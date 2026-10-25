// r4d_gemm_m2xfp4a8_tiled.hip - r4d_gemm_m2xfp4a8_tiled: the M2XFP4 x fp8 GEMM for prefill row
// counts, gfx1201. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the operands, weight layout, fold and
// tile of r4d_gemm_mxfp4a8_tiled, plus the weight-side Sg-EM (arXiv 2601.19213 §4.4.2) the decode
// row decodes: per group of 32 K a meta byte of four two-bit subgroup multipliers, folded into the
// e4m3 staging bytes through r4d_m2xfp4.h's second table index, so the WMMA chain, the Wref
// epilogue and the activation scale grids are the mxfp4 kernel's unchanged.
//
// The meta byte is staged beside the exponent byte: one byte per (weight row, 32 of K), in the
// scale plane's [K/32][N] order and row pitch, in the tail of the same stored weight (the relayout
// appends it after the fragment-order codes, r4d_layout.cpp). A staged slot's eight elements sit at
// an offset that is a multiple of 8 inside their 32-column group, so one two-bit field per slot
// covers it whole; the shift is fixed per slot across slabs and read with the geometry the scale
// dword already uses.
//
// The fold's bound: Wref[n] is the row's largest exponent byte as for MXFP4, so d >= 0 holds for
// every block, and the largest element any table cell then holds is 6 * 1.75 * 2^-d = 10.5 * 2^-d
// -- an e4m3 normal at every d >= 0 (r4d_m2xfp4.h). The table is four times the mxfp4 one, 512
// bytes, still nothing against sW; the LDS copy widens with it.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_m2xfp4.h"

// bf16 output as uint16 pairs (the emulator-compatible spelling, r4d_common.h): the OCP
// rows read __hip_bfloat16 from hip_bf16.h, whose bytes these are.
#include <cstdint>
#include <stdexcept>

#define R4D_M2T_TM    4         // 16-row fragments per wave
#define R4D_M2T_WM    4         // waves along M
#define R4D_M2T_WN    2         // waves along N
#define R4D_M2T_PAD   8         // LDS row pad: 16 lanes reading rows 64 B apart collide 8 ways
#define R4D_M2T_NTHR  (R4D_M2T_WM * R4D_M2T_WN * 32)
#define R4D_M2T_BMF   (R4D_M2T_WM * R4D_M2T_TM * 16)
#define R4D_M2T_KUNIT 64        // the narrowest K slab, so K must be a multiple of it
// tn=0's M crossover to TN=4 for this kernel, as its tn axis (tGemmMxfp4Tiled) states it.
#define R4D_M2T_TN4_MIN_M 2048

typedef int      m2t_v2i  __attribute__((ext_vector_type(2)));
typedef unsigned m2t_u2   __attribute__((ext_vector_type(2)));
typedef unsigned m2t_u4   __attribute__((ext_vector_type(4)));

// The re-tile: one thread per 8-byte slot of the fragment-ordered plane; writes are contiguous.
// (r4d_mxfp4a8_tile_a_kernel, shared work; this row runs its own copy so the two units stay
// independent.)
__global__ __launch_bounds__(256) void r4d_m2xfp4a8_tile_a_kernel(
    const unsigned char* __restrict__ A, unsigned char* __restrict__ AT, int M, int K,
    long long slots, int lda) {
  const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
  if (o >= slots) return;
  const int ks = K / 16;
  const int lane = (int)(o & 31);
  const long long f = o >> 5;
  const int mt = (int)(f / ks), kst = (int)(f % ks);
  const int r = mt * 16 + (lane & 15), k = kst * 16 + (lane >> 4) * 8;
  m2t_u2 v = { 0u, 0u };
  if (r < M) v = *(const m2t_u2*)(A + (size_t)r * lda + k);
  *(m2t_u2*)(AT + o * 8) = v;
}

template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(R4D_M2T_NTHR) void r4d_gemm_m2xfp4a8_tiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wm, const unsigned char* __restrict__ Wref,
    uint16_t* __restrict__ C, int M, int K, int N, int as_rs, int ldws, int ldc) {
  constexpr int TM = R4D_M2T_TM, WN = R4D_M2T_WN, NTHR = R4D_M2T_NTHR, BMF = R4D_M2T_BMF;
  constexpr int NS = LBK / 16, STR = LBK + R4D_M2T_PAD, BNF = WN * TN * 16;
  constexpr int NTILES_T = BNF / 16, TOT_SLOTS = NTILES_T * NS * 32;
  // 4 weight slots (one 16 B read) wherever the tile has enough of them, else 2 (8 B), else 1.
  constexpr int WSLOTS = (TOT_SLOTS >= NTHR * 4) ? 4 : (TOT_SLOTS >= NTHR * 2 ? 2 : 1);
  constexpr int NW = TOT_SLOTS / (NTHR * WSLOTS);
  static_assert(NW * NTHR * WSLOTS == TOT_SLOTS, "the W tile must split evenly over the block");
  __shared__ unsigned char sW[BNF * STR];
  __shared__ unsigned int sMag[128];     // kM2xfp4Mag: [16][4][2] words
  __shared__ float sR[ABLK ? BMF : 1];   // ABLK: a boundary's ratio a row, then each row's final scale

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps = K / 16, Mt = (M + 15) >> 4;

  if (tid < 128) sMag[tid] = (&kM2xfp4Mag[0][0][0])[tid];

  // One A base per M fragment, wave-uniform: readfirstlane keeps it in an SGPR so loads take the
  // SADDR form with a 32-bit lane offset. A tile past M reads the last real one.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned int aoff = (unsigned int)lane * 8;

  // The W slots this thread stages, fixed across slabs: WSLOTS adjacent rows of one (n-tile,
  // k-step), so the wave reads contiguous bytes; the clamp keeps the vector read aligned. The meta
  // shift moves with the slot: its eight elements' offset inside their 32-column group is
  // kst*16 + (lane>>4)*8, a multiple of 8 (r4d_m2xfp4.h).
  int wrow[NW], wkloc[NW], wrc[NW], wkst[NW], wlanec[NW], wksh[NW];
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHR * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % NS, ntl = rest / NS;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wkloc[it] = kst * 16 + (lane_ >> 4) * 8;
    wksh[it] = r4d_m2xfp4_kshift(wkloc[it]);
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
    wref[it] = WSLOTS == 4 ? *(const unsigned int*)(Wref + wrc[it])
             : WSLOTS == 2 ? *(const unsigned short*)(Wref + wrc[it])
                           : Wref[wrc[it]];
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  // ABLK: thread t owns row m0 + t (BMF == NTHR) and loads the next 128 boundary's two scales (the
  // block starting there and the one before it) a boundary early.
  const int trow = m0 + tid < M ? m0 + tid : M - 1;
  const float* __restrict__ asrow = As + (size_t)trow * as_rs;
  float ns = 0.f, np = 0.f;
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  lds_barrier();   // sMag

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // ABLK: at a 128 boundary each row's ratio goes to LDS, published by the staging barrier below
    // (sR's previous readers finished before the last slab's closing barrier). A zero block keeps
    // the carried scale; after a zero block the carried scale is looked up from the grid.
    const bool bnd = ABLK && k0 && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd)
        sR[tid] = ns > 0.f ? (np > 0.f ? np : r4d_mxfp4_carried(asrow, (k0 >> 7) - 2)) / ns : 1.f;
    }
    // A fragments for the whole slab, one clause, consumed only at the WMMAs.
    m2t_v2i af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const m2t_v2i*)(abase[i] +
                                      (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    // W slab -> LDS, folded to e4m3. The meta bytes ride the same dword reads as the exponents.
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int* src =
          &Wp[((size_t)(wrc[it] >> 4) * ksteps + kstep0 + wkst[it]) * 32 + wlanec[it]];
      const int blk = (k0 + wkst[it] * 16) / R4D_MXFP4_GROUP;
      unsigned int wq[WSLOTS], wsw, wmw;
      if constexpr (WSLOTS == 4) {
        const m2t_u4 v = *(const m2t_u4*)src;
        wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        wsw = *(const unsigned int*)(Ws + (size_t)blk * ldws + wrc[it]);
        wmw = *(const unsigned int*)(Wm + (size_t)blk * ldws + wrc[it]);
      } else if constexpr (WSLOTS == 2) {
        const m2t_u2 v = *(const m2t_u2*)src;
        wq[0] = v[0]; wq[1] = v[1];
        wsw = *(const unsigned short*)(Ws + (size_t)blk * ldws + wrc[it]);
        wmw = *(const unsigned short*)(Wm + (size_t)blk * ldws + wrc[it]);
      } else {
        wq[0] = *src;
        wsw = Ws[(size_t)blk * ldws + wrc[it]];
        wmw = Wm[(size_t)blk * ldws + wrc[it]];
      }
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const int d = r4d_mxfp4_fold_d((int)((wref[it] >> (8 * q)) & 0xFF),
                                       (int)((wsw >> (8 * q)) & 0xFF));
        const int k = (int)((wmw >> (8 * q + wksh[it])) & 3);
        const int ti = (d * 4 + k) * 2;
        unsigned int lo, hi;
        r4d_mxfp4_unpack8_t(wq[q], sMag[ti], sMag[ti + 1], &lo, &hi);
        *(m2t_u2*)(&sW[(wrow[it] + q) * STR + wkloc[it]]) = m2t_u2{lo, hi};
      }
    }
    lds_barrier();

    if constexpr (ABLK) {
      // Re-express each row's partial sum in the new block's scale: one ratio a lane a row.
      if (bnd) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const float r = sR[wm * TM * 16 + i * 16 + col];
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] *= r;
        }
      }
      // The next boundary's scales, issued behind this slab's loads and needed a boundary later.
      if ((k0 & 127) == 0 && k0 + 128 < K) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      m2t_v2i wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      // Keep one step's W fragments live at a time; otherwise the compiler hoists every step's
      // loads above the first WMMA.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf[j], af[i][step],
                                                                          acc[i][j]);
    }
    lds_barrier();
  }

  // Epilogue. A lane holds row m0 + 64*wm + 16i + col of M fragment i and eight consecutive columns
  // of N fragment j, one 16-byte store per (i, j). 2^(Wref[n]-127) is the float whose exponent
  // field is Wref[n]; N % 16 == 0 keeps a lane's eight columns all inside or all outside N.
  float sc[TM];
  if constexpr (ABLK) {
    // the scale each row ends in, once a row; the K loop's closing barrier retired the ratio reads
    sR[tid] = r4d_mxfp4_carried(asrow, (K - 1) >> 7);
    lds_barrier();
#pragma unroll
    for (int i = 0; i < TM; ++i) sc[i] = sR[wm * TM * 16 + i * 16 + col];
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      sc[i] = As[(size_t)(m < M ? m : M - 1) * as_rs];
    }
  }
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int nb = n0 + wn * TN * 16 + j * 16 + kb8;
    if (nb >= N) continue;
    const m2t_u2 rb = *(const m2t_u2*)(Wref + nb);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      if (m >= M) continue;
      // rounded as the bf16 store rounds, packed in whole registers (see bf16x2_rne_nan)
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float rf = __int_as_float((int)((rb[e >> 2] >> (8 * (e & 3))) & 0xFF) << 23);
        v[e] = acc[i][j][e] * rf * sc[i];
      }
      m2t_u4 o;
      o.x = bf16x2_rne_nan(v[0], v[1]);
      o.y = bf16x2_rne_nan(v[2], v[3]);
      o.z = bf16x2_rne_nan(v[4], v[5]);
      o.w = bf16x2_rne_nan(v[6], v[7]);
      *(m2t_u4*)(C + (size_t)m * ldc + nb) = o;
    }
  }
}

extern "C" int r4d_gemm_m2xfp4a8_tiled_group(void) { return R4D_MXFP4_GROUP; }

extern "C" long r4d_m2xfp4a8_tile_a_bytes(int M, int K) {
  return M < 1 || K < 16 ? 0 : (long)((M + 15) / 16) * 16 * K;
}

extern "C" void r4d_m2xfp4a8_tile_a(long a, long at, int M, int K, long stream, int lda) {
  if (!lda) lda = K;
  if (M < 1) throw std::runtime_error("r4d m2xfp4a8_tile_a: M must be >= 1");
  if (K < 16 || K % 16) throw std::runtime_error("r4d m2xfp4a8_tile_a: K must be a multiple of 16");
  const long long slots = (long long)((M + 15) / 16) * (K / 16) * 32;
  hipLaunchKernelGGL(r4d_m2xfp4a8_tile_a_kernel, dim3((unsigned)((slots + 255) / 256)), dim3(256),
                     0, (hipStream_t)stream, (const unsigned char*)a, (unsigned char*)at, M, K,
                     slots, lda);
}

extern "C" void r4d_gemm_m2xfp4a8_tiled(long at, long ascale, long wq, long ws, long wm, long wref,
                                        long c, int M, int K, int N, int TN, int as_group,
                                        int as_rs, long stream, int ldws, int ldc) {
  if (!ldws) ldws = N;
  if (!ldc) ldc = N;
  if (M < 1) throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: N must be a multiple of 16");
  if (K < R4D_M2T_KUNIT || K % R4D_M2T_KUNIT)
    throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: K must be a multiple of 64");
  // tn=0: TN=4 from the M crossover on, unless its 128-wide tile leaves fewer than 128 blocks
  // (narrow N), where TN=2's extra blocks win.
  if (TN == 0)
    TN = M >= R4D_M2T_TN4_MIN_M && (long)((N + 127) / 128) * ((M + R4D_M2T_BMF - 1) / R4D_M2T_BMF) >= 128
             ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("r4d gemm_m2xfp4a8_tiled: activation scale rows shorter than ceil(K/128)");
  const bool wide_slab = TN == 2 && K % 128 == 0;
  const int bnf = R4D_M2T_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_M2T_BMF - 1) / R4D_M2T_BMF), block(R4D_M2T_NTHR);
  auto Ap = (const unsigned char*)at;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Mp = (const unsigned char*)wm;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (uint16_t*)c;
  hipStream_t st = (hipStream_t)stream;
#define M2T_LAUNCH(TNv, LBKv, ABv)                                                              \
  hipLaunchKernelGGL((r4d_gemm_m2xfp4a8_tiled_kernel<TNv, LBKv, ABv>), grid, block, 0, st, Ap, Sp, \
                     Wp, Zp, Mp, Rp, Cp, M, K, N, as_rs, ldws, ldc)
  if (TN == 4)        { if (ablk) M2T_LAUNCH(4, 64, true);  else M2T_LAUNCH(4, 64, false); }
  else if (wide_slab) { if (ablk) M2T_LAUNCH(2, 128, true); else M2T_LAUNCH(2, 128, false); }
  else                { if (ablk) M2T_LAUNCH(2, 64, true);  else M2T_LAUNCH(2, 64, false); }
#undef M2T_LAUNCH
}

// ============================================================== the plugin ABI entry point
// The row-major activation is re-tiled into the op's scratch and the GEMM reads it there.
extern "C" int r4d_rad_gemm_m2xfp4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    /* No parts: the m2xfp4 declare makes none, so gemm_ops refuses pb1/pb2, as libref does. */
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent is required, as in r4d_rad_gemm_mxfp4a8_m64. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    if (g.K % R4D_M2T_KUNIT) return R4D_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs);
    if (rc != RAD_OK) return rc;
    const long long slab = r4d_m2xfp4a8_tile_a_bytes((int)g.M, (int)g.K);
    if (!a->scratch || a->scratch_bytes < slab) return R4D_NO(RAD_E_INVAL);
    R4dMxTileCfg t;
    r4d_cfg_mxfp4_tiled(a, &t);
    const long at = (long)(uintptr_t)a->scratch;
    // The meta plane rides the weight blob's tail, in the scale plane's [K/32][N] order.
    const long meta = r4d_addr(g.b) + (long)g.N * g.K / 2;
    R4D_TRY("gemm_m2xfp4a8_tiled",
        r4d_m2xfp4a8_tile_a(r4d_addr(g.a), at, (int)g.M, (int)g.K, (long)(uintptr_t)s, (int)g.K);
        r4d_gemm_m2xfp4a8_tiled(at, r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), meta,
                                r4d_addr(g.bref), r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N,
                                t.tn, as_group, as_rs, (long)(uintptr_t)s, (int)g.N, (int)g.N));
    return RAD_OK;
}

/* The op's scratch, sized at the band's widest M: the fragment-tiled activation in whole 16-row
 * fragments. */
extern "C" int64_t r4d_rad_gemm_m2xfp4a8_tiled_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), K = r4d_geti_or(a, "K", 0);
    return (int64_t)r4d_m2xfp4a8_tile_a_bytes((int)M, (int)K);
}
