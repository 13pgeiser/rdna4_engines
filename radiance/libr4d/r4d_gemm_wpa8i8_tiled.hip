// r4d_gemm_wpa8i8_tiled.hip - r4d_gemm_w4a8i8_tiled / r4d_gemm_w5a8i8_tiled: the 4- and 5-bit asymmetric
// weight x int8 activation GEMM for prefill row counts, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with
//
//     dequant(Wq)[n][k] = scale[n][k/128] * (code[n][k] - zero[n][k/128]),   code in 0..15 or 0..31
//
// (ParoQuant's int4 (z-lab) and int5, after the activation is rotated by givens_dense_quant) in
// 16x16x16 int8 WMMA, which issues at the fp8 rate. The f16 skinny kernel for these weights
// (r4d_gemm_w5a16_nt_m64) is capped by the f16 matrix instruction at about 75 TF/s. Arrangement as in
// r4d_gemm_mxfp4a8_tiled.hip:
//
//   * a 256 x 64 tile (4 x 2 waves, 4 x 2 fragments each); the activation is re-tiled by
//     r4d_mxfp4a8_tile_a into fragment order (one 16 x 16 fragment = 256 bytes) in the op's scratch and
//     read straight into registers; the weight slab is staged through LDS;
//   * a slab is one scale group (128 of K = 8 WMMA steps): the int32 accumulators restart each slab and
//     are folded into the float total as slab * (a_scale[row][g] * w_scale[col][g]), since the weight
//     scale varies by column;
//   * a partitioned weight (pb1/pb2) is one launch per part over windows of the planes.
//
// Why int8, not fp8: (code - zero) is -15..15 (int4) or -31..31 (int5), exact in int8, so staging is a
// byte-wise subtract; e4m3's three mantissa bits stop at 16. The activation is int8 with one scale per
// (row, 128 of K): after rotation a group is flat, so 127 uniform levels beat e4m3's relative 2^-4 for
// nearly every element (perplexity equals the f16 path's). Accumulators are exact int32: a slab is 128
// products of at most 31 * 127.
//
// Staging: for a code byte c (0..31) and zero z (0..31), (c | 0x80) - z = 128 + (c - z) never drops
// below 97 or borrows across bytes, and XOR 0x80 gives the two's complement byte of (c - z). Three ALU
// ops per dword of four weights.
//
// Weight blob as r4d_gemm_w5a16_nt_m64's (BITS 5): the nibble plane (w4's fragment order), then the
// fifth bits, one dword per (n-tile, 64 of K, lane): byte s = k step s, bit e = element e. BITS 4 is the
// nibble plane alone. The scale operand is the f16 kernel's dword plane [N/16][K/128][16]: f16 scale
// low, -(1024 + zero) high.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#define R4D_WPT_TM    4         // 16-row fragments per wave
#define R4D_WPT_TN    2         // 16-column fragments per wave
#define R4D_WPT_WM    4         // waves along M
#define R4D_WPT_WN    2         // waves along N
#define R4D_WPT_PAD   8         // LDS row pad: 16 lanes reading rows 136 B apart
#define R4D_WPT_NTHR  (R4D_WPT_WM * R4D_WPT_WN * 32)
#define R4D_WPT_BMF   (R4D_WPT_WM * R4D_WPT_TM * 16)
#define R4D_WPT_LBK   128       // one scale group a slab

typedef int      wpt_v2i  __attribute__((ext_vector_type(2)));
typedef int      wpt_v8i  __attribute__((ext_vector_type(8)));
typedef unsigned wpt_u4   __attribute__((ext_vector_type(4)));

// 4 bits -> bit 4 of each byte: the fifth bits of four elements, in the position above their nibbles
static __device__ __forceinline__ unsigned wpt_spread16(unsigned n) {
  return ((n * 0x00204081u) & 0x01010101u) << 4;
}

// four code bytes (0..31) and the zero point in every byte -> the four bytes of (code - zero), signed
static __device__ __forceinline__ unsigned wpt_sub_zero(unsigned codes4, unsigned zz) {
  return ((codes4 | 0x80808080u) - zz) ^ 0x80808080u;
}

template <int BITS>
__global__ __launch_bounds__(R4D_WPT_NTHR) void r4d_gemm_wpa8i8_tiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Whi, const unsigned* __restrict__ Wsz,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs, int ldc) {
  constexpr int TM = R4D_WPT_TM, TN = R4D_WPT_TN, WN = R4D_WPT_WN, NTHR = R4D_WPT_NTHR, BMF = R4D_WPT_BMF;
  constexpr int LBK = R4D_WPT_LBK, NS = LBK / 16, STR = LBK + R4D_WPT_PAD, BNF = WN * TN * 16;
  constexpr int NTILES_T = BNF / 16, TOT_SLOTS = NTILES_T * NS * 32;
  // A thread stages one lane's 16 bytes of a 64-wide k block (four k steps of one weight row, as the
  // plane is laid out: dword (t, kb, lane, s)), so a slab of two blocks is TOT_SLOTS / 4 reads.
  constexpr int NW = TOT_SLOTS / 4 / NTHR;
  static_assert(NW * NTHR * 4 == TOT_SLOTS, "the W tile must split evenly over the block");
  __shared__ unsigned char sW[BNF * STR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps = K / 16, kpb = K / 64, Mt = (M + 15) >> 4, nsz = K / 128;
  const int ntile_last = N / 16 - 1;

  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned int aoff = (unsigned int)lane * 8;

  // The activation scales this lane folds with: one per M fragment, at the row its accumulators hold.
  const float* __restrict__ arow[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int m = m0 + wm * TM * 16 + i * 16 + col;
    m = m < M ? m : M - 1;
    arow[i] = As + (size_t)m * as_rs;
  }

  // The W reads this thread makes, fixed across slabs: lane `wl` of n-tile `wt` (clamped to the last
  // real tile), 64-wide k block `wkb` of the slab, staged to the LDS row `wrow` at half `whalf`.
  int wrow[NW], wkb[NW], whalf[NW], wt[NW], wlane[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int idx = it * NTHR + tid;
    wlane[it] = idx & 31;
    wkb[it] = (idx >> 5) & 1;
    const int ntl = (idx >> 5) >> 1;
    wrow[it] = ntl * 16 + (wlane[it] & 15);
    whalf[it] = wlane[it] >> 4;
    const int t = (n0 >> 4) + ntl;
    wt[it] = t < ntile_last ? t : ntile_last;
  }

  float tot[TM][TN][8];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) tot[i][j][e] = 0.f;

  for (int k0 = 0, g = 0; k0 < K; k0 += LBK, ++g) {
    // A fragments for the slab, one clause, consumed only at the WMMAs.
    wpt_v2i af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const wpt_v2i*)(abase[i] +
                                      (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    // The scales of this slab, issued ahead of the staging that hides their latency.
    float sa[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) sa[i] = arow[i][g];
    float ws[TN][8];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int t = (n0 + wn * TN * 16 + j * 16) >> 4;
      t = t < ntile_last ? t : ntile_last;
      const wpt_u4* p = (const wpt_u4*)(Wsz + ((size_t)t * nsz + g) * 16 + kb8);
      const wpt_u4 v0 = p[0], v1 = p[1];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        ws[j][e]     = __half2float(__ushort_as_half((unsigned short)(v0[e] & 0xFFFFu)));
        ws[j][4 + e] = __half2float(__ushort_as_half((unsigned short)(v1[e] & 0xFFFFu)));
      }
    }
    // W slab -> LDS as int8 (code - zero), by true k.  A dword's element e is nibble 2e (e < 4) or
    // 2(e-4)+1, and k = 8*(e>>2) + 4*(lane>>4) + (e&3): a lane writes its elements 0..3 at
    // k = 4*(lane>>4) and 4..7 at +8, and the WMMA reads eight consecutive bytes.
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const size_t blk = (size_t)wt[it] * kpb + k0 / 64 + wkb[it];
      const wpt_u4 wv = *(const wpt_u4*)&Wq[blk * 128 + wlane[it] * 4];
      unsigned hb4 = 0;
      if constexpr (BITS == 5) hb4 = Whi[blk * 32 + wlane[it]];   // byte s: the fifth bits of k step s
      const unsigned sz = Wsz[((size_t)wt[it] * nsz + g) * 16 + (wlane[it] & 15)];
      const unsigned zz = ((sz >> 16) & 0x1Fu) * 0x01010101u;     // high half is -(1024 + zero) in f16
#pragma unroll
      for (int s_ = 0; s_ < 4; ++s_) {
        const unsigned w = wv[s_];
        unsigned cl = w & 0x0F0F0F0Fu, ch = (w >> 4) & 0x0F0F0F0Fu;      // elements 0..3, 4..7
        if constexpr (BITS == 5) {
          const unsigned hb = (hb4 >> (8 * s_)) & 0xFFu;
          cl |= wpt_spread16(hb & 15u);
          ch |= wpt_spread16(hb >> 4);
        }
        unsigned char* d = &sW[wrow[it] * STR + (wkb[it] * 4 + s_) * 16 + whalf[it] * 4];
        *(unsigned*)d = wpt_sub_zero(cl, zz);
        *(unsigned*)(d + 8) = wpt_sub_zero(ch, zz);
      }
    }
    lds_barrier();

    wpt_v8i gacc[TM][TN];
#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      wpt_v2i wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          gacc[i][j] = step == 0
              ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, wf[j], true, af[i][step], (wpt_v8i)(0), false)
              : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, wf[j], true, af[i][step], gacc[i][j], false);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) tot[i][j][e] = fmaf((float)gacc[i][j][e], sa[i] * ws[j][e], tot[i][j][e]);
    lds_barrier();
  }

#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int nb = n0 + wn * TN * 16 + j * 16 + kb8;
    if (nb >= N) continue;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      if (m >= M) continue;
      union { __hip_bfloat16 h[8]; wpt_u4 v; } o;
#pragma unroll
      for (int e = 0; e < 8; ++e) o.h[e] = (__hip_bfloat16)tot[i][j][e];
      *(wpt_u4*)(C + (size_t)m * ldc + nb) = o.v;
    }
  }
}

template <int BITS>
static void wpa8i8_launch(long at, long ascale, long wq, long whi, long wsz, long c, int M, int K, int N,
                          int as_rs, long stream, int ldc) {
  if (M < 1) throw std::runtime_error("r4d gemm_wpa8i8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_wpa8i8_tiled: N must be a multiple of 16");
  if (K % R4D_WPT_LBK) throw std::runtime_error("r4d gemm_wpa8i8_tiled: K must be a multiple of 128");
  const int bnf = R4D_WPT_WN * R4D_WPT_TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_WPT_BMF - 1) / R4D_WPT_BMF), block(R4D_WPT_NTHR);
  hipLaunchKernelGGL((r4d_gemm_wpa8i8_tiled_kernel<BITS>), grid, block, 0, (hipStream_t)stream,
                     (const unsigned char*)at, (const float*)ascale, (const unsigned*)wq,
                     (const unsigned*)whi, (const unsigned*)wsz, (__hip_bfloat16*)c, M, K, N, as_rs, ldc);
}

// ============================================================== the plugin ABI entry points
// The row-major int8 activation is re-tiled into the op's scratch (one part at a time) and the GEMM
// reads it there, as in mxfp4a8_tiled, including one launch per part of a partitioned weight.
template <int BITS>
static int wpa8i8_dispatch(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_WPT_LBK) return R4D_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    if (as_group != 128) return R4D_NO(RAD_E_SHAPE);       // one scale a (row, 128): the slab
    const long long slab = r4d_mxfp4a8_tile_a_bytes((int)g.M, (int)g.K);
    if (!a->scratch || a->scratch_bytes < slab) return R4D_NO(RAD_E_INVAL);
    const long long kb = g.K / 128;
    const long at = (long)(uintptr_t)a->scratch;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        R4D_TRY("gemm_wpa8i8_tiled",
            r4d_mxfp4a8_tile_a(r4d_addr(g.a) + p * g.K, at, (int)g.M, (int)g.K, (long)(uintptr_t)s,
                               (int)(mp.lda ? mp.lda : g.K));
            wpa8i8_launch<BITS>(at, r4d_addr(g.ascale) + p * kb * 4,
                                r4d_addr(g.b) + n0 * g.K / 2,
                                BITS == 5 ? r4d_addr(g.b) + g.N * g.K / 2 + n0 * g.K / 8 : 0L,
                                r4d_addr(g.bscale) + (n0 / 16) * kb * 16 * 4,
                                r4d_addr(g.y) + n0 * 2, (int)g.M, (int)g.K, (int)np, as_rs,
                                (long)(uintptr_t)s, (int)g.N));
    }
    return RAD_OK;
}

extern "C" int r4d_rad_gemm_w4a8i8_tiled(const RadArgs* a, RadStream s) { return wpa8i8_dispatch<4>(a, s); }
extern "C" int r4d_rad_gemm_w5a8i8_tiled(const RadArgs* a, RadStream s) { return wpa8i8_dispatch<5>(a, s); }
