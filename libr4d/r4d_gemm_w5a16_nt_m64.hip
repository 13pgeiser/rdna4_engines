// r4d_gemm_w5a16_nt_m64.hip - r4d_gemm_w5a16_nt_m64: 5-bit-weight skinny GEMM for M up to 64,
// gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// r4d_gemm_w4a16_nt_m64 with a fifth bit per weight, for ParoQuant's int5 (asymmetric, an f16 scale
// and a 5-bit integer zero per 128 of K, the rotation done on the activation in front of it).  The
// reading of that kernel's header holds word for word -- weight-streaming, memory-bound, f16
// because 1024 + n is exact -- and this file only says what the fifth bit changes.
//
// THE WEIGHT BLOB IS TWO PLANES.  The low four bits of every code are w4's fragment order exactly
// (offset binary here: the nibble is q & 15, no sign flip).  The fifth bits follow as a second
// plane of 32-bit words, ONE A (n-tile, 64-wide k block, lane):
//
//     hi[(t * (K/64) + kb) * 32 + lane]     byte s = k step s of the block, bit e = element e
//
// element e of a lane's fragment being the same one nibble 2e / 2(e-4)+1 of dword s holds.  So a
// lane's 16 bytes of nibbles and its 4 bytes of fifth bits cover the same 32 weights, and the
// second load is one dword a block.
//
// THE UNPACK ADDS TWO MULTIPLIES TO w4's.  The even elements' fifth bits are the byte's low nibble,
// the odd elements' its high one; a 4-bit mask n becomes one bit a byte by n * 0x00204081 (bit j
// lands at 8j and nothing else lands on a multiple of eight below bit 32), masked with 0x01010101
// and shifted to the 16s place of the nibble's byte.  The 5-bit code is then 0..31, 1024 + q is
// exact in f16, and the zero point -- an integer 0..31 in the scale dword's high half as
// -(1024 + zero) -- subtracts exactly.
//
// The dispatcher takes M past 64 as chunks of 64 rows (the weight is re-read per chunk, which at a
// 2048-row prefill chunk is near the compute bound of the f16 matrix instruction anyway), and
// partitioned weights (ParoQuant's gate|up: `pb1`/`pb2`, see r4d_gemm_args.h) as one launch per
// part over windows of the planes.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W4_MAX_M 64
#ifndef R4D_GEMM_W4_GROUP
#define R4D_GEMM_W4_GROUP 128        // K per (scale, zero) pair
#endif
#define R4D_GEMM_W4_KPB   64         // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

// NT reads the weight non-temporally. The weight is streamed once per step and never reused, while
// the activation is re-read once per column tile and wants to stay resident, so telling the cache
// not to keep the weight decides which of the two occupies the last level. It is a gain at M=8 and
// M=16 and a LOSS from M=24 up on the shapes with the longest K, so it is per-call, like every
// other geometry parameter, and the table in radiance_w4.py carries the value per M band.
//
// One copy of any of these weights fits this part's 64 MiB last level, so a bench that streams a
// SINGLE buffer in a loop measures a cache-resident machine and cannot see this effect at all; a
// pooled working set is what makes the difference visible.

// NPW (column tiles per WAVE) is a template parameter because it sizes the accumulator array, and a
// launch argument because the right value is a property of M: at M=8 the kernel is at the DRAM
// roofline and NPW>1 only costs blocks, while at M=64 the activation re-read is four times the
// weight and NPW=4 is what claws it back. Instantiated only for the two values that win in
// practice; anything else is rejected rather than silently rounded.

// Address-space-qualified so the loads stay global_load rather than flat_load: on gfx12 a flat
// load increments dscnt as well as loadcnt, so a later LDS wait would also wait on HBM.  See the
// same declaration in r4d_gemm_bf16_nt_m64.hip for why the element type is a scalar there.
typedef __attribute__((address_space(1))) const uint4v*          r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;
typedef __attribute__((address_space(1))) const unsigned*        r4d_gu1;

__device__ __forceinline__ v8h fragA(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// One dword of eight nibbles -> one 16x16x16 f16 B fragment, minus the zero point.
// t0/t1 split the even and odd nibbles into bytes; each v_perm_b32 pairs two of those bytes with
// the constant high byte 0x64, which is the f16 exponent field for 1024.  nz carries
// -(1024 + zero) in BOTH halves so one v_pk_add_f16 corrects a pair.
// The unpack is under a percent of this kernel at M=8, where it is at the DRAM roofline and the
// arithmetic hides in the load latency. It is a real fraction at M=64, but that band belongs to
// r4d_gemm_w4a8_nt_m64 -- and with no dequant at all the f16 form still trails the int8 one there,
// because the gap is the matrix instruction and not the dequant. Nothing here is worth a knob.
#ifndef R4D_GEMM_W4_TWOS
#define R4D_GEMM_W4_TWOS 1
#endif

// DO NOT FOLD THE GROUP SCALE INTO THE B FRAGMENT.  One v_pk_mul_f16 per pair on the finished
// integer q is exact in f16, removes the second accumulator array entirely and takes the M=64 shape
// from 192 VGPRs to 135 -- and it is SLOWER on most of the drafter's shapes.  The multiply lands
// inside the per-fragment dependent chain between the unpack and the WMMA, where the per-group
// epilogue it replaces sits outside it; this kernel is stall-bound, not register-bound.

__device__ __forceinline__ v8h dequant(unsigned w, __half2 nz) {
  const unsigned K64 = 0x64646464u;
#if R4D_GEMM_W4_TWOS
  // The stored nibble is a 4-bit TWO'S COMPLEMENT code, because that is what
  // r4d_gemm_w4a8_nt_m64 needs -- it shifts the nibble into the byte's high half and reads the
  // byte signed.  This kernel wants OFFSET BINARY, so that 1024 + n minus the constant 1024 + 8
  // is the value.  The two differ by bit 3 of the nibble, so one XOR per spread dword converts,
  // and the zero point is then pinned to 8 for every group.  Two instructions per fragment, on
  // the kernel whose whole unpack is a fraction of a percent of its time at M=8 -- which is why
  // the conversion lives here and not in the int8 kernel, where the unpack is three instructions
  // and does matter.  One quantised weight, two kernels, dispatched on M.
  const unsigned t0 = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
#else
  const unsigned t0 = w & 0x0F0F0F0Fu;
  const unsigned t1 = (w >> 4) & 0x0F0F0F0Fu;
#endif
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);   // nibbles 0, 2
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);   // nibbles 4, 6
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);   // nibbles 1, 3
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);   // nibbles 5, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

// One dword of eight nibbles plus the byte of their fifth bits -> one 16x16x16 f16 B fragment,
// minus the zero point.  Nibble 2e (e < 4) / 2(e-4)+1 is fragment element e, so bits 0..3 of `hb`
// are the fifth bits of the even nibbles' elements and bits 4..7 the odd nibbles'.
__device__ __forceinline__ unsigned spread16(unsigned n) {      // 4 bits -> bit 4 of each byte
  return ((n * 0x00204081u) & 0x01010101u) << 4;
}
__device__ __forceinline__ v8h dequant5(unsigned w, unsigned hb, __half2 nz) {
  const unsigned K64 = 0x64646464u;
  const unsigned t0 = (w & 0x0F0F0F0Fu) | spread16(hb & 15u);
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) | spread16(hb >> 4);
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);   // elements 0, 1
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);   // elements 2, 3
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);   // elements 4, 5
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);   // elements 6, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// MT = row tiles per BLOCK, as in the bf16 kernel: more of them amortise the weight fragment and
// its unpack over more WMMA, fewer put more blocks on the machine.
// HI false is the 4-bit asymmetric grid (z-lab's plain int4): the same kernel with no fifth-bit plane.
template <int MT, int NPW, bool NT, bool HI>
__global__ __launch_bounds__(1024) void r4d_gemm_w5a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Whi, const unsigned* __restrict__ Wsz,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int lda, int ldc) {
  extern __shared__ float red[];        // WV*NPW*SK*256 floats: a 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W4_GROUP;              // groups this split owns
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Activation: rows clamped, never masked -- a tile past M re-reads the last valid row and
    // computes a result nobody stores.
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; i++)
      aoff[i] = rowGo((unsigned)lda, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);

    // Weight: one uint4 per lane per four k steps, tiles already permuted, so the whole k loop is
    // global_load_b128 with a constant stride and no address arithmetic per step.
    const int kpb = K / R4D_GEMM_W4_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / R4D_GEMM_W4_GROUP;                // (scale, zero) dwords per row of tiles
    r4d_gw4 pw[NPW];
    r4d_gu1 ph[NPW];
    r4d_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      // Clamped, like the row tiles: a wave whose last column tiles run past N re-reads the last
      // real tile and computes a result the store discards, rather than reading memory it does not
      // own or diverging the whole k loop on a mask.
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      // Quantizer parameters: 16 distinct dwords per group, read by the 16 lanes that own the rows.
      ph[j] = (r4d_gu1)(const unsigned*)(HI ? Whi : Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gu1)(const unsigned*)(Wsz) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W4_GROUP / R4D_GEMM_W4_KPB;  // packed blocks per group

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
      __half2 nz[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned sz = ps[j][(size_t)g * 16];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
      }

      // The first WMMA of a group takes an inline zero for C, so the group accumulator costs no
      // instruction to clear.
      v8f g_acc[NPW][MT];
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
        unsigned hv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          hv[j] = HI ? (NT ? __builtin_nontemporal_load(&ph[j][o]) : ph[j][o]) : 0u;
        }
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int u = b * 4 + s;
          v8h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA(pa[i], u);   // loaded once, spent NPW times
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v8h bfrag = dequant5(wv[j][s], (hv[j] >> (8 * s)) & 0xFFu, nz[j]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = u == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f))
                                   : mmah(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += (R4D_GEMM_W4_GROUP / 16) * 4;
    }
  }

  // Split-K reduction and store: identical to r4d_gemm_bf16_nt_m64, one row tile at a time so the
  // LDS is WV*SK KiB whatever MT is, and in a fixed order over splits so the result does not depend
  // on how the waves were scheduled.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * ldc + n] = (__hip_bfloat16)v;
    }
  }
}

void r4d_gemm_w5a16_nt_m64(long a, long wq, long whi, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream, int lda, int ldc,
                           bool hi5) {
  if (!lda) lda = K;
  if (!ldc) ldc = N;
  if (M < 1 || M > R4D_GEMM_W4_MAX_M)
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4_GROUP))
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: K must be divisible by SK*128");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: NPW must be 1 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Hp = (const unsigned*)whi;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT) do { \
    if (hi5) hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<MT, NP, TT, true>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc); \
    else     hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<MT, NP, TT, false>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc); } while (0)
#define LAUNCH_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH(1, NP, TT); break; \
    case 2: LAUNCH(2, NP, TT); break; \
    case 3: LAUNCH(3, NP, TT); break; \
    case 4: LAUNCH(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH_MB(1, true); } else { LAUNCH_MB(1, false); }
  } else {
    if (NT) { LAUNCH_MB(4, true); } else { LAUNCH_MB(4, false); }
  }
#undef LAUNCH_MB
#undef LAUNCH
}

int r4d_gemm_w5a16_nt_m64_max_m(void) { return R4D_GEMM_W4_MAX_M; }
int r4d_gemm_w5a16_nt_m64_group(void) { return R4D_GEMM_W4_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
/* ------------------------------------------------------------------ f16-activation grids */

static int w5_dispatch(const RadArgs* a, RadStream s, bool hi5) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_GEMM_W4_GROUP) return R4D_NO(RAD_E_SHAPE);
    const int group = r4d_gemm_w5a16_nt_m64_group();
    // The row's M is the step's, not the kernel's: rows past 64 are chunks of 64, each its own pass
    // over the weight, so a prefill chunk costs the weight's bytes once a 64 rows.
    const long long mc = std::min<long long>(g.M, r4d_gemm_w5a16_nt_m64_max_m());
    Cfg c;
    rc = cfg_for(a, g.K, mc, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    const long long Ntot = g.N, K = g.K, lda = mp.lda ? mp.lda : g.K;
    const long long nsz = K / R4D_GEMM_W4_GROUP;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        for (long long m0 = 0; m0 < g.M; m0 += mc) {
            const long long mm = std::min<long long>(mc, g.M - m0);
            R4D_TRY("gemm_w5a16_nt_m64",
                r4d_gemm_w5a16_nt_m64(
                    r4d_addr(g.a) + (m0 * lda + p * K) * 2,
                    r4d_addr(g.b) + n0 * K / 2,                              /* the nibble plane */
                    hi5 ? r4d_addr(g.b) + Ntot * K / 2 + n0 * K / 8 : 0L,    /* the fifth-bit plane */
                    r4d_addr(g.bscale) + (n0 / 16) * nsz * 16 * 4,
                    r4d_addr(g.y) + (m0 * Ntot + n0) * 2,
                    (int)mm, (int)K, (int)np, c.wv, c.sk, c.mb, c.npw, c.nt, (long)(uintptr_t)s,
                    (int)lda, (int)Ntot, hi5));
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_gemm_w5a16_m64(const RadArgs* a, RadStream s) { return w5_dispatch(a, s, true); }
/* ...and the plain 4-bit asymmetric grid (z-lab's int4), whose blob is the nibble plane alone. */
extern "C" int r4d_rad_gemm_w4a16asym_m64(const RadArgs* a, RadStream s) { return w5_dispatch(a, s, false); }
