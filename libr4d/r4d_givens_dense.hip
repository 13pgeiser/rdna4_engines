// r4d_givens_dense.hip -- ParoQuant's rotation as a dense per-group matrix on the matrix cores.
//
// The Givens stages of a 128-column group compose to one orthogonal 128 x 128 matrix T, so
// y = T (x / D) is a small GEMM: out[m][j] = sum_i T[j][i] * (x[m][i] / D[i]).  The sequential kernel
// (r4d_givens_rotate.hip) does the same job in `stages` dependent LDS rounds a group and is
// latency bound (~100-130 GB/s of ~640); this one spends ~8x the multiply-adds on the f16 WMMA, which
// is 5 us of math for a 2048-row part, and quantises in the epilogue -- int8 (scale amax / 127, for the
// int8 prefill GEMMs) or E4M3 (amax / 448, for the mxfp4a8 GEMMs), the same scale grid as the
// sequential producer's.  About 3x faster at prefill sizes and no slower at decode.
//
// THE MATRICES ARE THE WEIGHT: rad-convert composes them from the container's `t.pairs` / `t.angle`
// tables (r4d_layout.cpp, r4d.givens.dense) into [parts * K / 128][128 (j)][128 (i)] f16, so the
// serving side never walks a Givens network.  The channel scale stays a separate table (`rscale`), a
// divide on the way in: folding 1/D into T would put entries of very different magnitude in one f16
// matrix.
//
// ONE BLOCK A (part, group) x row chunk.  The group's T (32 KiB of f16) is staged in LDS once and each
// of the block's four waves takes 16-row tiles: per k step one B fragment (the activation row's
// eight k, divided by D, as f16) meets the eight j tiles of T.  The accumulators are C[j][m]: lane l
// holds column m = l & 15 and rows j = 8 * (l >> 4) + e, so a row's 128 outputs are split over lane m
// and lane m + 16 and its amax is a max over 64 registers and one xor-16 shuffle.
//
// WHAT MOVED THE TIME (selftest, 2048 rows x K 5120): the first version waited on one activation load a
// k step and spilled ~190 registers (186 us).  A tile's eight loads are now issued together and the
// issue order is pinned per k step (sched_barrier), which is what keeps the A fragments from being
// hoisted whole into registers: 137 VGPRs, no spills, ~100 us.  Double-buffering the next tile and
// coalescing the stores through LDS were tried and did not move it.
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_common.h"
#include "r4d_q8.h"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kGroup = 128;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kWaves = kThreads / 32;
constexpr uint32_t kTStride = kGroup + 8;           // f16 per LDS row of T: 16 B pad, conflict-free b128 reads

typedef _Float16 v8h_t __attribute__((ext_vector_type(8)));

template <bool I8>
__global__ __launch_bounds__(kThreads) void givens_dense_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ tdense, const float* __restrict__ rscale,
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t M, uint32_t K, uint32_t nkb, uint32_t x_ld,
    uint32_t q_ld, uint32_t s_ld) {
    __shared__ uint16_t Tl[kGroup * kTStride];
    __shared__ float invd[kGroup];
    const uint32_t gp = blockIdx.x, part = gp / nkb, g = gp % nkb;
    const uint32_t tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;

    // all sixteen loads in flight before the first LDS store: the staging is latency, not bandwidth
    const uint4* tg = (const uint4*)(tdense + (size_t)gp * kGroup * kGroup);
    constexpr uint32_t kStage = kGroup * kGroup / 8 / kThreads;
    uint4 tv[kStage];
#pragma unroll
    for (uint32_t k = 0; k < kStage; ++k) tv[k] = tg[tid + k * kThreads];
#pragma unroll
    for (uint32_t k = 0; k < kStage; ++k) {
        const uint32_t i = tid + k * kThreads, j = i / (kGroup / 8), c = i % (kGroup / 8);
        *(uint4*)&Tl[j * kTStride + c * 8] = tv[k];
    }
    invd[tid] = 1.0f / rscale[(size_t)part * K + g * kGroup + tid];
    __syncthreads();

    const uint32_t h = lane >> 4, col = lane & 15;
    const uint32_t Mt = (M + 15) >> 4;
    const uint32_t step = gridDim.y * kWaves;
    for (uint32_t tile = blockIdx.y * kWaves + wave; tile < Mt; tile += step) {
        // The activation row a lane reads: lane (col, h) takes columns 8h .. 8h+7 of every 16-wide k step of
        // row (tile * 16 + col).  All eight k steps of a tile are loaded at once, so a tile costs one load
        // latency and not eight.
        uint32_t m = tile * 16 + col;
        const bool live = m < M;
        if (!live) m = M - 1;
        const uint16_t* xr = x + (size_t)m * x_ld + g * kGroup + 8 * h;
        uint4 raw[8];
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) raw[ks] = *(const uint4*)(xr + ks * 16);
        v8f acc[8];
#pragma unroll
        for (int jt = 0; jt < 8; ++jt) acc[jt] = (v8f)(0.f);
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) {
            const uint32_t c0 = ks * 16 + 8 * h;
            const uint32_t w[4] = { raw[ks].x, raw[ks].y, raw[ks].z, raw[ks].w };
            v8h_t bf;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t b = (w[e >> 1] >> (16 * (e & 1))) & 0xFFFFu;
                const float v = __uint_as_float(b << 16) * invd[c0 + e];
                bf[e] = (_Float16)fminf(fmaxf(v, -65504.0f), 65504.0f);
            }
#pragma unroll
            for (int jt = 0; jt < 8; ++jt) {
                const v8h_t af = *(const v8h_t*)&Tl[(jt * 16 + col) * kTStride + c0];
                acc[jt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf, acc[jt]);
            }
            __builtin_amdgcn_sched_barrier(0);        // keep a k step's A fragments from being hoisted over the last
        }
        float amax = 0.0f;
#pragma unroll
        for (int jt = 0; jt < 8; ++jt)
#pragma unroll
            for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(acc[jt][e]));
        amax = fmaxf(amax, __shfl_xor(amax, 16));
        const float sc = amax > 0.0f ? amax * (1.0f / q8_top<I8>()) : 1.0f;
        const float inv = 1.0f / sc;
        if (!live) continue;
        if (h == 0) s[(size_t)m * s_ld + (size_t)part * nkb + g] = sc;
        uint8_t* qr = q + (size_t)m * q_ld + (size_t)part * K + g * kGroup + 8 * h;
#pragma unroll
        for (int jt = 0; jt < 8; ++jt) {
            const uint32_t lo = q8_pack4<I8>(acc[jt][0], acc[jt][1], acc[jt][2], acc[jt][3], inv);
            const uint32_t hi = q8_pack4<I8>(acc[jt][4], acc[jt][5], acc[jt][6], acc[jt][7], inv);
            *(uint2*)(qr + jt * 16) = make_uint2(lo, hi);
        }
    }
}

}  // namespace

//   x       [M][K]                       bf16, `x_ld` elements a row (0 = K)
//   tdense  [parts * K/128][128][128]    f16, T[j][i] of each (part, group), laid out by r4d_layout.cpp
//   rscale  [parts][K]                   f32, the factor the stored row was multiplied by
//   q       [M][parts * K]               int8 or E4M3, `q_ld` elements a row (0 = parts * K)
//   s       [M][parts * K/128]           f32, `s_ld` elements a row (0 = parts * K/128)
enum { GD_X = 0, GD_T = 1, GD_RSCALE = 2, GD_Q = 3, GD_S = 4 };

static int givens_dense_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x  = r4d_opt(a, GD_X);
    const RadTensor* t  = r4d_opt(a, GD_T);
    const RadTensor* rs = r4d_opt(a, GD_RSCALE);
    const RadTensor* q  = r4d_opt(a, GD_Q);
    const RadTensor* sc = r4d_opt(a, GD_S);
    if (!x || !t || !rs || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2 || t->rank != 2 || rs->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    if (t->dtype != RAD_F16 || rs->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const bool i8 = q->dtype == RAD_I8;
    if (!i8 && q->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long parts = r4d_geti_or(a, "parts", 1), nkb = K / kGroup;
    if (r4d_geti_or(a, "group", kGroup) != kGroup || K % kGroup || parts < 1) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != parts * K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != parts * nkb) return R4D_NO(RAD_E_SHAPE);
    // tdense is a packed blob: its byte count is the layout hook's ([parts * K, 128] f16), not a shape to check
    if (rs->shape[0] != parts || rs->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(t->data, 16) || !r4d_aligned(x->data, 16) || !r4d_aligned(q->data, 8))
        return R4D_NO(RAD_E_ALIGN);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0 || (xp * 2) % 16) return R4D_NO(RAD_E_STRIDE);
    const long long mt = (M + 15) / 16;
    const uint32_t gy = (uint32_t)std::max<long long>(1, (mt + 4 * kWaves - 1) / (4 * kWaves));
    const dim3 grid((uint32_t)(parts * nkb), gy);
#define GD_EMIT(I8_) R4D_EMIT(sk, (givens_dense_kernel<I8_>), grid, dim3(kThreads), 0, \
        (const uint16_t*)x->data, (const uint16_t*)t->data, (const float*)rs->data, (uint8_t*)q->data, \
        (float*)sc->data, (uint32_t)M, (uint32_t)K, (uint32_t)nkb, (uint32_t)xp, (uint32_t)qp, (uint32_t)sp)
    const int rc = i8 ? GD_EMIT(true) : GD_EMIT(false);
#undef GD_EMIT
    if (rc != RAD_OK) return rc;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? RAD_E_DEVICE : RAD_OK;
}
R4D_ENTRY(r4d_rad_givens_dense_quant, givens_dense_body)
