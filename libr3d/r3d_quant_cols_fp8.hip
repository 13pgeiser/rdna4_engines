/* r3d_quant_cols_fp8.hip -- quant_cols_fp8: E4M3 codes with one scale per column over every row.
 *
 * An attention's V for its fp8 P.V: the formula is quant_act_fp8's (scale amax / 448, 1 for an
 * all-zero column, codes of x / scale clamped to +-448) with the group running down a column. A
 * column's scale needs every row, so two passes over the rows, split into kColChunks chunks:
 *   1. column amax partials, a workgroup a block of columns and a chunk of rows, the bf16
 *      magnitudes compared as integers (their order is the values');
 *   2. a workgroup the same block and chunk: the scales from the chunks' partials, then the codes,
 *      the chunk's rows walked from its end (the first pass read them last, the cache still holds
 *      them).
 *
 * A thread owns eight columns: one 16-byte load and one 8-byte store a row. RDNA3 has no fp8
 * hardware, so the codes use the software E4M3 codec (r3d_codecs.h), not the cvt_pk_fp8 builtin.
 *
 * Scratch: kColChunks * n floats (the column amax partials).
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr int kColChunks = 32;
constexpr int kColThreads = 128;
constexpr float kTopE4M3 = 448.0f;

// The two bf16 in a 32-bit word, sign masked off: the magnitude, ordered as the values.
__device__ __forceinline__ uint32_t col_mag(uint32_t w) { return w & 0x7fff7fffu; }
__device__ __forceinline__ float bf16w_lo(uint32_t w) { return bf16_to_f32((uint16_t)(w & 0xffffu)); }
__device__ __forceinline__ float bf16w_hi(uint32_t w) { return bf16_to_f32((uint16_t)(w >> 16)); }
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
// Two E4M3 codes packed into a 16-bit value, the software codec (RDNA3 has no fp8 builtin).
__device__ __forceinline__ uint16_t col_code2(float a, float b, float ia, float ib) {
    const uint8_t lo = f32_to_e4m3(clamp_top(a * ia, kTopE4M3));
    const uint8_t hi = f32_to_e4m3(clamp_top(b * ib, kTopE4M3));
    return (uint16_t)(lo | (hi << 8));
}

__global__ __launch_bounds__(kColThreads) void quant_cols_amax_kernel(
    const uint16_t* __restrict__ x, int M, int n, int x_ld, float* __restrict__ part) {
    const int c8 = blockIdx.x * kColThreads + threadIdx.x;
    if (c8 * 8 >= n) return;
    const int r0 = (int)((long)M * blockIdx.y / kColChunks);
    const int r1 = (int)((long)M * (blockIdx.y + 1) / kColChunks);
    const uint16_t* p = x + (long)c8 * 8;
    uint32_t m[4] = { 0, 0, 0, 0 };
    int r = r0;
    for (; r + 8 <= r1; r += 8) {
        uint4 u[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) u[k] = *(const uint4*)(p + (long)(r + k) * x_ld);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            m[0] = max(m[0], col_mag(u[k].x));
            m[1] = max(m[1], col_mag(u[k].y));
            m[2] = max(m[2], col_mag(u[k].z));
            m[3] = max(m[3], col_mag(u[k].w));
        }
    }
    for (; r < r1; ++r) {
        const uint4 u = *(const uint4*)(p + (long)r * x_ld);
        m[0] = max(m[0], col_mag(u.x));
        m[1] = max(m[1], col_mag(u.y));
        m[2] = max(m[2], col_mag(u.z));
        m[3] = max(m[3], col_mag(u.w));
    }
    float* d = part + (long)blockIdx.y * n + c8 * 8;
    *(float4*)d = make_float4(bf16w_lo(m[0]), bf16w_hi(m[0]), bf16w_lo(m[1]), bf16w_hi(m[1]));
    *(float4*)(d + 4) = make_float4(bf16w_lo(m[2]), bf16w_hi(m[2]), bf16w_lo(m[3]), bf16w_hi(m[3]));
}

__global__ __launch_bounds__(kColThreads) void quant_cols_code_kernel(
    const uint16_t* __restrict__ x, int M, int n, int x_ld, const float* __restrict__ part,
    uint8_t* __restrict__ y, int y_ld, float* __restrict__ scale) {
    const int c8 = blockIdx.x * kColThreads + threadIdx.x;
    if (c8 * 8 >= n) return;
    float am[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int ch = 0; ch < kColChunks; ++ch) {
        const float4 a = *(const float4*)(part + (long)ch * n + c8 * 8);
        const float4 b = *(const float4*)(part + (long)ch * n + c8 * 8 + 4);
        am[0] = fmaxf(am[0], a.x); am[1] = fmaxf(am[1], a.y);
        am[2] = fmaxf(am[2], a.z); am[3] = fmaxf(am[3], a.w);
        am[4] = fmaxf(am[4], b.x); am[5] = fmaxf(am[5], b.y);
        am[6] = fmaxf(am[6], b.z); am[7] = fmaxf(am[7], b.w);
    }
    float sc[8], inv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        sc[e] = am[e] > 0.0f ? am[e] * (1.0f / kTopE4M3) : 1.0f;
        inv[e] = 1.0f / sc[e];
    }
    if (blockIdx.y == 0) {
        *(float4*)(scale + c8 * 8) = make_float4(sc[0], sc[1], sc[2], sc[3]);
        *(float4*)(scale + c8 * 8 + 4) = make_float4(sc[4], sc[5], sc[6], sc[7]);
    }
    const int r0 = (int)((long)M * blockIdx.y / kColChunks);
    const int r1 = (int)((long)M * (blockIdx.y + 1) / kColChunks);
    const uint16_t* p = x + (long)c8 * 8;
    uint8_t* q = y + (long)c8 * 8;
    auto code = [&](uint4 u) -> uint2 {
        uint32_t lo = col_code2(bf16w_lo(u.x), bf16w_hi(u.x), inv[0], inv[1]);
        lo |= (uint32_t)col_code2(bf16w_lo(u.y), bf16w_hi(u.y), inv[2], inv[3]) << 16;
        uint32_t hi = col_code2(bf16w_lo(u.z), bf16w_hi(u.z), inv[4], inv[5]);
        hi |= (uint32_t)col_code2(bf16w_lo(u.w), bf16w_hi(u.w), inv[6], inv[7]) << 16;
        return make_uint2(lo, hi);
    };
    int r = r1;
    for (; r - 8 >= r0; r -= 8) {
        uint4 u[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) u[k] = *(const uint4*)(p + (long)(r - 1 - k) * x_ld);
#pragma unroll
        for (int k = 0; k < 8; ++k) *(uint2*)(q + (long)(r - 1 - k) * y_ld) = code(u[k]);
    }
    for (; r > r0; --r)
        *(uint2*)(q + (long)(r - 1) * y_ld) = code(*(const uint4*)(p + (long)(r - 1) * x_ld));
}

}  // namespace

extern "C" {

int64_t r3d_quant_cols_fp8_scratch(const RadArgs* a)
{
    const long long n = r3d_geti_or(a, "n", 0);
    if (n <= 0) return 0;
    return (int64_t)kColChunks * n * (int64_t)sizeof(float);
}

int r3d_rad_quant_cols_fp8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *y = r3d_opt(a, 1), *sc = r3d_opt(a, 2);
    if (!r3d_p(x) || !r3d_wp(y) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0);
    if (n <= 0 || n % 8) return R3D_NO(RAD_E_SHAPE);
    /* The rows are the leading axes, whatever their rank; the pitches are the leading strides.
     * All three are cast to int for the launch, so bound them. */
    if (n > (1LL << 30)) return R3D_NO(RAD_E_SHAPE);
    const long long x_ld = r3d_row_pitch(x);
    const long long y_ld = r3d_row_pitch(y);
    if (x_ld < n || y_ld < n || (x_ld & 7) || (y_ld & 7)) return R3D_NO(RAD_E_SHAPE);
    const long long xnel = r3d_numel(x);
    if (xnel % n) return R3D_NO(RAD_E_SHAPE);
    const long long M = xnel / n;
    if (M < 1 || M > (1LL << 30)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(y) < M * n || r3d_numel(sc) < n) return R3D_NO(RAD_E_SHAPE);
    const int64_t need = (int64_t)kColChunks * n * sizeof(float);
    if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    float* part = (float*)a->scratch;

    const dim3 grid((unsigned)((n / 8 + kColThreads - 1) / kColThreads), (unsigned)kColChunks);
    const dim3 blk(kColThreads);
    R3D_KLAUNCH(quant_cols_amax_kernel, grid, blk, (hipStream_t)s, (const uint16_t*)r3d_p(x),
                (int)M, (int)n, (int)x_ld, part);
    R3D_KLAUNCH(quant_cols_code_kernel, grid, blk, (hipStream_t)s, (const uint16_t*)r3d_p(x),
                (int)M, (int)n, (int)x_ld, part, (uint8_t*)r3d_wp(y), (int)y_ld,
                (float*)r3d_wp(sc));
    return (int)hipGetLastError();
}

}  // extern "C"
