/* r3d_had_quant_act_fp8.hip -- had_quant_act_fp8: a 128-wide block Hadamard, then per-group E4M3.
 *
 * The rotated form of quant_act_fp8: the 128-wide Hadamard shrinks the group maximum, so e4m3's
 * four exponent bits carry the outliers better after the rotation. The transform is the wave-local
 * FWHT of r3d_fwht.h (unnormalised; the scale carries the 1/sqrt(128) so the dequantised value is
 * the orthonormal rotation). Normalisation is folded into the scale: S = absmax * (1/sqrt(128)) /
 * 448, codes of (v * (128/absmax)) clamped to +-448.
 *
 * A separate op, not a parameter: it writes different numbers than quant_act_fp8, so a swapped
 * graph would run as a different model. The codes are row-major (the fp8 GEMM reads stored order).
 *
 * gfx1102: the FWHT's cross-lane stages are DPP, and a DPP exchange in a loop body miscompiles here
 * (a non-first wave drops out -- the same bug as the GDN recurrent kernel, the MoE router, and the
 * int8 Hadamard quantisers). This serves the single-group form, n == group == 128: one wave runs
 * the whole FWHT with no group loop. A wider n is declined.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_fwht.h"
#include <cmath>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr float kTopE4M3 = 448.0f;

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t e4m3_codes4(const float* v, float inv) {
    const uint32_t lo = e4m3x2_pack(clamp_top(v[0] * inv, kTopE4M3), clamp_top(v[1] * inv, kTopE4M3));
    const uint32_t hi = e4m3x2_pack(clamp_top(v[2] * inv, kTopE4M3), clamp_top(v[3] * inv, kTopE4M3));
    return lo | (hi << 16);
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void had_quant_act_fp8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n) {
    constexpr int HAD = 128, VPL = HAD / 32;
    __shared__ float lds[128];
    __shared__ float red[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* a = x + (size_t)row * n;
    uint8_t* qr = q + (size_t)row * n;
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

    float m = 0.0f;
    if (wave == 0) {   // one wave does the whole single-group FWHT (no group loop, gfx1102)
        const int base = lane * VPL;
        float v[VPL];
#pragma unroll
        for (int j = 0; j < VPL; ++j) v[j] = (base + j < (int)n) ? ld_bf(a, base + j) : 0.0f;
        r3d_fwht_wave<HAD>(v, lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) { lds[base + j] = v[j]; m = fmaxf(m, fabsf(v[j])); }
    }
    m = wave_max_f(m);
    if (lane == 0) red[wave] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) m = fmaxf(m, red[w]);
    const float scale = m > 0.0f ? m * (1.0f / kTopE4M3) : 1.0f;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) s[row] = scale * (1.0f / sqrtf((float)HAD));

    for (uint32_t c = lane * 4; c < n; c += kWave * 4) {
        if (wave == 0) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = lds[c + (uint32_t)j];
            *(uint32_t*)(qr + c) = e4m3_codes4(v, inv);
        }
    }
}

}  // namespace

extern "C" {

int r3d_rad_had_quant_act_fp8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2);
    if (!r3d_p(x) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || q->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x)) return R3D_NO(RAD_E_STRIDE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    if (K <= 0 || K % 4) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    // gfx1102: single-group form (see the file header).
    if (grp != 128 || K != 128) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(q) < M * K || r3d_numel(sc) < M) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kThreads);
    R3D_KLAUNCH(had_quant_act_fp8_kernel, grid, blk, (hipStream_t)s, (const uint16_t*)r3d_p(x),
                (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)K);
    return (int)hipGetLastError();
}

}  // extern "C"
