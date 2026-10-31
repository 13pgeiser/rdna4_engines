/* r3d_gated_had_quant_i8g.hip -- gated_had_quant_i8g: a gated product, a 128-wide block Hadamard,
 * then per-group int8 (row-major), in one pass.
 *
 * mode 0 (silu) is down_proj's input, mode 1 (sigmoid) a gated attention o_proj's. The gate and up
 * arrive as two halves of a fused gate_up of row pitch 2*n. The gated product is formed in fp32,
 * the 128-wide Hadamard shrinks the group maximum, and the codes are row-major int8 (the i8a8 dot2
 * GEMM reads stored order), with one scale per (row, group). The normalisation (1/sqrt(128)) is
 * folded into the scale.
 *
 * gfx1102: the FWHT's cross-lane stages are DPP, and a DPP exchange in a loop body miscompiles here
 * (a non-first wave drops out). This serves the single-group form, n == group == 128: one wave runs
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
constexpr float kTopI8 = 127.0f;

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

template <int MODE>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void gated_had_quant_i8g_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ yo, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    constexpr int HAD = 128, VPL = HAD / 32;
    __shared__ float lds[128];
    __shared__ float red[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* g = gu + (size_t)row * gu_ld;
    const uint16_t* u = g + n;
    uint8_t* qr = q + (size_t)row * n;
    uint16_t* yop = yo ? yo + (size_t)row * y_ld : nullptr;
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

    float m = 0.0f;
    if (wave == 0) {   // one wave does the whole single-group FWHT (no group loop, gfx1102)
        const int base = lane * VPL;
        float v[VPL];
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            const float xg = ld_bf(g, base + j);
            const float sg = 1.0f / (1.0f + expf(-xg));   /* full exp: the reference's sigmoid */
            v[j] = (MODE == 0 ? xg * sg : sg) * ld_bf(u, base + j);
        }
        r3d_fwht_wave<HAD>(v, lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            lds[base + j] = v[j];
            m = fmaxf(m, fabsf(v[j]));
            if (yop) yop[base + j] = f32_to_bf16(v[j]);
        }
    }
    m = wave_max_f(m);
    if (lane == 0) red[wave] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) m = fmaxf(m, red[w]);
    m = m > 0.0f ? m : 1e-8f;
    const float inv = kTopI8 / m;
    if (threadIdx.x == 0) s[row] = m * (1.0f / sqrtf((float)HAD)) / kTopI8;

    if (wave == 0)
        for (uint32_t c = lane * 4; c < n; c += kWave * 4) {
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = lds[c + (uint32_t)j];
            *(uint32_t*)(qr + c) = i8_codes4(v, inv);   // row-major (the i8a8 GEMM's layout)
        }
}

}  // namespace

extern "C" {

int r3d_rad_gated_had_quant_i8g(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    /* The schema's order: gate_up, q, scale, out_bf16. */
    const RadTensor* gu = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2),
                    *y = r3d_opt(a, 3);
    if (!r3d_p(gu) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (gu->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, n = 0;
    if (r3d_rows_of(q, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 4) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    if (grp != 128 || n != 128) return R3D_NO(RAD_E_SHAPE);   // single-group form (gfx1102)
    if (r3d_numel(gu) < M * 2 * n) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(sc) < M) return R3D_NO(RAD_E_SHAPE);
    uint32_t y_ld = n;
    if (y) {
        if (y->dtype != RAD_BF16 || r3d_numel(y) < M * n) return R3D_NO(RAD_E_SHAPE);
        if (y->rank >= 2 && y->shape[1] > n) y_ld = (uint32_t)y->shape[1];
    }

    const char* act = r3d_gets(a, "act");
    const long long mode = act && 0 == std::strcmp(act, "sigmoid") ? 1 : 0;
    const uint32_t gu_ld = (uint32_t)(gu->rank >= 2 ? gu->shape[1] : gu->stride[0]);
    dim3 grid((unsigned)M), blk(kThreads);
    if (mode == 0)
        R3D_KLAUNCH(R3D_KNAME(gated_had_quant_i8g_kernel<0>), grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(gu), (uint16_t*)(y ? r3d_wp(y) : nullptr),
                    (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)n, gu_ld, y_ld);
    else
        R3D_KLAUNCH(R3D_KNAME(gated_had_quant_i8g_kernel<1>), grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(gu), (uint16_t*)(y ? r3d_wp(y) : nullptr),
                    (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)n, gu_ld, y_ld);
    return (int)hipGetLastError();
}

}  // extern "C"
