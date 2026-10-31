/* r3d_gelu_mul_quant_i8.hip -- gelu_mul_quant_i8: gelu(gate) * up and the per-row symmetric int8
 * quantiser in one pass over the row.
 *
 * The down projection's input of a GELU-gated feed-forward (Gemma) on a w4a8 grid, replacing the
 * gelu_mul + quant_act_i8 pair and byte-identical to it: the product is formed in the tanh form
 * with the same constants, rounded to bf16, and the absmax, scale and codes are quant_act_i8's
 * over those bf16 values. The absmax is a max, so partitioning does not change it.
 *
 *   gate_up [M][2n] bf16 (gate first) -> q [M][n] i8 (WMMA step order), scale [M] f32, out_bf16?
 *
 * The codes are written in the int8 WMMA step order (i8_step_pos), the plane the w4a8/w8a8/w2a8
 * GEMMs read, so n is a multiple of 16.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include <cmath>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr float kTopI8 = 127.0f;
constexpr float kI8Floor = 1e-8f;

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
__device__ __forceinline__ uint32_t i8_step_pos(uint32_t c) {
    return (c & ~12u) | ((c & 4u) << 1) | ((c & 8u) >> 1);
}
__device__ __forceinline__ float block_max(float v, float* lds) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    v = wave_max(v);
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) t = fmaxf(t, lds[w]);
    return t;
}

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void gelu_mul_quant_i8_kernel(
    const uint16_t* __restrict__ gu, uint8_t* __restrict__ q, float* __restrict__ s,
    uint16_t* __restrict__ yo, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    extern __shared__ uint16_t ybuf[];      // n bf16 of the product
    __shared__ float lm[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* gr = gu + (size_t)row * gu_ld;
    uint8_t* qr = q + (size_t)row * n;
    uint16_t* yop = yo ? yo + (size_t)row * y_ld : nullptr;

    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float g = ld_bf(gr, i);
        const float gelu = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
        const uint16_t y = f32_to_bf16(gelu * ld_bf(gr, n + i));
        ybuf[i] = y;
        if (yop) yop[i] = y;
        m = fmaxf(m, fabsf(bf16_to_f32(y)));
    }
    m = block_max(m, lm);
    const float scale = fmaxf(m, kI8Floor) / kTopI8;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) s[row] = scale;

    for (uint32_t c = threadIdx.x * 4; c < n; c += kThreads * 4) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = (c + (uint32_t)j < n) ? bf16_to_f32(ybuf[c + (uint32_t)j])
                                                                  : 0.0f;
        *(uint32_t*)(qr + i8_step_pos(c)) = i8_codes4(v, inv);
    }
}

}  // namespace

extern "C" {

int r3d_rad_gelu_mul_quant_i8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* gu = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2),
                    *y = r3d_opt(a, 3);
    if (!r3d_p(gu) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (gu->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, n = 0;
    if (r3d_rows_of(q, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 16 || !r3d_contig(q)) return R3D_NO(RAD_E_SHAPE);
    // gate_up is [M, 2n]; accept a row pitch of at least 2n. The pitch is the leading stride
    // (r3d_row_pitch), which is shape[1] for a rank-2 row and the whole row extent for a rank-3.
    const long long gu0 = r3d_row_pitch(gu);
    if (gu0 < 2 * n) return R3D_NO(RAD_E_SHAPE);
    const uint32_t gu_ld = (uint32_t)gu0;
    if (r3d_numel(sc) < M) return R3D_NO(RAD_E_SHAPE);
    uint32_t y_ld = n;
    if (y) {
        if (y->dtype != RAD_BF16 || r3d_numel(y) < M * n) return R3D_NO(RAD_E_SHAPE);
        const long long yp = r3d_row_pitch(y);
        if (yp < n) return R3D_NO(RAD_E_STRIDE);
        y_ld = (uint32_t)yp;
    }
    const size_t lds = (size_t)n * sizeof(uint16_t);
    if (lds > 60 * 1024) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kThreads);
    R3D_KLAUNCH_S(gelu_mul_quant_i8_kernel, grid, blk, (size_t)lds, (hipStream_t)s,
                  (const uint16_t*)r3d_p(gu), (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc),
                  (uint16_t*)(y ? r3d_wp(y) : nullptr), (uint32_t)n, gu_ld, y_ld);
    return (int)hipGetLastError();
}

}  // extern "C"
