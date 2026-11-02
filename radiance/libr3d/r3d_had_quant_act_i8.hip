/* r3d_had_quant_act_i8.hip -- had_quant_act_i8: a block Hadamard along K, then per-row symmetric
 * int8 quantisation, written in the int8 WMMA step order.
 *
 * The rotation: plain per-token int8 activations are unusable on these models (KL against bf16 an
 * order of magnitude worse than per-token e4m3); a block Hadamard first makes int8 several times
 * better than e4m3. A rotation fixes range, not precision: int8 spends its grid covering a few
 * outlier channels, and mixing each value with HAD-1 others removes them. It must happen before
 * the absmax, since the benefit is a smaller row maximum, so it belongs in the quantiser.
 *
 * The transform is the wave-local FWHT of r3d_fwht.h: one wave owns a HAD-wide block, VPL = HAD/32
 * contiguous values per lane. Normalisation is folded into the scale: the stages compute the
 * unnormalised transform (sqrt(HAD) times the orthonormal one), and S[row] = absmax * (1/sqrt(HAD))
 * / 127 undoes it while inv = 127/absmax gives the normalised codes.
 *
 * gfx1102 note: the FWHT's cross-lane stages are DPP exchanges, and a DPP exchange inside a loop
 * body miscompiles on gfx1102 (a non-first wave drops out of the loop; the same bug as the GDN
 * recurrent kernel and the MoE router). This kernel therefore serves the single-block form,
 * K == group: one wave runs the whole FWHT with no block loop, the rest quantise from LDS. A
 * wider K (several rotation blocks) would put the transform in a loop and is declined.
 *
 * The codes are written in the int8 WMMA step order (i8_step_pos, r3d_quant_fp8.hip), the same
 * plane the w4a8/w8a8/w2a8 GEMMs read, so n is a multiple of 16.
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

template <int HAD>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void had_quant_act_i8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    float rnorm) {
    constexpr int VPL = HAD / 32;
    __shared__ float lds[1024];     // the transformed row (n <= 1024)
    __shared__ float red[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* a = x + (size_t)row * n;
    uint8_t* qr = q + (size_t)row * n;
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

    // One wave does the whole single-block FWHT: no block loop, so its DPP cross-lane stages stay
    // out of a loop body (a DPP exchange in a loop miscompiles on gfx1102 -- see the file header).
    float m = 0.0f;
    if (wave == 0) {
        const int base = lane * VPL;
        float v[VPL];
#pragma unroll
        for (int j = 0; j < VPL; ++j) v[j] = ld_bf(a, base + j);
        r3d_fwht_wave<HAD>(v, lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) { lds[base + j] = v[j]; m = fmaxf(m, fabsf(v[j])); }
    }
    m = block_max(m, red);
    m = fmaxf(m, kI8Floor);
    const float inv = kTopI8 / m;
    if (threadIdx.x == 0) s[row] = m * rnorm / kTopI8;

    // Write the codes in the WMMA step order, reading the transformed row from LDS.
    for (uint32_t c = threadIdx.x * 4; c < n; c += kThreads * 4) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = (c + (uint32_t)j < n) ? lds[c + (uint32_t)j] : 0.0f;
        *(uint32_t*)(qr + i8_step_pos(c)) = i8_codes4(v, inv);
    }
}

}  // namespace

extern "C" {

int r3d_rad_had_quant_act_i8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2);
    if (!r3d_p(x) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_contig(x) || !r3d_contig(q) || !r3d_contig(sc)) return R3D_NO(RAD_E_STRIDE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    if (K <= 0 || K % 16) return R3D_NO(RAD_E_SHAPE);

    const long long had = r3d_geti_or(a, "group", 128);
    if (had != 128 && had != 256 && had != 512) return R3D_NO(RAD_E_SHAPE);
    // gfx1102: the FWHT is DPP and must not run in a block loop (see the file header), so this
    // serves the single-block form, K == group. A wider K (several rotation blocks) is declined.
    if (K != had) return R3D_NO(RAD_E_SHAPE);

    const float rnorm = 1.0f / sqrtf((float)had);
    dim3 grid((unsigned)M), blk(kThreads);
    const uint32_t n = (uint32_t)K;
    switch (had) {
        case 256: R3D_KLAUNCH(had_quant_act_i8_kernel<256>, grid, blk, (hipStream_t)s,
                              (const uint16_t*)r3d_p(x), (uint8_t*)r3d_wp(q),
                              (float*)r3d_wp(sc), n, rnorm); break;
        case 512: R3D_KLAUNCH(had_quant_act_i8_kernel<512>, grid, blk, (hipStream_t)s,
                              (const uint16_t*)r3d_p(x), (uint8_t*)r3d_wp(q),
                              (float*)r3d_wp(sc), n, rnorm); break;
        default:  R3D_KLAUNCH(had_quant_act_i8_kernel<128>, grid, blk, (hipStream_t)s,
                              (const uint16_t*)r3d_p(x), (uint8_t*)r3d_wp(q),
                              (float*)r3d_wp(sc), n, rnorm); break;
    }
    return (int)hipGetLastError();
}

}  // extern "C"
