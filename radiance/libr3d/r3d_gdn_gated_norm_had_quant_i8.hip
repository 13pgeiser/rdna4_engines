/* r3d_gdn_gated_norm_had_quant_i8.hip -- the GDN block's gated RMS norm, block Hadamard and
 * per-row int8 quantiser in one pass: out_proj's input on the 48 linear-attention layers (the
 * fourth rotated site in a W4A8 layer).
 *
 *     out = x * rsqrt(mean_head(x^2) + eps) * w * silu(z)
 *
 * Two extents: the RMS is per head (head_dim = 128), while the rotation and the int8 scale are
 * per token over the flattened H*128. The reduction is segmented: a head is 128/VPL consecutive
 * lanes, and the xor masks below that stay inside one head.
 *
 *     HAD=128  VPL=4   32 lanes/head   one head per rotation block
 *     HAD=256  VPL=8   16 lanes/head   two heads per block
 *     HAD=512  VPL=16   8 lanes/head   four heads per block
 *
 * At width 128 the rotation mixes only within a head; wider widths mix across heads.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_fwht.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>

namespace {

constexpr int kHEAD = 128;   // head_dim

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

__host__ __device__ __forceinline__ int r3d_act_threads(int M, int K)
{
    if (M <= 32) return 1024;
    const int lim = K >= 8192 ? (1 << 24) : (K >= 4096 ? 120 : 72);
    return M <= lim ? 512 : 256;
}

template <int HAD, int THR>
__global__ __launch_bounds__(THR) void r3d_gdn_gated_norm_had_quant_i8_kernel(
    const unsigned short* __restrict__ X, const unsigned short* __restrict__ Z,
    const float* __restrict__ Wg, signed char* __restrict__ Q, float* __restrict__ S,
    int K, long long xrow, long long zrow, int zgrp, int zblk, float eps, float rnorm)
{
    constexpr int VPL = HAD / 32, NW = THR / 32;
    constexpr int LPH = kHEAD / VPL;          // lanes spanned by one head
    extern __shared__ float lds[];
    float* red = lds + K;
    const int row = blockIdx.x;
    const unsigned short* x = X + (size_t)row * xrow;
    const unsigned short* z = Z + (size_t)row * zrow;
    // zgrp != 0: Z is read in place from the interleaved qkvz projection (avoiding a `z_out[:] = z`
    // copy per layer): element i is at (i / zblk) * zgrp + (i % zblk). zblk is a multiple of HAD,
    // so a rotation block never straddles a group.
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

    float m = 0.f;
    for (int blk = wave; blk < K / HAD; blk += NW) {
        const int base = blk * HAD + lane * VPL;
        const int zbase = zgrp ? (base / zblk) * zgrp + (base % zblk) : base;
        float v[VPL];
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            v[j] = bf16_to_f32(x[base + j]);
            ss += v[j] * v[j];
        }
        // segmented: masks below LPH never cross a head
        ss = r3d_xor_sum<LPH>(ss);
        const float rms = __frsqrt_rn(ss * (1.f / (float)kHEAD) + eps);
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            const float g = bf16_to_f32(z[zbase + j]);
            v[j] = v[j] * rms * Wg[(base + j) & (kHEAD - 1)] * (g / (1.f + __expf(-g)));
        }
        r3d_fwht_wave<HAD>(v, lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            lds[base + j] = v[j];
            m = fmaxf(m, fabsf(v[j]));
        }
    }

    m = wave_max(m);
    if (lane == 0) red[wave] = m;
    R3D_BARRIER();
    m = red[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) m = fmaxf(m, red[i]);
    m = fmaxf(m, 1e-8f);
    const float inv = 127.f / m;
    if (threadIdx.x == 0) S[row] = m * rnorm / 127.f;

    signed char* q = Q + (size_t)row * K;
    for (int i = threadIdx.x; i < K; i += THR) {
        const int st = i >> 4, j = i & 15;
        const int sr = (st << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = __float2int_rn(lds[sr] * inv);
        q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}

int r3d_gdn_gated_norm_hq_i8_launch(const void* x, const void* z, const void* w, void* q,
                                    void* sc, int M, int K, long long xrow, long long zrow,
                                    int zgrp, int zblk, int had, float eps, hipStream_t stream)
{
    const float rn = 1.f / sqrtf((float)had);
    const long long xr = xrow ? xrow : K, zr = zrow ? zrow : K;
    const int thr = r3d_act_threads(M, K);
    const size_t shmem = (size_t)K * sizeof(float) + (size_t)(thr / 32) * sizeof(float);
    if (shmem > 65536) return -3;
#define R3D_GNHQ_LAUNCH(H, T) \
    R3D_KLAUNCH_S(R3D_KNAME((r3d_gdn_gated_norm_had_quant_i8_kernel<H, T>)), dim3((unsigned)M), \
                  dim3(T), shmem, stream, (const unsigned short*)x, (const unsigned short*)z, \
                  (const float*)w, (signed char*)q, (float*)sc, K, xr, zr, zgrp, zblk, eps, rn)
#define R3D_GNHQ_T(H) switch (thr) { \
    case 256:  R3D_GNHQ_LAUNCH(H, 256);  break; \
    case 512:  R3D_GNHQ_LAUNCH(H, 512);  break; \
    case 1024: R3D_GNHQ_LAUNCH(H, 1024); break; \
    default: return -2; }
    switch (had) {
        case 128:  R3D_GNHQ_T(128); break;
        case 256:  R3D_GNHQ_T(256); break;
        default:   R3D_GNHQ_T(512); break;
    }
#undef R3D_GNHQ_T
#undef R3D_GNHQ_LAUNCH
    return (int)hipGetLastError();
}

}  // namespace

/* ------------------------------------------------------------------ the plugin ABI */

enum { GNH_X = 0, GNH_Z, GNH_W, GNH_Q, GNH_S };

extern "C" int r3d_rad_gdn_gated_norm_hq_i8(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x  = r3d_opt(a, GNH_X);
    const RadTensor* z  = r3d_opt(a, GNH_Z);
    const RadTensor* w  = r3d_opt(a, GNH_W);
    const RadTensor* q  = r3d_opt(a, GNH_Q);
    const RadTensor* sc = r3d_opt(a, GNH_S);
    if (!r3d_p(x) || !r3d_p(z) || !r3d_p(w) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(x) || !r3d_is_bf16(z) || !r3d_is_f32(w)) return R3D_NO(RAD_E_DTYPE);
    if (q->dtype != RAD_I8 && q->dtype != RAD_U8) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(sc)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(q) || !r3d_contig(sc) || !r3d_contig(w)) return R3D_NO(RAD_E_STRIDE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != 0) return R3D_NO(RAD_E_SHAPE);
    const long long head = r3d_geti_or(a, "head_dim", kHEAD);
    if (head != kHEAD) return R3D_NO(RAD_E_SHAPE);
    if (K % kHEAD) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(w) != head) return R3D_NO(RAD_E_SHAPE);   // wgain is 128 floats, not K

    const long long had = r3d_geti_or(a, "group", 0);
    if (had != 128 && had != 256 && had != 512) return R3D_NO(RAD_E_SHAPE);
    // A rotation block cannot straddle a tensor-parallel shard boundary.
    if (K % had) return R3D_NO(RAD_E_SHAPE);

    const long long xrow = r3d_row_pitch(x);
    const long long zrow = r3d_row_pitch(z);
    if (xrow < 0 || zrow < 0) return R3D_NO(RAD_E_STRIDE);

    // Interleaved-gate layout (see the kernel); zgrp 0 means contiguous.
    const long long zgrp = r3d_geti_or(a, "zgrp", 0);
    const long long zblk = r3d_geti_or(a, "zblk", 0);
    if (zgrp && (zblk % had || zblk < had || zgrp < zblk)) return R3D_NO(RAD_E_INVAL);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);

    return r3d_gdn_gated_norm_hq_i8_launch(r3d_p(x), r3d_p(z), r3d_p(w), r3d_wp(q), r3d_wp(sc),
                                           (int)M, (int)K, xrow, zrow, (int)zgrp, (int)zblk,
                                           (int)had, eps, (hipStream_t)s);
}
