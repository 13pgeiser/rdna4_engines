/* r3d_gdn_gated_rmsnorm_h128.hip -- the gated RMS norm a gated-delta-net layer applies to its
 * own output before the output projection:
 *
 *     out[r][d] = x[r][d] * rsqrt(mean_d(x[r][:]^2) + eps) * w[d] * act(z[r][d])
 *
 * one row per (token, head), 128 wide; act is silu (the default) or sigmoid. Replaces FLA's
 * rmsnorm_fn (mamba/ops/layernorm_gated). A separate row because the chunked scan's workgroup
 * owns only 64 of a row's 128 v channels; the decode recurrent update owns whole rows and fuses
 * this into its epilogue.
 *
 * One wave per row, four channels per lane: a barrier-free five-shuffle wave sum, and one 8-byte
 * access per lane for each load and the store.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_args.h"
#include "r3d_codecs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

constexpr int kD     = 128;   // channels per row
constexpr int kDPL   = 4;     // channels per lane
constexpr int kNW    = 8;     // rows in flight per workgroup
constexpr int kNTHR  = kNW * 32;

__device__ __forceinline__ float gn_act(float z, int kind)
{
    const float s = 1.0f / (1.0f + __expf(-z));
    return kind ? s : z * s;      // 1 = sigmoid, 0 = silu / swish
}

template <int ACT>
__global__ __launch_bounds__(kNTHR)
void r3d_gdn_gated_rmsnorm_kernel(const unsigned short* __restrict__ x,
                                  const unsigned short* __restrict__ z,
                                  const float* __restrict__ w,
                                  unsigned short* __restrict__ o,
                                  long long rows, long long xrow, long long zrow,
                                  long long orow, long long zheads, long long ztok, float eps)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long r = (long long)blockIdx.x * kNW + warp;
    if (r >= rows) return;
    const int d0 = lane * kDPL;
    // The gate may be a column view of a fused [q|k|v|z] projection: within a token the `zheads`
    // gate rows are `zrow` apart and the next token is `ztok` away. zheads <= 0 is the flat case.
    const long long zoff = zheads > 0 ? (r / zheads) * ztok + (r % zheads) * zrow : r * zrow;

    const ushort4 xv = *(const ushort4*)(x + r * xrow + d0);
    float v[kDPL] = {bf16_to_f32(xv.x), bf16_to_f32(xv.y), bf16_to_f32(xv.z), bf16_to_f32(xv.w)};

    float ss = 0.0f;
#pragma unroll
    for (int c = 0; c < kDPL; ++c) ss += v[c] * v[c];
    ss = wave_sum(ss);
    const float scale = __frsqrt_rn(ss * (1.0f / kD) + eps);

    const ushort4 zv = *(const ushort4*)(z + zoff + d0);
    const float zz[kDPL] = {bf16_to_f32(zv.x), bf16_to_f32(zv.y),
                            bf16_to_f32(zv.z), bf16_to_f32(zv.w)};
    const float4 ww = *(const float4*)(w + d0);
    const float wv[kDPL] = {ww.x, ww.y, ww.z, ww.w};

    unsigned short out[kDPL];
#pragma unroll
    for (int c = 0; c < kDPL; ++c)
        out[c] = f32_to_bf16(v[c] * scale * wv[c] * gn_act(zz[c], ACT));
    *(ushort4*)(o + r * orow + d0) = make_ushort4(out[0], out[1], out[2], out[3]);
}

int r3d_gdn_gated_rmsnorm_launch(const void* x, const void* z, const void* w, void* o,
                                 long long rows, long long xrow, long long zrow, long long orow,
                                 long long zheads, long long ztok, int act, float eps,
                                 hipStream_t stream)
{
    dim3 grid((unsigned)((rows + kNW - 1) / kNW)), blk(kNTHR);
    if (act)
        R3D_KLAUNCH_S(R3D_KNAME((r3d_gdn_gated_rmsnorm_kernel<1>)), grid, blk, 0, stream,
                      (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                      (unsigned short*)o, rows, xrow, zrow, orow, zheads, ztok, eps);
    else
        R3D_KLAUNCH_S(R3D_KNAME((r3d_gdn_gated_rmsnorm_kernel<0>)), grid, blk, 0, stream,
                      (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                      (unsigned short*)o, rows, xrow, zrow, orow, zheads, ztok, eps);
    return (int)hipGetLastError();
}

}  // namespace

/* ------------------------------------------------------------------ the raw entry */

extern "C" int r3d_gdn_gated_rmsnorm_h128(long x, long z, long w, long o, long long rows,
                                          long long xrow, long long zrow, long long orow,
                                          long long zheads, long long ztok, int width,
                                          float eps, int act, long stream)
{
    if (!x || !z || !w || !o) return -1;
    if (width != kD || rows < 1) return -2;
    return r3d_gdn_gated_rmsnorm_launch((const void*)x, (const void*)z, (const void*)w, (void*)o,
                                        rows, xrow, zrow, orow, zheads, ztok, act, eps,
                                        (hipStream_t)stream);
}

/* ------------------------------------------------------------------ the plugin ABI */

enum { GN_X = 0, GN_Z, GN_W, GN_O };

/* The (token, head) row pitch of an operand whose last axis is `width`: the stride of the axis
 * above it (the head axis), -1 when the layout does not put consecutive rows that far apart. */
static long long gn_row_pitch(const RadTensor* t, long long width)
{
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] == width) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : width;
    }
    return r3d_contig(t) ? width : -1;
}

extern "C" int r3d_rad_gdn_gated_rmsnorm(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, GN_X);
    const RadTensor* z = r3d_opt(a, GN_Z);
    const RadTensor* w = r3d_opt(a, GN_W);
    const RadTensor* o = r3d_opt(a, GN_O);
    if (!r3d_p(x) || !r3d_p(z) || !r3d_p(w) || !r3d_wp(o)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(x) || !r3d_is_bf16(z) || !r3d_is_bf16(o) || !r3d_is_f32(w))
        return R3D_NO(RAD_E_DTYPE);
    if (x->rank < 2 || z->rank < 2 || o->rank < 2) return R3D_NO(RAD_E_SHAPE);

    if (!r3d_contig(w)) return R3D_NO(RAD_E_STRIDE);

    // One row is one (token, head). x and out are [tokens, heads, channels] (rank 3, as the chunk
    // scan requires) while the gate may be the flat [tokens, heads*channels] projection. So the
    // width comes from `channels` (or x's last axis), the row count from x, and each operand's
    // pitch between consecutive (token, head) rows is read per its shape.
    const long long width = r3d_geti_or(a, "channels", x->shape[x->rank - 1]);
    if (width <= 0 || r3d_numel(x) % width) return R3D_NO(RAD_E_SHAPE);
    const long long rows = r3d_numel(x) / width;
    if (r3d_numel(z) < rows * width || r3d_numel(o) < rows * width) return R3D_NO(RAD_E_SHAPE);

    const long long xrow = gn_row_pitch(x, width);
    const long long orow = gn_row_pitch(o, width);
    if (xrow < 0 || orow < 0) return R3D_NO(RAD_E_STRIDE);

    // The gate may also be a column slice of a wider [q|k|v|z] row (fp8 delta net): heads are
    // `width` apart within a token and tokens are the slice's stride apart, so the kernel also
    // takes the head count. zheads stays 0 for the flat layouts.
    long long zrow = gn_row_pitch(z, width), zheads = 0, ztok = 0;
    if (zrow < 0 && z->rank == 2 && z->stride[1] == 1 && z->shape[1] % width == 0) {
        zheads = z->shape[1] / width;
        ztok   = z->stride[0];
        zrow   = width;
        if (zheads <= 0 || rows % zheads) return R3D_NO(RAD_E_SHAPE);
        if (x->rank >= 3 && x->shape[x->rank - 2] != zheads) return R3D_NO(RAD_E_SHAPE);
    }
    if (zrow < 0) return R3D_NO(RAD_E_STRIDE);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const int   act = r3d_act_code(a, "act", nullptr, -1);
    if (act != 0 && act != 1) return R3D_NO(RAD_E_INVAL);

    return r3d_gdn_gated_rmsnorm_launch(r3d_p(x), r3d_p(z), r3d_p(w), r3d_wp(o),
                                        rows, xrow, zrow, orow, zheads, ztok,
                                        act, eps, (hipStream_t)s);
}
