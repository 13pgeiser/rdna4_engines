/* r3d_rmsnorm_quant_i8.hip -- rmsnorm_quant_i8: the RMS norm and the per-row int8 quantiser, in
 * one pass over the row, optionally with the residual junction in front of it: a post-norm on the
 * branch output, the residual add and a scalar on the sum.
 *
 * A w4a8 linear needs its activation as int8 codes in the GEMM's fragment order with one f32 scale
 * per row, produced by a norm. Unfused that is two dispatches, and a sandwich-norm model issues
 * four or five at every residual junction:
 *
 *     rmsnorm(branch; w_post) -> add(residual) -> [scale] -> rmsnorm(residual; w) -> quant
 *
 * This is the chain as one dispatch. Byte-identical to the chain, because every intermediate the
 * chain stores in bf16 is rounded to bf16 here and the next stage reads the rounded value: the
 * post-norm output, the residual sum, the scaled residual and the norm output the quantiser
 * reduces over. Both sums of squares use the standalone norm's partition (thread t folding columns
 * t, t+THR, ... in order, then the wave sum and the waves in order through LDS).
 *
 * The codes are in the w4a8 GEMM's fragment order: byte j of a k step comes from source
 * 8*((j&7)>>2) + 4*(j>>3) + (j&3) inside the step.
 *
 * THR is fixed at 256 (the re-read form): one workgroup a row, the row re-read in each pass. The
 * register form (decode heights) and the fused all-reduce forms are not served here.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include <cstdint>

#define R3D_BARRIER() __syncthreads(); asm volatile("" ::: "memory");

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr float kTopI8 = 127.0f;
constexpr float kFloor = 1e-8f;

__device__ __forceinline__ float bf2f(uint16_t h) { return bf16_to_f32(h); }

template <bool WF32>
__device__ __forceinline__ float gain(const void* w, uint32_t i) {
    return WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
}
__device__ __forceinline__ float scalar_of(const void* sc, uint32_t f32) {
    return f32 ? ((const float*)sc)[0] : bf16_to_f32(((const uint16_t*)sc)[0]);
}

// The row body from the junction on: `s1` is the post-norm's factor, already reduced over `xr`.
template <bool WF32, bool RES, bool POST, bool SCL>
__device__ __forceinline__ void junction_row(
    const uint16_t* __restrict__ xr, uint16_t* __restrict__ rr, const void* __restrict__ w,
    const void* __restrict__ wp, float sv, signed char* __restrict__ qr, float* __restrict__ Sr,
    uint16_t* __restrict__ yr, uint32_t n, float eps, float wadd, float s1, uint16_t* ybuf,
    float* l2, float* lm) {
    // ---- the junction (post-norm, add, scalar), stored, and the norm's sum of squares over it
    float ss2 = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        uint16_t v = xr[i];
        if (RES) {
            if (POST) v = f32_to_bf16(bf2f(v) * s1 * (gain<WF32>(wp, i) + wadd));
            v = f32_to_bf16(bf2f(rr[i]) + bf2f(v));
            if (SCL) v = f32_to_bf16(bf2f(v) * sv);
            rr[i] = v;
        }
        const float f = bf2f(v);
        ss2 += f * f;
    }
    ss2 = block_sum<kThreads>(ss2, l2);
    const float s2 = rsqrtf(ss2 / (float)n + eps);

    // ---- the norm output, rounded to bf16 as the standalone norm stores it, and its absmax
    const uint16_t* __restrict__ nr = RES ? rr : xr;
    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const uint16_t y = f32_to_bf16(bf2f(nr[i]) * s2 * (gain<WF32>(w, i) + wadd));
        ybuf[i] = y;
        if (yr) yr[i] = y;
        m = fmaxf(m, fabsf(bf2f(y)));
    }
    m = wave_max(m);
    if ((threadIdx.x & 31) == 0) lm[threadIdx.x >> 5] = m;
    R3D_BARRIER();   // also publishes ybuf to every thread
    m = lm[0];
#pragma unroll
    for (uint32_t i = 1; i < kWaves; ++i) m = fmaxf(m, lm[i]);
    const float scale = fmaxf(m, kFloor) / kTopI8;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) *Sr = scale;

    // ---- the codes, in the w4a8 GEMM's fragment order
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const uint32_t s = i >> 4, j = i & 15;
        const uint32_t src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = (int)lrintf(bf2f(ybuf[src]) * inv);
        qr[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}

template <bool WF32, bool RES, bool POST, bool SCL>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void rmsnorm_quant_i8_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo, uint32_t n,
    float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld) {
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[kWaves], l2[kWaves], lm[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;

    // ---- the post-norm's sum of squares, over the branch output
    float s1 = 0.0f;
    if (POST) {
        float ss = 0.0f;
        for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
            const float v = bf2f(xr[i]);
            ss += v * v;
        }
        ss = block_sum<kThreads>(ss, l1);
        s1 = rsqrtf(ss / (float)n + eps);
    }
    junction_row<WF32, RES, POST, SCL>(
        xr, RES ? res + (size_t)row * r_ld : nullptr, w, wp, SCL ? scalar_of(sc, sc_f32) : 1.0f,
        q + (size_t)row * n, S + row, yo ? yo + (size_t)row * y_ld : nullptr, n, eps, wadd, s1,
        ybuf, l2, lm);
}

}  // namespace

extern "C" {

int r3d_rad_rmsnorm_quant_i8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *rs = r3d_opt(a, 1), *w = r3d_opt(a, 2), *q = r3d_opt(a, 3),
                    *sc = r3d_opt(a, 4), *y = r3d_opt(a, 5), *wp = r3d_opt(a, 6),
                    *sv = r3d_opt(a, 7);
    if (!r3d_p(x) || !r3d_p(w) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if ((wp || sv) && !rs) return R3D_NO(RAD_E_INVAL);   // a junction without a stream to add into
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, n = 0;
    if (r3d_rows_of(x, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 16) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(w) != n) return R3D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (wp && (r3d_numel(wp) != n || wp->dtype != w->dtype)) return R3D_NO(RAD_E_SHAPE);
    if (sv && (r3d_numel(sv) != 1 || (sv->dtype != RAD_F32 && sv->dtype != RAD_BF16)))
        return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(q) < M * n || r3d_numel(sc) < M) return R3D_NO(RAD_E_SHAPE);
    if (y && r3d_numel(y) < M * n) return R3D_NO(RAD_E_SHAPE);
    if (rs && r3d_numel(rs) < M * n) return R3D_NO(RAD_E_SHAPE);

    const long long x_ld_ = r3d_row_pitch(x);
    if (x_ld_ < 0) return R3D_NO(RAD_E_STRIDE);
    const uint32_t x_ld = (uint32_t)x_ld_;
    uint32_t r_ld = (uint32_t)n;
    if (rs) {
        const long long rl = r3d_row_pitch(rs);
        if (rl < n) return R3D_NO(RAD_E_STRIDE);
        r_ld = (uint32_t)rl;
    }
    uint32_t y_ld = (uint32_t)n;
    if (y) {
        const long long yl = r3d_row_pitch(y);
        if (yl < n) return R3D_NO(RAD_E_STRIDE);
        y_ld = (uint32_t)yl;
    }
    const size_t lds = (size_t)n * sizeof(uint16_t);
    if (lds > 60 * 1024) return R3D_NO(RAD_E_SHAPE);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const uint32_t scf = sv ? (sv->dtype == RAD_F32) : 0u;
    const bool WF = w->dtype == RAD_F32, RS = rs != nullptr, PO = wp != nullptr, SC = sv != nullptr;

    dim3 grid((unsigned)M), blk(kThreads);
    const uint16_t* xp = (const uint16_t*)r3d_p(x);
    uint16_t* rwp = rs ? (uint16_t*)r3d_wp(rs) : nullptr;
    const void* wv = r3d_p(w);
    const void* wpp = r3d_p(wp);
    const void* svp = r3d_p(sv);
    signed char* qp = (signed char*)r3d_wp(q);
    float* sp = (float*)r3d_wp(sc);
    uint16_t* yp = y ? (uint16_t*)r3d_wp(y) : nullptr;

#define RQ8(WF_, RS_, PO_, SC_)                                                                  \
    R3D_KLAUNCH_S(R3D_KNAME(rmsnorm_quant_i8_kernel<WF_, RS_, PO_, SC_>), grid, blk, lds,        \
                  (hipStream_t)s, xp, rwp, wv, wpp, svp, scf, qp, sp, yp, (uint32_t)n, eps, wadd,\
                  x_ld, r_ld, y_ld)
    if (WF) {
        if (!RS) RQ8(true, false, false, false);
        else if (!PO && !SC) RQ8(true, true, false, false);
        else if (PO && !SC) RQ8(true, true, true, false);
        else if (PO && SC) RQ8(true, true, true, true);
        else RQ8(true, true, false, true);
    } else {
        if (!RS) RQ8(false, false, false, false);
        else if (!PO && !SC) RQ8(false, true, false, false);
        else if (PO && !SC) RQ8(false, true, true, false);
        else if (PO && SC) RQ8(false, true, true, true);
        else RQ8(false, true, false, true);
    }
#undef RQ8
    return (int)hipGetLastError();
}

}  // extern "C"
