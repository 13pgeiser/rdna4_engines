/* g4k_norm.hip -- Gemma 4's sandwich-norm joins, each one kernel instead of a chain of small ones.
 *
 * A decode step of the 26B-A4B launches ~250 rmsnorms and ~90 adds a step, each a few hundred
 * bytes of work behind a full launch: --profile-ops measured 2.0 ms and 0.85 ms of a 16.6 ms step.
 * Two places in every layer are pure chains of them:
 *
 *   norm_add_norm    the attention's join:   t = norm(a) * w_a;  x += t;  y = norm(x) * w_y
 *                    (post_attention_layernorm, the residual add, pre_feedforward_layernorm)
 *   join_add_scale   the feed-forward's:     d' = norm(d) * w_d;  r' = norm(r) * w_r;
 *                    j = norm(d' + r') * w_j;  x = (x + j) * s;  and optionally y = norm(x) * w_n
 *                    (post_feedforward_layernorm_1 and _2, the arms' sum, post_feedforward_layernorm,
 *                    the residual add, layer_scalar -- and the NEXT layer's input_layernorm, or the
 *                    final norm after the last layer, which read the x this writes)
 *
 * THE SAME FUNCTION AS THE CHAIN, ROUNDING FOR ROUNDING. Every value the chain stores as bf16 is
 * rounded to bf16 here at the same point, and every element is computed with libr4d's own
 * expression: a norm is bf16((v * rsqrtf(mean(v^2) + eps)) * g), an add bf16(u + v), a scale
 * bf16(x * s). The one thing that can differ is the ORDER the sum of squares is added in -- libr4d
 * picks its block width from a measured table -- which moves the f32 norm factor by an ulp at most
 * and, rarely, a bf16 output by one ulp. The arch plugin asks for these first and falls back to
 * the chain where nothing serves them (gemma4.cpp).
 *
 * One 256-thread block a row: a decode step is a handful of rows, so the row is latency-bound and
 * a block's three reductions beat seven launches. Values a later stage reuses stay in registers
 * (kMaxPer a thread, so n <= 256 * kMaxPer; the row table says so).
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "g4k.h"
#include "g4k_launch.h"
#include "rad_plugin.h"

namespace g4kn {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kMaxPer = 24;         /* elements a thread holds: n <= 6144 */

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }
/* round to nearest even, as libr4d's f32_to_bf16 */
__device__ __forceinline__ uint16_t f2bf(float f) {
    const unsigned u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31u); }
__device__ __forceinline__ float shfl_xor(float v, int m) {
    return __int_as_float(__builtin_amdgcn_ds_bpermute((lane_id() ^ m) << 2, __float_as_int(v)));
}
/* the block's sum, every thread the same value: a wave butterfly, then the waves in order */
__device__ __forceinline__ float block_sum(float v, float* lds) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v += shfl_xor(v, d);
    const int lane = lane_id(), wave = (int)(threadIdx.x >> 5);
    __syncthreads();                                 /* lds is free: the last read of it is done */
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) s += lds[w];
    return s;
}
__device__ __forceinline__ float norm_factor(float ss, int n, float eps) {
    return rsqrtf(ss / (float)n + eps);
}

struct NormArgs {
    const uint16_t* a;  long long a_ld;     /* norm_add_norm: the block output */
    const uint16_t* d;  long long d_ld;     /* join_add_scale: the dense partial */
    const uint16_t* r;  long long r_ld;     /*                 the routed partial */
    uint16_t*       x;  long long x_ld;     /* the residual stream, read and written */
    uint16_t*       y;  long long y_ld;     /* norm_add_norm: the next norm's output */
    const float*    w0;                     /* w_a, or w_d */
    const float*    w1;                     /* w_y, or w_r */
    const float*    w2;                     /*        w_j */
    const float*    w3;                     /*        w_n, the next norm's gain (or null) */
    const uint16_t* s;                      /*        the layer scalar, bf16, one element */
    int   n;
    float eps;
};

/* t = norm(a) * w_a;  x += t;  y = norm(x) * w_y */
__global__ __launch_bounds__(kThreads) void norm_add_norm_kernel(NormArgs A) {
    __shared__ float lds[kWaves];
    const int tid = (int)threadIdx.x, n = A.n;
    const long long m = (long long)blockIdx.x;
    const uint16_t* ar = A.a + m * A.a_ld;
    uint16_t* xr = A.x + m * A.x_ld;
    uint16_t* yr = A.y + m * A.y_ld;

    float av[kMaxPer], xv[kMaxPer];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        av[k] = i < n ? bf2f(ar[i]) : 0.0f;
        ss += av[k] * av[k];
    }
    const float sa = norm_factor(block_sum(ss, lds), n, A.eps);
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float t = rbf((av[k] * sa) * A.w0[i]);
            xv[k] = rbf(bf2f(xr[i]) + t);
            xr[i] = f2bf(xv[k]);
        } else {
            xv[k] = 0.0f;
        }
        ss += xv[k] * xv[k];
    }
    const float sx = norm_factor(block_sum(ss, lds), n, A.eps);
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) yr[i] = f2bf((xv[k] * sx) * A.w1[i]);
    }
}

/* x = (x + norm(norm(d) * w_d + norm(r) * w_r) * w_j) * s, and with NEXT y = norm(x) * w_n: the
 * next layer's input norm taken over the bf16 x this stores, which is what that norm would read. */
template <bool NEXT>
__global__ __launch_bounds__(kThreads) void join_add_scale_kernel(NormArgs A) {
    __shared__ float lds[kWaves];
    const int tid = (int)threadIdx.x, n = A.n;
    const long long m = (long long)blockIdx.x;
    const uint16_t* dr = A.d + m * A.d_ld;
    const uint16_t* rr = A.r + m * A.r_ld;
    uint16_t* xr = A.x + m * A.x_ld;

    float dv[kMaxPer], rv[kMaxPer];
    float sd2 = 0.0f, sr2 = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        dv[k] = i < n ? bf2f(dr[i]) : 0.0f;
        rv[k] = i < n ? bf2f(rr[i]) : 0.0f;
        sd2 += dv[k] * dv[k];
        sr2 += rv[k] * rv[k];
    }
    const float sd = norm_factor(block_sum(sd2, lds), n, A.eps);
    const float sr = norm_factor(block_sum(sr2, lds), n, A.eps);
    /* the join, kept in dv */
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float u = rbf((dv[k] * sd) * A.w0[i]);
            const float v = rbf((rv[k] * sr) * A.w1[i]);
            dv[k] = rbf(u + v);
        } else {
            dv[k] = 0.0f;
        }
        ss += dv[k] * dv[k];
    }
    const float sj = norm_factor(block_sum(ss, lds), n, A.eps);
    const float sc = bf2f(*A.s);
    ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPer; ++k) {
        const int i = tid + k * kThreads;
        if (i < n) {
            const float j = rbf((dv[k] * sj) * A.w2[i]);
            const float x1 = rbf(bf2f(xr[i]) + j);
            const uint16_t xn = f2bf(x1 * sc);
            xr[i] = xn;
            rv[k] = bf2f(xn);                        /* the new x, kept for the next norm */
        } else {
            rv[k] = 0.0f;
        }
        if (NEXT) ss += rv[k] * rv[k];
    }
    if (NEXT) {
        const float sn = norm_factor(block_sum(ss, lds), n, A.eps);
        uint16_t* yr = A.y + m * A.y_ld;
#pragma unroll
        for (int k = 0; k < kMaxPer; ++k) {
            const int i = tid + k * kThreads;
            if (i < n) yr[i] = f2bf((rv[k] * sn) * A.w3[i]);
        }
    }
}

/* ------------------------------------------------------------------ host: the launch adapters */

/* [rows..., n] with a dense last axis: its row count and pitch, or false */
static bool rows_of(const RadTensor* t, int64_t n, int64_t* rows, int64_t* ld) {
    if (!t || t->dtype != RAD_BF16 || t->rank < 1) return false;
    if (rad_tensor_lastdim(t) != n || (t->stride[t->rank - 1] != 1)) return false;
    *rows = rad_tensor_rows(t);
    *ld = rad_tensor_row_pitch(t);
    return *ld >= n || *rows <= 1;
}
static bool gain(const RadTensor* w, int64_t n) {
    return w && w->dtype == RAD_F32 && rad_tensor_numel(w) == n &&
           (w->rank == 0 || w->stride[w->rank - 1] == 1);
}

static int params(const RadArgs* a, int64_t* n, float* eps) {
    *n = rad_args_geti_or(a, "n", 0);
    *eps = (float)rad_args_getf_or(a, "eps", 1e-6);
    if (*n <= 0 || *n > (int64_t)kThreads * kMaxPer) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace g4kn */

extern "C" {

/* operands: a, w_a, x (inout), w_y, y */
int g4k_norm_add_norm_bf16(const RadArgs* a, RadStream stream) {
    using namespace g4kn;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    int64_t n = 0;
    float eps = 0.0f;
    if (int rc = params(a, &n, &eps)) return rc;
    const RadTensor *A_ = rad_arg_in(a, 0), *Wa = rad_arg_in(a, 1), *X = rad_arg_in(a, 2),
                    *Wy = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    if (!A_ || !X || !Y) return RAD_E_INVAL;
    if (!gain(Wa, n) || !gain(Wy, n)) return RAD_E_DTYPE;
    int64_t ra = 0, la = 0, rx = 0, lx = 0, ry = 0, ly = 0;
    if (!rows_of(A_, n, &ra, &la) || !rows_of(X, n, &rx, &lx) || !rows_of(Y, n, &ry, &ly))
        return RAD_E_SHAPE;
    const int64_t M = rx;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (ra < M || ry < M || M > 0x7fffffff) return RAD_E_SHAPE;
    NormArgs P{};
    P.a = (const uint16_t*)A_->data; P.a_ld = la;
    P.x = (uint16_t*)X->data;        P.x_ld = lx;
    P.y = (uint16_t*)Y->data;        P.y_ld = ly;
    P.w0 = (const float*)Wa->data;   P.w1 = (const float*)Wy->data;
    P.n = (int)n; P.eps = eps;
    G4K_LAUNCH(G4K_KNAME(norm_add_norm_kernel), dim3((unsigned)M), dim3(kThreads), (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* operands: d, w_d, r, w_r, w_j, x (inout), s, w_n?, y? -- the last two both or neither */
int g4k_join_add_scale_bf16(const RadArgs* a, RadStream stream) {
    using namespace g4kn;
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    int64_t n = 0;
    float eps = 0.0f;
    if (int rc = params(a, &n, &eps)) return rc;
    const RadTensor *D = rad_arg_in(a, 0), *Wd = rad_arg_in(a, 1), *R = rad_arg_in(a, 2),
                    *Wr = rad_arg_in(a, 3), *Wj = rad_arg_in(a, 4), *X = rad_arg_in(a, 5),
                    *Sc = rad_arg_in(a, 6);
    if (!D || !R || !X || !Sc) return RAD_E_INVAL;
    if (!gain(Wd, n) || !gain(Wr, n) || !gain(Wj, n)) return RAD_E_DTYPE;
    if (Sc->dtype != RAD_BF16 || rad_tensor_numel(Sc) != 1) return RAD_E_DTYPE;
    const RadTensor *Wn = a->n_t > 7 ? rad_arg_in(a, 7) : nullptr,
                    *Y  = a->n_t > 8 ? rad_arg_in(a, 8) : nullptr;
    if (!Wn != !Y) return RAD_E_INVAL;
    if (Wn && !gain(Wn, n)) return RAD_E_DTYPE;
    int64_t rd = 0, ld = 0, rr = 0, lr = 0, rx = 0, lx = 0, ry = 0, ly = 0;
    if (!rows_of(D, n, &rd, &ld) || !rows_of(R, n, &rr, &lr) || !rows_of(X, n, &rx, &lx))
        return RAD_E_SHAPE;
    if (Y && !rows_of(Y, n, &ry, &ly)) return RAD_E_SHAPE;
    const int64_t M = rx;                            /* the step's rows: the stream's view */
    if (M <= 0) return RAD_OK;
    if (rd < M || rr < M || (Y && ry < M) || M > 0x7fffffff) return RAD_E_SHAPE;
    NormArgs P{};
    P.d = (const uint16_t*)D->data;  P.d_ld = ld;
    P.r = (const uint16_t*)R->data;  P.r_ld = lr;
    P.x = (uint16_t*)X->data;        P.x_ld = lx;
    P.w0 = (const float*)Wd->data;   P.w1 = (const float*)Wr->data;   P.w2 = (const float*)Wj->data;
    P.s = (const uint16_t*)Sc->data;
    if (Y) { P.w3 = (const float*)Wn->data; P.y = (uint16_t*)Y->data; P.y_ld = ly; }
    P.n = (int)n; P.eps = eps;
    if (Y)
        G4K_LAUNCH(G4K_KNAME(join_add_scale_kernel<true>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    else
        G4K_LAUNCH(G4K_KNAME(join_add_scale_kernel<false>), dim3((unsigned)M), dim3(kThreads),
                   (hipStream_t)stream, P);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
