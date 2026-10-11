// r4d_rmsnorm_quant_i8.hip - r4d_rmsnorm_quant_i8: the RMS norm and the per-row int8 quantiser
// r4d_gemm_w4a8_nt_m64 reads, in one pass over the row -- optionally with the residual junction
// in front of it: a post-norm on the branch output, the residual add and a scalar on the sum.
//
// WHY.  A w4a8 linear needs its activation as int8 codes in the GEMM's fragment order with one f32
// scale per row (r4d_quant_act_i8), and the producer of that activation is a norm.  Unfused, a
// model whose linears take that grid issues the norm and the quantiser as two dispatches, and a
// sandwich-norm model (Gemma 2/3/4) issues four or five at every residual junction:
//
//     rmsnorm(branch; w_post) -> add(residual) -> [scale] -> rmsnorm(residual; w) -> quant_act_i8
//
// each one a single workgroup over one row at a decode batch, so the step pays the dispatch, not
// the bytes.  This is that chain as one dispatch.
//
// BYTE-IDENTICAL TO THE CHAIN IT REPLACES, not merely close, because the engine's greedy output is
// compared byte for byte across builds.  What makes that hold:
//
//   * every intermediate the chain stores in bf16 is rounded to bf16 here too, and the next stage
//     reads the rounded value: the post-norm output, the residual sum, the scaled residual and the
//     norm output the quantiser takes its absmax and codes over;
//   * both sums of squares are taken by the same partition the standalone norm uses -- the width
//     r4d_act_threads(M, n) picks, thread t folding columns t, t+THR, ... in ascending order, then
//     r4d_wave_sum and the waves in order through LDS (r4d_model_bf16.hip's rmsnorm_kernel and
//     block_sum, which this copies rather than shares only because that one lives in another unit);
//   * the arithmetic of every element is written the way the standalone kernels write it --
//     `x * s * (g + wadd)`, `a + b`, `x * scalar`, `fmaxf(m, 1e-8f) / 127.f` -- and the library is
//     built with -ffp-contract=off, so no stage picks up an fma the chain does not have.
//
// THE RESIDUAL IS READ AND WRITTEN, and nothing else reads it before this kernel is done: each
// thread rewrites only the columns it read, so the second pass reads its own writes back.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <cstdint>

namespace {

constexpr uint32_t kWave = 32;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// The standalone norm's reduction, verbatim (r4d_model_bf16.hip): the wave butterfly, then the
// waves summed IN ORDER by every thread out of LDS.  `lds` must not be reused by a second
// reduction without a barrier between them -- the kernel below gives each its own array.
template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

__device__ __forceinline__ float gain(const void* w, uint32_t i, bool f32) {
    return f32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
}

// RES:  a residual is present; the norm is taken of `res + d` (stored back into `res`), where d is
//       the branch `x` itself or, with POST, rmsnorm(x; w_post).  Without RES the norm is of `x`.
// SCL:  the sum is multiplied by the [1] scalar before it is stored and normalised.
// `ybuf` holds the row's norm output for the fragment-order store, which reads across threads.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR>
__global__ __launch_bounds__(THR) void rmsnorm_quant_i8_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    uint32_t n, float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld) {
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
    signed char* __restrict__ qr = q + (size_t)row * n;

    // ---- the post-norm's sum of squares, over the branch output
    float s1 = 0.0f;
    if (POST) {
        float ss = 0.0f;
        for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = bf2f(xr[i]); ss += v * v; }
        ss = block_sum<THR / kWave>(ss, l1);
        s1 = rsqrtf(ss / (float)n + eps);
    }

    // ---- the junction (post-norm, add, scalar), stored, and the norm's sum of squares over it
    float sv = 1.0f;
    if (SCL) sv = sc_f32 ? ((const float*)sc)[0] : bf2f(((const uint16_t*)sc)[0]);
    float ss2 = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        uint16_t v = xr[i];
        if (RES) {
            if (POST) v = f32_to_bf16(bf2f(v) * s1 * (gain(wp, i, WF32) + wadd));
            v = f32_to_bf16(bf2f(rr[i]) + bf2f(v));
            if (SCL) v = f32_to_bf16(bf2f(v) * sv);
            rr[i] = v;
        }
        const float f = bf2f(v);
        ss2 += f * f;
    }
    ss2 = block_sum<THR / kWave>(ss2, l2);
    const float s2 = rsqrtf(ss2 / (float)n + eps);

    // ---- the norm output, rounded to bf16 as the standalone norm stores it, and its absmax
    const uint16_t* __restrict__ nr = RES ? rr : xr;
    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint16_t y = f32_to_bf16(bf2f(nr[i]) * s2 * (gain(w, i, WF32) + wadd));
        ybuf[i] = y;
        if (yo) yo[(size_t)row * y_ld + i] = y;
        m = fmaxf(m, fabsf(bf2f(y)));
    }
    m = r4d_wave_max(m);
    if ((threadIdx.x & 31) == 0) lm[threadIdx.x >> 5] = m;
    __syncthreads();   // also publishes ybuf to every thread
    m = lm[0];
#pragma unroll
    for (uint32_t i = 1; i < THR / kWave; i++) m = fmaxf(m, lm[i]);
    const float scale = fmaxf(m, 1e-8f) / 127.f;
    const float inv = 1.f / scale;
    if (threadIdx.x == 0) S[row] = scale;

    // ---- the codes, in r4d_quant_act_i8's fragment order: byte j of a k step comes from
    // lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) + 4*(j>>3) + (j&3) inside the step.
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint32_t s = i >> 4, j = i & 15;
        const uint32_t src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = __float2int_rn(bf2f(ybuf[src]) * inv);
        qr[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}

}  // namespace

// ============================================================== the plugin ABI entry point
//
// Operands, in the schema's order (r4d_rows.cpp, oRmsNormQuantI8):
//   0 x         [M][n] bf16           the norm's input, or the branch output with a residual
//   1 residual  [M][n] bf16, optional READ AND WRITTEN
//   2 w         [n]    bf16 or f32    the norm's gain
//   3 q         [M][n] int8           fragment order, unpitched
//   4 scale     [M][1] f32            one per row
//   5 out_bf16  [M][n] bf16, optional the norm output the unfused norm would have stored
//   6 w_post    [n]    optional       a post-norm on x before the add; needs the residual
//   7 scalar    [1]    optional       multiplies the sum; needs the residual
enum { RQ_X = 0, RQ_RES, RQ_W, RQ_Q, RQ_S, RQ_Y, RQ_WP, RQ_SC };

static int rad_rmsnorm_quant_i8_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* x  = r4d_opt(a, RQ_X);
    const RadTensor* rs = r4d_opt(a, RQ_RES);
    const RadTensor* w  = r4d_opt(a, RQ_W);
    const RadTensor* q  = r4d_opt(a, RQ_Q);
    const RadTensor* s  = r4d_opt(a, RQ_S);
    const RadTensor* y  = r4d_opt(a, RQ_Y);
    const RadTensor* wp = r4d_opt(a, RQ_WP);
    const RadTensor* sc = r4d_opt(a, RQ_SC);
    if (!x || !w || !q || !s) return R4D_NO(RAD_E_INVAL);
    if ((wp || sc) && !rs) return R4D_NO(RAD_E_INVAL);   // a junction without a stream to add into
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || s->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    long long M = 0, n = 0;
    if (r4d_rows_of(x, &M, &n) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (n % 16) return R4D_NO(RAD_E_SHAPE);                  // a k step is sixteen bytes of one row
    if (r4d_numel(w) != n) return R4D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wp && (r4d_numel(wp) != n || wp->dtype != w->dtype)) return R4D_NO(RAD_E_SHAPE);
    if (sc && (r4d_numel(sc) != 1 || (sc->dtype != RAD_F32 && sc->dtype != RAD_BF16)))
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) != M * n || !r4d_contig(q)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(s) != M || !r4d_contig(s)) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) != M * n) return R4D_NO(RAD_E_SHAPE);
    if (rs && r4d_numel(rs) != M * n) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), rp = rs ? r4d_row_pitch(rs) : n, yp = y ? r4d_row_pitch(y) : n;
    if (xp < 0 || rp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    // The row's norm output is staged in LDS for the cross-thread fragment-order store.
    const size_t lds = (size_t)n * sizeof(uint16_t);
    if (lds > 60 * 1024) return R4D_NO(RAD_E_SHAPE);

    // THE SAME WIDTH AS rmsnorm_bf16 AND quant_act_i8 PICK, from the same function: it decides
    // which columns a thread folds into each sum of squares, which is the byte-identity.
    const int thr = r4d_act_threads((int)M, (int)n);
    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const uint32_t scf = sc ? (sc->dtype == RAD_F32) : 0u;

#define R4D_RQI8_T(WF, RS, PO, SC, T) emit_rc = R4D_EMIT(sk,                                     \
    (rmsnorm_quant_i8_kernel<WF, RS, PO, SC, T>), dim3((uint32_t)M), dim3(T), lds,               \
    (const uint16_t*)r4d_p(x), (uint16_t*)r4d_wp(rs), r4d_p(w), r4d_p(wp), r4d_p(sc), scf,       \
    (signed char*)r4d_wp(q), (float*)r4d_wp(s), (uint16_t*)r4d_wp(y), (uint32_t)n, eps, wadd,    \
    (uint32_t)xp, (uint32_t)rp, (uint32_t)yp)
#define R4D_RQI8(WF, RS, PO, SC)                                                                  \
    do { if (thr == 1024)     R4D_RQI8_T(WF, RS, PO, SC, 1024);                                   \
         else if (thr == 512) R4D_RQI8_T(WF, RS, PO, SC, 512);                                    \
         else if (thr == 256) R4D_RQI8_T(WF, RS, PO, SC, 256);                                    \
         else return R4D_NO(RAD_E_INVAL); } while (0)
#define R4D_RQI8_W(WF)                                                                            \
    do { if (!rs)            R4D_RQI8(WF, false, false, false);                                   \
         else if (!wp && !sc) R4D_RQI8(WF, true, false, false);                                   \
         else if (wp && !sc)  R4D_RQI8(WF, true, true, false);                                    \
         else if (wp && sc)   R4D_RQI8(WF, true, true, true);                                     \
         else                 R4D_RQI8(WF, true, false, true); } while (0)
    if (w->dtype == RAD_F32) R4D_RQI8_W(true);
    else                     R4D_RQI8_W(false);
#undef R4D_RQI8_W
#undef R4D_RQI8
#undef R4D_RQI8_T
    return emit_rc;
}
R4D_ENTRY(r4d_rad_rmsnorm_quant_i8, rad_rmsnorm_quant_i8_body)
