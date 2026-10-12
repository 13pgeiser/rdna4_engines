// r4d_qkv_norm_rope.hip - r4d_qkv_norm_rope: the per-head norms and the rotary of separate q, k and
// v projections in one dispatch -- Gemma 4's attention prologue.
//
// WHY.  Between its projections and the cache store a Gemma 4 attention block runs five single-pass
// kernels over a few dozen heads: the weightless value norm (rmsnorm_p), the q and the k norms
// (rmsnorm, one row a head) and the two rotaries (rope).  At a decode batch each is a handful of
// workgroups, so the block pays five dispatches for work the card does in a fraction of one.  This is
// the five as one, one workgroup a (token, head).
//
// BYTE-IDENTICAL TO THE FIVE, for the reason r4d_rmsnorm_quant_i8.hip gives.  What makes that hold:
//
//   * each norm's sum of squares is taken by THE PARTITION ITS STANDALONE KERNEL USES.  rmsnorm runs
//     one workgroup a row at r4d_act_threads(rows, head_dim) threads and rmsnorm_p at 256; thread t
//     folds columns t, t+THR, ... in order, the waves are summed in order through LDS.  Here a
//     workgroup is head_dim threads, and thread t < min(THR, head_dim) folds the same columns in the
//     same order while the rest hold zero -- which is what the standalone kernel's surplus waves hold
//     too, and adding +0.0 to a sum of squares changes no bit;
//   * the norm output is rounded to bf16 as the standalone norm stores it, and the rotary reads that;
//   * the rotary's per-pair arithmetic is rope_kernel's, written the same way (powf ladder,
//     sincosf, `x0*c - x1*s`, `x0*s + x1*c`), under the library's -ffp-contract=off.
//
// `v_from_k` IS GEMMA'S GLOBAL CLASS: k == v, so v is the value norm of the RAW k, taken before the
// k norm rewrites it -- the order the unfused block issues them in.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace {

constexpr uint32_t kWave = 32;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// THR threads' partition of one head's sum of squares, reproduced inside a head_dim-wide workgroup:
// thread t < P = min(THR, HD) folds t, t+P, ...; the waves are then summed in order.
template <uint32_t HD>
__device__ __forceinline__ float head_ss(const uint16_t* __restrict__ x, uint32_t P, float* lds) {
    float ss = 0.0f;
    if (threadIdx.x < P)
        for (uint32_t i = threadIdx.x; i < HD; i += P) { const float v = bf2f(x[i]); ss += v * v; }
    ss = r4d_wave_sum(ss);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = ss;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < HD / kWave; ++w) s += lds[w];
    return s;
}

// The rotary over a normed head staged in `nb`, written to `out`: rope_kernel's body for pair i.
template <uint32_t HD>
__device__ __forceinline__ void rotate(const uint16_t* __restrict__ nb, uint16_t* __restrict__ out,
                                       float p0, uint32_t rot, uint32_t nfreq, int den, float theta) {
    const uint32_t half = rot / 2, t = threadIdx.x;
    if (t < half) {
        float inv = 1.0f / powf(theta, (float)(2 * t) / (float)den);
        if (t >= nfreq) inv = 0.0f;
        const float ang = p0 * inv;
        float c, s;
        sincosf(ang, &s, &c);
        const float x0 = bf2f(nb[t]), x1 = bf2f(nb[t + half]);
        out[t]        = f32_to_bf16(x0 * c - x1 * s);
        out[t + half] = f32_to_bf16(x0 * s + x1 * c);
    }
    for (uint32_t i = rot + t; i < HD; i += HD) out[i] = nb[i];   // the unrotated tail
}

// blockIdx.y: [0, nq) a query head, [nq, nq+nkv) a key head (and with VK its value, from the raw
// key), [nq+nkv, nq+2nkv) a value head normalised in place (without VK).
template <uint32_t HD, bool WF32, bool VK>
__global__ __launch_bounds__(HD) void qkv_norm_rope_kernel(
    uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v,
    const int32_t* __restrict__ pos, const void* __restrict__ qw, const void* __restrict__ kw,
    uint32_t nq, uint32_t nkv, uint32_t pq, uint32_t pk, uint32_t pv,
    uint32_t q_ld, uint32_t k_ld, uint32_t v_ld, float eps, float wadd, float mult,
    uint32_t rot, uint32_t nfreq, int den, float theta, float rscale, uint32_t n_pos) {
    __shared__ float l1[HD / kWave], l2[HD / kWave];
    __shared__ uint16_t nb[HD];
    const uint32_t m = blockIdx.x, hh = blockIdx.y, t = threadIdx.x;
    const float div = rscale != 0.0f ? rscale : 1.0f;
    // rope rotates the rows its positions cover; a row past them is normed and left unrotated
    const float p0 = m < n_pos ? (float)pos[m] / div : 0.0f;

    if (hh >= nq + nkv) {   // a value head, in place: rmsnorm_p
        uint16_t* __restrict__ x = v + (size_t)m * v_ld + (size_t)(hh - nq - nkv) * HD;
        const float s = rsqrtf(head_ss<HD>(x, pv, l1) / (float)HD + eps);
        __syncthreads();   // every thread has read x before any writes it
        for (uint32_t i = t; i < HD; i += HD) x[i] = f32_to_bf16(bf2f(x[i]) * s);
        return;
    }
    const bool isq = hh < nq;
    const uint32_t h = isq ? hh : hh - nq;
    uint16_t* __restrict__ x = isq ? q + (size_t)m * q_ld + (size_t)h * HD
                                   : k + (size_t)m * k_ld + (size_t)h * HD;
    const void* __restrict__ w = isq ? qw : kw;

    if (VK && !isq) {   // v = rmsnorm_p(raw k), before the k norm rewrites k
        uint16_t* __restrict__ vo = v + (size_t)m * v_ld + (size_t)h * HD;
        const float sv = rsqrtf(head_ss<HD>(x, pv, l2) / (float)HD + eps);
        for (uint32_t i = t; i < HD; i += HD) vo[i] = f32_to_bf16(bf2f(x[i]) * sv);
    }
    const float s = rsqrtf(head_ss<HD>(x, isq ? pq : pk, l1) / (float)HD + eps);
    for (uint32_t i = t; i < HD; i += HD) {
        const float g = WF32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
        nb[i] = f32_to_bf16(bf2f(x[i]) * s * (g + wadd) * mult);
    }
    __syncthreads();   // the normed head is complete before any pair reads across it
    if (m < n_pos) rotate<HD>(nb, x, p0, rot, nfreq, den, theta);
    else for (uint32_t i = t; i < HD; i += HD) x[i] = nb[i];
}

}  // namespace

// ============================================================== the plugin ABI entry point
enum { QN_Q = 0, QN_K, QN_V, QN_POS, QN_QW, QN_KW };

static int rad_qkv_norm_rope_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* q  = r4d_opt(a, QN_Q);
    const RadTensor* k  = r4d_opt(a, QN_K);
    const RadTensor* v  = r4d_opt(a, QN_V);
    const RadTensor* ps = r4d_opt(a, QN_POS);
    const RadTensor* qw = r4d_opt(a, QN_QW);
    const RadTensor* kw = r4d_opt(a, QN_KW);
    if (!q || !k || !v || !ps || !qw || !kw) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_BF16 || k->dtype != RAD_BF16 || v->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ps->dtype != RAD_I32 || ps->rank != 1 || !r4d_contig(ps)) return R4D_NO(RAD_E_SHAPE);
    if (qw->dtype != kw->dtype || (qw->dtype != RAD_BF16 && qw->dtype != RAD_F32)) return R4D_NO(RAD_E_DTYPE);
    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nq  = r4d_geti_or(a, "n_head", 0);
    const long long nkv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot = r4d_geti_or(a, "rotary_dim", hd);
    const long long vk  = r4d_geti_or(a, "v_from_k", 0);
    if (hd != 256 && hd != 512) return R4D_NO(RAD_E_SHAPE);
    if (nq <= 0 || nkv <= 0 || rot <= 0 || rot > hd || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(qw) != hd || r4d_numel(kw) != hd) return R4D_NO(RAD_E_SHAPE);
    const char* mode = r4d_gets(a, "mode");
    if (mode && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_getf(a, "yarn_factor", 0.0) != 0.0) return R4D_NO(RAD_E_UNSUPPORTED);
    // rope's ladder, resolved exactly as r4d_rad_rope resolves it
    const char* lad = r4d_gets(a, "ladder");
    long long den = rot;
    if (lad) {
        if (!std::strcmp(lad, "head")) den = hd;
        else if (std::strcmp(lad, "rotary")) return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const long long nfreq = rot / 2;
    const long long rot_l = (den == hd && rot < hd) ? hd : rot;

    long long M = 0, Mk = 0, Mv = 0;
    const long long q_ld = r4d_rows_pitch(q, nq * hd, &M);
    const long long k_ld = r4d_rows_pitch(k, nkv * hd, &Mk);
    const long long v_ld = r4d_rows_pitch(v, nkv * hd, &Mv);
    if (q_ld < 0 || k_ld < 0 || v_ld < 0) return R4D_NO(RAD_E_STRIDE);
    const long long Mp = r4d_numel(ps);
    if (Mp <= 0 || M < Mp || Mk != M || Mv != M) return R4D_NO(RAD_E_SHAPE);

    // the standalone kernels' widths, from the same function: rmsnorm one row a head at
    // r4d_act_threads(rows, head_dim), rmsnorm_p at 256
    const uint32_t pq = (uint32_t)std::min<long long>(r4d_act_threads((int)(M * nq), (int)hd), hd);
    const uint32_t pk = (uint32_t)std::min<long long>(r4d_act_threads((int)(M * nkv), (int)hd), hd);
    const uint32_t pv = (uint32_t)std::min<long long>(256, hd);
    const float eps   = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd  = (float)r4d_getf(a, "wadd", 0.0);
    const float theta = (float)r4d_getf(a, "theta", 10000.0);
    const float rsc   = (float)r4d_getf(a, "scale", 1.0);
    const float mult  = 1.0f;
    const dim3 grid((uint32_t)M, (uint32_t)(nq + nkv + (vk ? 0 : nkv)));

#define R4D_QNR_T(HD, WF, VK) emit_rc = R4D_EMIT(sk, (qkv_norm_rope_kernel<HD, WF, VK>), grid, dim3(HD), 0, \
    (uint16_t*)r4d_wp(q), (uint16_t*)r4d_wp(k), (uint16_t*)r4d_wp(v), (const int32_t*)r4d_p(ps),           \
    r4d_p(qw), r4d_p(kw), (uint32_t)nq, (uint32_t)nkv, pq, pk, pv, (uint32_t)q_ld, (uint32_t)k_ld,          \
    (uint32_t)v_ld, eps, wadd, mult, (uint32_t)rot_l, (uint32_t)nfreq, (int)den, theta, rsc, (uint32_t)Mp)
#define R4D_QNR(HD)                                                                                     \
    do { if (qw->dtype == RAD_F32) { if (vk) R4D_QNR_T(HD, true, true);  else R4D_QNR_T(HD, true, false); } \
         else                      { if (vk) R4D_QNR_T(HD, false, true); else R4D_QNR_T(HD, false, false); } } while (0)
    if (hd == 256) R4D_QNR(256);
    else           R4D_QNR(512);
#undef R4D_QNR
#undef R4D_QNR_T
    return emit_rc;
}
R4D_ENTRY(r4d_rad_qkv_norm_rope, rad_qkv_norm_rope_body)
