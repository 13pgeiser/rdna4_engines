/* r3d_gdn_conv_w4_h128.hip -- the gated-delta-net depthwise causal convolution and the work fused
 * around it, one entry point per serving phase.
 *
 * PREFILL (gdn_conv_prep): everything the layer does between the qkv projection and the chunked
 * scan, in one kernel:
 *
 *     conv(x)        depthwise causal convolution, width 4, silu, varlen, with the conv-state
 *                    cache read at a sequence's first chunk and rewritten from its last tokens
 *     split          the packed [q | k | v] channel range into three contiguous tensors
 *     l2norm         over each 128-dim q and k head
 *     gate           g = -exp(A_log) . softplus(a + dt_bias), beta = sigmoid(b)
 *     cumsum         g summed along the chunk, which is what the scan actually consumes
 *
 * The conv output is rounded to bf16 before the norm and the store, so this kernel is a bit-exact
 * drop-in for the unfused causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum pair.
 *
 * DECODE (gdn_conv_update): the same filter, but a step's tokens are a speculative window and the
 * state cache is a rolling buffer of width-1 + num_spec entries: a step reads its window starting
 * at the slot the last accepted token left, and rewrites the buffer shifted by one.
 *
 * A workgroup is HPB waves, one wave per 128-dim head, four dims per lane, so a head's norm is a
 * barrier-free wave reduction and each lane does one 8-byte access. Tokens are blocked by the
 * 64-token chunk because the cumsum is per chunk.
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
#include <cstdint>

namespace {

constexpr int kHD   = 128;   // dims per head, q k and v alike
constexpr int kBT   = 64;    // token block = the scan's chunk, so the cumsum stays wave-local
constexpr int kW    = 4;     // conv width
constexpr int kST   = kW - 1;  // conv state depth (3 taps of history)
constexpr int kMAXT = 8;     // tokens the decode update preloads into registers
constexpr int kDPL  = 4;     // dims per lane (kHD / 32)

/* One convolution tap: rounded to bf16 by the caller. The reciprocal form is two instructions
 * against ~10 for an IEEE-exact divide, and v_rcp_f32 is 1 ULP against bf16's 2^-8 quantum. */
__device__ __forceinline__ float cp_silu(float z)
{
    return z * __builtin_amdgcn_rcpf(1.0f + __expf(-z));
}

/* softplus, in the two-sided form with the same large-x cutoff. */
__device__ __forceinline__ float cp_softplus(float x, float thr)
{
    if (x > thr) return x;
    return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

/* a and b are read where they lie: a chunk() of the ba projection, bf16 with row stride 2*H (or
 * fp32), avoiding a copy and two launches per layer. */
template <int ABF16>
__device__ __forceinline__ float cp_ab(const void* p, size_t i)
{
    return ABF16 ? bf16_to_f32(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

/* `has_init` says whether the slot holds a history; its element width (int32 or byte) is given
 * by the caller. Read as bytes, a ctx_len of 512 would look like 0 and the conv window would
 * silently start from zeros. */
__device__ __forceinline__ bool r3d_has_init(const void* p, int i32, int n)
{
    if (!p) return false;
    return i32 ? ((const int*)p)[n] != 0 : ((const unsigned char*)p)[n] != 0;
}

/* ================================ the prefill convolution ================================ */

template <int HPB, int ABF16>
__global__ __launch_bounds__(HPB * 32)
void r3d_gdn_conv_prep_kernel(const unsigned short* __restrict__ x, long long xpitch,
                              const unsigned short* __restrict__ wgt,
                              const unsigned short* __restrict__ bias,
                              unsigned short* __restrict__ cstate,
                              long long cs_seq, long long cs_dim, long long cs_tok,
                              const int* __restrict__ cache_idx, long long ci_stride,
                              const void* __restrict__ has_init, int hi_i32,
                              const void* __restrict__ av, const void* __restrict__ bv,
                              long long ab_stride,
                              const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                              unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                              unsigned short* __restrict__ v,
                              float* __restrict__ go, float* __restrict__ beta,
                              const int* __restrict__ cu, int N, int H, int Hg, float sp_thr)
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // flat chunk id -> (sequence, chunk); the grid is an upper bound.
    int n = -1, t0 = 0, bos = 0, rows = 0, slen = 0;
    {
        int acc = 0, want = blockIdx.x;
        for (int i = 0; i < N; ++i) {
            const int b = cu[i], e = cu[i + 1];
            const int nc = (e - b + kBT - 1) / kBT;
            if (want < acc + nc) { n = i; bos = b; slen = e - b; t0 = (want - acc) * kBT;
                                   rows = min(kBT, slen - t0); break; }
            acc += nc;
        }
        if (n < 0) return;
    }
    const bool first = (t0 == 0);

    // ---- the gating half, on one workgroup per chunk --------------------------------------
    if (blockIdx.y == 0) {
        for (int hv = warp; hv < H; hv += HPB) {
            const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];
            const int i0 = 2 * lane, i1 = i0 + 1;
            float g0 = 0.0f, g1 = 0.0f;
            if (i0 < rows)
                g0 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i0) * ab_stride + hv)
                                         + dtb, sp_thr);
            if (i1 < rows)
                g1 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i1) * ab_stride + hv)
                                         + dtb, sp_thr);
            // inclusive wave scan of the per-lane pair sums, made exclusive by subtracting the pair
            float p = g0 + g1;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const float o = __shfl_up(p, d, 32);
                if (lane >= d) p += o;
            }
            const float pre = p - (g0 + g1);
            if (i0 < rows) {
                go[(size_t)(bos + t0 + i0) * H + hv] = pre + g0;
                beta[(size_t)(bos + t0 + i0) * H + hv] = 1.0f / (1.0f + __expf(
                    -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i0) * ab_stride + hv)));
            }
            if (i1 < rows) {
                go[(size_t)(bos + t0 + i1) * H + hv] = pre + g0 + g1;
                beta[(size_t)(bos + t0 + i1) * H + hv] = 1.0f / (1.0f + __expf(
                    -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i1) * ab_stride + hv)));
            }
        }
    }

    // ---- the convolution half --------------------------------------------------------------
    const int unit = blockIdx.y * HPB + warp;
    const int d0 = unit * kHD + lane * kDPL;
    const unsigned short* xp = x + (size_t)(bos + t0) * xpitch + d0;

    float wt[kW][kDPL], bs[kDPL];
#pragma unroll
    for (int c = 0; c < kDPL; ++c) {
#pragma unroll
        for (int i = 0; i < kW; ++i) wt[i][c] = bf16_to_f32(wgt[(size_t)(d0 + c) * kW + i]);
        bs[c] = bias ? bf16_to_f32(bias[d0 + c]) : 0.0f;
    }

    // win[i] holds x[t - (W-1) + i]; seeded from the previous tokens, or from the conv state cache
    // at the sequence's first chunk.
    float win[kW][kDPL];
    {
        const bool init = first && r3d_has_init(has_init, hi_i32, n);
        const unsigned short* sp = init ? (cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                                           + (size_t)d0 * cs_dim) : nullptr;
#pragma unroll
        for (int i = 0; i < kST; ++i) {              // i = 0,1,2  ->  x[t0-3+i]
            const int src = t0 - kST + i;
#pragma unroll
            for (int c = 0; c < kDPL; ++c) {
                float val = 0.0f;
                if (src >= 0) val = bf16_to_f32(x[(size_t)(bos + src) * xpitch + d0 + c]);
                else if (init) val = bf16_to_f32(sp[(size_t)c * cs_dim + (size_t)(kST + src) * cs_tok]);
                win[i][c] = val;
            }
        }
    }

    // ---- write the new conv state, from x, at the sequence's first chunk --------------------
    if (first && cache_idx) {
        unsigned short* sp = cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                           + (size_t)d0 * cs_dim;
        const bool init = r3d_has_init(has_init, hi_i32, n);
        // Channel-major: the cache is [dim][slot] with slot contiguous, so a lane's kDPL*kST
        // shorts are one contiguous run.
#pragma unroll
        for (int c = 0; c < kDPL; ++c) {
#pragma unroll
            for (int i = 0; i < kST; ++i) {
                const int src = slen - kST + i;
                unsigned short val = 0;
                if (src >= 0) val = x[(size_t)(bos + src) * xpitch + d0 + c];
                else if (init) val = sp[(size_t)c * cs_dim + (size_t)(kST + src) * cs_tok];
                sp[(size_t)c * cs_dim + (size_t)i * cs_tok] = val;
            }
        }
    }

    const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
    const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

    for (int t = 0; t < rows; ++t) {
        float o[kDPL];
        {
            const unsigned short* xr = xp + (size_t)t * xpitch;
            const ushort4 raw = *(const ushort4*)xr;
            win[kW - 1][0] = bf16_to_f32(raw.x); win[kW - 1][1] = bf16_to_f32(raw.y);
            win[kW - 1][2] = bf16_to_f32(raw.z); win[kW - 1][3] = bf16_to_f32(raw.w);
        }
#pragma unroll
        for (int c = 0; c < kDPL; ++c) {
            float acc = bs[c];
#pragma unroll
            for (int i = 0; i < kW; ++i) acc += wt[i][c] * win[i][c];
            o[c] = bf16_to_f32(f32_to_bf16(cp_silu(acc)));
        }

        if (region < 2) {
            float ss = 0.0f;
#pragma unroll
            for (int c = 0; c < kDPL; ++c) ss += o[c] * o[c];
            ss += r3d_xor_lane<16>(ss); ss += r3d_xor_lane<8>(ss); ss += r3d_xor_lane<4>(ss);
            ss += r3d_xor_lane<2>(ss);  ss += r3d_xor_lane<1>(ss);
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            unsigned short* dst = (region ? k : q)
                                + ((size_t)(bos + t0 + t) * Hg + (region ? hk : hq)) * kHD
                                + lane * kDPL;
            *(ushort4*)dst = make_ushort4(f32_to_bf16(o[0] * inv), f32_to_bf16(o[1] * inv),
                                          f32_to_bf16(o[2] * inv), f32_to_bf16(o[3] * inv));
        } else {
            unsigned short* dst = v + ((size_t)(bos + t0 + t) * H + hv) * kHD + lane * kDPL;
            *(ushort4*)dst = make_ushort4(f32_to_bf16(o[0]), f32_to_bf16(o[1]),
                                          f32_to_bf16(o[2]), f32_to_bf16(o[3]));
        }

#pragma unroll
        for (int i = 0; i < kW - 1; ++i)
#pragma unroll
            for (int c = 0; c < kDPL; ++c) win[i][c] = win[i + 1][c];
    }
}

int r3d_gdn_conv_prep_launch(const void* x, long long xpitch, const void* wgt, const void* bias,
                             void* cstate, long long cs_seq, long long cs_dim, long long cs_tok,
                             const void* cache_idx, long long ci_stride, const void* has_init,
                             int hi_i32, const void* a, const void* b, long long ab_stride,
                             int ab_is_bf16, const void* A_log, const void* dt_bias, void* q,
                             void* k, void* v, void* g, void* beta, const void* cu,
                             int N, int T, int H, int Hg, int width, float softplus_thr,
                             hipStream_t stream)
{
    const int units = 2 * Hg + H;
    int hpb = 8;
    while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
    const int nchunk_ub = (T + kBT - 1) / kBT + N;
    dim3 grid(nchunk_ub, units / hpb), blk(hpb * 32);
#define CP_LAUNCH(P, AB) \
    R3D_KLAUNCH(R3D_KNAME((r3d_gdn_conv_prep_kernel<P, AB>)), grid, blk, stream, \
                (const unsigned short*)x, xpitch, (const unsigned short*)wgt, \
                (const unsigned short*)bias, (unsigned short*)cstate, \
                cs_seq, cs_dim, cs_tok, (const int*)cache_idx, ci_stride, \
                has_init, hi_i32, a, b, ab_stride, \
                (const float*)A_log, (const float*)dt_bias, (unsigned short*)q, \
                (unsigned short*)k, (unsigned short*)v, (float*)g, (float*)beta, \
                (const int*)cu, N, H, Hg, softplus_thr)
#define CP_SWITCH(AB) \
    switch (hpb) { \
        case 8: CP_LAUNCH(8, AB); break; \
        case 4: CP_LAUNCH(4, AB); break; \
        case 2: CP_LAUNCH(2, AB); break; \
        default: CP_LAUNCH(1, AB); break; \
    }
    if (ab_is_bf16) { CP_SWITCH(1); } else { CP_SWITCH(0); }
#undef CP_SWITCH
#undef CP_LAUNCH
    return (int)hipGetLastError();
}

/* ================================ the decode convolution ================================ */

template <int HPB>
__global__ __launch_bounds__(HPB * 32)
void r3d_gdn_conv_update_kernel(const unsigned short* __restrict__ x, long long xpitch,
                                const unsigned short* __restrict__ wgt,
                                const unsigned short* __restrict__ bias,
                                unsigned short* __restrict__ cstate,
                                long long cs_seq, long long cs_dim, long long cs_tok,
                                int state_len_max,
                                const int* __restrict__ cache_idx, long long ci_stride,
                                const int* __restrict__ naccept,
                                unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                                unsigned short* __restrict__ v,
                                const int* __restrict__ cu, int H, int Hg, int maxq)
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = blockIdx.x;
    const int bos = cu[n], slen = cu[n + 1] - bos;
    if (slen <= 0) return;
    const int slot = cache_idx[(size_t)n * ci_stride];
    if (slot < 0) return;

    const int off = naccept ? (naccept[n] - 1) : 0;
    const int slen_eff = state_len_max - (maxq - slen);

    const int unit = blockIdx.y * HPB + warp;
    const int d0 = unit * kHD + lane * kDPL;

    static_assert(kW == 4, "the weight load reads a channel's taps as one ushort4");
    float wt[kW][kDPL], bs[kDPL];
#pragma unroll
    for (int c = 0; c < kDPL; ++c) {
        const ushort4 w4 = *(const ushort4*)(wgt + (size_t)(d0 + c) * kW);
        const unsigned short* ws = (const unsigned short*)&w4;
#pragma unroll
        for (int i = 0; i < kW; ++i) wt[i][c] = bf16_to_f32(ws[i]);
        bs[c] = bias ? bf16_to_f32(bias[d0 + c]) : 0.0f;
    }

    const int csd = (int)cs_dim, cst = (int)cs_tok, xpi = (int)xpitch;
    unsigned short* const sp = cstate + (size_t)slot * cs_seq + (size_t)d0 * cs_dim;
    const unsigned short* const xrow = x + (size_t)bos * xpitch + d0;
    ushort4 xb[kMAXT];
    const bool xpre = (slen <= kMAXT);
#pragma unroll
    for (int t = 0; t < kMAXT; ++t)
        if (xpre && t < slen) xb[t] = *(const ushort4*)(xrow + t * xpi);
    float win[kW][kDPL];
    unsigned short hist[kST][kDPL];
    {
        const unsigned short* hp = sp + off * cst;
#pragma unroll
        for (int i = 0; i < kST; ++i)
#pragma unroll
            for (int c = 0; c < kDPL; ++c) {
                const unsigned short raw = hp[c * csd + i * cst];
                hist[i][c] = raw;
                win[i][c] = bf16_to_f32(raw);
            }
    }

    const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
    const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

    unsigned short* op = (region == 2)
        ? (v + ((size_t)bos * H + hv) * kHD + lane * kDPL)
        : ((region ? k : q) + ((size_t)bos * Hg + (region ? hk : hq)) * kHD + lane * kDPL);
    const int ostep = (region == 2) ? H * kHD : Hg * kHD;

#define CU_TOKEN_BODY(RAW)                                                       \
    {                                                                            \
        const ushort4 raw_ = (RAW);                                              \
        win[kW - 1][0] = bf16_to_f32(raw_.x); win[kW - 1][1] = bf16_to_f32(raw_.y); \
        win[kW - 1][2] = bf16_to_f32(raw_.z); win[kW - 1][3] = bf16_to_f32(raw_.w); \
        unsigned short out[kDPL];                                                \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < kDPL; ++c) {                                         \
            float acc = bs[c];                                                   \
            _Pragma("unroll")                                                    \
            for (int i = 0; i < kW; ++i) acc += wt[i][c] * win[i][c];            \
            out[c] = f32_to_bf16(cp_silu(acc));                                  \
        }                                                                        \
        *(ushort4*)op = make_ushort4(out[0], out[1], out[2], out[3]);            \
        op += ostep;                                                             \
        _Pragma("unroll")                                                        \
        for (int i = 0; i < kW - 1; ++i)                                         \
            _Pragma("unroll")                                                    \
            for (int c = 0; c < kDPL; ++c) win[i][c] = win[i + 1][c];            \
    }
    if (xpre) {
#pragma unroll
        for (int t = 0; t < kMAXT; ++t) {
            if (t < slen) CU_TOKEN_BODY(xb[t])
        }
    } else {
        for (int t = 0; t < slen; ++t) CU_TOKEN_BODY(*(const ushort4*)(xrow + t * xpi))
    }

    // Rewrite the cache shifted by one.
    const int VAL = kW - 2;
    const int nh = slen_eff - slen < 0 ? 0 : (slen_eff - slen);
#pragma unroll
    for (int c = 0; c < kDPL; ++c) {
        unsigned short* dst = sp + c * csd;
#pragma unroll
        for (int i = 0; i + 1 < kST; ++i)
            if (i < nh && i < slen_eff) dst[i * cst] = hist[i + 1][c];
        if (xpre) {
#pragma unroll
            for (int t = 0; t < kMAXT; ++t) {
                const int i = t + VAL;
                if (i >= nh && i < slen_eff && t < slen) {
                    const ushort4 vv = xb[t];
                    dst[i * cst] = (c == 0) ? vv.x : (c == 1) ? vv.y : (c == 2) ? vv.z : vv.w;
                }
            }
        }
        for (int i = nh; i < slen_eff; ++i) {
            const int ti = i - VAL;
            if (!xpre || ti < 0 || ti >= slen) dst[i * cst] = xrow[ti * xpi + c];
        }
    }
#undef CU_TOKEN_BODY
}

int r3d_gdn_conv_update_launch(const void* x, long long xpitch, const void* wgt, const void* bias,
                               void* cstate, long long cs_seq, long long cs_dim, long long cs_tok,
                               int state_len_max, const void* cache_idx, long long ci_stride,
                               const void* num_accepted, void* q, void* k, void* v,
                               const void* cu, int N, int H, int Hg, int width, int max_query_len,
                               hipStream_t stream)
{
    const int units = 2 * Hg + H;
    int hpb = 8;
    while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
    while (hpb > 1 && (long)N * (units / hpb) < 256) hpb >>= 1;
    dim3 grid(N, units / hpb), blk(hpb * 32);
#define CU_LAUNCH(P) \
    R3D_KLAUNCH(R3D_KNAME((r3d_gdn_conv_update_kernel<P>)), grid, blk, stream, \
                (const unsigned short*)x, xpitch, (const unsigned short*)wgt, \
                (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim, \
                cs_tok, state_len_max, (const int*)cache_idx, ci_stride, \
                (const int*)num_accepted, (unsigned short*)q, (unsigned short*)k, \
                (unsigned short*)v, (const int*)cu, H, Hg, max_query_len)
    switch (hpb) {
        case 8: CU_LAUNCH(8); break;
        case 4: CU_LAUNCH(4); break;
        case 2: CU_LAUNCH(2); break;
        default: CU_LAUNCH(1); break;
    }
#undef CU_LAUNCH
    return (int)hipGetLastError();
}

/* A contiguous cu (rank 1, >= 2) gives the sequence count; -1 on refusal. */
static int seqs_of(const RadTensor* cu, int* out)
{
    if (!cu || cu->rank != 1 || cu->shape[0] < 2) return -1;
    if (!r3d_contig(cu)) return -2;
    *out = (int)(cu->shape[0] - 1);
    return 0;
}

}  // namespace

/* ------------------------------------------------------------------ the plugin ABI: conv prep */

enum { CP_X = 0, CP_W, CP_B, CP_ALOG, CP_DTB, CP_CSTATE, CP_CU, CP_A, CP_BGATE, CP_CACHE_IDX,
       CP_HAS_INIT, CP_Q, CP_K, CP_V, CP_G, CP_BETA };

extern "C" int r3d_rad_gdn_conv_prep(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x    = r3d_opt(a, CP_X);
    const RadTensor* w    = r3d_opt(a, CP_W);
    const RadTensor* b    = r3d_opt(a, CP_B);
    const RadTensor* alog = r3d_opt(a, CP_ALOG);
    const RadTensor* dtb  = r3d_opt(a, CP_DTB);
    const RadTensor* cst  = r3d_opt(a, CP_CSTATE);
    const RadTensor* cu   = r3d_opt(a, CP_CU);
    const RadTensor* q    = r3d_opt(a, CP_Q);
    const RadTensor* k    = r3d_opt(a, CP_K);
    const RadTensor* v    = r3d_opt(a, CP_V);
    const RadTensor* g    = r3d_opt(a, CP_G);
    const RadTensor* beta = r3d_opt(a, CP_BETA);
    const RadTensor* ta   = r3d_opt(a, CP_A);
    const RadTensor* tb   = r3d_opt(a, CP_BGATE);
    const RadTensor* cidx = r3d_opt(a, CP_CACHE_IDX);
    const RadTensor* hin  = r3d_opt(a, CP_HAS_INIT);

    if (!r3d_p(x) || !r3d_p(w) || !r3d_p(alog) || !r3d_p(dtb) || !r3d_p(cst) || !r3d_p(cu) ||
        !r3d_wp(q) || !r3d_wp(k) || !r3d_wp(v) || !r3d_wp(g) || !r3d_wp(beta))
        return R3D_NO(RAD_E_INVAL);
    // Optional in the schema but required here: the kernel dereferences all three.
    if (!r3d_p(ta) || !r3d_p(tb) || !r3d_p(cidx)) return R3D_NO(RAD_E_INVAL);

    if (x->rank < 2 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R3D_NO(RAD_E_SHAPE);
    if (g->rank != 2 || beta->rank != 2 || cst->rank != 3 || ta->rank != 2) return R3D_NO(RAD_E_SHAPE);

    int N = 0;
    if (seqs_of(cu, &N) != 0) return R3D_NO(RAD_E_SHAPE);

    const long long xpitch = r3d_row_pitch(x);
    if (xpitch < 0) return R3D_NO(RAD_E_STRIDE);
    const long long ab_stride = r3d_row_pitch(ta);
    if (ab_stride < 0) return R3D_NO(RAD_E_STRIDE);
    if (r3d_row_pitch(tb) != ab_stride) return R3D_NO(RAD_E_STRIDE);

    if (!r3d_contig(q) || !r3d_contig(k) || !r3d_contig(v) || !r3d_contig(g) ||
        !r3d_contig(beta) || !r3d_contig(w) || !r3d_contig(alog) || !r3d_contig(dtb))
        return R3D_NO(RAD_E_STRIDE);
    if (cidx->rank < 1) return R3D_NO(RAD_E_SHAPE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R3D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T) return R3D_NO(RAD_E_SHAPE);
    if (g->shape[0] != T || g->shape[1] != H) return R3D_NO(RAD_E_SHAPE);
    if (beta->shape[0] != T || beta->shape[1] != H) return R3D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3D_NO(RAD_E_SHAPE);
    if (K != kHD || V != kHD) return R3D_NO(RAD_E_SHAPE);

    const int width = (int)r3d_geti_or(a, "conv_width", kW);
    if (width != kW) return R3D_NO(RAD_E_SHAPE);
    const float thr = (float)r3d_getf(a, "softplus_thr", 20.0);
    if ((float)r3d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R3D_NO(RAD_E_UNSUPPORTED);
    const int ab_is_bf16 = (ta->dtype == RAD_BF16) ? 1 : 0;
    if (tb->dtype != ta->dtype) return R3D_NO(RAD_E_DTYPE);
    if (!ab_is_bf16 && ta->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_bf16(x) || !r3d_is_bf16(w) || !r3d_is_bf16(cst) ||
        !r3d_is_bf16(q) || !r3d_is_bf16(k) || !r3d_is_bf16(v) ||
        !r3d_is_f32(alog) || !r3d_is_f32(dtb) || !r3d_is_f32(g) || !r3d_is_f32(beta))
        return R3D_NO(RAD_E_DTYPE);

    return r3d_gdn_conv_prep_launch(r3d_p(x), xpitch, r3d_p(w), b ? r3d_p(b) : nullptr, r3d_wp(cst),
                                    cst->stride[0], cst->stride[1], cst->stride[2],
                                    r3d_p(cidx), cidx->stride[0], hin ? r3d_p(hin) : nullptr,
                                    hin && (hin->dtype == RAD_I32 || hin->dtype == RAD_U32),
                                    r3d_p(ta), r3d_p(tb), ab_stride, ab_is_bf16,
                                    r3d_p(alog), r3d_p(dtb), r3d_wp(q), r3d_wp(k), r3d_wp(v),
                                    r3d_wp(g), r3d_wp(beta), r3d_p(cu),
                                    N, (int)T, (int)H, (int)Hg, width, thr, (hipStream_t)s);
}

/* ------------------------------------------------------------------ the plugin ABI: conv update */

enum { CU_X = 0, CU_W, CU_B, CU_CSTATE, CU_SIDX, CU_NACC, CU_CU, CU_Q, CU_K, CU_V };

extern "C" int r3d_rad_gdn_conv_update(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x    = r3d_opt(a, CU_X);
    const RadTensor* w    = r3d_opt(a, CU_W);
    const RadTensor* b    = r3d_opt(a, CU_B);
    const RadTensor* cst  = r3d_opt(a, CU_CSTATE);
    const RadTensor* sidx = r3d_opt(a, CU_SIDX);
    const RadTensor* nacc = r3d_opt(a, CU_NACC);
    const RadTensor* cu   = r3d_opt(a, CU_CU);
    const RadTensor* q    = r3d_opt(a, CU_Q);
    const RadTensor* k    = r3d_opt(a, CU_K);
    const RadTensor* v    = r3d_opt(a, CU_V);
    if (!r3d_p(x) || !r3d_p(w) || !r3d_p(cst) || !r3d_p(sidx) || !r3d_p(cu) ||
        !r3d_wp(q) || !r3d_wp(k) || !r3d_wp(v))
        return R3D_NO(RAD_E_INVAL);
    if (cst->rank != 3 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R3D_NO(RAD_E_SHAPE);
    if (sidx->rank < 1) return R3D_NO(RAD_E_SHAPE);

    int N = 0;
    if (seqs_of(cu, &N) != 0) return R3D_NO(RAD_E_SHAPE);

    const long long xpitch = r3d_row_pitch(x);
    if (xpitch < 0) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(q) || !r3d_contig(k) || !r3d_contig(v) || !r3d_contig(w))
        return R3D_NO(RAD_E_STRIDE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (H == 0 || Hg == 0 || H % Hg) return R3D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || k->shape[0] != T) return R3D_NO(RAD_E_SHAPE);
    if (K != kHD || V != kHD) return R3D_NO(RAD_E_SHAPE);

    if (!r3d_is_bf16(x) || !r3d_is_bf16(w) || !r3d_is_bf16(cst) ||
        !r3d_is_bf16(q) || !r3d_is_bf16(k) || !r3d_is_bf16(v))
        return R3D_NO(RAD_E_DTYPE);

    const int width = (int)r3d_geti_or(a, "conv_width", kW);
    const long long state_len_max = cst->shape[2];
    const long long mql = state_len_max - (width - 2);
    if (width != kW || mql < 1) return R3D_NO(RAD_E_SHAPE);

    return r3d_gdn_conv_update_launch(r3d_p(x), xpitch, r3d_p(w), b ? r3d_p(b) : nullptr,
                                      r3d_wp(cst), cst->stride[0], cst->stride[1], cst->stride[2],
                                      (int)state_len_max, r3d_p(sidx), sidx->stride[0],
                                      nacc ? r3d_p(nacc) : nullptr, r3d_wp(q), r3d_wp(k), r3d_wp(v),
                                      r3d_p(cu), N, (int)H, (int)Hg, width, (int)mql,
                                      (hipStream_t)s);
}
