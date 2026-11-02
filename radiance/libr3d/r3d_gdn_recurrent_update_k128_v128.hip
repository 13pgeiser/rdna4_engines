/* r3d_gdn_recurrent_update_k128_v128.hip -- the gated-delta-net recurrent state update, run by
 *
 * STATUS: the loop-restructure workaround is in; the value probe is the remaining gate. Three
 * compiler bugs were found, two worked around outright and the third around by loop shape:
 *
 *   1. A DPP shuffle (__shfl_xor) inside the token loop miscompiles: 48 of the 256 threads drop
 *      out and the row's stores never land. Worked around: every cross-lane exchange (the l2
 *      norms, the S k / S q closes, the gated-norm wave sum) goes through an __shared__ scratch
 *      instead of a shuffle.
 *
 *   2. As a template <int NORM, int ABF16> the kernel's device code is present and the launch
 *      succeeds, but the kernel never runs (not even a prologue sentinel lands), while a plain
 *      kernel in the same file runs fine. Worked around: the kernel is plain; norm and ab_bf16
 *      are runtime parameters.
 *
 *   3. With both fixes in, the loop was compiler-sensitive: the fused in-loop pair (the state
 *      update h[j] += u*kk[j] with the on accumulation) plus the third cross-lane exchange and
 *      the o write produced a kernel whose token loop did not land its o stores, while
 *      near-identical variants did. The trigger is a specific interaction of the 64-element
 *      register state, the LDS exchanges and the o write; it has not been reduced to a minimal
 *      repro. Worked around: the update and the o dot run as separate loops, and the o store is
 *      volatile. The value probe is the gate.
 *
 * decode (prefill runs the chunked scan). One kernel for the whole step:
 *
 *     g = -exp(A_log) . softplus(a + dt_bias)      beta = sigmoid(b)      q, k l2-normalised
 *     S  *= e^g
 *     u   = beta . (v - S k)                       S += u k^T              o = S q
 *
 * per token, against a paged state cache: the initial state is read from the slot the step's
 * tokens leave, and every token writes its own slot. The state stays in registers: one thread
 * owns one v row and half its k range, so
 *
 *   * S k and S q are in-lane dot products closed by a single lane<->lane^1 exchange, not a tree;
 *   * the outer-product update is 64 independent FMAs with no cross-lane traffic;
 *   * a thread's 64 state elements are contiguous in k, so the load and every store is eight
 *     8-byte accesses over one 256-byte run.
 *
 * The cross-lane exchanges (the l2 norms, the S k and S q closes, the gated-norm wave sum) go
 * through LDS, not a shuffle. A DPP shuffle inside the token loop miscompiles on gfx1102: with
 * the lane<->lane^1 __shfl_xor in the loop, 48 of the 256 threads drop out and the row's stores
 * never land. Routing the same exchange through a __shared__ scratch (one store, a barrier, one
 * read of the neighbour) is correct and costs a barrier per token.
 *
 * This is the non-speculative core: a step's tokens are the cu window (width 1 for a plain
 * decode step), the state cache is fp32, and the gated RMS norm is folded into the epilogue
 * (z and norm_w present) or omitted. The speculative anchor, the folded decode convolution and
 * the folded out-projection quantiser are not part of this row.
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

constexpr int kK    = 128;   // head_k
constexpr int kV    = 128;   // head_v
constexpr int kHK   = 64;    // k elements per thread (kK / 2)
constexpr int kTHR  = kV * 2;  // two threads per v row

typedef float ru_f4 __attribute__((ext_vector_type(4)));

template <int NT>
__device__ __forceinline__ void ru_load8(float* d, const float* p)
{
    const ru_f4* q = (const ru_f4*)p;
    *(ru_f4*)d       = NT ? __builtin_nontemporal_load(q)     : *q;
    *(ru_f4*)(d + 4) = NT ? __builtin_nontemporal_load(q + 1) : *(q + 1);
}

template <int NT>
__device__ __forceinline__ void ru_store8(float* p, const float* d)
{
    ru_f4* q = (ru_f4*)p;
    if (NT) {
        __builtin_nontemporal_store(*(const ru_f4*)d,       q);
        __builtin_nontemporal_store(*(const ru_f4*)(d + 4), q + 1);
    } else {
        *q       = *(const ru_f4*)d;
        *(q + 1) = *(const ru_f4*)(d + 4);
    }
}

__device__ __forceinline__ float ru_softplus(float x, float thr)
{
    if (x > thr) return x;
    return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

template <int ABF16>
__device__ __forceinline__ float ru_ab(const void* p, size_t i)
{
    return ABF16 ? bf16_to_f32(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

__device__ __forceinline__ float ru_act(float z, int kind)
{
    const float s = 1.0f / (1.0f + __expf(-z));
    return kind ? s : z * s;
}

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

/* norm: 0 = no epilogue norm, 1 = sigmoid, 2 = silu (the row's `act`). ab_bf16: a/b are bf16.
 * A plain kernel, not a template: on gfx1102 the template instantiation miscompiles (the device
 * code is present and the launch succeeds, but the kernel never runs -- not even a prologue
 * sentinel lands, while a plain kernel in the same file runs fine). */
__global__ __launch_bounds__(kTHR)
void r3d_gdn_recurrent_update_kernel(const unsigned short* __restrict__ q,
                                     const unsigned short* __restrict__ k,
                                     const unsigned short* __restrict__ v,
                                     const void* __restrict__ av, const void* __restrict__ bv,
                                     long long ab_stride,
                                     const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                     float* __restrict__ state,
                                     long long st_slot, long long st_head,
                                     const int* __restrict__ sidx, long long sidx_stride,
                                     const unsigned short* __restrict__ zg, long long zg_pitch,
                                     const float* __restrict__ nw,
                                     unsigned short* __restrict__ o,
                                     const int* __restrict__ cu, int H, int Hg,
                                     float scale, float sp_thr, float neps,
                                     int norm, int ab_bf16)
{
    const int tid = threadIdx.x;
    const int row = (tid >> 1), half = tid & 1, k0 = half * kHK;
    const int n = blockIdx.x, hv = blockIdx.y;
    const int bos = cu[n], T = cu[n + 1] - bos;
    if (T <= 0) return;
    const int hk = hv / (H / Hg);

    /* Read early so their latency overlaps the state load. */
    const float a_log_h = A_log[hv], dtb = dt_bias[hv];
    const float alog = __expf(a_log_h);

    float h[kHK];
    {
        const float* p = state + (size_t)sidx[(size_t)n * sidx_stride] * st_slot
                       + (size_t)hv * st_head + (size_t)row * kK + k0;
#pragma unroll
        for (int j = 0; j < kHK; j += 8) ru_load8<0>(&h[j], p + j);
    }

    // LDS scratch for the cross-lane exchanges. ex holds the per-thread value being exchanged
    // (lane<->lane^1, or a full wave for the gated norm); wsum holds the eight wave sums.
    __shared__ float ex[2 * kTHR];
    __shared__ float wsum[kTHR / 32];

    for (int t = 0; t < T; ++t) {
        const int tok = bos + t;
        const unsigned short* qp = q + ((size_t)tok * Hg + hk) * kK + k0;
        const unsigned short* kp = k + ((size_t)tok * Hg + hk) * kK + k0;

        float qq[kHK], kk[kHK];
        for (int j = 0; j < kHK; ++j) { qq[j] = bf16_to_f32(qp[j]); kk[j] = bf16_to_f32(kp[j]); }

        // l2 norms: 64 in-lane terms, then one exchange with the lane holding the other half.
        // (LDS, not a shuffle -- a DPP in this loop miscompiles on gfx1102.)
        float sq = 0.0f, sk = 0.0f;
        for (int j = 0; j < kHK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
        ex[tid] = sq; ex[tid + kTHR] = sk;
        R3D_BARRIER();
        sq += ex[tid ^ 1]; sk += ex[tid ^ 1 + kTHR];
        const float qs = __frsqrt_rn(sq + 1e-6f) * scale, ks = __frsqrt_rn(sk + 1e-6f);
        for (int j = 0; j < kHK; ++j) { qq[j] *= qs; kk[j] *= ks; }

        const size_t ab_i = (size_t)tok * ab_stride + hv;
        const float ra = ab_bf16 ? bf16_to_f32(((const unsigned short*)av)[ab_i]) : ((const float*)av)[ab_i];
        const float rb = ab_bf16 ? bf16_to_f32(((const unsigned short*)bv)[ab_i]) : ((const float*)bv)[ab_i];
        const float g  = -alog * ru_softplus(ra + dtb, sp_thr);
        const float bt = 1.0f / (1.0f + __expf(-rb));
        const float eg = __expf(g);

        float dot = 0.0f;
        for (int j = 0; j < kHK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
        ex[tid] = dot;
        R3D_BARRIER();
        dot += ex[tid ^ 1];

        const float vraw = bf16_to_f32(v[((size_t)tok * H + hv) * kV + row]);
        const float u = (vraw - dot) * bt;

        // The outer-product update and the o dot run as separate loops: fused, with the
        // third exchange and the o write live, the token loop's o stores stopped landing on
        // gfx1102 (a compiler interaction, not reduced to a minimal repro); split, they land.
        for (int j = 0; j < kHK; ++j) h[j] += u * kk[j];
        float on = 0.0f;
        for (int j = 0; j < kHK; ++j) on += h[j] * qq[j];
        ex[tid] = on;
        R3D_BARRIER();
        on += ex[tid ^ 1];

        if (norm) {
            ex[tid] = (half == 0) ? on * on : 0.0f;
            R3D_BARRIER();
            const int wave = tid >> 5;
            float s = 0.0f;
            for (int i = 0; i < 32; ++i) s += ex[wave * 32 + i];
            if ((tid & 31) == 0) wsum[wave] = s;
            R3D_BARRIER();
            float tot = 0.0f;
            for (int i = 0; i < kTHR / 32; ++i) tot += wsum[i];
            const float sc = __frsqrt_rn(tot * (1.0f / kV) + neps);
            const unsigned short zr = zg[(size_t)tok * zg_pitch + (size_t)hv * kV + row];
            on *= sc * nw[row] * ru_act(bf16_to_f32(zr), norm);
            R3D_BARRIER();
        }

        const unsigned short ob = f32_to_bf16(on);
        if (half == 0) {
            // Volatile: the plain store was the one the compiler dropped in the fused form.
            volatile unsigned short* ov = o;
            ov[((size_t)tok * H + hv) * kV + row] = ob;
        }

        const int so = sidx[(size_t)n * sidx_stride + t];
        if (so >= 0) {
            float* p = state + (size_t)so * st_slot + (size_t)hv * st_head
                       + (size_t)row * kK + k0;
            for (int j = 0; j < kHK; j += 8) ru_store8<0>(p + j, &h[j]);
        }
    }
}

int r3d_gdn_recurrent_update_launch(const void* q, const void* k, const void* v, const void* a,
                                    const void* b, long long ab_stride, const void* A_log,
                                    const void* dt_bias, void* state, long long st_slot,
                                    long long st_head, const void* sidx, long long sidx_stride,
                                    const void* zg, long long zg_pitch, const void* nw,
                                    void* o, const void* cu, int N, int H, int Hg, int norm,
                                    int ab_bf16, float scale, float sp_thr, float neps,
                                    hipStream_t stream)
{
    dim3 grid(N, H), blk(kTHR);
    // A plain kernel: norm and ab_bf16 are runtime parameters, not template arguments.
    R3D_KLAUNCH(r3d_gdn_recurrent_update_kernel, grid, blk, stream,
                (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
                a, b, ab_stride, (const float*)A_log, (const float*)dt_bias, (float*)state,
                st_slot, st_head, (const int*)sidx, sidx_stride, (const unsigned short*)zg,
                zg_pitch, (const float*)nw, (unsigned short*)o, (const int*)cu, H, Hg,
                scale, sp_thr, neps, norm, ab_bf16);
    return (int)hipGetLastError();
}

}  // namespace

/* ------------------------------------------------------------------ the plugin ABI */

enum { RU_Q = 0, RU_K, RU_V, RU_A, RU_B, RU_ALOG, RU_DTB, RU_STATE, RU_SIDX, RU_NACC, RU_CU,
       RU_Z, RU_NORMW, RU_O, RU_OQ, RU_OSCALE };

extern "C" int r3d_rad_gdn_recurrent_update(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* q    = r3d_opt(a, RU_Q);
    const RadTensor* k    = r3d_opt(a, RU_K);
    const RadTensor* v    = r3d_opt(a, RU_V);
    const RadTensor* ta   = r3d_opt(a, RU_A);
    const RadTensor* tb   = r3d_opt(a, RU_B);
    const RadTensor* alog = r3d_opt(a, RU_ALOG);
    const RadTensor* dtb  = r3d_opt(a, RU_DTB);
    const RadTensor* st   = r3d_opt(a, RU_STATE);
    const RadTensor* sidx = r3d_opt(a, RU_SIDX);
    const RadTensor* nacc = r3d_opt(a, RU_NACC);
    const RadTensor* cu   = r3d_opt(a, RU_CU);
    const RadTensor* z    = r3d_opt(a, RU_Z);
    const RadTensor* nw   = r3d_opt(a, RU_NORMW);
    const RadTensor* o    = r3d_opt(a, RU_O);
    // o_q and o_scale are the folded quantiser's outputs; this row does not fold it.
    if (!r3d_p(q) || !r3d_p(k) || !r3d_p(v) || !r3d_p(ta) || !r3d_p(tb) || !r3d_p(alog) ||
        !r3d_p(dtb) || !r3d_wp(st) || !r3d_p(sidx) || !r3d_p(cu) || !r3d_wp(o))
        return R3D_NO(RAD_E_INVAL);

    if (q->rank != 3 || k->rank != 3 || v->rank != 3 || st->rank != 4 || o->rank != 3)
        return R3D_NO(RAD_E_SHAPE);
    if (sidx->rank < 2) return R3D_NO(RAD_E_SHAPE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R3D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || o->shape[0] != T || o->shape[1] != H || o->shape[2] != V)
        return R3D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3D_NO(RAD_E_SHAPE);
    if (K != kK || V != kV) return R3D_NO(RAD_E_SHAPE);

    if (!r3d_is_bf16(q) || !r3d_is_bf16(k) || !r3d_is_bf16(v) || !r3d_is_bf16(o))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(alog) || !r3d_is_f32(dtb)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(st)) return R3D_NO(RAD_E_DTYPE);   // fp32 state cache
    if (ta->dtype != RAD_BF16 && ta->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (tb->dtype != ta->dtype) return R3D_NO(RAD_E_DTYPE);

    // state cache [slots, H, V, K]: the per-head and per-slot strides.
    if (st->shape[1] != H || st->shape[2] != kV || st->shape[3] != kK) return R3D_NO(RAD_E_SHAPE);
    const long long st_head = st->stride[1];
    const long long st_slot = st->stride[0];
    if (st_head != (long long)kV * kK) return R3D_NO(RAD_E_STRIDE);

    // a and b: [T, H] (the ba projection's rows), row pitch in elements.
    const long long ab_stride = r3d_row_pitch(ta);
    if (ab_stride < 0 || r3d_row_pitch(tb) != ab_stride) return R3D_NO(RAD_E_STRIDE);

    const long long sidx_stride = sidx->stride[sidx->rank - 1] == 1 ? sidx->stride[0] : -1;
    if (sidx_stride < 0) return R3D_NO(RAD_E_STRIDE);

    int N = 0;
    if (cu->rank != 1 || cu->shape[0] < 2 || !r3d_contig(cu)) return R3D_NO(RAD_E_SHAPE);
    N = (int)(cu->shape[0] - 1);

    const float scale = (float)r3d_getf(a, "scale", 1.0 / sqrt((double)kK));
    const float sp_thr = (float)r3d_getf(a, "softplus_thr", 20.0);
    const float neps = (float)r3d_getf(a, "eps", 1e-6);

    // The gated RMS norm epilogue: present when both z and norm_w are given.
    int norm = 0;
    long long zg_pitch = 0;
    if (r3d_p(z) && r3d_p(nw)) {
        if (z->rank != 3 || z->shape[0] != T || z->shape[1] != H || z->shape[2] != kV)
            return R3D_NO(RAD_E_SHAPE);
        if (!r3d_is_bf16(z) || !r3d_is_f32(nw) || r3d_numel(nw) != kV) return R3D_NO(RAD_E_DTYPE);
        const int act = r3d_act_code(a, "act", nullptr, 0);
        if (act != 0 && act != 1) return R3D_NO(RAD_E_INVAL);
        norm = (act == 0) ? 2 : 1;   // r3d_act_code: 0=silu, 1=sigmoid; kernel: 2=silu, 1=sigmoid
        zg_pitch = z->stride[0];
    }

    const int e = r3d_gdn_recurrent_update_launch(r3d_p(q), r3d_p(k), r3d_p(v), r3d_p(ta), r3d_p(tb),
                                           ab_stride, r3d_p(alog), r3d_p(dtb), r3d_wp(st),
                                           st_slot, st_head, r3d_p(sidx), sidx_stride,
                                           r3d_p(z), zg_pitch, r3d_p(nw), r3d_wp(o), r3d_p(cu),
                                           N, (int)H, (int)Hg, norm, (ta->dtype == RAD_BF16),
                                           scale, sp_thr, neps, (hipStream_t)s);
    /* The launcher's codes are raw: its own negatives and hipGetLastError's positives. The
     * plugin ABI takes RadStatus, so normalise. */
    if (e < 0) return R3D_NO(RAD_E_INVAL);
    return e ? R3D_NO(RAD_E_DEVICE) : RAD_OK;
}
