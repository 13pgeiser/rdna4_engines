/* r3d_moe_router_topk_bf16.hip -- router_topk: softmax over all n_expert logits, the top_k by
 * value, and -- if `norm` -- the kept weights renormalised to sum to one.
 *
 * One wave per row (one block per row); the selection is top_k passes of a wave minimum, not a
 * sort. That matches the reference exactly (largest remaining value, ties to the lower id); a sort
 * with another tie rule would be a different router on tied rows, which are not rare. Selection
 * runs on the logits: softmax is monotone, so the order is the same and the comparison exact.
 *
 * gfx1102: a wave reduction (the max, the sum, the per-round min) in a loop body miscompiles for
 * a non-first wave -- wave 0 is correct, wave 1+ drop out. Two things keep this safe:
 *   * one wave per block (one block per row): the __syncthreads is a single-wave barrier, so there
 *     is no cross-wave sync whose ordering the compiler can scramble;
 *   * a separate LDS slot per top_k round: the loop never reuses a slot, so a read-after-write
 *     across iterations cannot be folded (the max and the sum use their own slots, rounds 0 and 1).
 * Every wave reduction is an LDS pass, not a DPP butterfly (a DPP exchange miscompiles here).
 *
 * Verified: single-row (equal, distinct, random) and multi-row (M >= 2) all pick the right ids and
 * weights.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

namespace {

constexpr int kW = 32;

/* (order-preserving value, ~id); 0 = absent or NaN. The key's high half is the logit's bits with
 * the sign folded (so larger logit -> larger key) and the low half is the complemented id, so a
 * tie goes to the lower id. */
__device__ __forceinline__ uint32_t topk_key(uint16_t raw, uint32_t e, bool ok) {
    const bool nan = ((raw & 0x7F80u) == 0x7F80u) && (raw & 0x7Fu);
    const uint16_t u = (raw == 0x8000u) ? (uint16_t)0u : raw;   // -0.0 folds to +0.0
    const uint16_t k = (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
    return nan ? 0u : (((uint32_t)k << 16) | (~e & 0xFFFFu));
}
__device__ __forceinline__ uint16_t topk_key_raw(uint32_t key) {
    const uint32_t k = key >> 16;
    return (uint16_t)((k & 0x8000u) ? (k & 0x7FFFu) : ~k);
}

/* One row, one wave (the block is a single wave). `wmin` holds 32 slots per round: round 0 the
 * max, round 1 the sum, rounds 2..(1+top_k) the per-round minimums. A separate slot per round so
 * the loop never reuses one (gfx1102). */
template <int MAXOWN>
__device__ __forceinline__ void r3d_router_topk_row(
    const unsigned short* __restrict__ lg, int32_t* __restrict__ ids,
    unsigned short* __restrict__ w, uint32_t m, uint32_t n_expert, uint32_t top_k,
    uint32_t lg_ld, int norm, uint32_t lane, __shared__ uint32_t* wmin)
{
    const unsigned short* row = lg + (size_t)m * lg_ld;

    uint16_t raw[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        raw[i] = row[e < n_expert ? e : 0u];
    }
    float v[MAXOWN];
    uint32_t key[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        const bool ok = e < n_expert;
        v[i] = ok ? bf16_to_f32(raw[i]) : -INFINITY;
        key[i] = topk_key(raw[i], e, ok) & (0u - (uint32_t)ok);
    }

    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) mx = fmaxf(mx, v[i]);

    // Wave max through LDS (round 0).
    wmin[lane] = __builtin_bit_cast(uint32_t, mx);
    R3D_BARRIER();
#pragma unroll
    for (int i = 0; i < kW; ++i)
        mx = fmaxf(mx, __builtin_bit_cast(float, wmin[i]));

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) sum += __expf(v[i] - mx);
    // Wave sum through LDS (round 1).
    wmin[kW + lane] = __builtin_bit_cast(uint32_t, sum);
    R3D_BARRIER();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kW; ++i) s += __builtin_bit_cast(float, wmin[kW + i]);
    sum = s;
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    uint32_t P = 0xFFFFFFFFu, mykey = 0u, nsel = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint32_t d = 0xFFFFFFFFu;
#pragma unroll
        for (int i = 0; i < MAXOWN; ++i) {
            const uint32_t t = P - key[i];
            d = d < t ? d : t;
        }
        // Wave minimum through LDS, round (2 + r): a fresh slot each round.
        const uint32_t base = (2 + r) * kW;
        wmin[base + lane] = d;
        R3D_BARRIER();
        uint32_t mn = 0xFFFFFFFFu;
#pragma unroll
        for (int i = 0; i < kW; ++i) mn = mn < wmin[base + i] ? mn : wmin[base + i];
        const uint32_t best = P - mn;
        if ((best & 0xFFFFu) == 0u) break;
        if (lane == r) mykey = best;
        P = best - 1u;
        ++nsel;
    }
    const float p = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;
    const uint16_t pb = f32_to_bf16(p);
    float wsum = 0.0f;
    for (uint32_t r = 0; r < nsel; ++r)
        wsum += __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, p),
                                                                    (int)r));
    if (lane < nsel) {
        ids[(size_t)m * top_k + lane] = (int32_t)(~mykey & 0xFFFFu);
        w[(size_t)m * top_k + lane] =
            (norm && wsum > 0.0f) ? f32_to_bf16(bf16_to_f32(pb) * (1.0f / wsum)) : pb;
    }
}

template <int MAXOWN>
__global__ __launch_bounds__(kW) void r3d_router_topk_kernel(
    const unsigned short* __restrict__ lg, int32_t* __restrict__ ids,
    unsigned short* __restrict__ w, uint32_t M, uint32_t n_expert, uint32_t top_k,
    uint32_t lg_ld, int norm)
{
    __shared__ uint32_t wmin[kW * (2 + kW)];   // a slot per round, top_k <= kW
    const uint32_t lane = threadIdx.x;
    const uint32_t m = blockIdx.x;             // one block per row
    r3d_router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane, wmin);
}

}  // namespace



/* ------------------------------------------------------------------ router_sigmoid_topk
 *
 * The GLM-style router: scores[e] = sigmoid(logit[e]), the selection runs on
 * select[e] = scores[e] + bias[e] (a per-expert correction bias), and the kept weights are the
 * scores themselves, scaled by route_scale (and renormalised to sum to one if `norm`). Ties go to
 * the lower id, as in router_topk.
 *
 * The selection key is 64-bit (the 32-bit select's monotone bits, the low 32 the complemented id),
 * so a 64-bit LDS slot per round. The scores are staged in LDS so the weight is read back without a
 * global reload. */
template <int MAXOWN>
__device__ __forceinline__ void r3d_router_sigmoid_topk_row(
    const unsigned short* __restrict__ lg, const float* __restrict__ bias,
    int32_t* __restrict__ ids, unsigned short* __restrict__ w,
    uint32_t m, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld,
    float scale, int norm, uint32_t lane,
    __shared__ uint64_t* wmin, __shared__ float* s_score)
{
    const unsigned short* row = lg + (size_t)m * lg_ld;

    uint64_t key[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        const bool ok = e < n_expert;
        const float lgt = ok ? bf16_to_f32(row[e]) : -INFINITY;
        const float sc = 1.0f / (1.0f + __expf(-lgt));
        s_score[lane * MAXOWN + i] = sc;
        const float sel = ok ? (sc + bias[e]) : -INFINITY;
        const uint32_t bits = __builtin_bit_cast(uint32_t, sel);
        const uint32_t u = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        key[i] = (((uint64_t)u) << 32) | (~e & 0xFFFFFFFFull);
    }

    uint64_t P = ~0ull, mykey = 0ull, nsel = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint64_t d = ~0ull;  // min over (P - key[i]); P - key is smaller for the larger key
#pragma unroll
        for (int i = 0; i < MAXOWN; ++i) {
            const uint64_t t = P - key[i];
            d = d < t ? d : t;
        }
        const uint64_t base = (uint64_t)(2 + r) * kW;
        wmin[base + lane] = d;
        R3D_BARRIER();
        uint64_t mn = ~0ull;
#pragma unroll
        for (int i = 0; i < kW; ++i) mn = mn < wmin[base + i] ? mn : wmin[base + i];
        const uint64_t best = P - mn;
        if (best == 0ull) break;
        if (lane == r) mykey = best;
        P = best - 1ull;
        ++nsel;
    }

    const uint32_t id = (uint32_t)(~mykey & 0xFFFFFFFFull);
    const float sc = s_score[(id % kW) * MAXOWN + (id / kW)];
    // Sum the kept scores over lanes 0..nsel-1 (a wave reduce).
    wmin[lane] = (uint64_t)__builtin_bit_cast(uint32_t, sc);
    R3D_BARRIER();
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kW; ++i) {
        const uint32_t b = (uint32_t)wmin[i];
        if (i < nsel) sum += __builtin_bit_cast(float, b);
    }
    if (lane < nsel) {
        const float weight = sc * (norm ? (scale / (sum + 1e-20f)) : scale);
        ids[(size_t)m * top_k + lane] = (int32_t)id;
        w[(size_t)m * top_k + lane] = f32_to_bf16(weight);
    }
}

template <int MAXOWN>
__global__ __launch_bounds__(kW) void r3d_router_sigmoid_topk_kernel(
    const unsigned short* __restrict__ lg, const float* __restrict__ bias,
    int32_t* __restrict__ ids, unsigned short* __restrict__ w,
    uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, float scale, int norm)
{
    __shared__ uint64_t wmin[kW * (2 + kW)];
    __shared__ float s_score[kW * MAXOWN];
    const uint32_t lane = threadIdx.x;
    const uint32_t m = blockIdx.x;
    r3d_router_sigmoid_topk_row<MAXOWN>(lg, bias, ids, w, m, n_expert, top_k, lg_ld, scale,
                                        norm, lane, wmin, s_score);
}


extern "C" {

/* moe_scatter's public dispatch, reused by the fused router_topk_scatter. */
int r3d_rad_moe_scatter(const RadArgs*, RadStream);

int r3d_rad_moe_router_topk_bf16(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const long long M = r3d_geti_or(a, "M", 0), ne = r3d_geti_or(a, "n_expert", 0),
                    tk = r3d_geti_or(a, "top_k", 0);
    const long long norm_ll = r3d_geti_or(a, "norm", 0);
    const int norm = (int)norm_ll;
    if (M <= 0 || ne <= 0 || tk <= 0 || tk > kW) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);   // MAXOWN=16 for this row
    const RadTensor* L = r3d_opt(a, 0), *I = r3d_opt(a, 1), *W = r3d_opt(a, 2);
    if (!r3d_p(L) || !r3d_wp(I) || !r3d_wp(W)) return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kW);           // one block (one wave) per row
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else
        R3D_KLAUNCH(r3d_router_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    return (int)hipGetLastError();
}


/* router_topk_scatter: the pair as one op. Phase 1 is the top-k selection over the whole n_expert
 * (the router_topk kernel); phase 2 is the counting sort over this rank's slice (n_local), which
 * reuses moe_scatter's dispatch. Byte-identical to the two reference launches, in order. */
int r3d_rad_router_topk_scatter(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 6) return R3D_NO(RAD_E_INVAL);
    const long long M = r3d_geti_or(a, "M", 0), ne = r3d_geti_or(a, "n_expert", 0),
                    tk = r3d_geti_or(a, "top_k", 0), nl = r3d_geti_or(a, "n_local", ne);
    const int norm = (int)r3d_geti_or(a, "norm", 0);
    if (M <= 0 || ne <= 0 || tk <= 0 || tk > (long long)kW || tk > ne) return R3D_NO(RAD_E_SHAPE);
    if (nl <= 0 || nl > ne) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* L = r3d_opt(a, 0), *I = r3d_opt(a, 1), *W = r3d_opt(a, 2),
                    *ST = r3d_opt(a, 3), *OFF = r3d_opt(a, 4), *CNT = r3d_opt(a, 5);
    if (!r3d_p(L) || !r3d_wp(I) || !r3d_wp(W) || !r3d_wp(ST) || !r3d_wp(OFF) || !r3d_wp(CNT))
        return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);

    // Phase 1: the top-k selection (the router_topk kernel, over the whole n_expert).
    dim3 grid((unsigned)M), blk(kW);
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else
        R3D_KLAUNCH(r3d_router_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    if (hipGetLastError() != hipSuccess) return R3D_NO(RAD_E_INVAL);

    // Phase 2: the counting sort over this rank's slice (n_local), via moe_scatter's dispatch.
    RadTensor t2[4] = { *I, *ST, *OFF, *CNT };
    RadParam sp[4] = {
        { "n_expert", RAD_P_INT, (long long)nl, 0, nullptr, 0.0 },
        { "top_k", RAD_P_INT, (long long)tk, 0, nullptr, 0.0 },
        { "expert_base", RAD_P_INT, r3d_geti_or(a, "expert_base", 0), 0, nullptr, 0.0 },
        { "protect", RAD_P_STR, 0, 0, r3d_gets(a, "protect"), 0.0 },
    };
    RadArgs sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.t = t2; sa.n_t = 4; sa.p = sp; sa.n_p = 4;
    sa.rank = a->rank; sa.world_size = a->world_size;
    return r3d_rad_moe_scatter(&sa, s);
}


int r3d_rad_router_sigmoid_topk(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 4) return R3D_NO(RAD_E_INVAL);
    const long long M = r3d_geti_or(a, "M", 0), ne = r3d_geti_or(a, "n_expert", 0),
                    tk = r3d_geti_or(a, "top_k", 0);
    const int norm = (int)r3d_geti_or(a, "norm", 0);
    const double scale_d = r3d_getf(a, "route_scale", 0.0);
    const float scale = (float)scale_d;
    if (M <= 0 || ne <= 0 || tk <= 0 || tk > (long long)kW || tk > ne) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);
    if (scale <= 0.0f) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* L = r3d_opt(a, 0), *B = r3d_opt(a, 1), *I = r3d_opt(a, 2), *W = r3d_opt(a, 3);
    if (!r3d_p(L) || !r3d_p(B) || !r3d_wp(I) || !r3d_wp(W)) return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16 || B->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kW);
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    const float* bp = (const float*)r3d_p(B);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    else
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    return (int)hipGetLastError();
}

}  // extern "C"
