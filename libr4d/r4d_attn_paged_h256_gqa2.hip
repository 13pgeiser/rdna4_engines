// R4D attention dispatch for Gemma 4 dense's sliding-window layers: head_dim 256, 2 queries per KV
// head, paged block 16, window 1024 (50 of 60 layers). Instantiations of the h256/gqa6 kernel
// templates with WIN = 1, as the DFlash2 drafter leg does at head_dim 128.
//
// Both bands: decode (q_len * gqa <= 64, so q_len <= 32) and the query-tiled prefill kernel. Unlike
// the drafter leg, a trunk layer prefills, and the libref fallback is opt-in and refused by
// default (spec §17).
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel uses helpers from the prefill unit, so prefill is included first. Both kernels
// are instantiated here with WIN = 1.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   256
#define B_GQA          2
#define B_BLOCK_SIZE  16
#define B_TILE 16                   // decode: r4d_decode_launch_win (r4d_attn_decode_h256_gqa6.hip)
// The h256 decode option set, K prefetch 2 included. Same register use as gqa6 (256 VGPRs, small
// spill); not separately tuned for this geometry.
#define B_OPT  3429947

extern "C" long r4d_attn_decode_h256_gqa2_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits_win(a, B_TILE), B_OPT);
}

extern "C" int r4d_attn_decode_h256_gqa2_fp8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 1>(a, s);
}
// The same with q at R4DArgs::q_ld (O_QLD), for attn_paged_qrow; its own instances, so the dense
// ones above keep their registers.
extern "C" int r4d_attn_decode_h256_gqa2_qld_fp8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT | O_QLD, 0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_qld_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT | O_QLD, 1>(a, s);
}

// ---- prefill ---------------------------------------------------------------------------------
// The h256/gqa6 prefill option set but at K prefetch 4 (P_OPT there is 3954187); O_KWIDE is valid
// at head_dim 256. At gqa 2 a block covers NWARPS * 16 / 2 = 192 query positions. WIN is on;
// window <= 0 gives full attention.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211

template <int KVP>
static int prefill_launch_h256_gqa2(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / B_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                B_BLOCK_SIZE, KVP, P_OPT, 1>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK4();
    return 0;
}

extern "C" int r4d_attn_prefill_h256_gqa2_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<0>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<1>(a, s);
}

/* Prefill RadArgs wrappers, defined in this unit rather than the h256 dispatch. Absent descales
 * mean 1.0 (fill_paged leaves them null). */
extern "C" int r4d_rad_attn_prefill_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa2_fp8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_prefill_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}
