// R4D attention dispatch for the Gemma 4 dense model's sliding-window geometry: head_dim 256,
// 2 queries per KV head, paged block 16, and a SLIDING WINDOW (window 1024 in the model, 50 of its
// 60 layers). It is the same device code as the 27B target's h256/gqa6 dispatch -- the kernels are
// templates on head_dim and gqa, so this file is another set of instantiations, not a second
// kernel -- with the window bit (template arg WIN) turned on, the way the DFlash2 drafter leg does
// it at head_dim 128.
//
// Both bands, now. The decode band (q_len * gqa <= 64, i.e. q_len <= 32 at gqa 2) is the
// windowed decode kernel above. The prefill band (a query long enough to want the query-tiled
// prefill kernel) is the windowed prefill kernel below -- the same r4d_attn_prefill_kernel the
// h256/gqa6 and h512/gqa8 dispatches instantiate, with the window bit (template arg WIN) turned on,
// the way the decode leg does. A trunk layer prefills, so it needs the prefill band on the device:
// libref is an opt-in fallback the engine refuses by default (spec §17). The drafter leg is
// decode-only because a drafter never prefills.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode and prefill kernels share their query conversion and fragment helpers, so the prefill
// header comes first here for the same reason it does in the h256 dispatch. This translation unit
// instantiates BOTH r4d_attn_prefill_kernel (below, WIN=1) and the windowed decode kernel.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   256
#define B_GQA          2
#define B_BLOCK_SIZE  16
#define B_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold
#define B_TILE 16
// The same option set the h256 decode kernel ships, K prefetch of 2 included. At head_dim 256 it
// is the same register picture gqa6 has (256 VGPRs with a small spill), so this is a correctness
// instantiation here -- the register count is a 3b tuning question, not a 3a one. The window bit
// is already in the template; this file only turns it on.
#define B_OPT  3429947
#ifndef B_JPB
#define B_JPB 2
#endif
#ifndef B_VECMIN
#define B_VECMIN 256
#endif

// Split law, the drafter leg's with the same window-aware term. It aims at 192 workgroups, snaps
// DOWN to a power of two but never below 16, clamps to at least two tiles a segment, and then steps
// UP off the exact grid size `splits * units == 256`, a pothole in this geometry's timing.
//
// THE WINDOW-AWARE TERM is the one thing it does that the h256 full-attention law does not: under a
// sliding window the context the kernel actually walks is the window, not the sequence, so the tile
// count that feeds the machine-fill target and the thin-segment clamp comes from the window.
// Without that the law reads a 16K sequence and asks for segments two tiles deep off a 1024-token
// walk.
static int decode_splits_h256_gqa2(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int span  = (a->window > 0 && a->window + a->q_len < a->max_ctx)
                    ? a->window + a->q_len : a->max_ctx;
    const int tiles = (span + B_TILE - 1) / B_TILE;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    if (s * units == 256 && s * 2 <= tiles) s <<= 1;   // step off the pothole, see above
    return s;
}

extern "C" long r4d_attn_decode_h256_gqa2_scratch_bytes(const R4DArgs* a) {
    const int  splits = decode_splits_h256_gqa2(a);
    const long rows   = (long)r4d_q_total(*a) * a->q_heads;
    return (B_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)
                        : rows * splits * (long)(a->head_dim + 2) * 4L;
}

template <int KVP>
static int decode_launch_h256_gqa2(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    const int rows = a->q_len * B_GQA;
    if (rows > B_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = decode_splits_h256_gqa2(a);

#define B_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, B_TILE, B_HEAD_DIM, B_GQA, B_BLOCK_SIZE,    \
                                               KVP, B_OPT, 1>),                               \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) B_LAUNCH(1);
    else if (nw == 2) B_LAUNCH(2);
    else if (nw == 3) B_LAUNCH(3);
    else              B_LAUNCH(4);
#undef B_LAUNCH
    CHK4();

    // Same two merge forms, and the same rule for choosing between them, as the h256 dispatch:
    // a wave a row while there are enough rows to fill the part and the segment count fits a
    // wave, the general LDS-tree form otherwise. Both produce the same bytes, and both carry the
    // window bit here (the final template argument) for the same reason the decode kernel does.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= B_VECMIN) {
        const int blocks = (jobs3 + B_JPB - 1) / B_JPB;
        if (B_OPT & 32)
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<B_HEAD_DIM, 4, 1, B_JPB, 1>),
                               dim3(blocks), dim3(32 * B_JPB), 0, s, *a, splits, B_TILE);
        else
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<B_HEAD_DIM, 4, 0, B_JPB, 1>),
                               dim3(blocks), dim3(32 * B_JPB), 0, s, *a, splits, B_TILE);
    } else if (B_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<B_HEAD_DIM, 4, 1, 1>), dim3(jobs3),
                           dim3(B_HEAD_DIM), splits * sizeof(float), s, *a, splits, B_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<B_HEAD_DIM, 4, 0, 1>), dim3(jobs3),
                           dim3(B_HEAD_DIM), splits * sizeof(float), s, *a, splits, B_TILE);
    CHK4();
    return 0;
}

extern "C" int r4d_attn_decode_h256_gqa2_fp8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch_h256_gqa2<0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch_h256_gqa2<1>(a, s);
}

// ---- prefill ---------------------------------------------------------------------------------
// The h256/gqa6 prefill option set (P_OPT 3955211) is reused unchanged: at head_dim 256 the wide K
// fetch (O_KWIDE) is valid -- it is written for 16 channels = head_dim 256 exactly -- so unlike the
// h512 leg (which turns it off because 512 reads past the tile) this one keeps it. gqa 2 makes
// BLOCK_Q (NWARPS*16/2 = 192) larger than gqa 6's 64, so a workgroup covers more query positions:
// fewer grid.x blocks, the same KV tile. WIN is on (the last template arg); window <= 0 folds it
// away to the full-attention form, so a windowless launch is safe.
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

/* The prefill RadArgs wrappers, defined here (not in the h256 dispatch) for the same reason the
 * h512 leg's live in theirs: an entry point appended to a textual inclusion compiles into nothing.
 * Absent descales mean 1.0 (fill_paged leaves them null and the kernel reads 1.0). */
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
