// R4D attention dispatch for the Gemma 4 dense model's GLOBAL layers: head_dim 512, 8 queries per
// KV head, full attention (window 0), paged block 16. This is the one head_dim the library has not
// compiled before -- no h512 kernel existed -- so this file is a real new instantiation, not a
// rename. It is the same device code as the h256/gqa6 and gqa8 dispatches (the kernels are
// templates on head_dim and gqa), with the two changes head_dim 512 forces:
//
//   PREFILL -- TILE 16, not 48. The 16-bit K+V tiles are sK[TILE*(512+8)] + sV16[512*(TILE+8)];
//   at TILE 48 that is 107264 B, past the 64 KiB of LDS the part has, and only TILE 16 clears it
//   (41216 B). The prefill warps tile the QUERY dimension (BLOCK_Q = NWARPS*16/gqa) and loop the
//   KV in TILE chunks, so shrinking the KV tile costs loop iterations, not a different tiling.
//   `O_KWIDE` is off -- its thread-to-(key,channel) map is written for 16 channels, head_dim 256
//   exactly, and at 512 it would read past the tile; the h128 prefill runs the same set minus it,
//   so the non-wide path is already a shipped, correct one.
//
//   DECODE -- the split-KV combine is the HEAD-GENERIC plain form (one thread per channel, a block
//   of head_dim threads), not the vectorised form: that one pins a lane's width to eight or four
//   dwords, and head_dim 512 would give it sixteen and refuse to compile. The plain form is slower
//   but merges off the decode critical path; the trade is a 3b question.
//
// Register pressure at head_dim 512 is the one thing this file does not fix -- a thread carries
// head_dim/16 query fragments plus head_dim/8 accumulator floats, ~384 VGPRs, which the part tops
// out at, so both legs spill. That is correct and a tuning (3b) question, not a 3a one.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK5() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define H5_HEAD_DIM   512
#define H5_GQA          8
#define H5_BLOCK_SIZE  16
#define H5_MAX_DECODE_ROWS 64       // q_len * gqa a single decode workgroup can hold: q_len <= 8
// The h256 prefill option set (3955211) minus O_KWIDE -- the only head-512 option change; the h128
// prefill ships this exact number.
#define H5_P_NWARPS 24
#define H5_P_TILE   16              // the KV tile that fits 64 KiB LDS at head_dim 512 (see above)
#define H5_P_OPT    3430923
#define H5_D_TILE   16
#define H5_D_OPT    3429947         // the shared decode opt set; bit 32 (f16 partials) is set
#define H5_D_JPB    2
#define H5_D_VECMIN 256

// ---- prefill -----------------------------------------------------------------------------------
template <int KVP>
static int h5_prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != H5_HEAD_DIM || a->block_size != H5_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != H5_GQA) return -2;
    constexpr int BQ = (H5_P_NWARPS * 16) / H5_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<H5_P_NWARPS, H5_P_TILE, H5_HEAD_DIM, H5_GQA,
                                                H5_BLOCK_SIZE, KVP, H5_P_OPT>),
                       grid, dim3(H5_P_NWARPS * 32), 0, s, *a);
    CHK5();
    return 0;
}

extern "C" int r4d_attn_prefill_h512_gqa8_fp8kv(const R4DArgs* a, hipStream_t s) {
    return h5_prefill_launch<0>(a, s);
}
extern "C" int r4d_attn_prefill_h512_gqa8_bf16kv(const R4DArgs* a, hipStream_t s) {
    return h5_prefill_launch<1>(a, s);
}

// ---- decode ------------------------------------------------------------------------------------
// The h256 full-attention split law: as many workgroups as fit under 128, at least two tiles a
// segment. It reads kv_heads, num_seqs and max_ctx and nothing about gqa or head_dim, so there is
// nothing to re-derive for this geometry; a window does not exist at full attention.
static int h5_decode_splits(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + H5_D_TILE - 1) / H5_D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    int s = 128 / units;
    const int cap = tiles / 2;
    if (s > cap) s = cap;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

extern "C" long r4d_attn_decode_h512_gqa8_scratch_bytes(const R4DArgs* a) {
    const int  splits = h5_decode_splits(a);
    const long rows   = (long)r4d_q_total(*a) * a->q_heads;
    return (H5_D_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)
                           : rows * splits * (long)(a->head_dim + 2) * 4L;
}

template <int KVP>
static int h5_decode_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != H5_HEAD_DIM || a->block_size != H5_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != H5_GQA) return -2;
    const int rows = a->q_len * H5_GQA;
    if (rows > H5_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = h5_decode_splits(a);

#define H5_LAUNCH(NW)                                                                         \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, H5_D_TILE, H5_HEAD_DIM, H5_GQA,             \
                                                H5_BLOCK_SIZE, KVP, H5_D_OPT>),               \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) H5_LAUNCH(1);
    else if (nw == 2) H5_LAUNCH(2);
    else if (nw == 3) H5_LAUNCH(3);
    else              H5_LAUNCH(4);
#undef H5_LAUNCH
    CHK5();

    // The head-generic plain combine, always: a single thread per channel, one block of head_dim
    // threads, no DPL to trip on at 512. H5_D_OPT carries bit 32 (f16 partials), so it reads them
    // as such; the f32 half is spelled for the same guard the other dispatches use, not expected.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (H5_D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H5_HEAD_DIM, 4, 1>), dim3(jobs3),
                           dim3(H5_HEAD_DIM), splits * sizeof(float), s, *a, splits, H5_D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H5_HEAD_DIM, 4, 0>), dim3(jobs3),
                           dim3(H5_HEAD_DIM), splits * sizeof(float), s, *a, splits, H5_D_TILE);
    CHK5();
    return 0;
}

extern "C" int r4d_attn_decode_h512_gqa8_fp8kv(const R4DArgs* a, hipStream_t s) {
    return h5_decode_launch<0>(a, s);
}
extern "C" int r4d_attn_decode_h512_gqa8_bf16kv(const R4DArgs* a, hipStream_t s) {
    return h5_decode_launch<1>(a, s);
}

// ============================================================== the plugin ABI entry points
// Defined here, in this translation unit, for the same reason the h128 drafter's live in theirs:
// the prefill and decode headers are textual inclusions, and an entry point appended to either
// compiles into nothing, leaving the .so with an undefined symbol only r4d_selftest would catch.

extern "C" int r4d_rad_attn_prefill_h512_gqa8_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, 512);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twins in the h256 dispatch. */
    return r4d_attn_prefill_h512_gqa8_fp8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_prefill_h512_gqa8_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, 512);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h512_gqa8_bf16kv(&r, (hipStream_t)s);
}

/* The split-KV partial buffer, sized at declare from the batch's worst step, the way the other
 * decode geometries do it. The h256 split law makes splits * num_seqs flat, but its clamp to
 * `>= 2 tiles a segment` breaks that at large batches, so the sweep over every count to the bound
 * is still the right size even at full attention. */
extern "C" int64_t r4d_rad_attn_decode_h512_gqa8_scratch(const RadArgs* a) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim   = H5_HEAD_DIM;
        r.q_heads    = (int)n_head;
        r.kv_heads   = (int)(n_head / H5_GQA);
        r.num_seqs   = (int)max_seqs;
        r.q_len      = (int)r4d_geti_or(a, "q_len", 1);
        r.max_ctx    = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits     = sp;
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n) {
            r.num_seqs = (int)n;
            const int64_t need = r4d_attn_decode_h512_gqa8_scratch_bytes(&r);
            if (need > worst) worst = need;
        }
        return worst;
    }

    int rc = fill_paged(a, &r, H5_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    r.splits = sp;
    return r4d_attn_decode_h512_gqa8_scratch_bytes(&r);
}

static int h5_decode_common(const RadArgs* a, R4DArgs* r, int which) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    int rc = fill_paged(a, r, H5_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    r->splits = sp;
    const int64_t need = r4d_attn_decode_h512_gqa8_scratch_bytes(r);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

extern "C" int r4d_rad_attn_decode_h512_gqa8_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = h5_decode_common(a, &r, 0);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h512_gqa8_fp8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_decode_h512_gqa8_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = h5_decode_common(a, &r, 1);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h512_gqa8_bf16kv(&r, (hipStream_t)s);
}
