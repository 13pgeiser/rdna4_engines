// R4D paged attention dispatch for head_dim 256, 6 queries per KV head, paged block 16. The
// KV-dtype variants are template instantiations of one device code. The geometry is a hard gate:
// anything else is rejected.
//
// The query is bf16 in every variant. A single-term 8-bit query is ~10x less accurate against an
// fp32 reference (2.58e-02 vs ~2.3e-03 relative), and the two-term e4m3 form that reaches f16
// accuracy is 4-10% slower end to end. Two families ship, differing in KV dtype: bf16 and fp8
// (the production configuration).
//
// `opt` is a bitmask of optimisation steps:
//     prefill 3954187 = f16 operands | mask skip | folded query scale | contiguous-k fragments
//                     | prefetch 2 + scheduling barrier | dot2 row sum | block table in SGPRs
//                     | wide K fetch | V prefetch across the m-tile loop | LDS-scoped barrier
//     decode  3429947 = the same minus the wide K fetch (its K never reaches LDS), plus K read
//                       straight from the paged cache (bit 16) and f16 split-KV partials (32).
// Each bit is defined where it is used (r4d_attn_prefill_h256_gqa6.hip,
// r4d_attn_decode_h256_gqa6.hip) and can be turned off on its own after a geometry change.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

// The geometry of every kernel in this unit. BLOCK_SIZE is not in the entry-point names because
// it is 16 for all of them.
#define A_HEAD_DIM   256
#define A_GQA          6
#define A_BLOCK_SIZE  16
#define A_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold

extern "C" void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows) {
    if (head_dim)        *head_dim        = A_HEAD_DIM;
    if (gqa)             *gqa             = A_GQA;
    if (block_size)      *block_size      = A_BLOCK_SIZE;
    if (max_decode_rows) *max_decode_rows = A_MAX_DECODE_ROWS;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS at head_dim 256, 16-bit K and V: sK = TILE*264*2, sV = 256*(TILE+8)*2. TILE 64 would need
// 68096 B > 65536, so TILE is 48.
#define P_NWARPS 24
#define P_TILE   48
// K prefetch 2, not 4: four fragments in flight make the m-tile loop spill at six waves per SIMD
// (36-48 bytes per lane, ~4% of the prefill). At 2 the spill is 20 bytes, outside the WMMA groups.
#define P_OPT    3954187

// KVP: 0 = fp8 e4m3 cache, 1 = bf16 cache. Reached only through the two named wrappers below.
template <int KVP, int XOPT = 0>
static int prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / A_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    // R4D_ATTN_FP8: 0 = f16 (default), 1 = O_QK8, 2 = O_PV8, 3 = both; fp8, fp8_v4 and k6_v4 KV.
    int fp8mode = 0;
    if constexpr (KVP == 0) {
        static const int m = [] { const char* e = getenv("R4D_ATTN_FP8"); return e ? atoi(e) : 0; }();
        fp8mode = m;
    }
#define P_LAUNCH(OPTX)                                                                            \
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, \
                                                KVP, OPTX>),                                      \
                       grid, dim3(P_NWARPS * 32), 0, s, *a)
    if constexpr (KVP == 0) {
        if (fp8mode == 3)      P_LAUNCH(P_OPT | XOPT | O_QK8 | O_PV8);
        else if (fp8mode == 2) P_LAUNCH(P_OPT | XOPT | O_PV8);
        else if (fp8mode == 1) P_LAUNCH(P_OPT | XOPT | O_QK8);
        else                   P_LAUNCH(P_OPT | XOPT);
    } else {
        P_LAUNCH(P_OPT | XOPT);
    }
#undef P_LAUNCH
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<1>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_fp8v4kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0, O_V4>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_k6v4kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0, O_V4 | O_K6>(a, s);
}

// ---- decode ----------------------------------------------------------------------------------
#define D_TILE 16
// Decode prefetches two K fragments per WMMA group, not four. Four costs 8 more VGPRs at the peak
// of the QK loop and spills 18, reloading part of the query from scratch on every tile. Two fits in
// 255 VGPRs and is 1.7-7.5% faster, with identical output.
#define D_OPT  3429947
// Output rows (waves) per block in the vectorised merge. 1, 2 and 4 are within 1%; 2 is best on
// average.
#define D_JPB 2
// Minimum output rows for the vectorised merge. It gives a row 32 lanes instead of 256, trading
// parallelism for 8x fewer loads, which pays only when there are enough rows to fill the GPU. At
// one request (96 rows, most segments) it loses up to 23%; the crossover is ~192 rows and above
// 288 it wins by 1.2-20%. The cut is four rows per CU.
#define D_VECMIN 256

// Split law: r4d_decode_splits (r4d_attn_decode_h256_gqa6.hip).
extern "C" long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits(a, D_TILE), D_OPT);
}

template <int KVP, int XOPT = 0>
static int decode_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    const int rows = a->q_len * A_GQA;
    if (rows > A_MAX_DECODE_ROWS) return -4;    // verify width <= 10 at gqa 6
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = r4d_decode_splits(a, D_TILE);

#define D_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, D_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,    \
                                               KVP, D_OPT | XOPT>),                           \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    // The vectorised merge's wave sum reproduces the general LDS tree only while splits <= 64.
    // Above that (only when kv_heads * num_seqs is 1, giving 128 splits) use the general form.
    // Both produce identical bytes; the choice is speed only.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= D_VECMIN) {
        const int blocks = (jobs3 + D_JPB - 1) / D_JPB;
        if (D_OPT & 32)
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<A_HEAD_DIM, 4, 1, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
        else
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<A_HEAD_DIM, 4, 0, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
    } else if (D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 1>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 0>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_decode_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<1>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_fp8v4kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0, O_V4>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_k6v4kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0, O_V4 | O_K6>(a, s);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader is r4d_attn_args.h, shared by every paged geometry.
//
// Note: entry points belong in this translation unit, not in the r4d_attn_prefill_*.hip or
// r4d_attn_decode_*.hip files it includes. An entry point added there does not get compiled, and
// the .so carries an undefined symbol that only r4d_selftest catches.
/* ------------------------------------------------------------------ paged prefill */

extern "C" int r4d_rad_attn_prefill_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0, not an error: the kernels read them that way, vLLM defaults both
     * scales to 1.0, and kv_store_fp8 at its default scale needs no descale plane. The operands
     * are optional in the schema and the kernel, so they are optional here too. A calibrated pair
     * is used when passed. */
    return r4d_attn_prefill_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_prefill_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

/* ------------------------------------------------------------------ paged decode */

/* Split-KV partial buffer: written by the decode kernel, read by the combine, nothing else. Arena
 * scratch sized at declare from the band's upper bound (RadScratchFn).
 *
 * Declare passes parameters and no tensors (the batch is not known yet), so parameters are the
 * primary source and tensors the fallback. Reading tensors first would refuse every attn_paged
 * row with "invalid argument". q_len is the band's upper bound, since the region is sized once
 * for the worst step.
 *
 * The size is the maximum over every sequence count up to max_seqs: S * splits(S) is not monotone
 * in S (at kv_heads 2 the h256 law gives 60 at twelve sequences and 63 at nine, and the h128 law's
 * step-up doubles splits at one count only). The laws are closed forms, so the sweep is cheap. */
// Per-geometry byte counts over the shared partial layout, so one sweep serves all.
// which: 0 = h256/gqa6, 1 = h128/gqa4, 2 = h256/gqa2 (Gemma 4 windowed).
static int64_t decode_scratch_bytes_of(const R4DArgs* r, int which) {
    if (which == 1) return (int64_t)r4d_attn_decode_h128_gqa4_scratch_bytes(r);
    if (which == 2) return (int64_t)r4d_attn_decode_h256_gqa2_scratch_bytes(r);
    if (which == 3) return (int64_t)r4d_attn_decode_h128_gqa2w_scratch_bytes(r);
    return (int64_t)r4d_attn_decode_h256_gqa6_scratch_bytes(r);
}

/* `row`: a token's bytes in a (block, head) page (fill_paged_row), 0 for 2 * head_dim; `rep`: the
 * launch maps each kv head onto `rep` query groups (R4DArgs::kv_rep), so the split law counts
 * groups, gqa / rep of them a kv head; `q_pitched`: q may be a slice (attn_paged_qrow). */
static int64_t decode_scratch(const RadArgs* a, int head_dim, int which, long long row = 0,
                              int rep = 1, bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim  = head_dim;
        r.q_heads   = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads  = (int)(gqa > 0 ? n_head / gqa : n_head) * rep;
        r.kv_rep    = rep;
        r.num_seqs  = (int)max_seqs;
        r.q_len     = (int)r4d_geti_or(a, "q_len", 1);
        r.max_ctx   = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits    = sp;
        /* max_ctx feeds only the split law. Zero would pick one split and undersize the buffer
         * for a launch that later picks more, so refuse. */
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n) {
            r.num_seqs = (int)n;
            const int64_t need = decode_scratch_bytes_of(&r, which);
            if (need > worst) worst = need;
        }
        return worst;
    }

    int rc = fill_paged_row(a, &r, head_dim, row > 0 ? row : 2LL * head_dim, q_pitched);
    if (rc != RAD_OK) return rc;
    r.kv_heads *= rep;
    r.kv_rep = rep;
    r.splits = sp;
    return decode_scratch_bytes_of(&r, which);
}

extern "C" int64_t r4d_rad_attn_decode_h256_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0);
}

extern "C" int64_t r4d_rad_attn_decode_h128_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1);
}

static int decode_common(const RadArgs* a, R4DArgs* r, int head_dim, int which, long long row = 0,
                         int rep = 1, bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    int rc = fill_paged_row(a, r, head_dim, row > 0 ? row : 2LL * head_dim, q_pitched);
    if (rc != RAD_OK) return rc;
    /* a QLD instance reads q_ld whatever q is: dense is the pitch q_heads * head_dim */
    if (q_pitched && r->q_ld == 0) r->q_ld = (long)r->q_heads * head_dim;
    /* each cache head serves `rep` query groups: the kernel's grid and its gqa check count groups */
    r->kv_heads *= rep;
    r->kv_rep = rep;
    r->splits = sp;
    const int64_t need = decode_scratch_bytes_of(r, which);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

extern "C" int r4d_rad_attn_decode_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

/* The fp8_v4 cache: a token's row is 25 * 256 / 16 = 400 bytes (E4M3 K, int4 V, f16 scales). */
extern "C" int r4d_rad_attn_prefill_h256_fp8v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged_row(a, &r, 256, 400);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa6_fp8v4kv(&r, (hipStream_t)s);
}

extern "C" int64_t r4d_rad_attn_decode_h256_v4_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0, 400);
}

extern "C" int r4d_rad_attn_decode_h256_fp8v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0, 400);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_fp8v4kv(&r, (hipStream_t)s);
}

/* The k6_v4 cache: a token's row is 22 * 256 / 16 = 352 bytes (int6 K, int4 V, f16 scales). */
extern "C" int r4d_rad_attn_prefill_h256_k6v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged_row(a, &r, 256, 352);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h256_gqa6_k6v4kv(&r, (hipStream_t)s);
}

extern "C" int64_t r4d_rad_attn_decode_h256_k6_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0, 352);
}

extern "C" int r4d_rad_attn_decode_h256_k6v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0, 352);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_k6v4kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h128_gqa4_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_bf16kv(&r, (hipStream_t)s);
}

/* Gemma 4 dense sliding window (head_dim 256, gqa 2, window 1024) decode; its prefill entry points
 * are in r4d_attn_paged_h256_gqa2.hip. No check_dims_once: this is not the geometry that reports
 * dims. */
extern "C" int64_t r4d_rad_attn_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2);
}

extern "C" int r4d_rad_attn_decode_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twin above. */
    return r4d_attn_decode_h256_gqa2_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}

/* Gemma 3 sliding window (head_dim 128, gqa 2, window 1024) decode; its prefill entry points
 * are in r4d_attn_paged_h128_gqa2w.hip. */
extern "C" int64_t r4d_rad_attn_decode_h128_gqa2w_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 3);
}

extern "C" int r4d_rad_attn_decode_h128_gqa2w_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 3);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa2w_fp8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_gqa2w_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 3);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa2w_bf16kv(&r, (hipStream_t)s);
}

/* The same decode under attn_paged_qrow: q read at its own row pitch (R4DArgs::q_ld), the sliding
 * layers' q where Gemma 4 31B's fused q|k|v projection left it. Same kernel, same arithmetic. */
extern "C" int64_t r4d_rad_attn_qrow_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2, 0, 1, true);
}
extern "C" int r4d_rad_attn_qrow_decode_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 0, 1, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_fp8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_qrow_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 0, 1, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_bf16kv(&r, (hipStream_t)s);
}

/* Head_dim 128 at 8 queries per kv head, any window, causal or not, q_len up to 16, under
 * attn_paged_qrow: the gqa-4 kernel with each kv head mapped onto two query groups
 * (R4DArgs::kv_rep = 2), the 31B's DFlash drafter's attention over a cache that holds each kv head
 * once. The workgroup holds q_len * 4 rows, so a block of 16 fits where gqa 8 (128 rows) does not;
 * the two groups of a kv head read its blocks twice, as the copies they replace did. */
extern "C" int64_t r4d_rad_attn_qrow_h128_gqa8_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1, 0, 2, true);
}
extern "C" int r4d_rad_attn_qrow_h128_gqa8_fp8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1, 0, 2, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_rep_fp8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_qrow_h128_gqa8_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1, 0, 2, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_rep_bf16kv(&r, (hipStream_t)s);
}
