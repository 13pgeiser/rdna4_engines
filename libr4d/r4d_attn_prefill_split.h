// The prefill KV split, host side: the law, the scratch it declares and the merge launch, shared by
// every head-256 query-tiled prefill launcher (the h256/gqa6 unit, r4d_attn_paged_gqa8.hip's ratios
// 4, 6 over 4-token pages, 8 and 12, and the h256/gqa2 unit). The kernel side is O_PSPLIT in
// r4d_attn_prefill_h256_gqa6.hip. Include after r4d_attn_decode_h256_gqa6.hip, whose partial
// layout and merges it reuses. No entry points here: each unit defines its own, with its geometry.
//
// Why. The query-tiled grid is ceil(q_len / BQ) x kv_heads x num_seqs, one 768-thread workgroup a
// WGP (BQ = NWARPS * 16 / gqa query positions: 192 at gqa 2, 96 at 4, 64 at 6, 48 at 8, 32 at 12),
// and each workgroup walks its whole causal range serially. So below one workgroup per WGP a launch
// costs one full walk of the context whatever its row count: Qwen3.8-27B at TP2 (gqa 6, kv_heads 2)
// runs a 53-row tail over 188k keys as 2 workgroups on 32 WGPs, ~17.7 ms a layer, against ~21 ms a
// workgroup for a 1024-row piece filling all 32 (profiled tails, attention: 63 ms at 41k / 80 rows,
// 152 at 97k / 345, 283 at 188k / 53). The geometries with one KV head a rank are worse off:
// Flash-Next's gqa 12 at TP2 runs 53 rows as 2 workgroups as well, and any chunk up to 32 rows as
// one. Such launches split the KV and finish in the split-KV merge.
#pragma once

#include <cstdlib>

// Workgroups the card runs at once: one a WGP. HIP's multiProcessorCount counts WGPs on RDNA (32 on
// the R9700, see want_wg in r4d_hc_bf16.hip); 32 if it cannot be read.
static inline int r4d_prefill_wgps() {
    static const int v = [] {
        int dev = 0;
        hipDeviceProp_t pr{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&pr, dev) != hipSuccess)
            return 32;
        return pr.multiProcessorCount > 0 ? pr.multiProcessorCount : 32;
    }();
    return v;
}

// The split law, at the launch's own geometry: `bq` query positions a workgroup, `tile` keys a KV
// tile. `units` counts the workgroups that do work: a ragged batch's grid is sized from its longest
// sequence (q_len), but a shorter sequence's query tiles past its end exit at once, so the count is
// bounded by sum ceil(q_s / bq) <= (total_q + (bq - 1) n) / bq as well (exact for one sequence), so a
// ragged launch's longest sequence does not make its others look like full tiles. One S applies to
// every sequence of the launch; a mixed step's decode rows reach the split-KV decode row instead, by
// the arch's per-band issue (AttnGatedFP8::step). A launch over part of a step (q and out whole, cu
// past row 0) counts the step's rows in total_q, so the bound is loose there and may keep it
// unsplit.
// At `wgps` units or more the card is full: no split. Otherwise S rounds of equal workgroups cost
// rounds / S walks, so take one round (S1 = wgps / units) or two (S2 = 2 wgps / units), whichever
// is faster, the smaller S on a tie (fewer partials). At gqa 6 with 2 KV heads: 53 rows (2 units)
// -> 16 splits, 80 -> 8, 345 (12 units) -> 5 (60 workgroups, 0.4 walks against 0.5 at 2), 576 -> 3,
// 1024 and up -> 1. At gqa 12 with 1 KV head: up to 32 rows (1 unit) -> 32, 53 -> 16, 345 (11) -> 5.
// Each split keeps at least 8 tiles of max_ctx, the only context the host sees (the engine passes
// the model's bound, so in serving this caps nothing; a short context's splits past its tiles exit
// at once). At most 64, the vectorised merge's limit.
static inline int r4d_prefill_splits_law(int bq, int tile, int q_len, int total_q, int num_seqs,
                                         int kv_heads, int max_ctx, int wgps) {
    const long n  = std::max(1, num_seqs);
    const long qt = std::min(n * ((std::max(1, q_len) + bq - 1) / bq),
                             (std::max(1, total_q) + (long)(bq - 1) * n) / bq);
    const long units = (long)std::max(1, kv_heads) * std::max(1L, qt);
    if (units >= wgps) return 1;
    const int s1 = (int)(wgps / units), s2 = (int)(2L * wgps / units);
    int sp = s2 > 2 * s1 ? s2 : s1;
    const int cap = ((max_ctx > 0 ? max_ctx : (1 << 30) - 1) + tile - 1) / tile / 8;
    sp = std::min(sp, std::max(1, cap));
    return std::max(1, std::min(sp, 64));
}

// The split count the last prefill launch of this unit used, 1 unsplit. Read by each unit's test
// hook (r4d_selftest), which tells a split launch from an unsplit one whose bytes happen to agree.
// Not atomic.
static int g_prefill_last_splits = 0;

/* The KV split's count before the scratch bound: the case's `splits` parameter when it names one
 * (r4d_selftest; the prefill rows declare no such axis, so the engine passes none), else
 * R4D_PREFILL_SPLITS, else the law. R4D_PREFILL_SPLITS is a tuning and A/B knob, not
 * configuration: 0 or unset = the law, 1 = never split, N = N splits for every launch (at most 64,
 * and at most what the scratch holds, see r4d_prefill_splits). Read once. */
static inline int r4d_prefill_splits_want(const RadArgs* a, const R4DArgs* r, int bq, int tile) {
    const int sp = split_of(a);
    if (sp > 0) return std::min(sp, 64);
    static const int env = [] {
        const char* e = getenv("R4D_PREFILL_SPLITS");
        return e && *e ? atoi(e) : 0;
    }();
    if (env > 0) return std::min(env, 64);
    return r4d_prefill_splits_law(bq, tile, r->q_len, r4d_q_total(*r), r->num_seqs, r->kv_heads,
                                  r->max_ctx, r4d_prefill_wgps());
}

/* The launch's split count, 1 for the plain form: what r4d_prefill_splits_want asks, cut to the
 * segments the caller's scratch holds, so a forced count past the declared worst case runs with as
 * many as fit rather than writing past the region. No scratch (a row without the hook, a direct
 * caller of the r4d_attn_prefill_* entry points) is the plain form. */
static inline int r4d_prefill_splits(const RadArgs* a, const R4DArgs* r, int bq, int tile) {
    if (!a->scratch || a->scratch_bytes <= 0) return 1;
    int sp = r4d_prefill_splits_want(a, r, bq, tile);
    const long per = r4d_decode_scratch(r, 1, 0);          /* one segment of every row, f32 */
    if (sp > 1 && per > 0 && a->scratch_bytes / per < sp) sp = (int)(a->scratch_bytes / per);
    return std::max(1, sp);
}

/* Bytes of the split's f32 partials, at declare the most any launch in the band can take.
 *
 * Declare passes the band's upper bound as q_len and no tensors, so the worst case is swept over
 * every sequence count up to max_seqs and every row count up to the band, in both the uniform form
 * (q_len = rows / n) and the ragged one (q_len = min(band, rows), as fill_paged takes it). Only
 * launches with fewer than one workgroup a WGP split, and the units of a launch of `t` rows are at
 * least kv_heads * ceil(t / bq), so the sweep stops there. The law keeps units * S <= 2 wgps, so
 * rows * S * q_heads <= bq * gqa * 2 wgps = NWARPS * 16 * 2 wgps, the same for every ratio at 24
 * waves: 24576 (row, head) pairs of 1032 bytes at head_dim 256, 25.4 MB, for any kv_heads. Not
 * monotone in the rows, hence the sweep. A launch over part of a step's rows counts the whole
 * step's in its partial layout (total_q), more than this covers when it is long; r4d_prefill_splits
 * then cuts S to what fits, down to the plain form.
 *
 * Never an error: a refusal would unseat the prefill row, which ran without scratch before the
 * split. A shape it cannot read gets none, and its launches run the plain form. */
static inline int64_t r4d_prefill_scratch(const RadArgs* a, int head_dim, long long row, int bq,
                                          int tile) {
    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        const int kv   = (int)std::max(1LL, gqa > 0 ? n_head / gqa : n_head);
        const int band = (int)std::max(1LL, r4d_geti_or(a, "q_len", 1));
        const int mctx = (int)r4d_geti_or(a, "max_ctx", 0);
        const int w    = r4d_prefill_wgps();
        const int64_t per_row = (int64_t)n_head * (head_dim + 2) * 4;
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n)
            for (long long t = n; t <= band; ++t) {
                if ((long long)kv * ((t + bq - 1) / bq) >= w) break;
                int sp = r4d_prefill_splits_law(bq, tile, (int)std::min<long long>(band, t), (int)t,
                                                (int)n, kv, mctx, w);
                if (t % n == 0)
                    sp = std::max(sp, r4d_prefill_splits_law(bq, tile, (int)(t / n), (int)t, (int)n,
                                                              kv, mctx, w));
                if (sp > 1) worst = std::max(worst, (int64_t)t * sp * per_row);
            }
        return worst;
    }
    if (fill_paged_row(a, &r, head_dim, row) != RAD_OK) return 0;
    const int sp = r4d_prefill_splits_want(a, &r, bq, tile);
    return sp > 1 ? (int64_t)r4d_decode_scratch(&r, sp, 0) : 0;
}

// The split's merge: the decode merge over f32 partials (PF16 0), at the prefill tile, so it
// recomputes the kernel's partition; one wave a row from VECMIN rows (JPB rows a block), else the
// LDS-tree form, on the decode launchers' terms. SPAN: the launch may cover part of a step's rows
// (ArPieces' pieces, a band-split step's prefill half: cu_q starts past row 0, or ends before
// total_q), and the rest are another launch's.
template <int HEAD_DIM, int JPB, int VECMIN>
static inline int r4d_prefill_combine(const R4DArgs* a, int splits, int tile, hipStream_t s) {
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= VECMIN)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, 0, JPB, 0, 1>),
                           dim3((jobs3 + JPB - 1) / JPB), dim3(32 * JPB), 0, s, *a, splits, tile);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<HEAD_DIM, 4, 0, 0, 1>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, tile);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}
