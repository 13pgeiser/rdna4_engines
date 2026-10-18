/* metrics.h -- the scheduler's snapshot for the server's /metrics endpoint (spec §14).
 *
 * Plain numbers only; each exporter (Prometheus, the live view) does its own formatting. Fixed
 * arrays, so a scrape allocates nothing and never contends with the step path for the allocator.
 */
#pragma once
#include "rad_core.h"

namespace rad {

enum { RAD_SCHED_MAX_KV_GROUPS = 16 };

struct SchedMetrics {
    int64_t step = 0;

    /* Queue state, instantaneous. */
    int64_t queue_depth = 0;         /* requests waiting, including preempted ones */
    int64_t running = 0;
    int64_t preempted_waiting = 0;   /* of queue_depth, how many are awaiting recompute */

    /* Cumulative counters. Monotonic, so a scraper can rate() them. */
    int64_t admitted = 0;
    int64_t finished = 0;
    int64_t failed = 0;
    int64_t cancelled = 0;
    int64_t preemptions = 0;
    int64_t steps = 0;

    /* Steps by shape. `steps_mixed` counts steps carrying both a prefill chunk and decode rows.
     * `decode_rows_unspeculated` counts every decode row that took no draft, for any reason
     * (depth 0, a request that may not speculate, a drafter with nothing to propose). */
    int64_t steps_prefill = 0;
    int64_t steps_mixed = 0;
    int64_t steps_decode = 0;
    /* --prefill-interleave: decode-only steps the hold inserted, issued ahead or not; each kept a
     * ready chunk out. */
    int64_t steps_held = 0;
    int64_t decode_rows_unspeculated = 0;
    /* Steps whose forward pass was enqueued behind the previous step before it committed
     * (Scheduler::step_next). Every other step leaves the card idle during the host's commit and
     * staging. */
    int64_t steps_ahead = 0;
    /* Prefill chunks capped to leave room for decode rows or a waiting admission in the same
     * step (--busy-prefill-chunk). Zero under `off`. */
    int64_t prefill_chunks_capped = 0;

    /* Cumulative nanoseconds spent inside steps, total and for decode steps only. Wall time over
     * a step count is not a step time: it charges idle gaps between requests to steps and mixes
     * prefill with decode. `step_ns_decode / steps_decode` is the decode step time. Measured from
     * `step_start_`, the clock `decode_tps` uses, so no extra synchronisation (spec §16). */
    int64_t step_ns = 0;
    int64_t step_ns_decode = 0;

    /* The single-sequence decode floor, nanoseconds; 0 until a window of eligible steps fills.
     * A min over WINDOW MEANS of single-sequence decode steps (see DecodeFloor), not over raw
     * brackets -- a raw min latches a degenerate bracket forever. Single-sequence on purpose: a
     * decode step at eight sequences is a different measurement, so keying the floor to one row
     * keeps a batch arriving from being misread as drift. Scheduler::step_finished feeds DecodeFloor. */
    int64_t step_ns_decode_best = 0;

    /* The most recent completed window mean of single-sequence decode steps, nanoseconds; 0 until
     * the first gated window completes. The advisory's OTHER side: best() is how fast the engine
     * has ever been, recent() is how fast it is now, and both are window means of the same stream
     * at the same size -- so they cannot disagree by order statistics the way the engine's floor
     * disagreed with the browser's per-poll min (7.7% phantom drift on a flat lane, 2026-10-18).
     * Scheduler::step_finished records it. */
    int64_t step_ns_decode_recent = 0;

    /* Per KV group. `name` points into the Program, which outlives any scrape. */
    int n_kv_groups = 0;
    struct Group {
        const char* name = "";
        int64_t blocks_total = 0;
        int64_t blocks_free = 0;
        float   utilisation = 0.0f;   /* 1 - free/total */
        /* Tokens one block covers, 0 for a stateful group. */
        int64_t block_tokens = 0;
        /* A windowed group recycles blocks below the window, so its size bounds how far back a
         * sequence sees, not how much context fits. */
        bool    window = false;
        /* The group in bytes, the only unit that sums across groups: a stateful group has no
         * token axis, and paged groups share one, so summed tokens overcount. */
        int64_t bytes_total = 0;
        int64_t bytes_used = 0;
        /* Bytes the group was carved to address. bytes_total is what the card holds now; the
         * difference is the VRAM the elastic pool released behind unheld blocks. */
        int64_t bytes_carved = 0;
    } kv[RAD_SCHED_MAX_KV_GROUPS];

    /* Prefix caching (§7.3). Attention and linear are reported separately because they hit at
     * different granularities. */
    int64_t prompt_tokens = 0;
    int64_t attn_cached_tokens = 0;
    int64_t linear_cached_tokens = 0;
    float   attn_hit_rate = 0.0f;
    /* Cumulative, from the prefix cache. Rising evictions under a steady workload mean the pool
     * cannot hold the working set. */
    int64_t prefix_evictions = 0;
    int64_t prefix_ckpt_evictions = 0;
    /* prefix_ckpt_evictions by reason (retention policy, least recently used of a full store),
     * and reservations that went without a snapshot. */
    int64_t prefix_ckpt_retired = 0, prefix_ckpt_lru = 0, prefix_ckpt_refused = 0;
    int64_t prefix_expired = 0, prefix_ckpt_expired = 0;   /* --prefix-cache-ttl */
    /* Admissions whose attention hit was cut to nothing: no snapshot at or below it (recurrent
     * state or a window snapshot), a windowed trunk group no window snapshot covers, or blocks
     * that could not be adopted. `hit_cut_tokens` is the checkpoint shortfall: prompt tokens the
     * attention cache held and the admission computed again, because the hit resumed at a
     * snapshot below it or was cut to nothing. */
    int64_t hit_cut_no_snapshot = 0, hit_cut_window = 0, hit_cut_adopt = 0;
    int64_t hit_cut_tokens = 0;
    /* Tokens of context held only by the cache, i.e. what eviction would free. Excludes blocks
     * shared with a running sequence. Same token axis as the paged groups, so comparable with
     * kv_tokens_used. */
    int64_t prefix_held_tokens = 0;
    float   linear_hit_rate = 0.0f;
    int64_t checkpoints_written = 0;

    /* Throughput: tokens over wall time in a 1-second decaying window (Scheduler::step_finished);
     * the counters beside them are exact. */
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    float   prefill_tps = 0.0f;
    float   decode_tps = 0.0f;

    /* Speculation (§10). */
    int64_t draft_tokens = 0;
    int64_t draft_accepted = 0;
    /* Prefix hits ended early so a windowed block drafter has context (Scheduler::draft_refill_),
     * and the reusable tokens they gave back to be recomputed. */
    int64_t draft_refills = 0;
    int64_t draft_refill_tokens = 0;
    float   draft_acceptance = 0.0f;
    int     draft_depth = 0;          /* what the controller would choose right now */
    /* Per draft position: `reached` counts verifies where every earlier position was accepted,
     * `accepted` how often this one was then taken. Their ratio is the conditional acceptance,
     * which says what a deeper draft would buy. `draft_depth_seen` bounds the valid entries. */
    enum { RAD_METRICS_MAX_SPEC = 64 };
    int     draft_depth_seen = 0;
    int64_t draft_pos_reached [RAD_METRICS_MAX_SPEC] = {};
    int64_t draft_pos_accepted[RAD_METRICS_MAX_SPEC] = {};
    /* Adaptive window (--draft-window adaptive, draft.h); all zero while off. The gauge is the
     * mean window over running sequences; the cumulative histogram counts verifies by window,
     * index = drafts verified. */
    bool    draft_adaptive = false;
    float   draft_depth_adaptive = 0.0f;
    int64_t draft_verify_depth[RAD_METRICS_MAX_SPEC + 1] = {};
};


/* Estimator for the single-sequence decode floor (SchedMetrics::step_ns_decode_best).
 *
 * A min over RAW step brackets is unbounded below: a decode step issued ahead behind a step with
 * long host work (a prefill chunk, or a decode step that saves a KV checkpoint) has its forward
 * pass finished before its own bracket opens, so the bracket holds only readback and commit --
 * about a millisecond, no GPU work. A min latches that degenerate on the first occurrence and
 * never recovers. Measured live 2026-10-18: floor 1.34 ms against a 15.70 ms mean, after which
 * the dashboard's restart advice read "heavily drifted, pays back fast" on a healthy engine --
 * a systematically wrong advisory, worse than none, because it recommends a restart that cannot
 * help.
 *
 * The fix mirrors the construction the dashboard already proves clean on this lane: a min over
 * WINDOW MEANS, not over raw samples. Each full window of single-sequence decode steps
 * contributes one mean; a degenerate is diluted by the honest steps sharing its window, so the
 * floor tracks the real floor. Two biases on that floor partially cancel: a window MEAN sits
 * above the true floor (it averages, so it cannot reach a lone fast step), while min-over-windows
 * picks the window with the most degenerates and pulls back down. A simulation of the live lane
 * (real ~15.9 ms, true floor ~14.4 ms, degenerate ~1.34 ms) puts the estimate at 14.3 ms median
 * at the observed ~1.4% degenerate rate, worst case ~13.6 ms -- within a few percent of the true
 * floor, and never the 12x latch of the raw min. The residual can sit slightly BELOW the true
 * floor (a degenerate-heavy window), which mildly OVER-reports drift; it is bounded and small,
 * not the unbounded latch. kWindow is a structural count of samples to average, not a magic
 * threshold a real step could fall below. Scheduler::step_finished feeds it one update per
 * single-sequence decode step. */
struct DecodeFloor {
    /* Single-sequence decode steps averaged into one floor candidate. 32 approximates the number
     * of decode steps the dashboard's 500 ms poll spans at this lane's step time, so the engine
     * floor and the page's own c1 floor are computed over comparable windows. */
    static constexpr int kWindow = 32;

    void update(int64_t ns) {
        /* Reject degenerate brackets BEFORE they enter a window. A decode step issued ahead behind
         * a long host commit (a prefill chunk, or a KV checkpoint every 256 tokens) brackets at
         * ~9% of a real step; real single-sequence steps bracket at ~96%. The populations are ~10x
         * apart, so any threshold between them separates them -- 0.5 is the midpoint of that gap,
         * not a value tuned to one distribution. Filtering at the STEP level is essential: at the
         * observed ~1.4% rate a degenerate dilutes inside a 32-step window to a mean of ~0.91x,
         * which no window-level threshold would catch, and a min over hundreds of windows then
         * reliably lands on a degenerate-heavy window ~10% below the true floor.
         * No rejection until a window's worth of samples has been accepted, so the running mean is
         * never bootstrapped from a degenerate. */
        if (accepted_ >= kWindow && ns * 2 < mean_) return;
        ++accepted_;
        mean_ += (ns - mean_) / (double)accepted_;
        sum_ += ns;
        if (++count_ < kWindow) return;
        const int64_t mean = sum_ / count_;
        /* The FIRST window is recorded by neither side. The rejection gate needs a window's worth
         * of accepted samples before it can judge one, so the first window is by construction
         * un-gated and can carry degenerates; recording it would let the floor latch a value the
         * engine can no longer reach -- the same artifact the gate exists to remove, smuggled in
         * through bootstrap. Measured: at a 5% degenerate rate the un-gated first window put the
         * floor ~2% below the true floor and it stuck there forever. */
        if (warmed_) {
            if (best_ == 0 || mean < best_) best_ = mean;
            recent_ = mean;             /* most recent completed window, same size as best_ */
        }
        warmed_ = true;
        sum_ = 0;
        count_ = 0;
    }
    /* The floor: the fastest window mean ever seen. A restart is priced against this. */
    int64_t best() const { return best_; }
    /* The CURRENT speed: the most recent completed window mean, from the SAME stream and window
     * size as best(). The advisory compares best() against recent() -- both window means of one
     * population -- so the two sides cannot disagree by order statistics the way the engine's
     * floor disagreed with the browser's per-poll min (a 7.7% phantom gap on a flat lane,
     * 2026-10-18). 0 until the first window completes; the page must not divide by it. */
    int64_t recent() const { return recent_; }

private:
    int64_t sum_    = 0;
    int     count_  = 0;      /* samples in the window being accumulated */
    int64_t best_   = 0;
    int64_t recent_ = 0;
    bool     warmed_  = 0;    /* false until one window has completed (the un-gated bootstrap) */
    int64_t accepted_ = 0;    /* samples accepted across all windows (rejection gate's support) */
    double  mean_   = 0.0;    /* running mean of ACCEPTED steps -- the rejection reference */
};

}  /* namespace rad */
