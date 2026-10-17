/* geometry.h -- the prefill chunk geometry, computed from the resolved kernels.
 *
 * The attention block size comes from the resolved attention kernel (§7.2) and the linear-state
 * chunk length from every resolved kernel that marks a parameter RAD_PROLE_SEQ_CHUNK, so the
 * operator never has to supply a minimum --max-num-batched-tokens by hand (as vLLM requires for
 * hybrid models). If the budget is below what the geometry needs, startup prints the number and
 * refuses: a smaller budget means chunks never land on a checkpoint boundary and the linear
 * prefix hit rate is zero.
 */
#pragma once
#include "rad_core.h"

namespace rad {

struct ChunkGeometry {
    /* Attention block size, the lcm across every paged KV group. libr4d's paged attention is
     * compiled for 16 only. */
    int64_t attn_block = 1;

    /* Tile length of a linear-state scan (64 for libr4d's gated delta net). Taken from the
     * declared geometry and kernel constraints of every op whose schema marks a parameter
     * RAD_PROLE_SEQ_CHUNK. */
    int64_t state_chunk = 1;

    /* lcm(attn_block, state_chunk); every non-final prefill chunk is a multiple of it, so no
     * kernel is handed a partial tile. */
    int64_t quantum = 1;

    /* Linear-state checkpoint interval (§7.3): the operator's request rounded up to a multiple
     * of `quantum`, so a split at the interval is a legal chunk split. Zero without linear state
     * (see window_ckpt for the exception). */
    int64_t checkpoint_interval = 0;

    /* How many intervals one prefill chunk may carry from an interval boundary past the first
     * (Config::prefill_span); 1 without an interval grid. plan_chunk is where it acts. */
    int64_t span = 1;

    /* The budget of every step a span does not take (prefill_shared_tok). A span is for a prefill
     * alone in its step (Scheduler::step); a step carrying anything else is planned at this
     * budget as a deployment without a span plans it, so the span's extra rows never lengthen a
     * step other requests wait on. */
    int64_t shared_tok = 0;

    /* The smallest --max-num-batched-tokens this geometry can be served with. */
    int64_t min_max_tok = 1;

    /* Smallest upper bound a kernel places on the chunk length, or 0. Used only to report a
     * contradiction with `quantum`, never to clamp. */
    int64_t max_chunk = 0;

    /* Every contributor by name, for the startup line and the refusal. */
    std::string why;

    /* No linear state but a sliding window: the checkpoint snapshots only the window behind a
     * prompt's tail (KVManager::has_window_snapshot). Chunks are not split at the interval, since
     * a mid-prompt window snapshot gains nothing; the interval is only the shortest prompt worth
     * a snapshot. */
    bool window_ckpt = false;

    bool checkpoints() const { return checkpoint_interval > 0; }
};

/* The budget of a step a span does not take (ChunkGeometry::shared_tok), from the configuration
 * alone: --max-num-batched-tokens less the span's extra intervals, at least one interval, and the
 * whole budget without a span. The engine states it to the plugin before declare
 * (RadBuildCtx::max_mixed_tok) and the scheduler plans by it, so the two agree. */
int64_t prefill_shared_tok(const Config& cfg);

/* Resolve the geometry from the declared program and the operator's budgets.
 *
 * Returns RAD_E_INVAL when cfg.max_tok is too small (the required number is in
 * out->min_max_tok, the reason in out->why; the caller stops), and RAD_E_SHAPE when two kernels'
 * constraints contradict each other, which is a kernel-set bug. */
int chunk_geometry_resolve(const Program& prog, const Config& cfg, ChunkGeometry* out);

}  /* namespace rad */
