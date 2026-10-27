/* stage.h -- the one-layer prefill stage: per pass and layer, copy the live part of the layer's
 * mirror row into the group's stage view, on the card's DMA engine, one layer ahead.
 *
 * Ordering rule. Every layer aliases offset 0 of the stage, which is safe only because for each
 * layer L these happen in order:
 *
 *   1. the stage copy of L has landed;
 *   2. L's store writes this chunk's tokens into the stage (the copy reads the mirror, which does
 *      not hold them yet, so a copy landing later would put older bytes back over them);
 *   3. L's attention reads the stage;
 *   4. the copy of L+1 starts.
 *
 * The copy is a DMA list, which holds no CU slot, so the order cannot rest on one stream's issue
 * order:
 *
 *   - 1 before 2: the compute stream waits on an event the list itself completes (rad_event_dma
 *     hands the list the event's completion word as its `done`), issued immediately before L's
 *     store. The wait is a barrier packet, so nothing between it and the store can pass it.
 *   - 4 after 3: the list polls a ride (rad_ride) taken on the compute stream after every
 *     attention launch of L (rad_kv_stage_next), so it starts when L's reads are done and crosses
 *     the link under L's MLP and L+1's projections.
 *
 * What breaks it:
 *
 *   - taking the ride before all of L's attention launches (a prefill layer's attention is issued
 *     in `qsa_gq_rows`-wide chunks, so the seam is after the loop, not in the resolve);
 *   - waiting for the copy after the layer's store instead of before it;
 *   - arming the completion word from the host instead of in stream order: an older 0 would then
 *     release this layer's wait early, or a fresh 1 cut the previous wait short;
 *   - the copy on lane 1, or on a second stream with no recorded event pair (docs/KV-STREAMING.md
 *     §3.5.1);
 *   - replaying a recorded pass without re-running the arm and the submit: the list is not a
 *     queue packet, so Ctx::tape_kv_stage re-runs them with the tape paused, as xprefetch.cpp does.
 *
 * The hazard pass sees the same byte range for every aliased layer, and it never sees the DMA
 * copy at all, so the barriers above are the copy's whole ordering (core/runtime/issue.cpp,
 * haz_mark).
 *
 * Volume. The copy moves the mirror row up to the highest block id the pass can read, not the
 * whole row (`n_blocks * block_bytes` is the context allocation). The bound is
 * KVManager::held_top_blocks, one past the highest held block (kvstream_bound_live, per step), and
 * it must never be low: the attention would read whatever the previous layer left there, which is
 * fluent wrong text, not a fault. It is an upper bound because
 *
 *   - every id a block table holds is held, and the held histogram counts sequence and
 *     cache-retained blocks alike;
 *   - the stage is identity-addressed from block 0, so copying [0, bound) covers every id below
 *     it, dense or not;
 *   - it is re-read every step, after the step's blocks are ensured, so a block a chunk appends or
 *     a speculative token's block is inside it; a partial block is copied whole and its unwritten
 *     tail is the mirror's zeros.
 *
 * Allocation is not assumed ascending: a retained or restored block can sit above a free one. The
 * bound counts every held block, so a server with a long cached conversation stages more than one
 * request reads; 0 (nothing held) stages the whole row.
 *
 * A backend with no DMA queue (the host backend, a test) copies on the compute stream, in the same
 * order.
 */
#pragma once
#include "mem/kv.h"
#include "runtime/ctx.h"

namespace rad {

/* Select the view of every streamed group the pass names and reset the stage's per-pass state.
 * Called before each pass (core/runtime/ctx.cpp). Not in bind.h, so a caller that only builds a
 * step's view (core/sched/batch.cpp) does not link the driver. */
void kvstream_enter(Ctx* c, const RadBatch* b);

/* Set every streamed group's stage bound from `kv` (rank 0's manager, which hands out the ids) for
 * the step about to run. Called for every rank from Engine::prepare_release, with the ranks parked
 * and the step's blocks ensured. */
void kvstream_bound_live(Ctx* c, const KVManager& kv);

/* The driver, also re-run by a recorded pass (Ctx::tape_kv_stage). `after` is the call after the
 * layer's attention (rad_kv_stage_next) rather than before its store. */
int kvstage_run(Ctx* c, int group, int layer, bool after);

}  /* namespace rad */
