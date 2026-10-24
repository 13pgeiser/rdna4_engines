/* stage.h -- the one-layer prefill stage: per pass, per layer, copy the live part of the layer's
 * mirror row into the stage view of the group's pool, on the card's DMA engine, one layer ahead.
 *
 * ORDERING RULE -- a correctness invariant, not a preference.
 *
 * Aliasing every layer to offset 0 is safe only because, for each layer L, these four happen in
 * this order:
 *
 *   1. the stage copy of L has landed;
 *   2. L's store writes this chunk's tokens into the stage (the stage copy reads the mirror, which
 *      the store has not written yet, so a copy landing after the store would put the mirror's
 *      older bytes back over them);
 *   3. L's attention reads the stage;
 *   4. the stage copy of L+1 starts, and nothing reads the stage until its wait.
 *
 * The copy is a list on the card's DMA engine, which holds no CU slot, so the order does not rest
 * on one stream's issue order alone:
 *
 *   - 1 before 2 is a barrier the compute stream takes on an event the *list itself* completes
 *     (rad_event_dma hands the list the event's completion word as its `done`), issued immediately
 *     before the layer's store. `rad_event_wait` is a barrier packet, so the wait is device-side and
 *     in issue order: nothing between it and the store can slip in front of it;
 *   - 4 after 3 is a ride (`rad_ride`) taken on the compute stream once every attention launch of
 *     layer L has been issued, so the signal it completes is behind all of L's reads. The list polls
 *     that signal, so it starts when L's attention is done, and crosses the link under L's MLP and
 *     L+1's projections.
 *
 * What breaks it, in the order it is likely to happen:
 *
 *   - taking the ride *before* all of L's attention launches. A prefill layer's attention is issued
 *     in `qsa_gq_rows`-wide chunks, so a ride taken at the first chunk starts the copy of L+1 while
 *     the second chunk still reads the stage. That is why the seam is after the loop
 *     (`rad_kv_stage_next`), not in `resolve_sel`;
 *   - waiting for the copy *after* the layer's store instead of before it. The store's bytes are the
 *     newest ones and the copy's are older ones from the same mirror row;
 *   - arming the completion event on the host instead of on the compute stream. An older 0 from the
 *     previous layer's list would then release this layer's wait early, and a fresh 1 would cut the
 *     previous wait short -- one signal, two uses, no order between them. The record packet puts the
 *     arm in stream order, behind the previous wait;
 *   - the copy on lane 1, or on a bare second stream with no recorded event pair (see §3.5.1);
 *   - re-issuing the copy host-side without `tape_host`. The list is not a queue packet, so a
 *     recorded pass must re-run the arm and the submit where it made them; `Ctx::tape_kv_stage`
 *     does that with the tape paused, as `xprefetch.cpp` does for the prefetch.
 *
 * `haz_mark` sees the KV operand as the same byte range for every aliased layer, so a stage and the
 * previous layer's attention are never marked overlap. That is a relaxation check, not the
 * protection; do not weaken it (core/runtime/issue.cpp, "haz_mark"). The DMA copy is not a dispatch
 * at all, so the hazard pass cannot see it: the barriers above are its whole ordering.
 *
 * VOLUME -- and the one way this file can be silently wrong.
 *
 * The copy moves the mirror row up to the highest block id the coming attention can read, not the
 * whole row: `n_blocks * block_bytes` is the context allocation, and a 17k-token request was
 * paying 24 GiB of PCIe to stage 102 MiB of live KV (docs/KV-STREAMING.md §3.5, §5 Phase 3).
 *
 * `stage_live_bytes` (kvstream_bound_live, called once per released batch from the engine) is an
 * UPPER bound on the block ids the pass can name, and an UNDERESTIMATE IS NOT A CRASH: the attention
 * would read bytes the copy never moved -- whatever the previous layer left there -- and produce
 * fluent wrong text. It cannot be too small because:
 *
 *   - every id a block table can hold came out of the group's allocator and is held, and
 *     `KVManager::held_top_blocks` is one bucket past the highest held block it has counted. The
 *     histogram is updated on every acquisition (grab_block, reserve_specific_block) and on the last
 *     release, so it counts sequence blocks and cache-retained blocks alike;
 *   - the stage view is identity-addressed from block 0 for every layer, so copying [0, bound)
 *     covers every id below it, whether or not the ids are dense;
 *   - it is re-evaluated every step, so the chunk that appends a block raises it before the pass
 *     that reads it. A partial tail block and a speculative token live in blocks `ensure` took
 *     before the step was built, so they are held and inside the bound; the bound is in blocks, so
 *     a partial block is copied whole and its unwritten tail is the mirror's zeros.
 *
 * Ascending allocation is NOT assumed. The free list is a min-heap, so a fresh sequence usually
 * takes 0,1,2..., but a prefix-cache-retained block or a restore's `reserve_specific_block` can
 * leave a live block above a free one, and a reused id need not be the sequence's next. That is
 * why the bound is the allocator's high-water mark and not `ceil(end / block_size)`.
 *
 * What the bound costs: it is the whole group's held set, including blocks another sequence or the
 * prefix cache holds, so a server with a long cached conversation behind it stages more than the
 * request reads. It is never less, and a group with nothing held (0) stages the whole row as before.
 *
 * A backend with no DMA queue (the host backend, a test) copies on the compute stream exactly as
 * before, in the same four-step order.
 */
#pragma once
#include "mem/kv.h"
#include "runtime/ctx.h"

namespace rad {

/* Select the view of every streamed group a pass names, and reset the stage's per-pass state.
 * Called once before the pass loop (seams in core/runtime/ctx.cpp). Here rather than in bind.h
 * because it is the pass's opening move -- the driver's, not the layout's -- and because a caller
 * that only builds a step's view must not link the driver (core/sched/batch.cpp). */
void kvstream_enter(Ctx* c, const RadBatch* b);

/* Carry the block manager's upper bound on live block ids into every streamed group's stage view,
 * for the step about to run. Called for every rank from Engine::prepare_release, with the ranks
 * parked, after the step's blocks are ensured and before its layers are staged; rank 0's manager is
 * the one that hands the ids out. A group with nothing held leaves the view at 0, which copies the
 * whole row (stage.h, "VOLUME"). */
void kvstream_bound_live(Ctx* c, const KVManager& kv);

/* The driver, as a tape step re-runs it (Ctx::tape_kv_stage). `after` is the seam between the
 * layer's store and the next layer's copy. */
int kvstage_run(Ctx* c, int group, int layer, bool after);

}  /* namespace rad */

extern "C" {
/* Is this layer staged? 1: its row is whole for this pass (the caller must not resolve it). 0: the
 * group has no stage view, so the caller proceeds normally. Called from `resolve_sel`, which is the
 * pass's later seam: the wait itself happens before the layer's store, so this only reports. */
int rad_kv_stage_layer(RadCtx* c, rad_kvgroup group, const RadKVGroupBatch* kvb, int layer);
/* Issue the next bound layer's stage copy, after this layer's attention has been issued. */
int rad_kv_stage_next(RadCtx* c, rad_kvgroup group, int layer);
}
