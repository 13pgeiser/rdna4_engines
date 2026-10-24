/* bind.h -- the two views of a streamed group's one pool allocation.
 *
 * `layer_slot` stays the group-local position of every bound layer, because it is also the host
 * mirror's row (rad_kv_layer_index -> kv_stream_host). Aliasing is a *depth*: the stage view
 * reports depth 1, so every layer's operand resolves to offset 0 of the same allocation while the
 * mirror keeps its per-layer addressing (docs/KV-STREAMING.md §3.5.1).
 *
 * A pass selects a view from its phase *and row shape*, before it is recorded or replayed. A
 * one-row-a-sequence decode step (RAD_PHASE_DECODE within 1 + n_spec rows a sequence) takes the
 * ring: its selection is rewritten to resident slots and misses are copied. Every other step --
 * prefill, a mixed prefill+decode step (RAD_PHASE_MIXED), and a decode-classified pass with more
 * rows a sequence than a decode step can have (the serial drafter's history pass over a chunk) --
 * takes the stage: the stage is the group's whole context at identity addressing, so a decode row
 * in a mixed chunk reads its block ids directly, exactly as a prefill row does, and no selection can
 * overflow a ring. Selecting per pass is visible to the tape because `phase`, `n_tok`, `n_seq` and
 * `n_spec` are already in the pass key.
 */
#pragma once
#include "mem/kv.h"
#include "runtime/ctx.h"

namespace rad {

enum class KVView { Ring, Stage };

/* Fill both fixed views from the plan and make the ring active. Call at bind time, after the
 * plan's carve has set n_slots, n_blocks and depth. A non-paged group is left untouched. */
void kv_bind_views(const KVGroupPlan& p, KVPoolBinding* b);

/* The offset of a layer's cache in the active view. depth 1 aliases every layer to the pool base;
 * the `off + layer_stride <= bytes` bound in issue.cpp is what makes that legal, and it is left
 * exactly as it was. */
int64_t kv_binding_offset(const KVPoolBinding& b, int64_t layer);

/* The view a pass uses. `n_tok`, `n_seq` and `n_spec` are the pass's row shape, not just its
 * phase: a decode-classified pass with more rows a sequence than a decode step can have (the
 * serial drafter's history pass over a chunk) reads a prefill-shaped set and must take the stage.
 * See bind.cpp. */
KVView kv_pass_view(int phase, int64_t n_tok, int64_t n_seq, int n_spec);

/* Make `view` the binding's active view. Stage falls back to the ring when the group has none. */
void kv_binding_enter(KVPoolBinding& b, KVView view);

/* One rank's step view of a group's ring and mirror, as the batch carries it; null when the group
 * is not streamed. `src` is that rank's own manager view when the caller has per-rank views
 * (`per_rank`), else rank 0's, whose map pointers are rebased from rank 0's pool (`kv0`) into this
 * rank's (`kvr`).
 *
 * Each rank names its OWN manager's stream view. The device maps live in that rank's KV pool and
 * the host mirror is that rank's pinned allocation; sharing rank 0's view made two ranks write
 * different KV shards into the same mirror bytes, which a staged prefill then read back. A caller
 * that gave no per-rank views (tests) still gets the old offset rebase, so a follower's maps do not
 * point into its peer's pool.
 *
 * The addresses are fixed at bring-up, so a step's view is a copy of pointer values, not a per-step
 * description; the batch outlives it with the `out` cell it is written into. */
const RadKVStream* kvstream_step_view(RadKVStream* out, const RadKVStream* src, bool per_rank,
                                      const void* kv0, const void* kvr);

/* The identity of a step view, for a recorded pass's key (Ctx::pass_key): the memory a kernel reads,
 * not the address of the struct that carries it. The batch's view is a copy in one of the builder's
 * Out slots (batch.h kOut, 68 of them with speculation at its maximum), and the slot advances every
 * step, so keying on `&view` made every streamed decode step a pass the tape store had never seen:
 * it recorded a new tape each step, the least-recently-used 24 tapes dropped it long before its slot
 * came round again, and so no streamed pass ever played: the host walked the whole issue path every
 * step instead of replaying eight packets. The pointers below are fixed at bring-up
 * (kvstream_step_view copies them), so this is stable across steps and still separates two groups
 * that hand a kernel different memory. */
uint64_t kvstream_view_key(const RadKVStream* v);

}  /* namespace rad */
