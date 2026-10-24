/* plan.h -- the KV streaming policy, as pure arithmetic on a KVGroupPlan.
 *
 * A streamed group has two device views over one allocation (docs/KV-STREAMING.md §4.1):
 *
 *   ring   n_slots blocks deep, `n_layers` layer stripes, addressed through the residency map;
 *   stage  n_blocks blocks deep, ONE layer stripe, addressed by block id.
 *
 * The allocation is max(ring_bytes, stage_bytes), never the sum: a decode-only ring is tiny
 * against one layer of the context, so the stage sets the price and the ring fits inside it.
 * `depth` on the plan is the stripe count of the staged view (1); the ring view keeps the
 * group's own `n_layers` stripes. Nothing here touches a device, so this file is testable
 * beside mem/kv.h's arithmetic.
 *
 * The ceiling is the number the VRAM budget is resolved from (Engine::plan, before
 * cap_resident_tokens() runs). A streamed group must contribute its stage, not its whole
 * cache, or the phase frees nothing: that is why kvstream_cap_probe() is called on the
 * throwaway probe before probe.ceiling() (docs/KV-STREAMING.md §4.1, point 5).
 */
#pragma once
#include "mem/kv.h"

namespace rad {

/* Bytes of the decode ring view: `slots` blocks, every layer. Inline because it is one
 * multiplication on the plan and because a caller that only lays a view out (kvstream/bind.cpp)
 * must not link the probe below, which names KVManager: core/sched/batch.cpp is linked by tests
 * whose weak device fakes an extracted KVManager would displace. */
inline int64_t kv_ring_bytes(const KVGroupPlan& p, int64_t slots) {
    return slots > 0 ? slots * p.bytes_per_block : 0;
}

/* Blocks the group's context addresses: `max_seqs` sequences at `--max-model-len`, rounded up to a
 * whole block each. This is the one spelling of the id space, shared by the store's mirror/map
 * sizing (KVStreamStore::context_blocks) and the startup ring decision, so the two cannot disagree
 * about whether a ring covers the context. */
inline int64_t kv_context_blocks(int64_t block_size, int64_t max_ctx, int64_t max_seqs) {
    if (block_size <= 0) return 0;
    const int64_t seqs = max_seqs > 0 ? max_seqs : 1;
    const int64_t ctx  = max_ctx > 0 ? max_ctx : 0;
    return seqs * ((ctx + block_size - 1) / block_size);
}

/* The one allocation: max(ring, stage). `slots <= 0` or `slots >= n_blocks` is a group that does
 * not stream (no ring asked for, or one that covers the context), and it keeps its old
 * single-view cost. A non-paged group is unchanged. */
int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots);
int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots, int64_t context_blocks);
int64_t kv_pool_bytes(const KVGroupPlan& p);

/* The pool cost of a group at `blocks` blocks of context and `slots` resident blocks. This is
 * what ceiling() sums; with no streamed slot it is the old blocks * bytes_per_block. */
int64_t kv_ceiling_bytes(const KVGroupPlan& p, int64_t blocks, int64_t slots);

/* Whether the idle tiers may gather this group. A streamed group's pool holds one layer's bytes,
 * so a tier gather of layer L would be the last staged layer copied n_layers times
 * (core/runtime/tierexec.cpp); its mirror is the tier, and the residency map restores a block on
 * the next resolve (docs/KV-STREAMING.md §4.1). */
bool kv_tier_cached(const KVGroupPlan& p);

/* The group --kv-resident streams: the widest paged FULL group whose selection the plugin has
 * declared a streamed path for, as configure_kv chooses it. `prog` is the resolved program: a group
 * is a candidate only if kvstream_ops states its pitch and page, so the choice is a rule rather
 * than a width coincidence. dense_read groups are never candidates: a step reads them over the
 * whole live history, so a ring cannot hold them back. -1 when the model has none. */
int32_t kv_stream_group(const KVManager& kv, const Program* prog);

/* The same choice from a rank's declarations, before the plans exist (Engine::configure_kv): the
 * group by declaration is the one the plans will hold, so the policy is chosen once and the plan
 * below follows it. */
int32_t kv_stream_group(const std::vector<KVGroupInfo>& groups, const Program* prog = nullptr);

/* What --kv-resident makes of one group: the ring's depth from the flag, the group's id space from
 * --max-num-seqs and --max-model-len, and whether anything streams at all. Pure arithmetic on
 * numbers the engine already has, so the budget probe and the carve decision cannot disagree. */
struct KVStreamRing {
    int64_t ring_blocks = 0;   /* --kv-resident, rounded up to whole blocks and raised to the
                                * ceiling below when the flag asks for less */
    int64_t raised_blocks = 0; /* the depth the flag was raised to (kvstream_ceiling_blocks), or 0
                                * when it already asked for that much: the ring is not the
                                * operator's number when this is set */
    int64_t ctx_blocks  = 0;   /* max_seqs x ceil(ctx / block_size): the group's whole context */
    /* A ring that covers the context is not a streamed group. The pool then holds every block
     * with its own id, so the group keeps the base engine's path exactly: no mirror, no residency
     * maps, no stage, no resolve, and a null RadKVStream (docs/KV-STREAMING.md §5 Phase 4). */
    bool streams() const { return ring_blocks > 0 && ring_blocks < ctx_blocks; }
};

/* The blocks one decode step's selections can name, which is what the ring must hold -- and the
 * depth --kv-resident is raised to when it asks for less. A within-(1+n_spec) decode pass takes
 * the ring (`kv_pass_view`); a decode-classified pass with more rows a sequence than that (the
 * serial drafter's history pass over a prefill chunk) stages, so it is not counted here. Under
 * speculation a decode step issues one selection per row, so a step is `max_seqs x (1 + n_spec)`
 * rows and each names at most `sel_pitch` distinct blocks -- the arch's `topk + 1`. 0 when the
 * program declares no kv_resolve: a dense model's attention reads a block id as a pool slot, so
 * its ceiling is the whole context and nothing can be streamed. */
int64_t kvstream_ceiling_blocks(const Config& cfg, int64_t sel_pitch);

/* The ops a streamed group's issue path actually needs, read off the program the plugin declared
 * (the builder's resolved op list, not the architecture's name). A streamed group is only safe
 * when its attention rewrites `sel` from block ids into ring slots (`kv_resolve`) and its store
 * writes the host mirror (`kv_store_streamed`, or the fused prologue that folds the store in).
 * Without the resolve, `sel` still holds block ids while the attention reads them as pool page
 * indices -- past the ring when the pool is smaller than the id space, which is a fault, not a
 * slow path. Without the store nothing writes the mirror, so a miss reads zeros.
 *
 * `group` narrows the match to the group the plan will stream: a declaration states that group's
 * geometry (`block_bytes` on the resolve, `block_size` on the store), so a resolve emitted for a
 * narrower group cannot enable streaming of another. Null matches any declaration; a declaration
 * that states no geometry is accepted as-is, so a plugin that names neither is never refused for
 * it. A declaration the builder recorded with no kernel in any domain is not a path: the plugin's
 * handle was null and nothing would be issued (the same test as Builder::op_resolved). That is how
 * a packed cache reads -- the architecture emits the store and the kernel table has no row for the
 * layout -- and it must not count, or the group would stream with a null writer. */
struct KVStreamOps {
    bool    resolve   = false;   /* kv_resolve, stating this group's layer bytes and sel_pitch */
    bool    store     = false;   /* kv_store_streamed: the mirror writer every streamed layer has */
    bool    folded    = false;   /* qk_norm_rope_gate_kv_store_streamed: the same store, folded */
    int64_t sel_pitch = 0;       /* the blocks one selection row can name, from the resolve */
    /* The ops are there, so this group can stream. */
    bool streams() const { return resolve && (store || folded); }
    /* The first op the plugin is missing, for the one startup line that says why nothing streams;
     * null when streams(). */
    const char* missing() const {
        if (!resolve) return "kv_resolve";
        if (!store && !folded) return "kv_store_streamed";
        return nullptr;
    }
};
KVStreamOps kvstream_ops(const Program& prog, const KVGroupInfo* group);

/* The ring's depth and the group's id space with nothing else decided: whether a ring covers the
 * context, so that the caller can tell "this model has no streamed path" (a warning) from "this
 * ring is below what a decode step can select" (a raise) without one hiding the other. */
KVStreamRing kvstream_ring_shape(const Config& cfg, int64_t ctx, int64_t block_size);

/* Read --kv-resident against `group`'s context and fill `out`. A ring below what one decode step
 * can select is raised to it -- `out->ring_blocks` becomes the ceiling and `out->raised_blocks`
 * says the flag did not ask for that much -- not refused: the engine computes the floor and the
 * operator asked for streaming, not for a number. The caller states the one warning, once, and
 * startup proceeds. A ring that covers the context is not a raise either -- it is
 * `out->streams() == false`, and the group stays a resident pool. `block_size` is the group's own
 * page (1 for a group whose page is unknown). Never refuses: both callers reach it only after
 * kvstream_ops().streams(), so the ceiling is positive, and a caller that names a pitch of its own
 * on a model with no kv_resolve is a case its caller has already warned about and ignored. */
void kvstream_ring(const Config& cfg, int64_t ctx, int64_t block_size, int64_t sel_pitch,
                   KVStreamRing* out);

/* Teach the budget probe the streamed plan: the ring from --kv-resident and the stage from the
 * context. Call before probe.ceiling(ctx); the group's ceiling then falls from n_layers of its
 * context to max(ring, one layer of it), and `cfg.kv_streams` records that the pool is priced that
 * way (core/mem/vram_budget.cpp). Returns RAD_OK on a model with nothing to stream, before any
 * weight is loaded. The ring it prices is the raised one (kvstream_ring), so the budget and the
 * carve take the same depth; the warning for it is kvstream_configure's, which runs once rather
 * than once a rank. A group whose architecture declares no streamed ops (kvstream_ops) is not
 * priced for streaming at all: that is the same RAD_OK, with `cfg.kv_streams` left false, so the
 * configuration is exactly the one without --kv-resident. */
int kvstream_cap_probe(KVManager& probe, Config& cfg, const Program& prog, int64_t ctx);

}  /* namespace rad */
