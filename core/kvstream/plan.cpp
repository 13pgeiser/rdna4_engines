/* plan.cpp -- see plan.h. */
#include "kvstream/plan.h"

#include <algorithm>

namespace rad {

int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots) {
    return kv_pool_bytes(p, slots, p.n_blocks);
}

int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots, int64_t context_blocks) {
    if (!p.paged()) return p.n_states * p.bytes_per_state;
    const int64_t ring = kv_ring_bytes(p, slots);
    /* slots >= context_blocks is a ring that covers the context, which streams nothing: it is the
     * group's ordinary one-view pool. A streamed group (slots < context_blocks) also needs its
     * one-layer stage. The context is passed explicitly because paged_split runs before the carve
     * writes p.n_blocks. */
    if (slots <= 0 || slots >= context_blocks) return ring;
    return std::max(ring, context_blocks * p.layer_bytes_per_block());
}

int64_t kv_pool_bytes(const KVGroupPlan& p) {
    return kv_pool_bytes(p, p.pool_blocks());
}

int64_t kv_ceiling_bytes(const KVGroupPlan& p, int64_t blocks, int64_t slots) {
    if (!p.paged() || p.block_size <= 0) return 0;
    const int64_t ring = slots > 0 ? slots * p.bytes_per_block : 0;
    if (slots <= 0 || slots >= blocks) return blocks * p.bytes_per_block;
    /* The stage is context-deep and one layer wide. It and the ring are one reservation, so the
     * budget pays the larger, not the sum. */
    return std::max(ring, blocks * p.layer_bytes_per_block());
}

bool kv_tier_cached(const KVGroupPlan& p) {
    return !p.streamed();
}

namespace {
/* The one rule both overloads state: among the groups from index 1 on, a candidate is a full group
 * that owns bytes and is neither zero nor dense-read, whose declared ops stream it; the widest
 * page wins. Each overload supplies only how its own container spells those fields. */
bool kv_stream_candidate(int32_t kind, bool zero, bool dense_read, int64_t bytes_per_block) {
    return kind == RAD_KV_FULL && !zero && !dense_read && bytes_per_block > 0;
}

template <class Candidate>
int32_t kv_stream_widest(size_t n, const Candidate& candidate) {
    int32_t best = -1;
    int64_t widest = 0;
    for (size_t g = 1; g < n; ++g) {
        int64_t bytes = 0;
        if (!candidate(g, &bytes)) continue;
        if (bytes > widest) {
            widest = bytes;
            best = (int32_t)g;
        }
    }
    return best;
}
}  /* namespace */

int32_t kv_stream_group(const KVManager& kv, const Program* prog) {
    const std::vector<KVGroupPlan>& ps = kv.plans();
    return kv_stream_widest(ps.size(), [&](size_t g, int64_t* bytes) {
        const KVGroupPlan& p = ps[g];
        if (!kv_stream_candidate(p.kind, p.zero, p.dense_read, p.bytes_per_block)) return false;
        /* Only a group the program wires: a kv_resolve stating this group's per-layer pitch and a
         * streamed store stating its page. Every caller has the program (the builder's resolved
         * op list), so there is no width-only fallback here. */
        const KVGroupInfo* gi = g < prog->kv_groups.size() ? &prog->kv_groups[g] : nullptr;
        if (!kvstream_ops(*prog, gi).streams()) return false;
        *bytes = p.bytes_per_block;
        return true;
    });
}

int32_t kv_stream_group(const std::vector<KVGroupInfo>& gs, const Program* prog) {
    return kv_stream_widest(gs.size(), [&](size_t g, int64_t* bytes) {
        const KVGroupInfo& gi = gs[g];
        if (!kv_stream_candidate(gi.decl.kind, gi.zero, gi.dense_read, gi.bytes_per_block))
            return false;
        /* The declarations exist before the plans do; a caller with no program (the width-only
         * report fallback) keeps the width rule. */
        if (prog && !kvstream_ops(*prog, &gi).streams()) return false;
        *bytes = gi.bytes_per_block;
        return true;
    });
}

int64_t kvstream_ceiling_blocks(const Config& cfg, int64_t sel_pitch) {
    if (sel_pitch <= 0 || cfg.max_seqs <= 0) return 0;
    /* One selection per row: a sequence drafts n_spec tokens and a verify step runs them beside
     * the accepted position, so under speculation a step is (1 + n_spec) rows a sequence. */
    const int64_t rows = cfg.max_seqs * (1 + (cfg.n_spec > 0 ? cfg.n_spec : 0));
    return rows * sel_pitch;
}

KVStreamOps kvstream_ops(const Program& prog, const KVGroupInfo* group) {
    KVStreamOps ops;
    /* Whether this declaration has a kernel at all. `bands` being empty does NOT mean it has none:
     * the builder always leaves at least one band, and for an op no row matched it is a placeholder
     * with both domains null and a miss reason (core/build/rad_builder.cpp, resolve_bands). The same
     * predicate as Builder::op_resolved, read off the program a non-builder caller holds. */
    const auto resolved = [](const OpInfo& o) {
        for (const Band& b : o.bands)
            for (int d = 0; d < RAD_N_DOMAINS; ++d)
                if (b.dom[d]) return true;
        return false;
    };
    /* The group's per-layer block pitch, the figure kv_resolve states as `block_bytes`
     * (rad_kv_layer_bytes). bytes_per_block covers every bound layer. */
    const int64_t n_bound = group && !group->layers.empty() ? (int64_t)group->layers.size() : 1;
    const int64_t layer_bytes =
        group && group->bytes_per_block > 0 ? group->bytes_per_block / n_bound : 0;
    for (const OpInfo& op : prog.ops) {
        /* A declaration that resolved to nothing is not a streamed path: the plugin's handle is
         * null and nothing would be issued. A declared-but-unresolved store counts as no path,
         * which is the packed-cache shape -- the architecture emits the op and the kernel table has
         * no row for the layout -- and letting it count would stream a group whose store never
         * runs, so the mirror would stay empty and a miss would copy zeros. */
        if (!resolved(op)) continue;
        const bool is_resolve = op.op == "kv_resolve";
        const bool is_store   = op.op == "kv_store_streamed" ||
                                op.op == "qk_norm_rope_gate_kv_store_streamed";
        if (!is_resolve && !is_store) continue;
        /* The declaration states the geometry it was emitted for; a resolve or a store declared
         * for a narrower group must not enable streaming of this one. A declaration that states
         * neither is taken at its word. */
        long long stated = 0;
        if (is_resolve && layer_bytes > 0 && op.base.get_i("block_bytes", &stated) &&
            stated != layer_bytes)
            continue;
        if (is_store && group && group->block_size > 0 &&
            op.base.get_i("block_size", &stated) && stated != group->block_size)
            continue;
        if (is_resolve) {
            long long pitch = 0;
            if (!op.base.get_i("sel_pitch", &pitch) || pitch <= 0) continue;
            /* A resolve with no pitch cannot size the ring; the first one that states it wins, as
             * every layer of the group declares the same. */
            if (!ops.resolve) ops.sel_pitch = (int64_t)pitch;
            ops.resolve = true;
        } else if (op.op == "kv_store_streamed") {
            ops.store = true;
        } else {
            ops.folded = true;
        }
    }
    return ops;
}

KVStreamRing kvstream_ring_shape(const Config& cfg, int64_t ctx, int64_t block_size) {
    KVStreamRing out;
    const int64_t bs = block_size > 0 ? block_size : 1;
    out.ctx_blocks  = kv_context_blocks(bs, ctx, cfg.max_seqs);
    out.ring_blocks = cfg.kv_resident > 0 ? (cfg.kv_resident + bs - 1) / bs : 0;
    /* The shipped floor, unless the context is smaller than it (a model that cannot hold 32 768
     * tokens of this group streams nothing at all, which is the pre-existing behaviour). */
    if (out.ring_blocks > 0) {
        out.floor_blocks = (kStreamFloorTokens + bs - 1) / bs;
        if (out.floor_blocks > out.ctx_blocks) out.floor_blocks = out.ctx_blocks;
        if (out.ring_blocks < out.floor_blocks) {
            out.ring_blocks    = out.floor_blocks;
            out.raised_blocks  = out.floor_blocks;
        }
    }
    if (out.ring_blocks > out.ctx_blocks) out.ring_blocks = out.ctx_blocks;
    return out;
}

void kvstream_ring(const Config& cfg, int64_t ctx, int64_t block_size, int64_t sel_pitch,
                   KVStreamRing* out) {
    *out = kvstream_ring_shape(cfg, ctx, block_size);
    if (!out->streams()) return;   /* absent, 0, or a ring that covers the context */

    /* Anything under the computed amount is ignored and pulled up to it. A ring below what one
     * decode step can select is not a slower configuration, it is a request served from a
     * selection table that no longer addresses the pool -- and the engine computes that floor
     * itself, from --max-num-seqs, the drafter's depth and the arch's own sel_pitch. The operator
     * asked for streaming, not for a number, so the ring takes the floor and `raised_blocks` says
     * the flag did not ask for it: the caller prints the one line, and startup proceeds. Silent
     * here because this runs once a rank on the budget probe and once in bring-up, and the raise
     * is one startup fact, not three. The ceiling is always positive here: both callers reach this
     * only after kvstream_ops().streams(), which requires a kv_resolve with sel_pitch > 0, and
     * --max-num-seqs is parsed >= 1. */
    const int64_t ceiling = kvstream_ceiling_blocks(cfg, sel_pitch);
    if (out->ring_blocks < ceiling) {
        out->ring_blocks   = ceiling < out->ctx_blocks ? ceiling : out->ctx_blocks;
        out->raised_blocks = out->ring_blocks;
    }
}

/* The prefix cache's host tier cannot hold an entry that contains a streamed group: its rows are
 * addressed by the block id a sequence used, so they cannot be copied into a slot and their ids
 * cannot be given up for one (PrefixCache::demotable, docs/KV-STREAMING.md §4.1). A host tier
 * configured beside a streamed group would only be pinned and never used, so it is turned off
 * before anything is sized. `cfg.kv_streams` is what kvstream_cap_probe sets when the flag asks,
 * the architecture serves it and the ring is under the context -- a flag alone, a model with no
 * streamed path and a ring that covers the context all leave it false, and the tier stays on.
 * Returns the MiB taken off, for the caller's one line; 0 when nothing changed. */
int64_t kvstream_prefix_host_off(Config& cfg) {
    if (!cfg.kv_streams || cfg.prefix_cache_mib <= 0) return 0;
    const int64_t was = cfg.prefix_cache_mib;
    cfg.prefix_cache_mib = 0;
    return was;
}

int kvstream_cap_probe(KVManager& probe, Config& cfg, const Program& prog, int64_t ctx) {    cfg.kv_streams = false;
    if (cfg.kv_resident <= 0) return RAD_OK;
    const int32_t g = kv_stream_group(probe, &prog);
    if (g < 0) return RAD_OK;
    const KVGroupPlan* p = probe.plan(g);
    if (!p || p->block_size <= 0) return RAD_OK;
    /* This model's architecture declares no streamed path for the group the plan would stream:
     * price the pool the ordinary way, exactly as without the flag. The ring's arithmetic is for a
     * streamed group only, and the one warning this case deserves is printed by kvstream_configure
     * (this probe runs on every rank). */
    const KVStreamOps ops =
        kvstream_ops(prog, g < (int32_t)prog.kv_groups.size() ? &prog.kv_groups[(size_t)g] : nullptr);
    if (!ops.streams()) return RAD_OK;
    KVStreamRing ring;
    /* The ring this prices is the raised one (kvstream_ring), so the budget and the carve take the
     * same depth. */
    kvstream_ring(cfg, ctx, p->block_size, ops.sel_pitch, &ring);
    if (!ring.streams()) return RAD_OK;
    cfg.kv_streams = true;
    probe.cap_resident_tokens(g, ring.ring_blocks * p->block_size);
    return RAD_OK;
}

}  /* namespace rad */
