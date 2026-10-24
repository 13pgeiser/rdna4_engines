/* carve.cpp -- see carve.h. */
#include "kvstream/carve.h"

#include "kvstream/mirror.h"
#include "kvstream/plan.h"
#include "mem/kv.h"

namespace rad {

int64_t kvstream_maps_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                              int64_t max_ctx, int64_t max_seqs) {
    /* One loop for the budget's term and the report's size: KVStreamStore::cost() is the predicate
     * alloc() uses and the arithmetic kvstream_map_bytes states, so the two cannot drift. */
    return st.cost(plans, max_ctx, max_seqs).map_bytes;
}

bool kvstream_dense(const KVGroupPlan& p, const std::vector<KVGroupPlan>& plans,
                    const KVStreamStore& st) {
    if (!p.paged() || p.zero || p.kind != RAD_KV_FULL || !p.dense_read) return false;
    for (const auto& q : plans)
        if (!q.zero && q.paged() && st.windowed(q.index)) return true;
    return false;
}

int64_t kvstream_dense_blocks(const KVGroupPlan& p, int64_t max_ctx, int64_t max_seqs) {
    if (p.block_size <= 0 || max_ctx <= 0) return 0;
    const int64_t seqs = max_seqs > 0 ? max_seqs : 1;
    return seqs * ((max_ctx + p.block_size - 1) / p.block_size);
}

int64_t kvstream_dense_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                               int64_t max_ctx, int64_t max_seqs) {
    int64_t bytes = 0;
    for (const auto& p : plans) {
        if (!kvstream_dense(p, plans, st)) continue;
        bytes += align_up(kvstream_dense_blocks(p, max_ctx, max_seqs) * p.bytes_per_block,
                          RAD_ALIGN_UNIT);
    }
    return bytes;
}

void kvstream_split_windows(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                            int64_t max_ctx, int64_t max_seqs, int64_t* remaining,
                            std::vector<int64_t>* out) {
    for (const auto& p : plans) {
        if (!p.paged() || p.zero || !st.windowed(p.index)) continue;
        const int64_t slots = st.resident_blocks(p);
        *remaining -= kv_pool_bytes(p, slots, st.context_blocks(p, max_ctx, max_seqs));
        (*out)[p.index] = slots;
    }
    /* The maps are the rest of the arrangement. A probe that sizes the pool must pay them even
     * though it cannot reserve them yet, or the pool it chooses is short by exactly their bytes and
     * the first resident group carved against it falls short of --max-model-len. */
    *remaining -= kvstream_maps_charged(plans, st, max_ctx, max_seqs);
    /* A dense-read group does not take a share of the split: its requirement is every sequence's
     * context, which is a floor fixed by the run configuration. Charging it here (and leaving it
     * out of the shared split in paged_split) is what keeps the sibling's host mirror from
     * promising max_seqs sequences that the indexer cannot address. */
    for (const auto& p : plans) {
        if (!kvstream_dense(p, plans, st)) continue;
        const int64_t blocks = kvstream_dense_blocks(p, max_ctx, max_seqs);
        *remaining -= align_up(blocks * p.bytes_per_block, RAD_ALIGN_UNIT);
        (*out)[p.index] = blocks;
    }
}

int kvstream_window_fit(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                        int64_t fixed, int64_t budget, int64_t max_ctx, int64_t max_seqs) {
    int64_t window_bytes = 0;
    int64_t window_groups = 0;
    for (const auto& p : plans) {
        if (!p.paged() || p.zero || !st.windowed(p.index)) continue;
        /* A window group already bounds its own history: the ids it allocates are its window's,
         * while a streamed group's ids are its whole context. The two cannot both be true, and
         * context_blocks() would size the table of the first as if it were the second. */
        if (p.kind != RAD_KV_FULL) {
            RAD_ERR("KV group '%s' is a %s group and streaming is set for it: only a full "
                    "attention group streams. A window already bounds its own history, so the ids "
                    "it allocates are its window's, and a streamed group's ids are its whole "
                    "context -- the residency map cannot be right for both. Leave this group "
                    "unstreamed (--kv-resident 0 for it).",
                    p.name.c_str(), kv_kind_name(p.kind));
            return RAD_E_INVAL;
        }
        window_bytes += kv_pool_bytes(p, st.resident_blocks(p),
                                      st.context_blocks(p, max_ctx, max_seqs));
        ++window_groups;
    }
    const int64_t map_bytes   = kvstream_maps_charged(plans, st, max_ctx, max_seqs);
    const int64_t dense_bytes = kvstream_dense_charged(plans, st, max_ctx, max_seqs);
    int64_t dense_groups = 0;
    for (const auto& p : plans) if (kvstream_dense(p, plans, st)) ++dense_groups;
    if (fixed + window_bytes + map_bytes + dense_bytes > budget) {
        RAD_ERR("the resident KV windows and dense-read groups do not fit vram_kv:\n"
                "  resident windows   %10s  (%lld group(s) streaming)\n"
                "  ring maps          %10s  (the streamed groups' residency maps)\n"
                "  dense-read groups  %10s  (%lld group(s) at %lld sequences x %lld tokens)\n"
                "  linear/conv state  %10s  (%lld sequences)\n"
                "  vram_kv            %10s\n"
                "  short by           %10s\n"
                "Lower --kv-resident (the ring), or --max-model-len / --max-num-seqs (the one-layer "
                "stage and a dense-read group's floor are sized from them), or give the cache more "
                "of the card: a smaller --gpu-headroom-mib or a larger --host-pool-mib. The fixed "
                "costs in this table are not what is short: it is a streamed group's ring and "
                "stage, or the indexer beside it that must hold every sequence's context.",
                humanb(window_bytes).c_str(), (long long)window_groups,
                humanb(map_bytes).c_str(), humanb(dense_bytes).c_str(), (long long)dense_groups,
                (long long)max_seqs, (long long)max_ctx, humanb(fixed).c_str(),
                (long long)max_seqs, humanb(budget).c_str(),
                humanb(fixed + window_bytes + map_bytes + dense_bytes - budget).c_str());
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

int kvstream_apply(std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                   int64_t max_ctx, int64_t max_seqs) {
    for (auto& p : plans) {
        if (!p.paged()) continue;
        if (st.windowed(p.index) && !p.zero) {
            /* A streamed group's ids run over its context, which lives in host memory; the pool
             * backs only its resident window. So the split above gave this group its window (the
             * one number the pool may spend), and the context comes from the run configuration.
             * A window wider than the context is the context: there is nothing to stream. */
            p.n_blocks = st.context_blocks(p, max_ctx, max_seqs);
            p.n_slots  = st.resident_blocks(p);
            /* Only a ring smaller than the context needs the second, one-layer view. A ring that
             * covers the context streams nothing and never reaches this cap (kvstream_ring), so a
             * windowed group here always has the stage. */
            p.depth    = p.streamed() ? 1 : 0;
        } else {
            /* Not streamed: it addresses exactly the blocks it carves. */
            p.n_slots = p.n_blocks;
        }
        if (p.n_slots > p.n_blocks) p.n_slots = p.n_blocks;
        /* Streaming wins over replication for one group, and the resolution is not an error
         * because the two claims are about different stores. rad_kv_group_replicated says a tier
         * slot holds the group once; a streamed group has no tier entry at all (kv_tier_cached) --
         * its host mirror is the tier -- so the flag would only misdescribe the slot. A plugin
         * declares replication because the arithmetic of the group is the same on every rank (an
         * MLA latent), which stays true while it streams. */
        if (p.streamed()) p.replicated = false;
        /* A streamed group needs somewhere for the blocks the pool does not back. Refused here,
         * not at the first attention read: an unbacked id is written past the layer stride and the
         * result is fluent wrong output with no way to bisect it (docs/KV-STREAMING.md §3.1).
         * RAD_E_INVAL, not RAD_E_STATE: RAD_E_STATE stringifies as "called in the wrong phase",
         * and this is a caller handing the manager a cap without the backing for it. */
        if (p.streamed()) {
            const KVStreamStore::Map* m = st.map(p.index);
            const int64_t need = p.n_blocks * p.bytes_per_block;
            if (!m || !m->host || !m->res || m->host_bytes < need) {
                RAD_ERR("KV group '%s' streams -- %lld of its %lld blocks resident -- but nothing "
                        "holds the other blocks: a streamed group needs a host copy of %s and a "
                        "residency map before it is carved. This is a wiring bug, not a flag: "
                        "--kv-resident is set only where the engine also allocates both.",
                        p.name.c_str(), (long long)p.n_slots, (long long)p.n_blocks,
                        humanb(need).c_str());
                return RAD_E_INVAL;
            }
        }
    }
    return RAD_OK;
}

std::string kvstream_report_line(const KVGroupPlan& p) {
    /* A streamed group's pool is smaller than its context, so the two numbers differ and the
     * group's own line would read as a pool that holds the whole context. Both tiers in one line:
     * the ring and where it lives (the pool, at its offset), and the host mirror the misses are
     * copied from. The maps are the group's own small device allocation, taken from the same pool,
     * so they are named here too rather than being invisible. */
    return fmt("    %-14s %-6s streaming: %7lld blocks resident in vram_kv at "
               "+%lld = %9s, %7lld blocks = %9s mirrored in pinned host memory, "
               "maps %s\n", "", "", (long long)p.pool_blocks(),
               (long long)p.pool_offset, humanb(p.pool_bytes).c_str(),
               (long long)p.n_blocks,
               humanb(p.n_blocks * p.bytes_per_block).c_str(),
               humanb(kvstream_map_bytes(p.n_blocks, p.pool_blocks(), p.n_layers)).c_str());
}

}  /* namespace rad */
