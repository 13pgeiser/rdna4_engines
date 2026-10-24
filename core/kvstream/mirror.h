/* mirror.h -- the host mirror tier and the resident window (docs/KV-STREAMING.md §4.1).
 *
 * A streamed group's device pool backs a resident window, not its context; the blocks the pool
 * does not hold live in a pinned host copy, laid out layer-major and block-indexed exactly like the
 * pool, and a device residency map says which blocks are in the pool (`-1` for the ones that are
 * not). This file owns that pair, the cap that narrows a group's pool to a window, and the
 * arithmetic that sizes both, so mem/kv.{h,cpp} keeps one KVStreamStore member and one call per
 * method. Nothing here touches a device except init(), which fills the maps on the stream it is
 * given.
 */
#pragma once
#include "mem/pools.h"

#include <cstdint>
#include <vector>

namespace rad {

struct KVGroupPlan;

/* What the streamed groups need, in bytes, before anything is taken. `host_bytes` is pinned
 * host memory (`blocks * bytes_per_block` per group, every rank its own shard); `map_bytes` is
 * the residency map, the ring's reverse map and the control block, all device memory and all
 * taken from the KV pool itself (a handful of MiB), so they are budgeted with the cache they
 * describe rather than as an uncharged side allocation.
 *
 * Valid after plan_groups() and before carve(); the arithmetic is the one carve() uses. */
struct KVStreamCost {
    int64_t groups     = 0;
    int64_t host_bytes = 0;
    int64_t map_bytes  = 0;
};

/* The resolve's cumulative counters, as metrics read them. A server that does not stream
 * reports zeros; `resident_blocks` is the ring's depth from the carve. */
struct KVStreamStats {
    int64_t  resident_blocks = 0;
    uint64_t misses = 0;        /* blocks placed in the ring by resolve launches */
    uint64_t bytes_in = 0;      /* bytes those placements copied from the host mirror */
};

/* The bytes a streamed group needs, before anything is taken: the mirror is the context the pool
 * would have held, and the maps are the residency map (one entry a logical block), the ring's
 * reverse map, its per-slot state and the control block. Only a full
 * attention group streams -- carve() refuses a window one by name -- so that is what is counted
 * here; counting a group the carve will refuse would take its mirror first and report a size the
 * operator cannot act on. */
int64_t kvstream_map_bytes(int64_t blocks, int64_t slots, int64_t layers);

/* A streamed group's storage off the card: the pinned host copy and the device residency map.
 *
 * The host copy must cover the group's whole context: `n_blocks * bytes_per_block`, the size the
 * pool would have had. Needed before carve() for any group cap() narrowed.
 *
 * The manager keeps one of these per streamed group and hands it to the run phase as the ABI's
 * RadKVStream (view), so the fields the plugin reads and the fields alloc() fills are one
 * definition: `host_bytes` and `slots`/`block_bytes` are the mirror's and the ring's sizes.
 *
 * The manager's whole streaming state is here, by group handle, so the policy is in this module and
 * KVManager only delegates: the resident window cap, the mirror and maps, and the resolve's
 * counters. Empty vectors and zeros on a manager that does not stream. */
class KVStreamStore {
public:
    using Map = RadKVStream;

    /* How much of a paged group's context its device pool backs, in tokens; 0 (the default) is all
     * of it, which is every group on a server that does not stream. Only the group named, because
     * a streamed group is usually one whose reads a step can bound -- the sparse attention group --
     * while its neighbours (an indexer's block keys, read densely over the whole history every
     * step) must stay where they are (docs/KV-STREAMING.md §4). Rounded up to a whole block and
     * never above the group's own carve. */
    void cap(int32_t group, int64_t tokens);

    /* A group whose pool is a resident window rather than its context (cap): its block ids run
     * over the context and only `n_slots` of them are in device memory. */
    bool windowed(int32_t group) const;

    /* Blocks of the group's resident window, rounded up to a whole block. A window shorter than one
     * block is one block: a pool that holds nothing is not a streamed group, it is a group that
     * cannot run. */
    int64_t resident_blocks(const KVGroupPlan& p) const;

    /* Blocks the group's context addresses: what the pool would hold if every sequence it may run
     * reached --max-model-len at once, i.e. the paged part of ceiling(). The block table is already
     * pitched for this (BatchBuilder sizes it from max_ctx), so the ids exist whether the pool backs
     * them or not. */
    int64_t context_blocks(const KVGroupPlan& p, int64_t max_ctx, int64_t max_seqs) const;

    void set_map(int32_t group, const Map& m);
    const Map* map(int32_t group) const;

    /* The ABI view of a streamed group's ring and mirror, or null for one that is fully resident.
     * Stable from bring-up: the batch builder copies the pointer into every step's batch, and a
     * recorded pass keyed by it replays only while the addresses do not move. */
    const RadKVStream* view(const KVGroupPlan* p, int32_t group) const;

    KVStreamCost  cost(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs) const;

    /* Take the mirror and the ring's maps for every streamed group and set_map() each. Call after
     * plan_groups() and before carve(): the carve refuses a streamed group with no map, and the
     * maps' bytes leave the pool before its paged split. The mirror is zeroed here, because a
     * block the prefix cache has not written yet is read as a miss and an uninitialised block
     * would be fluent wrong output. The device maps are not touched (an elastic pool backs its
     * range at carve), so init() fills them after it. On failure the mirror is released again;
     * release() in the engine's teardown covers a partial bring-up. */
    int  alloc(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs,
               MemAllocator& host_alloc, Pool& kv_pool);

    /* Fill the device maps: `res` and `slot_block` with -1, `slot_state` and `ctl` with zero (the
     * state a resolve launch expects at bring-up). Call after carve(), with the rank's stream
     * current, and sync before the first pass. */
    int  init(const std::vector<KVGroupPlan>& plans, RadStream s);

    /* Release what alloc() took. The device maps came from the KV pool and go with it; the host
     * mirror is outside every pool and is freed here. Safe to call twice and after a failed
     * bring-up: a group with no mirror is skipped. */
    void release(MemAllocator& host_alloc);

    KVStreamStats stats(const std::vector<KVGroupPlan>& plans) const;

    /* Charged by whoever reads a resolve launch's ctl: the step path never waits on the card, so
     * the kernel's cumulative numbers are read where the next unit drains them. */
    void note_traffic(int64_t misses, int64_t bytes) {
        if (misses > 0) misses_ += (uint64_t)misses;
        if (bytes  > 0) bytes_in_ += (uint64_t)bytes;
    }

private:
    std::vector<int64_t> resident_tokens_;   /* by group handle: its pool's depth in tokens */
    std::vector<Map>     maps_;              /* set_map(), by group handle */
    uint64_t misses_ = 0;                    /* note_traffic() */
    uint64_t bytes_in_ = 0;
};

}  /* namespace rad */
