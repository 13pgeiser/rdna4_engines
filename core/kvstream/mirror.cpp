/* mirror.cpp -- see mirror.h. */
#include "kvstream/mirror.h"

#include <sys/mman.h>

#include "kvstream/plan.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

namespace {
int64_t ceil_div(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

/* Each array starts on an 8-byte boundary: the ABI's `ctl` counters are read as u64 pairs, and a
 * block count or a ring depth that is odd would leave a 4-byte boundary misaligned. The pool's base
 * is RAD_ALIGN_UNIT (4096) aligned, so rounding every part keeps the absolute addresses aligned. */
int64_t align8(int64_t n) { return (n + 7) & ~(int64_t)7; }

/* The reservation's one layout: the bytes of each array and its offset from the reservation's
 * base. alloc() cuts the reservation from it, init() resets each part at its own length, and
 * kvstream_map_bytes() is its total, so the budget, the carve and the reset cannot disagree. */
struct MapLayout {
    int64_t res_bytes = 0, slot_block_bytes = 0, slot_state_bytes = 0, ctl_bytes = 0;
    int64_t res = 0, slot_block = 0, slot_state = 0, ctl = 0;
    int64_t bytes = 0;   /* the whole reservation */
};

MapLayout map_layout(int64_t blocks, int64_t slots, int64_t layers) {
    if (layers < 1) layers = 1;
    MapLayout l;
    l.res_bytes        = layers * blocks * (int64_t)sizeof(int32_t);
    l.slot_block_bytes = layers * slots  * (int64_t)sizeof(int32_t);
    l.slot_state_bytes = layers * slots * 2 * (int64_t)sizeof(int32_t);
    l.ctl_bytes        = 16 * (int64_t)sizeof(int32_t);
    l.res        = 0;                                        /* res, block -> slot, one map a layer */
    l.slot_block = l.res        + align8(l.res_bytes);       /* slot_block, slot -> block */
    l.slot_state = l.slot_block + align8(l.slot_block_bytes);/* slot_state: epoch, heat */
    l.ctl        = l.slot_state + align8(l.slot_state_bytes);/* ctl, shared by the layers */
    l.bytes      = l.ctl        + align8(l.ctl_bytes);
    return l;
}
}  /* namespace */

int64_t kvstream_map_bytes(int64_t blocks, int64_t slots, int64_t layers) {
    return map_layout(blocks, slots, layers).bytes;
}

void KVStreamStore::cap(int32_t group, int64_t tokens) {
    if (group < 0) return;
    if ((size_t)group >= resident_tokens_.size()) resident_tokens_.resize((size_t)group + 1, 0);
    resident_tokens_[(size_t)group] = tokens > 0 ? tokens : 0;
}

bool KVStreamStore::windowed(int32_t group) const {
    return group >= 0 && (size_t)group < resident_tokens_.size() &&
           resident_tokens_[(size_t)group] > 0;
}

int64_t KVStreamStore::resident_blocks(const KVGroupPlan& p) const {
    const size_t g = (size_t)p.index;
    if (g >= resident_tokens_.size() || resident_tokens_[g] <= 0 || p.block_size <= 0) return 0;
    return ceil_div(resident_tokens_[g], p.block_size);
}

int64_t KVStreamStore::context_blocks(const KVGroupPlan& p, int64_t max_ctx,
                                      int64_t max_seqs) const {
    /* One spelling of the id space (kvstream/plan.h): the startup ring decision uses the same
     * arithmetic, so a ring that covers ctx_blocks covers what the store sizes. */
    return kv_context_blocks(p.block_size, max_ctx, max_seqs);
}

void KVStreamStore::set_map(int32_t group, const Map& m) {
    if (group < 0) return;
    if ((size_t)group >= maps_.size()) maps_.resize((size_t)group + 1);
    maps_[(size_t)group] = m;
}

const KVStreamStore::Map* KVStreamStore::map(int32_t group) const {
    if (group < 0 || (size_t)group >= maps_.size()) return nullptr;
    return &maps_[(size_t)group];
}

const RadKVStream* KVStreamStore::view(const KVGroupPlan* p, int32_t group) const {
    /* A group whose pool backs only a window of its context, and nothing else: a ring that covers
     * the context never reaches here (kvstream_ring leaves `group` at -1 and nothing is capped), so
     * such a group is not a streamed one and gets the base engine's null -- no resolve, no stage,
     * no mirror read (docs/KV-STREAMING.md §5 Phase 4). */
    if (!p || !p->streamed() || !windowed(group)) return nullptr;
    const Map* m = map(group);
    return m && m->host && m->res ? m : nullptr;
}

KVStreamCost KVStreamStore::cost(const std::vector<KVGroupPlan>& plans, int64_t max_ctx,
                                 int64_t max_seqs) const {
    KVStreamCost c;
    for (const auto& p : plans) {
        if (!windowed(p.index) || p.zero || p.kind != RAD_KV_FULL) continue;
        const int64_t blocks = context_blocks(p, max_ctx, max_seqs);
        const int64_t slots  = std::min(resident_blocks(p), blocks);
        if (blocks <= 0 || slots <= 0) continue;
        c.host_bytes += blocks * p.bytes_per_block;
        c.map_bytes  += kvstream_map_bytes(blocks, slots, p.n_layers);
        ++c.groups;
    }
    return c;
}

int KVStreamStore::alloc(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs,
                         MemAllocator& host_alloc, Pool& kv_pool) {
    for (const auto& p : plans) {
        if (!windowed(p.index) || p.zero || p.kind != RAD_KV_FULL) continue;
        const int64_t blocks = context_blocks(p, max_ctx, max_seqs);
        const int64_t slots  = std::min(resident_blocks(p), blocks);
        if (blocks <= 0 || slots <= 0) continue;

        Map m;
        m.slots             = slots;
        m.block_bytes       = p.layer_bytes_per_block();
        m.host_layer_stride = blocks * m.block_bytes;
        m.host_bytes        = blocks * p.bytes_per_block;
        m.host = static_cast<uint8_t*>(host_alloc.alloc(m.host_bytes, RAD_MEM_HOST_MAPPED));
        if (!m.host) {
            RAD_ERR("KV group '%s': the host mirror of %s (%lld blocks of %s) could not be "
                    "allocated", p.name.c_str(), humanb(m.host_bytes).c_str(), (long long)blocks,
                    humanb(p.bytes_per_block).c_str());
            release(host_alloc);
            return RAD_E_NOMEM;
        }
        /* 2 MiB pages for this mirror are already what the pinned allocator asks for
         * (device/aql.cpp, pin_pieces), before the pin faults the range in; the switch that turns
         * them off for the measurement is RADIANCE_NO_HUGE_PAGES there. Nothing to do here. */
        /* A block the prefix cache has not written yet is read as a miss and copied to the ring
         * before attention sees it, so the bytes must be defined: an uninitialised block would be
         * fluent wrong output with nothing to point at. */
        std::memset(m.host, 0, (size_t)m.host_bytes);

        /* One reservation for the group: the resolve's arrays are a few hundred KiB and separate
         * reserves would each round up to the pool's alignment. They come out of the KV pool, so
         * the paged split below sees them and the pool's report names them. */
        const int64_t layers = p.n_layers > 0 ? p.n_layers : 1;
        const MapLayout lay  = map_layout(blocks, slots, layers);
        const int64_t map_bytes = lay.bytes;
        std::string what = "kv_stream_maps:" + p.name;
        char* at = static_cast<char*>(kv_pool.reserve(what, map_bytes, RAD_ALIGN_UNIT, nullptr));
        if (!at) {
            RAD_ERR("KV group '%s': its residency map and ring bookkeeping (%s) did not fit the "
                    "KV pool", p.name.c_str(), humanb(map_bytes).c_str());
            host_alloc.free(m.host, RAD_MEM_HOST_MAPPED);
            release(host_alloc);
            return RAD_E_NOMEM;
        }
        /* The residency map is one slice a layer: a block's ring bytes are per layer, and the
         * resolve copies one layer's block, so a group-wide map would call a block resident for a
         * layer whose bytes were never copied (the first decode after a staged prefill). The
         * control block stays shared: its counters and overflow latch are the group's. */
        m.res        = reinterpret_cast<int32_t*>(at + lay.res);
        m.slot_block = reinterpret_cast<int32_t*>(at + lay.slot_block);
        m.slot_state = reinterpret_cast<int32_t*>(at + lay.slot_state);
        m.ctl        = reinterpret_cast<int32_t*>(at + lay.ctl);
        set_map(p.index, m);
    }
    return RAD_OK;
}

int KVStreamStore::init(const std::vector<KVGroupPlan>& plans, RadStream s) {
    if (!s) {
        RAD_ERR("KV streaming: the maps must be filled on a stream this engine owns; a null one "
                "is not usable");
        return RAD_E_INVAL;
    }
    for (const auto& p : plans) {
        const Map* m = map(p.index);
        if (!m || !m->host || !m->res) continue;
        /* The reset a resolve launch expects (libref/ref_kvstream.cpp's contract): both maps at
         * -1, the per-slot state and the control block at zero. There is no reset op because a
         * residency map lives as long as the pool. */
        const MapLayout lay = map_layout((int64_t)p.n_blocks, m->slots,
                                         p.n_layers > 0 ? p.n_layers : 1);
        int rc = RAD_OK;
        /* Every layer has its own map slice (see alloc): a block's bytes are copied per layer, so
         * residency is per layer too. A streamed group is by definition one whose pool is narrower
         * than its context, so every block starts absent and the first read of it is a miss. */
        rc = rad_memset_async(m->res, 0xff, lay.res_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->slot_block, 0xff, lay.slot_block_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->slot_state, 0, lay.slot_state_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->ctl, 0, lay.ctl_bytes, s);
        if (rc != RAD_OK) {
            RAD_ERR("KV group '%s': the residency map could not be initialised: %s",
                    p.name.c_str(), rad_dev_last_error());
            return rc;
        }
    }
    return RAD_OK;
}

void KVStreamStore::release(MemAllocator& host_alloc) {
    for (auto& m : maps_) {
        if (!m.host) continue;
        host_alloc.free(m.host, RAD_MEM_HOST_MAPPED);
        /* The device maps are carved from the KV pool and go with it. Clearing the view as well
         * makes a second call, or a teardown after a failed bring-up, a no-op. */
        m = Map{};
    }
}

KVStreamStats KVStreamStore::stats(const std::vector<KVGroupPlan>& plans) const {
    KVStreamStats st;
    st.misses   = misses_;
    st.bytes_in = bytes_in_;
    for (const auto& p : plans)
        if (p.streamed()) st.resident_blocks += p.pool_blocks();
    return st;
}

}  /* namespace rad */
