/* hostarena.h -- fixed-size slots of pinned host memory.
 *
 * Two stores are built from it: the KV tiers' block arena (core/mem/kvtier.h), where one slot is
 * one cache entry, and the linear-state snapshot store (KVManager::set_checkpoint_store), where
 * one slot is one snapshot. Under tensor parallelism a slot holds EVERY rank's shard of its
 * object, each rank at its own offset, so one slot id -- rank 0's to hand out -- names the object
 * on every rank.
 *
 * Pinned, because every byte of it is a PCIe transfer endpoint and a pageable one collapses: the
 * driver stages an async copy to pageable memory through its own buffer and, on some paths,
 * synchronises. Slot-allocated at one fixed size rather than malloc'd per object: every object a
 * store holds is the same size for a given model, so a bitmap is the whole allocator.
 *
 * RUNS, BECAUSE A TRANSFER IS PER RUN. A 4-token block makes a 200K-token conversation fifty
 * thousand entries, and a copy per entry is fifty thousand driver calls on the engine thread --
 * the call rate, not the link, is then what a restore costs. Slots taken together are given
 * consecutive ids wherever the arena has them, so a batch crosses the link as one strided copy
 * a run. */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

class HostArena {
public:
    ~HostArena();
    int  open(int64_t slot_bytes, int64_t cap_bytes);
    void close();
    bool enabled() const { return base_ != nullptr; }

    int64_t slot_bytes() const { return slot_bytes_; }
    /* Fixed when the arena is opened. Stored rather than counted from the bitmap, because an
     * HTTP thread reads it for the sessions view while the engine thread allocates and frees. */
    int64_t capacity()   const { return cap_slots_; }
    int64_t used_slots() const { return used_; }
    int64_t cap_bytes()  const { return slot_bytes_ * capacity(); }
    int64_t free_slots() const { return cap_slots_ - used_; }

    /* -1 when the arena is full. The caller evicts and retries; this does not choose victims,
     * because the policy that would choose them lives with the store's owner and an allocator
     * that reached into it would give each object two owners. */
    bool    full() const { return used_ >= cap_slots_; }
    int32_t alloc();
    /* UP TO `want` CONSECUTIVE SLOTS, the first free run at or after the cursor. `*got` is how
     * many; the caller asks again for the rest. -1 when the arena is full. */
    int32_t alloc_run(int32_t want, int32_t* got);
    void    free(int32_t slot);
    bool    in_use(int32_t slot) const { return slot >= 0 && slot < cap_slots_ && is_used(slot); }
    void*   at(int32_t slot) const;

private:
    bool    is_used(int64_t s) const { return (bits_[(size_t)(s >> 6)] >> (s & 63)) & 1ull; }
    void*   base_ = nullptr;
    int64_t slot_bytes_ = 0;
    int64_t used_ = 0;
    int64_t cap_slots_ = 0;
    int64_t cursor_ = 0;
    std::vector<uint64_t> bits_;
};

}  /* namespace rad */
