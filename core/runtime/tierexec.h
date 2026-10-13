/* tierexec.h -- the side of the KV tiers that actually moves bytes.
 *
 * core/mem/kvtier.h decides WHAT should move and hands out TierJobs; this executes them. The split
 * is the one KVManager already uses for BlockCopy and it is not decoration: core/mem has no stream
 * and must not acquire one, because a synchronisation on the step path is what this design spends
 * itself avoiding (spec §5.4).
 *
 * ============================== NOTHING HERE WAITS ON THE HOST ================================
 *
 * A copy out and a restore in are both ENQUEUED and ordered by events, never by a host wait:
 *
 *   - a copy out waits, on the transfer stream, for everything the compute stream had been given
 *     when it was issued -- the step that wrote the blocks is in there -- and signals an event the
 *     engine polls between steps. The engine thread goes on building steps the whole time, and the
 *     scheduler's lock is not held while it runs;
 *   - a restore in makes the COMPUTE stream wait for it, so the step that reads the restored
 *     blocks runs after they land without the host having waited for anything.
 *
 * ============================== THE TRANSFER, AND WHY IT IS SHAPED THIS WAY ====================
 *
 * A paged pool is LAYER-MAJOR: layer l of group g owns `[base + l*layer_stride, + layer_stride)`
 * and a block inside it is `block * layer_bytes_per_block()`. So a logical block is `n_layers`
 * fragments of a few hundred bytes, and moving one block at a time means a few hundred bytes a
 * transfer -- dispatch cost with the data as a rounding error.
 *
 * Turned around, WITHIN a layer the blocks are dense rows of a matrix. That makes packing an
 * arbitrary set of them exactly `gather_rows` and restoring them exactly `scatter_rows`, and the
 * transfer count becomes ONE LAUNCH PER (GROUP, LAYER) for a whole staging batch rather than per
 * block.
 *
 * THE STAGING BUFFER'S LAYOUT IS WHAT MAKES THE HOST COPY CONTIGUOUS. Device staging is
 * `max_batch * entry_bytes`, and one entry's bytes are [group][layer][fragment] laid end to end.
 * For a given (group, layer) the destination rows are then at a stride of `entry_bytes` from a
 * base inside the first entry -- which is exactly `gather_rows`'s `y` pitch, so one launch writes
 * every entry's fragment for that layer directly into the place it will be copied from. The host
 * side is then one strided copy per run of consecutive arena slots.
 */
#pragma once
#include "mem/kv.h"
#include "mem/kvtier.h"
#include "plugin/registry.h"
#include "rad_core.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class TierExec {
public:
    ~TierExec();
    /* `groups` is PrefixCache::cached_groups(): the paged groups an entry holds a block in, in the
     * order the entry's `blocks` vector uses. The staging layout is derived from them once here,
     * because a second derivation of the same offsets is a second place to get them wrong.
     *
     * `max_batch` is the staging buffer in entries, `max_copy` the most entries one copy out may
     * carry, and `max_chain` the longest chain one restore brings back -- the index buffers are
     * sized from the last two. */
    int  configure(Registry* reg, KVManager* kv, IdleTiers* tiers,
                   const std::vector<int32_t>& groups, int max_batch, int64_t max_copy,
                   int64_t max_chain, bool leader, int device, int64_t slot_off,
                   RadStream shared);
    void close();
    bool enabled() const { return enabled_; }
    std::string report() const { return report_; }

    /* THE SIZES A CONFIGURE IS GIVEN, from one entry's bytes on this rank, the longest context and
     * the block size, and the VRAM device_init() then takes for them. The VRAM budget reserves
     * device_bytes() before any of this exists and bringup configures with sizes(), so the two
     * cannot drift apart. */
    struct Sizes { int max_batch = 0; int64_t max_copy = 0, max_chain = 0; };
    static Sizes   sizes(int64_t entry_bytes, int64_t max_ctx, int64_t block_size);
    static int64_t device_bytes(int64_t entry_bytes, int n_groups, const Sizes& s);

    /* DECIDING AND DOING ARE SEPARATE, and under tensor parallelism that is the point. The policy
     * is one object describing CONVERSATIONS, which are not sharded; the transfer is per rank,
     * because each rank holds its own shard of every block. So the leader plans, every rank runs
     * the same job list, and the leader reports the outcome once.
     *
     * A follower that consulted the policy itself would find each record already pending -- or
     * already reported -- and would move nothing, silently, leaving its shard of the pool holding
     * whatever was there before. That is corrupt KV on one card and not the other, which is the
     * worst shape this failure can take. */

    /* ---------------- copy on idle ----------------
     *
     * Enqueue a batch of WriteBack jobs on this rank and return. The copy is ordered after
     * everything `compute` had been given, and `copy_state` says when it has landed. One batch
     * at a time: the engine does not issue the next until this one has reported, which is what
     * lets the staging and index buffers be reused without a wait. */
    int  issue_copy(std::vector<TierJob>& jobs, RadStream compute);
    /* 1 when nothing is out or the last batch has landed, 0 while it is in flight, negative when
     * the transfer stream reported an error. Taking the answer clears it. */
    int  copy_state();
    bool copy_out() const { return copy_out_; }
    /* copy_state, after waiting for the batch to land. For a caller that cannot go on until the
     * copy has a host slot to show for it -- an eviction that would otherwise lose the entry. */
    int  wait_copy();
    /* EVERYTHING THIS RANK ENQUEUED HAS LANDED. For a block given back on a failure path while a
     * copy into it may still be in flight: the pool would hand it to a sequence whose writes the
     * late copy then overwrites. */
    void quiesce();

    /* ---------------- restore ----------------
     *
     * Enqueue a chain of ToDevice jobs, in chain order, and make `compute` wait for them. The
     * leader chooses every block first, recalling the loan for the
     * whole chain at once; a chain the pool cannot hold is cut where the reservation ran out, and
     * the jobs past the cut are left unstarted. `ok[i]` is whether job i was fully enqueued on
     * this rank. */
    int  issue_restore(std::vector<TierJob>& jobs, RadStream compute, std::vector<char>* ok);
    /* Leader only. Gives back what a failed job held -- a restore's blocks -- and reports every
     * job to the policy. `now_ns` is when the pass ran: a promotion is a use. */
    void report_pass(std::vector<TierJob>& jobs, const std::vector<char>& ok, uint64_t now_ns);

    /* A lookup hit off-device: the job that would bring it back, or false when the entry is not
     * off-device or already has work outstanding. */
    bool plan_promote(const BlockHash& key, TierJob* out);           /* leader only */

    struct Stats { int64_t passes = 0, jobs = 0, bytes_out = 0, bytes_in = 0; };
    const Stats& stats() const { return st_; }

private:
    struct GroupInfo {
        int32_t  g = -1;
        int64_t  n_layers = 0;
        int64_t  frag = 0;          /* layer_bytes_per_block() */
        int64_t  layer_stride = 0;
        int64_t  n_blocks = 0;
        void*    base = nullptr;
        int64_t  off = 0;           /* this group's start inside one entry's staging bytes */
        /* rad_kv_group_zero: no shard on this rank. Kept in its place, because a job's block ids
         * are one per cached group in the same order on every rank; nothing is gathered or
         * scattered for it and it is no part of this rank's entry. */
        bool     zero = false;
        Resolved gather, scatter;
    };

    /* The staging buffer, the index buffers, the events and the transfer stream, built on the
     * first pass that has a job and never at bringup. The cost of building them eagerly is quoted
     * where it is defined; it is not small. */
    int  device_init();
    /* WHETHER EVERY ID THIS RANK IS ABOUT TO ADDRESS IS THE CACHE'S TO WRITE.
     *
     * THE KERNEL'S BOUND IS THE CARVE AND NOT THE LIVE MARK. gather_rows and scatter_rows refuse
     * an index outside [0, n_blocks), which is every id the carve created -- but the blocks above
     * live_blocks() are lent to the expert slab, and a restore into one writes over weights a
     * kernel is reading. That reads as fluent text with a new hash on whichever rank is behind
     * rather than as a failed transfer, so it is tested here, where the job can still be refused. */
    bool blocks_backed(const std::vector<int32_t>& blocks, const char* what);
    void warn_once(const char* what, int st);
    int  run_rows(const Resolved& r, bool scatter, const GroupInfo& gi, int64_t layer,
                  void* stage_base, const int32_t* d_idx, int64_t n_jobs);
    /* One strided copy per run of consecutive arena slots, `n` entries starting at job `first` of
     * `order`. To host when `out`, from host otherwise. */
    int  copy_runs(const std::vector<TierJob*>& order, size_t first, int64_t n, bool out);
    int  enqueue_copy(std::vector<TierJob>& jobs, RadStream compute);

    bool      enabled_ = false;
    bool      leader_ = false;   /* this rank chooses block ids; the others follow */
    bool      warned_ = false;
    /* Separate from `warned_`: a transfer that failed and an id this rank cannot address are
     * different reports, and the second is a broken invariant rather than a tier having trouble.
     * Said once, so a refusal the pass retries every tick does not fill the log with it. */
    bool      warned_unbacked_ = false;
    /* THE CARD THIS RANK OWNS. Every allocation and every launch below takes the CURRENT device,
     * and the engine loop leaves whatever device the last rank used current. Pinned here rather
     * than assumed. */
    int       device_ = -1;
    /* WHERE THIS RANK'S BYTES LIVE INSIDE A SHARED SLOT. One host slot holds an entry for the
     * WHOLE ENGINE, and under tensor parallelism an entry is n_ranks shards of KV that are not
     * interchangeable. Every rank writing at offset zero would mean the last one to run wins and
     * every rank scatters back somebody else's half. So the slot is n_ranks entries wide and a
     * rank only ever touches its own window. */
    int64_t   slot_off_ = 0;
    int64_t   slot_bytes_ = 0;   /* the whole shared slot: n_ranks * entry_bytes_ */
    Registry* reg_ = nullptr;
    KVManager* kv_ = nullptr;
    IdleTiers* tiers_ = nullptr;
    std::vector<GroupInfo> gi_;
    int64_t   entry_bytes_ = 0;
    int       max_batch_ = 0;
    int64_t   max_copy_ = 0;
    int64_t   max_chain_ = 0;
    void*     stage_ = nullptr;     /* device, max_batch_ * entry_bytes_ */
    /* ---- the block indices ----
     *
     * ONE REGION FOR THE COPY OUT AND A RING OF THEM FOR RESTORES, pinned on the host and mirrored
     * on the device. An index list is uploaded with the transfer it serves and read by kernels
     * later in the same stream, so a region may be refilled only once the upload that last read
     * it has run: one copy is out at a time, so its region is always free when the next is
     * issued; several restores can be issued in one step, so each takes the next region of the
     * ring and waits on that region's event -- the upload of a restore issued kRing restores ago,
     * which has long since run. */
    static constexpr int kRing = 4;
    int32_t*  idx_host_ = nullptr;  /* pinned: copy region, then kRing restore regions */
    int32_t*  idx_dev_ = nullptr;
    int64_t   copy_idx_ = 0;        /* entries a copy region holds, per group */
    int64_t   chain_idx_ = 0;       /* entries a restore region holds, per group */
    RadEvent  ring_ev_[kRing] = {};
    bool      ring_used_[kRing] = {};
    int       ring_next_ = 0;
    RadEvent  ev_compute_ = nullptr;  /* the compute stream, as a copy out found it */
    RadEvent  ev_copy_ = nullptr;     /* the last copy out, landed */
    RadEvent  ev_restore_ = nullptr;  /* the last restore, landed */
    bool      copy_out_ = false;
    /* THE TRANSFER STREAM, WHICH IS USUALLY SOMEBODY ELSE'S. A stream costs a hardware queue and
     * a fourth queue costs this card 65% of its decode step, so the tier takes the expert mover's
     * maintenance stream when there is one (Mover::transfer_stream) and only makes its own when
     * there is not. `own_stream_` is which of those happened, because close() must not destroy a
     * stream it was lent. */
    RadStream stream_ = nullptr;
    RadStream shared_ = nullptr;
    bool      own_stream_ = false;
    std::vector<TierJob*> order_;
    /* ---- RADIANCE_DEBUG_TIER_VERIFY ----
     *
     * DID THE BYTES SURVIVE THE ROUND TRIP. A tier that corrupts reads back as fluent, plausible,
     * wrong text and nothing about the engine says which of a dozen steps lost the bytes. This
     * hashes what went out and what came back and names the entry when they differ, which splits
     * the only question worth asking first: is the tier moving the wrong bytes, or is it moving
     * the right ones and something downstream is choosing differently?
     *
     * A DIAGNOSTIC, NOT A GUARD. It hashes on the host, which is a cost on every entry --
     * affordable when a run is being bisected and not otherwise. Off
     * unless the variable is set, and leader-only, because the map is keyed by an object the
     * followers do not own. A copy out is hashed when it has LANDED, which is when copy_state
     * first sees it done. */
    bool      verify_ = false;
    std::unordered_map<BlockHash, uint64_t, BlockHashHash> out_sum_;
    struct VerifyNote { BlockHash key; int32_t host_slot = -1; };
    std::vector<VerifyNote> verify_pending_;
    void      verify_note(const BlockHash& key, const void* p, int64_t n);
    void      verify_check(const BlockHash& key, const void* p, int64_t n);

    std::string report_;
    Stats     st_;
};

}  /* namespace rad */
