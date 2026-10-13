/* kvtier.h -- where a finished conversation's KV is kept once VRAM has a better use for it
 * (spec §7.3): VRAM, then host memory, and nowhere else.
 *
 * The prefix cache holds a KV reference on every block it indexes, so a finished turn's context
 * survives into the next one. That is the whole value of it, and it is also the whole cost: those
 * blocks are the most valuable thing in the pool while the session is live and the least valuable
 * thing in it once the session stops -- and on a card whose experts do not fit, VRAM holding an
 * idle conversation is VRAM every decode step pays for across the link.
 *
 * ============================== COPY ON IDLE, FREE ON DEMAND ====================================
 *
 * An entry is COPIED to host memory as soon as the request that published it finishes, and its
 * VRAM copy is kept. From then on the entry is CLEAN: the pool and the host hold the same bytes,
 * so giving the VRAM copy up costs nothing at all -- no transfer, no wait, only the cache's
 * reference -- and it is given up the moment something wants the memory: the expert slab's loan,
 * or an allocation the free list cannot serve. A conversation that comes back before either
 * happens resumes from VRAM for free; one that comes back after is restored from its host copy,
 * and KEEPS that copy, so the turn after it copies only the tokens that turn added.
 *
 *     published --(copy, at once)--> clean on device --(memory wanted)--> host
 *                                        ^                                  |
 *                                        +------------(a hit restores)------+
 *
 * HOST MEMORY IS THE LAST TIER. When a copy needs a slot and the arena is full, the oldest
 * entries only the host holds are dropped, as many as the copies refused. Nothing is written to a
 * drive: a tier there wears it at the rate the server prefills -- every block and every snapshot a
 * prompt produces -- to spare a re-prefill only for what outgrows VRAM and host memory together.
 *
 * AND EVERY ENTRY EXPIRES. One no request has used for --prefix-cache-ttl seconds leaves the cache
 * from whichever tier holds it (PrefixCache::expire); it is not moved anywhere first.
 *
 * THERE IS NO IDLE THRESHOLD ON THE WAY OUT OF VRAM, and that is the point of the design rather
 * than an omission. A threshold is a guess at when an entry stops being worth its VRAM, made in
 * seconds by an operator; a clean entry answers the question itself, because keeping it costs
 * nothing until the memory is wanted and releasing it costs nothing once it is.
 *
 * WHY MOVING BEATS DROPPING, stated once so the costs can be argued about in the right units:
 * dropping a 200K-token prefix costs a re-prefill, which is minutes. Restoring it costs a copy
 * over PCIe, which is a fraction of a second. The two are not the same order of magnitude.
 *
 * ============================== THIS OBJECT MOVES NO BYTES ======================================
 *
 * It decides WHAT should move and hands the work out as a list of TierJob. The engine executes
 * them on its transfer stream and reports back. That split is not decoration: `KVManager` already
 * works this way (BlockCopy, take_pending_copies) for the same reason -- core/mem has no stream,
 * and a synchronisation on the step path is what this design spends itself avoiding (spec §5.4).
 * A second mechanism for moving KV bytes would be a second place to get the pool's addressing
 * wrong.
 *
 * ============================== THE ADDRESSING, WHICH IS THE TRAP ==============================
 *
 * THE POOL IS LAYER-MAJOR (core/mem/kv.h). Layer i of a paged group owns
 * `[base + i*layer_stride, + layer_stride)`, and inside that a block is
 * `block_index * layer_bytes_per_block()`. A logical block is therefore NOT a contiguous extent:
 * it is `n_layers` fragments, one per layer, and on a production model a fragment is a few hundred
 * bytes. `base + block * bytes_per_block` addresses something else entirely -- roughly `n_layers`
 * consecutive blocks OF LAYER ZERO -- and a tier built on it moves other sequences' data and
 * restores a jumble. Corrupt KV does not read as babble; it reads as fluent, deterministic,
 * plausible, wrong text, and nothing short of a byte comparison against a known-good run finds it.
 *
 * So the unit of transfer here is not a block. It is A LAYER'S WORTH OF MANY BLOCKS, which is the
 * shape the pool is already in: within one layer the group's blocks are dense rows of
 * `layer_bytes_per_block()` each, so packing an arbitrary set of them is exactly `gather_rows`
 * and restoring them is exactly `scatter_rows`. One launch per (group, layer) per batch, however
 * many entries the batch holds.
 *
 * ============================== ONE RECORD, ONE TIER, ONE LIST ==================================
 *
 * An entry that can live in either of two tiers must not be on BOTH tiers' LRU lists: each tier's
 * eviction path then invalidates references the other holds, and every such pair is a
 * use-after-free. The rule here is that a key has exactly one Rec,
 * a Rec is in exactly one tier, and a Rec is in at most that tier's list. A CLEAN device record
 * is on no list: the copy it keeps off the card is what lets the VRAM copy go, and the tiers'
 * lists are for entries no longer in VRAM. A Rec with a transfer through the pool outstanding is
 * `pending` and is in no list either: it cannot be chosen as a victim, cannot be planned again,
 * and cannot be erased until the engine reports the job finished or failed.
 */
#pragma once
#include "mem/hostarena.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "rad_core.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rad {

enum class KVTier : uint8_t { Device = 0, Host = 1 };
const char* kv_tier_name(KVTier t);

/* What the engine is being asked to do. One job is one entry moving one step. The block ids are
 * carried in the job rather than looked up when it runs, because the whole point of `pending` is
 * that nothing may touch the record in between. */
struct TierJob {
    enum class Kind : uint8_t {
        WriteBack,      /* gather_rows out of the pool into a host slot; the entry stays in VRAM */
        ToDevice,       /* allocate blocks, scatter_rows in; the host copy is kept */
        DropHost,       /* the arena is full: this record's bytes go */
    };
    Kind      kind = Kind::WriteBack;
    BlockHash key;
    /* Per cached group, in PrefixCache::cached_groups() order. For a write-back these are the
     * blocks to read; for a promotion they are filled in by the engine with the blocks it
     * allocated. */
    std::vector<int32_t> blocks;
    int32_t   host_slot = -1;
    int64_t   bytes = 0;

    /* A PROMOTION THAT WAS NEVER STARTED, as opposed to one that failed part of the way through.
     * A restore reserves every block before it moves a byte, and a chain that did not fit is cut
     * where the reservation ran out: the entries past that point were not touched, their host
     * copies are exactly what they were, and they stay on the host tier. Only an entry whose
     * scatter may have run is dropped on failure. */
    bool      started = false;
};

/* ------------------------------------------------------------------ the policy */
class IdleTiers {
public:
    struct Config {
        int64_t host_cap_bytes  = 0;   /* 0 is off */
        int64_t entry_bytes     = 0;   /* one entry: sum over cached groups of bytes_per_block */
        /* ---- the linear half ----
         * `ckpt_bytes` is ONE SNAPSHOT, every rank's shard of it, laid out exactly like
         * `entry_bytes`. The snapshots live in the engine's checkpoint store, `ckpt_slots` deep,
         * which is reported here and not allocated: --prefix-cache-mib is the blocks' alone. */
        int64_t ckpt_bytes      = 0;
        int64_t ckpt_slots      = 0;   /* the checkpoint store's capacity, for the report and view */
    };

    int  configure(const Config& c);
    void close();
    bool host_on() const { return cfg_.host_cap_bytes > 0 && arena_.enabled(); }
    bool any_on()  const { return host_on(); }
    std::string report() const;

    /* Called by PrefixCache whenever an entry is inserted. The blocks come with it because a copy
     * needs them and the map is the only place they would otherwise be read from -- at which point
     * the tiers would hold a second copy of the cache's own bookkeeping and the two could
     * disagree. A record with no host copy is queued for one here: publishing is the moment a
     * request has stopped writing the entry, which is the moment it can be copied. */
    void note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns);
    /* THE HIT HALF OF THAT, for a lookup that matched `n` blocks of a chain. Only the idle clock
     * moves: a hit carries no new block ids, and it creates no record for an entry the tiers were
     * never told about. One lock for the whole chain, since a long prefix is thousands of blocks. */
    void note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns);
    /* The entry is gone from the cache for good -- evicted, or its blocks released. Any tier
     * storage it held is returned. A record with work outstanding is marked instead of erased and
     * is cleaned up when the job reports. */
    void forget(const BlockHash& key);
    /* AN OFF-DEVICE ENTRY WAS RECOMPUTED. A request that could not reach the host copy -- a
     * restore that did not fit, a chain past the point a promotion was cut -- computes the same
     * tokens again and publishes them, and the cache takes those blocks as the entry's device
     * copy. The tier copy goes: two copies of one entry computed by two different batches agree
     * only to the rounding the batch shape allows, and a record that called them the same bytes
     * would restore the older one over a pool that holds the newer. The record is a fresh device
     * record, queued for a copy of its own. */
    void rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns);

    /* WHERE IS IT. Device for anything this object has never been told about, which is the right
     * answer: an entry the cache holds and the tiers have not touched is in the pool. */
    KVTier tier_of(const BlockHash& key) const;
    bool   resident_off_device(const BlockHash& key) const;

    /* ---------------- copy on idle ----------------
     *
     * The records waiting for a host copy, up to `max_entries`, as WriteBack jobs with their host
     * slots already allocated. Oldest first, so a conversation that went idle before another is
     * copied before it. Blocks only: a snapshot is in host memory from the moment it is saved.
     *
     * Every record named by a returned job is `pending` until report() is called for it. What an
     * arena too full to take the batch cannot hold stays queued, and the next maintenance plan
     * drops that many of the oldest host-only entries. */
    void plan_writeback(int64_t max_entries, std::vector<TierJob>* out);
    bool writeback_waiting() const;
    /* Has the arena refused a copy since the last maintenance plan? That plan is then due now. */
    bool starved() const;

    /* THE VRAM COPIES THAT CAN GO, as the entries that became clean since the last call. The
     * cache decides whether each one actually goes -- a block a running sequence also holds would
     * free nothing -- and reports what it did through drop_commit. A key handed out here and not
     * dropped is the cache's to ask about again. */
    void take_clean(std::vector<BlockHash>* out);
    /* Can this entry's blocks leave VRAM for nothing, right now: its host copy has landed and no
     * job is out on the record. */
    bool droppable(const BlockHash& key) const;
    /* The cache released the blocks. The record is now a host record with its copy, on the host
     * tier's list like any other. Its snapshot stays in its slot: the store is not the pool, and
     * the blocks leaving frees nothing there. */
    void drop_commit(const BlockHash& key);

    /* ---------------- capacity ----------------
     *
     * The copies the arena refused since the last plan are made room for by dropping that many of
     * the oldest host-only entries, as DropHost jobs; `budget` bounds one plan. */
    void plan(int budget, std::vector<TierJob>* out);

    /* ---------------- expiry ----------------
     *
     * THE HOST RECORDS NO REQUEST HAS USED SINCE `cutoff_ns`, oldest first, up to `max`: what
     * --prefix-cache-ttl takes out of host memory. The cache drops each one (drop_offdevice) and
     * the record goes with it; a record with a job out is passed over and asked about again. */
    void expired(uint64_t cutoff_ns, size_t max, std::vector<BlockHash>* out) const;

    /* A promotion, asked for by a lookup that hit off-device. Returns false when the entry is not
     * in host memory or already has work outstanding. */
    bool plan_promote(const BlockHash& key, TierJob* out);

    /* The engine finished (or failed) a job. On failure the record returns to the state it was in
     * with its storage intact, EXCEPT for a failed ToDevice that had started, which cannot leave a
     * half-scattered pool behind and so drops the entry. A successful ToDevice restarts the
     * record's clock at `now_ns` when one is given: the promotion happened because something is
     * about to use it. */
    void report(const TierJob& job, bool ok, uint64_t now_ns = 0);

    /* WAS THIS KEY FORGOTTEN WHILE ITS JOB WAS OUT -- or is it not known at all. The engine asks
     * before acting on a finished job's blocks: an entry the cache evicted mid-transfer had its
     * references released by the eviction, and anything done in its name afterwards would be done
     * to blocks somebody else now holds. */
    bool doomed(const BlockHash& key) const;

    /* ============================ WHEN THE BYTES ARE GONE, SO IS THE ENTRY ====================
     *
     * A record erased while its bytes are OFF-DEVICE -- a capacity drop, a promotion that failed,
     * an expiry -- leaves the prefix cache holding an entry whose block ids the pool has already
     * given to somebody else. `resident_off_device` then answers false, because there is no
     * record any more, and the cache would serve those ids as an ordinary hit: one conversation
     * gets another's context, as fluent and plausible text.
     *
     * So the tiers tell the cache, at every site that erases an off-device record, and the cache
     * drops the entry WITHOUT releasing blocks -- there are none of its own left to release. */
    void set_drop_sink(std::function<void(const BlockHash&)> f) { on_drop_ = std::move(f); }

    /* ---------------- the linear half of an idle session ----------------
     *
     * Told by PrefixCache::commit_checkpoint and PrefixCache::retire_ckpt, so the session view can
     * say which conversations hold a snapshot and where a return resumes from. A snapshot lives in
     * its checkpoint store slot, in host memory, and never moves: the tiers only record it. A
     * record may hear of a snapshot BEFORE it hears note_used -- a checkpoint is committed
     * mid-request and the chain is published when the request closes -- so this creates the
     * record if it has to. */
    void note_snapshot(const BlockHash& key, int32_t slot, int64_t pos);
    /* The cache is done with the snapshot at `key`. The record's BLOCKS are untouched -- the two
     * halves are retired by different policies on different clocks, and a snapshot the retention
     * policy drops says nothing about the prefix. */
    void release_snapshot(const BlockHash& key);
    int64_t  ckpt_bytes() const { return cfg_.ckpt_bytes; }
    int64_t  ckpt_slots() const { return cfg_.ckpt_slots; }

    /* ---------------- the session view ----------------
     *
     * WHAT A SESSION IS HERE, because it is not what the cache is keyed by. The cache is keyed by
     * BLOCK, and a 200K-token conversation is twelve thousand of them -- an enumeration nobody can
     * read and not what anyone means by "session". A session is one PUBLISHED CHAIN: the blocks
     * one request inserted, named by its tip. That is exactly the leaf of the prefix trie, so two
     * conversations sharing a system prompt are two sessions and the shared blocks belong to both.
     *
     * THE SIZES THEREFORE OVERLAP AND THE TOTAL IS NOT A SUM. Sharing is the point of the cache;
     * a view that hid it would make a server look like it was storing several times what it is.
     * Anything reporting these numbers says so. */
    struct SessionInfo {
        uint64_t    id = 0;
        std::string model;
        std::string label;     /* the client's own name for the conversation, or "" */
        int64_t     n_tokens = 0;
        int64_t     bytes = 0;
        uint64_t    first_seen_ns = 0;
        uint64_t    last_used_ns = 0;
        int64_t     on_device = 0, on_host = 0;                /* blocks, by tier */
        /* LINEAR STATE, on a hybrid model. Counted separately from blocks because it is a
         * different resource with a different scarcity: a snapshot is thousands of blocks' worth
         * of bytes and there are only --checkpoint-slots of them in host memory. A session showing
         * blocks in host memory and no snapshot is one that will come back and re-run every
         * recurrent layer over its whole transcript, which is not what "restored" should mean. */
        int64_t     snapshots = 0;                             /* this session holds */
        int64_t     snap_bytes = 0;
        /* WHAT A RETURN REUSES: the token position of the deepest snapshot on the chain. On a
         * hybrid model the scheduler clamps a hit to the last linear checkpoint it can
         * reach, so this -- not n_tokens and not the blocks -- is the prefix the next turn skips,
         * and n_tokens minus it is what that turn prefills. Zero with no snapshot anywhere. */
        int64_t     resume_tokens = 0;
        bool        moving = false;                            /* a job is outstanding */
    };
    /* WHAT HOST MEMORY IS HOLDING FOR THE CACHE, each block counted once. The per-session figures
     * cannot be summed into this: sessions sharing a prefix share its blocks. The cap is what the
     * operator configured. A clean entry's host copy is counted: it occupies the arena whether or
     * not the VRAM copy has gone yet. */
    struct Occupancy {
        int64_t host_bytes = 0;
        int64_t host_cap_bytes = 0;
    };
    /* Publish one request's chain. Called from PrefixCache::insert, which is the only place that
     * knows a chain was completed rather than merely touched.
     *
     * KEPT WHETHER OR NOT ANY TIER IS ON, because what it records is not a tiering fact. A server
     * with no --prefix-cache-mib still stores conversations -- it stores them all in VRAM, which
     * is the case the table most wants to show -- and gating the record on the mover would make
     * the view report an empty server to every operator who has not opted into tiering.
     *
     * ONE CONVERSATION IS ONE ROW, and the rule that makes it one is the trie: a session is a
     * LEAF, so a chain that a later turn extends has stopped being one and is absorbed into it.
     * Each turn publishes the chain it grew under its own request id, so without that the table
     * is a row per turn -- a table nobody can read, holding a copy of every prefix of the
     * transcript for as long as the newest. */
    void note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                      uint64_t now_ns, const std::string& label = std::string());
    void set_model(const std::string& m) { model_ = m; }
    /* Newest activity first. Recomputes each session's tier histogram from the records, so it is
     * a view and never a second copy of the truth. The walk takes the lock in slices and lets
     * every other taker go first between them (YieldingMutex), so rows read at different slices
     * may describe moments a few microseconds apart. */
    std::vector<SessionInfo> sessions(Occupancy* occ = nullptr) const;
    void forget_session(uint64_t id);

    struct Stats {
        /* `to_host` is entries COPIED to host memory; `dropped` is entries whose VRAM copy was
         * then given up. They are different events: a copy is link traffic and a drop is VRAM
         * handed to something else, and a conversation that returns before anything wants its
         * VRAM is copied and never dropped. */
        int64_t to_host = 0, dropped = 0, promoted = 0;
        /* `host_drops` is entries the full arena gave up. */
        int64_t host_drops = 0, failures = 0;
        int64_t host_bytes = 0;
        int64_t host_hits = 0;
        /* WHAT THE STEP THREAD SPENT ON THEM, under the scheduler's lock: restores issued at
         * admission, and copies landed synchronously because pressure could not wait for them.
         * Host time, not link time -- what every other request waits through. */
        int64_t restores = 0, restore_ns = 0, flushes = 0, flush_ns = 0;
    };
    /* The step thread's own time on the tiers (see Stats). */
    void note_restore(int64_t ns) { std::lock_guard lk(mu_); ++stats_.restores; stats_.restore_ns += ns; }
    void note_flush(int64_t ns)   { std::lock_guard lk(mu_); ++stats_.flushes;  stats_.flush_ns += ns; }
    /* A copy, taken under the lock: an HTTP thread reads these while the engine thread moves
     * records. */
    Stats        stats() const;
    HostArena&   arena() { return arena_; }
    /* The size of ONE SLOT, which under tensor parallelism holds every rank's shard. */
    int64_t      entry_bytes() const { return cfg_.entry_bytes; }

private:
    struct Rec {
        KVTier   tier = KVTier::Device;
        uint64_t used_ns = 0;
        /* THE HOST COPY. On a device record it is a copy that makes the record clean; on a host
         * record it is where the bytes are read from. */
        int32_t  host_slot = -1;
        bool     pending = false;
        bool     doomed  = false;   /* forget() arrived while pending */
        bool     queued  = false;   /* in wb_q_: waiting for a copy */
        bool     clean_q = false;   /* in clean_q_: a drop the cache has not been asked about */
        bool     listed  = false;   /* on host_lru_; `it` is valid */
        std::vector<int32_t> blocks;
        /* ---- the linear half ----
         * `ck_slot` is the snapshot's checkpoint store slot and `ck_pos` its token position, for the
         * session view. -1 when the record carries no snapshot, which is the common case -- one
         * block in every (interval / block_size) is a checkpoint position. */
        int32_t  ck_slot = -1;
        int64_t  ck_pos = 0;
        std::multimap<uint64_t, BlockHash>::iterator it;
    };

    /* Does this record want a host copy it does not have yet? Its blocks are in VRAM with none. */
    bool wants_copy(const Rec& r) const;
    void queue_copy(const BlockHash& key, Rec& r);
    void queue_clean(const BlockHash& key, Rec& r);
    /* GIVE UP A HOST SLOT, the record's or a finished job's. Every block-arena free goes through
     * `give_slot`, because a freed slot is what unblocks a write-back that found the arena full.
     * `free_host` is the record's own. */
    void give_slot(int32_t slot);
    void free_host(Rec& r);
    void unlist(Rec& r);
    void relist(const BlockHash& key, Rec& r);

    struct Session {
        int64_t  n_tokens = 0;
        uint64_t first_seen_ns = 0;
        uint64_t last_used_ns = 0;
        std::vector<BlockHash> chain;
        std::string label;
    };

    /* Erase one row and its root index entry together. */
    void drop_session(uint64_t id);
    /* Every row whose chain starts at `key`, which has gone for good. */
    void drop_rooted(const BlockHash& key);
    /* The body of close(), for a caller that already holds the lock. */
    void reset();

    /* TWO THREADS REACH THIS OBJECT. The engine loop owns every mutation -- the cache publishes a
     * chain from the scheduler's close path, the maintenance pass moves records -- and an HTTP
     * thread drawing the sessions view walks the same containers through sessions(). An
     * unordered_map rehashed under an iterator is a crash rather than a stale number, so the
     * state the view reads is held under this and every mutator takes it.
     *
     * NOT A BOTTLENECK, and it is worth saying where the cost lands: the per-block calls on the
     * publish path (note_used) take it once each, uncontended, against a hash insert they were
     * already paying for. No method here calls another that takes it, which is what keeps a plain
     * mutex sufficient; the drop sinks are called while it is held and reach into the prefix
     * cache, which never calls back.
     *
     * AND THE VIEW'S WALK MUST NOT DELAY THE ENGINE. It visits every block of every stored
     * conversation, which is far longer than any engine-side hold, and the engine takes this lock
     * every step. So the walk takes it for a slice of records at a time and, between slices,
     * waits until nobody else is waiting for it: an engine-side taker that arrives during a slice
     * waits for that slice and never for the walk. A plain mutex cannot promise that, because
     * the walker re-locking the moment it unlocks can win against a woken waiter every time. */
    struct YieldingMutex {
        std::mutex       m;
        std::atomic<int> waiting{0};
        void lock() {
            waiting.fetch_add(1, std::memory_order_relaxed);
            m.lock();
            waiting.fetch_sub(1, std::memory_order_relaxed);
        }
        void unlock() { m.unlock(); }
        /* The view's: taken only while no other taker is waiting. */
        void lock_after_others() {
            for (;;) {
                while (waiting.load(std::memory_order_relaxed) > 0) std::this_thread::yield();
                m.lock();
                if (waiting.load(std::memory_order_relaxed) == 0) return;
                m.unlock();
            }
        }
    };
    mutable YieldingMutex mu_;

    Config   cfg_;
    HostArena arena_;
    std::unordered_map<BlockHash, Rec, BlockHashHash> recs_;
    /* THE HOST-ONLY RECORDS BY LAST USE, begin() the least recently used. Keyed by the clock rather
     * than kept in the order records arrive, because a record arrives here when its VRAM copy is
     * given up, which says nothing about when a request last read it -- and both the capacity drop
     * and the expiry take the oldest by use. */
    std::multimap<uint64_t, BlockHash> host_lru_;
    /* THE WORK QUEUES, in the order the work arrived. Each holds a key at most once -- the Rec's
     * flag says whether it is in -- and a key whose record changed under it is skipped when it is
     * reached rather than searched for, so both stay a push and a pop. */
    std::deque<BlockHash> wb_q_;       /* records wanting a host copy */
    std::vector<BlockHash> clean_q_;   /* records whose VRAM copy may now go */
    /* COPIES THE ARENA REFUSED since the last maintenance plan, which is what that plan frees
     * room for. A count and not a list: any host-only entry makes room for any copy. */
    int64_t  starved_ = 0;
    /* A WRITE-BACK FOUND THE ARENA FULL AND NOTHING TO GIVE UP. Waiting again changes nothing until
     * a slot is freed, so the copy is not planned until one is -- otherwise every tick pops the
     * queue, finds no slot and puts it back. */
    bool     wb_stuck_ = false;
    std::unordered_map<uint64_t, Session> sessions_;
    /* THE DEATH SIGNAL FOR A ROW, and the reason the table is bounded. A published chain is only
     * ever reachable from position zero, so the moment the cache evicts its FIRST block the
     * conversation cannot be hit at any length and the row describes blocks that are not there.
     * The cache announces its evictions and expiries through forget(), and the entries this object
     * drops itself -- a full arena's, a failed restore's -- go through drop_rooted as well, so the
     * root is indexed and the row goes with it. Several sessions share a root when
     * they share a system prompt and all of them are unreachable together, which is why the value
     * is a list. */
    std::unordered_map<BlockHash, std::vector<uint64_t>, BlockHashHash> roots_;
    std::string model_;
    std::function<void(const BlockHash&)> on_drop_;
    Stats    stats_;
    std::string report_;
};

}  /* namespace rad */
