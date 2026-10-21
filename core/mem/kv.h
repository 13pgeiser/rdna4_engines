/* kv.h -- the KV block manager (spec §7.2).
 *
 * Paged per KV group; each group has its own block geometry, pool and slot mapping. Block size
 * comes from KVGroupInfo::block_size, taken from the resolved attention kernel's constraints,
 * never from a core constant, so a kernel change needs no core edit.
 *
 * Four state kinds (hybrid models are the normal case):
 *
 *   RAD_KV_FULL    paged, per-token, growing
 *   RAD_KV_WINDOW  paged, per-token, bounded by the window; blocks below the window are recycled
 *   RAD_KV_LINEAR  per-sequence, fixed size: the GDN recurrent state
 *   RAD_KV_CONV    per-sequence, fixed size: a rolling window of conv_width-1 + n_spec entries
 *
 * The two stateful kinds are also allocated, mapped and freed here, so one allocator decides what
 * a sequence owns. With two, a freed sequence's state slot could still be handed to the next
 * request by the other.
 */
#pragma once
#include "mem/hostarena.h"
#include "mem/pools.h"
#include "rad_core.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ group sizing */
/* Layer bundling compared with vLLM. vLLM splits a hybrid model's layers into equal-sized groups
 * and pads each bundle to a multiple of that size with real memory: on a 48 linear / 16 full /
 * 5 drafter model its rule picks size 5, giving 15 groups x 5 = 75 slots for 69 layers and a
 * per-request round-up per group. A least-waste search recovers part of that.
 *
 * Here each state kind is one group with its own pool (spec §7.2), so there is no uniform page
 * size to pad to: padding is zero and the group count is minimal. This struct lets
 * `--debug-placement` print the comparison and lets a test assert it. */
struct BundleComparison {
    std::vector<int64_t> bundles;        /* layers per state kind, as declared */
    int64_t stock_group_size = 0;        /* vLLM's rule: the smallest bundle (with the 1.5x escape) */
    int64_t stock_groups = 0;
    int64_t stock_slots = 0;             /* allocated layer-slots, padding included */
    int64_t patched_group_size = 0;      /* the least-waste group size for the same rule */
    int64_t patched_groups = 0;
    int64_t patched_slots = 0;
    int64_t our_groups = 0;              /* == bundles.size() */
    int64_t our_slots = 0;               /* == sum(bundles); no padding */
    std::string report() const;
};
BundleComparison compare_bundling(const std::vector<int64_t>& bundles);

/* ------------------------------------------------------------------ per-group plan */
struct KVGroupPlan {
    int32_t     index = -1;            /* position in Program::kv_groups */
    std::string name;
    int         kind = RAD_KV_FULL;
    int64_t     n_layers = 0;

    int64_t     block_size = 0;        /* tokens per block, from the resolved kernel */
    int64_t     bytes_per_block = 0;   /* covers every layer bound to the group */
    int64_t     bytes_per_state = 0;   /* LINEAR/CONV: per sequence, every layer */
    int64_t     conv_slots = 0;        /* CONV: conv_width - 1 + n_spec */
    int64_t     window_blocks = 0;     /* WINDOW: the per-sequence block cap */
    /* WINDOW: a checkpoint slot carries this group's window (has_window_snapshot). Set only for a
     * trunk group of a model with no linear state. A drafter's window is recomputed by draft
     * refill, and a model with linear state re-prefills a hit on a windowed trunk, so neither
     * snapshot would be read. */
    bool        win_snap = false;

    int64_t     n_blocks = 0;          /* paged groups: blocks carved from the pool */
    /* rad_kv_group_zero: this rank keeps no cache for the group. The plan keeps its position and
     * block count (one block table serves every rank); its costs are zero and nothing is carved.
     * No op on this rank names it (issue refuses a group with no pool bound). */
    bool        zero = false;
    /* rad_kv_group_replicated: every rank holds the same bytes, so a tier slot carries this
     * group once, in a window at its front that rank 0 writes and every rank reads
     * (TierExec::slot_layout). The pool is carved as for any group. */
    bool        replicated = false;
    int64_t     n_states = 0;          /* LINEAR/CONV: n_seq_states * state_copies */
    /* Physical slots per logical state: 1, or 2 under speculation. A verify runs 1 + n_spec
     * tokens through the recurrence before acceptance is known; libr4d keeps the committed state
     * in copy 0 and the per-token replay factors in copy 1, and folds the accepted prefix in at
     * the next step. The factors (a few hundred floats a head, against a 128x128 state) fit in
     * one slot, so the pool grows 2x rather than (1 + n_spec)x. Copy k of logical slot s is
     * physical slot s + k * n_seq_states. */
    int64_t     state_copies = 1;
    int64_t n_seq_states() const { return state_copies > 0 ? n_states / state_copies : n_states; }
    int64_t     pool_offset = 0;
    int64_t     pool_bytes = 0;
    void*       base = nullptr;

    bool paged() const { return kind == RAD_KV_FULL || kind == RAD_KV_WINDOW; }
    bool stateful() const { return kind == RAD_KV_LINEAR || kind == RAD_KV_CONV; }

    /* The pool is layer-major, as the ABI requires: a RAD_OPK_KV operand carries the layer as its
     * offset and resolves to `base + layer * layer_stride` (RadOperand in rad_runtime.h,
     * KVPoolBinding in core/runtime/ctx.h), so a kernel gets one base pointer per layer and
     * indexes it with the block table and slot mapping.
     *
     * Layer i owns [base + i*layer_stride, + layer_stride): n_blocks blocks of
     * `bytes_per_block / n_layers` for a paged group, n_states states of
     * `bytes_per_state / n_layers` for a stateful one. */
    int64_t layer_bytes_per_block() const {
        return n_layers > 0 ? bytes_per_block / n_layers : bytes_per_block;
    }
    int64_t layer_bytes_per_state() const {
        return n_layers > 0 ? bytes_per_state / n_layers : bytes_per_state;
    }
    int64_t layer_stride() const {
        return paged() ? n_blocks * layer_bytes_per_block() : n_states * layer_bytes_per_state();
    }
};

/* A copy the caller must execute on its copy stream before the destination block is read. The
 * manager has no stream and issues no device copies, since a memcpy on the step path would
 * synchronise (spec §5.4). */
struct BlockCopy {
    int32_t group = -1;
    int32_t src_block = -1;
    int32_t dst_block = -1;
};

/* A state slot that still holds the previous tenant's state and must be zeroed before first use.
 * The manager names the work and the engine issues it, as with BlockCopy.
 *
 * A paged sequence reads only the blocks its table names, so a recycled block is unreachable
 * until written. A recurrent slot has no table: `gdn_chunk_scan` reads `h0` from the slot and has
 * no operand for "no initial state", so an uncleared slot continues the previous sequence's
 * history. Conv slots carry `has_init` but are cleared too, so one rule covers the pool. */
struct StateClear {
    int32_t group = -1;
    int32_t slot  = -1;
};

/* ------------------------------------------------------------------ manager */
class KVManager {
public:
    /* Sizes every group out of one pool, refusing by name and shortfall if the fixed costs alone
     * do not fit. The paged groups are sized to run out at the same total token count; any other
     * split strands memory in whichever group outlasts the others.
     *
     * The budget is whatever `kv_pool` has left (vram_kv), as the budget resolver set it.
     * Linear-state snapshots live in pinned host memory (set_checkpoint_store), so the pool holds
     * only the live cache (spec §6, §7.3). */
    int configure(const std::vector<KVGroupInfo>& groups, const Config& cfg, Pool& kv_pool);

    /* configure() in two steps for the budget resolver. plan_groups() sizes every group from the
     * declaration and run configuration (no pool), and ceiling() gives the cost of a full cache
     * for `ctx` tokens on max_seqs sequences. Pools are allocated before carve(), so the KV pool
     * is sized and capped from these two alone (core/mem/vram_budget.cpp); VRAM above the ceiling
     * would never be addressed. carve() spends the pool it is given. */
    int     plan_groups(const std::vector<KVGroupInfo>& groups, const Config& cfg);
    int64_t ceiling(int64_t ctx) const;
    /* The context-independent part of every ceiling: recurrent and conv state for max_seqs
     * sequences, plus alignment padding. Set by plan_groups(). */
    int64_t fixed_bytes() const { return fixed_; }
    /* The token count the paged groups settle on for `budget_bytes` of pool (carve()'s split,
     * without carving). One block table serves every rank, so all ranks must hold the same block
     * count, but under an uneven tensor-parallel split their pools differ. The engine asks every
     * rank and passes the least to cap_paged_tokens() before carve(); 0 is no cap.
     * kUnboundedTokens when every paged group on this rank is zero (rad_kv_group_zero): this rank
     * then limits nothing. */
    static constexpr int64_t kUnboundedTokens = INT64_MAX;
    int64_t paged_tokens(int64_t budget_bytes) const { return paged_split(budget_bytes, nullptr); }
    /* Inverse of paged_tokens: the least whole MiB of pool whose paged split reaches `tokens`,
     * searched up to max_mib; -1 if max_mib does not reach it. Pure arithmetic on the plan, used
     * by Engine::plan to size a cheaper rank to the token count of its dearest peer. Exact,
     * because paged_split is monotone in the budget. */
    int64_t pool_mib_for_tokens(int64_t tokens, int64_t max_mib) const;
    void    cap_paged_tokens(int64_t tokens) { tokens_cap_ = tokens > 0 ? tokens : 0; }
    int     carve(Pool& kv_pool);

    const std::vector<KVGroupPlan>& plans() const { return plans_; }
    const KVGroupPlan* plan(int32_t group) const;
    int32_t n_groups() const { return (int32_t)plans_.size(); }
    std::string report() const;

    /* ---------------- sequences ---------------- */
    int  add_sequence(uint64_t seq);
    bool has_sequence(uint64_t seq) const;
    void free_sequence(uint64_t seq);
    /* Count of sequence frees and preemptions over the manager's life. These are the only
     * refcount drops the prefix cache cannot see, so a change tells it a block it saw shared may
     * no longer be. */
    uint64_t seq_frees() const { return seq_frees_; }

    /* Grow the sequence's mapping to cover `n_tokens` in every group. Unshares a shared partial
     * block first (copy-on-write) and appends the copy request to the pending list.
     * RAD_E_FULL when a group's free list is empty -- the caller preempts and retries.
     *
     * `keep_from` is the position of the first query the coming step runs (the scheduler's
     * n_computed); -1 means the sequence's current end. A WINDOW group recycles only blocks below
     * keep_from - window + 1, not below the window of the step's end, which would drop keys the
     * first rows of a multi-block prefill chunk still read. */
    int  ensure(uint64_t seq, int64_t n_tokens, int64_t keep_from = -1);

    /* A WINDOW group's caps. A sequence holds the window behind the step's first query plus the
     * step's own tokens (at most window_blocks + ceil(max_tok / block) + 1), and one step carries
     * at most max_tok tokens in total, so the pool needs max_seqs windows plus one step. */
    int64_t window_seq_cap(const KVGroupPlan& p) const;
    int64_t window_pool_cap(const KVGroupPlan& p) const;
    /* ceil(max_tok / block): the blocks one step's tokens can fill. */
    int64_t step_blocks(const KVGroupPlan& p) const;

    /* Recycles a committed chunk's window blocks. ensure() keeps a chunk's blocks beside the
     * window and only the next ensure() recycles them, so a prefill waiting for its next chunk
     * would hold a whole chunk of blocks; several such prefills can exhaust the pool. The
     * scheduler calls this when it advances n_computed, passing the new n_computed. */
    void trim_window(uint64_t seq, int64_t keep_from);

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a whole multiple of the group's
     * block size: a partial block is never shared, because the next token written into it would
     * corrupt the cache entry. Bumps each block's refcount. */
    int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks, int64_t n_tokens);

    /* The child shares every whole block of the parent and copies the partial one on first write
     * (spec §7.3). A recurrent state cannot be shared, so the child gets its own slot and the
     * caller must restore a snapshot into it. */
    int  fork(uint64_t parent, uint64_t child);

    std::vector<BlockCopy> take_pending_copies();

    /* Prefix-cache and tier access to blocks. The cache holds a reference on every block it
     * indexes, so a cached block is never in the free list and cannot be handed to another
     * sequence under a cache entry. Nothing else may call these. */
    /* Takes a free block with no sequence attached, for the idle-session tiers: a restore needs
     * somewhere to put bytes before any sequence asks for them, and the prefix cache takes the
     * reference afterwards. The holder must release_block() if the restore fails. Does not
     * recall the loan; it stops at what the cache keeps in hand. */
    int  reserve_block(int32_t group, int32_t* out);
    /* Up to `n` blocks appended to `out`; returns how many. Recalls the loan for all of them up
     * front, since a block-at-a-time recall would drain the slab one buffer at a time. */
    int64_t reserve_blocks(int32_t group, int64_t n, std::vector<int32_t>* out);
    /* Takes one particular block another rank already chose. Block ids are shared across ranks
     * (RadBatch carries one table), so under tensor parallelism rank 0 picks and the others take
     * the same id. Refuses (RAD_E_STATE) a held block or one above the live mark rather than
     * substituting, since a rank restoring into a different block would corrupt only that rank's
     * output. */
    int  reserve_specific_block(int32_t group, int32_t block);
    /* The cache's reference on a block a sequence already holds. The block is marked as the
     * cache's for cache_only_blocks(). */
    int  retain_block(int32_t group, int32_t block);
    /* A reserved block's reference handed to the cache as it stands: marked, not counted again. */
    void cache_reserved_block(int32_t group, int32_t block);
    /* Releases the cache's reference on a marked block, or a reservation's on any other. */
    void release_block(int32_t group, int32_t block);
    int32_t block_refcount(int32_t group, int32_t block) const;
    /* Blocks only the cache holds: marked, at refcount one. This is what an eviction would free.
     * Maintained at every refcount change because it is read under the scheduler's lock, where a
     * walk of the cache would stall the step. */
    int64_t cache_only_blocks(int32_t group) const;

    const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const;
    int32_t state_slot(uint64_t seq, int32_t group) const;
    int     state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const;
    int64_t state_index_width(int32_t group) const;

    /* Slots handed to a sequence since the last drain, oldest first. The engine zeroes each on the
     * compute stream before the step that first reads it and then calls clear_state_clears(). */
    const std::vector<StateClear>& state_clears() const { return clears_; }
    void clear_state_clears() { clears_.clear(); }
    int64_t seq_tokens(uint64_t seq) const;

    /* ---------------- accounting ---------------- */
    int64_t free_blocks(int32_t g) const;

    /* Blocks of a group that something holds. Use this for occupancy, not `carve - free`:
     * free_blocks() is bounded by the live mark while the cache lends, so that difference would
     * count lent blocks as occupied. */
    int64_t held_blocks(int32_t g) const;

    /* The carve: the range every block id is addressed against. live_blocks() is what the cache
     * keeps unlent. */
    int64_t total_blocks(int32_t group) const;
    int64_t free_states(int32_t group) const;

    /* Pool size in bytes, the only unit in which groups can be summed: paged and stateful groups
     * count different objects, and paged groups share one token axis.
     *
     * pool_bytes() excludes lent blocks, which the expert slab reports as its own. Used bytes
     * count held blocks, owned by a sequence or referenced only by the prefix cache; a block
     * stops counting when it reaches the free list. */
    int64_t pool_bytes(int32_t group) const;
    int64_t pool_bytes_used(int32_t group) const;
    /* Bytes the budget carved for the group; the difference from pool_bytes() is lent to the
     * expert slab. */
    int64_t pool_bytes_carved(int32_t group) const;
    int64_t blocks_for(int32_t group, int64_t n_tokens) const;

    /* Whether a sequence of this length fits an empty pool. If not, preemption cannot help and
     * the request is failed rather than deadlocking the engine (spec §17). */
    bool can_ever_fit(int64_t n_tokens) const;

    /* ---------------- the loan ----------------
     *
     * The paged groups lend unheld blocks to the expert slab in place. The carve fixes and backs
     * every address once and nothing here unmaps it; the loan only lowers how many of a group's
     * blocks the cache may hand out (the live mark). The slab keeps expert units in the blocks
     * above the mark in every layer's stripe, so the MoE GEMM need not read them across the link.
     * No memory is allocated or freed either way, so the loan makes no driver call.
     *
     * The loan is in tokens, counted down from the top of the carve. FULL groups are sized to run
     * out at the same token count, so a loan of T tokens is the top floor(T / block_size) blocks
     * of every FULL group. WINDOW groups are sized per sequence and never lent.
     *
     * A held block (owned by a sequence or indexed by the prefix cache) is never lent, so a loan
     * evicts nothing. The min-heap free list keeps free blocks at the top, where a loan reaches
     * them.
     *
     * The buffer is the free blocks the cache keeps in hand so a request finds room without a
     * recall: the larger of one prefill chunk (the most one step consumes) and the largest recent
     * shortfall. The shortfall term decays, so an idle server settles at the floor. */
    int64_t buffer_bytes() const;

    /* True for the FULL paged groups only. */
    bool    lendable(int32_t group) const;

    /* Tokens the cache can lend now: over the lendable groups, the least distance from the top of
     * the carve down to the highest held block plus the buffer. Rank 0 only (a follower tracks no
     * held blocks). Also decays the buffer's shortfall term, so call it once per arbiter tick. */
    int64_t spare_tokens();

    /* Lends `tokens` from the top, or takes back what a smaller number no longer covers. Lending
     * past a held block is refused (RAD_E_STATE). Taking back always succeeds here; the caller
     * must have moved the slab out of those blocks first. */
    int     set_loan(int64_t tokens);
    int64_t loan_tokens() const { return loan_; }

    /* Recall hook. A request needing more blocks than the cache has in hand calls it with the
     * loan it can still afford, and the slab owner drains to that before returning, so admission
     * is decided against the whole carve. Without a hook, lent blocks are not recalled and the
     * request gets RAD_E_FULL. */
    void    set_reclaim(std::function<int(int64_t tokens)> f) { reclaim_ = std::move(f); }

    /* Copies the leader's loan onto a follower rank. Block ids are shared across ranks and only
     * rank 0 tracks sequences, so a follower takes the leader's mark instead of computing one;
     * its slab occupies the same block ids of its own pool. */
    int     follow(const KVManager& leader);

    /* Blocks of a group the cache may hand out: the carve less the loan. */
    int64_t live_blocks(int32_t group) const;

    /* ---------------- preemption by recompute (spec §7.1, §17) ---------------- */
    /* Frees the sequence's blocks and state slots and resets it to zero tokens. A recurrent
     * state that was not snapshotted cannot be resumed, so the sequence replays from its last
     * checkpoint or from zero. No "needs re-prefill" flag is kept: the scheduler's
     * `r.n_computed = 0` (preempt_back) drives the replay, and seq_tokens() == 0 reflects it. */
    int  preempt(uint64_t seq);

    /* ---------------- speculative rejection (spec §10) ---------------- */
    /* Shrinks the sequence to `n_tokens` and frees the paged blocks past it. A verify writes KV
     * for 1 + n_spec positions and the sampler accepts a prefix; `ensure` is monotone and cannot
     * undo the rest.
     *
     * Paged groups only. The linear state and conv window have already absorbed the rejected
     * tokens: the conv cache is a rolling window of conv_width-1 + n_spec entries, and the kernels
     * handle a rejection by moving the read offset (core/sample/accept.h). Freeing a state slot
     * here would destroy the state that offset points into. Unlike preempt, the sequence keeps
     * its state slots and window base. */
    int  rollback(uint64_t seq, int64_t n_tokens);

    /* The absolute token position block_table(seq, group)[0] covers: zero for a FULL group, the
     * recycled prefix for a WINDOW group. The slot mapping is (pos - first_block_pos) /
     * block_size; without the offset a sliding-window sequence writes to the wrong slot once it
     * passes its window. */
    int64_t first_block_pos(uint64_t seq, int32_t group) const;

    /* The first table entry changed since the last call; entries below it are unchanged, and
     * kNoDirty means none changed. The call clears the mark, so it has one reader: the batch
     * builder, which stages only the changed tail. An unseen sequence returns 0. */
    int64_t take_table_changes(uint64_t seq, int32_t group) const;
    static constexpr int64_t kNoDirty = INT64_MAX;

    /* Free victims, in the order given, until `seq` can be extended to `n_tokens`.
     *   RAD_OK        room was made; *preempted lists who paid for it
     *   RAD_E_FULL    even an empty pool cannot serve it; fail the request, do not retry
     * The caller supplies the victim order, since priority is the scheduler's policy. */
    int  make_room(uint64_t seq, int64_t n_tokens,
                   const std::vector<uint64_t>& victims,
                   std::vector<uint64_t>* preempted);

    /* ---------------- linear-state checkpoints (spec §7.3) ----------------
     *
     * Snapshots live in pinned host memory: no kernel reads them, they are only copied back into
     * a live state slot. A save is one copy across the link after the step that produced the
     * state, a restore one copy back before the step that resumes from it (a few ms for a
     * 50-100 MiB snapshot).
     *
     * The store is owned by the engine and shared by every rank: a slot holds each rank's shard
     * at its own offset (`rank_off`). Rank 0 hands out slot ids and the other ranks use them.
     * Without a store no slot is handed out and a hybrid model's prefix hits replay their linear
     * layers from zero. `checkpoint_bytes()` is this rank's shard. */
    void    set_checkpoint_store(HostArena* store, int64_t rank_off);
    int64_t checkpoint_bytes() const { return checkpoint_bytes_; }
    int64_t checkpoint_slots() const { return ck_store_ ? ck_store_->capacity() : 0; }
    int64_t free_checkpoint_slots() const { return ck_store_ ? ck_store_->free_slots() : 0; }
    int  alloc_checkpoint(int32_t* slot_out);
    void free_checkpoint(int32_t slot);
    void* checkpoint_ptr(int32_t slot) const;
    /* The store holds `slot`: it was allocated and not given back since. */
    bool  checkpoint_held(int32_t slot) const;

    /* Snapshot and resume a sequence's whole linear state. On a hybrid model a prefix hit carries
     * the attention half in adopted blocks and the recurrent half only through these calls;
     * without them a hit resumes from whatever the slot last held, which still reads as fluent
     * text.
     *
     * The pool is layer-major, so the blob is a gather: for each stateful group in `plans()`
     * order, for each layer, `layer_bytes_per_state()` bytes. Both directions walk it the same
     * way; a checkpoint is only read back by the same build. Only copy 0 (the committed state) is
     * saved; the other copies hold replay factors for unaccepted tokens.
     *
     * Enqueued on `s`, so stream order puts a save after the step that produced the state and a
     * restore before the step that reads it. One pitched copy per stateful group. RAD_E_INVAL on
     * a bad slot or an unknown sequence. */
    int checkpoint_save(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s,
                        const std::vector<int32_t>* win = nullptr);
    int checkpoint_restore(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s,
                           const std::vector<int32_t>* win = nullptr);

    /* ---------------- window snapshots (the sliding-window half of a prefix hit) ----------------
     * A WINDOW group keeps only the blocks its window can still read, so the prefix cache's block
     * lists do not describe it. A prompt extending a cached prefix at position P needs blocks
     * [lo, P/B), lo = max(0, P/B - (window_blocks - 1)), the floor recycle_window keeps for a
     * query at P. A checkpoint slot therefore carries, after the linear state, window_blocks pages
     * per snapshot group (every layer, layer-major rows), restored into fresh blocks seeded at
     * that floor.
     *
     * has_window_snapshot(): this rank's slot carries a window section.
     * window_rows(): from the leader's table, the block ids [lo, P/B) of every non-zero win_snap
     * group in plans() order, each padded with -1 to window_blocks; every rank addresses its own
     * pool with it, since followers keep no tables. False if the sequence no longer holds that
     * range.
     * seed_window(): gives a sequence with an empty table in each snapshot group the blocks
     * [lo, P/B) a restore will fill, before the ensure that extends it. Other WINDOW groups start
     * at the floor ensure sets. */
    bool has_window_snapshot() const { return window_ck_bytes_ > 0; }
    bool window_rows(uint64_t seq, int64_t pos, std::vector<int32_t>* out) const;
    int  seed_window(uint64_t seq, int64_t pos);
    /* The floor and end (block indices) of the window a query at `pos` reads in a group. */
    static int64_t window_lo(const KVGroupPlan& p, int64_t pos) {
        const int64_t b = pos / p.block_size - (p.window_blocks - 1);
        return b > 0 ? b : 0;
    }

    /* Byte address of a state slot, for the slot-mapping builder. Null when the group has no
     * base. */
    void*   state_ptr(int32_t group, int32_t layer, int32_t slot) const;

private:
    /* carve() wraps carve_into() and calls forget() on failure, so a carve that stops part way
     * cannot leave plans_ passing guards for block state that was never sized. plan_groups() also
     * starts from forget(). */
    int  carve_into(Pool& kv_pool);
    /* The token count at which the paged groups run out together for `budget_bytes` of pool,
     * under tokens_cap_, and each group's block count in `blocks` (by group index; a windowed
     * group capped at its window). carve_into() spends what this returns. */
    int64_t paged_split(int64_t budget_bytes, std::vector<int64_t>* blocks) const;
    /* A zero group's share of `tokens`: the block count of the shared table, capped at the window
     * for a windowed group. 0 for kUnboundedTokens, which is not a count. */
    int64_t zero_blocks(const KVGroupPlan& p, int64_t tokens) const;
    void forget();

    /* One traversal serving both checkpoint directions; see checkpoint_save. */
    int checkpoint_copy(const int32_t* state_slots, int n_slots, int32_t ck_slot, bool save,
                        RadStream s, const std::vector<int32_t>* win);

    struct Seq {
        uint64_t id = 0;
        int64_t  n_tokens = 0;
        std::vector<std::vector<int32_t>> blocks;   /* per group */
        std::vector<int64_t> first_block_pos;       /* WINDOW: token pos of blocks[0] */
        std::vector<int32_t> state;                 /* per group, -1 when not stateful */
        /* Per group, the lowest table entry written since the batch builder last staged this
         * sequence. An append or truncation lowers it to where it touched; a shift lowers it to
         * zero; kNoDirty is unchanged. Mutable because take_table_changes() clears it. */
        mutable std::vector<int64_t> dirty_from;
    };

    int  grab_block(int32_t g, int32_t* out);
    void drop_block(int32_t g, int32_t b);
    /* Increments a held block's refcount and keeps the cache-only count in step. */
    void add_ref(size_t g, int32_t b);
    int  grab_state(int32_t g, int32_t* out);
    void drop_state(int32_t g, int32_t s);
    Seq* find(uint64_t seq);
    const Seq* find(uint64_t seq) const;
    /* Recycle a WINDOW group's blocks no query at or past `keep_from` can read. */
    void recycle_window(Seq* s, int64_t keep_from);

    std::vector<KVGroupPlan> plans_;
    std::vector<std::vector<int32_t>> free_blocks_;   /* per group */
    std::vector<std::vector<int32_t>> refcount_;      /* per group, per block */
    /* Per group, per block: the prefix cache holds one of the references. A free block is never
     * marked. `cache_only_` is, per group, how many marked blocks are at a count of one. */
    std::vector<std::vector<uint8_t>> cached_;
    std::vector<int64_t>              cache_only_;
    std::vector<std::vector<int32_t>> free_states_;   /* per group */
    std::unordered_map<uint64_t, Seq> seqs_;
    std::vector<BlockCopy> pending_;

    HostArena* ck_store_ = nullptr;                   /* set_checkpoint_store */
    int64_t   ck_rank_off_ = 0;
    int64_t   checkpoint_bytes_ = 0;
    int64_t   window_ck_bytes_ = 0;   /* the window section of checkpoint_bytes_ */
    std::vector<StateClear> clears_;                  /* grabbed slots the engine must zero */
    void*     pool_base_ = nullptr;
    BundleComparison bundling_;
    std::string report_;
    int64_t   max_seqs_ = 0;
    int64_t   max_ctx_ = 0;
    int64_t   max_tok_ = 0;      /* --max-num-batched-tokens; the buffer floor is one of these */
    /* plan_groups() computes these; carve() spends against them and ceiling() reports on them. */
    int64_t   fixed_ = 0;      /* the non-negotiable floor: states, conv slots, padding */
    int64_t   ckpts_ = 0;      /* --checkpoint-slots, kept for the warning on a model with none */
    int64_t   tokens_cap_ = 0; /* cap_paged_tokens(): the least any rank can carve; 0 is none */

    /* ---- the loan (see the public block) ----
     *
     * `live_` is the carve less the loan. Every carved id stays in `free_blocks_`; ids at or
     * above live_ are just not handed out. The free list is a min-heap so the lowest free id goes
     * out first: held blocks pack at the bottom and the top, the only part a loan reaches, drains.
     * A stack would hand out the most recently freed id, often a high one. The heap also makes the
     * gate a single check: if any free id is below live_, the smallest one is.
     *
     * `held_hist_` counts held blocks per bucket, so finding the highest held block is a walk of
     * a few thousand counters, not a refcount scan. It bounds how far a loan may reach, so it is
     * maintained on the allocation path; a stale value would lend a held block. */
    /* The buffer per group, in blocks. `chunk_blocks_` is the floor (one prefill chunk, fixed at
     * plan time). `short_peak_` is the largest recent admission shortfall, raised by grow_for and
     * decayed by spare_tokens().
     *
     * Only the shortfall is recorded, never the recall: grow_for recalls shortfall plus buffer,
     * and feeding that back would grow the buffer from its own value until it reached the carve. */
    std::vector<int64_t>  chunk_blocks_;
    std::vector<int64_t>  short_peak_;
    Pool*    pool_ = nullptr;
    int64_t  loan_ = 0;                     /* tokens lent from the top of every lendable group */
    uint64_t seq_frees_ = 0;
    std::function<int(int64_t)> reclaim_;
    std::vector<int64_t>  live_;            /* per group: blocks the cache may hand out */
    std::vector<int64_t>  free_usable_;     /* per group: free ids below live_ */
    std::vector<int64_t>  bucket_blocks_;   /* per group: blocks one held_hist_ bucket covers */
    std::vector<std::vector<int32_t>> held_hist_;   /* per group, per bucket: held blocks */

    int64_t available_blocks(int32_t g) const;
    /* The buffer this group wants in hand, in blocks. See buffer_bytes(). */
    int64_t  buf_blocks(size_t i) const;
    /* One past the highest held block, rounded up to a bucket. */
    int64_t  held_top(size_t i) const;
    /* The blocks a loan of `tokens` takes off the top of group i. */
    int64_t  lent_blocks(size_t i, int64_t tokens) const;
    /* Moves every lendable group's mark to a loan of `tokens`, unchecked (set_loan checks). */
    void     apply_loan(int64_t tokens);
    int  grow_for(int32_t g, int64_t blocks);
    void note_held(int32_t g, int32_t b, int delta);
    bool      configured_ = false;
    static const std::vector<int32_t> kNoBlocks;
};

}  /* namespace rad */
