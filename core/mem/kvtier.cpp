#include "mem/kvtier.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace rad {

const char* kv_tier_name(KVTier t) {
    switch (t) {
        case KVTier::Device: return "device";
        case KVTier::Host:   return "host";
    }
    return "?";
}

/* ------------------------------------------------------------------ IdleTiers */

int IdleTiers::configure(const Config& c) {
    std::lock_guard lk(mu_);
    reset();
    cfg_ = c;

    if (cfg_.host_cap_bytes > 0 && cfg_.entry_bytes > 0) {
        const int st = arena_.open(cfg_.entry_bytes, cfg_.host_cap_bytes);
        if (st != RAD_OK) {
            RAD_WARN("kv tiers: no pinned host memory for %s, the host tier is off",
                     humanb(cfg_.host_cap_bytes).c_str());
            cfg_.host_cap_bytes = 0;
        }
    } else {
        cfg_.host_cap_bytes = 0;
    }

    char buf[512];
    if (!any_on()) {
        report_ = "kv tiers: off (no --prefix-cache-mib), so a finished conversation's KV stays "
                  "in VRAM until the cache evicts it under pressure -- it is never copied to host "
                  "memory, and what the pool is holding is mostly cache rather than live requests";
    } else {
        std::snprintf(buf, sizeof buf,
                      "kv tiers: a finished conversation's KV is copied to host memory at once "
                      "(%s, %lld entries) and its VRAM is given up when the expert loan or a "
                      "request wants it; a full host tier drops its least recently used entries. "
                      "One entry is %s",
                      humanb(arena_.cap_bytes()).c_str(), (long long)arena_.capacity(),
                      humanb(cfg_.entry_bytes).c_str());
        report_ = buf;
        if (cfg_.ckpt_bytes > 0) {
            std::snprintf(buf, sizeof buf,
                          "\nkv tiers: a session's linear state is saved to host memory -- %lld "
                          "snapshot slot(s) of %s -- and a full store retires the least recently "
                          "used",
                          (long long)cfg_.ckpt_slots, humanb(cfg_.ckpt_bytes).c_str());
            report_ += buf;
        }
    }
    return RAD_OK;
}

void IdleTiers::close() {
    std::lock_guard lk(mu_);
    reset();
}

/* The body of close(), for the caller that already holds the lock. */
void IdleTiers::reset() {
    recs_.clear();
    sessions_.clear();
    roots_.clear();
    host_lru_.clear();
    wb_q_.clear();
    clean_q_.clear();
    starved_ = 0;
    wb_stuck_ = false;
    arena_.close();
    cfg_ = Config{};
    report_.clear();
}

std::string IdleTiers::report() const { return report_; }

void IdleTiers::unlist(Rec& r) {
    if (!r.listed) return;
    host_lru_.erase(r.it);
    r.listed = false;
}

/* ONLY A RECORD WHOSE BYTES ARE ONLY OFF THE CARD. A device record -- clean or not -- is ordered by
 * the prefix cache's own LRU, and a second ordering of the same entries is the two-owner shape the
 * header's one-record rule exists to rule out. */
void IdleTiers::relist(const BlockHash& key, Rec& r) {
    if (r.pending || r.tier == KVTier::Device || r.listed) return;
    r.it = host_lru_.emplace_hint(host_lru_.end(), r.used_ns, key);
    r.listed = true;
}

bool IdleTiers::wants_copy(const Rec& r) const {
    if (r.pending || r.doomed) return false;
    return host_on() && r.tier == KVTier::Device && r.host_slot < 0 && !r.blocks.empty();
}

void IdleTiers::queue_copy(const BlockHash& key, Rec& r) {
    if (r.queued || !wants_copy(r)) return;
    wb_q_.push_back(key);
    r.queued = true;
}

void IdleTiers::give_slot(int32_t slot) {
    if (slot < 0) return;
    arena_.free(slot);
    wb_stuck_ = false;
}

void IdleTiers::free_host(Rec& r) {
    give_slot(r.host_slot);
    r.host_slot = -1;
}

void IdleTiers::queue_clean(const BlockHash& key, Rec& r) {
    if (r.clean_q) return;
    clean_q_.push_back(key);
    r.clean_q = true;
}

/* ------------------------------------------------------------------ the linear half */

/* RECORDED, NOT MOVED. A snapshot is in host memory from the moment it is saved and stays in its
 * store slot until the cache retires it; the tiers keep its slot and position for the session
 * view. */
void IdleTiers::note_snapshot(const BlockHash& key, int32_t slot, int64_t pos) {
    if (slot < 0) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    r.ck_slot = slot;
    r.ck_pos = pos;
}

void IdleTiers::release_snapshot(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    r.ck_slot = -1;
    /* A RECORD THAT ONLY EVER EXISTED FOR THE SNAPSHOT GOES WITH IT. note_snapshot creates one at
     * commit time, which is mid-request and long before the chain is published; a request that
     * never publishes -- a rebased block table, a cancel before the first whole block -- would
     * otherwise leave a record with no blocks, no storage and no way to be reached again. */
    if (!r.pending && r.tier == KVTier::Device && r.blocks.empty() && r.host_slot < 0)
        recs_.erase(it);
}

/* ------------------------------------------------------------------ the cache's notices */

void IdleTiers::note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns) {
    if (!any_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    r.used_ns = now_ns;
    if (r.tier == KVTier::Device && !r.pending) r.blocks = blocks;
    /* A record only moves on the host list when it is IN it. A device record is a timestamp and
     * nothing else, which is what keeps this cheap enough to sit on the publish path -- the hash
     * insert is the whole cost. */
    if (r.listed) {
        unlist(r);
        relist(key, r);
    }
    queue_copy(key, r);
}

void IdleTiers::note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns) {
    if (!any_on() || !keys || n <= 0) return;
    std::lock_guard lk(mu_);
    for (int64_t i = 0; i < n; ++i) {
        auto it = recs_.find(keys[i]);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        r.used_ns = now_ns;
        if (r.listed) {
            unlist(r);
            relist(keys[i], r);
        }
    }
}

void IdleTiers::rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns) {
    if (!any_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    if (r.pending) return;
    unlist(r);
    free_host(r);
    r.tier = KVTier::Device;
    r.blocks = blocks;
    r.used_ns = now_ns;
    queue_copy(key, r);
}

bool IdleTiers::doomed(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() || it->second.doomed;
}

IdleTiers::Stats IdleTiers::stats() const {
    std::lock_guard lk(mu_);
    Stats s = stats_;
    /* What the block arena holds, clean copies included: counted off the arena rather than kept
     * beside it, because every path that takes or returns a slot would otherwise have to remember
     * to move a second number. */
    s.host_bytes = arena_.used_slots() * cfg_.entry_bytes;
    return s;
}

/* AND EVERY CONVERSATION THAT STARTED HERE. A chain whose first block has gone cannot be hit at any
 * length -- a prefix hit starts at position zero. Leaving the row behind would report blocks the
 * pool has already handed to somebody else as this session's, and would hold its whole chain in
 * memory for a conversation that can never come back. Called wherever an entry goes for good: the
 * cache's forget(), and the sites here that drop an off-device entry and tell the cache. */
void IdleTiers::drop_rooted(const BlockHash& key) {
    auto ro = roots_.find(key);
    if (ro == roots_.end()) return;
    const std::vector<uint64_t> ids = ro->second;       /* drop_session rewrites the list */
    for (uint64_t id : ids) drop_session(id);
}

void IdleTiers::forget(const BlockHash& key) {
    std::lock_guard lk(mu_);
    drop_rooted(key);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.pending) {
        /* The engine is mid-transfer against this record. Erasing it now is exactly the
         * use-after-free the one-record rule exists to prevent, so mark it and let report()
         * finish the job and clean up. */
        r.doomed = true;
        return;
    }
    unlist(r);
    free_host(r);
    recs_.erase(it);
}

KVTier IdleTiers::tier_of(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() ? KVTier::Device : it->second.tier;
}

bool IdleTiers::resident_off_device(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    return !it->second.pending && it->second.tier != KVTier::Device;
}

/* ------------------------------------------------------------------ copy on idle */

bool IdleTiers::writeback_waiting() const {
    std::lock_guard lk(mu_);
    /* NOT WHILE NOTHING CAN MOVE. An arena full of the copies of entries still in VRAM has nothing
     * a capacity drop may take, so a plan would pop the queue, find no slot and put it back, every
     * tick, until a slot is freed or a VRAM copy is given up -- which is what ends the wait. */
    if (wb_stuck_ && arena_.full()) return false;
    return !wb_q_.empty() && host_on();
}

bool IdleTiers::starved() const {
    std::lock_guard lk(mu_);
    return starved_ > 0;
}

void IdleTiers::plan_writeback(int64_t max_entries, std::vector<TierJob>* out) {
    if (!host_on() || max_entries <= 0) return;
    std::lock_guard lk(mu_);
    const size_t first = out->size();
    while (!wb_q_.empty() && (int64_t)(out->size() - first) < max_entries) {
        const BlockHash key = wb_q_.front();
        wb_q_.pop_front();
        auto it = recs_.find(key);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        r.queued = false;
        if (!wants_copy(r)) continue;
        TierJob j;
        j.kind = TierJob::Kind::WriteBack;
        j.key = key;
        j.bytes = cfg_.entry_bytes;
        j.blocks = r.blocks;
        out->push_back(std::move(j));
    }

    /* THE HOST SLOTS, AS RUNS. Taken for the whole batch at once so that entries planned together
     * sit together in the arena and each run crosses the link as one copy. A batch the arena
     * cannot hold is cut where the arena ran out: the entries past that point go back to the
     * front of the queue, and the maintenance plan is told how many copies were refused so it can
     * make that much room. */
    const int64_t want = (int64_t)(out->size() - first);
    std::vector<int32_t> slots;
    slots.reserve((size_t)want);
    while ((int64_t)slots.size() < want) {
        int32_t got = 0;
        const int32_t s = arena_.alloc_run((int32_t)(want - (int64_t)slots.size()), &got);
        if (s < 0) break;
        for (int32_t k = 0; k < got; ++k) slots.push_back(s + k);
    }
    size_t next = 0;
    std::vector<BlockHash> refused;
    for (size_t i = first; i < out->size();) {
        TierJob& j = (*out)[i];
        if (next < slots.size()) {
            j.host_slot = slots[next++];
            ++i;
            continue;
        }
        ++starved_;
        refused.push_back(j.key);
        out->erase(out->begin() + (long)i);
    }
    if (want > 0 && slots.empty()) wb_stuck_ = true;
    for (size_t i = first; i < out->size(); ++i) {
        Rec& r = recs_[(*out)[i].key];
        unlist(r);
        r.pending = true;
    }
    /* Back where they were, at the front and in order, so the oldest conversation is still the
     * first copied once there is room. */
    for (auto it = refused.rbegin(); it != refused.rend(); ++it) {
        auto ri = recs_.find(*it);
        if (ri == recs_.end() || ri->second.queued) continue;
        wb_q_.push_front(*it);
        ri->second.queued = true;
    }
}

void IdleTiers::take_clean(std::vector<BlockHash>* out) {
    std::lock_guard lk(mu_);
    for (const BlockHash& k : clean_q_) {
        auto it = recs_.find(k);
        if (it == recs_.end()) continue;
        it->second.clean_q = false;
        out->push_back(k);
    }
    clean_q_.clear();
}

bool IdleTiers::droppable(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    const Rec& r = it->second;
    return !r.pending && !r.doomed && r.tier == KVTier::Device && r.host_slot >= 0;
}

void IdleTiers::drop_commit(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.tier != KVTier::Device) return;
    r.tier = KVTier::Host;
    ++stats_.dropped;
    wb_stuck_ = false;       /* a host record is what a capacity drop takes */
    relist(key, r);
}

/* ------------------------------------------------------------------ capacity and expiry */

void IdleTiers::plan(int budget, std::vector<TierJob>* out) {
    if (!host_on() || budget <= 0) return;
    std::lock_guard lk(mu_);
    /* HOW MANY, AND NOT "UNTIL THERE IS ROOM". A queued drop does not free its slot until the
     * engine reports it, so a loop that tested the arena would queue a drop for every entry on the
     * list and empty the whole tier to make room for one. The copies refused since the last plan
     * say how much room was wanted; this frees exactly that much, least recently used first. */
    const int64_t want = std::min<int64_t>(starved_, budget);
    starved_ = 0;
    for (int64_t k = 0; k < want && !host_lru_.empty();) {
        auto ri = recs_.find(host_lru_.begin()->second);
        if (ri == recs_.end()) { host_lru_.erase(host_lru_.begin()); continue; }
        Rec& r = ri->second;
        TierJob j;
        j.kind = TierJob::Kind::DropHost;
        j.key = ri->first;
        j.host_slot = r.host_slot;
        j.bytes = cfg_.entry_bytes;
        unlist(r);
        r.pending = true;
        out->push_back(std::move(j));
        ++k;
    }
}

void IdleTiers::expired(uint64_t cutoff_ns, size_t max, std::vector<BlockHash>* out) const {
    std::lock_guard lk(mu_);
    /* The list holds no record with a job out (relist), so every key it yields can go now. */
    for (auto it = host_lru_.begin();
         it != host_lru_.end() && it->first < cutoff_ns && out->size() < max; ++it)
        out->push_back(it->second);
}

bool IdleTiers::plan_promote(const BlockHash& key, TierJob* out) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    Rec& r = it->second;
    if (r.pending || r.doomed || r.tier != KVTier::Host) return false;

    *out = TierJob{};
    out->kind = TierJob::Kind::ToDevice;
    out->key = key;
    out->host_slot = r.host_slot;
    out->bytes = cfg_.entry_bytes;
    ++stats_.host_hits;
    unlist(r);
    r.pending = true;
    return true;
}

void IdleTiers::report(const TierJob& job, bool ok, uint64_t now_ns) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(job.key);
    if (it == recs_.end()) {
        /* Nothing names the slots a copy was handed but the job itself. */
        if (job.kind == TierJob::Kind::WriteBack) give_slot(job.host_slot);
        return;
    }
    Rec& r = it->second;
    r.pending = false;
    /* A promotion the chain was cut before is not a failure: nothing was tried. */
    if (!ok && (job.kind != TierJob::Kind::ToDevice || job.started)) ++stats_.failures;

    /* A record that was forgotten mid-transfer is finished off here and nowhere else. Doing it at
     * the top, before the per-kind handling, means no branch below has to remember to check. */
    if (r.doomed) {
        /* THE JOB'S SLOT AND THE RECORD'S ARE NOT THE SAME SLOT on every path. A copy's host slot
         * is the job's until it reports; freeing only the record's would leak it, and freeing the
         * job's as well where it IS the record's would return one slot twice. */
        if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
        free_host(r);
        recs_.erase(it);
        return;
    }

    switch (job.kind) {
        case TierJob::Kind::WriteBack: {
            if (job.host_slot >= 0) {
                if (ok && r.tier == KVTier::Device && r.host_slot < 0) {
                    r.host_slot = job.host_slot;
                    ++stats_.to_host;
                } else {
                    give_slot(job.host_slot);
                }
            }
            relist(job.key, r);
            /* Clean now: the cache is asked whether the VRAM copy can go. */
            if (r.tier == KVTier::Device && r.host_slot >= 0) queue_clean(job.key, r);
            /* A copy that failed is tried again. */
            queue_copy(job.key, r);
            return;
        }

        case TierJob::Kind::ToDevice:
            if (ok) {
                r.tier = KVTier::Device;
                r.blocks = job.blocks;
                /* THE HOST COPY IS KEPT, which is what makes the entry clean the moment it lands:
                 * its VRAM copy can go again at no cost, and the next copy this conversation needs
                 * is only of the tokens its next turn adds. */
                r.host_slot = job.host_slot;
                /* THE CLOCK RESTARTS: the entry came back because a request is about to hit it. */
                if (now_ns) r.used_ns = now_ns;
                ++stats_.promoted;
                queue_clean(job.key, r);
                return;
            }
            if (!job.started) {
                /* NEVER STARTED: the reservation ran out before this entry, and nothing about it
                 * was touched. It stays exactly where it was. */
                relist(job.key, r);
                return;
            }
            /* A FAILED PROMOTION DROPS THE ENTRY, AND THE CACHE MUST HEAR ABOUT IT. Every other
             * failure here can leave the record where it was, because nothing was consumed. This
             * one cannot: the scatter may have written some layers and not others, so the blocks
             * hold a mixture and the cached tensors are no longer a function of the tokens. A miss
             * is always safe and this is the only way to guarantee one -- but the prefix cache
             * still holds an entry naming blocks that are not ours any more, and left alone it
             * would serve them. */
            if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
            free_host(r);
            recs_.erase(it);
            drop_rooted(job.key);
            if (on_drop_) on_drop_(job.key);
            return;

        case TierJob::Kind::DropHost:
            free_host(r);
            ++stats_.host_drops;
            recs_.erase(it);
            drop_rooted(job.key);
            if (on_drop_) on_drop_(job.key);
            return;
    }
}


/* ------------------------------------------------------------------ the session view */

/* A PREFIX RELATION IS WHAT MAKES TWO CHAINS THE SAME CONVERSATION. The blocks are content
 * addressed, so turn N's chain is byte for byte turn N-1's with more on the end; anything that
 * diverges anywhere is a different transcript from that point on, however much of a system prompt
 * it shares. */
static bool chain_extends(const std::vector<BlockHash>& shorter,
                          const std::vector<BlockHash>& longer) {
    if (shorter.size() > longer.size()) return false;
    for (size_t i = 0; i < shorter.size(); ++i) if (shorter[i] != longer[i]) return false;
    return true;
}

void IdleTiers::note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                             uint64_t now_ns, const std::string& label) {
    if (chain.empty()) return;
    std::lock_guard lk(mu_);

    /* A SESSION IS A LEAF OF THE PREFIX TRIE, and that is what decides every case here. The turns
     * of one conversation arrive under different ids -- the id is the request's, and a
     * conversation is a new request every turn -- so without this the table grows a row a turn,
     * each holding its own copy of a chain that is a prefix of the next one's and none of them
     * ever revisited. Candidates are the sessions sharing this root, which is a handful at most.
     *
     * A CHAIN THIS ONE EXTENDS HAS STOPPED BEING A LEAF, so it is absorbed: its blocks are the
     * first blocks of this one, byte for byte, because the chain is content addressed.
     *
     * A CHAIN THAT EXTENDS THIS ONE MEANS THIS ONE IS NOT A LEAF EITHER. That happens when a
     * preempted request re-prefills and republishes SHORTER than the conversation it belongs to;
     * recording it would shrink the row to less than the pool is holding, and there may be
     * several conversations below it -- two clients past the same system prompt -- with nothing
     * to say which of them republished. So the most recent of them is touched and no row is
     * added, which is the one answer that is wrong about nothing. */
    uint64_t first_seen = 0;
    /* A conversation's label outlives the turn that sent it: a later turn without one keeps the
     * name an earlier one gave. */
    std::string kept_label = label;
    bool absorbed = false;
    std::vector<uint64_t>& at_root = roots_[chain[0]];
    for (size_t i = 0; i < at_root.size();) {
        auto it = sessions_.find(at_root[i]);
        if (it == sessions_.end()) { at_root.erase(at_root.begin() + (long)i); continue; }
        if (!chain_extends(it->second.chain, chain)) { ++i; continue; }
        if (first_seen == 0 || it->second.first_seen_ns < first_seen)
            first_seen = it->second.first_seen_ns;
        if (kept_label.empty()) kept_label = it->second.label;
        sessions_.erase(it);
        at_root.erase(at_root.begin() + (long)i);
        absorbed = true;
    }

    if (!absorbed) {
        Session* newest = nullptr;
        for (uint64_t sid : at_root) {
            auto it = sessions_.find(sid);
            if (it == sessions_.end() || !chain_extends(chain, it->second.chain)) continue;
            if (!newest || it->second.last_used_ns > newest->last_used_ns) newest = &it->second;
        }
        if (newest) {
            newest->last_used_ns = now_ns;
            if (!label.empty()) newest->label = label;
            return;
        }
    }

    Session& s = sessions_[id];
    s.last_used_ns = now_ns;
    s.n_tokens = n_tokens;
    s.first_seen_ns = first_seen ? first_seen : now_ns;
    s.chain = chain;
    s.label = std::move(kept_label);
    at_root.push_back(id);
}

void IdleTiers::drop_session(uint64_t id) {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) return;
    if (!it->second.chain.empty()) {
        auto ri = roots_.find(it->second.chain[0]);
        if (ri != roots_.end()) {
            std::vector<uint64_t>& v = ri->second;
            v.erase(std::remove(v.begin(), v.end(), id), v.end());
            if (v.empty()) roots_.erase(ri);
        }
    }
    sessions_.erase(it);
}

void IdleTiers::forget_session(uint64_t id) {
    std::lock_guard lk(mu_);
    drop_session(id);
}

std::vector<IdleTiers::SessionInfo> IdleTiers::sessions(Occupancy* occ) const {
    /* The view's hold on the lock, a slice at a time (YieldingMutex). */
    struct Slice {
        YieldingMutex& m;
        explicit Slice(YieldingMutex& mu) : m(mu) { m.lock_after_others(); }
        ~Slice() { m.unlock(); }
    };
    /* Records a slice visits: a few microseconds of hash lookups, the most an engine-side taker
     * arriving mid-slice waits. */
    constexpr size_t kSliceRecords = 64;

    std::vector<SessionInfo> out;
    {
        Slice lk(mu_);
        out.reserve(sessions_.size());
        for (const auto& kv : sessions_) {
            SessionInfo si;
            si.id = kv.first;
            si.model = model_;
            si.n_tokens = kv.second.n_tokens;
            si.first_seen_ns = kv.second.first_seen_ns;
            si.label = kv.second.label;
            si.last_used_ns = kv.second.last_used_ns;
            si.bytes = (int64_t)kv.second.chain.size() * cfg_.entry_bytes;
            out.push_back(std::move(si));
        }
    }
    /* SNAPSHOTS IN THE STORE, each once. The store is the engine's and is not read from this
     * thread -- but this walk visits every record a stored conversation reaches, so it counts them
     * here. They are sparse, one per checkpoint interval, so the set stays small beside the walk
     * that fills it. */
    std::unordered_set<BlockHash, BlockHashHash> host_snaps;
    size_t kept = 0;
    for (size_t k = 0; k < out.size(); ++k) {
        SessionInfo& si = out[k];
        /* The histogram is computed here and stored nowhere. A session's blocks move one tier at
         * a time under a bounded plan, so a second copy of this would be stale between the pass
         * that moves the first block and the pass that moves the last -- which is exactly the
         * window an operator is looking at the page during. The session is found again at every
         * slice: between slices it may have been absorbed into a longer one or dropped, and a
         * row that is gone is left out. */
        bool gone = false;
        for (size_t at = 0;;) {
            Slice lk(mu_);
            const auto it = sessions_.find(si.id);
            if (it == sessions_.end()) { gone = true; break; }
            const std::vector<BlockHash>& chain = it->second.chain;
            const size_t end = std::min(chain.size(), at + kSliceRecords);
            for (; at < end; ++at) {
                const BlockHash& h = chain[at];
                auto ri = recs_.find(h);
                if (ri == recs_.end()) { ++si.on_device; continue; }
                const Rec& rr = ri->second;
                if (rr.pending) si.moving = true;
                switch (rr.tier) {
                    case KVTier::Device: ++si.on_device; break;
                    case KVTier::Host:   ++si.on_host;   break;
                }
                /* THE LINEAR HALF, which is in host memory from the moment it is saved, whatever
                 * tier its blocks are on. */
                if (rr.ck_slot >= 0) {
                    ++si.snapshots;
                    if (occ) host_snaps.insert(h);
                    if (rr.ck_pos > si.resume_tokens) si.resume_tokens = rr.ck_pos;
                }
            }
            if (at >= chain.size()) break;
        }
        if (gone) continue;
        si.snap_bytes = si.snapshots * cfg_.ckpt_bytes;
        if (kept != k) out[kept] = std::move(si);
        ++kept;
    }
    out.resize(kept);
    std::sort(out.begin(), out.end(),
              [](const SessionInfo& a, const SessionInfo& b) {
                  return a.last_used_ns > b.last_used_ns;
              });
    if (occ) {
        Slice lk(mu_);
        occ->host_bytes = arena_.used_slots() * cfg_.entry_bytes +
                          (int64_t)host_snaps.size() * cfg_.ckpt_bytes;
        /* What the operator gave the block arena and the snapshot store -- the same scope as
         * `host_bytes` above, which counts both. */
        occ->host_cap_bytes = (host_on() ? cfg_.host_cap_bytes : 0) +
                              cfg_.ckpt_slots * cfg_.ckpt_bytes;
    }
    return out;
}
}  /* namespace rad */
