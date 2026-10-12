/* stager.cpp -- see stager.h. */
#include "place/stager.h"

#include <algorithm>

namespace rad {

PrefillStager::~PrefillStager() {
    for (Place& p : place_) {
        if (p.done) rad_event_destroy(p.done);
        if (p.free) rad_event_destroy(p.free);
    }
    for (RadEvent& g : gate_)
        if (g) rad_event_destroy(g);
}

int PrefillStager::init(const Program& prog, const Plan& plan, Mover* mover, char* region,
                        int64_t bytes, int64_t min_rows) {
    layers_.clear();
    if (!mover || !region || bytes <= 0) return RAD_OK;

    /* A ROUTED LAYER IS ITS EXPERT UNITS, and the ops that read it are the span their declared
     * uses cover: the first is the one that must find the copies, the last the one after which
     * nothing reads them again -- once the block's last pass of rows has issued it (after_op).
     * A declared draft head's layer, numbered past the trunk's, is not one (see the top of
     * stager.h). */
    const int32_t trunk = prog.drafter.kind != RAD_DRAFT_NONE && prog.meta.n_layers > 0
                              ? (int32_t)prog.meta.n_layers : INT32_MAX;
    std::map<int32_t, Layer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.expert < 0 || mu.layer < 0 || mu.first_use_op < 0 || mu.last_use_op < 0) continue;
        if (mu.layer >= trunk) continue;
        Layer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    for (auto& kv : by_layer) layers_.push_back(std::move(kv.second));
    std::sort(layers_.begin(), layers_.end(),
              [](const Layer& a, const Layer& b) { return a.first_op < b.first_op; });
    /* Two layers' spans must not overlap: a layer's copies are given back at the next layer's
     * first op, which must be past every op that reads them. */
    for (size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i].first_op <= layers_[i - 1].last_op) {
            RAD_WARN("prefill staging: routed layers %d and %d are read by interleaved ops (%d..%d "
                     "and %d..%d); staging is off", layers_[i - 1].layer, layers_[i].layer,
                     layers_[i - 1].first_op, layers_[i - 1].last_op, layers_[i].first_op,
                     layers_[i].last_op);
            layers_.clear();
            return RAD_OK;
        }
    if (layers_.empty()) return RAD_OK;

    first_of_.assign(prog.ops.size(), -1);
    last_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i) {
        if ((size_t)layers_[i].first_op < first_of_.size()) first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
        if ((size_t)layers_[i].last_op < last_of_.size()) last_of_[(size_t)layers_[i].last_op] = (int32_t)i;
    }

    mover_ = mover;
    region_ = region;
    ring_ = bytes / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT * 2;
    head_ = 0;
    min_rows_ = min_rows;
    n_early_ = 0;
    ring_order_.clear();
    spare_.clear();
    for (int i = kPlaces - 1; i >= 0; --i) {
        Place& p = place_[i];
        if (rad_event_create_local(&p.done) < 0 || rad_event_create_local(&p.free) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        spare_.push_back(i);
    }
    for (int i = 0; i < kGates; ++i) {
        if (rad_event_create_local(&gate_[i]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
        gate_li_[i] = -1;
    }
    /* The backend opens what its first list needs -- the card's DMA queue -- here at bring-up, not
     * inside the first long prompt's first step. */
    return rad_memcpy_list_async(nullptr, 0, 0, mover_->transfer_stream());
}

/* A TRUNK PASS AT THE FULL PLAN AND LARGE ENOUGH, and only that (see the top of stager.h). The
 * next trunk pass's first layers, staged at the end of the last one, are this pass's if it arms
 * and go back to the pool if it does not. */
bool PrefillStager::begin_pass(const RadBatch* b) {
    if (!b || b->draft_pass != 0) return false;
    armed_ = full_ && enabled() && b->n_tok >= min_rows_;
    staged_ = armed_;
    if (!armed_) {
        withdraw_early();
        return false;
    }
    first_op_ = true;
    next_ = 0;
    first_li_ = -1;
    for (int32_t& g : gate_li_) g = -1;
    return true;
}

/* The early layers' copies back to the pool: nothing has read them, so there is no `free` to
 * record and the next copy over their places needs no wait for them. */
void PrefillStager::withdraw_early() {
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired) retire(p);
    }
    n_early_ = 0;
}

/* The host half of giving a place back: its units are published at their pool slots again, which
 * the next residency sync hands to the weight tables. The device half is `free`, which the copy
 * that laps the place waits on. */
void PrefillStager::retire(Place& p) {
    mover_->unstage_units(p.staged);
    p.staged.clear();
    p.retired = true;
}

/* The oldest place leaves the ring, and the mover's stream waits on its last reader: whatever is
 * copied there later goes out behind that wait. */
int PrefillStager::recycle_front() {
    const int32_t i = ring_order_.front();
    Place& p = place_[i];
    if (p.free_rec) RAD_TRY(rad_event_wait(mover_->transfer_stream(), p.free));
    p.li = -1;
    p.free_rec = p.read = p.retired = false;
    ring_order_.pop_front();
    spare_.push_back(i);
    return RAD_OK;
}

/* A PLACE OF `want` BYTES at the head of the ring, or -- `partial` -- all the room the head can
 * reach when `want` does not fit. The head may write over retired places only: the oldest place
 * still published bounds it a whole ring ahead of where that place starts. *out is -1 when
 * nothing fits. */
int PrefillStager::take(int64_t want, bool partial, int32_t* out) {
    *out = -1;
    int64_t limit = head_ + ring_;
    for (int32_t i : ring_order_)
        if (!place_[i].retired) { limit = place_[i].beg + ring_; break; }
    int64_t len = std::min(want, limit - head_);
    if (len < want && !partial) return RAD_OK;
    if (len < RAD_ALIGN_UNIT) return RAD_OK;
    const int64_t end = head_ + len;
    /* Every place the new one laps leaves the ring first; then, with the table full, the oldest
     * retired ones -- their waits only order the mover's stream behind reads long finished. */
    while (!ring_order_.empty() && place_[ring_order_.front()].beg + ring_ < end)
        RAD_TRY(recycle_front());
    while (spare_.empty() && !ring_order_.empty() && place_[ring_order_.front()].retired)
        RAD_TRY(recycle_front());
    if (spare_.empty()) return RAD_OK;
    const int32_t i = spare_.back();
    spare_.pop_back();
    Place& p = place_[i];
    p.beg = head_;
    p.end = end;
    *out = i;
    return RAD_OK;
}

/* ONE LAYER INTO THE RING, behind `gate` when it is not null. `now` is the layer the run reaches
 * next, which takes whatever room there is up to all it needs; any other goes only whole and at
 * most half the ring -- so the next layer never finds less room than two buffers of half the ring
 * gave it, and a heavy layer is not cut short for having been reached early. *got is kStaged, kNone
 * when nothing of the layer is pooled, or kLater when the ring has no room for it yet. */
int PrefillStager::stage_layer(int32_t li, bool now, RadEvent gate, int* got) {
    *got = kLater;
    const Layer& L = layers_[(size_t)li];
    int64_t unit = 0;
    const int64_t need = mover_->stage_need(L.units, &unit);
    if (need <= 0) { *got = kNone; return RAD_OK; }
    if (!now && need > ring_ / 2) return RAD_OK;
    /* A place that runs past the region's end is two stretches, and the first can end in less
     * room than a unit: it asks for one unit more than the layer needs. */
    const int64_t want = head_ % ring_ + need > ring_ ? need + unit : need;
    int32_t i = -1;
    RAD_TRY(take(want, now, &i));
    if (i < 0) return RAD_OK;
    Place& p = place_[i];
    p.li = li;
    if (gate) RAD_TRY(rad_event_wait(mover_->transfer_stream(), gate));
    /* A place that runs past the region's end goes on at its start. */
    const int64_t off = p.beg % ring_, len = p.end - p.beg;
    Mover::StageSpan sp[2] = { { region_ + off, std::min(len, ring_ - off) },
                               { region_, len - std::min(len, ring_ - off) } };
    RAD_TRY(mover_->stage_units(L.units, sp, sp[1].cap > 0 ? 2 : 1, nullptr, p.done, &p.staged));
    if (p.staged.empty()) {
        p.li = -1;
        spare_.push_back(i);
        *got = kNone;
        return RAD_OK;
    }
    head_ = p.end;
    ring_order_.push_back(i);
    *got = kStaged;
    return RAD_OK;
}

/* STAGE THE LAYERS AFTER `cur` as far as the pace lets them go and the ring has room. A layer
 * nothing of which is pooled is passed over; one the ring has no room for yet, or whose pace
 * layer the run has not reached, stops the walk until a later op. */
int PrefillStager::stage_ahead(int32_t cur, bool* resync) {
    while ((size_t)next_ < layers_.size()) {
        const int32_t g = next_ - kLead;
        RadEvent gate = nullptr;
        if (g >= first_li_) {
            if (gate_li_[g % kGates] != g) break;
            gate = gate_[g % kGates];
        }
        int got = kLater;
        RAD_TRY(stage_layer(next_, next_ <= cur + 1, gate, &got));
        if (got == kLater) break;
        *resync = *resync || got == kStaged;
        ++next_;
    }
    return RAD_OK;
}

int PrefillStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;

    /* THE PASS'S FIRST ROUTED LAYER, staged at its first op of any kind, so the copies run behind
     * whatever the pass issues before the layer -- and the layers after it as far as the pace and
     * the ring allow. The layers the pass before staged for it are its own from here. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            next_ = (int32_t)(it - layers_.begin());
            first_li_ = next_;
            if (n_early_ > 0 && next_ != 0) {
                withdraw_early();
                *resync = true;
            }
            next_ = std::max(next_, n_early_);
            n_early_ = 0;
            RAD_TRY(stage_ahead(first_li_ - 1, resync));
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    /* The run's arrival at this layer, which the copies kLead layers on are paced by. */
    RAD_TRY(rad_event_record(gate_[li % kGates], s));
    gate_li_[li % kGates] = li;
    /* This layer's copies were published when they went out: the one wait that orders its ops
     * behind them. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li == li) RAD_TRY(rad_event_wait(s, p.done));
    }
    /* The layers before are done with their places: every pass of rows issued their last op, the
     * last of them recorded `free`, and a layer's ops do not interleave with the next one's. Given
     * back here and not at that op, whose next run would read copies already overwritten. */
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (!p.retired && p.li < li && p.read) retire(p);
    }
    /* And the copies go on ahead of it into the room that gave back. */
    if (next_ <= li) next_ = li + 1;
    const bool was = *resync;
    RAD_TRY(stage_ahead(li, resync));
    *resync = *resync || was;
    return RAD_OK;
}

/* THE LAYER'S LAST OP: `free` goes behind it, so a copy that laps its place can start as early as
 * it could -- and goes behind it again if the block issues the op for another pass of rows, which
 * is why the copies themselves are not given back until the next layer starts. */
int PrefillStager::after_op(rad_op h, RadStream s) {
    if ((size_t)h >= last_of_.size()) return RAD_OK;
    const int32_t li = last_of_[(size_t)h];
    if (li < 0) return RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired || p.li != li) continue;
        RAD_TRY(rad_event_record(p.free, s));
        p.free_rec = true;
        p.read = true;
    }
    return RAD_OK;
}

/* Nothing this pass staged outlives it. Then, when the next step goes on with a prompt, its trunk
 * pass's first layers go into the ring behind this pass's last ones -- layer k paced on this
 * pass's layer n - kLead + k, as the run would pace it if the two passes were one. */
int PrefillStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    int rc = RAD_OK;
    for (int32_t i : ring_order_) {
        Place& p = place_[i];
        if (p.retired) continue;
        const int r = rad_event_record(p.free, s);
        if (r < 0 && rc >= 0) rc = r;
        p.free_rec = r >= 0 || p.free_rec;
        retire(p);
    }
    if (rc < 0 || !next_prefill_) return rc;
    const int32_t n = (int32_t)layers_.size();
    for (int32_t k = 0; k < kLead && k < n; ++k) {
        const int32_t g = n - kLead + k;
        if (g < first_li_ || gate_li_[g % kGates] != g) break;
        int got = kLater;
        RAD_TRY(stage_layer(k, /*now=*/false, gate_[g % kGates], &got));
        if (got == kLater) break;
        n_early_ = k + 1;
    }
    return RAD_OK;
}

/* ================================================================== the file stager */
FileStager::~FileStager() {
    for (int b = 0; b < 2; ++b) {
        if (done_[b]) rad_event_destroy(done_[b]);
        if (free_[b]) rad_event_destroy(free_[b]);
    }
}

int FileStager::init(const Program& prog, const Plan& plan, Mover* mover) {
    layers_.clear();
    if (!mover) return RAD_OK;

    std::map<int32_t, Layer> by_layer;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (mu.tier != Tier::SSD || mu.access != RAD_ACCESS_CONDITIONAL || mu.layer < 0) continue;
        if (mu.first_use_op < 0 || mu.last_use_op < 0) continue;   /* declared and never read */
        Layer& L = by_layer[mu.layer];
        L.layer = mu.layer;
        L.units.push_back((int32_t)u);
        L.first_op = L.first_op < 0 ? mu.first_use_op : std::min(L.first_op, mu.first_use_op);
        L.last_op = std::max(L.last_op, mu.last_use_op);
    }
    for (auto& kv : by_layer) layers_.push_back(std::move(kv.second));
    if (layers_.empty()) return RAD_OK;
    std::sort(layers_.begin(), layers_.end(),
              [](const Layer& a, const Layer& b) { return a.first_op < b.first_op; });
    /* A layer's copies stay published until the next one starts, so two layers whose ops
     * interleave would need both buffers at once and a third for the read ahead. That is a plugin
     * shape this tier cannot serve, and a file-tier unit has no address to fall back on. */
    for (size_t i = 1; i < layers_.size(); ++i)
        if (layers_[i].first_op <= layers_[i - 1].last_op) {
            RAD_ERR("file stager: routed layers %d and %d are read by interleaved ops (%d..%d and "
                    "%d..%d), and a routed layer read from the container is published one layer "
                    "at a time", layers_[i - 1].layer, layers_[i].layer, layers_[i - 1].first_op,
                    layers_[i - 1].last_op, layers_[i].first_op, layers_[i].last_op);
            layers_.clear();
            return RAD_E_UNSUPPORTED;
        }

    buf_[0] = mover->file_buffer(0);
    buf_[1] = mover->file_buffer(1);
    cap_ = mover->file_buffer_bytes();
    if (!buf_[0] || !buf_[1] || cap_ < plan.file_layer_max) {
        RAD_ERR("file stager: the plan reads %zu routed layers from the container and the mover "
                "holds no buffers for them", layers_.size());
        layers_.clear();
        return RAD_E_STATE;
    }
    first_of_.assign(prog.ops.size(), -1);
    for (size_t i = 0; i < layers_.size(); ++i)
        if ((size_t)layers_[i].first_op < first_of_.size())
            first_of_[(size_t)layers_[i].first_op] = (int32_t)i;
    mover_ = mover;
    for (int b = 0; b < 2; ++b)
        if (rad_event_create_local(&done_[b]) < 0 || rad_event_create_local(&free_[b]) < 0) {
            layers_.clear();
            return RAD_E_DEVICE;
        }
    return RAD_OK;
}

/* EVERY PASS: there is no other address for a file-tier unit, so a pass the stager sat out would
 * reach its first routed layer with null expert pointers. */
bool FileStager::begin_pass(const RadBatch* b) {
    armed_ = enabled() && b;
    if (!armed_) return false;
    holds_[0] = holds_[1] = -1;
    cur_ = -1;
    first_op_ = true;
    pass_kind_ = b->draft_pass;
    last_issued_ = -1;
    return true;
}

bool FileStager::will_issue(int32_t li) const {
    const auto it = pass_last_.find(pass_kind_);
    return it == pass_last_.end() || layers_[(size_t)li].first_op <= it->second;
}

int FileStager::start(int32_t li) {
    const int b = li & 1;
    if (holds_[b] >= 0) return RAD_E_STATE;     /* released before the next read, by construction */
    const int rc = mover_->stage_file_units(layers_[(size_t)li].units, buf_[b], cap_,
                                            free_rec_[b] ? free_[b] : nullptr, done_[b],
                                            &staged_[b]);
    if (rc < 0) return rc;
    if (!staged_[b].empty()) holds_[b] = li;
    return RAD_OK;
}

int FileStager::finish(int32_t b, RadStream s) {
    RAD_TRY(rad_event_record(free_[b], s));
    free_rec_[b] = true;
    mover_->unstage_units(staged_[b]);
    staged_[b].clear();
    holds_[b] = -1;
    return RAD_OK;
}

int FileStager::before_op(rad_op h, RadStream s, bool* resync) {
    *resync = false;
    if ((int32_t)h > last_issued_) last_issued_ = (int32_t)h;

    /* THE PASS'S FIRST ROUTED LAYER, read at its first op of any kind. */
    if (first_op_) {
        first_op_ = false;
        const auto it = std::lower_bound(layers_.begin(), layers_.end(), (int32_t)h,
                                         [](const Layer& L, int32_t op) { return L.first_op < op; });
        if (it != layers_.end()) {
            const int32_t li = (int32_t)(it - layers_.begin());
            if (will_issue(li)) {
                RAD_TRY(start(li));
                *resync = true;
            }
        }
    }

    if ((size_t)h >= first_of_.size()) return RAD_OK;
    const int32_t li = first_of_[(size_t)h];
    if (li < 0 || li == cur_) return RAD_OK;

    /* Entering layer `li`: the layer before is behind every op issued so far on this stream. */
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0 && holds_[b] != li) { RAD_TRY(finish(b, s)); *resync = true; }
    /* A layer the read-ahead skipped -- the learned bound said this pass ends before it -- is read
     * here rather than issued without addresses. */
    if (holds_[li & 1] != li) {
        RAD_TRY(start(li));
        *resync = true;
    }
    if (holds_[li & 1] == li) RAD_TRY(rad_event_wait(s, done_[li & 1]));
    cur_ = li;

    const int32_t next = li + 1;
    if ((size_t)next < layers_.size() && will_issue(next)) {
        RAD_TRY(start(next));
        *resync = true;
    }
    return RAD_OK;
}

int FileStager::after_op(rad_op h, RadStream s) {
    (void)h;
    (void)s;
    return RAD_OK;
}

int FileStager::end_pass(RadStream s) {
    if (!armed_) return RAD_OK;
    armed_ = false;
    pass_last_[pass_kind_] = last_issued_;
    int rc = RAD_OK;
    for (int b = 0; b < 2; ++b)
        if (holds_[b] >= 0) {
            const int r = finish(b, s);
            if (r < 0 && rc >= 0) rc = r;
        }
    cur_ = -1;
    return rc;
}

}  /* namespace rad */
