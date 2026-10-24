/* stage.cpp -- see stage.h (the ordering rule lives there, beside the code that would break it). */
#include "kvstream/stage.h"

#include "device/device.h"
#include "kvstream/bind.h"
#include "runtime/wtab.h"

#include <algorithm>
#include <cstdlib>

namespace rad {

namespace {
/* The layer staged most recently on this rank thread; reset at every pass. */
thread_local int g_last_layer = -1;
/* What the pass that just closed staged, and over how many tokens: the counter is read at the next
 * enter, so the numbers are the ones a request actually paid (docs/KV-STREAMING.md §3.9). */
struct PassStage {
    uint64_t layers = 0, bytes = 0;
    int64_t  tokens = 0, ctx = 0, live_row = 0;   /* live_row: the history ONE layer of this pass reads */
    int64_t  bound_blocks = 0;                    /* the manager's bound the copy was clamped to */
    int      phase = -1;
};
thread_local PassStage g_pass;
bool trace_stage() {
    static const bool on = std::getenv("RADIANCE_KV_STAGE_TRACE") != nullptr;
    return on;
}

/* One line per pass that staged: the bytes the pass moved against the bytes the history it read
 * actually occupies. The ratio is the number that decides whether the stage is a bandwidth problem
 * or a per-chunk tax (see the note beside kv_stage_layer). */
void report_pass() {
    if (!g_pass.layers) return;
    const double live = (double)g_pass.live_row * (double)g_pass.layers;
    RAD_INFO("kv streaming: stage pass phase=%d tokens=%lld ctx=%lld: %llu layers, %.1f MiB moved, "
             "%.1f MiB of live history (%.1fx), bound %lld blocks",
             g_pass.phase, (long long)g_pass.tokens, (long long)g_pass.ctx,
             (unsigned long long)g_pass.layers, (double)g_pass.bytes / (1024.0 * 1024.0),
             live / (1024.0 * 1024.0), live > 0.0 ? (double)g_pass.bytes / live : 0.0,
             (long long)g_pass.bound_blocks);
}
}  /* namespace */

/* Forget the last staged layer. A pass visits its layers in order and `resolve_sel` may be called
 * once per attention launch, so the driver stages a layer once and treats a consecutive repeat as
 * already done. Called from kvstream_enter() at every pass. */
static void kvstage_begin_pass() {
    report_pass();
    g_last_layer = -1;
    g_pass = PassStage();
}

void kvstream_enter(Ctx* c, const RadBatch* b) {
    /* The stage is filled once per layer per pass; a new pass starts at the first layer again. */
    kvstage_begin_pass();
    if (!c || !b) return;
    g_pass.tokens = b->n_tok;
    g_pass.ctx    = b->max_ctx_len;
    g_pass.phase  = b->phase;
    for (int i = 0; i < b->n_kv_groups; ++i) {
        if (!b->kv[i].stream) continue;
        KVPoolBinding* kb = c->kv_binding(b->kv[i].group);
        if (!kb || !kb->has_stage) continue;
        /* A pass selects by phase and row shape (bind.h). */
        kv_binding_enter(*kb, kv_pass_view(b->phase, b->n_tok, b->n_seq, b->n_spec));
        /* The history one layer of this pass can read: every block up to the longest sequence's
         * end, at the same block size the resolve uses. That is the useful part of the stage row
         * the draw below copies in full. */
        if (kb->stage.bytes > 0 && b->kv[i].block_size > 0 && b->kv[i].stream->block_bytes > 0) {
            const int64_t reach = b->max_ctx_len + b->max_q_len;
            const int64_t live_blocks = (reach + b->kv[i].block_size - 1) / b->kv[i].block_size;
            const int64_t row = std::min(live_blocks * b->kv[i].stream->block_bytes, kb->stage.bytes);
            if (row > g_pass.live_row) g_pass.live_row = row;
            if (kb->stage_live_bytes > 0) {
                const int64_t bound = kb->stage_live_bytes / b->kv[i].stream->block_bytes;
                if (bound > g_pass.bound_blocks) g_pass.bound_blocks = bound;
            }
        }
    }
}

/* The allocator's high-water mark, as bytes of one layer: every id a table can name is below it
 * (stage.h, "VOLUME", states the argument and the failure if it is ever too small). */
void kvstream_bound_live(Ctx* c, const KVManager& kv) {
    if (!c) return;
    for (int g = 0; g < kv.n_groups(); ++g) {
        KVPoolBinding* b = c->kv_binding(g);
        if (!b || !b->has_stage) continue;
        const RadKVStream* st = kv.stream_view(g);
        if (!st || st->block_bytes <= 0) continue;
        const int64_t blocks = kv.held_top_blocks(g);
        b->stage_live_bytes = blocks > 0 ? blocks * st->block_bytes : 0;
    }
}

}  /* namespace rad */

namespace rad {
namespace {

/* Where a staged layer's bytes come from and how big they are: both views start at the pool base, so
 * the destination is always offset 0 and the source is this layer's row in the mirror, addressed by
 * the group-local position (`layer_slot`), not by the absolute model layer. */
struct StageSrc {
    void*          dst = nullptr;
    const uint8_t* src = nullptr;
    int64_t        bytes = 0;
};

bool stage_src(Ctx* ctx, int group, const RadKVGroupBatch* kvb, int layer, StageSrc* o) {
    const KVPoolBinding* b = ctx->kv_binding(group);
    const RadKVStream* st = kvb ? kvb->stream : nullptr;
    if (!b || !st) return false;
    const int64_t li = ctx->kv_layer_index(group, layer);
    if (!st->host || st->block_bytes <= 0 || st->host_layer_stride <= 0 || li < 0) return false;
    o->dst = b->base;
    o->src = st->host + li * st->host_layer_stride;
    o->bytes = b->bytes;
    /* Only the live prefix of the row: the block ids above the manager's highest held block cannot
     * be named by any table this pass reads (stage.h, "VOLUME" -- an underestimate here is wrong
     * text, not a fault, so this is the bound and not a guess). */
    if (b->stage_live_bytes > 0 && b->stage_live_bytes < o->bytes) o->bytes = b->stage_live_bytes;
    if (o->bytes <= 0 || o->bytes > st->host_layer_stride) o->bytes = st->host_layer_stride;
    return o->bytes > 0;
}

/* The DMA path: one layer of the mirror onto the card without holding a CU slot, overlapped with the
 * layer before it. `done` is an event the list completes instead of a dispatch; `mark` is the ride
 * on the compute stream the list waits for, or null (copy now). */
bool dma_stage(Ctx* c) {
    static const bool on = rad_xp_supported(c->device()) != 0;
    return on;
}

/* One list in flight per rank thread: the layer it fills, and the ride it waits on. */
struct StagePipe {
    RadEvent done = nullptr;
    int      inflight = -1;    /* the layer the outstanding list fills */
    int      group = -1;
    void*    mark = nullptr;
    int64_t  bytes = 0;        /* what that list was told to move, for the pass's counter */
};
thread_local StagePipe g_pipe;
thread_local bool g_warned_ride = false;

/* Arm the completion word, mark the compute stream and hand the list over.
 *
 * The arm is a write of 1 in stream order (the weight table's patch dispatch, runtime/wtab.h), and
 * it is also the dispatch the list's ride waits on: so the list starts when the arm has executed,
 * and the copy's 0 can never precede the 1 the wait is looking for. The arm is behind every
 * attention launch of the layer before, which is what makes the one-layer buffer safe to refill
 * (core/kvstream/stage.h). */
bool arm_and_submit(Ctx* c, int group, int layer, const StageSrc& s, void* mark) {
    uint32_t* word = nullptr;
    if (!g_pipe.done || rad_event_dma(g_pipe.done, c->stream(), &word) < 0) return false;
    WtabPatch arm;
    arm.n = 1;
    arm.dst[0] = (void**)(void*)word;
    arm.val[0] = (void*)(uintptr_t)1;
    if (wtab_patch(arm, c->stream()) < 0) return false;
    const uint64_t src = (uint64_t)(uintptr_t)s.src, dst = (uint64_t)(uintptr_t)s.dst;
    const uint64_t bytes = (uint64_t)s.bytes;
    /* No yield: this copy's wait is a barrier on the very stream whose link hold would release it
     * (docs/KV-STREAMING.md §6.1, the deadlock the prefetch's rule names). */
    if (rad_xp_list(c->device(), mark, 1, &src, &dst, &bytes, word, 0, RAD_XP_NO_YIELD) != RAD_OK)
        return false;
    g_pipe.inflight = layer;
    g_pipe.group = group;
    g_pipe.mark = mark;
    g_pipe.bytes = s.bytes;
    if (trace_stage())
        RAD_INFO("kv streaming: handed over layer %d (%lld bytes), behind %lld", layer,
                 (long long)s.bytes, (long long)rad_xp_behind(c->device()));
    return true;
}

/* The copy on the compute stream, for a backend with no DMA queue (the host backend, a test) and
 * for a ride the engine would not lend: a list holds no CU slot, a copy does, so this is the path
 * that pauses the layer rather than the one that rides behind it. */
int serial_stage(Ctx* c, const StageSrc& s) {
    RadStream st = c->stream();
    const int rc = (s.bytes & 15) == 0
                       ? rad_memcpy_ranges_async(s.dst, s.src, s.bytes, nullptr, 0, st)
                       : rad_memcpy_async(s.dst, s.src, s.bytes, st);
    return rc;
}

void count_stage(Ctx* c, int layer, int64_t bytes) {
    g_last_layer = layer;
    g_pass.layers += 1;
    g_pass.bytes  += (uint64_t)bytes;
    if (trace_stage() && c->rank() == 0)
        RAD_INFO("kv streaming: stage layer %d (%lld bytes)", layer, (long long)bytes);
}

/* Records the arm and the submit where a recorded pass made them: a DMA list is not a queue packet,
 * so a tape that replayed without them would read an unstaged layer. */
int stage_issue_host(Ctx* c, int group, int layer, bool after) {
    if (c->recording())
        return c->tape_kv_stage(after, group, layer);
    return kvstage_run(c, group, layer, after);
}

}  /* namespace */

/* The wait and, when the pipeline has nothing for this layer, the copy: issued before the layer's
 * store, so the store's newer bytes are the last writer of this chunk's positions (the copy reads a
 * mirror the store has not written yet). 1 when the layer is staged, 0 when the group has no stage
 * view -- the caller then resolves as usual. A failure is reported with rad_step_fail and returns
 * 1, so the caller stops rather than reading an unstaged layer. Internal: every caller is the
 * extern "C" seam below. */
static int kvstage_wait(Ctx* c, int group, const RadKVGroupBatch* kvb, int layer) {
    if (!c || !kvb || !kvb->stream) return 0;
    const KVPoolBinding* b = c->kv_binding(group);
    if (!b || !b->has_stage || b->depth != 1) return 0;
    if (layer == g_last_layer) return 1;
    return stage_issue_host(c, group, layer, false);
}

/* After a layer's attention has been issued and has finished reading the stage: start the next
 * bound layer's copy. The ride is completed by the next dispatch on the compute stream, which is
 * behind every attention launch of `layer`; nothing may read the stage again until the wait that
 * precedes the next layer's store. 0 always: it reports nothing to the arch, and the exported
 * rad_kv_stage_next keeps the int for the ABI (every caller (void)s it). Internal: the extern "C"
 * seam below is its only caller. */
static int kvstage_next(Ctx* c, int group, int layer) {
    if (!c) return 0;
    const KVPoolBinding* b = c->kv_binding(group);
    if (!b || !b->has_stage || b->depth != 1) return 0;
    return stage_issue_host(c, group, layer, true);
}

int kvstage_run(Ctx* c, int group, int layer, bool after) {
    if (!c) return 0;
    const RadBatch* b = c->batch();
    const RadKVGroupBatch* kvb = nullptr;
    if (b)
        for (int i = 0; i < b->n_kv_groups; ++i)
            if ((int)b->kv[i].group == group) { kvb = &b->kv[i]; break; }
    if (!kvb || !kvb->stream) return 0;
    const KVPoolBinding* kb = c->kv_binding(group);
    if (!kb || !kb->has_stage || kb->depth != 1) return 0;

    if (!after) {
        /* Make `layer`'s stage row whole. Pipelined: the list was handed over at the layer before,
         * so this is a barrier on the compute stream that the engine's done word releases. */
        if (g_pipe.inflight >= 0 && (g_pipe.inflight != layer || g_pipe.group != group)) {
            /* A list is filling the stage for a layer this pass does not read (a pass that ended
             * before it). Nothing may rewrite the buffer until it lands. */
            (void)rad_event_wait(c->stream(), g_pipe.done);
            g_pipe.inflight = -1;
            g_pipe.bytes = 0;
            rad_ride_release(g_pipe.mark, nullptr, 0);
            g_pipe.mark = nullptr;
        }
        if (g_pipe.inflight == layer && g_pipe.group == group) {
            if (rad_event_wait(c->stream(), g_pipe.done) < 0) {
                rad_step_fail(c, "kv streaming: the compute stream could not wait for the stage copy "
                                 "the DMA engine is filling; the layer's attention would read the "
                                 "previous layer's bytes");
                return 1;
            }
            /* What the list was handed, not the row's whole size: the counter is what the pass
             * paid, and the clamp is the whole point of the pass's bound. */
            const int64_t moved = g_pipe.bytes;
            g_pipe.inflight = -1;
            g_pipe.bytes = 0;
            /* The signal the list polled is nobody's until it goes back to the pool. */
            rad_ride_release(g_pipe.mark, nullptr, 0);
            g_pipe.mark = nullptr;
            count_stage(c, layer, moved);
            return 1;
        }
        StageSrc s;
        if (!stage_src(c, group, kvb, layer, &s)) {
            rad_step_fail(c, "kv streaming: a layer needs its prefill stage but the group has no host "
                             "mirror row for it, so the attention would read an unstaged layer");
            return 1;
        }
        /* Nothing was pipelined for this layer (the pass's first, or a group the pipeline has not
         * seen): copy it on the engine now and wait for it, so it costs the layer's prologue. */
        if (dma_stage(c)) {
            if (!g_pipe.done && rad_event_create_local(&g_pipe.done) < 0) g_pipe.done = nullptr;
            /* The arm dispatch is what the ride below is completed by; it is issued after this
             * layer's projections, so the copy has that much of a head start. */
            void* mark = nullptr;
            if (g_pipe.done && rad_ride(c->stream(), &mark) == RAD_OK) {
                if (arm_and_submit(c, group, layer, s, mark)) {
                    const bool ok = rad_event_wait(c->stream(), g_pipe.done) >= 0;
                    g_pipe.inflight = -1;
                    rad_ride_release(mark, nullptr, 0);
                    if (ok) {
                        count_stage(c, layer, s.bytes);
                        return 1;
                    }
                } else {
                    rad_ride_release(mark, nullptr, 0);
                }
            }
        }
        if (serial_stage(c, s) < 0) {
            rad_step_fail(c, "kv streaming: the prefill stage copy failed; the layer's attention "
                             "would read the previous layer's bytes");
            return 1;
        }
        count_stage(c, layer, s.bytes);
        return 1;
    }

    /* The layer after this one in this group's binding, if it has one. */
    int next = -1;
    const auto& ls = kb->layer_slot;
    for (int l = layer + 1; l < (int)ls.size(); ++l)
        if (c->kv_layer_index(group, l) >= 0) { next = l; break; }
    if (next < 0 || !dma_stage(c)) return 0;
    if (!g_pipe.done && rad_event_create_local(&g_pipe.done) < 0) g_pipe.done = nullptr;
    if (!g_pipe.done) return 0;
    StageSrc s;
    if (!stage_src(c, group, kvb, next, &s)) return 0;
    /* No spare ride: the next layer's wait copies it on the compute stream instead. */
    void* mark = nullptr;
    if (rad_ride(c->stream(), &mark) != RAD_OK) {
        if (!g_warned_ride) {
            g_warned_ride = true;
            RAD_INFO("rank %d: kv streaming: no spare ride for the stage's DMA list, so its layer "
                     "copies serially", c->rank());
        }
        return 0;
    }
    if (!arm_and_submit(c, group, next, s, mark)) {
        rad_ride_release(mark, nullptr, 0);
        return 0;
    }
    return 0;
}

}  /* namespace rad */

extern "C" int rad_kv_stage_layer(RadCtx* c, rad_kvgroup group, const RadKVGroupBatch* kvb,
                                  int layer) {
    using namespace rad;
    if (!c || !kvb || !kvb->stream) return 0;
    return kvstage_wait(static_cast<Ctx*>(c), group, kvb, layer);
}

/* The two exported names are the same question ("is this layer staged?"); the frozen ABI keeps
 * both signatures, but only one body. */
extern "C" int rad_kv_stage_wait(RadCtx* c, rad_kvgroup group, const RadKVGroupBatch* kvb,
                                 int layer) {
    return rad_kv_stage_layer(c, group, kvb, layer);
}

extern "C" int rad_kv_stage_next(RadCtx* c, rad_kvgroup group, int layer) {
    using namespace rad;
    if (!c) return 0;
    return kvstage_next(static_cast<Ctx*>(c), group, layer);
}
