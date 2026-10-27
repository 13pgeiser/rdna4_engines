/* stage.cpp -- see stage.h, which states the ordering rule. */
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
/* What the pass staged, reported at the next pass's start. */
struct PassStage {
    uint64_t layers = 0, bytes = 0;
    int64_t  tokens = 0, ctx = 0, live_row = 0;   /* live_row: the history one layer reads */
    int64_t  bound_blocks = 0;                    /* the manager's bound the copy was clamped to */
    int      phase = -1;
};
thread_local PassStage g_pass;
/* Waits out a stage list still in flight (defined beside g_pipe). */
static void stage_pipe_drain(Ctx* c);

bool trace_stage() {
    static const bool on = std::getenv("RADIANCE_KV_STAGE_TRACE") != nullptr;
    return on;
}

/* Under RADIANCE_KV_STAGE_TRACE, one line per pass that staged: the bytes it moved against the
 * bytes of history it could read. */
void report_pass() {
    if (!g_pass.layers || !trace_stage()) return;
    const double live = (double)g_pass.live_row * (double)g_pass.layers;
    RAD_INFO("kv streaming: stage pass phase=%d tokens=%lld ctx=%lld: %llu layers, %.1f MiB moved, "
             "%.1f MiB of live history (%.1fx), bound %lld blocks",
             g_pass.phase, (long long)g_pass.tokens, (long long)g_pass.ctx,
             (unsigned long long)g_pass.layers, (double)g_pass.bytes / (1024.0 * 1024.0),
             live / (1024.0 * 1024.0), live > 0.0 ? (double)g_pass.bytes / live : 0.0,
             (long long)g_pass.bound_blocks);
}
}  /* namespace */

/* A pass visits its layers in order and may ask for one layer once per attention launch, so a
 * layer is staged once and a consecutive repeat is already done. */
static void kvstage_begin_pass() {
    report_pass();
    g_last_layer = -1;
    g_pass = PassStage();
}

void kvstream_enter(Ctx* c, const RadBatch* b) {
    kvstage_begin_pass();
    if (!c || !b) return;
    g_pass.tokens = b->n_tok;
    g_pass.ctx    = b->max_ctx_len;
    g_pass.phase  = b->phase;
    for (int i = 0; i < b->n_kv_groups; ++i) {
        if (!b->kv[i].stream) continue;
        KVPoolBinding* kb = c->kv_binding(b->kv[i].group);
        if (!kb || !kb->has_stage) continue;
        const bool was_stage = kb->depth == 1;
        kv_binding_enter(*kb, kv_pass_view(b->phase, b->n_tok, b->n_seq, b->n_spec));
        /* Back on the ring after a staged pass, the residency maps describe bytes the stage has
         * overwritten. A staged pass copies whole layers to the pool at block-id identity (block b
         * at base + b * block_bytes), which overlaps the ring's slots of other layers, and the
         * copy leaves the maps alone: a decode that trusts them reads another layer's, or another
         * request's, KV without a fault. The maps are reset, so the first ring pass resolves every
         * live block from the mirror as a request's first pass does. `slots < blocks` holds for
         * every stream with a stage: bringup gives a stream whose ring covers its context no
         * mirror, no resolve and no stage. */
        if (was_stage && kb->depth == 0) {
            const RadKVStream* st = b->kv[i].stream;
            const int64_t bb     = st->block_bytes;
            const int64_t blocks = bb > 0 ? st->host_layer_stride / bb : 0;
            const int64_t layers = st->host_layer_stride > 0
                                       ? st->host_bytes / st->host_layer_stride : 0;
            if (blocks > 0 && layers > 0 && st->slots > 0 && st->slots < blocks) {
                /* A list of the pass that just ended lands before the refill, not after it. */
                stage_pipe_drain(c);
                (void)rad_memset_async(st->res, 0xff,
                                       layers * blocks * (int64_t)sizeof(int32_t), c->stream());
                (void)rad_memset_async(st->slot_block, 0xff,
                                       layers * st->slots * (int64_t)sizeof(int32_t),
                                       c->stream());
                (void)rad_memset_async(st->slot_state, 0,
                                       layers * st->slots * 2 * (int64_t)sizeof(int32_t),
                                       c->stream());
            }
        }
        /* The history one layer can read, up to the longest sequence's end, for the report. */
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

/* The allocator's high-water mark in bytes of one layer (stage.h, "Volume"). */
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

namespace {

/* A staged layer's copy: from its mirror row (by group-local position, not model layer) to the
 * pool base, where both views start. */
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
    /* Only up to the held bound (stage.h, "Volume"). */
    if (b->stage_live_bytes > 0 && b->stage_live_bytes < o->bytes) o->bytes = b->stage_live_bytes;
    if (o->bytes <= 0 || o->bytes > st->host_layer_stride) o->bytes = st->host_layer_stride;
    return o->bytes > 0;
}

/* Whether the card has the DMA queue the pipelined path needs. */
bool dma_stage(Ctx* c) {
    static const bool on = rad_xp_supported(c->device()) != 0;
    return on;
}

/* The one list in flight per rank thread: the layer it fills and the ride it waits on. */
struct StagePipe {
    RadEvent done = nullptr;
    int      inflight = -1;    /* the layer the outstanding list fills */
    int      group = -1;
    void*    mark = nullptr;
    int64_t  bytes = 0;        /* what that list was told to move, for the pass's counter */
};
thread_local StagePipe g_pipe;
thread_local bool g_warned_ride = false;

/* A list handed over by arm_and_submit() is normally waited for by the next kvstage_wait or
 * consumed by kvstage_next, but a pass can end with one in flight. On the transition back to the
 * ring such a list, still writing base + 0, would land on the ring's layer-0 slots after the
 * refill the map reset causes, and `res` would call another layer's bytes resident. */
static void stage_pipe_drain(Ctx* c) {
    if (g_pipe.inflight < 0) return;
    (void)rad_event_wait(c->stream(), g_pipe.done);
    g_pipe.inflight = -1;
    g_pipe.group    = -1;
    g_pipe.bytes    = 0;
    rad_ride_release(g_pipe.mark, nullptr, 0);
    g_pipe.mark = nullptr;
}

/* Arm the completion word in stream order and hand the list over. The arm (a 1 written by the
 * weight table's patch dispatch, runtime/wtab.h) is also what completes the list's ride, so the
 * list cannot write its 0 before the 1 the wait looks for. */
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
    /* No yield: the compute stream waits on this copy, so yielding the link to a hold whose release
     * is behind that wait would deadlock (docs/KV-STREAMING.md §6.1). */
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

/* The copy on the compute stream: for a backend with no DMA queue, and when no ride is spare. */
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

/* While recording, a host tape step that re-runs the arm and the submit at every play. */
int stage_issue_host(Ctx* c, int group, int layer, bool after) {
    if (c->recording())
        return c->tape_kv_stage(after, group, layer);
    return kvstage_run(c, group, layer, after);
}

}  /* namespace */

/* Before the layer's store: wait for the pipelined copy, or copy now. 1 when staged, 0 when the
 * group has no stage view. A failure fails the step and returns 1, so the caller stops rather
 * than read an unstaged layer. */
static int kvstage_wait(Ctx* c, int group, const RadKVGroupBatch* kvb, int layer) {
    if (!c || !kvb || !kvb->stream) return 0;
    const KVPoolBinding* b = c->kv_binding(group);
    if (!b || !b->has_stage || b->depth != 1) return 0;
    if (layer == g_last_layer) return 1;
    return stage_issue_host(c, group, layer, false);
}

/* After every attention launch of `layer`: start the copy of the group's next layer, riding the
 * next dispatch on the compute stream. Nothing reads the stage again before the next layer's
 * wait. */
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
        /* Make `layer`'s row whole. A pipelined list was handed over at the layer before; its
         * wait is a barrier on the compute stream. */
        if (g_pipe.inflight >= 0 && (g_pipe.inflight != layer || g_pipe.group != group)) {
            /* A list for a layer this pass does not read (one an earlier pass started): it must
             * land before anything rewrites the buffer. */
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
            const int64_t moved = g_pipe.bytes;
            g_pipe.inflight = -1;
            g_pipe.bytes = 0;
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
        /* Nothing pipelined (the pass's first layer): copy now and wait, in the layer's prologue. */
        if (dma_stage(c)) {
            if (!g_pipe.done && rad_event_create_local(&g_pipe.done) < 0) g_pipe.done = nullptr;
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

    /* The group's next bound layer, if any. */
    int next = -1;
    const auto& ls = kb->layer_slot;
    for (int l = layer + 1; l < (int)ls.size(); ++l)
        if (c->kv_layer_index(group, l) >= 0) { next = l; break; }
    if (next < 0 || !dma_stage(c)) return 0;
    if (!g_pipe.done && rad_event_create_local(&g_pipe.done) < 0) g_pipe.done = nullptr;
    if (!g_pipe.done) return 0;
    StageSrc s;
    if (!stage_src(c, group, kvb, next, &s)) return 0;
    /* No spare ride: the next layer's wait copies it on the compute stream. */
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

extern "C" int rad_kv_stage_next(RadCtx* c, rad_kvgroup group, int layer) {
    using namespace rad;
    if (!c) return 0;
    return kvstage_next(static_cast<Ctx*>(c), group, layer);
}
