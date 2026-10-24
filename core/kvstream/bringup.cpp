/* bringup.cpp -- see bringup.h. */
#include "kvstream/bringup.h"

#include "device/device.h"
#include "engine_hostmem.h"
#include "kvstream/plan.h"

#include <algorithm>
#include <cstring>

namespace rad {

/* Every op whose only job is the host mirror or the ring. `kv_store` is not here -- a group that
 * does not stream uses it -- and neither is the fused `qk_norm_rope_gate_kv_store`, whose streamed
 * form (below) is a second variant of a fold that already exists without it. */
static const char* const kStreamOnlyOps[] = {
    "kv_store_streamed",
    "qk_norm_rope_gate_kv_store_streamed",
    "kv_resolve",
};

bool kvstream_op_gated(const RadBuildCtx& ctx, const char* op) {
    if (!op) return false;
    if (ctx.kv_stream_ok != 0 && ctx.kv_resident_blocks > 0) return false; /* a ring is carved */
    for (const char* s : kStreamOnlyOps)
        if (std::strcmp(s, op) == 0) return true;
    return false;
}

int kvstream_configure(const Config& cfg, int64_t ctx, const std::vector<KVGroupInfo>& groups,
                       const Program& prog, KVStreamSetup* out) {
    *out = KVStreamSetup{};
    /* --kv-resident is the whole switch: absent or 0 and nothing streams, which is the base
     * engine's configuration and costs nothing. */
    if (cfg.kv_resident <= 0) return RAD_OK;
    /* One group could stream: the paged full-attention group with the largest page. On Flash-Next
     * that is `kv_attn`; the indexer's `kv_qsa_bkey` is a full group too but a fraction of the
     * width and `qsa_score` reads it densely over the whole history every step, so it must stay
     * resident (docs/KV-STREAMING.md §2). The writer maintains the host mirror
     * (`kv_store_streamed`), so a miss copies a block the engine wrote. A model whose paged groups
     * are all windowed has nothing a ring could hold back, so a ring on it is not an error: the
     * flag simply changes nothing. */
    int32_t g = kv_stream_group(groups, &prog);
    /* No group has a streamed path, so nothing will stream -- but a ring that would have streamed
     * still has to name the op the architecture is missing, and the message is written against a
     * group. Fall back to the old width rule for the report only; kvstream_ops below
     * then refuses the path exactly as before. */
    if (g < 0) g = kv_stream_group(groups);
    if (g < 0) return RAD_OK;
    const KVGroupInfo& gi = groups[(size_t)g];
    const int64_t bs = gi.block_size > 0 ? gi.block_size : 1;
    /* The ring's depth against the group's id space, without deciding anything yet: a ring that
     * covers the context is not a refusal -- the group stays a plain resident pool, decided here
     * from --max-num-seqs and --max-model-len, both known long before the first token. */
    const KVStreamRing shape = kvstream_ring_shape(cfg, ctx, bs);
    out->asked_tokens   = cfg.kv_resident;
    out->context_tokens = ctx * cfg.max_seqs;
    out->group_name     = gi.name;
    out->ring_blocks    = shape.ring_blocks;
    out->ring_tokens    = shape.ring_blocks * bs;
    out->block_size     = bs;
    out->ctx_blocks     = shape.ctx_blocks;
    if (!shape.streams()) return RAD_OK;
    /* Fail safe before anything is allocated: the group the plan wants to stream must have the
     * ops its streamed path needs, declared by the plugin that emitted this program. Without
     * kv_resolve a streamed group's `sel` still holds block ids while the attention reads them as
     * ring slots -- past a ring smaller than the id space, which is a fault or silent corruption,
     * not a slow path; without kv_store_streamed nothing writes the host mirror, so a miss reads
     * zeros. Two paths declare both today -- the gated-fp8 attention (arch/common/
     * rad_block_attn_gated_fp8.h) and GLM-5.3's MLA block (arch/common/rad_block_glm_mla.h),
     * which resolves at finish after its indexer selects. Ignored with one line, then exactly the
     * configuration without the flag: no cap, no mirror, no maps, no stage, a null RadKVStream. */
    const KVStreamOps ops = kvstream_ops(prog, &gi);
    if (!ops.streams()) {
        /* Two causes, two sentences. The architecture can declare no streamed path at all (the
         * gated path emits a plain kv_store and no resolve), or it can declare one the kernel
         * table has no row for: a packed cache is declared to the core as plain E4M3, so the dtype
         * gate below cannot see it, and the streamed store's rows are E4M3 and bf16 only. The
         * second case must not say "the architecture declares no store" -- it did -- and the
         * operator's answer is the cache layout, not the model. */
        if ((cfg.kv_cache_dtype == "fp8_v4" || cfg.kv_cache_dtype == "k6_v4") && ops.resolve &&
            !ops.store && !ops.folded)
            RAD_WARN("--kv-resident %lld ignored: group '%s' has a %s cache, and the streamed store "
                     "has no writer for that layout (E4M3 and bf16 only), so nothing streams. "
                     "Serving exactly as without --kv-resident: every block of the group stays on "
                     "the card, with no host mirror, no residency map and no stage view",
                     (long long)cfg.kv_resident, gi.name.c_str(), cfg.kv_cache_dtype.c_str());
        else
            RAD_WARN("--kv-resident %lld ignored: architecture '%s' declares no %s for group '%s', so "
                     "this model has no streamed path and nothing streams. Serving exactly as without "
                     "--kv-resident: every block of the group stays on the card, with no host mirror, "
                     "no residency map and no stage view",
                     (long long)cfg.kv_resident,
                     prog.meta.arch_id && *prog.meta.arch_id ? prog.meta.arch_id : "(unnamed)",
                     ops.missing(), gi.name.c_str());
        *out = KVStreamSetup{};
        return RAD_OK;
    }
    /* The streamed store writes the mirror with the cache's own conversion: E4M3 for the fp8 QSA
     * family, bf16 for the default cache dtype (kv_store_streamed_bf16). Streaming is
     * dtype-agnostic beyond the two writers -- the mirror and the ring are byte planes of one
     * paged layout whose pitch is `block_bytes` -- so a bf16 group streams. A packed fp8_v4/k6_v4
     * group is declared to the core as plain E4M3 and its architecture emits no streamed store for
     * the layout, so it is warned about and ignored above, never refused here. This gate is the
     * last line for a group whose core dtype is neither of the two -- f32 or f16, which no
     * architecture ships a full group in today. Checked only where something streams, so a
     * configuration that fits is never refused for it. */
    if (gi.decl.dtype != RAD_F8E4M3 && gi.decl.dtype != RAD_BF16) {
        RAD_ERR("--kv-resident %lld: group '%s' has a %s cache, and the streamed store has E4M3 "
                "and bf16 rows only (--kv-cache-dtype fp8 or bf16); a packed cache is a later "
                "unit's row. Leave --kv-resident off to keep every block on the card",
                (long long)cfg.kv_resident, out->group_name.c_str(),
                rad_dtype_name(gi.decl.dtype));
        return RAD_E_INVAL;
    }
    /* The ring's depth: a ring below what one decode step can select is raised to it, with the one
     * line that says what was asked, what it became and why. */
    KVStreamRing ring;
    kvstream_ring(cfg, ctx, bs, ops.sel_pitch, &ring);
    if (ring.raised_blocks > 0)
        /* Anything under the computed amount is ignored and pulled up to it, the way Strata sizes
         * this: the engine knows what a decode step names, and --kv-resident is a request for
         * streaming rather than for one number. A warning, so startup proceeds -- and a different
         * sentence from the ignored-flag one above, because this model does stream. */
        RAD_WARN("--kv-resident %lld raised to %lld tokens: a decode step can name %lld blocks of "
                 "%lld blocks each (--max-num-seqs %lld x %lld for speculation x %lld), and the ring "
                 "must hold them all; the ring takes the raised depth and startup proceeds",
                 (long long)out->asked_tokens, (long long)(ring.ring_blocks * bs),
                 (long long)ring.raised_blocks, (long long)bs,
                 (long long)(cfg.max_seqs > 0 ? cfg.max_seqs : 1),
                 (long long)(1 + (cfg.n_spec > 0 ? cfg.n_spec : 0)), (long long)ops.sel_pitch);
    /* The raised ring may be the whole context, which is not a streamed group at all: it keeps
     * every block on the card and the report says which context the ring covers. */
    out->ring_blocks   = ring.ring_blocks;
    out->ring_tokens   = ring.ring_blocks * bs;
    out->raised_tokens = ring.raised_blocks > 0 ? ring.ring_blocks * bs : 0;
    out->group         = ring.streams() ? g : -1;
    /* Prefill runs on the stage and never resolves: above `qsa_exact_to` one launch's selection is
     * a chunk's union, which no serving ring could hold, so the pass gets the group's context one
     * layer deep, addressed by block id. A within-(1+n_spec) decode pass takes the ring and
     * resolves; a decode-classified pass with more rows a sequence than that (the serial drafter's
     * history pass over a chunk) is prefill-shaped and stages, because its union is the histories it
     * spans. */
    out->stage_blocks = ring.streams() ? ring.ctx_blocks : 0;
    return RAD_OK;
}

void kvstream_host_add(KVStreamHostUse* use, const KVManager& kv) {
    const KVStreamCost c = kv.stream_cost();
    use->sum.groups     += c.groups;
    use->sum.host_bytes += c.host_bytes;
    use->sum.map_bytes  += c.map_bytes;
    use->max_rank_bytes = std::max(use->max_rank_bytes, c.host_bytes);
}

int kvstream_host_gate(const KVStreamHostUse& use, int n_ranks, int64_t still, int64_t headroom) {
    if (use.sum.host_bytes <= 0) return RAD_OK;
    std::string why;
    if (lockable_room(use.sum.host_bytes, still, headroom, false, &why)) return RAD_OK;
    RAD_ERR("kv streaming: the host mirror for %lld streamed KV group(s) is up to %s a rank "
            "(%s over %d ranks) of pinned host memory, and %s. It is sized by "
            "--max-model-len and --max-num-seqs (one full context a sequence), not by "
            "--kv-resident, which sizes the card's ring. Lower one of the first two, or "
            "--kv-resident 0 to keep every block on the card. --host-pool-mib is the "
            "experts' tier and holding less of it does not make room for this one.",
            (long long)use.sum.groups, humanb(use.max_rank_bytes).c_str(),
            humanb(use.sum.host_bytes).c_str(), n_ranks, why.c_str());
    return RAD_E_NOMEM;
}

int kvstream_fill(KVManager& kv) {
    /* After the carve, which is where an elastic pool backs its range: a fill before that
     * would touch pages the card has not been given. One bring-up stream, drained before
     * the first pass, since the maps are read from the first step on. */
    RadStream st = nullptr;
    int rc = rad_stream_create(&st, 0);
    if (rc == RAD_OK) rc = kv.init_stream(st);
    if (rc == RAD_OK) rc = rad_stream_sync(st);
    rad_stream_destroy(st);
    return rc;
}

void kvstream_report(const KVStreamSetup& s, const KVStreamHostUse& host, int n_ranks) {
    /* The flag is not the ring when the ceiling raised it, and this report is what people read
     * afterwards: the flag's number, the ring's, and why, before anything below quotes the ring. */
    const int64_t flag = s.raised_tokens > 0 ? s.raised_tokens : s.asked_tokens;
    if (s.raised_tokens > 0 && s.block_size > 0)
        RAD_INFO("kv streaming: the ring is the raised one, not the flag's: --kv-resident %lld "
                 "became %lld tokens (%lld -> %lld blocks of %lld tokens each), the fewest blocks "
                 "one decode step's selection fits in; what was carved below is that ring",
                 (long long)s.asked_tokens, (long long)s.raised_tokens,
                 (long long)((s.asked_tokens + s.block_size - 1) / s.block_size),
                 (long long)(s.raised_tokens / s.block_size), (long long)s.block_size);
    if (host.sum.groups > 0)
        RAD_INFO("kv streaming: %lld group(s) stream (%s): kv_store_streamed writes the host "
                 "mirror and the resident ring slots, kv_resolve rewrites each selection as ring "
                 "slots, up to %s of pinned host mirror a rank (%s over %d), %s of ring maps in "
                 "the KV pools; the pool is one reservation of max(ring, stage), with the ring "
                 "sized by --kv-resident",
                 (long long)host.sum.groups, s.group_name.c_str(),
                 humanb(host.max_rank_bytes).c_str(), humanb(host.sum.host_bytes).c_str(), n_ranks,
                 humanb(host.sum.map_bytes).c_str());
    if (s.stage_blocks > 0)
        RAD_INFO("kv streaming: prefill staging, group '%s' ring %lld blocks, one-layer stage %lld "
                 "blocks (one layer of its %lld-block context): a prefill or mixed step, and a decode "
                 "pass with more rows a sequence than a decode step can have, stages each layer and "
                 "never resolves; only a within-(1+n_spec) decode pass uses the ring",
                 s.group_name.c_str(), (long long)s.ring_blocks,
                 (long long)s.stage_blocks, (long long)s.ctx_blocks);
    if (s.group < 0 && s.asked_tokens > 0 && s.ctx_blocks > 0)
        RAD_INFO("kv streaming: nothing streams: --kv-resident %lld covers group '%s''s "
                 "%lld-token context (%lld blocks), so every block keeps its own slot and the "
                 "group runs the engine's ordinary resident path -- no host mirror, no residency "
                 "map, no stage and no resolve. Lower --kv-resident below %lld to stream it",
                 (long long)flag, s.group_name.c_str(), (long long)s.context_tokens,
                 (long long)s.ctx_blocks, (long long)s.context_tokens);
}

}  /* namespace rad */
