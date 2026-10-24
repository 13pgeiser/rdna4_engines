/* GLM-5.3's sparse latent attention (MLA, no RoPE): the projections and the latent cache store
 * around the absorbed core (GlmMla), and the core itself over the latent cache (MlaLatent), whose
 * complete heads shard as rows of the shared K/V weight. prepare() exposes the normalized q latent
 * for the indexer; finish() consumes the indexer's selection. The caller owns the mHC residual.
 *
 * Under tensor parallelism the head form splits the 64 heads over the ranks (GlmTp::heads): the
 * rows of q_b and kv_b, the columns of o, an all_reduce of the output; q_a, kv_a and the latent
 * cache are replicated. From GlmTp::kSplitRows a prefill-sized step takes the row form instead
 * (Wire::rows, Form::rows): each rank runs its run of the step's rows (GlmTp::run_of) through q_b,
 * absorption, attention, expansion and the output projection with all 64 heads, then an
 * all_gather lays the runs out as the whole step. The latent cache and the selection are whole on
 * every rank already, so the split needs no other exchange. The gather moves half the bytes of
 * the head form's all-reduce, and the attention loads each key once for all 64 heads of a row.
 * The price is the three projections at full width on every rank (126 MB a layer); RowsWire::q
 * unset declares neither them nor the form. */
#ifndef RAD_BLOCK_GLM_MLA_H
#define RAD_BLOCK_GLM_MLA_H
#include "rad_fp8.h"
#include "rad_glm_tp.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace rad {
namespace arch {
struct MlaLatent {
    struct Config {
        int64_t heads = 16, head_dim = 256, latent_dim = 512, blocks = 513;
    };
    struct Wire {
        rad_buf q = 0, q_latent = 0, attn_latent = 0, out = 0, selected = 0, length = 0;
        /* qsa_select's entry count (selected_count), one int per row. Not part of the attention's
         * operands and not a declare requirement: only the streamed resolve needs it, and it fails
         * the step if a streaming group's selection has no count. Never `length`, which is a token
         * count. */
        rad_buf nsel = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup kv = 0;
    rad_weight projection = 0;
    rad_op op_absorb = 0, op_attn = 0, op_expand = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                const Config& cfg, const Wire& wire, const char* source) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        kv = group;
        if (!b || !source || !group || c.heads < 1 || c.blocks < 1 || !w.q || !w.q_latent ||
            !w.attn_latent || !w.out || !w.selected || !w.length || w.q == w.q_latent ||
            w.q_latent == w.attn_latent || w.attn_latent == w.out || c.latent_dim != 512 ||
            c.head_dim != 256)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, group));
        const char* name = nm.f("blk.%d.attn_kv_b.weight", l);
        RAD_ARCH_TRY(map_copy(b, name, source));
        // The conversion recipe decodes this small projection only.
        projection = decl_w(b, name, RAD_BF16, { c.heads * 2 * c.head_dim, c.latent_dim },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        if (!projection) return RAD_E_INVAL;
        op_absorb = rw(
            b,
            RAD_OP(b, "mla_project",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("head_dim", c.head_dim), RAD_INT("latent_dim", c.latent_dim),
                              RAD_STR("mode", "absorb"), RAD_STR("dtype", "bf16")),
                   RAD_WEIGHTS(projection)),
            { w.q }, { w.q_latent });
        op_attn =
            rw(b,
               RAD_OP(b, "mla_attn",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("latent_dim", c.latent_dim),
                                 RAD_INT("heads", c.heads), RAD_INT("blocks", c.blocks),
                                 RAD_INT("block_size", 4),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)c.head_dim)),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               { w.q_latent, w.selected, w.length }, { w.attn_latent });
        if (!op_attn || rad_kv_block_size(b, group) != 4) return RAD_E_INVAL;
        op_expand = rw(
            b,
            RAD_OP(b, "mla_project",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("head_dim", c.head_dim), RAD_INT("latent_dim", c.latent_dim),
                              RAD_STR("mode", "expand"), RAD_STR("dtype", "bf16")),
                   RAD_WEIGHTS(projection)),
            { w.attn_latent }, { w.out });
        return op_absorb && op_attn && op_expand ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, int64_t rows) const {
        RAD_ISSUE_N(ctx, op_absorb, rows, brows(w.q, rows), RAD_W(projection),
                    brows(w.q_latent, rows));
        RAD_ISSUE_N(ctx, op_attn, rows, brows(w.q_latent, rows), kv_cache(kv, layer),
                    brows(w.selected, rows), brows(w.length, rows), brows(w.attn_latent, rows));
        RAD_ISSUE_N(ctx, op_expand, rows, brows(w.attn_latent, rows), RAD_W(projection),
                    brows(w.out, rows));
    }
};

struct GlmMla {
    /* How a prefill-sized step under tensor parallelism runs: `heads` keeps the head form at every
     * step size and spares the full-width weights' VRAM; `rows` is the row form. The plugin
     * chooses (RadBuildCtx::tp_attn_rows); the block takes the choice through whether it is given
     * RowsWire buffers. */
    enum class Form { heads, rows };
    /* The row form's buffers, `rows_for(max_tok)` rows each. `out` (Wire::out) must hold
     * world * rows_for(T) rows, since every rank's run is gathered at full length. */
    struct RowsWire {
        rad_buf q = 0, q_latent = 0, attn_latent = 0, out = 0;
        ActFP8 attn{};
    };
    struct Wire {
        ActFP8 input{}, q_low{}, attn{};
        rad_buf q_raw = 0, kv_raw = 0, latent = 0, out = 0;
        MlaLatent::Wire core{};
        RowsWire rows{};
    };
    /* The gathered row of the row form: 4096 bf16, so a run is ceil(T / world) rows. */
    static constexpr int64_t kOutRowBytes = 4096 * 2;
    static int64_t rows_for(int64_t T, int world) {
        return GlmTp::run_rows(T, world, kOutRowBytes);
    }
    Geom g{};
    Wire w{};
    int layer = 0;
    rad_kvgroup kv = 0;
    LinearFP8 q_a{}, kv_a{}, q_b{}, output{};
    NormQuantFP8 q_norm{};
    QuantFP8 attn_quant{};
    MlaLatent core{};
    rad_weight q_gain = 0, kv_gain = 0;
    rad_op op_kv_norm = 0, op_store = 0, op_store_stream = 0, op_resolve = 0, op_ar = 0;
    LinearFP8 rows_q_b{}, rows_output{};
    QuantFP8 rows_quant{};
    rad_weight rows_kv_b = 0;
    rad_op op_rows_absorb = 0, op_rows_attn = 0, op_rows_expand = 0, op_rows_gather = 0;
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                const char* prefix, rad_kvgroup group, bool reduce = true,
                bool defer_finish = false) {
        g = geom;
        w = wire;
        layer = l;
        kv = group;
        if (!b || !prefix || !kv || g.n_embd != 4096 || g.world < 1 || GlmTp::kHeads % g.world ||
            !w.input.x || !w.q_raw || !w.kv_raw || !w.latent || !w.out || !w.q_low.x || !w.attn.x ||
            w.attn.x != w.core.out)
            return RAD_E_INVAL;
        const auto wg = grp_layer(l);
        RAD_ARCH_TRY(q_a.declare(b, nm, g, nm.f("blk.%d.attn_q_a", l), 1536, 4096, RAD_SHARD_NONE,
                                 wg, { nm.ckpt("%s.q_a_proj.weight", prefix) }, w.input, w.q_raw));
        RAD_ARCH_TRY(kv_a.declare(b, nm, g, nm.f("blk.%d.attn_kv_a", l), 512, 4096, RAD_SHARD_NONE,
                                  wg, { nm.ckpt("%s.kv_a_proj_with_mqa.weight", prefix) }, w.input,
                                  w.kv_raw));
        const char* qn = nm.f("blk.%d.attn_q_a_norm.weight", l);
        const char* kn = nm.f("blk.%d.attn_kv_a_norm.weight", l);
        RAD_ARCH_TRY(map_copy(b, qn, nm.ckpt("%s.q_a_layernorm.weight", prefix)));
        RAD_ARCH_TRY(map_copy(b, kn, nm.ckpt("%s.kv_a_layernorm.weight", prefix)));
        q_gain = decl_w(b, qn, RAD_F32, { 1536 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, wg);
        kv_gain = decl_w(b, kn, RAD_F32, { 512 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, wg);
        if (!q_gain || !kv_gain) return RAD_E_INVAL;
        RAD_ARCH_TRY(q_norm.declare(b, g, q_gain, w.q_raw, w.q_low, 1536));
        op_kv_norm = rw(b,
                        RAD_OP(b, "rmsnorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 512),
                                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(kv_gain)),
                        { w.kv_raw }, { w.latent });
        if (!op_kv_norm) return RAD_E_INVAL;
        RAD_ARCH_TRY(q_b.declare(b, nm, g, nm.f("blk.%d.attn_q_b", l), GlmTp::mla_width(g.world),
                                 1536, RAD_SHARD_ROW, wg, { nm.ckpt("%s.q_b_proj.weight", prefix) },
                                 w.q_low, w.core.q));
        // Cache stores contiguous halves of the normalized latent, with no duplicated K/V.
        op_store = rw(b,
                      RAD_OP(b, "kv_store",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", 256),
                                        RAD_INT("n_head_kv", 1), RAD_INT("block_size", 4),
                                        RAD_STR("kv_dtype", "bf16")),
                             RAD_NOWEIGHTS),
                      { w.latent }, {});
        if (!op_store) return RAD_E_INVAL;
        return defer_finish ? RAD_OK : declare_finish(b, nm, prefix, reduce);
    }
    int declare_finish(RadBuilder* b, Names& nm, const char* prefix, bool reduce = true) {
        MlaLatent::Config cfg;
        cfg.heads = GlmTp::heads(g.world);
        RAD_ARCH_TRY(
            core.declare(b, nm, g, layer, kv, cfg, w.core, nm.ckpt("%s.kv_b_proj.weight", prefix)));
        RAD_ARCH_TRY(attn_quant.declare(b, g, w.attn, GlmTp::mla_width(g.world)));
        RAD_ARCH_TRY(output.declare(b, nm, g, nm.f("blk.%d.attn_output", layer), 4096,
                                    GlmTp::mla_width(g.world), RAD_SHARD_COL, grp_layer(layer),
                                    { nm.ckpt("%s.o_proj.weight", prefix) }, w.attn, w.out));
        /* The store of a group whose KV streams: kv_store's operands and parameters, plus the
         * stream's residency map and this layer's host mirror, which the batch carries and declare
         * cannot see (prepare adds them at issue). A streamed group's store must write the mirror,
         * so falling back to kv_store would leave it empty and a miss would copy zeros. */
        if (rad_op_declared(b, "kv_store_streamed") && rad_kv_block_size(b, kv) == 4)
            op_store_stream = rw(b,
                                RAD_OP(b, "kv_store_streamed",
                                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                                  RAD_INT("head_dim", 256), RAD_INT("n_head_kv", 1),
                                                  RAD_INT("block_size", 4),
                                                  RAD_STR("kv_dtype", "bf16")),
                                       RAD_NOWEIGHTS),
                                { w.latent }, {});
        /* The resolve that rewrites the selection from block ids into ring slots. It is issued in
         * finish, after this layer's indexer selects and before the attention reads `selected`;
         * that ordering is the whole correctness argument. `sel_pitch` is the selection row's
         * pitch, which is what MlaLatent declares its attention with. */
        op_resolve = kv_resolve_declare(b, g, kv, w.core.selected, w.core.nsel, core.c.blocks);
        if (reduce) op_ar = GlmTp::reduce(b, g, w.out);
        if (!op_store || ((reduce && g.world > 1) && !op_ar)) return RAD_E_INVAL;
        return reduce ? declare_rows(b, nm, prefix) : RAD_OK;
    }
    /* Declared whatever the step size, as every TP split here is: arena levels declare again at a
     * smaller max_tok and must get the same graph. The gather's message bound is the all-reduce's,
     * so both share one exchange instance and its scratch. */
    int declare_rows(RadBuilder* b, Names& nm, const char* prefix) {
        const auto& r = w.rows;
        if (!r.q || g.world < 2) return RAD_OK;
        if (!r.q_latent || !r.attn_latent || !r.out || !r.attn.x || r.attn.x == r.q) return RAD_E_INVAL;
        const int64_t R = rows_for(g.max_tok, g.world);
        const auto wg = grp_layer(layer);
        /* Whole views of the head form's container entries: one stored weight each, whose
         * encoding (block fp8 at f32 scales, the decoded bf16 kv_b) the views inherit. */
        rows_q_b.stored = nm.f("blk.%d.attn_q_b.weight", layer);
        RAD_ARCH_TRY(rows_q_b.declare(b, nm, g, nm.f("blk.%d.attn_q_b_rows", layer), 16384, 1536,
                                      RAD_SHARD_NONE, wg, { nm.ckpt("%s.q_b_proj.weight", prefix) },
                                      w.q_low, r.q, R));
        rows_kv_b = decl_view(b, nm.f("blk.%d.attn_kv_b_rows.weight", layer),
                              nm.f("blk.%d.attn_kv_b.weight", layer), nullptr, RAD_BF16,
                              { 64 * 2 * 256, 512 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, wg);
        if (!rows_kv_b) return RAD_E_INVAL;
        auto project = [&](const char* mode, rad_buf in, rad_buf out) {
            return rw(b,
                      RAD_OP(b, "mla_project",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("heads", 64),
                                        RAD_INT("head_dim", 256), RAD_INT("latent_dim", 512),
                                        RAD_STR("mode", mode), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(rows_kv_b)),
                      { in }, { out });
        };
        op_rows_absorb = project("absorb", r.q, r.q_latent);
        op_rows_attn =
            rw(b,
               RAD_OP(b, "mla_attn",
                      RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("latent_dim", 512),
                                 RAD_INT("heads", 64), RAD_INT("blocks", core.c.blocks),
                                 RAD_INT("block_size", 4), RAD_F64("scale", 1.0 / std::sqrt(256.0)),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               { r.q_latent, w.core.selected, w.core.length }, { r.attn_latent });
        op_rows_expand = project("expand", r.attn_latent, r.attn.x);
        RAD_ARCH_TRY(rows_quant.declare(b, g, r.attn, 16384));
        rows_output.stored = nm.f("blk.%d.attn_output.weight", layer);
        RAD_ARCH_TRY(rows_output.declare(b, nm, g, nm.f("blk.%d.attn_output_rows", layer), 4096,
                                         16384, RAD_SHARD_NONE, wg,
                                         { nm.ckpt("%s.o_proj.weight", prefix) }, r.attn, r.out,
                                         R));
        op_rows_gather = rw(b,
                            RAD_OP(b, "all_gather",
                                   RAD_PARAMS(RAD_INT("world_size", g.world),
                                              RAD_RANGE("numel", 4096, g.max_tok * 4096),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                            { r.out }, { w.out });
        return op_rows_absorb && op_rows_attn && op_rows_expand && op_rows_gather ? RAD_OK
                                                                                   : RAD_E_INVAL;
    }
    /* From the TP split threshold, and while the last rank's run is not empty (always at that
     * threshold for up to 8 ranks); the same answer on every rank, as the gather needs. */
    bool rows_take(int64_t T) const {
        return op_rows_gather && GlmTp::splits(T, g.world) &&
               (g.world - 1) * rows_for(T, g.world) < T;
    }
    void prepare(RadCtx* ctx, const RadBatch* batch) const {
        const auto* kb = kv_batch(batch, kv);
        if (!kb || !kb->slot_mapping) {
            rad_step_fail(ctx, "GLM MLA requires latent cache slot mapping");
            return;
        }
        const int64_t T = batch->n_tok;
        q_a.step(ctx, w.input, w.q_raw, T);
        kv_a.step(ctx, w.input, w.kv_raw, T);
        q_norm.step(ctx, T);
        RAD_ISSUE_N(ctx, op_kv_norm, T, brows(w.kv_raw, T), RAD_W(kv_gain), brows(w.latent, T));
        /* The row form splits the query projection into finish_rows (per rank), so q_b runs here
         * only in the head form; the latent store below is full length either way. */
        if (!rows_take(T)) q_b.step(ctx, w.q_low, w.core.q, T);
        const RadKVStream* st = kb->stream;
        if (!st) {
            RAD_ISSUE_N(ctx, op_store, T, bcol(w.latent, 0, 256, T), bcol(w.latent, 256, 256, T),
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
            return;
        }
        /* A streamed group's cache is the host mirror: the ring slot is written only when the block
         * is already resident. The wait must sit immediately before the store, so this chunk's
         * bytes are the stage's last writer (core/kvstream/stage.h ORDERING RULE 1-2). Falling back
         * to kv_store here would write the ring alone, which for a ring smaller than the context is
         * a block id used as a slot, so the step fails instead. */
        (void)kv_stage_wait(ctx, kv, kb, layer);
        const int64_t bbytes = st->block_bytes;
        const int64_t blocks = bbytes > 0 ? st->host_layer_stride / bbytes : 0;
        const int64_t li     = rad_kv_layer_index(ctx, kv, layer);
        const uint8_t* host  = li >= 0 ? st->host + li * st->host_layer_stride : nullptr;
        const int32_t* res   = li >= 0 ? st->res + li * blocks : nullptr;
        if (!op_store_stream || !host || !res || blocks <= 0) {
            rad_step_fail(ctx, "GLM MLA: this layer's KV group streams but the build has no "
                              "kv_store_streamed row to write its host mirror with, so the layer's "
                              "KV would never reach storage");
            return;
        }
        RAD_ISSUE_N(ctx, op_store_stream, T, bcol(w.latent, 0, 256, T),
                    bcol(w.latent, 256, 256, T), praw(kb->slot_mapping, RAD_I32, T),
                    kv_cache(kv, layer), praw(res, RAD_I32, blocks),
                    praw2(host, RAD_U8, blocks, bbytes));
        /* A staged pass reads the pool at block-id identity, and kv_store_streamed's device write is
         * gated on the residency map: a block is written only where the *ring* holds it. The map is
         * empty for every staged prefill before the group's first decode, so without this store the
         * mirror would advance and the stage would not, and the attention would read the previous
         * layer's bytes for this chunk. The identity store is the same bytes at the addressing this
         * view uses, so a staged pass issues it too, after the mirror writer. `kv_stage_layer` is the
         * predicate that gates the resolve below, answered here by the wait just taken: the layer is
         * staged and must not resolve. */
        if (kv_stage_layer(ctx, kv, kb, layer))
            RAD_ISSUE_N(ctx, op_store, T, bcol(w.latent, 0, 256, T), bcol(w.latent, 256, 256, T),
                        praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
    }
    void finish(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        const auto* kb = kv_batch(batch, kv);
        if (kb && kb->stream) {
            /* One `selected` buffer serves the whole model (glm5next.cpp assigns m.iw.selected =
             * m.aw.core.selected), so this layer's indexer must have re-emitted fresh ids
             * immediately above. Only one full layer resolves at a time, and only on a pass that
             * takes the ring; a staged pass addresses block ids directly through the identity
             * stage. Resolving outside that order makes layer n+1 resolve layer n's ring slots as
             * block ids: in range, fluent, wrong. */
            if (kv_stage_layer(ctx, kv, kb, layer)) {
                /* staged: this pass leaves `selected` as block ids and must not resolve */
            } else if (!w.core.nsel) {
                rad_step_fail(ctx, "GLM MLA: the KV group streams but the selection has no entry "
                                  "count (selected_count), so kv_resolve cannot rewrite it");
                return;
            } else if (!op_resolve) {
                rad_step_fail(ctx, "GLM MLA: this layer's KV group streams but the build has no "
                                  "kv_resolve row to rewrite its selection with, so the attention "
                                  "would read block ids as ring slots");
                return;
            } else if (T > 0) {
                const RadKVStream* st = kb->stream;
                const int64_t bbytes = st->block_bytes;
                const int64_t blocks = bbytes > 0 ? st->host_layer_stride / bbytes : 0;
                const int64_t slots  = st->slots;
                const int64_t li     = rad_kv_layer_index(ctx, kv, layer);
                /* `host` is null exactly when `li < 0`, so it is the whole test for a mirror row. */
                const uint8_t* host  = li >= 0 ? st->host + li * st->host_layer_stride : nullptr;
                if (!host || blocks <= 0 || slots <= 0) {
                    rad_step_fail(ctx, "GLM MLA: the KV group streams but this layer has no host "
                                      "mirror row or residency map to resolve against");
                    return;
                }
                /* The count operand is qsa_select's second output, the number of entries it wrote,
                 * never `length` (its third output, a token count). The device clamps a too-large
                 * count to sel_pitch and skips the -1 padding, so the wrong buffer serves the right
                 * slots on the card while the reference refuses the same call -- the oracle dies
                 * quietly (tests/arch_test.cpp asserts this handle). */
                RAD_ISSUE_N(ctx, op_resolve, T,
                            brow_slice(w.core.selected, 0, T, core.c.blocks),
                            brow_slice(w.core.nsel, 0, T, 1),
                            praw(st->res + li * blocks, RAD_I32, blocks),
                            praw(st->slot_block + li * slots, RAD_I32, slots),
                            praw2(st->slot_state + li * slots * 2, RAD_I32, slots, 2),
                            praw(st->ctl, RAD_I32, 16),
                            praw2(host, RAD_U8, blocks, bbytes),
                            kv_cache(kv, layer));
            }
        }
        /* The row form (upstream 1.12.0): a prefill-sized step under --tp-attn-split runs the
         * attention over a rank's own rows and gathers, so the resolve above already rewrote
         * `selected` for every row before either form reads it. */
        if (rows_take(T)) {
            finish_rows(ctx, T);
        } else {
            core.step(ctx, T);
            attn_quant.step(ctx, T);
            output.step(ctx, w.attn, w.out, T);
            if (op_ar) RAD_ISSUE_N(ctx, op_ar, T * 4096, brows(w.out, T), RAD_NONE);
        }
        /* Behind every attention launch of this layer, so the next layer's copy starts only once
         * the last read of the aliased stage is behind it (core/kvstream/stage.h). */
        if (kb && kb->stream) kv_stage_next(ctx, kv, layer);
    }
    /* This rank's rows [first, first + n) end to end; the last rank's run may be short, and the
     * gather sends its unwritten tail rows, which land past T. */
    void finish_rows(RadCtx* ctx, int64_t T) const {
        const auto& r = w.rows;
        const GlmTp::Run run = GlmTp::run_of(T, g.world, g.rank, kOutRowBytes);
        const int64_t first = run.first, n = run.rows;
        rows_q_b.step(ctx, w.q_low, r.q, T, first, n, 0);
        RAD_ISSUE_N(ctx, op_rows_absorb, n, brows(r.q, n), RAD_W(rows_kv_b), brows(r.q_latent, n));
        RAD_ISSUE_N(ctx, op_rows_attn, n, brows(r.q_latent, n), kv_cache(kv, layer),
                    brow_slice(w.core.selected, first, n, core.c.blocks),
                    brow_slice(w.core.length, first, n, 1), brows(r.attn_latent, n));
        RAD_ISSUE_N(ctx, op_rows_expand, n, brows(r.attn_latent, n), RAD_W(rows_kv_b),
                    brows(r.attn.x, n));
        rows_quant.step(ctx, n);
        rows_output.step(ctx, r.attn, r.out, n);
        RAD_ISSUE_N(ctx, op_rows_gather, run.run * 4096, brows(r.out, run.run),
                    brows(w.out, g.world * run.run));
    }
};
}
} // namespace rad::arch
#endif
