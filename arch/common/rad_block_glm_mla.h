/* GLM-5.3's sparse latent attention (MLA, no RoPE): the projections and the latent cache store
 * around the absorbed core (GlmMla), and the core itself over the latent cache (MlaLatent), whose
 * complete heads shard as rows of the shared K/V weight. prepare() exposes the normalized q latent
 * for the indexer; finish() consumes the indexer's selection. The caller owns the mHC residual.
 *
 * Under tensor parallelism the 64 heads split over the ranks (GlmTp::heads): the rows of q_b and
 * kv_b, the columns of o, an all_reduce of the output; q_a, kv_a and the latent cache are
 * replicated. */
#ifndef RAD_BLOCK_GLM_MLA_H
#define RAD_BLOCK_GLM_MLA_H
#include "rad_fp8.h"
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct MlaLatent {
    struct Config {
        int64_t heads = 16, head_dim = 256, latent_dim = 512, blocks = 513;
    };
    struct Wire {
        rad_buf q = 0, q_latent = 0, attn_latent = 0, out = 0, selected = 0, length = 0;
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
    struct Wire {
        ActFP8 input{}, q_low{}, attn{};
        rad_buf q_raw = 0, kv_raw = 0, latent = 0, out = 0;
        MlaLatent::Wire core{};
    };
    Geom g{};
    Wire w{};
    int layer = 0;
    rad_kvgroup kv = 0;
    LinearFP8 q_a{}, kv_a{}, q_b{}, output{};
    NormQuantFP8 q_norm{};
    QuantFP8 attn_quant{};
    MlaLatent core{};
    rad_weight q_gain = 0, kv_gain = 0;
    rad_op op_kv_norm = 0, op_store = 0, op_ar = 0;
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
        if (reduce) op_ar = GlmTp::reduce(b, g, w.out);
        return op_store && (!(reduce && g.world > 1) || op_ar) ? RAD_OK : RAD_E_INVAL;
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
        q_b.step(ctx, w.q_low, w.core.q, T);
        RAD_ISSUE_N(ctx, op_store, T, bcol(w.latent, 0, 256, T), bcol(w.latent, 256, 256, T),
                    praw(kb->slot_mapping, RAD_I32, T), kv_cache(kv, layer));
    }
    void finish(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        core.step(ctx, T);
        attn_quant.step(ctx, T);
        output.step(ctx, w.attn, w.out, T);
        if (op_ar) RAD_ISSUE_N(ctx, op_ar, T * 4096, brows(w.out, T), RAD_NONE);
    }
};
}
} // namespace rad::arch
#endif
