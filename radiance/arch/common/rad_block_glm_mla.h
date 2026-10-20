/* No-RoPE GLM MLA. prepare() exposes normalized Q-lora for the replicated
 * indexer; finish() consumes its selection. The caller owns the mHC residual. */
#ifndef RAD_BLOCK_GLM_MLA_H
#define RAD_BLOCK_GLM_MLA_H
#include "rad_fp8.h"
#include "rad_block_mla_latent.h"

namespace rad {
namespace arch {
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
        if (!b || !prefix || !kv || g.n_embd != 4096 || g.world < 1 || 64 % g.world || !w.input.x ||
            !w.q_raw || !w.kv_raw || !w.latent || !w.out || !w.q_low.x || !w.attn.x ||
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
        RAD_ARCH_TRY(q_b.declare(b, nm, g, nm.f("blk.%d.attn_q_b", l), 16384 / g.world, 1536,
                                 RAD_SHARD_ROW, wg, { nm.ckpt("%s.q_b_proj.weight", prefix) },
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
        cfg.heads = 64 / g.world;
        RAD_ARCH_TRY(
            core.declare(b, nm, g, layer, kv, cfg, w.core, nm.ckpt("%s.kv_b_proj.weight", prefix)));
        RAD_ARCH_TRY(attn_quant.declare(b, g, w.attn, 16384 / g.world));
        RAD_ARCH_TRY(output.declare(b, nm, g, nm.f("blk.%d.attn_output", layer), 4096,
                                    16384 / g.world, RAD_SHARD_COL, grp_layer(layer),
                                    { nm.ckpt("%s.o_proj.weight", prefix) }, w.attn, w.out));
        if (reduce && g.world > 1)
            op_ar = rw(b,
                       RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", 4096, g.max_tok * 4096),
                                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                         RAD_INT("min_bytes", g.wire_min_bytes)),
                              RAD_NOWEIGHTS),
                       { w.out }, { w.out });
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
