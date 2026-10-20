/* GLM KDA projections around the recurrent/cache core. These checkpoint weights
 * are BF16. The two low-rank gate paths have no intervening activation. The
 * caller supplies normalized input and owns the mHC residual connection. */
#ifndef RAD_BLOCK_GLM_KDA_H
#define RAD_BLOCK_GLM_KDA_H
#include "rad_fp8.h"
#include "rad_block_kda_state.h"

namespace rad {
namespace arch {
struct GlmKda {
    struct Wire {
        rad_buf input = 0, forget_low = 0, gate_low = 0, out = 0;
        KdaState::Wire core{};
    };
    Geom g{};
    Wire w{};
    KdaState core{};
    LinearFP8 qkv{}, beta{}, forget_a{}, forget_b{}, gate_a{}, gate_b{}, output{};
    rad_op op_ar = 0;
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const Wire& wire,
                const char* prefix, const KdaState::Src& src, rad_kvgroup state,
                rad_kvgroup history, bool reduce = true) {
        g = geom;
        w = wire;
        if (!b || !prefix || g.n_embd != 4096 || g.world < 1 || 64 % g.world || !w.input ||
            !w.forget_low || !w.gate_low || !w.out || w.input == w.out ||
            w.forget_low == w.gate_low)
            return RAD_E_INVAL;
        // All seven projections are plain BF16 in the pinned checkpoint.
        Geom bg = g;
        bg.w_bf16 = true;
        const int64_t width = 8192 / g.world, heads = 64 / g.world;
        const auto group = grp_layer(layer);
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        const auto input = act(w.input);
        RAD_ARCH_TRY(qkv.declare(
            b, nm, bg, nm.f("blk.%d.attn_qkv", layer), 3 * width, 4096, RAD_SHARD_ROW, group,
            { nm.ckpt("%s.q_proj.weight", prefix), nm.ckpt("%s.k_proj.weight", prefix),
              nm.ckpt("%s.v_proj.weight", prefix) },
            input, w.core.qkv, 0, { width, width, width }));
        RAD_ARCH_TRY(beta.declare(b, nm, bg, nm.f("blk.%d.attn_beta", layer), heads, 4096,
                                  RAD_SHARD_ROW, group, { nm.ckpt("%s.b_proj.weight", prefix) },
                                  input, w.core.beta));
        RAD_ARCH_TRY(forget_a.declare(
            b, nm, bg, nm.f("blk.%d.attn_forget_a", layer), 128, 4096, RAD_SHARD_NONE, group,
            { nm.ckpt("%s.f_a_proj.weight", prefix) }, input, w.forget_low));
        RAD_ARCH_TRY(forget_b.declare(
            b, nm, bg, nm.f("blk.%d.attn_forget_b", layer), width, 128, RAD_SHARD_ROW, group,
            { nm.ckpt("%s.f_b_proj.weight", prefix) }, act(w.forget_low), w.core.a));
        RAD_ARCH_TRY(gate_a.declare(b, nm, bg, nm.f("blk.%d.attn_gate_a", layer), 128, 4096,
                                    RAD_SHARD_NONE, group,
                                    { nm.ckpt("%s.g_a_proj.weight", prefix) }, input, w.gate_low));
        RAD_ARCH_TRY(gate_b.declare(b, nm, bg, nm.f("blk.%d.attn_gate_b", layer), width, 128,
                                    RAD_SHARD_ROW, group, { nm.ckpt("%s.g_b_proj.weight", prefix) },
                                    act(w.gate_low), w.core.gate));
        KdaState::Config cfg;
        cfg.heads = heads;
        RAD_ARCH_TRY(core.declare(b, nm, bg, layer, state, history, cfg, w.core, src));
        RAD_ARCH_TRY(output.declare(b, nm, bg, nm.f("blk.%d.attn_output", layer), 4096, width,
                                    RAD_SHARD_COL, group, { nm.ckpt("%s.o_proj.weight", prefix) },
                                    act(w.core.out), w.out));
        if (!qkv.bf16 || !beta.bf16 || !forget_a.bf16 || !forget_b.bf16 || !gate_a.bf16 ||
            !gate_b.bf16 || !output.bf16)
            return RAD_E_UNSUPPORTED;
        if (reduce && g.world > 1)
            op_ar = rw(b,
                       RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", 4096, g.max_tok * 4096),
                                         RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                         RAD_INT("min_bytes", g.wire_min_bytes)),
                              RAD_NOWEIGHTS),
                       { w.out }, { w.out });
        return reduce && g.world > 1 && !op_ar ? RAD_E_INVAL : RAD_OK;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        const auto input = act(w.input);
        qkv.step(ctx, input, w.core.qkv, T);
        beta.step(ctx, input, w.core.beta, T);
        forget_a.step(ctx, input, w.forget_low, T);
        forget_b.step(ctx, act(w.forget_low), w.core.a, T);
        gate_a.step(ctx, input, w.gate_low, T);
        gate_b.step(ctx, act(w.gate_low), w.core.gate, T);
        core.step(ctx, batch);
        output.step(ctx, act(w.core.out), w.out, T);
        if (op_ar) RAD_ISSUE_N(ctx, op_ar, T * 4096, brows(w.out, T), RAD_NONE);
    }
};
}
} // namespace rad::arch
#endif
