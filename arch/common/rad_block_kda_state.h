/* GLM's recurrent core over already projected inputs. ple_conv supplies exactly
 * causal depthwise convolution + SiLU, including cold start and accepted-history rollback. */
#ifndef RAD_BLOCK_KDA_STATE_H
#define RAD_BLOCK_KDA_STATE_H
#include "rad_arch.h"
#include <cmath>

namespace rad {
namespace arch {
struct KdaState {
    struct Config {
        int64_t heads = 16, dim = 128, width = 4;
        double lower = -5;
    };
    struct Src {
        const char *q_conv = nullptr, *k_conv = nullptr, *v_conv = nullptr, *alog = nullptr,
                   *bias = nullptr, *norm = nullptr;
    };
    struct Wire {
        rad_buf qkv = 0, conv = 0, a = 0, beta = 0, gate = 0, out = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup state = 0, history = 0;
    rad_weight taps = 0, alog = 0, bias = 0, norm = 0;
    rad_op op_conv = 0, op_verify = 0, op_prefill = 0, op_norm = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup st, rad_kvgroup cv,
                const Config& cfg, const Wire& wire, const Src& src) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        state = st;
        history = cv;
        if (!b || !st || !cv || c.heads < 1 || c.dim != 128 || c.width != 4 || g.max_spec > 31 ||
            !std::isfinite(c.lower) || c.lower >= 0 || !w.qkv || !w.conv || !w.a || !w.beta ||
            !w.gate || !w.out || w.qkv == w.conv || !src.q_conv || !src.k_conv || !src.v_conv ||
            !src.alog || !src.bias || !src.norm)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, st));
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, cv));
        const int64_t width = c.heads * c.dim;
        const char *tn = nm.f("blk.%d.ssm_conv.weight", l), *an = nm.f("blk.%d.ssm_a", l),
                   *bn = nm.f("blk.%d.ssm_dt_bias", l), *nn = nm.f("blk.%d.ssm_norm.weight", l);
        RAD_ARCH_TRY(map_concat(b, tn, { src.q_conv, src.k_conv, src.v_conv }));
        RAD_ARCH_TRY(map_copy(b, an, src.alog));
        RAD_ARCH_TRY(map_copy(b, bn, src.bias));
        RAD_ARCH_TRY(map_copy(b, nn, src.norm));
        taps = decl_w(b, tn, RAD_BF16, { 3 * width, c.width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                      grp_layer(l), 0, { width, width, width });
        alog =
            decl_w(b, an, RAD_F32, { c.heads }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        bias = decl_w(b, bn, RAD_F32, { width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        norm =
            decl_w(b, nn, RAD_F32, { c.dim }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        if (!taps || !alog || !bias || !norm) return RAD_E_INVAL;
        op_conv = rw(b,
                     RAD_OP(b, "ple_conv",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 3 * width),
                                       RAD_INT("width", c.width), RAD_INT("dilation", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(taps)),
                     { w.qkv }, { w.conv });
        for (int verify = 0; verify < 2; ++verify) {
            const rad_op op = rw(
                b,
                RAD_OP(b, "kda_update",
                       RAD_PARAMS(
                           RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                           RAD_INT("head_k", c.dim), RAD_INT("head_v", c.dim),
                           RAD_F64("lower_bound", c.lower),
                           RAD_F64("scale", 1.0 / std::sqrt((double)c.dim)),
                           RAD_F64("l2_eps", 1e-6), RAD_STR("mode", verify ? "verify" : "prefill"),
                           RAD_STR("dtype", "bf16"),
                           RAD_STR("state_form", g.max_spec > 0 ? "anchor" : "per_candidate")),
                       RAD_WEIGHTS(alog, bias)),
                { w.conv, w.a, w.beta }, { w.out });
            if (verify)
                op_verify = op;
            else
                op_prefill = op;
        }
        op_norm = rw(b,
                     RAD_OP(b, "gdn_gated_rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", c.dim),
                                       RAD_F64("eps", g.eps), RAD_STR("act", "sigmoid")),
                            RAD_WEIGHTS(norm)),
                     { w.out, w.gate }, { w.out });
        return op_conv && op_verify && op_prefill && op_norm ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const auto *st = kv_batch(batch, state), *cv = kv_batch(batch, history);
        if (!st || !cv || !st->state_index || !cv->state_index) {
            rad_step_fail(ctx, "KDA requires recurrent and convolution state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq, width = c.heads * c.dim;
        const int64_t pitch = st->state_index_pitch > 0 ? st->state_index_pitch : 1;
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        for (int verify = 1; verify >= 0; --verify) {
            const int64_t seq0 = verify ? 0 : D, seqs = verify ? D : S - D, rows = verify ? DT : T;
            if (!seqs) continue;
            const RadOperand cu = praw(batch->cu_seqlens + seq0, RAD_I32, seqs + 1);
            const RadOperand init = praw(batch->ctx_lens + seq0, RAD_I32, seqs);
            const RadOperand accepted =
                verify ? praw(batch->num_accepted, RAD_I32, seqs) : RAD_NONE;
            RAD_ISSUE_N(ctx, op_conv, rows, brows(w.qkv, rows), RAD_W(taps),
                        kv_cache(history, layer), cu, RAD_NONE,
                        praw(cv->state_index + seq0, RAD_I32, seqs), init, accepted,
                        brows(w.conv, rows));
            RAD_ISSUE_N(ctx, verify ? op_verify : op_prefill, rows, bcol(w.conv, 0, width, rows),
                        bcol(w.conv, width, width, rows), bcol(w.conv, 2 * width, width, rows),
                        brows(w.a, rows), brows(w.beta, rows), RAD_W(alog), RAD_W(bias),
                        kv_cache(state, layer),
                        praw2(st->state_index + seq0 * pitch, RAD_I32, seqs, pitch), cu, accepted,
                        init, brows(w.out, rows));
        }
        RAD_ISSUE_N(ctx, op_norm, T, brows(w.out, T), brows(w.gate, T), RAD_W(norm),
                    brows(w.out, T));
    }
};
}
} // namespace rad::arch
#endif
