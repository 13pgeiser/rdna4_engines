/* Replicated GLM indexer front end. Casts bridge ordinary linear rows to the
 * flattened per-head quantizer and join normalized keys with learned pool gates. */
#ifndef RAD_BLOCK_GLM_INDEXER_H
#define RAD_BLOCK_GLM_INDEXER_H
#include "rad_fp8.h"
#include "rad_block_dsa_cache.h"
#include "rad_tp_split.h"

namespace rad {
namespace arch {
struct GlmIndexer {
    struct Wire {
        rad_buf input = 0, q_low = 0, q_wide = 0, kg_raw = 0;
        DsaCache::Wire cache{};
    };
    Geom g{};
    Wire w{};
    LinearFP8 query{}, key_gate{};
    DsaCache cache{};
    rad_weight gain = 0, bias = 0, head_weight = 0;
    rad_op op_norm = 0, op_query_copy = 0, op_gate_copy = 0, op_head = 0;
    SplitF32Proj head_split{};
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                const char* prefix, const DsaCache::Config& cfg, rad_kvgroup attention,
                rad_kvgroup tail, rad_kvgroup* index) {
        g = geom;
        w = wire;
        if (!b || !prefix || g.n_embd != 4096 || !w.input || !w.q_low || !w.q_wide || !w.kg_raw)
            return RAD_E_INVAL;
        Geom bg = g;
        bg.w_bf16 = true;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        RAD_ARCH_TRY(query.declare(b, nm, bg, nm.f("blk.%d.index_query", l), 4096, 1536,
                                   RAD_SHARD_NONE, grp_layer(l),
                                   { nm.ckpt("%s.wq_b.weight", prefix) }, act(w.q_low), w.q_wide));
        RAD_ARCH_TRY(key_gate.declare(
            b, nm, bg, nm.f("blk.%d.index_key_gate", l), 256, 4096, RAD_SHARD_NONE, grp_layer(l),
            { nm.ckpt("%s.wk.weight", prefix), nm.ckpt("%s.index_kpool_compress_gate", prefix) },
            act(w.input), w.kg_raw, 0, { 128, 128 }));
        const char *gn = nm.f("blk.%d.index_norm.weight", l),
                   *bn = nm.f("blk.%d.index_norm.bias", l),
                   *hn = nm.f("blk.%d.index_head.weight", l);
        RAD_ARCH_TRY(map_copy(b, gn, nm.ckpt("%s.k_norm.weight", prefix)));
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.k_norm.bias", prefix)));
        RAD_ARCH_TRY(map_copy(b, hn, nm.ckpt("%s.weights_proj.weight", prefix)));
        gain = decl_w(b, gn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        bias = decl_w(b, bn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        head_weight = decl_w(b, hn, RAD_F32, { 32, 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                             grp_layer(l));
        if (!gain || !bias || !head_weight) return RAD_E_INVAL;
        op_norm = rw(b,
                     RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                       RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(gain, bias)),
                     { w.kg_raw }, { w.cache.kg });
        op_query_copy = rw(b,
                           RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                  RAD_NOWEIGHTS),
                           { w.q_wide }, { w.cache.query });
        op_gate_copy = rw(b,
                          RAD_OP(b, "cast",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                            RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                 RAD_NOWEIGHTS),
                          { w.kg_raw }, { w.cache.kg });
        op_head = rw(b,
                     RAD_OP(b, "gemm_nt_f32",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 32),
                                       RAD_INT("K", 4096), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(head_weight)),
                     { w.input }, { w.cache.gate });
        RAD_ARCH_TRY(head_split.declare(b, nm, g, nm.f("blk.%d.index_head", l), g.max_tok, 32, 4096,
                                        head_weight, w.input, w.cache.gate));
        RAD_ARCH_TRY(cache.declare(b, nm, g, l, cfg, w.cache, attention, tail, index,
                                   nm.ckpt("%s.index_kpool_compress_ape", prefix)));
        return op_norm && op_query_copy && op_gate_copy && op_head ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        query.step(ctx, act(w.q_low), w.q_wide, T);
        RAD_ISSUE_N(ctx, op_query_copy, T * 32, brows(w.q_wide, T), brows(w.cache.query, T * 32));
        key_gate.step(ctx, act(w.input), w.kg_raw, T);
        RAD_ISSUE_N(ctx, op_norm, T, bcol(w.kg_raw, 0, 128, T), RAD_W(gain), RAD_W(bias),
                    bcol(w.cache.kg, 0, 128, T));
        RAD_ISSUE_N(ctx, op_gate_copy, T, bcol(w.kg_raw, 128, 128, T),
                    bcol(w.cache.kg, 128, 128, T));
        if (head_split.takes(T))
            head_split.step(ctx, T, brows(w.input, T), head_weight, brows(w.cache.gate, T));
        else
            RAD_ISSUE_N(ctx, op_head, T, brows(w.input, T), RAD_W(head_weight),
                        brows(w.cache.gate, T));
        cache.step(ctx, batch);
    }
};
}
} // namespace rad::arch
#endif
