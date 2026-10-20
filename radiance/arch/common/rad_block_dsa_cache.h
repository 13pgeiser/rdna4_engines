/* GLM's stateful compressed-index core. The caller supplies projected queries,
 * signed F32 head gates, and [LayerNorm(key), pool_gate] BF16 rows. No RoPE.
 * Query projections are replicated across TP ranks; selected pages belong to MLA. */
#ifndef RAD_BLOCK_DSA_CACHE_H
#define RAD_BLOCK_DSA_CACHE_H
#include "rad_arch.h"

namespace rad {
namespace arch {
struct DsaCache {
    struct Config {
        int64_t blocks = 0, work = 0, ring = 0, topk = 512;
    };
    struct Wire {
        rad_buf query = 0, codes = 0, scales = 0, gate = 0, kg = 0;
        rad_buf stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, pool_scale = 0;
        rad_buf score = 0, selected = 0, nsel = 0, length = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup index = 0, tail = 0, attention = 0;
    rad_weight ape = 0;
    rad_op op_query = 0, op_work = 0, op_pool = 0, op_tail = 0, op_score = 0, op_select = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, rad_kvgroup attn, rad_kvgroup tail_group,
                rad_kvgroup* index_group, const char* ape_source) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        attention = attn;
        tail = tail_group;
        if (!b || !index_group || !ape_source || !attention || !tail || c.blocks < 1 ||
            c.work < (g.max_tok + 3) / 4 + g.max_seqs || c.ring < 4 + g.max_spec || c.topk != 512 ||
            !w.query || !w.codes || !w.scales || !w.gate || !w.kg || !w.stage || !w.page ||
            !w.bpos || !w.nc || !w.pooled || !w.pool_scale || !w.score || !w.selected || !w.nsel ||
            !w.length)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, tail));
        const char* name = nm.f("blk.%d.index_ape", l);
        RAD_ARCH_TRY(map_copy(b, name, ape_source));
        ape = decl_w(b, name, RAD_F32, { 4, 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
        if (!ape) return RAD_E_INVAL;
        // Query buffers are [tokens*32,128]; scoring uses their rank-three view.
        op_query = rw(b,
                      RAD_OP(b, "dsa_query",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                      { w.query }, { w.codes, w.scales });
        op_work = rw(
            b,
            RAD_OP(b, "qsa_work",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256), RAD_INT("ratio", 4),
                              RAD_INT("ring", c.ring), RAD_INT("seqs", g.max_seqs),
                              RAD_INT("work", c.work), RAD_STR("dtype", "bf16")),
                   RAD_NOWEIGHTS),
            { w.kg }, { w.stage, w.page, w.bpos, w.nc });
        // Declare immediately before the cache consumer: page size comes from that op.
        if (!*index_group) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_FULL;
            d.dtype = RAD_U8;
            d.n_head_kv = 1;
            d.head_dim = 17;
            *index_group = rad_decl_kv_group(b, nm.f("kv_dsa_index"), &d);
        }
        index = *index_group;
        if (!index) return RAD_E_INVAL;
        op_pool =
            rw(b,
               RAD_OP(b, "dsa_pool",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.work), RAD_INT("n", 128), RAD_INT("ratio", 4),
                                 RAD_INT("block_size", 4), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(ape)),
               { w.stage, w.page }, { w.pooled, w.pool_scale });
        if (!op_pool || rad_kv_block_size(b, index) != 4) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, index));
        op_tail = rw(b,
                     RAD_OP(b, "qsa_tail_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256),
                                       RAD_INT("ratio", 4), RAD_INT("ring", c.ring),
                                       RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.kg }, {});
        op_score = rw(b,
                      RAD_OP(b, "dsa_score",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                        RAD_INT("heads", 32), RAD_INT("blocks", c.blocks),
                                        RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "fp8_e4m3")),
                             RAD_NOWEIGHTS),
                      { w.codes, w.scales, w.gate, w.nc }, { w.score });
        op_select = rw(b,
                       RAD_OP(b, "qsa_select",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("blocks", c.blocks),
                                         RAD_INT("topk", 512), RAD_INT("ratio", 4),
                                         RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1)),
                              RAD_NOWEIGHTS),
                       { w.score, w.nc }, { w.selected, w.nsel, w.length });
        return op_query && op_work && op_tail && op_score && op_select ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const auto* ib = kv_batch(batch, index);
        const auto* tb = kv_batch(batch, tail);
        const auto* ab = kv_batch(batch, attention);
        if (!ib || !tb || !ab || !ib->block_table || !ab->block_table || !tb->state_index) {
            rad_step_fail(ctx, "DSA requires index/attention block tables and tail state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq;
        const int64_t work = (T + 3) / 4 + S,
                      pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
        RAD_ISSUE_N(ctx, op_query, T * 32, brows(w.query, T * 32), brows(w.codes, T * 32),
                    brows(w.scales, T * 32));
        RAD_ISSUE_N(ctx, op_work, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch),
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), RAD_NONE,
                    brows(w.stage, work * 4), brows(w.page, work), brows(w.bpos, work),
                    brows(w.nc, S));
        RAD_ISSUE_N(ctx, op_pool, work, brows(w.stage, work * 4), RAD_W(ape), brows(w.pooled, work),
                    brows(w.pool_scale, work), brows(w.page, work), kv_cache(index, layer));
        // Gather before overwriting the tail; the wider ring preserves rejected-token rollback.
        RAD_ISSUE_N(ctx, op_tail, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch));
        int64_t live = ((int64_t)batch->max_ctx_len + T + batch->n_spec + 4) / 4 + 1;
        if (live > c.blocks) live = c.blocks;
        // T is the total across sequences, so the conservative bound can exceed the
        // per-sequence page table. Columns without a table entry cannot be scored.
        if (live > ib->block_table_pitch) live = ib->block_table_pitch;
        if (live < 1) live = 1;
        RAD_ISSUE_N(ctx, op_score, T, brows(w.codes, T * 32), brows(w.scales, T * 32),
                    brows(w.gate, T), kv_cache(index, layer), RAD_NONE,
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, 0, c.blocks, 0, live, T));
        RAD_ISSUE_N(ctx, op_select, T, bcol_at(w.score, 0, c.blocks, 0, live, T),
                    praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch), brows(w.nc, S),
                    praw(batch->positions, RAD_I32, T), praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.selected, T), brows(w.nsel, T), brows(w.length, T));
    }
};
}
} // namespace rad::arch
#endif
