/* attn_llama.h -- one plain multi-head attention block, bf16 operands, paged KV. PRIVATE to
 * arch/unlimitedocr_bf16 (the arch/qwen35_bf16 + siglip2.h precedent: a header beside the plugin
 * that composes it).
 *
 *   rmsnorm -> q gemm, k gemm, v gemm -> rope(q), rope(k)
 *   -> kv_store -> paged attention (causal) -> o gemm -> [all_reduce] -> residual add
 *
 * It is rad_block_attn_gated.h with the Qwen3.5-specific parts removed, and the removals are the
 * model: Unlimited-OCR's decoder is a plain Llama-MoE (modeling_deepseekv2.py's
 * SlidingWindowLlamaAttention is MHA with `use_mla: false`), so
 *
 *   * there is NO output gate: `q_proj` is [n_embd, n_head*head_dim] and holds only the query, so
 *     no de-interleaving, no `sigmoid`, no `mul`;
 *   * there is NO per-head q/k norm: the config's `qk_nope_head_dim`/`qk_rope_head_dim` are 0 and
 *     the reference rotates the projections straight;
 *   * the rotary is FULL head_dim (rot = head_dim = 128) and NeoX-half (`rotate_half` in the
 *     reference), not partial, not mrope;
 *   * q, k and v are three [1280, 1280] projections kept as three GEMMs and three contiguous
 *     operands, exactly as rad_block_attn_gated.h argues: a fused [k|v] row is not expressible as
 *     an operand a head-wise reader can walk.
 *
 * Everything the gated block gets right about paged KV is copied through unchanged: the attention
 * declared BEFORE the store so block_size derives from the resolved kernel, the ranged `q_len`
 * band, the ragged-step cu_seqlens, the collective emitted by the plugin and not the core.
 */
#ifndef UNLIMITEDOCR_ATTN_LLAMA_H
#define UNLIMITEDOCR_ATTN_LLAMA_H

#include <arch/rad_arch.h>

namespace rad {
namespace arch {

struct AttnLlamaBF16 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd] the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd] normed input, then the o output */
        rad_buf q    = 0;   /* [max_tok, q_dim]   rotated */
        rad_buf k    = 0;   /* [max_tok, kv_dim]  rotated */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf attn = 0;   /* [max_tok, q_dim] */
    };

    struct Src {
        const char* norm = nullptr;   /* input_layernorm.weight */
        const char* q    = nullptr;
        const char* k    = nullptr;
        const char* v    = nullptr;
        const char* o    = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;

    rad_weight w_norm = 0, w_q = 0, w_k = 0, w_v = 0, w_o = 0;
    rad_op op_norm = 0, op_q = 0, op_k = 0, op_v = 0;
    rad_op op_rope_q = 0, op_rope_k = 0;
    rad_op op_attn = 0, op_kv_store = 0, op_o = 0, op_ar = 0, op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnLlamaBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                  rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                  const Src& src) {
    g = geom; w = wire; layer = l; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }
    /* The KV-head-replication refusal rad_block_attn_gated.h states: this block declares its k/v
     * with the plain even row-shard and passes no row span, so a world above the KV head count
     * would hand a rank half of one KV head. Say so at declare. */
    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q.weight", l),    src.q));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),    src.k));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),    src.v));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l), src.o));

    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),    RAD_F32,  {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_q    = decl_w(b, nm.f("blk.%d.attn_q.weight", l),       RAD_BF16, {q_dim, g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_k    = decl_w(b, nm.f("blk.%d.attn_k.weight", l),       RAD_BF16, {kv_dim, g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_v    = decl_w(b, nm.f("blk.%d.attn_v.weight", l),       RAD_BF16, {kv_dim, g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_o    = decl_w(b, nm.f("blk.%d.attn_output.weight", l),  RAD_BF16, {g.n_embd, q_dim},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    op_q = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", q_dim),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_q)),
              {w.h}, {w.q});
    op_k = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_k)),
              {w.h}, {w.k});
    op_v = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_v)),
              {w.h}, {w.v});

    /* Full-head rotary, NeoX halves: the reference's rotate_half. The q call names n_head query
     * heads and no kv heads (the buffers are separate); the k call names the kv heads. */
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, "neox"),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, "neox"),
                   {w.k}, {w.k});

    /* Declared before kv_store so block_size derives from the resolved attention kernel (spec
     * §7.2), pinned to RAD_KV_BLOCK for the same reason rad_arch.h states. P1 serves the plain
     * causal rule; the R-SWA window is P4 and re-declares this op with its own rule. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn});

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    op_o = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                             RAD_INT("K", q_dim), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_o)),
              {w.attn}, {w.h});

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});

    return RAD_OK;
}

inline void AttnLlamaBF16::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    RAD_ISSUE(c, op_q,    brows(w.h, T), RAD_W(w_q),    brows(w.q, T));
    RAD_ISSUE(c, op_k,    brows(w.h, T), RAD_W(w_k),    brows(w.k, T));
    RAD_ISSUE(c, op_v,    brows(w.h, T), RAD_W(w_v),    brows(w.v, T));

    RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));

    RAD_ISSUE(c, op_kv_store,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    RAD_ISSUE(c, op_o, brows(w.attn, T), RAD_W(w_o), brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* UNLIMITEDOCR_ATTN_LLAMA_H */
