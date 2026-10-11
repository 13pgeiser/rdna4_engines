/* gemma4_dflash.h -- the DFlash (v1) block-diffusion drafter, as z-lab trains it for Gemma 4.
 *
 * ============================== WHAT IT IS ==============================
 *
 * rad_block_dflash2.h WITHOUT THE TWO THINGS DFLASH2 ADDED: the per-layer dynamic convolutions
 * around attention and the feed-forward, and the candidate selector that walks a path through the
 * block. What is left is the original DFlash: a small Qwen3-shaped transformer that denoises one
 * anchor token plus `block - 1` mask tokens in ONE forward and reads every position's argmax off
 * the TARGET's lm_head.
 *
 *     context pass, one row a committed token:
 *         c = hidden_norm(fc(concat of the trunk's residual stream at target_layer_ids)))
 *         for each layer:  k = k_norm(k_proj(c)) then rope;  v = v_proj(c);  kv_store(k, v)
 *
 *     query pass, `block` rows a sequence:
 *         x = embed(anchor, mask, ...) * sqrt(hidden)         <- Gemma's embedding scale, see below
 *         for each layer:
 *             h = input_layernorm(x)
 *             q, k, v = q_proj(h), k_proj(h), v_proj(h);  q_norm, k_norm;  rope;  kv_store
 *             x += o_proj(attention(q, cache))        sliding layers CAUSAL over a 2048 window,
 *                                                     the full layer NON-causal over everything
 *             x += down(silu(gate(h')) * up(h')),  h' = post_attention_layernorm(x)
 *         x = norm(x)
 *         over the mask rows: logits = lm_head(x) [the TARGET's];  draft = argmax
 *
 * ============================== THE THREE GEMMA-SPECIFIC FACTS ==============================
 *
 * They are vLLM PR #41703's, and each one costs acceptance and nothing else when it is wrong --
 * every draft is verified against the trunk, so the model's output never depends on any of them.
 *
 *   THE EMBEDDING IS SCALED. Gemma multiplies its token embedding by sqrt(hidden_size), stored as a
 *   bf16 (53.0 at 2816), and the drafter was trained reading the target's embedding module, which
 *   applies it. `embed_lookup`'s `mult` carries it.
 *
 *   THE SLIDING LAYERS ARE CAUSAL AND THE FULL LAYER IS NOT. The checkpoint's `layer_types` names
 *   four sliding layers and one full one; the PR's own test pins `causal True` on the sliding
 *   layers and `False` on the full one. So this block declares TWO attention ops and issues the
 *   one each layer's type names.
 *
 *   THE LOGITS ARE SOFT-CAPPED, and that is ignored on purpose: the draft is an argmax, and
 *   cap * tanh(x / cap) is strictly increasing, so it cannot move one.
 *
 * ============================== THE HEAD ==============================
 *
 * The target's own head, read in bf16 by an ordinary `gemm_nt` over the gathered mask rows. Gemma
 * ties it to the embedding, so it is the same weight the trunk's `embed_lookup` and `logits_gemm`
 * read: no extra bytes are stored. It is vocab-sharded with the trunk, so the top-1 runs over this
 * rank's slice and the ranks' candidates are merged exactly (`row_topk` at R = 2 -> `all_gather` ->
 * `pair_argmax`; R = 2 because the gather moves 16-byte rows, see kTopR).
 *
 * Reading the bf16 head every draft pass is the expensive way to do it -- 262144 x 2816 x 2 bytes
 * over the ranks. The cheaper way is the 2-bit copy plus an exact rerank that DFlash2 uses, and
 * that needs the trunk's head in fp8 or int8, which a tied Gemma head is not. Stated rather than
 * hidden: it is the first thing to change once the model runs.
 *
 * ============================== THE LINEARS ARE FP8 ==============================
 *
 * The same as DFlash2 and for the same reason -- the drafter is read whole on every decode step --
 * so the container must be converted with a recipe that makes the six linears block fp8 (or
 * MXFP4). declare() refuses a bf16 one by name rather than letting the loader fail on a view.
 */
#ifndef GEMMA4_DFLASH_H
#define GEMMA4_DFLASH_H

#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include <cstdio>
#include <cstring>

namespace gemma4 {

using namespace rad::arch;

struct Dflash1 {
    enum { MAX_LAYERS = 8 };

    struct Config {
        int64_t n_taps = 0;      /* how many trunk layers `fc` concatenates */
        int64_t block  = 0;      /* query rows a sequence: the anchor plus block - 1 masks */
        int64_t window = 0;      /* the sliding layers' window; 0 for none */
        int     full[MAX_LAYERS] = {};   /* 1: a full-attention (non-causal, unwindowed) layer */
        double  escale = 1.0;    /* the target's embedding scale, rounded as the target holds it */
        int64_t steps() const { return block - 1; }
    };

    struct Wire {
        /* the context pass */
        rad_buf src = 0, srcq = 0, srcs = 0;   /* [max_tok, n_taps*n_embd] bf16 | e4m3 | f32 */
        rad_buf c = 0, cq = 0, cs = 0;         /* [max_tok, n_embd] */
        rad_buf ck = 0, cv = 0;                /* [max_tok, kv_dim] -- see rad_block_dflash2.h */
        /* the query pass */
        rad_buf x = 0, h = 0, hq = 0, hs = 0;  /* [qrows, n_embd] */
        rad_buf q = 0, k = 0, v = 0;           /* [qrows, q_dim | kv_dim | kv_dim] */
        rad_buf attn = 0, atq = 0, ats = 0;    /* [qrows, q_dim] */
        rad_buf gate_up = 0;                   /* [qrows, 2 * n_ff] */
        rad_buf ffn = 0, ffq = 0, ffs = 0;     /* [qrows, n_ff] */
        /* the head, one row a draft position */
        rad_buf sel_h = 0;                     /* [srows, n_embd] */
        rad_buf dlog = 0;                      /* [srows, n_vocab on this rank] bf16 */
        rad_buf cand = 0, unary = 0;           /* [srows, kTopR] i32 | f32 -- this rank's best */
        rad_buf dpair = 0, dgath = 0;          /* [srows, kTopR, 2] | [srows, world * kTopR, 2] f32 */
        rad_buf tokens = 0;                    /* [srows, 1] i32: THE DRAFT, pitch `steps` */
    };

    Geom    g{};          /* the drafter's geometry */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;
    int64_t qrows = 0, crows = 0, srows = 0, gated_rows = 0;
    rad_kvgroup kv = 0;
    bool    sharded = false;

    DenseFP8   w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_tok = 0;                     /* the TARGET's table: embedding AND head */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    DenseFP8   w_q[MAX_LAYERS]{}, w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};

    rad_op op_qsrc = 0, op_cfc = 0, op_cnorm = 0, op_qc = 0, op_crope = 0, op_cstore = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    rad_op op_embed = 0, op_embed_ar = 0, op_qh = 0, op_qat = 0, op_qff = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {};
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {}, op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_gug[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn_swa = 0, op_attn_full = 0;
    rad_op op_act = 0, op_ar = 0, op_add = 0, op_final = 0;
    rad_op op_gather = 0, op_head = 0, op_topk = 0, op_dgather = 0, op_pick = 0;

    /* CANDIDATES A RANK CONTRIBUTES under a sharded head. The draft needs one, but the gather moves
     * whole 16-byte words a row and an (int32 id, f32 value) pair is 8 bytes, so R = 1 is refused at
     * launch ("shape this kernel does not serve") -- the bound rad_block_mtp_hc.h states. Two pairs
     * are one word; `pair_argmax` reduces the world * 2 of them to the global top-1. */
    static constexpr int64_t kTopR = 2;

    static bool head_sharded(const Geom& g) { return vocab_head_sharded(g); }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck, const Wire& wire);
    void context(RadCtx* c, const RadBatch* batch) const;
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
    /* The six linears must be what the recipe made them. DenseFP8 declares E4M3 (or MXFP4) views
     * whatever the container holds, and the loader's complaint about a bf16 weight read through an
     * fp8 view names the view, not the missing recipe. */
    static int need_quantised(RadBuilder* b, const char* wn) {
        RadEncoding e{};
        if (!weight_enc(b, wn, &e) || e.n_planes <= 0) return RAD_OK;   /* nothing to ask yet */
        const uint32_t dt = e.plane[0].dtype;
        if (dt == RAD_F8E4M3 || dt == RAD_FP4E2M1) return RAD_OK;
        fprintf(stderr, "radiance: gemma4: the DFlash drafter's '%s' is stored %s. Its six linears "
                        "are served block fp8 (or MXFP4): convert with a recipe that names them, "
                        "e.g. `dflash.blk.*.attn_q.weight rtn codes=fp8_e4m3 block=128x128 "
                        "scale=bf16` and the same for fc, attn_k/v/output, ffn_gate_up and "
                        "ffn_down.\n", wn, rad_dtype_name(dt));
        return RAD_E_DTYPE;
    }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dflash1::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                            int base_layer, rad_kvgroup group, rad_weight tok_embd,
                            const char* ck, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: gemma4: the drafter has %lld layers; this block carries at most "
                        "%d\n", (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0) return RAD_E_INVAL;

    const int64_t qd = q_dim(), kvd = kv_dim();
    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;
    sharded = head_sharded(g);

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dflash.fc"), g.n_embd, cfg.n_taps * g.n_embd,
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(),
                              { nm.ckpt("%s.fc.weight", ck) }));
    RAD_ARCH_TRY(need_quantised(b, nm.f("dflash.fc.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.norm.weight"),        nm.ckpt("%s.norm.weight", ck)));
    w_hnorm = decl_w(b, nm.f("dflash.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dflash.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup gl = grp_layer(first_layer + i);
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));
        w_anorm[i] = decl_w(b, nm.f("dflash.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_fnorm[i] = decl_w(b, nm.f("dflash.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_qn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_kn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        /* The declared names are rad_block_dflash2.h's, deliberately: the recipe that quantises a
         * DFlash2 drafter's linears names exactly these, so one recipe serves both drafters. */
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dflash.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dflash.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dflash.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dflash.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_gate_up", i), 2 * g.n_ff,
                                     g.n_embd, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, gl,
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
        for (const char* s : { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up",
                               "ffn_down" })
            RAD_ARCH_TRY(need_quantised(b, nm.f("dflash.blk.%d.%s.weight", i, s)));
    }

    /* ---------------------------------------------------------------- attention: two kinds
     * Declared before either kv_store: `block_size` is read back off the resolved kernel. Both
     * read the one KV group, so whichever resolves first answers for it. */
    auto attn = [&](int causal, int64_t window) {
        return rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", causal),
                                RAD_INT("window", window), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                  {w.q}, {w.attn});
    };
    bool any_full = false, any_swa = false;
    for (int64_t l = 0; l < g.n_layer; ++l) (cfg.full[l] ? any_full : any_swa) = true;
    if (any_swa)  op_attn_swa  = attn(1, cfg.window);
    if (any_full) op_attn_full = attn(0, 0);

    /* ---------------------------------------------------------------- the context pass */
    op_qsrc = rw(b, RAD_OP(b, "quant_act_fp8",
                    RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.src}, {w.srcq, w.srcs});
    op_cfc = w_fc.gemm(b, crows, g.n_embd, cfg.n_taps * g.n_embd, {w.srcq, w.srcs}, {w.c});
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    op_qc = rw(b, RAD_OP(b, "quant_act_fp8",
                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.c}, {w.cq, w.cs});
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_ck[i] = w_k[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.ck});
        op_cv[i] = w_v[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.cv});
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    op_crope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off), RAD_F64("mult", cfg.escale)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.x});
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});
    op_qh = rw(b, RAD_OP(b, "quant_act_fp8",
                  RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.h}, {w.hq, w.hs});
    op_qat = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.attn}, {w.atq, w.ats});
    op_qff = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.ffn}, {w.ffq, w.ffs});

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_anorm[i])),
                        {w.x}, {w.h});
        op_qp[i] = w_q[i].gemm(b, qrows, qd, g.n_embd, {w.hq, w.hs}, {w.q});
        op_kp[i] = w_k[i].gemm(b, qrows, kvd, g.n_embd, {w.hq, w.hs}, {w.k});
        op_vp[i] = w_v[i].gemm(b, qrows, kvd, g.n_embd, {w.hq, w.hs}, {w.v});
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv),
                                    RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = w_o[i].gemm(b, qrows, g.n_embd, qd, {w.atq, w.ats}, {w.h});
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fnorm[i])),
                        {w.x}, {w.h});
        op_gu[i] = w_gu[i].gemm(b, qrows, 2 * g.n_ff, g.n_embd, {w.hq, w.hs}, {w.gate_up});
        /* The gate_up|silu|quant fold where it resolves, the pair where it does not --
         * rad_block_dflash2.h's rule, band included. */
        {
            const int64_t hi = qrows < 64 ? qrows : 64;
            if (hi > 0 && !w_gu[i].mx) {
                op_gug[i] = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * g.n_ff),
                                              RAD_INT("K", g.n_embd),
                                              RAD_INT("group", RAD_FP8_BLOCK),
                                              RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                               {w.hq, w.hs}, {w.ffq, w.ffs});
                if (op_gug[i]) gated_rows = hi;
            }
        }
        op_dn[i] = w_dn[i].gemm(b, qrows, g.n_embd, g.n_ff, {w.ffq, w.ffs}, {w.h});
    }
    op_qrope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.x, w.h}, {w.x});
    op_final = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* ---------------------------------------------------------------- the head
     * The MASK rows -- offsets 1..block-1 of each block; the core names them through out_ids --
     * gathered contiguous, projected onto THIS RANK's slice of the target's vocabulary, and the
     * argmax taken. Sharded, each rank's best kTopR are reported as global ids with their logits,
     * all-gathered, and reduced to the global top-1 by `pair_argmax` (gemma4k), ties to the lower
     * id -- exact, since the global winner is its own shard's first. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.sel_h});
    op_head = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                               RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_tok)),
                {w.sel_h}, {w.dlog});
    if (sharded) {
        op_topk = rw(b, RAD_OP(b, "row_topk",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                                   RAD_INT("R", kTopR), RAD_INT("vocab_off", g.vocab_off),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.dlog}, {w.cand, w.unary, w.dpair});
        /* ROW-INTERLEAVED, so one draft position's pairs from every rank land side by side. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * kTopR * 2), RAD_STR("dtype", "f32"),
                                      RAD_INT("row", kTopR * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_pick = rw(b, RAD_OP(b, "pair_argmax",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.world * kTopR),
                                   RAD_STR("dtype", "f32")),
                        RAD_NOWEIGHTS),
                    {w.dgath}, {w.tokens});
        if (!op_pick) {
            fprintf(stderr, "radiance: gemma4: the sharded draft head needs `pair_argmax`, which the "
                            "gemma4k kernel library serves -- load it (--kernels gemma4k,libr4d,...)\n");
            return RAD_E_UNSUPPORTED;
        }
    } else {
        op_topk = rw(b, RAD_OP(b, "row_topk",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                                   RAD_INT("R", 1), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.dlog}, {w.tokens, w.unary});
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dflash1::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    RAD_ISSUE_N(c, op_qsrc, T, brows(w.src, T), brows(w.srcq, T), brows(w.srcs, T));
    RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                brows(w.c, T), RAD_NONE, w_fc.bref());
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, w_v[i].bref());
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv, brows(w.ck, T), RAD_W(w_kn[i]),
                    brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T), praw(slot, RAD_I32, T),
                    kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dflash1::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const rad_op op_attn = cfg.full[i] ? op_attn_full : op_attn_swa;

        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_qh, T, brows(w.h, T), brows(w.hq, T), brows(w.hs, T));
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.hq, T), brows(w.hs, T),
                    RAD_W(w_q[i].w), RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, w_q[i].bref());
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.hq, T), brows(w.hs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.hq, T), brows(w.hs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, w_v[i].bref());
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T),
                    kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(w.attn, T));
        RAD_ISSUE_N(c, op_qat, T, brows(w.attn, T), brows(w.atq, T), brows(w.ats, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.atq, T), brows(w.ats, T),
                    RAD_W(w_o[i].w), RAD_W(w_o[i].s), brows(w.h, T), RAD_NONE, w_o[i].bref());
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h, T), brows(w.x, T));

        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_qh, T, brows(w.h, T), brows(w.hq, T), brows(w.hs, T));
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.hq, T), brows(w.hs, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.hq, T), brows(w.hs, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE,
                        w_gu[i].bref());
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ffn, T), brows(w.ffq, T), brows(w.ffs, T));
        }
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffq, T), brows(w.ffs, T),
                    RAD_W(w_dn[i].w), RAD_W(w_dn[i].s), brows(w.h, T), RAD_NONE, w_dn[i].bref());
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h, T), brows(w.x, T));
    }
    RAD_ISSUE_N(c, op_final, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h), praw(batch->out_ids, RAD_I32, S), brows(w.sel_h, S));
    RAD_ISSUE_N(c, op_head, S, brows(w.sel_h, S), RAD_W(w_tok), brows(w.dlog, S));
    if (sharded) {
        RAD_ISSUE_N(c, op_topk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_pick, S, brows(w.dgath, S), brows(w.tokens, S), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_topk, S, brows(w.dlog, S), brows(w.tokens, S), brows(w.unary, S),
                    RAD_NONE);
    }
}

}  /* namespace gemma4 */

#endif /* GEMMA4_DFLASH_H */
