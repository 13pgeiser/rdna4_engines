/* GLM-5.3-Flash: four mHC streams, KDA/latent MLA, mixed Quark weights,
 * width-sharded routed experts and a native serial next-token decoder.
 * This plugin uses only the public ABI and the installed common headers. */
#include <arch/rad_block_mhc.h>
#include <arch/rad_block_glm_kda.h>
#include <arch/rad_block_glm_mla.h>
#include <arch/rad_block_glm_indexer.h>
#include <arch/rad_block_mlp_fp8.h>
#include <arch/rad_block_moe_fp8.h>
#include <memory>

namespace glm5next {
using namespace rad::arch;
struct Layer {
    bool full = false, draft = false;
    ManifoldConn attn_hc{}, ffn_hc{};
    NormQuantFP8 norm{};
    GlmKda kda{};
    GlmMla mla{};
    GlmIndexer index{};
    MlpFP8 dense{};
    MoeFP8 moe{};
    rad_op residual = 0;
};
struct Model {
    Names nm{}, dnm{ "mtp" };
    Geom g{};
    std::vector<Layer> layers;
    Layer draft{};
    rad_kvgroup latent = 0, index = 0, tail = 0, state = 0, conv = 0;
    rad_buf h0 = 0, h1 = 0, x = 0, y = 0, post = 0, comb = 0, final_hidden = 0, logit_input = 0,
            logits = 0;
    ActFP8 act{}, ff{};
    GlmKda::Wire kw{};
    GlmMla::Wire aw{};
    GlmIndexer::Wire iw{};
    MlpFP8::Wire dw{};
    MoeFP8::Wire mw{};
    rad_weight token = 0, head = 0, final_gain = 0;
    rad_op embed = 0, embed_ar = 0, expand = 0, contract = 0, norm = 0, gather = 0, head_op = 0;
    rad_buf mtp_x = 0, mtp_h = 0, mtp_e = 0, mtp_concat = 0, mtp_final = 0;
    rad_weight en = 0, hn = 0, eh = 0, mnorm = 0;
    rad_op mprev = 0, membed = 0, membed_ar = 0, men = 0, mhn = 0, meh = 0, mnorm_op = 0,
           mselect = 0, mback = 0, mcopy = 0, mhead = 0;
    bool have_mtp = false;
    bool fuse_router = false, fuse_swiglu = false, gather_inverse = false;
};
static std::unique_ptr<Model> models[MAX_RANKS];
static thread_local std::unique_ptr<Model> probe_model;

static int64_t cfg(const RadModelMeta* m, const char* key, int64_t fallback) {
    char nested[160];
    std::snprintf(nested, sizeof nested, "text_config.%s", key);
    return rad_meta_geti(m, key, rad_meta_geti(m, nested, fallback));
}
static rad_op reduce(RadBuilder* b, const Geom& g, rad_buf x, bool exact = false) {
    if (g.world == 1) return 0;
    return rw(b,
              RAD_OP(b, "all_reduce",
                     RAD_PARAMS(
                         RAD_INT("world_size", g.world), RAD_RANGE("numel", 4096, g.max_tok * 4096),
                         RAD_STR("dtype", "bf16"), RAD_INT("exact", exact ? 1 : g.wire_exact),
                         RAD_INT("min_bytes", g.wire_min_bytes)),
                     RAD_NOWEIGHTS),
              { x }, { x });
}
static rad_op copy(RadBuilder* b, const Geom& g, rad_buf x, rad_buf y, int64_t n = 4096) {
    return rw(b,
              RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                                RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                     RAD_NOWEIGHTS),
              { x }, { y });
}
static rad_op norm(RadBuilder* b, const Geom& g, rad_weight w, rad_buf x, rad_buf y) {
    return rw(b,
              RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                     RAD_WEIGHTS(w)),
              { x }, { y });
}
static int buffers(RadBuilder* b, Model& m) {
    auto& g = m.g;
    auto& nm = m.nm;
    const int64_t T = g.max_tok, W = 8192 / g.world, F = 2048 / g.world;
    auto bf = [&](const char* n, int64_t width) {
        return decl_b(b, nm.f("%s", n), RAD_BF16, { T, width });
    };
    auto a = [&](ActFP8& v, const char* n, int64_t width) {
        v.x = bf(n, width);
        return v.declare_qs(b, nm, g, n, width);
    };
    m.h0 = bf("manifold", 16384);
    m.h1 = bf("manifold_mid", 16384);
    m.x = bf("collapsed", 4096);
    m.y = bf("attn_delta", 4096);
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, 4 });
    m.comb = decl_b(b, nm.f("mix"), RAD_F32, { T, 16 });
    RAD_ARCH_TRY(a(m.act, "normalized", 4096));
    m.final_hidden = decl_b(b, nm.f("final_hidden"), RAD_BF16, { T, 4096 }, RAD_BUF_PERSIST);
    m.logit_input = decl_b(b, nm.f("logit_input"), RAD_BF16, { g.max_logit_rows, 4096 });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));
    m.kw.input = m.act.x;
    m.kw.out = m.y;
    m.kw.forget_low = bf("forget_low", 128);
    m.kw.gate_low = bf("gate_low", 128);
    m.kw.core.qkv = bf("qkv", 3 * W);
    m.kw.core.conv = bf("convolved", 3 * W);
    m.kw.core.a = bf("forget", W);
    m.kw.core.beta = bf("beta", 64 / g.world);
    m.kw.core.gate = bf("gate", W);
    m.kw.core.out = bf("recurrent_out", W);
    m.aw.input = m.act;
    m.aw.q_raw = bf("q_raw", 1536);
    m.aw.kv_raw = bf("kv_raw", 512);
    RAD_ARCH_TRY(a(m.aw.q_low, "q_low", 1536));
    m.aw.latent = bf("latent", 512);
    m.aw.out = m.y;
    m.aw.core.q = bf("query", 16384 / g.world);
    m.aw.core.q_latent = decl_b(b, nm.f("q_latent"), RAD_BF16, { T, 64 / g.world, 512 });
    m.aw.core.attn_latent = decl_b(b, nm.f("attn_latent"), RAD_BF16, { T, 64 / g.world, 512 });
    RAD_ARCH_TRY(a(m.aw.attn, "expanded", 16384 / g.world));
    m.aw.core.out = m.aw.attn.x;
    m.aw.core.selected = decl_b(b, nm.f("selected"), RAD_I32, { T, 513 });
    m.aw.core.length = decl_b(b, nm.f("length"), RAD_I32, { T });
    const int64_t work = (T + 3) / 4 + g.max_seqs, blocks = (g.max_ctx + 3) / 4 + 1;
    m.iw.input = m.act.x;
    m.iw.q_low = m.aw.q_low.x;
    m.iw.q_wide = bf("index_query", 4096);
    m.iw.kg_raw = bf("index_key_gate", 256);
    auto& iw = m.iw.cache;
    iw.query = decl_b(b, nm.f("index_heads"), RAD_BF16, { T * 32, 128 });
    iw.codes = decl_b(b, nm.f("index_codes"), RAD_F8E4M3, { T * 32, 128 });
    iw.scales = decl_b(b, nm.f("index_scales"), RAD_F32, { T * 32, 1 });
    iw.gate = decl_b(b, nm.f("index_head_weights"), RAD_F32, { T, 32 });
    iw.kg = bf("index_normalized_key_gate", 256);
    iw.stage = decl_b(b, nm.f("pool_stage"), RAD_BF16, { work * 4, 256 });
    iw.page = decl_b(b, nm.f("pool_page"), RAD_I32, { work });
    iw.bpos = decl_b(b, nm.f("pool_pos"), RAD_I32, { work });
    iw.nc = decl_b(b, nm.f("pool_count"), RAD_I32, { g.max_seqs });
    iw.pooled = decl_b(b, nm.f("pool_codes"), RAD_F8E4M3, { work, 128 });
    iw.pool_scale = decl_b(b, nm.f("pool_scales"), RAD_F32, { work, 1 });
    iw.score = decl_b(b, nm.f("index_scores"), RAD_F32, { T, blocks });
    iw.selected = m.aw.core.selected;
    iw.length = m.aw.core.length;
    iw.nsel = decl_b(b, nm.f("selected_count"), RAD_I32, { T });
    m.dw.x = m.x;
    m.dw.h = m.act;
    m.dw.gate_up = bf("dense_gate_up", 2 * g.n_ff);
    RAD_ARCH_TRY(a(m.dw.ffn, "dense_ff", g.n_ff));
    const int64_t R = T;
    m.mw.x = m.x;
    m.mw.h = m.act;
    m.mw.logits = decl_b(b, nm.f("router_logits"), RAD_F32, { T, 288 });
    m.mw.ids = decl_b(b, nm.f("expert_ids"), RAD_I32, { T, 8 });
    m.mw.ew = decl_b(b, nm.f("expert_weights"), RAD_F32, { T, 8 });
    m.mw.eoff = decl_b(b, nm.f("expert_offsets"), RAD_I32, { 289 });
    m.mw.ecnt = decl_b(b, nm.f("expert_counts"), RAD_I32, { 288 });
    m.mw.sorted = decl_b(b, nm.f("expert_sorted"), RAD_I32, { R * 8 });
    if (m.gather_inverse) m.mw.inv = decl_b(b, nm.f("expert_inverse"), RAD_I32, { R * 8 });
    m.mw.egu = decl_b(b, nm.f("expert_gate_up"), RAD_BF16, { R * 8, 2 * F });
    m.mw.eff.x = decl_b(b, nm.f("expert_ff"), RAD_BF16, { R * 8, F });
    RAD_ARCH_TRY(m.mw.eff.declare_qs(b, nm, g, "expert_ff", F, R * 8));
    m.mw.edn = decl_b(b, nm.f("expert_down"), RAD_BF16, { R * 8, 4096 });
    m.mw.sgu = bf("shared_gate_up", 2 * F);
    RAD_ARCH_TRY(a(m.mw.sff, "shared_ff", F));
    m.mw.sout = bf("shared_out", 4096);
    m.mw.sgate = bf("shared_one", 1);
    return RAD_OK;
}
static int layer(RadBuilder* b, Model& m, Layer& lay, Names& nm, int l, bool draft) {
    const auto& g = m.g;
    lay.draft = draft;
    lay.full = draft || l % 4 == 3;
    const char* pre = nm.ckpt("model.language_model.layers.%d", l);
    const char* ap = nm.ckpt("%s.self_attn", pre);
    rad_buf input = draft ? m.mtp_x : m.x;
    if (!draft) {
        ManifoldConn::Config hc;
        hc.eps = g.eps;
        ManifoldConn::Wire a{ m.h0, m.x, m.y, m.h1, m.post, m.comb };
        ManifoldConn::Wire f{ m.h1, m.x, m.act.x, m.h0, m.post, m.comb };
        ManifoldConn::Src as{ nm.ckpt("%s.hc_attn_fn", pre), nm.ckpt("%s.hc_attn_scale", pre),
                              nm.ckpt("%s.hc_attn_base", pre) };
        ManifoldConn::Src fs{ nm.ckpt("%s.hc_ffn_fn", pre), nm.ckpt("%s.hc_ffn_scale", pre),
                              nm.ckpt("%s.hc_ffn_base", pre) };
        RAD_ARCH_TRY(lay.attn_hc.declare(b, nm, g, l, nm.f("blk.%d.hc_attn", l), hc, a, as, true));
        // Declare ffn mHC after the attention below: declaration order is liveness.
        (void)f;
        (void)fs;
    }
    const char* nn = nm.f("blk.%d.attn_norm.weight", l);
    RAD_ARCH_TRY(map_copy(b, nn, nm.ckpt("%s.input_layernorm.weight", pre)));
    const auto gain =
        decl_w(b, nn, RAD_F32, { 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    RAD_ARCH_TRY(lay.norm.declare(b, g, gain, input, m.act, 4096));
    if (lay.full) {
        RAD_ARCH_TRY(lay.mla.declare(b, nm, g, l, m.aw, ap, m.latent, true, true));
        DsaCache::Config c;
        c.blocks = (g.max_ctx + 3) / 4 + 1;
        c.work = (g.max_tok + 3) / 4 + g.max_seqs;
        c.ring = 4 + g.max_spec;
        RAD_ARCH_TRY(lay.index.declare(b, nm, g, l, m.iw, nm.ckpt("%s.indexer", ap), c, m.latent,
                                       m.tail, &m.index));
        RAD_ARCH_TRY(lay.mla.declare_finish(b, nm, ap));
    } else {
        KdaState::Src s{ nm.ckpt("%s.q_conv1d.weight", ap), nm.ckpt("%s.k_conv1d.weight", ap),
                         nm.ckpt("%s.v_conv1d.weight", ap), nm.ckpt("%s.A_log", ap),
                         nm.ckpt("%s.dt_bias", ap),         nm.ckpt("%s.o_norm.weight", ap) };
        RAD_ARCH_TRY(lay.kda.declare(b, nm, g, l, m.kw, ap, s, m.state, m.conv));
    }
    if (!draft) {
        RAD_ARCH_TRY(lay.attn_hc.declare_post(b));
        ManifoldConn::Config hc;
        hc.eps = g.eps;
        ManifoldConn::Wire f{ m.h1, m.x, m.act.x, m.h0, m.post, m.comb };
        ManifoldConn::Src fs{ nm.ckpt("%s.hc_ffn_fn", pre), nm.ckpt("%s.hc_ffn_scale", pre),
                              nm.ckpt("%s.hc_ffn_base", pre) };
        RAD_ARCH_TRY(lay.ffn_hc.declare(b, nm, g, l, nm.f("blk.%d.hc_ffn", l), hc, f, fs, true));
    } else {
        lay.residual = rw(b,
                          RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                          { m.mtp_x, m.y }, { m.mtp_x });
        // All history rows fill the draft cache; only proposal rows need its FFN.
        m.mselect = rw(b,
                       RAD_OP(b, "gather_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", 4096),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                       { m.mtp_x }, { m.mtp_h });
        m.mback = copy(b, g, m.mtp_h, m.mtp_x);
    }
    if (l < 3) {
        auto w = m.dw;
        w.x = input;
        MlpFP8::Src s{ nm.ckpt("%s.post_attention_layernorm.weight", pre),
                       nm.ckpt("%s.mlp.gate_proj.weight", pre),
                       nm.ckpt("%s.mlp.up_proj.weight", pre),
                       nm.ckpt("%s.mlp.down_proj.weight", pre) };
        lay.dense.swiglu_quant = m.fuse_swiglu;
        RAD_ARCH_TRY(lay.dense.declare(b, nm, g, l, w, s, false, draft, kArNone, false, 10));
    } else {
        auto w = m.mw;
        w.x = input;
        MoeFP8::Src s;
        s.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
        s.router = nm.ckpt("%s.mlp.gate.weight", pre);
        s.router_bias = nm.ckpt("%s.mlp.gate.e_score_correction_bias", pre);
        s.experts = nm.ckpt("%s.mlp.experts", pre);
        s.shared = nm.ckpt("%s.mlp.shared_experts", pre);
        s.cfg.n_expert = s.cfg.n_expert_all = 288;
        s.cfg.top_k = 8;
        s.cfg.n_ff_exp = s.cfg.n_ff_shared = 2048 / g.world;
        s.cfg.rows = g.max_tok;
        s.cfg.sigmoid_router = 1;
        s.cfg.route_scale = 2.5;
        s.cfg.swiglu_limit = 10;
        s.cfg.shared_ungated = 1;
        s.cfg.router_fuse = m.fuse_router;
        s.cfg.swiglu_quant = m.fuse_swiglu;
        for (int q = 0; q < 2; ++q) {
            s.cfg.ff_lo[q] = g.rank * (2048 / g.world);
            s.cfg.ff_hi[q] = (g.rank + 1) * (2048 / g.world);
        }
        RAD_ARCH_TRY(MoeFP8::probe(b, nm, l, s, &s.cfg));
        RAD_ARCH_TRY(lay.moe.declare(b, nm, g, l, w, s, false, draft));
    }
    if (!draft) RAD_ARCH_TRY(lay.ffn_hc.declare_post(b));
    return RAD_OK;
}
/* A decode fusion of the MoE block, on unless its variable is 0. Each is byte-identical to the
 * launches it replaces, so the switch is about speed alone; 0 restores the separate launches. */
static int fusion_switch(const char* name, bool* out) {
    const char* e = std::getenv(name);
    *out = true;
    if (!e || !*e || !std::strcmp(e, "1")) return RAD_OK;
    if (!std::strcmp(e, "0")) {
        *out = false;
        return RAD_OK;
    }
    fprintf(stderr, "radiance: %s='%s'; it is 0 or 1\n", name, e);
    return RAD_E_INVAL;
}
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->is_draft || ctx->rank < 0 || ctx->rank >= MAX_RANKS)
        return RAD_E_INVAL;
    const int world = ctx->world_size ? ctx->world_size : 1;
    if ((world != 1 && world != 2 && world != 4 && world != 8) || meta->n_embd != 4096 ||
        meta->n_layers != 45 || meta->n_vocab != 154880 ||
        cfg(meta, "num_nextn_predict_layers", 0) != 1 || cfg(meta, "hc_mult", 0) != 4 ||
        cfg(meta, "n_routed_experts", 0) != 288 || cfg(meta, "index_topk", 0) != 2048 ||
        ctx->max_spec > 31 || (ctx->kv_dtype && ctx->kv_dtype != RAD_BF16))
        return RAD_E_UNSUPPORTED;
    auto& slot = ctx->shape_probe ? probe_model : models[ctx->rank];
    slot = std::make_unique<Model>();
    auto& m = *slot;
    auto& g = m.g;
    m.nm = Names(ctx->scope);
    auto& nm = m.nm;
    g.n_layer = 45;
    g.n_embd = 4096;
    g.head_dim = 256;
    g.n_head_all = g.n_head_kv_all = 64;
    g.n_head = g.n_head_kv = 64 / world;
    g.n_ff_all = 12288;
    g.n_ff = 12288 / world;
    g.n_vocab_all = 154880;
    g.n_vocab = shard_extent(154880, world, ctx->rank);
    g.vocab_off = shard_begin(154880, world, ctx->rank);
    g.rank = ctx->rank;
    g.world = world;
    g.attn_world = world;
    g.attn_rank = ctx->rank;
    g.max_tok = ctx->max_tok;
    g.max_ctx = ctx->max_ctx;
    g.max_seqs = ctx->max_seqs;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = std::min<int64_t>(g.max_tok, g.max_seqs * (g.max_spec + 1));
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.eps = meta->rms_eps;
    g.wire_exact = !ctx->tp_wire_lossy;
    g.wire_min_bytes = ctx->tp_wire_min_bytes;
    if (g.max_tok < 1 || g.max_ctx < 1 || g.max_seqs < 1 || g.eps <= 0) return RAD_E_INVAL;
    RAD_ARCH_TRY(fusion_switch("RADIANCE_GLM_ROUTER_FUSE", &m.fuse_router));
    RAD_ARCH_TRY(fusion_switch("RADIANCE_GLM_SWIGLU_QUANT", &m.fuse_swiglu));
    RAD_ARCH_TRY(fusion_switch("RADIANCE_GLM_GATHER_INVERSE", &m.gather_inverse));
    RAD_ARCH_TRY(buffers(b, m));
    RadKVGroupDecl d{};
    d.kind = RAD_KV_FULL;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = 256;
    m.latent = rad_decl_kv_group(b, nm.f("kv_latent"), &d);
    d = {};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.state_dim[0] = 4 + g.max_spec;
    d.state_dim[1] = 256;
    m.tail = rad_decl_kv_group(b, nm.f("kv_index_tail"), &d);
    d = {};
    d.kind = RAD_KV_LINEAR;
    d.dtype = RAD_F32;
    d.n_head_kv = 64 / world;
    d.state_dim[0] = d.state_dim[1] = 128;
    m.state = rad_decl_kv_group(b, nm.f("kv_kda"), &d);
    d = {};
    d.kind = RAD_KV_CONV;
    d.dtype = RAD_BF16;
    d.n_head_kv = 1;
    d.head_dim = 3 * 8192 / world;
    d.conv_width = 4;
    m.conv = rad_decl_kv_group(b, nm.f("kv_kda_conv"), &d);
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
    RAD_ARCH_TRY(
        map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.language_model.norm.weight")));
    m.token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, 4096 }, RAD_ACCESS_VOCAB,
                     RAD_SHARD_ROW, grp_model());
    m.head = decl_w(b, nm.f("output.weight"), RAD_BF16, { g.n_vocab, 4096 }, RAD_ACCESS_VOCAB,
                    RAD_SHARD_ROW, grp_model());
    m.final_gain = decl_w(b, nm.f("output_norm.weight"), RAD_F32, { 4096 }, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_model());
    m.embed = rw(b,
                 RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", 4096),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.token)),
                 {}, { m.x });
    m.embed_ar = reduce(b, g, m.x, true);
    m.expand = copy(b, g, m.x, m.h0);
    m.layers.resize(45);
    for (int l = 0; l < 45; ++l) RAD_ARCH_TRY(layer(b, m, m.layers[l], nm, l, false));
    m.contract = rw(b,
                    RAD_OP(b, "mhc_contract",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                      RAD_INT("hc", 4), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                    { m.h0 }, { m.x });
    m.norm = norm(b, g, m.final_gain, m.x, m.final_hidden);
    m.gather = rw(b,
                  RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", 4096),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  { m.final_hidden }, { m.logit_input });
    m.head_op =
        rw(b,
           RAD_OP(b, "logits_gemm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                             RAD_INT("n_embd", 4096), RAD_STR("dtype", "bf16")),
                  RAD_WEIGHTS(m.head)),
           { m.logit_input }, { m.logits });
    if (g.max_spec > 0) {
        auto& dn = m.dnm;
        m.mtp_x = decl_b(b, dn.f("x"), RAD_BF16, { g.max_tok, 4096 });
        m.mtp_h = decl_b(b, dn.f("hidden"), RAD_BF16, { g.max_tok, 4096 }, RAD_BUF_PERSIST);
        m.mtp_e = decl_b(b, dn.f("embedding"), RAD_BF16, { g.max_tok, 4096 });
        m.mtp_concat = decl_b(b, dn.f("concat"), RAD_BF16, { g.max_tok, 8192 });
        m.mtp_final = decl_b(b, dn.f("final"), RAD_BF16, { g.max_tok, 4096 });
        const char* pre = "model.language_model.layers.45";
        auto weight = [&](const char* n, const char* src) {
            const char* name = dn.f("%s", n);
            if (map_copy(b, name, dn.ckpt("%s.%s", pre, src)) < 0) return (rad_weight)0;
            return decl_w(b, name, RAD_F32, { 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(45));
        };
        m.en = weight("enorm.weight", "enorm.weight");
        m.hn = weight("hnorm.weight", "hnorm.weight");
        m.mnorm = weight("output_norm.weight", "shared_head.norm.weight");
        RAD_ARCH_TRY(map_copy(b, dn.f("eh_proj.weight"), dn.ckpt("%s.eh_proj.weight", pre)));
        m.eh = decl_w(b, dn.f("eh_proj.weight"), RAD_BF16, { 4096, 8192 }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_NONE, grp_layer(45));
        m.mprev = rw(b,
                     RAD_OP(b, "gather_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 4096),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { m.final_hidden }, { m.mtp_h });
        m.membed = rw(b,
                      RAD_OP(b, "embed_lookup",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", 4096),
                                        RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                        RAD_INT("vocab_offset", g.vocab_off)),
                             RAD_WEIGHTS(m.token)),
                      {}, { m.mtp_e });
        m.membed_ar = reduce(b, g, m.mtp_e, true);
        m.men = norm(b, g, m.en, m.mtp_e, m.mtp_concat);
        m.mhn = norm(b, g, m.hn, m.mtp_h, m.mtp_concat);
        m.meh = rw(b,
                   RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 4096),
                                     RAD_INT("K", 8192), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(m.eh)),
                   { m.mtp_concat }, { m.mtp_x });
        RAD_ARCH_TRY(layer(b, m, m.draft, dn, 45, true));
        m.mnorm_op = norm(b, g, m.mnorm, m.mtp_x, m.mtp_final);
        m.mcopy = copy(b, g, m.mtp_final, m.mtp_h);
        m.mhead =
            rw(b,
               RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", 4096), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(m.head)),
               { m.mtp_h }, { m.logits });
        RadDrafterDecl dd{};
        dd.name = "mtp";
        dd.kind = RAD_DRAFT_SERIAL;
        dd.depth = g.max_spec;
        dd.mask_token = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        m.have_mtp = true;
    }
    rad_note(b,
             "GLM-5.3: TP%d, 45 mHC layers, 34 KDA + 11 sparse MLA, mixed MXFP4/FP8, native MTP %s",
             world, m.have_mtp ? "enabled" : "disabled");
    return RAD_OK;
}
static void attention(RadCtx* c, Layer& l, const RadBatch* batch) {
    l.norm.step(c, batch->n_tok);
    if (l.full) {
        l.mla.prepare(c, batch);
        l.index.step(c, batch);
        l.mla.finish(c, batch);
    } else
        l.kda.step(c, batch);
}
static void step(RadCtx* c, const RadBatch* batch) {
    auto& m = *models[rad_rank(c)];
    const int64_t T = batch->n_tok;
    if (batch->enc) {
        rad_step_fail(c, "GLM vision is not implemented by this text plugin");
        return;
    }
    if (batch->draft_pass) {
        if (!m.have_mtp || batch->draft_pass == 1) {
            rad_step_fail(c, "GLM MTP pass is not declared or has invalid round 1");
            return;
        }
        if (batch->draft_pass < 0 && batch->n_out > 0)
            RAD_ISSUE_N(c, m.mprev, batch->n_out, RAD_B(m.final_hidden),
                        praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.mtp_h, batch->n_out));
        RAD_ISSUE_N(c, m.membed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token),
                    brows(m.mtp_e, T));
        if (m.membed_ar) RAD_ISSUE_N(c, m.membed_ar, T * 4096, brows(m.mtp_e, T), RAD_NONE);
        RAD_ISSUE_N(c, m.men, T, brows(m.mtp_e, T), RAD_W(m.en), bcol(m.mtp_concat, 0, 4096, T));
        RAD_ISSUE_N(c, m.mhn, T, brows(m.mtp_h, T), RAD_W(m.hn), bcol(m.mtp_concat, 4096, 4096, T));
        RAD_ISSUE_N(c, m.meh, T, brows(m.mtp_concat, T), RAD_W(m.eh), brows(m.mtp_x, T));
        auto& l = m.draft;
        l.norm.step(c, T);
        l.mla.prepare(c, batch);
        l.index.step(c, batch);
        if (batch->draft_pass < 0 && batch->n_draft_out <= 0) return;
        l.mla.finish(c, batch);
        RAD_ISSUE_N(c, l.residual, T, brows(m.mtp_x, T), brows(m.y, T), brows(m.mtp_x, T));
        const int64_t L = batch->draft_pass < 0 ? batch->n_draft_out : T;
        if (batch->draft_pass < 0) {
            RAD_ISSUE_N(c, m.mselect, L, RAD_B(m.mtp_x), praw(batch->draft_out_ids, RAD_I32, L),
                        brows(m.mtp_h, L));
            RAD_ISSUE_N(c, m.mback, L, brows(m.mtp_h, L), brows(m.mtp_x, L));
        }
        l.moe.step_rows(c, L);
        RAD_ISSUE_N(c, m.mnorm_op, L, brows(m.mtp_x, L), RAD_W(m.mnorm), brows(m.mtp_final, L));
        RAD_ISSUE_N(c, m.mcopy, L, brows(m.mtp_final, L), brows(m.mtp_h, L));
        RAD_ISSUE_N(c, m.mhead, L, brows(m.mtp_h, L), RAD_W(m.head), brows(m.logits, L));
        return;
    }
    RAD_ISSUE_N(c, m.embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.token), brows(m.x, T));
    if (m.embed_ar) RAD_ISSUE_N(c, m.embed_ar, T * 4096, brows(m.x, T), RAD_NONE);
    for (int i = 0; i < 4; ++i)
        RAD_ISSUE_N(c, m.expand, T, brows(m.x, T), bcol(m.h0, i * 4096, 4096, T));
    for (int i = 0; i < 45; ++i) {
        auto& l = m.layers[i];
        l.attn_hc.read(c, T);
        attention(c, l, batch);
        l.attn_hc.write(c, T);
        l.ffn_hc.read(c, T);
        if (i < 3)
            l.dense.step(c, batch);
        else
            l.moe.step(c, batch);
        l.ffn_hc.write(c, T);
    }
    RAD_ISSUE_N(c, m.contract, T, brows(m.h0, T), brows(m.x, T));
    RAD_ISSUE_N(c, m.norm, T, brows(m.x, T), RAD_W(m.final_gain), brows(m.final_hidden, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out, brows(m.final_hidden, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.logit_input, batch->n_out));
        RAD_ISSUE_N(c, m.head_op, batch->n_out, brows(m.logit_input, batch->n_out), RAD_W(m.head),
                    brows(m.logits, batch->n_out));
    }
}
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (cfg(meta, "num_nextn_predict_layers", 0) == 1) out->draft_depth = 3;
    out->shape_probe_ok = 1;
    return RAD_OK;
}
} // namespace glm5next
#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(glm5next)
RAD_ARCH_PLUGIN(glm5next, "glm5next", "", "0.1.0",
                "GLM-5.3-Flash mixed MXFP4/FP8, mHC/KDA/MLA and native serial MTP")
#endif
