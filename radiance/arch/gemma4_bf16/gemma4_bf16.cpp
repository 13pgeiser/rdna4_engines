/* gemma4_bf16.cpp -- Gemma 4 dense, bf16, with its vision tower (rad_block_gemma4_vision.h).
 *
 * Two attention classes over one residual stream, alternating by layer. A "sliding" layer is
 * sliding-window attention over head_dim 256 with a v projection and standard RoPE; a "global"
 * layer is windowless attention over head_dim 512 with no v projection (k == v) and proportional
 * RoPE (the "head" ladder) over the partial rotary width. Norms are plain RMSNorm, x_hat * w (see
 * declare), with a second norm on each projection's output before its residual, a weightless RMS
 * norm on the value, and a GELU (not SwiGLU) feed-forward. The final logits are soft-capped to
 * cap * tanh(x / cap).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_gemma4_vision.h>
#include <arch/rad_gemma4_chat.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4_bf16 {

using namespace rad::arch;

/* Residual junctions as one op each. A w4a8 block opens with rmsnorm -> quant_act_i8 and closes
 * with rmsnorm (post) -> add -> [scale], so between two blocks the stream runs five single-row
 * dispatches: the previous post-norm, the add, the layer scalar, this block's norm and its
 * quantiser. libr4d's `rmsnorm_quant_i8` is that chain in one byte-identical dispatch, and q, k and
 * v read one quantised pair instead of three. `qkv_norm_rope` does the same for the attention
 * prologue's five ops, `gelu_mul_quant_i8` for down's input, and at two ranks
 * `ar_rmsnorm_quant_i8` also folds in the all-reduce before each junction. A kernel set without an
 * op keeps that op's chain. */

/* layer_types is optional. When present it must name every layer "full_attention" or
 * "sliding_attention"; when absent, layer l is global when (l + 1) % interval == 0. */
static int g4_parse_types(const char* lt, int64_t n, std::vector<int>& is_global) {
    RAD_ARCH_TRY(parse_layer_kinds("gemma4", lt, {"full_attention"}, "sliding_attention",
                                   &is_global));
    if ((int64_t)is_global.size() != n) {
        fprintf(stderr, "radiance: gemma4: layer_types names %zu layers and the model has %lld\n",
                is_global.size(), (long long)n);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int g4_schedule(const RadModelMeta* m, int64_t n, int64_t interval,
                       std::vector<int>& is_global) {
    const char* lt = meta_text_gets(m, "layer_types", nullptr);
    if (lt && *lt) {
        RAD_ARCH_TRY(g4_parse_types(lt, n, is_global));
    } else {
        if (interval <= 0) {
            fprintf(stderr, "radiance: gemma4: no layer_types and invalid full_attention_interval %lld\n",
                    (long long)interval);
            return RAD_E_INVAL;
        }
        is_global.resize((size_t)n);
        for (int64_t l = 0; l < n; ++l) is_global[(size_t)l] = is_full_attention(l, interval);
    }
    return RAD_OK;
}

/* A linear's weight is its checkpoint's `X.weight`. Google's CT-INT4 (QAT) release ships
 * `X.weight_packed` and `X.weight_scale`, which the checkpoint reader imports as `X.weight`
 * (core/format/checkpoint.h), so one name map serves both releases. */

/* The two attention classes share the residual stream and the FFN but differ in head geometry,
 * the value path and the RoPE. The main Geom (from geom_from) is the slider; the global class is
 * read from the container head and folded in here. */
struct ClassGeom {
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int64_t window = 0;
    double  theta  = 1.0;
    float   scale  = 1.0f;
    bool    head_ladder = false;   /* true: proportional over head_dim (global) */
    bool    has_v       = true;    /* false: k == v (global) */
    int64_t q_dim()  const { return n_head * head_dim; }
    int64_t kv_dim() const { return n_head_kv * head_dim; }
    int64_t gqa()    const { return n_head / n_head_kv; }
};

/* ---- one linear, plain bf16 or w4a8. The recipe quantises GEMM weights to int4 codes with an
 * f16 scale per [1 x 128]; a weight no rule matches stays bf16. The weight's encoding picks the
 * op: `gemm_nt` for plain bf16, `gemm_nt_q` at w4a8 for codes + scale. w4a8 reads an int8
 * activation and an f32 per-row scale, written by a quant_act_i8 declared here over the bf16
 * activation, in the GEMM's fragment byte order. Not quant_act_i8g (row-major, per-(row, group)
 * scale; i8a8's producer), and not w4a16, whose row expects an f16 activation and does not check
 * the operand dtype, so bf16 bits would be misread as f16. The name map is made before the weight
 * is queried (Builder::finish() needs the map entry before the declaration). */
struct G4Linear {
    rad_weight w = 0, ws = 0;   /* the codes + scale (w4a8), or the single bf16 weight */
    rad_op     op = 0, op_q = 0;   /* the quantised GEMM and, for w4a8, its activation quantiser */
    rad_buf    q8 = 0, s8 = 0;     /* the int8 pair (i8 codes + f32 row scale) the quantiser writes */
    int64_t    N = 0, K = 0;
    bool       quant = false;   /* true: w4a8 */
    /* Reads another linear's int8 pair. q, k and v quantise the same norm output on the same
     * per-row grid, so with `in_from` set at declare this linear reads that linear's pair and
     * declares no quantiser of its own: one dispatch instead of three. */
    bool       shared_in = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {},
                 const G4Linear* in_from = nullptr);
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
    /* the GEMM alone, for a caller that wrote the int8 pair itself (a fused norm + quantiser) */
    void gemm(RadCtx* c, rad_buf y, int64_t T) const;
};

inline int G4Linear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src,
                            int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                            rad_buf a, rad_buf y,
                            std::initializer_list<int64_t> row_parts,
                            const G4Linear* in_from) {
    N = n_out; K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1) RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else                RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    /* A builder with no model behind it (a test, a synthetic tool) answers RAD_E_NOTFOUND, and
     * the weight is read as plain bf16. */
    const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                  enc.plane[0].dtype == RAD_BF16);
    int64_t parts[RAD_MAX_ROW_PARTS] = {0};
    int np = 0;
    for (int64_t p : row_parts) if (np < RAD_MAX_ROW_PARTS) parts[np++] = p;

    if (plain) {
        w = decl_w(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, row_parts);
        if (!w) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w)),
                { a }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* A w4 weight: i4 codes and an f16 scale per [1 x 128]. gemm_nt_q at w4a8 reads an int8
     * activation and an f32 per-row scale, so a quantiser over the bf16 plane `a` writes the int8
     * pair (q8, s8). The codes + scale layout is the same as w4a16's, so containers serve
     * unchanged. */
    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
    /* An 8-bit weight on the same grid (i8 codes, f16 scale per [1 x 128]): gemm_nt_q at w8a8
     * reads the same int8 activation and per-row scale as w4a8, so the quantiser is shared. A
     * recipe uses it for linears int4 hurts (attention, down_proj). */
    const uint32_t cdt = known && enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    const bool w4 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 2 && !enc.transform[0] &&
                    (cdt == RAD_I4 || cdt == RAD_I8) && enc.plane[0].block[1] == 1 &&
                    sc && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == 128;
    if (w4) {
        if (N % 16 || K % 128) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a w4 weight takes rows in "
                            "16s and K in whole 128-column scale groups.\n", base,
                    (long long)N, (long long)K);
            return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        w  = decl_view(b, wn, nullptr, "codes", cdt, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        ws = decl_view(b, sn, wn, "scale", RAD_F16, { N, K / 128 }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        if (!w || !ws) return RAD_E_INVAL;
        if (in_from && in_from->quant && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, 1 });
        }
        if (!q8 || !s8) return RAD_E_INVAL;
        /* quant_act_i8, not quant_act_i8g: the w4a8 and w8a8 GEMMs read codes in the int8 WMMA's
         * step order with one f32 scale a row (docs/OPS.md). quant_act_i8g writes row-major codes
         * with a scale per (row, group) for i8a8; paired with these GEMMs it produces NaN logits.
         * `group` only sizes the optional code sums, which this symmetric grid does not take. */
        if (!shared_in)
            op_q = rw(b, RAD_OP(b, "quant_act_i8",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", 128), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", 128), RAD_STR("dtype", cdt == RAD_I8 ? "w8a8" : "w4a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        quant = true;
        return ((op_q || shared_in) && op) ? RAD_OK : RAD_E_INVAL;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, w4a8 or w8a8 -- "
                    "neither matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void G4Linear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        /* Quantise the bf16 activation into the int8 pair, then run the GEMM. quant_act_i8 takes
         * x, q, scale, asum; `asum` (the per-group row sums asymmetric grids subtract) is absent on
         * this symmetric grid and passed as a null tensor, since operands are positional. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        gemm(c, y, T);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
}

inline void G4Linear::gemm(RadCtx* c, rad_buf y, int64_t T) const {
    RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                brows(y, T), RAD_NONE, RAD_NONE);
}

/* ---- one attention block, parameterised by class. */
struct Gemma4Attn {
    struct Wire { rad_buf x = 0, h = 0, q = 0, k = 0, v = 0, attn = 0; };

    Wire      w{};
    Geom      g{};
    ClassGeom cls{};
    int         layer = 0;
    rad_kvgroup kv    = 0;

    rad_weight w_norm = 0, w_qn = 0, w_kn = 0, w_post = 0;   /* the norms (always bf16) */
    G4Linear   ln_q{}, ln_k{}, ln_v{}, ln_o{};              /* the GEMM weights (bf16 or w4a8) */
    rad_op op_norm = 0, op_vn = 0, op_qn = 0, op_kn = 0,
           op_rq = 0, op_rk = 0, op_attn = 0, op_kv = 0, op_post = 0, op_ar = 0, op_add = 0;
    /* the five prologue ops (v norm, q norm, k norm, two rotaries) as one, where it resolves */
    rad_op op_qnr = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                 rad_kvgroup group, const Wire& wire, const char* pre);
    /* `entry_done`: a junction op already wrote this block's quantised input (ln_q's pair).
     * `tail_fused`: the caller issues the post-norm and the add as a junction op. */
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
};

int Gemma4Attn::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                        rad_kvgroup group, const Wire& wire, const char* pre) {
    g = geom; cls = cgeo; layer = l; kv = group; w = wire;
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();

    /* The norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),      nm.ckpt("%s.%lld.input_layernorm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.q_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.k_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post_norm.weight", l), nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
    /* Declared bf16, not f32: the checkpoint holds them bf16, and a widened weight cannot be
     * served from the mapped tier. Reading the gain as bf16 gives the same values. */
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qn = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_kn = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.attn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* The GEMM weights: bf16 or w4a8, probed by G4Linear. */
    RAD_ARCH_TRY(ln_q.declare(b, nm, g, nm.f("blk.%d.attn_q", l),
                              { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                              qd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.q));
    const G4Linear* qin = &ln_q;   /* k and v read q's int8 pair */
    RAD_ARCH_TRY(ln_k.declare(b, nm, g, nm.f("blk.%d.attn_k", l),
                              { nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l) },
                              kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.k, {}, qin));
    if (cls.has_v)
        RAD_ARCH_TRY(ln_v.declare(b, nm, g, nm.f("blk.%d.attn_v", l),
                                  { nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l) },
                                  kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.v, {}, qin));
    RAD_ARCH_TRY(ln_o.declare(b, nm, g, nm.f("blk.%d.attn_o", l),
                              { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                              g.n_embd, qd, RAD_SHARD_COL, grp_layer(l), w.attn, w.h));

    /* input norm: x_hat * w over the residual stream. */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* Gemma normalises the value with a weightless RMS norm (the checkpoint has no v_norm gain).
     * A sliding class normalises v in place; a k_eq_v class derives v from the raw k into its own
     * buffer, which must happen before k_norm rewrites k. */
    op_vn = rw(b, RAD_OP(b, "rmsnorm_p",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head_kv), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {cls.has_v ? w.v : w.k}, {w.v});

    op_qn = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_qn)),
             {w.q}, {w.q});
    op_kn = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * cls.n_head_kv), RAD_INT("n", cls.head_dim),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_kn)),
             {w.k}, {w.k});

    /* sliding: standard RoPE over the full width, theta_s. global: proportional over the partial
     * width (the "head" ladder), theta_g. q names its query heads; k its kv heads. */
    op_rq = rw(b, RAD_OP(b, "rope",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                            RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", 0),
                            RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                            RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                            RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                 RAD_NOWEIGHTS),
                {w.q}, {w.q});
    /* The rope kernel rotates n_head + n_head_kv heads of a fused QKV row and requires
     * n_head > 0. K is a separate buffer, so it is rotated as a Q-only form over the K heads:
     * n_head = n_head_kv and n_head_kv = 0, a row of n_head_kv*head_dim, exactly K's width. Q and K
     * share theta and positions, so this matches the reference's separate K rope. */
    op_rk = rw(b, RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                             RAD_INT("n_head", cls.n_head_kv), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                             RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                  RAD_NOWEIGHTS),
                 {w.k}, {w.k});

    /* Gemma 4 scores Q.K without the 1/sqrt(head_dim) factor (HF Gemma4TextAttention and vLLM
     * use scaling = 1.0; the q/k norm gains absorb it). attn_paged defaults to 1/sqrt(head_dim),
     * so 1.0 is passed explicitly. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                   RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                              RAD_INT("gqa", cls.gqa()), RAD_INT("causal", 1), RAD_INT("window", cls.window),
                              RAD_INT("n_head", cls.n_head), RAD_INT("block_size", RAD_KV_BLOCK),
                              RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                              RAD_F64("scale", 1.0),
                              RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                   RAD_NOWEIGHTS),
                  {w.q}, {w.attn});

    /* The prologue as one op: libr4d's `qkv_norm_rope` is rmsnorm_p + rmsnorm + rmsnorm + rope +
     * rope over the three projections, byte-identical, one workgroup per (token, head). Declared
     * beside the unfused five and used only where it resolves; a multi-component or YaRN rotary
     * keeps the five, as does an engine whose libraries do not declare it (CPU-only), where asking
     * for it would stop the declare. */
    if (!g.rope_mc && g.yarn.factor == 0.0f && rad_op_declared(b, "qkv_norm_rope")) {
        op_qnr = RAD_OP(b, "qkv_norm_rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                                 RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", cls.n_head_kv),
                                 RAD_F64("theta", cls.theta), RAD_F64("eps", g.eps),
                                 RAD_F64("scale", cls.scale), RAD_STR("mode", "neox"),
                                 RAD_INT("rotary_dim", cls.n_rot),
                                 RAD_STR("ladder", cls.head_ladder ? "head" : "rotary"),
                                 RAD_F64("wadd", g.wadd_qk), RAD_INT("v_from_k", cls.has_v ? 0 : 1),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_qn, w_kn));
        /* A k_eq_v class derives v from the raw k: v is written, never read. */
        op_qnr = cls.has_v ? rw(b, op_qnr, {w.q, w.k, w.v}, {w.q, w.k, w.v})
                           : rw(b, op_qnr, {w.q, w.k}, {w.q, w.k, w.v});
    }

    op_kv = rw(b, RAD_OP(b, "kv_store",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                           RAD_INT("n_head_kv", cls.n_head_kv),
                           RAD_INT("block_size", rad_kv_block_size(b, kv)),
                           RAD_STR("kv_dtype", g.kv_dtype)),
                RAD_NOWEIGHTS),
                {w.k, w.v}, {});

    /* o_proj is column-sharded: the rank's partial sum is reduced first, then normed. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});

    /* The second norm: the reduced attention output is normed before the residual add. */
    op_post = rw(b, RAD_OP(b, "rmsnorm",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                            RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(w_post)),
                {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.x, w.h}, {w.x});
    return RAD_OK;
}

void Gemma4Attn::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                      bool ar_fused) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;

    if (entry_done) {
        ln_q.gemm(c, w.q, T);
        ln_k.gemm(c, w.k, T);
        if (cls.has_v) ln_v.gemm(c, w.v, T);
    } else {
        RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
        ln_q.step(c, w.h, w.q, T);
        ln_k.step(c, w.h, w.k, T);
        if (cls.has_v) ln_v.step(c, w.h, w.v, T);
    }

    if (op_qnr) {
        RAD_ISSUE(c, op_qnr, brows(w.q, T), brows(w.k, T), brows(w.v, T), rope_posmc(g, batch, T),
                  RAD_W(w_qn), RAD_W(w_kn));
    } else {
        /* Normalise v first: on a k_eq_v class v derives from the raw k, before k_norm rewrites it. */
        RAD_ISSUE_N(c, op_vn, T * cls.n_head_kv, brows(cls.has_v ? w.v : w.k, T), brows(w.v, T));
        RAD_ISSUE_N(c, op_qn, T * cls.n_head,    brows(w.q, T), RAD_W(w_qn), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn, T * cls.n_head_kv, brows(w.k, T), RAD_W(w_kn), brows(w.k, T));

        RAD_ISSUE(c, op_rq, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rk, brows(w.k, T), rope_posmc(g, batch, T));
    }

    RAD_ISSUE(c, op_kv, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    ln_o.step(c, w.attn, w.h, T);
    /* The all-reduce comes before the post-norm: o_proj is column-sharded, so each rank holds a
     * partial sum, and the norm is of the whole output (sum_r rmsnorm(p_r) != rmsnorm(sum_r p_r)). */
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,  brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

/* ---- the feed-forward: GELU (not SwiGLU), with a second norm on the down-projection output. */
struct Gemma4Mlp {
    struct Wire { rad_buf x = 0, h = 0, gu = 0, ffn = 0; };

    Wire  w{};
    Geom  g{};
    int   layer = 0;
    rad_weight w_norm = 0, w_post = 0;   /* the norms (always bf16) */
    G4Linear   ln_gu{}, ln_down{};      /* the GEMM weights (bf16 or w4a8) */
    rad_op op_norm = 0, op_act = 0, op_post = 0, op_ar = 0, op_add = 0;
    /* gelu_mul and down's quantiser as one, where it resolves (libr4d `gelu_mul_quant_i8`) */
    rad_op op_gq = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire, const char* pre);
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
};

int Gemma4Mlp::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire, const char* pre) {
    g = geom; layer = l; w = wire;

    /* The norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_norm.weight", l), nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_post_norm.weight", l), nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
    /* bf16, not f32, as for the attention norms: a widened weight cannot be served from the
     * mapped tier. */
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.ffn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* The GEMM weights: bf16 or w4a8, probed by G4Linear. gate_up is the gate|up fusion. */
    RAD_ARCH_TRY(ln_gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                               { nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                 nm.ckpt("%s.%lld.mlp.up_proj.weight", pre, (long long)l) },
                               2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.gu,
                               { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(ln_down.declare(b, nm, g, nm.f("blk.%d.ffn_down", l),
                                 { nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l) },
                                 g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l), w.ffn, w.h));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                           RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(w_norm)),
               {w.x}, {w.h});
    op_act = rw(b, RAD_OP(b, "gelu_mul",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.gu}, {w.ffn});
    if (ln_down.quant && !ln_down.shared_in && rad_op_declared(b, "gelu_mul_quant_i8"))
        op_gq = rw(b, RAD_OP(b, "gelu_mul_quant_i8",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                   {w.gu}, {ln_down.q8, ln_down.s8});
    /* down_proj is column-sharded: reduce first, then the post-norm. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_post = rw(b, RAD_OP(b, "rmsnorm",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                          RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
               RAD_WEIGHTS(w_post)),
              {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
              {w.x, w.h}, {w.x});
    return RAD_OK;
}

void Gemma4Mlp::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                     bool ar_fused) const {
    const int64_t T = batch->n_tok;
    if (entry_done) {
        ln_gu.gemm(c, w.gu, T);
    } else {
        RAD_ISSUE(c, op_norm,  brows(w.x, T),   RAD_W(w_norm),  brows(w.h, T));
        ln_gu.step(c, w.h, w.gu, T);
    }
    if (op_gq) {
        RAD_ISSUE_N(c, op_gq, T, brows(w.gu, T), brows(ln_down.q8, T), brows(ln_down.s8, T), RAD_NONE);
        ln_down.gemm(c, w.h, T);
    } else {
        RAD_ISSUE(c, op_act,   brows(w.gu, T),  brows(w.ffn, T));
        ln_down.step(c, w.ffn, w.h, T);
    }
    /* The all-reduce before the post-norm, as in the attention block: down_proj is column-sharded. */
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,   brows(w.x, T),   brows(w.h, T),  brows(w.x, T));
}

/* ---- one residual junction as one op: [post-norm of the branch `x`, the add into `res`, the
 * layer scalar,] the next block's norm and its int8 quantiser (libr4d `rmsnorm_quant_i8`). With
 * `res` null it is the plain norm of `x`, the first block's entry. */
struct G4Junction {
    rad_op     op = 0;
    rad_op     op_ar = 0;   /* the same with the two-rank exact all-reduce of `x` in front, or 0 */
    rad_buf    x = 0, res = 0, q8 = 0, s8 = 0;
    rad_weight w = 0, wp = 0, sc = 0;

    int declare(RadBuilder* b, const Geom& g, rad_buf x_, rad_buf res_, rad_weight w_,
                rad_weight wp_, rad_weight sc_, const G4Linear& into) {
        x = x_; res = res_; w = w_; wp = wp_; sc = sc_; q8 = into.q8; s8 = into.s8;
        if (!into.quant || into.shared_in || !q8 || !s8 || !w || (sc && !wp) || ((wp || sc) && !res))
            return RAD_E_INVAL;
        const RadParam p[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                               RAD_F64("wadd", g.wadd) };
        const rad_weight ws[3] = { w, wp, sc };
        const int nw = sc ? 3 : (wp ? 2 : 1);
        /* Not declared by any loaded library (CPU-only) or not resolved: the caller keeps the
         * unfused chain. */
        if (rad_op_declared(b, "rmsnorm_quant_i8"))
            op = rad_decl_op(b, "rmsnorm_quant_i8", p, (int)(sizeof p / sizeof p[0]), ws, nw);
        if (!op) return RAD_OK;
        const rad_buf rd[2] = { x, res };
        const rad_buf wr[3] = { q8, s8, res };
        rad_op_reads (b, op, rd, res ? 2 : 1);
        rad_op_writes(b, op, wr, res ? 3 : 2);
        /* With the all-reduce in front, at two ranks on the exact wire: `x` is then a rank's
         * partial sum of a column-sharded linear. With the lossy wire allowed, the standalone
         * all-reduce is kept, since it may choose that wire. */
        if (res && g.world == 2 && g.wire_exact && rad_op_declared(b, "ar_rmsnorm_quant_i8")) {
            const RadParam pa[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_F64("eps", g.eps), RAD_INT("world_size", g.world),
                                    RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd) };
            op_ar = rad_decl_op(b, "ar_rmsnorm_quant_i8", pa, (int)(sizeof pa / sizeof pa[0]), ws, nw);
            if (op_ar) {
                const rad_buf wa[4] = { x, q8, s8, res };
                rad_op_reads (b, op_ar, rd, 2);
                rad_op_writes(b, op_ar, wa, 4);
            }
        }
        return RAD_OK;
    }
    void step(RadCtx* c, int64_t T, bool with_ar = false) const {
        RAD_ISSUE_N(c, with_ar ? op_ar : op, T, brows(x, T), res ? brows(res, T) : RAD_NONE, RAD_W(w),
                    brows(q8, T), brows(s8, T), RAD_NONE,
                    wp ? RAD_W(wp) : RAD_NONE, sc ? RAD_W(sc) : RAD_NONE);
    }
};

/* ==================================================================== the assistant head
 *
 * Gemma 4's draft head, the "assistant" (Gemma4AssistantForCausalLM), merged under `draft.` by
 * `rad-convert --draft-model <gemma-4-*-it-assistant>`. From the checkpoint and three references:
 * transformers' modeling_gemma4_assistant.py with its candidate generator, vLLM's gemma4_mtp.py
 * with spec_decode/gemma4.py, and llama.cpp's gemma4-assistant.cpp:
 *
 *     x  = pre_projection( concat( E_trunk[tok] * sqrt(n_trunk), h_trunk ) )        2*n_trunk -> n
 *     x  = the head's layers over x (31B: sliding, sliding, sliding, full)
 *     hd = norm(x)
 *     logits = lm_head(hd)       the head's own table, tied to its embed_tokens; no soft-cap
 *     h' = post_projection(hd)                                                       n -> n_trunk
 *
 * `h_trunk` is the trunk's final-normed hidden state (the trunk lm_head's input) and `E_trunk` the
 * trunk's embedding table, scaled as the trunk scales it. The head's own embed_tokens serves only
 * as its lm_head (vLLM uses the trunk's table on the input side). The concatenation is embedding
 * first, then hidden state; all three references agree.
 *
 * A head layer is a trunk layer without key and value: q_proj, the per-head q norm, its class's
 * rotary, attention over the trunk's cache at scale 1, o_proj, the same four plain-gain norms, the
 * GELU feed-forward and the per-layer scalar. Each layer reads the last trunk layer of its class
 * (58 and 59 on the 31B; vLLM's _setup_gemma4_kv_sharing), so the head's q heads and the trunk's
 * kv heads must share geometry, which declare checks.
 *
 * The head keeps no cache of its own (rad_drafter_shared_kv): no history pass, one row per
 * drafting sequence, every round at the committed position over the committed context. Round 1
 * embeds the token the trunk chose and continues from the trunk's hidden state; round r embeds
 * round r-1's proposal and continues from its h'. Those two are the only path from one round to
 * the next, by design of the head.
 *
 * Tensor-parallel split as the trunk's: q heads row-sharded, so each rank's queries meet its share
 * of the trunk's kv heads; o_proj and the feed-forward down projection column-sharded, each
 * followed by an all-reduce before its post-norm. The two projections and every norm are
 * replicated, since they read and write the replicated stream. The lm_head is vocab-sharded like
 * the trunk's and goes through the same sampler chain, greedy. */
/* Where the head's rows sit for the rotary, relative to the committed position
 * (rad_drafter_shared_kv), and the depth `--num-speculative-tokens auto` takes. */
static constexpr int kHeadPosOffset = 0;
static constexpr int kHeadDepth     = 3;

struct Gemma4Head {
    struct HLayer {
        bool        global = false;
        int         trunk_layer = 0;   /* the trunk layer whose cache this layer reads */
        rad_kvgroup kv = 0;
        ClassGeom   cls{};
        rad_buf     q = 0, attn = 0;   /* the class's own rows: the two widths differ */
        rad_weight  w_norm = 0, w_qn = 0, w_post = 0, w_fnorm = 0, w_fpost = 0, w_scalar = 0;
        G4Linear    ln_q{}, ln_o{}, ln_gu{}, ln_down{};
        rad_op      op_norm = 0, op_qn = 0, op_rq = 0, op_attn = 0, op_ar = 0, op_post = 0,
                    op_add = 0, op_fnorm = 0, op_act = 0, op_far = 0, op_fpost = 0, op_fadd = 0,
                    op_scale = 0;
    };

    Geom    g{};            /* the trunk's, with max_tok cut to the head's rows */
    int64_t n = 0;          /* the head's width (1024 on the 31B) */
    int64_t n_trunk = 0;    /* the trunk's width, which the two projections join */
    int64_t n_ff = 0;       /* this rank's share of the feed-forward */
    int64_t rows = 0;       /* the most rows one pass carries: one per drafting sequence */
    double  softcap = 0.0;  /* 0: none, which is what the 31B's head says */
    std::vector<HLayer> layers;

    rad_buf src = 0;        /* the trunk's final-normed hidden state, [max_tok, n_trunk] */
    rad_buf logits = 0;     /* the trunk's logits plane: the sampler reads the head's from it */
    rad_buf b_c = 0, b_e = 0, b_x = 0, b_h = 0, b_gu = 0, b_ffn = 0, b_hd = 0, b_hn = 0;
    rad_weight w_tok = 0, w_norm = 0, w_lm = 0;
    G4Linear   ln_pre{}, ln_post{};
    rad_op op_hsrc = 0, op_hcopy = 0, op_embed = 0, op_embed_ar = 0, op_ecopy = 0, op_norm = 0,
           op_logits = 0, op_cap = 0;

    int  declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                 const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s, rad_kvgroup kv_g,
                 int last_sliding, int last_full, int64_t n_main, rad_weight tok_embd,
                 rad_buf trunk_h, rad_buf trunk_logits);
    void step(RadCtx* c, const RadBatch* batch) const;
};

/* Does the container carry an assistant head at all? */
static bool has_assistant(const RadModelMeta* m) {
    const char* a = rad_meta_gets(m, "draft.architectures", "");
    return a && std::strstr(a, "Gemma4Assistant") != nullptr;
}

int Gemma4Head::declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const Geom& trunk,
                        const ClassGeom& cs, const ClassGeom& cg, rad_kvgroup kv_s,
                        rad_kvgroup kv_g, int last_sliding, int last_full, int64_t n_main,
                        rad_weight tok_embd, rad_buf trunk_h, rad_buf trunk_logits) {
    g = trunk;
    rows = trunk.max_seqs > 0 ? trunk.max_seqs : 1;
    g.max_tok = rows;
    src = trunk_h;
    logits = trunk_logits;
    w_tok = tok_embd;

    /* ---- the head's own geometry, and what it must share with the trunk's. */
    n       = meta_text_geti(meta, "hidden_size", 0, "draft.");
    n_trunk = rad_meta_geti(meta, "draft.backbone_hidden_size", 0);
    const long long n_l      = meta_text_geti(meta, "num_hidden_layers", 0, "draft.");
    const long long n_h      = meta_text_geti(meta, "num_attention_heads", 0, "draft.");
    const long long n_kv     = meta_text_geti(meta, "num_key_value_heads", 0, "draft.");
    const long long n_kv_g   = meta_text_geti(meta, "num_global_key_value_heads", n_kv, "draft.");
    const long long head_s   = meta_text_geti(meta, "head_dim", 0, "draft.");
    const long long head_g   = meta_text_geti(meta, "global_head_dim", head_s, "draft.");
    const long long ff       = meta_text_geti(meta, "intermediate_size", 0, "draft.");
    const long long win      = meta_text_geti(meta, "sliding_window", 0, "draft.");
    const long long vocab    = meta_text_geti(meta, "vocab_size", 0, "draft.");
    const long long ordered  = rad_meta_geti(meta, "draft.use_ordered_embeddings", 0);
    const double    eps      = meta_text_getf(meta, "rms_norm_eps", 0.0, "draft.");
    const double    theta_s  = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0, "draft.");
    const double    theta_g  = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0, "draft.");
    double          prf      = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0, "draft.");
    softcap = meta_text_getf(meta, "final_logit_softcapping", 0.0, "draft.");
    if (!(prf > 0.0)) prf = 1.0;

    if (n <= 0 || n_l <= 0 || n_h <= 0 || ff <= 0 || !(eps > 0.0)) {
        fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config is incomplete (hidden %lld, "
                        "layers %lld, heads %lld, ffn %lld, eps %g)\n",
                (long long)n, n_l, n_h, ff, eps);
        return RAD_E_INVAL;
    }
    /* The head reads the trunk's cache, so everything the cache's shape depends on must agree:
     * the two head widths, the two kv head counts and the window. A head that disagrees was trained
     * with another trunk. */
    if (n_trunk != trunk.n_embd || vocab != trunk.n_vocab_all || head_s != cs.head_dim ||
        head_g != cg.head_dim || n_kv != cs.n_head_kv * trunk.world ||
        n_kv_g != cg.n_head_kv * trunk.world || win != cs.window) {
        fprintf(stderr, "radiance: gemma4 assistant: this head was not made for this trunk -- it "
                        "states width %lld, vocabulary %lld, heads %lld/%lld x %lld/%lld kv over a "
                        "window of %lld; the trunk's cache is %lld/%lld kv heads of %lld/%lld over "
                        "%lld, at width %lld and vocabulary %lld\n",
                (long long)n_trunk, vocab, n_kv, n_kv_g, head_s, head_g, win,
                (long long)(cs.n_head_kv * trunk.world), (long long)(cg.n_head_kv * trunk.world),
                (long long)cs.head_dim, (long long)cg.head_dim, (long long)cs.window,
                (long long)trunk.n_embd, (long long)trunk.n_vocab_all);
        return RAD_E_INVAL;
    }
    /* The queries must rotate as the trunk's keys did, or q.k measures the wrong distance. */
    const int64_t rot_g = (int64_t)(head_g * prf + 0.5);
    if (theta_s != cs.theta || theta_g != cg.theta || rot_g != cg.n_rot) {
        fprintf(stderr, "radiance: gemma4 assistant: its rotary (theta %g / %g, %lld of %lld global "
                        "dims) is not the trunk's (%g / %g, %lld): its queries would meet the "
                        "trunk's keys at the wrong angle\n",
                theta_s, theta_g, (long long)rot_g, head_g, cs.theta, cg.theta,
                (long long)cg.n_rot);
        return RAD_E_INVAL;
    }
    /* Centroid masking (use_ordered_embeddings) scores a top-k of vocabulary clusters instead of
     * the whole head. The 31B's head does not use it and it is not implemented, so it is refused. */
    if (ordered) {
        fprintf(stderr, "radiance: gemma4 assistant: use_ordered_embeddings (centroid-masked logits) "
                        "is not implemented\n");
        return RAD_E_UNSUPPORTED;
    }
    int64_t nh = 0;
    if (divide_or_fail(n_h, trunk.world, "draft num_attention_heads", &nh) < 0) return RAD_E_INVAL;
    if (divide_or_fail(ff, trunk.world, "draft intermediate_size", &n_ff) < 0) return RAD_E_INVAL;
    if (nh % cs.n_head_kv || nh % cg.n_head_kv) {
        fprintf(stderr, "radiance: gemma4 assistant: %lld query heads a rank do not group over %lld "
                        "sliding / %lld global kv heads\n",
                (long long)nh, (long long)cs.n_head_kv, (long long)cg.n_head_kv);
        return RAD_E_INVAL;
    }

    /* The layer schedule: `layer_types` names every layer, as the trunk's does. */
    std::vector<int> is_global;
    {
        const char* lt = rad_meta_gets(meta, "draft.text_config.layer_types",
                                       rad_meta_gets(meta, "draft.layer_types", nullptr));
        if (!lt || !*lt || g4_parse_types(lt, n_l, is_global) < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config's layer_types does not "
                            "name its %lld layers\n", n_l);
            return RAD_E_INVAL;
        }
    }

    const int64_t base = n_main;   /* the head's weight groups, past the trunk's layers */
    g.eps = (float)eps;

    /* ---- the rows. The two classes keep their own query and attention rows, as in the trunk. */
    ClassGeom hs = cs; hs.n_head = nh;
    ClassGeom hgc = cg; hgc.n_head = nh;
    b_c   = decl_b(b, nm.f("mtp_c"),   g.act_dtype, {rows, 2 * n_trunk});
    b_e   = decl_b(b, nm.f("mtp_e"),   g.act_dtype, {rows, n_trunk});
    b_x   = decl_b(b, nm.f("mtp_x"),   g.act_dtype, {rows, n});
    b_h   = decl_b(b, nm.f("mtp_h"),   g.act_dtype, {rows, n});
    b_gu  = decl_b(b, nm.f("mtp_gu"),  g.act_dtype, {rows, 2 * n_ff});
    b_ffn = decl_b(b, nm.f("mtp_ffn"), g.act_dtype, {rows, n_ff});
    b_hd  = decl_b(b, nm.f("mtp_hd"),  g.act_dtype, {rows, n});
    b_hn  = decl_b(b, nm.f("mtp_hn"),  g.act_dtype, {rows, n_trunk});
    const rad_buf q_s = decl_b(b, nm.f("mtp_qs"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf a_s = decl_b(b, nm.f("mtp_as"), g.act_dtype, {rows, hs.q_dim()});
    const rad_buf q_g = decl_b(b, nm.f("mtp_qg"), g.act_dtype, {rows, hgc.q_dim()});
    const rad_buf a_g = decl_b(b, nm.f("mtp_ag"), g.act_dtype, {rows, hgc.q_dim()});

    /* ---- the entry: the trunk's hidden state into the right half of `c`, the trunk's scaled
     * embedding into the left. A history pass gathers the hidden state from the trunk's rows; a
     * round copies the head's own h' from the previous round. Both write the half in place, since
     * gather and cast take the destination's row pitch from the operand. */
    op_hsrc = rw(b, RAD_OP(b, "gather_rows",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
                 {src}, {b_c});
    op_hcopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_hn}, {b_c});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", n_trunk),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", std::sqrt((double)n_trunk))),
                    RAD_WEIGHTS(w_tok)),
                  {}, {b_e});
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                             RAD_PARAMS(RAD_INT("world_size", g.world),
                                        RAD_RANGE("numel", n_trunk, rows * n_trunk),
                                        RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                             RAD_NOWEIGHTS),
                         {b_e}, {b_e});
    op_ecopy = rw(b, RAD_OP(b, "cast",
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_trunk),
                               RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                    RAD_NOWEIGHTS),
                  {b_e}, {b_c});
    if (!op_hsrc || !op_hcopy || !op_embed || !op_ecopy) return RAD_E_INVAL;

    /* pre_projection: the joined row to the head's width. Replicated, like the stream it reads
     * and writes. */
    RAD_ARCH_TRY(ln_pre.declare(b, nm, g, nm.f("mtp.pre_proj"), { nm.ckpt("draft.pre_projection.weight") },
                                n, 2 * n_trunk, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                b_c, b_x));

    /* The head's n-wide norm (plain gain), residual add and all-reduce. */
    auto norm = [&](rad_weight w, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w)),
                  {x}, {y});
    };
    auto add = [&]() {
        return rw(b, RAD_OP(b, "add",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                  {b_x, b_h}, {b_x});
    };
    auto all_reduce = [&]() {
        return rw(b, RAD_OP(b, "all_reduce",
                     RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                RAD_INT("min_bytes", g.wire_min_bytes)),
                     RAD_NOWEIGHTS),
                  {b_h}, {b_h});
    };

    /* ---- the layers. */
    layers.resize((size_t)n_l);
    for (int64_t l = 0; l < n_l; ++l) {
        HLayer& L = layers[(size_t)l];
        const int gl = (int)(base + l);
        L.global = is_global[(size_t)l] == 1;
        L.cls = L.global ? hgc : hs;
        L.trunk_layer = L.global ? last_full : last_sliding;
        L.kv = L.global ? kv_g : kv_s;
        L.q = L.global ? q_g : q_s;
        L.attn = L.global ? a_g : a_s;
        if (L.trunk_layer < 0) {
            fprintf(stderr, "radiance: gemma4 assistant: layer %lld reads a %s trunk layer and the "
                            "trunk has none\n", (long long)l, L.global ? "full" : "sliding");
            return RAD_E_INVAL;
        }
        const char* pre = "draft.model.layers";
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.input_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l),
                              nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l),
                              nm.ckpt("%s.%lld.layer_scalar", pre, (long long)l)));
        L.w_norm   = decl_w(b, nm.f("mtp.blk.%lld.attn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_qn     = decl_w(b, nm.f("mtp.blk.%lld.attn_q_norm.weight", (long long)l), RAD_BF16,
                            {L.cls.head_dim}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_post   = decl_w(b, nm.f("mtp.blk.%lld.attn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fnorm  = decl_w(b, nm.f("mtp.blk.%lld.ffn_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_fpost  = decl_w(b, nm.f("mtp.blk.%lld.ffn_post_norm.weight", (long long)l), RAD_BF16, {n},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));
        L.w_scalar = decl_w(b, nm.f("mtp.blk.%lld.layer_scalar", (long long)l), RAD_BF16, {1},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(gl));

        RAD_ARCH_TRY(L.ln_q.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_q", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                                    L.cls.q_dim(), n, RAD_SHARD_ROW, grp_layer(gl), b_h, L.q));
        RAD_ARCH_TRY(L.ln_o.declare(b, nm, g, nm.f("mtp.blk.%lld.attn_o", (long long)l),
                                    { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                                    n, L.cls.q_dim(), RAD_SHARD_COL, grp_layer(gl), L.attn, b_h));
        RAD_ARCH_TRY(L.ln_gu.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_gate_up", (long long)l),
                                     { nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                       nm.ckpt("%s.%lld.mlp.up_proj.weight",   pre, (long long)l) },
                                     2 * n_ff, n, RAD_SHARD_ROW, grp_layer(gl), b_h, b_gu,
                                     { n_ff, n_ff }));
        RAD_ARCH_TRY(L.ln_down.declare(b, nm, g, nm.f("mtp.blk.%lld.ffn_down", (long long)l),
                                       { nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l) },
                                       n, n_ff, RAD_SHARD_COL, grp_layer(gl), b_ffn, b_h));

        L.op_norm = norm(L.w_norm, b_x, b_h);
        L.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows * L.cls.n_head), RAD_INT("n", L.cls.head_dim),
                                  RAD_F64("eps", eps), RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(L.w_qn)),
                     {L.q}, {L.q});
        L.op_rq = rw(b, RAD_OP(b, "rope",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", L.cls.head_dim),
                                  RAD_INT("n_head", L.cls.n_head), RAD_INT("n_head_kv", 0),
                                  RAD_F64("theta", L.cls.theta), RAD_F64("scale", L.cls.scale),
                                  RAD_STR("mode", "neox"), RAD_INT("rotary_dim", L.cls.n_rot),
                                  RAD_STR("ladder", L.cls.head_ladder ? "head" : "rotary")),
                       RAD_NOWEIGHTS),
                     {L.q}, {L.q});
        /* One query row per sequence over the trunk's cache, at scale 1 like the trunk. The
         * context is the committed positions (rad_drafter_shared_kv), so only the decode band
         * exists. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", L.cls.head_dim),
                                    RAD_INT("gqa", L.cls.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", L.cls.window), RAD_INT("n_head", L.cls.n_head),
                                    RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", rows),
                                    RAD_INT("max_ctx", g.max_ctx), RAD_F64("scale", 1.0),
                                    RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                       {L.q}, {L.attn});
        if (g.world > 1) L.op_ar = all_reduce();
        L.op_post  = norm(L.w_post, b_h, b_h);
        L.op_add   = add();
        L.op_fnorm = norm(L.w_fnorm, b_x, b_h);
        L.op_act = rw(b, RAD_OP(b, "gelu_mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_ff), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                      {b_gu}, {b_ffn});
        if (g.world > 1) L.op_far = all_reduce();
        L.op_fpost = norm(L.w_fpost, b_h, b_h);
        L.op_fadd  = add();
        L.op_scale = rw(b, RAD_OP(b, "scale",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(L.w_scalar)),
                        {b_x}, {b_x});
        if (!L.op_norm || !L.op_qn || !L.op_rq || !L.op_attn || !L.op_post || !L.op_add ||
            !L.op_fnorm || !L.op_act || !L.op_fpost || !L.op_fadd || !L.op_scale ||
            (g.world > 1 && (!L.op_ar || !L.op_far)))
            return RAD_E_INVAL;
    }

    /* ---- the exit: the final norm, the head's own lm_head, and h' for the next round. */
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output_norm.weight"), nm.ckpt("draft.model.norm.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.output.weight"), nm.ckpt("draft.model.embed_tokens.weight")));
    w_norm = decl_w(b, nm.f("mtp.output_norm.weight"), RAD_BF16, {n}, RAD_ACCESS_PER_TOKEN,
                    RAD_SHARD_NONE, grp_layer((int)(base + n_l)));
    /* Vocab-sharded like the trunk's: the draft's logits go through the same sampler chain, which
     * merges the ranks' candidates. Declared at its codes' dtype (bf16 as shipped, or int8 / block
     * fp8 where a recipe makes it so), selecting the matching logits_gemm kernel. It is a round's
     * largest read, and quantising it only affects proposals, which the trunk verifies. */
    w_lm = decl_w(b, nm.f("mtp.output.weight"), weight_codes_dtype(b, nm.f("mtp.output.weight"), RAD_BF16),
                  {g.n_vocab, n}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_layer((int)(base + n_l)));
    op_norm = norm(w_norm, b_x, b_hd);
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_vocab", g.n_vocab),
                                RAD_INT("n_embd", n), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_lm)),
                   {b_hd}, {logits});
    if (softcap > 0.0)
        op_cap = rw(b, RAD_OP(b, "logit_softcap",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_vocab),
                                 RAD_F64("cap", softcap), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS),
                    {logits}, {logits});
    if (!op_norm || !op_logits || (softcap > 0.0 && !op_cap)) return RAD_E_INVAL;
    /* post_projection last: h' is the next round's input, read by the head's first op
     * (op_hcopy). Declared here, its buffer lives across the whole head, so nothing in a pass can
     * overwrite it before the next round reads it. */
    RAD_ARCH_TRY(ln_post.declare(b, nm, g, nm.f("mtp.post_proj"), { nm.ckpt("draft.post_projection.weight") },
                                 n_trunk, n, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                 b_hd, b_hn));

    rad_note(b, "the assistant head is declared: %lld layer(s) of width %lld at absolute %lld..%lld, "
                "reading the trunk's cache at layer %d (sliding) and %d (full); no cache of its own, "
                "every round at the committed position",
             (long long)n_l, (long long)n, (long long)base, (long long)(base + n_l - 1),
             last_sliding, last_full);
    return RAD_OK;
}

/* One pass, over the batch staged for a head with no cache of its own: one row per drafting
 * sequence, at the committed position, with the context ending there. */
void Gemma4Head::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    if (batch->draft_pass < 0) {
        /* Every row must be a drafting row, as rad_drafter_shared_kv requested; anything else
         * was built for a head with its own cache. */
        if (batch->n_draft_out != T || batch->n_out != T) {
            rad_step_fail(c, "the gemma4 assistant head was handed a history pass that is not one row "
                             "a drafting sequence; it keeps no cache of its own and drafts from that "
                             "shape only (rad_drafter_shared_kv)");
            return;
        }
        RAD_ISSUE_N(c, op_hsrc, T, RAD_B(src), praw(batch->out_ids, RAD_I32, T),
                    bcol(b_c, n_trunk, n_trunk, T));
    } else {
        RAD_ISSUE_N(c, op_hcopy, T, brows(b_hn, T), bcol(b_c, n_trunk, n_trunk, T));
    }
    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(b_e, T));
    if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * n_trunk, brows(b_e, T), RAD_NONE);
    RAD_ISSUE_N(c, op_ecopy, T, brows(b_e, T), bcol(b_c, 0, n_trunk, T));
    ln_pre.step(c, b_c, b_x, T);

    for (const HLayer& L : layers) {
        const RadKVGroupBatch* kvb = kv_batch(batch, L.kv);
        RAD_ISSUE(c, L.op_norm, brows(b_x, T), RAD_W(L.w_norm), brows(b_h, T));
        L.ln_q.step(c, b_h, L.q, T);
        RAD_ISSUE_N(c, L.op_qn, T * L.cls.n_head, brows(L.q, T), RAD_W(L.w_qn), brows(L.q, T));
        RAD_ISSUE(c, L.op_rq, brows(L.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE_N(c, L.op_attn, batch->max_q_len,
                    brows(L.q, T), kv_cache(L.kv, L.trunk_layer),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, batch->n_seq,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(L.attn, T));
        L.ln_o.step(c, L.attn, b_h, T);
        if (L.op_ar) RAD_ISSUE_N(c, L.op_ar, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_post, T, brows(b_h, T), RAD_W(L.w_post), brows(b_h, T));
        RAD_ISSUE(c, L.op_add, brows(b_x, T), brows(b_h, T), brows(b_x, T));

        RAD_ISSUE(c, L.op_fnorm, brows(b_x, T), RAD_W(L.w_fnorm), brows(b_h, T));
        L.ln_gu.step(c, b_h, b_gu, T);
        RAD_ISSUE(c, L.op_act, brows(b_gu, T), brows(b_ffn, T));
        L.ln_down.step(c, b_ffn, b_h, T);
        if (L.op_far) RAD_ISSUE_N(c, L.op_far, T * n, brows(b_h, T), RAD_NONE);
        RAD_ISSUE_N(c, L.op_fpost, T, brows(b_h, T), RAD_W(L.w_fpost), brows(b_h, T));
        RAD_ISSUE(c, L.op_fadd, brows(b_x, T), brows(b_h, T), brows(b_x, T));
        RAD_ISSUE(c, L.op_scale, brows(b_x, T), RAD_W(L.w_scalar), brows(b_x, T));
    }

    RAD_ISSUE(c, op_norm, brows(b_x, T), RAD_W(w_norm), brows(b_hd, T));
    RAD_ISSUE_N(c, op_logits, T, brows(b_hd, T), RAD_W(w_lm), brows(logits, T));
    if (op_cap) RAD_ISSUE_N(c, op_cap, T, brows(logits, T), brows(logits, T));
    ln_post.step(c, b_hd, b_hn, T);
}

struct Layer {
    bool       global = false;
    Gemma4Attn attn{};
    Gemma4Mlp  mlp{};
    rad_weight w_scalar = 0;   /* the per-layer [1] residual scalar, `cur = cur * scalar` */
    rad_op     op_scale = 0;
    /* The fused junctions (see "Residual junctions" above): attention -> this layer's
     * feed-forward, and this layer's feed-forward -> the next layer's attention, scalar included.
     * The last layer has no j_ma: its stream goes to the final norm, which feeds the lm_head, not
     * a quantiser. */
    G4Junction j_am{}, j_ma{};
};

struct Model {
    Names      nm{""};
    Geom       g{};
    ClassGeom  cs{};   /* sliding class */
    ClassGeom  cg{};   /* global class  */
    int64_t    n_main = 0;

    rad_kvgroup kv_s = 0, kv_g = 0;

    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    rad_buf b_qs = 0, b_ks = 0, b_vs = 0, b_attns = 0;   /* sliding rows */
    rad_buf b_qg = 0, b_kg = 0, b_vg = 0, b_atting = 0;  /* global rows  */
    rad_buf b_gu = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out = 0, w_lm = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final = 0, op_gather = 0, op_logits = 0, op_cap = 0;

    std::vector<Layer> layers;

    /* The assistant head, declared only when the container carries one and the deployment
     * speculates (`max_spec` > 0). */
    Gemma4Head head{};
    bool       have_head = false;

    /* The vision tower (rad_block_gemma4_vision.h), declared when the container carries one and
     * the deployment serves media; `op_mm` writes its rows over the placeholders' embeddings. */
    Gemma4Vision vis{};
    rad_op       op_mm = 0;

    G4Junction j0{};       /* the first block's entry: norm + quantiser of the embedding stream */
    bool       fuse = false;
    bool       fuse_ar = false;   /* every junction also carries the all-reduce in front of it */
};

static Model g_model[MAX_RANKS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: gemma4_bf16 refuses a checkpoint with %lld experts; that is not this arch\n",
                (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- geometry RadModelMeta does not name. */
    const int64_t kv_g_all    = meta_text_geti(meta, "num_global_key_value_heads", 0);
    const int64_t head_g      = meta_text_geti(meta, "global_head_dim", 0);
    const int64_t sliding_win = meta_text_geti(meta, "sliding_window", 0);
    const int64_t kv_shared   = meta_text_geti(meta, "num_kv_shared_layers", 0);
    const int64_t interval    = meta_text_geti(meta, "full_attention_interval", 0);
    /* Whether the global layers read V from K (attention_k_eq_v; the 31B's do, and have no
     * v_proj). A checkpoint that does not say is refused rather than guessed. */
    const long long k_eq_v    = meta_text_geti(meta, "attention_k_eq_v", -1);
    if (k_eq_v != 0 && k_eq_v != 1) {
        fprintf(stderr, "radiance: gemma4: the config does not say whether the global layers read "
                        "V from K (attention_k_eq_v)\n");
        return RAD_E_INVAL;
    }
    const double  theta_s     = meta_text_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    const double  theta_g     = meta_text_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    const double  softcap     = meta_text_getf(meta, "final_logit_softcapping", 0.0);
    double        prf         = meta_text_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0);

    if (head_g <= 0 || kv_g_all <= 0) {
        fprintf(stderr, "radiance: gemma4: global_head_dim %lld / num_global_key_value_heads %lld; both required\n",
                (long long)head_g, (long long)kv_g_all);
        return RAD_E_INVAL;
    }
    if (head_g & 1) { fprintf(stderr, "radiance: gemma4: global head_dim %lld is odd\n", (long long)head_g); return RAD_E_INVAL; }
    if (!(theta_s > 0.0) || !(theta_g > 0.0)) {
        fprintf(stderr, "radiance: gemma4: both rope_parameters...rope_theta are required\n");
        return RAD_E_INVAL;
    }
    if (!(softcap > 0.0)) { fprintf(stderr, "radiance: gemma4: final_logit_softcapping is required\n"); return RAD_E_INVAL; }
    if (kv_shared < 0) { fprintf(stderr, "radiance: gemma4: num_kv_shared_layers %lld is negative\n", (long long)kv_shared); return RAD_E_INVAL; }
    if (kv_shared > 0) {
        fprintf(stderr, "radiance: gemma4: num_kv_shared_layers %lld needs a shared-KV reuse path; refusing\n",
                (long long)kv_shared);
        return RAD_E_UNSUPPORTED;
    }
    if (sliding_win <= 0) { fprintf(stderr, "radiance: gemma4: sliding_window %lld is required\n", (long long)sliding_win); return RAD_E_INVAL; }
    if (!(prf > 0.0)) prf = 1.0;
    const int64_t n_rot_g = (int64_t)(head_g * prf + 0.5);
    if (n_rot_g <= 0 || n_rot_g > head_g || (n_rot_g & 1)) {
        fprintf(stderr, "radiance: gemma4: global rotary width %lld out of range for head_dim %lld\n",
                (long long)n_rot_g, (long long)head_g);
        return RAD_E_INVAL;
    }

    int64_t kv_g = 0;
    if (divide_or_fail(kv_g_all, g.world, "num_global_key_value_heads", &kv_g) < 0) return RAD_E_INVAL;

    /* Plain RMSNorm, x_hat * w, on every norm including the post-projection ones. Unlike
     * Gemma 2/3, Gemma 4 does not use the (1 + w) form: vLLM's stock RMSNorm, HF's Gemma4RMSNorm
     * (weight init 1.0) and llama.cpp's LLM_NORM_RMS all multiply by the weight alone. */
    g.wadd    = 0.0f;
    g.wadd_qk = 0.0f;
    const float final_wadd = 0.0f;

    m.n_main = g.n_layer;
    std::vector<int> is_global;
    RAD_ARCH_TRY(g4_schedule(meta, g.n_layer, interval, is_global));
    int n_global = 0;
    for (int v : is_global) n_global += v;

    /* The sliding class comes from geom_from (head_dim 256, 16 kv heads, full rotation, theta_s);
     * the global class is built from the container header. */
    m.cs.n_head = g.n_head;      m.cs.n_head_kv = g.n_head_kv; m.cs.head_dim = g.head_dim;
    m.cs.n_rot  = g.head_dim;    m.cs.window = sliding_win;    m.cs.theta = theta_s;
    m.cs.scale  = g.rope_scale;  m.cs.head_ladder = false;     m.cs.has_v = true;
    m.cg.n_head = g.n_head;      m.cg.n_head_kv = kv_g;        m.cg.head_dim = head_g;
    m.cg.n_rot  = n_rot_g;       m.cg.window = 0;              m.cg.theta = theta_g;
    m.cg.scale  = g.rope_scale;  m.cg.head_ladder = true;      m.cg.has_v = k_eq_v == 0;

    /* ---- the two KV groups. */
    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_WINDOW;  sd.dtype = kvdt;  sd.window = sliding_win;
    sd.n_head_kv = g.n_head_kv;  sd.head_dim = g.head_dim;
    m.kv_s = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    RadKVGroupDecl gd{};
    gd.kind = RAD_KV_FULL;    gd.dtype = kvdt;  gd.window = 0;
    gd.n_head_kv = kv_g;      gd.head_dim = head_g;
    m.kv_g = rad_decl_kv_group(b, nm.f("kv_global"), &gd);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (m.kv_s == 0 || m.kv_g == 0) { fprintf(stderr, "radiance: gemma4: a KV group failed to declare\n"); return RAD_E_INVAL; }

    /* ---- buffers. The two classes have different row widths, so they keep their own rows. */
    m.b_x  = decl_b(b, nm.f("x"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h  = decl_b(b, nm.f("h"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_qs    = decl_b(b, nm.f("attn_qs"),   g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_ks    = decl_b(b, nm.f("attn_ks"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_vs    = decl_b(b, nm.f("attn_vs"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_attns = decl_b(b, nm.f("attn_outs"), g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_qg    = decl_b(b, nm.f("attn_qg"),    g.act_dtype, {g.max_tok, m.cg.q_dim()});
    m.b_kg    = decl_b(b, nm.f("attn_kg"),    g.act_dtype, {g.max_tok, m.cg.kv_dim()});
    m.b_vg    = decl_b(b, nm.f("attn_vg"),    g.act_dtype, {g.max_tok, m.cg.kv_dim()});
    m.b_atting= decl_b(b, nm.f("attn_outg"),  g.act_dtype, {g.max_tok, m.cg.q_dim()});
    m.b_gu   = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn  = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    m.b_hout = decl_b(b, nm.f("h_out"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges. Gemma 4 ties the head to the embedding. */
    const char* pre = "model.language_model.layers";
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"), nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    /* bf16, not f32: the model group is pinned to the mapped tier (token_embd's embed_lookup is
     * host-only), and a widened output_norm could not be served from the mapping. */
    m.w_out = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* Declared at its codes' dtype, like the assistant head's lm_head: bf16 as shipped, or int8 /
     * block fp8 where a recipe (`--quant output.weight=...`) makes it so, selecting the matching
     * logits_gemm kernel. It is the trunk's largest single read at decode (n_vocab x n_embd per
     * rank). */
    m.w_lm  = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                     {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    /* The embedding is scaled by sqrt(n_embd) at the lookup, not in the table (as in llama.cpp's
     * build_inp_embd and HF's ScaledWordEmbedding), so the plain table stays tied to the head.
     * Without it the token signal in the residual stream would be ~73x too small. */
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", std::sqrt((double)g.n_embd))),
                    RAD_WEIGHTS(m.w_tok)),
                   {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_x});
    /* An image's rows overwrite its placeholders' embeddings after the (all-reduced) lookup and
     * before the first block: the reference scales token embeddings by sqrt(n_embd) and scatters
     * the image features in unscaled. */
    if (Gemma4Vision::present(meta) && ctx->max_enc_patches > 0)
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});

    /* ---- the layers. */
    Gemma4Attn::Wire sw{}; sw.x = m.b_x; sw.h = m.b_h; sw.q = m.b_qs; sw.k = m.b_ks; sw.v = m.b_vs;        sw.attn = m.b_attns;
    Gemma4Attn::Wire gw{}; gw.x = m.b_x; gw.h = m.b_h; gw.q = m.b_qg; gw.k = m.b_kg; gw.v = m.b_vg;        gw.attn = m.b_atting;
    Gemma4Mlp::Wire  mw{}; mw.x = m.b_x; mw.h = m.b_h; mw.gu = m.b_gu; mw.ffn = m.b_ffn;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.global = (is_global[(size_t)l] == 1);
        const ClassGeom&        cgeo = lay.global ? m.cg : m.cs;
        const rad_kvgroup       kg   = lay.global ? m.kv_g : m.kv_s;
        const Gemma4Attn::Wire& aw   = lay.global ? gw : sw;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kg));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, cgeo, kg, aw, pre));
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, pre));

        /* The per-layer scalar: `cur = cur * scalar`, a [1] bf16 over the residual stream. */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.layer_scalar", (int)l),
                              nm.ckpt("%s.%lld.layer_scalar", pre, (long long)l)));
        lay.w_scalar = decl_w(b, nm.f("blk.%d.layer_scalar", (int)l), RAD_BF16, {1},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer((int)l));
        lay.op_scale = rw(b, RAD_OP(b, "scale",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_scalar)),
                           {mw.x}, {mw.x});
    }

    /* ---- the junctions, declared once every layer exists (j_ma reads the next layer's gain).
     * Used only when all resolved and every linear a junction feeds reads its int8 pair (q, k and
     * v through q's, and gate_up): a junction writes that pair, not the bf16 norm output, so a bf16
     * k or v beside an int8 q would read a stale buffer. Otherwise the blocks keep their own norms,
     * quantisers, post-norms and adds. */
    if (m.n_main > 0) {
        bool ok = true;
        for (const Layer& lay : m.layers)
            ok = ok && lay.attn.ln_q.quant && lay.attn.ln_k.shared_in &&
                 (!lay.attn.cls.has_v || lay.attn.ln_v.shared_in) && lay.mlp.ln_gu.quant;
        if (ok) {
            RAD_ARCH_TRY(m.j0.declare(b, g, m.b_x, 0, m.layers[0].attn.w_norm, 0, 0, m.layers[0].attn.ln_q));
            ok = m.j0.op != 0;
            for (int64_t l = 0; ok && l < m.n_main; ++l) {
                Layer& lay = m.layers[(size_t)l];
                RAD_ARCH_TRY(lay.j_am.declare(b, g, m.b_h, m.b_x, lay.mlp.w_norm, lay.attn.w_post, 0,
                                              lay.mlp.ln_gu));
                ok = lay.j_am.op != 0;
                if (ok && l + 1 < m.n_main) {
                    const Layer& nx = m.layers[(size_t)l + 1];
                    RAD_ARCH_TRY(lay.j_ma.declare(b, g, m.b_h, m.b_x, nx.attn.w_norm, lay.mlp.w_post,
                                                  lay.w_scalar, nx.attn.ln_q));
                    ok = lay.j_ma.op != 0;
                }
            }
        }
        m.fuse = ok;
        bool ar = ok && g.world == 2;
        for (int64_t l = 0; ar && l < m.n_main; ++l) {
            const Layer& lay = m.layers[(size_t)l];
            ar = lay.j_am.op_ar != 0 && (l + 1 >= m.n_main || lay.j_ma.op_ar != 0);
        }
        m.fuse_ar = ar;
    }

    /* ---- the tail: final norm, the sampled rows, the lm_head, and the soft-cap. */
    m.op_final = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(m.w_out)),
                     {m.b_x}, {m.b_h});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(m.w_lm)),
                     {m.b_hout}, {m.b_logits});
    m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_vocab),
                                 RAD_F64("cap", softcap), RAD_STR("dtype", "f32")),
                      RAD_NOWEIGHTS),
                     {m.b_logits}, {m.b_logits});

    /* ---- the assistant head, when the container carries one (`rad-convert --draft-model`) and
     * the deployment speculates; otherwise its ~0.5 GB of weights stay off the card. */
    if (ctx->max_spec > 0 && has_assistant(meta)) {
        int last_s = -1, last_f = -1;
        for (int64_t l = 0; l < m.n_main; ++l) (is_global[(size_t)l] ? last_f : last_s) = (int)l;
        RAD_ARCH_TRY(m.head.declare(b, nm, meta, g, m.cs, m.cg, m.kv_s, m.kv_g, last_s, last_f,
                                    m.n_main, m.w_tok, m.b_h, m.b_logits));
        m.have_head = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal,
         * through the ordinary sampler chain (proposal 0). The head has no cache of its own, so
         * the core stages every pass at the committed position. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = 0;
        dd.proposal_pitch = 1;
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        RAD_ARCH_TRY(rad_drafter_shared_kv(b, kHeadPosOffset));
    }

    /* ---- the vision tower, its weight groups past the trunk's and the head's (counted even when
     * the head is not declared, so the groups do not move with max_spec). */
    {
        const int64_t head_groups = has_assistant(meta) ? meta_text_geti(meta, "num_hidden_layers", 0, "draft.") + 1 : 0;
        RAD_ARCH_TRY(m.vis.declare(b, meta, ctx, nm, g.n_embd, (int)(m.n_main + head_groups),
                                   "model.", "gemma4_bf16"));
    }

    rad_note(b, "rank %d/%d: %lld layers (%lld global, %lld sliding), sliding window %lld, "
                "slider %lld heads x %lld, global %lld heads x %lld, soft-capped logits",
             g.rank, g.world, (long long)g.n_layer, (long long)n_global,
             (long long)(g.n_layer - n_global), (long long)sliding_win,
              (long long)g.n_head, (long long)g.head_dim, (long long)g.n_head, (long long)head_g);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* An encoder pass runs only the tower. */
    if (batch->enc) {
        m.vis.step(c, batch);
        return;
    }
    /* A draft pass runs only the head. The trunk already left its final-normed hidden state in
     * `h`, which the head's history pass gathers from. */
    if (batch->draft_pass != 0) {
        if (!m.have_head) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` assistant, or declare ran at max_spec 0");
            return;
        }
        m.head.step(c, batch);
        return;
    }
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, batch->n_tok));

    const int64_t T = batch->n_tok;
    if (m.fuse) m.j0.step(c, T);
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& l = m.layers[li];
        const bool last = li + 1 == m.layers.size();
        l.attn.step(c, batch, m.fuse, m.fuse, m.fuse_ar);
        if (m.fuse) l.j_am.step(c, T, m.fuse_ar);
        l.mlp.step(c, batch, m.fuse, m.fuse && !last, m.fuse_ar);
        if (m.fuse && !last) {
            /* [all-reduce,] post-norm, add, layer scalar, next block's norm and quantiser: one op */
            l.j_ma.step(c, T, m.fuse_ar);
            continue;
        }
        /* The per-layer scalar, `cur = cur * scalar` over the residual stream, applied after the
         * mlp add, as vLLM's `hidden_states = hidden_states * self.layer_scalar`. The checkpoint's
         * values (0.03-0.99) are trained weights that rescale the residual toward O(1). */
        RAD_ISSUE(c, l.op_scale, brows(m.b_x, batch->n_tok), RAD_W(l.w_scalar),
                  brows(m.b_x, batch->n_tok));
    }

    RAD_ISSUE(c, m.op_final,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out), brows(m.b_h, batch->n_tok));

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm),
                    brows(m.b_logits, batch->n_out));
        RAD_ISSUE_N(c, m.op_cap, batch->n_out,
                    brows(m.b_logits, batch->n_out),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The reply format: the model's, shared with arch/gemma4 (arch/rad_gemma4_chat.h). */
const RadChatFormat* chat_format(const RadModelMeta*) { return gemma4_chat_format(); }

/* The default draft depth for `--num-speculative-tokens auto`. A serial head's depth is an
 * operating point, not a trained depth: every depth runs. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (has_assistant(meta)) {
        out->draft_depth = kHeadDepth;
        out->draft_depth_fixed = 0;
    }
    /* The tower's pooling window, so a pass size the core picks itself is whole windows. */
    if (Gemma4Vision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.pooling_kernel_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    return RAD_OK;
}

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(gemma4_bf16, "gemma4", "compressed-tensors", "0.1.0",
                "Gemma 4 dense (text and images), w4a8 from its bf16 or its CT-INT4 (QAT) release: "
                "sliding-window and proportional-RoPE global attention over a plain RMSNorm stream "
                "with GELU feed-forward, per-layer scalar scaling and soft-capped logits")
RAD_ARCH_CHAT_FORMAT(gemma4_bf16)
RAD_ARCH_PROBE(gemma4_bf16)
#endif

}  /* namespace gemma4_bf16 */
