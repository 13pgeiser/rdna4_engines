/* gemma3.cpp -- Gemma 3 dense text trunk (TranslateGemma 27B), bf16 or w4a8.
 *
 * Built from gemma4_bf16.cpp: the same sandwich-normed blocks over one residual stream, the same
 * GELU feed-forward, the same w4a8 linears and fused residual junctions. What differs, from HF's
 * modeling_gemma3.py (transformers 4.57):
 *
 *   - every norm is GemmaRMSNorm, x_hat * (1 + w), the per-head q/k norms and the final norm too;
 *   - one head geometry for both attention classes (head_dim 128), each with its own v projection
 *     and no value norm;
 *   - attention scores are scaled by query_pre_attn_scalar^-0.5 (1/sqrt(168) on the 27B), not by
 *     1/sqrt(head_dim);
 *   - sliding layers rotate at rope_local_base_freq (10k) unscaled; global layers at rope_theta
 *     (1M) with the config's rotary scaling (linear, factor 8), which HF applies to them only;
 *   - the embedding is scaled by sqrt(hidden) held in the weight dtype (HF's embed_scale buffer
 *     cast to bf16: 73.5, not 73.32, on the 27B);
 *   - no layer scalar, no logit soft-capping (both softcaps are null in the 27B's config), no KV
 *     sharing, no partial rotary.
 *
 * Text only: a checkpoint's vision tower and projector are not mapped and so never converted.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_block_dflash2.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace gemma3 {

using namespace rad::arch;

/* Where the language model's tensors are in a Gemma3ForConditionalGeneration checkpoint as
 * transformers 4.57 saves it (TranslateGemma). */
static const char* const kPre    = "language_model.model.";
static const char* const kLayers = "language_model.model.layers";

/* layer_types names every layer "full_attention" or "sliding_attention"; without it, every
 * `_sliding_window_pattern`-th layer is global (HF's legacy rule: (l + 1) % pattern == 0). */
static int g3_schedule(const RadModelMeta* m, int64_t n, std::vector<int>& is_global) {
    const char* lt = meta_text_gets(m, "layer_types", nullptr);
    if (lt && *lt) {
        RAD_ARCH_TRY(parse_layer_kinds("gemma3", lt, {"full_attention"}, "sliding_attention",
                                       &is_global));
        if ((int64_t)is_global.size() != n) {
            fprintf(stderr, "radiance: gemma3: layer_types names %zu layers and the model has %lld\n",
                    is_global.size(), (long long)n);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }
    const int64_t pattern = meta_text_geti(m, "_sliding_window_pattern",
                                           meta_text_geti(m, "sliding_window_pattern", 0));
    if (pattern <= 0) {
        fprintf(stderr, "radiance: gemma3: no layer_types and no sliding_window_pattern\n");
        return RAD_E_INVAL;
    }
    is_global.resize((size_t)n);
    for (int64_t l = 0; l < n; ++l) is_global[(size_t)l] = is_full_attention(l, pattern);
    return RAD_OK;
}

struct ClassGeom {
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0;
    int64_t window = 0;
    double  theta  = 1.0;
    float   scale  = 1.0f;   /* rotary position divisor (linear scaling) */
    int64_t q_dim()  const { return n_head * head_dim; }
    int64_t kv_dim() const { return n_head_kv * head_dim; }
    int64_t gqa()    const { return n_head / n_head_kv; }
};

/* ---- one linear: gemma4_bf16's G4Linear (plain bf16, or w4a8/w8a8 over a per-row int8
 * activation), plus two formats whose activation scale is per 128 columns, not per row.
 *
 * Gemma 3's activations carry a few channels hundreds of times the rest, and one int8 scale a row
 * spends its codes on them: on TranslateGemma 27B w8a8 alone measured KL 0.097 against bf16
 * where int8 weights alone cost 0.003. So:
 *   - `i8g`: int8 codes with a bf16 scale per [1 x 128] (libquant `codes=i8 group=128
 *     scale=bf16`), gemm_nt_q at i8a8 over quant_act_i8g's codes, an f32 scale per (row, 128);
 *   - `mx`: OCP-MXFP4 (e2m1 codes, an E8M0 exponent per [1 x 32]), gemm_nt_q at mxfp4a8 over
 *     quant_act_fp8's E4M3 codes, an f32 scale per (row, 128); the exponent plane is passed twice,
 *     as b_scale and as b_ref (arch/common/rad_fp8.h LinearFP8 says why).
 * Neither has a fused norm/quantiser here, so the junctions stay off for them. */
struct G3Linear {
    rad_weight w = 0, ws = 0, wref = 0;
    rad_op     op = 0, op_q = 0;
    rad_buf    q8 = 0, s8 = 0;   /* the activation's code and scale planes, whatever the format */
    int64_t    N = 0, K = 0;
    bool       quant = false;    /* w4a8 / w8a8 over a per-row int8 activation */
    bool       i8g = false;
    bool       mx = false;
    bool       a4 = false;       /* asymmetric int4 over the per-(row, 128) int8 activation */
    bool       shared_in = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {},
                 const G3Linear* in_from = nullptr);
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
    void gemm(RadCtx* c, rad_buf y, int64_t T) const;
};

inline int G3Linear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                             std::initializer_list<const char*> src,
                             int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                             rad_buf a, rad_buf y,
                             std::initializer_list<int64_t> row_parts,
                             const G3Linear* in_from) {
    N = n_out; K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1) RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else                RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
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

    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
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

    i8g = known && linear_is_w8(enc);
    mx  = known && linear_is_mxfp4(enc);
    /* Asymmetric int4 (libquant `codes=u4 zero=u8 group=128 scale=f16`), unrotated: gemm_nt_q at
     * w4a8i8_asym over the same per-(row, 128) int8 activation as i8g. LinearFP8 serves this grid
     * only under ParoQuant's rotation, which a tensor-parallel shard cannot cut; unrotated it shards
     * like any row- or column-split weight. */
    {
        const RadEncPlane* zp = rad_enc_plane(&enc, "zero");
        a4 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 3 && !enc.transform[0] &&
             enc.plane[0].dtype == RAD_U4 && enc.plane[0].block[1] == 1 && sc &&
             sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK && zp &&
             zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK;
    }
    if (a4) {
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an asymmetric int4 weight takes "
                            "rows in 16s and K in whole 128s\n", base, (long long)N, (long long)K);
            return RAD_E_INVAL;
        }
        w  = decl_view(b, wn, nullptr, "codes", RAD_U4, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                       RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        if (!w || !ws) return RAD_E_INVAL;
        if (in_from && (in_from->i8g || in_from->a4) && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8, { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, K / RAD_FP8_BLOCK });
            if (!q8 || !s8) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
            if (!op_q) return RAD_E_INVAL;
        }
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "w4a8i8_asym")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    if (i8g || mx) {
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; it takes rows in 16s and K in "
                            "whole %d-column activation scale groups\n", base, (long long)N,
                    (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        if (mx) {
            w    = decl_view(b, wn, nullptr, "codes", RAD_FP4E2M1, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            ws   = decl_view(b, sn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            wref = decl_view(b, nm.f("%s.ref", base), wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                             RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            if (!w || !ws || !wref) return RAD_E_INVAL;
        } else {
            w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            ws = decl_view(b, sn, wn, "scale", RAD_BF16, { N, K / RAD_FP8_BLOCK }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            if (!w || !ws) return RAD_E_INVAL;
        }
        if (in_from && in_from->mx == mx && (in_from->i8g || in_from->a4) == i8g && in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), mx ? RAD_F8E4M3 : RAD_I8, { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, K / RAD_FP8_BLOCK });
            if (!q8 || !s8) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, mx ? "quant_act_fp8" : "quant_act_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
            if (!op_q) return RAD_E_INVAL;
        }
        if (mx)
            op = rw(b, RAD_OP(b, "gemm_nt_q",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "mxfp4a8")),
                        RAD_WEIGHTS(w, ws, wref)),
                    { q8, s8 }, { y });
        else
            op = rw(b, RAD_OP(b, "gemm_nt_q",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "i8a8")),
                        RAD_WEIGHTS(w, ws)),
                    { q8, s8 }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, w4a8, w8a8, "
                    "int8 a 128 (i8a8) or MXFP4 -- none matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void G3Linear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        gemm(c, y, T);
    } else if (i8g || mx || a4) {
        /* quant_act_i8g / quant_act_fp8 take x, q, scale: no optional `asum`. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T));
        gemm(c, y, T);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
}

inline void G3Linear::gemm(RadCtx* c, rad_buf y, int64_t T) const {
    RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                brows(y, T), RAD_NONE, mx ? RAD_W(wref) : RAD_NONE);
}

/* ---- the GPTQ calibration tap (rad_arch.h's calib_*): the Gram of one linear's bf16 input,
 * summed over the corpus in a PERSIST [K, K] f32 buffer. Declared only with RADIANCE_CALIB_DIR set,
 * for the layers RADIANCE_G3_CALIB_LAYERS names ("lo:hi", default every layer): a down projection's
 * Gram is 10752^2 f32 a rank at TP2, so 62 layers of four taps do not fit beside the weights and a
 * calibration runs in slices of layers, the corpus pushed through once per slice. */
struct G3Tap {
    rad_buf h = 0;
    rad_op  op = 0;
    int64_t K = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* name, rad_buf a, int64_t k) {
        K = k;
        h = decl_b(b, nm.f("%s.gram", name), RAD_F32, { k, k }, RAD_BUF_PERSIST);
        if (!h) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gram_accum",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", k),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                    RAD_NOWEIGHTS),
                { a, h }, { h });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    /* gram_accum takes a, a_scale?, h: no scale for a bf16 input. */
    void step(RadCtx* c, rad_buf a, int64_t T, const bool* live) const {
        if (op && live && *live) RAD_ISSUE_N(c, op, T, brows(a, T), RAD_NONE, brows(h, K));
    }
};

/* ---- one attention block, parameterised by class. */
struct Gemma3Attn {
    struct Wire { rad_buf x = 0, h = 0, q = 0, k = 0, v = 0, attn = 0; };

    Wire      w{};
    Geom      g{};
    ClassGeom cls{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    double      attn_scale = 1.0;

    rad_weight w_norm = 0, w_qn = 0, w_kn = 0, w_post = 0;
    G3Linear   ln_q{}, ln_k{}, ln_v{}, ln_o{};
    rad_op op_norm = 0, op_qn = 0, op_kn = 0,
           op_rq = 0, op_rk = 0, op_attn = 0, op_kv = 0, op_post = 0, op_ar = 0, op_add = 0;
    /* the q/k norms and both rotaries as one op (libr4d `qkv_norm_rope` at v_norm 0), where it
     * resolves; byte-identical to the four */
    rad_op op_qnr = 0;
    /* Calibration: q|k|v's shared input and o_proj's (this rank's columns). */
    G3Tap       tap_in{}, tap_o{};
    const bool* tap_live = nullptr;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                 rad_kvgroup group, const Wire& wire, double scale);
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
    /* The branch's post-norm alone, for a caller fusing the add and the next norm after it. */
    void post(RadCtx* c, int64_t T) const {
        RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    }
};

int Gemma3Attn::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                        rad_kvgroup group, const Wire& wire, double scale) {
    g = geom; cls = cgeo; layer = l; kv = group; w = wire; attn_scale = scale;
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();
    const char* pre = kLayers;

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),      nm.ckpt("%s.%lld.input_layernorm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.q_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.k_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post_norm.weight", l), nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qn = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_kn = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.attn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    RAD_ARCH_TRY(ln_q.declare(b, nm, g, nm.f("blk.%d.attn_q", l),
                              { nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l) },
                              qd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.q));
    const G3Linear* qin = &ln_q;   /* k and v read q's int8 pair */
    RAD_ARCH_TRY(ln_k.declare(b, nm, g, nm.f("blk.%d.attn_k", l),
                              { nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l) },
                              kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.k, {}, qin));
    RAD_ARCH_TRY(ln_v.declare(b, nm, g, nm.f("blk.%d.attn_v", l),
                              { nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l) },
                              kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.v, {}, qin));
    RAD_ARCH_TRY(ln_o.declare(b, nm, g, nm.f("blk.%d.attn_o", l),
                              { nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l) },
                              g.n_embd, qd, RAD_SHARD_COL, grp_layer(l), w.attn, w.h));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* Per-head q and k norms, one row per head. Gemma 3 has no value norm. */
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

    /* NeoX rotary over the full head. K is rotated as a Q-only form over its own heads (see
     * gemma4_bf16: the rope kernel requires n_head > 0). `scale` divides the position: HF's
     * linear scaling divides inv_freq by the factor, which is the same angle. */
    op_rq = rw(b, RAD_OP(b, "rope",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                            RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", 0),
                            RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                            RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.head_dim)),
                 RAD_NOWEIGHTS),
                {w.q}, {w.q});
    op_rk = rw(b, RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                             RAD_INT("n_head", cls.n_head_kv), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.head_dim)),
                  RAD_NOWEIGHTS),
                 {w.k}, {w.k});

    if (rad_op_declared(b, "qkv_norm_rope")) {
        op_qnr = RAD_OP(b, "qkv_norm_rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                                 RAD_INT("n_head", cls.n_head), RAD_INT("n_head_kv", cls.n_head_kv),
                                 RAD_F64("theta", cls.theta), RAD_F64("eps", g.eps),
                                 RAD_F64("scale", cls.scale), RAD_STR("mode", "neox"),
                                 RAD_INT("rotary_dim", cls.head_dim), RAD_F64("wadd", g.wadd_qk),
                                 RAD_INT("v_norm", 0), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_qn, w_kn));
        if (op_qnr) op_qnr = rw(b, op_qnr, {w.q, w.k}, {w.q, w.k});
    }

    op_attn = rw(b, RAD_OP(b, "attn_paged",
                   RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                              RAD_INT("gqa", cls.gqa()), RAD_INT("causal", 1), RAD_INT("window", cls.window),
                              RAD_INT("n_head", cls.n_head), RAD_INT("block_size", RAD_KV_BLOCK),
                              RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                              RAD_F64("scale", attn_scale),
                              RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                   RAD_NOWEIGHTS),
                  {w.q}, {w.attn});

    op_kv = rw(b, RAD_OP(b, "kv_store",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                           RAD_INT("n_head_kv", cls.n_head_kv),
                           RAD_INT("block_size", rad_kv_block_size(b, kv)),
                           RAD_STR("kv_dtype", g.kv_dtype)),
                RAD_NOWEIGHTS),
                {w.k, w.v}, {});

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

void Gemma3Attn::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                      bool ar_fused) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;

    if (entry_done) {
        ln_q.gemm(c, w.q, T);
        ln_k.gemm(c, w.k, T);
        ln_v.gemm(c, w.v, T);
    } else {
        RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
        ln_q.step(c, w.h, w.q, T);
        ln_k.step(c, w.h, w.k, T);
        ln_v.step(c, w.h, w.v, T);
        tap_in.step(c, w.h, T, tap_live);
    }

    if (op_qnr) {
        /* v is an operand the op leaves alone at v_norm 0. */
        RAD_ISSUE(c, op_qnr, brows(w.q, T), brows(w.k, T), brows(w.v, T), rope_posmc(g, batch, T),
                  RAD_W(w_qn), RAD_W(w_kn));
    } else {
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

    tap_o.step(c, w.attn, T, tap_live);
    ln_o.step(c, w.attn, w.h, T);
    /* All-reduce before the post-norm: o_proj is column-sharded, and the norm is of the sum. */
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,  brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

/* ---- the feed-forward: GeGLU (gelu_pytorch_tanh), second norm on the down output. */
struct Gemma3Mlp {
    struct Wire { rad_buf x = 0, h = 0, gu = 0, ffn = 0; };

    Wire  w{};
    Geom  g{};
    int   layer = 0;
    rad_weight w_norm = 0, w_post = 0;
    G3Linear   ln_gu{}, ln_down{};
    rad_op op_norm = 0, op_act = 0, op_post = 0, op_ar = 0, op_add = 0;
    rad_op op_gq = 0;
    /* Calibration: gate|up's input and down's (this rank's columns). */
    G3Tap       tap_gu{}, tap_dn{};
    const bool* tap_live = nullptr;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire);
    void step(RadCtx* c, const RadBatch* batch, bool entry_done = false, bool tail_fused = false,
              bool ar_fused = false) const;
    void post(RadCtx* c, int64_t T) const {
        RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    }
};

int Gemma3Mlp::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire) {
    g = geom; layer = l; w = wire;
    const char* pre = kLayers;

    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_norm.weight", l), nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_post_norm.weight", l), nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.ffn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

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
    // Not under calibration: the tap reads the bf16 activation this fusion never writes.
    if (ln_down.quant && !ln_down.shared_in && !calib_dir() && rad_op_declared(b, "gelu_mul_quant_i8"))
        op_gq = rw(b, RAD_OP(b, "gelu_mul_quant_i8",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                   {w.gu}, {ln_down.q8, ln_down.s8});
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

void Gemma3Mlp::step(RadCtx* c, const RadBatch* batch, bool entry_done, bool tail_fused,
                     bool ar_fused) const {
    const int64_t T = batch->n_tok;
    if (entry_done) {
        ln_gu.gemm(c, w.gu, T);
    } else {
        RAD_ISSUE(c, op_norm,  brows(w.x, T),   RAD_W(w_norm),  brows(w.h, T));
        ln_gu.step(c, w.h, w.gu, T);
        tap_gu.step(c, w.h, T, tap_live);
    }
    if (op_gq) {
        RAD_ISSUE_N(c, op_gq, T, brows(w.gu, T), brows(ln_down.q8, T), brows(ln_down.s8, T), RAD_NONE);
        ln_down.gemm(c, w.h, T);
    } else {
        RAD_ISSUE(c, op_act,   brows(w.gu, T),  brows(w.ffn, T));
        tap_dn.step(c, w.ffn, T, tap_live);
        ln_down.step(c, w.ffn, w.h, T);
    }
    if (op_ar && !(tail_fused && ar_fused)) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (tail_fused) return;
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,   brows(w.x, T),   brows(w.h, T),  brows(w.x, T));
}

/* ---- one residual junction as one op (libr4d `rmsnorm_quant_i8`): [post-norm of the branch,
 * the add into the stream,] the next block's norm and its int8 quantiser. gemma4_bf16's
 * G4Junction without the layer scalar. */
struct G3Junction {
    rad_op     op = 0;
    rad_op     op_ar = 0;
    rad_buf    x = 0, res = 0, q8 = 0, s8 = 0;
    rad_weight w = 0, wp = 0;

    int declare(RadBuilder* b, const Geom& g, rad_buf x_, rad_buf res_, rad_weight w_,
                rad_weight wp_, const G3Linear& into) {
        x = x_; res = res_; w = w_; wp = wp_; q8 = into.q8; s8 = into.s8;
        if (!into.quant || into.shared_in || !q8 || !s8 || !w || (wp && !res))
            return RAD_E_INVAL;
        const RadParam p[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                               RAD_F64("wadd", g.wadd) };
        const rad_weight ws[2] = { w, wp };
        const int nw = wp ? 2 : 1;
        if (rad_op_declared(b, "rmsnorm_quant_i8"))
            op = rad_decl_op(b, "rmsnorm_quant_i8", p, (int)(sizeof p / sizeof p[0]), ws, nw);
        if (!op) return RAD_OK;
        const rad_buf rd[2] = { x, res };
        const rad_buf wr[3] = { q8, s8, res };
        rad_op_reads (b, op, rd, res ? 2 : 1);
        rad_op_writes(b, op, wr, res ? 3 : 2);
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
                    wp ? RAD_W(wp) : RAD_NONE, RAD_NONE);
    }
};

/* ---- the MXFP4 trunk's junction (libr4d `rmsnorm_quant_fp8`): the add of the branch `x` into
 * the stream `res`, the next block's norm and its E4M3 quantiser, in one op. It has no post-norm
 * slot (rmsnorm_quant_i8's w_post is Gemma 4's), so the branch's post-norm runs just before it:
 * four launches a junction become two. With `res` null it is the first block's norm of the
 * embedding stream. */
struct G3JunctionFP8 {
    rad_op     op = 0;
    /* At two ranks on the exact wire: the branch's all-reduce and post-norm folded in front
     * (ar_rmsnorm_post_quant_fp8), `x` then being this rank's partial sum. One launch. */
    rad_op     op_ar = 0;
    rad_buf    x = 0, res = 0, q = 0, s = 0;
    rad_weight w = 0, wp = 0;
    int64_t    n = 0;

    int declare(RadBuilder* b, const Geom& g, rad_buf x_, rad_buf res_, rad_weight w_,
                const G3Linear& into, rad_weight wp_ = 0) {
        x = x_; res = res_; w = w_; wp = wp_; q = into.q8; s = into.s8; n = g.n_embd;
        if (!into.mx || into.shared_in || !q || !s || !w) return RAD_E_INVAL;
        if (!rad_op_declared(b, "rmsnorm_quant_fp8")) return RAD_OK;
        op = RAD_OP(b, "rmsnorm_quant_fp8",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n), RAD_F64("eps", g.eps),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype),
                               RAD_F64("wadd", g.wadd)),
                    RAD_WEIGHTS(w));
        if (!op) return RAD_OK;
        const rad_buf rd[2] = { x, res };
        const rad_buf wr[3] = { q, s, res };
        rad_op_reads (b, op, rd, res ? 2 : 1);
        rad_op_writes(b, op, wr, res ? 3 : 2);
        if (res && wp && g.world == 2 && g.wire_exact && rad_op_declared(b, "ar_rmsnorm_post_quant_fp8")) {
            const rad_weight ws[2] = { w, wp };
            const RadParam pa[] = { RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n), RAD_F64("eps", g.eps),
                                    RAD_INT("group", RAD_FP8_BLOCK), RAD_INT("world_size", g.world),
                                    RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd) };
            op_ar = rad_decl_op(b, "ar_rmsnorm_post_quant_fp8", pa, (int)(sizeof pa / sizeof pa[0]), ws, 2);
            if (op_ar) {
                const rad_buf wa[4] = { x, q, s, res };
                rad_op_reads (b, op_ar, rd, 2);
                rad_op_writes(b, op_ar, wa, 4);
            }
        }
        return RAD_OK;
    }
    /* x, residual?, w -> q, scale, out_bf16?[, w_post]: the bf16 normed copy is not wanted. */
    void step(RadCtx* c, int64_t T, bool with_ar = false) const {
        if (with_ar)
            RAD_ISSUE_N(c, op_ar, T, brows(x, T), brows(res, T), RAD_W(w), brows(q, T), brows(s, T),
                        RAD_NONE, RAD_W(wp));
        else
            RAD_ISSUE_N(c, op, T, brows(x, T), res ? brows(res, T) : RAD_NONE, RAD_W(w),
                        brows(q, T), brows(s, T), RAD_NONE);
    }
};

struct Layer {
    bool       global = false;
    Gemma3Attn attn{};
    Gemma3Mlp  mlp{};
    G3Junction j_am{}, j_ma{};
    G3JunctionFP8 jf_am{}, jf_ma{};
};

struct Model {
    Names      nm{""};
    Geom       g{};
    ClassGeom  cs{};   /* sliding class */
    ClassGeom  cg{};   /* global class  */

    rad_kvgroup kv_s = 0, kv_g = 0;

    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    rad_buf b_q = 0, b_k = 0, b_v = 0, b_attn = 0;
    rad_buf b_gu = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out = 0, w_lm = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final = 0, op_gather = 0, op_logits = 0;

    std::vector<Layer> layers;

    G3Junction j0{};
    bool       fuse = false;
    G3JunctionFP8 jf0{};
    bool       fuse_mx = false;   /* the MXFP4 trunk's junctions (G3JunctionFP8) */
    bool       fuse_mx_ar = false;   /* ... each with the all-reduce and post-norm folded in */
    bool       fuse_ar = false;

    /* The DFlash2 drafter and the trunk taps it reads, only when the container carries one and
     * the deployment speculates (declare_dflash2). */
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    bool          have_df = false;
    enum { MAX_TAPS = 8 };
    int           n_taps = 0;
    int           tap_layer[MAX_TAPS] = {};
    rad_buf       b_dfsrc = 0;   /* [max_tok, n_taps * n_embd] */
    rad_op        op_tap = 0;

    /* The calibration tap: on while `tap_live`, over layers [calib_lo, calib_hi). */
    bool       calib = false, tap_live = false;
    int64_t    calib_lo = 0, calib_hi = 0;
    long long  calib_rows = 0;
};

static Model g_model[MAX_RANKS];

/* ================================================================== the DFlash2 drafter
 *
 * A second model at absolute layers n_layer.. with its own KV group (rad_block_dflash2.h), as
 * qwen35_fp8 declares it. What is Gemma's here:
 *   - the query block's embedding is scaled as the trunk's is (input_embedding_scale, sqrt(hidden)
 *     in bf16), which the block applies through embed_lookup's `mult`;
 *   - the head is tied: the 2-bit draft head is a copy of the embedding table.
 * The taps are the residual stream after the layers dflash_config.target_layer_ids names, which
 * NeMo's trainer captures as HF's hidden_states[i + 1]. */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer, const char* ck_embd) {
    Names& nm = m.nm;
    Geom   dg{};
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "gemma3 DFlash2"));

    Dflash2::Config dc{};
    dc.short_block = 1;   /* rad_drafter_short_block below */
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.sliding_window", 0);
    dc.escale   = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: gemma3: the DFlash2 drafter states no selector geometry\n");
        return RAD_E_INVAL;
    }
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: gemma3: DFlash2's query block has to be a power of two; this "
                        "serve asks for %d draft tokens (trained at %lld)\n", ctx->max_spec, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld and runs at %lld", trained,
                 (long long)dc.block);
    const double omul = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: gemma3: the drafter multiplies (%g) or soft-caps (%g) its logits, "
                        "which is not implemented\n", omul, scap);
        return RAD_E_UNSUPPORTED;
    }
    const double emult = rad_bf16_to_f32(rad_f32_to_bf16((float)std::sqrt((double)m.g.n_embd)));
    if (dc.escale != 1.0 && std::fabs(dc.escale - emult) > 1e-3)
        rad_note(b, "the drafter's input_embedding_scale %g is not the trunk's %g", dc.escale, emult);

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.b_dfsrc;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,    {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* Candidates are rescored against the trunk's head when logit_rerank can read its rows (fp8
     * or int8) and every candidate a rank proposes is one it owns; the planes are then kRerankR
     * wide and the selector reads the first sel_topk. */
    {
        const uint32_t hd = weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16);
        if ((hd == RAD_F8E4M3 || hd == RAD_I8) && (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            m.df.w_lm  = m.w_lm;
            m.df.top_w = Dflash2::kRerankR;
        }
    }
    const int64_t tw = m.df.w_lm ? m.df.top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("%s", ck_embd), dw));
    m.have_df = true;

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: gemma3: the DFlash2 checkpoint declares no mask_token_id\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (m.df.block_lo > 0) RAD_ARCH_TRY(rad_drafter_short_block(b, m.df.block_lo));
    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d.., %lld heads / %lld kv x %lld, ffn %lld, "
                "%s attention with a %lld-token window, %d trunk taps, embedding x %g, %s head",
             (long long)dg.n_layer, base_layer, (long long)dg.n_head, (long long)dg.n_head_kv,
             (long long)dg.head_dim, (long long)dg.n_ff, dc.causal ? "causal" : "non-causal",
             (long long)dc.window, m.n_taps, dc.escale, m.df.w_lm ? "reranked 2-bit" : "2-bit");
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    /* The rotary scaling (linear, factor 8 on the 27B) into g.rope_scale, for the global class. */
    RAD_ARCH_TRY(yarn_from(m.g, meta, "", m.g.head_dim));
    if (m.g.yarn.factor != 0.0f) {
        fprintf(stderr, "radiance: gemma3: YaRN rotary scaling is not what Gemma 3 uses; refusing\n");
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    const int64_t sliding_win = meta_text_geti(meta, "sliding_window", 0);
    const double  theta_s     = meta_text_getf(meta, "rope_local_base_freq", 0.0);
    const double  theta_g     = meta_text_getf(meta, "rope_theta", 0.0);
    const double  qpas        = meta_text_getf(meta, "query_pre_attn_scalar", 0.0);
    const double  cap_attn    = meta_text_getf(meta, "attn_logit_softcapping", 0.0);
    const double  cap_final   = meta_text_getf(meta, "final_logit_softcapping", 0.0);

    if (sliding_win <= 0) { fprintf(stderr, "radiance: gemma3: sliding_window %lld is required\n", (long long)sliding_win); return RAD_E_INVAL; }
    if (!(theta_s > 0.0) || !(theta_g > 0.0)) {
        fprintf(stderr, "radiance: gemma3: rope_local_base_freq and rope_theta are both required\n");
        return RAD_E_INVAL;
    }
    if (!(qpas > 0.0)) { fprintf(stderr, "radiance: gemma3: query_pre_attn_scalar is required\n"); return RAD_E_INVAL; }
    /* Gemma 2 soft-caps; Gemma 3 does not, and these kernels cannot. A config that asks is
     * refused rather than served without it. */
    if (cap_attn > 0.0 || cap_final > 0.0) {
        fprintf(stderr, "radiance: gemma3: logit soft-capping (attn %g, final %g) is not implemented\n",
                cap_attn, cap_final);
        return RAD_E_UNSUPPORTED;
    }

    /* GemmaRMSNorm everywhere: x_hat * (1 + w). */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    std::vector<int> is_global;
    RAD_ARCH_TRY(g3_schedule(meta, g.n_layer, is_global));
    int n_global = 0;
    for (int v : is_global) n_global += v;

    m.cs.n_head = g.n_head; m.cs.n_head_kv = g.n_head_kv; m.cs.head_dim = g.head_dim;
    m.cs.window = sliding_win; m.cs.theta = theta_s; m.cs.scale = 1.0f;
    m.cg = m.cs;
    m.cg.window = 0; m.cg.theta = theta_g; m.cg.scale = g.rope_scale;
    const double attn_scale = 1.0 / std::sqrt(qpas);

    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    RadKVGroupDecl sd{};
    sd.kind = RAD_KV_WINDOW;  sd.dtype = kvdt;  sd.window = sliding_win;
    sd.n_head_kv = g.n_head_kv;  sd.head_dim = g.head_dim;
    m.kv_s = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    RadKVGroupDecl gd{};
    gd.kind = RAD_KV_FULL;    gd.dtype = kvdt;  gd.window = 0;
    gd.n_head_kv = g.n_head_kv;  gd.head_dim = g.head_dim;
    m.kv_g = rad_decl_kv_group(b, nm.f("kv_global"), &gd);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (m.kv_s == 0 || m.kv_g == 0) { fprintf(stderr, "radiance: gemma3: a KV group failed to declare\n"); return RAD_E_INVAL; }

    /* One head geometry, so the two classes share their rows. */
    m.b_x    = decl_b(b, nm.f("x"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h    = decl_b(b, nm.f("h"), g.act_dtype, {g.max_tok, g.n_embd});
    m.b_q    = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_k    = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_v    = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, m.cs.kv_dim()});
    m.b_attn = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, m.cs.q_dim()});
    m.b_gu   = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn  = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    m.b_hout = decl_b(b, nm.f("h_out"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* The drafter's taps: one residual stream per tapped layer, copied into a column slice. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_layer_ids(rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                                       g.n_layer, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps <= 0) {
                fprintf(stderr, "radiance: gemma3: the DFlash drafter names no target_layer_ids\n");
                return RAD_E_INVAL;
            }
            m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype, {g.max_tok, (int64_t)m.n_taps * g.n_embd});
            m.op_tap = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_dfsrc});
        }
    }

    /* The vocabulary edges: the head is tied to the embedding. */
    char ck_embd[96], ck_norm[96];
    std::snprintf(ck_embd, sizeof ck_embd, "%sembed_tokens.weight", kPre);
    std::snprintf(ck_norm, sizeof ck_norm, "%snorm.weight", kPre);
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s", ck_embd)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s", ck_norm)));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"), nm.ckpt("%s", ck_embd)));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm  = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                     {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    /* HF scales the embedding by its embed_scale buffer cast to the weight dtype: sqrt(hidden)
     * rounded to bf16 (73.5 on the 27B). */
    const double emult = rad_bf16_to_f32(rad_f32_to_bf16((float)std::sqrt((double)g.n_embd)));
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off),
                               RAD_F64("mult", emult)),
                    RAD_WEIGHTS(m.w_tok)),
                   {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_x});

    Gemma3Attn::Wire aw{}; aw.x = m.b_x; aw.h = m.b_h; aw.q = m.b_q; aw.k = m.b_k; aw.v = m.b_v; aw.attn = m.b_attn;
    Gemma3Mlp::Wire  mw{}; mw.x = m.b_x; mw.h = m.b_h; mw.gu = m.b_gu; mw.ffn = m.b_ffn;

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.global = (is_global[(size_t)l] == 1);
        const ClassGeom&  cgeo = lay.global ? m.cg : m.cs;
        const rad_kvgroup kg   = lay.global ? m.kv_g : m.kv_s;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kg));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, cgeo, kg, aw, attn_scale));
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw));
    }

    /* The fused junctions, used only when every one resolves and every linear a junction feeds
     * reads its int8 pair (see gemma4_bf16). */
    // Neither junction family under calibration: the tap reads each linear's bf16 input, which
    // the fused junctions never write (an MXFP4 container wrote zero Grams for q/k/v and gate_up).
    if (g.n_layer > 0 && !calib_dir()) {
        bool ok = true;
        for (const Layer& lay : m.layers)
            ok = ok && lay.attn.ln_q.quant && lay.attn.ln_k.shared_in && lay.attn.ln_v.shared_in &&
                 lay.mlp.ln_gu.quant;
        if (ok) {
            RAD_ARCH_TRY(m.j0.declare(b, g, m.b_x, 0, m.layers[0].attn.w_norm, 0, m.layers[0].attn.ln_q));
            ok = m.j0.op != 0;
            for (int64_t l = 0; ok && l < g.n_layer; ++l) {
                Layer& lay = m.layers[(size_t)l];
                RAD_ARCH_TRY(lay.j_am.declare(b, g, m.b_h, m.b_x, lay.mlp.w_norm, lay.attn.w_post,
                                              lay.mlp.ln_gu));
                ok = lay.j_am.op != 0;
                if (ok && l + 1 < g.n_layer) {
                    const Layer& nx = m.layers[(size_t)l + 1];
                    RAD_ARCH_TRY(lay.j_ma.declare(b, g, m.b_h, m.b_x, nx.attn.w_norm, lay.mlp.w_post,
                                                  nx.attn.ln_q));
                    ok = lay.j_ma.op != 0;
                }
            }
        }
        m.fuse = ok;
        bool ar = ok && g.world == 2;
        for (int64_t l = 0; ar && l < g.n_layer; ++l) {
            const Layer& lay = m.layers[(size_t)l];
            ar = lay.j_am.op_ar != 0 && (l + 1 >= g.n_layer || lay.j_ma.op_ar != 0);
        }
        m.fuse_ar = ar;
    }

    /* ---- the MXFP4 trunk's junctions, when every block's inputs are E4M3 pairs it can write. */
    if (g.n_layer > 0 && !m.fuse && !calib_dir()) {
        bool ok = true;
        for (const Layer& lay : m.layers)
            ok = ok && lay.attn.ln_q.mx && !lay.attn.ln_q.shared_in && lay.attn.ln_k.shared_in &&
                 lay.attn.ln_v.shared_in && lay.mlp.ln_gu.mx && !lay.mlp.ln_gu.shared_in;
        if (ok) {
            RAD_ARCH_TRY(m.jf0.declare(b, g, m.b_x, 0, m.layers[0].attn.w_norm, m.layers[0].attn.ln_q));
            ok = m.jf0.op != 0;
            for (int64_t l = 0; ok && l < g.n_layer; ++l) {
                Layer& lay = m.layers[(size_t)l];
                RAD_ARCH_TRY(lay.jf_am.declare(b, g, m.b_h, m.b_x, lay.mlp.w_norm, lay.mlp.ln_gu,
                                               lay.attn.w_post));
                ok = lay.jf_am.op != 0;
                if (ok && l + 1 < g.n_layer) {
                    RAD_ARCH_TRY(lay.jf_ma.declare(b, g, m.b_h, m.b_x, m.layers[(size_t)l + 1].attn.w_norm,
                                                   m.layers[(size_t)l + 1].attn.ln_q, lay.mlp.w_post));
                    ok = lay.jf_ma.op != 0;
                }
            }
        }
        m.fuse_mx = ok;
        bool ar = ok && g.world == 2;
        for (int64_t l = 0; ar && l < g.n_layer; ++l) {
            const Layer& lay = m.layers[(size_t)l];
            ar = lay.jf_am.op_ar != 0 && (l + 1 >= g.n_layer || lay.jf_ma.op_ar != 0);
        }
        m.fuse_mx_ar = ar;
    }

    /* ---- the calibration tap. It reads the bf16 input of every linear, which the fused junctions
     * never write, so under it none is declared (above): any container calibrates, unfused. */
    if (calib_dir()) {
        m.calib_lo = 0;
        m.calib_hi = g.n_layer;
        if (const char* s = std::getenv("RADIANCE_G3_CALIB_LAYERS")) {
            long long lo = 0, hi = 0;
            int used = 0;
            if (std::sscanf(s, "%lld:%lld%n", &lo, &hi, &used) != 2 || s[used] != '\0' || lo < 0 ||
                hi > g.n_layer || lo >= hi) {
                fprintf(stderr, "radiance: gemma3: RADIANCE_G3_CALIB_LAYERS '%s' is not lo:hi within "
                                "0:%lld\n", s, (long long)g.n_layer);
                return RAD_E_INVAL;
            }
            m.calib_lo = lo;
            m.calib_hi = hi;
        }
        for (int64_t l = m.calib_lo; l < m.calib_hi; ++l) {
            Layer& lay = m.layers[(size_t)l];
            RAD_ARCH_TRY(lay.attn.tap_in.declare(b, nm, g, nm.f("blk.%d.attn_in", (int)l), m.b_h, g.n_embd));
            RAD_ARCH_TRY(lay.attn.tap_o.declare(b, nm, g, nm.f("blk.%d.attn_o", (int)l), m.b_attn, m.cs.q_dim()));
            RAD_ARCH_TRY(lay.mlp.tap_gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", (int)l), m.b_h, g.n_embd));
            RAD_ARCH_TRY(lay.mlp.tap_dn.declare(b, nm, g, nm.f("blk.%d.ffn_down", (int)l), m.b_ffn, g.n_ff));
        }
        m.calib = true;
        m.tap_live = true;
        for (Layer& lay : m.layers) { lay.attn.tap_live = &m.tap_live; lay.mlp.tap_live = &m.tap_live; }
    }

    m.op_final = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
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

    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)g.n_layer, ck_embd));

    rad_note(b, "rank %d/%d: %lld layers (%lld global, %lld sliding over %lld), %lld heads / %lld kv "
                "x %lld, attention scale 1/sqrt(%g), rotary %g / %g (global position / %g), "
                "embedding x %g%s",
             g.rank, g.world, (long long)g.n_layer, (long long)n_global,
             (long long)(g.n_layer - n_global), (long long)sliding_win, (long long)g.n_head,
             (long long)g.n_head_kv, (long long)g.head_dim, qpas, theta_s, theta_g,
             (double)g.rope_scale, emult,
             m.fuse ? ", fused junctions" :
             m.fuse_mx_ar ? ", fused fp8 junctions with the all-reduce" :
             m.fuse_mx ? ", fused fp8 junctions" : "");
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    if (batch->enc) {
        rad_step_fail(c, "gemma3 serves text only");
        return;
    }
    /* A drafter pass runs only the drafter: positive is the query pass proposing the whole block,
     * negative the context pass turning committed tokens into the drafter's own K and V. */
    if (batch->draft_pass != 0) {
        if (!m.have_df) {
            rad_step_fail(c, "the batch asks for a drafter pass and this gemma3 container declares none");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                      m.df.context(c, batch);
        return;
    }
    const int64_t T = batch->n_tok;
    /* The drafter conditions on a tapped layer's output, which the next layer overwrites: copy it
     * into the tap's column slice once the layer's residual is complete. */
    auto tap = [&](size_t li) {
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)li)
                RAD_ISSUE(c, m.op_tap, brows(m.b_x, T),
                          bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, T));
    };
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * m.g.n_embd, brows(m.b_x, T), RAD_NONE);

    if (m.fuse_mx) {
        /* MXFP4 trunk: [post-norm, then add + next norm + quantiser as one op] at every junction;
         * the last block's tail runs unfused into the final norm. */
        m.jf0.step(c, T);
        for (size_t li = 0; li < m.layers.size(); ++li) {
            const Layer& l = m.layers[li];
            const bool last = li + 1 == m.layers.size();
            const bool far = m.fuse_mx_ar;
            l.attn.step(c, batch, true, true, far);
            if (!far) l.attn.post(c, T);
            l.jf_am.step(c, T, far);
            l.mlp.step(c, batch, true, !last, far);
            if (!last) {
                if (!far) l.mlp.post(c, T);
                l.jf_ma.step(c, T, far);
            }
            tap(li);
        }
    }
    if (m.fuse) m.j0.step(c, T);
    for (size_t li = 0; !m.fuse_mx && li < m.layers.size(); ++li) {
        const Layer& l = m.layers[li];
        const bool last = li + 1 == m.layers.size();
        l.attn.step(c, batch, m.fuse, m.fuse, m.fuse_ar);
        if (m.fuse) l.j_am.step(c, T, m.fuse_ar);
        l.mlp.step(c, batch, m.fuse, m.fuse && !last, m.fuse_ar);
        if (m.fuse && !last) l.j_ma.step(c, T, m.fuse_ar);
        tap(li);
    }
    /* With fused junctions the last layer's tail ran unfused (tail_fused false), so the stream is
     * complete here either way. */
    RAD_ISSUE(c, m.op_final, brows(m.b_x, T), RAD_W(m.w_out), brows(m.b_h, T));

    /* The tap's rows, and its files once the corpus reached RADIANCE_CALIB_TOKENS. The row-split
     * linears' input is whole on every rank, so rank 0 writes it, once per linear that reads it.
     * The column-split ones (o, down) see this rank's columns only: each rank writes its block as
     * `<stem>.part`, which libquant's calib_sum assembles into the full K x K (block diagonal: the
     * cross-rank products are never formed on one card). */
    if (m.calib && m.tap_live) {
        m.calib_rows += T;
        if (m.calib_rows >= calib_target()) {
            const int rk = rad_rank(c);
            bool ok = true;
            for (int64_t l = m.calib_lo; l < m.calib_hi; ++l) {
                const Layer& lay = m.layers[(size_t)l];
                char stem[96];
                if (rk == 0) {
                    for (const char* which : { "attn_q", "attn_k", "attn_v" }) {
                        std::snprintf(stem, sizeof stem, "blk.%d.%s", (int)l, which);
                        ok = calib_write(c, lay.attn.tap_in.h, m.g.n_embd, m.calib_rows, stem, 0) && ok;
                    }
                    std::snprintf(stem, sizeof stem, "blk.%d.ffn_gate_up", (int)l);
                    ok = calib_write(c, lay.mlp.tap_gu.h, m.g.n_embd, m.calib_rows, stem, 0) && ok;
                }
                std::snprintf(stem, sizeof stem, "blk.%d.attn_o.part", (int)l);
                ok = calib_write(c, lay.attn.tap_o.h, lay.attn.tap_o.K, m.calib_rows, stem, 0) && ok;
                std::snprintf(stem, sizeof stem, "blk.%d.ffn_down.part", (int)l);
                ok = calib_write(c, lay.mlp.tap_dn.h, lay.mlp.tap_dn.K, m.calib_rows, stem, 0) && ok;
            }
            m.tap_live = false;
            std::fprintf(stderr, "radiance: gemma3: rank %d wrote the Hessians of layers %lld:%lld "
                                 "after %lld tokens to %s%s\n", rk, (long long)m.calib_lo,
                         (long long)m.calib_hi, m.calib_rows, calib_dir(), ok ? "" : ", with errors");
        }
    }

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The speculation depth this container's drafter supports, answered before declare: a DFlash2
 * drafter proposes its trained block in one pass (block_size 8: 7 drafts), as qwen35_fp8 answers;
 * 0 without a drafter. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash2")) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 8);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;   /* as qwen35_fp8: the trained block */
    }
    return RAD_OK;
}

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(gemma3)
RAD_ARCH_PLUGIN(gemma3, "gemma3", "", "0.1.0",
                "Gemma 3 dense text trunk (TranslateGemma), bf16 or w4a8 at convert: sliding-window "
                "and linearly-scaled global attention over a (1 + w) RMSNorm stream with GELU "
                "feed-forward")
#endif

}  /* namespace gemma3 */
