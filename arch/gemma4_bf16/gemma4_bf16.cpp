/* gemma4_bf16.cpp -- Gemma 4 dense, text-only, bf16.
 *
 * Two attention classes over ONE residual stream, alternating by layer. A "sliding" layer is
 * sliding-window attention over head_dim 256 with a v projection and standard RoPE; a "global"
 * layer is windowless attention over head_dim 512 with no v projection (k == v) and proportional
 * RoPE (the "head" ladder) over the partial rotary width. Every layer is GemmaRMSNorm -- x_hat *
 * (1 + w) -- with a SECOND norm on each projection's output before its residual, a weightless RMS
 * on the value, and a GELU (not SwiGLU) feed-forward. The final logits are soft-capped to
 * cap * tanh(x / cap).
 */
#include <arch/rad_arch.h>

#include "gemma4_chat_format.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4_bf16 {

using namespace rad::arch;

/* ---- container geometry RadModelMeta does not name. Both spellings are accepted: a .rad
 * converted from the HF safetensors carries the flat HuggingFace keys and one converted from a
 * GGUF carries llama.cpp's. */
static long long g4_i(const RadModelMeta* m, const char* bare, long long dflt) {
    long long v = rad_meta_geti(m, bare, dflt);
    if (v == dflt) v = rad_meta_geti(m, (std::string("text_config.") + bare).c_str(), dflt);
    return v;
}
static double g4_f(const RadModelMeta* m, const char* bare, double dflt) {
    double v = rad_meta_getf(m, bare, dflt);
    if (v == dflt) v = rad_meta_getf(m, (std::string("text_config.") + bare).c_str(), dflt);
    return v;
}

static constexpr int TRC_SITES = 5;   /* x0, h_attn, h_mlp, x_l, h_fin */
static inline bool trc_on() { static const bool on = std::getenv("RADIANCE_TRACE_RESID") != nullptr; return on; }
struct Model;                 /* forward; defined below -- the trace ring lives in Model */
static uint16_t* trc_ring(int rank);   /* forward: defined after the Model global below */
static const char* const TRC_N[TRC_SITES] = {"x0", "h_attn", "h_mlp", "x_l", "h_fin"};
/* Async copy of the LAST token row (T-1) of a bf16 [max_tok,n_embd] buffer into the pinned
 * ring at (ring-row, site). The ring is indexed by (row, site), NOT by step: each step OVERWRITES
 * the previous step's rows, so the scan at the top of step N+1 reads step N's data -- and must
 * drain the stream first (step() does, trace-gated), because the engine's step_end sync does not
 * wait for these out-of-band copies on the host build (the h_fin amax=0 the first traced run
 * showed = a not-yet-landed copy read as zero). Pinned dst is mandatory (host.cpp:1860). Works on
 * both backends: the host backend's rad_memcpy_async and rad_dev_alloc serve the same calls. */
static inline void trc_copy(RadCtx* c, uint16_t* ring, int64_t n_embd, int row, int site,
                            rad_buf b, int64_t T) {
#if !defined(RAD_ARCH_NO_EXPORTS)
    /* The arch_test harness (RAD_ARCH_NO_EXPORTS) compiles this file in without the runtime: it
     * supplies the declare-phase ABI and nothing of the step-phase one, so the trace's runtime
     * calls (rad_buf_ptr / rad_memcpy_async / rad_stream) exist only in a real build. */
    if (!trc_on() || !ring || T <= 0) return;
    void* p = rad_buf_ptr(c, b);
    if (!p) return;
    rad_memcpy_async(ring + ((size_t)row * TRC_SITES + (size_t)site) * (size_t)n_embd,
                     (const uint16_t*)p + (T - 1) * (size_t)n_embd, n_embd * 2, rad_stream(c));
    /* Synchronous for the trace: the source buffer (w.h / b_x / b_h) is REUSED by a later op in
     * the same step, so an async copy that lands after the reuse would read the wrong data (the
     * host backend's h_attn site read as 0 / stale for this reason). Draining here -- right after
     * the copy is issued, before the next op overwrites the buffer -- makes the ring hold the
     * value at this point in the step. The cost is a sync per traced site (acceptable: the trace
     * is a debug path, off by default). */
    rad_stream_sync(rad_stream(c));
#endif
}

/* layer_types is optional. When present it must name EVERY layer "full_attention" or
 * "sliding_attention"; when absent the (l + 1) % interval == 0 fallback is the model card. */
static int g4_parse_types(const char* lt, int64_t n, std::vector<int>& is_global) {
    is_global.assign((size_t)n, -1);
    size_t got = 0;
    const char* p = lt;
    while (*p) {
        if (!std::strncmp(p, "full_attention", 14)) {
            if (got >= (size_t)n) return RAD_E_INVAL;
            is_global[got++] = 1; p += 14;
        } else if (!std::strncmp(p, "sliding_attention", 17)) {
            if (got >= (size_t)n) return RAD_E_INVAL;
            is_global[got++] = 0; p += 17;
        } else {
            ++p;   /* separators: whitespace, quotes, [ ] , and / */
        }
    }
    if (got != (size_t)n) return RAD_E_INVAL;
    for (size_t i = 0; i < is_global.size(); ++i) if (is_global[i] < 0) return RAD_E_INVAL;
    return RAD_OK;
}

static int g4_schedule(const RadModelMeta* m, int64_t n, int64_t interval,
                       std::vector<int>& is_global) {
    const char* lt = rad_meta_gets(m, "layer_types", rad_meta_gets(m, "text_config.layer_types", nullptr));
    is_global.assign((size_t)n, 0);
    if (lt && *lt) {
        RAD_ARCH_TRY(g4_parse_types(lt, n, is_global));
    } else {
        if (interval <= 0) {
            fprintf(stderr, "radiance: gemma4: no layer_types and invalid full_attention_interval %lld\n",
                    (long long)interval);
            return RAD_E_INVAL;
        }
        for (int64_t l = 0; l < n; ++l) is_global[(size_t)l] = ((l + 1) % interval == 0) ? 1 : 0;
    }
    return RAD_OK;
}

/* THE CHECKPOINT'S NAME FOR A LINEAR'S WEIGHT. Google's CT-INT4 release stores the codes as
 * `X.weight_packed` (compressed-tensors, format pack-quantized); the bf16 release -- and every
 * finetune of it, which ships no QAT form -- stores `X.weight`. Read once in declare from the
 * checkpoint's own quantization_config, so a bf16 source converted with
 * `--set quantization_config.quant_method=compressed-tensors` (which selects this plugin) maps its
 * plain names, and the recipe then decides the grid. At serve time the container's declared names
 * are used and this is not consulted. */
static const char* g_wsfx = "weight_packed";

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

/* ---- one linear, plain bf16 or w4a8. The recipe re-quantises the GEMM weights to int4 codes
 * with an f16 scale per [1 x 128] (group 128); a weight no rule matches stays bf16. This probes
 * the weight's encoding and chooses the op by it: `gemm_nt` for a plain bf16 weight, `gemm_nt_q`
 * at dtype w4a8 for the codes + their scale. w4a8 is the int8-activation grid: it reads an int8
 * activation + an f32 per-row scale, which the model's bf16 activation has to be quantised into.
 * The quantiser is declared here (quant_act_i8) over the activation's bf16 plane and writes the
 * int8 pair (q8, s8) the GEMM reads -- in the GEMM's fragment byte order, with one scale per row.
 * quant_act_i8g (row-major, a per-(row,group) scale) is i8a8's producer, not this one's; pairing
 * it with w4a8 is the compounding-into-inf seed of the all-NaN logits (see the declare comment).
 * w4a16 was tried and is the WRONG grid: its row declares an f16 activation and
 * its launcher never checks the operand dtype, so feeding it the model's bf16 bits reinterprets
 * them as f16 -- a magnitude-compressed garbage stream (bf16 0.1 -> 1.45, 1.0 -> 1.875), finite
 * nonsense that propagates through the residual. The name map is made here, before the weight is
 * asked what it is (Builder::finish() needs the map entry before the declaration). */
struct G4Linear {
    rad_weight w = 0, ws = 0;   /* the codes + scale (w4a8), or the single bf16 weight */
    rad_op     op = 0, op_q = 0;   /* the quantised GEMM and, for w4a8, its activation quantiser */
    rad_buf    q8 = 0, s8 = 0;     /* the int8 pair (i8 codes + f32 row scale) the quantiser writes */
    int64_t    N = 0, K = 0;
    bool       quant = false;   /* true: w4a8 */

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {});
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
};

inline int G4Linear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src,
                            int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                            rad_buf a, rad_buf y,
                            std::initializer_list<int64_t> row_parts) {
    N = n_out; K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1) RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else                RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    /* a builder with no model behind it (a test, a synthetic tool) answers RAD_E_NOTFOUND, and a
     * weight declared there is read as plain in its declared dtype -- the bf16 path. */
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

    /* a w4 weight: i4 codes + an f16 scale per [1 x 128]. The GEMM (gemm_nt_q at w4a8) reads an
     * int8 activation + an f32 per-row scale, which the model's bf16 activation has to be quantised
     * into, so the quantiser (quant_act_i8g) is declared over the activation's bf16 plane `a` and
     * writes the int8 pair (q8, s8) the GEMM reads. The codes + scale layout is byte-for-byte the
     * w4a16 one -- the container serves unchanged; only the op dtype and the activation producer
     * differ. */
    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
    /* An 8-bit weight on the same grid -- i8 codes, an f16 scale per [1 x 128] -- is the same
     * pair at twice the radix: gemm_nt_q at w8a8 reads the identical int8 activation + per-row
     * scale the w4a8 GEMM reads, so the quantiser declaration is shared. It is how a recipe keeps
     * a linear the int4 grid hurts (the attention, down_proj) closer to the checkpoint. */
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
        q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8,  { g.max_tok, K });
        s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, 1 });
        if (!q8 || !s8) return RAD_E_INVAL;
        /* quant_act_i8, NOT quant_act_i8g: the w4a8 GEMM's A fragment reads a byte-shuffled plane
         * (r4d_gemm_w4a8_nt_m64.hip:46-52, "the quantiser writes it") and applies an f32 per-ROW
         * scale at the store (:61-63), so the producer must be the fragment-order per-row quantiser
         * (r4d_quant_act_i8.hip:1-2). quant_act_i8g writes PLAIN ROW-MAJOR codes with a
         * per-(row,group) scale (r4d_rows.cpp:2043-2046, "what the int8 trunk GEMMs (i8a8) read")
         * -- pairing it with this GEMM shuffles the dot and broadcasts the wrong scale grid, the
         * compounding-into-inf seed of the all-NaN logits. */
        /* group=K, NOT 128: the host quantisers (libref/libavx) are per-`group` -- one f32 scale
         * per group of `group` elements -- while the device one (r4d_quant_act_i8) is per-ROW and
         * ignores `group` in this 3-operand form (it only reads it in the asum form). group=K makes
         * the host ngrp=1 (one scale per row, the [M,1] buffer the arch declares) and is a no-op on
         * the device, so the same declaration is self-consistent on both backends. group=128 left
         * the host quantiser refusing the [M,1] scale (it wants [M, K/128]). */
        op_q = rw(b, RAD_OP(b, "quant_act_i8",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                               RAD_INT("group", K), RAD_STR("dtype", "bf16")),
                    RAD_NOWEIGHTS),
                { a }, { q8, s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", 128), RAD_STR("dtype", cdt == RAD_I8 ? "w8a8" : "w4a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        quant = true;
        return (op_q && op) ? RAD_OK : RAD_E_INVAL;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, w4a8 or w8a8 -- "
                    "neither matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void G4Linear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        /* quantise the bf16 activation into the int8 pair (fragment order + a per-row scale), then
         * read the pair. The scale plane is one f32 per row, browsed at T rows like the codes.
         * quant_act_i8's schema is 4 operands -- x, q, scale, asum -- and `asum` (the per-group row
         * sums the asymmetric grids subtract) is absent on this symmetric grid, so it is passed as
         * a null tensor, not dropped: operands are positional (issue.cpp). */
        RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                    brows(y, T), RAD_NONE, RAD_NONE);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
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

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                 rad_kvgroup group, const Wire& wire, const char* pre);
    void step(RadCtx* c, const RadBatch* batch) const;
};

int Gemma4Attn::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const ClassGeom& cgeo,
                        rad_kvgroup group, const Wire& wire, const char* pre) {
    g = geom; cls = cgeo; layer = l; kv = group; w = wire;
    const int64_t qd = cls.q_dim(), kvd = cls.kv_dim();

    /* the norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),      nm.ckpt("%s.%lld.input_layernorm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.q_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l),    nm.ckpt("%s.%lld.self_attn.k_norm.weight",   pre, (long long)l)));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post_norm.weight", l), nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
    /* declared bf16, not f32: the checkpoint holds them bf16 and no rule re-quantises them, so an
     * f32 declaration would widen them at load -- and a widened weight cannot be served from the
     * mapped tier, which the whole layer group is pinned to because the w4a16 GEMM has no device
     * kernel for the M > 64 prefill band. Reading the gain in its own (bf16) dtype is identical:
     * the widening was a lossless upcast of the same bf16 value. */
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qn = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_kn = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_BF16, {cls.head_dim},
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.attn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* the GEMM weights: bf16 or w4a8, probed by G4Linear. */
    RAD_ARCH_TRY(ln_q.declare(b, nm, g, nm.f("blk.%d.attn_q", l),
                              { nm.ckpt("%s.%lld.self_attn.q_proj.%s", pre, (long long)l, g_wsfx) },
                              qd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.q));
    RAD_ARCH_TRY(ln_k.declare(b, nm, g, nm.f("blk.%d.attn_k", l),
                              { nm.ckpt("%s.%lld.self_attn.k_proj.%s", pre, (long long)l, g_wsfx) },
                              kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.k));
    if (cls.has_v)
        RAD_ARCH_TRY(ln_v.declare(b, nm, g, nm.f("blk.%d.attn_v", l),
                                  { nm.ckpt("%s.%lld.self_attn.v_proj.%s", pre, (long long)l, g_wsfx) },
                                  kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.v));
    RAD_ARCH_TRY(ln_o.declare(b, nm, g, nm.f("blk.%d.attn_o", l),
                              { nm.ckpt("%s.%lld.self_attn.o_proj.%s", pre, (long long)l, g_wsfx) },
                              g.n_embd, qd, RAD_SHARD_COL, grp_layer(l), w.attn, w.h));

    /* input norm: x_hat * w over the residual stream. */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* Gemma normalises the value with a WEIGHTLESS rms (the checkpoint has no v_norm gain). A
     * sliding class normalises v in place; a k_eq_v class derives v from the RAW k into its own
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
    /* The rope kernel serves the fused QKV row and requires n_head > 0 (it rotates n_head +
     * n_head_kv heads over a row of (n_head + 2*n_head_kv)*head_dim). The K heads are a separate
     * buffer, so the K-only form is expressed as a Q-only form over the K heads: n_head is the
     * count of heads to rotate (here n_head_kv) and n_head_kv is 0, so the row is
     * n_head_kv*head_dim -- exactly the K buffer's width. RoPE is identical for Q and K (same
     * theta, same positions), so this is the reference's separate K rope (llama.cpp ggml_rope_ext
     * over the K buffer). */
    op_rk = rw(b, RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                             RAD_INT("n_head", cls.n_head_kv), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", cls.theta), RAD_F64("scale", cls.scale),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", cls.n_rot),
                             RAD_STR("ladder", cls.head_ladder ? "head" : "rotary")),
                  RAD_NOWEIGHTS),
                 {w.k}, {w.k});

    /* Gemma 4 scores Q.K WITHOUT the 1/sqrt(head_dim) factor: both the HF reference
     * (Gemma4TextAttention: self.scaling = 1.0) and the vLLM oracle use scaling = 1.0 --
     * the learnable q/k RMSNorm weights absorb the scaling, so no extra 1/sqrt(d) is applied.
     * attn_paged's default is 1/sqrt(head_dim), so pass 1.0 explicitly to match the reference. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                   RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", cls.head_dim),
                              RAD_INT("gqa", cls.gqa()), RAD_INT("causal", 1), RAD_INT("window", cls.window),
                              RAD_INT("n_head", cls.n_head), RAD_INT("block_size", RAD_KV_BLOCK),
                              RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                              RAD_F64("scale", 1.0),
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

    /* o_proj is column-sharded: the rank's partial sum is reduced FIRST, then normed. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                              RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                              RAD_INT("min_bytes", g.wire_min_bytes)),
                   RAD_NOWEIGHTS),
                  {w.h}, {w.h});

    /* the SECOND residual: the (reduced) attention output is itself normed before the add. */
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

void Gemma4Attn::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    ln_q.step(c, w.h, w.q, T);
    ln_k.step(c, w.h, w.k, T);
    if (cls.has_v) ln_v.step(c, w.h, w.v, T);

    /* normalise v FIRST: on a k_eq_v class v is derived from the raw k, before k_norm rewrites it. */
    RAD_ISSUE_N(c, op_vn, T * cls.n_head_kv, brows(cls.has_v ? w.v : w.k, T), brows(w.v, T));
    RAD_ISSUE_N(c, op_qn, T * cls.n_head,    brows(w.q, T), RAD_W(w_qn), brows(w.q, T));
    RAD_ISSUE_N(c, op_kn, T * cls.n_head_kv, brows(w.k, T), RAD_W(w_kn), brows(w.k, T));

    RAD_ISSUE(c, op_rq, brows(w.q, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, op_rk, brows(w.k, T), rope_posmc(g, batch, T));

    RAD_ISSUE(c, op_kv, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    ln_o.step(c, w.attn, w.h, T);
    trc_copy(c, trc_ring(rad_rank(c)), g.n_embd, layer, 1, w.h, T);  /* h_attn: pre-post-norm o-proj output */
    /* THE ALL-REDUCE BEFORE THE POST-NORM. o_proj is column-sharded, so each rank holds a PARTIAL
     * sum here; the norm is of the whole output (HF, and vLLM's RowParallelLinear, reduce first).
     * Normed per rank and then summed is sum_r rmsnorm(p_r), not rmsnorm(sum_r p_r). */
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
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

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire, const char* pre);
    void step(RadCtx* c, const RadBatch* batch) const;
};

int Gemma4Mlp::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire, const char* pre) {
    g = geom; layer = l; w = wire;

    /* the norms: always bf16 (no rule matches them), name map first. */
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_norm.weight", l), nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
    RAD_ARCH_TRY(map_copy (b, nm.f("blk.%d.ffn_post_norm.weight", l), nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
    /* bf16, not f32: same reason as the attention norms -- an f32 declaration widens them and a
     * widened weight cannot be served from the mapped tier the whole layer group is pinned to. */
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_BF16, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_post = decl_w(b, nm.f("blk.%d.ffn_post_norm.weight", l), RAD_BF16, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* the GEMM weights: bf16 or w4a8, probed by G4Linear. gate_up is the gate|up fusion. */
    RAD_ARCH_TRY(ln_gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                               { nm.ckpt("%s.%lld.mlp.gate_proj.%s", pre, (long long)l, g_wsfx),
                                 nm.ckpt("%s.%lld.mlp.up_proj.%s",   pre, (long long)l, g_wsfx) },
                               2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l), w.h, w.gu,
                               { g.n_ff, g.n_ff }));
    RAD_ARCH_TRY(ln_down.declare(b, nm, g, nm.f("blk.%d.ffn_down", l),
                                 { nm.ckpt("%s.%lld.mlp.down_proj.%s", pre, (long long)l, g_wsfx) },
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

void Gemma4Mlp::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, op_norm,  brows(w.x, T),   RAD_W(w_norm),  brows(w.h, T));
    ln_gu.step(c, w.h, w.gu, T);
    RAD_ISSUE(c, op_act,   brows(w.gu, T),  brows(w.ffn, T));
    ln_down.step(c, w.ffn, w.h, T);
    trc_copy(c, trc_ring(rad_rank(c)), g.n_embd, layer, 2, w.h, T);  /* h_mlp: pre-post-norm down-proj output */
    /* The all-reduce before the post-norm, as in the attention block: down_proj is column-sharded. */
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE_N(c, op_post, T, brows(w.h, T), RAD_W(w_post), brows(w.h, T));
    RAD_ISSUE(c, op_add,   brows(w.x, T),   brows(w.h, T),  brows(w.x, T));
}

/* ==================================================================== the assistant head
 *
 * GEMMA 4'S DRAFT HEAD, the "assistant" (Gemma4AssistantForCausalLM), merged under `draft.` by
 * `rad-convert --draft-model <gemma-4-*-it-assistant>`. Read off the checkpoint and three reference
 * implementations -- transformers' modeling_gemma4_assistant.py with its candidate generator,
 * vLLM's gemma4_mtp.py with spec_decode/gemma4.py, and llama.cpp's gemma4-assistant.cpp:
 *
 *     x  = pre_projection( concat( E_trunk[tok] * sqrt(n_trunk), h_trunk ) )        2*n_trunk -> n
 *     x  = the head's layers over x (31B: sliding, sliding, sliding, full)
 *     hd = norm(x)
 *     logits = lm_head(hd)       the head's OWN table, tied to its embed_tokens; no soft-cap
 *     h' = post_projection(hd)                                                       n -> n_trunk
 *
 * `h_trunk` is the trunk's FINAL-NORMED hidden state -- the trunk lm_head's input -- and `E_trunk`
 * the trunk's embedding table, scaled the way the trunk scales it. The head's own embed_tokens is
 * its lm_head and nothing else (vLLM swaps the trunk's table in for the input side). The
 * concatenation is the embedding FIRST, then the hidden state; all three references agree.
 *
 * A LAYER IS A TRUNK LAYER WITH THE KEY AND VALUE TAKEN AWAY: q_proj, the per-head q norm, the
 * rotary of its class, attention over the TRUNK'S cache at scale 1, o_proj, the same four
 * plain-gain norms, the GELU feed-forward and the per-layer scalar. Each layer reads the LAST trunk
 * layer of its own class -- the sliding ones the last sliding layer, the full one the last full
 * layer, 58 and 59 on the 31B (vLLM's _setup_gemma4_kv_sharing). So the head's q heads and the
 * trunk's kv heads must be the same geometry, and declare checks that they are.
 *
 * IT KEEPS NO CACHE OF ITS OWN, and that is what the core is told (rad_drafter_shared_kv): no
 * history pass, one row a drafting sequence, every round at the committed position over the
 * committed context. Round 1's row embeds the token the trunk chose and continues from the trunk's
 * hidden state; round r embeds round r-1's proposal and continues from its h'. What the head
 * proposed reaches the next round through those two and nothing else -- which is the whole design
 * of the head, not a shortcut taken here.
 *
 * THE TENSOR-PARALLEL SPLIT IS THE TRUNK'S: q heads row-sharded over the ranks, so each rank's
 * queries meet its own share of the trunk's kv heads; o_proj and the feed-forward down projection
 * column-sharded, each followed by an all-reduce BEFORE its post-norm (the norm of a partial sum is
 * not the partial of a norm). The two projections and every norm are replicated: they read and
 * write the replicated stream, and a split would cost an all-reduce to save a few MiB. The lm_head
 * is vocab-sharded like the trunk's and goes through the same sampler chain, greedy. */
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
    int64_t rows = 0;       /* the most rows one pass carries: one a drafting sequence */
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

/* A `draft.` key under the head's text config, where Gemma4AssistantConfig nests every dimension,
 * or at the top of it. */
static long long hd_i(const RadModelMeta* m, const char* k, long long d) {
    long long v = rad_meta_geti(m, (std::string("draft.text_config.") + k).c_str(), -0x7FFFFFFFll);
    return v != -0x7FFFFFFFll ? v : rad_meta_geti(m, (std::string("draft.") + k).c_str(), d);
}
static double hd_f(const RadModelMeta* m, const char* k, double d) {
    const double nan = std::nan("");
    double v = rad_meta_getf(m, (std::string("draft.text_config.") + k).c_str(), nan);
    return std::isnan(v) ? rad_meta_getf(m, (std::string("draft.") + k).c_str(), d) : v;
}

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
    n       = hd_i(meta, "hidden_size", 0);
    n_trunk = rad_meta_geti(meta, "draft.backbone_hidden_size", 0);
    const long long n_l      = hd_i(meta, "num_hidden_layers", 0);
    const long long n_h      = hd_i(meta, "num_attention_heads", 0);
    const long long n_kv     = hd_i(meta, "num_key_value_heads", 0);
    const long long n_kv_g   = hd_i(meta, "num_global_key_value_heads", n_kv);
    const long long head_s   = hd_i(meta, "head_dim", 0);
    const long long head_g   = hd_i(meta, "global_head_dim", head_s);
    const long long ff       = hd_i(meta, "intermediate_size", 0);
    const long long win      = hd_i(meta, "sliding_window", 0);
    const long long vocab    = hd_i(meta, "vocab_size", 0);
    const long long ordered  = rad_meta_geti(meta, "draft.use_ordered_embeddings", 0);
    const double    eps      = hd_f(meta, "rms_norm_eps", 0.0);
    const double    theta_s  = hd_f(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    const double    theta_g  = hd_f(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    double          prf      = hd_f(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0);
    softcap = hd_f(meta, "final_logit_softcapping", 0.0);
    if (!(prf > 0.0)) prf = 1.0;

    if (n <= 0 || n_l <= 0 || n_h <= 0 || ff <= 0 || !(eps > 0.0)) {
        fprintf(stderr, "radiance: gemma4 assistant: the `draft.` config is incomplete (hidden %lld, "
                        "layers %lld, heads %lld, ffn %lld, eps %g)\n",
                (long long)n, n_l, n_h, ff, eps);
        return RAD_E_INVAL;
    }
    /* IT READS THE TRUNK'S CACHE, so everything the cache's shape depends on must agree: the two
     * head widths, the two kv head counts and the window. A head that disagrees was trained beside
     * another trunk, and its attention would read keys that are not the ones it learned against. */
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
    /* THE QUERIES MUST ROTATE AS THE TRUNK'S KEYS DID, or q.k measures the wrong distance. */
    const int64_t rot_g = (int64_t)(head_g * prf + 0.5);
    if (theta_s != cs.theta || theta_g != cg.theta || rot_g != cg.n_rot) {
        fprintf(stderr, "radiance: gemma4 assistant: its rotary (theta %g / %g, %lld of %lld global "
                        "dims) is not the trunk's (%g / %g, %lld): its queries would meet the "
                        "trunk's keys at the wrong angle\n",
                theta_s, theta_g, (long long)rot_g, head_g, cs.theta, cg.theta,
                (long long)cg.n_rot);
        return RAD_E_INVAL;
    }
    /* CENTROID MASKING (use_ordered_embeddings) scores a top-k of vocabulary clusters instead of
     * the whole head. The 31B's head does not use it, and nothing here implements it; refusing is
     * better than a head that scores every token while its checkpoint expects a mask. */
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
     * round copies the head's own h' from the round before. Both write the half in place: a
     * gather and a cast read their destination's row pitch off the operand. */
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

    /* pre_projection: the joined row to the head's width. Replicated: it reads and writes the
     * replicated stream. */
    RAD_ARCH_TRY(ln_pre.declare(b, nm, g, nm.f("mtp.pre_proj"), { nm.ckpt("draft.pre_projection.weight") },
                                n, 2 * n_trunk, RAD_SHARD_NONE, grp_layer((int)(base + n_l)),
                                b_c, b_x));

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

        L.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                    RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(L.w_norm)),
                       {b_x}, {b_h});
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
        /* ONE QUERY ROW A SEQUENCE over the trunk's cache, at scale 1 like the trunk's. The
         * context is the committed positions (the core's doing: rad_drafter_shared_kv), so the
         * decode band is the only band there is. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", L.cls.head_dim),
                                    RAD_INT("gqa", L.cls.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", L.cls.window), RAD_INT("n_head", L.cls.n_head),
                                    RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", rows),
                                    RAD_INT("max_ctx", g.max_ctx), RAD_F64("scale", 1.0),
                                    RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                       {L.q}, {L.attn});
        if (g.world > 1)
            L.op_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                       RAD_INT("min_bytes", g.wire_min_bytes)),
                            RAD_NOWEIGHTS),
                         {b_h}, {b_h});
        L.op_post = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                    RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(L.w_post)),
                       {b_h}, {b_h});
        L.op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                      {b_x, b_h}, {b_x});
        L.op_fnorm = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                     RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(L.w_fnorm)),
                        {b_x}, {b_h});
        L.op_act = rw(b, RAD_OP(b, "gelu_mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n_ff), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                      {b_gu}, {b_ffn});
        if (g.world > 1)
            L.op_far = rw(b, RAD_OP(b, "all_reduce",
                             RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", n, rows * n),
                                        RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                        RAD_INT("min_bytes", g.wire_min_bytes)),
                             RAD_NOWEIGHTS),
                          {b_h}, {b_h});
        L.op_fpost = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                     RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(L.w_fpost)),
                        {b_h}, {b_h});
        L.op_fadd = rw(b, RAD_OP(b, "add",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                       {b_x, b_h}, {b_x});
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
    /* VOCAB-SHARDED, LIKE THE TRUNK'S: the draft's logits go through the same sampler chain, which
     * merges the ranks' candidate sets, so a rank holds its own slice of the vocabulary. Declared
     * at its codes' dtype: bf16 as Google ships it, or int8 / block fp8 where a recipe makes it so,
     * and the logits_gemm kernel that reads that encoding serves it. It is the largest read of a
     * round, and a draft is only a proposal: the trunk verifies every token with its own head. */
    w_lm = decl_w(b, nm.f("mtp.output.weight"), weight_codes_dtype(b, nm.f("mtp.output.weight"), RAD_BF16),
                  {g.n_vocab, n}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_layer((int)(base + n_l)));
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                              RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_norm)),
                 {b_x}, {b_hd});
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
    /* post_projection LAST: h' is the next round's input, and the op that reads it (op_hcopy) is
     * the head's first. Declared here, its life runs the whole head, so no op of a pass can be laid
     * over it before the round after has read it. */
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

/* One pass. The batch is the one the scheduler staged for a head with no cache of its own: one row
 * a drafting sequence, at the committed position, its context ending there. */
void Gemma4Head::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    if (batch->draft_pass < 0) {
        /* Every row of the pass is a drafting row: that is what rad_drafter_shared_kv asked the
         * scheduler for. Anything else is a batch built for a head with a cache of its own. */
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

    uint16_t* trc = nullptr;   /* pinned ring [n_main+1][TRC_SITES][n_embd] bf16; set only when tracing */
    bool      trc_done = false;

    /* The assistant head, declared only when the container carries one and the deployment
     * speculates (`max_spec` > 0). */
    Gemma4Head head{};
    bool       have_head = false;
};

static Model g_model[MAX_RANKS];

static uint16_t* trc_ring(int rank) { return g_model[rank].trc; }

/* bf16 bit classification: u = raw << 16; exp7 = u & 0x7F80; inf = exp7 && !frac; nan = exp7 && frac.
 * amax is over the NaN-free lanes only, so an all-NaN row prints amax=0 (the engine's -vv signature). */
static void trc_scan_print(const Model& m) {
    if (!m.trc) { fprintf(stderr, "radiance: resid trace: no ring\n"); return; }
    const int rows = (int)m.n_main + 1;
    const int64_t n = m.g.n_embd;
    for (int pass = 0; pass < 2; ++pass) {          /* pass 0: first inf; pass 1: first nan */
        for (int r = 0; r < rows && pass >= 0; ++r) {
            for (int s = 0; s < TRC_SITES; ++s) {
                const uint16_t* row = m.trc + ((size_t)r * TRC_SITES + (size_t)s) * (size_t)n;
                int n_nan = 0, n_inf = 0; double amax = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    const uint32_t u = (uint32_t)row[i] << 16;
                    float f; std::memcpy(&f, &u, 4);
                    const bool spe = (u & 0x7F80u) == 0x7F80u;
                    if (spe && (u & 0x007Fu)) ++n_nan;
                    else if (spe) ++n_inf;
                    else { const double a = std::fabs((double)f); if (a > amax) amax = a; }
                }
                const bool hit = pass ? n_nan > 0 : n_inf > 0;
                if (!hit) continue;
                fprintf(stderr, "TRC %s layer %2d site %-6s: nan=%d inf=%d amax=%.4g\n",
                        pass ? "NAN" : "INF", r, TRC_N[s], n_nan, n_inf, amax);
                break;
            }
        }
    }
    /* Always-on summary, so an all-finite run still proves the ring was filled (a trace whose
     * copies never land would print amax=0 here, not silence). */
    auto site_amax = [&](int r, int s) {
        const uint16_t* row = m.trc + ((size_t)r * TRC_SITES + (size_t)s) * (size_t)n;
        double a = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const uint32_t u = (uint32_t)row[i] << 16;
            if ((u & 0x7F80u) == 0x7F80u) return -1.0;   /* inf or nan: say so, do not classify */
            float f; std::memcpy(&f, &u, 4);
            const double v = std::fabs((double)f);
            if (v > a) a = v;
        }
        return a;
    };
    /* The ring rows: 0 = x0 + layer 0 (sites 1-3), 1..n_main-1 = the layers, n_main = h_fin.
     * -1.0 = an inf/nan in the row (the bisection signal); 0 = a zero row (a not-landed copy
     * after the sync fix means the sync is broken, not the data). */
    fprintf(stderr,
            "TRC summary: x0 %.4g | L0 h_attn %.4g h_mlp %.4g x_l %.4g | L%d x_l %.4g | h_fin %.4g\n",
            site_amax(0, 0),
            site_amax(0, 1), site_amax(0, 2), site_amax(0, 3),
            rows - 2, site_amax(rows - 2, 3),
            site_amax(rows - 1, 4));
}

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
    {
        const char* fm = rad_meta_gets(meta, "quantization_config.format", "");
        g_wsfx = (fm && std::strstr(fm, "pack-quantized")) ? "weight_packed" : "weight";
    }

    /* ---- geometry RadModelMeta does not name. */
    const int64_t kv_g_all    = g4_i(meta, "num_global_key_value_heads", 0);
    const int64_t head_g      = g4_i(meta, "global_head_dim", 0);
    const int64_t sliding_win = g4_i(meta, "sliding_window", 0);
    const int64_t kv_shared   = g4_i(meta, "num_kv_shared_layers", 0);
    const int64_t interval    = g4_i(meta, "full_attention_interval", 0);
    const double  theta_s     = g4_f(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    const double  theta_g     = g4_f(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    const double  softcap     = g4_f(meta, "final_logit_softcapping", 0.0);
    double        prf         = g4_f(meta, "rope_parameters.full_attention.partial_rotary_factor", -1.0);

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

    /* Plain RMSNorm: x_hat * w, on every norm including the post-projection ones. Gemma 4
     * (unlike Gemma 2/3) does NOT use the (1 + w) form: the vLLM 0.27.1 oracle (which serves
     * this checkpoint correctly) uses the stock RMSNorm (weight init 1.0), HF transformers'
     * Gemma4RMSNorm multiplies by the weight with no +1 (weight init 1.0), and llama.cpp's
     * gemma4 uses LLM_NORM_RMS with no load-time offset. wadd=0.0 = the plain form. */
    g.wadd    = 0.0f;
    g.wadd_qk = 0.0f;
    const float final_wadd = 0.0f;

    m.n_main = g.n_layer;
    std::vector<int> is_global;
    RAD_ARCH_TRY(g4_schedule(meta, g.n_layer, interval, is_global));
    int n_global = 0;
    for (int v : is_global) n_global += v;

    /* the slider comes off geom_from (head_dim 256, 16 kv heads, full rotation, theta_s); the global
     * class is built from the container head. */
    m.cs.n_head = g.n_head;      m.cs.n_head_kv = g.n_head_kv; m.cs.head_dim = g.head_dim;
    m.cs.n_rot  = g.head_dim;    m.cs.window = sliding_win;    m.cs.theta = theta_s;
    m.cs.scale  = g.rope_scale;  m.cs.head_ladder = false;     m.cs.has_v = true;
    m.cg.n_head = g.n_head;      m.cg.n_head_kv = kv_g;        m.cg.head_dim = head_g;
    m.cg.n_rot  = n_rot_g;       m.cg.window = 0;              m.cg.theta = theta_g;
    m.cg.scale  = g.rope_scale;  m.cg.head_ladder = true;      m.cg.has_v = false;

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
     * host-only), so an f32-declared output_norm would widen and fail to serve from the mapping. */
    m.w_out = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* DECLARED AT ITS CODES' DTYPE, as the assistant head's lm_head is: bf16 as the checkpoint ships
     * it, or int8 / block fp8 where a recipe (`--quant output.weight=...`) makes it so, and the
     * logits_gemm kernel for that encoding serves it. At a decode step it is the largest single read
     * of the trunk (n_vocab x n_embd a rank), so its encoding is a direct share of the step. */
    m.w_lm  = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                     {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    /* The embedding is scaled by sqrt(n_embd) at the LOOKUP, not baked into the table (the
     * reference's build_inp_embd and HF's ScaledWordEmbedding both do it there), and the
     * checkpoint's table is plain -- so the scale rides the op and the table stays tied to the
     * head's. Absent it, the residual stream carries a 73x-too-small token signal and the first
     * layer's scale-invariant norm hides nothing downstream. */
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

    /* ---- the layers. This loop is the model card. */
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

        /* the per-layer scalar: `cur = cur * scalar`, a [1] bf16 over the residual stream. */
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
     * the deployment speculates. A container converted without it, or a serve at max_spec 0,
     * declares none of it: half a gigabyte of weights stays off the card. */
    if (ctx->max_spec > 0 && has_assistant(meta)) {
        int last_s = -1, last_f = -1;
        for (int64_t l = 0; l < m.n_main; ++l) (is_global[(size_t)l] ? last_f : last_s) = (int)l;
        RAD_ARCH_TRY(m.head.declare(b, nm, meta, g, m.cs, m.cg, m.kv_s, m.kv_g, last_s, last_f,
                                    m.n_main, m.w_tok, m.b_h, m.b_logits));
        m.have_head = true;

        /* WHAT THE ENGINE DRIVES. Serial: `max_spec` dependent passes, each embedding what the
         * last one proposed, through the ordinary sampler chain (proposal 0) -- and a head with no
         * cache of its own, so the core stages every pass at the committed position. */
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

    rad_note(b, "rank %d/%d: %lld layers (%lld global, %lld sliding), sliding window %lld, "
                "slider %lld heads x %lld, global %lld heads x %lld, soft-capped logits",
             g.rank, g.world, (long long)g.n_layer, (long long)n_global,
             (long long)(g.n_layer - n_global), (long long)sliding_win,
              (long long)g.n_head, (long long)g.head_dim, (long long)g.n_head, (long long)head_g);
#if !defined(RAD_ARCH_NO_EXPORTS)
    if (trc_on()) {
        const int64_t bytes = (m.n_main + 1) * TRC_SITES * g.n_embd * 2;
        m.trc = (uint16_t*)rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
        fprintf(stderr, "radiance: gemma4 resid trace ring %s (%lld bytes)\n",
                m.trc ? "pinned" : "FAILED", (long long)bytes);
    }
#endif
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* A DRAFT PASS RUNS THE HEAD AND NOTHING ELSE. The trunk already ran this step and left its
     * final-normed hidden state in `h`, which the head's history pass gathers from. */
    if (batch->draft_pass != 0) {
        if (!m.have_head) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` assistant, or declare ran at max_spec 0");
            return;
        }
        m.head.step(c, batch);
        return;
    }
    /* Scan on every step from the second on: the ring holds the PREVIOUS step's rows. The
     * previous step's copies (x0 early, h_fin last) are drained HERE before the read: the
     * engine's step_end sync does not wait for the arch's out-of-band trc copies on the host
     * build, so a not-yet-landed copy would read as zero (the h_fin amax=0 the first traced
     * run showed). The sync is trace-gated: a normal run pays nothing. */
    if (trc_on() && m.trc) {
        if (m.trc_done) {
            rad_stream_sync(rad_stream(c));
            trc_scan_print(m);
        }
        m.trc_done = true;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    trc_copy(c, m.trc, m.g.n_embd, 0, 0, m.b_x, batch->n_tok);   /* site x0: post-embedding */

    for (const Layer& l : m.layers) {
        l.attn.step(c, batch);
        l.mlp.step(c, batch);
        /* the per-layer scalar: `cur = cur * scalar`, the [1] bf16 over the residual stream,
         * applied at the end of each layer (after the mlp add) -- exactly the vLLM oracle's
         * `hidden_states = hidden_states * self.layer_scalar`. The checkpoint's values (0.03-0.99,
         * the vLLM text-only loader remaps `model.language_model.` -> `model.` and loads the
         * buffer, gemma4.py:1419/1644) are the trained weights: they rescale the residual toward
         * O(1) (the RMSNorms normalise magnitude but keep direction, so the input signal survives
         * through the branch, not the identity path). */
        RAD_ISSUE(c, l.op_scale, brows(m.b_x, batch->n_tok), RAD_W(l.w_scalar),
                  brows(m.b_x, batch->n_tok));
        trc_copy(c, m.trc, m.g.n_embd, l.attn.layer, 3, m.b_x, batch->n_tok);  /* site x_l: post residual */
    }

    RAD_ISSUE(c, m.op_final,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out), brows(m.b_h, batch->n_tok));
    trc_copy(c, m.trc, m.g.n_embd, (int)m.n_main, 4, m.b_h, batch->n_tok);     /* site h_fin: final-norm, lm_head input */

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

/* The reply format: gemma4_chat_format.h. */
const RadChatFormat* chat_format(const RadModelMeta*) { return chat_format_decl(); }

/* WHAT THIS CONTAINER'S HEAD IS WORTH SPECULATING TO, for `--num-speculative-tokens auto`. A serial
 * head names an operating point, not a trained depth: one more round is one more pass of the head,
 * and every depth still runs. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (has_assistant(meta)) {
        out->draft_depth = kHeadDepth;
        out->draft_depth_fixed = 0;
    }
    return RAD_OK;
}

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(gemma4_bf16, "gemma4", "compressed-tensors", "0.1.0",
                "Gemma 4 dense (text-only), w4a8 from its CT-INT4 (compressed-tensors) release: "
                "sliding-window and proportional-RoPE global attention over a GemmaRMSNorm stream "
                "with GELU feed-forward, per-layer scalar scaling and soft-capped logits")
RAD_ARCH_CHAT_FORMAT(gemma4_bf16)
RAD_ARCH_PROBE(gemma4_bf16)
#endif

}  /* namespace gemma4_bf16 */
