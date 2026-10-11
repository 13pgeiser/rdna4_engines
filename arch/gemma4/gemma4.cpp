/* gemma4 -- Gemma 4 text decoders (26B-A4B MoE, and the dense members of the family): a bf16
 * trunk, routed experts bf16 or block fp8 (see FP8 EXPERTS), and the DFlash block-diffusion
 * drafter.
 *
 * ============================== THE SHAPE (gemma-4-26B-A4B-it) ==============================
 *
 *   30 layers, hidden 2816, vocab 262144 (tied to the embedding), rms_norm_eps 1e-6
 *   layer_types: five `sliding_attention` then one `full_attention`, five times over
 *     sliding: 16 heads / 8 KV heads x 256, rope theta 1e4 over the whole head, window 1024
 *     full:    16 heads / 2 KV heads x 512, PROPORTIONAL rope theta 1e6 over a quarter of it,
 *              and K IS V (`attention_k_eq_v`): there is no v_proj, V is the k projection before
 *              its norm and its rotary
 *   attention scale 1.0 -- the q/k norms carry the temperature; q and k RMS-normed per head with a
 *   gain, v with none
 *   Gemma's sandwich norms, plain gains (x_hat * w, not 1 + w)
 *   the feed-forward is TWO ARMS BESIDE EACH OTHER, both reading the same residual:
 *     dense    pre_feedforward_layernorm -> gate|up 2112 (gelu-tanh) -> down
 *              -> post_feedforward_layernorm_1
 *     routed   router over rmsnorm(x) * scale * hidden^-0.5 -> softmax -> top 8 of 128 ->
 *              renormalised -> times per_expert_scale[expert];
 *              pre_feedforward_layernorm_2 -> 128 experts of 704 (gelu-tanh) ->
 *              post_feedforward_layernorm_2
 *     sum -> post_feedforward_layernorm -> residual add
 *   layer_scalar multiplies the residual stream at the end of every layer
 *   embedding scaled by sqrt(hidden) AS A bf16 (53.0, not 53.066); final norm; tied lm_head;
 *   final_logit_softcapping 30
 *
 * The reference is transformers' modeling_gemma4 (Gemma4TextDecoderLayer, Gemma4TextAttention,
 * Gemma4TextRouter, _compute_proportional_rope_parameters), read at 5.19.
 *
 * ============================== THE READINGS THAT STAY PLAUSIBLE WHEN WRONG ==============================
 *
 *   PROPORTIONAL ROPE IS NOT A PARTIAL ROTARY. It rotates the first 64 frequency PAIRS of a
 *   512-wide head at theta^(-2j/512) -- the exponent divides by the HEAD, and pair j is channels
 *   (j, j + 256) -- and leaves the other 192 pairs at frequency zero. `rope` with rotary_dim 128
 *   would pair (j, j + 64) at theta^(-2j/128), which is a different function that still produces
 *   fluent text. It is declared here as a full-width NeoX rotary whose frequencies past pair 63
 *   are divided by 1e30 -- the YaRN blend `rope` already carries, with the ramp a step at 63.5..64
 *   -- which evaluates to exactly the reference's angles below the step and to cos 1, sin ~1e-31
 *   above it. See decl_full_rope().
 *
 *   V ON A FULL LAYER IS THE UN-NORMED, UN-ROTATED K PROJECTION, RMS-normed with a unit gain. So
 *   the value norm reads `k` BEFORE the key norm writes it, which is the issue order below.
 *
 *   THE ROUTER DOES NOT READ THE FEED-FORWARD'S INPUT. It reads the residual stream through a
 *   GAINLESS norm, times its own `scale` vector, times hidden^-0.5 -- one `rmsnorm` with `scale`
 *   as the gain and `mult` the constant -- while the experts read pre_feedforward_layernorm_2's
 *   output and the dense arm pre_feedforward_layernorm's.
 *
 *   per_expert_scale MULTIPLIES THE RENORMALISED WEIGHT, after the top-k. It is gathered by expert
 *   id with `embed_lookup` (a 128 x 1 table) and multiplied in with `mul`.
 *
 *   THE TWO POST-NORMS OF THE ARMS ARE PER ARM, so under tensor parallel each arm's partial sum is
 *   reduced BEFORE its norm. The two partials sit in one [2T, n_embd] buffer -- dense rows first,
 *   routed rows after -- so one all-reduce carries both.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * Heads split by rank, o and down by column with an all-reduce, the vocabulary by row. The full
 * layers' two KV heads are REPLICATED past two ranks -- at four, ranks {0,1} hold KV head 0 and
 * {2,3} KV head 1, each reading its head's row span of k_proj -- the scheme rad_arch.h's geom_from
 * describes. Routed experts are EXPERT-PARALLEL (whole experts, 128/world a rank): a 704-wide
 * expert is 5.5 fp8 blocks and has no block-aligned width split. The router is replicated, so
 * every rank agrees on the routing without a collective and the arm's all-reduce sums the ranks'
 * experts.
 *
 * ============================== FP8 EXPERTS ==============================
 *
 * The experts are ~90% of the weight bytes and the decode step reads whichever were routed, so
 * their width is the step's bandwidth. Each projection is served as the container holds it --
 * bf16 by `moe_gemm`, block fp8 (the recipe's `rtn codes=fp8_e4m3 block=128x128 scale=bf16`) by
 * `moe_gemm_q` at fp8a8 over an E4M3 input -- per layer and per projection (expert_format):
 *
 *   gate_up  [2 * 704, 2816]: 1408 rows are whole 64-row tiles and eleven 128-row scale blocks
 *            (one straddles gate and up, which only shares a scale); K is 22 blocks. fp8 at the
 *            checkpoint's own width.
 *   down     [2816, 704]: K is the expert width and moe_gemm_q reads K in whole 128-column
 *            blocks, which 704 is not. It stays bf16 -- unless the checkpoint is padded to 768
 *            first, outside the engine: zero rows of gate and up give gelu(0) * 0 = 0 in the
 *            padded columns, which meet zero columns of down, so the function is unchanged, and
 *            then both projections are fp8 (data/recipes/gemma4-26b-a4b-df-pad768.recipe).
 *
 * ============================== WHAT THIS PLUGIN DOES NOT YET DECLARE ==============================
 *
 *   the vision tower (Gemma 4's, which arch/embgemma2 already declares for an encoder) -- this
 *   serves text; a container that carries a tower is served with --mm-max-patches 0 and the
 *   server refuses a media part by name;
 *   an fp8 attention or dense arm -- under a tenth of the weight bytes, and the dense arm's 2112
 *   is not whole 128-blocks either.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_gemma4_chat.h>

#include "gemma4_dflash.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4 {

using namespace rad::arch;

/* The fused router's two bounds, as rad_block_moe_fp8.h states them: the most routing slots
 * libr4d's sort serves in one workgroup (its kScatterSmall), and the skinny bf16 GEMM's decode
 * band, where the fused op's GEMM is that GEMM's own arithmetic. */
static const int64_t kMoeSortRows = 1024;
static const int64_t kMoeRouterFusedRows = 64;

/* ============================================================== metadata */
/* Gemma4ForConditionalGeneration nests the text model's config under `text_config`; a
 * Gemma4ForCausalLM checkpoint does not. rad-convert flattens nested keys with dots, so every key
 * is asked for in both spellings, text_config's first. */
static long long tc_geti(const RadModelMeta* m, const char* k, long long dflt) {
    const long long sentinel = -0x7FFFFFFFll;
    char key[160];
    std::snprintf(key, sizeof key, "text_config.%s", k);
    long long v = rad_meta_geti(m, key, sentinel);
    if (v == sentinel) v = rad_meta_geti(m, k, sentinel);
    return v == sentinel ? dflt : v;
}
static double tc_getf(const RadModelMeta* m, const char* k, double dflt) {
    char key[160];
    std::snprintf(key, sizeof key, "text_config.%s", k);
    if (rad_meta_gets(m, key, nullptr)) return rad_meta_getf(m, key, dflt);
    return rad_meta_getf(m, k, dflt);
}
static const char* tc_gets(const RadModelMeta* m, const char* k, const char* dflt) {
    char key[160];
    std::snprintf(key, sizeof key, "text_config.%s", k);
    const char* v = rad_meta_gets(m, key, nullptr);
    return v ? v : rad_meta_gets(m, k, dflt);
}
/* A JSON boolean reaches the meta as 1/0 or as true/false, depending on the converter's path. */
static bool tc_getb(const RadModelMeta* m, const char* k, bool dflt) {
    const char* s = tc_gets(m, k, nullptr);
    if (!s) return dflt;
    if (!std::strcmp(s, "true") || !std::strcmp(s, "True")) return true;
    if (!std::strcmp(s, "false") || !std::strcmp(s, "False") || !std::strcmp(s, "null")) return false;
    return std::atoll(s) != 0;
}

/* A value as a bf16 tensor holds it: round to nearest even on the top 16 bits of the f32. The
 * reference's embedding scale is a bf16 buffer, so sqrt(2816) = 53.066 multiplies as 53.0. */
static double bf16_round(double v) {
    float f = (float)v;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&f, &u, 4);
    return (double)f;
}

/* "1 6 11 17 22 27" -> the trunk layers whose output the drafter reads. Out-of-range entries are
 * dropped (llama_fp8.cpp says why). */
static int parse_ids(const char* s, int64_t n, int* out, int max) {
    if (!s) return 0;
    int k = 0;
    for (const char* p = s; *p && k < max;) {
        while (*p == ' ' || *p == ',' || *p == '[' || *p == ']') ++p;
        if (!*p) break;
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        p = e;
        if (v >= 0 && v < n) out[k++] = (int)v;
    }
    return k;
}

/* "sliding_attention full_attention ..." -> 1 for each full layer. */
static int parse_types(const char* s, std::vector<int>* full) {
    full->clear();
    if (!s) return RAD_E_INVAL;
    for (const char* p = s; *p;) {
        while (*p == ' ' || *p == ',' || *p == '[' || *p == ']' || *p == '"') ++p;
        if (!*p) break;
        const char* t = p;
        while (*p && *p != ' ' && *p != ',' && *p != ']' && *p != '"') ++p;
        const size_t len = (size_t)(p - t);
        if (len == 14 && !std::strncmp(t, "full_attention", 14))         full->push_back(1);
        else if (len == 17 && !std::strncmp(t, "sliding_attention", 17)) full->push_back(0);
        else {
            fprintf(stderr, "radiance: gemma4: layer_types names '%.*s'; this plugin knows "
                            "full_attention and sliding_attention\n", (int)len, t);
            return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

/* ============================================================== the model */

/* One attention kind: the sliding and the full layers differ in head width, KV heads, rotary
 * and window, and in whether V has a projection of its own. */
struct Kind {
    int64_t hd = 0;          /* head_dim */
    int64_t nh = 0;          /* query heads, THIS rank */
    int64_t nkv = 0;         /* KV heads, THIS rank */
    int64_t nkv_all = 0;
    bool    kv_repl = false; /* KV heads replicated across ranks (world > nkv_all) */
    int64_t kv_off = 0;      /* this rank's first KV row in the unsharded k/v projection */
    double  theta = 0.0;
    int64_t window = 0;      /* attn_paged's: key j visible to query i when i - j < window */
    int64_t rot_pairs = 0;   /* frequency pairs that rotate; < hd/2 is proportional rope */
    bool    k_eq_v = false;
    rad_kvgroup kv = 0;
    int64_t q()  const { return nh * hd; }
    int64_t kv_dim() const { return nkv * hd; }
};

struct Layer {
    int full = 0;
    /* attention */
    rad_weight w_anorm = 0, w_qkv = 0, w_q = 0, w_k = 0, w_qn = 0, w_kn = 0, w_o = 0, w_apost = 0;
    rad_op op_anorm = 0, op_qkv = 0, op_qknr = 0;
    rad_op op_q = 0, op_k = 0, op_vnorm = 0, op_qn = 0, op_kn = 0, op_qrope = 0, op_krope = 0;
    rad_op op_store = 0, op_attn = 0, op_o = 0, op_ar_o = 0, op_apost = 0, op_add_a = 0;
    /* dense arm */
    rad_weight w_fnorm = 0, w_gu = 0, w_dn = 0, w_fpost1 = 0;
    rad_op op_fnorm = 0, op_gu = 0, op_act = 0, op_dn = 0;
    /* routed arm. Each projection's experts are plain bf16 or block fp8, whichever the container
     * holds (expert_format). A bf16 projection's handles are its experts in order, one table; an
     * fp8 one's are the EVEN local experts then the ODD ones, codes in w_egu/w_edn and the scale
     * views in w_egus/w_edns, so each parity is one run (see declare). */
    rad_weight w_rscale = 0, w_router = 0, w_escale = 0, w_enorm = 0, w_fpost2 = 0;
    std::vector<rad_weight> w_egu, w_edn, w_egus, w_edns;
    bool gu_fp8 = false, dn_fp8 = false;
    rad_op op_rnorm = 0, op_router = 0, op_topk = 0, op_escale = 0, op_emul = 0, op_enorm = 0;
    rad_op op_scatter = 0, op_egu = 0, op_eact = 0, op_eq = 0, op_edn = 0, op_gather = 0;
    /* libr4d's fused forms of the router, its top-k and the sort, where they resolve: the pair
     * (`router_topk_scatter`) up to `topk_scatter_rows`, and the router GEMM in front of it
     * (`router_gemm_topk_scatter`) up to `router_fused_rows`. A pass past their rows issues the
     * separate ops, which are declared either way. */
    rad_op  op_topk_scatter = 0, op_router_fused = 0;
    int64_t topk_scatter_rows = 0, router_fused_rows = 0;
    /* the join */
    rad_weight w_fpost = 0, w_scalar = 0;
    rad_op op_ar_f = 0, op_fpost1 = 0, op_fpost2 = 0, op_fsum = 0, op_fpost = 0, op_add_f = 0;
    rad_op op_scale = 0;
    /* THE SANDWICH JOINS, FUSED (gemma4k): op_join_a is apost + add_a + fnorm, op_join_f the
     * routed layer's fpost1 + fpost2 + fsum + fpost + add_f + scale. Zero where not served, and
     * then the chain above is declared and issued instead. */
    rad_op op_join_a = 0, op_join_f = 0;
    /* op_join_f also writes the NEXT norm -- the following layer's input norm, or the final norm
     * after the last layer -- into `h`, with that norm's gain `w_next`; the norm it replaces is
     * then neither declared nor issued. */
    bool next_norm = false;
    rad_weight w_next = 0;
    /* the drafter's tap, on a layer the drafter reads (see declare) */
    rad_op op_tap = 0;
};

struct Model {
    Names   nm{""};
    Geom    g{};             /* the sliding kind's geometry, and everything model-wide */
    Kind    ks{}, kf{};      /* sliding, full */
    int64_t n_layer = 0, E = 0;
    int64_t n_ff = 0;        /* dense arm, THIS rank */
    bool    moe = false;
    int64_t n_expert = 0, n_local = 0, expert_base = 0, top_k = 0, n_ffe = 0;
    int64_t moe_rows = 0;    /* tokens a routed pass */
    double  escale = 1.0, softcap = 0.0;

    rad_buf b_x = 0, b_h = 0, b_o = 0;
    rad_buf b_qkv = 0, b_atts = 0;                         /* sliding */
    rad_buf b_qf = 0, b_kf = 0, b_vf = 0, b_attf = 0;      /* full */
    rad_buf b_ff2 = 0, b_gu = 0, b_act = 0;                /* the arms' join; the dense arm */
    rad_buf b_rn = 0, b_logit = 0, b_ids = 0, b_ew = 0, b_es = 0, b_hx = 0;
    rad_buf b_sorted = 0, b_eoff = 0, b_ecnt = 0, b_egu = 0, b_eact = 0, b_edn = 0;
    /* the experts' E4M3 inputs, declared only where some layer's projection is fp8: gate_up's
     * (the normed block input, a row a token) and down's (the gated product, a row a slot) */
    rad_buf b_hq = 0, b_hqs = 0, b_eq = 0, b_eqs = 0;
    int     n_gu_fp8 = 0, n_dn_fp8 = 0;   /* layers whose projection is fp8 */
    rad_buf b_hout = 0, b_logits = 0;

    rad_weight w_tok = 0, w_onorm = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_onorm = 0, op_gather = 0, op_logits = 0, op_cap = 0;

    enum { MAX_TAPS = 8 };
    int     n_taps = 0;
    int     tap_layer[MAX_TAPS] = {};
    rad_buf b_dfsrc = 0;

    Dflash1       df{};
    Dflash1::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    bool          have_df = false;

    std::vector<Layer> layers;
};

static Model g_model[MAX_RANKS];

/* ============================================================== geometry */

static int split_heads(Kind& k, int64_t nh_all, int world, int rank, const char* what) {
    if (nh_all % world) {
        fprintf(stderr, "radiance: gemma4: %lld %s query heads do not divide over %d ranks\n",
                (long long)nh_all, what, world);
        return RAD_E_INVAL;
    }
    k.nh = nh_all / world;
    if (world <= k.nkv_all) {
        if (k.nkv_all % world) {
            fprintf(stderr, "radiance: gemma4: %lld %s KV heads do not divide over %d ranks\n",
                    (long long)k.nkv_all, what, world);
            return RAD_E_INVAL;
        }
        k.nkv = k.nkv_all / world;
        k.kv_repl = false;
        k.kv_off = rank * k.nkv * k.hd;
    } else {
        /* REPLICATED: ranks [i*r, (i+1)*r) share KV head i, r = world / nkv_all. A rank's query
         * heads [rank*nh, (rank+1)*nh) all belong to that head because gqa = nh_all / nkv_all is a
         * multiple of nh exactly when world is a multiple of nkv_all -- which is checked. */
        if (world % k.nkv_all) {
            fprintf(stderr, "radiance: gemma4: %d ranks cannot replicate %lld %s KV heads; the "
                            "world must be a multiple of the KV head count\n", world,
                    (long long)k.nkv_all, what);
            return RAD_E_INVAL;
        }
        k.nkv = 1;
        k.kv_repl = true;
        k.kv_off = (rank / (world / k.nkv_all)) * k.hd;
    }
    return RAD_OK;
}

/* THE FULL LAYERS' ROTARY (see the header). `rope` rotates n_head + n_head_kv heads of its operand,
 * so a q-only tensor is n_head = nh, n_head_kv = 0 and a k-only one n_head = nkv. */
static rad_op decl_full_rope(RadBuilder* b, const Kind& k, int64_t rows, int64_t heads) {
    const int64_t half = k.hd / 2;
    if (k.rot_pairs >= half)
        return RAD_OP(b, "rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", k.hd),
                                 RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                 RAD_F64("theta", k.theta), RAD_F64("scale", 1.0),
                                 RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd)),
                      RAD_NOWEIGHTS);
    /* frequency(inv, i) = inv / factor for i >= high, inv for i <= low (rad_yarn.h); the
     * magnitude multiplies cos and sin and stays 1. */
    return RAD_OP(b, "rope",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", k.hd),
                             RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                             RAD_F64("theta", k.theta), RAD_F64("scale", 1.0),
                             RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd),
                             RAD_F64("yarn_factor", 1e30),
                             RAD_F64("yarn_low", (double)k.rot_pairs - 0.5),
                             RAD_F64("yarn_high", (double)k.rot_pairs),
                             RAD_F64("yarn_magnitude", 1.0)),
                  RAD_NOWEIGHTS);
}

/* ============================================================== the drafter */

static int declare_dflash(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg = m.g;
    dg.n_layer       = rad_meta_geti(meta, "draft.num_hidden_layers", 0);
    dg.head_dim      = rad_meta_geti(meta, "draft.head_dim", 0);
    dg.n_head_all    = rad_meta_geti(meta, "draft.num_attention_heads", 0);
    dg.n_head_kv_all = rad_meta_geti(meta, "draft.num_key_value_heads", 0);
    dg.n_ff_all      = rad_meta_geti(meta, "draft.intermediate_size", 0);
    dg.eps           = (float)rad_meta_getf(meta, "draft.rms_norm_eps", 0.0);
    dg.theta         = (float)rad_meta_getf(meta, "draft.rope_parameters.rope_theta",
                                            rad_meta_getf(meta, "draft.rope_theta", 0.0));
    dg.wadd = dg.wadd_qk = 0.0f;
    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: gemma4: the container's `draft.` config does not describe a "
                        "DFlash drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, ffn "
                        "%lld, rms_eps %g, rope_theta %g\n",
                (long long)dg.n_layer, (long long)dg.n_head_all, (long long)dg.n_head_kv_all,
                (long long)dg.head_dim, (long long)dg.n_ff_all, (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    if (rad_meta_geti(meta, "draft.hidden_size", dg.n_embd) != dg.n_embd) {
        fprintf(stderr, "radiance: gemma4: the drafter is %lld wide and the target %lld; it reads "
                        "the target's embedding and head, so it is not this target's drafter\n",
                (long long)rad_meta_geti(meta, "draft.hidden_size", 0), (long long)dg.n_embd);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(yarn_from(dg, meta, "draft.", dg.head_dim));
    int s;
    if ((s = divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads", &dg.n_head))    < 0) return s;
    if ((s = divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads", &dg.n_head_kv)) < 0) return s;
    if ((s = divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",   &dg.n_ff))      < 0) return s;
    const long long dvocab = rad_meta_geti(meta, "draft.vocab_size", dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: gemma4: the drafter claims a %lld-token vocabulary and the "
                        "target has %lld\n", dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }

    Dflash1::Config dc{};
    dc.n_taps = m.n_taps;
    dc.escale = m.escale;
    /* v1 states its block at the top level; a SpecForge config nests it under dflash_config. */
    long long trained = rad_meta_geti(meta, "draft.block_size", 0);
    if (trained <= 0) trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (trained < 2) {
        fprintf(stderr, "radiance: gemma4: the DFlash checkpoint states no block_size\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec + 1 != trained) {
        fprintf(stderr, "radiance: gemma4: the DFlash drafter was trained at block_size %lld, "
                        "which is %lld draft tokens, and this serve asks for %d. A block-diffusion "
                        "drafter denoises a fixed number of mask positions; run it at "
                        "--num-speculative-tokens %lld (or `auto`).\n",
                trained, trained - 1, ctx->max_spec, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    dc.block = trained;
    if (dg.n_layer > Dflash1::MAX_LAYERS) return RAD_E_UNSUPPORTED;
    {
        std::vector<int> full;
        const char* lt = rad_meta_gets(meta, "draft.layer_types", nullptr);
        if (lt) {
            RAD_ARCH_TRY(parse_types(lt, &full));
            if ((int64_t)full.size() != dg.n_layer) {
                fprintf(stderr, "radiance: gemma4: the drafter's layer_types names %zu layers and "
                                "it has %lld\n", full.size(), (long long)dg.n_layer);
                return RAD_E_INVAL;
            }
        } else {
            full.assign((size_t)dg.n_layer, 1);
        }
        bool any_swa = false;
        for (int64_t l = 0; l < dg.n_layer; ++l) {
            dc.full[l] = full[(size_t)l];
            any_swa = any_swa || !full[(size_t)l];
        }
        if (any_swa) {
            dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                      rad_meta_geti(meta, "draft.sliding_window", 0));
            if (!rad_meta_geti(meta, "draft.use_sliding_window", 1) || dc.window <= 0) {
                fprintf(stderr, "radiance: gemma4: the drafter names sliding layers and no "
                                "window\n");
                return RAD_E_INVAL;
            }
        }
    }

    /* ONE CACHE FOR ALL FIVE LAYERS, unbounded, because the full layer reads every position. The
     * sliding layers bound themselves with their attention op's `window`, so the declared drafter
     * window is 0 and the core stages the whole context. A window group for the four sliding
     * layers would cap their share at the window; it is the obvious saving once this runs. */
    RadKVGroupDecl dd{};
    dd.kind      = RAD_KV_FULL;
    dd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    dd.n_head_kv = dg.n_head_kv;
    dd.head_dim  = dg.head_dim;
    m.kv_draft = rad_decl_kv_group(b, nm.f("kv_draft"), &dd);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, m.kv_draft));
    dg.kv_dtype = rad_dtype_name(dd.dtype);

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr = dg.max_seqs * dc.steps();
    const int64_t tw = dc.n_taps * dg.n_embd;

    Dflash1::Wire& dw = m.dfw;
    dw.src   = m.b_dfsrc;
    dw.srcq  = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3, {dg.max_tok, tw});
    dw.srcs  = decl_b(b, nm.f("df_srcs"), RAD_F32,    {dg.max_tok, fp8_blocks(tw)});
    dw.c     = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.cq    = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs    = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.ck    = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv    = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x     = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h     = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.hq    = decl_b(b, nm.f("df_hq"),   RAD_F8E4M3, {qr, dg.n_embd});
    dw.hs    = decl_b(b, nm.f("df_hs"),   RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.q     = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k     = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v     = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn  = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.atq   = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats   = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn   = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    dw.ffq   = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs   = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.dlog  = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, dg.n_vocab});
    if (Dflash1::head_sharded(dg)) {
        const int64_t R = Dflash1::kTopR;
        dw.cand  = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, R});
        dw.unary = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, R});
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, R, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * R, 2});
    } else {
        dw.unary = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, 1});
    }
    if (!Dflash1::head_sharded(dg) && dg.world > 1) {
        fprintf(stderr, "radiance: gemma4: the vocabulary (%lld) does not split evenly into whole "
                        "16-row blocks over %d ranks, which the drafter's sharded head needs\n",
                (long long)dg.n_vocab_all, dg.world);
        return RAD_E_UNSUPPORTED;
    }
    /* THE DRAFT, one row a draft position -- [n_seq * steps, 1], which is [n_seq, steps] with a
     * pitch of `steps` -- and the engine reads it by the declaration below. */
    dw.tokens = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {sr, 1});

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft", dw));
    m.have_df = true;

    RadDrafterDecl drf{};
    drf.name           = "dflash";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;          /* the anchor predicts nothing: block = depth + 1 */
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = 0;                 /* the full layer reads everything; see the group */
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id",
                                                rad_meta_geti(meta, "draft.mask_token_id", -1));
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: gemma4: the DFlash checkpoint declares no mask_token_id\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    int nfull = 0;
    for (int64_t l = 0; l < dg.n_layer; ++l) nfull += dc.full[l];
    rad_note(b, "the DFlash drafter is declared: %lld layers at absolute %d..%d (%d full, %lld "
                "sliding over %lld causal), %lld heads / %lld kv x %lld, ffn %lld, reading the "
                "trunk at %d tap(s). ONE forward over %lld rows proposes %lld tokens; the head is "
                "the target's tied bf16 table, argmax %s",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1, nfull,
             (long long)(dg.n_layer - nfull), (long long)dc.window, (long long)dg.n_head,
             (long long)dg.n_head_kv, (long long)dg.head_dim, (long long)dg.n_ff, m.n_taps,
             (long long)dc.block, (long long)dc.steps(),
             Dflash1::head_sharded(dg) ? "per vocab shard, merged" : "over the whole vocabulary");
    return RAD_OK;
}

/* ============================================================== the experts' encoding */

/* WHAT ONE EXPERT PROJECTION IS STORED AS -- the container's entry, or in rad-convert the recipe's
 * plan for it (rad_weight_encoding) -- which picks the GEMM: plain bf16 is `moe_gemm`, block fp8
 * (fp8_e4m3 codes, a bf16 scale a 128x128 block) is `moe_gemm_q` at fp8a8. Anything else is
 * refused by name. A builder with no model behind it answers nothing, and the declared dtype is
 * then what is read: bf16. */
enum { EXP_BF16 = 0, EXP_FP8 = 1 };
static int expert_format(RadBuilder* b, const char* wn, int* fmt) {
    RadEncoding e{};
    *fmt = EXP_BF16;
    if (!weight_enc(b, wn, &e)) return RAD_OK;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return RAD_OK;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (rad_enc_is(&e, "affine") && e.n_planes == 2 && !e.transform[0] &&
        e.plane[0].dtype == RAD_F8E4M3 && sc && sc->dtype == RAD_BF16 &&
        sc->block[0] == RAD_FP8_BLOCK && sc->block[1] == RAD_FP8_BLOCK) {
        *fmt = EXP_FP8;
        return RAD_OK;
    }
    char en[256];
    rad_enc_format(&e, en, sizeof en);
    fprintf(stderr, "radiance: gemma4: '%s' is %s; the routed experts are served plain bf16 or "
                    "block fp8 (fp8_e4m3*bf16[128x128]). Convert with no rule for them, or with "
                    "`rtn codes=fp8_e4m3 block=128x128 scale=bf16`.\n", wn, en);
    return RAD_E_DTYPE;
}

/* ONE ENCODING A PROJECTION A LAYER, because one table of experts takes one: the first local
 * expert's decides, and every other has to match it. `what` is "gate_up" or "down". */
static int layer_expert_format(RadBuilder* b, Names& nm, int64_t l, int64_t base, int64_t n,
                               const char* what, int* fmt) {
    for (int64_t e = 0; e < n; ++e) {
        const char* wn = nm.f("blk.%lld.ffn_%s_exps.%lld.weight", (long long)l, what,
                              (long long)(base + e));
        int f = EXP_BF16;
        RAD_ARCH_TRY(expert_format(b, wn, &f));
        if (e == 0) { *fmt = f; continue; }
        if (f != *fmt) {
            fprintf(stderr, "radiance: gemma4: '%s' is %s and expert %lld of the same layer is %s; "
                            "one layer's %s experts take one encoding (a recipe rule for them "
                            "matches all of a layer's or none)\n", wn,
                    f == EXP_FP8 ? "block fp8" : "bf16", (long long)base,
                    *fmt == EXP_FP8 ? "block fp8" : "bf16", what);
            return RAD_E_DTYPE;
        }
    }
    return RAD_OK;
}

/* ============================================================== declare */

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->is_draft) {
        fprintf(stderr, "radiance: gemma4: this plugin has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* q = meta->quant ? meta->quant : "";
    if (*q) {
        fprintf(stderr, "radiance: gemma4 serves a container converted from the bf16 checkpoint "
                        "(google/gemma-4-*-it), with the recipe choosing what is fp8, and this one "
                        "declares '%s'.\n", q);
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    Geom&  g  = m.g;

    /* ---- what this plugin is NOT, refused by name. */
    if (tc_geti(meta, "hidden_size_per_layer_input", 0) > 0 ||
        tc_geti(meta, "num_kv_shared_layers", 0) > 0 || tc_getb(meta, "use_double_wide_mlp", false) ||
        tc_getb(meta, "attention_bias", false)) {
        fprintf(stderr, "radiance: gemma4: this checkpoint uses per-layer embeddings, shared KV "
                        "layers, a double-wide MLP or attention biases (the E2B/E4B shape); this "
                        "plugin serves the 26B-A4B and 31B shape\n");
        return RAD_E_UNSUPPORTED;
    }
    {
        const char* act = tc_gets(meta, "hidden_activation", "");
        if (!act || std::strcmp(act, "gelu_pytorch_tanh")) {
            fprintf(stderr, "radiance: gemma4: hidden_activation '%s'; this plugin serves "
                            "gelu_pytorch_tanh\n", act ? act : "");
            return RAD_E_UNSUPPORTED;
        }
    }

    /* ---- the geometry. Read, never derived: head_dim is not hidden / heads on this family. */
    m.n_layer = meta->n_layers > 0 ? meta->n_layers : tc_geti(meta, "num_hidden_layers", 0);
    m.E       = tc_geti(meta, "hidden_size", meta->n_embd);
    const int64_t nh_all  = tc_geti(meta, "num_attention_heads", meta->n_head);
    const int64_t vocab   = tc_geti(meta, "vocab_size", meta->n_vocab);
    const int64_t ff_all  = tc_geti(meta, "intermediate_size", meta->n_ff);
    const double  eps     = tc_getf(meta, "rms_norm_eps", meta->rms_eps);
    m.ks.hd      = tc_geti(meta, "head_dim", meta->head_dim);
    m.ks.nkv_all = tc_geti(meta, "num_key_value_heads", meta->n_head_kv);
    m.kf.hd      = tc_geti(meta, "global_head_dim", m.ks.hd);
    m.kf.nkv_all = tc_geti(meta, "num_global_key_value_heads", m.ks.nkv_all);
    m.kf.k_eq_v  = tc_getb(meta, "attention_k_eq_v", false);
    m.ks.theta   = tc_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    m.kf.theta   = tc_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    m.ks.window  = tc_geti(meta, "sliding_window", 0);   /* HF: q - k < window, attn_paged's rule */
    m.softcap    = tc_getf(meta, "final_logit_softcapping", 0.0);
    if (m.n_layer <= 0 || m.E <= 0 || nh_all <= 0 || vocab <= 0 || ff_all <= 0 || !(eps > 0) ||
        m.ks.hd <= 0 || m.ks.nkv_all <= 0 || m.kf.hd <= 0 || m.kf.nkv_all <= 0 ||
        !(m.ks.theta > 0) || !(m.kf.theta > 0) || m.ks.window <= 0) {
        fprintf(stderr, "radiance: gemma4: the container does not describe the model: layers "
                        "%lld, hidden %lld, heads %lld, vocab %lld, ffn %lld, eps %g, sliding %lld "
                        "x %lld kv theta %g window %lld, full %lld x %lld kv theta %g\n",
                (long long)m.n_layer, (long long)m.E, (long long)nh_all, (long long)vocab,
                (long long)ff_all, eps, (long long)m.ks.hd, (long long)m.ks.nkv_all, m.ks.theta,
                (long long)m.ks.window, (long long)m.kf.hd, (long long)m.kf.nkv_all, m.kf.theta);
        return RAD_E_INVAL;
    }
    /* The two rotaries. The sliding one is the ordinary full-head NeoX; the full one is read for
     * its type and its proportion and refused if it is anything this plugin does not compute. */
    {
        const char* st = tc_gets(meta, "rope_parameters.sliding_attention.rope_type", "default");
        const char* ft = tc_gets(meta, "rope_parameters.full_attention.rope_type", "default");
        const double prf = tc_getf(meta, "rope_parameters.full_attention.partial_rotary_factor", 1.0);
        const double ff  = tc_getf(meta, "rope_parameters.full_attention.factor", 1.0);
        if (std::strcmp(st, "default") || (std::strcmp(ft, "default") && std::strcmp(ft, "proportional")) ||
            ff != 1.0 || !(prf > 0.0) || prf > 1.0) {
            fprintf(stderr, "radiance: gemma4: rotary '%s' (sliding) and '%s' factor %g "
                            "proportion %g (full); this plugin computes default and "
                            "proportional at factor 1\n", st, ft, ff, prf);
            return RAD_E_UNSUPPORTED;
        }
        m.ks.rot_pairs = m.ks.hd / 2;
        /* int(rope_proportion * head_dim // 2), exactly as transformers computes it. */
        m.kf.rot_pairs = !std::strcmp(ft, "proportional")
                       ? (int64_t)std::floor(prf * (double)m.kf.hd / 2.0)
                       : m.kf.hd / 2;
        if (m.kf.rot_pairs <= 0) return RAD_E_INVAL;
    }

    std::vector<int> full;
    RAD_ARCH_TRY(parse_types(tc_gets(meta, "layer_types", nullptr), &full));
    if ((int64_t)full.size() != m.n_layer) {
        fprintf(stderr, "radiance: gemma4: layer_types names %zu layers and the model has %lld\n",
                full.size(), (long long)m.n_layer);
        return RAD_E_INVAL;
    }

    /* The model-wide Geom, which the shared helpers (and the drafter's copy) read. */
    g.n_layer = m.n_layer;
    g.n_embd  = m.E;
    g.head_dim = m.ks.hd;
    g.n_head_all = nh_all;
    g.n_head_kv_all = m.ks.nkv_all;
    g.n_ff_all = ff_all;
    g.n_vocab_all = vocab;
    g.max_tok  = ctx->max_tok;
    g.max_seqs = ctx->max_seqs;
    g.max_ctx  = ctx->max_ctx;
    g.max_spec = ctx->max_spec;
    g.max_logit_rows = ctx->max_seqs * (ctx->max_spec > 0 ? ctx->max_spec + 1 : 1);
    if (g.max_logit_rows > ctx->max_tok) g.max_logit_rows = ctx->max_tok;
    if (ctx->max_out_rows > g.max_logit_rows) g.max_logit_rows = ctx->max_out_rows;
    g.rank  = ctx->rank;
    g.world = ctx->world_size > 0 ? ctx->world_size : 1;
    g.wire_exact = ctx->tp_wire_lossy ? 0 : 1;
    g.wire_min_bytes = ctx->tp_wire_min_bytes;
    g.eps   = (float)eps;
    g.theta = (float)m.ks.theta;
    g.rope_scale = 1.0f;
    g.dtype = "bf16";
    g.act_dtype = RAD_BF16;
    if (g.max_tok <= 0 || g.max_seqs <= 0) return RAD_E_INVAL;
    const int W = g.world, R = g.rank;
    RAD_ARCH_TRY(split_heads(m.ks, nh_all, W, R, "sliding"));
    RAD_ARCH_TRY(split_heads(m.kf, nh_all, W, R, "full"));
    g.n_head = m.ks.nh;
    g.n_head_kv = m.ks.nkv;
    RAD_ARCH_TRY(divide_or_fail(ff_all, W, "intermediate_size", &m.n_ff));
    g.n_ff = m.n_ff;
    g.n_vocab   = shard_extent(vocab, W, R);
    g.vocab_off = shard_begin(vocab, W, R);
    if (g.n_vocab <= 0) return RAD_E_INVAL;
    const int64_t E = m.E, T = g.max_tok;

    /* ---- the routed arm's geometry. */
    m.moe = tc_getb(meta, "enable_moe_block", false);
    if (m.moe) {
        m.n_expert = tc_geti(meta, "num_experts", meta->n_expert);
        m.top_k    = tc_geti(meta, "top_k_experts", meta->n_expert_used);
        m.n_ffe    = tc_geti(meta, "moe_intermediate_size", 0);
        if (m.n_expert <= 0 || m.top_k <= 0 || m.top_k > m.n_expert || m.n_ffe <= 0) {
            fprintf(stderr, "radiance: gemma4: enable_moe_block with %lld experts, top %lld, of "
                            "%lld\n", (long long)m.n_expert, (long long)m.top_k,
                    (long long)m.n_ffe);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(divide_or_fail(m.n_expert, W, "num_experts", &m.n_local));
        m.expert_base = (int64_t)R * m.n_local;
        /* A pass bounds the [rows * top_k, ...] expert planes; 2048 tokens is one pass at the
         * default chunk. */
        m.moe_rows = T < 2048 ? T : 2048;
    }

    /* ---- the embedding scale and the rotary scaling the converter may have written. */
    m.escale = bf16_round(std::sqrt((double)E));
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.ks.hd));
    if (g.yarn.factor != 0.0f || g.rope_scale != 1.0f) {
        fprintf(stderr, "radiance: gemma4: --rope-scaling is not implemented for this family's "
                        "two rotaries\n");
        return RAD_E_UNSUPPORTED;
    }

    /* ---- the KV groups, one a geometry: the sliding layers' (256-wide heads) and the full
     * layers' (512).
     *
     * THE SLIDING LAYERS' GROUP IS RAD_KV_FULL, NOT RAD_KV_WINDOW, and the window lives on the
     * attention op (`window` 1024), which reads only the keys inside it. A WINDOW group is not
     * prefix-cached (core/mem/prefix.cpp: a hit's blocks do not describe a cache whose floor
     * moves). Before the scheduler knew that, a hit gave the sliding layers a table whose reused
     * prefix was never written for this request -- recycled blocks of other conversations, and a
     * reply that answers someone else's question; now a hit on a model with a windowed trunk group
     * reuses nothing and re-prefills (core/sched). As FULL the group is cached with the others and
     * a hit is kept, exactly; the drafter's KV gets no hole either. What it costs is the recycling
     * below the window, which buys nothing here: the planner gives every paged group the same
     * block count, so the full group bounds the tokens a card holds either way. */
    {
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL;
        d.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
        d.window = 0;
        d.n_head_kv = m.ks.nkv;
        d.head_dim = m.ks.hd;
        m.ks.kv = rad_decl_kv_group(b, nm.f("kv_swa"), &d);
        d.kind = RAD_KV_FULL;
        d.window = 0;
        d.n_head_kv = m.kf.nkv;
        d.head_dim = m.kf.hd;
        m.kf.kv = rad_decl_kv_group(b, nm.f("kv_full"), &d);
        g.kv_dtype = rad_dtype_name(d.dtype);
    }

    /* Gemma4ForConditionalGeneration keeps the text model under model.language_model;
     * Gemma4ForCausalLM (a text-only export) under model. */
    const char* archs = rad_meta_gets(meta, "architectures", "");
    const char* L = (archs && std::strstr(archs, "Gemma4ForCausalLM")) ? "model"
                                                                       : "model.language_model";

    /* ---- THE EXPERTS' NAME MAPS, AND FROM THEM WHAT EACH LAYER'S EXPERTS ARE, before any buffer:
     * an fp8 projection reads an E4M3 input that a bf16 one does not, and a plane nothing reads
     * is still charged by the planner. One stacked tensor a projection a layer, sliced by the
     * GLOBAL expert index -- this rank declares its window into the checkpoint's experts. */
    m.layers.resize((size_t)m.n_layer);
    if (m.moe) {
        const int64_t NL = m.n_local;
        for (int64_t l = 0; l < m.n_layer; ++l) {
            Layer& y = m.layers[(size_t)l];
            const long long ll = (long long)l;
            const char* sgu = nm.ckpt("%s.layers.%lld.experts.gate_up_proj", L, ll);
            const char* sdn = nm.ckpt("%s.layers.%lld.experts.down_proj", L, ll);
            for (int64_t e = 0; e < NL; ++e) {
                const long long ge = (long long)(m.expert_base + e);
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%lld.ffn_gate_up_exps.%lld.weight", ll, ge), sgu, (int)ge));
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%lld.ffn_down_exps.%lld.weight", ll, ge), sdn, (int)ge));
            }
            int fgu = EXP_BF16, fdn = EXP_BF16;
            RAD_ARCH_TRY(layer_expert_format(b, nm, l, m.expert_base, NL, "gate_up", &fgu));
            RAD_ARCH_TRY(layer_expert_format(b, nm, l, m.expert_base, NL, "down", &fdn));
            y.gu_fp8 = fgu == EXP_FP8;
            y.dn_fp8 = fdn == EXP_FP8;
            /* moe_gemm_q's geometry, refused here by name rather than as a kernel miss. Its
             * experts are two tables by parity, so a rank holds an even count of them. */
            if ((y.gu_fp8 || y.dn_fp8) && NL % 2) {
                fprintf(stderr, "radiance: gemma4: block fp8 experts come in pairs on a rank (the "
                                "fp8 grouped GEMM takes them as two tables by parity); this rank "
                                "holds %lld\n", (long long)NL);
                return RAD_E_INVAL;
            }
            if (y.gu_fp8 && ((2 * m.n_ffe) % 64 || E % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: gemma4: layer %lld's gate_up experts are block fp8 at "
                                "%lld x %lld; the fp8 grouped GEMM takes N in 64s and K in whole "
                                "%d-column blocks\n", ll, (long long)(2 * m.n_ffe), (long long)E,
                        RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            if (y.dn_fp8 && (E % 64 || m.n_ffe % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: gemma4: layer %lld's down experts are block fp8, and "
                                "the experts are %lld wide: the fp8 grouped GEMM reads K in whole "
                                "%d-column scale blocks. Keep `ffn_down_exps` bf16 (no recipe "
                                "rule for it), or pad the experts to a multiple of %d (zero rows "
                                "of gate and up, zero columns of down) and convert the padded "
                                "checkpoint.\n", ll,
                        (long long)m.n_ffe, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            m.n_gu_fp8 += y.gu_fp8;
            m.n_dn_fp8 += y.dn_fp8;
        }
    }

    /* ---- activations. */
    const int64_t qs = m.ks.q(), kvs = m.ks.kv_dim(), qf = m.kf.q(), kvf = m.kf.kv_dim();
    m.b_x    = decl_b(b, nm.f("x"),     RAD_BF16, {T, E});
    m.b_h    = decl_b(b, nm.f("h"),     RAD_BF16, {T, E});
    m.b_o    = decl_b(b, nm.f("o"),     RAD_BF16, {T, E});
    m.b_qkv  = decl_b(b, nm.f("qkv_s"), RAD_BF16, {T, qs + 2 * kvs});
    m.b_atts = decl_b(b, nm.f("att_s"), RAD_BF16, {T, qs});
    m.b_qf   = decl_b(b, nm.f("q_f"),   RAD_BF16, {T, qf});
    m.b_kf   = decl_b(b, nm.f("k_f"),   RAD_BF16, {T, kvf});
    m.b_vf   = decl_b(b, nm.f("v_f"),   RAD_BF16, {T, kvf});
    m.b_attf = decl_b(b, nm.f("att_f"), RAD_BF16, {T, qf});
    m.b_ff2  = decl_b(b, nm.f("ff2"),   RAD_BF16, {2 * T, E});
    m.b_gu   = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * m.n_ff});
    m.b_act  = decl_b(b, nm.f("ffn"),   RAD_BF16, {T, m.n_ff});
    if (m.moe) {
        const int64_t P = m.moe_rows, K = m.top_k;
        m.b_rn     = decl_b(b, nm.f("moe_rn"),     RAD_BF16, {T, E});
        m.b_logit  = decl_b(b, nm.f("moe_logits"), RAD_BF16, {T, m.n_expert});
        m.b_ids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,  {T, K});
        m.b_ew     = decl_b(b, nm.f("moe_w"),      RAD_BF16, {T, K});
        m.b_es     = decl_b(b, nm.f("moe_es"),     RAD_BF16, {T, K});
        if (m.n_gu_fp8 < m.n_layer)
            m.b_hx = decl_b(b, nm.f("moe_h"),      RAD_BF16, {T, E});
        if (m.n_gu_fp8 > 0) {
            m.b_hq  = decl_b(b, nm.f("moe_hq"),    RAD_F8E4M3, {T, E});
            m.b_hqs = decl_b(b, nm.f("moe_hqs"),   RAD_F32,    {T, fp8_blocks(E)});
        }
        m.b_sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32,  {P * K});
        m.b_eoff   = decl_b(b, nm.f("moe_eoff"),   RAD_I32,  {m.n_local + 1});
        m.b_ecnt   = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,  {m.n_local});
        m.b_egu    = decl_b(b, nm.f("moe_egu"),    RAD_BF16, {P * K, 2 * m.n_ffe});
        m.b_eact   = decl_b(b, nm.f("moe_eact"),   RAD_BF16, {P * K, m.n_ffe});
        if (m.n_dn_fp8 > 0) {
            m.b_eq  = decl_b(b, nm.f("moe_eq"),    RAD_F8E4M3, {P * K, m.n_ffe});
            m.b_eqs = decl_b(b, nm.f("moe_eqs"),   RAD_F32,    {P * K, fp8_blocks(m.n_ffe)});
        }
        m.b_edn    = decl_b(b, nm.f("moe_edn"),    RAD_BF16, {P * K, E});
    }
    m.b_hout = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, E});

    /* ---- the drafter's taps: declared when the container carries a DFlash drafter AND this serve
     * speculates. DFlash2 is refused by name -- its convolutions and selector are another block. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash2")) {
            fprintf(stderr, "radiance: gemma4: the container carries a DFlash2 drafter; this plugin "
                            "serves the DFlash (v1) block z-lab publishes for Gemma 4\n");
            return RAD_E_UNSUPPORTED;
        }
        if (da && std::strstr(da, "DFlash")) {
            const char* ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
            if (!ids) ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
            m.n_taps = parse_ids(ids, m.n_layer, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps <= 0) {
                fprintf(stderr, "radiance: gemma4: the drafter names no trunk layer this model "
                                "has; it is paired with a different target\n");
                return RAD_E_UNSUPPORTED;
            }
            m.b_dfsrc = decl_b(b, nm.f("dflash_src"), RAD_BF16, {T, (int64_t)m.n_taps * E});
        }
    }

    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges: ONE table, row-sharded with the vocabulary, read as the embedding
     * by `embed_lookup` and as the head by `logits_gemm` -- Gemma ties them, so storing the head
     * a second time would be 1.4 GiB for nothing. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", L)));
    m.w_tok   = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E}, RAD_ACCESS_VOCAB,
                       RAD_SHARD_ROW, grp_model());
    m.w_onorm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {E}, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                   RAD_INT("vocab_offset", g.vocab_off), RAD_F64("mult", m.escale)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    if (W > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", W), RAD_RANGE("numel", E, T * E),
                                          RAD_STR("dtype", "bf16"), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    /* ---- small declaration helpers, every one rows-banded at the step's height unless told. */
    auto rmsnorm = [&](int64_t rows, int64_t n, rad_weight w, double mult, rad_buf in, rad_buf out) {
        rad_op h = w ? RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16"),
                                         RAD_F64("wadd", 0.0), RAD_F64("mult", mult)),
                              RAD_WEIGHTS(w))
                     : RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                         RAD_F64("eps", eps), RAD_STR("dtype", "bf16"),
                                         RAD_F64("wadd", 0.0), RAD_F64("mult", mult)),
                              RAD_NOWEIGHTS);
        return rw(b, h, {in}, {out});
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w, rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                    RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(w)),
                  {in}, {out});
    };
    auto binop = [&](const char* op, int64_t rows, int64_t n, rad_buf a, rad_buf c, rad_buf out) {
        return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                              RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {a, c}, {out});
    };
    auto act_mul = [&](int64_t rows, int64_t n, rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "act_mul",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                    RAD_STR("act", "gelu_tanh"), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  {in}, {out});
    };
    auto allreduce = [&](int64_t max_numel, rad_buf buf) {
        return rw(b, RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", W), RAD_RANGE("numel", E, max_numel),
                                    RAD_STR("dtype", "bf16"), RAD_INT("exact", g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes)),
                         RAD_NOWEIGHTS),
                  {buf}, {buf});
    };
    auto attn = [&](const Kind& k, rad_buf q, rad_buf out) {
        return rw(b, RAD_OP(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", k.hd),
                                    RAD_INT("gqa", k.nh / k.nkv), RAD_INT("causal", 1),
                                    RAD_INT("window", k.window), RAD_INT("n_head", k.nh),
                                    RAD_INT("block_size", RAD_KV_BLOCK),
                                    RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                    RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype),
                                    RAD_F64("scale", 1.0)),
                         RAD_NOWEIGHTS),
                  {q}, {out});
    };
    auto kv_store = [&](const Kind& k, rad_buf kb, rad_buf vb) {
        return rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.hd),
                                    RAD_INT("n_head_kv", k.nkv),
                                    RAD_INT("block_size", rad_kv_block_size(b, k.kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                  {kb, vb}, {});
    };

    /* THE SANDWICH JOINS: asked of the kernel libraries as one op each (gemma4k serves them) and
     * declared as the chain where nothing does -- or where RADIANCE_GEMMA4_FUSE=0 asks for the
     * chain, the A/B switch. The fused kernels round where the chain rounds (gemma4k/g4k_norm.hip).
     * libref has no row for the joins on purpose, so "nothing serves them" stays reachable. */
    const char* fuse_env = std::getenv("RADIANCE_GEMMA4_FUSE");
    const bool fuse_joins = !(fuse_env && fuse_env[0] == '0');
    int n_joins = 0;

    const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
    /* EVERY LAYER'S INPUT NORM GAIN, AHEAD OF THE LAYERS: a feed-forward join writes the next
     * layer's input norm (join_add_scale's `y`) and so names that layer's gain before the layer
     * itself is declared. */
    for (int64_t l = 0; l < m.n_layer; ++l) {
        const long long ll = (long long)l;
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_norm.weight", ll),
                              nm.ckpt("%s.layers.%lld.input_layernorm.weight", L, ll)));
        m.layers[(size_t)l].w_anorm = decl_w(b, nm.f("blk.%lld.attn_norm.weight", ll), RAD_F32,
                                             {E}, A, N, grp_layer((int)l));
    }

    for (int64_t l = 0; l < m.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.full = full[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const long long ll = (long long)l;
        const int li = (int)l;
        const RadWeightGroup gl = grp_layer(li);
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%lld.%s", L, ll, s); };
        RAD_ARCH_TRY(rad_bind_layer_kv(b, li, k.kv));

        /* ------------------------------------------------ attention: weights */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_q_norm.weight", ll), src("self_attn.q_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_k_norm.weight", ll), src("self_attn.k_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_output.weight", ll), src("self_attn.o_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_post_norm.weight", ll),
                              src("post_attention_layernorm.weight")));
        /* bf16 gains: `qk_norm_rope` reads its gains at the activation's width. */
        y.w_qn    = decl_w(b, nm.f("blk.%lld.attn_q_norm.weight", ll), RAD_BF16, {k.hd}, A, N, gl);
        y.w_kn    = decl_w(b, nm.f("blk.%lld.attn_k_norm.weight", ll), RAD_BF16, {k.hd}, A, N, gl);
        y.w_o     = decl_w(b, nm.f("blk.%lld.attn_output.weight", ll), RAD_BF16, {E, k.q()}, A,
                           RAD_SHARD_COL, gl);
        y.w_apost = decl_w(b, nm.f("blk.%lld.attn_post_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        if (!y.full) {
            /* ONE PROJECTION, q|k|v, three row parts: every rank takes its slice of each. */
            RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.attn_qkv.weight", ll),
                                    { src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                                      src("self_attn.v_proj.weight") }));
            if (!k.kv_repl) {
                y.w_qkv = decl_w(b, nm.f("blk.%lld.attn_qkv.weight", ll), RAD_BF16,
                                 {k.q() + 2 * k.kv_dim(), E}, A, RAD_SHARD_ROW, gl, 0,
                                 {k.q(), k.kv_dim(), k.kv_dim()});
            } else {
                y.w_qkv = decl_w(b, nm.f("blk.%lld.attn_qkv.weight", ll), RAD_BF16,
                                 {k.q() + 2 * k.kv_dim(), E}, A, RAD_SHARD_ROW, gl, 0,
                                 {k.q(), k.kv_dim(), k.kv_dim()});
                const int64_t lo[3] = { R * k.q(), k.kv_off, k.kv_off };
                const int64_t hi[3] = { lo[0] + k.q(), lo[1] + k.kv_dim(), lo[2] + k.kv_dim() };
                const int64_t tot[3] = { nh_all * k.hd, k.nkv_all * k.hd, k.nkv_all * k.hd };
                RAD_ARCH_TRY(rad_weight_shard_span_parts(b, y.w_qkv, lo, hi, tot, 3));
            }
        } else {
            if (!k.k_eq_v) {
                fprintf(stderr, "radiance: gemma4: full-attention layer %lld has a v_proj "
                                "(attention_k_eq_v false); this plugin serves K = V\n", ll);
                return RAD_E_UNSUPPORTED;
            }
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_q.weight", ll), src("self_attn.q_proj.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_k.weight", ll), src("self_attn.k_proj.weight")));
            y.w_q = decl_w(b, nm.f("blk.%lld.attn_q.weight", ll), RAD_BF16, {k.q(), E}, A,
                           RAD_SHARD_ROW, gl);
            y.w_k = decl_w(b, nm.f("blk.%lld.attn_k.weight", ll), RAD_BF16, {k.kv_dim(), E}, A,
                           RAD_SHARD_ROW, gl);
            if (k.kv_repl)
                RAD_ARCH_TRY(rad_weight_shard_span(b, y.w_k, k.kv_off, k.kv_off + k.kv_dim()));
        }

        /* ------------------------------------------------ attention: ops, in issue order (the
         * attention op is declared ahead of its kv_store, which reads the block size off it). */
        if (!(l > 0 && m.layers[(size_t)l - 1].next_norm))
            y.op_anorm = rmsnorm(T, E, y.w_anorm, 1.0, m.b_x, m.b_h);
        if (!y.full) {
            y.op_qkv  = gemm(T, k.q() + 2 * k.kv_dim(), E, y.w_qkv, m.b_h, m.b_qkv);
            y.op_qknr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.hd),
                                           RAD_INT("n_head", k.nh), RAD_INT("n_head_kv", k.nkv),
                                           RAD_F64("theta", k.theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "neox"), RAD_INT("rotary_dim", k.hd),
                                           RAD_F64("wadd", 0.0), RAD_INT("v_norm", 1)),
                                RAD_WEIGHTS(y.w_qn, y.w_kn)),
                            {m.b_qkv}, {m.b_qkv});
            /* Q IS READ WHERE THE PROJECTION LEFT IT, a column slice of the fused [q|k|v] row:
             * gemma4k's attention takes q at any row pitch, so there is no copy into a dense block
             * -- a launch and a [T, q] round trip a sliding layer. libr4d's attention reads q only
             * dense and refuses the slice by name (RAD_E_STRIDE), which is one more reason the
             * order is `--kernels gemma4k,libr4d` for this model. */
            y.op_attn  = attn(k, m.b_qkv, m.b_atts);
            y.op_store = kv_store(k, m.b_qkv, m.b_qkv);
            y.op_o     = gemm(T, E, k.q(), y.w_o, m.b_atts, m.b_o);
        } else {
            y.op_q     = gemm(T, k.q(), E, y.w_q, m.b_h, m.b_qf);
            y.op_k     = gemm(T, k.kv_dim(), E, y.w_k, m.b_h, m.b_kf);
            /* V FIRST: the unit-gain norm of the RAW k projection, before the key norm writes k. */
            y.op_vnorm = rmsnorm(T * k.nkv, k.hd, 0, 1.0, m.b_kf, m.b_vf);
            y.op_qn    = rmsnorm(T * k.nh, k.hd, y.w_qn, 1.0, m.b_qf, m.b_qf);
            y.op_kn    = rmsnorm(T * k.nkv, k.hd, y.w_kn, 1.0, m.b_kf, m.b_kf);
            y.op_qrope = rw(b, decl_full_rope(b, k, T, k.nh), {m.b_qf}, {m.b_qf});
            y.op_krope = rw(b, decl_full_rope(b, k, T, k.nkv), {m.b_kf}, {m.b_kf});
            y.op_attn  = attn(k, m.b_qf, m.b_attf);
            y.op_store = kv_store(k, m.b_kf, m.b_vf);
            y.op_o     = gemm(T, E, k.q(), y.w_o, m.b_attf, m.b_o);
        }
        if (W > 1) y.op_ar_o = allreduce(T * E, m.b_o);
        /* The post-norm AFTER the reduction: it is a norm of the whole sum. Declared below, with the
         * dense arm's pre-norm, as one `norm_add_norm` where gemma4k serves it. */

        /* ------------------------------------------------ the dense arm */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm.weight", ll), src("pre_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.ffn_gate_up.weight", ll),
                                { src("mlp.gate_proj.weight"), src("mlp.up_proj.weight") }));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_down.weight", ll), src("mlp.down_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm.weight", ll),
                              src("post_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.layer_scalar", ll), src("layer_scalar")));
        y.w_fnorm  = decl_w(b, nm.f("blk.%lld.ffn_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        y.w_gu     = decl_w(b, nm.f("blk.%lld.ffn_gate_up.weight", ll), RAD_BF16, {2 * m.n_ff, E},
                            A, RAD_SHARD_ROW, gl, 0, {m.n_ff, m.n_ff});
        y.w_dn     = decl_w(b, nm.f("blk.%lld.ffn_down.weight", ll), RAD_BF16, {E, m.n_ff}, A,
                            RAD_SHARD_COL, gl);
        y.w_fpost  = decl_w(b, nm.f("blk.%lld.ffn_post_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        /* bf16 and one element: the reference multiplies the bf16 stream by it. */
        y.w_scalar = decl_w(b, nm.f("blk.%lld.layer_scalar", ll), RAD_BF16, {1}, A, N, gl);

        if (fuse_joins)
            y.op_join_a = rw(b, RAD_OP(b, "norm_add_norm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                               RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(y.w_apost, y.w_fnorm)),
                             {m.b_o, m.b_x}, {m.b_x, m.b_h});
        if (!y.op_join_a) {
            y.op_apost = rmsnorm(T, E, y.w_apost, 1.0, m.b_o, m.b_o);
            y.op_add_a = binop("add", T, E, m.b_x, m.b_o, m.b_x);
            y.op_fnorm = rmsnorm(T, E, y.w_fnorm, 1.0, m.b_x, m.b_h);
        }
        y.op_gu    = gemm(T, 2 * m.n_ff, E, y.w_gu, m.b_h, m.b_gu);
        y.op_act   = act_mul(T, m.n_ff, m.b_gu, m.b_act);
        /* The dense partial into rows [0, T) of the join. */
        y.op_dn    = gemm(T, E, m.n_ff, y.w_dn, m.b_act, m.b_ff2);

        /* ------------------------------------------------ the routed arm */
        if (m.moe) {
            const int64_t P = m.moe_rows, K = m.top_k, NL = m.n_local;
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm1.weight", ll),
                                  src("post_feedforward_layernorm_1.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_post_norm2.weight", ll),
                                  src("post_feedforward_layernorm_2.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm_exps.weight", ll),
                                  src("pre_feedforward_layernorm_2.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_gate_inp.weight", ll), src("router.proj.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_gate_inp.scale", ll), src("router.scale")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_exp_scale", ll), src("router.per_expert_scale")));
            y.w_fpost1 = decl_w(b, nm.f("blk.%lld.ffn_post_norm1.weight", ll), RAD_F32, {E}, A, N, gl);
            y.w_fpost2 = decl_w(b, nm.f("blk.%lld.ffn_post_norm2.weight", ll), RAD_F32, {E}, A, N, gl);
            y.w_enorm  = decl_w(b, nm.f("blk.%lld.ffn_norm_exps.weight", ll), RAD_F32, {E}, A, N, gl);
            /* THE ROUTER IS REPLICATED (rad_block_moe_fp8.h argues it): every rank routes every
             * token the same way from the same residual stream. */
            y.w_router = decl_w(b, nm.f("blk.%lld.ffn_gate_inp.weight", ll), RAD_BF16,
                                {m.n_expert, E}, A, N, gl);
            y.w_rscale = decl_w(b, nm.f("blk.%lld.ffn_gate_inp.scale", ll), RAD_F32, {E}, A, N, gl);
            y.w_escale = decl_w(b, nm.f("blk.%lld.ffn_exp_scale", ll), RAD_BF16, {m.n_expert}, A,
                                N, gl);
            /* THE EXPERTS: whole experts, this rank's run of them (their name maps and encodings
             * were settled before the buffers). Each projection's handles are declared as the
             * runs its GEMM's tables are: bf16, one run in expert order; block fp8, the codes of
             * the even local experts, then the odd ones', then the scale views in the same two
             * runs -- `moe_gemm_q` takes a projection's experts as two tables by parity, and a
             * table is a run of consecutively declared handles. The codes and the scale plane are
             * two views of one stored weight and one movement unit (slot 0 gate_up, 1 down). */
            auto experts = [&](bool fp8, const char* what, int slot, int64_t rows, int64_t cols,
                               std::vector<rad_weight>& wc, std::vector<rad_weight>& ws) {
                wc.assign((size_t)NL, 0);
                ws.clear();
                if (!fp8) {
                    for (int64_t e = 0; e < NL; ++e) {
                        const long long ge = (long long)(m.expert_base + e);
                        wc[(size_t)e] = decl_w(b, nm.f("blk.%lld.ffn_%s_exps.%lld.weight", ll, what, ge),
                                               RAD_BF16, {rows, cols}, RAD_ACCESS_CONDITIONAL, N,
                                               grp_expert_slot(li, (int)e, slot));
                        if (!wc[(size_t)e]) return RAD_E_INVAL;
                    }
                    return RAD_OK;
                }
                ws.assign((size_t)NL, 0);
                const int64_t half = NL / 2;
                for (int pass = 0; pass < 2; ++pass)          /* codes, then scales */
                    for (int par = 0; par < 2; ++par)         /* even, then odd */
                        for (int64_t i = 0; i < half; ++i) {
                            const int64_t e = 2 * i + par;
                            const long long ge = (long long)(m.expert_base + e);
                            const char* wn = nm.f("blk.%lld.ffn_%s_exps.%lld.weight", ll, what, ge);
                            rad_weight& h = (pass ? ws : wc)[(size_t)(par * half + i)];
                            h = pass ? decl_view(b, nm.f("blk.%lld.ffn_%s_exps.%lld.scale", ll, what, ge),
                                                 wn, "scale", RAD_BF16,
                                                 {fp8_blocks(rows), fp8_blocks(cols)},
                                                 RAD_ACCESS_CONDITIONAL, N,
                                                 grp_expert_slot(li, (int)e, slot))
                                     : decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, {rows, cols},
                                                 RAD_ACCESS_CONDITIONAL, N,
                                                 grp_expert_slot(li, (int)e, slot));
                            if (!h) return RAD_E_INVAL;
                        }
                return RAD_OK;
            };
            RAD_ARCH_TRY(experts(y.gu_fp8, "gate_up", 0, 2 * m.n_ffe, E, y.w_egu, y.w_egus));
            RAD_ARCH_TRY(experts(y.dn_fp8, "down", 1, E, m.n_ffe, y.w_edn, y.w_edns));
            /* An fp8 projection's op takes its weights in its table operands' order: even codes,
             * even scales, odd codes, odd scales. */
            auto tables = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
                const size_t h = wc.size() / 2;
                std::vector<rad_weight> t;
                t.insert(t.end(), wc.begin(), wc.begin() + (long)h);
                t.insert(t.end(), ws.begin(), ws.begin() + (long)h);
                t.insert(t.end(), wc.begin() + (long)h, wc.end());
                t.insert(t.end(), ws.begin() + (long)h, ws.end());
                return t;
            };

            y.op_rnorm  = rmsnorm(P, E, y.w_rscale, 1.0 / std::sqrt((double)E), m.b_x, m.b_rn);
            y.op_router = gemm(P, m.n_expert, E, y.w_router, m.b_rn, m.b_logit);
            y.op_topk   = rw(b, RAD_OP(b, "router_topk",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_expert", m.n_expert),
                                             RAD_INT("top_k", K), RAD_INT("norm", 1),
                                             RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                              {m.b_logit}, {m.b_ids, m.b_ew});
            /* per_expert_scale[expert], gathered as a [n_expert, 1] table by the ids, then the
             * routing weight times it. */
            y.op_escale = rw(b, RAD_OP(b, "embed_lookup",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P * K), RAD_INT("n_embd", 1),
                                             RAD_INT("n_vocab", m.n_expert), RAD_STR("dtype", "bf16")),
                                  RAD_WEIGHTS(y.w_escale)),
                              {m.b_ids}, {m.b_es});
            y.op_emul   = binop("mul", P, K, m.b_ew, m.b_es, m.b_ew);
            /* THE EXPERTS' INPUT: pre_feedforward_layernorm_2 of the stream -- for an fp8 gate_up,
             * normed and quantised in one pass (`rmsnorm_quant_fp8`), with no bf16 copy, since the
             * fp8 GEMM reads only the codes and their scales. */
            if (y.gu_fp8) {
                y.op_enorm = RAD_OP(b, "rmsnorm_quant_fp8",
                                    RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", E),
                                               RAD_F64("eps", eps), RAD_INT("group", RAD_FP8_BLOCK),
                                               RAD_STR("dtype", "bf16"), RAD_F64("wadd", 0.0)),
                                    RAD_WEIGHTS(y.w_enorm));
                y.op_enorm = rw(b, y.op_enorm, {m.b_x}, {m.b_hq, m.b_hqs});
            } else {
                y.op_enorm = rmsnorm(P, E, y.w_enorm, 1.0, m.b_x, m.b_hx);
            }
            y.op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                                   RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_expert", NL),
                                              RAD_INT("top_k", K),
                                              RAD_INT("expert_base", m.expert_base)),
                                   RAD_NOWEIGHTS),
                               {m.b_ids}, {m.b_sorted, m.b_eoff, m.b_ecnt});
            /* THE ROUTER, ITS TOP-K AND THE SORT IN ONE LAUNCH at decode heights. At a decode step
             * each of the three is one small workgroup's work behind a full dispatch, and libr4d
             * fuses them: `router_topk_scatter` for the pair, bounded by the sort's single-workgroup
             * form (kMoeSortRows routing slots), and `router_gemm_topk_scatter` with the router's
             * GEMM in front, up to the skinny GEMM's decode band -- byte-identical to the launches
             * they stand in for (docs/OPS.md), so the routing does not move. The per-expert scale
             * and the experts' input norm sit between them in the issue order and read only the ids
             * and weights the fused op writes. Declared here, after every buffer they name: a late
             * declaration only lengthens a buffer's planned life (rad_block_moe_fp8.h). */
            {
                const int64_t sort_rows = P * K <= kMoeSortRows ? P : kMoeSortRows / K;
                if (sort_rows > 0)
                    y.op_topk_scatter = rw(b, RAD_OP(b, "router_topk_scatter",
                                               RAD_PARAMS(RAD_RANGE("M", 1, sort_rows),
                                                          RAD_INT("n_expert", m.n_expert),
                                                          RAD_INT("n_local", NL), RAD_INT("top_k", K),
                                                          RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
                                                          RAD_INT("expert_base", m.expert_base)),
                                               RAD_NOWEIGHTS),
                                           {m.b_logit}, {m.b_ids, m.b_ew, m.b_sorted, m.b_eoff, m.b_ecnt});
                if (y.op_topk_scatter) {
                    y.topk_scatter_rows = sort_rows;
                    const int64_t fr = sort_rows < kMoeRouterFusedRows ? sort_rows : kMoeRouterFusedRows;
                    y.op_router_fused = rw(b, RAD_OP(b, "router_gemm_topk_scatter",
                                               RAD_PARAMS(RAD_RANGE("M", 1, fr), RAD_INT("K", E),
                                                          RAD_INT("n_expert", m.n_expert),
                                                          RAD_INT("n_local", NL), RAD_INT("top_k", K),
                                                          RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
                                                          RAD_INT("expert_base", m.expert_base)),
                                               RAD_WEIGHTS(y.w_router)),
                                           {m.b_rn},
                                           {m.b_logit, m.b_ids, m.b_ew, m.b_sorted, m.b_eoff, m.b_ecnt});
                    if (y.op_router_fused) y.router_fused_rows = fr;
                }
            }
            /* THE TWO GROUPED GEMMs DISAGREE ABOUT a_order, and must: gate_up reads the block
             * input, a row a TOKEN, gathered through sorted_tok; down reads gate_up's product,
             * already a row a (token, slot) in the scatter's order (rad_block_moe_fp8.h). */
            if (y.gu_fp8) {
                const std::vector<rad_weight> tab = tables(y.w_egu, y.w_egus);
                y.op_egu = rw(b, rad_decl_op(b, "moe_gemm_q",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", 2 * m.n_ffe),
                                                RAD_INT("K", E), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_INT("group", RAD_FP8_BLOCK),
                                                RAD_STR("a_order", "token"), RAD_STR("dtype", "fp8a8")),
                                     tab.data(), (int)tab.size()),
                              {m.b_hq, m.b_hqs, m.b_sorted, m.b_eoff}, {m.b_egu});
            } else {
                y.op_egu = rw(b, rad_decl_op(b, "moe_gemm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", 2 * m.n_ffe),
                                                RAD_INT("K", E), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_STR("a_order", "token"),
                                                RAD_STR("dtype", "bf16")),
                                     y.w_egu.data(), (int)NL),
                              {m.b_hx, m.b_sorted, m.b_eoff}, {m.b_egu});
            }
            /* gelu_tanh(gate) * up over the two column halves. An fp8 down then quantises the
             * product on its own launch: libr4d's fused gated quantiser is compiled for silu. */
            y.op_eact = act_mul(P * K, m.n_ffe, m.b_egu, m.b_eact);
            if (y.dn_fp8) {
                y.op_eq = rw(b, RAD_OP(b, "quant_act_fp8",
                                    RAD_PARAMS(RAD_RANGE("M", 1, P * K), RAD_INT("n", m.n_ffe),
                                               RAD_INT("group", RAD_FP8_BLOCK),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                             {m.b_eact}, {m.b_eq, m.b_eqs});
                const std::vector<rad_weight> tab = tables(y.w_edn, y.w_edns);
                y.op_edn = rw(b, rad_decl_op(b, "moe_gemm_q",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", E),
                                                RAD_INT("K", m.n_ffe), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_INT("group", RAD_FP8_BLOCK),
                                                RAD_STR("a_order", "sorted"), RAD_STR("dtype", "fp8a8")),
                                     tab.data(), (int)tab.size()),
                              {m.b_eq, m.b_eqs, m.b_sorted, m.b_eoff}, {m.b_edn});
            } else {
                y.op_edn = rw(b, rad_decl_op(b, "moe_gemm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", E),
                                                RAD_INT("K", m.n_ffe), RAD_INT("n_expert", NL),
                                                RAD_INT("top_k", K), RAD_STR("a_order", "sorted"),
                                                RAD_STR("dtype", "bf16")),
                                     y.w_edn.data(), (int)NL),
                              {m.b_eact, m.b_sorted, m.b_eoff}, {m.b_edn});
            }
            {
                rad_buf rd[3] = { m.b_edn, m.b_ew, m.b_sorted };
                rad_buf wr[1] = { m.b_ff2 };
                y.op_gather = RAD_OP(b, "moe_gather",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", E),
                                                RAD_INT("top_k", K), RAD_STR("dtype", "bf16"),
                                                RAD_STR("act", "none")),
                                     RAD_NOWEIGHTS);
                if (!y.op_gather) return RAD_E_INVAL;
                rad_op_reads(b, y.op_gather, rd, 3);
                rad_op_writes(b, y.op_gather, wr, 1);
            }
            if (!y.op_enorm || !y.op_egu || !y.op_edn || (y.dn_fp8 && !y.op_eq)) return RAD_E_INVAL;
            if (W > 1) y.op_ar_f = allreduce(2 * T * E, m.b_ff2);
            if (fuse_joins)
            {
                /* ...and the norm that reads the x it writes: the next layer's input norm, or the
                 * final norm after the last layer -- one launch and one read of x the less. */
                const rad_weight w_next = l + 1 < m.n_layer ? m.layers[(size_t)l + 1].w_anorm
                                                            : m.w_onorm;
                y.op_join_f = rw(b, RAD_OP(b, "join_add_scale",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                                   RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                                        RAD_WEIGHTS(y.w_fpost1, y.w_fpost2, y.w_fpost, y.w_scalar,
                                                    w_next)),
                                 {m.b_ff2, m.b_x}, {m.b_x, m.b_h});
                if (y.op_join_f) { y.next_norm = true; y.w_next = w_next; }
            }
            if (!y.op_join_f) {
                y.op_fpost1 = rmsnorm(T, E, y.w_fpost1, 1.0, m.b_ff2, m.b_ff2);
                y.op_fpost2 = rmsnorm(T, E, y.w_fpost2, 1.0, m.b_ff2, m.b_ff2);
                y.op_fsum   = binop("add", T, E, m.b_ff2, m.b_ff2, m.b_h);
                y.op_fpost  = rmsnorm(T, E, y.w_fpost, 1.0, m.b_h, m.b_h);
            }
        } else {
            if (W > 1) y.op_ar_f = allreduce(T * E, m.b_ff2);
            y.op_fpost = rmsnorm(T, E, y.w_fpost, 1.0, m.b_ff2, m.b_h);
        }
        if (!y.op_join_f) {
            y.op_add_f = binop("add", T, E, m.b_x, m.b_h, m.b_x);
            y.op_scale = rw(b, RAD_OP(b, "mul_weight",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                              RAD_STR("dtype", "bf16"), RAD_INT("scalar", 1)),
                                   RAD_WEIGHTS(y.w_scalar)),
                               {m.b_x}, {m.b_x});
        }
        n_joins += (y.op_join_a ? 1 : 0) + (y.op_join_f ? 1 : 0);

        /* THE TAP: a row copy of the layer's output into the column slice a tap owns, DECLARED
         * HERE, after the layer's last op, because that is where step() issues it. The buffer
         * planner's liveness is declaration order. One tap op declared after the whole trunk made
         * `dflash_src` look born at the end of it, so the plan was free to put it on top of
         * buffers every layer writes -- and at --max-num-seqs 16 it did, under `moe_edn`, the
         * routed experts' output: every MoE layer after a tap overwrote what the drafter reads,
         * and on gemma-4-26B-A4B-it it accepted 2% of its drafts with the reply still fluent. */
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l && !y.op_tap)
                y.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
    }

    if (m.layers.empty() || !m.layers.back().next_norm)
        m.op_onorm = rmsnorm(T, E, m.w_onorm, 1.0, m.b_x, m.b_h);
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                      {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", E),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(m.w_tok)),
                      {m.b_hout}, {m.b_logits});
    /* FINAL LOGIT SOFTCAPPING, cap * tanh(x / cap) over the f32 logits: `logit_softcap`, which
     * libref, libavx and libr4d serve. Where nothing does the model still runs and greedy decoding
     * is unchanged -- the cap is strictly increasing -- but a sampled distribution is SHARPER than
     * the model's, so the startup log says so. */
    if (m.softcap > 0.0) {
        m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                      RAD_INT("n", g.n_vocab), RAD_F64("cap", m.softcap),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                       {m.b_logits}, {m.b_logits});
        if (!m.op_cap)
            rad_note(b, "WARNING: nothing serves `logit_softcap` (load libr4d or libavx). "
                        "The model's logits are %g * tanh(x / %g); without it greedy output is "
                        "unchanged and SAMPLED output draws from a sharper distribution than the "
                        "model's", m.softcap, m.softcap);
    }

    /* THE DRAFTER LAST: declaration order is the buffer planner's liveness (llama_fp8.cpp). */
    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dflash(b, meta, ctx, m, (int)m.n_layer));

    int nfull = 0;
    for (int f : full) nfull += f;
    rad_note(b, "rank %d/%d: %lld layers (%d full attention %lld x %lld over %lld kv%s, %lld "
                "sliding %lld x %lld over %lld kv, window %lld); dense ffn %lld; %s; vocab "
                "[%lld,%lld) tied",
             R, W, (long long)m.n_layer, nfull, (long long)m.kf.nh, (long long)m.kf.hd,
             (long long)m.kf.nkv, m.kf.kv_repl ? " replicated" : "",
             (long long)(m.n_layer - nfull), (long long)m.ks.nh, (long long)m.ks.hd,
             (long long)m.ks.nkv, (long long)m.ks.window, (long long)m.n_ff,
             m.moe ? "routed: experts [base, base + local) of the model's, top-k" : "no routed arm",
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "sandwich joins: %d of %lld fused (gemma4k norm_add_norm / join_add_scale)%s",
             n_joins, (long long)(m.moe ? 2 * m.n_layer : m.n_layer),
             !fuse_joins ? " -- RADIANCE_GEMMA4_FUSE=0, the rmsnorm / add / mul_weight chain"
             : n_joins == 0 ? " -- not served (load gemma4k), the rmsnorm / add / mul_weight chain" : "");
    if (m.moe) {
        auto how = [&](int n_fp8) {
            return n_fp8 == m.n_layer ? "block fp8" : n_fp8 == 0 ? "bf16" : "mixed";
        };
        rad_note(b, "routed arm: %lld local experts from %lld of %lld, top %lld, %lld wide; "
                    "gate_up %s (%d of %lld layers fp8), down %s (%d fp8); passes of %lld tokens",
                 (long long)m.n_local, (long long)m.expert_base, (long long)m.n_expert,
                 (long long)m.top_k, (long long)m.n_ffe, how(m.n_gu_fp8), m.n_gu_fp8,
                 (long long)m.n_layer, how(m.n_dn_fp8), m.n_dn_fp8, (long long)m.moe_rows);
    }
    if (m.kf.rot_pairs < m.kf.hd / 2)
        rad_note(b, "full layers: PROPORTIONAL rope -- %lld of %lld frequency pairs rotate at "
                    "theta %g over the whole %lld-wide head, the rest stay at frequency zero",
                 (long long)m.kf.rot_pairs, (long long)(m.kf.hd / 2), m.kf.theta,
                 (long long)m.kf.hd);
    return RAD_OK;
}

/* ============================================================== step */

static void routed(RadCtx* c, const Model& m, const Layer& y, int64_t T) {
    const int64_t E = m.E, K = m.top_k, NL = m.n_local, NE = m.n_expert;
    for (int64_t r0 = 0; r0 < T; r0 += m.moe_rows) {
        const int64_t n = (T - r0 < m.moe_rows) ? (T - r0) : m.moe_rows;
        int32_t* const counts = rad_route_counts(c, (int)(&y - m.layers.data()), NL);
        const RadOperand ecnt = counts ? praw(counts, RAD_I32, NL) : brows(m.b_ecnt, NL);
        RAD_ISSUE_N(c, y.op_rnorm, n, brow_slice(m.b_x, r0, n, E), RAD_W(y.w_rscale),
                    brow_slice(m.b_rn, r0, n, E));
        /* One launch, two or three: the router, its top-k and the sort (see declare). */
        const bool rfused = y.op_router_fused && n <= y.router_fused_rows;
        const bool pair = !rfused && y.op_topk_scatter && n <= y.topk_scatter_rows;
        if (rfused) {
            RAD_ISSUE_N(c, y.op_router_fused, n, brow_slice(m.b_rn, r0, n, E), RAD_W(y.w_router),
                        RAD_NONE, brow_slice(m.b_logit, r0, n, NE),
                        brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), ecnt, RAD_NONE);
        } else {
            RAD_ISSUE_N(c, y.op_router, n, brow_slice(m.b_rn, r0, n, E), RAD_W(y.w_router),
                        brow_slice(m.b_logit, r0, n, NE));
            if (pair)
                RAD_ISSUE_N(c, y.op_topk_scatter, n, brow_slice(m.b_logit, r0, n, NE),
                            brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K),
                            brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), ecnt);
            else
                RAD_ISSUE_N(c, y.op_topk, n, brow_slice(m.b_logit, r0, n, NE),
                            brow_slice(m.b_ids, r0, n, K), brow_slice(m.b_ew, r0, n, K));
        }
        /* The ids and the gathered scales read as flat [n * K] vectors: the same bytes. */
        int32_t* ids = (int32_t*)rad_buf_ptr(c, m.b_ids);
        uint16_t* es = (uint16_t*)rad_buf_ptr(c, m.b_es);
        RAD_ISSUE_N(c, y.op_escale, n * K, praw(ids ? ids + r0 * K : nullptr, RAD_I32, n * K),
                    RAD_W(y.w_escale), praw2(es ? es + r0 * K : nullptr, RAD_BF16, n * K, 1));
        RAD_ISSUE_N(c, y.op_emul, n, brow_slice(m.b_ew, r0, n, K), brow_slice(m.b_es, r0, n, K),
                    brow_slice(m.b_ew, r0, n, K));
        /* rmsnorm_quant_fp8's operands: x, residual (absent), w, q, scale, out_bf16 (absent). */
        if (y.gu_fp8)
            RAD_ISSUE_N(c, y.op_enorm, n, brow_slice(m.b_x, r0, n, E), RAD_NONE, RAD_W(y.w_enorm),
                        brow_slice(m.b_hq, r0, n, E), brow_slice(m.b_hqs, r0, n, E / RAD_FP8_BLOCK),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, y.op_enorm, n, brow_slice(m.b_x, r0, n, E), RAD_W(y.w_enorm),
                        brow_slice(m.b_hx, r0, n, E));
        if (!rfused && !pair)
            RAD_ISSUE_N(c, y.op_scatter, n, brow_slice(m.b_ids, r0, n, K), brows(m.b_sorted, n * K),
                        brows(m.b_eoff, NL + 1), ecnt);
        /* An fp8 projection's tables: the first handle of each parity's run and its length --
         * even codes, even scales, odd codes, odd scales (declare). */
        const int64_t H = NL / 2;
        if (y.gu_fp8)
            RAD_ISSUE_N(c, y.op_egu, n, brow_slice(m.b_hq, r0, n, E),
                        brow_slice(m.b_hqs, r0, n, E / RAD_FP8_BLOCK),
                        RAD_WTAB(y.w_egu[0], H), RAD_WTAB(y.w_egus[0], H),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1),
                        RAD_WTAB(y.w_egu[(size_t)H], H), RAD_WTAB(y.w_egus[(size_t)H], H),
                        brows(m.b_egu, n * K));
        else
            RAD_ISSUE_N(c, y.op_egu, n, brow_slice(m.b_hx, r0, n, E), RAD_WTAB(y.w_egu[0], NL),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), brows(m.b_egu, n * K));
        RAD_ISSUE_N(c, y.op_eact, n * K, bcol(m.b_egu, 0, m.n_ffe, n * K),
                    bcol(m.b_egu, m.n_ffe, m.n_ffe, n * K), brows(m.b_eact, n * K));
        if (y.dn_fp8) {
            RAD_ISSUE_N(c, y.op_eq, n * K, brows(m.b_eact, n * K), brows(m.b_eq, n * K),
                        brows(m.b_eqs, n * K));
            RAD_ISSUE_N(c, y.op_edn, n, brows(m.b_eq, n * K), brows(m.b_eqs, n * K),
                        RAD_WTAB(y.w_edn[0], H), RAD_WTAB(y.w_edns[0], H),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1),
                        RAD_WTAB(y.w_edn[(size_t)H], H), RAD_WTAB(y.w_edns[(size_t)H], H),
                        brows(m.b_edn, n * K));
        } else {
            RAD_ISSUE_N(c, y.op_edn, n, brows(m.b_eact, n * K), RAD_WTAB(y.w_edn[0], NL),
                        brows(m.b_sorted, n * K), brows(m.b_eoff, NL + 1), brows(m.b_edn, n * K));
        }
        /* The routed partial into rows [T + r0, T + r0 + n) of the join. */
        RAD_ISSUE_N(c, y.op_gather, n, brows(m.b_edn, n * K), brow_slice(m.b_ew, r0, n, K),
                    brows(m.b_sorted, n * K), RAD_NONE, RAD_NONE,
                    brow_slice(m.b_ff2, T + r0, n, E));
        if (r0 + n >= T) {
            RadRouting rt{};
            rt.expert_ids    = ids;
            rt.expert_w      = nullptr;   /* bf16 weights; the field is const float* (moe_fp8.h) */
            rt.expert_count  = counts ? counts : (const int32_t*)rad_buf_ptr(c, m.b_ecnt);
            rt.sorted_tok    = (const int32_t*)rad_buf_ptr(c, m.b_sorted);
            rt.expert_offset = (const int32_t*)rad_buf_ptr(c, m.b_eoff);
            rt.top_k         = K;
            rt.n_expert      = NL;
            rad_route_report(c, (int)(&y - m.layers.data()), &rt);
        }
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    if (batch->draft_pass != 0) {
        if (!m.have_df) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no DFlash `draft.` block, or declare ran at "
                             "max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                       m.df.context(c, batch);
        return;
    }
    if (batch->enc) {
        rad_step_fail(c, "gemma4 declares no encoder; serve this container with --mm-max-patches 0");
        return;
    }
    const int64_t T = batch->n_tok, E = m.E;
    if (T <= 0) return;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * E, brows(m.b_x, T), RAD_NONE);

    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& y = m.layers[li];
        const Kind& k = y.full ? m.kf : m.ks;
        const RadKVGroupBatch* kvb = kv_batch(batch, k.kv);
        const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
        const int32_t* table = kvb ? kvb->block_table  : nullptr;
        const int32_t* used  = kvb ? kvb->seqused      : nullptr;
        const RadOperand bt  = praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0);
        const RadOperand su  = praw(used, RAD_I32, batch->n_seq);

        if (y.op_anorm) RAD_ISSUE(c, y.op_anorm, brows(m.b_x, T), RAD_W(y.w_anorm), brows(m.b_h, T));
        if (!y.full) {
            const int64_t qd = k.q(), kvd = k.kv_dim();
            RAD_ISSUE(c, y.op_qkv, brows(m.b_h, T), RAD_W(y.w_qkv), brows(m.b_qkv, T));
            RAD_ISSUE(c, y.op_qknr, brows(m.b_qkv, T), pos, RAD_W(y.w_qn), RAD_W(y.w_kn));
            RAD_ISSUE(c, y.op_store, bcol(m.b_qkv, qd, kvd, T), bcol(m.b_qkv, qd + kvd, kvd, T),
                      praw(slot, RAD_I32, T), kv_cache(k.kv, (int)li));
            RAD_ISSUE_N(c, y.op_attn, batch->max_q_len, bcol(m.b_qkv, 0, qd, T), kv_cache(k.kv, (int)li),
                        bt, su, RAD_NONE, RAD_NONE, cu, brows(m.b_atts, T));
            RAD_ISSUE(c, y.op_o, brows(m.b_atts, T), RAD_W(y.w_o), brows(m.b_o, T));
        } else {
            RAD_ISSUE(c, y.op_q, brows(m.b_h, T), RAD_W(y.w_q), brows(m.b_qf, T));
            RAD_ISSUE(c, y.op_k, brows(m.b_h, T), RAD_W(y.w_k), brows(m.b_kf, T));
            RAD_ISSUE_N(c, y.op_vnorm, T * k.nkv, brows(m.b_kf, T), RAD_NONE, brows(m.b_vf, T));
            RAD_ISSUE_N(c, y.op_qn, T * k.nh, brows(m.b_qf, T), RAD_W(y.w_qn), brows(m.b_qf, T));
            RAD_ISSUE_N(c, y.op_kn, T * k.nkv, brows(m.b_kf, T), RAD_W(y.w_kn), brows(m.b_kf, T));
            RAD_ISSUE(c, y.op_qrope, brows(m.b_qf, T), pos);
            RAD_ISSUE(c, y.op_krope, brows(m.b_kf, T), pos);
            RAD_ISSUE(c, y.op_store, brows(m.b_kf, T), brows(m.b_vf, T), praw(slot, RAD_I32, T),
                      kv_cache(k.kv, (int)li));
            RAD_ISSUE_N(c, y.op_attn, batch->max_q_len, brows(m.b_qf, T), kv_cache(k.kv, (int)li),
                        bt, su, RAD_NONE, RAD_NONE, cu, brows(m.b_attf, T));
            RAD_ISSUE(c, y.op_o, brows(m.b_attf, T), RAD_W(y.w_o), brows(m.b_o, T));
        }
        if (y.op_ar_o) RAD_ISSUE_N(c, y.op_ar_o, T * E, brows(m.b_o, T), RAD_NONE);
        if (y.op_join_a) {
            RAD_ISSUE(c, y.op_join_a, brows(m.b_o, T), RAD_W(y.w_apost), brows(m.b_x, T),
                      RAD_W(y.w_fnorm), brows(m.b_h, T));
        } else {
            RAD_ISSUE(c, y.op_apost, brows(m.b_o, T), RAD_W(y.w_apost), brows(m.b_o, T));
            RAD_ISSUE(c, y.op_add_a, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
            RAD_ISSUE(c, y.op_fnorm, brows(m.b_x, T), RAD_W(y.w_fnorm), brows(m.b_h, T));
        }

        /* dense arm -> rows [0, T) of the join */
        RAD_ISSUE(c, y.op_gu, brows(m.b_h, T), RAD_W(y.w_gu), brows(m.b_gu, T));
        RAD_ISSUE(c, y.op_act, bcol(m.b_gu, 0, m.n_ff, T), bcol(m.b_gu, m.n_ff, m.n_ff, T),
                  brows(m.b_act, T));
        RAD_ISSUE(c, y.op_dn, brows(m.b_act, T), RAD_W(y.w_dn), brows(m.b_ff2, T));
        if (m.moe) {
            routed(c, m, y, T);
            if (y.op_ar_f) RAD_ISSUE_N(c, y.op_ar_f, 2 * T * E, brows(m.b_ff2, 2 * T), RAD_NONE);
            if (y.op_join_f) {
                RAD_ISSUE(c, y.op_join_f, brows(m.b_ff2, T), RAD_W(y.w_fpost1),
                          brow_slice(m.b_ff2, T, T, E), RAD_W(y.w_fpost2), RAD_W(y.w_fpost),
                          brows(m.b_x, T), RAD_W(y.w_scalar), RAD_W(y.w_next), brows(m.b_h, T));
            } else {
                RAD_ISSUE(c, y.op_fpost1, brows(m.b_ff2, T), RAD_W(y.w_fpost1), brows(m.b_ff2, T));
                RAD_ISSUE(c, y.op_fpost2, brow_slice(m.b_ff2, T, T, E), RAD_W(y.w_fpost2),
                          brow_slice(m.b_ff2, T, T, E));
                RAD_ISSUE(c, y.op_fsum, brows(m.b_ff2, T), brow_slice(m.b_ff2, T, T, E), brows(m.b_h, T));
                RAD_ISSUE(c, y.op_fpost, brows(m.b_h, T), RAD_W(y.w_fpost), brows(m.b_h, T));
            }
        } else {
            if (y.op_ar_f) RAD_ISSUE_N(c, y.op_ar_f, T * E, brows(m.b_ff2, T), RAD_NONE);
            RAD_ISSUE(c, y.op_fpost, brows(m.b_ff2, T), RAD_W(y.w_fpost), brows(m.b_h, T));
        }
        if (!y.op_join_f) {
            RAD_ISSUE(c, y.op_add_f, brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));
            RAD_ISSUE(c, y.op_scale, brows(m.b_x, T), RAD_W(y.w_scalar), brows(m.b_x, T));
        }

        /* The tap AFTER the scalar: HF's hidden_states[i + 1] is the decoder layer's return value,
         * which has been multiplied by it. */
        for (int t = 0; t < m.n_taps && y.op_tap; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, y.op_tap, brows(m.b_x, T),
                      bcol(m.b_dfsrc, (int64_t)t * E, E, T));
        }
    }

    if (m.op_onorm) RAD_ISSUE(c, m.op_onorm, brows(m.b_x, T), RAD_W(m.w_onorm), brows(m.b_h, T));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.b_h, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out, brows(m.b_hout, batch->n_out), RAD_W(m.w_tok),
                    brows(m.b_logits, batch->n_out));
        if (m.op_cap)
            RAD_ISSUE_N(c, m.op_cap, batch->n_out, brows(m.b_logits, batch->n_out),
                        brows(m.b_logits, batch->n_out));
    }
}

/* ============================================================== probe */
/* DFlash proposes its whole block in one pass, so its depth is the block it was trained at less
 * the anchor, and nothing else: 15 for z-lab's Gemma 4 drafters (block_size 16). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash") && !std::strstr(da, "DFlash2")) {
        long long bs = rad_meta_geti(meta, "draft.block_size", 0);
        if (bs <= 0) bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
        if (bs > 1) {
            out->draft_depth = (int)(bs - 1);
            out->draft_depth_fixed = 1;
        }
    }
    return RAD_OK;
}

/* ============================================================== the reply format
 * The model's, not this plugin's -- the thought channel, and calls written as a name and one
 * object in Gemma's dialect of JSON -- and shared with gemma4_bf16 (arch/rad_gemma4_chat.h). The
 * container's template is not visible from here (it lives in the vocab section, not the metadata),
 * so this answers for every Gemma 4 container. */
static const RadChatFormat* chat_format(const RadModelMeta*) { return gemma4_chat_format(); }

}  /* namespace gemma4 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(gemma4, "gemma4", "", "0.1.0",
                "Gemma 4 text decoders (26B-A4B MoE, dense 31B), bf16: sliding and full attention "
                "with proportional rope and K = V, a dense and a routed feed-forward side by side, "
                "and z-lab's DFlash block-diffusion drafter")
RAD_ARCH_PROBE(gemma4)
RAD_ARCH_CHAT_FORMAT(gemma4)
#endif
