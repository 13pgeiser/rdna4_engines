/* siglip2.h -- LFM2-VL's image encoder: the SigLIP2 NaFlex tower and the projector after it.
 *
 * WHAT IT COMPUTES, per encoder pass (RadBatch::enc):
 *
 *   x  = patches @ Wp^T + bp                     the patch embedding, a linear layer over a patch
 *                                                row (y, x, channel)
 *   x += pos_table resampled to each segment's grid   torch's antialiased half-pixel bilinear
 *                                                (grid_embed_aa), rounded before the add
 *   depth times:
 *     x += out(attn(qkv(LN1(x))))                full attention within a segment (one tile or the
 *                                                thumbnail), no rotary
 *     x += fc2(gelu_tanh(fc1(LN2(x))))
 *   y  = post_LN(x)
 *   out = l2(gelu(l1([LN](y viewed [P/merge^2, merge^2 * width]))))   the projector
 *
 * which is transformers' Siglip2VisionModel (no pooling head) followed by Lfm2VlMultiModalProjector.
 * The projector's pixel unshuffle is a VIEW here: the processor stores each segment's patches in
 * merge-block order (core/mm/processor.cpp, image_lfm2), so a block's merge^2 patches are
 * consecutive rows, dy then dx -- the order the unshuffle concatenates them in. The tower itself
 * does not care about the order: attention within a segment is permutation-equivariant and every
 * other layer is per patch, and the position table is resampled at each patch's own (row, column).
 *
 * q, k and v are one fused weight, concatenated at convert (map_concat), so a layer is one QKV
 * GEMM; the attention reads the three as column slices.
 *
 * Replicated, per-request weights and pass-sized activations, as arch/common/rad_block_vit.h's
 * tower: see the notes there.
 */
#ifndef LFM2_SIGLIP2_H
#define LFM2_SIGLIP2_H

#include <arch/rad_arch.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lfm2_bf16 {

using namespace rad::arch;

struct Siglip2 {
    int64_t depth = 0, width = 0, heads = 0, ff = 0, patch = 0, merge = 0, side = 0;
    int64_t head_dim = 0, patch_dim = 0, proj_hidden = 0, n_embd = 0, P = 0;
    float   eps = 1e-6f;
    bool    proj_norm = false, on = false;

    struct Layer {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0;
        rad_weight qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0;
        rad_weight up_w = 0, up_b = 0, down_w = 0, down_b = 0;
        rad_op ln1 = 0, qkv = 0, attn = 0, proj = 0, ln2 = 0, up = 0, down = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0, pn_w = 0, pn_b = 0;
    rad_weight mn_w = 0, mn_b = 0, m1_w = 0, m1_b = 0, m2_w = 0, m2_b = 0;
    rad_op op_patch = 0, op_pos = 0, op_post = 0, op_mln = 0, op_m1 = 0, op_m2 = 0;
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_mh = 0, b_mn = 0, b_mf = 0;
    rad_buf b_out = 0;

    /* A tower to declare: the container is an LFM2-VL one, whose vision_config is SigLIP2's. */
    static bool present(const RadModelMeta* m) {
        return !std::strcmp(rad_meta_gets(m, "vision_config.model_type", ""), "siglip2_vision_model") &&
               rad_meta_geti(m, "vision_config.num_hidden_layers", 0) > 0;
    }
    /* Input patches an output row: the projector's unshuffle. */
    static int64_t merge_of(const RadModelMeta* m) {
        const int64_t d = rad_meta_geti(m, "downsample_factor", 0);
        return d * d;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int Siglip2::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                            Names& nm, int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    depth  = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
    width  = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads  = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
    ff     = rad_meta_geti(m, "vision_config.intermediate_size", 0);
    patch  = rad_meta_geti(m, "vision_config.patch_size", 0);
    eps    = (float)rad_meta_getf(m, "vision_config.layer_norm_eps", 1e-6);
    const int64_t chans = rad_meta_geti(m, "vision_config.num_channels", 3);
    const int64_t npos  = rad_meta_geti(m, "vision_config.num_patches", 0);
    merge  = rad_meta_geti(m, "downsample_factor", 0);
    proj_hidden = rad_meta_geti(m, "projector_hidden_size", 0);
    n_embd = model_embd;
    side = (int64_t)std::llround(std::sqrt((double)npos));
    if (depth <= 0 || width <= 0 || heads <= 0 || ff <= 0 || patch <= 0 || merge <= 0 ||
        proj_hidden <= 0 || side <= 0 || side * side != npos || width % heads || chans != 3) {
        fprintf(stderr, "radiance: lfm2: the container's SigLIP2 vision_config is incomplete or "
                        "inconsistent (layers %lld, hidden %lld, heads %lld, ff %lld, patch %lld, "
                        "positions %lld, channels %lld; downsample_factor %lld, projector %lld)\n",
                (long long)depth, (long long)width, (long long)heads, (long long)ff,
                (long long)patch, (long long)npos, (long long)chans, (long long)merge,
                (long long)proj_hidden);
        return RAD_E_INVAL;
    }
    /* The pooling head is SigLIP2's contrastive readout; LFM2-VL reads the patch rows, and a
     * checkpoint that ships a head would load weights nothing runs. Refused by name. */
    if (rad_meta_geti(m, "vision_config.vision_use_head", 0) != 0 ||
        !std::strcmp(rad_meta_gets(m, "vision_config.vision_use_head", ""), "true")) {
        fprintf(stderr, "radiance: lfm2: this SigLIP2 tower has a pooling head, which LFM2-VL does "
                        "not read and this plugin does not serve\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* act = rad_meta_gets(m, "vision_config.hidden_act", "gelu_pytorch_tanh");
    const char* pact = rad_meta_gets(m, "projector_hidden_act", "gelu");
    if (std::strcmp(act, "gelu_pytorch_tanh") || std::strcmp(pact, "gelu")) {
        fprintf(stderr, "radiance: lfm2: the tower's activation '%s' and the projector's '%s' are "
                        "not gelu_pytorch_tanh and gelu, which is what this plugin serves\n",
                act, pact);
        return RAD_E_UNSUPPORTED;
    }
    const char* pb = rad_meta_gets(m, "projector_bias", "true");
    if (std::strcmp(pb, "true") && std::strcmp(pb, "1")) {
        fprintf(stderr, "radiance: lfm2: a projector without biases is not served\n");
        return RAD_E_UNSUPPORTED;
    }
    proj_norm = !std::strcmp(rad_meta_gets(m, "projector_use_layernorm", "true"), "true") ||
                !std::strcmp(rad_meta_gets(m, "projector_use_layernorm", "true"), "1");
    head_dim  = width / heads;
    patch_dim = chans * patch * patch;

    const int64_t mm = merge * merge, mw = mm * width;
    P = ctx->max_enc_patches;
    /* The longest segment attention sees: a tile or the largest thumbnail, which the processor
     * keeps within max_num_patches. A bound that picks the attention kernel's block height, so it
     * is the real one in a sizing declare too. */
    const int64_t seg = rad_meta_geti(m, "processor.image_processor.max_num_patches",
                                      rad_meta_geti(m, "max_num_patches", P));
    const int64_t max_seg = std::min<int64_t>(P, seg);
    if (ctx->shape_probe) {
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    if (P % mm) return RAD_E_SHAPE;
    const int64_t R = P / mm;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                  RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    const char* tw = "model.vision_tower.vision_model.";
    w_patch = bw(nm.f("v.patch.weight"), nm.ckpt("%sembeddings.patch_embedding.weight", tw),
                 { width, patch_dim }, grp_model());
    b_patch = bw(nm.f("v.patch.bias"), nm.ckpt("%sembeddings.patch_embedding.bias", tw),
                 { width }, grp_model());
    w_pos   = bw(nm.f("v.pos.weight"), nm.ckpt("%sembeddings.position_embedding.weight", tw),
                 { npos, width }, grp_model());
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        const char* e = "%sencoder.layers.%d.%s";
        l.n1w = bw(nm.f("v.blk.%d.norm1.weight", li), nm.ckpt(e, tw, li, "layer_norm1.weight"), { width }, g);
        l.n1b = bw(nm.f("v.blk.%d.norm1.bias", li),   nm.ckpt(e, tw, li, "layer_norm1.bias"),   { width }, g);
        l.n2w = bw(nm.f("v.blk.%d.norm2.weight", li), nm.ckpt(e, tw, li, "layer_norm2.weight"), { width }, g);
        l.n2b = bw(nm.f("v.blk.%d.norm2.bias", li),   nm.ckpt(e, tw, li, "layer_norm2.bias"),   { width }, g);
        /* q, k, v fused at convert: one GEMM a layer. */
        const char* qw = nm.f("v.blk.%d.qkv.weight", li);
        const char* qb = nm.f("v.blk.%d.qkv.bias", li);
        if (map_concat(b, qw, { nm.ckpt(e, tw, li, "self_attn.q_proj.weight"),
                                nm.ckpt(e, tw, li, "self_attn.k_proj.weight"),
                                nm.ckpt(e, tw, li, "self_attn.v_proj.weight") }) < 0 ||
            map_concat(b, qb, { nm.ckpt(e, tw, li, "self_attn.q_proj.bias"),
                                nm.ckpt(e, tw, li, "self_attn.k_proj.bias"),
                                nm.ckpt(e, tw, li, "self_attn.v_proj.bias") }) < 0)
            return RAD_E_INVAL;
        l.qkv_w = decl_w(b, qw, RAD_BF16, { 3 * width, width }, ACC, RAD_SHARD_NONE, g);
        l.qkv_b = decl_w(b, qb, RAD_BF16, { 3 * width }, ACC, RAD_SHARD_NONE, g);
        l.out_w = bw(nm.f("v.blk.%d.proj.weight", li), nm.ckpt(e, tw, li, "self_attn.out_proj.weight"), { width, width }, g);
        l.out_b = bw(nm.f("v.blk.%d.proj.bias", li),   nm.ckpt(e, tw, li, "self_attn.out_proj.bias"),   { width }, g);
        l.up_w  = bw(nm.f("v.blk.%d.fc1.weight", li),  nm.ckpt(e, tw, li, "mlp.fc1.weight"), { ff, width }, g);
        l.up_b  = bw(nm.f("v.blk.%d.fc1.bias", li),    nm.ckpt(e, tw, li, "mlp.fc1.bias"),   { ff }, g);
        l.down_w = bw(nm.f("v.blk.%d.fc2.weight", li), nm.ckpt(e, tw, li, "mlp.fc2.weight"), { width, ff }, g);
        l.down_b = bw(nm.f("v.blk.%d.fc2.bias", li),   nm.ckpt(e, tw, li, "mlp.fc2.bias"),   { width }, g);
        if (!l.n1w || !l.n1b || !l.n2w || !l.n2b || !l.qkv_w || !l.qkv_b || !l.out_w ||
            !l.out_b || !l.up_w || !l.up_b || !l.down_w || !l.down_b)
            return RAD_E_INVAL;
    }
    pn_w = bw(nm.f("v.post_norm.weight"), nm.ckpt("%spost_layernorm.weight", tw), { width }, grp_model());
    pn_b = bw(nm.f("v.post_norm.bias"),   nm.ckpt("%spost_layernorm.bias", tw),   { width }, grp_model());
    const char* pj = "model.multi_modal_projector.";
    if (proj_norm) {
        mn_w = bw(nm.f("v.proj.norm.weight"), nm.ckpt("%slayer_norm.weight", pj), { mw }, grp_model());
        mn_b = bw(nm.f("v.proj.norm.bias"),   nm.ckpt("%slayer_norm.bias", pj),   { mw }, grp_model());
        if (!mn_w || !mn_b) return RAD_E_INVAL;
    }
    m1_w = bw(nm.f("v.proj.fc1.weight"), nm.ckpt("%slinear_1.weight", pj), { proj_hidden, mw }, grp_model());
    m1_b = bw(nm.f("v.proj.fc1.bias"),   nm.ckpt("%slinear_1.bias", pj),   { proj_hidden }, grp_model());
    m2_w = bw(nm.f("v.proj.fc2.weight"), nm.ckpt("%slinear_2.weight", pj), { n_embd, proj_hidden }, grp_model());
    m2_b = bw(nm.f("v.proj.fc2.bias"),   nm.ckpt("%slinear_2.bias", pj),   { n_embd }, grp_model());
    if (!w_patch || !b_patch || !w_pos || !pn_w || !pn_b || !m1_w || !m1_b || !m2_w || !m2_b)
        return RAD_E_INVAL;

    /* ---- activations. `mh` is the post-norm's output declared at the MERGED width: the norm
     * writes P rows of `width` and the projector reads P/mm rows of mm * width, the same bytes. */
    b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, { P, width });
    b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, { P, width });
    b_qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
    b_att = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
    b_ff  = decl_b(b, nm.f("v.ff"),  RAD_BF16, { P, ff });
    b_mh  = decl_b(b, nm.f("v.mh"),  RAD_BF16, { R, mw });
    if (proj_norm) b_mn = decl_b(b, nm.f("v.mn"), RAD_BF16, { R, mw });
    b_mf  = decl_b(b, nm.f("v.mf"),  RAD_BF16, { R, proj_hidden });
    b_out = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_h || !b_qkv || !b_att || !b_ff || !b_mh || (proj_norm && !b_mn) || !b_mf ||
        !b_out)
        return RAD_E_INVAL;

    auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* a, rad_weight w,
                    rad_weight bias) -> rad_op {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                      RAD_WEIGHTS(w, bias));
    };
    auto lnorm = [&](int64_t rows, int64_t n, rad_weight w, rad_weight bias) -> rad_op {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                 RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };

    op_patch = rw(b, gemm(P, width, patch_dim, "none", w_patch, b_patch), {}, { b_x });
    op_pos = rw(b, RAD_OP(b, "grid_embed_aa",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                     RAD_INT("side", side), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w_pos)),
                { b_x }, { b_x });
    for (Layer& l : L) {
        l.ln1  = rw(b, lnorm(P, width, l.n1w, l.n1b), { b_x }, { b_h });
        l.qkv  = rw(b, gemm(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.attn = rw(b, RAD_OP(b, "attn_dense",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                         RAD_INT("max_seqlen", max_seg)),
                              RAD_NOWEIGHTS),
                    { b_qkv }, { b_att });
        l.proj = rw(b, gemm(P, width, width, "none", l.out_w, l.out_b), { b_att, b_x }, { b_x });
        l.ln2  = rw(b, lnorm(P, width, l.n2w, l.n2b), { b_x }, { b_h });
        l.up   = rw(b, gemm(P, ff, width, "gelu_tanh", l.up_w, l.up_b), { b_h }, { b_ff });
        l.down = rw(b, gemm(P, width, ff, "none", l.down_w, l.down_b), { b_ff, b_x }, { b_x });
    }
    op_post = rw(b, lnorm(P, width, pn_w, pn_b), { b_x }, { b_mh });
    if (proj_norm) op_mln = rw(b, lnorm(R, mw, mn_w, mn_b), { b_mh }, { b_mn });
    op_m1 = rw(b, gemm(R, proj_hidden, mw, "gelu", m1_w, m1_b), { proj_norm ? b_mn : b_mh }, { b_mf });
    op_m2 = rw(b, gemm(R, n_embd, proj_hidden, "none", m2_w, m2_b), { b_mf }, { b_out });

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "SigLIP2 tower: %lld layers of %lld (%lld heads x %lld), ff %lld, patch %lld, "
                    "%lld x %lld position table; projector %lld -> %lld -> %lld%s; %lld patches a "
                    "pass -> %lld rows",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)patch, (long long)side, (long long)side,
                 (long long)mw, (long long)proj_hidden, (long long)n_embd,
                 proj_norm ? " behind a layer norm" : "", (long long)P, (long long)R);
    }
    on = true;
    return RAD_OK;
}

inline void Siglip2::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge, Rn = Pn / mm;
    if (!on || batch->enc != RAD_MM_IMAGE || Pn <= 0 || Pn > P || Pn % mm || S <= 0) {
        rad_step_fail(c, "lfm2: an encoder pass this program's SigLIP2 tower cannot serve: no "
                         "tower, not images, more patches than it was declared for, or not whole "
                         "merged rows");
        return;
    }
    /* The patches, their coordinates -- a dense [4, P] of row, column, segment height and width,
     * which is what the position resample reads -- and the segment boundaries. */
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), RAD_W(b_patch), RAD_NONE, brows(b_x, Pn));
    RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd);
    for (const Layer& l : L) {
        RAD_ISSUE_N(c, l.ln1, Pn, brows(b_x, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.attn, Pn,
                    bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.proj, Pn, brows(b_att, Pn), RAD_W(l.out_w), RAD_W(l.out_b),
                    brows(b_x, Pn), brows(b_x, Pn));
        RAD_ISSUE_N(c, l.ln2, Pn, brows(b_x, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.up, Pn, brows(b_h, Pn), RAD_W(l.up_w), RAD_W(l.up_b), RAD_NONE,
                    brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.down, Pn, brows(b_ff, Pn), RAD_W(l.down_w), RAD_W(l.down_b),
                    brows(b_x, Pn), brows(b_x, Pn));
    }
    /* The post-norm writes Pn rows of `width` into a buffer declared at the merged width: the
     * same bytes the projector reads as Pn/mm rows -- the pixel unshuffle. */
    RAD_ISSUE_N(c, op_post, Pn, brows(b_x, Pn), RAD_W(pn_w), RAD_W(pn_b), brows(b_mh, Rn));
    if (proj_norm)
        RAD_ISSUE_N(c, op_mln, Rn, brows(b_mh, Rn), RAD_W(mn_w), RAD_W(mn_b), brows(b_mn, Rn));
    RAD_ISSUE_N(c, op_m1, Rn, brows(proj_norm ? b_mn : b_mh, Rn), RAD_W(m1_w), RAD_W(m1_b),
                RAD_NONE, brows(b_mf, Rn));
    RAD_ISSUE_N(c, op_m2, Rn, brows(b_mf, Rn), RAD_W(m2_w), RAD_W(m2_b), RAD_NONE,
                brows(b_out, Rn));
}

}  /* namespace lfm2_bf16 */

#endif /* LFM2_SIGLIP2_H */
