/* rad_block_sam.h -- Unlimited-OCR's DeepEncoder: SAM ViT-B/16, then CLIP ViT-L/14 on its map.
 *
 * The reference is the checkpoint's own deepencoder.py (cached at /tmp/kilo/uocr/), used by
 * modeling_unlimitedocr.py's image branch:
 *
 *   sam_model(1024x1024 picture)                                  -> [1, 1024, 16, 16]
 *     patch_embed: conv 16x16 stride 16 (768 out), + pos_embed [1,64,64,768]
 *     12 blocks, w 768, 12 heads x 64: LayerNorm, qkv, attention with DECOMPOSED RELATIVE
 *       POSITION (rel_pos_h/rel_pos_w, 27 rows) inside a 14x14 window of the 64x64 patch grid
 *       (padded to 70x70 = 25 windows), and one 64x64 window on blocks 2, 5, 8 and 11 (tables of
 *       127 rows); proj; LayerNorm; mlp 3072 exact gelu; mlp proj
 *     neck: conv 1x1 256 (no bias), LayerNorm2d 256, conv 3x3 256 (no bias), LayerNorm2d 256
 *     net_2: conv 3x3 stride 2 -> 512 at 32x32 (no bias); net_3: -> 1024 at 16x16 (no bias)
 *   vision_model(patches=the sam map)                             -> CLIP ViT-L/14, 257 tokens
 *     class token + the 16x16 map as patch_embeds (NO pixels) + position_embedding[257];
 *     pre_layrnorm eps 1e-5; 24 layers, w 1024, 16 heads x 64: LayerNorm, fused qkv, dense
 *       attention over all 257 tokens, out_proj; LayerNorm; fc1 4096 quick_gelu; fc2
 *   fuse: cat(clip rows 1:, the 16x16 map) -> 2048 -> the projector linear -> 1280
 *   layout: every map line of 16 rows is closed by `image_newline`, the view by
 *           `view_seperator` -> 16*(16+1) + 1 = 273 rows for one 1024x1024 view
 *
 * ============================== THE ONE OP THE ENGINE MUST ADD ==============================
 *
 * SAM's attention logits are
 *
 *     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
 *
 * with Rh/Rw read from the relative-position tables at the (query, key) grid offsets. The
 * per-pair term is a dot of the query with a vector that depends on BOTH rows, so it folds into
 * neither q, k nor v, and no declared op takes a bias operand (attn_dense's operands are
 * q, k, v, cu, out). This block therefore declares its attention as `sam_attn`:
 *
 *   params    M(range), head_dim, n_head, grid_h, grid_w, window, scale(f64), dtype
 *   operands  q, k, v, cu_seqlens (i32), out     -- rows n_head*head_dim each
 *   weights   rel_pos_h, rel_pos_w               -- [2*win-1 or 2*grid-1, head_dim]
 *   for each cu segment of grid_h*grid_w rows in row-major order: with `window` > 0 pad the grid
 *   to multiples of `window` with zero rows (deepencoder's window_partition pads AFTER norm1;
 *   the pad rows take part in the softmax exactly as the reference's do), cut it into
 *   window x window tiles, each tile attending to all of its rows; with `window` 0 one tile is
 *   the whole segment. logits as above, softmax in f32 over the tile's keys, out = sum_j p v_j,
 *   written back in the segment's row order with the pad rows dropped. Rh is deepencoder's
 *   get_rel_pos(table, q_h, k_h): the table at (i_h - j_h + len - 1), linearly interpolated when
 *   its length is not 2*max(q_h,k_h)-1 (all three geometries here match exactly).
 *
 * probe() answers whether that op resolves. When it does not, declare() still maps and declares
 * every weight (a converted container carries the tower) but declares NO tower op and NO encoder,
 * which leaves the plugin on P1's text-only behaviour. Dropping the tables instead is not a
 * usable approximation: against them the reference's rows move by cosine 0.35-0.77 (P2 report).
 *
 * ============================== ROWS AND PATCHES ==============================
 *
 * core/mm stages one view as a 64 x (64 + 4) grid of 16x16 patch rows: the last kDown = 4 patch
 * columns of every row are zeros the encoder must not read. The patch embed therefore runs one
 * grid row at a time -- a 64-row GEMM reading that row's first 64 staged patch rows where they
 * lie -- so the tower sees exactly the reference's raster 64x64 grid. A view's 272 rows are then
 * written with a `image_newline` row closing each 17-row map line, and the view-separator segment
 * (an empty kDown x kDown grid) one `view_seperator` row: 273 rows a view at merge 16 (4x4
 * patches a row). A pass may carry several views; every one has this geometry (the base mode's
 * 1024 picture). Anything else is refused by name: crops need a resampled position table
 * (bicubic antialias, which no op provides) and tiling is the plan's later gundam mode.
 */
#ifndef RAD_BLOCK_SAM_H
#define RAD_BLOCK_SAM_H

#include <arch/rad_arch.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct DeepEncoder {
    /* ---- the reference's geometry. */
    static constexpr int64_t kPatch     = 16;   /* SAM's patch size in pixels */
    static constexpr int64_t kDown      = 4;    /* patch columns per map token (net_2 x net_3) */
    static constexpr int64_t kGrid      = 64;   /* 1024 / 16 patches a side */
    static constexpr int64_t kMap       = 16;   /* kGrid / kDown */
    static constexpr int64_t kWin       = 14;   /* SAM's window on the patch grid */
    static constexpr int64_t kPadCols   = (kWin - kGrid % kWin) % kWin;   /* window_partition's pad */
    static constexpr int64_t kGridP     = kGrid + kPadCols;                /* 70: padded grid side */
    static constexpr int64_t kSamTokP   = kGridP * kGridP;                 /* 4900 padded tokens */
    static constexpr int64_t kSamW      = 768;
    static constexpr int64_t kSamHeads  = 12;
    static constexpr int64_t kSamHd     = 64;
    static constexpr int64_t kSamDepth  = 12;
    static constexpr int64_t kSamFf     = 3072; /* mlp_ratio 4 */
    static constexpr int64_t kSamTok    = kGrid * kGrid;              /* 4096 */
    static constexpr int64_t kClipW     = 1024;
    static constexpr int64_t kClipHeads = 16;
    static constexpr int64_t kClipHd    = 64;
    static constexpr int64_t kClipDepth = 24;
    static constexpr int64_t kClipFf    = 4096;
    static constexpr int64_t kClipTok   = kMap * kMap + 1;            /* 257: cls + the map */
    static constexpr int64_t kFused     = 2048;
    static constexpr int64_t kNEmbd     = 1280;
    static constexpr int64_t kPatchDim  = 3 * kPatch * kPatch;        /* 768: a staged patch row */
    static constexpr int64_t kRowsView  = kMap * (kMap + 1) + 1;      /* 273 rows of one view */
    static constexpr int64_t kPatchView = kGrid * (kGrid + kDown);    /* 4352 staged patches */
    static constexpr int64_t kSegView   = kPatchView + kDown * kDown; /* 4368 patches an item */
    static constexpr int64_t kMerge     = kDown * kDown;              /* 16 patches a row */

    static bool global_block(int64_t i) { return i == 2 || i == 5 || i == 8 || i == 11; }

    struct SamLayer {
        rad_weight w_n1 = 0, b_n1 = 0, w_qkv = 0, b_qkv = 0, w_proj = 0, b_proj = 0;
        rad_weight w_rh = 0, w_rw = 0;
        rad_weight w_n2 = 0, b_n2 = 0, w_l1 = 0, b_l1 = 0, w_l2 = 0, b_l2 = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_attn = 0, op_proj = 0, op_n2 = 0, op_l1 = 0, op_l2 = 0;
        rad_op op_padd = 0, op_ladd = 0;   /* the residual adds, rounded apart from the GEMM */
        rad_op op_qkvw = 0, op_attnw = 0;   /* the windowed path, over the padded grid */
        rad_op op_gpad = 0, op_greal = 0;   /* into window_partition's order, and back */
        int64_t rel_rows = 0;
        int64_t window = 0;      /* 0 = the whole grid */
    };
    struct ClipLayer {
        rad_weight w_n1 = 0, b_n1 = 0, w_qkv = 0, b_qkv = 0, w_out = 0, b_out = 0;
        rad_weight w_n2 = 0, b_n2 = 0, w_fc1 = 0, b_fc1 = 0, w_fc2 = 0, b_fc2 = 0;
        rad_op op_n1 = 0, op_qkv = 0, op_attn = 0, op_out = 0;
        rad_op op_cadd = 0, op_madd = 0;   /* the residual adds, rounded apart from the GEMM */
        rad_op op_n2 = 0, op_fc1 = 0, op_scale = 0, op_qg = 0, op_fc2 = 0;
    };

    std::vector<SamLayer>  sam;
    std::vector<ClipLayer> clip;

    /* stems, neck, projector, layout */
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0;
    rad_weight w_neck0 = 0, w_nk1 = 0, b_nk1 = 0, w_nk2 = 0, w_nk3 = 0, b_nk3 = 0;
    rad_weight w_net2 = 0, w_net3 = 0;
    rad_weight w_cls = 0, w_cpos = 0, w_playr = 0, b_playr = 0;
    rad_weight w_proj = 0, b_proj = 0, w_newline = 0, w_sep = 0;

    rad_op op_patch = 0, op_ids_sam = 0, op_sampos = 0, op_sampadd = 0;
    rad_op op_neck0 = 0, op_nk1 = 0, op_unf3 = 0, op_nk2 = 0, op_nk3 = 0;
    rad_op op_unf_net2 = 0, op_net2 = 0, op_unf_net3 = 0, op_net3 = 0;
    rad_op op_cls = 0, op_cmap = 0, op_ids_clip = 0, op_cpos = 0, op_cadd = 0, op_playr = 0;
    rad_op op_cp1 = 0;   /* clip[0].norm1 over the pre_layrnorm output */
    rad_op op_ccopy = 0; /* pre_layrnorm output into the residual stream */
    rad_op op_fuse_clip = 0, op_fuse_sam = 0, op_proj = 0, op_line = 0;
    rad_op op_dbg1024 = 0, op_dbg1280 = 0, op_dbg768 = 0, op_dbg256 = 0;
    int dbg = 0, dbg_row = 0;   /* RAD_UNLIMITEDOCR_DBG / _ROW: dump one stage, from a row */
    rad_op op_fill_one = 0, op_fill_one_c = 0, op_fill_1702 = 0;
    rad_op op_newline = 0, op_sep = 0;
    rad_op op_cu_sam = 0, op_cu_clip = 0;   /* {0, kSamTok} and {0, kClipTok} */
    rad_op op_cu_pad = 0;                   /* {0, kSamTokP} */
    rad_op op_gather_pad = 0, op_gather_real = 0;

    /* activations -- one view's, reused by every view a pass carries. */
    rad_buf b_x = 0, b_h = 0, b_qkv = 0, b_att = 0, b_ff = 0, b_sampos = 0;
    /* The windowed path's padded grid: h gathered into window_partition's 70x70 order with the pad
     * rows zero (so the qkv linear yields the reference's bias on them), then qkv, then attention.
     * `hw`/`attw` hold kSamTokP rows. */
    rad_buf b_hw = 0, b_qkvw = 0, b_attw = 0;
    rad_buf b_ids_pad = 0, b_ids_real = 0;   /* pad position -> real row (-1 pad); the inverse */
    rad_buf b_n0 = 0, b_n1 = 0, b_unf3 = 0, b_n2 = 0, b_n3 = 0;
    rad_buf b_unf_2 = 0, b_map2 = 0, b_unf_3 = 0, b_map = 0;
    rad_buf b_c = 0, b_ch = 0, b_cq = 0, b_ca = 0, b_cf = 0, b_cg = 0, b_cg2 = 0;
    rad_buf b_cpos = 0, b_fuse = 0, b_projbuf = 0;
    rad_buf b_ct = 0;   /* a CLIP block's projection output, before its residual add */
    /* CLIP's pre_layrnorm output: the first layer's norm1 reads this, every later one the residual
     * stream (deepencoder's VitModel normalises once, before the transformer). */
    rad_buf b_cp = 0;
    rad_buf b_one = 0, b_one_c = 0, b_c1702 = 0;
    rad_buf b_ids_sam = 0, b_ids_clip = 0;
    rad_buf b_cu_sam = 0, b_cu_clip = 0, b_cu_pad = 0;
    rad_buf b_out = 0;
    int64_t max_patches = 0;
    int64_t max_views   = 0;
    bool    on = false;

    /* ------------------------------------------------------------------ stage taps
     * RADIANCE_DEBUG_MM_BUFFERS names intermediate stages (comma separated; a name matches a tag
     * by substring) and RADIANCE_DEBUG_MM_DUMP's directory receives <tag>.bf16: raw bf16 rows, one
     * file a stage, for a stage-by-stage comparison against a reference encoder. The copy is
     * issued on the pass's stream behind the producing op and synced, so it reads the op's output
     * (unlike the after-pass dump in core/engine_mm.cpp, which sees each buffer's last writer). */
    static bool tap_names(const char** list, size_t* n) {
        static const char* const env = std::getenv("RADIANCE_DEBUG_MM_BUFFERS");
        static const char* const dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
        *list = env;
        *n = 0;
        if (!env || !*env || !dir || !*dir) return false;
        return true;
    }
    static bool tap_wants(const char* tag) {
        const char* list; size_t n;
        if (!tap_names(&list, &n)) return false;
        const size_t tl = std::strlen(tag);
        for (const char* s = list; *s; ) {
            const char* e = s;
            while (*e && *e != ',') ++e;
            const size_t l = (size_t)(e - s);
            if (l) {
                for (size_t i = 0; i + l <= tl; ++i)
                    if (std::strncmp(tag + i, s, l) == 0) return true;
            }
            s = *e ? e + 1 : e;
        }
        return false;
    }
    void tap(RadCtx* c, const char* tag, rad_buf b, int64_t rows, int64_t cols,
             int64_t off = 0) const {
        if (rows <= 0 || cols <= 0 || !tap_wants(tag)) return;
        void* p = rad_buf_ptr(c, b);
        if (!p) return;
        std::vector<uint16_t> h((size_t)(rows * cols));
        if (rad_memcpy_async(h.data(), (const char*)p + off * cols * 2,
                             (size_t)(rows * cols * 2), rad_stream(c)) < 0) return;
        rad_stream_sync(rad_stream(c));
        char path[512];
        std::snprintf(path, sizeof path, "%s/tap.%s.bf16", std::getenv("RADIANCE_DEBUG_MM_DUMP"),
                      tag);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(h.data(), sizeof(uint16_t), h.size(), f);
            std::fclose(f);
        }
    }

    /* ------------------------------------------------------------------ probe / weights */
    /* SAM's attention has no op unless one is added (the note above states it). */
    static bool probe(RadBuilder* b) { return rad_op_declared(b, "sam_attn") != 0; }

    static int wcopy(RadBuilder* b, const char* dn, const char* ck,
                     std::initializer_list<int64_t> sh, rad_weight* w, RadWeightGroup g,
                     int access = RAD_ACCESS_PER_REQUEST) {
        RAD_ARCH_TRY(map_copy(b, dn, ck));
        *w = decl_w(b, dn, RAD_BF16, sh, access, RAD_SHARD_NONE, g);
        return *w ? RAD_OK : RAD_E_INVAL;
    }

    /* Every vision tensor, at the checkpoint's names. Always declared, whether or not the tower
     * runs, so a converted container is complete for the pass that will. */
    int declare_weights(RadBuilder* b, Names& nm, int grp_base) {
        int rc;
        const RadWeightGroup gm = grp_model();
        if ((rc = wcopy(b, nm.f("sam.patch_embed.weight"),
                        nm.ckpt("model.sam_model.patch_embed.proj.weight"), {kSamW, kPatchDim},
                        &w_patch, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.patch_embed.bias"),
                        nm.ckpt("model.sam_model.patch_embed.proj.bias"), {kSamW}, &b_patch,
                        gm)) != RAD_OK) return rc;
        /* pos_embed is [1, 64, 64, 768]: one bf16 plane reshaped, which decl_weight admits. */
        if ((rc = wcopy(b, nm.f("sam.pos_embed"), nm.ckpt("model.sam_model.pos_embed"),
                        {kSamTok, kSamW}, &w_pos, gm, RAD_ACCESS_VOCAB)) != RAD_OK) return rc;

        sam.assign((size_t)kSamDepth, SamLayer{});
        for (int64_t i = 0; i < kSamDepth; ++i) {
            SamLayer& l = sam[(size_t)i];
            const RadWeightGroup g = grp_layer(grp_base + (int)i);
            const long long L = (long long)i;
            l.rel_rows = global_block(i) ? 2 * kGrid - 1 : 2 * kWin - 1;
            l.window   = global_block(i) ? 0 : kWin;
            /* One checkpoint tensor, its declared name and shape. */
            auto SW = [&](const char* dn, const char* ck, std::initializer_list<int64_t> sh,
                          rad_weight* w) -> int {
                char d[96], k[160];
                std::snprintf(d, sizeof d, "sam.blk.%lld.%s", L, dn);
                std::snprintf(k, sizeof k, "model.sam_model.blocks.%lld.%s", L, ck);
                return wcopy(b, nm.f("%s", d), nm.ckpt("%s", k), sh, w, g);
            };
            if ((rc = SW("attn_norm.weight", "norm1.weight", {kSamW}, &l.w_n1)) != RAD_OK) return rc;
            if ((rc = SW("attn_norm.bias", "norm1.bias", {kSamW}, &l.b_n1)) != RAD_OK) return rc;
            if ((rc = SW("attn.qkv.weight", "attn.qkv.weight", {3 * kSamW, kSamW},
                         &l.w_qkv)) != RAD_OK) return rc;
            if ((rc = SW("attn.qkv.bias", "attn.qkv.bias", {3 * kSamW}, &l.b_qkv)) != RAD_OK)
                return rc;
            if ((rc = SW("attn.proj.weight", "attn.proj.weight", {kSamW, kSamW},
                         &l.w_proj)) != RAD_OK) return rc;
            if ((rc = SW("attn.proj.bias", "attn.proj.bias", {kSamW}, &l.b_proj)) != RAD_OK)
                return rc;
            if ((rc = SW("attn.rel_pos_h", "attn.rel_pos_h", {l.rel_rows, kSamHd},
                         &l.w_rh)) != RAD_OK) return rc;
            if ((rc = SW("attn.rel_pos_w", "attn.rel_pos_w", {l.rel_rows, kSamHd},
                         &l.w_rw)) != RAD_OK) return rc;
            if ((rc = SW("mlp_norm.weight", "norm2.weight", {kSamW}, &l.w_n2)) != RAD_OK) return rc;
            if ((rc = SW("mlp_norm.bias", "norm2.bias", {kSamW}, &l.b_n2)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin1.weight", "mlp.lin1.weight", {kSamFf, kSamW},
                         &l.w_l1)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin1.bias", "mlp.lin1.bias", {kSamFf}, &l.b_l1)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin2.weight", "mlp.lin2.weight", {kSamW, kSamFf},
                         &l.w_l2)) != RAD_OK) return rc;
            if ((rc = SW("mlp.lin2.bias", "mlp.lin2.bias", {kSamW}, &l.b_l2)) != RAD_OK) return rc;
        }

        if ((rc = wcopy(b, nm.f("sam.neck.0.weight"), nm.ckpt("model.sam_model.neck.0.weight"),
                        {256, kSamW}, &w_neck0, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.1.weight"), nm.ckpt("model.sam_model.neck.1.weight"),
                        {256}, &w_nk1, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.1.bias"), nm.ckpt("model.sam_model.neck.1.bias"),
                        {256}, &b_nk1, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.2.weight"), nm.ckpt("model.sam_model.neck.2.weight"),
                        {256, 256 * 9}, &w_nk2, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.3.weight"), nm.ckpt("model.sam_model.neck.3.weight"),
                        {256}, &w_nk3, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.neck.3.bias"), nm.ckpt("model.sam_model.neck.3.bias"),
                        {256}, &b_nk3, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.net_2.weight"), nm.ckpt("model.sam_model.net_2.weight"),
                        {512, 256 * 9}, &w_net2, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("sam.net_3.weight"), nm.ckpt("model.sam_model.net_3.weight"),
                        {1024, 512 * 9}, &w_net3, gm)) != RAD_OK) return rc;

        const char* cp = "model.vision_model.";
        if ((rc = wcopy(b, nm.f("clip.cls"), nm.ckpt("%sembeddings.class_embedding", cp),
                        {kClipW}, &w_cls, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("clip.pos_embed"),
                        nm.ckpt("%sembeddings.position_embedding.weight", cp),
                        {kClipTok, kClipW}, &w_cpos, gm, RAD_ACCESS_VOCAB)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("clip.pre_norm.weight"), nm.ckpt("%spre_layrnorm.weight", cp),
                        {kClipW}, &w_playr, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("clip.pre_norm.bias"), nm.ckpt("%spre_layrnorm.bias", cp),
                        {kClipW}, &b_playr, gm)) != RAD_OK) return rc;

        clip.assign((size_t)kClipDepth, ClipLayer{});
        for (int64_t i = 0; i < kClipDepth; ++i) {
            ClipLayer& l = clip[(size_t)i];
            const RadWeightGroup g = grp_layer(grp_base + 32 + (int)i);
            const long long L = (long long)i;
            auto CW = [&](const char* dn, const char* ck, std::initializer_list<int64_t> sh,
                          rad_weight* w) -> int {
                char d[96], k[192];
                std::snprintf(d, sizeof d, "clip.blk.%lld.%s", L, dn);
                std::snprintf(k, sizeof k, "model.vision_model.transformer.layers.%lld.%s", L, ck);
                return wcopy(b, nm.f("%s", d), nm.ckpt("%s", k), sh, w, g);
            };
            if ((rc = CW("attn_norm.weight", "layer_norm1.weight", {kClipW}, &l.w_n1)) != RAD_OK)
                return rc;
            if ((rc = CW("attn_norm.bias", "layer_norm1.bias", {kClipW}, &l.b_n1)) != RAD_OK)
                return rc;
            if ((rc = CW("attn.qkv.weight", "self_attn.qkv_proj.weight", {3 * kClipW, kClipW},
                         &l.w_qkv)) != RAD_OK) return rc;
            if ((rc = CW("attn.qkv.bias", "self_attn.qkv_proj.bias", {3 * kClipW},
                         &l.b_qkv)) != RAD_OK) return rc;
            if ((rc = CW("attn.out.weight", "self_attn.out_proj.weight", {kClipW, kClipW},
                         &l.w_out)) != RAD_OK) return rc;
            if ((rc = CW("attn.out.bias", "self_attn.out_proj.bias", {kClipW}, &l.b_out)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp_norm.weight", "layer_norm2.weight", {kClipW}, &l.w_n2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp_norm.bias", "layer_norm2.bias", {kClipW}, &l.b_n2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc1.weight", "mlp.fc1.weight", {kClipFf, kClipW}, &l.w_fc1)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc1.bias", "mlp.fc1.bias", {kClipFf}, &l.b_fc1)) != RAD_OK) return rc;
            if ((rc = CW("mlp.fc2.weight", "mlp.fc2.weight", {kClipW, kClipFf}, &l.w_fc2)) != RAD_OK)
                return rc;
            if ((rc = CW("mlp.fc2.bias", "mlp.fc2.bias", {kClipW}, &l.b_fc2)) != RAD_OK) return rc;
        }

        if ((rc = wcopy(b, nm.f("proj.weight"), nm.ckpt("model.projector.layers.weight"),
                        {kNEmbd, kFused}, &w_proj, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("proj.bias"), nm.ckpt("model.projector.layers.bias"),
                        {kNEmbd}, &b_proj, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("layout.image_newline"), nm.ckpt("model.image_newline"),
                        {kNEmbd}, &w_newline, gm)) != RAD_OK) return rc;
        if ((rc = wcopy(b, nm.f("layout.view_seperator"), nm.ckpt("model.view_seperator"),
                        {kNEmbd}, &w_sep, gm)) != RAD_OK) return rc;
        return RAD_OK;
    }

    /* ------------------------------------------------------------------ op builders */
    static rad_op gemm_b(RadBuilder* b, int64_t max_m, int64_t N, int64_t K, const char* act,
                         rad_weight w, rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("act", act), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    }
    static rad_op gemm_u(RadBuilder* b, int64_t max_m, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    }
    static rad_op lnorm(RadBuilder* b, int64_t max_m, int64_t n, float eps, rad_weight w,
                        rad_weight bias) {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_m), RAD_INT("n", n), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    }
    /* A bias-free 3x3 convolution as conv2d_unfold + gemm_nt: the unfolded rows are exactly what
     * a GEMM against the checkpoint's [cout, cin, 3, 3] weight (read as [cout, cin*9]) needs. */
    static rad_op unfold(RadBuilder* b, int64_t h, int64_t w, int64_t cin, int64_t stride,
                         int64_t cu_div) {
        return RAD_OP(b, "conv2d_unfold",
                      RAD_PARAMS(RAD_INT("M", h), RAD_INT("width", w), RAD_INT("cin", cin),
                                 RAD_INT("kh", 3), RAD_INT("kw", 3), RAD_INT("stride", stride),
                                 RAD_INT("pad", 1), RAD_INT("cu_div", cu_div),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    }

    int declare_ops(RadBuilder* b) {
        /* ---- activations, one view's. */
        b_x   = decl_b(b, "v.sam.x",   RAD_BF16, {kSamTok, kSamW});
        b_h   = decl_b(b, "v.sam.h",   RAD_BF16, {kSamTok, kSamW});
        b_qkv = decl_b(b, "v.sam.qkv", RAD_BF16, {kSamTok, 3 * kSamW});
        b_att = decl_b(b, "v.sam.att", RAD_BF16, {kSamTok, kSamW});
        b_ff  = decl_b(b, "v.sam.ff",  RAD_BF16, {kSamTok, kSamFf});
        b_sampos  = decl_b(b, "v.sam.pos", RAD_BF16, {kSamTok, kSamW});
        b_ids_sam = decl_b(b, "v.sam.ids", RAD_I32,  {kSamTok});
        b_hw    = decl_b(b, "v.sam.hw",   RAD_BF16, {kSamTokP, kSamW});
        b_qkvw  = decl_b(b, "v.sam.qkvw", RAD_BF16, {kSamTokP, 3 * kSamW});
        b_attw  = decl_b(b, "v.sam.attw", RAD_BF16, {kSamTokP, kSamW});
        b_ids_pad  = decl_b(b, "v.sam.ids.pad",  RAD_I32, {kSamTokP}, RAD_BUF_PERSIST);
        b_ids_real = decl_b(b, "v.sam.ids.real", RAD_I32, {kSamTok}, RAD_BUF_PERSIST);

        b_n0   = decl_b(b, "v.neck.0",   RAD_BF16, {kSamTok, 256});
        b_n1   = decl_b(b, "v.neck.1",   RAD_BF16, {kSamTok, 256});
        b_unf3 = decl_b(b, "v.neck.unf", RAD_BF16, {kSamTok, 256 * 9});
        b_n2   = decl_b(b, "v.neck.2",   RAD_BF16, {kSamTok, 256});
        b_n3   = decl_b(b, "v.neck.3",   RAD_BF16, {kSamTok, 256});
        b_unf_2 = decl_b(b, "v.net2.unf", RAD_BF16, {kSamTok / 4, 256 * 9});
        b_map2  = decl_b(b, "v.net2.out", RAD_BF16, {kSamTok / 4, 512});
        b_unf_3 = decl_b(b, "v.net3.unf", RAD_BF16, {kMap * kMap, 512 * 9});
        b_map   = decl_b(b, "v.map",      RAD_BF16, {kMap * kMap, kClipW});

        b_c   = decl_b(b, "v.clip.x",   RAD_BF16, {kClipTok, kClipW});
        b_ch  = decl_b(b, "v.clip.h",   RAD_BF16, {kClipTok, kClipW});
        b_cq  = decl_b(b, "v.clip.qkv", RAD_BF16, {kClipTok, 3 * kClipW});
        b_ca  = decl_b(b, "v.clip.att", RAD_BF16, {kClipTok, kClipW});
        b_cf  = decl_b(b, "v.clip.ff",  RAD_BF16, {kClipTok, kClipFf});
        b_cg  = decl_b(b, "v.clip.g",   RAD_BF16, {kClipTok, kClipFf});
        b_cg2 = decl_b(b, "v.clip.g2",  RAD_BF16, {kClipTok, kClipFf});
        b_cpos = decl_b(b, "v.clip.pos", RAD_BF16, {kClipTok, kClipW});
        b_ct   = decl_b(b, "v.clip.tmp", RAD_BF16, {kClipTok, kClipW});
        b_cp   = decl_b(b, "v.clip.pre", RAD_BF16, {kClipTok, kClipW});
        b_ids_clip = decl_b(b, "v.clip.ids", RAD_I32, {kClipTok});

        b_fuse    = decl_b(b, "v.fuse", RAD_BF16, {kMap * kMap, kFused});
        b_projbuf = decl_b(b, "v.proj", RAD_BF16, {kMap * kMap, kNEmbd});

        b_one    = decl_b(b, "v.one",       RAD_BF16, {1, kNEmbd});
        b_one_c  = decl_b(b, "v.one.clip",  RAD_BF16, {1, kClipW});
        b_c1702  = decl_b(b, "v.one.1702",  RAD_BF16, {1, kClipFf});
        b_cu_sam  = decl_b(b, "v.cu.sam",  RAD_I32, {2});
        b_cu_clip = decl_b(b, "v.cu.clip", RAD_I32, {2});
        b_cu_pad  = decl_b(b, "v.cu.pad",  RAD_I32, {2});

        /* ---- the pass's constants, declared FIRST because they are issued first. Liveness is
         * computed over this op order, so an op declared after its readers leaves its buffer with
         * a live range that begins at the reader: the packer then aliases that buffer with an
         * activation written in between, and the value is gone by the time it is read (this is
         * exactly what corrupted the cu_seqlens tables while they were declared at the bottom). */
        op_fill_one = rw(b, RAD_OP(b, "fill",
                           RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kNEmbd),
                                      RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                           RAD_NOWEIGHTS),
                         {}, {b_one});
        op_fill_one_c = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kClipW),
                                        RAD_STR("dtype", "bf16"), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                           {}, {b_one_c});
        op_fill_1702 = rw(b, RAD_OP(b, "fill",
                            RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kClipFf),
                                       RAD_STR("dtype", "bf16"), RAD_F64("value", 1.702)),
                            RAD_NOWEIGHTS),
                          {}, {b_c1702});
        /* One cu_seqlens table each, {0, tokens}, one segment: `iota` writes both entries in one
         * op. The three unfolds read b_cu_sam through their `cu_div` (4096/64 = 64 rows for the
         * neck and net_2 maps, 4096/128 = 32 for net_3's), so one table serves all of them. */
        op_cu_sam = rw(b, RAD_OP(b, "iota",
                          RAD_PARAMS(RAD_INT("M", 2), RAD_INT("start", 0), RAD_INT("step", kSamTok)),
                          RAD_NOWEIGHTS),
                       {}, {b_cu_sam});
        op_cu_clip = rw(b, RAD_OP(b, "iota",
                           RAD_PARAMS(RAD_INT("M", 2), RAD_INT("start", 0),
                                      RAD_INT("step", kClipTok)),
                           RAD_NOWEIGHTS),
                        {}, {b_cu_clip});
        op_cu_pad = rw(b, RAD_OP(b, "iota",
                          RAD_PARAMS(RAD_INT("M", 2), RAD_INT("start", 0),
                                     RAD_INT("step", kSamTokP)),
                          RAD_NOWEIGHTS),
                       {}, {b_cu_pad});

        /* ---- SAM stem. The patch embed is one GEMM a grid row (the staged rows carry kDown
         * zero patch columns nothing reads), then the 64x64 position table added in: at this
         * grid the reference's get_abs_pos_sam returns the table unchanged, so a row gather is
         * exact. */
        op_patch = rw(b, gemm_b(b, kGrid, kSamW, kPatchDim, "none", w_patch, b_patch), {}, {b_x});
        op_ids_sam = rw(b, RAD_OP(b, "iota",
                         RAD_PARAMS(RAD_INT("M", kSamTok), RAD_INT("start", 0),
                                    RAD_INT("step", 1)),
                         RAD_NOWEIGHTS),
                     {}, {b_ids_sam});
        op_sampos = rw(b, RAD_OP(b, "embed_lookup",
                         RAD_PARAMS(RAD_RANGE("M", 1, kSamTok), RAD_INT("n_embd", kSamW),
                                    RAD_INT("n_vocab", kSamTok), RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(w_pos)),
                     {b_ids_sam}, {b_sampos});
        op_sampadd = rw(b, RAD_OP(b, "add",
                          RAD_PARAMS(RAD_RANGE("M", 1, kSamTok), RAD_INT("n", kSamW),
                                     RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS),
                      {b_x, b_sampos}, {b_x});

        for (SamLayer& l : sam) {
            l.op_n1 = rw(b, lnorm(b, kSamTok, kSamW, 1e-6f, l.w_n1, l.b_n1), {b_x}, {b_h});
            /* The windowed path: `h` into window_partition's padded 70x70 order with zero pad rows
             * (a negative gather index is a zero row), so the qkv linear gives the pad rows the
             * reference's bias -- deepencoder pads AFTER norm1 and runs the whole attention on the
             * padded window, so a pad key's k and v are the qkv bias, not zero. */
            l.op_gpad = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, kSamTokP), RAD_INT("n", kSamW),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                          {b_h, b_ids_pad}, {b_hw});
            l.op_qkv = rw(b, gemm_b(b, kSamTok, 3 * kSamW, kSamW, "none", l.w_qkv, l.b_qkv),
                          {b_h}, {b_qkv});
            l.op_qkvw = rw(b, gemm_b(b, kSamTokP, 3 * kSamW, kSamW, "none", l.w_qkv, l.b_qkv),
                           {b_hw}, {b_qkvw});
            l.op_attn = rw(b, RAD_OP(b, "sam_attn",
                              RAD_PARAMS(RAD_RANGE("M", 1, kSamTok),
                                         RAD_INT("head_dim", kSamHd),
                                         RAD_INT("n_head", kSamHeads),
                                         RAD_INT("grid_h", kGrid), RAD_INT("grid_w", kGrid),
                                         RAD_INT("window", l.window),
                                         RAD_F64("scale", 1.0 / 8.0), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(l.w_rh, l.w_rw)),
                          {b_qkv, b_cu_sam}, {b_att});
            l.op_attnw = rw(b, RAD_OP(b, "sam_attn",
                               RAD_PARAMS(RAD_RANGE("M", 1, kSamTokP),
                                          RAD_INT("head_dim", kSamHd),
                                          RAD_INT("n_head", kSamHeads),
                                          RAD_INT("grid_h", kGridP), RAD_INT("grid_w", kGridP),
                                          RAD_INT("window", kWin),
                                          RAD_F64("scale", 1.0 / 8.0), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(l.w_rh, l.w_rw)),
                           {b_qkvw, b_cu_pad}, {b_attw});
            l.op_greal = rw(b, RAD_OP(b, "gather_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, kSamTok), RAD_INT("n", kSamW),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                            {b_attw, b_ids_real}, {b_att});
            l.op_proj = rw(b, gemm_b(b, kSamTok, kSamW, kSamW, "none", l.w_proj, l.b_proj),
                           {b_att}, {b_h});
            /* deepencoder rounds the projection before adding the shortcut (`x + attn(norm(x))`
             * is two bf16 operations), so the residual add is its own op: fusing it into the
             * GEMM's f32 accumulator skips a rounding and drifts from the reference. */
            l.op_padd = rw(b, RAD_OP(b, "add",
                             RAD_PARAMS(RAD_RANGE("M", 1, kSamTok), RAD_INT("n", kSamW),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                          {b_h, b_x}, {b_x});
            l.op_n2 = rw(b, lnorm(b, kSamTok, kSamW, 1e-6f, l.w_n2, l.b_n2), {b_x}, {b_h});
            l.op_l1 = rw(b, gemm_b(b, kSamTok, kSamFf, kSamW, "gelu", l.w_l1, l.b_l1), {b_h},
                         {b_ff});
            l.op_l2 = rw(b, gemm_b(b, kSamTok, kSamW, kSamFf, "none", l.w_l2, l.b_l2),
                         {b_ff}, {b_h});
            l.op_ladd = rw(b, RAD_OP(b, "add",
                             RAD_PARAMS(RAD_RANGE("M", 1, kSamTok), RAD_INT("n", kSamW),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                          {b_h, b_x}, {b_x});
        }

        /* ---- neck. The 1x1 is a GEMM; the 3x3s are unfold + gemm because they carry no bias and
         * conv2d_nhwc's `b` is required. LayerNorm2d is a layernorm across a pixel's channels,
         * which are innermost. */
        op_neck0 = rw(b, gemm_u(b, kSamTok, 256, kSamW, w_neck0), {b_x}, {b_n0});
        op_nk1 = rw(b, lnorm(b, kSamTok, 256, 1e-6f, w_nk1, b_nk1), {b_n0}, {b_n1});
        op_unf3 = rw(b, unfold(b, kGrid, kGrid, 256, 1, kGrid), {b_n1, b_cu_sam}, {b_unf3});
        op_nk2 = rw(b, gemm_u(b, kSamTok, 256, 256 * 9, w_nk2), {b_unf3}, {b_n2});
        op_nk3 = rw(b, lnorm(b, kSamTok, 256, 1e-6f, w_nk3, b_nk3), {b_n2}, {b_n3});
        op_unf_net2 = rw(b, unfold(b, kGrid, kGrid, 256, 2, kGrid), {b_n3, b_cu_sam}, {b_unf_2});
        op_net2 = rw(b, gemm_u(b, kSamTok / 4, 512, 256 * 9, w_net2), {b_unf_2}, {b_map2});
        op_unf_net3 = rw(b, unfold(b, kGrid / 2, kGrid / 2, 512, 2, kGrid * 2),
                         {b_map2, b_cu_sam}, {b_unf_3});
        op_net3 = rw(b, gemm_u(b, kMap * kMap, kClipW, 512 * 9, w_net3), {b_unf_3}, {b_map});

        /* ---- CLIP. The class row goes first and the map's rows below it: absolute position
         * pairs row for row, and attention is permutation-equivariant, so the class row's place
         * is free. (The reference cats class first too.) */
        op_cls = rw(b, RAD_OP(b, "mul_weight",
                       RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kClipW), RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(w_cls)),
                    {b_one_c}, {b_c});
        op_cmap = rw(b, RAD_OP(b, "cast",
                        RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", kClipW),
                                   RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                        RAD_NOWEIGHTS),
                     {b_map}, {b_c});
        op_ids_clip = rw(b, RAD_OP(b, "iota",
                          RAD_PARAMS(RAD_INT("M", kClipTok), RAD_INT("start", 0),
                                     RAD_INT("step", 1)),
                          RAD_NOWEIGHTS),
                      {}, {b_ids_clip});
        op_cpos = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n_embd", kClipW),
                                   RAD_INT("n_vocab", kClipTok), RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w_cpos)),
                     {b_ids_clip}, {b_cpos});
        op_cadd = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n", kClipW),
                                   RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                     {b_c, b_cpos}, {b_c});
        op_playr = rw(b, lnorm(b, kClipTok, kClipW, 1e-5f, w_playr, b_playr), {b_c}, {b_cp});
        /* deepencoder normalises the embedding once (pre_layrnorm) and the transformer's residual
         * stream starts there, not at the embedding: layer 0's norm1 and its residual add both
         * read the pre_layrnorm output. */
        op_ccopy = rw(b, RAD_OP(b, "cast",
                        RAD_PARAMS(RAD_INT("M", kClipTok), RAD_INT("n", kClipW),
                                   RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                        RAD_NOWEIGHTS),
                     {b_cp}, {b_c});

        for (ClipLayer& l : clip) {
            l.op_n1 = rw(b, lnorm(b, kClipTok, kClipW, 1e-5f, l.w_n1, l.b_n1), {b_c}, {b_ch});
            l.op_qkv = rw(b, gemm_b(b, kClipTok, 3 * kClipW, kClipW, "none", l.w_qkv, l.b_qkv),
                          {b_ch}, {b_cq});
            l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                              RAD_PARAMS(RAD_RANGE("M", 1, kClipTok),
                                         RAD_INT("head_dim", kClipHd), RAD_INT("gqa", 1),
                                         RAD_INT("causal", 0), RAD_F64("scale", 1.0 / 8.0),
                                         RAD_INT("cu_div", 1), RAD_INT("max_seqlen", kClipTok),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                          {b_cq, b_cu_clip}, {b_ca});
            l.op_out = rw(b, gemm_b(b, kClipTok, kClipW, kClipW, "none", l.w_out, l.b_out),
                          {b_ca}, {b_ct});
            l.op_cadd = rw(b, RAD_OP(b, "add",
                             RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n", kClipW),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                          {b_ct, b_c}, {b_c});
            l.op_n2 = rw(b, lnorm(b, kClipTok, kClipW, 1e-5f, l.w_n2, l.b_n2), {b_c}, {b_ch});
            l.op_fc1 = rw(b, gemm_b(b, kClipTok, kClipFf, kClipW, "none", l.w_fc1, l.b_fc1),
                          {b_ch}, {b_cf});
            /* quick_gelu: fc1's output times sigmoid(1.702 * itself). */
            l.op_scale = rw(b, RAD_OP(b, "mul",
                              RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n", kClipFf),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                          {b_cf, b_c1702}, {b_cg});
            l.op_qg = rw(b, RAD_OP(b, "act_mul",
                           RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n", kClipFf),
                                      RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                        {b_cg, b_cf}, {b_cg2});
            l.op_fc2 = rw(b, gemm_b(b, kClipTok, kClipW, kClipFf, "none", l.w_fc2, l.b_fc2),
                          {b_cg2}, {b_ct});
            l.op_madd = rw(b, RAD_OP(b, "add",
                             RAD_PARAMS(RAD_RANGE("M", 1, kClipTok), RAD_INT("n", kClipW),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                          {b_ct, b_c}, {b_c});
        }

        /* ---- fuse, project. */
        op_fuse_clip = rw(b, RAD_OP(b, "cast",
                            RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", kClipW),
                                       RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                            RAD_NOWEIGHTS),
                         {b_c}, {b_fuse});
        op_fuse_sam = rw(b, RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", kClipW),
                                      RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                           RAD_NOWEIGHTS),
                        {b_map}, {b_fuse});
        op_proj = rw(b, gemm_b(b, kMap * kMap, kNEmbd, kFused, "none", w_proj, b_proj), {b_fuse},
                     {b_projbuf});
        /* Debug taps: one stage's 256 rows into the out buffer instead of the layout. */
        op_dbg1024 = rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", kClipW),
                                     RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                          RAD_NOWEIGHTS),
                        {b_map, b_c, b_unf_2}, {b_out});
        op_dbg768 = rw(b, RAD_OP(b, "cast",
                         RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", kSamW),
                                    RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_x, b_att}, {b_out});
        op_dbg256 = rw(b, RAD_OP(b, "cast",
                         RAD_PARAMS(RAD_INT("M", 256), RAD_INT("n", 256),
                                    RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                         RAD_NOWEIGHTS),
                       {b_n0, b_n1, b_n2, b_n3, b_map2, b_unf_2}, {b_out});
        op_dbg1280 = rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_INT("M", kMap * kMap), RAD_INT("n", kNEmbd),
                                     RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                          RAD_NOWEIGHTS),
                        {b_projbuf}, {b_out});

        /* A projected map line into its place among the layout rows. */
        op_line = rw(b, RAD_OP(b, "cast",
                        RAD_PARAMS(RAD_INT("M", kMap), RAD_INT("n", kNEmbd),
                                   RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                        RAD_NOWEIGHTS),
                     {b_projbuf}, {b_out});

        /* ---- constants and the two layout rows. */
        op_newline = rw(b, RAD_OP(b, "mul_weight",
                          RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kNEmbd),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w_newline)),
                        {b_one}, {b_out});
        op_sep = rw(b, RAD_OP(b, "mul_weight",
                      RAD_PARAMS(RAD_INT("M", 1), RAD_INT("n", kNEmbd), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w_sep)),
                    {b_one}, {b_out});
        return RAD_OK;
    }


    /* ------------------------------------------------------------------ declare */
    /* Maps and declares the tower's weights always; declares the ops and the encoder only when
     * SAM's attention op exists. `grp_base` is a free movement-group number. */
    int declare(RadBuilder* b, const RadBuildCtx* ctx, Names& nm, int64_t n_embd_model,
                int grp_base) {
        RAD_ARCH_TRY(declare_weights(b, nm, grp_base));
        if (n_embd_model != kNEmbd) {
            fprintf(stderr, "radiance: unlimitedocr: the decoder's hidden size %lld is not the "
                            "DeepEncoder's %lld\n", (long long)n_embd_model, (long long)kNEmbd);
            return RAD_E_UNSUPPORTED;
        }
        if (!probe(b)) {
            rad_note(b, "the DeepEncoder tower is NOT declared: SAM's attention needs the "
                        "`sam_attn` op (windowed 14x14 / global 64x64 attention with decomposed "
                        "rel_pos_h/rel_pos_w), which no loaded kernel library provides. Its "
                        "weights are mapped and declared, so a converted container carries them; "
                        "images are refused until the op lands");
            on = false;
            return RAD_OK;
        }
        max_patches = ctx->max_enc_patches;
        if (max_patches < kSegView) {
            /* --mm-max-patches 0 (text only) or a bound below one 1024x1024 view: the tower stays
             * off and the trunk serves text, which is what the flag asks for. */
            rad_note(b, "the DeepEncoder tower is NOT declared: one 1024x1024 view is %lld staged "
                        "patches and --mm-max-patches is %lld; images are refused",
                     (long long)kSegView, (long long)max_patches);
            on = false;
            return RAD_OK;
        }
        max_views = max_patches / kSegView;
        b_out = decl_b(b, "v.out", RAD_BF16, {max_patches / kMerge, kNEmbd});
        if (const char* e = getenv("RAD_UNLIMITEDOCR_DBG")) dbg = atoi(e);
        if (const char* e = getenv("RAD_UNLIMITEDOCR_DBG_ROW")) dbg_row = atoi(e);
        RAD_ARCH_TRY(declare_ops(b));
        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name        = "deepencoder";
            e.modalities  = RAD_MM_IMAGE;
            e.patch_dim   = kPatchDim;
            e.merge       = kMerge;
            e.max_patches = max_patches;
            e.n_embd      = kNEmbd;
            e.out         = b_out;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        }
        on = true;
        rad_note(b, "DeepEncoder: SAM ViT-B/16 at 1024 (%lld blocks of %lld, %lld heads x %lld, "
                    "window %lld on a %lldx%lld grid, global on 2/5/8/11, rel_pos tables 27/127 "
                    "rows) -> 16x16x1024 map -> CLIP ViT-L/14 (%lld layers of %lld, %lld heads x "
                    "%lld, %lld tokens) -> %lld -> projector -> %lld; %lld patches a view "
                    "(64x%lld staged, %lld patches a row) -> %lld rows a view",
                 (long long)kSamDepth, (long long)kSamW, (long long)kSamHeads, (long long)kSamHd,
                 (long long)kWin, (long long)kGrid, (long long)kGrid, (long long)kClipDepth,
                 (long long)kClipW, (long long)kClipHeads, (long long)kClipHd,
                 (long long)kClipTok, (long long)kFused, (long long)kNEmbd,
                 (long long)kPatchView, (long long)(kGrid + kDown), (long long)kMerge,
                 (long long)kRowsView);
        return RAD_OK;
    }

    /* ------------------------------------------------------------------ step */
    /* One view: the staged 64 x (64+4) patch grid -> 273 rows at `rowbase`. */
    void view(RadCtx* c, const RadBatch* batch, int64_t v) const {
        const int64_t pbase = v * kSegView;
        const int64_t rowbase = pbase / kMerge;
        const char* pix = (const char*)batch->enc_pixels + pbase * kPatchDim * 2;

        /* The patch embed, one grid row at a time: the kDown zero patch columns the processor
         * stages at the end of each row are skipped, not read. */
        for (int64_t r = 0; r < kGrid; ++r)
            RAD_ISSUE_N(c, op_patch, kGrid,
                        RAD_P_T2(pix + (r * (kGrid + kDown)) * kPatchDim * 2, RAD_BF16, kGrid,
                                 kPatchDim),
                        RAD_W(w_patch), RAD_W(b_patch), RAD_NONE,
                        brow_slice(b_x, r * kGrid, kGrid, kSamW));

        /* The 64x64 position table, a row for a token (a gather, exact at this grid), added in. */
        RAD_ISSUE(c, op_ids_sam, RAD_B(b_ids_sam));
        RAD_ISSUE_N(c, op_sampos, kSamTok, RAD_B(b_ids_sam), RAD_W(w_pos), brows(b_sampos, kSamTok));
        RAD_ISSUE_N(c, op_sampadd, kSamTok, brows(b_x, kSamTok), brows(b_sampos, kSamTok),
                    brows(b_x, kSamTok));
        tap(c, "st_x0", b_x, 256, kSamW);

        if (dbg == 10) { RAD_ISSUE(c, op_dbg768, brow_slice(b_x, dbg_row, 256, kSamW), bcol(b_out, 0, kSamW, 256)); return; }

        for (size_t li = 0; li < sam.size(); ++li) {
            const SamLayer& l = sam[li];
            RAD_ISSUE_N(c, l.op_n1, kSamTok, brows(b_x, kSamTok), RAD_W(l.w_n1), RAD_W(l.b_n1),
                        brows(b_h, kSamTok));
            if (l.window > 0) {
                /* The reference's window path: pad the normed grid to 70x70, run qkv over the pad
                 * rows too (they carry the qkv bias), attend inside each 14x14 window, and take
                 * the real rows back out. The pad rows take softmax mass with the bias as v. */
                RAD_ISSUE_N(c, l.op_gpad, kSamTokP, brows(b_h, kSamTok), RAD_B(b_ids_pad),
                            brows(b_hw, kSamTokP));
                RAD_ISSUE_N(c, l.op_qkvw, kSamTokP, brows(b_hw, kSamTokP), RAD_W(l.w_qkv),
                            RAD_W(l.b_qkv), RAD_NONE, brows(b_qkvw, kSamTokP));
                RAD_ISSUE_N(c, l.op_attnw, kSamTokP,
                            bcol(b_qkvw, 0, kSamW, kSamTokP), bcol(b_qkvw, kSamW, kSamW, kSamTokP),
                            bcol(b_qkvw, 2 * kSamW, kSamW, kSamTokP), RAD_B(b_cu_pad),
                            brows(b_attw, kSamTokP), RAD_W(l.w_rh), RAD_W(l.w_rw));
                RAD_ISSUE_N(c, l.op_greal, kSamTok, brows(b_attw, kSamTokP), RAD_B(b_ids_real),
                            brows(b_att, kSamTok));
            } else {
                RAD_ISSUE_N(c, l.op_qkv, kSamTok, brows(b_h, kSamTok), RAD_W(l.w_qkv),
                            RAD_W(l.b_qkv), RAD_NONE, brows(b_qkv, kSamTok));
                RAD_ISSUE_N(c, l.op_attn, kSamTok,
                            bcol(b_qkv, 0, kSamW, kSamTok), bcol(b_qkv, kSamW, kSamW, kSamTok),
                            bcol(b_qkv, 2 * kSamW, kSamW, kSamTok), RAD_B(b_cu_sam),
                            brows(b_att, kSamTok), RAD_W(l.w_rh), RAD_W(l.w_rw));
            }
            char tag[40];
            std::snprintf(tag, sizeof tag, "sam_att%lld_raw", (long long)li);
            tap(c, tag, b_att, 256, kSamW);
            if (dbg >= 50 && dbg < 50 + (int)kSamDepth && dbg == 50 + (int)li) {
                RAD_ISSUE(c, op_dbg768, brow_slice(b_att, dbg_row, 256, kSamW), bcol(b_out, 0, kSamW, 256));
                return;
            }
            RAD_ISSUE_N(c, l.op_proj, kSamTok, brows(b_att, kSamTok), RAD_W(l.w_proj),
                        RAD_W(l.b_proj), RAD_NONE, brows(b_h, kSamTok));
            RAD_ISSUE_N(c, l.op_padd, kSamTok, brows(b_h, kSamTok), brows(b_x, kSamTok),
                        brows(b_x, kSamTok));
            std::snprintf(tag, sizeof tag, "st_pre%lld", (long long)li);
            tap(c, tag, b_x, 256, kSamW);
            if (dbg >= 70 && dbg < 70 + (int)kSamDepth && dbg == 70 + (int)li) {
                RAD_ISSUE(c, op_dbg768, brow_slice(b_x, dbg_row, 256, kSamW), bcol(b_out, 0, kSamW, 256));
                return;
            }
            RAD_ISSUE_N(c, l.op_n2, kSamTok, brows(b_x, kSamTok), RAD_W(l.w_n2), RAD_W(l.b_n2),
                        brows(b_h, kSamTok));
            RAD_ISSUE_N(c, l.op_l1, kSamTok, brows(b_h, kSamTok), RAD_W(l.w_l1), RAD_W(l.b_l1),
                        RAD_NONE, brows(b_ff, kSamTok));
            RAD_ISSUE_N(c, l.op_l2, kSamTok, brows(b_ff, kSamTok), RAD_W(l.w_l2), RAD_W(l.b_l2),
                        RAD_NONE, brows(b_h, kSamTok));
            RAD_ISSUE_N(c, l.op_ladd, kSamTok, brows(b_h, kSamTok), brows(b_x, kSamTok),
                        brows(b_x, kSamTok));
            std::snprintf(tag, sizeof tag, "st_b%lld", (long long)li);
            tap(c, tag, b_x, 256, kSamW);
            if (li < 3) {
                std::snprintf(tag, sizeof tag, "st_full%lld", (long long)li);
                tap(c, tag, b_x, kSamTok, kSamW);
            }
            if (dbg == 11 + (int)li) {
                RAD_ISSUE(c, op_dbg768, brow_slice(b_x, dbg_row, 256, kSamW), bcol(b_out, 0, kSamW, 256));
                return;
            }
        }

        /* neck + net_2 + net_3. The 3x3s are a dense view of the map rows, unfolded. */
        RAD_ISSUE_N(c, op_neck0, kSamTok, brows(b_x, kSamTok), RAD_W(w_neck0),
                    brows(b_n0, kSamTok));
        tap(c, "st_n0", b_n0, 256, 256);
        if (dbg == 30) { RAD_ISSUE(c, op_dbg256, brows(b_n0, 256), bcol(b_out, 0, 256, 256)); return; }
        RAD_ISSUE_N(c, op_nk1, kSamTok, brows(b_n0, kSamTok), RAD_W(w_nk1), RAD_W(b_nk1),
                    brows(b_n1, kSamTok));
        tap(c, "st_n1", b_n1, 256, 256);
        if (dbg == 31) { RAD_ISSUE(c, op_dbg256, brows(b_n1, 256), bcol(b_out, 0, 256, 256)); return; }
        RAD_ISSUE(c, op_unf3, RAD_P_T2(rad_buf_ptr(c, b_n1), RAD_BF16, kGrid, kGrid * 256),
                  RAD_B(b_cu_sam), RAD_NONE, brows(b_unf3, kSamTok));
        RAD_ISSUE_N(c, op_nk2, kSamTok, brows(b_unf3, kSamTok), RAD_W(w_nk2),
                    brows(b_n2, kSamTok));
        tap(c, "st_n2", b_n2, 256, 256);
        if (dbg == 32) { RAD_ISSUE(c, op_dbg256, brows(b_n2, 256), bcol(b_out, 0, 256, 256)); return; }
        RAD_ISSUE_N(c, op_nk3, kSamTok, brows(b_n2, kSamTok), RAD_W(w_nk3), RAD_W(b_nk3),
                    brows(b_n3, kSamTok));
        tap(c, "st_n3", b_n3, 256, 256);
        if (dbg == 33) { RAD_ISSUE(c, op_dbg256, brows(b_n3, 256), bcol(b_out, 0, 256, 256)); return; }
        RAD_ISSUE(c, op_unf_net2, RAD_P_T2(rad_buf_ptr(c, b_n3), RAD_BF16, kGrid, kGrid * 256),
                  RAD_B(b_cu_sam), RAD_NONE, brows(b_unf_2, kSamTok / 4));
        if (dbg == 40) {
            RAD_ISSUE(c, op_dbg1024, brows(b_unf_2, 256), bcol(b_out, 0, kClipW, 256));
            return;
        }
        RAD_ISSUE_N(c, op_net2, kSamTok / 4, brows(b_unf_2, kSamTok / 4), RAD_W(w_net2),
                    brows(b_map2, kSamTok / 4));
        tap(c, "st_map2", b_map2, 256, 512);
        RAD_ISSUE(c, op_unf_net3,
                  RAD_P_T2(rad_buf_ptr(c, b_map2), RAD_BF16, kGrid / 2, (kGrid / 2) * 512),
                  RAD_B(b_cu_sam), RAD_NONE, brows(b_unf_3, kMap * kMap));
        if (dbg == 34) { RAD_ISSUE(c, op_dbg256, brows(b_map2, 256), bcol(b_out, 0, 256, 256)); return; }
        RAD_ISSUE_N(c, op_net3, kMap * kMap, brows(b_unf_3, kMap * kMap), RAD_W(w_net3),
                    brows(b_map, kMap * kMap));
        tap(c, "st_map", b_map, kMap * kMap, kClipW);

        /* CLIP at 257 tokens: the class row, the map, the absolute table. */
        RAD_ISSUE(c, op_cls, RAD_B(b_one_c), RAD_W(w_cls), brow_slice(b_c, 0, 1, kClipW));
        RAD_ISSUE(c, op_cmap, bcol(b_map, 0, kClipW, kMap * kMap),
                  brow_slice(b_c, 1, kMap * kMap, kClipW));
        RAD_ISSUE(c, op_ids_clip, RAD_B(b_ids_clip));
        RAD_ISSUE_N(c, op_cpos, kClipTok, RAD_B(b_ids_clip), RAD_W(w_cpos),
                    brows(b_cpos, kClipTok));
        RAD_ISSUE_N(c, op_cadd, kClipTok, brows(b_c, kClipTok), brows(b_cpos, kClipTok),
                    brows(b_c, kClipTok));
        tap(c, "clip_x", b_c, kClipTok, kClipW);
        RAD_ISSUE_N(c, op_playr, kClipTok, brows(b_c, kClipTok), RAD_W(w_playr),
                    RAD_W(b_playr), brows(b_cp, kClipTok));
        RAD_ISSUE_N(c, op_ccopy, kClipTok, brows(b_cp, kClipTok), brows(b_c, kClipTok));

        for (size_t ci = 0; ci < clip.size(); ++ci) {
            const ClipLayer& l = clip[ci];
            char ctag[40];
            RAD_ISSUE_N(c, l.op_n1, kClipTok, brows(b_c, kClipTok), RAD_W(l.w_n1),
                        RAD_W(l.b_n1), brows(b_ch, kClipTok));
            std::snprintf(ctag, sizeof ctag, "clip_pre%lld", (long long)ci);
            tap(c, ctag, b_ch, kClipTok, kClipW);
            RAD_ISSUE_N(c, l.op_qkv, kClipTok, brows(b_ch, kClipTok), RAD_W(l.w_qkv),
                        RAD_W(l.b_qkv), RAD_NONE, brows(b_cq, kClipTok));
            RAD_ISSUE_N(c, l.op_attn, kClipTok,
                        bcol(b_cq, 0, kClipW, kClipTok), bcol(b_cq, kClipW, kClipW, kClipTok),
                        bcol(b_cq, 2 * kClipW, kClipW, kClipTok), RAD_B(b_cu_clip),
                        brows(b_ca, kClipTok));
            std::snprintf(ctag, sizeof ctag, "clip_att%lld", (long long)ci);
            tap(c, ctag, b_ca, kClipTok, kClipW);
            RAD_ISSUE_N(c, l.op_out, kClipTok, brows(b_ca, kClipTok), RAD_W(l.w_out),
                        RAD_W(l.b_out), RAD_NONE, brows(b_ct, kClipTok));
            RAD_ISSUE_N(c, l.op_cadd, kClipTok, brows(b_ct, kClipTok), brows(b_c, kClipTok),
                        brows(b_c, kClipTok));
            std::snprintf(ctag, sizeof ctag, "clip_h%lld", (long long)ci);
            tap(c, ctag, b_c, kClipTok, kClipW);
            RAD_ISSUE_N(c, l.op_n2, kClipTok, brows(b_c, kClipTok), RAD_W(l.w_n2),
                        RAD_W(l.b_n2), brows(b_ch, kClipTok));
            RAD_ISSUE_N(c, l.op_fc1, kClipTok, brows(b_ch, kClipTok), RAD_W(l.w_fc1),
                        RAD_W(l.b_fc1), RAD_NONE, brows(b_cf, kClipTok));
            RAD_ISSUE_N(c, l.op_scale, kClipTok, brows(b_cf, kClipTok), brows(b_c1702, 1),
                        brows(b_cg, kClipTok));
            RAD_ISSUE_N(c, l.op_qg, kClipTok, brows(b_cg, kClipTok), brows(b_cf, kClipTok),
                        brows(b_cg2, kClipTok));
            RAD_ISSUE_N(c, l.op_fc2, kClipTok, brows(b_cg2, kClipTok), RAD_W(l.w_fc2),
                        RAD_W(l.b_fc2), RAD_NONE, brows(b_ct, kClipTok));
            RAD_ISSUE_N(c, l.op_madd, kClipTok, brows(b_ct, kClipTok), brows(b_c, kClipTok),
                        brows(b_c, kClipTok));
            std::snprintf(ctag, sizeof ctag, "clip_b%lld", (long long)ci);
            tap(c, ctag, b_c, kClipTok, kClipW);
        }

        tap(c, "clip_out", b_c, kMap * kMap, kClipW, 1);

        /* Fuse (CLIP's patch rows | the map) and project. */
        RAD_ISSUE(c, op_fuse_clip, brow_slice(b_c, 1, kMap * kMap, kClipW),
                  bcol(b_fuse, 0, kClipW, kMap * kMap));
        RAD_ISSUE(c, op_fuse_sam, bcol(b_map, 0, kClipW, kMap * kMap),
                  bcol(b_fuse, kClipW, kClipW, kMap * kMap));
        tap(c, "fuse", b_fuse, kMap * kMap, kFused);
        RAD_ISSUE_N(c, op_proj, kMap * kMap, brows(b_fuse, kMap * kMap), RAD_W(w_proj),
                    RAD_W(b_proj), RAD_NONE, brows(b_projbuf, kMap * kMap));
        tap(c, "proj", b_projbuf, kMap * kMap, kNEmbd);

        if (dbg) {
            const RadOperand src = dbg == 1 ? bcol(b_map, 0, kClipW, kMap * kMap)
                                 : dbg == 2 ? brow_slice(b_c, 1, kMap * kMap, kClipW)
                                            : brows(b_projbuf, kMap * kMap);
            const RadOperand dst = dbg == 4 ? bcol(b_out, 0, kNEmbd, kMap * kMap)
                                            : bcol(b_out, 0, kClipW, kMap * kMap);
            RAD_ISSUE(c, dbg == 4 ? op_dbg1280 : op_dbg1024, src, dst);
            return;
        }

        /* The layout: every map line of 16 rows is closed by `image_newline`, and the view by a
         * `view_seperator` row. */
        for (int64_t m = 0; m < kMap; ++m) {
            const int64_t dst = rowbase + m * (kMap + 1);
            RAD_ISSUE_N(c, op_line, kMap, brow_slice(b_projbuf, m * kMap, kMap, kNEmbd),
                        brow_slice(b_out, dst, kMap, kNEmbd));
            RAD_ISSUE(c, op_newline, RAD_B(b_one), RAD_W(w_newline),
                      brow_slice(b_out, dst + kMap, 1, kNEmbd));
        }
        RAD_ISSUE(c, op_sep, RAD_B(b_one), RAD_W(w_sep),
                  brow_slice(b_out, rowbase + kRowsView - 1, 1, kNEmbd));
    }

    void step(RadCtx* c, const RadBatch* batch) {
        if (!on) {
            rad_step_fail(c, "unlimitedocr: an image pass reached the DeepEncoder, which this "
                             "program did not declare");
            return;
        }
        const int64_t n_patch = batch->enc_n_patch;
        const int64_t n_seg   = batch->enc_n_seg;
        const int64_t views   = n_patch / kSegView;
        if (n_patch <= 0 || views <= 0 || views > max_views || n_seg != 2 * views) {
            rad_step_fail(c, "unlimitedocr: an encoder pass the DeepEncoder does not serve: one "
                             "1024x1024 view is 4352 staged patches + a separator segment, and "
                             "only that geometry is declared (crops need a resampled position "
                             "table, tiling is gundam mode)");
            return;
        }

        /* One view's constants for the whole pass. */
        RAD_ISSUE(c, op_fill_one, RAD_B(b_one));
        RAD_ISSUE(c, op_fill_one_c, RAD_B(b_one_c));
        RAD_ISSUE(c, op_fill_1702, RAD_B(b_c1702));
        RAD_ISSUE(c, op_cu_sam, RAD_B(b_cu_sam));
        RAD_ISSUE(c, op_cu_clip, RAD_B(b_cu_clip));
        RAD_ISSUE(c, op_cu_pad, RAD_B(b_cu_pad));

        /* window_partition's two index tables: padded position (r*kGridP + c) holds real row
         * r*kGrid + c, every other padded row is the pad (gather -1 = a zero row, which the qkv
         * linear turns into the qkv bias). Constants, written before the pass's ops. */
        static const std::vector<int32_t> pad_idx = [] {
            std::vector<int32_t> v((size_t)kSamTokP, -1);
            for (int64_t r = 0; r < kGrid; ++r)
                for (int64_t c = 0; c < kGrid; ++c)
                    v[(size_t)(r * kGridP + c)] = (int32_t)(r * kGrid + c);
            return v;
        }();
        static const std::vector<int32_t> real_idx = [] {
            std::vector<int32_t> v((size_t)kSamTok);
            for (int64_t r = 0; r < kGrid; ++r)
                for (int64_t c = 0; c < kGrid; ++c)
                    v[(size_t)(r * kGrid + c)] = (int32_t)(r * kGridP + c);
            return v;
        }();
        void* p_pad  = rad_buf_ptr(c, b_ids_pad);
        void* p_real = rad_buf_ptr(c, b_ids_real);
        if (!p_pad || !p_real) {
            rad_step_fail(c, "unlimitedocr: the padded-grid index table of the DeepEncoder did not "
                             "bind");
            return;
        }
        if (rad_memcpy_async(p_pad, pad_idx.data(),
                             (int64_t)(pad_idx.size() * sizeof(int32_t)),
                             rad_stream(c)) < 0 ||
            rad_memcpy_async(p_real, real_idx.data(),
                             (int64_t)(real_idx.size() * sizeof(int32_t)),
                             rad_stream(c)) < 0) {
            rad_step_fail(c, "unlimitedocr: could not stage the DeepEncoder's padded-grid index "
                             "table");
            return;
        }

        for (int64_t v = 0; v < views; ++v) view(c, batch, v);
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif  /* RAD_BLOCK_SAM_H */
