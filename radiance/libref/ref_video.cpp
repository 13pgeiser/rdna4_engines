/* ref_video.cpp -- the video generator's ops around its decoder (docs/OPS.md, "Video
 * generation"): the denoised latent's patch rows unfolded to voxels, the tiles a video decoder
 * runs over, gathered from the latent and stitched back into frames, and the rotary positions of
 * a tile's tokens.
 *
 * Per-request geometry (a video's frames, height and width) comes from a `geom` operand the run
 * phase writes on the device, never from the parameters, which are fixed at declare. The tiles
 * are diffusers' AutoencoderKLMiniMaxH3 tiling in latent units, and the stitch rounds where its
 * bf16 blends round.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <vector>

using namespace ref;

namespace {

inline int64_t ldi(const RadTensor* t, int64_t i) {
    return ld_int(t->dtype, t->data, offlin(t, i));
}

/* A space-separated list of floats, kept in f32: a video's latent and pixel statistics are f32
 * tensors upstream, unlike the bf16 lists chan_affine reads. */
int parse_f32_list(const char* s, std::vector<float>* out) {
    out->clear();
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        out->push_back((float)v);
        s = end;
    }
    return (int)out->size();
}

/* The tiles over one axis of `len` latent units, diffusers' _split_tiles in latent units: a tile
 * at least as long as the axis is the whole axis (and must be exactly as long, every tile being
 * tile x tile); otherwise the fewest tiles whose union covers the axis with every overlap at
 * least `overlap`, the slack added to the overlaps one unit at a time round-robin from the first,
 * and start[i + 1] = start[i] + tile - ov[i]. False where no such layout exists. */
bool split_tiles(int64_t len, int64_t tile, int64_t overlap, std::vector<int64_t>* start,
                 std::vector<int64_t>* ov) {
    start->clear();
    ov->clear();
    if (len <= 0 || tile <= 0 || overlap < 0 || overlap >= tile) return false;
    if (tile >= len) {
        start->push_back(0);
        return len == tile;
    }
    int64_t nt = (len + tile - 1) / tile;
    while (tile * nt - overlap * (nt - 1) - len < 0) ++nt;
    ov->assign((size_t)(nt - 1), overlap);
    const int64_t rem = tile * nt - overlap * (nt - 1) - len;
    for (int64_t i = 0; i < rem; ++i) (*ov)[(size_t)(i % (nt - 1))] += 1;
    start->push_back(0);
    for (int64_t i = 0; i + 1 < nt; ++i) start->push_back(start->back() + tile - (*ov)[(size_t)i]);
    return true;
}

}  /* namespace */

/* ------------------------------------------------------------------ patch_unfold */
/* A video latent's patch rows unfolded to voxel rows and denormalised. x [F Hp Wp, C pt ph pw]
 * holds patch (f, i, j), frame-major then row-major, its column (c, a, b, d) at
 * c pt ph pw + a ph pw + b pw + d; y [(F pt) (Hp ph) (Wp pw), C] holds voxel (t, yy, xx)
 * row-major:
 *
 *   y[(t, yy, xx), c] = f32(f32(x[(t / pt, yy / ph, xx / pw), (c, t % pt, yy % ph, xx % pw)]
 *                               * mul[c]) + add[c])
 *
 * `geom` is (F, Hp, Wp), the request's latent grid in patches. `mul` and `add` are C values each,
 * kept in f32 (the reference denormalises f32 latents with an f32 deviation and mean). y holds at
 * least the voxels' rows; rows past them are not written. */
int ref_patch_unfold(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    std::vector<float> mul, add;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32_list(p_str(a, "mul", ""), &mul) != C ||
        parse_f32_list(p_str(a, "add", ""), &add) != C || numel(geom) < 3)
        return RAD_E_SHAPE;
    const int64_t F = ldi(geom, 0), Hp = ldi(geom, 1), Wp = ldi(geom, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t Hv = Hp * ph, Wv = Wp * pw, K = C * pt * ph * pw, rows = F * pt * Hv * Wv;
    if (rows_of(y, C) < rows || rows_of(x, K) < F * Hp * Wp) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * C)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t t = r / (Hv * Wv), yy = (r / Wv) % Hv, xx = r % Wv;
        const int64_t patch = ((t / pt) * Hp + yy / ph) * Wp + xx / pw;
        const int64_t sub = ((t % pt) * ph + yy % ph) * pw + xx % pw;
        const int64_t xb = rowoff(x, patch, K), xs = laststride(x);
        const int64_t yb = rowoff(y, r, C), ys = laststride(y);
        for (int64_t c = 0; c < C; ++c)
            stt(y, yb + c * ys,
                ldt(x, xb + (c * pt * ph * pw + sub) * xs) * mul[(size_t)c] + add[(size_t)c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_pos3 */
/* A video decoder's rotary positions over consecutive tiles: row m is token k = m % period of its
 * tile, the tile's T x H x W tokens first ((t, i, j) row-major) and the period's remaining rows
 * (register tokens) at position 0. Each axis is the token's centre normalised to [-1, 1) and
 * scaled, in f32 as torch's 2 * (arange(0.5, n) / n) - 1 makes it:
 *
 *   g(c, n) = f32(f32(2 f32((c + 0.5) / n)) - 1)     y[axis][m] = f32(f32(scale) g)
 *
 * y is f32 [3, >= M], component-major, each component a row at the operand's own pitch (a [3, M]
 * view of a wider buffer); M is its row length. */
int ref_grid_pos3(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t T = p_int(a, "T", 0), H = p_int(a, "H", 0), W = p_int(a, "W", 0);
    const int64_t period = p_int(a, "period", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (T <= 0 || H <= 0 || W <= 0 || period < T * H * W || y->rank != 2 || y->shape[0] < 3)
        return RAD_E_SHAPE;
    const int64_t M = y->shape[1], rs = y->stride[0], cs = y->stride[1];
    if (M <= 0) return RAD_E_SHAPE;
    auto g = [](int64_t c, int64_t n) {
        const float q = ((float)c + 0.5f) / (float)n;
        return 2.0f * q - 1.0f;
    };
    REF_PAR_FOR(M * 3)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t k = m % period;
        float p[3] = { 0.0f, 0.0f, 0.0f };
        if (k < T * H * W) {
            p[0] = scale * g(k / (H * W), T);
            p[1] = scale * g((k / W) % H, H);
            p[2] = scale * g(k % W, W);
        }
        for (int c = 0; c < 3; ++c) stt(y, c * rs + m * cs, p[c]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_gather */
/* The tiles a video decoder runs over, gathered from voxel rows x [frames H W, n] ((t, yy, xx)
 * row-major), `geom` = (frames, H, W) in latent units, its frames those of the `frames`
 * parameter. Each spatial axis is split by split_tiles (`tile`, `overlap`) and the tiles are taken
 * row-major over (row band, column band). Tile k is rows [k period, (k + 1) period) of y: its
 * frames tile tile voxels (t, ty, tx) row-major, from x's row (t, start_y + ty, start_x + tx);
 * then reg's first `n_reg` rows (a decoder's register tokens) in y's dtype; then zero rows to the
 * period. y holds at least n_tiles period rows. */
int ref_vtile_gather(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *reg = &a->t[1], *geom = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), frames = p_int(a, "frames", 0);
    const int64_t period = p_int(a, "period", 0), n_reg = p_int(a, "n_reg", 0);
    const int64_t vox = frames * tile * tile;
    if (n <= 0 || tile <= 0 || frames <= 0 || n_reg < 0 || period < vox + n_reg ||
        numel(geom) < 3 || rows_of(reg, n) < n_reg)
        return RAD_E_SHAPE;
    const int64_t H = ldi(geom, 1), W = ldi(geom, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (ldi(geom, 0) != frames || !split_tiles(H, tile, overlap, &sy, &oy) ||
        !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nw = (int64_t)sx.size(), rows = (int64_t)sy.size() * nw * period;
    if (rows_of(y, n) < rows || rows_of(x, n) < frames * H * W) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * n)
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t k = row / period, p = row % period;
        const int64_t yb = rowoff(y, row, n), ys = laststride(y);
        if (p < vox) {
            const int64_t t = p / (tile * tile), ty = (p / tile) % tile, tx = p % tile;
            const int64_t src = (t * H + sy[(size_t)(k / nw)] + ty) * W + sx[(size_t)(k % nw)] + tx;
            const int64_t xb = rowoff(x, src, n), xs = laststride(x);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(x, xb + c * xs));
        } else if (p < vox + n_reg) {
            const int64_t rb = rowoff(reg, p - vox, n), rs = laststride(reg);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(reg, rb + c * rs));
        } else {
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, 0.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_stitch */
/* A decoded chunk's tiles stitched into frames, blended with the previous chunk's overlap, and
 * written as RGB bytes.
 *
 * x is the decoder's output over vtile_gather's tiles, [>= n_tiles period, 3 pt scale scale]:
 * tile k's row k period + (lt tile + ty) tile + tx, column
 * c pt scale^2 + a scale^2 + b scale + d, is colour c of pixel (ty scale + b, tx scale + d) of clip
 * frame lt pt + a. `geom` = (frames, H, W, first, last): the chunk's latent geometry, and whether
 * it is its video's first and last chunk (nonzero). Tiles are split_tiles' over H and W.
 *
 * Every product and sum below rounds to x's dtype (rd), as the reference's bf16 blends do, with
 * wa(p, e) = rd(1 - rd(p / e)) and wb(p, e) = rd(p / e). Pixel (Y, X) of clip frame fc belongs to
 * the tile (i, j) whose kept range holds it, [start_i scale, start_{i+1} scale) (the last tile to
 * its end), at (ly, lx) within it; T(i, j, ly, lx) is that tile's decoded value. The spatial
 * stitch blends the top ev = ov_y[i-1] scale rows with the tile above and then the left
 * eh = ov_x[j-1] scale columns with the tile to the left, both neighbours as decoded (never
 * blended):
 *
 *   V = rd(rd(T(i-1, j, ly + tile scale - ev, lx) wa(ly, ev)) + rd(T(i, j, ly, lx) wb(ly, ev)))
 *       when i > 0 and ly < ev, else T(i, j, ly, lx)
 *   S = rd(rd(T(i, j-1, ly, lx + tile scale - eh) wa(lx, eh)) + rd(V wb(lx, eh)))
 *       when j > 0 and lx < eh, else V
 *
 * Output frame k < keep is clip frame skip + k, its first `blend` frames blended with `prev`
 * unless the chunk is its video's first, v = rd(rd(prev[k] wa(k, blend)) + rd(S wb(k, blend)));
 * prev then becomes the frames S(2 skip + keep + k'), k' < blend, at prev's dtype; with `last`
 * those frames follow the kept ones in y. A byte is round-half-even(255 clamp(f32(f32(v std[c]) +
 * mean[c]), 0, 1)), `mean` and `std` three f32 values each. y is u8 [frames_out, H scale,
 * W scale, 3] and prev [blend, H scale, W scale, 3]; y holds at least the frames' rows. */
int ref_vtile_stitch(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *geom = &a->t[1], *prev = &a->t[2], *y = &a->t[3];
    const int64_t tile = p_int(a, "tile", 0), overlap = p_int(a, "overlap", 0);
    const int64_t S = p_int(a, "scale", 0), pt = p_int(a, "pt", 0), skip = p_int(a, "skip", 0);
    const int64_t keep = p_int(a, "keep", 0), blend = p_int(a, "blend", 0);
    const int64_t period = p_int(a, "period", 0);
    std::vector<float> mean, stdv;
    if (tile <= 0 || S <= 0 || pt <= 0 || skip < 0 || keep <= 0 || blend < 0 || blend > keep ||
        parse_f32_list(p_str(a, "mean", ""), &mean) != 3 ||
        parse_f32_list(p_str(a, "std", ""), &stdv) != 3 || numel(geom) < 5 || y->dtype != RAD_U8)
        return RAD_E_SHAPE;
    const int64_t frames = ldi(geom, 0), H = ldi(geom, 1), W = ldi(geom, 2);
    const bool first = ldi(geom, 3) != 0, last = ldi(geom, 4) != 0;
    std::vector<int64_t> sy, sx, oy, ox;
    if (frames <= 0 || 2 * skip + keep + blend > frames * pt || period < frames * tile * tile ||
        !split_tiles(H, tile, overlap, &sy, &oy) || !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size(), cols = 3 * pt * S * S;
    if (rows_of(x, cols) < nh * nw * period) return RAD_E_SHAPE;
    const int64_t Hp = H * S, Wp = W * S, npx = Hp * Wp;
    const int64_t fout = keep + (last ? blend : 0);
    if (rows_of(y, 3) < fout * npx || (blend > 0 && rows_of(prev, 3) < blend * npx))
        return RAD_E_SHAPE;
    /* The owner tile and the offset in it of every pixel row and column. */
    std::vector<int64_t> own_y((size_t)Hp), own_x((size_t)Wp);
    for (int64_t Y = 0, i = 0; Y < Hp; ++Y) {
        while (i + 1 < nh && Y >= sy[(size_t)(i + 1)] * S) ++i;
        own_y[(size_t)Y] = i;
    }
    for (int64_t X = 0, j = 0; X < Wp; ++X) {
        while (j + 1 < nw && X >= sx[(size_t)(j + 1)] * S) ++j;
        own_x[(size_t)X] = j;
    }
    const uint32_t dt = x->dtype;
    auto rd = [dt](float v) { return round_dt(dt, v); };
    auto wb = [&](int64_t p, int64_t e) { return rd((float)p / (float)e); };
    auto wa = [&](int64_t p, int64_t e) { return rd(1.0f - rd((float)p / (float)e)); };
    auto T = [&](int64_t i, int64_t j, int64_t ly, int64_t lx, int64_t fc, int64_t c) {
        const int64_t row = (i * nw + j) * period + ((fc / pt) * tile + ly / S) * tile + lx / S;
        const int64_t col = ((c * pt + fc % pt) * S + ly % S) * S + lx % S;
        return ldt(x, rowoff(x, row, cols) + col * laststride(x));
    };
    auto stitch = [&](int64_t fc, int64_t Y, int64_t X, int64_t c) {
        const int64_t i = own_y[(size_t)Y], j = own_x[(size_t)X];
        const int64_t ly = Y - sy[(size_t)i] * S, lx = X - sx[(size_t)j] * S;
        float v = T(i, j, ly, lx, fc, c);
        const int64_t ev = i > 0 ? oy[(size_t)(i - 1)] * S : 0;
        if (ly < ev) {
            const float up = T(i - 1, j, ly + tile * S - ev, lx, fc, c);
            v = rd(rd(up * wa(ly, ev)) + rd(v * wb(ly, ev)));
        }
        const int64_t eh = j > 0 ? ox[(size_t)(j - 1)] * S : 0;
        if (lx < eh) {
            const float left = T(i, j - 1, ly, lx + tile * S - eh, fc, c);
            v = rd(rd(left * wa(lx, eh)) + rd(v * wb(lx, eh)));
        }
        return v;
    };
    auto byte = [&](int64_t f, int64_t px, int64_t c, float v) {
        float p = v * stdv[(size_t)c] + mean[(size_t)c];
        p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        stt(y, rowoff(y, f * npx + px, 3) + c * laststride(y), p * 255.0f);
    };
    REF_PAR_FOR(npx * 3 * (fout + blend) * 4)
    for (int64_t px = 0; px < npx; ++px) {
        const int64_t Y = px / Wp, X = px % Wp;
        for (int64_t c = 0; c < 3; ++c) {
            for (int64_t k = 0; k < keep; ++k) {
                float v = stitch(skip + k, Y, X, c);
                if (!first && k < blend) {
                    const int64_t pb = rowoff(prev, k * npx + px, 3);
                    const float pv = ldt(prev, pb + c * laststride(prev));
                    v = rd(rd(pv * wa(k, blend)) + rd(v * wb(k, blend)));
                }
                byte(k, px, c, v);
            }
            for (int64_t k = 0; k < blend; ++k) {
                const float v = stitch(2 * skip + keep + k, Y, X, c);
                if (last) byte(keep + k, px, c, v);
                stt(prev, rowoff(prev, k * npx + px, 3) + c * laststride(prev), v);
            }
        }
    }
    return RAD_OK;
}
