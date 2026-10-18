/* avx_video.cpp -- the video generator's ops around its decoder (docs/OPS.md, "Video
 * generation"): the latent's patch rows unfolded to voxels, the tiles a video decoder runs over,
 * gathered from the latent and stitched back into frames, and the rotary positions of a tile's
 * tokens.
 *
 * Bit-exact with libref (libref/ref_video.cpp): they are copies and roundings, and a seed must
 * give the same video bytes on both host libraries. Where libref multiplies and then adds in two
 * roundings, `keep` stops the compiler from fusing the pair (this plugin is built with
 * -ffp-contract=fast, libref without contraction); the stitch's bf16 blends are passes of whole
 * rows through x's dtype (`narrow_row`), so a product is rounded before the sum reads it.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

inline int64_t at(const RadTensor* t, int64_t i) { return ldi(t, offlin(t, i)); }

/* v[0, n) rounded through dtype `dt` in place, as a store and reload rounds it. `tmp` holds n
 * floats. */
inline void narrow_row(uint32_t dt, float* v, int64_t n, float* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* A value the compiler may not fuse into the next add. */
#if AVX_LEVEL == 0
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#elif AVX_LEVEL == 3
inline float keep(float x) { __asm__("" : "+v"(x)); return x; }
#else
inline float keep(float x) { __asm__("" : "+x"(x)); return x; }
#endif

/* The f32 values of a space-separated list (libref's parse_f32_list): up to `cap` into out, the
 * count returned whatever it is. */
int64_t parse_f32(const char* s, float* out, int64_t cap) {
    int64_t k = 0;
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        if (k < cap) out[k] = (float)v;
        ++k;
        s = end;
    }
    return k;
}

/* libref's split_tiles: the tile starts and overlaps over `len` latent units. */
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

/* A row copied between operands, converted through f32 as libref's element loop is. Whole-byte
 * float rows of one dtype are copied as bytes, which is the same thing for every non-NaN value. */
inline void copy_row(const RadTensor* src, int64_t sb, const RadTensor* dst, int64_t db, int64_t n,
                     float* scr) {
    const uint32_t dt = src->dtype;
    const bool flt = dt == RAD_F32 || dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F8E4M3 ||
                     dt == RAD_F8E5M2;
    if (flt && dt == dst->dtype && laststride(src) == 1 && laststride(dst) == 1) {
        const size_t w = (size_t)(rad_dtype_bits(dt) / 8);
        std::memcpy((char*)dst->data + (size_t)db * w, (const char*)src->data + (size_t)sb * w,
                    (size_t)n * w);
        return;
    }
    out_row(dst, db, n, in_row(src, sb, n, scr));
}

/* out[e] = rd(rd(a[e] wa) + rd(b[e] wb)) over e < cnt, pixels of three values with a weight pair
 * each (`ws` 1) or one pair for every pixel (`ws` 0), rd a pass through `dt`. `out` may be `b`;
 * `p2` and `tmp` hold cnt floats. */
void blend_rgb(const float* a, const float* wa, const float* b, const float* wb, int64_t ws,
               int64_t cnt, float* out, float* p2, float* tmp, uint32_t dt) {
    for (int64_t e = 0; e < cnt; ++e) {
        p2[e] = b[e] * wb[(e / 3) * ws];
        out[e] = a[e] * wa[(e / 3) * ws];
    }
    narrow_row(dt, out, cnt, tmp);
    narrow_row(dt, p2, cnt, tmp);
    for (int64_t e = 0; e < cnt; ++e) out[e] += p2[e];
    narrow_row(dt, out, cnt, tmp);
}

}  /* namespace */

/* ------------------------------------------------------------------ patch_unfold */
/* Patch rows to denormalised voxel rows, f32(f32(x * mul[c]) + add[c]) (libref): a task is a
 * patch, its row read once and dealt into its pt ph pw voxel rows. */
AVX_KERNEL(patch_unfold) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *Y = &a->t[2];
    const int64_t C = p_int(a, "C", 0), pt = p_int(a, "pt", 0);
    const int64_t ph = p_int(a, "ph", 0), pw = p_int(a, "pw", 0);
    const char* ms = p_str(a, "mul", ""), *as = p_str(a, "add", "");
    /* Each number takes at least two characters but the last, so this bounds the count. */
    const int64_t cap = (int64_t)((std::strlen(ms) + std::strlen(as)) / 2 + 2);
    float* lists = (float*)scratch_raw((size_t)(2 * cap + 16) * sizeof(float));
    float *mul = lists, *add = lists + cap;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || parse_f32(ms, mul, cap) != C ||
        parse_f32(as, add, cap) != C || numel(G) < 3)
        return RAD_E_SHAPE;
    const int64_t F = at(G, 0), Hp = at(G, 1), Wp = at(G, 2);
    if (F <= 0 || Hp <= 0 || Wp <= 0) return RAD_E_SHAPE;
    const int64_t vol = pt * ph * pw, Hv = Hp * ph, Wv = Wp * pw, K = C * vol;
    if (rows_of(Y, C) < F * pt * Hv * Wv || rows_of(X, K) < F * Hp * Wp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(F * Hp * Wp * K >= AVX_PAR_MIN)
    for (int64_t pi = 0; pi < F * Hp * Wp; ++pi) {
        float* buf = scratch_f32((size_t)(K + C + 32));
        float *xs = buf, *row = buf + K + 16;
        const float* xr = in_row(X, rowoff(X, pi, K), K, xs);
        const int64_t f = pi / (Hp * Wp), i = (pi / Wp) % Hp, j = pi % Wp;
        for (int64_t sub = 0; sub < vol; ++sub) {
            const int64_t ta = sub / (ph * pw), yb = (sub / pw) % ph, xd = sub % pw;
            for (int64_t c = 0; c < C; ++c) row[c] = keep(xr[c * vol + sub] * mul[c]) + add[c];
            const int64_t r = ((f * pt + ta) * Hv + i * ph + yb) * Wv + j * pw + xd;
            out_row(Y, rowoff(Y, r, C), C, row);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_pos3 */
/* A tile's rotary positions, f32(scale) g(c, n), g(c, n) = f32(f32(2 f32((c + 0.5) / n)) - 1)
 * (libref): one value an axis index, taken once a launch. */
AVX_KERNEL(grid_pos3) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t T = p_int(a, "T", 0), H = p_int(a, "H", 0), W = p_int(a, "W", 0);
    const int64_t period = p_int(a, "period", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (T <= 0 || H <= 0 || W <= 0 || period < T * H * W || Y->rank != 2 || Y->shape[0] < 3)
        return RAD_E_SHAPE;
    const int64_t M = Y->shape[1], rs = Y->stride[0], cs = Y->stride[1];
    if (M <= 0) return RAD_E_SHAPE;
    float* g = (float*)scratch_raw((size_t)(T + H + W + 16) * sizeof(float));
    auto axis = [scale](int64_t c, int64_t n) {
        const float q = ((float)c + 0.5f) / (float)n;
        return scale * (keep(2.0f * q) - 1.0f);
    };
    for (int64_t c = 0; c < T; ++c) g[c] = axis(c, T);
    for (int64_t c = 0; c < H; ++c) g[T + c] = axis(c, H);
    for (int64_t c = 0; c < W; ++c) g[T + H + c] = axis(c, W);
    AVX_PARALLEL_FOR_IF(M * 3 >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t k = m % period;
        const bool tok = k < T * H * W;
        stt(Y, m * cs, tok ? g[k / (H * W)] : 0.0f);
        stt(Y, rs + m * cs, tok ? g[T + (k / W) % H] : 0.0f);
        stt(Y, 2 * rs + m * cs, tok ? g[T + H + k % W] : 0.0f);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_gather */
/* A video decoder's tiles out of the latent's voxel rows, then the register rows and zeros
 * (libref): one copied row a y row. */
AVX_KERNEL(vtile_gather) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *R = &a->t[1], *G = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), tile = p_int(a, "tile", 0);
    const int64_t overlap = p_int(a, "overlap", 0), frames = p_int(a, "frames", 0);
    const int64_t period = p_int(a, "period", 0), n_reg = p_int(a, "n_reg", 0);
    const int64_t vox = frames * tile * tile;
    if (n <= 0 || tile <= 0 || frames <= 0 || n_reg < 0 || period < vox + n_reg ||
        numel(G) < 3 || rows_of(R, n) < n_reg)
        return RAD_E_SHAPE;
    const int64_t H = at(G, 1), W = at(G, 2);
    std::vector<int64_t> sy, sx, oy, ox;
    if (at(G, 0) != frames || !split_tiles(H, tile, overlap, &sy, &oy) ||
        !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nw = (int64_t)sx.size(), rows = (int64_t)sy.size() * nw * period;
    if (rows_of(Y, n) < rows || rows_of(X, n) < frames * H * W) return RAD_E_SHAPE;
    const int64_t* py = sy.data();
    const int64_t* px = sx.data();

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t row = 0; row < rows; ++row) {
        float* scr = scratch_f32((size_t)n + 16);
        const int64_t k = row / period, p = row % period, yb = rowoff(Y, row, n);
        if (p < vox) {
            const int64_t t = p / (tile * tile), ty = (p / tile) % tile, tx = p % tile;
            const int64_t src = (t * H + py[k / nw] + ty) * W + px[k % nw] + tx;
            copy_row(X, rowoff(X, src, n), Y, yb, n, scr);
        } else if (p < vox + n_reg) {
            copy_row(R, rowoff(R, p - vox, n), Y, yb, n, scr);
        } else {
            for (int64_t c = 0; c < n; ++c) scr[c] = 0.0f;
            out_row(Y, yb, n, scr);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ vtile_stitch */
/* A decoded chunk's tiles stitched into frames, the chunk overlap blended and kept, RGB bytes out
 * (libref's rule and roundings). A task is one pixel row of one frame: each tile column's line of
 * the owner tile is read out of x, blended with the tile above's line and then the left tile's
 * (both as decoded), a whole line at a time, every product and sum a pass through x's dtype. The
 * kept frames are made first, reading `prev`; the frames that refill it after, in a second pass,
 * so no task reads a row another has rewritten. */
AVX_KERNEL(vtile_stitch) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *G = &a->t[1], *PV = &a->t[2], *Y = &a->t[3];
    const int64_t tile = p_int(a, "tile", 0), overlap = p_int(a, "overlap", 0);
    const int64_t S = p_int(a, "scale", 0), pt = p_int(a, "pt", 0), skip = p_int(a, "skip", 0);
    const int64_t keep_n = p_int(a, "keep", 0), blend = p_int(a, "blend", 0);
    const int64_t period = p_int(a, "period", 0);
    float mean[3], stdv[3];
    if (tile <= 0 || S <= 0 || pt <= 0 || skip < 0 || keep_n <= 0 || blend < 0 ||
        blend > keep_n || parse_f32(p_str(a, "mean", ""), mean, 3) != 3 ||
        parse_f32(p_str(a, "std", ""), stdv, 3) != 3 || numel(G) < 5 || Y->dtype != RAD_U8)
        return RAD_E_SHAPE;
    const int64_t frames = at(G, 0), H = at(G, 1), W = at(G, 2);
    const bool first = at(G, 3) != 0, last = at(G, 4) != 0;
    std::vector<int64_t> sy, sx, oy, ox;
    if (frames <= 0 || 2 * skip + keep_n + blend > frames * pt || period < frames * tile * tile ||
        !split_tiles(H, tile, overlap, &sy, &oy) || !split_tiles(W, tile, overlap, &sx, &ox))
        return RAD_E_SHAPE;
    const int64_t nh = (int64_t)sy.size(), nw = (int64_t)sx.size(), cols = 3 * pt * S * S;
    if (rows_of(X, cols) < nh * nw * period) return RAD_E_SHAPE;
    const int64_t Hp = H * S, Wp = W * S, npx = Hp * Wp;
    const int64_t fout = keep_n + (last ? blend : 0);
    if (rows_of(Y, 3) < fout * npx || (blend > 0 && rows_of(PV, 3) < blend * npx))
        return RAD_E_SHAPE;
    std::vector<int64_t> own_y((size_t)Hp);
    for (int64_t Yp = 0, i = 0; Yp < Hp; ++Yp) {
        while (i + 1 < nh && Yp >= sy[(size_t)(i + 1)] * S) ++i;
        own_y[(size_t)Yp] = i;
    }
    const uint32_t dt = X->dtype;
    const int64_t TW = tile * S, RW = Wp * 3, TW3 = TW * 3;
    const int64_t *psy = sy.data(), *psx = sx.data(), *poy = oy.data(), *pox = ox.data();
    const int64_t* pown = own_y.data();
    auto wb = [dt](int64_t p, int64_t e) { return narrow1(dt, (float)p / (float)e); };
    auto wa = [dt](int64_t p, int64_t e) {
        return narrow1(dt, 1.0f - narrow1(dt, (float)p / (float)e));
    };

    /* Tile (i, j)'s decoded line at row ly of clip frame fc: out[lx * 3 + c], lx < TW. */
    auto tile_line = [&](int64_t i, int64_t j, int64_t ly, int64_t fc, float* out) {
        const int64_t base = (i * nw + j) * period + ((fc / pt) * tile + ly / S) * tile;
        const int64_t xs = laststride(X);
        for (int64_t tx = 0; tx < tile; ++tx) {
            const int64_t xb = rowoff(X, base + tx, cols);
            for (int64_t c = 0; c < 3; ++c) {
                const int64_t col0 = ((c * pt + fc % pt) * S + ly % S) * S;
                for (int64_t d = 0; d < S; ++d)
                    out[(tx * S + d) * 3 + c] = ldt(X, xb + (col0 + d) * xs);
            }
        }
    };
    /* The stitched pixel row Yp of clip frame fc into row[0, RW). `l0`, `l1` hold TW3 floats,
     * `w` 2 TW, `p2` and `tmp` RW. */
    auto stitch_row = [&](int64_t fc, int64_t Yp, float* row, float* l0, float* l1, float* w,
                          float* p2, float* tmp) {
        const int64_t i = pown[Yp], ly = Yp - psy[i] * S;
        const int64_t ev = i > 0 ? poy[i - 1] * S : 0;
        const float wv[2] = { ly < ev ? wa(ly, ev) : 0.0f, ly < ev ? wb(ly, ev) : 0.0f };
        for (int64_t j = 0; j < nw; ++j) {
            const int64_t x0 = psx[j] * S, x1 = j + 1 < nw ? psx[j + 1] * S : Wp;
            tile_line(i, j, ly, fc, l0);
            if (ly < ev) {
                tile_line(i - 1, j, ly + TW - ev, fc, l1);
                blend_rgb(l1, &wv[0], l0, &wv[1], 0, TW3, l0, p2, tmp, dt);
            }
            const int64_t eh = j > 0 ? std::min(pox[j - 1] * S, x1 - x0) : 0;
            if (eh > 0) {
                const int64_t full = pox[j - 1] * S;
                tile_line(i, j - 1, ly, fc, l1);
                for (int64_t lx = 0; lx < eh; ++lx) {
                    w[lx] = wa(lx, full);
                    w[TW + lx] = wb(lx, full);
                }
                blend_rgb(l1 + (TW - full) * 3, w, l0, w + TW, 1, eh * 3, l0, p2, tmp, dt);
            }
            std::memcpy(row + x0 * 3, l0, (size_t)((x1 - x0) * 3) * sizeof(float));
        }
    };
    /* Frame f's pixel row Yp as bytes. */
    auto bytes = [&](int64_t f, int64_t Yp, const float* row) {
        for (int64_t X0 = 0; X0 < Wp; ++X0) {
            const int64_t yb = rowoff(Y, f * npx + Yp * Wp + X0, 3);
            for (int64_t c = 0; c < 3; ++c) {
                float p = keep(row[X0 * 3 + c] * stdv[c]) + mean[c];
                p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
                stt(Y, yb + c * laststride(Y), p * 255.0f);
            }
        }
    };
    const size_t need = (size_t)(3 * RW + 2 * TW3 + 2 * TW + 96);

    AVX_PARALLEL_FOR_IF(keep_n * npx * 3 * 8 >= AVX_PAR_MIN)
    for (int64_t task = 0; task < keep_n * Hp; ++task) {
        const int64_t k = task / Hp, Yp = task % Hp;
        float* buf = scratch_f32(need);
        float *row = buf, *p2 = row + RW + 16, *tmp = p2 + RW + 16, *l0 = tmp + RW + 16;
        float *l1 = l0 + TW3 + 16, *w = l1 + TW3 + 16;
        stitch_row(skip + k, Yp, row, l0, l1, w, p2, tmp);
        if (!first && k < blend) {
            float* pr = l0;   /* the overlap row, a line's worth at a time */
            const float pw[2] = { wa(k, blend), wb(k, blend) };
            for (int64_t x0 = 0; x0 < Wp; x0 += TW) {
                const int64_t cnt = std::min(TW, Wp - x0);
                for (int64_t e = 0; e < cnt * 3; ++e) {
                    const int64_t pxi = k * npx + Yp * Wp + x0 + e / 3;
                    pr[e] = ldt(PV, rowoff(PV, pxi, 3) + (e % 3) * laststride(PV));
                }
                blend_rgb(pr, &pw[0], row + x0 * 3, &pw[1], 0, cnt * 3, row + x0 * 3, p2, tmp, dt);
            }
        }
        bytes(k, Yp, row);
    }

    AVX_PARALLEL_FOR_IF(blend * npx * 3 * 8 >= AVX_PAR_MIN)
    for (int64_t task = 0; task < blend * Hp; ++task) {
        const int64_t k = task / Hp, Yp = task % Hp;
        float* buf = scratch_f32(need);
        float *row = buf, *p2 = row + RW + 16, *tmp = p2 + RW + 16, *l0 = tmp + RW + 16;
        float *l1 = l0 + TW3 + 16, *w = l1 + TW3 + 16;
        stitch_row(2 * skip + keep_n + k, Yp, row, l0, l1, w, p2, tmp);
        if (last) bytes(keep_n + k, Yp, row);
        for (int64_t X0 = 0; X0 < Wp; ++X0) {
            const int64_t pb = rowoff(PV, k * npx + Yp * Wp + X0, 3);
            for (int64_t c = 0; c < 3; ++c) stt(PV, pb + c * laststride(PV), row[X0 * 3 + c]);
        }
    }
    return RAD_OK;
}
