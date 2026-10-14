/* ref_gen.cpp -- the image generator's ops (docs/OPS.md, "Image generation").
 *
 * WHERE A ROUNDING IS WRITTEN OUT, IT IS THE REFERENCE MODEL'S. A diffusion transformer and its
 * decoder are run in bf16 by the reference implementation, so every tensor it materialises is
 * rounded there: `LN(x) * (1 + scale)` is three bf16 tensors, not one f32 expression, and the
 * oracle rounds where the reference does so a kernel that fuses the chain can be held to it.
 *
 * Geometry a step decides -- an image's height and width -- arrives in the OPERANDS, never in the
 * parameters: a 2-D image operand is [height, width * channels], dense, and the op reads both off
 * its shape. A parameter is fixed at declare and an image's size is a request's.
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"
#include "rad_philox.h"

#include <algorithm>
#include <vector>
#include <string>

using namespace ref;

namespace {

/* An entry's flags (docs/OPS.md, flow_euler): RadGenBatch's RAD_GEN_F_* values, restated because a
 * kernel library compiles against rad_abi.h and not the run phase's header. */
enum { kFirst = 1, kInpaint = 2, kPreview = 4 };

inline float rnd(uint32_t dt, float v) {
    uint16_t h;
    switch (dt) {
        case RAD_BF16: h = f32_to_bf16(v); return bf16_to_f32(h);
        case RAD_F16:  h = f32_to_f16(v);  return f16_to_f32(h);
        default:       return v;
    }
}
inline float rbf(float v) { return bf16_to_f32(f32_to_bf16(v)); }

inline int64_t cu_at(const RadTensor* cu, int64_t i) {
    return ld_int(cu->dtype, cu->data, offlin(cu, i));
}

/* The sequence that owns dense row `r`: linear, because there are a handful. */
inline int64_t seq_of(const RadTensor* cu, int64_t n_seq, int64_t r) {
    int64_t s = 0;
    while (s + 1 < n_seq && cu_at(cu, s + 1) <= r) ++s;
    return s;
}

inline int64_t rows_n(const RadTensor* t, int64_t n) {
    if (!t || n <= 0) return 0;
    if (t->rank >= 2 && t->shape[t->rank - 1] == n) {
        int64_t r = 1;
        for (uint32_t i = 0; i + 1 < t->rank; ++i) r *= t->shape[i];
        return r;
    }
    return numel(t) / n;
}

/* A space-separated list of floats, each rounded to bf16 as the reference's bf16 tensor of it is. */
int parse_list(const char* s, std::vector<float>* out) {
    out->clear();
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        out->push_back(rbf((float)v));
        s = end;
    }
    return (int)out->size();
}

}  /* namespace */

/* ------------------------------------------------------------------ ln_mod */
/* y = bf16( bf16(LN(x)) * bf16(1 + s[seq]) ): a layer norm with no gain and no bias, then the
 * modulation row of the row's sequence. `s` is [n_seq, >= n] at its own row pitch -- a column
 * slice of the modulation projection. */
int ref_ln_mod(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *s = &a->t[1], *cu = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_n(x, n) < rows) return RAD_E_SHAPE;
    const int64_t s_rs = row_stride(s, n), s_cs = laststride(s);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t e = seq_of(cu, n_seq, r);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum += ldt(x, xb + i * xs);
        const float mean = sum / (float)n;
        float var = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float d = ldt(x, xb + i * xs) - mean;
            var += d * d;
        }
        const float rs = 1.0f / std::sqrt(var / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            const float ln = rnd(y->dtype, (ldt(x, xb + i * xs) - mean) * rs);
            const float sc = rnd(y->dtype, 1.0f + ldt(s, e * s_rs + i * s_cs));
            stt(y, yb + i * ys, ln * sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gate_add */
/* x = bf16( x + bf16( bf16(tanh(g[seq])) * y ) ): a residual added through its sequence's gate. */
int ref_gate_add(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *y = &a->t[0], *g = &a->t[1], *cu = &a->t[2], *x = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n), n_seq = numel(cu) - 1;
    if (rows <= 0 || n_seq <= 0 || rows_n(x, n) < rows) return RAD_E_SHAPE;
    const int64_t g_rs = row_stride(g, n), g_cs = laststride(g);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t e = seq_of(cu, n_seq, r);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            const float t = rnd(x->dtype, std::tanh(ldt(g, e * g_rs + i * g_cs)));
            const float p = rnd(x->dtype, t * ldt(y, yb + i * ys));
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + p);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ln_mod_quant_fp8 */
/* gate_add (with `gated` 1), ln_mod and quant_act_fp8 over each row, each step's output rounded
 * to its stored dtype as the three ops would store it: the residual to x's, the norm to bf16, and
 * the codes as ref_quant_act_fp8 writes them over that bf16 row. */
int ref_ln_mod_quant_fp8(const RadArgs* a, RadStream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *y = t_in(a, 0), *g = t_in(a, 1), *s = &a->t[2], *cu = &a->t[3];
    const RadTensor *x = &a->t[4], *q = &a->t[5], *sc = &a->t[6], *z = t_in(a, 7);
    const int64_t n = p_int(a, "n", 0), grp = p_int(a, "group", 0);
    const bool gated = p_int(a, "gated", 0) != 0;
    const float eps = p_f32(a, "eps", 1e-6f);
    if (n <= 0 || grp <= 0 || n % grp || (gated && (!y || !g))) return RAD_E_SHAPE;
    const int64_t rows = rows_n(q, n), n_seq = numel(cu) - 1, ngrp = n / grp;
    if (rows <= 0 || n_seq <= 0 || rows_n(x, n) < rows || numel(sc) < rows * ngrp)
        return RAD_E_SHAPE;
    const int64_t s_rs = row_stride(s, n), s_cs = laststride(s);
    const int64_t g_rs = gated ? row_stride(g, n) : 0, g_cs = gated ? laststride(g) : 0;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t e = seq_of(cu, n_seq, r);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        std::vector<float> v((size_t)n), o((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            float xv = ldt(x, xb + i * xs);
            if (gated) {
                const float t = rnd(x->dtype, std::tanh(ldt(g, e * g_rs + i * g_cs)));
                const float d = ldt(y, rowoff(y, r, n) + i * laststride(y));
                xv = rnd(x->dtype, xv + rnd(x->dtype, t * d));
                stt(x, xb + i * xs, xv);
            }
            v[(size_t)i] = xv;
        }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum += v[(size_t)i];
        const float mean = sum / (float)n;
        float var = 0.0f;
        for (int64_t i = 0; i < n; ++i) { const float d = v[(size_t)i] - mean; var += d * d; }
        const float rs = 1.0f / std::sqrt(var / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            const float ln = rnd(RAD_BF16, (v[(size_t)i] - mean) * rs);
            const float sm = rnd(RAD_BF16, 1.0f + ldt(s, e * s_rs + i * s_cs));
            o[(size_t)i] = rnd(RAD_BF16, ln * sm);
            if (z) stt(z, rowoff(z, r, n) + i * laststride(z), o[(size_t)i]);
        }
        const int64_t qb = rowoff(q, r, n), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            float amax = 0.0f;
            for (int64_t i = gi * grp; i < (gi + 1) * grp; ++i)
                amax = std::max(amax, std::fabs(o[(size_t)i]));
            const float scl = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / scl;
            for (int64_t i = gi * grp; i < (gi + 1) * grp; ++i) {
                float w = o[(size_t)i] * inv;
                if (w > 448.0f) w = 448.0f;
                if (w < -448.0f) w = -448.0f;
                st_dt(q->dtype, q->data, qb + i * qs, w);
            }
            stt(sc, rowoff(sc, r, ngrp) + gi * laststride(sc), scl);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ timestep_embed */
/* y[m] = [cos(t' f_0) .. cos(t' f_{h-1}) | sin(t' f_0) .. sin(t' f_{h-1})], h = n / 2, where
 * f_i = exp(-ln(max_period) * i / h) in f32 and t' = scale * t -- or, with `round` 1, the value
 * diffusers' pipeline hands its transformer: t rounded to bf16 at scale, divided back and rounded
 * again, then scaled: scale * bf16(bf16(t * scale) / scale). */
int ref_timestep_embed(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *t = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float scale = p_f32(a, "scale", 1000.0f);
    const double mp = p_f32(a, "max_period", 10000.0f);
    const int round = (int)p_int(a, "round", 0);
    if (n <= 0 || (n & 1)) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || numel(t) < rows) return RAD_E_SHAPE;
    const int64_t h = n / 2;
    const float coef = (float)(-std::log(mp));
    for (int64_t m = 0; m < rows; ++m) {
        const float tv = ldt(t, offlin(t, m));
        const float tt = round ? scale * rbf(rbf(tv * scale) / scale) : scale * tv;
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        for (int64_t i = 0; i < h; ++i) {
            const float f = std::exp(coef * (float)i / (float)h);
            const float arg = tt * f;
            stt(y, yb + i * ys, std::cos(arg));
            stt(y, yb + (h + i) * ys, std::sin(arg));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows2 */
/* y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]] -- a sequence assembled from two sources. */
int ref_gather_rows2(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *idx = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || numel(idx) < rows) return RAD_E_SHAPE;
    const int64_t ra = rows_n(A, n), rbn = rows_n(B, n);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, i));
        const RadTensor* src = r >= 0 ? A : B;
        const int64_t row = r >= 0 ? r : -1 - r;
        const int64_t yb = rowoff(y, i, n), ys = laststride(y);
        if (row >= (r >= 0 ? ra : rbn)) {
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, 0.0f);
            continue;
        }
        const int64_t sb = rowoff(src, row, n), ss = laststride(src);
        for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(src, sb + c * ss));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lane_gather / lane_store */
/* lane_gather: y[cu[e] + k] = src[lane[e] * stride + k] for the rows of each entry. */
int ref_lane_gather(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *cu = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, src_rows = rows_n(src, n);
    if (n_seq <= 0 || numel(lane) < n_seq) return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), b1 = cu_at(cu, e + 1);
        if (L < 0 || b1 - b0 > stride || (L + 1) * stride > src_rows) return RAD_E_SHAPE;
        for (int64_t k = 0; k < b1 - b0; ++k) {
            const int64_t sb = rowoff(src, L * stride + k, n), ss = laststride(src);
            const int64_t yb = rowoff(y, b0 + k, n), ys = laststride(y);
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, ldt(src, sb + c * ss));
        }
    }
    return RAD_OK;
}

/* lane_store: every listed lane's rows [0, rows) become src's rows. */
int ref_lane_store(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *src = &a->t[0], *lane = &a->t[1], *dst = &a->t[2];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(src, n), dst_rows = rows_n(dst, n), nl = numel(lane);
    if (rows <= 0 || rows > stride) return RAD_E_SHAPE;
    for (int64_t l = 0; l < nl; ++l) {
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, l));
        if (L < 0 || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
        for (int64_t k = 0; k < rows; ++k) {
            const int64_t sb = rowoff(src, k, n), ss = laststride(src);
            const int64_t db = rowoff(dst, L * stride + k, n), ds = laststride(dst);
            for (int64_t c = 0; c < n; ++c) stt(dst, db + c * ds, ldt(src, sb + c * ss));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ randn_lanes */
/* The first step of an image: its lane's rows [0, cu[e+1] - cu[e]) become the seed's normal
 * stream (abi/rad_philox.h), element (k, c) the stream's element k * n + c. Entries whose flags
 * lack RAD_GEN_F_FIRST are left alone. */
int ref_randn_lanes(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *fl = &a->t[0], *seed = &a->t[1], *lane = &a->t[2], *cu = &a->t[3];
    const RadTensor *dst = &a->t[4];
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, dst_rows = rows_n(dst, n);
    if (n_seq <= 0 || numel(fl) < n_seq || numel(lane) < n_seq || numel(seed) < 2 * n_seq)
        return RAD_E_SHAPE;
    for (int64_t e = 0; e < n_seq; ++e) {
        if (!(ld_int(fl->dtype, fl->data, offlin(fl, e)) & kFirst)) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t len = cu_at(cu, e + 1) - cu_at(cu, e);
        if (L < 0 || len > stride || (L + 1) * stride > dst_rows) return RAD_E_SHAPE;
        const uint64_t sd = (uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e)) |
                            ((uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e + 1)) << 32);
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t db = rowoff(dst, L * stride + k, n), ds = laststride(dst);
            for (int64_t c = 0; c < n; ++c)
                stt(dst, db + c * ds, rad_philox_normal(sd, (uint64_t)(k * n + c)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ flow_euler */
/* One Euler step of a flow-matching sampler for every entry that moves a lane (pair != -2):
 *
 *   v'  = pair >= 0 ? bf16(neg + bf16(cfg * bf16(pos - neg))) : pos   the guided velocity
 *   x'  = bf16(x + bf16(bf16(sigma_next - sigma) * v'))              diffusers' step, its roundings
 *
 * and, by flag, kPreview writes bf16(x - sigma * v') -- the step's estimate of the clean
 * latent -- to the lane's preview rows; kInpaint puts the kept region back afterwards:
 * x' = bf16(m x' + (1 - m)((1 - sigma_next) src + sigma_next noise)), m the lane's mask row and
 * noise the image's own stream (randn_lanes). At the last step sigma_next is 0 and the kept
 * region is the picture's latent exactly. */
int ref_flow_euler(const RadArgs* a, RadStream) {
    if (!have(a, 13)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *sg = &a->t[1], *sgn = &a->t[2], *cfg = &a->t[3];
    const RadTensor *pair = &a->t[4], *lane = &a->t[5], *fl = &a->t[6], *seed = &a->t[7];
    const RadTensor *cu = &a->t[8];
    const RadTensor *inp = t_in(a, 9), *mask = t_in(a, 10);
    const RadTensor *st = &a->t[11];
    const RadTensor *pv = t_in(a, 12);
    const int64_t n = p_int(a, "n", 0), stride = p_int(a, "stride", 0);
    if (n <= 0 || stride <= 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1, st_rows = rows_n(st, n);
    if (n_seq <= 0) return RAD_E_SHAPE;
    const uint32_t dt = st->dtype;
    for (int64_t e = 0; e < n_seq; ++e) {
        const int64_t pr = ld_int(pair->dtype, pair->data, offlin(pair, e));
        if (pr == -2) continue;
        const int64_t L = ld_int(lane->dtype, lane->data, offlin(lane, e));
        const int64_t b0 = cu_at(cu, e), len = cu_at(cu, e + 1) - b0;
        const int64_t p0 = pr >= 0 ? cu_at(cu, pr) : 0;
        if (L < 0 || len > stride || (L + 1) * stride > st_rows) return RAD_E_SHAPE;
        if (pr >= 0 && cu_at(cu, pr + 1) - p0 != len) return RAD_E_SHAPE;
        const float s  = ldt(sg, offlin(sg, e)), sn = ldt(sgn, offlin(sgn, e));
        const float w  = ldt(cfg, offlin(cfg, e));
        const int64_t f = ld_int(fl->dtype, fl->data, offlin(fl, e));
        const uint64_t sd = (uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e)) |
                            ((uint64_t)(uint32_t)ld_int(seed->dtype, seed->data, offlin(seed, 2 * e + 1)) << 32);
        const float dts = rnd(dt, sn - s);
        if ((f & kInpaint) && (!inp || !mask)) return RAD_E_INVAL;
        if ((f & kPreview) && !pv) return RAD_E_INVAL;
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < len; ++k) {
            const int64_t vb = rowoff(v, b0 + k, n), vs = laststride(v);
            const int64_t nb = pr >= 0 ? rowoff(v, p0 + k, n) : 0;
            const int64_t sb = rowoff(st, L * stride + k, n), ss = laststride(st);
            for (int64_t c = 0; c < n; ++c) {
                float vv = ldt(v, vb + c * vs);
                if (pr >= 0) {
                    const float vn = ldt(v, nb + c * vs);
                    vv = rnd(dt, vn + rnd(dt, w * rnd(dt, vv - vn)));
                }
                const float x = ldt(st, sb + c * ss);
                if (f & kPreview) {
                    const int64_t pb = rowoff(pv, L * stride + k, n);
                    stt(pv, pb + c * laststride(pv), x - s * vv);
                }
                float xn = rnd(dt, x + rnd(dt, dts * vv));
                if (f & kInpaint) {
                    const float m = ldt(mask, offlin(mask, L * stride + k));
                    const float src = ldt(inp, rowoff(inp, L * stride + k, n) + c * laststride(inp));
                    const float nz = rad_philox_normal(sd, (uint64_t)(k * n + c));
                    const float kept = (1.0f - sn) * src + sn * nz;
                    xn = rnd(dt, m * xn + (1.0f - m) * kept);
                }
                stt(st, sb + c * ss, xn);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ chan_affine */
/* A per-channel affine from two lists the PARAMETERS carry (a latent's mean and deviation are
 * configuration, not weights), each value rounded to bf16 as the reference's bf16 tensor of it:
 * mode `fma` is bf16(bf16(x * mul) + add), `inv` is bf16(bf16(x - add) / mul). */
int ref_chan_affine(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    std::vector<float> mul, add;
    if (n <= 0 || parse_list(p_str(a, "mul", ""), &mul) != n ||
        parse_list(p_str(a, "add", ""), &add) != n)
        return RAD_E_SHAPE;
    const char* mode = p_str(a, "mode", "fma");
    const bool inv = mode && !std::strcmp(mode, "inv");
    if (!inv && mode && std::strcmp(mode, "fma")) return RAD_E_INVAL;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || rows_n(x, n) < rows) return RAD_E_SHAPE;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(x, xb + c * xs);
            stt(y, yb + c * ys, inv ? rnd(y->dtype, v - add[c]) / mul[c]
                                    : rnd(y->dtype, v * mul[c]) + add[c]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d_nhwc */
/* An image convolution, channels innermost: x is [Hin, Win * cin], y [Hout, Wout * cout], both
 * dense, and the sizes are read off them. w is PyTorch's [cout, cin, k, k] and b [cout].
 *
 *   up 2      the input is first upsampled 2x nearest-neighbour (`nearest-exact` at an exact
 *             factor of two is the same): the conv reads x[y / 2, x / 2]
 *   pad same  k / 2 zeros on every side
 *   pad down  no zeros above and left, one below and right -- ZeroPad2d((0, 1, 0, 1)) then the
 *             conv, a stride-2 downsample's
 *
 * f32 accumulation over (kh, kw, cin), the bias added, one rounding. */
int ref_conv2d_nhwc(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *b = &a->t[2], *y = &a->t[3];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), k = p_int(a, "k", 0);
    const int64_t stride = p_int(a, "stride", 1), up = p_int(a, "up", 1);
    const char* pad = p_str(a, "pad", "same");
    const bool down = pad && !std::strcmp(pad, "down");
    if (cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t Hin = x->shape[0], Win = x->shape[1] / cin;
    const int64_t Ho = y->shape[0], Wo = y->shape[1] / cout;
    const int64_t Hv = Hin * up, Wv = Win * up;
    const int64_t pt = down ? 0 : k / 2, pb = down ? 1 : k / 2;
    if ((Hv + pt + pb - k) / stride + 1 != Ho || (Wv + pt + pb - k) / stride + 1 != Wo)
        return RAD_E_SHAPE;
    if (numel(w) < cout * cin * k * k || numel(b) < cout) return RAD_E_SHAPE;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    #pragma omp parallel for schedule(static)
    for (int64_t oy = 0; oy < Ho; ++oy)
        for (int64_t ox = 0; ox < Wo; ++ox)
            for (int64_t co = 0; co < cout; ++co) {
                float acc = 0.0f;
                for (int64_t kh = 0; kh < k; ++kh) {
                    const int64_t iy = oy * stride + kh - pt;
                    if (iy < 0 || iy >= Hv) continue;
                    for (int64_t kw = 0; kw < k; ++kw) {
                        const int64_t ix = ox * stride + kw - pt;
                        if (ix < 0 || ix >= Wv) continue;
                        const int64_t xb = (iy / up) * xr + (ix / up) * cin;
                        for (int64_t ci = 0; ci < cin; ++ci)
                            acc += ldt(w, ((co * cin + ci) * k + kh) * k + kw) * ldt(x, xb + ci);
                    }
                }
                stt(y, oy * yr + ox * cout + co, acc + ldt(b, co));
            }
    return RAD_OK;
}

/* ------------------------------------------------------------------ l2norm_chan */
/* An image norm over the channels of each pixel, the reference's three bf16 tensors:
 * n1 = bf16(x / max(||x||, 1e-12)), n2 = bf16(n1 * sqrt(n)), y = bf16(n2 * gamma), then `act`
 * (none, or silu: bf16(y * sigmoid(y))). */
int ref_l2norm_chan(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *g = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || numel(g) < n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool silu = act && !std::strcmp(act, "silu");
    if (!silu && act && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || rows_n(x, n) < rows) return RAD_E_SHAPE;
    const float sq = (float)std::sqrt((double)n);
    const uint32_t dt = y->dtype;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        float ss = 0.0f;
        for (int64_t c = 0; c < n; ++c) { const float v = ldt(x, xb + c * xs); ss += v * v; }
        const float nrm = std::max(std::sqrt(ss), 1e-12f);
        for (int64_t c = 0; c < n; ++c) {
            const float n1 = rnd(dt, ldt(x, xb + c * xs) / nrm);
            const float n2 = rnd(dt, n1 * sq);
            float v = rnd(dt, n2 * ldt(g, c));
            if (silu) v = v / (1.0f + std::exp(-v));
            stt(y, yb + c * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ d2s_add / s2d_avg_add */
/* d2s_add, the decoder's upsampling shortcut (Wan's DupUp3D on one frame): x [H, W * cin] repeated
 * `rep = cout * F / cin` times per channel and dealt into a 2x larger y [2H, 2W * cout], added:
 * y[2r+i, 2c+j, o] += x[r, c, (o * F + off + 2i + j) / rep], F = 4 * ft, off = 4 when ft is 2
 * (the second of the two frames a temporal factor makes is the one a first chunk keeps). */
int ref_d2s_add(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = x->shape[0], W = x->shape[1] / cin;
    if (y->shape[0] != 2 * H || y->shape[1] / cout != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft, off = ft == 2 ? 4 : 0;
    if ((cout * F) % cin) return RAD_E_SHAPE;
    const int64_t rep = cout * F / cin;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    #pragma omp parallel for schedule(static)
    for (int64_t oy = 0; oy < 2 * H; ++oy)
        for (int64_t ox = 0; ox < 2 * W; ++ox)
            for (int64_t o = 0; o < cout; ++o) {
                const int64_t i = oy & 1, j = ox & 1;
                const int64_t ci = (o * F + off + 2 * i + j) / rep;
                const int64_t yo = oy * yr + ox * cout + o;
                stt(y, yo, ldt(y, yo) + ldt(x, (oy / 2) * xr + (ox / 2) * cin + ci));
            }
    return RAD_OK;
}

/* s2d_avg_add, the encoder's downsampling shortcut (Wan's AvgDown3D on one frame): each 2x2 block
 * of x [2H, 2W * cin] dealt into channels and averaged in groups of gs = cin * F / cout into
 * y [H, W * cout], added. A channel of the dealt row is (c, t, i, j) at c * F + 4t + 2i + j; with
 * ft 2 the frame t == 0 is the zero frame the reference pads in front, which the mean still counts. */
int ref_s2d_avg_add(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0), ft = p_int(a, "ft", 1);
    if (cin <= 0 || cout <= 0 || (ft != 1 && ft != 2)) return RAD_E_SHAPE;
    if (x->rank != 2 || y->rank != 2 || x->shape[1] % cin || y->shape[1] % cout) return RAD_E_SHAPE;
    const int64_t H = y->shape[0], W = y->shape[1] / cout;
    if (x->shape[0] != 2 * H || x->shape[1] / cin != 2 * W) return RAD_E_SHAPE;
    const int64_t F = 4 * ft;
    if ((cin * F) % cout) return RAD_E_SHAPE;
    const int64_t gs = cin * F / cout;
    const int64_t xr = x->stride[0], yr = y->stride[0];
    #pragma omp parallel for schedule(static)
    for (int64_t oy = 0; oy < H; ++oy)
        for (int64_t ox = 0; ox < W; ++ox)
            for (int64_t o = 0; o < cout; ++o) {
                float acc = 0.0f;
                for (int64_t gg = 0; gg < gs; ++gg) {
                    const int64_t g = o * gs + gg;
                    const int64_t c = g / F, rem = g % F, t = rem / 4, i = (rem % 4) / 2, j = rem % 2;
                    if (ft == 2 && t == 0) continue;
                    acc += ldt(x, (2 * oy + i) * xr + (2 * ox + j) * cin + c);
                }
                const int64_t yo = oy * yr + ox * cout + o;
                stt(y, yo, ldt(y, yo) + rnd(y->dtype, acc / (float)gs));
            }
    return RAD_OK;
}

/* ------------------------------------------------------------------ to_rgba8 */
/* A decoded image to bytes the way diffusers' postprocess makes them: v = clamp(x, -1, 1),
 * u = clamp(bf16(bf16(v / 2) + 0.5), 0, 1), byte = round-half-even(f32(u) * 255). */
int ref_to_rgba8(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || rows_n(x, n) < rows || y->dtype != RAD_U8) return RAD_E_SHAPE;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            float v = ldt(x, xb + c * xs);
            v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            float u = rbf(rbf(v * 0.5f) + 0.5f);
            u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
            ((uint8_t*)y->data)[yb + c * ys] = (uint8_t)std::nearbyintf(u * 255.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_tanh, scatter_add_rows */
int ref_gelu_tanh(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_n(y, n);
    if (rows <= 0 || rows_n(x, n) < rows) return RAD_E_SHAPE;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t c = 0; c < n; ++c) {
            const float v = ldt(x, xb + c * xs);
            const float t = std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v));
            stt(y, yb + c * ys, 0.5f * v * (1.0f + t));
        }
    }
    return RAD_OK;
}

/* x[idx[i]] = x[idx[i]] + v[i], rounded to x's dtype; a negative index is a skip. */
int ref_scatter_add_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_n(v, n), dst_rows = rows_n(x, n);
    if (M <= 0 || numel(idx) < M) return RAD_E_SHAPE;
    for (int64_t m = 0; m < M; ++m)
        if (ld_int(idx->dtype, idx->data, offlin(idx, m)) >= dst_rows) return RAD_E_INVAL;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), vs = laststride(v);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(x, xb + i * xs, ldt(x, xb + i * xs) + ldt(v, vb + i * vs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_prefix_self */
/* Attention over two key sources: the sequence's PREFIX, `seqused[s]` positions in the paged cache,
 * and the step's OWN rows, `k` and `v` beside `q` and not in the cache. Self row i of a sequence
 * sees every prefix key and, with `causal`, its own rows j <= i -- otherwise all of them.
 *
 * It is the attention of a sequence whose rows are recomputed each step against a prefix that is
 * not: an image's latent rows against the conditioning, or one segment of the conditioning against
 * the segments before it. */
/* quant_act_i8g's rule for one row of `n`: the scale (amax/127, 1 for an all-zero row) and the
 * codes rint(clamp(x * (1/scale), +-127)), in f32 as the device computes them. `code` may be `x`:
 * each code is written after its value is read. */
static float ref_q8_row(const float* x, int64_t n, float* code) {
    float am = 0.0f;
    for (int64_t d = 0; d < n; ++d) am = std::max(am, std::fabs(x[d]));
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t d = 0; d < n; ++d)
        code[d] = std::rint(std::min(std::max(x[d] * inv, -127.0f), 127.0f));
    return sc;
}

int ref_attn_prefix_self(const RadArgs* a, RadStream) {
    if (a->n_t < 12) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *kv = &a->t[3];
    const RadTensor *bt = &a->t[4], *su = &a->t[5], *cu = &a->t[6], *o = &a->t[11];
    const RadTensor *kdes = t_in(a, 7), *vdes = t_in(a, 8);
    /* THE INT8 Q.K FORM: the own keys' codes and scales given, everything else quantised here. */
    const RadTensor *k8 = t_in(a, 9), *ks = t_in(a, 10);
    const char* qk = p_str(a, "qk", "bf16");
    const bool i8 = std::strcmp(qk, "i8") == 0;
    if (i8 != (k8 != nullptr) || i8 != (ks != nullptr)) return RAD_E_INVAL;
    if (i8 && (kdes || vdes)) return RAD_E_UNSUPPORTED;
    const int64_t head_dim = p_int(a, "head_dim", 0), gqa = p_int(a, "gqa", 1);
    const int causal = (int)p_int(a, "causal", 0);
    const int64_t block_size = kv->rank == 4 ? kv->shape[2] : p_int(a, "block_size", 0);
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > 1024) return RAD_E_UNSUPPORTED;
    const int64_t kv_heads = kv->rank == 4 ? kv->shape[1] : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = kv_heads * gqa, n_seq = numel(cu) - 1;
    if (n_seq <= 0 || numel(su) < n_seq) return RAD_E_SHAPE;
    /* The rows are cu's own span, which need not start at 0: one segment of a longer sequence is
     * passed as the two offsets that bound it. */
    const int64_t r_first = cu_at(cu, 0), rows = cu_at(cu, n_seq) - r_first;
    const int64_t max_blocks = bt->rank >= 2 ? bt->shape[1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt->rank >= 2 ? bt->stride[0] : max_blocks;
    const int64_t q_rs = row_stride(q, q_heads * head_dim), k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_rs = row_stride(v, kv_heads * head_dim), o_rs = row_stride(o, q_heads * head_dim);
    const int64_t k8_rs = i8 ? row_stride(k8, kv_heads * head_dim) : 0;
    const int64_t ks_rs = i8 ? row_stride(ks, kv_heads) : 0;
    const int64_t sb = kv->stride[0], sh = kv->stride[1], so = kv->stride[2];
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < rows * q_heads; ++idx) {
        const int64_t h = idx % q_heads, r = r_first + idx / q_heads, kh = h / gqa;
        const int64_t s = seq_of(cu, n_seq, r), b0 = cu_at(cu, s), b1 = cu_at(cu, s + 1);
        const int64_t i = r - b0, P = ld_int(su->dtype, su->data, offlin(su, s));
        const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
        const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;
        const int64_t qb = r * q_rs + h * head_dim;
        const int64_t nself = causal ? i + 1 : b1 - b0;
        auto pkey = [&](int64_t j, int64_t d, bool val) -> float {
            const int64_t blk = ld_int(bt->dtype, bt->data, s * bt_rs + j / block_size);
            if (blk < 0) return 0.0f;
            return ldt(kv, blk * sb + kh * sh + (j % block_size) * so + (val ? head_dim : 0) + d) *
                   (val ? vds : kds);
        };
        float xq[1024], cq[1024], ck[1024];
        float sq = 1.0f;
        if (i8) {
            for (int64_t d = 0; d < head_dim; ++d) xq[d] = ldt(q, qb + d);
            sq = ref_q8_row(xq, head_dim, cq);
        }
        auto dot = [&](int64_t j) -> float {
            if (i8) {
                float sk = 0.0f;
                if (j < P) {
                    for (int64_t d = 0; d < head_dim; ++d) ck[d] = pkey(j, d, false);
                    sk = ref_q8_row(ck, head_dim, ck);
                } else {
                    const int64_t row = b0 + j - P;
                    for (int64_t d = 0; d < head_dim; ++d)
                        ck[d] = ldt(k8, row * k8_rs + kh * head_dim + d);
                    sk = ldt(ks, row * ks_rs + kh);
                }
                /* Integers below 2^24 in magnitude: the f32 sum is exact. */
                float acc = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) acc += cq[d] * ck[d];
                return acc * sq * sk * scale;
            }
            float acc = 0.0f;
            if (j < P) {
                for (int64_t d = 0; d < head_dim; ++d) acc += ldt(q, qb + d) * pkey(j, d, false);
            } else {
                const int64_t kb = (b0 + j - P) * k_rs + kh * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) acc += ldt(q, qb + d) * ldt(k, kb + d);
            }
            return acc * scale;
        };
        const int64_t nk = P + nself;
        float mx = -INFINITY;
        for (int64_t j = 0; j < nk; ++j) mx = std::max(mx, dot(j));
        float acc[1024];
        for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t j = 0; j < nk; ++j) {
            const float wgt = ref_exp(dot(j) - mx);
            sum += wgt;
            if (j < P) {
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += wgt * pkey(j, d, true);
            } else {
                const int64_t vb = (b0 + j - P) * v_rs + kh * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += wgt * ldt(v, vb + d);
            }
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const int64_t ob = r * o_rs + h * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d, acc[d] * inv);
    }
    return RAD_OK;
}
