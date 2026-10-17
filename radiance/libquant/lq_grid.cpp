/* lq_grid.cpp -- parsing a grid, choosing a block's scale, and round-to-nearest on it. See
 * lq_grid.h for the options and why the rules must not change. */
#include "lq_grid.h"
#include "lq_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lq {

namespace {

std::string fmt(const char* f, const char* a = "", long long b = 0) {
    char m[256];
    std::snprintf(m, sizeof m, f, a, b);
    return m;
}

/* "128x128", "1x*", "*x*" -> rows and columns, 0 for `*`. */
bool parse_block(const char* s, int64_t* br, int64_t* bc) {
    const char* x = std::strchr(s, 'x');
    if (!x) return false;
    auto one = [](const char* a, const char* e, int64_t* out) {
        if (e - a == 1 && *a == '*') { *out = 0; return true; }
        char* end = nullptr;
        const long long v = std::strtoll(a, &end, 10);
        if (end != e || v <= 0) return false;
        *out = v;
        return true;
    };
    return one(s, x, br) && one(x + 1, s + std::strlen(s), bc);
}

bool is_float_code(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_FP6E2M3 || dt == RAD_FP6E3M2 ||
           dt == RAD_FP4E2M1;
}

/* A scale dtype the planes can hold and a reader can widen. */
bool is_scale_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F8E4M3 || dt == RAD_E8M0;
}

/* The smallest positive value of a float scale dtype; a scale that would round to zero becomes
 * this, so its reciprocal stays finite. */
float tiny_of(uint32_t dt) {
    switch (dt) {
        case RAD_F16:    return 0x1p-24f;
        case RAD_F8E4M3: return 0x1p-9f;
        case RAD_BF16:   return 0x1p-133f;
        default:         return 0x1p-149f;
    }
}

/* `v` rounded to a float scale dtype, never to zero: general-path scales are always finite and
 * nonzero. */
float round_nz(uint32_t dt, float v) {
    float r = round_to(dt, v);
    if (r == 0.0f || !std::isfinite(r)) r = std::copysign(tiny_of(dt), v == 0.0f ? 1.0f : v);
    return r;
}

/* ------------------------------------------------------------------ the named codebooks */
struct NamedTable { const char* name; std::vector<float> v; };

const std::vector<NamedTable>& named_tables() {
    static const std::vector<NamedTable> t = {
        /* QLoRA's NormalFloat-4: the quantiles of a unit normal, with an exact zero and the
         * extremes at +-1 (bitsandbytes' table, in code order). */
        { "nf4", { -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
                   -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
                   0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
                   0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
                   0.7229568362236023f, 1.0f } },
        /* llama.cpp's IQ4_NL values: a non-linear 4-bit grid with no zero and unequal ends. */
        { "iq4nl", { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 } },
        /* Sixteen levels for Hadamard-rotated weights: Lloyd's quantiser fitted to the rotated,
         * amax-normalised Qwen3.8-Flash-Next experts (the same in every layer and both
         * projections), each level rounded to the nearest E4M3 value. Being exact in E4M3, a code
         * expands with no rounding to the byte libr4d's w4nla8h fp8 WMMA reads. Its table
         * (r4d_args.h, kW4nlTable) holds these values in this code order (magnitudes ascending,
         * then their negatives), and a container whose table differs is refused at load. About
         * 11% less output error than -8..7 under the same searched scale, at the same bytes. */
        { "w4nl", { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                    -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f } },
        /* The 32-level form for libr4d's w5nl64a8h: sixteen magnitudes fitted the same way at
         * groups of 64 (again shared by every layer and projection), rounded to E4M3 at the top
         * level that keeps all sixteen distinct with least movement. Magnitudes ascending, then
         * their negatives, so a code's high bit is its sign. About 72% less output error energy
         * than w4nl under the same scales, GPTQ and coordinate descent, for one more bit. */
        { "w5nl", { 0.8125f, 2.5f, 4.0f, 5.5f, 7.5f, 9.0f, 11.0f, 12.0f,
                    14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 30.0f,
                    -0.8125f, -2.5f, -4.0f, -5.5f, -7.5f, -9.0f, -11.0f, -12.0f,
                    -14.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -30.0f } },
        /* A one-bit weight: a sign and the block's scale. */
        { "binary", { -1.0f, 1.0f } },
        /* BitNet's ternary weight, a two-bit code with one code unused. */
        { "ternary", { -1.0f, 0.0f, 1.0f } },
    };
    return t;
}

/* A codebook from its option: a name above, or the values themselves, `;`-separated. */
bool parse_table(const char* s, std::vector<float>* out, std::string* why) {
    for (const NamedTable& t : named_tables())
        if (!std::strcmp(s, t.name)) { *out = t.v; return true; }
    out->clear();
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        const float v = std::strtof(p, &end);
        if (end == p || !std::isfinite(v)) {
            if (why) *why = fmt("table=%s is neither nf4, iq4nl, w4nl, w5nl, binary, ternary nor a "
                                "`;`-separated list of values", s);
            return false;
        }
        out->push_back(v);
        p = end;
        if (*p == ';') ++p;
        else if (*p) {
            if (why) *why = fmt("table=%s: the values are separated by `;`", s);
            return false;
        }
    }
    if (out->size() < 2 || out->size() > 4096) {
        if (why) *why = fmt("table=%s holds %lld values; a codebook holds 2 to 4096", s,
                            (long long)out->size());
        return false;
    }
    return true;
}

/* The narrowest unsigned code that indexes `n` entries. */
uint32_t index_dtype(size_t n) {
    static const uint32_t u[] = { RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6,
                                  RAD_U7, RAD_U8, RAD_U9, RAD_U10, RAD_U11, RAD_U12 };
    int bits = 1;
    while (bits < 12 && ((size_t)1 << bits) < n) ++bits;
    return u[bits - 1];
}

/* The integer sub-scale dtypes a first level may be under a second. */
bool l1_int_range(uint32_t dt, int* lo, int* hi) {
    int sgn = 0;
    int bits = rad_dtype_packed_int(dt, &sgn);
    if (dt == RAD_U8) { bits = 8; sgn = 0; }
    if (dt == RAD_I8) { bits = 8; sgn = 1; }
    if (bits < 2 || bits > 8) return false;
    if (sgn) { *lo = -(1 << (bits - 1)); *hi = (1 << (bits - 1)) - 1; }
    else     { *lo = 0; *hi = (1 << bits) - 1; }
    return true;
}

/* ------------------------------------------------------------------ one code, the general path */

/* The unit value (before the scale) of code `c`. */
double unit_of(const Grid& g, int c) {
    if (!g.table.empty()) return g.table[(size_t)c];
    if (g.is_float) {
        uint8_t buf[4] = { (uint8_t)c, 0, 0, 0 };
        return rad_load_f32(buf, g.codes, 0);
    }
    return (double)c;
}

/* The codebook entry nearest `u`, ties to the lower value. */
int nearest_entry(const Grid& g, float u) {
    const std::vector<int>& o = g.table_order;
    size_t a = 0, b = o.size();
    while (a < b) {                                   /* first entry not below u */
        const size_t m = (a + b) / 2;
        if (g.table[(size_t)o[m]] < u) a = m + 1;
        else b = m;
    }
    if (a == 0) return o[0];
    if (a == o.size()) return o.back();
    const float lo = g.table[(size_t)o[a - 1]], hi = g.table[(size_t)o[a]];
    return (u - lo <= hi - u) ? o[a - 1] : o[a];
}

/* The code of `x` against scale `s` and, for an asymmetric grid, zero `z`. */
int code_at(const Grid& g, double x, double s, int z) {
    if (!g.table.empty()) return nearest_entry(g, (float)(x / s));
    const float u = (float)(x / s);
    if (g.is_float) {
        if (g.codes == RAD_F8E4M3) return f32_to_e4m3(u);
        if (g.codes == RAD_FP4E2M1) return f32_to_e2m1(u);
        return (int)f32_to_small_float(u, g.codes);
    }
    const double q = x / s;
    const double r = q < 0.0 ? std::ceil(q - 0.5) : std::floor(q + 0.5);   /* half away */
    const double c = r + (g.asym ? z : 0);
    return (int)std::min<double>(std::max<double>(c, g.qmin), g.qmax);
}

/* ------------------------------------------------------------------ rule=search */

/* The weighted squared error of a block coded against scale `s` (and zero `z`), and the two sums
 * its least-squares refit is formed from. */
struct Fit { double err = 0, sxd = 0, sdd = 0; };

Fit block_fit(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1, int64_t c0,
              int64_t c1, const float* imp, double s, int z) {
    Fit f;
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t c = c0; c < c1; ++c) {
            const double x = w[r * cols + c];
            const double wt = imp ? (double)imp[c] : 1.0;
            const int code = code_at(g, x, s, z);
            const double d = g.asym ? (double)(code - z) : unit_of(g, code);
            const double e = x - s * d;
            f.err += wt * e * e;
            f.sxd += wt * x * d;
            f.sdd += wt * d * d;
        }
    return f;
}

/* Symmetric search: 21 multiples of the absmax scale from 0.70 to 1.10, plus each candidate's
 * least-squares refit (the scale minimising the error for the codes it chose). Each is rounded to
 * the scale dtype before being judged, unless a second level will round it later, in which case
 * the best is returned unrounded. The sweep is llama.cpp's make_qx_quants idea over any code; the
 * refit makes a coarse sweep sufficient.
 *
 * `t2` is the second level once known: candidates are then judged as stored, the first level
 * rounded relative to t2 and clamped to its dtype's range, so only decodable scales are chosen. */
double search_sym(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1, int64_t c0,
                  int64_t c1, const float* imp, double s0, double t2 = 0.0) {
    if (s0 == 0.0) return s0;
    auto rd = [&](double s) {
        if (t2 > 0.0) {
            const double q = std::min(std::max(s / t2, -(double)g.l1max), (double)g.l1max);
            return (double)round_nz(g.scale_dt, (float)q) * t2;
        }
        return g.lv2 ? s : (double)round_nz(g.scale_dt, (float)s);
    };
    double best = rd(s0), best_err = block_fit(g, w, cols, r0, r1, c0, c1, imp, best, 0).err;
    for (int k = -15; k <= 5; ++k) {
        const double s = rd(s0 * (1.0 + 0.02 * k));
        const Fit f = block_fit(g, w, cols, r0, r1, c0, c1, imp, s, 0);
        if (f.err < best_err) { best_err = f.err; best = s; }
        if (f.sdd > 0.0 && f.sxd != 0.0) {
            const double t = rd(f.sxd / f.sdd);
            const double e = block_fit(g, w, cols, r0, r1, c0, c1, imp, t, 0).err;
            if (e < best_err) { best_err = e; best = t; }
        }
    }
    return best;
}

/* Asymmetric search: both range ends shrunk to six fractions from 1.00 to 0.80; the pair with
 * least error is kept. The zero is chosen against each candidate's rounded scale, as under the
 * absmax rule. */
void search_asym(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1, int64_t c0,
                 int64_t c1, const float* imp, float* lo, float* hi) {
    const float lo0 = *lo, hi0 = *hi;
    double best_err = -1.0;
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) {
            const float l = lo0 * (1.0f - 0.04f * (float)a), h = hi0 * (1.0f - 0.04f * (float)b);
            float s = (h - l) / (float)g.qmax;
            if (!(s > 0.0f)) continue;
            const float sr = g.lv2 ? s : round_nz(g.scale_dt, s);
            const int z = clampi((int)(-l / sr + 0.5f), 0, g.qmax);
            const double e = block_fit(g, w, cols, r0, r1, c0, c1, imp, sr, z).err;
            if (best_err < 0.0 || e < best_err) { best_err = e; *lo = l; *hi = h; }
        }
}

/* A first-level scale as stored under a transform. The plane holds s / N in the scale dtype and the
 * reader undoes the transform after multiplying by it, so a code is decoded against N times the
 * stored value, and codes are chosen against that. Dividing by N (a power of two) is exact unless
 * it leaves the dtype's normal range (an f16 or E4M3 scale in its subnormals, or below them); only
 * then does the scale the codes see move, and the stored value is never zero, so a non-empty block
 * decodes to something. Without a transform the stored value is the scale itself. */
void store_first_level(const Grid& g, float resid, BlockScale* b) {
    b->stored = round_to(g.scale_dt, b->s * resid);
    if (resid == 1.0f || b->stored / resid == b->s) return;
    if (b->stored == 0.0f) b->stored = round_nz(g.scale_dt, b->s * resid);
    b->s = b->stored / resid;
    b->inv = 1.0f / b->s;
    b->inv_d = 1.0 / (double)b->s;
    if (g.asym) b->zero = clampi((int)(-b->lo / b->s + 0.5f), 0, g.qmax);
}

}  /* namespace */

int grid_parse(const RadParam* o, int n_o, Grid* g, std::string* why) {
    *g = Grid{};
    const char* codes = rad_param_gets(o, n_o, "codes", nullptr);
    const char* tab = rad_param_gets(o, n_o, "table", nullptr);
    if (tab) {
        if (!parse_table(tab, &g->table, why)) return RAD_E_INVAL;
        if (rad_param_gets(o, n_o, "zero", nullptr)) {
            if (why) *why = "a codebook takes no zero point: its entries are the values";
            return RAD_E_INVAL;
        }
        const uint32_t need = index_dtype(g->table.size());
        if (codes && rad_dtype_parse(codes) != need) {
            if (why) *why = fmt("table=%s has %lld entries, which a code of exactly ", tab,
                                (long long)g->table.size()) +
                            rad_dtype_name(need) + " indexes; codes= says otherwise";
            return RAD_E_INVAL;
        }
        g->codes = need;
        bool ints = true;
        float mn = g->table[0], mx = g->table[0];
        for (float v : g->table) {
            if (v != std::nearbyint(v) || v < -128.0f || v > 127.0f) ints = false;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        g->table_dt = ints ? RAD_I8 : RAD_F32;
        g->table_order.resize(g->table.size());
        for (size_t i = 0; i < g->table.size(); ++i) g->table_order[i] = (int)i;
        std::stable_sort(g->table_order.begin(), g->table_order.end(),
                         [&](int a, int b) { return g->table[(size_t)a] < g->table[(size_t)b]; });
        if (!(mx > mn)) {
            if (why) *why = fmt("table=%s holds one value", tab);
            return RAD_E_INVAL;
        }
        g->tpeak = -mn >= mx ? mn : mx;
        g->tsigned = -mn != mx;
    } else if (!codes) {
        if (why) *why = "codes= is required: the code dtype, e.g. i4, u2, fp8_e4m3, fp4_e2m1 -- "
                        "or table= for a codebook";
        return RAD_E_INVAL;
    }
    if (!tab) g->codes = rad_dtype_parse(codes);
    const int bits = rad_dtype_bits(g->codes);
    g->is_float = tab ? false : is_float_code(g->codes);
    int sgn = 0;
    bool packed = rad_dtype_packed_int(g->codes, &sgn) > 0 || g->codes == RAD_I8 ||
                  g->codes == RAD_U8;
    if (g->codes == RAD_I8) sgn = 1;
    if (g->codes == RAD_I16 || g->codes == RAD_I32 || g->codes == RAD_U16) {
        packed = true;
        g->wide = true;
        sgn = g->codes != RAD_U16;
    }
    if (!tab && !g->is_float && !packed) {
        if (why) *why = fmt("codes=%s is not a code dtype this grid writes", codes);
        return RAD_E_INVAL;
    }

    if (const char* z = rad_param_gets(o, n_o, "zero", nullptr)) {
        g->zero_dt = rad_dtype_parse(z);
        int zs = 0;
        if (!(rad_dtype_packed_int(g->zero_dt, &zs) > 0 || g->zero_dt == RAD_U8 ||
              g->zero_dt == RAD_U16) || zs) {
            if (why) *why = fmt("zero=%s: an integer zero point is unsigned", z);
            return RAD_E_INVAL;
        }
        g->asym = true;
    }
    if (tab) {
        /* The codebook already set the code dtype. */
    } else if (!g->is_float) {
        if (g->asym && sgn) {
            if (why) *why = fmt("codes=%s is signed; an asymmetric grid's codes are unsigned "
                                "(u%lld)", codes, bits);
            return RAD_E_INVAL;
        }
        if (!g->asym && !sgn) {
            if (why) *why = fmt("codes=%s is unsigned and no zero= was given: an unsigned code "
                                "needs a zero point to reach a negative weight", codes);
            return RAD_E_INVAL;
        }
        if (g->asym) {
            g->qmin = 0;
            g->qmax = (int)((1ll << bits) - 1);
        } else {
            g->qmax = (int)((1ll << (bits - 1)) - 1);
            const char* cl = rad_param_gets(o, n_o, "clamp", "full");
            if (!std::strcmp(cl, "full")) g->qmin = (int)(-(1ll << (bits - 1)));
            else if (!std::strcmp(cl, "sym")) g->qmin = -g->qmax;
            else {
                if (why) *why = fmt("clamp=%s is neither full nor sym", cl);
                return RAD_E_INVAL;
            }
        }
    } else {
        if (g->asym) {
            if (why) *why = "a float code takes no zero point";
            return RAD_E_INVAL;
        }
        switch (g->codes) {
            case RAD_F8E4M3:  g->fmax = 448.0f;   g->emax = 8;  break;
            case RAD_F8E5M2:  g->fmax = 57344.0f; g->emax = 15; break;
            case RAD_FP6E2M3: g->fmax = 7.5f;     g->emax = 2;  break;
            case RAD_FP6E3M2: g->fmax = 28.0f;    g->emax = 4;  break;
            default:          g->fmax = 6.0f;     g->emax = 2;  break;   /* fp4 e2m1 */
        }
    }

    /* The second level first, because it decides which dtypes the first may use. */
    if (const char* s2 = rad_param_gets(o, n_o, "scale2", nullptr)) {
        g->lv2 = true;
        g->scale2_dt = rad_dtype_parse(s2);
        if (!is_scale_dt(g->scale2_dt)) {
            if (why) *why = fmt("scale2=%s is not a scale dtype (f32, bf16, f16, fp8_e4m3, e8m0)",
                                s2);
            return RAD_E_INVAL;
        }
        const long long g2 = rad_param_geti(o, n_o, "group2", 0);
        const char* b2 = rad_param_gets(o, n_o, "block2", nullptr);
        if (g2 > 0 && b2) {
            if (why) *why = "group2= and block2= both say where the second level sits; give one";
            return RAD_E_INVAL;
        }
        if (g2 > 0) { g->br2 = 1; g->bc2 = g2; }
        else if (b2 && !parse_block(b2, &g->br2, &g->bc2)) {
            if (why) *why = fmt("block2=%s is not RxC, with * for a whole extent", b2);
            return RAD_E_INVAL;
        }
        g->fixed2 = rad_param_getf(o, n_o, "scale2_value", 0.0);
        if (g->fixed2 != 0.0 &&
            (!(g->fixed2 > 0.0) || g->scale2_dt == RAD_E8M0 ||
             (double)round_to(g->scale2_dt, (float)g->fixed2) != g->fixed2)) {
            if (why) {
                char v[48];
                std::snprintf(v, sizeof v, "%.17g", g->fixed2);
                *why = std::string("scale2_value=") + v + " is not a positive value " +
                       rad_dtype_name(g->scale2_dt) + " holds exactly (an e8m0 second level is "
                       "a shared exponent and is not fixed)";
            }
            return RAD_E_INVAL;
        }
    } else if (rad_param_gets(o, n_o, "block2", nullptr) || rad_param_geti(o, n_o, "group2", 0) ||
               rad_param_getf(o, n_o, "scale2_value", 0.0) != 0.0) {
        if (why) *why = "block2=, group2= and scale2_value= describe a second scale level, and "
                        "scale2= names none";
        return RAD_E_INVAL;
    }

    const char* sdt = rad_param_gets(o, n_o, "scale", "bf16");
    g->scale_dt = rad_dtype_parse(sdt);
    if (g->lv2 && l1_int_range(g->scale_dt, &g->l1_lo, &g->l1_hi)) {
        g->l1_int = true;
        g->l1max = (float)g->l1_hi;
    } else if (!is_scale_dt(g->scale_dt)) {
        if (why) *why = fmt(g->lv2 ? "scale=%s is not a scale dtype (f32, bf16, f16, fp8_e4m3, or "
                                     "under scale2= an integer u4..u8, i6, i8)"
                                   : "scale=%s is not a scale dtype (f32, bf16, f16, fp8_e4m3, "
                                     "e8m0)", sdt);
        return RAD_E_INVAL;
    } else if (g->lv2) {
        if (g->scale_dt == RAD_E8M0) {
            if (why) *why = "scale=e8m0 under scale2=: a shared exponent is a level of its own";
            return RAD_E_INVAL;
        }
        g->l1max = g->scale_dt == RAD_F8E4M3 ? 448.0f : 1.0f;
    }
    if (g->tsigned && g->l1_int && g->l1_lo == 0) {
        if (why) *why = std::string("table=") + tab + " has ends of unequal magnitude, so a "
                        "block's scale carries a sign, and scale=" + sdt + " cannot hold one";
        return RAD_E_INVAL;
    }
    if (g->tsigned && g->scale_dt == RAD_E8M0) {
        if (why) *why = "a codebook with ends of unequal magnitude takes a signed scale, and e8m0 "
                        "is a bare exponent";
        return RAD_E_INVAL;
    }

    const long long group = rad_param_geti(o, n_o, "group", 0);
    const char* blk = rad_param_gets(o, n_o, "block", nullptr);
    if (group > 0 && blk) {
        if (why) *why = "group= and block= both say where the scales sit; give one";
        return RAD_E_INVAL;
    }
    if (group > 0) {
        g->br = 1;
        g->bc = group;
    } else if (blk && !parse_block(blk, &g->br, &g->bc)) {
        if (why) *why = fmt("block=%s is not RxC, with * for a whole extent", blk);
        return RAD_E_INVAL;
    }
    if (g->lv2) {
        /* Each second-level block must be a whole number of first-level ones. */
        auto nests = [](int64_t b1, int64_t b2) {
            if (b1 == 0) return b2 == 0;
            return b2 == 0 || b2 % b1 == 0;
        };
        if (!nests(g->br, g->br2) || !nests(g->bc, g->bc2)) {
            if (why) *why = "the second level's block is not a whole number of the first's";
            return RAD_E_INVAL;
        }
    }

    if (const char* t = rad_param_gets(o, n_o, "transform", nullptr)) {
        long long n = 0;
        if (std::strncmp(t, "fwht", 4) != 0 || std::sscanf(t + 4, "%lld", &n) != 1 || n < 2 ||
            (n & (n - 1))) {
            if (why) *why = fmt("transform=%s: the grid applies fwhtN, N a power of two", t);
            return RAD_E_INVAL;
        }
        g->fwht = n;
    }

    const char* rule = rad_param_gets(o, n_o, "rule", g->scale_dt == RAD_E8M0 ? "mx" : "absmax");
    if (!std::strcmp(rule, "absmax")) g->rule = Grid::ABSMAX;
    else if (!std::strcmp(rule, "mx")) g->rule = Grid::MX;
    else if (!std::strcmp(rule, "fixed")) g->rule = Grid::FIXED;
    else if (!std::strcmp(rule, "search")) g->rule = Grid::SEARCH;
    else {
        if (why) *why = fmt("rule=%s is not absmax, mx, fixed or search", rule);
        return RAD_E_INVAL;
    }
    if ((g->rule == Grid::MX) != (g->scale_dt == RAD_E8M0)) {
        if (why) *why = "rule=mx and scale=e8m0 go together: a shared exponent is a power of two";
        return RAD_E_INVAL;
    }
    if (g->rule == Grid::MX && (!g->is_float || g->asym)) {
        if (why) *why = "rule=mx is the OCP shared exponent over a float code";
        return RAD_E_INVAL;
    }
    if (g->rule == Grid::FIXED) {
        g->fixed = rad_param_getf(o, n_o, "scale_value", 0.0);
        if (!(g->fixed > 0.0)) {
            if (why) *why = "rule=fixed needs scale_value= greater than zero";
            return RAD_E_INVAL;
        }
        if (g->asym) {
            if (why) *why = "rule=fixed has no zero point to fix";
            return RAD_E_INVAL;
        }
        if (g->lv2) {
            if (why) *why = "rule=fixed has one scale; scale2= would be a second";
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

RadEncoding grid_encoding(const Grid& g) {
    RadEncoding e = rad_enc_affine(g.codes, g.scale_dt, g.br, g.bc);
    if (g.asym) rad_enc_add_plane(&e, "zero", g.zero_dt, g.br, g.bc);
    if (g.lv2) rad_enc_add_plane(&e, "scale.1", g.scale2_dt, g.br2, g.bc2);
    if (!g.table.empty()) rad_enc_add_table(&e, "table", g.table_dt, 1, (int64_t)g.table.size());
    if (g.fwht) {
        char t[RAD_ENC_STR];
        std::snprintf(t, sizeof t, "fwht%lld", (long long)g.fwht);
        rad_enc_copy_str(e.transform, t);
    }
    return e;
}

int64_t grid_row_block(const Grid& g) {
    if (g.lv2) return g.br2;                 /* a multiple of the first level's, or 0 */
    if (g.br > 0) return g.br;
    return g.rule == Grid::FIXED ? 1 : 0;
}

BlockScale grid_block_scale(const Grid& g, const float* w, int64_t cols, int64_t r0, int64_t r1,
                            int64_t c0, int64_t c1, const float* imp) {
    BlockScale b;
    const float resid = g.fwht ? 1.0f / (float)g.fwht : 1.0f;

    if (g.rule == Grid::FIXED) {
        /* The reciprocal is of the unrounded value, in double; the stored scale is the rounded
         * value. This is the n-gram table's rule, whose scale is exact in bf16 so the two agree;
         * an inexact value is quantised against what was asked, not its rounding. */
        b.s = round_to(g.scale_dt, (float)g.fixed);
        b.inv_d = 1.0 / g.fixed;
        b.inv = (float)b.inv_d;
        b.stored = round_to(g.scale_dt, b.s * resid);
        return b;
    }

    if (g.general()) {
        /* General path: the block's ideal scale for its code (signed for a codebook with unequal
         * ends), searched under rule=search, then rounded to the scale dtype, never to zero.
         * Under a second level it is returned unrounded and grid_scales rounds it relative to
         * that level. */
        if (g.asym) {
            float lo = 0.0f, hi = 0.0f;
            for (int64_t r = r0; r < r1; ++r)
                for (int64_t c = c0; c < c1; ++c) {
                    const float v = w[r * cols + c];
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
            if (g.rule == Grid::SEARCH && hi > lo)
                search_asym(g, w, cols, r0, r1, c0, c1, imp, &lo, &hi);
            float s = (hi - lo) / (float)g.qmax;
            if (!(s > 0.0f)) s = 1e-12f;
            b.lo = lo;
            b.s = g.lv2 ? s : round_nz(g.scale_dt, s);
            b.inv = 1.0f / b.s;
            b.inv_d = 1.0 / (double)b.s;
            b.zero = clampi((int)(-lo / b.s + 0.5f), 0, g.qmax);
            if (g.lv2) b.stored = b.s;
            else store_first_level(g, resid, &b);
            return b;
        }
        float amax = 0.0f, peak = 0.0f;
        for (int64_t r = r0; r < r1; ++r)
            for (int64_t c = c0; c < c1; ++c) {
                const float v = w[r * cols + c];
                const float a = std::fabs(v);
                if (a > amax) { amax = a; peak = v; }
            }
        double s0;
        if (!g.table.empty()) s0 = g.tsigned ? (double)peak / g.tpeak : (double)amax / std::fabs(g.tpeak);
        else if (g.is_float) s0 = (double)amax / g.fmax;
        else s0 = (double)amax / (double)g.qmax;
        if (g.rule == Grid::SEARCH) s0 = search_sym(g, w, cols, r0, r1, c0, c1, imp, s0);
        if (s0 == 0.0) s0 = g.lv2 ? 0.0 : 1e-12;
        b.s = g.lv2 ? (float)s0 : round_nz(g.scale_dt, (float)s0);
        b.inv = b.s != 0.0f ? 1.0f / b.s : 0.0f;
        b.inv_d = b.s != 0.0f ? 1.0 / (double)b.s : 0.0;
        if (g.lv2) b.stored = b.s;
        else store_first_level(g, resid, &b);
        return b;
    }

    if (g.asym) {
        /* [lo, hi] is widened to include 0 so an all-positive block still has an exact zero. The
         * scale spans it in (levels - 1) steps and is rounded to its dtype; the integer zero is
         * chosen against the rounded scale, so (code - zero) * scale decodes to the grid point the
         * code was chosen as. */
        float lo = 0.0f, hi = 0.0f;
        bool first = true;
        for (int64_t r = r0; r < r1; ++r)
            for (int64_t c = c0; c < c1; ++c) {
                const float v = w[r * cols + c];
                if (first || v < lo) lo = v;
                if (first || v > hi) hi = v;
                first = false;
            }
        if (lo > 0.0f) lo = 0.0f;
        if (hi < 0.0f) hi = 0.0f;
        float s = (hi - lo) / (float)(g.qmax);
        if (!(s > 0.0f)) s = 1e-12f;
        b.lo = lo;
        b.s = round_to(g.scale_dt, s);
        b.inv = 1.0f / b.s;
        b.zero = clampi((int)(-lo / b.s + 0.5f), 0, g.qmax);
        store_first_level(g, resid, &b);
        return b;
    }

    float amax = 0.0f;
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t c = c0; c < c1; ++c) {
            const float v = w[r * cols + c];
            const float a = v < 0.0f ? -v : v;
            if (a > amax) amax = a;
        }

    if (g.rule == Grid::MX) {
        /* The OCP rule, saturation included: X = 2^(floor(log2 amax) - emax), so amax / X lands in
         * [2^emax, 2^(emax+1)) and the top of that range saturates to the code's largest value.
         * Bumping the exponent instead would double the step for every other element. */
        int exp = -127;
        if (amax > 0.0f) {
            uint32_t u;
            std::memcpy(&u, &amax, 4);
            exp = (int)((u >> 23) & 0xFFu) - 127 - g.emax;
        }
        b.e8 = clampi(exp + 127, 0, 254);
        /* 2^-(e8 - 127) built in the exponent field: exact, no libm call. */
        const uint32_t ui = (uint32_t)clampi(254 - b.e8, 1, 254) << 23;
        std::memcpy(&b.inv, &ui, 4);
        b.inv_d = b.inv;
        b.s = rad_e8m0_to_f32((uint8_t)b.e8);
        b.stored = b.s;
        if (g.fwht) {
            /* The exponent stored is the transform's order lower, exactly; codes are chosen
             * against N times it, which differs only where the exponent's floor clamps it. */
            int sh = 0;
            while (((int64_t)1 << sh) < g.fwht) ++sh;
            b.e8 = clampi(b.e8 - sh, 0, 254);
            b.stored = rad_e8m0_to_f32((uint8_t)b.e8);
            b.s = b.stored * (float)g.fwht;
            b.inv = 1.0f / b.s;
            b.inv_d = b.inv;
        }
        return b;
    }

    if (g.is_float) {
        /* An all-zero block gets scale 1, so its reciprocal is finite and its codes zero. The
         * quotient multiplies by 1/fmax for a scale narrower than f32 and divides for f32: the
         * forms the block-fp8 linears and the fp8 hyper-connection rows were quantised with. They
         * differ in the last f32 place and can round a narrow scale differently. */
        float s = 1.0f;
        if (amax > 0.0f) s = g.scale_dt == RAD_F32 ? amax / g.fmax : amax * (1.0f / g.fmax);
        b.s = round_to(g.scale_dt, s);
        b.inv = 1.0f / b.s;
    } else {
        /* An all-zero block gets 1e-12: finite reciprocal, zero codes. */
        float s = amax / (float)g.qmax;
        if (!(s > 0.0f)) s = 1e-12f;
        b.s = round_to(g.scale_dt, s);
        b.inv = 1.0f / b.s;
    }
    b.inv_d = b.inv;
    store_first_level(g, resid, &b);
    return b;
}

int grid_code(const Grid& g, const BlockScale& b, float v) {
    if (g.rule == Grid::FIXED) {
        const float x = (float)((double)v * b.inv_d);
        if (!g.table.empty()) return nearest_entry(g, x);
        if (g.wide) return code_at(g, (double)v * b.inv_d, 1.0, 0);
        if (g.codes == RAD_F8E4M3) return rad_f32_to_fp8e4m3(x);
        if (g.is_float) return g.codes == RAD_FP4E2M1 ? f32_to_e2m1(x)
                                                      : (int)f32_to_small_float(x, g.codes);
        return clampi(round_away(x), g.qmin, g.qmax);
    }
    if (g.general()) {
        if (!g.table.empty()) return nearest_entry(g, v * b.inv);
        if (g.wide) return code_at(g, (double)v * b.inv_d, 1.0, g.asym ? b.zero : 0);
    }
    const float x = v * b.inv;
    if (g.is_float) {
        if (g.codes == RAD_F8E4M3) return f32_to_e4m3(x);
        if (g.codes == RAD_FP4E2M1) return f32_to_e2m1(x);
        return (int)f32_to_small_float(x, g.codes);
    }
    if (g.asym) return clampi(round_away(x) + b.zero, g.qmin, g.qmax);
    return clampi(round_away(x), g.qmin, g.qmax);
}

float grid_decode(const Grid& g, const BlockScale& b, int code) {
    if (!g.table.empty()) return g.table[(size_t)code] * b.s;
    if (g.is_float) {
        uint8_t buf[4] = { (uint8_t)code, 0, 0, 0 };
        return rad_load_f32(buf, g.codes, 0) * b.s;
    }
    if (g.wide) return (float)((double)(code - (g.asym ? b.zero : 0)) * (double)b.s);
    if (g.asym) return (float)(code - b.zero) * b.s;
    return (float)code * b.s;
}

void grid_store_scale(const Grid& g, const BlockScale& b, void* scale_row, void* zero_row,
                      int64_t i) {
    if (g.l1_int) {
        rad_store_code(scale_row, g.scale_dt, i, (int)b.stored);
    } else {
        switch (g.scale_dt) {
            case RAD_F32:    ((float*)scale_row)[i] = b.stored; break;
            case RAD_BF16:   ((uint16_t*)scale_row)[i] = f32_to_bf16(b.stored); break;
            case RAD_F16:    ((uint16_t*)scale_row)[i] = f32_to_f16(b.stored); break;
            case RAD_F8E4M3: ((uint8_t*)scale_row)[i] = f32_to_e4m3(b.stored); break;
            case RAD_E8M0:   ((uint8_t*)scale_row)[i] = (uint8_t)b.e8; break;
            default: break;
        }
    }
    if (g.asym && zero_row) rad_store_code(zero_row, g.zero_dt, i, b.zero);
}

Planes grid_planes(const Grid& g, int64_t cols, void* const* planes) {
    Planes p;
    const int64_t sc = g.bc > 0 ? (cols + g.bc - 1) / g.bc : 1;
    p.codes = (uint8_t*)planes[0];
    p.scale = (uint8_t*)planes[1];
    p.zero  = g.asym ? (uint8_t*)planes[2] : nullptr;
    p.codes_row  = rad_dtype_bytes(g.codes, cols);
    p.scale_row  = rad_dtype_bytes(g.scale_dt, sc);
    p.zero_row   = g.asym ? rad_dtype_bytes(g.zero_dt, sc) : 0;
    p.scale_cols = sc;
    if (g.lv2) {
        p.scale2 = (uint8_t*)planes[g.plane_scale2()];
        p.scale2_cols = g.bc2 > 0 ? (cols + g.bc2 - 1) / g.bc2 : 1;
        p.scale2_row = rad_dtype_bytes(g.scale2_dt, p.scale2_cols);
    }
    if (!g.table.empty()) p.table = (uint8_t*)planes[g.plane_table()];
    return p;
}

void grid_transform_rows(const Grid& g, float* w, int64_t rows, int64_t cols) {
    if (!g.fwht) return;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; c += g.fwht) fwht_inplace(w + r * cols + c, g.fwht);
}

Scales grid_scales(const Grid& g, const float* w, int64_t rows, int64_t cols, const float* imp) {
    Scales s;
    s.br = g.br > 0 ? g.br : rows;
    s.bc = g.bc > 0 ? g.bc : cols;
    s.nr = (rows + s.br - 1) / s.br;
    s.nc = (cols + s.bc - 1) / s.bc;
    s.b.resize((size_t)(s.nr * s.nc));
    const float* im = g.fwht ? nullptr : imp;   /* importances name columns before a rotation */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (s.nr * s.nc > 64)
#endif
    for (int64_t k = 0; k < s.nr * s.nc; ++k) {
        const int64_t i = k / s.nc, j = k % s.nc;
        const int64_t r0 = i * s.br, c0 = j * s.bc;
        s.b[(size_t)k] = grid_block_scale(g, w, cols, r0, std::min(r0 + s.br, rows), c0,
                                          std::min(c0 + s.bc, cols), im);
    }
    if (!g.lv2) return s;

    /* Second level: each block takes the largest first-level magnitude under it divided by the
     * first level's dtype maximum, so the widest sub-scale sits at the dtype's top. It is rounded
     * to its own dtype (up to a power of two for e8m0, so it never rounds down past the top).
     * Each first-level scale is then stored relative to it: half away from zero for an integer
     * sub-scale, nearest even for a float one, and never zero for a non-empty block (which would
     * decode to nothing). Codes are chosen against the product, as the reader multiplies it. The
     * transform's 1/N goes into the second level. */
    const int64_t br2 = g.br2 > 0 ? g.br2 : rows, bc2 = g.bc2 > 0 ? g.bc2 : cols;
    s.nr2 = (rows + br2 - 1) / br2;
    s.nc2 = (cols + bc2 - 1) / bc2;
    std::vector<float> peak((size_t)(s.nr2 * s.nc2), 0.0f), s2r(peak.size(), 1.0f);
    for (int64_t i = 0; i < s.nr; ++i)
        for (int64_t j = 0; j < s.nc; ++j) {
            const size_t k2 = (size_t)(((i * s.br) / br2) * s.nc2 + (j * s.bc) / bc2);
            peak[k2] = std::max(peak[k2], std::fabs(s.b[(size_t)(i * s.nc + j)].s));
        }
    const float resid = g.fwht ? 1.0f / (float)g.fwht : 1.0f;
    s.s2.resize(peak.size());
    for (size_t k = 0; k < peak.size(); ++k) {
        if (g.fixed2 > 0.0) {
            /* Fixed: store the requested value; the first level is relative to it divided by the
             * transform's residual, which the reader's product undoes. */
            s.s2[k] = (float)g.fixed2;
            s2r[k] = (float)(g.fixed2 / (double)resid);
            continue;
        }
        float v = peak[k] / g.l1max;
        if (!(v > 0.0f) || !std::isfinite(v)) v = 1.0f;
        if (g.scale2_dt == RAD_E8M0) {
            int e = 0;
            const float m = std::frexp(v, &e);           /* v = m 2^e, m in [0.5, 1) */
            v = std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
            v = std::min(std::max(v, 0x1p-127f), 0x1p127f);
            s2r[k] = v;
            s.s2[k] = std::min(std::max(v * resid, 0x1p-127f), 0x1p127f);
        } else {
            s2r[k] = round_nz(g.scale2_dt, v);
            s.s2[k] = round_nz(g.scale2_dt, s2r[k] * resid);
        }
        /* As for a first level alone (store_first_level): the first level is taken relative to
         * what the reader multiplies by, which is N times the stored second level. */
        s2r[k] = s.s2[k] / resid;
    }
    /* A searched float first level is searched again against the now-known second level, over
     * candidates as stored: rounding moves an E4M3 sub-scale by up to 1/16, most of what the first
     * search gained. The second level stays where the first search's peaks put it. */
    if (g.rule == Grid::SEARCH && !g.l1_int && !g.asym) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (s.nr * s.nc > 64)
#endif
        for (int64_t k = 0; k < s.nr * s.nc; ++k) {
            const int64_t i = k / s.nc, j = k % s.nc;
            BlockScale& b = s.b[(size_t)k];
            if (b.s == 0.0f) continue;
            const double t = s2r[(size_t)(((i * s.br) / br2) * s.nc2 + (j * s.bc) / bc2)];
            const int64_t r0 = i * s.br, c0 = j * s.bc;
            b.s = (float)search_sym(g, w, cols, r0, std::min(r0 + s.br, rows), c0,
                                    std::min(c0 + s.bc, cols), im, (double)b.s, t);
        }
    }
    int64_t held = 0;
    for (int64_t i = 0; i < s.nr; ++i)
        for (int64_t j = 0; j < s.nc; ++j) {
            BlockScale& b = s.b[(size_t)(i * s.nc + j)];
            const float t = s2r[(size_t)(((i * s.br) / br2) * s.nc2 + (j * s.bc) / bc2)];
            held += std::fabs(b.s / t) > g.l1max;
            const float q = std::min(std::max(b.s / t, -g.l1max), g.l1max);
            float l1;
            if (g.l1_int) {
                int c = clampi(round_away(q), g.l1_lo, g.l1_hi);
                if (c == 0 && b.s != 0.0f) c = clampi(q < 0.0f ? -1 : 1, g.l1_lo, g.l1_hi);
                l1 = (float)c;
            } else {
                l1 = b.s != 0.0f ? round_nz(g.scale_dt, q) : 0.0f;
            }
            b.stored = l1;
            b.s = l1 * t;
            if (b.s == 0.0f) b.s = t;                    /* a block of zeros: any finite scale */
            b.inv = 1.0f / b.s;
            b.inv_d = 1.0 / (double)b.s;
            if (g.asym) b.zero = clampi((int)(-b.lo / b.s + 0.5f), 0, g.qmax);
        }
    /* Only a fixed second level can push a first-level scale past its dtype's top; a derived one
     * maps the largest under it exactly to the top. */
    if (held)
        std::fprintf(stderr, "libquant: %lld of %lld scales sat past %s's top under scale2_value=%g "
                             "and were held there\n", (long long)held, (long long)(s.nr * s.nc),
                     rad_dtype_name(g.scale_dt), g.fixed2);
    return s;
}

void grid_store_code(const Grid& g, uint8_t* row, int64_t i, int code) {
    if (g.codes == RAD_I32) { ((int32_t*)row)[i] = code; return; }
    rad_store_code(row, g.codes, i, code);
}

void grid_store_scales(const Grid& g, const Scales& s, const Planes& p) {
    for (int64_t i = 0; i < s.nr; ++i)
        for (int64_t j = 0; j < s.nc; ++j)
            grid_store_scale(g, s.b[(size_t)(i * s.nc + j)], p.scale + i * p.scale_row,
                             p.zero ? p.zero + i * p.zero_row : nullptr, j);
    if (g.lv2)
        for (int64_t i = 0; i < s.nr2; ++i)
            for (int64_t j = 0; j < s.nc2; ++j) {
                const float v = s.s2[(size_t)(i * s.nc2 + j)];
                uint8_t* row = p.scale2 + i * p.scale2_row;
                if (g.scale2_dt == RAD_E8M0) {
                    uint32_t u;
                    std::memcpy(&u, &v, 4);
                    row[j] = (uint8_t)((u >> 23) & 0xFFu);
                } else {
                    rad_store_f32(row, g.scale2_dt, j, v);
                }
            }
    if (p.table)
        for (size_t k = 0; k < g.table.size(); ++k) {
            if (g.table_dt == RAD_I8) ((int8_t*)p.table)[k] = (int8_t)g.table[k];
            else ((float*)p.table)[k] = g.table[k];
        }
}

void grid_fake(const Grid& g, const float* w, int64_t rows, int64_t cols, float* out,
               const float* imp) {
    const Scales sc = grid_scales(g, w, rows, cols, imp);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
            const BlockScale& b = sc.at(r, c);
            out[r * cols + c] = grid_decode(g, b, grid_code(g, b, w[r * cols + c]));
        }
}

BlockScale grid_finish(const Grid& g, double s, float lo) {
    BlockScale b;
    const float resid = g.fwht ? 1.0f / (float)g.fwht : 1.0f;
    b.lo = lo;
    b.s = round_nz(g.scale_dt, (float)(s != 0.0 ? s : 1e-12));
    b.inv = 1.0f / b.s;
    b.inv_d = 1.0 / (double)b.s;
    if (g.asym) b.zero = clampi((int)(-lo / b.s + 0.5f), 0, g.qmax);
    store_first_level(g, resid, &b);
    return b;
}

int grid_rtn(const Grid& g, const float* w, int64_t row0, int64_t rows, int64_t cols,
             const Planes& p, const float* imp) {
    if (g.general()) {
        const int64_t rb = grid_row_block(g);
        if (rb > 0 && row0 % rb) return RAD_E_SHAPE;
        const Scales sc = grid_scales(g, w, rows, cols, imp);
        grid_store_scales(g, sc, p);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < rows; ++r) {
            uint8_t* cr = p.codes + r * p.codes_row;
            for (int64_t c = 0; c < cols; ++c)
                grid_store_code(g, cr, c, grid_code(g, sc.at(r, c), w[r * cols + c]));
        }
        return RAD_OK;
    }

    const int64_t br = g.br > 0 ? g.br : (g.rule == Grid::FIXED ? 1 : rows);
    const int64_t bc = g.bc > 0 ? g.bc : cols;
    if (g.br > 0 && row0 % g.br) return RAD_E_SHAPE;
    const int64_t nband = (rows + br - 1) / br;
    /* A fixed scale over the whole extent is one element, written once here rather than by every
     * band concurrently. */
    const bool one_scale = g.br == 0;
    const bool int4 = g.codes == RAD_I4 && !g.is_float && !g.asym && g.rule != Grid::FIXED &&
                      g.rule != Grid::MX;
    if (one_scale && g.rule == Grid::FIXED) {
        const BlockScale b = grid_block_scale(g, w, cols, 0, 0, 0, 0);
        grid_store_scale(g, b, p.scale, p.zero, 0);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t band = 0; band < nband; ++band) {
        const int64_t r0 = band * br, r1 = r0 + br < rows ? r0 + br : rows;
        /* The scale plane's row for this band, relative to the call's first. */
        const int64_t srow = g.br > 0 ? band : 0;
        uint8_t* sr = p.scale + srow * p.scale_row;
        uint8_t* zr = p.zero ? p.zero + srow * p.zero_row : nullptr;
        for (int64_t c0 = 0, j = 0; c0 < cols; c0 += bc, ++j) {
            const int64_t c1 = c0 + bc < cols ? c0 + bc : cols;
            const BlockScale b = grid_block_scale(g, w, cols, r0, r1, c0, c1);
            if (!(one_scale && g.rule == Grid::FIXED)) grid_store_scale(g, b, sr, zr, j);
            /* Fast path for symmetric int4 over an even span (every routed expert of a w4 model):
             * grid_code's arithmetic without per-element branches, two codes written as one whole
             * byte instead of a nibble read-modify-write. */
            if (int4 && !(c0 & 1) && !((c1 - c0) & 1)) {
                for (int64_t r = r0; r < r1; ++r) {
                    uint8_t* cr = p.codes + r * p.codes_row;
                    const float* wr = w + r * cols;
                    for (int64_t c = c0; c < c1; c += 2) {
                        const int lo = clampi(round_away(wr[c] * b.inv), g.qmin, g.qmax);
                        const int hi = clampi(round_away(wr[c + 1] * b.inv), g.qmin, g.qmax);
                        cr[c >> 1] = (uint8_t)((lo & 15) | ((hi & 15) << 4));
                    }
                }
                continue;
            }
            for (int64_t r = r0; r < r1; ++r) {
                uint8_t* cr = p.codes + r * p.codes_row;
                for (int64_t c = c0; c < c1; ++c)
                    rad_store_code(cr, g.codes, c, grid_code(g, b, w[r * cols + c]));
            }
        }
    }
    return RAD_OK;
}

}  /* namespace lq */
