/* ref_host.cpp -- ref_host.h's functions: the rules and roundings of r4d_pair_argmax_f32.hip,
 * r4d_norm_join_bf16.hip and r4d_expert_scale_bf16.hip, one element at a time. */
#include "ref_host.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstring>
#include <vector>

/* pair_argmax: the device kernel's rules in plain C++. */
int host_pair_argmax(const RadArgs* a) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor* p = rad_arg_in(a, 0);
    const RadTensor* t = rad_arg_in(a, 1);
    const RadTensor* v = rad_arg_in(a, 2);
    if (!p || !t) return RAD_E_INVAL;
    if (p->dtype != RAD_F32 || t->dtype != RAD_I32 || (v && v->dtype != RAD_F32)) return RAD_E_DTYPE;
    const long long n = rad_args_geti_or(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    if (p->rank == 3) {
        if (p->shape[1] < n || p->shape[2] != 2 || p->stride[2] != 1 || p->stride[1] != 2) return RAD_E_SHAPE;
    } else if (p->rank == 2) {
        if (p->shape[1] < 2 * n || p->stride[1] != 1) return RAD_E_SHAPE;
    } else {
        return RAD_E_SHAPE;
    }
    const int64_t rows = p->shape[0];
    if (t->rank < 1 || t->shape[0] < rows || (v && (v->rank < 1 || v->shape[0] < rows))) return RAD_E_SHAPE;
    const int64_t prs = p->stride[0], trs = t->rank >= 2 ? t->stride[0] : 1;
    const int64_t vrs = v ? (v->rank >= 2 ? v->stride[0] : 1) : 0;
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = (const float*)p->data + r * prs;
        int32_t bid = -1;
        float bv = -INFINITY;
        for (long long j = 0; j < n; ++j) {
            int32_t id;
            std::memcpy(&id, &row[2 * j], 4);
            const float val = row[2 * j + 1];
            if (id < 0) continue;
            if (bid < 0 || val > bv || (val == bv && id < bid)) { bid = id; bv = val; }
        }
        ((int32_t*)t->data)[r * trs] = bid;
        if (v) ((float*)v->data)[r * vrs] = bv;
    }
    return RAD_OK;
}

/* The joins: each value the unfused chain stores is rounded to bf16 where it stores it. Summed in
 * index order, so a ULP from the device's block order. */
static float hb2f(uint16_t h) { const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t hf2b(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float hrbf(float f) { return hb2f(hf2b(f)); }
static float hnorm(const float* v, int64_t n, double eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; ++i) ss += v[i] * v[i];
    return 1.0f / std::sqrt(ss / (float)n + (float)eps);
}
static bool hrows(const RadTensor* t, int64_t n, int64_t* rows, int64_t* ld) {
    if (!t || t->dtype != RAD_BF16 || t->rank < 1 || rad_tensor_lastdim(t) != n) return false;
    if (t->stride[t->rank - 1] != 1) return false;
    *rows = rad_tensor_rows(t);
    *ld = rad_tensor_row_pitch(t);
    return true;
}
static bool hgain(const RadTensor* w, int64_t n) {
    return w && w->dtype == RAD_F32 && rad_tensor_numel(w) == n;
}

/* [rows..., n] of `dt`, dense last axis, for the route form's fp8 codes and f32 scales */
static bool hrows_as(const RadTensor* t, uint32_t dt, int64_t n, int64_t* rows, int64_t* ld) {
    if (!t || t->dtype != dt || t->rank < 1 || rad_tensor_lastdim(t) != n) return false;
    if (t->stride[t->rank - 1] != 1) return false;
    *rows = rad_tensor_rows(t);
    *ld = rad_tensor_row_pitch(t);
    return true;
}

/* norm_add_norm, and with `route` norm_add_norm_route's three more outputs */
static int nan_host(const RadArgs* a, bool route) {
    if (!a || a->n_t < (route ? 10 : 5)) return RAD_E_INVAL;
    const int64_t n = rad_args_geti_or(a, "n", 0);
    const double eps = rad_args_getf_or(a, "eps", 1e-6);
    const float mult = (float)rad_args_getf_or(a, "mult", 1.0);
    const RadTensor *A_ = rad_arg_in(a, 0), *Wa = rad_arg_in(a, 1), *X = rad_arg_in(a, 2),
                    *Wy = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    const RadTensor *Wr = route ? rad_arg_in(a, 5) : nullptr, *R = route ? rad_arg_in(a, 6) : nullptr;
    const RadTensor *We = route ? rad_arg_in(a, 7) : nullptr, *Q = route ? rad_arg_in(a, 8) : nullptr;
    const RadTensor *S = route ? rad_arg_in(a, 9) : nullptr;
    if (n <= 0 || !A_ || !X || !Y || (route && (!R || !Q || !S))) return RAD_E_INVAL;
    if (route && (n % 128 || rad_args_geti_or(a, "group", 0) != 128)) return RAD_E_SHAPE;
    if (!hgain(Wa, n) || !hgain(Wy, n) || (route && (!hgain(Wr, n) || !hgain(We, n))))
        return RAD_E_DTYPE;
    int64_t ra, la, rx, lx, ry, ly, rr = 0, lr = 0, rq = 0, lq = 0, rs = 0, ls = 0;
    if (!hrows(A_, n, &ra, &la) || !hrows(X, n, &rx, &lx) || !hrows(Y, n, &ry, &ly)) return RAD_E_SHAPE;
    if (route && (!hrows(R, n, &rr, &lr) || !hrows_as(Q, RAD_F8E4M3, n, &rq, &lq) ||
                  !hrows_as(S, RAD_F32, n / 128, &rs, &ls)))
        return RAD_E_SHAPE;
    if (ra < rx || ry < rx || (route && (rr < rx || rq < rx || rs < rx))) return RAD_E_SHAPE;
    const float *wa = (const float*)Wa->data, *wy = (const float*)Wy->data;
    std::vector<float> v((size_t)n), xv((size_t)n);
    for (int64_t m = 0; m < rx; ++m) {
        const uint16_t* ar = (const uint16_t*)A_->data + m * la;
        uint16_t* xr = (uint16_t*)X->data + m * lx;
        uint16_t* yr = (uint16_t*)Y->data + m * ly;
        for (int64_t i = 0; i < n; ++i) v[(size_t)i] = hb2f(ar[i]);
        const float sa = hnorm(v.data(), n, eps);
        for (int64_t i = 0; i < n; ++i) {
            const float t = hrbf((v[(size_t)i] * sa) * wa[i]);
            xv[(size_t)i] = hrbf(hb2f(xr[i]) + t);
            xr[i] = hf2b(xv[(size_t)i]);
        }
        const float sx = hnorm(xv.data(), n, eps);
        for (int64_t i = 0; i < n; ++i) yr[i] = hf2b((xv[(size_t)i] * sx) * wy[i]);
        if (!route) continue;
        /* the router's row, and the experts' as rmsnorm_quant_fp8 writes it (v holds it) */
        const float *wr = (const float*)Wr->data, *we = (const float*)We->data;
        uint16_t* rrow = (uint16_t*)R->data + m * lr;
        uint8_t* qrow = (uint8_t*)Q->data + m * lq;
        float* srow = (float*)S->data + m * ls;
        for (int64_t i = 0; i < n; ++i) {
            rrow[i] = hf2b(((xv[(size_t)i] * sx) * wr[i]) * mult);
            v[(size_t)i] = hrbf((xv[(size_t)i] * sx) * we[i]);
        }
        for (int64_t g = 0; g < n / 128; ++g) {
            float am = 0.0f;
            for (int64_t i = g * 128; i < g * 128 + 128; ++i)
                am = std::fmax(am, std::fabs(v[(size_t)i]));
            const float sc = am > 0.0f ? am * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            srow[g] = sc;
            for (int64_t i = g * 128; i < g * 128 + 128; ++i) {
                const float c = std::fmin(std::fmax(v[(size_t)i] * inv, -448.0f), 448.0f);
                qrow[i] = rad_f32_to_fp8e4m3(c);
            }
        }
    }
    return RAD_OK;
}
int host_norm_add_norm(const RadArgs* a) { return nan_host(a, false); }
int host_norm_add_norm_route(const RadArgs* a) { return nan_host(a, true); }

int host_join_add_scale(const RadArgs* a) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *Wn = a->n_t > 7 ? rad_arg_in(a, 7) : nullptr,
                    *Y  = a->n_t > 8 ? rad_arg_in(a, 8) : nullptr;
    if (!Wn != !Y) return RAD_E_INVAL;
    const int64_t n = rad_args_geti_or(a, "n", 0);
    const double eps = rad_args_getf_or(a, "eps", 1e-6);
    const RadTensor *D = rad_arg_in(a, 0), *Wd = rad_arg_in(a, 1), *R = rad_arg_in(a, 2),
                    *Wr = rad_arg_in(a, 3), *Wj = rad_arg_in(a, 4), *X = rad_arg_in(a, 5),
                    *Sc = rad_arg_in(a, 6);
    if (n <= 0 || !D || !R || !X || !Sc) return RAD_E_INVAL;
    if (!hgain(Wd, n) || !hgain(Wr, n) || !hgain(Wj, n)) return RAD_E_DTYPE;
    if (Sc->dtype != RAD_BF16 || rad_tensor_numel(Sc) != 1) return RAD_E_DTYPE;
    if (Wn && !hgain(Wn, n)) return RAD_E_DTYPE;
    int64_t rd, ld, rr, lr, rx, lx, ry = 0, ly = 0;
    if (!hrows(D, n, &rd, &ld) || !hrows(R, n, &rr, &lr) || !hrows(X, n, &rx, &lx)) return RAD_E_SHAPE;
    if (Y && !hrows(Y, n, &ry, &ly)) return RAD_E_SHAPE;
    if (rd < rx || rr < rx || (Y && ry < rx)) return RAD_E_SHAPE;
    const float *wd = (const float*)Wd->data, *wr = (const float*)Wr->data, *wj = (const float*)Wj->data;
    const float sc = hb2f(*(const uint16_t*)Sc->data);
    std::vector<float> dv((size_t)n), rv((size_t)n);
    for (int64_t m = 0; m < rx; ++m) {
        const uint16_t* dr = (const uint16_t*)D->data + m * ld;
        const uint16_t* rrw = (const uint16_t*)R->data + m * lr;
        uint16_t* xr = (uint16_t*)X->data + m * lx;
        for (int64_t i = 0; i < n; ++i) { dv[(size_t)i] = hb2f(dr[i]); rv[(size_t)i] = hb2f(rrw[i]); }
        const float sd = hnorm(dv.data(), n, eps), sr = hnorm(rv.data(), n, eps);
        for (int64_t i = 0; i < n; ++i)
            dv[(size_t)i] = hrbf(hrbf((dv[(size_t)i] * sd) * wd[i]) + hrbf((rv[(size_t)i] * sr) * wr[i]));
        const float sj = hnorm(dv.data(), n, eps);
        for (int64_t i = 0; i < n; ++i) {
            const float x1 = hrbf(hb2f(xr[i]) + hrbf((dv[(size_t)i] * sj) * wj[i]));
            xr[i] = hf2b(x1 * sc);
            rv[(size_t)i] = hb2f(xr[i]);
        }
        if (Y) {
            const float* wn = (const float*)Wn->data;
            uint16_t* yr = (uint16_t*)Y->data + m * ly;
            const float sn = hnorm(rv.data(), n, eps);
            for (int64_t i = 0; i < n; ++i) yr[i] = hf2b((rv[(size_t)i] * sn) * wn[i]);
        }
    }
    return RAD_OK;
}

/* expert_scale: the device kernel's one product. */
int host_expert_scale(const RadArgs* a) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const int64_t k = rad_args_geti_or(a, "top_k", 0), ne = rad_args_geti_or(a, "n_expert", 0);
    if (k <= 0 || ne <= 0) return RAD_E_SHAPE;
    const RadTensor *I = rad_arg_in(a, 0), *S = rad_arg_in(a, 1), *W = rad_arg_in(a, 2);
    if (!I || !S || !W) return RAD_E_INVAL;
    if (I->dtype != RAD_I32 || S->dtype != RAD_BF16 || W->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (rad_tensor_numel(S) < ne || rad_tensor_lastdim(I) != k || rad_tensor_lastdim(W) != k)
        return RAD_E_SHAPE;
    if (I->stride[I->rank - 1] != 1 || W->stride[W->rank - 1] != 1) return RAD_E_STRIDE;
    const int64_t rows = rad_tensor_rows(W);
    if (rad_tensor_rows(I) < rows) return RAD_E_SHAPE;
    const int64_t li = rad_tensor_row_pitch(I), lw = rad_tensor_row_pitch(W);
    const uint16_t* sc = (const uint16_t*)S->data;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < k; ++c) {
            const int32_t id = ((const int32_t*)I->data)[r * li + c];
            const float s = (id >= 0 && id < ne) ? hb2f(sc[id]) : 0.0f;
            uint16_t* w = (uint16_t*)W->data + r * lw + c;
            *w = hf2b(hb2f(*w) * s);
        }
    return RAD_OK;
}
