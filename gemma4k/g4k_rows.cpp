/* g4k_rows.cpp -- gemma4k's row table, the op schemas it defines, and the host arms of those ops.
 *
 * gemma4k is the kernel library for Gemma 4's text model beyond what libr4d serves it with:
 *
 *   attn_paged at head_dim 256 and 512, gqa 1/2/4/8, ANY window, causal or not, bf16 or fp8 cache.
 *     Gemma 4's sliding layers are h256/gqa2 with a 1024-token window and its full layers h512 at
 *     gqa 8 (TP1/TP2), 4 (TP4) or 2 (TP8).
 *   pair_argmax: the top-1 of a row of all-gathered (id, value) pairs, one i32 a row -- the
 *     vocab-sharded DFlash head's merge, which needs R >= 2 on the wire (16-byte gather rows) and a
 *     contiguous proposal out (g4k_pick.hip says why). libref declares it too, and is its oracle.
 *   norm_add_norm and join_add_scale, ops this library DEFINES: Gemma 4's two sandwich-norm joins,
 *     each one kernel in place of a chain of rmsnorm / add / mul_weight launches, rounding where the
 *     chain rounds (g4k_norm.hip). The arch plugin asks for them and falls back to the chain.
 *
 * Put it AHEAD of libr4d for Gemma 4 (`--kernels gemma4k,libr4d,...`). The hierarchy is absolute,
 * and libr4d has rows at h256/gqa2 and h512/gqa8 too -- written for the dense 31B, which has no
 * drafter, and not measured on the 26B-A4B or under a DFlash verify. These rows are the ones every
 * Gemma 4 number in this tree was measured with. Every other row here is the only one in the tree
 * for its geometry, so the order changes nothing else: libr4d keeps the drafter's h128/gqa4
 * attention, say. (Gemma's final logit softcap is libr4d's and libavx's `logit_softcap`; this
 * library has no row for it.)
 *
 * Pure host code: no HIP header. The entry points it names are in g4k_attn.hip, g4k_pick.hip and
 * g4k_norm.hip.
 */
#include "g4k.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifndef G4K_BUILD_TARGET
#define G4K_BUILD_TARGET "gfx12"
#endif

#define ROW(a) (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NOTUNE nullptr, 0, nullptr
#define C_EQ(k, v) { (k), RAD_C_EQ, (long long)(v), nullptr }
#define C_GE(k, v) { (k), RAD_C_GE, (long long)(v), nullptr }
#define C_LE(k, v) { (k), RAD_C_LE, (long long)(v), nullptr }
#define C_IN(k, s) { (k), RAD_C_IN, 0, (s) }

namespace {

/* ==================================================================== the ops this library declares
 *
 * pair_argmax is libref's schema restated field for field (the first declarer in a hierarchy fixes
 * it, and libref is in every hierarchy). The two joins are this library's own: libref has no row
 * for them ON PURPOSE, because a reference row would resolve them wherever gemma4k is not loaded,
 * and the architecture's fallback to the rmsnorm / add / mul_weight chain -- which libr4d and
 * libavx serve -- would never be taken. docs/OPS.md carries all three rows, which is what the
 * ops_doc test holds them to. */
static const RadParamSpec pPick[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "n",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "dtype", RAD_P_STR, RAD_REQUIRED, 0 },
};
static const RadOperandSpec oPick[] = {
    { "pairs", RAD_OPD_IN,  0 },
    { "idx",   RAD_OPD_OUT, 0 },
    { "val",   RAD_OPD_OUT, 1 },
};
static const RadParamSpec pJoin[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "n",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "eps",   RAD_P_F64, RAD_REQUIRED, 0 },
    { "dtype", RAD_P_STR, RAD_REQUIRED, 0 },
};
static const RadOperandSpec oNan[] = {
    { "a",   RAD_OPD_IN,     0 },
    { "w_a", RAD_OPD_WEIGHT, 0 },
    { "x",   RAD_OPD_INOUT,  0 },
    { "w_y", RAD_OPD_WEIGHT, 0 },
    { "y",   RAD_OPD_OUT,    0 },
};
static const RadOperandSpec oJas[] = {
    { "d",   RAD_OPD_IN,     0 },
    { "w_d", RAD_OPD_WEIGHT, 0 },
    { "r",   RAD_OPD_IN,     0 },
    { "w_r", RAD_OPD_WEIGHT, 0 },
    { "w_j", RAD_OPD_WEIGHT, 0 },
    { "x",   RAD_OPD_INOUT,  0 },
    { "s",   RAD_OPD_WEIGHT, 0 },
    { "w_n", RAD_OPD_WEIGHT, 1 },
    { "y",   RAD_OPD_OUT,    1 },
};
static const RadParamSpec pEscale[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED, 0 },
    { "n_expert", RAD_P_INT, RAD_REQUIRED, 0 },
    { "top_k",    RAD_P_INT, RAD_REQUIRED, 0 },
    { "dtype",    RAD_P_STR, RAD_REQUIRED, 0 },
};
static const RadOperandSpec oEscale[] = {
    { "ids", RAD_OPD_IN,     0 },
    { "w_e", RAD_OPD_WEIGHT, 0 },
    { "ew",  RAD_OPD_INOUT,  0 },
};
/* attn_paged_qrow is attn_paged's schema restated field for field under a name only this library
 * declares. The attention kernels below take q at any row pitch, so q can be read where a fused
 * [q|k|v] projection left it; libr4d's paged attention reads q only dense and refuses the slice by
 * name (RAD_E_STRIDE), which stops the engine at the first step. Under attn_paged, whichever
 * library comes first in `--kernels` wins the op, so an architecture handing it the slice depends
 * on an order it cannot see. Under its own name it resolves to this library or to nothing, and the
 * architecture falls back to the lift and attn_paged where it is nothing. No libref row, as the
 * joins, for the same reason. */
static const RadParamSpec pAttnQrow[] = {
    { "q_len",      RAD_P_INT, RAD_REQUIRED, 0 },
    { "head_dim",   RAD_P_INT, RAD_REQUIRED, 0 },
    { "gqa",        RAD_P_INT, RAD_REQUIRED, 0 },
    { "block_size", RAD_P_INT, RAD_DERIVED,  0 },
    { "causal",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "window",     RAD_P_INT, RAD_REQUIRED, 0 },
    { "q_dtype",    RAD_P_STR, RAD_REQUIRED, 0 },
    { "kv_dtype",   RAD_P_STR, RAD_REQUIRED, 0 },
    { "scale",      RAD_P_F64, RAD_OPTIONAL, 0 },
    { "max_ctx",    RAD_P_INT, RAD_OPTIONAL, 0 },
    { "n_head",     RAD_P_INT, RAD_OPTIONAL, 0 },
    { "max_seqs",   RAD_P_INT, RAD_OPTIONAL, 0 },
};
static const RadOperandSpec oAttnQrow[] = {
    { "q",           RAD_OPD_IN,  0 },
    { "kv_cache",    RAD_OPD_IN,  0 },
    { "block_table", RAD_OPD_IN,  0 },
    { "seqused",     RAD_OPD_IN,  0 },
    { "k_descale",   RAD_OPD_IN,  1 },
    { "v_descale",   RAD_OPD_IN,  1 },
    { "cu_seqlens",  RAD_OPD_IN,  1 },
    { "out",         RAD_OPD_OUT, 0 },
};
static const RadOpSchema kSchemas[] = {
    { "pair_argmax", ROW(pPick), ROW(oPick),
      "idx[m] = the id of the largest value among pairs[m, 0..n), each pair an int32 id in the first "
      "f32 slot and its value in the second (row_topk's `pairs`, all-gathered): ties to the lower "
      "id, a negative id skipped, -1 and -inf where nothing is valid. row_topk_merge at R = 1, "
      "written one element a row at the output's own pitch." },
    { "norm_add_norm", ROW(pJoin), ROW(oNan),
      "t = rmsnorm(a) * w_a; x = x + t; y = rmsnorm(x) * w_y, over [M, n] rows; every value the "
      "unfused rmsnorm, add, rmsnorm chain stores is rounded to the activation dtype at the same "
      "point. A post-norm, the residual add and the next pre-norm (Gemma's sandwich)." },
    { "join_add_scale", ROW(pJoin), ROW(oJas),
      "x = (x + rmsnorm(rmsnorm(d) * w_d + rmsnorm(r) * w_r) * w_j) * s, over [M, n] rows, s one "
      "element; and with w_n and y, y = rmsnorm(x) * w_n over the x written. Rounding where the "
      "unfused chain rounds. Two arms' post-norms, their sum, its norm, the residual add, a "
      "per-layer scalar and the next layer's input norm (Gemma 4's feed-forward join)." },
    { "expert_scale", ROW(pEscale), ROW(oEscale),
      "ew[t, k] = ew[t, k] * w_e[ids[t, k]] over [M, top_k], w_e an [n_expert] table, an id outside "
      "it reading 0; one product rounded once, so the embed_lookup + mul pair's value bit for bit. "
      "Gemma 4's per_expert_scale on the renormalised routing weights." },
    { "attn_paged_qrow", ROW(pAttnQrow), ROW(oAttnQrow),
      "attn_paged, parameter for parameter and value for value, with q read at its own row pitch: "
      "a column slice of a wider row -- the q part of a fused [q|k|v] projection -- read where it "
      "lies rather than lifted into a dense block first." },
};

/* ==================================================================== operand descriptions
 *
 * So rad-kbench can call these kernels cold and check them against libref (docs/PLUGIN.md §9).
 * attn_paged's is libr4d's (r4d_shapes.cpp) restated: ONE sequence whose context is its query, or
 * `ctx` deep when a sweep sets it. */
RadOpdDesc opd(uint32_t dt, int64_t d0, int64_t d1 = 0, int64_t d2 = 0, int64_t d3 = 0) {
    RadOpdDesc d{};
    d.dtype = dt;
    d.fill = RAD_FILL_NORMAL;
    d.idx_const = -1;
    const int64_t e[4] = { d0, d1, d2, d3 };
    for (int i = 0; i < 4 && e[i] != 0; ++i) d.shape[d.rank++] = e[i];
    return d;
}
RadOpdDesc opd_idx(int64_t d0, int64_t d1, int64_t idx_max, int64_t konst = -1) {
    RadOpdDesc d = opd((uint32_t)RAD_I32, d0, d1);
    d.fill = RAD_FILL_INDEX;
    d.idx_max = idx_max;
    d.idx_const = konst;
    return d;
}
RadOpdDesc opd_none() {
    RadOpdDesc d{};
    d.idx_const = -1;
    d.flags = RAD_OPD_F_ABSENT;
    return d;
}
int pick(RadOpdDesc* out, int operand, const RadOpdDesc* list, int n) {
    if (!out || !list) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (list[i].flags & RAD_OPD_F_ABSENT) continue;
        if (list[i].rank == 0) return RAD_E_UNSUPPORTED;
        for (uint32_t d = 0; d < list[i].rank; ++d)
            if (list[i].shape[d] <= 0) return RAD_E_UNSUPPORTED;
    }
    if (operand < 0 || operand >= n) return RAD_E_SHAPE;
    *out = list[operand];
    return RAD_OK;
}
#define PICK(l) return pick(out, operand, (l), (int)(sizeof(l) / sizeof((l)[0])))
int64_t pi(const RadParam* p, int n_p, const char* k, int64_t dflt = 0) {
    return (int64_t)rad_param_getdim(p, n_p, k, (long long)dflt);
}
uint32_t dt_act(const RadParam* p, int n_p, const char* k) {
    const uint32_t d = rad_param_getdt(p, n_p, k);
    return d == RAD_DT_INVALID ? (uint32_t)RAD_BF16 : d;
}

int shape_attn_paged(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t q_len = pi(p, n_p, "q_len"), hd = pi(p, n_p, "head_dim");
    const int64_t gqa = pi(p, n_p, "gqa"), bs = pi(p, n_p, "block_size", 16), nh = pi(p, n_p, "n_head");
    if (q_len <= 0 || hd <= 0 || gqa <= 0 || bs <= 0 || nh <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
    int64_t ctx = pi(p, n_p, "ctx", 0);
    if (ctx < q_len) ctx = q_len;
    const int64_t nblk = (ctx + bs - 1) / bs;
    const uint32_t Q = dt_act(p, n_p, "q_dtype"), KV = dt_act(p, n_p, "kv_dtype");
    const RadOpdDesc l[] = {
        opd(Q, q_len, nh, hd),
        opd(KV, nblk, nh / gqa, bs, 2 * hd),
        opd_idx(1, nblk, nblk),
        opd_idx(1, 0, 0, ctx),
        opd_none(),
        opd_none(),
        opd_none(),
        opd(Q, q_len, nh, hd),
    };
    PICK(l);
}

/* pair_argmax as libref describes it (ref_shapes.cpp): the gathered plane is f32, so the value
 * lane draws proper floats and the id lane their bit patterns -- negative ones pads, positive ones
 * legal ids -- and ties do not occur. */
int shape_pick(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, n, 2),
        opd((uint32_t)RAD_I32, M),
        opd((uint32_t)RAD_F32, M),
    };
    PICK(l);
}

/* The host arm of pair_argmax, the device kernel's rules in plain C++. Not libref's row: that one
 * is a reference, which the engine refuses to serve, and this op has no chain to fall back to. */
int pick_host(const RadArgs* a, RadStream) {
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

/* ==================================================================== the host arms of the joins
 *
 * The device kernels' arithmetic in plain C++: each value the unfused chain stores is rounded to
 * bf16 where it stores it. Summed in index order, so a ULP from the device's block order. */
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

int norm_add_norm_host(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const int64_t n = rad_args_geti_or(a, "n", 0);
    const double eps = rad_args_getf_or(a, "eps", 1e-6);
    const RadTensor *A_ = rad_arg_in(a, 0), *Wa = rad_arg_in(a, 1), *X = rad_arg_in(a, 2),
                    *Wy = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    if (n <= 0 || !A_ || !X || !Y) return RAD_E_INVAL;
    if (!hgain(Wa, n) || !hgain(Wy, n)) return RAD_E_DTYPE;
    int64_t ra, la, rx, lx, ry, ly;
    if (!hrows(A_, n, &ra, &la) || !hrows(X, n, &rx, &lx) || !hrows(Y, n, &ry, &ly)) return RAD_E_SHAPE;
    if (ra < rx || ry < rx) return RAD_E_SHAPE;
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
    }
    return RAD_OK;
}

int join_add_scale_host(const RadArgs* a, RadStream) {
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

/* The host arm of expert_scale, the device kernel's one product in plain C++. */
int expert_scale_host(const RadArgs* a, RadStream) {
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

/* ==================================================================== constraints
 *
 * The attention rows serve their (head_dim, gqa) at ANY q_len -- one kernel does decode, verify and
 * prefill, splitting the key range when there are too few tiles to fill the card -- so `q_len`
 * carries no constraint and cuts no band. `causal` and `window` are runtime fields of the kernel.
 * block_size is the kernel's (RAD_DERIVED): 16.
 *
 * TWO ROWS HERE OVERLAP libr4d's ON PURPOSE: h256/gqa2 (any window) and h512/gqa8, Gemma 4's own
 * geometries, which this library serves ahead of libr4d (see the top of this file). Nothing else
 * overlaps, so putting this library first cannot take one of libr4d's other geometries: libr4d
 * serves h256 at gqa 4/6/8/12 with window 0, so the h256 gqa4 and gqa8 rows below take a window
 * only (C_GE 1), and gqa 1 and h512 at gqa 1/2/4, which libr4d does not compile, take anything. */
#define G4K_ATTN_C(NAME, HD, G, KV, WMIN)                                                          \
    static const RadConstraint NAME[] = {                                                          \
        C_EQ("head_dim", HD), C_EQ("gqa", G), C_EQ("block_size", 16), C_GE("causal", 0),           \
        C_GE("window", WMIN), C_IN("q_dtype", "bf16"), C_IN("kv_dtype", KV),                       \
    };
G4K_ATTN_C(cH256x1B, 256, 1, "bf16", 0) G4K_ATTN_C(cH256x1F, 256, 1, "fp8_e4m3", 0)
G4K_ATTN_C(cH256x2B, 256, 2, "bf16", 0) G4K_ATTN_C(cH256x2F, 256, 2, "fp8_e4m3", 0)
G4K_ATTN_C(cH256x4B, 256, 4, "bf16", 1) G4K_ATTN_C(cH256x4F, 256, 4, "fp8_e4m3", 1)
G4K_ATTN_C(cH256x8B, 256, 8, "bf16", 1) G4K_ATTN_C(cH256x8F, 256, 8, "fp8_e4m3", 1)
G4K_ATTN_C(cH512x1B, 512, 1, "bf16", 0) G4K_ATTN_C(cH512x1F, 512, 1, "fp8_e4m3", 0)
G4K_ATTN_C(cH512x2B, 512, 2, "bf16", 0) G4K_ATTN_C(cH512x2F, 512, 2, "fp8_e4m3", 0)
G4K_ATTN_C(cH512x4B, 512, 4, "bf16", 0) G4K_ATTN_C(cH512x4F, 512, 4, "fp8_e4m3", 0)
G4K_ATTN_C(cH512x8B, 512, 8, "bf16", 0) G4K_ATTN_C(cH512x8F, 512, 8, "fp8_e4m3", 0)
#undef G4K_ATTN_C

/* the device kernels hold 24 elements a thread over a 256-thread block */
static const RadConstraint cJoin[]    = { C_GE("n", 1), C_LE("n", 6144), C_IN("dtype", "bf16") };
static const RadConstraint cJoinHost[] = { C_GE("n", 1), C_IN("dtype", "bf16") };
static const RadConstraint cPick[]    = { C_GE("n", 1), C_IN("dtype", "f32 fp32") };
static const RadConstraint cEscale[]  = { C_GE("top_k", 1), C_LE("top_k", 64), C_GE("n_expert", 1),
                                          C_IN("dtype", "bf16") };

/* ==================================================================== the rows */
#define G4K_ATTN_ROW(HD, G, KVN, C, KVDOC)                                                         \
    { "attn_paged_h" #HD "_gqa" #G "_" #KVN, "attn_paged", "attn",                                \
      "paged attention, any window, causal or not; split-KV with a merge where tiles are few",     \
      "head_dim " #HD ", " #G " queries per kv head, paged block 16, any q_len, varlen",           \
      "bf16 query, " KVDOC " kv cache, bf16 out, f32 partials",                                     \
      RAD_DOMAIN_DEVICE, 10, ROW(C), NOTUNE,                                                       \
      nullptr, nullptr, g4k_attn_h##HD##_gqa##G##_##KVN, g4k_attn_h##HD##_scratch, nullptr,      \
      nullptr, shape_attn_paged, nullptr, nullptr, nullptr },
#define G4K_ATTN_PAIR(HD, G)                                                                       \
    G4K_ATTN_ROW(HD, G, bf16kv, cH##HD##x##G##B, "bf16")                                          \
    G4K_ATTN_ROW(HD, G, fp8kv, cH##HD##x##G##F, "fp8-e4m3 (per-(seq,head) descales)")
/* attn_paged_qrow: the same kernel and the same constraints. NO opd_shape -- no libref row, so no
 * oracle; g4k_emu's attention cases already take q as a pitched slice of a wider row. */
#define G4K_QROW_ROW(HD, G, KVN, C, KVDOC)                                                         \
    { "attn_paged_qrow_h" #HD "_gqa" #G "_" #KVN, "attn_paged_qrow", "attn",                      \
      "paged attention reading q at its own row pitch, any window, causal or not",                \
      "head_dim " #HD ", " #G " queries per kv head, paged block 16, any q_len, varlen",           \
      "bf16 query (any row pitch), " KVDOC " kv cache, bf16 out, f32 partials",                     \
      RAD_DOMAIN_DEVICE, 10, ROW(C), NOTUNE,                                                       \
      nullptr, nullptr, g4k_attn_h##HD##_gqa##G##_##KVN, g4k_attn_h##HD##_scratch, nullptr,      \
      nullptr, nullptr, nullptr, nullptr, nullptr },
#define G4K_QROW_PAIR(HD, G)                                                                       \
    G4K_QROW_ROW(HD, G, bf16kv, cH##HD##x##G##B, "bf16")                                          \
    G4K_QROW_ROW(HD, G, fp8kv, cH##HD##x##G##F, "fp8-e4m3 (per-(seq,head) descales)")

static const RadKernelInfo kKernels[] = {
    G4K_ATTN_PAIR(256, 1) G4K_ATTN_PAIR(256, 2) G4K_ATTN_PAIR(256, 4) G4K_ATTN_PAIR(256, 8)
    G4K_ATTN_PAIR(512, 1) G4K_ATTN_PAIR(512, 2) G4K_ATTN_PAIR(512, 4) G4K_ATTN_PAIR(512, 8)
    G4K_QROW_PAIR(256, 1) G4K_QROW_PAIR(256, 2) G4K_QROW_PAIR(256, 4) G4K_QROW_PAIR(256, 8)
    G4K_QROW_PAIR(512, 1) G4K_QROW_PAIR(512, 2) G4K_QROW_PAIR(512, 4) G4K_QROW_PAIR(512, 8)

    /* Described as libref describes it, so rad-kbench runs both arms against the oracle. */
    { "pair_argmax_f32", "pair_argmax", "select",
      "the top-1 of a row of (id, value) pairs, ties to the lower id, negative ids skipped",
      "any M and n; one wave a row; the outputs one element a row at their own pitch",
      "f32 pairs (int32 id bits, f32 value), i32 idx, f32 val",
      RAD_DOMAIN_DEVICE, 10, ROW(cPick), NOTUNE,
      nullptr, nullptr, g4k_pair_argmax_f32, nullptr, nullptr, nullptr, shape_pick,
      nullptr, nullptr, nullptr },
    /* NO opd_shape: libref has no row for these (see the schemas), so there is no oracle to run
     * them against. The host arms state the arithmetic, and tests/g4k_emu checks the device
     * kernels against them and against the unfused chain, bit for bit. */
    { "norm_add_norm_bf16", "norm_add_norm", "norm",
      "post-norm, residual add, next pre-norm: one block a row, rounding where the chain rounds",
      "n <= 6144, any M, row pitches honoured", "bf16 activations, f32 gains",
      RAD_DOMAIN_DEVICE, 10, ROW(cJoin), NOTUNE,
      nullptr, nullptr, g4k_norm_add_norm_bf16, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    { "norm_add_norm_bf16_host", "norm_add_norm", "norm",
      "post-norm, residual add, next pre-norm, rounding where the chain rounds", "any [M, n]",
      "bf16 activations, f32 gains",
      RAD_DOMAIN_HOST, 10, ROW(cJoinHost), NOTUNE,
      nullptr, nullptr, norm_add_norm_host, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    { "join_add_scale_bf16", "join_add_scale", "norm",
      "two arms' post-norms, sum, norm, residual add, layer scalar: one block a row",
      "n <= 6144, any M, row pitches honoured", "bf16 activations, f32 gains, bf16 scalar",
      RAD_DOMAIN_DEVICE, 10, ROW(cJoin), NOTUNE,
      nullptr, nullptr, g4k_join_add_scale_bf16, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    { "join_add_scale_bf16_host", "join_add_scale", "norm",
      "two arms' post-norms, sum, norm, residual add, layer scalar", "any [M, n]",
      "bf16 activations, f32 gains, bf16 scalar",
      RAD_DOMAIN_HOST, 10, ROW(cJoinHost), NOTUNE,
      nullptr, nullptr, join_add_scale_host, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    /* NO opd_shape, as the joins: no libref row, so no oracle; g4k_emu holds both arms to the
     * embed_lookup + mul pair. */
    { "expert_scale_bf16", "expert_scale", "moe",
      "routing weight times its expert's scale, in place: one thread an element",
      "any M, top_k <= 64, row pitches honoured", "i32 ids, bf16 table and weights",
      RAD_DOMAIN_DEVICE, 10, ROW(cEscale), NOTUNE,
      nullptr, nullptr, g4k_expert_scale_bf16, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    { "expert_scale_bf16_host", "expert_scale", "moe",
      "routing weight times its expert's scale, in place", "any M, top_k <= 64",
      "i32 ids, bf16 table and weights",
      RAD_DOMAIN_HOST, 10, ROW(cEscale), NOTUNE,
      nullptr, nullptr, expert_scale_host, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr },
    { "pair_argmax_f32_host", "pair_argmax", "select",
      "the top-1 of a row of (id, value) pairs, ties to the lower id, negative ids skipped",
      "any M and n", "f32 pairs (int32 id bits, f32 value), i32 idx, f32 val",
      RAD_DOMAIN_HOST, 10, ROW(cPick), NOTUNE,
      nullptr, nullptr, pick_host, nullptr, nullptr, nullptr, shape_pick,
      nullptr, nullptr, nullptr },
};
#undef G4K_QROW_PAIR
#undef G4K_QROW_ROW
#undef G4K_ATTN_PAIR
#undef G4K_ATTN_ROW

}  /* namespace */

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo i = {
        RAD_PLUGIN_KERNEL, "gemma4k", "0.1.0",
        "Gemma 4 text: paged attention at head_dim 256/512 (gqa 1/2/4/8, any window), the "
        "sharded draft head's pair_argmax, and the sandwich-norm joins",
        G4K_BUILD_TARGET,
    };
    return &i;
}

int rad_kernel_schema_count(void) { return (int)(sizeof(kSchemas) / sizeof(kSchemas[0])); }
const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < rad_kernel_schema_count()) ? &kSchemas[i] : nullptr;
}
int rad_kernel_count(void) { return (int)(sizeof(kKernels) / sizeof(kKernels[0])); }
const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < rad_kernel_count()) ? &kKernels[i] : nullptr;
}

/* Every kernel here reads and writes its operands and its scratch, and nothing else: no state
 * between launches, no allocation of its own (rad_abi.h, rad_kernel_concurrent). */
int rad_kernel_concurrent(int) { return 1; }

}  /* extern "C" */
