/* lora_cases.h -- the runtime-LoRA case matrix and what every implementation is held to, shared by
 * tests/lora_ops_test.cpp (libref and libavx, at every ISA level the machine runs) and
 * tests/r4d_emu/test_lora.cpp (libr4d's source on the SIMT emulator, beside libref and the host
 * restatement in ref_host.cpp). One matrix and one checker, so every implementation runs the same
 * cases.
 *
 * The matrix: M in {1, 3, 17, 64, 130}, R in {16, 32, 48}, S = 4, K (and N) in {64, 256, 1000};
 * `a` bf16, or E4M3 with an f32 scale every 128 columns (1000 leaves a partial group, 64 a row
 * narrower than one group); `y` bf16 and f32. Then the other dtypes once over a smaller grid: `a`
 * f16, f32, E5M2 and i8 (the last two with scales), `y` f16. `a`, its scale stream and `y` are
 * column slices of wider buffers. Slots cycle through -1, -5, 0..3 and 4 (= S): three spellings of
 * a base row beside the adapter rows, M = 1 once as an adapter row and twice as a base one.
 *
 * Checked per implementation, per case:
 *   - Base rows are untouched, by bytes. Shrink leaves the sentinel NaN pattern `t` was filled
 *     with; expand leaves `y` (every third element a -0, which `y + 0` would turn into +0) and
 *     every column outside the slice as they were.
 *   - Adapter rows against a double-precision sum of the same f32 terms (x(m, k) is the code times
 *     its scale rounded to f32, as the contract defines it). Shrink: |t - sum| <= (K + 2) 2^-24
 *     sum|terms|, the worst case of an f32 dot in any summation order, so no implementation can
 *     fail it by choosing an order and every one fails it by reading a wrong element. Expand:
 *     |y - want| <= h (|want| + e) + 2 e with e = 4 (R + 2) 2^-24 (|y| + |scale| sum|terms|) and h
 *     half an ulp of y's dtype relative (2^-8 bf16, 2^-11 f16, 0 f32).
 *   - An implementation marked `ref_bytes` must also produce, for expand, exactly the bytes of the
 *     first implementation in the list, the claim libr4d's expand makes about libref's order.
 *
 * A shrink row with a scratch hook (libr4d's WMMA row keeps its K split's partials there) is handed
 * the scratch the hook asks for, filled with a non-zero pattern, as the core would hand it a slice of
 * its arena.
 */
#pragma once

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace lora_cases {

typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);

struct Impl {
    const char* name;
    LaunchFn    shrink, expand;
    bool        ref_bytes;          /* expand must match the first implementation bit for bit */
    ScratchFn   shrink_scratch;     /* the row's scratch hook, or null: none */
};

/* The scratch a launch is handed, sized by the row's own hook from the call's parameters -- what
 * the core does at declare with the band's M. Host memory: every implementation here runs on the
 * host. */
static inline void with_scratch(RadArgs& a, ScratchFn fn, std::vector<uint8_t>& keep) {
    a.scratch = nullptr;
    a.scratch_bytes = 0;
    const int64_t need = fn ? fn(&a) : 0;
    if (need <= 0) return;
    keep.assign((size_t)need, 0xA5);               /* not zeros: a split must write its partials */
    a.scratch = keep.data();
    a.scratch_bytes = need;
}

struct Tally {
    int cases = 0, fails = 0;
    double worst_shrink = 0, worst_expand = 0;     /* the largest |err| / tolerance seen */
};

static const int kS = 4;
static const int32_t kSlots[] = { -1, 0, 1, 2, 3, 4, 2, -5, 1 };
static inline int32_t slot_of(int m, int off) { return kSlots[(m * 7 + off) % 9]; }
static inline bool base(int32_t s) { return s < 0 || s >= kS; }
static const uint32_t kSentinel = 0x7fc0deadu;     /* a quiet NaN with a payload: "unwritten" */

static inline size_t esize(uint32_t dt) { return (size_t)rad_dtype_bits(dt) / 8; }

static inline void put(uint32_t dt, void* p, size_t i, float v) {
    switch (dt) {
        case RAD_BF16:   ((uint16_t*)p)[i] = rad_f32_to_bf16(v); break;
        case RAD_F16:    ((uint16_t*)p)[i] = rad_f32_to_f16(v); break;
        case RAD_F32:    ((float*)p)[i] = v; break;
        case RAD_F8E4M3: ((uint8_t*)p)[i] = rad_f32_to_fp8e4m3(v); break;
        case RAD_F8E5M2: ((uint8_t*)p)[i] = rad_f32_to_fp8e5m2(v); break;
        case RAD_I8: {
            float r = std::nearbyintf(v);
            r = r < -127.f ? -127.f : (r > 127.f ? 127.f : r);
            ((int8_t*)p)[i] = (int8_t)r;
            break;
        }
        default: break;
    }
}

static inline float get(uint32_t dt, const void* p, size_t i) {
    switch (dt) {
        case RAD_BF16:   return rad_bf16_to_f32(((const uint16_t*)p)[i]);
        case RAD_F16:    return rad_f16_to_f32(((const uint16_t*)p)[i]);
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F8E4M3: return rad_fp8e4m3_to_f32(((const uint8_t*)p)[i]);
        case RAD_F8E5M2: return rad_fp8e5m2_to_f32(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        default:         return 0.0f;
    }
}

static inline bool coded(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_I8;
}

static inline RadTensor mk(void* data, uint32_t dt, std::vector<int64_t> shape,
                           std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = data; t.dtype = dt; t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

struct Params {
    std::vector<RadParam> p;
    std::vector<std::string> keep;
    Params() { keep.reserve(8); }                  /* sval points into `keep`: no reallocation */
    void i(const char* k, long long v) {
        RadParam q{}; q.key = k; q.kind = RAD_P_INT; q.ival = v; p.push_back(q);
    }
    void s(const char* k, const char* v) {
        keep.emplace_back(v);
        RadParam q{}; q.key = k; q.kind = RAD_P_STR; q.sval = keep.back().c_str(); p.push_back(q);
    }
};

static inline void fail(Tally& T, const Impl& im, const char* what, int M, int R, int K,
                        uint32_t dt, int m, int j) {
    ++T.fails;
    if (T.fails <= 20)
        std::printf("FAIL %-8s %s: M %d R %d K/N %d %s at row %d col %d\n", im.name, what, M, R, K,
                    rad_dtype_name(dt), m, j);
}

/* ------------------------------------------------------------------------------- lora_shrink */
static inline void shrink_case(Tally& T, const std::vector<Impl>& impls, std::mt19937& rng, int M,
                               int R, int K, uint32_t adt, int off) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.01f, 0.05f);
    const bool sc = coded(adt);
    const int group = 128, groups = (K + group - 1) / group;
    const int lda = K + 7, a_off = 3, lds = groups + 3, s_off = 1;
    const size_t es = esize(adt);
    const float amp = adt == RAD_I8 ? 40.0f : (sc ? 4.0f : 1.0f);
    std::vector<uint8_t> abuf((size_t)(M * lda + a_off) * es);
    for (int i = 0; i < M * lda + a_off; ++i) put(adt, abuf.data(), (size_t)i, amp * nd(rng));
    std::vector<float> sbuf((size_t)(M * lds + s_off));
    for (auto& v : sbuf) v = ud(rng);
    std::vector<int32_t> slot((size_t)M);
    for (int m = 0; m < M; ++m) slot[(size_t)m] = slot_of(m, off);
    std::vector<uint16_t> la((size_t)kS * R * K);
    for (auto& v : la) v = rad_f32_to_bf16(0.05f * nd(rng));

    RadTensor t[5]{};
    t[0] = mk(abuf.data() + a_off * es, adt, {M, K}, {lda, 1});
    if (sc) t[1] = mk(sbuf.data() + s_off, RAD_F32, {M, groups}, {lds, 1});
    t[2] = mk(slot.data(), RAD_I32, {M}, {1});
    t[3] = mk(la.data(), RAD_BF16, {kS, R, K}, {(int64_t)R * K, K, 1});
    Params P;
    P.i("M", M); P.i("K", K); P.i("R", R); P.i("S", kS); P.i("group", sc ? group : 0);
    P.s("dtype", rad_dtype_name(adt));
    RadArgs a{}; a.t = t; a.n_t = 5; a.p = P.p.data(); a.n_p = (int)P.p.size(); a.world_size = 1;

    /* The double sums, once: they do not depend on the implementation. */
    std::vector<double> want((size_t)M * R, 0.0), tol((size_t)M * R, 0.0);
    for (int m = 0; m < M; ++m) {
        const int32_t s = slot[(size_t)m];
        if (base(s)) continue;
        for (int j = 0; j < R; ++j) {
            double sum = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                float x = get(adt, abuf.data(), (size_t)(a_off + m * lda + k));
                if (sc) x = x * sbuf[(size_t)(s_off + m * lds + k / group)];
                const double term = (double)x * rad_bf16_to_f32(la[((size_t)s * R + j) * K + k]);
                sum += term; mag += std::fabs(term);
            }
            want[(size_t)m * R + j] = sum;
            tol[(size_t)m * R + j] = (K + 2) * std::ldexp(1.0, -24) * mag + 1e-30;
        }
    }
    for (const Impl& im : impls) {
        if (!im.shrink) continue;
        std::vector<float> tv((size_t)M * R);
        for (auto& x : tv) std::memcpy(&x, &kSentinel, 4);
        t[4] = mk(tv.data(), RAD_F32, {M, R}, {R, 1});
        ++T.cases;
        std::vector<uint8_t> scr;
        with_scratch(a, im.shrink_scratch, scr);
        const int rc = im.shrink(&a, nullptr);
        if (rc != RAD_OK) {
            fail(T, im, "shrink returned an error", M, R, K, adt, rc, -1);
            continue;
        }
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < R; ++j) {
                const size_t o = (size_t)m * R + j;
                if (base(slot[(size_t)m])) {
                    if (std::memcmp(&tv[o], &kSentinel, 4))
                        fail(T, im, "shrink wrote a base row", M, R, K, adt, m, j);
                    continue;
                }
                const double e = std::fabs((double)tv[o] - want[o]);
                if (e / tol[o] > T.worst_shrink) T.worst_shrink = e / tol[o];
                if (!(e <= tol[o])) fail(T, im, "shrink value", M, R, K, adt, m, j);
            }
    }
}

/* ------------------------------------------------------------------------------- lora_expand */
static inline void expand_case(Tally& T, const std::vector<Impl>& impls, std::mt19937& rng, int M,
                               int R, int N, uint32_t ydt, int off) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 2.0f);
    const int ldy = N + 5, y_off = 2;
    const size_t es = esize(ydt), ny = (size_t)(M * ldy + y_off);
    std::vector<uint8_t> y0(ny * es);
    for (size_t i = 0; i < ny; ++i) put(ydt, y0.data(), i, i % 3 == 0 ? -0.0f : nd(rng));
    std::vector<int32_t> slot((size_t)M);
    for (int m = 0; m < M; ++m) slot[(size_t)m] = slot_of(m, off);
    std::vector<float> tv((size_t)M * R);
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < R; ++j) {
            float v = 0.5f * nd(rng);
            if (base(slot[(size_t)m])) std::memcpy(&v, &kSentinel, 4);   /* never read */
            tv[(size_t)m * R + j] = v;
        }
    std::vector<uint16_t> lb((size_t)kS * N * R);
    for (auto& v : lb) v = rad_f32_to_bf16(0.05f * nd(rng));
    std::vector<float> scl((size_t)kS * N);
    for (auto& v : scl) v = ud(rng);

    RadTensor t[5]{};
    t[0] = mk(tv.data(), RAD_F32, {M, R}, {R, 1});
    t[1] = mk(slot.data(), RAD_I32, {M}, {1});
    t[2] = mk(lb.data(), RAD_BF16, {kS, N, R}, {(int64_t)N * R, R, 1});
    t[3] = mk(scl.data(), RAD_F32, {kS, N}, {N, 1});
    Params P;
    P.i("M", M); P.i("N", N); P.i("R", R); P.i("S", kS); P.s("dtype", rad_dtype_name(ydt));
    RadArgs a{}; a.t = t; a.n_t = 5; a.p = P.p.data(); a.n_p = (int)P.p.size(); a.world_size = 1;

    const double h = ydt == RAD_BF16 ? std::ldexp(1.0, -8) : ydt == RAD_F16 ? std::ldexp(1.0, -11)
                                                                             : 0.0;
    std::vector<uint8_t> first;
    for (size_t ii = 0; ii < impls.size(); ++ii) {
        const Impl& im = impls[ii];
        if (!im.expand) continue;
        std::vector<uint8_t> yb = y0;
        t[4] = mk(yb.data() + y_off * es, ydt, {M, N}, {ldy, 1});
        ++T.cases;
        const int rc = im.expand(&a, nullptr);
        if (rc != RAD_OK) {
            fail(T, im, "expand returned an error", M, R, N, ydt, rc, -1);
            continue;
        }
        const bool cmp_first = im.ref_bytes && !first.empty();
        for (size_t i = 0; i < ny; ++i) {
            const long long q = (long long)i - y_off;
            const int m = q < 0 ? -1 : (int)(q / ldy), n = q < 0 ? -1 : (int)(q % ldy);
            if (cmp_first && std::memcmp(&yb[i * es], &first[i * es], es))
                fail(T, im, "expand is not the first implementation's bytes", M, R, N, ydt, m, n);
            const bool inside = m >= 0 && m < M && n < N && !base(slot[(size_t)m]);
            if (!inside) {
                if (std::memcmp(&yb[i * es], &y0[i * es], es))
                    fail(T, im, "expand touched a base row or the slice's neighbours", M, R, N, ydt,
                         m, n);
                continue;
            }
            const int32_t s = slot[(size_t)m];
            double sum = 0, mag = 0;
            for (int j = 0; j < R; ++j) {
                const double term = (double)tv[(size_t)m * R + j] *
                                    rad_bf16_to_f32(lb[((size_t)s * N + n) * R + j]);
                sum += term; mag += std::fabs(term);
            }
            const double yin = get(ydt, y0.data(), i), sc = scl[(size_t)s * N + n];
            const double want = yin + sc * sum;
            const double e =
                4.0 * (R + 2) * std::ldexp(1.0, -24) * (std::fabs(yin) + std::fabs(sc) * mag);
            const double tol = h * (std::fabs(want) + e) + 2 * e + 1e-30;
            const double err = std::fabs((double)get(ydt, yb.data(), i) - want);
            if (err / tol > T.worst_expand) T.worst_expand = err / tol;
            if (!(err <= tol)) fail(T, im, "expand value", M, R, N, ydt, m, n);
        }
        if (first.empty()) first = yb;
    }
}

/* The whole matrix over `impls`. */
static inline void run_matrix(Tally& T, const std::vector<Impl>& impls, unsigned seed) {
    std::mt19937 rng(seed);
    for (int M : {1, 3, 17, 64, 130})
        for (int R : {16, 32, 48})
            for (int K : {64, 256, 1000})
                for (int off : {2, 0, 5}) {
                    if (off != 2 && M != 1) continue;      /* M = 1 also as a -1 and an S row */
                    shrink_case(T, impls, rng, M, R, K, RAD_BF16, off);
                    shrink_case(T, impls, rng, M, R, K, RAD_F8E4M3, off);
                    expand_case(T, impls, rng, M, R, K, RAD_BF16, off);
                    expand_case(T, impls, rng, M, R, K, RAD_F32, off);
                }
    for (int M : {1, 17})
        for (int K : {256, 1000}) {
            for (uint32_t adt : {(uint32_t)RAD_F16, (uint32_t)RAD_F32, (uint32_t)RAD_F8E5M2,
                                 (uint32_t)RAD_I8})
                shrink_case(T, impls, rng, M, 32, K, adt, 2);
            expand_case(T, impls, rng, M, 32, K, RAD_F16, 2);
        }
}

}  /* namespace lora_cases */
