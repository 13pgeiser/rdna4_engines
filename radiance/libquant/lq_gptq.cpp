/* lq_gptq.cpp -- the Hessian side of GPTQ and the sequential loop. See lq_gptq.h. */
#include "lq_gptq.h"
#include "lq_hess.h"

#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace lq {

namespace {

/* H is damped, inverted and re-factorised into upper U with U^T U = H^-1, the form the sequential
 * loop wants: column i's error is divided by U[i][i] and spread over columns j > i with weight
 * U[i][j] (Frantar et al.'s formulation). Double throughout: a Cholesky is K^3/6 accumulations
 * deep, and f32 loses real conditioning on a matrix near-singular in the directions a rotation
 * flattened. */
#ifdef RAD_LQ_LAPACK
/* LAPACK, column-major, Fortran calling convention (OpenBLAS). See the three functions below. */
extern "C" void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
extern "C" void dpotri_(const char* uplo, const int* n, double* a, const int* lda, int* info);

/* dpotrf rejects a non-positive pivot but lets a NaN through, where the loops' !(s > 0) refuses
 * it: a factor whose diagonal is not finite and positive is refused here as there. */
static bool diag_ok(const std::vector<double>& a, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        const double d = a[(size_t)(i * k + i)];
        if (!(d > 0.0) || !std::isfinite(d)) return false;
    }
    return true;
}

/* Zero the row-major strict upper (upper = true) or strict lower triangle. */
static void zero_triangle(std::vector<double>& a, int64_t k, bool upper) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < k; ++i) {
        if (upper) for (int64_t j = i + 1; j < k; ++j) a[(size_t)(i * k + j)] = 0.0;
        else       for (int64_t j = 0; j < i; ++j)     a[(size_t)(i * k + j)] = 0.0;
    }
}

bool chol_lower_lapack(std::vector<double>& a, int64_t k) {
    const int n = (int)k;
    int info = 0;
    dpotrf_("U", &n, a.data(), &n, &info);
    if (info != 0) return false;
    zero_triangle(a, k, true);
    return diag_ok(a, k);
}

void inv_from_chol_lapack(std::vector<double>& l, int64_t k) {
    const int n = (int)k;
    int info = 0;
    dpotri_("U", &n, l.data(), &n, &info);
    /* Column-major upper is row-major lower: mirror it into the row-major upper half. */
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = i + 1; j < k; ++j) l[(size_t)(i * k + j)] = l[(size_t)(j * k + i)];
}

bool chol_upper_lapack(std::vector<double>& a, int64_t k) {
    const int n = (int)k;
    int info = 0;
    dpotrf_("L", &n, a.data(), &n, &info);
    if (info != 0) return false;
    zero_triangle(a, k, false);
    return diag_ok(a, k);
}
#endif  /* RAD_LQ_LAPACK */

/* The single-threaded loops: the reference, and the only path without OpenBLAS. */
bool chol_lower_loop(std::vector<double>& a, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            double s = a[(size_t)(i * k + j)];
            for (int64_t t = 0; t < j; ++t) s -= a[(size_t)(i * k + t)] * a[(size_t)(j * k + t)];
            if (i == j) {
                if (!(s > 0.0)) return false;
                a[(size_t)(i * k + i)] = std::sqrt(s);
            } else {
                a[(size_t)(i * k + j)] = s / a[(size_t)(j * k + j)];
            }
        }
        for (int64_t j = i + 1; j < k; ++j) a[(size_t)(i * k + j)] = 0.0;
    }
    return true;
}

/* A = L L^T, factored in place -> A^-1, full and symmetric: L^-1 first, then L^-T L^-1. */
void inv_from_chol_loop(std::vector<double>& l, int64_t k) {
    for (int64_t j = 0; j < k; ++j) {
        l[(size_t)(j * k + j)] = 1.0 / l[(size_t)(j * k + j)];
        for (int64_t i = j + 1; i < k; ++i) {
            double s = 0.0;
            for (int64_t t = j; t < i; ++t) s += l[(size_t)(i * k + t)] * l[(size_t)(t * k + j)];
            l[(size_t)(i * k + j)] = -s / l[(size_t)(i * k + i)];
        }
    }
    std::vector<double> out((size_t)(k * k), 0.0);
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (int64_t t = i; t < k; ++t) s += l[(size_t)(t * k + i)] * l[(size_t)(t * k + j)];
            out[(size_t)(i * k + j)] = s;
            out[(size_t)(j * k + i)] = s;
        }
    l.swap(out);
}

/* A -> U upper with A = U^T U. */
bool chol_upper_loop(std::vector<double>& a, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        double d = a[(size_t)(i * k + i)];
        for (int64_t t = 0; t < i; ++t) {
            const double v = a[(size_t)(t * k + i)];
            d -= v * v;
        }
        if (!(d > 0.0)) return false;
        const double u = std::sqrt(d);
        a[(size_t)(i * k + i)] = u;
        for (int64_t j = i + 1; j < k; ++j) {
            double s = a[(size_t)(i * k + j)];
            for (int64_t t = 0; t < i; ++t) s -= a[(size_t)(t * k + i)] * a[(size_t)(t * k + j)];
            a[(size_t)(i * k + j)] = s / u;
        }
        for (int64_t j = 0; j < i; ++j) a[(size_t)(i * k + j)] = 0.0;
    }
    return true;
}

#ifdef RAD_LQ_LAPACK
/* LAPACK where OpenBLAS was found; RAD_LQ_LAPACK=0 takes the loops instead, the reference the
 * LAPACK path is checked against (quant_algo_test). */
bool use_lapack() {
    const char* e = std::getenv("RAD_LQ_LAPACK");
    return !(e && e[0] == '0');
}
#endif

bool chol_lower(std::vector<double>& a, int64_t k) {
#ifdef RAD_LQ_LAPACK
    if (use_lapack()) return chol_lower_lapack(a, k);
#endif
    return chol_lower_loop(a, k);
}

void inv_from_chol(std::vector<double>& l, int64_t k) {
#ifdef RAD_LQ_LAPACK
    if (use_lapack()) return inv_from_chol_lapack(l, k);
#endif
    inv_from_chol_loop(l, k);
}

bool chol_upper(std::vector<double>& a, int64_t k) {
#ifdef RAD_LQ_LAPACK
    if (use_lapack()) return chol_upper_lapack(a, k);
#endif
    return chol_upper_loop(a, k);
}

/* Layer factors kept at once. */
constexpr size_t kFactors = 4;

struct Cache {
    std::mutex mu;
    std::map<std::string, Gram> f;   /* null = looked and found nothing */
    std::deque<std::string> order;   /* f's keys, oldest first */
};

Cache& cache() {
    static Cache c;
    return c;
}

}  /* namespace */

std::vector<int32_t> gptq_act_order(const char* dir, const char* name, int64_t K, int64_t domain) {
    std::vector<int32_t> perm((size_t)K);
    for (int64_t i = 0; i < K; ++i) perm[(size_t)i] = (int32_t)i;
    if (!dir || !*dir || !name || !*name || K <= 0) return perm;
    std::vector<double> acc;
    if (!calib_sum(dir, calib_resolve(dir, name, K), K, domain, &acc)) return perm;
    /* Largest input energy first, ties in column order. The recursion absorbs error best early,
     * so the costliest columns are quantised while every later column can still take it. */
    std::stable_sort(perm.begin(), perm.end(), [&](int32_t x, int32_t y) {
        return acc[(size_t)x * (size_t)K + (size_t)x] > acc[(size_t)y * (size_t)K + (size_t)y];
    });
    return perm;
}

Gram gptq_factor(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm) {
    if (!dir || !*dir || !name || !*name || K <= 0) return nullptr;
    const std::string stem = calib_resolve(dir, name, K);
    const bool layer = stem == calib_stem(name);
    char dk[32];
    std::snprintf(dk, sizeof dk, "%.17g", damp);
    const std::string key = std::string(dir) + "|" + stem + "|" + std::to_string(K) + "|" +
                            std::to_string(domain) + "|" + dk + (perm ? "|act" : "");

    /* A layer's factor is shared by its experts, so it is built once under the lock and cached.
     * An expert's own is built for its one call outside the lock, so experts factor in parallel. */
    Cache& c = cache();
    std::unique_lock<std::mutex> lk(c.mu, std::defer_lock);
    if (layer) {
        lk.lock();
        auto it = c.f.find(key);
        if (it != c.f.end()) return it->second;
    }
    auto keep = [&](Gram u) {
        /* A few layer factors, the oldest dropped first: a dense model reads each layer's
         * once, and a 21504-wide factor is 1.85 GB in f32 (every one kept OOM-killed
         * TranslateGemma 27B's conversion at layer 25). A MoE layer's experts run in parallel
         * and alternate between its gate_up and down stems, so one kept factor would be
         * recomputed over and over. */
        if (layer && !c.f.count(key)) {
            while (c.order.size() >= kFactors) {
                c.f.erase(c.order.front());
                c.order.pop_front();
            }
            c.f.emplace(key, u);
            c.order.push_back(key);
        }
        return u;
    };

    std::vector<double> acc;
    const int found = calib_sum(dir, stem, K, domain, &acc);
    if (!found) {
        std::fprintf(stderr, "libquant: no calibration for '%s' in %s (looked for gram.r*.%s.bin); "
                             "rounding it to nearest\n", name, dir, stem.c_str());
        return keep(nullptr);
    }
    /* Under act order the factor is of P H P^T, since the recursion runs over columns in
     * quantisation order. */
    if (perm) {
        std::vector<double> p((size_t)(K * K));
        for (int64_t i = 0; i < K; ++i)
            for (int64_t j = 0; j < K; ++j)
                p[(size_t)(i * K + j)] = acc[(size_t)perm[i] * (size_t)K + (size_t)perm[j]];
        acc.swap(p);
    }

    double tr = 0.0;
    int64_t dead = 0;
    for (int64_t i = 0; i < K; ++i) {
        if (!(acc[(size_t)(i * K + i)] > 0.0)) ++dead;
        tr += acc[(size_t)(i * K + i)];
    }
    /* A dead column is a channel the corpus never exercised: its diagonal is zero, the Cholesky
     * would fail on it, and the calibration says nothing about it. It gets the mean diagonal, so
     * the recursion does not spend other columns' error on it. */
    const double mean = tr / (double)K;
    const double lam  = damp * mean;
    for (int64_t i = 0; i < K; ++i) {
        double& d = acc[(size_t)(i * K + i)];
        if (!(d > 0.0)) d = mean;
        d += lam;
    }
    if (dead)
        std::fprintf(stderr, "libquant: %lld of %lld columns of '%s' were never exercised by the "
                             "calibration corpus\n", (long long)dead, (long long)K, stem.c_str());
    if (!chol_lower(acc, K)) {
        std::fprintf(stderr, "libquant: the Hessian for '%s' is not positive definite even "
                             "damped; rounding it to nearest\n", stem.c_str());
        return keep(nullptr);
    }
    inv_from_chol(acc, K);
    if (!chol_upper(acc, K)) {
        std::fprintf(stderr, "libquant: the inverse Hessian for '%s' would not factor; rounding "
                             "it to nearest\n", stem.c_str());
        return keep(nullptr);
    }
    auto u = std::make_shared<std::vector<float>>((size_t)(K * K));
    for (size_t i = 0; i < u->size(); ++i) (*u)[i] = (float)acc[i];
    if (layer)
        std::fprintf(stderr, "libquant: GPTQ quantises '%s' against %d calibration file(s), "
                             "K=%lld\n", stem.c_str(), found, (long long)K);
    return keep(u);
}

void gptq_quantise(const Grid& g, float* w, int32_t* codes, const Scales& scales, int64_t N,
                   int64_t K, const float* u) {
    if (!w || !codes || N <= 0 || K <= 0) return;
    auto one = [&](int64_t n, int64_t i, float v) -> float {
        const BlockScale& b = scales.at(n, i);
        const int c = grid_code(g, b, v);
        codes[(size_t)(n * K + i)] = c;
        return grid_decode(g, b, c);
    };

    if (!u) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t n = 0; n < N; ++n)
            for (int64_t i = 0; i < K; ++i) w[(size_t)(n * K + i)] = one(n, i, w[(size_t)(n * K + i)]);
        return;
    }

    /* Column blocks outside, rows inside. Rows are independent and are the parallelism, but a row
     * running all of K alone would stream all K^2 of `u`. With the block loop outside, all rows
     * work on the same columns together and the shared 128 x K slice of `u` stays in cache. */
    const int64_t B = 128;
    std::vector<float> err((size_t)(N * B));
    for (int64_t b0 = 0; b0 < K; b0 += B) {
        const int64_t b1 = (b0 + B < K) ? b0 + B : K;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t n = 0; n < N; ++n) {
            float* wr = w + (size_t)(n * K);
            float* er = err.data() + (size_t)(n * B);
            for (int64_t i = b0; i < b1; ++i) {
                const float* ur = u + (size_t)(i * K);
                const float d = ur[i];
                const float v = wr[i];
                const float dq = one(n, i, v);
                wr[i] = dq;
                /* d is U[i][i], strictly positive: chol_upper refuses otherwise. */
                const float e = (v - dq) / d;
                er[i - b0] = e;
                for (int64_t j = i + 1; j < b1; ++j) wr[j] -= e * ur[j];
            }
            for (int64_t t = b0; t < b1; ++t) {
                const float e = er[t - b0];
                const float* ur = u + (size_t)(t * K);
                for (int64_t j = b1; j < K; ++j) wr[j] -= e * ur[j];
            }
        }
    }
}

void gptq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                 int64_t N, int64_t K, const float* h, double damp, int passes) {
    if (!x || !w || !codes || !h || N <= 0 || K <= 0 || passes <= 0) return;
    /* Same damping as gptq_factor, so both minimise the same form: lambda is `damp` times the mean
     * diagonal, and an unexercised column takes the mean as its diagonal. `ad` is what the form
     * adds to H's diagonal, `dg` the resulting diagonal. */
    double tr = 0.0;
    for (int64_t i = 0; i < K; ++i) tr += h[(size_t)(i * K + i)];
    const double mean = tr / (double)K;
    const float lam = (float)(damp * mean);
    std::vector<float> ad((size_t)K), dg((size_t)K);
    for (int64_t i = 0; i < K; ++i) {
        const float d = h[(size_t)(i * K + i)];
        ad[(size_t)i] = (d > 0.0f ? 0.0f : (float)mean) + lam;
        dg[(size_t)i] = (d > 0.0f ? d : (float)mean) + lam;
    }
    /* G = E (H + diag(ad)) for all rows by the deterministic blocked product; each row then keeps
     * its G current by adding one row of it per code that moves. */
    std::vector<float> e((size_t)(N * K)), gm((size_t)(N * K));
    for (size_t i = 0; i < e.size(); ++i) e[i] = w[i] - x[i];
    mat_hess(e.data(), N, K, h, gm.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t n = 0; n < N; ++n) {
        float* er = e.data() + (size_t)(n * K);
        float* gr = gm.data() + (size_t)(n * K);
        const float* xr = x + (size_t)(n * K);
        float* wr = w + (size_t)(n * K);
        int32_t* cr = codes + (size_t)(n * K);
        for (int64_t j = 0; j < K; ++j) gr[j] += ad[(size_t)j] * er[j];
        for (int pass = 0; pass < passes; ++pass) {
            int64_t moved = 0;
            for (int64_t j = 0; j < K; ++j) {
                const float hj = dg[(size_t)j];
                /* The row's gradient at j less column j's own term, over the diagonal. */
                const float t = xr[j] - (gr[j] - er[j] * hj) / hj;
                const BlockScale& b = scales.at(n, j);
                const int c = grid_code(g, b, t);
                if (c == cr[j]) continue;
                const float dq = grid_decode(g, b, c);
                const float d = (dq - xr[j]) - er[j];
                /* Only a strict improvement moves a code: the form changes by d (2 g_j + d hj),
                 * g_j the current gradient; a tie keeps GPTQ's choice. */
                if (!(d * (2.0f * gr[j] + d * hj) < 0.0f)) continue;
                const float* hr = h + (size_t)(j * K);
                for (int64_t t2 = 0; t2 < K; ++t2) gr[t2] += d * hr[t2];
                gr[j] += ad[(size_t)j] * d;
                er[j] += d;
                wr[j] = dq;
                cr[j] = c;
                ++moved;
            }
            if (!moved) break;
        }
    }
}

}  /* namespace lq */
