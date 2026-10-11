/* tests/g4k_emu: the sandwich-norm joins (g4k_norm.hip) on the host SIMT emulator.
 *
 * Two references. The CHAIN is the unfused sequence the arch plugin would otherwise issue -- rmsnorm,
 * add, rmsnorm (or rmsnorm x2, add, rmsnorm, add, mul_weight) -- in libr4d's per-element arithmetic,
 * its sums of squares added in this kernel's block order: the fused kernel must match it BIT FOR
 * BIT, which is the claim that fusing changes nothing but the launch count. The HOST ARM sums in
 * index order -- what a different libr4d block width does too -- and may differ by a flipped
 * rounding, measured against the row's scale (see cmp). Padding columns and rows past M must be
 * left alone. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "g4k.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
extern "C" const RadKernelInfo* rad_kernel_at(int);
extern "C" int rad_kernel_count(void);

static std::mt19937 rng(11);
static float b2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2b(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float rb(float f) { return b2f(f2b(f)); }

/* the kernel's sum of squares: 256 threads strided over the row, a butterfly a wave, waves in order */
static float block_factor(const std::vector<float>& v, int n, float eps) {
    float part[256];
    for (int t = 0; t < 256; ++t) {
        float s = 0.0f;
        for (int k = 0; k < 24; ++k) { const int i = t + 256 * k; const float x = i < n ? v[(size_t)i] : 0.0f; s += x * x; }
        part[t] = s;
    }
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) {
        float l[32];
        for (int i = 0; i < 32; ++i) l[i] = part[w * 32 + i];
        for (int d = 16; d >= 1; d >>= 1) {
            float nl[32];
            for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
            std::memcpy(l, nl, sizeof l);
        }
        total += l[0];
    }
    return 1.0f / std::sqrt(total / (float)n + eps);
}
static std::vector<float> chain_norm(const std::vector<float>& v, const float* g, int n, float eps) {
    const float f = block_factor(v, n, eps);
    std::vector<float> y((size_t)n);
    for (int i = 0; i < n; ++i) y[(size_t)i] = rb((v[(size_t)i] * f) * g[i]);
    return y;
}

static RadTensor rows(void* p, uint32_t dt, int64_t M, int64_t n, int64_t ld) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = M; t.shape[1] = n; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor vec(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
static const RadKernelInfo* host_row(const char* op) {
    for (int i = 0; i < rad_kernel_count(); ++i)
        if (rad_kernel_at(i)->domain == RAD_DOMAIN_HOST && !std::strcmp(rad_kernel_at(i)->op, op)) return rad_kernel_at(i);
    return nullptr;
}
static std::vector<uint16_t> randb(size_t k, float sd) {
    std::normal_distribution<float> d(0.f, sd);
    std::vector<uint16_t> v(k);
    for (auto& x : v) x = f2b(d(rng));
    return v;
}
static std::vector<float> gains(int n) {
    std::normal_distribution<float> d(1.f, 0.4f);
    std::vector<float> g((size_t)n);
    for (auto& x : g) x = d(rng);
    return g;
}
/* vs the host arm, the difference in units of the row's RMS: a different summation order moves the
 * norm factor by an f32 ulp, which flips a bf16 rounding here and there -- and where the residual
 * add cancels, a flipped ulp of the addend is many ulps of a small sum, so ulps of the OUTPUT are
 * the wrong yardstick and the row's scale is the right one. */
struct Tally { long chain_bad = 0, host_diff = 0, host_bad = 0, pad = 0; double worst = 0.0; };
static void cmp(const uint16_t* got, const uint16_t* chain, const uint16_t* host, int n, Tally& t) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)b2f(host[i]) * b2f(host[i]);
    const double rms = std::sqrt(ss / n) + 1e-30;
    for (int i = 0; i < n; ++i) {
        if (got[i] != chain[i]) ++t.chain_bad;
        if (got[i] == host[i]) continue;
        ++t.host_diff;
        const double e = std::fabs((double)b2f(got[i]) - b2f(host[i])) / rms;
        if (e > t.worst) t.worst = e;
        /* one bf16 ulp of an element at 4x the row's rms is up to 3% of it: 1/16 passes a flipped
         * rounding of a large element and fails anything a real bug produces */
        if (e > 1.0 / 16) ++t.host_bad;
    }
}

static int case_nan(int M, int n, int pad, int extra_rows) {
    const int ld = n + pad, R = M + extra_rows;
    const float eps = 1e-6f;
    auto a = randb((size_t)R * ld, 3.f), x = randb((size_t)R * ld, 1.f), y0 = randb((size_t)R * ld, 1.f);
    auto wa = gains(n), wy = gains(n);
    auto xd = x, yd = y0, xh = x, yh = y0;
    RadTensor t[5] = { rows(a.data(), RAD_BF16, M, n, ld), vec(wa.data(), RAD_F32, n),
                       rows(xd.data(), RAD_BF16, M, n, ld), vec(wy.data(), RAD_F32, n),
                       rows(yd.data(), RAD_BF16, M, n, ld) };
    RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n", RAD_P_INT, n};
    p[2].key = "eps"; p[2].kind = RAD_P_F64; p[2].dval = eps; p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
    RadArgs ar{}; ar.t = t; ar.n_t = 5; ar.p = p; ar.n_p = 4;
    if (g4k_norm_add_norm_bf16(&ar, nullptr) != RAD_OK) { std::puts("norm_add_norm: device launch failed"); return 1; }
    t[2].data = xh.data(); t[4].data = yh.data();
    if (host_row("norm_add_norm")->launch(&ar, nullptr) != RAD_OK) { std::puts("norm_add_norm: host launch failed"); return 1; }
    Tally tx, ty;
    for (int m = 0; m < R; ++m) {
        const size_t o = (size_t)m * ld;
        if (m >= M) { for (int i = 0; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]); continue; }
        for (int i = n; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]);
        std::vector<float> av((size_t)n), xv((size_t)n);
        for (int i = 0; i < n; ++i) av[(size_t)i] = b2f(a[o + i]);
        const auto t1 = chain_norm(av, wa.data(), n, eps);                 /* rmsnorm */
        std::vector<uint16_t> cx((size_t)n), cy((size_t)n);
        for (int i = 0; i < n; ++i) { xv[(size_t)i] = rb(b2f(x[o + i]) + t1[(size_t)i]); cx[(size_t)i] = f2b(xv[(size_t)i]); }   /* add */
        const auto t2 = chain_norm(xv, wy.data(), n, eps);                 /* rmsnorm */
        for (int i = 0; i < n; ++i) cy[(size_t)i] = f2b(t2[(size_t)i]);
        cmp(&xd[o], cx.data(), &xh[o], n, tx);
        cmp(&yd[o], cy.data(), &yh[o], n, ty);
    }
    const bool ok = tx.chain_bad == 0 && ty.chain_bad == 0 && tx.host_bad == 0 && ty.host_bad == 0 && tx.pad == 0;
    std::printf("norm_add_norm  M %3d n %4d pitch %4d: vs chain x %ld y %ld differ; vs host arm %ld/%ld differ, worst "
                "%.1e of the row rms; padding touched %ld -> %s\n", M, n, ld, tx.chain_bad, ty.chain_bad, tx.host_diff,
                ty.host_diff, tx.worst > ty.worst ? tx.worst : ty.worst, tx.pad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* `next`: also the next layer's input norm, y = norm(x) * w_n over the x the join writes */
static int case_jas(int M, int n, int pad, int extra_rows, bool next = false) {
    const int ld = n + pad, R = M + extra_rows;
    const float eps = 1e-6f;
    /* the two arms as one [2M, n] buffer, as the plugin holds them: dense rows, then routed */
    auto ff = randb((size_t)2 * R * ld, 2.f), x = randb((size_t)R * ld, 4.f), y0 = randb((size_t)R * ld, 1.f);
    auto wd = gains(n), wr = gains(n), wj = gains(n), wn = gains(n);
    uint16_t sc = f2b(0.4375f + (float)(rng() % 64) / 128.f);
    auto xd = x, xh = x, yd = y0, yh = y0;
    RadTensor t[9] = { rows(ff.data(), RAD_BF16, M, n, ld), vec(wd.data(), RAD_F32, n),
                       rows(ff.data() + (size_t)M * ld, RAD_BF16, M, n, ld), vec(wr.data(), RAD_F32, n),
                       vec(wj.data(), RAD_F32, n), rows(xd.data(), RAD_BF16, M, n, ld), vec(&sc, RAD_BF16, 1),
                       vec(wn.data(), RAD_F32, n), rows(yd.data(), RAD_BF16, M, n, ld) };
    RadParam p[4]{}; p[0] = {"M", RAD_P_INT, M}; p[1] = {"n", RAD_P_INT, n};
    p[2].key = "eps"; p[2].kind = RAD_P_F64; p[2].dval = eps; p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
    RadArgs ar{}; ar.t = t; ar.n_t = next ? 9 : 7; ar.p = p; ar.n_p = 4;
    if (g4k_join_add_scale_bf16(&ar, nullptr) != RAD_OK) { std::puts("join_add_scale: device launch failed"); return 1; }
    t[5].data = xh.data(); t[8].data = yh.data();
    if (host_row("join_add_scale")->launch(&ar, nullptr) != RAD_OK) { std::puts("join_add_scale: host launch failed"); return 1; }
    Tally tx, ty;
    for (int m = 0; m < R; ++m) {
        const size_t o = (size_t)m * ld;
        if (m >= M) { for (int i = 0; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]); continue; }
        for (int i = n; i < ld; ++i) tx.pad += (xd[o + i] != x[o + i]) + (yd[o + i] != y0[o + i]);
        std::vector<float> dv((size_t)n), rv((size_t)n), sv((size_t)n);
        for (int i = 0; i < n; ++i) { dv[(size_t)i] = b2f(ff[o + i]); rv[(size_t)i] = b2f(ff[(size_t)M * ld + o + i]); }
        const auto d1 = chain_norm(dv, wd.data(), n, eps);                  /* rmsnorm (arm 1) */
        const auto r1 = chain_norm(rv, wr.data(), n, eps);                  /* rmsnorm (arm 2) */
        for (int i = 0; i < n; ++i) sv[(size_t)i] = rb(d1[(size_t)i] + r1[(size_t)i]);   /* add */
        const auto j = chain_norm(sv, wj.data(), n, eps);                   /* rmsnorm */
        std::vector<uint16_t> cx((size_t)n);
        for (int i = 0; i < n; ++i) cx[(size_t)i] = f2b(rb(b2f(x[o + i]) + j[(size_t)i]) * b2f(sc));   /* add, mul_weight */
        cmp(&xd[o], cx.data(), &xh[o], n, tx);
        if (next) {                                                         /* the next rmsnorm */
            std::vector<float> xv((size_t)n);
            for (int i = 0; i < n; ++i) xv[(size_t)i] = b2f(cx[(size_t)i]);
            const auto yn = chain_norm(xv, wn.data(), n, eps);
            std::vector<uint16_t> cy((size_t)n);
            for (int i = 0; i < n; ++i) cy[(size_t)i] = f2b(yn[(size_t)i]);
            cmp(&yd[o], cy.data(), &yh[o], n, ty);
        }
    }
    const bool ok = tx.chain_bad == 0 && ty.chain_bad == 0 && tx.host_bad == 0 && ty.host_bad == 0 && tx.pad == 0;
    std::printf("join_add_scale%s M %3d n %4d pitch %4d: vs chain %ld/%ld differ; vs host arm %ld/%ld differ, worst %.1e "
                "of the row rms; padding touched %ld -> %s\n", next ? "+next" : "     ", M, n, ld, tx.chain_bad,
                ty.chain_bad, tx.host_diff, ty.host_diff, tx.worst > ty.worst ? tx.worst : ty.worst, tx.pad,
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    /* Gemma 4 26B's 2816 at decode and verify heights, a prefill height, odd widths, the limit */
    const int Ms[] = {1, 4, 16, 37};
    for (int M : Ms) fails += case_nan(M, 2816, M % 2 ? 64 : 0, 2);
    for (int M : Ms) fails += case_jas(M, 2816, M % 2 ? 0 : 128, 2);
    fails += case_nan(3, 1000, 24, 1) + case_jas(3, 1000, 24, 1);
    fails += case_nan(2, 6144, 0, 0) + case_jas(2, 6144, 16, 0);
    fails += case_nan(5, 256, 8, 1) + case_jas(5, 5376, 0, 1);
    /* the edge between the 12-a-thread kernels and the 24: 3072 the last of the first */
    fails += case_nan(2, 3072, 0, 1) + case_nan(2, 3073, 8, 1);
    fails += case_jas(2, 3072, 0, 1, true) + case_jas(2, 3073, 8, 1, true) + case_jas(3, 3073, 0, 1);
    /* and with the next layer's input norm folded in */
    for (int M : Ms) fails += case_jas(M, 2816, M % 2 ? 64 : 0, 2, true);
    fails += case_jas(3, 1000, 24, 1, true) + case_jas(2, 6144, 16, 0, true);
    std::printf("norm joins: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
