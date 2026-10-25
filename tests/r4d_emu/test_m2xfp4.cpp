/* tests/r4d_emu: the M2XFP4 x fp8 GEMMs (r4d_gemm_m2xfp4a8_decode.hip, r4d_gemm_m2xfp4a8_tiled.hip)
 * on the host SIMT emulator. The container's m2xfp4 encoding (the paper's weight-side Sg-EM,
 * arXiv 2601.19213 §4.4.2) is synthesised from random f32 weights by an in-test scalar quantiser
 * implementing the spec: per group of 32 columns an E8M0 exponent with the adaptive bias (-1, 0, 1,
 * drawn at random) folded into the stored byte, and a meta byte of four two-bit subgroup
 * multipliers with k = 3 in heavy rotation; the codes are the e2m1 roundings of the exact
 * dequantiser's inverse.
 * The stored bytes are then built the way r4d_rad_relayout_m2xfp4 builds them (fragment-order
 * codes, the meta plane appended in the exponents' [K/32][N] order), and both kernels run against
 * an f64 CPU reference of the exact decode, at the FP8 WMMA's tolerance plus the table's accepted
 * k = 3 rounding (rel_l2 < 4e-2, see run_cases). Cases cover both activation scale grids (per row,
 * per (row, 128) with a zero block), M across the decode band's band edges (1, 8, 16, 40, 128) and
 * the tiled row's, the one-fragment prefetch form (M <= 16), both K slabs (64 and 128), and a
 * split-K launch on the decode kernel.
 *
 * The table itself is held to arithmetic first: every kM2xfp4Mag[d][k] cell must be the
 * round-to-nearest-even e4m3 of e2m1mag * (1 + k/4) * 2^-d; the k = 0..2 rows must be exact while
 * the product is a normal e4m3 value, and the k = 3 odd-mantissa entries must be the tie's even
 * side (2.5, 5, 10 scaled by 2^-d). The unpack is held to the same arithmetic over every code at
 * every (d, k).
 *
 * The declaration side (r4d_rows.cpp) is not in the emu build, so the two cfg readers it defines,
 * which the units' plugin entries call, are stubbed in cfg_stub.cpp at their axis defaults (0 = the
 * kernels' own rules), which is what an untuned call gets. This test calls the raw entries, so it
 * runs no stub; everything it runs is the library's own code.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d.h"
#include "r4d_m2xfp4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

/* ------------------------------------------------------------------ arithmetic e4m3 / e2m1 */

/* Round-to-nearest-even e4m3 (OCP FP8, 1-4-3, bias 7, subnormals at 2^-9, 0x7F/0xFF NaN), the
 * conversion the device tables and the WMMA's own converts use. */
static uint8_t rne_e4m3(double v) {
    if (v == 0) return std::signbit(v) ? 0x80 : 0;   /* the kernels keep a sign on zero */
    const uint8_t sign = v < 0 ? 0x80 : 0;
    double a = std::fabs(v);
    if (a > 480) return (uint8_t)(sign | 0x7E);
    int e = 0;
    while (a >= 2.0) { a *= 0.5; ++e; }
    while (a < 1.0) { a *= 2.0; --e; }
    int m;
    if (e < -6) {                                   /* subnormal: 2^-9 steps of the value itself */
        double q = std::fabs(v) * 512.0;
        double fl = std::floor(q);
        const double frac = q - fl;
        int qi = (int)fl;
        if (frac > 0.5 || (frac == 0.5 && (qi & 1))) ++qi;
        if (qi >= 8) return (uint8_t)(sign | 0x08); /* up to min normal 2^-6 */
        return (uint8_t)(sign | qi);
    }
    double q = (a - 1.0) * 8.0;
    m = (int)q;
    const double frac = q - m;
    if (frac > 0.5 || (frac == 0.5 && (m & 1))) ++m;
    if (m == 8) { m = 0; ++e; }
    return (uint8_t)(sign | ((e + 7) << 3) | m);
}

static double e4m3_f64(uint8_t b) {
    const double s = b & 0x80 ? -1.0 : 1.0;
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 0) return s * m * 0.001953125;                     /* 2^-9 steps */
    return s * std::ldexp(1.0 + m / 8.0, e - 7);
}

/* The e2m1 magnitudes, and the quantiser's RNE code for a magnitude against them (ties to the even
 * code, as the hardware's convert and the CPU reference agree). */
static const double kE2m1Mag[8] = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
static uint8_t rne_e2m1(double v) {
    double a = std::fabs(v);
    int best = 0;
    double bd = 1e300;
    for (int i = 0; i < 8; ++i) {
        const double d = std::fabs(a - kE2m1Mag[i]);
        if (d < bd || (d == bd && !(best & 1))) { bd = d; best = i; }
    }
    return (uint8_t)(best | (v < 0 ? 8 : 0));
}
static double e2m1_f64(uint8_t code) {
    return (code & 8 ? -1.0 : 1.0) * kE2m1Mag[code & 7];
}

/* ------------------------------------------------------------------ the table checks */

static int check_lut() {
    int fails = 0;
    for (int d = 0; d < 16; ++d)
        for (int k = 0; k < 4; ++k)
            for (int m = 0; m < 8; ++m) {
                const uint8_t got = (uint8_t)((kM2xfp4Mag[d][k][m >> 2] >> (8 * (m & 3))) & 0xFF);
                const double exact = kE2m1Mag[m] * (1.0 + k / 4.0) * std::ldexp(1.0, -d);
                const uint8_t want = rne_e4m3(exact);
                if (got != want) {
                    std::printf("FAIL lut d=%d k=%d mag=%d: %02x, want %02x (%.6g)\n", d, k, m,
                                got, want, exact);
                    ++fails;
                    continue;
                }
                /* k = 0..2 exact wherever the product is a normal e4m3 value; k = 3 only the
                 * odd-mantissa magnitudes round, and to the tie's even side. */
                if (e4m3_f64(got) != exact) {
                    const bool k3tie = k == 3 && (m == 3 || m == 5 || m == 7);
                    const double rounded = k3tie ? std::ldexp(m == 3 ? 2.5 : m == 5 ? 5.0 : 10.0, -d)
                                                 : exact;
                    if (k3tie && e4m3_f64(got) == rounded) continue;
                    /* the subnormal floor: like r4d_mxfp4.h's table one binade earlier, the 2^-9
                     * step rounds once the product falls in it; exactness is only claimed while
                     * the product is normal (>= 2^-6), and every k = 0..2 normal product is */
                    if (k <= 2 && exact >= 0.015625 && std::fabs(exact) > 0) {
                        std::printf("FAIL lut k=%d d=%d mag=%d: %02x decodes %.6g, want %.6g\n", k,
                                    d, m, got, e4m3_f64(got), exact);
                        ++fails;
                    }
                }
            }
    /* and the k = 0 rows are r4d_mxfp4.h's, byte for byte (a memcmp would walk the [k] axis) */
    static const unsigned kM0[16][2] = {
        {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u}, {0x2c282000u, 0x3c383430u},
        {0x24201800u, 0x34302c28u}, {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
        {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u}, {0x03020100u, 0x0c080604u},
        {0x02010000u, 0x06040302u}, {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
        {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
        {0x00000000u, 0x00000000u},
    };
    for (int d = 0; d < 16; ++d)
        if (kM2xfp4Mag[d][0][0] != kM0[d][0] || kM2xfp4Mag[d][0][1] != kM0[d][1]) {
            std::printf("FAIL lut: the k=0 row d=%d is not kMxfp4Mag\n", d);
            ++fails;
        }
    return fails;
}

/* The unpack against arithmetic, over every 16-bit pattern at both halves of a word and random
 * words, at every (d, k). */
static int check_unpack() {
    int fails = 0;
    auto one = [&](uint32_t word, int d, int k) {
        uint32_t lo = 0, hi = 0;
        r4d_m2xfp4_unpack8(word, d, k, &lo, &hi);
        for (int j = 0; j < 8; ++j) {
            const uint8_t code = (uint8_t)((word >> (4 * j)) & 0xF);
            const uint8_t got = (uint8_t)((j < 4 ? (lo >> (8 * j)) : (hi >> (8 * (j - 4)))) & 0xFF);
            const double exact = e2m1_f64(code) * (1.0 + k / 4.0) * std::ldexp(1.0, -d);
            const uint8_t want = rne_e4m3(exact);
            if (got != want) {
                if (fails < 12)
                    std::printf("FAIL unpack w=%08x d=%d k=%d j=%d: %02x, want %02x\n", word, d, k,
                                j, got, want);
                ++fails;
            }
        }
    };
    for (int d = 0; d < 16; ++d)
        for (int k = 0; k < 4; ++k) {
            for (uint32_t v = 0; v < 65536; ++v) {
                one(v, d, k);
                if (v < 256) { one(v << 24, d, k); one(v << 16, d, k); }
            }
        }
    std::mt19937 rng(11);
    for (int i = 0; i < 20000; ++i) one(rng(), (int)(rng() % 16), (int)(rng() % 4));
    return fails;
}

/* ------------------------------------------------------------------ the GEMM cases */

struct Case {
    const char* name;
    const char* kernel;    // the decode row's two forms, or the tiled row
    int M, N, K;
    bool ablk;             // the activation scale per (row, 128 of K), with a zero block
    int ks;                // the decode launch's split (0 = the kernel's rule)
};

/* The in-test quantiser's planes for one weight [N][K]: codes (packed e2m1, element 2j in bits
 * 0-3 of byte j), the per-(row, 32) exponents with the bias folded in, the meta bytes, and the
 * f64 exact dequant the reference multiplies. */
struct M2Weight {
    std::vector<uint8_t> codes;   // [N][K/2]
    std::vector<uint8_t> scale;   // [N][K/32]
    std::vector<uint8_t> meta;    // [N][K/32]
    std::vector<double> exact;    // [N][K], the spec's decode
};

static M2Weight m2_quantise(const std::vector<float>& w, int N, int K, std::mt19937& rng) {
    const int nblk = K / 32;
    M2Weight m;
    m.codes.assign((size_t)N * K / 2, 0);
    m.scale.assign((size_t)N * nblk, 0);
    m.meta.assign((size_t)N * nblk, 0);
    m.exact.assign((size_t)N * K, 0.0);
    std::uniform_int_distribution<int> bias(-1, 1);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nblk; ++b) {
            double absmax = 0.0;
            for (int j = 0; j < 32; ++j) absmax = std::max(absmax, (double)std::fabs(w[(size_t)n * K + b * 32 + j]));
            /* the exponent the six-magnitude grid needs, plus the adaptive bias, folded into the
             * stored byte; a zero group keeps a small exponent, its codes all zero */
            int e8 = absmax > 0 ? (int)std::ceil(std::log2(absmax / 6.0)) : -30;
            e8 += bias(rng);
            if (e8 < -30) e8 = -30;
            if (e8 > 30) e8 = 30;
            const uint8_t sbyte = (uint8_t)(e8 + 127);
            uint8_t mbyte = 0;
            for (int s = 0; s < 4; ++s) {
                /* k = 3 in heavy rotation, the multiplier the rounding lives on */
                const int k = (n + b + s * 3 + (int)(rng() % 2)) & 3;
                mbyte |= k << (2 * s);
                for (int j = 0; j < 8; ++j) {
                    const int col = b * 32 + s * 8 + j;
                    const double scale = std::ldexp(1.0 + k / 4.0, e8);
                    const uint8_t code = rne_e2m1(w[(size_t)n * K + col] / scale);
                    m.codes[(size_t)n * K / 2 + col / 2] |=
                        (uint8_t)(code << (4 * (col & 1)));
                    m.exact[(size_t)n * K + col] =
                        e2m1_f64(code) * std::ldexp(1.0, e8) * (1.0 + k / 4.0);
                }
            }
            m.scale[(size_t)n * nblk + b] = sbyte;
            m.meta[(size_t)n * nblk + b] = mbyte;
        }
    return m;
}

/* The stored weight, as r4d_rad_relayout_m2xfp4 writes it: codes in WMMA fragment order, then the
 * meta plane in the exponents' [K/32][N] order, and the exponent plane and per-row reference
 * beside them. */
static void m2_store(const M2Weight& m, int N, int K, std::vector<uint8_t>* wq,
                     std::vector<uint8_t>* ws, std::vector<uint8_t>* wm,
                     std::vector<uint8_t>* wref) {
    const int nblk = K / 32;
    wq->assign((size_t)N * K / 2 + (size_t)nblk * N, 0);
    ws->assign((size_t)nblk * N, 0);
    wm->assign((size_t)nblk * N, 0);
    wref->assign(N, 0);
    /* fragment order: slot = ((t * ks + ksi) * 32 + lh * 16 + r) * 4 + bin, element 2j in the low
     * nibble of byte j (r4d_layout.cpp, mx_slot) */
    const int ks16 = K / 16;
    for (int n = 0; n < N; ++n) {
        const int t = n / 16, r = n % 16;
        uint8_t ref = 0;
        for (int b = 0; b < nblk; ++b) {
            const uint8_t s = m.scale[(size_t)n * nblk + b];
            (*ws)[(size_t)b * N + n] = s;
            (*wm)[(size_t)b * N + n] = m.meta[(size_t)n * nblk + b];
            if (s > ref) ref = s;
        }
        (*wref)[n] = ref;
        for (int k = 0; k < K; ++k) {
            const uint8_t code = (uint8_t)((m.codes[(size_t)n * K / 2 + k / 2] >> (4 * (k & 1))) & 0xF);
            const int ksi = k / 16, kin = k % 16;
            const int lh = kin / 8, bin = (kin % 8) / 2;
            const size_t idx = (size_t)((t * ks16 + ksi) * 32 + lh * 16 + r) * 4 + bin;
            (*wq)[idx] |= (uint8_t)(code << (4 * (k & 1)));
        }
    }
}

static double bf16_f64(uint16_t h) {
    const double s = h & 0x8000 ? -1.0 : 1.0;
    const int e = (h >> 7) & 0xFF, m = h & 0x7F;
    if (e == 0) return s * m * std::ldexp(1.0, -133);          /* zero / bf16 subnormal */
    return s * std::ldexp(1.0 + m / 128.0, e - 127);
}

static int run_cases() {
    static const Case all[] = {
        { "decode1, K 128, rule split 16, per-row scales",
          "(r4d_gemm_m2xfp4a8_decode1_kernel<128, 16, false, false>)", 1, 48, 128, false, 0 },
        { "decode1, K 64, split 4", "(r4d_gemm_m2xfp4a8_decode1_kernel<64, 4, false, false>)",
          8, 16, 64, false, 4 },
        { "decode1, M 16, block scales", "(r4d_gemm_m2xfp4a8_decode1_kernel<128, 2, false, true>)",
          16, 128, 256, true, 2 },
        { "decode, M 40 three fragments, odd K slab 192",
          "(r4d_gemm_m2xfp4a8_decode_kernel<64, 1, 3, false, false>)", 40, 272, 192, false, 1 },
        { "decode, M 128 all fragments, block scales with a zero block",
          "(r4d_gemm_m2xfp4a8_decode_kernel<128, 1, 8, false, true>)", 128, 128, 128, true, 1 },
        { "tiled, TN 2 just past the decode band, wide slab",
          "(r4d_gemm_m2xfp4a8_tiled_kernel<2, 128, false>)", 129, 64, 128, false, 0 },
        { "tiled, TN 2, K 192 takes the 64 slab",
          "(r4d_gemm_m2xfp4a8_tiled_kernel<2, 64, false>)", 300, 64, 192, false, 0 },
        { "tiled, TN 2, block scales", "(r4d_gemm_m2xfp4a8_tiled_kernel<2, 128, true>)",
          256, 128, 128, true, 0 },
    };
    int fails = 0, cases = 0;
    for (const Case& c : all) {
        std::mt19937 rng(2000 + cases);
        std::normal_distribution<double> nd(0.0, 1.0);
        const int nkb = (c.K + 127) / 128;
        /* random f32 weights -> the m2xfp4 planes */
        std::vector<float> wf((size_t)c.N * c.K);
        for (auto& v : wf) v = (float)(3.5 * nd(rng));
        const M2Weight m = m2_quantise(wf, c.N, c.K, rng);
        std::vector<uint8_t> wq, ws, wm, wref;
        m2_store(m, c.N, c.K, &wq, &ws, &wm, &wref);
        /* fp8 activations and their grid. The bytes are the quantised values; the grid's scale is
         * applied once, where the kernel applies it: per row in the epilogue, per (row, 128) at
         * each boundary -- the reference multiplies the same way. */
        const int PA = c.ablk ? nkb : 1;
        std::vector<uint8_t> a((size_t)c.M * c.K);
        std::vector<float> as((size_t)c.M * PA);
        for (int i = 0; i < c.M; ++i)
            for (int q = 0; q < PA; ++q) {
                /* block 1 of every fourth row is zero: the carried-scale path */
                const float s = c.ablk && i % 4 == 1 && q == 1
                                    ? 0.0f
                                    : std::ldexp(1.0f, (int)((rng() % 7) - 3)) * (1.0f + (rng() % 97) / 97.0f);
                as[(size_t)i * PA + q] = s;
            }
        for (int i = 0; i < c.M; ++i)
            for (int k = 0; k < c.K; ++k) {
                const float block_scale = as[(size_t)i * PA + (c.ablk ? k / 128 : 0)];
                const float v = (float)(nd(rng) * 2.0);
                /* a zero-scale block's activation bytes are zero (the block-scaled quantisers
                 * produce nothing there), and the kernels carry the previous scale through */
                a[(size_t)i * c.K + k] =
                    block_scale == 0.0f ? 0 : rne_e4m3(v / block_scale);
            }
        /* the f64 reference: the activation's bytes (times the block grid, when there is one) and
         * the exact weight decode, which already carries its 2^(scale - 127) */
        std::vector<double> reff((size_t)c.M * c.N);
        for (int i = 0; i < c.M; ++i)
            for (int n = 0; n < c.N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < c.K; ++k)
                    acc += e4m3_f64(a[(size_t)i * c.K + k]) *
                           (c.ablk ? (double)as[(size_t)i * PA + k / 128] : 1.0) *
                           m.exact[(size_t)n * c.K + k];
                if (!c.ablk) acc *= (double)as[(size_t)i];
                reff[(size_t)i * c.N + n] = acc;
            }
        /* launch. The scratch is sized for the widest split the band's rule can pick (16), which
         * the ABI's scratch hook bounds the same way; the arrival counters are the unit's own,
         * one block per output buffer. */
        std::vector<uint16_t> got((size_t)c.M * c.N, 0x7777);
        const long wq_ptr = (long)(uintptr_t)wq.data();
        const long scratch_bytes = 16l * c.M * c.N * (long)sizeof(float);
        std::vector<float> scratch((size_t)(scratch_bytes / sizeof(float)), 0.0f);
        emu_last_kernel = "";
        ++cases;
        if (std::strstr(c.kernel, "tiled")) {
            std::vector<uint8_t> at((size_t)r4d_m2xfp4a8_tile_a_bytes(c.M, c.K));
            r4d_m2xfp4a8_tile_a((long)(uintptr_t)a.data(), (long)(uintptr_t)at.data(), c.M, c.K,
                                0, 0);
            r4d_gemm_m2xfp4a8_tiled((long)(uintptr_t)at.data(), (long)(uintptr_t)as.data(),
                                    wq_ptr, (long)(uintptr_t)ws.data(), (long)(uintptr_t)wm.data(),
                                    (long)(uintptr_t)wref.data(), (long)(uintptr_t)got.data(),
                                    c.M, c.K, c.N, 0, c.ablk ? 128 : 0, PA,
                                    0, c.N, c.N);
        } else {
            r4d_gemm_m2xfp4a8_decode((long)(uintptr_t)a.data(), (long)(uintptr_t)as.data(),
                                     wq_ptr, (long)(uintptr_t)ws.data(), (long)(uintptr_t)wm.data(),
                                     (long)(uintptr_t)wref.data(), (long)(uintptr_t)got.data(),
                                     c.M, c.K, c.N, c.ks, 0, 0, c.ablk ? 128 : 0, PA,
                                     (long)(uintptr_t)scratch.data(), scratch_bytes, 0, 0, c.N,
                                     c.N);
        }
        /* every launch lands on one form; a moved bound shows by name */
        if (!std::strstr(emu_last_kernel, c.kernel)) {
            std::printf("FAIL %s: ran %s, want %s\n", c.name,
                        emu_last_kernel ? emu_last_kernel : "(nothing)", c.kernel);
            ++fails;
            continue;
        }
        double num = 0, den = 0, worst = 0;
        for (size_t i = 0; i < reff.size(); ++i) {
            const double d = bf16_f64(got[i]) - reff[i];
            num += d * d;
            den += reff[i] * reff[i];
            worst = std::max(worst, std::fabs(d) / (std::fabs(reff[i]) + 1e-3));
        }
        const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
        /* The FP8 WMMA's tolerance, plus the table's accepted k = 3 rounding: the mxfp4 row's
         * 1e-2 with the ~1.2% whole-dot contribution a full rotation of k = 3 subgroups adds
         * (a quarter of the weights at up to 2^-4.4 each). Cancellation can make a single
         * element's relative error unbounded, so the aggregate is what is held. */
        if (rel >= 4e-2) {
            std::printf("FAIL %s: rel_l2 %.3e, worst %.3e\n", c.name, rel, worst);
            ++fails;
        }
    }
    std::printf("%d gemm case(s), %d failure(s)\n", cases, fails);
    return fails;
}

int main() {
    int fails = check_lut();
    fails += check_unpack();
    fails += run_cases();
    if (fails) std::printf("FAIL m2xfp4 (%d)\n", fails);
    return fails ? 1 : 0;
}
