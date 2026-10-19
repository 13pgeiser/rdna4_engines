/* tests/r4d_emu: the MXFP4 expert GEMM (r4d_moe_mxfp4.hip) on the host SIMT emulator. Every form
 * the RADIANCE_MXFP4_* switches reach must match the arithmetic-decode row-group form bit for bit,
 * and both a host restatement of the per-element sequence: per 128 of K in order, per 32-wide group
 * a fresh 32-product sum in k order (the emulated WMMA pair), folded as raw * activation scale *
 * E8M0 scale with two roundings and an add. Runs of 0, 1, 15, 16, 17, 63, 64, 65 and 130 rows,
 * dropped slots inside runs and past the routed rows, columns an odd expert does not have, a
 * shorter K for odd experts, E8M0 codes 0 and 255, parts 2, token and sorted rows, stacked planes
 * and pointer tables. The P64 forms (RADIANCE_MXFP4_PREFILL_P64) run at every BM, BN, stage and
 * column-tile count, pinned below their row bound, with jobs cut at 32 and 64 rows, column
 * blocks past N and the output through LDS at aligned and odd pitches. The table decode is also
 * held to the arithmetic one over every code. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_mxfp4.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern const char* emu_last_kernel;

/* The arithmetic decode the kernels used before the table (r4d_moe_mxfp4.hip, fp4x4 and its
 * byte interleave), restated as the reference of r4d_mxfp4_unpack8_exact. */
static uint32_t fp4x4_ref(uint32_t n) {
    const uint32_t mag = n & 0x07070707u, flag = (mag + 0x06060606u) & 0x08080808u;
    const uint32_t mask = (flag << 5) - (flag >> 3);
    return (((mag + 0x0c0c0c0cu) << 2) & mask) | (((mag & 0x01010101u) * 0x30u) & ~mask) |
           ((n & 0x08080808u) << 4);
}
static int check_decode() {
    int fails = 0;
    auto one = [&](uint32_t w) {
        const uint32_t low = fp4x4_ref(w & 0x0f0f0f0fu), high = fp4x4_ref((w >> 4) & 0x0f0f0f0fu);
        const uint32_t lo_ref = __byte_perm(low, high, 0x5140),
                       hi_ref = __byte_perm(low, high, 0x7362);
        uint32_t lo, hi, tlo, thi;
        r4d_mxfp4_unpack8_exact(w, &lo, &hi);
        r4d_mxfp4_unpack8(w, 0, &tlo, &thi);
        if (lo != lo_ref || hi != hi_ref || lo != tlo || hi != thi) {
            if (fails < 4)
                std::printf("FAIL decode of %08x: table %08x %08x, arithmetic %08x %08x\n", w, hi,
                            lo, hi_ref, lo_ref);
            ++fails;
        }
        /* each byte against the e2m1 value it encodes */
        for (int e = 0; e < 8; ++e) {
            const uint32_t c = (w >> (4 * e)) & 15u;
            static const float mag[8] = { 0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
            const float want = (c & 8u ? -1.f : 1.f) * mag[c & 7u];
            const uint8_t b = (uint8_t)(((e < 4 ? lo : hi) >> (8 * (e & 3))) & 0xffu);
            if (rad_fp8e4m3_to_f32(b) != want || (b & 0x80u) != (c & 8u ? 0x80u : 0u)) {
                if (fails < 4) std::printf("FAIL code %x at element %d -> byte %02x\n", c, e, b);
                ++fails;
            }
        }
    };
    /* the bytes decode independently: every pair of adjacent bytes exhaustively, at both halves
     * of the word, and random words over all four */
    for (uint32_t v = 0; v < 65536; ++v) {
        one(v);
        one(v << 16);
        one(v << 8);
    }
    std::mt19937 rng(7);
    for (int i = 0; i < 200000; ++i) one((uint32_t)rng());
    return fails;
}

struct Case {
    const char* name;
    int N, K, N_odd, K_odd, parts, top;
    bool sorted_a, tables, exceptional;
    std::vector<int> runs;
    int tail;           // dropped slots past the routed rows
};

/* Byte equality with every bf16 NaN taken as one value. On x86 inf - inf gives the negative
 * default NaN, and an add of two NaNs returns its first operand, whose side the compiler may swap;
 * the card generates the positive quiet NaN, so a NaN's sign carries no information here. */
static bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }
static size_t first_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i] && !(nan16(a[i]) && nan16(b[i]))) return i;
    return a.size() == b.size() ? a.size() : std::min(a.size(), b.size());
}
static bool same16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    return a.size() == b.size() && first_diff(a, b) == a.size();
}

static RadTensor ten(void* p, uint32_t dt, std::vector<int64_t> shape,
                     std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        t.shape[i] = shape[i];
        t.stride[i] = stride[i];
    }
    return t;
}

int main() {
    int fails = check_decode(), cases = 1;
    if (fails) std::printf("FAIL table decode: %d mismatches\n", fails);

    const std::vector<int> runs9 = { 0, 1, 15, 16, 17, 63, 64, 65, 130, 141 };
    const Case all[] = {
        { "gate_up-like, token rows", 128, 384, 128, 384, 2, 4, false, false, false, runs9, 6 },
        { "odd experts narrower and shorter", 128, 384, 64, 256, 2, 4, false, true, false, runs9,
          2 },
        { "down-like, sorted rows", 64, 256, 64, 256, 1, 2, true, true, false, runs9, 4 },
        { "exceptional E8M0 codes", 64, 256, 64, 256, 2, 4, false, false, true, runs9, 4 },
        { "decode band, row groups of 4",
          64,
          384,
          64,
          384,
          2,
          2,
          false,
          true,
          false,
          { 0, 1, 15, 16, 17, 30 },
          3 },
        { "decode band, row groups of 1",
          64,
          256,
          32,
          256,
          1,
          2,
          true,
          false,
          true,
          { 1, 0, 3, 15, 1 },
          2 },
        { "P64 jobs at 32 and 64 rows, N under a block",
          96,
          512,
          96,
          384,
          1,
          2,
          false,
          false,
          false,
          { 32, 33, 31, 96, 128, 129, 0, 2 },
          9 },
        { "P64 resident A, column tiles cut",
          192,
          256,
          128,
          128,
          2,
          4,
          true,
          true,
          true,
          { 64, 63, 65, 1, 0, 127, 17 },
          5 },
    };
    struct Form {
        const char* env;
        int stage, tiles, lut;
        const char* kernel;
        bool prefill;
        bool pipe = false;
    };
    /* env: DECODE_EXPERTS, ALIGNED_JOBS, PREFILL_N32, PREFILL_M32, PREFILL_N64, PREFILL_CACHE_ROWS */
    const Form forms[] = {
        { "100000", 256, 1, 0, "moe_mxfp4_expert_kernel<4, 256, false>", false },
        { "100000", 256, 1, 1, "moe_mxfp4_expert_kernel<4, 256, true>", false },
        { "100000", 256, 4, 0, "moe_mxfp4_expert_staged_kernel<256, 4, false>", false },
        { "100000", 256, 4, 1, "moe_mxfp4_expert_staged_kernel<256, 4, true>", false },
        { "100000", 128, 1, 0, "moe_mxfp4_expert_staged_kernel<128, 1, false>", false },
        { "100000", 128, 1, 1, "moe_mxfp4_expert_staged_kernel<128, 1, true>", false },
        { "100000", 128, 4, 0, "moe_mxfp4_expert_staged_kernel<128, 4, false>", false },
        { "100000", 128, 4, 1, "moe_mxfp4_expert_staged_kernel<128, 4, true>", false },
        { "000000", 0, 0, 1, "moe_mxfp4_fp8_kernel<1, 1, false, true>", false },
        { "010000", 0, 0, 1, "moe_mxfp4_fp8_kernel<1, 1, false, true>", true },
        { "001000", 0, 0, 1, "moe_mxfp4_fp8_kernel<2, 1, false, true>", true },
        { "011000", 0, 0, 1, "moe_mxfp4_fp8_kernel<2, 1, false, true>", true },
        { "011100", 0, 0, 1, "moe_mxfp4_fp8_kernel<2, 2, false, true>", true },
        { "011110", 0, 0, 1, "moe_mxfp4_fp8_kernel<4, 2, false, true>", true },
        { "011111", 0, 0, 0, "moe_mxfp4_fp8_kernel<4, 2, true, false>", true },
        { "011111", 0, 0, 1, "moe_mxfp4_fp8_kernel<4, 2, true, true>", true },
        { "011111", 0, 0, 0, "moe_mxfp4_prefill_kernel<false>", true, true },
        { "011111", 0, 0, 1, "moe_mxfp4_prefill_kernel<true>", true, true },
    };
    /* P64 at every raster point, column tiles 4 (the resident form's default) and 1, and the LDS
     * output at the default; the K of a case picks streamed or resident A. */
    struct P64 {
        int bm, bn, sk, ct, out, db = 0;
    };
    std::vector<P64> p64s;
    for (int bm : { 32, 64 })
        for (int bn : { 64, 128 })
            for (int sk : { 128, 256 })
                for (int ct : { 1, 4 }) p64s.push_back({ bm, bn, sk, ct, 0 });
    p64s.push_back({ 64, 128, 128, 4, 1 });
    p64s.push_back({ 32, 64, 256, 1, 1 });
    /* The double buffer at every BM x BN x stage: where two stages do not fit in LDS, and with
     * resident A, the single-buffer form runs. */
    for (int bm : { 32, 64 })
        for (int bn : { 64, 128 })
            for (int sk : { 128, 256 }) p64s.push_back({ bm, bn, sk, 4, 0, 1 });
    p64s.push_back({ 64, 128, 128, 4, 1, 1 });
    static const char* const kP64[] = { "RADIANCE_MXFP4_PREFILL_P64",
                                        "RADIANCE_MXFP4_PREFILL_P64_MIN_T_GATE_UP",
                                        "RADIANCE_MXFP4_PREFILL_P64_MIN_T_DOWN",
                                        "RADIANCE_MXFP4_PREFILL_P64_BM",
                                        "RADIANCE_MXFP4_PREFILL_P64_BN",
                                        "RADIANCE_MXFP4_PREFILL_P64_STAGE_K",
                                        "RADIANCE_MXFP4_PREFILL_P64_COLTILES",
                                        "RADIANCE_MXFP4_PREFILL_P64_STAGE_OUT",
                                        "RADIANCE_MXFP4_PREFILL_P64_T_BM64",
                                        "RADIANCE_MXFP4_PREFILL_P64_T_BN128",
                                        "RADIANCE_MXFP4_PREFILL_P64_T_DOWN_WIDE",
                                        "RADIANCE_MXFP4_PREFILL_P64_DB" };
    /* The rule's tile with nothing pinned, by its row bounds (unset: this test's few rows are
     * under all of them); the widest tiles take the LDS output. */
    struct Rule {
        const char* t_bm64;
        const char* t_bn128;
        const char* t_down;
        const char* streamed;
        const char* resident;
    };
    const Rule rules[] = { { nullptr, nullptr, nullptr, "<32, 64, 256, 1, false>",
                             "<32, 64, 128, 4, true>" },
                           { "0", nullptr, nullptr, "<64, 64, 256, 1, false>", "<32, 64, 128, 4, true>" },
                           { "0", "0", "0", "<64, 128, 128, 1, false>", "<64, 128, 128, 4, true>" } };
    static const char* const kEnv[] = {
        "RADIANCE_MXFP4_DECODE_EXPERTS", "RADIANCE_MXFP4_ALIGNED_JOBS",
        "RADIANCE_MXFP4_PREFILL_N32",    "RADIANCE_MXFP4_PREFILL_M32",
        "RADIANCE_MXFP4_PREFILL_N64",    "RADIANCE_MXFP4_PREFILL_CACHE_ROWS"
    };

    for (const Case& c : all) {
        const int E = (int)c.runs.size(), NK = std::max(c.K, c.K_odd), OUT = std::max(c.N, c.N_odd);
        int routed = 0;
        for (int r : c.runs) routed += r;
        const int T = (routed + c.tail + c.top - 1) / c.top * c.top;
        // An odd pitch on every other case: the LDS output's 16-byte stores need an aligned row.
        const int Arows = c.sorted_a ? T : T / c.top, ap = NK + 16, asp = NK / 128 + 1,
                  yp = OUT + ((&c - all) % 2 ? 5 : 8);
        std::mt19937 rng(1000 + cases);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> oct(-6.f, 6.f);
        std::vector<uint8_t> a((size_t)Arows * ap);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng) * 2.f);
        std::vector<float> as((size_t)Arows * asp);
        for (auto& v : as) v = std::exp2(oct(rng));
        std::vector<int> offsets(E + 1, 0), sorted(T, -1), owner(T, -1);
        for (int e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + c.runs[e];
        for (int e = 0; e < E; ++e)
            for (int r = offsets[e]; r < offsets[e + 1]; ++r) {
                owner[r] = e;
                // A dropped slot about every eleventh row, the first rows of some runs included.
                sorted[r] = (r * 13 + e) % 11 == 0 ? -1 : (int)(rng() % (uint32_t)T);
            }
        const int n[2] = { c.N, c.N_odd }, k[2] = { c.K, c.K_odd },
                  nexp[2] = { (E + 1) / 2, E / 2 };
        std::vector<uint8_t> w[2], s[2];
        std::vector<const void*> wt[2], st[2];
        for (int q = 0; q < 2; ++q) {
            w[q].resize((size_t)nexp[q] * n[q] * k[q] / 2);
            s[q].resize((size_t)nexp[q] * n[q] * k[q] / 32);
            for (auto& v : w[q]) v = (uint8_t)rng();
            static const uint8_t special[] = { 0, 255, 1, 254, 127 };
            for (size_t i = 0; i < s[q].size(); ++i)
                s[q][i] = c.exceptional && rng() % 9 == 0 ? special[rng() % 5]
                                                          : (uint8_t)(118 + rng() % 16);
            for (int x = 0; x < nexp[q]; ++x) {
                wt[q].push_back(w[q].data() + (size_t)x * n[q] * k[q] / 2);
                st[q].push_back(s[q].data() + (size_t)x * n[q] * k[q] / 32);
            }
        }
        auto plane = [&](int q, bool scale) {
            const int64_t cols = scale ? k[q] / 32 : k[q];
            void* p = c.tables ? (void*)(scale ? st[q].data() : wt[q].data())
                               : (void*)(scale ? s[q].data() : w[q].data());
            RadTensor t = ten(p, scale ? RAD_E8M0 : RAD_FP4E2M1, { nexp[q], n[q], cols },
                              { c.tables ? 0 : (int64_t)n[q] * cols, cols, 1 });
            return t;
        };

        /* the per-element sequence, on the host */
        std::vector<uint16_t> ref((size_t)T * yp, 0x7777);
        for (int r = 0; r < T; ++r)
            for (int col = 0; col < OUT; ++col) {
                uint16_t& out = ref[(size_t)r * yp + col];
                out = 0;
                if (owner[r] < 0 || sorted[r] < 0) continue;
                const int e = owner[r], q = e & 1, ar = c.sorted_a ? r : sorted[r] / c.top;
                const int pw = OUT / c.parts, cw = n[q] / c.parts, within = col % pw;
                const bool valid = within < cw;
                const int wn = (col / pw) * cw + within;
                const uint8_t* wr =
                    w[q].data() + (size_t)(e / 2) * n[q] * k[q] / 2 + (size_t)wn * k[q] / 2;
                const uint8_t* sr =
                    s[q].data() + (size_t)(e / 2) * n[q] * k[q] / 32 + (size_t)wn * k[q] / 32;
                float total = 0.f;
                for (int kb = 0; kb < k[q]; kb += 128)
                    for (int g = 0; g < 4; ++g) {
                        float raw = 0.f;
                        for (int kk = kb + g * 32; kk < kb + g * 32 + 32; ++kk) {
                            const uint32_t code = valid ? (wr[kk / 2] >> (4 * (kk & 1))) & 15u : 0u;
                            uint32_t lo, hi;
                            r4d_mxfp4_unpack8_exact(code, &lo, &hi);
                            raw += rad_fp8e4m3_to_f32(a[(size_t)ar * ap + kk]) *
                                   rad_fp8e4m3_to_f32((uint8_t)(lo & 0xffu));
                        }
                        const uint32_t x = valid ? sr[kb / 32 + g] : 127u;
                        uint32_t bits = x == 255 ? 0x7fc00000u : x ? x << 23 : 0x00400000u;
                        float scale;
                        std::memcpy(&scale, &bits, 4);
                        float t = raw * as[(size_t)ar * asp + kb / 128];
                        t = t * scale;
                        total = total + t;
                    }
                uint32_t u;
                std::memcpy(&u, &total, 4);
                out = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >>
                                 16);   // f32_to_bf16, r4d_common.h
            }

        RadTensor t[9] = { ten(a.data(), RAD_F8E4M3, { Arows, NK }, { ap, 1 }),
                           ten(as.data(), RAD_F32, { Arows, NK / 128 }, { asp, 1 }),
                           plane(0, false),
                           plane(0, true),
                           ten(sorted.data(), RAD_I32, { T }, { 1 }),
                           ten(offsets.data(), RAD_I32, { E + 1 }, { 1 }),
                           plane(1, false),
                           plane(1, true),
                           RadTensor{} };
        RadParam p[11]{};
        auto pi = [](const char* key, long long v) {
            RadParam x{};
            x.key = key;
            x.kind = RAD_P_INT;
            x.ival = v;
            return x;
        };
        auto ps = [](const char* key, const char* v) {
            RadParam x{};
            x.key = key;
            x.kind = RAD_P_STR;
            x.sval = v;
            return x;
        };
        p[0] = pi("M", T / c.top);
        p[1] = pi("N", c.N);
        p[2] = pi("K", c.K);
        p[3] = pi("N_odd", c.N_odd);
        p[4] = pi("K_odd", c.K_odd);
        p[5] = pi("n_expert", E);
        p[6] = pi("top_k", c.top);
        p[7] = pi("parts", c.parts);
        p[8] = pi("group", 128);
        p[9] = ps("a_order", c.sorted_a ? "sorted" : "token");
        p[10] = ps("dtype", "mxfp4a8");
        auto run = [&](const Form& f, std::vector<uint16_t>& y) {
            y.assign((size_t)T * yp, 0x7777);
            t[8] = ten(y.data(), RAD_BF16, { T, OUT }, { yp, 1 });
            for (int i = 0; i < 6; ++i) setenv(kEnv[i], f.env[i] == '1' ? "1" : "0", 1);
            if (f.stage) {
                setenv("RADIANCE_MXFP4_EXPERTS_STAGE_K", std::to_string(f.stage).c_str(), 1);
                setenv("RADIANCE_MXFP4_EXPERTS_COLTILES", std::to_string(f.tiles).c_str(), 1);
            }
            setenv("RADIANCE_MXFP4_LUT", f.lut ? "1" : "0", 1);
            setenv("RADIANCE_MXFP4_PREFILL_PIPE", f.pipe ? "1" : "0", 1);
            RadArgs ar{};
            ar.t = t;
            ar.n_t = 9;
            ar.p = p;
            ar.n_p = 11;
            emu_last_kernel = "";
            return r4d_rad_moe_mxfp4(&ar, nullptr);
        };
        auto run_p64 = [&](const P64& f, std::vector<uint16_t>& y) {
            const int v[8] = { 1, 0, 0, f.bm, f.bn, f.sk, f.ct, f.out };
            for (int i = 0; i < 8; ++i) setenv(kP64[i], std::to_string(v[i]).c_str(), 1);
            setenv(kP64[11], f.db ? "1" : "0", 1);
            const Form base = { "000000", 0, 0, 1, "", false };
            const int rc = run(base, y);
            setenv(kP64[0], "0", 1);
            return rc;
        };
        const Form alt = { "000000", 0, 0, 0, "moe_mxfp4_fp8_kernel<1, 1, false, false>", false };
        std::vector<uint16_t> legacy, got;
        ++cases;
        int rc = run(alt, legacy);
        if (rc != RAD_OK || !std::strstr(emu_last_kernel, alt.kernel)) {
            std::printf("FAIL %s: reference form rc %d kernel %s\n", c.name, rc, emu_last_kernel);
            ++fails;
            continue;
        }
        if (!same16(legacy, ref)) {
            const size_t i = first_diff(legacy, ref);
            std::printf("FAIL %s: reference form against the host sequence at row %zu col %zu: "
                        "%04x, want %04x\n",
                        c.name, i / yp, i % yp, legacy[i], ref[i]);
            ++fails;
        }
        int live = 0;
        for (uint16_t v : ref) live += v != 0 && v != 0x7777;
        if (live < T * OUT / 3) {
            std::printf("FAIL %s: output mostly zero\n", c.name);
            ++fails;
        }
        for (const Form& f : forms) {
            if (f.prefill && T < 512) continue;
            ++cases;
            rc = run(f, got);
            const bool reached = std::strstr(emu_last_kernel, f.kernel) != nullptr;
            if (rc != RAD_OK || !reached || !same16(got, legacy)) {
                const size_t i = first_diff(got, legacy);
                std::printf(
                    "FAIL %s: form %s stage %d tiles %d lut %d ran %s (want %s), rc %d, first "
                    "word %zu\n",
                    c.name, f.env, f.stage, f.tiles, f.lut, emu_last_kernel, f.kernel, rc, i);
                ++fails;
            }
        }
        const bool resident = NK <= 256;
        for (const Rule& u : rules) {
            ++cases;
            for (const char* v : kP64) unsetenv(v);
            setenv(kP64[0], "1", 1);
            setenv(kP64[1], "0", 1);
            setenv(kP64[2], "0", 1);
            if (u.t_bm64) setenv(kP64[8], u.t_bm64, 1);
            if (u.t_bn128) setenv(kP64[9], u.t_bn128, 1);
            if (u.t_down) setenv(kP64[10], u.t_down, 1);
            const Form base = { "000000", 0, 0, 1, "", false };
            rc = run(base, got);
            for (const char* v : kP64) unsetenv(v);
            const std::string want =
                std::string("moe_mxfp4_p64_kernel") + (resident ? u.resident : u.streamed);
            if (rc != RAD_OK || !std::strstr(emu_last_kernel, want.c_str()) ||
                !same16(got, legacy)) {
                std::printf("FAIL %s: P64 rule ran %s (want %s), rc %d, first word %zu\n", c.name,
                            emu_last_kernel, want.c_str(), rc, first_diff(got, legacy));
                ++fails;
            }
        }
        for (const P64& f : p64s) {
            ++cases;
            rc = run_p64(f, got);
            const bool db = f.db && !resident && (f.sk == 128 || (f.bm == 32 && f.bn == 64));
            const std::string want = "moe_mxfp4_p64_kernel<" + std::to_string(f.bm) + ", " +
                                     std::to_string(f.bn) + ", " + std::to_string(f.sk) + ", " +
                                     (resident ? std::to_string(f.ct) : "1") + ", " +
                                     (resident ? "true" : "false") + (db ? ", true>" : ">");
            const bool reached = std::strstr(emu_last_kernel, want.c_str()) != nullptr;
            if (rc != RAD_OK || !reached || !same16(got, legacy)) {
                const size_t i = first_diff(got, legacy);
                std::printf("FAIL %s: P64 bm %d bn %d stage %d tiles %d out %d db %d ran %s (want "
                            "%s), rc %d, "
                            "first word %zu (row %zu col %zu: %04x, want %04x)\n",
                            c.name, f.bm, f.bn, f.sk, f.ct, f.out, f.db, emu_last_kernel,
                            want.c_str(), rc, i, i / yp, i % yp, i < got.size() ? got[i] : 0,
                            i < legacy.size() ? legacy[i] : 0);
                ++fails;
            }
        }
    }
    for (const char* v : kEnv) unsetenv(v);
    for (const char* v : kP64) unsetenv(v);
    unsetenv("RADIANCE_MXFP4_EXPERTS_STAGE_K");
    unsetenv("RADIANCE_MXFP4_EXPERTS_COLTILES");
    unsetenv("RADIANCE_MXFP4_LUT");
    unsetenv("RADIANCE_MXFP4_PREFILL_PIPE");
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
