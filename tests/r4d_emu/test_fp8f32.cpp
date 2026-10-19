/* tests/r4d_emu: the dense E4M3 GEMM with F32 block scales (r4d_fp8_f32scale.hip) on the host SIMT
 * emulator. The decode and row-tile forms (RADIANCE_FP8F32_FORM, each R x S x C and row count) must
 * match the legacy 16x64 tile
 * bit for bit: the same fragments, the same mma chain per 128-K block and the same fold order.
 * Ragged row counts, pitched rows and a K that leaves the decode form's last wave group partly
 * idle are covered; the output's pitch padding stays untouched. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static RadTensor mat(void* p, uint32_t dt, int64_t rows, int64_t cols, int64_t ld) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = ld;
    t.stride[1] = 1;
    return t;
}

int main() {
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape {
        int M, N, K;
    };
    const Shape shapes[] = { { 1, 256, 512 },  { 6, 128, 1536 }, { 17, 256, 512 },
                             { 48, 128, 512 }, { 70, 128, 384 }, { 130, 128, 256 },
                             { 3, 128, 4096 } };
    const char* forms[] = { "decode",      "decode1x4",   "decode1x32",  "decode2x8",
                            "decode3x16",  "decode1x8x2", "decode1x4x4", "decode2x8x2",
                            "decode3x4x2", "decode2x4x4", "rows1",       "rows2",
                            "rows4",       nullptr };
    int fails = 0, cases = 0;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K, kb = K / 128, ap = K + 16, yp = N + 8;
        std::vector<uint8_t> a((size_t)M * ap), w((size_t)N * K);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng));
        for (auto& v : w) v = rad_f32_to_fp8e4m3(nd(rng) * 0.5f);
        std::vector<float> as((size_t)M * kb), ws((size_t)(N / 128) * kb);
        // Scales over twelve octaves: K blocks of very different weight cancel, so a changed
        // summation order shows through the bf16 output.
        std::uniform_real_distribution<float> oct(-6.f, 6.f);
        for (auto& v : as) v = std::exp2(oct(rng));
        for (auto& v : ws) v = std::exp2(oct(rng)) * 0.01f;
        RadTensor t[7] = { mat(a.data(), RAD_F8E4M3, M, K, ap),
                           mat(as.data(), RAD_F32, M, kb, kb),
                           mat(w.data(), RAD_F8E4M3, N, K, K),
                           mat(ws.data(), RAD_F32, N / 128, kb, kb),
                           RadTensor{},
                           RadTensor{},
                           RadTensor{} };
        RadParam p[5]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "N", RAD_P_INT, N };
        p[2] = { "K", RAD_P_INT, K };
        p[3] = { "group", RAD_P_INT, 128 };
        p[4].key = "dtype";
        p[4].kind = RAD_P_STR;
        p[4].sval = "fp8a8f32";
        auto run = [&](const char* form, std::vector<uint16_t>& y) {
            y.assign((size_t)M * yp, 0x7777);
            t[4] = mat(y.data(), RAD_BF16, M, N, yp);
            setenv("RADIANCE_FP8F32_FORM", form, 1);
            RadArgs ar{};
            ar.t = t;
            ar.n_t = 7;
            ar.p = p;
            ar.n_p = 5;
            return r4d_rad_fp8_f32scale(&ar, nullptr);
        };
        std::vector<uint16_t> legacy, got;
        ++cases;
        if (run("legacy", legacy) != RAD_OK) {
            std::printf("FAIL legacy M=%d N=%d K=%d\n", M, N, K);
            ++fails;
            continue;
        }
        int nonzero = 0;
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c) nonzero += legacy[(size_t)r * yp + c] != 0;
        if (nonzero < M * N / 2) {
            std::printf("FAIL legacy output mostly zero at M=%d\n", M);
            ++fails;
        }
        for (const char* const* f = forms; *f; ++f) {
            ++cases;
            const int rc = run(*f, got);
            if (rc != RAD_OK || got != legacy) {
                size_t i = 0;
                while (rc == RAD_OK && i < got.size() && got[i] == legacy[i]) ++i;
                std::printf("FAIL form %s at M=%d N=%d K=%d (rc %d, first word %zu)\n", *f, M, N, K,
                            rc, i);
                ++fails;
            }
        }
    }
    unsetenv("RADIANCE_FP8F32_FORM");
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
