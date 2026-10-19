/* tests/r4d_emu: the dense E4M3 GEMM with F32 block scales (r4d_fp8_f32scale.hip) on the host SIMT
 * emulator. The form the rule picks at each shape (the decode form at one and two column tiles a
 * wave and one to eight waves a block, the 16- and 32-row tiles) must match the legacy 16x64 tile,
 * which serves an activation pitch off 16 bytes, bit for bit: the same fragments, the same mma
 * chain per 128-K block and the same fold order. Ragged row counts, pitched rows and a K that
 * leaves the decode form's last wave group partly idle are covered; the output's pitch padding
 * stays untouched. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

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
        const char* kernel;
    };
    const char* const dec1 = "(fp8_f32scale_decode<1, 1>)";
    const char* const dec2 = "(fp8_f32scale_decode<1, 2>)";
    const char* const rows = "fp8_f32scale_rows<R>";
    // Decode at eight, four, two and one waves a block (K blocks 12/32, 4, 2, 1), at two column
    // tiles a wave from 17 rows over N 1024 and 640; the 16-row tile, and the 32-row one at N 4096.
    const Shape shapes[] = { { 1, 256, 512, dec1 },   { 6, 128, 1536, dec1 },
                             { 17, 256, 512, dec1 },  { 48, 128, 512, dec1 },
                             { 2, 128, 256, dec1 },   { 5, 256, 128, dec1 },
                             { 3, 128, 4096, dec1 },  { 40, 1024, 512, dec2 },
                             { 20, 640, 256, dec2 },  { 70, 128, 384, rows },
                             { 130, 128, 256, rows }, { 33, 4096, 512, rows },
                             { 70, 4096, 128, rows } };
    int fails = 0, cases = 0;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K, kb = K / 128, ap = K + 16, lp = K + 8, yp = N + 8;
        std::vector<uint8_t> a((size_t)M * ap), al((size_t)M * lp), w((size_t)N * K);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng));
        for (int r = 0; r < M; ++r) std::memcpy(&al[(size_t)r * lp], &a[(size_t)r * ap], K);
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
        // The pitch off 16 bytes takes the legacy tile.
        auto run = [&](bool legacy, std::vector<uint16_t>& y) {
            y.assign((size_t)M * yp, 0x7777);
            t[0] = legacy ? mat(al.data(), RAD_F8E4M3, M, K, lp)
                          : mat(a.data(), RAD_F8E4M3, M, K, ap);
            t[4] = mat(y.data(), RAD_BF16, M, N, yp);
            RadArgs ar{};
            ar.t = t;
            ar.n_t = 7;
            ar.p = p;
            ar.n_p = 5;
            return r4d_rad_fp8_f32scale(&ar, nullptr);
        };
        std::vector<uint16_t> legacy, got;
        ++cases;
        if (run(true, legacy) != RAD_OK || std::strcmp(emu_last_kernel, "fp8_f32scale")) {
            std::printf("FAIL legacy M=%d N=%d K=%d ran %s\n", M, N, K, emu_last_kernel);
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
        ++cases;
        const int rc = run(false, got);
        if (rc != RAD_OK || std::strcmp(emu_last_kernel, sh.kernel) || got != legacy) {
            size_t i = 0;
            while (rc == RAD_OK && i < got.size() && got[i] == legacy[i]) ++i;
            std::printf("FAIL %s (want %s) at M=%d N=%d K=%d (rc %d, first word %zu)\n",
                        emu_last_kernel, sh.kernel, M, N, K, rc, i);
            ++fails;
        }
    }
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
