// Real AQL GEMM checks across registry, grid and 32-bit activation-offset boundaries.
// Sparse dyadic inputs have an exact independent four-term oracle, including split-K.
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_device.h"
#include "device/device.h"
#include "plugin/registry.h"
#include "rad_plugin.h"
#include <dlfcn.h>
#include <array>
#include <climits>

namespace {
bool device() {
    RadDeviceProps p{};
    return rad_dev_props(0, &p) == RAD_OK && !p.is_host_backend && rad_dev_set(0) == RAD_OK;
}
struct Library {
    void* h;
    Library() {
        const char* home = std::getenv("RADIANCE_HOME");
        h = dlopen((std::string(home ? home : "") + "/kernels/libr4d.so").c_str(),
                   RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(h); }
    const RadKernelInfo* row(const char* name) {
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        for (int i = 0; i < count(); ++i)
            if (!std::strcmp(at(i)->name, name)) return at(i);
        throw std::runtime_error(name);
    }
};
RadTensor tensor(void* p, int64_t rows, int64_t cols) {
    RadTensor t{};
    t.data = p;
    t.dtype = RAD_BF16;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = cols;
    t.stride[1] = 1;
    return t;
}
RadParam param(const char* k, int v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
float av(int64_t r, int j) {
    return (float)((r * (j + 1) + 3 * j) % 29 - 14) / 16;
}
float wv(int n, int j) {
    return (float)((n * (j + 2) + j) % 17 - 8) / 8;
}
void run(int M, int N, int K, const std::vector<std::array<int, 4>>& configs) {
    Library lib;
    RadStream s = nullptr;
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const size_t abytes = (size_t)M * K * 2, ybytes = (size_t)M * N * 2;
    void* ap = rad_dev_alloc(abytes + 256, RAD_MEM_DEVICE);
    void* wp = rad_dev_alloc((size_t)N * K * 2, RAD_MEM_DEVICE);
    void* yp = rad_dev_alloc(ybytes + 256, RAD_MEM_DEVICE);
    REQUIRE(ap && wp && yp);
    std::vector<uint8_t> guard(128, 0x5a);
    for (auto pair : {std::pair<void*, size_t>{ap, abytes}, {yp, ybytes}}) {
        REQUIRE_EQ(rad_memcpy_async(pair.first, guard.data(), 128, s), RAD_OK);
        REQUIRE_EQ(rad_memcpy_async((uint8_t*)pair.first + 128 + pair.second, guard.data(), 128, s),
                   RAD_OK);
    }
    const int indices[] = {0, 1, 32, K - 1};
    constexpr int block = 64;
    std::vector<uint16_t> a((size_t)block * K, 0), w((size_t)N * K, 0);
    for (int n = 0; n < N; ++n)
        for (int j = 0; j < 4; ++j)
            w[(size_t)n * K + indices[j]] = rad_f32_to_bf16(wv(n, j));
    REQUIRE_EQ(rad_memcpy_async(wp, w.data(), w.size() * 2, s), RAD_OK);
    for (int64_t r = 0; r < M; r += block) {
        int rows = (int)std::min<int64_t>(block, M - r);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < 4; ++j)
                a[(size_t)i * K + indices[j]] = rad_f32_to_bf16(av(r + i, j));
        REQUIRE_EQ(
            rad_memcpy_async((uint8_t*)ap + 128 + r * K * 2, a.data(), (size_t)rows * K * 2, s),
            RAD_OK);
        REQUIRE_EQ(rad_stream_sync(s),
                   RAD_OK); // keep the pageable staging buffer alive until consumed
    }
    std::vector<uint16_t> got((size_t)M * N), expected(got.size());
    for (int64_t r = 0; r < M; ++r)
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int j = 0; j < 4; ++j)
                sum += av(r, j) * wv(n, j);
            expected[(size_t)r * N + n] = rad_f32_to_bf16(sum);
        }
    RadTensor t[] = {tensor((uint8_t*)ap + 128, M, K), tensor(wp, N, K),
                     tensor((uint8_t*)yp + 128, M, N)};
    for (auto c : configs) {
        // c = tiled variant (0 for skinny), split count, row tiles, column waves.
        const auto* k = lib.row(c[0] ? "gemm_bf16_nt_tiled" : "gemm_bf16_nt_m64");
        RadParam p[] = {param("sk", c[1]), param("mb", c[2]), param("wv", c[3]),
                        param("tile", c[0])};
        RadArgs args{};
        args.t = t;
        args.n_t = 3;
        args.p = p;
        args.n_p = 4;
        REQUIRE_EQ(k->launch(&args, s), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        REQUIRE_EQ(rad_memcpy_async(got.data(), t[2].data, ybytes, s), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        size_t bad = 0;
        while (bad < got.size() && got[bad] == expected[bad])
            ++bad;
        fprintf(stderr, "    %s M=%d N=%d K=%d tile=%d sk=%d mb=%d wv=%d first_mismatch=%zu/%zu\n",
                k->name, M, N, K, c[0], c[1], c[2], c[3], bad, got.size());
        CHECK_EQ(bad, got.size());
        for (auto pair : {std::pair<void*, size_t>{ap, abytes}, {yp, ybytes}}) {
            std::vector<uint8_t> before(128), after(128);
            REQUIRE_EQ(rad_memcpy_async(before.data(), pair.first, 128, s), RAD_OK);
            REQUIRE_EQ(
                rad_memcpy_async(after.data(), (uint8_t*)pair.first + 128 + pair.second, 128, s),
                RAD_OK);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
            CHECK(before == guard);
            CHECK(after == guard);
        }
    }
    rad_dev_free(ap, RAD_MEM_DEVICE);
    rad_dev_free(wp, RAD_MEM_DEVICE);
    rad_dev_free(yp, RAD_MEM_DEVICE);
    rad_stream_destroy(s);
}
}
TEST(bf16_capacity_preserves_existing_kernel_selection) {
    if (!device()) SKIP("requires real device");
    rad::Registry r;
    REQUIRE_EQ(r.load_dir(home() + "/kernels", {"libr4d", "libref"}), RAD_OK);
    // Nomic/Qwen projection widths, a narrow output, and a vision-style K tail. Enlarging
    // the row range must preserve the old small/prefill ladder, including its seam values.
    struct Case { int m, n, k; const char* name; };
    const Case cases[] = {
        {1, 2304, 768, "gemm_bf16_nt_m16"},
        {16, 3584, 1024, "gemm_bf16_nt_m16"},
        {17, 3584, 1024, "gemm_bf16_nt_m64"},
        {64, 2304, 768, "gemm_bf16_nt_m64"},
        {65, 2304, 768, "gemm_bf16_nt_tiled"},
        {511, 768, 3072, "gemm_bf16_nt_tiled"},
        {65536, 1024, 3584, "gemm_bf16_nt_tiled"},
        {65, 17, 768, "gemm_bf16_nt_m64"},
        {65, 768, 1040, "gemm_bf16_nt_m64"},
    };
    for (const auto& c : cases) {
        rad::Geometry g;
        g.set_i("M", c.m);
        g.set_i("N", c.n);
        g.set_i("K", c.k);
        g.set_s("dtype", "bf16");
        const auto* row = r.select("gemm_nt", g, RAD_DOMAIN_DEVICE);
        REQUIRE(row);
        CHECK_EQ(std::string(row->info->name), c.name);
    }
}
TEST(bf16_capacity_old_boundary_and_ragged_tails) {
    if (!device()) SKIP("requires real device");
    for (int m : {65536, 65537, 131073})
        run(m, 129, 64, {{0, 1, 1, 1}, {0, 4, 3, 2}, {1, 1, 1, 1}, {2, 4, 1, 1}});
}
TEST(bf16_capacity_grid_row_rebase) {
    if (!device()) SKIP("requires real device");
    run(65535 * 16 + 1, 17, 64, {{0, 1, 1, 1}});
}
TEST(bf16_capacity_unsigned_activation_offset_rebase) {
    if (!device()) SKIP("requires real device");
    // >2^32 BF16 elements (~8 GiB). Host staging stays bounded at 8 MiB.
    run(65539, 129, 65536, {{0, 4, 1, 1}, {1, 4, 1, 1}, {2, 1, 1, 1}});
}
TEST(bf16_capacity_preserves_chained_decode_dependencies) {
    if (!device()) SKIP("requires real device");
    Library lib;
    const auto* k = lib.row("gemm_bf16_nt_m64");
    constexpr int m = 17, n = 64;
    std::vector<uint16_t> x(m * n), w(n * n, 0), got(m * n);
    for (int i = 0; i < m * n; ++i)
        x[i] = rad_f32_to_bf16((float)(i % 23 - 11) / 16);
    for (int i = 0; i < n; ++i)
        w[i * n + i] = rad_f32_to_bf16(2.f);
    void* xp = rad_dev_alloc(x.size() * 2, RAD_MEM_DEVICE);
    void* wp = rad_dev_alloc(w.size() * 2, RAD_MEM_DEVICE);
    void* yp = rad_dev_alloc(x.size() * 2, RAD_MEM_DEVICE);
    void* zp = rad_dev_alloc(x.size() * 2, RAD_MEM_DEVICE);
    REQUIRE(xp && wp && yp && zp);
    RadStream stream = nullptr;
    REQUIRE_EQ(rad_stream_create(&stream, 0), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(xp, x.data(), x.size() * 2, stream), RAD_OK);
    REQUIRE_EQ(rad_memcpy_async(wp, w.data(), w.size() * 2, stream), RAD_OK);
    RadParam p[] = {param("chain", 1), param("mb", 1), param("sk", 4), param("wv", 1)};
    RadTensor first[] = {tensor(xp, m, n), tensor(wp, n, n), tensor(yp, m, n)};
    RadTensor second[] = {tensor(yp, m, n), tensor(wp, n, n), tensor(zp, m, n)};
    RadArgs a{};
    a.n_t = 3;
    a.p = p;
    a.n_p = 4;
    rad_dev_chain_infer(1);
    REQUIRE_EQ(rad_dev_chain_enabled(), 1);
    for (int repeat = 0; repeat < 8; ++repeat) {
        a.t = first;
        REQUIRE_EQ(k->launch(&a, stream), RAD_OK);
        a.t = second;
        REQUIRE_EQ(k->launch(&a, stream), RAD_OK);
    }
    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
    rad_dev_chain_infer(0);
    REQUIRE_EQ(rad_memcpy_async(got.data(), zp, got.size() * 2, stream), RAD_OK);
    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
    for (size_t i = 0; i < got.size(); ++i)
        CHECK_EQ(got[i], rad_f32_to_bf16(4 * rad_bf16_to_f32(x[i])));
    for (void* v : {xp, wp, yp, zp})
        rad_dev_free(v, RAD_MEM_DEVICE);
    rad_stream_destroy(stream);
}
TEST(bf16_capacity_invalid_and_multilaunch_chain_fail_before_issue) {
    if (!device()) SKIP("requires real device");
    Library lib;
    RadTensor t[] = {tensor((void*)0x1000, 65539, 65536), tensor((void*)0x2000, 129, 65536),
                     tensor((void*)0x3000, 65539, 129)};
    RadParam p[] = {param("chain", 1), param("mb", 1), param("sk", 1), param("wv", 1)};
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 4;
    CHECK_EQ(lib.row("gemm_bf16_nt_m64")->launch(&a, nullptr), RAD_E_UNSUPPORTED);
    p[0].ival = 0;
    t[0].shape[0] = t[2].shape[0] = (int64_t)INT_MAX + 1;
    CHECK_EQ(lib.row("gemm_bf16_nt_m64")->launch(&a, nullptr), RAD_E_SHAPE);
    CHECK_EQ(lib.row("gemm_bf16_nt_tiled")->launch(&a, nullptr), RAD_E_SHAPE);
    t[0] = tensor((void*)0x1000, 65, 65536);
    t[2] = tensor((void*)0x3000, 65, 65536);
    CHECK_EQ(lib.row("gemm_bf16_nt_m64")->launch(&a, nullptr), RAD_E_SHAPE); // W span exceeds u32
    CHECK_EQ(lib.row("gemm_bf16_nt_tiled")->launch(&a, nullptr), RAD_E_SHAPE);
}
RAD_TEST_MAIN()
