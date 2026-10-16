/* lora_ops_test.cpp -- lora_shrink and lora_expand in libref and libavx, over one case matrix.
 *
 * The matrix and the checks are tests/r4d_emu/lora_cases.h's, the same ones the r4d emulator test
 * holds libr4d's kernels to: M in {1, 3, 17, 64, 130}, R in {16, 32, 48}, S = 4, K and N in
 * {64, 256, 1000}, `a` bf16 or E4M3 with a 128-column scale stream (and f16, f32, E5M2, i8 once),
 * `y` bf16, f32 (and f16), `a`/`y` as column slices, slots mixing -1, -5, 0..3 and S. Each library
 * is held separately to the same double-precision sums -- so libref and libavx agree with each
 * other to twice the bound -- and to the base-row contract by bytes: shrink leaves a sentinel NaN
 * in a base row of `t`, expand leaves a base row of `y` exactly as it was, -0 included.
 *
 * avx_check already compares libavx with libref at every ISA level on drawn geometries; what this
 * adds is the hand-built part a draw cannot reach: the -1 and -5 spellings of a base row (a draw
 * never produces a negative slot), the sentinel and -0 bytes, and wide row strides on `a`, its
 * scale and `y`. libavx is run at every level this machine executes.
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "r4d_emu/lora_cases.h"

#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

struct Lib {
    void* h = nullptr;
    lora_cases::LaunchFn shrink = nullptr, expand = nullptr;
    int (*isa_force)(int) = nullptr;
    const char* (*isa_name)(int) = nullptr;
};

Lib open_lib(const char* so) {
    Lib L;
    std::vector<std::string> paths;
    if (const char* home = std::getenv("RADIANCE_HOME"))
        paths.push_back(std::string(home) + "/kernels/" + so);
    paths.push_back(std::string("radiance_home/kernels/") + so);
    paths.push_back(std::string("build/radiance_home/kernels/") + so);
    for (auto& p : paths) {
        L.h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (L.h) break;
    }
    if (!L.h) return L;
    /* As the loader does: libavx builds its fp8 decode table here, and a launch before it reads
     * every scalar-path fp8 code as 0. */
    auto open = (int (*)(void))dlsym(L.h, "rad_plugin_open");
    if (open && open() < 0) return L;
    auto cnt = (int (*)(void))dlsym(L.h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(L.h, "rad_kernel_at");
    L.isa_force = (int (*)(int))dlsym(L.h, "avx_isa_force");
    L.isa_name = (const char* (*)(int))dlsym(L.h, "avx_isa_name");
    if (!cnt || !at) return L;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (!std::strcmp(k->op, "lora_shrink")) L.shrink = k->launch;
        if (!std::strcmp(k->op, "lora_expand")) L.expand = k->launch;
    }
    return L;
}

}  /* namespace */

TEST(lora_libref_and_libavx_over_the_matrix) {
    const Lib ref = open_lib("libref.so");
    REQUIRE(ref.shrink != nullptr);
    REQUIRE(ref.expand != nullptr);
    const Lib avx = open_lib("libavx.so");
    REQUIRE(avx.shrink != nullptr);
    REQUIRE(avx.expand != nullptr);
    /* Every ISA level the machine executes; a level the CPU lacks is clamped by avx_isa_force and
     * then not counted, as avx_check does. Without the symbol, the level libavx picked itself. */
    const int levels = avx.isa_force ? 4 : 1;
    for (int lvl = 0; lvl < levels; ++lvl) {
        if (avx.isa_force && avx.isa_force(lvl) != lvl) continue;
        std::vector<lora_cases::Impl> impls = {
            { "libref", ref.shrink, ref.expand, false },
            { "libavx", avx.shrink, avx.expand, false },
        };
        lora_cases::Tally T;
        lora_cases::run_matrix(T, impls, 11 + (unsigned)lvl);
        std::fprintf(stderr, "    libavx at %s: %d implementation-cases, %d failure(s); worst "
                     "|err|/tol shrink %.3f, expand %.3f\n",
                     avx.isa_name ? avx.isa_name(lvl) : "its own level", T.cases, T.fails,
                     T.worst_shrink, T.worst_expand);
        CHECK(T.cases > 0);
        CHECK_EQ(T.fails, 0);
    }
    if (avx.isa_force) avx.isa_force(3);           /* back to the highest the machine allows */
}

RAD_TEST_MAIN()
