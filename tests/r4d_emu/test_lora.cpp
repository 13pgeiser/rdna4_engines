/* tests/r4d_emu: lora_shrink / lora_expand (r4d_lora_bf16.hip) on the host SIMT emulator.
 *
 * The case matrix and every check are lora_cases.h's, shared with tests/lora_ops_test.cpp (libref and
 * libavx): base rows untouched by bytes, adapter rows against a double-precision sum within the
 * bounds stated there. Here the implementations are the emulated kernels, ref_host.cpp's
 * restatement of libref's arithmetic, and libref itself when $RADIANCE_HOME holds one (ctest sets
 * it) -- and the emulated EXPAND is held to the first of those bit for bit, because its
 * j-ascending sum and its roundings are libref's. Shrink sums k in 32 lane strides and a fixed
 * butterfly, so it is held to the bound rather than to libref's bytes.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "lora_cases.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using lora_cases::Impl;
using lora_cases::LaunchFn;

static int host_shrink(const RadArgs* a, RadStream) { return host_lora_shrink(a); }
static int host_expand(const RadArgs* a, RadStream) { return host_lora_expand(a); }

/* libref's two rows, through the plugin's own exports, or nulls. */
static void load_libref(LaunchFn* shrink, LaunchFn* expand) {
    *shrink = *expand = nullptr;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return;
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");      /* as the loader does */
    if (open && open() < 0) return;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (!std::strcmp(k->op, "lora_shrink")) *shrink = k->launch;
        if (!std::strcmp(k->op, "lora_expand")) *expand = k->launch;
    }
}

int main() {
    LaunchFn rs = nullptr, re = nullptr;
    load_libref(&rs, &re);
    std::printf("lora: libref %s\n", rs && re
                ? "loaded: the emulated kernels are held to it as well"
                : "not found ($RADIANCE_HOME): held to ref_host.cpp and the double sums only");
    std::vector<Impl> impls = { { "host", host_shrink, host_expand, true } };
    if (rs && re) impls.push_back({ "libref", rs, re, true });
    impls.push_back({ "r4d-emu", r4d_rad_lora_shrink, r4d_rad_lora_expand, true });
    lora_cases::Tally T;
    lora_cases::run_matrix(T, impls, 11);
    std::printf("lora: %d implementation-cases, %d failure(s); worst |err|/tol shrink %.3f, "
                "expand %.3f\n", T.cases, T.fails, T.worst_shrink, T.worst_expand);
    return T.fails != 0;
}
