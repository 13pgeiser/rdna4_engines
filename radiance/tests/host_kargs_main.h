/* host_kargs_main.h -- main() for a device test that can also run with its kernel arguments in
 * pinned host memory.
 *
 * `--host-kargs` asks the AQL backend (rad::aql_force_host_kargs) to keep every card's kernel
 * argument rings and recorded-pass argument blocks in pinned host memory, the placement a card whose
 * PCI BAR is smaller than its VRAM gets, so that path is tested on a card with a large BAR. The rest
 * of the command line goes to the harness (rad_test.h).
 */
#pragma once

#include "rad_test.h"
#include "device/device.h"

#include <cstdio>
#include <cstring>

inline int radtest_host_kargs_main(int argc, char** argv) {
    int n = 0;
    for (int i = 0; i < argc; ++i) {
        if (i > 0 && !std::strcmp(argv[i], "--host-kargs")) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
            if (rad::aql_force_host_kargs() < 0) {
                std::fprintf(stderr, "--host-kargs: the device backend has already started\n");
                return 2;
            }
#else
            std::fprintf(stderr, "--host-kargs: this build has no AQL backend\n");
            return 2;
#endif
            continue;
        }
        argv[n++] = argv[i];
    }
    argv[n] = nullptr;
    return ::radtest::run_all(n, argv);
}

#define RAD_TEST_MAIN_HOST_KARGS() \
    int main(int argc, char** argv) { return radtest_host_kargs_main(argc, argv); }
