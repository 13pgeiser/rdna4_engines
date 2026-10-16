/* The cases harness_test runs through the harness's command line. Not a test of its own: several
 * cases fail on purpose, and harness_test selects them by name. */
#include "rad_test.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>

TEST(fails_a_check) { CHECK(1 + 1 == 3); }

TEST(throws_an_exception) {
    CHECK(true);
    throw std::runtime_error("boom");
}

TEST(passes_one) { CHECK(true); }

TEST(passes_two) { CHECK_EQ(2, 2); }

TEST(skips) { SKIP("no widget on this machine"); }

TEST(needs_an_artifact) {
    NEED_ARTIFACT("/nonexistent/radtest/libwidget.so");
    CHECK(true);
}

/* Prints the directory on stdout, so the caller can look for it after the binary exits. */
TEST(writes_a_temp_file) {
    std::ofstream(radtest::tmp_path("data.txt")) << "payload\n";
    std::printf("tmpdir=%s\n", radtest::tmpdir().c_str());
    std::fflush(stdout);
    CHECK(access(radtest::tmp_path("data.txt").c_str(), R_OK) == 0);
}

/* Spins until the process has spent 0.11 s of CPU, so the time and the CPU budget are certain to
 * be crossed on any machine. */
TEST(burns_cpu) {
    const double until = radtest::cpu_seconds() + 0.11;
    volatile unsigned x = 0;
    while (radtest::cpu_seconds() < until) x = x + 1;
    CHECK(true);
}

TEST(reports_the_level) {
    std::printf("level=%d scaled=%d\n", radtest::level(), radtest::scaled(10, 20));
    std::fflush(stdout);
    CHECK(true);
}

TEST(perf_a_timed_case) { CHECK(true); }

RAD_TEST_MAIN()
