/* The test harness itself (rad_test.h): case selection, sharding, exit codes, skips, the temporary
 * directory and the CPU budget. ctest registrations depend on each of these: a shard that drops a
 * case or a filter that matches the wrong ones loses coverage without failing anything, so the
 * harness is checked through the command line of a fixture binary (harness_fixture.cpp). */
#include "rad_test.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

namespace {

struct Run {
    int         status = -1;   /* the exit status, or -1 when the process did not exit normally */
    std::string out;           /* stdout and stderr, interleaved */
};

/* Runs the fixture with `args`, under `env` (VAR=value words) and none of the harness's variables
 * inherited from ctest. */
Run fixture(const std::string& args, const std::string& env = "") {
    const std::string cmd = "env -u RAD_TEST_FILTER -u RAD_TEST_STRICT -u RAD_TEST_LEVEL "
                            "-u RAD_TEST_CPU_BUDGET " + env + " '" RAD_HARNESS_FIXTURE "' " +
                            args + " 2>&1";
    Run r;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return r;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) r.out.append(buf, n);
    const int st = pclose(p);
    r.status = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return r;
}

std::vector<std::string> lines(const std::string& s) {
    std::vector<std::string> v;
    size_t at = 0;
    while (at < s.size()) {
        const size_t nl = s.find('\n', at);
        v.push_back(s.substr(at, nl == std::string::npos ? std::string::npos : nl - at));
        if (nl == std::string::npos) break;
        at = nl + 1;
    }
    return v;
}

const std::vector<std::string> kAll = {
    "fails_a_check", "throws_an_exception", "passes_one", "passes_two", "skips",
    "needs_an_artifact", "writes_a_temp_file", "burns_cpu", "reports_the_level",
    "perf_a_timed_case"};

}  /* namespace */

/* With no pattern every case is selected, in declaration order, which is the order shards index. */
TEST(list_names_every_case_in_declaration_order) {
    const Run r = fixture("--list");
    CHECK_EQ(r.status, 0);
    CHECK(lines(r.out) == kAll);
}

/* A pattern keeps the cases containing it, a -pattern drops them, and RAD_TEST_FILTER adds to the
 * command line in the same form. The ctest entries rely on `-perf_` dropping exactly the timed
 * cases. */
TEST(a_pattern_keeps_matching_cases_and_a_dash_pattern_drops_them) {
    using V = std::vector<std::string>;
    CHECK(lines(fixture("--list passes").out) == (V{"passes_one", "passes_two"}));
    CHECK(lines(fixture("--list passes -two").out) == (V{"passes_one"}));
    CHECK(lines(fixture("--list skips fails").out) == (V{"fails_a_check", "skips"}));
    CHECK(lines(fixture("--list", "RAD_TEST_FILTER=passes,-one").out) == (V{"passes_two"}));
    CHECK(lines(fixture("--list -two", "'RAD_TEST_FILTER=passes skips'").out) ==
          (V{"passes_one", "skips"}));
    V no_perf = kAll;
    no_perf.pop_back();
    CHECK(lines(fixture("--list -perf_").out) == no_perf);
    CHECK(lines(fixture("--list perf_").out) == (V{"perf_a_timed_case"}));
}

/* For every shard count, including one above the number of cases, the shards' union is the
 * selection with each case exactly once, and no shard is more than one case larger than another. A
 * case lost between shards would never run. */
TEST(the_shards_cover_every_selected_case_exactly_once) {
    for (const char* filter : {"", "-perf_"}) {
        const std::vector<std::string> want = lines(fixture(std::string("--list ") + filter).out);
        for (int n : {2, 3, 4, 11}) {
            std::vector<std::string> seen;
            size_t lo = SIZE_MAX, hi = 0;
            for (int k = 0; k < n; ++k) {
                const Run r = fixture(std::string("--list --shard ") + std::to_string(k) + "/" +
                                      std::to_string(n) + " " + filter);
                CHECK_EQ(r.status, 0);
                const std::vector<std::string> part = lines(r.out);
                lo = std::min(lo, part.size());
                hi = std::max(hi, part.size());
                seen.insert(seen.end(), part.begin(), part.end());
            }
            std::vector<std::string> sorted_want = want;
            std::sort(sorted_want.begin(), sorted_want.end());
            std::sort(seen.begin(), seen.end());
            CHECK(seen == sorted_want);
            CHECK(hi - lo <= 1);
        }
    }
}

/* A failed check fails the binary, and the cases after it still run. */
TEST(a_failing_case_fails_the_binary_and_later_cases_still_run) {
    const Run r = fixture("fails_a_check passes_one");
    CHECK_EQ(r.status, 1);
    CHECK(has(r.out, "  FAIL fails_a_check\n"));
    CHECK(has(r.out, "CHECK(1 + 1 == 3)"));
    CHECK(has(r.out, "  ok   passes_one\n"));
    CHECK(has(r.out, "1 failure(s)"));
}

/* An exception escaping a case is that case's failure, not the end of the binary. */
TEST(an_exception_fails_its_case_and_later_cases_still_run) {
    const Run r = fixture("throws_an_exception passes_two");
    CHECK_EQ(r.status, 1);
    CHECK(has(r.out, "  FAIL throws_an_exception\n    exception: boom\n"));
    CHECK(has(r.out, "  ok   passes_two\n"));
}

/* A binary that checked nothing is ctest's skip (77), never a pass; one where some case checked
 * something passes on that case. A filter that selects nothing says so. */
TEST(a_binary_that_checked_nothing_is_a_skip_not_a_pass) {
    Run r = fixture("skips");
    CHECK_EQ(r.status, 77);
    CHECK(has(r.out, "  SKIP skips: no widget on this machine\n"));
    CHECK(has(r.out, "  skip skips\n"));
    r = fixture("skips passes_one");
    CHECK_EQ(r.status, 0);
    CHECK(has(r.out, "(1 skipped)"));
    r = fixture("no_case_has_this_name");
    CHECK_EQ(r.status, 77);
    CHECK(has(r.out, "no case matches the filter"));
}

/* A missing in-tree artifact skips when run by hand and fails under RAD_TEST_STRICT, which ctest
 * sets: an artifact this build makes is never legitimately absent there. */
TEST(a_missing_artifact_skips_by_hand_and_fails_under_strict) {
    Run r = fixture("needs_an_artifact");
    CHECK_EQ(r.status, 77);
    CHECK(has(r.out, "  SKIP needs_an_artifact: missing in-tree artifact: "
                     "/nonexistent/radtest/libwidget.so\n"));
    r = fixture("needs_an_artifact", "RAD_TEST_STRICT=1");
    CHECK_EQ(r.status, 1);
    CHECK(has(r.out, "  FAIL needs_an_artifact\n"));
    CHECK(has(r.out, "RAD_TEST_STRICT"));
    r = fixture("needs_an_artifact", "RAD_TEST_STRICT=0");
    CHECK_EQ(r.status, 77);
}

/* The temporary directory is made under RAD_TEST_TMPDIR, removed when the binary passes and kept
 * for inspection when it fails. */
TEST(the_temp_directory_is_removed_after_a_pass_and_kept_after_a_failure) {
    const std::string base = radtest::tmpdir();
    auto dir_of = [](const std::string& out) {
        for (const std::string& l : lines(out))
            if (l.rfind("tmpdir=", 0) == 0) return l.substr(7);
        return std::string();
    };
    Run r = fixture("writes_a_temp_file", "RAD_TEST_TMPDIR=" + base);
    CHECK_EQ(r.status, 0);
    std::string dir = dir_of(r.out);
    CHECK(dir.rfind(base + "/radtest-", 0) == 0);
    CHECK(!dir.empty() && access(dir.c_str(), F_OK) != 0);

    r = fixture("writes_a_temp_file fails_a_check", "RAD_TEST_TMPDIR=" + base);
    CHECK_EQ(r.status, 1);
    dir = dir_of(r.out);
    REQUIRE(dir.rfind(base + "/radtest-", 0) == 0);
    CHECK(access((dir + "/data.txt").c_str(), R_OK) == 0);
    CHECK(has(r.out, "kept " + dir));
}

/* RAD_TEST_LEVEL selects scaled()'s exhaustive size at 2 and above. */
TEST(the_level_comes_from_rad_test_level) {
    CHECK(has(fixture("reports_the_level").out, "level=1 scaled=10"));
    CHECK(has(fixture("reports_the_level", "RAD_TEST_LEVEL=2").out, "level=2 scaled=20"));
    CHECK(has(fixture("reports_the_level", "RAD_TEST_LEVEL=0").out, "level=0 scaled=10"));
}

/* A binary over its CPU budget fails and says why, and a case of 0.1 s or more prints its time
 * and heads the slowest list. The budget is the suite's tripwire for its own speed. */
TEST(a_binary_over_its_cpu_budget_fails) {
    Run r = fixture("burns_cpu passes_one", "RAD_TEST_CPU_BUDGET=0.05");
    CHECK_EQ(r.status, 1);
    CHECK(has(r.out, "over its budget of 0.05 s"));
    CHECK(has(r.out, "  ok   burns_cpu ("));
    CHECK(has(r.out, "slowest: burns_cpu "));
    r = fixture("passes_one", "RAD_TEST_CPU_BUDGET=60");
    CHECK_EQ(r.status, 0);
}

/* A command line the harness does not understand is refused rather than read as a filter. */
TEST(a_bad_command_line_is_refused) {
    CHECK_EQ(fixture("--shard 4/4").status, 2);
    CHECK_EQ(fixture("--shard 1").status, 2);
    CHECK_EQ(fixture("--shard -1/2").status, 2);
    CHECK_EQ(fixture("--shardx").status, 2);
    CHECK_EQ(fixture("--list --shard=1/2").status, 0);
}

/* CHECKs on worker threads are all counted. The threads start together, and the compiler barrier
 * makes each CHECK's increment a separate memory update, as it is in a real test, rather than one
 * sum the compiler folds the loop into. */
TEST(checks_on_worker_threads_are_all_counted) {
    const int before = radtest::checks();
    std::atomic<int> ready{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < 8; ++t)
        pool.emplace_back([&ready] {
            ready.fetch_add(1);
            while (ready.load() < 8) std::this_thread::yield();
            for (int i = 0; i < 20000; ++i) {
                CHECK(i >= 0);
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        });
    for (std::thread& t : pool) t.join();
    const int counted = radtest::checks() - before;
    CHECK_EQ(counted, 8 * 20000);
}

RAD_TEST_MAIN()
