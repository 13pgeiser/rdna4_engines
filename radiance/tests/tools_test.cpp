/* tools_test.cpp -- the command-line plumbing every tool shares (tools/iface.h).
 *
 * A tool reads numbers and kernel lists from the same operator who runs the engine, and a value
 * the tool reads differently from the engine is a silent difference: `rad-convert --max-spec auto`
 * read as 0 leaves the draft head out of a container, `--max-ctx 12k` declares at 12, and a kernel
 * list split by another rule loads other libraries than the engine will. So the tools refuse what
 * the engine refuses, and split what the engine splits, by the same code or against it here.
 */
#include "rad_test.h"
#include "iface.h"
#include "rad_core.h"

#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

using namespace rad;

namespace {

bool parses(const char* text, int64_t lo, int64_t hi, int64_t* out = nullptr) {
    int64_t v = 0;
    std::string why;
    const bool ok = rad_parse_int(text, lo, hi, &v, &why);
    if (ok && out) *out = v;
    return ok;
}

/* The engine's own reading of --kernels, for comparison with the tools'. */
std::vector<std::string> engine_kernels(const char* list) {
    Config c;
    Argv a({ "--model", "/m/x.rad", "--kernels", list });
    if (config_parse(a.argc(), a.argv.data(), &c) != RAD_OK) return { "<refused>" };
    return c.kernel_hierarchy;
}

}  /* namespace */

/* A whole decimal number is read exactly, at both ends of its range. */
TEST(a_whole_number_in_range_is_read_exactly) {
    int64_t v = -1;
    CHECK(parses("0", 0, 10, &v));
    CHECK_EQ(v, (int64_t)0);
    CHECK(parses("10", 0, 10, &v));
    CHECK_EQ(v, (int64_t)10);
    CHECK(parses("-3", -5, 5, &v));
    CHECK_EQ(v, (int64_t)-3);
    CHECK(parses("9223372036854775807", 0, INT64_MAX, &v));
    CHECK_EQ(v, (int64_t)INT64_MAX);
    /* Base 10 always: a leading zero is not octal. */
    CHECK(parses("010", 0, 100, &v));
    CHECK_EQ(v, (int64_t)10);
}

/* Everything atoi would have read as its leading digits, or as 0, is refused. */
TEST(a_value_that_is_not_a_whole_number_is_refused) {
    for (const char* bad : { "", "auto", "12k", "8x", "1.5", "0x10", "abc", "4 ", "--max-tok",
                             "99999999999999999999", "-99999999999999999999" })
        CHECK(!parses(bad, INT64_MIN, INT64_MAX));
    CHECK(!parses(nullptr, 0, 10));
}

/* A number outside the flag's range is refused with the range in the reason. */
TEST(a_number_out_of_range_is_refused_with_the_range) {
    std::string why;
    int64_t v = 7;
    CHECK(!rad_parse_int("0", 1, INT32_MAX, &v, &why));
    CHECK(has(why, "[1, 2147483647]"));
    CHECK(!rad_parse_int("-1", 0, INT64_MAX, &v, &why));
    CHECK(has(why, ">= 0"));
    CHECK(!rad_parse_int("12k", 0, 10, &v, &why));
    CHECK(has(why, "'12k'"));
    CHECK_EQ(v, (int64_t)7);                       /* untouched on refusal */
}

/* The tools split a kernel list exactly as the engine does, so a list validated with rad-kbench
 * loads the same libraries, in the same order, when it serves. */
TEST(the_tools_split_a_kernel_list_as_the_engine_does) {
    for (const char* list : { "libr4d", "libr4d,libref", "mine:libr4d:libref", "a,b:c",
                              ",a,,b,", ":a::b:", "/opt/k/libmine.so,libref" }) {
        const std::vector<std::string> tools = rad_split_list(list);
        CHECK(tools == engine_kernels(list));
        if (tools != engine_kernels(list)) std::fprintf(stderr, "    differs: '%s'\n", list);
    }
    CHECK(rad_split_list("mine:libr4d") == (std::vector<std::string>{ "mine", "libr4d" }));
}

/* $RADIANCE_KERNELS and $RADIANCE_HOME: set but empty is unset, for the tools as for the engine. */
TEST(an_empty_environment_variable_is_unset_for_the_tools) {
    const char* k = std::getenv("RADIANCE_KERNELS");
    const char* h = std::getenv("RADIANCE_HOME");
    const std::string sk = k ? k : "", sh = h ? h : "";

    setenv("RADIANCE_KERNELS", "", 1);
    CHECK(rad_hierarchy_from_env().empty());
    setenv("RADIANCE_KERNELS", "x:y", 1);
    CHECK(rad_hierarchy_from_env() == (std::vector<std::string>{ "x", "y" }));

    unsetenv("RADIANCE_HOME");
    const std::string dflt = rad_home();
    setenv("RADIANCE_HOME", "", 1);
    CHECK_EQ(rad_home(), dflt);
    setenv("RADIANCE_HOME", "/somewhere", 1);
    CHECK_EQ(rad_home(), std::string("/somewhere"));

    if (k) setenv("RADIANCE_KERNELS", sk.c_str(), 1); else unsetenv("RADIANCE_KERNELS");
    if (h) setenv("RADIANCE_HOME", sh.c_str(), 1); else unsetenv("RADIANCE_HOME");
}

RAD_TEST_MAIN()
