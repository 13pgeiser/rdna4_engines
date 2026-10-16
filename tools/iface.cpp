/* iface.cpp -- CLI plumbing shared by the tools.
 *
 * The plugin loader and declare phase stay in core/build/startup.h, shared with the engine: a
 * second copy could disagree on hierarchy order, which decides the layout baked into a container.
 */
#include "iface.h"
#include "util/kernel_list.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>

namespace rad {

/* Same fallback as the engine (core/config.cpp): RAD_DEFAULT_HOME. Do not derive it from the
 * install prefix; the home is a directory below it, so the tools would find a different plugin
 * tree. */
std::string rad_home() {
    if (const char* e = std::getenv("RADIANCE_HOME")) if (*e) return e;
#ifdef RAD_DEFAULT_HOME
    return RAD_DEFAULT_HOME;
#else
    return ".";
#endif
}

std::vector<std::string> rad_split_list(const std::string& s) { return rad_split_kernel_list(s); }

std::vector<std::string> rad_hierarchy_from_env() {
    const char* e = std::getenv("RADIANCE_KERNELS");
    return (e && *e) ? rad_split_list(e) : std::vector<std::string>{};
}

bool rad_parse_int(const char* text, int64_t lo, int64_t hi, int64_t* out, std::string* why) {
    char* end = nullptr;
    errno = 0;
    const long long n = text ? std::strtoll(text, &end, 10) : 0;
    if (!text || !*text || !end || *end || errno == ERANGE) {
        if (why) *why = fmt("takes a whole number, not '%s'", text ? text : "");
        return false;
    }
    if (n < lo || n > hi) {
        if (why) *why = hi == INT64_MAX ? fmt("must be >= %lld, not %lld", (long long)lo, n)
                                        : fmt("must be in [%lld, %lld], not %lld", (long long)lo,
                                              (long long)hi, n);
        return false;
    }
    *out = n;
    return true;
}

int64_t rad_flag_int(const char* tool, const char* flag, const char* text, int64_t lo, int64_t hi) {
    int64_t v = 0;
    std::string why;
    if (!rad_parse_int(text, lo, hi, &v, &why)) {
        std::fprintf(stderr, "%s: %s %s\n", tool, flag, why.c_str());
        std::exit(2);
    }
    return v;
}

}  /* namespace rad */
