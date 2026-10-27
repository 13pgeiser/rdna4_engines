/* spechub_wiring_test.cpp -- what the loader and the reference gate see of the spechub plugin.
 *
 * The device rows must be production rows and the host rows reference rows; were the whole
 * library marked reference, the reference gate would refuse the coupled ops at bringup. The check
 * loads the real DSO as the loader does (dlopen and the exported symbols) and asserts the domain
 * and the reference answer per row. A host build has only the reference rows and must stay
 * classified as such.
 */
#include "rad_test.h"

#include "rad_abi.h"

#include <dlfcn.h>

#include <map>
#include <string>
#include <vector>

namespace {

struct Plugin {
    void* handle = nullptr;
    const RadPluginInfo* (*info)() = nullptr;
    int (*reference)() = nullptr;
    int (*kern_ref)(int) = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;

    bool open(const char* path) {
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        info      = (const RadPluginInfo* (*)())dlsym(handle, "rad_plugin_info");
        reference = (int (*)())dlsym(handle, "rad_plugin_reference");
        kern_ref  = (int (*)(int))dlsym(handle, "rad_kernel_reference");
        count     = (int (*)())dlsym(handle, "rad_kernel_count");
        at        = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return info && reference && count && at;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

/* The rule rad_reference_ops applies, over the rows as the loader would classify them: an op is on
 * the reference when no row for it is a non-reference row. */
std::vector<std::string> reference_only_ops(const Plugin& p, const std::vector<int>& is_ref) {
    std::map<std::string, bool> has_real;
    for (int i = 0; i < p.count(); ++i) {
        const RadKernelInfo* k = p.at(i);
        if (!k) continue;
        bool& real = has_real[k->op];
        real = real || !is_ref[(size_t)i];
    }
    std::vector<std::string> out;
    for (const auto& kv : has_real)
        if (!kv.second) out.push_back(kv.first);
    return out;
}

}  /* namespace */

TEST(spechub_wiring_the_device_rows_are_production_and_the_oracle_rows_are_reference) {
    Plugin p;
    REQUIRE(p.open(RAD_SPECHUB_LIB));

    CHECK_EQ(p.info()->kind, (int)RAD_PLUGIN_KERNEL);
    CHECK_EQ(std::string(p.info()->name), std::string("spechub"));

    /* The per-row answer must exist: without it the loader falls back to the library answer, which
     * is what mis-classified the device rows. */
    REQUIRE(p.kern_ref != nullptr);
    /* And the library as a whole is NOT a reference library -- otherwise the loader sorts it below
     * the kernel libraries it serves beside and every row inherits the flag. */
    CHECK_EQ(p.reference(), 0);

    std::vector<int> is_ref;
    int dev = 0, host = 0;
    for (int i = 0; i < p.count(); ++i) {
        const RadKernelInfo* k = p.at(i);
        REQUIRE(k != nullptr);
        const int r = p.kern_ref(i);
        is_ref.push_back(r);
        if (k->domain == RAD_DOMAIN_DEVICE) {
            ++dev;
            /* A device row is a kernel, never a reference: this is the line that decides whether a
             * coupled run may start at all. */
            CHECK_EQ(r, 0);
            CHECK(std::string(k->name).rfind("ref_", 0) != 0);
        } else {
            ++host;
            CHECK_EQ(k->domain, (int)RAD_DOMAIN_HOST);
            /* The host oracle beside it keeps its reference status. */
            CHECK_EQ(r, 1);
            CHECK(std::string(k->name).rfind("ref_", 0) == 0);
        }
    }
    CHECK_EQ(dev + host, p.count());
#ifdef RAD_SPECHUB_WITH_DEVICE
    CHECK_EQ(dev, 3);
    CHECK_EQ(host, 3);
#else
    CHECK_EQ(dev, 0);
    CHECK_EQ(host, 3);
#endif

    /* Every op must have at least one production row on a device build. This is the property the
     * gate reads: with it, the coupling is servable without --debug-accept-reference-kernels. */
    const std::vector<std::string> refonly = reference_only_ops(p, is_ref);
#ifdef RAD_SPECHUB_WITH_DEVICE
    CHECK_EQ(refonly.size(), (size_t)0);
#else
    /* No card: every op is oracle-only, and the gate must still see them as such. */
    CHECK_EQ(refonly.size(), (size_t)3);
#endif
}

RAD_TEST_MAIN()
