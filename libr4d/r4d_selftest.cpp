/* r4d_selftest -- dump the libr4d row table, cross-check it, and prove every row links and runs.
 *
 * A transcription error in a constraint table stays invisible until a model silently picks the
 * wrong kernel. So the table is printed in full and checked against itself:
 *
 *   - every row's `op` has a schema declared by this plugin, or nothing can select it;
 *   - every constraint key is a declared parameter of that op. libr4d's select() and radiance's
 *     schemas (docs/OPS.md) use different vocabularies, and a key renamed in only one place
 *     yields a row that matches nothing;
 *   - names are unique, launches exist, an init has a fini, a declared variant list is non-empty.
 *
 * Then, on a device, every launch is called with a minimal valid input, so a linkage error, a
 * missing libr4d symbol or an argument-order mistake surfaces here rather than in a serve. Some
 * cases are real numerical checks, e.g.:
 *
 *   - the W4 weight arrangement against libr4d's own dequant (`dequant_w4_bf16` undoes the
 *     fragment order), so the host relayout and the device kernel must agree;
 *   - the two-rank all-reduce across two real cards through the plugin's init/fini rendezvous,
 *     with a known sum: the only test of the peer handshake.
 *
 * The plugin is dlopened rather than linked, so an undefined symbol in the .so is caught too.
 *
 * One child process per row. A bad pointer handed to a kernel is a GPU page fault, which ROCr
 * reports by printing a line and aborting the process. In one process the first bad row would end
 * the run, and with stdout block-buffered down a pipe the fault line can land next to the wrong
 * row. So the parent runs the static phase and then re-execs itself once per row with
 * `--case <name>`; a child killed by a signal is a FAIL naming the signal, and the run continues.
 * posix_spawn rather than fork, because the parent already has the HIP runtime's threads up and
 * only exec resets them. The row list is the plugin's own table, so a row without a smoke case
 * reports `no smoke case`.
 *
 * `--case <name>` is also the reproducer: it runs one row, alone, in the foreground.
 */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_args.h"
#include "r4d_hc_fp8.h"
#include "r4d_fp8_frag.h"
#include "r4d_fp8_lm.h"

#include <dlfcn.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

/* posix_spawn's environment; declared here rather than relying on unistd.h's _GNU_SOURCE guard. */
extern char** environ;

#ifdef RAD_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

static int g_fail = 0;
static int g_skip = 0;

/* The row this process was asked to run, or null for the parent. Set from --case. */
static const char* g_case = nullptr;

/* Whether any case block claimed g_case; a row no case mentions is reported as a coverage hole. */
static bool g_matched = false;

static bool sel(const char* name) {
    if (!g_case) return true;
    if (std::strcmp(g_case, name) != 0) return false;
    g_matched = true;
    return true;
}

/* Child exit codes: 0 clean, 1 at least one FAIL, 2 only skips. Anything else, or a signal, is a
 * crash, which the parent reports as a FAIL for that row. */
enum { CH_OK = 0, CH_FAIL = 1, CH_SKIP = 2 };

static void ok(const char* what, bool cond, const std::string& detail = std::string()) {
    if (cond) { std::printf("  ok    %s\n", what); return; }
    ++g_fail;
    std::printf("  FAIL  %s%s%s\n", what, detail.empty() ? "" : " -- ", detail.c_str());
}

static void skip(const char* what, const char* why) {
    ++g_skip;
    std::printf("  skip  %s -- %s\n", what, why);
}

static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}



/* ------------------------------------------------------------------ the plugin surface */

struct Plugin {
    void* h = nullptr;
    uint32_t (*abi)(void) = nullptr;
    const RadPluginInfo* (*info)(void) = nullptr;
    int (*schema_count)(void) = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*kernel_count)(void) = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    /* Test hooks, not part of the plugin ABI: libr4d's own, resolved from the same .so. r4d.h
     * explains why the wave count cannot be compared any other way. */
    void (*pin_mw)(int) = nullptr;
    void (*pin_nt)(int) = nullptr;
    void (*pin_qsa)(int) = nullptr;
    void (*pin_moe)(int) = nullptr;
    void (*pin_fk)(int) = nullptr;
    void (*pin_dtile)(int) = nullptr;
    void (*pin_gather)(int) = nullptr;
    void (*pin_ddec)(int) = nullptr;
    void (*pin_hc)(int) = nullptr;
    void (*pin_qsel)(int) = nullptr;
    int (*record)(const void**, int) = nullptr;
};

/* Defined after the tiny-tensor helpers it needs; called from the static checks above them. */

template <typename T>
static bool sym(void* h, const char* n, T* out) {
    dlerror();
    void* p = dlsym(h, n);
    const char* e = dlerror();
    if (e || !p) { std::printf("  FAIL  dlsym(%s): %s\n", n, e ? e : "null"); ++g_fail; return false; }
    std::memcpy(out, &p, sizeof(p));
    return true;
}

static bool load(Plugin* pl, const char* path) {
    pl->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!pl->h) { std::printf("  FAIL  dlopen(%s): %s\n", path, dlerror()); ++g_fail; return false; }
    bool okall = true;
    okall &= sym(pl->h, "rad_plugin_abi_version", &pl->abi);
    okall &= sym(pl->h, "rad_plugin_info", &pl->info);
    okall &= sym(pl->h, "rad_kernel_schema_count", &pl->schema_count);
    okall &= sym(pl->h, "rad_kernel_schema_at", &pl->schema_at);
    okall &= sym(pl->h, "rad_kernel_count", &pl->kernel_count);
    okall &= sym(pl->h, "rad_kernel_at", &pl->kernel_at);
    okall &= sym(pl->h, "r4d_gemm_fp8a8_pin_mw", &pl->pin_mw);
    okall &= sym(pl->h, "r4d_gemm_fp8a8_pin_nt", &pl->pin_nt);
    okall &= sym(pl->h, "r4d_qsa_score_pin_form", &pl->pin_qsa);
    okall &= sym(pl->h, "r4d_moe_gemm_pin_prefill", &pl->pin_moe);
    okall &= sym(pl->h, "r4d_moe_gemm_pin_fk", &pl->pin_fk);
    okall &= sym(pl->h, "r4d_moe_gemm_pin_down_tile", &pl->pin_dtile);
    okall &= sym(pl->h, "r4d_moe_gather_pin_form", &pl->pin_gather);
    okall &= sym(pl->h, "r4d_moe_gemm_pin_down_dec", &pl->pin_ddec);
    okall &= sym(pl->h, "r4d_hc_read_pin_prefill", &pl->pin_hc);
    okall &= sym(pl->h, "r4d_qsa_select_pin_form", &pl->pin_qsel);
    okall &= sym(pl->h, "r4d_record_dispatches", &pl->record);
    return okall;
}

static const char* cop_name(int op) {
    switch (op) {
        case RAD_C_EQ:  return "==";
        case RAD_C_LE:  return "<=";
        case RAD_C_GE:  return ">=";
        case RAD_C_DIV: return "%";
        case RAD_C_IN:  return "in";
        default:        return "?";
    }
}

/* ------------------------------------------------------------------ the catalogue */

static void dump(const Plugin& pl) {
    const RadPluginInfo* i = pl.info();
    std::printf("plugin  %s %s (%s)\n        %s\n\n", i->name, i->version, i->build_target,
                i->description);

    std::printf("op schemas (%d)\n", pl.schema_count());
    for (int s = 0; s < pl.schema_count(); ++s) {
        const RadOpSchema* sc = pl.schema_at(s);
        std::printf("  %-30s", sc->op);
        for (int j = 0; j < sc->n_params; ++j)
            std::printf(" %s%s%s", sc->params[j].required ? "" : "[",
                        sc->params[j].key, sc->params[j].required ? "" : "]");
        std::printf("\n%34s", "");
        for (int j = 0; j < sc->n_operands; ++j) {
            static const char* role[] = { "in", "out", "inout", "w" };
            std::printf(" %s:%s%s", sc->operands[j].name,
                        role[sc->operands[j].role & 3], sc->operands[j].optional ? "?" : "");
        }
        std::printf("\n");
    }

    std::printf("\nkernels (%d)\n", pl.kernel_count());
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        std::printf("  %-44s op=%-28s fam=%-7s prio=%-3d dom=%s%s%s%s%s\n",
                    r->name, r->op, r->family, r->priority,
                    r->domain == RAD_DOMAIN_DEVICE ? "device" : "host",
                    r->init ? " init" : "", r->scratch ? " scratch" : "",
                    r->layout ? " layout" : "",
                    r->relayout ? (r->unrelayout ? " relayout+inverse" : " relayout") : "");
        std::printf("        ");
        for (int c = 0; c < r->n_constraints; ++c) {
            const RadConstraint& cc = r->constraints[c];
            if (cc.op == RAD_C_IN) std::printf("%s in {%s}  ", cc.key, cc.sval);
            else if (cc.op == RAD_C_DIV) std::printf("%s %% %lld == 0  ", cc.key, cc.ival);
            else std::printf("%s %s %lld  ", cc.key, cop_name(cc.op), cc.ival);
        }
        std::printf("\n");
        if (r->n_tunables) {
            int64_t space = 1;
            std::printf("        tunables(%d):", r->n_tunables);
            for (int v = 0; v < r->n_tunables; ++v) {
                const RadTunable& t = r->tunables[v];
                space *= t.n_values > 0 ? t.n_values : 1;
                std::printf(" %s=%lld/%d", t.key, (long long)t.deflt, t.n_values);
            }
            std::printf("   space %lld%s\n", (long long)space,
                        r->tune_valid ? " (pruned)" : "");
        }
    }
    std::printf("\n");
}

/* ------------------------------------------------------------------ the static checks */

static void cross_check(const Plugin& pl) {
    std::printf("static checks\n");
    ok("ABI version matches the header this was built against", pl.abi() == RAD_ABI_VERSION);
    ok("plugin declares itself a kernel plugin", pl.info()->kind == RAD_PLUGIN_KERNEL);

    const int ns = pl.schema_count(), nk = pl.kernel_count();
    ok("at least one op schema", ns > 0);
    ok("at least one kernel row", nk > 0);

    /* Unique names. */
    int dup_schema = 0, dup_kernel = 0;
    for (int a = 0; a < ns; ++a)
        for (int b = a + 1; b < ns; ++b)
            if (!std::strcmp(pl.schema_at(a)->op, pl.schema_at(b)->op)) ++dup_schema;
    for (int a = 0; a < nk; ++a)
        for (int b = a + 1; b < nk; ++b)
            if (!std::strcmp(pl.kernel_at(a)->name, pl.kernel_at(b)->name)) ++dup_kernel;
    ok("op schema names are unique", dup_schema == 0);
    ok("kernel names are unique", dup_kernel == 0);

    int no_schema = 0, bad_key = 0, no_launch = 0, init_no_fini = 0, bad_domain = 0, no_p2p = 0;
    int bad_axis = 0, bad_deflt = 0, dup_axis = 0;
    std::string axisdetail;
    std::string keydetail, p2pdetail;
    for (int k = 0; k < nk; ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!r->launch) { ++no_launch; continue; }
        if (r->init && !r->fini) ++init_no_fini;
        /* Every row is device domain; a stray host row would silently move a step off the card.
         * Host kernels are libavx's. */
        if (r->domain != RAD_DOMAIN_DEVICE) {
            ++bad_domain;
            keydetail += std::string(r->name) + " is not RAD_DOMAIN_DEVICE; ";
        }
        /* Every axis's default is one of its own values. The loader also refuses a plugin that
         * breaks this, but at startup; this names the row and key at authoring time. */
        for (int v = 0; v < r->n_tunables; ++v) {
            const RadTunable& t = r->tunables[v];
            if (!t.key || !t.key[0] || !t.values || t.n_values < 1) {
                ++bad_axis;
                if (axisdetail.empty())
                    axisdetail = std::string(r->name) + " axis " + std::to_string(v);
                continue;
            }
            int found = 0;
            for (int i = 0; i < t.n_values; ++i) if (t.values[i] == t.deflt) found = 1;
            if (!found) {
                ++bad_deflt;
                if (axisdetail.empty())
                    axisdetail = std::string(r->name) + ":" + t.key + " default "
                               + std::to_string((long long)t.deflt) + " is not one of its values";
            }
            for (int w = 0; w < v; ++w)
                if (r->tunables[w].key && !std::strcmp(r->tunables[w].key, t.key)) {
                    ++dup_axis;
                    if (axisdetail.empty())
                        axisdetail = std::string(r->name) + " declares " + t.key + " twice";
                }
        }

        const RadOpSchema* sc = nullptr;
        for (int s = 0; s < ns; ++s)
            if (!std::strcmp(pl.schema_at(s)->op, r->op)) { sc = pl.schema_at(s); break; }
        if (!sc) {
            ++no_schema;
            keydetail += std::string(r->name) + " has no schema for op '" + r->op + "'; ";
            continue;
        }
        /* A constraint on a key the op does not declare can never be satisfied, so the row is
         * silently unreachable. A "device." key is not a parameter: it is matched against the
         * machine (rad_types.h). */
        static const size_t kFact = std::strlen(RAD_DEVICE_FACT_PREFIX);
        for (int c = 0; c < r->n_constraints; ++c) {
            if (!std::strncmp(r->constraints[c].key, RAD_DEVICE_FACT_PREFIX, kFact)) continue;
            bool found = false;
            for (int j = 0; j < sc->n_params; ++j)
                if (!std::strcmp(sc->params[j].key, r->constraints[c].key)) { found = true; break; }
            if (!found) {
                ++bad_key;
                keydetail += std::string(r->name) + " constrains '" + r->constraints[c].key +
                             "' which op '" + r->op + "' does not declare; ";
            }
        }
        /* Every multi-card row states its transport: device.p2p == 1 for a kernel that accesses
         * peer memory directly, device.p2p == 0 for a host twin with a pinned host mailbox.
         * Without it, a machine without P2P would resolve a P2P kernel that faults or hangs, and
         * a host twin would compete with its original where P2P works. */
        bool multi = false, p2p = false;
        for (int c = 0; c < r->n_constraints; ++c) {
            const RadConstraint& rc = r->constraints[c];
            if (!std::strcmp(rc.key, "world_size") &&
                (rc.op == RAD_C_EQ || rc.op == RAD_C_GE) && rc.ival >= 2)
                multi = true;
            if (!std::strcmp(rc.key, RAD_FACT_P2P) && rc.op == RAD_C_EQ &&
                (rc.ival == 1 || rc.ival == 0))
                p2p = true;
        }
        if (multi && !p2p) {
            ++no_p2p;
            p2pdetail += std::string(r->name) + "; ";
        }
    }
    ok("every row has a launch", no_launch == 0);
    ok("every row with init has fini", init_no_fini == 0);
    ok("every row is RAD_DOMAIN_DEVICE but the named host gather", bad_domain == 0, keydetail);
    ok("every tunable axis has a key and at least one value", bad_axis == 0, axisdetail);
    ok("every tunable default is one of its own axis values", bad_deflt == 0, axisdetail);
    ok("no row declares the same axis key twice", dup_axis == 0, axisdetail);
    ok("every row's op has a schema", no_schema == 0, keydetail);
    ok("every constraint key is a declared parameter of its op", bad_key == 0, keydetail);
    ok("every row that needs two or more ranks requires device.p2p 1 or 0", no_p2p == 0,
       p2pdetail);

    /* The libr4d orderings the integer priority has to preserve. */
    int p_ts = -1, p_os = -1, p_dec = -1, p_pre = -1, p_m16 = -1, p_m64 = -1, p_bt = -1;
    int p_g2dec = -1, p_g2pre = -1, p_g2win = -1;
    int p_lsc = -1, p_lsw = -1, p_lsq = -1, p_lec = -1, p_let = -1;
    for (int k = 0; k < nk; ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!std::strcmp(r->name, "ar_twoshot_4rank_exact")) p_ts = r->priority;
        if (!std::strcmp(r->name, "ar_oneshot_4rank_exact")) p_os = r->priority;
        if (!std::strcmp(r->name, "attn_decode_h256_gqa6_fp8kv"))  p_dec = r->priority;
        if (!std::strcmp(r->name, "attn_prefill_h256_gqa6_fp8kv")) p_pre = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_m16")) p_m16 = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_m64")) p_m64 = r->priority;
        if (!std::strcmp(r->name, "gemm_bf16_nt_tiled")) p_bt = r->priority;
        if (!std::strcmp(r->name, "attn_decode_h256_gqa2_fp8kv"))  p_g2dec = r->priority;
        if (!std::strcmp(r->name, "attn_prefill_h256_gqa2_fp8kv")) p_g2pre = r->priority;
        if (!std::strcmp(r->name, "attn_window_h256_gqa2_fp8kv"))  p_g2win = r->priority;
        if (!std::strcmp(r->name, "lora_shrink_scalar")) p_lsc = r->priority;
        if (!std::strcmp(r->name, "lora_shrink_wmma"))   p_lsw = r->priority;
        if (!std::strcmp(r->name, "lora_shrink_wmma_q")) p_lsq = r->priority;
        if (!std::strcmp(r->name, "lora_expand_scalar")) p_lec = r->priority;
        if (!std::strcmp(r->name, "lora_expand_tiled"))  p_let = r->priority;
    }
    ok("two-shot all-reduce outranks one-shot (libr4d row order)", p_ts > p_os && p_ts > 0);
    ok("paged decode outranks paged prefill in the shared band", p_dec > p_pre && p_dec > 0);
    ok("gemm_bf16_nt_m16 outranks _m64 (libr4d row order, see the ladder)", p_m16 > p_m64);
    ok("gemm_bf16_nt_tiled outranks _m64 and not _m16 (M > 64 is its band)",
       p_bt > p_m64 && p_bt < p_m16);
    ok("the h256/gqa2 legs outrank the windowed kernel in both their bands",
       p_g2win > 0 && p_g2dec > p_g2win && p_g2pre > p_g2win);
    ok("the tiled LoRA rows outrank the scalar rows they shadow (at every M)",
       p_lsc > 0 && p_lec > 0 && p_lsw > p_lsc && p_lsq > p_lsc && p_let > p_lec);
    std::printf("\n");
}

/* ================================================================== the device phase */
/* --cold MIB: bytes read through the caches before every timed launch; see st_time_us. */
static int64_t g_cold = 0;

#ifdef RAD_HAVE_HIP

/* Every hipError_t in this file is checked. A kernel fault leaves a sticky error, so the
 * device-to-host copy of the answer fails; unchecked, the comparison then runs on stale host
 * memory, which in a loop over geometries holds the previous geometry's correct answer and passes.
 *
 * Forwarding macros keep call sites ordinary and supply __LINE__; failures count in `g_fail`,
 * which main() returns. The wrapper returns int, not hipError_t: ROCm puts `nodiscard` on the type
 * itself, so a hipError_t return would warn at every macro. hipSuccess is 0, so sites that test the
 * result read unchanged. */
static int hip_ck(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::printf("  FAIL  %s at r4d_selftest.cpp:%d: %s\n", what, line, hipGetErrorString(e));
        ++g_fail;
    }
    return (int)e;
}
#define st_memcpy(...)      hip_ck(hipMemcpy(__VA_ARGS__), "hipMemcpy", __LINE__)
#define st_memcpy2d(...)    hip_ck(hipMemcpy2D(__VA_ARGS__), "hipMemcpy2D", __LINE__)
#define st_memset(...)      hip_ck(hipMemset(__VA_ARGS__), "hipMemset", __LINE__)
#define st_free(...)        hip_ck(hipFree(__VA_ARGS__), "hipFree", __LINE__)
#define st_set_device(...)  hip_ck(hipSetDevice(__VA_ARGS__), "hipSetDevice", __LINE__)
#define st_stream_sync(...) hip_ck(hipStreamSynchronize(__VA_ARGS__), "hipStreamSynchronize", \
                                   __LINE__)
#define st_ev_create(...)   hip_ck(hipEventCreate(__VA_ARGS__), "hipEventCreate", __LINE__)
#define st_ev_destroy(...)  hip_ck(hipEventDestroy(__VA_ARGS__), "hipEventDestroy", __LINE__)
#define st_ev_record(...)   hip_ck(hipEventRecord(__VA_ARGS__), "hipEventRecord", __LINE__)
#define st_ev_sync(...)     hip_ck(hipEventSynchronize(__VA_ARGS__), "hipEventSynchronize", \
                                   __LINE__)
#define st_ev_elapsed(...)  hip_ck(hipEventElapsedTime(__VA_ARGS__), "hipEventElapsedTime", \
                                   __LINE__)

static std::vector<void*> g_allocs;

static void* dalloc(size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) bytes = 16;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    st_memset(p, 0, bytes);
    g_allocs.push_back(p);
    return p;
}

/* ---- the timed window, warm or cold (--cold MIB).
 *
 * Warm, a perf mode launches `iters` times back to back on one set of operands, so a weight that
 * fits in L2 (8 MiB) or the last-level cache (64 MiB) is read from cache after the first launch,
 * unlike in the engine, which streams weights cold every step.
 *
 * Cold, each launch follows a read of g_cold bytes through the hierarchy and is timed alone between
 * two events; the median is reported. The read copies into one kColdChunk destination, since
 * copying the full span would leave dirty lines for the timed launch to write back. The flush
 * buffers live outside dalloc because dfree_all runs between shapes. */
constexpr int64_t kColdChunk = 4ll << 20;

template <class F>
static double st_time_us(F&& launch, int iters, hipStream_t s) {
    if (iters <= 0) return 0.0;
    if (g_cold <= 0) {
        hipEvent_t e0, e1;
        st_ev_create(&e0); st_ev_create(&e1);
        st_ev_record(e0, s);
        for (int i = 0; i < iters; ++i) launch(i);
        st_ev_record(e1, s);
        st_ev_sync(e1);
        float ms = 0.0f;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0); st_ev_destroy(e1);
        return (double)ms * 1000.0 / iters;
    }
    static void* src = nullptr;
    static void* dst = nullptr;
    if (!src && (hipMalloc(&src, (size_t)g_cold) != hipSuccess ||
                 hipMalloc(&dst, (size_t)std::min(g_cold, kColdChunk)) != hipSuccess)) {
        std::printf("--cold: no %lld MiB of device memory for the flush\n",
                    (long long)(g_cold >> 20));
        std::exit(1);
    }
    std::vector<hipEvent_t> ev(2 * (size_t)iters);
    for (hipEvent_t& e : ev) st_ev_create(&e);
    for (int i = 0; i < iters; ++i) {
        for (int64_t off = 0; off < g_cold; off += kColdChunk)
            hip_ck(hipMemcpyAsync(dst, (const char*)src + off,
                                  (size_t)std::min(kColdChunk, g_cold - off),
                                  hipMemcpyDeviceToDevice, s), "hipMemcpyAsync", __LINE__);
        st_ev_record(ev[2 * (size_t)i], s);
        launch(i);
        st_ev_record(ev[2 * (size_t)i + 1], s);
    }
    st_ev_sync(ev.back());
    std::vector<float> t((size_t)iters);
    for (int i = 0; i < iters; ++i) st_ev_elapsed(&t[(size_t)i], ev[2 * (size_t)i], ev[2 * (size_t)i + 1]);
    for (hipEvent_t e : ev) st_ev_destroy(e);
    std::sort(t.begin(), t.end());
    return (double)t[t.size() / 2] * 1000.0;
}

static void dfree_all() {
    for (void* p : g_allocs) st_free(p);
    g_allocs.clear();
}

static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = d;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}

/* moe_gemm_q takes its experts as two tables by parity (operands 2/3 even, 6/7 odd). A stacked
 * plane of every expert is both: the even entries at twice its leading stride, the odd ones the
 * same from one entry in. `wbits`/`sbits` are the planes' element widths, which a plugin-private
 * dtype cannot report. */
static void moe_parity_tables(std::vector<RadTensor>& ts, int64_t ne, int wbits, int sbits) {
    const RadTensor w = ts[2], s = ts[3];
    RadTensor wo = w, so = s;
    ts[2].shape[0] = ts[3].shape[0] = (ne + 1) / 2;
    ts[2].stride[0] *= 2;
    ts[3].stride[0] *= 2;
    wo.shape[0] = so.shape[0] = ne / 2;
    wo.stride[0] *= 2;
    so.stride[0] *= 2;
    wo.data = (char*)w.data + w.stride[0] * wbits / 8;
    so.data = (char*)s.data + s.stride[0] * sbits / 8;
    /* In front of the output, which is the last operand. */
    ts.insert(ts.end() - 1, wo);
    ts.insert(ts.end() - 1, so);
}

/* An absent optional operand is a null `data`, not a zero rank: r4d_opt() keys on the pointer
 * (rad_abi.h), so clearing only the rank would still take the present branch. */
static void absent(RadTensor& t) { t.data = nullptr; t.rank = 0; }

/* Fills an i32 device buffer, e.g. cumulative sequence lengths. A zeroed cu (as dalloc returns)
 * means T == 0 for every sequence and the gdn kernels exit immediately, so smoke tests fill it. */
static void fill_i32(void* d, const std::vector<int32_t>& v) {
    st_memcpy(d, v.data(), v.size() * sizeof(int32_t), hipMemcpyHostToDevice);
}

static std::vector<int32_t> cu_even(int64_t seqs, int64_t total) {
    std::vector<int32_t> cu((size_t)seqs + 1, 0);
    for (int64_t i = 0; i <= seqs; ++i) cu[(size_t)i] = (int32_t)(total * i / seqs);
    return cu;
}

static RadParam PI(const char* k, long long v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
static RadParam PS(const char* k, const char* v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
/* A real-valued parameter, RAD_P_F64 as in the schemas (docs/OPS.md). This file calls launches
 * directly with no declare to catch a kind mismatch, so a string would silently fall back to the
 * shim's default. */
static RadParam PF(const char* k, double v) {
    RadParam p; std::memset(&p, 0, sizeof(p));
    p.key = k; p.kind = RAD_P_F64; p.dval = v; return p;
}

/* Bytes per element for the dtypes this file allocates. Local because the core's rad_dtype_bytes
 * is not linkable from here (r4d_plugin.h). */
static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: case RAD_U32: return 4;
        case RAD_BF16: case RAD_F16: case RAD_I16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

/* An avalanche hash for test data: a value that depends on every bit of its index, so a wrong
 * pitch or lane map reads numbers that are unrelated rather than a shifted copy of the right ones. */
static uint32_t st_mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

static void* alloc_for(const RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) n *= t.shape[i];
    return dalloc((size_t)n * esz(t.dtype));
}

/* An i32 operand whose contents matter (cu, state_idx, num_accepted), built and filled before
 * run_case, which would otherwise zero it. A zero cu means no tokens, and a zero num_accepted
 * makes the recurrent update read column -1 of its state-index row. */
static RadTensor TI32(const std::vector<int64_t>& shape, const std::vector<int32_t>& v) {
    RadTensor t = T(nullptr, RAD_I32, shape);
    t.data = alloc_for(t);
    if (t.data) fill_i32(t.data, v);
    return t;
}

/* n copies of one value: state slots and accepted counts, which are per-sequence constants here. */
static std::vector<int32_t> rep_i32(int64_t n, int32_t v) {
    return std::vector<int32_t>((size_t)n, v);
}

/* An f32 operand filled with one value, e.g. attention descales: a zero descale would zero every
 * score, and a uniform softmax would pass even with the dequantise multiply deleted. */
static RadTensor TF32(const std::vector<int64_t>& shape, float v) {
    RadTensor t = T(nullptr, RAD_F32, shape);
    t.data = alloc_for(t);
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    const std::vector<float> h((size_t)n, v);
    if (t.data) st_memcpy(t.data, h.data(), (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    return t;
}

/* 0, 1, ... n-1: one distinct cache slot per sequence, so two sequences do not race on one. */
static std::vector<int32_t> iota_i32(int64_t n) {
    std::vector<int32_t> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (int32_t)i;
    return v;
}

/* ------------------------------------------------------------------ the tuning space
 *
 * A row declares axes; one point in their cross product is one instantiation of the kernel. This
 * enumerates the product as the core does at tune time (all-defaults first, so a sweep's first
 * line is what an untuned install runs) and prunes points with the row's own tune_valid.
 * Enumerating rather than listing points keeps the harness from drifting from the plugin. */
/* How many points an interactive sweep walks (rad-tune walks the whole space). Raise it with
 * R4D_PERF_POINTS when an axis needs the whole ladder. */
static const size_t kPerfCap = [] {
    const char* e = std::getenv("R4D_PERF_POINTS");
    const long v = (e && *e) ? std::strtol(e, nullptr, 10) : 0;
    return (size_t)(v > 0 ? v : 64);
}();

struct TunePoint {
    std::vector<RadParam> choice;
    std::string           label;
};

static std::vector<TunePoint> tune_points(const RadKernelInfo* r, const std::vector<RadParam>& geom,
                                          size_t cap = 0) {
    std::vector<TunePoint> out;
    if (!r || r->n_tunables <= 0) { out.push_back(TunePoint{}); out.back().label = "-"; return out; }

    const int n = r->n_tunables;
    std::vector<int> idx((size_t)n, 0);
    for (;;) {
        TunePoint tp;
        tp.choice.resize((size_t)n);
        for (int i = 0; i < n; ++i) {
            const RadTunable& t = r->tunables[i];
            std::memset(&tp.choice[(size_t)i], 0, sizeof(RadParam));
            tp.choice[(size_t)i].key  = t.key;
            tp.choice[(size_t)i].kind = RAD_P_INT;
            tp.choice[(size_t)i].ival = t.values[idx[(size_t)i]];
            if (i) tp.label += ',';
            tp.label += std::string(t.key) + "=" + std::to_string((long long)t.values[idx[(size_t)i]]);
        }
        const int legal = !r->tune_valid ||
                          r->tune_valid(geom.data(), (int)geom.size(),
                                        tp.choice.data(), (int)tp.choice.size());
        if (legal) {
            /* The all-defaults point goes first, so a sweep's baseline is its first line. */
            int is_deflt = 1;
            for (int i = 0; i < n; ++i)
                if (tp.choice[(size_t)i].ival != r->tunables[i].deflt) { is_deflt = 0; break; }
            if (is_deflt) out.insert(out.begin(), std::move(tp));
            else          out.push_back(std::move(tp));
        }
        int i = n - 1;
        while (i >= 0 && ++idx[(size_t)i] >= r->tunables[i].n_values) { idx[(size_t)i] = 0; --i; }
        if (i < 0) break;
        if (cap && out.size() >= cap) break;
    }
    if (out.empty()) { out.push_back(TunePoint{}); out.back().label = "-"; }
    return out;
}

/* Appends a point's choices to a case's parameters, as the core does at issue. */
static std::vector<RadParam> with_point(const std::vector<RadParam>& ps, const TunePoint& tp) {
    std::vector<RadParam> out = ps;
    out.insert(out.end(), tp.choice.begin(), tp.choice.end());
    return out;
}

/* Run one row's launch with the operands and parameters the caller built. Allocates every tensor
 * whose data is null, sizes and allocates scratch from the row's own hook, then synchronises so a
 * launch failure is attributed here and not to whatever runs next. */
static void run_case(const RadKernelInfo* r, std::vector<RadTensor>& ts,
                     std::vector<RadParam>& ps, int expect = RAD_OK, void* instance = nullptr) {
    for (auto& t : ts) if (!t.data && t.rank) t.data = alloc_for(t);

    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    a.instance = instance;

    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need < 0) { ok(r->name, false, "scratch hook returned " + std::to_string(need)); return; }
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }

    hipStream_t st = nullptr;
    const int rc = r->launch(&a, (RadStream)st);
    const hipError_t he = (hipError_t)st_stream_sync(st);
    std::string d = "rc=" + std::to_string(rc);
    if (he != hipSuccess) d += std::string(" hip=") + hipGetErrorString(he);
    ok(r->name, rc == expect && he == hipSuccess, d);
}

static const RadKernelInfo* find(const Plugin& pl, const char* name) {
    for (int k = 0; k < pl.kernel_count(); ++k)
        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);
    return nullptr;
}

/* ------------------------------------------------------------------ per-family smoke tests */

static void attn_case(const Plugin& pl, const char* name, int64_t head_dim, int64_t gqa,
                      int64_t q_len, const char* kv_dtype, int prefill, int window = 0) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    const int64_t seqs = 2, kv_heads = 2, blocks = 8, bs = 16;
    const int64_t q_heads = kv_heads * gqa, total_q = seqs * q_len;
    const uint32_t kvdt = std::strcmp(kv_dtype, "bf16") == 0 ? RAD_BF16 : RAD_F8E4M3;
    /* One disjoint run of blocks per sequence, and a context that fills them. Zeroed (as dalloc
     * returns), the block table would point every sequence at block 0, and seqused would mean no
     * context, so both grids would exit at once. */
    const int64_t max_blocks = blocks / seqs, ctx = max_blocks * bs;

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { total_q, q_heads, head_dim }),
        T(nullptr, kvdt,     { blocks, kv_heads, bs, 2 * head_dim }),
        TI32({ seqs, max_blocks }, iota_i32(seqs * max_blocks)),
        TI32({ seqs }, rep_i32(seqs, (int32_t)ctx)),
        TF32({ seqs, kv_heads }, 1.0f),
        TF32({ seqs, kv_heads }, 1.0f),
        /* cu_seqlens. oAttnPaged has eight slots; omitting this one would shift `out` into it
         * and every row would fail with RAD_E_INVAL. Absent means a uniform batch, as here. */
        T(nullptr, RAD_I32,  { seqs + 1 }),
        T(nullptr, RAD_BF16, { total_q, q_heads, head_dim }),
    };
    if (kvdt == RAD_BF16) { absent(ts[4]); absent(ts[5]); }
    absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("q_len", q_len), PI("head_dim", head_dim), PI("gqa", gqa), PI("block_size", bs),
        PI("causal", 1), PI("window", window), PS("q_dtype", "bf16"), PS("kv_dtype", kv_dtype),
    };
    (void)prefill;
    run_case(r, ts, ps);
}

static void gemm_case(const Plugin& pl, const char* name, const char* dtype, int64_t M,
                      int64_t N, int64_t K, uint32_t adt, int need_ascale, int need_asum,
                      int need_bref, int64_t wbytes, int64_t sbytes) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    std::vector<RadTensor> ts = {
        T(nullptr, adt,      { M, K }),
        T(nullptr, RAD_F32,  { M }),
        T(nullptr, RAD_U8,   { wbytes }),
        T(nullptr, RAD_U8,   { sbytes }),
        T(nullptr, RAD_BF16, { M, N }),
        T(nullptr, RAD_F32,  { (K / 128) * ((M + 15) / 16) * 16 }),
        T(nullptr, RAD_U8,   { N }),
    };
    if (!need_ascale) absent(ts[1]);
    if (!need_asum)   absent(ts[5]);
    if (!need_bref)   absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dtype),
    };
    run_case(r, ts, ps);
}

/* ================================================================== the block-scaled fp8 family
 *
 * The gemm_case helper hands `b` and `b_scale` as flat U8 buffers, which the fp8 shim refuses: it
 * checks the scale plane's extents, since a scale at the wrong grid reads plausible numbers. So the
 * fp8 rows get their own builder with [N][K] and [ceil(N/128)][ceil(K/128)] shapes. */
static void fp8_gemm_case(const Plugin& pl, const char* name, const char* dtype,
                          int64_t M, int64_t N, int64_t K, int need_ascale,
                          const TunePoint* tp = nullptr) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    const int64_t sn = (N + 127) / 128, sk = (K + 127) / 128;
    std::vector<RadTensor> ts = {
        T(nullptr, need_ascale ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16, { M, K }),
        T(nullptr, RAD_F32,    { M, sk }),
        T(nullptr, RAD_F8E4M3, { N, K }),
        T(nullptr, RAD_BF16,   { sn, sk }),
        T(nullptr, RAD_BF16,   { M, N }),
        T(nullptr, RAD_F32,    { 1 }),
        T(nullptr, RAD_U8,     { 1 }),
    };
    if (!need_ascale) absent(ts[1]);
    absent(ts[5]);
    absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dtype),
    };
    if (tp) ps = with_point(ps, *tp);
    run_case(r, ts, ps);
}


/* ---- host mirrors, for the numeric check below --------------------------------------------- */
static float h_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}
static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* E4M3 -> f32 in software; exponent 0 is subnormal (no implicit leading one). The oracle for the
 * device's v_cvt_f32_fp8, so it is written from the format, not from the kernel. */
static float h_e4m3(uint8_t b) {
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const uint32_t ex = (uint32_t)(b >> 3) & 0x0Fu, mn = (uint32_t)b & 0x07u;
    if (ex == 0) return sign * (float)mn * (1.0f / 512.0f);
    return sign * (1.0f + (float)mn / 8.0f) * std::ldexp(1.0f, (int)ex - 7);
}

/* Two avalanche-random values for the numeric attention check: a bounded bf16 in [-1, 1] and an
 * e4m3 code, each a function of its index so a wrong pitch or lane map reads unrelated numbers. */
static uint16_t st_rand_bf16(uint32_t i) {
    const uint32_t x = st_mix32(i);
    const float f = ((x & 0x00ffffffu) / 16777215.0f) * 2.0f - 1.0f;
    return h_to_bf16(f);
}
static uint8_t st_rand_e4m3(uint32_t i) {
    const uint32_t x = st_mix32(i + 0x9e3779b9u);
    /* Bounded to [-1, 1] like st_rand_bf16. With the full e4m3 range (up to 448) the attention
     * scores grow so large that a 1-ulp difference between v_cvt_f32_fp8 and h_e4m3 flips the
     * softmax max into a large output error; bounded, the scores stay near 22. */
    static const uint8_t mag[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,   /* subnormal, 0 .. 7/64 */
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,   /* exp 4, 0.125 .. 0.234375 */
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,   /* exp 5, 0.25 .. 0.46875 */
        0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,   /* exp 6, 0.5 .. 0.9375 */
        0x38,                                             /* exp 7, 1.0 */
    };
    const uint8_t m = mag[(x >> 1) % (sizeof mag / sizeof mag[0])];
    return (uint8_t)(m | (((x >> 8) & 1u) << 7));
}

/* Inline paged-attention oracle: a port of ref_attn_paged (libref/ref_attn.cpp) for the uniform
 * batch, bf16 q, fp8-e4m3 or bf16 k/v, f32 descales, with the same mask, two-pass softmax and k/v
 * dequant (the selftest does not link libref). One (row, head) at a time: `i` is the query's index
 * within its sequence's query, and `pos = ctx - qlen + i`. */
static void attn_oracle_one(const uint16_t* qh, const uint8_t* kv, int kv_is_fp8,
                            int64_t head_dim, int64_t kv_heads, int64_t block_size, int64_t kh,
                            const int* block_table, int64_t max_blocks, int ctx,
                            float kds, float vds, int causal, int window, float scale,
                            int64_t qlen, int64_t i, float* out) {
    const int ew = kv_is_fp8 ? 1 : 2;
    const int64_t pos = ctx - qlen + i;
    const int64_t sb = kv_heads * block_size * 2 * head_dim;
    const int64_t sh = block_size * 2 * head_dim;
    const int64_t so = 2 * head_dim;
    const auto kslot = [&](int64_t blk, int64_t slot, int64_t d) {
        const uint8_t* p = kv + (blk * sb + kh * sh + slot * so + d) * ew;
        return kv_is_fp8 ? h_e4m3(p[0]) : h_bf16(*(const uint16_t*)p);
    };
    const auto vslot = [&](int64_t blk, int64_t slot, int64_t d) {
        const uint8_t* p = kv + (blk * sb + kh * sh + slot * so + head_dim + d) * ew;
        return kv_is_fp8 ? h_e4m3(p[0]) : h_bf16(*(const uint16_t*)p);
    };
    const auto visible = [&](int64_t j) {
        if (causal ? (j > pos) : (j >= ctx)) return false;
        if (window > 0) { const int64_t d = pos - j;
            if (causal ? (d >= window) : (d >= window || d <= -window)) return false; }
        return true;
    };
    float mx = -INFINITY;
    for (int64_t j = 0; j < ctx; ++j) {
        if (!visible(j)) continue;
        if (j / block_size >= max_blocks) break;
        int blk = block_table[j / block_size];
        if (blk < 0) continue;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) dot += h_bf16(qh[d]) * (kslot(blk, j % block_size, d) * kds);
        dot *= scale;
        if (dot > mx) mx = dot;
    }
    if (mx == -INFINITY) { for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f; return; }
    float sum = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) out[d] = 0.0f;
    for (int64_t j = 0; j < ctx; ++j) {
        if (!visible(j)) continue;
        if (j / block_size >= max_blocks) break;
        int blk = block_table[j / block_size];
        if (blk < 0) continue;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) dot += h_bf16(qh[d]) * (kslot(blk, j % block_size, d) * kds);
        const float w = std::exp(dot * scale - mx);
        sum += w;
        for (int64_t d = 0; d < head_dim; ++d) out[d] += w * (vslot(blk, j % block_size, d) * vds);
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) out[d] *= inv;
}

/* Numeric attention check: fill q and the paged k/v with avalanche-random data, run the row, and
 * compare the bf16 output against the oracle above. A negative control corrupts part of the
 * reference and requires the comparison to fail. */
static void attn_numeric_case(const Plugin& pl, const char* name, int64_t head_dim, int64_t gqa,
                              int64_t q_len, const char* kv_dtype, int window, int64_t ctx = 64) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }
    const int64_t seqs = 1, kv_heads = 2, bs = 16;
    const int64_t q_heads = kv_heads * gqa;
    const int64_t max_blocks = ctx / bs;
    const int64_t total_q = seqs * q_len;
    const int kv_is_fp8 = std::strcmp(kv_dtype, "bf16") != 0;
    const uint32_t kvdt = kv_is_fp8 ? RAD_F8E4M3 : RAD_BF16;
    const int ew = kv_is_fp8 ? 1 : 2;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::vector<uint16_t> qh((size_t)total_q * q_heads * head_dim);
    for (size_t i = 0; i < qh.size(); ++i) qh[i] = st_rand_bf16(i);
    const int64_t nelem = max_blocks * kv_heads * bs * 2 * head_dim;
    std::vector<uint8_t> kv((size_t)nelem * ew);
    for (int64_t e = 0; e < nelem; ++e) {
        const size_t bo = (size_t)e * ew;
        if (kv_is_fp8) kv[bo] = st_rand_e4m3(e);
        else { const uint16_t h = st_rand_bf16(e); kv[bo] = (uint8_t)h; kv[bo + 1] = (uint8_t)(h >> 8); }
    }
    const std::vector<int32_t> bt = iota_i32(max_blocks);

    void* d_q  = dalloc((size_t)total_q * q_heads * head_dim * 2);
    st_memcpy(d_q, qh.data(), qh.size() * 2, hipMemcpyHostToDevice);
    void* d_kv = dalloc(kv.size());
    st_memcpy(d_kv, kv.data(), kv.size(), hipMemcpyHostToDevice);
    void* d_out = dalloc((size_t)total_q * q_heads * head_dim * 2);

    std::vector<RadTensor> ts = {
        T(d_q,   RAD_BF16, { total_q, q_heads, head_dim }),
        T(d_kv,  kvdt,     { max_blocks, kv_heads, bs, 2 * head_dim }),
        TI32({ seqs, max_blocks }, bt),
        TI32({ seqs }, rep_i32(seqs, (int32_t)ctx)),
        TF32({ seqs, kv_heads }, 1.0f),
        TF32({ seqs, kv_heads }, 1.0f),
        T(nullptr, RAD_I32,  { seqs + 1 }),
        T(d_out, RAD_BF16, { total_q, q_heads, head_dim }),
    };
    if (kvdt == RAD_BF16) { absent(ts[4]); absent(ts[5]); }
    absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("q_len", q_len), PI("head_dim", head_dim), PI("gqa", gqa), PI("block_size", bs),
        PI("causal", 1), PI("window", window), PS("q_dtype", "bf16"), PS("kv_dtype", kv_dtype),
    };
    run_case(r, ts, ps);

    std::vector<uint16_t> oh((size_t)total_q * q_heads * head_dim);
    st_memcpy(oh.data(), d_out, oh.size() * 2, hipMemcpyDeviceToHost);

    const float abs_tol = 1e-3f, rel_tol = 2e-2f;
    std::vector<float> oref((size_t)total_q * q_heads * head_dim);
    float max_err = 0.0f; int nbad = 0;
    for (int64_t tok = 0; tok < total_q; ++tok)
      for (int64_t h = 0; h < q_heads; ++h) {
        const int64_t kh = h / gqa;
        float o[512];
        attn_oracle_one(&qh[(tok * q_heads + h) * head_dim], kv.data(), kv_is_fp8,
                        head_dim, kv_heads, bs, kh, bt.data(), max_blocks, ctx,
                        1.0f, 1.0f, 1, window, scale, q_len, tok, o);
        for (int64_t d = 0; d < head_dim; ++d) {
            const size_t idx = (size_t)(tok * q_heads + h) * head_dim + d;
            oref[idx] = o[d];
            const float got = h_bf16(oh[idx]);
            const float diff = std::fabs(got - o[d]);
            if (diff > abs_tol + rel_tol * std::fabs(o[d])) ++nbad;
            if (diff > max_err) max_err = diff;
        }
    }
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric", name);
    ok(nm, nbad == 0,
       fmt("max_err=%.5g bad=%d/%d", max_err, nbad, (int)oref.size()));

    /* Negative control: corrupt part of the reference and require the comparison to fail. */
    for (size_t idx = 0; idx < oref.size(); ++idx)
        if ((idx & 7) == 0) oref[idx] += 0.5f;
    float nmax = 0.0f; int nbadc = 0;
    for (size_t idx = 0; idx < oref.size(); ++idx) {
        const float got = h_bf16(oh[idx]);
        const float diff = std::fabs(got - oref[idx]);
        if (diff > abs_tol + rel_tol * std::fabs(oref[idx])) ++nbadc;
        if (diff > nmax) nmax = diff;
    }
    std::snprintf(nm, sizeof nm, "%s numeric-negctl", name);
    ok(nm, nbadc > 0,
       fmt("corrupted ref must fail: max_err=%.5g bad=%d", nmax, nbadc));
}


/* ================================================================== a weight's canonical planes
 *
 * A container holds a weight as the planes of its encoding, row-major (spec §4.1), and a kernel's
 * layout hooks arrange them at load. Cases build the planes here, get the kernel's bytes from the
 * row's own hooks (`stored`), and compute the reference from the planes, so the arrangement is
 * checked by the kernel rather than against itself.
 *
 * The quantisation (absmax per scale block) is not under test; what is checked is that the kernel
 * reads the numbers the planes hold. */
struct Planes {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;   /* one per plane of `enc` */
    std::vector<int64_t> rows, cols;          /* each plane's extents, in its own elements */

    int find(const char* role) const { return rad_enc_find(&enc, role); }
    RadTensor t(int i) const {
        RadTensor x;
        std::memset(&x, 0, sizeof(x));
        x.data  = const_cast<uint8_t*>(data[(size_t)i].data());
        x.dtype = enc.plane[i].dtype;
        x.rank  = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    /* Size every plane for a [n, k] weight, zeroed. */
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

/* gemm_nt_q's `pb1`/`pb2` over N rows (ParoQuant): rows [0, e1), [e1, e2), [e2, N) meet
 * activation copies 0, 1, 2, side by side in a row; `parts` counts the copies. 0 leaves a bound
 * out, as the kernels read it. */
struct PartCut {
    int64_t e1, e2;
    int     parts;
    PartCut(int64_t N, int64_t pb1, int64_t pb2)
        : e1(pb1 > 0 ? std::min(pb1, N) : N),
          e2(pb2 > 0 ? std::max(e1, std::min(pb2, N)) : N),
          parts(e2 < N ? 3 : (e1 < N ? 2 : 1)) {}
    int part_of(int64_t n) const { return (int)(n >= e1) + (int)(n >= e2); }
};

/* An asymmetric grid's planes for a [N, K] weight: `bits`-bit codes (a bit stream along K), an f16
 * scale and a u8 zero point 0 .. 2^bits - 1 per 128 of K, drawn from st_mix32 of the
 * coordinates. */
static Planes asym_planes(int bits, int64_t N, int64_t K) {
    Planes p;
    p.enc = rad_enc_affine(bits == 4 ? RAD_U4 : RAD_U5, RAD_F16, 1, 128);
    rad_enc_add_plane(&p.enc, "zero", RAD_U8, 1, 128);
    p.size(N, K);
    const int ic = p.find("codes"), is = p.find("scale"), iz = p.find("zero");
    const int64_t rb = p.row_bytes(ic), G = K / 128;
    const uint32_t top = (1u << bits) - 1;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k)
            rad_store_code(p.data[(size_t)ic].data() + n * rb, p.enc.plane[ic].dtype, k,
                           (int)(st_mix32((uint32_t)(n * K + k) ^ 0x5eedu) & top));
        for (int64_t g = 0; g < G; ++g) {
            const uint32_t h = st_mix32((uint32_t)(n * G + g) ^ 0x2545u);
            rad_store_f32(p.data[(size_t)is].data(), RAD_F16, n * G + g, 0.002f * (float)(1 + h % 40));
            p.data[(size_t)iz][(size_t)(n * G + g)] = (uint8_t)((h >> 8) & top);
        }
    }
    return p;
}

/* The unnormalised Sylvester Walsh-Hadamard transform of `n` values, in place: (a + b) to the lower
 * index and (a - b) to the upper, the transform an "fwhtN" encoding names. H_n H_n = n I. */
static void h_fwht(float* v, int64_t n) {
    for (int64_t len = 1; len < n; len <<= 1)
        for (int64_t i = 0; i < n; i += len << 1)
            for (int64_t j = i; j < i + len; ++j) {
                const float a = v[j], b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
}

/* Block fp8 over [N, K], the checkpoint format: E4M3 codes and a bf16 scale per 128x128 block
 * (absmax / 448). Codes are taken against the rounded scale, so code times stored scale is the
 * weight the planes describe. */
static Planes fp8_block_planes(const std::vector<float>& w, int64_t N, int64_t K) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    rad_enc_add_plane(&p.enc, "codes", RAD_F8E4M3, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", RAD_BF16, 128, 128);
    p.size(N, K);
    const int64_t sn = p.rows[1], sk = p.cols[1];
    for (int64_t bi = 0; bi < sn; ++bi)
        for (int64_t bj = 0; bj < sk; ++bj) {
            float amax = 0.0f;
            for (int64_t r = bi * 128; r < std::min<int64_t>(N, (bi + 1) * 128); ++r)
                for (int64_t c = bj * 128; c < std::min<int64_t>(K, (bj + 1) * 128); ++c)
                    amax = std::max(amax, std::fabs(w[(size_t)(r * K + c)]));
            rad_store_f32(p.data[1].data(), RAD_BF16, bi * sk + bj, amax > 0 ? amax / 448.0f : 1.0f);
            const float s = rad_load_f32(p.data[1].data(), RAD_BF16, bi * sk + bj);
            for (int64_t r = bi * 128; r < std::min<int64_t>(N, (bi + 1) * 128); ++r)
                for (int64_t c = bj * 128; c < std::min<int64_t>(K, (bj + 1) * 128); ++c)
                    rad_store_f32(p.data[0].data(), RAD_F8E4M3, r * K + c, w[(size_t)(r * K + c)] / s);
        }
    return p;
}

/* A symmetric 4-bit grid over [N, K]: i4 codes -8..7 and a scale of dtype `sdt` per row per
 * `group` columns (absmax / 7). With `rot`, each group is multiplied by the `group`-wide Hadamard
 * first and the stored scale carries the residual 1/group (<H w, H a> = group <w, a>), so codes
 * times scales against a rotated activation give the unrotated product. The encoding names the
 * rotation, which the kernel's hook checks. */
static Planes int4_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t group,
                          uint32_t sdt, bool rot) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    if (rot) {
        char t[RAD_ENC_STR];
        std::snprintf(t, sizeof t, "fwht%lld", (long long)group);
        rad_enc_copy_str(p.enc.transform, t);
    }
    rad_enc_add_plane(&p.enc, "codes", RAD_I4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", sdt, 1, group);
    p.size(N, K);
    const int64_t ng = K / group, crow = p.row_bytes(0);
    std::vector<float> v((size_t)group);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            for (int64_t j = 0; j < group; ++j) v[(size_t)j] = w[(size_t)(n * K + g * group + j)];
            if (rot) h_fwht(v.data(), group);
            float amax = 0.0f;
            for (float x : v) amax = std::max(amax, std::fabs(x));
            float s = amax > 0 ? amax / 7.0f : 1.0f;
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s);
            s = rad_load_f32(p.data[1].data(), sdt, n * ng + g);
            for (int64_t j = 0; j < group; ++j) {
                const float q = std::nearbyint(v[(size_t)j] / s);
                const int code = (int)std::max(-8.0f, std::min(7.0f, q));
                rad_store_code(p.data[0].data() + n * crow, RAD_I4, g * group + j, code);
            }
            if (rot) rad_store_f32(p.data[1].data(), sdt, n * ng + g, s / (float)group);
        }
    return p;
}

/* The same rotated grid with libquant's w4nl codebook: u4 codes indexing a sixteen-entry f32
 * `table` plane (r4d_args.h's kW4nlTable), scale = amax / largest entry (22), nearest entry chosen,
 * residual 1/group folded into the stored scale. */
static Planes nl_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t group,
                        uint32_t sdt) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    char t[RAD_ENC_STR];
    std::snprintf(t, sizeof t, "fwht%lld", (long long)group);
    rad_enc_copy_str(p.enc.transform, t);
    rad_enc_add_plane(&p.enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", sdt, 1, group);
    rad_enc_add_table(&p.enc, "table", RAD_F32, 1, 16);
    p.size(N, K);
    std::memcpy(p.data[2].data(), kW4nlTable, sizeof kW4nlTable);
    const int64_t ng = K / group, crow = p.row_bytes(0);
    std::vector<float> v((size_t)group);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            for (int64_t j = 0; j < group; ++j) v[(size_t)j] = w[(size_t)(n * K + g * group + j)];
            h_fwht(v.data(), group);
            float amax = 0.0f;
            for (float x : v) amax = std::max(amax, std::fabs(x));
            float s = amax > 0 ? amax / 22.0f : 1.0f;
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s);
            s = rad_load_f32(p.data[1].data(), sdt, n * ng + g);
            for (int64_t j = 0; j < group; ++j) {
                const float u = v[(size_t)j] / s;
                int best = 0;
                for (int c = 1; c < 16; ++c)
                    if (std::fabs(u - kW4nlTable[c]) < std::fabs(u - kW4nlTable[best])) best = c;
                rad_store_code(p.data[0].data() + n * crow, RAD_U4, g * group + j, best);
            }
            rad_store_f32(p.data[1].data(), sdt, n * ng + g, s / (float)group);
        }
    return p;
}

/* The same codebook at w4nl64a8h's scales (`gs` 64) or w4nl32a8h's (32): the weight rotated by 128,
 * each `gs`-column part of a rotated group scaled on its own (amax / largest entry, residual 1/128
 * folded in, stored as E4M3 under r4d_args.h's fixed kW4G64Scale), nearest entry chosen against
 * the stored scale. Planes: codes, scale (E4M3 [1 x gs]), scale.1 (f32 constant), table. With
 * `w5`, w5nl64a8h's: u5 codes into kW5nlTable, whose largest entry is 30. */
static Planes nl64_planes(const std::vector<float>& w, int64_t N, int64_t K, int64_t gs = 64,
                          bool w5 = false) {
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    const int nt = w5 ? 32 : 16;
    const float top = w5 ? 30.0f : 22.0f;
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    rad_enc_copy_str(p.enc.transform, "fwht128");
    rad_enc_add_plane(&p.enc, "codes", w5 ? RAD_U5 : RAD_U4, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", RAD_F8E4M3, 1, gs);
    rad_enc_add_plane(&p.enc, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&p.enc, "table", RAD_F32, 1, nt);
    p.size(N, K);
    std::memcpy(p.data[2].data(), &kW4G64Scale, sizeof kW4G64Scale);
    std::memcpy(p.data[3].data(), tab, sizeof(float) * (size_t)nt);
    const int64_t crow = p.row_bytes(0), srow = p.row_bytes(1);
    std::vector<float> v(128);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < K / 128; ++g) {
            for (int64_t j = 0; j < 128; ++j) v[(size_t)j] = w[(size_t)(n * K + g * 128 + j)];
            h_fwht(v.data(), 128);
            const int64_t nh = 128 / gs;
            for (int64_t h = 0; h < nh; ++h) {
                float amax = 0.0f;
                for (int64_t j = 0; j < gs; ++j) amax = std::max(amax, std::fabs(v[(size_t)(h * gs + j)]));
                const uint8_t code = rad_f32_to_fp8e4m3(amax > 0 ? amax / top / 128.0f / kW4G64Scale
                                                                 : 1.0f);
                p.data[1][(size_t)(n * srow + g * nh + h)] = code;
                const float s = h_e4m3(code) * kW4G64Scale * 128.0f;   /* the rotated domain's */
                for (int64_t j = 0; j < gs; ++j) {
                    const float u = v[(size_t)(h * gs + j)] / s;
                    int best = 0;
                    for (int c = 1; c < nt; ++c)
                        if (std::fabs(u - tab[c]) < std::fabs(u - tab[best])) best = c;
                    rad_store_code(p.data[0].data() + n * crow, w5 ? RAD_U5 : RAD_U4,
                                   g * 128 + h * gs + j, best);
                }
            }
        }
    return p;
}

/* A symmetric 8-bit grid over [N, K]: i8 codes -127..127 and an f16 scale of amax / 127 per row per
 * 128 columns, as libquant writes for `codes=i8 group=128 scale=f16 clamp=sym` and the w8a16
 * layout hook takes. */
static Planes int8_planes(const std::vector<float>& w, int64_t N, int64_t K) {
    Planes p;
    p.enc = rad_enc_affine(RAD_I8, RAD_F16, 1, 128);
    p.size(N, K);
    const int64_t ng = K / 128;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(w[(size_t)(n * K + g * 128 + j)]));
            rad_store_f32(p.data[1].data(), RAD_F16, n * ng + g, amax > 0 ? amax / 127.0f : 1.0f);
            const float s = rad_load_f32(p.data[1].data(), RAD_F16, n * ng + g);
            for (int64_t j = 0; j < 128; ++j) {
                const float q = std::nearbyint(w[(size_t)(n * K + g * 128 + j)] / s);
                p.data[0][(size_t)(n * K + g * 128 + j)] =
                    (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
        }
    return p;
}

/* Code `k` of row `n` of a packed integer codes plane, as its value. */
static int code_at(const Planes& p, int64_t n, int64_t k) {
    return (int)rad_load_f32(p.data[0].data() + n * p.row_bytes(0), p.enc.plane[0].dtype, k);
}

/* The rectangle [r0, r1) x [c0, c1) of a logical weight as the planes covering it, each cut on its
 * own blocks as the loader takes a tensor-parallel rank's share. Byte-wide planes only. */
static Planes share(const Planes& p, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    Planes q;
    q.enc = p.enc;
    q.data.resize(p.data.size());
    q.rows.resize(p.data.size());
    q.cols.resize(p.data.size());
    for (int i = 0; i < p.enc.n_planes; ++i) {
        const RadEncPlane& pl = p.enc.plane[i];
        const int64_t br = pl.block[0] > 0 ? pl.block[0] : 1, bc = pl.block[1] > 0 ? pl.block[1] : 1;
        const int64_t pr0 = r0 / br, pr1 = (r1 + br - 1) / br, pc0 = c0 / bc, pc1 = (c1 + bc - 1) / bc;
        const int64_t eb = rad_dtype_bits(pl.dtype) / 8;
        q.rows[(size_t)i] = pr1 - pr0;
        q.cols[(size_t)i] = pc1 - pc0;
        q.data[(size_t)i].resize((size_t)((pr1 - pr0) * (pc1 - pc0) * eb));
        for (int64_t r = pr0; r < pr1; ++r)
            std::memcpy(q.data[(size_t)i].data() + (r - pr0) * (pc1 - pc0) * eb,
                        p.data[(size_t)i].data() + r * p.row_bytes(i) + pc0 * eb,
                        (size_t)((pc1 - pc0) * eb));
    }
    return q;
}

/* What row `r` reads for `operand` from the planes named `roles`: RAD_OK with its arrangement in
 * `out` (relayout), or RAD_E_UNSUPPORTED with the one selected plane unchanged (a hook's "stored as
 * is", or a row with no hook). Anything else is the hook's refusal. */
static int stored(const RadKernelInfo* r, const std::vector<RadParam>& lp, int operand,
                  const Planes& p, std::initializer_list<const char*> roles,
                  std::vector<uint8_t>* out, RadLayout* lay = nullptr) {
    int sel[RAD_ENC_MAX_PLANES];
    RadTensor pt[RAD_ENC_MAX_PLANES];
    int n = 0;
    for (const char* role : roles) {
        if (n >= RAD_ENC_MAX_PLANES) return RAD_E_INVAL;
        sel[n] = p.find(role);
        if (sel[n] < 0) return RAD_E_SHAPE;
        pt[n] = p.t(sel[n]);
        ++n;
    }
    RadLayout L;
    std::memset(&L, 0, sizeof(L));
    const int rc = r->layout ? r->layout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, &L)
                             : RAD_E_UNSUPPORTED;
    if (lay) *lay = L;
    if (rc == RAD_E_UNSUPPORTED) {
        if (n != 1) return RAD_E_SHAPE;
        *out = p.data[(size_t)sel[0]];
        return RAD_E_UNSUPPORTED;
    }
    if (rc != RAD_OK) return rc;
    if (!r->relayout || L.bytes <= 0) return RAD_E_STATE;
    out->assign((size_t)L.bytes, 0);
    return r->relayout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, out->data(),
                       L.bytes);
}

/* The row's inverse of `stored`, which must give back exactly the planes it was handed. `why`
 * says what differed, or that the row has no inverse. */
static bool restores(const RadKernelInfo* r, const std::vector<RadParam>& lp, int operand,
                     const Planes& p, std::initializer_list<const char*> roles,
                     const std::vector<uint8_t>& bytes, std::string* why) {
    if (!r->unrelayout) { *why = "the row publishes no inverse"; return false; }
    int sel[RAD_ENC_MAX_PLANES];
    RadTensor pt[RAD_ENC_MAX_PLANES];
    std::vector<std::vector<uint8_t>> back;
    int n = 0;
    for (const char* role : roles) {
        if (n >= RAD_ENC_MAX_PLANES) return false;
        sel[n] = p.find(role);
        if (sel[n] < 0) { *why = std::string("no plane '") + role + "'"; return false; }
        back.emplace_back(p.data[(size_t)sel[n]].size(), 0xA5);
        ++n;
    }
    for (int i = 0; i < n; ++i) {
        pt[i] = p.t(sel[i]);
        pt[i].data = back[(size_t)i].data();
    }
    const int rc = r->unrelayout(lp.data(), (int)lp.size(), operand, &p.enc, sel, bytes.data(),
                                 (int64_t)bytes.size(), pt, n);
    if (rc != RAD_OK) { *why = "unrelayout rc=" + std::to_string(rc); return false; }
    for (int i = 0; i < n; ++i)
        if (back[(size_t)i] != p.data[(size_t)sel[i]]) {
            *why = std::string("plane '") + p.enc.plane[sel[i]].role + "' differs";
            return false;
        }
    return true;
}

/* ------------------------------------------------- static YaRN, in double
 *
 * cos and sin of one rotary pair at one position: the inverse frequency theta^(-2i/rot), blended
 * toward inv/factor over the linear ramp [low, high] (vLLM's YaRNScalingRotaryEmbedding; its
 * correction bounds are held to yarn_from by tests/yarn_test.cpp), both scaled by the magnitude.
 *
 * `tol` is how far an f32 evaluation of the same formula may differ: 8 ulp of the angle plus the
 * rounding of the result. Where the ramp sits it is a few 1e-5 even at position 524287, so a wrong
 * factor, bound or blend order falls far outside it; it is loose only in the fastest pairs at long
 * positions, which YaRN leaves unscaled and where f32 cannot do better than the angle's ulp. */
struct YarnRef { double c, s, tol; };
static YarnRef yarn_ref(double pos, int i, int rot, double theta, double factor, double low,
                        double high, double magnitude) {
    const double inv  = std::pow(theta, -2.0 * i / rot);
    const double ramp = std::min(1.0, std::max(0.0, (i - low) / (high == low ? 0.001 : high - low)));
    const double a    = pos * (inv / factor * ramp + inv * (1.0 - ramp));
    return { std::cos(a) * magnitude, std::sin(a) * magnitude,
             magnitude * (std::fabs(a) * 0x1p-20 + 0x1p-21) };
}
/* The YaRN cases' parameters: factor 2 over 262144 trained positions at theta 1e7, and the
 * positions a 512K context reaches -- the ends, the trained length on both sides, and past it. */
static const double  kYarnMag = 1.069314718055994531;   /* 1 + 0.1 ln 2 */
static const int32_t kYarnPos[] = { 0, 1, 4096, 262143, 262144, 500000, 524287 };
static double yarn_low(int64_t rot)  { return rot == 64 ? 14 : rot == 128 ? 28 : 56; }
static double yarn_high(int64_t rot) { return rot == 64 ? 22 : rot == 128 ? 43 : 85; }

/* ------------------------------------------------- the fused attention prologue, against its pair
 *
 * Bit equality: `qk_norm_rope_gate` replaces rmsnorm(q) + rmsnorm(k) + rope(q) + rope(k), and
 * rounds the norm output to bf16 before rotating so that it computes the same function.
 *
 * The table is part of the claim: the fused kernel takes cos/sin, and a table with different
 * angles is a different rope. So `rope_table` runs here as the fused path's first stage, and the
 * comparison is of what the engine issues.
 *
 * In the engine the q side is a strided view (the [q|gate] projection is interleaved per head);
 * see the reference block below. */
static void qk_prologue_case(const Plugin& pl, uint32_t gain_dt, bool yarn = false, int64_t rot = 64) {
    if (!sel("qk_norm_rope_gate") && !sel("rope_table_f32")) return;
    /* Both gain widths run: architecture plugins here declare f32 norm gains, vLLM's bf16, and one
     * kernel flag picks between them. The smoke test only passes bf16. */
    const bool gf32 = gain_dt == RAD_F32;
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    const RadKernelInfo* rn = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* rr = find(pl, "rope_bf16");
    if (!rf || !rt || !rn || !rr) { ok("qk_norm_rope_gate vs its pair", false, "row absent"); return; }

    /* The engine's head counts, and k rotated in place as the block issues it: the launcher's
     * wave count depends on nq + nkv, so other head counts would exercise a different launch. */
    const int64_t M = 512, nq = 12, nkv = 2, hd = 256, ctx = yarn ? 524288 : 4096;
    auto add_yarn = [&](std::vector<RadParam>& ps) {
        if (!yarn) return;
        ps.push_back(PF("yarn_factor", 2)); ps.push_back(PF("yarn_low", yarn_low(rot)));
        ps.push_back(PF("yarn_high", yarn_high(rot))); ps.push_back(PF("yarn_magnitude", kYarnMag));
    };
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;
    /* Many rows: two summation orders of one mean square differ by an f32 ULP that rarely crosses
     * a bf16 boundary, so a mismatch shows in about one element in tens of thousands. 512 rows x
     * 14 heads x 256 lanes is 1.8M elements a side; positions span 0, small and near the top. */
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < M; ++i) pos[(size_t)i] = (int32_t)((i * 7 + i * i) % ctx);
    if (yarn) std::copy(std::begin(kYarnPos), std::end(kYarnPos), pos.begin());

    auto rnd = [](int64_t i) {                       /* deterministic, and spans the exponent */
        const uint32_t x = (uint32_t)(i * 2654435761u + 12345u);
        return (float)((int32_t)(x >> 16) - 32768) * (1.0f / 8192.0f);
    };
    std::vector<uint16_t> hqg((size_t)M * nq * 2 * hd), hk((size_t)M * nkv * hd);
    std::vector<uint16_t> hq_src((size_t)M * nq * hd), hgate((size_t)M * nq * hd);
    for (int64_t t = 0; t < M; ++t)
        for (int64_t h = 0; h < nq; ++h)
            for (int64_t d = 0; d < hd; ++d) {
                const uint16_t q = h_to_bf16(rnd(((t * nq) + h) * hd + d));
                const uint16_t g = h_to_bf16(rnd(1000000 + ((t * nq) + h) * hd + d));
                hqg[(size_t)((t * nq + h) * 2 * hd + d)]      = q;
                hqg[(size_t)((t * nq + h) * 2 * hd + hd + d)] = g;
                hq_src[(size_t)((t * nq + h) * hd + d)] = q;
                hgate[(size_t)((t * nq + h) * hd + d)]  = g;
            }
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = h_to_bf16(rnd(2000000 + (int64_t)i));
    /* The gain is built in bf16 and widened for the f32 run, so both runs use the same values. */
    std::vector<uint16_t> hwq((size_t)hd), hwk((size_t)hd);
    std::vector<float> fwq((size_t)hd), fwk((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) {
        hwq[(size_t)d] = h_to_bf16(rnd(3000000 + d) * 0.1f);
        hwk[(size_t)d] = h_to_bf16(rnd(4000000 + d) * 0.1f);
        fwq[(size_t)d] = h_bf16(hwq[(size_t)d]);
        fwk[(size_t)d] = h_bf16(hwk[(size_t)d]);
    }

    const size_t qb = hq_src.size() * 2, kb = hk.size() * 2;
    void* d_qg = dalloc(hqg.size() * 2);
    void* d_k  = dalloc(kb);
    const size_t gsz = gf32 ? 4u : 2u;
    void* d_wq = dalloc((size_t)hd * gsz);
    void* d_wk = dalloc((size_t)hd * gsz);
    void* d_p  = dalloc(pos.size() * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    void* d_qr = dalloc(qb);            /* the reference q and k, normed then rotated in place */
    void* d_kr = dalloc(kb);
    void* d_qo = dalloc(qb);
    void* d_ko = dalloc(kb);
    void* d_go = dalloc(qb);
    if (!d_qg || !d_k || !d_wq || !d_wk || !d_p || !d_cs || !d_qr || !d_kr ||
        !d_qo || !d_ko || !d_go) { ok("qk_norm_rope_gate vs its pair", false, "hipMalloc failed"); return; }
    st_memcpy(d_qg, hqg.data(), hqg.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_k,  hk.data(),  kb, hipMemcpyHostToDevice);
    if (gf32) {
        st_memcpy(d_wq, fwq.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
        st_memcpy(d_wk, fwk.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    } else {
        st_memcpy(d_wq, hwq.data(), (size_t)hd * 2, hipMemcpyHostToDevice);
        st_memcpy(d_wk, hwk.data(), (size_t)hd * 2, hipMemcpyHostToDevice);
    }
    st_memcpy(d_p,  pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);

    /* ---- the pair the fusion replaces ---- */
    {
        /* The engine feeds this norm a strided view: the q half of each head inside the
         * interleaved [q|gate] projection, rows 2*head_dim apart. The reference uses the same
         * operand, in case the norm differs on strided and dense rows. */
        RadTensor qsrc = T(d_qg, RAD_BF16, { M * nq, hd });
        qsrc.stride[0] = 2 * hd; qsrc.stride[1] = 1;
        std::vector<RadTensor> ts = { qsrc,
                                      T(d_wq, gain_dt, { hd }),
                                      T(d_qr, RAD_BF16, { M * nq, hd }) };
        std::vector<RadParam> ps = { PI("M", M * nq), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_k, RAD_BF16, { M * nkv, hd }),
                                      T(d_wk, gain_dt, { hd }),
                                      T(d_kr, RAD_BF16, { M * nkv, hd }) };
        std::vector<RadParam> ps = { PI("M", M * nkv), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    for (int side = 0; side < 2; ++side) {
        const int64_t nh = side ? nkv : nq;
        std::vector<RadTensor> ts = { T(side ? d_kr : d_qr, RAD_BF16, { M, nh * hd }),
                                      T(d_p, RAD_I32, { M }) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", nh),
                                     PI("n_head_kv", 0), PF("theta", theta), PF("scale", 1.0),
                                     PS("mode", "neox"), PI("rotary_dim", rot) };
        add_yarn(ps);
        run_case(rr, ts, ps);
    }

    /* ---- the fused path, table and all ---- */
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        add_yarn(ps);
        run_case(rt, ts, ps);
        if (yarn) {
            /* The table's rows at the YaRN positions against the double-precision formula, as a
             * fraction of what f32 may differ by (yarn_ref): at or under 1 passes. */
            std::vector<float> row((size_t)rot);
            double worst = 0.0;
            for (int32_t p : kYarnPos) {
                st_memcpy(row.data(), (float*)d_cs + (size_t)p * rot, (size_t)rot * sizeof(float),
                          hipMemcpyDeviceToHost);
                for (int i = 0; i < rot / 2; ++i) {
                    const YarnRef y = yarn_ref(p, i, (int)rot, theta, 2.0, yarn_low(rot),
                                               yarn_high(rot), kYarnMag);
                    worst = std::max(worst, std::fabs(row[(size_t)i] - y.c) / y.tol);
                    worst = std::max(worst, std::fabs(row[(size_t)(rot / 2 + i)] - y.s) / y.tol);
                }
            }
            char what[96], got[96];
            std::snprintf(what, sizeof what, "YaRN table, rot %lld, vs a double YaRN through 524287",
                          (long long)rot);
            std::snprintf(got, sizeof got, "worst %.3g of the f32 bound", worst);
            ok(what, worst <= 1.0, got);
        }
    }
    {
        std::vector<RadTensor> ts = {
            T(d_qg, RAD_BF16, { M, nq * 2 * hd }), T(d_k, RAD_BF16, { M, nkv * hd }),
            T(d_cs, RAD_F32,  { ctx, rot }),       T(d_p,  RAD_I32,  { M }),
            T(d_wq, gain_dt, { hd }),             T(d_wk, gain_dt, { hd }),
            T(d_qo, RAD_BF16, { M, nq * hd }),     T(d_k,  RAD_BF16, { M, nkv * hd }),
            T(d_go, RAD_BF16, { M, nq * hd }),
        };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", nq),
                                     PI("n_head_kv", nkv), PI("rot", rot), PS("q_dtype", "bf16"),
                                     PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
        run_case(rf, ts, ps);
    }

    std::vector<uint16_t> gq(hq_src.size()), gk(hk.size()), gg(hgate.size());
    std::vector<uint16_t> wq_(hq_src.size()), wk_(hk.size());
    st_memcpy(gq.data(),  d_qo, qb, hipMemcpyDeviceToHost);
    st_memcpy(gk.data(),  d_k,  kb, hipMemcpyDeviceToHost);   /* rotated in place */
    st_memcpy(gg.data(),  d_go, qb, hipMemcpyDeviceToHost);
    st_memcpy(wq_.data(), d_qr, qb, hipMemcpyDeviceToHost);
    st_memcpy(wk_.data(), d_kr, kb, hipMemcpyDeviceToHost);

    /* Bit equality holds because both fold the sum of squares the same way: `r4d_rmsnorm_bf16`
     * takes its thread width from `r4d_act_threads(M, n)`, which decides the columns each thread
     * folds, and the fused prologue reproduces that fold.
     *
     * The failure message reports a relative worst case: a rotated element is a difference of two
     * products, so a ULP can be a few percent of a small number, while a wrong angle or pairing is
     * O(1). */
    auto cmp = [&](const char* what, const std::vector<uint16_t>& got,
                   const std::vector<uint16_t>& want, int64_t width) {
        int64_t n_off = 0;
        double  worst = 0.0, scale = 0.0;
        std::string where;
        for (size_t i = 0; i < want.size(); ++i) {
            const double v = h_bf16(want[i]);
            if (v > scale) scale = v; else if (-v > scale) scale = -v;
        }
        if (scale <= 0.0) scale = 1.0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] == want[i]) continue;
            ++n_off;
            const double d = h_bf16(got[i]) - h_bf16(want[i]);
            const double ad = (d < 0 ? -d : d) / scale;
            if (ad > worst) {
                worst = ad;
                where = "worst at row " + std::to_string(i / (size_t)width) + " lane " +
                        std::to_string(i % (size_t)width) + ": " + std::to_string(h_bf16(got[i])) +
                        " vs " + std::to_string(h_bf16(want[i]));
            }
        }
        const std::string tag = std::string(yarn ? "YaRN, " : "") + (gf32 ? "f32 gain, " : "bf16 gain, ") + what;
        char rate[128];
        std::snprintf(rate, sizeof rate, "%lld of %zu differ, worst %.2e of full scale",
                      (long long)n_off, got.size(), worst);
        /* Byte equality: no element may differ. `worst` in the message separates a rounding
         * from a wrong angle. */
        ok(("qk_norm_rope_gate " + tag + " is byte-identical to the unfused pair").c_str(),
           n_off == 0, std::string(rate) + (n_off == 0 ? "" : " -- " + where));
    };
    cmp("q", gq, wq_, hd);
    cmp("k", gk, wk_, hd);
    cmp("gate", gg, hgate, hd);
}
/* ------------------------------------ the prologue with the cache store folded on, against the pair
 *
 * `qk_norm_rope_gate_kv_store` is `qk_norm_rope_gate` followed by `kv_store` into the E4M3 or bf16
 * cache, and must be byte-identical to that pair: q, the rotated k, and every byte of the cache.
 * Both paths get the same inputs on separate buffers (k is rotated in place) and caches starting
 * from the same random bytes, so a misplaced or spurious store shows.
 *
 * The inputs span the clamp (v up to +-2000, k normed and rotated), reaching E4M3's +-448
 * saturation and its subnormals. A pad row (slot -1) and a slot past the cache's end must store
 * nothing. The engine's geometry: 12 query heads and 2 kv heads of 256, rot 64. */
static void qk_kv_fold_case(const Plugin& pl, bool bf16) {
    const char* row = bf16 ? "qk_norm_rope_gate_kv_store_bf16" : "qk_norm_rope_gate_kv_store_fp8";
    if (!sel(row)) return;
    const char* name = bf16
        ? "qk_norm_rope_gate_kv_store_bf16 is byte-identical to the prologue + kv_store"
        : "qk_norm_rope_gate_kv_store_fp8 is byte-identical to the prologue + kv_store";
    const uint32_t cdt = bf16 ? RAD_BF16 : RAD_F8E4M3;
    const char* cname = bf16 ? "bf16" : "fp8_e4m3";
    const RadKernelInfo* rk = find(pl, row);
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rs = find(pl, bf16 ? "kv_store_bf16" : "kv_store_fp8");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    if (!rk || !rf || !rs || !rt) { ok(name, false, "row absent"); return; }

    const int64_t M = 48, nq = 12, nkv = 2, hd = 256, rot = 64, ctx = 4096, bs = 16, nb = 6;
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;
    std::vector<int32_t> pos((size_t)M), slot((size_t)M);
    for (int64_t t = 0; t < M; ++t) {
        pos[(size_t)t]  = (int32_t)((t * 97 + 5) % ctx);
        slot[(size_t)t] = (int32_t)((t * 7 + 3) % (nb * bs));     /* 7 is coprime with 96 */
    }
    slot[5]  = -1;                                                 /* a pad row */
    slot[11] = (int32_t)(nb * bs);                                 /* past the cache */

    uint32_t st = 0x9e3779b9u;
    auto rnd = [&](float span) {
        st = st * 1664525u + 1013904223u;
        return ((float)(st >> 8) / 16777216.0f * 2.0f - 1.0f) * span;
    };
    std::vector<uint16_t> hqg((size_t)M * nq * 2 * hd), hk((size_t)M * nkv * hd),
                          hv((size_t)M * nkv * hd);
    for (auto& x : hqg) x = h_to_bf16(rnd(4.0f));
    for (auto& x : hk)  x = h_to_bf16(rnd(4.0f));
    for (auto& x : hv)  x = h_to_bf16(rnd(2000.0f));
    std::vector<float> fwq((size_t)hd), fwk((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) { fwq[(size_t)d] = rnd(0.5f); fwk[(size_t)d] = rnd(0.5f); }
    const size_t cbytes = (size_t)(nb * nkv * bs * 2 * hd) * (bf16 ? 2 : 1);
    std::vector<uint8_t> hc(cbytes);
    for (auto& x : hc) x = (uint8_t)(rnd(128.0f) + 128.0f);

    const size_t qgb = hqg.size() * 2, kb = hk.size() * 2, qb = (size_t)M * nq * hd * 2;
    void* d_qg = dalloc(qgb);  void* d_v = dalloc(kb);
    void* d_kA = dalloc(kb);   void* d_kB = dalloc(kb);
    void* d_qA = dalloc(qb);   void* d_qB = dalloc(qb);
    void* d_cA = dalloc(cbytes); void* d_cB = dalloc(cbytes);
    void* d_wq = dalloc((size_t)hd * 4); void* d_wk = dalloc((size_t)hd * 4);
    void* d_p  = dalloc((size_t)M * 4);  void* d_sl = dalloc((size_t)M * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    if (!d_qg || !d_v || !d_kA || !d_kB || !d_qA || !d_qB || !d_cA || !d_cB || !d_wq || !d_wk ||
        !d_p || !d_sl || !d_cs) { ok(name, false, "hipMalloc failed"); return; }
    st_memcpy(d_qg, hqg.data(), qgb, hipMemcpyHostToDevice);
    st_memcpy(d_v,  hv.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_kA, hk.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_kB, hk.data(),  kb, hipMemcpyHostToDevice);
    st_memcpy(d_cA, hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_cB, hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_wq, fwq.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_wk, fwk.data(), (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_p,  pos.data(), (size_t)M * 4, hipMemcpyHostToDevice);
    st_memcpy(d_sl, slot.data(), (size_t)M * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        run_case(rt, ts, ps);
    }
    std::vector<RadParam> pq = { PI("M", M), PI("head_dim", hd), PI("n_head", nq),
                                 PI("n_head_kv", nkv), PI("rot", rot), PS("q_dtype", "bf16"),
                                 PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
    auto prologue = [&](void* k, void* q) {
        return std::vector<RadTensor>{
            T(d_qg, RAD_BF16, { M, nq * 2 * hd }), T(k, RAD_BF16, { M, nkv * hd }),
            T(d_cs, RAD_F32, { ctx, rot }),        T(d_p, RAD_I32, { M }),
            T(d_wq, RAD_F32, { hd }),              T(d_wk, RAD_F32, { hd }),
            T(q, RAD_BF16, { M, nq * hd }),        T(k, RAD_BF16, { M, nkv * hd }),
            RadTensor{} };
    };
    auto cache = [&](void* c) { return T(c, cdt, { nb, nkv, bs, 2 * hd }); };

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = prologue(d_kA, d_qA);
        run_case(rf, ts, pq);
    }
    {
        std::vector<RadTensor> ts = { T(d_kA, RAD_BF16, { M, nkv * hd }),
                                      T(d_v, RAD_BF16, { M, nkv * hd }),
                                      T(d_sl, RAD_I32, { M }), cache(d_cA) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head_kv", nkv),
                                     PI("block_size", bs), PS("kv_dtype", cname) };
        run_case(rs, ts, ps);
    }
    /* ---- the fold ---- */
    {
        std::vector<RadTensor> ts = prologue(d_kB, d_qB);
        ts.push_back(T(d_v, RAD_BF16, { M, nkv * hd }));
        ts.push_back(T(d_sl, RAD_I32, { M }));
        ts.push_back(cache(d_cB));
        std::vector<RadParam> ps = pq;
        ps.push_back(PI("block_size", bs));
        ps.push_back(PS("kv_dtype", cname));
        run_case(rk, ts, ps);
    }

    std::vector<uint8_t> qa(qb), qbv(qb), ka(kb), kbv(kb), ca(cbytes), cb(cbytes);
    st_memcpy(qa.data(), d_qA, qb, hipMemcpyDeviceToHost);
    st_memcpy(qbv.data(), d_qB, qb, hipMemcpyDeviceToHost);
    st_memcpy(ka.data(), d_kA, kb, hipMemcpyDeviceToHost);
    st_memcpy(kbv.data(), d_kB, kb, hipMemcpyDeviceToHost);
    st_memcpy(ca.data(), d_cA, cbytes, hipMemcpyDeviceToHost);
    st_memcpy(cb.data(), d_cB, cbytes, hipMemcpyDeviceToHost);
    auto ndiff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int64_t n = 0;
        for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
        return n;
    };
    /* And the store happened: the pair's cache differs from the initial bytes. */
    const int64_t dq = ndiff(qa, qbv), dk = ndiff(ka, kbv), dc = ndiff(ca, cb), moved = ndiff(ca, hc);
    char msg[160];
    std::snprintf(msg, sizeof msg, "q %lld, k %lld, cache %lld of %zu bytes differ; the pair wrote "
                  "%lld cache bytes", (long long)dq, (long long)dk, (long long)dc, cbytes,
                  (long long)moved);
    ok(name, dq == 0 && dk == 0 && dc == 0 && moved > 0, msg);
}
/* ------------------------------------------------ the k-only form, against the pair it replaces
 *
 * `n_head` 0 is what the QSA indexer asks for: its query heads are already plain `head_dim` runs
 * inside the `heads + 1` projection, so there is no [q|gate] split and no q operand. It replaces
 * `heads` rmsnorms on a strided view plus one rope, and must be byte-identical to them.
 *
 * The stride is why this is a separate case: the head here is 128 columns inside a 640-column row,
 * the indexer's layout. A shim taking the row pitch from the head count rather than the operand
 * would pass the case above and fail this one.
 *
 * The model's geometry: 4 heads of 128 at rot 64. The projection's fifth head is the raw key, which
 * this op must leave untouched; that is checked explicitly. */
static void qsa_qprep_case(const Plugin& pl) {
    if (!sel("qk_norm_rope_gate") && !sel("rope_table_f32")) return;
    const RadKernelInfo* rf = find(pl, "qk_norm_rope_gate");
    const RadKernelInfo* rt = find(pl, "rope_table_f32");
    const RadKernelInfo* rn = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* rr = find(pl, "rope_bf16");
    if (!rf || !rt || !rn || !rr) { ok("qk_norm_rope_gate k-only", false, "row absent"); return; }

    const int64_t M = 512, heads = 4, hd = 128, rot = 64, ctx = 4096;
    const int64_t nw = (heads + 1) * hd, qw = heads * hd;
    const double  theta = 1e7, eps = 1e-6, wadd = 1.0;

    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < M; ++i) pos[(size_t)i] = (int32_t)((i * 7 + i * i) % ctx);

    auto rnd = [](int64_t i) {
        const uint32_t x = (uint32_t)(i * 2654435761u + 12345u);
        return (float)((int32_t)(x >> 16) - 32768) * (1.0f / 8192.0f);
    };
    std::vector<uint16_t> hqk((size_t)M * nw);
    for (size_t i = 0; i < hqk.size(); ++i) hqk[i] = h_to_bf16(rnd(5000000 + (int64_t)i));
    std::vector<float> hw((size_t)hd);
    for (int64_t d = 0; d < hd; ++d) hw[(size_t)d] = h_bf16(h_to_bf16(rnd(6000000 + d) * 0.1f));

    const size_t qkb = hqk.size() * 2, qb = (size_t)M * qw * 2;
    void* d_qk = dalloc(qkb);
    void* d_w  = dalloc((size_t)hd * 4);
    void* d_p  = dalloc(pos.size() * 4);
    void* d_cs = dalloc((size_t)ctx * rot * 4);
    void* d_qr = dalloc(qb);          /* the reference: heads norms, then one rope in place */
    void* d_qo = dalloc(qb);          /* the fused output */
    if (!d_qk || !d_w || !d_p || !d_cs || !d_qr || !d_qo) {
        ok("qk_norm_rope_gate k-only", false, "hipMalloc failed"); return;
    }
    st_memcpy(d_qk, hqk.data(), qkb, hipMemcpyHostToDevice);
    st_memcpy(d_w,  hw.data(),  (size_t)hd * 4, hipMemcpyHostToDevice);
    st_memcpy(d_p,  pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memset(d_cs, 0, (size_t)ctx * rot * 4);
    st_memset(d_qr, 0, qb);
    st_memset(d_qo, 0, qb);

    /* ---- the pair: one norm a head off the strided projection, then one rope over all of them */
    for (int64_t h = 0; h < heads; ++h) {
        RadTensor src = T((void*)((uint16_t*)d_qk + h * hd), RAD_BF16, { M, hd });
        src.stride[0] = nw; src.stride[1] = 1;
        RadTensor dst = T((void*)((uint16_t*)d_qr + h * hd), RAD_BF16, { M, hd });
        dst.stride[0] = qw; dst.stride[1] = 1;
        std::vector<RadTensor> ts = { src, T(d_w, RAD_F32, { hd }), dst };
        std::vector<RadParam> ps = { PI("M", M), PI("n", hd), PF("eps", eps),
                                     PS("dtype", "bf16"), PF("wadd", wadd) };
        run_case(rn, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_qr, RAD_BF16, { M, qw }), T(d_p, RAD_I32, { M }) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", heads),
                                     PI("n_head_kv", 0), PF("theta", theta), PF("scale", 1.0),
                                     PS("mode", "neox"), PI("rotary_dim", rot) };
        run_case(rr, ts, ps);
    }

    /* ---- the fused path, table and all ---- */
    {
        std::vector<RadTensor> ts = { T(d_p, RAD_I32, { M }), T(d_cs, RAD_F32, { ctx, rot }) };
        std::vector<RadParam> ps = { PI("M", M), PI("rot", rot), PF("theta", theta),
                                     PF("scale", 1.0), PS("dtype", "f32") };
        run_case(rt, ts, ps);
    }
    {
        std::vector<RadTensor> ts = {
            T(d_qk, RAD_BF16, { M, nw }),            /* q_gate -- made absent below */
            T(d_qk, RAD_BF16, { M, nw }),            /* k: the whole projection, at its pitch */
            T(d_cs, RAD_F32,  { ctx, rot }),         T(d_p, RAD_I32, { M }),
            T(d_w,  RAD_F32,  { hd }),               /* q_w -- absent */
            T(d_w,  RAD_F32,  { hd }),
            T(d_qo, RAD_BF16, { M, qw }),            /* q_out -- absent */
            T(d_qo, RAD_BF16, { M, qw }),
            T(d_qo, RAD_BF16, { M, qw }),            /* gate_out -- absent */
        };
        absent(ts[0]); absent(ts[4]); absent(ts[6]); absent(ts[8]);
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", 0),
                                     PI("n_head_kv", heads), PI("rot", rot), PS("q_dtype", "bf16"),
                                     PF("eps", eps), PF("wadd", wadd), PI("cs_f32", 1) };
        run_case(rf, ts, ps);
    }

    std::vector<uint16_t> got((size_t)M * qw), want((size_t)M * qw);
    st_memcpy(got.data(),  d_qo, qb, hipMemcpyDeviceToHost);
    st_memcpy(want.data(), d_qr, qb, hipMemcpyDeviceToHost);
    int64_t n_off = 0;
    std::string where;
    for (size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i]) {
            if (!n_off)
                where = " -- first at row " + std::to_string(i / (size_t)qw) + " head " +
                        std::to_string((i % (size_t)qw) / (size_t)hd) + " lane " +
                        std::to_string(i % (size_t)hd) + ": " + std::to_string(h_bf16(got[i])) +
                        " vs " + std::to_string(h_bf16(want[i]));
            ++n_off;
        }
    char rate[96];
    std::snprintf(rate, sizeof rate, "%lld of %zu differ", (long long)n_off, got.size());
    ok("qk_norm_rope_gate k-only is byte-identical to heads norms plus a rope",
       n_off == 0, std::string(rate) + where);

    /* The head it must not touch: `k_out` is `heads` wide, so the raw key at input columns
     * [heads*hd, nw) is never an output, but an over-long head loop would norm it in place. */
    std::vector<uint16_t> back((size_t)M * nw);
    st_memcpy(back.data(), d_qk, qkb, hipMemcpyDeviceToHost);
    ok("qk_norm_rope_gate k-only leaves the projection's untouched heads alone",
       back == hqk, "the raw-key head must survive byte for byte");
}

/* -------------------------------------------- router_topk + moe_scatter, against the pair
 *
 * Byte equality on all five outputs. The fused kernel calls the same two device functions as the
 * standalone kernels, so this checks the wiring around them.
 *
 * The two expert counts differ: the selection runs over the whole (replicated) router, the sort
 * over this rank's slice. The case runs with `n_local < n_expert` and a non-zero `expert_base`, so
 * a fused op using one count for both would fail.
 *
 * Rows cross every wave assignment: the standalone selection gives a workgroup NW = 4 rows, the
 * fused one a wave per row with 8, 16 or 32 waves, wrapping above 32. 5, 13 and 20 rows reach each
 * width with a partial final pass, and 45 wraps.
 */
static void topk_scatter_at(const RadKernelInfo* rf, const RadKernelInfo* rt,
                            const RadKernelInfo* rs, int64_t M, const char* prot = nullptr);

static void topk_scatter_case(const Plugin& pl) {
    /* Row names, not op names: `sel` matches the row the sweep names. Guarding on an op name
     * whose row carries a suffix would never run the case. (qk_prologue_case's row is spelled
     * like its op.) */
    if (!sel("router_topk_scatter_bf16") && !sel("router_topk_bf16") && !sel("moe_scatter_i32"))
        return;
    const RadKernelInfo* rf = find(pl, "router_topk_scatter_bf16");
    const RadKernelInfo* rt = find(pl, "router_topk_bf16");
    const RadKernelInfo* rs = find(pl, "moe_scatter_i32");
    if (!rf || !rt || !rs) { ok("router_topk_scatter", false, "row absent"); return; }
    for (const int64_t M : { 5, 13, 20, 45 }) topk_scatter_at(rf, rt, rs, M);
    /* And with protected experts sorted last: the first, a middle and the last of the slice. */
    for (const int64_t M : { 13, 45 }) topk_scatter_at(rf, rt, rs, M, "0,17,101,253");
}

static void topk_scatter_at(const RadKernelInfo* rf, const RadKernelInfo* rt,
                            const RadKernelInfo* rs, int64_t M, const char* prot) {
    const int64_t ne = 512, nl = 254, base = 3, k = 10;
    const std::string at = " (M=" + std::to_string(M) + (prot ? ", protect " + std::string(prot) : "") +
                           ")";
    const int64_t NS = M * k;   /* routing slots in the step */

    /* Logits with ties and a wide exponent range, so the tie rule and the exact-on-logits
     * comparison both decide something. */
    std::vector<uint16_t> hlg((size_t)M * ne);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t e = 0; e < ne; ++e) {
            const int32_t t = (int32_t)((m * 7919 + e * 31) % 97) - 48;
            hlg[(size_t)(m * ne + e)] = h_to_bf16((float)t * 0.125f);
        }

    void* d_lg = dalloc(hlg.size() * 2);
    void* d_id[2] = { dalloc((size_t)NS * 4), dalloc((size_t)NS * 4) };
    void* d_w [2] = { dalloc((size_t)NS * 2), dalloc((size_t)NS * 2) };
    void* d_so[2] = { dalloc((size_t)NS * 4), dalloc((size_t)NS * 4) };
    void* d_of[2] = { dalloc((size_t)(nl + 1) * 4), dalloc((size_t)(nl + 1) * 4) };
    void* d_cn[2] = { dalloc((size_t)nl * 4), dalloc((size_t)nl * 4) };
    if (!d_lg || !d_id[0] || !d_id[1] || !d_w[0] || !d_w[1] || !d_so[0] || !d_so[1] ||
        !d_of[0] || !d_of[1] || !d_cn[0] || !d_cn[1]) {
        ok(("router_topk_scatter" + at).c_str(), false, "hipMalloc failed"); return;
    }
    st_memcpy(d_lg, hlg.data(), hlg.size() * 2, hipMemcpyHostToDevice);
    for (int i = 0; i < 2; ++i) {
        st_memset(d_id[i], 0xFF, (size_t)NS * 4);  st_memset(d_w[i], 0, (size_t)NS * 2);
        st_memset(d_so[i], 0xFF, (size_t)NS * 4);  st_memset(d_of[i], 0xFF, (size_t)(nl + 1) * 4);
        st_memset(d_cn[i], 0xFF, (size_t)nl * 4);
    }

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = { T(d_lg, RAD_BF16, { M, ne }),
                                      T(d_id[0], RAD_I32, { M, k }),
                                      T(d_w[0], RAD_BF16, { M, k }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", ne), PI("top_k", k),
                                     PI("norm", 1), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
    }
    {
        std::vector<RadTensor> ts = { T(d_id[0], RAD_I32, { M, k }),
                                      T(d_so[0], RAD_I32, { NS }),
                                      T(d_of[0], RAD_I32, { nl + 1 }),
                                      T(d_cn[0], RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                     PI("expert_base", base) };
        if (prot) ps.push_back(PS("protect", prot));
        run_case(rs, ts, ps);
    }

    /* ---- the fused op ---- */
    {
        std::vector<RadTensor> ts = { T(d_lg, RAD_BF16, { M, ne }),
                                      T(d_id[1], RAD_I32, { M, k }),
                                      T(d_w[1], RAD_BF16, { M, k }),
                                      T(d_so[1], RAD_I32, { NS }),
                                      T(d_of[1], RAD_I32, { nl + 1 }),
                                      T(d_cn[1], RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", ne), PI("n_local", nl),
                                     PI("top_k", k), PI("norm", 1), PS("dtype", "bf16"),
                                     PI("expert_base", base) };
        if (prot) ps.push_back(PS("protect", prot));
        run_case(rf, ts, ps);
    }

    auto same_i32 = [&](const char* what, void* a, void* b, int64_t n) {
        std::vector<int32_t> x((size_t)n), y((size_t)n);
        st_memcpy(x.data(), a, (size_t)n * 4, hipMemcpyDeviceToHost);
        st_memcpy(y.data(), b, (size_t)n * 4, hipMemcpyDeviceToHost);
        int64_t off = 0; std::string where;
        for (int64_t i = 0; i < n; ++i)
            if (x[(size_t)i] != y[(size_t)i]) {
                if (!off) where = " -- first at " + std::to_string(i) + ": " +
                                  std::to_string(y[(size_t)i]) + " vs " + std::to_string(x[(size_t)i]);
                ++off;
            }
        ok((std::string("router_topk_scatter ") + what + " matches the unfused pair" + at).c_str(),
           off == 0, std::to_string(off) + " of " + std::to_string(n) + " differ" + where);
    };
    same_i32("expert_ids", d_id[0], d_id[1], NS);
    same_i32("sorted_tok", d_so[0], d_so[1], NS);
    same_i32("expert_offset", d_of[0], d_of[1], nl + 1);
    same_i32("expert_count", d_cn[0], d_cn[1], nl);
    {
        std::vector<uint16_t> x((size_t)NS), y((size_t)NS);
        st_memcpy(x.data(), d_w[0], (size_t)NS * 2, hipMemcpyDeviceToHost);
        st_memcpy(y.data(), d_w[1], (size_t)NS * 2, hipMemcpyDeviceToHost);
        ok(("router_topk_scatter expert_w is byte-identical to the unfused pair" + at).c_str(),
           x == y, "the weights carry the renormalisation, which reads back a stored bf16");
    }
}

/* -------------------------------- router_gemm_topk_scatter, against the launches it stands in for
 *
 * Byte equality with three launches: the router's gemm_nt, the shared gate's gemm_nt at N = 1,
 * and router_topk_scatter over the logits; all seven outputs are compared, logits and gate
 * included. An element of the skinny bf16 GEMM depends only on its split-K count, and the fused op
 * asks the unfused launch's law for that count at each N (four at 512 x 2560, twenty for the
 * one-column gate; eight and sixteen at 256 x 2048).
 *
 * Rows cover every wave count (8, 16, 32) and row-tile count (one to four 16-row tiles, partial
 * included) at the served geometry (512 experts, top_k 10, K 2560) and a 256-expert one (the other
 * selection instantiation). Also an expert-parallel slice with a protect list, and no gate.
 *
 * Each shape runs twice back to back on one output: the last workgroup is found by a
 * self-resetting counter, owned by the instance the row's init creates, which all launches here
 * share while the grid changes between them. A launch with no instance is refused. */
static void router_gemm_fused_case(const Plugin& pl) {
    if (!sel("router_gemm_topk_scatter_bf16")) return;
    const RadKernelInfo* rg  = find(pl, "router_gemm_topk_scatter_bf16");
    const RadKernelInfo* m16 = find(pl, "gemm_bf16_nt_m16");
    const RadKernelInfo* m64 = find(pl, "gemm_bf16_nt_m64");
    const RadKernelInfo* rf  = find(pl, "router_topk_scatter_bf16");
    if (!rg || !m16 || !m64 || !rf) { ok("router_gemm_topk_scatter", false, "row absent"); return; }

    struct RgCase {
        int64_t ne, K, k, nl, base;
        const char* prot;
        bool gate;
        std::vector<int64_t> Ms;
    };
    const std::vector<RgCase> cfgs = {
        { 512, 2560, 10, 512, 0, nullptr, true,  { 1, 4, 5, 8, 13, 16, 20, 33, 45, 64 } },
        { 512, 2560, 10, 512, 0, nullptr, false, { 4, 20 } },
        { 512, 2560, 10, 254, 3, "0,17,101,253", true, { 5, 45 } },
        { 512, 2560, 10, 254, 3, nullptr, false, { 13 } },
        { 256, 2048,  8, 256, 0, nullptr, true,  { 4, 33 } },
    };
    const int64_t maxM = 64, maxNe = 512, maxK = 2560, maxNS = 64 * 10;

    uint64_t rs64 = 0x2545F4914F6CDD1Dull;
    auto rnd = [&]() { rs64 ^= rs64 << 13; rs64 ^= rs64 >> 7; rs64 ^= rs64 << 17; return rs64; };
    auto uni = [&](float lo, float hi) {
        return lo + (hi - lo) * (float)((rnd() >> 40) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    };

    void* d_a  = dalloc((size_t)maxM * maxK * 2);
    void* d_w  = dalloc((size_t)maxNe * maxK * 2);
    void* d_wg = dalloc((size_t)maxK * 2);
    void* d_lg[2] = { dalloc((size_t)maxM * maxNe * 2), dalloc((size_t)maxM * maxNe * 2) };
    void* d_g [2] = { dalloc((size_t)maxM * 2),         dalloc((size_t)maxM * 2) };
    void* d_id[2] = { dalloc((size_t)maxNS * 4), dalloc((size_t)maxNS * 4) };
    void* d_ew[2] = { dalloc((size_t)maxNS * 2), dalloc((size_t)maxNS * 2) };
    void* d_so[2] = { dalloc((size_t)maxNS * 4), dalloc((size_t)maxNS * 4) };
    void* d_of[2] = { dalloc((size_t)(maxNe + 1) * 4), dalloc((size_t)(maxNe + 1) * 4) };
    void* d_cn[2] = { dalloc((size_t)maxNe * 4), dalloc((size_t)maxNe * 4) };
    if (!d_a || !d_w || !d_wg || !d_lg[0] || !d_lg[1] || !d_g[0] || !d_g[1] || !d_id[0] ||
        !d_id[1] || !d_ew[0] || !d_ew[1] || !d_so[0] || !d_so[1] || !d_of[0] || !d_of[1] ||
        !d_cn[0] || !d_cn[1]) {
        ok("router_gemm_topk_scatter", false, "hipMalloc failed");
        return;
    }
    void* inst = nullptr;
    {
        std::vector<RadParam> ip = { PI("M", maxM), PI("K", maxK), PI("n_expert", maxNe),
                                     PI("top_k", 10), PI("norm", 1), PS("dtype", "bf16") };
        const int irc = rg->init && rg->fini
                            ? rg->init(ip.data(), (int)ip.size(), 0, 1, &inst) : RAD_E_INVAL;
        ok("router_gemm_topk_scatter init takes an instance", irc == RAD_OK && inst != nullptr,
           "rc=" + std::to_string(irc));
        if (irc != RAD_OK || !inst) return;
    }

    for (const RgCase& c : cfgs) {
        /* Logits a few units wide, like a trained router's, with a tie planted in every row. */
        std::vector<uint16_t> hw((size_t)c.ne * c.K), hwg((size_t)c.K);
        for (auto& v : hw)  v = h_to_bf16(uni(-0.06f, 0.06f));
        for (auto& v : hwg) v = h_to_bf16(uni(-0.06f, 0.06f));
        for (int64_t kk = 0; kk < c.K; ++kk) hw[(size_t)(7 * c.K + kk)] = hw[(size_t)(3 * c.K + kk)];
        st_memcpy(d_w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_wg, hwg.data(), hwg.size() * 2, hipMemcpyHostToDevice);

        for (const int64_t M : c.Ms) {
            const int64_t NS = M * c.k;
            const std::string at = fmt(" (n_expert %lld, n_local %lld, M=%lld%s%s)",
                                       (long long)c.ne, (long long)c.nl, (long long)M,
                                       c.gate ? ", gate" : "",
                                       c.prot ? ", protect" : "");
            std::vector<uint16_t> ha((size_t)M * c.K);
            for (auto& v : ha) v = h_to_bf16(uni(-2.0f, 2.0f));
            st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
            for (int i = 0; i < 2; ++i) {
                st_memset(d_lg[i], 0xFF, (size_t)maxM * maxNe * 2);
                st_memset(d_g[i], 0xFF, (size_t)maxM * 2);
                st_memset(d_id[i], 0xFF, (size_t)maxNS * 4);
                st_memset(d_ew[i], 0, (size_t)maxNS * 2);
                st_memset(d_so[i], 0xFF, (size_t)maxNS * 4);
                st_memset(d_of[i], 0xFF, (size_t)(maxNe + 1) * 4);
                st_memset(d_cn[i], 0xFF, (size_t)maxNe * 4);
            }

            /* ---- the three launches */
            const RadKernelInfo* gr = M <= 16 ? m16 : m64;
            {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_w, RAD_BF16, { c.ne, c.K }),
                                              T(d_lg[0], RAD_BF16, { M, c.ne }) };
                std::vector<RadParam> ps = { PI("M", M), PI("N", c.ne), PI("K", c.K),
                                             PS("dtype", "bf16") };
                run_case(gr, ts, ps);
            }
            if (c.gate) {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_wg, RAD_BF16, { 1, c.K }),
                                              T(d_g[0], RAD_BF16, { M, 1 }) };
                std::vector<RadParam> ps = { PI("M", M), PI("N", 1), PI("K", c.K),
                                             PS("dtype", "bf16") };
                run_case(gr, ts, ps);
            }
            {
                std::vector<RadTensor> ts = { T(d_lg[0], RAD_BF16, { M, c.ne }),
                                              T(d_id[0], RAD_I32, { M, c.k }),
                                              T(d_ew[0], RAD_BF16, { M, c.k }),
                                              T(d_so[0], RAD_I32, { NS }),
                                              T(d_of[0], RAD_I32, { c.nl + 1 }),
                                              T(d_cn[0], RAD_I32, { c.nl }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n_expert", c.ne),
                                             PI("n_local", c.nl), PI("top_k", c.k), PI("norm", 1),
                                             PS("dtype", "bf16"), PI("expert_base", c.base) };
                if (c.prot) ps.push_back(PS("protect", c.prot));
                run_case(rf, ts, ps);
            }

            /* ---- the fused op, twice */
            for (int rep = 0; rep < 2; ++rep) {
                std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, c.K }),
                                              T(d_w, RAD_BF16, { c.ne, c.K }),
                                              T(d_wg, RAD_BF16, { 1, c.K }),
                                              T(d_lg[1], RAD_BF16, { M, c.ne }),
                                              T(d_id[1], RAD_I32, { M, c.k }),
                                              T(d_ew[1], RAD_BF16, { M, c.k }),
                                              T(d_so[1], RAD_I32, { NS }),
                                              T(d_of[1], RAD_I32, { c.nl + 1 }),
                                              T(d_cn[1], RAD_I32, { c.nl }),
                                              T(d_g[1], RAD_BF16, { M, 1 }) };
                if (!c.gate) { absent(ts[2]); absent(ts[9]); }
                std::vector<RadParam> ps = { PI("M", M), PI("K", c.K), PI("n_expert", c.ne),
                                             PI("n_local", c.nl), PI("top_k", c.k), PI("norm", 1),
                                             PS("dtype", "bf16"), PI("expert_base", c.base) };
                if (c.prot) ps.push_back(PS("protect", c.prot));
                if (rep == 1) {
                    /* The second launch must not find anything the first one left. */
                    st_memset(d_id[1], 0xFF, (size_t)maxNS * 4);
                    st_memset(d_so[1], 0xFF, (size_t)maxNS * 4);
                    st_memset(d_of[1], 0xFF, (size_t)(maxNe + 1) * 4);
                }
                run_case(rg, ts, ps, RAD_OK, inst);

                const std::string tag = at + (rep ? ", second launch" : "");
                auto same = [&](const char* what, void* a, void* b, size_t bytes) {
                    std::vector<uint8_t> x(bytes), y(bytes);
                    st_memcpy(x.data(), a, bytes, hipMemcpyDeviceToHost);
                    st_memcpy(y.data(), b, bytes, hipMemcpyDeviceToHost);
                    size_t first = bytes;
                    int64_t n_off = 0;
                    for (size_t i = 0; i < bytes; ++i)
                        if (x[i] != y[i]) { if (first == bytes) first = i; ++n_off; }
                    ok((std::string("router_gemm_topk_scatter ") + what +
                        " is byte-identical to the unfused launches" + tag).c_str(),
                       n_off == 0,
                       n_off ? fmt("%lld byte(s) differ, first at byte %zu", (long long)n_off,
                                   first) : std::string());
                };
                same("logits", d_lg[0], d_lg[1], (size_t)M * c.ne * 2);
                if (c.gate) same("shared gate", d_g[0], d_g[1], (size_t)M * 2);
                same("expert_ids", d_id[0], d_id[1], (size_t)NS * 4);
                same("expert_w", d_ew[0], d_ew[1], (size_t)NS * 2);
                same("sorted_tok", d_so[0], d_so[1], (size_t)NS * 4);
                same("expert_offset", d_of[0], d_of[1], (size_t)(c.nl + 1) * 4);
                same("expert_count", d_cn[0], d_cn[1], (size_t)c.nl * 4);
            }
        }
    }
    /* A shape past the decode band is refused, not served at another arithmetic. */
    {
        const int64_t M = 65;
        std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { M, 2560 }),
                                      T(d_w, RAD_BF16, { 512, 2560 }),
                                      T(d_wg, RAD_BF16, { 1, 2560 }),
                                      T(nullptr, RAD_BF16, { M, 512 }),
                                      T(nullptr, RAD_I32, { M, 10 }),
                                      T(nullptr, RAD_BF16, { M, 10 }),
                                      T(nullptr, RAD_I32, { M * 10 }),
                                      T(nullptr, RAD_I32, { 513 }),
                                      T(nullptr, RAD_I32, { 512 }),
                                      T(nullptr, RAD_BF16, { M, 1 }) };
        std::vector<RadParam> ps = { PI("M", M), PI("K", 2560), PI("n_expert", 512),
                                     PI("top_k", 10), PI("norm", 1), PS("dtype", "bf16") };
        run_case(rg, ts, ps, RAD_E_SHAPE, inst);
    }
    /* A served shape with no instance is refused: there is no counter to name the last block. */
    {
        const int64_t M = 4;
        std::vector<RadTensor> ts = { T(d_a, RAD_BF16, { M, 2560 }),
                                      T(d_w, RAD_BF16, { 512, 2560 }),
                                      T(d_wg, RAD_BF16, { 1, 2560 }),
                                      T(d_lg[1], RAD_BF16, { M, 512 }),
                                      T(d_id[1], RAD_I32, { M, 10 }),
                                      T(d_ew[1], RAD_BF16, { M, 10 }),
                                      T(d_so[1], RAD_I32, { M * 10 }),
                                      T(d_of[1], RAD_I32, { 513 }),
                                      T(d_cn[1], RAD_I32, { 512 }),
                                      T(d_g[1], RAD_BF16, { M, 1 }) };
        std::vector<RadParam> ps = { PI("M", M), PI("K", 2560), PI("n_expert", 512),
                                     PI("top_k", 10), PI("norm", 1), PS("dtype", "bf16") };
        run_case(rg, ts, ps, RAD_E_STATE);
    }
    rg->fini(inst);
}

/* Protected experts are sorted last, checked against a host stable sort. Their rows must follow
 * every other expert's, in the listed order, with the rest closed up and every run stable. Both
 * standalone sort forms run (450 slots: one workgroup; 3000: three passes), with the first, a
 * middle and the last expert of a slice protected. topk_scatter_case covers the fused kernel.
 *
 * The general form also runs at prefill sizes: 79 and 80 tiles of 256 slots (several batches of
 * loads per scan thread, a short last one), on a rank's slice and with all 512 experts local (the
 * mod4 split's case), where every scan thread is an expert. */
static void scatter_protect_at(const RadKernelInfo* rs, int64_t M, int64_t nl, int64_t base);

static void scatter_protect_case(const Plugin& pl) {
    if (!sel("moe_scatter_i32")) return;
    const RadKernelInfo* rs = find(pl, "moe_scatter_i32");
    if (!rs) { ok("moe_scatter protect", false, "row absent"); return; }
    for (const int64_t M : { 45, 300, 2000 }) scatter_protect_at(rs, M, 254, 3);
    scatter_protect_at(rs, 2048, 512, 0);
}

static void scatter_protect_at(const RadKernelInfo* rs, int64_t M, int64_t nl, int64_t base) {
    const int64_t ne = 512, k = 10;
    const std::vector<int64_t> prot = { 0, 17, 101, 253 };
    std::vector<int64_t> place((size_t)nl);
    {
        int64_t next = 0;
        for (int64_t e = 0; e < nl; ++e)
            if (std::find(prot.begin(), prot.end(), e) == prot.end()) place[(size_t)e] = next++;
        for (int64_t p : prot) place[(size_t)p] = next++;
    }
    {
        const int64_t NS = M * k;
        const std::string at = " (" + std::to_string(NS) + " slots, slice " + std::to_string(nl) +
                               ")";
        /* Each row's k distinct experts from the whole router, a quarter in this slice, the
         * protected ones over-represented so their runs are long. */
        std::vector<int32_t> ids((size_t)NS);
        uint64_t rs64 = 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
        auto rnd = [&]() { rs64 ^= rs64 << 13; rs64 ^= rs64 >> 7; rs64 ^= rs64 << 17; return rs64; };
        for (int64_t m = 0; m < M; ++m)
            for (int64_t j = 0; j < k; ++j) {
                int32_t g;
                bool dup;
                do {
                    const uint64_t r = rnd();
                    g = (r % 5 == 0) ? (int32_t)(base + prot[(size_t)((r >> 8) % prot.size())])
                                     : (int32_t)((r >> 8) % ne);
                    dup = false;
                    for (int64_t q = 0; q < j; ++q) dup |= ids[(size_t)(m * k + q)] == g;
                } while (dup);
                ids[(size_t)(m * k + j)] = g;
            }
        std::vector<int32_t> want_s((size_t)NS, -1), want_o((size_t)(nl + 1), 0),
                             want_c((size_t)nl, 0);
        for (int64_t i = 0; i < NS; ++i) {
            const int64_t l = ids[(size_t)i] - base;
            if (l >= 0 && l < nl) ++want_c[(size_t)place[(size_t)l]];
        }
        for (int64_t p = 0; p < nl; ++p) want_o[(size_t)(p + 1)] = want_o[(size_t)p] + want_c[(size_t)p];
        {
            std::vector<int32_t> cur(want_o.begin(), want_o.end() - 1);
            for (int64_t i = 0; i < NS; ++i) {
                const int64_t l = ids[(size_t)i] - base;
                if (l >= 0 && l < nl) want_s[(size_t)cur[(size_t)place[(size_t)l]]++] = (int32_t)i;
            }
        }
        void* d_id = dalloc((size_t)NS * 4);
        void* d_so = dalloc((size_t)NS * 4);
        void* d_of = dalloc((size_t)(nl + 1) * 4);
        void* d_cn = dalloc((size_t)nl * 4);
        if (!d_id || !d_so || !d_of || !d_cn) { ok(("moe_scatter protect" + at).c_str(), false, "alloc"); return; }
        st_memcpy(d_id, ids.data(), (size_t)NS * 4, hipMemcpyHostToDevice);
        st_memset(d_so, 0x5A, (size_t)NS * 4);
        st_memset(d_of, 0x5A, (size_t)(nl + 1) * 4);
        st_memset(d_cn, 0x5A, (size_t)nl * 4);
        std::vector<RadTensor> ts = { T(d_id, RAD_I32, { M, k }), T(d_so, RAD_I32, { NS }),
                                      T(d_of, RAD_I32, { nl + 1 }), T(d_cn, RAD_I32, { nl }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                     PI("expert_base", base), PS("protect", "0,17,101,253") };
        run_case(rs, ts, ps);
        std::vector<int32_t> got_s((size_t)NS), got_o((size_t)(nl + 1)), got_c((size_t)nl);
        st_memcpy(got_s.data(), d_so, (size_t)NS * 4, hipMemcpyDeviceToHost);
        st_memcpy(got_o.data(), d_of, (size_t)(nl + 1) * 4, hipMemcpyDeviceToHost);
        st_memcpy(got_c.data(), d_cn, (size_t)nl * 4, hipMemcpyDeviceToHost);
        const int64_t prot_rows = want_o[(size_t)nl] - want_o[(size_t)(nl - prot.size())];
        ok(("moe_scatter puts the protected experts last" + at).c_str(),
           got_s == want_s && got_o == want_o && got_c == want_c,
           std::to_string(prot_rows) + " protected rows of " + std::to_string(want_o[(size_t)nl]) +
               " live; sorted " + (got_s == want_s ? "ok" : "DIFFERS") + ", offsets " +
               (got_o == want_o ? "ok" : "DIFFER") + ", counts " + (got_c == want_c ? "ok" : "DIFFER"));
        /* A list that is not ascending, or reaches past the slice, is refused. */
        const std::string past = "0," + std::to_string(nl);
        for (const char* badp : { "17,0", past.c_str() }) {
            std::vector<RadParam> pb = { PI("M", M), PI("n_expert", nl), PI("top_k", k),
                                         PI("expert_base", base), PS("protect", badp) };
            run_case(rs, ts, pb, RAD_E_SHAPE);
        }
    }
}

/* moe_gather's forms, against each other byte for byte.
 *
 * Past the self-scan cap the op builds the slot inverse in scratch and sums each token's slots,
 * using gather_wide_kernel (a wave per token, sixteen bytes per lane) where the slots fit its
 * registers and every row starts on sixteen bytes, and gather_kernel (a column per thread)
 * elsewhere. r4d_moe_gather_pin_form pins each (1 column per thread, 2 wave per token), and every
 * form must write the column-per-thread bytes: with and without the shared arm, under both gates,
 * with dropped slots, at a width ending inside a column group, and at Qwen3.8-Flash-Next's
 * 2048-token chunk on a rank of four (also timed). A decode step (under the cap), a top_k past the
 * wide form's registers, and a width not tiled by sixteen-byte pieces must refuse the wide pin.
 * `wide` says whether the shape takes the wide form. */
struct MoeGatherShape { int64_t M, n, k; const char* act; bool wide; };

static void moe_gather_at(const Plugin& pl, const RadKernelInfo* r, const MoeGatherShape& sh) {
    const int64_t M = sh.M, n = sh.n, k = sh.k, NS = M * k;
    const bool shared = sh.act != nullptr;
    const std::string at = " (M=" + std::to_string(M) + ", n=" + std::to_string(n) + ", top " +
                           std::to_string(k) +
                           (shared ? std::string(", shared arm, ") + sh.act : std::string()) + ")";
    uint64_t rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)M * 1315423911ull) ^
                  ((uint64_t)n * 2654435761ull) ^ (uint64_t)k;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    /* Each slot to one of 64 experts, or dropped one in thirty-one; `sorted` is the stable sort by
     * expert, live slots first and -1 after, as moe_scatter writes it. */
    std::vector<int32_t> ex((size_t)NS), sorted((size_t)NS, -1);
    for (int64_t i = 0; i < NS; ++i) {
        const uint64_t v = rnd();
        ex[(size_t)i] = (v % 31 == 0) ? -1 : (int32_t)((v >> 8) % 64);
    }
    {
        int64_t pos = 0;
        for (int32_t e = 0; e < 64; ++e)
            for (int64_t i = 0; i < NS; ++i)
                if (ex[(size_t)i] == e) sorted[(size_t)pos++] = (int32_t)i;
    }
    std::vector<uint16_t> ye((size_t)(NS * n)), ew((size_t)(M * k)), shv, sgv;
    for (auto& v : ye) v = h_to_bf16((float)((int64_t)(rnd() % 4001) - 2000) / 512.0f);
    for (auto& v : ew) v = h_to_bf16((float)(rnd() % 1000 + 1) / 1024.0f);
    if (shared) {
        shv.resize((size_t)(M * n));
        sgv.resize((size_t)M);
        for (auto& v : shv) v = h_to_bf16((float)((int64_t)(rnd() % 4001) - 2000) / 256.0f);
        for (auto& v : sgv) v = h_to_bf16((float)((int64_t)(rnd() % 801) - 400) / 100.0f);
    }
    void* d_ye = dalloc(ye.size() * 2);
    void* d_ew = dalloc(ew.size() * 2);
    void* d_so = dalloc(sorted.size() * 4);
    void* d_sh = shared ? dalloc(shv.size() * 2) : nullptr;
    void* d_sg = shared ? dalloc(sgv.size() * 2) : nullptr;
    void* d_y  = dalloc((size_t)(M * n * 2));
    if (!d_ye || !d_ew || !d_so || !d_y || (shared && (!d_sh || !d_sg))) {
        ok(("moe_gather forms" + at).c_str(), false, "hipMalloc failed");
        return;
    }
    st_memcpy(d_ye, ye.data(), ye.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_ew, ew.data(), ew.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_so, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    if (shared) {
        st_memcpy(d_sh, shv.data(), shv.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_sg, sgv.data(), sgv.size() * 2, hipMemcpyHostToDevice);
    }
    std::vector<RadTensor> ts = { T(d_ye, RAD_BF16, { NS, n }), T(d_ew, RAD_BF16, { M, k }),
                                  T(d_so, RAD_I32, { NS }),
                                  T(d_sh, RAD_BF16, { M, n }), T(d_sg, RAD_BF16, { M, 1 }),
                                  T(d_y, RAD_BF16, { M, n }) };
    if (!shared) { absent(ts[3]); absent(ts[4]); }
    std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("top_k", k) };
    if (shared) ps.push_back(PS("act", sh.act));
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int64_t need = r->scratch ? r->scratch(&a) : 0;
    if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }

    /* One launch at a pinned form, over stale bytes; its return code. */
    auto run = [&](int form, std::vector<uint16_t>& out) -> int {
        std::vector<uint16_t> junk((size_t)(M * n), 0x7f7fu);
        st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        pl.pin_gather(form);
        const int rc = r->launch(&a, nullptr);
        const hipError_t he = (hipError_t)st_stream_sync(nullptr);
        pl.pin_gather(0);
        if (rc == RAD_OK && he != hipSuccess) return RAD_E_DEVICE;
        out.resize((size_t)(M * n));
        st_memcpy(out.data(), d_y, out.size() * 2, hipMemcpyDeviceToHost);
        return rc;
    };
    std::vector<uint16_t> base, got;
    if (const int rc = run(1, base); rc != RAD_OK) {
        ok(("moe_gather, a column a thread" + at).c_str(), false, "rc=" + std::to_string(rc));
        return;
    }
    struct Var { int form; const char* name; bool wide; };
    static const Var kVar[] = {
        { 0, "the rule",                                  false },
        { 2, "a wave a token",                            true  },
    };
    for (const Var& v : kVar) {
        const std::string label = "moe_gather, " + std::string(v.name) + at;
        const int rc = run(v.form, got);
        if (v.wide && !sh.wide) {
            /* A shape the wide form cannot take refuses the pin rather than run another form. */
            ok((label + " is refused").c_str(), rc == RAD_E_UNSUPPORTED, "rc=" + std::to_string(rc));
            continue;
        }
        if (rc != RAD_OK) { ok(label.c_str(), false, "rc=" + std::to_string(rc)); continue; }
        int64_t diff = 0;
        std::string first;
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != base[i]) {
                if (!diff)
                    first = ", first at token " + std::to_string((int64_t)i / n) + " col " +
                            std::to_string((int64_t)i % n);
                ++diff;
            }
        ok(label.c_str(), diff == 0,
           std::to_string(diff) + " of " + std::to_string(M * n) +
               " differ from a column a thread" + first);
    }

    if (M >= 2048 && sh.wide) {
        hipEvent_t e0, e1;
        st_ev_create(&e0);
        st_ev_create(&e1);
        auto time = [&](int form) -> float {
            const int it = 20;
            pl.pin_gather(form);
            for (int w = 0; w < 3; ++w) r->launch(&a, nullptr);
            st_ev_record(e0, nullptr);
            for (int i = 0; i < it; ++i) r->launch(&a, nullptr);
            st_ev_record(e1, nullptr);
            st_ev_sync(e1);
            float ms = 0.0f;
            st_ev_elapsed(&ms, e0, e1);
            pl.pin_gather(0);
            return ms * 1000.0f / (float)it;
        };
        const float t1 = time(1), t2 = time(2);
        std::printf("      moe_gather at %lld routed rows x %lld: a column a thread %.1f us, a wave "
                    "a token %.1f us a call, prepass included\n",
                    (long long)NS, (long long)n, t1, t2);
        st_ev_destroy(e0);
        st_ev_destroy(e1);
    }
}

static void moe_gather_case(const Plugin& pl) {
    if (!sel("moe_gather_bf16")) return;
    const RadKernelInfo* r = find(pl, "moe_gather_bf16");
    if (!r) { ok("moe_gather_bf16", false, "row absent"); return; }
    if (!pl.pin_gather) { ok("moe_gather_bf16 forms", false, "pin hook absent"); return; }
    static const MoeGatherShape kSh[] = {
        {  600, 2560, 10, "sigmoid", true  },
        {  600, 2560, 10, nullptr,   true  },
        {   77,  520, 10, "none",    true  },
        {  300,  264,  8, "sigmoid", true  },
        {   40, 2560, 10, "sigmoid", false },   /* a decode step: the self-scan form */
        {  100,  512, 20, "sigmoid", false },   /* top_k past the wide form's registers */
        {  100,  260, 10, nullptr,   false },   /* a width sixteen-byte pieces do not tile */
        { 2048, 2560, 10, "sigmoid", true  },   /* Qwen3.8-Flash-Next's chunk, timed */
    };
    for (const MoeGatherShape& sh : kSh) moe_gather_at(pl, r, sh);
}

/* ------------------------------------------------------- the vocab-parallel top-R, end to end
 *
 * A top-R over one plane must equal the merge of the two half-plane top-Rs, index for index and
 * value for value. This covers the `vocab_off` shift, the `pairs` plane row_topk writes and the
 * merge at once; a sharded draft head must propose exactly what the replicated one does.
 *
 * The values are full of ties (a 23-value alphabet over 2048 columns), because a dense top-R
 * breaks ties on the lower column and a merge that broke them on rank order would disagree.
 *
 * The gathered plane is assembled as `all_gather` with `row = R*2` does: rank r's R pairs at
 * [r*R, (r+1)*R) of each row. */
static void rowtopk_shard_case(const Plugin& pl) {
    if (!sel("rowtopk_merge") && !sel("rowtopk_bf16")) return;
    const RadKernelInfo* rt = find(pl, "rowtopk_bf16");
    const RadKernelInfo* rm = find(pl, "rowtopk_merge");
    if (!rt || !rm) { ok("rowtopk_merge", false, "row absent"); return; }

    const int64_t M = 5, N = 2048, R = 8, W = 2, SH = N / W;
    std::vector<uint16_t> hx((size_t)M * N);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t c = 0; c < N; ++c)
            hx[(size_t)(m * N + c)] = h_to_bf16((float)((m * 31 + c * 7) % 23) - 11.0f);

    void* d_x  = dalloc((size_t)M * N * 2);
    void* d_i0 = dalloc((size_t)M * R * 4);
    void* d_v0 = dalloc((size_t)M * R * 4);
    void* d_i  = dalloc((size_t)M * R * 4);
    void* d_v  = dalloc((size_t)M * R * 4);
    void* d_p  = dalloc((size_t)M * R * 2 * 4);
    void* d_g  = dalloc((size_t)M * W * R * 2 * 4);
    void* d_h  = dalloc((size_t)M * SH * 2);
    if (!d_x || !d_i0 || !d_v0 || !d_i || !d_v || !d_p || !d_g || !d_h) {
        ok("rowtopk_merge", false, "hipMalloc failed"); return;
    }
    st_memcpy(d_x, hx.data(), (size_t)M * N * 2, hipMemcpyHostToDevice);

    /* The answer to agree with: one top-R over the whole plane. */
    {
        std::vector<RadTensor> ts = { T(d_x, RAD_BF16, { M, N }), T(d_i0, RAD_I32, { M, R }),
                                      T(d_v0, RAD_F32, { M, R }) };
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("R", R), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
    }
    std::vector<int32_t> wi((size_t)M * R);
    std::vector<float>   wv((size_t)M * R);
    st_memcpy(wi.data(), d_i0, (size_t)M * R * 4, hipMemcpyDeviceToHost);
    st_memcpy(wv.data(), d_v0, (size_t)M * R * 4, hipMemcpyDeviceToHost);

    /* Each rank's own slice, reporting global columns, plus the pairs an all_gather would move.
     * The slice is copied out: the top-R refuses a strided plane, and under a real shard each
     * rank's plane is contiguous and `SH` wide anyway. */
    std::vector<uint16_t> half((size_t)M * SH);
    std::vector<float> gath((size_t)M * W * R * 2);
    for (int64_t r = 0; r < W; ++r) {
        for (int64_t m = 0; m < M; ++m)
            for (int64_t c = 0; c < SH; ++c)
                half[(size_t)(m * SH + c)] = hx[(size_t)(m * N + r * SH + c)];
        st_memcpy(d_h, half.data(), half.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = {
            T(d_h, RAD_BF16, { M, SH }),
            T(d_i, RAD_I32, { M, R }), T(d_v, RAD_F32, { M, R }),
            T(d_p, RAD_F32, { M, R, 2 }),
        };
        std::vector<RadParam> ps = { PI("M", M), PI("N", SH), PI("R", R),
                                     PI("vocab_off", r * SH), PS("dtype", "bf16") };
        run_case(rt, ts, ps);
        std::vector<float> pr((size_t)M * R * 2);
        st_memcpy(pr.data(), d_p, (size_t)M * R * 2 * 4, hipMemcpyDeviceToHost);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < R * 2; ++k)
                gath[(size_t)((m * W + r) * R * 2 + k)] = pr[(size_t)(m * R * 2 + k)];
    }
    st_memcpy(d_g, gath.data(), gath.size() * 4, hipMemcpyHostToDevice);

    {
        std::vector<RadTensor> ts = { T(d_g, RAD_F32, { M, W * R, 2 }), T(d_i, RAD_I32, { M, R }),
                                      T(d_v, RAD_F32, { M, R }) };
        std::vector<RadParam> ps = { PI("M", M), PI("R", R), PI("world_size", W),
                                     PS("dtype", "bf16") };
        run_case(rm, ts, ps);
    }
    std::vector<int32_t> gi((size_t)M * R);
    std::vector<float>   gv((size_t)M * R);
    st_memcpy(gi.data(), d_i, (size_t)M * R * 4, hipMemcpyDeviceToHost);
    st_memcpy(gv.data(), d_v, (size_t)M * R * 4, hipMemcpyDeviceToHost);

    int bad = 0;
    std::string first;
    for (size_t i = 0; i < gi.size() && !bad; ++i)
        if (gi[i] != wi[i] || gv[i] != wv[i]) {
            ++bad;
            first = "row " + std::to_string(i / (size_t)R) + " slot " +
                    std::to_string(i % (size_t)R) + ": merged (" + std::to_string(gi[i]) + ", " +
                    std::to_string(gv[i]) + ") want (" + std::to_string(wi[i]) + ", " +
                    std::to_string(wv[i]) + ")";
        }
    ok("rowtopk_merge matches an unsharded top-R", bad == 0, bad ? first : "");
}
/* --------------------------- the attention with the output gate and quantiser in its merge, vs the pair
 *
 * `attn_decode_gq_h256_gqa12_<kv>` is `attn_decode_h256_gqa12_<kv>` followed by `gate_quant_fp8`
 * over its output, for E4M3 and bf16 caches, and must be byte-identical to that pair: the gated
 * bf16 product in `out`, every code and every scale. The engine's sparse-path geometry: 12 query
 * heads over one kv head of 256, 4-token pages, one query row per sequence, the gate a column view
 * of a [q|gate] plane as the block passes it.
 *
 * Two batch sizes cover the merge's two forms: 3 rows (36 (row, head) jobs, one thread per column)
 * and 24 rows (288, one wave per row). Each has one padded row (no context), which gets zeros, a
 * zero scale group and a scale of 1. The E4M3 cache is random bytes except NaN codes, the bf16 one
 * random finite values, and the descales are not 1.
 *
 * `i8` runs both at int8 codes (the operand an int8 out projection reads), with the same claim. */
static void attn_gq_fold_case(const Plugin& pl, int64_t S, bool bf16, bool i8 = false) {
    const std::string kvs = bf16 ? "bf16kv" : "fp8kv";
    const std::string fold = "attn_decode_gq_h256_gqa12_" + kvs;
    const std::string pair = "attn_decode_h256_gqa12_" + kvs;
    if (!sel(fold.c_str())) return;
    const std::string name = fold + " is byte-identical to the pair at " +
                             std::to_string((long long)S) + " rows" + (i8 ? ", int8 codes" : "");
    const uint32_t cdt = i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
    const RadKernelInfo* rk = find(pl, fold.c_str());
    const RadKernelInfo* ra = find(pl, pair.c_str());
    const RadKernelInfo* rg = find(pl, "gate_quant_fp8");
    if (!rk || !ra || !rg) { ok(name.c_str(), false, "row absent"); return; }

    const int64_t nq = 12, nkv = 1, hd = 256, bs = 4, nb = 96, maxb = 40;
    uint32_t st = 0x2545f491u ^ (uint32_t)S;
    auto rnd = [&](float span) {
        st = st * 1664525u + 1013904223u;
        return ((float)(st >> 8) / 16777216.0f * 2.0f - 1.0f) * span;
    };
    std::vector<int32_t> table((size_t)(S * maxb)), used((size_t)S);
    for (int64_t r = 0; r < S; ++r) {
        for (int64_t b = 0; b < maxb; ++b)
            table[(size_t)(r * maxb + b)] = (int32_t)((r * 13 + b * 7 + 1) % nb);
        used[(size_t)r] = (int32_t)(maxb * bs - (r % 5) * 3);
    }
    used[1] = 0;                                                   /* a padded row */
    std::vector<uint16_t> hq((size_t)(S * nq * hd)), hqg((size_t)(S * nq * 2 * hd));
    for (auto& x : hq)  x = h_to_bf16(rnd(2.0f));
    for (auto& x : hqg) x = h_to_bf16(rnd(6.0f));
    const size_t cbytes = (size_t)(nb * nkv * bs * 2 * hd) * (bf16 ? 2 : 1);
    std::vector<uint8_t> hc(cbytes);
    if (bf16) {
        for (size_t i = 0; i + 1 < cbytes; i += 2) {
            const uint16_t h = h_to_bf16(rnd(3.0f));
            std::memcpy(&hc[i], &h, 2);
        }
    } else {
        for (auto& x : hc) {
            x = (uint8_t)(rnd(128.0f) + 128.0f);
            if ((x & 0x7f) == 0x7f) x ^= 1;                        /* E4M3's NaN codes */
        }
    }
    std::vector<float> hds((size_t)(S * nkv));
    for (auto& x : hds) x = 0.75f + 0.5f * (rnd(1.0f) + 1.0f) * 0.5f;

    const size_t qb = hq.size() * 2, ob = (size_t)(S * nq * hd) * 2;
    const size_t q8b = (size_t)(S * nq * hd), sb = (size_t)(S * nq * (hd / 128)) * 4;
    void* d_q  = dalloc(qb);          void* d_qg = dalloc(hqg.size() * 2);
    void* d_c  = dalloc(cbytes);
    void* d_kd = dalloc(hds.size() * 4); void* d_vd = dalloc(hds.size() * 4);
    void* d_oA = dalloc(ob);  void* d_oB = dalloc(ob);
    void* d_qA = dalloc(q8b); void* d_qB = dalloc(q8b);
    void* d_sA = dalloc(sb);  void* d_sB = dalloc(sb);
    if (!d_q || !d_qg || !d_c || !d_kd || !d_vd || !d_oA || !d_oB || !d_qA || !d_qB || !d_sA ||
        !d_sB) { ok(name.c_str(), false, "hipMalloc failed"); return; }
    st_memcpy(d_q,  hq.data(),  qb, hipMemcpyHostToDevice);
    st_memcpy(d_qg, hqg.data(), hqg.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_c,  hc.data(),  cbytes, hipMemcpyHostToDevice);
    st_memcpy(d_kd, hds.data(), hds.size() * 4, hipMemcpyHostToDevice);
    std::reverse(hds.begin(), hds.end());
    st_memcpy(d_vd, hds.data(), hds.size() * 4, hipMemcpyHostToDevice);

    /* The gate: head h's second half of the [q|gate] row, one pitch of 2 * head_dim a head. */
    RadTensor gate = T((uint8_t*)d_qg + hd * 2, RAD_BF16, { S, nq, hd });
    gate.stride[0] = nq * 2 * hd;
    gate.stride[1] = 2 * hd;
    auto attn_ops = [&](void* out) {
        return std::vector<RadTensor>{
            T(d_q, RAD_BF16, { S, nq, hd }),
            T(d_c, bf16 ? RAD_BF16 : RAD_F8E4M3, { nb, nkv, bs, 2 * hd }),
            TI32({ S, maxb }, table),
            TI32({ S }, used),
            T(d_kd, RAD_F32, { S, nkv }),
            T(d_vd, RAD_F32, { S, nkv }),
            RadTensor{},
            T(out, RAD_BF16, { S, nq * hd }) };
    };
    std::vector<RadParam> pa = {
        PI("q_len", 1), PI("head_dim", hd), PI("gqa", nq / nkv), PI("block_size", bs),
        PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),
        PS("kv_dtype", bf16 ? "bf16" : "fp8_e4m3"),
        PI("n_head", nq), PI("max_seqs", S), PI("max_ctx", maxb * bs),
    };

    /* ---- the pair ---- */
    {
        std::vector<RadTensor> ts = attn_ops(d_oA);
        run_case(ra, ts, pa);
    }
    {
        std::vector<RadTensor> ts = { gate, T(d_oA, RAD_BF16, { S * nq, hd }),
                                      T(d_qA, cdt, { S * nq, hd }),
                                      T(d_sA, RAD_F32, { S * nq, hd / 128 }),
                                      T(d_oA, RAD_BF16, { S * nq, hd }) };
        std::vector<RadParam> ps = { PI("M", S * nq), PI("n", hd), PI("group", 128),
                                     PS("dtype", "bf16"), PS("act", "sigmoid") };
        run_case(rg, ts, ps);
    }
    /* ---- the fold ---- */
    {
        std::vector<RadTensor> ts = attn_ops(d_oB);
        ts.push_back(gate);
        ts.push_back(T(d_qB, cdt, { S, nq * hd }));
        ts.push_back(T(d_sB, RAD_F32, { S, nq * hd / 128 }));
        std::vector<RadParam> ps = pa;
        ps.push_back(PI("group", 128));
        ps.push_back(PS("act", "sigmoid"));
        run_case(rk, ts, ps);
    }

    std::vector<uint8_t> oa(ob), obv(ob), qa(q8b), qbv(q8b), sa(sb), sbv(sb);
    st_memcpy(oa.data(),  d_oA, ob,  hipMemcpyDeviceToHost);
    st_memcpy(obv.data(), d_oB, ob,  hipMemcpyDeviceToHost);
    st_memcpy(qa.data(),  d_qA, q8b, hipMemcpyDeviceToHost);
    st_memcpy(qbv.data(), d_qB, q8b, hipMemcpyDeviceToHost);
    st_memcpy(sa.data(),  d_sA, sb,  hipMemcpyDeviceToHost);
    st_memcpy(sbv.data(), d_sB, sb,  hipMemcpyDeviceToHost);
    auto ndiff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        int64_t n = 0;
        for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
        return n;
    };
    /* And the pair wrote something: its codes are not all zero. */
    int64_t live = 0;
    for (uint8_t x : qa) live += x != 0;
    const int64_t n_o = ndiff(oa, obv), n_q = ndiff(qa, qbv), n_s = ndiff(sa, sbv);
    char msg[200];
    std::snprintf(msg, sizeof msg, "out %lld of %zu, codes %lld of %zu, scales %lld of %zu bytes "
                  "differ; %lld nonzero codes", (long long)n_o, ob, (long long)n_q, q8b,
                  (long long)n_s, sb, (long long)live);
    ok(name.c_str(), n_o == 0 && n_q == 0 && n_s == 0 && live > 0, msg);
}

/* kv_store, both widths, against an exactly representable set so the check is equality: halves
 * and integers up to 8 survive E4M3 (three mantissa bits) and bf16 unchanged, so `h_e4m3` or
 * `h_bf16` of what landed must equal what went in.
 *
 * The op scatters into kv_cache[n_blocks, kv_heads, block_size, 2*head_dim], K at [.., d] and V at
 * [.., head_dim + d], at (slot / block_size, slot % block_size). Each value is built from (token,
 * head, dim) and K and V have disjoint sign, so a k/v swap, head transposition or off-by-a-block
 * shows.
 *
 * A negative or out-of-range slot (a pad row) must store nothing: the cache is prefilled with a
 * sentinel and those slots must still hold it. Slot 0 is a real slot (as in libref and
 * r4d_gdn_conv_w4_h128_bf16.hip) and is used deliberately. */
static void kv_store_case(const Plugin& pl, const char* name, const char* kv_dtype) {
    if (!sel(name)) return;
    const RadKernelInfo* r = find(pl, name);
    if (!r) { ok(name, false, "row absent"); return; }

    const bool     fp8        = std::strcmp(kv_dtype, "bf16") != 0;
    const int64_t  head_dim   = 256, kv_heads = 2, block_size = 16, n_blocks = 4;
    const int64_t  lane       = kv_heads * head_dim;
    const int64_t  cache_elts = n_blocks * kv_heads * block_size * 2 * head_dim;
    /* Slot 0 is real; -1 is the pad; the last is past the end of the four blocks. */
    const std::vector<int32_t> slots = { 0, 17, 63, -1, (int32_t)(n_blocks * block_size), 33 };
    const int64_t  T_tok      = (int64_t)slots.size();

    /* Exact in E4M3 and bf16. Sign separates K from V, the magnitude encodes head and column, so
     * no two confusable (token, head, dim) triples share a value. */
    auto want = [&](int64_t t, int64_t h, int64_t d, bool is_v) {
        static const float mag[8] = { 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 8.0f };
        const float m = mag[(size_t)((t * 3 + h * 5 + d) % 8)];
        return is_v ? -m : m;
    };

    std::vector<uint16_t> hk((size_t)T_tok * lane), hv((size_t)T_tok * lane);
    for (int64_t t = 0; t < T_tok; ++t)
        for (int64_t h = 0; h < kv_heads; ++h)
            for (int64_t d = 0; d < head_dim; ++d) {
                hk[(size_t)(t * lane + h * head_dim + d)] = h_to_bf16(want(t, h, d, false));
                hv[(size_t)(t * lane + h * head_dim + d)] = h_to_bf16(want(t, h, d, true));
            }

    const size_t cbytes = (size_t)cache_elts * (fp8 ? 1u : 2u);
    void* d_k  = dalloc((size_t)T_tok * lane * 2);
    void* d_v  = dalloc((size_t)T_tok * lane * 2);
    void* d_sm = dalloc((size_t)T_tok * 4);
    void* d_c  = dalloc(cbytes);
    if (!d_k || !d_v || !d_sm || !d_c) { ok(name, false, "hipMalloc failed"); return; }
    st_memcpy(d_k, hk.data(), (size_t)T_tok * lane * 2, hipMemcpyHostToDevice);
    st_memcpy(d_v, hv.data(), (size_t)T_tok * lane * 2, hipMemcpyHostToDevice);
    st_memcpy(d_sm, slots.data(), (size_t)T_tok * 4, hipMemcpyHostToDevice);
    /* 0x7f is NaN in E4M3 and large in bf16; the store cannot produce it from these inputs, so
     * anything still holding it was not written. */
    st_memset(d_c, 0x7f, cbytes);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { T_tok, lane }),
        T(d_v,  RAD_BF16, { T_tok, lane }),
        T(d_sm, RAD_I32,  { T_tok }),
        T(d_c,  fp8 ? RAD_F8E4M3 : RAD_BF16, { n_blocks, kv_heads, block_size, 2 * head_dim }),
    };
    std::vector<RadParam> ps = {
        PI("M", T_tok), PI("head_dim", head_dim), PI("n_head_kv", kv_heads),
        PI("block_size", block_size), PS("kv_dtype", kv_dtype),
    };
    run_case(r, ts, ps);

    std::vector<uint8_t> got(cbytes);
    st_memcpy(got.data(), d_c, cbytes, hipMemcpyDeviceToHost);
    auto at = [&](int64_t i) {
        return fp8 ? h_e4m3(got[(size_t)i])
                   : h_bf16(*(const uint16_t*)(got.data() + (size_t)i * 2));
    };

    int bad = 0, pad_touched = 0;
    std::string first;
    for (int64_t t = 0; t < T_tok && bad < 1; ++t) {
        const int32_t s = slots[(size_t)t];
        const bool    live = s >= 0 && s / block_size < n_blocks;
        for (int64_t h = 0; h < kv_heads && !bad; ++h)
            for (int64_t d = 0; d < head_dim; ++d) {
                if (!live) continue;
                const int64_t blk = s / block_size, off = s % block_size;
                const int64_t base = ((blk * kv_heads + h) * block_size + off) * 2 * head_dim;
                const float gk = at(base + d), gv = at(base + head_dim + d);
                const float wk = want(t, h, d, false), wv = want(t, h, d, true);
                if (gk != wk || gv != wv) {
                    ++bad;
                    first = "tok " + std::to_string((long long)t) + " head " +
                            std::to_string((long long)h) + " dim " + std::to_string((long long)d) +
                            ": k " + std::to_string(gk) + " want " + std::to_string(wk) +
                            ", v " + std::to_string(gv) + " want " + std::to_string(wv);
                    break;
                }
            }
    }
    /* Every element no live slot names must still be the sentinel, checked on raw storage: 0x7f
     * is E4M3's NaN, which h_e4m3 decodes as 480.0, so a decoded comparison would misreport. */
    for (int64_t i = 0; i < cache_elts; ++i) {
        const int64_t blk = i / (kv_heads * block_size * 2 * head_dim);
        const int64_t off = (i / (2 * head_dim)) % block_size;
        bool named = false;
        for (int32_t s : slots)
            if (s >= 0 && s / block_size < n_blocks && s / block_size == blk && s % block_size == off)
                named = true;
        if (named) continue;
        const bool intact = fp8 ? got[(size_t)i] == 0x7f
                                : *(const uint16_t*)(got.data() + (size_t)i * 2) == 0x7f7f;
        if (!intact) {
            ++pad_touched;
            first = "element " + std::to_string((long long)i) + " of block " +
                    std::to_string((long long)blk) + " offset " + std::to_string((long long)off);
            break;
        }
    }

    ok((std::string(name) + " scatters k and v to the slot's block and offset").c_str(),
       bad == 0, first);
    ok((std::string(name) + " leaves a pad slot and every unnamed offset untouched").c_str(),
       pad_touched == 0, pad_touched ? "written: " + first : "");
}

/* The anchor state form, end to end.
 *
 * `gdn_recurrent_update` serves two state contracts with the same arithmetic. Under
 * `per_candidate` every candidate token writes its own state slot and the next step reads the one
 * the last accepted token wrote. Under `anchor` a sequence keeps one state, the step's tokens are
 * recorded beside it as replay factors, and the next step folds the accepted prefix back in. Both
 * give the same output from different caches, so this drives a multi-step schedule against a host
 * walk of the delta rule with rollback:
 *
 *     the state a step starts from is the state after the previous step's first `num_accepted`
 *     tokens.
 *
 * The window moves all three ways: a paged caller re-derives the page index every step, so the six
 * columns slide by -1, 0 or +1 and the kernel finds its anchor through a marker in the factor
 * block. The schedule takes +1, then 0, then -1.
 *
 * A second host chain always rolls back to the first token (what a kernel ignoring `num_accepted`
 * would produce) and must disagree with the device, so the check is sensitive.
 */
static void gdn_recurrent_anchor(const Plugin& pl) {
    const char* row = "gdn_recurrent_update_k128_v128_bf16_fp32state";
    const char* nm  = "gdn_recurrent_update anchor replays the accepted prefix";

    if (!sel(row) && !sel(nm)) return;
    const RadKernelInfo* r = find(pl, row);
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t K = 128, V = 128, H = 2, Hg = 1, N = 2, NT = 3, Td = N * NT;
    const int64_t COLS = 6;               /* three copies of the anchor, three of the factors */
    const int64_t PAGES = 7;              /* the widest window the schedule below slides over */
    const int64_t SLOTS = N * PAGES;
    const double  scale = 0.088388, neps = 1e-6;
    /* `win` is the window's first page, `nacc` what the previous step committed. The first step
     * bootstraps: its factor block is a fresh slot's zeros, the marker reads as absent, and
     * replaying one zeroed token leaves the anchor alone. */
    struct Step { int nacc, win; };
    const Step sched[] = { {1, 0}, {2, 1}, {3, 1}, {1, 0} };
    const int NS = (int)(sizeof(sched) / sizeof(sched[0]));

    std::vector<uint16_t> hq((size_t)(Td * Hg * K)), hkk((size_t)(Td * Hg * K)),
                          hv((size_t)(Td * H * V)), ha((size_t)(Td * H)), hb((size_t)(Td * H)),
                          hz((size_t)(Td * H * V)), hdev((size_t)(Td * H * V));
    std::vector<float> halog((size_t)H), hdtb((size_t)H), hnw((size_t)V);
    std::vector<float> hstate((size_t)(SLOTS * H * V * K), 0.0f);
    std::vector<int32_t> hsidx((size_t)(N * COLS)), hnacc((size_t)N);

    uint32_t rs = 0x9e3779b9u;
    auto rnd = [&rs]() {
        rs = rs * 1664525u + 1013904223u;
        return (float)((rs >> 9) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    /* A_log gives a decay of a few percent per token (exp(A_log) * softplus(a + dt_bias) ~ 0.05),
     * so the state neither freezes nor forgets within twelve tokens. */
    for (int64_t h = 0; h < H; ++h) { halog[(size_t)h] = -2.0f + 0.25f * (float)h;
                                      hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h; }
    for (int64_t j = 0; j < V; ++j) hnw[(size_t)j] = 1.0f + 0.25f * rnd();
    /* The state the first step must find at its anchor column. */
    for (int64_t n = 0; n < N; ++n) {
        float* p = &hstate[(size_t)((n * PAGES + sched[0].win + 2) * H * V * K)];
        for (int64_t i = 0; i < H * V * K; ++i) p[(size_t)i] = 0.1f * rnd();
    }

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16,   { Td, Hg, K }),      /* q       */
        T(nullptr, RAD_BF16,   { Td, Hg, K }),      /* k       */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* v       */
        T(nullptr, RAD_BF16,   { Td, H }),          /* a       */
        T(nullptr, RAD_BF16,   { Td, H }),          /* b       */
        T(nullptr, RAD_F32,    { H }),              /* A_log   */
        T(nullptr, RAD_F32,    { H }),              /* dt_bias */
        T(nullptr, RAD_F32,    { SLOTS, H, V, K }), /* state   */
        T(nullptr, RAD_I32,    { N, COLS }),        /* state_idx: six columns, stride 6 */
        T(nullptr, RAD_I32,    { N }),              /* num_accepted */
        T(nullptr, RAD_I32,    { N + 1 }),          /* cu      */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* z       */
        T(nullptr, RAD_F32,    { V }),              /* norm_w  */
        T(nullptr, RAD_BF16,   { Td, H, V }),       /* o       */
        T(nullptr, RAD_F8E4M3, { Td, H * V }),      /* o_q     */
        T(nullptr, RAD_F32,    { Td, H }),          /* o_scale */
    };
    for (auto& t : ts) { t.data = alloc_for(t); if (!t.data) { ok(nm, false, "hipMalloc"); return; } }
    st_memcpy(ts[5].data, halog.data(), halog.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(ts[6].data, hdtb.data(),  hdtb.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(ts[7].data, hstate.data(), hstate.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(ts[12].data, hnw.data(),  hnw.size() * 4,   hipMemcpyHostToDevice);
    {
        const std::vector<int32_t> cu = cu_even(N, Td);
        st_memcpy(ts[10].data, cu.data(), cu.size() * 4, hipMemcpyHostToDevice);
    }

    std::vector<RadParam> ps = {
        PI("q_len", NT), PI("head_k", K), PI("head_v", V), PI("n_head_v", H),
        PF("scale", scale), PF("eps", neps), PS("act", "silu"), PS("state_form", "anchor"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;

    /* Two host chains over the same data: one rolls back to the accepted token, one always to
     * the first. `sn` keeps a state per candidate token for the next step's rollback. */
    const size_t selems = (size_t)(N * H * V * K);
    std::vector<float> S(selems), Sw(selems);
    std::vector<float> sn((size_t)NT * selems), snw((size_t)NT * selems);
    std::vector<float> ref((size_t)(Td * H * V)), refw((size_t)(Td * H * V));
    for (int64_t n = 0; n < N; ++n) {
        const float* p = &hstate[(size_t)((n * PAGES + sched[0].win + 2) * H * V * K)];
        std::memcpy(&S[(size_t)(n * H * V * K)],  p, (size_t)(H * V * K) * 4);
        std::memcpy(&Sw[(size_t)(n * H * V * K)], p, (size_t)(H * V * K) * 4);
    }

    auto walk = [&](std::vector<float>& St, std::vector<float>& snp, std::vector<float>& out) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t h = 0; h < H; ++h) {
                float* s = &St[(size_t)((n * H + h) * V * K)];
                const int64_t hg = h / (H / Hg);
                for (int64_t t = 0; t < NT; ++t) {
                    const int64_t tok = n * NT + t;
                    float qv[128], kv[128];
                    double sq = 0.0, sk = 0.0;
                    for (int64_t j = 0; j < K; ++j) {
                        qv[j] = h_bf16(hq [(size_t)((tok * Hg + hg) * K + j)]);
                        kv[j] = h_bf16(hkk[(size_t)((tok * Hg + hg) * K + j)]);
                        sq += (double)qv[j] * qv[j];
                        sk += (double)kv[j] * kv[j];
                    }
                    const float qs = (float)(1.0 / std::sqrt(sq + 1e-6)) * (float)scale;
                    const float ks = (float)(1.0 / std::sqrt(sk + 1e-6));
                    for (int64_t j = 0; j < K; ++j) { qv[j] *= qs; kv[j] *= ks; }
                    const float x   = h_bf16(ha[(size_t)(tok * H + h)]) + hdtb[(size_t)h];
                    const float spl = x > 20.0f ? x
                                    : (x > 0.0f ? x + std::log1p(std::exp(-x))
                                                : std::log1p(std::exp(x)));
                    const float eg  = std::exp(-std::exp(halog[(size_t)h]) * spl);
                    const float bt  = 1.0f / (1.0f + std::exp(-h_bf16(hb[(size_t)(tok * H + h)])));
                    float on[128];
                    double tot = 0.0;
                    for (int64_t rw = 0; rw < V; ++rw) {
                        float* sr = s + rw * K;
                        double dot = 0.0;
                        for (int64_t j = 0; j < K; ++j) { sr[j] *= eg; dot += (double)sr[j] * kv[j]; }
                        const float u = (h_bf16(hv[(size_t)((tok * H + h) * V + rw)]) - (float)dot) * bt;
                        double o1 = 0.0;
                        for (int64_t j = 0; j < K; ++j) { sr[j] += u * kv[j]; o1 += (double)sr[j] * qv[j]; }
                        on[rw] = (float)o1;
                        tot += o1 * o1;
                    }
                    const float sc = (float)(1.0 / std::sqrt(tot / (double)V + neps));
                    for (int64_t rw = 0; rw < V; ++rw) {
                        const float zz = h_bf16(hz[(size_t)((tok * H + h) * V + rw)]);
                        out[(size_t)((tok * H + h) * V + rw)] =
                            on[rw] * sc * hnw[(size_t)rw] * (zz / (1.0f + std::exp(-zz)));
                    }
                    std::memcpy(&snp[(size_t)t * selems + (size_t)((n * H + h) * V * K)],
                                s, (size_t)(V * K) * 4);
                }
            }
    };

    double worst = 0.0, apart = 0.0;
    int bad_rc = 0;
    for (int s = 0; s < NS && !bad_rc; ++s) {
        for (size_t i = 0; i < hq.size(); ++i)  { hq[i]  = h_to_bf16(rnd()); hkk[i] = h_to_bf16(rnd()); }
        for (size_t i = 0; i < hv.size(); ++i)  { hv[i]  = h_to_bf16(rnd()); hz[i]  = h_to_bf16(rnd()); }
        for (size_t i = 0; i < ha.size(); ++i)  { ha[i]  = h_to_bf16(rnd()); hb[i]  = h_to_bf16(rnd()); }
        st_memcpy(ts[0].data, hq.data(),  hq.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[1].data, hkk.data(), hkk.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(ts[2].data, hv.data(),  hv.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[3].data, ha.data(),  ha.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[4].data, hb.data(),  hb.size() * 2,  hipMemcpyHostToDevice);
        st_memcpy(ts[11].data, hz.data(), hz.size() * 2,  hipMemcpyHostToDevice);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < COLS; ++c)
                hsidx[(size_t)(n * COLS + c)] = (int32_t)(n * PAGES + sched[s].win + c);
            hnacc[(size_t)n] = sched[s].nacc;
        }
        st_memcpy(ts[8].data, hsidx.data(), hsidx.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(ts[9].data, hnacc.data(), hnacc.size() * 4, hipMemcpyHostToDevice);
        st_memset(ts[13].data, 0, hdev.size() * 2);

        const int rc = r->launch(&a, (RadStream) nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            ok(nm, false, "step " + std::to_string(s) + " rc=" + std::to_string(rc));
            bad_rc = 1;
            break;
        }
        st_memcpy(hdev.data(), ts[13].data, hdev.size() * 2, hipMemcpyDeviceToHost);

        if (s > 0) {
            const size_t back = (size_t)(sched[s].nacc - 1);
            std::memcpy(S.data(),  &sn [back * selems],  selems * 4);
            std::memcpy(Sw.data(), &snw[0],              selems * 4);  /* the prefix ignored */
        }
        walk(S,  sn,  ref);
        walk(Sw, snw, refw);
        /* Relative L2, not a per-element maximum: the gated norm leaves many outputs near zero,
         * where a per-element ratio would inflate rounding differences. */
        double en = 0.0, ew = 0.0, rn = 0.0;
        for (size_t i = 0; i < hdev.size(); ++i) {
            const double d = h_bf16(hdev[i]);
            en += (d - ref[i])  * (d - ref[i]);
            ew += (d - refw[i]) * (d - refw[i]);
            rn += (double)ref[i] * ref[i];
        }
        rn = rn > 0.0 ? std::sqrt(rn) : 1.0;
        worst = std::max(worst, std::sqrt(en) / rn);
        if (s == NS - 1) apart = std::sqrt(ew) / rn;
    }
    if (bad_rc) return;

    char d[192];
    std::snprintf(d, sizeof d, "relative L2 %.3g over %d steps, %lld tokens; a replay that "
                  "ignored num_accepted would be %.3g away", worst, NS, (long long)(NS * Td), apart);
    ok(nm, worst < 5e-3 && apart > 0.05, d);
}

/* Codes or scales that differ from quant_act_i8g's over the bf16 rows `x`: one f32 scale per
 * (row, 128 columns) at amax / 127 (1 for a zero group), each code the value over it rounded to
 * nearest-even. The producers that write int8 codes themselves are held to exactly this. */
static int64_t i8_code_misses(const uint16_t* x, int64_t x_ld, const int8_t* q, int64_t q_ld,
                              const float* s, int64_t s_ld, int64_t rows, int64_t n) {
    int64_t miss = 0;
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t g = 0; g < n / 128; ++g) {
            const uint16_t* xr = x + i * x_ld + g * 128;
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(h_bf16(xr[j])));
            const float sc = amax > 0 ? amax * (1.0f / 127.0f) : 1.0f;
            miss += s[i * s_ld + g] != sc;
            const float inv = 1.0f / sc;
            for (int64_t j = 0; j < 128; ++j) {
                const float v = std::max(-127.0f, std::min(127.0f, h_bf16(xr[j]) * inv));
                miss += q[i * q_ld + g * 128 + j] != (int)std::nearbyint(v);
            }
        }
    return miss;
}

/* The decode convolution folded into the recurrent update, against the pair it replaces.
 *
 * gdn_conv_recurrent_update must leave every byte gdn_conv_update + gdn_recurrent_update leave:
 * output, fp8 codes and scales, recurrent state and conv state. Both run from identical copies
 * and all five are compared over two steps; the second catches an arrival counter that does not
 * reset (the key heads' conv state would then not be rewritten).
 *
 * Two value heads per key head (so the last-arriver rewrite runs), uneven windows and read
 * offsets, sequence 1 at conv slot 0 (a real slot a `<= 0` test would drop), and a last sequence
 * of padding with every slot negative. `x` and `z` are column views of one projection row, the
 * gate after the convolution's columns, as the block lays it out.
 *
 * The key heads' conv state must have moved, so the check is sensitive. The anchor form runs in a
 * child process, as in gdn_recurrent_anchor. `i8` runs both at int8 codes and also holds the pair's
 * codes to quant_act_i8g over its `o`. */
static void gdn_conv_fold_run(const Plugin& pl, const char* nm, const char* form,
                              bool i8 = false) {
    const RadKernelInfo* rf = find(pl, "gdn_conv_recurrent_update_k128_v128_bf16");
    const RadKernelInfo* rc = find(pl, "gdn_conv_update_w4_h128_bf16");
    const RadKernelInfo* rr = find(pl, "gdn_recurrent_update_k128_v128_bf16_fp32state");
    if (!rf || !rc || !rr) { ok(nm, false, "a row is absent"); return; }

    const int64_t K = 128, V = 128, H = 4, Hg = 2, N = 4, W = 4, MQ = 4, SL = W - 2 + MQ;
    const int64_t conv_dim = (2 * Hg * K + H * V), in_w = conv_dim + H * V;
    const int64_t COLS = 6, SSLOTS = N * COLS, CSLOTS = 5;
    const int32_t lens[N] = { 4, 2, 3, 1 }, nacc[N] = { 2, 1, 3, 1 }, cidx[N] = { 3, 0, 1, -1 };
    int64_t Td = 0;
    std::vector<int32_t> cu(1, 0);
    for (int64_t n = 0; n < N; ++n) { Td += lens[n]; cu.push_back((int32_t)Td); }

    uint32_t seed = 0x51ed270bu;
    auto rnd = [&seed]() {
        seed = st_mix32(seed + 0x9e3779b9u);
        return (float)(seed & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    auto bf = [&](size_t n, float s) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = h_to_bf16(s * rnd());
        return v;
    };
    const std::vector<uint16_t> hw = bf((size_t)(conv_dim * W), 0.5f);
    const std::vector<uint16_t> hcs = bf((size_t)(CSLOTS * conv_dim * SL), 1.0f);
    std::vector<float> hst((size_t)(SSLOTS * H * V * K)), halog((size_t)H), hdtb((size_t)H),
                       hnw((size_t)V);
    for (auto& e : hst) e = 0.1f * rnd();
    for (int64_t h = 0; h < H; ++h) { halog[(size_t)h] = -2.0f + 0.25f * (float)h;
                                      hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h; }
    for (auto& e : hnw) e = 1.0f + 0.25f * rnd();
    std::vector<int32_t> hsidx((size_t)(N * COLS));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < COLS; ++c)
            hsidx[(size_t)(n * COLS + c)] = cidx[n] < 0 ? -1 : (int32_t)(n * COLS + c);

    /* Two of everything the ops write; the shared inputs once. */
    auto dev = [&](const void* h, size_t bytes) {
        void* d = dalloc(bytes);
        if (d && h) st_memcpy(d, h, bytes, hipMemcpyHostToDevice);
        return d;
    };
    const size_t in_b = (size_t)(Td * in_w) * 2, st_b = hst.size() * 4, cs_b = hcs.size() * 2;
    const size_t o_b = (size_t)(Td * H * V) * 2, oq_b = (size_t)(Td * H * V), os_b = (size_t)(Td * H) * 4;
    void* d_in = dev(nullptr, in_b);
    void* d_ab = dev(nullptr, (size_t)(Td * 2 * H) * 2);
    void* d_w = dev(hw.data(), hw.size() * 2);
    void* d_alog = dev(halog.data(), halog.size() * 4);
    void* d_dtb = dev(hdtb.data(), hdtb.size() * 4);
    void* d_nw = dev(hnw.data(), hnw.size() * 4);
    void* d_cidx = dev(cidx, sizeof(cidx));
    void* d_sidx = dev(hsidx.data(), hsidx.size() * 4);
    void* d_nacc = dev(nacc, sizeof(nacc));
    void* d_cu = dev(cu.data(), cu.size() * 4);
    void* d_q = dev(nullptr, (size_t)(Td * Hg * K) * 2);
    void* d_k = dev(nullptr, (size_t)(Td * Hg * K) * 2);
    void* d_v = dev(nullptr, (size_t)(Td * H * V) * 2);
    void* d_cs[2]; void* d_st[2]; void* d_o[2]; void* d_oq[2]; void* d_os[2];
    for (int i = 0; i < 2; ++i) {
        d_cs[i] = dev(hcs.data(), cs_b);  d_st[i] = dev(hst.data(), st_b);
        d_o[i] = dev(nullptr, o_b); d_oq[i] = dev(nullptr, oq_b); d_os[i] = dev(nullptr, os_b);
    }
    if (!d_in || !d_ab || !d_cs[1] || !d_os[1]) { ok(nm, false, "hipMalloc"); return; }

    auto view = [](void* base, uint32_t dt, int64_t rows, int64_t cols, int64_t pitch,
                   int64_t col0) {
        RadTensor t = T((char*)base + (size_t)col0 * esz(dt), dt, { rows, cols });
        t.stride[0] = pitch;
        return t;
    };
    const RadTensor tx = view(d_in, RAD_BF16, Td, conv_dim, in_w, 0);
    const RadTensor tz = view(d_in, RAD_BF16, Td, H * V, in_w, conv_dim);
    const RadTensor ta = view(d_ab, RAD_BF16, Td, H, 2 * H, 0);
    const RadTensor tb = view(d_ab, RAD_BF16, Td, H, 2 * H, H);
    const RadTensor tw = T(d_w, RAD_BF16, { conv_dim, W });
    const RadTensor tal = T(d_alog, RAD_F32, { H }), tdt = T(d_dtb, RAD_F32, { H });
    const RadTensor tnw = T(d_nw, RAD_F32, { V });
    const RadTensor tci = T(d_cidx, RAD_I32, { N }), tsi = T(d_sidx, RAD_I32, { N, COLS });
    const RadTensor tna = T(d_nacc, RAD_I32, { N }), tcu = T(d_cu, RAD_I32, { N + 1 });
    RadTensor none; std::memset(&none, 0, sizeof(none));
    auto cs = [&](int i) { return T(d_cs[i], RAD_BF16, { CSLOTS, conv_dim, SL }); };
    auto st = [&](int i) { return T(d_st[i], RAD_F32, { SSLOTS, H, V, K }); };
    auto to = [&](int i) { return T(d_o[i], RAD_BF16, { Td, H, V }); };
    auto toq = [&](int i) { return T(d_oq[i], i8 ? RAD_I8 : RAD_F8E4M3, { Td, H * V }); };
    auto tos = [&](int i) { return T(d_os[i], RAD_F32, { Td, H }); };
    const RadTensor tq = T(d_q, RAD_BF16, { Td, Hg, K }), tk = T(d_k, RAD_BF16, { Td, Hg, K }),
                    tv = T(d_v, RAD_BF16, { Td, H, V });

    std::vector<RadParam> pc = { PI("q_len", MQ), PI("head_k", K), PI("head_v", V),
                                 PI("conv_width", W) };
    std::vector<RadParam> pr = { PI("q_len", MQ), PI("head_k", K), PI("head_v", V),
                                 PI("n_head_v", H), PF("scale", 0.088388), PF("eps", 1e-6),
                                 PS("act", "silu"), PS("state_form", form) };
    std::vector<RadParam> pf = pr;
    pf.push_back(PI("conv_width", W));
    auto run = [&](const RadKernelInfo* r, std::vector<RadTensor> ts, std::vector<RadParam>& ps) {
        RadArgs a; std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        const int rc_ = r->launch(&a, (RadStream) nullptr);
        return rc_ != RAD_OK ? rc_ : (st_stream_sync(nullptr) != hipSuccess ? -1000 : RAD_OK);
    };
    auto fetch = [](void* d, size_t b) {
        std::vector<uint8_t> h(b);
        st_memcpy(h.data(), d, b, hipMemcpyDeviceToHost);
        return h;
    };

    std::string why;
    bool moved = false;
    for (int step = 0; step < 2 && why.empty(); ++step) {
        const std::vector<uint16_t> hin = bf((size_t)(Td * in_w), 1.0f);
        const std::vector<uint16_t> hab = bf((size_t)(Td * 2 * H), 1.0f);
        st_memcpy(d_in, hin.data(), in_b, hipMemcpyHostToDevice);
        st_memcpy(d_ab, hab.data(), hab.size() * 2, hipMemcpyHostToDevice);
        int e = run(rc, { tx, tw, none, cs(0), tci, tna, tcu, tq, tk, tv }, pc);
        if (e == RAD_OK)
            e = run(rr, { tq, tk, tv, ta, tb, tal, tdt, st(0), tsi, tna, tcu, tz, tnw, to(0),
                          toq(0), tos(0) }, pr);
        if (e != RAD_OK) { why = "pair rc=" + std::to_string(e); break; }
        e = run(rf, { tx, tw, none, cs(1), tci, ta, tb, tal, tdt, st(1), tsi, tna, tcu, tz, tnw,
                      to(1), toq(1), tos(1) }, pf);
        if (e != RAD_OK) { why = "fold rc=" + std::to_string(e); break; }

        const struct { const char* what; void* a; void* b; size_t n; } cmp[] = {
            { "o", d_o[0], d_o[1], o_b }, { "o_q", d_oq[0], d_oq[1], oq_b },
            { "o_scale", d_os[0], d_os[1], os_b }, { "state", d_st[0], d_st[1], st_b },
            { "conv_state", d_cs[0], d_cs[1], cs_b },
        };
        for (const auto& c : cmp) {
            const std::vector<uint8_t> A = fetch(c.a, c.n), B = fetch(c.b, c.n);
            size_t diff = 0;
            for (size_t i = 0; i < c.n; ++i) diff += A[i] != B[i];
            if (diff) {
                why = "step " + std::to_string(step) + ": " + c.what + " differs in " +
                      std::to_string(diff) + " of " + std::to_string(c.n) + " bytes";
                break;
            }
        }
        /* At int8 codes, the real sequences' tokens (all but the padded last) against
         * quant_act_i8g over the pair's `o`. A (token, head) row is one group. */
        if (i8 && why.empty()) {
            const std::vector<uint8_t> O = fetch(d_o[0], o_b), Q = fetch(d_oq[0], oq_b),
                                       S = fetch(d_os[0], os_b);
            const int64_t rows = (int64_t)cu[(size_t)N - 1] * H;
            const int64_t miss = i8_code_misses((const uint16_t*)O.data(), V,
                                                (const int8_t*)Q.data(), V,
                                                (const float*)S.data(), 1, rows, V);
            if (miss) why = "step " + std::to_string(step) + ": " + std::to_string(miss) +
                            " int8 codes or scales differ from quant_act_i8g over o";
        }
        /* The key heads' channels of a real sequence's conv slot, against where they started. */
        const std::vector<uint8_t> C = fetch(d_cs[1], cs_b);
        const size_t kq = (size_t)(3 * conv_dim * SL) * 2;           /* sequence 0's slot */
        if (std::memcmp(C.data() + kq, hcs.data() + kq / 2, (size_t)(2 * Hg * K * SL) * 2) != 0)
            moved = true;
    }
    if (why.empty() && !moved) why = "the key heads' conv state never moved";
    char d[160];
    std::snprintf(d, sizeof d, "%s", why.c_str());
    ok(nm, why.empty(), why.empty() ? "" : d);
}

static void gdn_conv_fold(const Plugin& pl) {
    const char* row = "gdn_conv_recurrent_update_k128_v128_bf16";
    const char* nmp = "gdn_conv_recurrent_update is byte-identical to the pair, per_candidate";
    const char* nma = "gdn_conv_recurrent_update is byte-identical to the pair, anchor";
    if (sel(row)) {
        gdn_conv_fold_run(pl, nmp, "per_candidate");
        gdn_conv_fold_run(pl, "gdn_conv_recurrent_update is byte-identical to the pair at int8 "
                              "codes, which are quant_act_i8g's, per_candidate",
                          "per_candidate", true);
    }
    if (sel(row) || sel(nma)) gdn_conv_fold_run(pl, nma, "anchor");
}

/* The gated norm's gate as a column slice, which one pitch cannot describe.
 *
 * The fp8 delta net projects [q|k|v|z] in one GEMM (arch/common/rad_block_gdn_fp8.h), so the gate
 * is the right-hand columns of a wider row: its (token, head) rows are `channels` apart within a
 * token and the projection's width apart between tokens. A single stride walks into the next token
 * after `n_head_v` rows, silently. This runs the same numbers dense and as the tail of a row twice
 * as wide and requires identical bytes; values are multiples of an eighth, so equality is exact. */
static void gdn_gated_norm_sliced_gate(const Plugin& pl) {
    if (!sel("gdn_gated_rmsnorm_h128_bf16")) return;
    const char* nm = "gdn_gated_rmsnorm_h128_bf16 reads a column-sliced gate";
    const RadKernelInfo* r = find(pl, "gdn_gated_rmsnorm_h128_bf16");
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t nt = 6, H = 5, ch = 128, rows = nt * H, wide = 2 * H * ch;
    std::vector<uint16_t> hx((size_t)(rows * ch)), hz((size_t)(rows * ch)),
                          hw2((size_t)(nt * wide), 0), ha((size_t)(rows * ch), 0),
                          hb((size_t)(rows * ch), 0);
    std::vector<float>    hw((size_t)ch);
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t c = 0; c < ch; ++c) {
                const int64_t i = (t * H + h) * ch + c;
                hx[(size_t)i] = h_to_bf16((float)(((t * 7 + h * 5 + c) % 17) - 8) * 0.125f);
                hz[(size_t)i] = h_to_bf16((float)(((t * 3 + h * 11 + c) % 13) - 6) * 0.25f);
                /* The same gate values, planted in the tail of a row twice as wide. */
                hw2[(size_t)(t * wide + H * ch + h * ch + c)] = hz[(size_t)i];
            }
    for (int64_t c = 0; c < ch; ++c) hw[(size_t)c] = (float)((c % 5) - 2) * 0.5f;

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { rows, ch }), T(nullptr, RAD_BF16, { rows, ch }),
        T(nullptr, RAD_F32,  { ch }),       T(nullptr, RAD_BF16, { rows, ch }),
    };
    RadTensor twide = T(nullptr, RAD_BF16, { nt, wide });
    for (auto& t : ts) t.data = alloc_for(t);
    twide.data = alloc_for(twide);
    if (!ts[0].data || !ts[1].data || !ts[2].data || !ts[3].data || !twide.data) {
        ok(nm, false, "hipMalloc failed");
        return;
    }
    st_memcpy(ts[0].data, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hz.data(), hz.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[2].data, hw.data(), hw.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(twide.data, hw2.data(), hw2.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = {
        PI("M", rows), PI("channels", ch), PF("eps", 1e-6), PS("act", "silu"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;

    int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "dense arm rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(ha.data(), ts[3].data, ha.size() * 2, hipMemcpyDeviceToHost);

    /* The gate operand, respelled: [tokens, heads*channels] starting H*ch into a `wide` row. */
    ts[1].data      = (void*)((uint16_t*)twide.data + H * ch);
    ts[1].rank      = 2;
    ts[1].shape[0]  = nt;
    ts[1].shape[1]  = H * ch;
    ts[1].stride[0] = wide;
    ts[1].stride[1] = 1;
    st_memset(ts[3].data, 0, ha.size() * 2);
    rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "sliced arm rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hb.data(), ts[3].data, hb.size() * 2, hipMemcpyDeviceToHost);

    int64_t bad = 0;
    for (size_t i = 0; i < ha.size(); ++i) if (ha[i] != hb[i]) ++bad;
    char d[128];
    std::snprintf(d, sizeof d, "%lld of %lld elements differ", (long long)bad,
                  (long long)ha.size());
    ok(nm, bad == 0, d);
}

/* The grouped dynamic convolution against a host mirror of DFlash2's `_grouped_conv`:
 *
 *   out[t,h] = SUM over k of (base[k][h] + delta[t][k*NG + h/group]) * x[t-k][h] * (t%block >= k)
 *
 * The smoke case runs on zeroed buffers and cannot tell a convolution from a copy. This checks the
 * group index (which delta belongs to which channel), the block-position mask, and whether base is
 * indexed by channel or by group. */
static void dflash_conv_numeric(const Plugin& pl, int64_t nt, int64_t H, int64_t group,
                                int64_t taps, int64_t bsz) {
    if (!sel("dflash_conv_t2_g16_bf16")) return;
    const int64_t NG = H / group;
    char nm[128];
    std::snprintf(nm, sizeof nm, "dflash_conv_t2_g16_bf16 numeric T=%lld H=%lld block=%lld",
                  (long long)nt, (long long)H, (long long)bsz);
    const RadKernelInfo* r = find(pl, "dflash_conv_t2_g16_bf16");
    if (!r) { ok(nm, false, "row absent"); return; }

    const int64_t dpitch = 2 * taps * NG;
    std::vector<uint16_t> hx((size_t)(nt * H)), hd((size_t)(nt * dpitch)),
                          hb((size_t)(taps * H)), hy((size_t)(nt * H), 0);
    /* Quarters and eighths, so every product and sum is exact in f32. */
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h)
            hx[(size_t)(t * H + h)] = h_to_bf16((float)(((t * 5 + h * 3) % 17) - 8) * 0.25f);
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t j = 0; j < dpitch; ++j)
            hd[(size_t)(t * dpitch + j)] = h_to_bf16((float)(((t * 3 + j * 7) % 9) - 4) * 0.125f);
    for (int64_t k = 0; k < taps; ++k)
        for (int64_t h = 0; h < H; ++h)
            hb[(size_t)(k * H + h)] = h_to_bf16((float)(((k * 11 + h) % 13) - 6) * 0.25f);

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { nt, H }), T(nullptr, RAD_BF16, { nt, dpitch }),
        T(nullptr, RAD_BF16, { taps, H }), T(nullptr, RAD_BF16, { nt, H }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data || !ts[3].data) {
        ok(nm, false, "hipMalloc failed");
        return;
    }
    st_memcpy(ts[0].data, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hd.data(), hd.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[2].data, hb.data(), hb.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = {
        PI("T", nt), PI("hidden_size", H), PI("taps", taps), PI("group_size", group),
        PI("NG", NG), PI("block_size", bsz), PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), ts[3].data, hy.size() * 2, hipMemcpyDeviceToHost);

    int64_t bad = 0;
    double worst = 0;
    std::string first;
    for (int64_t t = 0; t < nt; ++t)
        for (int64_t h = 0; h < H; ++h) {
            float want = 0;
            for (int64_t k = 0; k < taps; ++k) {
                if ((t & (bsz - 1)) < k || t - k < 0) continue;
                const float coef = h_bf16(hb[(size_t)(k * H + h)]) +
                                   h_bf16(hd[(size_t)(t * dpitch + k * NG + h / group)]);
                want += coef * h_bf16(hx[(size_t)((t - k) * H + h)]);
            }
            const float got = h_bf16(hy[(size_t)(t * H + h)]);
            const double e = std::fabs((double)got - (double)h_bf16(h_to_bf16(want)));
            if (e > 0) {
                if (!bad) first = " first at t=" + std::to_string(t) + " h=" + std::to_string(h) +
                                  " want " + std::to_string(want) + " got " + std::to_string(got);
                ++bad;
            }
            worst = std::max(worst, e);
        }
    char d[192];
    std::snprintf(d, sizeof d, "%lld of %lld differ, worst %.4g%s", (long long)bad,
                  (long long)(nt * H), worst, first.c_str());
    ok(nm, bad == 0, d);
}

/* The skinny bf16 GEMM against a host reference at a given K. K is the dimension this kernel
 * decomposes along (pick_sk splits it, rowGo strides rows by it, the unrolled fragment loop walks
 * it), so it is a parameter. The weight is plain row-major [N, K]: the row has no layout hook. */
static void bf16_gemm_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K) {
    if (!sel(row)) return;
    char nm[128];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld", row,
                  (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    if (!r) { ok(nm, false, "row absent"); return; }

    /* Deterministic, in the drafter fc's range: activations to a couple of hundred, weights
     * within one. The sign alternates on a period coprime with 16, since a pattern symmetric in k
     * would make every split-K partial sum equal. */
    std::vector<uint16_t> ha((size_t)(M * K)), hw((size_t)(N * K)), hy((size_t)(M * N), 0);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k)
            ha[(size_t)(i * K + k)] =
                h_to_bf16((float)(((i * 7 + k * 13) % 401) - 200) * 0.5f * ((k % 3) ? 1.f : -1.f));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            hw[(size_t)(n * K + k)] =
                h_to_bf16((float)(((n * 11 + k * 5) % 199) - 99) * 0.01f * ((k % 5) ? 1.f : -1.f));

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
        T(nullptr, RAD_BF16, { M, N }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(ts[0].data, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), ts[2].data, hy.size() * 2, hipMemcpyDeviceToHost);

    /* The reference reads the same bf16 values, so only arithmetic and addressing can differ. A
     * wide N is checked on a strided sample of columns (the host cost is M*N*K); which weight row a
     * column reads depends only on its index, so sampling finds a misread. */
    const int64_t nstep = N > 64 ? N / 64 : 1;
    double num = 0, den = 0, worst = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += nstep) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)h_bf16(ha[(size_t)(i * K + k)]) * (double)h_bf16(hw[(size_t)(n * K + k)]);
            const double got = (double)h_bf16(hy[(size_t)(i * N + n)]);
            num += (got - acc) * (got - acc);
            den += acc * acc;
            worst = std::max(worst, std::fabs(got - acc) / (std::fabs(acc) + 1e-6));
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[96];
    std::snprintf(d, sizeof d, "rel_l2 %.3e, worst %.3e", rel, worst);
    ok(nm, rel < 2e-2, d);
}


/* The 8-bit-weight skinny GEMM against the planes' meaning. The row reads fragment-order codes
 * and (scale, zero) dwords from its own w8a16 hooks; the reference dequantises the planes on the
 * host. `adt` is the activation dtype: the row takes f16 and refuses others. */
static void w8a16_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K,
                          uint32_t adt) {
    if (!sel(row)) return;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric %s M=%lld N=%lld K=%lld", row,
                  adt == RAD_BF16 ? "bf16" : "f16", (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || !r->relayout) { ok(nm, false, "row or hooks absent"); return; }

    std::vector<float> wf((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            wf[(size_t)(n * K + k)] = std::ldexp(1.0f, (int)((n / 16) % 5) - 2 - (int)((k / 128) % 3))
                                      * (0.031f * (float)((n * 37 + k * 11) % 61) - 0.9f);
    const Planes p = int8_planes(wf, N, K);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wsz;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &wsz);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }

    std::vector<uint16_t> ha((size_t)(M * K)), hy((size_t)(M * N), 0);
    std::vector<float> av((size_t)(M * K));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            const float v = (float)(((i * 7 + k * 13) % 401) - 200) * 0.05f * ((k % 3) ? 1.f : -1.f);
            if (adt == RAD_BF16) {
                ha[(size_t)(i * K + k)] = h_to_bf16(v);
                av[(size_t)(i * K + k)] = h_bf16(ha[(size_t)(i * K + k)]);
            } else {
                uint16_t h;
                rad_store_f32(&h, RAD_F16, 0, v);
                ha[(size_t)(i * K + k)] = h;
                av[(size_t)(i * K + k)] = rad_load_f32(&h, RAD_F16, 0);
            }
        }

    void* d_a = dalloc(ha.size() * 2);
    void* d_w = dalloc(wq.size());
    void* d_s = dalloc(wsz.size());
    void* d_y = dalloc(hy.size() * 2);
    if (!d_a || !d_w || !d_s || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, wsz.data(), wsz.size(), hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_a, adt, { M, K }), T(nullptr, RAD_F32, { M }),
        T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_s, RAD_U8, { (int64_t)wsz.size() }),
        T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[1]);
    absent(ts[5]);
    absent(ts[6]);
    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                 PS("dtype", "w8a16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);

    const int64_t ng = K / 128;
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += (n + nstep < N ? nstep : 1)) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                const float sc = rad_load_f32(p.data[1].data(), RAD_F16, n * ng + k / 128);
                acc += (double)av[(size_t)(i * K + k)] *
                       (double)(int8_t)p.data[0][(size_t)(n * K + k)] * (double)sc;
            }
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[64];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    ok(nm, rel < 1e-2, d);
}


/* An OCP-MXFP4 weight's planes, drawn from st_mix32 of each coordinate and `seed`: codes over all
 * sixteen e2m1 values, and an exponent per (row, 32 of K) within three binades of 127. With
 * `deep`, about one block in eleven sits 6..12 binades lower (the folded table's subnormal
 * tail). */
static void mxfp4_fill(Planes& p, int ic, int is, int64_t N, int64_t K, uint32_t seed, bool deep) {
    const int64_t nblk = K / 32;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; k += 2) {
            const uint32_t h = st_mix32((uint32_t)(n * K + k) ^ (seed * 0x9e3779b9u));
            p.data[(size_t)ic][(size_t)((n * K + k) / 2)] =
                (uint8_t)((h & 15) | (((h >> 8) & 15) << 4));
        }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t b = 0; b < nblk; ++b) {
            const uint32_t h = st_mix32((uint32_t)(n * nblk + b) ^ (seed * 0x85ebca6bu) ^ 0x2545u);
            int e = 127 + (int)(h % 7) - 3;
            if (deep && (h >> 8) % 11 == 0) e -= 6 + (int)((h >> 16) % 7);
            p.data[(size_t)is][(size_t)(n * nblk + b)] = (uint8_t)e;
        }
}

/* An E4M3-representable activation value in [-4, 4], drawn the same way. */
static float mxfp4_act(int64_t i, int64_t k, uint32_t seed) {
    const uint32_t h = st_mix32((uint32_t)(i * 0x10001 + k) ^ (seed * 0xc2b2ae35u));
    return (float)((int)(h % 401) - 200) * 0.02f;
}

/* The MXFP4 GEMMs against the planes' meaning. The weight is an OCP-MXFP4 plane pair (e2m1 codes,
 * an E8M0 exponent per 32 K) stored through the row's own hooks, so fragment order, [K/32][N]
 * exponents and the per-row reference all come from r4d_layout.cpp. The host reference decodes
 * the planes with no fold.
 *
 * Exponents are spread so a row's blocks sit up to 12 binades under its reference: d <= 8 is exact
 * in the folded e4m3 table, d = 9..12 rounds (the subnormal tail). M and N are ragged against the
 * tiled row's 256-row block and 64/128-column tile, so clamped staging and the guarded epilogue
 * run. `tn` pins the tiled row's column width; 0 leaves it to the row's M rule (the skinny row
 * ignores it).
 *
 * `ablk` passes the activation scale on the block-scaled fp8 grid, one f32 per (row, 128 of K), as
 * fused quantisers write it. Scales span two decades, and every eleventh block's is zero with zero
 * codes (an all-zero activation block), where the kernels carry their previous scale through. */
static void mxfp4a8_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K,
                            int tn, bool ablk = false, std::vector<RadParam> extra = {},
                            int64_t pb1 = 0, int64_t pb2 = 0) {
    if (!sel(row)) return;
    /* Parts (ParoQuant): weight rows [0, pb1), [pb1, pb2), [pb2, N) each meet their own copy of
     * the activation, side by side in a row. `K` is one copy's width. */
    const PartCut cut(N, pb1, pb2);
    const int parts = cut.parts;
    char nm[160];
    std::string ex;
    for (const RadParam& p : extra) ex += std::string(" ") + p.key + "=" + std::to_string(p.ival);
    if (parts > 1) ex += " parts=" + std::to_string(parts);
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld tn=%d%s%s", row, (long long)M,
                  (long long)N, (long long)K, tn, ablk ? " ascale=row x 128" : "", ex.c_str());
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || !r->relayout) { ok(nm, false, "row or hooks absent"); return; }
    const int64_t nblk = K / 32;

    Planes p;
    p.enc = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    p.size(N, K);
    const int ic = p.find("codes"), is = p.find("scale");
    if (ic < 0 || is < 0) { ok(nm, false, "encoding has no codes/scale plane"); return; }
    mxfp4_fill(p, ic, is, N, K, 0x51u, true);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 32) };
    std::vector<uint8_t> wq, ws, wr;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &ws);
    const int rr = stored(r, lp, 6, p, { "scale" }, &wr);
    if (rw != RAD_OK || rs != RAD_OK || rr != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs) + "/" +
                      std::to_string(rr));
        return;
    }

    const int64_t nkb = (K + 127) / 128, asc = ablk ? nkb : 1;
    const int64_t PK = parts * K, PA = parts * asc;           /* a row of `a`, and of its scale */
    std::vector<uint8_t> ha((size_t)(M * PK));
    std::vector<float> av((size_t)(M * PK)), as((size_t)(M * PA));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t c = 0; c < PA; ++c) {
            const uint32_t h = st_mix32((uint32_t)(i * PA + c) ^ 0xa5u);
            as[(size_t)(i * PA + c)] =
                ablk ? ((i + c) % 11 == 5 ? 0.0f : 0.002f * (float)(1 + h % 97))
                     : 0.0125f * (float)(1 + h % 5);
        }
        for (int64_t k = 0; k < PK; ++k) {
            const bool zero = ablk && as[(size_t)(i * PA + (k / K) * asc + (k % K) / 128)] == 0.0f;
            const float v = zero ? 0.0f : mxfp4_act(i, k, 0x3cu);
            rad_store_f32(ha.data(), RAD_F8E4M3, i * PK + k, v);
            av[(size_t)(i * PK + k)] = rad_load_f32(ha.data(), RAD_F8E4M3, i * PK + k);
        }
    }
    std::vector<uint16_t> hy((size_t)(M * N), 0);
    void* d_a = dalloc(ha.size());
    void* d_s = dalloc(as.size() * 4);
    void* d_w = dalloc(wq.size());
    void* d_e = dalloc(ws.size());
    void* d_r = dalloc(wr.size());
    void* d_y = dalloc(hy.size() * 2);
    if (!d_a || !d_s || !d_w || !d_e || !d_r || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_e, ws.data(), ws.size(), hipMemcpyHostToDevice);
    st_memcpy(d_r, wr.data(), wr.size(), hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_a, RAD_F8E4M3, { M, PK }),
        ablk ? T(d_s, RAD_F32, { M, PA }) : T(d_s, RAD_F32, { M }),
        T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_e, RAD_U8, { (int64_t)ws.size() }),
        T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(d_r, RAD_U8, { (int64_t)wr.size() }),
    };
    absent(ts[5]);
    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 32),
                                 PS("dtype", "mxfp4a8") };
    if (tn) ps.push_back(PI("tn", tn));
    if (pb1) ps.push_back(PI("pb1", pb1));
    if (pb2) ps.push_back(PI("pb2", pb2));
    for (const RadParam& p : extra) ps.push_back(p);
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    /* A row with a scratch hook (the decode row's split partial) gets what it asks for, as from
     * the core; the decode row refuses a split with too little. */
    if (r->scratch) {
        const int64_t sb = r->scratch(&a);
        if (sb > 0) {
            a.scratch = dalloc((size_t)sb);
            a.scratch_bytes = sb;
            if (!a.scratch) { ok(nm, false, "hipMalloc failed"); return; }
        }
    }
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        ok(nm, false, "rc=" + std::to_string(rc));
        return;
    }
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);

    /* Every column for small N, a stride otherwise, always including the last column (the clamped
     * tail of the tiled row's weight staging). */
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0, worst = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n = (n + 1 == N) ? N : std::min(n + nstep, N - 1)) {
            double acc = 0;
            const int pt = cut.part_of(n);
            for (int64_t k = 0; k < K; ++k) {
                const double w = (double)rad_load_f32(p.data[(size_t)ic].data(), RAD_FP4E2M1, n * K + k) *
                                 (double)rad_load_f32(p.data[(size_t)is].data(), RAD_E8M0, n * nblk + k / 32);
                acc += (double)av[(size_t)(i * PK + pt * K + k)] * w *
                       (ablk ? (double)as[(size_t)(i * PA + pt * asc + k / 128)] : 1.0);
            }
            if (!ablk) acc *= (double)as[(size_t)i];
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
            worst = std::max(worst, std::fabs(d) / (std::fabs(acc) + 1e-3));
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[96];
    std::snprintf(d, sizeof d, "rel_l2 %.3e, worst %.3e", rel, worst);
    ok(nm, rel < 1e-2, d);
}


/* What givens_numeric runs: the sequential rotation to f16, or the dense rotation with either
 * quantiser. */
enum class GvCase { Rotate, DenseFp8, DenseI8 };

/* ParoQuant's activation quantiser against a double-precision rotation. The tables are real
 * pairings (each stage of each group a random permutation of its 128 columns read as 64 pairs),
 * angles over (-pi, pi), channel scales over (0.5, 2). The reference divides, rotates in f64 and
 * takes the absmax. The kernel's scale must match to the rounding of its f32 sincos, and each
 * dequantised element must lie within the reference's E4M3 grid (half a step at the top of the
 * range is 1/16 of the group maximum, so the 7% bound rejects a wrong pair or angle). */
static void givens_numeric(const Plugin& pl, int64_t M, int64_t K, int parts, int stages, GvCase mode) {
    const bool rot_only = mode == GvCase::Rotate, i8 = mode == GvCase::DenseI8, dense = !rot_only;
    const char* row = dense ? "givens_dense_quant" : "givens_rotate";
    if (!sel(row)) return;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s %s numeric M=%lld K=%lld parts=%d stages=%d", row,
                  dense ? (i8 ? "i8" : "fp8") : "", (long long)M, (long long)K, parts, stages);
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->launch) { ok(nm, false, "row absent"); return; }
    const int64_t G = 128, ng = K / G;
    std::vector<uint16_t> hx((size_t)(M * K));
    std::vector<float> xv((size_t)(M * K));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t h = st_mix32((uint32_t)(i * 0x10001 + k) ^ 0x77u);
            const float v = (float)((int)(h % 4001) - 2000) * 0.002f * (i % 4 == 3 ? 40.0f : 1.0f);
            hx[(size_t)(i * K + k)] = h_to_bf16(v);
            xv[(size_t)(i * K + k)] = h_bf16(hx[(size_t)(i * K + k)]);
        }
    const int64_t NPS = (int64_t)parts * stages;
    std::vector<int32_t> pairs((size_t)(NPS * K));
    std::vector<float> ang((size_t)(NPS * K / 2)), rs((size_t)(parts * K));
    for (int64_t ps = 0; ps < NPS; ++ps) {
        for (int64_t g = 0; g < ng; ++g) {
            std::vector<int32_t> perm((size_t)G);
            for (int i = 0; i < G; ++i) perm[(size_t)i] = i;
            for (int i = (int)G - 1; i > 0; --i)
                std::swap(perm[(size_t)i],
                          perm[st_mix32((uint32_t)(ps * 7919 + g * 104729 + i)) % (uint32_t)(i + 1)]);
            for (int i = 0; i < G; ++i) pairs[(size_t)(ps * K + g * G + i)] = (int32_t)(g * G + perm[(size_t)i]);
        }
        for (int64_t j = 0; j < K / 2; ++j)
            ang[(size_t)(ps * K / 2 + j)] =
                (float)(st_mix32((uint32_t)(ps * 31337 + j) ^ 0x5bu) % 6283) / 1000.0f - 3.1415f;
    }
    for (int64_t i = 0; i < parts * K; ++i)
        rs[(size_t)i] = 0.5f + (float)(st_mix32((uint32_t)i ^ 0xc3u) % 1500) / 1000.0f;

    void* d_x = dalloc(hx.size() * 2);
    void* d_p = dalloc(pairs.size() * 4);
    void* d_a = dalloc(ang.size() * 4);
    void* d_r = dalloc(rs.size() * 4);
    void* d_q = dalloc((size_t)(M * parts * K) * (rot_only ? 2 : 1));
    void* d_s = dalloc((size_t)(M * parts * ng) * 4);
    void* d_t = nullptr;
    if (!d_x || !d_p || !d_a || !d_r || !d_q || !d_s) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_p, pairs.data(), pairs.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_a, ang.data(), ang.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_r, rs.data(), rs.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_x, RAD_BF16, { M, K }), T(d_p, RAD_I32, { NPS, K }), T(d_a, RAD_F32, { NPS, K / 2 }),
        T(d_r, RAD_F32, { (int64_t)parts, K }),
        T(d_q, rot_only ? (uint32_t)RAD_F16 : i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3, { M, parts * K }),
        T(d_s, RAD_F32, { M, parts * ng }),
    };
    if (rot_only) ts.pop_back();
    std::vector<RadParam> ps = { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                                 PI("stages", stages), PI("parts", parts) };
    if (dense) {
        // The matrices as a container stores them (the "t.dense" table, abi/rad_encoding.h): one
        // 128x128 f16 matrix per (part, group), row p * ng + g, element [j][i] the image of column
        // i under the stages, composed in f64 as rad-convert does; through the row's layout hook.
        std::vector<uint16_t> dm((size_t)(parts * ng * G * G));
        std::vector<double> v((size_t)G);
        for (int p = 0; p < parts; ++p)
            for (int64_t g = 0; g < ng; ++g)
                for (int64_t i = 0; i < G; ++i) {
                    std::fill(v.begin(), v.end(), 0.0);
                    v[(size_t)i] = 1.0;
                    for (int st = 0; st < stages; ++st)
                        for (int64_t j = g * G / 2; j < (g + 1) * G / 2; ++j) {   // group g's pairs
                            const int64_t ia = pairs[(size_t)((p * stages + st) * K + 2 * j)] - g * G;
                            const int64_t ib = pairs[(size_t)((p * stages + st) * K + 2 * j + 1)] - g * G;
                            const double t = ang[(size_t)((p * stages + st) * (K / 2) + j)];
                            const double c = std::cos(t), sn = std::sin(t);
                            const double va = v[(size_t)ia], vb = v[(size_t)ib];
                            v[(size_t)ia] = va * c + vb * sn;
                            v[(size_t)ib] = vb * c - va * sn;
                        }
                    for (int64_t j = 0; j < G; ++j)
                        dm[(size_t)(((p * ng + g) * G + j) * G + i)] = rad_f32_to_f16((float)v[(size_t)j]);
                }
        Planes pp;
        pp.enc = rad_enc_plain(RAD_BF16);
        rad_enc_add_table(&pp.enc, "t.dense", RAD_F16, parts * ng, G * G);
        pp.data.assign(2, {});
        pp.rows = { 0, parts * ng };
        pp.cols = { 0, G * G };
        pp.data[1].assign((const uint8_t*)dm.data(), (const uint8_t*)(dm.data() + dm.size()));
        std::vector<RadParam> lp = { PI("n", K), PI("parts", parts) };
        std::vector<uint8_t> blob;
        const int rl = stored(r, lp, 1, pp, { "t.dense" }, &blob);
        if (rl != RAD_OK) { ok(nm, false, "relayout rc " + std::to_string(rl)); return; }
        d_t = dalloc(blob.size());
        if (!d_t) { ok(nm, false, "hipMalloc failed"); return; }
        st_memcpy(d_t, blob.data(), blob.size(), hipMemcpyHostToDevice);
        ts = { T(d_x, RAD_BF16, { M, K }), T(d_t, RAD_F16, { parts * K, 128 }), T(d_r, RAD_F32, { (int64_t)parts, K }),
               T(d_q, i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3, { M, parts * K }), T(d_s, RAD_F32, { M, parts * ng }) };
        ps = { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"), PI("parts", parts),
               PS("out", i8 ? "i8" : "fp8") };
    }
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) { ok(nm, false, "rc=" + std::to_string(rc)); return; }
    std::vector<uint8_t> hq((size_t)(M * parts * K) * (rot_only ? 2 : 1));
    std::vector<float> hs((size_t)(M * parts * ng));
    st_memcpy(hq.data(), d_q, hq.size(), hipMemcpyDeviceToHost);
    st_memcpy(hs.data(), d_s, hs.size() * 4, hipMemcpyDeviceToHost);

    double worst_scale = 0, worst_val = 0;
    std::vector<double> b((size_t)K);
    for (int64_t i = 0; i < M; ++i)
        for (int p = 0; p < parts; ++p) {
            for (int64_t k = 0; k < K; ++k) b[(size_t)k] = (double)xv[(size_t)(i * K + k)] / (double)rs[(size_t)(p * K + k)];
            for (int st = 0; st < stages; ++st)
                for (int64_t j = 0; j < K / 2; ++j) {
                    const int64_t ia = pairs[(size_t)((p * stages + st) * K + 2 * j)];
                    const int64_t ib = pairs[(size_t)((p * stages + st) * K + 2 * j + 1)];
                    const double t = ang[(size_t)((p * stages + st) * (K / 2) + j)], c = std::cos(t), sn = std::sin(t);
                    const double va = b[(size_t)ia], vb = b[(size_t)ib];
                    b[(size_t)ia] = va * c + vb * sn;
                    b[(size_t)ib] = vb * c - va * sn;
                }
            for (int64_t g = 0; g < ng; ++g) {
                double amax = 0;
                for (int k = 0; k < G; ++k) amax = std::max(amax, std::fabs(b[(size_t)(g * G + k)]));
                if (rot_only) {                 // f16: relative 2^-11 of the value, 1e-3 of the group max
                    for (int k = 0; k < G; ++k) {
                        const double y = rad_load_f32(hq.data(), RAD_F16, i * parts * K + p * K + g * G + k);
                        worst_val = std::max(worst_val, std::fabs(y - b[(size_t)(g * G + k)]) / (amax + 1e-30));
                    }
                    continue;
                }
                const double sc = hs[(size_t)(i * parts * ng + p * ng + g)];
                const double qmax = i8 ? 127.0 : 448.0;
                worst_scale = std::max(worst_scale, std::fabs(sc - amax / qmax) / (amax / qmax + 1e-30));
                for (int k = 0; k < G; ++k) {
                    const double deq = (double)rad_load_f32(hq.data(), i8 ? RAD_I8 : RAD_F8E4M3, i * parts * K + p * K + g * G + k) * sc;
                    worst_val = std::max(worst_val, std::fabs(deq - b[(size_t)(g * G + k)]) / (amax + 1e-30));
                }
            }
        }
    char d[128];
    std::snprintf(d, sizeof d, "scale rel %.2e, value/amax %.3f", worst_scale, worst_val);
    if (M >= 1024) {                                  // a prefill chunk: say how fast it ran
        hipEvent_t e0, e1;
        st_ev_create(&e0); st_ev_create(&e1);
        for (int i = 0; i < 3; ++i) r->launch(&a, nullptr);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < 20; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0); st_ev_destroy(e1);
        std::printf("        %.1f us a launch   (%s)\n", (double)ms * 50.0, d);
    }
    // The dense form carries f16 matrices: its scale is ~1e-3 off the f32 network's, and its
    // values stay within the quantiser's own step.
    ok(nm, rot_only ? worst_val < 1e-3
                    : (worst_scale < (dense ? 3e-3 : 1e-4) && worst_val < (i8 ? 0.01 : 0.07)), d);
}


/* ParoQuant's 5-bit skinny GEMM against the planes' meaning: u5 codes (a bit stream along K), an
 * f16 scale and a zero point 0..31 per 128 of K, an f16 activation, the weight stored through the
 * row's hooks. `pb1`/`pb2` give a partitioned weight. M past 64 runs the dispatcher's 64-row
 * chunks. */
static void w5a16_numeric(const Plugin& pl, int64_t M, int64_t N, int64_t K, int64_t pb1 = 0,
                          int64_t pb2 = 0) {
    const char* row = "gemm_w5a16_nt_m64";
    if (!sel(row)) return;
    const PartCut cut(N, pb1, pb2);
    const int parts = cut.parts;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld parts=%d", row, (long long)M,
                  (long long)N, (long long)K, parts);
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || !r->relayout) { ok(nm, false, "row or hooks absent"); return; }
    const int64_t G = K / 128;
    const Planes p = asym_planes(5, N, K);
    const int ic = p.find("codes"), is = p.find("scale"), iz = p.find("zero");
    const int64_t rb = p.row_bytes(ic);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, ws;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale", "zero" }, &ws);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    const int64_t PK = parts * K;
    std::vector<uint16_t> ha((size_t)(M * PK));
    std::vector<float> av((size_t)(M * PK));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < PK; ++k) {
            const uint32_t h = st_mix32((uint32_t)(i * 0x10001 + k) ^ 0x3cu);
            const float v = (float)((int)(h % 401) - 200) * 0.01f;
            rad_store_f32(&ha[(size_t)(i * PK + k)], RAD_F16, 0, v);
            av[(size_t)(i * PK + k)] = rad_load_f32(&ha[(size_t)(i * PK + k)], RAD_F16, 0);
        }
    std::vector<uint16_t> hy((size_t)(M * N), 0);
    void* d_a = dalloc(ha.size() * 2);
    void* d_w = dalloc(wq.size());
    void* d_s = dalloc(ws.size());
    void* d_y = dalloc(hy.size() * 2);
    if (!d_a || !d_w || !d_s || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, ws.data(), ws.size(), hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_a, RAD_F16, { M, PK }), T(nullptr, RAD_F32, { 1 }),
        T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_s, RAD_U8, { (int64_t)ws.size() }),
        T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_F32, { 1 }),
    };
    absent(ts[1]); absent(ts[5]); absent(ts[6]);
    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                 PS("dtype", "w5a16") };
    if (pb1) ps.push_back(PI("pb1", pb1));
    if (pb2) ps.push_back(PI("pb2", pb2));
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) { ok(nm, false, "rc=" + std::to_string(rc)); return; }
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n = (n + 1 == N) ? N : std::min(n + nstep, N - 1)) {
            const int pt = cut.part_of(n);
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double q = rad_load_f32(p.data[(size_t)ic].data() + n * rb, RAD_U5, k);
                const double z = p.data[(size_t)iz][(size_t)(n * G + k / 128)];
                const double sc = rad_load_f32(p.data[(size_t)is].data(), RAD_F16, n * G + k / 128);
                acc += (double)av[(size_t)(i * PK + pt * K + k)] * (q - z) * sc;
            }
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[64];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    ok(nm, rel < 1e-2, d);
}


/* Timing of the 5/4-bit GEMM at one prefill chunk over the row's tuned axes; `--case w5a16_perf`
 * only (a measurement, not a check). The weight is replicated past the last-level cache and walked
 * in turn, as in perf_row, so the printed GB/s is the weight stream's. */
static void w5a16_perf(const Plugin& pl, int bits, int64_t M, int64_t N, int64_t K) {
    if (!g_case || std::strcmp(g_case, "w5a16_perf")) return;
    const char* row = bits == 5 ? "gemm_w5a16_nt_m64" : "gemm_w4a16_asym_nt_m64";
    const RadKernelInfo* r = find(pl, row);
    if (!r) { std::printf("row absent\n"); return; }
    const int64_t G = K / 128;
    Planes p;
    p.enc = rad_enc_affine(bits == 5 ? RAD_U5 : RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&p.enc, "zero", RAD_U8, 1, 128);
    p.size(N, K);
    for (auto& d : p.data[0]) d = (uint8_t)st_mix32((uint32_t)(&d - p.data[0].data()));
    for (int64_t i = 0; i < N * G; ++i) {
        rad_store_f32(p.data[1].data(), RAD_F16, i, 0.01f);
        p.data[2][(size_t)i] = 8;
    }
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128),
                                 PS("dtype", bits == 5 ? "w5a16" : "w4a16_asym") };
    std::vector<uint8_t> wq, ws;
    if (stored(r, lp, 2, p, { "codes" }, &wq) != RAD_OK || stored(r, lp, 3, p, { "scale", "zero" }, &ws) != RAD_OK) {
        std::printf("relayout failed\n");
        return;
    }
    const size_t copies = std::max<size_t>(2, (size_t)(256u << 20) / wq.size() + 1);
    void* d_w = dalloc(wq.size() * copies);
    void* d_s = dalloc(ws.size() * copies);
    void* d_a = dalloc((size_t)(M * K * 2));
    void* d_y = dalloc((size_t)(M * N * 2));
    if (!d_w || !d_s || !d_a || !d_y) { std::printf("hipMalloc failed\n"); return; }
    for (size_t c = 0; c < copies; ++c) {
        st_memcpy((uint8_t*)d_w + c * wq.size(), wq.data(), wq.size(), hipMemcpyHostToDevice);
        st_memcpy((uint8_t*)d_s + c * ws.size(), ws.data(), ws.size(), hipMemcpyHostToDevice);
    }
    std::printf("%s M=%lld N=%lld K=%lld  (weight %.1f MiB)\n", row, (long long)M, (long long)N,
                (long long)K, wq.size() / 1048576.0);
    struct Cfg { int wv, sk, mb, npw, nt; };
    std::vector<Cfg> cfgs;
    for (int sk : { 0, 1, 2, 4, 5, 8 })
        for (int mb : { 0, 1, 2, 4 })
            for (int npw : { 1, 4 })
                for (int wv : { 1, 2, 4 }) cfgs.push_back({ wv, sk, mb, npw, 1 });
    double best = 1e30;
    Cfg bc{};
    for (const Cfg& c : cfgs) {
        std::vector<RadTensor> ts = {
            T(d_a, RAD_F16, { M, K }), T(nullptr, RAD_F32, { 1 }), T(d_w, RAD_U8, { (int64_t)wq.size() }),
            T(d_s, RAD_U8, { (int64_t)ws.size() }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
            T(nullptr, RAD_F32, { 1 }) };
        absent(ts[1]); absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                     PS("dtype", bits == 5 ? "w5a16" : "w4a16_asym"),
                                     PI("wv", c.wv), PI("sk", c.sk), PI("mb", c.mb), PI("npw", c.npw),
                                     PI("nt", c.nt) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->launch(&a, nullptr) != RAD_OK || st_stream_sync(nullptr) != hipSuccess) continue;
        hipEvent_t e0, e1;
        st_ev_create(&e0); st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        const int iters = 12;
        for (int i = 0; i < iters; ++i) {
            ts[2].data = (uint8_t*)d_w + (size_t)(i % copies) * wq.size();
            ts[3].data = (uint8_t*)d_s + (size_t)(i % copies) * ws.size();
            r->launch(&a, nullptr);
        }
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0); st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        const double tf = 2.0 * M * N * K / (us * 1e6);
        std::printf("  wv=%d sk=%d mb=%d npw=%d  %9.1f us  %6.1f TF/s  %6.1f GB/s\n", c.wv, c.sk, c.mb,
                    c.npw, us, tf, (double)wq.size() / (us * 1e3));
        if (us < best) { best = us; bc = c; }
    }
    std::printf("best wv=%d sk=%d mb=%d npw=%d: %.1f us\n", bc.wv, bc.sk, bc.mb, bc.npw, best);
}


/* The 4/5-bit asymmetric x int8 prefill GEMM against the planes' meaning: u4/u5 codes, an f16
 * scale and a u8 zero per 128 of K, an int8 activation with an f32 scale per (row, 128), parts
 * side by side. */
static void wpa8i8_numeric(const Plugin& pl, int bits, int64_t M, int64_t N, int64_t K, int64_t pb1 = 0,
                            int64_t pb2 = 0) {
    const char* row = bits == 4 ? "gemm_w4a8i8_tiled" : "gemm_w5a8i8_tiled";
    if (!sel(row)) return;
    const PartCut cut(N, pb1, pb2);
    const int parts = cut.parts;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld parts=%d", row, (long long)M,
                  (long long)N, (long long)K, parts);
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || !r->relayout) { ok(nm, false, "row or hooks absent"); return; }
    const int64_t G = K / 128;
    const Planes p = asym_planes(bits, N, K);
    const int ic = p.find("codes"), is = p.find("scale"), iz = p.find("zero");
    const int64_t rb = p.row_bytes(ic);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128), PS("dtype", bits == 4 ? "w4a8i8_asym" : "w5a8i8") };
    std::vector<uint8_t> wq, ws;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale", "zero" }, &ws);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    const int64_t PK = parts * K, PG = parts * G;
    std::vector<uint8_t> ha((size_t)(M * PK));
    std::vector<float> av((size_t)(M * PK)), as((size_t)(M * PG));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t c = 0; c < PG; ++c)
            as[(size_t)(i * PG + c)] = 0.002f * (float)(1 + st_mix32((uint32_t)(i * PG + c) ^ 0xa5u) % 97);
        for (int64_t k = 0; k < PK; ++k) {
            const int q8 = (int)(st_mix32((uint32_t)(i * PK + k) ^ 0x3cu) % 255u) - 127;
            ha[(size_t)(i * PK + k)] = (uint8_t)(int8_t)q8;
            av[(size_t)(i * PK + k)] = (float)q8;
        }
    }
    std::vector<uint16_t> hy((size_t)(M * N), 0);
    void* d_a = dalloc(ha.size());
    void* d_s = dalloc(as.size() * 4);
    void* d_w = dalloc(wq.size());
    void* d_e = dalloc(ws.size());
    void* d_y = dalloc(hy.size() * 2);
    if (!d_a || !d_s || !d_w || !d_e || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_e, ws.data(), ws.size(), hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_a, RAD_I8, { M, PK }), T(d_s, RAD_F32, { M, PG }),
        T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_e, RAD_U8, { (int64_t)ws.size() }),
        T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_F32, { 1 }),
    };
    absent(ts[5]); absent(ts[6]);
    std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                 PS("dtype", bits == 4 ? "w4a8i8_asym" : "w5a8i8") };
    if (pb1) ps.push_back(PI("pb1", pb1));
    if (pb2) ps.push_back(PI("pb2", pb2));
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    if (r->scratch) {
        const int64_t sb = r->scratch(&a);
        if (sb > 0) { a.scratch = dalloc((size_t)sb); a.scratch_bytes = sb; }
    }
    const int rc = r->launch(&a, (RadStream) nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) { ok(nm, false, "rc=" + std::to_string(rc)); return; }
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; i += (M > 64 ? 3 : 1))
        for (int64_t n = 0; n < N; n = (n + 1 == N) ? N : std::min(n + nstep, N - 1)) {
            const int pt = cut.part_of(n);
            double acc = 0;
            for (int64_t g = 0; g < G; ++g) {
                double grp = 0;
                const double z = p.data[(size_t)iz][(size_t)(n * G + g)];
                const double sc = rad_load_f32(p.data[(size_t)is].data(), RAD_F16, n * G + g);
                for (int64_t k = g * 128; k < (g + 1) * 128; ++k) {
                    const double q = rad_load_f32(p.data[(size_t)ic].data() + n * rb, p.enc.plane[ic].dtype, k);
                    grp += (double)av[(size_t)(i * PK + pt * K + k)] * (q - z);
                }
                acc += grp * sc * (double)as[(size_t)(i * PG + pt * G + g)];
            }
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[96];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    if (M >= 1024) {                                  // a prefill chunk: say how fast it ran
        hipEvent_t e0, e1;
        st_ev_create(&e0); st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < 10; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0); st_ev_destroy(e1);
        std::snprintf(d, sizeof d, "rel_l2 %.3e, %.0f us, %.1f TF/s", rel, ms * 100.0,
                      2.0 * M * N * K / ((double)ms * 100.0 * 1e6));
        std::printf("        %s\n", d);
    }
    ok(nm, rel < 1e-2, d);
}


/* Two decode GEMMs the engine may overlap (gemm_mxfp4a8_decode is a concurrent row: its split
 * partial is the op's scratch and its arrival counters are keyed on the output). Each GEMM has its
 * own operands and a disjoint slice of one scratch allocation, with a split (ks > 1) so the partial
 * and counters are live. Each runs alone for a reference, then pairs run on two unordered
 * non-blocking streams, repeatedly, and every output must be byte-identical (the reducer sums
 * splits in a fixed order, so a lost or shared counter or partial shows as a different byte or a
 * hang). A final serial launch checks the counters were left at zero. */
struct MxdOverlapGemm {
    int64_t M, N, K;
    int ks;
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    void* y = nullptr;
    int64_t scratch = 0;
    bool ready = false;
};

static bool mxd_overlap_build(const RadKernelInfo* r, MxdOverlapGemm& g, int seed) {
    const int64_t M = g.M, N = g.N, K = g.K, nkb = K / 128;
    Planes p;
    p.enc = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    p.size(N, K);
    const int ic = p.find("codes"), is = p.find("scale");
    if (ic < 0 || is < 0) return false;
    mxfp4_fill(p, ic, is, N, K, (uint32_t)seed, false);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 32) };
    std::vector<uint8_t> wq, ws, wr;
    if (stored(r, lp, 2, p, { "codes" }, &wq) != RAD_OK ||
        stored(r, lp, 3, p, { "scale" }, &ws) != RAD_OK ||
        stored(r, lp, 6, p, { "scale" }, &wr) != RAD_OK)
        return false;
    std::vector<uint8_t> ha((size_t)(M * K));
    std::vector<float> as((size_t)(M * nkb));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t c = 0; c < nkb; ++c)
            as[(size_t)(i * nkb + c)] =
                0.002f * (float)(1 + st_mix32((uint32_t)(i * nkb + c) ^ (uint32_t)seed) % 97);
        for (int64_t k = 0; k < K; ++k)
            rad_store_f32(ha.data(), RAD_F8E4M3, i * K + k, mxfp4_act(i, k, (uint32_t)seed));
    }
    void* d_a = dalloc(ha.size());
    void* d_s = dalloc(as.size() * 4);
    void* d_w = dalloc(wq.size());
    void* d_e = dalloc(ws.size());
    void* d_r = dalloc(wr.size());
    g.y = dalloc((size_t)(M * N) * 2);
    if (!d_a || !d_s || !d_w || !d_e || !d_r || !g.y) return false;
    st_memcpy(d_a, ha.data(), ha.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_e, ws.data(), ws.size(), hipMemcpyHostToDevice);
    st_memcpy(d_r, wr.data(), wr.size(), hipMemcpyHostToDevice);
    g.ts = { T(d_a, RAD_F8E4M3, { M, K }), T(d_s, RAD_F32, { M, nkb }),
             T(d_w, RAD_U8, { (int64_t)wq.size() }), T(d_e, RAD_U8, { (int64_t)ws.size() }),
             T(g.y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
             T(d_r, RAD_U8, { (int64_t)wr.size() }) };
    absent(g.ts[5]);
    g.ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 32), PS("dtype", "mxfp4a8"),
             PI("ks", g.ks), PI("bk", 128), PI("nt", 1) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = g.ts.data(); a.n_t = (int)g.ts.size();
    a.p = g.ps.data(); a.n_p = (int)g.ps.size();
    a.world_size = 1;
    g.scratch = r->scratch ? r->scratch(&a) : 0;
    g.ready = g.scratch >= (int64_t)g.ks * M * N * 4;
    return g.ready;
}

static int mxd_overlap_launch(const RadKernelInfo* r, MxdOverlapGemm& g, void* scratch,
                              hipStream_t st, int64_t scratch_bytes = -1) {
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = g.ts.data(); a.n_t = (int)g.ts.size();
    a.p = g.ps.data(); a.n_p = (int)g.ps.size();
    a.world_size = 1;
    a.scratch = scratch;
    a.scratch_bytes = scratch_bytes < 0 ? g.scratch : scratch_bytes;
    return r->launch(&a, (RadStream)st);
}

static void mxd_overlap_case(const Plugin& pl) {
    if (!sel("gemm_mxfp4a8_decode_overlap") && !sel("gemm_mxfp4a8_decode")) return;
    const char* nm = "gemm_mxfp4a8_decode: two split GEMMs on two streams, disjoint scratch, "
                     "byte-identical to serial";
    const RadKernelInfo* r = find(pl, "gemm_mxfp4a8_decode");
    if (!r || !r->layout || !r->relayout || !r->scratch) { ok(nm, false, "row or hooks absent"); return; }
    MxdOverlapGemm g[3];
    g[0] = MxdOverlapGemm{}; g[0].M = 8;  g[0].N = 400;  g[0].K = 2560; g[0].ks = 4;
    g[1] = MxdOverlapGemm{}; g[1].M = 8;  g[1].N = 272;  g[1].K = 2560; g[1].ks = 2;
    g[2] = MxdOverlapGemm{}; g[2].M = 24; g[2].N = 1040; g[2].K = 1152; g[2].ks = 4;
    for (int i = 0; i < 3; ++i)
        if (!mxd_overlap_build(r, g[i], 1 + 2 * i)) {
            ok(nm, false, "operands or scratch hook (ks > 1 needs a split partial)");
            return;
        }
    /* One allocation, a slice each, 256-byte aligned: the core's disjoint scratch slices. */
    int64_t off[3], tot = 0;
    for (int i = 0; i < 3; ++i) { off[i] = tot; tot += (g[i].scratch + 255) & ~int64_t(255); }
    char* base = (char*)dalloc((size_t)tot);
    if (!base) { ok(nm, false, "hipMalloc failed"); return; }
    /* The serial reference gets its own scratch, sharing nothing with the overlapped runs. */
    std::vector<std::vector<uint16_t>> want(3);
    for (int i = 0; i < 3; ++i) {
        void* sc = dalloc((size_t)g[i].scratch);
        if (!sc) { ok(nm, false, "hipMalloc failed"); return; }
        const int rc = mxd_overlap_launch(r, g[i], sc, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            ok(nm, false, "serial rc=" + std::to_string(rc));
            return;
        }
        want[(size_t)i].assign((size_t)(g[i].M * g[i].N), 0);
        st_memcpy(want[(size_t)i].data(), g[i].y, want[(size_t)i].size() * 2, hipMemcpyDeviceToHost);
    }
    hipStream_t s1 = nullptr, s2 = nullptr;
    if (hipStreamCreateWithFlags(&s1, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&s2, hipStreamNonBlocking) != hipSuccess) {
        ok(nm, false, "hipStreamCreate failed");
        return;
    }
    int bad = 0, iters = 0;
    std::string why;
    for (int it = 0; it < 200 && !bad; ++it) {
        const int a = it % 3, b = (it + 1 + (it / 3) % 2) % 3;   /* every ordered pair of distinct GEMMs */
        for (int i : { a, b }) st_memset(g[i].y, 0, (size_t)(g[i].M * g[i].N) * 2);
        /* The streams are non-blocking, so nothing orders them after the legacy-stream memset. */
        (void)hipDeviceSynchronize();
        const int ra = mxd_overlap_launch(r, g[a], base + off[a], s1);
        const int rb = mxd_overlap_launch(r, g[b], base + off[b], s2);
        const bool sa = st_stream_sync(s1) == hipSuccess, sb = st_stream_sync(s2) == hipSuccess;
        if (ra != RAD_OK || rb != RAD_OK || !sa || !sb) {
            bad = 1;
            why = "launch rc " + std::to_string(ra) + "/" + std::to_string(rb);
            break;
        }
        for (int i : { a, b }) {
            std::vector<uint16_t> got((size_t)(g[i].M * g[i].N));
            st_memcpy(got.data(), g[i].y, got.size() * 2, hipMemcpyDeviceToHost);
            if (std::memcmp(got.data(), want[(size_t)i].data(), got.size() * 2) != 0) {
                bad = 1;
                size_t nd = 0, first = got.size(), nz = 0;
                for (size_t j = 0; j < got.size(); ++j) {
                    if (got[j] != want[(size_t)i][j]) { ++nd; if (first == got.size()) first = j; }
                    if (got[j] == 0) ++nz;
                }
                why = "iter " + std::to_string(it) + ": GEMM " + std::to_string(i) + " (ks=" +
                      std::to_string(g[i].ks) + ") differs from serial: " + std::to_string(nd) +
                      "/" + std::to_string(got.size()) + " elements, first at " +
                      std::to_string(first) + " (m=" + std::to_string(first / (size_t)g[i].N) +
                      " n=" + std::to_string(first % (size_t)g[i].N) + "), " +
                      std::to_string(nz) + " zero, partner GEMM " + std::to_string(i == a ? b : a);
            }
        }
        ++iters;
    }
    (void)hipStreamDestroy(s1);
    (void)hipStreamDestroy(s2);
    ok(nm, !bad, bad ? why : std::to_string(iters) + " overlapped pairs identical");
    /* The counters must be zero again: a serial launch afterwards is still byte-identical. */
    bool again = true;
    for (int i = 0; i < 3 && again; ++i) {
        st_memset(g[i].y, 0, (size_t)(g[i].M * g[i].N) * 2);
        const int rc = mxd_overlap_launch(r, g[i], base + off[i], nullptr);
        std::vector<uint16_t> got((size_t)(g[i].M * g[i].N));
        again = rc == RAD_OK && st_stream_sync(nullptr) == hipSuccess;
        if (again) {
            st_memcpy(got.data(), g[i].y, got.size() * 2, hipMemcpyDeviceToHost);
            again = std::memcmp(got.data(), want[(size_t)i].data(), got.size() * 2) == 0;
        }
    }
    ok("gemm_mxfp4a8_decode: serial after the overlap storm is still identical (counters left at "
       "zero)", again);
    /* A split given less scratch than its partial is refused, not run unsplit: the hook sized it
     * for the whole band, so a short one is a caller error. */
    const int rc = mxd_overlap_launch(r, g[0], base + off[0], nullptr, g[0].scratch / 2);
    (void)st_stream_sync(nullptr);
    ok("gemm_mxfp4a8_decode: a split with half its scratch is refused", rc != RAD_OK,
       "rc=" + std::to_string(rc));
}

/* The int8 trunk chain: quant_act_i8g on a bf16 activation, then an i8a8 GEMM on its output,
 * against the planes' meaning. The weight is i8 with a bf16 scale per row per 128 K (libquant's
 * `codes=i8 group=128 scale=bf16 clamp=sym`), stored through the row's hooks. The reference uses
 * the codes the quantiser wrote, read back, so only the GEMM is under test; the quantiser is
 * checked separately, code for code. Ragged M reaches the guarded fold. */
static void i8a8_numeric(const Plugin& pl, const char* row, int64_t M, int64_t N, int64_t K) {
    if (!sel(row) && !sel("quant_act_i8g")) return;
    char nm[160];
    std::snprintf(nm, sizeof nm, "%s numeric M=%lld N=%lld K=%lld", row, (long long)M,
                  (long long)N, (long long)K);
    const RadKernelInfo* r = find(pl, row);
    const RadKernelInfo* qz = find(pl, "quant_act_i8g");
    if (!r || !qz || !r->layout || !r->relayout) { ok(nm, false, "rows or hooks absent"); return; }
    const int64_t nkb = K / 128;

    std::vector<float> wf((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            wf[(size_t)(n * K + k)] = std::ldexp(1.0f, (int)((n / 16) % 5) - 2 - (int)((k / 128) % 3))
                                      * (0.031f * (float)((n * 37 + k * 11) % 61) - 0.9f);
    Planes p;
    p.enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    p.size(N, K);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < nkb; ++g) {
            float amax = 0.0f;
            for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(wf[(size_t)(n * K + g * 128 + j)]));
            rad_store_f32(p.data[1].data(), RAD_BF16, n * nkb + g, amax > 0 ? amax / 127.0f : 1.0f);
            const float sc = rad_load_f32(p.data[1].data(), RAD_BF16, n * nkb + g);
            for (int64_t j = 0; j < 128; ++j) {
                const float q = std::nearbyint(wf[(size_t)(n * K + g * 128 + j)] / sc);
                p.data[0][(size_t)(n * K + g * 128 + j)] =
                    (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
        }
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wsc;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &wsc);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok(nm, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    {
        std::string why;
        const std::string what = std::string(row) + " i8 codes and tile-major scales invert";
        ok(what.c_str(), restores(r, lp, 2, p, { "codes" }, wq, &why) &&
                         restores(r, lp, 3, p, { "scale" }, wsc, &why), why);
    }

    std::vector<uint16_t> ha((size_t)(M * K));
    for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K; ++k)
            ha[(size_t)(i * K + k)] = h_to_bf16((float)(((i * 7 + k * 13) % 401) - 200) * 0.05f *
                                                ((k % 3) ? 1.f : -1.f) * ((k % 517 == 3) ? 30.f : 1.f));
    void* d_a  = dalloc(ha.size() * 2);
    void* d_q  = dalloc((size_t)(M * K));
    void* d_qs = dalloc((size_t)(M * nkb) * 4);
    void* d_w  = dalloc(wq.size());
    void* d_s  = dalloc(wsc.size());
    void* d_y  = dalloc((size_t)(M * N) * 2);
    if (!d_a || !d_q || !d_qs || !d_w || !d_s || !d_y) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(d_a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_s, wsc.data(), wsc.size(), hipMemcpyHostToDevice);

    auto launch = [&](const RadKernelInfo* k, std::vector<RadTensor> ts, std::vector<RadParam> ps) {
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        void* scr = nullptr;
        if (k->scratch) {
            const int64_t need = k->scratch(&a);
            if (need > 0) { scr = dalloc((size_t)need); a.scratch = scr; a.scratch_bytes = need; }
        }
        const int rc = k->launch(&a, (RadStream) nullptr);
        return rc == RAD_OK && st_stream_sync(nullptr) == hipSuccess ? RAD_OK : (rc ? rc : -1);
    };
    int rc = launch(qz, { T(d_a, RAD_BF16, { M, K }), T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, nkb }) },
                    { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc != RAD_OK) { ok(nm, false, "quant_act_i8g rc " + std::to_string(rc)); return; }
    std::vector<int8_t> hq((size_t)(M * K));
    std::vector<float> hqs((size_t)(M * nkb));
    st_memcpy(hq.data(), d_q, hq.size(), hipMemcpyDeviceToHost);
    st_memcpy(hqs.data(), d_qs, hqs.size() * 4, hipMemcpyDeviceToHost);
    {
        int64_t bad = 0;
        for (int64_t i = 0; i < M; ++i)
            for (int64_t g = 0; g < nkb; ++g) {
                float amax = 0.0f;
                for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(h_bf16(ha[(size_t)(i * K + g * 128 + j)])));
                const float sc = amax > 0 ? amax * (1.0f / 127.0f) : 1.0f;
                bad += hqs[(size_t)(i * nkb + g)] != sc;
                const float inv = 1.0f / sc;
                for (int64_t j = 0; j < 128; ++j) {
                    const float v = h_bf16(ha[(size_t)(i * K + g * 128 + j)]) * inv;
                    const int c = (int)std::nearbyint(std::max(-127.0f, std::min(127.0f, v)));
                    bad += hq[(size_t)(i * K + g * 128 + j)] != c;
                }
            }
        const std::string what =
            "quant_act_i8g vs a host quantiser, M=" + std::to_string(M) + " K=" + std::to_string(K);
        ok(what.c_str(), bad == 0, std::to_string(bad) + " codes or scales differ");
    }

    std::vector<RadTensor> ts = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, nkb }), T(d_w, RAD_I8, { N, K }),
        T(d_s, RAD_BF16, { N, nkb }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[5]); absent(ts[6]);
    rc = launch(r, ts, { PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "i8a8") });
    if (rc != RAD_OK) { ok(nm, false, "rc=" + std::to_string(rc)); return; }
    std::vector<uint16_t> hy((size_t)(M * N));
    st_memcpy(hy.data(), d_y, hy.size() * 2, hipMemcpyDeviceToHost);
    const int64_t nstep = N > 96 ? N / 96 : 1;
    double num = 0, den = 0;
    for (int64_t i = 0; i < M; ++i)
        for (int64_t n = 0; n < N; n += (n + nstep < N ? nstep : 1)) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)hq[(size_t)(i * K + k)] * hqs[(size_t)(i * nkb + k / 128)] *
                       (double)(int8_t)p.data[0][(size_t)(n * K + k)] *
                       rad_load_f32(p.data[1].data(), RAD_BF16, n * nkb + k / 128);
            const double d = (double)h_bf16(hy[(size_t)(i * N + n)]) - acc;
            num += d * d;
            den += acc * acc;
        }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    char d[64];
    std::snprintf(d, sizeof d, "rel_l2 %.3e", rel);
    ok(nm, rel < 5e-3, d);
}


/* The tiled bf16 GEMM against the skinny one, byte for byte, at every tile variant. An op over
 * M 1..max_tok is served by the skinny row at M <= 64 and the tiled one above, so they must agree
 * exactly or a decode step and a prefill chunk would disagree about the same row. The shapes reach
 * one split-K slice and several (N 512 gives 4; K 64 one per k step), with ragged M and N so a
 * stored clamped fragment row would show. */
static void bf16_tiled_identity(const Plugin& pl, int64_t M, int64_t N, int64_t K) {
    if (!sel("gemm_bf16_nt_tiled")) return;
    char nm[128];
    std::snprintf(nm, sizeof nm, "gemm_bf16_nt_tiled == gemm_bf16_nt_m64  M=%lld N=%lld K=%lld",
                  (long long)M, (long long)N, (long long)K);
    const RadKernelInfo* rt = find(pl, "gemm_bf16_nt_tiled");
    const RadKernelInfo* rs = find(pl, "gemm_bf16_nt_m64");
    if (!rt || !rs) { ok(nm, false, "row absent"); return; }

    std::vector<uint16_t> ha((size_t)(M * K)), hw((size_t)(N * K));
    for (size_t i = 0; i < ha.size(); ++i)
        ha[i] = h_to_bf16((float)((int)(st_mix32((uint32_t)i) % 401) - 200) * 0.5f);
    for (size_t i = 0; i < hw.size(); ++i)
        hw[i] = h_to_bf16((float)((int)(st_mix32((uint32_t)i ^ 0x9e3779b9u) % 199) - 99) * 0.01f);

    std::vector<RadTensor> ts = {
        T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
        T(nullptr, RAD_BF16, { M, N }),
    };
    for (auto& t : ts) t.data = alloc_for(t);
    if (!ts[0].data || !ts[1].data || !ts[2].data) { ok(nm, false, "hipMalloc failed"); return; }
    st_memcpy(ts[0].data, ha.data(), ha.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(ts[1].data, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);

    auto run = [&](const RadKernelInfo* r, int64_t tile, std::vector<uint16_t>& out) {
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
        if (tile >= 0) ps.push_back(PI("tile", tile));
        /* Poisoned before each run, since a stale output would compare equal. */
        if (hipMemset(ts[2].data, 0xA5, (size_t)(M * N) * 2) != hipSuccess) return -1;
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        const int rc = r->launch(&a, (RadStream) nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return rc != RAD_OK ? rc : -1;
        out.assign((size_t)(M * N), 0);
        st_memcpy(out.data(), ts[2].data, out.size() * 2, hipMemcpyDeviceToHost);
        return 0;
    };
    std::vector<uint16_t> ref, got;
    if (run(rs, -1, ref) != 0) { ok(nm, false, "skinny launch failed"); return; }
    std::string d;
    bool all = true;
    for (int64_t tile = 0; tile <= 2; ++tile) {
        const int rc = run(rt, tile, got);
        size_t diff = 0;
        if (rc == 0)
            for (size_t i = 0; i < ref.size(); ++i) diff += ref[i] != got[i];
        const bool pass = rc == 0 && diff == 0;
        all = all && pass;
        char b[64];
        std::snprintf(b, sizeof b, "%stile %lld: %s", d.empty() ? "" : ", ", (long long)tile,
                      rc ? "launch failed" : (diff ? (std::to_string(diff) + " differ").c_str()
                                                   : "same"));
        d += b;
    }
    ok(nm, all, d);
}

/* logit_softcap, against an f64 evaluation of the same formula.
 *
 * Two caps: the model's (Gemma 4 issues 30.0) and a smaller one, so a hard-wired 30.0 or a cap
 * swapped with its inverse fails one of them. Inputs are 0, 0.5, 1, 2 and 5 times the cap, signed,
 * so x/cap is exact and only the kernel's f32 arithmetic and bf16 store can differ. Tolerance: 1%
 * relative, 1e-3 absolute. */
static void logit_softcap_case(const Plugin& pl) {
    const char* what = "logit_softcap vs an f64 evaluation of the same formula";
    /* Both dtypes: bf16 (the GEMM's) and f32 (the sampling chain's logits). The f32 leg is
     * compared in f32, not through a bf16 round trip. */
    for (const uint32_t dt : { (uint32_t)RAD_BF16, (uint32_t)RAD_F32 }) {
        const char* row = dt == RAD_BF16 ? "logit_softcap_bf16" : "logit_softcap_f32";
        const size_t es = dt == RAD_BF16 ? 2 : 4;
        const RadKernelInfo* r = find(pl, row);
        if (!r) { ok(what, false, std::string("row absent: ") + row); return; }

        const int64_t M = 7, n = 128;
        for (const double capd : { 30.0, 10.0 }) {
            std::vector<uint16_t> xh((size_t)M * n);
            std::vector<float>    xf((size_t)M * n);
            std::vector<double>  want((size_t)M * n);
            const double mags[5] = { 0.0, 0.5, 1.0, 2.0, 5.0 };
            for (int64_t i = 0; i < M * n; ++i) {
                const double mag = mags[(int)((i * 31) % 5)];
                const double sgn = (mag == 0.0 || (i * 17) % 2 == 0) ? 1.0 : -1.0;
                const double xv = sgn * mag * capd;
                xh[(size_t)i] = h_to_bf16((float)xv);
                xf[(size_t)i] = (float)xv;
                want[(size_t)i] = capd * std::tanh(xv / capd);
            }

            void* d_x = dalloc((size_t)(M * n) * es);
            void* d_o = dalloc((size_t)(M * n) * es);
            if (!d_x || !d_o) { ok(what, false, "hipMalloc failed"); return; }
            st_memcpy(d_x, dt == RAD_BF16 ? (const void*)xh.data() : (const void*)xf.data(),
                      (size_t)(M * n) * es, hipMemcpyHostToDevice);

            std::vector<RadTensor> ts = {
                T(d_x, dt, { M, n }),
                T(d_o, dt, { M, n }),
            };
            std::vector<RadParam> ps = { PI("n", n), PF("cap", capd) };
            RadArgs a;
            std::memset(&a, 0, sizeof(a));
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps.data(); a.n_p = (int)ps.size();

            const int rc = r->launch(&a, nullptr);
            st_stream_sync(nullptr);
            if (rc != RAD_OK) {
                ok(what, false, "launch rc=" + std::to_string(rc) + " cap=" + std::to_string(capd));
                return;
            }

            std::vector<uint16_t> got((size_t)(M * n));
            std::vector<float>    gotf((size_t)(M * n));
            st_memcpy(dt == RAD_BF16 ? (void*)got.data() : (void*)gotf.data(), d_o,
                      (size_t)(M * n) * es, hipMemcpyDeviceToHost);
            double worst = 0.0;
            int64_t bad = 0;
            for (int64_t i = 0; i < M * n; ++i) {
                const double v = dt == RAD_BF16 ? (double)h_bf16(got[(size_t)i])
                                                : (double)gotf[(size_t)i];
                const double e = v - want[(size_t)i];
                if (std::fabs(e) > worst) worst = std::fabs(e);
                if (std::fabs(e) > 1e-2 * (std::fabs(want[(size_t)i]) + 1e-3)) ++bad;
            }
            ok(what, bad == 0, std::string(row) + " cap=" + std::to_string(capd) +
                 " worst_abs=" + std::to_string(worst) + " bad=" + std::to_string(bad));
        }
    }
}

/* rmsnorm_p, against an f64 evaluation of the weightless normalisation.
 *
 * Two eps values: the model's (Gemma 4 issues 1e-6) and 1e-2, so a hard-wired 1e-6 or a dropped
 * +eps fails the larger one. The inputs are a fixed set of signed magnitudes, and the f64 oracle
 * x * rsqrt(mean(x^2) + eps) is computed from the bf16 values the kernel reads. Tolerance: 1%
 * relative, 1e-3 absolute. */
static void rmsnorm_p_case(const Plugin& pl) {
    const char* what = "rmsnorm_p vs an f64 evaluation of the weightless normalisation";
    const RadKernelInfo* r = find(pl, "rmsnorm_p_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 7, n = 640;
    for (const double epsd : { 1e-6, 1e-2 }) {
        std::vector<uint16_t> xh((size_t)M * n);
        const double mags[5] = { 0.0, 0.5, 1.0, 2.0, 5.0 };
        for (int64_t i = 0; i < M * n; ++i) {
            const double mag = mags[(int)((i * 31) % 5)];
            const double sgn = (mag == 0.0 || (i * 17) % 2 == 0) ? 1.0 : -1.0;
            xh[(size_t)i] = h_to_bf16((float)(sgn * mag));
        }

        // f64 oracle from the bf16 values the kernel reads: y = x * rsqrt(mean(x^2) + eps).
        std::vector<double> want((size_t)M * n);
        for (int64_t m = 0; m < M; ++m) {
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const double v = (double)h_bf16(xh[(size_t)m * n + i]);
                ss += v * v;
            }
            const double s = 1.0 / std::sqrt(ss / (double)n + epsd);
            for (int64_t i = 0; i < n; ++i)
                want[(size_t)m * n + i] = (double)h_bf16(xh[(size_t)m * n + i]) * s;
        }

        void* d_x = dalloc(xh.size() * 2);
        void* d_o = dalloc((size_t)(M * n) * 2);
        if (!d_x || !d_o) { ok(what, false, "hipMalloc failed"); return; }
        st_memcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice);

        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { M, n }),
            T(d_o, RAD_BF16, { M, n }),
        };
        std::vector<RadParam> ps = { PI("n", n), PF("eps", epsd) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();

        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) {
            ok(what, false, "launch rc=" + std::to_string(rc) + " eps=" + std::to_string(epsd));
            return;
        }

        std::vector<uint16_t> got((size_t)(M * n));
        st_memcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        for (int64_t i = 0; i < M * n; ++i) {
            const double e = (double)h_bf16(got[(size_t)i]) - want[(size_t)i];
            if (std::fabs(e) > worst) worst = std::fabs(e);
            if (std::fabs(e) > 1e-2 * (std::fabs(want[(size_t)i]) + 1e-3)) ++bad;
        }
        ok(what, bad == 0, "eps=" + std::to_string(epsd) +
             " worst_abs=" + std::to_string(worst) + " bad=" + std::to_string(bad));
    }
}

/* gelu_mul, against an f64 evaluation of the tanh-approximation GELU.
 *
 * Fixed signed magnitudes for the gate and up halves; the f64 oracle
 * 0.5*g*(1 + tanh(0.7978845608028654*(g + 0.044715*g^3))) * u uses the bf16 values the kernel
 * reads, the gate being the first half of the [M, 2n] row. Tolerance: 1% relative, 1e-3
 * absolute. */
static void gelu_mul_case(const Plugin& pl) {
    const char* what = "gelu_mul vs an f64 evaluation of the tanh GELU";
    const RadKernelInfo* r = find(pl, "gelu_mul_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 7, n = 640;
    std::vector<uint16_t> gh((size_t)M * 2 * n);
    const double mags[5] = { 0.0, 0.5, 1.0, 2.0, 5.0 };
    for (int64_t i = 0; i < M * 2 * n; ++i) {
        const double mag = mags[(int)((i * 31) % 5)];
        const double sgn = (mag == 0.0 || (i * 17) % 2 == 0) ? 1.0 : -1.0;
        gh[(size_t)i] = h_to_bf16((float)(sgn * mag));
    }

    // f64 oracle from the bf16 values the kernel reads: y = gelu_tanh(gate) * up, gate the first half.
    std::vector<double> want((size_t)M * n);
    const double k0 = 0.7978845608028654, k1 = 0.044715;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t i = 0; i < n; ++i) {
            const double g = (double)h_bf16(gh[(size_t)m * 2 * n + i]);
            const double u = (double)h_bf16(gh[(size_t)m * 2 * n + n + i]);
            const double gelu = 0.5 * g * (1.0 + std::tanh(k0 * (g + k1 * g * g * g)));
            want[(size_t)m * n + i] = gelu * u;
        }
    }

    void* d_g = dalloc(gh.size() * 2);
    void* d_o = dalloc((size_t)(M * n) * 2);
    if (!d_g || !d_o) { ok(what, false, "hipMalloc failed"); return; }
    st_memcpy(d_g, gh.data(), gh.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_g, RAD_BF16, { M, 2 * n }),
        T(d_o, RAD_BF16, { M, n }),
    };
    std::vector<RadParam> ps = { PI("n", n) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) {
        ok(what, false, "launch rc=" + std::to_string(rc));
        return;
    }

    std::vector<uint16_t> got((size_t)(M * n));
    st_memcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost);
    double worst = 0.0;
    int64_t bad = 0;
    for (int64_t i = 0; i < M * n; ++i) {
        const double e = (double)h_bf16(got[(size_t)i]) - want[(size_t)i];
        if (std::fabs(e) > worst) worst = std::fabs(e);
        if (std::fabs(e) > 1e-2 * (std::fabs(want[(size_t)i]) + 1e-3)) ++bad;
    }
    ok(what, bad == 0, "worst_abs=" + std::to_string(worst) + " bad=" + std::to_string(bad));
}

/* scale, against an f64 evaluation of the same multiply.
 *
 * Three scalars (Gemma 4's per-layer scalar is 0.03-0.99): two positive of different magnitude,
 * catching a hard-wired or squared scalar, and a negative one, catching a dropped sign. Inputs are
 * fixed signed magnitudes. Tolerance: 1% relative, 1e-3 absolute. The scalar is stored as f32; the
 * bf16 `s` branch is not exercised (same multiply). */
static void scale_case(const Plugin& pl) {
    const char* what = "scale vs an f64 evaluation of the same multiply";
    const RadKernelInfo* r = find(pl, "scale_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 7, n = 128;
    for (const double sv : { 2.0, 0.5, -1.5 }) {
        std::vector<uint16_t> xh((size_t)M * n);
        std::vector<double>  want((size_t)M * n);
        const double mags[5] = { 0.0, 0.5, 1.0, 2.0, 3.0 };
        for (int64_t i = 0; i < M * n; ++i) {
            const double mag = mags[(int)((i * 31) % 5)];
            const double sgn = (mag == 0.0 || (i * 17) % 2 == 0) ? 1.0 : -1.0;
            const double xv = sgn * mag;
            xh[(size_t)i] = h_to_bf16((float)xv);
            want[(size_t)i] = xv * sv;
        }

        void* d_x = dalloc(xh.size() * 2);
        void* d_s = dalloc(4);
        void* d_o = dalloc((size_t)(M * n) * 2);
        if (!d_x || !d_s || !d_o) { ok(what, false, "hipMalloc failed"); return; }
        st_memcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice);
        const float sf = (float)sv;
        st_memcpy(d_s, &sf, 4, hipMemcpyHostToDevice);

        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { M, n }),
            T(d_s, RAD_F32, { 1 }),
            T(d_o, RAD_BF16, { M, n }),
        };
        std::vector<RadParam> ps = { PI("n", n) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();

        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) {
            ok(what, false, "launch rc=" + std::to_string(rc) + " s=" + std::to_string(sv));
            return;
        }

        std::vector<uint16_t> got((size_t)(M * n));
        st_memcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        for (int64_t i = 0; i < M * n; ++i) {
            const double e = (double)h_bf16(got[(size_t)i]) - want[(size_t)i];
            if (std::fabs(e) > worst) worst = std::fabs(e);
            if (std::fabs(e) > 1e-2 * (std::fabs(want[(size_t)i]) + 1e-3)) ++bad;
        }
        ok(what, bad == 0, "s=" + std::to_string(sv) +
             " worst_abs=" + std::to_string(worst) + " bad=" + std::to_string(bad));
    }
}

/* rope_bf16 against an f32 mirror of rope_kernel at each ladder width. `den` is the ladder's
 * width (`rot` for `rotary`/absent, `head_dim` for `head`). The V head must stay untouched. rel_l2
 * catches a wrong angle, which magnitude alone would not (rotation preserves length). */
static void rope_ladder_case(const Plugin& pl) {
    const char* what = "rope ladder vs an f32 mirror of rope_kernel at each width";
    const RadKernelInfo* r = find(pl, "rope_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 5, hd = 64, rot = 32, half = rot / 2;
    const int64_t n_head = 2, nkv = 1, n_rope = n_head + nkv;   /* 3 rotate, one V does not */
    const int64_t row = (n_head + 2 * nkv) * hd;
    const float theta = 10000.0f;
    const int32_t pos[5] = { 0, 1, 7, 128, 4096 };

    for (const char* lad : std::initializer_list<const char*>{ nullptr, "rotary", "head" }) {
        const int64_t den = (lad && !std::strcmp(lad, "head")) ? hd : rot;
        const char* tag = lad ? lad : "absent";

        std::vector<uint16_t> saved((size_t)M * row);
        for (int64_t i = 0; i < M * row; ++i) {
            float v = (float)(((i * 5) % 17) - 8) / 9.0f;
            if (i % 13 == 0) v *= 4.0f;                        /* a few big channels */
            saved[(size_t)i] = h_to_bf16(v);
        }

        void* d_qkv = dalloc(saved.size() * 2);
        void* d_pos = dalloc(M * 4);
        if (!d_qkv || !d_pos) { ok(what, false, "hipMalloc failed"); return; }
        st_memcpy(d_qkv, saved.data(), saved.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_pos, pos, M * 4, hipMemcpyHostToDevice);

        std::vector<RadTensor> ts = { T(d_qkv, RAD_BF16, { M, row }), T(d_pos, RAD_I32, { M }) };
        std::vector<RadParam> ps = { PI("M", M), PI("head_dim", hd), PI("n_head", n_head),
                                     PI("n_head_kv", nkv), PF("theta", (double)theta),
                                     PF("scale", 1.0), PS("mode", "neox"),
                                     PI("rotary_dim", rot) };
        if (lad) ps.push_back(PS("ladder", lad));
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();

        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, std::string("ladder=") + tag + " rc=" + std::to_string(rc)); return; }

        std::vector<uint16_t> got((size_t)M * row);
        st_memcpy(got.data(), d_qkv, got.size() * 2, hipMemcpyDeviceToHost);

        double num = 0.0, dnorm = 0.0;
        int64_t untouched_bad = 0;
        for (int64_t m = 0; m < M; ++m) {
            const size_t base = (size_t)m * row;
            for (int64_t h = 0; h < n_rope; ++h) {
                for (int64_t i = 0; i < half; ++i) {
                    const size_t i0 = base + (size_t)h * hd + i;
                    /* `head` is HF's proportional RoPE: pair i is (i, i + head_dim/2). */
                    const size_t i1 = i0 + ((lad && !std::strcmp(lad, "head")) ? hd / 2 : half);
                    const float x0 = h_bf16(saved[(size_t)i0]), x1 = h_bf16(saved[(size_t)i1]);
                    const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)den);
                    float s, c;
                    sincosf((float)pos[(size_t)m] * inv, &s, &c);
                    const float w0 = x0 * c - x1 * s, w1 = x0 * s + x1 * c;
                    const float g0 = h_bf16(got[(size_t)i0]), g1 = h_bf16(got[(size_t)i1]);
                    num += (double)(g0 - w0) * (double)(g0 - w0) +
                           (double)(g1 - w1) * (double)(g1 - w1);
                    dnorm += (double)w0 * w0 + (double)w1 * w1;
                }
            }
            for (int64_t e = 0; e < hd; ++e)   /* the V head is untouched, so exact */
                if (got[base + (size_t)n_rope * hd + e] != saved[base + (size_t)n_rope * hd + e])
                    ++untouched_bad;
        }
        const double rel = dnorm > 0.0 ? std::sqrt(num / dnorm) : 0.0;
        ok(what, rel < 2e-3 && untouched_bad == 0, std::string("ladder=") + tag + " rel_l2=" +
             std::to_string(rel) + " vhead-moved=" + std::to_string(untouched_bad));
    }
}

/* ---- the two fp8 fusions, against the pair each one replaces -------------------------------- */
//
// Byte-identical, so the comparison is against the unfused kernels, not a host mirror: generated
// text must not move across builds, and a last-place code difference on a few channels would pass
// any tolerance.
//
// The unfused pair writes a bf16 norm output and the quantiser reads it back, so amax and codes are
// taken over bf16-rounded values. A fused kernel quantising its f32 register value is more
// accurate, and wrong.
static int launch_row(const RadKernelInfo* r, std::vector<RadTensor> ts,
                      std::vector<RadParam> ps) {
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    const int rc = r->launch(&a, (RadStream)nullptr);
    if (rc != RAD_OK) return rc;
    return st_stream_sync(nullptr) == hipSuccess ? RAD_OK : RAD_E_STATE;
}

static void fused_quant_fp8_numeric(const Plugin& pl) {
    const RadKernelInfo* nrm = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* smu = find(pl, "silu_mul_bf16");
    const RadKernelInfo* qz  = find(pl, "quant_act_fp8");
    const RadKernelInfo* rq  = find(pl, "rmsnorm_quant_fp8");
    const RadKernelInfo* gq  = find(pl, "gated_quant_fp8");
    const RadKernelInfo* sg  = find(pl, "sigmoid_bf16");
    const RadKernelInfo* ml  = find(pl, "mul_bf16");
    const RadKernelInfo* tq  = find(pl, "gate_quant_fp8");
    if (!nrm || !smu || !qz || !rq || !gq || !sg || !ml || !tq) {
        ok("fp8 fusions vs the unfused pair", false, "rows absent");
        return;
    }
    /* M is a speculative decode step's token count; K is five 128-column groups, not a power of
     * two, so a wrong group stride lands somewhere visible. */
    const int64_t M = 8, K = 640, nkb = K / 128;

    std::vector<uint16_t> hx((size_t)M * K), hgu((size_t)M * 2 * K);
    std::vector<float>    hw((size_t)K);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < K; ++c) {
            /* Per-group magnitudes an order of magnitude apart, so one group's amax cannot stand
               in for its neighbour's, and one channel per row far above the rest (a dropped clamp
               turns it into a NaN). */
            const float m = std::ldexp(1.0f, (int)(c / 128) - 2);
            const float v = m * (0.037f * (float)((r * 29 + c * 13) % 71) - 1.3f);
            hx[(size_t)r * K + c]        = h_to_bf16(c == (r * 37) % K ? v * 19.0f : v);
            hgu[(size_t)r * 2 * K + c]   = h_to_bf16(v * 0.75f);
            hgu[(size_t)r * 2 * K + K + c] = h_to_bf16(m * (0.011f * (float)((r + c * 7) % 43) - 0.2f));
        }
    /* A zero-centred gain, as GemmaRMSNorm stores, so `wadd` matters. */
    for (int64_t c = 0; c < K; ++c) hw[(size_t)c] = 0.004f * (float)((c * 19) % 51) - 0.1f;

    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_gu = dalloc((size_t)M * 2 * K * 2);
    void* d_w  = dalloc((size_t)K * 4);
    void* d_y[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
    void* d_q[2] = { dalloc((size_t)M * K),     dalloc((size_t)M * K) };
    void* d_s[2] = { dalloc((size_t)M * nkb * 4), dalloc((size_t)M * nkb * 4) };
    if (!d_x || !d_gu || !d_w || !d_y[1] || !d_q[1] || !d_s[1]) {
        ok("fp8 fusion alloc", false, "hipMalloc failed");
        return;
    }
    st_memcpy(d_x,  hx.data(),  (size_t)M * K * 2, hipMemcpyHostToDevice);
    st_memcpy(d_gu, hgu.data(), (size_t)M * 2 * K * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w,  hw.data(),  (size_t)K * 4, hipMemcpyHostToDevice);

    const double eps = 1e-6, wadd = 1.0;

    auto same = [&](const char* what) {
        std::vector<uint16_t> y0((size_t)M * K), y1((size_t)M * K);
        std::vector<uint8_t>  q0((size_t)M * K), q1((size_t)M * K);
        std::vector<float>    s0((size_t)M * nkb), s1((size_t)M * nkb);
        st_memcpy(y0.data(), d_y[0], y0.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(y1.data(), d_y[1], y1.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(q0.data(), d_q[0], q0.size(), hipMemcpyDeviceToHost);
        st_memcpy(q1.data(), d_q[1], q1.size(), hipMemcpyDeviceToHost);
        st_memcpy(s0.data(), d_s[0], s0.size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(s1.data(), d_s[1], s1.size() * 4, hipMemcpyDeviceToHost);
        int ny = 0, nq = 0, ns = 0;
        for (size_t i = 0; i < y0.size(); ++i) ny += y0[i] != y1[i];
        for (size_t i = 0; i < q0.size(); ++i) nq += q0[i] != q1[i];
        for (size_t i = 0; i < s0.size(); ++i) ns += std::memcmp(&s0[i], &s1[i], 4) != 0;
        /* The codes must also be non-trivial: a run that wrote nothing would compare equal. */
        int nz = 0;
        for (size_t i = 0; i < q1.size(); ++i) nz += q1[i] != 0;
        ok(what, ny == 0 && nq == 0 && ns == 0 && nz > (int)q1.size() / 2,
           "bf16 differs " + std::to_string(ny) + ", codes " + std::to_string(nq) +
           ", scales " + std::to_string(ns) + ", nonzero codes " + std::to_string(nz));
    };

    /* --- rmsnorm + quant_act_fp8 --- */
    int rc = launch_row(nrm, { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
                               T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"),
                          PF("wadd", wadd) });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, {}),
                              T(d_w, RAD_F32, { K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128),
                          PS("dtype", "bf16"), PF("wadd", wadd) });
    if (rc != RAD_OK) ok("rmsnorm_quant_fp8 vs rmsnorm + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("rmsnorm_quant_fp8 is byte-identical to rmsnorm + quant_act_fp8");

    /* --- silu_mul + quant_act_fp8 --- */
    rc = launch_row(smu, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_y[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(gq, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_F8E4M3, { M, K }),
                              T(d_s[1], RAD_F32, { M, nkb }), T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                          PS("act", "silu") });
    if (rc != RAD_OK) ok("gated_quant_fp8 vs silu_mul + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("gated_quant_fp8 is byte-identical to silu_mul + quant_act_fp8");

    /* --- sigmoid + mul + quant_act_fp8, against gate_quant_fp8 ---
     *
     * The gated attention's epilogue. The gate is the first half of `gu` and the scaled value the
     * second; the engine's strided form (a column view of a [q|gate] projection) differs only in
     * the pitch the shim reads off the operand. */
    rc = launch_row(sg, { T(d_gu, RAD_BF16, { M, K }), T(d_y[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(ml, { T(d_y[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                              T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(tq, { T(d_gu, RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                          PS("act", "sigmoid") });
    if (rc != RAD_OK) ok("gate_quant_fp8 vs sigmoid + mul + quant_act_fp8", false,
                         "rc=" + std::to_string(rc));
    else same("gate_quant_fp8 is byte-identical to sigmoid + mul + quant_act_fp8");

    /* --- the same two at int8 codes, against the same products and quant_act_i8g ---
     *
     * With an I8 codes operand the producer writes int8 codes for an int8 trunk linear: the bf16
     * product must not move, and codes and scales must equal quant_act_i8g's over it. */
    if (const RadKernelInfo* qi = find(pl, "quant_act_i8g")) {
        rc = launch_row(smu, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(qi, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_I8, { M, K }),
                                  T(d_s[0], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(gq, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_I8, { M, K }),
                                  T(d_s[1], RAD_F32, { M, nkb }), T(d_y[1], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "silu") });
        if (rc != RAD_OK) ok("gated_quant_fp8 int8 vs silu_mul + quant_act_i8g", false,
                             "rc=" + std::to_string(rc));
        else same("gated_quant_fp8 at int8 codes is byte-identical to silu_mul + quant_act_i8g");

        rc = launch_row(sg, { T(d_gu, RAD_BF16, { M, K }), T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(ml, { T(d_y[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                                  T(d_y[0], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(qi, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_I8, { M, K }),
                                  T(d_s[0], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(tq, { T(d_gu, RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                                  T(d_q[1], RAD_I8, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                                  T(d_y[1], RAD_BF16, { M, K }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "sigmoid") });
        if (rc != RAD_OK) ok("gate_quant_fp8 int8 vs sigmoid + mul + quant_act_i8g", false,
                             "rc=" + std::to_string(rc));
        else same("gate_quant_fp8 at int8 codes is byte-identical to sigmoid + mul + quant_act_i8g");

        /* The rotated form pairs with E4M3 experts only and refuses int8 codes. */
        if (const RadKernelInfo* gh = find(pl, "gated_had_quant_fp8")) {
            rc = launch_row(gh, { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q[1], RAD_I8, { M, K }),
                                  T(d_s[1], RAD_F32, { M, nkb }) },
                            { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                              PS("act", "silu") });
            ok("gated_had_quant_fp8 refuses int8 codes", rc != RAD_OK, "rc=" + std::to_string(rc));
        }
    }

    /* --- add + rmsnorm + quant_act_fp8, against the same three fused ---
     *
     * The residual sum is f32, and the replaced pair rounds it to bf16 in `add`'s store before
     * `rmsnorm` reads it. Keeping the f32 sum (as the int8 sibling does) would move the text. */
    const RadKernelInfo* ad = find(pl, "add_bf16");
    if (!ad) { ok("rmsnorm_quant_fp8 residual", false, "add_bf16 absent"); return; }
    void* d_r[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
    if (!d_r[1]) { ok("fp8 fusion residual alloc", false, "hipMalloc failed"); return; }
    /* A structured residual, copied for each run: the fused form updates it in place. */
    std::vector<uint16_t> hr((size_t)M * K);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < K; ++c)
            hr[(size_t)r * K + c] = h_to_bf16(0.019f * (float)((r * 13 + c * 29) % 67) - 0.6f);
    st_memcpy(d_r[0], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_r[1], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);

    rc = launch_row(ad, { T(d_r[0], RAD_BF16, { M, K }), T(d_x, RAD_BF16, { M, K }),
                          T(d_r[0], RAD_BF16, { M, K }) },
                    { PI("M", M), PI("n", K), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(nrm, { T(d_r[0], RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
                               T(d_y[0], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"),
                          PF("wadd", wadd) });
    if (rc == RAD_OK)
        rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_F8E4M3, { M, K }),
                              T(d_s[0], RAD_F32, { M, nkb }) },
                        { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") });
    if (rc == RAD_OK)
        rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(d_r[1], RAD_BF16, { M, K }),
                              T(d_w, RAD_F32, { K }),
                              T(d_q[1], RAD_F8E4M3, { M, K }), T(d_s[1], RAD_F32, { M, nkb }),
                              T(d_y[1], RAD_BF16, { M, K }) },
                        { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128),
                          PS("dtype", "bf16"), PF("wadd", wadd) });
    if (rc != RAD_OK) {
        ok("rmsnorm_quant_fp8 residual vs add + rmsnorm + quant_act_fp8", false,
           "rc=" + std::to_string(rc));
        return;
    }
    same("rmsnorm_quant_fp8 residual is byte-identical to add + rmsnorm + quant_act_fp8");
    {
        std::vector<uint16_t> r0((size_t)M * K), r1((size_t)M * K);
        st_memcpy(r0.data(), d_r[0], r0.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(r1.data(), d_r[1], r1.size() * 2, hipMemcpyDeviceToHost);
        int nd = 0;
        for (size_t i = 0; i < r0.size(); ++i) nd += r0[i] != r1[i];
        ok("rmsnorm_quant_fp8 leaves the same residual behind", nd == 0,
           "differs " + std::to_string(nd));
    }
}

/* ---- gelu_mul_quant_i8, against gelu_mul + quant_act_i8 -------------------------------------
 * Gemma 4 31B's per-rank feed-forward width (10752 of 21504 at two ranks), both batch widths. */
static void gelu_mul_quant_i8_numeric(const Plugin& pl) {
    const RadKernelInfo* gm = find(pl, "gelu_mul_bf16");
    const RadKernelInfo* qz = find(pl, "quant_act_i8");
    const RadKernelInfo* fz = find(pl, "gelu_mul_quant_i8");
    if (!gm || !qz || !fz) { ok("gelu_mul_quant_i8 vs its pair", false, "rows absent"); return; }
    for (const int64_t M : { (int64_t)1, (int64_t)4, (int64_t)40 }) {
        const int64_t n = 10752;
        std::vector<uint16_t> hgu((size_t)M * 2 * n);
        for (size_t i = 0; i < hgu.size(); ++i)
            hgu[i] = h_to_bf16(std::ldexp(1.0f, (int)((i / 512) % 5) - 2) * (0.029f * (float)((i * 41) % 103) - 1.5f));
        void* dgu = dalloc(hgu.size() * 2);
        void* dy[2] = { dalloc((size_t)M * n * 2), dalloc((size_t)M * n * 2) };
        void* dq[2] = { dalloc((size_t)M * n), dalloc((size_t)M * n) };
        void* ds[2] = { dalloc((size_t)M * 4), dalloc((size_t)M * 4) };
        if (!dgu || !dy[1] || !dq[1] || !ds[1]) { ok("gelu_mul_quant_i8 alloc", false, "hipMalloc failed"); return; }
        st_memcpy(dgu, hgu.data(), hgu.size() * 2, hipMemcpyHostToDevice);
        int rc = launch_row(gm, { T(dgu, RAD_BF16, { M, 2 * n }), T(dy[0], RAD_BF16, { M, n }) },
                            { PI("M", M), PI("n", n), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(qz, { T(dy[0], RAD_BF16, { M, n }), T(dq[0], RAD_I8, { M, n }), T(ds[0], RAD_F32, { M }),
                                  T(nullptr, RAD_F32, {}) },
                            { PI("M", M), PI("n", n), PI("group", n), PS("dtype", "bf16") });
        if (rc == RAD_OK)
            rc = launch_row(fz, { T(dgu, RAD_BF16, { M, 2 * n }), T(dq[1], RAD_I8, { M, n }), T(ds[1], RAD_F32, { M }),
                                  T(dy[1], RAD_BF16, { M, n }) },
                            { PI("M", M), PI("n", n), PS("dtype", "bf16") });
        const std::string what = "gelu_mul_quant_i8 M=" + std::to_string(M);
        if (rc != RAD_OK) { ok((what + " vs gelu_mul + quant_act_i8").c_str(), false, "rc=" + std::to_string(rc)); continue; }
        std::vector<uint16_t> y0((size_t)M * n), y1((size_t)M * n);
        std::vector<uint8_t>  q0((size_t)M * n), q1((size_t)M * n);
        std::vector<float>    s0((size_t)M), s1((size_t)M);
        st_memcpy(y0.data(), dy[0], y0.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(y1.data(), dy[1], y1.size() * 2, hipMemcpyDeviceToHost);
        st_memcpy(q0.data(), dq[0], q0.size(), hipMemcpyDeviceToHost);
        st_memcpy(q1.data(), dq[1], q1.size(), hipMemcpyDeviceToHost);
        st_memcpy(s0.data(), ds[0], s0.size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(s1.data(), ds[1], s1.size() * 4, hipMemcpyDeviceToHost);
        int ny = 0, nq = 0, ns = 0, nz = 0;
        for (size_t i = 0; i < y0.size(); ++i) ny += y0[i] != y1[i];
        for (size_t i = 0; i < q0.size(); ++i) { nq += q0[i] != q1[i]; nz += q1[i] != 0; }
        for (size_t i = 0; i < s0.size(); ++i) ns += std::memcmp(&s0[i], &s1[i], 4) != 0;
        ok((what + " is byte-identical to gelu_mul + quant_act_i8").c_str(), ny == 0 && nq == 0 && ns == 0 && nz > (int)q1.size() / 2,
           "bf16 differs " + std::to_string(ny) + ", codes " + std::to_string(nq) + ", scales " + std::to_string(ns) +
           ", nonzero codes " + std::to_string(nz));
    }
}

/* ---- qkv_norm_rope, against the five ops it replaces ------------------------------------------
 *
 * rmsnorm_p (v, or the raw k under v_from_k), rmsnorm over the q heads and the k heads, then rope
 * over q and k, kernel by kernel, against the one dispatch. Both of Gemma 4 31B's classes at one
 * rank of two: sliding (head_dim 256, 16 q over 8 kv heads, full rotary, separate v) and global
 * (512, 16 over 2, the proportional `head` ladder over 128, k == v), at batch heights reaching
 * every width r4d_act_threads answers, including 256 at head_dim 512 (two channels per thread). */
static void qkv_norm_rope_numeric(const Plugin& pl) {
    const RadKernelInfo* np  = find(pl, "rmsnorm_p_bf16");
    const RadKernelInfo* nrm = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* rp  = find(pl, "rope_bf16");
    const RadKernelInfo* fz  = find(pl, "qkv_norm_rope_bf16");
    if (!np || !nrm || !rp || !fz) { ok("qkv_norm_rope vs its five", false, "rows absent"); return; }
    struct Cls { int64_t hd, nq, nkv, rot; const char* ladder; double theta; bool vk; };
    for (const Cls& cl : { Cls{ 256, 16, 8, 256, "rotary", 10000.0, false },
                           Cls{ 512, 16, 2, 128, "head", 1000000.0, true } }) {
        for (const int64_t M : { (int64_t)1, (int64_t)4, (int64_t)40 }) {
            const int64_t QD = cl.nq * cl.hd, KD = cl.nkv * cl.hd;
            std::vector<uint16_t> hq((size_t)M * QD), hk((size_t)M * KD), hv((size_t)M * KD);
            std::vector<uint16_t> hqw((size_t)cl.hd), hkw((size_t)cl.hd);
            std::vector<int32_t>  hp((size_t)M);
            for (size_t i = 0; i < hq.size(); ++i) hq[i] = h_to_bf16(0.031f * (float)((i * 37) % 97) - 1.4f);
            for (size_t i = 0; i < hk.size(); ++i) hk[i] = h_to_bf16(0.027f * (float)((i * 53) % 89) - 1.1f);
            for (size_t i = 0; i < hv.size(); ++i) hv[i] = h_to_bf16(0.023f * (float)((i * 29) % 83) - 0.9f);
            for (int64_t c = 0; c < cl.hd; ++c) {
                hqw[(size_t)c] = h_to_bf16(0.6f + 0.004f * (float)((c * 19) % 51));
                hkw[(size_t)c] = h_to_bf16(0.5f + 0.005f * (float)((c * 11) % 47));
            }
            for (int64_t m = 0; m < M; ++m) hp[(size_t)m] = (int32_t)(1000 + 7 * m);
            void* dq[2] = { dalloc(hq.size() * 2), dalloc(hq.size() * 2) };
            void* dk[2] = { dalloc(hk.size() * 2), dalloc(hk.size() * 2) };
            void* dv[2] = { dalloc(hv.size() * 2), dalloc(hv.size() * 2) };
            void* dqw = dalloc(hqw.size() * 2);
            void* dkw = dalloc(hkw.size() * 2);
            void* dp  = dalloc(hp.size() * 4);
            if (!dq[1] || !dk[1] || !dv[1] || !dqw || !dkw || !dp) { ok("qkv_norm_rope alloc", false, "hipMalloc failed"); return; }
            for (int j = 0; j < 2; ++j) {
                st_memcpy(dq[j], hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
                st_memcpy(dk[j], hk.data(), hk.size() * 2, hipMemcpyHostToDevice);
                st_memcpy(dv[j], hv.data(), hv.size() * 2, hipMemcpyHostToDevice);
            }
            st_memcpy(dqw, hqw.data(), hqw.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(dkw, hkw.data(), hkw.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(dp, hp.data(), hp.size() * 4, hipMemcpyHostToDevice);
            const double eps = 1e-6, wadd = 0.0;
            /* --- the five, in the order the unfused block issues them --- */
            int rc = launch_row(np, { T(cl.vk ? dk[0] : dv[0], RAD_BF16, { M * cl.nkv, cl.hd }), T(dv[0], RAD_BF16, { M * cl.nkv, cl.hd }) },
                                { PI("M", M * cl.nkv), PI("n", cl.hd), PF("eps", eps), PS("dtype", "bf16") });
            if (rc == RAD_OK)
                rc = launch_row(nrm, { T(dq[0], RAD_BF16, { M * cl.nq, cl.hd }), T(dqw, RAD_BF16, { cl.hd }), T(dq[0], RAD_BF16, { M * cl.nq, cl.hd }) },
                                { PI("M", M * cl.nq), PI("n", cl.hd), PF("eps", eps), PS("dtype", "bf16"), PF("wadd", wadd) });
            if (rc == RAD_OK)
                rc = launch_row(nrm, { T(dk[0], RAD_BF16, { M * cl.nkv, cl.hd }), T(dkw, RAD_BF16, { cl.hd }), T(dk[0], RAD_BF16, { M * cl.nkv, cl.hd }) },
                                { PI("M", M * cl.nkv), PI("n", cl.hd), PF("eps", eps), PS("dtype", "bf16"), PF("wadd", wadd) });
            for (int j = 0; j < 2 && rc == RAD_OK; ++j)
                rc = launch_row(rp, { T(j ? dk[0] : dq[0], RAD_BF16, { M, (j ? cl.nkv : cl.nq) * cl.hd }), T(dp, RAD_I32, { M }) },
                                { PI("M", M), PI("head_dim", cl.hd), PI("n_head", j ? cl.nkv : cl.nq), PI("n_head_kv", 0),
                                  PF("theta", cl.theta), PF("scale", 1.0), PS("mode", "neox"), PI("rotary_dim", cl.rot),
                                  PS("ladder", cl.ladder) });
            /* --- the one --- */
            if (rc == RAD_OK)
                rc = launch_row(fz, { T(dq[1], RAD_BF16, { M, QD }), T(dk[1], RAD_BF16, { M, KD }), T(dv[1], RAD_BF16, { M, KD }),
                                      T(dp, RAD_I32, { M }), T(dqw, RAD_BF16, { cl.hd }), T(dkw, RAD_BF16, { cl.hd }) },
                                { PI("M", M), PI("head_dim", cl.hd), PI("n_head", cl.nq), PI("n_head_kv", cl.nkv),
                                  PF("theta", cl.theta), PF("eps", eps), PF("scale", 1.0), PS("mode", "neox"),
                                  PI("rotary_dim", cl.rot), PS("ladder", cl.ladder), PF("wadd", wadd),
                                  PI("v_from_k", cl.vk ? 1 : 0), PS("dtype", "bf16") });
            const std::string what = std::string("qkv_norm_rope hd=") + std::to_string(cl.hd) + " M=" + std::to_string(M);
            if (rc != RAD_OK) { ok((what + " vs its five").c_str(), false, "rc=" + std::to_string(rc)); continue; }
            int nd = 0, moved = 0;
            for (int t = 0; t < 3; ++t) {
                void* const* d = t == 0 ? dq : (t == 1 ? dk : dv);
                const size_t n = t == 0 ? hq.size() : hk.size();
                std::vector<uint16_t> a(n), b(n);
                st_memcpy(a.data(), d[0], n * 2, hipMemcpyDeviceToHost);
                st_memcpy(b.data(), d[1], n * 2, hipMemcpyDeviceToHost);
                const std::vector<uint16_t>& h0 = t == 0 ? hq : (t == 1 ? hk : hv);
                for (size_t i = 0; i < n; ++i) { nd += a[i] != b[i]; moved += b[i] != h0[i]; }
            }
            ok((what + " is byte-identical to rmsnorm_p + rmsnorm x2 + rope x2").c_str(), nd == 0 && moved > 0,
               "differs " + std::to_string(nd) + ", changed from the input " + std::to_string(moved));
        }
    }
}

/* ---- rmsnorm_quant_i8, against the chain it replaces ------------------------------------------
 *
 * Byte-identical, as the fp8 fusions above, in the three forms a sandwich-norm model issues: the
 * plain norm + quantiser (a first block's entry), the junction without a scalar (attention ->
 * feed-forward) and with one (feed-forward -> next layer). The unfused chain, kernel by kernel, is
 * rmsnorm(x; w_post) -> add(residual) -> scale -> rmsnorm(residual; w) -> quant_act_i8. bf16 gains
 * (as Gemma's checkpoint stores), and two batch heights covering both r4d_act_threads widths a
 * decode step meets (1024 at M <= 32, 512 above). */
static void rmsnorm_quant_i8_numeric(const Plugin& pl) {
    const RadKernelInfo* nrm = find(pl, "rmsnorm_bf16");
    const RadKernelInfo* ad  = find(pl, "add_bf16");
    const RadKernelInfo* scl = find(pl, "scale_bf16");
    const RadKernelInfo* qz  = find(pl, "quant_act_i8");
    const RadKernelInfo* rq  = find(pl, "rmsnorm_quant_i8");
    if (!nrm || !ad || !scl || !qz || !rq) { ok("rmsnorm_quant_i8 vs its chain", false, "rows absent"); return; }
    for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)40 }) {
        const int64_t K = 5376;   /* Gemma 4 31B's hidden width: 336 sixteen-byte k steps */
        std::vector<uint16_t> hx((size_t)M * K), hr((size_t)M * K), hw((size_t)K), hp((size_t)K);
        for (int64_t r = 0; r < M; ++r)
            for (int64_t c = 0; c < K; ++c) {
                const float m = std::ldexp(1.0f, (int)((c / 384) % 6) - 3);
                const float v = m * (0.037f * (float)((r * 29 + c * 13) % 71) - 1.3f);
                hx[(size_t)r * K + c] = h_to_bf16(c == (r * 37) % K ? v * 19.0f : v);
                hr[(size_t)r * K + c] = h_to_bf16(0.019f * (float)((r * 13 + c * 29) % 67) - 0.6f);
            }
        for (int64_t c = 0; c < K; ++c) {
            hw[(size_t)c] = h_to_bf16(0.004f * (float)((c * 19) % 51) - 0.1f);
            hp[(size_t)c] = h_to_bf16(0.9f + 0.003f * (float)((c * 7) % 41));
        }
        const uint16_t hs = h_to_bf16(0.373f);
        void* d_x = dalloc((size_t)M * K * 2);
        void* d_w = dalloc((size_t)K * 2);
        void* d_p = dalloc((size_t)K * 2);
        void* d_sc = dalloc(4);
        void* d_t = dalloc((size_t)M * K * 2);
        void* d_r[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
        void* d_y[2] = { dalloc((size_t)M * K * 2), dalloc((size_t)M * K * 2) };
        void* d_q[2] = { dalloc((size_t)M * K), dalloc((size_t)M * K) };
        void* d_s[2] = { dalloc((size_t)M * 4), dalloc((size_t)M * 4) };
        if (!d_x || !d_w || !d_p || !d_sc || !d_t || !d_r[1] || !d_y[1] || !d_q[1] || !d_s[1]) {
            ok("rmsnorm_quant_i8 alloc", false, "hipMalloc failed");
            return;
        }
        st_memcpy(d_x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_p, hp.data(), hp.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_sc, &hs, 2, hipMemcpyHostToDevice);
        const double eps = 1e-6, wadd = 0.0;
        const std::vector<RadParam> pn = { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"), PF("wadd", wadd) };
        const std::vector<RadParam> pe = { PI("M", M), PI("n", K), PS("dtype", "bf16") };
        const std::vector<RadParam> pq = { PI("M", M), PI("n", K), PI("group", K), PS("dtype", "bf16") };

        auto compare = [&](const std::string& what, bool res) {
            std::vector<uint16_t> y0((size_t)M * K), y1((size_t)M * K), r0((size_t)M * K), r1((size_t)M * K);
            std::vector<uint8_t>  q0((size_t)M * K), q1((size_t)M * K);
            std::vector<float>    s0((size_t)M), s1((size_t)M);
            st_memcpy(y0.data(), d_y[0], y0.size() * 2, hipMemcpyDeviceToHost);
            st_memcpy(y1.data(), d_y[1], y1.size() * 2, hipMemcpyDeviceToHost);
            st_memcpy(q0.data(), d_q[0], q0.size(), hipMemcpyDeviceToHost);
            st_memcpy(q1.data(), d_q[1], q1.size(), hipMemcpyDeviceToHost);
            st_memcpy(s0.data(), d_s[0], s0.size() * 4, hipMemcpyDeviceToHost);
            st_memcpy(s1.data(), d_s[1], s1.size() * 4, hipMemcpyDeviceToHost);
            st_memcpy(r0.data(), d_r[0], r0.size() * 2, hipMemcpyDeviceToHost);
            st_memcpy(r1.data(), d_r[1], r1.size() * 2, hipMemcpyDeviceToHost);
            int ny = 0, nq = 0, ns = 0, nr = 0, nz = 0;
            for (size_t i = 0; i < y0.size(); ++i) ny += y0[i] != y1[i];
            for (size_t i = 0; i < q0.size(); ++i) nq += q0[i] != q1[i];
            for (size_t i = 0; i < s0.size(); ++i) ns += std::memcmp(&s0[i], &s1[i], 4) != 0;
            if (res) for (size_t i = 0; i < r0.size(); ++i) nr += r0[i] != r1[i];
            for (size_t i = 0; i < q1.size(); ++i) nz += q1[i] != 0;
            ok((what + " M=" + std::to_string(M)).c_str(), ny == 0 && nq == 0 && ns == 0 && nr == 0 && nz > (int)q1.size() / 2,
               "bf16 differs " + std::to_string(ny) + ", codes " + std::to_string(nq) + ", scales " +
               std::to_string(ns) + ", residual " + std::to_string(nr) + ", nonzero codes " + std::to_string(nz));
        };
        auto norm_quant = [&](void* src) {   /* rmsnorm(src; w) -> quant_act_i8, into slot 0 */
            int rc = launch_row(nrm, { T(src, RAD_BF16, { M, K }), T(d_w, RAD_BF16, { K }), T(d_y[0], RAD_BF16, { M, K }) }, pn);
            if (rc == RAD_OK)
                rc = launch_row(qz, { T(d_y[0], RAD_BF16, { M, K }), T(d_q[0], RAD_I8, { M, K }),
                                      T(d_s[0], RAD_F32, { M }), T(nullptr, RAD_F32, {}) }, pq);
            return rc;
        };

        /* --- the plain form --- */
        int rc = norm_quant(d_x);
        if (rc == RAD_OK)
            rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, {}), T(d_w, RAD_BF16, { K }),
                                  T(d_q[1], RAD_I8, { M, K }), T(d_s[1], RAD_F32, { M }), T(d_y[1], RAD_BF16, { M, K }),
                                  T(nullptr, RAD_BF16, {}), T(nullptr, RAD_BF16, {}) }, pn);
        if (rc != RAD_OK) ok("rmsnorm_quant_i8 vs rmsnorm + quant_act_i8", false, "rc=" + std::to_string(rc));
        else compare("rmsnorm_quant_i8 is byte-identical to rmsnorm + quant_act_i8", false);

        /* --- the junction, without and with the scalar --- */
        for (const bool with_sc : { false, true }) {
            st_memcpy(d_r[0], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);
            st_memcpy(d_r[1], hr.data(), hr.size() * 2, hipMemcpyHostToDevice);
            rc = launch_row(nrm, { T(d_x, RAD_BF16, { M, K }), T(d_p, RAD_BF16, { K }), T(d_t, RAD_BF16, { M, K }) }, pn);
            if (rc == RAD_OK)
                rc = launch_row(ad, { T(d_r[0], RAD_BF16, { M, K }), T(d_t, RAD_BF16, { M, K }), T(d_r[0], RAD_BF16, { M, K }) }, pe);
            if (rc == RAD_OK && with_sc)
                rc = launch_row(scl, { T(d_r[0], RAD_BF16, { M, K }), T(d_sc, RAD_BF16, { 1 }), T(d_r[0], RAD_BF16, { M, K }) }, pe);
            if (rc == RAD_OK) rc = norm_quant(d_r[0]);
            if (rc == RAD_OK)
                rc = launch_row(rq, { T(d_x, RAD_BF16, { M, K }), T(d_r[1], RAD_BF16, { M, K }), T(d_w, RAD_BF16, { K }),
                                      T(d_q[1], RAD_I8, { M, K }), T(d_s[1], RAD_F32, { M }), T(d_y[1], RAD_BF16, { M, K }),
                                      T(d_p, RAD_BF16, { K }),
                                      with_sc ? T(d_sc, RAD_BF16, { 1 }) : T(nullptr, RAD_BF16, {}) }, pn);
            const std::string what = with_sc ? "rmsnorm_quant_i8 junction with scalar" : "rmsnorm_quant_i8 junction";
            if (rc != RAD_OK) ok((what + " vs rmsnorm + add + scale + rmsnorm + quant_act_i8").c_str(), false, "rc=" + std::to_string(rc));
            else compare(what + " is byte-identical to its chain", true);
        }
    }
}

/* The fp8 lm_head, end to end through its own hooks.
 *
 * Unlike the linears' separate [sn][sk] plane, each stored row carries its own bf16 scales in its
 * tail, so this checks the interleave (a scale read from the wrong row), the f32 store (a bf16
 * write into the f32 plane) and the m_real mask on instantiations that round M up.
 *
 * The weight is block fp8's two planes, relaid by the row's hooks; the oracle is code times its
 * block's scale, and the inverse must return the planes. `i8` selects the int8 head
 * (logits_gemm_i8): i8 codes with a bf16 scale per row per 128 columns. */
static void fp8_logits_numeric(const Plugin& pl, int64_t N, int64_t K, int64_t M, bool i8 = false) {
    const char* row = i8 ? "logits_gemm_i8" : "logits_gemm_fp8";
    const RadKernelInfo* lg = find(pl, row);
    if (!lg || !lg->layout || !lg->relayout) {
        ok(i8 ? "i8 logits vs host reference" : "fp8 logits vs host reference", false,
           "row or hooks absent");
        return;
    }
    const int64_t stride = r4d_fp8::lm_row_bytes(K);
    const std::string tag = " [N=" + std::to_string(N) + " M=" + std::to_string(M) + "]";

    std::vector<float> wf((size_t)N * K), xf((size_t)M * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wf[(size_t)r * K + c] =
                std::ldexp(1.0f, (int)(((r / 128) % 5) * 3 - ((c / 128) % 4)))
                * (0.031f * (float)((r * 37 + c * 11) % 61) - 0.9f);
    std::vector<uint16_t> xb((size_t)M * K);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t c = 0; c < K; ++c) {
            const float v = 0.017f * (float)((m * 53 + c * 7) % 83) - 0.7f;
            xb[(size_t)m * K + c] = h_to_bf16(v);
            xf[(size_t)m * K + c] = h_bf16(xb[(size_t)m * K + c]);
        }

    Planes p;
    if (!i8) {
        p = fp8_block_planes(wf, N, K);
    } else {
        p.enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        p.size(N, K);
        const int64_t nkb = K / 128;
        for (int64_t r = 0; r < N; ++r)
            for (int64_t gi = 0; gi < nkb; ++gi) {
                float amax = 0.0f;
                for (int64_t j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(wf[(size_t)(r * K + gi * 128 + j)]));
                rad_store_f32(p.data[1].data(), RAD_BF16, r * nkb + gi, amax > 0 ? amax / 127.0f : 1.0f);
                const float sc = rad_load_f32(p.data[1].data(), RAD_BF16, r * nkb + gi);
                for (int64_t j = 0; j < 128; ++j) {
                    const float q = std::nearbyint(wf[(size_t)(r * K + gi * 128 + j)] / sc);
                    p.data[0][(size_t)(r * K + gi * 128 + j)] =
                        (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
                }
            }
    }
    std::vector<RadParam> lp = { PI("n_vocab", N), PI("n_embd", K) };
    RadLayout lay;
    std::vector<uint8_t> wq;
    const int lrc = stored(lg, lp, 1, p, { "codes", "scale" }, &wq, &lay);
    if (lrc != RAD_OK) {
        ok(i8 ? "i8 logits relayout" : "fp8 logits relayout", false, "rc=" + std::to_string(lrc) + tag);
        return;
    }
    ok(i8 ? "i8 logits layout is codes plus the row's scale tail" : "fp8 logits layout is codes plus a per-row scale tail",
       lay.bytes == N * stride && lay.shape[0] == N && lay.shape[1] == stride &&
       (int64_t)wq.size() == N * stride,
       "bytes " + std::to_string(lay.bytes) + " want " + std::to_string(N * stride) + tag);
    {
        std::string why;
        ok(i8 ? "i8 logits relayout inverts to the planes it was handed" : "fp8 logits relayout inverts to the planes it was handed",
           restores(lg, lp, 1, p, { "codes", "scale" }, wq, &why), why + tag);
    }

    /* The oracle: the planes, code times the scale of its 128x128 block. */
    const int64_t sk = p.cols[1];
    std::vector<double> ref((size_t)M * N);
    for (int64_t r = 0; r < N; ++r) {
        const uint8_t* row = p.data[0].data() + (size_t)r * K;
        const uint16_t* sc = (const uint16_t*)p.data[1].data() + (i8 ? r : r / 128) * sk;
        for (int64_t m = 0; m < M; ++m) {
            double acc = 0;
            for (int64_t c = 0; c < K; ++c)
                acc += (i8 ? (double)(int8_t)row[c] : (double)h_e4m3(row[c])) *
                       (double)h_bf16(sc[c / 128]) * xf[(size_t)m * K + c];
            ref[(size_t)m * N + r] = acc;
        }
    }

    void* d_w = dalloc(wq.size());
    void* d_x = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 4);
    if (!d_w || !d_x || !d_y) { ok(i8 ? "i8 logits alloc" : "fp8 logits alloc", false, std::string("hipMalloc failed") + tag); return; }
    st_memcpy(d_w, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
    /* Poisoned, so a kernel writing bf16 pairs into the f32 plane cannot pass on zeroed high
     * halves. */
    st_memset(d_y, 0x5a, (size_t)M * N * 4);

    // The declared shape [N][K], not the stored [N][K + 2*nkb]: core/runtime/ctx.cpp builds a
    // weight's tensor from decl.shape with the layout hook's dtype.
    const int rc = launch_row(lg, { T(d_x, RAD_BF16, { M, K }),
                                    T(d_w, i8 ? RAD_I8 : RAD_F8E4M3, { N, K }),
                                    T(d_y, RAD_F32, { M, N }) },
                              { PI("M", M), PI("n_vocab", N), PI("n_embd", K),
                                PS("dtype", "bf16") });
    if (rc != RAD_OK) { ok(i8 ? "i8 logits launch" : "fp8 logits launch", false, "rc=" + std::to_string(rc) + tag); return; }

    std::vector<float> got((size_t)M * N);
    st_memcpy(got.data(), d_y, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    const double e = den > 0 ? std::sqrt(num / den) : 1.0;
    ok(i8 ? "logits_gemm_i8 vs host ref over its planes" : "logits_gemm_fp8 vs host ref over its planes",
       e < 2e-3, "rel_l2 " + std::to_string(e) + tag);

    /* The drafter's rescoring reads the same stored head, so it is checked over the same bytes:
     * R candidates per row (one an id past the shard, which must come back a pad), each scored
     * against the planes, sorted by value. */
    const RadKernelInfo* rr = find(pl, i8 ? "logit_rerank_i8" : "logit_rerank");
    if (!rr) return;
    const int64_t R = std::min<int64_t>(8, N);
    std::vector<int32_t> cand((size_t)(M * R));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t r = 0; r < R; ++r)
            cand[(size_t)(m * R + r)] = r == R - 1 ? (int32_t)N : (int32_t)((m * 131 + r * 977) % N);
    void* d_in = dalloc(cand.size() * 4);
    void* d_oi = dalloc(cand.size() * 4);
    void* d_ov = dalloc(cand.size() * 4);
    const std::string rn = std::string(i8 ? "logit_rerank_i8" : "logit_rerank") +
                           " vs host ref over the stored head" + tag;
    if (!d_in || !d_oi || !d_ov) { ok(rn.c_str(), false, "hipMalloc failed"); return; }
    st_memcpy(d_in, cand.data(), cand.size() * 4, hipMemcpyHostToDevice);
    const int rrc = launch_row(rr, { T(d_x, RAD_BF16, { M, K }),
                                     T(d_w, i8 ? RAD_I8 : RAD_F8E4M3, { N, K }),
                                     T(d_in, RAD_I32, { M, R }), T(d_oi, RAD_I32, { M, R }),
                                     T(d_ov, RAD_F32, { M, R }) },
                               { PI("M", M), PI("R", R), PI("n_vocab", N), PI("n_embd", K),
                                 PI("vocab_off", 0), PS("dtype", "bf16") });
    if (rrc != RAD_OK) { ok(rn.c_str(), false, "rc=" + std::to_string(rrc)); return; }
    std::vector<int32_t> oi(cand.size());
    std::vector<float> ov(cand.size());
    st_memcpy(oi.data(), d_oi, oi.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(ov.data(), d_ov, ov.size() * 4, hipMemcpyDeviceToHost);
    int64_t bad = 0;
    for (int64_t m = 0; m < M; ++m) {
        double amax = 0;
        for (int64_t n = 0; n < N; ++n) amax = std::max(amax, std::fabs(ref[(size_t)(m * N + n)]));
        for (int64_t r = 0; r < R; ++r) {
            const int32_t id = oi[(size_t)(m * R + r)];
            const float v = ov[(size_t)(m * R + r)];
            if (r == R - 1) { bad += id != -1; continue; }      /* the pad sorts last */
            const bool known = std::find(cand.begin() + m * R, cand.begin() + m * R + R - 1, id) !=
                               cand.begin() + m * R + R - 1;
            bad += !known || std::fabs((double)v - ref[(size_t)(m * N + id)]) > 1e-3 * amax + 1e-4;
            if (r > 0) bad += v > ov[(size_t)(m * R + r - 1)];
        }
    }
    ok(rn.c_str(), bad == 0, std::to_string(bad) + " wrong ids, values or orderings");
}

/* The fp8 GEMM against a host reference on the same bytes (spec §17). A weight's two block-fp8
 * planes are arranged by the plugin's own layout hooks, the planes are dequantised on the host,
 * and the results compared. This catches a wrong arrangement, wrong scale indexing (an off-by-one
 * `>> 7` reads a neighbouring tile), a misplaced fold in the K loop and, for fp8a8, a wrong
 * activation quantiser or epilogue transpose.
 *
 * Both sides use the same quantised weight, so only summation order may differ. The bound is
 * bf16's resolution (2^-8 relative) loosened to 3e-2, since a K=512 reduction reordered across 32
 * lanes moves the last bf16 bit near a rounding boundary; a transposed operand or misread scale
 * lands two orders of magnitude away. The W8A8 rows also quantise the activation to E4M3 (about 6%
 * per element before averaging), so their error is higher. The two W8A8 rows put the weight on
 * opposite operands with opposite scale shapes and epilogues, so their failure modes differ. */
static void fp8_numeric(const Plugin& pl, int64_t N, int64_t K) {
    const RadKernelInfo* mv = find(pl, "gemm_fp8a16_nt_m1");
    const RadKernelInfo* tl = find(pl, "gemm_fp8a8_tiled");
    const RadKernelInfo* qz = find(pl, "quant_act_fp8");
    const RadKernelInfo* n16 = find(pl, "gemm_fp8a8_nt_m16");
    if (!mv || !tl || !qz || !n16 || !mv->layout || !n16->layout || !n16->relayout) {
        ok("fp8 numeric vs host reference", false, "rows or hooks absent");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;

    /* A weight with an asymmetric, per-block-varying magnitude; a constant or symmetric pattern
     * cannot distinguish a transposed fragment or a neighbouring block's scale.
     *
     * The exponent wraps: unwrapped, `(r/128)*3 - (c/128)` reaches 405 by row 17408 and
     * ldexp(1.0f, 405) is infinity. Wrapped, neighbouring 128-row blocks still differ by 8x and the
     * range stays within 2^12. */
    std::vector<float> wf((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wf[(size_t)r * K + c] =
                std::ldexp(1.0f, (int)(((r / 128) % 5) * 3 - ((c / 128) % 4)))
                * (0.031f * (float)((r * 37 + c * 11) % 61) - 0.9f);

    /* Two arrangements of one pair of planes: the fp8a16 matvec reads them row-major, the fp8a8
     * GEMMs read the codes in WMMA fragment order, each as its own layout hook says. One oracle
     * serves all: the planes dequantised on the host. */
    const Planes p = fp8_block_planes(wf, N, K);
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 128) };
    std::vector<uint8_t> wq, wqf, wsb, wsf;
    RadLayout flay;
    const int rw = stored(mv, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(mv, lp, 3, p, { "scale" }, &wsb);
    const int rf = stored(n16, lp, 2, p, { "codes" }, &wqf, &flay);
    const int rfs = stored(n16, lp, 3, p, { "scale" }, &wsf);
    ok("fp8 matvec reads both planes as they are",
       rw == RAD_E_UNSUPPORTED && rs == RAD_E_UNSUPPORTED,
       "codes rc " + std::to_string(rw) + ", scale rc " + std::to_string(rs));
    ok("fp8 fragment order is the codes rearranged, and the scales as they are",
       rf == RAD_OK && flay.bytes == N * K && (int64_t)wqf.size() == N * K &&
       rfs == RAD_E_UNSUPPORTED && wsf == wsb,
       "codes rc " + std::to_string(rf) + " bytes " + std::to_string(flay.bytes) +
       ", scale rc " + std::to_string(rfs));
    if (rw != RAD_E_UNSUPPORTED || rs != RAD_E_UNSUPPORTED || rf != RAD_OK) return;
    {
        std::string why;
        ok("fp8 fragment order inverts to the codes", restores(n16, lp, 2, p, { "codes" }, wqf, &why),
           why);
    }
    std::vector<uint16_t> ws((size_t)sn * sk);
    std::memcpy(ws.data(), wsb.data(), ws.size() * 2);

    std::vector<float> wd((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wd[(size_t)r * K + c] =
                h_e4m3(wq[(size_t)r * K + c]) * h_bf16(ws[(size_t)(r / 128) * sk + (c / 128)]);

    std::vector<uint16_t> xb((size_t)K);
    for (int64_t c = 0; c < K; ++c)
        xb[(size_t)c] = h_to_bf16(0.021f * (float)((c * 17) % 53) - 0.55f);
    std::vector<float> ref((size_t)N, 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        double acc = 0;
        for (int64_t c = 0; c < K; ++c) acc += (double)wd[(size_t)r * K + c] * h_bf16(xb[(size_t)c]);
        ref[(size_t)r] = (float)acc;
    }

    void* d_w = dalloc((size_t)N * K);
    void* d_wf = dalloc((size_t)N * K);   /* the same weight in fragment order, for the fp8a8 pair */
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_x = dalloc((size_t)K * 2);
    void* d_y = dalloc((size_t)N * 2);
    if (!d_w || !d_wf || !d_s || !d_x || !d_y) { ok("fp8 numeric alloc", false, "hipMalloc failed"); return; }
    st_memcpy(d_wf, wqf.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_w, wq.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_s, ws.data(), (size_t)sn * sk * 2, hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)K * 2, hipMemcpyHostToDevice);

    auto rel = [&](const std::vector<uint16_t>& got) {
        double num = 0, den = 0;
        for (int64_t r = 0; r < N; ++r) {
            const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)r];
            num += d * d;
            den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
        }
        return den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    };

    /* --- the matvec, at every point of its declared space: `rb` changes which rows share an
     *     activation window, and a wrong RB reads the neighbouring row's scale. */
    for (const TunePoint& tp : tune_points(mv, std::vector<RadParam>{ PI("M", 1), PI("N", N),
                                                                     PI("K", K), PI("group", 128),
                                                                     PS("dtype", "fp8a16") })) {
        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { 1, K }), T(nullptr, RAD_F32, { 1, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { 1, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[1]); absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = with_point(std::vector<RadParam>{
            PI("M", 1), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
        }, tp);
        st_memset(d_y, 0, (size_t)N * 2);
        run_case(mv, ts, ps);
        std::vector<uint16_t> got((size_t)N);
        st_memcpy(got.data(), d_y, (size_t)N * 2, hipMemcpyDeviceToHost);
        const double e = rel(got);
        const std::string what =
            std::string("gemm_fp8a16_nt_m1 vs host ref, ") + tp.label;
        ok(what.c_str(), e < 3e-2, "rel_l2 " + std::to_string(e));
    }

    /* --- the multi-row matvec, as a speculative verify step runs it: M = 1 + n_spec token rows
     *     against one weight read. The rows are distinct, each with its own host reference, to
     *     catch a row reading another's activation (a wrong `a_ld`, a hoisted pointer). M = 5
     *     (n_spec 4) is not a power of two, so a token count rounded up to an instantiation
     *     shows. */
    {
        const int64_t M = 5;
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_ym = dalloc((size_t)M * N * 2);
        if (!d_xm || !d_ym) { ok("fp8a16 multi-row alloc", false, "hipMalloc failed"); }
        else {
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c)
                    xm[(size_t)m * K + c] = xb[(size_t)((c + m * 37) % K)];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);

            std::vector<double> refm((size_t)M * N);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t r = 0; r < N; ++r) {
                    double acc = 0;
                    for (int64_t c = 0; c < K; ++c)
                        acc += (double)wd[(size_t)r * K + c] * h_bf16(xm[(size_t)m * K + c]);
                    refm[(size_t)m * N + r] = acc;
                }

            std::vector<RadTensor> ts = {
                T(d_xm, RAD_BF16, { M, K }), T(nullptr, RAD_F32, { M, sk }),
                T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
                T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[1]); absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
            };
            st_memset(d_ym, 0, (size_t)M * N * 2);
            run_case(mv, ts, ps);
            std::vector<uint16_t> got((size_t)M * N);
            st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
            double worst = 0;
            for (int64_t m = 0; m < M; ++m) {
                double num = 0, den = 0;
                for (int64_t r = 0; r < N; ++r) {
                    const double d =
                        (double)h_bf16(got[(size_t)m * N + r]) - refm[(size_t)m * N + r];
                    num += d * d;
                    den += refm[(size_t)m * N + r] * refm[(size_t)m * N + r];
                }
                const double er = den > 0 ? std::sqrt(num / den) : 1.0;
                if (er > worst) worst = er;
            }
            ok("gemm_fp8a16_nt_mt vs host ref, worst of 5 DISTINCT token rows",
               worst < 3e-2, "rel_l2 " + std::to_string(worst));
        }
    }

    /* --- the narrow W8A8 row, where two hazards meet: the epilogue transposes (the weight is the
     *     A operand, so D is [N][M] and C must be [M][N]), and split-K is on (N=256, K=512 gives
     *     ksplit=2, exercising the partial plane and the reduce). Ragged M (not a multiple of the
     *     16-wide token tile) is what a speculative batch runs. Every row must match, since a
     *     wrong transposed stride writes one right row. */
    {
        /* Every token-tile count and both sides of `Full`. The narrow tile covers NT * 16 tokens
         * per weight read, NT chosen from M: 1 and 8 give NT = 1, 24 and 32 NT = 2, 40 and 48
         * NT = 3, 64 NT = 4.
         *
         * `Full` drops the epilogue's column guard and means "all NT tiles are live", not "M is a
         * multiple of 16": at M = 48 with NT = 4 the second holds and the first does not, and a
         * kernel conflating them writes columns 48..63 past a split-K scratch sized for 48, over
         * the arrival counter, and the finisher hangs. 24 and 48 cover that; 40 is ragged under
         * NT = 3 as well. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)24, (int64_t)32, (int64_t)40,
                                 (int64_t)48, (int64_t)64 }) {
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_ym = dalloc((size_t)M * N * 2);
        const RadKernelInfo* nr = find(pl, "gemm_fp8a8_nt_m16");
        if (!nr) { ok("gemm_fp8a8_nt_m16 numeric", false, "row absent"); }
        else if (!d_xm || !d_q || !d_qs || !d_ym) { ok("fp8 narrow alloc", false, "hipMalloc failed"); }
        else {
            /* Distinct rows, each with its own reference, so a token tile reading another tile's
             * activation shows. */
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c)
                    xm[(size_t)m * K + c] = xb[(size_t)((c + m * 37) % K)];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
            std::vector<double> refn((size_t)M * N);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t r = 0; r < N; ++r) {
                    double acc = 0;
                    for (int64_t c = 0; c < K; ++c)
                        acc += (double)wd[(size_t)r * K + c] * h_bf16(xm[(size_t)m * K + c]);
                    refn[(size_t)m * N + r] = acc;
                }
            std::vector<RadTensor> qt = {
                T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
                T(d_qs, RAD_F32, { M, sk }),
            };
            std::vector<RadParam> qp = {
                PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
            };
            run_case(qz, qt, qp);

            std::vector<RadTensor> ts = {
                T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
                T(d_wf, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
                T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
            };
            st_memset(d_ym, 0, (size_t)M * N * 2);
            run_case(nr, ts, ps);
            std::vector<uint16_t> got((size_t)M * N);
            st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
            double worst = 0;
            for (int64_t m = 0; m < M; ++m) {
                double num = 0, den = 0;
                for (int64_t r = 0; r < N; ++r) {
                    const double d =
                        (double)h_bf16(got[(size_t)m * N + r]) - refn[(size_t)m * N + r];
                    num += d * d;
                    den += refn[(size_t)m * N + r] * refn[(size_t)m * N + r];
                }
                const double er = den > 0 ? std::sqrt(num / den) : 1.0;
                if (er > worst) worst = er;
            }
            ok((std::string("gemm_fp8a8_nt_m16 vs host ref, ") + std::to_string((long long)M) +
                " DISTINCT rows (transposed store + split-K)").c_str(),
               worst < 8e-2, "rel_l2 " + std::to_string(worst));

            /* The wave count must not move a byte. MW is a launch heuristic chosen from (N, M)
             * inside libr4d, pinned here through the test hook; a rel_l2 check would pass a
             * last-bit change that moves the emitted text.
             *
             * A different grid may differ: `narrow_ksplit` decides from `gx * mw`, which is N/16
             * only when N divides MW*16; otherwise the split repartitions K. So only the wave
             * counts whose grid the shape admits are compared (the rule's own `n % 128` and
             * `n % 256` test). */
            if (pl.pin_mw) {
                const std::vector<uint16_t> base = got;
                std::vector<uint16_t> alt((size_t)M * N);
                std::string why;
                for (const int mw : { 1, 2, 4, 8, 16 }) {
                    if (N % (mw * 16)) continue;
                    /* A K not a multiple of 256 runs the 128-wide staged set, which compiles
                     * only the wave counts the rule answers (4, 8, 16); other pins are refused
                     * (R4D_F8D_NT128 in r4d_gemm_fp8a8.hip). */
                    if (K % 256 && mw != 4 && mw != 8 && mw != 16) continue;
                    pl.pin_mw(mw);
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    st_memcpy(alt.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                    if (alt != base) { why = "mw " + std::to_string(mw); break; }
                }
                pl.pin_mw(0);
                ok((std::string("gemm_fp8a8_nt_m16 is byte-identical at every wave count its "
                                "grid admits, M ") + std::to_string((long long)M)).c_str(),
                   why.empty(), why);
            }

            /* Nor may the token-tile count. NT also feeds `Full`: at M = 48, NT = 3 makes Full
             * true and NT = 4 false, so the pin sweeps two epilogue paths. The pin only goes up:
             * NT*16 is the column extent the epilogue may write, so NT below ceil(M/16) would
             * drop rows. */
            if (pl.pin_nt) {
                const std::vector<uint16_t> base = got;
                std::vector<uint16_t> alt((size_t)M * N);
                std::string why;
                for (int nt = 1; nt <= 4; ++nt) {
                    if ((int64_t)nt * 16 < M) continue;
                    pl.pin_nt(nt);
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    st_memcpy(alt.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                    if (alt != base) { why = "nt " + std::to_string(nt); break; }
                }
                pl.pin_nt(0);
                ok((std::string("gemm_fp8a8_nt_m16 is byte-identical at every token-tile count "
                                "that covers M, M ") + std::to_string((long long)M)).c_str(),
                   why.empty(), why);
            }

            /* ---- ...and with the gate and the quantiser folded into its epilogue -------------
             *
             * Byte equality: the engine substitutes the fold for the pair (this GEMM's bf16
             * output read back by `gated_quant_fp8`), so any last-bit difference would be a second
             * numerical path. Over the same M list and N = 256, 5120 and 17408 (gate_up's
             * per-rank width). */
            const RadKernelInfo* gq = find(pl, "gated_quant_fp8");
            const RadKernelInfo* gg = find(pl, "gemm_fp8a8_gated_nt_m16");
            /* The fold stages K 256 wide and refuses other K; the engine's one K % 256 shape,
             * the shared expert's down projection, is never gated. */
            if (gq && gg && N % 256 == 0 && K % 256 == 0) {
                const int64_t nout = N / 2, gkb = nout / 128;
                /* The fold refuses a shape the split rule would split (it has no partial plane):
                 * `narrow_ksplit` splits when `K >= 512` and `N/16 < 128`, so N = 256 must be
                 * refused. Asserted, since serving it while ignoring the split would be wrong. */
                const bool splits = K >= 512 && N < 2048;
                void* d_gq[2] = { dalloc((size_t)M * nout),     dalloc((size_t)M * nout) };
                void* d_gs[2] = { dalloc((size_t)M * gkb * 4),  dalloc((size_t)M * gkb * 4) };
                void* d_gy[2] = { dalloc((size_t)M * nout * 2), dalloc((size_t)M * nout * 2) };
                if (!d_gq[1] || !d_gs[1] || !d_gy[1]) {
                    ok("gated fold alloc", false, "hipMalloc failed");
                } else {
                    /* Re-run the GEMM: d_ym holds the last pinned wave count's output. */
                    st_memset(d_ym, 0, (size_t)M * N * 2);
                    run_case(nr, ts, ps);
                    int rc = launch_row(gq, { T(d_ym, RAD_BF16, { M, N }),
                                              T(d_gq[0], RAD_F8E4M3, { M, nout }),
                                              T(d_gs[0], RAD_F32, { M, gkb }),
                                              T(d_gy[0], RAD_BF16, { M, nout }) },
                                        { PI("M", M), PI("n", nout), PI("group", 128),
                                          PS("dtype", "bf16"), PS("act", "silu") });
                    if (rc == RAD_OK) {
                        st_memset(d_gq[1], 0, (size_t)M * nout);
                        st_memset(d_gs[1], 0, (size_t)M * gkb * 4);
                        st_memset(d_gy[1], 0, (size_t)M * nout * 2);
                        rc = launch_row(gg, { T(d_q, RAD_F8E4M3, { M, K }),
                                              T(d_qs, RAD_F32, { M, sk }),
                                              T(d_wf, RAD_F8E4M3, { N, K }),
                                              T(d_s, RAD_BF16, { sn, sk }),
                                              T(d_gq[1], RAD_F8E4M3, { M, nout }),
                                              T(d_gs[1], RAD_F32, { M, gkb }),
                                              T(d_gy[1], RAD_BF16, { M, nout }) },
                                        { PI("M", M), PI("N", N), PI("K", K), PI("group", 128),
                                          PS("dtype", "fp8a8"), PS("act", "silu") });
                    }
                    if (splits) {
                        ok((std::string("gemm_fp8a8_gated_nt_m16 refuses a grid the split rule "
                                        "would widen, N ") + std::to_string((long long)N) +
                            " M " + std::to_string((long long)M)).c_str(),
                           rc != RAD_OK, "rc=" + std::to_string(rc));
                    } else if (rc != RAD_OK) {
                        ok("gemm_fp8a8_gated_nt_m16 vs the GEMM + gated_quant_fp8", false,
                           "rc=" + std::to_string(rc));
                    } else {
                        std::vector<uint8_t>  c0((size_t)M * nout), c1((size_t)M * nout);
                        std::vector<float>    z0((size_t)M * gkb), z1((size_t)M * gkb);
                        std::vector<uint16_t> b0((size_t)M * nout), b1((size_t)M * nout);
                        st_memcpy(c0.data(), d_gq[0], c0.size(), hipMemcpyDeviceToHost);
                        st_memcpy(c1.data(), d_gq[1], c1.size(), hipMemcpyDeviceToHost);
                        st_memcpy(z0.data(), d_gs[0], z0.size() * 4, hipMemcpyDeviceToHost);
                        st_memcpy(z1.data(), d_gs[1], z1.size() * 4, hipMemcpyDeviceToHost);
                        st_memcpy(b0.data(), d_gy[0], b0.size() * 2, hipMemcpyDeviceToHost);
                        st_memcpy(b1.data(), d_gy[1], b1.size() * 2, hipMemcpyDeviceToHost);
                        int nc = 0, nz = 0, nb = 0, live = 0;
                        for (size_t i = 0; i < c0.size(); ++i) nc += c0[i] != c1[i];
                        for (size_t i = 0; i < z0.size(); ++i)
                            nz += std::memcmp(&z0[i], &z1[i], 4) != 0;
                        for (size_t i = 0; i < b0.size(); ++i) nb += b0[i] != b1[i];
                        /* The codes must also be non-trivial: zeroed buffers compare equal. */
                        for (size_t i = 0; i < c1.size(); ++i) live += c1[i] != 0;
                        ok((std::string("gemm_fp8a8_gated_nt_m16 is byte-identical to the GEMM + "
                                        "gated_quant_fp8, N ") + std::to_string((long long)N) +
                            " M " + std::to_string((long long)M)).c_str(),
                           nc == 0 && nz == 0 && nb == 0 && live > (int)c1.size() / 2,
                           "codes differ " + std::to_string(nc) + ", scales " +
                           std::to_string(nz) + ", bf16 " + std::to_string(nb) + ", nonzero " +
                           std::to_string(live));
                    }
                }
            }
        }
        }
    }

    /* --- a row shard, as the loader makes one: rank 1's share of each canonical plane (rows
     *     [N/2, N) of the codes and the scale rows covering them, cut on 128-row blocks), relaid
     *     by the row's hook as an N/2-row weight, must reproduce rows [N/2, N) of the full run.
     *     The share is cut before the arrangement, so a relayout reading rows it was not handed
     *     shows. */
    if (N % 32 == 0 && (N / 2) % 128 == 0) {
        const int64_t Nh = N / 2, snh = Nh / 128, M = 8;
        const Planes half = share(p, Nh, N, 0, K);
        std::vector<RadParam> hp = { PI("N", Nh), PI("K", K), PI("group", 128) };
        std::vector<uint8_t> hw, hs;
        const int hrc = stored(n16, hp, 2, half, { "codes" }, &hw);
        const int hsc = stored(n16, hp, 3, half, { "scale" }, &hs);
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_yh = dalloc((size_t)M * Nh * 2);
        void* d_hw = dalloc((size_t)Nh * K);
        void* d_hs = dalloc((size_t)snh * sk * 2);
        const RadKernelInfo* nr = n16;
        if (hrc != RAD_OK || hsc != RAD_E_UNSUPPORTED) {
            ok("fragment order survives a row shard at N/2", false,
               "relayout of the share rc " + std::to_string(hrc) + "/" + std::to_string(hsc));
        } else if (!d_xm || !d_q || !d_qs || !d_yh || !d_hw || !d_hs) {
            ok("fp8 shard alloc", false, "missing");
        } else {
            st_memcpy(d_hw, hw.data(), hw.size(), hipMemcpyHostToDevice);
            st_memcpy(d_hs, hs.data(), hs.size(), hipMemcpyHostToDevice);
            std::vector<uint16_t> xm((size_t)M * K);
            for (int64_t m = 0; m < M; ++m)
                for (int64_t c = 0; c < K; ++c) xm[(size_t)m * K + c] = xb[(size_t)c];
            st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);
            std::vector<RadTensor> qt = {
                T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
                T(d_qs, RAD_F32, { M, sk }),
            };
            std::vector<RadParam> qp = {
                PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
            };
            run_case(qz, qt, qp);
            /* Rank 1's share, relaid. */
            std::vector<RadTensor> ts = {
                T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
                T(d_hw, RAD_F8E4M3, { Nh, K }),
                T(d_hs, RAD_BF16, { snh, sk }),
                T(d_yh, RAD_BF16, { M, Nh }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
            };
            absent(ts[5]); absent(ts[6]);
            std::vector<RadParam> ps = {
                PI("M", M), PI("N", Nh), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
            };
            st_memset(d_yh, 0, (size_t)M * Nh * 2);
            run_case(nr, ts, ps);
            std::vector<uint16_t> got((size_t)M * Nh);
            st_memcpy(got.data(), d_yh, (size_t)M * Nh * 2, hipMemcpyDeviceToHost);
            double num = 0, den = 0;
            for (int64_t r = 0; r < Nh; ++r) {
                const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)(Nh + r)];
                num += d * d;
                den += (double)ref[(size_t)(Nh + r)] * (double)ref[(size_t)(Nh + r)];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            ok("fragment order survives a row shard at N/2", e < 8e-2,
               "rel_l2 " + std::to_string(e));
        }
    }

    /* --- a column shard, where a fragment plane and a row-major one stop agreeing.
     *
     * The loader cuts the canonical planes (a column share is a plain rectangle of codes and
     * scales) and the hook arranges each share as a K/2 weight, rather than slicing the arranged
     * plane (a pitch-K*16 copy). Both shares run as independent weights; their sum, as the
     * all-reduce forms it, must equal the unsharded product. */
    if (K % 256 == 0) {
        const int64_t Kh = K / 2, skh = Kh / 128, M = 8;
        void* d_xh = dalloc((size_t)M * Kh * 2);
        void* d_q  = dalloc((size_t)M * Kh);
        void* d_qs = dalloc((size_t)M * skh * 4);
        void* d_wh = dalloc((size_t)N * Kh);
        void* d_sh = dalloc((size_t)sn * skh * 2);
        void* d_yh = dalloc((size_t)M * N * 2);
        const RadKernelInfo* nr = n16;
        if (!d_xh || !d_q || !d_qs || !d_wh || !d_sh || !d_yh) {
            ok("fp8 col-shard alloc", false, "missing");
        } else {
            std::vector<double> acc((size_t)N, 0.0);
            bool laid = true;
            for (int rank = 0; rank < 2 && laid; ++rank) {
                /* This rank's share of the canonical planes, arranged by the row's own hook. */
                const Planes part = share(p, 0, N, rank * Kh, (rank + 1) * Kh);
                std::vector<RadParam> kp = { PI("N", N), PI("K", Kh), PI("group", 128) };
                std::vector<uint8_t> cw, cs;
                const int crc = stored(nr, kp, 2, part, { "codes" }, &cw);
                const int csc = stored(nr, kp, 3, part, { "scale" }, &cs);
                if (crc != RAD_OK || csc != RAD_E_UNSUPPORTED ||
                    (int64_t)cw.size() != N * Kh || (int64_t)cs.size() != sn * skh * 2) {
                    ok("fragment order survives a column shard at K/2", false,
                       "relayout of the share rc " + std::to_string(crc) + "/" +
                       std::to_string(csc));
                    laid = false;
                    break;
                }
                st_memcpy(d_wh, cw.data(), cw.size(), hipMemcpyHostToDevice);
                st_memcpy(d_sh, cs.data(), cs.size(), hipMemcpyHostToDevice);
                std::vector<uint16_t> xm((size_t)M * Kh);
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t c = 0; c < Kh; ++c)
                        xm[(size_t)m * Kh + c] = xb[(size_t)(rank * Kh + c)];
                st_memcpy(d_xh, xm.data(), (size_t)M * Kh * 2, hipMemcpyHostToDevice);
                std::vector<RadTensor> qt = {
                    T(d_xh, RAD_BF16, { M, Kh }), T(d_q, RAD_F8E4M3, { M, Kh }),
                    T(d_qs, RAD_F32, { M, skh }),
                };
                std::vector<RadParam> qp = {
                    PI("M", M), PI("n", Kh), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(qz, qt, qp);
                std::vector<RadTensor> ts = {
                    T(d_q, RAD_F8E4M3, { M, Kh }), T(d_qs, RAD_F32, { M, skh }),
                    T(d_wh, RAD_F8E4M3, { N, Kh }), T(d_sh, RAD_BF16, { sn, skh }),
                    T(d_yh, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
                    T(nullptr, RAD_U8, { 1 }),
                };
                absent(ts[5]); absent(ts[6]);
                std::vector<RadParam> ps = {
                    PI("M", M), PI("N", N), PI("K", Kh), PI("group", 128), PS("dtype", "fp8a8"),
                };
                st_memset(d_yh, 0, (size_t)M * N * 2);
                run_case(nr, ts, ps);
                std::vector<uint16_t> got((size_t)M * N);
                st_memcpy(got.data(), d_yh, (size_t)M * N * 2, hipMemcpyDeviceToHost);
                for (int64_t r = 0; r < N; ++r) acc[(size_t)r] += (double)h_bf16(got[(size_t)r]);
            }
            double num = 0, den = 0;
            for (int64_t r = 0; r < N; ++r) {
                const double d = acc[(size_t)r] - (double)ref[(size_t)r];
                num += d * d;
                den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            if (laid)
                ok("fragment order survives a column shard at K/2", e < 8e-2,
                   "rel_l2 " + std::to_string(e));
        }
    }

    /* --- the tiled W8A8 path: quantise the activation with the library's own quantiser, then
     *     GEMM. Every row holds the same activation, so a wrong epilogue index shows as one row
     *     right and the rest wrong. */
    {
        /* A prefill chunk: 2048 rows is sixteen 128-row tiles, so a per-M-block staging bug
         * shows. */
        const int64_t M = 2048;
        void* d_xm = dalloc((size_t)M * K * 2);
        void* d_q  = dalloc((size_t)M * K);
        void* d_qs = dalloc((size_t)M * sk * 4);
        void* d_ym = dalloc((size_t)M * N * 2);
        if (!d_xm || !d_q || !d_qs || !d_ym) { ok("fp8 tiled alloc", false, "hipMalloc failed"); return; }
        std::vector<uint16_t> xm((size_t)M * K);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t c = 0; c < K; ++c) xm[(size_t)m * K + c] = xb[(size_t)c];
        st_memcpy(d_xm, xm.data(), (size_t)M * K * 2, hipMemcpyHostToDevice);

        std::vector<RadTensor> qt = {
            T(d_xm, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
        };
        std::vector<RadParam> qp = {
            PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
        };
        run_case(qz, qt, qp);

        std::vector<RadTensor> ts = {
            T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
            T(d_wf, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_ym, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
        };
        st_memset(d_ym, 0, (size_t)M * N * 2);
        run_case(tl, ts, ps);
        std::vector<uint16_t> got((size_t)M * N);
        st_memcpy(got.data(), d_ym, (size_t)M * N * 2, hipMemcpyDeviceToHost);
        /* Every row must match (a wrong epilogue stride writes one right row). The tolerance is
         * looser than the matvec's: the activation is quantised too, and E4M3's three mantissa
         * bits are 6% of a value. */
        double worst = 0;
        for (int64_t m = 0; m < M; ++m) {
            double num = 0, den = 0;
            for (int64_t r = 0; r < N; ++r) {
                const double d = (double)h_bf16(got[(size_t)m * N + r]) - (double)ref[(size_t)r];
                num += d * d;
                den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
            }
            const double e = den > 0 ? std::sqrt(num / den) : 1.0;
            if (e > worst) worst = e;
        }
        ok("gemm_fp8a8_tiled vs host ref, worst row", worst < 8e-2,
           "rel_l2 " + std::to_string(worst));
    }
}

/* The same check on a real checkpoint weight (spec §17 without the .rad container).
 *
 * `fp8_numeric` builds its own weight, so kernel and test could share a wrong idea of the format.
 * This runs the same comparison on bytes read from a safetensors file: real magnitudes, scale
 * variation, subnormals and zero blocks. The file's two tensors are block fp8's canonical planes,
 * and the matvec and the oracle both read them row-major.
 *
 * Arguments: the file, the weight's byte offset, the scale plane's byte offset, N and K. The
 * header length is a u64 at byte 0 followed by the JSON header, so the data region starts at 8 +
 * that and each tensor's `data_offsets[0]` is relative to it:
 *
 *     N=$(od -An -tu8 -N8 w.safetensors | tr -d ' ')
 *     dd if=w.safetensors bs=1 skip=8 count=$N status=none | tr ',' '\n' | grep -A2 gate_proj
 *
 * then pass 8 + $N + data_offsets[0] for the weight and for its scale plane. */
static void fp8_real(const Plugin& pl, const char* file, long long woff, long long soff,
                     int64_t N, int64_t K) {
    const RadKernelInfo* mv = find(pl, "gemm_fp8a16_nt_m1");
    if (!mv) { ok("fp8 real-weight check", false, "row absent"); return; }
    if (N % 128 || K % 128) { ok("fp8 real-weight check", false, "N and K must be 128-aligned"); return; }
    const int64_t sn = N / 128, sk = K / 128;

    std::FILE* f = std::fopen(file, "rb");
    if (!f) { ok("fp8 real-weight check", false, std::string("cannot open ") + file); return; }
    std::vector<uint8_t> wq((size_t)N * K);
    std::vector<uint16_t> ws((size_t)sn * sk);
    bool okread = std::fseek(f, (long)woff, SEEK_SET) == 0 &&
                  std::fread(wq.data(), 1, wq.size(), f) == wq.size() &&
                  std::fseek(f, (long)soff, SEEK_SET) == 0 &&
                  std::fread(ws.data(), 2, ws.size(), f) == (size_t)ws.size();
    std::fclose(f);
    if (!okread) { ok("fp8 real-weight check", false, "short read"); return; }

    /* A checkpoint block scale is ~1e-4 and a weight byte spans E4M3's whole range, so a plane
     * read at the wrong offset shows here first. */
    double smin = 1e30, smax = -1e30;
    for (size_t i = 0; i < ws.size(); ++i) {
        const double v = h_bf16(ws[i]);
        if (v < smin) smin = v;
        if (v > smax) smax = v;
    }
    ok("fp8 real scale plane is a plausible dequant multiplier",
       smin > 0 && smax < 1.0 && smax / smin < 1e4,
       "min " + std::to_string(smin) + " max " + std::to_string(smax));

    std::vector<float> wd((size_t)N * K);
    for (int64_t r = 0; r < N; ++r)
        for (int64_t c = 0; c < K; ++c)
            wd[(size_t)r * K + c] =
                h_e4m3(wq[(size_t)(r * K + c)]) * h_bf16(ws[(size_t)(r / 128) * sk + (c / 128)]);

    std::vector<uint16_t> xb((size_t)K);
    for (int64_t c = 0; c < K; ++c)
        xb[(size_t)c] = h_to_bf16(0.021f * (float)((c * 17) % 53) - 0.55f);
    std::vector<float> ref((size_t)N, 0.0f);
    for (int64_t r = 0; r < N; ++r) {
        double acc = 0;
        for (int64_t c = 0; c < K; ++c) acc += (double)wd[(size_t)r * K + c] * h_bf16(xb[(size_t)c]);
        ref[(size_t)r] = (float)acc;
    }

    void* d_w = dalloc((size_t)N * K);
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_x = dalloc((size_t)K * 2);
    void* d_y = dalloc((size_t)N * 2);
    if (!d_w || !d_s || !d_x || !d_y) { ok("fp8 real alloc", false, "hipMalloc failed"); return; }
    st_memcpy(d_w, wq.data(), (size_t)N * K, hipMemcpyHostToDevice);
    st_memcpy(d_s, ws.data(), (size_t)sn * sk * 2, hipMemcpyHostToDevice);
    st_memcpy(d_x, xb.data(), (size_t)K * 2, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_x, RAD_BF16, { 1, K }), T(nullptr, RAD_F32, { 1, sk }),
        T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
        T(d_y, RAD_BF16, { 1, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
    };
    absent(ts[1]); absent(ts[5]); absent(ts[6]);
    std::vector<RadParam> ps = {
        PI("M", 1), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
    };
    run_case(mv, ts, ps);
    std::vector<uint16_t> got((size_t)N);
    st_memcpy(got.data(), d_y, (size_t)N * 2, hipMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int64_t r = 0; r < N; ++r) {
        const double d = (double)h_bf16(got[(size_t)r]) - (double)ref[(size_t)r];
        num += d * d;
        den += (double)ref[(size_t)r] * (double)ref[(size_t)r];
    }
    const double e = den > 0 ? std::sqrt(num / den) : 1.0;
    /* Printed even on a pass (`ok` shows detail only on failure), so the margin stays visible. */
    std::printf("        N=%lld K=%lld  rel_l2 %.6g  (bound 3e-2; bf16 output rounding alone is "
                "~2e-3)\n", (long long)N, (long long)K, e);
    ok("gemm_fp8a16_nt_m1 vs host ref on REAL checkpoint bytes", e < 3e-2,
       "rel_l2 " + std::to_string(e));
}

/* ---- the decode GEMM, timed ------------------------------------------------------------------
 *
 * Timing per shape for rows rad-tune cannot reach (the fp8a8 rows declare no axes).
 *
 * The weight is replicated past the last-level cache and the launches walk the copies in turn:
 * back-to-back launches on one buffer would measure cache, while decode reads every weight once.
 *
 * The reported bandwidth is the weight stream only: at M <= 16 the activation (16*K) and output
 * (16*N) are under a percent of the traffic, so it compares directly with the part's plain-stream
 * bandwidth (638 GB/s).
 */
static void perf_row(const Plugin& pl, const char* name, std::vector<RadTensor> ts,
                     std::vector<RadParam> ps, void* wbase, size_t wbytes, int copies,
                     double bytes, int iters, int widx = 2, const char* point = nullptr) {
    const RadKernelInfo* r = find(pl, name);
    if (!r) { std::printf("  %-22s row absent\n", name); return; }
    /* The label is the row, or the tuned point in a sweep. `ps` already carries that point; this
     * only names it. */
    char lbl[64];
    std::snprintf(lbl, sizeof lbl, "%s", (point && *point && std::strcmp(point, "-")) ? point : name);
    name = lbl;
    for (auto& t : ts) if (!t.data && t.rank) t.data = alloc_for(t);

    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.world_size = 1;
    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need < 0) { std::printf("  %-22s scratch %lld\n", name, (long long)need); return; }
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }

    /* One warm launch, which also checks the row serves this shape; a refusal is reported, not
     * timed. */
    const int rc = r->launch(&a, nullptr);
    if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
        std::printf("  %-22s rc=%d (shape not served)\n", name, rc);
        return;
    }

    const double us = st_time_us([&](int i) {
        ts[widx].data = (uint8_t*)wbase + (size_t)(i % copies) * wbytes;
        r->launch(&a, nullptr);
    }, iters, nullptr);
    std::printf("  %-22s %8.1f us   %6.1f GB/s\n", name, us, bytes / (us * 1e3));
}


/* ------------------------------------------------------------------ the small ops, timed
 *
 * Most of a decode step's kernels are small ops moving a few tens of kilobytes, so their cost is
 * launch overhead. The per-op profiler cannot measure it, since it adds a synchronisation per call
 * of the same order. So this times back-to-back launches with no host synchronisation over
 * independent buffers: the floor a fusion would have to beat. */
static void smallop_perf(const Plugin& pl, int64_t M, int64_t K, int iters) {
    if (iters <= 0) iters = 2000;
    const int64_t nkb = K / 128;
    std::printf("small ops at M=%lld K=%lld, %d launches each, no host sync between\n",
                (long long)M, (long long)K, iters);

    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_r  = dalloc((size_t)M * K * 2);
    void* d_gu = dalloc((size_t)M * 2 * K * 2);
    void* d_w  = dalloc((size_t)K * 4);
    void* d_y  = dalloc((size_t)M * K * 2);
    void* d_q  = dalloc((size_t)M * K);
    void* d_s  = dalloc((size_t)M * nkb * 4);
    if (!d_x || !d_r || !d_gu || !d_w || !d_y || !d_q || !d_s) {
        std::printf("  alloc failed\n");
        return;
    }
    st_memset(d_x, 0x3c, (size_t)M * K * 2);
    st_memset(d_r, 0x3c, (size_t)M * K * 2);
    st_memset(d_gu, 0x3c, (size_t)M * 2 * K * 2);
    st_memset(d_w, 0, (size_t)K * 4);

    auto time_row = [&](const char* label, const char* row, std::vector<RadTensor> ts,
                        std::vector<RadParam> ps, double bytes) {
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-24s row absent\n", label); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-24s rc=%d (shape not served)\n", label, rc);
            return;
        }
        hipEvent_t e0, e1;
        st_ev_create(&e0);
        st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0);
        st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("  %-24s %7.2f us   %6.1f GB/s\n", label, us, bytes / (us * 1e3));
    };

    const double eps = 1e-6, wadd = 1.0;
    const double b_norm = (double)M * K * 2 + (double)M * K + (double)M * K * 2;

    time_row("rmsnorm_quant_fp8", "rmsnorm_quant_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, {}), T(d_w, RAD_F32, { K }),
               T(d_q, RAD_F8E4M3, { M, K }), T(d_s, RAD_F32, { M, nkb }),
               T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128), PS("dtype", "bf16"),
               PF("wadd", wadd) }, b_norm);
    time_row("rmsnorm_quant_fp8 (res)", "rmsnorm_quant_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(d_r, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }),
               T(d_q, RAD_F8E4M3, { M, K }), T(d_s, RAD_F32, { M, nkb }),
               T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PI("group", 128), PS("dtype", "bf16"),
               PF("wadd", wadd) }, b_norm + (double)M * K * 4);
    time_row("gated_quant_fp8", "gated_quant_fp8",
             { T(d_gu, RAD_BF16, { M, 2 * K }), T(d_q, RAD_F8E4M3, { M, K }),
               T(d_s, RAD_F32, { M, nkb }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") },
             (double)M * 2 * K * 2 + (double)M * K * 3);
    time_row("quant_act_fp8", "quant_act_fp8",
             { T(d_x, RAD_BF16, { M, K }), T(d_q, RAD_F8E4M3, { M, K }),
               T(d_s, RAD_F32, { M, nkb }) },
             { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") },
             (double)M * K * 3);
    time_row("rmsnorm_bf16", "rmsnorm_bf16",
             { T(d_x, RAD_BF16, { M, K }), T(d_w, RAD_F32, { K }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PF("eps", eps), PS("dtype", "bf16"), PF("wadd", wadd) },
             (double)M * K * 4);
    time_row("add_bf16", "add_bf16",
             { T(d_x, RAD_BF16, { M, K }), T(d_r, RAD_BF16, { M, K }), T(d_y, RAD_BF16, { M, K }) },
             { PI("M", M), PI("n", K), PS("dtype", "bf16") }, (double)M * K * 6);
}

/* ------------------------------------------------------------- the QSA indexer, timed at length
 *
 * The indexer is the only part of a decode step that grows with context: `qsa_score` and
 * `qsa_select` scale with the compressed blocks of history and run once per layer, so they are
 * timed at length.
 *
 * `blocks` is the declared width (max_ctx / ratio) and `ctx` the live one: the launch geometry
 * comes from the declaration and the work from `nc`, as in the deployment.
 *
 * M is 1 + n_spec, not the sequence count: every draft row selects its own set. */
static void qsa_perf(const Plugin& pl, int64_t ctx, int64_t M, int iters) {
    if (iters <= 0) iters = 200;
    const int64_t ratio = 4, heads = 4, n = 128, topk = 512;
    const int64_t max_ctx = 200000, nb = max_ctx / ratio;      /* the declared width */
    const int64_t live = ctx / ratio;                          /* complete blocks */
    if (live < 1 || live > nb) { std::printf("--perfqsa: ctx out of range\n"); return; }
    std::printf("qsa indexer at ctx=%lld (%lld of %lld blocks live), M=%lld, heads=%lld n=%lld "
                "topk=%lld, %d launches\n", (long long)ctx, (long long)live, (long long)nb,
                (long long)M, (long long)heads, (long long)n, (long long)topk, iters);

    const int64_t pages = live;
    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 29 + 7) % 83) - 41) / 47.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 41 + 13) % 71) - 35) / 39.0f);
    /* A scattered table, as a paged pool hands out pages; the block key read is the kernel's only
     * random access. */
    std::vector<int32_t> bth((size_t)nb, -1);
    for (int64_t b = 0; b < live; ++b) bth[(size_t)b] = (int32_t)((b * 7919 + 13) % pages);
    /* And the same table in page order: the difference is the cost of the paged pool, i.e. what a
     * contiguous block-key store could save. */
    std::vector<int32_t> bti((size_t)nb, -1);
    for (int64_t b = 0; b < live; ++b) bti[(size_t)b] = (int32_t)b;
    std::vector<int32_t> nch(1, (int32_t)live), cuh(2), posh((size_t)M);
    cuh[0] = 0; cuh[1] = (int32_t)M;
    for (int64_t m = 0; m < M; ++m) posh[(size_t)m] = (int32_t)(ctx - M + m);

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_bi = dalloc(bti.size() * 4);
    void* d_nc = dalloc(4);
    void* d_cu = dalloc(8);
    void* d_po = dalloc(posh.size() * 4);
    void* d_sc = dalloc((size_t)(M * nb) * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    if (!d_q || !d_bk || !d_bt || !d_bi || !d_nc || !d_cu || !d_po || !d_sc || !d_se || !d_ns ||
        !d_su) {
        std::printf("  alloc failed\n");
        return;
    }
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bi, bti.data(), bti.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), 8, hipMemcpyHostToDevice);
    st_memcpy(d_po, posh.data(), posh.size() * 4, hipMemcpyHostToDevice);

    /* `row` finds the kernel and `name` labels the line, so one row can be timed in two forms. */
    auto run = [&](const char* row, const char* name, std::vector<RadTensor> ts,
                   std::vector<RadParam> ps, double bytes) {
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-18s row absent\n", name); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need < 0) { std::printf("  %-18s scratch %lld\n", name, (long long)need); return; }
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-18s rc=%d (shape not served)\n", name, rc);
            return;
        }
        /* Warm the clocks, not just the caches: an idle card drops to a low-power state, and a
         * ~30 ms timed window falls inside the ramp back up (up to ~2x slower after an idle gap).
         * Half a second of launches first, so the timed window runs at deployment clocks. */
        {
            hipEvent_t w0, w1;
            st_ev_create(&w0); st_ev_create(&w1);
            st_ev_record(w0, nullptr);
            float wms = 0.0f;
            for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                for (int i = 0; i < 32; ++i) r->launch(&a, nullptr);
                st_ev_record(w1, nullptr);
                st_ev_sync(w1);
                st_ev_elapsed(&wms, w0, w1);
            }
            st_ev_destroy(w0); st_ev_destroy(w1);
        }
        const double us = st_time_us([&](int) { r->launch(&a, nullptr); }, iters, nullptr);
        std::printf("  %-30s %9.2f us   %6.1f GB/s\n", name, us, bytes / (us * 1e3));
    };

    /* The score plane is narrowed to the live reach as the architecture does (arch/common/
     * rad_qsa.h): pitch the declared `nb`, column count what the context can have reached. Both
     * kernels size their grid from it. */
    RadTensor sc_t = T(d_sc, RAD_F32, { M, nb });
    sc_t.shape[1] = live + 1;

    /* `live * n * 2` bytes, not `M *` that: the kernel tiles `kQT` query rows through LDS and reads
     * the key history once per tile. */
    const double score_bytes = (double)live * n * 2.0;
    auto score_args = [&]() {
        return std::vector<RadTensor>{ T(d_q,  RAD_BF16, { M, heads * n }),
                                       T(d_bk, RAD_BF16, { pages, n }),
                                       T(d_bt, RAD_I32,  { 1, nb }), T(d_cu, RAD_I32,  { 2 }),
                                       T(d_nc, RAD_I32,  { 1 }), sc_t };
    };
    const std::vector<RadParam> score_params = { PI("M", M), PI("n", n), PI("heads", heads),
                                                 PI("blocks", nb), PI("seqs", 1),
                                                 PS("dtype", "bf16") };
    /* Both forms on one input, timed and compared. They sum the same products in different
     * orders, so the live plane is compared within a tolerance, and an absent block's -inf must
     * match exactly. */
    if (pl.pin_qsa) {
        std::vector<float> plane[2];
        const char* label[2] = { "qsa_score_bf16 scalar form", "qsa_score_bf16 matrix form" };
        for (int f = 0; f < 2; ++f) {
            pl.pin_qsa(f + 1);
            run("qsa_score_bf16", label[f], score_args(), score_params, score_bytes);
            plane[f].resize((size_t)(M * nb));
            st_memcpy(plane[f].data(), d_sc, plane[f].size() * 4, hipMemcpyDeviceToHost);
        }
        pl.pin_qsa(0);
        double maxrel = 0.0, num = 0.0, den = 0.0;
        long long inf_mismatch = 0;
        for (int64_t m = 0; m < M; ++m)
            for (int64_t b = 0; b < live; ++b) {
                const float a = plane[0][(size_t)(m * nb + b)], c = plane[1][(size_t)(m * nb + b)];
                if (std::isinf(a) || std::isinf(c)) { inf_mismatch += (a != c); continue; }
                const double d = (double)a - (double)c;
                num += d * d;
                den += (double)a * a;
                const double sc = std::fabs((double)a) > 1e-3 ? std::fabs((double)a) : 1e-3;
                maxrel = std::max(maxrel, std::fabs(d) / sc);
            }
        std::printf("  forms agree to rel_l2 %.3e, max rel %.3e, %lld -inf mismatches\n",
                    den > 0 ? std::sqrt(num / den) : 0.0, maxrel, inf_mismatch);
    }
    run("qsa_score_bf16", "qsa_score_bf16 (pages scattered)", score_args(), score_params,
        score_bytes);
    run("qsa_score_bf16", "qsa_score_bf16 (pages in order)",
        { T(d_q,  RAD_BF16, { M, heads * n }), T(d_bk, RAD_BF16, { pages, n }),
          T(d_bi, RAD_I32,  { 1, nb }),        T(d_cu, RAD_I32,  { 2 }),
          T(d_nc, RAD_I32,  { 1 }),            sc_t },
        { PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb), PI("seqs", 1),
          PS("dtype", "bf16") },
        score_bytes);
    /* The launcher's rule, and the single-workgroup form it replaces at a wide launch. */
    if (pl.pin_qsel) pl.pin_qsel(1);
    run("qsa_select_i32", "qsa_select_i32 (single-workgroup form)",
        { sc_t, T(d_bt, RAD_I32, { 1, nb }), T(d_nc, RAD_I32, { 1 }),
          T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { 2 }),
          T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) },
        { PI("M", M), PI("blocks", nb), PI("topk", topk), PI("ratio", ratio), PI("seqs", 1) },
        (double)M * live * 8.0);
    if (pl.pin_qsel) pl.pin_qsel(0);
    run("qsa_select_i32", "qsa_select_i32",
        { sc_t, T(d_bt, RAD_I32, { 1, nb }), T(d_nc, RAD_I32, { 1 }),
          T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { 2 }),
          T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) },
        { PI("M", M), PI("blocks", nb), PI("topk", topk), PI("ratio", ratio), PI("seqs", 1) },
        (double)M * live * 8.0);
}


/* The routed-expert GEMM at a prefill chunk, both projections, every prefill form.
 *
 * rad-kbench declares at world 1 (no width split) and does not serve the rotated w4 dtype, so the
 * engine's tp2 shape is timed here: rank 0's at Qwen3.8-Flash-Next, gate_up with three 128-column
 * blocks of an even expert and two of an odd one (parts 2), and the down projection with the
 * matching K slices over the sorted rows.
 *
 * The routing is skewed, since cost depends on run lengths: expert popularity is lognormal (sigma
 * 0.8) and each token takes its top_k by a Gumbel draw. Weights are random codes; traffic and
 * arithmetic do not depend on values.
 *
 * Every form (run-aligned prefill, (tile, run) decode, uniform prefill grid) runs on the same bytes
 * and must match the first byte for byte, since all keep each element's accumulation order. */
static void moe_perf(const Plugin& pl, int64_t Mtok, int iters, int64_t E = 512, int64_t top_k = 10,
                     int pool_pct = 0) {
    if (iters <= 0) iters = 20;
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r) { std::printf("--perfmoe: row moe_gemm_w4a8 absent\n"); return; }
    const int64_t group = 128, Tr = Mtok * top_k;

    uint64_t rs = 0x9E3779B97F4A7C15ull;
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < Mtok; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    int64_t rmin = Tr, rmax = 0, rows_c[2] = { 0, 0 };
    for (int64_t e = 0; e < E; ++e) {
        const int64_t n = (int64_t)by_e[(size_t)e].size();
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)n;
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
        rmin = std::min(rmin, n); rmax = std::max(rmax, n); rows_c[e & 1] += n;
    }
    std::printf("moe_gemm_w4a8 at a %lld-token chunk: Tr=%lld routed rows over %lld experts, a run "
                "%lld..%lld rows (mean %.1f), %d launches\n", (long long)Mtok, (long long)Tr,
                (long long)E, (long long)rmin, (long long)rmax, (double)Tr / (double)E, iters);

    void* d_s = dalloc((size_t)Tr * 4);
    void* d_o = dalloc((size_t)(E + 1) * 4);
    st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);

    struct Proj { const char* name; int64_t N0, K0, N1, K1, parts; bool srt; };
    const Proj projs[2] = { { "gate_up", 768, 2560, 512, 2560, 2, false },
                            { "down",    2560, 384, 2560, 256, 1, true } };
    /* Form 4, where pinnable, is the decode form at two 128-wide K blocks per pass (the timing
     * kMoeGuFk256Rows is set from). Only gate_up has it (K a multiple of 256 and a format that
     * folds scale groups); elsewhere it runs as form 2. */
    const int nforms = pl.pin_moe ? (pl.pin_fk ? 4 : 3) : 1;
    for (const Proj& P : projs) {
        const int64_t N = std::max(P.N0, P.N1), K = std::max(P.K0, P.K1), kb = K / group;
        const int64_t Ma = P.srt ? Tr : Mtok;
        const int64_t nec[2] = { E / 2, E / 2 };
        const int64_t Nc[2] = { P.N0, P.N1 }, Kc[2] = { P.K0, P.K1 };
        std::vector<uint8_t> hb;
        auto fill = [&](size_t n, uint8_t lo, uint8_t span) {
            hb.resize(n);
            for (size_t i = 0; i < n; ++i) hb[i] = (uint8_t)(lo + (uint8_t)(rnd() % span));
        };
        void* d_a = dalloc((size_t)(Ma * K));
        fill((size_t)(Ma * K), 8, 111);
        for (size_t i = 0; i < hb.size(); ++i) if (i & 1) hb[i] |= 0x80u;
        st_memcpy(d_a, hb.data(), hb.size(), hipMemcpyHostToDevice);
        std::vector<float> asc((size_t)(Ma * kb), 0.013f);
        void* d_as = dalloc(asc.size() * 4);
        st_memcpy(d_as, asc.data(), asc.size() * 4, hipMemcpyHostToDevice);
        void* d_w[2]; void* d_ws[2]; void* d_tab[2] = { nullptr, nullptr };
        for (int c = 0; c < 2; ++c) {
            /* Five bits per element: the five-bit pass reads the plane at its own pitch, the
             * four-bit passes the front of each expert's share. */
            d_w[c] = dalloc((size_t)(nec[c] * Nc[c] * Kc[c] / 8 * 5));
            fill((size_t)(nec[c] * Nc[c] * Kc[c] / 8 * 5), 0, 255);
            st_memcpy(d_w[c], hb.data(), hb.size(), hipMemcpyHostToDevice);
            /* Pooled experts, as in the engine: a pointer table with `pool_pct` percent of entries
             * in pinned host memory read over the link. The least routed ones are pooled, as
             * residency keeps hot experts on the card. */
            if (pool_pct > 0) {
                const size_t eb = (size_t)(Nc[c] * Kc[c] / 2);
                std::vector<const void*> tab((size_t)nec[c]);
                std::vector<int64_t> order((size_t)nec[c]);
                for (int64_t x = 0; x < nec[c]; ++x) order[(size_t)x] = x;
                std::sort(order.begin(), order.end(), [&](int64_t p0, int64_t p1) {
                    return by_e[(size_t)(2 * p0 + c)].size() < by_e[(size_t)(2 * p1 + c)].size();
                });
                std::vector<char> pool((size_t)nec[c], 0);
                for (int64_t i = 0; i < nec[c] * pool_pct / 100; ++i) pool[(size_t)order[(size_t)i]] = 1;
                int64_t pooled = 0, pooled_rows = 0;
                for (int64_t x = 0; x < nec[c]; ++x) {
                    tab[(size_t)x] = (const char*)d_w[c] + (size_t)x * eb;
                    if (pool[(size_t)x]) {
                        void* h = nullptr;
                        if (hipHostMalloc(&h, eb, hipHostMallocMapped) != hipSuccess) {
                            std::printf("  %s: pinned allocation %lld failed\n", P.name, (long long)pooled);
                            break;
                        }
                        std::memcpy(h, hb.data() + (size_t)x * eb, eb);
                        void* hd = nullptr;
                        if (hipHostGetDevicePointer(&hd, h, 0) != hipSuccess || !hd) {
                            std::printf("  %s: no device address for pinned block %lld\n", P.name,
                                        (long long)pooled);
                            break;
                        }
                        tab[(size_t)x] = hd;
                        ++pooled;
                        pooled_rows += (int64_t)by_e[(size_t)(2 * x + c)].size();
                    }
                }
                std::printf("  %s class %d: %lld of %lld experts pooled, %lld routed rows on them\n",
                            P.name, c, (long long)pooled, (long long)nec[c], (long long)pooled_rows);
                d_tab[c] = dalloc(tab.size() * sizeof(void*));
                st_memcpy(d_tab[c], tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);
            }
            /* Room for the 32-wide grid's four E4M3 bytes per 128 (the bf16 grid's two bytes). */
            std::vector<uint16_t> wsc((size_t)(2 * nec[c] * Nc[c] * (Kc[c] / group)), h_to_bf16(0.02f));
            d_ws[c] = dalloc(wsc.size() * 2);
            st_memcpy(d_ws[c], wsc.data(), wsc.size() * 2, hipMemcpyHostToDevice);
        }
        void* d_y = dalloc((size_t)(Tr * N * 2));
        if (!d_s || !d_o || !d_a || !d_as || !d_w[0] || !d_w[1] || !d_ws[0] || !d_ws[1] || !d_y) {
            std::printf("  alloc failed\n");
            return;
        }
        std::vector<RadTensor> ts = {
            T(d_a,     RAD_F8E4M3,   { Ma, K }),
            T(d_as,    RAD_F32,      { Ma, kb }),
            T(d_tab[0] ? d_tab[0] : d_w[0], R4D_DT_MOEW4, { nec[0], P.N0, P.K0 }),
            T(d_ws[0], RAD_BF16,     { nec[0], P.N0, P.K0 / group }),
            T(d_s,     RAD_I32,      { Tr }),
            T(d_o,     RAD_I32,      { E + 1 }),
            T(d_tab[1] ? d_tab[1] : d_w[1], R4D_DT_MOEW4, { nec[1], P.N1, P.K1 }),
            T(d_ws[1], RAD_BF16,     { nec[1], P.N1, P.K1 / group }),
            T(d_y,     RAD_BF16,     { Tr, N }),
        };
        /* A pointer table is told from a stacked plane by a leading stride of zero. */
        if (d_tab[0]) ts[2].stride[0] = 0;
        if (d_tab[1]) ts[6].stride[0] = 0;
        std::vector<RadParam> ps = {
            PI("M", Mtok), PI("N", P.N0), PI("K", P.K0), PI("N_odd", P.N1), PI("K_odd", P.K1),
            PI("parts", P.parts), PI("n_expert", E), PI("top_k", top_k), PI("group", group),
            PS("a_order", P.srt ? "sorted" : "token"), PS("dtype", "w4a8h"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        /* The aligned prefill form's run map, sized regardless of the hook: inside the decode band
         * the hook sizes none, and the pinned aligned form would fall back to the uniform grid. */
        {
            const int64_t need = std::max<int64_t>(r->scratch ? r->scratch(&a) : 0,
                                                   (1 + 2 * (E + Tr / 32 + 2)) * 4);
            a.scratch = dalloc((size_t)need);
            a.scratch_bytes = need;
        }
        const double flop = 2.0 * ((double)rows_c[0] * P.N0 * P.K0 + (double)rows_c[1] * P.N1 * P.K1);
        const double welems = (double)nec[0] * P.N0 * P.K0 + (double)nec[1] * P.N1 * P.K1;
        std::vector<uint16_t> first((size_t)(Tr * N)), got((size_t)(Tr * N));
        /* Every scale grid over the same bytes: w4a8h's bf16 per (row, 128 of K), w4nl64a8h's two
         * E4M3 per 128 and w4nl32a8h's four (timing only the extra accumulators and fold), and
         * w5nl64a8h (`sub` 5), which adds the sign plane's loads and merge. Not over pooled
         * experts, whose blocks are sized for four bits. */
        for (const int sub : { 1, 2, 4, 5 }) {
        const bool g64 = sub > 1, w5 = sub == 5;
        if (w5 && (d_tab[0] || d_tab[1])) continue;
        const int64_t sg = w5 ? 2 : sub;
        ts[2] = T(d_tab[0] ? d_tab[0] : d_w[0], w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4,
                  { nec[0], P.N0, P.K0 });
        ts[6] = T(d_tab[1] ? d_tab[1] : d_w[1], w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4,
                  { nec[1], P.N1, P.K1 });
        if (d_tab[0]) ts[2].stride[0] = 0;
        if (d_tab[1]) ts[6].stride[0] = 0;
        ts[3] = g64 ? T(d_ws[0], RAD_F8E4M3, { nec[0], P.N0, sg * P.K0 / group })
                    : T(d_ws[0], RAD_BF16, { nec[0], P.N0, P.K0 / group });
        ts[7] = g64 ? T(d_ws[1], RAD_F8E4M3, { nec[1], P.N1, sg * P.K1 / group })
                    : T(d_ws[1], RAD_BF16, { nec[1], P.N1, P.K1 / group });
        ps.back().sval = w5 ? "w5nl64a8h" : sub == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h" : "w4a8h";
        const double wbytes = welems * (w5 ? 0.625 : 0.5);
        const std::string pname = std::string(P.name) + (w5 ? " w5" : sub == 4 ? " g32"
                                                         : g64 ? " g64" : "");
        for (int f = 0; f < nforms; ++f) {
            if (pl.pin_moe) pl.pin_moe(f == 3 ? 2 : f + 1);
            if (pl.pin_fk) pl.pin_fk(f == 3 ? 2 : 1);
            const int rc = r->launch(&a, nullptr);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                std::printf("  %-12s form %d  rc=%d (not served)\n", pname.c_str(), f + 1, rc);
                continue;
            }
            {
                hipEvent_t w0, w1;
                st_ev_create(&w0); st_ev_create(&w1);
                st_ev_record(w0, nullptr);
                float wms = 0.0f;
                for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                    for (int i = 0; i < 4; ++i) r->launch(&a, nullptr);
                    st_ev_record(w1, nullptr);
                    st_ev_sync(w1);
                    st_ev_elapsed(&wms, w0, w1);
                }
                st_ev_destroy(w0); st_ev_destroy(w1);
            }
            const double us = st_time_us([&](int) { r->launch(&a, nullptr); }, iters, nullptr);
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
            std::string same = "";
            if (f == 0) {
                first = got;
                /* A hash of the first form's bytes, to compare builds; the inputs are a fixed
                 * function of the arguments. */
                uint64_t hsh = 1469598103934665603ull;
                for (uint16_t v : got) { hsh ^= v; hsh *= 1099511628211ull; }
                same = "  fnv " + std::to_string(hsh);
            } else {
                int64_t diff = 0;
                for (size_t i = 0; i < got.size(); ++i) diff += got[i] != first[i];
                same = diff ? ("  " + std::to_string(diff) + " elements differ from form 1")
                            : "  byte-identical to form 1";
            }
            std::printf("  %-12s form %d  %9.1f us  %6.1f TFLOP/s useful  %6.1f GB/s of weight%s\n",
                        pname.c_str(), f + 1, us, flop / (us * 1e6), wbytes / (us * 1e3),
                        same.c_str());
        }
        }
        if (pl.pin_moe) pl.pin_moe(0);
        if (pl.pin_fk) pl.pin_fk(0);
    }
}

/* The launcher's kPf2MinRows: from this band the rule takes the PF2 form and the scratch hook
 * sizes `u` for its weight planes; below it the pin to PF2 is refused. */
static constexpr int64_t kHcPf2MinRows = 512;

/* The gated residual's read at int8 codes. The same inputs run with the codes plane as E4M3 and
 * as I8: the bf16 block input and the rotated E4M3 twin must be byte-identical, and the int8 codes
 * must equal quant_act_i8g's over the bf16 input (per (row, 128) at amax / 127). Row counts reach
 * the per-row kernels, decode tiles and prefill tile; the null stream takes the unfused path, a
 * real one the fused finishers. `form` pins one kernel family where exported (0: the rule). */
static void hc_read_i8(const Plugin& pl, int64_t M, bool real_stream, int form) {
    const RadKernelInfo* r = find(pl, "hc_read_e4m3");
    if (!r) { skip("hc_read_e4m3 int8 codes", "row absent"); return; }
    const int64_t n = 2560, hc = 4, lr = 320, HN = hc * n, grp = 128, nkb = n / grp;
    const int64_t dld = r4d_hc8::row_bytes(HN, r4d_hc8::down_group(HN));
    const int64_t uld = r4d_hc8::row_bytes(lr, r4d_hc8::up_group(lr));
    uint64_t rs = 0x9E3779B97F4A7C15ull ^ (uint64_t)M;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    std::vector<uint16_t> hh((size_t)(M * HN));
    for (auto& v : hh) v = h_to_bf16((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    std::vector<float> wg((size_t)HN);
    for (auto& v : wg) v = (float)((int)(rnd() % 201) - 100) / 1000.0f;
    auto plane = [&](int64_t rows, int64_t K, int64_t G, int64_t ld) {
        std::vector<uint8_t> b((size_t)(rows * ld), 0);
        for (int64_t r0 = 0; r0 < rows; ++r0) {
            uint8_t* row = b.data() + r0 * ld;
            for (int64_t k = 0; k < K; ++k) row[k] = (uint8_t)(8 + rnd() % 111) | ((rnd() & 1) ? 0x80u : 0u);
            float* sc = (float*)(row + K);
            for (int64_t g = 0; g < K / G; ++g) sc[g] = 0.002f + 0.004f * (float)(rnd() % 1000) / 1000.0f;
        }
        return b;
    };
    std::vector<uint8_t> down = plane(lr, HN, r4d_hc8::down_group(HN), dld);
    std::vector<uint8_t> up   = plane(HN, lr, r4d_hc8::up_group(lr), uld);
    std::vector<uint16_t> injw((size_t)(hc * HN));
    for (auto& v : injw) v = h_to_bf16((float)((int)(rnd() % 201) - 100) / 50000.0f);
    const size_t xb = (size_t)(M * n * 2), ib = (size_t)(M * hc * 2), qb = (size_t)(M * n),
                 sb = (size_t)(M * nkb * 4);
    void* d_h = dalloc(hh.size() * 2);
    void* d_w = dalloc(wg.size() * 4);
    void* d_dn = dalloc(down.size());
    void* d_up = dalloc(up.size());
    void* d_iw = dalloc(injw.size() * 2);
    void* d_x = dalloc(xb);
    void* d_inj = dalloc(ib);
    void* d_q = dalloc(qb);
    void* d_qs = dalloc(sb);
    void* d_rq = dalloc(qb);
    void* d_rs = dalloc(sb);
    static const char* const kForm[] = { "the launcher's rule", "the decode tiles' kernels",
                                         "the wide kernels", "the PF2 kernels" };
    const std::string nm = fmt("hc_read_e4m3 int8 codes at M=%lld (%s, %s)", (long long)M,
                               real_stream ? "a real stream" : "the null stream", kForm[form]);
    if (!d_h || !d_w || !d_dn || !d_up || !d_iw || !d_x || !d_inj || !d_q || !d_qs || !d_rq || !d_rs) {
        ok(nm.c_str(), false, "hipMalloc failed");
        return;
    }
    st_memcpy(d_h, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wg.data(), wg.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_dn, down.data(), down.size(), hipMemcpyHostToDevice);
    st_memcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice);
    st_memcpy(d_iw, injw.data(), injw.size() * 2, hipMemcpyHostToDevice);
    hipStream_t hs = nullptr;
    if (real_stream && hipStreamCreate(&hs) != hipSuccess) { ok(nm.c_str(), false, "stream"); return; }
    struct Out { std::vector<uint8_t> x, inj, q, qs, rq, rs; };
    Out out[2];
    if (pl.pin_hc) pl.pin_hc(form);
    for (int pass = 0; pass < 2; ++pass) {
        const uint32_t qdt = pass ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
        std::vector<RadTensor> ts = {
            T(d_h,  RAD_BF16,   { M, HN }),     T(d_w,  RAD_F32,    { HN }),
            T(d_dn, RAD_F8E4M3, { lr, HN }),    T(d_up, RAD_F8E4M3, { HN, lr }),
            T(d_iw, RAD_BF16,   { hc, HN }),    T(d_x,  RAD_BF16,   { M, n }),
            T(d_inj, RAD_BF16,  { M, hc }),     T(d_q,  qdt,        { M, n }),
            T(d_qs, RAD_F32,    { M, nkb }),    T(d_rq, RAD_F8E4M3, { M, n }),
            T(d_rs, RAD_F32,    { M, nkb }),
        };
        std::vector<RadParam> ps = {
            PI("M", M), PI("n", n), PI("hc", hc), PI("lowrank", lr), PF("eps", 1e-6),
            PS("dtype", "bf16"), PF("wadd", 1.0), PI("inject", 1), PI("group", grp),
            PI("rotate", 1), PS("mix", "e4m3"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        st_memset(d_x, 0, xb); st_memset(d_q, 0, qb); st_memset(d_qs, 0, sb);
        st_memset(d_rq, 0, qb); st_memset(d_rs, 0, sb); st_memset(d_inj, 0, ib);
        const int rc = r->launch(&a, (RadStream)hs);
        if (rc != RAD_OK || st_stream_sync(hs) != hipSuccess) {
            ok(nm.c_str(), false, fmt("pass %d rc=%d", pass, rc));
            if (pl.pin_hc) pl.pin_hc(0);
            if (hs) (void)hipStreamDestroy(hs);
            return;
        }
        auto get = [&](std::vector<uint8_t>& v, void* d, size_t b) {
            v.resize(b); st_memcpy(v.data(), d, b, hipMemcpyDeviceToHost);
        };
        get(out[pass].x, d_x, xb); get(out[pass].inj, d_inj, ib); get(out[pass].q, d_q, qb);
        get(out[pass].qs, d_qs, sb); get(out[pass].rq, d_rq, qb); get(out[pass].rs, d_rs, sb);
    }
    if (pl.pin_hc) pl.pin_hc(0);
    if (hs) (void)hipStreamDestroy(hs);
    std::string bad;
    if (out[0].x != out[1].x)     bad += " x";
    if (out[0].inj != out[1].inj) bad += " inj";
    if (out[0].rq != out[1].rq) bad += " rq";
    if (out[0].rs != out[1].rs) bad += " rs";
    const int64_t miss = i8_code_misses((const uint16_t*)out[1].x.data(), n,
                                        (const int8_t*)out[1].q.data(), n,
                                        (const float*)out[1].qs.data(), nkb, M, n);
    int64_t nz = 0;
    for (uint8_t v : out[1].q) nz += v != 0;
    ok(nm.c_str(), bad.empty() && miss == 0 && nz > (int64_t)qb / 2,
       bad.empty() ? fmt("%lld codes or scales differ from the host quantiser, %lld non-zero",
                         (long long)miss, (long long)nz)
                   : "the int8 run changed" + bad);
}

/* The gated residual's read at a prefill chunk: hc_read_e4m3 at Qwen3.8-Flash-Next's shape (n
 * 2560, four streams, lowrank 320, inject gains, the fp8 twin and its rotated twin). Several
 * launches at prefill, so also a harness for libr4d/isa/trace.sh. Random codes, small positive
 * scales.
 *
 * Every form of the prefill tile runs through the r4d_hc_read_pin_prefill hook (the decode tiles'
 * kernels at 128 rows, the wide kernels, and from kHcPf2MinRows the PF2 kernels), compared byte
 * for byte. `timed` also times each form.
 *
 * `f8` picks hc_read_e4m3, or hc_read_bf16 over bf16 mixing matrices; at bf16, PF2's prologue
 * copies mix_down instead of decoding it, and its up product reads the stored mix_up. */
static void hc_prefill(const Plugin& pl, int64_t M, int iters, bool timed, bool f8 = true) {
    if (iters <= 0) iters = 20;
    const char* const rn = f8 ? "hc_read_e4m3" : "hc_read_bf16";
    const RadKernelInfo* r = find(pl, rn);
    if (!r) { skip(fmt("%s prefill forms", rn).c_str(), "row absent"); return; }
    const int64_t n = 2560, hc = 4, lr = 320, HN = hc * n, grp = 128;
    const int64_t dld = r4d_hc8::row_bytes(HN, r4d_hc8::down_group(HN));
    const int64_t uld = r4d_hc8::row_bytes(lr, r4d_hc8::up_group(lr));
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ (uint64_t)M;
    auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };

    std::vector<uint16_t> hh((size_t)(M * HN));
    for (auto& v : hh) v = h_to_bf16((float)((int)(rnd() % 2001) - 1000) / 400.0f);
    std::vector<float> wg((size_t)HN);
    for (auto& v : wg) v = (float)((int)(rnd() % 201) - 100) / 1000.0f;
    auto plane = [&](int64_t rows, int64_t K, int64_t G, int64_t ld) {
        std::vector<uint8_t> b((size_t)(rows * ld), 0);
        for (int64_t r0 = 0; r0 < rows; ++r0) {
            uint8_t* row = b.data() + r0 * ld;
            for (int64_t k = 0; k < K; ++k) row[k] = (uint8_t)(8 + rnd() % 111) | ((rnd() & 1) ? 0x80u : 0u);
            float* sc = (float*)(row + K);
            for (int64_t g = 0; g < K / G; ++g) sc[g] = 0.002f + 0.004f * (float)(rnd() % 1000) / 1000.0f;
        }
        return b;
    };
    /* bf16 weights of the E4M3 planes' magnitude: |w| up to 0.25. */
    auto bplane = [&](int64_t count) {
        std::vector<uint8_t> b((size_t)(count * 2));
        uint16_t* v = (uint16_t*)b.data();
        for (int64_t i = 0; i < count; ++i)
            v[i] = h_to_bf16((float)((int)(rnd() % 2001) - 1000) / 4000.0f);
        return b;
    };
    std::vector<uint8_t> down = f8 ? plane(lr, HN, r4d_hc8::down_group(HN), dld) : bplane(lr * HN);
    std::vector<uint8_t> up   = f8 ? plane(HN, lr, r4d_hc8::up_group(lr), uld) : bplane(HN * lr);
    std::vector<uint16_t> injw((size_t)(hc * HN));
    for (auto& v : injw) v = h_to_bf16((float)((int)(rnd() % 201) - 100) / 50000.0f);

    const size_t xb = (size_t)(M * n * 2), ib = (size_t)(M * hc * 2), qb = (size_t)(M * n),
                 sb = (size_t)(M * (n / grp) * 4);
    void* d_h = dalloc(hh.size() * 2);
    void* d_w = dalloc(wg.size() * 4);
    void* d_dn = dalloc(down.size());
    void* d_up = dalloc(up.size());
    void* d_iw = dalloc(injw.size() * 2);
    void* d_x = dalloc(xb);
    void* d_inj = dalloc(ib);
    void* d_q = dalloc(qb);
    void* d_qs = dalloc(sb);
    void* d_rq = dalloc(qb);
    void* d_rs = dalloc(sb);
    if (!d_h || !d_w || !d_dn || !d_up || !d_iw || !d_x || !d_inj || !d_q || !d_qs || !d_rq || !d_rs) {
        ok(fmt("%s prefill forms: allocation", rn).c_str(), false);
        return;
    }
    st_memcpy(d_h, hh.data(), hh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wg.data(), wg.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_dn, down.data(), down.size(), hipMemcpyHostToDevice);
    st_memcpy(d_up, up.data(), up.size(), hipMemcpyHostToDevice);
    st_memcpy(d_iw, injw.data(), injw.size() * 2, hipMemcpyHostToDevice);

    /* The mixing matrices at their declared shape with the stored dtype, as the engine passes
     * them; the launcher derives the stored row pitch. */
    const uint32_t mdt = f8 ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
    std::vector<RadTensor> ts = {
        T(d_h,  RAD_BF16,   { M, HN }),     T(d_w,  RAD_F32,    { HN }),
        T(d_dn, mdt,        { lr, HN }),    T(d_up, mdt,        { HN, lr }),
        T(d_iw, RAD_BF16,   { hc, HN }),    T(d_x,  RAD_BF16,   { M, n }),
        T(d_inj, RAD_BF16,  { M, hc }),     T(d_q,  RAD_F8E4M3, { M, n }),
        T(d_qs, RAD_F32,    { M, n / grp }), T(d_rq, RAD_F8E4M3, { M, n }),
        T(d_rs, RAD_F32,    { M, n / grp }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("hc", hc), PI("lowrank", lr), PF("eps", 1e-6),
        PS("dtype", "bf16"), PF("wadd", 1.0), PI("inject", 1), PI("group", grp),
        PI("rotate", 1), PS("mix", f8 ? "e4m3" : "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    if (r->scratch) {
        const int64_t need = r->scratch(&a);
        if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
    }
    /* A real stream: the fused forms key their arrival counters on it, and the null stream takes
     * the unfused path. */
    hipStream_t hs = nullptr;
    if (hipStreamCreate(&hs) != hipSuccess) {
        ok(fmt("%s prefill forms: stream", rn).c_str(), false);
        return;
    }
    const RadStream rst = (RadStream)hs;

    struct Out { std::vector<uint8_t> x, inj, q, qs, rq, rs; };
    const int f_lo = pl.pin_hc ? 1 : 0, f_hi = pl.pin_hc ? (M >= kHcPf2MinRows ? 3 : 2) : 0;
    static const char* const kForm[] = { "the launcher's rule", "the decode tiles' kernels",
                                         "the wide kernels", "the PF2 kernels" };
    std::vector<Out> outs;
    for (int form = f_lo; form <= f_hi; ++form) {
        if (pl.pin_hc) pl.pin_hc(form);
        st_memset(d_x, 0, xb); st_memset(d_inj, 0, ib); st_memset(d_q, 0, qb);
        st_memset(d_qs, 0, sb); st_memset(d_rq, 0, qb); st_memset(d_rs, 0, sb);
        const int rc = r->launch(&a, rst);
        if (rc != RAD_OK || st_stream_sync(hs) != hipSuccess) {
            ok(fmt("%s prefill form '%s' at M=%lld is served", rn, kForm[form],
                   (long long)M).c_str(), false, fmt("rc=%d", rc));
            if (pl.pin_hc) pl.pin_hc(0);
            (void)hipStreamDestroy(hs);
            return;
        }
        Out o;
        auto get = [&](std::vector<uint8_t>& v, void* d, size_t b) {
            v.resize(b); st_memcpy(v.data(), d, b, hipMemcpyDeviceToHost);
        };
        get(o.x, d_x, xb); get(o.inj, d_inj, ib); get(o.q, d_q, qb);
        get(o.qs, d_qs, sb); get(o.rq, d_rq, qb); get(o.rs, d_rs, sb);
        outs.push_back(std::move(o));
    }
    /* Form f against the decode tiles' kernels, outs[0]. */
    auto same = [&](size_t f, const char* what) {
        std::string bad;
        if (outs[0].x != outs[f].x)     bad += " x";
        if (outs[0].inj != outs[f].inj) bad += " inj";
        if (outs[0].q != outs[f].q)     bad += " q";
        if (outs[0].qs != outs[f].qs)   bad += " qs";
        if (outs[0].rq != outs[f].rq)   bad += " rq";
        if (outs[0].rs != outs[f].rs)   bad += " rs";
        /* Non-zero output required: two forms that wrote nothing would compare equal. */
        size_t nz = 0;
        for (uint8_t v : outs[f].q) nz += v != 0;
        ok(fmt("%s prefill tile at M=%lld, %s: byte-identical to the decode tiles' kernels", rn,
               (long long)M, what).c_str(),
           bad.empty() && nz > qb / 2, bad.empty() ? fmt("%zu of %zu codes non-zero", nz, qb)
                                                   : "differs in" + bad);
    };
    for (size_t f = 1; f < outs.size(); ++f) same(f, kForm[f_lo + (int)f]);

    if (timed) {
        for (int form = f_lo; form <= f_hi; ++form) {
            if (pl.pin_hc) pl.pin_hc(form);
            hipEvent_t w0, w1;
            st_ev_create(&w0); st_ev_create(&w1);
            st_ev_record(w0, hs);
            float wms = 0.0f;
            for (int guard = 0; guard < 100000 && wms < 500.0f; ++guard) {
                for (int i = 0; i < 8; ++i) r->launch(&a, rst);
                st_ev_record(w1, hs);
                st_ev_sync(w1);
                st_ev_elapsed(&wms, w0, w1);
            }
            st_ev_destroy(w0); st_ev_destroy(w1);
            const double us = st_time_us([&](int) { r->launch(&a, rst); }, iters, hs);
            const double flop = 2.0 * (double)M * (double)HN * (double)(2 * lr + hc);
            std::printf("%s at M=%lld (n %lld, hc %lld, lowrank %lld, inject, fp8 twin + "
                        "rotated), %s: %.1f us a call, %.1f TFLOP/s\n", rn, (long long)M,
                        (long long)n, (long long)hc, (long long)lr, kForm[form], us,
                        flop / (us * 1e6));
        }
    }
    if (pl.pin_hc) pl.pin_hc(0);
    (void)hipStreamDestroy(hs);
}

/* The bf16 skinny GEMM at one shape, every variant, side by side, with the working set rotated
 * past the last-level cache (rad-tune's absolute numbers vary too much between runs to choose a
 * default from).
 *
 * The shapes the model's graph reaches: N = 48 (the GDN a|b projection, the only trunk caller),
 * N = 1280 and N = 256 (the DFlash2 drafter's). A default that changes the drafter's bits changes
 * only its proposals; one that changes the trunk's changes the model. */
static void bf16_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 16 || K % 16 || M < 1) {
        std::printf("--perfbf16: N and K must be 16-aligned and M >= 1\n");
        return;
    }
    const RadKernelInfo* r = find(pl, M <= 16 ? "gemm_bf16_nt_m16" : "gemm_bf16_nt_m64");
    if (!r) { std::printf("--perfbf16: row absent\n"); return; }

    const size_t wbytes = (size_t)N * (size_t)K * 2;
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 16) copies = 16;

    void* d_w = dalloc(wbytes * (size_t)copies);
    void* d_a = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 2);
    if (!d_w || !d_a || !d_y) {
        std::printf("--perfbf16: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    {
        std::vector<uint16_t> w(wbytes / 2);
        for (size_t i = 0; i < w.size(); ++i) w[i] = h_to_bf16((float)((int)(i % 17) - 8) * 0.125f);
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> a((size_t)M * K);
        for (size_t i = 0; i < a.size(); ++i) a[i] = h_to_bf16((float)((int)(i % 13) - 6) * 0.25f);
        st_memcpy(d_a, a.data(), a.size() * 2, hipMemcpyHostToDevice);
    }

    std::printf("\nbf16 skinny GEMM  %s  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, "
                "%d iters\n", r->name, (long long)N, (long long)K, (long long)M,
                (double)wbytes / (1 << 20), copies, iters);

    /* Every legal point of the declared space (~900, pruned to a few hundred at one K by
     * tune_valid), capped at `kPerfCap` for readability. */
    std::vector<RadParam> geom = {
        PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16"),
    };
    /* Above the skinny band the tiled row (same op) is timed too, at each tile variant. */
    const RadKernelInfo* rt = M > 64 && N >= 128 && K % 64 == 0 ? find(pl, "gemm_bf16_nt_tiled")
                                                                 : nullptr;
    for (const RadKernelInfo* row : { r, rt }) {
        if (!row) continue;
        const std::vector<TunePoint> pts = tune_points(row, geom, kPerfCap);
        std::printf("  %s: %zu point(s) of the declared space\n", row->name, pts.size());
        for (const TunePoint& tp : pts) {
            std::vector<RadTensor> ts = {
                T(d_a, RAD_BF16, { M, K }), T(d_w, RAD_BF16, { N, K }), T(d_y, RAD_BF16, { M, N }),
            };
            std::vector<RadParam> ps = with_point(geom, tp);
            perf_row(pl, row->name, ts, ps, d_w, wbytes, copies, (double)wbytes, iters, 1,
                     tp.label.c_str());
        }
    }
}
/* The int2 draft head at every variant. rad-tune never reaches this row (on the DFlash2
 * container its requests are `gemm_nt` and `all_reduce`), and it runs well under the part's
 * 638 GB/s. Tuning it is safe: the head only proposes, and greedy verify takes the trunk's argmax,
 * so changed output bits change the acceptance rate, not the text.
 *
 * The weight is one packed blob (N*K/4 bytes of codes plus N*(K/128)*4 of scale/zero pairs) and
 * the activation the quantiser's int8 with per-group sums: the gemm_nt_q operand list. Contents are
 * shape-correct garbage; only the stream is timed. */
static void w2_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 16 || K % 128 || M < 1) {
        std::printf("--perfw2: N must be 16-aligned, K 128-aligned, M >= 1\n");
        return;
    }
    const RadKernelInfo* r = find(pl, "gemm_w2a8_nt");
    if (!r) { std::printf("--perfw2: row absent\n"); return; }

    const int64_t nkb = K / 128;
    const size_t  wbytes = (size_t)N * (size_t)K / 4;
    const size_t  sbytes = (size_t)N * (size_t)nkb * 4;
    /* Copies past the last-level cache, as in fp8_perf, capped for large weights. */
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 16) copies = 16;

    void* d_w  = dalloc(wbytes * (size_t)copies);
    void* d_s  = dalloc(sbytes);
    void* d_a  = dalloc((size_t)M * K);
    void* d_as = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)nkb * (size_t)(((M + 15) / 16) * 16) * 4);
    void* d_y  = dalloc((size_t)M * N * 2);
    if (!d_w || !d_s || !d_a || !d_as || !d_su || !d_y) {
        std::printf("--perfw2: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    /* Non-zero scales: all-zero dequants could hit a denormal fast path and mismeasure. */
    {
        std::vector<uint32_t> sz((size_t)N * (size_t)nkb, 0x00003C00u);   /* zp 0, scale 1.0 */
        st_memcpy(d_s, sz.data(), sz.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint8_t> w(wbytes, 0x1Bu);        /* codes 0,1,2,3 */
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<int8_t> a((size_t)M * K, 3);
        st_memcpy(d_a, a.data(), a.size(), hipMemcpyHostToDevice);
        std::vector<float> as((size_t)M, 1.0f / 127.0f);
        st_memcpy(d_as, as.data(), as.size() * 4, hipMemcpyHostToDevice);
        std::vector<float> su((size_t)nkb * (size_t)(((M + 15) / 16) * 16), 3.0f * 128.0f);
        st_memcpy(d_su, su.data(), su.size() * 4, hipMemcpyHostToDevice);
    }

    const double bytes = (double)(wbytes + sbytes);
    std::printf("\nint2 draft head  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, %d iters\n",
                (long long)N, (long long)K, (long long)M,
                (double)(wbytes + sbytes) / (1 << 20), copies, iters);

    std::vector<RadParam> geom = {
        PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "w2a8"),
    };
    const std::vector<TunePoint> pts = tune_points(r, geom, kPerfCap);
    std::printf("  %zu point(s) of the declared space\n", pts.size());
    for (const TunePoint& tp : pts) {
        std::vector<RadTensor> ts = {
            T(d_a, RAD_I8, { M, K }), T(d_as, RAD_F32, { M }),
            T(d_w, RAD_U8, { (int64_t)wbytes }), T(d_s, RAD_U8, { (int64_t)sbytes }),
            T(d_y, RAD_BF16, { M, N }),
            T(d_su, RAD_F32, { nkb * ((M + 15) / 16) * 16 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[6]);
        std::vector<RadParam> ps = with_point(geom, tp);
        perf_row(pl, "gemm_w2a8_nt", ts, ps, d_w, wbytes, copies, bytes, iters, 2,
                 tp.label.c_str());
    }
}
static void fp8_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 128 || K % 128 || M < 1) {
        std::printf("--perf: N and K must be 128-aligned and M >= 1\n");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;
    const size_t wbytes = (size_t)N * K;
    /* Enough copies to push the working set past the last-level cache, capped for large
     * weights. */
    int copies = (int)(((size_t)768 << 20) / wbytes) + 1;
    if (copies > 64) copies = 64;

    void* d_w = dalloc(wbytes * (size_t)copies);
    void* d_s = dalloc((size_t)sn * sk * 2);
    void* d_q = dalloc((size_t)M * K);
    void* d_qs = dalloc((size_t)M * sk * 4);
    void* d_x = dalloc((size_t)M * K * 2);
    void* d_y = dalloc((size_t)M * N * 2);
    if (!d_w || !d_s || !d_q || !d_qs || !d_x || !d_y) {
        std::printf("--perf: hipMalloc failed (%zu MiB wanted)\n",
                    (wbytes * (size_t)copies) >> 20);
        return;
    }
    /* Non-zero, checkpoint-like scales, so no dequant is zero (a possible fast path). */
    {
        std::vector<uint16_t> s((size_t)sn * sk, h_to_bf16(3.0e-4f));
        st_memcpy(d_s, s.data(), s.size() * 2, hipMemcpyHostToDevice);
        std::vector<uint8_t> w(wbytes, 0x38);   /* E4M3 1.0 */
        for (int c = 0; c < copies; ++c)
            st_memcpy((uint8_t*)d_w + (size_t)c * wbytes, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint8_t> q((size_t)M * K, 0x38);
        st_memcpy(d_q, q.data(), q.size(), hipMemcpyHostToDevice);
        std::vector<float> qs((size_t)M * sk, 1.0f);
        st_memcpy(d_qs, qs.data(), qs.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint16_t> x((size_t)M * K, h_to_bf16(0.5f));
        st_memcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    }

    const double bytes = (double)wbytes;
    std::printf("\nfp8 decode GEMM  N=%lld K=%lld M=%lld  %.1f MiB weight, %d copies, %d iters\n",
                (long long)N, (long long)K, (long long)M, (double)wbytes / (1 << 20), copies, iters);

    {
        std::vector<RadTensor> ts = {
            T(d_x, RAD_BF16, { M, K }), T(nullptr, RAD_F32, { M, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[1]); absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a16"),
        };
        perf_row(pl, "gemm_fp8a16_nt_m1", ts, ps, d_w, wbytes, copies, bytes, iters);
    }
    {
        std::vector<RadTensor> ts = {
            T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }),
            T(d_w, RAD_F8E4M3, { N, K }), T(d_s, RAD_BF16, { sn, sk }),
            T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }),
        };
        absent(ts[5]); absent(ts[6]);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"),
        };
        perf_row(pl, "gemm_fp8a8_nt_m16", ts, ps, d_w, wbytes, copies, bytes, iters);
        perf_row(pl, "gemm_fp8a8_tiled", ts, ps, d_w, wbytes, copies, bytes, iters);
    }

    /* The lm head, a decode GEMM too. Its weight is [N][K + 2*nkb] E4M3 with each row's bf16
     * scales in its tail (r4d_fp8_layout.cpp), so it has its own buffer, and the weight is the
     * second operand. */
    if (M <= 64) {
        const int64_t lstride = r4d_fp8::lm_row_bytes(K);
        const size_t  lwb = (size_t)N * (size_t)lstride;
        int lcopies = (int)(((size_t)768 << 20) / lwb) + 1;
        if (lcopies > 16) lcopies = 16;
        void* d_lw = dalloc(lwb * (size_t)lcopies);
        void* d_ly = dalloc((size_t)M * N * 4);
        if (!d_lw || !d_ly) {
            std::printf("  %-22s hipMalloc failed (%zu MiB)\n", "logits_gemm_fp8",
                        (lwb * (size_t)lcopies) >> 20);
        } else {
            std::vector<uint8_t> plane(lwb);
            for (int64_t n = 0; n < N; ++n) {
                uint8_t* r = plane.data() + (size_t)n * lstride;
                std::memset(r, 0x38, (size_t)K);                  /* E4M3 1.0 */
                uint16_t* tail = (uint16_t*)(r + K);
                for (int64_t b = 0; b < sk; ++b) tail[b] = h_to_bf16(3.0e-4f);
            }
            for (int c = 0; c < lcopies; ++c)
                st_memcpy((uint8_t*)d_lw + (size_t)c * lwb, plane.data(), lwb,
                          hipMemcpyHostToDevice);
            /* The declared shape, [N][K], as the shim receives it; it derives the stored width
             * from n_embd (r4d_gemm_fp8a16_nt_m1.hip). */
            std::vector<RadTensor> ts = {
                T(d_x, RAD_BF16, { M, K }),
                T(d_lw, RAD_F8E4M3, { N, K }),
                T(d_ly, RAD_F32, { M, N }),
            };
            std::vector<RadParam> ps = {
                PI("n_vocab", N), PI("n_embd", K), PI("M", M), PS("dtype", "bf16"),
            };
            perf_row(pl, "logits_gemm_fp8", ts, ps, d_lw, lwb, lcopies, (double)lwb, iters, 1);
        }
    }
}

/* The 8-bit trunk GEMMs side by side at one shape, each with its useful TFLOP/s: block E4M3 W8A8
 * (the fp8 trunk), the int8 W8A8 tiled kernel, and the fp8 kernels' int8 twins (gemm_i8a8_*); at
 * M <= 64 also the two narrow rows. Operands hold constants (timing does not depend on values). */
static void w8_perf(const Plugin& pl, int64_t N, int64_t K, int64_t M, int iters) {
    if (N % 256 || K % 128 || M < 1) {
        std::printf("--perfw8: N a multiple of 256, K of 128, M >= 1\n");
        return;
    }
    const int64_t sn = N / 128, sk = K / 128;
    const size_t wbytes = (size_t)N * K;
    void* d_w  = dalloc(wbytes);
    void* d_fs = dalloc((size_t)sn * sk * 2);           /* E4M3 block scales, bf16 */
    void* d_is = dalloc((size_t)N * sk * 4);            /* the int8 rows' scale planes */
    void* d_q  = dalloc((size_t)M * K);
    void* d_qs = dalloc((size_t)M * sk * 4);
    void* d_x  = dalloc((size_t)M * K * 2);
    void* d_y  = dalloc((size_t)M * N * 2);
    if (!d_w || !d_fs || !d_is || !d_q || !d_qs || !d_x || !d_y) {
        std::printf("--perfw8: hipMalloc failed\n");
        return;
    }
    {
        std::vector<uint8_t> w(wbytes, 0x38);
        st_memcpy(d_w, w.data(), wbytes, hipMemcpyHostToDevice);
        std::vector<uint16_t> fs((size_t)sn * sk, h_to_bf16(3.0e-4f));
        st_memcpy(d_fs, fs.data(), fs.size() * 2, hipMemcpyHostToDevice);
        std::vector<uint32_t> is((size_t)N * sk, 0x0C001C00u);
        st_memcpy(d_is, is.data(), is.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint8_t> q((size_t)M * K, 0x38);
        st_memcpy(d_q, q.data(), q.size(), hipMemcpyHostToDevice);
        std::vector<float> qs((size_t)M * sk, 1.0f);
        st_memcpy(d_qs, qs.data(), qs.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint16_t> x((size_t)M * K, h_to_bf16(0.5f));
        st_memcpy(d_x, x.data(), x.size() * 2, hipMemcpyHostToDevice);
    }
    const double flop = 2.0 * (double)M * (double)N * (double)K;
    std::printf("\n8-bit trunk GEMMs  N=%lld K=%lld M=%lld  %d iters\n", (long long)N, (long long)K,
                (long long)M, iters);
    auto run = [&](const char* row, const char* dt, std::vector<RadTensor> ts) {
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", dt),
        };
        const RadKernelInfo* r = find(pl, row);
        if (!r) { std::printf("  %-22s row absent\n", row); return; }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("  %-22s rc=%d (shape not served)\n", row, rc);
            return;
        }
        hipEvent_t e0, e1;
        st_ev_create(&e0);
        st_ev_create(&e1);
        st_ev_record(e0, nullptr);
        for (int i = 0; i < iters; ++i) r->launch(&a, nullptr);
        st_ev_record(e1, nullptr);
        st_ev_sync(e1);
        float ms = 0;
        st_ev_elapsed(&ms, e0, e1);
        st_ev_destroy(e0);
        st_ev_destroy(e1);
        const double us = (double)ms * 1000.0 / iters;
        std::printf("  %-22s %9.1f us  %6.1f TFLOP/s\n", row, us, flop / (us * 1e6));
    };
    std::vector<RadTensor> fp8 = {
        T(d_q, RAD_F8E4M3, { M, K }), T(d_qs, RAD_F32, { M, sk }), T(d_w, RAD_F8E4M3, { N, K }),
        T(d_fs, RAD_BF16, { sn, sk }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(fp8[5]); absent(fp8[6]);
    run("gemm_fp8a8_tiled", "fp8a8", fp8);
    std::vector<RadTensor> i8 = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M }), T(d_w, RAD_U8, { (int64_t)wbytes }),
        T(d_is, RAD_U8, { N * sk * 2 }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(i8[5]); absent(i8[6]);
    run("gemm_w8a8_tiled", "w8a8", i8);
    /* The int8 twin of the fp8 row: i8 activation with an f32 scale per (row, 128), i8 weight with
     * a bf16 scale per row per 128; the fp8 row's scale buffers fit both. */
    std::vector<uint16_t> rs((size_t)N * sk, h_to_bf16(3.0e-4f));
    void* d_rs = dalloc(rs.size() * 2);
    if (!d_rs) { std::printf("--perfw8: hipMalloc failed\n"); return; }
    st_memcpy(d_rs, rs.data(), rs.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> i8g = {
        T(d_q, RAD_I8, { M, K }), T(d_qs, RAD_F32, { M, sk }), T(d_w, RAD_I8, { N, K }),
        T(d_rs, RAD_BF16, { N, sk }), T(d_y, RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }),
        T(nullptr, RAD_U8, { 1 }),
    };
    absent(i8g[5]); absent(i8g[6]);
    run("gemm_i8a8_tiled", "i8a8", i8g);
    if (M <= 64) {
        run("gemm_fp8a8_nt_m16", "fp8a8", fp8);
        run("gemm_i8a8_nt_m16", "i8a8", i8g);
    }
}

/* The w4 arrangement against libr4d's own dequant: the relayout writes fragment order and (scale,
 * zero) dwords and dequant_w4_bf16 undoes both on the card. The planes are an i4 grid with an f16
 * scale per 128, and the result must be code times scale. */
static void w4_roundtrip(const Plugin& pl) {
    const RadKernelInfo* wrow = find(pl, "gemm_w4a8_nt_m64");
    const RadKernelInfo* drow = find(pl, "dequant_w4_bf16");
    if (!wrow || !drow || !wrow->layout || !wrow->relayout) {
        ok("w4 relayout vs dequant_w4_bf16", false, "rows or hooks absent");
        return;
    }
    const int64_t N = 32, K = 256, group = 128;
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group) };

    std::vector<float> src((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            src[(size_t)(n * K + k)] = (float)(((n * 31 + k * 17) % 61) - 30) / 30.0f;
    const Planes p = int4_planes(src, N, K, group, RAD_F16, false);

    std::vector<uint8_t> wq, sz;
    RadLayout lay;
    const int rw = stored(wrow, lp, 2, p, { "codes" }, &wq, &lay);
    const int rs = stored(wrow, lp, 3, p, { "scale" }, &sz);
    if (rw != RAD_OK || rs != RAD_OK) {
        ok("w4 relayout", false, "rc " + std::to_string(rw) + "/" + std::to_string(rs));
        return;
    }
    ok("w4 layout is the codes at half a byte each", lay.bytes == N * K / 2,
       "bytes " + std::to_string(lay.bytes));
    {
        std::string why;
        const bool cw = restores(wrow, lp, 2, p, { "codes" }, wq, &why);
        std::string why_s;
        const bool cs = restores(wrow, lp, 3, p, { "scale" }, sz, &why_s);
        ok("w4 relayout inverts to the planes it was handed", cw && cs, why + " " + why_s);
    }

    void* dwq = dalloc(wq.size());
    void* dsz = dalloc(sz.size());
    void* dout = dalloc((size_t)(N * K * 2));
    st_memcpy(dwq, wq.data(), wq.size(), hipMemcpyHostToDevice);
    st_memcpy(dsz, sz.data(), sz.size(), hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(dwq,  RAD_U8,   { (int64_t)wq.size() }),
        T(dsz,  RAD_U8,   { (int64_t)sz.size() }),
        T(dout, RAD_BF16, { N, K }),
    };
    std::vector<RadParam> ps = {
        PI("N", N), PI("K", K), PI("group", group), PS("dtype", "w4"), PI("twos", 1),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = drow->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok("dequant_w4_bf16 launch", false, "rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> got((size_t)(N * K));
    st_memcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost);

    /* Code times scale as the planes hold them, rounded to bf16 like the kernel's output. */
    const int64_t ng = K / group;
    double worst = 0.0;
    int64_t bad = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t i = n * K + k;
            const uint32_t u = (uint32_t)got[(size_t)i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            const float want = h_bf16(h_to_bf16(
                (float)code_at(p, n, k) * rad_load_f32(p.data[1].data(), RAD_F16, n * ng + k / group)));
            const double err = std::fabs((double)f - (double)want);
            /* one bf16 step either way: the two sides may round one product differently */
            if (!(err <= 1e-6 + 1e-2 * std::fabs((double)want))) { ++bad; if (err > worst) worst = err; }
        }
    ok("w4 relayout round-trips through libr4d's own dequant_w4_bf16", bad == 0,
       std::to_string(bad) + " elements off, worst " + std::to_string(worst));
}


/* One geometry per class (even and odd experts) and the number of stacked parts along the weight
 * rows and the output. One class shape is the plain grouped GEMM; two are a rank's uneven slice of
 * every expert, where the narrower class's gate/up pair writes zeros past its part and its down
 * projection reads a shorter K. `sorted_a`: the down projection's operand, one row per routed slot
 * in sorted order (a_order sorted); otherwise token rows (a_order token). */
struct MoeW4Shape { const char* name; int64_t N0, K0, N1, K1, parts, ne; bool sorted_a = false; };

/* The last kernel one launch dispatches, as r4d_record_dispatches saw it; null if it dispatched
 * none. */
static const void* last_dispatch(const Plugin& pl, const RadKernelInfo* row, RadArgs& a, int* rc) {
    const void* fns[8] = {};
    pl.record(fns, 8);
    *rc = row->launch(&a, nullptr);
    const int n = pl.record(nullptr, 0);
    return n >= 1 && n <= 8 ? fns[n - 1] : nullptr;
}

/* The decode form's staged K both ways, byte for byte: one 128-wide K block per pass and two.
 * Two pins dispatching one function fail, so a form is never compared against itself. Both
 * launches are pinned, for the same reason. `fk2` says whether the shape can take two blocks per
 * pass (token operand, a format that folds scale groups, every class's K a multiple of 256);
 * otherwise there is one form, already checked by the form loop. */
static void moe_dec_variants(const Plugin& pl, const RadKernelInfo* row, RadArgs& a, void* d_y,
                             size_t elems, bool fk2, const std::string& what) {
    if (!fk2 || !pl.pin_fk || !pl.pin_moe) return;
    struct Var { int fk; const char* name; };
    static const Var kVar[] = { { 1, "one K block a pass" }, { 2, "two K blocks a pass" } };
    constexpr int kN = (int)(sizeof(kVar) / sizeof(kVar[0]));
    const void* fn[kN] = {};
    std::vector<uint16_t> base;
    for (int v = 0; v < kN; ++v) {
        const std::string label = what + " decode form, " + kVar[v].name;
        {
            std::vector<uint16_t> junk(elems, 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        pl.pin_moe(2);
        pl.pin_fk(kVar[v].fk);
        int rc = RAD_OK;
        fn[v] = last_dispatch(pl, row, a, &rc);
        st_stream_sync(nullptr);
        pl.pin_moe(0);
        pl.pin_fk(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); return; }
        int same = -1;
        for (int u = 0; u < v; ++u)
            if (fn[u] && fn[u] == fn[v]) same = u;
        if (same >= 0) {
            ok(label.c_str(), false, std::string("dispatched the same kernel as ") +
                                     kVar[same].name);
            continue;
        }
        std::vector<uint16_t> got(elems);
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        if (v == 0) { base = got; continue; }     /* checked against the host by the form loop */
        int64_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += got[i] != base[i];
        ok(label.c_str(), diff == 0, std::to_string(diff) + " of " + std::to_string(elems) +
                                     " elements differ from " + kVar[0].name);
    }
}

/* The run-aligned form's two column tiles on the down projection, byte for byte. Where the sorted
 * operand, one part and every class's width in whole 256-column tiles let the rule take the
 * 256-column tile, the 128-column tile (r4d_moe_gemm_pin_down_tile 1) must write the same bytes.
 * Each launch records its two dispatches (the run map, then the GEMM): the rule and the 256 pin
 * must dispatch one GEMM and the 128 pin another. Where the 256-column tile does not fit, its pin is refused. `base` is the
 * aligned form's bytes under the rule, already checked against the host.
 *
 * Past 512 experts there is no aligned launch (the map holds kMaxExpert + 1 offsets), so such a
 * shape is skipped. */
static void moe_aligned_down_tile(const Plugin& pl, const RadKernelInfo* row, RadArgs& a,
                                  void* d_y, void* d_map, int64_t map_bytes, size_t elems,
                                  const std::vector<uint16_t>& base, bool dn_wide, int64_t ne,
                                  const std::string& what) {
    if (!pl.pin_dtile || !pl.pin_moe || base.empty()) return;
    const std::string head = what + " aligned prefill form";
    if (ne > 512) {
        skip((head + ", column tiles").c_str(), "no run map past 512 experts, so no aligned launch");
        return;
    }
    void* const sc0 = a.scratch;
    const int64_t sb0 = a.scratch_bytes;
    a.scratch = d_map;
    a.scratch_bytes = map_bytes;
    pl.pin_moe(1);
    auto restore = [&] {
        pl.pin_dtile(0);
        pl.pin_moe(0);
        a.scratch = sc0;
        a.scratch_bytes = sb0;
    };
    /* One pinned launch into stale bytes; `gemm` gets its second dispatch, the GEMM. */
    auto launch = [&](int pin, std::vector<uint16_t>* got, const void** gemm) {
        {
            std::vector<uint16_t> junk(elems, 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        const void* fns[4] = {};
        pl.pin_dtile(pin);
        pl.record(fns, 4);
        const int rc = row->launch(&a, nullptr);
        const int n = pl.record(nullptr, 0);
        st_stream_sync(nullptr);
        pl.pin_dtile(0);
        if (gemm) *gemm = n == 2 ? fns[1] : nullptr;
        if (got) {
            got->assign(elems, 0);
            st_memcpy(got->data(), d_y, elems * 2, hipMemcpyDeviceToHost);
        }
        return rc;
    };
    if (!dn_wide) {
        const int rc = launch(2, nullptr, nullptr);
        restore();
        ok((head + ", the 256-column down tile is refused on this shape").c_str(),
           rc == RAD_E_UNSUPPORTED, "rc=" + std::to_string(rc));
        return;
    }
    const void* fn[3] = {};
    std::vector<uint16_t> got[3];
    for (int pin = 0; pin < 3; ++pin) {
        const int rc = launch(pin, &got[pin], &fn[pin]);
        if (rc != RAD_OK || !fn[pin]) {
            restore();
            ok((head + ", column tiles").c_str(), false,
               "pin " + std::to_string(pin) + ": launch rc " + std::to_string(rc) +
                   (fn[pin] ? "" : ", not two dispatches"));
            return;
        }
    }
    restore();
    if (fn[0] != fn[2] || fn[1] == fn[2]) {
        ok((head + ", column tiles").c_str(), false,
           std::string("the rule ") + (fn[0] == fn[2] ? "dispatches" : "does not dispatch") +
               " the 256-column GEMM; the 128 pin " +
               (fn[1] == fn[2] ? "dispatches it too" : "dispatches another"));
        return;
    }
    struct Var { int pin; const char* name; };
    static const Var kVar[] = { { 1, "the 128-column down tile" },
                                { 2, "the 256-column down tile, pinned" } };
    for (const Var& v : kVar) {
        int64_t diff = 0;
        for (size_t i = 0; i < elems; ++i) diff += got[v.pin][i] != base[i];
        ok((head + ", " + v.name).c_str(), diff == 0,
           std::to_string(diff) + " of " + std::to_string(elems) +
               " elements differ from the rule's 256-column tile");
    }
}

/* The 4-bit grouped MoE GEMM against a host product over the same planes (the check the
 * `moe_gemm_w4a8` row promises instead of a kbench case). The relayout reorders each dword's
 * nibbles and the kernel expands them with V_PERM_B32 into LDS; they must agree on nibble order,
 * row pitch (the launcher halves an element stride into bytes) and scale grid (one bf16 per row
 * per 128, not per 128x128 tile). The reference reads the canonical planes, independent of either
 * arrangement.
 *
 * Both bands: `T <= 128` runs a 16-row M tile, above it a 128-row one, differing in wave grid and
 * in how many experts a block's row range straddles (the binary search and run loop).
 *
 * Every launcher form runs against the host: the run-aligned prefill form (a run map in its
 * scratch), the (tile, run) decode form, and the uniform prefill grid of a caller with no scratch.
 *
 * `sub` 2 runs the codebook at E4M3 scales per 64 of K (w4nl64a8h), 4 per 32 (w4nl32a8h): each part
 * of a 128-wide group has its own scale. `w5` with `sub` 2 runs the five-bit codebook (w5nl64a8h),
 * whose stored row is its nibbles then its signs. `four` also runs `moe_gemm_w4a8_mod4` over the
 * same experts and holds it to the two-class op's bytes. */
static void moe_w4_gemm(const Plugin& pl, int64_t Tr, const MoeW4Shape& sh, int sub = 0,
                        bool w5 = false, bool four = false) {
    const bool g64 = sub > 1;
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r || !r->layout || !r->relayout) {
        ok((std::string("moe_gemm_w4a8 (") + sh.name + ")").c_str(), false, "row or hooks absent");
        return;
    }
    const std::string what = std::string("moe_gemm_w4a8 vs a host decode (T=") +
                             std::to_string(Tr) + ", " + sh.name +
                             (w5 ? ", w5nl64a8h" : sub == 4 ? ", w4nl32a8h"
                              : g64 ? ", w4nl64a8h" : "") + ")";
    const char* dt = w5 ? "w5nl64a8h" : sub == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h" : "w4a8";

    const int64_t ne = sh.ne, group = 128, top_k = 2;
    const int64_t Nc[2] = { sh.N0, sh.N1 }, Kc[2] = { sh.K0, sh.K1 };
    const int64_t nec[2] = { (ne + 1) / 2, ne / 2 };
    const int64_t N = std::max(sh.N0, sh.N1), K = std::max(sh.K0, sh.K1);
    const int64_t kb = K / group, M = sh.sorted_a ? Tr : Tr / top_k;
    const char* aord = sh.sorted_a ? "sorted" : "token";
    const int64_t fy = N / sh.parts;
    std::vector<RadParam> lp = { PI("N", sh.N0), PI("K", sh.K0), PI("N_odd", sh.N1),
                                 PI("K_odd", sh.K1), PI("group", group), PS("dtype", dt) };

    /* Each class's stored bytes from the hooks via its own operands (2/3 and 6/7), so the odd
     * table's view is tested too. Per expert: a float source, its planes (kept for the reference)
     * and the arrangement. */
    std::vector<uint8_t>  wq[2];
    std::vector<uint16_t> ws[2];
    std::vector<Planes>   cp[2];
    for (int c = 0; c < 2; ++c) {
        const int ow = c ? 6 : 2, os = c ? 7 : 3;
        const int64_t kbc = Kc[c] / group;
        const int64_t wbytes = w5 ? Nc[c] * Kc[c] / 8 * 5 : Nc[c] * Kc[c] / 2;
        wq[c].assign((size_t)(nec[c] * wbytes), 0);
        ws[c].assign((size_t)(nec[c] * Nc[c] * kbc * (sub == 4 ? 2 : 1)), 0);
        std::vector<float> wsrc((size_t)(Nc[c] * Kc[c]));
        for (int64_t x = 0; x < nec[c]; ++x) {
            const int64_t e = 2 * x + c;
            for (int64_t n = 0; n < Nc[c]; ++n)
                for (int64_t k = 0; k < Kc[c]; ++k)
                    wsrc[(size_t)(n * Kc[c] + k)] =
                        (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
            cp[c].push_back(g64 ? nl64_planes(wsrc, Nc[c], Kc[c], 128 / sub, w5)
                                : int4_planes(wsrc, Nc[c], Kc[c], group, RAD_BF16, false));
            std::vector<uint8_t> w, sc;
            const int rw = g64 ? stored(r, lp, ow, cp[c].back(), { "codes", "table" }, &w)
                               : stored(r, lp, ow, cp[c].back(), { "codes" }, &w);
            const int rs = g64 ? stored(r, lp, os, cp[c].back(), { "scale", "scale.1" }, &sc)
                               : stored(r, lp, os, cp[c].back(), { "scale" }, &sc);
            if (rw != RAD_OK || rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what.c_str(), false, "relayout rc " + std::to_string(rw) + "/" +
                                        std::to_string(rs));
                return;
            }
            if ((int64_t)w.size() != wbytes ||
                (int64_t)sc.size() != Nc[c] * kbc * (sub == 4 ? 4 : 2)) {
                ok(what.c_str(), false, "stored byte counts are not N*K/2 (5/8 at five bits) and "
                                        "N*(K/group)*2 (*4 at a 32)");
                return;
            }
            std::memcpy(wq[c].data() + (size_t)(x * wbytes), w.data(), w.size());
            std::memcpy((uint8_t*)ws[c].data() + (size_t)x * sc.size(), sc.data(), sc.size());
        }
    }

    /* The activation as E4M3 bytes and a per-(row, group) f32 scale, so host and device read the
     * same numbers. The scale varies along rows and groups, so reading another row's scale
     * shows. */
    std::vector<uint8_t> aq((size_t)(M * K));
    std::vector<float>   as((size_t)(M * kb));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < kb; ++g)
            as[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g + 0.0007f * (float)(m % 11);
        for (int64_t k = 0; k < K; ++k) {
            /* 8..126: no subnormal and no 0x7F (E4M3's NaN). */
            const uint32_t c = (uint32_t)((m * 13 + k * 5) % 119) + 8;
            aq[(size_t)(m * K + k)] = (uint8_t)(((m + k) & 1) ? (c | 0x80u) : c);
        }
    }

    /* Uneven runs per expert and a dropped tail, for the run loop and dropped-slot zeroing. */
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_a  = dalloc(aq.size());
    void* d_as = dalloc(as.size() * 4);
    void* d_w[2]  = { dalloc(wq[0].size()), dalloc(wq[1].size()) };
    void* d_ws[2] = { dalloc(ws[0].size() * 2), dalloc(ws[1].size() * 2) };
    void* d_s  = dalloc(sorted.size() * 4);
    void* d_o  = dalloc(off.size() * 4);
    void* d_y  = dalloc((size_t)(Tr * N * 2));
    st_memcpy(d_a,  aq.data(),     aq.size(),         hipMemcpyHostToDevice);
    st_memcpy(d_as, as.data(),     as.size() * 4,     hipMemcpyHostToDevice);
    for (int c = 0; c < 2; ++c) {
        st_memcpy(d_w[c],  wq[c].data(), wq[c].size(),     hipMemcpyHostToDevice);
        st_memcpy(d_ws[c], ws[c].data(), ws[c].size() * 2, hipMemcpyHostToDevice);
    }
    st_memcpy(d_s,  sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o,  off.data(),    off.size() * 4,    hipMemcpyHostToDevice);
    /* Stale bytes where the narrower class writes zeros, so a pad tile left unwritten shows. */
    {
        std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
        st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
    }

    /* The stacked weight-table form (a non-zero leading stride); the engine passes a pointer
     * array, and the kernel reads both. */
    std::vector<RadTensor> ts = {
        T(d_a,     RAD_F8E4M3,   { M, K }),
        T(d_as,    RAD_F32,      { M, kb }),
        T(d_w[0],  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { nec[0], sh.N0, sh.K0 }),
        g64 ? T(d_ws[0], RAD_F8E4M3, { nec[0], sh.N0, sub * sh.K0 / group })
            : T(d_ws[0], RAD_BF16,   { nec[0], sh.N0, sh.K0 / group }),
        T(d_s,     RAD_I32,      { Tr }),
        T(d_o,     RAD_I32,      { ne + 1 }),
        T(d_w[1],  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { nec[1], sh.N1, sh.K1 }),
        g64 ? T(d_ws[1], RAD_F8E4M3, { nec[1], sh.N1, sub * sh.K1 / group })
            : T(d_ws[1], RAD_BF16,   { nec[1], sh.N1, sh.K1 / group }),
        T(d_y,     RAD_BF16,     { Tr, N }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", sh.N0), PI("K", sh.K0), PI("N_odd", sh.N1), PI("K_odd", sh.K1),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", aord), PS("dtype", dt),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    /* Room for the run map at the smallest prefill tile regardless of the hook, which sizes none
     * inside the decode band. */
    const int64_t map_bytes = (1 + 2 * (ne + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);

    /* The reference, once: every form must reproduce it. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < Tr; ++i) {
        const int32_t f = sorted[(size_t)i];
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        const int c = (int)(e & 1);
        const int64_t x = e / 2, fc = Nc[c] / sh.parts, kbc = Kc[c] / group;
        for (int64_t n = 0; n < N; ++n) {
            double want = 0.0;
            const int64_t pp = n / fy, j = n - pp * fy;
            if (f >= 0 && i < live && j < fc) {
                const int64_t tok = sh.sorted_a ? i : f / top_k, wr = pp * fc + j;
                for (int64_t g = 0; g < kbc; ++g) {
                    if (g64) {
                        /* `sub` scale groups per 128, each E4M3 under the fixed constant. */
                        const Planes& q = cp[c][(size_t)x];
                        const int64_t gw = group / sub;
                        for (int64_t h = 0; h < sub; ++h) {
                            double part = 0.0;
                            for (int64_t k = g * group + h * gw; k < g * group + (h + 1) * gw; ++k)
                                part += (double)tab[code_at(q, wr, k)] *
                                        (double)h_e4m3(aq[(size_t)(tok * K + k)]);
                            const double sf = (double)h_e4m3(q.data[1][(size_t)(wr * q.row_bytes(1) +
                                                                                 sub * g + h)]) *
                                              (double)kW4G64Scale;
                            want += part * sf * (double)as[(size_t)(tok * kb + g)];
                        }
                        continue;
                    }
                    double sub = 0.0;
                    for (int64_t k = g * group; k < (g + 1) * group; ++k)
                        sub += (double)code_at(cp[c][(size_t)x], wr, k) *
                               (double)h_e4m3(aq[(size_t)(tok * K + k)]);
                    const float sf = rad_load_f32(cp[c][(size_t)x].data[1].data(), RAD_BF16,
                                                  wr * kbc + g);
                    want += sub * (double)sf * (double)as[(size_t)(tok * kb + g)];
                }
            }
            ref[(size_t)(i * N + n)] = want;
        }
    }

    const int f_lo = pl.pin_moe ? 1 : 0, f_hi = pl.pin_moe ? 3 : 0;
    std::vector<uint16_t> two_cls[4];      /* each form's bytes, for the four-class pass below */
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what + " " + kForm[form];
        if (pl.pin_moe) pl.pin_moe(form);
        /* Stale bytes again, so a form cannot inherit the previous form's zeros. */
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a.scratch = form == 3 ? nullptr : d_map;
        a.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        std::string first;
        for (int64_t i = 0; i < Tr; ++i)
            for (int64_t n = 0; n < N; ++n) {
                const double want = ref[(size_t)(i * N + n)];
                const uint32_t u = (uint32_t)got[(size_t)(i * N + n)] << 16;
                float gf;
                std::memcpy(&gf, &u, 4);
                /* A bf16 relative half-ulp (1/512) plus room for a reordered 256-term f32 sum.
                 * Written so a NaN on either side counts as a miss. */
                const double err = std::fabs((double)gf - want);
                if (!(err <= 0.004 * std::fabs(want) + 1e-4)) {
                    if (!bad++)
                        first = ", first at row " + std::to_string(i) + " col " +
                                std::to_string(n) + ": " + std::to_string(gf) + " for " +
                                std::to_string(want);
                    if (!(err <= worst)) worst = err;
                }
            }
        ok(label.c_str(), bad == 0, std::to_string(bad) + " of " + std::to_string(Tr * N) +
                                    " elements off, worst " + std::to_string(worst) + first);
        two_cls[form] = got;
    }
    /* The decode form against the uniform prefill grid, byte for byte (every form keeps each
     * element's accumulation order). The decode form finds runs by a one-round-trip search up to
     * 512 experts and a chunked search above; the grid uses a binary search, a third way. */
    if (!two_cls[2].empty() && !two_cls[3].empty()) {
        int64_t diff = 0;
        for (size_t i = 0; i < two_cls[2].size(); ++i) diff += two_cls[2][i] != two_cls[3][i];
        ok((what + " decode form, byte for byte with the uniform prefill grid").c_str(),
           diff == 0, std::to_string(diff) + " of " + std::to_string(Tr * N) +
                      " elements differ");
    }
    /* The aligned form's two column tiles on the down projection where every class is whole
     * 256-column tiles; elsewhere the 256-column pin is refused. */
    const bool dn_wide = sh.sorted_a && sh.parts == 1 && sh.N0 % 256 == 0 && sh.N1 % 256 == 0;
    moe_aligned_down_tile(pl, r, a, d_y, d_map, map_bytes, (size_t)(Tr * N), two_cls[1], dn_wide,
                          ne, what);
    /* The decode form's staged K both ways, where the shape can take two blocks per pass (never on
     * the sorted operand). */
    const bool fk2 = !sh.sorted_a && g64 && sh.K0 % 256 == 0 && sh.K1 % 256 == 0;
    moe_dec_variants(pl, r, a, d_y, (size_t)(Tr * N), fk2, what);
    if (!four) return;

    /* The same experts as four classes by number mod 4 (`moe_gemm_q_mod4`): each parity table read
     * at twice its stride from its first and second entries gives two classes, so classes 0 and 2
     * are the even table's halves and 1 and 3 the odd one's. Every form must reproduce the
     * two-class op's bytes, checking the class and entry each run resolves to. */
    const RadKernelInfo* r4 = find(pl, "moe_gemm_w4a8_mod4");
    const std::string what4 = what + ", four classes";
    if (!r4) {
        ok(what4.c_str(), false, "row moe_gemm_w4a8_mod4 absent");
        return;
    }
    auto halves = [](const RadTensor& t, int bits, RadTensor* lo, RadTensor* hi) {
        const int64_t n = t.shape[0];
        *lo = t;
        *hi = t;
        lo->shape[0] = (n + 1) / 2;
        hi->shape[0] = n / 2;
        lo->stride[0] = t.stride[0] * 2;
        hi->stride[0] = t.stride[0] * 2;
        hi->data = (char*)t.data + t.stride[0] * bits / 8;
    };
    const int wbits = w5 ? 5 : 4, sbits = g64 ? 8 : 16;
    RadTensor c4w[4], c4s[4];
    halves(ts[2], wbits, &c4w[0], &c4w[2]);
    halves(ts[3], sbits, &c4s[0], &c4s[2]);
    halves(ts[6], wbits, &c4w[1], &c4w[3]);
    halves(ts[7], sbits, &c4s[1], &c4s[3]);
    std::vector<RadTensor> ts4 = { ts[0], ts[1], c4w[0], c4s[0], ts[4], ts[5], c4w[1], c4s[1],
                                   c4w[2], c4s[2], c4w[3], c4s[3], ts[8] };
    std::vector<RadParam> ps4 = {
        PI("M", M), PI("N", sh.N0), PI("K", sh.K0), PI("N_1", sh.N1), PI("K_1", sh.K1),
        PI("N_2", sh.N0), PI("K_2", sh.K0), PI("N_3", sh.N1), PI("K_3", sh.K1),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", aord), PS("dtype", dt),
    };
    RadArgs a4;
    std::memset(&a4, 0, sizeof(a4));
    a4.t = ts4.data(); a4.n_t = (int)ts4.size();
    a4.p = ps4.data(); a4.n_p = (int)ps4.size();
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what4 + " " + kForm[form];
        if (two_cls[form].empty()) continue;      /* the two-class launch failed (reported) */
        if (pl.pin_moe) pl.pin_moe(form);
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a4.scratch = form == 3 ? nullptr : d_map;
        a4.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r4->launch(&a4, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        int64_t diff = 0;
        for (size_t i = 0; i < got.size(); ++i) diff += got[i] != two_cls[form][i];
        ok(label.c_str(), diff == 0, std::to_string(diff) + " of " + std::to_string(Tr * N) +
                                     " elements differ from the two-class op's");
    }
    moe_aligned_down_tile(pl, r4, a4, d_y, d_map, map_bytes, (size_t)(Tr * N), two_cls[1],
                          dn_wide, ne, what4);
    moe_dec_variants(pl, r4, a4, d_y, (size_t)(Tr * N), fk2, what4);
}

/* The four-class grouped GEMM, each class at its own geometry, against a host decode.
 *
 * moe_w4_gemm's four-class pass pairs geometries (0 with 2, 1 with 3), with class 0 always wide.
 * A rank of four holds one wide class (two blocks of each five-block expert, one of the others),
 * which is class 0 only on rank 0. Here each class is built at its own [N_q, K_q], stored through
 * the row's hooks at its own operands (2/3, 6/7, 8/9, 10/11) and checked against the planes, so a
 * wide class 1, 2 or 3 must read its own width and pitch and a narrow class 0 must write zeros past
 * its part. `dt`: fp8a8 runs `moe_gemm_fp8a8_mod4` (block E4M3, a bf16 scale per 128x128 tile);
 * the four-bit formats run `moe_gemm_w4a8_mod4`. Every form, as in moe_w4_gemm. */
struct MoeQ4Shape { const char* name; int64_t N[4], K[4], parts, ne; };

/* A rank of four's slices of a five-block expert: gate/up (two parts) at two 128-column blocks per
 * part for the wide class and one for the others, down at 256 and 128 of K. The wide class is 1, 2
 * or 3 (ranks other than 0). Ten experts make classes 0 and 1 one expert longer; three hundred
 * spread a decode step's runs over mostly empty experts. Every part is whole 128-row blocks of the
 * fp8 scale grid or inside one, since block fp8 reads one scale per tile. */
static const MoeQ4Shape kMoeQ4Shapes[] = {
    { "gate/up, class 1 wide", { 256, 512, 256, 256 }, { 256, 256, 256, 256 }, 2, 10 },
    { "gate/up, class 2 wide", { 256, 256, 512, 256 }, { 256, 256, 256, 256 }, 2, 10 },
    { "down, class 3 wide",    { 128, 128, 128, 128 }, { 128, 128, 128, 256 }, 1, 10 },
    { "300 experts, gate/up, class 3 wide", { 128, 128, 128, 256 }, { 128, 128, 128, 128 }, 2,
      300 },
};

static void moe_q4_gemm(const Plugin& pl, int64_t Tr, const MoeQ4Shape& sh, const char* dt) {
    const bool fp8 = std::strcmp(dt, "fp8a8") == 0, w5 = std::strcmp(dt, "w5nl64a8h") == 0;
    /* Scale groups per 128 of K at the E4M3-scaled forms (moe_w4_gemm's `sub`), 0 otherwise. */
    const int64_t sub = std::strcmp(dt, "w4nl32a8h") == 0 ? 4
                      : (w5 || std::strcmp(dt, "w4nl64a8h") == 0) ? 2 : 0;
    const bool g64 = sub > 1;
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const char* row = fp8 ? "moe_gemm_fp8a8_mod4" : "moe_gemm_w4a8_mod4";
    const std::string what = std::string(row) + " vs a host decode (T=" + std::to_string(Tr) +
                             ", " + sh.name + ", " + dt + ")";
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || (!fp8 && !r->relayout)) {
        ok(what.c_str(), false, "row or hooks absent");
        return;
    }

    const int64_t ne = sh.ne, group = 128, top_k = 2;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        N = std::max(N, sh.N[q]);
        K = std::max(K, sh.K[q]);
    }
    const int64_t kb = K / group, M = Tr / top_k, fy = N / sh.parts;
    const std::vector<RadParam> lp = {
        PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]), PI("K_1", sh.K[1]),
        PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]), PI("K_3", sh.K[3]),
        PI("group", group), PS("dtype", dt),
    };

    /* Each class's experts e = 4x + q from moe_w4_gemm's float source: their planes (kept for the
     * reference) and the hooks' stored bytes at the class's operands. Block fp8 is stored as its
     * planes; four-bit codes and the E4M3 scale view are relaid; a bf16 scale plane is stored
     * as is. */
    std::vector<uint8_t> wq[4], ws[4];
    std::vector<Planes>  cp[4];
    int64_t nec[4] = { 0, 0, 0, 0 };
    for (int q = 0; q < 4; ++q) {
        const int64_t Nq = sh.N[q], Kq = sh.K[q], kbq = Kq / group;
        const int ow = q ? 4 + 2 * q : 2, os = ow + 1;
        const int64_t wbytes = fp8 ? Nq * Kq : w5 ? Nq * Kq / 8 * 5 : Nq * Kq / 2;
        const int64_t sbytes = fp8 ? (Nq + group - 1) / group * kbq * 2
                                   : Nq * kbq * (g64 ? sub : 2);
        nec[q] = (ne + 3 - q) / 4;
        std::vector<float> wsrc((size_t)(Nq * Kq));
        for (int64_t x = 0; x < nec[q]; ++x) {
            const int64_t e = 4 * x + q;
            for (int64_t n = 0; n < Nq; ++n)
                for (int64_t k = 0; k < Kq; ++k)
                    wsrc[(size_t)(n * Kq + k)] =
                        (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
            cp[q].push_back(fp8 ? fp8_block_planes(wsrc, Nq, Kq)
                            : g64 ? nl64_planes(wsrc, Nq, Kq, group / sub, w5)
                                  : int4_planes(wsrc, Nq, Kq, group, RAD_BF16, false));
            std::vector<uint8_t> w, sc;
            const int rw = g64 ? stored(r, lp, ow, cp[q].back(), { "codes", "table" }, &w)
                               : stored(r, lp, ow, cp[q].back(), { "codes" }, &w);
            const int rs = g64 ? stored(r, lp, os, cp[q].back(), { "scale", "scale.1" }, &sc)
                               : stored(r, lp, os, cp[q].back(), { "scale" }, &sc);
            if (rw != (fp8 ? RAD_E_UNSUPPORTED : RAD_OK) ||
                rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what.c_str(), false, "class " + std::to_string(q) + " relayout rc " +
                                        std::to_string(rw) + "/" + std::to_string(rs));
                return;
            }
            if ((int64_t)w.size() != wbytes || (int64_t)sc.size() != sbytes) {
                ok(what.c_str(), false, "class " + std::to_string(q) + " stored " +
                                        std::to_string(w.size()) + " and " +
                                        std::to_string(sc.size()) + " bytes an expert, for " +
                                        std::to_string(wbytes) + " and " + std::to_string(sbytes));
                return;
            }
            wq[q].insert(wq[q].end(), w.begin(), w.end());
            ws[q].insert(ws[q].end(), sc.begin(), sc.end());
        }
    }

    /* moe_w4_gemm's activation and routing: E4M3 bytes without subnormals or NaN, an f32 scale per
     * (row, group) varying along both, uneven runs and a dropped tail. */
    std::vector<uint8_t> aq((size_t)(M * K));
    std::vector<float>   as((size_t)(M * kb));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t g = 0; g < kb; ++g)
            as[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g + 0.0007f * (float)(m % 11);
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t c = (uint32_t)((m * 13 + k * 5) % 119) + 8;
            aq[(size_t)(m * K + k)] = (uint8_t)(((m + k) & 1) ? (c | 0x80u) : c);
        }
    }
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_a  = dalloc(aq.size());
    void* d_as = dalloc(as.size() * 4);
    void* d_w[4];
    void* d_ws[4];
    for (int q = 0; q < 4; ++q) {
        d_w[q]  = dalloc(wq[q].size());
        d_ws[q] = dalloc(ws[q].size());
        st_memcpy(d_w[q],  wq[q].data(), wq[q].size(), hipMemcpyHostToDevice);
        st_memcpy(d_ws[q], ws[q].data(), ws[q].size(), hipMemcpyHostToDevice);
    }
    void* d_s = dalloc(sorted.size() * 4);
    void* d_o = dalloc(off.size() * 4);
    void* d_y = dalloc((size_t)(Tr * N * 2));
    st_memcpy(d_a,  aq.data(),     aq.size(),         hipMemcpyHostToDevice);
    st_memcpy(d_as, as.data(),     as.size() * 4,     hipMemcpyHostToDevice);
    st_memcpy(d_s,  sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o,  off.data(),    off.size() * 4,    hipMemcpyHostToDevice);

    /* The stacked table form, as moe_w4_gemm hands it, each class at its own extents. */
    auto codes_t = [&](int q) {
        return T(d_w[q], fp8 ? (uint32_t)RAD_F8E4M3 : w5 ? (uint32_t)R4D_DT_MOEW5
                                                         : (uint32_t)R4D_DT_MOEW4,
                 { nec[q], sh.N[q], sh.K[q] });
    };
    auto scale_t = [&](int q) {
        const int64_t kbq = sh.K[q] / group;
        return fp8 ? T(d_ws[q], RAD_BF16, { nec[q], (sh.N[q] + group - 1) / group, kbq })
             : g64 ? T(d_ws[q], RAD_F8E4M3, { nec[q], sh.N[q], sub * kbq })
                   : T(d_ws[q], RAD_BF16, { nec[q], sh.N[q], kbq });
    };
    std::vector<RadTensor> ts = {
        T(d_a, RAD_F8E4M3, { M, K }), T(d_as, RAD_F32, { M, kb }), codes_t(0), scale_t(0),
        T(d_s, RAD_I32, { Tr }), T(d_o, RAD_I32, { ne + 1 }), codes_t(1), scale_t(1),
        codes_t(2), scale_t(2), codes_t(3), scale_t(3), T(d_y, RAD_BF16, { Tr, N }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]), PI("K_1", sh.K[1]),
        PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]), PI("K_3", sh.K[3]),
        PI("parts", sh.parts), PI("n_expert", ne), PI("top_k", top_k), PI("group", group),
        PS("a_order", "token"), PS("dtype", dt),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t map_bytes = (1 + 2 * (ne + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);

    /* The reference, from the planes: row i's expert e is class e % 4, entry e / 4, at that
     * class's width and K; a part's columns past the class's width are zero. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < Tr; ++i) {
        const int32_t f = sorted[(size_t)i];
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        const int q = (int)(e % 4);
        const Planes& p = cp[q][(size_t)(e / 4)];
        const int64_t fc = sh.N[q] / sh.parts, Kq = sh.K[q], kbq = Kq / group;
        for (int64_t n = 0; n < N; ++n) {
            double want = 0.0;
            const int64_t pp = n / fy, j = n - pp * fy;
            if (f >= 0 && i < live && j < fc) {
                const int64_t tok = f / top_k, wr = pp * fc + j;
                for (int64_t g = 0; g < kbq; ++g) {
                    const double a_s = (double)as[(size_t)(tok * kb + g)];
                    if (fp8) {
                        /* E4M3 codes and the bf16 scale of the element's 128x128 tile. */
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)h_e4m3(p.data[0][(size_t)(wr * Kq + k)]) *
                                    (double)h_e4m3(aq[(size_t)(tok * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            (wr / group) * kbq + g) * a_s;
                    } else if (g64) {
                        /* `sub` scale groups per 128, each E4M3 under the fixed constant. */
                        const int64_t gw = group / sub;
                        for (int64_t h = 0; h < sub; ++h) {
                            double part = 0.0;
                            for (int64_t k = g * group + h * gw; k < g * group + (h + 1) * gw; ++k)
                                part += (double)tab[code_at(p, wr, k)] *
                                        (double)h_e4m3(aq[(size_t)(tok * K + k)]);
                            const uint8_t sb =
                                p.data[1][(size_t)(wr * p.row_bytes(1) + sub * g + h)];
                            want += part * (double)h_e4m3(sb) * (double)kW4G64Scale * a_s;
                        }
                    } else {
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)code_at(p, wr, k) *
                                    (double)h_e4m3(aq[(size_t)(tok * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            wr * kbq + g) * a_s;
                    }
                }
            }
            ref[(size_t)(i * N + n)] = want;
        }
    }

    const int f_lo = pl.pin_moe ? 1 : 0, f_hi = pl.pin_moe ? 3 : 0;
    std::vector<uint16_t> aligned;     /* the aligned form's bytes, for the column-tile pass */
    for (int form = f_lo; form <= f_hi; ++form) {
        const std::string label = what + " " + kForm[form];
        if (pl.pin_moe) pl.pin_moe(form);
        /* Stale bytes where a narrow class writes zeros, so a pad tile left unwritten shows. */
        {
            std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
            st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
        }
        a.scratch = form == 3 ? nullptr : d_map;
        a.scratch_bytes = form == 3 ? 0 : map_bytes;
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (pl.pin_moe) pl.pin_moe(0);
        if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double worst = 0.0;
        int64_t bad = 0;
        std::string first;
        for (int64_t i = 0; i < Tr; ++i)
            for (int64_t n = 0; n < N; ++n) {
                const double want = ref[(size_t)(i * N + n)];
                const double err = std::fabs((double)h_bf16(got[(size_t)(i * N + n)]) - want);
                /* moe_w4_gemm's tolerance; a NaN on either side is a miss. */
                if (!(err <= 0.004 * std::fabs(want) + 1e-4)) {
                    if (!bad++)
                        first = ", first at row " + std::to_string(i) + " col " +
                                std::to_string(n) + ": " +
                                std::to_string(h_bf16(got[(size_t)(i * N + n)])) + " for " +
                                std::to_string(want);
                    if (!(err <= worst)) worst = err;
                }
            }
        ok(label.c_str(), bad == 0, std::to_string(bad) + " of " + std::to_string(Tr * N) +
                                    " elements off, worst " + std::to_string(worst) + first);
        if (form == 1) aligned = got;
    }
    /* The token operand takes no 256-column down tile, and the pin to it is refused. */
    moe_aligned_down_tile(pl, r, a, d_y, d_map, map_bytes, (size_t)(Tr * N), aligned, false, ne,
                          what);
    /* The decode form's variants, each class at its own geometry; fp8a8 runs W4 0 here. */
    bool fk2 = g64;
    for (int q = 0; q < 4; ++q) fk2 = fk2 && sh.K[q] % 256 == 0;
    moe_dec_variants(pl, r, a, d_y, (size_t)(Tr * N), fk2, what);
}

/* A decode step's routing as router_topk_scatter leaves it: M tokens with `top_k` distinct experts
 * each (half from thirty-two popular experts) and, with `hot`, every token's first slot on one
 * expert, whose M-row run crosses 16-row tiles. Every 37th slot goes to another rank (dropped).
 * Then the scatter's stable counting sort: runs in expert order, slots in slot order. */
static void moe_route(int64_t M, int64_t top_k, int64_t ne, uint64_t seed, bool hot,
                      std::vector<int32_t>& off, std::vector<int32_t>& sorted) {
    const int64_t T = M * top_k;
    std::vector<int32_t> ids((size_t)T, -1);
    uint64_t s = seed;
    auto rnd = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 33);
    };
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                const uint32_t x = rnd();
                e = (x & 1u) ? (int32_t)((((x >> 1) % 32u) * 13u) % (uint32_t)ne)
                             : (int32_t)((x >> 1) % (uint32_t)ne);
                if (hot && k == 0) e = (int32_t)(ne / 3);
                dup = false;
                for (int64_t q = 0; q < k; ++q) dup = dup || ids[(size_t)(m * top_k + q)] == e;
            }
            ids[(size_t)(m * top_k + k)] = e;
        }
    for (int64_t i = 5; i < T; i += 37) ids[(size_t)i] = -1;
    std::vector<int32_t> cnt((size_t)ne, 0);
    for (const int32_t e : ids)
        if (e >= 0) ++cnt[(size_t)e];
    off.assign((size_t)(ne + 1), 0);
    for (int64_t e = 0; e < ne; ++e) off[(size_t)(e + 1)] = off[(size_t)e] + cnt[(size_t)e];
    sorted.assign((size_t)T, -1);
    std::vector<int32_t> at(off.begin(), off.end() - 1);
    for (int64_t i = 0; i < T; ++i)
        if (ids[(size_t)i] >= 0) sorted[(size_t)(at[(size_t)ids[(size_t)i]]++)] = (int32_t)i;
}

/* The sorted operand (the down projection) at decode: every form against a host decode, and
 * moe_down_dec_kernel (the rule's decode form here) against the (tile, run) form byte for byte.
 *
 * Down reads the activation in sorted order, one row per routed slot, with a decode-shaped routing
 * (moe_route): T = 40 (one sequence at depth 3) and T = 640 (sixteen), a run longer than a tile,
 * runs crossing tiles, a dropped tail. Each class has its own [N_q, K_q]; one narrower than the
 * output must zero the column tile past its width. The decode kernel runs at several column groups,
 * three uneven over the 64-column tiles, and the (tile, run) form is checked against the uniform
 * prefill grid on this sorted-operand path. Each pinned variant must dispatch a kernel of its own.
 *
 * Also an output naming fewer rows than its buffer while the offsets name the whole routing: both
 * decode forms must write those rows as the full launch did and nothing past them. */
struct MoeDnShape { const char* name; int64_t N[4], K[4], ne, top_k; };

static void moe_down_dec_case(const Plugin& pl, const MoeDnShape& sh, const char* dt,
                              std::initializer_list<int64_t> Ts) {
    const bool fp8 = std::strcmp(dt, "fp8a8") == 0, w5 = std::strcmp(dt, "w5nl64a8h") == 0;
    const int64_t sub = std::strcmp(dt, "w4nl32a8h") == 0 ? 4
                      : (w5 || std::strcmp(dt, "w4nl64a8h") == 0) ? 2 : 0;
    const bool g64 = sub > 1;
    const float* tab = w5 ? kW5nlTable : kW4nlTable;
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const char* row = fp8 ? "moe_gemm_fp8a8_mod4" : "moe_gemm_w4a8_mod4";
    const std::string head = std::string(row) + ", sorted operand (" + sh.name + ", " + dt + ")";
    const RadKernelInfo* r = find(pl, row);
    if (!r || !r->layout || (!fp8 && !r->relayout)) {
        ok(head.c_str(), false, "row or hooks absent");
        return;
    }
    if (!pl.pin_moe || !pl.pin_ddec) {
        skip(head.c_str(), "the plugin exports no form pins");
        return;
    }

    const int64_t ne = sh.ne, group = 128, top_k = sh.top_k;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        N = std::max(N, sh.N[q]);
        K = std::max(K, sh.K[q]);
    }
    const int64_t kb = K / group;
    const std::vector<RadParam> lp = {
        PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]), PI("K_1", sh.K[1]),
        PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]), PI("K_3", sh.K[3]),
        PI("group", group), PS("dtype", dt),
    };

    /* The experts as moe_q4_gemm builds them: e = 4x + q, each class at its own geometry, stored
     * by the row's hooks at the class's operands. */
    std::vector<uint8_t> wq[4], ws[4];
    std::vector<Planes>  cp[4];
    int64_t nec[4] = { 0, 0, 0, 0 };
    for (int q = 0; q < 4; ++q) {
        const int64_t Nq = sh.N[q], Kq = sh.K[q], kbq = Kq / group;
        const int ow = q ? 4 + 2 * q : 2, os = ow + 1;
        const int64_t wbytes = fp8 ? Nq * Kq : w5 ? Nq * Kq / 8 * 5 : Nq * Kq / 2;
        const int64_t sbytes = fp8 ? (Nq + group - 1) / group * kbq * 2
                                   : Nq * kbq * (g64 ? sub : 2);
        nec[q] = (ne + 3 - q) / 4;
        std::vector<float> wsrc((size_t)(Nq * Kq));
        for (int64_t x = 0; x < nec[q]; ++x) {
            const int64_t e = 4 * x + q;
            for (int64_t n = 0; n < Nq; ++n)
                for (int64_t k = 0; k < Kq; ++k)
                    wsrc[(size_t)(n * Kq + k)] =
                        (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
            cp[q].push_back(fp8 ? fp8_block_planes(wsrc, Nq, Kq)
                            : g64 ? nl64_planes(wsrc, Nq, Kq, group / sub, w5)
                                  : int4_planes(wsrc, Nq, Kq, group, RAD_BF16, false));
            std::vector<uint8_t> w, sc;
            const int rw = g64 ? stored(r, lp, ow, cp[q].back(), { "codes", "table" }, &w)
                               : stored(r, lp, ow, cp[q].back(), { "codes" }, &w);
            const int rs = g64 ? stored(r, lp, os, cp[q].back(), { "scale", "scale.1" }, &sc)
                               : stored(r, lp, os, cp[q].back(), { "scale" }, &sc);
            if (rw != (fp8 ? RAD_E_UNSUPPORTED : RAD_OK) ||
                rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED) ||
                (int64_t)w.size() != wbytes || (int64_t)sc.size() != sbytes) {
                ok(head.c_str(), false, "class " + std::to_string(q) + " relayout rc " +
                                        std::to_string(rw) + "/" + std::to_string(rs) + ", " +
                                        std::to_string(w.size()) + " and " +
                                        std::to_string(sc.size()) + " bytes an expert");
                return;
            }
            wq[q].insert(wq[q].end(), w.begin(), w.end());
            ws[q].insert(ws[q].end(), sc.begin(), sc.end());
        }
    }
    void* d_w[4];
    void* d_ws[4];
    for (int q = 0; q < 4; ++q) {
        d_w[q]  = dalloc(wq[q].size());
        d_ws[q] = dalloc(ws[q].size());
        st_memcpy(d_w[q],  wq[q].data(), wq[q].size(), hipMemcpyHostToDevice);
        st_memcpy(d_ws[q], ws[q].data(), ws[q].size(), hipMemcpyHostToDevice);
    }
    auto codes_t = [&](int q) {
        return T(d_w[q], fp8 ? (uint32_t)RAD_F8E4M3 : w5 ? (uint32_t)R4D_DT_MOEW5
                                                         : (uint32_t)R4D_DT_MOEW4,
                 { nec[q], sh.N[q], sh.K[q] });
    };
    auto scale_t = [&](int q) {
        const int64_t kbq = sh.K[q] / group;
        return fp8 ? T(d_ws[q], RAD_BF16, { nec[q], (sh.N[q] + group - 1) / group, kbq })
             : g64 ? T(d_ws[q], RAD_F8E4M3, { nec[q], sh.N[q], sub * kbq })
                   : T(d_ws[q], RAD_BF16, { nec[q], sh.N[q], kbq });
    };

    for (const int64_t Tr : Ts) {
        const int64_t M = Tr / top_k;
        const std::string what = head + " T=" + std::to_string(Tr);
        std::vector<int32_t> off, sorted;
        moe_route(M, top_k, ne, 0x9E3779B97F4A7C15ull ^ (uint64_t)Tr, M > 16, off, sorted);
        const int64_t live = off[(size_t)ne];
        int64_t nrun = 0, longest = 0;
        for (int64_t e = 0; e < ne; ++e) {
            const int64_t n = off[(size_t)(e + 1)] - off[(size_t)e];
            nrun += n > 0;
            longest = std::max(longest, n);
        }

        /* One activation row per routed slot: E4M3 bytes without subnormals or NaN, an f32 scale
         * per (row, group) varying along both. */
        std::vector<uint8_t> aq((size_t)(Tr * K));
        std::vector<float>   as((size_t)(Tr * kb));
        for (int64_t m = 0; m < Tr; ++m) {
            for (int64_t g = 0; g < kb; ++g)
                as[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g + 0.0007f * (float)(m % 11);
            for (int64_t k = 0; k < K; ++k) {
                const uint32_t c = (uint32_t)((m * 13 + k * 5) % 119) + 8;
                aq[(size_t)(m * K + k)] = (uint8_t)(((m + k) & 1) ? (c | 0x80u) : c);
            }
        }
        void* d_a  = dalloc(aq.size());
        void* d_as = dalloc(as.size() * 4);
        void* d_s  = dalloc(sorted.size() * 4);
        void* d_o  = dalloc(off.size() * 4);
        void* d_y  = dalloc((size_t)(Tr * N * 2));
        st_memcpy(d_a,  aq.data(),     aq.size(),         hipMemcpyHostToDevice);
        st_memcpy(d_as, as.data(),     as.size() * 4,     hipMemcpyHostToDevice);
        st_memcpy(d_s,  sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_o,  off.data(),    off.size() * 4,    hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = {
            T(d_a, RAD_F8E4M3, { Tr, K }), T(d_as, RAD_F32, { Tr, kb }), codes_t(0), scale_t(0),
            T(d_s, RAD_I32, { Tr }), T(d_o, RAD_I32, { ne + 1 }), codes_t(1), scale_t(1),
            codes_t(2), scale_t(2), codes_t(3), scale_t(3), T(d_y, RAD_BF16, { Tr, N }),
        };
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", sh.N[0]), PI("K", sh.K[0]), PI("N_1", sh.N[1]),
            PI("K_1", sh.K[1]), PI("N_2", sh.N[2]), PI("K_2", sh.K[2]), PI("N_3", sh.N[3]),
            PI("K_3", sh.K[3]), PI("parts", 1), PI("n_expert", ne), PI("top_k", top_k),
            PI("group", group), PS("a_order", "sorted"), PS("dtype", dt),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int64_t map_bytes = (1 + 2 * (ne + Tr / 16 + 2)) * 4;
        void* d_map = dalloc((size_t)map_bytes);

        /* The reference: sorted row i reads activation row i, its expert e from the offsets. */
        std::vector<double> ref((size_t)(Tr * N), 0.0);
        for (int64_t i = 0; i < live; ++i) {
            if (sorted[(size_t)i] < 0) continue;
            int64_t e = 0;
            while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
            const int q = (int)(e % 4);
            const Planes& p = cp[q][(size_t)(e / 4)];
            const int64_t fc = sh.N[q], Kq = sh.K[q], kbq = Kq / group;
            for (int64_t n = 0; n < fc; ++n) {
                double want = 0.0;
                for (int64_t g = 0; g < kbq; ++g) {
                    const double a_s = (double)as[(size_t)(i * kb + g)];
                    if (fp8) {
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)h_e4m3(p.data[0][(size_t)(n * Kq + k)]) *
                                    (double)h_e4m3(aq[(size_t)(i * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            (n / group) * kbq + g) * a_s;
                    } else if (g64) {
                        const int64_t gw = group / sub;
                        for (int64_t h = 0; h < sub; ++h) {
                            double part = 0.0;
                            for (int64_t k = g * group + h * gw; k < g * group + (h + 1) * gw; ++k)
                                part += (double)tab[code_at(p, n, k)] *
                                        (double)h_e4m3(aq[(size_t)(i * K + k)]);
                            const uint8_t sb = p.data[1][(size_t)(n * p.row_bytes(1) + sub * g + h)];
                            want += part * (double)h_e4m3(sb) * (double)kW4G64Scale * a_s;
                        }
                    } else {
                        double part = 0.0;
                        for (int64_t k = g * group; k < (g + 1) * group; ++k)
                            part += (double)code_at(p, n, k) *
                                    (double)h_e4m3(aq[(size_t)(i * K + k)]);
                        want += part * (double)rad_load_f32(p.data[1].data(), RAD_BF16,
                                                            n * kbq + g) * a_s;
                    }
                }
                ref[(size_t)(i * N + n)] = want;
            }
        }

        /* One pinned launch into stale bytes, returning its output and its last dispatch. `pins`
         * is (form, the sorted operand's decode kernel: -1 the (tile, run) form, 0 the rule, n
         * its column tiles). */
        auto run = [&](const int (&pins)[2], std::vector<uint16_t>* got,
                       const void** fn = nullptr) {
            {
                std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
                st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
            }
            a.scratch = pins[0] == 3 ? nullptr : d_map;
            a.scratch_bytes = pins[0] == 3 ? 0 : map_bytes;
            pl.pin_moe(pins[0]);
            pl.pin_ddec(pins[1]);
            int rc = RAD_OK;
            const void* last = last_dispatch(pl, r, a, &rc);
            st_stream_sync(nullptr);
            pl.pin_moe(0);
            pl.pin_ddec(0);
            if (fn) *fn = last;
            got->assign((size_t)(Tr * N), 0);
            st_memcpy(got->data(), d_y, got->size() * 2, hipMemcpyDeviceToHost);
            return rc;
        };

        /* Every form with the decode kernel off, against the host. */
        std::vector<uint16_t> base, grid;
        const void* fn0 = nullptr;    /* the (tile, run) form's kernel */
        for (int form = 1; form <= 3; ++form) {
            const std::string label = what + " " + kForm[form];
            const int pins[2] = { form, -1 };
            std::vector<uint16_t> got;
            const int rc = run(pins, &got, form == 2 ? &fn0 : nullptr);
            if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
            double worst = 0.0;
            int64_t bad = 0;
            std::string first;
            for (int64_t i = 0; i < Tr; ++i)
                for (int64_t n = 0; n < N; ++n) {
                    const double want = ref[(size_t)(i * N + n)];
                    const double err = std::fabs((double)h_bf16(got[(size_t)(i * N + n)]) - want);
                    if (!(err <= 0.004 * std::fabs(want) + 1e-4)) {
                        if (!bad++)
                            first = ", first at row " + std::to_string(i) + " col " +
                                    std::to_string(n) + ": " +
                                    std::to_string(h_bf16(got[(size_t)(i * N + n)])) + " for " +
                                    std::to_string(want);
                        if (!(err <= worst)) worst = err;
                    }
                }
            ok(label.c_str(), bad == 0, std::to_string(bad) + " of " + std::to_string(Tr * N) +
                                        " elements off, worst " + std::to_string(worst) + first);
            if (form == 2) base = got;
            if (form == 3) grid = got;
        }
        if (base.empty()) continue;

        /* The (tile, run) form on the sorted operand against the uniform prefill grid, byte for
         * byte. */
        if (!grid.empty()) {
            int64_t diff = 0;
            for (size_t i = 0; i < grid.size(); ++i) diff += grid[i] != base[i];
            ok((what + " decode form, byte for byte with the uniform prefill grid").c_str(),
               diff == 0, std::to_string(diff) + " of " + std::to_string(Tr * N) +
                          " elements differ");
        }

        /* The sorted operand's decode kernel under the rule and at several column groups (sixteen,
         * the cap, exceeds the output's five tiles), each against the (tile, run) form's bytes. */
        struct Var { int dn; const char* name; };
        static const Var kVar[] = {
            { 0, "down kernel, the rule's column tiles" },
            { 1, "down kernel, one column tile a block" },
            { 2, "down kernel, two column tiles a block" },
            { 3, "down kernel, three column tiles a block" },
            { 4, "down kernel, four column tiles a block" },
            { 8, "down kernel, eight column tiles a block" },
            { 16, "down kernel, sixteen column tiles a block" },
        };
        for (const Var& v : kVar) {
            const std::string label = what + " decode form, " + v.name + " (" +
                                      std::to_string(nrun) + " runs, longest " +
                                      std::to_string(longest) + ")";
            const int pins[2] = { 2, v.dn };
            std::vector<uint16_t> got;
            const void* fn = nullptr;
            const int rc = run(pins, &got, &fn);
            if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
            if (!fn || fn == fn0) {
                ok(label.c_str(), false, fn ? "dispatched the (tile, run) form's kernel"
                                            : "dispatched nothing");
                continue;
            }
            int64_t diff = 0;
            for (size_t i = 0; i < got.size(); ++i) diff += got[i] != base[i];
            ok(label.c_str(), diff == 0, std::to_string(diff) + " of " + std::to_string(Tr * N) +
                                         " elements differ from the (tile, run) form");
        }

        /* The output seven rows short: Y declared at Tr - 7 rows over the same buffer with the
         * offsets unchanged. Each decode form must write its rows as the full launch did and leave
         * the rest stale. */
        {
            const int64_t Tv = Tr - 7;
            ts[12] = T(d_y, RAD_BF16, { Tv, N });
            for (const int dn : { -1, 0 }) {
                const std::string label = what + " decode form, " +
                                          (dn < 0 ? "(tile, run)" : "down kernel") +
                                          ", an output " + std::to_string(Tv) + " rows long, " +
                                          std::to_string(off[(size_t)ne]) + " routed";
                const int pins[2] = { 2, dn };
                std::vector<uint16_t> got;
                const int rc = run(pins, &got);
                if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
                int64_t diff = 0, past = 0;
                for (int64_t i = 0; i < Tr * N; ++i) {
                    if (i < Tv * N) diff += got[(size_t)i] != base[(size_t)i];
                    else past += got[(size_t)i] != 0x7f7fu;
                }
                ok(label.c_str(), diff == 0 && past == 0,
                   std::to_string(diff) + " of the output's elements differ from the full launch, " +
                       std::to_string(past) + " written past its last row");
            }
            ts[12] = T(d_y, RAD_BF16, { Tr, N });
        }
    }
}

/* Qwen3.8-Flash-Next's down projection at rank 0 of four, narrowed: 512 experts at top 10, K 256
 * for class 0, 128 for classes 1-2 and 384 for class 3 (three K blocks per run), N five 64-column
 * tiles with class 2 one short. The other formats at 64 experts, top 4. */
static const MoeDnShape kMoeDnShapes[] = {
    { "512 experts, top 10", { 320, 320, 256, 320 }, { 256, 128, 128, 384 }, 512, 10 },
    { "64 experts, top 4",   { 320, 320, 256, 320 }, { 256, 128, 128, 384 }, 64, 4 },
};

/* The arranged plane against the format it states, and back.
 *
 * A checker reads `moe_gemm_w4a8`'s weight through the row's inverse, so the inverse must return
 * exactly the planes the relayout was given. The arrangement is checked against r4d_layout.cpp's
 * statement, not either hook: byte b of a packed dword holds element b low and b+4 high, two's
 * complement. The scale plane is stored as is, so its hook must answer RAD_E_UNSUPPORTED
 * ("nothing to arrange"). */
static void moe_w4_unlayout(const Plugin& pl) {
    const char* what = "moe_gemm_w4a8's arranged plane, vs the stated format and its inverse";
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    if (!r || !r->layout || !r->relayout || !r->unrelayout) {
        ok(what, false, "row or hooks absent");
        return;
    }
    const int64_t N = 64, K = 256, group = 128;
    std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group) };

    std::vector<float> src((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            src[(size_t)(n * K + k)] = (float)(((n * 37 + k * 11) % 97) - 48) / 53.0f;
    const Planes p = int4_planes(src, N, K, group, RAD_BF16, false);

    std::vector<uint8_t> wq, ws;
    const int rw = stored(r, lp, 2, p, { "codes" }, &wq);
    const int rs = stored(r, lp, 3, p, { "scale" }, &ws);
    if (rw != RAD_OK) { ok(what, false, "codes relayout rc=" + std::to_string(rw)); return; }
    if (rs != RAD_E_UNSUPPORTED) {
        ok(what, false, "the scale plane is stored as read and must answer RAD_E_UNSUPPORTED");
        return;
    }

    int64_t bad_code = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t pos  = k & 7;
            const int64_t byte = n * (K / 2) + (k >> 3) * 4 + (pos & 3);
            const int nib  = (wq[(size_t)byte] >> (pos >= 4 ? 4 : 0)) & 15;
            const int code = nib >= 8 ? nib - 16 : nib;
            if (code != code_at(p, n, k)) ++bad_code;
        }
    std::string why;
    const bool back = restores(r, lp, 2, p, { "codes" }, wq, &why);
    ok(what, bad_code == 0 && back,
       std::to_string(bad_code) + " code(s) not where the format puts them" +
       (back ? std::string() : "; inverse: " + why));

    /* w5nl64a8h's row against the same statement: the low four bits of every code where w4a8's
     * nibble would be, in a K/2 + K/8-byte row, and the high bit (the sign) of element 4m+b of a
     * 32 at bit 7-m of byte b of that 32's dword, from byte K/2 on. The table is the kernel's, and
     * the inverse restores both planes. */
    const char* what5 = "moe_gemm_w4a8's five-bit arranged plane, vs the stated format and its "
                        "inverse";
    std::vector<RadParam> lp5 = { PI("N", N), PI("K", K), PI("group", group),
                                  PS("dtype", "w5nl64a8h") };
    const Planes p5 = nl64_planes(src, N, K, 64, true);
    std::vector<uint8_t> w5;
    const int r5 = stored(r, lp5, 2, p5, { "codes", "table" }, &w5);
    const int64_t rb = K / 8 * 5;
    if (r5 != RAD_OK || (int64_t)w5.size() != N * rb) {
        ok(what5, false, "codes relayout rc=" + std::to_string(r5) + ", " +
                         std::to_string(w5.size()) + " bytes");
        return;
    }
    int64_t bad5 = 0, neg = 0;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t pos = k & 7;
            const int nib = (w5[(size_t)(n * rb + (k >> 3) * 4 + (pos & 3))] >> (pos >= 4 ? 4 : 0)) & 15;
            const int64_t j = k % 32;
            const int sgn = (w5[(size_t)(n * rb + K / 2 + (k / 32) * 4 + (j & 3))] >>
                             (7 - (j >> 2))) & 1;
            const int code = code_at(p5, n, k);
            neg += code >= 16;
            if ((nib | (sgn << 4)) != code) ++bad5;
        }
    std::string why5;
    const bool back5 = restores(r, lp5, 2, p5, { "codes", "table" }, w5, &why5);
    /* Half the source is negative, so sign bits are exercised. */
    ok(what5, bad5 == 0 && back5 && neg > N * K / 4,
       std::to_string(bad5) + " code(s) not where the format puts them, " + std::to_string(neg) +
       " negative" + (back5 ? std::string() : "; inverse: " + why5));
}
/* ---- the rotated expert format, end to end, against the float it approximates.
 *
 * moe_w4_gemm checks against the planes, so it would pass if the weight and the activation were
 * rotated by different Hadamards. The quantiser's transform (h_fwht here, libquant's
 * fwht_inplace) and r4d_fwht.h's r4d_fwht_wave are written independently (a host loop versus
 * five in-register shuffles across a wave), so the oracle here is the exact dot product of the
 * source weight and activation in original coordinates, against which both formats are measured.
 * A mismatched rotation gives rel_l2 around 5 instead of around 0.1.
 *
 * The printed numbers also compare the two formats, so the source has a few outlier channels:
 * covering an outlier with sixteen levels is the error rotation removes. */
static void moe_w4_rot(const Plugin& pl) {
    const char* what = "moe_gemm_w4a8h/w4nla8h/w4nl64a8h/w5nl64a8h + had_quant_act_fp8 vs the f32 "
                       "they approximate";
    const RadKernelInfo* r  = find(pl, "moe_gemm_w4a8");
    const RadKernelInfo* qp = find(pl, "quant_act_fp8");
    const RadKernelInfo* qh = find(pl, "had_quant_act_fp8");
    if (!r || !r->layout || !r->relayout || !qp || !qh) {
        ok(what, false, "row or hooks absent");
        return;
    }

    const int64_t ne = 3, N = 64, K = 256, group = 128, top_k = 2, Tr = 24;
    const int64_t kb = K / group, M = Tr / top_k;

    /* The source weight and activation in float, with outliers every 37th and 23rd column at six
     * times the body (milder than a real MoE input, enough to separate the grids). */
    std::vector<float>    wsrc((size_t)(ne * N * K));
    std::vector<uint16_t> xbf ((size_t)(M * K));
    std::vector<float>    xsrc((size_t)(M * K));
    for (int64_t e = 0; e < ne; ++e)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k) {
                float v = (float)(((e * 7 + n * 31 + k * 17) % 61) - 30) / 37.0f;
                if (k % 37 == 0) v *= 6.0f;
                wsrc[(size_t)((e * N + n) * K + k)] = v;
            }
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;
            /* Rounded through bf16 here, so the oracle reads the device's value and the residual
             * is the format's. */
            const uint16_t h = h_to_bf16(v);
            xbf [(size_t)(m * K + k)] = h;
            xsrc[(size_t)(m * K + k)] = h_bf16(h);
        }

    /* The same uneven routing the byte-level case uses, tail dropped. */
    std::vector<int32_t> off((size_t)(ne + 1), 0), sorted((size_t)Tr, -1);
    const int64_t live = Tr - 3;
    for (int64_t e = 0; e <= ne; ++e) off[(size_t)e] = (int32_t)((live * e) / ne);
    off[(size_t)ne] = (int32_t)live;
    for (int64_t i = 0; i < live; ++i) sorted[(size_t)i] = (int32_t)((i * 7 + 3) % Tr);

    void* d_x  = dalloc(xbf.size() * 2);
    void* d_s  = dalloc(sorted.size() * 4);
    void* d_o  = dalloc(off.size() * 4);
    st_memcpy(d_x, xbf.data(),    xbf.size() * 2,    hipMemcpyHostToDevice);
    st_memcpy(d_s, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(),    off.size() * 4,    hipMemcpyHostToDevice);

    /* The exact answer in the original coordinates: no rotation, no grid, f64 accumulation. */
    std::vector<double> want((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < live; ++i) {
        const int64_t tok = sorted[(size_t)i] / top_k;
        int64_t e = 0;
        while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)wsrc[(size_t)((e * N + n) * K + k)] *
                       (double)xsrc[(size_t)(tok * K + k)];
            want[(size_t)(i * N + n)] = acc;
        }
    }

    /* Six formats: plain; rotated; rotated with the w4nl codebook (the codes view takes the table
     * and the kernel decodes it); the codebook at E4M3 scales per 64 and per 32 of K (the scale
     * view takes the fixed second level); and the five-bit codebook per 64, with its signs in a
     * separate plane. */
    double rel[6] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
    for (int pass = 0; pass < 6; ++pass) {
        const bool rot = pass >= 1, nl = pass >= 2, g64 = pass >= 3, w5 = pass == 5;
        const int64_t sub = pass == 4 ? 4 : g64 ? 2 : 1;   /* scale groups per 128 */
        const char* dt = w5 ? "w5nl64a8h" : pass == 4 ? "w4nl32a8h" : g64 ? "w4nl64a8h"
                       : nl ? "w4nla8h" : rot ? "w4a8h" : "w4a8";

        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", group), PS("dtype", dt) };
        const int64_t wbytes = w5 ? N * K / 8 * 5 : N * K / 2;
        std::vector<uint8_t>  wq((size_t)(ne * wbytes), 0);
        std::vector<uint16_t> ws((size_t)(ne * N * kb * (sub > 2 ? 2 : 1)), 0);   /* 2 or 4 B/128 */
        for (int64_t e = 0; e < ne; ++e) {
            const std::vector<float> we(wsrc.begin() + e * N * K, wsrc.begin() + (e + 1) * N * K);
            const Planes pe = g64 ? nl64_planes(we, N, K, 128 / sub, w5)
                            : nl  ? nl_planes(we, N, K, group, RAD_BF16)
                                  : int4_planes(we, N, K, group, RAD_BF16, rot);
            std::vector<uint8_t> w, sc;
            RadLayout lw;
            const int rw = nl ? stored(r, lp, 2, pe, { "codes", "table" }, &w, &lw)
                              : stored(r, lp, 2, pe, { "codes" }, &w, &lw);
            const int rs = g64 ? stored(r, lp, 3, pe, { "scale", "scale.1" }, &sc)
                               : stored(r, lp, 3, pe, { "scale" }, &sc);
            if (rw != RAD_OK || rs != (g64 ? RAD_OK : RAD_E_UNSUPPORTED)) {
                ok(what, false, "relayout rc " + std::to_string(rw) + "/" + std::to_string(rs));
                return;
            }
            /* The encoding is the only guard against a mis-served container (arrangement,
             * extents and byte count are identical across formats), so the hook must accept the
             * encoding its dtype names and refuse the other. */
            if (e == 0) {
                const Planes other = int4_planes(we, N, K, group, RAD_BF16, nl || !rot);
                std::vector<uint8_t> scratch;
                const int ro = stored(r, lp, 2, other, { "codes" }, &scratch);
                const char* want_tag = w5 ? "moew5nh" : nl ? "moew4nh" : "moew4h";
                if (ro != RAD_E_DTYPE || !lw.tag ||
                    (std::strstr(lw.tag, want_tag) != nullptr) != rot) {
                    ok(what, false, std::string("the hook does not tell the formats apart: tag ") +
                                    (lw.tag ? lw.tag : "(none)") + ", the other encoding rc " +
                                    std::to_string(ro));
                    return;
                }
            }
            if ((int64_t)w.size() != wbytes) {
                ok(what, false, std::string(dt) + " stored " + std::to_string(w.size()) +
                                " code bytes for " + std::to_string(wbytes));
                return;
            }
            std::memcpy(wq.data() + (size_t)(e * wbytes), w.data(), w.size());
            std::memcpy((uint8_t*)ws.data() + (size_t)e * sc.size(), sc.data(), sc.size());
        }

        /* The activation goes through the device quantiser, whose rotation must match the one
         * the weight's planes were quantised under. */
        void* d_q  = dalloc((size_t)(M * K));
        void* d_qs = dalloc((size_t)(M * kb) * 4);
        std::vector<RadTensor> qt = {
            T(d_x,  RAD_BF16,    { M, K }),
            T(d_q,  RAD_F8E4M3,  { M, K }),
            T(d_qs, RAD_F32,     { M, kb }),
        };
        std::vector<RadParam> qps = {
            PI("M", M), PI("n", K), PI("group", group), PS("dtype", "bf16"),
        };
        RadArgs qa;
        std::memset(&qa, 0, sizeof(qa));
        qa.t = qt.data(); qa.n_t = (int)qt.size();
        qa.p = qps.data(); qa.n_p = (int)qps.size();
        const int qrc = (rot ? qh : qp)->launch(&qa, nullptr);
        st_stream_sync(nullptr);
        if (qrc != RAD_OK) {
            ok(what, false, std::string("quantiser rc=") + std::to_string(qrc)); return;
        }

        void* d_w  = dalloc(wq.size());
        void* d_ws = dalloc(ws.size() * 2);
        void* d_y  = dalloc((size_t)(Tr * N * 2));
        st_memcpy(d_w,  wq.data(), wq.size(),     hipMemcpyHostToDevice);
        st_memcpy(d_ws, ws.data(), ws.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = {
            T(d_q,  RAD_F8E4M3,   { M, K }),
            T(d_qs, RAD_F32,      { M, kb }),
            T(d_w,  w5 ? R4D_DT_MOEW5 : R4D_DT_MOEW4, { ne, N, K }),
            g64 ? T(d_ws, RAD_F8E4M3, { ne, N, sub * kb }) : T(d_ws, RAD_BF16, { ne, N, kb }),
            T(d_s,  RAD_I32,      { Tr }),
            T(d_o,  RAD_I32,      { ne + 1 }),
            T(d_y,  RAD_BF16,     { Tr, N }),
        };
        moe_parity_tables(ts, ne, w5 ? 5 : 4, g64 ? 8 : 16);
        std::vector<RadParam> ps = {
            PI("M", M), PI("N", N), PI("K", K), PI("n_expert", ne), PI("top_k", top_k),
            PI("group", group), PS("a_order", "token"), PS("dtype", dt),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) {
            ok(what, false, std::string("launch rc=") + std::to_string(rc)); return;
        }

        std::vector<uint16_t> got((size_t)(Tr * N));
        st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < live * N; ++i) {
            const double d = (double)h_bf16(got[(size_t)i]) - want[(size_t)i];
            num += d * d;
            den += want[(size_t)i] * want[(size_t)i];
        }
        rel[pass] = den > 0.0 ? std::sqrt(num / den) : -1.0;
    }

    /* 0.6 is deliberately loose, well above the formats' error, so a better quantiser needs no
     * change here. A mis-served container (rotated activation, unrotated weight) measures around 5,
     * since the rotated values are also sqrt(128) larger. */
    bool pass = true;
    for (double x : rel) pass = pass && x >= 0.0 && x < 0.6;
    /* Printed even on a pass, to compare the formats (`ok` shows detail only on failure). A kernel
     * decoding w4nl codes as -8..7 would measure near 1. */
    std::printf("        rel_l2 vs f32: w4a8 %.4f, w4a8h %.4f, w4nla8h %.4f, w4nl64a8h %.4f, "
                "w4nl32a8h %.4f, w5nl64a8h %.4f\n", rel[0], rel[1], rel[2], rel[3], rel[4], rel[5]);
    /* The finer groups must improve on the coarser form (a fold reading the wrong group's scale,
     * or bytes in another order, would not), and the fifth bit must too: a sign plane misread or
     * ignored flips half the signs (near 1), and the wrong row's lands well above w4nl64a8h. */
    pass = pass && rel[3] < rel[2] && rel[4] < rel[3] && rel[5] < 0.75 * rel[3];
    ok(what, pass, "rel_l2 w4a8 " + std::to_string(rel[0]) + ", w4a8h " + std::to_string(rel[1]) +
                   ", w4nla8h " + std::to_string(rel[2]) + ", w4nl64a8h " + std::to_string(rel[3]) +
                   ", w4nl32a8h " + std::to_string(rel[4]) + ", w5nl64a8h " +
                   std::to_string(rel[5]));
}

/* qsa_work, against a host walk of the same batch.
 *
 * A mixed step: a long prefill chunk that completes many blocks and straddles a boundary (its
 * first block takes keys from the tail and the chunk), a short chunk stopping inside a block, a
 * four-row speculative decode completing a block from two tail keys and two of its own, and a
 * fresh three-token prompt ending before its first block. One launch.
 *
 * Pinned on purpose:
 *
 *   The state slots are permuted and wider than the batch ({5, 0, 8, 3} of nine). The scheduler
 *   reorders rows between steps, so a kernel indexing the tail by batch row reads another
 *   sequence's keys.
 *
 *   The ring is 7 and the ratio 4, so `q % ring` and `q % ratio` name different rows for every
 *   tail key.
 *
 *   The three-token prompt: a block bound from truncating division would count block 0 complete
 *   and gather a key past the sequence's end, where the padding must say -1.
 *
 * Also checked: every operand the block-key kernel reads (page, first-token position, the `ratio`
 * staged rows per block) against its source, and -1 in the slots past the real count.
 */
static void qsa_work_case(const Plugin& pl) {
    const char* what = "qsa_work vs a host walk of a mixed prefill/decode batch";
    const RadKernelInfo* r = find(pl, "qsa_work_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t n = 32, ratio = 4, ring = 7, seqs = 4, btw = 64, slots = 9;
    const int32_t cuh[5] = { 0, 22, 25, 29, 32 };
    const int64_t NT = cuh[4];
    /* Not the batch rows, and wider than the batch (see the header). */
    const int32_t sxh[4] = { 5, 0, 8, 3 };
    /* Sequence 0 starts at 11, mid-block, so its first complete block (8..11) takes 8-10 from the
     * tail and 11 from the chunk. Sequence 1 starts on a boundary. Sequence 2 is a four-row
     * speculative decode whose block takes two of each. Sequence 3 is positions 0..2, one short of
     * its first block. */
    const int64_t start[4] = { 11, 60, 130, 0 };

    std::vector<uint16_t> kh((size_t)(NT * n));
    std::vector<int32_t>  ph((size_t)NT);
    for (int64_t s = 0, t = 0; s < seqs; ++s)
        for (int64_t i = 0; t < cuh[s + 1]; ++i, ++t) {
            ph[(size_t)t] = (int32_t)(start[s] + i);
            for (int64_t d = 0; d < n; ++d)
                kh[(size_t)(t * n + d)] = h_to_bf16((float)((t * 31 + d * 17) % 89) / 23.0f);
        }
    /* The tail pool, [slots][ring][n], every slot holding different values. */
    std::vector<uint16_t> th((size_t)(slots * ring * n));
    for (int64_t sl = 0; sl < slots; ++sl)
        for (int64_t row = 0; row < ring; ++row)
            for (int64_t d = 0; d < n; ++d)
                th[(size_t)((sl * ring + row) * n + d)] =
                    h_to_bf16(-1.0f - (float)sl * 3.0f - (float)row / 8.0f - (float)(d % 5) / 64.0f);
    std::vector<int32_t> bth((size_t)(seqs * btw));
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t b = 0; b < btw; ++b)
            bth[(size_t)(s * btw + b)] = (int32_t)((s * 211 + b * 13 + 5) % 4096);

    /* The host walk. */
    struct W { int64_t s, b; };
    std::vector<W> want;
    for (int64_t s = 0; s < seqs; ++s) {
        const int64_t p0 = ph[(size_t)cuh[s]], p1 = ph[(size_t)(cuh[s + 1] - 1)];
        const int64_t lo = p0 / ratio;                /* first block ending at or after p0 */
        const int64_t hi = (p1 + 1) / ratio - 1;      /* last block ending at or before p1 */
        for (int64_t b = lo; b <= hi; ++b) want.push_back({ s, b });
    }
    const int64_t W_ = (NT + ratio - 1) / ratio + seqs;     /* the host bound */

    void* d_k  = dalloc(kh.size() * 2);
    void* d_p  = dalloc(ph.size() * 4);
    void* d_cu = dalloc(sizeof cuh);
    void* d_tl = dalloc(th.size() * 2);
    void* d_sx = dalloc(sizeof sxh);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_st = dalloc((size_t)(W_ * ratio * n) * 2);
    void* d_pg = dalloc((size_t)W_ * 4);
    void* d_bp = dalloc((size_t)W_ * 4);
    void* d_nc = dalloc((size_t)seqs * 4);
    st_memcpy(d_k,  kh.data(),  kh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_p,  ph.data(),  ph.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh,        sizeof cuh,     hipMemcpyHostToDevice);
    st_memcpy(d_tl, th.data(),  th.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_sx, sxh,        sizeof sxh,     hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { NT, n }),
        T(d_p,  RAD_I32,  { NT }),
        T(d_cu, RAD_I32,  { seqs + 1 }),
        T(d_tl, RAD_BF16, { slots, 1, ring, n }),
        T(d_sx, RAD_I32,  { seqs, 1 }),
        T(d_bt, RAD_I32,  { seqs, btw }),
        T(nullptr, RAD_I32, { 3, NT }),          /* rope: absent, the one-component form */
        T(d_st, RAD_BF16, { W_ * ratio, n }),
        T(d_pg, RAD_I32,  { W_ }),
        T(d_bp, RAD_I32,  { W_ }),
        T(d_nc, RAD_I32,  { seqs }),
    };
    std::vector<RadParam> ps = {
        PI("M", NT), PI("n", n), PI("ratio", ratio), PI("ring", ring), PI("seqs", seqs),
        PI("work", W_), PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> gs((size_t)(W_ * ratio * n));
    std::vector<int32_t>  gp((size_t)W_), gb((size_t)W_);
    st_memcpy(gs.data(), d_st, gs.size() * 2, hipMemcpyDeviceToHost);
    st_memcpy(gp.data(), d_pg, gp.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(gb.data(), d_bp, gb.size() * 4, hipMemcpyDeviceToHost);
    std::vector<int32_t> gnc((size_t)seqs);
    st_memcpy(gnc.data(), d_nc, gnc.size() * 4, hipMemcpyDeviceToHost);

    /* Complete blocks per sequence, (last position + 1) / ratio, not the blocks completed this
     * step (the two agree only for a sequence starting at zero). */
    bool nc_ok = true;
    for (int64_t s = 0; s < seqs; ++s) {
        const int32_t w2 = (int32_t)((ph[(size_t)(cuh[s + 1] - 1)] + 1) / ratio);
        if (gnc[(size_t)s] != w2) nc_ok = false;
    }
    bool page_ok = true, pos_ok = true, key_ok = true, pad_ok = true;
    long from_tail = 0, ring_differs = 0;
    for (int64_t w = 0; w < W_; ++w) {
        if (w >= (int64_t)want.size()) { if (gp[(size_t)w] != -1) pad_ok = false; continue; }
        const int64_t s = want[(size_t)w].s, b = want[(size_t)w].b;
        if (gp[(size_t)w] != bth[(size_t)(s * btw + b)]) page_ok = false;
        if (gb[(size_t)w] != (int32_t)(b * ratio)) pos_ok = false;
        const int64_t p0 = ph[(size_t)cuh[s]];
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t q = b * ratio + i;
            const uint16_t* src;
            if (q >= p0) src = kh.data() + (size_t)((cuh[s] + (q - p0)) * n);
            else {
                src = th.data() + (size_t)((sxh[s] * ring + (q % ring)) * n);
                ++from_tail;
                if (q % ring != q % ratio) ++ring_differs;
            }
            for (int64_t d = 0; d < n; ++d)
                if (gs[(size_t)((w * ratio + i) * n + d)] != src[(size_t)d]) key_ok = false;
        }
    }
    /* Under M-RoPE, with the step's rotary planes: `bpos` is [work, 3]. A block whose first
     * token is in this step reads that token's three components; one committed earlier is text,
     * at its index plus the shift of the sequence's first row. Shifts differ per sequence (as
     * after a picture), and the components differ from each other. */
    const int32_t shift[4] = { 0, -9, 17, 5 };
    std::vector<int32_t> rh((size_t)(3 * NT));
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t t = cuh[s]; t < cuh[s + 1]; ++t) {
            const int32_t p = ph[(size_t)t] + shift[s];
            rh[(size_t)t] = p;
            rh[(size_t)(NT + t)] = p + 3 + (int32_t)(t % 4);
            rh[(size_t)(2 * NT + t)] = p + 11 + (int32_t)(t % 7);
        }
    void* d_rp = dalloc(rh.size() * 4);
    void* d_b3 = dalloc((size_t)W_ * 3 * 4);
    st_memcpy(d_rp, rh.data(), rh.size() * 4, hipMemcpyHostToDevice);
    std::vector<int32_t> sentinel((size_t)W_ * 3, -777);
    st_memcpy(d_b3, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> tr = ts;
    tr[6] = T(d_rp, RAD_I32, { 3, NT });
    tr[9] = T(d_b3, RAD_I32, { W_, 3 });
    RadArgs ar = a;
    ar.t = tr.data();
    bool mc_ok = r->launch(&ar, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<int32_t> g3((size_t)W_ * 3);
    st_memcpy(g3.data(), d_b3, g3.size() * 4, hipMemcpyDeviceToHost);
    long mc_text = 0;
    for (int64_t w = 0; w < W_ && mc_ok; ++w) {
        for (int c = 0; c < 3; ++c) {
            int32_t e = 0;
            if (w < (int64_t)want.size()) {
                const int64_t s = want[(size_t)w].s, b = want[(size_t)w].b;
                const int64_t t0 = cuh[s], p0 = ph[(size_t)t0], f = b * ratio;
                if (f >= p0) e = rh[(size_t)(c * NT + t0 + (f - p0))];
                else { e = (int32_t)(f + (rh[(size_t)t0] - p0)); if (c == 0) ++mc_text; }
            }
            if (g3[(size_t)(w * 3 + c)] != e) mc_ok = false;
        }
    }
    /* And a pass without the planes, as a text-only step: [work, 1] at a row pitch of 3.
     * Component 0 is written; the other two columns keep their values. */
    st_memcpy(d_b3, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> tt = ts;
    RadTensor col = T(d_b3, RAD_I32, { W_, 1 });
    col.stride[0] = 3;
    tt[9] = col;
    RadArgs at = a;
    at.t = tt.data();
    bool col_ok = r->launch(&at, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    st_memcpy(g3.data(), d_b3, g3.size() * 4, hipMemcpyDeviceToHost);
    for (int64_t w = 0; w < W_ && col_ok; ++w) {
        if (g3[(size_t)(w * 3)] != gb[(size_t)w]) col_ok = false;
        if (g3[(size_t)(w * 3 + 1)] != -777 || g3[(size_t)(w * 3 + 2)] != -777) col_ok = false;
    }
    std::printf("        M-RoPE: [work, 3] positions %s (%ld from committed text); one column at "
                "pitch 3 %s\n", mc_ok ? "ok" : "NO", mc_text, col_ok ? "ok" : "NO");

    /* `from_tail` requires a straddling block, `ring_differs` that ring and ratio rows differ. */
    const bool pass = page_ok && pos_ok && key_ok && pad_ok && nc_ok && from_tail > 0 &&
                      ring_differs > 0 && !want.empty() && mc_ok && mc_text > 0 && col_ok;
    std::printf("        %zu blocks (%ld keys from the tail, %ld at a row the old modulus would "
                "have missed), page %s, pos %s, keys %s, padding %s, nc %s\n", want.size(),
                from_tail, ring_differs, page_ok ? "ok" : "NO", pos_ok ? "ok" : "NO",
                key_ok ? "ok" : "NO", pad_ok ? "ok" : "NO", nc_ok ? "ok" : "NO");
    ok(what, pass, std::string("page=") + (page_ok ? "1" : "0") + " pos=" + (pos_ok ? "1" : "0") +
                   " key=" + (key_ok ? "1" : "0") + " pad=" + (pad_ok ? "1" : "0") +
                   " tail_keys=" + std::to_string(from_tail) +
                   " ring_rows=" + std::to_string(ring_differs));
}

/* qsa_tail_store, against a token-by-token loop.
 *
 * Positions p and p + ring share a row, so a long prefill chunk has several tokens per row and
 * the kernel may write only the last. The fixture has a long chunk, a short one and a four-row
 * speculative decode; the oracle is the serial loop (last writer wins). Writing every token would
 * race on the long chunk, so the case also runs twice and requires identical bytes.
 *
 * Pinned on purpose: the state slots are permuted and wider than the batch ({5, 0, 8} of nine),
 * since the row is step-local and the slot belongs to the sequence; and the ring is 7 with ratio
 * 4, so `pos % ring` and `pos % ratio` disagree.
 *
 * Rows nothing targets must keep their values.
 */
static void qsa_tail_store_case(const Plugin& pl) {
    const char* what = "qsa_tail_store vs the serial last-writer-wins loop";
    const RadKernelInfo* r = find(pl, "qsa_tail_store_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t n = 64, ratio = 4, ring = 7, seqs = 3, slots = 9;
    /* A long prefill chunk, a short one, and a four-row speculative decode: 19 + 3 + 4. */
    const int32_t cuh[4] = { 0, 19, 22, 26 };
    const int64_t NT = cuh[3];
    const int32_t sxh[3] = { 5, 0, 8 };
    /* Different starts, two of them mid-block, since `pos % ring` is the row. */
    const int64_t start[3] = { 40, 7, 131 };

    std::vector<uint16_t> kh((size_t)(NT * n));
    std::vector<int32_t>  ph((size_t)NT);
    for (int64_t s = 0, t = 0; s < seqs; ++s)
        for (int64_t i = 0; t < cuh[s + 1]; ++i, ++t) {
            ph[(size_t)t] = (int32_t)(start[s] + i);
            for (int64_t d = 0; d < n; ++d)
                kh[(size_t)(t * n + d)] = h_to_bf16((float)((t * 13 + d * 7) % 97) / 31.0f);
        }

    /* A pre-filled pool; untargeted rows, including the six unnamed slots, must keep it. */
    std::vector<uint16_t> pre((size_t)(slots * ring * n));
    for (size_t i = 0; i < pre.size(); ++i) pre[i] = h_to_bf16(-3.5f - (float)(i % 5));
    std::vector<uint16_t> want = pre;
    for (int64_t s = 0; s < seqs; ++s)
        for (int64_t t = cuh[s]; t < cuh[s + 1]; ++t) {          /* in order: last writer wins */
            const int64_t row = ph[(size_t)t] % ring;
            for (int64_t d = 0; d < n; ++d)
                want[(size_t)((sxh[s] * ring + row) * n + d)] = kh[(size_t)(t * n + d)];
        }

    void* d_k  = dalloc(kh.size() * 2);
    void* d_p  = dalloc(ph.size() * 4);
    void* d_cu = dalloc(sizeof cuh);
    void* d_tl = dalloc(pre.size() * 2);
    void* d_sx = dalloc(sizeof sxh);
    st_memcpy(d_k,  kh.data(), kh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_p,  ph.data(), ph.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh,       sizeof cuh,    hipMemcpyHostToDevice);
    st_memcpy(d_sx, sxh,       sizeof sxh,    hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_k,  RAD_BF16, { NT, n }),
        T(d_p,  RAD_I32,  { NT }),
        T(d_cu, RAD_I32,  { seqs + 1 }),
        T(d_tl, RAD_BF16, { slots, 1, ring, n }),
        T(d_sx, RAD_I32,  { seqs, 1 }),
    };
    std::vector<RadParam> ps = {
        PI("M", NT), PI("n", n), PI("ratio", ratio), PI("ring", ring), PI("seqs", seqs),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    std::vector<uint16_t> got[2];
    for (int pass = 0; pass < 2; ++pass) {
        st_memcpy(d_tl, pre.data(), pre.size() * 2, hipMemcpyHostToDevice);
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign(pre.size(), 0);
        st_memcpy(got[pass].data(), d_tl, got[pass].size() * 2, hipMemcpyDeviceToHost);
    }

    long kept = 0, kept_want = 0;
    for (size_t i = 0; i < pre.size(); i += (size_t)n) {
        if (got[0][i] == pre[i]) ++kept;
        if (want[i] == pre[i]) ++kept_want;
    }
    /* Slots the batch does not name must be untouched; a batch-row index would write them.
     * Counted separately from the aggregate above. */
    bool other_ok = true;
    for (int64_t sl = 0; sl < slots; ++sl) {
        bool named = false;
        for (int64_t s = 0; s < seqs; ++s) if (sxh[s] == sl) named = true;
        if (named) continue;
        for (int64_t i = 0; i < ring * n; ++i)
            if (got[0][(size_t)(sl * ring * n + i)] != pre[(size_t)(sl * ring * n + i)])
                other_ok = false;
    }
    const bool match = got[0] == want;
    const bool det = got[0] == got[1];
    const bool pass = match && det && other_ok && kept == kept_want && kept_want > 0;
    std::printf("        rows %s, untouched %ld of %lld (want %ld), unnamed slots %s, "
                "two runs identical %s\n", match ? "ok" : "DIFFER", kept,
                (long long)(slots * ring), kept_want, other_ok ? "ok" : "WRITTEN",
                det ? "ok" : "NO");
    ok(what, pass, std::string("match=") + (match ? "1" : "0") + " det=" + (det ? "1" : "0") +
                   " other=" + (other_ok ? "1" : "0") +
                   " kept=" + std::to_string(kept) + "/" + std::to_string(kept_want));
}

/* sample_argmax, against a host scan.
 *
 * Checks the tie rule and both load bodies. Greedy breaks ties to the lowest token id (so greedy
 * and top-1 agree); the kernel packs (key, ~id) into one word, monotone in the key and
 * anti-monotone in the id. Each row's maximum occurs at four ids.
 *
 * The float4 body runs when both row base and row pitch are 16-byte aligned, the scalar body
 * otherwise. `n_vocab` is not a multiple of four, so the vector body has a two-entry scalar tail,
 * which holds one of the tied ids. Pass 0's pitch is a multiple of four, pass 1's is not (every
 * row after the first misaligned); both must give the same answer.
 */
static void sample_argmax_case(const Plugin& pl) {
    const char* what = "sample_argmax vs a host scan, with ties and both load bodies";
    const RadKernelInfo* r = find(pl, "sample_argmax_f32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 5, V = 24578;          /* 4 * 6144 + 2 */
    std::vector<std::vector<float>> rows((size_t)M, std::vector<float>((size_t)V, 0.0f));
    std::vector<int32_t> want((size_t)M, 0);
    uint32_t rs = 0x1234567u;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t i = 0; i < V; ++i) {
            rs = rs * 1664525u + 1013904223u;
            rows[(size_t)m][(size_t)i] = (float)((int32_t)((rs >> 8) % 20001u) - 10000) * 0.001f;
        }
        /* Four ids hold the maximum; the last is in the vector body's tail. */
        const int64_t ties[4] = { 7, V / 3, V - 6, V - 1 };
        for (int k = 0; k < 4; ++k) rows[(size_t)m][(size_t)ties[k]] = 11.0f + (float)m;
        want[(size_t)m] = (int32_t)ties[0];
    }

    bool pass_ok[2] = { true, true };
    std::vector<int32_t> got[2];
    for (int pass = 0; pass < 2; ++pass) {
        const int64_t P = (pass == 0) ? V + 2 : V + 1;   /* 24580 vector, 24579 scalar */
        std::vector<float> flat((size_t)(M * P), 0.0f);
        for (int64_t m = 0; m < M; ++m)
            std::memcpy(&flat[(size_t)(m * P)], rows[(size_t)m].data(), (size_t)V * 4);
        void* d_lg = dalloc(flat.size() * 4);
        void* d_pa = dalloc(64);
        void* d_tk = dalloc((size_t)M * 4);
        st_memcpy(d_lg, flat.data(), flat.size() * 4, hipMemcpyHostToDevice);
        std::vector<RadTensor> ts = { T(d_lg, RAD_F32, { M, V }), T(d_pa, RAD_F32, { 16 }),
                                      T(d_tk, RAD_I32, { M }) };
        ts[0].stride[0] = P;
        std::vector<RadParam> ps = { PI("M", M), PI("n_vocab", V) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign((size_t)M, -1);
        st_memcpy(got[pass].data(), d_tk, (size_t)M * 4, hipMemcpyDeviceToHost);
        for (int64_t m = 0; m < M; ++m)
            if (got[pass][(size_t)m] != want[(size_t)m]) pass_ok[pass] = false;
    }
    const bool agree = got[0] == got[1];
    if (!g_case)
        std::printf("  argmax  aligned=%s unaligned=%s agree=%s  (row 0 got %d want %d)\n",
                    pass_ok[0] ? "ok" : "NO", pass_ok[1] ? "ok" : "NO", agree ? "ok" : "NO",
                    got[0].empty() ? -1 : got[0][0], want[0]);
    ok(what, pass_ok[0] && pass_ok[1] && agree,
       std::string("vec=") + (pass_ok[0] ? "1" : "0") + " scalar=" + (pass_ok[1] ? "1" : "0") +
       " agree=" + (agree ? "1" : "0"));
}

/* qsa_select, against a host sort of the same scores.
 *
 * Focused on ties and the tail. Block scores (sums of relus over a shared grid) tie often, so the
 * fixture gives the threshold value to more blocks than there are slots left, and the oracle is a
 * full sort by (score descending, index ascending), the rule the kernel's prefix counts implement.
 *
 * Checked:
 *   the set matches a host sort;
 *   the partial page is last, so the attention's causal bound masks inside it only;
 *   a page id of -1 inside the scored range is never selected, whatever its score;
 *   two runs give byte-identical tables (atomic placement would not).
 */
static void qsa_select_case(const Plugin& pl) {
    const char* what = "qsa_select vs a host sort, with ties, a tail and a dead page";
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 6, nb = 900, topk = 64, pages = 4096, ratio = 4;

    std::vector<float>   sc((size_t)(M * nb), 0.0f);
    std::vector<int32_t> bt((size_t)(M * nb), -1), nc((size_t)M, 0);
    for (int64_t m = 0; m < M; ++m) {
        /* Rows 0 and 1 have at most topk live blocks (the fast path, all selected); the rest run
         * the radix. Rows alternate on whether a partial page exists. */
        const int64_t live = (m == 0) ? 9 : (m == 1) ? topk : (100 + m * 137);
        nc[(size_t)m] = (int32_t)live;
        for (int64_t b = 0; b < nb; ++b) {
            const size_t i = (size_t)(m * nb + b);
            if (b < live) {
                /* A coarse grid: 40 values over hundreds of blocks puts many on the threshold. */
                sc[i] = (float)((b * 17 + m * 7) % 40) * 0.25f;
                bt[i] = (int32_t)((b * 31 + m * 11) % pages);
                /* A dead page with a winning score, inside the scored range. */
                if (b == live / 3) { sc[i] = 1e6f; bt[i] = -1; }
            } else if (b == live) {
                bt[i] = (m % 2 == 0) ? (int32_t)((m * 97 + 5) % pages) : -1;   /* the tail */
            }
        }
    }

    /* The oracle: sort the live, non-dead blocks by (score desc, index asc), take topk, then the
     * partial page. */
    std::vector<std::vector<int32_t>> want((size_t)M);
    std::vector<int32_t> want_n((size_t)M, 0), want_tail((size_t)M, -1);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nc[(size_t)m];
        std::vector<std::pair<float, int64_t>> v;
        for (int64_t b = 0; b < live; ++b)
            if (bt[(size_t)(m * nb + b)] >= 0) v.push_back({ sc[(size_t)(m * nb + b)], b });
        std::stable_sort(v.begin(), v.end(), [](const std::pair<float, int64_t>& a,
                                                const std::pair<float, int64_t>& b) {
            return a.first > b.first;                       /* stable => index ascending on ties */
        });
        const size_t take = v.size() < (size_t)topk ? v.size() : (size_t)topk;
        for (size_t i = 0; i < take; ++i)
            want[(size_t)m].push_back(bt[(size_t)(m * nb + v[i].second)]);
        want_tail[(size_t)m] = bt[(size_t)(m * nb + live)];
        want_n[(size_t)m] = (int32_t)take + (want_tail[(size_t)m] >= 0 ? 1 : 0);
    }

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    /* The query's own position: the last token of the partial page when there is one, and the
     * last token of the last complete block when there is not. */
    std::vector<int32_t> pos((size_t)M);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nc[(size_t)m];
        pos[(size_t)m] = (int32_t)(want_tail[(size_t)m] >= 0 ? live * ratio + (m % 3)
                                                             : live * ratio - 1);
    }
    /* One query per sequence (decode shape, `seq(m) == m`), as the oracle assumes;
     * qsa_select_prefill_case covers the join. */
    std::vector<int32_t> cuh((size_t)(M + 1));
    for (int64_t i = 0; i <= M; ++i) cuh[(size_t)i] = (int32_t)i;

    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }),
        T(d_bt, RAD_I32, { M, nb }),
        T(d_nc, RAD_I32, { M }),
        T(d_po, RAD_I32, { M }),
        T(d_cu, RAD_I32, { M + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }),
        T(d_ns, RAD_I32, { M }),
        T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", M) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    std::vector<int32_t> got[2], gn[2], gs[2];
    for (int pass = 0; pass < 2; ++pass) {
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        got[pass].assign((size_t)(M * (topk + 1)), 0);
        gn[pass].assign((size_t)M, 0);
        st_memcpy(got[pass].data(), d_se, got[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(gn[pass].data(),  d_ns, gn[pass].size() * 4,  hipMemcpyDeviceToHost);
        gs[pass].assign((size_t)M, 0);
        st_memcpy(gs[pass].data(),  d_su, gs[pass].size() * 4,  hipMemcpyDeviceToHost);
    }

    bool set_ok = true, tail_ok = true, dead_ok = true, n_ok = true;
    for (int64_t m = 0; m < M; ++m) {
        const int32_t ns = gn[0][(size_t)m];
        if (ns != want_n[(size_t)m]) { n_ok = false; continue; }
        std::vector<int32_t> mine(got[0].begin() + (size_t)(m * (topk + 1)),
                                  got[0].begin() + (size_t)(m * (topk + 1) + ns));
        /* The tail is last, by contract. */
        if (want_tail[(size_t)m] >= 0 && (ns == 0 || mine.back() != want_tail[(size_t)m]))
            tail_ok = false;
        if (want_tail[(size_t)m] >= 0) mine.pop_back();
        for (int32_t v : mine) if (v < 0) dead_ok = false;
        std::vector<int32_t> ref = want[(size_t)m];
        std::sort(mine.begin(), mine.end());
        std::sort(ref.begin(), ref.end());
        if (mine != ref) set_ok = false;
    }
    /* The causal bound for the attention: the flat KV is `nsel` pages of `ratio`, partial page
     * last, so the query sits at `(nsel-1)*ratio + pos%ratio` and the length is one past it. Off
     * by one, the attention would skip the query's own token or read beyond it. */
    bool su_ok = true;
    for (int64_t m = 0; m < M; ++m) {
        const int32_t ns = gn[0][(size_t)m];
        const int32_t want_su = ns == 0 ? 0
                              : (int32_t)((ns - 1) * ratio + (pos[(size_t)m] % ratio) + 1);
        if (gs[0][(size_t)m] != want_su) su_ok = false;
    }
    const bool det = got[0] == got[1] && gn[0] == gn[1] && gs[0] == gs[1];
    const bool pass = set_ok && tail_ok && dead_ok && n_ok && su_ok && det;
    std::printf("        set %s, tail last %s, dead page excluded %s, count %s, seqused %s, "
                "two runs identical %s\n", set_ok ? "ok" : "DIFFERS", tail_ok ? "ok" : "NO",
                dead_ok ? "ok" : "NO", n_ok ? "ok" : "NO", su_ok ? "ok" : "NO",
                det ? "ok" : "NO");
    ok(what, pass, std::string("set=") + (set_ok ? "1" : "0") + " tail=" + (tail_ok ? "1" : "0") +
                   " dead=" + (dead_ok ? "1" : "0") + " n=" + (n_ok ? "1" : "0") +
                   " su=" + (su_ok ? "1" : "0") + " det=" + (det ? "1" : "0"));
}

/* qsa_select at prefill: many queries over one history, each bounded by its own position.
 *
 * At decode the sequence's complete-block count equals the query's and `bt[nc]` is -1. Mid-chunk,
 * the sequence's count runs to the chunk's end while the query's runs to the query, and the query's
 * page holds later tokens of the same chunk; appending it with `seqused` past the query would let
 * the query attend to its own future (an implausibly good prefill).
 *
 * Checked: the per-query bound `n = min(nc[seq], (pos+1)/ratio)`, and that the partial page exists
 * exactly when `(pos + 1) % ratio != 0`.
 */
static void qsa_select_prefill_case(const Plugin& pl) {
    const char* what = "qsa_select at prefill: per-query bound and the partial-page rule";
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t S = 2, nb = 96, topk = 5, pages = 512, ratio = 4;
    /* Two chunks of different lengths over two histories, as the engine builds them, so a query's
     * row index and position differ. */
    const int64_t q0 = 9, q1 = 14, M = q0 + q1;
    const int64_t base[2] = { 137, 4 };          /* each sequence's first position in the chunk */

    std::vector<int32_t> cuh = { 0, (int32_t)q0, (int32_t)(q0 + q1) };
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < q0; ++i) pos[(size_t)i] = (int32_t)(base[0] + i);
    for (int64_t i = 0; i < q1; ++i) pos[(size_t)(q0 + i)] = (int32_t)(base[1] + i);

    /* `nc` is the sequence's count after the chunk: every block its last token completed. */
    std::vector<int32_t> nc((size_t)S);
    nc[0] = (int32_t)((base[0] + q0) / ratio);
    nc[1] = (int32_t)((base[1] + q1) / ratio);

    /* The table and scores are per sequence, the same for every query in it; queries differ only
     * in how much of the row they may see. */
    std::vector<float>   sc((size_t)(M * nb), 0.0f);
    std::vector<int32_t> bt((size_t)(S * nb), -1);
    for (int64_t s = 0; s < S; ++s)
        for (int64_t b = 0; b < nb; ++b)
            /* Live up to the block containing the chunk's last token, so `bt[n]` is a real page
             * for every query with a partial one. */
            bt[(size_t)(s * nb + b)] =
                b <= (base[s] + (s == 0 ? q0 : q1) - 1) / ratio ? (int32_t)((b * 29 + s * 7) % pages)
                                                                : -1;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b)
            sc[(size_t)(m * nb + b)] = (float)((b * 23 + m * 5) % 37) * 0.5f;

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    st_memcpy(d_sc, sc.data(),  sc.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(),  bt.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(),  nc.size() * 4,  hipMemcpyHostToDevice);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }),
        T(d_bt, RAD_I32, { S, nb }),
        T(d_nc, RAD_I32, { S }),
        T(d_po, RAD_I32, { M }),
        T(d_cu, RAD_I32, { S + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }),
        T(d_ns, RAD_I32, { M }),
        T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", S) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<int32_t> got((size_t)(M * (topk + 1))), gn((size_t)M), gs((size_t)M);
    st_memcpy(got.data(), d_se, got.size() * 4, hipMemcpyDeviceToHost);
    st_memcpy(gn.data(),  d_ns, gn.size() * 4,  hipMemcpyDeviceToHost);
    st_memcpy(gs.data(),  d_su, gs.size() * 4,  hipMemcpyDeviceToHost);

    bool set_ok = true, n_ok = true, su_ok = true, future_ok = true, varied = false;
    int64_t seen_partial = 0, seen_none = 0;
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s   = m < q0 ? 0 : 1;
        const int64_t p   = pos[(size_t)m];
        const int64_t own = (p + 1) / ratio;
        int64_t n = nc[(size_t)s] < nb ? nc[(size_t)s] : nb;
        if (own < n) n = own;
        if (own != nc[(size_t)s]) varied = true;      /* the bound is doing something */

        std::vector<std::pair<float, int64_t>> v;
        for (int64_t b = 0; b < n; ++b)
            if (bt[(size_t)(s * nb + b)] >= 0) v.push_back({ sc[(size_t)(m * nb + b)], b });
        std::stable_sort(v.begin(), v.end(), [](const std::pair<float, int64_t>& x,
                                                const std::pair<float, int64_t>& y) {
            return x.first > y.first;
        });
        const size_t take = v.size() < (size_t)topk ? v.size() : (size_t)topk;
        std::vector<int32_t> ref;
        for (size_t i = 0; i < take; ++i) ref.push_back(bt[(size_t)(s * nb + v[i].second)]);

        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? bt[(size_t)(s * nb + n)] : -1;
        if (part >= 0) ++seen_partial; else ++seen_none;
        const int32_t want_n = (int32_t)take + (part >= 0 ? 1 : 0);
        if (gn[(size_t)m] != want_n) { n_ok = false; continue; }

        std::vector<int32_t> mine(got.begin() + (size_t)(m * (topk + 1)),
                                  got.begin() + (size_t)(m * (topk + 1) + want_n));
        if (part >= 0) {
            if (mine.back() != part) set_ok = false;
            mine.pop_back();
        }
        std::sort(mine.begin(), mine.end());
        std::sort(ref.begin(), ref.end());
        if (mine != ref) set_ok = false;

        const int32_t want_su = want_n == 0 ? 0
                              : (int32_t)((want_n - 1) * ratio + (p % ratio) + 1);
        if (gs[(size_t)m] != want_su) su_ok = false;
        /* The flat KV is `nsel` pages of `ratio`; `seqused` must not reach past the query's own
         * token in the last one. */
        if (want_n > 0 && gs[(size_t)m] > (int32_t)((want_n - 1) * ratio + (p % ratio) + 1))
            future_ok = false;
    }
    const bool pass = set_ok && n_ok && su_ok && future_ok && varied &&
                      seen_partial > 0 && seen_none > 0;
    std::printf("        set %s, count %s, seqused %s, no future page %s; %lld queries with a "
                "partial page, %lld without\n", set_ok ? "ok" : "DIFFERS", n_ok ? "ok" : "NO",
                su_ok ? "ok" : "NO", future_ok ? "ok" : "NO",
                (long long)seen_partial, (long long)seen_none);
    ok(what, pass, std::string("set=") + (set_ok ? "1" : "0") + " n=" + (n_ok ? "1" : "0") +
                   " su=" + (su_ok ? "1" : "0") + " fut=" + (future_ok ? "1" : "0") +
                   " varied=" + (varied ? "1" : "0"));
}

/* qsa_select's wide-launch form against its single-workgroup form, byte for byte: table, count
 * and seqused. Two sequences, 120 queries each bounded by its own position (some below topk), dead
 * pages and -inf scores scattered through the history, and five score distributions: continuous;
 * coarse with heavy ties; two packed into one first-digit bin, one fitting the LDS list and one
 * past it (the fallback); and fine ties, where the sampled list is short and ends in a tie. */
static void qsa_select_wide_case(const Plugin& pl) {
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r || !pl.pin_qsel) { ok("qsa_select wide form", false, "row or hook absent"); return; }
    const int64_t S = 2, nb = 24000, topk = 512, ratio = 4, pages = 1 << 20;
    const int64_t q0 = 70, q1 = 50, M = q0 + q1;
    /* The second sequence's queries see 300 to 337 blocks, fewer than topk. */
    const int64_t base[2] = { 4 * 23000, 4 * 300 };
    std::vector<int32_t> cuh = { 0, (int32_t)q0, (int32_t)M };
    std::vector<int32_t> pos((size_t)M);
    for (int64_t i = 0; i < q0; ++i) pos[(size_t)i] = (int32_t)(base[0] + 37 * i);
    for (int64_t i = 0; i < q1; ++i) pos[(size_t)(q0 + i)] = (int32_t)(base[1] + 3 * i);
    std::vector<int32_t> nc = { (int32_t)std::min<int64_t>((base[0] + 37 * q0) / ratio, nb),
                                (int32_t)((base[1] + 3 * q1) / ratio) };
    uint64_t x = 0x243F6A8885A308D3ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<int32_t> bt((size_t)(S * nb));
    for (auto& v : bt) v = (rnd() % 97 == 0) ? -1 : (int32_t)(rnd() % pages);
    static const char* const kDist[] = { "continuous", "coarse, heavy ties",
                                         "one first-digit bin, in LDS",
                                         "one first-digit bin, past LDS",
                                         "fine ties at the threshold" };
    for (int dist = 0; dist < 5; ++dist) {
        std::vector<float> sc((size_t)(M * nb));
        for (auto& v : sc) {
            const uint64_t u = rnd();
            if (u % 211 == 0) { v = -INFINITY; continue; }
            switch (dist) {
                case 0: v = (float)(u % 1000003) / 1000.0f; break;
                case 1: v = (float)(u % 23) * 0.25f; break;
                /* 1.0 to 1.06 is one first-digit bin (1/16 of an octave); a few thousand scores
                 * fall in it. */
                case 2: v = (u % 8 == 0) ? 1.0f + (float)(u % 60000) * 1e-6f
                                         : (float)(u % 1000) / 2000.0f; break;
                case 3: v = 1.0f + (float)(u % 60000) * 1e-6f; break;
                /* ~5000 distinct scores, a few times each: the sampled list is short and the
                 * threshold is a tie. */
                default: v = (float)(u % 4999) * 0.01f; break;
            }
        }
        void* d_sc = dalloc(sc.size() * 4);
        void* d_bt = dalloc(bt.size() * 4);
        void* d_nc = dalloc(nc.size() * 4);
        void* d_po = dalloc(pos.size() * 4);
        void* d_cu = dalloc(cuh.size() * 4);
        st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
        std::vector<int32_t> out[2][3];
        for (int f = 1; f <= 2; ++f) {
            void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
            void* d_ns = dalloc((size_t)M * 4);
            void* d_su = dalloc((size_t)M * 4);
            std::vector<RadTensor> ts = {
                T(d_sc, RAD_F32, { M, nb }), T(d_bt, RAD_I32, { S, nb }), T(d_nc, RAD_I32, { S }),
                T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { S + 1 }),
                T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) };
            std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                         PI("ratio", ratio), PI("seqs", S) };
            RadArgs a;
            std::memset(&a, 0, sizeof(a));
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps.data(); a.n_p = (int)ps.size();
            pl.pin_qsel(f);
            const int rc = r->launch(&a, nullptr);
            pl.pin_qsel(0);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                ok(fmt("qsa_select wide form, %s: form %d is served", kDist[dist], f).c_str(),
                   false, fmt("rc=%d", rc));
                return;
            }
            out[f - 1][0].resize((size_t)(M * (topk + 1)));
            out[f - 1][1].resize((size_t)M);
            out[f - 1][2].resize((size_t)M);
            st_memcpy(out[f - 1][0].data(), d_se, out[f - 1][0].size() * 4, hipMemcpyDeviceToHost);
            st_memcpy(out[f - 1][1].data(), d_ns, (size_t)M * 4, hipMemcpyDeviceToHost);
            st_memcpy(out[f - 1][2].data(), d_su, (size_t)M * 4, hipMemcpyDeviceToHost);
        }
        int64_t full = 0, diff = 0;
        for (int32_t v : out[0][1]) full += v >= (int32_t)topk;
        for (int k = 0; k < 3; ++k)
            for (size_t i = 0; i < out[0][k].size(); ++i) diff += out[0][k][i] != out[1][k][i];
        ok(fmt("qsa_select wide form, %s: the table, the count and seqused are the single "
               "workgroup's, byte for byte", kDist[dist]).c_str(),
           diff == 0 && full > 0 && full < M,
           fmt("%lld entries differ; %lld of %lld queries select a full table", (long long)diff,
               (long long)full, (long long)M));
    }
}

/* qsa_select with the block list split across the device, against the same launch unsplit.
 *
 * One workgroup per query uses 6% of the device, so the launcher splits the block list into
 * chunks, takes each chunk's top-k and merges. That is exact because a chunk's local top-k holds
 * every global top-k member in it (proof above the split in r4d_qsa_select_bf16.hip), and the
 * result must be bit-identical.
 *
 * The two forms are the same launch with and without scratch: without the arena the launcher falls
 * back to the single-workgroup kernel. `nb` is over four thousand so the chunk threshold is
 * crossed, and the scores come from a small set so ties (lowest block index first, by prefix
 * count) are everywhere. */
static void qsa_select_split_one(const Plugin& pl, const char* what, int64_t S, int64_t per) {
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = S * per, nb = 20001, topk = 512, ratio = 4, pages = 8192;
    std::vector<float>   sc((size_t)(M * nb));
    /* One table per sequence with different live counts: the chunk pass reads both per sequence
     * while its grid is indexed by query. `per` is a sequence's query rows (`1 + n_spec` at
     * decode). */
    std::vector<int32_t> bt((size_t)(S * nb)), nc((size_t)S), pos((size_t)M),
                         cuh((size_t)(S + 1));
    const int64_t n = 19000;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t ns = n - s * 1300;             /* live blocks; one page in 64 is dead */
        nc[(size_t)s] = (int32_t)ns;
        for (int64_t b = 0; b < nb; ++b)
            bt[(size_t)(s * nb + b)] =
                b >= ns ? -1 : ((b % 64) == 37 ? -1 : (int32_t)((b * 4099 + 11 + s * 71) % pages));
        cuh[(size_t)s] = (int32_t)(s * per);
        for (int64_t j = 0; j < per; ++j)
            pos[(size_t)(s * per + j)] = (int32_t)(ns * ratio + j - 2);
    }
    cuh[(size_t)S] = (int32_t)M;
    /* Only 96 distinct scores over 19000 blocks, so the tie rule decides the answer. */
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b)
            sc[(size_t)(m * nb + b)] = (float)(((b * 31 + m * 17) % 96)) * 0.125f;

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    if (!d_sc || !d_bt || !d_nc || !d_po || !d_cu || !d_se || !d_ns || !d_su) {
        ok(what, false, "alloc failed"); return;
    }
    st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }), T(d_bt, RAD_I32, { S, nb }), T(d_nc, RAD_I32, { S }),
        T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { S + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", S) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t need = r->scratch ? r->scratch(&a) : 0;
    void* d_sk = need > 0 ? dalloc((size_t)need) : nullptr;
    if (need > 0 && !d_sk) { ok(what, false, "scratch alloc failed"); return; }

    /* Three runs: split, split again (determinism), and unsplit. */
    std::vector<int32_t> se[3], ns[3], su[3];
    for (int pass = 0; pass < 3; ++pass) {
        a.scratch = pass == 2 ? nullptr : d_sk;
        a.scratch_bytes = pass == 2 ? 0 : need;
        st_memset(d_se, 0xAA, (size_t)(M * (topk + 1)) * 4);
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
        se[pass].assign((size_t)(M * (topk + 1)), 0);
        ns[pass].assign((size_t)M, 0);
        su[pass].assign((size_t)M, 0);
        st_memcpy(se[pass].data(), d_se, se[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(ns[pass].data(), d_ns, ns[pass].size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(su[pass].data(), d_su, su[pass].size() * 4, hipMemcpyDeviceToHost);
    }

    /* The runs must take different forms: without the scratch hook all would be the fallback. */
    const bool split_ran = need > 0;
    const bool same = se[0] == se[2] && ns[0] == ns[2] && su[0] == su[2];
    const bool det  = se[0] == se[1] && ns[0] == ns[1] && su[0] == su[1];
    /* A set comparison beside the byte one: a wrong set is a wrong selection, while a right set in
     * another order changes only the attention's summation order. */
    bool set_ok = true;
    long slots = 0;
    long long first = -1;
    for (int64_t m = 0; m < M; ++m) {
        std::vector<int32_t> A(se[0].begin() + (size_t)(m * (topk + 1)),
                               se[0].begin() + (size_t)((m + 1) * (topk + 1)));
        std::vector<int32_t> B(se[2].begin() + (size_t)(m * (topk + 1)),
                               se[2].begin() + (size_t)((m + 1) * (topk + 1)));
        for (size_t i = 0; i < A.size(); ++i)
            if (A[i] != B[i]) { ++slots; if (first < 0) first = (long long)(m * (topk + 1) + i); }
        std::sort(A.begin(), A.end());
        std::sort(B.begin(), B.end());
        if (A != B) set_ok = false;
    }
    bool nsel_ok = ns[0] == ns[2] && su[0] == su[2];
    const bool pass = split_ran && same && det;
    std::printf("        %lld seq x %lld rows, %lld blocks, scratch %lld B; split vs unsplit %s "
                "(set %s, nsel/seqused %s, %ld slot(s) differ, first at %lld), "
                "two split runs identical %s\n",
                (long long)S, (long long)per, (long long)nb, (long long)need,
                same ? "ok" : "DIFFERS",
                set_ok ? "ok" : "DIFFERS", nsel_ok ? "ok" : "DIFFERS", slots, first,
                det ? "ok" : "NO");
    ok(what, pass, std::string("split_ran=") + (split_ran ? "1" : "0") +
                   " same=" + (same ? "1" : "0") + " det=" + (det ? "1" : "0"));
}

/* qsa_select over more than topk blocks with topk or fewer live, in every form, against libref's
 * rule byte for byte: the table (sentinel-filled before each launch, so a slot the kernel leaves
 * alone shows), the count and seqused.
 *
 * With topk or fewer live (dead pages, -inf scores), every live block is taken in block order and
 * the count says how many. The radix's first digit is the only one that sees every live key; a
 * later digit run on too few would narrow into bin 0, where a score of exactly zero lives, and
 * tie on it. Each row below is one way in:
 *   0  few live pages, continuous scores           1  few live, most of them scoring zero or less
 *   2  live pages, most scores -inf                3  exactly topk live, a third tied at zero
 *   4  all live keys in the first chunk, more than topk, ties at the threshold (the split merge
 *      sees exactly topk candidates and must still put the ties last)
 *   5  nothing live                                6  at most topk blocks (control)
 *   7  everything live (control)
 * -0.0 scores are planted in rows 1 and 4: libref ranks them as zero. */
static void qsa_select_few_live_case(const Plugin& pl) {
    const char* what = "qsa_select with topk or fewer live blocks, every form vs libref's rule";
    const RadKernelInfo* r = find(pl, "qsa_select_i32");
    if (!r || !pl.pin_qsel) { ok(what, false, "row or hook absent"); return; }

    const int64_t S = 8, M = S, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
    const int64_t nrow[S] = { 8000, 8000, 8000, 8000, 2000, 8000, 300, 8000 };
    std::vector<float>   sc((size_t)(M * nb), -INFINITY);
    std::vector<int32_t> bt((size_t)(S * nb), -1), nc((size_t)S), pos((size_t)M),
                         cuh((size_t)(S + 1));
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int64_t s = 0; s < S; ++s) {
        const int64_t n = nrow[s];
        nc[(size_t)s] = (int32_t)n;
        cuh[(size_t)s] = (int32_t)s;
        pos[(size_t)s] = (int32_t)(n * ratio + (s % 3));        /* mid-block: a partial page */
        for (int64_t b = 0; b < n; ++b) {
            float*   v  = &sc[(size_t)(s * nb + b)];
            int32_t* pg = &bt[(size_t)(s * nb + b)];
            const uint64_t u = rnd();
            const float cont = (float)(u % 1000003) / 1000.0f + 0.001f;
            *pg = (int32_t)(u % pages);
            switch (s) {
                case 0: if (b % 80 != 3) *pg = -1; *v = cont; break;             /* 100 live */
                /* 400 live, three in four scoring zero or less: more zeros than the slots the
                 * live keys leave, so a radix that kept narrowing past the first digit would
                 * find a "threshold" among them. */
                case 1: if (b % 20 != 7) *pg = -1;
                        *v = (b / 20) % 4 == 0 ? cont
                           : (b / 20) % 3 == 0 ? -2.5f : (b / 20) % 3 == 1 ? -0.0f : 0.0f;
                        break;
                case 2: *v = (b % 40 == 1) ? cont : -INFINITY; break;            /* 200 live */
                case 3: if (b % 15 != 2 || b >= 15 * 512) *pg = -1;              /* 512 live */
                        *v = (b % 45 == 2) ? 0.0f : cont; break;
                case 4: *v = (b % 7 == 0) ? cont : (b % 2 ? 0.0f : -0.0f); break;
                case 5: *pg = -1; *v = cont; break;
                case 6: if (b % 9 == 4) *pg = -1; *v = (float)(u % 13) * 0.5f; break;
                default: *v = cont; break;
            }
        }
        bt[(size_t)(s * nb + n)] = (s % 2 == 0) ? (int32_t)(rnd() % pages) : -1;  /* the tail */
    }
    cuh[(size_t)S] = (int32_t)M;

    /* libref's rule (ref_qsa_select): every live block when topk or fewer are live, else the
     * blocks above the topk-th largest value in block order, then its ties in block order. */
    std::vector<int32_t> want((size_t)(M * (topk + 1)), -1), want_n((size_t)M), want_su((size_t)M);
    for (int64_t m = 0; m < M; ++m) {
        const int64_t n = nrow[m];
        std::vector<int64_t> live;
        std::vector<float> val((size_t)n, 0.0f);
        for (int64_t b = 0; b < n; ++b) {
            const float v = sc[(size_t)(m * nb + b)];
            if (bt[(size_t)(m * nb + b)] < 0 || !(v > -1e30f)) continue;
            val[(size_t)b] = v > 0.0f ? v : 0.0f;
            live.push_back(b);
        }
        std::vector<int64_t> take;
        if ((int64_t)live.size() <= topk) {
            take = live;
        } else {
            std::vector<float> ord;
            for (int64_t b : live) ord.push_back(val[(size_t)b]);
            std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(),
                             [](float l, float r) { return l > r; });
            const float thr = ord[(size_t)(topk - 1)];
            for (int64_t b : live) if (val[(size_t)b] > thr) take.push_back(b);
            for (int64_t b : live)
                if ((int64_t)take.size() < topk && val[(size_t)b] == thr) take.push_back(b);
        }
        int32_t* row = &want[(size_t)(m * (topk + 1))];
        int64_t out = 0;
        for (int64_t b : take) row[out++] = bt[(size_t)(m * nb + b)];
        const int32_t part = (n < nb && (pos[(size_t)m] + 1) % ratio != 0)
                           ? bt[(size_t)(m * nb + n)] : -1;
        if (part >= 0) row[out++] = part;
        want_n[(size_t)m] = (int32_t)out;
        want_su[(size_t)m] = out == 0 ? 0
                           : (int32_t)((out - 1) * ratio + pos[(size_t)m] % ratio + 1);
    }

    void* d_sc = dalloc(sc.size() * 4);
    void* d_bt = dalloc(bt.size() * 4);
    void* d_nc = dalloc(nc.size() * 4);
    void* d_po = dalloc(pos.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_se = dalloc((size_t)(M * (topk + 1)) * 4);
    void* d_ns = dalloc((size_t)M * 4);
    void* d_su = dalloc((size_t)M * 4);
    if (!d_sc || !d_bt || !d_nc || !d_po || !d_cu || !d_se || !d_ns || !d_su) {
        ok(what, false, "alloc failed"); return;
    }
    st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nc.data(), nc.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_po, pos.data(), pos.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_sc, RAD_F32, { M, nb }), T(d_bt, RAD_I32, { S, nb }), T(d_nc, RAD_I32, { S }),
        T(d_po, RAD_I32, { M }),     T(d_cu, RAD_I32, { S + 1 }),
        T(d_se, RAD_I32, { M, topk + 1 }), T(d_ns, RAD_I32, { M }), T(d_su, RAD_I32, { M }) };
    std::vector<RadParam> ps = { PI("M", M), PI("blocks", nb), PI("topk", topk),
                                 PI("ratio", ratio), PI("seqs", S) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int64_t need = r->scratch ? r->scratch(&a) : 0;
    void* d_sk = need > 0 ? dalloc((size_t)need) : nullptr;
    if (need <= 0 || !d_sk) { ok(what, false, "the split path's scratch is unavailable"); return; }

    static const char* const kForm[] = { "single workgroup", "wide", "split" };
    for (int f = 0; f < 3; ++f) {
        a.scratch       = f == 2 ? d_sk : nullptr;
        a.scratch_bytes = f == 2 ? need : 0;
        st_memset(d_se, 0x55, (size_t)(M * (topk + 1)) * 4);
        st_memset(d_ns, 0x55, (size_t)M * 4);
        st_memset(d_su, 0x55, (size_t)M * 4);
        pl.pin_qsel(f == 2 ? 0 : f + 1);
        const int rc = r->launch(&a, nullptr);
        pl.pin_qsel(0);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            ok(fmt("%s: the %s form is served", what, kForm[f]).c_str(), false, fmt("rc=%d", rc));
            continue;
        }
        std::vector<int32_t> se((size_t)(M * (topk + 1))), ns((size_t)M), su((size_t)M);
        st_memcpy(se.data(), d_se, se.size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(ns.data(), d_ns, ns.size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(su.data(), d_su, su.size() * 4, hipMemcpyDeviceToHost);
        std::string bad;
        for (int64_t m = 0; m < M; ++m) {
            int64_t diff = 0, first = -1;
            for (int64_t i = 0; i <= topk; ++i)
                if (se[(size_t)(m * (topk + 1) + i)] != want[(size_t)(m * (topk + 1) + i)]) {
                    ++diff;
                    if (first < 0) first = i;
                }
            if (diff || ns[(size_t)m] != want_n[(size_t)m] || su[(size_t)m] != want_su[(size_t)m])
                bad += fmt(" row %lld: nsel %d (want %d), %lld slot(s) differ from %lld;",
                           (long long)m, ns[(size_t)m], want_n[(size_t)m], (long long)diff,
                           (long long)first);
        }
        ok(fmt("%s: the %s form", what, kForm[f]).c_str(), bad.empty(),
           bad.empty() ? std::string("8 rows exact") : bad);
    }
}

/* qsa_score, against an f64 evaluation of the same formula.
 *
 * Two controls:
 *
 *   sum-then-relu instead of relu-then-sum: both rank blocks plausibly, but it is a different
 *   model (rel_l2 of order one).
 *
 *   a padded entry scoring zero instead of -inf: a real score can be zero, so padding would
 *   outrank blocks every head rejected. The padding lanes are checked directly, since an l2 over
 *   a mostly live row would hide them.
 */
static void qsa_score_case(const Plugin& pl) {
    const char* what = "qsa_score vs an f64 evaluation of the same formula";
    const RadKernelInfo* r = find(pl, "qsa_score_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 5, n = 128, heads = 4, nb = 37, pages = 64;

    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    std::vector<int32_t>  bth((size_t)(M * nb));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 29 + 7) % 83) - 41) / 47.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 41 + 13) % 71) - 35) / 39.0f);
    /* A scattered table with padding at the end of every row, as a sequence shorter than the
     * declared width produces. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t live = nb - 3 - m * 4;
        for (int64_t b = 0; b < nb; ++b)
            bth[(size_t)(m * nb + b)] = b < live ? (int32_t)((b * 13 + m * 5) % pages) : -1;
    }

    std::vector<double> want((size_t)(M * nb), 0.0), sumrelu((size_t)(M * nb), 0.0);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t b = 0; b < nb; ++b) {
            const int32_t p = bth[(size_t)(m * nb + b)];
            if (p < 0) { want[(size_t)(m * nb + b)] = -1e30; sumrelu[(size_t)(m * nb + b)] = -1e30;
                         continue; }
            double s = 0.0, raw = 0.0;
            for (int64_t h = 0; h < heads; ++h) {
                double d = 0.0;
                for (int64_t j = 0; j < n; ++j)
                    d += (double)h_bf16(qh[(size_t)((m * heads + h) * n + j)]) *
                         (double)h_bf16(bkh[(size_t)(p * n + j)]);
                s += d > 0.0 ? d : 0.0;     /* relu per (block, head), then the head sum */
                raw += d;                   /* the control: sum first, clamp after */
            }
            const double inv = 1.0 / std::sqrt((double)n);
            want[(size_t)(m * nb + b)] = s * inv;
            sumrelu[(size_t)(m * nb + b)] = (raw > 0.0 ? raw : 0.0) * inv;
        }

    /* One query per sequence (decode shape, `seq(m) == m`), as the oracle assumes. Many queries
     * per history are covered by qsa_score_rows_case and qsa_select_prefill_case. */
    std::vector<int32_t> cuh((size_t)(M + 1)), nch((size_t)M, (int32_t)nb);
    for (int64_t i = 0; i <= M; ++i) cuh[(size_t)i] = (int32_t)i;

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_nc = dalloc(nch.size() * 4);
    void* d_o  = dalloc((size_t)(M * nb) * 4);
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), nch.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_BF16, { M, heads * n }),
        T(d_bk, RAD_BF16, { pages, n }),
        T(d_bt, RAD_I32,  { M, nb }),
        T(d_cu, RAD_I32,  { M + 1 }),
        T(d_nc, RAD_I32,  { M }),
        T(d_o,  RAD_F32,  { M, nb }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb), PI("seqs", M),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<float> got((size_t)(M * nb));
    st_memcpy(got.data(), d_o, got.size() * 4, hipMemcpyDeviceToHost);

    /* The l2 covers live entries only; the padding (-inf) is checked as a predicate. */
    auto rel = [&](const std::vector<double>& ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] <= -1e29) continue;
            const double d = (double)got[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        return den > 0.0 ? std::sqrt(num / den) : -1.0;
    };
    long pad = 0, pad_ninf = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (want[i] <= -1e29) { ++pad; if (got[i] < -1e30f) ++pad_ninf; }

    const double r_ok = rel(want), r_ctl = rel(sumrelu);
    const bool pass = r_ok >= 0.0 && r_ok < 1e-5 && r_ctl > 100.0 * r_ok && pad > 0 &&
                      pad_ninf == pad;
    std::printf("        rel_l2 vs f64: %.3e; sum-then-relu %.3e; %ld padded entries, %ld at "
                "-inf\n", r_ok, r_ctl, pad, pad_ninf);
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_ctl) +
                   ", padding " + std::to_string(pad_ninf) + "/" + std::to_string(pad));
}

/* qsa_score over sequences of many rows, against the same f64 formula.
 *
 * In a mixed step a 16-row query tile can hold the end of one sequence and the start of the next.
 * Rows {7, 20, 5}: the second sequence starts inside the first tile and ends inside the second,
 * and each sequence has its own table row, padding and live count. The control scores every row
 * against sequence 0's table, as a tile that ignored the boundary would. */
static void qsa_score_rows_case(const Plugin& pl) {
    const char* what = "qsa_score, three sequences of 7, 20 and 5 rows, vs f64";
    const RadKernelInfo* r = find(pl, "qsa_score_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t S = 3, n = 128, heads = 4, nb = 45, pages = 80;
    const int64_t rows[S] = { 7, 20, 5 };
    std::vector<int32_t> cuh((size_t)(S + 1), 0), nch((size_t)S);
    for (int64_t i = 0; i < S; ++i) cuh[(size_t)(i + 1)] = cuh[(size_t)i] + (int32_t)rows[i];
    const int64_t M = cuh[(size_t)S];

    std::vector<uint16_t> qh((size_t)(M * heads * n)), bkh((size_t)(pages * n));
    for (int64_t i = 0; i < M * heads * n; ++i)
        qh[(size_t)i] = h_to_bf16((float)(((i * 31 + 5) % 89) - 44) / 43.0f);
    for (int64_t i = 0; i < pages * n; ++i)
        bkh[(size_t)i] = h_to_bf16((float)(((i * 43 + 11) % 73) - 36) / 37.0f);
    std::vector<int32_t> bth((size_t)(S * nb));
    for (int64_t sq = 0; sq < S; ++sq) {
        const int64_t live = nb - 2 - sq * 9;
        nch[(size_t)sq] = (int32_t)(live - 1);           /* the table has one entry past nc */
        for (int64_t b = 0; b < nb; ++b)
            bth[(size_t)(sq * nb + b)] =
                b < live && (b % 11) != 4 ? (int32_t)((b * 17 + sq * 7) % pages) : -1;
    }

    auto score = [&](int64_t m, int64_t sq, int64_t b) {
        const int32_t p = bth[(size_t)(sq * nb + b)];
        if (p < 0) return -1e30;
        double s = 0.0;
        for (int64_t h = 0; h < heads; ++h) {
            double d = 0.0;
            for (int64_t j = 0; j < n; ++j)
                d += (double)h_bf16(qh[(size_t)((m * heads + h) * n + j)]) *
                     (double)h_bf16(bkh[(size_t)(p * n + j)]);
            s += d > 0.0 ? d : 0.0;
        }
        return s / std::sqrt((double)n);
    };

    void* d_q  = dalloc(qh.size() * 2);
    void* d_bk = dalloc(bkh.size() * 2);
    void* d_bt = dalloc(bth.size() * 4);
    void* d_cu = dalloc(cuh.size() * 4);
    void* d_nc = dalloc(nch.size() * 4);
    void* d_o  = dalloc((size_t)(M * nb) * 4);
    st_memcpy(d_q,  qh.data(),  qh.size() * 2,  hipMemcpyHostToDevice);
    st_memcpy(d_bk, bkh.data(), bkh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_bt, bth.data(), bth.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_cu, cuh.data(), cuh.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_nc, nch.data(), nch.size() * 4, hipMemcpyHostToDevice);
    /* A sentinel in every entry catches a store past a sequence's live count. */
    std::vector<float> sentinel((size_t)(M * nb), 12345.0f);
    st_memcpy(d_o, sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_BF16, { M, heads * n }), T(d_bk, RAD_BF16, { pages, n }),
        T(d_bt, RAD_I32,  { S, nb }),        T(d_cu, RAD_I32,  { S + 1 }),
        T(d_nc, RAD_I32,  { S }),            T(d_o,  RAD_F32,  { M, nb }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("heads", heads), PI("blocks", nb),
                                 PI("seqs", S), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
    std::vector<float> got((size_t)(M * nb));
    st_memcpy(got.data(), d_o, got.size() * 4, hipMemcpyDeviceToHost);

    double num = 0.0, den = 0.0, cnum = 0.0;
    long pad = 0, pad_ninf = 0, past = 0, past_kept = 0;
    for (int64_t sq = 0; sq < S; ++sq)
        for (int64_t m = cuh[(size_t)sq]; m < cuh[(size_t)(sq + 1)]; ++m)
            for (int64_t b = 0; b < nb; ++b) {
                const float g = got[(size_t)(m * nb + b)];
                if (b >= nch[(size_t)sq]) { ++past; past_kept += g == 12345.0f; continue; }
                const double w = score(m, sq, b);
                if (w <= -1e29) { ++pad; pad_ninf += g < -1e30f; continue; }
                const double c = score(m, 0, b);
                num += ((double)g - w) * ((double)g - w);
                den += w * w;
                if (c > -1e29) cnum += (c - w) * (c - w);
            }
    const double r_ok = den > 0.0 ? std::sqrt(num / den) : -1.0;
    const double r_ctl = den > 0.0 ? std::sqrt(cnum / den) : -1.0;
    const bool pass = r_ok >= 0.0 && r_ok < 1e-5 && r_ctl > 100.0 * r_ok && pad > 0 &&
                      pad_ninf == pad && past > 0 && past_kept == past;
    std::printf("        rel_l2 vs f64: %.3e; every row on sequence 0's table %.3e; %ld padded "
                "entries, %ld at -inf; %ld past the live count, %ld untouched\n",
                r_ok, r_ctl, pad, pad_ninf, past, past_kept);
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_ctl));
}

/* qsa_block_key, against an f64 evaluation of the same formula.
 *
 * The reference pools in f32 and rounds to bf16 before the norm; carrying f32 through gives a
 * different number. Both oracles are built and their distance reported, and the kernel is checked
 * against the rounding one.
 *
 * The negative control is the position: `pos0` off by the ratio (the block's last token instead of
 * its first) rotates every key by one block's angle, invisible in magnitude but caught by
 * rel_l2. */
static void qsa_block_key_case(const Plugin& pl) {
    const char* what = "qsa_block_key vs an f64 evaluation of the same formula";
    const RadKernelInfo* r = find(pl, "qsa_block_key_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    const int64_t M = 17, n = 128, ratio = 4, rot = 64, pos0 = 36;
    const double eps = 1e-6, theta = 10000000.0, rscale = 1.0, wadd = 1.0;

    std::vector<uint16_t> kh((size_t)(M * ratio * n));
    std::vector<float>    wh((size_t)n);
    for (int64_t i = 0; i < M * ratio; ++i)
        for (int64_t d = 0; d < n; ++d) {
            float v = (float)(((i * 37 + d * 11) % 97) - 48) / 53.0f;
            if (d % 31 == 0) v *= 5.0f;         /* outlier channels, as in a real key */
            kh[(size_t)(i * n + d)] = h_to_bf16(v);
        }
    for (int64_t d = 0; d < n; ++d) wh[(size_t)d] = 0.1f + (float)((d * 7) % 13) / 40.0f;

    /* `round_mean` false is the order the reference does not take. */
    auto oracle = [&](bool round_mean, int64_t p0, std::vector<double>* out) {
        out->assign((size_t)(M * n), 0.0);
        std::vector<double> v((size_t)n);
        for (int64_t b = 0; b < M; ++b) {
            for (int64_t d = 0; d < n; ++d) {
                float acc = 0.0f;
                for (int64_t j = 0; j < ratio; ++j)
                    acc += h_bf16(kh[(size_t)((b * ratio + j) * n + d)]);
                acc /= (float)ratio;
                v[(size_t)d] = round_mean ? (double)h_bf16(h_to_bf16(acc)) : (double)acc;
            }
            double ss = 0.0;
            for (int64_t d = 0; d < n; ++d) ss += v[(size_t)d] * v[(size_t)d];
            const double inv = 1.0 / std::sqrt(ss / (double)n + eps);
            std::vector<double> nz((size_t)n);
            for (int64_t d = 0; d < n; ++d) nz[(size_t)d] = v[(size_t)d] * inv * (wadd + wh[(size_t)d]);
            const double pp = (double)(p0 + b * ratio) / rscale;
            for (int64_t d = 0; d < n; ++d) {
                double res = nz[(size_t)d];
                if (d < rot) {
                    const int64_t half = rot / 2;
                    const bool first = d < half;
                    const int64_t pt = first ? d + half : d - half;
                    const int64_t ci = first ? d : d - half;
                    const double iv = 1.0 / std::pow(theta, (double)(2 * ci) / (double)rot);
                    const double c = std::cos(pp * iv), s = std::sin(pp * iv);
                    res = first ? (nz[(size_t)d] * c - nz[(size_t)pt] * s)
                                : (nz[(size_t)pt] * s + nz[(size_t)d] * c);
                }
                (*out)[(size_t)(b * n + d)] = res;
            }
        }
    };

    std::vector<double> want, unrounded, shifted;
    oracle(true,  pos0, &want);
    oracle(false, pos0, &unrounded);
    oracle(true,  pos0 + ratio, &shifted);

    void* d_k = dalloc(kh.size() * 2);
    void* d_w = dalloc(wh.size() * 4);
    void* d_o = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_k, kh.data(), kh.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice);

    /* `page` absent: an entry with a null pointer, not a shorter list (operands are positional).
     * The scattered form is the pass below. */
    std::vector<RadTensor> ts = {
        T(d_k, RAD_BF16, { M * ratio, n }),
        T(d_w, RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        T(nullptr, RAD_I32, { M }),
        T(d_o, RAD_BF16, { M, n }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", n), PI("ratio", ratio), PI("rotary_dim", rot), PI("pos0", pos0),
        PF("eps", eps), PF("theta", theta), PF("scale", rscale), PF("wadd", wadd),
        PS("dtype", "bf16"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    const int rc = r->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }

    std::vector<uint16_t> got((size_t)(M * n));
    st_memcpy(got.data(), d_o, got.size() * 2, hipMemcpyDeviceToHost);

    /* The scattered form the engine uses: a page list sends block b to page `pg[b]` of a store as
     * tall as the pool, and -1 means no page (nothing written). Same oracle; under test are the
     * destination and that the block count comes from the page list, not from the taller `out`. */
    const int64_t store = M * 3 + 11;
    std::vector<int32_t> pg((size_t)M);
    for (int64_t b = 0; b < M; ++b)
        pg[(size_t)b] = (b == M / 2) ? -1 : (int32_t)((b * 7 + 3) % store);
    void* d_pg = dalloc(pg.size() * 4);
    void* d_o2 = dalloc((size_t)(store * n) * 2);
    st_memcpy(d_pg, pg.data(), pg.size() * 4, hipMemcpyHostToDevice);
    std::vector<uint16_t> fill((size_t)(store * n), h_to_bf16(-7.0f));
    st_memcpy(d_o2, fill.data(), fill.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts2 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(d_pg, RAD_I32,  { M }),
        T(nullptr, RAD_I32, { M }),
        T(d_o2, RAD_BF16, { store, n }),
    };
    RadArgs a2 = a;
    a2.t = ts2.data(); a2.n_t = (int)ts2.size();
    bool scatter_ok = r->launch(&a2, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got2((size_t)(store * n));
    st_memcpy(got2.data(), d_o2, got2.size() * 2, hipMemcpyDeviceToHost);
    long untouched = 0;
    for (int64_t b = 0; b < M && scatter_ok; ++b) {
        if (pg[(size_t)b] < 0) continue;
        for (int64_t d = 0; d < n; ++d)
            if (got2[(size_t)(pg[(size_t)b] * n + d)] != got[(size_t)(b * n + d)])
                scatter_ok = false;
    }
    for (int64_t p = 0; p < store; ++p) {
        bool named = false;
        for (int64_t b = 0; b < M; ++b) if (pg[(size_t)b] == (int32_t)p) named = true;
        if (!named && got2[(size_t)(p * n)] == h_to_bf16(-7.0f)) ++untouched;
    }
    /* Every unnamed page must still hold the fill, including where the -1 block would land if the
     * guard were missing. */
    const bool scatter_clean = scatter_ok && untouched == store - (M - 1);

    /* The work-list form: `pos` given per block in shuffled order, as at decode, where the blocks
     * completing in one launch belong to different sequences. The positions are permuted and each
     * block is checked against the oracle at its permuted position, catching keys read at the wrong
     * position. */
    std::vector<int32_t> bpos((size_t)M);
    std::vector<int64_t> perm((size_t)M);
    for (int64_t b = 0; b < M; ++b) perm[(size_t)b] = (b * 5 + 2) % M;
    for (int64_t b = 0; b < M; ++b)
        bpos[(size_t)b] = (int32_t)(pos0 + perm[(size_t)b] * ratio);
    void* d_bp = dalloc(bpos.size() * 4);
    void* d_o3 = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_bp, bpos.data(), bpos.size() * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts3 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        T(d_bp, RAD_I32,  { M }),
        T(d_o3, RAD_BF16, { M, n }),
    };
    RadArgs a3 = a;
    a3.t = ts3.data(); a3.n_t = (int)ts3.size();
    bool list_ok = r->launch(&a3, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got3((size_t)(M * n));
    st_memcpy(got3.data(), d_o3, got3.size() * 2, hipMemcpyDeviceToHost);
    {
        /* Block b reads rows [b*ratio, (b+1)*ratio) as before and only the angle moves, so the
         * oracle is built at the permuted position for the same keys. */
        std::vector<double> wantp;
        oracle(true, pos0, &wantp);          /* shape only; recomputed per block below */
        for (int64_t b = 0; b < M && list_ok; ++b) {
            std::vector<double> one;
            oracle(true, pos0 + (perm[(size_t)b] - b) * ratio, &one);
            for (int64_t d = 0; d < n; ++d) {
                const double e = (double)h_bf16(got3[(size_t)(b * n + d)]) -
                                 one[(size_t)(b * n + d)];
                if (std::fabs(e) > 1e-2 * (std::fabs(one[(size_t)(b * n + d)]) + 1e-3))
                    list_ok = false;
            }
        }
    }

    /* Under M-RoPE: `mode` imrope with Qwen3.8's sections, each block's three components in an
     * [M, 3] view of a buffer with row pitch 4 (a dense reading picks the wrong block). Frequency i
     * reads H when i % 3 == 1 and i < 33, W when i % 3 == 2 and i < 30, T otherwise; the components
     * all differ, so a wrong component shows. */
    const int64_t pitch = 4;
    std::vector<int32_t> b4((size_t)(M * pitch), -1);
    for (int64_t b = 0; b < M; ++b) {
        const int32_t t = (int32_t)(pos0 + b * ratio);
        b4[(size_t)(b * pitch + 0)] = t;
        b4[(size_t)(b * pitch + 1)] = t + 5 + (int32_t)(b % 3);
        b4[(size_t)(b * pitch + 2)] = t + 13 + (int32_t)(b % 5);
    }
    void* d_b4 = dalloc(b4.size() * 4);
    void* d_o4 = dalloc((size_t)(M * n) * 2);
    st_memcpy(d_b4, b4.data(), b4.size() * 4, hipMemcpyHostToDevice);
    RadTensor b4v = T(d_b4, RAD_I32, { M, 3 });
    b4v.stride[0] = pitch;
    std::vector<RadTensor> ts4 = {
        T(d_k,  RAD_BF16, { M * ratio, n }),
        T(d_w,  RAD_F32,  { n }),
        T(nullptr, RAD_I32, { M }),
        b4v,
        T(d_o4, RAD_BF16, { M, n }),
    };
    std::vector<RadParam> ps4 = ps;
    ps4.push_back(PS("mode", "imrope"));
    ps4.push_back(PS("sections", "11 11 10"));
    RadArgs a4 = a;
    a4.t = ts4.data(); a4.n_t = (int)ts4.size();
    a4.p = ps4.data(); a4.n_p = (int)ps4.size();
    bool mrope_ok = r->launch(&a4, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got4((size_t)(M * n));
    st_memcpy(got4.data(), d_o4, got4.size() * 2, hipMemcpyDeviceToHost);
    double mr_num = 0.0, mr_den = 0.0, mr_t = 0.0;
    for (int64_t b = 0; b < M && mrope_ok; ++b) {
        /* The one-component oracle at each component's position; per frequency, the interleave
         * picks one. */
        std::vector<double> oc[3];
        for (int c = 0; c < 3; ++c) {
            std::vector<double> full;
            oracle(true, pos0 + (b4[(size_t)(b * pitch + c)] - (pos0 + b * ratio)), &full);
            oc[c].assign(full.begin() + b * n, full.begin() + (b + 1) * n);
        }
        for (int64_t d = 0; d < n; ++d) {
            const int64_t half = rot / 2;
            const int64_t ci = d < rot ? (d < half ? d : d - half) : 0;
            const int comp = d >= rot ? 0 : (ci % 3 == 1 && ci < 33) ? 1 : (ci % 3 == 2 && ci < 30) ? 2 : 0;
            const double e = oc[comp][(size_t)d], g2 = (double)h_bf16(got4[(size_t)(b * n + d)]);
            mr_num += (g2 - e) * (g2 - e);
            mr_den += e * e;
            mr_t += (g2 - oc[0][(size_t)d]) * (g2 - oc[0][(size_t)d]);
        }
    }
    const double r_mr = mr_den > 0.0 ? std::sqrt(mr_num / mr_den) : -1.0;
    const double r_mr_t = mr_den > 0.0 ? std::sqrt(mr_t / mr_den) : -1.0;
    /* The same buffer's first column, as a text-only step passes it: must equal the tiled answer
     * bit for bit, since column 0 holds the tiling. */
    RadTensor b1v = T(d_b4, RAD_I32, { M, 1 });
    b1v.stride[0] = pitch;
    ts4[3] = b1v;
    bool col_same = r->launch(&a4, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    st_memcpy(got4.data(), d_o4, got4.size() * 2, hipMemcpyDeviceToHost);
    col_same = col_same && got4 == got;
    mrope_ok = mrope_ok && r_mr >= 0.0 && r_mr < 0.01 && r_mr_t > 5.0 * r_mr && col_same;
    std::printf("        M-RoPE imrope: rel_l2 %.3e (all-T control %.3e); one column = tiled %s\n",
                r_mr, r_mr_t, col_same ? "yes" : "NO");

    auto rel = [&](const std::vector<double>& ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = (double)h_bf16(got[i]) - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        return den > 0.0 ? std::sqrt(num / den) : -1.0;
    };
    double sep = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = want[i] - unrounded[i];
        sep += d * d; den += want[i] * want[i];
    }
    sep = den > 0.0 ? std::sqrt(sep / den) : -1.0;

    const double r_ok = rel(want), r_shift = rel(shifted), r_unr = rel(unrounded);
    /* 0.01 reflects the bf16 output: a correct kernel lands within the grid spacing (2^-9) of the
     * exact answer. The position control (off by one block) must be over 20x worse. The rounding
     * control is subtle: the kernel must be closer to the oracle that rounds the pooled mean to
     * bf16 before the norm than to the one that does not; the two differ by only ~35%, the only
     * discrimination a bf16 output allows. */
    const bool pass = r_ok >= 0.0 && r_ok < 0.01 && r_shift > 20.0 * r_ok &&
                      r_unr > 0.0 && r_ok < 0.9 * r_unr && scatter_clean && list_ok && mrope_ok;
    std::printf("        rel_l2 vs f64: %.3e (unrounded-mean order %.3e, the two orders %.3e "
                "apart); pos0 off by one block %.3e; scattered form %s\n",
                r_ok, r_unr, sep, r_shift, scatter_clean ? "ok" : "DIFFERS");
    std::printf("        work-list positions %s\n", list_ok ? "ok" : "DIFFER");
    ok(what, pass, "rel " + std::to_string(r_ok) + ", control " + std::to_string(r_shift));
}

/* qsa_block_key with YaRN, against yarn_ref. Identical unit keys make the block mean exactly one,
 * and unit gain with eps 0 makes the norm one too, so what is left is the rotation of the
 * compressed key at each block's position: c - s in the first half of the rotary, s + c in the
 * second, and the unrotated tail at one. The bound is yarn_ref's for both terms plus the bf16
 * rounding of the output. */
static void qsa_block_key_yarn_case(const Plugin& pl) {
    const char* what = "qsa_block_key YaRN vs a double YaRN through 524287";
    const RadKernelInfo* r = find(pl, "qsa_block_key_bf16");
    if (!r) { ok(what, false, "row absent"); return; }
    const int64_t M = (int64_t)(sizeof kYarnPos / sizeof kYarnPos[0]), n = 128, ratio = 4, rot = 64;
    std::vector<uint16_t> keys((size_t)(M * ratio * n), h_to_bf16(1.0f));
    std::vector<float> gains((size_t)n, 1.0f);
    void* dk = dalloc(keys.size() * 2), *dw = dalloc(gains.size() * 4);
    void* dp = dalloc((size_t)M * 4), *dout = dalloc((size_t)(M * n * 2));
    st_memcpy(dk, keys.data(), keys.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(dw, gains.data(), gains.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(dp, kYarnPos, (size_t)M * 4, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = { T(dk, RAD_BF16, { M * ratio, n }), T(dw, RAD_F32, { n }),
                                  T(nullptr, RAD_I32, { M }), T(dp, RAD_I32, { M }),
                                  T(dout, RAD_BF16, { M, n }) };
    std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("ratio", ratio), PI("rotary_dim", rot),
                                 PF("eps", 0), PF("theta", 1e7), PF("scale", 1), PF("wadd", 0),
                                 PS("dtype", "bf16"), PF("yarn_factor", 2),
                                 PF("yarn_low", yarn_low(rot)), PF("yarn_high", yarn_high(rot)),
                                 PF("yarn_magnitude", kYarnMag) };
    RadArgs a{}; a.t = ts.data(); a.n_t = ts.size(); a.p = ps.data(); a.n_p = ps.size();
    bool passed = r->launch(&a, nullptr) == RAD_OK;
    st_stream_sync(nullptr);
    std::vector<uint16_t> got((size_t)(M * n));
    st_memcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost);
    double worst = 0.0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t i = 0; i < n; ++i) {
            double want = 1.0, tol = 0.0;
            if (i < rot) {
                const YarnRef y = yarn_ref(kYarnPos[m], (int)(i % (rot / 2)), (int)rot, 1e7, 2.0,
                                           yarn_low(rot), yarn_high(rot), kYarnMag);
                want = i < rot / 2 ? y.c - y.s : y.c + y.s;
                tol  = 2.0 * y.tol;
            }
            tol += std::fabs(want) * 0x1p-8 + 1e-6;
            const double have = h_bf16(got[(size_t)(m * n + i)]);
            passed = passed && std::isfinite(have);
            worst = std::max(worst, std::fabs(have - want) / tol);
        }
    char note[64];
    std::snprintf(note, sizeof note, "worst %.3g of the bf16 + f32 bound", worst);
    ok(what, passed && worst <= 1.0, note);
}

/* gram_accum, against an f64 host Gram of the same dequantised bytes.
 *
 * The accumulate is what matters: a corpus is many launches into one buffer. So `h` is seeded with
 * a non-zero pattern and the kernel launched twice; the expectation is seed + 2*want, which a
 * kernel that stores, or drops the seed, fails.
 *
 * The activation goes through the device quantiser, pinning the two ops' scale convention:
 * gram_accum reads `scale` as a dequant multiplier (the other reading is off by 448^2).
 *
 * K = 260 is not a multiple of the 64-wide output tile (four live columns in the last) or of the
 * 128-wide scale group (four in the last), but is a multiple of four (the staging dword). */
static void gram_accum_case(const Plugin& pl) {
    const char* what = "gram_accum vs an f64 host Gram of the same bytes, accumulated twice";
    const RadKernelInfo* g = find(pl, "gram_accum");
    const RadKernelInfo* q = find(pl, "quant_act_fp8");
    if (!g || !q) { ok(what, false, "row absent"); return; }

    const int64_t M = 200, K = 260, group = 128, kb = (K + group - 1) / group;

    std::vector<uint16_t> xbf((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;          /* outliers, so the scales differ by group */
            xbf[(size_t)(m * K + k)] = h_to_bf16(v);
        }

    void* d_x  = dalloc(xbf.size() * 2);
    void* d_q  = dalloc((size_t)(M * K) + 4);    /* the row is owned out to a multiple of 4 */
    void* d_qs = dalloc((size_t)(M * kb) * 4);
    st_memcpy(d_x, xbf.data(), xbf.size() * 2, hipMemcpyHostToDevice);

    std::vector<RadTensor> qt = {
        T(d_x,  RAD_BF16,   { M, K }),
        T(d_q,  RAD_F8E4M3, { M, K }),
        T(d_qs, RAD_F32,    { M, kb }),
    };
    std::vector<RadParam> qps = {
        PI("M", M), PI("n", K), PI("group", group), PS("dtype", "bf16"),
    };
    RadArgs qa;
    std::memset(&qa, 0, sizeof(qa));
    qa.t = qt.data(); qa.n_t = (int)qt.size();
    qa.p = qps.data(); qa.n_p = (int)qps.size();
    const int qrc = q->launch(&qa, nullptr);
    st_stream_sync(nullptr);
    if (qrc != RAD_OK) { ok(what, false, "quantiser rc=" + std::to_string(qrc)); return; }

    /* The oracle uses the codes read back, not the bf16 source, so only the Gram is measured. */
    std::vector<uint8_t> qc((size_t)(M * K));
    std::vector<float>   qsc((size_t)(M * kb));
    st_memcpy(qc.data(),  d_q,  qc.size(),      hipMemcpyDeviceToHost);
    st_memcpy(qsc.data(), d_qs, qsc.size() * 4, hipMemcpyDeviceToHost);
    std::vector<double> xd((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k)
            xd[(size_t)(m * K + k)] = (double)h_e4m3(qc[(size_t)(m * K + k)]) *
                                      (double)qsc[(size_t)(m * kb + k / group)];

    std::vector<double> want((size_t)(K * K), 0.0);
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = 0; k2 < K; ++k2) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m)
                acc += xd[(size_t)(m * K + k1)] * xd[(size_t)(m * K + k2)];
            want[(size_t)(k1 * K + k2)] = acc;
        }

    std::vector<float> seed((size_t)(K * K));
    for (int64_t i = 0; i < K * K; ++i) seed[(size_t)i] = (float)((i % 17) - 8) * 0.25f;
    void* d_h = dalloc(seed.size() * 4);
    st_memcpy(d_h, seed.data(), seed.size() * 4, hipMemcpyHostToDevice);

    std::vector<RadTensor> ts = {
        T(d_q,  RAD_F8E4M3, { M, K }),
        T(d_qs, RAD_F32,    { M, kb }),
        T(d_h,  RAD_F32,    { K, K }),
    };
    std::vector<RadParam> ps = {
        PI("M", M), PI("n", K), PI("group", group), PS("dtype", "fp8a8"),
    };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();
    for (int i = 0; i < 2; ++i) {
        const int rc = g->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what, false, "launch rc=" + std::to_string(rc)); return; }
    }

    std::vector<float> got((size_t)(K * K));
    st_memcpy(got.data(), d_h, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int64_t i = 0; i < K * K; ++i) {
        const double exp = (double)seed[(size_t)i] + 2.0 * want[(size_t)i];
        const double d = (double)got[(size_t)i] - exp;
        num += d * d; den += exp * exp;
        const double r = std::fabs(d) / (std::fabs(exp) + 1e-6);
        if (r > worst) worst = r;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    /* f32 accumulation over 200 rows against f64: 1e-5 leaves three orders of margin, well below
     * what the consumer's damping would notice. */
    const bool pass = rel >= 0.0 && rel < 1e-5;
    std::printf("        rel_l2 %.3e, worst entry %.3e\n", rel, worst);
    ok(what, pass, "rel_l2 " + std::to_string(rel) + ", worst " + std::to_string(worst));
}

/* gram_accum at bf16, against an f64 host Gram of the same bf16 values.
 *
 * The same seeded two-launch accumulate for the form with no scale operand, plus symmetry: GPTQ's
 * Cholesky assumes it, and each entry and its mirror are the same products summed in the same row
 * order by two blocks, so the output must be symmetric to the bit.
 *
 * Shapes: K = 260 for the masking (see the fp8 case) with M not a whole number of staged row
 * blocks, and the routed down projection's width over sorted rows. */
static void gram_accum_bf16_case(const Plugin& pl, int64_t M, int64_t K) {
    const std::string what = "gram_accum_bf16 vs an f64 host Gram (M=" + std::to_string(M) +
                             ", K=" + std::to_string(K) + "), accumulated twice";
    const RadKernelInfo* g = find(pl, "gram_accum_bf16");
    if (!g) { ok(what.c_str(), false, "row absent"); return; }

    std::vector<uint16_t> xbf((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < K; ++k) {
            float v = (float)(((m * 29 + k * 11) % 53) - 26) / 41.0f;
            if (k % 23 == 0) v *= 6.0f;
            xbf[(size_t)(m * K + k)] = h_to_bf16(v);
        }
    std::vector<double> xd((size_t)(M * K));
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = (double)h_bf16(xbf[i]);
    std::vector<double> want((size_t)(K * K), 0.0);
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = k1; k2 < K; ++k2) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m) acc += xd[(size_t)(m * K + k1)] * xd[(size_t)(m * K + k2)];
            want[(size_t)(k1 * K + k2)] = want[(size_t)(k2 * K + k1)] = acc;
        }

    void* d_x = dalloc(xbf.size() * 2);
    void* d_h = dalloc((size_t)(K * K) * 4);
    if (!d_x || !d_h) { ok(what.c_str(), false, "alloc failed"); return; }
    st_memcpy(d_x, xbf.data(), xbf.size() * 2, hipMemcpyHostToDevice);
    std::vector<RadTensor> ts = {
        T(d_x, RAD_BF16, { M, K }),
        RadTensor{},
        T(d_h, RAD_F32,  { K, K }),
    };
    std::vector<RadParam> ps = { PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16") };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps.data(); a.n_p = (int)ps.size();

    /* One launch into zeros: the symmetry check reads this. */
    int rc = g->launch(&a, nullptr);
    st_stream_sync(nullptr);
    if (rc != RAD_OK) { ok(what.c_str(), false, "launch rc=" + std::to_string(rc)); return; }
    std::vector<float> one((size_t)(K * K));
    st_memcpy(one.data(), d_h, one.size() * 4, hipMemcpyDeviceToHost);
    int64_t asym = 0;
    for (int64_t k1 = 0; k1 < K; ++k1)
        for (int64_t k2 = k1 + 1; k2 < K; ++k2)
            if (std::memcmp(&one[(size_t)(k1 * K + k2)], &one[(size_t)(k2 * K + k1)], 4)) ++asym;
    ok((what + ": symmetric to the bit").c_str(), asym == 0,
       std::to_string(asym) + " mirrored pairs differ");

    /* Seed, then two launches: expect seed + 2 * want. */
    std::vector<float> seed((size_t)(K * K));
    for (int64_t i = 0; i < K * K; ++i) seed[(size_t)i] = (float)((i % 17) - 8) * 0.25f;
    st_memcpy(d_h, seed.data(), seed.size() * 4, hipMemcpyHostToDevice);
    for (int i = 0; i < 2; ++i) {
        rc = g->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK) { ok(what.c_str(), false, "launch rc=" + std::to_string(rc)); return; }
    }
    std::vector<float> got((size_t)(K * K));
    st_memcpy(got.data(), d_h, got.size() * 4, hipMemcpyDeviceToHost);
    double num = 0.0, den = 0.0, worst = 0.0;
    for (int64_t i = 0; i < K * K; ++i) {
        const double exp = (double)seed[(size_t)i] + 2.0 * want[(size_t)i];
        const double d = (double)got[(size_t)i] - exp;
        num += d * d; den += exp * exp;
        const double r = std::fabs(d) / (std::fabs(exp) + 1e-6);
        if (r > worst) worst = r;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
    std::printf("        rel_l2 %.3e, worst entry %.3e\n", rel, worst);
    ok(what.c_str(), rel >= 0.0 && rel < 1e-5,
       "rel_l2 " + std::to_string(rel) + ", worst " + std::to_string(worst));

    /* A scale handed to the bf16 form is refused. */
    ts[1] = T(d_h, RAD_F32, { M, 1 });
    rc = g->launch(&a, nullptr);
    ok((what + ": refuses a scale operand").c_str(), rc != RAD_OK, "rc=" + std::to_string(rc));
    dfree_all();
}

/* The bf16 grouped MoE GEMM against a host product, every form against every other.
 *
 * The routing is skewed as in moe_perf (lognormal popularity, Gumbel top-k), and the last sorted
 * rows are dropped slots (-1, past the live count) whose output must be zeros, not the seeded
 * stale bytes. Every launcher form runs on the same bytes and must match the first byte for byte:
 * each element is one wave's accumulation over K in one order, whichever block computes it.
 *
 * The weights are also passed as a pointer table, as the engine does, with the experts in permuted
 * order behind it where two copies fit, so an entry addressed by arithmetic rather than through
 * the table reads another expert.
 *
 * Negative control: the down projection with `a_order` swapped to "token" is legal but gathers the
 * wrong rows, and must fail the comparison. */
struct MoeBf16Shape { const char* name; int64_t N, K, ne, top_k; bool srt; };

static void moe_gemm_bf16_case(const Plugin& pl, int64_t Mtok, const MoeBf16Shape& sh,
                               bool control = false, int64_t seg = 0) {
    static const char* const kForm[] = { "rule", "aligned prefill form", "decode form",
                                         "uniform prefill grid" };
    const RadKernelInfo* r = find(pl, "moe_gemm_bf16");
    const std::string what = std::string("moe_gemm_bf16 vs an f64 host product (") + sh.name +
                             ", " + std::to_string(Mtok) + " tokens" +
                             (seg ? ", serving the last " + std::to_string(seg) + " experts" : "") +
                             ")" + (control ? " CONTROL: a_order swapped" : "");
    if (!r) { ok(what.c_str(), false, "row absent"); return; }

    const int64_t N = sh.N, K = sh.K, E = sh.ne, top_k = sh.top_k, Tr = Mtok * top_k;
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ (uint64_t)(N * 131 + K * 7 + E + Mtok);
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };

    /* Each token's top_k distinct experts, then a stable sort by expert; the last three tokens'
     * slots are dropped. */
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 1.0 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    const int64_t kept = Mtok > 4 ? Mtok - 3 : Mtok;
    for (int64_t t = 0; t < kept; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    int64_t empty = 0, rmax = 0;
    for (int64_t e = 0; e < E; ++e) {
        const int64_t n = (int64_t)by_e[(size_t)e].size();
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)n;
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
        empty += n == 0;
        rmax = std::max(rmax, n);
    }
    const int64_t live = off[(size_t)E];
    sorted.resize((size_t)Tr, -1);
    /* A call serving a trailing segment (a layer's protected experts, sorted last) gets the
     * offsets from the segment's first expert and its weights; rows below the segment belong to
     * another call and must keep the seeded bytes. */
    const int64_t e0 = seg ? E - seg : 0, En = seg ? seg : E;
    const int64_t s0 = off[(size_t)e0];

    /* The activation: one row per token, or per (token, slot) for the down projection. */
    const int64_t AR = sh.srt ? Tr : Mtok;
    std::vector<uint16_t> ah((size_t)(AR * K)), wh((size_t)(E * N * K));
    for (auto& v : ah) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 700.0f);
    for (auto& v : wh) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 9000.0f);

    /* The reference for every output element, in f64 over the exact bf16 inputs. */
    std::vector<double> ref((size_t)(Tr * N), 0.0);
    for (int64_t i = 0; i < live; ++i) {
        const int32_t f = sorted[(size_t)i];
        if (f < 0) continue;
        int64_t e = 0;
        while (e + 1 < E && off[(size_t)(e + 1)] <= i) ++e;
        const int64_t row = sh.srt ? i : f / top_k;
        const uint16_t* ar = &ah[(size_t)(row * K)];
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t* wr = &wh[(size_t)((e * N + n) * K)];
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) acc += (double)h_bf16(ar[k]) * (double)h_bf16(wr[k]);
            ref[(size_t)(i * N + n)] = acc;
        }
    }

    /* Where small, two weight copies: the stacked plane in expert order and a permuted one for the
     * pointer table. Otherwise one copy, which the table points into. */
    const size_t eb = (size_t)(N * K) * 2;
    const bool permute = E * (int64_t)eb <= ((int64_t)16 << 20);
    void* d_w  = dalloc((size_t)E * eb);
    void* d_wp = permute ? dalloc((size_t)E * eb) : d_w;
    void* d_a  = dalloc(ah.size() * 2);
    void* d_s  = dalloc((size_t)Tr * 4);
    void* d_o  = dalloc((size_t)(E + 1) * 4);
    void* d_y  = dalloc((size_t)(Tr * N) * 2);
    void* d_tab = dalloc((size_t)E * sizeof(void*));
    const int64_t map_bytes = (1 + 2 * (E + Tr / 16 + 2)) * 4;
    void* d_map = dalloc((size_t)map_bytes);
    if (!d_w || !d_wp || !d_a || !d_s || !d_o || !d_y || !d_tab || !d_map) {
        ok(what.c_str(), false, "alloc failed");
        dfree_all();
        return;
    }
    st_memcpy(d_w, wh.data(), wh.size() * 2, hipMemcpyHostToDevice);
    std::vector<const void*> tab((size_t)E);
    if (permute) {
        std::vector<int64_t> slot((size_t)E);
        for (int64_t e = 0; e < E; ++e) slot[(size_t)e] = (e * 37 + 11) % E;
        bool bij = true;
        std::vector<char> seen((size_t)E, 0);
        for (int64_t e = 0; e < E; ++e) bij &= !seen[(size_t)slot[(size_t)e]]++;
        for (int64_t e = 0; e < E; ++e) {
            const int64_t s = bij ? slot[(size_t)e] : (E - 1 - e);
            st_memcpy((char*)d_wp + (size_t)s * eb, &wh[(size_t)(e * N * K)], eb,
                      hipMemcpyHostToDevice);
            tab[(size_t)e] = (const char*)d_wp + (size_t)s * eb;
        }
    } else {
        for (int64_t e = 0; e < E; ++e) tab[(size_t)e] = (const char*)d_w + (size_t)e * eb;
    }
    st_memcpy(d_tab, tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);
    st_memcpy(d_a, ah.data(), ah.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_s, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), off.size() * 4, hipMemcpyHostToDevice);

    const char* order = (sh.srt != control) ? "sorted" : "token";
    std::vector<RadParam> ps = {
        PI("M", Mtok), PI("N", N), PI("K", K), PI("n_expert", En), PI("top_k", top_k),
        PS("a_order", order), PS("dtype", "bf16"),
    };
    std::vector<uint16_t> first, got((size_t)(Tr * N));
    int64_t bad_total = 0;
    /* Stacked (form loop) and then the pointer table at the rule's form. */
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<RadTensor> ts = {
            T(d_a, RAD_BF16, { AR, K }),
            T(pass ? (void*)((const void**)d_tab + e0) : (void*)((char*)d_w + (size_t)e0 * eb),
              RAD_BF16, { En, N, K }),
            T(d_s, RAD_I32, { Tr }),
            T((int32_t*)d_o + e0, RAD_I32, { En + 1 }),
            T(d_y, RAD_BF16, { Tr, N }),
        };
        if (pass) ts[1].stride[0] = 0;
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int f_lo = (pass == 0 && pl.pin_moe) ? 1 : 0, f_hi = (pass == 0 && pl.pin_moe) ? 3 : 0;
        for (int form = f_lo; form <= f_hi; ++form) {
            const std::string label = what + " " + (pass ? "pointer table" : kForm[form]);
            if (pl.pin_moe) pl.pin_moe(form);
            {
                std::vector<uint16_t> junk((size_t)(Tr * N), 0x7f7fu);
                st_memcpy(d_y, junk.data(), junk.size() * 2, hipMemcpyHostToDevice);
            }
            a.scratch = form == 3 ? nullptr : d_map;
            a.scratch_bytes = form == 3 ? 0 : map_bytes;
            const int rc = r->launch(&a, nullptr);
            st_stream_sync(nullptr);
            if (pl.pin_moe) pl.pin_moe(0);
            if (rc != RAD_OK) { ok(label.c_str(), false, "launch rc=" + std::to_string(rc)); continue; }
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);

            double worst = 0.0, rel_max = 0.0;
            int32_t ulp_max = 0;
            int64_t bad = 0;
            std::string firstbad;
            for (int64_t i = 0; i < Tr; ++i)
                for (int64_t n = 0; n < N; ++n) {
                    if (i < s0) {
                        if (got[(size_t)(i * N + n)] != 0x7f7fu && !bad++)
                            firstbad = ", row " + std::to_string(i) + " below the segment was "
                                       "written";
                        continue;
                    }
                    const double want = ref[(size_t)(i * N + n)];
                    const double gf = (double)h_bf16(got[(size_t)(i * N + n)]);
                    /* Half a bf16 ulp plus room for a K-term f32 sum in the WMMA's order; NaN is a
                     * miss. `rel_max`, the worst element as a fraction of that, is printed. */
                    const double err = std::fabs(gf - want);
                    const double tol = 0.004 * std::fabs(want) + 2e-3;
                    if (err / tol > rel_max) rel_max = err / tol;
                    /* And in bf16 ulps from the once-rounded f64 product, on the sign-magnitude
                     * ordering of the bits. */
                    {
                        const uint16_t rb = h_to_bf16((float)want), gb = got[(size_t)(i * N + n)];
                        const int32_t ro = (rb & 0x8000) ? -(int32_t)(rb & 0x7fff) : (int32_t)rb;
                        const int32_t go = (gb & 0x8000) ? -(int32_t)(gb & 0x7fff) : (int32_t)gb;
                        const int32_t du = ro > go ? ro - go : go - ro;
                        if (du > ulp_max) ulp_max = du;
                    }
                    if (!(err <= tol)) {
                        if (!bad++)
                            firstbad = ", first at row " + std::to_string(i) + " col " +
                                       std::to_string(n) + ": " + std::to_string(gf) + " for " +
                                       std::to_string(want);
                        if (!(err <= worst)) worst = err;
                    }
                }
            bad_total += bad;
            if (control) continue;
            std::string same;
            if (first.empty()) {
                first = got;
                std::printf("        worst element %.3f of the tolerance, %d bf16 ulp from the rounded "
                            "f64 product\n", rel_max, (int)ulp_max);
            }
            else if (std::memcmp(first.data(), got.data(), got.size() * 2) != 0) {
                int64_t ndiff = 0;
                for (size_t q = 0; q < got.size(); ++q) ndiff += got[q] != first[q];
                same = ", " + std::to_string(ndiff) + " elements differ from the first form";
                ++bad;
            }
            ok(label.c_str(), bad == 0,
               std::to_string(bad) + " of " + std::to_string(Tr * N) + " elements off, worst " +
                   std::to_string(worst) + firstbad + same);
        }
        if (control) break;
    }
    if (control)
        ok(what.c_str(), bad_total > 0,
           "the swapped order agreed with the reference, so the comparison sees nothing");
    else
        std::printf("        %lld routed rows, %lld live, %lld experts empty, longest run %lld\n",
                    (long long)Tr, (long long)live, (long long)empty, (long long)rmax);
    dfree_all();
}

/* The bf16 expert GEMM at a prefill chunk: Qwen3.8-Flash-Next's whole experts (gate_up 1280 x 2560
 * over token rows, down 2560 x 640 over sorted rows) under a skewed routing, every form timed and
 * compared with the first byte for byte.
 *
 * `distinct` planes back the table entries (entry e reads plane e % distinct). Equal to `experts`
 * it is the real weight traffic; smaller, the planes may sit in the last-level cache and the time
 * is a floor, as the printed line says. */
static void moe16_perf(const Plugin& pl, int64_t Mtok, int iters, int64_t E, int64_t top_k,
                       int64_t distinct) {
    if (iters <= 0) iters = 20;
    if (distinct <= 0 || distinct > E) distinct = E;
    const RadKernelInfo* r = find(pl, "moe_gemm_bf16");
    if (!r) { std::printf("--perfmoe16: row moe_gemm_bf16 absent\n"); return; }
    const int64_t Tr = Mtok * top_k;

    uint64_t rs = 0x9E3779B97F4A7C15ull;
    auto rnd  = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
    auto unif = [&]() { return ((double)(rnd() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    std::vector<double> logp((size_t)E);
    for (int64_t e = 0; e < E; ++e)
        logp[(size_t)e] = 0.8 * std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307 * unif());
    std::vector<std::vector<int32_t>> by_e((size_t)E);
    std::vector<std::pair<double, int>> sc((size_t)E);
    for (int64_t t = 0; t < Mtok; ++t) {
        for (int64_t e = 0; e < E; ++e)
            sc[(size_t)e] = { logp[(size_t)e] - std::log(-std::log(unif())), (int)e };
        std::partial_sort(sc.begin(), sc.begin() + top_k, sc.end(),
                          [](const std::pair<double, int>& x, const std::pair<double, int>& y) {
                              return x.first > y.first;
                          });
        for (int64_t j = 0; j < top_k; ++j)
            by_e[(size_t)sc[(size_t)j].second].push_back((int32_t)(t * top_k + j));
    }
    std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
    sorted.reserve((size_t)Tr);
    for (int64_t e = 0; e < E; ++e) {
        off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)by_e[(size_t)e].size();
        sorted.insert(sorted.end(), by_e[(size_t)e].begin(), by_e[(size_t)e].end());
    }
    std::printf("moe_gemm_bf16 at a %lld-token chunk: %lld routed rows over %lld experts, %lld "
                "distinct weight planes%s, %d launches\n", (long long)Mtok, (long long)Tr,
                (long long)E, (long long)distinct,
                distinct < E ? " (planes alias: a cache-resident floor, not the real traffic)" : "",
                iters);

    void* d_s = dalloc((size_t)Tr * 4);
    void* d_o = dalloc((size_t)(E + 1) * 4);
    if (!d_s || !d_o) { std::printf("  alloc failed\n"); return; }
    st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
    st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);

    struct Proj { const char* name; int64_t N, K; bool srt; };
    const Proj projs[2] = { { "gate_up", 1280, 2560, false }, { "down", 2560, 640, true } };
    const int nforms = pl.pin_moe ? 3 : 1;
    for (const Proj& P : projs) {
        const int64_t Ma = P.srt ? Tr : Mtok;
        const size_t eb = (size_t)(P.N * P.K) * 2;
        std::vector<uint16_t> hb((size_t)std::max<int64_t>(Ma * P.K, P.N * P.K));
        for (auto& v : hb) v = h_to_bf16((float)((double)(int64_t)(rnd() % 2001) - 1000.0) / 3000.0f);
        void* d_a = dalloc((size_t)(Ma * P.K) * 2);
        void* d_w = dalloc((size_t)distinct * eb);
        void* d_tab = dalloc((size_t)E * sizeof(void*));
        void* d_y = dalloc((size_t)(Tr * P.N) * 2);
        if (!d_a || !d_w || !d_tab || !d_y) {
            std::printf("  %-8s alloc failed (%.1f MiB of planes)\n", P.name,
                        (double)distinct * (double)eb / 1048576.0);
            dfree_all();
            return;
        }
        st_memcpy(d_a, hb.data(), (size_t)(Ma * P.K) * 2, hipMemcpyHostToDevice);
        for (int64_t x = 0; x < distinct; ++x)
            st_memcpy((char*)d_w + (size_t)x * eb, hb.data(), eb, hipMemcpyHostToDevice);
        std::vector<const void*> tab((size_t)E);
        for (int64_t e = 0; e < E; ++e) tab[(size_t)e] = (const char*)d_w + (size_t)(e % distinct) * eb;
        st_memcpy(d_tab, tab.data(), tab.size() * sizeof(void*), hipMemcpyHostToDevice);

        std::vector<RadTensor> ts = {
            T(d_a, RAD_BF16, { Ma, P.K }),
            T(d_tab, RAD_BF16, { E, P.N, P.K }),
            T(d_s, RAD_I32, { Tr }),
            T(d_o, RAD_I32, { E + 1 }),
            T(d_y, RAD_BF16, { Tr, P.N }),
        };
        ts[1].stride[0] = 0;
        std::vector<RadParam> ps = {
            PI("M", Mtok), PI("N", P.N), PI("K", P.K), PI("n_expert", E), PI("top_k", top_k),
            PS("a_order", P.srt ? "sorted" : "token"), PS("dtype", "bf16"),
        };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        {
            const int64_t need = std::max<int64_t>(r->scratch ? r->scratch(&a) : 0,
                                                   (1 + 2 * (E + Tr / 16 + 2)) * 4);
            a.scratch = dalloc((size_t)need);
            a.scratch_bytes = need;
        }
        const double flop = 2.0 * (double)Tr * (double)P.N * (double)P.K;
        std::vector<uint16_t> first((size_t)(Tr * P.N)), got((size_t)(Tr * P.N));
        for (int f = 0; f < nforms; ++f) {
            if (pl.pin_moe) pl.pin_moe(f + 1);
            const int rc = r->launch(&a, nullptr);
            if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
                std::printf("  %-8s form %d  rc=%d (not served)\n", P.name, f + 1, rc);
                if (pl.pin_moe) pl.pin_moe(0);
                continue;
            }
            for (int i = 0; i < 3; ++i) r->launch(&a, nullptr);
            const double us = st_time_us([&](int) { r->launch(&a, nullptr); }, iters, nullptr);
            if (pl.pin_moe) pl.pin_moe(0);
            st_memcpy(got.data(), d_y, got.size() * 2, hipMemcpyDeviceToHost);
            std::string same;
            if (f == 0) first = got;
            else same = std::memcmp(first.data(), got.data(), got.size() * 2) ? "  DIFFERS from form 1"
                                                                            : "  = form 1";
            std::printf("  %-8s form %d  %9.1f us  %7.1f TFLOP/s  %7.1f GB/s of planes%s\n", P.name,
                        f + 1, us, flop / (us * 1e6), (double)E * (double)eb / (us * 1e3),
                        same.c_str());
        }
        dfree_all();
        d_s = dalloc((size_t)Tr * 4);
        d_o = dalloc((size_t)(E + 1) * 4);
        if (!d_s || !d_o) return;
        st_memcpy(d_s, sorted.data(), (size_t)Tr * 4, hipMemcpyHostToDevice);
        st_memcpy(d_o, off.data(), (size_t)(E + 1) * 4, hipMemcpyHostToDevice);
    }
}

/* ------------------------------------------------------------------ the two-rank all-reduce */

struct ArThread {
    const RadKernelInfo* row;
    int rank, dev;
    int64_t numel;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_E_STATE;
    std::vector<float> out;
    /* The all-gather row width in elements: 0 is plain concatenation, otherwise the
     * row-interleaved placement the vocab-parallel sampler needs. */
    int64_t row_elems = 0;
    /* Ranks in the collective: 2, or 3 for the wide family at a width that is not a power of
     * two. */
    int world = 2;
};

static void ar_worker(ArThread* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", t->numel), PS("dtype", "fp32"), PI("exact", 1),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;

    void* in = nullptr;
    void* out = nullptr;
    if (hipMalloc(&in, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    if (hipMalloc(&out, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    std::vector<float> h((size_t)t->numel, (float)(t->rank + 1));
    st_memcpy(in, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    st_memset(out, 0, (size_t)t->numel * 4);

    RadTensor ts[2] = { T(in, RAD_F32, { t->numel }), T(out, RAD_F32, { t->numel }) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts; a.n_t = 2;
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.instance = t->inst;
    a.rank = t->rank;
    a.world_size = t->world;
    t->rc_launch = t->row->launch(&a, nullptr);
    st_stream_sync(nullptr);

    t->out.assign((size_t)t->numel, 0.0f);
    st_memcpy(t->out.data(), out, t->out.size() * 4, hipMemcpyDeviceToHost);
    st_free(in);
    st_free(out);
}

static void ar_two_rank(const Plugin& pl, const std::vector<int>& devs, const char* name) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if (devs.size() < 2) { skip(name, "fewer than two gfx1201 cards"); return; }

    /* 8192 fp32 elements: whole 16-byte pushes, small enough to exercise the handshake rather
     * than the link. */
    const int64_t numel = 8192;
    ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel };
    std::thread ta(ar_worker, &a), tb(ar_worker, &b);
    ta.join();
    tb.join();

    ok("2-rank all-reduce init (peer rendezvous)",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-reduce launch",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch));

    int64_t bad = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (a.out[(size_t)i] != 3.0f) ++bad;
        if (b.out[(size_t)i] != 3.0f) ++bad;
    }
    ok("2-rank all-reduce sums 1 + 2 == 3 on both ranks, bit for bit", bad == 0,
       std::to_string(bad) + " elements wrong");

    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }
    ok("2-rank all-reduce fini ran", true);
}

/* ------------------------------------------------------------------ changing block counts */
/* Back-to-back all-reduces on one instance at different block counts. The block count follows the
 * message, so ops sharing an instance run different grids, and consecutive launches must alternate
 * scratch slots as a whole (per-block slots would let one launch push into bytes the previous one
 * is still reducing; r4d_ar_seq.h). Nothing synchronises between launches and each has its own
 * values and output, so a shared slot shows as a wrong sum and a counter mismatch as a hang. The
 * sizes give 11, 4 and 7 blocks under the default block rule. */
static const int64_t kArSweepNumel[] = { 65536, 8192, 40000, 65536, 12000, 24576, 65536, 8192 };
constexpr int kArSweepN = (int)(sizeof(kArSweepNumel) / sizeof(kArSweepNumel[0]));
/* Rounds of the sweep with new values each, so every slot is reused many times; a reader served a
 * slot's previous contents (a stale cached line, a flag seen before its bytes) yields a wrong sum,
 * since every launch's value is distinct. */
constexpr int kArSweepRounds = 32;

struct ArSweep {
    const RadKernelInfo* row;
    int rank, dev;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    int64_t bad = 0;
    int world = 2;
};

static void ar_sweep_worker(ArSweep* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    int64_t most = 0;
    for (int64_t n : kArSweepNumel) most = n > most ? n : most;
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", most), PS("dtype", "fp32"), PI("exact", 1),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;
    /* Rank r contributes (r + 1) * v, so a launch with value v sums to (1 + 2 + .. + world) * v. */
    const float tri = (float)(t->world * (t->world + 1) / 2);

    void* in[kArSweepN] = {};
    void* out[kArSweepN] = {};
    for (int j = 0; j < kArSweepN; ++j) {
        const size_t bytes = (size_t)kArSweepNumel[j] * 4;
        if (hipMalloc(&in[j], bytes) != hipSuccess || hipMalloc(&out[j], bytes) != hipSuccess) {
            t->rc_launch = RAD_E_NOMEM;
            return;
        }
    }
    for (int round = 0; round < kArSweepRounds && t->rc_launch == RAD_OK; ++round) {
        for (int j = 0; j < kArSweepN; ++j) {
            const size_t bytes = (size_t)kArSweepNumel[j] * 4;
            const float v = (float)(round * kArSweepN + j + 1);
            std::vector<float> h((size_t)kArSweepNumel[j], (float)(t->rank + 1) * v);
            st_memcpy(in[j], h.data(), bytes, hipMemcpyHostToDevice);
            st_memset(out[j], 0, bytes);
        }
        for (int j = 0; j < kArSweepN && t->rc_launch == RAD_OK; ++j) {
            RadTensor ts[2] = { T(in[j], RAD_F32, { kArSweepNumel[j] }),
                                T(out[j], RAD_F32, { kArSweepNumel[j] }) };
            RadArgs a;
            std::memset(&a, 0, sizeof(a));
            a.t = ts; a.n_t = 2;
            a.p = ps.data(); a.n_p = (int)ps.size();
            a.instance = t->inst;
            a.rank = t->rank;
            a.world_size = t->world;
            t->rc_launch = t->row->launch(&a, nullptr);
        }
        st_stream_sync(nullptr);
        for (int j = 0; j < kArSweepN; ++j) {
            const float v = (float)(round * kArSweepN + j + 1);
            std::vector<float> h((size_t)kArSweepNumel[j]);
            st_memcpy(h.data(), out[j], h.size() * 4, hipMemcpyDeviceToHost);
            for (float x : h) if (x != tri * v) ++t->bad;
        }
    }
    for (int j = 0; j < kArSweepN; ++j) {
        st_free(in[j]);
        st_free(out[j]);
    }
}

static void ar_two_rank_sweep(const Plugin& pl, const std::vector<int>& devs, const char* name) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if (devs.size() < 2) { skip(name, "fewer than two gfx1201 cards"); return; }
    ArSweep a{ row, 0, devs[0] }, b{ row, 1, devs[1] };
    std::thread ta(ar_sweep_worker, &a), tb(ar_sweep_worker, &b);
    ta.join();
    tb.join();
    ok("2-rank all-reduce init at the widest message",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-reduce, 32 rounds of launches back to back at changing block counts, every "
       "sum exact on both ranks",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK && a.bad == 0 && b.bad == 0,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch) + ", " +
       std::to_string(a.bad + b.bad) + " elements wrong");
    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }
}

/* ------------------------------------------------------------------ the two-rank all-gather */
/* The payload is what the vocab-parallel sampler gathers: (u32 global token id, f32 logit) pairs.
 * A token id's bit pattern read as f32 is a denormal, so an all-gather emulated by an all-reduce
 * over zero-padded planes would flush ids to zero where denormals flush. This checks the bytes on
 * both ranks, for both halves. */
static void gather_worker(ArThread* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", t->numel), PS("dtype", "f32"),
    };
    if (t->row_elems > 0) ps.push_back(PI("row", t->row_elems));
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;

    void* in = nullptr;
    void* out = nullptr;
    if (hipMalloc(&in, (size_t)t->numel * 4) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    if (hipMalloc(&out, (size_t)t->numel * 4 * t->world) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }

    std::vector<uint32_t> h((size_t)t->numel);
    for (int64_t i = 0; i < t->numel; i += 2) {
        h[(size_t)i] = (uint32_t)(t->rank * 124160 + (int)i + 1);          /* a real token id */
        const float v = (float)(t->rank + 1) * 0.5f + (float)i;            /* its logit */
        std::memcpy(&h[(size_t)i + 1], &v, 4);
    }
    st_memcpy(in, h.data(), h.size() * 4, hipMemcpyHostToDevice);
    st_memset(out, 0xAB, (size_t)t->numel * 4 * t->world);   /* marks untouched slices */

    RadTensor ts[2] = { T(in, RAD_F32, { t->numel }), T(out, RAD_F32, { t->numel * t->world }) };
    RadArgs a;
    std::memset(&a, 0, sizeof(a));
    a.t = ts; a.n_t = 2;
    a.p = ps.data(); a.n_p = (int)ps.size();
    a.instance = t->inst;
    a.rank = t->rank;
    a.world_size = t->world;
    t->rc_launch = t->row->launch(&a, nullptr);
    st_stream_sync(nullptr);

    t->out.assign((size_t)t->numel * t->world, 0.0f);
    st_memcpy(t->out.data(), out, t->out.size() * 4, hipMemcpyDeviceToHost);
    st_free(in);
    st_free(out);
}

static void gather_two_rank(const Plugin& pl, const std::vector<int>& devs, const char* name) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if (devs.size() < 2) { skip(name, "fewer than two gfx1201 cards"); return; }

    const int64_t numel = 8192;   /* the sampler's own message at 4 rows and 1024 candidates */
    ArThread a{ row, 0, devs[0], numel }, b{ row, 1, devs[1], numel };
    std::thread ta(gather_worker, &a), tb(gather_worker, &b);
    ta.join();
    tb.join();

    ok("2-rank all-gather init (peer rendezvous)",
       a.rc_init == RAD_OK && b.rc_init == RAD_OK,
       "rc " + std::to_string(a.rc_init) + "/" + std::to_string(b.rc_init));
    if (a.rc_init != RAD_OK || b.rc_init != RAD_OK) return;
    ok("2-rank all-gather launch",
       a.rc_launch == RAD_OK && b.rc_launch == RAD_OK,
       "rc " + std::to_string(a.rc_launch) + "/" + std::to_string(b.rc_launch));

    /* What each rank sent, rebuilt on the host rather than read from the device. */
    auto sent = [numel](int rank, int64_t i) -> uint32_t {
        if ((i & 1) == 0) return (uint32_t)(rank * 124160 + (int)i + 1);
        uint32_t u; const float v = (float)(rank + 1) * 0.5f + (float)(i - 1);
        std::memcpy(&u, &v, 4);
        return u;
    };
    int64_t bad = 0, bad_id = 0;
    for (const ArThread* t : { &a, &b }) {
        for (int r = 0; r < 2; ++r) {
            for (int64_t i = 0; i < numel; ++i) {
                uint32_t got;
                std::memcpy(&got, &t->out[(size_t)(r * numel + i)], 4);
                if (got == sent(r, i)) continue;
                ++bad;
                if ((i & 1) == 0) ++bad_id;
            }
        }
    }
    ok("2-rank all-gather places both ranks' bytes on both ranks, bit for bit", bad == 0,
       std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");

    if (row->fini) { row->fini(a.inst); row->fini(b.inst); }

    /* The row-interleaved placement the sampler uses: input [rows][row], output [rows][2][row],
     * rank 1's row m directly after rank 0's. A concatenating gather passes the check above and
     * fails this one. */
    const int64_t rowe = 2048;                     /* 4 rows of 1024 (id, logit) pairs */
    ArThread c{ row, 0, devs[0], numel }, d{ row, 1, devs[1], numel };
    c.row_elems = d.row_elems = rowe;
    std::thread tc(gather_worker, &c), td(gather_worker, &d);
    tc.join();
    td.join();
    if (c.rc_init != RAD_OK || d.rc_init != RAD_OK || c.rc_launch != RAD_OK ||
        d.rc_launch != RAD_OK) {
        ok("2-rank all-gather, row-interleaved", false,
           "rc " + std::to_string(c.rc_init) + "/" + std::to_string(d.rc_init) + " init, " +
           std::to_string(c.rc_launch) + "/" + std::to_string(d.rc_launch) + " launch");
        return;
    }
    const int64_t nrows = numel / rowe;
    bad = bad_id = 0;
    for (const ArThread* t : { &c, &d }) {
        for (int64_t m = 0; m < nrows; ++m) {
            for (int r = 0; r < 2; ++r) {
                for (int64_t j = 0; j < rowe; ++j) {
                    uint32_t got;
                    std::memcpy(&got, &t->out[(size_t)(m * 2 * rowe + r * rowe + j)], 4);
                    if (got == sent(r, m * rowe + j)) continue;
                    ++bad;
                    if ((j & 1) == 0) ++bad_id;
                }
            }
        }
    }
    ok("2-rank all-gather interleaves by row, so one row carries both ranks", bad == 0,
       std::to_string(bad) + " words wrong, " + std::to_string(bad_id) + " of them token ids");

    if (row->fini) { row->fini(c.inst); row->fini(d.inst); }
    ok("2-rank all-gather fini ran", true);
}

/* ------------------------------------------------------------------ the wide (3/4/8-rank) family */
/* The wide kernels are templated on the world size; these cases run them at 3 and 4 ranks (4 is
 * the TP4 width). Checked: a rendezvous of `world`, a sum of 1 + 2 + .. + world on every rank bit
 * for bit, slot parity surviving a block-count sweep, and the gather placing every slice in both
 * layouts. Skipped on fewer cards than the width, with the width named in the skip. */
static void ar_wide_rank(const Plugin& pl, const std::vector<int>& devs, const char* name,
                         int world) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if ((int)devs.size() < world) {
        skip(name, (std::string("needs ") + std::to_string(world) + " gfx1201 cards").c_str());
        return;
    }
    const int64_t numel = 8192;
    const std::string w = std::to_string(world);
    std::vector<ArThread> t;
    std::vector<std::thread> th;
    for (int r = 0; r < world; ++r) t.push_back(ArThread{ row, r, devs[(size_t)r], numel });
    for (int r = 0; r < world; ++r) { t[(size_t)r].world = world; th.emplace_back(ar_worker, &t[(size_t)r]); }
    for (std::thread& x : th) x.join();
    bool up = true;
    std::string rcs;
    for (const ArThread& a : t) { up = up && a.rc_init == RAD_OK; rcs += std::to_string(a.rc_init) + "/"; }
    ok((w + "-rank all-reduce init (peer rendezvous)").c_str(), up, rcs);
    if (!up) return;
    up = true; rcs.clear();
    for (const ArThread& a : t) { up = up && a.rc_launch == RAD_OK; rcs += std::to_string(a.rc_launch) + "/"; }
    ok((w + "-rank all-reduce launch").c_str(), up, rcs);
    /* Rank r contributes (r + 1), so the fold is the triangular number -- 6 at three, 10 at four. */
    const float want = (float)(world * (world + 1) / 2);
    int64_t bad = 0;
    for (const ArThread& a : t)
        for (int64_t i = 0; i < numel; ++i) if (a.out[(size_t)i] != want) ++bad;
    ok((w + "-rank all-reduce sums 1 + 2 + .. + " + w + " == " + std::to_string((int)want) +
        " on every rank, bit for bit").c_str(), bad == 0,
       std::to_string(bad) + " elements wrong");
    if (row->fini) for (ArThread& a : t) row->fini(a.inst);
    ok((w + "-rank all-reduce fini ran").c_str(), true);

    /* Block counts that change launch to launch on one instance: the wide family takes its slot
     * parity from the shared counters exactly as the pair does, and this is the check that it
     * does so at a width whose flag array is [block][world]. */
    std::vector<ArSweep> s;
    std::vector<std::thread> sh2;
    for (int r = 0; r < world; ++r) s.push_back(ArSweep{ row, r, devs[(size_t)r] });
    for (int r = 0; r < world; ++r) { s[(size_t)r].world = world; sh2.emplace_back(ar_sweep_worker, &s[(size_t)r]); }
    for (std::thread& x : sh2) x.join();
    up = true; rcs.clear();
    int64_t bad2 = 0;
    for (const ArSweep& a : s) {
        up = up && a.rc_init == RAD_OK && a.rc_launch == RAD_OK;
        bad2 += a.bad;
        rcs += std::to_string(a.rc_init) + "/" + std::to_string(a.rc_launch) + " ";
    }
    ok((w + "-rank all-reduce, 32 rounds of eight launches at changing block counts on one "
        "instance").c_str(),
       up && bad2 == 0,
       up ? std::to_string(bad2) + " elements wrong" : "init/launch rc " + rcs);
    if (row->fini) for (ArSweep& a : s) if (a.inst) row->fini(a.inst);
}

/* The three-rank all-reduce rows, at the sizes that select the two-shot and sizes that do not
 * divide.
 *
 * Every element has its own value, so a word reduced by the wrong owner, gathered from the wrong
 * slot or dropped at a short shard's end shows at a known index. The sizes walk the uneven split
 * (words % 3 of 0, 1 and 2) from 16 words to past 20971520, in place and once out of place, back
 * to back on one instance (slot parity, changing block counts). The oracle is the kernel's fold:
 * fp32 in ascending rank order, rounded once, bit for bit; it is also the one-shot's answer, so
 * the band edge between the rows changes no bits. */
struct ArPattern {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0;
    int world = 3;
    const char* dtype = nullptr;       /* "bf16" or "fp32" */
    std::vector<int64_t> sizes;
    int out_of_place = -1;             /* index into sizes, or -1 */
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<std::vector<float>> out;
};

static float ar_pattern_value(int rank, int64_t i) {
    /* Integers up to 254 over 64: exact in bf16 (8 significant bits), while sums of three need
     * ten, so the bf16 sum's rounding is checked too. */
    return (float)((int)((i * 37 + (int64_t)rank * 101) % 509) - 254) / 64.0f;
}

static void ar_pattern_worker(ArPattern* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    const bool bf = !std::strcmp(t->dtype, "bf16");
    const int esz = bf ? 2 : 4;
    int64_t most = 0;
    for (int64_t n : t->sizes) most = n > most ? n : most;
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", most), PS("dtype", t->dtype), PI("exact", 1),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;
    void* in = nullptr;
    void* out = nullptr;
    if (hipMalloc(&in, (size_t)most * esz) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    if (hipMalloc(&out, (size_t)most * esz) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    std::vector<uint8_t> h((size_t)most * esz);
    for (size_t k = 0; k < t->sizes.size(); ++k) {
        const int64_t n = t->sizes[k];
        for (int64_t i = 0; i < n; ++i) {
            const float v = ar_pattern_value(t->rank, i);
            if (bf) { const uint16_t b = h_to_bf16(v); std::memcpy(&h[(size_t)i * 2], &b, 2); }
            else    std::memcpy(&h[(size_t)i * 4], &v, 4);
        }
        st_memcpy(in, h.data(), (size_t)n * esz, hipMemcpyHostToDevice);
        const bool oop = (int)k == t->out_of_place;
        if (oop) st_memset(out, 0, (size_t)n * esz);
        const uint32_t dt = bf ? RAD_BF16 : RAD_F32;
        RadTensor ts[2] = { T(in, dt, { n }), T(out, dt, { n }) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = oop ? 2 : 1;
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst;
        a.rank = t->rank;
        a.world_size = t->world;
        const int rc = t->row->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK && t->rc_launch == RAD_OK) t->rc_launch = rc;
        st_memcpy(h.data(), oop ? out : in, (size_t)n * esz, hipMemcpyDeviceToHost);
        std::vector<float> got((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            if (bf) { uint16_t b; std::memcpy(&b, &h[(size_t)i * 2], 2); got[(size_t)i] = h_bf16(b); }
            else    std::memcpy(&got[(size_t)i], &h[(size_t)i * 4], 4);
        }
        t->out.push_back(std::move(got));
    }
    st_free(in);
    st_free(out);
}

static void ar_wide_pattern(const Plugin& pl, const std::vector<int>& devs, const char* name,
                            int world) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if ((int)devs.size() < world) {
        skip(name, (std::string("needs ") + std::to_string(world) + " gfx1201 cards").c_str());
        return;
    }
    struct Run { const char* dtype; std::vector<int64_t> sizes; int oop; };
    /* The three-rank list walks the UNEVEN split (words % 3 of 0, 1 and 2); the four-rank rows
     * cannot be given one -- cArTsExact4/cArTi8Ts4 constrain numel so the words divide evenly
     * (r4d_rows.cpp), and the two-shot's own entry refuses any other message -- so its list keeps
     * the whole-word sizes and the band edges instead. */
    std::vector<Run> runs;
    if (world == 3) {
        runs = {
            /* bf16, the residual: words = numel / 8. 2048 x 2048 rows is words % 3 == 2, x 2049 is
             * 0, x 2050 is 1; 128 is 16 words (1), 24008 is 3001 (1); 20971520 the old ceiling
             * (1); 46137344, Gemma 4's feed-forward reduce at an 8192-token chunk, past it (1). */
            { "bf16", { 4194304, 128, 4196352, 24008, 4198400, 20971520, 46137344, 4194304 }, 2 },
            /* fp32: words = numel / 4. 8196 -> 2049 (0), 4100 -> 1025 (2), 1048580 -> 262145 (2). */
            { "fp32", { 8196, 4100, 1048580 }, 0 },
        };
    } else {
        runs = {
            /* bf16: 524288 the one-shot band's top, 524352 the first message the two-shot's rows
             * take (numel % 64 == 0 either way, so both divide); the shipped chunk sizes; and
             * 4194304 again to close the run on a size already folded. */
            { "bf16", { 4194304, 128, 524288, 524352, 4196352, 20971520, 46137344, 4194304 }, 2 },
            /* fp32: words = numel / 4, whole groups of four either side of the same edge. */
            { "fp32", { 1048580 - 4, 524352 }, 0 },
        };
    }
    for (const Run& run : runs) {
        std::vector<ArPattern> t((size_t)world);
        std::vector<std::thread> th;
        for (int r = 0; r < world; ++r) {
            t[(size_t)r].row = row; t[(size_t)r].rank = r; t[(size_t)r].dev = devs[(size_t)r];
            t[(size_t)r].world = world;
            t[(size_t)r].dtype = run.dtype; t[(size_t)r].sizes = run.sizes;
            t[(size_t)r].out_of_place = run.oop;
            th.emplace_back(ar_pattern_worker, &t[(size_t)r]);
        }
        for (std::thread& x : th) x.join();
        const std::string what = std::string(name) + " " + run.dtype;
        bool up = true, ran = true;
        std::string rcs;
        for (const ArPattern& a : t) {
            up = up && a.rc_init == RAD_OK; ran = ran && a.rc_launch == RAD_OK;
            rcs += std::to_string(a.rc_init) + "/" + std::to_string(a.rc_launch) + " ";
        }
        ok((what + " init (peer rendezvous)").c_str(), up, "init/launch rc " + rcs);
        if (!up) return;
        ok((what + " launches").c_str(), ran, "init/launch rc " + rcs);
        if (ran) {
            const bool bf = !std::strcmp(run.dtype, "bf16");
            for (size_t k = 0; k < run.sizes.size(); ++k) {
                const int64_t n = run.sizes[k];
                int64_t bad = 0, first = -1;
                for (int64_t i = 0; i < n; ++i) {
                    float acc = 0.0f;
                    for (int r = 0; r < world; ++r) {
                        const float v = ar_pattern_value(r, i);
                        acc += bf ? h_bf16(h_to_bf16(v)) : v;
                    }
                    const float want = bf ? h_bf16(h_to_bf16(acc)) : acc;
                    for (int r = 0; r < world; ++r)
                        if (t[(size_t)r].out.size() <= k || t[(size_t)r].out[k][(size_t)i] != want) {
                            if (first < 0) first = i;
                            ++bad;
                        }
                }
                const int64_t words = n * (bf ? 2 : 4) / 16;
                const std::string label =
                    what + " numel " + std::to_string(n) + " (" + std::to_string(words) +
                    " words, % " + std::to_string(world) + " = " +
                    std::to_string(words % world) +
                    ((int)k == run.oop ? ", out of place" : ", in place") +
                    ") == fp32 ascending-rank fold on every rank, bit for bit";
                ok(label.c_str(), bad == 0,
                   std::to_string(bad) + " elements wrong, first at " + std::to_string(first));
            }
        }
        if (row->fini) for (ArPattern& a : t) if (a.inst) row->fini(a.inst);
    }
    /* The four-rank rows' even-shard rule, checked from the one place that can: a message whose
     * 16-byte words do not divide by four must be REFUSED (r4d_ar_twoshot_Nrank_exact.hip shards
     * it evenly and the entry throws on any residual), not launched with a truncated tail. The
     * three-rank row has no such rule -- it splits unevenly by design. */
    if (world == 4 && std::strstr(name, "twoshot")) {
        ArPattern t[4];
        for (int r = 0; r < 4; ++r) {
            t[r].row = row; t[r].rank = r; t[r].dev = devs[(size_t)r]; t[r].world = 4;
            t[r].dtype = "bf16"; t[r].sizes = { 524288, 40 };   /* 40 elems = 5 words: 5 % 4 != 0 */
            t[r].out_of_place = 1;
        }
        std::thread a(ar_pattern_worker, &t[0]), b(ar_pattern_worker, &t[1]),
                    c(ar_pattern_worker, &t[2]), d(ar_pattern_worker, &t[3]);
        a.join(); b.join(); c.join(); d.join();
        int refused = 0;
        for (const ArPattern& x : t) {
            if (x.rc_init == RAD_OK && x.rc_launch == RAD_E_SHAPE) ++refused;
            if (x.inst) row->fini(x.inst);
        }
        ok((std::string(name) + " refuses a 5-word message at 4 ranks (RAD_E_SHAPE, not a launch)").c_str(),
           refused == 4,
           std::to_string(refused) + " of 4 ranks refused, rc " + std::to_string(t[0].rc_launch));
    }
}

/* The three-rank tiered-int8 wire.
 *
 * Exact: every rank ends with the same bytes (the owner stores the decode of what it published),
 * and a message the wire declines (under `min_bytes`, or not whole 32-element groups) equals the
 * exact one-shot's fold bit for bit. Statistical: the rest is within quantisation error of the
 * fp32 sum. The input is the per-element pattern plus an outlier every 997 elements on rank 1
 * (tier 2 keeps a dominant element exact); sizes walk groups % 3 of 0, 1 and 2 up to 20971520, so
 * a wrong owner or slot gives a relative RMS near 1, versus a fraction of a percent for the
 * wire. */
struct ArTi8 {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0;
    int world = 3;
    std::vector<int64_t> sizes;
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<std::vector<uint16_t>> out;
};

static float ar_ti8_value(int rank, int64_t i) {
    if (rank == 1 && i % 997 == 13) return 900.0f;   /* exact in bf16 */
    return ar_pattern_value(rank, i);
}

static void ar_ti8_worker(ArTi8* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    int64_t most = 0;
    for (int64_t n : t->sizes) most = n > most ? n : most;
    std::vector<RadParam> ps = {
        PI("world_size", t->world), PI("numel", most), PS("dtype", "bf16"), PI("exact", 0),
        PI("min_bytes", 131072),
    };
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;
    void* buf = nullptr;
    if (hipMalloc(&buf, (size_t)most * 2) != hipSuccess) { t->rc_launch = RAD_E_NOMEM; return; }
    std::vector<uint16_t> h((size_t)most);
    for (int64_t n : t->sizes) {
        for (int64_t i = 0; i < n; ++i) h[(size_t)i] = h_to_bf16(ar_ti8_value(t->rank, i));
        st_memcpy(buf, h.data(), (size_t)n * 2, hipMemcpyHostToDevice);
        RadTensor ts[1] = { T(buf, RAD_BF16, { n }) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts; a.n_t = 1;
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst;
        a.rank = t->rank;
        a.world_size = t->world;
        const int rc = t->row->launch(&a, nullptr);
        st_stream_sync(nullptr);
        if (rc != RAD_OK && t->rc_launch == RAD_OK) t->rc_launch = rc;
        std::vector<uint16_t> got((size_t)n);
        st_memcpy(got.data(), buf, (size_t)n * 2, hipMemcpyDeviceToHost);
        t->out.push_back(std::move(got));
    }
    st_free(buf);
}

static void ar_wide_ti8(const Plugin& pl, const std::vector<int>& devs, const char* name,
                        int world) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if ((int)devs.size() < world) {
        skip(name, (std::string("needs ") + std::to_string(world) + " gfx1201 cards").c_str());
        return;
    }
    /* 32768 is under the 128 KiB floor and 65544 is not whole groups (at four ranks, not whole
     * SHARD groups -- the 4-rank wire tiles numel by 128): both exact. The rest are
     * lossy: 65536 sits ON the floor (2048 groups, % 3 = 2), then 2048 x 2048 / 2049 / 2050 rows
     * (groups % 3 = 2, 0, 1) and the ceiling (655360 groups, % 3 = 1), and 2048 rows again. */
    const std::vector<int64_t> sizes = { 4194304, 32768, 65536, 4196352, 65544, 4198400, 20971520,
                                         4194304 };
    /* The wire's tile: at three any whole group of 32 goes lossy (the 3-rank entry only asks
     * numel % 32 == 0); at four the shard must be whole groups too, so the fallback rule is
     * numel % (32 * 4) == 0 (r4d_rad_ar_twoshot_4rank_ti8). */
    const int64_t tile = world == 3 ? 32 : 32 * (int64_t)world;
    std::vector<ArTi8> t((size_t)world);
    std::vector<std::thread> th;
    for (int r = 0; r < world; ++r) {
        t[(size_t)r].row = row; t[(size_t)r].rank = r; t[(size_t)r].dev = devs[(size_t)r];
        t[(size_t)r].world = world; t[(size_t)r].sizes = sizes;
        th.emplace_back(ar_ti8_worker, &t[(size_t)r]);
    }
    for (std::thread& x : th) x.join();
    const std::string nm(name);
    bool up = true, ran = true;
    std::string rcs;
    for (const ArTi8& a : t) {
        up = up && a.rc_init == RAD_OK; ran = ran && a.rc_launch == RAD_OK;
        rcs += std::to_string(a.rc_init) + "/" + std::to_string(a.rc_launch) + " ";
    }
    ok((nm + " init (peer rendezvous)").c_str(), up, "init/launch rc " + rcs);
    if (!up) return;
    ok((nm + " launches").c_str(), ran, "init/launch rc " + rcs);
    if (ran) {
        for (size_t k = 0; k < sizes.size(); ++k) {
            const int64_t n = sizes[k];
            const bool lossy = n * 2 >= 131072 && n % tile == 0;
            int64_t diff_ranks = 0, inexact = 0;
            double se = 0.0, sx = 0.0, worst = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const uint16_t g0 = t[0].out[k][(size_t)i];
                for (int r = 1; r < world; ++r) if (t[(size_t)r].out[k][(size_t)i] != g0) ++diff_ranks;
                float acc = 0.0f;
                for (int r = 0; r < world; ++r) acc += h_bf16(h_to_bf16(ar_ti8_value(r, i)));
                const double e = (double)h_bf16(g0) - (double)acc;
                if (!lossy && g0 != h_to_bf16(acc)) ++inexact;
                se += e * e; sx += (double)acc * acc;
                const double rel = std::fabs(e) / (std::fabs((double)acc) + 1.0);
                if (rel > worst) worst = rel;
            }
            const double rrms = sx > 0 ? std::sqrt(se / sx) : 0.0;
            const std::string what = nm + " numel " + std::to_string(n) + " (" +
                                     std::to_string(n / 32) + " groups, % " +
                                     std::to_string(world) + " = " +
                                     std::to_string((n / 32) % world) + ")";
            ok((what + ": every rank holds the same bytes").c_str(), diff_ranks == 0,
               std::to_string(diff_ranks) + " elements differ between ranks");
            if (!lossy) {
                ok((what + ": declined by the wire, so the exact fold, bit for bit").c_str(),
                   inexact == 0, std::to_string(inexact) + " elements not the exact fold");
            } else {
                char d[160];
                std::snprintf(d, sizeof(d), "relative RMS %.4f%%, worst |err|/(|sum|+1) %.4f",
                              rrms * 100.0, worst);
                std::printf("        %s: %s\n", what.c_str(), d);
                ok((what + ": within the int8 wire's error of the fp32 sum (relative RMS < 1%)").c_str(),
                   rrms < 0.01, d);
            }
        }
    }
    if (row->fini) for (ArTi8& a : t) if (a.inst) row->fini(a.inst);
}

static void gather_wide_rank(const Plugin& pl, const std::vector<int>& devs, const char* name,
                             int world) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if ((int)devs.size() < world) {
        skip(name, (std::string("needs ") + std::to_string(world) + " gfx1201 cards").c_str());
        return;
    }
    const int64_t numel = 8192;   /* the sampler's own message at 4 rows and 1024 candidates */
    const int W = world;
    const std::string w = std::to_string(world);
    /* What each rank sent, rebuilt here rather than read back off the device (gather_two_rank). */
    auto sent = [](int rank, int64_t i) -> uint32_t {
        if ((i & 1) == 0) return (uint32_t)(rank * 124160 + (int)i + 1);
        uint32_t u; const float v = (float)(rank + 1) * 0.5f + (float)(i - 1);
        std::memcpy(&u, &v, 4);
        return u;
    };
    {
        std::vector<ArThread> t((size_t)W);
        std::vector<std::thread> th;
        for (int r = 0; r < W; ++r) {
            t[(size_t)r] = ArThread{ row, r, devs[(size_t)r], numel };
            t[(size_t)r].world = W;
            th.emplace_back(gather_worker, &t[(size_t)r]);
        }
        for (std::thread& x : th) x.join();
        bool up = true;
        std::string rcs;
        for (const ArThread& a : t) { up = up && a.rc_init == RAD_OK; rcs += std::to_string(a.rc_init) + "/"; }
        ok((w + "-rank all-gather init (peer rendezvous)").c_str(), up, "rc " + rcs);
        if (!up) return;
        bool ran = true;
        rcs.clear();
        for (const ArThread& a : t) { ran = ran && a.rc_launch == RAD_OK; rcs += std::to_string(a.rc_launch) + "/"; }
        ok((w + "-rank all-gather launch").c_str(), ran, "rc " + rcs);
        int64_t bad = 0, bad_id = 0;
        for (const ArThread& a : t) {
            for (int r = 0; r < W; ++r) {
                for (int64_t i = 0; i < numel; ++i) {
                    uint32_t got;
                    std::memcpy(&got, &a.out[(size_t)(r * numel + i)], 4);
                    if (got == sent(r, i)) continue;
                    ++bad;
                    if ((i & 1) == 0) ++bad_id;
                }
            }
        }
        ok((w + "-rank all-gather places every rank's bytes on every rank, bit for bit").c_str(),
           bad == 0, std::to_string(bad) + " words wrong, " + std::to_string(bad_id) +
                          " of them token ids");
        if (row->fini) for (ArThread& a : t) row->fini(a.inst);
    }

    /* The row-interleaved placement the sampler asks for: [rows][world][row]. */
    const int64_t rowe = 2048;                     /* 4 rows of 1024 (id, logit) pairs */
    std::vector<ArThread> t((size_t)W);
    std::vector<std::thread> th;
    for (int r = 0; r < W; ++r) {
        t[(size_t)r] = ArThread{ row, r, devs[(size_t)r], numel };
        t[(size_t)r].world = W;
        t[(size_t)r].row_elems = rowe;
        th.emplace_back(gather_worker, &t[(size_t)r]);
    }
    for (std::thread& x : th) x.join();
    bool up = true, ran = true;
    std::string rcs;
    for (const ArThread& a : t) {
        up = up && a.rc_init == RAD_OK; ran = ran && a.rc_launch == RAD_OK;
        rcs += std::to_string(a.rc_init) + "/" + std::to_string(a.rc_launch) + " ";
    }
    if (!up || !ran) {
        ok((w + "-rank all-gather, row-interleaved").c_str(), false, "init/launch rc " + rcs);
        return;
    }
    const int64_t nrows = numel / rowe;
    int64_t bad = 0, bad_id = 0;
    for (const ArThread& a : t) {
        for (int64_t m = 0; m < nrows; ++m) {
            for (int r = 0; r < W; ++r) {
                for (int64_t j = 0; j < rowe; ++j) {
                    uint32_t got;
                    std::memcpy(&got, &a.out[(size_t)(m * W * rowe + r * rowe + j)], 4);
                    if (got == sent(r, m * rowe + j)) continue;
                    ++bad;
                    if ((j & 1) == 0) ++bad_id;
                }
            }
        }
    }
    ok((w + "-rank all-gather interleaves by row, so one row carries all " + w + " ranks").c_str(),
       bad == 0, std::to_string(bad) + " words wrong, " + std::to_string(bad_id) +
                     " of them token ids");
    if (row->fini) for (ArThread& a : t) if (a.inst) row->fini(a.inst);
    ok((w + "-rank all-gather fini ran").c_str(), true);
}

/* ------------------------------------------------------------------ the fused write (ar_hc_write*) */
/* The fused all-reduce + gated-residual-write rows, with the MoE gather optionally in front (the
 * rows that end the link hold themselves, r4d_rows.cpp). The oracle is the unfused chain rebuilt
 * on the host in the kernels' own arithmetic: moe_gather's gather word (r4d_ar_gather_word: fp32
 * ascending-slot accumulate, round-to-nearest-even narrows, the shared arm's product rounded
 * before the add), then ar_vsum's fp32 ascending-rank fold rounded once to bf16, then hc_write's
 * word (r4d_ar_hc_write_word, whose pack is round-half-away f2bf rather than the reduce's RTNE,
 * hence its own helper below). The rows promise bytes identical to the chain they replace, so the
 * check is byte-exact. Exception: the sigmoid gate is __expf on the card and std::exp here, which
 * may differ by an f32 ulp and move one bf16 rounding (the tolerance silu_bf16's check names), so
 * a sigmoid shape counts moved bytes and fails only beyond an ulp.
 *
 * Both gather arms are walked: a small M keeps the block slot table in LDS (the launcher widens
 * its own grid), a prefill-sized M takes the inverse table in the scratch (the same launcher's
 * by_inv rule), and each rank is given the scratch or not per shape -- the same M, n and scratch
 * on every rank is what makes the two ranks decide the arm identically. */
struct ArHc {
    const RadKernelInfo* row = nullptr;
    int rank = 0, dev = 0, world = 2;
    int64_t M = 0, n = 0, hc = 2, top_k = 0;   /* top_k 0: the plain fused write */
    const char* act = nullptr;                 /* gather's shared arm: null operand, "none", "sigmoid" */
    bool use_scratch = false;                  /* hand the inverse table's scratch to the gather */
    void* inst = nullptr;
    int rc_init = RAD_E_STATE, rc_launch = RAD_OK;
    std::vector<uint16_t> y_in, inj, h0;       /* what we hand the kernel */
    std::vector<uint16_t> ye, ew, sh, sg;      /* the gather's inputs */
    std::vector<int32_t> sorted;
    std::vector<uint16_t> y_out, h_out;        /* what it leaves behind */
};

static void ar_hc_worker(ArHc* t) {
    if (st_set_device(t->dev) != hipSuccess) { t->rc_init = RAD_E_DEVICE; return; }
    const bool gather = t->top_k > 0;
    const int64_t Tn = t->M * t->top_k;   /* Tn, not T: T() is this file's RadTensor helper below */
    std::vector<RadParam> ps = { PI("world_size", t->world), PI("M", t->M), PI("n", t->n),
                                 PI("hc", t->hc), PS("dtype", "bf16"), PI("wire", 0) };
    if (gather) ps.push_back(PI("top_k", t->top_k));
    if (t->act) ps.push_back(PS("act", t->act));
    t->rc_init = t->row->init(ps.data(), (int)ps.size(), t->rank, t->world, &t->inst);
    if (t->rc_init != RAD_OK) return;

    /* Allocated and freed here, not through dalloc: g_allocs is one host-side list and four
     * rank threads would race on it (every other cross-rank worker does the same). */
    void* y = nullptr; void* inj = nullptr; void* h = nullptr;
    void* ye = nullptr; void* ew = nullptr; void* so = nullptr; void* sh = nullptr; void* sg = nullptr;
    void* scr = nullptr;
    const size_t yb = (size_t)(t->M * t->n * 2), hb = (size_t)(t->M * t->hc * t->n * 2);
    bool mem = hipMalloc(&y, yb) == hipSuccess && hipMalloc(&inj, (size_t)(t->M * t->hc * 2)) ==
               hipSuccess && hipMalloc(&h, hb) == hipSuccess;
    if (gather) mem = mem && hipMalloc(&ye, (size_t)(Tn * t->n * 2)) == hipSuccess &&
                     hipMalloc(&ew, (size_t)(Tn * 2)) == hipSuccess &&
                     hipMalloc(&so, (size_t)(Tn * 4)) == hipSuccess &&
                     (!t->act || (hipMalloc(&sh, yb) == hipSuccess &&
                                  hipMalloc(&sg, (size_t)(t->M * 2)) == hipSuccess));
    if (!mem) { t->rc_launch = RAD_E_NOMEM; }
    else {
        st_memcpy(y, t->y_in.data(), yb, hipMemcpyHostToDevice);
        st_memcpy(inj, t->inj.data(), (size_t)(t->M * t->hc * 2), hipMemcpyHostToDevice);
        st_memcpy(h, t->h0.data(), hb, hipMemcpyHostToDevice);
        if (gather) {
            st_memcpy(ye, t->ye.data(), (size_t)(Tn * t->n * 2), hipMemcpyHostToDevice);
            st_memcpy(ew, t->ew.data(), (size_t)(Tn * 2), hipMemcpyHostToDevice);
            st_memcpy(so, t->sorted.data(), (size_t)(Tn * 4), hipMemcpyHostToDevice);
            if (t->act) {
                st_memcpy(sh, t->sh.data(), yb, hipMemcpyHostToDevice);
                st_memcpy(sg, t->sg.data(), (size_t)(t->M * 2), hipMemcpyHostToDevice);
            }
        }
        std::vector<RadTensor> ts;
        if (gather) {
            ts = { T(ye, RAD_BF16, { Tn, t->n }), T(ew, RAD_BF16, { t->M, t->top_k }),
                   T(so, RAD_I32, { Tn }), T(sh, RAD_BF16, { t->M, t->n }),
                   T(sg, RAD_BF16, { t->M, 1 }), T(y, RAD_BF16, { t->M, t->n }),
                   T(inj, RAD_BF16, { t->M, t->hc }), T(h, RAD_BF16, { t->M, t->hc * t->n }) };
            if (!t->act) { absent(ts[3]); absent(ts[4]); }
        } else {
            ts = { T(y, RAD_BF16, { t->M, t->n }), T(inj, RAD_BF16, { t->M, t->hc }),
                   T(h, RAD_BF16, { t->M, t->hc * t->n }) };
        }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.instance = t->inst;
        a.rank = t->rank;
        a.world_size = t->world;
        const int64_t need = gather && t->row->scratch ? t->row->scratch(&a) : 0;
        if (gather && t->use_scratch) {
            /* The inverse table's bytes exactly: has_inv is `scratch && scratch_bytes >= Tn * 4`,
             * and the row's own hook is what says how big that is. */
            if (need <= 0 || hipMalloc(&scr, (size_t)need) != hipSuccess) {
                t->rc_launch = RAD_E_NOMEM;
            } else {
                st_memset(scr, 0, (size_t)need);
                a.scratch = scr; a.scratch_bytes = need;
            }
        }
        if (t->rc_launch == RAD_OK) {
            t->rc_launch = t->row->launch(&a, nullptr);
            st_stream_sync(nullptr);
            t->y_out.resize((size_t)(t->M * t->n));
            t->h_out.resize((size_t)(t->M * t->hc * t->n));
            st_memcpy(t->y_out.data(), y, yb, hipMemcpyDeviceToHost);
            st_memcpy(t->h_out.data(), h, hb, hipMemcpyDeviceToHost);
        }
    }
    for (void* p : { y, inj, h, ye, ew, so, sh, sg, scr }) if (p) st_free(p);
}

/* hc_write's pack (r4d_gdn_wmma.h f2bf): round-half-AWAY, one add. The reduce's from_f and
 * arg_bf16_rne are round-to-nearest-even; this file's h_to_bf16 is that one, so the write's
 * oracle needs this. */
static uint16_t h_bf16_away(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x8000u) >> 16);
}

static void ar_hc_write_case(const Plugin& pl, const std::vector<int>& devs, const char* name,
                             int world, bool want_gather) {
    const RadKernelInfo* row = find(pl, name);
    if (!row) { ok(name, false, "row absent"); return; }
    if ((int)devs.size() < world) {
        skip(name, (std::string("needs ") + std::to_string(world) + " gfx1201 cards").c_str());
        return;
    }
    struct Shape { int64_t M, n, hc, top_k; const char* act; bool scratch; };
    std::vector<Shape> shapes;
    if (!want_gather) {
        /* A decode step's rows and a chunk's, both under the one-shot bands. */
        shapes = { { 8, 1024, 4, 0, nullptr, false }, { 64, 2560, 2, 0, nullptr, false } };
    } else {
        /* The first two stay under the launcher's by_inv line (its own widening keeps the LDS
         * table at or below eight rows a block); the third is a prefill-sized message where only
         * the inverse table fits. The shared arm: absent, then "none" (the gate multiplies as it
         * is -- exact against the host), then "sigmoid" (the ulp-tolerant arm). */
        shapes = { { 8, 256, 2, 10, nullptr, false }, { 40, 512, 4, 6, "none", false },
                   { 200, 2048, 2, 10, "sigmoid", true } };
    }
    const std::string nm(name);
    for (const Shape& sh : shapes) {
        const int64_t M = sh.M, n = sh.n, hc = sh.hc, k = sh.top_k, T = M * k;
        const bool gather = k > 0;
        const bool sig = sh.act && !std::strcmp(sh.act, "sigmoid");
        const std::string at = " M=" + std::to_string(M) + " n=" + std::to_string(n) +
                               " hc=" + std::to_string(hc) +
                               (gather ? " top_k=" + std::to_string(k) +
                                 (sh.act ? std::string(", shared ") + sh.act
                                         : std::string(", no shared arm"))
                                       : std::string()) +
                               (gather && sh.scratch ? ", inverse table" : "");

        /* The routing is replicated across ranks -- every rank sorts the same slots -- while the
         * routed rows, the weights and the streams are each rank's own. `sorted` is the stable
         * sort by expert with the live slots first and -1 after, as moe_scatter writes it. */
        std::vector<int32_t> sorted((size_t)(gather ? T : 0), -1);
        if (gather) {
            std::vector<int32_t> ex((size_t)T);
            uint64_t rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)M * 1315423911ull) ^
                          ((uint64_t)n * 2654435761ull) ^ (uint64_t)k;
            auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
            for (int64_t i = 0; i < T; ++i) {
                const uint64_t v = rnd();
                ex[(size_t)i] = (v % 31 == 0) ? -1 : (int32_t)((v >> 8) % 64);
            }
            int64_t pos = 0;
            for (int32_t e = 0; e < 64; ++e)
                for (int64_t i = 0; i < T; ++i)
                    if (ex[(size_t)i] == e) sorted[(size_t)pos++] = (int32_t)i;
        }
        /* ye's rows are in sorted order, so the kernel resolves token tok, slot j to the sorted
         * ROW holding slot f = tok*k+j -- the inverse of `sorted` (s_i[f - lo] = i in
         * r4d_ar_oneshot_2rank_exact.hip's scan, inv[f] = i in r4d_ar_gather_inv_kernel). The
         * oracle inverts the same way; reading sorted forward would gather the wrong row. */
        std::vector<int32_t> inv((size_t)(gather ? T : 0), -1);
        for (int64_t i = 0; gather && i < T; ++i) {
            const int32_t f = sorted[(size_t)i];
            if (f >= 0 && f < T) inv[(size_t)f] = (int32_t)i;
        }

        std::vector<ArHc> t((size_t)world);
        std::vector<std::thread> th;
        for (int r = 0; r < world; ++r) {
            ArHc& x = t[(size_t)r];
            x.row = row; x.rank = r; x.dev = devs[(size_t)r]; x.world = world;
            x.M = M; x.n = n; x.hc = hc; x.top_k = k; x.act = sh.act; x.use_scratch = sh.scratch;
            x.y_in.resize((size_t)(M * n));
            x.inj.resize((size_t)(M * hc));
            x.h0.resize((size_t)(M * hc * n));
            uint64_t rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)(r + 1) * 0x9E3779B97F4A7C15ull) ^
                          ((uint64_t)M * 1315423911ull) ^ ((uint64_t)n * 2654435761ull);
            auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
            for (int64_t i = 0; i < M * n; ++i)
                x.y_in[(size_t)i] = gather ? (uint16_t)0x7f7fu      /* overwritten or it is junk */
                                           : h_to_bf16(ar_pattern_value(r, i));
            for (int64_t i = 0; i < M * hc; ++i)
                x.inj[(size_t)i] = h_to_bf16(ar_pattern_value(r + 1, i));
            for (int64_t i = 0; i < M * hc * n; ++i)
                x.h0[(size_t)i] = h_to_bf16(((int64_t)(rnd() % 4001) - 2000) / 512.0f);
            if (gather) {
                x.sorted = sorted;
                x.ye.resize((size_t)(T * n));
                x.ew.resize((size_t)T);
                for (auto& v : x.ye) v = h_to_bf16(((int64_t)(rnd() % 4001) - 2000) / 512.0f);
                for (auto& v : x.ew) v = h_to_bf16((float)(rnd() % 1000 + 1) / 1024.0f);
                if (sh.act) {
                    x.sh.resize((size_t)(M * n));
                    x.sg.resize((size_t)M);
                    for (auto& v : x.sh) v = h_to_bf16(((int64_t)(rnd() % 4001) - 2000) / 256.0f);
                    for (auto& v : x.sg) v = h_to_bf16(((int64_t)(rnd() % 801) - 400) / 100.0f);
                }
            }
            th.emplace_back(ar_hc_worker, &x);
        }
        for (std::thread& x : th) x.join();

        bool up = true, ran = true;
        std::string rcs;
        for (const ArHc& x : t) {
            up = up && x.rc_init == RAD_OK; ran = ran && x.rc_launch == RAD_OK;
            rcs += std::to_string(x.rc_init) + "/" + std::to_string(x.rc_launch) + " ";
        }
        ok((nm + at + " init (peer rendezvous)").c_str(), up, "init/launch rc " + rcs);
        if (!up) return;
        ok((nm + at + " launch").c_str(), ran, "init/launch rc " + rcs);
        if (!ran) { if (row->fini) for (ArHc& x : t) if (x.inst) row->fini(x.inst); continue; }

        /* The gather half, per rank, exactly as r4d_ar_gather_word folds it. */
        std::vector<std::vector<uint16_t>> ymsg((size_t)world, std::vector<uint16_t>((size_t)(M * n)));
        for (int r = 0; r < world; ++r) {
            const ArHc& x = t[(size_t)r];
            if (!gather) { ymsg[(size_t)r] = x.y_in; continue; }
            for (int64_t tok = 0; tok < M; ++tok)
                for (int64_t e = 0; e < n; ++e) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < k; ++j) {
                        const int32_t idx = inv[(size_t)(tok * k + j)];
                        if (idx < 0) continue;
                        acc += h_bf16(x.ew[(size_t)(tok * k + j)]) *
                               h_bf16(x.ye[(size_t)(idx * n + e)]);
                    }
                    uint16_t out = h_to_bf16(acc);
                    if (sh.act) {
                        const float gate = sig ? 1.0f / (1.0f + std::exp(-h_bf16(x.sg[(size_t)tok])))
                                               : h_bf16(x.sg[(size_t)tok]);
                        const uint16_t pr = h_to_bf16(h_bf16(x.sh[(size_t)(tok * n + e)]) * gate);
                        out = h_to_bf16(h_bf16(pr) + h_bf16(out));
                    }
                    ymsg[(size_t)r][(size_t)(tok * n + e)] = out;
                }
        }
        /* The fold, ascending rank, rounded once -- ar_vsum. Then the write -- hc_write_word. */
        int64_t bad_y = 0, moved_y = 0, bad_h = 0, moved_h = 0, first = -1;
        double worst = 0.0;
        for (int64_t i = 0; i < M * n; ++i) {
            float acc = 0.0f;
            for (int r = 0; r < world; ++r) acc += h_bf16(ymsg[(size_t)r][(size_t)i]);
            const uint16_t want_y = h_to_bf16(acc);
            const int64_t tok = i / n, e = i % n;
            for (int r = 0; r < world; ++r) {
                ArHc& x = t[(size_t)r];
                const uint16_t got = x.y_out[(size_t)i];
                if (got != want_y) {
                    const double rel = std::fabs((double)h_bf16(got) - (double)acc) /
                                       (std::fabs((double)acc) + 1.0);
                    if (sig && rel <= 0.02) { ++moved_y; if (rel > worst) worst = rel; }
                    else { ++bad_y; if (first < 0) first = i; }
                }
                for (int64_t c = 0; c < hc; ++c) {
                    const int64_t o = (tok * hc + c) * n + e;
                    const float wantf = h_bf16(x.h0[(size_t)o]) +
                                        h_bf16(x.inj[(size_t)(tok * hc + c)]) * h_bf16(want_y);
                    const uint16_t want = h_bf16_away(wantf);
                    const uint16_t g = x.h_out[(size_t)o];
                    if (g != want) {
                        const double rel = std::fabs((double)h_bf16(g) - (double)wantf) /
                                           (std::fabs((double)wantf) + 1.0);
                        if (sig && rel <= 0.02) { ++moved_h; if (rel > worst) worst = rel; }
                        else { ++bad_h; if (first < 0) first = o; }
                    }
                }
            }
        }
        char d[160];
        std::snprintf(d, sizeof d, "%lld + %lld wrong (first at %lld); %lld + %lld moved by "
                      "__expf vs std::exp, worst rel %.5f", (long long)bad_y, (long long)bad_h,
                      (long long)first, (long long)moved_y, (long long)moved_h, worst);
        ok((nm + at + ": every rank holds the folded y and the written h of the unfused chain, "
                   "byte for bit").c_str(),
           bad_y == 0 && bad_h == 0, d);
        if (gather && !sig) {
            /* The y the fold leaves is the same bytes on every rank -- the ascending-rank fold is
             * rank-order symmetric. A rank-local gather output that differs is fine; the folded
             * one must not. */
            int64_t skew = 0;
            for (int64_t i = 0; i < M * n; ++i)
                for (int r = 1; r < world; ++r)
                    if (t[(size_t)r].y_out[(size_t)i] != t[0].y_out[(size_t)i]) ++skew;
            ok((nm + at + ": the folded message is the same bytes on every rank").c_str(),
               skew == 0, std::to_string(skew) + " elements differ between ranks");
        }
        if (row->fini) for (ArHc& x : t) if (x.inst) row->fini(x.inst);
    }
}

/* ------------------------------------------------------------------ the device sweep */

/* embed_lookup_q against a host gather of the same bytes, through the host side's own conversions.
 *
 * Bit-exact: libavx's gather decodes a code as rad_f32_to_bf16(rad_fp8e4m3_to_f32(code) * scale),
 * and the device kernel must match for every code, or the n-gram embedding would depend on the
 * tier --ngram-placement chose. The table holds every code except the two NaN encodings.
 *
 * Five launches: the quad walk and the entry walk (widths 160, the model's, and 13); an E4M3 table
 * with a scale and a plain bf16 one (rows unchanged); the table in VRAM and in pinned host memory
 * read over the link. The ids include padding (-1) and one past the table, both writing a zero
 * row. */
static void embed_lookup_q_case(const Plugin& pl) {
    if (!sel("embed_lookup_q_bf16")) return;
    const char* what = "embed_lookup_q vs a host gather, to the bit";
    const RadKernelInfo* r = find(pl, "embed_lookup_q_bf16");
    if (!r) { ok(what, false, "row absent"); return; }

    struct Shape { int64_t ne; bool e4m3; bool host; };
    const Shape shapes[] = { { 160, true, false }, { 160, true, true }, { 13, true, false },
                             { 160, false, true }, { 13, false, false } };
    const int64_t NV = 1031, M = 37;
    const float scale = rad_bf16_to_f32(rad_f32_to_bf16(1.99317932e-4f));
    std::vector<void*> pinned;
    std::string bad;
    for (const Shape& sh : shapes) {
        const int64_t es = sh.e4m3 ? 1 : 2;
        const size_t tbytes = (size_t)(NV * sh.ne * es);
        std::vector<uint8_t> table(tbytes);
        uint32_t rs = 0x9e3779b9u;
        for (int64_t i = 0; i < NV * sh.ne; ++i) {
            rs = rs * 1664525u + 1013904223u;
            if (sh.e4m3) {
                uint8_t c = (uint8_t)((i * 167 + (rs >> 24)) & 0xFF);
                if ((c & 0x7F) == 0x7F) c ^= 0x01;           /* 0x7F and 0xFF are NaN */
                table[(size_t)i] = c;
            } else {
                const uint16_t b = rad_f32_to_bf16((float)((int32_t)(rs >> 9) - (1 << 22)) * 1e-6f);
                std::memcpy(&table[(size_t)(2 * i)], &b, 2);
            }
        }
        std::vector<int32_t> ids((size_t)M);
        for (int64_t m = 0; m < M; ++m) ids[(size_t)m] = (int32_t)((m * 131 + 7) % NV);
        ids[3] = -1;
        ids[7] = (int32_t)NV;
        ids[11] = (int32_t)(NV - 1);

        void* d_tab = nullptr;
        if (sh.host) {
            void* h = nullptr;
            if (hipHostMalloc(&h, tbytes, hipHostMallocMapped) != hipSuccess) {
                ok(what, false, "pinned table allocation failed");
                return;
            }
            pinned.push_back(h);
            std::memcpy(h, table.data(), tbytes);
            if (hipHostGetDevicePointer(&d_tab, h, 0) != hipSuccess || !d_tab) {
                ok(what, false, "no device address for the pinned table");
                return;
            }
        } else {
            d_tab = dalloc(tbytes);
            st_memcpy(d_tab, table.data(), tbytes, hipMemcpyHostToDevice);
        }
        const uint16_t sc_bits = rad_f32_to_bf16(scale);
        void* d_sc = dalloc(2);
        st_memcpy(d_sc, &sc_bits, 2, hipMemcpyHostToDevice);
        void* d_tok = dalloc((size_t)M * 4);
        st_memcpy(d_tok, ids.data(), (size_t)M * 4, hipMemcpyHostToDevice);
        void* d_x = dalloc((size_t)(M * sh.ne * 2));
        st_memset(d_x, 0xFF, (size_t)(M * sh.ne * 2));    /* zero rows must be written */

        std::vector<RadTensor> ts = { T(d_tok, RAD_I32, { M }),
                                      T(d_tab, sh.e4m3 ? RAD_F8E4M3 : RAD_BF16, { NV, sh.ne }),
                                      T(sh.e4m3 ? d_sc : nullptr, RAD_BF16, { 1 }),
                                      T(d_x, RAD_BF16, { M, sh.ne }) };
        std::vector<RadParam> ps = { PI("M", M), PI("n_embd", sh.ne), PI("n_vocab", NV),
                                     PS("dtype", "bf16") };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        st_stream_sync(nullptr);
        std::vector<uint16_t> got((size_t)(M * sh.ne), 0);
        st_memcpy(got.data(), d_x, got.size() * 2, hipMemcpyDeviceToHost);

        int64_t wrong = 0;
        for (int64_t m = 0; m < M; ++m) {
            const int32_t id = ids[(size_t)m];
            for (int64_t i = 0; i < sh.ne; ++i) {
                uint16_t want = 0;
                if (id >= 0 && id < NV) {
                    const size_t at = (size_t)(id * sh.ne + i);
                    if (sh.e4m3) {
                        want = rad_f32_to_bf16(rad_fp8e4m3_to_f32(table[at]) * scale);
                    } else {
                        std::memcpy(&want, &table[2 * at], 2);
                    }
                }
                if (got[(size_t)(m * sh.ne + i)] != want) ++wrong;
            }
        }
        if (rc != RAD_OK || wrong)
            bad += fmt("%s n_embd=%lld %s: rc=%d, %lld of %lld entries differ; ",
                       sh.e4m3 ? "e4m3" : "bf16", (long long)sh.ne,
                       sh.host ? "pinned host table" : "vram table", rc, (long long)wrong,
                       (long long)(M * sh.ne));
    }
    for (void* h : pinned) hip_ck(hipHostFree(h), "hipHostFree", __LINE__);
    ok(what, bad.empty(), bad);
}

/* ngram_ids' `ahead_ids`: the prompt following the step, hashed in the same launch. Step one runs
 * [5, 6] and [11, 42], with three more tokens of the last sequence, [7, 8, 9], in `tok` behind
 * them. Their ids must equal what step two computes for [7, 8, 9] against the window step one
 * left; that window must be the step's own; and the step's ids must match a launch without
 * `ahead_ids`. ref_test holds libref to the same property. */
static void ngram_ahead_case(const Plugin& pl) {
    if (!sel("ngram_ids_i32")) return;
    const RadKernelInfo* r = find(pl, "ngram_ids_i32");
    if (!r) { ok("ngram_ids_i32 ahead", false, "row absent"); return; }
    const int64_t kEos = 248044, H = 16, NG = 3;
    std::vector<int64_t> mult = { 23703573157769LL, 20109073645365LL, 8052911324071LL }, vsz, off;
    int64_t acc = 0;
    for (int i = 0; i < 16; ++i) { vsz.push_back(20000003 + 10 * i); off.push_back(acc); acc += vsz.back(); }
    auto dev = [&](const void* h, size_t b) { void* d = dalloc(b); st_memcpy(d, h, b, hipMemcpyHostToDevice); return d; };
    void* d_mult = dev(mult.data(), 3 * 8);
    void* d_vsz  = dev(vsz.data(), 16 * 8);
    void* d_off  = dev(off.data(), 16 * 8);
    auto launch = [&](const std::vector<int32_t>& tok, std::vector<int32_t>& state,
                      const std::vector<int32_t>& cu, const std::vector<int32_t>& cidx, int64_t M,
                      int64_t A, std::vector<int32_t>* ids, std::vector<int32_t>* ahead) {
        void* d_tok = dev(tok.data(), tok.size() * 4);
        void* d_st  = dev(state.data(), state.size() * 4);
        void* d_cu  = dev(cu.data(), cu.size() * 4);
        void* d_ci  = dev(cidx.data(), cidx.size() * 4);
        std::vector<int32_t> hini(cidx.size(), 1);
        void* d_hi  = dev(hini.data(), hini.size() * 4);
        void* d_ids = dalloc((size_t)(M * H * 4));
        void* d_ahd = A > 0 ? dalloc((size_t)(A * H * 4)) : nullptr;
        std::vector<RadTensor> ts = {
            T(d_tok, RAD_I32, { (int64_t)tok.size() }), T(d_st, RAD_I32, { (int64_t)state.size() / 2, 2 }),
            T(d_cu, RAD_I32, { (int64_t)cu.size() }), T(d_mult, RAD_I64, { 3 }),
            T(d_vsz, RAD_I64, { 16 }), T(d_off, RAD_I64, { 16 }),
            T(d_ci, RAD_I32, { (int64_t)cidx.size() }), T(d_hi, RAD_I32, { (int64_t)hini.size() }),
            T(nullptr, RAD_I32, { 1 }), T(d_ids, RAD_I32, { M, H }), T(d_ahd, RAD_I32, { A, H }) };
        absent(ts[8]);
        if (A == 0) absent(ts[10]);
        std::vector<RadParam> ps = { PI("M", M), PI("heads", H), PI("ngram", NG), PI("eos", kEos) };
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return false;
        ids->resize((size_t)(M * H));
        st_memcpy(ids->data(), d_ids, ids->size() * 4, hipMemcpyDeviceToHost);
        st_memcpy(state.data(), d_st, state.size() * 4, hipMemcpyDeviceToHost);
        if (A > 0) {
            ahead->resize((size_t)(A * H));
            st_memcpy(ahead->data(), d_ahd, ahead->size() * 4, hipMemcpyDeviceToHost);
        }
        return true;
    };
    std::vector<int32_t> st1 = { 900, 901, 902, 903 }, st0 = st1, st2;
    std::vector<int32_t> ids1, ahd1, ids0, ids2;
    const bool l1 = launch({ 5, 6, 11, 42, 7, 8, 9 }, st1, { 0, 2, 4 }, { 0, 1 }, 4, 3, &ids1, &ahd1);
    const bool l0 = launch({ 5, 6, 11, 42 }, st0, { 0, 2, 4 }, { 0, 1 }, 4, 0, &ids0, nullptr);
    st2 = st1;
    const bool l2 = l1 && launch({ 7, 8, 9 }, st2, { 0, 3 }, { 1 }, 3, 0, &ids2, nullptr);
    if (!l1 || !l0 || !l2) { ok("ngram_ids_i32 ahead: served", false); return; }
    ok("ngram_ids_i32 ahead: the step's own ids are the ones a launch without it writes", ids1 == ids0);
    ok("ngram_ids_i32 ahead: the window is the step's own", st1 == st0 && st1[2] == 11 && st1[3] == 42);
    ok("ngram_ids_i32 ahead: the ids ahead are the ones the next step computes", ahd1 == ids2);
}

/* ---- RUNTIME LoRA: a lora_shrink row then a lora_expand row, as the engine issues them, against an
 * f64 host sum and libref's bytes.
 *
 * Both halves in one case, because expand reads the `t` shrink wrote. A base row (slot -1, S,
 * -5) keeps the 0xFF fill of `t` and every byte of `y` (a -0 included), and the column slices'
 * neighbours are untouched. Adapter rows: shrink within (K + 2) 2^-24 sum|terms| of the f64 sum
 * (an f32 dot's worst case in any order); expand exactly the bytes of libref's f32 order (j
 * ascending, sum then scale then add, one RTNE) over the `t` the device wrote. The v1 pair runs
 * its four small shapes and two real ones; the tiled pairs run the real shapes -- M 1 / 8 / 2048,
 * K 2048 / 5120 / 6144 / 17408, N 2560 / 12288 / 17408, R 16 / 48 / 64 / 192 -- with slots in
 * runs the way a prefill chunk holds several requests and base rows, plus:
 *
 *   * BATCH INVARIANCE ON THE CARD: rows computed alone (M = 1) and inside batches of 8, 37 and
 *     2048 with other slots and base rows around them give the same t and y bits. This is the
 *     check the emulator cannot make -- that the WMMA unit's answer for a row does not depend on
 *     the tile position it lands in or on the rows beside it.
 *   * A LAUNCH OF BASE ROWS ONLY touches no byte of t or y. */
struct LoraProb {
    int M = 0, K = 0, N = 0, R = 0, S = 4;
    bool fp8 = false;
    uint32_t ydt = RAD_BF16;
    int lda = 0, lds = 0, ldy = 0, yoff = 3, group = 128;
    std::vector<uint8_t> a, y0;
    std::vector<float> as, sc;
    std::vector<int32_t> slot;
    std::vector<uint16_t> la, lb;
};

static size_t lora_ey(uint32_t ydt) { return ydt == RAD_F32 ? 4 : 2; }

/* The contract's x(m, k): a code times its group's scale (one f32 product), or the bf16 value. */
static float lora_x(const LoraProb& P, int m, int k) {
    const size_t ai = (size_t)m * P.lda + k;
    if (P.fp8) return rad_fp8e4m3_to_f32(P.a[ai]) * P.as[(size_t)m * P.lds + k / P.group];
    uint16_t b;
    std::memcpy(&b, &P.a[2 * ai], 2);
    return rad_bf16_to_f32(b);
}

static float lora_ld_y(uint32_t ydt, const uint8_t* p) {
    if (ydt == RAD_F32) { float v; std::memcpy(&v, p, 4); return v; }
    uint16_t b;
    std::memcpy(&b, p, 2);
    return ydt == RAD_F16 ? rad_f16_to_f32(b) : rad_bf16_to_f32(b);
}

/* The slots of a prefill chunk: runs of 1..400 rows, each a request on some adapter or none
 * (-1, S, -5 are three spellings of a base row). Short M cycles a fixed pattern instead. */
static std::vector<int32_t> lora_slots(int M, int S, uint32_t seed) {
    static const int32_t kCyc[] = { -1, 0, 1, 2, 3, 4, 2, -5, 1 };
    std::vector<int32_t> v((size_t)M);
    if (M <= 130) {
        for (int m = 0; m < M; ++m) {
            const int32_t s = kCyc[(m * 7 + 2) % 9];
            v[(size_t)m] = (s >= 0 && s < 4 && s >= S) ? s % S : (s == 4 ? S : s);
        }
        return v;
    }
    uint32_t st = seed;
    auto nx = [&]() { st = st * 1664525u + 1013904223u; return st >> 8; };
    for (int m = 0; m < M;) {
        const int len = 1 + (int)(nx() % 400);
        const int k = (int)(nx() % (uint32_t)(S + 3));
        const int32_t s = k < S ? k : (k == S ? -1 : (k == S + 1 ? S : -5));
        for (int i = 0; i < len && m < M; ++i, ++m) v[(size_t)m] = s;
    }
    return v;
}

static LoraProb lora_make(int M, int K, int N, int R, bool fp8, uint32_t ydt, uint32_t seed,
                          const std::vector<int32_t>* slots = nullptr) {
    LoraProb P;
    P.M = M; P.K = K; P.N = N; P.R = R; P.fp8 = fp8; P.ydt = ydt;
    const int groups = (K + P.group - 1) / P.group;
    P.lda = K + 8; P.lds = groups + 4; P.ldy = N + 6;
    uint32_t st = seed;
    auto rnd = [&]() {                           /* uniform in [-1, 1) */
        st = st * 1664525u + 1013904223u;
        return (float)((int32_t)(st >> 8) - (1 << 23)) / (float)(1 << 23);
    };
    const size_t ea = fp8 ? 1 : 2, ey = lora_ey(ydt);
    P.a.resize((size_t)M * P.lda * ea);
    for (size_t i = 0; i < (size_t)M * P.lda; ++i) {
        if (fp8) P.a[i] = rad_f32_to_fp8e4m3(8.0f * rnd());
        else { const uint16_t b = rad_f32_to_bf16(2.0f * rnd()); std::memcpy(&P.a[2 * i], &b, 2); }
    }
    P.as.resize((size_t)M * P.lds);
    for (float& v : P.as) v = 0.01f + 0.02f * (rnd() + 1.0f);
    P.slot = slots ? *slots : lora_slots(M, P.S, seed ^ 0x9e3779b9u);
    P.la.resize((size_t)P.S * R * K);
    P.lb.resize((size_t)P.S * N * R);
    for (uint16_t& v : P.la) v = rad_f32_to_bf16(0.1f * rnd());
    for (uint16_t& v : P.lb) v = rad_f32_to_bf16(0.1f * rnd());
    P.sc.resize((size_t)P.S * N);
    for (float& v : P.sc) v = 1.0f + 0.5f * rnd();
    P.y0.resize((size_t)M * P.ldy * ey);
    for (size_t i = 0; i < (size_t)M * P.ldy; ++i) {
        const float v = i % 3 == 0 ? -0.0f : rnd();
        if (ydt == RAD_F32) {
            std::memcpy(&P.y0[4 * i], &v, 4);
        } else {
            const uint16_t b = ydt == RAD_F16 ? rad_f32_to_f16(v) : rad_f32_to_bf16(v);
            std::memcpy(&P.y0[2 * i], &b, 2);
        }
    }
    return P;
}

/* Row m of `src` into row `dst` of `P`: x, its scales, its slot and its y (same K / N / dtypes). */
static void lora_copy_row(LoraProb& P, int dst, const LoraProb& src, int m) {
    const size_t ea = P.fp8 ? 1 : 2, ey = lora_ey(P.ydt);
    std::memcpy(&P.a[(size_t)dst * P.lda * ea], &src.a[(size_t)m * src.lda * ea], (size_t)P.lda * ea);
    std::memcpy(&P.as[(size_t)dst * P.lds], &src.as[(size_t)m * src.lds], (size_t)P.lds * 4);
    std::memcpy(&P.y0[(size_t)dst * P.ldy * ey], &src.y0[(size_t)m * src.ldy * ey],
                (size_t)P.ldy * ey);
    P.slot[(size_t)dst] = src.slot[(size_t)m];
}

/* Every row of [0, n) through f, on the host's cores: the f64 sums at M = 2048 are billions. */
template <class F>
static void lora_par(int64_t n, F&& f) {
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; ++i)
        th.emplace_back([&, i]() { for (int64_t r = i; r < n; r += nt) f(r); });
    for (auto& t : th) t.join();
}

/* Shrink then expand on the device over P; t ([M, R], 0xFF-filled first) and y come back. */
static bool lora_run(const RadKernelInfo* rs, const RadKernelInfo* re, const LoraProb& P,
                     std::vector<float>* t, std::vector<uint8_t>* y, int* rc1, int* rc2) {
    const int M = P.M, K = P.K, N = P.N, R = P.R, S = P.S;
    const int groups = (K + P.group - 1) / P.group;
    const size_t ey = lora_ey(P.ydt);
    void* d_a = dalloc(P.a.size());
    void* d_as = dalloc(P.as.size() * 4);
    void* d_sl = dalloc(P.slot.size() * 4);
    void* d_la = dalloc(P.la.size() * 2);
    void* d_lb = dalloc(P.lb.size() * 2);
    void* d_sc = dalloc(P.sc.size() * 4);
    void* d_t = dalloc((size_t)M * R * 4);
    void* d_y = dalloc(P.y0.size());
    if (!d_a || !d_as || !d_sl || !d_la || !d_lb || !d_sc || !d_t || !d_y) return false;
    st_memcpy(d_a, P.a.data(), P.a.size(), hipMemcpyHostToDevice);
    st_memcpy(d_as, P.as.data(), P.as.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_sl, P.slot.data(), P.slot.size() * 4, hipMemcpyHostToDevice);
    st_memcpy(d_la, P.la.data(), P.la.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_lb, P.lb.data(), P.lb.size() * 2, hipMemcpyHostToDevice);
    st_memcpy(d_sc, P.sc.data(), P.sc.size() * 4, hipMemcpyHostToDevice);
    st_memset(d_t, 0xFF, (size_t)M * R * 4);            /* a base row keeps this */
    st_memcpy(d_y, P.y0.data(), P.y0.size(), hipMemcpyHostToDevice);

    /* `a` columns [0, K) of [M, lda]; its scale [M, groups] of [M, lds]; `y` columns [yoff, +N). */
    RadTensor ta = T(d_a, P.fp8 ? RAD_F8E4M3 : RAD_BF16, { M, K });
    ta.stride[0] = P.lda;
    RadTensor tsc = T(P.fp8 ? d_as : nullptr, RAD_F32, { M, groups });
    tsc.stride[0] = P.lds;
    RadTensor ty = T((uint8_t*)d_y + (size_t)P.yoff * ey, P.ydt, { M, N });
    ty.stride[0] = P.ldy;
    std::vector<RadTensor> t1 = { ta, tsc, T(d_sl, RAD_I32, { M }), T(d_la, RAD_BF16, { S, R, K }),
                                  T(d_t, RAD_F32, { M, R }) };
    std::vector<RadParam> p1 = { PI("M", M), PI("K", K), PI("R", R), PI("S", S),
                                 PI("group", P.fp8 ? P.group : 0),
                                 PS("dtype", P.fp8 ? "fp8_e4m3" : "bf16") };
    std::vector<RadTensor> t2 = { T(d_t, RAD_F32, { M, R }), T(d_sl, RAD_I32, { M }),
                                  T(d_lb, RAD_BF16, { S, N, R }), T(d_sc, RAD_F32, { S, N }), ty };
    std::vector<RadParam> p2 = { PI("M", M), PI("N", N), PI("R", R), PI("S", S),
                                 PS("dtype", rad_dtype_name(P.ydt)) };
    RadArgs a1, a2;
    std::memset(&a1, 0, sizeof(a1)); std::memset(&a2, 0, sizeof(a2));
    a1.t = t1.data(); a1.n_t = (int)t1.size(); a1.p = p1.data(); a1.n_p = (int)p1.size();
    a2.t = t2.data(); a2.n_t = (int)t2.size(); a2.p = p2.data(); a2.n_p = (int)p2.size();
    a1.world_size = a2.world_size = 1;
    if (rs->scratch) {
        const int64_t need = rs->scratch(&a1);
        if (need > 0) { a1.scratch = dalloc((size_t)need); a1.scratch_bytes = need; }
    }
    *rc1 = rs->launch(&a1, nullptr);
    *rc2 = re->launch(&a2, nullptr);
    st_stream_sync(nullptr);
    t->resize((size_t)M * R);
    st_memcpy(t->data(), d_t, t->size() * 4, hipMemcpyDeviceToHost);
    y->resize(P.y0.size());
    st_memcpy(y->data(), d_y, y->size(), hipMemcpyDeviceToHost);
    dfree_all();
    return true;
}

static void lora_pair_case(const Plugin& pl, const char* sname, const char* ename,
                           const LoraProb& P) {
    const RadKernelInfo* rs = find(pl, sname);
    const RadKernelInfo* re = find(pl, ename);
    const std::string what = fmt("%s + %s M %d R %d K %d N %d %s / %s y", sname, ename, P.M, P.R,
                                 P.K, P.N, P.fp8 ? "e4m3+scale" : "bf16", rad_dtype_name(P.ydt));
    if (!rs || !re) { ok(what.c_str(), false, "row absent"); return; }
    std::vector<float> t;
    std::vector<uint8_t> y;
    int rc1 = 0, rc2 = 0;
    if (!lora_run(rs, re, P, &t, &y, &rc1, &rc2)) { ok(what.c_str(), false, "hipMalloc"); return; }
    const int R = P.R, K = P.K, N = P.N, S = P.S;
    const size_t ey = lora_ey(P.ydt);
    std::vector<int64_t> bb((size_t)P.M, 0), bt((size_t)P.M, 0), byy((size_t)P.M, 0);
    std::vector<double> worst((size_t)P.M, 0.0);
    lora_par(P.M, [&](int64_t mi) {
        const int m = (int)mi;
        const int32_t s = P.slot[(size_t)m];
        const bool base = s < 0 || s >= S;
        for (int j = 0; j < R; ++j) {
            const float v = t[(size_t)m * R + j];
            uint32_t bits; std::memcpy(&bits, &v, 4);
            if (base) { bb[(size_t)m] += bits != 0xFFFFFFFFu; continue; }
            double sum = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
                const double term = (double)lora_x(P, m, k) *
                                    rad_bf16_to_f32(P.la[((size_t)s * R + j) * K + k]);
                sum += term; mag += std::fabs(term);
            }
            const double tol = (K + 2) * std::ldexp(1.0, -24) * mag + 1e-30;
            worst[(size_t)m] = std::max(worst[(size_t)m], std::fabs(v - sum) / tol);
            bt[(size_t)m] += !(std::fabs(v - sum) <= tol);
        }
        for (int c = 0; c < P.ldy; ++c) {
            const size_t i = (size_t)m * P.ldy + c;
            const int n = c - P.yoff;
            if (base || n < 0 || n >= N) {
                bb[(size_t)m] += std::memcmp(&y[i * ey], &P.y0[i * ey], ey) != 0;
                continue;
            }
            float acc = 0.0f;                    /* libref's order over the device's t */
            for (int j = 0; j < R; ++j)
                acc += t[(size_t)m * R + j] * rad_bf16_to_f32(P.lb[((size_t)s * N + n) * R + j]);
            const float d = P.sc[(size_t)s * N + n] * acc;
            const float w = lora_ld_y(P.ydt, &P.y0[i * ey]) + d;
            uint8_t want[4];
            if (P.ydt == RAD_F32) {
                std::memcpy(want, &w, 4);
            } else {
                const uint16_t b = P.ydt == RAD_F16 ? rad_f32_to_f16(w) : rad_f32_to_bf16(w);
                std::memcpy(want, &b, 2);
            }
            byy[(size_t)m] += std::memcmp(&y[i * ey], want, ey) != 0;
        }
    });
    int64_t bad_base = 0, bad_t = 0, bad_y = 0;
    double w = 0;
    for (int m = 0; m < P.M; ++m) {
        bad_base += bb[(size_t)m]; bad_t += bt[(size_t)m]; bad_y += byy[(size_t)m];
        w = std::max(w, worst[(size_t)m]);
    }
    ok(what.c_str(), rc1 == RAD_OK && rc2 == RAD_OK && !bad_base && !bad_t && !bad_y,
       fmt("rc %d/%d, %lld base-row bytes touched, %lld t outside the bound (worst %.3f), "
           "%lld y not libref's bytes", rc1, rc2, (long long)bad_base, (long long)bad_t, w,
           (long long)bad_y));
}

/* BATCH INVARIANCE ON THE CARD. For each of `n_tgt` target rows of a batch (several slots, at
 * tile starts, ends and middles), the same row alone -- its x, scales, slot and y, the same pools
 * -- must come back with the same t and y bits. */
static void lora_invariance_case(const Plugin& pl, const char* sname, const char* ename, int K,
                                 int N, int R, bool fp8, uint32_t ydt) {
    const RadKernelInfo* rs = find(pl, sname);
    const RadKernelInfo* re = find(pl, ename);
    if (!rs || !re) { ok(fmt("%s batch invariance", sname).c_str(), false, "row absent"); return; }
    const size_t ey = lora_ey(ydt);
    for (int M : { 8, 37, 2048 }) {
        const LoraProb B = lora_make(M, K, N, R, fp8, ydt, 0x51ed270bu + (uint32_t)M);
        std::vector<float> tb;
        std::vector<uint8_t> yb;
        int r1 = 0, r2 = 0;
        if (!lora_run(rs, re, B, &tb, &yb, &r1, &r2) || r1 || r2) {
            ok(fmt("%s batch invariance M %d", sname, M).c_str(), false,
               fmt("rc %d/%d", r1, r2));
            continue;
        }
        int64_t diff_t = 0, diff_y = 0, rows = 0;
        const int cand[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33, M / 2, M - 2, M - 1 };
        for (int p : cand) {
            if (p < 0 || p >= M) continue;
            const int32_t s = B.slot[(size_t)p];
            if (s < 0 || s >= B.S) continue;
            LoraProb O = lora_make(1, K, N, R, fp8, ydt, 7u, &B.slot);
            O.M = 1;
            O.slot.assign(1, s);
            O.la = B.la; O.lb = B.lb; O.sc = B.sc;
            O.a.resize((size_t)O.lda * (fp8 ? 1 : 2));
            O.as.resize((size_t)O.lds);
            O.y0.resize((size_t)O.ldy * ey);
            lora_copy_row(O, 0, B, p);
            std::vector<float> to;
            std::vector<uint8_t> yo;
            if (!lora_run(rs, re, O, &to, &yo, &r1, &r2) || r1 || r2) { ++diff_t; continue; }
            diff_t += std::memcmp(&to[0], &tb[(size_t)p * R], (size_t)R * 4) != 0;
            diff_y += std::memcmp(&yo[0], &yb[(size_t)p * B.ldy * ey], (size_t)B.ldy * ey) != 0;
            ++rows;
        }
        ok(fmt("%s + %s batch invariance: K %d R %d N %d %s, %lld rows alone vs in M %d",
               sname, ename, K, R, N, fp8 ? "e4m3" : "bf16", (long long)rows, M).c_str(),
           rows > 0 && !diff_t && !diff_y,
           fmt("%lld t rows and %lld y rows differ from their M=1 bits", (long long)diff_t,
               (long long)diff_y));
    }
}

/* A chunk of base rows only: nothing is read past the slots, nothing is written. */
static void lora_base_case(const Plugin& pl, const char* sname, const char* ename) {
    const RadKernelInfo* rs = find(pl, sname);
    const RadKernelInfo* re = find(pl, ename);
    if (!rs || !re) { ok("lora base-only launch", false, "row absent"); return; }
    std::vector<int32_t> slots(2048);
    for (size_t i = 0; i < slots.size(); ++i) slots[i] = i % 3 == 0 ? -1 : (i % 3 == 1 ? 4 : -5);
    const LoraProb P = lora_make(2048, 2048, 2560, 48, false, RAD_BF16, 99u, &slots);
    std::vector<float> t;
    std::vector<uint8_t> y;
    int r1 = 0, r2 = 0;
    const bool ran = lora_run(rs, re, P, &t, &y, &r1, &r2);
    int64_t bad = 0;
    for (float v : t) { uint32_t b; std::memcpy(&b, &v, 4); bad += b != 0xFFFFFFFFu; }
    ok(fmt("%s + %s: M 2048 base rows only, t and y untouched", sname, ename).c_str(),
       ran && !r1 && !r2 && !bad && y == P.y0,
       fmt("rc %d/%d, %lld t words written, y %s", r1, r2, (long long)bad,
           y == P.y0 ? "untouched" : "CHANGED"));
}

static void lora_case(const Plugin& pl) {
    const bool sc_s = sel("lora_shrink_scalar"), sc_e = sel("lora_expand_scalar");
    const bool tw = sel("lora_shrink_wmma"), tq = sel("lora_shrink_wmma_q");
    const bool te = sel("lora_expand_tiled");
    if (sc_s || sc_e) {
        const char *s = "lora_shrink_scalar", *e = "lora_expand_scalar";
        lora_pair_case(pl, s, e, lora_make(1, 1000, 640, 16, false, RAD_BF16, 0x2545f491u));
        lora_pair_case(pl, s, e, lora_make(17, 1000, 300, 48, true, RAD_BF16, 0x2545f492u));
        lora_pair_case(pl, s, e, lora_make(130, 256, 1000, 32, true, RAD_F32, 0x2545f493u));
        lora_pair_case(pl, s, e, lora_make(130, 64, 64, 16, false, RAD_F32, 0x2545f494u));
        lora_pair_case(pl, s, e, lora_make(8, 2048, 2560, 48, false, RAD_BF16, 0x2545f495u));
        lora_pair_case(pl, s, e, lora_make(300, 6144, 2048, 16, true, RAD_F16, 0x2545f496u));
    }
    if (tw || te) {
        const char *s = "lora_shrink_wmma", *e = "lora_expand_tiled";
        lora_pair_case(pl, s, e, lora_make(1, 2048, 2560, 48, false, RAD_BF16, 0x1001u));
        lora_pair_case(pl, s, e, lora_make(8, 2048, 12288, 32, false, RAD_BF16, 0x1002u));
        lora_pair_case(pl, s, e, lora_make(8, 6144, 2048, 16, false, RAD_BF16, 0x1003u));
        lora_pair_case(pl, s, e, lora_make(2048, 2048, 2560, 48, false, RAD_BF16, 0x1004u));
        lora_pair_case(pl, s, e, lora_make(2048, 6144, 2048, 16, false, RAD_F32, 0x1005u));
        lora_pair_case(pl, s, e, lora_make(2048, 2048, 17408, 64, false, RAD_BF16, 0x1006u));
        lora_pair_case(pl, s, e, lora_make(37, 17408, 2560, 16, false, RAD_BF16, 0x1007u));
        lora_pair_case(pl, s, e, lora_make(3, 2048, 384, 192, false, RAD_F16, 0x1008u));
        lora_pair_case(pl, s, e, lora_make(130, 1000, 64, 16, false, RAD_BF16, 0x1009u));
        lora_invariance_case(pl, s, e, 2048, 2560, 48, false, RAD_BF16);
        lora_invariance_case(pl, s, e, 6144, 2048, 16, false, RAD_F32);
        lora_base_case(pl, s, e);
    }
    if (tq || te) {
        const char *s = "lora_shrink_wmma_q", *e = "lora_expand_tiled";
        lora_pair_case(pl, s, e, lora_make(1, 5120, 2560, 48, true, RAD_BF16, 0x2001u));
        lora_pair_case(pl, s, e, lora_make(8, 5120, 17408, 64, true, RAD_BF16, 0x2002u));
        lora_pair_case(pl, s, e, lora_make(2048, 5120, 17408, 64, true, RAD_BF16, 0x2003u));
        lora_pair_case(pl, s, e, lora_make(2048, 6144, 2048, 16, true, RAD_F16, 0x2004u));
        lora_pair_case(pl, s, e, lora_make(17, 1000, 300, 48, true, RAD_F32, 0x2005u));
        lora_invariance_case(pl, s, e, 5120, 1024, 64, true, RAD_BF16);
    }
}

/* ---- --perflora M K N R S PATTERN [iters]: every lora_shrink row, then every lora_expand row,
 * at one shape, warm (or cold with --cold). The pattern lays the slots out:
 *
 *   base   every row a base row -- what a base request pays for the adapter ops being there
 *   one    every row adapter 0 -- one request, or a prefill chunk of one
 *   half   even rows adapter 0, odd rows base -- base and adapter requests interleaved
 *   mix    row m on adapter m % S -- every adapter in every tile, the most passes a tile has
 *   runs   contiguous runs of ceil(M / S) rows a slot -- a chunk holding S requests
 *
 * `a` is bf16, or E4M3 with a scale every 128 when R4D_PERFLORA_A=fp8_e4m3; `y` bf16, or the
 * dtype R4D_PERFLORA_Y names (f16, f32). Rows that do not serve the dtype say so. Each shrink row's
 * t is compared with the first row's (max |diff| over max |t|), and each expand row's y bytes with
 * the first's over the same t -- equal bytes is the claim the tiled expand makes. The GB/s column
 * is the bytes the op cannot avoid (each active adapter's pool once, x or y once) over the time. */
static void lora_perf(const Plugin& pl, int64_t M, int64_t K, int64_t N, int64_t R, int64_t S,
                      const char* pat, int iters) {
    if (M < 1 || K < 1 || N < 1 || R < 1 || S < 1 || iters < 1) {
        std::printf("--perflora: M K N R S must be >= 1\n");
        return;
    }
    const char* ae = std::getenv("R4D_PERFLORA_A");
    const char* ye = std::getenv("R4D_PERFLORA_Y");
    const bool fp8 = ae && !std::strcmp(ae, "fp8_e4m3");
    const uint32_t ydt = ye && !std::strcmp(ye, "f32") ? RAD_F32
                       : (ye && !std::strcmp(ye, "f16") ? RAD_F16 : RAD_BF16);
    std::vector<int32_t> slot((size_t)M);
    const int64_t run = (M + S - 1) / S;
    for (int64_t m = 0; m < M; ++m) {
        int32_t s;
        if (!std::strcmp(pat, "base"))      s = -1;
        else if (!std::strcmp(pat, "one"))  s = 0;
        else if (!std::strcmp(pat, "half")) s = (m & 1) ? -1 : 0;
        else if (!std::strcmp(pat, "mix"))  s = (int32_t)(m % S);
        else if (!std::strcmp(pat, "runs")) s = (int32_t)(m / run);
        else { std::printf("--perflora: PATTERN is base, one, half, mix or runs\n"); return; }
        slot[(size_t)m] = s;
    }
    std::vector<bool> used((size_t)S, false);
    int64_t act_rows = 0;
    for (int32_t s : slot) if (s >= 0 && s < S) { used[(size_t)s] = true; ++act_rows; }
    int64_t n_act = 0;
    for (bool u : used) n_act += u;

    const int64_t group = 128, groups = (K + group - 1) / group;
    const size_t ea = fp8 ? 1 : 2, ey = lora_ey(ydt);
    void* d_a = dalloc((size_t)(M * K) * ea);
    void* d_as = dalloc((size_t)(M * groups) * 4);
    void* d_sl = dalloc((size_t)M * 4);
    void* d_la = dalloc((size_t)(S * R * K) * 2);
    void* d_lb = dalloc((size_t)(S * N * R) * 2);
    void* d_sc = dalloc((size_t)(S * N) * 4);
    void* d_t = dalloc((size_t)(M * R) * 4);
    void* d_y = dalloc((size_t)(M * N) * ey);
    void* d_y0 = dalloc((size_t)(M * N) * ey);
    if (!d_a || !d_as || !d_sl || !d_la || !d_lb || !d_sc || !d_t || !d_y || !d_y0) {
        std::printf("--perflora: hipMalloc failed\n");
        return;
    }
    {
        uint32_t st = 0x2545f491u;
        auto rnd = [&]() {
            st = st * 1664525u + 1013904223u;
            return (float)((int32_t)(st >> 8) - (1 << 23)) / (float)(1 << 23);
        };
        std::vector<uint8_t> a((size_t)(M * K) * ea);
        for (size_t i = 0; i < (size_t)(M * K); ++i) {
            if (fp8) a[i] = rad_f32_to_fp8e4m3(8.0f * rnd());
            else { const uint16_t b = rad_f32_to_bf16(2.0f * rnd()); std::memcpy(&a[2 * i], &b, 2); }
        }
        st_memcpy(d_a, a.data(), a.size(), hipMemcpyHostToDevice);
        std::vector<float> as((size_t)(M * groups));
        for (float& v : as) v = 0.01f + 0.02f * (rnd() + 1.0f);
        st_memcpy(d_as, as.data(), as.size() * 4, hipMemcpyHostToDevice);
        st_memcpy(d_sl, slot.data(), slot.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint16_t> la((size_t)(S * R * K)), lb((size_t)(S * N * R));
        for (uint16_t& v : la) v = rad_f32_to_bf16(0.1f * rnd());
        for (uint16_t& v : lb) v = rad_f32_to_bf16(0.1f * rnd());
        st_memcpy(d_la, la.data(), la.size() * 2, hipMemcpyHostToDevice);
        st_memcpy(d_lb, lb.data(), lb.size() * 2, hipMemcpyHostToDevice);
        std::vector<float> sc((size_t)(S * N));
        for (float& v : sc) v = 1.0f + 0.5f * rnd();
        st_memcpy(d_sc, sc.data(), sc.size() * 4, hipMemcpyHostToDevice);
        std::vector<uint8_t> y((size_t)(M * N) * ey);
        for (size_t i = 0; i < (size_t)(M * N); ++i) {
            const float v = rnd();
            if (ydt == RAD_F32) std::memcpy(&y[4 * i], &v, 4);
            else {
                const uint16_t b = ydt == RAD_F16 ? rad_f32_to_f16(v) : rad_f32_to_bf16(v);
                std::memcpy(&y[2 * i], &b, 2);
            }
        }
        st_memcpy(d_y0, y.data(), y.size(), hipMemcpyHostToDevice);
    }

    std::printf("\nruntime LoRA  M=%lld K=%lld N=%lld R=%lld S=%lld  pattern %s: %lld adapter rows, "
                "%lld adapter(s) active  a %s, y %s, %d iters%s\n", (long long)M, (long long)K,
                (long long)N, (long long)R, (long long)S, pat, (long long)act_rows,
                (long long)n_act, fp8 ? "fp8_e4m3/128" : "bf16", rad_dtype_name(ydt), iters,
                g_cold > 0 ? ", cold" : ", warm");
    const double sh_bytes = (double)n_act * R * K * 2 + (double)act_rows * K * ea;
    const double ex_bytes = (double)n_act * N * R * 2 + (double)act_rows * (N * 2.0 * ey + R * 4.0);

    std::vector<RadTensor> t1 = { T(d_a, fp8 ? RAD_F8E4M3 : RAD_BF16, { M, K }),
                                  T(fp8 ? d_as : nullptr, RAD_F32, { M, groups }),
                                  T(d_sl, RAD_I32, { M }), T(d_la, RAD_BF16, { S, R, K }),
                                  T(d_t, RAD_F32, { M, R }) };
    std::vector<RadParam> p1 = { PI("M", M), PI("K", K), PI("R", R), PI("S", S),
                                 PI("group", fp8 ? group : 0),
                                 PS("dtype", fp8 ? "fp8_e4m3" : "bf16") };
    std::vector<RadTensor> t2 = { T(d_t, RAD_F32, { M, R }), T(d_sl, RAD_I32, { M }),
                                  T(d_lb, RAD_BF16, { S, N, R }), T(d_sc, RAD_F32, { S, N }),
                                  T(d_y, ydt, { M, N }) };
    std::vector<RadParam> p2 = { PI("M", M), PI("N", N), PI("R", R), PI("S", S),
                                 PS("dtype", rad_dtype_name(ydt)) };

    /* A row whose band names other dtypes is skipped by name: lora_shrink_wmma_q shares the
     * tiled entry point, and timing it on a bf16 `a` would print the bf16 row twice. */
    auto takes = [](const RadKernelInfo* r, const char* dt) {
        for (int c = 0; c < r->n_constraints; ++c) {
            const RadConstraint& q = r->constraints[c];
            if (q.op != RAD_C_IN || std::strcmp(q.key, "dtype") || !q.sval) continue;
            const std::string set = std::string(" ") + q.sval + " ";
            return set.find(std::string(" ") + dt + " ") != std::string::npos;
        }
        return true;
    };

    /* lora_shrink: every row, each one's t against the first's. */
    std::vector<float> t_first, t_cur((size_t)(M * R));
    std::printf("  lora_shrink rows (%.2f MB the op must read):\n", sh_bytes / 1e6);
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!r->op || std::strcmp(r->op, "lora_shrink") || !r->launch) continue;
        if (!takes(r, fp8 ? "fp8_e4m3" : "bf16")) {
            std::printf("    %-22s (its band does not take this `a`)\n", r->name);
            continue;
        }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = t1.data(); a.n_t = (int)t1.size(); a.p = p1.data(); a.n_p = (int)p1.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        st_memset(d_t, 0, (size_t)(M * R) * 4);
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("    %-22s rc=%d (not served at this dtype / shape)\n", r->name, rc);
            continue;
        }
        st_memcpy(t_cur.data(), d_t, t_cur.size() * 4, hipMemcpyDeviceToHost);
        double dmax = 0, tmax = 0;
        if (t_first.empty()) t_first = t_cur;
        for (size_t i = 0; i < t_cur.size(); ++i) {
            if (slot[i / (size_t)R] < 0 || slot[i / (size_t)R] >= S) continue;
            dmax = std::max(dmax, (double)std::fabs(t_cur[i] - t_first[i]));
            tmax = std::max(tmax, (double)std::fabs(t_first[i]));
        }
        const double us = st_time_us([&](int) { r->launch(&a, nullptr); }, iters, nullptr);
        std::printf("    %-22s %9.2f us  %8.1f GB/s   t vs first row: max|diff|/max|t| %.2e\n",
                    r->name, us, sh_bytes / (us * 1e3), tmax > 0 ? dmax / tmax : 0.0);
    }
    if (t_first.empty()) { std::printf("  no lora_shrink row served\n"); return; }
    st_memcpy(d_t, t_first.data(), t_first.size() * 4, hipMemcpyHostToDevice);

    /* lora_expand: every row over the first shrink row's t, each one's y bytes against the first. */
    std::vector<uint8_t> y_first, y_cur((size_t)(M * N) * ey);
    std::printf("  lora_expand rows (%.2f MB the op must move):\n", ex_bytes / 1e6);
    for (int k = 0; k < pl.kernel_count(); ++k) {
        const RadKernelInfo* r = pl.kernel_at(k);
        if (!r->op || std::strcmp(r->op, "lora_expand") || !r->launch) continue;
        if (!takes(r, rad_dtype_name(ydt))) {
            std::printf("    %-22s (its band does not take this `y`)\n", r->name);
            continue;
        }
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = t2.data(); a.n_t = (int)t2.size(); a.p = p2.data(); a.n_p = (int)p2.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        st_memcpy(d_y, d_y0, (size_t)(M * N) * ey, hipMemcpyDeviceToDevice);
        const int rc = r->launch(&a, nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) {
            std::printf("    %-22s rc=%d (not served at this dtype / shape)\n", r->name, rc);
            continue;
        }
        st_memcpy(y_cur.data(), d_y, y_cur.size(), hipMemcpyDeviceToHost);
        if (y_first.empty()) y_first = y_cur;
        int64_t diff = 0;
        for (size_t i = 0; i < y_cur.size(); i += ey)
            diff += std::memcmp(&y_cur[i], &y_first[i], ey) != 0;
        /* Timed on a y that keeps accumulating: the cost of an add does not depend on y. */
        const double us = st_time_us([&](int) { r->launch(&a, nullptr); }, iters, nullptr);
        std::printf("    %-22s %9.2f us  %8.1f GB/s   y vs first row: %lld element(s) differ\n",
                    r->name, us, ex_bytes / (us * 1e3), (long long)diff);
    }
}

static void device_phase(const Plugin& pl) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
        skip("device phase", "no HIP device");
        return;
    }
    /* An integrated GPU can appear beside the discrete cards, reporting system RAM as "VRAM".
     * Devices are filtered on the arch string, since a rank on the integrated part would fail in a
     * way that looks like a kernel bug. */
    std::vector<int> devs;
    for (int d = 0; d < n; ++d) {
        hipDeviceProp_t p;
        if (hipGetDeviceProperties(&p, d) != hipSuccess) continue;
        if (!g_case) std::printf("  device %d: %s\n", d, p.gcnArchName);
        if (std::strncmp(p.gcnArchName, "gfx1201", 7) == 0) devs.push_back(d);
    }
    if (devs.empty()) { skip("device phase", "no gfx1201 device"); return; }
    st_set_device(devs[0]);

    if (!g_case) std::printf("\ndevice smoke tests (device %d)\n", devs[0]);

    ngram_ahead_case(pl);
    embed_lookup_q_case(pl);
    lora_case(pl);

    /* --- the paged cache's write side --- */
    kv_store_case(pl, "kv_store_bf16", "bf16");
    kv_store_case(pl, "kv_store_fp8",  "fp8_e4m3");

    /* --- attention --- */
    attn_case(pl, "attn_prefill_h256_gqa6_fp8kv",  256, 6, 32, "fp8_e4m3", 1);
    attn_case(pl, "attn_prefill_h256_gqa6_bf16kv", 256, 6, 32, "bf16",     1);
    attn_case(pl, "attn_decode_h256_gqa6_fp8kv",   256, 6,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h256_gqa6_bf16kv",  256, 6,  1, "bf16",     0);
    attn_case(pl, "attn_decode_h128_gqa4_fp8kv",   128, 4,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h128_gqa4_bf16kv",  128, 4,  1, "bf16",     0);
    attn_case(pl, "attn_prefill_h64_gqa4_bf16kv",   64, 4, 32, "bf16",     1);
    attn_case(pl, "attn_prefill_h64_gqa4_bf16kv",   64, 4,  1, "bf16",     1);
    /* Gemma 4 dense sliding window, decode, with the model's window of 1024: exercises the
     * windowed split law and the WIN=1 template. max_ctx is below the window, so the span clamps to
     * the context. */
    attn_case(pl, "attn_decode_h256_gqa2_fp8kv",   256, 2,  1, "fp8_e4m3", 0, 1024);
    attn_case(pl, "attn_decode_h256_gqa2_bf16kv", 256, 2,  1, "bf16",     0, 1024);
    /* Gemma 4 dense sliding window, prefill (q_len above 32), on the windowed query-tiled kernel
     * with WIN=1 (window 1024). max_ctx (64) is below the window, so the span clamps to the context
     * and the launch is the full-attention form. */
    attn_case(pl, "attn_prefill_h256_gqa2_fp8kv",  256, 2, 32, "fp8_e4m3", 1, 1024);
    attn_case(pl, "attn_prefill_h256_gqa2_bf16kv", 256, 2, 32, "bf16",     1, 1024);
    /* Gemma 4 dense global: head_dim 512, full attention, on the windowed kernel (one row for every
     * q_len): a prefill-shaped launch (q_len 32) and a decode one (1); numeric checks follow. */
    attn_case(pl, "attn_window_h512_gqa8_fp8kv",  512, 8, 32, "fp8_e4m3", 1);
    attn_case(pl, "attn_window_h512_gqa8_bf16kv", 512, 8, 32, "bf16",     1);
    attn_case(pl, "attn_window_h512_gqa8_fp8kv",  512, 8,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_window_h512_gqa8_bf16kv", 512, 8,  1, "bf16",     0);
    /* Numeric checks of the same rows: random data against the inline oracle, with a negative
     * control. */
    attn_numeric_case(pl, "attn_window_h512_gqa8_fp8kv",  512, 8, 32, "fp8_e4m3", 0);
    attn_numeric_case(pl, "attn_window_h512_gqa8_bf16kv", 512, 8, 32, "bf16",     0);
    attn_numeric_case(pl, "attn_window_h512_gqa8_fp8kv",  512, 8,  1, "fp8_e4m3", 0);
    attn_numeric_case(pl, "attn_window_h512_gqa8_bf16kv", 512, 8,  1, "bf16",     0);
    /* The h256/gqa2 prefill numeric check with ctx 2048 > window 1024, so the window applies: for
     * query row 0, pos = 2048-64 = 1984 and the visible keys are [961..1984], so wlo0 = 961 and
     * tstart = 961/48 = 20. This exercises the window mask and the below-window tile skip (tiles
     * 0..19 are never walked). */
    attn_numeric_case(pl, "attn_prefill_h256_gqa2_fp8kv",    256, 2, 64, "fp8_e4m3", 1024, 2048);
    attn_numeric_case(pl, "attn_prefill_h256_gqa2_bf16kv",   256, 2, 64, "bf16",     1024, 2048);
    attn_numeric_case(pl, "attn_decode_h256_gqa2_fp8kv",    256, 2,  1, "fp8_e4m3", 1024);
    attn_numeric_case(pl, "attn_decode_h256_gqa2_bf16kv",   256, 2,  1, "bf16",     1024);
    /* Two queries per KV head at 128: Qwen3-1.7B, Qwen3-ASR's decoder. */
    attn_case(pl, "attn_prefill_h128_gqa2_fp8kv",  128, 2, 32, "fp8_e4m3", 1);
    attn_case(pl, "attn_prefill_h128_gqa2_bf16kv", 128, 2, 32, "bf16",     1);
    attn_case(pl, "attn_decode_h128_gqa2_fp8kv",   128, 2,  1, "fp8_e4m3", 0);
    attn_case(pl, "attn_decode_h128_gqa2_bf16kv",  128, 2,  1, "bf16",     0);
    attn_numeric_case(pl, "attn_prefill_h128_gqa2_fp8kv",  128, 2, 64, "fp8_e4m3", 0, 256);
    attn_numeric_case(pl, "attn_prefill_h128_gqa2_bf16kv", 128, 2, 64, "bf16",     0, 256);
    attn_numeric_case(pl, "attn_decode_h128_gqa2_fp8kv",   128, 2,  1, "fp8_e4m3", 0, 256);
    attn_numeric_case(pl, "attn_decode_h128_gqa2_bf16kv",  128, 2,  1, "bf16",     0, 256);
    attn_numeric_case(pl, "attn_decode_h128_gqa2_bf16kv",  128, 2, 24, "bf16",     0, 256);
    attn_gq_fold_case(pl, 3, false);
    attn_gq_fold_case(pl, 24, false);
    attn_gq_fold_case(pl, 3, true);
    attn_gq_fold_case(pl, 24, true);
    for (const bool bf16 : { false, true }) {
        attn_gq_fold_case(pl, 3, bf16, true);
        attn_gq_fold_case(pl, 24, bf16, true);
    }
    if (sel("attn_vit_h72_bf16")) {
        const RadKernelInfo* r = find(pl, "attn_vit_h72_bf16");
        if (!r) ok("attn_vit_h72_bf16", false, "row absent");
        else {
            const int64_t total = 64, heads = 4, hd = 72;
            std::vector<RadTensor> ts = {
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
                T(nullptr, RAD_I32,  { 3 }),
                T(nullptr, RAD_BF16, { total, heads, hd }),
            };
            std::vector<RadParam> ps = {
                PI("M", total), PI("head_dim", hd), PI("gqa", 1), PI("causal", 0),
                PS("dtype", "bf16"), PI("max_seqlen", 32),
            };
            run_case(r, ts, ps);
        }
    }
    /* The encoder attention rows: a fused [q|k|v] projection read through column slices, at each
     * compiled (head_dim, gqa), with and without a window. Launch only; numerics are rad-kbench's
     * (libref is the oracle). */
    {
        struct G { const char* name; int64_t hd, nh, nkv; };
        for (const G& g : { G{ "attn_dense_h64_bf16", 64, 4, 4 },
                            G{ "attn_dense_h256_gqa2_bf16", 256, 4, 2 },
                            G{ "attn_dense_h512_gqa4_bf16", 512, 4, 1 } }) {
            if (!sel(g.name)) continue;
            const RadKernelInfo* r = find(pl, g.name);
            if (!r) { ok(g.name, false, "row absent"); continue; }
            for (const int64_t window : { (int64_t)0, (int64_t)9 }) {
                const int64_t total = 48, wq = g.nh * g.hd, wkv = g.nkv * g.hd;
                const int64_t width = wq + 2 * wkv;
                void* qkv = dalloc((size_t)(total * width * 2));
                auto col = [&](int64_t first, int64_t w) {
                    RadTensor t = T(qkv, RAD_BF16, { total, w });
                    t.data = (void*)((uint16_t*)qkv + first);
                    t.stride[0] = width;
                    return t;
                };
                std::vector<RadTensor> ts = {
                    col(0, wq), col(wq, wkv), col(wq + wkv, wkv),
                    T(nullptr, RAD_I32, { 3 }), T(nullptr, RAD_BF16, { total, wq }),
                };
                std::vector<RadParam> ps = {
                    PI("M", total), PI("head_dim", g.hd), PI("gqa", g.nh / g.nkv),
                    PI("causal", 0), PS("dtype", "bf16"), PF("scale", 1.0),
                    PI("max_seqlen", 32), PI("window", window),
                };
                run_case(r, ts, ps);
            }
        }
    }
    if (sel("act_mul_bf16")) {
        const RadKernelInfo* r = find(pl, "act_mul_bf16");
        if (!r) ok("act_mul_bf16", false, "row absent");
        else
            for (const char* act : { "silu", "gelu", "gelu_tanh" }) {
                std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 5, 645 }),
                                              T(nullptr, RAD_BF16, { 5, 645 }),
                                              T(nullptr, RAD_BF16, { 5, 645 }) };
                std::vector<RadParam> ps = { PI("M", 5), PI("n", 645), PS("act", act),
                                             PS("dtype", "bf16") };
                run_case(r, ts, ps);
            }
    }
    if (sel("seg_mean_bf16")) {
        const RadKernelInfo* r = find(pl, "seg_mean_bf16");
        if (!r) ok("seg_mean_bf16", false, "row absent");
        else {
            std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 23, 645 }),
                                          T(nullptr, RAD_I32, { 4 }),
                                          T(nullptr, RAD_BF16, { 3, 645 }) };
            std::vector<RadParam> ps = { PI("M", 23), PI("n", 645), PS("dtype", "bf16") };
            run_case(r, ts, ps);
        }
    }
    if (sel("qk_norm_rope_bf16")) {
        const RadKernelInfo* r = find(pl, "qk_norm_rope_bf16");
        if (!r) ok("qk_norm_rope_bf16", false, "row absent");
        else
            for (const int64_t hd : { (int64_t)256, (int64_t)512 }) {
                std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 7, 8 * hd }),
                                              T(nullptr, RAD_I32, { 7 }),
                                              T(nullptr, RAD_BF16, { hd }),
                                              T(nullptr, RAD_BF16, { hd }) };
                std::vector<RadParam> ps = { PI("M", 7), PI("head_dim", hd), PI("n_head", 4),
                                             PI("n_head_kv", 2), PF("theta", 10000.0),
                                             PF("eps", 1e-6), PS("mode", "neox"),
                                             PI("v_norm", 1) };
                run_case(r, ts, ps);
            }
    }
    if (sel("mul_weight_bf16")) {
        const RadKernelInfo* r = find(pl, "mul_weight_bf16");
        if (!r) ok("mul_weight_bf16", false, "row absent");
        else
            for (const int64_t wn : { (int64_t)1, (int64_t)512 }) {
                std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 5, 512 }),
                                              T(nullptr, RAD_BF16, { wn }),
                                              T(nullptr, RAD_BF16, { 5, 512 }) };
                std::vector<RadParam> ps = { PI("M", 5), PI("n", 512), PS("dtype", "bf16"),
                                             PI("scalar", wn == 1) };
                run_case(r, ts, ps);
            }
    }
    if (sel("qk_norm_rope_bf16")) {
        const RadKernelInfo* r = find(pl, "qk_norm_rope_bf16");
        if (r) {
            /* blocks, the vision form: 12 heads of 64, x and y halves, components (row, col). */
            std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 9, 36 * 64 }),
                                          T(nullptr, RAD_I32, { 2, 9 }),
                                          T(nullptr, RAD_BF16, { 64 }),
                                          T(nullptr, RAD_BF16, { 64 }) };
            std::vector<RadParam> ps = { PI("M", 9), PI("head_dim", 64), PI("n_head", 12),
                                         PI("n_head_kv", 12), PF("theta", 100.0), PF("eps", 1e-6),
                                         PS("mode", "blocks"), PS("sections", "16 16"),
                                         PS("comps", "1 0"), PI("v_norm", 1) };
            run_case(r, ts, ps);
        }
    }
    if (sel("pool_rows_bf16")) {
        const RadKernelInfo* r = find(pl, "pool_rows_bf16");
        if (!r) ok("pool_rows_bf16", false, "row absent");
        else {
            std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 27, 768 }),
                                          T(nullptr, RAD_BF16, { 3, 768 }) };
            std::vector<RadParam> ps = { PI("M", 27), PI("n", 768), PI("k", 9),
                                         PF("mult", 27.712812921102035), PS("dtype", "bf16") };
            run_case(r, ts, ps);
        }
    }
    /* --- the audio tower's ops, numerically: each kernel against the formula libref implements,
     * computed on the host from the same bf16 inputs (rad-kbench's replay fixture predates these
     * ops). Relative L2 against a bf16 rounding of the answer. */
    {
        uint64_t seed = 0x5eedull;
        auto rnd = [&]() {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            return (float)((int64_t)(seed >> 40) % 2001 - 1000) / 1000.0f;
        };
        auto up = [&](std::vector<float>& h) {
            std::vector<uint16_t> b(h.size());
            for (size_t i = 0; i < h.size(); ++i) { b[i] = h_to_bf16(h[i]); h[i] = h_bf16(b[i]); }
            void* d = dalloc(b.size() * 2);
            if (d) st_memcpy(d, b.data(), b.size() * 2, hipMemcpyHostToDevice);
            return d;
        };
        auto down = [&](void* d, size_t n) {
            std::vector<uint16_t> b(n);
            st_memcpy(b.data(), d, n * 2, hipMemcpyDeviceToHost);
            std::vector<float> f(n);
            for (size_t i = 0; i < n; ++i) f[i] = h_bf16(b[i]);
            return f;
        };
        auto rel = [](const std::vector<float>& got, const std::vector<float>& want) {
            double e = 0, r = 0;
            for (size_t i = 0; i < want.size(); ++i) {
                const double w = h_bf16(h_to_bf16(want[i]));
                e += (got[i] - w) * (got[i] - w);
                r += w * w;
            }
            return r > 0 ? std::sqrt(e / r) : std::sqrt(e);
        };
        auto check = [&](const char* name, const std::vector<float>& got,
                         const std::vector<float>& want, double tol) {
            const double r = rel(got, want);
            char d[96];
            std::snprintf(d, sizeof d, "rel_l2 %.3g", r);
            ok(name, r <= tol, d);
        };
        auto randv = [&](size_t n, float scale) {
            std::vector<float> v(n);
            for (auto& x : v) x = rnd() * scale;
            return v;
        };

        if (sel("relu_bf16") || sel("silu_bf16") || sel("clamp_bf16")) {
            const int64_t M = 5, n = 300;
            std::vector<float> x = randv((size_t)(M * n), 2.0f);
            void* dx = up(x);
            void* dy = dalloc((size_t)(M * n) * 2);
            if (const RadKernelInfo* r = find(pl, "relu_bf16")) {
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, n }), T(dy, RAD_BF16, { M, n }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n", n), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want(x.size());
                for (size_t i = 0; i < x.size(); ++i) want[i] = x[i] > 0 ? x[i] : 0;
                check("relu_bf16 numerics", down(dy, x.size()), want, 0.0);
            } else ok("relu_bf16", false, "row absent");
            if (const RadKernelInfo* r = find(pl, "silu_bf16")) {
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, n }), T(dy, RAD_BF16, { M, n }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n", n), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    const float v = h_bf16(h_to_bf16(x[i]));
                    want[i] = v * (1.0f / (1.0f + std::exp(-v)));
                }
                /* expf and std::exp may differ by an f32 ulp, moving a bf16 rounding. */
                check("silu_bf16 numerics", down(dy, x.size()), want, 1e-3);
            } else ok("silu_bf16", false, "row absent");
            /* Past the grid's y limit: the audio subsampler's rectifier runs over 96000 rows of
             * 128, beyond the 65535 a one-grid-row-per-matrix-row launch reaches. */
            if (const RadKernelInfo* r = find(pl, "relu_bf16")) {
                const int64_t Mb = 70000, nb = 32;
                std::vector<float> xb = randv((size_t)(Mb * nb), 2.0f);
                void* dxb = up(xb);
                void* dyb = dalloc((size_t)(Mb * nb) * 2);
                std::vector<RadTensor> ts = { T(dxb, RAD_BF16, { Mb, nb }), T(dyb, RAD_BF16, { Mb, nb }) };
                std::vector<RadParam> ps = { PI("M", Mb), PI("n", nb), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want(xb.size());
                for (size_t i = 0; i < xb.size(); ++i) want[i] = xb[i] > 0 ? xb[i] : 0;
                check("relu_bf16 numerics past 65535 rows", down(dyb, xb.size()), want, 0.0);
            }
            if (const RadKernelInfo* r = find(pl, "clamp_bf16")) {
                std::vector<float> lo = { -0.5f }, hi = { 0.7f };
                void* dl = up(lo);
                void* dh = up(hi);
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, n }), T(dl, RAD_BF16, { 1 }),
                                              T(dh, RAD_BF16, { 1 }), T(dy, RAD_BF16, { M, n }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n", n), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                    want[i] = x[i] < lo[0] ? lo[0] : (x[i] > hi[0] ? hi[0] : x[i]);
                check("clamp_bf16 numerics", down(dy, x.size()), want, 0.0);
            } else ok("clamp_bf16", false, "row absent");
        }
        if (sel("act_mul_bf16")) {
            const RadKernelInfo* r = find(pl, "act_mul_bf16");
            const int64_t M = 4, n = 200;
            std::vector<float> g = randv((size_t)(M * n), 4.0f), u = randv((size_t)(M * n), 1.0f);
            void* dg = up(g);
            void* du = up(u);
            void* dy = dalloc((size_t)(M * n) * 2);
            std::vector<RadTensor> ts = { T(dg, RAD_BF16, { M, n }), T(du, RAD_BF16, { M, n }),
                                          T(dy, RAD_BF16, { M, n }) };
            std::vector<RadParam> ps = { PI("M", M), PI("n", n), PS("act", "sigmoid"),
                                         PS("dtype", "bf16") };
            if (r) run_case(r, ts, ps);
            std::vector<float> want(g.size());
            for (size_t i = 0; i < g.size(); ++i)
                want[i] = h_bf16(h_to_bf16(1.0f / (1.0f + std::exp(-g[i])))) * u[i];
            check("act_mul_bf16 sigmoid numerics", down(dy, g.size()), want, 1e-2);
        }
        if (sel("rmsnorm_bf16")) {
            const RadKernelInfo* r = find(pl, "rmsnorm_bf16");
            const int64_t M = 3, n = 1024;
            std::vector<float> x = randv((size_t)(M * n), 3.0f), w = randv((size_t)n, 1.0f);
            void* dx = up(x);
            void* dw = up(w);
            void* dy = dalloc((size_t)(M * n) * 2);
            std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, n }), T(dw, RAD_BF16, { n }),
                                          T(dy, RAD_BF16, { M, n }) };
            std::vector<RadParam> ps = { PI("M", M), PI("n", n), PF("eps", 1e-6),
                                         PS("dtype", "bf16"), PF("mult", 0.5) };
            if (r) run_case(r, ts, ps);
            std::vector<float> want(x.size());
            for (int64_t m = 0; m < M; ++m) {
                double ss = 0;
                for (int64_t i = 0; i < n; ++i) ss += (double)x[(size_t)(m * n + i)] * x[(size_t)(m * n + i)];
                const float s = 1.0f / std::sqrt((float)(ss / n) + 1e-6f);
                for (int64_t i = 0; i < n; ++i)
                    want[(size_t)(m * n + i)] = x[(size_t)(m * n + i)] * s * w[(size_t)i] * 0.5f;
            }
            check("rmsnorm_bf16 mult numerics", down(dy, x.size()), want, 1e-2);
        }
        if (sel("conv2d_bf16")) {
            const RadKernelInfo* r = find(pl, "conv2d_bf16");
            if (!r) ok("conv2d_bf16", false, "row absent");
            /* The subsampler's two layers, two segments each, with a masked row. */
            struct G { int64_t M, W, cin, cout, cdiv, vst; };
            for (const G& g : { G{ 24, 128, 1, 128, 1, 1 }, G{ 12, 64, 128, 32, 2, 2 } }) {
                if (!r) break;
                const int64_t kh = 3, kw = 3, st = 2, pad = 1;
                const int64_t Wo = (g.W + 2 * pad - kw) / st + 1;
                const int64_t T0 = g.M / 2;                         /* rows per segment */
                const int64_t To = (T0 + 2 * pad - kh) / st + 1, Mo = 2 * To;
                std::vector<float> x = randv((size_t)(g.M * g.W * g.cin), 1.0f);
                std::vector<float> w = randv((size_t)(g.cout * g.cin * kh * kw), 0.3f);
                std::vector<int32_t> cu = { 0, (int32_t)(T0 * g.cdiv), (int32_t)(g.M * g.cdiv) };
                std::vector<int32_t> valid((size_t)(g.M * g.vst), 1);
                valid[(size_t)(3 * g.vst)] = 0;
                void* dx = up(x);
                void* dw = up(w);
                void* dcu = dalloc(cu.size() * 4);
                void* dv = dalloc(valid.size() * 4);
                fill_i32(dcu, cu);
                fill_i32(dv, valid);
                void* dy = dalloc((size_t)(Mo * Wo * g.cout) * 2);
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { g.M, g.W * g.cin }),
                                              T(dw, RAD_BF16, { g.cout, g.cin, kh, kw }),
                                              T(dcu, RAD_I32, { 3 }),
                                              T(dv, RAD_I32, { g.M * g.vst }),
                                              T(dy, RAD_BF16, { Mo, Wo * g.cout }) };
                std::vector<RadParam> ps = { PI("M", g.M), PI("width", g.W), PI("cin", g.cin),
                                             PI("cout", g.cout), PI("kh", kh), PI("kw", kw),
                                             PI("stride", st), PI("pad", pad), PI("cu_div", g.cdiv),
                                             PI("valid_stride", g.vst), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want((size_t)(Mo * Wo * g.cout), 0.0f);
                for (int64_t s = 0; s < 2; ++s)
                    for (int64_t to = 0; to < To; ++to)
                        for (int64_t wo = 0; wo < Wo; ++wo)
                            for (int64_t co = 0; co < g.cout; ++co) {
                                float acc = 0.0f;
                                for (int64_t i = 0; i < kh; ++i) {
                                    const int64_t ti = to * st - pad + i;
                                    if (ti < 0 || ti >= T0) continue;
                                    const int64_t tg = s * T0 + ti;
                                    if (!valid[(size_t)(tg * g.vst)]) continue;
                                    for (int64_t j = 0; j < kw; ++j) {
                                        const int64_t wi = wo * st - pad + j;
                                        if (wi < 0 || wi >= g.W) continue;
                                        for (int64_t ci = 0; ci < g.cin; ++ci)
                                            acc += x[(size_t)((tg * g.W + wi) * g.cin + ci)] *
                                                   w[(size_t)(((co * g.cin + ci) * kh + i) * kw + j)];
                                    }
                                }
                                want[(size_t)(((s * To + to) * Wo + wo) * g.cout + co)] = acc;
                            }
                check(g.cin == 1 ? "conv2d_bf16 layer0 numerics" : "conv2d_bf16 layer1 numerics",
                      down(dy, want.size()), want, 1e-2);
            }
            /* Qwen3-ASR's: a [mel, time] kernel (time last) with a bias and gelu, over seconds
             * staged 13 rows apart with the alignment rows masked; and the 7680 -> 1024 linear as
             * a 1 x W convolution over the frequency axis. Small channel counts, same indexing. */
            struct Q { int64_t M, W, cin, cout, kh, kw, st, pad, tl; bool bias, gelu; };
            for (const Q& q : { Q{ 26, 16, 6, 10, 3, 3, 2, 1, 1, true, true },
                                Q{ 13, 8, 6, 12, 1, 8, 1, 0, 0, false, false } }) {
                if (!r) break;
                const int64_t Wo = (q.W + 2 * q.pad - q.kw) / q.st + 1;
                const int64_t To = (q.M + 2 * q.pad - q.kh) / q.st + 1;
                std::vector<float> x = randv((size_t)(q.M * q.W * q.cin), 1.0f);
                std::vector<float> w = randv((size_t)(q.cout * q.cin * q.kh * q.kw), 0.3f);
                std::vector<float> bias = randv((size_t)q.cout, 0.5f);
                std::vector<int32_t> cu = { 0, (int32_t)q.M };
                std::vector<int32_t> valid((size_t)q.M, 1);
                if (q.bias) for (int64_t t = 0; t < q.M; ++t) valid[(size_t)t] = t % 13 < 11;
                void* dx = up(x);
                void* dw = up(w);
                void* db = up(bias);
                void* dcu = dalloc(cu.size() * 4);
                void* dv = dalloc(valid.size() * 4);
                fill_i32(dcu, cu);
                fill_i32(dv, valid);
                void* dy = dalloc((size_t)(To * Wo * q.cout) * 2);
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { q.M, q.W * q.cin }),
                                              T(dw, RAD_BF16, { q.cout, q.cin, q.kh, q.kw }),
                                              T(dcu, RAD_I32, { 2 }),
                                              q.bias ? T(dv, RAD_I32, { q.M }) : RadTensor{},
                                              T(dy, RAD_BF16, { To, Wo * q.cout }),
                                              q.bias ? T(db, RAD_BF16, { q.cout }) : RadTensor{} };
                std::vector<RadParam> ps = { PI("M", q.M), PI("width", q.W), PI("cin", q.cin),
                                             PI("cout", q.cout), PI("kh", q.kh), PI("kw", q.kw),
                                             PI("stride", q.st), PI("pad", q.pad),
                                             PS("dtype", "bf16"), PI("t_last", q.tl),
                                             PS("act", q.gelu ? "gelu" : "none") };
                run_case(r, ts, ps);
                std::vector<float> want((size_t)(To * Wo * q.cout), 0.0f);
                for (int64_t to = 0; to < To; ++to)
                    for (int64_t wo = 0; wo < Wo; ++wo)
                        for (int64_t co = 0; co < q.cout; ++co) {
                            float acc = 0.0f;
                            for (int64_t i = 0; i < q.kh; ++i) {
                                const int64_t ti = to * q.st - q.pad + i;
                                if (ti < 0 || ti >= q.M || (q.bias && !valid[(size_t)ti])) continue;
                                for (int64_t j = 0; j < q.kw; ++j) {
                                    const int64_t wi = wo * q.st - q.pad + j;
                                    if (wi < 0 || wi >= q.W) continue;
                                    for (int64_t ci = 0; ci < q.cin; ++ci) {
                                        const int64_t wix = q.tl
                                            ? ((co * q.cin + ci) * q.kw + j) * q.kh + i
                                            : ((co * q.cin + ci) * q.kh + i) * q.kw + j;
                                        acc += x[(size_t)((ti * q.W + wi) * q.cin + ci)] *
                                               w[(size_t)wix];
                                    }
                                }
                            }
                            if (q.bias) acc += bias[(size_t)co];
                            if (q.gelu) {
                                const float v = h_bf16(h_to_bf16(acc));
                                acc = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
                            }
                            want[(size_t)((to * Wo + wo) * q.cout + co)] = acc;
                        }
                check(q.gelu ? "conv2d_bf16 time-last bias gelu numerics"
                             : "conv2d_bf16 1 x W projection numerics",
                      down(dy, want.size()), want, 1e-2);
            }
        }
        if (sel("conv2d_unfold_bf16")) {
            /* The receptive fields of a [mel, time] 3x3 stride-2 convolution over one segment with
             * a masked row, columns in the time-last weight's order. A copy: exact. */
            const RadKernelInfo* r = find(pl, "conv2d_unfold_bf16");
            if (!r) ok("conv2d_unfold_bf16", false, "row absent");
            else {
                const int64_t M = 10, W = 6, cin = 5, Wo = 3, To = 5, K = cin * 9;
                std::vector<float> x = randv((size_t)(M * W * cin), 1.0f);
                std::vector<int32_t> cu = { 0, (int32_t)M };
                std::vector<int32_t> valid((size_t)M, 1);
                valid[4] = 0;
                void* dx = up(x);
                void* dcu = dalloc(8);
                void* dv = dalloc((size_t)M * 4);
                fill_i32(dcu, cu);
                fill_i32(dv, valid);
                void* dy = dalloc((size_t)(To * Wo * K) * 2);
                std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, W * cin }), T(dcu, RAD_I32, { 2 }),
                                              T(dv, RAD_I32, { M }),
                                              T(dy, RAD_BF16, { To * Wo, K }) };
                std::vector<RadParam> ps = { PI("M", M), PI("width", W), PI("cin", cin),
                                             PI("kh", 3), PI("kw", 3), PI("stride", 2),
                                             PI("pad", 1), PI("t_last", 1), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want((size_t)(To * Wo * K), 0.0f);
                for (int64_t to = 0; to < To; ++to)
                    for (int64_t wo = 0; wo < Wo; ++wo)
                        for (int64_t ci = 0; ci < cin; ++ci)
                            for (int64_t j = 0; j < 3; ++j)
                                for (int64_t i = 0; i < 3; ++i) {
                                    const int64_t ti = to * 2 - 1 + i, wi = wo * 2 - 1 + j;
                                    float v = 0.0f;
                                    if (ti >= 0 && ti < M && wi >= 0 && wi < W && valid[(size_t)ti])
                                        v = x[(size_t)((ti * W + wi) * cin + ci)];
                                    want[(size_t)((to * Wo + wo) * K + (ci * 3 + j) * 3 + i)] = v;
                                }
                check("conv2d_unfold_bf16 numerics", down(dy, want.size()), want, 1e-6);
            }
        }
        if (sel("seg_sinusoid_bf16")) {
            const RadKernelInfo* r = find(pl, "seg_sinusoid_bf16");
            if (!r) ok("seg_sinusoid_bf16", false, "row absent");
            else {
                /* Two clips of 30 and 17 rows in units of 8, positions mod 13. */
                const int64_t M = 47, n = 64;
                std::vector<int32_t> cu = { 0, 240, 376 };
                void* dcu = dalloc(cu.size() * 4);
                fill_i32(dcu, cu);
                void* dy = dalloc((size_t)(M * n) * 2);
                std::vector<RadTensor> ts = { T(dcu, RAD_I32, { 3 }), T(dy, RAD_BF16, { M, n }) };
                std::vector<RadParam> ps = { PI("M", M), PI("n", n), PI("period", 13),
                                             PI("cu_div", 8), PS("dtype", "bf16") };
                run_case(r, ts, ps);
                std::vector<float> want((size_t)(M * n));
                const float inc = (float)(std::log(10000.0) / (double)(n / 2 - 1));
                for (int64_t row = 0; row < M; ++row) {
                    const int64_t p = (row < 30 ? row : row - 30) % 13;
                    for (int64_t i = 0; i < n / 2; ++i) {
                        const float ang = (float)p * std::exp((float)i * -inc);
                        want[(size_t)(row * n + i)] = std::sin(ang);
                        want[(size_t)(row * n + n / 2 + i)] = std::cos(ang);
                    }
                }
                check("seg_sinusoid_bf16 numerics", down(dy, want.size()), want, 5e-3);
            }
        }
        if (sel("attn_dense_h64_bf16")) {
            /* Blocks: two clips of 30 and 17 rows (cu_seqlens in units of 8), each cut into
             * blocks of 13 rows from its start, attending within the block; 2 heads of 64 in one
             * fused [q|k|v] row. Against a host softmax over the same bf16 inputs. */
            const RadKernelInfo* r = find(pl, "attn_dense_h64_bf16");
            if (r) {
                const int64_t M = 47, H = 2, D = 64, width = 3 * H * D, blk = 13;
                std::vector<float> qkv = randv((size_t)(M * width), 1.0f);
                std::vector<int32_t> cu = { 0, 240, 376 };
                void* dq = up(qkv);
                void* dcu = dalloc(cu.size() * 4);
                fill_i32(dcu, cu);
                void* dout = dalloc((size_t)(M * H * D) * 2);
                auto col = [&](int64_t first) {
                    RadTensor t = T(dq, RAD_BF16, { M, H * D });
                    t.data = (void*)((uint16_t*)dq + first);
                    t.stride[0] = width;
                    return t;
                };
                std::vector<RadTensor> ts = { col(0), col(H * D), col(2 * H * D),
                                              T(dcu, RAD_I32, { 3 }),
                                              T(dout, RAD_BF16, { M, H * D }) };
                std::vector<RadParam> ps = { PI("M", M), PI("head_dim", D), PI("gqa", 1),
                                             PI("causal", 0), PS("dtype", "bf16"),
                                             PF("scale", 0.125), PI("max_seqlen", M),
                                             PI("cu_div", 8), PI("block", blk) };
                run_case(r, ts, ps);
                std::vector<float> want((size_t)(M * H * D), 0.0f);
                for (int64_t i = 0; i < M; ++i) {
                    const int64_t s0 = i < 30 ? 0 : 30, s1 = i < 30 ? 30 : M;
                    const int64_t b0 = s0 + (i - s0) / blk * blk;
                    const int64_t b1 = std::min(s1, b0 + blk);
                    for (int64_t h = 0; h < H; ++h) {
                        std::vector<double> sc((size_t)(b1 - b0));
                        double mx = -1e30;
                        for (int64_t j = b0; j < b1; ++j) {
                            double d = 0;
                            for (int64_t e = 0; e < D; ++e)
                                d += qkv[(size_t)(i * width + h * D + e)] *
                                     qkv[(size_t)(j * width + H * D + h * D + e)];
                            sc[(size_t)(j - b0)] = d * 0.125;
                            mx = std::max(mx, sc[(size_t)(j - b0)]);
                        }
                        double sum = 0;
                        for (auto& v : sc) { v = std::exp(v - mx); sum += v; }
                        for (int64_t e = 0; e < D; ++e) {
                            double a = 0;
                            for (int64_t j = b0; j < b1; ++j)
                                a += sc[(size_t)(j - b0)] *
                                     qkv[(size_t)(j * width + 2 * H * D + h * D + e)];
                            want[(size_t)(i * H * D + h * D + e)] = (float)(a / sum);
                        }
                    }
                }
                check("attn_dense_h64_bf16 blocks numerics", down(dout, want.size()), want, 2e-2);
            }
        }
        if (sel("conv1d_dw_bf16")) {
            const RadKernelInfo* r = find(pl, "conv1d_dw_bf16");
            const int64_t M = 20, C = 1024, K = 5;
            std::vector<float> x = randv((size_t)(M * C), 1.0f), w = randv((size_t)(C * K), 0.5f);
            std::vector<int32_t> cu = { 0, 7, 20 };
            void* dx = up(x);
            void* dw = up(w);
            void* dcu = dalloc(12);
            fill_i32(dcu, cu);
            void* dy = dalloc((size_t)(M * C) * 2);
            std::vector<RadTensor> ts = { T(dx, RAD_BF16, { M, C }), T(dw, RAD_BF16, { C, 1, K }),
                                          T(dcu, RAD_I32, { 3 }), T(dy, RAD_BF16, { M, C }) };
            std::vector<RadParam> ps = { PI("M", M), PI("n", C), PI("k", K), PS("dtype", "bf16") };
            if (r) run_case(r, ts, ps);
            std::vector<float> want((size_t)(M * C));
            for (int64_t t = 0; t < M; ++t) {
                const int64_t t0 = t < 7 ? 0 : 7;
                for (int64_t c = 0; c < C; ++c) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < K; ++j) {
                        const int64_t tt = t - (K - 1) + j;
                        if (tt >= t0) acc += x[(size_t)(tt * C + c)] * w[(size_t)(c * K + j)];
                    }
                    want[(size_t)(t * C + c)] = acc;
                }
            }
            check("conv1d_dw_bf16 numerics", down(dy, want.size()), want, 1e-2);
        }
        if (sel("rel_sinusoid_bf16")) {
            const RadKernelInfo* r = find(pl, "rel_sinusoid_bf16");
            const int64_t M = 13, n = 1024, h = n / 2;
            void* dy = dalloc((size_t)(M * n) * 2);
            std::vector<RadTensor> ts = { T(dy, RAD_BF16, { M, n }) };
            std::vector<RadParam> ps = { PI("M", M), PI("n", n), PF("max_timescale", 10000.0),
                                         PS("dtype", "bf16") };
            if (r) run_case(r, ts, ps);
            const float inc = (float)(std::log(10000.0) / (double)(h - 1));
            std::vector<float> want((size_t)(M * n));
            for (int64_t d = 0; d < M; ++d)
                for (int64_t i = 0; i < h; ++i) {
                    const float ang = (float)d * std::exp((float)i * -inc);
                    want[(size_t)(d * n + i)] = std::sin(ang);
                    want[(size_t)(d * n + h + i)] = std::cos(ang);
                }
            check("rel_sinusoid_bf16 numerics", down(dy, want.size()), want, 1e-2);
        }
        if (sel("attn_local_rel_bf16")) {
            const RadKernelInfo* r = find(pl, "attn_local_rel_bf16");
            const int64_t M = 40, H = 8, D = 128, L = 12, W = H * D;
            const float cap = 50.0f;
            std::vector<float> q = randv((size_t)(M * W), 2.0f), k = randv((size_t)(M * W), 2.0f);
            std::vector<float> v = randv((size_t)(M * W), 1.0f);
            std::vector<float> rk = randv((size_t)((L + 1) * W), 1.0f), pds = randv((size_t)D, 3.0f);
            std::vector<int32_t> cu = { 0, 17, 40 };
            void* dq = up(q);
            void* dk = up(k);
            void* dv = up(v);
            void* drk = up(rk);
            void* dp = up(pds);
            void* dcu = dalloc(12);
            fill_i32(dcu, cu);
            void* dout = dalloc((size_t)(M * W) * 2);
            std::vector<RadTensor> ts = { T(dq, RAD_BF16, { M, W }), T(dk, RAD_BF16, { M, W }),
                                          T(dv, RAD_BF16, { M, W }), T(drk, RAD_BF16, { L + 1, W }),
                                          T(dp, RAD_BF16, { D }), T(dcu, RAD_I32, { 3 }),
                                          T(dout, RAD_BF16, { M, W }) };
            std::vector<RadParam> ps = { PI("M", M), PI("head_dim", D), PI("n_head", H),
                                         PI("left", L), PI("right", 0), PF("softcap", cap),
                                         PS("dtype", "bf16") };
            if (r) run_case(r, ts, ps);
            const float qs = (float)(1.0 / std::sqrt((double)D) / std::log(2.0));
            const float ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
            std::vector<float> want((size_t)(M * W));
            for (int64_t i = 0; i < M; ++i)
                for (int64_t hh = 0; hh < H; ++hh) {
                    const int64_t t0 = i < 17 ? 0 : 17, j0 = std::max(t0, i - L);
                    std::vector<float> lg;
                    float mx = -1e30f;
                    for (int64_t j = j0; j <= i; ++j) {
                        float ss = 0.0f;
                        for (int64_t d = 0; d < D; ++d) {
                            const float x = pds[(size_t)d];
                            const float sp = h_bf16(h_to_bf16(
                                x > 20.0f ? x : (x > 0 ? x + std::log1p(std::exp(-x))
                                                       : std::log1p(std::exp(x)))));
                            const float qf = q[(size_t)(i * W + hh * D + d)] * qs * sp;
                            ss += qf * (k[(size_t)(j * W + hh * D + d)] * ks) +
                                  qf * rk[(size_t)((i - j) * W + hh * D + d)];
                        }
                        lg.push_back(cap * std::tanh(ss / cap));
                        mx = std::max(mx, lg.back());
                    }
                    float sum = 0.0f;
                    for (float& l : lg) { l = std::exp(l - mx); sum += l; }
                    for (int64_t d = 0; d < D; ++d) {
                        float acc = 0.0f;
                        for (int64_t j = j0; j <= i; ++j)
                            acc += lg[(size_t)(j - j0)] / sum * v[(size_t)(j * W + hh * D + d)];
                        want[(size_t)(i * W + hh * D + d)] = acc;
                    }
                }
            check("attn_local_rel_bf16 numerics", down(dout, want.size()), want, 1e-2);
        }
    }
    if (sel("cast_bf16_f32")) {
        const RadKernelInfo* r = find(pl, "cast_bf16_f32");
        if (!r) ok("cast_bf16_f32", false, "row absent");
        else {
            std::vector<RadTensor> ts = { T(nullptr, RAD_BF16, { 3, 768 }),
                                          T(nullptr, RAD_F32, { 3, 768 }) };
            std::vector<RadParam> ps = { PI("M", 3), PI("n", 768), PS("from", "bf16"),
                                         PS("to", "f32") };
            run_case(r, ts, ps);
        }
    }


    /* --- cast: the strided row copy an intermediate-hidden-state tap writes with -------------
     *
     * The destination is a column slice of a wider buffer, as when five taps of the trunk's
     * residual stream go side by side in one [T, 5 * n_embd] row; a wrong pitch writes the wrong
     * columns. Checked byte for byte, including that the untouched columns keep their fill. */
    if (sel("cast_bf16_bf16")) {
        const RadKernelInfo* r = find(pl, "cast_bf16_bf16");
        if (!r) ok("cast_bf16_bf16", false, "row absent");
        else {
            const int64_t M = 7, n = 40, wide = 3 * n, slot = 1;   /* the middle third */
            std::vector<uint16_t> src((size_t)(M * n));
            for (int64_t i = 0; i < M * n; ++i)
                src[(size_t)i] = h_to_bf16(0.031f * (float)((i * 13) % 97) - 1.25f);
            void* d_src = dalloc((size_t)M * n * 2);
            void* d_dst = dalloc((size_t)M * wide * 2);
            if (!d_src || !d_dst) { ok("cast alloc", false, "hipMalloc failed"); }
            else {
                st_memcpy(d_src, src.data(), (size_t)M * n * 2, hipMemcpyHostToDevice);
                /* A fill every untouched column must keep. */
                std::vector<uint16_t> fill((size_t)(M * wide), h_to_bf16(-7.5f));
                st_memcpy(d_dst, fill.data(), (size_t)M * wide * 2, hipMemcpyHostToDevice);

                RadTensor ty = T(d_dst, RAD_BF16, { M, n });
                ty.data   = (void*)((uint16_t*)d_dst + slot * n);   /* the slice's first element */
                ty.stride[0] = wide;                                /* at the wide row pitch */
                std::vector<RadTensor> ts = { T(d_src, RAD_BF16, { M, n }), ty };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", n), PS("from", "bf16"), PS("to", "bf16"),
                };
                run_case(r, ts, ps);

                std::vector<uint16_t> got((size_t)(M * wide));
                st_memcpy(got.data(), d_dst, (size_t)M * wide * 2, hipMemcpyDeviceToHost);
                bool copied = true, spilled = false;
                for (int64_t m = 0; m < M && copied && !spilled; ++m)
                    for (int64_t c = 0; c < wide; ++c) {
                        const uint16_t g = got[(size_t)(m * wide + c)];
                        if (c >= slot * n && c < (slot + 1) * n) {
                            if (g != src[(size_t)(m * n + c - slot * n)]) { copied = false; break; }
                        } else if (g != fill[0]) { spilled = true; break; }
                    }
                ok("cast_bf16_bf16 copies into a column slice, byte for byte", copied);
                ok("cast_bf16_bf16 leaves the columns outside the slice alone", !spilled);
            }
        }
    }
    /* --- fill: the op a rank with nothing to compute writes a defined buffer with --- */
    if (sel("fill_bf16")) {
        const RadKernelInfo* r = find(pl, "fill_bf16");
        if (!r) ok("fill_bf16", false, "row absent");
        else {
            const int64_t M = 5, n = 48, wide = 2 * n;   /* a column slice at a wide pitch */
            void* d = dalloc((size_t)M * wide * 2);
            if (!d) ok("fill alloc", false, "hipMalloc failed");
            else {
                std::vector<uint16_t> pre((size_t)(M * wide), h_to_bf16(-7.5f));
                st_memcpy(d, pre.data(), (size_t)M * wide * 2, hipMemcpyHostToDevice);
                RadTensor ty = T(d, RAD_BF16, { M, n });
                ty.stride[0] = wide;
                std::vector<RadTensor> ts = { ty };
                /* The default value, zero, must be exactly 0x0000. */
                std::vector<RadParam> p0 = { PI("M", M), PI("n", n), PS("dtype", "bf16") };
                run_case(r, ts, p0);
                std::vector<uint16_t> got((size_t)(M * wide));
                st_memcpy(got.data(), d, (size_t)M * wide * 2, hipMemcpyDeviceToHost);
                bool zero = true, spilled = false;
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t c = 0; c < wide; ++c) {
                        const uint16_t g = got[(size_t)(m * wide + c)];
                        if (c < n) { if (g != 0) zero = false; }
                        else if (g != pre[0]) spilled = true;
                    }
                ok("fill_bf16 writes exactly zero into the slice by default", zero);
                ok("fill_bf16 leaves the columns past the pitch alone", !spilled);
                /* A value: rounds to nearest even once, like every store in this library. */
                std::vector<RadParam> p1 = { PI("M", M), PI("n", n), PS("dtype", "bf16"),
                                             PF("value", 1.37) };
                run_case(r, ts, p1);
                st_memcpy(got.data(), d, (size_t)M * wide * 2, hipMemcpyDeviceToHost);
                const uint16_t want = h_to_bf16(1.37f);
                bool val = true;
                for (int64_t m = 0; m < M && val; ++m)
                    for (int64_t c = 0; c < n; ++c) if (got[(size_t)(m * wide + c)] != want) { val = false; break; }
                ok("fill_bf16 writes a given value, RNE-rounded", val);
            }
        }
    }
    /* --- byte rows: the KV tier's transfer pair --- */
    /* Pack and unpack a pool, the real layout in miniature: a "layer" is `rows` blocks of `n`
     * bytes, a session's blocks are scattered through it, and exactly those blocks must come back.
     * Bytes, since a KV fragment holds arbitrary data and must compare exactly. */
    if (sel("gather_rows_u8") || sel("scatter_rows_u8")) {
        const RadKernelInfo* rg = find(pl, "gather_rows_u8");
        const RadKernelInfo* rs = find(pl, "scatter_rows_u8");
        if (!rg || !rs) ok("byte row pair", false, "row absent");
        else {
            const int64_t rows = 64, n = 256, M = 5;
            const int32_t sel_blocks[5] = { 63, 0, 17, 40, 9 };   /* scattered, and not in order */
            std::vector<uint8_t> pool((size_t)(rows * n));
            for (int64_t i = 0; i < rows * n; ++i) pool[(size_t)i] = (uint8_t)((i * 31 + 7) & 0xff);

            void* d_pool = dalloc((size_t)rows * n);
            void* d_pack = dalloc((size_t)M * n);
            void* d_idx  = dalloc(sizeof sel_blocks);
            if (!d_pool || !d_pack || !d_idx) ok("byte row alloc", false, "hipMalloc failed");
            else {
                st_memcpy(d_pool, pool.data(), (size_t)rows * n, hipMemcpyHostToDevice);
                st_memcpy(d_idx, sel_blocks, sizeof sel_blocks, hipMemcpyHostToDevice);
                std::vector<uint8_t> poison((size_t)(M * n), 0xA5);
                st_memcpy(d_pack, poison.data(), (size_t)M * n, hipMemcpyHostToDevice);

                std::vector<RadTensor> tg = { T(d_pool, RAD_U8, { rows, n }),
                                              T(d_idx,  RAD_I32, { M }),
                                              T(d_pack, RAD_U8, { M, n }) };
                std::vector<RadParam> pg = { PI("M", M), PI("n", n), PS("dtype", "u8") };
                run_case(rg, tg, pg);

                std::vector<uint8_t> packed((size_t)(M * n));
                st_memcpy(packed.data(), d_pack, (size_t)M * n, hipMemcpyDeviceToHost);
                bool gathered = true;
                for (int64_t m = 0; m < M && gathered; ++m)
                    for (int64_t i = 0; i < n; ++i)
                        if (packed[(size_t)(m * n + i)] !=
                            pool[(size_t)(sel_blocks[m] * n + i)]) { gathered = false; break; }
                ok("gather_rows_u8 packs scattered blocks byte for byte", gathered);

                /* Scatter them into a different pool prefilled with a marker; unnamed blocks must
                 * keep it, since the destination is shared with other sessions. */
                std::vector<uint8_t> mark((size_t)(rows * n), 0x5C);
                st_memcpy(d_pool, mark.data(), (size_t)rows * n, hipMemcpyHostToDevice);
                std::vector<RadTensor> ts2 = { T(d_pack, RAD_U8, { M, n }),
                                               T(d_idx,  RAD_I32, { M }),
                                               T(d_pool, RAD_U8, { rows, n }) };
                run_case(rs, ts2, pg);

                std::vector<uint8_t> back((size_t)(rows * n));
                st_memcpy(back.data(), d_pool, (size_t)rows * n, hipMemcpyDeviceToHost);
                bool restored = true, clobbered = false;
                std::vector<bool> named((size_t)rows, false);
                for (int64_t m = 0; m < M; ++m) named[(size_t)sel_blocks[m]] = true;
                for (int64_t b = 0; b < rows && restored && !clobbered; ++b)
                    for (int64_t i = 0; i < n; ++i) {
                        const uint8_t g = back[(size_t)(b * n + i)];
                        if (named[(size_t)b]) {
                            if (g != pool[(size_t)(b * n + i)]) { restored = false; break; }
                        } else if (g != 0x5C) { clobbered = true; break; }
                    }
                ok("scatter_rows_u8 restores every named block byte for byte", restored);
                ok("scatter_rows_u8 leaves every block it was not given alone", !clobbered);
            }
        }
    }
    /* --- gated delta net --- */
    {
        const int64_t T_ = 128, H = 4, Hg = 2, K = 128, V = 128, chunk = 64, N = 2;
        if (sel("gdn_kkt_solve_k128_c64_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_kkt_solve_k128_c64_bf16");
            if (!r) ok("gdn_kkt_solve_k128_c64_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, Hg, K }), T(nullptr, RAD_F32, { T_, H }),
                    T(nullptr, RAD_F32, { T_, H }),      TI32({ N + 1 }, cu_even(N, T_)),
                    T(nullptr, RAD_BF16, { T_, H, chunk }),
                };
                std::vector<RadParam> ps = { PI("M", T_), PI("head_k", K), PI("chunk", chunk) };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_chunk_scan_k128_v128_c64_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_chunk_scan_k128_v128_c64_bf16");
            if (!r) ok("gdn_chunk_scan_k128_v128_c64_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, Hg, K }), T(nullptr, RAD_BF16, { T_, Hg, K }),
                    T(nullptr, RAD_BF16, { T_, H, V }),  T(nullptr, RAD_BF16, { T_, H, chunk }),
                    T(nullptr, RAD_F32,  { T_, H }),     T(nullptr, RAD_F32,  { T_, H }),
                    T(nullptr, RAD_F32,  { N, H, V, K }), TI32({ N + 1 }, cu_even(N, T_)),
                    T(nullptr, RAD_BF16, { T_, H, V }),  T(nullptr, RAD_F32, { N, H, V, K }),
                };
                std::vector<RadParam> ps = {
                    PI("M", T_), PI("head_k", K), PI("head_v", V), PI("chunk", chunk),
                    PF("scale", 0.088388),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_conv_prep_w4_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_conv_prep_w4_h128_bf16");
            if (!r) ok("gdn_conv_prep_w4_h128_bf16", false, "row absent");
            else {
                const int64_t width = 4, xdim = Hg * K * 2 + H * V;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_, xdim }),          /* x                */
                    T(nullptr, RAD_BF16, { xdim, width }),       /* conv weight      */
                    T(nullptr, RAD_BF16, { xdim }),              /* conv bias        */
                    T(nullptr, RAD_F32,  { H }),                 /* A_log            */
                    T(nullptr, RAD_F32,  { H }),                 /* dt_bias          */
                    T(nullptr, RAD_BF16, { N, xdim, width }),    /* conv state       */
                    TI32({ N + 1 }, cu_even(N, T_)),             /* cu               */
                    T(nullptr, RAD_BF16, { T_, H }),             /* a                */
                    T(nullptr, RAD_BF16, { T_, H }),             /* b_gate           */
                    TI32({ N }, iota_i32(N)),                    /* cache_idx        */
                    T(nullptr, RAD_U8,   { N }),                 /* has_init         */
                    T(nullptr, RAD_BF16, { T_, Hg, K }),         /* q                */
                    T(nullptr, RAD_BF16, { T_, Hg, K }),         /* k                */
                    T(nullptr, RAD_BF16, { T_, H, V }),          /* v                */
                    T(nullptr, RAD_F32,  { T_, H }),             /* g                */
                    T(nullptr, RAD_F32,  { T_, H }),             /* beta             */
                };
                std::vector<RadParam> ps = {
                    PI("M", T_), PI("head_k", K), PI("head_v", V), PI("chunk", chunk),
                    PI("conv_width", width),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_conv_update_w4_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_conv_update_w4_h128_bf16");
            if (!r) ok("gdn_conv_update_w4_h128_bf16", false, "row absent");
            else {
                const int64_t width = 4, xdim = Hg * K * 2 + H * V, Td = N;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Td, xdim }),
                    T(nullptr, RAD_BF16, { xdim, width }),
                    T(nullptr, RAD_BF16, { xdim }),
                    T(nullptr, RAD_BF16, { N, xdim, width }),
                    TI32({ N }, iota_i32(N)),                    /* state_idx        */
                    TI32({ N }, rep_i32(N, 1)),                  /* num_accepted     */
                    TI32({ N + 1 }, cu_even(N, Td)),             /* cu               */
                    T(nullptr, RAD_BF16, { Td, Hg, K }),
                    T(nullptr, RAD_BF16, { Td, Hg, K }),
                    T(nullptr, RAD_BF16, { Td, H, V }),
                };
                std::vector<RadParam> ps = {
                    PI("q_len", 1), PI("head_k", K), PI("head_v", V), PI("conv_width", width),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_recurrent_update_k128_v128_bf16_fp32state")) {
            const RadKernelInfo* r = find(pl, "gdn_recurrent_update_k128_v128_bf16_fp32state");
            if (!r) ok("gdn_recurrent_update_k128_v128_bf16_fp32state", false, "row absent");
            else {
                const int64_t Td = N;                            /* one candidate a sequence */
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Td, Hg, K }),         /* q                */
                    T(nullptr, RAD_BF16, { Td, Hg, K }),         /* k                */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* v                */
                    T(nullptr, RAD_BF16, { Td, H }),             /* a                */
                    T(nullptr, RAD_BF16, { Td, H }),             /* b                */
                    T(nullptr, RAD_F32,  { H }),                 /* A_log            */
                    T(nullptr, RAD_F32,  { H }),                 /* dt_bias          */
                    T(nullptr, RAD_F32,  { N, H, V, K }),        /* state            */
                    TI32({ N }, iota_i32(N)),                    /* state_idx        */
                    /* num_accepted indexes the state-index row as `nacc - 1`, so zero would read
                     * one int before the buffer; one accepted token is a plain decode step. */
                    TI32({ N }, rep_i32(N, 1)),                  /* num_accepted     */
                    TI32({ N + 1 }, cu_even(N, Td)),             /* cu               */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* z                */
                    T(nullptr, RAD_F32,  { V }),                 /* norm_w           */
                    T(nullptr, RAD_BF16, { Td, H, V }),          /* o                */
                    /* The folded quantiser, present so it is covered. head_v is the fp8 block
                     * width, so the scale plane is one f32 per (token, value head), and ref
                     * quantises the bf16 it stored, so the two agree byte for byte. */
                    T(nullptr, RAD_F8E4M3, { Td, H * V }),       /* o_q              */
                    T(nullptr, RAD_F32,    { Td, H }),           /* o_scale          */
                };
                std::vector<RadParam> ps = {
                    PI("q_len", 1), PI("head_k", K), PI("head_v", V), PF("scale", 1.0),
                    PF("eps", 1e-6), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        gdn_recurrent_anchor(pl);
        gdn_conv_fold(pl);
        if (sel("gdn_gated_rmsnorm_h128_bf16")) {
            const RadKernelInfo* r = find(pl, "gdn_gated_rmsnorm_h128_bf16");
            if (!r) ok("gdn_gated_rmsnorm_h128_bf16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { T_ * H, 128 }), T(nullptr, RAD_BF16, { T_ * H, 128 }),
                    T(nullptr, RAD_F32,  { 128 }),         T(nullptr, RAD_BF16, { T_ * H, 128 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", T_ * H), PI("channels", 128), PF("eps", 1e-6), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        gdn_gated_norm_sliced_gate(pl);
    }

    /* --- dflash conv --- */
    if (sel("dflash_conv_t2_g16_bf16")) {
        const RadKernelInfo* r = find(pl, "dflash_conv_t2_g16_bf16");
        if (!r) ok("dflash_conv_t2_g16_bf16", false, "row absent");
        else {
            const int64_t T_ = 64, H = 256, NG = H / 16, taps = 2;
            std::vector<RadTensor> ts = {
                T(nullptr, RAD_BF16, { T_, H }),
                T(nullptr, RAD_BF16, { T_, 2 * taps * NG }),
                T(nullptr, RAD_BF16, { taps, NG }),
                T(nullptr, RAD_BF16, { T_, H }),
            };
            std::vector<RadParam> ps = {
                PI("T", T_), PI("hidden_size", H), PI("taps", taps), PI("group_size", 16),
                PI("NG", NG), PI("block_size", 16), PS("dtype", "bf16"),
            };
            run_case(r, ts, ps);
        }
        /* The drafter's own shape (block 8, 5120 channels) and a block of 16, whose tap mask
         * differs. */
        dflash_conv_numeric(pl, 64, 256, 16, 2, 16);
        dflash_conv_numeric(pl, 32, 5120, 16, 2, 8);
    }

    /* --- dflash select, numerically: the chaining is what is tested.
     *
     * Step l's score depends on the token step l-1 chose. A kernel scoring every step against the
     * anchor would still emit plausible ids, and the drafter would just accept less. So the case
     * asserts the chained path and that an anchor-pinned walk over the same data differs.
     *
     * Every value is a power of two times a quarter, so f32 accumulation is exact in any order. */
    if (sel("dflash_select_bf16")) {
        const RadKernelInfo* r = find(pl, "dflash_select_bf16");
        if (!r) ok("dflash_select_bf16", false, "row absent");
        else {
            const int64_t M = 2, steps = 3, K = 4, R = 32, V = 64, rows = M * steps;
            auto pv   = [](int64_t t) { return t % 3 == 0 ? 0.5f : (t % 3 == 1 ? 1.0f : 2.0f); };
            auto sv   = [](int64_t t) { return 0.25f * (float)(t % 4); };
            auto cnd  = [&](int64_t row, int64_t c) { return (int32_t)((row * 7 + c * 10) % V); };
            auto unf  = [&](int64_t row, int64_t c) { return 8.0f * (float)((row + c * 3) % 5); };
            /* Strided, as the drafter passes it: the step's token_ids, `block` ids per sequence
             * with the anchor first. The entries between anchors (mask tokens) hold a value that
             * changes the answer if the stride is ignored. */
            const int64_t astride = steps + 1;
            const int32_t anchor[2] = { 2, 2 };
            std::vector<int32_t> h_anc((size_t)(M * astride), 7);
            for (int64_t s = 0; s < M; ++s) h_anc[(size_t)(s * astride)] = anchor[s];

            std::vector<int32_t> h_cand((size_t)(rows * K));
            std::vector<float>   h_un((size_t)(rows * K));
            for (int64_t i = 0; i < rows; ++i)
                for (int64_t c = 0; c < K; ++c) {
                    h_cand[(size_t)(i * K + c)] = cnd(i, c);
                    h_un[(size_t)(i * K + c)]   = unf(i, c);
                }
            std::vector<uint16_t> h_hp((size_t)(rows * R), h_to_bf16(1.0f));
            std::vector<uint16_t> h_pred((size_t)(V * R)), h_succ((size_t)(V * R));
            for (int64_t t = 0; t < V; ++t)
                for (int64_t j = 0; j < R; ++j) {
                    h_pred[(size_t)(t * R + j)] = h_to_bf16(pv(t));
                    h_succ[(size_t)(t * R + j)] = h_to_bf16(sv(t));
                }

            /* The reference walk, and the same walk with the predecessor pinned to the anchor. */
            int32_t want[2][8] = {}, pinned[2][8] = {};
            for (int pin = 0; pin < 2; ++pin)
                for (int64_t s = 0; s < M; ++s) {
                    int64_t chosen = 0;
                    for (int64_t l = 0; l < steps; ++l) {
                        const int64_t row = s * steps + l;
                        const int64_t pid = (l == 0 || pin) ? anchor[s] : cnd(row - 1, chosen);
                        int64_t best = 0;
                        float   bv   = -1e30f;
                        for (int64_t c = 0; c < K; ++c) {
                            const float sc = unf(row, c) + (float)R * pv(pid) * sv(cnd(row, c));
                            if (sc > bv) { bv = sc; best = c; }
                        }
                        chosen = best;
                        (pin ? pinned : want)[s][l] = cnd(row, best);
                    }
                }
            bool distinguishes = false;
            for (int64_t s = 0; s < M; ++s)
                for (int64_t l = 0; l < steps; ++l)
                    if (want[s][l] != pinned[s][l]) distinguishes = true;

            void* d_cand = dalloc((size_t)rows * K * 4);
            void* d_un   = dalloc((size_t)rows * K * 4);
            void* d_hp   = dalloc((size_t)rows * R * 2);
            void* d_anc  = dalloc(h_anc.size() * 4);
            void* d_pred = dalloc((size_t)V * R * 2);
            void* d_succ = dalloc((size_t)V * R * 2);
            void* d_out  = dalloc((size_t)M * steps * 4);
            if (!d_cand || !d_un || !d_hp || !d_anc || !d_pred || !d_succ || !d_out) {
                ok("dflash_select alloc", false, "hipMalloc failed");
            } else {
                st_memcpy(d_cand, h_cand.data(), (size_t)rows * K * 4, hipMemcpyHostToDevice);
                st_memcpy(d_un,   h_un.data(),   (size_t)rows * K * 4, hipMemcpyHostToDevice);
                st_memcpy(d_hp,   h_hp.data(),   (size_t)rows * R * 2, hipMemcpyHostToDevice);
                st_memcpy(d_anc,  h_anc.data(),  h_anc.size() * 4,     hipMemcpyHostToDevice);
                st_memcpy(d_pred, h_pred.data(), (size_t)V * R * 2,    hipMemcpyHostToDevice);
                st_memcpy(d_succ, h_succ.data(), (size_t)V * R * 2,    hipMemcpyHostToDevice);

                std::vector<RadTensor> ts = {
                    T(d_cand, RAD_I32,  { rows, K }), T(d_un,   RAD_F32,  { rows, K }),
                    T(d_hp,   RAD_BF16, { rows, R }), T(d_anc,  RAD_I32,  { M * astride }),
                    T(d_pred, RAD_BF16, { V, R }),    T(d_succ, RAD_BF16, { V, R }),
                    T(d_out,  RAD_I32,  { M, steps }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("steps", steps), PI("top_k", K), PI("rank", R),
                    PI("n_vocab", V), PI("anchor_stride", astride), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);

                std::vector<int32_t> got((size_t)(M * steps));
                st_memcpy(got.data(), d_out, (size_t)M * steps * 4, hipMemcpyDeviceToHost);
                bool same = true;
                for (int64_t s = 0; s < M && same; ++s)
                    for (int64_t l = 0; l < steps; ++l)
                        if (got[(size_t)(s * steps + l)] != want[s][l]) { same = false; break; }
                ok("dflash_select_bf16 walks the chained path", same);
                ok("dflash_select_bf16 case can tell a chained walk from an anchor-pinned one",
                   distinguishes);

                /* The bigram form (`hp` absent), DSpark's Markov head. `hp` above is exact ones,
                 * so absent must give bit-identical tokens. */
                std::vector<RadTensor> tb = ts;
                /* `absent`, not just a null `data`: run_case allocates any tensor with a rank and
                 * no pointer, which would pass uninitialised memory instead of no operand. */
                absent(tb[2]);
                st_memcpy(d_out, h_cand.data(), (size_t)M * steps * 4, hipMemcpyHostToDevice);
                run_case(r, tb, ps);
                std::vector<int32_t> bg((size_t)(M * steps));
                st_memcpy(bg.data(), d_out, (size_t)M * steps * 4, hipMemcpyDeviceToHost);
                bool bsame = true;
                for (int64_t s = 0; s < M && bsame; ++s)
                    for (int64_t l = 0; l < steps; ++l)
                        if (bg[(size_t)(s * steps + l)] != want[s][l]) { bsame = false; break; }
                ok("dflash_select_bf16 absent hp is a plane of ones (the bigram form)", bsame);
            }
        }
    }

    /* --- GEMM --- */
    {
        const int64_t M = 16, N = 256, K = 512;
        if (sel("gemm_bf16_nt_m16")) {
            const RadKernelInfo* r = find(pl, "gemm_bf16_nt_m16");
            if (!r) ok("gemm_bf16_nt_m16", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { N, K }),
                    T(nullptr, RAD_BF16, { M, N }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gemm_bf16_nt_m64")) {
            const RadKernelInfo* r = find(pl, "gemm_bf16_nt_m64");
            if (!r) ok("gemm_bf16_nt_m64", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { 64, K }), T(nullptr, RAD_BF16, { N, K }),
                    T(nullptr, RAD_BF16, { 64, N }),
                };
                std::vector<RadParam> ps = {
                    PI("M", 64), PI("N", N), PI("K", K), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        gemm_case(pl, "gemm_w4a16_nt_m64",   "w4a16",   M, N, K, RAD_F16, 0, 0, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w8a16_nt_m64",   "w8a16",   M, N, K, RAD_F16, 0, 0, 0, N*K,   N*(K/128)*4);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 5, 272, 384);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 40, 2560, 6144);
        i8a8_numeric(pl, "gemm_i8a8_nt_m16", 64, 512, 640);
        i8a8_numeric(pl, "gemm_i8a8_tiled", 200, 256, 640);
        i8a8_numeric(pl, "gemm_i8a8_tiled", 2048, 512, 2560);
        w8a16_numeric(pl, "gemm_w8a16_nt_m64", 5, 272, 384, RAD_F16);
        w8a16_numeric(pl, "gemm_w8a16_nt_m64", 64, 2560, 6144, RAD_F16);
        gemm_case(pl, "gemm_w4a8_nt_m64",    "w4a8",    M, N, K, RAD_I8,  1, 0, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w4a8_nt_m64",    "w4a8_asym", M, N, K, RAD_I8, 1, 1, 0, N*K/2, N*(K/128)*4);
        gemm_case(pl, "gemm_w8a8_nt_m64",    "w8a8",    M, N, K, RAD_I8,  1, 0, 0, N*K,   N*(K/128)*2);
        gemm_case(pl, "gemm_w2a8_nt",        "w2a8",    M, N, K, RAD_I8,  1, 1, 0, N*K/4, N*(K/128)*4);
        gemm_case(pl, "gemm_mxfp4a8_nt_m64", "mxfp4a8", M, N, K, RAD_F8E4M3, 1, 0, 1, N*K/2, (K/32)*N);
        gemm_case(pl, "gemm_mxfp4a8_tiled",  "mxfp4a8", 256, 256, K, RAD_F8E4M3, 1, 0, 1, 256*K/2, (K/32)*256);
        /* The skinny row first (validating the check), then the tiled row at both widths, ragged
         * in M and N, and at the size where tn=0 crosses over to TN=4. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_nt_m64", 40, 272, 640, 0);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 40, 272, 640, 2);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 272, 640, 2);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 400, 640, 4);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 2100, 512, 2560, 0);
        /* And with the fp8 family's (row, 128) activation grid, as architectures feed them. K = 640
         * is five blocks and 2560 twenty, so split-K segments and slab pairs cross block
         * boundaries. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_nt_m64", 40, 272, 640, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_nt_m64", 7, 512, 2560, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 272, 640, 2, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 400, 640, 4, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 2100, 512, 2560, 0, true);
        /* K not a multiple of 128: the TN=2 kernel takes the 64-wide slab. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 272, 576, 2);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 272, 576, 2, true);
        /* The decode row over its band: the split rule's choice, then every split and slab forced,
         * at both activation grids; N past a 128-column block, K a multiple of 64 but not 128 (the
         * slab falls back to 64), and DTM 1..4. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 8, 272, 640, 0, false);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 8, 272, 640, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 40, 400, 2560, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 64, 512, 2560, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 5, 1040, 576, 0, true);
        mxd_overlap_case(pl);
        /* Parts (ParoQuant): each weight part against its own activation copy, with bounds off any
         * kernel's tile, two and three parts, at every band. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_nt_m64", 7, 400, 640, 0, true, {}, 144);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_nt_m64", 40, 400, 640, 0, true, {}, 144, 288);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 8, 400, 2560, 0, true, {}, 128);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 40, 512, 2560, 0, true, {}, 128, 384);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 4, 34816, 5120, 0, true, {}, 17408);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 400, 640, 2, true, {}, 192);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 400, 640, 4, true, {}, 96, 304);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 2100, 512, 2560, 0, true, {}, 256);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_tiled", 300, 34816, 5120, 0, true, {}, 17408);
        /* Above 64 rows, every branch of the split table: 110+ column blocks, 48+, 8 or fewer
         * (with K not a multiple of 128, so the slab falls back to 64), a deep K on both sides of
         * M=80, and the plain case; then every split and slab forced. */
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 96, 14336, 640, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 128, 6144, 1152, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 100, 1024, 576, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 72, 5120, 6144, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 120, 5120, 6144, 0, true);
        mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 112, 5120, 2560, 0, false);
        for (int ks : { 1, 2, 4, 8, 16 })
            for (int bk : { 64, 128 })
                mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 96, 400, 2560, 0, true,
                                { PI("ks", ks), PI("bk", bk), PI("nt", 1) });
        for (int ks : { 1, 2, 4, 8, 16 })
            for (int bk : { 64, 128 })
                mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 24, 400, 2560, 0, true,
                                { PI("ks", ks), PI("bk", bk), PI("nt", ks == 2 ? 2 : 1) });
        /* And the one-fragment form (DTM = 1: register prefetch, 16 B loads) at every split, slab
         * and load hint, on both activation grids, N past a column block. */
        for (int ks : { 1, 2, 4, 8, 16 })
            for (int bk : { 64, 128 })
                for (int nt : { 1, 2 }) {
                    mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 8, 400, 2560, 0, true,
                                    { PI("ks", ks), PI("bk", bk), PI("nt", nt) });
                    mxfp4a8_numeric(pl, "gemm_mxfp4a8_decode", 13, 272, 1152, 0, false,
                                    { PI("ks", ks), PI("bk", bk), PI("nt", nt) });
                }
        gemm_case(pl, "gemm_w4a8_tiled",     "w4a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K/2, 256*(K/128)*4);
        gemm_case(pl, "gemm_w8a8_tiled",     "w8a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K,   256*(K/128)*2);
        gemm_case(pl, "gemm_w4a8_prefill",   "w4a8",    256, 256, K, RAD_I8, 1, 0, 0, 256*K/2, 256*(K/128)*4);

        /* Block-scaled fp8. The matvec runs at every point of its space (each `rb` is a different
         * instantiation of the K loop). The shapes give the narrow W8A8 row K a multiple of 256
         * and the tiled one N a multiple of 128. */
        for (const TunePoint& tp : tune_points(find(pl, "gemm_fp8a16_nt_m1"),
                                               std::vector<RadParam>{ PI("M", 1), PI("N", N),
                                                                      PI("K", K), PI("group", 128),
                                                                      PS("dtype", "fp8a16") }))
            fp8_gemm_case(pl, "gemm_fp8a16_nt_m1",  "fp8a16", 1,  N, K, 0, &tp);
        fp8_gemm_case(pl, "gemm_fp8a8_nt_m16",      "fp8a8",  8,  N, K, 1);
        fp8_gemm_case(pl, "gemm_fp8a8_nt_m16",      "fp8a8",  16, N, K, 1);
        fp8_gemm_case(pl, "gemm_fp8a8_tiled",       "fp8a8",  256, 256, K, 1);
    }

    /* --- the fp8 activation quantiser and the device sampler --- */
    {
        if (sel("quant_act_fp8")) {
            const RadKernelInfo* r = find(pl, "quant_act_fp8");
            if (!r) ok("quant_act_fp8", false, "row absent");
            else {
                const int64_t M = 32, K = 512;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16,    { M, K }),
                    T(nullptr, RAD_F8E4M3,  { M, K }),
                    T(nullptr, RAD_F32,     { M, K / 128 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("sample_chain_f32")) {
            const RadKernelInfo* r = find(pl, "sample_chain_f32");
            if (!r) ok("sample_chain_f32", false, "row absent");
            else {
                /* Not a power-of-two vocabulary, which could hide an off-by-one in the stride. */
                const int64_t M = 4, V = 1000;
                /* Two uniforms per position in [0,1). Zeros (as dalloc returns) would always select
                 * the first kept token and pass with the walk broken. */
                std::vector<float> uh((size_t)(2 * M));
                for (size_t i = 0; i < uh.size(); ++i) uh[i] = 0.11f + 0.17f * (float)i;
                RadTensor tu = T(nullptr, RAD_F32, { 2 * M });
                tu.data = alloc_for(tu);
                if (tu.data) st_memcpy(tu.data, uh.data(), uh.size() * 4, hipMemcpyHostToDevice);
                /* `draw` is three 32-bit words per position (a struct), sized in words. */
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_F32, { M, V }),
                    T(nullptr, RAD_U32, { M }),
                    tu,
                    T(nullptr, RAD_U32, { M, 3 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n_vocab", V), PF("temp", 1.0), PF("p", 0.95),
                    PF("min_p", 0.0), PI("k", 40),
                };
                run_case(r, ts, ps);
                /* Temperature 0 must give the argmax with greedy's lowest-id tie-break; a zeroed
                 * logits plane is the all-ties input that checks it. */
                std::vector<RadParam> ps0 = {
                    PI("M", M), PI("n_vocab", V), PF("temp", 0.0), PF("p", 1.0),
                    PF("min_p", 0.0), PI("k", 0),
                };
                run_case(r, ts, ps0);
                /* And with no draft proposal, as at the last position of a verify block. */
                absent(ts[1]);
                run_case(r, ts, ps);
            }
        }
    }

    /* --- quantiser and the fusions --- */
    {
        const int64_t M = 32, K = 512;
        if (sel("quant_act_i8")) {
            const RadKernelInfo* r = find(pl, "quant_act_i8");
            if (!r) ok("quant_act_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_I8, { M, K }),
                    T(nullptr, RAD_F32,  { M }),    T(nullptr, RAD_F32, { (K/128) * 32 }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
                absent(ts[3]);                           /* the plain form, without the row sums */
                run_case(r, ts, ps);
            }
        }
        if (sel("had_quant_act_i8")) {
            const RadKernelInfo* r = find(pl, "had_quant_act_i8");
            if (!r) ok("had_quant_act_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_I8, { M, K }),
                    T(nullptr, RAD_F32,  { M }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("rmsnorm_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "rmsnorm_had_quant_i8");
            if (!r) ok("rmsnorm_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { M, K }),
                    T(nullptr, RAD_BF16, { K }),    T(nullptr, RAD_I8,   { M, K }),
                    T(nullptr, RAD_F32,  { M }),    T(nullptr, RAD_BF16, { M, K }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PF("eps", 1e-6), PI("group", 128),
                    PS("dtype", "bf16"), PF("wadd", 0.0),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gated_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "gated_had_quant_i8");
            if (!r) ok("gated_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, 2 * K }), T(nullptr, RAD_BF16, { 0 }),
                    T(nullptr, RAD_I8,   { M, K }),     T(nullptr, RAD_F32,  { M }),
                };
                absent(ts[1]);                           /* the packed gate_up form */
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("group", 128), PS("dtype", "bf16"), PS("act", "silu"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("gdn_gated_norm_had_quant_i8")) {
            const RadKernelInfo* r = find(pl, "gdn_gated_norm_had_quant_i8");
            if (!r) ok("gdn_gated_norm_had_quant_i8", false, "row absent");
            else {
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, K }), T(nullptr, RAD_BF16, { M, K }),
                    T(nullptr, RAD_F32,  { 128 }),  T(nullptr, RAD_I8,   { M, K }),
                    T(nullptr, RAD_F32,  { M }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("n", K), PI("head_dim", 128), PI("group", 128),
                    PF("eps", 1e-6), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        if (sel("qk_norm_rope_gate")) {
            const RadKernelInfo* r = find(pl, "qk_norm_rope_gate");
            if (!r) ok("qk_norm_rope_gate", false, "row absent");
            else {
                const int64_t nq = 4, nkv = 2, hd = 128, rot = 64;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { M, nq * hd * 2 }),   /* [q|gate] per head */
                    T(nullptr, RAD_BF16, { M, nkv * hd }),
                    T(nullptr, RAD_BF16, { 4096, rot }),
                    T(nullptr, RAD_I32,  { M }),
                    T(nullptr, RAD_BF16, { hd }),
                    T(nullptr, RAD_BF16, { hd }),
                    T(nullptr, RAD_BF16, { M, nq * hd }),
                    T(nullptr, RAD_BF16, { M, nkv * hd }),
                    T(nullptr, RAD_BF16, { M, nq * hd }),
                };
                std::vector<RadParam> ps = {
                    PI("M", M), PI("head_dim", hd), PI("n_head", nq), PI("n_head_kv", nkv),
                    PI("rot", rot), PS("q_dtype", "bf16"), PF("eps", 1e-6),
                };
                run_case(r, ts, ps);
            }
        }
        qk_prologue_case(pl, RAD_BF16);
        qk_prologue_case(pl, RAD_F32);
        qk_prologue_case(pl, RAD_F32, true);
        qk_prologue_case(pl, RAD_F32, true, 128);
        qk_kv_fold_case(pl, false);
        qk_kv_fold_case(pl, true);
        qsa_qprep_case(pl);
        topk_scatter_case(pl);
        router_gemm_fused_case(pl);
        scatter_protect_case(pl);
        moe_gather_case(pl);
        if (sel("rowtopk_bf16")) {
            const RadKernelInfo* r = find(pl, "rowtopk_bf16");
            if (!r) ok("rowtopk_bf16", false, "row absent");
            else {
                const int64_t Mr = 32, Nr = 2048, R = 8;
                std::vector<RadTensor> ts = {
                    T(nullptr, RAD_BF16, { Mr, Nr }), T(nullptr, RAD_I32, { Mr, R }),
                    T(nullptr, RAD_F32,  { Mr, R }),
                };
                std::vector<RadParam> ps = {
                    PI("M", Mr), PI("N", Nr), PI("R", R), PS("dtype", "bf16"),
                };
                run_case(r, ts, ps);
            }
        }
        rowtopk_shard_case(pl);
    }

    /* dequant_w4_bf16's case is the w4 round trip, named for the row so the coverage check sees
     * it. */
    if (sel("dequant_w4_bf16")) {
        if (!g_case) std::printf("\nweight layout, against libr4d's own inverse\n");
        w4_roundtrip(pl);
    }

    if (sel("sample_argmax_f32")) {
        if (!g_case) std::printf("\nthe greedy sampler, against a host scan\n");
        sample_argmax_case(pl);
    }

    if (sel("qsa_work_bf16")) {
        if (!g_case) std::printf("\nthe QSA work list, against a host walk of the batch\n");
        qsa_work_case(pl);
    }

    if (sel("qsa_tail_store_bf16")) {
        if (!g_case) std::printf("\nthe QSA raw-key tail, against a serial loop\n");
        qsa_tail_store_case(pl);
    }

    if (sel("qsa_select_i32")) {
        if (!g_case) std::printf("\nthe QSA selection, against a host sort\n");
        qsa_select_case(pl);
        qsa_select_prefill_case(pl);
        qsa_select_wide_case(pl);
        qsa_select_split_one(pl, "qsa_select split across chunks vs the same launch unsplit",
                             1, 4);
        qsa_select_split_one(pl, "qsa_select split, two sequences of four draft rows", 2, 4);
        qsa_select_few_live_case(pl);
    }

    if (sel("qsa_score_bf16")) {
        if (!g_case) std::printf("\nthe QSA block score, against the formula it implements\n");
        /* Every form of the row: the launcher picks by shape, and another model's shape may get
         * a form the rule does not pick here. */
        for (int f = pl.pin_qsa ? 1 : 0; f <= (pl.pin_qsa ? 2 : 0); ++f) {
            if (pl.pin_qsa) {
                pl.pin_qsa(f);
                std::printf("      qsa_score, %s form\n", f == 1 ? "scalar" : "matrix");
            }
            qsa_score_case(pl);
            qsa_score_rows_case(pl);
        }
        if (pl.pin_qsa) pl.pin_qsa(0);
    }

    if (sel("qsa_block_key_bf16")) {
        if (!g_case) std::printf("\nthe QSA compressed key, against the formula it implements\n");
        qsa_block_key_case(pl);
        qsa_block_key_yarn_case(pl);
    }

    if (sel("logit_softcap_bf16")) {
        if (!g_case) std::printf("\nthe final-logit softcap, against the formula it implements\n");
        logit_softcap_case(pl);
    }

    if (sel("rmsnorm_p_bf16")) {
        if (!g_case) std::printf("\nthe weightless RMS norm, against the formula it implements\n");
        rmsnorm_p_case(pl);
    }

    if (sel("gelu_mul_bf16")) {
        if (!g_case) std::printf("\nthe gelu-gated product, against the formula it implements\n");
        gelu_mul_case(pl);
    }

    if (sel("scale_bf16")) {
        if (!g_case) std::printf("\nthe per-tensor scalar scale, against the multiply it implements\n");
        scale_case(pl);
    }

    if (sel("rope_bf16")) {
        if (!g_case) std::printf("\nthe rotary ladder, against an f32 mirror of the kernel at each width\n");
        rope_ladder_case(pl);
    }

    /* Each row needs a case under its own name: a child runs per row and `sel` matches only that
     * name, so a case nested in another row's block would never run. */
    if (sel("moe_gemm_w4a8")) {
        if (!g_case) std::printf("\nthe 4-bit grouped MoE GEMM, against a host decode\n");
        /* One geometry; a gate/up pair whose odd experts are half as wide; a down projection whose
         * odd experts are half as long; and a decode-shaped routing (a few runs over hundreds of
         * mostly empty experts) for the decode grid's run search. */
        static const MoeW4Shape kShapes[] = {
            { "one class",   64, 256,  64, 256, 1, 5 },
            { "gate/up 2:1", 256, 256, 128, 256, 2, 5 },
            { "down 2:1",    64, 256,  64, 128, 1, 5 },
            { "300 experts, gate/up 2:1", 256, 128, 128, 128, 2, 300 },
            /* Three passes of 256 along K, so the two-block pass also runs its one-ahead
             * prefetch. */
            { "40 experts, gate/up 2:1, K 768", 256, 768, 128, 768, 2, 40 },
            /* Past 512 experts: the decode form's chunked run search. Runs land on both sides of
             * 512, and K 256 lets the two-block pass run on that search too. */
            { "640 experts, gate/up 2:1, K 256", 256, 256, 128, 256, 2, 640 },
            /* The down projection as a rank of four runs it: sorted operand, one part, classes at
             * 256 and 128 of K, width in whole 256-column tiles (the aligned form's down tile).
             * Then with the odd class half as wide (that tile is padding for odd experts), and at
             * 512 experts, whose n_expert + 1 counts the run map scans in three 256-count tiles. */
            { "down, sorted rows, 512 x K 256:128", 512, 256, 512, 128, 1, 10, true },
            { "down, sorted rows, 512:256 x K 256:128", 512, 256, 256, 128, 1, 10, true },
            { "512 experts, down, sorted rows, 256 x K 128", 256, 128, 256, 128, 1, 512, true },
        };
        for (const MoeW4Shape& sh : kShapes) {
            moe_w4_gemm(pl, 24, sh);
            moe_w4_gemm(pl, 200, sh);
            moe_w4_gemm(pl, 600, sh);
            for (const int sub : { 2, 4 })
                for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                    moe_w4_gemm(pl, t, sh, sub);
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                moe_w4_gemm(pl, t, sh, 2, true);
        }
    }

    if (sel("moe_gemm_w4a8_mod4")) {
        if (!g_case) std::printf("\nthe 4-bit grouped MoE GEMM over four classes, against its "
                                 "two-class form and a host decode\n");
        /* The same shapes read as four classes (the even table's halves are classes 0 and 2, the
         * odd one's 1 and 3), against the two-class launch byte for byte. */
        static const MoeW4Shape kShapes4[] = {
            { "one class",   64, 256,  64, 256, 1, 5 },
            { "gate/up 2:1", 256, 256, 128, 256, 2, 10 },
            { "down 2:1",    64, 256,  64, 128, 1, 10 },
            { "300 experts, gate/up 2:1", 256, 128, 128, 128, 2, 300 },
            { "40 experts, gate/up 2:1, K 768", 256, 768, 128, 768, 2, 40 },
            { "640 experts, gate/up 2:1, K 256", 256, 256, 128, 256, 2, 640 },
            { "down, sorted rows, 512 x K 256:128", 512, 256, 512, 128, 1, 10, true },
            { "40 experts, down, sorted rows, 256 x K 128", 256, 128, 256, 128, 1, 40, true },
            { "down, sorted rows, 512:256 x K 256:128", 512, 256, 256, 128, 1, 10, true },
            { "512 experts, down, sorted rows, 256 x K 128", 256, 128, 256, 128, 1, 512, true },
        };
        for (const MoeW4Shape& sh : kShapes4)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 }) {
                moe_w4_gemm(pl, t, sh, 0, false, true);
                moe_w4_gemm(pl, t, sh, 2, false, true);
                moe_w4_gemm(pl, t, sh, 4, false, true);
                moe_w4_gemm(pl, t, sh, 2, true, true);
            }
        /* And every class at its own geometry, one of classes 1 to 3 the only wide one. */
        for (const MoeQ4Shape& sh : kMoeQ4Shapes)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                for (const char* dt : { "w4a8", "w4nl64a8h", "w4nl32a8h", "w5nl64a8h" })
                    moe_q4_gemm(pl, t, sh, dt);
        /* The sorted operand at decode, its own kernel against the (tile, run) form: one sequence
         * at depth 3 (T 40), then five, sixteen and nineteen, either side of both of the rule's
         * column-tile thresholds. */
        moe_down_dec_case(pl, kMoeDnShapes[0], "w4nl64a8h", { 40, 200, 640, 760 });
        for (const char* dt : { "w4a8", "w4nl64a8h", "w4nl32a8h", "w5nl64a8h" })
            moe_down_dec_case(pl, kMoeDnShapes[1], dt, { 40, 96 });
    }

    if (sel("moe_gemm_fp8a8_mod4")) {
        if (!g_case) std::printf("\nthe block-fp8 grouped MoE GEMM over four classes, against a "
                                 "host decode\n");
        for (const MoeQ4Shape& sh : kMoeQ4Shapes)
            for (const int64_t t : { (int64_t)24, (int64_t)200, (int64_t)600 })
                moe_q4_gemm(pl, t, sh, "fp8a8");
        moe_down_dec_case(pl, kMoeDnShapes[1], "fp8a8", { 40, 96 });
    }

    if (sel("hc_read_e4m3")) {
        if (!g_case) std::printf("\nthe gated residual's read at the prefill tile, every form "
                                 "against the decode tiles'\n");
        /* 700 and 1000 put PF2's K chunks across a stream boundary and row tiles past the last
         * row; 4096 is the one-chunk epilogue. */
        for (int64_t m : { 65, 300, 700, 1000, 2048, 4096 }) hc_prefill(pl, m, 0, false);
        /* 1 to 7 rows each hit a different up-product instantiation below eight. */
        for (int64_t m : { 1, 2, 3, 4, 5, 6, 7, 33, 65, 300, 2048 }) {
            hc_read_i8(pl, m, false, 0);
            hc_read_i8(pl, m, true, 0);
        }
        if (pl.pin_hc) {
            for (int64_t m : { 65, 2048 })
                for (int form : { 1, 2 }) hc_read_i8(pl, m, true, form);
            for (int64_t m : { 700, 2048 }) hc_read_i8(pl, m, true, 3);
        }
    }
    if (sel("hc_read_bf16")) {
        if (!g_case) std::printf("\nthe gated residual's read over bf16 mixing matrices at the "
                                 "prefill tile, every form against the decode tiles'\n");
        for (int64_t m : { 300, 700, 2048 }) hc_prefill(pl, m, 0, false, false);
    }

    if (sel("moe_gemm_w4a8")) {
        if (!g_case) std::printf("\nthe arranged plane, against the stated format and its "
                                 "inverse\n");
        moe_w4_unlayout(pl);
    }

    if (sel("had_quant_act_fp8")) {
        if (!g_case) std::printf("\nthe ROTATED 4-bit expert format, against the f32 it "
                                 "approximates\n");
        moe_w4_rot(pl);
    }

    if (g_case && !std::strcmp(g_case, "w5a16_perf")) {
        w5a16_perf(pl, 4, 64, 17408, 5120);
        w5a16_perf(pl, 4, 64, 5120, 17408);
        w5a16_perf(pl, 4, 16, 17408, 5120);
    }
    for (int bits : { 4, 5 }) {
        const char* row = bits == 4 ? "gemm_w4a8i8_tiled" : "gemm_w5a8i8_tiled";
        if (!sel(row)) continue;
        if (!g_case) std::printf("\nParoQuant int%d prefill GEMM (int8 WMMA), against the planes' meaning\n", bits);
        wpa8i8_numeric(pl, bits, 300, 400, 640);
        wpa8i8_numeric(pl, bits, 70, 272, 1280);
        wpa8i8_numeric(pl, bits, 300, 400, 640, 192);
        wpa8i8_numeric(pl, bits, 600, 512, 2560, 128, 384);
        wpa8i8_numeric(pl, bits, 2100, 34816, 5120, 17408);
        wpa8i8_numeric(pl, bits, 2048, 17408, 5120);
        wpa8i8_numeric(pl, bits, 2048, 5120, 17408);
    }

    if (sel("gemm_w5a16_nt_m64")) {
        if (!g_case) std::printf("\nParoQuant's 5-bit GEMM, against the planes' meaning\n");
        w5a16_numeric(pl, 8, 272, 640);
        w5a16_numeric(pl, 64, 400, 2560);
        w5a16_numeric(pl, 5, 400, 640, 144);
        w5a16_numeric(pl, 40, 512, 2560, 128, 384);
        w5a16_numeric(pl, 300, 400, 640, 192);
        w5a16_numeric(pl, 4, 34816, 5120, 17408);
        w5a16_numeric(pl, 150, 5120, 17408);
    }

    if (sel("givens_rotate")) {
        if (!g_case) std::printf("\nParoQuant's rotation to f16, against an f64 rotation\n");
        givens_numeric(pl, 3, 5120, 1, 8, GvCase::Rotate);
        givens_numeric(pl, 5, 5120, 2, 8, GvCase::Rotate);
        givens_numeric(pl, 300, 6144, 2, 8, GvCase::Rotate);
    }

    if (sel("givens_dense_quant")) {
        if (!g_case) std::printf("\nParoQuant's rotation as dense per-group matrices, int8 and E4M3 output, against libm's rotation\n");
        for (const GvCase mode : { GvCase::DenseFp8, GvCase::DenseI8 }) {
            givens_numeric(pl, 1, 128, 1, 8, mode);
            givens_numeric(pl, 5, 5120, 2, 8, mode);
            givens_numeric(pl, 8, 5120, 2, 8, mode);
            givens_numeric(pl, 16, 5120, 1, 8, mode);
            givens_numeric(pl, 300, 6144, 2, 8, mode);
            givens_numeric(pl, 2048, 5120, 1, 8, mode);
            givens_numeric(pl, 2048, 5120, 2, 8, mode);
            givens_numeric(pl, 64, 17408, 1, 8, mode);
            givens_numeric(pl, 2048, 17408, 1, 8, mode);
            givens_numeric(pl, 7, 5120, 3, 2, mode);
        }
    }

    if (sel("gram_accum")) {
        if (!g_case) std::printf("\nthe GPTQ calibration accumulator, against an f64 host Gram\n");
        gram_accum_case(pl);
    }

    if (sel("gram_accum_bf16")) {
        if (!g_case) std::printf("\nthe calibration accumulator over a plain bf16 activation\n");
        gram_accum_bf16_case(pl, 200, 260);
        gram_accum_bf16_case(pl, 2049, 640);
    }

    if (sel("moe_gemm_bf16")) {
        if (!g_case) std::printf("\nthe bf16 grouped MoE GEMM, against an f64 host product\n");
        /* Qwen3.8-Flash-Next's and Qwen3.6-35B-A3B's whole-expert widths at a few experts each, and
         * expert counts exercising the run map and decode run search at small widths. A width
         * without the 128-column tile runs the 64-column one. */
        static const MoeBf16Shape kShapes[] = {
            { "Flash-Next gate_up 1280x2560", 1280, 2560,   4,  2, false },
            { "Flash-Next down 2560x640",     2560,  640,   4,  2, true  },
            { "35B gate_up 1024x2048",        1024, 2048,   4,  2, false },
            { "35B down 2048x512",            2048,  512,   4,  2, true  },
            { "256 experts top 8",             128,  128, 256,  8, false },
            { "512 experts top 10, sorted",     64,   64, 512, 10, true  },
            { "64-column tile",                192,  192,   6,  2, false },
        };
        for (const MoeBf16Shape& sh : kShapes) {
            /* One decode-band step and one or two prefill passes. */
            const int64_t dec = std::max<int64_t>(1, (3 * sh.ne) / (2 * sh.top_k));
            moe_gemm_bf16_case(pl, dec, sh);
            moe_gemm_bf16_case(pl, 100, sh);
            if (sh.ne >= 256) moe_gemm_bf16_case(pl, 300, sh);
        }
        moe_gemm_bf16_case(pl, 100, kShapes[1], true);
        /* The trailing segment alone, at both row orders and both bands. */
        for (const int i : { 0, 4, 5 }) {
            const MoeBf16Shape& sh = kShapes[i];
            const int64_t dec = std::max<int64_t>(1, (3 * sh.ne) / (2 * sh.top_k));
            for (const int64_t m : { dec, (int64_t)100 }) moe_gemm_bf16_case(pl, m, sh, false, 2);
        }
    }

    if (sel("dequant_w4_bf16")) {
        /* N = 256 (the k and v projections) and 17408 (gate_up). narrow_mw depends on M, and
         * fp8_numeric's M list crosses its threshold (1 and 8 take MW = 4, 24-64 take MW = 8). */
        fp8_numeric(pl, 256, 512);
        fp8_numeric(pl, 17408, 512);
        fp8_numeric(pl, 5120, 5120);
        /* Many K tiles with split-K off: gate_up, 17408 x 5120, is twenty tiles with gx*mw over the
         * split threshold, so the prefetch fires nineteen times ((256,512) and (17408,512) have
         * two tiles; (5120,5120) splits). */
        fp8_numeric(pl, 17408, 5120);
        /* The remaining production K values: the GDN out-projection (K = v_dim = 1536 per rank)
         * and down_proj (K = n_ff = 8704). */
        fp8_numeric(pl, 5120, 1536);
        fp8_numeric(pl, 5120, 8704);
        /* A shared expert split by whole scale blocks: Qwen3.8-Flash-Next's 640 columns go 384 /
         * 256, so the down projection runs at K = 384 (a multiple of 128, not 256: the narrow
         * tile's 128-wide leg) and K = 256, and gate/up at N = 768 and 512 over K = 2560. */
        fp8_numeric(pl, 2560, 384);
        fp8_numeric(pl, 2560, 256);
        fp8_numeric(pl, 768, 2560);
        fp8_numeric(pl, 512, 2560);
        /* The lm_head. 5120 rows stand in for 124160 (rows are walked with a grid-stride loop);
         * M spans the exact instantiations, the rounded-up ones and the cap. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)13, (int64_t)32, (int64_t)33,
                                 (int64_t)64 })
            fp8_logits_numeric(pl, 5120, 5120, M);
        /* And K = 2560, the padded row: its stored width 2600 is not a multiple of 16, so
         * r4d_fp8_lm.h rounds it to 2608 (5120's 5200 needs no pad). M = 1 takes the matvec, M = 8
         * the WMMA tile, which index the row differently. */
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)33 })
            fp8_logits_numeric(pl, 2560, 2560, M);
    }
    /* The int8 head over the same stored row: both forms, and the padded K = 2560 row. */
    if (sel("logits_gemm_i8") || sel("logit_rerank_i8")) {
        for (const int64_t M : { (int64_t)1, (int64_t)8, (int64_t)33, (int64_t)64 })
            fp8_logits_numeric(pl, 5120, 5120, M, true);
        for (const int64_t M : { (int64_t)1, (int64_t)8 })
            fp8_logits_numeric(pl, 2560, 2560, M, true);
    }

    /* The fused quantisers are checked against the kernels they replace, gated on their own
     * rows. */
    if (sel("rmsnorm_quant_fp8") || sel("gated_quant_fp8") || sel("gate_quant_fp8")) {
        if (!g_case) std::printf("\nthe fp8 fusions, against the pair each one replaces\n");
        fused_quant_fp8_numeric(pl);
    }
    if (sel("gelu_mul_quant_i8")) {
        if (!g_case) std::printf("\ngelu_mul_quant_i8, against the pair it replaces\n");
        gelu_mul_quant_i8_numeric(pl);
    }
    if (sel("qkv_norm_rope_bf16")) {
        if (!g_case) std::printf("\nqkv_norm_rope, against the five ops it replaces\n");
        qkv_norm_rope_numeric(pl);
    }
    if (sel("rmsnorm_quant_i8")) {
        if (!g_case) std::printf("\nrmsnorm_quant_i8, against the chain it replaces\n");
        rmsnorm_quant_i8_numeric(pl);
    }

    /* The MTP head's fc is K=10240; the DFlash2 drafter's is 25600, the largest K any shape asks
     * these kernels for. Both rows run: an fc declared over M 1..64 resolves to _m16 for small
     * issues and _m64 for wide ones, separate kernels with separate split laws. */
    if (sel("gemm_bf16_nt_m16") || sel("gemm_bf16_nt_m64")) {
        if (!g_case) std::printf("\nthe skinny bf16 GEMMs, against a host reference\n");
        for (const char* row : { "gemm_bf16_nt_m16", "gemm_bf16_nt_m64" }) {
            bf16_gemm_numeric(pl, row, 5, 32, 10240);
            bf16_gemm_numeric(pl, row, 5, 32, 25600);
            /* The drafter's fc exactly: the weight offset row * K is large only for the pair. */
            bf16_gemm_numeric(pl, row, 5, 5120, 10240);
            bf16_gemm_numeric(pl, row, 5, 5120, 25600);
            bf16_gemm_numeric(pl, row, 16, 5120, 25600);
        }
        bf16_gemm_numeric(pl, "gemm_bf16_nt_m64", 64, 5120, 25600);
    }
    if (sel("gemm_bf16_nt_tiled")) {
        if (!g_case) std::printf("\nthe tiled bf16 GEMM, against the skinny one\n");
        bf16_tiled_identity(pl, 2048, 512, 2560);     /* the MoE router: four slices */
        bf16_tiled_identity(pl, 1000, 2560, 2560);    /* the PLE value projection: one slice */
        bf16_tiled_identity(pl, 333, 1040, 2560);     /* ragged M and N, two slices */
        bf16_tiled_identity(pl, 77, 128, 64);         /* one slice per k step, the band's floor */
        bf16_gemm_numeric(pl, "gemm_bf16_nt_tiled", 300, 1040, 2560);
    }

    /* Every collective has a host twin (r4d_rows.cpp): the same kernel with its mailbox in pinned
     * host memory. A case is the pair's, so the twin is held to exactly the original's checks. */
    /* The 2-rank pair keeps its own harness: its flag layout and its block ceiling are not the
     * wide family's, and TP2 is what the fleet runs today. Everything at three and four goes
     * through the width-parameterized functions, which take the world from the base's own name and
     * skip themselves, named for the width they need, when the box has fewer cards. */
    for (const char* base : { "ar_oneshot_2rank_exact", "ar_gather_2rank", "ar_oneshot_3rank_exact",
                              "ar_twoshot_3rank_exact", "ar_twoshot_3rank_ti8", "ar_gather_3rank",
                              "ar_oneshot_4rank_exact", "ar_twoshot_4rank_exact",
                              "ar_twoshot_4rank_ti8", "ar_gather_4rank" }) {
        for (const std::string& n : { std::string(base), std::string(base) + "_host" }) {
            if (!sel(n.c_str())) continue;
            const int world = std::strstr(base, "4rank") ? 4
                            : std::strstr(base, "3rank") ? 3 : 2;
            if (!g_case) {
                if (world == 2) std::printf("\ncross-rank\n");
                else std::printf("\ncross-rank, %d\n", world);
            }
            std::printf("  %s\n", n.c_str());
            if (world == 2) {
                if (!std::strcmp(base, "ar_oneshot_2rank_exact")) {
                    ar_two_rank(pl, devs, n.c_str());
                    ar_two_rank_sweep(pl, devs, n.c_str());
                } else {
                    gather_two_rank(pl, devs, n.c_str());
                }
            /* Its own rows, so the sweep names them (a row the sweep never mentions is reported as
             * a coverage hole) and so `--case ar_oneshot_4rank_exact` reproduces one alone. */
            } else if (std::strstr(base, "gather")) {
                gather_wide_rank(pl, devs, n.c_str(), world);
            } else if (std::strstr(base, "ti8")) {
                ar_wide_ti8(pl, devs, n.c_str(), world);
            } else if (std::strstr(base, "twoshot")) {
                ar_wide_pattern(pl, devs, n.c_str(), world);
            } else {
                ar_wide_rank(pl, devs, n.c_str(), world);
                /* The one-shot through the same per-element oracle: both rows of the op agree
                 * with it, so they agree with each other at every size either could be given. */
                ar_wide_pattern(pl, devs, n.c_str(), world);
            }
        }
    }
    /* The fused write rows at both widths and both gathers. The `_host` twins take the same
     * calls; they differ only in where the mailbox lives. */
    for (const char* base : { "ar_hc_write", "ar_gather_hc_write",
                              "ar_hc_write_4rank", "ar_gather_hc_write_4rank" }) {
        for (const std::string& n : { std::string(base), std::string(base) + "_host" }) {
            if (!sel(n.c_str())) continue;
            if (!g_case) std::printf("\ncross-rank, fused with the gated residual write\n");
            std::printf("  %s\n", n.c_str());
            ar_hc_write_case(pl, devs, n.c_str(), std::strstr(base, "4rank") ? 4 : 2,
                             std::strstr(base, "gather") != nullptr);
        }
    }
    /* The 8-rank rows cannot be exercised on a four-card box. Reported as skipped rather
     * than quietly passing: a row that has never been launched is not a row that works. */
    auto skip_pair = [](const char* base, const char* why) {
        for (const std::string& n : { std::string(base), std::string(base) + "_host" })
            if (sel(n.c_str())) skip(n.c_str(), why);
    };
    /* Only the 8-rank rows now: the four-rank ones are dispatched above and skip themselves with
     * the width they need. */
    for (const char* n : { "ar_oneshot_8rank_exact", "ar_twoshot_8rank_exact", "ar_gather_8rank" })
        skip_pair(n, "needs 8 cards");
    skip_pair("ar_oneshot_2rank_wht6", "lossy wire: needs a tolerance harness, not a smoke test");
    skip_pair("ar_ln_had_quant_i8", "two-rank fusion: needs the same harness as the wht6 wire");
    /* Also skipped: the norm half is rmsnorm_quant_fp8's device function, checked there, and the
     * handshake and reduced sum need a peer rank. */
    skip_pair("ar_rmsnorm_quant_fp8", "two-rank fusion: needs a peer rank");
    skip_pair("ar_rmsnorm_quant_i8", "two-rank fusion: needs a peer rank");

    dfree_all();
}

/* ------------------------------------------------------------------ the isolating driver
 *
 * The row table is the case list. Every case is named for a row, so a row no case mentions comes
 * back from its child as "no smoke case".
 */

/* The signal's number and name (a GPU page fault is not always SIGABRT). The fault line itself
 * is on stderr, unbuffered, just above. */
static std::string signame(int sig) {
    const char* n = nullptr;
    switch (sig) {
        case SIGABRT: n = "SIGABRT"; break;
        case SIGSEGV: n = "SIGSEGV"; break;
        case SIGBUS:  n = "SIGBUS";  break;
        case SIGILL:  n = "SIGILL";  break;
        case SIGFPE:  n = "SIGFPE";  break;
        case SIGKILL: n = "SIGKILL"; break;
        default:      n = "signal";  break;
    }
    return std::string(n) + " (" + std::to_string(sig) + ")";
}

static void run_isolated(const Plugin& pl, const char* exe, const char* plugin) {
    /* The device inventory is printed once, by the parent (an integrated part can appear beside
     * the discrete cards). */
    int nd = 0;
    if (hipGetDeviceCount(&nd) == hipSuccess) {
        for (int d = 0; d < nd; ++d) {
            hipDeviceProp_t p;
            if (hipGetDeviceProperties(&p, d) == hipSuccess)
                std::printf("  device %d: %s\n", d, p.gcnArchName);
        }
    }

    std::printf("\ndevice smoke tests (one child process per row)\n");
    std::fflush(stdout);

    for (int k = 0; k < pl.kernel_count(); ++k) {
        const std::string name = pl.kernel_at(k)->name;
        char* const argv[] = { const_cast<char*>(exe), const_cast<char*>(plugin),
                               const_cast<char*>("--case"), const_cast<char*>(name.c_str()),
                               nullptr };
        pid_t pid = 0;
        /* posix_spawn, not fork: the HIP runtime's threads are up, and only exec resets them. */
        if (posix_spawn(&pid, exe, nullptr, nullptr, argv, environ) != 0) {
            ok(name.c_str(), false, "posix_spawn failed");
            continue;
        }
        int st = 0;
        if (waitpid(pid, &st, 0) < 0) { ok(name.c_str(), false, "waitpid failed"); continue; }
        std::fflush(stdout);

        if (WIFSIGNALED(st)) {
            ok(name.c_str(), false, std::string("child died: ") + signame(WTERMSIG(st)));
            continue;
        }
        if (!WIFEXITED(st)) { ok(name.c_str(), false, "child did not exit normally"); continue; }
        switch (WEXITSTATUS(st)) {
            case CH_OK:   break;                       /* the child printed its ok line */
            case CH_FAIL: ++g_fail; break;             /* and its FAIL line, with detail  */
            case CH_SKIP: ++g_skip; break;
            default:
                ok(name.c_str(), false,
                   "child exit " + std::to_string(WEXITSTATUS(st)));
        }
    }
}

#else   /* !RAD_HAVE_HIP */
static void device_phase(const Plugin&) { skip("device phase", "built without HIP"); }
static void run_isolated(const Plugin&, const char*, const char*) {
    skip("device phase", "built without HIP");
}
#endif

/* ------------------------------------------------------------------ main */

static void usage(void) {
    std::printf("usage: r4d_selftest <plugin.so> [--case <row>]\n"
                "       r4d_selftest <plugin.so> --real <file> <woff> <soff> <N> <K>\n"
                "  no --case: the catalogue, the static checks, then one child process per row\n"
                "  --case:    that row's smoke test alone, in this process -- the reproducer\n"
                "  --real:    the block-scaled fp8 matvec against a REAL checkpoint weight, at the\n"
                "             given byte offsets in a safetensors file, compared with a host\n"
                "             reference over the same bytes. An offset is 8 + <header length, a\n"
                "             u64 at byte 0> + the tensor's own data_offsets[0].\n"
                "  --perf N K M [iters]:  time the fp8 decode GEMMs at one shape and print the\n"
                "             weight bandwidth each reaches.\n"
                "  --perfbf16 N K M [iters]:  the bf16 skinny GEMM at every variant it has.\n"
                "  --perfw8 N K M [iters]:  the 8-bit trunk GEMMs -- block E4M3 W8A8 and the int8\n"
                "             rows -- at one shape, each with its TFLOP/s.\n"
                "  --perfw2 N K M [iters]:  the int2 DFlash2 draft head at every variant it has.\n"
                "  --perfhc M [iters]:  hc_read_e4m3 at Qwen3.8-Flash-Next's shape and M rows, every\n"
                "             prefill-tile form (PF2 from M=512) compared byte for byte and timed\n"
                "             as one op each;\n"
                "             run it under libr4d/isa/trace.sh for its launches.\n"
                "  --cold MIB:  with a --perf mode, time cold: read MIB through the caches before\n"
                "             every timed launch (256 empties this card's L2 and last level) and\n"
                "             time each launch alone, the median reported. Warm, a weight under\n"
                "             64 MiB is read from cache after the first launch; the engine reads\n"
                "             its weights cold.\n"
                "  --perfmoe M [iters] [experts] [top_k] [pooled%%]:  the routed-expert w4 GEMM,\n"
                "             both projections at rank 0's tp2 width split, over a skewed routing of\n"
                "             an M-token chunk (512 experts, top 10 by default); every form, each\n"
                "             compared with the first. pooled%% of the experts are read out of\n"
                "             pinned host memory through a pointer table, as the engine's are.\n"
                "  --perfmoe16 M [iters] [experts] [top_k] [distinct]:  the bf16 expert GEMM at\n"
                "             Qwen3.8-Flash-Next's whole-expert widths (256 experts, top 10 by\n"
                "             default), every form compared with the first; `distinct` weight\n"
                "             planes back the table, and fewer than the experts is a cache-resident\n"
                "             floor rather than the real traffic.\n"
                "  --perflora M K N R S PATTERN [iters]:  every lora_shrink row, then every\n"
                "             lora_expand row, at one shape; PATTERN lays the slots out: base, one,\n"
                "             half (even rows adapter 0, odd base), mix (row m on m %% S) or runs\n"
                "             (S contiguous runs). R4D_PERFLORA_A=fp8_e4m3 for a coded `a` (scale\n"
                "             every 128), R4D_PERFLORA_Y=f16|f32 for `y`. Each row's t and y are\n"
                "             compared with the first row's; --cold applies.\n"
                "  --perfqsa CTX [M] [iters]:  the QSA indexer at a real context length. It is\n"
                "             the only decode op that grows with context, and at 200K it is\n"
                "             about a third of the step. M is 1 + n_spec.\n"
                "             rad-tune's declared graph does not reach a gemm_nt_q row, so this\n"
                "             is the only way to sweep it. Safe to chase hard: the head only\n"
                "             proposes, so its output bits change acceptance, not text.\n");
}

int main(int argc, char** argv) {
    const char* path = nullptr;
    const char* real_file = nullptr;
    long long real_woff = 0, real_soff = 0, real_n = 0, real_k = 0;
    long long perf_n = 0, perf_k = 0, perf_m = 0;
    int perf_it = 50, perf_w2 = 0, perf_bf = 0, perf_w8 = 0;
    long long sop_m = 0, sop_k = 0;
    int sop_it = 2000;
    long long qsa_ctx = 0, qsa_m = 4;
    int qsa_it = 200;
    long long hc_m = 0;
    int hc_it = 20;
    long long moe_m = 0, moe_e = 512, moe_k = 10;
    int moe_pool = 0;
    int moe_it = 20;
    long long m16_m = 0, m16_e = 256, m16_k = 10, m16_d = 0;
    int m16_it = 20;
    long long lo_m = 0, lo_k = 0, lo_n = 0, lo_r = 0, lo_s = 0;
    const char* lo_pat = nullptr;
    int lo_it = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) {
            if (i + 5 >= argc) { usage(); return 2; }
            real_file = argv[++i];
            real_woff = std::strtoll(argv[++i], nullptr, 0);
            real_soff = std::strtoll(argv[++i], nullptr, 0);
            real_n    = std::strtoll(argv[++i], nullptr, 0);
            real_k    = std::strtoll(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perf")) {
            if (i + 3 >= argc) { usage(); return 2; }
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfbf16")) {
            /* The bf16 skinny GEMM at every variant (see bf16_perf). */
            if (i + 3 >= argc) { usage(); return 2; }
            perf_bf = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfw8")) {
            if (i + 3 >= argc) { usage(); return 2; }
            perf_w8 = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfw2")) {
            /* The int2 draft head at every variant (see w2_perf). */
            if (i + 3 >= argc) { usage(); return 2; }
            perf_w2 = 1;
            perf_n = std::strtoll(argv[++i], nullptr, 0);
            perf_k = std::strtoll(argv[++i], nullptr, 0);
            perf_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') perf_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfqsa")) {
            /* The indexer at a real context length (the only decode op that grows with it). */
            if (i + 1 >= argc) { usage(); return 2; }
            qsa_ctx = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) qsa_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) qsa_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--cold")) {
            if (i + 1 >= argc) { usage(); return 2; }
            g_cold = std::strtoll(argv[++i], nullptr, 0) << 20;
        } else if (!std::strcmp(argv[i], "--perfmoe")) {
            /* The routed-expert GEMM at a prefill chunk of M tokens, every prefill form. */
            if (i + 1 >= argc) { usage(); return 2; }
            moe_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_it = (int)std::strtol(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_e = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) moe_pool = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfmoe16")) {
            /* The bf16 expert GEMM at a prefill chunk of M tokens, every form. */
            if (i + 1 >= argc) { usage(); return 2; }
            m16_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_it = (int)std::strtol(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_e = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) m16_d = std::strtoll(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perflora")) {
            /* The runtime-LoRA pair at one shape and slot layout, every row of each op. */
            if (i + 6 >= argc) { usage(); return 2; }
            lo_m = std::strtoll(argv[++i], nullptr, 0);
            lo_k = std::strtoll(argv[++i], nullptr, 0);
            lo_n = std::strtoll(argv[++i], nullptr, 0);
            lo_r = std::strtoll(argv[++i], nullptr, 0);
            lo_s = std::strtoll(argv[++i], nullptr, 0);
            lo_pat = argv[++i];
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) lo_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfhc")) {
            /* The gated residual's read at M rows. */
            if (i + 1 >= argc) { usage(); return 2; }
            hc_m = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != 0x2d) hc_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--perfops")) {
            /* The elementwise ops at a decode shape: launch cost, not GEMM bandwidth. */
            if (i + 2 >= argc) { usage(); return 2; }
            sop_m = std::strtoll(argv[++i], nullptr, 0);
            sop_k = std::strtoll(argv[++i], nullptr, 0);
            if (i + 1 < argc && argv[i + 1][0] != '-') sop_it = (int)std::strtol(argv[++i], nullptr, 0);
        } else if (!std::strcmp(argv[i], "--case")) {
            if (i + 1 >= argc) { usage(); return 2; }
            g_case = argv[++i];
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage();
            return 0;
        } else if (!path) {
            path = argv[i];
        } else {
            usage();
            return 2;
        }
    }
    if (!path) path = "libr4d.so";

    Plugin pl;

    /* Timing modes run alone, without the catalogue. */
    if (hc_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        hc_prefill(pl, hc_m, hc_it, true);
        dfree_all();
#else
        skip("gated residual read timing", "built without HIP");
#endif
        return 0;
    }
    if (lo_pat) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        lora_perf(pl, lo_m, lo_k, lo_n, lo_r, lo_s, lo_pat, lo_it);
        dfree_all();
#else
        skip("runtime LoRA timing", "built without HIP");
#endif
        return 0;
    }
    if (moe_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        moe_perf(pl, moe_m, moe_it, moe_e, moe_k, moe_pool);
        dfree_all();
#else
        skip("routed-expert GEMM timing", "built without HIP");
#endif
        return 0;
    }
    if (m16_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        moe16_perf(pl, m16_m, m16_it, m16_e, m16_k, m16_d);
        dfree_all();
#else
        skip("bf16 expert GEMM timing", "built without HIP");
#endif
        return 0;
    }
    if (qsa_ctx) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        qsa_perf(pl, qsa_ctx, qsa_m, qsa_it);
        dfree_all();
#else
        skip("qsa indexer timing", "built without HIP");
#endif
        return 0;
    }
    if (sop_m) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        smallop_perf(pl, sop_m, sop_k, sop_it);
        dfree_all();
#else
        skip("small-op timing", "built without HIP");
#endif
        return 0;
    }

    if (perf_n) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        if (perf_bf)      bf16_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else if (perf_w8) w8_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else if (perf_w2) w2_perf(pl, perf_n, perf_k, perf_m, perf_it);
        else         fp8_perf(pl, perf_n, perf_k, perf_m, perf_it);
        dfree_all();
#else
        skip("fp8 decode GEMM timing", "built without HIP");
#endif
        return 0;
    }

    /* The real-checkpoint check runs alone, against one given tensor. */
    if (real_file) {
        if (!load(&pl, path)) return 1;
#ifdef RAD_HAVE_HIP
        fp8_real(pl, real_file, real_woff, real_soff, real_n, real_k);
        dfree_all();
#else
        skip("fp8 real-weight check", "built without HIP");
#endif
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail;
    }

    /* The child: one row, no catalogue or static checks, and an exit code for the parent, since
     * a faulting row may print nothing. */
    if (g_case) {
        if (!load(&pl, path)) return CH_FAIL;
        device_phase(pl);
        if (!g_matched) skip(g_case, "no smoke case");
        std::fflush(stdout);
        return g_fail ? CH_FAIL : (g_skip ? CH_SKIP : CH_OK);
    }

    std::printf("r4d_selftest  %s\n\n", path);
    if (!load(&pl, path)) {
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail ? g_fail : 1;
    }
    dump(pl);
    cross_check(pl);

    /* /proc/self/exe rather than argv[0], which the caller may set to anything. */
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
        ok("re-exec path (/proc/self/exe)", false, "readlink failed");
        std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
        return g_fail;
    }
    self[n] = '\0';
    run_isolated(pl, self, path);

    std::printf("\n%d failure(s), %d skipped\n", g_fail, g_skip);
    return g_fail;
}
