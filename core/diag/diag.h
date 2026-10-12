/* diag.h -- the startup report: what was configured, what it runs on, and what looks wrong.
 *
 * WHY ONE REPORT AND NOT THE LINES ALREADY IN THE LOG. Every decision startup makes is said
 * somewhere on the way up, in the order it was made, between a hundred other lines. A log a user
 * pastes into an issue is read by someone who has to find the line about the embedding, the one
 * about the VRAM split and the one about the card's link, and cannot find the line that was never
 * printed because the value was a default. The report puts all of it in one block at the end of
 * startup: every option the command line takes with the value in force and where it came from,
 * the machine as the kernel describes it, and a list of the things on it that cost speed or
 * correctness with what to do about each.
 *
 * THREE PARTS, each built from data rather than from a list kept beside the code:
 *
 *   CONFIGURATION  every row of the option table (core/config.cpp), so an option added there
 *                  appears here without anyone remembering to; a value startup resolved carries
 *                  the reason it was resolved that way (Decision).
 *   HARDWARE       read from /proc and /sys under a ROOT, so a test can hand the collector a
 *                  directory tree of its own and a container built FROM scratch -- no shell, no
 *                  lspci, no rocm-smi -- reads the same files the host does.
 *   FINDINGS       each a small function over the two above (checks.cpp), unit-testable without
 *                  a card.
 *
 * Unknown is a value. A field that could not be read is reported as unknown rather than left out:
 * a report that silently drops a line reads as a machine that does not have the thing.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rad {
struct Config;
namespace diag {

/* ------------------------------------------------------------------ decisions */
/* ONE VALUE STARTUP RESOLVED, and why. `key` is the option it resolves (`--placement`) when there
 * is one, so the configuration table can show it on that option's row, or a topic of its own
 * (`vram split`) for what no single option states. Recording the same key again replaces the
 * earlier answer: a later stage that revises a choice (the n-gram copy that could not be made)
 * is the answer in force. */
struct Decision {
    std::string key;
    std::string value;
    std::string why;
};

void decide(std::vector<Decision>* list, const std::string& key, const std::string& value,
            const std::string& why);

/* ------------------------------------------------------------------ the configuration table */
/* One option as the report shows it: the value in force, where it came from (`default`, `command
 * line`, an environment variable, or `auto` for a value startup derived), and the reason when
 * startup derived it. */
struct ConfigRow {
    std::string flag;
    std::string value;
    std::string source;
    std::string why;
};

/* Every option in the command line's table (core/config.cpp), in its order, with the value `c`
 * holds and its source from Config::given. Secrets are not values: --api-key is "set" or "not
 * set". */
std::vector<ConfigRow> config_rows(const Config& c);

/* The options the table holds, for a test that every one reaches the report. */
std::vector<std::string> config_option_names();

/* ------------------------------------------------------------------ hardware */
/* One PCIe link on the way from the root port to a card. Speeds are the kernel's strings
 * ("32.0 GT/s PCIe"), widths lanes; -1 / "" where the file is absent, as it is on a bridge
 * internal to a card's own switch on some boards. */
struct PciHop {
    std::string bdf;
    std::string cur_speed, max_speed;
    int         cur_width = -1, max_width = -1;
};

struct CardFacts {
    std::string bdf;                 /* 0000:03:00.0 */
    std::string vendor, pci_device;  /* 0x1002, 0x7551 */
    std::string driver;              /* the bound kernel driver, "" for none */
    std::string drm;                 /* card1, "" when the driver made no DRM node */
    std::string gfx;                 /* gfx1201, from the KFD topology; "" when unknown */
    int         simd = -1;           /* SIMDs the KFD topology reports */
    int         numa_node = -1;
    int64_t     vram_total = -1, vram_visible = -1;   /* the BAR the host and the peers see */
    int         busy_percent = -1;
    std::string runtime_status;      /* active | suspended, "" when unknown */
    std::string runtime_control;     /* auto | on */
    std::string iommu_type;          /* the card's IOMMU group: identity, DMA, DMA-FQ */
    std::vector<PciHop> chain;       /* root port first, the card's own link last */
    std::vector<std::string> dpm_pcie;   /* the SMU's link levels, as listed */
    int         dpm_pcie_cur = -1;   /* the level marked current, -1 unknown */
    std::string sclk, mclk;          /* "500-2350 MHz", the lowest and highest DPM level */
    int64_t     power_cap_uw = -1, power_cap_default_uw = -1, power_cap_max_uw = -1;
    std::string vddgfx_offset;       /* "-75 mV" from pp_od_clk_voltage, "" when unknown */
    bool        integrated = false;  /* the KFD topology's node has no local memory of its own */

    /* What the engine knows and the kernel does not: filled by the engine for its own cards. */
    std::string name;                /* the runtime's product name */
    int         rank = -1;           /* the tensor-parallel rank it serves, -1 none */
    int         device = -1;         /* the backend's device index, -1 none */
    std::string dma;                 /* what carries its copy lists (dev_dma_lane), "" unknown */
};

struct HostFacts {
    std::string cpu_model;
    int         cores = -1, threads = -1;
    int64_t     mem_total = -1, mem_available = -1, swap_total = -1, swap_free = -1;
    std::string kernel, distro;
    std::string container;           /* "docker", "podman", "" for none seen */
    int64_t     cgroup_mem_limit = -1;   /* bytes; -2 is "max", -1 unknown */
    int64_t     memlock_soft = -1;       /* bytes; -2 is unlimited, -1 unknown */
    std::string thp_enabled, thp_defrag, thp_shmem;   /* the selected word of each */
    std::string iommu;               /* passthrough | translated | off | unknown */
    std::string iommu_why;           /* what that was read from */
    std::string cmdline;             /* the boot parameters that bear on a card, space separated */
    int64_t     ttm_pages_limit = -1;    /* pages; TTM's limit on the system memory it allocates */
    int64_t     page_size = 4096;
    int         cwsr_enable = -1;
    std::string io_uring;            /* available | refused (why) | unknown */
    std::string cpu_governor, cpu_epp, cpu_driver;
    std::string runtime;             /* the device runtime's version, "" for none */
};

struct Facts {
    HostFacts              host;
    std::vector<CardFacts> cards;    /* every display-class PCI device, in address order */
};

/* Reads /proc and /sys under `root` ("" is the live system). Never fails: what cannot be read is
 * left unknown.
 *
 * `managed`, when given, names the cards (by PCI address) whose power-management files are read
 * -- the DPM tables, the overdrive block, the busy percentage and the hwmon power limits; every
 * other card is described from its PCI and memory files alone. The driver answers those files
 * from the card's SMU under its power-management lock, and wakes a card that was runtime-
 * suspended to do it, so a server reading the machine again while it serves reads them for the
 * cards it serves on and no others. Null reads them for every card. */
Facts collect(const std::string& root, const std::vector<std::string>* managed = nullptr);

/* What only a system call answers, filled for the live process: the memlock limit and whether
 * io_uring is allowed here (a container's seccomp profile refuses it). */
void probe_process(HostFacts* h);

/* The names of the variables in `env` (an environ-shaped, null-terminated array) that start with
 * `prefix`, sorted. Names only: a value can be a path, a host name or a token, and the report is
 * written to be pasted into an issue. */
std::vector<std::string> env_names(char* const* env, const char* prefix);

/* ------------------------------------------------------------------ what the engine did */
struct RunFacts {
    std::string version;                 /* the engine's, for the report's first line */
    bool        device_build = true;     /* built with a device runtime (HIP) */
    int         tp = 1;
    bool        host_backend = false;
    bool        failed = false;          /* startup stopped; the report is what it got to */
    int         p2p = -1;                /* 1 peer to peer, 0 through host memory, -1 n/a */
    bool        p2p_forced_off = false;  /* --p2p off */
    std::string p2p_blocked;             /* the pairs the driver refused */
    std::vector<std::string> ignored;    /* devices left out, each with the reason */
    bool        tp_wire_lossy = false;
    int64_t     tp_wire_min_kb = 0;
    bool        prefix_cache = true;
    /* Why the engine turned prefix caching off itself (the KL mode, an embedding model); "" when
     * it is on or the operator turned it off. */
    std::string prefix_cache_off_why;
    int64_t     prefix_host_mib = 0, prefix_disk_mib = 0;
    int64_t     host_pool_mib = 0;       /* a rank, as configured */
    int64_t     pinned_bytes = -1;       /* every pinned host allocation the process made */
    int64_t     host_weight_bytes = 0;   /* weights a rank reads from the pinned host pool */
    int64_t     file_weight_bytes = 0;   /* and from the container on disk */
    int64_t     locked_bytes = 0;        /* memory locked for a host gather (an n-gram copy) */
    std::string ngram_placement;         /* where an n-gram table went, "" for a model without one */
    bool        reference_kernels = false;
    bool        profile_ops = false;
    bool        deterministic = false;
    /* The RADIANCE_* variables set in the environment, by name (env_names). The option table
     * covers what the command line takes; these change behaviour beside it -- the log level, a
     * pinned tuning, an instrument left on -- and a report of every setting in force names them. */
    std::vector<std::string> environment;
};

/* ------------------------------------------------------------------ findings */
struct Finding {
    enum Level { Note = 0, Warn = 1 };
    Level       level = Note;
    std::string id;     /* stable, for a test or a grep: "link-below-max" */
    std::string what;   /* what was seen, with the numbers */
    std::string why;    /* what it costs */
    std::string fix;    /* what to do: a flag, a setting, a command */
};

std::vector<Finding> check(const Facts& f, const RunFacts& r);

/* ------------------------------------------------------------------ the report */
std::string render(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                   const Facts& f, const RunFacts& r, const std::vector<Finding>& findings);

/* The same, as a JSON object in text: {"configuration", "derived", "hardware", "findings"}. */
std::string to_json(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                    const Facts& f, const RunFacts& r, const std::vector<Finding>& findings);

/* Merges the decisions into the option rows: a decision whose key is an option's flag supplies
 * that row's value and reason (and `auto` as its source when the option was not given). The
 * decisions no row took are returned in `derived`, in the order they were made. */
void merge(std::vector<ConfigRow>* rows, const std::vector<Decision>& decisions,
           std::vector<Decision>* derived);

/* "32.0 GT/s PCIe" -> 32.0; 0 when it does not parse. */
double link_gts(const std::string& speed);

}  /* namespace diag */
}  /* namespace rad */
