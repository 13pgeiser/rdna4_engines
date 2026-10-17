/* kbench_timer.h -- rad-kbench's timing pass (--bench).
 *
 * Correctness runs first. A device case that passes is handed to the DeviceTimer with its operands
 * still on the device, and timed later, back to back with the other passing cases, after one
 * sustained warm-up, with no host work between its batches. A host case is timed where it was
 * checked (time_host): the CPU needs no warm-up the correctness work would undo.
 *
 * What a device time is: device timestamps around a batch of back-to-back launches, divided by the
 * launch count. The batch is queued behind a gate (rad_stream_gate) the host opens once all of it
 * is queued, so neither the host's submission rate nor its synchronisation is in the interval; a
 * keep-alive copy runs ahead of the gate so the device is not idle (and its clock not falling)
 * while the host submits. Cold timing rotates the weights over copies spanning the footprint, so
 * every launch reads them from memory; in-place operands are restored from a pristine copy before
 * every launch, and the restores' own time, measured in the same window, is subtracted. Each case
 * runs until its median is known to 1% or the rep cap; a batch during which a throttler came on
 * that was off at idle and after the warm-up is discarded and repeated. The rules are pure
 * functions in kbench_core.h.
 */
#pragma once
#include "kbench_core.h"
#include "rad_abi.h"
#include "rad_device.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace rad {
namespace kb {

struct TimerOpts {
    int64_t footprint = -1;     /* rotation footprint in bytes: 0 warm, < 0 the default */
    int     fixed = 0;          /* --bench-iters: launches a batch; 0 sizes each batch */
    int     max_reps = 20;      /* --bench-reps: the most batches a case takes */
    int     warm_ms = 2000;     /* --bench-warm-ms: sustained load before each timing pass */
    int64_t budget = 0;         /* device bytes held for timing before a timing pass runs */
    bool    verbose = false;
};

/* A timed case's numbers. */
struct Timing {
    TimeStats   st;                  /* microseconds a launch */
    double      clock_mhz = -1;      /* median of the batches' averaged graphics clock */
    double      restore_pct = -1;    /* in-place restores, as a percent of the timed batch */
    int         threads = 0;         /* host rows: the OpenMP threads the library runs on */
    int         retried = 0;         /* batches repeated because a throttler came on */
    bool        gated = false;       /* a device case timed behind the gate */
    std::string rotation;            /* "33 copies, 2 weight table(s) through pointer tables" */
    std::string note;                /* why a number is missing or worth less */
};

/* One operand as the checked launch saw it. */
struct TimedOperand {
    RadTensor t{};                   /* device pointers for a device row */
    int       role = 0;
    int64_t   bytes = 0;             /* the buffer t.data points at */
    /* An in-place operand's bytes before the checked launch, host memory, read during add() or
     * time_host only. */
    const uint8_t* pristine = nullptr;
    /* An input's timing draw (kb::timing_input), host memory; empty for the recorded draw. */
    std::vector<uint8_t> variant;
};

struct TimedCase {
    const RadKernelInfo* info = nullptr;
    Geometry g;
    std::vector<TimedOperand> opd;
    void*   scratch = nullptr;
    int64_t scratch_bytes = 0;
    /* Device allocations (rad_dev_alloc, RAD_MEM_DEVICE) the timer frees once the case is timed. */
    std::vector<void*> frees;
    int64_t held = 0;                /* their bytes, against the timer's budget */
    uint64_t seed = 0;               /* for the weight tables' permutations */
    std::string what;                /* for verbose lines */
};

/* The device half: holds passing cases and times them in passes. Not thread-safe; one stream. */
class DeviceTimer {
public:
    DeviceTimer(const TimerOpts& o, RadStream s, Phases* ph);
    ~DeviceTimer();
    DeviceTimer(const DeviceTimer&) = delete;
    DeviceTimer& operator=(const DeviceTimer&) = delete;

    /* Takes the case over (its allocations included) and returns the id its Timing will have. A
     * case that would take the held bytes past the budget first times what is held. */
    int64_t add(TimedCase&& c);
    /* Times every held case and frees them. */
    void flush();
    /* Forgets a case: frees it if still held, and drops its result. For a case whose verdict,
     * reached after it was handed over, says it is not worth timing. */
    void drop(int64_t id);
    /* The result for an id, after the flush that timed it; null before. */
    const Timing* result(int64_t id) const;

    /* For the report header: how the times were taken, and the calibration. */
    const std::vector<std::string>& header() const { return header_; }
    /* Non-empty when the calibration says cold timings would not be cold: no case is timed. */
    const std::string& withheld() const { return withheld_; }
    bool calibrated() const { return set_up_; }

private:
    struct Prepared;
    struct Rep;
    struct Held {
        int64_t   id = 0;
        TimedCase c;
        std::vector<void*> pristine;   /* per operand: its device copy before the checked launch */
    };

    void setup();
    void warm_up();
    void calibrate();
    void sweep(int64_t bytes);
    bool prepare(Held& h, Prepared* p, std::string* why);
    void release(Prepared* p);
    void unit(Prepared& p, bool restore, int* rc);
    Rep  rep(Prepared& p, int64_t n, bool gated);
    void time_case(Held& h);
    void free_case(Held& h);
    double clock_now(uint64_t* throttle);
    double read_rate(const uint8_t* src, int64_t bytes, int64_t copies, const void* ranges,
                     int ranges_n);

    TimerOpts o_;
    RadStream s_ = nullptr;
    Phases*   ph_ = nullptr;
    std::vector<Held> held_;
    int64_t   held_bytes_ = 0;
    int64_t   next_id_ = 0;
    std::map<int64_t, Timing> results_;

    bool set_up_ = false, setup_failed_ = false;
    std::string withheld_;
    std::vector<std::string> header_;
    int64_t   footprint_ = 0;        /* 0 in warm timing */
    int64_t   scale_ = 0;            /* the footprint the probes and keep-alive are sized by */
    RadEvent  ev_[8] = {};
    void*     sweep_src_ = nullptr;
    int64_t   sweep_src_bytes_ = 0;
    void*     sweep_dst_ = nullptr;
    int64_t   sweep_bytes_ = 0, sweep_max_ = 0, sweep_off_ = 0;
    void*     word_ = nullptr;       /* the launch floor's 256-byte target */
    double    floor_us_ = -1, floor_bracketed_us_ = -1;
    std::string metrics_path_;
    /* Throttler bits on at idle (some firmware keeps one set for good) and on after the warm-up's
     * sustained load (a power limit that binds throughout): neither makes a batch worth repeating,
     * since the next batch would see them too. */
    uint64_t  idle_throttle_ = 0, load_throttle_ = 0;
    double    warm_clock_ = -1;
    std::chrono::steady_clock::time_point last_busy_{};
};

/* Times a host row on the calling thread: per-launch steady_clock times (or batches of launches
 * past 200 us when nothing is restored), one untimed launch first unless the checked launch took a
 * second or more, in-place operands restored outside the timed region, weights rotated over
 * `footprint` bytes (0: warm). Stops at the median's 1% interval, the rep cap, or two seconds.
 * Every operand the launch writes is a private copy, so the caller's buffers keep the checked
 * launch's output. `checked_s` is that launch's wall time. */
Timing time_host(const TimedCase& c, const TimerOpts& o, int64_t footprint, double checked_s);

/* The bytes of the host's last-level caches, every distinct instance counted once (sysfs), or 0. */
int64_t host_llc_bytes();

}  /* namespace kb */
}  /* namespace rad */
