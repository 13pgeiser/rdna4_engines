/* work_counter.h -- measuring work instead of time, for tests that bound a cost.
 *
 * Work::of(fn) counts the instructions the calling thread retires in user space while `fn` runs
 * (perf_event_open, PERF_COUNT_HW_INSTRUCTIONS, kernel and hypervisor excluded). The count depends
 * on the code path and the input, not on machine load, clock frequency or cache state, so a bound
 * on it does not flake when other tests run beside it. Tests compare ratios of two counts, or a
 * count per item against a bound far from the failure it catches.
 *
 * Where the counter is refused (no PMU in a VM, perf_event_paranoid above 2), the fallback is the
 * thread's CPU time in nanoseconds, which still excludes time spent descheduled. exact() says
 * which is in use. A multiplexed counter is scaled by its enabled/running times.
 */
#pragma once
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <functional>

namespace radtest {

class Work {
public:
    Work() {
        perf_event_attr a;
        std::memset(&a, 0, sizeof a);
        a.size = sizeof a;
        a.type = PERF_TYPE_HARDWARE;
        a.config = PERF_COUNT_HW_INSTRUCTIONS;
        a.disabled = 1;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        fd_ = (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
    }
    ~Work() {
        if (fd_ >= 0) close(fd_);
    }
    Work(const Work&) = delete;
    Work& operator=(const Work&) = delete;

    /* Instructions retired, or CPU nanoseconds without a counter. */
    bool exact() const { return fd_ >= 0; }
    const char* unit() const { return fd_ >= 0 ? "instructions" : "cpu ns"; }

    uint64_t of(const std::function<void()>& fn) {
        if (fd_ < 0) {
            timespec a{}, b{};
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &a);
            fn();
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
            return (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec));
        }
        ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
        fn();
        ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t v[3] = { 0, 0, 0 };   /* value, time enabled, time running */
        if (read(fd_, v, sizeof v) != (ssize_t)sizeof v) return 0;
        if (v[2] == 0) return 0;
        return v[2] < v[1] ? (uint64_t)((double)v[0] * (double)v[1] / (double)v[2]) : v[0];
    }

private:
    int fd_ = -1;
};

/* One counter per thread that uses it. */
inline Work& work() {
    static thread_local Work w;
    return w;
}

}  /* namespace radtest */
