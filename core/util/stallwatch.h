/* stallwatch.h -- naming where a phase that has gone quiet is waiting.
 *
 * A startup that stops printing leaves the hardest report to act on: the log ends on the last line
 * that was printed, and whatever came after it -- a driver call that never returned, a lock inside a
 * runtime library -- leaves no trace. The machine it happens on is often one this project cannot
 * run (a guest kernel, a card passed through to a VM), so where it waits has to come from the
 * process itself.
 *
 * A StallWatch is held across a phase by the thread doing it. When nothing has been logged for the
 * quiet interval it says what that thread is doing:
 *   - its scheduler state and kernel wait channel;
 *   - the system call it is in, an ioctl with its device and request, and the library that made
 *     the call;
 *   - when the thread waits in user space (a lock, a sleep), its stack as module+offset frames,
 *     which addr2line symbolises against the released binary. A thread inside an ioctl is not
 *     signalled: the call already names where it waits, and a signal could end a driver wait
 *     early;
 *   - the process's other threads that are inside an ioctl, since a lock in a runtime library is
 *     usually held by one of them.
 * It says it again each time the silence doubles, so a later log shows whether the thread moved.
 * The watch reads only its own process's /proc entries and costs one wake-up a second while held.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <sys/types.h>

namespace rad {

class StallWatch {
public:
    /* `phase` names what is being watched in the report ("startup"). */
    explicit StallWatch(const char* phase, uint64_t quiet_ms = 60000);
    ~StallWatch();
    StallWatch(const StallWatch&) = delete;
    StallWatch& operator=(const StallWatch&) = delete;

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
};

/* What thread `tid` of this process is doing, as the report words it: state, wait channel, system
 * call. With `stack`, a thread waiting in user space is also asked for its frames. */
std::string stall_describe_thread(pid_t tid, bool stack);

/* One line of /proc/<pid>/task/<tid>/syscall in words: "running", "not in a system call", or the
 * call with its arguments decoded where they name something (an ioctl's device and request). */
std::string stall_decode_syscall(const std::string& line);

/* `addr` as the module that maps it and its offset from the module's load base, "libfoo.so+0x1a2b":
 * the number addr2line takes for that module. */
std::string stall_module_offset(uintptr_t addr);

}  /* namespace rad */
