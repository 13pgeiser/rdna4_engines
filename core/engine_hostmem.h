/* engine_hostmem.h -- the memory cgroup limits bring-up checks before it locks a table into host
 * memory (engine_bringup.cpp, lockable_room). */
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace rad {

/* A cgroup's limit less the usage reclaim cannot give back: memory.current less file pages (v2)
 * or usage_in_bytes less cache (v1), since page cache yields to a locked allocation and anonymous
 * memory does not. "max" and v1's near-INT64_MAX sentinel mean no limit. True, with `room` set,
 * when `dir` has a limit. */
bool cgroup_room(const std::string& dir, bool v2, int64_t* room);

/* Every level of the memory cgroup at `path` (as /proc/self/cgroup names it), from that level up
 * to the hierarchy's root mounted at `root`, leaf first: `visit(dir, room)` for each level with a
 * limit, until it returns false. Each level is a limit of its own. */
void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* Whether `need` more bytes fit in host memory after `still` more this process will allocate first
 * and `margin` (--host-headroom-mib) stays free, or why not. Checks MemAvailable; every memory
 * cgroup from the process's own up to the visible root (Docker's --memory, enforced by the OOM
 * killer); and, for an mlock()ed copy (`mlocks`), RLIMIT_MEMLOCK unless the process has
 * CAP_IPC_LOCK. Pinned memory a card reads is a driver userptr mapping, which RLIMIT_MEMLOCK does
 * not count. An unreadable limit is ignored, never guessed. Shared with kvstream/bringup.cpp,
 * which checks the streamed group's host mirror before any rank takes one. */
bool lockable_room(int64_t need, int64_t still, int64_t margin, bool mlocks, std::string* why);

}  /* namespace rad */
