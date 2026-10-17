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

}  /* namespace rad */
