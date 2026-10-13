// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Flash-Next expert-parallel balanced ownership (`HIPFIRE_EP_BALANCED_OWNERS=1`).
//
// One workgroup of 512 threads (thread `e` = expert `e`, n_exp == 512), run
// once per MoE call right after the router wrote `ids[0..slots)` (slot order,
// `rows * k` ids, i32 bit patterns in the router's F32 scratch):
//
//   routed[e] = some slot has id e
//   idx(e)    = number of routed experts with id < e          (exclusive scan)
//   owner(e)  = idx(e) % n_ranks                              (identical on every rank)
//   live_*[e] = (routed[e] && owner(e) == self_rank) ? full_*[e] : dummy_*
//   slot_owner[s] = owner(ids[s])   (-1 for a negative or out-of-range id)
//
// `live_*` are the rank's pointer tables the routed producers read (all 512
// entries rewritten, nothing stale); `full_*` are the layer's read-only full
// tables (owned experts -> local compact tensor, others -> the owner peer's
// compact tensor); `dummy_*` are this rank's zero dummy expert. Distinct routed
// experts are dealt to ranks alternately in ascending id order, so per-rank
// distinct counts differ by at most one. Pure integer code: the host reference
// `qwen4_ep_owner_map` (rdna-compute `moe.rs`) is the same function.

#define OWNER_EXPERTS 512

extern "C" __global__ void __launch_bounds__(OWNER_EXPERTS)
qwen4_ep_owner_assign(const int *__restrict__ ids,
                      const unsigned long long *__restrict__ full_gate_up,
                      const unsigned long long *__restrict__ full_down,
                      unsigned long long *__restrict__ live_gate_up,
                      unsigned long long *__restrict__ live_down,
                      int *__restrict__ slot_owner, unsigned long long dummy_gate_up,
                      unsigned long long dummy_down, int slots, int n_ranks,
                      int self_rank) {
    __shared__ int routed[OWNER_EXPERTS];
    __shared__ int scan[OWNER_EXPERTS];
    __shared__ int before[OWNER_EXPERTS];
    const int e = (int)threadIdx.x;

    routed[e] = 0;
    __syncthreads();
    if (e < slots) {
        const int id = ids[e];
        if (id >= 0 && id < OWNER_EXPERTS)
            routed[id] = 1; // every writer stores the same value
    }
    __syncthreads();

    // Inclusive Hillis-Steele scan of `routed`, then the exclusive count.
    const int mine = routed[e];
    scan[e] = mine;
    __syncthreads();
    for (int offset = 1; offset < OWNER_EXPERTS; offset <<= 1) {
        const int add = e >= offset ? scan[e - offset] : 0;
        __syncthreads();
        scan[e] += add;
        __syncthreads();
    }
    const int idx = scan[e] - mine;
    before[e] = idx;
    __syncthreads();

    const int owner = idx % n_ranks;
    const bool take = mine != 0 && owner == self_rank;
    live_gate_up[e] = take ? full_gate_up[e] : dummy_gate_up;
    live_down[e] = take ? full_down[e] : dummy_down;

    if (e < slots) {
        const int id = ids[e];
        slot_owner[e] = (id >= 0 && id < OWNER_EXPERTS) ? before[id] % n_ranks : -1;
    }
}
