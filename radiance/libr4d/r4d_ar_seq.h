#pragma once
// r4d_ar_seq.h: the first step of every all-reduce and all-gather kernel (its handshake sequence
// number and scratch slot) and the wrap-safe wait after the push. Shared by the 2-rank and wide
// families: kernels sharing an instance's buffers must take the step identically or the ranks
// disagree.
//
// Counter allocation: 2 * nbw words, zeroed once (nbw = block capacity the flag array was sized
// for), then the landed counter (r4d_ar_link_landed):
//
//   [0, nbw)       Handshake counters, one per block. Block b advances word b once per launch it
//                  runs in, publishes it to the peer and waits for the peer's. Per block so a
//                  block idle through narrow launches resumes one step past its peer's last flag;
//                  the wrap-safe compare needs that gap far under 2^31.
//
//   [nbw, 2*nbw)   Parity counters. Every launch advances all nbw of them once, whatever its block
//                  count (block b takes words b, b + nb, b + 2*nb, ...), so each equals the launch
//                  count and a block reads the slot parity off its own word.
//
//   [2*nbw]        Landed counter: blocks of the running launch that have seen their peer's push
//                  land; back at zero when the launch ends.
//
// The slot cannot come from the handshake counter. Consecutive launches on one instance must use
// opposite scratch slots, since launch k+1 may push while the peer still reduces launch k. Block
// counts and per-block byte ranges vary by launch, so handshake counters drift apart per block,
// and a block of launch k+1 could push into the slot a different block of launch k is still
// reading. The parity counters move in lockstep, so the slot alternates per launch for every block.
//
// Cost: one more returning atomic, issued from a second wave so its round trip overlaps the
// handshake's, plus nbw / nb fire-and-forget atomics over the other threads. Those only need to
// land before the next launch, which the kernel boundary guarantees. All state is device-resident,
// so a captured graph replays.
#include <hip/hip_runtime.h>

// Sets `s_seq` to this block's handshake value and `s_slot` to the launch's scratch slot (0 or 1),
// then barriers so every thread can read both. `nbw` is the capacity the counters were allocated
// for, and gridDim.x must not exceed it.
//
// The barrier fences LDS only, since it publishes just the two LDS words. A full __syncthreads
// would wait for the outstanding parity atomics and invalidate the vector caches, putting that
// traffic on the path to the first push.
__device__ __forceinline__ void r4d_ar_seq_take(unsigned int* __restrict__ ctrs, unsigned int nbw,
                                                unsigned int& s_seq, unsigned int& s_slot) {
  const unsigned int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  unsigned int* const par = ctrs + nbw;
  const unsigned int pt = nt > 32u ? 32u : 0u;   // first lane of the second wave, if there is one
  if (tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
  if (tid == pt) s_slot = (atomicAdd(&par[b], 1u) + 1u) & 1u;
  for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The engine's hold on the link, released by the launch (rad_abi.h, rad_kernel_collective
// returning 2). The word is odd exactly while the engine holds the link for this launch and
// nothing else writes it meanwhile, so each block's first thread reads it once at the start
// (r4d_ar_link_word), where the host-memory read overlaps the push instead of delaying the reduce.
//
// Every block calls r4d_ar_link_landed once its handshake has seen the peer's push land. The last
// block knows all of the peer's bytes have arrived, so the link is free for the rest of the
// kernel; it decrements the word if odd, atomically so it commutes with any barrier counting the
// same word down. `ctr` counts blocks and returns to zero at the end of the launch. The DMA engine
// polls the word past every cache, so it is accessed at system scope.
__device__ __forceinline__ long long r4d_ar_link_word(const long long* gate, unsigned int tid) {
  return gate && tid == 0 ? __hip_atomic_load(gate, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                          : 0;
}

__device__ __forceinline__ void r4d_ar_link_landed(long long* gate, long long word,
                                                   unsigned int* ctr, unsigned int nb,
                                                   unsigned int tid) {
  if (!gate || tid != 0) return;
  const unsigned int prev = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_AGENT);
  if (prev + 1u != nb) return;
  __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (word & 1) __hip_atomic_fetch_add(gate, -1LL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// True while `flag` has not reached `want`. The u32 counters wrap, so compare the signed
// difference: a peer at or past `want` (possibly a launch ahead) reads as arrived across the wrap,
// where a plain `flag < want` would stop waiting once `want` wraps to a small number.
__device__ __forceinline__ bool r4d_ar_seq_behind(unsigned int flag, unsigned int want) {
  return (int)(flag - want) < 0;
}
