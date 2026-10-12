#pragma once
// r4d_ar_seq.h: the step every all-reduce and all-gather kernel takes first -- its handshake
// sequence number and the scratch slot it pushes into -- and the wrap-safe wait that follows the
// push. One definition for the 2-rank and the wide families, because every kernel that shares an
// instance's buffers has to take the step identically or the two ranks stop agreeing.
//
// THE COUNTER ALLOCATION IS 2 * nbw WORDS, zeroed once, where nbw is the block capacity the flag
// array was sized for, and the landed counter after them (r4d_ar_link_landed):
//
//   [0, nbw)       HANDSHAKE counters, one a block. Block b advances word b once per launch it
//                  runs in, publishes the new value to the peer and waits for the peer's. Kept per
//                  block so that a block which sat out a run of narrow launches resumes exactly
//                  one step past the flag its peer last published: the wrap-safe compare below is
//                  only sound while that gap stays far under 2^31, and a counter every block
//                  shared would let an idle block's flag fall arbitrarily far behind.
//
//   [nbw, 2*nbw)   PARITY counters. Every launch advances ALL nbw of them exactly once, whatever
//                  its own block count -- block b takes words b, b + nb, b + 2*nb, ... -- so all
//                  of them always equal the number of launches, and each block reads the slot
//                  parity off its own word without talking to any other block.
//
//   [2*nbw]        the LANDED counter: the blocks of the running launch that have seen their
//                  peer's push land, back at zero when the launch ends.
//
// WHY THE SLOT CANNOT COME OFF THE HANDSHAKE COUNTER. The double-buffered scratch is safe only if
// consecutive launches on one instance use opposite slots: launch k+1 may push while the peer is
// still reducing launch k, and it must not land in the bytes being read. Block counts differ from
// launch to launch -- the standalone all-reduce scales with the message, the fused norm takes one
// block a row -- and so does the byte range a block owns. The handshake counters therefore drift
// apart block by block, and a slot taken from them lets a block of launch k+1 push into the slot a
// DIFFERENT block of launch k is still reading on the peer, whenever their byte ranges overlap
// and their counters happen to share a parity. The parity counters move in lockstep, so the slot
// alternates per launch for every block, as the double buffer requires.
//
// THE COST is one more returning atomic, issued from a second wave so its round trip overlaps the
// handshake one rather than following it, and nbw / nb fire-and-forget atomics spread over the
// other threads. Nothing waits on those: the barrier below orders LDS only, so they land in the
// background, and all they need is to land before the next launch on the stream, which the kernel
// boundary guarantees. Everything stays device-resident, so a captured graph still replays.
#include <hip/hip_runtime.h>

// Sets `s_seq` to this block's handshake value and `s_slot` to the launch's scratch slot (0 or 1),
// then barriers so every thread can read both. `nbw` is the capacity the counters were allocated
// for, and gridDim.x must not exceed it.
//
// The barrier is fenced on LDS alone. The only thing it publishes is the two words in LDS; a full
// __syncthreads would also wait for every thread's outstanding parity atomics and invalidate the
// vector caches, which puts the fire-and-forget traffic on the path to the first push.
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

// THE ENGINE'S HOLD ON THE LINK, ENDED BY THE LAUNCH (rad_abi.h, rad_kernel_collective answering
// 2). The word is odd exactly while the engine holds the link for this launch, and nothing else
// writes it while the launch runs, so its value is read once, at the start, by each block's first
// thread (r4d_ar_link_word): the read crosses to host memory, and at the start it overlaps the push
// where at the end it would hold up the reduce.
//
// r4d_ar_link_landed is called by every block once its handshake has seen the peer's push to it
// land. The last block to get there knows every byte the peer sends this card has arrived, so the
// link is free for the rest of the kernel's run -- the reduce and whatever it writes -- and takes
// one off the word if it was odd, with an atomic so it commutes with any barrier counting the same
// word down. `ctr` counts the blocks and is back at zero when the launch ends, ready for the next
// one on the stream. The word is in memory the DMA engine polls past every cache, so it is read and
// written at system scope.
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

// True while `flag` has not yet reached `want`. The counters are u32 and wrap, so the comparison is
// on the signed difference: a peer at or past `want` -- it may already be a launch ahead -- reads
// as arrived on either side of the wrap, where a plain `flag < want` stops waiting altogether once
// `want` wraps to a small number.
__device__ __forceinline__ bool r4d_ar_seq_behind(unsigned int flag, unsigned int want) {
  return (int)(flag - want) < 0;
}
