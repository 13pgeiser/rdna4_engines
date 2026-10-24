// kv_resolve: turn a streamed KV group's selected block ids into the ring slots that hold them,
// copy a block in from the pinned host mirror when it is not resident, and rewrite the selection
// in place (docs/OPS.md "kv_resolve", docs/KV-STREAMING.md §3.4 and §4.2).
//
// `qsa_select` already writes the group's pool page id into `sel`, and `attn_paged` reads every
// entry as a raw pool index. A streamed group's device pool is `slots` blocks deep while the id
// space is `blocks`, so an id is no longer an address; after this op each entry is a slot of
// `kv_cache` and the attention is unchanged. `sel` is the only place the sparse path names KV
// storage, which is why one op between select and attention suffices.
//
// The order is the contract (libref/ref_kvstream.cpp), because two implementations must place the
// same blocks in the same slots and so produce the same table:
//
//   1. bump the epoch (ctl[0] + 1);
//   2. one lookup per entry below nsel[q]: a hit takes the epoch and, the first time this call
//      stamps that slot, bumps its two-bit heat counter by one (clamped at 3) -- one bump a slot a
//      call, not one an entry, so a block many rows name gains one step of heat, the same as one a
//      single row names. A miss claims its map entry once (-1 -> -2, atomicCAS), so the map is its
//      own set. That CAS is also the distinct-claim count: it succeeds exactly once per block,
//      which is the
//      `s_distinct` step 3 would otherwise have to scan the map for. The value read is kept, one
//      int an entry, for step 7 below;
//   3. compact the claimed blocks out of `res` in ASCENDING block id. The claims are made by
//      whichever thread reaches them first, but this scan is what fixes the miss order, so
//      atomic claim order never reaches the placement. A thread owns one contiguous chunk of the
//      map and reads it four ids at a time, so the list is still ascending by construction. Only
//      taken when the call claimed something: with no claim there is no `-2` and nothing to
//      place (step 2's count says so);
//   4. one victim per miss, in the hand order from ctl[1]: a FREE slot (slot_block < 0) first, so
//      no resident block is evicted while the ring has room (the free count is taken in parallel
//      once per call); only a call with no free slot falls back to CLOCK's second chance, where a
//      slot stamped with this call's epoch is passed and keeps its counter, a warm slot (counter
//      non-zero) has its counter HALVED and is passed, and a cold slot (counter 0) is taken. The
//      hand PICKS the victim: this is a per-visit decision, so the policy costs no reduction, no
//      barrier and no extra traffic -- an add-and-clamp and a shift where a constant store and a
//      zero store would do. An exact LRU would need the minimum over the ring per miss, a
//      collective reduction that pays more than the choice is worth. The hand ends just past the
//      last slot taken; the sweep visits at most 3*slots, because every write clamps the counter
//      to 3 and two halvings take any value to 0;
//   5. re-point both maps as each victim is taken (`res[old] = -1`, then slot -> b and b -> slot,
//      with the slot's counter bumped by one as a use: the victim's is 0 by construction, so it is
//      1 in the common case, and the clamp keeps a slot taken by identity or from the free region
//      inside [0, 3]);
//   6. copy `block_bytes` of host[b] into kv_cache[s] for each placed miss, as bytes, in this one
//      launch;
//   7. rewrite every entry below nsel[q] whose block is resident as its slot; entries past nsel[q]
//      keep qsa_select's -1 padding. An entry step 2 saw resident uses the slot it read there, so
//      `res` is read once a selection entry, not twice (see "THE ENTRY CACHE").
//
// OVERFLOW. The ring must hold the union of the call's selection (`hits + misses <= slots`). When it
// does not, the sweep cannot place every miss: the unplaced claims are reverted to -1 (so a later
// call may claim them again), ctl[3] is latched to 1, and their selection entries are rewritten to
// SLOT 0 rather than left holding the block id. The id would be an address past the ring, which
// attn_paged dereferences with no bounds check, so an overflow would fault or read another layer;
// slot 0 is defined memory, wrong numbers the caller discards on ctl[3].
//
// STATE. The epoch, the hand, the two-bit heat counters and the counters live between launches, so
// this row is not concurrent (r4d_rows.cpp's kStateful[]): two launches must never overlap, and the
// engine orders them. It is still recordable: the maps, the ring and the mirror are fixed-size
// from the group's declaration, so addresses and pitches never move.
//
// COST. A decode step issues this once per QSA layer -- twelve launches, in a step whose whole
// budget is ~33 ms -- so a launch's own cost is the miss path, not the dispatch: ~20 to ~60 misses
// a launch at 16k-160k of history. Everything that path touches is therefore sized from the call's
// own entries and not from `blocks`:
//
//   - the claim list comes out of the lookup pass, which already sees every claim, so no pass walks
//     `res` over `blocks`;
//   - the union is a metric (`ctl[10]`), counted on the first stamp per slot, so no pass walks
//     `slot_state` over `slots`: a ring is sized by misses and not by slots, and a ring larger than
//     the history does not cost more than a small one;
//   - the copies are independent, so each miss has a warp of its own and every miss's read is in
//     flight at once.
//
// What is left is the host read itself: a miss copies 2 KiB from a pinned mirror page the card has
// not touched, so it is PCIe transactions with a page walk behind them, and no arrangement of the
// copy loop changes that. The CLOCK sweep is not the cost either: each launch halves the counters
// of the slots it passes, so the ring arrives at the next launch mostly cold and a sweep visits
// about one slot a miss. The free-slot scan ahead of it costs a visit per slot it passes, and the
// count that says whether to scan at all (phase 3b) is one coalesced pass over the two slot maps per
// call, so a ring with no free slot -- where every eviction is unavoidable -- pays neither a full
// turn of scanning per miss nor one serial scan per launch.
//
// SCRATCH. `next_pow2(slots)` block ids then `slots` slots (the row's scratch hook sizes it). The
// miss list is a per-call temporary and would cost the arena planner, the recorded pass and the
// tape key bytes nothing outside this launch reads. The claim list is dynamic LDS (`cap` ids) while
// it fits -- the sort reads every element from every thread, so it wants shared memory -- and the
// scratch's first plane when it does not, which is why that plane is padded to a power of two. The
// entry cache is dynamic LDS too, after the claim list, and absent when it does not fit.
//
// THE CLAIM LIST'S SIZE IS DERIVED, NOT WRITTEN DOWN. A claim is a distinct absent block the
// selection names, so one call makes at most `M * sel_pitch` of them -- and at most `lookups =
// sum_q nsel[q]` in fact -- which is a fact about the call and not about the cache: `blocks` is the
// id space, `max_seqs * ceil(max_ctx / block_size)` and never the size of a launch. So the
// launcher sizes the list at `next_pow2(min(M * sel_pitch, the room a workgroup has))`, and for
// `--max-num-seqs 4` at `sel_pitch` 513 that is 4096 ids, where a constant would have to be chosen
// for the largest configuration or hand that one the map walk below. Nothing here mentions
// `max_seqs`: the engine declares `sel_pitch` and launches `M` rows, and any --max-num-seqs follows
// from those two. `ctl[12]` records the largest claim count one call has made, so a list that is
// ever too shallow is visible in the counters rather than only in the launch's time.
//
// Only the first `slots` claimed blocks are needed: the sweep cannot place more than `slots`, and
// every claim past that is still reverted to -1 by the final pass, so a capped list changes no
// placement.
//
// ABOVE THE LIST. A launch wider than the list -- 28 or more rows at `sel_pitch` 513, since a
// workgroup has tens of thousands of ids of room -- gathers its claims from the selection into the
// scratch and sorts them there (phase 3). The walk is over the call's entries, `lookups` of them,
// never over `blocks`: the map is only the set that says whether an id is a claim and that keeps a
// block four rows name out of the list four times. Same ids, same ascending order, same placements
// as the fast path. The one call this cannot serve is one claiming more blocks than the ring holds:
// no structure there can hold the whole claim set, so that call -- which has overflowed and whose
// step the caller discards -- keeps the map walk that puts the smallest `slots` claims in order.
//
// THE ENTRY CACHE. Step 2 and step 7 each read `res[b]` for every selection entry, and those reads
// are scattered over `blocks` i32: at decode that is two 513-load waves over a map the launch has
// already touched once. But step 7's answer for a HIT is step 2's
// answer: the only thing that clears a `res` entry in a call is an eviction, and the sweep never
// takes a slot stamped with this call's epoch (the `slot_state` test in both the free scan and the
// CLOCK sweep), which is exactly the stamp step 2 puts on every hit's slot. So a hit's slot cannot move, and only an entry step 2 saw absent can
// change -- a claimed block becomes its placed slot, or stays -1 when the sweep could not place it
// (step 6a's revert), or in the above-the-list path passes through -3 and back. Step 2 therefore
// stores the value it read, one int an entry, and step 7 uses it when it is >= 0 and reads the map
// only otherwise: the same slots, the same `sel`, one scattered wave instead of two.
//
// The cache is `M * sel_pitch` ints of dynamic LDS, the launch's whole entry count, and only while
// the workgroup has that room beside the claim list; a prefill-shaped launch that has not gets
// zero and step 7 reads the map instead. Nothing outside the launch sees it: no operand,
// no scratch, no contract. Entries whose block id is out of range are skipped in both steps and
// never reach the cache, and the read is by `sel`'s entry index, which is the same index -- and the
// same thread -- in both steps, so no barrier is needed between them.
//
// STATE FAULTS. libref refuses a `res` entry outside [-1, slots), a `slot_block` outside
// [-1, blocks), a hand outside [0, slots) and a selection entry outside [0, blocks) before it
// writes anything. A device launch cannot return a host status from inside its kernel, so this
// kernel guards instead: an out-of-range hand starts at 0, a map value outside its domain is not
// used to address memory, an out-of-range block id is left in `sel`, and nsel is clamped to
// `sel_pitch` so a bad count cannot steer a row into its neighbour. Those paths are the state
// fault libref refuses; a legal call never takes them and is bit-identical to the reference.
//
// The shape checks (ranks, dtypes, pitches, byte extent per block) belong to the shim below. Like
// every step-sized kernel it takes the launch's row count and row pitch from `sel`'s own extent,
// not from the ranged `M`/`sel_pitch` parameters: `M` arrives as the band's ceiling, so a one-row
// decode and a 625-token prefill would otherwise be refused (arch/common/rad_arch.h's "step-sized
// operands"). `M` is still checked as the ceiling it is, so a genuinely over-wide launch is refused.
// The ring depth is the operand's for the same reason: `slot_block`'s extent is the ring this call
// may address, and the declared `slots` is the ceiling it may not pass -- the pool is planned after
// declare, so a ring the engine sized smaller (--kv-resident) arrives shorter.
// It reads the two byte planes by row pitch in bytes, as libref does, so the same kernel serves
// every cache layout (`fp8`, `fp8_v4`, `k6_v4`, `bf16`) and the copy converts nothing.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_launch.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 1024;

/* A 16-byte vector as a builtin type, not `uint4`: the class copies through a memcpy call, and the
 * miss copy needs its four loads to stay four loads (phase 6). */
typedef unsigned int kv_v16 __attribute__((ext_vector_type(4)));

/* The next power of two at or above `n`, clamped to what an i32 can index: the sort pads to one and
 * the row's scratch is sized from the same expression. */
static long long kv_pow2ceil(long long n) {
    long long p = 1;
    while (p < n && p <= (1LL << 30)) p <<= 1;
    return p;
}

/* The claim list's capacity, in ids, from the launch's own operands and the device's LDS room (the
 * header above, "the claim list's size is derived"). `0` means a workgroup has no room at all, which
 * the shim refuses. `*room_out` is the dynamic LDS room in ints, so the entry cache can take what the
 * claim list leaves (header, "the entry cache"). */
static int kv_claim_cap(long long M, long long pitch, int* room_out) {
    /* Static LDS: the two reduction arrays and the scalars. Everything else is dynamic. */
    const size_t stat = (size_t)kThreads * 2 * sizeof(int) + 64;
    static const int room = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 0;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 0;
        if (pr.sharedMemPerBlock <= stat) return 0;
        size_t r = (pr.sharedMemPerBlock - stat) / sizeof(int);
        size_t p = 1;
        while (p * 2 <= r) p *= 2;      /* a power of two: the sort pads to one, and must not pad
                                         * past the buffer */
        return (int)p;
    }();
    *room_out = room;
    if (room < 1) return 0;
    /* The largest power of two the room holds: `cap` is always one, because the sort pads the list
     * to a power of two and must not pad past the buffer. */
    int top = 1;
    while (top <= room / 2) top *= 2;
    const long long bound = M * pitch;   /* the call's claim ceiling */
    int p = 1;
    while ((long long)p < bound && p < top) p <<= 1;
    return p;
}

/* The heat counter's two writes, the same function in all three implementations (libref's
 * heat_up/heat_down): a use bumps by one and saturates at 3, a sweep pass halves, and a negative
 * value -- never written by a legal call -- reads as cold rather than surviving every pass. */
__device__ __forceinline__ int kv_heat_up(int h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
__device__ __forceinline__ int kv_heat_down(int h) { return h > 0 ? (h >> 1) : 0; }

/* Ascending order of `n` distinct claim ids in the claim list, padded to a power of two with
 * INT32_MAX, which sorts past every block id. A caller's buffer, because both paths sort: the whole
 * claim set when it fits the list and one window of the id space when it does not. Block ids are
 * distinct, so no equal pair decides anything. */
__device__ __forceinline__ void kv_sort_claims(int* list, int n, int tid) {
    int n2 = 1;
    while (n2 < n) n2 <<= 1;
    for (int i = n + tid; i < n2; i += kThreads) list[i] = INT32_MAX;
    __syncthreads();
    for (int k = 2; k <= n2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n2; i += kThreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const int a = list[i], b = list[ixj];
                    /* Ascending overall: a pair whose lower index sits in the first half of a
                     * k-block sorts up, the other half sorts down (bitonic). */
                    const bool up = (i & k) == 0;
                    if (up ? a > b : a < b) { list[i] = b; list[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(kThreads) void kv_resolve_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ nsel,
    int32_t* __restrict__ res, int32_t* __restrict__ slot_block,
    int32_t* __restrict__ slot_state, int32_t* __restrict__ ctl,
    const uint8_t* __restrict__ host, uint8_t* __restrict__ cache,
    int32_t* __restrict__ miss_block, int32_t* __restrict__ miss_slot,
    int M, int blocks, int slots, int pitch, int bbytes, int sst_ld, int cap, int ecache) {

    __shared__ int s_epoch, s_lookups, s_nmiss, s_distinct, s_placed, s_overflow, s_claims, s_hits;
    __shared__ int s_free;
    __shared__ int s_cnt[kThreads], s_off[kThreads];
    /* The claim list, before the sort, `cap` ids of dynamic LDS (kv_claim_cap). Shared rather than
     * the row's scratch because the sort reads every element from every thread; the scratch still
     * holds the placed miss list, which is what the row declares and what the sweep and the copies
     * walk in order. `ecache` ints follow it: the value this call's lookup read for each selection
     * entry, which step 7 reuses (header, "the entry cache"). */
    R4D_DYN_LDS(int, s_dyn);
    int* const s_list = s_dyn;
    int* const s_ent = s_dyn + cap;
    /* The claim gather's fill counter. */
    __shared__ int s_wc;

    const int tid = (int)threadIdx.x;

    /* ---- 1. the epoch ---- */
    if (tid == 0) {
        s_epoch = ctl[0] + 1;
        s_lookups = 0;
        s_claims = 0;
        s_hits = 0;
        s_distinct = 0;
        s_nmiss = 0;
        s_placed = 0;
        s_overflow = 0;
        s_free = 0;
        /* This call's miss count. A call that claims nothing returns it as zero below without
         * reaching the sweep, so it is cleared here rather than left as the last call's. */
        ctl[2] = 0;
    }
    __syncthreads();
    const int epoch = s_epoch;

    /* ---- 2. one lookup per entry; the map is its own set ---- */
    int lookups = 0, hits = 0;
    for (int q = 0; q < M; ++q) {
        int n = nsel[q];
        if (n < 0) n = 0;
        if (n > pitch) n = pitch;
        lookups += n;
        for (int i = tid; i < n; i += kThreads) {
            const size_t e = (size_t)q * pitch + i;
            const int b = sel[e];
            if ((unsigned)b >= (unsigned)blocks) continue;
            const int s = res[b];
            /* The value this lookup saw, for step 7: a hit's slot cannot move in this call (the
             * sweep passes every slot this pass stamps with the epoch), so only an entry that was
             * absent here needs the map read again. Header, "the entry cache". */
            if (ecache > 0) s_ent[e] = s;
            if (s >= 0 && s < slots) {
                /* The stamp is this call's mark on the slot and the union's counter at once: the
                 * exchange says whether this thread was the first to stamp it, so the ring is
                 * counted here instead of walked after the fact. */
                if (atomicExch(&slot_state[(size_t)s * sst_ld], epoch) != epoch) {
                    /* One bump a slot a call: the exchange says this thread was the first to stamp
                     * it, so a block twenty rows name gains one step of heat, not twenty. */
                    slot_state[(size_t)s * sst_ld + 1] =
                        kv_heat_up(slot_state[(size_t)s * sst_ld + 1]);
                    ++hits;
                }
            } else if (s == -1) {
                /* One CAS wins per block, so this is the distinct-claim count, and the claim list
                 * needs no walk over `res` to find them. */
                if (atomicCAS(&res[b], -1, -2) == -1) {
                    const int k = atomicAdd(&s_claims, 1);
                    if (k < cap) s_list[k] = b;
                }
            }
        }
    }
    /* The launch's hit count: a per-thread total, then the same warp-first reduction the union
     * ends with. */
    s_cnt[tid] = hits;
    __syncthreads();
    if (tid < 32) {
        int t = 0;
        for (int i = tid; i < kThreads; i += 32) t += s_cnt[i];
        s_off[tid] = t;
    }
    __syncthreads();
    if (tid == 0) {
        int used = 0;
        for (int t = 0; t < 32; ++t) used += s_off[t];
        s_hits = used;
        s_lookups = lookups;
        s_distinct = s_claims;
    }
    __syncthreads();

    /* ---- 3. the canonical miss order: ascending block id ---- */
    const int claims = s_distinct;
    const int want = claims < slots ? claims : slots;
    if (claims > 0) {
        if (claims <= cap) {
            /* The list holds every claim, so sorting it says exactly what walking `res` said --
             * ascending block id -- over `claims` elements instead of `blocks`. */
            kv_sort_claims(s_list, claims, tid);
            for (int i = tid; i < want; i += kThreads) miss_block[i] = s_list[i];
        } else {
            /* Above the list: the launch is wide enough that `M * sel_pitch` -- the claim ceiling
             * it declares -- passes the workgroup's LDS room, so the list is not held in shared
             * memory. The claims are gathered from the selection into the row's scratch instead and
             * sorted there, over `lookups` entries: the same ids, the same ascending order, and
             * still nothing about `blocks` in the cost. Only a call that fits its ring takes this
             * path; a call with more claims than the ring has slots cannot be listed at all and
             * keeps the map walk below, because a sorted list of the smallest `slots` claims has to
             * come from the whole claim set. */
            if (claims <= slots) {
                if (tid == 0) s_wc = 0;
                __syncthreads();
                for (int q = 0; q < M; ++q) {
                    int n = nsel[q];
                    if (n < 0) n = 0;
                    if (n > pitch) n = pitch;
                    for (int i = tid; i < n; i += kThreads) {
                        const int b = sel[(size_t)q * pitch + i];
                        if ((unsigned)b >= (unsigned)blocks) continue;
                        /* One claim per distinct block: the map is the set, so a block four rows
                         * name enters the list once. -3 is "collected", restored below. */
                        if (atomicCAS(&res[b], -2, -3) == -2) {
                            const int k = atomicAdd(&s_wc, 1);
                            if (k < slots) miss_block[k] = b;
                        }
                    }
                }
                __syncthreads();
                for (int q = 0; q < M; ++q) {
                    int n = nsel[q];
                    if (n < 0) n = 0;
                    if (n > pitch) n = pitch;
                    for (int i = tid; i < n; i += kThreads) {
                        const int b = sel[(size_t)q * pitch + i];
                        if ((unsigned)b >= (unsigned)blocks) continue;
                        if (res[b] == -3) res[b] = -2;
                    }
                }
                __syncthreads();
                /* `claims` ids, padded to a power of two: the row's scratch is sized for exactly
                 * that (r4d_rad_kv_resolve_scratch). */
                kv_sort_claims(miss_block, claims, tid);
            } else {
                /* More claims than the ring can hold: this call has overflowed whatever it places,
                 * so the sweep stops at its first unplaceable miss and `ctl[3]` latches. The list
                 * here is `blocks`-wide because the ascending order of the WHOLE claim set is the
                 * contract, and with no structure able to hold it the map is where it lives. A legal
                 * call -- one the ring can hold -- never reaches this. */
                const int chunk = (blocks + kThreads - 1) / kThreads;
                const int lo = (int)((long long)tid * chunk);
                const int hi = lo + chunk < blocks ? lo + chunk : blocks;
                const bool vec4 = (lo & 3) == 0 && (hi & 3) == 0 && ((uintptr_t)res & 15u) == 0;
                int c = 0;
                if (vec4) {
                    const int4* const r4 = (const int4*)res;
                    for (int i = lo / 4; i < hi / 4; ++i) {
                        const int4 v = r4[i];
                        c += (v.x == -2) + (v.y == -2) + (v.z == -2) + (v.w == -2);
                    }
                } else {
                    for (int b = lo; b < hi; ++b) c += (res[b] == -2);
                }
                s_cnt[tid] = c;
                s_off[tid] = c;
                __syncthreads();
                for (int step = 1; step < kThreads; step <<= 1) {
                    const int v = tid >= step ? s_off[tid - step] : 0;
                    __syncthreads();
                    s_off[tid] += v;
                    __syncthreads();
                }
                __syncthreads();
                int pos = s_off[tid] - s_cnt[tid];
                if (vec4) {
                    const int4* const r4 = (const int4*)res;
                    for (int i = lo / 4; i < hi / 4; ++i) {
                        const int4 v = r4[i];
                        if (v.x == -2 && pos < slots) miss_block[pos++] = 4 * i;
                        if (v.y == -2 && pos < slots) miss_block[pos++] = 4 * i + 1;
                        if (v.z == -2 && pos < slots) miss_block[pos++] = 4 * i + 2;
                        if (v.w == -2 && pos < slots) miss_block[pos++] = 4 * i + 3;
                    }
                } else {
                    for (int b = lo; b < hi; ++b)
                        if (res[b] == -2 && pos < slots) miss_block[pos++] = b;
                }
                __syncthreads();
            }
        }
        if (tid == 0) {
            /* The list is the first `want` claims in ascending order whichever path built it. */
            s_nmiss = want;
            s_overflow = claims > slots;
            s_placed = 0;
        }
        __syncthreads();
    }

    /* ---- 3b. how many slots are free ---- */
    /* A free slot is claimed before any resident block is evicted (phase 4), and this is what says
     * whether there is one to claim: a call whose ring is full -- where every slot holds a block --
     * must not pay a turn of scanning per miss to discover it, and above that a full ring never
     * becomes free again during the call. Counting in parallel costs one coalesced pass over the
     * two slot maps; the sweep then reads the count and, when it is zero, goes straight to CLOCK.
     * A slot this call stamped is not free whatever it holds: it is the call's own. */
    if (claims > 0) {
        int nf = 0;
        for (int s = tid; s < slots; s += kThreads)
            if (slot_block[s] < 0 && slot_state[(size_t)s * sst_ld] != epoch) ++nf;
        if (nf != 0) atomicAdd(&s_free, nf);
    }
    __syncthreads();

    /* ---- 4. one victim per miss: the block's own slot first, else a free slot, else CLOCK ---- */
    /* Identity first: a block whose id fits the ring and whose own slot is free goes there, so
     * res[b] = b for a context shorter than the ring. The attention's gather is a long run of
     * adjacent block ids, and a layer's slots are 2 KiB apart inside one stripe, so identity turns
     * that run into a sequential walk of one stripe instead of the order the blocks entered the
     * selection.
     *
     * It is only a placement preference, never a replacement one: it takes a slot only when that
     * slot is free, and it does not move the hand, so a ring that cannot give identity the slot
     * falls through to the walk below with the hand exactly where it was. The fallback is free slot
     * first, then CLOCK, so a depth where most ids are past `slots` places the same blocks in the
     * same slots.
     *
     * A free slot (nothing resident in it) is not a victim at all: taking it costs no block, so
     * every free slot is claimed before a live one is even considered. CLOCK's second chance over
     * the heat counter is what a ring with no free slot falls back to, and the free scan has no
     * side effect when it finds nothing (it halves no counter and leaves the hand where it
     * started), so a full ring takes exactly the victim the sweep below names.
     *
     * `free_left` is the call's free-slot count from phase 3b, decremented as the sweep fills one.
     * Once it is zero nothing can free a slot -- a placement makes a free slot live and an eviction
     * re-points a live one -- so a full ring pays the count once, not a scan per miss. */
    if (tid == 0 && claims > 0) {
        int hand = ctl[1];
        /* A hand outside the ring is a state fault libref refuses; start at the beginning
         * rather than address outside it. */
        if (hand < 0 || hand >= slots) hand = 0;
        int placed = 0;
        bool overflow = s_overflow;
        int free_left = s_free;
        const int nmiss = s_nmiss;
        for (int k = 0; k < nmiss; ++k) {
            const int b = miss_block[k];
            int victim = -1;
            /* Identity: the block's own slot, if the id fits the ring and the slot is free (or
             * already holds it). Leaving the hand alone is what keeps the fallback identical. */
            if (b < slots) {
                const int want = b;
                if (slot_state[(size_t)want * sst_ld] != epoch &&
                    (slot_block[want] < 0 || slot_block[want] == b))
                    victim = want;
            }
            /* Free first, in the same hand order CLOCK walks, so a ring with frees to give
             * places every miss identically in both implementations. A stamp of this call's epoch
             * is never taken, free or not: it is a slot the call itself already uses. */
            if (victim < 0 && free_left > 0) {
                int h = hand;
                for (int scanned = 0; scanned < slots; ++scanned) {
                    const int s = h;
                    h = h + 1 == slots ? 0 : h + 1;
                    if (slot_state[(size_t)s * sst_ld] == epoch) continue;  /* used by this call */
                    if (slot_block[s] >= 0) continue;                       /* holds a block */
                    victim = s;
                    hand = h;
                    break;
                }
                if (victim < 0) free_left = 0;
            }
            if (victim < 0) {
                for (long long scanned = 0; scanned < 3LL * slots; ++scanned) {
                    const int s = hand;
                    hand = hand + 1 == slots ? 0 : hand + 1;
                    if (slot_state[(size_t)s * sst_ld] == epoch) continue;  /* used by this call */
                    const int old = slot_block[s];
                    if (old >= 0 && old < blocks &&
                        slot_state[(size_t)s * sst_ld + 1] > 0) {            /* second chance */
                        slot_state[(size_t)s * sst_ld + 1] =
                            kv_heat_down(slot_state[(size_t)s * sst_ld + 1]);
                        continue;
                    }
                    victim = s;
                    break;
                }
            }
            if (victim < 0) { overflow = true; break; }
            /* ---- 5. re-point both maps ---- */
            const int s = victim;
            const int old = slot_block[s];
            if (old < 0 && free_left > 0) --free_left;   /* a free slot the sweep just filled */
            if (old >= 0 && old < blocks) res[old] = -1;
            slot_block[s] = b;
            slot_state[(size_t)s * sst_ld] = epoch;
            slot_state[(size_t)s * sst_ld + 1] =
                kv_heat_up(slot_state[(size_t)s * sst_ld + 1]);
            res[b] = s;
            miss_slot[k] = s;
            ++placed;
        }
        ctl[1] = hand;
        ctl[2] = placed;
        if (overflow) ctl[3] = 1;
        s_placed = placed;
    }
    __syncthreads();

    /* A miss the sweep could not place is not resident: undo every remaining claim, so the next
     * call may claim them again. A `res` entry still -2 here is by definition such a claim (a
     * placed one holds its slot), and every claim is named by at least one selection entry, so the
     * selection reaches all of them: `lookups` entries instead of `blocks` map entries, and the
     * write is idempotent, so an entry repeating a block is harmless. The list is the shorter pass
     * when it holds every claim, which is the normal case. */
    if (claims > 0) {
        if (s_distinct <= slots) {
            for (int k = tid; k < s_distinct; k += kThreads)
                if (res[miss_block[k]] == -2) res[miss_block[k]] = -1;
        } else {
            for (int q = 0; q < M; ++q) {
                int n = nsel[q];
                if (n < 0) n = 0;
                if (n > pitch) n = pitch;
                for (int i = tid; i < n; i += kThreads) {
                    const int b = sel[(size_t)q * pitch + i];
                    if ((unsigned)b >= (unsigned)blocks) continue;
                    if (res[b] == -2) res[b] = -1;
                }
            }
        }
    }
    __syncthreads();

    /* ---- 6. the copies, one block each, in miss order, in this launch ---- */
    /* One warp a miss, so every miss's host read is in flight at once and each warp's lanes are
     * independent. One 2 KiB copy out of the host mirror is PCIe transactions at ~1 us of latency
     * each, and a decode launch takes tens of them: a loop that waits out one miss at a time, or a
     * flattened loop that re-reads `miss_block`/`miss_slot` from the scratch for every lane, spends
     * the launch waiting. The copies are independent -- each writes its own slot -- so the order
     * between them is the miss order only for the record.
     *
     * Four loads per thread, issued before any of them is stored from. A wave issues in order and a
     * store waits on its load, so `dst[i] = src[i]` at a stride of 32 makes the next iteration's load
     * wait behind the previous iteration's store and the 2 KiB copy pays the mirror's read latency
     * four times over instead of once (the trip count is `block_bytes`, a runtime value, so the
     * compiler cannot unroll it). Loading the four vectors of one thread's quarter of the block
     * first puts four requests in flight, and a block is
     * a whole number of them at the declared geometry (2 KiB = 128 vectors = one full turn of the
     * group), so the common path never takes the tail. Four more VGPRs: the kernel goes 32 -> 36
     * with no spill, and it runs one workgroup by design.
     *
     * The vectors are `kv_v16`, not `uint4`: that class's assignment is a 16-byte memcpy call, and a
     * memcpy is a load and a store that are adjacent by construction however the four are written in
     * the source, which is the serialised shape this loop exists to avoid. Builtin vectors leave the
     * four as four loads and four stores, which is what the scheduler needs. Same bytes either way. */
    const int placed = s_placed;
    const int lane = tid & 31;
    if (bbytes >= 16 && bbytes % 16 == 0) {
        const int vec = bbytes / 16;
        for (int k = tid >> 5; k < placed; k += kThreads / 32) {
            const kv_v16* __restrict__ const src =
                (const kv_v16*)host + (size_t)miss_block[k] * vec;
            kv_v16* __restrict__ const dst = (kv_v16*)cache + (size_t)miss_slot[k] * vec;
            int i = lane;
            for (; i + 96 < vec; i += 128) {
                const kv_v16 a = src[i], b = src[i + 32], c = src[i + 64], d = src[i + 96];
                dst[i] = a;
                dst[i + 32] = b;
                dst[i + 64] = c;
                dst[i + 96] = d;
            }
            /* A block that is not a whole number of four-wide turns -- `block_bytes` not a multiple
             * of 2048 -- finishes at the original one-vector stride, so every lane sequence is
             * covered exactly once whatever the pitch. */
            for (; i < vec; i += 32) dst[i] = src[i];
        }
    } else {
        for (int k = tid >> 5; k < placed; k += kThreads / 32) {
            const uint8_t* src = host + (size_t)miss_block[k] * bbytes;
            uint8_t* const dst = cache + (size_t)miss_slot[k] * bbytes;
            for (int i = lane; i < bbytes; i += 32) dst[i] = src[i];
        }
    }

    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    for (int q = 0; q < M; ++q) {
        int n = nsel[q];
        if (n < 0) n = 0;
        if (n > pitch) n = pitch;
        for (int i = tid; i < n; i += kThreads) {
            const size_t e = (size_t)q * pitch + i;
            const int b = sel[e];
            if ((unsigned)b >= (unsigned)blocks) continue;
            /* A hit's slot is the one step 2 read; anything absent then (-1, -2, -3) only the map
             * knows where the sweep placed it. Same index, same thread, so no barrier is needed. */
            int s = ecache > 0 ? s_ent[e] : -1;
            if (s < 0) s = res[b];
            sel[e] = s >= 0 ? s : 0;
        }
    }

    /* The distinct blocks this call touched, for the union the caller reports: the slots this call
     * hit or filled (counted in phase 2 and by the sweep) plus the claims it could not place. Both
     * halves are known by now -- distinct hit slots and distinct claimed blocks -- and an entry
     * count would overcount them, because many of a launch's rows share swaths of the history.
     * Nothing walks the ring for it. */
    if (tid == 0) {
        const int placed = s_placed;
        const int uni = s_hits + s_distinct;
        if (uni > ctl[10]) ctl[10] = uni;
        if (placed > ctl[11]) ctl[11] = placed;
        /* The depth the claim list had to hold, which is what the next size of it is derived from. */
        if (s_distinct > ctl[12]) ctl[12] = s_distinct;
        uint64_t* const ctr = (uint64_t*)(ctl + 4);
        ctr[0] += (uint64_t)placed;
        ctr[1] += (uint64_t)s_lookups;
        ctr[2] += 1u;
        ctl[0] = epoch;
    }
}

}  // namespace

/* `sel`         [M][sel_pitch]     i32, block ids in, ring slots out
 * `nsel`        [M]                i32, valid entries of each row
 * `res`         [blocks]           i32, block -> slot, -1 absent, -2 claimed by this call
 * `slot_block`  [slots]            i32, slot -> block, -1 free
 * `slot_state`  [slots][sst_ld]    i32, [s][0] epoch of last use, [s][1] two-bit heat counter
 * `ctl`         [16]               i32, [0] epoch, [1] hand, [2] placed, [3] overflow, u64 from [4]
 * `host`        [blocks][bbytes]   u8, the layer's host mirror
 * `cache`       [slots][bbytes]    u8, the layer's device ring
 * `miss_block`, `miss_slot`        the row scratch, `next_pow2(slots)` ids then `slots` slots
 * `cap`                            the claim list's capacity, from kv_claim_cap
 * `ecache`                         the entry cache's capacity in ints: `M * sel_pitch`, or 0 when
 *                                  the claim list leaves no room for it (header, "the entry cache")
 *
 * The two planes are addressed by `bbytes` alone, so the shim has already checked that each
 * block's row pitch in bytes is exactly `block_bytes` and that its trailing extents fill it. */
static int r4d_kv_resolve_emit(long sel, long nsel, long res, long slot_block, long slot_state,
                               long ctl, long host, long cache, long miss_block, long miss_slot,
                               int M, int blocks, int slots, int pitch, int bbytes, int sst_ld,
                               RadStream stream) {
    if (!sel || !nsel || !res || !slot_block || !slot_state || !ctl || !host || !cache ||
        !miss_block || !miss_slot)
        return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || blocks <= 0 || slots <= 0 || pitch <= 0 || bbytes <= 0 || sst_ld < 2)
        return R4D_NO(RAD_E_SHAPE);
    /* The list is sized from this launch's own operands. A workgroup with no LDS room for even one
     * id cannot run the op at all, which is a shape the caller cannot satisfy rather than a slow
     * launch: refuse it. */
    int room = 0;
    const int cap = kv_claim_cap(M, pitch, &room);
    if (cap < 1) return R4D_NO(RAD_E_SHAPE);
    /* One int a selection entry for the entry cache (the header), while the room the claim list
     * leaves holds them all: a decode launch caches its 513, a prefill this wide does not and step 7
     * reads the map again. */
    const long long entries = (long long)M * pitch;
    const int ecache = entries <= (long long)(room - cap) ? (int)entries : 0;
    return R4D_EMIT(stream, kv_resolve_kernel, dim3(1), dim3(kThreads),
                    (uint32_t)(cap + ecache) * sizeof(int),
                    (int32_t*)sel, (const int32_t*)nsel, (int32_t*)res, (int32_t*)slot_block,
                    (int32_t*)slot_state, (int32_t*)ctl, (const uint8_t*)host, (uint8_t*)cache,
                    (int32_t*)miss_block, (int32_t*)miss_slot, M, blocks, slots, pitch,
                    bbytes, sst_ld, cap, ecache);
}

/* ============================================================== the plugin ABI entry point */

enum { KR_SEL = 0, KR_NSEL = 1, KR_RES = 2, KR_SBLK = 3, KR_SST = 4, KR_CTL = 5, KR_HOST = 6,
       KR_CACHE = 7 };

/* Fan a byte extent into the shape's trailing elements: one block a row, dense, so the row pitch
 * in bytes and the product of the trailing extents' element bytes both equal `block_bytes`. */
static int r4d_kv_block_row_bytes(const RadTensor* t, long long* out) {
    long long acc = 1;
    for (unsigned d = 1; d < t->rank; ++d) acc *= t->shape[d];
    *out = acc * (long long)rad_dtype_bytes(t->dtype, 1);
    return t->stride[0] * (long long)rad_dtype_bytes(t->dtype, 1) == *out ? RAD_OK : RAD_E_STRIDE;
}

extern "C" int r4d_rad_kv_resolve(const RadArgs* a, RadStream stream) {
    const RadTensor* se = r4d_opt(a, KR_SEL);
    const RadTensor* ns = r4d_opt(a, KR_NSEL);
    const RadTensor* re = r4d_opt(a, KR_RES);
    const RadTensor* sb = r4d_opt(a, KR_SBLK);
    const RadTensor* st = r4d_opt(a, KR_SST);
    const RadTensor* ct = r4d_opt(a, KR_CTL);
    const RadTensor* ho = r4d_opt(a, KR_HOST);
    const RadTensor* ca = r4d_opt(a, KR_CACHE);
    if (!se || !ns || !re || !sb || !st || !ct || !ho || !ca)
        return R4D_NO(RAD_E_INVAL);

    /* `M` is a RANGED parameter, so the resolved geometry carries the band's ceiling (max_tok),
     * not this launch's rows: a 625-token prefill and a one-row decode both arrive with 2048 here.
     * The operand names the call, as arch/common/rad_arch.h's "step-sized operands" rule says, so
     * the row count is `sel`'s own extent and the parameter is only the ceiling it may not pass. */
    const long long band  = r4d_geti_or(a, "M", 0);
    const long long M     = se->shape[0];
    const long long blocks = r4d_geti_or(a, "blocks", 0);
    /* The ring depth is the operand's, for the same reason: the pool is planned after declare, so
     * the declared `slots` is the context bound and a ring the engine sized smaller
     * (--kv-resident) arrives as a shorter `slot_block`. The parameter is the ceiling a
     * call may not pass (libref/ref_kvstream.cpp, "The parameters"). */
    const long long slots_decl = r4d_geti_or(a, "slots", 0);
    const long long slots = sb->shape[0];
    /* The row pitch is the operand's too. Reading it from `sel_pitch` would let a view whose rows
     * are wider or narrower than its declared geometry be rewritten into its neighbour. */
    const long long declared_pitch = r4d_geti_or(a, "sel_pitch", 0);
    const long long pitch = r4d_row_pitch(se);
    const long long bbytes = r4d_geti_or(a, "block_bytes", 0);
    if (M <= 0 || blocks <= 0 || slots <= 0 || slots > slots_decl || pitch <= 0 || bbytes <= 0)
        return R4D_NO(RAD_E_SHAPE);
    if (M > INT32_MAX || blocks > INT32_MAX || slots > INT32_MAX || pitch > INT32_MAX ||
        bbytes > INT32_MAX)
        return R4D_NO(RAD_E_SHAPE);
    /* A launch above the band was resolved against constraints it does not satisfy. */
    if (band > 0 && M > band) return R4D_NO(RAD_E_SHAPE);
    if (declared_pitch > 0 && pitch != declared_pitch) return R4D_NO(RAD_E_STRIDE);

    /* The map and the counts are one format, written by one op and read by the writer and the
     * sweep: i32, dense. */
    const RadTensor* state[] = { se, ns, re, sb, st, ct };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (se->rank != 2 || st->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (se->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ns) < M || ns->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(re) < blocks || re->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (sb->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (st->shape[0] < slots || st->stride[1] != 1 || st->stride[0] < 2)
        return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ct) < 16 || ct->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (ho->rank < 2 || ca->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (ho->shape[0] < blocks || ca->shape[0] < slots) return R4D_NO(RAD_E_SHAPE);

    /* One block of one layer: the mirror's row and the pool's slot are the same bytes, and the
     * copy length `block_bytes` is both pitches. The pool arrives as a KV group's cache, so its
     * dtype is the cache layout's, not u8; only the byte pitch is fixed. */
    long long hdense = 0, pdense = 0;
    if (r4d_kv_block_row_bytes(ho, &hdense) != RAD_OK || hdense != bbytes)
        return R4D_NO(RAD_E_STRIDE);
    if (r4d_kv_block_row_bytes(ca, &pdense) != RAD_OK || pdense != bbytes)
        return R4D_NO(RAD_E_STRIDE);
    /* The counters are read as u64 in an i32 block. */
    if (((uintptr_t)ct->data & 7u) != 0) return R4D_NO(RAD_E_STRIDE);

    /* The row's scratch: `nlist` ids (the miss list, padded to a power of two when a wide launch
     * sorts it there), then `slots` slots. */
    const int64_t nlist = kv_pow2ceil(slots);
    const long long need = (nlist + slots) * 4;
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_INVAL);
    int32_t* const miss_block = (int32_t*)a->scratch;
    int32_t* const miss_slot = (int32_t*)((char*)a->scratch + nlist * 4);

    return r4d_kv_resolve_emit(r4d_addr(se), r4d_addr(ns), r4d_addr(re), r4d_addr(sb),
                               r4d_addr(st), r4d_addr(ct), r4d_addr(ho), r4d_addr(ca),
                               (long)(uintptr_t)miss_block, (long)(uintptr_t)miss_slot,
                               (int)M, (int)blocks, (int)slots, (int)pitch, (int)bbytes,
                               (int)st->stride[0], stream);
}

/* The miss list -- `next_pow2(slots)` block ids, because a launch too wide for the shared claim list
 * sorts the list in this scratch -- then `slots` slots (the header above). */
extern "C" int64_t r4d_rad_kv_resolve_scratch(const RadArgs* a) {
    const long long slots = r4d_geti_or(a, "slots", 0);
    if (slots <= 0 || slots > INT32_MAX) return 0;
    return (kv_pow2ceil(slots) + slots) * 4;
}
