/* ref_kvstream.cpp -- kv_resolve: the residency map between a streamed KV group's block ids and
 * the ring slots its device pool actually holds (docs/KV-STREAMING.md §3.4 and §4.3, docs/OPS.md
 * "Sparse attention (QSA)").
 *
 * ============================== What it sits between ==============================
 *
 * `qsa_select` already dereferences the attention group's block table: it writes
 * `sel[q][i] = tb[b]`, the group's POOL PAGE ID (`r4d_qsa_select_bf16.hip:254`, `:686`), and
 * `attn_paged` uses each entry as a raw pool page index with no indirection
 * (`r4d_attn_decode_h256_gqa6.hip:268`, `:322`; `r4d_attn_paged_gqa8.hip`). When the group's KV
 * streams, the device pool is `slots` blocks deep and the block-id space is `blocks`, so a block id
 * is no longer an address: `blk * kv_block_stride` would walk out of the layer and silently read
 * another layer's bytes. `kv_resolve` turns every selected block id into the ring slot holding it,
 * copies the block in from the pinned host mirror when it is not resident, and rewrites `sel` in
 * place. Nothing below the rewrite changes: `sel` is the only place the sparse path names KV
 * storage, so `attn_paged` reads a resolved table exactly as it reads a fully resident one.
 *
 * `nsel` and `seqused` stay what `qsa_select` wrote them: the attention's length over the table is
 * a token count inside the blocks, not an address, so the rewrite leaves both alone.
 *
 * ============================== The operands, exactly ==============================
 *
 *   #  name         role   dtype  shape                       what it is
 *   0  sel          inout  i32    [M, sel_pitch]              block ids in, ring slots out
 *   1  nsel         in     i32    [M]                        valid entries of each row
 *   2  res          inout  i32    [blocks]                   block -> slot, or -1; -2 in flight
 *   3  slot_block   inout  i32    [slots]                    slot -> block, or -1 free
 *   4  slot_state   inout  i32    [slots, 2]                 [s,0] epoch of last use, [s,1] heat
 *   5  ctl          inout  i32    [16]                       the control block (below)
 *   6  host         in     u8     [blocks, block_bytes]      the layer's host mirror, identity layout
 *   7  kv_cache     inout  bytes  [slots, block_bytes]       the layer's ring, as RAD_KV(group, layer)
 *
 * Order follows the read order of the algorithm, with the cache last as in `kv_store`. `sel` and
 * `nsel` keep `qsa_select`'s positions 5 and 6 relative to each other, so the pair reads the same
 * way in both ops.
 *
 * **The miss list is not an operand.** It is a per-call temporary, and a temporary is the row's
 * scratch in this codebase (`qsa_select`'s split workspace, docs/OPS.md "qsa_select"): an operand
 * would enter the arena planner, the recorded pass's ranges and the tape key for bytes nothing
 * outside one launch reads. A device row that needs one declares
 * `(next_pow2(slots) + slots) * 4` bytes through `RadKernelInfo::scratch`, laid out
 * `int32 miss_block[next_pow2(slots)]` -- the ascending miss list, padded because a launch too wide
 * for the kernel's shared claim list sorts it here -- then `int32 miss_slot[slots]`; this reference
 * sorts locally and needs none.
 *
 * ============================== The parameters ==============================
 *
 *   M            query rows to resolve; ranged, since a prefill chunk and a decode step differ
 *   blocks       the group's logical block count (the id space, `res`'s extent)
 *   slots        the ring depth (the device pool's extent, `slot_block`'s extent)
 *   sel_pitch    the selection's row pitch in entries (topk + 1); also the bound `nsel` may not pass
 *   block_bytes  bytes one block of ONE layer occupies: the host mirror's row pitch, the pool's slot
 *                pitch, and the length of every copy
 *
 * `block_bytes` rather than `block_size` (the block's token count) because the op moves bytes and
 * never reads an element: it is the same code at every cache layout (`fp8`, `fp8_v4`, `k6_v4`,
 * `bf16`), and it is the number a vectorised copy has to constrain (a block is a whole number of
 * 16-byte lanes). It is deliberately NOT a `block_size` key: `Builder::find_kv_consumer` binds a KV
 * group to the first unclaimed op after it that carries `block_size`, and the group's block size
 * must keep coming off the resolved attention kernel, not off the op that only copies its pages.
 *
 * `sel_pitch` and `block_bytes` are also pitches the operands carry (`sel.stride[0]`,
 * `host.stride[0]`), and this reference requires them to agree: a resolve that read one pitch and
 * wrote another would rewrite a row into its neighbour. Stating them in the schema puts them in the
 * resolved geometry, where a kernel may constrain them, and in the shape hook, which needs them to
 * size `sel` and the two planes.
 *
 * `slots` is the ring depth, and the OPERAND states it, exactly as `sel`'s extent states `M` and
 * its row pitch states `sel_pitch`: `slot_block`'s extent is the ring the call may address and the
 * parameter is the ceiling the caller declared. Declare cannot know the ring -- the pool is planned
 * after the program is built, and a plugin sees only the context bound -- so the declared `slots` is
 * the context and a ring the engine sized smaller (--kv-resident) arrives as a shorter
 * operand. A call may never address more than the declaration: `slots > slots_declared` is refused.
 *
 * ============================== The control block ==============================
 *
 * `ctl` is 16 int32s, the same [0..9] layout the reference design uses
 * (`Strata-main/include/strata/kernels/kv_stream.hpp:78`), 8-byte aligned so the counters can be
 * read as u64:
 *
 *   [0]  epoch        the value the last call used (0 in a fresh block, so the first call is 1)
 *   [1]  hand         the CLOCK hand, in [0, slots)
 *   [2]  misses       blocks placed by the last call
 *   [3]  overflow     1 once a call could not place every miss; latched, never cleared here
 *   [4..5]  u64 misses   every block ever placed
 *   [6..7]  u64 lookups  every selection entry ever resolved
 *   [8..9]  u64 calls
 *   [10] maximum union  the largest number of DISTINCT blocks one call has selected
 *                        (hits + misses), so a ring's depth can be sized from measurement
 *   [11] maximum misses the largest number of blocks one call has placed
 *   [12] maximum claims the largest number of DISTINCT absent blocks one call has claimed: the
 *                        length of the sorted miss list, so it is what a claim-list capacity has
 *                        to hold. One call's claims are bounded by its lookups, `sum_q nsel[q] <=
 *                        M * sel_pitch` (every claim is a selection entry), and the engine's
 *                        worst case is `max_seqs * sel_pitch`; a device kernel sizes its list from
 *                        those operands rather than a constant (libr4d/r4d_kv_resolve.hip), and
 *                        this is the counter that says whether it did.
 *   [13..15] zero, reserved
 *
 * `res`, `slot_block` and `slot_state` are state as well, and the caller brings them up: `res` and
 * `slot_block` FILLED WITH -1, `slot_state` and `ctl` ZEROED (one `fill` each, or the group's
 * bring-up). That is the reset: there is no separate reset op, because a residency map is fixed at
 * bring-up and lives as long as the pool does. Entering a call with any `res` entry of -2, a
 * `slot_block` outside [-1, blocks), a `res` outside [-1, slots) or a hand outside [0, slots) is a
 * state fault and is refused rather than guessed at. The epoch is a wrap-around i32 counter, only
 * ever compared for equality; it would take 2^31 calls - over two years of decoding - for a stale
 * stamp to read as a fresh one, and `slot_state[s,1]` would still be the sweep's second chance.
 *
 * `slot_state[s,1]` is a **two-bit heat counter in [0, 3]** inside an i32 (31 bits unused, so the
 * policy needs no new field, no extra per-slot memory and no ABI change). It is bumped by one, and
 * clamped at 3, on every use of the slot -- a hit or a placement -- once a slot a call, and halved
 * when the sweep passes the slot. Zeroed at bring-up and written nowhere else, so a fresh ring
 * starts cold and a negative value is not reachable from a legal state; a reader treats it as cold
 * rather than letting a fault strand the sweep.
 *
 * ============================== The semantics, in order ==============================
 *
 *  1. **epoch bump.** epoch = ctl[0] + 1.
 *  2. **one lookup per distinct block.** For row q ascending, then entry i ascending, over
 *     i in [0, nsel[q]) (entries at or past nsel[q] are `qsa_select`'s -1 padding and are not
 *     touched): b = sel[q][i].
 *       res[b] >= 0  -> a hit: slot_state[res[b],0] = epoch and, the first time this call stamps
 *                       that slot, slot_state[res[b],1] = min(old + 1, 3). One bump a SLOT a
 *                       call, not one an entry: a block twenty rows name gains one step of heat,
 *                       the same as one a single row names, so the counter measures the calls
 *                       that wanted the block and not the width of the selection.
 *       res[b] == -1 -> a miss, claimed once: res[b] = -2 and b appended to the miss list.
 *       res[b] == -2 -> already claimed by an earlier entry; nothing. This is what makes the pass
 *                       one lookup per block without a second structure: the map is its own set.
 *  3. **the miss list is ascending block id.** Not first-appearance order: a block claimed by row 3
 *     must not evict a different slot from a block claimed by row 0 depending on which thread got
 *     there first, or two implementations of one contract would place the same blocks in different
 *     slots and `sel` would differ. Ascending is a fact about the claim set, not about the map, so
 *     a device implementation orders the claim list itself (a sort of at most `M * sel_pitch` ids);
 *     nothing here needs a pass over `blocks`, which is the id space and not the call's size.
 *  4. **one victim per miss, in that order: identity, a free slot, then a CLOCK (second chance)
 *     sweep.** For each missed b in ascending order, if `b < slots` and slot b is available --
 *     `slot_state[b,0] != epoch` and `slot_block[b]` is either -1 or b itself -- the victim is slot
 *     b, and the hand does not move. This is a placement preference only: it takes a slot only when
 *     that slot is free, so it can never evict, and leaving the hand alone means a miss it cannot
 *     place falls through to exactly the walk below, from exactly the hand it would have seen. Two
 *     things make it worth having: it maps a block's id to its own slot while the context is
 *     shorter than the ring, and adjacent block ids then sit in adjacent slots of one layer's
 *     stripe, which is the order the attention's gather reads them in. Otherwise, visit slots from
 *     the hand, one at a time, wrapping. A FREE slot -- slot_block[s] == -1 -- is taken before any
 *     resident block is considered: evicting costs a copy on the next call and a free slot costs
 *     nothing, so the ring fills its empty region first. A slot with slot_state[s,0] == epoch was
 *     claimed or hit by this call and is passed, free or not, and keeps its heat counter; this is
 *     the "never take a slot this call itself uses" rule. When a whole turn finds no free slot, the
 *     same walk is the CLOCK sweep, with the two-bit heat counter in place of the one-bit
 *     reference:
 *       - a slot holding a block whose counter is non-zero: halve it (old >> 1) and pass it. A
 *         saturated 3 survives exactly two passes (3 -> 1 -> 0), a freshly used 1 exactly one,
 *         and a slot nothing has wanted for two sweeps is the victim on the next visit.
 *       - otherwise (a block, counter 0): take it.
 *     The hand still PICKS the victim -- this is a per-visit decision, not a search -- so the
 *     policy needs no reduction, no barrier and no extra traffic: an add-and-clamp where a
 *     constant store was, and a shift where a zero was. That is the whole reason it is affordable
 *     where an exact LRU is not: the epoch-ordered minimum costs a collective reduction per miss.
 *     The hand advances one slot per visit, so it ends just past the last slot taken. How many
 *     slots are free is counted before the sweep; a placement into a free slot makes it live and an
 *     eviction re-points a live one, so the count only falls to zero and a full ring goes straight
 *     to CLOCK. The free
 *     scan is side-effect free when it finds nothing -- it halves no counter and leaves the hand
 *     where it started -- so a full ring's victim is the sweep's alone. A free slot is taken
 *     without regard to its stale counter. The CLOCK sweep may
 *     visit at most 3 * slots slots; every write clamps the counter to 3, so two passes halve any
 *     value to 0 (docs/KV-STREAMING.md §3.4) and the third places the miss.
 *  5. **re-point both maps** as each victim is taken: if the slot held block `old`, res[old] = -1;
 *     slot_block[s] = b, slot_state[s] = {epoch, min(old + 1, 3)}, res[b] = s. A placement is a
 *     use, so it carries the same bump a hit does; the victim's counter is 0 by construction (the
 *     sweep only takes a cold slot), so this is 1 in the common case, and the `min` is what keeps
 *     the two-bit range true for a slot taken by identity or from the free region.
 *  6. **copy**: for each placed (b, s), block_bytes from host[b] into the pool's slot s. A copy is
 *     a byte copy; nothing is converted, so a streamed block is bit-identical to the block the
 *     writer put in the mirror.
 *  7. **rewrite**: for every entry in [0, nsel[q]), sel[q][i] = res[b] when that is >= 0. Entries
 *     at or past nsel[q] keep their -1.
 *
 * `ctl[2] = placed`, `ctl[0] = epoch`, `ctl[1] = hand`, and the counters grow by `placed`, the
 * number of entries read (one a lookup, as phase 2 defines a lookup: per entry, not per distinct
 * block) and one. The three maxima are the sizes this op exists to measure: `ctl[10]` is the
 * largest union one call has selected -- the number a ring must hold -- `ctl[11]` the largest number
 * of blocks one call has placed, i.e. how much of that union was not already resident, and
 * `ctl[12]` the largest number of claims one call has made, which is the depth the miss list had to
 * hold.
 *
 * ============================== Overflow ==============================
 *
 * The ring must hold every DISTINCT block one call selects: a hit holds a slot and a miss needs
 * another, so the condition is `hits + misses <= slots`. What that union is depends on the launch:
 * at decode it is at most `max_seqs * (sel_pitch - 1)`, while above `qsa_exact_to` prefill issues a
 * row per query and a launch's union can be a chunk's worth of the history, which is why the engine
 * measures it (ctl[10]) instead of assuming a bound.
 *
 * It is not silent when it does: a miss left unplaced by the sweep reverts to `res[b] = -1`, and
 * `ctl[3]` is latched to 1 with `ctl[2]` counting only what was placed. The selection rewrite then
 * writes SLOT 0 for every entry whose block is not resident -- a safe address inside the ring, not
 * the block's slot. The entry cannot be left holding the block id: on device that value is an
 * address past the ring which `attn_paged` dereferences without a bounds check, so a step whose
 * resolve overflowed would fault or read another layer's bytes. Slot 0 is defined memory holding a
 * real block's KV, so the step produces wrong numbers rather than a fault -- and a caller that finds
 * `ctl[3]` set must DISCARD the step, which is the only reading that keeps the engine honest: it
 * never serves output computed against a table whose entries do not name their blocks.
 *
 * ============================== Statefulness ==============================
 *
 * The op keeps state across calls - the epoch, the hand, the heat counters and its counters - so it
 * is not concurrent: two of its launches must never overlap, and the engine must not reorder one
 * against another. That is the `rad_kernel_concurrent` promise (`abi/rad_abi.h:493`), which is
 * given per row; a plugin's row list lives beside it, and libr4d's is the `kStateful[]` prefix
 * table inside `rad_kernel_concurrent` (`libr4d/r4d_rows.cpp:8748-8770`, "A new row with
 * stream-ordered state belongs in the list above"). Both kernel rows for this op therefore stay
 * out of that list. The absence of the promise is always safe: without it the engine orders the
 * launch behind everything before it, which is exactly what the map needs.
 *
 * Being stateful is not the same as being unrecordable. The op is a pure function of
 * device-resident state whose addresses and pitches never change after bring-up (the maps, the
 * ring and the mirror are fixed-size from the group's declaration), so a recorded pass replays it
 * correctly; rewriting contents is fine, changing a pitch would force a re-record (docs/
 * KV-STREAMING.md §3.4, core/runtime/ctx.cpp's pass key).
 *
 * ============================== Why this file is the oracle ==============================
 *
 * It is the obvious serial form of the contract above: one pass over the entries, one sorted miss
 * list, one second-chance walk of the slots, one memcpy per miss, one rewrite. Two device
 * implementations compared against it must agree on which slot every block lands in, which is the
 * part a fast parallel sweep could get right on hits and wrong on victims.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* The heat counter's two writes, the same function in all three implementations: a use bumps by
 * one and saturates at 3, a sweep pass halves, and a negative value (never written by a legal call)
 * reads as cold rather than letting it survive every pass. */
int32_t heat_up(int32_t h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
int32_t heat_down(int32_t h) { return h > 0 ? (h >> 1) : 0; }

/* `ctl`'s size, and the index of its first counter (three u64s from [4]). */
enum { kKvCtlInts = 16, kKvCtlMisses = 4, kKvCtlLookups = 6, kKvCtlCalls = 8 };
static_assert(kKvCtlLookups == kKvCtlMisses + 2 && kKvCtlCalls == kKvCtlMisses + 4,
              "the three counters are consecutive u64s from ctl[4]");

/* The byte pitch of a rank-2-or-deeper operand's rows, and whether its trailing extents fill
 * exactly `bytes` densely. Both the mirror and the pool are one block a row; a padded row would
 * copy a block's bytes over its neighbour's. */
int64_t row_bytes(const RadTensor* t, int64_t* dense) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) acc *= t->shape[i];
    *dense = acc;
    return t->stride[0] * rad_dtype_bytes(t->dtype, 1);
}

}  /* namespace */

int ref_kv_resolve(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sel = &a->t[0], *nsel = &a->t[1], *res = &a->t[2], *sblk = &a->t[3];
    const RadTensor *sst = &a->t[4], *ctl = &a->t[5], *host = &a->t[6], *pool = &a->t[7];
    const int64_t M = p_int(a, "M", 0), blocks = p_int(a, "blocks", 0);
    const int64_t slots_decl = p_int(a, "slots", 0), pitch = p_int(a, "sel_pitch", 0);
    const int64_t bbytes = p_int(a, "block_bytes", 0);
    /* Zero rows is not a call (docs/OPS.md). */
    if (M <= 0) return RAD_E_SHAPE;
    if (blocks <= 0 || slots_decl <= 0 || pitch <= 0 || bbytes <= 0) return RAD_E_SHAPE;

    /* The map and the counts are one format, written by one op and read by the writer and the
     * sweep: i32, dense. */
    const RadTensor* state[] = { sel, nsel, res, sblk, sst, ctl };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return RAD_E_DTYPE;
    if (sel->rank != 2 || sst->rank != 2) return RAD_E_SHAPE;
    if (sel->shape[0] < M || sel->shape[1] < pitch) return RAD_E_SHAPE;
    if (sel->stride[1] != 1 || sel->stride[0] != pitch) return RAD_E_STRIDE;
    if (numel(nsel) < M || nsel->stride[0] != 1) return RAD_E_SHAPE;
    if (numel(res) < blocks || res->stride[0] != 1) return RAD_E_SHAPE;
    /* The operand states the ring; the parameter is the ceiling it may not pass (see "The
     * parameters"). */
    const int64_t slots = numel(sblk);
    if (slots < 1 || slots > slots_decl || sblk->stride[0] != 1) return RAD_E_SHAPE;
    if (sst->shape[0] < slots || sst->stride[1] != 1 || sst->stride[0] < 2) return RAD_E_STRIDE;
    if (numel(ctl) < kKvCtlInts || ctl->stride[0] != 1) return RAD_E_SHAPE;
    if (host->rank < 2 || pool->rank < 2) return RAD_E_SHAPE;
    if (host->shape[0] < blocks || pool->shape[0] < slots) return RAD_E_SHAPE;

    /* One block of one layer: the mirror's row and the pool's slot are the same bytes, and the
     * copy length `block_bytes` is both pitches. */
    int64_t hdense = 0, pdense = 0;
    if (row_bytes(host, &hdense) != bbytes || hdense * rad_dtype_bytes(host->dtype, 1) != bbytes)
        return RAD_E_STRIDE;
    if (row_bytes(pool, &pdense) != bbytes || pdense * rad_dtype_bytes(pool->dtype, 1) != bbytes)
        return RAD_E_STRIDE;

    int32_t* const R  = (int32_t*)res->data;
    int32_t* const SB = (int32_t*)sblk->data;
    int32_t* const ST = (int32_t*)sst->data;
    int32_t* const C  = (int32_t*)ctl->data;
    const int64_t stp = sst->stride[0];
    /* The counters are read and written as u64 in an i32 block. */
    if (((uintptr_t)C & 7u) != 0) return RAD_E_STRIDE;

    /* ---- everything that can refuse, before anything is written; the selection is read once and
     * kept, so phase 7 writes values it did not have to re-read ---- */
    std::vector<int64_t> off, ent;
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = ld_int(nsel->dtype, nsel->data, q);
        if (n < 0 || n > pitch) return RAD_E_SHAPE;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = ld_int(sel->dtype, sel->data, q * pitch + i);
            if (b < 0 || b >= blocks) return RAD_E_INVAL;
            off.push_back(q * pitch + i);
            ent.push_back(b);
        }
    }
    for (int64_t b = 0; b < blocks; ++b)
        if (R[b] < -1 || R[b] >= slots) return RAD_E_INVAL;   /* -2 is never left by a call */
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < -1 || SB[s] >= blocks) return RAD_E_INVAL;
    if (C[1] < 0 || C[1] >= slots) return RAD_E_INVAL;

    /* ---- 1. the epoch ---- */
    const int32_t epoch = (int32_t)((uint32_t)C[0] + 1u);

    /* ---- 2. hits, and one claim per distinct miss (the map is its own set) ---- */
    std::vector<int32_t> miss;
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t b = (int32_t)ent[e];
        const int32_t s = R[b];
        if (s >= 0) {
            /* One bump a slot a call (see "The semantics", step 2). */
            if (ST[s * stp] != epoch) {
                ST[s * stp] = epoch;
                ST[s * stp + 1] = heat_up(ST[s * stp + 1]);
            }
        } else if (s == -1) {
            R[b] = -2;
            miss.push_back(b);
        }
    }
    const int64_t lookups = (int64_t)ent.size();

    /* ---- 3. the canonical miss order ---- */
    std::sort(miss.begin(), miss.end());

    /* ---- 4. one victim per miss: identity, a free slot, then CLOCK, from the hand, never a slot
     * this call uses ---- */
    /* Identity first (see "The semantics", step 4): a missed block whose id fits the ring and whose
     * own slot is free takes that slot and leaves the hand alone, so the fallback below runs from
     * the hand it would always have run from. A free slot is not a victim (see step 4): the ring's
     * empty region fills before any resident block is evicted. The free scan halves no counter
     * and does not move the hand when it finds nothing, so a full ring takes the CLOCK victim it
     * always took. The count below is what lets a full ring go straight to CLOCK: no placement
     * frees a slot and no eviction does either, so a count of zero holds for the rest of the call. */
    int64_t free_left = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < 0 && ST[s * stp] != epoch) ++free_left;
    int64_t hand = C[1];
    int64_t placed = 0;
    bool overflow = false;
    for (size_t k = 0; k < miss.size(); ++k) {
        const int32_t b = miss[k];
        int64_t victim = -1;
        if ((int64_t)b < slots && ST[(int64_t)b * stp] != epoch && (SB[b] < 0 || SB[b] == b))
            victim = b;
        if (victim < 0 && free_left > 0) {
            int64_t h = hand;
            for (int64_t scanned = 0; scanned < slots; ++scanned) {
                const int64_t s = h;
                h = h + 1 == slots ? 0 : h + 1;
                if (ST[s * stp] == epoch) continue;                  /* used by this call */
                if (SB[s] >= 0) continue;                            /* holds a block */
                victim = s;
                hand = h;
                break;
            }
            if (victim < 0) free_left = 0;
        }
        if (victim < 0) {
            for (int64_t scanned = 0; scanned < 3 * slots; ++scanned) {
                const int64_t s = hand;
                hand = hand + 1 == slots ? 0 : hand + 1;
                if (ST[s * stp] == epoch) continue;                  /* used by this call */
                if (SB[s] >= 0 && ST[s * stp + 1] > 0) {             /* second chance */
                    ST[s * stp + 1] = heat_down(ST[s * stp + 1]);
                    continue;
                }
                victim = s;
                break;
            }
        }
        if (victim < 0) { overflow = true; break; }
        /* ---- 5. re-point both maps ---- */
        if (SB[victim] < 0 && free_left > 0) --free_left;            /* a free slot just filled */
        if (SB[victim] >= 0) R[SB[victim]] = -1;
        SB[victim] = b;
        ST[victim * stp] = epoch;
        ST[victim * stp + 1] = heat_up(ST[victim * stp + 1]);
        R[b] = (int32_t)victim;
        ++placed;
    }
    /* A miss the sweep could not place is not resident: undo the claim, so the next call may claim
     * it again, and leave its selection entries naming the block (see "Overflow"). */
    for (size_t k = (size_t)placed; k < miss.size(); ++k) R[miss[k]] = -1;

    /* ---- 6. the copies, one block each, in miss order ---- */
    const uint8_t* const hb = (const uint8_t*)host->data;
    uint8_t* const pb = (uint8_t*)pool->data;
    for (int64_t k = 0; k < placed; ++k) {
        const int32_t b = miss[(size_t)k];
        std::memcpy(pb + (int64_t)R[b] * bbytes, hb + (int64_t)b * bbytes, (size_t)bbytes);
    }

    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t s = R[ent[e]];
        /* An entry of a miss the sweep could not place names slot 0, not its block id: the id is
         * an address outside the ring (see "Overflow"). ctl[3] is what says the table is unusable. */
        st_int(sel->dtype, sel->data, off[e], s >= 0 ? s : 0);
    }

    /* ---- the bookkeeping ---- */
    /* The union is over DISTINCT blocks, so it is counted from the slots this call stamped (every
     * block it hit or placed leaves the call's epoch there) plus the misses it could not place:
     * entries repeat, a selection names one block per row and many rows share swaths of history. */
    int64_t used = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (ST[s * stp] == epoch) ++used;
    const int64_t uni = used + (int64_t)miss.size() - placed;
    C[0] = epoch;
    C[1] = (int32_t)hand;
    C[2] = (int32_t)placed;
    if (overflow) C[3] = 1;
    /* The sizes the counters report: the union this call selected and the part of it that had to
     * be copied in.
     * A ring sized for the largest union a launch selects is the whole answer to "how large does the
     * ring have to be" (docs/KV-STREAMING.md §3.4). */
    if (uni > C[10]) C[10] = (int32_t)uni;
    if (placed > C[11]) C[11] = (int32_t)placed;
    if ((int64_t)miss.size() > C[12]) C[12] = (int32_t)miss.size();
    uint64_t* const ctr = (uint64_t*)(C + kKvCtlMisses);
    ctr[0] += (uint64_t)placed;
    ctr[1] += (uint64_t)lookups;
    ctr[2] += 1u;
    return RAD_OK;
}
