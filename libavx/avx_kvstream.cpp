/* avx_kvstream.cpp -- kv_resolve: the residency map between a streamed KV group's block ids and
 * the ring slots its pool holds (docs/OPS.md "kv_resolve", docs/KV-STREAMING.md §3.4).
 *
 * libref/ref_kvstream.cpp is the contract, and this file follows it line by line: the same
 * refusals in the same order, the same epoch, the same ascending miss order, the same victim from
 * the same CLOCK sweep over the same two-bit heat counter, so the same blocks land in the
 * same slots. That is the part a residency
 * map cannot get "close" on: `sel` becomes the attention's addressing, and two implementations
 * that agreed on residency but disagreed on a victim would read different KV for the same query.
 *
 * Why its own file rather than a section of avx_qsa.cpp: the op sits beside `qsa_select` in
 * docs/OPS.md but is not part of the indexer -- it walks no scores and rotates nothing, it moves
 * bytes and owns state across calls (epoch, hand, heat counters). libref keeps the same split
 * (ref_qsa.cpp / ref_kvstream.cpp), so the two libraries' files diff against each other.
 *
 * ============================== what is and is not vector work ==============================
 *
 * The phases that can run wide do: the rewrite of a `sel` row (a gather of `res[b]` with the
 * negative results left in place) and the block copy (one block is a whole number of 16-byte lanes
 * at every cache layout, docs/OPS.md). The passes that cannot are the lookup/claim pass, the claim
 * gather and the CLOCK sweep, and none of them is a missed opportunity:
 *
 *   the map is its own set (phase 2). Two lanes of one vector naming the same absent block would
 *   both read -1 and both claim it, which is exactly the double claim -2 exists to prevent, and
 *   two lanes naming the same resident slot would race on its epoch. The pass is one load and one
 *   store per entry and is bounded by the selection, not by the cache. The claim gather and the
 *   revert are the same pass twice more (`lookups` entries each) for the same reason: the list is a
 *   fact about the selection, and the whole point of this form is that `blocks` -- the id space --
 *   never enters a launch's cost.
 *
 *   the sweep is a serial second-chance walk with a moving hand whose result is the placement
 *   itself. Its cost is one pass over the slots per miss, and a miss is a cold block: the copies
 *   dominate.
 *
 * No OpenMP anywhere: the op carries state from call to call (ctl's epoch, hand and counters), so
 * two of its launches must never overlap and one launch is a single serial unit of work. This is
 * why its row is absent from libr4d's concurrent prefix table; libavx's rows make no concurrency
 * promise at all.
 *
 * Scratch is the arena the other kernels use (avx_common.h), not RadArgs::scratch: the miss list
 * and the claim list are per-call temporaries, and the arena is already the plugin's answer for
 * one. The layout is `int32 miss_block[slots]`, `int32 miss_slot[slots]` -- what a device row
 * declares through RadKernelInfo::scratch -- followed by `int32 claim[lookups]`, the gathered claim
 * set this file needs only because it has no workgroup to sort in.
 */
#include "avx_common.h"
#include "avx_vec.h"

#include <algorithm>

using namespace avx;

namespace {

/* The heat counter's two writes, the same function in all three implementations (libref's
 * heat_up/heat_down): a use bumps by one and saturates at 3, a sweep pass halves, and a negative
 * value -- never written by a legal call -- reads as cold. */
int32_t kv_heat_up(int32_t h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
int32_t kv_heat_down(int32_t h) { return h > 0 ? (h >> 1) : 0; }

/* `ctl`'s size and the index of its first u64 counter (three consecutive u64s from [4]). */
enum { kKvCtlInts = 16, kKvCtlMisses = 4, kKvCtlLookups = 6, kKvCtlCalls = 8 };
static_assert(kKvCtlLookups == kKvCtlMisses + 2 && kKvCtlCalls == kKvCtlMisses + 4,
              "the three counters are consecutive u64s from ctl[4]");

/* The byte pitch of a rank-2-or-deeper operand's rows, and whether its trailing extents fill that
 * pitch densely (libref's row_bytes). Both planes are one block a row: a padded row would copy a
 * block's bytes over its neighbour's. */
int64_t kv_row_bytes(const RadTensor* t, int64_t* dense) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) acc *= t->shape[i];
    *dense = acc;
    return t->stride[0] * rad_dtype_bytes(t->dtype, 1);
}

/* `block_bytes` of the mirror into a slot of the pool. A copy is a byte copy at every cache
 * layout (the op never reads an element), so the width is the ISA's, not the block's dtype. */
void kv_copy_block(uint8_t* dst, const uint8_t* src, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL >= 3
    for (; i + 64 <= n; i += 64)
        _mm512_storeu_si512((void*)(dst + i), _mm512_loadu_si512((const void*)(src + i)));
#elif AVX_LEVEL >= 2
    for (; i + 32 <= n; i += 32)
        _mm256_storeu_si256((__m256i*)(dst + i),
                            _mm256_loadu_si256((const __m256i*)(src + i)));
#elif AVX_LEVEL >= 1
    for (; i + 16 <= n; i += 16)
        _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((const __m128i*)(src + i)));
#endif
    if (i < n) std::memcpy(dst + i, src + i, (size_t)(n - i));
}

/* The claim list: every distinct block the selection claims, in block-id order. A claim exists only
 * where an entry names a block -- the lookup pass left it `-2` -- so the list is gathered from the
 * SELECTION, whose length is `lookups = sum_q nsel[q]`: a property of the call. It is deliberately
 * not a walk of `res` over `blocks`, which is the group's whole id space and says nothing about how
 * big one launch is. The map stays the dedup set: the first entry naming a claim marks it `-3`, so a
 * block several rows name enters the list once, and the marks are restored before the sweep sees the
 * map.
 *
 * The caller sorts the list; `cap` is the buffer's length and is `lookups` at least, because a
 * distinct claim costs at least one entry, so every claim is written. The contract needs the first
 * `slots` of that ascending order and nothing past them. */
int64_t kv_claims(int32_t* res, int32_t* out, int64_t cap,
                  const int32_t* sel, const int32_t* nsel, int64_t M, int64_t pitch) {
    int64_t n = 0;
    for (int64_t q = 0; q < M; ++q) {
        const int32_t* const row = sel + q * pitch;
        for (int64_t i = 0; i < nsel[q]; ++i) {
            const int32_t b = row[i];
            if (res[b] != -2) continue;              /* a hit, or taken by an earlier entry */
            res[b] = -3;
            if (n < cap) out[n++] = b;
        }
    }
    for (int64_t k = 0; k < n; ++k) res[out[k]] = -2;
    return n;
}

/* Every claim the sweep could not place goes back to -1: the block is not resident, and the next
 * call may claim it again. A `res` entry still -2 here is by definition such a claim (a placed one
 * holds its slot), and it is named by at least one selection entry, so one pass over the selection
 * reaches all of them -- `lookups` entries instead of `blocks` map entries. The write is idempotent,
 * so entries that repeat a block are harmless. */
void kv_relax_claims(int32_t* res, const int32_t* sel, const int32_t* nsel, int64_t M,
                     int64_t pitch) {
    for (int64_t q = 0; q < M; ++q) {
        const int32_t* const row = sel + q * pitch;
        for (int64_t i = 0; i < nsel[q]; ++i)
            if (res[row[i]] == -2) res[row[i]] = -1;
    }
}

/* The rewrite: entry i becomes the slot holding its block, and an entry whose block could not be
 * placed becomes slot 0 -- a defined address inside the ring, because leaving the block id would
 * make the attention dereference past it. ctl[3] is what marks the table unusable ("Overflow" in
 * docs/OPS.md).
 *
 * The gather is safe by construction: the validation pass refused any entry outside [0, blocks)
 * before a byte was written. An entry at or past nsel[q] is never passed in, so `qsa_select`'s -1
 * padding survives. */
void kv_rewrite_row(int32_t* row, int64_t n, const int32_t* res) {
    int64_t i = 0;
#if AVX_LEVEL >= 3
    const __m512i zero = _mm512_setzero_si512();
    for (; i + 16 <= n; i += 16) {
        const __m512i b = _mm512_loadu_si512((const void*)(row + i));
        const __m512i s = _mm512_mask_i32gather_epi32(zero, (__mmask16)0xFFFF, b,
                                                      (const void*)res, 4);
        /* A gather that lands on a negative is a block this call could not place (-1) or one it
         * just claimed (-2): slot 0, not the id, exactly as the scalar tail writes it. Clamping
         * says that for the whole vector; masking the store instead would leave the id in place. */
        _mm512_storeu_si512((void*)(row + i), _mm512_max_epi32(s, zero));
    }
#elif AVX_LEVEL >= 2
    const __m256i all = _mm256_set1_epi32(-1);
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_loadu_si256((const __m256i*)(row + i));
        const __m256i s = _mm256_mask_i32gather_epi32(zero, res, b, all, 4);
        /* s > -1 is s >= 0; a negative lane becomes 0 ("Overflow"), never the block id it holds. */
        const __m256i ok = _mm256_cmpgt_epi32(s, all);
        _mm256_storeu_si256((__m256i*)(row + i), _mm256_blendv_epi8(zero, s, ok));
    }
#endif
    for (; i < n; ++i) {
        const int32_t s = res[row[i]];
        row[i] = s >= 0 ? s : 0;
    }
}

}  /* namespace */

/* ------------------------------------------------------------------ kv_resolve */
/* One call: bump the epoch; one lookup per distinct block (a hit stamps the slot with the epoch and
 * bumps its heat counter, an absent block becomes a claim and joins the miss list once); place one
 * victim per miss -- the block's own slot when its id fits the ring and that slot is free (a
 * context shorter than the ring maps block id to slot), otherwise a free slot first, then CLOCK from
 * the hand over the two-bit heat counter, which never takes a slot this call has stamped; re-point
 * both maps; copy each missed block from the mirror; rewrite every valid entry. `nsel` and the
 * attention's length are untouched:
 * a selection entry is an address, not a count.
 *
 * `kv_cache` is the RAD_KV operand the attention and `kv_store` take, and it is INOUT rather than
 * OUT: only the slots a miss lands in are written and every other slot of the ring must survive
 * for the blocks still resident there.
 *
 * An entry of a miss the sweep could not place is rewritten to slot 0, not left holding its block
 * id: on device the id is an address past the ring that `attn_paged` dereferences unchecked, and a
 * measured overflow must produce wrong numbers the caller discards (ctl[3]), not a fault. */
AVX_KERNEL(kv_resolve) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
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
    const RadTensor* const state[] = { sel, nsel, res, sblk, sst, ctl };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return RAD_E_DTYPE;
    if (sel->rank != 2 || sst->rank != 2) return RAD_E_SHAPE;
    if (sel->shape[0] < M || sel->shape[1] < pitch) return RAD_E_SHAPE;
    if (sel->stride[1] != 1 || sel->stride[0] != pitch) return RAD_E_STRIDE;
    if (numel(nsel) < M || nsel->stride[0] != 1) return RAD_E_SHAPE;
    if (numel(res) < blocks || res->stride[0] != 1) return RAD_E_SHAPE;
    /* The operand states the ring and the parameter is its declared ceiling, as `sel` states `M`
     * (libref/ref_kvstream.cpp, "The parameters"): the pool is planned after declare, so a ring the
     * engine sized smaller (--kv-resident) arrives as a shorter `slot_block`. */
    const int64_t slots = numel(sblk);
    if (slots < 1 || slots > slots_decl || sblk->stride[0] != 1) return RAD_E_SHAPE;
    if (sst->shape[0] < slots || sst->stride[1] != 1 || sst->stride[0] < 2) return RAD_E_STRIDE;
    if (numel(ctl) < kKvCtlInts || ctl->stride[0] != 1) return RAD_E_SHAPE;
    if (host->rank < 2 || pool->rank < 2) return RAD_E_SHAPE;
    if (host->shape[0] < blocks || pool->shape[0] < slots) return RAD_E_SHAPE;

    /* One block of one layer: the mirror's row and the pool's slot are the same bytes, and
     * `block_bytes` is both pitches -- the length of every copy. */
    int64_t hdense = 0, pdense = 0;
    if (kv_row_bytes(host, &hdense) != bbytes ||
        hdense * rad_dtype_bytes(host->dtype, 1) != bbytes)
        return RAD_E_STRIDE;
    if (kv_row_bytes(pool, &pdense) != bbytes ||
        pdense * rad_dtype_bytes(pool->dtype, 1) != bbytes)
        return RAD_E_STRIDE;

    int32_t* const R  = (int32_t*)res->data;
    int32_t* const SB = (int32_t*)sblk->data;
    int32_t* const ST = (int32_t*)sst->data;
    int32_t* const C  = (int32_t*)ctl->data;
    const int32_t* const NS = (const int32_t*)nsel->data;
    int32_t* const SE = (int32_t*)sel->data;
    const int64_t stp = sst->stride[0];
    /* The counters are read and written as u64 inside an i32 block. */
    if (((uintptr_t)C & 7u) != 0) return RAD_E_STRIDE;

    /* ---- everything that can refuse, before anything is written; the entry count is summed here
     * so the counter and the claim list's length need no second look at `nsel` ---- */
    int64_t lookups = 0;
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = NS[q];
        if (n < 0 || n > pitch) return RAD_E_SHAPE;
        const int32_t* const row = SE + q * pitch;
        for (int64_t i = 0; i < n; ++i)
            if (row[i] < 0 || row[i] >= blocks) return RAD_E_INVAL;
        lookups += n;
    }
    /* The map's own domain, over the whole of it. This is a state fault the contract refuses
     * (`-2` is never left by a call; a slot index outside the ring would address past it), not a
     * fallback: an entry the map holds and no selection entry names can only be found by looking
     * at the map, so this pass is `blocks` long in the reference and here alike. Everything that
     * acts on a claim is bounded by the selection, as above. */
    for (int64_t b = 0; b < blocks; ++b)
        if (R[b] < -1 || R[b] >= slots) return RAD_E_INVAL;   /* -2 is never left by a call */
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < -1 || SB[s] >= blocks) return RAD_E_INVAL;
    if (C[1] < 0 || C[1] >= slots) return RAD_E_INVAL;

    /* ---- 1. the epoch ---- */
    const int32_t epoch = (int32_t)((uint32_t)C[0] + 1u);

    /* ---- 2. hits, and one claim per distinct miss (the map is its own set) ---- */
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = NS[q];
        const int32_t* const row = SE + q * pitch;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t b = row[i];
            const int32_t s = R[b];
            if (s >= 0) {
                /* The epoch stamps the slot; the heat counter gains one step, once a slot a call
                 * (libref's step 2), so the counter measures calls that wanted the block. */
                if (ST[s * stp] != epoch) {
                    ST[s * stp] = epoch;
                    ST[s * stp + 1] = kv_heat_up(ST[s * stp + 1]);
                }
            } else if (s == -1) {
                R[b] = -2;
            }
        }
    }

    /* ---- 3. the canonical miss order ---- */
    int32_t* const miss_block = (int32_t*)scratch_raw((size_t)(2 * slots + lookups) * sizeof(int32_t));
    int32_t* const miss_slot = miss_block + slots;
    int32_t* const claim = miss_slot + slots;
    const int64_t nclaims = kv_claims(R, claim, lookups, SE, NS, M, pitch);
    std::sort(claim, claim + nclaims);
    const int64_t nmiss = nclaims < slots ? nclaims : slots;
    for (int64_t k = 0; k < nmiss; ++k) miss_block[k] = claim[k];

    /* ---- 4. one victim per miss: identity, a free slot, then CLOCK, from the hand, never a slot
     * this call uses ---- */
    /* Identity first: a block whose id fits the ring and whose own slot is free goes there, so a
     * context shorter than the ring gets res[b] = b and the attention's run of adjacent ids reads
     * adjacent slots of one layer's stripe (libref's step 4; +2.07% at 16k, §26). It takes only a
     * free slot and does not move the hand, so it never evicts and the fallback below is the old
     * policy exactly, from the hand it would always have seen.
     *
     * Free before live: the ring's empty region fills before any resident block is evicted (see
     * libref's step 4). The free scan has no side effect when it finds nothing, so a full ring
     * takes the same CLOCK victim it always took. The count is what keeps a full ring from paying
     * that scan per miss: a placement makes a free slot live and an eviction re-points a live one,
     * so a zero count holds for the rest of the call. */
    int64_t free_left = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < 0 && ST[s * stp] != epoch) ++free_left;
    int64_t hand = C[1];
    int64_t placed = 0;
    /* More claims than the ring can hold means some miss was left unplaced by construction -- the
     * list here is the first `slots` of the claim set, so the sweep cannot discover it by running
     * out of misses -- and libref latches exactly that. */
    bool overflow = nclaims > slots;
    for (int64_t k = 0; k < nmiss; ++k) {
        const int32_t b = miss_block[k];
        int64_t victim = -1;
        if ((int64_t)b < slots && ST[(int64_t)b * stp] != epoch && (SB[b] < 0 || SB[b] == b))
            victim = b;
        if (victim < 0 && free_left > 0) {
            int64_t h = hand;
            for (int64_t scanned = 0; scanned < slots; ++scanned) {
                const int64_t s = h;
                h = h + 1 == slots ? 0 : h + 1;
                if (ST[s * stp] == epoch) continue;                 /* used by this call */
                if (SB[s] >= 0) continue;                           /* holds a block */
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
                if (ST[s * stp] == epoch) continue;                 /* used by this call */
                if (SB[s] >= 0 && ST[s * stp + 1] > 0) {             /* second chance */
                    ST[s * stp + 1] = kv_heat_down(ST[s * stp + 1]);
                    continue;
                }
                victim = s;
                break;
            }
        }
        if (victim < 0) {
            /* Every slot of the ring is stamped with this call's epoch: nothing left to take, and
             * taking a slot this call filled would break the table it is about to hand out. */
            overflow = true;
            break;
        }
        /* ---- 5. re-point both maps ---- */
        if (SB[victim] < 0 && free_left > 0) --free_left;            /* a free slot just filled */
        if (SB[victim] >= 0) R[SB[victim]] = -1;
        SB[victim] = b;
        ST[victim * stp] = epoch;
        ST[victim * stp + 1] = kv_heat_up(ST[victim * stp + 1]);
        R[b] = (int32_t)victim;
        miss_slot[k] = (int32_t)victim;
        ++placed;
    }
    kv_relax_claims(R, SE, NS, M, pitch);

    /* ---- 6. the copies, one block each, in miss order ---- */
    const uint8_t* const hb = (const uint8_t*)host->data;
    uint8_t* const pb = (uint8_t*)pool->data;
    for (int64_t k = 0; k < placed; ++k)
        kv_copy_block(pb + (int64_t)miss_slot[k] * bbytes,
                      hb + (int64_t)miss_block[k] * bbytes, bbytes);

    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    for (int64_t q = 0; q < M; ++q) kv_rewrite_row(SE + q * pitch, NS[q], R);

    /* ---- the bookkeeping ---- */
    /* Distinct blocks, counted from the slots this call stamped: a hit or a placed miss leaves the
     * call's epoch in its slot, and the misses it could not place are the rest of the union. Entry
     * counts would overcount, because many of a launch's rows share swaths of the history. */
    int64_t used = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (ST[s * stp] == epoch) ++used;
    /* The union is over DISTINCT blocks, so it is the slots this call stamped plus the claims it
     * could not place -- exactly libref's count, and now that the whole claim set is known rather
     * than a list clipped at `slots`, an overflow reports the union it selected too. */
    const int64_t uni = used + nclaims - placed;
    C[0] = epoch;
    C[1] = (int32_t)hand;
    C[2] = (int32_t)placed;
    if (overflow) C[3] = 1;
    /* The measured sizes: this call's selection union, the part of it that had to be copied in, and
     * the depth of its claim list. */
    if (uni > C[10]) C[10] = (int32_t)uni;
    if (placed > C[11]) C[11] = (int32_t)placed;
    if (nclaims > C[12]) C[12] = (int32_t)nclaims;
    uint64_t* const ctr = (uint64_t*)(C + kKvCtlMisses);
    ctr[0] += (uint64_t)placed;
    ctr[1] += (uint64_t)lookups;
    ctr[2] += 1u;
    return RAD_OK;
}
