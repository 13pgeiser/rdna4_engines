/* advance_impl.h -- one row of an advance, shared by the kernel (advance.hip) and the host backend
 * (advance.cpp). See advance.h for the modes.
 *
 * Rows are independent: each reads its own slot's state and writes only its own positions, tokens
 * and slots, so the kernel runs one thread per row and the host a loop.
 */
#pragma once
#include "advance.h"

#if defined(__HIPCC__)
#define RAD_ADV_FN __host__ __device__ inline
#else
#define RAD_ADV_FN inline
#endif

namespace rad {

RAD_ADV_FN int32_t* adv_i32(const AdvanceArgs& a, int64_t off) {
    return reinterpret_cast<int32_t*>(a.slab + off);
}

RAD_ADV_FN bool adv_is_eog(const AdvanceArgs& a, int32_t t) {
    int32_t lo = 0, hi = a.n_eog;
    while (lo < hi) {
        const int32_t mid = (lo + hi) / 2;
        if (a.eog[mid] < t) lo = mid + 1; else hi = mid;
    }
    return lo < a.n_eog && a.eog[lo] == t;
}

/* Every group's slots for the row's `n` positions (from the positions just written) and its
 * length from `base`. A position past the block table gets slot -1 ("no slot"); the builder sizes
 * tables for the furthest reachable position, so only a row whose output is unused hits this.
 *
 * A runtime window (e.ring, KVManager::set_window) must be computed exactly as the host builder
 * staged it (BatchBuilder::build_groups), because this runs after the build and overwrites what it
 * wrote:
 *   - `used` stops at the table: the ring reuses its blocks, so the sequence's held span is the
 *     table's, not the context's. Without this the decode walks bt[(used-1)/block] past the end,
 *     reads the -1 padding the builder fills and faults the card on a negative block;
 *   - a position at or past ring_pos lives in the ring, at block
 *     ring_pos/block + ((p - ring_pos)/block) % ring_blocks, not at (p - first)/block, which runs
 *     off the table once the ring wraps.
 *
 * `len`, when given, is the length's count from `base` in place of `n` (ADV_HIST: rows that
 * repeat a position must not lengthen the context). */
RAD_ADV_FN void adv_groups(const AdvanceArgs& a, int i, int32_t k, int32_t base, int32_t n,
                           int32_t len = -1) {
    const AdvGroup* G = reinterpret_cast<const AdvGroup*>(a.slab + a.grp_off);
    const int32_t* pos = adv_i32(a, a.pos_off);
    for (int g = 0; g < a.n_groups; ++g) {
        const AdvGroup& e = G[g];
        const int32_t first = adv_i32(a, e.first_off)[i];
        /* Per sequence: each prompt has its own floor and its own table (advance.h). */
        const int32_t rpos = e.ring ? adv_i32(a, e.rpos_off)[i] : 0;
        const int32_t tlen = e.ring ? adv_i32(a, e.tlen_off)[i] : 0;
        const bool ring = e.ring && tlen > 0;
        int64_t used = (int64_t)base + (len >= 0 ? len : n) - first;
        if (ring) {
            const int64_t span = (int64_t)tlen * e.block_size;
            if (used > span) used = span;
        }
        adv_i32(a, e.used_off)[i] = (int32_t)used;
        if (!e.paged) continue;
        const int32_t* bt = adv_i32(a, e.bt_off) + (int64_t)i * e.pitch;
        int32_t* slot = adv_i32(a, e.slot_off);
        for (int32_t j = 0; j < n; ++j) {
            const int32_t p = pos[k + j];
            int32_t bi = (p - first) / e.block_size;
            if (ring && p >= rpos && e.ring_blocks > 0)
                bi = rpos / e.block_size +
                     ((p - rpos) / e.block_size) % e.ring_blocks;
            slot[k + j] = (p >= first && bi >= 0 && bi < e.pitch && bt[bi] >= 0)
                              ? bt[bi] * e.block_size + p % e.block_size : -1;
        }
    }
}

/* Writes a row's position and moves its rotary components by the same amount. A row whose
 * position does not change keeps the host's components. */
RAD_ADV_FN void adv_pos(const AdvanceArgs& a, int32_t* pos, int32_t row, int32_t p) {
    const int32_t was = pos[row];
    pos[row] = p;
    if (a.rope_off < 0 || was == p) return;
    int32_t* r = adv_i32(a, a.rope_off);
    const int32_t shift = r[row] - was;
    r[row] = p + shift;
    r[a.rope_pitch + row] = p + shift;
    r[2 * a.rope_pitch + row] = p + shift;
}

RAD_ADV_FN void adv_row(const AdvanceArgs& a, int i) {
    const int32_t* rows = adv_i32(a, a.row_off);
    const int32_t rtok = rows[a.row_pitch + i];
    const int32_t rout = rows[2 * a.row_pitch + i];
    const int32_t aux  = rows[3 * a.row_pitch + i];
    /* Acceptance runs over every row that sampled, whoever staged it; every other mode over the
     * rows the card owns. */
    if (a.mode == ADV_ACCEPT ? rout < 0 : !rows[i]) return;

    int32_t* tok = adv_i32(a, a.tok_off);
    int32_t* pos = adv_i32(a, a.pos_off);
    int32_t* ctx = adv_i32(a, a.ctx_off);
    const int32_t slot = adv_i32(a, a.sid_off)[i];
    const int32_t k = adv_i32(a, a.cu_off)[i];
    const int32_t n = adv_i32(a, a.q_off)[i];
    StepSlot& S = a.st[slot];

    switch (a.mode) {
    case ADV_ACCEPT: {
        /* accept_greedy's rule, then collect_step's end-of-generation cut: a draft is accepted
         * while it equals the target's pick at its position, and the run stops at an
         * end-of-generation token, not after it. */
        const int32_t* y = a.src + rout;
        const int32_t* d = tok + rtok + 1;
        int32_t acc = 0;
        while (acc < aux && d[acc] == y[acc]) ++acc;
        for (int32_t j = 0; j <= acc; ++j)
            if (adv_is_eog(a, y[j])) { acc = j; break; }
        const int32_t head = ctx[i] + (rtok - k) + acc;
        S.acc = acc;
        S.head = head;
        S.ctx = head + 1;
        S.n_acc = acc + 1;
        S.tok[0] = y[acc];
        return;
    }
    case ADV_TRUNK: {
        const int32_t base = S.ctx;
        for (int32_t j = 0; j < n; ++j) { tok[k + j] = S.tok[j]; adv_pos(a, pos, k + j, base + j); }
        ctx[i] = base;
        adv_i32(a, a.acc_off)[i] = S.n_acc;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_HIST: {
        if (a.shared_kv) {
            /* A head that reads the trunk's cache and keeps none (rad_drafter_shared_kv). With no
             * history to fill, the pass is round 1 alone: one row at the committed position, with
             * the trunk's row there and the token chosen after it. Its context ends at the
             * committed position, not the verify step's whole run, because the cache past it
             * holds K/V of rejected drafts. The host staged the run's first trunk row; the card
             * adds the accepted drafts. */
            int32_t* out = adv_i32(a, a.out_off);
            tok[k] = a.src[rout + S.acc];
            adv_pos(a, pos, k, S.head + a.rot_shift);
            out[k] += S.acc;
            ctx[i] = S.head;
            adv_groups(a, i, k, S.head, 1);
            return;
        }
        /* `aux` host-known positions, then the verify step's positions up to the last committed
         * one, each with the token committed after it (the sampler's pick there). The pass has a
         * row for every position the step could have committed; rows past the last committed
         * repeat it, so no rejected draft's hidden state enters the head's cache. The repeat also
         * makes the run's last row draft round 1's input whatever the acceptance was
         * (RadBatch::draft_out_ids). */
        const int32_t base = S.head - S.acc - aux;
        const int32_t last = aux + S.acc;
        int32_t* out = adv_i32(a, a.out_off);
        const int32_t row0 = out[k];
        for (int32_t j = 0; j < n; ++j) {
            const int32_t jj = j < last ? j : last;
            if (jj >= aux) tok[k + j] = a.src[rout + jj - aux];
            adv_pos(a, pos, k + j, base + jj);
            out[k + j] = row0 + jj;
        }
        ctx[i] = base;
        /* The context ends at the last committed position, not at the run's end. Attention places
         * row j of a q_len-row run at seqused - q_len + j and reads keys up to there (attn_paged's
         * contract). With the run's whole length the last row, the only one whose output is used,
         * would also read the n_spec - acc positions after it, which the head has not written for
         * this sequence (stale rounds, or another request's K and V). Earlier rows then count
         * lower than their positions, but nothing reads their output, and every row's K and V
         * still store at its own position's slot. */
        adv_groups(a, i, k, base, n, last + 1);
        return;
    }
    case ADV_ROUND: {
        /* Round 2 onward (round 1 is the history pass's last row); each round embeds the previous
         * round's proposal. A head with no cache of its own stays at the committed position over
         * the committed context; its proposal reaches the next round through its hidden state. */
        const int32_t base = a.shared_kv ? S.head : S.head + a.round - 1;
        const int32_t t = a.src[(int64_t)i * a.src_pitch];
        S.tok[a.round - 1] = t;
        tok[k] = t;
        adv_pos(a, pos, k, a.shared_kv ? base + a.rot_shift : base);
        ctx[i] = base;
        adv_groups(a, i, k, base, 1);
        return;
    }
    case ADV_TAKE:
        S.tok[a.round] = a.src[(int64_t)i * a.src_pitch];
        return;
    case ADV_BLOCK: {
        /* A block drafter's query row: the anchor (the token the verify just chose) at the
         * committed end, where the next verify starts, then `aux`, the mask id, at every position
         * after it. The host staged the furthest the anchor can be. */
        const int32_t base = S.ctx;
        tok[k] = S.tok[0];
        for (int32_t j = 1; j < n; ++j) tok[k + j] = aux;
        for (int32_t j = 0; j < n; ++j) adv_pos(a, pos, k + j, base + j);
        ctx[i] = base;
        adv_groups(a, i, k, base, n);
        return;
    }
    case ADV_BTAKE:
        /* Row i is the query pass's i-th block, which is the i-th drafting sequence. */
        for (int32_t j = 0; j < a.round; ++j)
            S.tok[1 + j] = a.src[(int64_t)i * a.src_pitch + j];
        return;
    default:
        return;
    }
}

}  /* namespace rad */
