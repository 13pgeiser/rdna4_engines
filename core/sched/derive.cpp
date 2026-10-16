/* derive.cpp -- the derivation table, one row per name.
 */
#include "derive.h"
#include "rad_core.h"

#include <cstring>

namespace rad {

/* ------------------------------------------------------------------ seq_start_index */
/* [n_seq+1] -- where each sequence's tokens begin in this step's flat token array, and the total.
 * A copy of cu_seqlens, so a plugin can name it as an operand instead of reaching into RadBatch. */
static int d_seq_start_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    int64_t n = in.n_seq + 1;
    if (n > cap) return RAD_E_FULL;
    std::memcpy(out, in.cu_seqlens, (size_t)n * sizeof(int32_t));
    *n_out = n;
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv_state_index */
/* [n_seq] -- the conv state cache entry this sequence's update reads from.
 *
 * The conv state is a rolling window of `width-1 + num_spec` entries read at the slot the last
 * accepted token left, so a speculative rejection moves the read offset instead of recomputing
 * (spec §10). Base is the sequence's state slot; offset is the cursor the scheduler advances by
 * 1 + num_accepted per verify. Both live in request bookkeeping, hence a scheduler derivation. */
static int d_conv_state_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    if (in.n_seq > cap) return RAD_E_FULL;
    if (!in.conv_slot) return RAD_E_STATE;
    const int32_t W = in.conv_window > 0 ? in.conv_window : 1;
    for (int64_t i = 0; i < in.n_seq; ++i) {
        const StepEntry& e = in.plan->e[(size_t)i];
        int32_t slot = in.conv_slot[i];
        out[i] = slot < 0 ? -1 : (int32_t)(slot * W + (e.conv_cursor % W));
    }
    *n_out = in.n_seq;
    return RAD_OK;
}

/* ------------------------------------------------------- the linear-state chunk walk */
/* Chunked linear-attention kernels tile each sequence in fixed chunks from the sequence's own
 * origin, not the step's token array. A chunk never crosses a sequence boundary, and interior
 * boundaries land on absolute multiples of the chunk length, or the carried recurrent state would
 * be that of a partial tile. Prefill chunks already end on the quantum (geometry.h); the walk is
 * still needed because a mixed step's decode rows start at arbitrary positions. */
template <bool WantSeq>
static int chunk_walk(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    const int64_t C = in.state_chunk > 0 ? in.state_chunk : 1;
    int64_t k = 0;

    if (!WantSeq) {
        if (cap < 1) return RAD_E_FULL;
        out[k++] = 0;
    }
    for (int64_t i = 0; i < in.n_seq; ++i) {
        int64_t b = in.cu_seqlens[i], e = in.cu_seqlens[i + 1];
        if (e <= b) continue;
        int64_t p0 = in.positions[b];
        /* First interior boundary: the next absolute multiple of C strictly after p0. */
        int64_t next = ((p0 / C) + 1) * C;
        for (int64_t abs = next; abs < p0 + (e - b); abs += C) {
            if (k >= cap) return RAD_E_FULL;
            out[k++] = WantSeq ? (int32_t)i : (int32_t)(b + (abs - p0));
        }
        if (k >= cap) return RAD_E_FULL;
        out[k++] = WantSeq ? (int32_t)i : (int32_t)e;
    }
    *n_out = k;
    return RAD_OK;
}

/* [n_chunks+1] token indices: chunk c covers [out[c], out[c+1]). out[0] is 0 and the last entry is
 * n_tok, like cu_seqlens. */
static int d_state_chunk_bounds(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<false>(in, out, cap, n_out);
}

/* [n_chunks] -- which sequence each chunk belongs to. Derivable from the bounds, but one shared
 * pass here saves a per-chunk search in every layer. */
static int d_state_chunk_seq(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<true>(in, out, cap, n_out);
}

/* ------------------------------------------------------------------ lora_slot */
/* [n_tok] -- the runtime LoRA adapter slot each token's row is multiplied with, -1 for the base
 * model (core/lora/). A sequence's slot is its request's, fixed when the request was admitted, so
 * every row of one sequence carries the same value and a step mixing adapters needs nothing but
 * this array: the LoRA kernels read the slot of each row they touch and leave a -1 row alone. */
static int d_lora_slot(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    if (in.n_tok > cap || !in.plan || (int64_t)in.plan->e.size() < in.n_seq) return RAD_E_FULL;
    for (int64_t i = 0; i < in.n_seq; ++i) {
        const Request* r = in.plan->e[(size_t)i].req;
        const int32_t slot = r ? r->lora_slot : -1;
        const int64_t b = in.cu_seqlens[i], e = in.cu_seqlens[i + 1];
        if (b < 0 || e < b || e > in.n_tok) return RAD_E_STATE;
        for (int64_t k = b; k < e; ++k) out[k] = slot;
    }
    *n_out = in.n_tok;
    return RAD_OK;
}

/* ------------------------------------------------------------------ the table */
static const DeriveRow g_rows[] = {
    { "seq_start_index",  d_seq_start_index,
      "[n_seq+1] first token index of each sequence in this step, and the total" },
    { "conv_state_index", d_conv_state_index,
      "[n_seq] conv rolling-window entry to read, base slot * window + accepted cursor" },
    { "state_chunk_bounds", d_state_chunk_bounds,
      "[n_chunks+1] token indices bounding each linear-state chunk; never crosses a sequence" },
    { "state_chunk_seq",    d_state_chunk_seq,
      "[n_chunks] the sequence each chunk belongs to" },
    { "lora_slot",          d_lora_slot,
      "[n_tok] the runtime LoRA adapter slot of each token's sequence, -1 for the base model" },
};

int derive_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < (int)(sizeof g_rows / sizeof g_rows[0]); ++i)
        if (std::strcmp(g_rows[i].name, name) == 0) return i;
    return -1;
}

const DeriveRow* derive_row(int i) {
    return (i >= 0 && i < (int)(sizeof g_rows / sizeof g_rows[0])) ? &g_rows[i] : nullptr;
}

int derive_count() { return (int)(sizeof g_rows / sizeof g_rows[0]); }

}  /* namespace rad */
