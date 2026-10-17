/* gbnf_oracle.h -- upstream's candidate walk, the oracle the GBNF mask engine is tested against.
 *
 * fill_mask serves a mask from a plan precomputed at compile time (core/sample/gbnf_mask.cpp).
 * This computes the same mask as llama.cpp's llama_grammar_apply_impl: every vocabulary piece is
 * decoded against the carried partial character and rescanned per stack, per code point. Both
 * compute one pure function of the grammar state, so tests require bit-for-bit and status-for-
 * status equality.
 *
 * Linked into the tests and the acceptance tool only. It rescans the whole vocabulary per stack
 * per mask and must never be reachable from a serving path.
 */
#pragma once
#include "sample/gbnf.h"

#include <cstdint>

namespace rad {

/* The mask for `g`'s current state, with fill_mask's status: RAD_OK, RAD_E_STATE when the grammar
 * has no live parse or admits no token, RAD_E_SHAPE when `n_words` is short, or RAD_E_FULL when the
 * rescan exceeds `work_limit` (0: the engine's per-operation bound). RAD_E_FULL means no comparison
 * was possible, not a disagreement. A lazy grammar awaiting its trigger admits everything. */
int gbnf_oracle_mask(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words,
                     uint64_t work_limit = 0);

/* The state accepting `token` leaves, built as upstream's llama_grammar_accept_impl does: every
 * stack advanced one code point at a time through accept_chr and the epsilon closure. The engine
 * steps code-point stacks through the mask plan's automaton instead; both must leave the same stack
 * set and carried partial character. `*out` is sorted. Returns RAD_OK, RAD_E_STATE when no stack
 * survives, or RAD_E_FULL / RAD_E_NOMEM past the engine's bounds. Only for a grammar not awaiting
 * its trigger and a non-end-of-generation token; accept_token answers those without the stacks. */
int gbnf_oracle_accept(const GbnfGrammar& g, int32_t token, GbnfStacks* out,
                       PartialUtf8* partial);

}  /* namespace rad */
