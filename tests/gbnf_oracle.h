/* gbnf_oracle.h -- upstream's candidate walk, kept as the oracle the GBNF mask engine is tested
 * against.
 *
 * fill_mask serves a mask from a plan precomputed at compile time (core/sample/gbnf_mask.cpp).
 * This computes the same mask the way llama.cpp's llama_grammar_apply_impl does: every piece in
 * the vocabulary decoded against the carried partial character and rescanned per stack, per code
 * point. The two are implementations of one pure function of the grammar's state, so the test is
 * equality, bit for bit and status for status, and any difference is a defect in the engine.
 *
 * It is linked into the tests and the acceptance tool only. It costs a full rescan of the
 * vocabulary per stack per mask, which is what the engine exists to avoid, and it must never be
 * reachable from a serving path.
 */
#pragma once
#include "sample/gbnf.h"

#include <cstdint>

namespace rad {

/* The mask for `g`'s current state. Returns what fill_mask must return for it: RAD_OK, RAD_E_STATE
 * when the grammar has no live parse or admits no token, RAD_E_SHAPE when `n_words` is short --
 * or RAD_E_FULL when the rescan outgrows `work_limit` (the engine's per-operation bound when 0),
 * which is a comparison the caller could not make rather than a disagreement. A lazy grammar
 * still waiting for its trigger admits everything. */
int gbnf_oracle_mask(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words,
                     uint64_t work_limit = 0);

/* THE STATE ACCEPTING `token` LEAVES, the way the stack machinery builds it: every stack advanced
 * one code point at a time through accept_chr and the epsilon closure, as upstream's
 * llama_grammar_accept_impl does. The engine steps the stacks that match code points through the
 * mask plan's automaton instead, and the two must leave the same stack set and the same carried
 * partial character. `*out` comes back sorted. Returns RAD_OK, RAD_E_STATE when no stack
 * survives, or RAD_E_FULL / RAD_E_NOMEM past the engine's bounds. For a grammar not waiting on
 * its trigger, and a token that is not end-of-generation -- accept_token answers those without
 * the stacks. */
int gbnf_oracle_accept(const GbnfGrammar& g, int32_t token, GbnfStacks* out,
                       PartialUtf8* partial);

}  /* namespace rad */
