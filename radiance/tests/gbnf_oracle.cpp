/* gbnf_oracle.cpp -- upstream's candidate walk (llama.cpp `llama_grammar_reject_candidates` and
 * `llama_grammar_apply_impl`, at the commit gbnf.h names), over the engine's own stack machinery.
 * See gbnf_oracle.h.
 *
 * The closure, character tests and UTF-8 decoding are the engine's (gbnf_internal.h). Only mask
 * assembly from a stack set is reimplemented, so a disagreement points at that assembly.
 */
#include "gbnf_oracle.h"

#include "sample/gbnf_internal.h"

#include <algorithm>
#include <new>

namespace rad {

namespace {

using namespace gbnf;

struct Candidate {
    size_t          index;
    const uint32_t* code_points;
    PartialUtf8     partial_utf8;
    int32_t         id;
};
using Candidates = std::vector<Candidate>;

/* Work charged per candidate per character-range element, against the engine's per-operation
 * bound. */
constexpr uint64_t kCandidateCost = 4;

Candidates reject_for_stack(const GbnfRuleSet& rules, const GbnfStack& stack,
                            const Candidates& candidates, OpWork& work);

Candidates reject_candidates(const GbnfRuleSet& rules, const GbnfStacks& stacks,
                             const Candidates& candidates, OpWork& work) {
    if (candidates.empty() || stacks.empty()) return {};
    Candidates rejects = reject_for_stack(rules, stacks.front(), candidates, work);
    for (size_t i = 1; i < stacks.size(); ++i)
        rejects = reject_for_stack(rules, stacks[i], rejects, work);
    return rejects;
}

Candidates reject_for_stack(const GbnfRuleSet& rules, const GbnfStack& stack,
                            const Candidates& candidates, OpWork& work) {
    {
        const GbnfElement* t = stack.empty() ? nullptr : stack.back();
        const bool range = t && !is_token_elem(t);
        work.add(candidates.size() * kCandidateCost * (range ? char_range_len(t) : 1) +
                 kStackCost);
    }
    Candidates rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        /* A completed stack accepts nothing further, so anything with content left is rejected. */
        for (const auto& t : candidates)
            if (*t.code_points != 0 || t.partial_utf8.n_remain != 0) rejects.push_back(t);
        return rejects;
    }

    const GbnfElement* top = stack.back();

    if (is_token_elem(top)) {
        for (const auto& t : candidates) {
            if (*t.code_points == 0) {
                if (t.partial_utf8.n_remain != 0) rejects.push_back(t);
            } else if (!match_token(top, t.id)) {
                rejects.push_back(t);
            }
        }
        return rejects;
    }

    Candidates next;
    next.reserve(candidates.size());
    for (const auto& t : candidates) {
        if (*t.code_points == 0) {
            if (t.partial_utf8.n_remain != 0 && !match_partial_char(top, t.partial_utf8))
                rejects.push_back(t);
        } else if (match_char(top, *t.code_points).first) {
            next.push_back({ t.index, t.code_points + 1, t.partial_utf8, t.id });
        } else {
            rejects.push_back(t);
        }
    }

    const GbnfElement* after = match_char(top, 0).second;
    GbnfStack stack_after(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(after)) stack_after.push_back(after);

    GbnfStacks next_stacks;
    SetBuild   next_set(work);
    advance_stack(rules, stack_after, next_stacks, next_set);

    for (const auto& t : reject_candidates(rules, next_stacks, next, work))
        rejects.push_back({ t.index, t.code_points - 1, t.partial_utf8, t.id });

    return rejects;
}

int oracle_impl(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words, uint64_t work_limit) {
    const VocabView&   vocab = *g.vocab();
    const GbnfRuleSet& rules = program_rules(g.program());
    const int32_t      n_tok = vocab.n_tokens();
    const bool         allow_eog = g.complete();

    /* Decode every piece against the carried partial, admit every candidate, then clear the
     * rejects of one reject walk over the whole vocabulary. */
    std::vector<std::pair<std::vector<uint32_t>, PartialUtf8>> decoded;
    decoded.reserve((size_t)n_tok);
    Candidates cands;
    cands.reserve((size_t)n_tok);
    std::vector<int32_t> index_to_id;
    index_to_id.reserve((size_t)n_tok);

    for (int32_t id = 0; id < n_tok; ++id) {
        if (vocab.is_eog(id)) {
            if (allow_eog) bitmask[id >> 5] |= 1u << ((uint32_t)id & 31u);
            continue;
        }
        const std::string& piece = vocab.token_piece(id);
        if (piece.empty() || piece[0] == 0) continue;
        const auto& d = decoded.emplace_back(decode_utf8(piece, g.partial()));
        cands.push_back({ index_to_id.size(), d.first.data(), d.second, id });
        index_to_id.push_back(id);
    }

    for (int32_t id : index_to_id) bitmask[id >> 5] |= 1u << ((uint32_t)id & 31u);

    OpWork work;
    if (work_limit) work.limit = work_limit;
    for (const auto& r : reject_candidates(rules, g.stacks(), cands, work)) {
        const int32_t id = index_to_id[r.index];
        bitmask[id >> 5] &= ~(1u << ((uint32_t)id & 31u));
    }

    for (int64_t w = 0; w < n_words; ++w)
        if (bitmask[w]) return RAD_OK;
    return RAD_E_STATE;
}

void accept_chr(const GbnfRuleSet& rules, const GbnfStack& stack, uint32_t chr,
                GbnfStacks& out, SetBuild& build) {
    if (stack.empty()) return;
    const GbnfElement* pos = stack.back();
    if (is_token_elem(pos)) return;
    auto m = match_char(pos, chr);
    build.work.add((size_t)(m.second - pos));
    if (!m.first) return;
    GbnfStack next(stack.begin(), stack.end() - 1);
    if (!is_end_of_sequence(m.second)) next.push_back(m.second);
    advance_stack(rules, next, out, build);
}

int accept_impl(const GbnfGrammar& g, int32_t token, GbnfStacks* out, PartialUtf8* partial) {
    const GbnfRuleSet& rules = program_rules(g.program());
    const auto decoded = decode_utf8(g.vocab()->token_piece(token), g.partial());
    const auto& cps = decoded.first;

    OpWork     work;
    GbnfStacks new_stacks;
    SetBuild   new_set(work);
    GbnfStacks current;
    for (const auto& stack : g.stacks()) {
        if (stack.empty()) continue;
        const GbnfElement* pos = stack.back();
        if (is_token_elem(pos)) {
            if (!match_token(pos, token)) continue;
            GbnfStack next(stack.begin(), stack.end() - 1);
            if (!is_end_of_sequence(pos + 1)) next.push_back(pos + 1);
            advance_stack(rules, next, new_stacks, new_set);
        } else {
            current.push_back(stack);
        }
    }
    for (auto it = cps.begin(), end = cps.end() - 1; it != end && !current.empty(); ++it) {
        GbnfStacks next;
        SetBuild   next_set(work);
        for (const auto& cs : current) accept_chr(rules, cs, *it, next, next_set);
        current = std::move(next);
    }
    for (auto& s : current) new_set.insert(new_stacks, std::move(s));
    std::sort(new_stacks.begin(), new_stacks.end());
    *out = std::move(new_stacks);
    *partial = decoded.second;
    return out->empty() ? RAD_E_STATE : RAD_OK;
}

}  /* namespace */

int gbnf_oracle_accept(const GbnfGrammar& g, int32_t token, GbnfStacks* out,
                       PartialUtf8* partial) {
    if (!out || !partial || !g.vocab()) return RAD_E_INVAL;
    try {
        return accept_impl(g, token, out, partial);
    } catch (const gbnf::GbnfLimit&) {
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        return RAD_E_NOMEM;
    }
}

int gbnf_oracle_mask(const GbnfGrammar& g, uint32_t* bitmask, int64_t n_words,
                     uint64_t work_limit) {
    if (!bitmask || !g.vocab()) return RAD_E_INVAL;
    if (n_words < bitmask_words(g.vocab()->n_tokens())) return RAD_E_SHAPE;
    if (g.awaiting_trigger()) {
        std::fill(bitmask, bitmask + n_words, 0xFFFFFFFFu);
        return RAD_OK;
    }
    std::fill(bitmask, bitmask + n_words, 0u);
    if (g.stacks().empty()) return RAD_E_STATE;
    try {
        return oracle_impl(g, bitmask, n_words, work_limit);
    } catch (const gbnf::GbnfLimit&) {
        return RAD_E_FULL;
    } catch (const std::bad_alloc&) {
        return RAD_E_NOMEM;
    }
}

}  /* namespace rad */
