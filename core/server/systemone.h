/* systemone.h -- POST /v1/systemone: LiquidAI d1's decisions, a distribution read off one position.
 *
 * d1 is a decision model. It is given a state (text, any JSON value, pictures, or a mix) and named
 * questions, and each answer is a softmax over a handful of tokens at the position right after the
 * assistant header: `yes` / `no` for a noul, one code a option for a choice, a digit a level for a
 * score. Nothing is generated. The API and the prompt are the model's own (its prompt.py and api.py
 * on the Hub); this file is that adapter, and the engine knows nothing of it beyond
 * Request::token_scores -- "read these ids' raw logits at the prompt's last position".
 *
 * A container serves it only when it says it is d1 (`decision.type` = `d1`, set at convert with
 * `--set decision.type=d1`); the HF config does not declare it.
 */
#pragma once

#include "oai.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct DecisionQuestion {
    std::string name, type;                     /* type: noul | choice | score */
    /* One entry an answer, in the response's order: "true"/"false", the option names, the
     * levels "0".."n-1". */
    std::vector<std::string> labels;
    std::vector<std::string> legend;            /* score: each level's description */
    /* The ids read at the answer slot, and for each answer the indices into them of the
     * spellings it scores as: an answer's score is its best spelling's. */
    std::vector<int32_t>             candidates;
    std::vector<std::vector<size_t>> groups;
    std::vector<int32_t>             tokens;    /* the whole prompt */
};

struct DecisionRequest {
    std::vector<DecisionQuestion> questions;
    /* The pictures, placed in the shared head of every question's prompt. */
    std::shared_ptr<mm::PromptMedia> media;
    /* How many leading tokens every question's prompt shares: the state, tokenised once. 0 for
     * a single question, which d1 tokenises as one text. */
    int32_t trunk = 0;
    /* What the model reads, as d1 counts it: the shared head once and each question's own tail. */
    int64_t input_tokens = 0;
};

/* Parse and render. `max_ctx` bounds every question's prompt (one position stays free for the
 * token the request samples and discards). */
int parse_decision_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                           const std::string& model_id, DecisionRequest* out, ApiError* err);

/* One answer from its candidates' raw logits: each answer scores its best spelling, a softmax over
 * the answers is the distribution. Non-finite logits are an error, not a distribution. */
int decision_answer(const DecisionQuestion& q, const std::vector<float>& logits, json* out,
                    ApiError* err);

}  /* namespace server */
}  /* namespace rad */
