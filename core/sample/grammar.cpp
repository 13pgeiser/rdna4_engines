/* grammar.cpp -- the GBNF grammar backend and the JSON Schema entry point onto it.
 */
#include "grammar.h"
#include "gbnf.h"

#include <json-schema-to-grammar.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <exception>
#include <new>

namespace rad {

int64_t grammar_mask_words(int64_t n_vocab) { return bitmask_words(n_vocab); }

/* Maximum schema nesting, in objects and arrays. The converter recurses once per level and names
 * each rule after its full path, so output grows with the square of the depth (a 72 KB schema
 * 3,000 levels deep gives 67 MB of grammar) and deeper ones exhaust the admitting thread's memory
 * or stack. At this depth a schema converts in tens of milliseconds to a couple of megabytes. */
static constexpr size_t kMaxSchemaDepth = 512;

/* Iterative, because the values it must catch are too deep to recurse over. */
static size_t json_depth(const nlohmann::ordered_json& root) {
    std::vector<std::pair<const nlohmann::ordered_json*, size_t>> todo{ { &root, 1 } };
    size_t deepest = 0;
    while (!todo.empty()) {
        const auto [j, d] = todo.back();
        todo.pop_back();
        if (!j->is_structured()) continue;
        deepest = std::max(deepest, d);
        for (const auto& c : *j) todo.push_back({ &c, d + 1 });
    }
    return deepest;
}

/* Uses the vendored converter, the only JSON Schema compiler in the tree; its hardening for
 * client data is in vendor/patches 0003 and 0004. Client schemas reach it from /v1/completions
 * (`json_schema`, `response_format`) through the server's grammar bridge; on
 * /v1/chat/completions the chat template compiles them instead. The converter throws; errors
 * are returned as statuses. */
int grammar_from_json_schema(const std::string& schema_json, const std::string& root,
                             std::string* out_gbnf, std::string* err) {
    if (!out_gbnf) return RAD_E_INVAL;
    /* The converter always names its top rule "root"; any other name is refused rather than
     * silently ignored. */
    if (!root.empty() && root != "root") {
        if (err) *err = "this converter always names the top rule 'root'";
        return RAD_E_UNSUPPORTED;
    }
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        if (json_depth(schema) > kMaxSchemaDepth) {
            if (err) *err = "the schema nests more than " + std::to_string(kMaxSchemaDepth) +
                            " levels deep";
            return RAD_E_INVAL;
        }
        *out_gbnf = json_schema_to_grammar(schema);
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return RAD_E_INVAL;
    }
    if (out_gbnf->empty()) {
        if (err) *err = "the schema compiled to an empty grammar";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* ================================================================== the GBNF backend */

namespace {

class GbnfBackend : public GrammarBackend {
public:
    explicit GbnfBackend(std::unique_ptr<GbnfGrammar> g) : g_(std::move(g)) {}

    int  accept_token(int32_t token) override { return g_->accept_token(token); }
    int  fill_mask(uint32_t* m, int64_t n) const override { return g_->fill_mask(m, n); }
    bool complete() const override { return g_->complete(); }
    bool constrains() const override { return !g_->awaiting_trigger(); }
    uint64_t masks_walked() const override { return g_->masks_walked(); }

    /* The state is a stack set and a carried UTF-8 partial, both copyable, so a rewind is a copy
     * whose cost does not depend on how far the window went. */
    bool can_speculate() const override { return true; }
    int  push_state() override { g_->push_state(); return RAD_OK; }
    int  rollback() override   { return g_->rollback(); }

private:
    std::unique_ptr<GbnfGrammar> g_;
};

int gbnf_ctor(const GrammarSpec& spec, const VocabView& vocab,
              std::unique_ptr<GrammarBackend>* out, std::string* err) {
    std::string text = spec.text;

    /* A JSON Schema is compiled to GBNF and run on the same machine, not a second backend. */
    if (spec.kind == GrammarKind::JsonSchema) {
        std::string gbnf;
        const int st = grammar_from_json_schema(spec.text, spec.root, &gbnf, err);
        if (st < 0) return st;
        text = std::move(gbnf);
    }

    std::unique_ptr<GbnfGrammar> g;
    const int st = GbnfGrammar::create(text, spec.root.empty() ? "root" : spec.root, &vocab,
                                       spec.lazy, spec.trigger_words, spec.trigger_tokens,
                                       &g, err);
    if (st < 0) return st;

    if (!spec.prefix.empty()) {
        const int ps = g->accept_str(spec.prefix);
        if (ps < 0) {
            /* The grammar rejects the generation prompt; masking from here would use a state
             * that does not describe this request, so refuse. */
            if (err) *err = "the grammar does not admit this template's generation prompt, so it "
                            "cannot be positioned at the start of the reply";
            return ps;
        }
    }

    *out = std::make_unique<GbnfBackend>(std::move(g));
    return RAD_OK;
}

}  /* namespace */

int grammar_create(const GrammarSpec& spec, const VocabView& vocab,
                   std::unique_ptr<GrammarBackend>* out, std::string* err) {
    if (!out) return RAD_E_INVAL;
    if (spec.text.empty()) {
        if (err) *err = "empty grammar";
        return RAD_E_INVAL;
    }

    /* The compiler returns its own failures as statuses; only bad_alloc can escape. This runs on
     * the admission and scheduler threads, where an escaping exception would end every request. */
    try {
        return gbnf_ctor(spec, vocab, out, err);
    } catch (const std::bad_alloc&) {
        if (err) *err = "out of memory building the grammar";
        return RAD_E_NOMEM;
    }
}

}  /* namespace rad */
