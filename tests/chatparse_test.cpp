/* chatparse_test.cpp -- the streaming reply parser, against the legacy parser in tests/legacy/.
 *
 * core/text/chatparse.h reads a reply once, as it streams, from a format description. The legacy
 * path re-parses the whole reply on every chunk with the PEG parser llama.cpp's auto-parser builds
 * from the chat template, then trims and lifts calls in a second pass. It is the oracle: every
 * case runs both on the same request rendered through each served model's real template
 * (tests/data/chat).
 *
 * Comparison rule (chatdiff::judge): where both read a reply within the format's grammar, the
 * answers are identical byte for byte (content, reasoning, every call's name and arguments).
 * Where the legacy grammar rejects the reply, it returns the whole reply as content with calls
 * lifted by hand once complete; a streaming parser cannot retract what it sent, so it recovers
 * locally instead (strict() reports this). The legacy path also lifts a call written without the
 * format's opener out of content (`bare_lift`); the streaming parser leaves it as content.
 *
 * Streaming contract, checked at every cut (whole, every two-way split, byte by byte, token-sized):
 * each delta is a prefix of the final value, the deltas sum to it, it equals the one-shot parse,
 * and held bytes never exceed the format's bound.
 *
 * Cost: streaming a long reply byte by byte is a constant factor of parsing it once, and cost is
 * linear in reply length.
 */
#include "rad_test.h"

#include "chatdiff.h"
#include "rad_builder.h"
#include "sample/gbnf.h"
#include "sample/vocab_view.h"
#include "server/adapters.h"
#include "server/json_partial.h"
#include "server/oai.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::chatdiff;
using json = nlohmann::ordered_json;

/* ================================================================== fixtures */

static std::string fixture(const std::string& dir) {
    std::ifstream f(std::string(RAD_CHAT_FIXTURES) + "/" + dir + "/chat_template.jinja");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* One bench per (model, request), built once. A fixture that does not load fails rather than
 * skips, since the templates are in the repository. */
static Bench* bench(const Model& m, const ChatRequest& rq, const json& tools = agent_tools()) {
    static std::map<std::string, std::unique_ptr<Bench>> cache;
    const std::string key = m.name + "|" + describe(rq) + "|" + rq.schema + "|" + tools.dump();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    auto b = std::make_unique<Bench>();
    std::string why;
    if (b->load(m, fixture(m.dir), &why) < 0 || b->bind(rq, tools, &why) < 0) {
        ::radtest::fail(__FILE__, __LINE__, m.name + " (" + describe(rq) + "): " + why);
        return nullptr;
    }
    return (cache[key] = std::move(b)).get();
}

/* The parser for format `f` under `ro`, as the server builds it: the format compiled, then
 * bound to the request. Null after a failed check when either step refuses. */
static std::shared_ptr<const ReplyParser> parser_of(const ChatFormat& f, const ReplyOptions& ro) {
    std::string why;
    std::shared_ptr<const ReplyFormat> rf;
    std::shared_ptr<const ReplyParser> rp;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    if (rf) CHECK_OK(ReplyParser::create(rf, ro, &rp, &why));
    return rp;
}

/* A reply quoted for a failure message. Mutated replies need not be UTF-8. */
static std::string quote(const std::string& s) {
    return json(s).dump(-1, ' ', false, json::error_handler_t::replace);
}

/* The requests every corpus is read under: an agent's request, and each switch that changes what
 * a reply means. */
static std::vector<ChatRequest> requests() {
    std::vector<ChatRequest> v;
    v.push_back(ChatRequest{});
    ChatRequest r;
    r = ChatRequest{}; r.tools = false;       v.push_back(r);
    r = ChatRequest{}; r.thinking = false;    v.push_back(r);
    r = ChatRequest{}; r.reasoning = false;   v.push_back(r);
    r = ChatRequest{}; r.parallel = false;    v.push_back(r);
    return v;
}

/* ================================================================== the streaming checks */

static std::vector<size_t> every(size_t n, size_t step) {
    std::vector<size_t> c;
    for (size_t i = step; i < n; i += step) c.push_back(i);
    return c;
}

/* Token-sized pieces, on code-point boundaries, as the decoder hands them over. */
static std::vector<size_t> tokenish(const std::string& s, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<size_t> c;
    size_t at = 0;
    while (at < s.size()) {
        at += 1 + rng() % 6;
        while (at < s.size() && ((unsigned char)s[at] & 0xC0) == 0x80) ++at;
        if (at < s.size()) c.push_back(at);
    }
    return c;
}

/* Every relevant cut of the reply, against its one-shot parse. Two-way splits are exhaustive for
 * replies up to `split_cap` bytes. */
static std::string stream_all(const ReplyParser& p, const std::string& text, size_t split_cap = 4096) {
    const ReplyMessage expect = reply_parse(p, text);
    std::string err = check_stream(p, text, {}, expect);
    if (err.empty()) err = check_stream(p, text, every(text.size(), 1), expect);
    for (uint32_t seed = 1; err.empty() && seed <= 3; ++seed)
        err = check_stream(p, text, tokenish(text, seed), expect);
    if (text.size() <= split_cap)
        for (size_t k = 1; err.empty() && k < text.size(); ++k)
            err = check_stream(p, text, { k }, expect);
    return err;
}

/* Read `text` under bench `b`: the differential verdict must be allowed and every cut must stream
 * to the same answer. Returns the verdict for further checks. */
static Verdict run(Bench* b, const std::string& text, const char* what, size_t split_cap = 4096) {
    Verdict v;
    if (!b) return v;
    v = compare(*b, text);
    const std::string j = judge(v);
    if (!j.empty())
        ::radtest::fail(__FILE__, __LINE__, std::string(what) + " [" + b->model().name + ", " +
                        describe(b->request()) + "]: " + j + "\n  reply: " +
                        quote(text) + "\n" + v.detail);
    const std::string s = stream_all(b->parser(), text, split_cap);
    if (!s.empty())
        ::radtest::fail(__FILE__, __LINE__, std::string(what) + " [" + b->model().name + ", " +
                        describe(b->request()) + "]: " + s + "\n  reply: " + quote(text));
    ++::radtest::checks();
    return v;
}

/* ================================================================== corpus 1: rendered replies
 *
 * An assistant message rendered through the model's template is the text the model is trained to
 * write. Both parsers must agree, read it within the grammar, and return the message's calls. */

/* An argument as rendered and read back: a string is its text, anything else JSON, including
 * booleans and null, which Jinja renders Python-style (`True`, `None`). Exception: MiniCPM's
 * template prints values with a bare `{{ param_value }}`, so lists and objects lose their brackets;
 * those are covered by the differential check rather than compared with the message. */
static bool round_trips(const json& want, const json& got, bool containers) {
    if (!got.is_object() || want.size() != got.size()) return false;
    for (auto it = want.begin(); it != want.end(); ++it) {
        if (!got.contains(it.key())) return false;
        if (!containers && it.value().is_structured()) continue;
        if (got.at(it.key()) != it.value()) return false;
    }
    return true;
}

/* A rendered reply the format's grammar does not admit, and why: without parallel calls the
 * grammar admits one call, and MiniCPM's grammar ends a value at the first `</param>`, so a CDATA
 * value containing that tag is split there by both parsers. */
static bool outside_grammar(const Model& m, const ChatRequest& rq, const json& msg) {
    const size_t ncalls = msg.contains("tool_calls") ? msg["tool_calls"].size() : 0;
    if (!rq.parallel && ncalls > 1) return true;
    if (!is_qwen(m) && msg.dump().find("</param>") != std::string::npos) return true;
    return false;
}

TEST(rendered_replies_read_back_identically_under_every_request) {
    const std::vector<json> corpus = rendered_corpus();
    for (const Model& m : models()) {
        for (const ChatRequest& rq : requests()) {
            Bench* b = bench(m, rq);
            if (!b) continue;
            for (size_t i = 0; i < corpus.size(); ++i) {
                const json& msg = corpus[i];
                if (!rq.tools && msg.contains("tool_calls")) continue;
                const std::string reply = b->render_reply(msg);
                if (reply.empty()) continue;   /* the template does not continue its prompt */
                const std::string what = "rendered reply " + std::to_string(i);
                const Verdict v = run(b, reply, what.c_str());
                if (outside_grammar(m, rq, msg)) continue;
                if (!v.strict || !v.same)
                    ::radtest::fail(__FILE__, __LINE__, what + " [" + m.name + ", " + describe(rq) +
                                    "] did not read within the grammar\n  reply: " + quote(reply) +
                                    "\n" + compare(*b, reply).detail);
                const ReplyMessage r = b->new_parse(reply);
                const size_t ncalls = msg.contains("tool_calls") ? msg["tool_calls"].size() : 0;
                CHECK_EQ(r.calls.size(), ncalls);
                for (size_t k = 0; k < std::min(ncalls, r.calls.size()); ++k) {
                    const json& want = msg["tool_calls"][k]["function"];
                    CHECK_EQ(r.calls[k].name, want["name"].get<std::string>());
                    CHECK(r.calls[k].closed);
                    const json got = json::parse(r.calls[k].arguments, nullptr, false);
                    if (!is_qwen(m) && got.is_discarded()) continue;
                    if (!round_trips(json::parse(want["arguments"].get<std::string>()), got,
                                     is_qwen(m)))
                        ::radtest::fail(__FILE__, __LINE__, what + " [" + m.name +
                                        "]: arguments came back as " + r.calls[k].arguments);
                }
            }
        }
    }
}

/* Every rendered reply truncated after every byte reads as the legacy path reads it: a call in
 * progress stays a call with the arguments that arrived. */
TEST(every_truncation_of_a_rendered_reply_reads_as_the_old_path_did) {
    const std::vector<json> corpus = rendered_corpus();
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        for (const json& msg : corpus) {
            const std::string reply = b->render_reply(msg);
            for (size_t n = 0; n <= reply.size(); ++n) {
                /* The decoder hands over whole code points only. */
                if (n < reply.size() && ((unsigned char)reply[n] & 0xC0) == 0x80) continue;
                const Verdict v = compare(*b, reply.substr(0, n));
                const std::string j = judge(v);
                if (!j.empty()) {
                    ::radtest::fail(__FILE__, __LINE__, m.name + ": " + j + "\n  reply: " +
                                    quote(reply.substr(0, n)) + "\n" + v.detail);
                    break;
                }
            }
            ++::radtest::checks();
        }
    }
}

/* ================================================================== corpus 2: adversarial replies
 *
 * Hand-written in each format's spelling: shapes models write, and those a streaming parser most
 * easily gets wrong. */

TEST(adversarial_replies_agree_with_the_old_path_or_differ_as_documented) {
    for (const Model& m : models()) {
        for (const ChatRequest& rq : requests()) {
            Bench* b = bench(m, rq);
            if (!b) continue;
            for (const Adversarial& a : adversarial(m)) run(b, a.text, a.name.c_str());
        }
    }
}

/* Adversarial shapes whose reading is a design decision, with the decided result. */
static ReplyMessage read_adv(const Model& m, const std::string& name) {
    Bench* b = bench(m, ChatRequest{});
    for (const Adversarial& a : adversarial(m))
        if (a.name == name) return b->new_parse(a.text);
    ::radtest::fail(__FILE__, __LINE__, "no adversarial case " + name);
    return ReplyMessage{};
}

TEST(the_decided_shapes_read_as_decided) {
    for (const Model& m : models()) {
        const bool q = is_qwen(m);
        ReplyMessage r;

        /* A reply cut off inside a call is a call, open, with what arrived. */
        r = read_adv(m, "a truncated reply inside a call");
        CHECK_EQ(r.calls.size(), (size_t)1);
        if (!r.calls.empty()) CHECK(!r.calls[0].closed);

        /* Qwen's reasoning ends where a call starts; MiniCPM's analysis names no call opener to
         * end it at, so an unclosed block keeps the call as reasoning, as the legacy path does. */
        r = read_adv(m, "unclosed think before a call");
        CHECK_EQ(r.calls.size(), q ? (size_t)1 : (size_t)0);

        /* A number with units is not a number: the argument is the string. */
        r = read_adv(m, "an integer with units");
        CHECK_EQ(r.calls.size(), (size_t)1);
        if (!r.calls.empty())
            CHECK_EQ(r.calls[0].arguments, std::string("{\"command\":\"sleep\",\"timeout\":\"30 seconds\"}"));

        /* A typed argument always reads as JSON: Python literals become JSON's (inside containers
         * too), an empty value becomes null, and text that is no JSON value becomes a string.
         * Qwen's format reads non-string arguments as JSON; MiniCPM's reads every one as text. */
        if (q) {
            r = read_adv(m, "Python literals");
            CHECK_EQ(r.calls.size(), (size_t)1);
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"verbose\":true,\"nothing\":null}"));
            r = read_adv(m, "a Python dict with literals");
            if (!r.calls.empty()) {
                const json a = json::parse(r.calls[0].arguments, nullptr, false);
                CHECK(!a.is_discarded());
                if (!a.is_discarded()) {
                    CHECK(a["extra"]["a"] == json(true));
                    CHECK(a["extra"]["b"].is_null());
                    CHECK(a["extra"]["c"] == json("False"));
                }
            }
            r = read_adv(m, "an empty typed value");
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"level\":null}"));
            r = read_adv(m, "a typed value that is no JSON");
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"verbose\":\"yes please\"}"));
            r = read_adv(m, "a number that is not one");
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"path\":\"/a\",\"limit\":\"01\"}"));
            /* A raw line break or tab inside a string is escaped, since JSON parsers refuse the
             * raw character. */
            r = read_adv(m, "JSON strings holding a raw line break and tab");
            CHECK_EQ(r.calls.size(), (size_t)1);
            if (!r.calls.empty()) {
                const json a = json::parse(r.calls[0].arguments, nullptr, false);
                CHECK(!a.is_discarded());
                if (!a.is_discarded()) {
                    CHECK(a["extra"]["a"] == json("line1\nline2"));
                    CHECK(a["extra"]["b"] == json("\tx"));
                }
            }
            /* A container is normalised JSON, or else its text as a string (trailing comma, bare
             * word, missing bracket), so the arguments stay JSON. */
            r = read_adv(m, "Python containers that are no JSON");
            CHECK_EQ(r.calls.size(), (size_t)1);
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"extra\":\"{'a': 1, 'b': [2, 3],}\","
                                                           "\"tags\":\"['x', nan]\"}"));
            r = read_adv(m, "a container missing its close");
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"extra\":\"{'a': [1, 2}\"}"));
            r = read_adv(m, "an escaped quote in a double-quoted string");
            if (!r.calls.empty())
                CHECK_EQ(r.calls[0].arguments, std::string("{\"tags\":[\"it's\", \"x\"]}"));
            /* The grammar admits a number beyond double range and an unpaired surrogate escape,
             * which clients' JSON parsers refuse, so such a value becomes its text. */
            r = read_adv(m, "JSON a client cannot read");
            if (!r.calls.empty()) {
                const json a = json::parse(r.calls[0].arguments, nullptr, false);
                CHECK(!a.is_discarded());
                if (!a.is_discarded()) {
                    CHECK(a["level"] == json("1e400"));
                    CHECK(a["extra"] == json(R"({"a": "\ud800"})"));
                }
            }
        }

        /* A Python-quoted array is normalised to JSON. */
        r = read_adv(m, "a Python-quoted array");
        if (!r.calls.empty()) {
            const json a = json::parse(r.calls[0].arguments, nullptr, false);
            CHECK(!a.is_discarded());
            if (!a.is_discarded()) CHECK_EQ(a["edits"][0]["newText"].get<std::string>(), std::string("b's"));
        }

        /* A function that is not a tool is still delivered as a call. */
        r = read_adv(m, "a function that is not a tool");
        CHECK_EQ(r.calls.size(), (size_t)1);
        if (!r.calls.empty()) CHECK_EQ(r.calls[0].name, std::string("which gcc cc clang make"));

        /* CDATA is punctuation. */
        r = read_adv(m, "CDATA with brackets and tags");
        if (r.calls.size() == 1) {
            const json a = json::parse(r.calls[0].arguments, nullptr, false);
            CHECK_EQ(a.value("content", std::string()),
                     std::string("if (a[b[0]]) { x = \"]]\"; }\n  <tag>\n    indented\n"));
        }

        /* Text after a call is content, whichever format. */
        r = read_adv(m, "calls then trailing text");
        CHECK_EQ(r.calls.size(), (size_t)1);
        CHECK_EQ(r.content, std::string("Done now."));
        CHECK_EQ(r.reasoning, std::string("Thinking it over.\n"));

        /* A close tag without the newline the template writes after it still closes the call. */
        if (q) {
            r = read_adv(m, "a close without its newline");
            CHECK_EQ(r.calls.size(), (size_t)1);
            if (!r.calls.empty()) {
                CHECK(r.calls[0].closed);
                CHECK_EQ(r.calls[0].arguments, std::string("{\"command\":\"ls\"}"));
            }
            CHECK(r.content.empty());

            /* A call whose own close is missing is ended by its wrapper's, and the next call is
             * a call of its own rather than more arguments for this one. */
            r = read_adv(m, "a call closed by its wrapper alone");
            CHECK_EQ(r.calls.size(), (size_t)2);
            if (r.calls.size() == 2) {
                CHECK(r.calls[0].closed);
                CHECK_EQ(r.calls[0].arguments, std::string("{\"command\":\"ls\"}"));
                CHECK_EQ(r.calls[1].name, std::string("read"));
                CHECK_EQ(r.calls[1].arguments, std::string("{\"path\":\"/a\"}"));
            }
            CHECK(r.content.empty());
        }

        /* The terminator is not the answer, nor is trailing whitespace. */
        CHECK_EQ(read_adv(m, "a turn terminator at the end").content, std::string("Answer."));
        CHECK_EQ(read_adv(m, "trailing spaces and newlines").content, std::string("Answer."));
    }
}

/* An argument name repeating its tag's opening, `<parameter=parameter=command>` (models write it
 * and the undeclared arm admits it), is read leniently as the argument it names; otherwise the
 * tool would get `parameter=command` and no command. A name that does not resolve to a declared
 * one stays as written. */
TEST(an_argument_name_repeating_its_tag_is_the_argument_it_names) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string T = "Thinking it over.\n</think>\n\n";
        const bool q = is_qwen(m);
        const std::string rep = q ? "parameter=" : "param name=\"";
        const std::string text = T + C(m, "bash", { { rep + "command", "ls -la" },
                                                    { rep + rep + "timeout", "30" },
                                                    { rep + "nope", "x" } });
        bool strict = true;
        const ReplyMessage r = b->new_parse(text, &strict);
        CHECK(!strict);
        REQUIRE(r.calls.size() == 1);
        const bool typed = b->format().other_value == ChatValue::Json;
        CHECK_EQ(r.calls[0].arguments, "{\"command\":\"ls -la\",\"timeout\":" +
                                           std::string(typed ? "30" : "\"30\"") + "," +
                                           json(rep + "nope").dump() + ":\"x\"}");
        const std::string s = stream_all(b->parser(), text);
        CHECK(s.empty());
        if (!s.empty()) fprintf(stderr, "    %s\n", s.c_str());
    }
}

/* A call the reply ended inside reads back as content in its written form: the content before it,
 * then the call's bytes, never the reasoning or an earlier closed call. A buffered response uses
 * this when the call cannot be reported as a call. */
TEST(a_call_the_reply_ended_inside_reads_back_as_its_text) {
    auto trimmed = [](std::string s) {
        while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
        return s;
    };
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string T = "Thinking it over.\n</think>\n\n";
        const std::string done = C(m, "bash", { { "command", "ls" } });
        const std::string call = C(m, "write", { { "path", "/a" }, { "content", "partial fi" } });
        for (const size_t cut : { call.find("partial") + 3, call.find("</function>") + 3 }) {
            const std::string open = call.substr(0, cut);
            struct Case { std::string text, content; size_t calls; };
            std::vector<Case> cases = {
                { T + "Writing.\n" + open, "Writing.\n" + trimmed(open), 1 },
                { T + "Writing.\n" + done + "\n" + open, "Writing.\n" + trimmed(open), 2 },
                { T + open, trimmed(open), 1 },
            };
            /* Qwen's reasoning block ends where a call starts. */
            if (is_qwen(m)) cases.push_back({ "I will write it.\n" + open, trimmed(open), 1 });
            for (const Case& k : cases) {
                std::unique_ptr<ReplyStream> s = b->parser().open();
                for (size_t i = 0; i < k.text.size(); ++i)
                    s->feed(std::string_view(k.text).substr(i, 1), nullptr);
                s->finish(nullptr);
                const ReplyMessage& r = s->message();
                CHECK_EQ(r.calls.size(), k.calls);
                if (r.calls.size() != k.calls) continue;
                CHECK(!r.calls.back().closed);
                if (k.calls == 2) CHECK(r.calls[0].closed);
                CHECK_EQ(s->content_with_open_call(k.text), k.content);
                CHECK(stream_all(b->parser(), k.text).empty());
            }
        }
        /* With no open call the content is the content. */
        const std::string whole = T + "Writing.\n" + done;
        std::unique_ptr<ReplyStream> s = b->parser().open();
        s->feed(whole, nullptr);
        s->finish(nullptr);
        CHECK_EQ(s->content_with_open_call(whole), std::string("Writing."));
    }
}

/* ================================================================== calls without the wrapper
 *
 * Qwen's lazy grammar engages only at `<tool_call>\n`, so a call written without that exact
 * opener is unconstrained and the likeliest to arrive malformed. Agent transcripts contain a bare
 * `<function=NAME>` ... `</function>`, or `<tool_call>` run straight into `<function=`. The legacy
 * path lifts these in a second pass; the parser reads them as calls in one pass once what follows
 * the name tag shows a call, and reports that the reply left its grammar.
 *
 * Each shape is read whole and at every cut, and compared with the legacy path. */

static std::string qbare(const std::string& name,
                         const std::vector<std::pair<std::string, std::string>>& args) {
    std::string s = "<function=" + name + ">\n";
    for (const auto& a : args) s += "<parameter=" + a.first + ">\n" + a.second + "\n</parameter>\n";
    return s + "</function>";
}

struct Shape {
    std::string name, text;
    size_t calls;
    bool strict;       /* the reply stays within the grammar: nothing was read as a loose call */
    /* The opener is inside the reasoning block the generation prompt opened. With thinking off
     * the prompt already closed it, so the text is content and the call is a call. */
    bool in_reasoning = false;
};

static std::vector<Shape> bare_shapes() {
    const std::string T = "Thinking it over.\n</think>\n\n";
    const std::string ls = qbare("bash", { { "command", "ls -la /work" } });
    const std::string rd = qbare("read", { { "path", "/work/a.py" } });
    return {
        { "a bare call", T + ls, 1, false },
        { "a bare call followed by its newline", T + ls + "\n", 1, false },
        { "a bare call after prose", T + "Let me look at the tree first.\n" + ls, 1, false },
        { "a bare call between sentences", T + "Looking.\n" + ls + "\nThen the file.", 1, false },
        { "two bare calls", T + ls + "\n" + rd, 2, false },
        { "two bare calls with a sentence between", T + ls + "\nNow the file.\n" + rd, 2, false },
        { "a bare call with no arguments", T + "<function=which gcc cc>\n</function>", 1, false },
        /* The first argument's name confirms the call, so it is held while read: a tag name of
         * any length, not a delimiter. */
        { "a bare call whose first argument name is long",
          T + "<function=bash>\n<parameter=" + std::string(200, 'x') + ">\nls\n</parameter>\n</function>",
          1, false },
        /* Unconstrained, the model may omit the newlines the template puts around each tag, so
         * that whitespace is optional in a loose call. */
        { "a bare call written without newlines",
          T + "<function=bash><parameter=command>ls -la</parameter></function>", 1, false },
        { "the wrapper without its newline",
          T + "<tool_call>" + ls + "\n</tool_call>", 1, false },
        { "the wrapper followed by a space",
          T + "Listing.\n<tool_call> " + ls + "\n</tool_call>\nDone.", 1, false },
        { "two wrappers without their newlines",
          T + "<tool_call>" + ls + "\n</tool_call>\n<tool_call>" + rd + "\n</tool_call>", 2, false },
        { "a bare opener inside a closed reasoning block",
          "I will list it.\n" + ls + "\n</think>\n\nDone.", 0, true, true },
        { "a bare opener inside an unclosed reasoning block", "I will list it.\n" + ls, 0, true,
          true },
        { "the bare opener in prose", T + "The `<function=f>` tag names the tool.", 0, true },
        { "the wrapper in prose", T + "Write `<tool_call>` and then the call.", 0, true },
        { "the wrapper and opener in prose", T + "Write `<tool_call><function=f>` first.", 0, true },
        { "a bare opener whose tag never closes", T + "See <function=bash and more.", 0, true },
        { "a bare opener at the very end", T + "Next: <function=ba", 0, true },
        /* An argument name that runs into a tag is not an argument; the opener was text. */
        { "a bare opener whose argument name runs into a tag",
          T + "<function=bash>\n<parameter=com\nls\n</parameter>\n</function>", 0, true },
        { "a bare opener whose long argument name runs into a tag",
          T + "<function=bash>\n<parameter=" + std::string(200, 'x') + "\nls\n</parameter>\n</function>",
          0, true },
        { "a mangled bare opener before a wrapped call",
          T + "<function=bash>\n<parameter=com\nls\n</parameter>\n</function>\n<tool_call>\n" + rd +
              "\n</tool_call>", 1, true },
        { "the wrapper at the very end", T + "Next: <tool_call>", 0, true },
    };
}

TEST(a_qwen_call_without_its_wrapper_is_a_call_as_the_old_path_read_it) {
    for (const Model& m : models()) {
        if (!is_qwen(m)) continue;
        for (const ChatRequest& rq : { ChatRequest{}, requests()[2] }) {
            Bench* b = bench(m, rq);
            if (!b) continue;
            for (Shape s : bare_shapes()) {
                if (s.in_reasoning && !rq.thinking) { s.calls = 1; s.strict = false; }
                const Verdict v = run(b, s.text, s.name.c_str());
                const std::string where = s.name + " [" + m.name + ", " + describe(rq) + "]";
                if (!v.same)
                    ::radtest::fail(__FILE__, __LINE__, where + ": differs from the old path\n" +
                                    v.detail);
                if (v.strict != s.strict)
                    ::radtest::fail(__FILE__, __LINE__, where + ": strict " +
                                    std::to_string(v.strict));
                bool strict = true;
                const ReplyMessage r = b->new_parse(s.text, &strict);
                CHECK_EQ(r.calls.size(), s.calls);
                for (const ReplyCall& c : r.calls) {
                    CHECK(c.closed);
                    CHECK(server::json_complete(c.arguments));
                }
                /* No part of a loosely read call is left in the content. */
                if (s.calls > 0 && !s.strict) {
                    CHECK(r.content.find("<function=") == std::string::npos);
                    CHECK(r.content.find("<parameter=") == std::string::npos);
                    CHECK(r.content.find("<tool_call>") == std::string::npos);
                }
            }
        }
    }
}

/* Where the legacy path answers differently, and why. Both answers are asserted so a change to
 * either is noticed. */
TEST(a_bare_call_differs_from_the_old_path_only_where_the_old_path_was_wrong) {
    for (const Model& m : models()) {
        if (!is_qwen(m)) continue;
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string T = "Thinking it over.\n</think>\n\n";

        /* The legacy lifter has no schema, so typed arguments come back as strings, and it trims
         * every value, losing an edit's first-line indentation (vendor/patches/0005 fixes this on
         * the grammar path). The parser types values from the schema and strips only the
         * format's delimiting newline, as inside the wrapper. */
        {
            const std::string text =
                T + qbare("read", { { "path", "/work/a.py" }, { "limit", "40" } }) + "\n" +
                qbare("write", { { "path", "/work/b.py" }, { "content", "    return a + b" } });
            run(b, text, "typed and indented");
            const legacy::Answer o = b->old_parse(text);
            const ReplyMessage r = b->new_parse(text);
            REQUIRE(o.calls.size() == 2 && r.calls.size() == 2);
            CHECK_EQ(o.calls[0].arguments, std::string("{\"path\":\"/work/a.py\",\"limit\":\"40\"}"));
            CHECK_EQ(r.calls[0].arguments, std::string("{\"path\":\"/work/a.py\",\"limit\":40}"));
            CHECK_EQ(o.calls[1].arguments,
                     std::string("{\"path\":\"/work/b.py\",\"content\":\"return a + b\"}"));
            CHECK_EQ(r.calls[1].arguments,
                     std::string("{\"path\":\"/work/b.py\",\"content\":\"    return a + b\"}"));
            /* Inside the wrapper the same call gives the same arguments. */
            const ReplyMessage w = b->new_parse(
                T + "<tool_call>\n" + qbare("read", { { "path", "/work/a.py" }, { "limit", "40" } }) +
                "\n</tool_call>");
            REQUIRE(w.calls.size() == 1);
            CHECK_EQ(w.calls[0].arguments, r.calls[0].arguments);
        }

        /* The legacy lifter runs only when nothing else parses as a call, so it leaves a bare
         * call beside a wrapped one in the content as XML. */
        {
            const std::string text = T + qbare("bash", { { "command", "ls" } }) +
                                     "\n<tool_call>\n" + qbare("read", { { "path", "/a" } }) +
                                     "\n</tool_call>";
            const Verdict v = run(b, text, "bare then wrapped");
            CHECK(!v.same);
            CHECK(!v.strict);
            const legacy::Answer o = b->old_parse(text);
            const ReplyMessage r = b->new_parse(text);
            CHECK_EQ(o.calls.size(), (size_t)1);
            CHECK(o.content.find("<function=bash>") != std::string::npos);
            CHECK_EQ(r.calls.size(), (size_t)2);
            CHECK(r.content.empty());
        }

        /* The legacy lifter drops a whole call when a later argument name runs into a tag. The
         * call has already streamed by then, so it ends at that argument and the rest is
         * content. */
        {
            const std::string text = T + "<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                                         "<parameter=x<y>\n1\n</parameter>\n</function>";
            const Verdict v = run(b, text, "a later argument name runs into a tag");
            CHECK(!v.same);
            CHECK(!v.strict);
            const ReplyMessage r = b->new_parse(text);
            REQUIRE(r.calls.size() == 1);
            CHECK(r.calls[0].closed);
            CHECK_EQ(r.calls[0].arguments, std::string("{\"command\":\"ls\"}"));
            CHECK_EQ(r.content, std::string("<parameter=x<y>\n1\n</parameter>\n</function>"));
            CHECK(b->old_parse(text).calls.empty());
        }

        /* The legacy lifter leaves a bare call cut off mid-argument as content. A stream has
         * already announced the call, so it is a call with unfinished arguments, as a wrapped
         * call cut off there is. */
        {
            const std::string text = T + "<function=bash>\n<parameter=command>\nls -";
            const Verdict v = run(b, text, "truncated bare call");
            CHECK(!v.same);
            const ReplyMessage r = b->new_parse(text);
            REQUIRE(r.calls.size() == 1);
            CHECK(!r.calls[0].closed);
            CHECK_EQ(r.calls[0].arguments, std::string("{\"command\":\"ls -"));
            CHECK(b->old_parse(text).calls.empty());
        }
    }
}

/* MiniCPM's calls are unwrapped: its opener is `<function name="`, which its grammar already
 * opens a call with. So none of the above applies: a call is strict and Qwen's spellings in its
 * content are content. */
TEST(minicpm_reads_with_no_loose_opener) {
    const Model& m = models()[2];
    REQUIRE(!is_qwen(m));
    Bench* b = bench(m, ChatRequest{});
    REQUIRE(b != nullptr);
    CHECK(b->format().call_open.empty());
    const std::string T = "Thinking it over.\n</think>\n\n";
    const std::vector<std::pair<std::string, size_t>> replies = {
        { T + "<function name=\"bash\"><param name=\"command\">ls</param></function>", 1 },
        { T + "Look.\n<function name=\"bash\"><param name=\"command\">ls</param></function>\n"
              "<function name=\"read\"><param name=\"path\">/a</param></function>", 2 },
        { T + "Text with `<function=f>` and `<tool_call>` in it.", 0 },
    };
    for (const auto& [text, n] : replies) {
        const Verdict v = run(b, text, "minicpm");
        CHECK(v.same);
        CHECK(v.strict);
        CHECK_EQ(b->new_parse(text).calls.size(), n);
    }
    /* A Qwen-spelled call in a MiniCPM reply is content, left in place. */
    const std::string q = T + "<tool_call>\n" + qbare("bash", { { "command", "ls" } }) + "\n</tool_call>";
    bool strict = false;
    const ReplyMessage r = b->new_parse(q, &strict);
    CHECK(strict);
    CHECK(r.calls.empty());
    CHECK(r.content.find("<function=bash>") != std::string::npos);
}

/* ================================================================== corpus 3: fuzz
 *
 * Random replies assembled from each format's fragments (to reach deep states) and random byte
 * mutations of every corpus reply. The verdict must be allowed and every cut must stream to the
 * same answer. Mutations may break UTF-8, which the decoder always sanitises, so the comparison
 * runs on sanitised text and the raw bytes are fed only to check invariants. */

TEST(fuzzed_replies_never_break_the_contract) {
    std::mt19937 rng(20260924);
    std::map<std::string, int> tally;
    for (const Model& m : models()) {
        const std::vector<std::string> frag = fragments(m);
        std::vector<std::string> seeds;
        for (const json& msg : rendered_corpus()) seeds.push_back(bench(m, ChatRequest{})->render_reply(msg));
        for (const Adversarial& a : adversarial(m)) seeds.push_back(a.text);
        for (const ChatRequest& rq : { ChatRequest{}, requests()[2], requests()[4] }) {
            Bench* b = bench(m, rq);
            if (!b) continue;
            for (int i = 0; i < 1500; ++i) {
                std::string text;
                if (i % 2 == 0) {
                    const int n = (int)(rng() % 24);
                    for (int k = 0; k < n; ++k) text += frag[rng() % frag.size()];
                } else {
                    text = mutate(rng, seeds[rng() % seeds.size()], frag);
                }
                /* Raw bytes: no crash, and the stream agrees with itself. */
                const std::string raw = stream_all(b->parser(), text, 0);
                if (!raw.empty())
                    ::radtest::fail(__FILE__, __LINE__, m.name + " raw fuzz: " + raw + "\n  reply: " +
                                    quote(text));
                const std::string clean = sanitize(text);
                const Verdict v = compare(*b, clean);
                const std::string j = judge(v);
                if (!j.empty()) {
                    ::radtest::fail(__FILE__, __LINE__, m.name + " fuzz (" + describe(rq) + "): " + j +
                                    "\n  reply: " + quote(clean) + "\n" + v.detail);
                    return;
                }
                ++tally[v.same ? "same" : v.bare_lift ? "bare-lift" : "recovered"];
                ++::radtest::checks();
            }
        }
    }
    fprintf(stderr, "    fuzz: %d identical, %d recovered where the old grammar failed, %d bare "
                    "calls the old lifter took from content\n",
            tally["same"], tally["recovered"], tally["bare-lift"]);
}

/* ================================================================== response formats
 *
 * With a response format the reply is one JSON document, optionally fenced, and is the content. */
TEST(response_format_replies_read_as_the_old_path_read_them) {
    const std::string schema = R"({"type":"object","properties":{"answer":{"type":"string"},)"
                               R"("n":{"type":"integer"}},"required":["answer"]})";
    const std::vector<std::string> replies = {
        "Thinking.\n</think>\n\n{\"answer\": \"yes\", \"n\": 3}",
        "Thinking.\n</think>\n\n```json\n{\"answer\": \"yes\"}\n```",
        "Thinking.\n</think>\n\n  {\"answer\": \"a ``` b\"}  \n",
        "Thinking.\n</think>\n\n{\"answer\": \"trunc",
        "Thinking.\n</think>\n\n```json\n{\"answer\": \"x\"}",
        "Thinking.\n</think>\n\n{\"answer\": \"x\"} trailing",
        "Thinking.\n</think>\n\nnot json at all",
        "Thinking.\n</think>\n\n```JSON\n{}\n```",
        "Thinking.\n</think>\n\n```json\n{}\n```\n",
        "unclosed <tool_call>\n{\"answer\": 1}",
        "",
    };
    for (const Model& m : models()) {
        for (bool tools : { false, true }) {
            ChatRequest rq;
            rq.tools = tools;
            rq.schema = schema;
            Bench* b = bench(m, rq);
            if (!b) continue;
            for (const std::string& r : replies) run(b, r, "response format");
        }
    }
}

/* ================================================================== the grammar
 *
 * For a template-analysed format the grammar generated from the description equals the vendored
 * generator's byte for byte and accepts and rejects the same call text. */

static std::vector<json> tool_sets() {
    std::vector<json> v;
    v.push_back(agent_tools());
    v.push_back(json::parse(R"([{"type":"function","function":{"name":"get_weather",)"
                            R"("parameters":{"type":"object","properties":{"city":{"type":"string"}},)"
                            R"("required":["city"]}}}])"));
    v.push_back(json::parse(R"([{"type":"function","function":{"name":"my.tool_v2",)"
                            R"("parameters":{"type":"object","properties":{"n":{"type":"integer"},)"
                            R"("ref":{"$ref":"#/$defs/r"}},"$defs":{"r":{"type":"object",)"
                            R"("properties":{"x":{"type":"string"}}}}}}},)"
                            R"({"type":"function","function":{"name":"bare"}}])"));
    return v;
}

/* Rules in only one grammar, marked - for vendored and + for generated. Both list rules sorted by
 * name, one per line. */
static std::string line_diff(const std::string& a, const std::string& b) {
    auto lines = [](const std::string& s) {
        std::vector<std::string> v;
        std::istringstream in(s);
        for (std::string l; std::getline(in, l);) v.push_back(l);
        return v;
    };
    const std::vector<std::string> la = lines(a), lb = lines(b);
    std::string out;
    for (const std::string& l : la)
        if (std::find(lb.begin(), lb.end(), l) == lb.end()) out += "  - " + l + "\n";
    for (const std::string& l : lb)
        if (std::find(la.begin(), la.end(), l) == la.end()) out += "  + " + l + "\n";
    return out.empty() ? std::string("  (same rules, different text)\n") : out;
}

TEST(the_generated_grammar_is_the_vendored_grammar) {
    int compared = 0;
    for (const Model& m : models()) {
        for (const json& tools : tool_sets()) {
            for (bool parallel : { true, false }) {
                for (const std::string& schema : { std::string(), std::string(R"({"type":"object"})") }) {
                    ChatRequest rq;
                    rq.parallel = parallel;
                    rq.schema = schema;
                    Bench* b = bench(m, rq, tools);
                    if (!b) continue;
                    ChatGrammarRequest gq;
                    gq.tools = &tools;
                    const json js = schema.empty() ? json() : json::parse(schema);
                    gq.json_schema = schema.empty() ? nullptr : &js;
                    gq.generation_prompt = b->prompt().generation_prompt;
                    gq.parallel_tool_calls = parallel;
                    ChatGrammar g;
                    std::string why;
                    CHECK_OK(chat_format_grammar(b->format(), gq, &g, &why));
                    if (g.text != b->prompt().grammar)
                        ::radtest::fail(__FILE__, __LINE__, m.name + " (" + describe(rq) + "):\n" +
                                        line_diff(b->prompt().grammar, g.text));
                    CHECK_EQ(g.lazy, b->prompt().grammar_lazy);
                    std::vector<std::string> words;
                    for (const auto& t : b->prompt().grammar_triggers) words.push_back(t.value);
                    CHECK(g.trigger_words == words);
                    ++compared;
                }
            }
        }
    }
    CHECK(compared > 0);
}

/* Both grammars admit every rendered reply's call section, and a mutated one by both or
 * neither. */
TEST(the_generated_grammar_accepts_what_the_vendored_one_accepts) {
    log_set_level(Log::Error);
    std::mt19937 rng(7);
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b || b->prompt().grammar.empty()) continue;   /* MiniCPM: no lazy grammar */
        ChatGrammarRequest gq;
        gq.tools = &agent_tools();
        gq.generation_prompt = b->prompt().generation_prompt;
        ChatGrammar g;
        std::string why;
        CHECK_OK(chat_format_grammar(b->format(), gq, &g, &why));
        const std::string opener = b->format().call_open;
        const std::vector<std::string> frag = fragments(m);
        /* A vocabulary of printable bytes: the grammar walks text and no mask is requested. */
        static const SimpleVocab vocab = [] {
            SimpleVocab v;
            for (int c = 32; c < 127; ++c) v.add(std::string(1, (char)c));
            v.add("\n");
            return v;
        }();
        auto admits = [](const std::string& gbnf, const std::string& text) {
            std::unique_ptr<GbnfGrammar> gr;
            std::string err;
            if (GbnfGrammar::create(gbnf, "root", &vocab, false, {}, {}, &gr, &err) < 0) return -1;
            if (gr->accept_str(text) < 0) return 0;
            return gr->complete() ? 1 : 0;
        };
        for (const json& msg : rendered_corpus()) {
            const std::string reply = b->render_reply(msg);
            const size_t at = reply.find(opener);
            if (at == std::string::npos) continue;
            const std::string calls = reply.substr(at);
            CHECK_EQ(admits(b->prompt().grammar, calls), 1);
            CHECK_EQ(admits(g.text, calls), 1);
            for (int k = 0; k < 20; ++k) {
                const std::string bad = mutate(rng, calls, frag);
                CHECK_EQ(admits(b->prompt().grammar, sanitize(bad)), admits(g.text, sanitize(bad)));
            }
        }
    }
    log_set_level(Log::Warn);
}

/* A required call constrains the whole reply: after the generation prompt, the reasoning block and
 * then the calls, with no prose before or after, so the reply ends at the last call. A named
 * function admits only that function; `parallel_tool_calls: false` admits one call. */
TEST(a_required_call_grammar_is_the_whole_reply) {
    log_set_level(Log::Error);
    static const SimpleVocab vocab = [] {
        SimpleVocab v;
        for (int c = 32; c < 127; ++c) v.add(std::string(1, (char)c));
        v.add("\n");
        return v;
    }();
    int checked = 0;
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b || b->prompt().grammar.empty()) continue;
        const std::string gp = b->prompt().generation_prompt;
        auto grammar = [&](bool parallel, const std::string& only) {
            ChatGrammarRequest gq;
            gq.tools = &agent_tools();
            gq.generation_prompt = gp;
            gq.parallel_tool_calls = parallel;
            gq.require_call = true;
            gq.only_tool = only;
            ChatGrammar g;
            std::string why;
            CHECK_OK(chat_format_grammar(b->format(), gq, &g, &why));
            CHECK(!g.lazy);
            CHECK(g.trigger_words.empty());
            return g.text;
        };
        /* 1 complete, 0 admitted but unfinished or refused. The sampler feeds the generation
         * prompt first (SamplingParams::grammar_prefix). */
        auto admits = [&](const std::string& gbnf, const std::string& reply) {
            std::unique_ptr<GbnfGrammar> gr;
            std::string err;
            if (GbnfGrammar::create(gbnf, "root", &vocab, false, {}, {}, &gr, &err) < 0) return -1;
            if (gr->accept_str(gp) < 0 || gr->accept_str(reply) < 0) return 0;
            return gr->complete() ? 1 : 0;
        };
        const std::string any = grammar(true, ""), one = grammar(false, ""),
                          read = grammar(true, "read");
        const std::string T = "Thinking it over.\n</think>\n\n";
        const std::string bash = C(m, "bash", { { "command", "ls" } });
        const std::string rd = C(m, "read", { { "path", "/a" } });

        CHECK_EQ(admits(any, T + bash), 1);
        CHECK_EQ(admits(any, T + bash + "\n" + rd), 1);
        CHECK_EQ(admits(any, T + "The answer is 42."), 0);
        CHECK_EQ(admits(any, T + "Checking.\n" + bash), 0);
        CHECK_EQ(admits(any, T + bash + "\nDone now."), 0);
        CHECK_EQ(admits(any, T), 0);
        CHECK_EQ(admits(one, T + bash), 1);
        CHECK_EQ(admits(one, T + bash + "\n" + rd), 0);
        CHECK_EQ(admits(read, T + rd), 1);
        CHECK_EQ(admits(read, T + bash), 0);

        ChatGrammarRequest gq;
        gq.tools = &agent_tools();
        gq.require_call = true;
        gq.only_tool = "nope";
        ChatGrammar g;
        std::string why;
        CHECK(chat_format_grammar(b->format(), gq, &g, &why) == RAD_E_INVAL);
        CHECK(why.find("nope") != std::string::npos);
        ++checked;
    }
    CHECK(checked > 0);
    log_set_level(Log::Warn);
}

/* ================================================================== the description */

TEST(the_derived_formats_are_the_ones_in_service) {
    for (const Model& m : models()) {
        ChatFormat f;
        std::string refused, why;
        CHECK_OK(chat_format_from_template(fixture(m.dir), m.bos, m.eos, &f, &refused, &why));
        CHECK(refused.empty());
        CHECK_EQ(f.reasoning_open, std::string("<think>\n"));
        CHECK_EQ(f.reasoning_close, std::string("\n</think>\n\n"));
        if (is_qwen(m)) {
            CHECK_EQ(f.call_open, std::string("<tool_call>\n"));
            CHECK_EQ(f.call_close, std::string("</tool_call>"));
            CHECK_EQ(f.name_prefix, std::string("<function="));
            CHECK_EQ(f.name_suffix, std::string(">\n"));
            CHECK_EQ(f.call_end, std::string("</function>\n"));
            CHECK_EQ(f.arg_prefix, std::string("<parameter="));
            CHECK_EQ(f.arg_name_suffix, std::string(">\n"));
            CHECK_EQ(f.arg_value_suffix, std::string("</parameter>\n"));
            CHECK(f.triggers == std::vector<std::string>{ "<tool_call>\n" });
            CHECK(f.reasoning_breaks == std::vector<std::string>{ "<tool_call>\n" });
        } else {
            CHECK(f.call_open.empty());
            CHECK_EQ(f.name_prefix, std::string("<function name=\""));
            CHECK_EQ(f.name_suffix, std::string("\">"));
            CHECK_EQ(f.call_end, std::string("</function>"));
            CHECK_EQ(f.arg_prefix, std::string("<param name=\""));
            CHECK(f.arg_name_suffix.empty());
            CHECK_EQ(f.arg_value_prefix, std::string("\">"));
            CHECK_EQ(f.arg_value_suffix, std::string("</param>"));
            CHECK(f.triggers.empty());
            CHECK(f.reasoning_breaks.empty());
        }
        /* The decoder gets the same markers the vendored analysis named. */
        Bench* b = bench(m, ChatRequest{});
        if (b) CHECK(chat_format_markers(f) == b->prompt().preserved_tokens);
    }
}

/* The ChatML fallback (a model with no template) has neither reasoning nor calls. */
TEST(no_template_is_a_format_with_no_reasoning_and_no_calls) {
    ChatFormat f;
    std::string refused, why;
    CHECK_OK(chat_format_from_template("", "", "", &f, &refused, &why));
    CHECK(!f.has_reasoning());
    CHECK(!f.has_tools());
    std::shared_ptr<const ReplyFormat> rf;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    ReplyOptions ro;
    std::shared_ptr<const ReplyParser> rp;
    CHECK_OK(ReplyParser::create(rf, ro, &rp, &why));
    const ReplyMessage r = reply_parse(*rp, "  <think>x</think> <tool_call>\n");
    CHECK_EQ(r.content, std::string("  <think>x</think> <tool_call>"));
    /* A request with tools is refused with the reason. */
    const json tools = agent_tools();
    ro.tools = &tools;
    CHECK(ReplyParser::create(rf, ro, &rp, &why) == RAD_E_UNSUPPORTED);
    CHECK(!why.empty());
}

/* ================================================================== the plugin's declaration */

static const char* const kBreaks[] = { "<tool_call>\n" };
static const char* const kTriggers[] = { "<tool_call>\n" };

static RadChatFormat qwen_abi() {
    RadChatFormat c;
    std::memset(&c, 0, sizeof c);
    c.struct_size = sizeof c;
    c.name = "qwen-declared";
    c.reasoning_open = "<think>\n";
    c.reasoning_close = "\n</think>\n\n";
    c.reasoning_breaks = kBreaks;
    c.n_reasoning_breaks = 1;
    c.call_open = "<tool_call>\n";
    c.call_close = "</tool_call>";
    c.name_prefix = "<function=";
    c.name_suffix = ">\n";
    c.call_end = "</function>\n";
    c.arg_prefix = "<parameter=";
    c.arg_name_suffix = ">\n";
    c.arg_value_prefix = "";
    c.arg_value_suffix = "</parameter>\n";
    c.string_value = RAD_CHAT_VALUE_CDATA;
    c.other_value = RAD_CHAT_VALUE_JSON;
    c.value_newlines = 1;
    c.triggers = kTriggers;
    c.n_triggers = 1;
    return c;
}

/* Where the reply starts relative to its reasoning block, read from the generation prompt as the
 * stream reads it: opened, opened and closed (thinking off), or not opened, with markers trimmed of
 * optional whitespace. The engine counts and closes the block by these, so a reply reported open
 * must be one the stream reads as reasoning. */
TEST(the_parser_says_whether_a_reply_starts_inside_its_reasoning_block) {
    RadChatFormat c = qwen_abi();
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(&c, &f, &why));
    std::shared_ptr<const ReplyFormat> rf;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    struct Case { const char* prompt; bool open; };
    for (const Case& k : { Case{ "<|im_start|>assistant\n<think>\n", true },
                           Case{ "<|im_start|>assistant\n<think>\n\n</think>\n\n", false },
                           Case{ "<|im_start|>assistant\n", false } }) {
        ReplyOptions ro;
        ro.generation_prompt = k.prompt;
        std::shared_ptr<const ReplyParser> rp;
        CHECK_OK(ReplyParser::create(rf, ro, &rp, &why));
        const ReasoningMarkers m = rp->reasoning();
        CHECK_EQ(m.open, std::string("<think>"));
        CHECK_EQ(m.close, std::string("</think>"));
        CHECK_EQ(m.starts_open, k.open);
        const ReplyMessage r = reply_parse(*rp, "x</think>\n\ny");
        CHECK_EQ(r.reasoning.empty(), !k.open);
    }

    ChatFormat none;
    CHECK_OK(chat_format_from_template("", "", "", &none, &why, &why));
    const std::shared_ptr<const ReplyParser> rp = parser_of(none, ReplyOptions{});
    REQUIRE(rp);
    CHECK(rp->reasoning().close.empty());
    CHECK(!rp->reasoning().starts_open);
}

/* A plugin-declared format behaves exactly as the template-derived one: same parser, same
 * grammar. */
TEST(a_declared_format_reads_and_constrains_like_the_derived_one) {
    const Model& m = models()[0];
    RadChatFormat c = qwen_abi();
    ChatFormat declared;
    std::string why;
    CHECK_OK(chat_format_from_abi(&c, &declared, &why));
    CHECK_OK(chat_format_check(declared, &why));
    Bench* b = bench(m, ChatRequest{});
    if (!b) return;
    ChatFormat derived = b->format();
    derived.name = declared.name;
    CHECK(chat_format_describe(derived) == chat_format_describe(declared));
    CHECK(chat_format_markers(derived) == chat_format_markers(declared));

    ReplyOptions ro;
    ro.generation_prompt = b->prompt().generation_prompt;
    ro.tools = &agent_tools();
    ro.terminators = m.terminators;
    const std::shared_ptr<const ReplyParser> rp = parser_of(declared, ro);
    REQUIRE(rp);
    for (const json& msg : rendered_corpus()) {
        const std::string reply = b->render_reply(msg);
        CHECK(same(reply_parse(*rp, reply), b->new_parse(reply)));
    }
    ChatGrammarRequest gq;
    gq.tools = &agent_tools();
    gq.generation_prompt = b->prompt().generation_prompt;
    ChatGrammar g;
    CHECK_OK(chat_format_grammar(declared, gq, &g, &why));
    CHECK_EQ(g.text, b->prompt().grammar);
}

/* oh-my-pi's `task` tool in batch form: `"model?": "never"`, which Arktype exports as
 * `{"not": {}}`. A schema no value satisfies gets no arm, and the declared and derived formats
 * agree on it. */
TEST(an_argument_no_value_satisfies_has_no_arm_in_either_grammar) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"task",
      "description":"Spawn subagents.","parameters":{"type":"object","properties":{
       "context":{"type":"string"},"model":{"not":{}},
       "tasks":{"type":"array","items":{"type":"object","properties":{
        "agent":{"type":"string"},"task":{"type":"string"},"solutionSpace":{"type":"string"},
        "model":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]}},
        "required":["agent","solutionSpace","task"]}}},
       "required":["context","tasks"]}}}])");
    RadChatFormat c = qwen_abi();
    ChatFormat declared;
    std::string why;
    CHECK_OK(chat_format_from_abi(&c, &declared, &why));
    ChatGrammarRequest gq;
    gq.tools = &tools;
    ChatGrammar g;
    CHECK_OK(chat_format_grammar(declared, gq, &g, &why));
    CHECK(!g.text.empty());
    CHECK(g.text.find("task-arg-model") == std::string::npos);

    Bench* b = bench(models()[0], ChatRequest{}, tools);
    if (!b) return;
    gq.generation_prompt = b->prompt().generation_prompt;
    CHECK_OK(chat_format_grammar(declared, gq, &g, &why));
    CHECK_EQ(g.text, b->prompt().grammar);

    /* The parser reads such an argument via the undeclared arm, as the grammar does: as text,
     * even when it spells JSON. */
    const std::string reply = "Thinking it over.\n</think>\n\n" +
        C(models()[0], "task", { { "context", "c" }, { "tasks", "[]" }, { "model", "123" } });
    const Verdict v = run(b, reply, "an argument no value satisfies");
    CHECK(v.strict);
    const ReplyMessage r = b->new_parse(reply);
    REQUIRE(r.calls.size() == 1);
    CHECK_EQ(r.calls[0].arguments, std::string("{\"context\":\"c\",\"tasks\":[],\"model\":\"123\"}"));
}

/* oh-my-pi 18.6.0's tools as the client sends them (tests/data/chat), without descriptions. Its
 * `task` tool spells the optional `model` as `not` of a union of all six JSON types, which is also
 * `never`; the declared format gives it no arm and agrees with the derived format on the whole
 * tool set. */
TEST(oh_my_pis_tools_build_the_same_grammar_in_either_format) {
    std::ifstream f(std::string(RAD_CHAT_FIXTURES) + "/oh-my-pi-18.6.0-tools.json");
    std::ostringstream ss;
    ss << f.rdbuf();
    const json tools = json::parse(ss.str());
    REQUIRE(tools.is_array() && tools.size() == 11);

    RadChatFormat c = qwen_abi();
    ChatFormat declared;
    std::string why;
    CHECK_OK(chat_format_from_abi(&c, &declared, &why));
    ChatGrammarRequest gq;
    gq.tools = &tools;
    ChatGrammar g;
    CHECK_OK(chat_format_grammar(declared, gq, &g, &why));
    if (!why.empty()) fprintf(stderr, "    %s\n", why.c_str());
    CHECK(!g.text.empty());
    CHECK(g.text.find("task-arg-model") == std::string::npos);
    CHECK(g.text.find("task-arg-tasks") != std::string::npos);

    Bench* b = bench(models()[0], ChatRequest{}, tools);
    if (!b) return;
    gq.generation_prompt = b->prompt().generation_prompt;
    CHECK_OK(chat_format_grammar(declared, gq, &g, &why));
    CHECK_EQ(g.text, b->prompt().grammar);
}

/* The struct grows at its end. A plugin built against an older header reports a smaller
 * struct_size and the missing fields read as zero; extra fields from a newer header are ignored.
 * Nothing is read past the reported size. */
TEST(the_declaration_reads_only_what_the_plugin_compiled) {
    RadChatFormat c = qwen_abi();
    ChatFormat f;
    std::string why;

    c.struct_size = (uint32_t)offsetof(RadChatFormat, string_value);
    CHECK_OK(chat_format_from_abi(&c, &f, &why));
    CHECK_EQ(f.name_prefix, std::string("<function="));
    CHECK(f.string_value == ChatValue::Raw);
    CHECK(f.triggers.empty());

    struct Bigger { RadChatFormat base; const char* later_field; } big;
    std::memset(&big, 0, sizeof big);
    big.base = qwen_abi();
    big.base.struct_size = sizeof big;
    big.later_field = "ignored";
    CHECK_OK(chat_format_from_abi(&big.base, &f, &why));
    CHECK(f.triggers == std::vector<std::string>{ "<tool_call>\n" });

    c.struct_size = 2;
    CHECK(chat_format_from_abi(&c, &f, &why) == RAD_E_INVAL);
    CHECK(chat_format_from_abi(nullptr, &f, &why) == RAD_E_INVAL);
}

/* What the parser cannot read is refused by name, never guessed. */
TEST(a_format_the_parser_cannot_read_is_refused_by_name) {
    auto refused = [](void (*edit)(RadChatFormat&), const char* word) {
        RadChatFormat c = qwen_abi();
        edit(c);
        ChatFormat f;
        std::string why;
        if (chat_format_from_abi(&c, &f, &why) < 0) return true;
        const int st = chat_format_check(f, &why);
        if (st >= 0) return false;
        return why.find(word) != std::string::npos;
    };
    CHECK(refused([](RadChatFormat& c) { c.section_open = "<calls>"; }, "section"));
    CHECK(refused([](RadChatFormat& c) { c.call_separator = ", "; }, "separator"));
    CHECK(refused([](RadChatFormat& c) { c.name_suffix = ""; }, "name_suffix"));
    CHECK(refused([](RadChatFormat& c) { c.call_end = ""; }, "call_end"));
    CHECK(refused([](RadChatFormat& c) { c.arg_value_suffix = ""; }, "arg_value_suffix"));
    CHECK(refused([](RadChatFormat& c) { c.arg_value_prefix = "\""; }, "both set"));
    CHECK(refused([](RadChatFormat& c) { c.arg_prefix = "</function>\n"; }, "told apart"));
    CHECK(refused([](RadChatFormat& c) { c.reasoning_close = ""; }, "reasoning_close"));
    static const char* const other[] = { "<call>" };
    CHECK(refused([](RadChatFormat& c) { c.reasoning_breaks = other; }, "call opener"));
    CHECK(refused([](RadChatFormat& c) { c.string_value = RAD_CHAT_VALUE_JSON; }, "string_value"));
}

/* A tool whose name contains the name-ending delimiter is ambiguous in this format; the request
 * is refused at admission, naming the tool. */
TEST(a_tool_name_holding_the_format_delimiter_is_refused_at_admission) {
    Bench* b = bench(models()[0], ChatRequest{});
    if (!b) return;
    std::shared_ptr<const ReplyFormat> rf;
    std::string why;
    CHECK_OK(ReplyFormat::compile(b->format(), &rf, &why));
    const json bad = json::parse(R"([{"type":"function","function":{"name":"a>b"}}])");
    ReplyOptions ro;
    ro.tools = &bad;
    std::shared_ptr<const ReplyParser> rp;
    CHECK(ReplyParser::create(rf, ro, &rp, &why) == RAD_E_INVAL);
    CHECK(why.find("a>b") != std::string::npos);
    const json badp = json::parse(R"([{"type":"function","function":{"name":"f","parameters":)"
                                  R"({"type":"object","properties":{"x>\ny":{"type":"string"}}}}}])");
    ro.tools = &badp;
    CHECK(ReplyParser::create(rf, ro, &rp, &why) == RAD_E_INVAL);
    CHECK(why.find("x>") != std::string::npos);
}

/* ================================================================== the wire
 *
 * What a streaming client assembles from the SSE chunks (content, reasoning, each call's id, name
 * and arguments) equals the buffered answer, for every reply cut a byte at a time. Delta chunks
 * are written directly rather than through a JSON tree, and must match nlohmann's bytes. */
TEST(what_a_streaming_client_assembles_is_the_buffered_answer) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        ReplyOptions ro;
        ro.generation_prompt = b->prompt().generation_prompt;
        ro.tools = &agent_tools();
        ro.terminators = m.terminators;
        const std::shared_ptr<const ReplyParser> rp = parser_of(b->format(), ro);
        server::ReplyParserBridge bridge(rp);
        std::vector<std::string> replies;
        for (const json& msg : rendered_corpus()) replies.push_back(b->render_reply(msg));
        for (const Adversarial& a : adversarial(m)) replies.push_back(a.text);
        for (const std::string& reply : replies) {
            server::OaiRequest r;
            r.id = "chatcmpl-test";
            r.model = "m";
            r.parse_tools = true;
            server::ChatDeltaStream st(r, 0, &bridge);
            std::string sse = st.first_chunk();
            for (size_t i = 0; i < reply.size(); ++i)
                sse += st.push(std::string_view(reply).substr(i, 1), false);
            sse += st.push(std::string_view(), true);
            sse += st.final_chunk(server::Finish::Stop, server::TimingsOut(), server::UsageOut());

            std::string content, reasoning, finish;
            std::vector<std::string> ids, names, args;
            size_t at = 0;
            while ((at = sse.find("data: ", at)) != std::string::npos) {
                const size_t end = sse.find("\n\n", at);
                const std::string payload = sse.substr(at + 6, end - at - 6);
                const json j = json::parse(payload);
                CHECK_EQ(j.dump(), payload);
                at = end;
                const json& c = j["choices"][0];
                if (!c["finish_reason"].is_null()) finish = c["finish_reason"].get<std::string>();
                const json& d = c["delta"];
                if (d.contains("content")) content += d["content"].get<std::string>();
                if (d.contains("reasoning_content")) reasoning += d["reasoning_content"].get<std::string>();
                if (!d.contains("tool_calls")) continue;
                const json& tc = d["tool_calls"][0];
                const size_t k = tc["index"].get<size_t>();
                if (tc.contains("id")) {
                    CHECK_EQ(k, ids.size());
                    ids.push_back(tc["id"].get<std::string>());
                    names.push_back(tc["function"]["name"].get<std::string>());
                    args.emplace_back();
                }
                if (k < args.size() && tc["function"].contains("arguments"))
                    args[k] += tc["function"]["arguments"].get<std::string>();
            }
            const server::ParsedMessage& p = st.parsed();
            const ReplyMessage want = reply_parse(*rp, reply);
            CHECK_EQ(content, want.content);
            CHECK_EQ(reasoning, want.reasoning);
            CHECK_EQ(p.content, want.content);
            CHECK_EQ(names.size(), want.calls.size());
            for (size_t k = 0; k < std::min(names.size(), want.calls.size()); ++k) {
                CHECK_EQ(names[k], want.calls[k].name);
                CHECK_EQ(args[k], want.calls[k].arguments);
                CHECK_EQ(p.tool_calls[k].id, ids[k]);
            }
            /* A call is reported only when it closed and its arguments parse. */
            bool complete = !want.calls.empty() && want.calls.back().closed;
            for (const ReplyCall& c : want.calls) complete = complete && server::json_complete(c.arguments);
            CHECK_EQ(finish, std::string(complete ? "tool_calls" : "stop"));
        }
    }
}

/* ================================================================== the chat bridge
 *
 * Everything the server gets for a request derives from one format: the grammar and its triggers,
 * the tokens the decoder renders, and the parser. A plugin's declaration of the same format yields
 * the same request; an unrunnable declaration stops the chat endpoint at startup; a template whose
 * calls cannot be described has its tool requests refused. */

static std::shared_ptr<Vocab> vocab_with(const std::string& tmpl) {
    VocabBuild vb;
    vb.tokens = { "a", "<|im_end|>" };
    vb.types = { RAD_TT_NORMAL, RAD_TT_CONTROL };
    vb.eos = 1;
    vb.chat_template = tmpl;
    auto v = std::make_shared<Vocab>();
    return v->load(std::move(vb)) >= 0 ? v : nullptr;
}

static server::ChatRender bridge_render(const server::ChatTemplateBridge& br, const json& tools,
                                        std::string* why = nullptr) {
    server::ChatRenderOptions opt;
    server::ChatRender out;
    std::string w;
    const json msgs = json::array({ { { "role", "user" }, { "content", "Fix the failing test in /work." } } });
    if (br.apply(msgs, tools, opt, &out, why ? why : &w) < 0) out.prompt.clear();
    return out;
}

TEST(the_chat_bridge_derives_the_whole_request_from_one_format) {
    const Model& m = models()[0];
    std::shared_ptr<Vocab> v = vocab_with(fixture(m.dir));
    REQUIRE(v != nullptr);
    Bench* b = bench(m, ChatRequest{});
    REQUIRE(b != nullptr);

    server::ChatTemplateBridge derived;
    CHECK_OK(derived.init(*v));
    CHECK(derived.tool_refusal().empty());
    const server::ChatRender r = bridge_render(derived, agent_tools());
    REQUIRE(!r.prompt.empty());
    CHECK_EQ(r.grammar, b->prompt().grammar);
    CHECK(r.grammar_lazy);
    REQUIRE(r.grammar_triggers.size() == 1);
    CHECK_EQ(r.grammar_triggers[0].type, 1);
    CHECK_EQ(r.grammar_triggers[0].value, std::string("<tool_call>\n"));
    CHECK(r.preserved_tokens == b->prompt().preserved_tokens);
    REQUIRE(r.parser != nullptr);

    std::unique_ptr<server::IReplyReader> rd = r.parser->open();
    const std::string reply = b->render_reply(message("Look.", "", { call("read", { { "path", "/a" }, { "limit", 3 } }) }));
    rd->feed(reply, nullptr);
    rd->finish(nullptr);
    const server::ParsedMessage pm = rd->message();
    CHECK_EQ(pm.reasoning, std::string("Look.\n"));
    REQUIRE(pm.tool_calls.size() == 1);
    CHECK_EQ(pm.tool_calls[0].arguments, std::string("{\"path\":\"/a\",\"limit\":3}"));

    /* The same format declared by a plugin yields the same request. */
    RadChatFormat c = qwen_abi();
    server::ChatTemplateBridge declared;
    CHECK_OK(declared.init(*v, &c, "radtest"));
    const server::ChatRender d = bridge_render(declared, agent_tools());
    CHECK_EQ(d.grammar, r.grammar);
    CHECK(d.grammar_triggers.size() == 1 && d.grammar_triggers[0].value == "<tool_call>\n");
    CHECK(d.preserved_tokens == r.preserved_tokens);

    /* An unrunnable declaration is refused at startup, naming the field. */
    c.call_separator = ", ";
    server::ChatTemplateBridge broken;
    CHECK(broken.init(*v, &c, "radtest") < 0);

    /* An ambiguous tool is refused at admission, by name. */
    std::string why;
    const server::ChatRender bad = bridge_render(
        derived, json::parse(R"([{"type":"function","function":{"name":"a>b"}}])"), &why);
    CHECK(bad.prompt.empty());
    CHECK(why.find("a>b") != std::string::npos);
}

/* A template writing each call as a JSON object is not a describable format: the model still
 * chats with reasoning split, and a request with tools is refused with the reason. */
TEST(a_template_whose_calls_cannot_be_described_refuses_tools_and_still_chats) {
    const std::string tmpl =
        "{%- if tools -%}<|im_start|>system\n# Tools\n{%- for t in tools %}\n{{ t | tojson }}"
        "{%- endfor %}\nCall a tool as <tool_call>\n{\"name\": ..., \"arguments\": ...}\n"
        "</tool_call><|im_end|>\n{%- endif -%}"
        "{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
        "{%- if m.role == 'assistant' and m.reasoning_content -%}<think>\n{{ m.reasoning_content }}"
        "\n</think>\n\n{%- endif -%}{{ m.content or '' }}"
        "{%- if m.tool_calls -%}{%- for tc in m.tool_calls -%}\n<tool_call>\n"
        "{\"name\": \"{{ tc.function.name }}\", \"arguments\": {{ tc.function.arguments | tojson }}}"
        "\n</tool_call>{%- endfor -%}{%- endif -%}<|im_end|>\n{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>\n{%- endif -%}";
    std::shared_ptr<Vocab> v = vocab_with(tmpl);
    REQUIRE(v != nullptr);
    server::ChatTemplateBridge br;
    CHECK_OK(br.init(*v));
    if (!br.supports_tools()) return;   /* no tool support probed, nothing to refuse */
    CHECK(!br.tool_refusal().empty());
    const server::ChatRender r = bridge_render(br, json::array());
    REQUIRE(r.parser != nullptr);
    std::unique_ptr<server::IReplyReader> rd = r.parser->open();
    rd->feed("Hello.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>", nullptr);
    rd->finish(nullptr);
    CHECK(rd->message().tool_calls.empty());
    CHECK(rd->message().content.find("Hello.") == 0);
    /* A render with tools is refused too, should one pass admission. */
    std::string why;
    const server::ChatRender t = bridge_render(br, agent_tools(), &why);
    CHECK(t.prompt.empty());
    CHECK(!why.empty());
}

/* ================================================================== the object family
 *
 * A call written as its name and one object of arguments (ChatArgs::Object): JSON after a name tag
 * with no end or wrapper (Mistral's `[TOOL_CALLS]NAME[ARGS]{...}`), or after a function tag with
 * its own close. The dialect fields (Gemma 4's strings and bare keys) are tested in
 * chatparse_gemma4_test. */

static std::shared_ptr<const ReplyParser> object_format(bool tagged_end, ChatFormat* out = nullptr) {
    static const char* const kT1[] = { "[TOOL_CALLS]" };
    static const char* const kT2[] = { "<tool_call>" };
    RadChatFormat c{};
    c.struct_size = sizeof(RadChatFormat);
    c.args_form = RAD_CHAT_ARGS_OBJECT;
    if (!tagged_end) {
        c.name = "test:name-tag";
        c.name_prefix = "[TOOL_CALLS]";
        c.name_suffix = "[ARGS]";
        c.triggers = kT1;
    } else {
        c.name = "test:function-tag";
        c.call_open = "<tool_call>\n";
        c.call_close = "</tool_call>";
        c.name_prefix = "<function=";
        c.name_suffix = ">";
        c.call_end = "</function>\n";
        c.triggers = kT2;
    }
    c.n_triggers = 1;
    ChatFormat f;
    std::string why;
    std::shared_ptr<const ReplyFormat> rf;
    std::shared_ptr<const ReplyParser> p;
    if (chat_format_from_abi(&c, &f, &why) < 0 || ReplyFormat::compile(f, &rf, &why) < 0) {
        ::radtest::fail(__FILE__, __LINE__, why);
        return nullptr;
    }
    if (out) *out = f;
    ReplyOptions ro;
    ro.tools = &agent_tools();
    if (ReplyParser::create(rf, ro, &p, &why) < 0) ::radtest::fail(__FILE__, __LINE__, why);
    return p;
}

static int object_admits(const std::string& gbnf, const std::string& text) {
    static const SimpleVocab vocab = [] {
        SimpleVocab v;
        for (int c = 32; c < 127; ++c) v.add(std::string(1, (char)c));
        v.add("\n");
        return v;
    }();
    std::unique_ptr<GbnfGrammar> gr;
    std::string err;
    if (GbnfGrammar::create(gbnf, "root", &vocab, false, {}, {}, &gr, &err) < 0) return -1;
    if (gr->accept_str(text) < 0) return 0;
    return gr->complete() ? 1 : 0;
}

TEST(json_arguments_behind_a_name_tag_read_as_one_object) {
    std::shared_ptr<const ReplyParser> p = object_format(false);
    REQUIRE(p != nullptr);
    bool strict = false;
    const std::string reply = "Sure.[TOOL_CALLS]bash[ARGS]{\"command\": \"ls -la\", \"timeout\": 30}"
                              "[TOOL_CALLS]read[ARGS]{\"path\": \"/a\", \"limit\": [1, {\"x\": null}]}";
    const ReplyMessage m = reply_parse(*p, reply, &strict);
    CHECK(strict);
    CHECK_EQ(m.content, std::string("Sure."));
    REQUIRE(m.calls.size() == 2);
    CHECK_EQ(m.calls[0].arguments, std::string("{\"command\":\"ls -la\",\"timeout\":30}"));
    CHECK_EQ(m.calls[1].arguments, std::string("{\"path\":\"/a\",\"limit\":[1,{\"x\":null}]}"));
    const std::string err = stream_all(*p, reply);
    if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err);
    /* A bare key is outside JSON: read, and reported non-strict. */
    const ReplyMessage b = reply_parse(*p, "[TOOL_CALLS]read[ARGS]{path: \"/a\"}", &strict);
    CHECK(!strict);
    REQUIRE(b.calls.size() == 1);
    CHECK_EQ(b.calls[0].arguments, std::string("{\"path\":\"/a\"}"));
    /* No object after the tag: a call without arguments, and the rest is content. */
    const ReplyMessage n = reply_parse(*p, "[TOOL_CALLS]ping[ARGS] and then", &strict);
    CHECK(!strict);
    REQUIRE(n.calls.size() == 1);
    CHECK_EQ(n.calls[0].arguments, std::string("{}"));
    CHECK_EQ(n.content, std::string("and then"));
}

TEST(json_arguments_behind_a_function_tag_end_at_its_close) {
    ChatFormat f;
    std::shared_ptr<const ReplyParser> p = object_format(true, &f);
    REQUIRE(p != nullptr);
    bool strict = false;
    const std::string reply = "<tool_call>\n<function=bash>{\"command\": \"ls\"}</function>\n</tool_call>"
                              "<tool_call>\n<function=read>{\"path\": \"/a\"}\n</function>\n</tool_call>";
    const ReplyMessage m = reply_parse(*p, reply, &strict);
    CHECK(strict);
    REQUIRE(m.calls.size() == 2);
    CHECK_EQ(m.calls[0].arguments, std::string("{\"command\":\"ls\"}"));
    CHECK_EQ(m.calls[1].arguments, std::string("{\"path\":\"/a\"}"));
    const std::string err = stream_all(*p, reply);
    if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err);
    /* The function's close also closes a still-open object. */
    const ReplyMessage c = reply_parse(
        *p, "<tool_call>\n<function=read>{\"path\": \"/a\"</function>\n</tool_call>", &strict);
    CHECK(!strict);
    REQUIRE(c.calls.size() == 1);
    CHECK(c.calls[0].closed);
    CHECK_EQ(c.calls[0].arguments, std::string("{\"path\":\"/a\"}"));

    /* The grammar: the tags as written with a JSON object between. */
    ChatGrammarRequest gq;
    gq.tools = &agent_tools();
    ChatGrammar g;
    std::string why;
    CHECK_OK(chat_format_grammar(f, gq, &g, &why));
    CHECK(g.lazy);
    log_set_level(Log::Error);
    CHECK_EQ(object_admits(g.text, "<tool_call>\n<function=bash>{\"command\": \"ls\", \"n\": [1, {\"a\": null}]}"
                                   "</function>\n</tool_call>"), 1);
    CHECK_EQ(object_admits(g.text, "<tool_call>\n<function=bash>{command: \"ls\"}</function>\n</tool_call>"), 0);
    CHECK_EQ(object_admits(g.text, "<tool_call>\n<function=nope>{}</function>\n</tool_call>"), 0);
    log_set_level(Log::Warn);
}

/* reasoning_anywhere: a block opening after the answer is still reasoning, and a close with no
 * block open is dropped. Without it, both are content. */
TEST(a_reasoning_block_may_open_again_later_when_the_format_says_so) {
    for (bool anywhere : { false, true }) {
        RadChatFormat c = qwen_abi();
        c.reasoning_anywhere = anywhere ? 1 : 0;
        ChatFormat f;
        std::string why;
        CHECK_OK(chat_format_from_abi(&c, &f, &why));
        ReplyOptions ro;
        ro.generation_prompt = "<|im_start|>assistant\n<think>\n";
        ro.tools = &agent_tools();
        const std::shared_ptr<const ReplyParser> p = parser_of(f, ro);
        REQUIRE(p);
        const std::string reply = "Plan.\n</think>\n\nAnswer.<think>\nmore\n</think>\n\nTail.</think>";
        bool strict = false;
        const ReplyMessage m = reply_parse(*p, reply, &strict);
        if (anywhere) {
            CHECK_EQ(m.reasoning, std::string("Plan.\nmore\n"));
            CHECK_EQ(m.content, std::string("Answer.Tail."));
            CHECK(!strict);   /* the last </think> closes nothing */
        } else {
            CHECK_EQ(m.reasoning, std::string("Plan.\n"));
            CHECK_EQ(m.content, std::string("Answer.<think>\nmore\n</think>\n\nTail.</think>"));
        }
        const std::string err = stream_all(*p, reply);
        if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err);
    }
}

TEST(an_object_format_the_parser_cannot_read_is_refused) {
    auto refused = [](void (*edit)(RadChatFormat&), const char* word) {
        static const char* const kT[] = { "[TOOL_CALLS]" };
        RadChatFormat c{};
        c.struct_size = sizeof(RadChatFormat);
        c.args_form = RAD_CHAT_ARGS_OBJECT;
        c.name_prefix = "[TOOL_CALLS]";
        c.name_suffix = "[ARGS]";
        c.triggers = kT;
        c.n_triggers = 1;
        edit(c);
        ChatFormat f;
        std::string why;
        if (chat_format_from_abi(&c, &f, &why) < 0) return false;
        return chat_format_check(f, &why) == RAD_E_UNSUPPORTED && why.find(word) != std::string::npos;
    };
    CHECK(refused([](RadChatFormat&) {}, "") == false);   /* the base is a format */
    CHECK(refused([](RadChatFormat& c) { c.arg_value_suffix = "</a>"; }, "arg_*"));
    CHECK(refused([](RadChatFormat& c) { c.object_string_open = "<s>"; }, "together"));
    CHECK(refused([](RadChatFormat& c) { c.object_string_open = c.object_string_close = "'"; },
                  "bare word"));
    CHECK(refused([](RadChatFormat& c) { c.call_end = "<,>"; }, "punctuation"));
    CHECK(refused([](RadChatFormat& c) { c.call_end = " \n"; }, "only whitespace"));
    CHECK(refused([](RadChatFormat& c) { c.name_prefix = ""; c.triggers = nullptr; c.n_triggers = 0; },
                  "name_prefix is empty"));
}

/* ================================================================== cost
 *
 * The legacy path re-parses the whole reply per chunk, O(n^2). The parser's own count of what it
 * reads twice checks the claim deterministically; the `perf_` cases time it, with budgets that
 * tolerate a loaded machine but no quadratic path (re-parsing a 200 KB reply per byte would read
 * 20 GB). The timed cases belong in a serial run, apart from the unit tests. */

/* Each step of the parser consumes a byte or queues bytes to be read again
 * (ReplyStream::reread), so what it reads twice bounds its work per byte. Plain prose and a
 * well-formed call are read once. A near miss of the call syntax is held and read again, a fixed
 * amount each time, whatever the chunking: a path that re-read what it holds on every chunk, or
 * everything since its last decision, would grow with the chunk count or the reply's square. */
TEST(the_parser_reads_a_byte_twice_only_where_a_near_miss_is_read_again) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        auto reread = [&](const std::string& text, size_t piece) {
            std::unique_ptr<ReplyStream> s = b->parser().open();
            for (size_t at = 0; at < text.size(); at += piece)
                s->feed(std::string_view(text).substr(at, piece), nullptr);
            s->finish(nullptr);
            return s->reread();
        };
        const std::string plain = long_reply(m, 200 * 1024);
        for (const size_t piece : { plain.size(), (size_t)4, (size_t)1 })
            CHECK_EQ(reread(plain, piece), (size_t)0);

        /* Openers that turn out to be text, then an argument that turns out not to be JSON. */
        const std::string unit =
            (is_qwen(m) ? std::string("See <function=read and <function= more <tool_call> text, "
                                      "<function=x>\nnot a call. ")
                        : std::string("Use <function name=\"x\" oops and <function name= and "
                                      "<param name=\"a\">v</param> here. ")) +
            C(m, "configure", { { "verbose", "yes please" }, { "extra", "{'a': [1, 2}" } }) + "\n";
        size_t at64 = 0;
        for (const size_t copies : { (size_t)64, (size_t)256 }) {
            std::string text = "Thinking it over.\n</think>\n\n";
            for (size_t k = 0; k < copies; ++k) text += unit;
            const size_t whole = reread(text, text.size());
            CHECK_EQ(reread(text, 4), whole);
            CHECK_EQ(reread(text, 1), whole);
            CHECK(whole >= copies && whole < text.size());
            if (copies == 64) at64 = whole;
            else CHECK(whole <= 4 * at64 + 64);
        }
    }
}

static double seconds_to(const std::function<void()>& f, int reps = 3) {
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

static double stream_bytes(const ReplyParser& p, const std::string& text, size_t piece,
                           std::vector<double>* per_call = nullptr, int reps = 3) {
    return seconds_to([&] {
        std::unique_ptr<ReplyStream> s = p.open();
        std::vector<ReplyEvent> ev;
        for (size_t at = 0; at < text.size(); at += piece) {
            const auto t0 = std::chrono::steady_clock::now();
            s->feed(std::string_view(text).substr(at, std::min(piece, text.size() - at)), &ev);
            if (per_call)
                per_call->push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            ev.clear();
        }
        s->finish(&ev);
    }, per_call ? 1 : reps);
}

TEST(perf_a_long_reply_streamed_byte_by_byte_costs_a_constant_factor_of_parsing_it_once) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string text = long_reply(m, 200 * 1024);
        const ReplyMessage once = b->new_parse(text);
        CHECK_EQ(once.calls.size(), (size_t)1);
        const double t_once = seconds_to([&] { (void)b->new_parse(text); });
        const double t_bytes = stream_bytes(b->parser(), text, 1);
        const double t_tokens = stream_bytes(b->parser(), text, 4);
        fprintf(stderr, "    %s: %zu bytes -- once %.3f ms (%.0f MB/s), byte by byte %.3f ms "
                        "(%.1f ns/byte), 4-byte pieces %.3f ms\n",
                m.name.c_str(), text.size(), t_once * 1e3, text.size() / t_once / 1e6,
                t_bytes * 1e3, t_bytes / text.size() * 1e9, t_tokens * 1e3);
        /* A constant factor: per-call cost of feeding one byte versus a memchr skip. */
        CHECK(t_bytes < 200.0 * std::max(t_once, 50e-6));
        /* An absolute per-byte ceiling, far below one re-parse per byte. */
        CHECK(t_bytes / text.size() < 1e-6);
    }
}

/* Four times the reply costs four times the time, and a quadratic path sixteen; the bound of 8 is
 * a factor of two from each. The sizes are timed alternately, best of five, so bursts of machine
 * load affect both equally. */
TEST(perf_four_times_the_reply_costs_four_times_the_time) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string a = long_reply(m, 100 * 1024), c = long_reply(m, 400 * 1024);
        double ta = 1e30, tc = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            ta = std::min(ta, stream_bytes(b->parser(), a, 1, nullptr, 1));
            tc = std::min(tc, stream_bytes(b->parser(), c, 1, nullptr, 1));
        }
        const double ratio = tc / ta;
        fprintf(stderr, "    %s: %zu -> %zu bytes byte by byte: %.3f -> %.3f ms (x%.2f)\n",
                m.name.c_str(), a.size(), c.size(), ta * 1e3, tc * 1e3, ratio);
        CHECK(ratio < 8.0);
    }
}

TEST(perf_a_chunk_costs_microseconds_at_the_99th_percentile) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string text = long_reply(m, 200 * 1024);
        std::vector<double> per;
        stream_bytes(b->parser(), text, 4, &per);
        std::sort(per.begin(), per.end());
        const double p50 = per[per.size() / 2], p99 = per[per.size() * 99 / 100];
        fprintf(stderr, "    %s: %zu chunks of 4 bytes, p50 %.2f us, p99 %.2f us, max %.1f us\n",
                m.name.c_str(), per.size(), p50 * 1e6, p99 * 1e6, per.back() * 1e6);
        CHECK(p99 < 50e-6);
    }
}

/* Held bytes stay constant over a long plain-text reply: no buffer grows with it. */
TEST(a_long_reply_holds_a_constant_number_of_bytes) {
    for (const Model& m : models()) {
        Bench* b = bench(m, ChatRequest{});
        if (!b) continue;
        const std::string text = long_reply(m, 200 * 1024);
        std::unique_ptr<ReplyStream> s = b->parser().open();
        size_t most = 0;
        for (size_t at = 0; at < text.size(); ++at) {
            s->feed(std::string_view(text).substr(at, 1), nullptr);
            const ReplyHeld h = s->held();
            most = std::max(most, h.delimiter + h.space + h.structure);
        }
        CHECK(most <= b->parser().hold_bound() + 64);
    }
}

RAD_TEST_MAIN()
