/* chat_test.cpp -- the chat template and the reply parser, against the real template.
 *
 * server_test covers the endpoint with fake IChatTemplate and IReplyParser implementations. This
 * file covers what decides whether an agent turn works: what the template renders, and whether
 * the reply splits into reasoning, content and tool calls.
 *
 * Templates are read from disk: tests/data/chat holds each byte for byte as the model ships it,
 * and RAD_TEST_CHAT_JINJA / RAD_TEST_MINICPM_JINJA point the cases at a checkpoint's own copy to
 * check a revised template before replacing the fixture. Replies are read as the server reads
 * them: the format derived from the template (chatfmt.h), bound to the request's tools, streamed
 * through chatparse.h.
 *
 * Every case is a shape the model emitted in a served agent turn. Qwen's tool calls are XML
 * (<tool_call><function=..><parameter=..>); the template opens the reasoning block and the model
 * closes it. The cases target where these interact: a call inside an unclosed think block, a
 * second </think>, a call with no reasoning. A parser handling only the happy path drops tool
 * calls in production.
 */
#include "rad_test.h"

#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

using namespace rad;
using json = nlohmann::ordered_json;

/* ------------------------------------------------------------------ the template under test */

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* A template under test: the committed fixture `rel` unless `env` names a checkpoint's copy.
 * It is compiled once and kept for the process: loading parses the Jinja and probes its
 * capabilities, too slow to repeat per case. A template that does not load fails every case that
 * needs it. One that cannot be read is a skip when `env` named it (a model file) and a missing
 * in-tree artifact when it is the committed fixture. */
struct TemplateFixture {
    const char* env;
    bool overridden = false;
    std::string path, source;
    ChatTemplate* t = nullptr;

    TemplateFixture(const char* env_, const char* rel, const char* bos, const char* eos)
        : env(env_) {
        const char* p = std::getenv(env);
        overridden = p != nullptr;
        path = p ? p : std::string(RAD_CHAT_FIXTURES) + "/" + rel;
        source = read_file(path);
        if (source.empty()) return;
        t = new ChatTemplate();
        if (t->load(source, bos, eos) < 0) {
            delete t;
            t = nullptr;
        }
    }

    /* False, with the case marked skipped or failed, when the template is not ready; the case
     * then leaves. */
    bool need() const {
        if (t && t->ready()) return true;
        if (overridden && source.empty())
            ::radtest::skip("no chat template at %s (named by %s)", path.c_str(), env);
        else if (source.empty())
            ::radtest::artifact_missing(__FILE__, __LINE__, "the chat template at " + path);
        else
            ::radtest::fail(__FILE__, __LINE__, "the chat template at " + path + " does not load");
        return false;
    }
};

static const TemplateFixture& qwen() {
    static const TemplateFixture f("RAD_TEST_CHAT_JINJA", "qwen3.8/chat_template.jinja", "", "");
    return f;
}

static json tools_json() {
    return json::parse(R"([{
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"limit":{"type":"integer"}},
                       "required":["path"]}}},
     {"type": "function",
      "function": {
        "name": "edit",
        "description": "Replace an exact string in a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"old_string":{"type":"string"},
                                     "new_string":{"type":"string"}},
                       "required":["path","old_string","new_string"]}}}])");
}

/* Render one turn, passing `effort` as the template's reasoning_effort when set. */
static int render_turn(ChatPrompt* p, bool with_tools, const char* effort) {
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Read /work/a.py and summarise it."}});
    ChatOptions opt;
    opt.add_generation_prompt = true;
    opt.enable_thinking = true;
    if (effort) opt.template_kwargs["reasoning_effort"] = json(effort).dump();
    return qwen().t->apply(msgs, with_tools ? tools_json() : json(), opt, p);
}

static bool render(ChatPrompt* p, bool with_tools = true) {
    return render_turn(p, with_tools, nullptr) >= 0;
}

/* The reply parser for a render, built as the server builds it: the format derived from the
 * template, bound to the request's tools and to the render's generation prompt, which tells it
 * the reply starts inside a reasoning block. */
static std::shared_ptr<const ReplyParser> parser_for(const std::string& jinja, const char* bos,
                                                     const char* eos, const ChatPrompt& p,
                                                     const json* tools, bool parallel) {
    ChatFormat f;
    std::string refused, why;
    if (chat_format_from_template(jinja, bos, eos, &f, &refused, &why) < 0) {
        fprintf(stderr, "    no reply format: %s\n", why.c_str());
        return nullptr;
    }
    std::shared_ptr<const ReplyFormat> rf;
    if (ReplyFormat::compile(f, &rf, &why) < 0) {
        fprintf(stderr, "    reply format does not compile: %s\n", why.c_str());
        return nullptr;
    }
    ReplyOptions ro;
    ro.generation_prompt   = p.generation_prompt;
    ro.tools               = tools;
    ro.parallel_tool_calls = parallel;
    ro.terminators         = { "<|im_end|>", "<|endoftext|>" };
    std::shared_ptr<const ReplyParser> rp;
    if (ReplyParser::create(rf, ro, &rp, &why) < 0) {
        fprintf(stderr, "    no reply parser: %s\n", why.c_str());
        return nullptr;
    }
    return rp;
}

/* Read a reply whole, or (`partial`) as far as a stream has delivered it. */
static ReplyMessage read_reply(const ReplyParser& rp, const std::string& text, bool partial) {
    std::unique_ptr<ReplyStream> s = rp.open();
    s->feed(text, nullptr);
    if (!partial) s->finish(nullptr);
    return s->message();
}

/* ------------------------------------------------------------------ what the template renders */

TEST(template_opens_the_think_block_itself) {
    if (!qwen().need()) return;
    ChatPrompt p;
    CHECK(render(&p));
    /* The template, not the model, writes the opening tag at the end of the prompt, so the reply
     * starts inside a reasoning block. A parser unaware of this treats the turn as content and
     * leaks the model's </think> to the user. */
    CHECK(p.prompt.find("<|im_start|>assistant") != std::string::npos);
    CHECK(p.prompt.size() >= 7 &&
          p.prompt.compare(p.prompt.size() - 8, 8, "<think>\n") == 0);
    CHECK(p.generation_prompt.find("<think>") != std::string::npos);
}

TEST(template_renders_the_xml_tool_preamble) {
    if (!qwen().need()) return;
    ChatPrompt p;
    CHECK(render(&p));
    CHECK(p.prompt.find("<tool_call>") != std::string::npos);
    CHECK(p.prompt.find("<function=example_function_name>") != std::string::npos);
    CHECK(qwen().t->caps().tools);
    CHECK(qwen().t->caps().tool_calls);
}

/* A tool call the client sends back has `arguments` as a JSON string, per the OpenAI schema. The
 * stock template applies `|items` to it, a type error on a string. A multi-turn tool loop must
 * still render the parameter XML the model was trained on. */
TEST(replayed_tool_call_arguments_render_as_parameter_xml) {
    if (!qwen().need()) return;
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "Read /work/a.py."}});
    json call = {{"id", "c1"}, {"type", "function"},
                 {"function", {{"name", "read"},
                               {"arguments", "{\"path\": \"/work/a.py\", \"limit\": 40}"}}}};
    msgs.push_back({{"role", "assistant"}, {"content", ""}, {"tool_calls", json::array({call})}});
    msgs.push_back({{"role", "tool"}, {"tool_call_id", "c1"}, {"content", "print(1)\n"}});
    msgs.push_back({{"role", "user"}, {"content", "Now with limit 80."}});

    ChatPrompt p;
    ChatOptions opt;
    CHECK_OK(qwen().t->apply(msgs, tools_json(), opt, &p));
    CHECK(p.prompt.find("<function=read>") != std::string::npos);
    CHECK(p.prompt.find("<parameter=path>") != std::string::npos);
    CHECK(p.prompt.find("/work/a.py") != std::string::npos);
    /* The raw JSON string must not survive. The needle is the exact string sent, spaces
     * included, because the tool schema in the preamble legitimately contains `"path":`. */
    CHECK(p.prompt.find("{\"path\": \"/work/a.py\"") == std::string::npos);
}

/* ------------------------------------------------------------------ golden renders
 *
 * Every shipped template renders the requests in tests/data/chat/golden/cases.json byte for byte
 * as Python's jinja2 does under transformers' settings (tests/mkchatgolden.py wrote the goldens).
 * The prompt is the one input that fails silently: a vendored-engine refresh that changes
 * whitespace control, `tojson` spacing, a filter, or how history reasoning and tool calls reach
 * the template still yields answers, only worse ones. The requests arrive as the server passes
 * them (arguments as JSON strings, media as marker parts), so the conversion to what the template
 * reads is checked too. */

/* The first byte where two renders differ, with a little context, for a failure message. */
static std::string first_difference(const std::string& want, const std::string& got) {
    size_t i = 0;
    while (i < want.size() && i < got.size() && want[i] == got[i]) ++i;
    const size_t from = i > 40 ? i - 40 : 0;
    return "at byte " + std::to_string(i) + "\n      reference: " +
           json(want.substr(from, 100)).dump(-1, ' ', false, json::error_handler_t::replace) +
           "\n      engine:    " +
           json(got.substr(from, 100)).dump(-1, ' ', false, json::error_handler_t::replace);
}

TEST(every_shipped_template_renders_as_reference_jinja_does) {
    const std::string root = RAD_CHAT_FIXTURES;
    const json spec = json::parse(read_file(root + "/golden/cases.json"), nullptr, false);
    REQUIRE(spec.is_object());
    size_t compared = 0;
    for (const auto& [model, info] : spec["models"].items()) {
        ChatTemplate t;
        CHECK_OK(t.load(read_file(root + "/" + model + "/chat_template.jinja"),
                        info["bos"].get<std::string>(), info["eos"].get<std::string>()));
        if (!t.ready()) continue;
        const std::string media = info["media"].get<std::string>();
        for (const json& c : spec["cases"]) {
            const std::string name = c["name"].get<std::string>();
            if (c.contains("models")) {
                bool listed = false;
                for (const json& m : c["models"]) listed |= m.get<std::string>() == model;
                if (!listed) continue;
            }
            const std::string base = root + "/" + model + "/golden/" + name;
            const std::string ref = read_file(base + ".txt"), refusal = read_file(base + ".err");
            if (ref.empty() && refusal.empty()) {
                ::radtest::fail(__FILE__, __LINE__, "no golden for " + model + "/" + name +
                                " (run tests/mkchatgolden.py)");
                continue;
            }
            ChatOptions opt;
            opt.add_generation_prompt = c.value("add_generation_prompt", true);
            opt.enable_thinking = c.value("enable_thinking", true);
            opt.quiet = true;
            if (c.contains("kwargs"))
                for (const auto& [k, v] : c["kwargs"].items()) opt.template_kwargs[k] = v.dump();
            ChatPrompt p;
            const int st = t.apply(c["messages"], c.value("tools", false) ? spec["tools"] : json(),
                                   opt, &p);
            ++compared;
            if (!refusal.empty()) {
                /* The template refused in Python, so it must refuse here, in its own words. */
                if (st >= 0 || p.error.find(refusal) == std::string::npos)
                    ::radtest::fail(__FILE__, __LINE__, model + "/" + name + ": the reference "
                                    "refused (\"" + refusal + "\"); the engine returned " +
                                    std::to_string(st) + " \"" + p.error + "\"");
                continue;
            }
            if (st < 0) {
                ::radtest::fail(__FILE__, __LINE__, model + "/" + name + ": the engine refused: " +
                                p.error);
                continue;
            }
            /* The engine writes the marker where the template writes its own placeholder; the
             * processor replaces the marker with the placeholder's tokens. */
            std::string got = p.prompt;
            for (size_t at; !media.empty() && (at = got.find("<__media__>")) != std::string::npos;)
                got.replace(at, std::strlen("<__media__>"), media);
            if (got != ref)
                ::radtest::fail(__FILE__, __LINE__, model + "/" + name + " differs from the "
                                "reference render " + first_difference(ref, got));
        }
    }
    /* Five templates, every case each one renders. */
    CHECK(compared >= 73);
}

/* ------------------------------------------------------------------ reading the reply back */

/* The parser for the agent turn above (tools, one call per reply). Built once: the render and the
 * derived format are functions of the template alone, so every reply reads with the same one. */
static const ReplyParser* qwen_parser() {
    static const std::shared_ptr<const ReplyParser> rp = []() -> std::shared_ptr<const ReplyParser> {
        ChatPrompt p;
        if (!qwen().t || !render(&p)) return nullptr;
        const json tools = tools_json();
        return parser_for(qwen().source, "", "", p, &tools, false);
    }();
    return rp.get();
}

static ReplyMessage parse_reply(const std::string& text, bool partial = false) {
    const ReplyParser* rp = qwen_parser();
    if (!rp) {
        ::radtest::fail(__FILE__, __LINE__, "no reply parser for the Qwen template");
        return ReplyMessage{};
    }
    return read_reply(*rp, text, partial);
}

TEST(reply_splits_reasoning_from_content) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply("The user wants a summary.\n</think>\n\nIt prints one.");
    CHECK(r.reasoning.find("The user wants a summary.") != std::string::npos);
    CHECK_EQ(r.content, std::string("It prints one."));
    CHECK_EQ(r.calls.size(), (size_t)0);
    /* The closing tag is a delimiter, never text. */
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.reasoning.find("</think>") == std::string::npos);
}

TEST(reply_with_tool_call_after_reasoning) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply(
        "I should read it.\n</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK_EQ(r.calls[0].name, std::string("read"));
        CHECK(r.calls[0].arguments.find("/work/a.py") != std::string::npos);
    }
    CHECK(r.content.find("<tool_call>") == std::string::npos);
    CHECK(r.content.find("</think>") == std::string::npos);
}

/* An agent turn: long reasoning, the closing tag, then a tool call. It differs from the case
 * above only in length, so a different result means the parser is length- or
 * newline-sensitive. */
TEST(long_reasoning_then_tool_call_still_parses) {
    if (!qwen().need()) return;
    std::string reasoning;
    for (int i = 0; i < 60; ++i)
        reasoning += "Checking the file again, because the previous edit may not have applied. ";
    ReplyMessage r = parse_reply(
        reasoning + "\nNext, I'll fix the test with the exact old text. Let me check the rest:\n"
        "</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/test_a.py\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.content.find("<tool_call>") == std::string::npos);
    CHECK(r.content.find("<function=") == std::string::npos);
}

/* The model skips the closing tag and calls from inside the think block (patched templates
 * self-heal this shape, so models emit it). The call must still be extracted; the tool XML is
 * unambiguous either way. */
TEST(tool_call_inside_an_unclosed_think_block) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply(
        "I need to look at the file first.\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("<tool_call>") == std::string::npos);
}

/* A reply with no reasoning at all: the block the template opened is closed immediately. */
TEST(reply_with_empty_reasoning) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply("\n</think>\n\nIt prints one.");
    CHECK_EQ(r.content, std::string("It prints one."));
    CHECK(r.content.find("</think>") == std::string::npos);
}

/* Streaming parses every prefix. A partial tool call is not an error, and no prefix may leak a
 * delimiter into content. */
TEST(partial_reply_does_not_leak_delimiters) {
    if (!qwen().need()) return;
    const std::string full =
        "Reading it.\n</think>\n\n"
        "<tool_call>\n<function=read>\n<parameter=path>\n/work/a.py\n</parameter>\n"
        "</function>\n</tool_call>";
    for (size_t n = 1; n <= full.size(); ++n) {
        ReplyMessage r = parse_reply(full.substr(0, n), /*partial=*/true);
        if (r.content.find("</think>") != std::string::npos ||
            r.content.find("<tool_call>") != std::string::npos ||
            r.content.find("<function=") != std::string::npos) {
            ::radtest::fail(__FILE__, __LINE__,
                            "partial prefix of " + std::to_string(n) + " bytes leaked a "
                            "delimiter into content: \"" + r.content + "\"");
            break;
        }
    }
}

/* ------------------------------------------------------------- argument values are verbatim
 *
 * An agent's edit tool matches `old_string` against the file byte for byte. The XML carries the
 * value between two delimiting newlines:
 *
 *     <parameter=old_string>\n    return a + b\n</parameter>
 *
 * Only those two newlines are delimiters; everything between, including leading indentation, is
 * the value. Stripping any other leading whitespace breaks every indented edit. */
TEST(indented_argument_keeps_every_leading_space) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply(
        "Fixing it.\n</think>\n\n"
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=old_string>\n    return a + b\n</parameter>\n"
        "<parameter=new_string>\n    return a + b + 1\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() != 1) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("path", std::string()), std::string("/work/a.py"));
    CHECK_EQ(a.value("old_string", std::string()), std::string("    return a + b"));
    CHECK_EQ(a.value("new_string", std::string()), std::string("    return a + b + 1"));
}

/* Multi-line with blank lines and a tab: inner blank lines survive, and only the first and last
 * newline belong to the XML. */
TEST(multiline_argument_survives_verbatim) {
    if (!qwen().need()) return;
    const std::string body = "def add(a, b):\n\n\t# sum\n    return a + b\n  ";
    ReplyMessage r = parse_reply(
        "Fixing it.\n</think>\n\n"
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\n/work/a.py\n</parameter>\n"
        "<parameter=old_string>\n" + body + "\n</parameter>\n"
        "<parameter=new_string>\nx\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() != 1) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("old_string", std::string()), body);
}

/* ------------------------------------------------------------- reasoning effort
 *
 * OpenAI's `reasoning_effort` ladder is none/minimal/low/medium/high; this model's template accepts
 * xhigh/medium/low (default xhigh) and raises on anything else. The template, not the server,
 * decides: policing OpenAI's list would refuse "xhigh", and "high" is refused by the template.
 */
TEST(the_templates_own_effort_values_render) {
    if (!qwen().need()) return;
    for (const char* e : {"xhigh", "medium", "low"}) {
        ChatPrompt p;
        CHECK_OK(render_turn(&p, true, e));
        CHECK(p.error.empty());
        /* medium injects no instruction at all. */
        if (std::string(e) == "medium")
            CHECK(p.prompt.find("Reasoning effort is set to") == std::string::npos);
        else
            CHECK(p.prompt.find(std::string("Reasoning effort is set to ") + e) !=
                  std::string::npos);
    }
    /* The default, for clients that send none. */
    ChatPrompt d;
    CHECK_OK(render_turn(&d, true, nullptr));
    CHECK(d.prompt.find("Reasoning effort is set to xhigh") != std::string::npos);
}

TEST(a_refused_effort_comes_back_in_the_templates_own_words) {
    if (!qwen().need()) return;
    ChatPrompt p;
    CHECK(render_turn(&p, true, "high") < 0);
    /* The template's own message reaches the caller, naming the field and the accepted values. */
    CHECK(!p.error.empty());
    CHECK(p.error.find("reasoning effort") != std::string::npos);
    CHECK(p.error.find("xhigh") != std::string::npos);
}

/* ================================================================== MiniCPM5: the tagged format
 *
 * A second template on a different path. Qwen's calls are `<tool_call><function=f><parameter=p>`;
 * MiniCPM5's are `<function name="f"><param name="p">`, classified TAG_WITH_TAGGED and handled by
 * a different generator. Any mismatch on that path loses every tool call, so the cases replay what
 * the model actually emitted; a single call with one scalar argument proves nothing.
 */

static const TemplateFixture& minicpm() {
    static const TemplateFixture f("RAD_TEST_MINICPM_JINJA", "minicpm5/chat_template.jinja", "<s>",
                                   "</s>");
    return f;
}

/* `edits` is an array on purpose: non-string arguments go through the schema-constrained JSON
 * parser, as with a real agent's edit tool. */
static json mc_tools_json() {
    return json::parse(R"([{
      "type": "function",
      "function": {
        "name": "ls",
        "description": "List a directory",
        "parameters": {"type":"object","properties":{"path":{"type":"string"}},
                       "required":["path"]}}},
     {"type": "function",
      "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {"type":"object","properties":{"command":{"type":"string"}},
                       "required":["command"]}}},
     {"type": "function",
      "function": {
        "name": "write",
        "description": "Write a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},"content":{"type":"string"}},
                       "required":["path","content"]}}},
     {"type": "function",
      "function": {
        "name": "edit",
        "description": "Apply edits to a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"},
                                     "edits":{"type":"array","items":{"type":"object",
                                       "properties":{"oldText":{"type":"string"},
                                                     "newText":{"type":"string"}},
                                       "required":["oldText","newText"]}}},
                       "required":["path","edits"]}}}])");
}

/* The parser for an agent turn rendered with these tools, built once like qwen_parser(). */
static const ReplyParser* mc_parser() {
    static const std::shared_ptr<const ReplyParser> rp = []() -> std::shared_ptr<const ReplyParser> {
        if (!minicpm().t) return nullptr;
        ChatPrompt p;
        json msgs = json::array();
        msgs.push_back({{"role", "user"}, {"content", "Implement tic tac toe in python."}});
        ChatOptions opt;
        opt.add_generation_prompt = true;
        opt.enable_thinking = true;
        /* As the server renders it (ChatRenderOptions defaults to true); it decides whether the
         * generator repeats over calls. */
        opt.parallel_tool_calls = true;
        const int st = minicpm().t->apply(msgs, mc_tools_json(), opt, &p);
        if (st < 0) {
            fprintf(stderr, "    mc_parser: apply failed (%d): %s\n", st, p.error.c_str());
            return nullptr;
        }
        const json tools = mc_tools_json();
        return parser_for(minicpm().source, "<s>", "</s>", p, &tools, true);
    }();
    return rp.get();
}

static ReplyMessage mc_parse(const std::string& text, bool partial = false) {
    const ReplyParser* rp = mc_parser();
    /* Report the failure; an empty message would read as a parser that found no calls. */
    if (!rp) {
        ::radtest::fail(__FILE__, __LINE__, "no reply parser for the MiniCPM5 template");
        return ReplyMessage{};
    }
    return read_reply(*rp, text, partial);
}

/* No format marker may survive into content. */
static void mc_content_is_clean(const ReplyMessage& r) {
    CHECK(r.content.find("<function") == std::string::npos);
    CHECK(r.content.find("<param") == std::string::npos);
    CHECK(r.content.find("</think>") == std::string::npos);
    CHECK(r.content.find("CDATA") == std::string::npos);
    /* The turn terminator is an eog token: the decoder renders it only when the format needs it,
     * and the parser then trims it from content. */
    CHECK(r.content.find("<|im_end|>") == std::string::npos);
}

/* ===================================== replies that leave the grammar
 *
 * Real reply shapes the template's grammar does not match: a function that is not a tool, a call
 * written inside prose. The parser still reads them as calls, because the agent can answer "no
 * such tool" usefully, while a call returned as prose loses the turn. */
static void mc_recovered(const ReplyMessage& r, size_t calls) {
    CHECK_EQ(r.calls.size(), calls);
    mc_content_is_clean(r);
}

TEST(a_call_whose_function_is_not_a_tool_at_all_is_still_a_call) {
    if (!minicpm().need()) return;
    /* A shell command as the function name, which no schema-driven grammar matches. */
    ReplyMessage r = mc_parse("Checking.\n</think>\n\nLet me check the compiler version.\n\n"
                              "<function name=\"which gcc cc clang make\"></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("which gcc cc clang make"));
    CHECK_EQ(r.calls[0].arguments, std::string("{}"));
    CHECK_EQ(r.content, std::string("Let me check the compiler version."));
}

TEST(a_call_with_no_arguments_is_an_empty_object) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Notes.\n</think>\n\nI'll check my notes.\n\n"
                              "<function name=\"memory\"></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("memory"));
    CHECK_EQ(r.calls[0].arguments, std::string("{}"));
}

TEST(a_cdata_value_arrives_verbatim_without_its_delimiters) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Writing.\n</think>\n\nWriting it.\n\n<function name=\"write\">"
                              "<param name=\"path\">/work/a.c</param>"
                              "<param name=\"content\"><![CDATA[int main(void) { return 0; }\n"
                              "/* <not a tag> & \"quotes\" */\n]]></param></function>");
    mc_recovered(r, 1);
    if (r.calls.empty()) return;
    json a = json::parse(r.calls[0].arguments, nullptr, false);
    CHECK(!a.is_discarded());
    if (a.is_discarded()) return;
    CHECK_EQ(a.value("content", std::string()),
             std::string("int main(void) { return 0; }\n/* <not a tag> & \"quotes\" */\n"));
    CHECK_EQ(r.content, std::string("Writing it."));
}

TEST(two_calls_keep_the_text_between_them) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Both.\n</think>\n\nFirst.\n"
                              "<function name=\"bash\"><param name=\"cmd\">ls</param></function>"
                              "\nThen.\n<function name=\"bash\"><param name=\"cmd\">pwd</param>"
                              "</function>");
    mc_recovered(r, 2);
    if (r.calls.size() != 2) return;
    CHECK_EQ(r.calls[0].arguments, std::string("{\"cmd\":\"ls\"}"));
    CHECK_EQ(r.calls[1].arguments, std::string("{\"cmd\":\"pwd\"}"));
    CHECK_EQ(r.content, std::string("First.\nThen."));
}

/* Qwen's format with a sentence after the call, which Qwen's grammar does not admit. */
TEST(a_qwen_call_followed_by_a_sentence_is_still_a_call) {
    if (!qwen().need()) return;
    ReplyMessage r = parse_reply("Looking.\n</think>\n\n<tool_call>\n<function=read>\n"
                                 "<parameter=path>\n/work/a.py\n</parameter>\n</function>\n"
                                 "</tool_call>\nThat is the file.");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("read"));
    CHECK_EQ(r.calls[0].arguments, std::string("{\"path\":\"/work/a.py\"}"));
    CHECK_EQ(r.content, std::string("That is the file."));
    CHECK_EQ(r.reasoning, std::string("Looking.\n"));
}

/* A truncated call is a call in progress, never content, since the stream already announced it.
 * Its arguments are what arrived, unterminated; because they do not parse, the server reports
 * finish_reason "length" or "stop", not "tool_calls" (json_partial.h). */
TEST(a_half_written_call_is_a_call_whose_arguments_did_not_finish) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Writing.\n</think>\n\nWriting.\n<function name=\"write\">"
                              "<param name=\"content\"><![CDATA[int main(void) {");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.empty()) return;
    CHECK(!r.calls[0].closed);
    CHECK_EQ(r.calls[0].arguments, std::string("{\"content\":\"int main(void) {"));
    CHECK(json::parse(r.calls[0].arguments, nullptr, false).is_discarded());
    CHECK_EQ(r.content, std::string("Writing."));
}

TEST(a_reply_that_only_mentions_the_syntax_is_left_alone) {
    if (!minicpm().need()) return;
    const std::string text = "A hash map stores key-value pairs. The `<function>` syntax is how "
                             "this model writes a call, but this sentence is not one.";
    ReplyMessage r = mc_parse("Hm.\n</think>\n\n" + text);
    CHECK_EQ(r.calls.size(), (size_t)0);
    CHECK_EQ(r.content, text);
}

/* Streamed content cannot be retracted, so a call's opener never reaches content at any prefix,
 * including one ending inside the opener. */
TEST(a_streamed_call_never_shows_its_opening_as_content) {
    if (!minicpm().need()) return;
    const std::string full = "Writing.\n</think>\n\nWriting it now.\n<function name=\"write\">"
                             "<param name=\"path\">/work/a.c</param></function>";
    for (size_t n = 1; n <= full.size(); ++n) {
        ReplyMessage r = mc_parse(full.substr(0, n), /*partial=*/true);
        if (std::string("Writing it now.").compare(0, r.content.size(), r.content) != 0) {
            ::radtest::fail(__FILE__, __LINE__, "prefix of " + std::to_string(n) +
                            " bytes emitted content \"" + r.content + "\"");
            break;
        }
    }
    CHECK_EQ(mc_parse(full).content, std::string("Writing it now."));
}

/* An argument name the schema does not declare. Schema arms match parameter names literally, so
 * a strict grammar would reject the call, e.g.
 *
 *     <function name="read"><param name="action">read</param></function>
 *
 * and the second case would lose an otherwise valid call over one extra argument. Both must reach
 * the tool, which can reject the argument properly. */
TEST(mc_an_undeclared_argument_does_not_lose_the_call) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Reading it.\n</think>\n\n"
                             "<function name=\"ls\"><param name=\"action\">read</param></function>");
    CHECK_EQ((int)r.calls.size(), 1);
    if (r.calls.empty()) return;
    CHECK_EQ(r.calls[0].name, std::string("ls"));
    CHECK(r.calls[0].arguments.find("action") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_an_extra_argument_does_not_lose_a_valid_call) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Listing.\n</think>\n\n"
                             "<function name=\"ls\"><param name=\"path\">/work</param>"
                             "<param name=\"action\">read</param></function>");
    CHECK_EQ((int)r.calls.size(), 1);
    if (r.calls.empty()) return;
    /* The declared argument still binds to its own schema-checked arm; the generic arm is tried
     * last. */
    CHECK(r.calls[0].arguments.find("\"path\"") != std::string::npos);
    CHECK(r.calls[0].arguments.find("/work") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_one_call_with_one_argument) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "I will look at the directory.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK_EQ(r.calls[0].name, std::string("ls"));
        CHECK(r.calls[0].arguments.find("/work") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* Two calls separated by a newline; a parallel-call rule expecting a JSON-array ", " separator
 * would miss the second. */
TEST(mc_two_calls_separated_by_a_newline) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "Both, then.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>\n"
        "<function name=\"bash\"><param name=\"command\">pwd</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)2);
    if (r.calls.size() == 2) {
        CHECK_EQ(r.calls[0].name, std::string("ls"));
        CHECK_EQ(r.calls[1].name, std::string("bash"));
        CHECK(r.calls[1].arguments.find("pwd") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* A sentence before the call. This format declares no section or per-call opener, so content
 * must end at the call itself. */
TEST(mc_a_sentence_before_the_call) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "The workspace is empty.\n</think>\n\n"
        "The workspace is empty. I will create the file.\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\">print(1)</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("I will create the file.") != std::string::npos);
    mc_content_is_clean(r);
}

/* A sentence after the call, and between two calls. */
TEST(mc_a_sentence_after_the_call) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "Writing it.\n</think>\n\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\">print(1)</param></function>\n"
        "Created the file.");
    CHECK_EQ(r.calls.size(), (size_t)1);
    CHECK(r.content.find("Created the file.") != std::string::npos);
    mc_content_is_clean(r);
}

TEST(mc_a_sentence_between_two_calls) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "First one, then the other.\n</think>\n\n"
        "<function name=\"ls\"><param name=\"path\">/work</param></function>\n"
        "Next, I will check the current directory.\n"
        "<function name=\"bash\"><param name=\"command\">pwd</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)2);
    CHECK(r.content.find("Next, I will check") != std::string::npos);
    mc_content_is_clean(r);
}

/* A tagged argument carries its own name, so arguments may come in any order, not just schema
 * order. */
TEST(mc_arguments_in_any_order) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "Writing.\n</think>\n\n"
        "<function name=\"write\"><param name=\"content\">print(1)</param>"
        "<param name=\"path\">/work/t.py</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        CHECK(r.calls[0].arguments.find("/work/t.py") != std::string::npos);
        CHECK(r.calls[0].arguments.find("print(1)") != std::string::npos);
    }
    mc_content_is_clean(r);
}

/* CDATA is punctuation. The template tells the model to wrap any value holding <, & or a newline
 * in it, so it is the normal shape of a written file; a delimiter reaching the tool would be
 * written into the file. */
TEST(mc_cdata_is_not_part_of_the_value) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "Writing.\n</think>\n\n"
        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
        "<param name=\"content\"><![CDATA[def f():\n    return \"<b>\"\n]]></param>"
        "</function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        const std::string& a = r.calls[0].arguments;
        CHECK(a.find("CDATA") == std::string::npos);
        CHECK(a.find("]]>") == std::string::npos);
        CHECK(a.find("def f()") != std::string::npos);
    }
}

/* A streamed value is re-parsed every chunk and only new bytes are sent, so a partial parse must
 * never emit what it would later retract, such as a partial `<![CDATA` opener. */
TEST(mc_a_partial_cdata_value_emits_no_delimiter) {
    if (!minicpm().need()) return;
    for (const char* head : { "<", "<![CDATA", "<![CDATA[", "<![CDATA[def f():" }) {
        ReplyMessage r = mc_parse(
            std::string("Writing.\n</think>\n\n"
                        "<function name=\"write\"><param name=\"path\">/work/t.py</param>"
                        "<param name=\"content\">") + head, /*partial=*/true);
        if (r.calls.empty()) continue;   /* emitting nothing yet is also correct */
        CHECK(r.calls[0].arguments.find("CDATA") == std::string::npos);
    }
}

/* A non-string argument written as a Python literal rather than JSON. A strict JSON parser
 * rejects the single quote and loses the call, and the model then retries the same call
 * repeatedly. */
TEST(mc_a_python_quoted_array_argument_still_parses) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse(
        "Fixing the typo.\n</think>\n\n"
        "<function name=\"edit\"><param name=\"edits\">"
        "[{'oldText': 'def minimise:', 'newText': 'def minimax:'}]</param>"
        "<param name=\"path\">/work/t.py</param></function>");
    CHECK_EQ(r.calls.size(), (size_t)1);
    if (r.calls.size() == 1) {
        const std::string& a = r.calls[0].arguments;
        CHECK_EQ(r.calls[0].name, std::string("edit"));
        /* Normalised to JSON, as `arguments` requires. */
        CHECK(a.find("\"oldText\"") != std::string::npos);
        CHECK(a.find("def minimax:") != std::string::npos);
        CHECK(a.find("'") == std::string::npos);
    }
    mc_content_is_clean(r);
}

/* A plain answer stays plain: no invented call and no marker left. */
TEST(mc_a_reply_with_no_call_is_just_content) {
    if (!minicpm().need()) return;
    ReplyMessage r = mc_parse("Thinking about it.\n</think>\n\nA hash table maps keys to values.");
    CHECK_EQ(r.calls.size(), (size_t)0);
    CHECK(r.content.find("A hash table maps keys to values.") != std::string::npos);
    mc_content_is_clean(r);
}

/* ===================================== the template face, with no checkpoint
 *
 * These cases need no vendor template: ChatML ships in the binary (`ChatTemplate::chatml_source`),
 * so the seam to the vendored Jinja engine is covered even when the cases above skip. They check
 * the three properties chat.h promises:
 *
 *   Errors are values. Upstream throws on a bad template, a rejected message or a malformed
 *   request; every one must come back as a negative RAD_E_*, never cross the request loop.
 *
 *   The fallback is announced. An empty source means the prompt format is a guess.
 *
 *   Capabilities are probed by rendering synthetic inputs, not assumed; a default answer would
 *   report tool support for a template without it.
 */

TEST(an_empty_template_is_the_chatml_fallback_and_says_which_one_it_is) {
    ChatTemplate t;
    CHECK_OK(t.load("", "", ""));
    CHECK(t.ready());
    /* Named, so /props can show the operator that the fallback is in use. */
    CHECK_EQ(t.source(), std::string(ChatTemplate::chatml_source()));
}

TEST(a_template_is_not_ready_until_one_is_loaded) {
    ChatTemplate t;
    CHECK(!t.ready());
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});
    ChatPrompt p;
    /* An error value, not a throw: this is reached from the request loop. */
    CHECK(t.apply(msgs, json(), ChatOptions{}, &p) < 0);
}

TEST(chatml_renders_every_turn_in_order_and_marks_the_assistant_prefix) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));

    json msgs = json::array();
    msgs.push_back({{"role", "system"}, {"content", "You are terse."}});
    msgs.push_back({{"role", "user"}, {"content", "First question."}});
    msgs.push_back({{"role", "assistant"}, {"content", "First answer."}});
    msgs.push_back({{"role", "user"}, {"content", "Second question."}});

    ChatOptions opt;
    opt.add_generation_prompt = true;
    ChatPrompt p;
    CHECK_OK(t.apply(msgs, json(), opt, &p));

    const size_t sys = p.prompt.find("You are terse.");
    const size_t q1  = p.prompt.find("First question.");
    const size_t a1  = p.prompt.find("First answer.");
    const size_t q2  = p.prompt.find("Second question.");
    CHECK(sys != std::string::npos);
    CHECK(q1 != std::string::npos);
    CHECK(a1 != std::string::npos);
    CHECK(q2 != std::string::npos);
    /* In the order sent; positions, not mere presence, are checked. */
    CHECK(sys < q1);
    CHECK(q1 < a1);
    CHECK(a1 < q2);

    /* The generation prompt is a suffix of the prompt; the scheduler and the reply parser rely
     * on it, and the parser cuts the reply there. */
    CHECK(!p.generation_prompt.empty());
    CHECK(p.prompt.size() >= p.generation_prompt.size());
    CHECK_EQ(p.prompt.compare(p.prompt.size() - p.generation_prompt.size(),
                              p.generation_prompt.size(), p.generation_prompt), 0);
}

TEST(without_a_generation_prompt_nothing_is_appended_for_the_assistant) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});

    ChatOptions on;
    on.add_generation_prompt = true;
    ChatPrompt with;
    CHECK_OK(t.apply(msgs, json(), on, &with));

    ChatOptions off;
    off.add_generation_prompt = false;
    ChatPrompt without;
    CHECK_OK(t.apply(msgs, json(), off, &without));

    CHECK(without.prompt.size() < with.prompt.size());
    CHECK_EQ(with.prompt.compare(0, without.prompt.size(), without.prompt), 0);
}

/* The add-BOS flag strips the template's BOS rather than adding one. The container says whether
 * the tokeniser adds BOS; if it does and the template also renders `bos_token`, the render drops
 * the template's so exactly one reaches the model. A doubled BOS is a prompt the model was never
 * trained on.
 *
 * ChatML renders no bos_token, so a minimal inline template is used. */
TEST(a_bos_the_tokeniser_will_add_is_not_also_rendered_into_the_prompt) {
    static const char* src =
        "{{- bos_token -}}"
        "{%- for message in messages -%}{{- message.content -}}{%- endfor -%}";

    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hello"}});

    /* The tokeniser adds none, so the template's stays. */
    ChatTemplate keeps;
    CHECK_OK(keeps.load(src, "<s>", "</s>", false, false));
    ChatPrompt a;
    CHECK_OK(keeps.apply(msgs, json(), ChatOptions{}, &a));
    CHECK_EQ(a.prompt.compare(0, 3, "<s>"), 0);

    /* The tokeniser adds one, so the template's is dropped. */
    ChatTemplate drops;
    CHECK_OK(drops.load(src, "<s>", "</s>", true, false));
    ChatPrompt b;
    CHECK_OK(drops.apply(msgs, json(), ChatOptions{}, &b));
    CHECK(b.prompt.compare(0, 3, "<s>") != 0);
    CHECK_EQ(a.prompt.size() - b.prompt.size(), (size_t)3);

    /* A request may set it for a container that did not; the two are ORed, so a caller can turn
     * it on but not off. */
    ChatPrompt c;
    ChatOptions opt;
    opt.add_bos = true;
    CHECK_OK(keeps.apply(msgs, json(), opt, &c));
    CHECK_EQ(c.prompt, b.prompt);
}
/* The probe answers from the template: ChatML reads only `message.content`, so it must not report
 * tool support. */
TEST(the_capability_probe_reads_what_the_template_actually_does) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    const ChatCaps& c = t.caps();
    CHECK(!c.tools);
    CHECK(!c.tool_calls);
    /* It reads string content and renders any role. */
    CHECK(c.string_content);
    CHECK(c.system_role);
}

/* jinja2 reads `obj.0` as `obj[0]`, and GLM-5's template tests a tool result with
 * `m.content.0.type`. The engine threw on it, and the probe, which swallows a throw, reported a
 * template that renders tools as one that cannot: every request with `tools` was refused. */
TEST(a_dotted_integer_is_an_index_as_in_jinja2) {
    ChatTemplate glm;
    CHECK_OK(glm.load(
        "{%- if tools %}<tools>{% for t in tools %}{{ t.function.name }};{% endfor %}</tools>{% endif %}"
        "{%- for m in messages %}"
        "{%- if m.role == 'tool' and m.content is not mapping and m.content.0.type == 'tool_reference' %}<ref/>"
        "{%- else %}<{{ m.role }}>{{ m.content }}"
        "{%- if m.tool_calls %}{% for c in m.tool_calls %}<call>{{ c.function.name }}</call>{% endfor %}{% endif %}"
        "</{{ m.role }}>{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}<assistant>{% endif %}", "", ""));
    CHECK(glm.caps().tools);
    CHECK(glm.caps().tool_calls);
    /* The string-content probe hands the template an empty tools array, not [null]: this
     * template iterates `tools`, threw on the null, and lost string content. */
    CHECK(glm.caps().string_content);

    ChatTemplate first;
    CHECK_OK(first.load("{{ messages.0.content }}|{{ messages.1.role }}", "", ""));
    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hi"}});
    msgs.push_back({{"role", "assistant"}, {"content", "hello"}});
    ChatPrompt p;
    CHECK_OK(first.apply(msgs, json(), ChatOptions{}, &p));
    CHECK_EQ(p.prompt, std::string("hi|assistant"));
}

/* A template that takes one system message, first, as many do, against conversations the Chat
 * Completions API allows: a system message after a user turn, and two at the head. They are
 * rendered with the system messages merged, in order, into one at the head (#137). Any other
 * refusal comes back in the template's own words, without the parser-generation wrapper. */
TEST(system_messages_a_template_takes_only_first_are_merged_at_the_head) {
    ChatTemplate t;
    CHECK_OK(t.load(
        "{%- for m in messages %}"
        "{%- if m.role == 'system' and not loop.first %}"
        "{{- raise_exception('System message must be at the beginning.') }}{%- endif %}"
        "{%- if m.role == 'user' and m.content == 'boom' %}{{- raise_exception('no boom') }}{%- endif %}"
        "<{{ m.role }}>{{ m.content }}</{{ m.role }}>"
        "{%- endfor %}{%- if add_generation_prompt %}<assistant>{% endif %}", "", ""));
    ChatOptions opt;
    opt.quiet = true;
    auto render = [&](const json& msgs, ChatPrompt* p) { return t.apply(msgs, json(), opt, p); };
    ChatPrompt a, b, c, d;
    CHECK_OK(render(json::parse(R"([{"role":"system","content":"s"},{"role":"user","content":"hi"}])"), &a));
    CHECK_EQ(a.prompt, std::string("<system>s</system><user>hi</user><assistant>"));
    CHECK_OK(render(json::parse(R"([{"role":"user","content":"hi"},{"role":"system","content":"s"}])"), &b));
    CHECK_EQ(b.prompt, std::string("<system>s</system><user>hi</user><assistant>"));
    CHECK_OK(render(json::parse(R"([{"role":"system","content":"s1"},{"role":"system",
        "content":[{"type":"text","text":"s2"}]},{"role":"user","content":"hi"}])"), &c));
    CHECK_EQ(c.prompt, std::string("<system>s1\n\ns2</system><user>hi</user><assistant>"));
    CHECK(render(json::parse(R"([{"role":"system","content":"s"},{"role":"user","content":"boom"}])"),
                 &d) < 0);
    CHECK(d.error.find("no boom") != std::string::npos);
    CHECK(d.error.find("Unable to generate parser") == std::string::npos);
}

/* Unparseable Jinja (common in hand-built containers) is reported at load time as an error value;
 * a throw would cross the bringup path. */
TEST(a_template_that_does_not_parse_is_a_value_and_not_a_throw) {
    ChatTemplate t;
    CHECK(t.load("{%- for message in messages -%}{{ message.content }}", "", "") < 0);
    CHECK(!t.ready());
}

/* Messages come from an HTTP request, so any malformed shape is refused, never thrown. */
TEST(a_message_list_that_is_not_one_is_refused_rather_than_thrown) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    ChatPrompt p;

    /* An object where an array belongs. */
    CHECK(t.apply(json::object({{"role", "user"}}), json(), ChatOptions{}, &p) < 0);
    /* A message with no role. */
    json no_role = json::array();
    no_role.push_back({{"content", "hello"}});
    CHECK(t.apply(no_role, json(), ChatOptions{}, &p) < 0);
    /* A null, as a missing field deserialises. */
    CHECK(t.apply(json(), json(), ChatOptions{}, &p) < 0);
}

/* The JSON-schema converter recurses once per nesting level, so a schema thousands of levels
 * deep (tens of KB of request) would exhaust the stack. It is refused on depth before rendering. */
TEST(a_schema_nested_past_the_limit_is_refused_before_it_is_rendered) {
    ChatTemplate t;
    CHECK_OK(t.load(ChatTemplate::chatml_source(), "", ""));
    std::string deep;
    for (int i = 0; i < 20000; ++i) deep += R"({"type":"array","items":)";
    deep += R"({"type":"integer"})";
    for (int i = 0; i < 20000; ++i) deep += "}";

    json msgs = json::array();
    msgs.push_back({{"role", "user"}, {"content", "hi"}});
    ChatPrompt p;
    ChatOptions opt;
    opt.json_schema = deep;
    CHECK_EQ(t.apply(msgs, json(), opt, &p), RAD_E_INVAL);
    CHECK(p.error.find("levels deep") != std::string::npos);

    json tools = json::array();
    tools.push_back({{"type", "function"},
                     {"function", {{"name", "f"}, {"parameters", json::parse(deep)}}}});
    ChatPrompt q;
    CHECK_EQ(t.apply(msgs, tools, ChatOptions{}, &q), RAD_E_INVAL);
}

RAD_TEST_MAIN()
