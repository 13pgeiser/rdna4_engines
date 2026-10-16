/* tokenizer_perf_test.cpp -- the tokeniser's cost, held to budgets.
 *
 * Tokenisation is on the request path: an agent re-sends its whole conversation every turn, and
 * every byte is encoded before the first token is generated. Checked here, in work
 * (tests/work_counter.h) so the bounds hold on a loaded machine:
 *
 *   - linear scaling: four times the text is about four times the work, on ordinary text and on
 *     inputs a broken pre-tokeniser is quadratic on (one huge word, a huge whitespace run,
 *     digits, alternating classes);
 *   - the encoder against the reference encoder (tools/tokref) on the same text;
 *   - the segment cache: the last turn of a long conversation hits for every earlier turn and
 *     costs a fraction of a full encode.
 *
 * Cases named perf_ hold wall-time budgets (an absolute throughput floor, pattern compile time);
 * they belong to the serial performance tier.
 */
#include "rad_test.h"
#include "work_counter.h"
#include "tokenizer_fixture.h"
#include "tokref.h"

#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>

using namespace rad;
using namespace tokfix;

namespace {

double best_of(int n, const std::function<void()>& f) {
    double best = 1e30;
    for (int i = 0; i < n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

/* Prose and code in the proportions an agent's conversation has them. */
std::string document(uint32_t seed, size_t bytes) {
    static const char* words[] = {
        "the", "cache", "is", "sharded", "by", "key", "and", "each", "shard", "has", "its", "own",
        "lock", "so", "that", "readers", "never", "wait", "on", "writers", "of", "another", "we",
        "measured", "it", "at", "requests", "per", "second", "with", "p99", "latency", "under",
        "load", "tokenizer", "encode", "returns", "ids", "for", "text", "value", "struct", "func",
        "return", "if", "else", "nil", "err", "string", "int", "len", "make", "map", "append",
        "更新", "缓存", "naïve", "café", "über", "Straße", "ünïcode", "don't", "it's", "we'll",
    };
    static const char* punct[] = { ".", ",", ";", ":", "(", ")", "{", "}", "[", "]", " =", " :=",
                                   " ==", " !=", " //", " ->", "\"", "'", "?", "!" };
    std::mt19937 rng(seed);
    std::string s;
    while (s.size() < bytes) {
        const int n = 3 + rng() % 14;
        for (int k = 0; k < n; ++k) {
            if (k) s += ' ';
            s += words[rng() % (sizeof(words) / sizeof(words[0]))];
            if (rng() % 5 == 0) s += punct[rng() % (sizeof(punct) / sizeof(punct[0]))];
            if (rng() % 17 == 0) s += std::to_string(rng() % 100000);
        }
        s += rng() % 3 ? ".\n" : (rng() % 2 ? "\n\t" : "\n\n    ");
    }
    return s;
}

std::shared_ptr<const Vocab> vocab() {
    static std::shared_ptr<const Vocab> v = [] {
        VocabBuild vb = train_bytelevel(document(1, 300000), k_qwen35, 1500);
        vb.steps = { simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() };
        add_specials(vb, { { "<|im_start|>", RAD_TT_CONTROL }, { "<|im_end|>", RAD_TT_CONTROL } });
        return load(std::move(vb));
    }();
    return v;
}

struct Pathological {
    const char* name;
    std::function<std::string(size_t)> make;
};

const std::vector<Pathological>& pathological_inputs() {
    static const std::vector<Pathological> v = {
        { "one word", [](size_t n) { return std::string(n, 'a'); } },
        { "whitespace", [](size_t n) { return std::string(n, ' ') + "x"; } },
        { "mixed whitespace", [](size_t n) { std::string s; while (s.size() < n) s += " \t \n"; return s + "x"; } },
        { "digits", [](size_t n) { std::string s; while (s.size() < n) s += "0123456789"; return s; } },
        { "alternating", [](size_t n) { std::string s; while (s.size() < n) s += "a1!"; return s; } },
        { "apostrophes", [](size_t n) { std::string s; while (s.size() < n) s += "''s"; return s; } },
    };
    return v;
}

double encode_seconds(const Tokenizer& t, const std::string& text, int runs = 5) {
    std::vector<int32_t> ids;
    return best_of(runs, [&] { ids.clear(); t.encode(text, ids, false, true); });
}

/* The work of one encode, in radtest::Work's unit. Without an instruction counter the unit is CPU
 * time, which a busy machine inflates run to run, so the least of three runs is taken. */
uint64_t encode_work(const Tokenizer& t, const std::string& text) {
    std::vector<int32_t> ids;
    const int runs = radtest::work().exact() ? 1 : 3;
    uint64_t best = UINT64_MAX;
    for (int r = 0; r < runs; ++r) {
        ids.clear();
        best = std::min(best, radtest::work().of([&] { t.encode(text, ids, false, true); }));
    }
    return best;
}

}  /* namespace */

/* Four times the text is about four times the work. The texts differ, so a cache of one cannot
 * help the other; the encoder's per-call word cache makes the larger one slightly cheaper per byte
 * (about x3.3). Counted in work, so the ratio does not depend on the machine's load or on where the
 * working set leaves the core's cache. */
TEST(encoding_work_is_linear_in_the_text) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const std::string one = document(3, 256 << 10), four = document(4, 1 << 20);
    const uint64_t a = encode_work(t, one), b = encode_work(t, four);
    std::fprintf(stderr, "    256 KB %llu, 1 MB %llu %s, ratio %.2f\n", (unsigned long long)a,
                 (unsigned long long)b, radtest::work().unit(), (double)b / (double)a);
    CHECK((double)b / (double)a > 2.6);
    CHECK((double)b / (double)a < 4.6);
}

/* Inputs a splitter goes quadratic on. A backtracking `\s+(?!\S)` rescans a whitespace run from
 * every position in it, and a merge quadratic in word length shows on one huge word; both must
 * stay linear. 125 and 500 KB: linear is 4 (a whitespace run is x4.3), quadratic 16. Counted in CPU
 * time the larger text also pays for leaving the core's cache (x7.4 on a whitespace run with every
 * core busy), so that bound is wider; both stay far below quadratic. */
TEST(pathological_inputs_stay_linear) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    for (const auto& in : pathological_inputs()) {
        const std::string one = in.make(125000), four = in.make(500000);
        const uint64_t a = encode_work(t, one), b = encode_work(t, four);
        std::fprintf(stderr, "    %-17s 125 KB %llu, 500 KB %llu, ratio %.2f\n", in.name,
                     (unsigned long long)a, (unsigned long long)b, (double)b / (double)a);
        CHECK((double)b / (double)a < (radtest::work().exact() ? 6.0 : 9.0));
    }
}

/* The encoder does a fraction of the reference encoder's work (tools/tokref, the plain
 * formulation it is checked against) on the same text. */
TEST(the_encoder_does_a_fraction_of_the_reference_s_work) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const std::string text = document(2, 512 << 10);
    const uint64_t e = encode_work(t, text);
    std::vector<int32_t> ids;
    const uint64_t r = radtest::work().of([&] { tokref::encode(*v, text, ids, false, true); });
    std::fprintf(stderr, "    512 KB: encoder %llu, reference %llu %s (%.1fx)\n",
                 (unsigned long long)e, (unsigned long long)r, radtest::work().unit(),
                 (double)r / (double)e);
    CHECK((double)r / (double)e > 4.0);
}

/* The last turn of a long conversation costs only what is new in it: every earlier turn is a
 * segment the cache holds, so encoding the last prompt hits once per earlier turn and misses once,
 * for the new one, and does a fraction of an uncached encode's work. */
TEST(a_resent_conversation_costs_its_new_turn) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer plain, cached;
    plain.init(v);
    plain.set_segment_cache(false);
    cached.init(v);
    cached.clear_cache();
    std::string conv;
    std::vector<std::string> prompts;
    for (int k = 0; k < 40; ++k) {
        conv += std::string("<|im_start|>") + (k % 2 ? "assistant" : "user") + "\n" +
                document(100 + k, 15000) + "<|im_end|>\n";
        prompts.push_back(conv + "<|im_start|>assistant\n");
    }
    std::vector<int32_t> ids;
    for (size_t k = 0; k + 1 < prompts.size(); ++k) { ids.clear(); cached.encode(prompts[k], ids, false, true); }
    const std::string& last = prompts.back();
    const Tokenizer::CacheStats before = cached.cache_stats();
    std::vector<int32_t> warm_ids, cold_ids;
    const uint64_t warm = radtest::work().of([&] { cached.encode(last, warm_ids, false, true); });
    const Tokenizer::CacheStats after = cached.cache_stats();
    const uint64_t cold = radtest::work().of([&] { plain.encode(last, cold_ids, false, true); });
    std::fprintf(stderr, "    %zu bytes: uncached %llu, cached %llu %s (%.0fx)\n", last.size(),
                 (unsigned long long)cold, (unsigned long long)warm, radtest::work().unit(),
                 (double)cold / (double)warm);
    CHECK(warm_ids == cold_ids);
    CHECK_EQ(after.hits - before.hits, (uint64_t)39);
    CHECK_EQ(after.misses - before.misses, (uint64_t)1);
    CHECK(warm * 5 < cold);
}

/* Wall-time budgets, for the serial performance tier: an absolute throughput floor on prose and
 * code, and the encoder ahead of the reference in time as well as in work. */
TEST(perf_throughput_has_a_floor_and_beats_the_reference) {
    auto v = vocab();
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    const std::string text = document(2, 2 << 20);
    const double s = encode_seconds(t, text);
    const double mbs = text.size() / s / 1e6;
    std::vector<int32_t> ids;
    const double r = best_of(2, [&] { ids.clear(); tokref::encode(*v, text, ids, false, true); });
    std::fprintf(stderr, "    %.1f MB/s (reference %.1f MB/s, %.1fx)\n", mbs, text.size() / r / 1e6, r / s);
    CHECK(mbs > 25.0);
    CHECK(r / s > 4.0);
}

/* Compiling a shipped pattern is a load-time cost, paid once; it stays small. */
TEST(perf_compiling_a_shipped_pattern_is_cheap) {
    for (const char* p : { k_qwen35, k_minicpm, k_llama3, k_gpt2 }) {
        std::unique_ptr<TokRegex> re;
        std::string why;
        const double s = best_of(3, [&] { re.reset(); TokRegex::compile(p, &re, &why); });
        CHECK(re != nullptr);
        CHECK(s < 0.05);
    }
}

RAD_TEST_MAIN()
