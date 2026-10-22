/* precompiled_test.cpp -- extended grapheme clusters and SentencePiece's precompiled normaliser
 * (core/text/tokenizer_grapheme.h) against Unicode's own test file and HuggingFace tokenizers'
 * Precompiled normaliser (goldens: mk_precompiled_goldens.py). */
#include "rad_test.h"
#include "text/tokenizer.h"
#include "text/tokenizer_grapheme.h"

#include "precompiled_golden.h"

#include <string>
#include <vector>

using namespace rad;

/* Every case of Unicode 16.0.0's GraphemeBreakTest.txt: the clusters grapheme_end walks out of the
 * sequence start where the file breaks it. */
TEST(grapheme_clusters_follow_uax29) {
    int bad = 0;
    for (const GbCase& g : kGbCases) {
        uint32_t starts = 0;
        for (size_t i = 0; i < g.n;) {
            if (i) starts |= 1u << i;
            const size_t e = grapheme_end(g.cp, g.n, i);
            if (e <= i || e > g.n) { ++bad; break; }
            i = e;
        }
        if (starts != g.starts) {
            if (++bad <= 5) {
                std::string s;
                for (uint32_t k = 0; k < g.n; ++k) s += fmt(" %04X", g.cp[k]);
                fprintf(stderr, "    [%s ] starts 0x%x, want 0x%x\n", s.c_str(), starts, g.starts);
            }
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(sizeof kGbCases / sizeof kGbCases[0] > 1000);
}

/* A cluster under six bytes goes whole to its shortest mapped prefix's replacement (a combining
 * accent after a mapped letter is dropped); otherwise each character goes to its own. */
TEST(precompiled_normaliser_matches_tokenizers) {
    PrecompiledMap pm;
    std::string why;
    REQUIRE_EQ(pm.load(kPcCharsmap, &why), (int)RAD_OK);
    for (const PcCase& c : kPcCases) {
        const std::string got = pm.normalize(c.in);
        if (got != c.out) fprintf(stderr, "    '%s' -> '%s', want '%s'\n", c.in, got.c_str(), c.out);
        CHECK(got == c.out);
    }
}

TEST(precompiled_map_refuses_what_does_not_decode) {
    PrecompiledMap pm;
    std::string why;
    CHECK_EQ(pm.load("not base64!", &why), RAD_E_FORMAT);
    CHECK_EQ(pm.load("AAA=", &why), RAD_E_FORMAT);
    /* A trie byte count past the map's end. */
    CHECK_EQ(pm.load("AAEAAA==", &why), RAD_E_FORMAT);
    CHECK(!pm.ready());
}

/* The normaliser inside a Unigram vocabulary, as T5's tokenizer.json declares it: Precompiled,
 * a right Strip and a Replace of space runs, then Metaspace. With no tokenizer_config.json beside
 * it the end token's id is unset, so encoding adds none (HuggingFace's ids: the same pieces, then
 * </s>). */
TEST(precompiled_runs_in_a_tokenizer_json_chain) {
    const std::string json = std::string(R"({
      "version": "1.0", "added_tokens": [
        {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 2, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
      "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Precompiled", "precompiled_charsmap": ")") + kPcCharsmap + R"("},
        {"type": "Strip", "strip_left": false, "strip_right": true},
        {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": "▁"}]},
      "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true},
      "post_processor": {"type": "TemplateProcessing",
        "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "special_tokens": {"</s>": {"id": "</s>", "ids": [1], "tokens": ["</s>"]}}},
      "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
      "model": {"type": "Unigram", "unk_id": 2, "vocab": [
        ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0], ["▁", -2.0], ["▁a", -3.0], ["a", -4.0],
        ["▁big", -3.0], ["b", -5.0], ["i", -5.0], ["g", -5.0], ["▁apple", -3.0], ["p", -5.0],
        ["l", -5.0], ["e", -5.0], ["é", -5.0], ["▁café", -3.0], ["c", -5.0], ["f", -5.0]]}
    })";
    VocabBuild vb;
    REQUIRE_EQ(parse_tokenizer_json_buf(json, "t5-like", vb), (int)RAD_OK);
    auto v = std::make_shared<Vocab>();
    REQUIRE_EQ(v->load(std::move(vb)), (int)RAD_OK);
    Tokenizer tok;
    REQUIRE_EQ(tok.init(v), (int)RAD_OK);
    std::vector<int32_t> ids;
    REQUIRE_EQ(tok.encode("A big Apple", ids, true, true), (int)RAD_OK);
    CHECK(ids == std::vector<int32_t>({ 4, 6, 10 }));
    ids.clear();
    /* The fullwidth A normalises to an A the vocabulary lacks, and the decomposed accent to the
     * piece it holds. */
    REQUIRE_EQ(tok.encode("\xef\xbc\xa1 cafe\xcc\x81", ids, true, true), (int)RAD_OK);
    CHECK(ids == std::vector<int32_t>({ 3, 2, 15 }));
}

RAD_TEST_MAIN()
