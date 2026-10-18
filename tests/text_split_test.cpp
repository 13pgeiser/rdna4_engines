/* tests/text_split_test.cpp -- sentences and clauses of text that arrives in pieces
 * (core/server/text_split.h), which the realtime speech session speaks one at a time.
 */
#include "rad_test.h"

#include "server/text_split.h"

#include <string>
#include <vector>

using namespace rad::server;

namespace {

using Units = std::vector<std::string>;

/* The units of `text` given whole, the rest left pending (or taken too with `final`). */
Units units(const std::string& text, TextSplit mode, bool final, std::string* rest = nullptr) {
    std::string p = text;
    Units u = take_text_units(&p, mode, final);
    if (rest) *rest = p;
    return u;
}

}  /* namespace */

/* A sentence ends at its stop once whitespace follows; the stop at the end of what has arrived
 * waits for the next piece, and the end of the input ends the last sentence however it ends. */
TEST(a_sentence_ends_when_the_text_after_it_shows_it_did) {
    std::string rest;
    CHECK(units("Hello world. How are", TextSplit::Sentence, false, &rest) ==
          Units({ "Hello world." }));
    CHECK_EQ(rest, std::string("How are"));
    CHECK(units("Hello.", TextSplit::Sentence, false, &rest).empty());
    CHECK_EQ(rest, std::string("Hello."));
    CHECK(units("Hello. ", TextSplit::Sentence, false, &rest) == Units({ "Hello." }));
    CHECK(units("Hello world. How are", TextSplit::Sentence, true, &rest) ==
          Units({ "Hello world.", "How are" }));
    CHECK(rest.empty());
    CHECK(units("Wait!  Is it?\nYes", TextSplit::Sentence, true) ==
          Units({ "Wait!", "Is it?", "Yes" }));
}

/* What does not end a sentence: a decimal point, a common abbreviation, an initial. A run of
 * stops and closing quotes or brackets stays with its sentence. */
TEST(decimals_abbreviations_and_initials_do_not_end_a_sentence) {
    CHECK(units("It costs 3.14 dollars. Then", TextSplit::Sentence, false) ==
          Units({ "It costs 3.14 dollars." }));
    CHECK(units("Dr. Smith and Mrs. Jones left. Then", TextSplit::Sentence, false) ==
          Units({ "Dr. Smith and Mrs. Jones left." }));
    CHECK(units("Fruit, e.g. apples, etc. is fine. Then", TextSplit::Sentence, false) ==
          Units({ "Fruit, e.g. apples, etc. is fine." }));
    CHECK(units("J.R. Tolkien and J. Smith wrote. Then", TextSplit::Sentence, false) ==
          Units({ "J.R. Tolkien and J. Smith wrote." }));
    CHECK(units("Really?! Yes... Fine.", TextSplit::Sentence, true) ==
          Units({ "Really?!", "Yes...", "Fine." }));
    CHECK(units("He said \"stop.\" Then (he left.) Done", TextSplit::Sentence, true) ==
          Units({ "He said \"stop.\"", "Then (he left.)", "Done" }));
}

/* CJK, Indic and Arabic stops end a sentence at the next character, since those scripts put no
 * space after them; at the end of what has arrived they too wait. */
TEST(cjk_indic_and_arabic_stops_need_no_space) {
    std::string rest;
    CHECK(units("你好。今天天气很好！我们", TextSplit::Sentence, false, &rest) ==
          Units({ "你好。", "今天天气很好！" }));
    CHECK_EQ(rest, std::string("我们"));
    CHECK(units("你好。", TextSplit::Sentence, false, &rest).empty());
    CHECK_EQ(rest, std::string("你好。"));
    CHECK(units("नमस्ते। आप कैसे हैं", TextSplit::Sentence, false) == Units({ "नमस्ते।" }));
    CHECK(units("مرحبا؟ كيف حالك", TextSplit::Sentence, false) == Units({ "مرحبا؟" }));
    CHECK(units("「好。」然后", TextSplit::Sentence, false) == Units({ "「好。」" }));
}

/* Clauses also end at commas and semicolons, full-width and Arabic ones included; `none` never
 * cuts, and gives the whole text at the end. */
TEST(clauses_cut_at_commas_and_none_never_cuts) {
    CHECK(units("one, two; three. four", TextSplit::Clause, true) ==
          Units({ "one,", "two;", "three.", "four" }));
    CHECK(units("one, two; three. four", TextSplit::Sentence, true) ==
          Units({ "one, two; three.", "four" }));
    CHECK(units("一，二；三", TextSplit::Clause, true) == Units({ "一，", "二；", "三" }));
    std::string rest;
    CHECK(units("One. Two. ", TextSplit::None, false, &rest).empty());
    CHECK_EQ(rest, std::string("One. Two. "));
    CHECK(units("  One. Two.  ", TextSplit::None, true) == Units({ "One. Two." }));
}

/* A unit with nothing to say is dropped, and blank input gives nothing. */
TEST(a_unit_of_only_punctuation_is_dropped) {
    CHECK(units("... ok. !!", TextSplit::Sentence, true) == Units({ "ok." }));
    CHECK(units("   ", TextSplit::Sentence, true).empty());
    CHECK(units("", TextSplit::Clause, true).empty());
}

/* Text fed a byte at a time (cutting UTF-8 sequences) gives the units it gives whole. */
TEST(pieces_give_the_units_the_whole_text_gives) {
    const std::string text = "Dr. Who said 3.5 things. 你好。Then, \"yes!\" he left; ok. Done";
    for (TextSplit mode : { TextSplit::Sentence, TextSplit::Clause, TextSplit::None }) {
        const Units whole = units(text, mode, true);
        Units got;
        std::string p;
        for (char c : text) {
            p += c;
            for (std::string& u : take_text_units(&p, mode, false)) got.push_back(u);
        }
        for (std::string& u : take_text_units(&p, mode, true)) got.push_back(u);
        CHECK(got == whole);
        CHECK(!whole.empty());
    }
}

RAD_TEST_MAIN()
