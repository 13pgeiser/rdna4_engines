/* recipe_test.cpp -- the rules rad-convert quantises by (core/format/recipe.h).
 *
 * A recipe decides every weight of a conversion that can take hours, and a rule that matches the
 * wrong weights (or none) shows only when the container is measured afterwards. These cases pin
 * the matching, the order and the two spellings to spec §4.4.
 */
#include "rad_test.h"
#include "format/recipe.h"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

using namespace rad;

TEST(a_glob_matches_the_whole_name) {
    CHECK(recipe_glob("blk.*.ffn_*_exps.*.weight", "blk.12.ffn_down_exps.511.weight"));
    CHECK(recipe_glob("*", "output.weight"));
    CHECK(recipe_glob("output.weight", "output.weight"));
    CHECK(recipe_glob("blk.?.attn_q.weight", "blk.7.attn_q.weight"));
    CHECK(!recipe_glob("blk.?.attn_q.weight", "blk.17.attn_q.weight"));   /* ? is one character */
    CHECK(recipe_glob("blk.*.weight", "blk.1.2.3.weight"));               /* * crosses dots */
    CHECK(!recipe_glob("blk.*.weight", "blk.1.scale"));
    CHECK(!recipe_glob("output", "output.weight"));                        /* not a prefix match */
    CHECK(recipe_glob("*_exps.*", "blk.0.ffn_gate_up_exps.3.weight"));
    CHECK(recipe_glob("a*b*c", "aXXbYYbZc"));                              /* backtracks */
    CHECK(!recipe_glob("a*b*c", "aXXbYY"));
}

TEST(the_first_rule_that_matches_decides) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("blk.*.ffn_down_exps.*.weight  gptq codes=i4 group=128\n"
                          "blk.*.weight                  rtn  codes=fp8_e4m3 block=128x128\n",
                          "r", &why), RAD_OK);
    const RecipeRule* a = r.match("blk.3.ffn_down_exps.7.weight");
    const RecipeRule* b = r.match("blk.3.attn_q.weight");
    REQUIRE(a && b);
    CHECK_EQ(a->quantizer, std::string("gptq"));
    CHECK_EQ(b->quantizer, std::string("rtn"));
    CHECK(r.match("output.weight") == nullptr);                 /* kept as the checkpoint holds it */
}

TEST(a_command_line_rule_goes_ahead_of_the_file) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128,scale=bf16", &why),
               RAD_OK);
    REQUIRE_EQ(r.add_text("* rtn codes=i8 group=128\n", "file", &why), RAD_OK);
    const RecipeRule* m = r.match("output.weight");
    REQUIRE(m != nullptr);
    CHECK_EQ(m->origin, std::string("--quant"));
    CHECK_EQ(m->options.size(), (size_t)3);
    CHECK_EQ(m->options[0].first, std::string("codes"));
    CHECK_EQ(m->options[2].second, std::string("bf16"));
    CHECK_EQ(r.match("blk.0.attn_q.weight")->origin, std::string("file:1"));
}

TEST(comments_and_blank_lines_are_not_rules_and_a_bad_line_is_named) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("# a recipe\n\n   \nout* rtn codes=i8  # trailing\n", "x.recipe", &why),
               RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)1);
    CHECK_EQ(r.rules()[0].options.size(), (size_t)1);
    CHECK_EQ(r.rules()[0].origin, std::string("x.recipe:4"));

    Recipe bad;
    CHECK_EQ(bad.add_text("a rtn\nlonely\n", "y.recipe", &why), RAD_E_FORMAT);
    CHECK(why.find("y.recipe:2") != std::string::npos);
    CHECK_EQ(bad.add_text("a rtn group\n", "z", &why), RAD_E_FORMAT);   /* not key=value */
    CHECK_EQ(bad.add_text("a rtn k=1 k=2\n", "z", &why), RAD_E_FORMAT); /* given twice */
    CHECK(why.find("twice") != std::string::npos);
    CHECK_EQ(bad.add_flag("noquantiser", &why), RAD_E_FORMAT);
    CHECK_EQ(bad.add_flag("p=:k=v", &why), RAD_E_FORMAT);
}

/* $NAME expands from the environment, so one recipe serves every machine. An unset name is
 * refused: quantising against an empty calibration directory would silently do something else. */
TEST(a_variable_is_the_environments_and_an_unset_one_is_refused) {
    ::setenv("RAD_RECIPE_TEST_DIR", "/data/calib", 1);
    ::unsetenv("RAD_RECIPE_TEST_UNSET");
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_DIR/q38 tag=${RAD_RECIPE_TEST_DIR}x\n",
                          "v", &why), RAD_OK);
    CHECK_EQ(r.rules()[0].options[0].second, std::string("/data/calib/q38"));
    CHECK_EQ(r.rules()[0].options[1].second, std::string("/data/calibx"));
    CHECK_EQ(r.add_text("* gptq calib=$RAD_RECIPE_TEST_UNSET\n", "v", &why), RAD_E_FORMAT);
    CHECK(why.find("RAD_RECIPE_TEST_UNSET") != std::string::npos);
    CHECK_EQ(r.add_text("* gptq calib=${RAD_RECIPE_TEST_DIR\n", "v", &why), RAD_E_FORMAT);
}

/* A container records the options in one canonical spelling, so --reuse compares two
 * conversions by what they asked for, not by how the rule was typed. */
TEST(the_recorded_options_do_not_depend_on_how_they_were_written) {
    Recipe a, b;
    std::string why;
    REQUIRE_EQ(a.add_text("* rtn scale=bf16 codes=i4 group=128\n", "a", &why), RAD_OK);
    REQUIRE_EQ(b.add_flag("*=rtn:group=128,codes=i4,scale=bf16", &why), RAD_OK);
    CHECK_EQ(a.rules()[0].options_text(), b.rules()[0].options_text());
    CHECK_EQ(a.rules()[0].options_text(), std::string("codes=i4,group=128,scale=bf16"));
}

/* The recipe a container carries reads back as the same rules. */
TEST(the_recorded_recipe_reads_back_as_the_same_rules) {
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_flag("output.weight=rtn:codes=fp8_e4m3,block=128x128", &why), RAD_OK);
    REQUIRE_EQ(r.add_text("blk.*.weight gptq codes=i4 group=128 damp=0.02\n", "f", &why), RAD_OK);
    Recipe back;
    REQUIRE_EQ(back.add_text(r.text(), "header", &why), RAD_OK);
    REQUIRE_EQ(back.rules().size(), r.rules().size());
    for (size_t i = 0; i < r.rules().size(); ++i) {
        CHECK_EQ(back.rules()[i].pattern, r.rules()[i].pattern);
        CHECK_EQ(back.rules()[i].quantizer, r.rules()[i].quantizer);
        CHECK(back.rules()[i].options == r.rules()[i].options);
    }
}

namespace {
bool has_opt(const RecipeRule* r, const char* k, const char* v) {
    for (const auto& [kk, vv] : r->options)
        if (kk == k && vv == v) return true;
    return false;
}
}  /* namespace */

/* The NVFP4 recipe as it ships: it parses, and its patterns decide the 27B architecture's
 * declared names. The linears (fused MLPs, attention, the delta net's in- and out-projections)
 * go to MXFP4, the head to block fp8, and the norms, the convolution, the a/b projection and the
 * embedding keep the checkpoint's form. */
TEST(the_nvfp4_recipe_decides_the_model_it_was_written_for) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/q38-27b-nvfp4-df2.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "q38-27b-nvfp4-df2.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)17);

    /* every linear, across the layer range, by the fused names the builder declares */
    const char* linears[] = { "blk.47.ffn_gate_up.weight", "blk.47.ffn_down.weight",
                              "blk.3.ffn_gate_up.weight", "blk.3.attn_qg.weight",
                              "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                              "blk.3.attn_output.weight", "blk.47.ssm_inz.weight",
                              "blk.47.ssm_out.weight" };
    for (const char* n : linears) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "fp4_e2m1"));
        CHECK(has_opt(m, "group", "32"));
        CHECK(has_opt(m, "scale", "e8m0"));
    }
    /* the head to block fp8 */
    const RecipeRule* head = r.match("output.weight");
    REQUIRE(head != nullptr);
    CHECK(has_opt(head, "codes", "fp8_e4m3"));
    CHECK(has_opt(head, "block", "128x128"));
    /* kept as the checkpoint ships them (the a/b projection is bf16 there) */
    const char* kept[] = { "blk.0.ffn_norm.weight", "blk.0.attn_norm.weight", "blk.47.ssm_conv1d.weight",
                           "blk.47.ssm_a_log", "blk.47.ssm_ab.weight", "token_embd.weight", "output_norm.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
    /* the drafter's own linears */
    const RecipeRule* fc = r.match("dflash.fc.weight");
    REQUIRE(fc != nullptr);
    CHECK(has_opt(fc, "codes", "fp4_e2m1"));
    const RecipeRule* dh = r.match("dflash.draft_head.weight");
    REQUIRE(dh != nullptr);
    CHECK(has_opt(dh, "codes", "u2"));
}

/* The Sage-A1.5 recipe as it ships: every trunk linear -- attention, the delta net's in- and
 * out-projections, the routed and shared experts of all 40 blocks and the MTP layer at its own
 * index 40 -- goes to block fp8, the head likewise, and the routers, the shared-expert gates,
 * the delta net's parameters, the norms, the embedding and the vision tower keep the
 * checkpoint's form. */
TEST(the_sage_recipe_decides_the_model_it_was_written_for) {
    std::string dir = __FILE__;
    dir = dir.substr(0, dir.find_last_of('/')) + "/../data/recipes";
    std::ifstream in(dir + "/sage-a1.5-fp8.recipe");
    REQUIRE(in.good());
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Recipe r;
    std::string why;
    REQUIRE_EQ(r.add_text(text, "sage-a1.5-fp8.recipe", &why), RAD_OK);
    CHECK_EQ(r.rules().size(), (size_t)12);

    const char* linears[] = { "blk.0.attn_qg.weight", "blk.3.attn_qg.weight", "blk.39.attn_qg.weight",
                              "blk.40.attn_qg.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
                              "blk.39.attn_output.weight", "blk.0.ssm_inz.weight", "blk.39.ssm_out.weight",
                              "blk.0.ffn_gate_up_exps.0.weight", "blk.19.ffn_down_exps.255.weight",
                              "blk.40.ffn_gate_up_exps.0.weight", "blk.40.ffn_down_exps.255.weight",
                              "blk.0.ffn_gate_up_shexp.weight", "blk.40.ffn_down_shexp.weight" };
    for (const char* n : linears) {
        const RecipeRule* m = r.match(n);
        REQUIRE(m != nullptr);
        CHECK_EQ(m->quantizer, std::string("rtn"));
        CHECK(has_opt(m, "codes", "fp8_e4m3"));
        CHECK(has_opt(m, "block", "128x128"));
        CHECK(has_opt(m, "scale", "bf16"));
    }
    const RecipeRule* head = r.match("output.weight");
    REQUIRE(head != nullptr);
    CHECK(has_opt(head, "codes", "fp8_e4m3"));
    CHECK(has_opt(head, "block", "128x128"));
    const RecipeRule* dh = r.match("mtp.draft_head.weight");
    REQUIRE(dh != nullptr);
    CHECK(has_opt(dh, "codes", "u2"));
    CHECK(has_opt(dh, "zero", "u8"));
    CHECK(has_opt(dh, "group", "128"));
    CHECK(has_opt(dh, "scale", "f16"));

    /* kept as the checkpoint ships them: the routers, the gates, the delta net's parameters,
     * the norms, the embedding and the whole vision tower */
    const char* kept[] = { "blk.0.ffn_norm.weight", "blk.0.attn_norm.weight", "blk.0.ssm_ab.weight",
                           "blk.0.ssm_conv1d.weight", "blk.0.ssm_a_log", "blk.0.ssm_dt.bias",
                           "blk.0.ssm_norm.weight", "blk.0.ffn_gate_inp.weight",
                           "blk.0.ffn_gate_shexp.weight", "blk.40.ffn_gate_inp.weight",
                           "token_embd.weight", "output_norm.weight", "mtp.fc.weight",
                           "mtp.norm.weight", "v.blk.0.qkv.weight", "v.merger.fc1.weight" };
    for (const char* n : kept)
        CHECK(r.match(n) == nullptr);
}

RAD_TEST_MAIN()
