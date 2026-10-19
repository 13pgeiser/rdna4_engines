/* gen_test.cpp -- the image generator's request side (core/gen/gen.h) and its pass driver's
 * admission (core/gen/driver.h), against sched_fakes.h's KV manager and device layer.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/driver.h"
#include "gen/gen.h"

#include <cmath>
#include <string>
#include <vector>

using namespace rad;

namespace {

RadGeneratorDecl decl_base() {
    RadGeneratorDecl d{};
    d.name = "test";
    d.layout = RAD_GEN_LAYOUT_JOINT_GRID;
    d.latent_scale = 16;
    d.size_multiple = 32;
    d.pixel_channels = 4;
    d.slot_rows = 4;
    d.max_tokens = 1024;
    d.max_refs = 2;
    d.ref_area = 256 * 256;
    d.default_steps = 4;
    d.shift = 1.0;
    d.max_lanes = 4;
    return d;
}

/* The schedule diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps gives, in double:
 * linspace(1, 1/n, n), shifted (exp(mu) / (exp(mu) + 1/t - 1) dynamically, s*t / (1 + (s-1)*t)
 * statically), stretched so the last sigma is the terminal, then a closing 0. */
std::vector<double> diffusers_sigmas(const RadGeneratorDecl& d, int n, int64_t tokens) {
    std::vector<double> s;
    for (int i = 0; i < n; ++i) s.push_back(n == 1 ? 1.0 : 1.0 + (1.0 / n - 1.0) * i / (n - 1));
    if (d.dyn_shift) {
        const double m = (d.max_shift - d.base_shift) / (double)(d.max_seq - d.base_seq);
        const double mu = (double)tokens * m + d.base_shift - m * (double)d.base_seq;
        const double e = d.shift_exp ? std::exp(mu) : mu;
        for (double& t : s) t = e / (e + (1.0 / t - 1.0));
    } else {
        for (double& t : s) t = d.shift * t / (1.0 + (d.shift - 1.0) * t);
    }
    if (d.shift_terminal > 0) {
        const double scale = (1.0 - s.back()) / (1.0 - d.shift_terminal);
        for (double& t : s) t = 1.0 - (1.0 - t) / scale;
    }
    s.push_back(0.0);
    return s;
}

bool close_to(const std::vector<float>& got, const std::vector<double>& want, double tol) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs((double)got[i] - want[i]) > tol) return false;
    return true;
}

}  /* namespace */

/* ================================================================== the schedule */
/* The sigmas follow diffusers' FlowMatchEuler for both shifts, the exponential and linear
 * dynamic forms and the terminal stretch, and end in 0. A wrong sigma degrades every image with
 * no error anywhere. */
TEST(the_schedule_is_diffusers_flow_match_euler) {
    RadGeneratorDecl d = decl_base();
    d.dyn_shift = 1; d.shift_exp = 1;
    d.base_shift = 0.5; d.max_shift = 0.9; d.base_seq = 256; d.max_seq = 8192;
    d.shift_terminal = 0.02;
    for (int n : { 1, 4, 30 })
        for (int64_t tokens : { (int64_t)256, (int64_t)1024, (int64_t)4096 }) {
            const std::vector<float> s = gen::schedule(d, {}, n, tokens);
            CHECK(close_to(s, diffusers_sigmas(d, n, tokens), 2e-6));
            CHECK_EQ(s.back(), 0.0f);
            if (n > 1) CHECK_NEAR(s[(size_t)n - 1], 0.02, 2e-6);   /* the terminal */
        }
    d.shift_exp = 0;
    CHECK(close_to(gen::schedule(d, {}, 6, 2048), diffusers_sigmas(d, 6, 2048), 2e-6));
    d.dyn_shift = 0; d.shift_terminal = 0; d.shift = 3.0;
    CHECK(close_to(gen::schedule(d, {}, 5, 999), diffusers_sigmas(d, 5, 999), 2e-6));
    /* 0 steps is the model's default. */
    CHECK_EQ(gen::schedule(d, {}, 0, 64).size(), (size_t)d.default_steps + 1);
}

/* A fixed schedule is taken as declared (with shift 1), then closed with 0. */
TEST(a_fixed_schedule_is_used_as_declared) {
    RadGeneratorDecl d = decl_base();
    const std::vector<float> s = gen::schedule(d, { 0.9, 0.5, 0.125 }, 10, 64);
    REQUIRE_EQ(s.size(), (size_t)4);
    CHECK_EQ(s[0], 0.9f); CHECK_EQ(s[1], 0.5f); CHECK_EQ(s[2], 0.125f); CHECK_EQ(s[3], 0.0f);
}

/* ================================================================== the sequence */
/* An h x w image at frame `pos`: every row at that frame, height and width centred on zero. */
TEST(image_rotary_positions_centre_the_grid_on_its_frame) {
    int32_t p[3 * 12];
    gen::image_rope(7, 3, 4, p, 12);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            const int k = r * 4 + c;
            CHECK_EQ(p[k], 7);
            CHECK_EQ(p[12 + k], r - 2);
            CHECK_EQ(p[24 + k], c - 2);
        }
}

/* The generator's sequence: the prompt past its fixed prefix, text rows at consecutive rotary
 * positions, each picture's slots replaced by its latent rows on a grid at one frame, and a
 * segment per run (bidirectional for a picture). Refused, by name, when a picture sits in the
 * prefix or its slots do not tile its latent grid. */
TEST(the_generators_sequence_lays_out_text_and_pictures) {
    gen::Branch b;
    b.tokens = { 90, 91, 92, 3, 4, 500, 500, 7, 8 };
    b.n_drop = 3;
    b.media = std::make_shared<mm::PromptMedia>();
    mm::RopeLayout::Run run;
    run.tok = 5; run.n = 2; run.item = 0;
    b.media->rope.runs.push_back(run);
    gen::Layout L;
    std::string why;
    REQUIRE_EQ(gen::build_layout(b, /*slot_rows=*/2, { { 2, 2 } }, &L, &why), (int)RAD_OK);
    REQUIRE_EQ(L.rows, (int64_t)8);
    const std::vector<int32_t> src = { 3, 4, -1, -2, -3, -4, 7, 8 };
    CHECK(L.src == src);
    const int32_t t[8] = { 0, 1, 2, 2, 2, 2, 4, 5 };
    const int32_t h[8] = { 0, 1, -1, -1, 0, 0, 4, 5 };
    const int32_t w[8] = { 0, 1, -1, 0, -1, 0, 4, 5 };
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(L.rope[(size_t)i], t[i]);
        CHECK_EQ(L.rope[(size_t)(8 + i)], h[i]);
        CHECK_EQ(L.rope[(size_t)(16 + i)], w[i]);
    }
    CHECK(L.seg == std::vector<int32_t>({ 0, 2, 6, 8 }));
    CHECK(L.bidi == std::vector<int32_t>({ 0, 1, 0 }));
    CHECK_EQ(L.next_pos, 6);

    CHECK_EQ(gen::build_layout(b, 2, { { 3, 2 } }, &L, &why), (int)RAD_E_INVAL);
    CHECK(has(why, "do not tile"));
    CHECK_EQ(gen::build_layout(b, 2, {}, &L, &why), (int)RAD_E_INVAL);
    CHECK(has(why, "were encoded for it"));
    b.media->rope.runs[0].tok = 2;
    CHECK_EQ(gen::build_layout(b, 2, { { 2, 2 } }, &L, &why), (int)RAD_E_INVAL);
    CHECK(has(why, "inside the prompt's fixed prefix"));
    b.media.reset();
    b.n_drop = 9;
    CHECK_EQ(gen::build_layout(b, 2, {}, &L, &why), (int)RAD_E_INVAL);
    CHECK(has(why, "no longer than its fixed prefix"));
}

/* ================================================================== the ticket */
/* Images pushed before the end are all delivered, the first finish wins, and an empty ticket
 * that has not ended times out rather than reporting the end. */
TEST(a_ticket_delivers_every_image_before_its_end) {
    gen::Ticket t;
    gen::Image im;
    bool timed_out = false;
    CHECK(!t.next(&im, 0, &timed_out));
    CHECK(timed_out);
    gen::Image a; a.index = 0;
    gen::Image b; b.index = 1;
    t.push(std::move(a));
    t.push(std::move(b));
    t.finish(RAD_OK);
    t.finish(RAD_E_STATE, "late");
    CHECK(!t.finished());
    CHECK(t.next(&im, 0, &timed_out) && im.index == 0);
    CHECK(t.next(&im, 0, &timed_out) && im.index == 1);
    CHECK(!t.next(&im, 0, &timed_out));
    CHECK(!timed_out);
    CHECK(t.finished());
    CHECK_EQ(t.status(), (int)RAD_OK);
}

/* ================================================================== the driver */
namespace {
struct GenFixture {
    Program prog;
    FakeKV  kv;
    gen::Driver drv;
    void build(int64_t max_tok = 2048, int64_t kv_blocks = 4096) {
        prog.generator = decl_base();
        prog.ctx.max_tok = max_tok;
        prog.ctx.max_seqs = 8;
        prog.kv_groups.emplace_back();
        KVGroupInfo g;
        g.name = "cond";
        g.decl.kind = RAD_KV_FULL;
        g.block_size = 16;
        prog.kv_groups.push_back(g);
        kv.setup(prog.kv_groups, 1 << 20, kv_blocks, 4);
    }
    int start() { return drv.init(prog, &kv, { gen::Driver::RankIO{} }); }
    std::unique_ptr<gen::Request> req(int w, int h, int n = 1) {
        auto r = std::make_unique<gen::Request>();
        r->width = w;
        r->height = h;
        r->n = n;
        r->pos.tokens = { 1, 2, 3, 4 };
        r->ticket = std::make_shared<gen::Ticket>();
        return r;
    }
};
}  /* namespace */

/* A request that can never run is refused at submission, naming the field: the size multiple,
 * the largest image, the rows a step with guidance would need, n, the picture count, the prompt's
 * length, the partial image count and an inpainting mask of the wrong size. */
TEST(the_generator_refuses_a_request_that_can_never_run_by_field) {
    GenFixture f;
    f.build(/*max_tok=*/1500);
    CHECK_OK(f.start());
    std::string why;
    auto refused = [&](std::unique_ptr<gen::Request> r, const char* says) {
        why.clear();
        CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
        const bool named = has(why, says);
        CHECK(named);
        if (!named) std::fprintf(stderr, "    refusal said: %s\n", why.c_str());
    };
    refused(f.req(100, 64), "whole multiples of 32");
    refused(f.req(1024, 1024), "the largest this model makes is 1024 latent tokens");
    auto g = f.req(512, 512);
    g->has_neg = true;
    g->neg.tokens = { 1 };
    refused(std::move(g), "with guidance needs 2048 rows");
    refused(f.req(64, 64, 0), "n must be 1..4");
    refused(f.req(64, 64, 5), "n must be 1..4");
    auto refs = f.req(64, 64);
    refs->refs.resize(3);
    refused(std::move(refs), "at most 2 pictures");
    auto longp = f.req(64, 64);
    longp->pos.tokens.assign(1501, 7);
    refused(std::move(longp), "the prompt is 1501 tokens");
    auto prev = f.req(64, 64);
    prev->previews = 4;
    refused(std::move(prev), "partial_images must be 0..3");
    auto inp = f.req(64, 64);
    inp->inpaint = true;
    inp->inp_src.w = 64; inp->inp_src.h = 64;
    inp->inp_mask.assign(3, 1.0f);
    refused(std::move(inp), "the mask's picture is not at the size being made");
    CHECK_OK(f.drv.submit(f.req(512, 512), &why));
    CHECK_EQ(f.drv.queued(), (int64_t)1);
}

/* A request's passes in order: its conditioning, then denoise steps over both images at once, a
 * preview of each image at the step its partial image falls on (decoded before the next step
 * overwrites the preview latent), then each final image, which ends the ticket and frees the
 * lanes for the next request. Image i carries seed + i. */
TEST(a_request_runs_its_conditioning_its_steps_its_previews_and_its_images_in_order) {
    GenFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(64, 64, 2);
    r->previews = 1;
    r->steps = 4;
    r->seed = 100;
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    std::string kinds;
    std::vector<gen::Image> got;
    for (int i = 0; i < 20 && !t->finished(); ++i) {
        const RadBatch* b = f.drv.next(i);
        REQUIRE(b != nullptr);
        kinds += b->gen == RAD_GEN_PASS_COND ? 'C' : b->gen == RAD_GEN_PASS_DENOISE ? 'S'
                                                   : b->gen == RAD_GEN_PASS_DECODE ? 'D' : '?';
        CHECK_OK(f.drv.done());
        gen::Image im;
        bool to = false;
        while (t->next(&im, 0, &to)) got.push_back(std::move(im));
    }
    CHECK_EQ(kinds, std::string("CSSDDSSDD"));
    REQUIRE_EQ(got.size(), (size_t)4);
    for (int i = 0; i < 2; ++i) {
        CHECK(got[(size_t)i].preview);
        CHECK_EQ(got[(size_t)i].index, i);
        CHECK(!got[(size_t)(2 + i)].preview);
        CHECK_EQ(got[(size_t)(2 + i)].index, i);
        CHECK_EQ(got[(size_t)(2 + i)].seed, (uint64_t)(100 + i));
    }
    CHECK_EQ(t->status(), (int)RAD_OK);
    /* Every lane is free again: a request for all four images is admitted at once. */
    CHECK_OK(f.drv.submit(f.req(64, 64, 4), &why));
    const RadBatch* b = f.drv.next(20);
    CHECK(b != nullptr && b->gen == RAD_GEN_PASS_COND);
    CHECK_EQ(f.drv.running(), (int64_t)1);
    CHECK_EQ(f.drv.queued(), (int64_t)0);
}

/* Admission is in order: a request that cannot get lanes for every image blocks the requests
 * behind it, even one that would fit, so a large request is not starved by small ones. A
 * shutdown answers every ticket, queued or running, with the status that stopped the engine. */
TEST(the_generator_admits_in_order_and_shutdown_answers_every_ticket) {
    GenFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto a = f.req(64, 64, 3), b = f.req(64, 64, 2), c = f.req(64, 64, 1);
    auto ta = a->ticket, tb = b->ticket, tc = c->ticket;
    CHECK_OK(f.drv.submit(std::move(a), &why));
    CHECK_OK(f.drv.submit(std::move(b), &why));
    CHECK_OK(f.drv.submit(std::move(c), &why));
    const RadBatch* p = f.drv.next(0);                /* A's conditioning */
    CHECK(p != nullptr);
    CHECK_EQ(f.drv.running(), (int64_t)1);            /* A holds 3 of 4 lanes */
    CHECK_EQ(f.drv.queued(), (int64_t)2);             /* B needs 2; C, behind it, waits too */
    CHECK_EQ(ta->steps, (int32_t)4);                  /* the default schedule */
    CHECK_EQ(ta->out_tokens, (int64_t)(4 * 4 * 3));   /* 64x64 at scale 16, three images */

    f.drv.shutdown("the engine stopped", RAD_E_UNSUPPORTED);
    for (const auto& t : { ta, tb, tc }) {
        gen::Image im;
        bool to = false;
        CHECK(!t->next(&im, 0, &to));
        CHECK(t->finished());
        CHECK_EQ(t->status(), (int)RAD_E_UNSUPPORTED);
        CHECK_EQ(t->why(), std::string("the engine stopped"));
    }
    CHECK_EQ(f.drv.submit(f.req(64, 64), &why), (int)RAD_E_UNSUPPORTED);
    CHECK(f.drv.next(1) == nullptr);
}

/* The joint layout (RAD_GEN_LAYOUT_JOINT_STEP): no KV group; the conditioning's prompt padded to
 * the declared multiple (padding at position 1, valid_cu bounding the prompt), the reference
 * picture's tokens after it with its own l and scaled positions, and stored for every lane of the
 * request. A denoising entry is that conditioning, read from its lane and branch's store rows,
 * then the image, both in the pass order and mode by mode. */
TEST(the_joint_layout_carries_the_conditioning_in_every_step) {
    Program prog;
    prog.generator = decl_base();
    RadGeneratorDecl& d = prog.generator;
    d.layout = RAD_GEN_LAYOUT_JOINT_STEP;
    d.cond_rows = 64;
    d.pad_multiple = 8;
    d.pad_token = 99;
    d.rope_hw_scale = 0.5;
    d.rope_text_l = 1;
    d.ref_area = 32 * 32;
    prog.ctx.max_tok = 2048;
    prog.ctx.max_seqs = 8;
    prog.kv_groups.emplace_back();
    gen::Driver drv;
    REQUIRE_EQ(drv.init(prog, nullptr, { gen::Driver::RankIO{} }), (int)RAD_OK);
    auto r = std::make_unique<gen::Request>();
    r->width = 64;
    r->height = 32;
    r->n = 2;
    r->pos.tokens = { 1, 2, 3, 4, 5 };
    r->pos.n_drop = 1;
    r->has_neg = true;
    r->neg.tokens = { 1, 6 };
    r->neg.n_drop = 1;
    r->cfg = 3.0f;
    gen::Pixels px;
    px.w = 32;
    px.h = 32;
    px.ch = 4;
    px.bf16.assign(32 * 32 * 4, 0);
    r->refs.push_back(px);
    r->ticket = std::make_shared<gen::Ticket>();
    std::string why;
    REQUIRE_EQ(drv.submit(std::move(r), &why), (int)RAD_OK);

    /* The positive conditioning: 4 text rows padded to 8, then the picture's 2 x 2 tokens. */
    const RadBatch* b = drv.next(0);
    REQUIRE(b != nullptr);
    REQUIRE_EQ(b->gen, (int32_t)RAD_GEN_PASS_COND);
    const RadGenBatch* G = b->gen_b;
    CHECK_EQ(b->n_kv_groups, (int32_t)0);
    CHECK_EQ(b->n_tok, (int64_t)9);                   /* five tokens and four padding */
    CHECK_EQ(G->rows, (int64_t)12);                   /* 8 text rows and 4 picture tokens */
    CHECK_EQ(G->n_pad, (int32_t)4);
    CHECK_EQ(G->branch, (int32_t)0);
    CHECK_EQ(G->valid_cu[1], (int32_t)5);
    CHECK_EQ(b->token_ids[8], (int32_t)99);
    CHECK_EQ(b->positions[8], (int32_t)1);
    CHECK_EQ(G->n_lane, (int64_t)2);
    CHECK_EQ(G->src[0], (int32_t)1);                  /* the prompt row after the drop */
    CHECK_EQ(G->src[8], (int32_t)-1);                 /* the picture's first latent row */
    CHECK_EQ(G->rope_f[3 * 12 + 2], 2.0f);            /* text row 2: l = 2 */
    CHECK_EQ(G->rope_f[3 * 12 + 9], -1.0f);           /* the picture: l = -1 */
    CHECK_EQ(G->rope_f[2 * 12 + 9], 0.5f);            /* its token (0, 1): w = 0.5 */
    CHECK_OK(drv.done());
    b = drv.next(1);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen_b->branch, (int32_t)1);
    CHECK_EQ(b->gen_b->rows, (int64_t)12);            /* one token padded to 8, the picture */
    CHECK_OK(drv.done());

    /* A step: four entries (two images, each with its negative), 12 + 8 rows each. */
    b = drv.next(2);
    REQUIRE(b != nullptr);
    REQUIRE_EQ(b->gen, (int32_t)RAD_GEN_PASS_DENOISE);
    G = b->gen_b;
    REQUIRE_EQ(b->n_seq, (int64_t)4);
    CHECK_EQ(b->n_tok, (int64_t)80);
    CHECK_EQ(G->n_img, (int64_t)32);
    CHECK_EQ(b->cu_seqlens[1], (int32_t)20);
    const int32_t lane0 = G->lane_dh[0];
    CHECK_EQ(G->src[0], (int32_t)(2 * lane0 * 64));             /* lane 0's positive row 0 */
    CHECK_EQ(G->src[20], (int32_t)((2 * lane0 + 1) * 64));      /* its negative row 0 */
    CHECK_EQ(G->src[12], (int32_t)-1);                          /* the first image row */
    CHECK_EQ(G->img_rows[0], (int32_t)12);
    CHECK_EQ(G->img_cu[1], (int32_t)8);
    CHECK_EQ(G->mode_cu[1], (int32_t)8);                        /* entry 0: text 0..8 */
    CHECK_EQ(G->mode_cu[2], (int32_t)12);                       /* pictures 8..12 */
    CHECK_EQ(G->mode_cu[3], (int32_t)20);                       /* image to 20 */
    CHECK_EQ(G->n_mode_h[0], (int64_t)32);
    CHECK_EQ(G->n_mode_h[1], (int64_t)16);
    CHECK_EQ(G->n_mode_h[2], (int64_t)32);
    CHECK_EQ(G->mode_rows[32], (int32_t)8);                     /* the first picture row */
    CHECK_EQ(G->mode_ccu[1 * 5 + 1], (int32_t)4);               /* pictures: entry 1 from 4 */
    CHECK_EQ(G->mode_rope_f[4 * 32 + 3 * 16 + 0], -1.0f);       /* the pictures' l plane */
    CHECK_EQ(G->rope_f[1 * 80 + 12 + 4], 0.5f);                 /* image token (1, 0): h = 0.5 */
    CHECK_OK(drv.done());
}

RAD_TEST_MAIN()
