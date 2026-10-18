/* video_gen_test.cpp -- the video generator's pass driver (core/gen/video_driver.h) and the
 * request shape and schedule it shares with the server (core/gen/video.h). The plugin's side of
 * each pass is played by the test: it writes the decode passes' outputs where the driver reads
 * them back, and a fake rad_arch_video_positions lays the rows out.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/video_driver.h"

#include <cmath>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* Two frames a chunk and one more at the end, a latent frame a chunk and one more, two latent
 * frames a decode; 32-pixel sides to 64 x 64; 40 soundtrack latents a second at 800 Hz. */
RadVideoGenDecl decl_small() {
    RadVideoGenDecl d{};
    d.max_text = 8;
    d.fps_num = 4;
    d.fps_den = 1;
    d.frames_per_chunk = 2;
    d.frames_tail = 1;
    d.latents_per_chunk = 1;
    d.latents_tail = 1;
    d.dec_latents = 2;
    d.min_chunks = 1;
    d.max_chunks = 3;
    d.latent_scale = 16;
    d.latent_channels = 4;
    d.patch = 2;
    d.size_multiple = 32;
    d.min_side = 32;
    d.max_area = 64 * 64;
    d.max_aspect = 2;
    d.default_width = 64;
    d.default_height = 32;
    d.default_chunks = 1;
    d.sample_rate = 800;
    d.audio_rate = 40;
    d.audio_channels = 2;
    d.audio_latent_channels = 4;
    d.default_steps = 3;
    d.max_steps = 5;
    d.shift_video = 12.0;
    d.shift_audio = 3.0;
    d.max_lanes = 2;
    d.frames_out = 1;
    d.audio_out = 2;
    return d;
}

/* The packed layout's rows, each position the row's index; a decode pass, ten rows a frame. */
int64_t fake_positions(int which, const RadVideoGeom* g, float* pos, int64_t cap) {
    const int64_t rows = g->n_text + 2 * (int64_t)g->n_audio +
                         (int64_t)g->lat_frames * (g->lat_h / 2) * (g->lat_w / 2);
    if (which == RAD_VPOS_DECODE) return 10 * (int64_t)g->frames;
    if (pos) {
        if (cap < rows) return RAD_E_INVAL;
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t i = 0; i < rows; ++i) pos[c * rows + i] = (float)(c * 1000 + i);
    }
    return rows;
}

struct VideoFixture {
    Program prog;
    gen::VideoDriver drv;
    void build() {
        prog.video_gen = decl_small();
        prog.vgen_str.name = "test";
        prog.vgen_str.tmpl = "{prompt}";
        prog.ctx.max_tok = 256;
        prog.ctx.max_seqs = 1;
    }
    int start() { return drv.init(prog, fake_positions, { gen::VideoDriver::RankIO{} }); }
    std::unique_ptr<gen::VideoRequest> req(int32_t chunks = 2, int32_t steps = 0) {
        auto r = std::make_unique<gen::VideoRequest>();
        r->tokens = { 1, 2, 3 };
        r->width = 64;
        r->height = 32;
        r->chunks = chunks;
        r->steps = steps;
        r->seed = 7;
        r->ticket = std::make_shared<gen::VideoTicket>();
        return r;
    }
    /* The plugin's answer to a decode pass: each byte or sample its index plus `base`. */
    void answer(int base) {
        for (const gen::IPassDriver::Readback& rb : drv.readbacks()) {
            if (rb.buf == prog.video_gen.frames_out) {
                uint8_t* x = static_cast<uint8_t*>(rb.host);
                for (int64_t i = 0; i < rb.bytes; ++i) x[i] = (uint8_t)(base + i);
            } else {
                float* x = static_cast<float*>(rb.host);
                for (int64_t i = 0; i < rb.bytes / 4; ++i) x[i] = (float)(base + i);
            }
        }
    }
};

char kind_of(int gen) {
    switch (gen) {
    case RAD_GEN_PASS_VCOND:   return 'C';
    case RAD_GEN_PASS_VSTEP:   return 'S';
    case RAD_GEN_PASS_ADECODE: return 'A';
    case RAD_GEN_PASS_VDECODE: return 'D';
    case RAD_GEN_PASS_VUPSCALE: return 'U';
    default:                   return '?';
    }
}

}  /* namespace */

/* A request's shape: frames and latent frames from its chunks, the soundtrack's latents its length
 * at the soundtrack's rate (ties to even), the rows of a step. */
TEST(a_request_shape_follows_the_declaration) {
    const RadVideoGenDecl d = decl_small();
    const gen::VideoShape s = gen::video_shape(d, 64, 32, 2);
    CHECK_EQ(s.frames, 5);                 /* 2 * 2 + 1 */
    CHECK_EQ(s.lat_frames, 3);             /* 1 * 2 + 1 */
    CHECK_EQ(s.lat_h, 2);
    CHECK_EQ(s.lat_w, 4);
    CHECK_EQ(s.n_audio, 50);               /* 5 frames at 4 a second, 40 latents a second */
    CHECK_EQ(s.video_rows, (int64_t)3 * 1 * 2);
    CHECK_EQ(s.audio_rows, (int64_t)100);
    /* MiniMax-H3's: 124 frames are 206.67 soundtrack latents, which round to 207. */
    RadVideoGenDecl h = d;
    h.fps_num = 24;
    h.frames_per_chunk = 17;
    h.frames_tail = 5;
    h.latents_per_chunk = 5;
    h.latents_tail = 2;
    const gen::VideoShape hs = gen::video_shape(h, 1344, 768, 7);
    CHECK_EQ(hs.frames, 124);
    CHECK_EQ(hs.lat_frames, 37);
    CHECK_EQ(hs.n_audio, 207);
    CHECK_EQ(hs.video_rows, (int64_t)37 * 24 * 42);
}

/* The schedule: `steps` grid points from sigma 1 to 0, each shifted, one evaluation fewer; the
 * video's and the soundtrack's shifts give two schedules of the same length. */
TEST(the_schedule_counts_grid_points_and_shifts_each_modality) {
    const RadVideoGenDecl d = decl_small();
    const gen::VideoSchedule s = gen::video_schedule(d, 50);
    REQUIRE_EQ(s.sig_v.size(), (size_t)50);
    REQUIRE_EQ(s.sig_a.size(), (size_t)50);
    CHECK_EQ(s.steps, 49);
    CHECK_EQ(s.sig_v.front(), 1.0f);
    CHECK_EQ(s.sig_v.back(), 0.0f);
    CHECK_EQ(s.sig_a.back(), 0.0f);
    for (size_t i = 1; i < s.sig_v.size(); ++i) {
        CHECK(s.sig_v[i] < s.sig_v[i - 1]);
        CHECK(s.sig_a[i] < s.sig_a[i - 1]);
    }
    /* A grid point past the middle is linspace's from the end: 1 - 30/49 there. */
    const double b = 1.0 - 30.0 / 49.0;
    CHECK(std::fabs(s.sig_v[30] - 12.0 * b / (1.0 + 11.0 * b)) < 1e-6);
    CHECK(std::fabs(s.sig_a[30] - 3.0 * b / (1.0 + 2.0 * b)) < 1e-6);
    CHECK_EQ(gen::video_schedule(d, 2).steps, 1);
}

/* A trained schedule (a distilled checkpoint's): its rungs in f32, each modality's shift applied
 * as the reference's tensor arithmetic does (f32 throughout), then the terminal 0; one evaluation a
 * rung, and no other step count. */
TEST(a_trained_schedule_shifts_its_rungs_and_fixes_the_step_count) {
    static const double kRungs[] = { 0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125 };
    RadVideoGenDecl d = decl_small();
    d.shift_video = 10.0;
    d.n_rungs = 8;
    d.rungs = kRungs;
    d.default_steps = d.max_steps = 9;
    const gen::VideoSchedule s = gen::video_schedule(d, 9);
    REQUIRE_EQ(s.sig_v.size(), (size_t)9);
    REQUIRE_EQ(s.sig_a.size(), (size_t)9);
    CHECK_EQ(s.steps, 8);
    CHECK_EQ(s.sig_v.back(), 0.0f);
    CHECK_EQ(s.sig_a.back(), 0.0f);
    for (int i = 0; i < 8; ++i) {
        const float b = (float)kRungs[i];
        CHECK_EQ(s.sig_v[(size_t)i], (10.0f * b) / (1.0f + 9.0f * b));
        CHECK_EQ(s.sig_a[(size_t)i], (3.0f * b) / (1.0f + 2.0f * b));
    }
    CHECK_EQ(gen::video_schedule(d, 8).steps, 0);
    CHECK_EQ(gen::video_schedule(d, 50).steps, 0);

    VideoFixture f;
    f.build();
    f.prog.video_gen = d;
    f.prog.video_gen.rungs = nullptr;
    f.prog.vgen_str.rungs.assign(kRungs, kRungs + 8);
    CHECK_OK(f.start());
    std::string why;
    CHECK_EQ(f.drv.submit(f.req(1, 5), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "num_inference_steps must be 9"));
    why.clear();
    CHECK_OK(f.drv.submit(f.req(1, 0), &why));
}

/* One request's passes: its prompt with the whole schedule, a step per evaluation (the first
 * naming itself the first, each with its two timesteps and sigmas and the packed rows' positions),
 * the soundtrack, then the chunks in order, the last one longer. The ticket carries the steps,
 * the soundtrack and the frames, in that order, then the end. */
TEST(a_request_runs_prompt_steps_soundtrack_and_chunks_in_order) {
    VideoFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(2);
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    CHECK_EQ(t->frames, 5);
    CHECK_EQ(t->steps, 2);
    std::string kinds;
    std::vector<int32_t> frames_out;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        const RadVideoGenBatch* V = b->vgen_b;
        REQUIRE(V != nullptr);
        kinds += kind_of(b->gen);
        CHECK_EQ(V->geom.n_text, 3);
        CHECK_EQ(V->geom.frames, 5);
        if (b->gen == RAD_GEN_PASS_VCOND) {
            CHECK_EQ(b->n_tok, (int64_t)3);
            CHECK_EQ(V->n_steps, 2);
        } else if (b->gen == RAD_GEN_PASS_VSTEP) {
            CHECK_EQ(b->n_tok, (int64_t)(3 + 100 + 6));
        } else if (b->gen == RAD_GEN_PASS_VDECODE) {
            CHECK_EQ(b->n_tok, (int64_t)50);
            frames_out.push_back(V->n_frames_out);
        }
        f.answer(10 * i);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("CSSADD"));
    CHECK(frames_out == std::vector<int32_t>({ 2, 3 }));
    std::string ev;
    gen::VideoEvent e;
    bool to = false;
    int32_t next_frame = 0;
    while (t->next(&e, 0, &to)) {
        ev += e.kind == gen::VideoEvent::STEP ? 's' : e.kind == gen::VideoEvent::AUDIO ? 'a' : 'f';
        if (e.kind == gen::VideoEvent::AUDIO) {
            CHECK_EQ(e.samples.size(), (size_t)(50 * 20 * 2));
        }
        if (e.kind == gen::VideoEvent::FRAMES) {
            CHECK_EQ(e.first, next_frame);
            CHECK_EQ(e.rgb.size(), (size_t)e.n * 64 * 32 * 3);
            next_frame += e.n;
        }
    }
    CHECK_EQ(ev, std::string("ssaff"));
    CHECK_EQ(next_frame, 5);
    CHECK_EQ(t->status(), (int)RAD_OK);
}

/* A second request is conditioned while the first waits for its first step, and denoised once the
 * first has finished: requests are denoised one after another. */
TEST(a_second_request_is_conditioned_early_and_denoised_after_the_first) {
    VideoFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    CHECK_OK(f.drv.submit(f.req(1, 2), &why));
    CHECK_OK(f.drv.submit(f.req(1, 2), &why));
    std::string kinds;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        kinds += kind_of(b->gen);
        f.answer(0);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("CCSADSAD"));
}

/* Latent upscaling's small size: each side over the factor, rounded half up to the size multiple,
 * at least one multiple and at most the side. */
TEST(the_small_size_is_each_side_over_the_factor_at_the_multiple) {
    const RadVideoGenDecl d = decl_small();
    int32_t w = 0, h = 0;
    gen::video_small_size(d, 1344, 768, 2.0, &w, &h);
    CHECK_EQ(w, 672);
    CHECK_EQ(h, 384);
    gen::video_small_size(d, 1344, 768, 4.0, &w, &h);
    CHECK_EQ(w, 352);                      /* 336 is 10.5 multiples, rounded up */
    CHECK_EQ(h, 192);
    gen::video_small_size(d, 1280, 768, 4.0, &w, &h);
    CHECK_EQ(w, 320);
    CHECK_EQ(h, 192);
    gen::video_small_size(d, 64, 32, 4.0, &w, &h);
    CHECK_EQ(w, 32);                       /* never below one multiple */
    CHECK_EQ(h, 32);
    gen::video_small_size(d, 64, 64, 1.2, &w, &h);
    CHECK_EQ(w, 64);                       /* 1.67 multiples round back to the side */
    CHECK_EQ(h, 64);
}

/* An upscaled request: its first evaluations at the small size, the last of them stepping to the
 * clean estimate, then the upscale pass with both geometries, the factor and the first full-size
 * step's sigmas, then the full-size steps and the decodes at the full size. */
TEST(an_upscaled_request_runs_small_steps_then_the_upscale_then_full_steps) {
    VideoFixture f;
    f.build();
    f.prog.video_gen.upscale_max = 4.0;
    f.prog.video_gen.upscale_refine = 1;
    CHECK_OK(f.start());
    const gen::VideoSchedule sc = gen::video_schedule(f.prog.video_gen, 4);
    REQUIRE_EQ(sc.steps, 3);
    std::string why;
    auto r = f.req(1, 4);
    r->width = 64;
    r->height = 64;
    r->upscale = 2.0;
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    CHECK_EQ(t->small_width, 32);
    CHECK_EQ(t->small_height, 32);
    CHECK_EQ(t->refine, 1);
    CHECK_EQ(t->steps, 3);
    /* 3 frames: 2 latent frames, 30 soundtrack latents a channel. Small: 1 x 1 patches a frame;
     * full: 2 x 2. */
    const int64_t small_rows = 3 + 60 + 2 * 1, full_rows = 3 + 60 + 2 * 4;
    std::string kinds;
    int steps_seen = 0;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        const RadVideoGenBatch* V = b->vgen_b;
        REQUIRE(V != nullptr);
        kinds += kind_of(b->gen);
        if (b->gen == RAD_GEN_PASS_VSTEP) {
            const bool small = steps_seen < 2;
            CHECK_EQ(b->n_tok, small ? small_rows : full_rows);
            CHECK_EQ(V->geom.lat_h, small ? 2 : 4);
            CHECK_EQ(V->geom.lat_w, small ? 2 : 4);
            CHECK_EQ(V->geom_d[RAD_VGEOM_LAT_H], small ? 2 : 4);
            CHECK_EQ(V->seg_cu[3], (int32_t)(small ? small_rows : full_rows));
            CHECK_EQ(V->sched[1], sc.sig_v[(size_t)steps_seen]);
            /* The last small step lands on the clean estimate. */
            CHECK_EQ(V->sched[2], steps_seen == 1 ? 0.0f : sc.sig_v[(size_t)steps_seen + 1]);
            CHECK_EQ(V->sched[5], steps_seen == 1 ? 0.0f : sc.sig_a[(size_t)steps_seen + 1]);
            ++steps_seen;
        } else if (b->gen == RAD_GEN_PASS_VUPSCALE) {
            CHECK_EQ(b->n_tok, (int64_t)0);
            CHECK_EQ(V->geom.lat_h, 4);
            CHECK_EQ(V->geom_from.lat_h, 2);
            CHECK_EQ(V->geom_from.lat_w, 2);
            CHECK_EQ(V->upscale[0], 2.0f);
            CHECK_EQ(V->sched[1], sc.sig_v[2]);
            CHECK_EQ(V->sched[4], sc.sig_a[2]);
            const int32_t* g = V->geom_d;
            CHECK(std::vector<int32_t>(g + RAD_VGEOM_UP_SRC, g + RAD_VGEOM_UP_SRC + 14) ==
                  std::vector<int32_t>({ 2, 2, 2, 4, 4, 2, 4, 4, 2, 1, 1, 2, 2, 2 }));
        } else if (b->gen == RAD_GEN_PASS_VDECODE) {
            CHECK_EQ(V->geom.lat_h, 4);
        }
        f.answer(0);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("CSSUSAD"));
    CHECK(t->t_small_ns > 0 && t->t_upscaled_ns >= t->t_small_ns);
    CHECK_EQ(t->status(), (int)RAD_OK);
}

/* Upscaling is refused when the model has none, past its largest factor, when the factor leaves
 * the size as it is, and when the refinement leaves no small step or takes none. */
TEST(an_upscale_that_cannot_run_is_refused_at_submit) {
    VideoFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1, 4);
    r->upscale = 2.0;
    CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "no latent upscaler"));

    VideoFixture g;
    g.build();
    g.prog.video_gen.upscale_max = 4.0;
    g.prog.video_gen.upscale_refine = 1;
    CHECK_OK(g.start());
    auto refused = [&](double factor, int32_t refine, const char* what) {
        auto q = g.req(1, 4);
        q->width = 64;
        q->height = 64;
        q->upscale = factor;
        q->refine = refine;
        why.clear();
        CHECK_EQ(g.drv.submit(std::move(q), &why), (int)RAD_E_INVAL);
        CHECK(has(why, what));
    };
    refused(5.0, 0, "at most 4");
    refused(1.2, 0, "leaves 64x64");
    refused(2.0, 3, "1..2");
    auto ok = g.req(1, 4);
    ok->width = 64;
    ok->height = 64;
    ok->upscale = 2.0;
    ok->refine = 2;
    CHECK_OK(g.drv.submit(std::move(ok), &why));
}

/* What can never run is refused at submit: an empty or long prompt, a size off the multiple,
 * below the least side, over the area or the aspect, a length outside the chunks, steps outside
 * 2..max. */
TEST(a_request_that_cannot_run_is_refused_at_submit) {
    VideoFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto refused = [&](std::unique_ptr<gen::VideoRequest> r, const char* what) {
        why.clear();
        CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
        CHECK(has(why, what));
    };
    auto r = f.req();
    r->tokens.clear();
    refused(std::move(r), "prompt");
    r = f.req();
    r->tokens.assign(9, 1);
    refused(std::move(r), "1..8");
    r = f.req();
    r->width = 48;
    refused(std::move(r), "size");
    r = f.req();
    r->width = 96;
    r->height = 32;
    refused(std::move(r), "size");
    r = f.req();
    r->width = 64;
    r->height = 96;
    refused(std::move(r), "size");
    refused(f.req(4), "chunks");
    refused(f.req(0), "chunks");
    refused(f.req(1, 6), "num_inference_steps");
    refused(f.req(1, 1), "num_inference_steps");
}

RAD_TEST_MAIN()
