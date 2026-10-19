/* video_gen_test.cpp -- the video generator's pass driver (core/gen/video_driver.h) and the
 * request shape and schedule it shares with the server (core/gen/video.h). The plugin's side of
 * each pass is played by the test: it writes the decode passes' outputs where the driver reads
 * them back, and a fake rad_arch_video_positions lays the rows out.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/video_driver.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstring>
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

/* The soundtrack's rows a latent in fake_positions' layout: the declaration's audio_streams, or its
 * two channels. */
int g_audio_streams = 2;

/* The packed layout's rows, each position the row's index; a decode pass, ten rows a frame. */
int64_t fake_positions(int which, const RadVideoGeom* g, float* pos, int64_t cap) {
    const int64_t rows = g->n_text + g_audio_streams * (int64_t)g->n_audio +
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
 * 2..max. A length of 0 chunks is none named. */
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
    refused(f.req(1, 6), "num_inference_steps");
    refused(f.req(1, 1), "num_inference_steps");
    /* No length named is the model's default. */
    auto none = f.req(0);
    gen::VideoTicket* t = none->ticket.get();
    CHECK_OK(f.drv.submit(std::move(none), &why));
    CHECK_EQ(t->frames, 3);
}

/* Decoders (RadVideoGenDecl::decoders): a request's index reaches every VDECODE pass; an index past
 * the list, or a frame over the decoder's area, is refused at submit. */
TEST(the_decoder_a_request_names_reaches_its_decode_passes) {
    VideoFixture f;
    f.build();
    f.prog.vgen_str.decoders = { "vae", "diffusion" };
    f.prog.vgen_str.decoder_max_area = { 64 * 64, 32 * 32 };
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1);
    r->decoder = 1;
    CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);   /* 64 x 32 over 32 x 32 */
    CHECK(why.find("diffusion") != std::string::npos);
    r = f.req(1);
    r->decoder = 2;
    CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
    r = f.req(1);
    r->decoder = 1;
    r->width = 32;
    r->height = 32;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    int decodes = 0;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_VDECODE) {
            CHECK_EQ(b->vgen_b->decoder, 1);
            ++decodes;
        }
        f.answer(0);
        CHECK_OK(f.drv.done());
    }
    CHECK(decodes > 0);
}

/* A length head (RadVideoGenDecl::duration_out): a request that names no length is shaped at the
 * default until its VCOND pass, whose prediction (ln seconds, read back) the driver turns into
 * chunks -- clamped to the declared seconds, rounded half to even to frames, floored to the chunk
 * grid and lifted a chunk under the minimum -- then held to the chunks the model makes. */
TEST(a_length_head_sets_the_length_after_the_prompt_pass) {
    struct Case { double sec; int32_t frames; };
    /* 4 fps, 2 frames a chunk + 1, 1..3 chunks, 0.5..2 s (2..8 frames): 1.3 s is 5.2 frames, 5;
     * 1.125 s is 4.5, which rounds to 4 (half to even) and floors to 3; 0.1 s is held to 2
     * frames, which floor to 1 and are lifted to 3; 9 s is held to 8 frames, 7 on the grid. */
    const Case cases[] = { { 1.3, 5 }, { 1.125, 3 }, { 0.1, 3 }, { 9.0, 7 } };
    for (const Case& k : cases) {
        VideoFixture f;
        f.build();
        f.prog.video_gen.duration_out = 3;
        f.prog.video_gen.duration_min = 0.5;
        f.prog.video_gen.duration_max = 2.0;
        f.prog.video_gen.duration_dtype = RAD_F32;
        CHECK_OK(f.start());
        std::string why;
        auto r = f.req(0);
        std::shared_ptr<gen::VideoTicket> t = r->ticket;
        CHECK_OK(f.drv.submit(std::move(r), &why));
        CHECK_EQ(t->frames, 3);                       /* the default, one chunk */
        const RadBatch* b = f.drv.next(0);
        REQUIRE(b != nullptr);
        CHECK_EQ(b->gen, (int)RAD_GEN_PASS_VCOND);
        REQUIRE_EQ(f.drv.readbacks().size(), (size_t)1);
        CHECK_EQ(f.drv.readbacks()[0].buf, (rad_buf)3);
        *static_cast<float*>(f.drv.readbacks()[0].host) = (float)std::log(k.sec);
        CHECK_OK(f.drv.done());
        CHECK_EQ(t->frames, k.frames);
        gen::VideoEvent e;
        bool to = false;
        REQUIRE(t->next(&e, 0, &to));
        CHECK_EQ(e.kind, (int)gen::VideoEvent::LENGTH);
        CHECK_EQ(e.n, k.frames);
        /* The steps run at that length. */
        b = f.drv.next(1);
        REQUIRE(b != nullptr);
        CHECK_EQ(b->gen, (int)RAD_GEN_PASS_VSTEP);
        CHECK_EQ(b->vgen_b->geom.frames, k.frames);
    }
}

/* ---- form 1 (LTX-2's FlowMatchEulerDiscrete): evaluations, not grid points */

static uint32_t bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

/* The dev schedule bit for bit against diffusers' scheduler (numpy 2.5): 30 evaluations of a
 * 6144-token video shift by mu 2.7833 and stretch to end at 0.1; and 7 of a 1500-token one. */
TEST(form_one_is_ltx2s_dynamically_shifted_schedule_bit_for_bit) {
    RadVideoGenDecl d = decl_small();
    d.sched_kind = 1;
    d.dyn_shift = 1;
    d.base_shift = 0.95;
    d.max_shift = 2.05;
    d.base_seq = 1024;
    d.max_seq = 4096;
    d.shift_terminal = 0.1;
    const gen::VideoSchedule s = gen::video_schedule(d, 30, 6144);
    REQUIRE_EQ(s.sig_v.size(), (size_t)31);
    CHECK_EQ(s.steps, 30);
    const std::pair<int, uint32_t> want[] = {
        { 0, 0x3f800000u }, { 1, 0x3f7f3c87u }, { 2, 0x3f7e6c02u }, { 15, 0x3f6b19eeu },
        { 27, 0x3eff5f82u }, { 28, 0x3eb2f452u }, { 29, 0x3dccccd0u }, { 30, 0u },
    };
    for (const auto& w : want) CHECK_EQ(bits(s.sig_v[(size_t)w.first]), w.second);
    CHECK(s.sig_a == s.sig_v);
    const gen::VideoSchedule s7 = gen::video_schedule(d, 7, 1500);
    const uint32_t want7[] = { 0x3f800000u, 0x3f6e0d50u, 0x3f57d30eu, 0x3f3b945eu, 0x3f167dc8u,
                               0x3ec74586u, 0x3dccccd0u, 0u };
    REQUIRE_EQ(s7.sig_v.size(), (size_t)8);
    for (int i = 0; i < 8; ++i) CHECK_EQ(bits(s7.sig_v[(size_t)i]), want7[i]);
    /* One evaluation is a schedule of form 1. */
    CHECK_EQ(gen::video_schedule(d, 1, 6144).steps, 1);
}

static const double kDistilled[] = { 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725,
                                     0.421875 };
static const double kStage2[] = { 0.909375, 0.725, 0.421875 };

RadVideoGenDecl decl_distilled() {
    RadVideoGenDecl d = decl_small();
    d.sched_kind = 1;
    d.n_rungs = 8;
    d.rungs = kDistilled;
    d.default_steps = d.max_steps = 8;
    d.upscale_max = 2.0;
    d.n_refine_rungs = 3;
    d.refine_rungs = kStage2;
    d.upscale_refine = 3;
    return d;
}

/* A trained schedule of form 1 is its rungs as given, and with refinement the second stage's rungs
 * follow the first's in place of the 0. */
TEST(form_one_rungs_are_used_as_given_and_the_refinement_follows_them) {
    const RadVideoGenDecl d = decl_distilled();
    const gen::VideoSchedule s = gen::video_schedule(d, 8);
    REQUIRE_EQ(s.sig_v.size(), (size_t)9);
    CHECK_EQ(s.steps, 8);
    for (int i = 0; i < 8; ++i) CHECK_EQ(s.sig_v[(size_t)i], (float)kDistilled[i]);
    CHECK_EQ(s.sig_v[8], 0.0f);
    CHECK_EQ(gen::video_schedule(d, 9).steps, 0);
    const gen::VideoSchedule r = gen::video_schedule(d, 8, 0, true);
    REQUIRE_EQ(r.sig_v.size(), (size_t)12);
    CHECK_EQ(r.steps, 11);
    CHECK_EQ(r.sig_v[8], (float)kStage2[0]);
    CHECK_EQ(r.sig_v[10], (float)kStage2[2]);
    CHECK_EQ(r.sig_v[11], 0.0f);
}

/* Two stages: every first-stage rung at the small size, the last to the clean estimate; the
 * upscale re-noises to the second stage's first sigma; the second stage's rungs at the full size.
 * The request's refinement is the second stage's length or nothing. */
TEST(a_two_stage_request_runs_the_second_stages_own_rungs_after_the_upscale) {
    VideoFixture f;
    f.build();
    f.prog.video_gen = decl_distilled();
    f.prog.video_gen.rungs = nullptr;
    f.prog.video_gen.refine_rungs = nullptr;
    f.prog.vgen_str.rungs.assign(kDistilled, kDistilled + 8);
    f.prog.vgen_str.refine_rungs.assign(kStage2, kStage2 + 3);
    CHECK_OK(f.start());
    std::string why;
    auto bad = f.req(1, 0);
    bad->width = bad->height = 64;
    bad->upscale = 2.0;
    bad->refine = 2;
    CHECK_EQ(f.drv.submit(std::move(bad), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "upscale_refine_steps must be 3"));
    auto r = f.req(1, 0);
    r->width = r->height = 64;
    r->upscale = 2.0;
    auto t = r->ticket;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    CHECK_EQ(t->steps, 11);
    CHECK_EQ(t->refine, 3);
    std::string kinds;
    int n = 0;
    for (int i = 0; i < 40; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        const RadVideoGenBatch* V = b->vgen_b;
        kinds += kind_of(b->gen);
        if (b->gen == RAD_GEN_PASS_VSTEP) {
            const float cur = n < 8 ? (float)kDistilled[n] : (float)kStage2[n - 8];
            const float nxt = n == 7 || n == 10 ? 0.0f
                            : n < 7 ? (float)kDistilled[n + 1] : (float)kStage2[n - 7];
            CHECK_EQ(V->sched[1], cur);
            CHECK_EQ(V->sched[2], nxt);
            CHECK_EQ(V->geom.lat_h, n < 8 ? 2 : 4);
            ++n;
        } else if (b->gen == RAD_GEN_PASS_VUPSCALE) {
            CHECK_EQ(V->sched[1], (float)kStage2[0]);
            CHECK_EQ(V->sched[4], (float)kStage2[0]);
        }
        f.answer(0);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("CSSSSSSSSUSSSAD"));
    CHECK_EQ(t->status(), (int)RAD_OK);
}

/* A guided model: the request's terms over the model's defaults reach every pass; the negative
 * prompt rides in the VCOND pass after the prompt, each from position 0, only while a scale asks
 * for the unconditional branch; the text rows of a step are the model's fixed count. */
TEST(guidance_and_the_negative_prompt_reach_the_passes) {
    VideoFixture f;
    f.build();
    RadVideoGenDecl& d = f.prog.video_gen;
    d.guidance = 1;
    const double defaults[8] = { 3.0, 1.0, 3.0, 0.7, 7.0, 1.0, 3.0, 0.7 };
    std::memcpy(d.guidance_default, defaults, sizeof defaults);
    d.step_text_rows = 6;
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1, 3);
    r->neg_tokens = { 9, 8 };
    r->guidance[1] = 0.0;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VCOND);
    const RadVideoGenBatch* V = b->vgen_b;
    CHECK_EQ(b->n_tok, (int64_t)5);
    CHECK_EQ(b->n_seq, 2);
    CHECK_EQ(V->n_neg, 2);
    CHECK(std::vector<int32_t>(b->cu_seqlens, b->cu_seqlens + 3) == std::vector<int32_t>({ 0, 3, 5 }));
    CHECK(std::vector<int32_t>(b->token_ids, b->token_ids + 5) ==
          std::vector<int32_t>({ 1, 2, 3, 9, 8 }));
    CHECK(std::vector<int32_t>(b->positions, b->positions + 5) ==
          std::vector<int32_t>({ 0, 1, 2, 0, 1 }));
    CHECK_EQ(V->geom.n_text, 6);
    REQUIRE(V->guidance_h != nullptr);
    CHECK_EQ(V->guidance_h[0], 3.0f);
    CHECK_EQ(V->guidance_h[1], 0.0f);
    CHECK_EQ(V->guidance_h[3], 0.7f);
    CHECK_OK(f.drv.done());
    b = f.drv.next(1);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VSTEP);
    CHECK_EQ(b->n_tok, (int64_t)(6 + 60 + 2 * 2));
    CHECK_EQ(b->vgen_b->guidance_h[4], 7.0f);
    /* The device copy the ops read. */
    REQUIRE(b->vgen_b->guidance != nullptr);
    CHECK_EQ(b->vgen_b->guidance[0], 3.0f);
    CHECK_EQ(b->vgen_b->guidance[4], 7.0f);
    CHECK_OK(f.drv.done());

    /* Both scales 1: no unconditional branch, so no negative prompt. */
    VideoFixture g;
    g.build();
    g.prog.video_gen = d;
    CHECK_OK(g.start());
    auto u = g.req(1, 3);
    u->neg_tokens = { 9 };
    u->guidance[0] = 1.0;
    u->guidance[4] = 1.0;
    CHECK_OK(g.drv.submit(std::move(u), &why));
    b = g.drv.next(0);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VCOND);
    CHECK_EQ(b->n_tok, (int64_t)3);
    CHECK_EQ(b->n_seq, 1);
    CHECK_EQ(b->vgen_b->n_neg, 0);

    /* A guided request with no negative prompt cannot run; an unguided model takes no terms. */
    auto m = g.req(1, 3);
    why.clear();
    CHECK_EQ(g.drv.submit(std::move(m), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "negative prompt"));
    VideoFixture h;
    h.build();
    CHECK_OK(h.start());
    auto q = h.req(1, 3);
    q->guidance[2] = 2.0;
    why.clear();
    CHECK_EQ(h.drv.submit(std::move(q), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "no guidance"));
}

/* A first frame reaches the VCOND pass as bf16 x / 255 * 2 - 1 at the size the first steps run
 * at, and every pass of the request says it has one; a model without image conditioning, or a
 * frame of another size, is refused. */
TEST(a_first_frame_reaches_the_prompt_pass_at_the_first_steps_size) {
    VideoFixture f;
    f.build();
    f.prog.video_gen.image_cond = 1;
    f.prog.video_gen.upscale_max = 2.0;
    f.prog.video_gen.upscale_refine = 1;
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1, 3);
    r->width = r->height = 64;
    r->upscale = 2.0;
    r->image.assign(32 * 32 * 3, 0);
    r->image[0] = 255;
    r->image[1] = 128;
    auto wrong = f.req(1, 3);
    wrong->image.assign(64 * 32 * 3, 0);
    wrong->width = wrong->height = 64;
    wrong->upscale = 2.0;
    CHECK_EQ(f.drv.submit(std::move(wrong), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "32x32"));
    CHECK_OK(f.drv.submit(std::move(r), &why));
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VCOND);
    const RadVideoGenBatch* V = b->vgen_b;
    REQUIRE(V->image != nullptr);
    CHECK_EQ(V->has_image, 1);
    CHECK_EQ(V->image[0], rad_f32_to_bf16(1.0f));
    CHECK_EQ(V->image[1], rad_f32_to_bf16(2.0f * (128.0f / 255.0f) - 1.0f));
    CHECK_EQ(V->image[2], rad_f32_to_bf16(-1.0f));
    CHECK_OK(f.drv.done());
    b = f.drv.next(1);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VSTEP);
    CHECK(b->vgen_b->image == nullptr);
    CHECK_EQ(b->vgen_b->has_image, 1);

    VideoFixture g;
    g.build();
    CHECK_OK(g.start());
    auto q = g.req(1, 3);
    q->image.assign(64 * 32 * 3, 0);
    why.clear();
    CHECK_EQ(g.drv.submit(std::move(q), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "picture"));
}

/* A left-padded encoder's prompt ends at text_left_pad - 1, the negative prompt too; one stream of
 * soundtrack rows a latent; an exact upscaler takes its factor only, over sides it divides. */
TEST(left_padded_prompts_one_audio_stream_and_an_exact_upscaler) {
    VideoFixture f;
    f.build();
    RadVideoGenDecl& d = f.prog.video_gen;
    d.guidance = 1;
    d.guidance_default[0] = 3.0;
    d.guidance_default[4] = 1.0;
    d.text_left_pad = 8;
    d.audio_streams = 1;
    d.upscale_max = 2.0;
    d.upscale_refine = 1;
    d.upscale_exact = 1;
    g_audio_streams = 1;
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(1, 3);
    r->neg_tokens = { 9, 8 };
    CHECK_OK(f.drv.submit(std::move(r), &why));
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VCOND);
    CHECK(std::vector<int32_t>(b->positions, b->positions + 5) ==
          std::vector<int32_t>({ 5, 6, 7, 6, 7 }));
    CHECK_OK(f.drv.done());
    b = f.drv.next(1);
    REQUIRE(b && b->gen == RAD_GEN_PASS_VSTEP);
    /* 3 text rows, 30 soundtrack latents in one stream, 2 x 2 video patches. */
    CHECK_EQ(b->n_tok, (int64_t)(3 + 30 + 2 * 2));
    /* The soundtrack as a (30, 1, 1) volume; the video's rows cut at its first latent frame. */
    const int32_t* gd = b->vgen_b->geom_d;
    CHECK(std::vector<int32_t>(gd + RAD_VGEOM_AUDIO_VOL, gd + RAD_VGEOM_AUDIO_VOL + 3) ==
          std::vector<int32_t>({ 30, 1, 1 }));
    CHECK_EQ(gd[RAD_VGEOM_FIRST_CU], 0);
    CHECK_EQ(gd[RAD_VGEOM_FIRST_CU + 2], 4);
    CHECK_EQ(gd[RAD_VGEOM_FIRST_CU + 1] * gd[RAD_VGEOM_LAT_FRAMES], 4);
    CHECK_OK(f.drv.done());

    auto u = f.req(1, 3);
    u->width = 64;
    u->height = 32;
    u->upscale = 2.0;
    u->neg_tokens = { 9 };
    why.clear();
    CHECK_EQ(f.drv.submit(std::move(u), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "multiples of 64"));
    auto v = f.req(1, 3);
    v->width = v->height = 64;
    v->upscale = 1.5;
    v->neg_tokens = { 9 };
    why.clear();
    CHECK_EQ(f.drv.submit(std::move(v), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "exactly 2"));
    g_audio_streams = 2;
}

RAD_TEST_MAIN()
