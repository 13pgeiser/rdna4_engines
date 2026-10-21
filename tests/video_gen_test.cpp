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

/* The pack's boundary table for this fake's layout: [0 | text | text + stereo | total]. */
int64_t fake_segs(const RadVideoGeom* g, int32_t* cu, int64_t cap) {
    const int64_t audio = 2 * (int64_t)g->n_audio;
    const int64_t rows = g->n_text + audio +
                         (int64_t)g->lat_frames * (g->lat_h / 2) * (g->lat_w / 2);
    const int32_t t[4] = { 0, g->n_text, (int32_t)(g->n_text + audio), (int32_t)rows };
    if (!cu) return 4;
    if (cap < 4) return RAD_E_INVAL;
    for (int i = 0; i < 4; ++i) cu[i] = t[i];
    return 4;
}
/* Same count, wrong boundaries -- the driver must stage its own table, not this one. */
int64_t fake_segs_wrong(const RadVideoGeom* g, int32_t* cu, int64_t cap) {
    if (!cu) return 4;
    if (cap < 4) return RAD_E_INVAL;
    cu[0] = 0; cu[1] = g->n_text + 1; cu[2] = 999999; cu[3] = 12345;
    return 4;
}
/* A segmentation R_SEG cannot hold -- also falls back to the driver's own table. */
int64_t fake_segs_more(const RadVideoGeom* g, int32_t* cu, int64_t cap) {
    if (!cu) return 5;
    if (cap < 5) return RAD_E_INVAL;
    cu[0] = 0; cu[1] = g->n_text; cu[2] = g->n_text + 4; cu[3] = g->n_text + 8; cu[4] = 99;
    return 5;
}

/* The reference-aware layout export for this fake: it lays out the plain sequence only (a
 * reference table would be the real plugin's to present), answering with the rows and writing
 * the boundaries and positions when roomy. */
int64_t fake_pack(const RadVideoGeom* g, const int32_t* ref_tab, int32_t* seg, int64_t cap_seg,
                  float* pos, int64_t cap_pos) {
    if (ref_tab) return RAD_E_INVAL;
    const int64_t audio = 2 * (int64_t)g->n_audio;
    const int64_t rows = g->n_text + audio +
                         (int64_t)g->lat_frames * (g->lat_h / 2) * (g->lat_w / 2);
    if (seg) {
        if (cap_seg < 4) return RAD_E_INVAL;
        seg[0] = 0; seg[1] = g->n_text; seg[2] = (int32_t)(g->n_text + audio);
        seg[3] = (int32_t)rows;
    }
    if (pos) {
        if (cap_pos < rows) return RAD_E_INVAL;
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t i = 0; i < rows; ++i) pos[c * rows + i] = (float)(c * 1000 + i);
    }
    return rows;
}
/* Right-shaped but the rows do not match the geometry -- the driver stages its own table. */
int64_t fake_pack_wrong(const RadVideoGeom* g, const int32_t* ref_tab, int32_t* seg,
                        int64_t cap_seg, float* pos, int64_t cap_pos) {
    (void)ref_tab; (void)pos; (void)cap_pos;
    if (!seg) return 999999;
    if (cap_seg < 4) return RAD_E_INVAL;
    seg[0] = 0; seg[1] = g->n_text + 1; seg[2] = 999999; seg[3] = 12345;
    return 12345;
}
/* A plain table with an entry past the fourth: the rows are right and the first boundaries
 * match, but the extra word poisons the table the geometry backs -- fallback again. */
int64_t fake_pack_more(const RadVideoGeom* g, const int32_t* ref_tab, int32_t* seg,
                       int64_t cap_seg, float* pos, int64_t cap_pos) {
    (void)ref_tab; (void)pos; (void)cap_pos;
    const int64_t audio = 2 * (int64_t)g->n_audio;
    const int64_t rows = g->n_text + audio +
                         (int64_t)g->lat_frames * (g->lat_h / 2) * (g->lat_w / 2);
    if (!seg) return rows;
    if (cap_seg < 5) return RAD_E_INVAL;
    seg[0] = 0; seg[1] = g->n_text; seg[2] = (int32_t)(g->n_text + audio);
    seg[3] = (int32_t)rows; seg[4] = 99;
    return rows;
}

/* The reference-aware layout: when a reference table rides, lay out
 * [text | one block per keyframe | one block per reference | audio target | video target] --
 * boundaries are segment starts with the total rows last, so the video target runs
 * [seg[nseg-2], seg[nseg-1]). Keyframes precede references, matching the real pack. Each keyframe
 * contributes a fixed 2 rows and each picture a fixed 4: this fake lays out the structure the
 * driver stages, not the reference's real visual-token arithmetic. With no table it lays the plain
 * sequence, so the encode/decode passes that call the pack without references get their plain
 * table. With nk = 0 the layout is exactly the reference-only one the older tests assert. */
int64_t fake_pack_ref(const RadVideoGeom* g, const int32_t* ref_tab, int32_t* seg, int64_t cap_seg,
                      float* pos, int64_t cap_pos) {
    const int64_t audio = 2 * (int64_t)g->n_audio;
    const int64_t vtarget = (int64_t)g->lat_frames * (g->lat_h / 2) * (g->lat_w / 2);
    if (!ref_tab) {
        const int64_t rows = g->n_text + audio + vtarget;
        if (seg) {
            if (cap_seg < 4) return RAD_E_INVAL;
            seg[0] = 0; seg[1] = g->n_text; seg[2] = (int32_t)(g->n_text + audio);
            seg[3] = (int32_t)rows;
        }
        if (pos) {
            if (cap_pos < 3 * rows) return RAD_E_INVAL;
            for (int64_t c = 0; c < 3; ++c)
                for (int64_t i = 0; i < rows; ++i) pos[c * rows + i] = (float)(c * 1000 + i);
        }
        return rows;
    }
    const int32_t nr = ref_tab[0], nk = ref_tab[1];
    const int64_t rper = 4, kper = 2;
    const int64_t rows = g->n_text + (int64_t)nk * kper + (int64_t)nr * rper + audio + vtarget;
    const int64_t nseg = (int64_t)nk + (int64_t)nr + 4;
    if (!seg) return rows;
    if (cap_seg < nseg) return RAD_E_INVAL;
    seg[0] = 0;
    int64_t cur = g->n_text;
    for (int32_t k = 0; k < nk; ++k) { seg[1 + k] = (int32_t)cur; cur += kper; }
    for (int32_t r = 0; r < nr; ++r) { seg[1 + nk + r] = (int32_t)cur; cur += rper; }
    seg[1 + nk + nr] = (int32_t)cur;                             /* audio target start */
    seg[2 + nk + nr] = (int32_t)(cur + audio);                   /* video target start */
    seg[3 + nk + nr] = (int32_t)(cur + audio + vtarget);         /* total rows */
    if (pos) {
        if (cap_pos < 3 * rows) return RAD_E_INVAL;
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t i = 0; i < rows; ++i) pos[c * rows + i] = (float)(c * 1000 + i);
    }
    return rows;
}

/* The encode-geometry export for this fake: it agrees with what the driver staged — a video is
 * padded to its own frame count (gd[23]), an audio to its own sample count (gd[0]) — so an encode
 * pass accepts the staged volume and moves on. Only what the driver validates matters here. */
int64_t fake_encode_geom(int32_t kind, const int64_t* in, int32_t* gd, int64_t cap) {
    if (!gd) return kind == RAD_GEN_PASS_AENCODE ? 13 : 25;
    const int64_t need = kind == RAD_GEN_PASS_AENCODE ? 13 : 25;
    if (cap < need) return RAD_E_INVAL;
    for (int64_t i = 0; i < need; ++i) gd[i] = 0;
    if (kind == RAD_GEN_PASS_AENCODE) gd[0] = (int32_t)in[0];   /* n_pad = the sample count */
    else gd[23] = (int32_t)in[0];                               /* T_pad = the frame count */
    return need;
}

/* The encode geometry of the real VAEs, so a video or audio reference gets its true padded volume
 * and latent counts (fake_encode_geom above only echoes its input, which is enough for a still
 * picture). A video of T frames runs ceil(T/17) clips, padded to clips*17 frames (gd[23]) and
 * reporting clips*5-3 latent frames (gd[24]), one frame for a still; an audio of n samples is
 * right-padded to a whole hop (gd[0]) and reports n/800 rounded up as its latent frames (gd[12]).
 * The driver clip-pads / zero-pads to these figures and reads gd[24] / gd[12] for the table. */
int64_t fake_encode_vae(int32_t kind, const int64_t* in, int32_t* gd, int64_t cap) {
    if (!gd) return kind == RAD_GEN_PASS_AENCODE ? 13 : 25;
    const int64_t need = kind == RAD_GEN_PASS_AENCODE ? 13 : 25;
    if (cap < need) return RAD_E_INVAL;
    for (int64_t i = 0; i < need; ++i) gd[i] = 0;
    if (kind == RAD_GEN_PASS_AENCODE) {
        const int64_t lat = (in[0] + 799) / 800;   /* vae_audio_latents: ceil(n / hop) */
        gd[0] = (int32_t)(lat * 800);              /* samples padded to a whole hop */
        gd[12] = (int32_t)lat;                     /* latent frames */
    } else {
        const int64_t clips = (in[0] + 16) / 17;   /* ceil(T / clip length) */
        gd[22] = (int32_t)clips;
        gd[23] = (int32_t)(clips * 17);            /* frames padded to whole clips */
        gd[24] = in[0] <= 1 ? 1 : (int32_t)(clips * 5 - 3);   /* vae_latent_frames */
    }
    return need;
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
    int start_with(gen::VideoDriver::SegsFn segs) {
        return drv.init(prog, fake_positions, { gen::VideoDriver::RankIO{} }, segs);
    }
    int start_with_pack(gen::VideoDriver::PackFn pack) {
        return drv.init(prog, fake_positions, { gen::VideoDriver::RankIO{} }, nullptr, pack);
    }
    int start_with_encode_gen(gen::VideoDriver::EncodeGeomFn enc) {
        return drv.init(prog, fake_positions, { gen::VideoDriver::RankIO{} }, nullptr, nullptr,
                        nullptr, enc);
    }
    int start_with_pack_encode(gen::VideoDriver::PackFn pack, gen::VideoDriver::EncodeGeomFn enc) {
        return drv.init(prog, fake_positions, { gen::VideoDriver::RankIO{} }, nullptr, pack,
                        nullptr, enc);
    }
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

/* A request's shift re-shapes ONLY the stream it names; the stream it leaves at 0 keeps the
 * declaration's trained shift (decl_small is video 12, audio 3). */
TEST(a_request_shift_overrides_the_declaration_per_stream) {
    const RadVideoGenDecl d = decl_small();
    const gen::VideoSchedule base = gen::video_schedule(d, 50);
    const gen::VideoSchedule s = gen::video_schedule(d, 50, 6.0, 0.0);   /* video -> 6, audio keep 3 */
    REQUIRE_EQ(s.sig_v.size(), base.sig_v.size());
    REQUIRE_EQ(s.sig_a.size(), base.sig_a.size());
    const double b = 1.0 - 30.0 / 49.0;
    CHECK(std::fabs(s.sig_v[30] - 6.0 * b / (1.0 + 5.0 * b)) < 1e-6);    /* re-shifted by 6 */
    CHECK(std::fabs(s.sig_a[30] - 3.0 * b / (1.0 + 2.0 * b)) < 1e-6);    /* still the decl's 3 */
    CHECK(s.sig_v[30] != base.sig_v[30]);                                /* video actually moved */
    CHECK(std::fabs(s.sig_a[30] - base.sig_a[30]) < 1e-9);               /* audio untouched */
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

/* A plugin exporting the pack's boundary table stages it into R_SEG; one exporting a table the
 * geometry does not back, or more segments than the register holds, does not -- the driver's own
 * plain table goes instead. All three stage the same bytes for a plain request. */
TEST(the_plugin_segment_table_is_staged_and_a_disagreeing_one_falls_back) {
    const gen::VideoDriver::SegsFn fns[3] = { fake_segs, fake_segs_wrong, fake_segs_more };
    for (gen::VideoDriver::SegsFn fn : fns) {
        VideoFixture f;
        f.build();
        CHECK_OK(f.start_with(fn));
        std::string why;
        auto r = f.req(1);
        CHECK_OK(f.drv.submit(std::move(r), &why));
        bool saw_step = false;
        for (int i = 0; i < 20 && !saw_step; ++i) {
            const RadBatch* b = f.drv.next(i);
            if (!b) break;
            if (b->gen == RAD_GEN_PASS_VSTEP) {
                const RadVideoGenBatch* V = b->vgen_b;
                REQUIRE(V != nullptr);
                const int32_t audio = 2 * V->geom.n_audio;
                CHECK_EQ(V->seg_cu[0], 0);
                CHECK_EQ(V->seg_cu[1], V->geom.n_text);
                CHECK_EQ(V->seg_cu[2], (int32_t)(V->geom.n_text + audio));
                CHECK_EQ(V->seg_cu[3], (int32_t)b->n_tok);
                saw_step = true;
            }
            f.answer(10 * i);
            CHECK_OK(f.drv.done());
        }
        CHECK(saw_step);
    }
}

/* The reference-aware export is preferred over rad_arch_video_segs for the plain sequence it
 * also lays out; a pack whose rows the geometry does not back, or that writes past the plain
 * table's fourth boundary, falls back -- and a plain request stages the same bytes in every
 * case. The batch carries no reference table: the driver's is empty until references ride. */
TEST(the_reference_aware_pack_is_preferred_and_a_disagreeing_one_falls_back) {
    const gen::VideoDriver::PackFn fns[3] = { fake_pack, fake_pack_wrong, fake_pack_more };
    for (gen::VideoDriver::PackFn fn : fns) {
        VideoFixture f;
        f.build();
        CHECK_OK(f.start_with_pack(fn));
        std::string why;
        auto r = f.req(1);
        CHECK_OK(f.drv.submit(std::move(r), &why));
        bool saw_step = false;
        for (int i = 0; i < 20 && !saw_step; ++i) {
            const RadBatch* b = f.drv.next(i);
            if (!b) break;
            if (b->gen == RAD_GEN_PASS_VSTEP) {
                const RadVideoGenBatch* V = b->vgen_b;
                REQUIRE(V != nullptr);
                const int32_t audio = 2 * V->geom.n_audio;
                CHECK_EQ(V->seg_cu[0], 0);
                CHECK_EQ(V->seg_cu[1], V->geom.n_text);
                CHECK_EQ(V->seg_cu[2], (int32_t)(V->geom.n_text + audio));
                CHECK_EQ(V->seg_cu[3], (int32_t)b->n_tok);
                CHECK(V->ref_tab == nullptr);
                saw_step = true;
            }
            f.answer(10 * i);
            CHECK_OK(f.drv.done());
        }
        CHECK(saw_step);
    }
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

namespace {

/* A synthetic picture as the processor hands one over: eight patches in one temporal group,
 * merged two by two into two output rows. `expand` is the two placeholder tokens the driver's
 * rope layout sits on; `pixels` are an index pattern so a byte-parity check has something to
 * check against when the tower path is exercised end to end. */
std::shared_ptr<mm::Item> synthetic_picture(int64_t patch_dim) {
    auto it = std::make_shared<mm::Item>();
    it->kind = mm::Kind::Image;
    it->grid_t = 1;
    it->grid_h = 2;
    it->grid_w = 4;
    it->n_patch = 8;
    it->patch_dim = patch_dim;
    it->pixels.resize((size_t)(it->n_patch * patch_dim));
    for (size_t i = 0; i < it->pixels.size(); ++i) it->pixels[i] = (uint16_t)i;
    it->n_seg = 1;
    it->seg_rows = 2;
    it->rows_h = 1;
    it->rows_w = 2;
    it->expand = { 700, 701 };
    it->seg_at = { 0 };
    return it;
}

gen::VideoRefMedia ref_part(gen::VideoRefMedia::Kind kind, const char* name, int32_t index) {
    gen::VideoRefMedia m;
    m.kind = kind;
    m.index = index;
    m.name = name;
    m.data = "x";
    return m;
}

/* The encoder view of a request whose prompt is { 1, expand..., 2, 3 }. */
std::shared_ptr<mm::PromptMedia> picture_media(int64_t patch_dim, int32_t tok) {
    auto pm = std::make_shared<mm::PromptMedia>();
    mm::Placed pl;
    pl.item = synthetic_picture(patch_dim);
    pl.tok = tok;
    pm->items.push_back(std::move(pl));
    pm->layout();
    return pm;
}

void with_tower(VideoFixture& f, int64_t max_enc) {
    f.prog.encoder.modalities = RAD_MM_IMAGE;
    f.prog.encoder.patch_dim = 6;
    f.prog.ctx.max_enc_patches = max_enc;
}

}  /* namespace */

/* A reference picture's conditioning pass carries the tower's input and the M-RoPE grid exactly
 * as the image driver stages them, and nothing of it leaks into the step passes that follow. */
TEST(a_reference_image_conditions_the_video_prompt) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 700, 701, 2, 3 };
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    r->media = picture_media(6, 1);
    CHECK_OK(f.drv.submit(std::move(r), &why));

    /* The first pass conditions: its media fields carry the picture. */
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int)RAD_GEN_PASS_VCOND);
    CHECK_EQ(b->n_tok, (int64_t)5);
    CHECK_EQ(b->rope_mixed, 1);
    CHECK(b->rope_pos != nullptr);
    CHECK_EQ(b->n_mm_rows, (int64_t)2);
    CHECK(b->mm_rows != nullptr);
    CHECK_EQ(b->enc, 0);
    CHECK_EQ(b->enc_n_patch, (int64_t)8);
    CHECK(b->enc_pixels != nullptr);
    CHECK(b->enc_coord != nullptr);
    CHECK_EQ(b->enc_n_seg, 1);
    CHECK(b->enc_cu != nullptr);
    CHECK_EQ(b->enc_max_seg, 8);
    CHECK(b->vgen_b != nullptr);
    f.answer(0);
    CHECK_OK(f.drv.done());

    /* The steps after it carry no media: every pass starts from a zeroed batch, so the tower
     * never sees a stale conditioning. */
    std::string kinds = "C";
    for (int i = 1; i < 10; ++i) {
        b = f.drv.next(i);
        if (!b) break;
        kinds += kind_of(b->gen);
        if (b->gen == RAD_GEN_PASS_VSTEP) {
            CHECK(b->rope_pos == nullptr);
            CHECK_EQ(b->rope_mixed, 0);
            CHECK_EQ(b->n_mm_rows, (int64_t)0);
            CHECK_EQ(b->enc_n_patch, (int64_t)0);
            CHECK_EQ(b->n_tok, (int64_t)(5 + 100 + 6));
        }
        f.answer(0);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(kinds, std::string("CSSADD"));
}

/* With the encoders declared, a reference picture queues a Vencode pass and enters the pack's
 * reference table: the picture in its OWN latent grid (the encoder's sides over the VAE ratio), at
 * the slot its encode pass numbers, one latent frame, no soundtrack. Until step_dit consumes it the
 * table rides the batch untouched, but it must be exactly what the pack scatters from. */
TEST(a_reference_picture_fills_the_pack_reference_table) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;
    d.enc_ratio = 16;
    d.max_enc_frames = 1;
    d.max_enc_side = 16;
    CHECK_OK(f.start_with_encode_gen(fake_encode_geom));
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 700, 701, 2, 3 };
    gen::VideoRefMedia m = ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1);
    m.enc_frames = 1;
    m.enc_h = 16;
    m.enc_w = 16;
    m.enc_pixels.assign(16 * 16 * 3, 1.0f);
    r->refs.push_back(m);
    r->media = picture_media(6, 1);
    CHECK_OK(f.drv.submit(std::move(r), &why));

    /* The first pass encodes the picture, before any conditioning or denoising. */
    const RadBatch* b = f.drv.next(0);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int)RAD_GEN_PASS_VENCODE);
    CHECK(b->vgen_b != nullptr);
    CHECK(b->vgen_b->ref_tab != nullptr);
    CHECK_EQ(b->vgen_b->enc_kind, (int32_t)RAD_ENC_IMAGE);
    CHECK_EQ(b->vgen_b->enc_slot, (int32_t)1);
    CHECK_OK(f.drv.done());

    /* The reference table: {n_refs, n_keys} then one 6-word picture row {kind, slot, h, w, frames,
     * audio} — a picture (kind 0), slot 1, a 1x1 latent grid (16/16), one latent frame, no sound. */
    const int32_t* tab = b->vgen_b->ref_tab;
    CHECK_EQ(tab[0], 1);
    CHECK_EQ(tab[1], 0);
    CHECK_EQ(tab[2], 0);
    CHECK_EQ(tab[3], 1);
    CHECK_EQ(tab[4], 1);
    CHECK_EQ(tab[5], 1);
    CHECK_EQ(tab[6], 1);
    CHECK_EQ(tab[7], 0);
}

/* A generator whose denoiser consumes the reference table lays each conditioned step at the pack's
 * widened layout: the reference block rides between the text and the target streams, the step
 * stages the pack's full segment-boundary table (video grid last) and its [3, rows] positions, and
 * the sampler's target counts are the pack's video and audio segments -- not the whole widened
 * step (the reference rows are pinned, never sampled). The encode pass that precedes it keeps its
 * plain table; only the step is widened. */
TEST(a_reference_conditioned_step_widens_to_the_pack_layout) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;
    d.enc_ratio = 16;
    d.max_enc_frames = 1;
    d.max_enc_side = 16;
    d.ref_rows = 1;                     // the denoiser consumes the table: turn the widening on
    CHECK_OK(f.start_with_pack_encode(fake_pack_ref, fake_encode_geom));
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 700, 701, 2, 3 };   /* 5 text tokens */
    gen::VideoRefMedia m = ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1);
    m.enc_frames = 1;
    m.enc_h = 16;
    m.enc_w = 16;
    m.enc_pixels.assign(16 * 16 * 3, 1.0f);
    r->refs.push_back(m);
    r->media = picture_media(6, 1);
    CHECK_OK(f.drv.submit(std::move(r), &why));

    const gen::VideoShape s = gen::video_shape(d, 64, 32, 2);
    const int64_t n_text = 5, audio = 2 * s.n_audio, vtarget = s.video_rows, rper = 4;
    const int64_t widened = n_text + rper + audio + vtarget;

    bool saw_step = false, saw_encode = false;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_VSTEP) {
            const RadVideoGenBatch* V = b->vgen_b;
            REQUIRE(V != nullptr);
            saw_step = true;
            CHECK(V->ref_tab != nullptr);
            CHECK_EQ(b->n_tok, widened);                       /* widened by the reference block */
            CHECK_EQ(V->seg_cu[0], 0);
            CHECK_EQ(V->seg_cu[1], (int32_t)n_text);           /* text | reference */
            CHECK_EQ(V->seg_cu[2], (int32_t)(n_text + rper));  /* audio target start */
            CHECK_EQ(V->seg_cu[3], (int32_t)(n_text + rper + audio));   /* video target start */
            CHECK_EQ(V->seg_cu[4], (int32_t)widened);
            CHECK_EQ(V->lat_cu[1], (int32_t)vtarget);          /* sampler owns the target streams */
            CHECK_EQ(V->lat_cu[3], (int32_t)audio);            /* only, not the reference rows */
        } else if (b->gen == RAD_GEN_PASS_VENCODE) {
            saw_encode = true;
        }
        f.answer(10 * i);
        CHECK_OK(f.drv.done());
    }
    CHECK(saw_step);
    CHECK(saw_encode);
}

/* An endpoint-keyframe completion (I2VA/L2VA/FL2VA) rides the same machinery a reference does, but
 * fills the pack's KEYFRAME table instead of its reference rows. Each first/last anchor the server
 * resized to the target canvas folds into a Vencode pass at the IMAGE slot its keyframe ORDER names
 * (1, then 2 -- free, since a keyframe request carries no references), and the step widens to
 * [text | keyframe block | keyframe block | audio target | video target]. The table is {0 references,
 * 2 keys} then the two key triples {pixel frame, video_t, audio_t}: the first pins pixel frame 0,
 * the last the final frame, each one latent frame of video and no soundtrack -- the resolved frame
 * index the reference's MiniMaxH3ImageToVideo assigns. The sampler still owns only the two target
 * streams; the pinned rows are never sampled. */
TEST(a_keyframe_completes_widen_the_step_and_pin_the_ends) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;
    d.enc_ratio = 16;
    d.max_enc_frames = 1;
    d.max_enc_side = 16;
    d.ref_rows = 1;                     // the denoiser consumes the table: turn the widening on
    CHECK_OK(f.start_with_pack_encode(fake_pack_ref, fake_encode_geom));
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 2, 3 };            /* 3 text tokens: pins are not references, so no tower run */
    gen::VideoRefMedia first = ref_part(gen::VideoRefMedia::FIRST_FRAME, "first_frame", 1);
    first.enc_frames = 1;
    first.enc_h = 16;
    first.enc_w = 16;
    first.enc_pixels.assign(16 * 16 * 3, 1.0f);
    gen::VideoRefMedia last = ref_part(gen::VideoRefMedia::LAST_FRAME, "last_frame", 2);
    last.enc_frames = 1;
    last.enc_h = 16;
    last.enc_w = 16;
    last.enc_pixels.assign(16 * 16 * 3, 2.0f);
    r->refs.push_back(first);
    r->refs.push_back(last);
    CHECK_OK(f.drv.submit(std::move(r), &why));

    const gen::VideoShape s = gen::video_shape(d, 64, 32, 2);
    const int64_t n_text = 3, audio = 2 * s.n_audio, vtarget = s.video_rows, kper = 2;
    const int64_t widened = n_text + 2 * kper + audio + vtarget;

    int enc_seen = 0;
    bool saw_first = false, saw_last = false, saw_step = false;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_VENCODE) {
            const RadVideoGenBatch* V = b->vgen_b;
            REQUIRE(V != nullptr);
            CHECK(V->ref_tab != nullptr);                 /* the keyframe table rides the encode too */
            CHECK_EQ(V->enc_kind, (int32_t)RAD_ENC_IMAGE); /* a picture anchor at an image slot */
            CHECK_EQ(V->enc_slot, (int32_t)(++enc_seen));  /* order 1, then 2 */
            if (enc_seen == 1) saw_first = true;
            else if (enc_seen == 2) saw_last = true;
        } else if (b->gen == RAD_GEN_PASS_VSTEP) {
            const RadVideoGenBatch* V = b->vgen_b;
            REQUIRE(V != nullptr);
            saw_step = true;
            CHECK_EQ(b->n_tok, widened);                          /* widened by the two key blocks */
            CHECK_EQ(V->seg_cu[0], 0);
            CHECK_EQ(V->seg_cu[1], (int32_t)n_text);              /* text | key 0 */
            CHECK_EQ(V->seg_cu[2], (int32_t)(n_text + kper));     /* key 0 | key 1 */
            CHECK_EQ(V->seg_cu[3], (int32_t)(n_text + 2 * kper));         /* key 1 | audio target */
            CHECK_EQ(V->seg_cu[4], (int32_t)(n_text + 2 * kper + audio)); /* audio | video target */
            CHECK_EQ(V->seg_cu[5], (int32_t)widened);                    /* video grid ends the row */
            CHECK_EQ(V->lat_cu[1], (int32_t)vtarget);             /* sampler owns the target streams */
            CHECK_EQ(V->lat_cu[3], (int32_t)audio);              /* only, never the pinned rows */
            /* The keyframe table: {n_refs = 0, n_keys = 2}, then two 3-word key triples. Checked
             * while the step's job is still alive -- the driver retires a finished job (freeing this
             * table) on the next() that follows its last pass. */
            const int32_t* tab = V->ref_tab;
            REQUIRE(tab != nullptr);
            CHECK_EQ(tab[0], 0);                 /* no references */
            CHECK_EQ(tab[1], 2);                 /* two endpoint keys */
            CHECK_EQ(tab[2], 0);                 /* first pins output frame 0 */
            CHECK_EQ(tab[3], 1);                 /* one latent frame of video */
            CHECK_EQ(tab[4], 0);                 /* no soundtrack */
            CHECK_EQ(tab[5], s.frames - 1);      /* last pins the final frame */
            CHECK_EQ(tab[6], 1);
            CHECK_EQ(tab[7], 0);
        }
        f.answer(10 * i);
        CHECK_OK(f.drv.done());
    }
    CHECK(saw_first);
    CHECK(saw_last);
    CHECK(saw_step);
}

/* The keyframe's SECOND channel: when the container carries a vision tower, a keyframe request that
 * also carries processed media runs a conditioning pass that stages the tower's rows (the anchor
 * tokens) ALONGSIDE the VAE pin. So a request yields BOTH a VENCODE pass (the pin at image slot 1) and
 * a VCOND pass (the tower run), exactly as a reference picture conditions -- the keyframe is not merely
 * a pin once a tower is present. */
TEST(a_keyframe_completion_also_conditions_through_the_tower) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;
    d.enc_ratio = 16;
    d.max_enc_frames = 1;
    d.max_enc_side = 16;
    d.ref_rows = 1;
    CHECK_OK(f.start_with_pack_encode(fake_pack_ref, fake_encode_geom));
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 700, 701, 2, 3 };    /* a <Picture 1> anchor run sits in the prompt */
    gen::VideoRefMedia first = ref_part(gen::VideoRefMedia::FIRST_FRAME, "first_frame", 1);
    first.enc_frames = 1;
    first.enc_h = 16;
    first.enc_w = 16;
    first.enc_pixels.assign(16 * 16 * 3, 1.0f);
    r->refs.push_back(first);
    r->media = picture_media(6, 1);       /* the SAME keyframe also feeds the tower */
    CHECK_OK(f.drv.submit(std::move(r), &why));

    bool saw_cond = false, saw_pin = false;
    for (int i = 0; i < 20; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_VCOND) {
            saw_cond = true;
            CHECK_EQ(b->n_mm_rows, (int64_t)2);    /* the keyframe's tower run */
            CHECK(b->mm_rows != nullptr);
            CHECK_EQ(b->enc_n_patch, (int64_t)8);  /* its patches staged for the tower */
            CHECK(b->enc_pixels != nullptr);
        } else if (b->gen == RAD_GEN_PASS_VENCODE) {
            saw_pin = true;
            CHECK_EQ(b->vgen_b->enc_kind, (int32_t)RAD_ENC_IMAGE);
            CHECK_EQ(b->vgen_b->enc_slot, (int32_t)1);
        }
        f.answer(10 * i);
        CHECK_OK(f.drv.done());
    }
    CHECK(saw_cond);                    /* the anchor channel ran ... */
    CHECK(saw_pin);                     /* ... alongside the VAE pin */
}

/* A keyframe is a VAE-latent pin, so it conditions through the encoders, NOT the vision tower: a
 * container converted WITHOUT its tower still renders it (pin-only). Only reference media -- whose
 * identity soft tokens come from the tower -- are refused on such a container. */
TEST(a_keyframe_completion_needs_no_vision_tower) {
    VideoFixture f;
    f.build();                          /* no with_tower: the container carries no vision tower */
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;                      /* it does carry the VAE encoders, which the pin needs */
    d.enc_ratio = 16;
    d.max_enc_frames = 1;
    d.max_enc_side = 16;
    d.ref_rows = 1;
    CHECK_OK(f.start_with_pack_encode(fake_pack_ref, fake_encode_geom));
    std::string why;
    auto r = f.req(2);
    r->tokens = { 1, 2, 3 };            /* no tower run: no media placed */
    gen::VideoRefMedia first = ref_part(gen::VideoRefMedia::FIRST_FRAME, "first_frame", 1);
    first.enc_frames = 1;
    first.enc_h = 16;
    first.enc_w = 16;
    first.enc_pixels.assign(16 * 16 * 3, 1.0f);
    r->refs.push_back(first);
    CHECK_OK(f.drv.submit(std::move(r), &why));   /* the pin-only path admits without a tower */
}

/* What the container cannot consume is refused at submit, naming the part or the budget it breaks —
 * never silently dropped into a weaker video. A reference video, soundtrack or audio needs the VAE
 * encoders; on a container converted without them (encodes 0) they are refused by name here, and on
 * one that carries them they are encoded and consume — see the next test. */
TEST(reference_media_the_generator_cannot_consume_are_refused_naming_the_part) {
    std::string why;

    /* A container converted without its tower takes no reference media at all. */
    VideoFixture f;
    f.build();
    CHECK_OK(f.start());
    auto r = f.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    CHECK_EQ(f.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "vision tower"));

    VideoFixture g;
    g.build();
    with_tower(g, 16);
    CHECK_OK(g.start());
    auto refused = [&](std::unique_ptr<gen::VideoRequest> q, const char* what) {
        why.clear();
        CHECK_EQ(g.drv.submit(std::move(q), &why), (int)RAD_E_INVAL);
        CHECK(has(why, what));
    };
    /* This container carries the tower but not the VAEs (encodes 0): a reference that reaches the
     * model only through them is refused by name rather than dropped. */
    r = g.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::VIDEO, "ref_video_1", 1));
    refused(std::move(r), "ref_video_1");
    r = g.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    r->refs.push_back(ref_part(gen::VideoRefMedia::VIDEO_AUDIO, "ref_video_audio_1", 1));
    refused(std::move(r), "ref_video_audio_1");
    r = g.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::AUDIO, "ref_audio_2", 2));
    refused(std::move(r), "ref_audio_2");
    /* An image part with nothing behind it would become a text-only video. */
    r = g.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    refused(std::move(r), "no processed media");
    /* A run placed past the prompt, and more patches than one conditioning pass encodes. */
    r = g.req();
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    r->media = picture_media(6, 3);   /* expand would run to token 5, past the 3-token prompt */
    refused(std::move(r), "past the end");
    VideoFixture h;
    h.build();
    with_tower(h, 4);
    CHECK_OK(h.start());
    r = h.req();
    r->tokens = { 1, 700, 701, 2, 3 };
    r->refs.push_back(ref_part(gen::VideoRefMedia::IMAGE, "ref_image_1", 1));
    r->media = picture_media(6, 1);
    why.clear();
    CHECK_EQ(h.drv.submit(std::move(r), &why), (int)RAD_E_INVAL);
    CHECK(has(why, "at most 4"));
}

/* The other half of the guard above: a container that DOES carry its VAE encoders consumes a
 * reference video, its soundtrack and a standalone audio. Each queues an encode pass that folds its
 * latents into a fixed slot and enters the pack's reference table. A video and its separately-given
 * soundtrack are ONE row — the soundtrack folded into the video's `audio` word — encoded at the
 * VIDEO's slot, audio first then video; a standalone audio is its own row. The request carries no
 * `media` (only images feed the text encoder), and the driver hands each encoder its padded volume:
 * a 5-frame clip repeated out to 17, an 850-sample audio zero-padded up to two 800-sample hops. */
TEST(reference_videos_and_audios_are_encoded_into_the_reference_table) {
    VideoFixture f;
    f.build();
    with_tower(f, 16);
    RadVideoGenDecl& d = f.prog.video_gen;
    d.encodes = 1;
    d.enc_ratio = 16;
    d.max_enc_frames = 289;
    d.max_enc_side = 256;
    d.max_enc_samples = 500000;
    CHECK_OK(f.start_with_encode_gen(fake_encode_vae));
    std::string why;
    auto r = f.req(2, 2);   /* 2 steps keeps the walk short; the encodes all run before it */
    /* A reference video: the 5-frame trimmed minimum at 16x16 (short final clip -> repeat frame). */
    gen::VideoRefMedia v = ref_part(gen::VideoRefMedia::VIDEO, "ref_video_1", 1);
    v.enc_frames = 5; v.enc_h = 16; v.enc_w = 16;
    v.enc_pixels.assign(5 * 16 * 16 * 3, 1.0f);
    /* Its soundtrack: 1600 samples over 2 channels — already whole hops, so unpadded. */
    gen::VideoRefMedia a = ref_part(gen::VideoRefMedia::VIDEO_AUDIO, "ref_video_audio_1", 1);
    a.enc_n = 1600; a.enc_channels = 2;
    a.enc_samples.assign(2 * 1600, 0.25f);
    /* A standalone audio: 850 samples, not a whole hop, so the driver zero-pads 750 onto it. */
    gen::VideoRefMedia s = ref_part(gen::VideoRefMedia::AUDIO, "ref_audio_2", 2);
    s.enc_n = 850; s.enc_channels = 2;
    s.enc_samples.assign(2 * 850, -0.5f);
    r->refs.push_back(v);
    r->refs.push_back(a);
    r->refs.push_back(s);
    CHECK_OK(f.drv.submit(std::move(r), &why));

    int saw_venc = 0, saw_vaudio = 0, saw_saudio = 0;
    const int32_t* tab = nullptr;
    for (int i = 0; i < 12 && saw_venc + saw_vaudio + saw_saudio < 3; ++i) {
        const RadBatch* b = f.drv.next(0);
        if (!b) break;
        const RadVideoGenBatch* V = b->vgen_b;
        if (b->gen == RAD_GEN_PASS_VENCODE || b->gen == RAD_GEN_PASS_AENCODE) {
            REQUIRE(V != nullptr);
            if (b->gen == RAD_GEN_PASS_VENCODE) {
                CHECK_EQ(V->enc_kind, (int32_t)RAD_ENC_VIDEO);
                CHECK_EQ(V->enc_slot, (int32_t)1);
                CHECK_EQ(V->enc_frames, 17);        /* 5 frames clip-padded to one whole clip */
                ++saw_venc;
            } else if (V->enc_kind == RAD_ENC_VAUDIO) {   /* the soundtrack, at the VIDEO's slot */
                CHECK_EQ(V->enc_slot, (int32_t)1);
                CHECK_EQ(V->enc_n_samples, 1600);         /* whole hops: unpadded */
                ++saw_vaudio;
            } else {                                     /* the standalone audio, its own slot */
                CHECK_EQ(V->enc_kind, (int32_t)RAD_ENC_AUDIO);
                CHECK_EQ(V->enc_slot, (int32_t)2);
                CHECK_EQ(V->enc_n_samples, 1600);         /* 850 zero-padded up to two hops */
                ++saw_saudio;
            }
        }
        if (V && V->ref_tab && !tab) tab = V->ref_tab;
        f.answer(10 * i);
        CHECK_OK(f.drv.done());
    }
    CHECK_EQ(saw_venc, 1);
    CHECK_EQ(saw_vaudio, 1);
    CHECK_EQ(saw_saudio, 1);
    REQUIRE(tab != nullptr);
    /* Two rows: the video (its soundtrack folded into the `audio` word) and the standalone audio. The
     * VIDEO_AUDIO is not a row of its own. Video: kind 2, slot 1, 1x1 grid, 5 frames -> 2 latent
     * frames, soundtrack 1600/800 = 2. Audio: kind 1, slot 2, no grid, 850 -> 2 latent frames. */
    CHECK_EQ(tab[0], 2);
    CHECK_EQ(tab[1], 0);
    CHECK_EQ(tab[2], 2); CHECK_EQ(tab[3], 1); CHECK_EQ(tab[4], 1); CHECK_EQ(tab[5], 1);
    CHECK_EQ(tab[6], 2); CHECK_EQ(tab[7], 2);
    CHECK_EQ(tab[8], 1); CHECK_EQ(tab[9], 2); CHECK_EQ(tab[10], 0); CHECK_EQ(tab[11], 0);
    CHECK_EQ(tab[12], 0); CHECK_EQ(tab[13], 2);
}

RAD_TEST_MAIN()
