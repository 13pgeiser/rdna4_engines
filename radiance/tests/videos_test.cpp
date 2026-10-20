/* tests/videos_test.cpp -- OpenAI's video jobs over a video generator (core/server/videos.cpp).
 *
 * The generator is faked here: it finishes every request on submit with two steps, a soundtrack and
 * three small frames, so these pin what the server owns: which requests are refused and under which
 * field, the request it hands the generator (the prompt, the size from `size` or SGLang's
 * `target`, the length rounded up to whole chunks, the steps, a seed), the job's life seen through
 * the video object, the MP4 and thumbnail it serves, listing and deleting, and a failure reported
 * on the job.
 */
#include "server_fakes.h"

#include "gen/video.h"
#include "mm/media.h"

#include <chrono>
#include <thread>

namespace {

struct FakeVideoGen : IVideoGen {
    VideoGenInfo inf;
    int         finish_status = RAD_OK;
    std::string finish_why;
    std::vector<std::unique_ptr<gen::VideoRequest>> got;
    mutable std::mutex got_m;

    FakeVideoGen() {
        inf.name = "fake-video";
        inf.tmpl = "<v>{prompt}";
        inf.max_text = 64;
        inf.fps_num = 24;
        inf.fps_den = 1;
        inf.frames_per_chunk = 17;
        inf.frames_tail = 5;
        inf.min_chunks = 7;
        inf.max_chunks = 20;
        inf.size_multiple = 32;
        inf.min_side = 256;
        inf.max_area = 1344 * 768;
        inf.max_aspect = 4;
        inf.default_width = 1344;
        inf.default_height = 768;
        inf.default_chunks = 7;
        inf.sample_rate = 32000;
        inf.channels = 2;
        inf.default_steps = 50;
        inf.max_steps = 65;
        inf.max_lanes = 2;
    }
    const VideoGenInfo& info() const override { return inf; }
    int submit(std::unique_ptr<gen::VideoRequest> r, std::string*) override {
        std::lock_guard<std::mutex> lk(got_m);
        gen::VideoTicket& t = *r->ticket;
        /* Three frames of the request's size and a tenth of a second of sound, whatever it asked:
         * enough to make a file. */
        t.frames = 3;
        t.width = r->width;
        t.height = r->height;
        t.steps = 2;
        t.sample_rate = 32000;
        t.channels = 2;
        t.fps = 24.0;
        if (r->upscale > 1.0) {
            t.small_width = (int32_t)(r->width / r->upscale) / 32 * 32;
            t.small_height = (int32_t)(r->height / r->upscale) / 32 * 32;
            t.refine = r->refine > 0 ? r->refine : (int32_t)inf.upscale_refine;
        }
        if (finish_status == RAD_OK) {
            /* A request that named no length learns the one the model chose first. */
            if (r->chunks == 0) {
                gen::VideoEvent l;
                l.kind = gen::VideoEvent::LENGTH;
                l.n = 3;
                t.push(std::move(l));
            }
            for (int s = 1; s <= 2; ++s) {
                gen::VideoEvent e;
                e.kind = gen::VideoEvent::STEP;
                e.steps_done = s;
                t.push(std::move(e));
            }
            gen::VideoEvent a;
            a.kind = gen::VideoEvent::AUDIO;
            for (int i = 0; i < 3200; ++i) {
                a.samples.push_back(0.1f * (float)std::sin(i * 0.05));
                a.samples.push_back(-0.1f * (float)std::sin(i * 0.05));
            }
            t.push(std::move(a));
            gen::VideoEvent f;
            f.kind = gen::VideoEvent::FRAMES;
            f.first = 0;
            f.n = 3;
            f.rgb.resize((size_t)3 * r->width * r->height * 3);
            for (size_t i = 0; i < f.rgb.size(); ++i) f.rgb[i] = (uint8_t)(i * 7);
            t.push(std::move(f));
        }
        t.finish(finish_status, finish_why);
        got.push_back(std::move(r));
        return RAD_OK;
    }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
    const gen::VideoRequest& last() const {
        std::lock_guard<std::mutex> lk(got_m);
        return *got.back();
    }
};

struct Videos {
    FakeVideoGen gen;
    std::unique_ptr<Fixture> f;
    Videos() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.videos = &gen;
            o.video_model = true;
        } }));
    }
    static json body(json extra = json::object()) {
        json b = { { "prompt", "a fox in the snow" }, { "size", "256x256" } };
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
        return b;
    }
    HttpResponse create(const json& b) { return f->post("/v1/videos", b.dump()); }
    std::string refused(const json& b, int status = 400) {
        HttpResponse r = create(b);
        CHECK_EQ(r.status, status);
        return str_at(json::parse(r.body, nullptr, false)["error"], "param");
    }
    /* The job until it has finished, polled the way a client does. */
    json wait(const std::string& id) {
        json o;
        for (int i = 0; i < 400; ++i) {
            HttpResponse r = f->get(("/v1/videos/" + id).c_str());
            o = json::parse(r.body, nullptr, false);
            const std::string st = str_at(o, "status");
            if (st == "completed" || st == "failed") break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return o;
    }
};

}  /* namespace */

/* A request that cannot be served is a 400 naming its field before anything is queued. */
TEST(a_video_request_that_cannot_be_served_names_its_field) {
    Videos v;
    CHECK_EQ(v.refused(Videos::body({ { "temperature", 1 } })), std::string("temperature"));
    CHECK_EQ(v.refused({ { "size", "256x256" } }), std::string("prompt"));
    CHECK_EQ(v.refused(Videos::body({ { "prompt", " \n" } })), std::string("prompt"));
    CHECK_EQ(v.refused(Videos::body({ { "size", "250x256" } })), std::string("size"));
    CHECK_EQ(v.refused(Videos::body({ { "size", "1344" } })), std::string("size"));
    CHECK_EQ(v.refused(Videos::body({ { "size", "4096x4096" } })), std::string("size"));
    CHECK_EQ(v.refused(Videos::body({ { "seconds", "16" } })), std::string("seconds"));
    CHECK_EQ(v.refused(Videos::body({ { "seconds", "-1" } })), std::string("seconds"));
    CHECK_EQ(v.refused(Videos::body({ { "num_inference_steps", 1 } })),
             std::string("num_inference_steps"));
    CHECK_EQ(v.refused(Videos::body({ { "num_inference_steps", 66 } })),
             std::string("num_inference_steps"));
    CHECK_EQ(v.refused(Videos::body({ { "input_reference", "file-1" } })),
             std::string("input_reference"));
    CHECK_EQ(v.refused(Videos::body({ { "task", "fl2va" } })), std::string("task"));
    CHECK_EQ(v.refused(Videos::body({ { "conditions", json::array({ json::object() }) } })),
             std::string("conditions"));
    CHECK_EQ(v.refused({ { "prompt", "p" }, { "target", { { "aspect_ratio", "wide" } } } }),
             std::string("target"));
    CHECK_EQ(v.refused(Videos::body({ { "target", { { "aspect_ratio", "16:9" } } } })),
             std::string("size"));
    CHECK_EQ(v.refused(Videos::body({ { "seconds", "5" },
                                      { "target", { { "duration_seconds", 5 } } } })),
             std::string("seconds"));
    CHECK_EQ(v.gen.got.size(), (size_t)0);
}

/* The request the generator gets: the prompt in the template (the template's text read with
 * special tokens, the prompt without), the size, the length in whole chunks rounded up from the
 * seconds asked, the steps and the seed. */
TEST(the_generator_gets_the_size_the_chunks_and_the_steps) {
    Videos v;
    HttpResponse r = v.create(Videos::body({ { "seconds", "8" }, { "num_inference_steps", 20 },
                                             { "seed", 11 } }));
    REQUIRE_EQ(r.status, 200);
    const gen::VideoRequest& q = v.gen.last();
    std::string prompt;
    for (int32_t c : q.tokens) prompt += (char)c;
    CHECK_EQ(prompt, std::string("<v>a fox in the snow"));
    CHECK_EQ(q.width, 256);
    CHECK_EQ(q.height, 256);
    CHECK_EQ(q.chunks, 11);                /* 8 s is 192 frames, 17 * 11 + 5 */
    CHECK_EQ(q.steps, 20);
    CHECK_EQ(q.seed, (uint64_t)11);
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(str_at(o, "object"), std::string("video"));
    CHECK_EQ(str_at(o, "seconds"), std::string("8.00"));
    CHECK_EQ(str_at(o, "size"), std::string("256x256"));
    CHECK(str_at(o, "id").rfind("video_", 0) == 0);
    v.wait(str_at(o, "id"));

    /* Five seconds is not a whole chunk: rounded up to 124 frames. Without a size, the canvas. */
    r = v.create({ { "prompt", "p" }, { "seconds", 5 } });
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().chunks, 7);
    CHECK_EQ(v.gen.last().width, 1344);
    CHECK_EQ(v.gen.last().height, 768);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(str_at(o, "seconds"), std::string("5.17"));
    v.wait(str_at(o, "id"));

    /* SGLang's target: the canvas over a ratio, its short edge and the length. */
    r = v.create({ { "prompt", "p" }, { "task", "t2va" }, { "conditions", json::array() },
                   { "target", { { "aspect_ratio", "9:16" }, { "short_edge", 768 },
                                 { "duration_seconds", 10 } } } });
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().width, 768);
    CHECK_EQ(v.gen.last().height, 1344);
    CHECK_EQ(v.gen.last().chunks, 14);     /* 240 frames: 17 * 14 + 5 = 243 */
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
}

/* A model with a length head chooses the length of a request that names none or says "auto": the
 * generator gets 0 chunks, the video object says "auto" until the model has chosen and then the
 * length chosen. A length named is as for any model; without a head, none named is the default. */
TEST(a_length_head_chooses_the_length_of_a_request_that_names_none) {
    Videos v;
    HttpResponse r = v.create(Videos::body());
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().chunks, 7);
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));

    v.gen.inf.length_head = true;
    for (const json& b : { Videos::body(), Videos::body({ { "seconds", "auto" } }) }) {
        r = v.create(b);
        REQUIRE_EQ(r.status, 200);
        CHECK_EQ(v.gen.last().chunks, 0);
        json o = json::parse(r.body, nullptr, false);
        CHECK_EQ(str_at(o, "seconds"), std::string("auto"));
        o = v.wait(str_at(o, "id"));
        CHECK_EQ(str_at(o, "status"), std::string("completed"));
        CHECK_EQ(str_at(o, "seconds"), std::string("0.12"));    /* 3 frames at 24 a second */
    }
    r = v.create(Videos::body({ { "seconds", "8" } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().chunks, 11);
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
}

/* A model with more than one decoder takes `decoder` by name, the default first: the generator gets
 * its index, the video object names a decoder other than the default, an unknown name or a frame
 * over the decoder's area is a 400 naming the field. A model with one decoder takes no name. */
TEST(a_decoder_is_named_and_held_to_its_area) {
    Videos v;
    CHECK_EQ(v.refused(Videos::body({ { "decoder", "vae" } })), std::string("decoder"));
    v.gen.inf.decoders = { "vae", "diffusion" };
    v.gen.inf.decoder_max_area = { 1344 * 768, 512 * 256 };
    HttpResponse r = v.create(Videos::body({ { "decoder", "diffusion" }, { "size", "512x256" } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().decoder, 1);
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(str_at(o, "decoder"), std::string("diffusion"));
    v.wait(str_at(o, "id"));
    r = v.create(Videos::body({ { "decoder", "vae" } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().decoder, 0);
    CHECK(!json::parse(r.body, nullptr, false).contains("decoder"));
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
    CHECK_EQ(v.refused(Videos::body({ { "decoder", "magic" } })), std::string("decoder"));
    CHECK_EQ(v.refused(Videos::body({ { "decoder", "diffusion" }, { "size", "768x512" } })),
             std::string("decoder"));
}

/* Latent upscaling: refused by a model without an upscaler; with one, the factor and the refine
 * steps reach the generator, a factor of 1 is none, and the refine steps need a factor. */
TEST(latent_upscale_reaches_the_generator_where_the_model_has_an_upscaler) {
    Videos v;
    CHECK_EQ(v.refused(Videos::body({ { "latent_upscale", 2 } })), std::string("latent_upscale"));
    v.gen.inf.upscale_max = 4.0;
    v.gen.inf.upscale_refine = 3;
    CHECK_EQ(v.refused(Videos::body({ { "latent_upscale", 4.5 } })), std::string("latent_upscale"));
    CHECK_EQ(v.refused(Videos::body({ { "latent_upscale", 0.5 } })), std::string("latent_upscale"));
    CHECK_EQ(v.refused(Videos::body({ { "upscale_refine_steps", 2 } })),
             std::string("upscale_refine_steps"));
    CHECK_EQ(v.refused(Videos::body({ { "latent_upscale", 2 }, { "upscale_refine_steps", 1.5 } })),
             std::string("upscale_refine_steps"));
    HttpResponse r = v.create(Videos::body({ { "size", "512x512" }, { "latent_upscale", "2" },
                                             { "upscale_refine_steps", 4 } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().upscale, 2.0);
    CHECK_EQ(v.gen.last().refine, 4);
    /* The video object says how it was made, at creation and when it is fetched. */
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o["latent_upscale"].get<double>(), 2.0);
    CHECK_EQ(o["upscale_refine_steps"].get<int>(), 4);
    CHECK_EQ(str_at(o, "upscale_from"), std::string("256x256"));
    json done = v.wait(str_at(o, "id"));
    CHECK_EQ(str_at(done, "upscale_from"), std::string("256x256"));
    /* Without refine steps named, the generator's default is what the object reports. */
    r = v.create(Videos::body({ { "size", "512x512" }, { "latent_upscale", 2 } }));
    REQUIRE_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o["upscale_refine_steps"].get<int>(), 3);
    v.wait(str_at(o, "id"));
    r = v.create(Videos::body({ { "latent_upscale", 1 } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().upscale, 0.0);
    CHECK_EQ(v.gen.last().refine, 0);
    o = json::parse(r.body, nullptr, false);
    CHECK(!o.contains("latent_upscale") && !o.contains("upscale_refine_steps") &&
          !o.contains("upscale_from"));
    v.wait(str_at(o, "id"));
}

/* A form is read as JSON's fields are (OpenAI's clients send the create call as one). */
TEST(a_form_creates_a_video_too) {
    Videos v;
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/videos";
    q.form = { FormPart{ "prompt", "", "", "a boat" }, FormPart{ "size", "", "", "512x256" },
               FormPart{ "seconds", "", "", "6" } };
    HttpResponse r = v.f->srv->router().dispatch(q);
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().width, 512);
    CHECK_EQ(v.gen.last().height, 256);
    CHECK_EQ(v.gen.last().chunks, 9);      /* 144 frames: 17 * 9 + 5 = 158 */
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
    q.form.push_back(FormPart{ "input_reference", "cat.png", "image/png", "..." });
    CHECK_EQ(v.f->srv->router().dispatch(q).status, 400);
}

/* The job's life: queued at once, completed with progress 100 and an expiry, the MP4 and its
 * first frame served; listing it, deleting it, and a 404 after. */
TEST(a_video_job_completes_and_serves_its_file) {
    Videos v;
    HttpResponse r = v.create(Videos::body());
    REQUIRE_EQ(r.status, 200);
    const std::string id = str_at(json::parse(r.body, nullptr, false), "id");
    json o = v.wait(id);
    if (str_at(o, "status") == "failed" && has(str_at(o["error"], "message"), "no libavcodec"))
        SKIP("this build has no libavcodec to write the MP4 with");
    REQUIRE_EQ(str_at(o, "status"), std::string("completed"));
    CHECK_EQ(o["progress"].get<int>(), 100);
    CHECK(o["completed_at"].is_number());
    CHECK(o["expires_at"].get<int64_t>() > o["completed_at"].get<int64_t>());
    CHECK(o["error"].is_null());
    HttpResponse c = v.f->get(("/v1/videos/" + id + "/content").c_str());
    REQUIRE_EQ(c.status, 200);
    CHECK_EQ(c.content_type, std::string("video/mp4"));
    REQUIRE(c.body.size() > 64);
    CHECK_EQ(c.body.substr(4, 4), std::string("ftyp"));
    /* The moov atom precedes the media data, so a player can start before the file has arrived. */
    CHECK(c.body.find("moov") < c.body.find("mdat"));
    HttpResponse th = v.f->get(("/v1/videos/" + id + "/content?variant=thumbnail").c_str());
    REQUIRE_EQ(th.status, 200);
    CHECK_EQ(th.content_type, std::string("image/webp"));
    CHECK_EQ(th.body.substr(0, 4), std::string("RIFF"));
    CHECK_EQ(v.f->get(("/v1/videos/" + id + "/content?variant=spritesheet").c_str()).status, 400);

    HttpResponse r2 = v.create(Videos::body());
    const std::string id2 = str_at(json::parse(r2.body, nullptr, false), "id");
    v.wait(id2);
    json l = json::parse(v.f->get("/v1/videos").body, nullptr, false);
    CHECK_EQ(str_at(l, "object"), std::string("list"));
    REQUIRE_EQ(l["data"].size(), (size_t)2);
    CHECK_EQ(str_at(l["data"][0], "id"), id2);        /* newest first */
    CHECK_EQ(str_at(l, "first_id"), id2);
    CHECK_EQ(str_at(l, "last_id"), id);
    l = json::parse(v.f->get(("/v1/videos?order=asc&limit=1&after=" + id).c_str()).body, nullptr,
                    false);
    REQUIRE_EQ(l["data"].size(), (size_t)1);
    CHECK_EQ(str_at(l["data"][0], "id"), id2);
    CHECK(!l["has_more"].get<bool>());

    HttpResponse d = v.f->srv->router().call("DELETE", "/v1/videos/" + id);
    REQUIRE_EQ(d.status, 200);
    json dj = json::parse(d.body, nullptr, false);
    CHECK_EQ(str_at(dj, "object"), std::string("video.deleted"));
    CHECK(dj["deleted"].get<bool>());
    CHECK_EQ(v.f->get(("/v1/videos/" + id).c_str()).status, 404);
    CHECK_EQ(v.f->get(("/v1/videos/" + id + "/content").c_str()).status, 404);
}

/* A request the generator fails is a failed job carrying the reason. */
TEST(a_failed_generation_is_a_failed_job_with_its_reason) {
    Videos v;
    v.gen.finish_status = RAD_E_STATE;
    v.gen.finish_why = "the engine stopped";
    HttpResponse r = v.create(Videos::body());
    REQUIRE_EQ(r.status, 200);
    json o = v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
    CHECK_EQ(str_at(o, "status"), std::string("failed"));
    CHECK_EQ(str_at(o["error"], "code"), std::string("generation_failed"));
    CHECK(has(str_at(o["error"], "message"), "the engine stopped"));
}

/* Without a video model the routes are 501. */
TEST(the_video_routes_are_501_without_a_video_model) {
    Fixture f;
    CHECK_EQ(f.post("/v1/videos", Videos::body().dump()).status, 501);
    CHECK_EQ(f.get("/v1/videos").status, 501);
}

namespace {

/* A PNG of w x h, every pixel (r, g, b). */
std::string png_rgb(int32_t w, int32_t h, uint8_t r, uint8_t g, uint8_t b) {
    std::vector<uint8_t> px;
    for (int64_t i = 0; i < (int64_t)w * h; ++i) px.insert(px.end(), { r, g, b });
    std::vector<uint8_t> file;
    std::string why;
    if (mm::encode_image(px.data(), w, h, 3, "png", 0, &file, &why) < 0) return std::string();
    return std::string(file.begin(), file.end());
}

std::string png_url(const std::string& bytes) {
    return "data:image/png;base64," + base64_encode(bytes.data(), bytes.size());
}

/* A guided model that starts from a picture, as LTX-2.5's dev weights. */
void guided(Videos& v, int32_t crf) {
    VideoGenInfo& in = v.gen.inf;
    in.guidance = true;
    const double d[8] = { 3.0, 1.0, 3.0, 0.7, 7.0, 1.0, 3.0, 0.7 };
    std::copy(d, d + 8, in.guidance_default);
    in.default_negative = "ugly";
    in.image_cond = true;
    in.image_crf = crf;
    in.step_evals = true;
    in.upscale_max = 2.0;
    in.upscale_refine = 3;
}

std::string text_of(const std::vector<int32_t>& ids) {
    std::string s;
    for (int32_t c : ids) s += (char)c;
    return s;
}

}  /* namespace */

/* Guidance terms and the negative prompt: refused by a model without guidance; on a guided one the
 * named terms reach the generator (the rest NaN, the model's default), the negative prompt is
 * rendered in the template (the model's own when none is named), and the video object reports the
 * resolved terms. Steps count evaluations there, from 1. */
TEST(guidance_and_the_negative_prompt_reach_a_guided_model) {
    {
        Videos v;
        CHECK_EQ(v.refused(Videos::body({ { "negative_prompt", "x" } })),
                 std::string("negative_prompt"));
        CHECK_EQ(v.refused(Videos::body({ { "stg_scale", 1 } })), std::string("stg_scale"));
    }
    Videos v;
    guided(v, 0);
    CHECK_EQ(v.refused(Videos::body({ { "guidance_rescale", 1.5 } })),
             std::string("guidance_rescale"));
    CHECK_EQ(v.refused(Videos::body({ { "audio_guidance_scale", -1 } })),
             std::string("audio_guidance_scale"));
    CHECK_EQ(v.refused(Videos::body({ { "negative_prompt", 3 } })), std::string("negative_prompt"));
    HttpResponse r = v.create(Videos::body({ { "guidance_scale", 5 }, { "audio_stg_scale", "0" },
                                             { "negative_prompt", "blurry" },
                                             { "num_inference_steps", 1 } }));
    REQUIRE_EQ(r.status, 200);
    const gen::VideoRequest& q = v.gen.last();
    CHECK_EQ(q.guidance[0], 5.0);
    CHECK_EQ(q.guidance[5], 0.0);
    CHECK(std::isnan(q.guidance[1]) && std::isnan(q.guidance[4]));
    CHECK(q.has_neg);
    CHECK_EQ(text_of(q.neg_tokens), std::string("<v>blurry"));
    CHECK_EQ(q.steps, 1);
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o["guidance_scale"].get<double>(), 5.0);
    CHECK_EQ(o["guidance_rescale"].get<double>(), 0.7);
    CHECK_EQ(o["audio_guidance_scale"].get<double>(), 7.0);
    CHECK_EQ(str_at(o, "negative_prompt"), std::string("blurry"));
    v.wait(str_at(o, "id"));

    r = v.create(Videos::body());
    REQUIRE_EQ(r.status, 200);
    CHECK(!v.gen.last().has_neg);
    CHECK_EQ(text_of(v.gen.last().neg_tokens), std::string("<v>ugly"));
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));

    /* Both scales 1: unguided, so the object names no negative prompt. */
    r = v.create(Videos::body({ { "guidance_scale", 1 }, { "audio_guidance_scale", 1 } }));
    REQUIRE_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK(!o.contains("negative_prompt"));
    v.wait(str_at(o, "id"));
}

/* input_reference: refused by a text-only model; on a model that starts from a picture, a data URL
 * in JSON or a file in a form becomes the first frame at the size the first steps run at (the
 * small size under latent upscaling), an even colour staying that colour; task i2va goes with it. */
TEST(input_reference_becomes_the_first_frame_at_the_first_steps_size) {
    const std::string pic = png_rgb(300, 170, 200, 40, 90);
    {
        Videos v;
        CHECK_EQ(v.refused(Videos::body({ { "input_reference", png_url(pic) } })),
                 std::string("input_reference"));
    }
    Videos v;
    guided(v, 0);
    CHECK_EQ(v.refused(Videos::body({ { "input_reference", "data:image/png;base64,AAAA" } })),
             std::string("input_reference"));
    CHECK_EQ(v.refused(Videos::body({ { "input_reference", png_url(pic) }, { "task", "t2va" } })),
             std::string("task"));
    HttpResponse r = v.create(Videos::body({ { "input_reference", { { "image_url", png_url(pic) } } },
                                             { "task", "i2va" } }));
    REQUIRE_EQ(r.status, 200);
    const gen::VideoRequest& q = v.gen.last();
    REQUIRE_EQ(q.image.size(), (size_t)256 * 256 * 3);
    CHECK_EQ((int)q.image[0], 200);
    CHECK_EQ((int)q.image[1], 40);
    CHECK_EQ((int)q.image[q.image.size() - 1], 90);
    json o = json::parse(r.body, nullptr, false);
    CHECK(o.value("input_reference", false));
    v.wait(str_at(o, "id"));

    r = v.create(Videos::body({ { "input_reference", png_url(pic) }, { "size", "512x512" },
                                { "latent_upscale", 2 } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().image.size(), (size_t)256 * 256 * 3);
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));

    HttpRequest f;
    f.method = "POST";
    f.path = "/v1/videos";
    f.form = { FormPart{ "prompt", "", "", "a boat" }, FormPart{ "size", "", "", "512x256" },
               FormPart{ "input_reference", "boat.png", "image/png", pic } };
    r = v.f->srv->router().dispatch(f);
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(v.gen.last().image.size(), (size_t)512 * 256 * 3);
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));
}

/* With the model's H.264 quality the picture goes through the round trip first: an even colour
 * comes back within a few levels (yuv420p's rounding), an odd size losing its last column. */
TEST(input_reference_takes_the_models_h264_round_trip) {
    Videos v;
    guided(v, 18);
    const std::string pic = png_rgb(301, 171, 200, 40, 90);
    HttpResponse r = v.create(Videos::body({ { "input_reference", png_url(pic) } }));
    if (r.status == 503) return;   /* a build without libx264 refuses it, as it should */
    REQUIRE_EQ(r.status, 200);
    const gen::VideoRequest& q = v.gen.last();
    REQUIRE_EQ(q.image.size(), (size_t)256 * 256 * 3);
    const int want[3] = { 200, 40, 90 };
    for (size_t i = 0; i < q.image.size(); i += 997)
        CHECK(std::abs((int)q.image[i] - want[i % 3]) <= 3);
    v.wait(str_at(json::parse(r.body, nullptr, false), "id"));

    mm::Frame in, out;
    in.w = 7;
    in.h = 5;
    in.rgb.assign(7 * 5 * 3, 128);
    std::string why;
    REQUIRE_EQ(mm::h264_roundtrip(in, 18, &out, &why), (int)RAD_OK);
    CHECK_EQ(out.w, 6);
    CHECK_EQ(out.h, 4);
}

RAD_TEST_MAIN()
