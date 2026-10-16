/* tests/images_test.cpp -- OpenAI's images API over an image generator (core/server/images.cpp).
 *
 * The generator is faked (server_fakes.h: FakeImageGen finishes every request on submit), so
 * these pin what the server owns: which requests are refused and under which field, the request
 * it hands the generator (sizes rounded to the model's multiple, steps from `quality`, a seed a
 * JavaScript client can read back), the usage split between text and pictures, the background an
 * `auto` resolves to, failures mapped to 400 or 500 in both the body and a stream, multipart
 * uploads, and the stream's events. Cases that encode or decode a picture need the media decoder
 * and skip without it.
 */
#include "server_fakes.h"

#include "mm/media.h"

namespace {

struct Images {
    FakeImageGen img;
    std::unique_ptr<Fixture> f;
    Images() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.images = &img;
            o.image_model = true;
        } }));
    }
    HttpResponse post(const char* path, const json& body) { return f->post(path, body.dump()); }
    json gen(const json& body, int status = 200) {
        HttpResponse r = post("/v1/images/generations", body);
        CHECK_EQ(r.status, status);
        return json::parse(r.body, nullptr, false);
    }
    const gen::Request& last() const { return *img.got.back(); }
};

/* A PNG of w x h with every pixel `rgba`. */
std::string png(int32_t w, int32_t h, uint8_t alpha_left, uint8_t alpha_right) {
    std::vector<uint8_t> px((size_t)w * h * 4, 200);
    for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w; ++x) px[((size_t)y * w + x) * 4 + 3] = x < w / 2 ? alpha_left : alpha_right;
    std::vector<uint8_t> file;
    std::string why;
    if (mm::encode_image(px.data(), w, h, 4, "png", 0, &file, &why) < 0) return std::string();
    return std::string(file.begin(), file.end());
}

std::string data_url(const std::string& bytes) {
    return "data:image/png;base64," + base64_encode(bytes.data(), bytes.size());
}

}  /* namespace */

/* ================================================================== refusals */

/* A request this deployment cannot serve is a 400 naming the field it is about, before anything
 * is queued: unknown fields, a missing or non-string prompt, n outside 1..lanes, url output (no
 * picture is stored), a transparent jpeg, previews without a stream, a step count on a model with
 * its own schedule, half a size (naming the half that is missing), a malformed or oversized size,
 * guidance without a negative prompt, and pictures sent to the endpoint that takes none. */
TEST(an_images_request_that_cannot_be_served_names_its_field) {
    Images s;
    s.img.inf.fixed_schedule = true;
    s.img.inf.fixed_steps = 8;
    struct Case { json body; const char* param; };
    const Case cases[] = {
        { { { "prompt", "x" }, { "wobble", 1 } }, "wobble" },
        { json::object(), "prompt" },
        { { { "prompt", 5 } }, "prompt" },
        { { { "prompt", "x" }, { "n", 0 } }, "n" },
        { { { "prompt", "x" }, { "n", 5 } }, "n" },
        { { { "prompt", "x" }, { "response_format", "url" } }, "response_format" },
        { { { "prompt", "x" }, { "output_format", "jpeg" }, { "background", "transparent" } }, "background" },
        { { { "prompt", "x" }, { "partial_images", 2 } }, "partial_images" },
        { { { "prompt", "x" }, { "num_inference_steps", 20 } }, "num_inference_steps" },
        { { { "prompt", "x" }, { "width", 256 } }, "height" },
        { { { "prompt", "x" }, { "height", 256 } }, "width" },
        { { { "prompt", "x" }, { "size", "12x" } }, "size" },
        { { { "prompt", "x" }, { "size", "2048x2048" } }, "size" },
        { { { "prompt", "x" }, { "true_cfg_scale", 4.0 } }, "true_cfg_scale" },
        { { { "prompt", "x" }, { "image", "data:image/png;base64,AAAA" } }, "image" },
    };
    for (const Case& c : cases) {
        const json e = s.gen(c.body, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string(c.param));
        if (str_at(e["error"], "param") != c.param) fprintf(stderr, "    %s\n", c.body.dump().c_str());
    }
    CHECK(s.gen({ { "prompt", "x" }, { "response_format", "url" } }, 400)["error"]["supported"] ==
          json::array({ "b64_json" }));
    CHECK(s.img.got.empty());
    /* An edit needs a picture; a model that reads none takes no edit at all. */
    CHECK_EQ(str_at(json::parse(s.post("/v1/images/edits", { { "prompt", "x" } }).body)["error"], "param"),
             std::string("image"));
    s.img.inf.max_refs = 0;
    CHECK_EQ(s.post("/v1/images/edits", { { "prompt", "x" } }).status, 501);
}

/* An image model generates no text, so the text endpoints point at the images endpoints, and its
 * model entry says what it does: image generation and editing, no grammar, no video. */
TEST(an_image_model_names_the_images_endpoints_and_its_capabilities) {
    Images s;
    HttpResponse r = s.f->post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 501);
    CHECK(has(r.body, "/v1/images/generations"));
    CHECK_EQ(s.f->post("/v1/completions", R"({"prompt":"x"})").status, 501);
    const json caps = json::parse(s.f->get("/v1/models").body)["data"][0]["capabilities"];
    CHECK(caps.value("image_generation", false));
    CHECK(caps.value("image_edit", false));
    CHECK(!caps.value("grammar", true));
    CHECK(!caps.value("video", true));
}

/* ================================================================== the request handed over */

/* What reaches the generator: sides rounded down to the model's multiple, steps from `quality`
 * (half the default for low, three quarters for medium) unless named, the seed given or a fresh
 * one under 2^53 (returned for reuse, and JavaScript reads JSON numbers as doubles), and each
 * image's seed counting up from it. */
TEST(the_generator_gets_rounded_sizes_quality_steps_and_a_readable_seed) {
    if (!mm::decoder_available()) return;
    Images s;
    const json a = s.gen({ { "prompt", "x" }, { "size", "300x200" }, { "quality", "low" },
                           { "seed", 5 }, { "n", 2 } });
    CHECK_EQ(s.last().width, 288);
    CHECK_EQ(s.last().height, 192);
    CHECK_EQ(s.last().steps, 10);
    CHECK_EQ(s.last().seed, (uint64_t)5);
    CHECK_EQ(a.value("size", ""), std::string("288x192"));
    REQUIRE(a["data"].size() == 2);
    CHECK_EQ(a["data"][1]["seed"].get<uint64_t>(), (uint64_t)6);
    std::vector<uint8_t> file;
    REQUIRE(base64_decode(a["data"][0]["b64_json"].get<std::string>(), &file));
    CHECK(file.size() > 8 && file[1] == 'P' && file[2] == 'N' && file[3] == 'G');

    s.gen({ { "prompt", "x" }, { "quality", "medium" } });
    CHECK_EQ(s.last().steps, 15);
    CHECK_EQ(s.last().width, 256);
    CHECK_EQ(s.last().height, 128);
    s.gen({ { "prompt", "x" }, { "quality", "low" }, { "num_inference_steps", 7 } });
    CHECK_EQ(s.last().steps, 7);
    for (const json& seed : { json(nullptr), json(-1) }) {
        json body = { { "prompt", "x" } };
        if (!seed.is_null()) body["seed"] = seed;
        const json r = s.gen(body);
        CHECK(s.last().seed < ((uint64_t)1 << 53));
        CHECK(r["data"][0]["seed"].get<uint64_t>() < ((uint64_t)1 << 53));
    }
}

/* Usage counts the pictures' token runs as image tokens, once in each branch that carries them:
 * twice with guidance (the negative prompt is a second conditioning). An edit's pictures are
 * resized to the model's reference area at their own aspect before the encoder sees them. */
TEST(usage_counts_each_pictures_tokens_in_every_branch) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    REQUIRE(!pic.empty());
    const json plain = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    const json& u = plain["usage"];
    CHECK_EQ(u["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)3);
    CHECK_EQ(u["input_tokens_details"].value("text_tokens", (int64_t)-1),
             u.value("input_tokens", (int64_t)0) - 3);
    REQUIRE(!s.img.frames.empty());
    CHECK(s.img.frames.back() == std::make_pair(96, 32));
    CHECK_EQ(plain.value("size", ""), std::string("256x128"));
    const json guided = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) }, { "negative_prompt", "" },
          { "true_cfg_scale", 4.0 } }).body);
    CHECK_EQ(guided["usage"]["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)6);
    CHECK(s.last().has_neg);
}

/* `background: auto` keeps an edited picture's transparency and is opaque otherwise; a mask must
 * be the first picture's size. */
TEST(an_auto_background_follows_the_pictures_alpha) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string opaque = png(40, 20, 255, 255), half = png(40, 20, 0, 255);
    REQUIRE(!opaque.empty() && !half.empty());
    auto edit = [&](const std::string& pic) {
        return json::parse(s.post("/v1/images/edits", { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    };
    CHECK_EQ(edit(opaque).value("background", ""), std::string("opaque"));
    CHECK_EQ(edit(half).value("background", ""), std::string("transparent"));
    CHECK_EQ(s.gen({ { "prompt", "x" } }).value("background", ""), std::string("opaque"));
    const json m = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(opaque) }, { "mask", data_url(png(10, 10, 0, 0)) } }).body);
    CHECK_EQ(str_at(m["error"], "param"), std::string("mask"));
}

/* ================================================================== failures */

/* The generator's refusal at submit is the client's 400 when it says the request can never run
 * (RAD_E_INVAL) and a 503 otherwise; a request that failed while running is a 500 carrying the
 * generator's reason, or a 400 for its admission refusal. A stream ends in an error event of the
 * same type. A prompt the model's template does not open with its fixed turn is the server's
 * failure, a 500 typed as one. */
TEST(a_generators_failure_is_a_400_or_a_500_streamed_or_not) {
    if (!mm::decoder_available()) return;
    Images s;
    s.img.submit_status = RAD_E_INVAL;
    s.img.submit_why = "too long for the cache";
    CHECK(has(s.gen({ { "prompt", "x" } }, 400)["error"].value("message", ""), "too long"));
    s.img.submit_status = RAD_E_IO;
    s.gen({ { "prompt", "x" } }, 503);
    s.img.submit_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    for (int st : { RAD_E_INVAL, RAD_E_IO }) {
        s.img.finish_status = st;
        s.img.finish_why = "the decoder ran out of memory";
        const json e = s.gen({ { "prompt", "x" } }, st == RAD_E_INVAL ? 400 : 500)["error"];
        CHECK_EQ(e.value("message", ""), s.img.finish_why);
        CHECK_EQ(e.value("type", ""), std::string(st == RAD_E_INVAL ? "invalid_request_error"
                                                                    : "server_error"));
        HttpResponse r = s.post("/v1/images/generations", { { "prompt", "x" }, { "stream", true } });
        CHECK_EQ(r.status, 200);
        const std::vector<json> ev = sse_events(drain(r));
        REQUIRE(!ev.empty());
        CHECK_EQ(ev.back().value("type", ""), std::string("error"));
        CHECK_EQ(ev.back()["error"].value("type", ""), e.value("type", ""));
    }
    s.img.finish_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    s.img.inf.drop_prefix = "<not the template's turn>";
    const json e = s.gen({ { "prompt", "x" } }, 500)["error"];
    CHECK_EQ(e.value("type", ""), std::string("internal_server_error"));
}

/* ================================================================== the wire forms */

/* An upload as multipart form-data: fields arrive as text (n as "2") and pictures as `image[]`
 * parts. */
TEST(a_multipart_edit_reads_text_fields_and_picture_parts) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/edits";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "n", "", "", "2" },
               FormPart{ "image[]", "a.png", "image/png", pic },
               FormPart{ "image[]", "b.png", "image/png", pic } };
    HttpResponse r = s.f->srv->router().dispatch(q);
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["data"].size(), (size_t)2);
    CHECK_EQ(s.last().n, 2);
    CHECK_EQ(s.last().refs.size(), (size_t)2);
    q.form[1].data = "two";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* A stream sends each preview as a partial_image event numbered from 0, then the image as
 * `completed` with its usage, per image. */
TEST(a_streamed_generation_sends_previews_then_the_image) {
    if (!mm::decoder_available()) return;
    Images s;
    HttpResponse r = s.post("/v1/images/generations",
                            { { "prompt", "x" }, { "stream", true }, { "partial_images", 2 } });
    CHECK_EQ(r.status, 200);
    const std::vector<json> ev = sse_events(drain(r));
    REQUIRE(ev.size() == 3);
    CHECK_EQ(ev[0].value("type", ""), std::string("image_generation.partial_image"));
    CHECK_EQ(ev[0].value("partial_image_index", -1), 0);
    CHECK_EQ(ev[1].value("partial_image_index", -1), 1);
    CHECK_EQ(ev[2].value("type", ""), std::string("image_generation.completed"));
    CHECK(ev[2].contains("usage"));
    CHECK(!ev[2].contains("partial_image_index"));
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

RAD_TEST_MAIN()
