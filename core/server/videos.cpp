/* videos.cpp -- OpenAI's video endpoints over a video generator (core/gen/video_driver.h): a video
 * and its soundtrack from a prompt.
 *
 * A video takes minutes, so it is a job, as OpenAI's API has it: POST /v1/videos queues one and
 * answers with the video object at once, GET /v1/videos/{id} reports its status and progress, and
 * GET /v1/videos/{id}/content downloads the MP4 once it is completed (`variant` thumbnail gives
 * the first frame as WebP). GET /v1/videos lists the jobs, newest first, and DELETE removes one,
 * cancelling it when it is still running. A finished video is kept for `video_ttl_s` seconds
 * (`expires_at`), at most `max_videos` of them.
 *
 * The request takes OpenAI's fields (`prompt`, `model`, `seconds`, `size`) as JSON or as a
 * multipart form, and SGLang's for the same model: `task` (t2va), `target` {`short_edge`,
 * `aspect_ratio`, `duration_seconds`}, `conditions` (empty), `seed`, and `num_inference_steps`.
 * `size` is the frame's width x height, multiples of the model's size multiple; `aspect_ratio`
 * (`w:h`) instead lays the model's own canvas over that ratio: the short edge (the model's, or
 * `short_edge`), the area capped at the model's largest, both sides rounded to the multiple.
 * A video's length is a whole number of the model's chunks: `seconds` is rounded up to the next.
 * A model with a length head (LTX-2.5's duration head) chooses the length of a request that names
 * none, or says `"seconds": "auto"`; the video object reports it once chosen. A model with more
 * than one decoder takes `decoder`, one of their names (LTX-2.5: `vae`, the default, or
 * `diffusion`).
 * A model that can start from a picture takes OpenAI's `input_reference`, a file in the form or a
 * data URL (a string or {image_url}) in JSON, as the video's first frame, prepared as the model's
 * pipeline prepares it: an H.264 round trip at the model's quality, then a bilinear resize to the
 * size the first steps run at. Non-empty `conditions` are refused.
 *
 * A guided model takes `negative_prompt` and its guidance terms, each the model's default when
 * absent: `guidance_scale`, `stg_scale`, `modality_scale` and `guidance_rescale` for the video,
 * the same with `audio_` in front for the soundtrack. A scale of 1 turns that modality's
 * unconditional branch off; the negative prompt is the model's own unless named.
 *
 * A model with a latent upscaler also takes `latent_upscale`, a factor from 1 to its largest: the
 * first steps run at `size` divided by it and the latent is enlarged to `size` for the last
 * `upscale_refine_steps` (the model's default when absent; abi/rad_builder.h, RadVideoGenDecl).
 * The video object of an upscaled video says how it was made: the factor, the refine steps and the
 * small size (`upscale_from`).
 *
 * The prompt is the model's template with `{prompt}` filled; the template is tokenised with its
 * special tokens and the prompt without, so a caller cannot write control tokens into it.
 */
#include "server.h"
#include "loras.h"
#include "oai.h"

#include "gen/video.h"
#include "mm/imaging.h"
#include "mm/media.h"
#include "mm/mp4.h"
#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <random>
#include <thread>

namespace rad {
namespace server {

struct Server::VideoJob {
    std::string id, model, prompt, seconds, size;
    int64_t     created_at = 0;
    uint64_t    seed = 0;
    std::shared_ptr<gen::VideoTicket> ticket;
    std::thread worker;
    /* Under Server::videos_m_ from here on. */
    std::string status = "queued";
    int         progress = 0;
    int64_t     completed_at = 0, expires_at = 0;
    bool        failed = false;
    std::string err_code, err_message;
    std::string mp4;
    std::vector<uint8_t> first;   /* the first frame, RGB, for the thumbnail */
    /* Latent upscaling, as the generator resolved it: the factor, the evaluations after it and the
     * small size; refine 0 when the video is not upscaled. */
    double      upscale = 0.0;
    int32_t     refine = 0;
    std::string upscale_from;
    /* A guided model's terms as resolved against its defaults, the negative prompt when a scale
     * used it, and whether the video started from a picture. */
    bool        guided = false;
    double      guidance[8] = {};
    std::string negative;
    bool        has_negative = false;
    bool        from_image = false;
    std::string decoder;          /* a decoder the request named other than the default */
    int32_t     w = 0, h = 0;
    bool        worker_done = false;   /* its thread has nothing left to do but return */
};

namespace {

/* The largest picture `input_reference` may decode to, which bounds the memory a request can make
 * the server allocate before the resize. */
constexpr int64_t kMaxRefPixels = (int64_t)1 << 26;

std::string new_video_id() {
    static const char* hex = "0123456789abcdef";
    std::random_device rd;
    std::string id = "video_";
    for (int i = 0; i < 32; ++i) id.push_back(hex[rd() & 15]);
    return id;
}

/* "1344x768" -> (1344, 768); false for anything else. */
bool parse_size(const std::string& s, int64_t* w, int64_t* h) {
    const size_t x = s.find_first_of("xX");
    if (x == std::string::npos || x == 0 || x + 1 >= s.size()) return false;
    char* e = nullptr;
    *w = std::strtoll(s.c_str(), &e, 10);
    if (e != s.c_str() + x) return false;
    *h = std::strtoll(s.c_str() + x + 1, &e, 10);
    return *e == '\0' && *w > 0 && *h > 0;
}

/* "16:9" -> 16/9 as two positives; false for anything else. */
bool parse_ratio(const std::string& s, double* a, double* b) {
    const size_t c = s.find(':');
    if (c == std::string::npos) return false;
    char* e = nullptr;
    *a = std::strtod(s.c_str(), &e);
    if (e != s.c_str() + c) return false;
    *b = std::strtod(s.c_str() + c + 1, &e);
    return *e == '\0' && *a > 0 && *b > 0;
}

/* The model's canvas for a ratio: the short edge, the area capped, both sides rounded to the
 * multiple (ties to even, as the reference's float round). */
void canvas_for(double rw, double rh, int64_t short_edge, int64_t max_area, int64_t mult,
                int64_t* w, int64_t* h) {
    const double ratio = rw / rh;
    double W = ratio >= 1.0 ? short_edge * ratio : (double)short_edge;
    double H = ratio >= 1.0 ? (double)short_edge : short_edge / ratio;
    if (W * H > (double)max_area) {
        const double sc = std::sqrt((double)max_area / (W * H));
        W *= sc;
        H *= sc;
    }
    *w = std::max<int64_t>(mult, (int64_t)std::nearbyint(W / mult) * mult);
    *h = std::max<int64_t>(mult, (int64_t)std::nearbyint(H / mult) * mult);
}

}  /* namespace */

HttpResponse Server::handle_video_create(const HttpRequest& req) {
    if (!deps_.videos || !deps_.tok)
        return error_response(err_not_implemented(
            "/v1/videos requires a video generation model; this one does not make video"),
                              req.peer);
    const VideoGenInfo& info = deps_.videos->info();

    /* JSON, or a form whose fields are strings (OpenAI's clients send the create call as a form,
     * since it may carry a file). */
    json body = json::object();
    std::vector<uint8_t> ref_bytes;
    bool has_ref = false;
    if (!req.form.empty()) {
        for (const FormPart& p : req.form) {
            if (p.name == "input_reference" && !p.filename.empty()) {
                ref_bytes.assign(p.data.begin(), p.data.end());
                has_ref = true;
                continue;
            }
            if (!p.filename.empty())
                return error_response(err_bad(p.name, "'" + p.name + "' takes text, not a file"),
                                      req.peer);
            json v = p.data;
            if (p.name == "target" || p.name == "conditions" ||
                ((p.name == "loras" || p.name == "lora") && !p.data.empty() &&
                 (p.data[0] == '[' || p.data[0] == '{'))) {
                v = json::parse(p.data, nullptr, false);
                if (v.is_discarded())
                    return error_response(err_bad(p.name, "expected JSON"), req.peer);
            }
            body[p.name] = v;
        }
    } else {
        ApiError err;
        if (parse_request_object(req.body, lim_, &body, &err) < 0)
            return error_response(err, req.peer);
    }
    static const char* kKnown[] = {
        "model", "prompt", "seconds", "size", "input_reference", "seed", "num_inference_steps",
        "task", "target", "conditions", "user", "latent_upscale", "upscale_refine_steps",
        "negative_prompt", "decoder", "loras", "lora",
        /* RadVideoGenBatch::guidance_h's order */
        "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
        "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale", "audio_guidance_rescale",
    };
    for (auto it = body.begin(); it != body.end(); ++it) {
        bool known = false;
        for (const char* k : kKnown) known = known || it.key() == k;
        if (!known)
            return error_response(err_bad(it.key(), "'" + it.key() + "' is not a field of "
                                                     "/v1/videos"), req.peer);
    }
    auto bad = [&](const char* k, const std::string& why) { return error_response(err_bad(k, why), req.peer); };
    auto has = [&](const json& o, const char* k) { return o.contains(k) && !o[k].is_null(); };
    /* A number, or a string holding one (a form's fields are strings). */
    auto number = [&](const json& v, double* out) {
        if (v.is_number()) { *out = v.get<double>(); return true; }
        if (!v.is_string()) return false;
        const std::string s = v.get<std::string>();
        char* e = nullptr;
        *out = std::strtod(s.c_str(), &e);
        return !s.empty() && *e == '\0';
    };

    if (!has(body, "prompt") || !body["prompt"].is_string() ||
        body["prompt"].get<std::string>().find_first_not_of(" \t\r\n") == std::string::npos)
        return bad("prompt", "the prompt is required, as a non-empty string");
    std::string prompt = body["prompt"].get<std::string>();
    /* Runtime LoRA adapters: `loras` and <lora:...> tags, which leave the prompt. */
    std::vector<std::pair<std::string, double>> loras;
    {
        std::string field, why;
        if (parse_loras(body, &prompt, &loras, &field, &why) < 0) return bad(field.c_str(), why);
        if (!loras.empty() && !opt_.gen_lora_acquire)
            return error_response(err_unsupported(field.c_str(), "this server was started without "
                                                                 "adapters (--lora, --lora-dir)"),
                                  req.peer);
    }
    if (has(body, "input_reference")) {
        /* JSON: a data URL, as a string or an image_url object. */
        const json& v = body["input_reference"];
        std::string u;
        if (v.is_string()) u = v.get<std::string>();
        else if (v.is_object() && v.contains("image_url") && v["image_url"].is_string())
            u = v["image_url"].get<std::string>();
        else if (v.is_object() && v.contains("image_url") && v["image_url"].is_object() &&
                 v["image_url"].contains("url") && v["image_url"]["url"].is_string())
            u = v["image_url"]["url"].get<std::string>();
        else
            return bad("input_reference", "a picture is a file, a data: URL, or an object with an "
                                          "image_url");
        ApiError err;
        if (decode_media_url(u, "input_reference", &ref_bytes, &err) < 0)
            return error_response(err, req.peer);
        has_ref = true;
    }
    if (has_ref && !info.image_cond)
        return error_response(err_unsupported("input_reference", "this model makes video from "
                                              "text only"), req.peer);
    const std::string task_want = has_ref ? "i2va" : "t2va";
    if (has(body, "task") && (!body["task"].is_string() ||
                              body["task"].get<std::string>() != task_want)) {
        ApiError e = err_unsupported("task", has_ref ? "a video from a picture is i2va"
                                                     : "this request makes video from text: t2va");
        e.supported = { task_want };
        return error_response(e, req.peer);
    }
    if (has(body, "conditions") && (!body["conditions"].is_array() || !body["conditions"].empty()))
        return error_response(err_unsupported("conditions", "this server makes video from text "
                                              "only; conditions must be empty"), req.peer);
    json target = json::object();
    if (has(body, "target")) {
        if (!body["target"].is_object()) return bad("target", "expected an object");
        target = body["target"];
        for (auto it = target.begin(); it != target.end(); ++it)
            if (it.key() != "short_edge" && it.key() != "aspect_ratio" &&
                it.key() != "duration_seconds")
                return bad("target", "'" + it.key() + "' is not a field of target");
    }

    /* ---- the size */
    int64_t W = info.default_width, H = info.default_height;
    const int64_t short_def = std::min(info.default_width, info.default_height);
    if (has(body, "size")) {
        if (!body["size"].is_string() || !parse_size(body["size"].get<std::string>(), &W, &H))
            return bad("size", "expected WIDTHxHEIGHT, e.g. " + std::to_string(info.default_width) +
                                   "x" + std::to_string(info.default_height));
        if (has(target, "aspect_ratio") || has(target, "short_edge"))
            return bad("size", "give size or target's aspect_ratio and short_edge, not both");
    } else if (has(target, "aspect_ratio") || has(target, "short_edge")) {
        double rw = (double)info.default_width, rh = (double)info.default_height;
        if (has(target, "aspect_ratio")) {
            if (!target["aspect_ratio"].is_string())
                return bad("target", "aspect_ratio is a string, w:h or auto");
            const std::string ar = target["aspect_ratio"].get<std::string>();
            if (ar != "auto" && !parse_ratio(ar, &rw, &rh))
                return bad("target", "aspect_ratio must be w:h (16:9) or auto");
        }
        int64_t se = short_def;
        if (has(target, "short_edge")) {
            double v = 0;
            if (!number(target["short_edge"], &v) || v < 1 || v != std::floor(v))
                return bad("target", "short_edge must be a positive whole number of pixels");
            se = (int64_t)v;
        }
        if (std::max(rw, rh) > (double)info.max_aspect * std::min(rw, rh))
            return bad("target", "the aspect ratio may be at most " +
                                     std::to_string(info.max_aspect) + ":1 either way");
        canvas_for(rw, rh, se, info.max_area, info.size_multiple, &W, &H);
    }
    if (W % info.size_multiple || H % info.size_multiple || W < info.min_side ||
        H < info.min_side || W * H > info.max_area ||
        std::max(W, H) > info.max_aspect * std::min(W, H)) {
        int64_t sw = 0, sh = 0;
        canvas_for((double)W, (double)H, short_def, info.max_area, info.size_multiple, &sw, &sh);
        return bad("size", std::to_string(W) + "x" + std::to_string(H) + " is not a size this "
                   "model makes: sides are multiples of " + std::to_string(info.size_multiple) +
                   " from " + std::to_string(info.min_side) + ", at most " +
                   std::to_string(info.max_area) + " pixels and " +
                   std::to_string(info.max_aspect) + ":1; its canvas at that ratio is " +
                   std::to_string(sw) + "x" + std::to_string(sh));
    }

    /* ---- the length, in whole chunks */
    const double fps = (double)info.fps_num / (double)info.fps_den;
    const double min_s = (double)(info.frames_per_chunk * info.min_chunks + info.frames_tail) / fps;
    const double max_s = (double)(info.frames_per_chunk * info.max_chunks + info.frames_tail) / fps;
    int64_t chunks = info.default_chunks;
    const bool has_sec = has(body, "seconds"), has_dur = has(target, "duration_seconds");
    if (has_sec && has_dur)
        return bad("seconds", "give seconds or target's duration_seconds, not both");
    /* A model with a length head chooses the length of a request that names none, or `auto`. */
    const bool auto_len = info.length_head &&
        (!(has_sec || has_dur) ||
         (has_sec && body["seconds"].is_string() && body["seconds"].get<std::string>() == "auto"));
    if (auto_len) {
        chunks = 0;
    } else if (has_sec || has_dur) {
        double sec = 0;
        if (!number(has_sec ? body["seconds"] : target["duration_seconds"], &sec) || !(sec > 0))
            return bad(has_sec ? "seconds" : "target", "the length is a positive number of seconds");
        const int64_t want = (int64_t)std::ceil(sec * fps - 1e-9);
        chunks = std::max<int64_t>(info.min_chunks,
                                   (want - info.frames_tail + info.frames_per_chunk - 1) /
                                       info.frames_per_chunk);
        if (chunks > info.max_chunks)
            return bad(has_sec ? "seconds" : "target",
                       "this model makes " + rad::fmt("%.2f", min_s) + " to " +
                           rad::fmt("%.2f", max_s) + " seconds");
    }
    /* The length the job reports until the model has chosen one. */
    const int64_t frames =
        info.frames_per_chunk * (chunks > 0 ? chunks : info.default_chunks) + info.frames_tail;

    std::unique_ptr<gen::VideoRequest> r = std::make_unique<gen::VideoRequest>();
    r->width = (int32_t)W;
    r->height = (int32_t)H;
    r->chunks = (int32_t)chunks;
    /* ---- the decoder, by name, the model's first when absent */
    std::string decoder;
    if (has(body, "decoder")) {
        if (!body["decoder"].is_string())
            return bad("decoder", "the decoder is a name");
        decoder = body["decoder"].get<std::string>();
        std::string names;
        int32_t idx = -1;
        for (size_t i = 0; i < info.decoders.size(); ++i) {
            names += (i ? ", " : "") + info.decoders[i];
            if (info.decoders[i] == decoder) idx = (int32_t)i;
        }
        if (idx < 0)
            return bad("decoder", info.decoders.empty()
                                      ? std::string("this model has one decoder and takes no name")
                                      : "this model decodes with " + names);
        if (W * H > info.decoder_max_area[(size_t)idx])
            return bad("decoder", "the " + decoder + " decoder makes frames of " +
                                      std::to_string(info.decoder_max_area[(size_t)idx]) +
                                      " pixels at most");
        r->decoder = idx;
        if (idx == 0) decoder.clear();
    }
    if (has(body, "num_inference_steps")) {
        double v = 0;
        const char* unit = info.step_evals ? " model evaluations"
                                           : " sigma grid points, the last one 0";
        const int64_t lo = info.step_evals ? 1 : 2;
        if (info.fixed_steps && (!number(body["num_inference_steps"], &v) ||
                                 v != (double)info.max_steps))
            return bad("num_inference_steps",
                       "this model runs the schedule it was trained on: " +
                           std::to_string(info.max_steps) + unit);
        if (!number(body["num_inference_steps"], &v) || v != std::floor(v) || v < lo ||
            v > (double)info.max_steps)
            return bad("num_inference_steps", "expected an integer in " + std::to_string(lo) +
                                                  ".." + std::to_string(info.max_steps) + " (" +
                                                  (unit + 1) + ")");
        r->steps = (int32_t)v;
    }
    /* ---- guidance */
    static const char* kGuide[8] = {
        "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
        "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale", "audio_guidance_rescale",
    };
    for (int i = 0; i < 8; ++i) {
        if (!has(body, kGuide[i])) continue;
        if (!info.guidance)
            return error_response(err_unsupported(kGuide[i], "this model takes no guidance"),
                                  req.peer);
        double v = 0;
        const bool rescale = i % 4 == 3;
        if (!number(body[kGuide[i]], &v) || !(v >= 0.0) || v > (rescale ? 1.0 : 100.0))
            return bad(kGuide[i], rescale ? "expected a number from 0 to 1"
                                          : "expected a number from 0 to 100");
        r->guidance[i] = v;
    }
    std::string negative = info.default_negative;
    if (has(body, "negative_prompt")) {
        if (!info.guidance)
            return error_response(err_unsupported("negative_prompt", "this model takes no "
                                                  "guidance"), req.peer);
        if (!body["negative_prompt"].is_string())
            return bad("negative_prompt", "expected a string");
        negative = body["negative_prompt"].get<std::string>();
        r->has_neg = true;
    }
    if (has(body, "latent_upscale")) {
        double v = 0;
        if (!(info.upscale_max > 0.0))
            return error_response(err_unsupported("latent_upscale", "this model has no latent "
                                                  "upscaler"), req.peer);
        if (!number(body["latent_upscale"], &v) || !(v >= 1.0) || v > info.upscale_max)
            return bad("latent_upscale", "expected a factor from 1 to " +
                                             rad::fmt("%g", info.upscale_max));
        r->upscale = v > 1.0 ? v : 0.0;
    }
    if (has(body, "upscale_refine_steps")) {
        double v = 0;
        if (r->upscale <= 1.0)
            return bad("upscale_refine_steps", "counts the steps after a latent upscale; give "
                                               "latent_upscale above 1 with it");
        if (!number(body["upscale_refine_steps"], &v) || v != std::floor(v) || v < 1 || v > 1e6)
            return bad("upscale_refine_steps", "expected a positive integer");
        r->refine = (int32_t)v;
    }
    bool seed_given = false;
    if (has(body, "seed")) {
        double v = 0;
        if (!number(body["seed"], &v) || v != std::floor(v)) return bad("seed", "expected an integer");
        r->seed = (uint64_t)(int64_t)v;
        seed_given = v != -1;
    }
    if (!seed_given) {
        /* 53 bits, as the other generators choose one: JavaScript reads a number as a double. */
        std::random_device rd;
        r->seed = (((uint64_t)rd() << 32) ^ rd()) & (((uint64_t)1 << 53) - 1);
    }

    /* ---- the prompt: the template's text with special tokens, the prompt without */
    {
        static const std::string kSlot = "{prompt}";
        const std::string& t = info.tmpl;
        const size_t p = t.find(kSlot);
        auto render = [&](const std::string& text, std::vector<int32_t>* out) {
            auto put = [&](const std::string& s, bool special) {
                if (s.empty()) return;
                const auto ids = deps_.tok->encode(s, false, special);
                out->insert(out->end(), ids.begin(), ids.end());
            };
            put(t.substr(0, p), true);
            put(text, false);
            put(t.substr(p + kSlot.size()), true);
        };
        render(prompt, &r->tokens);
        if ((int64_t)r->tokens.size() > info.max_text)
            return bad("prompt", "the prompt is " + std::to_string(r->tokens.size()) +
                                     " tokens; this model takes at most " +
                                     std::to_string(info.max_text));
        if (info.guidance) {
            render(negative, &r->neg_tokens);
            if ((int64_t)r->neg_tokens.size() > info.max_text)
                return bad("negative_prompt", "the negative prompt is " +
                                                  std::to_string(r->neg_tokens.size()) +
                                                  " tokens; this model takes at most " +
                                                  std::to_string(info.max_text));
        }
    }

    /* ---- the first frame, at the size the first steps run at */
    if (has_ref) {
        int32_t fw = r->width, fh = r->height;
        if (r->upscale > 1.0) {
            RadVideoGenDecl geo{};   /* video_small_size reads the size multiple only */
            geo.size_multiple = info.size_multiple;
            gen::video_small_size(geo, r->width, r->height, r->upscale, &fw, &fh);
        }
        mm::Frame f;
        std::string why;
        if (mm::decode_image(ref_bytes.data(), ref_bytes.size(), kMaxRefPixels, &f, &why) < 0)
            return bad("input_reference", "the picture could not be read: " + why);
        if (info.image_crf > 0) {
            mm::Frame rt;
            const int e = mm::h264_roundtrip(f, info.image_crf, &rt, &why);
            if (e < 0) {
                ApiError err = err_bad("input_reference", "the picture could not be prepared: " + why);
                if (e != RAD_E_INVAL) { err.status = 503; err.type = "server_error"; }
                return error_response(err, req.peer);
            }
            f = std::move(rt);
        }
        mm::FrameRGBA a, sized;
        a.w = f.w;
        a.h = f.h;
        a.rgba.resize((size_t)f.w * f.h * 4);
        for (size_t i = 0, n = (size_t)f.w * f.h; i < n; ++i) {
            std::memcpy(&a.rgba[4 * i], &f.rgb[3 * i], 3);
            a.rgba[4 * i + 3] = 255;
        }
        mm::resize_bilinear_rgba(a, fw, fh, &sized);
        r->image.resize((size_t)fw * fh * 3);
        for (size_t i = 0, n = (size_t)fw * fh; i < n; ++i)
            std::memcpy(&r->image[3 * i], &sized.rgba[4 * i], 3);
    }

    /* ---- admitted and queued: the job holds its place until its file is written */
    if (!loras.empty()) {
        std::string why, note;
        const int s = opt_.gen_lora_acquire(loras, &r->lora_slot, &r->lora_hold, &note, &why);
        if (s < 0) {
            ApiError e = err_bad("loras", why);
            if (s == RAD_E_NOTFOUND) { e.status = 404; e.type = "not_found_error"; }
            else if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    if (auto no = admit(1, req.peer)) return *no;
    auto j = std::make_shared<VideoJob>();
    j->id = new_video_id();
    j->model = opt_.model_id;
    j->prompt = prompt;
    j->seconds = auto_len ? std::string("auto") : rad::fmt("%.2f", (double)frames / fps);
    j->size = std::to_string(W) + "x" + std::to_string(H);
    j->created_at = unix_now();
    j->seed = r->seed;
    j->ticket = std::make_shared<gen::VideoTicket>();
    r->ticket = j->ticket;
    const int64_t n_tok = (int64_t)r->tokens.size();
    const int32_t steps = r->steps;
    j->guided = info.guidance;
    for (int i = 0; i < 8; ++i)
        j->guidance[i] = std::isnan(r->guidance[i]) ? info.guidance_default[i] : r->guidance[i];
    j->has_negative = info.guidance && (j->guidance[0] != 1.0 || j->guidance[4] != 1.0);
    if (j->has_negative) j->negative = negative;
    j->from_image = !r->image.empty();
    j->decoder = decoder;
    const double upscale = r->upscale;
    {
        std::string w;
        const int s = deps_.videos->submit(std::move(r), &w);
        if (s < 0) {
            release(1);
            ApiError e = err_bad("", w.empty() ? std::string(rad_strerror(s)) : w);
            if (s != RAD_E_INVAL) { e.status = 503; e.type = "server_error"; }
            return error_response(e, req.peer);
        }
    }
    const gen::VideoTicket& tk = *j->ticket;
    if (tk.refine > 0) {
        j->upscale = upscale;
        j->refine = tk.refine;
        j->upscale_from = std::to_string(tk.small_width) + "x" + std::to_string(tk.small_height);
    }
    RAD_INFO("video %s: %s, %s, %lld prompt tokens, %d steps%s, seed=%llu",
             j->id.c_str(), j->size.c_str(),
             auto_len ? "the length the model chooses"
                      : rad::fmt("%lld frames (%s s)", (long long)frames, j->seconds.c_str()).c_str(),
             (long long)n_tok, steps > 0 ? steps : (int)tk.steps,
             tk.refine > 0 ? rad::fmt(" (the first %d at %dx%d, upscaled for the last %d)",
                                      (int)(tk.steps - tk.refine), (int)tk.small_width,
                                      (int)tk.small_height, (int)tk.refine).c_str()
                           : "",
             (unsigned long long)j->seed);
    std::lock_guard<std::mutex> lk(videos_m_);
    sweep_videos_locked(unix_now());
    videos_.push_back(j);
    j->worker = std::thread([this, j] { video_work(j); });
    return HttpResponse::json_body(200, json_dump(video_json(*j)));
}

/* The job's thread. Progress is the steps' share of 90, then the frames' of the last 10. */
void Server::video_work(std::shared_ptr<VideoJob> j) {
    gen::VideoTicket& t = *j->ticket;
    const VideoGenInfo& info = deps_.videos->info();
    mm::Mp4Writer mp4;
    std::string why;
    int st = mp4.open(t.width, t.height, info.fps_num, info.fps_den, t.sample_rate, t.channels,
                      &why);
    int32_t frames = 0;
    bool aborted = false;
    while (st >= 0) {
        gen::VideoEvent e;
        bool to = false;
        if (!t.next(&e, 500, &to)) {
            if (!to) break;
            if (stopping_.load()) {
                t.cancel.store(true);
                aborted = true;
                break;
            }
            continue;
        }
        if (e.kind == gen::VideoEvent::STEP) {
            std::lock_guard<std::mutex> lk(videos_m_);
            j->status = "in_progress";
            j->progress = t.steps > 0 ? (int)(90LL * e.steps_done / t.steps) : 0;
        } else if (e.kind == gen::VideoEvent::LENGTH) {
            std::lock_guard<std::mutex> lk(videos_m_);
            j->seconds = rad::fmt("%.2f", (double)e.n * info.fps_den / info.fps_num);
        } else if (e.kind == gen::VideoEvent::AUDIO) {
            if (t.channels > 0)
                st = mp4.add_audio(e.samples.data(), (int64_t)e.samples.size() / t.channels, &why);
        } else if (e.kind == gen::VideoEvent::FRAMES) {
            if (e.first == 0 && e.n > 0) {
                std::lock_guard<std::mutex> lk(videos_m_);
                j->first.assign(e.rgb.begin(), e.rgb.begin() + (size_t)t.width * t.height * 3);
                j->w = t.width;
                j->h = t.height;
            }
            st = mp4.add_frames(e.rgb.data(), e.n, &why);
            frames += e.n;
            std::lock_guard<std::mutex> lk(videos_m_);
            j->progress = 90 + (t.frames > 0 ? (int)(10LL * frames / t.frames) : 0);
        }
    }
    if (st < 0) t.cancel.store(true);
    std::string file;
    if (st >= 0 && !aborted && t.status() >= 0 && frames == t.frames)
        st = mp4.finish(&file, &why);
    release(1);

    std::lock_guard<std::mutex> lk(videos_m_);
    const int64_t now = unix_now();
    if (st >= 0 && !aborted && t.status() >= 0 && frames == t.frames) {
        j->status = "completed";
        j->progress = 100;
        j->completed_at = now;
        j->expires_at = now + opt_.video_ttl_s;
        j->mp4 = std::move(file);
        auto sec = [](int64_t a, int64_t b) { return (double)(b - a) / 1e9; };
        const std::string steps = t.t_upscaled_ns > 0
            ? rad::fmt("small steps %.1f s, upscale %.2f s, full steps %.1f s",
                       sec(t.t_cond_ns, t.t_small_ns), sec(t.t_small_ns, t.t_upscaled_ns),
                       sec(t.t_upscaled_ns, t.t_denoised_ns))
            : rad::fmt("steps %.1f s", sec(t.t_cond_ns, t.t_denoised_ns));
        RAD_INFO("video %s: completed, %.1f MB (%s), %.1f s: queue and prompt %.1f s, %s, "
                 "decode %.1f s", j->id.c_str(), (double)j->mp4.size() / 1e6,
                 mp4.video_encoder().c_str(), sec(t.t_submit_ns, t.t_done_ns),
                 sec(t.t_submit_ns, t.t_cond_ns), steps.c_str(), sec(t.t_denoised_ns, t.t_done_ns));
    } else {
        j->status = "failed";
        j->failed = true;
        j->completed_at = now;
        if (st < 0) {
            j->err_code = "encoding_failed";
            j->err_message = "the video could not be written: " + why;
        } else if (aborted || t.status() == RAD_E_CANCELLED) {
            j->err_code = "cancelled";
            j->err_message = aborted ? "the server is shutting down" : "the video was cancelled";
        } else {
            j->err_code = t.status() == RAD_E_INVAL ? "invalid_request" : "generation_failed";
            j->err_message = t.why().empty() ? std::string(rad_strerror(t.status())) : t.why();
        }
        RAD_WARN("video %s: failed: %s", j->id.c_str(), j->err_message.c_str());
    }
    j->worker_done = true;
}

void Server::sweep_videos_locked(int64_t now) {
    /* Deleted jobs whose threads are done are joined. */
    for (size_t i = 0; i < videos_gone_.size();) {
        if (!videos_gone_[i]->worker_done) { ++i; continue; }
        if (videos_gone_[i]->worker.joinable()) videos_gone_[i]->worker.join();
        videos_gone_.erase(videos_gone_.begin() + (std::ptrdiff_t)i);
    }
    /* Expired files go first, then the oldest finished jobs past the count. A running job is
     * never dropped here: it holds an admission place and its thread. */
    auto drop = [&](size_t i) {
        if (videos_[i]->worker.joinable()) videos_[i]->worker.join();
        videos_.erase(videos_.begin() + (std::ptrdiff_t)i);
    };
    for (size_t i = 0; i < videos_.size();) {
        const VideoJob& j = *videos_[i];
        const bool done = j.status == "completed" || j.status == "failed";
        if (done && j.expires_at > 0 && j.expires_at <= now) drop(i);
        else if (done && j.failed && j.completed_at + opt_.video_ttl_s <= now) drop(i);
        else ++i;
    }
    int64_t finished = 0;
    for (const auto& j : videos_) finished += j->status == "completed" || j->status == "failed";
    for (size_t i = 0; i < videos_.size() && finished > opt_.max_videos;) {
        const VideoJob& j = *videos_[i];
        if (j.status == "completed" || j.status == "failed") {
            drop(i);
            --finished;
        } else {
            ++i;
        }
    }
}

void Server::stop_videos() {
    std::vector<std::shared_ptr<VideoJob>> all;
    {
        std::lock_guard<std::mutex> lk(videos_m_);
        all = videos_;
        all.insert(all.end(), videos_gone_.begin(), videos_gone_.end());
    }
    for (auto& j : all) j->ticket->cancel.store(true);
    for (auto& j : all)
        if (j->worker.joinable()) j->worker.join();
}

/* The video object, OpenAI's shape. */
json Server::video_json(const VideoJob& j) {
    json o;
    o["id"] = j.id;
    o["object"] = "video";
    o["model"] = j.model;
    o["status"] = j.status;
    o["progress"] = j.progress;
    o["created_at"] = j.created_at;
    o["completed_at"] = j.completed_at ? json(j.completed_at) : json(nullptr);
    o["expires_at"] = j.expires_at ? json(j.expires_at) : json(nullptr);
    o["prompt"] = j.prompt;
    o["seconds"] = j.seconds;
    o["size"] = j.size;
    o["seed"] = j.seed;
    if (j.refine > 0) {
        o["latent_upscale"] = j.upscale;
        o["upscale_refine_steps"] = j.refine;
        o["upscale_from"] = j.upscale_from;
    }
    if (j.guided) {
        static const char* kGuide[8] = {
            "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
            "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale",
            "audio_guidance_rescale",
        };
        for (int i = 0; i < 8; ++i) o[kGuide[i]] = j.guidance[i];
        if (j.has_negative) o["negative_prompt"] = j.negative;
    }
    if (j.from_image) o["input_reference"] = true;
    if (!j.decoder.empty()) o["decoder"] = j.decoder;
    o["remixed_from_video_id"] = nullptr;
    if (j.failed) {
        json e;
        e["code"] = j.err_code;
        e["message"] = j.err_message;
        o["error"] = e;
    } else {
        o["error"] = nullptr;
    }
    return o;
}

HttpResponse Server::handle_video_get(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::lock_guard<std::mutex> lk(videos_m_);
    for (const auto& j : videos_)
        if (j->id == id) return HttpResponse::json_body(200, json_dump(video_json(*j)));
    ApiError e = err_bad("id", "no video '" + id + "'; it may have expired");
    e.status = 404;
    e.type = "not_found_error";
    return error_response(e, req.peer);
}

HttpResponse Server::handle_video_list(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    int64_t limit = 20;
    if (auto it = req.query.find("limit"); it != req.query.end()) {
        char* e = nullptr;
        limit = std::strtoll(it->second.c_str(), &e, 10);
        if (*e || limit < 1 || limit > 100)
            return error_response(err_bad("limit", "expected an integer in 1..100"), req.peer);
    }
    const auto ord = req.query.find("order");
    const bool asc = ord != req.query.end() && ord->second == "asc";
    if (ord != req.query.end() && ord->second != "asc" && ord->second != "desc")
        return error_response(err_bad("order", "expected asc or desc"), req.peer);
    const auto after = req.query.find("after");
    std::lock_guard<std::mutex> lk(videos_m_);
    sweep_videos_locked(unix_now());
    std::vector<std::shared_ptr<VideoJob>> v = videos_;
    if (!asc) std::reverse(v.begin(), v.end());
    size_t at = 0;
    if (after != req.query.end()) {
        while (at < v.size() && v[at]->id != after->second) ++at;
        if (at == v.size())
            return error_response(err_bad("after", "no video '" + after->second + "'"), req.peer);
        ++at;
    }
    json data = json::array();
    for (size_t i = at; i < v.size() && (int64_t)data.size() < limit; ++i)
        data.push_back(video_json(*v[i]));
    json o;
    o["object"] = "list";
    o["data"] = data;
    o["first_id"] = data.empty() ? json(nullptr) : data.front()["id"];
    o["last_id"] = data.empty() ? json(nullptr) : data.back()["id"];
    o["has_more"] = at + data.size() < v.size();
    return HttpResponse::json_body(200, json_dump(o));
}

HttpResponse Server::handle_video_delete(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::lock_guard<std::mutex> lk(videos_m_);
    std::shared_ptr<VideoJob> j;
    for (size_t i = 0; i < videos_.size(); ++i)
        if (videos_[i]->id == id) {
            j = videos_[i];
            videos_.erase(videos_.begin() + (std::ptrdiff_t)i);
            break;
        }
    if (!j) {
        ApiError e = err_bad("id", "no video '" + id + "'");
        e.status = 404;
        e.type = "not_found_error";
        return error_response(e, req.peer);
    }
    /* A running job is cancelled, and its thread ends once the driver has seen that (at its next
     * pass, which may be seconds away); it is joined then. */
    j->ticket->cancel.store(true);
    videos_gone_.push_back(j);
    sweep_videos_locked(unix_now());
    json o;
    o["id"] = id;
    o["object"] = "video.deleted";
    o["deleted"] = true;
    return HttpResponse::json_body(200, json_dump(o));
}

HttpResponse Server::handle_video_content(const HttpRequest& req) {
    if (!deps_.videos)
        return error_response(err_not_implemented("/v1/videos requires a video generation model"),
                              req.peer);
    const std::string id = req.param("id");
    std::string variant = "video";
    if (auto it = req.query.find("variant"); it != req.query.end()) variant = it->second;
    if (variant != "video" && variant != "thumbnail") {
        ApiError e = err_unsupported("variant", "this server gives the video or its thumbnail");
        e.supported = { "video", "thumbnail" };
        return error_response(e, req.peer);
    }
    std::lock_guard<std::mutex> lk(videos_m_);
    for (const auto& j : videos_) {
        if (j->id != id) continue;
        if (variant == "thumbnail") {
            if (j->first.empty())
                return error_response(err_bad("id", "video '" + id + "' has no frames yet"),
                                      req.peer);
            std::vector<uint8_t> file;
            std::string why;
            if (mm::encode_image(j->first.data(), j->w, j->h, 3, "webp", 90, &file, &why) < 0)
                return error_response(err_internal("the thumbnail could not be encoded: " + why),
                                      req.peer);
            HttpResponse res;
            res.content_type = "image/webp";
            res.body.assign((const char*)file.data(), file.size());
            return res;
        }
        if (j->status != "completed")
            return error_response(err_bad("id", "video '" + id + "' is " + j->status +
                                                    (j->failed ? ": " + j->err_message : "")),
                                  req.peer);
        HttpResponse res;
        res.content_type = "video/mp4";
        res.body = j->mp4;
        return res;
    }
    ApiError e = err_bad("id", "no video '" + id + "'; it may have expired");
    e.status = 404;
    e.type = "not_found_error";
    return error_response(e, req.peer);
}

}  /* namespace server */
}  /* namespace rad */
