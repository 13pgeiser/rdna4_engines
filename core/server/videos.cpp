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
 * A model that takes reference media (VideoGenInfo) also serves SGLang's `task` value `ref2va`:
 * the media are form file parts named `ref_image_N` (at most 9), `ref_video_N` (3), a video's
 * soundtrack as `ref_video_audio_N` and standalone audios as `ref_audio_N` (3), N 1-based and
 * contiguous within each kind. `input_reference` and non-empty `conditions` are still refused:
 * the parts carry the bytes, so no JSON array of media is read.
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
#include "oai.h"

#include "gen/video.h"
#include "mm/media.h"
#include "mm/mp4.h"
#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <thread>
#include <vector>

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
    /* Reference conditioning: the reference parts' names in conditioning order, empty for a
     * text-only job. */
    std::vector<std::string> refs;
    /* Latent upscaling, as the generator resolved it: the factor, the evaluations after it and the
     * small size; refine 0 when the video is not upscaled. */
    double      upscale = 0.0;
    int32_t     refine = 0;
    std::string upscale_from;
    int32_t     w = 0, h = 0;
    bool        worker_done = false;   /* its thread has nothing left to do but return */
};

namespace {

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

/* What a ref2va request may carry whatever a model declares: the reference media the source
 * workflows were built with — nine images, three videos each with an optional soundtrack, three
 * standalone audios. A model's declaration can only lower these. */
constexpr int32_t kMaxRefImages = 9, kMaxRefVideos = 3, kMaxRefAudios = 3;

/* A form file part's name for task ref2va: `ref_image_N`, `ref_video_N`, `ref_video_audio_N`
 * (a video's separately-given soundtrack) or `ref_audio_N`, N plain and 1-based. False for any
 * other name; `ref_video_audio_` is tried before `ref_video_` because one contains the other. */
bool parse_ref_name(const std::string& name, gen::VideoRefMedia::Kind* kind, int32_t* index) {
    static const struct { const char* pfx; gen::VideoRefMedia::Kind kind; } kKinds[] = {
        { "ref_image_",       gen::VideoRefMedia::IMAGE       },
        { "ref_video_audio_", gen::VideoRefMedia::VIDEO_AUDIO },
        { "ref_video_",       gen::VideoRefMedia::VIDEO       },
        { "ref_audio_",       gen::VideoRefMedia::AUDIO       },
    };
    for (const auto& k : kKinds) {
        const size_t n = std::strlen(k.pfx);
        if (name.compare(0, n, k.pfx) != 0 || name.size() == n) continue;
        char* e = nullptr;
        const long v = std::strtol(name.c_str() + n, &e, 10);
        if (*e == '\0' && v >= 1 && v <= 99) { *kind = k.kind; *index = (int32_t)v; return true; }
    }
    return false;
}

/* True when the file ends in one of these extensions, case-insensitively — what the client
 * called the file, which the media readers make the first word of what it holds. */
bool has_ext(const std::string& f, const std::vector<std::string>& exts) {
    const size_t d = f.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = f.substr(d + 1);
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    for (const std::string& x : exts) if (e == x) return true;
    return false;
}

std::string ext_list(const std::vector<std::string>& exts) {
    std::string s;
    for (const std::string& x : exts) { if (!s.empty()) s += ", "; s += "." + x; }
    return s;
}

/* The reference parts of a ref2va request, and their order. Refused with `*err` naming the part
 * when one carries no bytes or an extension the media readers do not take, when a kind is over
 * the API's or the model's count, when a kind's numbers leave a gap or repeat, or when a
 * soundtrack has no video of the same number. On return `refs` is sorted by (kind, number): the
 * order the generator builds the conditioning in. */
int validate_refs(std::vector<gen::VideoRefMedia>& refs, const VideoGenInfo& info, ApiError* err) {
    if (refs.empty()) {
        *err = err_bad("task", "ref2va takes at least one reference part: ref_image_N, "
                               "ref_video_N, ref_video_audio_N or ref_audio_N");
        return -1;
    }
    const struct {
        const char* pfx;
        gen::VideoRefMedia::Kind kind;
        int32_t cap;
        std::vector<std::string> exts;
    } kinds[] = {
        { "ref_image",       gen::VideoRefMedia::IMAGE,
          std::min(kMaxRefImages, (int32_t)info.max_ref_images),
          { "png", "jpg", "jpeg", "webp" } },
        { "ref_video",       gen::VideoRefMedia::VIDEO,
          std::min(kMaxRefVideos, (int32_t)info.max_ref_videos),
          { "mp4", "mov" } },
        { "ref_video_audio", gen::VideoRefMedia::VIDEO_AUDIO,
          std::min(kMaxRefVideos, (int32_t)info.max_ref_videos),
          { "mp4", "mov", "wav", "mp3", "m4a", "aac", "flac", "ogg" } },
        { "ref_audio",       gen::VideoRefMedia::AUDIO,
          std::min(kMaxRefAudios, (int32_t)info.max_ref_audios),
          { "wav", "mp3", "m4a", "aac", "flac", "ogg" } },
    };
    for (const auto& k : kinds) {
        std::vector<bool> got(k.cap + 1, false);
        int n = 0;
        for (const gen::VideoRefMedia& m : refs) {
            if (m.kind != k.kind) continue;
            ++n;
            if (m.data.empty())
                return *err = err_bad(m.name, "'" + m.name + "' carries an empty file"), -1;
            if (!has_ext(m.filename, k.exts))
                return *err = err_bad(m.name, "'" + m.name + "' must be one of " +
                                                  ext_list(k.exts)), -1;
            if (m.index > k.cap)
                return *err = err_bad(m.name, k.cap == 0
                                                  ? "this model takes no parts named " +
                                                    std::string(k.pfx) + "_N"
                                                  : "this model takes at most " +
                                                    std::to_string(k.cap) + " parts named " +
                                                    std::string(k.pfx) + "_N"), -1;
            if (got[m.index])
                return *err = err_bad(m.name, "two parts are numbered " + std::string(k.pfx) +
                                                  "_" + std::to_string(m.index)), -1;
            got[m.index] = true;
        }
        for (int i = 1; i <= n; ++i)
            if (!got[i])
                return *err = err_bad(std::string(k.pfx) + "_" + std::to_string(i),
                                      "the " + std::string(k.pfx) + "_N parts must be numbered 1 "
                                      "to " + std::to_string(n) + " without gaps or repeats"), -1;
    }
    for (const gen::VideoRefMedia& m : refs) {
        if (m.kind != gen::VideoRefMedia::VIDEO_AUDIO) continue;
        bool paired = false;
        for (const gen::VideoRefMedia& v : refs)
            paired = paired || (v.kind == gen::VideoRefMedia::VIDEO && v.index == m.index);
        if (!paired)
            return *err = err_bad(m.name, "'" + m.name + "' is the soundtrack of a reference "
                                          "video; give ref_video_" + std::to_string(m.index) +
                                          " with it"), -1;
    }
    std::stable_sort(refs.begin(), refs.end(),
                     [](const gen::VideoRefMedia& a, const gen::VideoRefMedia& b) {
                         return a.kind != b.kind ? a.kind < b.kind : a.index < b.index;
                     });
    return 0;
}

}  /* namespace */

/* The caps the /v1/models and /server_info capability block reports for a video model. The
 * counts are the ones validate_refs() will apply — the model's declaration under the API's,
 * since a declaration can only lower those — so what a client gates on is what it will meet. */
void Server::video_caps(json& caps, const IVideoGen* gen) {
    const VideoGenInfo& info = gen->info();
    caps["video_max_text"]       = info.max_text;
    caps["video_max_ref_images"] = std::min(kMaxRefImages, (int32_t)info.max_ref_images);
    caps["video_max_ref_videos"] = std::min(kMaxRefVideos, (int32_t)info.max_ref_videos);
    caps["video_max_ref_audios"] = std::min(kMaxRefAudios, (int32_t)info.max_ref_audios);
}

HttpResponse Server::handle_video_create(const HttpRequest& req) {
    if (!deps_.videos || !deps_.tok)
        return error_response(err_not_implemented(
            "/v1/videos requires a video generation model; this one does not make video"),
                              req.peer);
    const VideoGenInfo& info = deps_.videos->info();

    const bool takes_refs = info.max_ref_images > 0 || info.max_ref_videos > 0 ||
                            info.max_ref_audios > 0;
    /* JSON, or a form whose fields are strings (OpenAI's clients send the create call as a form,
     * since it may carry a file). A form may carry the reference media of a ref2va request as
     * named file parts; they are collected here and checked once the task is known. */
    std::vector<gen::VideoRefMedia> refs;
    /* Endpoint keyframes: the "first_frame"/"last_frame" file parts, kept apart from the reference
     * media because they are conditioned on differently (pinned rows, not reference rows) and are
     * validated and encoded on their own path below. At most one of each; together they are FL2VA,
     * alone they are I2VA (first) or L2VA (last). */
    std::vector<gen::VideoRefMedia> keys;
    json body = json::object();
    if (!req.form.empty()) {
        for (const FormPart& p : req.form) {
            if (p.name == "input_reference")
                return error_response(err_unsupported("input_reference", takes_refs
                                                      ? "this server reads reference media as "
                                                        "ref_image_N file parts"
                                                      : "this server makes video from text "
                                                        "only"), req.peer);
            if (p.name == "first_frame" || p.name == "last_frame") {
                if (p.filename.empty())
                    return error_response(err_bad(p.name, "'" + p.name + "' must carry an image "
                                                          "file"), req.peer);
                gen::VideoRefMedia m;
                m.kind = p.name == "first_frame" ? gen::VideoRefMedia::FIRST_FRAME
                                                 : gen::VideoRefMedia::LAST_FRAME;
                m.index = 1;
                m.name = p.name;
                m.filename = p.filename;
                m.content_type = p.content_type;
                m.data = p.data;
                keys.push_back(std::move(m));
                continue;
            }
            if (!p.filename.empty()) {
                gen::VideoRefMedia m;
                if (!parse_ref_name(p.name, &m.kind, &m.index))
                    return error_response(err_bad(p.name, "'" + p.name + "' takes text, not a "
                                                          "file"), req.peer);
                m.name = p.name;
                m.filename = p.filename;
                m.content_type = p.content_type;
                m.data = p.data;
                refs.push_back(std::move(m));
                continue;
            }
            json v = p.data;
            if (p.name == "target" || p.name == "conditions") {
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
    const std::string prompt = body["prompt"].get<std::string>();
    if (has(body, "input_reference"))
        return error_response(err_unsupported("input_reference", takes_refs
                                              ? "this server reads reference media as ref_image_N "
                                                "file parts"
                                              : "this server makes video from text only"),
                              req.peer);
    /* The task: text always; reference conditioning for a model that takes reference media. */
    bool ref2va = false;
    std::string kf_task;   /* an explicitly named keyframe task (i2va / l2va / fl2va), if any */
    if (has(body, "task")) {
        const std::string task = body["task"].is_string() ? body["task"].get<std::string>() : "";
        if (task == "ref2va" && takes_refs) ref2va = true;
        /* The keyframe-completion tasks: a first and/or last frame pinned at the output's ends.
         * They are recognised whatever a model declares (any model with a video encoder can pin a
         * keyframe); which frames are actually present is decided by the file parts, cross-checked
         * against this name below. t2va with a first_frame part is I2VA; a named task missing its
         * frame is refused there. */
        else if (task == "i2va" || task == "l2va" || task == "fl2va") kf_task = task;
        else if (task != "t2va") {
            ApiError e = err_unsupported("task", takes_refs
                                         ? "this server makes video from text, from reference "
                                           "media, or between keyframes: t2va, ref2va, i2va, l2va, "
                                           "fl2va"
                                         : "this server makes video from text or between keyframes: "
                                           "t2va, i2va, l2va, fl2va");
            e.supported = takes_refs
                ? std::vector<std::string>{ "t2va", "ref2va", "i2va", "l2va", "fl2va" }
                : std::vector<std::string>{ "t2va", "i2va", "l2va", "fl2va" };
            return error_response(e, req.peer);
        }
    }
    if (has(body, "conditions") && (!body["conditions"].is_array() || !body["conditions"].empty()))
        return error_response(err_unsupported("conditions", takes_refs
                                              ? "reference media arrive as ref_* file parts; "
                                                "conditions must be empty"
                                              : "this server makes video from text only; "
                                                "conditions must be empty"), req.peer);
    /* The reference parts, in conditioning order, before anything else reads them. Files with no
     * task are a caller's mistake about the task, not about the part. */
    if (!refs.empty() && !ref2va)
        return bad("task", "reference files are given: send them with task ref2va, or send "
                           "none");
    if (ref2va) {
        ApiError e;
        if (validate_refs(refs, info, &e) < 0) return error_response(e, req.peer);
    }
    /* Endpoint keyframes: at most one of each, not alongside reference media (a request either
     * completes between two pinned frames or conditions on references -- the two conditioning
     * paths are different rows), and only on a model that encodes (a keyframe reaches the DiT as a
     * pinned VAE latent). */
    {
        int n_first = 0, n_last = 0;
        for (const gen::VideoRefMedia& m : keys)
            if (m.kind == gen::VideoRefMedia::FIRST_FRAME) ++n_first; else ++n_last;
        if (n_first > 1) return bad("first_frame", "at most one first_frame part");
        if (n_last > 1) return bad("last_frame", "at most one last_frame part");
        if (!keys.empty()) {
            if (!info.encodes)
                return bad("first_frame", "this model has no video encoder, so it cannot pin a "
                                          "keyframe");
            if (!refs.empty())
                return bad("first_frame", "a keyframe completion conditions on its pinned frames, "
                                          "not on reference media: give first_frame/last_frame or "
                                          "the ref_* parts, not both");
        }
        /* A named keyframe task must carry the frame(s) it promises; t2va with parts is fine, the
         * parts alone decide the mode. */
        if (!kf_task.empty()) {
            if ((kf_task == "i2va" || kf_task == "fl2va") && n_first < 1)
                return bad("first_frame", "task '" + kf_task + "' pins a first frame: give a "
                                          "first_frame part");
            if ((kf_task == "l2va" || kf_task == "fl2va") && n_last < 1)
                return bad("last_frame", "task '" + kf_task + "' pins a last frame: give a "
                                         "last_frame part");
        }
    }
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
    if (has_sec || has_dur) {
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
    const int64_t frames = info.frames_per_chunk * chunks + info.frames_tail;

    std::unique_ptr<gen::VideoRequest> r = std::make_unique<gen::VideoRequest>();
    r->width = (int32_t)W;
    r->height = (int32_t)H;
    r->chunks = (int32_t)chunks;
    if (has(body, "num_inference_steps")) {
        double v = 0;
        if (info.fixed_steps && (!number(body["num_inference_steps"], &v) ||
                                 v != (double)info.max_steps))
            return bad("num_inference_steps",
                       "this model runs the schedule it was trained on: " +
                           std::to_string(info.max_steps) + " sigma grid points, the last one 0");
        if (!number(body["num_inference_steps"], &v) || v != std::floor(v) || v < 2 ||
            v > (double)info.max_steps)
            return bad("num_inference_steps", "expected an integer in 2.." +
                                                  std::to_string(info.max_steps) +
                                                  " (sigma grid points, the last one 0)");
        r->steps = (int32_t)v;
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
    r->refs = std::move(refs);
    /* Keyframes ride the reference vector for the driver's sake (its encode queue indexes it), but
     * always AFTER every reference: the driver lays the reference rows down first, then folds each
     * FIRST_FRAME/LAST_FRAME into the pack's keyframe table. first before last, so they keyframe-order
     * 0 and 1 and take image slots 1 and 2. */
    r->refs.insert(r->refs.end(), keys.begin(), keys.end());

    /* ---- the prompt: the template's text with special tokens, the reference media in its
     * presentation order, the prompt last (docs/PORT-REF2VA.md §3: images, then videos each
     * after its soundtrack's label, then the standalone audios). Each `<Picture k>: ` label is
     * encoded as its own string, as the tokenizer the source workflows were checked against sees
     * it; the picture's token run follows its label, taken from the processed item. */
    {
        static const std::string kSlot = "{prompt}";
        const std::string& t = info.tmpl;
        const size_t p = t.find(kSlot);
        auto put = [&](const std::string& s, bool special) {
            if (s.empty()) return;
            const auto ids = deps_.tok->encode(s, false, special);
            r->tokens.insert(r->tokens.end(), ids.begin(), ids.end());
        };
        /* The reference media are presented between the template's head and the prompt. Only the
         * images reach the language model as soft-token rows, so only they are decoded and placed
         * here; a reference video and every audio reach the model through the VAEs (decoded into the
         * encoder's form below), so in the prompt they contribute only their spoken label. That
         * label still rides ahead of the prompt in the reference's presentation order, because the
         * text encoder sees `<Video k>` / `<Audio j>` markers even for media it only names: a
         * soundtrack's `<Audio j>` label is emitted right before its video's `<Video k>` label, and
         * `audio_ord` counts soundtracks and standalone audios together, in the order they are
         * spoken (docs/PORT-REF2VA.md §3). */
        if (ref2va && deps_.mm && deps_.mm->accepts(MediaKind::Image)) {
            /* Every picture is decoded and processed before any token is laid down. A reference
             * video or audio needs no vision item here — it reaches the model through the VAEs, so
             * it contributes only its presentation label to the prompt (the 2 fps timestamped
             * frames the reference also shows Qwen are deferred; see docs/NOTE-REF2VA.md). */
            std::vector<std::shared_ptr<mm::Item>> pics;
            MediaOpts mo;
            mo.max_pixels = (int64_t)W * (int64_t)H;   /* `match`: scaled to the output's area */
            for (const gen::VideoRefMedia& m : r->refs) {
                if (m.kind != gen::VideoRefMedia::IMAGE) continue;
                const std::vector<uint8_t> bytes(m.data.begin(), m.data.end());
                std::shared_ptr<mm::Item> it;
                std::string why;
                if (deps_.mm->prepare(MediaKind::Image, bytes, mo, &it, &why) < 0 || !it)
                    return bad(m.name.c_str(), "'" + m.name + "' could not be used: " + why);
                pics.push_back(std::move(it));
            }
            /* validate_refs numbered each kind 1..n without gaps, so a part's number is its
             * presentation ordinal, and paired every soundtrack to a video of the same number. The
             * reference presentation order (docs/PORT-REF2VA.md §3) is: every picture, then each
             * video behind its soundtrack's label, then the standalone audios. `<Audio j>` counts
             * soundtracks and standalone audios together, in the order they are spoken. */
            auto has_soundtrack = [&](int32_t idx) {
                for (const gen::VideoRefMedia& m : r->refs)
                    if (m.kind == gen::VideoRefMedia::VIDEO_AUDIO && m.index == idx) return true;
                return false;
            };
            put(t.substr(0, p), true);
            std::shared_ptr<mm::PromptMedia> media;
            if (!pics.empty()) media = std::make_shared<mm::PromptMedia>();
            size_t i = 0;
            for (const gen::VideoRefMedia& m : r->refs) {   /* sorted by (kind, index): images first */
                if (m.kind != gen::VideoRefMedia::IMAGE) continue;
                put("<Picture " + std::to_string(m.index) + ">: ", false);
                mm::Placed pl;
                pl.item = pics[i++];
                pl.tok = (int32_t)r->tokens.size();
                r->tokens.insert(r->tokens.end(), pl.item->expand.begin(),
                                 pl.item->expand.end());
                media->items.push_back(std::move(pl));
            }
            int audio_ord = 0;
            for (const gen::VideoRefMedia& m : r->refs) {
                if (m.kind != gen::VideoRefMedia::VIDEO) continue;
                if (has_soundtrack(m.index))
                    put("<Audio " + std::to_string(++audio_ord) + ">: ", false);
                put("<Video " + std::to_string(m.index) + ">: ", false);
            }
            for (const gen::VideoRefMedia& m : r->refs) {
                if (m.kind != gen::VideoRefMedia::AUDIO) continue;
                put("<Audio " + std::to_string(++audio_ord) + ">: ", false);
            }
            if (media) { media->layout(); r->media = std::move(media); }
            put(prompt, false);
            put(t.substr(p + kSlot.size()), true);
        } else {
            put(t.substr(0, p), true);
            put(prompt, false);
            put(t.substr(p + kSlot.size()), true);
        }
        if ((int64_t)r->tokens.size() > info.max_text)
            return bad("prompt", "the prompt" + std::string(ref2va ? " with its reference media"
                                                                    : "") +
                                     " is " + std::to_string(r->tokens.size()) +
                                     " tokens; this model takes at most " +
                                     std::to_string(info.max_text));
    }

    /* ---- the encoder's own view of every reference the VAEs consume: pictures and videos through
     * the video VAE (enc_pixels), soundtracks and standalone audios through the audio VAE
     * (enc_samples). The text encoder saw the pictures above for identity; now each reference is
     * decoded to the encoder's own grid so the driver can stage it into a Vencode or Aencode pass.
     * These are the NATURAL decoded shapes — the driver's encode pass, which owns the encoder's
     * geometry, clip-pads a video's frames and zero-pads an audio's samples to what it reports. The
     * processor decodes under the same guard it used for the text encoder, so a reference that got
     * this far decodes here too. */
    if ((ref2va || !keys.empty()) && info.encodes && deps_.mm) {
        for (gen::VideoRefMedia& m : r->refs) {
            std::string why;
            const uint8_t* p = (const uint8_t*)m.data.data();
            const size_t   n = m.data.size();
            if (m.kind == gen::VideoRefMedia::FIRST_FRAME ||
                m.kind == gen::VideoRefMedia::LAST_FRAME) {
                /* A keyframe anchor resized to the EXACT output canvas: first a plain stretch
                 * ("disabled"), last a centre crop ("center") -- the reference's own resize. Its
                 * latent grid is then the target's, so the pack's pinned rows (frame_rows wide)
                 * match what the encoder writes into the image slot. */
                mm::Frame fr;
                if (deps_.mm->encoder_image_canvas(p, n, (int32_t)W, (int32_t)H,
                                                  m.kind == gen::VideoRefMedia::LAST_FRAME,
                                                  &fr, &why) < 0)
                    return bad(m.name.c_str(), "'" + m.name + "' could not be encoded: " + why);
                m.enc_frames = 1;
                m.enc_h = fr.h;
                m.enc_w = fr.w;
                m.enc_pixels.resize((size_t)fr.w * fr.h * 3);
                for (size_t i = 0; i < m.enc_pixels.size(); ++i)
                    m.enc_pixels[i] = (float)fr.rgb[i];
            } else if (m.kind == gen::VideoRefMedia::IMAGE) {
                mm::Frame fr;
                if (deps_.mm->encoder_image(p, n, (int32_t)info.enc_ratio,
                                            (int32_t)info.max_enc_side, &fr, &why) < 0)
                    return bad(m.name.c_str(), "'" + m.name + "' could not be encoded: " + why);
                m.enc_frames = 1;
                m.enc_h = fr.h;
                m.enc_w = fr.w;
                /* un-normalised RGB 0..255, row-major; the video VAE applies ImageNet mean/std itself */
                m.enc_pixels.resize((size_t)fr.w * fr.h * 3);
                for (size_t i = 0; i < m.enc_pixels.size(); ++i)
                    m.enc_pixels[i] = (float)fr.rgb[i];
            } else if (m.kind == gen::VideoRefMedia::VIDEO) {
                /* The frames the temporal VAE sees, sampled and resized to the encoder grid. The
                 * count is the reference's trimmed length (n % 17 == 5); the driver's encode pass
                 * repeats the final frame out to a whole clip, so this stays the natural count. */
                std::vector<mm::Frame> fr;
                if (deps_.mm->encoder_video(p, n, (int32_t)info.enc_ratio,
                                            (int32_t)info.max_enc_side, (int32_t)info.max_enc_frames,
                                            &fr, &why) < 0 || fr.empty())
                    return bad(m.name.c_str(), "'" + m.name + "' could not be encoded: " + why);
                m.enc_frames = (int32_t)fr.size();
                m.enc_h = fr.front().h;
                m.enc_w = fr.front().w;
                const size_t area = (size_t)m.enc_w * m.enc_h * 3;
                m.enc_pixels.resize(area * fr.size());
                for (size_t f = 0; f < fr.size(); ++f)
                    for (size_t i = 0; i < area; ++i)
                        m.enc_pixels[f * area + i] = (float)fr[f].rgb[i];
            } else {
                /* A reference audio, or a reference video's soundtrack: mono at the model's audio
                 * rate, then spread across the encoder's channels (the audio VAE encodes one
                 * channel a time; a mono source becomes both). The driver zero-pads to a hop. */
                std::vector<float> mono;
                if (deps_.mm->encoder_audio(p, n, (int32_t)info.sample_rate, info.max_enc_samples,
                                            &mono, &why) < 0 || mono.empty())
                    return bad(m.name.c_str(), "'" + m.name + "' could not be decoded: " + why);
                const int32_t ch = (int32_t)info.channels;
                m.enc_channels = ch;
                m.enc_n = (int64_t)mono.size();
                m.enc_samples.assign((size_t)ch * mono.size(), 0.0f);
                for (int32_t c = 0; c < ch; ++c)
                    for (size_t i = 0; i < mono.size(); ++i)
                        m.enc_samples[(size_t)c * mono.size() + i] = mono[i];
            }
        }
    }

    /* ---- admitted and queued: the job holds its place until its file is written */
    if (auto no = admit(1, req.peer)) return *no;
    auto j = std::make_shared<VideoJob>();
    j->id = new_video_id();
    j->model = opt_.model_id;
    j->prompt = prompt;
    j->seconds = rad::fmt("%.2f", (double)frames / fps);
    j->size = std::to_string(W) + "x" + std::to_string(H);
    j->created_at = unix_now();
    j->seed = r->seed;
    for (const gen::VideoRefMedia& m : r->refs) j->refs.push_back(m.name);
    j->ticket = std::make_shared<gen::VideoTicket>();
    r->ticket = j->ticket;
    const int64_t n_tok = (int64_t)r->tokens.size();
    const int32_t steps = r->steps;
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
    RAD_INFO("video %s: %s, %lld frames (%s s), %lld prompt tokens, %d steps%s%s, seed=%llu",
             j->id.c_str(), j->size.c_str(), (long long)frames, j->seconds.c_str(),
             (long long)n_tok, steps > 0 ? steps : (int)tk.steps,
             tk.refine > 0 ? rad::fmt(" (the first %d at %dx%d, upscaled for the last %d)",
                                      (int)(tk.steps - tk.refine), (int)tk.small_width,
                                      (int)tk.small_height, (int)tk.refine).c_str()
                           : "",
             j->refs.empty() ? "" : rad::fmt(", %zu reference parts", j->refs.size()).c_str(),
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
    if (!j.refs.empty()) {
        o["task"] = "ref2va";
        o["refs"] = j.refs;   /* the reference parts, in conditioning order */
    }
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
