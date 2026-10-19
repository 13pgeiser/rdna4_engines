/* video_driver.cpp -- the video generator's pass driver (video_driver.h), the ticket a caller
 * waits on, and a request's shape and schedule (video.h). */
#include "video_driver.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace gen {

/* ================================================================== the ticket */
void VideoTicket::push(VideoEvent&& e) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(e));
    }
    cv_.notify_all();
}

void VideoTicket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

bool VideoTicket::next(VideoEvent* out, int ms, bool* timed_out) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timed_out) *timed_out = false;
    if (q_.empty() && !done_)
        cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return !q_.empty() || done_; });
    if (!q_.empty()) {
        *out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    if (!done_ && timed_out) *timed_out = true;
    return false;
}

bool VideoTicket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_ && q_.empty();
}

int VideoTicket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string VideoTicket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

/* ================================================================== shape and schedule */
VideoShape video_shape(const RadVideoGenDecl& d, int32_t width, int32_t height, int32_t chunks) {
    VideoShape s;
    s.frames = (int32_t)(d.frames_per_chunk * chunks + d.frames_tail);
    s.lat_frames = (int32_t)(d.latents_per_chunk * chunks + d.latents_tail);
    s.lat_h = (int32_t)(height / d.latent_scale);
    s.lat_w = (int32_t)(width / d.latent_scale);
    /* round(frames / fps * audio_rate), ties to even as the reference's float round; with an
     * integer numerator and denominator a tie is exact. */
    const int64_t num = (int64_t)s.frames * d.fps_den * d.audio_rate, den = d.fps_num;
    int64_t q = num / den;
    const int64_t r2 = 2 * (num % den);
    if (r2 > den || (r2 == den && (q & 1))) ++q;
    s.n_audio = (int32_t)q;
    s.video_rows = (int64_t)s.lat_frames * (s.lat_h / d.patch) * (s.lat_w / d.patch);
    s.audio_rows = d.audio_channels * (int64_t)s.n_audio;
    return s;
}

void video_small_size(const RadVideoGenDecl& d, int32_t width, int32_t height, double factor,
                      int32_t* small_w, int32_t* small_h) {
    const int64_t m = d.size_multiple;
    auto side = [&](int32_t full) {
        const int64_t k = (int64_t)std::floor((double)full / factor / (double)m + 0.5);
        return (int32_t)(std::min<int64_t>(std::max<int64_t>(k, 1), full / m) * m);
    };
    *small_w = side(width);
    *small_h = side(height);
}

/* The shift of one base time, in f32 as the reference's tensor arithmetic. */
static float shift_sigma(double shift, float b) {
    const float sh = (float)shift, sh1 = (float)(shift - 1.0);
    return (sh * b) / (1.0f + sh1 * b);
}

/* torch.linspace(1, 0, n) in f32, which computes the first half up from the start and the
 * second half down from the end, then the exponential shift, consecutive duplicates dropped. */
static std::vector<float> shifted_sigmas(double shift, int64_t n) {
    std::vector<float> s;
    if (n < 2) return s;
    const float start = 1.0f, end = 0.0f;
    const float step = (end - start) / (float)(n - 1);
    const int64_t half = n / 2;
    for (int64_t i = 0; i < n; ++i) {
        const float b = i < half ? start + step * (float)i : end - step * (float)(n - i - 1);
        const float v = shift_sigma(shift, b);
        if (s.empty() || v != s.back()) s.push_back(v);
    }
    return s;
}

/* A trained schedule: each rung taken to f32 and shifted, then the terminal 0. */
static std::vector<float> rung_sigmas(double shift, const double* rungs, int32_t n) {
    std::vector<float> s;
    for (int32_t i = 0; i < n; ++i) s.push_back(shift_sigma(shift, (float)rungs[i]));
    s.push_back(0.0f);
    return s;
}

VideoSchedule video_schedule(const RadVideoGenDecl& d, int32_t steps) {
    VideoSchedule v;
    if (d.n_rungs > 0) {
        if (steps != d.n_rungs + 1) return v;
        v.sig_v = rung_sigmas(d.shift_video, d.rungs, d.n_rungs);
        v.sig_a = rung_sigmas(d.shift_audio, d.rungs, d.n_rungs);
        v.steps = d.n_rungs;
        return v;
    }
    v.sig_v = shifted_sigmas(d.shift_video, steps);
    v.sig_a = shifted_sigmas(d.shift_audio, steps);
    const size_t n = std::min(v.sig_v.size(), v.sig_a.size());
    v.steps = n > 0 ? (int32_t)(n - 1) : 0;
    return v;
}

/* ================================================================== sizing */
namespace {

constexpr int64_t kAlign = 256;

/* The most segment boundaries R_SEG holds: the plain presentation has four, and a full
 * reference-conditioned one (text, a keyframe pair, two segments for every reference at the
 * server's 9/3/3 caps, the two target streams) stays well under this. The plugin's pack is the
 * only source once references ride; the VSA kernels take the count from the staged rows and
 * accept up to 4096. */
constexpr int64_t kSegCap = 64;

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

/* Region offsets and sizes, in the enum's order (video_driver.h), and the total; shared by the
 * allocation and the VRAM budget. `ep` and `pd` are the conditioning encoder's pass bound and
 * patch width (RadBuildCtx::max_enc_patches, RadEncoderDecl::patch_dim), 0 and 0 without a
 * vision encoder: the media regions then hold nothing. */
int64_t carve(const RadVideoGenDecl& d, int64_t max_tok, int64_t ep, int64_t pd,
              int64_t (*bytes)[2], int n) {
    const int64_t want[] = {
        d.max_text * 4,          /* R_TOK */
        d.max_text * 4,          /* R_POS */
        2 * 4,                   /* R_CU */
        3 * max_tok * 4,         /* R_ROPE: a step's rotary positions, a conditioning's [3,T] M-RoPE */
        RAD_VGEOM_N * 4,         /* R_GEOM */
        6 * 4,                   /* R_SCHED */
        d.max_steps * 4,         /* R_TV */
        d.max_steps * 4,         /* R_TA */
        4 * 4,                   /* R_SEED */
        4,                       /* R_LANE */
        4,                       /* R_FIRST */
        kSegCap * 4,             /* R_SEG: the pack's boundary table, widened for references */
        4 * 4,                   /* R_LAT */
        4,                       /* R_UP */
        d.max_text * 4,          /* R_MMROW: the media rows, one per placeholder token */
        ep * pd * 2,             /* R_ENCPIX: [ep, pd] bf16 */
        4 * ep * 4,              /* R_ENCCRD: [4, ep] i32 component-major */
        (ep + 2) * 4,            /* R_ENCCU: segment boundaries; a segment is at least one patch */
        /* The VAE encode staging: one media item's pixels or samples and its encode geometry,
         * sized to the declaration's per-pass ceilings so a single pass holds any media the
         * export accepts. All 0 without the encoders (decl.encodes 0 leaves the ceilings 0). */
        d.max_enc_frames * d.max_enc_side * d.max_enc_side * 3 * 4,   /* R_ENCPX: [f,h,w,3] f32 */
        d.max_enc_samples * d.audio_channels * 4,                     /* R_ENCSM: [ch,n] f32 */
        25 * 4,                        /* R_ENCG: rad_arch_encode_geom's 25 i32 (the video geom) */
        kSegCap * 4,                   /* R_MOD: the merged modulation-group boundaries (<= R_SEG) */
    };
    int64_t off = 0;
    for (int r = 0; r < n; ++r) {
        bytes[r][0] = off;
        bytes[r][1] = want[r];
        off = align_up(off + want[r]);
    }
    return off;
}

int64_t frames_bytes_of(const RadVideoGenDecl& d) {
    return (d.frames_per_chunk + d.frames_tail) * d.max_area * 3;
}

/* The samples a channel of a soundtrack under `frames` frames: its latents rounded up, which
 * bounds the rounding video_shape does. */
int64_t samples_of(const RadVideoGenDecl& d, int32_t frames) {
    const int64_t num = (int64_t)frames * d.fps_den * d.audio_rate;
    return (num / d.fps_num + 1) * (d.sample_rate / d.audio_rate);
}

}  /* namespace */

/* The conditioning encoder's staging bounds: a program with no declared vision encoder stages
 * no media, and the driver refuses requests that carry any. */
namespace {
void encoder_bounds(const Program& prog, int64_t* ep, int64_t* pd) {
    *ep = prog.encoder.modalities != 0 ? prog.ctx.max_enc_patches : 0;
    *pd = prog.encoder.modalities != 0 ? prog.encoder.patch_dim : 0;
}
}  /* namespace */

int64_t VideoDriver::device_bytes(const Program& prog) {
    if (!prog.is_video_generator()) return 0;
    int64_t ep = 0, pd = 0;
    encoder_bounds(prog, &ep, &pd);
    int64_t b[R_N][2];
    return carve(prog.video_gen, prog.ctx.max_tok, ep, pd, b, R_N);
}

int VideoDriver::init(const Program& prog, PositionsFn positions,
                      const std::vector<RankIO>& ranks, SegsFn segs, PackFn pack,
                      ModSegsFn mod_segs, EncodeGeomFn enc_geom) {
    fini();
    if (!prog.is_video_generator()) return RAD_E_STATE;
    decl_  = prog.video_gen;
    rungs_ = prog.vgen_str.rungs;
    decl_.rungs = rungs_.empty() ? nullptr : rungs_.data();
    name_  = prog.vgen_str.name;
    ranks_ = ranks;
    positions_ = positions;
    segs_ = segs;
    pack_ = pack;
    mod_segs_ = mod_segs;
    enc_geom_ = enc_geom;
    max_tok_ = prog.ctx.max_tok;
    encoder_bounds(prog, &max_enc_, &patch_dim_);
    if (!positions_) {
        RAD_ERR("video generator '%s': the plugin exports no rad_arch_video_positions, which a "
                "video generator must", name_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    if (max_tok_ < decl_.max_text) {
        RAD_ERR("video generator '%s': a pass carries %lld rows and a prompt may be %lld tokens",
                name_.c_str(), (long long)max_tok_, (long long)decl_.max_text);
        return RAD_E_UNSUPPORTED;
    }
    int64_t b[R_N][2];
    bytes_ = carve(decl_, max_tok_, max_enc_, patch_dim_, b, R_N);
    for (int r = 0; r < R_N; ++r) { reg_[r].off = b[r][0]; reg_[r].bytes = b[r][1]; reg_[r].used = 0; }

    frames_bytes_ = frames_bytes_of(decl_);
    audio_bytes_ = samples_of(decl_, (int32_t)(decl_.frames_per_chunk * decl_.max_chunks +
                                               decl_.frames_tail)) *
                   decl_.audio_channels * 4;
    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    frames_host_ = static_cast<uint8_t*>(rad_dev_alloc(frames_bytes_, RAD_MEM_HOST_PINNED));
    audio_host_ = static_cast<float*>(rad_dev_alloc(audio_bytes_, RAD_MEM_HOST_PINNED));
    if (!host_ || !frames_host_ || !audio_host_) {
        RAD_ERR("video generator: no pinned host memory for the pass staging and outputs (%lld + "
                "%lld + %lld bytes)", (long long)bytes_, (long long)frames_bytes_,
                (long long)audio_bytes_);
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("video generator: rank %zu has no room for the pass staging (%lld bytes)", r,
                    (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    vout_.assign(ranks_.size(), RadVideoGenBatch{});
    free_lanes_.clear();
    for (int64_t i = decl_.max_lanes - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    /* Debug-only readback of the encoder outputs (R4D_ENC_DEBUG). Only when the environment names
     * it AND the container carries encoders; a normal run leaves enc_dbg_ false and never reads
     * decl_.dbg_* or touches enc_dbg_host_. */
    if (std::getenv("R4D_ENC_DEBUG") &&
        (decl_.dbg_enca_bytes > 0 || decl_.dbg_evc_bytes > 0)) {
        enc_dbg_bytes_ = std::max(decl_.dbg_enca_bytes, decl_.dbg_evc_bytes);
        enc_dbg_host_ = static_cast<char*>(rad_dev_alloc(enc_dbg_bytes_, RAD_MEM_HOST_PINNED));
        enc_dbg_ = enc_dbg_host_ != nullptr;
        if (!enc_dbg_)
            RAD_ERR("video generator: R4D_ENC_DEBUG set but no pinned host for the %lld-byte "
                    "encoder readback", (long long)enc_dbg_bytes_);
    }
    return RAD_OK;
}

void VideoDriver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    if (frames_host_) rad_dev_free(frames_host_, RAD_MEM_HOST_PINNED);
    if (audio_host_) rad_dev_free(audio_host_, RAD_MEM_HOST_PINNED);
    if (enc_dbg_host_) rad_dev_free(enc_dbg_host_, RAD_MEM_HOST_PINNED);
    host_ = nullptr;
    frames_host_ = nullptr;
    audio_host_ = nullptr;
    enc_dbg_host_ = nullptr;
}

/* ================================================================== requests */
int VideoDriver::submit(std::unique_ptr<VideoRequest> r, std::string* why) {
    if (!r || !r->ticket) return RAD_E_INVAL;
    const RadVideoGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int64_t T = (int64_t)r->tokens.size();
    if (T < 1 || T > d.max_text)
        return no("the prompt is " + std::to_string(T) + " tokens; this model takes 1.." +
                  std::to_string(d.max_text));
    /* Reference conditioning reaches the model two ways. Reference images enter the language model
     * through `media` (the server built it from the parts, gen/video.h; build_cond stages it into
     * the conditioning pass) AND contribute VAE rows. A reference video, its soundtrack, and a
     * standalone audio reach the model only as VAE rows: the server decodes them into the encoder's
     * own form, one encode pass each folds their latents into the lane's fixed slots, and the step's
     * pack lays those slots between the text and the target streams (RadVideoGenBatch::ref_tab).
     * Every kind the declaration allows is consumed; the guard below is only about a container
     * lacking its vision tower, which could encode nothing. */
    if (!r->refs.empty() && max_enc_ == 0)
        return no("this container was converted without its vision tower, so no request can "
                  "carry reference media");
    /* A reference video, soundtrack or standalone audio reaches the model only through the VAE
     * encoders (the video/audio Vencode/Aencode passes), which a container carries exactly when it
     * declares `encodes`. Converted without them there is nothing to condition on, and the reference
     * would drop silently into a text-only video — the very silent weakening this guard exists to
     * prevent — so refuse it, naming the part. Images need only the tower and are checked below. */
    if (!d.encodes)
        for (const VideoRefMedia& m : r->refs)
            if (m.kind != VideoRefMedia::IMAGE)
                return no("'" + m.name + "' is a reference " +
                          (m.kind == VideoRefMedia::VIDEO ? "video"
                           : m.kind == VideoRefMedia::VIDEO_AUDIO ? "soundtrack" : "audio") +
                          ", but this container was converted without its VAE encoders, so nothing "
                          "can condition on it");
    /* Only reference images add language-model soft-token rows, so only they need processed media.
     * A reference video or audio reaches the model purely through the VAE encode rows the pack lays
     * from the reference table, so a request of those alone carries no `media` and needs none. */
    bool has_image = false;
    for (const VideoRefMedia& m : r->refs)
        if (m.kind == VideoRefMedia::IMAGE) has_image = true;
    if (has_image && !r->media)
        return no("the request carries reference images but no processed media to condition on");
    if (r->media) {
        int64_t n_patch = 0;
        for (const mm::Placed& p : r->media->items) {
            if ((int64_t)p.tok + (int64_t)p.item->expand.size() > T)
                return no("a reference's placeholder run is placed past the end of the prompt");
            n_patch += p.item->n_patch;
        }
        if (n_patch > max_enc_)
            return no("the references are " + std::to_string(n_patch) + " patches; this model "
                      "encodes at most " + std::to_string(max_enc_) + " in one pass");
    }
    if (r->chunks < d.min_chunks || r->chunks > d.max_chunks)
        return no("the length must be " + std::to_string(d.min_chunks) + ".." +
                  std::to_string(d.max_chunks) + " chunks");
    const int64_t W = r->width, H = r->height;
    if (W % d.size_multiple || H % d.size_multiple || W < d.min_side || H < d.min_side ||
        W * H > d.max_area || std::max(W, H) > d.max_aspect * std::min(W, H))
        return no("the size " + std::to_string(W) + "x" + std::to_string(H) + " is not one this "
                  "model makes: sides multiples of " + std::to_string(d.size_multiple) + " from " +
                  std::to_string(d.min_side) + ", at most " + std::to_string(d.max_area) +
                  " pixels, the long side at most " + std::to_string(d.max_aspect) +
                  " times the short");
    const int32_t steps = r->steps > 0 ? r->steps : (int32_t)d.default_steps;
    if (d.n_rungs > 0 && steps != d.n_rungs + 1)
        return no("num_inference_steps must be " + std::to_string(d.n_rungs + 1) + ": this model "
                  "runs the schedule it was trained on, " + std::to_string(d.n_rungs) +
                  " evaluations");
    if (steps < 2 || steps > d.max_steps)
        return no("num_inference_steps must be 2.." + std::to_string(d.max_steps) + " (sigma grid "
                  "points, the last one 0)");

    auto j = std::make_unique<Job>();
    const VideoShape s = video_shape(d, r->width, r->height, r->chunks);
    j->geom.n_text = (int32_t)T;
    j->geom.frames = s.frames;
    j->geom.lat_frames = s.lat_frames;
    j->geom.lat_h = s.lat_h;
    j->geom.lat_w = s.lat_w;
    j->geom.n_audio = s.n_audio;
    /* The pack's reference table (RadVideoGenBatch::ref_tab): every reference the DiT conditions
     * on, in presentation order, in ITS OWN latent grid -- {kind, slot, h, w, frames, audio} six
     * words after the reference and keyframe counts. `kind` is a pack Ref2VARefKind (picture 0,
     * standalone audio 1, video 2), the server's VIDEO_AUDIO being folded into its video's `audio`
     * word rather than a row of its own. Each reference's slot is the one its encode pass folds its
     * latents into, and its latent counts come from the encoder's own geometry, not hand arithmetic:
     * a picture is one latent frame (its grid the encoder's sides over the VAE ratio); a video's
     * frames and its soundtrack's are the Vencode/Aencode latent counts rad_arch_encode_geom gives
     * (a video's are the same before and after the encode pass clip-pads its short final clip). The
     * table must exist before the step layout is decided, because when the denoiser consumes it the
     * reference rows are part of that layout. The keyframe count stays 0 (continuation is separate
     * work), so the table holds only reference rows. */
    if (d.encodes) {
        bool need_geom = false;
        for (const VideoRefMedia& m : r->refs)
            if (m.kind != VideoRefMedia::IMAGE) need_geom = true;
        if (need_geom && !enc_geom_)
            return no("the request carries a reference video or audio, but this plugin exports no "
                      "rad_arch_encode_geom, so their encoder geometry cannot be sized");
        int32_t eg[25];
        /* A reference's latent frame counts, straight from the encoder's geometry. */
        auto video_lat = [&](const VideoRefMedia& m) -> int64_t {
            int64_t in[4] = { m.enc_frames, m.enc_h, m.enc_w, 3 };
            return enc_geom_(RAD_GEN_PASS_VENCODE, in, eg, 25) < 0 ? -1 : (int64_t)eg[24];
        };
        auto audio_lat = [&](const VideoRefMedia& m) -> int64_t {
            int64_t in[4] = { m.enc_n, 0, 0, m.enc_channels };
            return enc_geom_(RAD_GEN_PASS_AENCODE, in, eg, 25) < 0 ? -1 : (int64_t)eg[12];
        };
        /* Pictures and videos share a side-grid check against the VAE ratio. */
        auto bad_side = [&](const VideoRefMedia& m) {
            return d.enc_ratio < 1 || m.enc_h < d.enc_ratio || m.enc_w < d.enc_ratio ||
                   m.enc_h % d.enc_ratio || m.enc_w % d.enc_ratio;
        };
        for (size_t i = 0; i < r->refs.size(); ++i) {
            const VideoRefMedia& m = r->refs[i];
            if (m.kind == VideoRefMedia::IMAGE) {
                if (m.enc_pixels.empty())
                    return no("'" + m.name + "' was not prepared for the video encoder");
                if (m.index < 1)
                    return no("'" + m.name + "' has no slot in the conditioning");
                if (bad_side(m))
                    return no("'" + m.name + "' has encoder sides (" + std::to_string(m.enc_h) +
                              "x" + std::to_string(m.enc_w) + ") the VAE grid of " +
                              std::to_string(d.enc_ratio) + " does not divide");
                j->enc_q.push_back({RAD_ENC_IMAGE, m.index, (int32_t)i});
                if (j->ref_tab.empty()) j->ref_tab.assign(2, 0);   /* [n_refs, n_keys] */
                ++j->ref_tab[0];
                j->ref_tab.insert(j->ref_tab.end(),
                                  {0 /*a picture*/, m.index, (int32_t)(m.enc_h / d.enc_ratio),
                                   (int32_t)(m.enc_w / d.enc_ratio), 1 /*one latent frame*/,
                                   0 /*no soundtrack*/});
            } else if (m.kind == VideoRefMedia::VIDEO) {
                if (m.enc_pixels.empty())
                    return no("'" + m.name + "' was not prepared for the video encoder");
                if (m.index < 1 || m.index > 3)
                    return no("'" + m.name + "' has no video slot (1..3)");
                if (bad_side(m))
                    return no("'" + m.name + "' has encoder sides (" + std::to_string(m.enc_h) +
                              "x" + std::to_string(m.enc_w) + ") the VAE grid of " +
                              std::to_string(d.enc_ratio) + " does not divide");
                /* A soundtrack given for this video (ref_video_audio_N) is folded into THIS row's
                 * `audio` word, not a row of its own; the pack lays its block right before the
                 * video's and the scatter reads the two back as the pair they are. It is encoded at
                 * the VIDEO's slot, since aenc_slot_row numbers a soundtrack by its video. */
                int64_t a_t = 0;
                for (size_t k = 0; k < r->refs.size(); ++k) {
                    const VideoRefMedia& a = r->refs[k];
                    if (a.kind != VideoRefMedia::VIDEO_AUDIO || a.index != m.index) continue;
                    if (a.enc_samples.empty() || a.enc_n < 1 || a.enc_channels < 1)
                        return no("'" + a.name + "' (the soundtrack of '" + m.name +
                                  "') was not prepared for the audio encoder");
                    a_t = audio_lat(a);
                    if (a_t < 1)
                        return no("the audio encoder refuses the soundtrack '" + a.name + "'");
                    j->enc_q.push_back({RAD_ENC_VAUDIO, m.index, (int32_t)k});
                }
                const int64_t v_t = video_lat(m);
                if (v_t < 1) return no("the video encoder refuses '" + m.name + "'");
                j->enc_q.push_back({RAD_ENC_VIDEO, m.index, (int32_t)i});
                if (j->ref_tab.empty()) j->ref_tab.assign(2, 0);
                ++j->ref_tab[0];
                j->ref_tab.insert(j->ref_tab.end(),
                                  {2 /*a video*/, m.index, (int32_t)(m.enc_h / d.enc_ratio),
                                   (int32_t)(m.enc_w / d.enc_ratio), (int32_t)v_t, (int32_t)a_t});
            } else if (m.kind == VideoRefMedia::AUDIO) {
                if (m.enc_samples.empty() || m.enc_n < 1 || m.enc_channels < 1)
                    return no("'" + m.name + "' was not prepared for the audio encoder");
                if (m.index < 1 || m.index > 3)
                    return no("'" + m.name + "' has no audio slot (1..3)");
                const int64_t a_t = audio_lat(m);
                if (a_t < 1) return no("the audio encoder refuses '" + m.name + "'");
                j->enc_q.push_back({RAD_ENC_AUDIO, m.index, (int32_t)i});
                if (j->ref_tab.empty()) j->ref_tab.assign(2, 0);
                ++j->ref_tab[0];
                j->ref_tab.insert(j->ref_tab.end(),
                                  {1 /*a standalone audio*/, m.index, 0, 0, 0, (int32_t)a_t});
            }
            /* VIDEO_AUDIO rows are folded into their video above, so they take no row here. */
        }
    }
    /* The step's packed sequence. When the denoiser consumes the reference table the pack is the
     * only source of its rows, segment boundaries and positions -- the reference rows sit between
     * the text and the target streams, so the plain `text + audio + video` arithmetic and the plain
     * position layout no longer describe it. rad_arch_video_pack returns the total rows and, when
     * roomy, the boundaries and the full [3, rows] planar RoPE in one call. Otherwise the plain
     * layout is computed exactly as before, and a staged reference table (if any) rides inert. */
    const int64_t nd = positions_(RAD_VPOS_DECODE, &j->geom, nullptr, 0);
    if (nd <= 0 || nd > max_tok_)
        return no("the model lays this video's decoder in " + std::to_string(nd) + " rows, which "
                  "this server cannot run");
    j->dec_rows = nd;
    if (d.ref_rows && !j->ref_tab.empty()) {
        if (r->upscale > 1.0)
            return no("reference conditioning does not yet combine with latent upscale; ask for "
                      "the size you want without latent_upscale");
        if (!pack_)
            return no("the generator says it conditions on references but exports no "
                      "rad_arch_video_pack");
        const int64_t pr = pack_(&j->geom, j->ref_tab.data(), nullptr, 0, nullptr, 0);
        if (pr <= 0)
            return no("the model refused the reference conditioning layout");
        if (pr > max_tok_)
            return no("a conditioned step is " + std::to_string(pr) + " rows and a pass carries " +
                      std::to_string(max_tok_) + "; ask for a shorter or smaller video, or fewer "
                      "references");
        j->seg.assign((size_t)kSegCap, 0);
        j->pos.assign((size_t)(3 * pr), 0.0f);
        if (pack_(&j->geom, j->ref_tab.data(), j->seg.data(), kSegCap, j->pos.data(), 3 * pr) != pr)
            return no("the model could not lay the reference conditioning out");
        int nseg = -1;
        for (int k = 0; k < kSegCap; ++k)
            if (j->seg[(size_t)k] == (int32_t)pr) { nseg = k + 1; break; }   /* video grid ends here */
        if (nseg < 0 || nseg > kSegCap)
            return no("the reference conditioning layout has no closing segment boundary");
        j->seg.resize((size_t)nseg);
        j->nseg = nseg;
        /* The modulation groups. A modulation cell is fixed by a segment's kind, so the denoiser
         * merges runs of consecutive same-cell segments and pitches its fixed-width table by the
         * GROUP count, not the reference count -- which is how nine reference photos (four groups:
         * prompt, all-photos, soundtrack, video) ride a table the fine layout (twelve segments)
         * would overrun. The plugin merges for us (rad_arch_video_mod_segs); one that exports none
         * merges nothing, so the group table is the fine one and the older per-segment ceiling
         * applies. Refuse over-wide merges here, at submit, where a clean answer is still possible. */
        int nmod = -1;
        if (mod_segs_) {
            j->mod_seg.assign((size_t)kSegCap, 0);
            nmod = (int)mod_segs_(&j->geom, j->ref_tab.data(), j->mod_seg.data(), kSegCap);
            if (nmod < 0 || nmod > kSegCap)
                return no("the model could not merge the reference conditioning into its "
                          "modulation groups");
            j->mod_seg.resize((size_t)nmod);
        } else {
            nmod = nseg;                 /* nothing merges: the groups are the fine segments */
            j->mod_seg = j->seg;
        }
        j->nmod = nmod;
        const int ngroups = nmod - 1;    /* `mod_seg` is boundaries: one more than the groups */
        if (d.max_ref_segs > 0 && ngroups > d.max_ref_segs)
            return no("these references change the conditioning's modulation cell " +
                      std::to_string(ngroups) + " times, but this deployment's modulation table holds " +
                      std::to_string(d.max_ref_segs) + " cells -- ask for references that share their "
                      "modalities and target, or fewer of them");
        /* `rows` stays the plain text+audio+video count: every non-step pass (encode, decode,
         * audio, upscale) stages its own segment table and target counts from it and must see the
         * bytes it sees today. Only build_step runs at the widened `step_rows`. */
        j->rows = T + s.audio_rows + s.video_rows;
        j->step_rows = pr;
        /* The last segment (the video target) runs [seg[nseg-2], seg[nseg-1]); seg[nseg-1] is the
         * total rows, so the video start is the boundary before it -- the same one the VSA takes as
         * its video base (it works in segments, one fewer than the boundaries it is handed). */
        j->lat_v = pr - j->seg[(size_t)(nseg - 2)];
        j->lat_a = decl_.audio_channels * (int64_t)j->geom.n_audio;
        j->ref_wide = true;
    } else {
        j->rows = T + s.audio_rows + s.video_rows;
        if (j->rows > max_tok_)
            return no("a step of this video is " + std::to_string(j->rows) + " rows and a pass "
                      "carries " + std::to_string(max_tok_) + "; ask for a shorter or smaller video");
        const int64_t nr = positions_(RAD_VPOS_STEP, &j->geom, nullptr, 0);
        if (nr != j->rows)
            return no("the model lays this video out in " + std::to_string(nr) + " step rows, "
                      "which this server cannot run");
        j->pos.assign((size_t)(3 * nr), 0.0f);
        if (positions_(RAD_VPOS_STEP, &j->geom, j->pos.data(), nr) != nr)
            return no("the model could not lay this video out");
        j->step_rows = j->rows;
    }
    j->sched = video_schedule(d, steps);
    if (j->sched.steps < 1) return no("the schedule has no steps");
    if (r->upscale > 1.0) {
        if (!(d.upscale_max > 1.0)) return no("this model has no latent upscaler");
        if (r->upscale > d.upscale_max)
            return no("latent_upscale is at most " + fmt("%g", d.upscale_max) + " for this model");
        int32_t sw = 0, sh = 0;
        video_small_size(d, r->width, r->height, r->upscale, &sw, &sh);
        if (sw == r->width && sh == r->height)
            return no("latent_upscale " + fmt("%g", r->upscale) + " leaves " +
                      std::to_string(r->width) + "x" + std::to_string(r->height) + " as it is: "
                      "the small size is each side over the factor, rounded to a multiple of " +
                      std::to_string(d.size_multiple));
        const int32_t evals = j->sched.steps;
        int32_t refine = r->refine;
        if (refine == 0) refine = std::min<int32_t>(d.upscale_refine, evals - 1);
        if (refine < 1 || refine >= evals)
            return no("upscale_refine_steps must be 1.." + std::to_string(evals - 1) + ": the "
                      "schedule takes " + std::to_string(evals) + " evaluations and the small "
                      "size at least one of them");
        const VideoShape ss = video_shape(d, sw, sh, r->chunks);
        j->geom_small = j->geom;
        j->geom_small.lat_h = ss.lat_h;
        j->geom_small.lat_w = ss.lat_w;
        j->rows_small = T + ss.audio_rows + ss.video_rows;
        const int64_t ns = positions_(RAD_VPOS_STEP, &j->geom_small, nullptr, 0);
        if (ns != j->rows_small)
            return no("the model lays the small size " + std::to_string(sw) + "x" +
                      std::to_string(sh) + " out in " + std::to_string(ns) + " step rows, which "
                      "this server cannot run");
        j->pos_small.assign((size_t)(3 * ns), 0.0f);
        if (positions_(RAD_VPOS_STEP, &j->geom_small, j->pos_small.data(), ns) != ns)
            return no("the model could not lay the small size out");
        j->split = evals - refine;
        j->factor = (float)(((double)r->width / sw + (double)r->height / sh) / 2.0);
        r->ticket->small_width = sw;
        r->ticket->small_height = sh;
        r->ticket->refine = refine;
    }
    j->n_chunks = r->chunks;
    j->n_samples = (int64_t)s.n_audio * (d.sample_rate / d.audio_rate);
    /* The reference table and the VAE encode queue were built above, before the step layout was
     * chosen. Each reference the server decoded into encoder form rides one encode pass at the slot
     * its index numbers within its kind -- a picture or video a Vencode pass, a standalone audio and
     * every video's soundtrack an Aencode pass (a soundtrack at its video's slot, since the encoder
     * and the scatter both number it by its video). The keyframe count stays 0: continuation is
     * separate work. */
    /* The soundtrack's length, exactly the samples the decode will hand the encoder — the
     * number a caller that chains clips against an audio floor checks. A build without
     * libavcodec never gets the muxed file to read it back from, so it is logged here. */
    RAD_INFO("video: soundtrack %.2f s, %lld samples at %lld Hz x %lld ch",
             (double)j->n_samples / d.sample_rate, (long long)j->n_samples,
             (long long)d.sample_rate, (long long)d.audio_channels);

    VideoTicket& t = *r->ticket;
    t.frames = s.frames;
    t.width = r->width;
    t.height = r->height;
    t.steps = j->sched.steps;
    t.sample_rate = d.sample_rate;
    t.channels = d.audio_channels;
    t.fps = (double)d.fps_num / (double)d.fps_den;
    t.in_tokens = T;
    t.rows = j->step_rows;   /* the conditioned step's real length, references included */
    t.t_submit_ns = (int64_t)rad_mono_ns();
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) {
        if (why) *why = closed_why_;
        return closed_status_;
    }
    j->id = next_id_++;
    j->req = std::move(r);
    jobs_.push_back(std::move(j));
    return RAD_OK;
}

void VideoDriver::shutdown(const char* why, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_why_ = why ? why : "the engine stopped";
    closed_status_ = status;
    for (auto& j : jobs_) j->req->ticket->finish(status, closed_why_);
}

int64_t VideoDriver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (!j->admitted) ++n;
    return n;
}

int64_t VideoDriver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (j->admitted) ++n;
    return n;
}

VideoDriver::Job* VideoDriver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void VideoDriver::release_locked(Job& j) {
    if (j.lane >= 0) {
        free_lanes_.push_back(j.lane);
        j.lane = -1;
    }
}

void VideoDriver::fail_locked(Job& j, int status, const std::string& why) {
    j.failed = true;
    j.req->ticket->finish(status, why);
}

/* In arrival order: a request needs only a lane. */
void VideoDriver::admit_locked() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted || j.failed) continue;
        if (free_lanes_.empty()) break;
        j.lane = free_lanes_.back();
        free_lanes_.pop_back();
        j.admitted = true;
    }
}

/* ================================================================== staging */
int VideoDriver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r)
        for (int g = 0; g < R_N; ++g) {
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
    return RAD_OK;
}

/* The geometry a pass's ops read off the device (RAD_VGEOM_*) for a pass at geometry `g` of
 * `rows` step rows, with chunk `chunk`'s span when it is a decode's and both grids when the job
 * upscales. */
void VideoDriver::stage_geom(const Job& j, const RadVideoGeom& g0, int64_t rows, int32_t chunk,
                             bool ref_step) {
    int32_t* g = hptr<int32_t>(R_GEOM);
    std::memset(g, 0, RAD_VGEOM_N * 4);
    g[RAD_VGEOM_LAT_FRAMES] = g0.lat_frames;
    g[RAD_VGEOM_LAT_H] = g0.lat_h;
    g[RAD_VGEOM_LAT_W] = g0.lat_w;
    g[RAD_VGEOM_N_TEXT] = g0.n_text;
    g[RAD_VGEOM_N_AUDIO] = g0.n_audio;
    g[RAD_VGEOM_CHUNK_LAT0] = (int32_t)(decl_.latents_per_chunk * chunk);
    g[RAD_VGEOM_CHUNK_FRAMES] = (int32_t)decl_.dec_latents;
    g[RAD_VGEOM_CHUNK_H] = g0.lat_h;
    g[RAD_VGEOM_CHUNK_W] = g0.lat_w;
    g[RAD_VGEOM_CHUNK_FIRST] = chunk == 0;
    g[RAD_VGEOM_CHUNK_LAST] = chunk == j.n_chunks - 1;
    g[RAD_VGEOM_PATCH_FRAMES] = (int32_t)decl_.dec_latents;
    g[RAD_VGEOM_PATCH_H] = (int32_t)(g0.lat_h / decl_.patch);
    g[RAD_VGEOM_PATCH_W] = (int32_t)(g0.lat_w / decl_.patch);
    if (j.split > 0) {
        const RadVideoGeom& a = j.geom_small;
        const RadVideoGeom& b = j.geom;
        const int32_t p = (int32_t)decl_.patch;
        const int32_t up[] = { a.lat_frames, a.lat_h, a.lat_w, b.lat_h, b.lat_w,
                               b.lat_frames, b.lat_h, b.lat_w,
                               a.lat_frames, a.lat_h / p, a.lat_w / p,
                               b.lat_frames, b.lat_h / p, b.lat_w / p };
        std::memcpy(g + RAD_VGEOM_UP_SRC, up, sizeof up);
    }
    reg_[R_GEOM].used = RAD_VGEOM_N * 4;
    const bool wide = j.ref_wide && ref_step;
    const int64_t audio_rows = decl_.audio_channels * (int64_t)g0.n_audio;
    /* With references riding, the sampler owns only the target video rows -- not the reference
     * rows the pack placed ahead of them -- so the target count is the pack's, not the plain
     * rows-minus-text-minus-audio. The audio target is the whole soundtrack either way. */
    const int64_t video_rows = wide ? j.lat_v : rows - g0.n_text - audio_rows;
    *hptr<int32_t>(R_LANE) = j.lane;
    *hptr<int32_t>(R_FIRST) = j.step == 0;
    int32_t* seg = hptr<int32_t>(R_SEG);
    if (wide) {
        /* The pack laid the reference rows out ahead of the target streams, video grid last. Stage
         * its boundaries verbatim: the VSA derives its segment count from the operand length, so
         * every reference becomes a dense prefix sink with no kernel change. Only the step pass
         * stages this table; the decode, encode and upscale passes stage the plain layout below. */
        std::memcpy(seg, j.seg.data(), j.seg.size() * sizeof(int32_t));
        reg_[R_SEG].used = (int64_t)j.seg.size() * 4;
    } else {
        /* The pack lays the segments out (rad_arch_video_segs, or the reference-aware
         * rad_arch_video_pack once refs ride); for a plain request its table is the one computed
         * below, so a plugin without the exports -- or one whose table disagrees with this
         * geometry -- stages the same bytes either way. The pack answers with the rows, not a
         * count, so a poison word at the fifth slot is what proves a plain table has no entries
         * past the fourth: a pack that writes past it disagrees with this geometry. */
        bool packed = false;
        if (pack_) {
            seg[4] = -1;
            const int64_t pr = pack_(&g0, nullptr, seg, kSegCap, nullptr, 0);
            packed = pr == rows && seg[4] == -1 && seg[0] == 0 && seg[1] == g0.n_text &&
                     seg[2] == (int32_t)(g0.n_text + audio_rows) && seg[3] == (int32_t)rows;
        } else if (segs_) {
            packed = segs_(&g0, seg, 4) == 4 && seg[0] == 0 && seg[1] == g0.n_text &&
                     seg[2] == (int32_t)(g0.n_text + audio_rows) && seg[3] == (int32_t)rows;
        }
        if (!packed) {
            seg[0] = 0;
            seg[1] = g0.n_text;
            seg[2] = (int32_t)(g0.n_text + audio_rows);
            seg[3] = (int32_t)rows;
        }
        reg_[R_SEG].used = 4 * 4;
    }
    /* The norm's merged boundaries. On a widened step that is the plugin's merged group table (a
     * subset of the fine one just staged into R_SEG); on the plain path and every non-step pass it
     * is the fine table verbatim -- nothing merges there, so the norm reads the same bytes as the
     * attention and the generated output is exactly what it was before this operand split. */
    int32_t* mod = hptr<int32_t>(R_MOD);
    if (wide && j.nmod > 0) {
        std::memcpy(mod, j.mod_seg.data(), j.mod_seg.size() * sizeof(int32_t));
        reg_[R_MOD].used = (int64_t)j.mod_seg.size() * 4;
    } else {
        std::memcpy(mod, seg, reg_[R_SEG].used);
        reg_[R_MOD].used = reg_[R_SEG].used;
    }
    int32_t* lat = hptr<int32_t>(R_LAT);
    lat[0] = 0;
    lat[1] = (int32_t)video_rows;
    lat[2] = 0;
    lat[3] = (int32_t)audio_rows;
    reg_[R_LANE].used = 4;
    reg_[R_FIRST].used = 4;
    reg_[R_LAT].used = 4 * 4;
}

void VideoDriver::base_batches(const Job& j, const RadVideoGeom& g, int64_t n_tok, int step) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadVideoGenBatch& V = vout_[r];
        const int ri = (int)r;
        B = RadBatch{};
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = 1;
        B.max_q_len = (int32_t)n_tok;
        B.max_q_len_prefill = (int32_t)n_tok;
        B.token_ids = dptr<int32_t>(ri, R_TOK);
        B.positions = dptr<int32_t>(ri, R_POS);
        B.cu_seqlens = dptr<int32_t>(ri, R_CU);
        V = RadVideoGenBatch{};
        V.lane = j.lane;
        V.geom = g;
        V.geom_from = j.geom_small;
        V.geom_d = dptr<int32_t>(ri, R_GEOM);
        V.lane_d = dptr<int32_t>(ri, R_LANE);
        V.first_d = dptr<int32_t>(ri, R_FIRST);
        V.seg_cu = dptr<int32_t>(ri, R_SEG);
        V.mod_cu = dptr<int32_t>(ri, R_MOD);
        V.lat_cu = dptr<int32_t>(ri, R_LAT);
        V.n_steps = j.sched.steps;
        V.t_video = dptr<float>(ri, R_TV);
        V.t_audio = dptr<float>(ri, R_TA);
        V.step = j.step;
        V.sched = dptr<float>(ri, R_SCHED);
        V.pos = dptr<float>(ri, R_ROPE);
        V.seed = dptr<uint32_t>(ri, R_SEED);
        V.chunk = j.chunk;
        V.n_chunks = j.n_chunks;
        V.dec_rows = j.dec_rows;
        V.n_samples = j.n_samples;
        V.upscale = dptr<float>(ri, R_UP);
        V.ref_tab = j.ref_tab.empty() ? nullptr : j.ref_tab.data();
        B.vgen_b = &V;
    }
}

/* ================================================================== the passes */
const RadBatch* VideoDriver::next(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    pass_ = Pass{};
    rb_.clear();
    if (closed_) return nullptr;
    /* Retire finished, failed and cancelled jobs, returning their lanes before admitting. Safe
     * because next() runs between passes, with nothing in flight. */
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        Job& j = **it;
        const bool cancelled = j.req->ticket->cancel.load(std::memory_order_acquire);
        if (!(j.failed || cancelled || j.finished)) { ++it; continue; }
        if (cancelled && !j.failed && !j.finished)
            j.req->ticket->finish(RAD_E_CANCELLED, "the request was cancelled");
        release_locked(j);
        it = jobs_.erase(it);
    }
    admit_locked();

    /* 1. The decodes of a denoised lane: the soundtrack, then the chunks in order. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || j.step < j.sched.steps) continue;
        if (!j.audio_done) return build_audio(j, step);
        if (j.chunk < j.n_chunks) return build_chunk(j, step);
    }
    /* 2. A VAE encode of a reference-conditioned lane's media, before its prompt: one pass per
     *    reference, so the encoder's latents are in the lane's slot rows by the time the prompt
     *    and its steps run (the step's pack scatters them into the sequence). */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || j.cond_done) continue;
        if (j.enc_next < j.enc_q.size()) return build_encode(j, step);
    }
    /* 3. A prompt. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && !j.cond_done) return build_cond(j, step);
    }
    /* 4. The oldest lane's next step, or its upscale once its small steps are done. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || !j.cond_done || j.step >= j.sched.steps) continue;
        if (small_now(j) && j.step == j.split) return build_upscale(j, step);
        return build_step(j, step);
    }
    return nullptr;
}

const RadBatch* VideoDriver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

const RadBatch* VideoDriver::build_cond(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const int64_t T = j.geom.n_text;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    std::memcpy(tok, j.req->tokens.data(), (size_t)T * 4);
    for (int64_t i = 0; i < T; ++i) pos[i] = (int32_t)i;
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)T;
    float* tv = hptr<float>(R_TV);
    float* ta = hptr<float>(R_TA);
    for (int32_t i = 0; i < j.sched.steps; ++i) {
        tv[i] = 1.0f - j.sched.sig_v[(size_t)i];
        ta[i] = 1.0f - j.sched.sig_a[(size_t)i];
    }
    reg_[R_TOK].used = T * 4;
    reg_[R_POS].used = T * 4;
    reg_[R_CU].used = 2 * 4;
    reg_[R_TV].used = j.sched.steps * 4;
    reg_[R_TA].used = j.sched.steps * 4;
    const RadVideoGeom& g = small_now(j) ? j.geom_small : j.geom;
    stage_geom(j, g, small_now(j) ? j.rows_small : j.rows, 0);

    /* Reference media for the tower and the conditioning's M-RoPE, staged exactly as the image
     * driver stages a conditioned prompt (core/gen/driver.cpp's build_cond). R_ROPE holds the
     * [3,T] i32 grid here and a step's f32 positions in build_step — never both in one pass, and
     * step_cond reads rope_pos instead of positions when it is set. */
    int64_t n_patch = 0, n_seg = 0, n_mm = 0;
    int32_t max_seg = 0;
    if (j.req->media) {
        const mm::PromptMedia& pm = *j.req->media;
        int32_t* rope = hptr<int32_t>(R_ROPE);
        for (int64_t i = 0; i < T; ++i) {
            int32_t p3[3] = { (int32_t)i, (int32_t)i, (int32_t)i };
            pm.rope.at(i, p3);
            rope[i] = p3[0];
            rope[T + i] = p3[1];
            rope[2 * T + i] = p3[2];
        }
        reg_[R_ROPE].used = 3 * T * 4;
        int64_t total = 0;
        for (const mm::Placed& p : pm.items) total += p.item->n_patch;
        uint16_t* pix = hptr<uint16_t>(R_ENCPIX);
        int32_t* crd = hptr<int32_t>(R_ENCCRD);
        int32_t* ecu = hptr<int32_t>(R_ENCCU);
        ecu[0] = 0;
        std::vector<int32_t> tmp;
        for (const mm::Placed& p : pm.items) {
            const mm::Item& it = *p.item;
            std::memcpy(pix + n_patch * patch_dim_, it.pixels.data(),
                        (size_t)(it.n_patch * patch_dim_) * 2);
            tmp.resize((size_t)(4 * it.n_patch));
            it.coords(0, it.n_patch, tmp.data());
            for (int cc = 0; cc < 4; ++cc)
                std::memcpy(crd + cc * total + n_patch, tmp.data() + cc * it.n_patch,
                            (size_t)it.n_patch * 4);
            for (int32_t s = 0; s < it.n_seg; ++s) {
                const int64_t a = it.patch_at(s), e = it.patch_at(s + 1);
                ecu[n_seg + 1] = (int32_t)(n_patch + e);
                max_seg = std::max<int32_t>(max_seg, (int32_t)(e - a));
                ++n_seg;
            }
            n_patch += it.n_patch;
        }
        int32_t* mmr = hptr<int32_t>(R_MMROW);
        for (const mm::RopeLayout::Run& run : pm.rope.runs)
            for (int32_t k = 0; k < run.n; ++k) mmr[n_mm++] = run.tok + k;
        reg_[R_ENCPIX].used = n_patch * patch_dim_ * 2;
        reg_[R_ENCCRD].used = 4 * total * 4;
        reg_[R_ENCCU].used = (n_seg + 1) * 4;
        reg_[R_MMROW].used = n_mm * 4;
    }

    if (upload() < 0) return nullptr;
    base_batches(j, g, T, step);
    if (j.req->media) {
        for (size_t r = 0; r < ranks_.size(); ++r) {
            RadBatch& B = out_[r];
            const int ri = (int)r;
            B.rope_pos = dptr<int32_t>(ri, R_ROPE);
            B.rope_mixed = 1;
            B.n_mm_rows = n_mm;
            B.mm_rows = n_mm ? dptr<int32_t>(ri, R_MMROW) : nullptr;
            B.enc = 0;
            B.enc_n_patch = n_patch;
            B.enc_pixels = n_patch ? dptr<uint16_t>(ri, R_ENCPIX) : nullptr;
            B.enc_coord = n_patch ? dptr<int32_t>(ri, R_ENCCRD) : nullptr;
            B.enc_n_seg = n_seg;
            B.enc_cu = n_seg ? dptr<int32_t>(ri, R_ENCCU) : nullptr;
            B.enc_max_seg = max_seg;
        }
    }
    for (auto& B : out_) B.gen = RAD_GEN_PASS_VCOND;
    pass_.kind = RAD_GEN_PASS_VCOND;
    pass_.job = j.id;
    return &out_[0];
}

/* Step i's timesteps and sigmas, the video's then the soundtrack's (RadVideoGenBatch::sched);
 * with `to_clean` the next sigmas are 0, so the step lands on its clean estimate. */
void VideoDriver::stage_sched(const Job& j, int32_t i, bool to_clean) {
    float* s = hptr<float>(R_SCHED);
    s[0] = 1.0f - j.sched.sig_v[(size_t)i];
    s[1] = j.sched.sig_v[(size_t)i];
    s[2] = to_clean ? 0.0f : j.sched.sig_v[(size_t)i + 1];
    s[3] = 1.0f - j.sched.sig_a[(size_t)i];
    s[4] = j.sched.sig_a[(size_t)i];
    s[5] = to_clean ? 0.0f : j.sched.sig_a[(size_t)i + 1];
    reg_[R_SCHED].used = 6 * 4;
}

/* The video's noise seed is the request's; the soundtrack's the same with its top bit flipped, so
 * the two streams never coincide. */
void VideoDriver::stage_seed(const Job& j) {
    uint32_t* sd = hptr<uint32_t>(R_SEED);
    const uint64_t seed = j.req->seed;
    sd[0] = (uint32_t)seed;
    sd[1] = (uint32_t)(seed >> 32);
    sd[2] = (uint32_t)seed;
    sd[3] = (uint32_t)(seed >> 32) ^ 0x80000000u;
    reg_[R_SEED].used = 4 * 4;
}

const RadBatch* VideoDriver::build_step(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const bool small = small_now(j);
    const RadVideoGeom& g = small ? j.geom_small : j.geom;
    const int64_t rows = small ? j.rows_small : j.step_rows;   /* widened by references when they ride */
    const std::vector<float>& pos = small ? j.pos_small : j.pos;
    stage_sched(j, j.step, small && j.step == j.split - 1);
    std::memcpy(hptr<float>(R_ROPE), pos.data(), pos.size() * 4);
    reg_[R_ROPE].used = (int64_t)pos.size() * 4;
    stage_seed(j);
    /* The packed sequence is one attention document. */
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)rows;
    reg_[R_CU].used = 2 * 4;
    /* A reference-conditioned step stages the pack's reference-aware segment table; the plain and
     * small (pre-upscale) steps stage the plain four-boundary table. ref_wide never coincides with
     * small -- submit refuses upscale together with references. */
    stage_geom(j, g, rows, 0, j.ref_wide);
    if (upload() < 0) return nullptr;
    base_batches(j, g, rows, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_VSTEP;
    pass_.kind = RAD_GEN_PASS_VSTEP;
    pass_.job = j.id;
    return &out_[0];
}

/* The small latent to the full size, both latents re-noised to the first full-size step's
 * sigmas. */
const RadBatch* VideoDriver::build_upscale(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_sched(j, j.split, false);
    stage_seed(j);
    *hptr<float>(R_UP) = j.factor;
    reg_[R_UP].used = 4;
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_VUPSCALE;
    pass_.kind = RAD_GEN_PASS_VUPSCALE;
    pass_.job = j.id;
    return &out_[0];
}

const RadBatch* VideoDriver::build_audio(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_ADECODE;
    rb_.assign(1, Readback{ decl_.audio_out, j.n_samples * decl_.audio_channels * 4, audio_host_ });
    pass_.kind = RAD_GEN_PASS_ADECODE;
    pass_.job = j.id;
    return &out_[0];
}

const RadBatch* VideoDriver::build_chunk(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_geom(j, j.geom, j.rows, j.chunk);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, j.dec_rows, step);
    const bool last = j.chunk == j.n_chunks - 1;
    const int32_t n = (int32_t)(decl_.frames_per_chunk + (last ? decl_.frames_tail : 0));
    for (size_t r = 0; r < out_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_VDECODE;
        vout_[r].n_frames_out = n;
    }
    const int64_t bytes = (int64_t)n * j.req->height * j.req->width * 3;
    rb_.assign(1, Readback{ decl_.frames_out, bytes, frames_host_ });
    pass_.kind = RAD_GEN_PASS_VDECODE;
    pass_.job = j.id;
    pass_.n_frames = n;
    return &out_[0];
}

/* One media item through its VAE encoder: stage its encoder-ready bytes and the encode geometry
 * the plugin's rad_arch_encode_geom derives from them, then name the pass's kind and slot so the
 * encoder folds the latents into the lane's fixed slot rows. The server sends the NATURAL decoded
 * media; the padding the geom demands (a video's frames to a whole clip, an audio's samples to a
 * whole hop) is applied here, since only the driver has the geom, and the encoder is handed exactly
 * that padded volume. */
const RadBatch* VideoDriver::build_encode(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const Job::EncItem& it = j.enc_q[j.enc_next];
    const VideoRefMedia& m = j.req->refs[(size_t)it.ref];
    const bool audio = it.kind == RAD_ENC_VAUDIO || it.kind == RAD_ENC_AUDIO;
    auto fail = [&](const std::string& w) {
        fail_locked(j, RAD_E_STATE, w);
        return nullptr;
    };
    if (!enc_geom_)
        return fail("the request carries reference media to encode, but this plugin exports no "
                    "rad_arch_encode_geom");
    /* The encoder kernels recompute their geometry from the batch's frame / sample counts and
     * require them to be the padded figures rad_arch_encode_geom reports (a video's frames out to a
     * whole clip, an audio's samples out to a whole hop). The server sends the NATURAL decoded
     * media, so the padding lives here, where the geometry is known: a video's short final clip is
     * filled by repeating its last frame, an audio's tail by zeros. A single picture is already
     * clip- and hop-complete (one frame, no hop), so its bytes and staging are unchanged. */
    int32_t* gd = hptr<int32_t>(R_ENCG);
    int64_t stage_frames = 0, stage_samples = 0;
    if (audio) {
        const int64_t n = m.enc_n;
        const int32_t ch = m.enc_channels;
        if (n <= 0 || ch <= 0) return fail("a reference audio has no samples or no channels");
        int64_t in[4] = { n, 0, 0, ch };
        if (enc_geom_(RAD_GEN_PASS_AENCODE, in, gd, 25) < 0)
            return fail("the audio encoder refuses " + std::to_string(n) + " samples over " +
                        std::to_string(ch) + " channels");
        stage_samples = gd[0];   /* n rounded up to a whole hop; the encoder reads exactly this */
        if (stage_samples * ch * 4 > reg_[R_ENCSM].bytes)
            return fail("a reference audio does not fit one encode pass");
        float* sm = hptr<float>(R_ENCSM);
        std::memset(sm, 0, (size_t)stage_samples * ch * 4);   /* right-pad each channel with zeros */
        for (int32_t c = 0; c < ch; ++c)
            std::memcpy(sm + (size_t)c * stage_samples, m.enc_samples.data() + (size_t)c * n,
                        (size_t)n * 4);
        reg_[R_ENCSM].used = stage_samples * ch * 4;
    } else {
        const int64_t nf = m.enc_frames, H = m.enc_h, W = m.enc_w;
        if (nf <= 0 || H <= 0 || W <= 0)
            return fail("a reference picture or video has no frames or no sides");
        int64_t in[4] = { nf, H, W, 3 };
        if (enc_geom_(RAD_GEN_PASS_VENCODE, in, gd, 25) < 0)
            return fail("the video encoder refuses a reference at " + std::to_string(H) + "x" +
                        std::to_string(W) + " over " + std::to_string(nf) + " frames");
        const int64_t area = H * W * 3;
        stage_frames = gd[23];   /* nf clip-padded to a whole number of 17-frame clips */
        if (stage_frames * area * 4 > reg_[R_ENCPX].bytes)
            return fail("a reference picture or video does not fit one encode pass");
        float* px = hptr<float>(R_ENCPX);
        std::memcpy(px, m.enc_pixels.data(), (size_t)nf * area * 4);
        /* Repeat the final frame out to the padded clip count: the causal encoder wants every clip
         * full, and the reference pads the short last clip exactly this way. */
        for (int64_t f = nf; f < stage_frames; ++f)
            std::memcpy(px + f * area, px + (nf - 1) * area, (size_t)area * 4);
        reg_[R_ENCPX].used = stage_frames * area * 4;
    }
    reg_[R_ENCG].used = 25 * 4;
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const int ri = (int)r;
        RadVideoGenBatch& V = vout_[(size_t)r];
        V.enc_geom_d = dptr<int32_t>(ri, R_ENCG);
        V.enc_kind = it.kind;
        V.enc_slot = it.slot;
        if (audio) {
            V.enc_samples = dptr<float>(ri, R_ENCSM);
            V.enc_n_samples = stage_samples;
            V.enc_channels = m.enc_channels;
        } else {
            V.enc_pixels = dptr<float>(ri, R_ENCPX);
            V.enc_frames = stage_frames;
            V.enc_height = m.enc_h;
            V.enc_width = m.enc_w;
        }
    }
    const int32_t kind = audio ? RAD_GEN_PASS_AENCODE : RAD_GEN_PASS_VENCODE;
    for (auto& B : out_) B.gen = kind;
    pass_.kind = kind;
    pass_.job = j.id;
    if (enc_dbg_) {
        const rad_buf   h = audio ? decl_.dbg_enca : decl_.dbg_evc;
        const int64_t   n = audio ? decl_.dbg_enca_bytes : decl_.dbg_evc_bytes;
        enc_dbg_used_ = 0;
        if (h && n > 0) {
            rb_.assign(1, Readback{h, n, enc_dbg_host_});
            enc_dbg_used_ = n;
        }
    }
    return &out_[0];
}

int VideoDriver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t now = (int64_t)rad_mono_ns();
    Job* j = job_locked(pass_.job);
    switch (j ? pass_.kind : 0) {
    case RAD_GEN_PASS_VCOND:
        j->cond_done = true;
        j->req->ticket->t_cond_ns = now;
        break;
    case RAD_GEN_PASS_VENCODE:
    case RAD_GEN_PASS_AENCODE:
        /* The media's latents are now in the lane's slot rows; take the next reference, if any. */
        if (enc_dbg_ && enc_dbg_used_ > 0) {
            const int64_t nf = enc_dbg_used_ / 4;
            const float* z = reinterpret_cast<const float*>(enc_dbg_host_);
            uint64_t h = 1469598103934665603ULL;   /* FNV-1a over the readback bytes */
            for (int64_t i = 0; i < enc_dbg_used_; ++i)
                h = (h ^ (uint64_t)(uint8_t)enc_dbg_host_[i]) * 1099511628211ULL;
            int64_t nz = 0;
            float mx = 0.0f;
            double sum = 0.0;
            for (int64_t i = 0; i < nf; ++i) {
                if (z[i] != 0.0f) ++nz;
                const float a = z[i] < 0 ? -z[i] : z[i];
                if (a > mx) mx = a;
                sum += z[i];
            }
            RAD_ERR("enc_dbg %s slot %d: fnv %016llx floats %lld nonzero %lld maxabs %.6g "
                    "sum %.6g  head[%.6g %.6g %.6g %.6g]",
                    pass_.kind == RAD_GEN_PASS_AENCODE ? "AENCA(b_enca)" : "VENCV(b_evc)",
                    (int)enc_dbg_used_, (unsigned long long)h, (long long)nf, (long long)nz,
                    (double)mx, sum, (double)z[0], (double)z[1], (double)z[2], (double)z[3]);
        }
        ++j->enc_next;
        break;
    case RAD_GEN_PASS_VSTEP: {
        ++j->step;
        VideoEvent e;
        e.kind = VideoEvent::STEP;
        e.steps_done = j->step;
        j->req->ticket->push(std::move(e));
        if (j->split > 0 && j->step == j->split) j->req->ticket->t_small_ns = now;
        if (j->step >= j->sched.steps) j->req->ticket->t_denoised_ns = now;
        break;
    }
    case RAD_GEN_PASS_VUPSCALE:
        j->upscaled = true;
        j->req->ticket->t_upscaled_ns = now;
        break;
    case RAD_GEN_PASS_ADECODE: {
        VideoEvent e;
        e.kind = VideoEvent::AUDIO;
        e.samples.assign(audio_host_, audio_host_ + j->n_samples * decl_.audio_channels);
        j->req->ticket->push(std::move(e));
        j->audio_done = true;
        break;
    }
    case RAD_GEN_PASS_VDECODE: {
        VideoEvent e;
        e.kind = VideoEvent::FRAMES;
        e.first = j->frames_out;
        e.n = pass_.n_frames;
        const size_t bytes = (size_t)pass_.n_frames * j->req->height * j->req->width * 3;
        e.rgb.assign(frames_host_, frames_host_ + bytes);
        j->req->ticket->push(std::move(e));
        j->frames_out += pass_.n_frames;
        if (++j->chunk == j->n_chunks) {
            j->finished = true;
            j->req->ticket->t_done_ns = now;
            j->req->ticket->finish(RAD_OK);
        }
        break;
    }
    default:
        break;
    }
    pass_ = Pass{};
    rb_.clear();
    return RAD_OK;
}

}  /* namespace gen */
}  /* namespace rad */
