/* video_driver.cpp -- the video generator's pass driver (video_driver.h), the ticket a caller
 * waits on, and a request's shape and schedule (video.h). */
#include "video_driver.h"
#include "rad_internal.h"
#include "rad_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    s.audio_rows = (d.audio_streams > 0 ? d.audio_streams : d.audio_channels) * (int64_t)s.n_audio;
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

/* Form 1 (RadVideoGenDecl::sched_kind): FlowMatchEulerDiscreteScheduler.set_timesteps with
 * LTX-2's sigmas. numpy's linspace in f64 (i * step + start, the last exactly the stop) taken to
 * f32; the shift and the terminal stretch in f32, as numpy computes on an f32 array with Python
 * scalars. */
static std::vector<float> flow_sigmas(const RadVideoGenDecl& d, int32_t n, int64_t tokens) {
    std::vector<float> s((size_t)n);
    const double stop = 1.0 / n, step = n > 1 ? (stop - 1.0) / (n - 1) : 0.0;
    for (int32_t i = 0; i < n; ++i) s[(size_t)i] = (float)(i == n - 1 && n > 1 ? stop : i * step + 1.0);
    if (d.dyn_shift) {
        const double m = (d.max_shift - d.base_shift) / (double)(d.max_seq - d.base_seq);
        const double mu = (double)tokens * m + (d.base_shift - m * (double)d.base_seq);
        const float e = (float)std::exp(mu);
        for (float& v : s) v = e / (e + (1.0f / v - 1.0f));
    }
    if (d.shift_terminal > 0.0) {
        const float sf = (1.0f - s.back()) / (float)(1.0 - d.shift_terminal);
        for (float& v : s) v = 1.0f - (1.0f - v) / sf;
    }
    s.push_back(0.0f);
    return s;
}

VideoSchedule video_schedule(const RadVideoGenDecl& d, int32_t steps, int64_t video_tokens,
                             bool refine, double shift_video, double shift_audio) {
    /* A request's shift overrides the declaration's per stream; 0 keeps the trained value. Form 1
     * ignores both: its shift is the declaration's, dynamic or terminal. */
    const double sv = shift_video > 0.0 ? shift_video : d.shift_video;
    const double sa = shift_audio > 0.0 ? shift_audio : d.shift_audio;
    VideoSchedule v;
    if (d.sched_kind == 1) {
        if (d.n_rungs > 0) {
            if (steps != d.n_rungs) return v;
            for (int32_t i = 0; i < d.n_rungs; ++i) v.sig_v.push_back((float)d.rungs[i]);
            if (refine)
                for (int32_t i = 0; i < d.n_refine_rungs; ++i)
                    v.sig_v.push_back((float)d.refine_rungs[i]);
            v.sig_v.push_back(0.0f);
        } else {
            if (steps < 1) return v;
            v.sig_v = flow_sigmas(d, steps, video_tokens);
        }
        v.sig_a = v.sig_v;
        v.steps = (int32_t)v.sig_v.size() - 1;
        return v;
    }
    if (d.n_rungs > 0) {
        if (steps != d.n_rungs + 1) return v;
        v.sig_v = rung_sigmas(sv, d.rungs, d.n_rungs);
        v.sig_a = rung_sigmas(sa, d.rungs, d.n_rungs);
        v.steps = d.n_rungs;
        return v;
    }
    v.sig_v = shifted_sigmas(sv, steps);
    v.sig_a = shifted_sigmas(sa, steps);
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
    /* A guided model's VCOND carries the negative prompt after the positive. */
    const int64_t prompts = d.guidance ? 2 : 1;
    const int64_t want[] = {
        prompts * d.max_text * 4,   /* R_TOK */
        prompts * d.max_text * 4,   /* R_POS */
        3 * 4,                   /* R_CU */
        3 * max_tok * 4,         /* R_ROPE: a step's rotary positions, a conditioning's [3,T] M-RoPE */
        RAD_VGEOM_N * 4,         /* R_GEOM */
        16 * 4,                  /* R_SCHED: the Euler scalars [0..5], then the res_multistep
                                  * coeffs -- video [6..10], audio [11..15] */
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
        d.image_cond ? d.max_area * 3 * 2 : 0,   /* R_IMG: a first frame, bf16 */
        8 * 4,                   /* R_GUIDE */
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
    decoder_names_ = prog.vgen_str.decoders;
    decoder_area_ = prog.vgen_str.decoder_max_area;
    refine_rungs_ = prog.vgen_str.refine_rungs;
    decl_.refine_rungs = refine_rungs_.empty() ? nullptr : refine_rungs_.data();
    default_negative_ = prog.vgen_str.default_negative;
    decl_.default_negative = default_negative_.c_str();
    const int64_t prompts = decl_.guidance ? 2 : 1;
    if (max_tok_ < prompts * decl_.max_text) {
        RAD_ERR("video generator '%s': a pass carries %lld rows and %s may be %lld tokens",
                name_.c_str(), (long long)max_tok_, prompts == 2 ? "a prompt and its negative" :
                "a prompt", (long long)(prompts * decl_.max_text));
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
    duration_host_ = static_cast<float*>(rad_dev_alloc(64, RAD_MEM_HOST_PINNED));
    if (!host_ || !frames_host_ || !audio_host_ || !duration_host_) {
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
        /* The slab must hold the largest single shadow AND the full "all" multi-shadow slab. */
        const int64_t all_sum = decl_.dbg_ve_trunk_bytes + decl_.dbg_ve_nout_bytes +
                                decl_.dbg_ve_q48_bytes + decl_.dbg_ve_q2_bytes +
                                decl_.dbg_ve_l0_bytes + decl_.dbg_ve_l1_bytes +
                                decl_.dbg_ve_l2_bytes + decl_.dbg_ve_l3_bytes + decl_.dbg_ve_l4_bytes;
        enc_dbg_bytes_ = std::max<std::int64_t>(
            { decl_.dbg_enca_bytes, decl_.dbg_evc_bytes, decl_.dbg_ve_cin_bytes, all_sum });
        enc_dbg_host_ = static_cast<char*>(rad_dev_alloc(enc_dbg_bytes_, RAD_MEM_HOST_PINNED));
        enc_dbg_ = enc_dbg_host_ != nullptr;
        if (!enc_dbg_)
            RAD_ERR("video generator: R4D_ENC_DEBUG set but no pinned host for the %lld-byte "
                    "encoder readback", (long long)enc_dbg_bytes_);
    }
    /* Debug-only DiT step dump (#61, R4D_DIT_SEL): the slab for the step-0 readback -- the plugin's
     * f32 shadows (h0 | temb | mod) plus the step's velocity head output. The plugin declares the
     * shadows only when it sees the same env, so a mismatch shows up as dbg_dit_h0 == 0 and simply
     * disables the dump. The value of R4D_DIT_SEL names the step to dump; only step 0 (what the
     * shadows carry) is honoured, anything else warns and stays off. */
    if (const char* sel = std::getenv("R4D_DIT_SEL")) {
        const long want = std::strtol(sel, nullptr, 10);
        dit_sel_ = want > 0 ? (int)want : 0;   /* #54 debug: also dump this step, dit_sK_* files */
        if (want < 0)
            RAD_ERR("video generator: R4D_DIT_SEL=%s: negative step; ignoring", sel);
        else if (decl_.dbg_dit_h0 && decl_.dbg_dit_vel) {
            dit_h0_b_ = decl_.dbg_dit_h0_bytes;
            dit_vel_b_ = decl_.dbg_dit_vel_bytes;
            dit_temb_b_ = decl_.dbg_dit_temb_bytes;
            dit_mod_b_ = decl_.dbg_dit_mod_bytes;
            dit_blk_b_ = decl_.dbg_dit_blk_bytes;
            dit_hn_b_ = decl_.dbg_dit_hn_bytes;
            dit_nout_b_ = decl_.dbg_dit_nout_bytes;
            /* #63: the decoder parity chain rides the same slab, appended after the DiT buffers.
             * It lags none of them -- the decode pass readback happens in its own pass. */
            dec_lat_b_ = decl_.dbg_vd_lat_bytes;
            dec_x0_b_ = decl_.dbg_vd_x0_bytes;
            dec_blk_b_ = decl_.dbg_vd_blk_bytes;
            dec_n_b_ = decl_.dbg_vd_n_bytes;
            dec_pix_b_ = decl_.dbg_vd_pix_bytes;
            const int64_t sum = dit_h0_b_ + dit_vel_b_ + dit_temb_b_ + dit_mod_b_ + dit_blk_b_
                              + dit_hn_b_ + dit_nout_b_ + dec_lat_b_ + dec_x0_b_ + dec_blk_b_
                              + dec_n_b_ + dec_pix_b_;
            dit_host_ = static_cast<char*>(rad_dev_alloc(sum, RAD_MEM_HOST_PINNED));
            dit_dbg_ = dit_host_ != nullptr;
            if (!dit_dbg_)
                RAD_ERR("video generator: R4D_DIT_SEL set but no pinned host for the %lld-byte "
                        "DiT dump", (long long)sum);
        } else {
            RAD_ERR("video generator: R4D_DIT_SEL set but the plugin exported no DiT shadows -- "
                    "restart it with the same env so declare_dit sees it");
        }
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
    if (dit_host_) rad_dev_free(dit_host_, RAD_MEM_HOST_PINNED);
    if (duration_host_) rad_dev_free(duration_host_, RAD_MEM_HOST_PINNED);
    duration_host_ = nullptr;
    host_ = nullptr;
    frames_host_ = nullptr;
    audio_host_ = nullptr;
    enc_dbg_host_ = nullptr;
    dit_host_ = nullptr;
}

/* ================================================================== requests */
/* A length head's prediction (RadVideoGenDecl::duration_out) in chunks: diffusers'
 * LTX2DurationHead.predict_num_frames on the chunk grid -- the seconds clamped to the declared
 * bounds, rounded to frames, floored to the grid and lifted one chunk when that falls under the
 * minimum -- then held to min_chunks..max_chunks. `note` says what was clamped. */
int32_t VideoDriver::predicted_chunks(double log_seconds, std::string* note) const {
    const RadVideoGenDecl& d = decl_;
    const double fps = (double)d.fps_num / (double)d.fps_den;
    float e = std::exp((float)log_seconds);
    if (d.duration_dtype == RAD_BF16) e = rad_bf16_to_f32(rad_f32_to_bf16(e));
    const double sec = e;
    auto round_even = [](double v) { return (int64_t)std::nearbyint(v); };
    const int64_t lo = std::max<int64_t>(1, round_even(d.duration_min * fps));
    const int64_t hi = round_even(d.duration_max * fps);
    const int64_t want = std::isfinite(sec) ? round_even(sec * fps) : hi;
    const int64_t frames = std::max(lo, std::min(want, hi));
    int64_t n = std::max<int64_t>(0, (frames - d.frames_tail) / d.frames_per_chunk);
    if (d.frames_per_chunk * n + d.frames_tail < lo &&
        d.frames_per_chunk * (n + 1) + d.frames_tail <= hi)
        ++n;
    std::string what = fmt(": %.3f s predicted", sec);
    if (!(sec >= d.duration_min && sec <= d.duration_max))
        what += fmt(", held to %g..%g s", d.duration_min, d.duration_max);
    if (n < d.min_chunks || n > d.max_chunks) {
        n = std::max<int64_t>(d.min_chunks, std::min<int64_t>(n, d.max_chunks));
        what += fmt(", within the %lld..%lld frames this model makes",
                    (long long)(d.frames_per_chunk * d.min_chunks + d.frames_tail),
                    (long long)(d.frames_per_chunk * d.max_chunks + d.frames_tail));
    }
    if (note) *note = what;
    return (int32_t)n;
}

/* The request's shape at `chunks` long: its geometry, rows and positions, its schedule, the
 * upscale's split and what its ticket says it makes. Run at submit, and again for a request whose
 * length the model predicts once its VCOND pass has written the prediction. */
int VideoDriver::shape_job(Job& j, const VideoRequest& r, int32_t chunks, int32_t steps,
                           std::string* why) {
    const RadVideoGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int64_t T = (int64_t)r.tokens.size();
    const VideoShape s = video_shape(d, r.width, r.height, chunks);
    j.geom.n_text = (int32_t)(d.step_text_rows > 0 ? d.step_text_rows : T);
    j.geom.frames = s.frames;
    j.geom.lat_frames = s.lat_frames;
    j.geom.lat_h = s.lat_h;
    j.geom.lat_w = s.lat_w;
    j.geom.n_audio = s.n_audio;
    /* The step's row count and its positions. A reference-conditioned step widens them; the
     * encode queue and the pack's table are built first because the layout depends on them. */
    j.ref_tab.clear();
    j.enc_q.clear();
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
     * reference rows are part of that layout. Endpoint keyframes (a first and/or last frame pinned
     * at the output's ends) are appended after the reference rows as {frame, video_t, audio_t}
     * triples; a request carries either references or keyframes, never both. */
    if (d.encodes) {
        bool need_geom = false;
        for (const VideoRefMedia& m : r.refs)
            if (m.kind != VideoRefMedia::IMAGE && m.kind != VideoRefMedia::FIRST_FRAME &&
                m.kind != VideoRefMedia::LAST_FRAME) need_geom = true;
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
        /* A picture or an endpoint keyframe folds through the VAE as a ONE-frame clip, which is
         * why it rides at a far larger side than a video can -- but "larger" is not "unbounded":
         * the encoder's per-level buffers hold a full clip's worth of the DEEPEST temporal stage,
         * so one frame cannot exceed a fifth of the pixels budget no matter how few frames it
         * has. The geometry call answers that; asking it here refuses an over-cap media at
         * accept, where a step-time refusal would stop the whole engine. */
        auto one_frame = [&](const VideoRefMedia& m) {
            const int64_t in[4] = { 1, m.enc_h, m.enc_w, 3 };
            return !enc_geom_ || enc_geom_(RAD_GEN_PASS_VENCODE, in, eg, 25) >= 0;
        };
        for (size_t i = 0; i < r.refs.size(); ++i) {
            const VideoRefMedia& m = r.refs[i];
            if (m.kind == VideoRefMedia::IMAGE) {
                if (m.enc_pixels.empty())
                    return no("'" + m.name + "' was not prepared for the video encoder");
                if (m.index < 1)
                    return no("'" + m.name + "' has no slot in the conditioning");
                if (bad_side(m))
                    return no("'" + m.name + "' has encoder sides (" + std::to_string(m.enc_h) +
                              "x" + std::to_string(m.enc_w) + ") the VAE grid of " +
                              std::to_string(d.enc_ratio) + " does not divide");
                if (!one_frame(m))
                    return no("'" + m.name + "' is " + std::to_string(m.enc_h) + "x" +
                              std::to_string(m.enc_w) +
                              ", a single frame larger than the video encoder's capacity");
                j.enc_q.push_back({RAD_ENC_IMAGE, m.index, (int32_t)i});
                if (j.ref_tab.empty()) j.ref_tab.assign(2, 0);   /* [n_refs, n_keys] */
                ++j.ref_tab[0];
                j.ref_tab.insert(j.ref_tab.end(),
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
                for (size_t k = 0; k < r.refs.size(); ++k) {
                    const VideoRefMedia& a = r.refs[k];
                    if (a.kind != VideoRefMedia::VIDEO_AUDIO || a.index != m.index) continue;
                    if (a.enc_samples.empty() || a.enc_n < 1 || a.enc_channels < 1)
                        return no("'" + a.name + "' (the soundtrack of '" + m.name +
                                  "') was not prepared for the audio encoder");
                    a_t = audio_lat(a);
                    if (a_t < 1)
                        return no("the audio encoder refuses the soundtrack '" + a.name + "'");
                    j.enc_q.push_back({RAD_ENC_VAUDIO, m.index, (int32_t)k});
                }
                const int64_t v_t = video_lat(m);
                if (v_t < 1) return no("the video encoder refuses '" + m.name + "'");
                j.enc_q.push_back({RAD_ENC_VIDEO, m.index, (int32_t)i});
                if (j.ref_tab.empty()) j.ref_tab.assign(2, 0);
                ++j.ref_tab[0];
                j.ref_tab.insert(j.ref_tab.end(),
                                  {2 /*a video*/, m.index, (int32_t)(m.enc_h / d.enc_ratio),
                                   (int32_t)(m.enc_w / d.enc_ratio), (int32_t)v_t, (int32_t)a_t});
            } else if (m.kind == VideoRefMedia::AUDIO) {
                if (m.enc_samples.empty() || m.enc_n < 1 || m.enc_channels < 1)
                    return no("'" + m.name + "' was not prepared for the audio encoder");
                if (m.index < 1 || m.index > 3)
                    return no("'" + m.name + "' has no audio slot (1..3)");
                const int64_t a_t = audio_lat(m);
                if (a_t < 1) return no("the audio encoder refuses '" + m.name + "'");
                j.enc_q.push_back({RAD_ENC_AUDIO, m.index, (int32_t)i});
                if (j.ref_tab.empty()) j.ref_tab.assign(2, 0);
                ++j.ref_tab[0];
                j.ref_tab.insert(j.ref_tab.end(),
                                  {1 /*a standalone audio*/, m.index, 0, 0, 0, (int32_t)a_t});
            }
            /* VIDEO_AUDIO rows are folded into their video above, so they take no row here. */
        }
        /* Endpoint keyframes: each FIRST_FRAME/LAST_FRAME the server resized to the target canvas
         * and staged in enc_pixels folds into the pack's KEYFRAME table -- not a reference row. The
         * DiT reads its latent rows PINNED at that frame's moment on the target timeline (denoised
         * never): the first pins output frame 0, the last the final frame, and the gap between is
         * diffused. The key triple is {pixel frame, video_t, audio_t}; a picture anchor is video-only
         * (video_t = one latent frame, audio_t = 0), and its pixel frame is 0 or frames-1 -- the same
         * resolved frame index the reference's MiniMaxH3ImageToVideo assigns. Its latents ride the
         * image slot its keyframe ORDER names (1, then 2): a keyframe request carries no references,
         * so those slots are free, and ref_parts -- and the scatter -- read them back by the same
         * order. Keyframe rows are appended after every reference row, which is where ref_pack_of
         * reads the table's key block. */
        for (size_t i = 0; i < r.refs.size(); ++i) {
            const VideoRefMedia& m = r.refs[i];
            const bool first = m.kind == VideoRefMedia::FIRST_FRAME;
            if (!first && m.kind != VideoRefMedia::LAST_FRAME) continue;
            if (m.enc_pixels.empty())
                return no("'" + m.name + "' was not prepared for the video encoder");
            if (bad_side(m))
                return no("'" + m.name + "' has encoder sides (" + std::to_string(m.enc_h) +
                          "x" + std::to_string(m.enc_w) + ") the VAE grid of " +
                          std::to_string(d.enc_ratio) + " does not divide");
            if (!one_frame(m))
                return no("'" + m.name + "' is " + std::to_string(m.enc_h) + "x" +
                          std::to_string(m.enc_w) +
                          ", a single frame larger than the video encoder's capacity; a keyframe "
                          "encodes at the output resolution, so ask for a smaller size");
            if (j.ref_tab.empty()) j.ref_tab.assign(2, 0);   /* [n_refs, n_keys] */
            const int32_t slot = ++j.ref_tab[1];              /* keyframe order -> image slot 1, 2 */
            j.enc_q.push_back({RAD_ENC_IMAGE, slot, (int32_t)i});
            j.ref_tab.insert(j.ref_tab.end(),
                              {first ? 0 : s.frames - 1, 1 /*one latent frame*/, 0 /*no soundtrack*/});
        }
    }
    /* The step's packed sequence. When the denoiser consumes the reference table the pack is the
     * only source of its rows, segment boundaries and positions -- the reference rows sit between
     * the text and the target streams, so the plain `text + audio + video` arithmetic and the plain
     * position layout no longer describe it. rad_arch_video_pack returns the total rows and, when
     * roomy, the boundaries and the full [3, rows] planar RoPE in one call. Otherwise the plain
     * layout is computed exactly as before, and a staged reference table (if any) rides inert. */
    const int64_t nd = positions_(RAD_VPOS_DECODE, &j.geom, nullptr, 0);
    if (nd <= 0 || nd > max_tok_)
        return no("the model lays this video's decoder in " + std::to_string(nd) + " rows, which "
                  "this server cannot run");
    j.dec_rows = nd;
    if (d.ref_rows && !j.ref_tab.empty()) {
        if (r.upscale > 1.0)
            return no("reference conditioning does not yet combine with latent upscale; ask for "
                      "the size you want without latent_upscale");
        if (!pack_)
            return no("the generator says it conditions on references but exports no "
                      "rad_arch_video_pack");
        const int64_t pr = pack_(&j.geom, j.ref_tab.data(), nullptr, 0, nullptr, 0);
        if (pr <= 0)
            return no("the model refused the reference conditioning layout");
        if (pr > max_tok_)
            return no("a conditioned step is " + std::to_string(pr) + " rows and a pass carries " +
                      std::to_string(max_tok_) + "; ask for a shorter or smaller video, or fewer "
                      "references");
        j.seg.assign((size_t)kSegCap, 0);
        j.pos.assign((size_t)(3 * pr), 0.0f);
        if (pack_(&j.geom, j.ref_tab.data(), j.seg.data(), kSegCap, j.pos.data(), 3 * pr) != pr)
            return no("the model could not lay the reference conditioning out");
        int nseg = -1;
        for (int k = 0; k < kSegCap; ++k)
            if (j.seg[(size_t)k] == (int32_t)pr) { nseg = k + 1; break; }   /* video grid ends here */
        if (nseg < 0 || nseg > kSegCap)
            return no("the reference conditioning layout has no closing segment boundary");
        j.seg.resize((size_t)nseg);
        j.nseg = nseg;
        /* The modulation groups. A modulation cell is fixed by a segment's kind, so the denoiser
         * merges runs of consecutive same-cell segments and pitches its fixed-width table by the
         * GROUP count, not the reference count -- which is how nine reference photos (four groups:
         * prompt, all-photos, soundtrack, video) ride a table the fine layout (twelve segments)
         * would overrun. The plugin merges for us (rad_arch_video_mod_segs); one that exports none
         * merges nothing, so the group table is the fine one and the older per-segment ceiling
         * applies. Refuse over-wide merges here, at submit, where a clean answer is still possible. */
        int nmod = -1;
        if (mod_segs_) {
            j.mod_seg.assign((size_t)kSegCap, 0);
            nmod = (int)mod_segs_(&j.geom, j.ref_tab.data(), j.mod_seg.data(), kSegCap);
            if (nmod < 0 || nmod > kSegCap)
                return no("the model could not merge the reference conditioning into its "
                          "modulation groups");
            j.mod_seg.resize((size_t)nmod);
        } else {
            nmod = nseg;                 /* nothing merges: the groups are the fine segments */
            j.mod_seg = j.seg;
        }
        j.nmod = nmod;
        const int ngroups = nmod - 1;    /* `mod_seg` is boundaries: one more than the groups */
        if (d.max_ref_segs > 0 && ngroups > d.max_ref_segs)
            return no("these references change the conditioning's modulation cell " +
                      std::to_string(ngroups) + " times, but this deployment's modulation table holds " +
                      std::to_string(d.max_ref_segs) + " cells -- ask for references that share their "
                      "modalities and target, or fewer of them");
        /* `rows` stays the plain text+audio+video count: every non-step pass (encode, decode,
         * audio, upscale) stages its own segment table and target counts from it and must see the
         * bytes it sees today. Only build_step runs at the widened `step_rows`. */
        j.rows = j.geom.n_text + s.audio_rows + s.video_rows;
        j.step_rows = pr;
        /* The last segment (the video target) runs [seg[nseg-2], seg[nseg-1]); seg[nseg-1] is the
         * total rows, so the video start is the boundary before it -- the same one the VSA takes as
         * its video base (it works in segments, one fewer than the boundaries it is handed). */
        j.lat_v = pr - j.seg[(size_t)(nseg - 2)];
        j.lat_a = decl_.audio_channels * (int64_t)j.geom.n_audio;
        j.ref_wide = true;
    } else {
        j.rows = j.geom.n_text + s.audio_rows + s.video_rows;
        if (j.rows > max_tok_)
            return no("a step of this video is " + std::to_string(j.rows) + " rows and a pass "
                      "carries " + std::to_string(max_tok_) + "; ask for a shorter or smaller video");
        const int64_t nr = positions_(RAD_VPOS_STEP, &j.geom, nullptr, 0);
        if (nr != j.rows)
            return no("the model lays this video out in " + std::to_string(nr) + " step rows, "
                      "which this server cannot run");
        j.pos.assign((size_t)(3 * nr), 0.0f);
        if (positions_(RAD_VPOS_STEP, &j.geom, j.pos.data(), nr) != nr)
            return no("the model could not lay this video out");
        j.step_rows = j.rows;
    }

    /* The schedule follows the upscale's split, which needs the small size first. */
    const bool upscaling = r.upscale > 1.0;
    const bool own_refine = upscaling && d.n_refine_rungs > 0;
    int64_t first_tokens = s.video_rows;
    if (upscaling) {
        int32_t sw = 0, sh = 0;
        video_small_size(d, r.width, r.height, r.upscale, &sw, &sh);
        first_tokens = video_shape(d, sw, sh, chunks).video_rows;
    }
    j.sched = video_schedule(d, steps, first_tokens, own_refine, r.shift_video, r.shift_audio);
    if (j.sched.steps < 1) return no("the schedule has no steps");
    int32_t first_w = r.width, first_h = r.height;
    if (upscaling) {
        if (!(d.upscale_max > 1.0)) return no("this model has no latent upscaler");
        if (d.upscale_exact) {
            const int64_t m = (int64_t)std::llround(d.size_multiple * d.upscale_max);
            if (r.upscale != d.upscale_max || r.width % m || r.height % m)
                return no("this model's latent upscaler enlarges by exactly " +
                          fmt("%g", d.upscale_max) + ": latent_upscale must be that, and both "
                          "sides multiples of " + std::to_string(m));
        }
        if (r.upscale > d.upscale_max)
            return no("latent_upscale is at most " + fmt("%g", d.upscale_max) + " for this model");
        int32_t sw = 0, sh = 0;
        video_small_size(d, r.width, r.height, r.upscale, &sw, &sh);
        if (sw == r.width && sh == r.height)
            return no("latent_upscale " + fmt("%g", r.upscale) + " leaves " +
                      std::to_string(r.width) + "x" + std::to_string(r.height) + " as it is: "
                      "the small size is each side over the factor, rounded to a multiple of " +
                      std::to_string(d.size_multiple));
        const int32_t evals = j.sched.steps;
        int32_t refine = r.refine;
        if (own_refine) {
            if (refine != 0 && refine != d.n_refine_rungs)
                return no("upscale_refine_steps must be " + std::to_string(d.n_refine_rungs) +
                          ": this model refines the upscaled video over the schedule it was "
                          "trained on");
            refine = d.n_refine_rungs;
        }
        if (refine == 0) refine = std::min<int32_t>(d.upscale_refine, evals - 1);
        if (refine < 1 || refine >= evals)
            return no("upscale_refine_steps must be 1.." + std::to_string(evals - 1) + ": the "
                      "schedule takes " + std::to_string(evals) + " evaluations and the small "
                      "size at least one of them");
        const VideoShape ss = video_shape(d, sw, sh, chunks);
        j.geom_small = j.geom;
        j.geom_small.lat_h = ss.lat_h;
        j.geom_small.lat_w = ss.lat_w;
        j.rows_small = j.geom.n_text + ss.audio_rows + ss.video_rows;
        const int64_t ns = positions_(RAD_VPOS_STEP, &j.geom_small, nullptr, 0);
        if (ns != j.rows_small)
            return no("the model lays the small size " + std::to_string(sw) + "x" +
                      std::to_string(sh) + " out in " + std::to_string(ns) + " step rows, which "
                      "this server cannot run");
        j.pos_small.assign((size_t)(3 * ns), 0.0f);
        if (positions_(RAD_VPOS_STEP, &j.geom_small, j.pos_small.data(), ns) != ns)
            return no("the model could not lay the small size out");
        j.split = evals - refine;
        j.factor = (float)(((double)r.width / sw + (double)r.height / sh) / 2.0);
        r.ticket->small_width = sw;
        r.ticket->small_height = sh;
        r.ticket->refine = refine;
        first_w = sw;
        first_h = sh;
    }
    if (!r.image.empty()) {
        if (!d.image_cond) return no("this model cannot start from a picture");
        if (r.image.size() != (size_t)first_w * first_h * 3)
            return no("the first frame is not " + std::to_string(first_w) + "x" +
                      std::to_string(first_h) + " RGB");
    }
    j.n_chunks = chunks;
    j.n_samples = (int64_t)s.n_audio * (d.sample_rate / d.audio_rate);
    VideoTicket& t = *r.ticket;
    t.frames = s.frames;
    t.steps = j.sched.steps;
    t.rows = j.step_rows;   /* the conditioned step's real length, references included */
    return RAD_OK;
}

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
    /* Reference media need the vision tower (their identity soft tokens). Endpoint keyframes do not:
     * they condition through the VAE-latent pin, so a container converted without its tower can still
     * render them -- the server simply lays no `media` for them and the pin-only path runs. (Keyframes
     * still need the VAE encoders, checked just below.) */
    bool has_ref_media = false;
    for (const VideoRefMedia& m : r->refs)
        if (m.kind != VideoRefMedia::FIRST_FRAME && m.kind != VideoRefMedia::LAST_FRAME)
            has_ref_media = true;
    if (has_ref_media && max_enc_ == 0)
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
                           : m.kind == VideoRefMedia::VIDEO_AUDIO ? "soundtrack"
                           : m.kind == VideoRefMedia::FIRST_FRAME ||
                             m.kind == VideoRefMedia::LAST_FRAME ? "keyframe" : "audio") +
                          ", but this container was converted without its VAE encoders, so nothing "
                          "can condition on it");
    /* Only pictures (reference images and endpoint keyframes) add language-model soft-token rows, so
     * only they need processed media. A reference video or audio reaches the model purely through the
     * VAE encode rows the pack lays from the reference table, so a request of those alone carries no
     * `media` and needs none. A keyframe with no tower also carries no `media` — that is the deliberate
     * pin-only fallback, not a silent weakening (its VAE pin still conditions the DiT). */
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
    if (r->chunks != 0 && (r->chunks < d.min_chunks || r->chunks > d.max_chunks))
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
    const bool evals = d.sched_kind == 1;   /* form 1 counts evaluations, form 0 grid points */
    const int32_t rung_steps = evals ? d.n_rungs : d.n_rungs + 1;
    if (d.n_rungs > 0 && steps != rung_steps)
        return no("num_inference_steps must be " + std::to_string(rung_steps) + ": this model "
                  "runs the schedule it was trained on, " + std::to_string(d.n_rungs) +
                  " evaluations");
    if (steps < (evals ? 1 : 2) || steps > d.max_steps)
        return no("num_inference_steps must be " + std::to_string(evals ? 1 : 2) + ".." +
                  std::to_string(d.max_steps) + (evals ? " (model evaluations)" :
                  " (sigma grid points, the last one 0)"));

    auto j = std::make_unique<Job>();
    /* Guidance: the request's terms over the model's defaults; a negative prompt only when a
     * scale asks for the unconditional branch. */
    bool named = r->has_neg;
    for (int i = 0; i < 8; ++i) {
        named = named || !std::isnan(r->guidance[i]);
        j->guidance[i] = (float)(std::isnan(r->guidance[i]) ? d.guidance_default[i] : r->guidance[i]);
        if (!(j->guidance[i] >= 0.0f) || !std::isfinite(j->guidance[i]))
            return no("guidance terms must be finite and not negative");
    }
    if (!d.guidance && named)
        return no("this model takes no guidance or negative prompt");
    j->n_neg = 0;
    if (d.guidance && (j->guidance[0] != 1.0f || j->guidance[4] != 1.0f)) {
        const int64_t N = (int64_t)r->neg_tokens.size();
        if (N < 1 || N > d.max_text)
            return no("the negative prompt is " + std::to_string(N) + " tokens; this model takes "
                      "1.." + std::to_string(d.max_text));
        j->n_neg = (int32_t)N;
    }
    if (r->decoder < 0 || (r->decoder > 0 && (size_t)r->decoder >= decoder_area_.size()))
        return no("this model has no decoder " + std::to_string(r->decoder));
    if (r->decoder > 0 && W * H > decoder_area_[(size_t)r->decoder])
        return no("the " + decoder_names_[(size_t)r->decoder] + " decoder makes frames of " +
                  std::to_string(decoder_area_[(size_t)r->decoder]) + " pixels at most");
    j->auto_len = r->chunks == 0 && d.duration_out != RAD_NULL_HANDLE;
    const int32_t chunks = r->chunks == 0 ? (int32_t)d.default_chunks : r->chunks;
    if (shape_job(*j, *r, chunks, steps, why) < 0) return RAD_E_INVAL;
    /* The soundtrack's length, exactly the samples the decode will hand the encoder -- the
     * number a caller that chains clips against an audio floor checks. A build without
     * libavcodec never gets the muxed file to read it back from, so it is logged here. */
    RAD_INFO("video: soundtrack %.2f s, %lld samples at %lld Hz x %lld ch",
             (double)j->n_samples / d.sample_rate, (long long)j->n_samples,
             (long long)d.sample_rate, (long long)d.audio_channels);

    VideoTicket& t = *r->ticket;
    t.width = r->width;
    t.height = r->height;
    t.sample_rate = d.sample_rate;
    t.channels = d.audio_channels;
    t.fps = (double)d.fps_num / (double)d.fps_den;
    t.in_tokens = T;
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
    g[RAD_VGEOM_AUDIO_VOL] = g0.n_audio;
    g[RAD_VGEOM_AUDIO_VOL + 1] = 1;
    g[RAD_VGEOM_AUDIO_VOL + 2] = 1;
    const int32_t ppf = (int32_t)((g0.lat_h / decl_.patch) * (g0.lat_w / decl_.patch));
    g[RAD_VGEOM_FIRST_CU] = 0;
    g[RAD_VGEOM_FIRST_CU + 1] = ppf;
    g[RAD_VGEOM_FIRST_CU + 2] = ppf * g0.lat_frames;
    reg_[R_GEOM].used = RAD_VGEOM_N * 4;
    const bool wide = j.ref_wide && ref_step;
    const int64_t audio_rows =
        (decl_.audio_streams > 0 ? decl_.audio_streams : decl_.audio_channels) * (int64_t)g0.n_audio;
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
    std::memcpy(hptr<float>(R_GUIDE), j.guidance, sizeof j.guidance);
    reg_[R_GUIDE].used = sizeof j.guidance;
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
        V.sampler = j.req->sampler;
        V.sched = dptr<float>(ri, R_SCHED);
        V.pos = dptr<float>(ri, R_ROPE);
        V.seed = dptr<uint32_t>(ri, R_SEED);
        V.chunk = j.chunk;
        V.n_chunks = j.n_chunks;
        V.dec_rows = j.dec_rows;
        V.n_samples = j.n_samples;
        V.upscale = dptr<float>(ri, R_UP);
        V.ref_tab = j.ref_tab.empty() ? nullptr : j.ref_tab.data();
        V.guidance_h = j.guidance;
        V.guidance = dptr<float>(ri, R_GUIDE);
        V.has_image = !j.req->image.empty();
        V.decoder = j.req->decoder;
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
    /* The prompt, then the negative prompt when the request is guided against one, each its own
     * sequence from position 0. */
    const int64_t T = (int64_t)j.req->tokens.size(), N = j.n_neg;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    std::memcpy(tok, j.req->tokens.data(), (size_t)T * 4);
    if (N > 0) std::memcpy(tok + T, j.req->neg_tokens.data(), (size_t)N * 4);
    /* A left-padded encoder's prompt ends at text_left_pad - 1. */
    const int64_t lp = decl_.text_left_pad;
    for (int64_t i = 0; i < T; ++i) pos[i] = (int32_t)(lp > 0 ? lp - T + i : i);
    for (int64_t i = 0; i < N; ++i) pos[T + i] = (int32_t)(lp > 0 ? lp - N + i : i);
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)T;
    cu[2] = (int32_t)(T + N);
    /* The first frame as the reference's image processor hands it to the VAE: x / 255 * 2 - 1 in
     * f32, then bf16. */
    const std::vector<uint8_t>& img = j.req->image;
    if (!img.empty()) {
        uint16_t* px = hptr<uint16_t>(R_IMG);
        for (size_t i = 0; i < img.size(); ++i)
            px[i] = rad_f32_to_bf16(2.0f * ((float)img[i] / 255.0f) - 1.0f);
        reg_[R_IMG].used = (int64_t)img.size() * 2;
    }
    float* tv = hptr<float>(R_TV);
    float* ta = hptr<float>(R_TA);
    for (int32_t i = 0; i < j.sched.steps; ++i) {
        tv[i] = 1.0f - j.sched.sig_v[(size_t)i];
        ta[i] = 1.0f - j.sched.sig_a[(size_t)i];
    }
    reg_[R_TOK].used = (T + N) * 4;
    reg_[R_POS].used = (T + N) * 4;
    reg_[R_CU].used = (N > 0 ? 3 : 2) * 4;
    reg_[R_TV].used = j.sched.steps * 4;
    reg_[R_TA].used = j.sched.steps * 4;
    if (dit_dbg_)
        std::fprintf(stderr, "dit_dbg: t arrays staged: tv[0]=%.4f tv[%d]=%.4f ta[0]=%.4f ta[%d]=%.4f\n",
                     tv[0], (int)j.sched.steps - 1, tv[j.sched.steps - 1],
                     ta[0], (int)j.sched.steps - 1, ta[j.sched.steps - 1]);
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
    base_batches(j, g, T + N, step);
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
    for (size_t r = 0; r < out_.size(); ++r) {
        RadBatch& B = out_[r];
        B.gen = RAD_GEN_PASS_VCOND;
        B.n_seq = N > 0 ? 2 : 1;
        B.max_q_len = B.max_q_len_prefill = (int32_t)std::max(T, N);
        vout_[r].n_neg = (int32_t)N;
        if (!img.empty()) vout_[r].image = dptr<uint16_t>((int)r, R_IMG);
    }
    if (j.auto_len) rb_.assign(1, Readback{ decl_.duration_out, 4, duration_host_ });
    pass_.kind = RAD_GEN_PASS_VCOND;
    pass_.job = j.id;
    pass_.lora_slot = j.req->lora_slot;
    return &out_[0];
}

/* Step i's timesteps, sigmas and sampler coefficients (RadVideoGenBatch::sched). With `to_clean` the
 * next sigmas are 0, so the step lands on its clean estimate. Both samplers' scalars are staged
 * whichever runs, so the plugin reads the slots its sampler names:
 *   [0..5]   Euler: video (t, sigma, sigma_next) then soundtrack.
 *   [6..10]  res_multistep, video: sigma (= the velocity's weight), r = sigma_next/sigma, hb1, hb2,
 *            order (1 or 2).
 *   [11..15] res_multistep, soundtrack: the same five.
 * The coefficients are DPM-Solver++2M data-prediction / k-diffusion res_multistep at eta=0: with
 * h = log(sigma/sigma_next), h_last = log(sigma_prev/sigma) and rs = h_last/h,
 *   hb1 = (1 - r)(1 + 1/(2 rs)),  hb2 = -(1 - r)/(2 rs),  order = 2.
 * They are used only on a step that has a previous estimate on the SAME latent grid, so `order` is 1
 * (plain Euler, hb unused) on the first step, on the first full-size step after a latent upscale
 * (i == split, whose grid and rows differ from the small run's stale history) and on a step whose
 * next sigma is 0 (the last, or the last small step). hb1 + hb2 == 1 - r on the two-estimate steps,
 * so every branch shares the Euler limit. */
void VideoDriver::stage_sched(const Job& j, int32_t i, bool to_clean) {
    float* s = hptr<float>(R_SCHED);
    s[0] = 1.0f - j.sched.sig_v[(size_t)i];
    s[1] = j.sched.sig_v[(size_t)i];
    s[2] = to_clean ? 0.0f : j.sched.sig_v[(size_t)i + 1];
    s[3] = 1.0f - j.sched.sig_a[(size_t)i];
    s[4] = j.sched.sig_a[(size_t)i];
    s[5] = to_clean ? 0.0f : j.sched.sig_a[(size_t)i + 1];
    /* A stream's res_multistep coefficients, written to out[0..4]. `prev` is the step before this
     * one's sigma (>= i so i > 0), and the restart step (i == split) forces Euler. */
    auto coeffs = [&](const std::vector<float>& sig, float* out) {
        const float sg = sig[(size_t)i];
        const float sgn = to_clean ? 0.0f : sig[(size_t)i + 1];
        const float r = sg > 0.0f ? sgn / sg : 0.0f;
        float hb1 = 1.0f - r, hb2 = 0.0f, order = 1.0f;
        const bool two = sgn > 0.0f && i > 0 && !(j.split > 0 && i == j.split);
        if (two) {
            const float h = std::log(sg / sgn);
            const float h_last = std::log(sig[(size_t)i - 1] / sg);
            const float rs = h_last / h;
            hb1 = (1.0f - r) * (1.0f + 1.0f / (2.0f * rs));
            hb2 = -(1.0f - r) / (2.0f * rs);
            order = 2.0f;
        }
        out[0] = sg;
        out[1] = r;
        out[2] = hb1;
        out[3] = hb2;
        out[4] = order;
    };
    coeffs(j.sched.sig_v, s + 6);
    coeffs(j.sched.sig_a, s + 11);
    reg_[R_SCHED].used = 16 * 4;
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

/* DiT step dump (#61, R4D_DIT_SEL + R4D_ENC_DUMPDIR): write one raw file named verbatim. Separate
 * from enc_dbg_dump_file because the harness names carry their own dtype suffixes (.f32 / .i32). */
static void dit_dump_file(const std::string& name, const void* p, size_t bytes) {
    const char* dir = std::getenv("R4D_ENC_DUMPDIR");
    if (!dir || !p || bytes == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(p, 1, bytes, f);
        std::fclose(f);
    }
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
    /* DiT step dump (#61, R4D_DIT_SEL): the step-0 readback of the plugin's f32 shadows and the
     * velocity head, plus the host-side bytes the CPU reference replay needs (the planar RoPE
     * positions and the packed segment boundaries -- final in their regions since upload() ran).
     * Plain step 0 only: the harness replays the fixed [text | audio | video] layout and the full
     * grid, and the arch issues the shadows only under the same conditions. */
    dit_used_ = false;
    if (dit_dbg_ && (j.step == 0 || j.step == 1 || j.step == dit_sel_) && !small &&
        !j.ref_wide && j.ref_tab.empty()) {
        dit_step_ = j.step;   /* namespaced dump: step 0 -> dit_*, step K>0 -> dit_sK_* */
        char* p = dit_host_;
        int64_t off = 0;
        const rad_buf hh[7] = { decl_.dbg_dit_h0, decl_.dbg_dit_temb, decl_.dbg_dit_mod,
                                decl_.dbg_dit_blk, decl_.dbg_dit_hn, decl_.dbg_dit_nout,
                                decl_.dbg_dit_vel };
        const int64_t nn[7] = { dit_h0_b_, dit_temb_b_, dit_mod_b_, dit_blk_b_, dit_hn_b_,
                                dit_nout_b_, dit_vel_b_ };
        for (int i = 0; i < 7; ++i)
            if (hh[i] && nn[i] > 0) {
                rb_.push_back(Readback{ hh[i], nn[i], p + off });
                off += nn[i];
            }
        dit_used_ = true;
        dit_rows_ = rows;
        const std::string tag = dit_step_ ? "dit_s" + std::to_string(dit_step_) + "_" : "dit_";
        dit_dump_file(tag + "pos.f32", pos.data(), (size_t)pos.size() * 4);
        dit_dump_file(tag + "seg_cu.i32", hptr<int32_t>(R_SEG), (size_t)reg_[R_SEG].used);
        dit_dump_file(tag + "tvid.f32", hptr<float>(R_TV), (size_t)j.sched.steps * 4);
        dit_dump_file(tag + "tarr.f32", hptr<float>(R_TA), (size_t)j.sched.steps * 4);
    }
    pass_.kind = RAD_GEN_PASS_VSTEP;
    pass_.job = j.id;
    pass_.lora_slot = j.req->lora_slot;
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
    stage_seed(j);   /* a decoder that denoises draws from the request's seed */
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
    /* #63: the decoder fingerprint chain, staged once (chunk 0) after the frames readback. */
    if (dit_dbg_ && dec_lat_b_ > 0 && j.chunk == 0) {
        char* p = dit_host_ + dit_h0_b_ + dit_vel_b_ + dit_temb_b_ + dit_mod_b_ + dit_blk_b_
                  + dit_hn_b_ + dit_nout_b_;
        rb_.push_back(Readback{ decl_.dbg_vd_lat, dec_lat_b_, p });
        p += dec_lat_b_;
        rb_.push_back(Readback{ decl_.dbg_vd_x0, dec_x0_b_, p });
        p += dec_x0_b_;
        rb_.push_back(Readback{ decl_.dbg_vd_blk, dec_blk_b_, p });
        p += dec_blk_b_;
        rb_.push_back(Readback{ decl_.dbg_vd_n, dec_n_b_, p });
        p += dec_n_b_;
        rb_.push_back(Readback{ decl_.dbg_vd_pix, dec_pix_b_, p });
        dec_geom_[0] = j.geom.lat_h;
        dec_geom_[1] = j.geom.lat_w;
        dec_geom_[2] = n;
        dec_used_ = true;
    }
    pass_.kind = RAD_GEN_PASS_VDECODE;
    pass_.job = j.id;
    pass_.n_frames = n;
    return &out_[0];
}

/* Debug-only raw dump (R4D_ENC_DEBUG + R4D_ENC_DUMPDIR): write the encoder's input / output bytes to
 * a file so tests/data/ref2va/ref_aenc_parity.py can compare the device's encode kernels against the
 * reference HF encoder on the SAME input. Inert unless both environment variables are set — a normal
 * run never enters here. Part of the R4D_ENC_DEBUG scaffolding, stripped at PR time. */
static void enc_dbg_dump_file(const std::string& tag, const void* p, size_t bytes) {
    const char* dir = std::getenv("R4D_ENC_DUMPDIR");
    if (!dir || !p || bytes == 0) return;
    const std::string path = std::string(dir) + "/" + tag + ".f32";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(p, 1, bytes, f);
        std::fclose(f);
    }
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
        if (enc_dbg_)
            enc_dbg_dump_file("encsm_" + std::to_string((int)it.kind) + "_" +
                              std::to_string((int)it.slot), sm, (size_t)stage_samples * ch * 4);
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
        if (enc_dbg_)
            enc_dbg_dump_file("encpx_" + std::to_string((int)it.kind) + "_" +
                              std::to_string((int)it.slot), px, (size_t)stage_frames * area * 4);
    }
    reg_[R_ENCG].used = 25 * 4;
    if (enc_dbg_)
        enc_dbg_dump_file("encg_" + std::to_string((int)it.kind) + "_" +
                          std::to_string((int)it.slot), gd, 25 * 4);
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
        /* R4D_ENC_SEL picks which audio-pass buffer the readback names, so one binary can dump the
         * whole final chain without a rebuild: "enca" (default, the normalised latent), "mp" (the
         * mean_proj output, i.e. the latent before the fixed (z-mean)/std norm), "amb" (the mean_proj
         * INPUT -- the pre_block output cast to f32), or "pre" (the raw bf16 pre_block buffer b_ax2).
         * The split localises a latent-scale gap: if the gain is already present at "mp"/"amb", the
         * norm op is innocent and the body carries it. "amb" is the RELIABLE f32 twin of "pre": the
         * bf16 b_ax2 is arena-rotated scratch whose readback can lag the pass (b_amb2 is finite while
         * b_ax2 dumps ±1e38 -- a stale read), so the f32 b_amb is what to trust for the pre_block
         * value. The video pass and the default audio pass are untouched; only an env var selects. */
        const char* sel = std::getenv("R4D_ENC_SEL");
        const bool mp  = sel &&  audio && std::strcmp(sel, "mp")  == 0 && decl_.dbg_ae_mp_bytes  > 0;
        const bool amb = sel &&  audio && std::strcmp(sel, "amb") == 0 && decl_.dbg_ae_amb_bytes > 0;
        const bool pre = sel &&  audio && std::strcmp(sel, "pre") == 0 && decl_.dbg_ae_pre_bytes > 0;
        const bool q2  = sel && !audio && std::strcmp(sel, "q2")  == 0 && decl_.dbg_ve_q2_bytes  > 0;
        const bool q48 = sel && !audio && std::strcmp(sel, "q48") == 0 && decl_.dbg_ve_q48_bytes > 0;
        const bool nout = sel && !audio && std::strcmp(sel, "nout") == 0 && decl_.dbg_ve_nout_bytes > 0;
        const bool cin  = sel && !audio && std::strcmp(sel, "cin")  == 0 && decl_.dbg_ve_cin_bytes  > 0;
        /* "all" reads the three video shadows back in ONE pass into one slab (nout | q48 | q2), so a
         * single request localises a latent-scale gain across norm_out / conv_out / quant_conv. */
        const bool all = sel && !audio && std::strcmp(sel, "all") == 0 &&
                         decl_.dbg_ve_trunk_bytes > 0 && decl_.dbg_ve_nout_bytes > 0 &&
                         decl_.dbg_ve_q48_bytes > 0 && decl_.dbg_ve_q2_bytes > 0 &&
                         decl_.dbg_ve_l0_bytes > 0 && decl_.dbg_ve_l1_bytes > 0 &&
                         decl_.dbg_ve_l2_bytes > 0 && decl_.dbg_ve_l3_bytes > 0 &&
                         decl_.dbg_ve_l4_bytes > 0;
        enc_dbg_used_ = 0;
        if (all && enc_dbg_bytes_ >= decl_.dbg_ve_trunk_bytes + decl_.dbg_ve_nout_bytes +
                                    decl_.dbg_ve_q48_bytes + decl_.dbg_ve_q2_bytes +
                                    decl_.dbg_ve_l0_bytes + decl_.dbg_ve_l1_bytes +
                                    decl_.dbg_ve_l2_bytes + decl_.dbg_ve_l3_bytes + decl_.dbg_ve_l4_bytes) {
            /* One slab, fixed order: trunk | nout | q48 | q2 | l0 | l1 | l2 | l3 | l4. */
            const rad_buf  hh[9] = { decl_.dbg_ve_trunk, decl_.dbg_ve_nout, decl_.dbg_ve_q48,
                                     decl_.dbg_ve_q2, decl_.dbg_ve_l0, decl_.dbg_ve_l1,
                                     decl_.dbg_ve_l2, decl_.dbg_ve_l3, decl_.dbg_ve_l4 };
            const int64_t  nn[9] = { decl_.dbg_ve_trunk_bytes, decl_.dbg_ve_nout_bytes,
                                     decl_.dbg_ve_q48_bytes, decl_.dbg_ve_q2_bytes,
                                     decl_.dbg_ve_l0_bytes, decl_.dbg_ve_l1_bytes,
                                     decl_.dbg_ve_l2_bytes, decl_.dbg_ve_l3_bytes, decl_.dbg_ve_l4_bytes };
            char* p = enc_dbg_host_;
            int64_t off = 0;
            rb_.clear();
            for (int i = 0; i < 9; ++i) { rb_.push_back(Readback{ hh[i], nn[i], p + off }); off += nn[i]; }
            enc_dbg_sel_ = "all";
            enc_dbg_used_ = off;
        } else {
            const rad_buf h = mp ? decl_.dbg_ae_mp : amb ? decl_.dbg_ae_amb : pre ? decl_.dbg_ae_pre
                                : nout ? decl_.dbg_ve_nout : q48 ? decl_.dbg_ve_q48 : q2 ? decl_.dbg_ve_q2
                                : cin ? decl_.dbg_ve_cin
                                : audio ? decl_.dbg_enca : decl_.dbg_evc;
            const int64_t n = mp ? decl_.dbg_ae_mp_bytes : amb ? decl_.dbg_ae_amb_bytes
                                : pre ? decl_.dbg_ae_pre_bytes : nout ? decl_.dbg_ve_nout_bytes
                                : q48 ? decl_.dbg_ve_q48_bytes : q2 ? decl_.dbg_ve_q2_bytes
                                : cin ? decl_.dbg_ve_cin_bytes
                                : audio ? decl_.dbg_enca_bytes : decl_.dbg_evc_bytes;
            enc_dbg_sel_ = mp ? "mp" : amb ? "amb" : pre ? "pre" : nout ? "nout" : q48 ? "q48" : q2 ? "q2"
                             : cin ? "cin" : "";
            if (h && n > 0) {
                rb_.assign(1, Readback{h, n, enc_dbg_host_});
                enc_dbg_used_ = n;
            }
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
        if (j->auto_len) {
            j->auto_len = false;
            std::string note, why;
            int32_t n = predicted_chunks((double)duration_host_[0], &note);
            /* Shaped as at submit, which passed at default_chunks; a longer video than a pass
             * can carry is shortened until it fits. */
            const int32_t steps = j->req->steps > 0 ? j->req->steps : (int32_t)decl_.default_steps;
            const int32_t at_submit = j->n_chunks;
            while (n > at_submit && shape_job(*j, *j->req, n, steps, &why) < 0) --n;
            if (n <= at_submit && shape_job(*j, *j->req, n, steps, &why) < 0) {
                fail_locked(*j, RAD_E_INVAL, why);
                break;
            }
            RAD_INFO("video: the model gives the prompt %d frames (%.2f s)%s", j->geom.frames,
                     (double)j->geom.frames * decl_.fps_den / decl_.fps_num, note.c_str());
            VideoEvent e;
            e.kind = VideoEvent::LENGTH;
            e.n = j->geom.frames;
            j->req->ticket->push(std::move(e));
        }
        break;
    case RAD_GEN_PASS_VENCODE:
    case RAD_GEN_PASS_AENCODE:
        /* The media's latents are now in the lane's slot rows; take the next reference, if any. */
        if (enc_dbg_ && enc_dbg_used_ > 0 && enc_dbg_sel_ && std::strcmp(enc_dbg_sel_, "all") == 0) {
            /* Nine shadows in one slab -> nine files. */
            const char* nm8[9] = { "evctrunk", "evcnout", "evcq48", "evcq2",
                                   "evcl0", "evcl1", "evcl2", "evcl3", "evcl4" };
            const int64_t nn8[9] = { decl_.dbg_ve_trunk_bytes, decl_.dbg_ve_nout_bytes,
                                     decl_.dbg_ve_q48_bytes, decl_.dbg_ve_q2_bytes,
                                     decl_.dbg_ve_l0_bytes, decl_.dbg_ve_l1_bytes,
                                     decl_.dbg_ve_l2_bytes, decl_.dbg_ve_l3_bytes, decl_.dbg_ve_l4_bytes };
            char* p = enc_dbg_host_;
            int64_t off = 0;
            for (int i = 0; i < 9; ++i) {
                enc_dbg_dump_file(nm8[i], p + off, (size_t)nn8[i]);
                off += nn8[i];
            }
            RAD_ERR("enc_dbg all: dumped %lld bytes across trunk/nout/q48/q2/l0/l1/l2/l3/l4",
                    (long long)off);
        } else if (enc_dbg_ && enc_dbg_used_ > 0) {
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
            std::string tagstr = pass_.kind != RAD_GEN_PASS_AENCODE
                               ? (*enc_dbg_sel_ ? std::string("evc") + enc_dbg_sel_ : std::string("evc"))
                               : (*enc_dbg_sel_ ? std::string("enc") + enc_dbg_sel_ : std::string("enca"));
            const char* tag = tagstr.c_str();
            RAD_ERR("enc_dbg %s: fnv %016llx floats %lld nonzero %lld maxabs %.6g "
                    "sum %.6g  head[%.6g %.6g %.6g %.6g]",
                    tag, (unsigned long long)h, (long long)nf, (long long)nz,
                    (double)mx, sum, (double)z[0], (double)z[1], (double)z[2], (double)z[3]);
            enc_dbg_dump_file(tag, enc_dbg_host_, (size_t)enc_dbg_used_);
        }
        ++j->enc_next;
        break;
    case RAD_GEN_PASS_VSTEP: {
        ++j->step;
        if (dit_used_) {
            /* The step-0 dump landed: slab (h0 | temb | mod | blk | hn | nout | vel) to raw
             * files, plus the few facts the driver knows (R4D_ENC_DUMPDIR; the prep script
             * supplies D=5376 and the harness .meta files). */
            const std::string tag = dit_step_ ? "dit_s" + std::to_string(dit_step_) + "_" : "dit_";
            char* p = dit_host_;
            dit_dump_file(tag + "h0.f32", p, (size_t)dit_h0_b_);
            p += dit_h0_b_;
            dit_dump_file(tag + "temb.f32", p, (size_t)dit_temb_b_);
            p += dit_temb_b_;
            dit_dump_file(tag + "mod.f32", p, (size_t)dit_mod_b_);
            p += dit_mod_b_;
            dit_dump_file(tag + "blk.f32", p, (size_t)dit_blk_b_);
            p += dit_blk_b_;
            dit_dump_file(tag + "hn.f32", p, (size_t)dit_hn_b_);
            p += dit_hn_b_;
            dit_dump_file(tag + "nout.f32", p, (size_t)dit_nout_b_);
            p += dit_nout_b_;
            dit_dump_file(tag + "vel.f32", p, (size_t)dit_vel_b_);
            if (std::getenv("R4D_ENC_DUMPDIR")) {
                const std::string path = std::string(std::getenv("R4D_ENC_DUMPDIR")) + "/" +
                                         tag + "info.txt";
                if (std::FILE* f = std::fopen(path.c_str(), "w")) {
                    std::fprintf(f, "rows %lld\nn_text %lld\nn_audio %lld\nsteps %d\nseed %llu\n",
                                 (long long)dit_rows_, (long long)j->geom.n_text,
                                 (long long)j->geom.n_audio, (int)j->sched.steps,
                                 (unsigned long long)j->req->seed);
                    std::fclose(f);
                }
            }
            RAD_ERR("dit_dbg: dumped step 0 -- rows %lld h0 %lld temb %lld mod %lld blk %lld "
                    "hn %lld nout %lld vel %lld",
                    (long long)dit_rows_, (long long)dit_h0_b_, (long long)dit_temb_b_,
                    (long long)dit_mod_b_, (long long)dit_blk_b_, (long long)dit_hn_b_,
                    (long long)dit_nout_b_, (long long)dit_vel_b_);
            dit_used_ = false;
        }
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
        if (dec_used_) {
            /* The chunk-0 decoder fingerprint (lat | x0 | blk | n | pix) to raw files for the CPU
             * replay of comfy's ViT3DDecoder (R4D_ENC_DUMPDIR; #63). */
            char* p = dit_host_ + dit_h0_b_ + dit_vel_b_ + dit_temb_b_ + dit_mod_b_ + dit_blk_b_
                      + dit_hn_b_ + dit_nout_b_;
            dit_dump_file("vdec_lat.bf16", p, (size_t)dec_lat_b_);
            p += dec_lat_b_;
            dit_dump_file("vdec_x0.bf16", p, (size_t)dec_x0_b_);
            p += dec_x0_b_;
            dit_dump_file("vdec_blk.f32", p, (size_t)dec_blk_b_);
            p += dec_blk_b_;
            dit_dump_file("vdec_n.f32", p, (size_t)dec_n_b_);
            p += dec_n_b_;
            dit_dump_file("vdec_pix.f32", p, (size_t)dec_pix_b_);
            if (std::getenv("R4D_ENC_DUMPDIR")) {
                const std::string path =
                    std::string(std::getenv("R4D_ENC_DUMPDIR")) + "/vdec_info.txt";
                if (std::FILE* f = std::fopen(path.c_str(), "w")) {
                    std::fprintf(f, "lat_h %lld\nlat_w %lld\nn_frames %lld\nseed %llu\n",
                                 (long long)dec_geom_[0], (long long)dec_geom_[1],
                                 (long long)dec_geom_[2], (unsigned long long)j->req->seed);
                    std::fclose(f);
                }
            }
            RAD_ERR("vd_dbg: dumped decode of chunk 0 -- lat %lld x0 %lld blk %lld n %lld pix %lld",
                    (long long)dec_lat_b_, (long long)dec_x0_b_, (long long)dec_blk_b_,
                    (long long)dec_n_b_, (long long)dec_pix_b_);
            dec_used_ = false;
        }
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
