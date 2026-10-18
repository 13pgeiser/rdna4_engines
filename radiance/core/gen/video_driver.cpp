/* video_driver.cpp -- the video generator's pass driver (video_driver.h), the ticket a caller
 * waits on, and a request's shape and schedule (video.h). */
#include "video_driver.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

int64_t align_up(int64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

/* Region offsets and sizes, in the enum's order (video_driver.h), and the total; shared by the
 * allocation and the VRAM budget. */
int64_t carve(const RadVideoGenDecl& d, int64_t max_tok, int64_t (*bytes)[2], int n) {
    const int64_t want[] = {
        d.max_text * 4,          /* R_TOK */
        d.max_text * 4,          /* R_POS */
        2 * 4,                   /* R_CU */
        3 * max_tok * 4,         /* R_ROPE: a step's rotary positions */
        RAD_VGEOM_N * 4,         /* R_GEOM */
        6 * 4,                   /* R_SCHED */
        d.max_steps * 4,         /* R_TV */
        d.max_steps * 4,         /* R_TA */
        4 * 4,                   /* R_SEED */
        4,                       /* R_LANE */
        4,                       /* R_FIRST */
        4 * 4,                   /* R_SEG */
        4 * 4,                   /* R_LAT */
        4,                       /* R_UP */
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

int64_t VideoDriver::device_bytes(const Program& prog) {
    if (!prog.is_video_generator()) return 0;
    int64_t b[R_N][2];
    return carve(prog.video_gen, prog.ctx.max_tok, b, R_N);
}

int VideoDriver::init(const Program& prog, PositionsFn positions,
                      const std::vector<RankIO>& ranks) {
    fini();
    if (!prog.is_video_generator()) return RAD_E_STATE;
    decl_  = prog.video_gen;
    rungs_ = prog.vgen_str.rungs;
    decl_.rungs = rungs_.empty() ? nullptr : rungs_.data();
    name_  = prog.vgen_str.name;
    ranks_ = ranks;
    positions_ = positions;
    max_tok_ = prog.ctx.max_tok;
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
    bytes_ = carve(decl_, max_tok_, b, R_N);
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
    host_ = nullptr;
    frames_host_ = nullptr;
    audio_host_ = nullptr;
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
    j->rows = T + s.audio_rows + s.video_rows;
    if (j->rows > max_tok_)
        return no("a step of this video is " + std::to_string(j->rows) + " rows and a pass "
                  "carries " + std::to_string(max_tok_) + "; ask for a shorter or smaller video");
    const int64_t nr = positions_(RAD_VPOS_STEP, &j->geom, nullptr, 0);
    const int64_t nd = positions_(RAD_VPOS_DECODE, &j->geom, nullptr, 0);
    if (nr != j->rows || nd <= 0 || nd > max_tok_)
        return no("the model lays this video out in " + std::to_string(nr) + " step rows and " +
                  std::to_string(nd) + " decoder rows, which this server cannot run");
    j->dec_rows = nd;
    j->pos.assign((size_t)(3 * nr), 0.0f);
    if (positions_(RAD_VPOS_STEP, &j->geom, j->pos.data(), nr) != nr)
        return no("the model could not lay this video out");
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

    VideoTicket& t = *r->ticket;
    t.frames = s.frames;
    t.width = r->width;
    t.height = r->height;
    t.steps = j->sched.steps;
    t.sample_rate = d.sample_rate;
    t.channels = d.audio_channels;
    t.fps = (double)d.fps_num / (double)d.fps_den;
    t.in_tokens = T;
    t.rows = j->rows;
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
void VideoDriver::stage_geom(const Job& j, const RadVideoGeom& g0, int64_t rows, int32_t chunk) {
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
    const int64_t audio_rows = decl_.audio_channels * (int64_t)g0.n_audio;
    const int64_t video_rows = rows - g0.n_text - audio_rows;
    *hptr<int32_t>(R_LANE) = j.lane;
    *hptr<int32_t>(R_FIRST) = j.step == 0;
    int32_t* seg = hptr<int32_t>(R_SEG);
    seg[0] = 0;
    seg[1] = g0.n_text;
    seg[2] = (int32_t)(g0.n_text + audio_rows);
    seg[3] = (int32_t)rows;
    int32_t* lat = hptr<int32_t>(R_LAT);
    lat[0] = 0;
    lat[1] = (int32_t)video_rows;
    lat[2] = 0;
    lat[3] = (int32_t)audio_rows;
    reg_[R_LANE].used = 4;
    reg_[R_FIRST].used = 4;
    reg_[R_SEG].used = 4 * 4;
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
    /* 2. A prompt. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && !j.cond_done) return build_cond(j, step);
    }
    /* 3. The oldest lane's next step, or its upscale once its small steps are done. */
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
    if (upload() < 0) return nullptr;
    base_batches(j, g, T, step);
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
    const int64_t rows = small ? j.rows_small : j.rows;
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
    stage_geom(j, g, rows, 0);
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

int VideoDriver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t now = (int64_t)rad_mono_ns();
    Job* j = job_locked(pass_.job);
    switch (j ? pass_.kind : 0) {
    case RAD_GEN_PASS_VCOND:
        j->cond_done = true;
        j->req->ticket->t_cond_ns = now;
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
