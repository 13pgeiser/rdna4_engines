/* gen.cpp -- the request side of image generation: the ticket a caller waits on, the sampler's
 * schedule, and the generator's sequence built from a conditioning prompt (gen.h). */
#include "gen.h"

#include <chrono>
#include <cmath>
#include <cstring>

namespace rad {
namespace gen {

/* ================================================================== the ticket */
void Ticket::push(Image&& im) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(im));
    }
    cv_.notify_all();
}

void Ticket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

void Ticket::step(int32_t done, int32_t total) {
    (void)done;
    (void)total;
}

bool Ticket::next(Image* out, int ms, bool* timed_out) {
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

bool Ticket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_ && q_.empty();
}

int Ticket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string Ticket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

/* ================================================================== the schedule */
/* diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps with explicit sigmas, as the pipeline
 * calls it, in the precision it runs at: the sigmas are an f32 array and every shift is numpy f32
 * arithmetic, the Python floats in it (exp(mu), the shift, the terminal) taking the array's type. */
std::vector<float> schedule(const RadGeneratorDecl& d, const std::vector<double>& fixed,
                            int32_t steps, int64_t tokens) {
    std::vector<float> s;
    if (!fixed.empty()) {
        for (double v : fixed) s.push_back((float)v);
    } else {
        const int32_t n = steps > 0 ? steps : (int32_t)d.default_steps;
        /* np.linspace(1.0, 1 / n, n) in f64, then astype(f32). */
        const double lo = 1.0 / (double)n;
        for (int32_t i = 0; i < n; ++i) {
            const double v = n == 1 ? 1.0 : 1.0 + (lo - 1.0) * (double)i / (double)(n - 1);
            s.push_back((float)v);
        }
    }
    if (d.dyn_shift) {
        const double m = (d.max_shift - d.base_shift) / (double)(d.max_seq - d.base_seq);
        const double b = d.base_shift - m * (double)d.base_seq;
        const double mu = (double)tokens * m + b;
        const float  e = d.shift_exp ? (float)std::exp(mu) : (float)mu;
        for (float& t : s) t = e / (e + (1.0f / t - 1.0f));
    } else {
        const float sh = (float)d.shift;
        for (float& t : s) t = sh * t / (1.0f + (sh - 1.0f) * t);
    }
    if (d.shift_terminal > 0 && !s.empty()) {
        const float scale = (1.0f - s.back()) / (float)(1.0 - d.shift_terminal);
        for (float& t : s) t = 1.0f - ((1.0f - t) / scale);
    }
    s.push_back(0.0f);
    return s;
}

/* ================================================================== the sequence */
void image_rope(int32_t pos, int32_t h, int32_t w, int32_t* planes, int64_t n_total) {
    const int32_t h0 = -(h - h / 2), w0 = -(w - w / 2);
    int64_t k = 0;
    for (int32_t r = 0; r < h; ++r)
        for (int32_t c = 0; c < w; ++c, ++k) {
            planes[k] = pos;
            planes[n_total + k] = h0 + r;
            planes[2 * n_total + k] = w0 + c;
        }
}

int build_layout(const Branch& b, int64_t slot_rows,
                 const std::vector<std::pair<int32_t, int32_t>>& ref_hw, Layout* out,
                 std::string* why) {
    *out = Layout{};
    const int64_t T = (int64_t)b.tokens.size();
    if (b.n_drop < 0 || b.n_drop >= T) {
        if (why) *why = "the conditioning prompt is no longer than its fixed prefix";
        return RAD_E_INVAL;
    }
    /* The media runs, in prompt order: [tok, tok + n) placeholder rows of item `item`. */
    struct Run { int64_t tok, n; int32_t item; };
    std::vector<Run> runs;
    if (b.media)
        for (const mm::RopeLayout::Run& r : b.media->rope.runs)
            runs.push_back({ r.tok, r.n, r.item });
    if (runs.size() != ref_hw.size()) {
        if (why) *why = "the prompt places " + std::to_string(runs.size()) + " picture(s) and " +
                        std::to_string(ref_hw.size()) + " were encoded for it";
        return RAD_E_INVAL;
    }
    /* Each picture's rows and where its latent rows start among all the references'. */
    std::vector<int64_t> lat0(ref_hw.size() + 1, 0);
    for (size_t i = 0; i < ref_hw.size(); ++i)
        lat0[i + 1] = lat0[i] + (int64_t)ref_hw[i].first * ref_hw[i].second;
    int64_t rows = 0;
    {
        int64_t text = T - b.n_drop;
        for (const Run& r : runs) {
            if (r.tok < b.n_drop) {
                if (why) *why = "a picture sits inside the prompt's fixed prefix";
                return RAD_E_INVAL;
            }
            text -= r.n;
        }
        rows = text;
        for (size_t i = 0; i < runs.size(); ++i) {
            const int64_t lat = (int64_t)ref_hw[i].first * ref_hw[i].second;
            if (runs[i].n * slot_rows != lat) {
                if (why) *why = "picture " + std::to_string(i + 1) + " has " +
                                std::to_string(runs[i].n) + " encoder rows, which do not tile its " +
                                std::to_string(ref_hw[i].first) + " x " +
                                std::to_string(ref_hw[i].second) + " latent grid";
                return RAD_E_INVAL;
            }
            rows += lat;
        }
    }
    out->rows = rows;
    out->src.assign((size_t)rows, 0);
    out->rope.assign((size_t)rows * 3, 0);
    out->seg.push_back(0);
    int32_t pos = 0;
    int64_t row = 0;
    size_t ri = 0;
    int64_t t = b.n_drop;
    bool in_text = false;
    auto close_text = [&]() {
        if (in_text) {
            out->seg.push_back((int32_t)row);
            out->bidi.push_back(0);
            in_text = false;
        }
    };
    while (t < T) {
        if (ri < runs.size() && t == runs[ri].tok) {
            close_text();
            const int32_t h = ref_hw[ri].first, w = ref_hw[ri].second;
            const int64_t n = (int64_t)h * w;
            for (int64_t k = 0; k < n; ++k) out->src[(size_t)(row + k)] = -(int32_t)(1 + lat0[ri] + k);
            std::vector<int32_t> pl((size_t)n * 3);
            image_rope(pos, h, w, pl.data(), n);
            for (int c = 0; c < 3; ++c)
                std::memcpy(&out->rope[(size_t)(c * rows + row)], &pl[(size_t)(c * n)],
                            (size_t)n * sizeof(int32_t));
            pos += h > w ? h : w;
            row += n;
            out->seg.push_back((int32_t)row);
            out->bidi.push_back(1);
            t += runs[ri].n;
            ++ri;
            continue;
        }
        in_text = true;
        out->src[(size_t)row] = (int32_t)t;
        for (int c = 0; c < 3; ++c) out->rope[(size_t)(c * rows + row)] = pos;
        ++pos;
        ++row;
        ++t;
    }
    close_text();
    out->next_pos = pos;
    if (row != rows) {
        if (why) *why = "the generator's sequence did not come out at the length it was sized for";
        return RAD_E_STATE;
    }
    return RAD_OK;
}

}  /* namespace gen */
}  /* namespace rad */
