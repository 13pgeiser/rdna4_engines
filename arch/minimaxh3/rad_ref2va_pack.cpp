/* The Ref2VA packing, ported rule for rule from the reference implementation's PackedLayout
 * (docs/NOTE-P3-PACK.md carries the line-referenced extraction; the golden vectors in
 * tests/data/ref2va are generated from the reference source itself). Where the reference computes
 * in float64 so does this -- the coordinates feed a rotary, and an area-normalised axis drawn with
 * float32 steps would bend it. The stored positions are f32 because that is what R_ROPE holds. */

#include "rad_ref2va_pack.h"

#include <algorithm>
#include <cmath>

namespace minimaxh3 {

/* Pixel frames the k-th video latent of a run covers, cycling; the pixel-frames-per-audio-latent
 * rescale; and the near-clean timesteps the pinned rows sit at. */
static const int kFramePerToken[5] = { 1, 4, 4, 4, 4 };
static const double kFrameRescale = 5.0 / 3.0;
static const double kVisualPin = 0.999;
static const double kAudioPin = 1.0;

static const char* const kKindNames[kR2VKinds] = { "text", "cond", "cond_audio", "ref_img",
                                                   "ref_audio", "audio", "video" };

const char* ref2va_kind_name(int kind) {
    return kind >= 0 && kind < kR2VKinds ? kKindNames[kind] : "?";
}

int ref2va_seg_tag(int kind) {
    switch (kind) {
        case kR2VText: return 1;
        case kR2VAudio: case kR2VCondAudio: case kR2VRefAudio: return 2;
        default: return 0;                               /* video, cond, ref_img */
    }
}

int ref2va_is_img(int kind) {
    return kind == kR2VCond || kind == kR2VRefImg || kind == kR2VVideo;
}

int ref2va_is_audio(int kind) {
    return kind == kR2VCondAudio || kind == kR2VRefAudio || kind == kR2VAudio;
}

double ref2va_t_label(int kind, double t_v, double t_a) {
    switch (kind) {
        case kR2VCond: case kR2VRefImg: return std::max(t_v, kVisualPin);
        case kR2VCondAudio: case kR2VRefAudio: return std::max(t_a, kAudioPin);
        case kR2VAudio: return t_a;
        default: return t_v;                             /* text, video */
    }
}

/* The spatial axis of a latent side `dim` at 2x2 patches: n coords spreading over
 * [(1-ratio)/2, (1+ratio)/2] * 32, ratio = dim / sqrt(h*w) -- so a square latent maps both axes
 * onto the same [~8, ~24] band and a wide one trades bands against its neighbour, keeping the
 * rotary's combined reach the same for every aspect. */
static std::vector<double> axis_from_sqrt_area(int dim, int patch, double sqrt_area) {
    const double ratio = (double)dim / sqrt_area;
    const int n = dim / patch;
    std::vector<double> v((size_t)n);
    for (int i = 0; i < n; ++i)
        v[(size_t)i] = ((double)i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
    return v;
}

/* One latent frame's patch rows: h-major (row, col) pairs over the 2x2 grid, plus the column axis
 * on its own -- its first and last coordinate pin the stereo channels onto the grid's width. */
static void frame_grid(int h, int w, std::vector<double>* frame, std::vector<double>* w_grid) {
    const double area = std::sqrt((double)h * (double)w);
    std::vector<double> hg = axis_from_sqrt_area(h, 2, area), wg = axis_from_sqrt_area(w, 2, area);
    frame->clear();
    frame->reserve(hg.size() * wg.size() * 2);
    for (const double& r : hg)
        for (const double& c : wg) { frame->push_back(r); frame->push_back(c); }
    *w_grid = std::move(wg);
}

/* The time a run of n video latents occupies on the audio-latent scale. */
static double video_t_span_sum(int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += kFrameRescale * (double)kFramePerToken[k % 5];
    return s;
}

/* Each video latent's time coordinate: origin, then origin + each shorter latent prefix's spans
 * (exclusive cumsum -- the k-th latent sits where its predecessors end). */
static std::vector<double> video_t_grid(int n, double origin) {
    std::vector<double> g((size_t)n);
    double t = origin;
    for (int k = 0; k < n; ++k) {
        g[(size_t)k] = t;
        t += kFrameRescale * (double)kFramePerToken[k % 5];
    }
    return g;
}

/* The time a reference block occupies ahead of the target timeline: a photo one frame, a standalone
 * soundtrack its own length, a clip whichever of its two streams reaches further. */
static double ref_t_span(const Ref2VARef& r) {
    if (r.kind == kR2VRefImage) return 1.0;
    if (r.kind == kR2VRefAudioOnly) return (double)r.audio_t;
    return std::max((double)r.audio_t, video_t_span_sum(r.latent_t));
}

/* Append a block's stereo audio rows: channel-major -- all T frames of the left channel, then the
 * right's -- time advancing one audio latent a row, width pinned to the grid's first coordinate
 * for one channel and its last for the other, height always 0. */
static void audio_grid(double cursor, int t, double w_low, double w_high, std::vector<float>* pos) {
    for (int ch = 0; ch < 2; ++ch) {
        const double w = ch == 0 ? w_low : w_high;
        for (int i = 0; i < t; ++i) {
            pos->push_back((float)(cursor + (double)i));
            pos->push_back(0.0f);
            pos->push_back((float)w);
        }
    }
}

/* Append a block's video rows: every latent frame's patch rows at that frame's time coordinate. */
static void video_grid(int vt, const std::vector<double>& frame, double origin,
                       std::vector<float>* pos) {
    const std::vector<double> tg = video_t_grid(vt, origin);
    for (int k = 0; k < vt; ++k)
        for (size_t j = 0; j < frame.size(); j += 2) {
            pos->push_back((float)tg[(size_t)k]);
            pos->push_back((float)frame[j]);
            pos->push_back((float)frame[j + 1]);
        }
}

int ref2va_build(int text_len, const Ref2VAGeom& g, const std::vector<Ref2VAKeyframe>& keyframes,
                 const std::vector<Ref2VARef>& refs, Ref2VAPack* out) {
    if (text_len < 0 || g.latent_t < 0 || g.latent_h < 0 || g.latent_w < 0 || g.audio_t < 0)
        return -1;
    for (const Ref2VAKeyframe& kf : keyframes)
        if (kf.frame < 0 || kf.video_t < 0 || kf.audio_t < 0) return -1;
    for (const Ref2VARef& r : refs) {
        if (r.kind < kR2VRefImage || r.kind > kR2VRefVideoAudio) return -1;
        if (r.lat_h < 0 || r.lat_w < 0 || r.latent_t < 0 || r.audio_t < 0) return -1;
    }

    Ref2VAPack p;
    std::vector<double> frame, w_grid;
    frame_grid(g.latent_h, g.latent_w, &frame, &w_grid);
    const size_t frame_rows = frame.size() / 2;
    const double w_low = w_grid.empty() ? 0.0 : w_grid.front();
    const double w_high = w_grid.empty() ? 0.0 : w_grid.back();

    /* The prompt first: time = position in the prompt, space unused. */
    p.segs.push_back({ kR2VText, 0, (int64_t)text_len, 0 });
    p.pos.reserve((size_t)text_len * 3);
    for (int i = 0; i < text_len; ++i) {
        p.pos.push_back((float)i);
        p.pos.push_back(0.0f);
        p.pos.push_back(0.0f);
    }
    int64_t row = text_len;

    /* References pack between the prompt and the targets, so the target timeline -- and every
     * keyframe anchored to it -- starts after all of their spans. */
    double cursor = (double)text_len;
    for (const Ref2VARef& r : refs) cursor += ref_t_span(r);

    /* Keyframes: a keyframe's latent rows share the TARGET grid (they were resized before the
     * VAE), anchored at its pixel frame's moment on the target timeline. */
    for (const Ref2VAKeyframe& kf : keyframes) {
        const double cond_t = cursor + kFrameRescale * (double)kf.frame;
        if (kf.video_t > 0) {
            const int64_t n = (int64_t)kf.video_t * (int64_t)frame_rows;
            video_grid(kf.video_t, frame, cond_t, &p.pos);
            p.segs.push_back({ kR2VCond, row, row + n, 0 });
            row += n;
        }
        if (kf.audio_t > 0) {
            const int64_t n = (int64_t)kf.audio_t * 2;
            audio_grid(cond_t, kf.audio_t, w_low, w_high, &p.pos);
            p.segs.push_back({ kR2VCondAudio, row, row + n, 0 });
            row += n;
        }
    }

    /* Reference blocks, in request order, each at a running cursor of its own. */
    cursor = (double)text_len;
    for (const Ref2VARef& r : refs) {
        if (r.kind == kR2VRefImage) {
            std::vector<double> rf, rw;
            frame_grid(r.lat_h, r.lat_w, &rf, &rw);
            const int64_t n = (int64_t)rf.size() / 2;
            for (int64_t i = 0; i < n; ++i) {
                p.pos.push_back((float)cursor);
                p.pos.push_back((float)rf[(size_t)i * 2]);
                p.pos.push_back((float)rf[(size_t)i * 2 + 1]);
            }
            p.segs.push_back({ kR2VRefImg, row, row + n, 0 });
            row += n;
            cursor += 1.0;
        } else if (r.kind == kR2VRefAudioOnly) {
            if (r.audio_t > 0) {
                const int64_t n = (int64_t)r.audio_t * 2;
                audio_grid(cursor, r.audio_t, w_low, w_high, &p.pos);
                p.segs.push_back({ kR2VRefAudio, row, row + n, 0 });
                row += n;
            }
            cursor += (double)r.audio_t;
        } else {
            /* A clip's soundtrack packs immediately before its video, both at the same time
             * origin, and its audio channels pin onto the CLIP's own width extremes -- it is the
             * clip's space, not the target's. */
            std::vector<double> rf, rw;
            frame_grid(r.lat_h, r.lat_w, &rf, &rw);
            if (r.audio_t > 0) {
                const int64_t n = (int64_t)r.audio_t * 2;
                audio_grid(cursor, r.audio_t, rw.empty() ? 0.0 : rw.front(),
                           rw.empty() ? 0.0 : rw.back(), &p.pos);
                p.segs.push_back({ kR2VRefAudio, row, row + n, 0 });
                row += n;
            }
            const int64_t n = (int64_t)r.latent_t * (int64_t)(rf.size() / 2);
            video_grid(r.latent_t, rf, cursor, &p.pos);
            p.segs.push_back({ kR2VRefImg, row, row + n, 0 });
            row += n;
            cursor += std::max((double)r.audio_t, video_t_span_sum(r.latent_t));
        }
    }

    /* The target streams, always last and always the sampler's whole state: the soundtrack, then
     * the video, both at the timeline the references left room for. */
    {
        const int64_t n = (int64_t)g.audio_t * 2;
        audio_grid(cursor, g.audio_t, w_low, w_high, &p.pos);
        p.segs.push_back({ kR2VAudio, row, row + n, 1 });
        row += n;
    }
    {
        const int64_t n = (int64_t)g.latent_t * (int64_t)frame_rows;
        video_grid(g.latent_t, frame, cursor, &p.pos);
        p.segs.push_back({ kR2VVideo, row, row + n, 1 });
        row += n;
    }

    p.rows = row;
    {
        /* The blocks were appended as (t, h, w) triples; R_ROPE holds the axes apart, so lay the
         * buffer planar now and staging is a copy. */
        std::vector<float> tri = std::move(p.pos);
        p.pos.assign((size_t)row * 3, 0.0f);
        for (int64_t r = 0; r < row; ++r) {
            p.pos[(size_t)r] = tri[(size_t)r * 3];
            p.pos[(size_t)row + r] = tri[(size_t)r * 3 + 1];
            p.pos[2 * (size_t)row + r] = tri[(size_t)r * 3 + 2];
        }
    }
    *out = std::move(p);
    return 0;
}

int ref2va_mod_rows(const Ref2VAPack& p, double t_v, double t_a, std::vector<int>* out) {
    std::vector<double> uniq;
    for (const Ref2VASeg& s : p.segs)
        if (s.end > s.start) uniq.push_back(ref2va_t_label(s.kind, t_v, t_a));
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    out->resize(p.segs.size());
    for (size_t i = 0; i < p.segs.size(); ++i) {
        const Ref2VASeg& s = p.segs[i];
        if (s.end <= s.start) { (*out)[i] = 0; continue; }
        const double lbl = ref2va_t_label(s.kind, t_v, t_a);
        const size_t t_row = (size_t)(std::lower_bound(uniq.begin(), uniq.end(), lbl) - uniq.begin());
        (*out)[i] = (int)(t_row * 3 + (size_t)ref2va_seg_tag(s.kind));
    }
    return 0;
}

void ref2va_ada_cell(const Ref2VAPack& p, double t_v, double t_a, std::vector<int>* md,
                     std::vector<int>* trow) {
    md->resize(p.segs.size());
    trow->resize(p.segs.size());
    for (size_t i = 0; i < p.segs.size(); ++i) {
        const Ref2VASeg& s = p.segs[i];
        if (s.end <= s.start) { (*md)[i] = 0; (*trow)[i] = kR2VTVideo; continue; }
        const int tag = ref2va_seg_tag(s.kind);
        (*md)[i] = tag;
        /* The modality fixes which stream's schedule the row reads; the label decides whether it
         * reads that schedule's own timestep or, once the sampler has decayed below the kind's
         * near-clean floor (ref2va_t_label is std::max, so the label is bit-identical to one of
         * the two), the pinned row. Deciding by modality rather than by the bare label is what
         * keeps video and audio apart when a probe happens to set t_v == t_a. */
        const double lbl = ref2va_t_label(s.kind, t_v, t_a);
        if (tag == 2) (*trow)[i] = (lbl == t_a) ? kR2VTAudio : kR2VTPinAudio;
        else          (*trow)[i] = (lbl == t_v) ? kR2VTVideo : kR2VTPinVideo;
    }
}

}   /* namespace minimaxh3 */
