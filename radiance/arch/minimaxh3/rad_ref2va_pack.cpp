/* rad_ref2va_pack.cpp -- see rad_ref2va_pack.h. The positions are computed in f64 as ComfyUI
 * computes them and stored in f32. */

#include "rad_ref2va_pack.h"

#include <algorithm>
#include <cmath>

namespace minimaxh3 {

/* The pixel frames the k-th latent frame of a video covers, cycling, and the time a pixel frame
 * spans in soundtrack latents. */
static const int kFramePerToken[5] = { 1, 4, 4, 4, 4 };
static const double kFrameRescale = 5.0 / 3.0;

int ref2va_seg_tag(int kind) {
    switch (kind) {
        case kR2VText: return 1;
        case kR2VAudio: case kR2VCondAudio: case kR2VRefAudio: return 2;
        default: return 0;
    }
}

int ref2va_is_img(int kind) {
    return kind == kR2VCond || kind == kR2VRefImg || kind == kR2VVideo;
}

int ref2va_is_audio(int kind) {
    return kind == kR2VCondAudio || kind == kR2VRefAudio || kind == kR2VAudio;
}

/* A side of `dim` latents in 2x2 patches: dim / 2 positions over
 * [(1 - ratio) / 2, (1 + ratio) / 2) x 32, ratio = dim / sqrt(h w), so every aspect spans the same
 * area. */
static std::vector<double> axis_from_sqrt_area(int dim, int patch, double sqrt_area) {
    const double ratio = (double)dim / sqrt_area;
    const int n = dim / patch;
    std::vector<double> v((size_t)n);
    for (int i = 0; i < n; ++i)
        v[(size_t)i] = ((double)i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
    return v;
}

/* A latent frame's patches as (row, column) pairs, row-major, and the column axis, whose ends the
 * two soundtrack channels sit at. */
static void frame_grid(int h, int w, std::vector<double>* frame, std::vector<double>* w_grid) {
    const double area = std::sqrt((double)h * (double)w);
    std::vector<double> hg = axis_from_sqrt_area(h, 2, area), wg = axis_from_sqrt_area(w, 2, area);
    frame->clear();
    frame->reserve(hg.size() * wg.size() * 2);
    for (const double& r : hg)
        for (const double& c : wg) { frame->push_back(r); frame->push_back(c); }
    *w_grid = std::move(wg);
}

/* The time n latent frames of video span. */
static double video_t_span_sum(int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += kFrameRescale * (double)kFramePerToken[k % 5];
    return s;
}

/* Each latent frame's time: where the frames before it end. */
static std::vector<double> video_t_grid(int n, double origin) {
    std::vector<double> g((size_t)n);
    double t = origin;
    for (int k = 0; k < n; ++k) {
        g[(size_t)k] = t;
        t += kFrameRescale * (double)kFramePerToken[k % 5];
    }
    return g;
}

/* The time a reference spans: a picture 1, a soundtrack its length, a video the longer of its two
 * streams. */
static double ref_t_span(const Ref2VARef& r) {
    if (r.kind == kR2VRefImage) return 1.0;
    if (r.kind == kR2VRefAudioOnly) return (double)r.audio_t;
    return std::max((double)r.audio_t, video_t_span_sum(r.latent_t));
}

/* A soundtrack's rows, the first channel's then the second's, a latent each at `cursor` onwards;
 * the channels sit at the two ends of the width axis, row 0. */
static void audio_grid(double cursor, int t, double w_low, double w_high, std::vector<float>* pos) {
    if (!pos) return;
    for (int ch = 0; ch < 2; ++ch) {
        const double w = ch == 0 ? w_low : w_high;
        for (int i = 0; i < t; ++i) {
            pos->push_back((float)(cursor + (double)i));
            pos->push_back(0.0f);
            pos->push_back((float)w);
        }
    }
}

/* A video's rows: each latent frame's patches at its time. */
static void video_grid(int vt, const std::vector<double>& frame, double origin,
                       std::vector<float>* pos) {
    if (!pos) return;
    const std::vector<double> tg = video_t_grid(vt, origin);
    for (int k = 0; k < vt; ++k)
        for (size_t j = 0; j < frame.size(); j += 2) {
            pos->push_back((float)tg[(size_t)k]);
            pos->push_back((float)frame[j]);
            pos->push_back((float)frame[j + 1]);
        }
}

int ref2va_build(int text_len, const Ref2VAGeom& g, const std::vector<Ref2VAKeyframe>& keyframes,
                 const std::vector<Ref2VARef>& refs, Ref2VAPack* out, bool positions) {
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

    /* The prompt: its token positions on the time axis. */
    p.segs.push_back({ kR2VText, 0, (int64_t)text_len, 0 });
    std::vector<float>* pos = positions ? &p.pos : nullptr;
    for (int i = 0; i < text_len && pos; ++i) {
        p.pos.push_back((float)i);
        p.pos.push_back(0.0f);
        p.pos.push_back(0.0f);
    }
    int64_t row = text_len;

    /* The video's clock, and the keyframes on it, starts after every reference's span. */
    double cursor = (double)text_len;
    for (const Ref2VARef& r : refs) cursor += ref_t_span(r);

    /* A keyframe is on the video's grid, at its frame's time. */
    for (const Ref2VAKeyframe& kf : keyframes) {
        const double cond_t = cursor + kFrameRescale * (double)kf.frame;
        if (kf.video_t > 0) {
            const int64_t n = (int64_t)kf.video_t * (int64_t)frame_rows;
            video_grid(kf.video_t, frame, cond_t, pos);
            p.segs.push_back({ kR2VCond, row, row + n, 0 });
            row += n;
        }
        if (kf.audio_t > 0) {
            const int64_t n = (int64_t)kf.audio_t * 2;
            audio_grid(cond_t, kf.audio_t, w_low, w_high, pos);
            p.segs.push_back({ kR2VCondAudio, row, row + n, 0 });
            row += n;
        }
    }

    /* The references in request order, each after the last. */
    cursor = (double)text_len;
    for (const Ref2VARef& r : refs) {
        if (r.kind == kR2VRefImage) {
            std::vector<double> rf, rw;
            frame_grid(r.lat_h, r.lat_w, &rf, &rw);
            const int64_t n = (int64_t)rf.size() / 2;
            for (int64_t i = 0; i < n && pos; ++i) {
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
                audio_grid(cursor, r.audio_t, w_low, w_high, pos);
                p.segs.push_back({ kR2VRefAudio, row, row + n, 0 });
                row += n;
            }
            cursor += (double)r.audio_t;
        } else {
            /* A video's soundtrack, at the video's time and the ends of the video's own width
             * axis, comes right before it. */
            std::vector<double> rf, rw;
            frame_grid(r.lat_h, r.lat_w, &rf, &rw);
            if (r.audio_t > 0) {
                const int64_t n = (int64_t)r.audio_t * 2;
                audio_grid(cursor, r.audio_t, rw.empty() ? 0.0 : rw.front(),
                           rw.empty() ? 0.0 : rw.back(), pos);
                p.segs.push_back({ kR2VRefAudio, row, row + n, 0 });
                row += n;
            }
            const int64_t n = (int64_t)r.latent_t * (int64_t)(rf.size() / 2);
            video_grid(r.latent_t, rf, cursor, pos);
            p.segs.push_back({ kR2VRefImg, row, row + n, 0 });
            row += n;
            cursor += std::max((double)r.audio_t, video_t_span_sum(r.latent_t));
        }
    }

    /* The soundtrack and the video the sampler steps, last. */
    {
        const int64_t n = (int64_t)g.audio_t * 2;
        audio_grid(cursor, g.audio_t, w_low, w_high, pos);
        p.segs.push_back({ kR2VAudio, row, row + n, 1 });
        row += n;
    }
    {
        const int64_t n = (int64_t)g.latent_t * (int64_t)frame_rows;
        video_grid(g.latent_t, frame, cursor, pos);
        p.segs.push_back({ kR2VVideo, row, row + n, 1 });
        row += n;
    }

    p.rows = row;
    if (positions) {
        /* From (t, h, w) triples to [3, rows]. */
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

/* A kind's modulation cell (rad_ref2va_pack.h): the latents and the prompt read their schedule's
 * embeddings, a keyframe or reference its stream's pinned ones. */
static void cell_of_kind(int kind, int64_t S, int* md, int64_t* r0) {
    switch (kind) {
        case kR2VAudio:     *md = 2; *r0 = S;     break;
        case kR2VCondAudio:
        case kR2VRefAudio:  *md = 2; *r0 = 3 * S; break;
        case kR2VText:      *md = 1; *r0 = 0;     break;
        case kR2VCond:
        case kR2VRefImg:    *md = 0; *r0 = 2 * S; break;
        default:            *md = 0; *r0 = 0;     break;
    }
}

void ref2va_mod_groups(const Ref2VAPack& p, int64_t S, const std::vector<Ref2VATagRun>& runs,
                       Ref2VAModGroups* out) {
    out->bound.clear(); out->col.clear(); out->md.clear(); out->r0.clear();
    std::vector<Ref2VATagRun> sorted;
    for (const Ref2VATagRun& r : runs)
        if (r.end > r.start) sorted.push_back(r);
    std::sort(sorted.begin(), sorted.end(),
              [](const Ref2VATagRun& a, const Ref2VATagRun& b) { return a.start < b.start; });

    /* The rows in pieces of one cell each: a segment is one, except the prompt, which the runs cut
     * into text and pad pieces, every one on the text's block. */
    struct Piece { int64_t start, end; int md; int64_t r0; };
    std::vector<Piece> pieces;
    for (const Ref2VASeg& s : p.segs) {
        if (s.end <= s.start) continue;
        if (s.kind != kR2VText) {
            Piece pc{ s.start, s.end, 0, 0 };
            cell_of_kind(s.kind, S, &pc.md, &pc.r0);
            pieces.push_back(pc);
            continue;
        }
        int64_t at = s.start;
        for (const Ref2VATagRun& r : sorted) {
            const int64_t a = std::max(r.start, at), b = std::min(r.end, s.end);
            if (a >= s.end) break;
            if (b <= a) continue;
            if (a > at) pieces.push_back({ at, a, 1, 0 });
            pieces.push_back({ a, b, r.tag, 0 });
            at = b;
        }
        if (at < s.end) pieces.push_back({ at, s.end, 1, 0 });
    }

    /* A group opens where the cell changes; cells are numbered in their first group's order. */
    auto column_of = [&](int md, int64_t r0) {
        for (size_t c = 0; c < out->md.size(); ++c)
            if (out->md[c] == md && out->r0[c] == r0) return (int)c;
        out->md.push_back(md);
        out->r0.push_back(r0);
        return (int)out->md.size() - 1;
    };
    int cur = -1;
    bool open = false;
    int64_t prev_end = 0;
    for (const Piece& pc : pieces) {
        const int col = column_of(pc.md, pc.r0);
        if (!open || col != cur || pc.start != prev_end) {
            out->bound.push_back(open ? pc.start : 0);
            cur = col;
            open = true;
            out->col.push_back(col);
        }
        prev_end = pc.end;
    }
    if (open) out->bound.push_back(p.rows);
}

}   /* namespace minimaxh3 */
