/* rad_ref2va_pack.h -- MiniMax-H3's packed sequence with keyframes and reference media, on the
 * host: ComfyUI's PackedLayout (comfy/ldm/minimax/model.py), tested against that code's own numbers
 * (tests/data/ref2va).
 *
 * The sequence is [text | keyframes | references in request order | soundtrack | video]. The
 * video's clock starts after every reference's time span; a keyframe sits on it at 5/3 a pixel
 * frame; a reference clip's soundtrack rows come right before its video rows, at the same time
 * origin. Keyframes and references keep a near-clean timestep; the sampler steps the last two
 * segments only.
 */

#ifndef RAD_ARCH_MINIMAXH3_REF2VA_PACK_H
#define RAD_ARCH_MINIMAXH3_REF2VA_PACK_H

#include <cstdint>
#include <vector>

namespace minimaxh3 {

/* Segment kinds, named as the reference implementation names them. */
enum Ref2VAKind {
    kR2VText = 0,       /* the prompt's hidden states */
    kR2VCond,           /* a keyframe's video latent rows */
    kR2VCondAudio,      /* a keyframe's soundtrack latent rows */
    kR2VRefImg,         /* a reference photo, or a reference clip's video */
    kR2VRefAudio,       /* a reference clip's (or standalone audio's) soundtrack */
    kR2VAudio,          /* the target soundtrack -- sampler-owned */
    kR2VVideo,          /* the target video -- sampler-owned */
    kR2VKinds
};

/* A reference's kind; a video with a soundtrack of its own is either of the last two. */
enum Ref2VARefKind { kR2VRefImage = 0, kR2VRefAudioOnly = 1, kR2VRefVideo = 2, kR2VRefVideoAudio = 3 };

/* A keyframe: the frame of the video it is pinned at and its latent frames of each stream. */
struct Ref2VAKeyframe {
    int frame;      /* >= 0 */
    int video_t;    /* 0: none */
    int audio_t;    /* 0: none */
};

/* A reference, in latents: its own grid and frames for a picture or video, its soundtrack's
 * length for one that has sound. */
struct Ref2VARef {
    int kind;       /* Ref2VARefKind */
    int lat_h, lat_w;
    int latent_t;
    int audio_t;
};

/* The target, in latent units. */
struct Ref2VAGeom {
    int latent_t, latent_h, latent_w, audio_t;
};

struct Ref2VASeg {
    int kind;
    int64_t start, end;   /* rows [start, end) */
    int update;           /* 1 for the soundtrack and the video the sampler steps */
};

struct Ref2VAPack {
    std::vector<Ref2VASeg> segs;
    int64_t rows = 0;
    /* The rotary positions, [3, rows]: every row's time, then row, then column. */
    std::vector<float> pos;
};

/* A kind's modality in the modulation (ComfyUI's seg_tag): 0 video, 1 text, 2 sound. */
int ref2va_seg_tag(int kind);
/* Whether a kind's rows are video latents or soundtrack latents; text is neither. */
int ref2va_is_img(int kind);
int ref2va_is_audio(int kind);
/* A keyframe's or reference's timestep is the larger of its stream's and these. */
constexpr double kR2VVisualPin = 0.999, kR2VAudioPin = 1.0;

/* The modulation groups: runs of consecutive rows that read the same modulation cell, which the
 * segment's kind fixes (`md` its modality's projection, `r0` its block of timestep embeddings) --
 * except inside the prompt, where a picture's vision pads read the video projection on the text's
 * block, as ComfyUI tags them (`runs`). The modulated norm takes the group boundaries, the attention
 * the segments'. Groups that share a cell read one column of the modulation table (`col`), so the
 * table grows with the distinct cells, not with the references or the pads: nine pictures add one
 * cell. The groups cover the sequence; an empty segment opens none. */
struct Ref2VATagRun {
    int64_t start, end;   /* prompt rows [start, end) */
    int tag;              /* the projection they read: 0 video for vision pads */
};
struct Ref2VAModGroups {
    std::vector<int64_t> bound;   /* [groups + 1], from 0 to the rows */
    std::vector<int> col;         /* [groups]: the cell the group reads */
    std::vector<int> md;          /* [cells]: 0 video, 1 text, 2 sound */
    std::vector<int64_t> r0;      /* [cells] */
};
/* `S` is the steps a schedule has, and the embeddings are [0, S) the video's, [S, 2S) the
 * soundtrack's, then each again with its pin, [2S, 3S) and [3S, 4S). The boundaries do not depend
 * on S. `runs` may come in any order; the parts outside the prompt are ignored. Without runs every
 * group has a cell of its own. */
void ref2va_mod_groups(const Ref2VAPack& p, int64_t S, const std::vector<Ref2VATagRun>& runs,
                       Ref2VAModGroups* out);

/* The sequence, and its positions when `positions`; -1 for a negative size or an unknown kind. */
int ref2va_build(int text_len, const Ref2VAGeom& g, const std::vector<Ref2VAKeyframe>& keyframes,
                 const std::vector<Ref2VARef>& refs, Ref2VAPack* out, bool positions = true);

}   /* namespace minimaxh3 */

#endif
