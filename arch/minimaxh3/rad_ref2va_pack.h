/* minimaxh3 -- the Ref2VA packed sequence, on the host.

 * A reference-to-video-or-audio request packs more than the plain [text | soundtrack | video]:
 * every reference photo, reference clip and soundtrack -- and a continuation's keyframes -- rides
 * the DiT sequence as rows of its own, ahead of the streams being denoised. This TU decides WHERE
 * those rows sit, what their (time, row, column) rotary coordinates are, which of them the sampler
 * owns and which stay pinned, and therefore which modulation row each reads. It is pure host
 * arithmetic -- no device, no core -- so it is unit-testable against the reference implementation's
 * own numbers (tests/data/ref2va, docs/NOTE-P3-PACK.md).

 * The rules replicated here, in one breath: the sequence is
 *   [text | keyframe rows | reference blocks in request order | audio | video],
 * the target timeline starts after every reference's time span (a first pass sums them), keyframes
 * anchor onto that timeline at FRAME_RESCALE per pixel frame, a reference clip's audio rows pack
 * immediately before its video rows sharing one time origin, pinned rows sit at a near-clean
 * timestep with their modality's tag, and the two target segments -- always last, always the
 * sampler's whole state -- ride the live timesteps.
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

/* The four reference part kinds the API admits. */
enum Ref2VARefKind { kR2VRefImage = 0, kR2VRefAudioOnly = 1, kR2VRefVideo = 2, kR2VRefVideoAudio = 3 };

/* One keyframe: its latent frame index in the target (frame 0 anchors at the target timeline's
 * origin) and the latent rows it carries, 0 where that stream is absent. */
struct Ref2VAKeyframe {
    int frame;      /* resolved frame index, >= 0 */
    int video_t;    /* latent frames of its video latent; <= 0 = none */
    int audio_t;    /* audio latent frames of its soundtrack latent; <= 0 = none */
};

/* One reference block, in latent units: h/w for the kinds that carry video (a photo's or clip's
 * OWN latent sides -- the grid is the reference's, not the target's), audio_t for the kinds that
 * carry sound. */
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
    int64_t start, end;   /* rows [start, end) of the packed sequence */
    int update;           /* 1: the sampler owns these rows (the two target streams) */
};

struct Ref2VAPack {
    std::vector<Ref2VASeg> segs;
    int64_t rows = 0;
    /* The rotary coordinates, PLANAR -- all times, then all rows, then all columns -- exactly the
     * [3, rows] shape R_ROPE holds, so staging is a copy. */
    std::vector<float> pos;
};

const char* ref2va_kind_name(int kind);
/* The modulation table's modality column a kind rides (the reference implementation's seg_tag):
 * video-stream rows 0, text 1, every audio-stream row 2. */
int ref2va_seg_tag(int kind);
/* Which latent space a kind's rows live in: img (cond/ref_img/video), audio (cond_audio/
 * ref_audio/audio), or neither (text). The sampler scatters the latents by these. */
int ref2va_is_img(int kind);
int ref2va_is_audio(int kind);
/* The timestep label a kind carries at video/audio timesteps t_v/t_a: the streams their own, the
 * pinned rows clamped to their near-clean floor. */
double ref2va_t_label(int kind, double t_v, double t_a);
/* Per-segment modulation-table row, reference-implementation style: t_row * 3 + tag, where t_row
 * indexes the sorted distinct labels actually present among the pack's non-empty segments.
 * Empty segments (only possible for a degenerate target) get 0. */
int ref2va_mod_rows(const Ref2VAPack& p, double t_v, double t_a, std::vector<int>* out);

/* Which timestep-embedding row a segment's modulation reads, once the reference pins are added to
 * the streams' schedules. The embedding is laid [0,S) = the video schedule, [S,2S) = the audio
 * schedule, [2S] = the visual pin, [2S+1] = the audio pin -- the two pinned rows the reference
 * conditioning needs that the plain path leaves off. A segment reads the schedule row when its
 * label IS the schedule's current timestep, and a pinned row only once the pin (its near-clean
 * floor, ref2va_t_label) sits above the schedule's decaying timestep. */
enum Ref2VATRow { kR2VTVideo = 0, kR2VTAudio = 1, kR2VTPinVideo = 2, kR2VTPinAudio = 3 };
/* Per-segment modulation cell, as step_ada fills it for the current step's timesteps t_v/t_a:
 * `md` is the modality's ada projection (== ref2va_seg_tag) and `trow` is the Ref2VATRow whose
 * embedding feeds the cell. So the cell (md,trow) is exactly (kSegMod[e], {0 or S}) on a plain
 * pack, and picks a pinned row for a reference row once its schedule falls under the pin. Empty
 * segments (end <= start) get (0, kR2VTVideo) -- their cell is never read. */
void ref2va_ada_cell(const Ref2VAPack& p, double t_v, double t_a, std::vector<int>* md,
                     std::vector<int>* trow);

/* Lay the sequence out. Returns 0, or -1 on a negative dimension (the driver has already bounded
 * media against the arch's ceilings; this only refuses nonsense). */
int ref2va_build(int text_len, const Ref2VAGeom& g, const std::vector<Ref2VAKeyframe>& keyframes,
                 const std::vector<Ref2VARef>& refs, Ref2VAPack* out);

}   /* namespace minimaxh3 */

#endif
