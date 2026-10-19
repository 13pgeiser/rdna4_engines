/* Encode geometry for the MiniMax-H3 VAEs: how many latent frames an encode produces and where
 * the spatial tiles sit. Pure host arithmetic -- no device, no weights. The kernels that actually
 * run the encoders live in the P2 device work; this TU answers the sizing questions they and the
 * driver ask ("how many latents will T frames give me", "which pixel window is tile k"), so the
 * numbers can be pinned against the reference implementation without any GPU. docs/NOTE-P2-ENCODE.md
 * carries the line-referenced extraction; tests/data/vae_geom are golden vectors generated from the
 * reference's own code. */

#ifndef RADIANCE_ARCH_MINIMAXH3_RAD_VAE_GEOM_H
#define RADIANCE_ARCH_MINIMAXH3_RAD_VAE_GEOM_H

#include <cstdint>
#include <vector>

namespace minimaxh3 {

/* Video VAE. Spatial downsamples prod(2,2,2,2,1,1) = 16, temporally the encode chain goes
 * T -> ceil(T/2) -> ceil(T/2): 17 pixel frames become 5 latents (17 -> 9 -> 5). */
constexpr int kVaeRatio = 16;        /* pixels per latent side */
constexpr int kVaeRatioT = 8;        /* nominal pixel frames per latent (17-frame clips are off-by-one: first latent holds 1 frame) */
constexpr int kVaeClipLength = 17;   /* pixel frames per temporal clip */
constexpr int kVaeClipLatents = 5;   /* latents one full clip encodes to */
constexpr int kVaeTokenDrop = 3;     /* trailing latents every multi-clip encode drops */
constexpr int kVaeTileSize = 256;    /* reference "tile size", pixels, always on */
constexpr int kVaeTileOverlapMin = 64;

/* Audio VAE: encoder strides (2,4,4,5,5) -> one latent per 800 samples at 32 kHz (40/s), and its
 * wrapper right-pads the waveform with zeros to a multiple of that before encoding. */
constexpr int kAudioHop = 800;
constexpr int kAudioLatentChannels = 32;

/* Latent frames an encode of `frames` pixel frames produces. A single image encodes alone and
 * yields its one latent; two or more frames chunk into ceil(T/17) clips (the short last clip is
 * padded by repeating its final frame), each yielding 5 latents, and the encode drops the last 3.
 * Returns -1 for a non-positive frame count. */
int64_t vae_latent_frames(int64_t frames);

/* Latents one clip of `frames` pixel frames encodes to before any concatenation/drop: the causal
 * chain is T -> ceil(T/2) -> ceil(T/2) (a single frame truncates its temporal taps and stays
 * one). A full clip is 17 -> 9 -> 5, so this times the clip count minus the drop is exactly
 * vae_latent_frames for T >= 2. */
int64_t vae_clip_latents(int64_t frames);

/* Audio latent frames an encode of `samples` samples produces (right zero-pad to the hop, then
 * one latent per hop). Returns -1 for a negative sample count. */
int64_t vae_audio_latents(int64_t samples);

/* One axis of the spatial tiling. `start`/`len` are the pixel window the tile encoder receives
 * (`len` is the real slice length -- the reference's Python slice clamps the nominal 256 at the
 * tail); `overlap` is the pixel overlap this tile shares with the NEXT tile, which the blend cuts
 * out of this tile's latents (`overlap / 16` latent columns/rows). */
struct VaeTile {
    int64_t start;
    int64_t len;
    int64_t overlap;
};

/* Port of the reference's split_tiles(): if one 256 window covers the axis, it does; otherwise
 * the smallest window count whose overlaps (64 px minimum, extra 16 px units spread round-robin
 * so the windows reach exactly the end) fit, starts cumulating 256 - overlap. The result covers
 * [0, length) exactly whenever length is a multiple of 16 (the encoder's own alignment). */
void vae_split_tiles(int64_t length, std::vector<VaeTile>* out);

/* The video latents' per-channel de-normalization constants, transcribed from
 * comfy/ldm/minimax/vae.py (LATENTS_MEAN / LATENTS_STD, 24 channels): a normalized latent is
 * (mean_moments - LATENTS_MEAN) / LATENTS_STD. */
extern const double kVaeLatentsMean[24];
extern const double kVaeLatentsStd[24];

/* Pixel normalization before the encoder: pixels in [-1,1] go through (x+1)/2, then per-channel
 * (x - IMAGENET_MEAN) / IMAGENET_STD. */
extern const double kVaePixelMean[3];
extern const double kVaePixelStd[3];

}   /* namespace minimaxh3 */

#endif
