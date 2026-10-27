/* rad_vae_geom.h -- the latents MiniMax-H3's VAE encoders make of a media's frames or samples,
 * and their normalisation constants, on the host. ComfyUI's comfy/ldm/minimax/vae.py and
 * audio_vae.py, tested against those encoders' own outputs (tests/data/vae_geom). */

#ifndef RADIANCE_ARCH_MINIMAXH3_RAD_VAE_GEOM_H
#define RADIANCE_ARCH_MINIMAXH3_RAD_VAE_GEOM_H

#include <cstdint>

namespace minimaxh3 {

/* The video VAE: 16 pixels a latent side, and in time clips of 17 frames to 5 latent frames
 * (17 -> 9 -> 5), of which an encode of more than one frame drops the last 3. */
constexpr int kVaeRatio = 16;
constexpr int kVaeClipLength = 17;
constexpr int kVaeClipLatents = 5;
constexpr int kVaeTokenDrop = 3;

/* The audio VAE: strides 2 4 4 5 5, a latent per 800 samples (40 a second at 32 kHz), the samples
 * padded with zeros to a whole latent. */
constexpr int kAudioHop = 800;

/* The latent frames of `frames` pixel frames: 1 for a picture, else 5 a clip of 17 (the last
 * padded by repeating its last frame) less 3. -1 for no frames. */
int64_t vae_latent_frames(int64_t frames);

/* The latent frames one clip of `frames` encodes to, ceil(T / 4) (one frame stays one). */
int64_t vae_clip_latents(int64_t frames);

/* The latents of `samples` samples; -1 for a negative count. */
int64_t vae_audio_latents(int64_t samples);

/* A video latent is normalised as (moments - mean) / std, vae.py's LATENTS_MEAN and LATENTS_STD. */
extern const double kVaeLatentsMean[24];
extern const double kVaeLatentsStd[24];

/* A pixel x in [0, 1] goes in as (x - mean) / std, ImageNet's. */
extern const double kVaePixelMean[3];
extern const double kVaePixelStd[3];

}   /* namespace minimaxh3 */

#endif
