/* rad_vae_geom.cpp -- see rad_vae_geom.h. */

#include "rad_vae_geom.h"

namespace minimaxh3 {

/* vae.py's encode_temporal for more than one frame, its plain encode for one. */
int64_t vae_latent_frames(int64_t frames) {
    if (frames <= 0) return -1;
    if (frames == 1) return 1;
    const int64_t clips = (frames + kVaeClipLength - 1) / kVaeClipLength;
    return clips * kVaeClipLatents - kVaeTokenDrop;
}

int64_t vae_clip_latents(int64_t frames) {
    if (frames <= 0) return -1;
    if (frames == 1) return 1;
    return (frames + 3) / 4;   /* ceil(ceil(T / 2) / 2) */
}

/* Every strided convolution (k = 2 s, pad ceil(s / 2)) divides a padded length exactly. */
int64_t vae_audio_latents(int64_t samples) {
    if (samples < 0) return -1;
    return (samples + kAudioHop - 1) / kAudioHop;
}

const double kVaeLatentsMean[24] = {
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
};

const double kVaePixelMean[3] = { 0.485, 0.456, 0.406 };
const double kVaePixelStd[3] = { 0.229, 0.224, 0.225 };

const double kVaeLatentsStd[24] = {
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
};

}   /* namespace minimaxh3 */
