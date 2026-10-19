/* Encode geometry, ported rule for rule from the reference VAEs (docs/NOTE-P2-ENCODE.md carries
 * the line-referenced extraction; the golden vectors in tests/data/vae_geom are generated from the
 * reference source itself -- including its real encoder runs, so the latent counts here are the
 * network's own outputs, not a hand-derived formula). */

#include "rad_vae_geom.h"

#include <algorithm>

namespace minimaxh3 {

/* Encode() for 2+ frames calls encode_temporal(): ceil(T/clip_length) clips, the short last one
 * padded by repeating its final frame to a full clip (so every clip contributes exactly
 * kVaeClipLatents latents -- the causal chain 17 -> ceil(17/2) -> ceil(9/2) = 5), concatenated,
 * minus token_drop. A single image encodes alone and keeps its one latent (the chain at T=1
 * truncates its temporal taps and never downsamples below one frame). */
int64_t vae_latent_frames(int64_t frames) {
    if (frames <= 0) return -1;
    if (frames == 1) return 1;
    const int64_t clips = (frames + kVaeClipLength - 1) / kVaeClipLength;
    return clips * kVaeClipLatents - kVaeTokenDrop;
}

int64_t vae_clip_latents(int64_t frames) {
    if (frames <= 0) return -1;
    if (frames == 1) return 1;
    return (frames + 3) / 4;   /* ceil(ceil(T/2)/2) collapses to ceil(T/4) */
}

/* encode(): right_pad = ceil(L/hop)*hop - L samples of zeros, then the stride chain divides the
 * padded length by the hop exactly (every strided conv k = 2s, p = ceil(s/2) is exact on lengths
 * divisible by s, and the padded length is divisible by the product). */
int64_t vae_audio_latents(int64_t samples) {
    if (samples < 0) return -1;
    return (samples + kAudioHop - 1) / kAudioHop;
}

void vae_split_tiles(int64_t length, std::vector<VaeTile>* out) {
    out->clear();
    if (length <= kVaeTileSize) {
        out->push_back({ 0, length, 0 });
        return;
    }
    /* Grow the window count until the minimum 64 px overlaps fit: each extra window adds 256 px
     * of reach but one more overlap. */
    int64_t n = (length + kVaeTileSize - 1) / kVaeTileSize;
    std::vector<int64_t> overlaps;
    for (;;) {
        overlaps.assign((size_t)(n - 1), kVaeTileOverlapMin);
        int64_t sum = 0;
        for (const int64_t o : overlaps) sum += o;
        const int64_t remaining = kVaeTileSize * n - sum - length;
        if (remaining < 0) { ++n; continue; }
        /* Slack the overlaps leave is spread in latent-width (16 px) units, round-robin, so the
         * windows reach exactly the end of the axis. */
        for (int64_t i = 0; i < remaining / kVaeRatio; ++i)
            overlaps[(size_t)(i % (n - 1))] += kVaeRatio;
        break;
    }
    int64_t start = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t len = std::min<int64_t>(kVaeTileSize, length - start);
        const int64_t ov = i < n - 1 ? overlaps[(size_t)i] : 0;
        out->push_back({ start, len, ov });
        if (i < n - 1) start += kVaeTileSize - overlaps[(size_t)i];
    }
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
