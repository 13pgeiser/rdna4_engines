/* vae_geom_test.cpp -- the encode geometry (arch/minimaxh3/rad_vae_geom.cpp) against golden vectors
 * produced by the reference VAEs themselves.
 *
 * The TU is pure host arithmetic, so it is compiled straight in here -- no rad_core, no device.
 * tests/data/vae_geom/golden.json comes from gen_golden.py, which runs the reference's real
 * encoders on CPU (video: EncoderFCN3D + the wrapper's own encode()/encode_temporal()/
 * split_tiles()/tiled_encode() with the reference constructor arguments; audio: the DAC encoder):
 * the latent counts below are the networks' own output shapes, the tile plans the reference's own
 * returns, and the constants its own tables. A mismatch is the port disagreeing with the model.
 */
#include "rad_test.h"
#include "rad_vae_geom.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace minimaxh3;

namespace {

nlohmann::json& golden() {
    static nlohmann::json g = [] {
        std::ifstream f(std::string(RAD_VAE_GOLDEN) + "/golden.json");
        if (!f.is_open()) {
            fprintf(stderr, "  FAIL golden.json missing -- regenerate with "
                            "tests/data/vae_geom/gen_golden.py\n");
            std::exit(1);
        }
        return nlohmann::json::parse(f);
    }();
    return g;
}

}   /* namespace */

TEST(vae_encode_latent_counts) {
    /* Every frame count goes through the reference encode() end to end: chunking, the padded last
     * clip, the trailing drop. The spatial entries pin the ratio (generated at 32x32) and the
     * latent width pins z_channels. */
    for (const auto& c : golden()["video"]["encode"]) {
        const int64_t t = c["T"];
        const auto& z = c["z_shape"];
        CHECK_EQ(vae_latent_frames(t), z[2].get<int64_t>());
        CHECK_EQ(z[1].get<int64_t>(), 24);
        CHECK_EQ(z[3].get<int64_t>() * kVaeRatio, 32);
        CHECK_EQ(z[4].get<int64_t>() * kVaeRatio, 32);
    }
}

TEST(vae_clip_chain_composes_with_encode) {
    /* A full clip's latents times the clip count, minus the drop, is the encode count -- and the
     * causal chain the golden's tiled_encode cases pin (no chunking, no drop: the raw clip) is
     * vae_clip_latents of the pixel frames. */
    for (const auto& c : golden()["video"]["encode"]) {
        const int64_t t = c["T"];
        if (t < 2) continue;
        const int64_t clips = (t + kVaeClipLength - 1) / kVaeClipLength;
        CHECK_EQ(clips * vae_clip_latents(kVaeClipLength) - kVaeTokenDrop, c["z_shape"][2].get<int64_t>());
    }
    for (const auto& c : golden()["video"]["tiled_encode"])
        CHECK_EQ(vae_clip_latents(c["x_shape"][2].get<int64_t>()), c["moments_shape"][2].get<int64_t>());
    /* A single frame truncates its temporal taps instead of padding them. */
    CHECK_EQ(vae_clip_latents(1), 1);
}

TEST(vae_split_tiles_matches_reference) {
    for (const auto& c : golden()["video"]["split_tiles"]) {
        const int64_t len = c["input_len"];
        std::vector<VaeTile> tiles;
        vae_split_tiles(len, &tiles);

        const auto& gs = c["starts"];
        const auto& go = c["overlaps"];
        CHECK_EQ(tiles.size(), gs.size());
        if (tiles.size() != gs.size()) continue;
        for (size_t i = 0; i < tiles.size(); ++i) {
            CHECK_EQ(tiles[i].start, gs[i].get<int64_t>());
            CHECK_EQ(tiles[i].overlap, i < go.size() ? go[i].get<int64_t>() : 0);
            /* The reference slices [start, start+256) and lets the slice clamp at the axis end --
             * that clamped length is what the tile encoder really receives. */
            CHECK_EQ(tiles[i].len, std::min<int64_t>(kVaeTileSize, len - tiles[i].start));
        }
        /* Consecutive windows share exactly their overlap, so the latents the blend keeps sum to
         * the axis when the encoder's own 16 px alignment holds. */
        if (len % kVaeRatio == 0) {
            int64_t covered = 0;
            for (const VaeTile& t : tiles) covered += t.len - t.overlap;
            CHECK_EQ(covered, len);
        }
    }
}

TEST(vae_tiled_encode_grid_keeps_size) {
    /* The reference always tiles; the blend cuts keep the moments at exactly ratio-reduced size.
     * That is the tile plan plus arithmetic -- the same statement the real network run above
     * makes, port-side. */
    for (const auto& c : golden()["video"]["tiled_encode"]) {
        const auto& x = c["x_shape"];
        const auto& m = c["moments_shape"];
        for (int axis = 0; axis < 2; ++axis) {
            const int64_t px = axis == 0 ? x[3].get<int64_t>() : x[4].get<int64_t>();
            const int64_t lat = axis == 0 ? m[3].get<int64_t>() : m[4].get<int64_t>();
            std::vector<VaeTile> tiles;
            vae_split_tiles(px, &tiles);
            int64_t total = 0;
            for (const VaeTile& t : tiles) {
                CHECK_EQ(t.len % kVaeRatio, 0);      /* whole latents, or the cut would slice one */
                CHECK_EQ(t.overlap % kVaeRatio, 0);
                total += t.len / kVaeRatio - t.overlap / kVaeRatio;
            }
            CHECK_EQ(total, lat);
        }
    }
}

TEST(vae_audio_latents_match_encoder) {
    /* The wrapper right-pads to the hop and the stride chain divides exactly: the reference
     * encoder's own frame count is ceil(samples / 800). */
    for (const auto& c : golden()["audio"]["latents"]) {
        const int64_t s = c["samples"];
        CHECK_EQ(vae_audio_latents(s), c["T"].get<int64_t>());
        CHECK_EQ(vae_audio_latents(s) * kAudioHop, c["padded"].get<int64_t>());
    }
}

TEST(vae_constants_are_the_reference_tables) {
    /* The de-normalization the latents carry: doubles compare bit for bit, since the golden
     * carries the same decimal literals. */
    const auto& k = golden()["video"]["constants"];
    const auto mean = k["latents_mean"].get<std::vector<double>>();
    const auto stdv = k["latents_std"].get<std::vector<double>>();
    CHECK_EQ(mean.size(), 24u);
    CHECK_EQ(stdv.size(), 24u);
    for (size_t i = 0; i < mean.size() && i < 24; ++i) {
        CHECK(kVaeLatentsMean[i] == mean[i]);
        CHECK(kVaeLatentsStd[i] == stdv[i]);
    }
    const auto pm = k["pixel_mean"].get<std::vector<double>>();
    const auto ps = k["pixel_std"].get<std::vector<double>>();
    for (size_t i = 0; i < 3; ++i) {
        CHECK(kVaePixelMean[i] == pm[i]);
        CHECK(kVaePixelStd[i] == ps[i]);
    }
}

TEST(vae_geom_rejects_nonsense_and_covers_small_axes) {
    CHECK_EQ(vae_latent_frames(0), -1);
    CHECK_EQ(vae_latent_frames(-5), -1);
    CHECK_EQ(vae_clip_latents(0), -1);
    CHECK_EQ(vae_audio_latents(-1), -1);
    CHECK_EQ(vae_audio_latents(0), 0);
    std::vector<VaeTile> tiles;
    vae_split_tiles(1, &tiles);
    CHECK_EQ(tiles.size(), 1u);
    CHECK_EQ(tiles[0].start, 0);
    CHECK_EQ(tiles[0].len, 1);
    CHECK_EQ(tiles[0].overlap, 0);
    vae_split_tiles(kVaeTileSize, &tiles);
    CHECK_EQ(tiles.size(), 1u);
    CHECK_EQ(tiles[0].len, kVaeTileSize);
}

RAD_TEST_MAIN()
