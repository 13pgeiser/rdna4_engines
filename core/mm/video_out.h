/* video_out.h -- a video and its soundtrack written to a file in memory, muxed into one of a few
 * containers: MP4 (H.264 video in yuv420p + AAC, the moov atom first) or WebM (VP9 or AV1 video +
 * Opus audio). The frames arrive in order, in any number of calls; the soundtrack may come before,
 * between or after them, whole or in pieces. Nothing is written until finish(), which encodes the
 * soundtrack and interleaves it with the video: a video is at most tens of seconds, and its frames
 * are encoded as they come so the RGB is never all held at once.
 *
 * The video encoder is the best the linked libavcodec has for the format: MP4 prefers libx264
 * (constant quality), else libopenh264, else whatever it registers for H.264; WebM uses libvpx-vp9
 * or libaom-av1 (else the generic encoder for that codec). Audio is AAC for MP4 and Opus for WebM;
 * since Opus only encodes at 48 kHz, a soundtrack at any other rate is resampled with libswresample
 * (AAC takes the soundtrack's own rate). Without libavcodec every call returns RAD_E_UNSUPPORTED;
 * without a format's encoder or muxer, open() returns it with a message naming what is missing. */
#pragma once
#include "rad_core.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace mm {

/* The container and codec pair open() writes. Mp4 is the default everywhere, so a caller that does
 * not choose one gets the H.264 + AAC file Radiance has always produced. */
enum class VideoFormat {
    Mp4,       /* H.264 + AAC, moov atom first */
    WebmVp9,   /* VP9 + Opus */
    WebmAv1,   /* AV1 + Opus */
};

class VideoWriter {
public:
    VideoWriter();
    ~VideoWriter();
    VideoWriter(const VideoWriter&) = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;

    /* A video of `width` x `height` (even) at fps_num / fps_den frames a second, with a soundtrack
     * of `channels` channels at `sample_rate` (0 channels: none), written in `fmt`. `fmt` trails the
     * error out-param and defaults to MP4 so the older MP4-only call sites keep compiling. */
    int open(int32_t width, int32_t height, int64_t fps_num, int64_t fps_den, int64_t sample_rate,
             int64_t channels, std::string* why, VideoFormat fmt = VideoFormat::Mp4);
    /* `n` frames, each [height, width, 3] u8 RGB, after those already added. */
    int add_frames(const uint8_t* rgb, int32_t n, std::string* why);
    /* Samples interleaved over the channels, in [-1, 1], at the rate passed to open(), after those
     * already added. */
    int add_audio(const float* samples, int64_t n_frames, std::string* why);
    /* Flushes both encoders and writes the file into `out`. */
    int finish(std::string* out, std::string* why);

    /* The video encoder in use, for the log. */
    const std::string& video_encoder() const { return venc_name_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::string           venc_name_;
};

}  /* namespace mm */
}  /* namespace rad */
