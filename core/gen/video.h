/* video.h -- video generation: the request and the channel its video comes back on.
 *
 * A video generator (abi/rad_builder.h, rad_declare_video_generator) makes a video and its
 * soundtrack in one denoising loop over a packed sequence, then decodes the soundtrack and the
 * frames, a chunk of frames at a time. core/gen/video_driver.h turns a request into those passes;
 * this is what a caller hands it and what it hands back: progress as the steps run, the
 * soundtrack, the frames chunk by chunk in order, then the end.
 */
#pragma once
#include "rad_core.h"
#include "mm/processor.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace gen {

/* One thing a request has made. */
struct VideoEvent {
    enum Kind { STEP = 1, AUDIO = 2, FRAMES = 3 };
    int                  kind = 0;
    int32_t              steps_done = 0;   /* STEP: denoising steps finished */
    /* AUDIO: the whole soundtrack, interleaved, in [-1, 1]. */
    std::vector<float>   samples;
    /* FRAMES: frames [first, first + n) of the video, each [height, width, 3] u8 RGB. */
    int32_t              first = 0, n = 0;
    std::vector<uint8_t> rgb;
};

/* As Ticket (gen.h) is for an image: the driver thread writes, the caller reads. */
class VideoTicket {
public:
    /* driver side */
    void push(VideoEvent&& e);
    void finish(int status, const std::string& why = std::string());

    /* caller side: the next event, or false once the request has ended and every event was
     * taken. Waits at most `ms` milliseconds; `*timed_out` marks a false that was a timeout. */
    bool next(VideoEvent* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* What the request makes, written by the driver at submit: the frames and their size, the
     * steps, and the soundtrack's rate and channels. */
    int32_t frames = 0, width = 0, height = 0, steps = 0;
    int64_t sample_rate = 0, channels = 0;
    double  fps = 0.0;
    /* With latent upscaling, the size the first evaluations run at and the evaluations after the
     * upscale; zero without. */
    int32_t small_width = 0, small_height = 0, refine = 0;
    /* Usage, written by the driver before finish(): prompt tokens, rows a step and per-phase
     * times (t_small_ns the last small step done, t_upscaled_ns the upscale; zero without). */
    int64_t in_tokens = 0, rows = 0;
    int64_t t_submit_ns = 0, t_cond_ns = 0, t_small_ns = 0, t_upscaled_ns = 0, t_denoised_ns = 0,
            t_done_ns = 0;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<VideoEvent>  q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
};

/* One file of a reference-conditioned request (task "ref2va"): an image or a video that
 * conditions identity and motion, a video's separately-given soundtrack, or a standalone audio.
 * The server reads the parts' bytes, bounds them and orders them; decoding them and turning
 * them into conditioning rows is the generator's. `index` is the part's 1-based number within
 * its kind, which fixes its slot in the conditioning; VIDEO_AUDIO #i is the soundtrack of
 * VIDEO #i. */
struct VideoRefMedia {
    /* FIRST_FRAME / LAST_FRAME are endpoint KEYFRAMES, not references: a keyframe completion pins
     * one output frame (first to frame 0, last to the final frame) and diffuses the gap between.
     * They are never conditioned on as identities, so they take no reference row -- the driver folds
     * each into the pack's keyframe table and encodes it into an image slot (a keyframe request
     * carries no references, so those slots are free). The server resizes each to the target canvas
     * (first a plain stretch, last a centre crop) and sets enc_pixels to one frame at that size, so
     * its latent grid is the target's exactly and its pinned rows are frame_rows wide. */
    enum Kind : int32_t { IMAGE = 0, VIDEO = 1, VIDEO_AUDIO = 2, AUDIO = 3, FIRST_FRAME = 4, LAST_FRAME = 5 };
    Kind        kind = IMAGE;
    int32_t     index = 0;
    std::string name, filename, content_type;   /* the part's name and what the client called the file */
    std::string data;                           /* the file's bytes */

    /* The reference in the encoder's own form, prepared by the server (which owns the decoder) so
     * the driver stages bytes without ever touching a codec; they are empty until the server decodes
     * the part. These are the NATURAL decoded shapes — the driver's encode pass, which owns the
     * encoder's geometry, does the padding the encoder kernels demand (clip-padding a video's
     * frames, zero-padding an audio's samples). IMAGE/VIDEO fill the pixels, AUDIO/VIDEO_AUDIO the
     * samples.
     *   enc_pixels: f32 [enc_frames][enc_h][enc_w][3] row-major RGB in 0..255, UN-normalised —
     *     the video VAE applies ImageNet mean/std itself. Each side is snapped down to a multiple
     *     of the VAE's spatial ratio and capped at the declaration's max_enc_side, aspect kept,
     *     because our encoder runs one volume and cannot tile the way the reference does. Frames are
     *     the reference's sampled, trimmed count (for a still image, 1); the final short clip is
     *     padded by the driver, not here.
     *   enc_samples: f32 [enc_channels][enc_n] channel-major at the model's audio rate. A mono
     *     source is spread across the channels (the audio VAE encodes one channel at a time); the
     *     tail is right-padded with zeros to a whole hop by the driver. */
    std::vector<float> enc_pixels;
    int32_t            enc_frames = 0, enc_h = 0, enc_w = 0;
    std::vector<float> enc_samples;
    int64_t            enc_n = 0;
    int32_t            enc_channels = 0;
};

struct VideoRequest {
    std::vector<int32_t> tokens;   /* the prompt, rendered and tokenised */
    int32_t  width = 0, height = 0;   /* pixels, within the declaration's limits */
    int32_t  chunks = 0;           /* the length in chunks (RadVideoGenDecl) */
    int32_t  steps = 0;            /* sigma grid points; 0: the model's default */
    /* Latent upscaling (abi/rad_builder.h, RadVideoGenDecl): the factor, 0 for none, and the
     * evaluations at the full size after it, 0 for the model's default. */
    double   upscale = 0.0;
    int32_t  refine = 0;
    uint64_t seed = 0;
    /* Reference conditioning, sorted by (kind, index); empty for a text-only request. The bytes
     * are the VAE's (P2): reference videos and audios become conditioning rows through the
     * video/audio VAEs, not the TE. */
    std::vector<VideoRefMedia> refs;
    /* The references' TE view: every image and video reference processed to patches, placed in
     * `tokens` at their presentation markers, with the M-RoPE layout over the whole conditioning
     * (§3's presentation order). The server decodes and processes; the driver stages it into the
     * conditioning pass's encoder fields (RadBatch::enc_* / rope_pos / mm_rows) exactly as the
     * image generator's driver does. Null for a request whose references carry nothing the TE
     * sees — audios are label-only, and a text-only request has none. */
    std::shared_ptr<mm::PromptMedia> media;
    std::shared_ptr<VideoTicket> ticket;
};

/* A request's shape under a declaration: frames, latent grid, soundtrack latents and the rows of
 * a step. Shared by the driver and the server, which checks a request before preparing it. */
struct VideoShape {
    int32_t frames = 0, lat_frames = 0, lat_h = 0, lat_w = 0, n_audio = 0;
    int64_t video_rows = 0, audio_rows = 0;
};
VideoShape video_shape(const RadVideoGenDecl& d, int32_t width, int32_t height, int32_t chunks);

/* The small size a request's first evaluations run at under latent upscaling by `factor`: each
 * side divided by it and rounded to the nearest multiple of the size multiple, at least one
 * multiple and at most the side. */
void video_small_size(const RadVideoGenDecl& d, int32_t width, int32_t height, double factor,
                      int32_t* small_w, int32_t* small_h);

/* The two schedules a request of `steps` grid points runs (RadVideoGenDecl, "The sampler"): each
 * the f32 sigmas, last 0, and the model evaluations it takes, the shorter of the two. With a
 * trained schedule any other step count than n_rungs + 1 gives no steps. */
struct VideoSchedule {
    std::vector<float> sig_v, sig_a;
    int32_t            steps = 0;
};
VideoSchedule video_schedule(const RadVideoGenDecl& d, int32_t steps);

}  /* namespace gen */
}  /* namespace rad */
