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
