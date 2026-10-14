/* gen.h -- image generation: what a request is, what comes back, and the driver that turns one
 * into the passes an image generator declares (rad_declare_generator, abi/rad_builder.h).
 *
 * A GENERATOR IS NOT DRIVEN BY TOKEN STEPS. Its work is three kinds of pass -- conditioning,
 * denoising, decoding -- and an image is tens of denoising passes over thousands of latent rows,
 * every one of them read whole, none of them sampled. The scheduler that serves text owns token
 * positions, prefix reuse, preemption and speculation, and none of that has a meaning here, so the
 * generator gets a driver of its own (this file) and shares everything below the step with the rest
 * of the engine: the ranks, the plugin's step function, the KV manager the conditioning lives in.
 *
 * THE PASSES, as the driver runs them:
 *
 *   COND     once for every conditioning sequence a request has -- its prompt, and with guidance
 *            its negative prompt -- the prompt (and its reference pictures) encoded into the
 *            generator's KV group. A request is admitted once its lanes and its KV are free.
 *   DENOISE  one step for each image in flight that fits the pass, any mix of requests, sizes and
 *            step counts: each image carries its own sigma. With guidance an image is two entries,
 *            its positive and its negative conditioning, which the plugin combines.
 *   DECODE   one image's latent to pixels, read back and handed to its request.
 *
 * A request's images run in LANES -- the plugin's own latent state, `max_lanes` of them -- so one
 * request of n images takes n lanes and shares one conditioning.
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
struct IKVManager;
namespace gen {

/* A picture as the generator's own encoder reads it: RGBA (or `ch` channels), row-major, each value
 * bf16 in [-1, 1]. */
struct Pixels {
    int32_t w = 0, h = 0, ch = 4;
    std::vector<uint16_t> bf16;
};

/* ONE CONDITIONING SEQUENCE: the prompt as tokens, with the media it carries placed, and how many
 * of its first tokens the generator does not see (RadGeneratorDecl::drop_prefix). */
struct Branch {
    std::vector<int32_t>             tokens;
    std::shared_ptr<mm::PromptMedia> media;    /* null for a prompt with no pictures */
    int32_t                          n_drop = 0;
};

/* ONE IMAGE, or one preview of one, on its way back. */
struct Image {
    int32_t              index = 0;     /* which of the request's n */
    int32_t              w = 0, h = 0, ch = 4;
    std::vector<uint8_t> px;            /* [h, w, ch] */
    bool                 preview = false;
    int32_t              preview_no = 0;
    uint64_t             seed = 0;
};

/* THE CHANNEL BACK TO THE CALLER: images as they finish, then the end. The driver's thread writes,
 * the caller's reads; a mutex is enough -- an image is megabytes and there are a handful a
 * request. `cancel` is the caller's, read by the driver at its next pass. */
class Ticket {
public:
    /* driver side */
    void push(Image&& im);
    void finish(int status, const std::string& why = std::string());
    void step(int32_t done, int32_t total);

    /* caller side: the next image, or false once the request has ended and every image was taken.
     * Waits at most `ms` milliseconds; `*timed_out` says a false was a timeout, not the end. */
    bool next(Image* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* What the request cost, for the response: the conditioning's tokens, the latent rows made, and
     * the time each phase took. Written by the driver before finish(). */
    int64_t in_tokens = 0, out_tokens = 0;
    int64_t t_submit_ns = 0, t_cond_ns = 0, t_first_step_ns = 0, t_denoised_ns = 0, t_done_ns = 0;
    int32_t steps = 0;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<Image>       q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
};

struct Request {
    Branch   pos, neg;
    bool     has_neg = false;
    float    cfg = 1.0f;
    int32_t  width = 0, height = 0;     /* pixels, whole multiples of the declared multiple */
    int32_t  steps = 0;                 /* 0: the model's default (or its fixed schedule) */
    int32_t  n = 1;
    uint64_t seed = 0;                  /* image i draws its noise from seed + i */
    int32_t  previews = 0;              /* partial images per image, 0..3 */
    /* The reference pictures' encoder view, one per media item of `pos`, in order. */
    std::vector<Pixels> refs;
    /* REPAINTING (an edit with a mask): the picture at the size being made, and per latent token
     * how much of it is repainted (1) and how much kept (0). */
    bool                 inpaint = false;
    Pixels               inp_src;
    std::vector<float>   inp_mask;
    std::shared_ptr<Ticket> ticket;
};

/* ================================================================== the sampler's schedule */
/* The sigmas a request steps through, n + 1 of them, the last 0: the decl's fixed schedule, or
 * `steps` (0: the default) shifted the way the decl says. `tokens` is the image's latent token
 * count, which a dynamic shift reads. f32, as the reference holds them. */
std::vector<float> schedule(const RadGeneratorDecl& d, const std::vector<double>& fixed,
                            int32_t steps, int64_t tokens);

/* ================================================================== the generator's sequence */
/* ONE CONDITIONING SEQUENCE AS THE GENERATOR SEES IT (RadGeneratorDecl::layout): which prompt row
 * or reference latent row each of its rows is, its rotary positions, its segments, and where the
 * image being made starts. */
struct Layout {
    int64_t              rows = 0;
    std::vector<int32_t> src;          /* [rows] */
    std::vector<int32_t> rope;         /* [3, rows] component-major */
    std::vector<int32_t> seg;          /* [n_seg + 1] */
    std::vector<int32_t> bidi;         /* [n_seg] */
    int32_t              next_pos = 0; /* the rotary count after the last row: the image's frame */
};
/* `ref_hw` is each reference's latent grid (h, w), in media order. RAD_E_INVAL with `why` when the
 * prompt does not start with its drop prefix or a picture's slots do not tile its latent grid. */
int build_layout(const Branch& b, int64_t slot_rows, const std::vector<std::pair<int32_t, int32_t>>& ref_hw,
                 Layout* out, std::string* why);
/* The rotary positions of an h x w image at frame `pos`: [3, h * w] component-major. */
void image_rope(int32_t pos, int32_t h, int32_t w, int32_t* planes, int64_t n_total);

}  /* namespace gen */
}  /* namespace rad */
