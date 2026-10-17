/* gen.h -- image generation: the request, what comes back, and the driver that turns a request
 * into the passes an image generator declares (rad_declare_generator, abi/rad_builder.h).
 *
 * A generator is not driven by token steps: an image is tens of denoising passes over thousands of
 * latent rows, all read whole, none sampled. Token positions, prefix reuse, preemption and
 * speculation have no meaning here, so the generator has its own driver and shares only what lies
 * below the step: the ranks, the plugin's step function and the KV manager holding conditioning.
 *
 * Passes, as the driver runs them:
 *
 *   COND     once per conditioning sequence (the prompt, plus the negative prompt with guidance):
 *            the prompt and its reference pictures encoded into the generator's KV group. A
 *            request is admitted once its lanes and its KV are free.
 *   DENOISE  one step for each in-flight image that fits the pass, any mix of requests, sizes and
 *            step counts; each image carries its own sigma. With guidance an image is two entries,
 *            positive and negative conditioning, which the plugin combines.
 *   DECODE   one image's latent to pixels, read back and handed to its request.
 *
 * Images run in lanes, the plugin's own latent state (`max_lanes` of them): a request of n images
 * takes n lanes and shares one conditioning.
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

/* One conditioning sequence: the prompt tokens with their media placed, and how many leading
 * tokens the generator does not see (RadGeneratorDecl::drop_prefix). */
struct Branch {
    std::vector<int32_t>             tokens;
    std::shared_ptr<mm::PromptMedia> media;    /* null for a prompt with no pictures */
    int32_t                          n_drop = 0;
};

/* One finished image, or a preview of one. */
struct Image {
    int32_t              index = 0;     /* which of the request's n */
    int32_t              w = 0, h = 0, ch = 4;
    std::vector<uint8_t> px;            /* [h, w, ch] */
    bool                 preview = false;
    int32_t              preview_no = 0;
    uint64_t             seed = 0;
};

/* The channel back to the caller: images as they finish, then the end. The driver thread writes,
 * the caller reads; a mutex suffices for a handful of megabyte images per request. `cancel` is set
 * by the caller and read by the driver at its next pass. */
class Ticket {
public:
    /* driver side */
    void push(Image&& im);
    void finish(int status, const std::string& why = std::string());

    /* caller side: the next image, or false once the request has ended and every image was taken.
     * Waits at most `ms` milliseconds; `*timed_out` marks a false that was a timeout, not the end. */
    bool next(Image* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* Usage for the response: conditioning tokens, latent rows made, and per-phase times.
     * Written by the driver before finish(). */
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
    /* Inpainting (an edit with a mask): the source picture at the output size, and per latent
     * token how much is repainted (1) versus kept (0). */
    bool                 inpaint = false;
    Pixels               inp_src;
    std::vector<float>   inp_mask;
    std::shared_ptr<Ticket> ticket;
};

/* ================================================================== the sampler's schedule */
/* The n + 1 sigmas a request steps through, the last 0: the decl's fixed schedule, or `steps`
 * (0: the default) shifted as the decl says. `tokens` is the image's latent token count, read by a
 * dynamic shift. f32, matching the reference. */
std::vector<float> schedule(const RadGeneratorDecl& d, const std::vector<double>& fixed,
                            int32_t steps, int64_t tokens);

/* ================================================================== the generator's sequence */
/* One conditioning sequence as the generator sees it (RadGeneratorDecl::layout): the prompt row or
 * reference latent row behind each row, rotary positions, segments, and the output image's frame. */
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
