/* video_driver.h -- the video generator's pass driver (video.h; abi/rad_builder.h's
 * RadVideoGenDecl describes the passes).
 *
 * Driven by the engine thread like the other drivers (driver.h): `next()` picks and stages a
 * pass, `done()` folds its result back once every rank has run it. Requests arrive from HTTP
 * threads through `submit()` and leave through their VideoTicket.
 *
 * A request holds a lane from admission to its last chunk. Its geometry, schedule and rotary
 * positions are fixed at submit, so nothing is computed between passes but the staging.
 *
 * Pass priority, one pass at a time:
 *
 *   1. ADECODE, then each VDECODE chunk in order, of a lane whose denoising is done: they finish
 *      a request and free its lane.
 *   2. VCOND of an admitted request, so its steps can follow the current request's without a gap.
 *   3. VSTEP of the oldest lane with steps left, or its VUPSCALE once its small steps are done.
 *      Requests are denoised one after another, not in turns: a pass is one request's whole
 *      sequence, so taking turns would only delay the first.
 */
#pragma once
#include "video.h"
#include "pass.h"
#include "device/device.h"

#include <mutex>

namespace rad {
namespace gen {

class VideoDriver final : public IPassDriver {
public:
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
    };
    using PositionsFn = int64_t (*)(int, const RadVideoGeom*, float*, int64_t);

    /* `positions` is the plugin's rad_arch_video_positions. */
    int  init(const Program& prog, PositionsFn positions, const std::vector<RankIO>& ranks);
    void fini();
    ~VideoDriver() override { fini(); }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog);

    /* ---------------------------------------------------------- HTTP threads */
    /* Queue a request, refused at once with `why` when it can never run. */
    int  submit(std::unique_ptr<VideoRequest> r, std::string* why);
    const RadVideoGenDecl& decl() const { return decl_; }
    void shutdown(const char* why, int status) override;

    /* ---------------------------------------------------------- the engine thread */
    const RadBatch* next(int step) override;
    const RadBatch* batch_for(int rank) const override;
    const std::vector<Readback>& readbacks() const override { return rb_; }
    int  done() override;

    int64_t queued() const override;
    int64_t running() const override;

private:
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<VideoRequest> req;
        RadVideoGeom  geom{};
        int64_t       rows = 0, dec_rows = 0;
        int32_t       n_chunks = 0;
        int64_t       n_samples = 0;     /* the soundtrack's samples a channel */
        VideoSchedule sched;
        std::vector<float> pos;          /* [3, rows] */
        /* Latent upscaling: the evaluations at the small size (0: none), its geometry, rows and
         * positions, the factor the network is told, and whether the upscale has run. */
        int32_t       split = 0;
        RadVideoGeom  geom_small{};
        int64_t       rows_small = 0;
        std::vector<float> pos_small;
        float         factor = 0.0f;
        bool          upscaled = false;
        int32_t  lane = -1;
        bool     admitted = false;
        bool     cond_done = false;
        int32_t  step = 0;               /* steps done */
        bool     audio_done = false;
        int32_t  chunk = 0;              /* chunks decoded */
        int32_t  frames_out = 0;         /* frames handed to the caller */
        bool     failed = false;
        bool     finished = false;
    };
    /* The pass in flight, for done(). */
    struct Pass {
        int      kind = 0;
        uint64_t job = 0;
        int32_t  n_frames = 0;
    };

    /* Staging regions: their offsets in the host span and every rank's slab. */
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    enum { R_TOK, R_POS, R_CU, R_ROPE, R_GEOM, R_SCHED, R_TV, R_TA, R_SEED, R_LANE, R_FIRST,
           R_SEG, R_LAT, R_UP, R_N };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    /* The geometry and rows of the job's next step: the small ones before its upscale. */
    bool small_now(const Job& j) const { return j.split > 0 && !j.upscaled; }
    void base_batches(const Job& j, const RadVideoGeom& g, int64_t n_tok, int step);
    void stage_geom(const Job& j, const RadVideoGeom& g, int64_t rows, int32_t chunk);
    void stage_sched(const Job& j, int32_t i, bool to_clean);
    void stage_seed(const Job& j);

    void admit_locked();
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why);
    Job* job_locked(uint64_t id);

    const RadBatch* build_cond(Job& j, int step);
    const RadBatch* build_step(Job& j, int step);
    const RadBatch* build_upscale(Job& j, int step);
    const RadBatch* build_audio(Job& j, int step);
    const RadBatch* build_chunk(Job& j, int step);

    RadVideoGenDecl     decl_{};
    std::vector<double> rungs_;        /* decl_.rungs points here */
    std::string         name_;
    PositionsFn         positions_ = nullptr;
    int64_t             max_tok_ = 0;
    std::vector<RankIO> ranks_;

    char*               host_ = nullptr;   /* pinned, same layout as each slab */
    int64_t             bytes_ = 0;
    std::vector<void*>  dev_;
    Region              reg_[R_N];
    uint8_t*            frames_host_ = nullptr;
    int64_t             frames_bytes_ = 0;
    float*              audio_host_ = nullptr;
    int64_t             audio_bytes_ = 0;
    std::vector<Readback> rb_;

    std::vector<RadBatch>         out_;      /* one per rank */
    std::vector<RadVideoGenBatch> vout_;
    Pass                          pass_;

    mutable std::mutex               mu_;
    std::deque<std::unique_ptr<Job>> jobs_;  /* arrival order */
    std::vector<int32_t>             free_lanes_;
    uint64_t                         next_id_ = 1;
    bool                             closed_ = false;
    std::string                      closed_why_;
    int                              closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
