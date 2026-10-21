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
 *   2. VENCODE or AENCODE of an admitted request's reference media, one per reference, so the
 *      video and audio VAE latents are in the lane's slots before its prompt runs.
 *   3. VCOND of an admitted request, so its steps can follow the current request's without a gap.
 *   4. VSTEP of the oldest lane with steps left, or its VUPSCALE once its small steps are done.
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
    using SegsFn      = int64_t (*)(const RadVideoGeom*, int32_t*, int64_t);
    using PackFn      = int64_t (*)(const RadVideoGeom*, const int32_t*, int32_t*, int64_t,
                                    float*, int64_t);
    using ModSegsFn   = int64_t (*)(const RadVideoGeom*, const int32_t*, int32_t*, int64_t);
    using EncodeGeomFn = int64_t (*)(int32_t, const int64_t*, int32_t*, int64_t);

    /* `positions` is the plugin's rad_arch_video_positions; `segs`, optional, is its
     * rad_arch_video_segs (null = the driver stages the plain segment table itself); `pack`,
     * optional, is its rad_arch_video_pack (null = reference-conditioned sequences cannot be
     * laid out and such requests are refused); `mod_segs`, optional, is its
     * rad_arch_video_mod_segs (null = the driver stages the fine segment table into the norm's
     * operand too, correct for any pack that merges nothing); `enc_geom`, optional, is its
     * rad_arch_encode_geom (null = no encode pass runs). */
    int  init(const Program& prog, PositionsFn positions, const std::vector<RankIO>& ranks,
              SegsFn segs = nullptr, PackFn pack = nullptr, ModSegsFn mod_segs = nullptr,
              EncodeGeomFn enc_geom = nullptr);
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
        /* The VAE references of a reference-conditioned request (RadVideoGenBatch::ref_tab):
         * the pack's input, empty until the path consumes them (it is refused at submit
         * before that). The step reads it host-side, so nothing stages it. */
        std::vector<int32_t> ref_tab;
        /* When the generator's denoiser consumes that table (`decl_.ref_rows`), the step's packed
         * sequence carries reference rows between the text and the target streams, laid out by
         * rad_arch_video_pack. These hold that step layout: `ref_wide` marks it, `seg` the pack's
         * segment boundaries with the video grid last, `nseg` their count, and `lat_v`/`lat_a` the
         * TARGET video and audio row counts -- the rows the sampler owns and the Euler steps update,
         * which the plain `rows - text - audio` arithmetic no longer recovers once references ride
         * ahead of the streams. Left empty/0 on the plain path and on every non-step pass (decode,
         * encode, upscale), which keep their own ref-less segment table. */
        bool                 ref_wide = false;
        int32_t              nseg = 0;
        std::vector<int32_t> seg;
        /* The same layout merged to modulation groups: consecutive segments sharing a cell collapse,
         * so `mod_seg` holds a SUBSET of `seg`'s boundaries and `nmod` (its count) is what the
         * modulation table -- `nmod - 1` cells -- is pitched by. The attention still runs on the fine
         * `seg`; only the norm reads `mod_seg`. Empty on the plain path (there the driver stages the
         * plain table into both operands). */
        std::vector<int32_t> mod_seg;
        int32_t              nmod = 0;
        int64_t              lat_v = 0, lat_a = 0;
        /* The denoising step's row count: the widened total once references ride, otherwise equal
         * to `rows`. Only build_step uses it -- the encode, decode, audio and upscale passes keep
         * staging against the plain `rows` (their segment table and target counts are their own,
         * never the reference layout), so widening the step cannot disturb them. `pos` widens with
         * it, since build_step is the only pass that stages R_ROPE from `pos`. */
        int64_t              step_rows = 0;
        /* The VAE encode queue: the reference media whose bytes reach the model through a video or
         * audio VAE, one item per Vencode/Aencode pass, in presentation order, built at submit for
         * the parts the server decoded into encoder form (gen/video.h enc_pixels/enc_samples).
         * `enc_kind` is the RAD_ENC_* the pass names and `enc_slot` its 1-based number within that
         * kind (the reference's index); `ref` indexes req->refs. `enc_next` is the next to run;
         * once it reaches enc_q.size() the media are all encoded. Empty for a request the encoder
         * has nothing to do with. */
        struct EncItem { int32_t kind, slot, ref; };
        std::vector<EncItem> enc_q;
        size_t               enc_next = 0;
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
    /* R_MMROW/R_ENCPIX/R_ENCCRD/R_ENCCU stage the conditioning pass's reference media, as the
     * image generator's driver does (core/gen/driver.cpp); R_ROPE carries a step's f32 positions
     * and a conditioning's [3,T] i32 M-RoPE, never in the same pass. */
    /* R_ENCPX/R_ENCSM/R_ENCG stage one media item for a Vencode/Aencode pass: the encoder-ready
     * pixels ([frames,h,w,3] f32) or samples ([ch,n] f32) and the encode geometry the plugin's
     * rad_arch_encode_geom returned, staged verbatim as device i32 for the kernels to read. */
    /* R_MOD is the norm's segment boundaries: the merged modulation-group table (a subset of
     * R_SEG's), which the modulated norm reads for its per-cell modulation while the attention
     * reads the fine R_SEG. The step stages the merged table; every other pass, and the plain
     * path, stages R_SEG's bytes into it as well. */
    enum { R_TOK, R_POS, R_CU, R_ROPE, R_GEOM, R_SCHED, R_TV, R_TA, R_SEED, R_LANE, R_FIRST,
           R_SEG, R_LAT, R_UP, R_MMROW, R_ENCPIX, R_ENCCRD, R_ENCCU,
           R_ENCPX, R_ENCSM, R_ENCG, R_MOD, R_N };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    /* The geometry and rows of the job's next step: the small ones before its upscale. */
    bool small_now(const Job& j) const { return j.split > 0 && !j.upscaled; }
    void base_batches(const Job& j, const RadVideoGeom& g, int64_t n_tok, int step);
    void stage_geom(const Job& j, const RadVideoGeom& g, int64_t rows, int32_t chunk,
                    bool ref_step = false);
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
    /* One Vencode or Aencode pass for the job's next queued reference (enc_q[enc_next]): stages
     * that part's encoder-ready pixels or samples and the encode geometry, and names its kind and
     * slot. Returns null and fails the job if the bytes no longer fit a pass. */
    const RadBatch* build_encode(Job& j, int step);

    RadVideoGenDecl     decl_{};
    std::vector<double> rungs_;        /* decl_.rungs points here */
    std::string         name_;
    PositionsFn         positions_ = nullptr;
    SegsFn              segs_ = nullptr;
    PackFn              pack_ = nullptr;
    ModSegsFn           mod_segs_ = nullptr;
    EncodeGeomFn        enc_geom_ = nullptr;
    int64_t             max_tok_ = 0;
    /* The conditioning's encoder view: the tower's per-patch width and the patches one pass
     * carries. Zero when the program declares no vision encoder, which bounds the media regions
     * and rejects any request that carries references. */
    int64_t             patch_dim_ = 0, max_enc_ = 0;
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

    /* Debug-only (R4D_ENC_DEBUG): a pinned slab the encoder's audio / video latent buffer is read
     * back into after a Vencode / Aencode pass, so done() can log a digest. Allocated and used
     * only when the environment is set and the generator named its encoders; otherwise enc_dbg_
     * is false and these stay null / zero, so a normal request never touches them. */
    bool                  enc_dbg_ = false;
    char*                 enc_dbg_host_ = nullptr;
    int64_t               enc_dbg_bytes_ = 0;   /* the slab's capacity */
    int64_t               enc_dbg_used_ = 0;    /* bytes the last encode readback filled */
    const char*           enc_dbg_sel_ = "";    /* R4D_ENC_SEL: "", "mp" or "pre" (audio) */

    /* Debug-only (#61, R4D_DIT_SEL): a pinned slab holding the DiT step-0 dump -- the f32 shadows
     * (h0 | temb | mod) the plugin declares when it sees the same env, plus the step's velocity
     * head output b_vel -- so done() can write them to R4D_ENC_DUMPDIR for the CPU reference DiT
     * replay. Staged only on a plain request's step 0; a plugin without the env leaves
     * decl_.dbg_dit_h0 null and dit_dbg_ stays false, so a normal run never touches these. */
    bool                  dit_dbg_ = false;
    char*                 dit_host_ = nullptr;
    int64_t               dit_h0_b_ = 0, dit_vel_b_ = 0, dit_temb_b_ = 0, dit_mod_b_ = 0;
    bool                  dit_used_ = false;    /* the running pass staged a dump */
    int64_t               dit_rows_ = 0;        /* packed rows R of the dumped step */

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
