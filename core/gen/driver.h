/* driver.h -- the image generator's pass driver (gen.h says what the passes are).
 *
 * ONE THREAD DRIVES IT, the engine's, and it asks two things a pass: `next()` decides what runs
 * and stages it, `done()` folds the result back once every rank has run it. Requests arrive from
 * HTTP threads through `submit()` and leave through their Ticket; the queue is the only thing the
 * two kinds of thread share, behind one mutex.
 *
 * WHAT RUNS NEXT, in this order, one pass at a time:
 *
 *   1. a DECODE an image is owed -- a finished image, or a preview its last step marked. First,
 *      because a finished image frees its lane and answers its caller, and a preview must be read
 *      before the lane's next step writes the preview latent again.
 *   2. a COND an admitted request is owed: its conditioning, so its images can join the next
 *      denoise pass.
 *   3. a DENOISE over as many images in flight as fit the step's rows, oldest request first, each
 *      at its own step.
 *
 * A request is ADMITTED when it can get lanes for every image and KV for every conditioning
 * sequence; until then it waits, in order -- a request that does not fit blocks the ones behind
 * it, so a large request is not starved by a stream of small ones.
 *
 * THE STAGING is one region per rank on its card and one pinned host region with the same layout.
 * Every pass is synchronised before the next is built (Engine::gen_loop), so one host region is
 * enough: an image pass is tens of milliseconds and more, and the host work between two is
 * microseconds.
 */
#pragma once
#include "gen.h"
#include "device/device.h"

#include <mutex>
#include <unordered_map>

namespace rad {
namespace gen {

class Driver {
public:
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
    };

    /* `kv` is rank 0's KV manager: one block table serves every rank. */
    int  init(const Program& prog, const Config& cfg, IKVManager* kv, const std::vector<RankIO>& ranks);
    void fini();
    ~Driver() { fini(); }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog, const Config& cfg);

    /* ---------------------------------------------------------- HTTP threads */
    /* Queue a request. Refused at once -- with `why` -- when it can never run: a size past the
     * largest image, more pictures than the model takes, a conditioning longer than a pass. */
    int  submit(std::unique_ptr<Request> r, std::string* why);
    /* What a request may ask for: the model's own limits, for the server to check before it does
     * the expensive part of preparing one. */
    const RadGeneratorDecl&    decl() const { return decl_; }
    const std::vector<double>& fixed_sigmas() const { return fixed_; }
    int64_t max_cond_tokens() const { return max_tok_; }
    /* Fail every request queued or in flight, and refuse new ones. */
    void shutdown(const char* why, int status);

    /* ---------------------------------------------------------- the engine thread */
    /* The next pass, staged on every rank's stream, or null when there is nothing to run.
     * `step` is the engine's pass counter, carried into RadBatch::step. */
    const RadBatch* next(int step);
    const RadBatch* batch_for(int rank) const;
    /* The bytes of the generator's image buffer the pass just staged makes, 0 for a pass that
     * makes none: the engine copies them to `host()` behind the pass. */
    int64_t decode_bytes() const { return dec_bytes_; }
    void*   host_image() const { return img_host_; }
    /* The pass has run on every rank and its streams are drained. */
    int  done();
    bool busy() const;

    /* For /health, /stats and the dashboard. */
    int64_t queued() const;
    int64_t running() const;

private:
    struct Cond {
        Layout   lay;
        uint64_t seq = 0;          /* the KV manager's key */
        bool     have_kv = false;
        bool     encoded = false;
        std::vector<int32_t> rope_img;   /* [3, image tokens]: the image's rows at this sequence */
    };
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<Request> req;
        int      n_branch = 1;
        Cond     cond[2];
        int32_t  h_lat = 0, w_lat = 0;
        int64_t  tokens = 0;
        std::vector<float> sigmas;
        std::vector<int32_t> lanes;
        std::vector<std::pair<int32_t, int32_t>> ref_hw;
        int32_t  images_done = 0;
        bool     admitted = false;
        bool     failed = false;
    };
    struct Lane {
        int64_t  job = -1;          /* index into jobs_ by id; -1 free */
        uint64_t job_id = 0;
        int32_t  index = 0;
        int32_t  step = 0;
        uint64_t seed = 0;
        std::vector<int32_t> preview_at;   /* the steps after which a preview is decoded */
        int32_t  previews_done = 0;
        bool     preview_due = false;
        bool     final_due = false;
        bool     finished = false;
    };
    /* What the pass being run is, for done(). */
    struct Pass {
        int      kind = 0;
        uint64_t job = 0;
        int      branch = 0;
        int32_t  lane = -1;
        bool     preview = false;
        std::vector<int32_t> lanes;   /* DENOISE: the lanes it stepped, one per image */
        std::vector<char>    marks;   /* ... and whether each was marked for a preview */
    };

    /* Staging regions: their offsets in the host span and every rank's slab. */
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    enum { R_TOK, R_POS, R_ROPE, R_CU, R_MMROW, R_ENCPIX, R_ENCCRD, R_ENCCU, R_SRC, R_GROPE,
           R_SEG, R_SLOT, R_PX, R_MASK, R_SIG, R_SIGN, R_CFG, R_PAIR, R_LANE, R_FLAGS, R_SEED,
           R_BT, R_USED, R_N };

    int  layout_regions();
    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    void finish_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int step);

    int  admit_locked();
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why);
    Job* job_locked(uint64_t id);

    const RadBatch* build_cond(Job& j, int branch, int step);
    const RadBatch* build_denoise(const std::vector<int32_t>& lanes, int step);
    const RadBatch* build_decode(int32_t lane, bool preview, int step);

    RadGeneratorDecl      decl_{};
    std::vector<double>   fixed_;
    int64_t               max_tok_ = 0, max_seqs_ = 0, max_enc_ = 0, patch_dim_ = 0;
    int64_t               max_rows_ = 0;     /* the longest conditioning sequence */
    int64_t               max_px_ = 0;       /* pixels the staging holds */
    int64_t               block_size_ = 0, max_blocks_ = 0;
    int32_t               kv_group_ = 0;
    std::vector<RankIO>   ranks_;
    IKVManager*           kv_ = nullptr;

    char*                 host_ = nullptr;   /* pinned, same layout as each slab */
    int64_t               bytes_ = 0;
    std::vector<void*>    dev_;
    Region                reg_[R_N];
    void*                 img_host_ = nullptr;
    int64_t               img_bytes_ = 0;
    int64_t               dec_bytes_ = 0;

    std::vector<RadBatch>        out_;     /* one per rank */
    std::vector<RadGenBatch>     gout_;
    std::vector<RadKVGroupBatch> kout_;
    Pass                         pass_;
    /* The host arrays a COND pass's plugin reads on the host (RadGenBatch's _h fields). */
    std::vector<int32_t>         host_bidi_, host_px_, host_lanes_;

    mutable std::mutex           mu_;
    std::deque<std::unique_ptr<Job>> jobs_;  /* arrival order */
    std::vector<Lane>            lanes_;
    std::vector<int32_t>         free_lanes_;
    uint64_t                     next_id_ = 1;
    uint64_t                     next_seq_ = 1;
    bool                         closed_ = false;
    std::string                  closed_why_;
    int                          closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
