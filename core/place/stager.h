/* stager.h -- the prefill stager: routed layers' non-resident experts copied onto the card ahead
 * of the layers that read them.
 *
 * ---- WHY A PREFILL NEEDS IT -------------------------------------------------------------------
 *
 * An expert the slab has no room for lives in the pinned host pool, and a GEMM that routes to it
 * reads it across the link in place (Site::DeviceZeroCopy). At decode that is the right trade: a
 * step routes a few dozen rows to a few dozen experts and reads a few of them from the host. A
 * prefill chunk routes tens of thousands of rows to nearly EVERY expert of every layer, so the
 * whole pooled part of each layer -- ~85 MB a card on Qwen3.8-Flash-Next -- crosses the link inside
 * the layer's two GEMMs, at the link's rate and on the critical path, while the rest of the layer's
 * work leaves the link idle. Measured with r4d_selftest --perfmoe, the gate_up GEMM of a 2048-token
 * chunk is 1.53 ms with every expert on the card and 4.34 ms with the coldest 13% in host memory.
 *
 * So at a step large enough, the stager copies the pooled experts of the layers ahead into a VRAM
 * ring while the layers before them run, publishes the copies for each layer's ops, and publishes
 * the host addresses again once those ops are behind it. The copy is the same bytes the GEMM would have read, so the output
 * is bit-identical; only when the link carries them moves.
 *
 * ---- WHERE THE RING IS, AND WHEN IT IS ITS ---------------------------------------------------
 *
 * One region, the size of two layers' share of the experts the budget leaves off the card, at the
 * top of the activation arena (Program::arena_stage_off, sized by vram_budget_resolve). A step
 * smaller than the arena's full plan LENDS that top to the expert slab, so the region is the
 * stager's only in a step at the full plan -- which is exactly the steps worth staging for -- and
 * the rest of the time it holds resident experts like any other lent slot. The scheduler says
 * which it is, pass by pass (set_full_arena).
 *
 * ---- WHY A RING AND NOT TWO BUFFERS ------------------------------------------------------------
 *
 * Layers do not leave equal parts off the card: at depth, with the KV cache's loan taken back, the
 * last third of a Qwen3.8-Flash-Next stack pools about twice the median layer. With two buffers of
 * one share each, layer L+1's copy waits for layer L-1's last op, so the engine idled once a light
 * layer's copy was done -- 60 to 80 ms of a ~280 ms step -- and a heavy layer's copy, longer than
 * the layer before it, made every layer after it wait. The ring packs each layer at its own size,
 * so the copies run as far ahead as the light layers leave room for and the heavy ones start
 * early. A layer staged further ahead than the next one takes at most half the ring, so the next
 * layer always has the room two buffers gave it; the next one takes whatever is free.
 *
 * ---- THE ORDER, AND WHAT KEEPS IT SAFE ---------------------------------------------------------
 *
 *   the pass's first op     stage the first routed layer the pass issues, and the ones after it
 *                           as far as the pace and the ring allow
 *   a layer's first op      wait on its copies; publish the host addresses of the layers before
 *                           it again; stage further layers into the room that gave back
 *   a layer's last op       record "free" behind it on the op's stream
 *   the pass's end          the same for anything still staged -- and, when the next step goes on
 *                           with a prompt, stage the first layers of its trunk pass
 *
 * A layer's place in the ring is counted in bytes the ring has handed out since bring-up, so a
 * place laps the region and a copy finds the places it overwrites from the count alone. Before a
 * copy goes out, the mover's stream waits on the "free" event of every earlier place it laps --
 * recorded behind the last op that read it, in this pass or an earlier one. The layer's ops wait on
 * the copies' event through ONE stream wait, issued before the layer's first op: the copies are
 * published settled (Mover::stage_units says why), and nothing but the layer's own ops ever
 * resolves them.
 *
 * ---- HOW FAR AHEAD ----------------------------------------------------------------------------
 *
 * A layer's copies go out once the compute stream has reached the MoE of the layer kLead before it,
 * through an event recorded at that layer's first op. Not as soon as the ring has room: copies that
 * land long before their layer is read cost the layers they overlap more than they save. Measured at
 * TP2 on Qwen3.8-Flash-Next, released as soon as there was room, every class of op ran slower and a
 * 26K-token prompt lost 0.6%; held to one layer ahead, the heavy late layers of a 160K-token prompt
 * waited again and it lost 2.4%. Three layers ahead covered the late layers at the smallest cost
 * to the short prompt.
 *
 * ---- THE NEXT STEP'S FIRST LAYERS -------------------------------------------------------------
 *
 * A step that carries a chunk of a prompt is issued only once the step before it has been read
 * back, so the copies of its first layers would start with it and its first layer would wait for
 * all of them. When the engine says the next step goes on with a prompt (set_next_prefill), the
 * pass's end stages the first layers of the next trunk pass instead, paced on this pass's last
 * layers as if the two passes were one. They are published at once. The next trunk pass adopts them
 * when it is armed; a trunk pass that is not publishes their host addresses again in begin_pass,
 * which the run phase calls before it brings the weight tables up to date. A draft pass in between
 * reads none of the trunk's layers and leaves them be. With both, against two buffers at TP2: 160K
 * 1.0-1.4% faster, 26K unchanged; at TP1, where the link carries every pooled expert each step,
 * 26K 0.6% faster.
 *
 * ---- ONLY THE TRUNK ---------------------------------------------------------------------------
 *
 * A draft head's routed layer is not the stager's. The head runs in passes of its own -- over a
 * prefill chunk, the history pass, which needs the head's experts on a prompt's last chunk alone
 * and there only for the rows that draft, a handful however long the chunk -- so its pooled
 * experts are read in place like a decode step's. The head's layer is numbered past the
 * trunk's layers (RadModelMeta::n_layers), left out at bring-up, and a draft pass is never armed.
 */
#pragma once
#include "../runtime/stager.h"
#include "mover.h"
#include "planner.h"

#include <deque>
#include <map>
#include <vector>

namespace rad {

class PrefillStager final : public OpStager {
public:
    ~PrefillStager() override;

    /* The routed layers of `plan` and the ops that read them, and `bytes` of device memory at
     * `region` as the ring. `min_rows` is the smallest pass it arms for. Leaves it disabled --
     * enabled() false -- when the plan has no routed layer with a pooled unit to stage. */
    int init(const Program& prog, const Plan& plan, Mover* mover, char* region, int64_t bytes,
             int64_t min_rows);

    /* Per pass, before it is released: whether the step runs at the arena's full plan, which is
     * when the region is not lent. */
    void set_full_arena(bool full) { full_ = full; }
    /* Per step, before it is released: whether the step after it goes on with a prompt in a pass
     * this stager arms for, so this step's trunk pass stages that pass's first layers. */
    void set_next_prefill(bool more) { next_prefill_ = more; }

    bool    enabled() const { return !layers_.empty(); }
    int64_t ring_bytes() const { return ring_; }
    /* The smallest pass (in tokens) the stager arms for: a step under it reads every pooled
     * expert of a routed layer synchronously. The scheduler keeps a held-back prefill chunk at
     * or above it (Scheduler::set_stage_rows). */
    int64_t min_rows() const { return min_rows_; }
    size_t  n_layers() const { return layers_.size(); }
    /* Whether the last trunk pass armed: its routed layers' pooled experts were copied onto the
     * card through the ring. A draft pass leaves it as the trunk pass set it. */
    bool    staged() const { return staged_; }
    static constexpr int32_t lead() { return kLead; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    struct Layer {
        int32_t              layer = -1;
        std::vector<int32_t> units;
        int32_t              first_op = -1, last_op = -1;
    };
    std::vector<Layer>   layers_;      /* the routed layers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */
    std::vector<int32_t> last_of_;     /* by op: the layer whose last op it is, or -1 */

    /* ONE LAYER'S PLACE IN THE RING. [beg, end) is in bytes handed out since bring-up; the
     * offset in the region is beg modulo its size, and a place never straddles the region's end. */
    struct Place {
        int32_t  li = -1;                      /* index into layers_ */
        int64_t  beg = 0, end = 0;
        RadEvent done = nullptr;               /* behind its copies, on the mover stream */
        RadEvent free = nullptr;               /* behind its last reader */
        bool     free_rec = false;
        /* Its last op has been issued at least once and `free` is recorded behind it. A step
         * longer than the MoE block's pass of rows issues a layer's ops once per pass, so the
         * last op can run again: the copies are published away at the NEXT layer's first op,
         * which is past every pass, while each run of the last op moves `free` behind itself. */
        bool     read = false;
        bool     retired = false;              /* its units are published at their pool slots */
        std::vector<int32_t> staged;
    };
    /* More places than a ring of two shares can hold layers of the smallest share a pass stages;
     * a full table only stops the copies going further ahead. */
    static constexpr int kPlaces = 16;
    Place    place_[kPlaces];
    std::deque<int32_t>  ring_order_;  /* places in the ring, oldest first; retired ones stay until lapped */
    std::vector<int32_t> spare_;       /* places not in the ring */

    Mover*   mover_ = nullptr;
    char*    region_ = nullptr;
    int64_t  ring_ = 0;
    int64_t  head_ = 0;                /* bytes handed out since bring-up */
    int64_t  min_rows_ = 0;

    /* THE PACE (see the top of this file): layer k's copies wait on the event recorded at the
     * first op of layer k - kLead, on the compute stream. A layer within kLead of the pass's first
     * routed layer has no such layer in its pass and goes at once. */
    static constexpr int32_t kLead = 3;
    static constexpr int kGates = 64;
    RadEvent gate_[kGates] = {};
    int32_t  gate_li_[kGates] = {};    /* the layer each gate was recorded at this pass, or -1 */
    int32_t  first_li_ = -1;           /* the pass's first routed layer */

    bool     full_ = false;
    bool     armed_ = false;
    /* armed_ as the last trunk pass's begin_pass set it, which end_pass leaves alone: the engine
     * asks after the pass has ended. */
    bool     staged_ = false;
    bool     next_prefill_ = false;
    bool     first_op_ = false;        /* the pass has issued nothing yet */
    int32_t  next_ = 0;                /* the next layer to stage this pass */
    /* The next trunk pass's first layers, staged at this pass's end: layers [0, n_early_). */
    int32_t  n_early_ = 0;

    enum { kStaged, kNone, kLater };
    int  stage_ahead(int32_t cur, bool* resync);
    int  stage_layer(int32_t li, bool now, RadEvent gate, int* got);
    void withdraw_early();
    int  take(int64_t want, bool partial, int32_t* out);
    int  recycle_front();
    void retire(Place& p);
};

/* ---- THE FILE STAGER: routed layers read from the container -------------------------------------
 *
 * A routed unit the plan put on the file tier (--weights-disk-tier) has no address of its own, and
 * a grouped GEMM is handed every expert's pointer at issue. So on EVERY pass -- a decode step as
 * much as a prefill chunk -- each routed layer's file-tier units are read into one of the mover's
 * two file buffers before the layer's first expert op and published there, and published as null
 * again when the next routed layer starts. Which experts the router picks cannot be known on the
 * host without a round trip, so the whole layer's file-tier part is read; a prefill chunk routes to
 * nearly all of it anyway, and a decode step is paced by the drive -- the trade the flag names.
 *
 *   the pass's first op      read the first routed layer the pass issues
 *   a layer's first op       release the layer before (record its buffer free on the stream, publish
 *                            null), wait on this layer's copies, and read the NEXT layer into the
 *                            buffer just released -- its copies queue behind that free event
 *   the pass's end           release whatever is still held
 *
 * A LAYER IS RELEASED WHEN THE NEXT ONE STARTS, not at its own last op: a routed block that slices a
 * large step into passes of `rows` tokens issues its expert ops once a slice, so the last declared
 * op of the layer comes round again, and a release there would leave the second slice reading null.
 * Repeats of the current layer's first op do nothing.
 *
 * THE READ IS ON THE ISSUING THREAD and blocks it; the card meanwhile runs whatever was issued
 * before it. A routed layer's file-tier part is gigabytes and its compute milliseconds, so the step
 * is the drive's either way, and a reader thread would buy the few percent the layer's own compute
 * is worth at the cost of a second writer of the residency table.
 *
 * A layer is read ahead only when the pass will issue it -- the same learned bound PrefillStager
 * keeps (by draft_pass), so a trunk pass does not read the MTP head's layer for nothing. */
class FileStager final : public OpStager {
public:
    ~FileStager() override;

    /* The routed layers of `plan` that hold a unit on the file tier and the ops that read them,
     * over the mover's two file buffers. Leaves it disabled -- enabled() false -- when the plan
     * puts no routed unit there; refuses a plan whose routed layers are read by interleaved ops,
     * because a file-tier unit has no other address to fall back on. */
    int init(const Program& prog, const Plan& plan, Mover* mover);

    bool    enabled() const { return !layers_.empty(); }
    size_t  n_layers() const { return layers_.size(); }
    int64_t buffer_bytes() const { return cap_; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    struct Layer {
        int32_t              layer = -1;
        std::vector<int32_t> units;
        int32_t              first_op = -1, last_op = -1;
    };
    std::vector<Layer>   layers_;      /* the routed layers with file-tier units, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */

    Mover*   mover_ = nullptr;
    char*    buf_[2] = { nullptr, nullptr };
    int64_t  cap_ = 0;
    RadEvent done_[2] = { nullptr, nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[2] = { nullptr, nullptr };   /* behind the ops that read a buffer's layer */
    bool     free_rec_[2] = { false, false };
    std::vector<int32_t> staged_[2];
    int32_t  holds_[2] = { -1, -1 };             /* the layer a buffer holds, or -1 */
    int32_t  cur_ = -1;                          /* the layer whose ops are being issued */

    bool     armed_ = false;
    bool     first_op_ = false;
    int32_t  pass_kind_ = 0;
    int32_t  last_issued_ = -1;
    std::map<int32_t, int32_t> pass_last_;

    bool will_issue(int32_t li) const;
    int  start(int32_t li);
    int  finish(int32_t b, RadStream s);
};

}  /* namespace rad */
