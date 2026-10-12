/* xprefetch.h -- the next-layer expert prefetch at decode: what the host half (xprefetch.cpp) and the
 * kernels (xprefetch.hip) share.
 *
 * WHAT IT DOES. Under expert-tiered placement a routed layer's GEMMs read the experts that are not on
 * the card from the pinned host pool, across the link, while they run. A layer's router input predicts
 * the NEXT routed layer's choice well: that layer's router applied to it ranks the experts the next
 * layer will route. So at each routed layer the prefetch applies the next layer's router to this
 * layer's router input, takes the likeliest experts that sit in the pinned pool, copies them onto the
 * card with the DMA engine while the rest of this layer runs, and points the next layer's weight tables
 * at the copies just before its GEMMs. The GEMMs then read those experts from the card.
 *
 * NOTHING IT DOES CHANGES A RESULT. A copy holds the bytes of the pool slot it was made from, and an
 * entry is pointed at it only where it still names that slot (a compare-and-swap: a heat-engine move
 * since then wins) and only once the copy has landed. Late copies are not used; nothing waits for them.
 *
 * THE PIECES, per routed layer l predicting layer l+1:
 *   - at l's router input (Ctx::xp_predict): on a stream of its own, behind the compute stream's
 *     dispatch that reads the input (a ride), xp_logits applies l+1's router to the input rows and
 *     xp_plan picks the copies and writes them to a mailbox in pinned host memory;
 *   - after l's routed down GEMM (Ctx::xp_mark): a ride on the compute stream that the DMA list polls,
 *     so the copies cross the link under l+1's attention rather than under l's own streamed experts;
 *   - a host thread hands each mailbox to the card's prefetch DMA queue: wait for the mark, copy, then
 *     write the request's number to the group's done word;
 *   - ahead of l+1's routed GEMMs (Ctx::xp_step, on the compute stream): xp_step puts the previous
 *     layer's entries back (its GEMMs are behind it in the stream) and points l+1's at the copies if
 *     the done word says they have landed.
 * The ring holds kXpGroups layers' copies, kXpSlots each; a group is copied into again three layers
 * later, after the step that put its entries back. */
#pragma once
#include <cstdint>

namespace rad {

constexpr int kXpSlots    = 2;    /* copies a layer: the link fits about two units between the marks */
constexpr int kXpGroups   = 3;
constexpr int kXpMaxRows  = 16;   /* decode passes; a wider pass makes no prediction */
constexpr int kXpMaxK     = 16;
constexpr int kXpMaxNe    = 512;
constexpr int kXpMaxPools = 4;
constexpr int kXpParts    = 4;    /* the weights one expert's unit is made of */

/* One copied unit: its table entries, what they held and what they hold while patched, and where the
 * layer's routing counts say whether the step used it. */
struct XpRec {
    void**         ent[kXpParts];
    void*          oldv[kXpParts];
    void*          newv[kXpParts];
    const int32_t* cnt;
    int32_t        place;
    int32_t        pad;
};
struct XpJob {
    uint64_t src, dst, bytes, tag;    /* tag: the request's number, written after the rest */
};
/* A group's mailbox, in pinned host memory: xp_plan writes the jobs, then `seq`. */
struct XpBox {
    XpJob             jobs[kXpSlots];
    int32_t           n;
    volatile uint32_t seq;
};
struct XpPools {
    uint64_t lo[kXpMaxPools], hi[kXpMaxPools], stride[kXpMaxPools];
    int32_t  n;
};
/* Restored records: [0] units copied, [1] of them the step routed; [2] units planned. */
struct XpStats {
    unsigned int copied, routed, planned, pad;
};

/* What the prefetch takes on the card for a model of `layers` routed layers whose largest expert unit
 * is `unit` bytes: the ring and the buffers beside it (Ctx::xp_open allocates exactly these), in the
 * 2 MiB grains the card's allocations come in -- the ring its own, the small buffers sharing theirs. */
inline int64_t xp_device_bytes(int64_t unit, int64_t layers) {
    constexpr int64_t kGrain = (int64_t)2 << 20;
    auto grains = [](int64_t b) { return (b + kGrain - 1) / kGrain * kGrain; };
    const int64_t ring  = (int64_t)kXpGroups * kXpSlots * unit;
    const int64_t small = (int64_t)sizeof(XpRec) * kXpGroups * kXpSlots + 4 * kXpGroups + (int64_t)sizeof(XpStats) +
                          4 * (int64_t)kXpGroups * kXpMaxRows * kXpMaxNe + layers * kXpMaxNe * kXpParts * 8 +
                          layers * kXpMaxNe * 4;
    return grains(ring) + grains(small);
}

}  /* namespace rad */

extern "C" int rad_xp_logits_launch(const uint16_t* h, const uint16_t* w, float* logits, uint32_t rows,
                                    uint32_t n_embd, uint32_t ne, void* stream);
extern "C" int rad_xp_plan_launch(const float* logits, uint32_t rows, uint32_t ne, uint32_t k,
                                  void** const* map, const int32_t* place, const int32_t* cnt,
                                  rad::XpPools pools, uint64_t ring, uint64_t ring_stride, rad::XpRec* rec,
                                  int32_t* nrec, rad::XpBox* box, uint32_t seq, rad::XpStats* stats,
                                  void* stream);
extern "C" int rad_xp_step_launch(const rad::XpRec* rrec, const int32_t* rn, const rad::XpRec* prec,
                                  const int32_t* pn, const uint32_t* done, uint32_t seq, void* stream);
