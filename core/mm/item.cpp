/* item.cpp -- the parts of Item the step path uses. The batch builder links only this unit from
 * core/mm, so a test that fakes the device layer does not pull in the processor and, through its
 * metadata reader, the real device layer. */
#include "mm/processor.h"

namespace rad {
namespace mm {

void Item::coords(int64_t first, int64_t count, int32_t* planes) const {
    /* An audio clip is 1-D (RadBatch::enc): frame index, whether the frame takes part in the
     * convolution, the clip's real frame count and its staged frame count. A clip staged in
     * chunks (AudioFamily::Qwen3ASR) reads each chunk's first `chunk_read` frames; the rest of the
     * chunk is alignment the encoder reads as zero. */
    if (kind == Kind::Audio) {
        for (int64_t k = 0; k < count; ++k) {
            const int64_t f = first + k;
            planes[k]             = (int32_t)f;
            planes[count + k]     = chunk_staged > 0 ? (f % chunk_staged < chunk_read ? 1 : 0)
                                                     : (f < valid ? 1 : 0);
            planes[2 * count + k] = valid;
            planes[3 * count + k] = grid_h;
        }
        return;
    }
    /* Merge-block order inverted: a patch's index within its segment is
     * ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col. */
    const int32_t merge = grid_h / rows_h;
    int32_t s = 0;
    while (s + 1 < n_seg && patch_at(s + 1) <= first) ++s;
    for (int64_t k = 0; k < count; ++k) {
        while (s + 1 < n_seg && patch_at(s + 1) <= first + k) ++s;
        const int64_t within = first + k - patch_at(s);
        const int64_t bw = seg_w(s) / merge;
        const int64_t in_col = within % merge;
        const int64_t in_row = (within / merge) % merge;
        const int64_t blk = within / (merge * merge);
        const int64_t row = (blk / bw) * merge + in_row;
        const int64_t col = (blk % bw) * merge + in_col;
        planes[k]             = (int32_t)row;
        planes[count + k]     = (int32_t)col;
        planes[2 * count + k] = seg_h(s);
        planes[3 * count + k] = seg_w(s);
    }
}

}  /* namespace mm */
}  /* namespace rad */
