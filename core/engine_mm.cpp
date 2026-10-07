/* engine_mm.cpp -- the encoder driver (spec §11).
 *
 * The scheduler holds back a request whose media is not encoded (Scheduler::media_ready) and hands
 * its items here. They are encoded BETWEEN steps, never inside one: a pass is issued with no step
 * in flight, read back, and the item marked done, so the next step's admission finds its rows in
 * host memory and the batch builder copies them in with the chunk that reaches them. Nothing on the
 * step path ever waits on the tower.
 *
 * A PASS IS WHOLE SEGMENTS. An image is one segment and a video one per temporal group, and the
 * encoder attends within a segment, so a pass may end between two segments of a video and never
 * inside one; the processor has already kept every segment within a pass. Several items share a
 * pass when they fit, which is what a request with a handful of small pictures wants.
 *
 * AN AUDIO CLIP IS ONE SEGMENT OF THE AUDIO ENCODER, whose passes are separate from the vision
 * tower's: a program declares one of each at most, and an item goes to the one its kind names.
 *
 * EVERY RANK RUNS THE WHOLE TOWER on its own card from its own copy of the patches. The rows are
 * needed on every rank anyway -- each rank embeds every token -- and running it everywhere costs the
 * encoder's compute once per card instead of a transfer of its output between them.
 */
#include "engine_priv.h"

#include "mm/processor.h"

#include <cstdio>
#include <cstdlib>

namespace rad {

int Engine::run_encodes() {
    std::vector<std::shared_ptr<mm::Item>> items;
    sched_.take_encodes(&items);
    if (items.empty()) return RAD_OK;

    /* A DIAGNOSTIC: RADIANCE_DEBUG_MM_DUMP=<dir> writes each item's patches (`.pix.bf16`, the
     * processor's output in merge-block order) and its rows (`.bf16`) there as raw bf16, named by
     * its content key -- the tower's input and output, for comparison against a reference encoder
     * fed the same input. */
    static const char* const dump_dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
    auto dump = [&](const mm::Item& it, const char* ext, const uint16_t* p, size_t n) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/mm_%016llx%016llx_%dx%dx%d%s", dump_dir,
                      (unsigned long long)it.key_hi, (unsigned long long)it.key_lo, it.grid_t,
                      it.grid_h, it.grid_w, ext);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(p, sizeof(uint16_t), n, f);
            std::fclose(f);
            RAD_INFO("encoder: wrote %s", path);
        }
    };

    Rank& r0 = *ranks_[0];
    const Program& prog = r0.program;
    for (auto& ip : items) {
        if (ip->encoded) continue;
        const RadEncoderDecl& e = ip->kind == mm::Kind::Audio ? prog.audio_encoder : prog.encoder;
        ip->embd.assign((size_t)(ip->n_rows() * e.n_embd), 0);
        if (dump_dir && *dump_dir) dump(*ip, ".pix.bf16", ip->pixels.data(), ip->pixels.size());
    }

    /* THE PASSES, ENCODER BY ENCODER: an audio clip goes through the audio encoder and a picture
     * or a video through the other, each with its own pass size and output buffer. */
    for (int which = 0; which < 2; ++which) {
        const bool audio = which == 1;
        const RadEncoderDecl& enc = audio ? prog.audio_encoder : prog.encoder;
        const std::string& enc_name = audio ? prog.audio_encoder_name : prog.encoder_name;
        std::vector<mm::Item*> mine;
        for (auto& ip : items)
            if (!ip->encoded && (ip->kind == mm::Kind::Audio) == audio) mine.push_back(ip.get());
        if (mine.empty()) continue;
        if (enc.modalities == 0) {
            RAD_ERR("encoder: a %s item reached the encoder and this program declares no %s "
                    "encoder", audio ? "audio" : "picture", audio ? "audio" : "image");
            return RAD_E_STATE;
        }
        const int64_t max_p = enc.max_patches;
        const int64_t n_embd = enc.n_embd;

        size_t i = 0;
        int32_t seg = 0;
        while (i < mine.size()) {
            std::vector<BatchBuilder::EncoderPart> parts;
            int64_t np = 0;
            while (i < mine.size()) {
                mm::Item& it = *mine[i];
                const int64_t seg_p = (int64_t)it.grid_h * it.grid_w;
                if (seg_p > max_p) {
                    RAD_ERR("encoder: a segment of %lld patches does not fit a pass of %lld",
                            (long long)seg_p, (long long)max_p);
                    return RAD_E_STATE;
                }
                const int64_t room = (max_p - np) / seg_p;
                const int32_t can = (int32_t)std::min<int64_t>(it.n_seg - seg, room);
                if (can <= 0) break;
                parts.push_back({ &it, seg, can });
                np += (int64_t)can * seg_p;
                seg += can;
                if (seg == it.n_seg) { ++i; seg = 0; }
            }
            if (parts.empty()) break;

            const RadBatch* b = sched_.build_encoder(parts, enc_step_++, enc);
            if (!b) return RAD_E_STATE;
            RAD_TRY(prepare_release(b));
            /* THE OUTPUT'S ADDRESS IS THE PASS'S LEVEL'S, read after prepare_release: a pass runs
             * at the arena's full plan (Engine::arena_step), and a buffer sits at another offset
             * in every level. An address taken at the level the last step left would read some
             * other buffer, or memory lent to the expert slab, in place of the rows. */
            void* out = r0.ctx.buf_ptr(enc.out);
            if (!out) {
                RAD_ERR("encoder '%s': its output buffer was declared but never planned",
                        enc_name.c_str());
                return RAD_E_STATE;
            }
            /* kIssueModel: the pass and nothing behind it -- no sampler, and no tiering tick,
             * which belongs to steps. */
            issue_what_.store(kIssueModel, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();
            barrier_->arrive_and_wait();
            const int st = step_status_.load(std::memory_order_acquire);
            if (st < 0) {
                RAD_ERR("encoder pass failed: %s -- the engine cannot continue", rad_strerror(st));
                return st;
            }
            /* The rows, in pass order: each part's segments are consecutive rows of its item. */
            int64_t row = 0;
            for (const BatchBuilder::EncoderPart& p : parts) {
                mm::Item& it = *const_cast<mm::Item*>(p.item);
                const int64_t rows = (int64_t)p.n_seg * it.seg_rows;
                const int64_t bytes = rows * n_embd * (int64_t)sizeof(uint16_t);
                RAD_TRY(rad_memcpy_async(it.embd.data() + (int64_t)p.seg0 * it.seg_rows * n_embd,
                                         static_cast<const char*>(out) +
                                             row * n_embd * (int64_t)sizeof(uint16_t),
                                         bytes, r0.ctx.stream()));
                row += rows;
            }
            /* Every rank's stream, not only rank 0's: the next pass rewrites the staging every
             * rank's copy reads, and the arena every rank's tower wrote. */
            for (auto& rk : ranks_) RAD_TRY(rad_stream_sync(rk->ctx.stream()));
        }
    }

    for (auto& ip : items) {
        mm::Item& it = *ip;
        if (it.encoded) continue;
        it.encoded = true;
        it.pixels.clear();
        it.pixels.shrink_to_fit();
        if (dump_dir && *dump_dir) dump(it, ".bf16", it.embd.data(), it.embd.size());
    }
    RAD_DEBUG("encoder: %zu item(s) encoded", items.size());
    return RAD_OK;
}

}  /* namespace rad */
