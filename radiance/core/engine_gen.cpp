/* engine_gen.cpp -- the image generator's loop: the driver's passes (core/gen/driver.h) run on
 * the ranks like token steps, one at a time, each drained before the next.
 *
 * Draining is cheap: a pass is tens to thousands of milliseconds of device work, and the next
 * pass depends only on the driver's host bookkeeping, so the wait costs the microseconds of
 * building it. Issuing ahead would need a second staging region and a commit protocol.
 */
#include "engine_priv.h"
#include "gen/driver.h"

#include <chrono>
#include <thread>

namespace rad {

int Engine::gen_loop() {
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("image generator loop: could not bind rank 0's device");
    int step = 0, idle = 0;
    const rad_buf img = ranks_[0]->program.generator.image_out;
    /* An error ends the driver's requests, then the loop. */
    auto stop = [&](int st) {
        gen_->shutdown("the engine stopped on an error and is not serving", st);
        return fail_loop(st);
    };
    while (!stopping_.load(std::memory_order_acquire)) {
        const RadBatch* b = gen_->next(step);
        if (!b) {
            step_batch_.store(nullptr, std::memory_order_release);
            if (idle < 200) ++idle;
            std::this_thread::sleep_for(std::chrono::microseconds(idle * 50));
            continue;
        }
        idle = 0;
        if (const int ps = prepare_release(b); ps < 0) return stop(ps);
        issue_what_.store(kIssueModel, std::memory_order_release);
        step_batch_.store(b, std::memory_order_release);
        barrier_->arrive_and_wait();      /* release the ranks */
        barrier_->arrive_and_wait();      /* wait for them */
        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("generator pass %d failed: %s -- the engine cannot continue", step,
                    rad_strerror(st));
            return stop(st);
        }
        /* A decode pass's image, copied behind it on rank 0's stream; every rank holds the same
         * pixels. */
        if (const int64_t n = gen_->decode_bytes(); n > 0) {
            const void* src = ranks_[0]->ctx.buf_ptr(img);
            if (!src) {
                RAD_ERR("the generator's image buffer was declared but never planned");
                return stop(RAD_E_STATE);
            }
            st = rad_memcpy_async(gen_->host_image(), src, n, ranks_[0]->ctx.stream());
        }
        for (auto& rk : ranks_)
            if (st >= 0) st = rad_stream_sync(rk->ctx.stream());
        if (st < 0) {
            RAD_ERR("generator pass %d: the device failed: %s", step, rad_strerror(st));
            return stop(st);
        }
        dump_buffers(ranks_[0]->ctx, step);
        gen_->done();
        ++step;
    }
    return RAD_OK;
}

}  /* namespace rad */
