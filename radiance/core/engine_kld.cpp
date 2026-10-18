/* engine_kld.cpp -- the KL mode on the Engine: the driver that replaces the HTTP server, and the
 * copy of a step's scored rows off the cards. The arithmetic and files are core/kld.h's. */
#include "engine_priv.h"
#include "kld.h"

#include "device/device.h"
#include "server/sink.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace rad {

int Engine::run_kld() {
    /* Each rank's logits buffer must hold a step's tokens after the sampler's rows
     * (RadBuildCtx::max_out_rows); a plugin that ignored the field is refused here rather than
     * overrun. */
    const int64_t need = cfg_.max_tok + cfg_.max_seqs;
    const int64_t nv = meta_.n_vocab;
    kld_vocab0_.clear();
    kld_width_.clear();
    for (const auto& rk : ranks_) {
        const Program& p = rk->program;
        if (p.logits_buf == RAD_NULL_HANDLE || (size_t)p.logits_buf >= p.buffers.size()) {
            RAD_ERR("kld: rank %d declared no logits buffer", rk->index);
            return RAD_E_UNSUPPORTED;
        }
        const RadBufDecl& d = p.buffers[p.logits_buf].decl;
        if (d.dtype != RAD_F32 || d.rank != 2 || d.shape[0] < need) {
            RAD_ERR("kld: the '%s' plugin's logits buffer is %lld rows of %s; the KL mode needs "
                    "f32 rows for a whole step, %lld (it reads RadBuildCtx::max_out_rows)",
                    plugins_.arch_plugin.c_str(), (long long)d.shape[0],
                    d.dtype == RAD_F32 ? "f32" : "another dtype", (long long)need);
            return RAD_E_UNSUPPORTED;
        }
        /* The columns a rank owns: the sampler's split, which declare_sampler checked, or the
         * whole vocabulary on every rank (then read from rank 0 only). */
        const int64_t w = d.shape[1];
        if (w >= nv) {
            if (!kld_width_.empty()) continue;
            kld_vocab0_.push_back(0);
            kld_width_.push_back(nv);
            continue;
        }
        kld_vocab0_.push_back(rk->sampler.config().vocab_off);
        kld_width_.push_back(w);
    }
    int64_t cover = 0;
    for (int64_t w : kld_width_) cover += w;
    if (cover != nv) {
        RAD_ERR("kld: the ranks' logits columns add up to %lld of a %lld-token vocabulary",
                (long long)cover, (long long)nv);
        return RAD_E_STATE;
    }
    kld_host_.assign(kld_width_.size(), nullptr);
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        kld_host_[r] = static_cast<float*>(
            rad_dev_alloc(cfg_.max_tok * kld_width_[r] * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        if (!kld_host_[r]) {
            RAD_ERR("kld: no pinned host memory for a step's logits (%lld MiB a rank); lower "
                    "--max-num-batched-tokens", (long long)(cfg_.max_tok * kld_width_[r] * 4 >> 20));
            return RAD_E_NOMEM;
        }
    }

    auto kld = std::make_unique<Kld>();
    RAD_TRY(kld->open(cfg_, vocab_, nv, cfg_.model));
    /* Published before add_all takes the scheduler's lock; the step loop reads it only on steps
     * planned under that lock afterwards. */
    kld_ = std::move(kld);

    const size_t n = kld_->n_docs();
    /* A document the KV pool cannot hold even alone would wait at the head of the queue forever:
     * admission takes only a chunk the pool can hold, and nothing ever frees enough. Refused up
     * front, naming every such document. The pool is the least that a full paged group holds. */
    {
        const KVManager& kv = ranks_[0]->kv;
        int64_t pool = INT64_MAX;
        for (int32_t g = 0; g < kv.n_groups(); ++g) {
            const KVGroupPlan* p = kv.plan(g);
            if (p && p->paged() && p->window_blocks == 0)
                pool = std::min<int64_t>(pool, p->n_blocks * p->block_size);
        }
        int too_long = 0;
        for (size_t d = 0; d < n; ++d) {
            const int64_t len = (int64_t)kld_->tokens(d).size();
            if (kv.can_ever_fit(len)) continue;
            RAD_ERR("kld: document %zu is %lld tokens and the KV pool holds %lld, so it can never "
                    "be scored; give the cache more room or shorten the document", d,
                    (long long)len, (long long)pool);
            ++too_long;
        }
        if (too_long) return RAD_E_FULL;
    }

    /* --kld-lora: every document runs through one runtime LoRA adapter, held for the whole run --
     * how an adapter is scored against a reference recorded from its merged or HF counterpart. */
    int32_t  lora_slot = -1;
    uint64_t lora_uid = 0;
    std::shared_ptr<void> lora_hold;
    if (!cfg_.kld_lora.empty() &&
        lora_acquire(cfg_.kld_lora, &lora_slot, &lora_uid, &lora_hold) != RAD_OK) {
        RAD_ERR("kld: --kld-lora '%s' is not a loaded adapter (--lora NAME=PATH)", cfg_.kld_lora.c_str());
        return RAD_E_NOTFOUND;
    }
    if (lora_slot >= 0)
        RAD_INFO("kld: scoring through the runtime LoRA adapter '%s' (slot %d)", cfg_.kld_lora.c_str(),
                 lora_slot);

    /* Submit every document under one hold of the queue, so batching depends only on the corpus
     * and the flags and a rerun batches identically. */
    std::vector<std::unique_ptr<server::Sink>> sinks(n);
    std::vector<std::unique_ptr<Request>> reqs(n);
    for (size_t d = 0; d < n; ++d) {
        sinks[d] = std::make_unique<server::Sink>(64);
        auto r = std::make_unique<Request>();
        r->id         = d + 1;
        r->prompt     = kld_->tokens(d);
        r->max_tokens = 1;
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->score      = true;
        /* --kld-lora-sources: the other documents run as the base, in the same steps. */
        const bool via = cfg_.kld_lora_sources.empty() ||
                         kld_->source(d).rfind(cfg_.kld_lora_sources, 0) == 0;
        r->lora_slot  = via ? lora_slot : -1;
        r->lora_uid   = via ? lora_uid : 0;
        r->sink       = sinks[d].get();
        sinks[d]->id  = r->id;
        reqs[d] = std::move(r);
    }
    std::vector<int> rc(n, RAD_OK);
    sched_.add_all(reqs.data(), n, rc.data());
    for (size_t d = 0; d < n; ++d) {
        if (rc[d] >= 0) continue;
        RAD_ERR("kld: the scheduler refused document %zu (%zu tokens): %s", d,
                kld_->tokens(d).size(), rad_strerror(rc[d]));
        return rc[d];
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    for (size_t first = 0;;) {
        while (first < n && sinks[first]->done()) ++first;
        if (first == n) break;
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        sinks[first]->wait(200);
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(15)) {
            last = now;
            const double s = std::chrono::duration<double>(now - t0).count();
            RAD_INFO("kld: %lld / %lld positions, %zu / %zu documents, %.0f s",
                     (long long)kld_->rows_done(), (long long)kld_->n_rows(), first, n, s);
        }
    }
    int bad = 0;
    for (size_t d = 0; d < n; ++d) {
        const server::Finish f = sinks[d]->finish_reason();
        if (f == server::Finish::Error || f == server::Finish::Cancelled) {
            RAD_ERR("kld: document %zu did not finish: %s", d, rad_strerror(sinks[d]->status()));
            ++bad;
        }
        sched_.reap(sinks[d]->id);
    }
    if (bad) return RAD_E_STATE;
    RAD_INFO("kld: %zu documents in %.0f s, %lld preemption(s)", n,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
             (long long)sched_.metrics().preemptions);
    return kld_->finish();
}

/* Copy a step's scored rows off the cards. They follow the sampler's rows in the logits buffer in
 * the batch builder's entry order (core/sched/batch.cpp), so the walk below repeats that loop and
 * checks the count. Each rank copies on its own stream behind the step; KL mode does not
 * pipeline, so the next step waits for this. */
int Engine::score_rows(const RadBatch* b, const StepOut& so) {
    const int64_t n = b->n_out - so.n_rows;
    if (n <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    std::vector<KldSpan> spans;
    int64_t row = 0;
    for (const StepEntry& e : plan.e) {
        if (!(e.req && e.req->score && e.is_prefill && e.hidden_row < 0)) continue;
        spans.push_back({ e.req->id, (int64_t)e.ctx_len, (int64_t)e.n_tokens, row });
        row += e.n_tokens;
    }
    if (row != n || n > cfg_.max_tok) {
        RAD_ERR("kld: step %d names %lld scored rows and its plan %lld", b->step, (long long)n,
                (long long)row);
        return RAD_E_STATE;
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        Rank& rk = *ranks_[r];
        RAD_TRY(rad_dev_set(rk.device));
        const auto* src = static_cast<const float*>(rk.ctx.buf_ptr(rk.program.logits_buf));
        if (!src) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(kld_host_[r], src + (int64_t)so.n_rows * kld_width_[r],
                                 n * kld_width_[r] * (int64_t)sizeof(float), rk.ctx.stream()));
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r]->device));
        RAD_TRY(rad_stream_sync(ranks_[r]->ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    std::vector<KldShard> shards(kld_width_.size());
    for (size_t r = 0; r < shards.size(); ++r)
        shards[r] = { kld_host_[r], kld_vocab0_[r], kld_width_[r] };
    return kld_->score(spans, shards);
}

}  /* namespace rad */
