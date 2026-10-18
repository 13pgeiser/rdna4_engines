/* engine_lora.cpp -- runtime LoRA on the engine side: which modules adapters may target (fixed
 * before declare), and loading an adapter into a slot of every rank's pools.
 *
 * A load packs the adapter for each rank (lora::load_adapter, host only), refuses a module no rank
 * placed, copies each rank's bytes into the slot on a stream of its own, waits, and only then
 * publishes the name. A request can name the adapter only after the publish, so no step reads a
 * slot while it is written. */
#include "engine_priv.h"
#include "lora/lora.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace rad {

namespace {

/* adapter_config.json's target_modules as leaf names. PEFT also accepts a regex string and
 * "all-linear"; neither names leaves this engine can check before declare, so both are refused
 * with the flag that does. */
int read_targets(const std::string& dir, std::set<std::string>* out, std::string* err) {
    const std::string path = dir + "/adapter_config.json";
    std::ifstream f(path);
    if (!f) { *err = "cannot read " + path; return RAD_E_NOTFOUND; }
    nlohmann::json j;
    try { f >> j; } catch (const std::exception& e) {
        *err = path + ": " + e.what();
        return RAD_E_INVAL;
    }
    const auto it = j.find("target_modules");
    if (it == j.end() || it->is_null()) { *err = path + " names no target_modules"; return RAD_E_INVAL; }
    if (it->is_string()) {
        *err = path + ": target_modules is '" + it->get<std::string>() +
               "', not a list of module names; give them with --lora-target-modules";
        return RAD_E_UNSUPPORTED;
    }
    if (!it->is_array()) { *err = path + ": target_modules is not a list"; return RAD_E_INVAL; }
    for (const auto& m : *it) {
        if (!m.is_string()) { *err = path + ": a target_modules entry is not a string"; return RAD_E_INVAL; }
        std::string s = m.get<std::string>();
        const size_t dot = s.rfind('.');
        if (dot != std::string::npos) s = s.substr(dot + 1);
        if (!s.empty()) out->insert(s);
    }
    return RAD_OK;
}

}  /* namespace */

int Engine::lora_targets() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    lora_targets_ = std::set<std::string>(cfg_.lora.targets.begin(), cfg_.lora.targets.end());
    if (lora_targets_.empty()) {
        for (const auto& [name, dir] : cfg_.lora.adapters) {
            std::string err;
            const int s = read_targets(dir, &lora_targets_, &err);
            if (s < 0) {
                RAD_ERR("lora: adapter '%s': %s", name.c_str(), err.c_str());
                return s;
            }
        }
    }
    lora_table_.reset(cfg_.lora.slots(), &lora_epoch_);
    return RAD_OK;
}

int Engine::lora_start() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    size_t n_sites = 0;
    for (const auto& r : ranks_) n_sites += r->program.lora_sites.size();
    if (n_sites == 0) {
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_ERR("lora: no GEMM of this model multiplies a weight of the targeted modules (%s)",
                t.c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* Every target is a module some rank carries adapters on. A leaf the model lacks, or holds
     * only outside the trunk (embedding, lm_head, a tower, the drafter), would target nothing. */
    {
        std::set<std::string> carried;
        for (const auto& r : ranks_)
            for (const lora::Site& site : r->lora_sites)
                for (const lora::SiteSource& src : site.sources)
                    if (src.targeted) carried.insert(src.module.substr(src.module.rfind('.') + 1));
        for (const std::string& x : lora_targets_)
            if (!carried.count(x)) {
                RAD_ERR("lora: the target module '%s' is no trunk linear of this model, so no "
                        "adapter could land on it; leave it out of --lora-target-modules (or of "
                        "the adapters' target_modules)", x.c_str());
                return RAD_E_UNSUPPORTED;
            }
    }
    /* The adapter ops run in their GEMM's domain and read the pools on the card, so a targeted
     * GEMM placed on the host could serve no adapter request. On the host backend every op and
     * the pools are in host memory. */
    if (!device_is_host())
        for (const auto& r : ranks_)
            for (const LoraIssueSite& site : r->program.lora_sites)
                if (r->ctx.op_domain((rad_op)site.gemm) != RAD_DOMAIN_DEVICE) {
                    RAD_ERR("lora: rank %d: the GEMM over '%s' is placed on the host, and its "
                            "adapter ops read pools on the card; keep the targeted weights in "
                            "VRAM or leave the module out of --lora-target-modules", r->index,
                            r->program.lora_pools[(size_t)site.pool].weight.c_str());
                    return RAD_E_UNSUPPORTED;
                }
    {
        const Program& P = ranks_[0]->program;
        int64_t slot_bytes = 0;
        for (const LoraPool& p : P.lora_pools) slot_bytes += 2 * p.R * p.K + 2 * p.N * p.R + 4 * p.N;
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_INFO("lora: %zu GEMMs over %zu weights carry adapters on rank 0 (%s); %d slots of rank "
                 "%d, %s a slot, %s on the card", P.lora_sites.size(), P.lora_pools.size(), t.c_str(),
                 cfg_.lora.slots(), cfg_.lora.max_rank, humanb(slot_bytes).c_str(),
                 humanb(slot_bytes * cfg_.lora.slots()).c_str());
    }
    for (const auto& [name, dir] : cfg_.lora.adapters) {
        std::string err;
        const int s = lora_load(name, dir, &err);
        if (s < 0) {
            RAD_ERR("lora: adapter '%s' (%s): %s", name.c_str(), dir.c_str(), err.c_str());
            return s;
        }
    }
    return RAD_OK;
}

int Engine::lora_load(const std::string& name, const std::string& dir, std::string* err) {
    std::lock_guard<std::mutex> lk(lora_mu_);
    if (!cfg_.lora.enabled()) { *err = "the engine was started without adapters"; return RAD_E_STATE; }
    if (name.empty()) { *err = "an adapter needs a name"; return RAD_E_INVAL; }
    /* A slot that is still cooling (lora/slots.h) is waited for here rather than answered with a
     * retry: the step loop goes round idle or busy, so the wait is milliseconds, and a hot swap
     * (unload, then load at once) would otherwise fail under traffic. Only other loads wait on
     * lora_mu_ meanwhile; requests and unloads use the table alone. */
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int slot = -1;
    for (;;) {
        bool cooling = false;
        const int s = lora_table_.reserve(name, &slot, &cooling, err);
        if (s >= 0) break;
        if (s != RAD_E_FULL || !cooling || stopping_.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= give_up)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    /* ---- pack, per rank (host only) */
    std::vector<lora::AdapterInfo> infos(ranks_.size());
    std::vector<std::vector<lora::PackedSite>> packed(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        const int s = lora::load_adapter(dir, ranks_[ri]->lora_sites, cfg_.lora.max_rank, &infos[ri],
                                         &packed[ri], err);
        if (s < 0) {
            if (ranks_.size() > 1) *err = fmt("rank %zu: ", ri) + *err;
            return s;
        }
    }
    {
        const int s = lora::check_placed(infos, err, &ranks_[0]->program, &lora_targets_);
        if (s < 0) return s;
    }

    /* ---- copy, per rank, on a stream of its own (the compute stream may be recording a pass) */
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        Rank& r = *ranks_[ri];
        const Program& P = r.program;
        if (packed[ri].size() != P.lora_pools.size()) {
            *err = fmt("rank %zu: the adapter was packed for %zu pools, the program has %zu", ri,
                       packed[ri].size(), P.lora_pools.size());
            return RAD_E_STATE;
        }
        if (rad_dev_set(r.device) < 0) { *err = fmt("rank %zu: cannot bind its device", ri); return RAD_E_DEVICE; }
        RadStream st{};
        if (rad_stream_create(&st, 0) < 0) { *err = fmt("rank %zu: no copy stream", ri); return RAD_E_DEVICE; }
        int rc = RAD_OK;
        for (size_t p = 0; p < P.lora_pools.size() && rc >= 0; ++p) {
            const LoraPool& lp = P.lora_pools[p];
            const lora::PackedSite& ps = packed[ri][p];
            const int64_t ab = 2 * lp.R * lp.K, bb = 2 * lp.N * lp.R, sb = 4 * lp.N;
            char* da = (char*)P.buffers[lp.a].ptr;
            char* db = (char*)P.buffers[lp.b].ptr;
            char* ds = (char*)P.buffers[lp.scale].ptr;
            if (!da || !db || !ds) { rc = RAD_E_STATE; *err = "a pool has no address"; break; }
            da += (int64_t)slot * ab; db += (int64_t)slot * bb; ds += (int64_t)slot * sb;
            /* A pool this adapter does not touch is zeroed, not left as it was: the bytes are what
             * the slot's last adapter or the arena held, and a scale of 0 times a NaN is a NaN. */
            if (ps.empty) {
                if (rad_memset_async(da, 0, ab, st) < 0 || rad_memset_async(db, 0, bb, st) < 0 ||
                    rad_memset_async(ds, 0, sb, st) < 0)
                    rc = RAD_E_DEVICE;
                continue;
            }
            if ((int64_t)ps.a.size() * 2 != ab || (int64_t)ps.b.size() * 2 != bb ||
                (int64_t)ps.scale.size() * 4 != sb) {
                *err = fmt("rank %zu: pool '%s' was packed as %zu/%zu/%zu elements, its slot holds "
                           "%lld/%lld/%lld", ri, lp.weight.c_str(), ps.a.size(), ps.b.size(),
                           ps.scale.size(), (long long)(ab / 2), (long long)(bb / 2), (long long)(sb / 4));
                rc = RAD_E_STATE;
                break;
            }
            if (rad_memcpy_async(da, ps.a.data(), ab, st) < 0 ||
                rad_memcpy_async(db, ps.b.data(), bb, st) < 0 ||
                rad_memcpy_async(ds, ps.scale.data(), sb, st) < 0)
                rc = RAD_E_DEVICE;
        }
        const int sy = rad_stream_sync(st);
        rad_stream_destroy(st);
        if (rc >= 0 && sy < 0) rc = RAD_E_DEVICE;
        if (rc < 0) {
            if (err->empty()) *err = fmt("rank %zu: the copy into slot %d failed", ri, slot);
            return rc;
        }
    }

    lora_table_.publish(slot, name, dir);
    const lora::AdapterInfo& a = infos[0];
    RAD_INFO("lora: '%s' in slot %d -- %zu modules, rank %lld, alpha %g%s, base %s", name.c_str(),
             slot, a.modules.size(), (long long)a.max_module_rank, a.alpha, a.rslora ? " (rsLoRA)" : "",
             a.base_model.empty() ? "(unnamed)" : a.base_model.c_str());
    return RAD_OK;
}

int Engine::lora_acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                         std::shared_ptr<void>* hold) {
    return lora_table_.acquire(name, slot, uid, hold);
}

int Engine::lora_unload(const std::string& name, std::string* err, int* in_flight) {
    return lora_table_.unload(name, in_flight, err);
}

std::vector<std::string> Engine::lora_names() const {
    return lora_table_.names();
}

}  /* namespace rad */
