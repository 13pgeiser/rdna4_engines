/* lora/config.h -- what runtime LoRA was started with (--lora, --max-loras, --max-lora-rank,
 * --lora-target-modules). Read by declare (which sites and pools exist), by the adapter manager
 * (what may be loaded into them) and by nothing else.
 *
 * Off unless asked for: with no adapter named and no --max-loras, `enabled()` is false, declare
 * appends nothing, the program has no pool and no extra op, and no step carries an adapter slot. */
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace rad {

struct LoraConfig {
    /* NAME=PATH pairs to load at startup, in the order given. */
    std::vector<std::pair<std::string, std::string>> adapters;
    /* Adapter slots resident at once. 0 with adapters named means "as many as were named". */
    int max_loras = 0;
    /* The rank every slot is sized for; an adapter of lower rank is zero-padded into it. */
    int max_rank = 16;
    /* HF module leaf names (q_proj, gate_proj, in_proj_qkv, ...) the sites are built for, sorted
     * and unique. Empty: the union of the startup adapters' target_modules. */
    std::vector<std::string> targets;

    bool enabled() const { return max_loras > 0 || !adapters.empty(); }
    int  slots() const {
        const int named = (int)adapters.size();
        return max_loras > 0 ? (max_loras > named ? max_loras : named) : named;
    }
};

}  /* namespace rad */
