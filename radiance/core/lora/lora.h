/* lora.h -- runtime LoRA: which declared GEMMs an adapter lands in, and the bytes of one adapter for
 * one rank (research/radiance/lora/DESIGN.md §2, §5).
 *
 * TWO PURE STEPS, both host-only and both read nothing but the declared Program:
 *
 *   build_sites()   after declare: every LoRA-capable trunk GEMM whose weight a target names becomes
 *                   a Site -- the weight's per-rank geometry (K slice, output width, which rows of
 *                   which PEFT module land where in the GEMM's y) read off the facts the plugin
 *                   recorded on the weight: its shard mode, row parts, shard / part spans and its
 *                   name-map sources. Never a plugin's own split rule. One Site per GEMM op; the
 *                   Sites of one declared weight share one POOL (one slot layout per weight).
 *   load_adapter()  a PEFT directory -> one PackedSite per pool: exactly the bytes of one slot.
 *
 * Nothing here allocates device memory or touches a device; the caller declares the pools and copies.
 *
 * TARGETS ARE HF MODULE LEAF NAMES ("q_proj", "gate_proj", "in_proj_qkv", ...). A declared weight is
 * targeted when one of its name-map sources is `<prefix>.<leaf>.weight`, `leaf` in the set, and the
 * weight is not in a non-trunk scope (weight_targeted() states the rule; build_sites and
 * find_unsupported_uses use exactly that predicate, and the builder may call it mid-declare).
 */
#pragma once
#include "../rad_core.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace lora {

/* HF module LEAF names. A std::set so two callers that build it in different orders get the same
 * answer. Empty targets nothing. */
using TargetSet = std::set<std::string>;

/* ONE CONTIGUOUS RUN OF ONE PEFT MODULE'S ROWS INSIDE THIS SITE'S y, ON THIS RANK.
 * A module may need SEVERAL parts: in_proj_qkv covers three row parts of the delta net's fused
 * ssm_inz, and a rank holds a slice of EACH, so its B rows on that rank are three disjoint runs
 * landing at three y columns. Every part of one module carries the same t_off. */
struct SitePart {
    std::string module;            /* HF module path without ".weight", as the name map spells it */
    int64_t src_lo = 0, src_hi = 0;/* rows [lo,hi) of the module's full output (= rows of lora_B) */
    int64_t y_col = 0;             /* column of this site's per-rank y where row src_lo lands */
    int64_t t_off = 0;             /* first t column of the module's block = module slot * max_rank */
    /* ADDED: the weight's row part this run lies in (0 for a weight of one part). */
    int32_t row_part = 0;
};

/* ADDED: one name-map source of the site's weight, in concatenation order -- targeted or not,
 * because an untargeted source still occupies rows of the fused weight. */
struct SiteSource {
    std::string module;            /* source without ".weight" */
    bool        targeted = false;
    int32_t     slot = -1;         /* module slot in the site (t_off = slot * max_rank); -1 untargeted */
    /* The run of row parts [part_lo, part_hi) the source occupies, or -1/-1 when the declared
     * facts do not decide it (two sources over four parts) and the adapter's B rows must. */
    int32_t     part_lo = -1, part_hi = -1;
};

struct Site {
    uint32_t gemm = 0;                          /* the gemm op handle (anchor) */
    uint32_t lora_shrink = 0, lora_expand = 0;  /* appended op handles (stream G) */
    std::string weight;                         /* declared weight name */
    int     shard = 0;                          /* RAD_SHARD_ROW / COL / NONE of the base weight */
    int64_t K_r = 0, k_lo = 0, K_full = 0;      /* x columns on this rank; COL: A columns
                                                 * [k_lo, k_lo + K_r). K_full 0 = not recorded on
                                                 * the weight (a COL span): taken from lora_A. */
    int64_t N_r = 0;                            /* per-rank y width the gemm writes */
    int64_t R = 0;                              /* capacity: n_modules * max_rank (see report:
                                                 * DESIGN said parts.size() * max_rank) */
    std::vector<SitePart> parts;                /* empty while !parts_resolved */
    uint32_t pool_a = 0, pool_b = 0, pool_scale = 0;   /* PERSIST buffers; 0 here, the caller's */

    /* ---- ADDED */
    int32_t     pool = -1;           /* index of the weight in build_sites' per-weight pool list */
    uint32_t    w = 0;               /* the declared weight's handle */
    std::string op;                  /* the gemm's op name */
    int32_t     n_modules = 0;       /* targeted sources */
    int32_t     max_rank = 0;
    bool        parts_resolved = true;   /* false: the fused layout is decided by B rows at load */
    bool        res_unchecked = false;   /* gemm_nt_bias: its optional `res` operand is bound at
                                          * issue, not at declare -- stream G checks it there */
    std::vector<SiteSource> sources;
    /* Per row part g of the weight (P = max(1, n_row_parts) entries): the part's rows in the
     * container (0 = not recorded: an equal-part single span), this rank's rows [lo, hi) of it in
     * the part's own units, and the y column where row `lo` lands. */
    std::vector<int64_t> part_total, part_lo, part_hi, part_y;
};

/* One adapter, one POOL (= one declared weight), one rank -- exactly the bytes of one slot. */
struct PackedSite {
    std::vector<uint16_t> a;       /* [R, K_r] bf16, zero rows past the module's rank in each block */
    std::vector<uint16_t> b;       /* [N_r, R] bf16, zero outside the module's rows / rank block */
    std::vector<float>    scale;   /* [N_r]: the module's alpha/r (alpha/sqrt(r) rsLoRA) on its rows */
    /* ADDED */
    int32_t pool = -1;
    bool    empty = true;          /* no module of this adapter lands in this pool */
};

/* What an adapter is, independent of the name it is served under. */
struct AdapterInfo {
    struct Module {
        std::string name;          /* adapter's module path (base_model.model. stripped) */
        std::string base;          /* the base's spelling it matched on this rank ("" = absent here) */
        int64_t     rank = 0;      /* from the tensors (== rank_pattern / r) */
        double      alpha = 0.0;   /* alpha_pattern / lora_alpha */
        float       scale = 0.0f;  /* alpha / rank, or alpha / sqrt(rank) with rsLoRA */
        int32_t     pool = -1;     /* pool it was packed into on this rank, -1 = absent here */
    };
    std::string dir;
    std::string base_model;        /* base_model_name_or_path */
    std::string peft_type;         /* "LORA" */
    int64_t     r = 0;             /* config r */
    double      alpha = 0.0;       /* config lora_alpha */
    bool        rslora = false;
    std::vector<std::string> target_modules;                    /* config, as written */
    std::vector<std::pair<std::string, int64_t>> rank_pattern;  /* config order */
    std::vector<std::pair<std::string, double>>  alpha_pattern;
    std::vector<Module>      modules;     /* sorted by name */
    std::set<std::string>    leaves;      /* leaf names present in the tensors (targets present) */
    std::vector<std::string> absent;      /* modules no site on THIS rank took (see check_placed) */
    int64_t                  max_module_rank = 0;
    uint64_t                 content_hash = 0;   /* FNV-1a 64 over the config bytes and the tensors */
};

/* A targeted weight read by an op that is not a LoRA site. The caller refuses at startup. */
struct UnsupportedUse {
    std::string weight;            /* declared weight */
    std::string module;            /* the targeted source module(s), comma separated */
    std::string op;                /* the op that reads it */
    uint32_t    op_index = 0;
    std::string why;
};

/* ---------------------------------------------------------------- the one predicate
 * Is declared weight `w` a LoRA target? Works on a PARTIALLY declared Program (the builder calls
 * it from decl_op): it reads only the weight's own declaration and the name-map entry (or CONCAT
 * pieces) whose `declared` is the weight's source -- false when that does not exist yet.
 *
 * SCOPE RULE (never targets):
 *   - access RAD_ACCESS_VOCAB (embedding, lm_head) or RAD_ACCESS_PER_REQUEST (encoder towers);
 *   - a layer group at or above meta.n_layers (the drafter's / MTP head's layers, the rule
 *     KVGroupInfo::drafter uses) when meta.n_layers > 0;
 *   - a declared name in a non-trunk builder scope: first component draft / vision / audio / mtp;
 *   - a source whose first component is `mtp` or `draft`, or that passes through a vision/audio
 *     tower component (visual, vision_tower, vision_model, audio_tower, audio_model, embed_vision,
 *     embed_audio, multi_modal_projector, mm_projector), or whose leaf is embed_tokens / lm_head.
 * A routed expert IS targeted when its leaf is (so its moe op is reported, never skipped). */
bool weight_targeted(const Program& prog, rad_weight w, const TargetSet& targets);

/* Is this op a LoRA-capable GEMM over weight handle `w` (as its `b` operand)? `why` says why not.
 * gemm_nt; gemm_nt_bias without `act`; gemm_nt_q without `pb1`/`pb2`, with K == the weight's K,
 * over a weight whose known encoding has no `transform`; gemm_nt_q_bias (same encoding rule). */
bool op_is_site(const Program& prog, const OpInfo& op, rad_weight* b_out, std::string* why);

/* Every op that reads a targeted weight (any view of it) and is not a capable site over it. */
int find_unsupported_uses(const Program& prog, const TargetSet& targets,
                          std::vector<UnsupportedUse>* out);

/* The Sites of one rank's Program. `pool_weights` (optional) receives the per-weight pool list:
 * Site::pool indexes it, and its order is the order the pools' first sites appear in. RAD_OK or a
 * negative code with `err` naming the weight. Pool handles are left 0. */
int build_sites(const Program& prog, const TargetSet& targets, int max_rank,
                std::vector<Site>* out, std::string* err,
                std::vector<std::string>* pool_weights = nullptr);

/* Every leaf whose every targeted weight in `prog` would be a site: what "all LoRA-capable trunk
 * linears" means when no adapter names the targets. */
TargetSet all_capable_targets(const Program& prog);

/* One adapter for one rank: `packed` gets one PackedSite per pool (index = Site::pool), in the
 * slot layout of the pool's sites. Refusals return a negative code and name module, expected and
 * found. A module with no site on THIS rank is not refused here -- an attention-zero rank holds no
 * q/k/v/o -- it is listed in info->absent; check_placed() refuses one no rank took. */
int load_adapter(const std::string& dir, const std::vector<Site>& sites, int max_rank,
                 AdapterInfo* info, std::vector<PackedSite>* packed, std::string* err);

/* After load_adapter on every rank: refuse a module that no rank placed. With `prog`/`targets`
 * (any one rank's) the message says why it is not a site (base lacks it, scope, target list, op). */
int check_placed(const std::vector<AdapterInfo>& per_rank, std::string* err,
                 const Program* prog = nullptr, const TargetSet* targets = nullptr);

/* Why a module path is not a site on this Program, in one sentence ("" when it is one). */
std::string explain_module(const Program& prog, const TargetSet& targets, const std::string& module);

}  /* namespace lora */
}  /* namespace rad */
