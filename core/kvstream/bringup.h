/* bringup.h -- the engine-side bring-up of a streamed group (docs/KV-STREAMING.md §4.1, §5 Phase 4).
 *
 * `--kv-resident N` is the whole switch: absent or 0 and nothing streams, and a ring that covers
 * the group's context streams nothing either, because such a group keeps the base engine's path
 * with no overhead at all. Which group streams, the two refusals (a model with no full group, a
 * cache with no streamed store row), the ring's ceiling, the cap that narrows every rank's pool,
 * the host-RAM gate for the mirrors and the startup report all live here, so
 * core/engine_bringup.cpp keeps one call at each seam. Nothing touches a device except fill(),
 * which creates and drains the bring-up stream of the maps the carve has just placed.
 */
#pragma once
#include "mem/kv.h"

#include <string>
#include <vector>

namespace rad {

struct Config;

/* The plan --kv-resident resolves to, as Engine::configure_kv carries it to the report. `group` is
 * -1 when nothing streams, which is every server without the flag and also a server whose ring
 * covers the group's context; the other fields then describe what was asked for and why nothing
 * streams, which is all the report needs. */
struct KVStreamSetup {
    int32_t     group = -1;            /* the group that streams; -1 when nothing does */
    std::string group_name;            /* the group that would have: named by the report */
    int64_t     asked_tokens = 0;      /* --kv-resident as given; 0 is no streaming at all */
    int64_t     raised_tokens = 0;     /* what the flag became when the ring's ceiling raised
                                        * it; 0 when it was not raised */
    int64_t     context_tokens = 0;    /* --max-model-len x --max-num-seqs */
    int64_t     ring_tokens = 0;       /* the ring that is carved, in tokens: the raised flag,
                                        * rounded to whole blocks */
    int64_t     ring_blocks = 0;       /* the ring's depth, in blocks */
    int64_t     block_size = 0;        /* the group's page, in tokens: the ring's block */
    int64_t     ctx_blocks = 0;        /* the group's id space, in blocks */
    int64_t     stage_blocks = 0;      /* the stage view's depth, in blocks; 0 when there is none */
};

/* The host tier the mirrors take, summed over the ranks. `host_bytes` is every rank's pinned
 * mirror; `max_rank_bytes` is the largest, which is the figure an operator acts on (an uneven
 * tensor-parallel split gives the ranks different shard sizes). */
struct KVStreamHostUse {
    KVStreamCost sum;
    int64_t      max_rank_bytes = 0;
};

/* Whether `op` must not be declared at all (it is not an error: the architecture takes its
 * non-streaming path). The ops in kStreamOnlyOps (bringup.cpp) exist only to
 * move KV between host memory and a ring, so a run that carves no ring must not have them: their
 * declaration alone changes the graph, the arena and which kernels the planner may choose, and an
 * operator who never asks for streaming must get exactly the build they had before it existed.
 * True only when the architecture says it streams (RadArchProbe::kv_stream_ok, carried in
 * RadBuildCtx::kv_stream_ok) AND --kv-resident asked for a ring. One binary serves both runs, so
 * this is where the two questions meet; every architecture's `rad_op_declared` falls through to
 * its non-streaming path by itself, and no model file carries the flag. */
bool kvstream_op_gated(const RadBuildCtx& ctx, const char* op);

/* Read --kv-resident, choose the group from rank 0's declarations, decide whether the ring covers
 * its context (nothing streams then), refuse a cache with no streamed store row, apply the ring's
 * ceiling (kvstream_ring, with `sel_pitch` read off the declared kv_resolve) and fill `out`.
 * RAD_OK with group -1 when nothing streams, which is the ordinary configuration and caps nothing.
 *
 * The ceiling raises the ring when the flag asks for a ring smaller than one decode step's
 * selection, and says so in one line: a warning, not a refusal, because the ring is a size the
 * engine computes and the operator asked for streaming. `out->asked_tokens` and
 * `out->raised_tokens` keep what the operator typed and what the ring became, for the report;
 * `--kv-resident` itself is left as given, so nothing else re-reads a number the operator did not
 * write.
 *
 * Fail-safe: when the architecture declares no streamed ops for the group the plan would stream
 * (kvstream_ops -- no kv_resolve, or no streamed store), the flag is ignored with one warning that
 * names the architecture and the missing op, and `out` is left exactly as if --kv-resident had not
 * been given: group -1, so no ring is capped, no mirror or residency map is taken, no stage view is
 * laid and the batch hands the arch no RadKVStream. That is not the raised ring above: an
 * architecture without a streamed path cannot be fixed by the operator, while a ring under the
 * ceiling is fixed here. A cache layout the streamed store has no writer for (--kv-cache-dtype
 * fp8_v4 / k6_v4, whose architecture declares no streamed store for it) takes the same
 * warn-and-ignore, with a sentence that names the layout rather than the architecture. Where the
 * ops are there, the E4M3/bf16 gate still applies: the streamed store has rows for those two only,
 * so a full group in any other core dtype would have no writer at all. */
int kvstream_configure(const Config& cfg, int64_t ctx, const std::vector<KVGroupInfo>& groups,
                       const Program& prog, KVStreamSetup* out);

/* Accumulate one rank's stream_cost() into the host tier's total. */
void kvstream_host_add(KVStreamHostUse* use, const KVManager& kv);

/* The host-RAM gate for the mirrors: a streamed group's mirror is `max_seqs x max_ctx` of its KV
 * per rank -- on Flash-Next at 262144 tokens and 2 sequences that is 3.3 GiB a rank, the largest
 * host allocation in the engine after the experts' pool. Checked once, before any rank takes it,
 * against everything still to be pinned after it (`still`: the prefix cache's host tier, the
 * activation arenas, the checkpoint store when its size is already fixed). The card side is the
 * ring, which carve() sizes from --kv-resident; a refusal names this tier and the flag that started
 * streaming, because telling an operator to lower --host-pool-mib would be about the wrong pool:
 * that one holds experts, and shrinking it cannot make room here. */
int kvstream_host_gate(const KVStreamHostUse& use, int n_ranks, int64_t still, int64_t headroom);

/* After the carve: fill the maps on a bring-up stream and drain it. */
int kvstream_fill(KVManager& kv);

/* The startup lines: the raised ring when the ceiling moved it (the flag is no longer what was
 * carved, and the report is what people read afterwards), which group streams and what it took on
 * both tiers, the stage, and -- when the flag was given but nothing streams -- which group's
 * context the ring covers. */
void kvstream_report(const KVStreamSetup& s, const KVStreamHostUse& host, int n_ranks);

}  /* namespace rad */
