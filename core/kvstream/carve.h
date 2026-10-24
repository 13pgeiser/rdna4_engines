/* carve.h -- what a carve does with a streamed group (docs/KV-STREAMING.md §4.1, §4.2).
 *
 * carve() spends a pool against the plan, and three of its steps are streaming's: the resident
 * windows leave the budget before the other groups split the rest; after the split each paged
 * group's n_slots (and a streamed group's context and depth) is written, with a refusal if nothing
 * holds the blocks the pool does not back; and the pool report gains the line naming both tiers.
 * Those three live here, so mem/kv.cpp keeps one call at each seam. Pure arithmetic on the plan:
 * nothing here touches a device.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

struct KVGroupPlan;
class KVStreamStore;

/* Bytes a streamed group's residency maps take from the KV pool: exactly what alloc() reserves,
 * as KVStreamStore::cost() states it. They are part of the group's arrangement, not a side
 * allocation: the maps are reserved before carve() (KVStreamStore::alloc), so the budget that
 * sizes the pool must charge them. The budget probe has no pool and cannot reserve them, while
 * the carve has already taken them; charging them in paged_split() and adding them back to the
 * carve's budget is what makes both paths see the same bytes (docs/KV-STREAMING.md §3.9). */
int64_t kvstream_maps_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                              int64_t max_ctx, int64_t max_seqs);

/* A dense-read group (rad_kv_group_dense_read: an indexer's block keys) is walked over its whole
 * live history every decode step, so it cannot stream. While any group does stream, its pool must
 * hold every sequence's context: the host mirror beside it is sized for max_seqs sequences, and an
 * indexer funded for one would make the advertised capacity a promise the engine cannot keep. False
 * when nothing streams -- that configuration deliberately sizes the cache for one full-length
 * sequence, and charging the floor there would take the experts' VRAM for capacity nothing uses. */
bool kvstream_dense(const KVGroupPlan& p, const std::vector<KVGroupPlan>& plans,
                    const KVStreamStore& st);

/* Blocks a dense-read group must hold: max_seqs sequences, each ceil(max_ctx / block_size)
 * blocks long. 0 for a group kvstream_dense() does not select. */
int64_t kvstream_dense_blocks(const KVGroupPlan& p, int64_t max_ctx, int64_t max_seqs);

/* The bytes those blocks take from the pool, rounded as reserve() rounds them. */
int64_t kvstream_dense_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                               int64_t max_ctx, int64_t max_seqs);

/* Take the streamed groups' resident windows, their maps and the dense-read groups' full contexts
 * out of a paged split before the other groups compete for the pool: `remaining` loses their bytes
 * and `out[group]` becomes the ring's depth (or the dense-read group's context) in blocks. A streamed group does not compete for the pool: its device share is its resident
 * window, fixed by --kv-resident, and the blocks it addresses are its context, which lives in host
 * memory. The caller leaves those groups out of the split and reports this as their carved blocks
 * -- `carve_into` reads it as the pool's depth, while the group's context comes from the run
 * configuration. */
void kvstream_split_windows(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                            int64_t max_ctx, int64_t max_seqs, int64_t* remaining,
                            std::vector<int64_t>* out);

/* Refuse a resident window that does not fit `budget`, before anything is split. A streamed group's
 * window leaves the budget before the other groups split the rest, so a window that does not fit
 * silently leaves them zero blocks and they are refused for a pool too small -- an operator told to
 * raise --host-pool-mib when the term that is too large is --kv-resident, and who then tunes the
 * wrong one. Refuse the window itself instead. The fixed costs are their own term and were checked
 * by the caller. A window group is refused by name here too: it already bounds its own history. */
int kvstream_window_fit(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                        int64_t fixed, int64_t budget, int64_t max_ctx, int64_t max_seqs);

/* After the split: write every paged group's n_slots, which is its carve unless it streams, and for
 * a streamed group its n_blocks as the context and its depth. Refuse a streamed group whose host
 * copy or residency map is missing: an unbacked id is written past the layer stride. */
int kvstream_apply(std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                   int64_t max_ctx, int64_t max_seqs);

/* The report continuation for a streamed group: its pool is smaller than its context, so the
 * group's own line would read as a pool that holds the whole context. Both tiers in one line. */
std::string kvstream_report_line(const KVGroupPlan& p);

}  /* namespace rad */
