# KV streaming: host-authoritative KV with a resident VRAM window

The full context's KV cache lives in pinned host RAM; VRAM holds only the part attention is
about to read, and the VRAM that frees up goes to experts. `--max-model-len` sizes the RAM
side, `--kv-resident` the VRAM side.

The design is tied to **sparse selection**: streaming pays only if the bytes a step must read
are far smaller than the context. That holds for radiance's indexer-selected attention (the
QSA family and GLM-5.3's MLA block) and fails for a dense full-attention layer, where a step
reads the whole context and the link is the wall (§2). A dense model is refused by name.

Two views sit over one allocation: the **ring**, `n_slots` blocks deep and `n_layers` stripes,
addressed through a residency map; and the **stage**, the group's context deep, one stripe,
addressed by block id. The allocation is `max(ring, stage)`, never the sum (§3.1, §4.1), and a
pass's view follows from its phase and row shape, so it is in the pass key, not a runtime mode.

The operator's view is [GUIDE.md](GUIDE.md) §4.4.1. Section numbers here are load-bearing
(source comments cite them): §5.1 answers "§5 Phase 1" and "§5 Phase 4", §5.2 "§5 Phase 4",
§5.3 "§5 Phase 3".

---

## 1. The tier that exists, and why it is not the stream

The VRAM paged pool (`core/mem/kv.{h,cpp}`) is the only place a live sequence's KV ever lives.
The idle tiers (`core/mem/kvtier.*`, `core/runtime/tierexec.*`) copy a *finished* conversation's
blocks to pinned host RAM and back on a prefix-cache hit: a **hot cache, not a stream**, whose unit
is one published prefix entry (`plan_writeback`, `tierexec.h:22-25`), which fires on publish and
admission rather than inside a request (`engine.cpp:1214`, `:1292`), moves bytes between identical
layouts (`tierexec.cpp:326`, `:458`), and lives in `RAD_MEM_HOST_PINNED` memory (`hostarena.cpp:16`)
no kernel may dereference (`mem/kv.h:381`). Streaming needs tens of thousands of small, per-launch,
selection-driven copies a second with a residency map, so **do not extend the tier**; carry over
only its two rules (never block the host on the step path; order copies with events, not syncs —
`spec.md` §5.4).

---

## 2. What bounds a step's read

Per rank at `--tp 2`, fp8 KV (`docs/QSA.md:54-59`, `docs/GUIDE.md:718-725`).
Qwen3.8-Flash-Next (`arch/qwen4exp_fp8`) has 12 QSA layers plus the MTP head, so 13 layers bind
to `kv_attn`:

| group | per 4-token block | per token | at 262 144 tokens |
|---|---|---|---|
| `kv_attn` (fp8, 2 kv heads, head_dim 256, 13 layers) | 26.00 KiB | 6.50 KiB | 1.66 GiB |
| `kv_qsa_bkey` (bf16, 1 head, head_dim 16) | 3.25 KiB | 0.83 KiB | 208 MiB |

Decode is sparse at every context: `sparse` is forced on by `T == batch->n_seq`
(`rad_block_attn_gated_fp8.h:535-536`) and the selection keeps `topk = budget / ratio = 512` blocks
(`docs/QSA.md:36-38`), so a step reads 2048 tokens, 13 MiB across 13 layers, plus the partial tail
block — essentially free at 30 tok/s. The indexer is the catch: `qsa_score` walks **every complete
block** of the live history each step, bounded only by `nc` (`r4d_qsa_score_bf16.hip:51-60`) — 208
MiB a step at 262K, ~6 GB/s — so `kv_qsa_bkey` stays resident and cannot candidate for a ring
(`dense_read`, §3.1).

A dense family (Qwen3.8-27B, `arch/qwen35_*`) reads the whole context every step — 16 KiB a
token per rank at fp8, 120 GB/s at 30 tok/s at 262K — so the ring only subtracts from that read:
break-even is roughly `resident + 65k` tokens at 30 tok/s. A dense model declares no streamed
path, so `--kv-resident` is warned about and ignored by name and the group stays fully resident
(§4.5).

---

## 3. The contracts

### 3.1 One id space, two depths

The paged pool is dense and **block-id indexed**, layer-major:

```
layer i at  base + i * layer_stride,  layer_stride = n_blocks * layer_bytes_per_block
block b at  block_index * layer_bytes_per_block     (core/mem/kv.h:92-108)
```

`slot()` is `block_table[bi] * block_size + pos % block_size` (`core/sched/mem_iface.h:226-231`)
— no indirection anywhere, and the kernels bound only by the carve (`tierexec.h:122-123`). A
block id outside `[0, n_blocks)` corrupts the next layer, silently, with fluent text.

`KVGroupPlan` therefore carries two depths: `n_blocks`, the logical id space (the context),
which the manager, the refcounts, the block table, the prefix cache and every `id < n_blocks`
guard count; and `n_slots`, the blocks the device pool backs. `pool_blocks()` is
`n_slots ? n_slots : n_blocks`, `streamed()` is `paged() && n_slots > 0 && n_slots < n_blocks`
(`core/mem/kv.h:175-182`), and `layer_stride()` uses `n_slots`. Only *addressing* moves.

Every site that must know the split — the places where `n_blocks` was the address bound — is the
same list that carry the arithmetic today:
`layer_stride()`/`layer_bytes_per_block()` (`kv.h:106-108`), the refcount/free-list sized by the
carve (`kv.cpp:449-471`), the loan strip offset (`engine_bringup.cpp:3345-3350`), the kernel shape
and tier `GroupInfo` (`kvbind.cpp:36`, `tierexec.cpp:183`), `ceiling()`/`paged_split()` and
`VramFacts` (`kv.cpp:266`, `:300`, `:349`; `vram_budget.h:69-78`), and the `id < n_blocks` guards,
which stay valid because the *carve* is no longer their bound.

The pool is **one reservation of `max(ring, stage)`**, not the sum: the stage is one layer of
the context, so it sets the price and the ring fits inside it above ~24k of context. The count
is `kv_pool_bytes(plan, slots)`, the ceiling sums `kv_ceiling_bytes(plan, blocks, slots)`, and
`carve()` writes `n_blocks`, `n_slots` and `depth` together (`kvstream_apply`). A streamed plan
whose host copy or residency map is missing is refused with `RAD_E_INVAL` rather than handing
out ids the pool cannot address.

A streamed group takes its id space from the context
(`kv_context_blocks(block_size, max_ctx, max_seqs)`) instead of the pool split, so it stops
competing for VRAM; its device share is its resident window, fixed by `--kv-resident`.
`lendable()` is false for it (`kv.cpp:968-976`) and the tier must not gather it (§4.1).
Refusals, each by name so an operator tunes the right flag:

- **a resident window that does not fit the budget** is refused against `--kv-resident` before
  anything is split: a window that silently ate the split would leave a neighbour with zero
  blocks and be reported as a pool too small (`kvstream_window_fit`).
- **a streamed `RAD_KV_WINDOW` group**, because a window group already bounds its own history.
- **the dense-read floor.** A group a step reads over its whole live history (an indexer's
  block keys; `rad_kv_group_dense_read`) cannot stream, and while a sibling streams its carve
  is `max_seqs` full contexts, not the split's shared token count (`kvstream_dense`,
  `kvstream_dense_blocks`): the mirror beside it is sized for `max_seqs` sequences, so an
  indexer funded for one would make the advertised capacity a promise the engine cannot keep.
  With nothing streaming the floor is false — that configuration deliberately sizes the cache
  for one full-length sequence, and charging it there would take the experts' VRAM for capacity
  nothing uses.
- **the maps are charged before the pool is sized.** They are reserved from the KV pool before
  `carve()`, but the budget probe that sizes `vram_kv` has no pool to reserve them from;
  `kvstream_maps_charged()` takes them out of the paged split in both paths and `carve_into()`
  adds back what `alloc_stream()` already reserved.

A ring that covers the group's context (`ceil(max_seqs * max_ctx / block_size) <= ring_blocks`)
is **not a streamed group**: no mirror, no maps, no stage, no `kv_resolve`, a null
`RadKVStream`. The base engine's path exactly, decided at startup before the first token.

### 3.2 The host mirror

`HostArena` is slot-per-entry and `RAD_MEM_HOST_PINNED`, copy-only. Streaming needs a
**layer-major, block-indexed mirror in `RAD_MEM_HOST_MAPPED`**, so a device kernel reads it over
the link in place — the property the host pool, the activation arena, the token embedding and
the n-gram table already rely on (`core/mem/pools.cpp:218-227`, `ctx.cpp:164-180`,
`planner.cpp:938-955`).

The layout is the pool's: a layer's mirror is a dense run of `blocks` rows of `block_bytes`,
`host_layer_stride = blocks * block_bytes`, so layer `l`'s row for block `b` is
`host + l * host_layer_stride + b * block_bytes`. `block_bytes` is both the `kv_resolve` row
pitch and the length of every copy into the ring. The mirror is `n_blocks * bytes_per_block`,
zeroed at bring-up, so a block no store has written reads as a miss. It is kept **per rank**,
each with its own maps in its own pool (`kvstream_step_view`, `per_rank`). Every existing
reader of the mapped property is read-only; this is the first in-place *writer* of host memory
of a size that matters, so its coherence is pinned by a test, not assumed (§6.2).

### 3.3 The writer maintains both copies

`kv_store` and `qk_norm_rope_gate_kv_store` write the pool at `slot_mapping[i]` and nothing else
(`libr4d/r4d_rows.cpp:1913-1931`; `r4d_model_bf16.hip:505-516`). `kv_store_streamed`, and its
folded form `qk_norm_rope_gate_kv_store_streamed`, writes **the host mirror always** and the
resident slot **only when `res[block] >= 0`**, with the mirror and the map as extra operands. A
slot never goes stale, so nothing needs invalidating — not on a rejected draft, not on a
restore.

The residency test is in the writer, not only in the resolve, because at decode the block a
step writes is usually the block the previous step made resident: a writer that only wrote the
mirror would leave the ring's copy missing this step's tokens, and the resolve, finding the
block resident, would not copy it. The alternative — mirror only, and the resolve copies every
block it needs, tail page included — needs no predicate, but the resolve cannot tell which
entry is the partial page (the last is partial only when `(pos + 1) % ratio != 0`), so it would
re-copy every selected block or take a new operand to find the tail. The test is one compare
on a hot path, not a race: `res` is written by `kv_resolve` alone and only read here.

A layer's order is **store, then resolve, then attention**. The indexer cannot resolve: it runs
before the store, so a non-resident tail block would be copied short of the tokens just written
(`arch/common/rad_qsa.h`). The streamed store has rows for E4M3 and bf16 only; any other core
dtype is refused, and a packed cache (`fp8_v4`, `k6_v4`) declares no streamed store for its
layout and is warned about and ignored (§4.5).

### 3.4 The resolve

`qsa_select` **already dereferences the attention table**: it writes `sel[q][i] = tb[b]`, the
group's pool page id (`r4d_qsa_select_bf16.hip:254`, `:686`), and `attn_paged` uses each entry
as a raw pool page index with no bounds check (`r4d_attn_decode_h256_gqa6.hip:268-272`). `sel`
is the only place the sparse path names KV storage, so one op between select and attention
suffices.

`kv_resolve` (`libr4d/r4d_kv_resolve.hip`) reads `sel`, `nsel`, `res`, `slot_*`, `ctl`, the
host base and the pool, claims a free slot or a CLOCK victim, copies each miss and rewrites
`sel`. Those eight operands are in read order, the cache last as in `kv_store`; `M`, `blocks`,
`slots`, `sel_pitch` and `block_bytes` are parameters, and the unit is `block_bytes` rather
than `block_size` because the op moves bytes and never reads an element. The miss list is
scratch, not an operand: an operand would enter the arena planner, the recorded pass's ranges
and the tape key for bytes nothing outside one launch reads.

**The order is the contract, and it is a total order:**

1. bump the epoch (`ctl[0] + 1`);
2. one lookup per entry below `nsel[q]`: a hit takes the epoch and, the first time this call
   stamps the slot, bumps its two-bit heat counter (clamped at 3) — one bump a slot a call, not
   one an entry; a miss claims its map entry once (`-1 -> -2`), so the map is its own set;
3. the **miss list is in ascending block id**, never first-appearance order;
4. one victim per miss, in that order: identity (block `b` into slot `b` when that slot is
   free), then a free slot, then a CLOCK sweep where a slot stamped with this call's epoch is
   passed, a warm slot is halved and passed, and a cold slot is taken. The hand picks the
   victim — a per-visit decision, so no reduction, no barrier, no extra traffic; an exact LRU
   would need a collective minimum per miss;
5. re-point both maps, a placement counting as a use;
6. copy `block_bytes` from `host[b]` into the pool's slot, as bytes: a streamed block is
   bit-identical to the block the writer put in the mirror;
7. rewrite every entry below `nsel[q]` as its slot; the `-1` padding past it stays.

Step 3 is what makes the resolve **deterministic — the same blocks in the same slots for the
same inputs**. Without it a block claimed by row 3 could evict a different slot depending on
which thread got there first, and two implementations of one contract would produce different
`sel`. Ascending is a fact about the claim set, not about the map, so a device implementation
sorts the claim list itself (the kernel's `atomicCAS` is the distinct-claim count; the
compaction scan fixes the order) and never scans `blocks`, which would make a launch's cost
depend on the cache instead of on the selection. The kernel keeps the value it read per entry,
so `res` is read once a selection entry.

**Overflow.** The ring must hold the union of the call's selection (`hits + misses <= slots`).
When it cannot, unplaced claims revert to `-1`, `ctl[3]` latches, and those entries are rewritten to
**slot 0** — defined memory holding a real block's KV, so the step produces wrong numbers rather
than a fault, since a block id there would be an address past the ring that `attn_paged`
dereferences unchecked. The engine reads `ctl` (one 64-byte copy a group a step, behind the step's
last resolve and before the step-end event, so the step path never waits) and **fails every request
that was prefilling in the step** with the ring's size and the union that did not fit, then clears
the latch so the next request is measured on its own. No step is served from a table whose entries
do not name their blocks.

**State, concurrency, recording.** The epoch, hand, heat counters and counters live between
calls, so two launches must never overlap: the row is not in `rad_kernel_concurrent`'s stateful
prefix and the engine orders it. Stateful is not unrecordable: the op is a pure function of
device-resident state whose addresses and pitches never change after bring-up, so a recorded
pass replays it correctly. The map is per rank and the block ids are shared, so ranks resolve
the same shared ids through maps brought up identically and one rank's counters are every
rank's; each rank names its own view, and a follower that inherited its peer's maps took a
page-not-present fault on its first staged pass.

**The tape key.** `Ctx::pass_key` mixes `phase`, `n_tok`, `n_seq`, `n_spec`, `block_table` and its
pitch, `slot_mapping`, `seqused` and `kvstream_view_key` (`ctx.cpp:919-971`).
`kvstream_view_key` is the memory a kernel reads, not the address of the struct that carries it: the
batch's view is a copy in a rotating builder Out slot, so keying on `&view` made every streamed step
a pass the tape store had never seen, and the 24-tape LRU dropped each one long before replay.
Rewriting the maps' contents is fine; changing a pointer or a pitch is not. A view is fixed from
bring-up.

**Three implementations, one contract.** `libref/ref_kvstream.cpp` is the serial definition —
one pass over the entries, one sorted miss list, one second-chance walk, one memcpy a miss, one
rewrite — and is the oracle. `libavx/avx_kvstream.cpp` follows it line by line: the same
refusals in the same order, the same epoch, ascending order and victim from the same sweep, so
the same blocks land in the same slots; it vectorises the rewrite and the block copy only,
because the lookup, the claim gather and the sweep are serial by construction.
`libr4d/r4d_kv_resolve.hip` is the card's row, and `r4d_selftest --case kv_resolve` compares it
against the libref oracle case by case: hits, ascending misses, eviction, free-before-live,
second chance, saturation, overflow, 2052 claims, a whole block byte for byte, and the refusals.

### 3.5 Prefill: the union, and the stage

Above `qsa_exact_to = budget + ratio - 1 = 2051` (`qwen4exp_fp8.cpp:621-624`, `:1085`) prefill
is **per-query sparse**: every row of a chunk selects its own 512 blocks
(`rad_block_attn_gated_fp8.h:523-536`, `:582-594`; `docs/QSA.md:140-148`). What one launch's
union is (`scripts/kv-union.sh`, Flash-Next `--tp 2`, one request a context, ring raised until
no launch overflowed):

| prompt tokens | ring that held | resolves | largest union | largest 1-call misses | prefill |
|---|---|---|---|---|---|
| 3 958 | 2 048 blocks | 108 | **990** | 976 | 1.83 s |
| 16 048 | 4 096 | 180 | **4 000** | 1 024 | 6.10 s |
| 66 131 | 16 384 | 480 | **14 608** | 1 024 | 25.29 s |
| 133 331 | 32 768 | 864 | **23 429** | 1 024 | 52.93 s |

The union is **70-100% of the live history and grows with it** — it does not flatten, so no fixed
"a few times bigger" ring serves long prefill. Decode never exceeds 513 blocks, and a decode-sized
ring overflows in prefill: 2 560 blocks at 16k into a 2 048-block ring, 16 542 at 128k into 16 384.
The stage is the answer: **one layer's whole history** in an identity-layout buffer, so prefill needs
no selection awareness and its pool is a single layer deep. At 262K, fp8, TP2 that is 128 MiB a
layer, 1.66 GiB of PCIe per 5120-token chunk against ~8.7 s of compute — under 1%.

**Dense prefill is a separate hole.** A prompt below `qsa_exact_to` runs dense `attn_paged`,
which reads a block id straight out of the block table and never resolves, while
`kv_store_streamed` writes the ring only for resident blocks. A streamed group therefore cannot
serve a dense prefill at all: a 1 015-token prompt into a 1 024-block ring is refused rather
than producing wrong text. Any ring smaller than the context must refuse that case explicitly
until it is handled.

### 3.5.1 The aliased stage: verified, and the ordering rule

`tests/runtime_test.cpp` (`a_streamed_groups_layers_may_all_alias_one_pool_slot`: 13 layers, a
pool one `layer_stride` deep, every bound layer's slot 0, 187 checks) pins the premise.

**The operand resolution accepts it.** `RAD_OPK_KV` looks the absolute layer up in
`KVPoolBinding::layer_slot` and takes `off = layer * layer_stride`, bounded only by
`off + layer_stride <= bytes` (`issue.cpp:781-789`) — which a one-layer pool satisfies at `off = 0`
for all 13 layers. An *identity* table over the same pool is refused ("outside the ... bytes it was
allocated"), which is that bound and not a layer-distinctness check, and `bind_kv_pools` only
*produces* a positional table (`kvbind.cpp:21-25`). Both are tested.

**`layer_slot` cannot carry the aliasing.** The same table is the host mirror's row number:
`rad_kv_layer_index` (`issue.cpp:1223-1228`) is what `kv_stream_host` indexes with
(`rad_arch.h:1028-1032`), and `kv_store_streamed` and `kv_resolve` both take that pointer
(`rad_block_attn_gated_fp8.h:527`, `:633`), so mapping every layer to 0 would write all 13 layers'
KV into layer 0's mirror row. Aliasing is a **field on the binding**, the stage's depth (0 =
`n_layers`, 1 = every layer at offset 0), with `layer_slot` left alone.

**The carve is `n_layers` deep, not one.** `bytes_per_block` covers every bound layer by
declaration (`rad_builder.cpp:2094-2098`), so `pool_blocks() * bytes_per_block` is 13 layer
stripes (`kv.cpp:713`, `:1093`, `ceiling()` at `:266`). A one-layer group divides that
multiplication by its depth — one function, and the page definition does not move.

**Ordering: the hazard analysis does not protect it, and does not have to.** The KV operand is
external memory, not a `Program::buffers` entry, so the buffer plan's aliasing analysis never
sees it (`rad_bufplan.cpp` colours declared activations only). Two things do: **one stream, in
issue order** — store, stage and attention of a layer are separate dispatches on the batch's
compute stream, and a stream is a total order, so a layer's stage copy sits between the
previous layer's attention and this one's (`rad_block_attn_gated_fp8.h:620-626`: "it sits on
one stream: issue order is the ordering"); and **`haz_mark`, as a relaxation check only** — it
does see a KV operand (`haz_note` builds ranges from the issued tensors, and a KV operand
arrives as a real pointer whose span is exactly one layer's bytes,
`issue.cpp:1078-1091`), and with every layer aliased that is the *same* range for all of them,
so a stage declared as writing it conflicts with the previous layer's attention read and the
pair is never marked overlap (`ctx.cpp:1084-1093`). The worst case left is read-read between
two layers' attentions, harmless because a write lies between them — `kv_resolve` is not a
`rad_kernel_concurrent` row (`r4d_rows.cpp:8852-8862`) and a ranged copy is a dispatch no op
owns. Do not weaken it. What breaks it, in the order it is likely to happen:

- **the ride taken before all of L's attention launches:** a prefill layer's attention is
  issued in `qsa_gq_rows`-wide chunks, so a ride taken at the first chunk starts L+1's copy
  while the second chunk still reads the stage. Hence the seam after the loop, not in
  `resolve_sel`.
- **waiting after the layer's store:** the copy reads the mirror, which the store has not
  written yet, so it would put older bytes back over this chunk's tokens.
- **`pieces` and lane 1:** `mix_ar_piece` runs on lane 1 (`qwen4exp_fp8.cpp:1424-1428`) while
  lane 0 keeps issuing this layer's rows; a stage moved to lane 1 needs `rad_lane_join(1, 0)`
  before its reader.
- **a host-side stage:** a copy re-issued per play (`xprefetch.cpp:275-291`) rests its order on
  `cur_stream()` and the `kTapeHost` join (`ctx.cpp:1187-1199`), not on the layer's place in the
  pass. `Ctx::tape_kv_stage` re-runs the arm and the submit with the tape paused instead.
- **a sliced KV operand:** `issue.cpp:807` overwrites `shape[0]` with `o.rows`, narrowing an op's
  hazard range — true for an identity pool, a lie for an aliased one. No op does this today.
- **a binding that moves:** `KVPoolBinding` is not in the pass key, so a changed binding between two
  passes sharing a key would replay against the wrong addresses; depth and ring are fixed views
  from bring-up, and the phase is in the key (`b->phase`, `ctx.cpp:929`).

The rule beside the code: **the stage copy is issued on the compute stream, or a DMA list the
compute stream waits on, inside the layer, between the store and the attention.**

**What a staged pass is.** It needs no residency map and no resolve: `sel` already holds block
ids, and a context-deep identity stage addresses a block by its id, so the pass fills the stage
instead of resolving. That covers prefill rows *and* the decode rows a mixed chunk carries.
**A staged pass therefore addresses the pool by block id, so the store writes the device copy on
that pass too**: the "ring" operand `kv_store_streamed` writes *is* the stage view (same op, same
operands, the pass's binding), and its resident-slot write is redundant but in bounds and
harmless — which is why there is no third store variant. The ring then drops back to what decode
needs (`max_seqs x (topk + 1)`).

**Volume, and the one way this path can be silently wrong.** The copy moves the mirror row up
to the highest block id the coming attention can read, not the whole row: `n_blocks *
block_bytes` is the context allocation. `stage_live_bytes` (`kvstream_bound_live`, once a
released batch, ranks parked, after the step's blocks are ensured) is an **upper bound** on the
ids the pass can name, and an underestimate is not a crash — the attention would read bytes the
copy never moved and produce fluent wrong text. It cannot be too small: every id a block table
can hold came out of the group's allocator and is held; `KVManager::held_top_blocks` is one
bucket past the highest held block, updated on every acquisition and on the last release; and
the stage is identity-addressed from block 0, so `[0, bound)` covers every id below it whether
or not ids are dense. Ascending allocation is **not** assumed — a prefix-cache-retained block or
a restore's `reserve_specific_block` can leave a live block above a free one — hence the
allocator's high-water mark and not `ceil(end / block_size)`. It costs the group's whole held
set; a group with nothing held stages the whole row, which is always correct.

### 3.6 The ABI surface

`RadKVStream` (`abi/rad_runtime.h:46-77`) is one group's fixed view: the pinned mirror, the
per-layer residency maps (`res`: block -> slot or -1; `slot_block`: slot -> block or -1;
`slot_state`: epoch and heat; one slice a layer), the shared control block, `block_bytes`,
`host_layer_stride`, the ring's `slots` and `host_bytes`. Nothing in it changes address for the
engine's life, which lets a recorded pass replay a resolve. Resident
per layer, not per group, because the resolve copies one layer's bytes.

`RadKVGroupBatch` gains one trailing, zero-defaulted `const RadKVStream* stream` (`:79-97`).
Null means fully resident: no ring, no map, no host copy. A pointer rather than fields because
the data is per group and fixed at bring-up while the batch is rebuilt every step for every
rank; one trailing pointer keeps every field a non-streaming plugin reads at its old offset.
The build side: `RadBuildCtx::kv_resident_blocks` (`--kv-resident`, in blocks, so the builder
needs no block size) and `RadBuildCtx::kv_stream_ok` (the architecture's capability, carried to
the declare); `RadArchProbe::kv_stream_ok` states it once (`abi/rad_builder.h:1381-1387`).

ABI history: `RAD_ABI_VERSION 26` adds the batch's trailing pointer, `kv_resident_blocks` and
`kv_stream_ok` — every earlier field keeps its offset, the structs grow, so out-of-tree plugins
must be rebuilt. `kv_stream_ok` reached `RadArchProbe` in 28; 29 moved `control_vector_ok` to
80 because upstream had taken the offset `kv_stream_ok` used, and the version is `29u` today
(`abi/rad_abi.h:20-43`). A field, a retype or a reorder needs a bump; an op or a function does
not — `rad_kv_stage_layer` (`abi/rad_runtime.h:768`) and
`rad_kv_stage_wait`/`rad_kv_stage_next` (`arch/common/rad_arch.h`, with the inline
`kv_stage_*` wrappers an arch plugin calls) move no version.

### 3.7 Flags, report, metrics

Flags are declared once in `core/config.cpp:45` and shown in `kShow[]` at `:989`, with sync tests
(`config_test.cpp:396-560`, `diag_test.cpp:534-552`) that fail by name when either half is missing.
`--kv-resident N` is the whole switch. The KV report (`kv.cpp:509-531`) gains the streamed group's
two-tier line, and the receipt names the ring, the stage, the host tier and, when nothing streams,
which group's context the flag covers. Metrics: `radiance:kv_resident_blocks`,
`radiance:kv_stream_misses_total`, `radiance:kv_stream_bytes_total` (`metrics.cpp:267-280`),
`kv_util` unchanged. The per-request line carries cumulative resolves, lookups, misses, the ring's
depth and the measured maxima (`ctl[10..12]`).

### 3.8 The host-RAM budget and the gate

The mirror is the largest host allocation after the experts' pool: up to 4.69 GiB a rank at the
four-sequence target, 9.38 GiB over two. `kvstream_host_gate` checks it once, before any rank takes
it, against everything still to be pinned after it (the prefix cache's host tier, the activation
arenas, the checkpoint store when fixed) plus `--host-headroom-mib`, and a refusal names this tier
and the flag that started streaming, not `--host-pool-mib`: that pool holds experts and shrinking it
cannot make room here.

The prefix cache's host tier **stays** — it holds finished conversations while the mirror holds
live ones — and the mirror is budgeted alongside it:

    kv streaming: the host mirror for 2 streamed KV group(s) is up to 4.69 GiB a rank (9.38 GiB
    over 2) ... and this wants 9.38 GiB with 4.00 GiB to spare (--host-headroom-mib)

At the target the gate had 8.2 GiB spare (16.18 GiB of `MemAvailable`, less 4.02 GiB still to
allocate, less the 4.00 GiB headroom), so that configuration is refused, short by ~1.2 GiB; with
`--prefix-cache-mib 2048` it serves, leaving 4.48 GiB `MemAvailable` and 1.07 GiB `MemFree`. The
lever is that flag — lowering the prefix tier, never deleting it. The resident indexer is sized
from the same `vram_kv` the streamed group no longer spends: **51 280 blocks = 205 120 tokens**,
above the 51 200 / 204 800 one `--max-model-len` conversation needs, so the per-conversation cap is
the advertised 204 800.

### 3.9 What it costs

Two floors bound `--kv-resident`:

- **Correctness floor**: the largest block union one decode launch selects, 513 blocks (2052
  tokens) at 16k, 64k and 160k of history. A ring under it is **raised to it** at startup, with
  one warning naming both numbers, because a request under it cannot be served from the selection
  table. Under speculation it is `max_seqs x (1 + n_spec) x 513` — **16 416 tokens at 2 sequences
  and spec 3** — so an 8 192-token request under speculation becomes a 16k ring, and any "8k"
  label on such an arm is wrong.
- **Performance floor**: the 24 576 default. The reference implementation measured
  10 x 2048 = 20 480 cells as the point a smaller ring starts costing, and the decode union is
  four times smaller, so sizing from the union alone lands under it: at 64k of history 24 576 runs
  **37.9 tok/s with 81 908 misses**, 8 192 runs **37.1 with 143 071**, and a ring larger than the
  history buys nothing (49 152 at 160k: 36.97 against 24 576's 36.94, 15% fewer misses). Only the
  overflow check is a hard requirement. **Run 32 768** (8 192 blocks): above the decode selection
  and well under a 64k context. Starvation is done by the context, never by shrinking the ring —
  a shorter ring is a configuration the engine raises or that no operator deploys.

A marginal miss is **~0.72-0.83 us** and `kv_resolve`'s per-call cost is **flat across a 6x range
of slot counts**, so ring size is nearly free; the gap is the miss copies. A decode launch body is
**~4.0 us of device time** (~48 us of a 27 ms step, ~0.2%), and the twelve cannot be merged (L's
resolve follows L's store and precedes L's attention), so streaming costs **~3% of decode at
`--tp 2` and 0% at `--tp 1`**. Placement is the lever that pays: putting block `b` in slot `b`
when that slot is free makes the attention's 513-block gather walk memory in order (**+2.03%** at
16k, identical misses and bytes; at depth the ids exceed the slot space and the first-free-then-
CLOCK path applies, +0.26%, inside the spread). With it, streaming at 16k is **-3.0% against
resident** (39.92 against 38.72 tok/s). Do not warm the ring: the fill is cheap (~1.6 us a cold
miss against 0.83 warm), but filling it from the last prefill chunk made decode 3.8% slower
(36.7 against 38.25 tok/s at 16k, identical bytes) — slot order, not the copies.

**Capacity is bought in host RAM.** Measured at `--kv-resident 24576` (6 144 blocks of 24.00 KiB,
one block being 4 tokens over 12 layers), per rank, Flash-Next fp8 on two ranks,
`--max-num-seqs 2 --max-model-len 204800` (409 600 tokens retained) and the four-sequence target
(819 200 retained):

| configuration | VRAM KV per card | host RAM per rank |
|---|---|---|
| streaming off, 2 x 204800 | 1.43 GiB `vram_kv`, 1.0 session retained | - |
| streaming off, 4 x 204800 | 1.53 GiB `vram_kv`, 204 648 tokens of 819 200 (25%, 1.0 session) | - |
| streamed, 2 x 204800 | ring 144 MiB + stage 200 MiB (pool = max = 200 MiB) + indexer 300 MiB -> `vram_kv` **618 MiB** | 2.34 GiB |
| streamed, 4 x 204800 (819 200 retained) | ring 144 + stage 400 (pool = max) + indexer 600 -> `vram_kv` **1 232 MiB**, four conversations funded | 4.69 GiB |

The card side is not constant, and it is not the ring that sets it: the stage is one layer of the
whole retained context, 512 B per retained token per rank (200 MiB at 2 x 204800, 400 MiB at the
target, against a ring of 144 MiB), growing at 1/12th the rate of the cache; the mirror grows at
**6.00 KiB a token per rank**, ~600 MiB per 100k tokens. Per card, once the ring fits the budget, a
`vram_kv` of B MiB retains about `2048 x B` tokens (375 MiB → **~768k tokens**, 750 MiB ->
**~1.54M**), plus the ~32 MiB of fixed costs that also scale (the maps, ~21.6 MiB at 4 x 204 800,
and the step slab).

Three prices per token of retained capacity: **mirror 6.00 KiB of host RAM** (`kv_attn` only),
**indexer 0.83 KiB of VRAM** (`kv_qsa_bkey`, read densely every step), **stage 0.50 KiB of VRAM**
(the pool, and the real card-side price). Only the ring does not scale with capacity. At the
target four conversations run concurrently (four requests of 204 107 tokens, TTFTs 204 / 308 /
412 / 413 s), and funding the indexer for four rather than one cost **449.77 MiB** from the expert
slab (11.06 -> 10.62 GiB, 94% -> 90%, 374 pinned slots a rank at 1.204 MiB), which the ring could
not have paid since the pool is the stage.

**Prefill comes out above the off arm**, because the pool that streaming no longer needs goes to
the expert slab and at `--deterministic` a non-resident expert is read in place from the pinned host
pool over PCIe: at 2 x 204800 `vram_kv` falls 1.43 GiB -> 467 MiB, the experts get 11.37 instead
of 10.40 GiB (9 623 instead of 8 795 units), and 17.64 instead of 18.61 GiB of expert weights
crosses the link per rank. That ~1 GiB, not the ring or the copy, is the 3-7% gain.
**The stage copy's volume, not its overlap, is what it costs**: a pass copies the context
allocation once a layer (2.34 GiB at 2 x 204800), linear in prompt length at a ~200x constant, and
at 29.6-34.7 GB/s that is the link — hence "overlap recovers a quarter to a third". Clamped to the
group's highest held block it moves **633 MiB at 16k (37.9x less)** and **44.9 GiB at 160k (4.5x
less)**, within 0.3% of the live KV restaged, and prefill is **above** the off arm at every depth
(+5.4% / +3.2% / +2.8%). Decode's misses move 88.4 MB (16k) to 168 MB (64k) a request.

Capacity is bought through `--max-num-seqs`, not a longer context: four conversations
(`--max-num-seqs 4 --max-model-len 204800`) stay inside the trained 262 144 window and need **no
rope change**; beyond it is YaRN territory (`docs/YARN.md`). Quality, measured separately:
perplexity 3.994 streaming off, 4.015 with a window covering the context, 4.014 with a fraction of
it, against a cold-start floor of 0.021 KL.

---

## 4. Architecture

Two modes over one host copy; layer-major addressing, block ids and `slot_mapping` are
untouched for every non-streamed group.

### 4.1 The module

The code is a module of its own, `core/kvstream/`; the existing files gain one call each and
nothing is moved.

| file | owns |
|---|---|
| `plan.h`, `.cpp` | the policy: whether a group is staged (`depth`), which view a pass uses, the sizes, the refusals (a window group, a non-E4M3 cache, a dense prefill into a streamed group). Pure arithmetic. |
| `bind.h`, `.cpp` | the layout: both views over a plan, `layer_slot` kept as the mirror's row, `kv_binding_offset`, the per-pass view choice, `kvstream_step_view`, `kvstream_view_key`. |
| `stage.h`, `.cpp` | the driver: per pass, per layer, the ranged copy of that layer's live blocks out of the mirror into the stage, on the DMA engine (§5.3); the pass-scoped "already staged" state and the tape note. |
| `mirror.h`, `.cpp` | the host tier: the mirror and the maps, their sizes (`kvstream_map_bytes`), life cycle (`cost`, `alloc`, `init`, `release`, `stats`) and the window cap (`cap_resident_tokens`). `KVManager` holds one `KVStreamStore` and delegates. |
| `carve.h`, `.cpp` | what a carve does with a streamed group: windows and the dense-read floor leaving the budget first, the window-fit refusal, the write of `n_blocks`/`n_slots`/`depth`, the maps' charge, the two-tier report line. |
| `bringup.h`, `.cpp` | the engine side: which group streams, the flag refusals (§4.5), the ring's ceiling, the host-RAM gate, the startup lines. |

`stage.cpp` is the only one that touches a device, with `rad_memcpy_ranges_async` — already what
the batch upload uses (`core/sched/batch.cpp:1044-1052`) — so no new kernel, no new libr4d row
and no new libref oracle for the stage. `mirror.cpp` fills the maps with
`rad_memcpy_async`/`rad_memset_async`.

What the module deliberately does not take over: **the block ids, refcounts, free lists and prefix
adopt** (a stage addresses every logical block identically, so it needs no second address space, and
`cached_groups()` is unchanged so a prefix hit adopts the same ids); **`kv_resolve`'s rows and the
oracle** (a staged pass does not resolve); **`TierExec`'s gather/scatter** (`kv_tier_cached` narrows
the list, because a gather of layer L from a one-layer pool would be "the last staged layer, 13
times", and the residency map restores a block on the next resolve); **`PrefillStager`/`Mover`**
(`lendable()` is false for a streamed group, and a 30-line driver issuing one ranged copy a layer is
smaller than teaching the expert stager KV geometry); **`qsa_select`** (`sel` already holds ids).

### 4.2 Core changes

1. **Split the carve.** `n_slots` beside `n_blocks`; `layer_stride()` uses `n_slots`; the manager
   keeps allocating ids in `[0, n_blocks)`, so the sites of §3.1 keep working.
2. **Host mirror.** A `RAD_MEM_HOST_MAPPED` buffer of `n_blocks * bytes_per_block` per streamed
   group, laid out like the device pool; budgeted as a host term and checked with `lockable_room`
   (`engine_bringup.cpp:319`, `:548`).
3. **Residency map.** A device `res: int32[n_blocks]` (-1 = not resident) plus a control block
   (epoch, clock hand, miss counters, overflow), appended to the group's batch and the ABI.
4. **Report, flag, metric.** A `kv streaming:` line beside the pool report, the flag in tokens, the
   three metrics; `kv_util` unchanged.

### 4.3 Device changes

1. **`kv_resolve`** (one libr4d kernel, one libavx reference, one libref oracle), as in §3.4:
   the reference design's resolve minus the K8V4/Q4 layouts radiance does not need.
2. **`kv_store_streamed`**: the host copy always, the slot only when `res[block] >= 0`, with the
   mirror and the map as extra operands; `qk_norm_rope_gate_kv_store_streamed` folds it into
   the q/k norm+rope+gate prologue.
3. **Layer-bulk stage**: a ranged copy of one layer's live blocks between the store and
   `attn_paged` of that layer. No new kernel.

### 4.4 Arch changes

The resolve is declared and issued in `arch/common/rad_block_attn_gated_fp8.h`, in the attention
block, after that layer's store (`store_cache`) and immediately before each sparse launch; the
indexer does not resolve (`arch/common/rad_qsa.h`). `rad_block_attn_gated_fp8.h:523-536` already
has the `sparse` predicate and the `reach` bound to hang the decision on, and above
`qsa_exact_to` the layer-bulk path takes over. The store is declared beside `kv_store` and
issued instead of it when the batch hands the group a stream. GLM-5.3's MLA block
(`rad_block_glm_mla.h`) declares the same pair and resolves at finish, after its indexer
selects.

### 4.5 The capability gate, and what a run without the flag is

Streaming is a property of the architecture before it is a choice of the operator. A plugin
answers it once with `RadArchProbe::kv_stream_ok`; the engine copies the answer into
`RadBuildCtx::kv_stream_ok` for every declare, so all layers of a build see the same table
entry. `qwen4exp_fp8` and `glm5next` set it. A model file states only what its attention can do;
whether a ring is carved is the operator's.

Three ops exist only to move KV between the host mirror and the ring: `kv_store_streamed`, its
folded form and `kv_resolve`. They are declared only when the architecture is capable **and**
`--kv-resident` asked for a ring. The conjunction is `kvstream_op_gated`, consulted from
`Builder::op_declared`, so the gate is one place, no model file knows whether the operator
asked, and every architecture's `rad_op_declared` falls through to the path it had. Fail-safe,
one sentence each and no refusal (nothing is allocated yet):

- a group that declares no `kv_resolve` or no streamed store is ignored: without a resolve `sel`
  still holds block ids while the attention reads them as ring slots, past a smaller pool — a fault
  or silent corruption; without a store nothing writes the mirror, so a miss reads zeros.
- a packed cache (`fp8_v4`, `k6_v4`) is refused the same way, naming the layout rather than the
  architecture: it is declared to the core as plain E4M3, the streamed store's rows are E4M3 and
  bf16 only, and the operator's answer is the cache layout, not the model.
- an architecture whose probe answers no is turned off at startup, naming it, rather than a ring no
  store can write being found after declare.
- a streamed store declared for another group's geometry does not enable this one, and a
  declaration recorded with no kernel in any domain is not a path.

What the no-flag guarantee is, and is not. Without the flag no streamed op is declared, no
runtime path changes and no budget changes: `kvstream_dense()` is false when nothing streams, so
no resident group is charged a floor of `max_seqs` contexts. It is **not** byte-identity with
the release before it: `qsa_select` gained its `nsel` output, so a model that declares a
selection declares its count too (`selected_count` in the GLM indexer, `qsa_nsel` in the QSA
family), i32, one per token — 16 KiB at `--max-num-batched-tokens 4096` — whether or not
anything streams. The graph is the one it had plus a buffer the unstreamed path never reads.

---

## 5. The stage copy's transport, and the ring's default

### 5.1 The ring's depth

`--kv-resident` is off by default. When it is used, size it at **32 768**, the performance floor
(§3.9), not at the decode union. The union (<= 513 blocks, 2052 tokens) is the validation
ceiling, and a flag under it is raised to it at startup with one warning; under speculation it
is `max_seqs x (1 + n_spec) x 513`.

`kv_pass_view` decides the view: a decode pass within `1 + n_spec` rows a sequence takes the
ring; every other pass — prefill, a mixed prefill+decode step, and a decode-classified pass
wider than a decode step can be (the serial drafter's history pass over a prefill chunk) —
takes the stage. The choice is visible to the tape without a new key term because `phase`,
`n_tok`, `n_seq` and `n_spec` are already in the pass key. Capacity is priced in host RAM, not
in the ring; a raised ring that does not fit is the budget's refusal, by name and size.

### 5.2 There is no streaming switch

`--kv-resident N` *is* the feature: absent (or 0) and nothing streams, present and the ring is N
tokens. There is no `--kv-streaming` to understand (it is refused by name), no `auto` to
second-guess, and no mode where an operator can ask for streaming that cannot help. A removed
`--kv-stream-threshold` is refused by name too: the base path for a live set that fits the ring
measured 2.6-2.9 % slower a step on the post-tape-fix tree (record §74), so the streamed path is
the cheap one at every depth. And when the ring covers the group's context it does not stream at
all (§3.1): a configuration whose context fits pays *nothing* — the same code path, not a cheap
one.

### 5.3 The transport

The copy of layer L+1 is one `rad_xp_list` on the card's prefetch DMA queue, marked on a ride
(`rad_ride`, `Ctx::xp_mark`) taken once every attention launch of layer L has been issued, so it
starts when L's attention is done with the aliased buffer and crosses the link under L's out
projection and L+1's projections, holding no CU slot. The compute stream waits for it through
an event the *list* completes: `rad_event_dma` gives a completion word that the list's fence
packet writes 0 to, and `rad_event_wait` is the barrier. The word is armed by a write of 1 in
stream order — the weight-table patch dispatch, which is also what the ride completes on —
because a host-side arm races a wait still queued for the previous layer, and an event record
cannot be used at all (a record completes by decrementing its signal to 0, exactly the value
the wait looks for).

Two consequences. **The wait is before the layer's store**: the copy reads the mirror, which the
store has not written yet, so a copy landing after it would put older bytes back over this
chunk's tokens; the seams are `rad_kv_stage_wait` immediately before `store_cache` and
`rad_kv_stage_next` after the attention loop. And **the list must not yield to a link hold**:
`rad_xp_list` takes a `flags` argument and this list passes `RAD_XP_NO_YIELD`, because the
hold's release would be behind the wait this stream takes on the copy — the deadlock
`core/device/device.h` names. A backend with no DMA queue (the host backend, a test) copies on
the compute stream in the same four-step order, and `Ctx::tape_kv_stage` re-runs the arm and the
submit with the tape paused, as `xprefetch.cpp` does.

### 5.4 What is not reused

The transport is the expert prefetch's; the policy is not. `XpRec`, `xp_plan` and `xp_step` patch
weight-table entries and have no analogue in KV, and the geometry is expert-sized (`kXpSlots 2`,
`kXpGroups 3`, `kXpMaxRows 16`), so the KV policy is written fresh in `core/kvstream/stage.*`.

**Experts speculate, KV waits.** A prefetch copy that lands late is unused, because the expert is
read over the link instead; a KV stage cannot be skipped, so it must be a copy the compute stream
waits on. They do not compete by design: the prefetch is decode-only (`kXpMaxRows 16`) and staging
is prefill-shaped. The alternatives, kept so a later change can be measured against them: the
one-layer aliased view (chosen); the same view carved from the arena's lent top (rejected — a KV
base inside another owner's pool, and the loan is step-granular while the carve is
bring-up-granular); a full `n_layers`-deep buffer lent to the experts between requests (costs
1.66 GiB at 262K *during* prefill, and `reclaim()` is a host wait); the whole context resident
during prefill (saves nothing: the group still carves `n_layers x context`).

---

## 6. Failure modes, risks, and the tests that hold them

| failure mode | what prevents it | test |
|---|---|---|
| a block id used as an address past the ring | `kv_resolve` rewrites every entry; overflow writes slot 0 and latches `ctl[3]`; the request is failed | `tests/ref_test.cpp`'s `kv_resolve_*`; `r4d_selftest --case kv_resolve` |
| an unbacked streamed plan handing out ids the pool cannot address | `kvstream_apply` refuses with `RAD_E_INVAL` | `a_streamed_group_is_refused_at_carve_without_its_host_copy` |
| a pool whose depth disagrees with its id space | `n_blocks`/`n_slots`/`depth` written by one function | `a_streamed_group_addresses_its_context_but_backs_only_the_resident_window` |
| a streamed window group, or a window that does not fit | refused by name against the flag | `a_streamed_window_group_is_refused_by_name` |
| a ring under one decode step's selection | raised at startup, one warning, both numbers | `a_ring_below_one_decode_step_s_selection_is_raised_to_the_ceiling` |
| a covering ring paying for streaming | `streams()` false: ordinary resident path, null `RadKVStream` | `a_ring_that_covers_the_context_streams_nothing_and_the_engine_pays_nothing`; `scripts/kv-parity.sh` |
| a wide decode-classified pass resolving a ring it cannot fit | `kv_pass_view` takes the stage | `a_decode_pass_wider_than_a_verify_stages_instead_of_resolving_the_ring` |
| a mirror nothing writes (a miss reads zeros) | the op is declared only with a store row for the group's geometry | `a_declared_streamed_store_with_no_kernel_is_not_a_streamed_path`; `a_packed_cache_warns_that_the_layout_has_no_writer_and_serves_non_streamed`; `a_resolve_declared_for_another_group_does_not_enable_streaming` |
| the flag honoured where no streamed path exists | the gate: warn and ignore | `a_model_whose_architecture_declares_no_streamed_ops_ignores_the_flag`; `a_streamed_op_exists_only_for_a_capable_architecture_and_a_ring` |
| a stage reading the previous layer's cache | the four-step order (§3.5.1) | `a_streamed_groups_layers_may_all_alias_one_pool_slot`; `scripts/kv-stage-parity-64k.sh` |
| a stage copying more than it must, or less than it may | the `held_top_blocks` upper bound | `scripts/kv-stage-parity-64k.sh` (64k prompt, 32 768-token ring) |
| maps that leave the pool short of the budget probe | `kvstream_maps_charged` in both paths | `a_streamed_groups_maps_are_charged_before_the_pool_is_sized` |
| the dense-read sibling funded for one sequence | the `kvstream_dense` floor | `a_dense_read_groups_floor_holds_every_sequence_while_a_sibling_streams` |
| a recorded pass replayed against a different view | `kvstream_view_key` by memory, not struct address; fixed pitches | `Ctx::pass_key`, with its static assertion on `RadKVGroupBatch`'s fields |
| a follower lane or a host re-issued copy breaking the order | the barriers in §3.5.1 and §5.3 | reviewed, not tested |

### 6.1 The link, the lanes, and the hold

At `--tp 2 --tp-wire exact` the all-reduce, the KV copies and the prefill expert stager share the
link. `rad_xp_list` copies run on the DMA engine, the one path that takes its share at full rate
(`RAD_COPY_YIELD`, `core/device/device.h:64-88`); a staging copy issued as an ordinary stream
memcpy would hold a CU slot and starve the kernels. With the correction the transport needed — the
list must not yield, because the compute stream waits on it — the stage competes as an equal, which
is why overlap recovers only a quarter to a third of its cost (§3.9). A stage on lane 1 needs
`rad_lane_join(1, 0)` before its reader.

### 6.2 The first in-place device write to mapped host memory

Every precedent is read-only, so the streamed store's write is held to libref's bytes before
anything depends on it: `r4d_selftest --case kv_store_streamed_fp8` puts the mirror in mapped
pinned host memory, runs the kernel against it and compares the bytes the *host* reads after
the sync; `--case kv_store_streamed_bf16` does the same for the bf16 plane;
`scripts/kv-stage-parity-64k.sh` exercises the same write in the served path.

### 6.3 Rank coverage

The map is per rank and the block ids are shared, so the leader decides residency and every rank
applies the same decision, the way `IdleTiers` splits planning from transfer (`tierexec.h:55-58`).
Exercised at `--tp 1` and `--tp 2` only, which is what the box has: `--tp 3`, `--tp 4`, the uneven
split (`vram_budget_rank_split`), the attention-zero ranks (`rad_kv_group_zero`) and the
`r4d_selftest` four-card cases are **untested**. The one real defect tp2 caught was the per-rank
rebasing of `RadKVStream` (§3.4), so exercise that path before claiming four ranks.

### 6.4 The draft model

The MTP head binds the same `kv_attn` group (`qwen4exp_fp8.cpp:656`, `:1331`) and its own K/V is a
window ring; whether the drafter reads the same residency map or gets its own needs deciding with
speculation on. A verify step issues one selection a row and takes the ring, not the stage, which is
why the ceiling is `max_seqs x (1 + n_spec) x 513` (§3.9) and why a decode-classified pass wider than
that must stage (§5.1).
