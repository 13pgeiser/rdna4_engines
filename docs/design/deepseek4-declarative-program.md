# DeepSeek4 declarative layer program (#666 G9)

Status: single-lane decode cut over (`decode_step_body`, which the eager,
hipGraph and Redline capture routes share), and the existing EP handlers
(`Deepseek4Bindings::run_attend*` / `run_moe*`) run the same sublayer step
builders. Batched prefill and verify, MTP, DSpark and the hand-scheduled
heterogeneous gfx1100+gfx1151 route keep their own sequencing; they call the
same moved operations through thin binders and are the next G9 slices.

## Program

`crates/hipfire-arch-deepseek4/src/program.rs` builds one step list per layer.
The architecture crate binds resident weights, per-layer caches and scratch;
`hipfire_dispatch::pipeline::mla` owns every kernel sequence and route choice.

| Block | Steps |
|---|---|
| Attention sublayer | `HcPre` → `QLoraPrepare` → [`PackedProjections`] → `QLoraProject` → `KvJoint` → `ApplyTailRope` → (ratio > 0) `KvCompress` main → (ratio 4) [`PackedProjections`] → `KvCompress` indexer → `SparseIndex` → `SparseLatentAttention` → `HcPost` |
| FFN sublayer | `HcPre` → `ExpertFfn` (or `Clear` under `HIPFIRE_DEEPSEEK4_SKIP_FFN`) → `HcPost` |
| Head | `HcHead` → `PreparedProjection` (lm_head) |

`PackedProjections` appear only on the exact-gfx1201 MQ2R route, where the
projections that share the attention input run as one mixed-row launch and
their own steps skip the projection.

With HC ping-pong the attention mix writes `residual_streams_next` and the FFN
mix writes back, so a layer ends in the buffer it started in and needs no swap.
Without ping-pong each mix writes the `q` scratch and copies back
(`HcPostOp::writeback`). The per-sublayer EP handlers execute the same builders
on `state.residual_streams` and swap after a folded mix, as before.

Lazy allocation moved out of the operations: `program::ensure_layer` runs
every `ensure_*` binder of a layer (same sizes, dtypes and order as the old
first call), then `program::layer_program` binds immutable steps.
`state.pos_buf` is pointed at the layer's position slot during the ensure phase,
as `apply_tail_rope` did.

## Typed operations (`hipfire-dispatch`, `pipeline::mla`)

| Step | Operation | Notes |
|---|---|---|
| `HcPre` (`hc::HcPreOp`) | Manifold HC control (`c = α·(X·W) + base`), sigmoid / Sinkhorn finalize, input map | fused-control, fused-finalize, input-map and rsqrt-once routes as fields |
| `HcPost` (`hc::HcPostOp`) | `X' = comb·X + post·y` | branch: one output, or gfx1201 TP3/TP4 peer partials |
| `HcHead` (`hc::HcHeadOp`) | head HC combine, MTP capture of the full streams, output norm, FWHT | |
| `QLoraPrepare`, `QLoraProject` | attention RMSNorm (plain + rotated), `wq_a` → `q_norm` → `wq_b` → per-head norm | fused q_norm + rotate route |
| `KvJoint`, `ApplyTailRope` | joint latent KV + `kv_norm`; tail YaRN RoPE of Q and KV | YaRN parameters bound per layer |
| `KvCompress` (`compressor::KvCompressOp`) | strided compressor: projections + APE, ring write, softmax pool, norm, RoPE, F32/F16 store, overlap shift | device-slot route (decode/capture) and host-gated route (batched prefill fallback) |
| `SparseIndex` (`attention::SparseIndexOp`) | ratio-4 indexer: query + RoPE, head weights, relu scores, top-k | two-stage, owner-parallel (gfx942) and portable top-k |
| `SparseLatentAttention` | SWA ring write, joint softmax over window + gathered compressed rows + sink, inverse RoPE, grouped O-LoRA | F32 / F16 / block-cyclic sharded caches; grouped wo_a per arch |
| `ExpertFfn` (`expert_ffn::ExpertFfnOp`) | FFN norm, shared SwiGLU expert, `sqrt(softplus)` router, bias-aware or hash routed experts | serial, two-stream overlap and retained-split schedules; EP partial redirection |
| `PreparedProjection`, `PackedProjections` | one projection of an architecture-prepared (plain + rotated) input; several in one gfx1201 mixed-row launch | |

Model-owned exact-device kernels (the gfx942 grouped O-LoRA and parallel
indexer top-k) reach dispatch through `attention::LatentAttentionBackend`,
implemented by the loaded model's `Mq2rBackend`, as the bias-aware MoE already
does through `MoeBiasAwareMq2Backend`.

## Numerics contract

The move is bit-exact by construction: every operation launches the same
kernels, in the same order, with the same arguments and under the same branch
conditions as the pre-port hand functions. Developer captures
(`HIPFIRE_DS4_DENSE_ACT_DIR`, `HIPFIRE_DEEPSEEK4_STAGE_NORM`,
`HIPFIRE_DEEPSEEK4_DUMP_INDEXER`, `HIPFIRE_DS4_ROUTE_DUMP`) run the program one
step at a time and read their tensors after the step that produced them; each
captured tensor is unchanged by the rest of its step.

Removed: the hand layer loop and the #397 single-lane super-op path
(`decode_step_body_lowered`, `HIPFIRE_FORWARD_LOWERED` for DeepSeek4).

## Evidence

gfx1151 (halo), fixture `deepseek-v4-flash-0731.mq2r` (registry
`deepseek-v4-flash`, sha256 `cbf2bbcf…`, 82,191,359,851 bytes), against beta
`09540d8c4`:

- 2,300 committed positions (2,200 teacher-forced + 100 greedy): input and
  argmax tokens, full-logit hash and HC residual-stream hash at every
  position; every layer's SWA rings, main and indexer compressor caches, KV
  and score rings and `mtp_last_hidden` every 32 positions and at the end —
  all **bitwise identical**, eager and hipGraph. The run passes the
  128-token sliding window, the ratio-128 compressor commits and the 512-row
  indexer top-k saturation (position 2,048).
- Beta itself is deterministic on this harness (two eager runs and the
  hipGraph run hash identically).

Hardware-blocked on halo: the gfx1201 (packed projections, TP3/TP4 peer HC,
sharded caches), gfx942 and heterogeneous gfx1100+gfx1151 routes, and physical
EP. Fixture-missing on halo: the MTP sidecar.
