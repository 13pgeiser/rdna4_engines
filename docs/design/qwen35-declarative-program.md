# Qwen3.5 declarative layer program (#666 G8)

Status: slice 1 (single-GPU decode layers, HIPF-117) and slice 2 (batched
prefill/verify bodies, HIPF-118) are branch-implemented. They recut #795 from
current `beta`. #795 is the design reference only, and none of its bodies
are cherry-picked.

## Goal

Qwen3.5 (and 3.6/3.8, same architecture) declares each decoder layer as
typed `Step`s, as Gemma 4 and Qwen4 do. `hipfire-dispatch` owns the executor
bodies and every arch-, shape- and dtype-gated route choice. The architecture
crate keeps:

- weight, recurrent-state and KV ownership;
- the per-layer program shape (`src/qwen35/program.rs`): which op, bound to
  which tensors;
- the lifecycle: hipGraph, Redline, DFlash capture and the chunk loop.

## Slice 1: decode layers

Single-GPU decode (`forward_scratch_layers_lowered`) runs each layer as

    [DeltaNetMixer | GatedAttention, GatedMlp | Moe]

The mixers live in `hipfire-dispatch/src/pipeline/hybrid.rs`; the dense FFN
is the shared `GatedMlp` op on its `Product` ladder
(`hipfire-dispatch/src/pipeline/gated_mlp.rs`, HIPF-142, design note
`docs/design/gated-mlp-op.md` slice 4):

| Step | Stages | Route choices owned by dispatch |
|---|---|---|
| `DeltaNetMixer` | `project` → `prepare` → `recur` → `gated_norm` → `output` | QKVZA scalar prep, conv + QK norm, conv scalar prep, compact QK (2:1, 3:1), FP32/Q8/Q8-compact/Q4 state, fused gated norm + MQ rotate |
| `GatedAttention` | `project` → `attend` → `output` | fused FA prep, Q8 KV-write fold, TriAttention tap, compaction offset, fused gate + MQ-rotate epilogue |
| `GatedMlp` (`Product`) | `project` → `down` | `Residual`: MQ SiLU·rotate + SwiGLU-residual GEMV, HFQ residual GEMV, plain GEMV + add for the rotated formats (MFP4, MQ8, MQ4G128, ParoQ4G128, …), `memory.offload_exec=cpu` split. `Write` (dense-TP partial): SiLU·mul + plain GEMV |

MoE layers keep `moe_ffn_dispatch` (the sealed MoE program with its CPU
expert splice).

The same `Product` op serves the batched rows (prefill, DFlash verify, the
dense-TP prefill partial): `gated_mlp::execute_product_rows` takes it with a
`ProductRows` context (the chunk's `BatchScratch`, the WMMA flags,
`DflashFusionCtx`, `DenseBatchMath`) and runs slice 2's batched producer/GEMM
ladder, `Residual` as the residual epilogue and `Write(partial)` as the TP
`Partial` epilogue. It is the only entry to that ladder; the qwen35 binding is
`{DeltaNet,FullAttn}LayerWeights::ffn_rows`.

The executor bodies were moved from current beta's `Qwen35Bindings` super-op
handlers. Every gate exists once:

- `hybrid::{gdn_compact_qk_div, gated_norm_mq_rotate, fused_attention_prep,
  fused_attention_epilogue, qkvza_scalar_prep, conv_scalar_prep,
  conv_qknorm, gfx1151_radiowave, gfx1201_state_fusions}` read
  `HybridDims` (built by `program::hybrid_dims`).
- The qwen35 copies are deleted. `kv_cache_attention_dispatch` (hand arms,
  dense TP, the batched-prefill FA fallback) calls `hybrid::kv_attention`.
  `forward_slots/vmm/exact.rs` calls the hybrid predicates. The dense 27B
  shape predicate is `HybridDims::is_dense_27b_shape` (the widened prefill
  admission uses it too).
- `qkv/qkvza/gate_up_via_execute_steps` forward to `hybrid::normed_projection`.

`lower_variant` and `Qwen35Bindings` stay for the EP decode executor
(`forward_ep`) and the exact VMM slot route. Their handlers delegate to the
same hybrid stages, so each body exists once. Paths that keep their own
bodies, all older than this port: the PP decode loop
(`forward_scratch_layers_multi`), the dense TP partials, and the gfx1201
exact-VMM row twins (`exact.rs`), which must change in lockstep with
`DeltaNetMixerOp::prepare` and `GatedAttentionOp::attend`.

Left for later slices: the hand arms (`HIPFIRE_FORWARD_LOWERED=0`, DFlash
hidden-state capture, VL mrope, `memory.offload_exec=cpu` spills that own
host CPU weights) and the MTP head.
`hipfire_runtime::llama::weight_gemv_swiglu_residual` still serves those
hand arms and the other architectures. The `Product` ladder's residual down
is its dispatch twin: same kernels, the same calibration taps per token
(one for MQ, two for HFQ, three for the plain-GEMV fallback). In the
fallback arm, `GemvFamily::run_auto` prepares each rotation's own sign
tables where the runtime's `weight_gemv` does it explicitly.

The decode error text changed: the step program wraps a failing launch's
error once, where the super-op executor wrapped it twice (`… (hipError=0)`
instead of `… (hipError=0) (hipError=0)`). No caller classifies errors by
that text.

## Slice 2: batched prefill and verify bodies

The dense batched (rows > 1) bodies of prefill and speculative verify moved
from `qwen35/prefill.rs` into three dispatch modules, as the rows > 1
executors of the slice-1 ops:

| Module | Executor (slice-1 op) | Stages and library |
|---|---|---|
| `pipeline::batched` | `execute_swiglu_ffn_batched` (now behind `GatedMlp` `Product` rows, `gated_mlp::execute_product_rows`) | gate/up and down stages; the IU4/A8/FP8-stream activation producers, `dispatch_batched_gemm_epilogue`, the key-selected GEMM runners (`run_*_key`), `DenseBatchMath`/`verify_dense_math` with the explicit `*VerifyExact` keys, packed-MQ4 and F1-lite routes |
| `pipeline::batched_attention` | `batch_chunk_full_attn_attn` (`GatedAttention`) | input projection, prepare (fused FA prep, TriAttention tap), attend (tile attend, ragged FP8Q packets, multirow Q8, gfx1100 split verify, A4 epilogue), output projection |
| `pipeline::batched_deltanet` | `batch_chunk_delta_net_attn` (`DeltaNetMixer`) | qkvza projection, pre-GDN (conv, gates, DFlash tape copies), sequential / chunk-scan / tree recurrence, gated norm and output projection |

The bodies moved verbatim from current beta, so every beta edit came along:
`DenseBatchMath` and the PM FP8 verify scope, `ChainVerifySplit` and the
gfx1100 split verify, IU4 verify, the FP8-stream family opt-outs, the gfx1201
lane-batched verify and the gfx1100/gfx1101 campaign gates. The two
byte-identical dense FFN bodies (DeltaNet and full-attention layers) are one
executor now; they differed only in their error text.

The arch crate binds field-compatible views in `program.rs`
(`PrefillBatchScratch::view`, `{DeltaNet,FullAttn}LayerWeights::{mixer_view,
ffn_view}`, `kv_view`, `flash_scratch`, `deltanet_state_view`,
`gdn_tape_view`) and keeps the chunk loop (`forward_batch_chunk_impl`), the
widened-prefill planner, embedding, final logits and the multi-request verify
lifecycle (`prefill_multi.rs`, which keeps its per-request contexts and calls
the moved stages directly).

Verify context: `DflashFusionCtx` (`Off`, `ChainVerify`, `ChainVerifySplit`)
and `TreeVerifyCtx`, `BatchSemantics`, `valid_lane_mask` and
`WIDENED_COMMIT_ROWS` are dispatch types now; the moved bodies take the enum,
never a bool. `StateQuant` is `hipfire_dispatch::ops::delta_net::StateQuant`.
The arch crate re-exports these at their old paths.

Every gate exists once: the moved fusion and route predicates
(`q8_multirow_attn_admitted`, `fa_pertoken_min_ctx`, `gfx12_fa_prep_admitted`,
`dflash_fa_batch_fusion_admitted`, `mq_f16_projection_fast_route`,
`s4_residual_fast`, the FP8-stream admissions, the gfx1100 split-verify
model) live in dispatch; the arch keeps one-line Qwen adapters for its public
`gfx1100_split_verify_{min_ctx,admitted}`. The fused qkv/qkvza key selectors
moved to `families::fused_qkv` beside the gate/up selector, and the runtime's
batched AWQ rotation helpers delegate to `pipeline::batched`. The beta
helpers shared by the moved dense bodies and the arch MoE bodies (the GDN
chunk-scan views/layer and the A3B PM KV codec / F5 Q-prep route) also live
once in dispatch. The TriAttention tap is one arch binding
(`program::batched_attention_tap(config).get()`, `Some` exactly when the tap
is enabled), passed to the attention executor and shared with the MoE
attention prep.

Remaining arch-side batched paths (slice 3):

- `batch_chunk_full_attn_fallback` / `run_fa_layer_body`: the per-token
  full-attention fallback for weights the batched path does not admit;
- the MoE layer bodies `batch_chunk_delta_net_moe` and
  `batch_chunk_full_attn_moe{,_prep,_finish}` (they call the moved stages,
  but their projections, PARO arms and always-residual output projection
  stay in the arch);
- the F2 chunk pair (`forward_prefill_chunk_pair`, `FaPairStage`,
  `batch_chunk_fa_attend_merged`), which stages two chunks into one attend.

## Gates

Each slice byte-compares `mq4_prefill_state_oracle` dumps (prefill logits,
Q8 KV, GDN state + EF, conv state, every decode step's logits) against its
beta base, on `qwen3.5-9b.mq4`, `qwen3.8-27b.mq4-xt` and
`qwen3.6-35b-a3b.mq4r`, with at least two chunk boundaries plus an odd tail.
It also needs greedy serve A/B rows for the routes it touches, and Redline
`--kv-mode q8 --pm4` shadow parity with an unchanged decode tape hash.
Evidence: HIPF-117 (slice 1), HIPF-118 (slice 2; also the
`prefill_ragged_oracle` hashes. `cb_vmm_state_oracle` does not run on gfx1151:
its k=1 executor hits a nonfinite logit on beta too, so it needs a gfx1201
receipt). The gfx1100/gfx1201-only routes of slice 2 (gfx1100
`ChainVerifySplit`, gfx1201 lane-batched CB verify, PM FP8 verify scope, IU4
verify, FP8-stream producers, gfx12 FA prep, A3B PM/F5, A3B native FP8 FA2
prefill) are token-identical moves (Loki `fnaudit.py`: 57 token-identical fns,
76 differing only in bindings/renames; the beta `483e614b2`/`93b67256b` FP8
port is audited the same way in HIPF-118); their hardware receipts are
outstanding: a hiptrx runbook (`cb_vmm_state_oracle` k=1,2,3, a 27B DFlash
serve A/B, and an A3B native FP8 prefill A/B beta vs stack) and a hipx row
(27B DFlash with split verify).

## Dense tensor parallelism on the generic engine (#927 item 4, HIPF-142)

`qwen35::mesh_tp` declares one step program for prefill and decode. Per
layer: the mixer (`SplitRule::HeadMixer`, declared input projections
`HeadSharded`, output projection `RowShard { axis: 1 }`) writing `mix_out`,
a residual `Accumulate`, `GatedMlp` (`SplitRule::GatedMlp`, `Product` ladder,
gate/up `ColumnShard { axis: 0 }`, down `RowShard { axis: 1 }`) writing
`ffn_out`, and a second `Accumulate`. On a pure TP mesh the splitter turns
each `Accumulate` into one rank-ordered `AllReduceAdd` into `x`, so a rank
runs `[embed, mixer] AllReduceAdd [ffn] AllReduceAdd [mixer] … [head]` —
the hand dense-TP route's two reductions per layer.

Each rank holds an even contiguous block of the query, KV, DeltaNet key and
value heads and of the FFN intermediate (`tp_rank_layouts`:
`dense_tp_rank_layouts` admitted only for even splits, loaded through
`load_weights_dense_tp_rank`). A decode epoch binds the decode scratch and
runs the mixers' `MixerOutput::Write` arm; a prefill epoch binds the rank's
batch scratch and runs the batched mixers and `Product` rows with the
`Partial` epilogue (last-row logits). `Qwen35TpBindings` is tied to the plan
of exactly those shards and refuses a prefill past the batch scratch,
sessions, checkpoints and prefix repair.

### Why one compound mixer rule

`SplitRule::HeadMixer` mirrors `GatedMlp`: whole input, head-split input
projections and state, row-split output projection, Partial output reduced
by its consumer. The alternative, `HeadLocal` plus a separate row-split
`Project` for the output projection, needs the mixer's attention output as a
graph resource between two ops. The hybrid mixers fuse that boundary
(fused gate + MQ-rotate attention epilogue feeding a prerotated GEMV,
gated norm + MQ rotate in DeltaNet), and `HeadLocal` has no partial output.
Qwen4's attention and GDN blocks are the intended second consumer; they are
still `Opaque` today.

### The `MixerOutput::Write` arm

`DeltaNetMixerOp` and `GatedAttentionOp` take `output: MixerOutput`.
`Residual` is the fused single-device body. `Write(out)` is the former hand
`dense_tp_{deltanet,attention}_partial` body, moved verbatim: the plain
chain (normed projection, unfused prep, plain recurrence or attention,
GEMV into `out`), with compacted KV and a prerotated input refused. The hand
dense-TP route now runs its decode mixers through this arm.

### What stays on the hand route

The product dense-TP route (`EpArch::Qwen35DenseTp` serve, bench, MTP and
DFlash dense TP) stays on `forward_{scratch,prefill}_dense_tp*` until the
generic engine has what it uses: per-segment hipGraph capture and replay of
decode, the uneven TP3/TP5 layouts (Q-group splits and replicated KV heads),
the peer-rooted reduction route, and the verify captures (hidden-state ring,
GDN tape, row capture and state rollback) that MTP and DFlash need. Each is
an executor capability, not a Qwen3.5 one.
