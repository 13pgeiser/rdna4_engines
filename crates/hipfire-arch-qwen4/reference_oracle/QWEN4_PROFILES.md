# Qwen4 tiny profiles (S0 source contract)

Developer-only CPU references: `tiny-moe` (pinned upstream equations) and `tiny-dense-dev`
(non-upstream dense equation). Neither is a production admission and neither proves a real
Qwen4-27B config; both exist so S1/S2/S3 can lower complete tiny checkpoints against per-op dumps.

## Commands (clean CPU venv, no GPU, no network if the vLLM source cache is warm)

```bash
cd <worktree>
export HIPFIRE_QWEN4_ORACLE_CACHE=$HOME/.hipfire/datasets/cachequality-qwen4-upstream   # or --cache-dir
PY=/home/kaden/qcal/release-0.4.2/qwen4-27b/s0-cpu-venv/bin/python
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-moe       --out /tmp/q4-tiny-moe   --seed 3800
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/q4-tiny-dense --seed 3800
# width-5120 knob (1-2 layers only; reduced projections/FFN)
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --hidden-size 5120 --layers 1 --out /tmp/q4-dense-5120
$PY -I -m unittest discover -s crates/hipfire-arch-qwen4/reference_oracle -p 'test_*.py'   # run from the repo root
```

`-I` ignores `PYTHONPATH`; `HIPFIRE_*` variables still apply. The system Debian torch crashes
(allocator error); use only the clean venv. Output per profile: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors` (BF16 + three I64 PLE buffers), `forward.npz`
(every op), `states.npz` (helper-capture states, plus DynamicCache states after prefill and chunked decode when the stack has an attention layer), `mtp_step0.npz`,
`manifest.json` (envelope, seed, source hashes, token ids, full tensor inventory with
shape/dtype/sha256, per-dump shape/logical dtype/sha256, negative-control results).

## Frozen envelope (root of `config.json`)

```json
{"hipfire_profile": {"id": "tiny-moe" | "tiny-dense-dev", "developer_only": true,
  "equation_source": "pinned-upstream" | "non-upstream-dense",
  "ffn": {"kind": "dense", "intermediate_size": N}
       | {"kind": "moe", "num_experts": E, "top_k": K, "intermediate_size": N,
          "shared_intermediate_size": M | null, "norm_topk_prob": bool}}}
```

PLE stored-geometry marker (S1 schema; values are read from the instantiated pinned `Qwen4ExpTextNGramEmbedding`, not constants):
`hipfire_profile.ple = {"valid_rows": 18}` and top-level
`qwen4_ple = {"version": 1, "multipliers": [3 x i64], "head_vocab_sizes": [7, 11], "head_offsets": [0, 7], "padded_rows": 24}`.
No other marker keys are emitted; row width, shard count and head counts come from `text_config`
(`ple_embed_dim`, `split_ngram_parts`, `heads_per_ngram`). Exactly one PLE table is supported by this marker.

Outer `qwen4_exp` / `Qwen4ExpForConditionalGeneration`, inner `qwen4_exp_text`; layer types use the
checkpoint spelling (`full_attention`; HF normalises it to `qwen_sparse_attention`). The config
also carries `full_attention_interval=2`, `mamba_ssm_dtype`, `mtp{num_hidden_layers,layer_types,hybrid,rope_theta}`,
`mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings=false`. Tiny vision tower tensors are
inventory component `vision` (never executed); filter by `component` for the text-only inventory.
`shared_intermediate_size = null` is **not constructible** from pinned HF (the sparse block always
builds a shared expert) and is rejected, not faked. Dense config omits every MoE key.

## Declared gate construction and shrink constraints

hidden16 vocab32 depth2 heads2/kv1 head8; GDN key1/value2 dims 8/8 conv2; layers `[linear, full]`;
experts3 top2 intermediate8 shared8; HC2 lowrank4; PLE on layer 1 (embed16, ngram3, heads/ngram1,
vocab-base7, divisible-by8, split1, conv2, eos2); QSA heads1 kv1 dim4 budget4 ratio2; rope
theta 10000 partial .5; max positions 64; output gate sigmoid; eager attention.
Constraints the shrink must respect (all enforced by pinned HF validation or observed):

- PLE: `ple_embed_dim % ((ngram_size-1)*heads_per_ngram) == 0` (16 % 2); PLE only on linear layers; eos id required.
  Primes after `base-1`: heads 7 and 11 -> **valid 18 rows, padded 24** (divisible by 8), row width 8, 3 I64 buffers.
- QSA: `indexer_kv_heads == 1`, `budget % ratio == 0`, `rotary_dim (= head_dim*partial) <= indexer_head_dim` (4 <= 4).
  Selection only bites with >= 3 complete blocks (>= 6 visible tokens): the fixed text has 9 tokens.
- MoE: HF needs `num_experts > 0` and a shared expert; `top_k < experts` so the router boundary exists.
- Fixed text `hi<|endoftext|>to hi.` -> 9 ids, EOS (id 2) mid-sequence to exercise EOS-aware n-gram history.
- The requested seed is used exactly. QSA/router top-k boundary ties are not screened: selection is whatever pinned
  `torch.topk` does on CPU, deterministic within the pinned environment.
- Linear-attention-only stacks (1 layer) cannot run cached/chunked decode in pinned HF
  (`DynamicCache.get_seq_length` needs an attention layer): no `prefill.*`/`chunked.*` states. `states.npz` is still emitted with
  `captured.layer<L>.{conv_state0,recurrent_state0,conv_state1,conv_state2}` derived from the helper captures (GDN conv input and
  final recurrent state, PLE short-conv input, last token ids); with a cache these are asserted bit-equal to the `DynamicCache` states.
- Width 5120 is bounded to 1-2 layers; every projection except the hidden axis is reduced. The two
  `[H,H]` MTP fusion projections (2 x 5120^2 = 52M elements, ~105 MB BF16) are the structural floor; guard 80M elements.

## Source / dense boundary

- Sources: the three Transformers files and three vLLM files are SHA-256 verified against
  `upstream.UPSTREAM_SOURCE_SPECS` **before use**; the verified bytes (not whatever import finds) are
  `exec`-ed under the real module names, and an already-imported unverified module is refused. A matching
  version string is never accepted alone (`transformers 5.16.1`, commit `93c8b7b4...`; vLLM `e126687a...`).
- `tiny-moe`: the real `Qwen4ExpForConditionalGeneration`, BF16, eager. Dumps are forward-hook / helper-tap captures of
  GDN conv+recurrence, PLE n-gram ids/embedding/conv, HC mix/inject, QSA indexer mask/pooling norms/attention, router/experts/shared expert, final mixer, logits.
- `tiny-dense-dev`: pinned decoder layers with the MLP slot replaced by
  `equations.dense_ffn_non_upstream` (SiLU gate*up -> down, BF16 rounding after every projection/activation/product/output)
  at the declared `intermediate_size`. The pinned `Qwen4ExpTextSparseMoeBlock` is swapped before construction, so no router/expert/shared
  tensor exists. Pinned HF has **no** dense decoder (`num_experts <= 0` rejected; layer always builds the sparse block);
  the dense profile is hipfire's hypothesis, labelled `non-upstream-dense`. It matched `Qwen4ExpTextMLP` bit-for-bit on the tiny case (reusable math only).
- MTP (`mtp_step0.npz`, inventory `mtp.*`): pinned HF ignores `^mtp.*` and has no MTP model. The branch executes the real vLLM
  `Qwen4ExpMultiTokenPredictor.forward` body from `amd/mtp.py` (AST-compiled, `self.*` set and key statements asserted against drift), with the pinned HF decoder
  layer (QSA + FFN + both HC modules) and HC final mixer behind vLLM's `(hidden, mlp_out, injection)` / `combine_and_mix` contracts. Label:
  `pinned-vllm-mtp-forward+hf-operators`. Not covered: later draft steps (`skip_topk` QSA index reuse, draft paged KV), vLLM Triton HC/QSA
  kernels (equivalence rests on vLLM `common/hyperconnection.py` torch equations matching HF `GatedResidual`, asserted). Embedding and LM head are shared with the main model.

## Negative controls (each must change named dumps; identity must change none)

`identity_rebuild`, `gdn_output_gate` (sigmoid<->silu), `ple_eos_history`, `ple_cached_history` (same suffix, different cached prefix),
MoE: `moe_router_gate`, `moe_norm_topk_prob`, `bf16_vs_f32_rounding`; dense: `dense_ffn_rounding`, `dense_gate_projection`.
The exact requested seed is used (no seed search): tiny-moe and tiny-dense-dev at `--seed 3800` pass every control on torch 2.9.1+cpu, 1 thread.
Dumps are bit-stable on one torch build/thread count; cross-build comparison should use BF16 unit roundoff (2^-7).
