# Static YaRN

Increasing `--max-model-len` reserves a larger serving window; it does not change
the model's rotary embeddings. A checkpoint trained at 262144 positions can
explicitly request static YaRN for a 524288-token deployment. Whether that extension
preserves useful accuracy must be measured on the model and workload.

One reader takes every architecture's rotary scaling (`yarn_from`,
`arch/common/rad_arch.h`). For the target it reads `text_config.rope_parameters`,
then `rope_parameters`, then the legacy `text_config.rope_scaling` and `rope_scaling`,
in that order; the first object with `rope_type` (or legacy `type`) is authoritative.
A drafter (DFlash2, DSpark) reads its own `draft.rope_parameters` /
`draft.rope_scaling`; it does not inherit the target's scaling. The vision tower keeps
its original axial rotations. A GGUF's `rope.scaling.*` keys are carried in as the
same `rope_scaling` keys when it is read.

**The factor convention is the config's**, as vLLM and transformers read it:
`factor` 2 over `original_max_position_embeddings` 262144 reaches 524288 positions.

Validated so far: the Qwen3.8 27B and 35B-A3B containers and both Flash-Next
variants at factor 2, with retrieval at ~500K tokens, and TP3 perplexity with
`--rope-scaling` within roughly ±1% of stock at factor 2. The LLaMA plugins read the
same keys and declare the same ops; YaRN there has had no model run.

## Metadata

| Key within the rotary object | Meaning |
| --- | --- |
| `rope_type` | `yarn`; `default`, absent or legacy `mrope` is no scaling; `linear` divides positions by `factor` |
| `factor` | finite number >= 1, required for YaRN |
| `original_max_position_embeddings` | positive integer, required; the **training** length, not the enlarged deployment/table length |
| `beta_fast`, `beta_slow` | defaults 32 and 1; finite, fast >= slow > 0 |
| `attention_factor` | positive finite cos/sin multiplier; default 1 + 0.1 ln(factor) for factor > 1, otherwise 1 |
| `truncate` | true/1 (default): floor/ceil correction bounds; false/0: retain fractional bounds |

Every other type -- `llama3`, `longrope`, `dynamic` -- is refused by name at load,
as are the `mscale` and `mscale_all_dim` variants (specify `attention_factor`). A
GGUF that sets llama.cpp's YaRN `attn_factor`, `ext_factor` or log multiplier to
anything but its neutral value is refused when read. Rotary width (including
partial rotary), theta and M-RoPE section selection are unchanged.

## Convert a separate container

Use the original target and drafter checkpoints, the same quantization recipe,
and a **new output path**. `--reuse` is valid only for the same source weights and
recipe; it is not a way to mix weights from different model revisions.

```sh
rad-convert /checkpoints/Qwen3.8-27B-FP8 \
  --draft-model /checkpoints/Qwen3.8-27B-DFlash2 \
  --recipe data/recipes/q38-27b-fp8-df2.recipe \
  --reuse /models/qwen3.8-27b-fp8.rad \
  --set text_config.rope_parameters.rope_type=yarn \
  --set text_config.rope_parameters.factor=2 \
  --set text_config.rope_parameters.original_max_position_embeddings=262144 \
  --set draft.rope_parameters.rope_type=yarn \
  --set draft.rope_parameters.factor=2 \
  --set draft.rope_parameters.original_max_position_embeddings=262144 \
  --max-ctx 524288 --kv-cache-dtype bf16 \
  -o /models/qwen3.8-27b-fp8-yarn2.rad
```

Run with `--plan-only` first. `--max-ctx` is the converter's graph bound, not a
replacement for the explicit YaRN metadata or the server's `--max-model-len`.
The bundled `.rad` by itself is not a converter input; keep the source checkpoints.
Check both target and draft settings with `rad-info --meta` after conversion.

Serve on enough memory for the requested window:

```sh
radiance --model /models/qwen3.8-27b-fp8-yarn2.rad \
  --tp 4 --max-model-len 524288 --max-num-seqs 32 --kv-cache-dtype bf16
```

That cache flag controls **both** target and drafter KV. Startup capacity and
successful actual long requests must be checked; the command is not a promise
of concurrency or accuracy. Keep the unmodified model and serving setup until
long-context quality and speed meet the deployment's requirements.

## Serve an existing container: `--rope-scaling`

The same metadata can be written when the container is loaded instead of converted in:

```sh
radiance --model /models/qwen3.8-next-flash-fp8.rad \
  --rope-scaling yarn:2 --max-model-len 409600
```

`yarn:FACTOR[:ORIG]` sets `rope_type = yarn`, `factor` and `original_max_position_embeddings`
(`ORIG`, default the container's own trained length) in the rotary object the reader above
takes first -- for a Qwen3.8 container that is `text_config.rope_parameters`, which already says
`default`; YaRN written under plain `rope_parameters` would sit behind it and never be read. A
drafter gets the same factor over its own trained length. Over a checkpoint's own YaRN
the factor is replaced, its trained length kept unless `ORIG` is given, and an explicit
`attention_factor` (or `mscale` variant) dropped so the magnitude is derived for the new factor.
`none` takes a checkpoint's YaRN off. A linear or other explicit rotary type is refused rather
than stacked over. The result is resolved and checked exactly as a converted container's is
(`core/format/rope_override.h`, held to `yarn_from` in `tests/yarn_test.cpp`).

Startup says what the rotary is (`rope: YaRN factor 2 over 262144 trained positions -- the rotary
reaches 524288`) and warns when `--max-model-len` is past what it covers -- also with no scaling
at all, past the trained length. A persisted prefix cache (`--prefix-cache-dir`) is fingerprinted
with the resolved rotary of the target and the drafter, so a store written under a different
one -- another `--rope-scaling`, or a container converted with YaRN -- is not reused: every cached
key was rotated differently.

The flag is for trying a factor, or serving a container whose source checkpoints are not at hand;
a deployment that settles on one can convert it in as above and drop the flag.

## Kernel and test contract

`rope`, `rope_table` and `qsa_block_key` take optional resolved coefficients
`yarn_factor`, `yarn_low`, `yarn_high`, `yarn_magnitude` (docs/OPS.md). Absent factor
is the original path, with the original op identities. With YaRN, `scale` must be 1:
inverse frequencies are blended rather than all positions being divided. Axial
rotary with YaRN is refused. Rebuild the bundled plugins together because their
shared op schemas must match.

The math follows the static YaRN formulation (Peng et al.) as vLLM's
`YaRNScalingRotaryEmbedding` implements it. `tests/yarn_test.cpp` pins the
correction bounds to vLLM's integers (14/22, 28/43, 56/85 for rotary widths 64, 128,
256 at factor 2, theta 1e7, 262144 trained positions); `tests/ref_test.cpp` and
`r4d_selftest` compare libref and libr4d against a double-precision evaluation of the
same formula at positions through 524287, within 8 ulp of each angle -- tight where
the ramp sits, loose only in the fastest pairs at long positions, which no f32 rotary
places better than its angle's own ulp. libavx is held to libref by `avx_check`, and
fused versus unfused HIP rotation is checked byte-for-byte separately.
