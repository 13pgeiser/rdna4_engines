# Static YaRN

A larger `--max-model-len` reserves a larger serving window; it does not change the model's rotary
embedding. A checkpoint trained at 262144 positions can request static YaRN explicitly for a
524288-token deployment. Whether the extension keeps useful accuracy must be measured on the model
and workload.

Every architecture reads its rotary scaling through one function (`yarn_from`,
`arch/common/rad_arch.h`). For the target it reads `text_config.rope_parameters`, then
`rope_parameters`, then the legacy `text_config.rope_scaling` and `rope_scaling`; the first object
with a `rope_type` (or legacy `type`) key is authoritative. A drafter (DFlash2, DSpark) reads its
own `draft.rope_parameters` / `draft.rope_scaling` and does not inherit the target's scaling. The
vision tower keeps its axial rotary. A GGUF's `rope.scaling.*` keys are read in as the same
`rope_scaling` keys.

The factor convention is the config's, as vLLM and transformers read it: `factor` 2 over
`original_max_position_embeddings` 262144 reaches 524288 positions.

Validated: the Qwen3.8-27B and Qwen3.6-35B-A3B containers and both Flash-Next variants at factor 2,
with retrieval at about 500K tokens, and TP3 perplexity with `--rope-scaling` within about ±1% of
the unscaled model at factor 2. The LLaMA plugins read the same keys and declare the same ops, but
YaRN has not been run on a LLaMA model.

## Metadata

| Key within the rotary object | Meaning |
| --- | --- |
| `rope_type` | `yarn`; `default`, absent or legacy `mrope` is no scaling; `linear` divides positions by `factor` |
| `factor` | finite number >= 1, required for YaRN |
| `original_max_position_embeddings` | positive integer, required; the training length, not the enlarged deployment or table length |
| `beta_fast`, `beta_slow` | defaults 32 and 1; finite, fast >= slow > 0 |
| `attention_factor` | positive finite cos/sin multiplier; default 1 + 0.1 ln(factor) for factor > 1, otherwise 1 |
| `truncate` | true/1 (default): floor/ceil correction bounds; false/0: keep fractional bounds |

Every other type (`llama3`, `longrope`, `dynamic`) is refused by name at load, as are the `mscale`
and `mscale_all_dim` keys (use `attention_factor`). A GGUF that sets llama.cpp's YaRN
`attn_factor`, `ext_factor` or log multiplier to anything but its neutral value is refused when
read. Rotary width (including partial rotary), theta and the M-RoPE sections are unchanged.

## Converting a separate container

Use the original target and drafter checkpoints, the same recipe, and a new output path. `--reuse`
is valid only for the same source weights and recipe; it cannot mix weights from different model
revisions.

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

Run it with `--plan-only` first. `--max-ctx` is the converter's graph bound; it replaces neither
the YaRN metadata nor the server's `--max-model-len`. A `.rad` is not a converter input, so keep
the source checkpoints. After conversion, check the target and draft keys with `rad-info --meta`.

Serve on enough memory for the requested window:

```sh
radiance --model /models/qwen3.8-27b-fp8-yarn2.rad \
  --tp 4 --max-model-len 524288 --max-num-seqs 32 --kv-cache-dtype bf16
```

`--kv-cache-dtype` sets both the target's and the drafter's KV cache. Check the startup capacity
and real long requests; the command guarantees neither concurrency nor accuracy. Keep the
unmodified model and serving setup until long-context quality and speed meet the deployment's
requirements.

## Serving an existing container: `--rope-scaling`

The same metadata can be applied when the container is loaded instead of at conversion:

```sh
radiance --model /models/qwen3.8-next-flash-fp8.rad \
  --rope-scaling yarn:2 --max-model-len 409600
```

`yarn:FACTOR[:ORIG]` (FACTOR above 1) sets `rope_type = yarn`, `factor` and
`original_max_position_embeddings` (`ORIG`, default the container's own trained length) in the
object the reader takes first. For a Qwen3.8 container that is `text_config.rope_parameters`,
which already says `default`; YaRN written under plain `rope_parameters` would never be read. A
drafter gets the same factor over its own trained length.

Over a checkpoint's own YaRN, the factor is replaced, the trained length is kept unless `ORIG` is
given, and an explicit `attention_factor` (or `mscale` key) is dropped so the magnitude is derived
for the new factor. `none` turns a checkpoint's YaRN off. A `linear` or other explicit rotary type
is refused, not stacked over. The result is resolved and checked as a converted container's is
(`core/format/rope_override.h`, held to `yarn_from` by `tests/yarn_test.cpp`).

Startup logs the rotary:

```
rope: YaRN factor 2 over 262144 trained positions -- the rotary reaches 524288; --max-model-len 409600 (--rope-scaling)
```

It warns when `--max-model-len` is past what the rotary covers, and, with no scaling, past the
trained length.

Use the flag to try a factor, or to serve a container whose source checkpoints are not at hand. A
deployment that settles on a factor can convert it in as above and drop the flag.

## Kernel and test contract

`rope`, `rope_table` and `qsa_block_key` take the optional resolved coefficients `yarn_factor`,
`yarn_low`, `yarn_high` and `yarn_magnitude` ([OPS.md](OPS.md)). Without them an op is declared
as for an unscaled rotary, so op identities and tune keys do not change. With YaRN, `scale` must be 1, since YaRN
blends inverse frequencies instead of dividing positions. Axial rotary with YaRN is refused.
Rebuild the bundled plugins together, because their shared op schemas must match.

The math follows the static YaRN formulation (Peng et al.) as vLLM's
`YaRNScalingRotaryEmbedding` implements it. `tests/yarn_test.cpp` pins the correction bounds to
vLLM's integers (14/22, 28/43, 56/85 for rotary widths 64, 128, 256 at factor 2, theta 1e7, 262144
trained positions). `tests/ref_test.cpp` and `r4d_selftest` compare libref and libr4d against a
double-precision evaluation of the same formula at positions up to 524287, within 8 ulp of each
angle. That bound is tight where the ramp sits and loose only in the fastest pairs at long
positions, where no f32 rotary can do better than the angle's own ulp. `avx_check` holds libavx
to libref, and fused against unfused HIP rotation is checked byte for byte.
