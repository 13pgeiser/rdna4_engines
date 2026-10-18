# Architectures: geometry and traps

An architecture plugin claims one **(architecture id, quantisation descriptor)** pair, the
selection key of `spec.md` §2.4, so a family with two checkpoint formats is two `.so` files rather
than one that branches. `rad-convert` normalises what it reads (safetensors or GGUF) into the id a
plugin claims.

| plugin | id | quantisation | what it serves |
|---|---|---|---|
| `arch/llama_fp8` | `llama` | `fp8_e4m3` | the plain LLaMA decoder, from a checkpoint that holds E4M3 codes and a `weight_scale_inv` plane |
| `arch/llama_dense_fp8` | `llama` | *(empty)* | the same graph from a bf16 checkpoint, linears made block fp8 by a recipe or kept bf16. MiniCPM5-2B |
| `arch/qwen35_bf16` | `qwen35` | *(empty)* | the Qwen3.5/3.6/3.8 dense hybrid, served bf16 |
| `arch/qwen35_fp8` | `qwen35` | `fp8_e4m3` | the same, block-scaled FP8 weights and activations |
| `arch/qwen35moe_fp8` | `qwen35moe` | `fp8_e4m3` | `qwen35_fp8` with the feed-forward routed |
| `arch/qwen4exp_fp8` | `qwen4exp` | *(empty)* | Qwen4-Exp (Qwen3.8-Flash-Next), a bf16 checkpoint quantised at convert |
| `arch/lfm2_bf16` | `lfm2` | *(empty)* | LiquidAI's LFM2 and LFM2-VL, bf16, one rank |
| `arch/embgemma2` | `embeddinggemma2` | *(empty)* | EmbeddingGemma 2: a bidirectional encoder with Gemma 4's vision tower and audio conformer, pooled to one vector |
| `arch/gemma4_bf16` | `gemma4` | `compressed-tensors` | Gemma 4 31B dense with its vision tower: sliding-window and proportional-RoPE global (`attention_k_eq_v`) attention, GELU ffn, softcapped logits; int4 codes with an f16 scale per 128 from Google's CT-INT4 (QAT) release, served w4a8. Optional draft head: Google's assistant (`rad-convert --draft-model`), with no cache of its own (`rad_drafter_shared_kv`) |
| `arch/gemma4` | `gemma4` | *(empty)* | Gemma 4 text from Google's bf16 checkpoint (the 26B-A4B MoE and the dense sizes without per-layer embeddings) with z-lab's DFlash drafter; routed experts bf16 or block fp8 |
| `arch/qwenimage21` | `qwenimage21` | *(empty)* | Qwen-Image 2.1 and 2.1-Turbo, an image generator, from the diffusers pipeline directory; DiT and text encoder bf16 or block fp8 |
| `arch/qwen3asr_bf16` | `qwen3asr` | *(empty)* | Qwen3-ASR: a Whisper-style audio encoder over a dense Qwen3 decoder, speech to text |
| `arch/minimaxmusic3` | `minimaxmusic3` | *(empty)* | MiniMax-Music3, songs from a description and lyrics: a Qwen3-8B frame model, a depth decoder, a flow-matching DiT and a DAC vocoder, from the diffusers pipeline directory, bf16 |
| `arch/breeze` | `breeze` | *(empty)* | Breeze-TTS-2, speech from text and a description of the voice: a T5Gemma 2 text encoder, a Qwen3 backbone with a 16-codebook depth decoder and the Qwen3-TTS 12 Hz codec, from the release directory, bf16, one card |
| `arch/minimaxh3` | `minimaxh3` | *(empty)* | MiniMax-H3, a video with its stereo soundtrack from text: Qwen3-VL-32B's language model as the text encoder, a 33B single-stream diffusion transformer, a tiled ViT video decoder and a BigVGAN audio decoder, from the diffusers pipeline directory; linears block fp8 (`data/recipes/minimax-h3-fp8.recipe`) |

The traps below are readings that are wrong without any error: the model produces fluent output.

## Common rule

**`head_dim` is read, never derived.** On the Qwen hybrids `head_dim` is 256 while
`hidden_size / num_attention_heads` is 213, and a derived value silently breaks the projection
widths, the KV page size and the attention kernel's band. `core/engine_bringup.cpp` takes it from
the container's meta.

---

# `llama`: the plain decoder

`model_type: llama` with `LlamaForCausalLM`. Every layer is full attention; unlike `qwen35_fp8`
there is no gated delta net, layer schedule, output gate or per-head q/k norm.

MiniCPM5-2B: 42 layers, hidden 2048, 16 query heads over 2 KV heads at `head_dim` 128, SwiGLU at
6144, RMS norms, full rotary at theta 5e6, an untied 130560-row head, no biases.

**Two checkpoint formats, one graph.** OpenBMB publishes MiniCPM5-2B only in bf16, so
`llama_dense_fp8` declares the empty quantisation descriptor and the recipe
`data/recipes/minicpm5-2b-dspark.recipe` quantises at convert (libquant's `rtn`, one amax per
128x128 block, `scale = amax/448` rounded to bf16). The container then holds what an fp8 release
would, both plugins declare the same ops over it, and nothing downstream can tell which one ran. A
container whose linears are left bf16 (`minicpm5-2b-bf16-dspark.recipe`) is read by `gemm_nt`.

fp8: `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`, each with a bf16 `[N/128][K/128]`
scale plane. Not fp8: every norm and the embedding. The lm_head is declared at whatever the source
holds (bf16 from a checkpoint, fp8 where the recipe made it so), and the GEMM that reads that
encoding is selected. Both vocabulary edges are row-sharded.

---

# `qwen35`: the hybrid Gated DeltaNet / gated-attention family

Qwen3.5, Qwen3.6 and Qwen3.8 dense models. HuggingFace calls it `model_type: qwen3_5`
(`Qwen3_5ForConditionalGeneration`); GGUF calls it `qwen35`.

## Qwen3.8-27B-FP8

```
64 layers, full_attention_interval 4  ->  layers 3,7,...,63 are FULL ATTENTION (16 of them)
                                          the other 48 are GATED DELTA NET
hidden_size 5120      intermediate_size 17408      vocab_size 248320
num_attention_heads 24   num_key_value_heads 4   head_dim 256      ->  GQA 6
attn_output_gate true                                              ->  gated attention
partial_rotary_factor 0.25  ->  RoPE over 64 of the 256 head dims
mrope_interleaved true, mrope_section [11, 11, 10]                 ->  MROPE, not plain rope
rope_theta 1e7        rms_norm_eps 1e-6        max_position_embeddings 262144

linear_num_key_heads   16    linear_key_head_dim   128
linear_num_value_heads 48    linear_value_head_dim 128    ->  GDN GQA 3:1
linear_conv_kernel_dim 4     mamba_ssm_dtype float32      output_gate_type swish
mtp_num_hidden_layers 1      tie_word_embeddings false
quantization: fp8 e4m3, activation_scheme "dynamic", per-weight `weight_scale_inv`
vision: 27 blocks, hidden 1152, 16 heads -> head_dim 72, patch 16, out_hidden_size 5120
```

### libr4d entry points

| model geometry | libr4d entry point |
|---|---|
| head_dim 256, GQA 6, paged block 16 | `attn_prefill_h256_gqa6_*`, `attn_decode_h256_gqa6_*` |
| linear k128 / v128, chunk 64 | `gdn_chunk_scan_k128_v128_c64_bf16`, `gdn_kkt_solve_k128_c64_bf16` |
| conv width 4, k128 | `gdn_conv_prep_w4_h128_bf16`, `gdn_conv_update_w4_h128_bf16` |
| recurrent state fp32 | `gdn_recurrent_update_k128_v128_bf16_fp32state` |
| gated attention, interleaved q/gate | `qk_norm_rope_gate` (reads the q/gate interleave off the operand's row pitch) |
| GDN gated norm over an interleaved z | `gdn_gated_norm_had_quant_i8` (`zgrp`/`zblk`) |
| vision head_dim 72 | `attn_vit_h72_bf16` |

## Tensor names: the FP8 safetensors checkpoint

Sharded one file per layer (`layers-N.safetensors`), plus `outside.safetensors` and
`mtp.safetensors`. Every FP8 weight has a companion `weight_scale_inv`; the norms, biases, `A_log`,
`dt_bias` and `conv1d` stay bf16. Linear shapes below are `[in, out]`, as the plugins write them;
the safetensors tensors are stored `[out, in]`.

**Full-attention layer** (`model.language_model.layers.N.`):
```
input_layernorm.weight              post_attention_layernorm.weight
self_attn.q_proj.weight  (+_scale_inv)   [5120, 12288]  = 24 heads x 256 x 2, query and GATE
self_attn.k_proj.weight  (+_scale_inv)   [5120,  1024]
self_attn.v_proj.weight  (+_scale_inv)   [5120,  1024]
self_attn.o_proj.weight  (+_scale_inv)   [6144,  5120]
self_attn.q_norm.weight [256]       self_attn.k_norm.weight [256]
mlp.{gate,up}_proj.weight (+_scale_inv)  [5120, 17408]
mlp.down_proj.weight      (+_scale_inv)  [17408, 5120]
```

**Linear (GDN) layer**:
```
input_layernorm.weight              post_attention_layernorm.weight
linear_attn.in_proj_qkv.weight (+_scale_inv)  [5120, 10240]  = 2*16*128 + 48*128
linear_attn.in_proj_z.weight   (+_scale_inv)  [5120,  6144]  = 48*128
linear_attn.in_proj_a.weight                  [5120,    48]
linear_attn.in_proj_b.weight                  [5120,    48]
linear_attn.conv1d.weight                     [10240, 1, 4]
linear_attn.A_log [48]    linear_attn.dt_bias [48]    linear_attn.norm.weight [128]
linear_attn.out_proj.weight (+_scale_inv)     [6144, 5120]
mlp.{gate,up,down}_proj  as above
```

**Model level**: `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`,
`lm_head.weight` (not tied on this model), and the MTP head: `mtp.fc.weight`, `mtp.norm.weight`,
`mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`.

## Full-attention layer: gated attention

`q_proj` produces the query **and a gate, interleaved per head**: for each of the 24 heads the
first 256 elements are the query and the next 256 the gate, so both are strided views with pitch
512.

```
qg      = gemm(x, q_proj)                  [n_tok, 12288]
k, v    = gemm(x, k_proj), gemm(x, v_proj) [n_tok, 1024]
q       = rmsnorm(view(qg, off 0,   pitch 512), q_norm)     per head, over 256
k       = rmsnorm(k, k_norm)                                per head
gate    = view(qg, off 256, pitch 512)
q, k    = mrope(q, k, positions, sections [11,11,10], rotary dims 64 of 256)
o       = attn_paged(q, k, v)              scale = 1/sqrt(256)
o       = o * sigmoid(gate)                <-- the gate. output_gate_type "swish" names the
                                               GDN gate, not this one; this one is a sigmoid
out     = gemm(o, o_proj)
```

`qk_norm_rope_gate` does the two norms, the rope and the gate extraction in one kernel, taking the
interleave from the `q_gate` operand's row pitch, so the projection is declared as laid out rather
than converted at load. Declare it first and fall back to the unfused sequence.

## Linear layer: gated delta net

```
qkv     = gemm(x, in_proj_qkv)             [n_tok, 10240]
z       = gemm(x, in_proj_z)               [n_tok,  6144]
b       = sigmoid(gemm(x, in_proj_b))      [n_tok, 48]     -> beta
a       = gemm(x, in_proj_a)               [n_tok, 48]
g       = softplus(a + dt_bias) * A_log_term
conv    = silu(causal_conv1d(qkv, conv1d, conv_state))     depthwise, width 4, 10240 channels
q,k,v   = strided views of conv:  q at 0 (16x128), k at 2048 (16x128), v at 4096 (48x128)
q, k    = l2_norm(q), l2_norm(k)           <-- L2, NOT rmsnorm
o       = delta rule: chunked scan at prefill, recurrent update at decode
o       = rmsnorm(o, linear_attn.norm) * silu(z)           the gated norm
out     = gemm(o, out_proj)
```

`linear_num_value_heads` (48) is three times `linear_num_key_heads` (16), so each q/k head serves
three value heads; libr4d's GDN kernels take `H` and `Hg` separately for this.

## Traps

- **`q`/`k` use L2 normalisation, not RMS.**
- **`A_log` is a log.** In the GGUF conversion `ssm_a` is already `-exp(A_log)`; in this safetensors
  checkpoint the tensor is `A_log` with the exponent not taken. The converter decides which
  convention reaches the kernel, and must say so.
- **The GDN output gate is `silu` here and `sigmoid` on Qwen4-Exp.** `RMSNormGated` defaults to silu
  and every Qwen3.5/3.6 release keeps it; Qwen4-Exp sets `output_gate_type: "sigmoid"`. The wrong
  gate rescales every value head's output. `arch/common/rad_block_gdn_fp8.h` refuses any other
  name.
- **Readings no per-op check can settle.** The rope convention (`neox`, halves, against `gptj`,
  adjacent pairs), which half of each head's `q|gate` pair is the query (the sigmoid accepts
  either), and the MTP head's concat order (hidden-first or embedding-first) are all
  self-consistent either way. Each is a declared knob at the site that reads it.

## KV groups

Three groups, which is why hybrid KV groups are first-class:

| group | kind | geometry |
|---|---|---|
| full attention | `RAD_KV_FULL` | 16 layers, head_dim 256, 4 KV heads; block size from the resolved kernel |
| GDN recurrent state | `RAD_KV_LINEAR` | 48 layers, 48 states of 128 x 128 |
| GDN conv state | `RAD_KV_CONV` | 48 layers, 2*16 + 48 = 80 heads x 128 = 10240 channels, a rolling `conv_width - 1 + n_spec` window |

**The recurrent state is f32**: 48 x 128 x 128 x 4 = 3 MiB per layer, **144 MiB per sequence**
across the 48 linear layers, and it is what the prefix-cache checkpoint of `spec.md` §7.3
snapshots. Its reads and writes dominate the two GDN kernels. libr4d also compiles an f16 form
(elements are O(0.1); the f16 round trip is about 5e-4 relative) that would halve that traffic,
but the plugins declare f32 because f16 changes the generated text. How a speculative step keeps a
state it can roll back is `docs/OPS.md`'s `state_form`.

## FP8

`quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`, and a `weight_scale_inv` beside
every quantised weight: DeepSeek-style 128x128 block scaling. Activations are quantised at runtime
("dynamic"), so the activation quantiser is an op. The vision tower's first several blocks are in
`modules_to_not_convert` and stay bf16.

**libr4d reads those bytes directly, so `rad-convert` copies them**: no permutation, repacking or
dequantisation, and a `.rad` costs what the checkpoint does. Widening a 27B FP8 checkpoint to bf16
would double the weight bytes; `qwen35_bf16` is for checkpoints that are bf16 to begin with. The
rows are `gemm_fp8a16_nt_m1`, `gemm_fp8a8_nt_m16`, `gemm_fp8a8_gated_nt_m16` and
`gemm_fp8a8_tiled`.

**The shipped path is `fp8a8`.** `dtype` is one string per declared op, so a linear is `fp8a16` or
`fp8a8`, not both. `fp8a16` reads the bf16 activation directly with no quantise launch, but its
kernel serves only `M <= 1` (single-sequence decode). `fp8a8`'s two kernels cover every M, at the
price of one `quant_act_fp8` launch per activation that feeds a GEMM. That cost is per activation,
not per linear, which is why the quantiser is its own struct rather than a step inside the linear.

**The scale plane is a weight operand**: `gemm_nt_q` takes `a, a_scale, b, b_scale`, and `b` and
`b_scale` are two declarations over the planes of one logical weight (`RadWeightDecl::planes`).
Each is placed, moved and sharded on its own; the engine cuts a row-sharded scale plane on its
block boundary as it cuts the codes.

## `qwen35moe`

The same architecture with the MLP replaced by a router plus experts; the gated attention, the
gated delta net, the layer schedule, the rope, the vocabulary edges and the MTP head are identical,
and every difference is behind `#if QWEN35_MOE` in the shared source. It is a second translation
unit rather than a branch on `n_expert` because the id is half the selection key, and one `.so`
cannot claim both ids.

---

# `qwen4exp`: Qwen4-Exp / Qwen3.8-Flash-Next

`model_type: qwen4_exp`, a subclass of `Qwen3_5MoeTextModel`, so most of it composes the blocks
above. New: the gated residual, the n-gram/PLE layer, QSA's indexer, stacked dense expert tensors,
GQA 12, and the delta net's sigmoid output gate.

```
48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE)
hidden_size 2560      hc_count 4  ->  the residual stream is 10240
512 experts of 640 at top-10, plus a shared expert of 640
24 query heads, 2 KV heads, head_dim 256     ->  GQA 12
vocab 248320          max_position_embeddings 262144
indexer_n_heads 4  indexer_head_dim 128  indexer_budget 2048  indexer_compress_ratio 4
mtp_num_hidden_layers 1        output_gate_type sigmoid
```

**The residual stream is four times as wide, and the blocks do not own their norms.** The
checkpoint has no `input_layernorm`, `post_attention_layernorm` or final `norm`. Each block has a
gated residual connection instead: a grouped norm over the 10240 stream, a low-rank data-dependent
read gate that mixes the four sub-streams down to 2560, and four data-dependent scalars that add
the block's output back into all four. There are 96, plus a 97th with no write half that collapses
the stream for the lm_head. The blocks run in their "input already normed and quantised" mode and
so give up two fusions: the residual add and the all-reduce folded into the next norm.

**Each weight's storage is the recipe's choice.** The checkpoint is bf16 with no
`quantization_config`, so the descriptor is empty; `rad-convert` quantises by a recipe, and the
plugin reads each weight's encoding from the container and declares the op that reads it:

- block linears (`self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv, in_proj_z, out_proj}`,
  the shared expert's three): fp8 or int8 (served W8A8);
- routed experts' gate_up and down: fp8, bf16, or one of the four- and five-bit forms of
  `docs/MOE-W4.md`;
- each connection's two mixing matrices and the MTP head's `fc_hidden` / `fc_embedding`: E4M3 rows
  or bf16. The low-rank width is 320, 2.5 scale blocks, so they are scaled a row group at a time;
- the lm_head: whatever a `logits_gemm` kernel reads;
- bf16: every norm gain, `A_log`, `dt_bias`, `conv1d`, `in_proj_a` and `in_proj_b` (48 rows is not
  a whole scale block), the router, `shared_expert_gate` and `embed_tokens`.

`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` is the served choice: four-bit codebook experts with
ten protected experts left bf16, int8 block linears and lm_head, E4M3 mixing and MTP fc matrices,
and the n-gram table at one fixed E4M3 scale.

**Tensor parallelism splits every expert's width.** `moe_intermediate_size` is 640 = 5 x 128, and
the fp8 scale block, the four-bit scale group and the rotation all sit on 128-column blocks, so an
even tp2 share (320) does not work. Each parity's five blocks are cut contiguously across the
ranks, and the rank taking the extra block alternates with the parity: at tp2 a rank takes three
blocks of the experts of one parity and two of the other's (`moe_gemm_q`'s two tables in
`docs/OPS.md`). The router is replicated, each rank computes its slice of every routed expert, and
the block's existing all-reduce sums the partials: no all-to-all, no extra collective. A world
above the block count is refused at declare, since an empty slice would still join the
all-reduce. The shared expert splits as evenly as whole blocks allow: three blocks on rank 0 and
two on rank 1 at tp2; at tp4, rank 2 `[0, 256)` and one block each on ranks 3, 0 and 1.

**At four ranks the classes are four.** Parity alone would give ranks 0 and 1 three blocks of every
expert pair and ranks 2 and 3 two. So at tp4 the cut is made per class of the expert's number
mod 4: rank `r` takes blocks `[0, 2)` of class `r` and one block of each other class, five blocks
of every four experts on every rank (`moe_gemm_q_mod4`, four slice tables). A layer that keeps two
experts plain bf16 has 510 quantised ones, and its first two classes are one expert longer. The
protected bf16 experts' ten 64-wide tiles go 3, 3, 2, 2.

Expert parallelism (whole experts per rank) is not used: random routing leaves ranks whole experts
apart and waiting at the all-reduce, and it would break the parity tables, for which the recipe
protects bf16 experts in pairs.

**Attention splits by KV head.** There are 2 KV heads of 256 rows; each rank declares
`n_head_kv 1` and reads one head's row span of `k_proj` and `v_proj`. At tp2 that is the ordinary
row shard. At tp4 the heads are replicated in rank pairs (ranks {0,1} carry KV head 0, {2,3} KV
head 1; both ranks of a pair hold identical KV weights, at no extra VRAM per rank). `n_head` is 12
per rank at tp2 and 6 at tp4, so tp4 resolves a different instantiation
(`attn_{prefill,decode}_h256_gqa6_bs4`, the QSA page of 4) than tp2's gqa12 pair. A world that
neither divides nor is a multiple of the KV head count (3, 5) runs attention on the largest
multiple below it and leaves the other ranks attention-zero; `docs/TP3.md` describes this and the
uneven delta-net head split.

**Fused all-reduce forms at four ranks.** The fused forms are two-rank except on the exact wire:
`ar_hc_write_4rank` (the 4-rank one-shot with the gated-residual write as its epilogue) and
`ar_gather_hc_write_4rank` (the same with `moe_gather` as its prologue, taken by the MoE
connections when the step is one MoE pass). Both are declared over the rows of a decode or verify
step, `--max-num-seqs` x (1 + `--num-speculative-tokens`), and used whenever a step within that
bound sends its message exact: always with `--tp-wire exact`, and with `--tp-wire wht6` under
`--tp-wire-min-kb`. They run on an all-reduce instance keyed by that bound. A prefill chunk, and
every lossy message, uses the standalone sequence: `moe_gather`, `all_reduce` (4-rank one-shot or
two-shot by size, or ti8 on the lossy wire), then `hc_write_bf16`. The unfused sequence is
declared beside the fused one, so nothing is unresolved at boot. The sparse decode's fused output
gate and quantiser (`attn_paged_gate_quant`) has gqa-6, page-4 rows.

**Candidate gather.** The lm_head is vocab-sharded, so the sampler gathers the ranks' candidate
sets rather than their logits (`core/sample/sampler.cpp`), and every step at a world above one runs
an `all_gather`. libr4d serves it at 2 ranks (`r4d_ar_gather_2rank.hip`) and at 3, 4 and 8
(`r4d_ar_gather_Nrank.hip`, moving (u32 id, f32 logit) pairs without adding them). Where no device
row resolves, only libref's host gather does, and the serve is refused.

**Four-bit experts are what make it fit.** 512 experts of 640 over 48 layers is 112.5 GiB of expert
weight at E4M3 and twice that at bf16. `--weights-disk-tier` serves what fits neither VRAM nor the
host pool from the container file, but then a routed expert is read a layer at a time and a step is
paced by the drive. At four bits an expert unit is 2.42 MiB instead of 4.69, and the per-rank share
fits VRAM plus a few GiB of pinned host. `docs/MOE-W4.md` is the format.

**PLE, the n-gram embedding, is 51B of the model's 180B parameters**: a 320,001,536-row table of
160-dim embeddings, gathered 16 rows per token by a hash of the last three token ids, read once at
layer 1 and injected into the wide stream.

**QSA** picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens for each
query and attends to those alone. Below `indexer_budget + indexer_compress_ratio - 1` = 2051 tokens
of context every complete block is selected and QSA equals dense causal attention bit for bit, so
a prefill below the bound runs the cheaper dense attention, and every change is tested against
that equality. `docs/QSA.md` is the decomposition.

The vision tower is declared when the container carries one and `--mm-max-patches` is not 0
(`arch/common/rad_block_vit.h`, `spec.md` §11).

---

# Sample size

Top-1 agreement between two numeric formats is one Bernoulli trial per prompt, so at p ≈ 0.93 its
standard error is 2.8 points at N = 80 and 2.0 at N = 160: formats that differ by less than that
rank in any order. Rank with a graded statistic that has thousands of samples (the mean |delta|
over the logits row, or the per-position KL of `radiance --kld-ref`), and confirm any ordering at
N ≥ 160 before acting on it.

---

# `embeddinggemma2`: an embedding model

`arch/embgemma2` declares a pooled output (`rad_declare_pooled`) and no logits, which makes the
program an embedding program: the engine runs it with no sampler and no KV cache, carries every
input whole in one step, and returns each sequence's pooled row on `/v1/embeddings`
([GUIDE §5.6](GUIDE.md)). One card (`--tp 1`).

```
text      24 layers, hidden 512, ffn 2048 (gelu-tanh, gated), vocab 262144
          layer_types: five sliding (|i-j| <= 512, 4 x 256, 2 kv heads, theta 1e4) then one full
          (4 x 512, 1 kv head, theta 1e6), four times; BIDIRECTIONAL everywhere, scale 1
          q, k RMS-normed with a gain, v without; sandwich norms with plain gains
          per-layer inputs: the embedding projected to 24 x 512 and normed, gating a 512 bottleneck
          a learned scalar a layer on the residual; mean over every token, then 512 -> 768
vision    Gemma 4's: 16 layers of 768 (12 x 64), a learned row/column table, a two-block 2-D rotary
          (x on the first half of a head, y on the second, theta 100), 3 x 3 mean pooling * sqrt(768),
          an unscaled RMS norm and 768 -> 512; up to 280 soft tokens an image by default
audio     Gemma 4's USM conformer: two stride-2 3 x 3 convolutions over (time, mel), 12 blocks
          (half-weight feed-forwards, local attention 11 back with a relative-position term and a
          softcap of 50, a gated depthwise causal convolution), clipped linears; 1024 -> 1536 ->
          embedder -> 512; 4 frames (40 ms) a token, clips cut at 30 s
```

Traps:

- **The embedding scale is the bf16 of sqrt(512)**, 22.625, because the reference stores it as a
  bf16 buffer. sqrt(512) itself is off by about 1e-4 relative on every embedding.
- **A media row replaces its placeholder's embedding before the per-layer inputs are projected.**
  An image token's per-layer input is computed from its image row, not from the placeholder id's
  embedding.
- **The vision rotary is two independent neox rotaries**, one over each half of a head
  (`qk_norm_rope` mode `blocks`). `rope` mode `axial` pairs across the halves and is a different
  function.
- **The vision pooler stores the mean in bf16 before scaling it by sqrt(768)** (`pool_rows`). The
  processor lays each 3 x 3 window's patches out consecutively, so the pooler is a mean of nine
  consecutive rows; attention and both position terms read each patch's own coordinates.
- **The audio subsampler masks its input to the clip's real frames before each convolution**: the
  second convolution reads the first one's output at frame 2t's validity (`conv2d`'s `valid`
  operand, stride 2), which zeroes the one row past the real ones that zero padding alone would
  not.
- **The audio attention reaches 11 rows back, not 12.** The config says `attention_context_left`
  13 and the chunked layout fetches 12 keys behind each block, but transformers' mask keeps a key
  only when `0 <= q - k < context_left - 1`. Twelve back drifts a few percent per block and leaves
  rows at cosine 0.6 by the last one.
- **An audio clip's tokens are ceil(real frames / 4)**, real frames being those whose last sample
  is real; the processor stages exactly those plus zero frames to whole rows, which reproduces
  every row the reference keeps.

---

# `gemma4`: Gemma 4 text and its DFlash drafter

`arch/gemma4` serves a container converted from Google's bf16 checkpoint
(`Gemma4ForConditionalGeneration` or `Gemma4ForCausalLM`), the recipe choosing what is fp8:
gemma-4-26B-A4B-it, and the dense sizes without per-layer embeddings. The CT-INT4 31B release is
`arch/gemma4_bf16`'s. A z-lab DFlash drafter is merged at convert time (`--draft-model`); the
recipes are `data/recipes/gemma4-26b-a4b-df*.recipe`. The reference is transformers'
`modeling_gemma4`.

```
gemma-4-26B-A4B-it
  30 layers, hidden 2816, vocab 262144 tied to the embedding, rms_norm_eps 1e-6
  layer_types: five sliding_attention then one full_attention, five times
    sliding  16 heads / 8 kv x 256, rope theta 1e4 over the whole head, window 1024
    full     16 heads / 2 kv x 512, PROPORTIONAL rope theta 1e6 over a quarter of it, K = V
  attention scale 1: q and k RMS-normed per head with a gain, v with none
  sandwich norms, plain gains (x_hat * w)
  the feed-forward is two arms reading the same residual, summed:
    dense    pre_feedforward_layernorm -> gate|up 2112 (gelu-tanh) -> down -> post_feedforward_layernorm_1
    routed   router -> softmax -> top 8 of 128 -> renormalised -> * per_expert_scale[expert];
             pre_feedforward_layernorm_2 -> 128 experts of 704 (gelu-tanh) -> post_feedforward_layernorm_2
    sum -> post_feedforward_layernorm -> residual add
  layer_scalar multiplies the residual stream at the end of every layer
  embedding * sqrt(2816) as a bf16; final norm; tied head; final_logit_softcapping 30
DFlash    a small Qwen3-shaped block drafter: an anchor and block - 1 mask tokens denoised in one
          pass over the trunk's hidden states at target_layer_ids; drafts read off the TARGET's head
```

Traps:

- **Proportional rope is not a partial rotary.** The full layers rotate the first 64 frequency
  pairs of the 512-wide head at theta^(-2j/512) (the exponent divides by the head width), with pair
  j at channels (j, j + 256), and leave the other 192 pairs at frequency zero. `rope` with
  `rotary_dim` 128 pairs (j, j + 64) at theta^(-2j/128), a different function that still produces
  fluent text. The plugin declares a full-width NeoX `rope` whose frequencies past pair 63 are
  divided by 1e30 through the YaRN ramp `rope` already carries, a step at 63.5..64.
- **V on a full layer is the un-normed, un-rotated k projection**, RMS-normed with a unit gain, so
  the value norm reads `k` before the key norm writes it.
- **The router does not read the feed-forward's input.** It reads the residual stream through a
  gainless norm times its own `scale` vector times hidden^-0.5, while the experts read
  pre_feedforward_layernorm_2's output. `per_expert_scale` multiplies the weight after the
  renormalised top-k.
- **The arms' two post-norms are per arm**, so under tensor parallel each arm's partial sum is
  reduced before its norm. The partials share one [2T, hidden] buffer, dense rows first (each
  slice's own run of it when a prefill chunk is cut in two), so one all-reduce carries both.
- **The embedding scale is sqrt(hidden) rounded to bf16**: 53.0, not 53.066. The drafter reads the
  same scaled embedding.
- **The sliding layers' KV group is `RAD_KV_FULL`**, with the window on the attention op. A
  `RAD_KV_WINDOW` group is not prefix-cached, so a hit on it reuses nothing and re-prefills. The
  planner gives every paged group the same block count, so the recycling a window group would buy
  is worth nothing here.
- **The drafter's sliding layers are causal and its full layer is not**, and its logits' softcap
  is ignored on purpose: a draft is an argmax and the cap is strictly increasing.

**Tensor parallel.** Heads split by rank, o and down by column with an all-reduce, the vocabulary
by row. The full layers' two KV heads are replicated past two ranks. Routed experts are
expert-parallel, 128 / world whole experts per rank, because a 704-wide expert has no block-aligned
width split; the router is replicated, so every rank agrees on the routing without a collective.
The drafter's head is vocab-sharded with the trunk's: each rank's top-2 are all-gathered and
`pair_argmax` picks the winner (two per rank, because a row-interleaved `all_gather` moves 16-byte
words and one pair is 8 bytes).

**Prefill overlap.** At more than one rank on the exact wire, with every join fused, a prefill chunk
of two routed passes or more (4096 tokens at the 2048-token pass) runs as two slices, and the
trunk's all-reduces ride the runtime's second lane beside the other slice's compute (slice A's
attention reduction under slice B's output projection, and so on). Attention runs whole, since it
reads every row's keys. The slices issue the same kernels over the same rows, so the output equals
the whole step's bit for bit. On 4x R9700 the reductions are 47% of a 20K-token prefill under
`--profile-ops`, and the overlap is modelled at up to a quarter off; `--profile-ops` syncs every op
and cannot show it, so time a prefill without it. A step that runs whole (every decode and verify
step) puts the dense arm on the second lane instead, beside the routed arm, with its reductions on
lane 0. The startup notes say which form each step takes.

**The draft head in fp8.** The drafter's head is the widest read on the draft path: the target's
bf16 table, 352 MiB per rank at tp 4, every draft pass. A container converted with the line
`dflash.draft_head.weight rtn codes=fp8_e4m3 block=128x128 scale=bf16` (the three `-df` recipes
carry it) stores a block-fp8 copy under that name, and the head reads it at fp8a8 instead: half
the bytes, for 176 MiB per rank more. Drafts are verified, so the coarser logits can change what is
proposed, never what is accepted; `draft_n_accepted` shows whether they propose worse. Without the
copy the head reads the bf16 table, and the startup note names which head is in use.

**FP8 experts.** The experts are ~90% of the weight bytes, so each projection is served as the
container holds it, per layer: bf16 by `moe_gemm`, block fp8 by `moe_gemm_q`. gate_up
`[1408, 2816]` is whole 64-row tiles and 128-blocks. down `[2816, 704]` is not (the fp8 grouped
GEMM reads K in whole 128-column blocks), so it stays bf16 unless the checkpoint's experts were
padded to 768 before conversion (zero rows of gate and up, zero columns of down: exact;
`gemma4-26b-a4b-df-pad768.recipe`). The fp8 tables are split by expert parity, so a rank holds an
even number of experts.

**Kernels.** libr4d serves all of the trunk's attention on its windowed kernels (the sliding layers
at h256, gqa 2, and the full layers at gqa 8, 4 and 2 for one or two, four and eight ranks), plus
`pair_argmax` and the fusions:

- the two sandwich-norm joins (`norm_add_norm`, `join_add_scale`);
- `norm_add_norm_route`: on a layer whose gate_up is block fp8, the attention join also writes the
  router's input and the experts' fp8 codes from the norm it already takes, in place of an
  `rmsnorm` and an `rmsnorm_quant_fp8`;
- `expert_scale`;
- `attn_paged_qrow`, the sliding layers' attention reading q where the fused qkv projection left it.

Where nothing serves a fusion the plugin declares its unfused form (the rmsnorm / add chain,
`embed_lookup` + `mul`, a `cast` lifting q for `attn_paged`), and the startup notes say which were
taken. libref and libavx do not declare these ops, so the plugin checks `rad_op_declared` first,
and a deployment without libr4d's device rows (a CPU-only build, the host backend) runs the chains.
On a card the fused joins and libr4d's chain differ in reduction order, not bit for bit; the joins
round as the host chain does (`tests/r4d_emu`). The drafter's residual add, norm and fp8 quantiser
of x are one `rmsnorm_quant_fp8` launch per input (ten per draft pass), or the `add` / `rmsnorm` /
`quant_act_fp8` chain. The sliding layers' attention stays on the windowed kernel because only it
reads q in place; plain `attn_paged` at h256/gqa2 (the dense 31B's sliding layers) takes libr4d's
h256/gqa2 leg, which rad-kbench times at 0.26-0.71 of the windowed kernel at decode and 0.43-0.67
at prefill. A drafter head sharded over more than one rank needs `pair_argmax` on the card.

**A broken drafter still writes fluent text**, since every draft is verified; judge it by
`draft_n_accepted` (about a third of `draft_n` on prose, at about two tokens per step), not by the
reply. The buffer planner takes liveness from declaration order, so the tap copy into `dflash_src`
is declared at the end of the layer it taps, where it is issued. Declared after the trunk, it would
let the planner share `dflash_src` with `moe_edn`, which every MoE layer writes, and acceptance
would collapse silently. arch_test holds the two live ranges to overlap.

**Not declared.** The vision tower (`arch/gemma4_bf16` and `arch/embgemma2` declare Gemma 4's): a
container that carries one is served with `--mm-max-patches 0`. An fp8 attention or dense arm:
under a tenth of the weight bytes, and the dense arm's 2112 is not whole 128-blocks either.

---

# `qwenimage21`: Qwen-Image 2.1 and 2.1-Turbo, an image generator

`arch/qwenimage21` declares a generator (`rad_declare_generator`), which makes the program a
three-pass one the engine drives through `core/gen`: COND once per request (the text encoder over
the prompt and its pictures, then the DiT over that prefix into the KV cache), DENOISE once per step
(the image's latent rows against the cached prefix), DECODE once per image (the VAE). It is served
on `/v1/images/generations` and `/v1/images/edits` ([GUIDE §5.6](GUIDE.md)); fp8 on one or two
cards, bf16 on two. The checkpoint is the diffusers pipeline directory, read as one.

```
text      Qwen3-VL-8B-Instruct as released: 36 layers of 4096 (32 x 128, 8 kv heads), interleaved
          M-RoPE over (t, h, w) at 24, 20, 20 pairs, theta 5e6; the vision tower's 27 layers with
          deepstack taps 8, 16, 24 added to text layers 0..2 at the picture rows
          OUTPUT: the last layer BEFORE the final norm, without the 14 tokens of the system turn
DiT       32 single-stream blocks of 4096 (32 x 128, gqa 1), SwiGLU 12288, q and k RMS-normed, no
          biases; ONE modulation linear for every block (scale1, gate1, scale2, gate2 -- no shift,
          tanh gates) over a LayerNorm without affine; rope over (t, h, w) at 8, 28, 28 adjacent
          pairs, theta 1e4, the picture grid centred on zero; txt_in = RMSNorm (gain 1 + w),
          linear, gelu-tanh, linear; timestep embedding at t * 1000, rounded
          THE CONDITION IS A CAUSAL PREFIX: text and reference latents, modulated at t = 0,
          block-causal (text causal, a picture bidirectional within itself); written to the KV
          cache once, and a step runs only the image's rows, non-causal over prefix and self
VAE       Wan 2.2's on one frame: RGBA in and out, 64 latent channels, 16x down, decoder widths
          1152..144, one single-head 1152-wide attention in each middle block
schedule  flow-matching Euler, exponential dynamic shift (0.5 at 256 tokens to 0.9 at 8192),
          shift_terminal 0.02, 40 steps; Turbo: the 8 sigmas model_index.json lists, no guidance
```

Traps:

- **The noise is radiance's, not torch's.** A seed is a Philox stream (`abi/rad_philox.h`), element
  (k, c) of image row k drawn at k * 64 + c, identical on every card count. Parity against diffusers
  means handing diffusers the same latents; a NumPy port of the stream does that.
- **The conditioning is modulated at t = 0**, not at the step's timestep: the prefix is written once,
  and re-modulating it would differ from the reference at every step after the first.
- **The text encoder's output skips the final norm.** The normed output scales every row by the
  norm's gain and gives a plausible, wrong picture.
- **A reference picture is read twice**: by the VAE as RGBA in [-1, 1], and by the vision tower
  composited on white, both after PIL's premultiplied Lanczos resize to the 1024x1024 area at its
  own aspect, sides rounded to 32. A different resampler changes every token the picture becomes.
- **The Euler step rounds where diffusers does**: x' = bf16(x + bf16(bf16(sigma_next - sigma) * v)),
  and guidance bf16(neg + bf16(cfg * bf16(pos - neg))). Unrounded is a different trajectory, not a
  more accurate one.
- **The VAE config's `temperal_downsample` is a boolean array**; a reader that drops booleans makes
  every level look unscaled.
- **Alpha off the decoder is never exactly 255** (243..255 on an opaque picture). Read as
  transparency, it makes every picture the model made transparent when edited again; the server's
  cutoff is alpha 128, the mask's.

Against diffusers' own pipeline in bf16, from the same latents (1024x1024, 20 steps, PSNR): bf16
TP2 33.9 dB for a prompt and 42.2 for an edit, where diffusers differs from itself by 32.6 to 34.4
dB across its own attention kernels; fp8 28.7 to 29.3 and 37.6 to 38.4; Turbo fp8 33.4 and 38.9 at
its 8 steps.

---

# `qwen3asr`: speech to text

`model_type: qwen3_asr` (`Qwen3ASRForConditionalGeneration`), served from the checkpoint as it
ships: Qwen3-ASR 1.7B is a Whisper-style audio encoder, a two-layer projector and a dense Qwen3-1.7B
decoder, all bf16. The reference is transformers' `modeling_qwen3_asr` with its
`Qwen3ASRFeatureExtractor`, which core/mm reproduces (`AudioFamily::Qwen3ASR`).

```
features   Whisper log-mel, 128 bins: n_fft 400, hop 160, torch.stft centred with reflected padding,
           power spectra, the last frame dropped (n samples -> n / 160 frames), Slaney mel bank and
           area norm, log10 floored at the clip's maximum less 8, then (x + 4) / 4; a clip under
           8000 samples zero-padded to 8000 first
encoder    each second (100 frames) convolved on its own: 3 x Conv2d 3x3 / stride 2 / pad 1, 480
           channels, exact gelu, over [mel, time] -> 16 mel x 13 rows a second; 7680 -> 1024 linear
           (no bias); sinusoidal positions 0..12 added to every second's rows; 24 pre-LayerNorm
           layers, d_model 1024, 16 heads of 64, biased q/k/v/o and FFN 4096 with exact gelu,
           bidirectional in blocks of 104 rows (8 s) from the clip's start; LayerNorm
projector  1024 -> 1024, gelu, 1024 -> 2048
decoder    28 layers, hidden 2048, 16 heads over 8 KV heads of 128 (GQA 2), per-head q/k RMS norms,
           NeoX rope over the whole head at theta 1e6, SwiGLU 6144, RMS eps 1e-6, vocab 151936,
           lm_head tied to the embedding
prompt     <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|> pads
           <|audio_end|><|im_end|>\n<|im_start|>assistant\n[language English<asr_text>]
```

Traps (each loads and transcribes plausibly when wrong):

- **A second's convolution must not see the next second.** The reference pads every second to 100
  frames and convolves the batch; one convolution over the clip with no seam reads one second's
  frames into the next's. core/mm stages a whole second as 104 frames whose last four the encoder
  reads as zero (`conv2d`'s `valid`, at strides 1, 2 and 4), and the last, shorter second as 8 x
  the rows the reference keeps of it, so the rows the tower writes are the rows the reference
  keeps and nothing is compacted. The zero frames inside the last second's window are real input
  (the reference's padding) and are read: its last kept row depends on them.
- **The kernels are [mel, time]**, so PyTorch's weight is `[cout, cin, mel, time]` with time last,
  where `conv2d` convolves (time, width). `t_last` reads it as stored; reading it the other way is a
  transposed kernel, which still produces text.
- **The 7680 -> 1024 linear takes channel-major features** (`c * 16 + f`) where the convolution
  writes frequency-major ones. Declared as a 1 x 16 convolution over frequency, its weight is
  PyTorch's `[1024, 480, 1, 16]`; declared as a GEMM, it would need a permutation the name map has
  no form for.
- **Positions restart every second** (`seg_sinusoid`, period 13) and **attention runs in blocks of
  104 rows**, the last one short (`attn_dense`'s `block`). Positions counted over the whole clip,
  or attention over all of it, is a different model that still writes text.
- **The decoder's positions are the tokens' indices**: plain Qwen3, no M-RoPE; the audio
  placeholders count up like text.

A pass carries 30 s (3120 staged frames); `/v1/audio/transcriptions` cuts longer recordings into
clips at their pauses (docs/GUIDE.md §5.14). Against transformers on LibriSpeech test-clean and
test-other clips, the encoder's rows are within 1.1-1.3% relative L2 of the bf16 reference's; on
the one clip where they differed by 5%, the reference itself was 5.3% from a float32 run of the
same model and this engine 0.8%. The transcripts match word for word.

---

# `minimaxmusic3`: MiniMax-Music3, songs

`arch/minimaxmusic3` declares an audio generator (`rad_declare_audio_generator`), which makes the
program a four-pass one the engine drives through `core/gen/audio_driver.h`: PROMPT runs the
language model over prompt chunks, FRAME samples one frame of every song in flight and runs the
language model over its embedding, WINDOW steps the denoiser over a window of frames, VOCODE plays
a finished window. A song is a lane; up to four are made at once. It is served on
`/v1/audio/speech` ([GUIDE §5.15](GUIDE.md)), bf16 on one card or two. The checkpoint is the
diffusers pipeline directory, read as one.

```
LM        Qwen3-8B with its embedding and head adapted to music codes: 36 layers of 4096 (32 x 128,
          8 kv heads), theta 1e6. Two branches a song, conditional and unconditional. Each frame's
          first code: one of 16384 semantic codes (ids 151675..) or the end of the song (151670),
          from u + 1.5 (c - u); the candidates are the top 50 of c, of which the top 50 of the
          guided logits are drawn
depth     4 layers of 4096, 16 heads of 256, SwiGLU 6144, learned positions: the frame's other
          seven codes (1024 each) one at a time, each with the same guidance, the sequence
          recomputed whole at every step. The next LM input is the eight codes' embeddings summed,
          times 8^-0.5
cond      the frame's hidden planes mixed by softmax(learned logits), scaled; a window's rows
          through a 3-tap conv to 2048 channels, resampled nearest to 441/128 latents a frame
DiT       36 layers of 2048 (32 x 64), LayerNorm with affine, SwiGLU 16384 with biases, RoPE over
          the first 32 of 64 dims; input [latent 128 | zeros 128 | cond 2048] plus a 1x1-conv
          residual, the timestep token prepended, non-causal. 30 Euler steps at t = i / 30,
          guidance 1.7 against an all-zero conditioning
windows   200 frames, a new one every 100; each binds its first 172 latents to its predecessor's
vocoder   DAC's decoder: the latent's two 64-channel halves as two streams (left, right), four
          transposed convolutions (8, 8, 4, 2), dilated residual units with snake; 512 samples a
          latent, 44.1 kHz stereo
```

Traps:

- **The randomness is radiance's.** Each pick draws `rad_philox_uniform_s(seed, 2^32 + frame,
  sub-step)` by inverse CDF in descending guided logit (ties to the lower index), and window k's
  noise is `rad_philox_normal_s(seed, 1 + k, row * 128 + c)` (`abi/rad_philox.h`). Parity against
  diffusers means handing diffusers the same picks and noise, which a NumPy port of the streams
  does.
- **The unconditional prompt is the conditional one with its middle blanked**: its first token and
  last two kept, every token between them the filler 151654, so both branches have one length.
  An empty or shortened prompt is a different guidance and still sings.
- **The first frame after the prompt is consumed, not played**: it only moves the language model
  past the prompt, and no conditioning row is stored for it.
- **The DiT's positions start at the timestep token**: position 0 is that token and the latents
  follow from 1. Starting the latents at 0 shifts every rotation by one and the song still plays.
- **A window's join runs in one order**: its head is set from the previous window's kept tail
  first, and only then is the next window's tail taken from the joined latent. The other order
  binds a window to its own tail, and the song drops in level for a moment at every join.
- **The model card says 32 kHz; the vocoder plays 44.1 kHz** (512 samples a latent), as diffusers'
  port and its config do.
- **Under tensor parallelism, a fused projection's stacked bias is cut per part**: a rank-1
  `[value | gate]` vector holds its share of each half, as the matrix it goes with does. A
  contiguous slice of the stack runs and is 5-6% off in every DiT velocity.

Against diffusers' own pipeline in bf16, from the same picks and noise: with radiance's codes
teacher-forced into the reference, the language model's last hidden row is 1.55% from the
reference's (relative L2; the reference is 1.27% from itself across bf16 and f32), the window
conditioning 1.33% (1.10%), a DiT step's velocity 1.3% (1.8%). At one card a 4 s song is 29.0 dB
SNR from the reference's audio, against 28.7 dB between the reference's own attention kernels; at
two cards a 12 s song of two windows is 26.3 dB (32.0), and the 4 s song 20.8 dB, one passage of it
having taken another turn, which a rounding-level change makes in the reference too. A pick is chaotic: with the same history,
the reference and radiance choose differently at 318 of 808 sub-steps, and the reference chooses
differently from itself across precisions at 314.

---

# `breeze`: Breeze-TTS-2, speech

`arch/breeze` declares an audio generator in its chunked form (`rad_declare_audio_generator` with
`voc_chunk`): PROMPT runs the text encoder and the backbone over both branches' whole prompts,
FRAME samples one frame of every speech in flight and runs the backbone over its embedding, VOCODE
plays a lane's newest frames, four at most, as soon as they are made. It is served on
`/v1/audio/speech` and `/v1/audio/speech/stream` ([GUIDE §5.15](GUIDE.md)), bf16 on one card. The
checkpoint is the release directory; its codec, in `audio_tokenizer/`, is read with it.

```
text      T5Gemma 2's text encoder in Gemma 3 1B's shape: 26 layers of 1152, 4 query heads over one
          KV head of 256 (q/k RMS-normed, scale 1/16), sandwich RMS norms with a (1 + w) gain,
          gelu-tanh GLU 6912, the embedding times 34.0 (bf16 of sqrt 1152). BIDIRECTIONAL over the
          whole prompt: sliding layers see 255 keys behind and 256 ahead (theta 1e4); layers 5,
          11, 17 and 23 see everything (theta 1e6, positions divided by 8). Final norm, then
          1152 -> 2048
backbone  Qwen3: 28 layers of 2048, 16 heads over 8 KV heads of 128, q/k norms, theta 1e6, eps
          1e-6. Its prompt rows ARE the encoder's projection; a frame's input is the bf16 of the
          f32 sum of its 16 codes' rows of one table [16 x 2051, 2048]. Head 2052: 2048 codes,
          three never drawn, the end of the speech
depth     12 layers of 1024, 8 heads over 2 KV heads of 128, Llama 3's rotary (theta 5e5, factor
          32, original length 16, low 1/512, high 1/128), no q/k norms: position 0 is the
          backbone's hidden state, position p code p - 1's row of the table, both through one
          2048 -> 1024 projection; 15 heads, one a codebook
sampling  u + g (c - u) against an unconditional prompt without the voice description (g 1, the
          default: none); top 50 at temperature 0.9; the first code also takes a repetition penalty
          of 1.1 over the speech's earlier first codes
codec     the Qwen3-TTS 12 Hz tokenizer's decoder: EMA codebooks (the first code's, the other
          fifteen's) projected 256 -> 512 and summed; a causal conv k3 to 1024; 8 layers of 512
          (16 heads of 64, rotary at the frame's index, causal over 72 frames, layer scales);
          twice a transposed conv k2 s2 and a ConvNeXt block (causal depthwise k7, LayerNorm,
          exact GELU over 4096); a causal conv k7 to 1536; four blocks of SnakeBeta, a causal
          transposed conv (8, 5, 4, 3) and three residual units (dilations 1, 3, 9); SnakeBeta and
          a conv k7 to one channel: 1920 samples a frame, 24 kHz mono
```

Traps (each still speaks, wrongly):

- **The backbone's configuration is `backbone_config`.** The top level's Llama 3 rotary at theta
  5e5 and eps 1e-5 belong to a native backbone the release does not build; the Qwen3 backbone runs
  at theta 1e6 and eps 1e-6.
- **The backbone never embeds text.** Every prompt row is the encoder's output projected; the
  checkpoint's `embed_text_tokens` (0.5B parameters) is unused, and so is the Mimi codec in the
  main file.
- **The encoder reads the whole prompt at once**, so a prompt is never split across passes
  (`prompt_whole`), and its sliding window is lopsided: transformers' mask at 512 sees 255 keys
  behind and 256 ahead. A symmetric window differs on prompts over 256 tokens.
- **The depth heads are stored (in, out)**, `[15, 1024, 2051]`, and are transposed at convert
  (`RAD_MAP_TRANSPOSE`); the codec's codebooks are EMA sums, divided by their usage at convert
  (`RAD_MAP_EMA`).
- **A frame's sixteen embeddings are summed in f32 and rounded once**, as torch sums a bf16 tensor;
  rounding after each would move every backbone input.
- **The codec's first two transposed convolutions have a kernel of one stride**, the four after
  them of two, all causal: the full output's last rows dropped, no padding in front.
- **The reference's streaming codec is not a reference.** Its 72-slot KV ring shifts with an
  overlapping in-place copy that races on the GPU: three identical decodes agree at 20-26 dB.
  Compare against the offline decode, which radiance's chunks reproduce: every codec layer is
  causal, so a chunk decoded again over the 85 frames its outputs reach back to is exact. The
  reference's own codec decoded on radiance's schedule is 117 dB from its whole decode (f32
  rounding); with 60 frames of context it is 30 dB.

Against BreezeBlue's eager reference in bf16, teacher-forced with radiance's codes (a CPU run):
the backbone's hidden rows are 1.04% from the reference's (relative L2, median over the prompt and
72 frames; the reference is 0.94% from itself across bf16 and f32). The reference would have drawn
another code at 323 of 1168 picks, all near-ties: the code radiance drew had a median 0.998 of
the reference's own pick's probability, and the first code differed once in 73 frames. Through
the codec alone, radiance's bf16 audio is 38.4 dB SNR from the reference's f32 decode of the same
codes. Whisper large-v3 transcribes radiance's speech word for word.

---

# `minimaxh3`: MiniMax-H3, video with sound

`arch/minimaxh3` declares a video generator (`rad_declare_video_generator`), which makes the
program a four-pass one the engine drives through `core/gen/video_driver.h`: VCOND runs the text
encoder and the refiner over the prompt and computes every step's modulation, VSTEP is one
denoising step of one request's packed sequence, ADECODE plays the soundtrack, VDECODE decodes
one temporal chunk of frames. A request is a lane; two hold state at once (one denoising, one
conditioned and waiting). It is served on `/v1/videos` ([GUIDE §5.16](GUIDE.md)), block fp8 on two
cards with `--vision-tower-placement ram`. The checkpoint is the diffusers pipeline directory,
read as one; only workflow t2va is served.

```
text      Qwen3-VL-32B's language model: 64 layers of 5120 (64 q / 8 kv heads of 128, ff 25600,
          eps 1e-6, theta 5e6, interleaved M-RoPE, which is NeoX for text). The conditioning is
          the hidden state AFTER LAYER 50, before any norm; layers 50..63 are never read and the
          container leaves them out. The prompt is tokenised as written, no template, no special
          tokens
refiner   context_embedder 5120 -> 5376 (bias), two plain pre-norm blocks over the prompt (full
          attention, no rotary, q/k RMS norms), a final RMSNorm
DiT       50 blocks of 5376 (56 heads of 128), SwiGLU 14336, RMSNorm eps 1e-5, no biases, one
          packed sequence [text | soundtrack ch0 | soundtrack ch1 | video] under full attention.
          Each row is modulated by its modality (0 video, 1 text, 2 audio) at its timestep (text
          rows take the video's): six vectors a block from Linear(silu(temb), 3 x 6 x 5376).
          norm_out's shift and scale are per timestep. Video latents 24 channels in 1x2x2 patches
          (96 values a row, proj_in f32), the soundtrack's 32 channels a row (f32). Guidance-
          distilled: one forward a step
rotary    3 axes (time, row, column), 16 frequencies each (theta 1e4) over the first 96 of 128
          dims, rotate-half pairs (i, i + 48); f32 positions
schedule  per modality: sigma = s x / (1 + (s - 1) x) over linspace(1, 0, N) (N grid points, N - 1
          evaluations), shift 12 for the video and 3 for the soundtrack; t = 1 - sigma; x0 =
          x + (1 - t) v, then x' = r x + (1 - r) x0 with r = sigma_next / sigma
video     ViT decoder, 36 layers of 2048 (32 heads of 64), f32 residual, RMSNorm, q/k RMS norms
VAE       without weights, rotary 8/8/8 over 48 of 64 dims (theta 100, angles 2 pi (2 (i + .5) /
          n - 1)), layer scales, SwiGLU 8192 with biases, 4 registers and a zero token a tile.
          A token is a latent: 4 frames x 16 x 16 pixels. Tiles of 16 x 16 latents, overlap 4 at
          least; chunks of 7 latent frames every 5. Pixels come out ImageNet-normalised
audio     dec_in_proj 32 -> 2048, BigVGAN: conv_pre k7 to 1024, 7 transposed convolutions (5 5 2 2
VAE       2 2 2), each followed by the mean of three AMP blocks (k 3/7/11, dilations 1/3/5) with
          anti-aliased SnakeBeta (x2 up, x2 down, 12-tap Kaiser filters); conv_post k7 to one
          channel, clamped. 800 samples a latent, 32 kHz, each stereo channel its own sequence
length    24 fps; a video of n chunks is 17 n + 5 frames over 5 n + 2 latent frames, n 7..20
          (5.17 to 14.4 s); the soundtrack is round-half-even(frames x 40 / 24) latents a channel
```

Traps:

- **The feed-forward's value half comes first** in the checkpoint (`[value | gate]`, out =
  `value * silu(gate)`), in the DiT, the refiner and the video decoder alike. The halves are
  swapped at convert into the `[gate | value]` that `silu_mul` reads.
- **The modulation gate has no tanh**, and the residual is `h += gate * branch`. The timestep
  embedding reads t in [0, 1] as it is, not scaled to 1000.
- **The grid's second half is linspace's from the end**: torch computes `linspace(1, 0, N)` half
  from each end in f32, which `gen::video_schedule` reproduces.
- **The packed rows' positions are f64 rounded to f32**: a latent frame spans 5/3 x (1, 4, 4, 4,
  4)[i mod 5] time units from `n_text`, the spatial axes are each side's latents over the square
  root of the frame's area, centred on the unit interval, times 32, and the soundtrack's two
  channels sit at the two ends of the width axis.
- **A tile is blended with its neighbours as they were decoded**: the vertical blend reads the
  original tile above, and the horizontal blend the original tile to the left, not the blended
  ones. Chunks keep frames [3, 20) of their 28 and blend their last 5 linearly into the next
  chunk's first 5.

Tensor parallel: the text encoder, refiner and DiT blocks split by heads and feed-forward width.
A block's three modulation projections split by rows, three of the six vectors a rank, and
VCOND writes every step's modulation into the lane once a request, so the 13B modulation
parameters and the text encoder (RAD_ACCESS_PER_REQUEST) live in pinned host memory under
`--vision-tower-placement ram`. Each step gathers its row of the table across the ranks. The video
decoder is replicated and splits its tiles between the ranks, gathering the decoded pixels; the
audio decoder runs whole on each.

Against diffusers' pipeline in bf16 from the same noise, a 256 x 256 clip of 4 grid points: the
fp8 text encoder's conditioning is 4.15% from the reference's (relative L2). With radiance's
conditioning given to both, the first evaluation's velocity is 5.4% (video) and 4.3% (soundtrack)
from the reference's, at slope 0.99 and cosine 0.9985; the reference itself, with radiance's
block-fp8 rounding emulated on the same linears, is 6.8% and 4.6% from its own bf16 and shows the
same per-channel bias. Fed the same latents, the video decoder is 36.9 dB PSNR from the
reference's frames (H.264 included) and the audio decoder 51.0 dB SNR from its samples. The same
seed gives the same latents, run to run.

Attention precision: the DiT's attention takes its scores in int8 (the keys quantised per row and
head by `quant_act_i8g`, the query inside the op) and its P.V in fp8 (the values quantised per
column by `quant_cols_fp8`, the weights `e4m3(448 p)` against an exactly tracked row max), so both
halves run on the 8-bit matrix instruction. Along an exact trajectory in diffusers (256 x 256), the
int8 scores move the velocity 1.9-5.9% and the fp8 P.V another 1-3 points, where exact f32
softmax and SDPA's bf16 flash kernel already differ by 1.7-4.4%; finished clips keep their shot,
motion and detail.

## FastH3: eight trained steps with video sparse attention

FastVideo's FastH3 releases (FastVideo/FastVideo-FastH3-8-Step-V2) are the same denoiser distilled
(DMD2) to eight evaluations with video sparse attention, and name both in their
`fastvideo_inference.json`, which the converter reads under `fastvideo_inference.` keys. The
plugin takes from it:

```
schedule  the trained timesteps 999 874 749 624 500 375 250 125 over a 1000-step clock, each
          shifted per modality (10 for the video, checked against scheduler_config.json, 3 for the
          soundtrack), then 0: nine grid points, eight evaluations, no other count
          (RadVideoGenDecl::rungs)
attention VIDEO_SPARSE_ATTN_H3 at sparsity 0.8 and 64-row tiles: the prompt's and the
          soundtrack's rows in 64-row chunks of their own (dense: their queries see every key, and
          every query sees them), the video's patches in 4 x 4 x 4 boxes of (time, row, column);
          a video query tile keeps the ceil(0.2 V) video tiles of best mean-q . mean-k score, per
          head. A fourth projection beside q, k and v (attn.to_gate_compress, appended to the
          fused qkv) gates in an attention over the tiles' means: vsa_select, then attn_vsa
          (docs/OPS.md)
name      served as `fastvideo-fasth3-8-step-v2`
```

Its text encoder and both decoders are MiniMax-H3's files byte for byte, so a container of the base
model supplies them (`--reuse`, the transformer's tensors listed as `--changed`; see the recipe's
header). The distilled model's first evaluation moves 21-23% under any perturbation (the reference
itself between bf16 and emulated fp8, or with the fp8 text encoder's conditioning), so its per-step
numbers against the reference say little; radiance's frames and the reference's agree by eye.

## The latent upscaler

LBH-123-AI's `Minimax_h3_latent_Upscaler` (v1, `LatentResizer3D`, 345M parameters, Apache-2.0) is
read as a companion folder of either pipeline, `latent_upscaler/` (its config.json and bf16
weights), and declared when the container carries it (RadVideoGenDecl::upscale_max is then its
largest factor). The architecture is the ComfyUI node's; the config states the same numbers:

```
input     the clean small latent's voxels (patch rows unfolded), normalised a second time with the
          upscaler's own per-channel mean and deviation (the node's (x - mean) / std on ComfyUI's
          already-normalised latent), bf16
factor    (W / W' + H / H') / 2; Linear(1, 64) of (factor - 1), SiLU, Linear(64, 64), in f32;
          each residual block's SiLU and Linear(64, 1024) gives its (scale, shift)
network   conv_in 3x3x3 24 -> 512; 18 blocks at the small size; F.interpolate(trilinear,
          align_corners=False) to the full size with the frame count kept, which is bilinear on
          each frame (resize_bilinear); 18 blocks at the full size; GroupNorm(32), SiLU,
          conv_out 3x3x3 512 -> 24. A block list is 12 residual blocks, a temporal block after
          the 1st, 3rd, .. 11th
residual  GroupNorm(32) SiLU conv3d; GroupNorm(32) * (1 + scale) + shift, SiLU, conv3d; + x
          (group_norm, conv3d_ndhwc with the residual added in its store)
temporal  GroupNorm(32) SiLU, a depthwise (5, 1, 1) convolution over frames (tconv_dw), a 1x1x1
          one (gemm_nt_bias with the residual)
output    denormalised, folded back into patch rows (patch_fold), then the video latent and the
          soundtrack's re-noised to the first full-size step's sigmas with stream 1 of their seeds
          (flow_renoise)
```

The node runs the network on whole clips here (its temporal chunking is off in the example
workflows); every rank runs it whole, its weights in VRAM on each.
