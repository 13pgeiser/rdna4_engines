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
| `arch/qwen35_w8a8` | `qwen35` | `w8a8_fwht128` | `qwen35_fp8` with an int8 trunk: i8 weights with a bf16 scale per 128 of K, rotated by a 128-wide Hadamard, against int8 activations under the same rotation (`data/recipes/q38-27b-w8a8-df2.recipe`) |
| `arch/qwen35moe_fp8` | `qwen35moe` | `fp8_e4m3` | `qwen35_fp8` with the feed-forward routed |
| `arch/qwen4exp_fp8` | `qwen4exp` | *(empty)* | Qwen4-Exp (Qwen3.8-Flash-Next), a bf16 checkpoint quantised at convert |
| `arch/glm5next` | `glm5next` | *(empty)* | GLM-5.3-Flash from AMD's Quark MXFP4 export: four mHC residual streams over 34 KDA and 11 sparse latent-attention (MLA) layers with a DSA indexer, 288 MXFP4 experts at top-8 under sigmoid routing, block-FP8 attention and dense layers at f32 scales, each weight served as stored; its MTP layer is a serial drafter. Pictures through its vision tower (`arch/common/rad_block_glm_vision.h`); bf16 KV |
| `arch/lfm2_bf16` | `lfm2` | *(empty)* | LiquidAI's LFM2 and LFM2-VL, bf16, one rank |
| `arch/embgemma2` | `embeddinggemma2` | *(empty)* | EmbeddingGemma 2: a bidirectional encoder with Gemma 4's vision tower and audio conformer, pooled to one vector |
| `arch/gemma4_bf16` | `gemma4` | `compressed-tensors` | Gemma 4 31B dense with its vision tower: sliding-window and proportional-RoPE global (`attention_k_eq_v`) attention, GELU ffn, softcapped logits; int4 codes with an f16 scale per 128 from Google's CT-INT4 (QAT) release or from bf16, served w4a8, or int8 codes on the same grid from bf16 (w8a8, the Q8 container). Optional draft model (`rad-convert --draft-model`): Google's assistant, with no cache of its own (`rad_drafter_shared_kv`), or z-lab's DFlash drafter, the block `arch/gemma4` serves (`arch/common/rad_gemma4_dflash.h`), or RedHatAI's DFlash2 speculator (`arch/common/rad_block_dflash2.h`) |
| `arch/gemma3` | `gemma3` | *(empty)* | Gemma 3 dense text (TranslateGemma 27B) from the bf16 checkpoint: sliding-window and global attention at one head geometry, GemmaRMSNorm, GELU ffn; linears bf16, or int8, MXFP4 or asymmetric int4 over an activation scaled per (row, 128), as the recipe stores them. Optional DFlash2 drafter (`rad-convert --draft-model`) |
| `arch/gemma4` | `gemma4` | *(empty)* | Gemma 4 text from Google's bf16 checkpoint (the 26B-A4B MoE and the dense sizes without per-layer embeddings) with z-lab's DFlash drafter; routed experts bf16 or block fp8 |
| `arch/qwenimage21` | `qwenimage21` | *(empty)* | Qwen-Image 2.1 and 2.1-Turbo, an image generator, from the diffusers pipeline directory; DiT and text encoder bf16 or block fp8 |
| `arch/krea2` | `krea2` | *(empty)* | Krea 2 Raw and Turbo, an image generator: Qwen3-VL-4B's tapped layers through a text-fusion transformer, a 12.9B single-stream DiT and Wan 2.1's VAE, from the diffusers pipeline directory; bf16 or block fp8 |
| `arch/sdxl` | `sdxl` | *(empty)* | Stable Diffusion XL and its finetunes (Illustrious, NoobAI, Pony), an image generator: CLIP ViT-L and OpenCLIP bigG text encoders, the 2.6B UNet sampled by k-diffusion's Euler, Euler ancestral or DPM++ 2M on ComfyUI's schedules, the SDXL VAE; from the single-file checkpoint, bf16, one card |
| `arch/flux3` | `flux3` | *(empty)* | FLUX.3 [klein] and its base, an image generator and editor: Qwen3-VL-4B's tapped layers, a 5.6B three-mode DiT over the text, up to ten reference pictures and the image, FLUX.2's VAE; bf16 or block fp8 |
| `arch/anima` | `anima` | *(empty)* | Anima (Base, Aesthetic, Turbo), an image generator: Qwen3-0.6B-Base through the LLM adapter over the prompt's T5 tokens, Cosmos-Predict2's 2B cross-attention DiT and the Qwen-Image VAE, from the release's single files; bf16 or block fp8, one card |
| `arch/flux3distilled` | `flux3distilled` | *(empty)* | the same plugin under the name FLUX.3 [klein]'s step-distilled release gives itself (`Flux3DistilledModularPipeline`; its base is a `Flux3ModularPipeline`) |
| `arch/qwen3asr_bf16` | `qwen3asr` | *(empty)* | Qwen3-ASR: a Whisper-style audio encoder over a dense Qwen3 decoder, speech to text |
| `arch/minimaxmusic3` | `minimaxmusic3` | *(empty)* | MiniMax-Music3, songs from a description and lyrics: a Qwen3-8B frame model, a depth decoder, a flow-matching DiT and a DAC vocoder, from the diffusers pipeline directory, bf16 |
| `arch/breeze` | `breeze` | *(empty)* | Breeze-TTS-2, speech from text and a description of the voice: a T5Gemma 2 text encoder, a Qwen3 backbone with a 16-codebook depth decoder and the Qwen3-TTS 12 Hz codec, from the release directory, bf16, one card |
| `arch/minimaxh3` | `minimaxh3` | *(empty)* | MiniMax-H3, a video with its stereo soundtrack from text: Qwen3-VL-32B's language model as the text encoder, a 33B single-stream diffusion transformer, a tiled ViT video decoder and a BigVGAN audio decoder, from the diffusers pipeline directory; linears block fp8 (`data/recipes/minimax-h3-fp8.recipe`) |
| `arch/ltx2` | `ltx2` | *(empty)* | LTX-2.5 (base and distilled), a video with its stereo soundtrack from text or from text and a first frame: Gemma 4's language model and two text connectors, the duration head, a dual-stream audio-video DiT with four-way guidance, the causal video VAE (tiled past the default canvas) or the diffusion decoder, the audio VAE with a BigVGAN vocoder and bandwidth extension, the x2 latent upsampler; bf16 or block fp8, one card |

The traps below are readings that are wrong without any error: the model produces fluent output.

## Common rule

**`head_dim` is read, never derived.** On the Qwen hybrids `head_dim` is 256 while
`hidden_size / num_attention_heads` is 213, and a derived value silently breaks the projection
widths, the KV page size and the attention kernel's band. `core/engine_bringup.cpp` takes it from
the container's meta.

## Adapter key names of the generators

A generator that serves LoRA files ([GUIDE §4.9.2](GUIDE.md)) declares, before its GEMMs, how
those files name its modules (`rad_lora_root`, `rad_lora_alias`; `lora_names` in
`arch/common/rad_arch.h`). Files follow the model's original code or ComfyUI's port of it, while
containers keep the diffusers names, and the two disagree in ways that load without an error and
generate garbage: a fused `qkv` against separate q, k and v, a SwiGLU projection whose gate half
comes first in one and second in the other. The tables, by plugin:

| plugin | adapter files | checkpoint |
|---|---|---|
| `krea2` | `blocks.N.attn.wq/wk/wv/wo/gate`, `mlp.gate/up/down`, `txtfusion.{layerwise,refiner}_blocks.N.*`, `first`, `last_linear`, `tmlp.0/2`, `tproj.1`, `txtmlp.1/3` | `transformer_blocks.N.attn.to_q/...`, `ff.*`, `text_fusion.*`, `img_in`, `final_layer.linear`, `time_embed.linear_1/2`, `time_mod_proj`, `txt_in.linear_1/2` |
| `qwenimage21` | the checkpoint's names, or `img_mlp.gate_up` = [gate; up] | `img_mlp.gate_layer`, `img_mlp.proj` |
| `minimaxh3` | `blocks.N.attn.qkv_proj` (q, k, v), `out_proj`, `mlp.fc1` = [gate; up], `fc2`; `token_refiner.blocks.N.*`; `video_patch_proj`, `audio_patch_proj`, `condition_proj`, `time_embedder.proj_in/out`, `final_layer.video_out/audio_out` | `attn.to_q/k/v`, `to_out.0`, `ff.net.0.proj` = [up; gate], `ff.net.2`; `token_refiner.refiner_blocks.N.*`; `proj_in`, `audio_proj_in`, `context_embedder`, `time_embedder.linear_1/2`, `proj_out`, `audio_proj_out` |
| `flux3` | `single_blocks.N`, `txt_mode_blocks.N`, `content_mode_blocks.{image,image_cond}.N` with `q_proj/k_proj/v_proj` or `qkv`, `attn_out`, `mlp_in`, `mlp_out` | `single_transformer_blocks.N`, `txt_mode_blocks.N`, `mode_blocks.{image,image_cond}.N` with `attn.to_q/k/v`, `attn.to_out.0`, `ff.linear_in/out` |
| `ltx2` | the checkpoint's names | |
| `anima` | the checkpoint's names behind `diffusion_model.` / `lora_unet_`; the text encoder as `lora_te_layers_N_...` | DiT under `net.` or `model.diffusion_model.`, Qwen3 under `model.layers.` |
| `sdxl` | kohya's `lora_unet_` over the single-file UNet paths (convolutions included), `lora_te1_` / `lora_te2_` over the CLIPs' transformers names | UNet under `model.diffusion_model.`, CLIP-L and bigG under `conditioner.embedders.{0,1}.transformer.` |

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
| full attention | `RAD_KV_FULL` | 16 layers, head_dim 256, 4 KV heads; block size from the resolved kernel. `--kv-cache-dtype fp8_v4` / `k6_v4` (`qwen35_fp8`; `qwen4exp_fp8` too) declare it at head_dim 200 / 176, the bytes of a token's quantised row over two (docs/OPS.md) |
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

**Drafters.** The container's own MTP head is the default, and `rad-convert --draft-model` can
replace it with a DFlash2 block drafter; the shared source declares it (`declare_dflash2`) and
does not declare the MTP head when a DFlash2 drafter is present, so the routed trunk carries a dense drafter
without further work. `sage-a1.5-fp8-df2.rad` is such a container: Sage-A1.5's routed trunk with
`incoai/Qwen3.6-35B-A3B-DFlash2`, whose six 2048-wide layers read eight taps
(`target_layer_ids` 1 6 11 16 22 27 32 37) and propose a block of seven tokens in one pass. At
depth 7 it accepts ~49% of the proposals and decodes 1.40x faster than the same trunk with its MTP
head (393 against 282 tok/s on two R9700s). `--num-speculative-tokens auto` takes the depth the
drafter states.

## `qwen35_w8a8`

The dense build again, behind `#if QWEN35_W8A8`, for a container `data/recipes/q38-27b-w8a8-df2.recipe`
converts from the bf16 checkpoint: the trunk's linears (the set Qwen's FP8 release quantises) as
`i8*bf16[1x128]/fwht128`, each 128 of K rotated by the unnormalised Hadamard before rounding, served
by `gemm_nt_q` at `i8a8h`. The descriptor `w8a8_fwht128` is what the convert's `--set` keys state;
a probe of the first delta-net input projection holds the weights to it before the activations are
declared, since only an int8 trunk gives them an int8 code pair. The descriptor names the rotation, so
a plain int8 trunk is refused.

- **The 1/128 is the weight's.** `had_quant_act_i8g` rotates the activation and folds nothing, as
  `had_quant_act_fp8` does; the per-row `had_*_i8` forms fold 1/sqrt(128) into the activation's
  scale and against this weight give an output sqrt(128) too small.
- **The encoding is the only guard.** Rotated and plain int8 have the same bytes and extents, and
  the rotated weight against the plain quantiser measures rel_l2 ~1.0 (r4d_selftest, `i8a8h_rot`),
  so each dtype's layout hook takes only its own transform.
- **Each rotated linear quantises its own input** from the bf16 plane its producer writes: q, k and
  v quantise one input three times, and no producer writes rotated codes yet. The MLP's gated fold
  is an fp8a8 tile over E4M3 codes, so an int8 gate_up runs the pair.
- **It declines imported checkpoints** (`RAD_ARCH_SERVES_IMPORTS(0)`). An imported quantisation
  goes to the family's one quantised plugin, and a second one would refuse every Quark, ParoQuant,
  GPTQ or NVFP4 Qwen3.8 release; they stay with `qwen35_fp8`.

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

**The attention cache takes `--kv-cache-dtype fp8_v4` or `k6_v4`.** The QSA layers' cache (and the
MTP head's layer, which shares it) is then declared at head_dim 200 or 176, a token's quantised row
over two, and read by the gqa12 rows (gqa6 with 4-token pages at four ranks) of those layouts
(docs/OPS.md). The indexer's key cache and tail keep their widths.

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

# `glm5next`: GLM-5.3-Flash

`arch/glm5next` serves the language model of GLM-5.3-Flash (`model_type: glm5_next`) from AMD's
Quark export, `amd/GLM-5.3-Flash-Quark-MXFP4`. Quark is an import path, not an execution format:
the plugin claims the empty descriptor and reads each weight's encoding from the container
(`data/recipes/glm53-mxfp4.recipe` changes one weight a layer, below). Its vision tower is
declared beside it when the container carries the tower and the deployment asks for an encoder
(below). The reference is SGLang's `glm5_next.py` and `glm5_next_nextn.py` at AMD's pinned commit
654df43; for the tower and its processor, transformers' `glm5_next`.

The plugin is one file over blocks in `arch/common`, one block a file; it declares the buffers
the blocks share, places them and issues the step's order.

| file | block | what |
|---|---|---|
| `rad_block_glm_mhc.h` | `ManifoldConn` | the four-stream residual's read and write |
| `rad_block_glm_kda.h` | `GlmKda`, `KdaState` | the 34 linear layers and their recurrence |
| `rad_block_glm_mla.h` | `GlmMla`, `MlaLatent` | the 11 sparse MLA layers and their absorbed core |
| `rad_block_glm_indexer.h` | `GlmIndexer`, `DsaCache` | the DSA indexer, its key cache and selection |
| `rad_block_glm_router.h` | `GlmRouter` | the correction-bias router, through `MoeFP8` |
| `rad_block_glm_mtp.h` | `GlmMtp` | the draft head around the plugin's layer 45 |
| `rad_block_glm_vision.h` | `GlmVision`, `MediaRows` | the tower, and its rows over the embeddings |
| `rad_glm_tp.h` | `GlmTp`, `SplitF32Proj` | the shards, the split threshold, the column split of a replicated f32 projection |

```
45 layers, hidden 4096, vocab 154880 (embedding and lm_head separate, bf16)
mHC: 4 residual streams, a 16384-wide row; 20 Sinkhorn passes, hc_eps 1e-6; the embedding is
  copied into every stream, and the mean of the streams feeds the final norm
layers 3, 7, ..., 43 (l % 4 == 3): sparse MLA, no RoPE
  q_a 4096 -> 1536, RMS-normed; kv_a 4096 -> 512, RMS-normed: the cached latent
  64 heads, q/k and v head_dim 256: q_b 1536 -> 16384, kv_b 512 -> 64 x (256 + 256),
    o 16384 -> 4096, scores scaled by 1/sqrt(256), absorbed (attention over the latent)
  DSA indexer: 32 heads of 128 from the normed q latent (wq_b 1536 -> 4096); a key of 128
    (LayerNorm, eps 1e-6, affine) and 128 pool-gate logits from the layer input; 4 keys
    pooled to 1, Hadamard-rotated, E4M3; head gates f32 (32 x 4096), signed;
    index_topk 2048 = 512 blocks of 4 tokens
the other 34 layers: KDA, 64 heads of 128
  q, k, v 4096 -> 3 x 8192, causal depthwise conv of width 4 + SiLU, q and k L2-normed,
    scores scaled by 1/sqrt(128)
  forget gate per channel through rank 128 (4096 -> 128 -> 8192), lower bound -5;
    beta per head (4096 -> 64); output gate through rank 128, sigmoid-gated RMS norm per head
feed-forward: layers 0-2 dense SwiGLU 2 x 12288; layers 3-44 MoE
  288 routed experts of 2048 at top-8, sigmoid scores, correction bias (noaux_tc, one group),
    weights normalized and scaled by 2.5; one shared expert of 2048 without a gate
  every SwiGLU clamped at 10
MTP: checkpoint layer 45 -- enorm, hnorm, eh_proj 8192 -> 4096, one MLA + indexer layer with
  MoE, shared_head.norm; no mHC; the trunk's embedding and lm_head
max_position_embeddings 1048576
```

Traps:

- **`comb` is indexed `[source, destination]`.** Each connection's write is
  `out[dst] = post[dst] * y + sum_src comb[src, dst] * h[src]`; the transpose has every shape
  valid and is another model. `out` must be a separate buffer: each output stream reads every
  input stream (`rad_block_glm_mhc.h`).
- **The router and the index head gates stay f32.** Both projections are f32 weights against a
  bf16 activation with f32 results (`gemm_nt_f32`), and the routing weights reach the gather
  as f32. Rounding any of them to bf16 changes which experts and which blocks are picked, and a
  weight rounded before the gather's sum can cancel to 0 where f32 keeps 0.125 (the libref
  fixture). The correction bias ranks the experts only; the mixture weights are the selected
  unbiased sigmoids.
- **The index scores are signed.** The ReLU comes before the head weighting, and a gate can be
  negative, so a block's score can be. `qsa_select` runs with `signed` 1; QSA's ranking clamps
  at zero and picks another set whenever a live score is negative (docs/OPS.md).
- **The SwiGLU is clamped, asymmetrically.** The gate is clipped above at 10 before the SiLU and
  not below; the up projection on both sides. The unclamped FP8 epilogue of the dense
  feed-forward does not apply, and `MlpFP8` turns it off when a limit is given.
- **Quark's excluded layers are block FP8, not bf16.** The export is MXFP4 globally and names
  the attention and MTP layers in `layer_quant_config` as E4M3 with an f32 `weight_scale` per
  128 x 128, multiplicative. Read as bf16-scaled `fp8a8`, a scale bf16 cannot hold would round;
  they are served `fp8a8f32`. The importer accepts an override only if it is MXFP4 or this
  block FP8 and refuses any other, a null or empty one included, by name.
- **`head_dim` is 0 in the config.** The KDA and MLA widths come from their own keys; the
  plugin fixes 128 and 256 and refuses a checkpoint whose hidden size, layer, vocabulary,
  stream, expert, `index_topk` or MTP layer counts are not these.
- **MTP concatenates the normalized embedding first,** the normalized previous hidden second.
- **No RoPE anywhere** (`qk_rope_head_dim` 0); the indexer's only rotation is the Hadamard.

**KV groups.** Five, all bf16 or wider; a `--kv-cache-dtype` other than bf16 is refused at
declare.

| group | kind | per layer | layers |
|---|---|---|---|
| `kv_latent` | `RAD_KV_FULL`, bf16, 1 head of 256, block 4 | the 512-wide normed latent, its two halves in the K and V slots | 11 MLA + MTP |
| `kv_dsa_index` | `RAD_KV_FULL`, u8, block 4 | one page a pooled key: 128 E4M3 codes and their f32 scale (`dsa_pool`'s cache form) | 11 MLA + MTP |
| `kv_index_tail` | `RAD_KV_LINEAR`, bf16, `[4 + spec, 256]` | the key and gate rows not yet pooled; the ring is wider by the draft depth so a rejected draft rolls back | 11 MLA + MTP |
| `kv_kda` | `RAD_KV_LINEAR`, f32, 128 x 128 a head | the recurrent state, 64 / tp heads a rank | 34 KDA |
| `kv_kda_conv` | `RAD_KV_CONV`, bf16, width 4 | the convolution history, 3 x 8192 / tp channels a rank | 34 KDA |

The latent and the index are replicated: every rank keeps the whole latent cache, and the plugin
declares both so (`rad_kv_group_replicated`), which lets the host tier (`--prefix-cache-mib`) hold
each once a slot: 49.59 KiB a 4-token block at any rank count (`kv_latent` 48.00 KiB over its 12
layers, `kv_dsa_index` 1.59 KiB), 12.4 KiB a token, where a copy per rank would make a slot
396.75 KiB at eight. The model is a hybrid, so a prefix-cache hit is usable only from a state
checkpoint at or below it, and `--checkpoint-interval` decides how much of a shared system prompt
is reused.

**Linears.** Each weight is served as the export stores it:

| weights | stored | served |
|---|---|---|
| routed and shared experts, layers 3-44 | MXFP4: E2M1, an E8M0 scale per 32 | `moe_gemm_q` mxfp4a8, an E4M3 activation scaled per (row, 128) |
| MLA q_a, kv_a, q_b, o (trunk and MTP); dense feed-forward, layers 0-2 | E4M3, an f32 scale per 128 x 128 | `gemm_nt_q` fp8a8f32 |
| MTP layer's experts | E4M3, an f32 scale per 128 x 128 | `moe_gemm_q` fp8a8f32 |
| MLA kv_b | quantised | decoded to bf16 by the recipe; `mla_project` absorbs and expands with it |
| KDA's seven projections; indexer wq_b, wk with the pool gate | bf16 | `gemm_nt` |
| router, index head gates, mHC projections | f32 | `gemm_nt_f32`, `mhc_pre` |
| embedding, lm_head, MTP eh_proj | bf16 | `embed_lookup`, `logits_gemm`, `gemm_nt` |

The text container holds 25845 weights, 171.32 GiB, the one with the tower 26143 and 172.37 GiB;
at eight ranks each card holds 22.06 GiB of them, 19.67 GiB the routed experts.

**MTP.** The plugin declares checkpoint layer 45 as a serial drafter (`RAD_DRAFT_SERIAL`), depth
3 unless `--num-speculative-tokens` says otherwise (at most 31). Its history pass writes the draft
layer's latent, index and tail for every row and runs its MoE on the proposal rows only.

**Tensor parallelism.** At 1, 2, 4 or 8 ranks; declare refuses any other. `tests/glm5next_test.cpp`
declares and steps the whole plugin at 1, 4 and 8, against pinned graph and step hashes at 1 and
8, the connection, router and index-gate splits at 1, 2, 4 and 8, the indexer's row split at 2
and 8, the row rule at every world, each issue's operand count against the loaded kernel
libraries' schemas, and every declared weight against the served container's names and encodings.

One row rule (`GlmTp::run_rows`): from a pass of `GlmTp::kSplitRows` = 256 rows, a replicated
piece of work splits by rows over the ranks instead, each rank taking one run of ceil(T / world)
rows rounded up to whole 16-byte words of the row it gathers, because an `all_gather` moves whole
words: multiples of four rows for the indexer's one i32 length a row. The last run may be short;
a rank whose run lies wholly past a small step takes the first run and computes dead rows, so
every plane it gathers is written (`GlmTp::run_of`). The indexer also splits a smaller step, from
16 rows once rows x live blocks reach 2^18 (`GlmTp::kSplitWork`). Below the thresholds, as in
decode, the collective costs more than it saves and every rank runs the replicated form.

- The routed and the shared experts split their 2048 width into equal contiguous slices, 256 a
  rank at eight; each rank computes its slice of every expert, and the block's all-reduce sums
  them. The dense feed-forward splits its 12288 the same way.
- MLA splits by whole heads: the rows of q_b and kv_b (8 heads a rank at eight) and the columns
  of o. q_a, kv_a and the latent are replicated.
- KDA splits its 64 heads, with them the q/k/v rows, the convolution taps and the state; the two
  rank-128 projections into the gates are replicated.
- The indexer's key side is replicated: every rank keeps the whole index cache and its tail. Its
  query side splits by rows from the threshold: each rank scores its run under the row rule,
  R = 4 * ceil(T / (4 * ranks)) query rows (`DsaCache::part_rows`), against the whole cache
  (`dsa_score` with `row_part`, the last run's rows past the step scoring `-inf`) and selects for
  the rows of its run that exist, under the bounds the scorer rebased to the run's first row
  (`cu_part`, docs/OPS.md); two `all_gather`s lay the runs' page tables and lengths end to end,
  513 and 1 i32 a row, 8 MiB a layer at a 4096-row chunk. The run's queries are all a rank needs,
  so the query projection, its quantiser and the head gates run on the run's rows too, and the
  gates need no gather; the bf16 projection stays on the step's rows where a run would be 64 rows
  or fewer, the skinny band whose split count a tuned configuration may set. Each row's selection
  is the same kernels' answer to the same operands, so the joined table is bit-identical to one
  rank's and MLA reads it unchanged. The scoring grows with the context (every row scores every
  complete block of its sequence), so this is where long prompts gain: the score plane shrinks to
  one run (256 MiB to 32 MiB at 64k of context and a 4096-row chunk). A smaller step splits too
  from 16 rows once rows x live blocks reach 2^18 (`GlmTp::kSplitWork`): eight sequences
  verifying five drafts from about 22k of context, where the scoring of 48 rows outweighs the two
  gathers; a single sequence's decode never does, and below both thresholds the indexer runs
  whole.
- The vocabulary is row-sharded, 19360 rows a rank at eight; the embedding's all-reduce is exact.
- The f32 projections (the router's 288 x 4096, the index head gates' 32 x 4096, each
  connection's mHC projection of 24 x 16384) are replicated. From a pass of 256 rows
  (`GlmTp::kSplitRows`; `SplitF32Proj` in `arch/common/rad_glm_tp.h`), each rank computes a disjoint part
  with the replicated kernel and an `all_gather` joins the parts, so every value is
  bit-identical: the router splits its output columns, the mHC projection its 16384-wide input
  into eight 2048-column chunks, whose partials `mhc_pre_finish` completes on every rank, and the
  head gates go with the indexer's rows (above). Below the threshold, as in decode, the
  collective costs more than it saves.

**Vision.** The tower (`arch/common/rad_block_glm_vision.h`, transformers' `Glm5NextVisionModel`)
is declared last, so an encoder pass shares the arena with the step's buffers. Every rank runs it
whole: its weights are replicated, `RAD_ACCESS_PER_REQUEST`, a group a block from 46, past the MTP
layer's. Its rows replace the `<|image|>` placeholders' embeddings unscaled, before the embedding's
reduction (`MediaRows`), in the trunk and in the draft head's lookup, as SGLang feeds its draft
model the target's image rows.

```
24 blocks, width 1024, 16 heads of 64, SwiGLU 4096; patch 14 x 14, two frames: 1176-wide rows
patch embedding: the Conv3d as a linear with bias; no position table
block: RMSNorm; qkv with bias; q and k RMS-normed per head with a gain; a 2-D rotary over
  (row, column), 16 frequencies an axis at theta 10000; attention within a picture at 1/8; proj
  with bias; RMSNorm; gate|up with bias, silu(min(g, 10)) * clamp(u, -10, 10), down with bias
post_layernorm (RMSNorm), then a 2 x 2 stride-2 Conv2d 1024 -> 4096 over each merged block
merger: proj 4096 -> 4096, LayerNorm (affine, eps 1e-5), exact GELU, the clipped SwiGLU through
  10240; no biases but the norm's. One row a merged block: an image of P patches is P / 4 tokens
```

The processor is `core/mm`'s `Glm5Next` family (`core/mm/processor.h`). A picture's canvas has sides
in multiples of 28 and 16 to `max_image_tokens` tokens; the picture keeps its aspect ratio, is
resized only when it is under the floor or over the budget (torchvision's bicubic), and the rest
of the canvas, right and bottom, is black. It is spelled `<|begin_of_image|>`, one `<|image|>` a
token, `<|end_of_image|>`. The checkpoint allows 8000 tokens (32000 patches); one encoder pass
bounds it, so `--mm-max-patches auto` (16384) serves at most 4096, and `--mm-max-patches 32000`
the checkpoint's budget. `--image-tokens` lowers it. Video is refused.

Traps:

- **The downsample reads the processor's merge-block order as stored.** The reference views the
  post-norm rows as `[blocks, 2, 2, 1024]`, which in merge-block order is four consecutive rows;
  `conv2d_unfold` (width 2, two input rows a block) writes each block as one row in the weight's
  own column order `(channel, kernel row, kernel column)`, so the checkpoint's `[4096, 1024, 2, 2]`
  is read as a `[4096, 4096]` linear unchanged. The 4096-wide view of the four rows without the
  unfold is `(kernel row, kernel column, channel)`, and multiplied by the stored weight it is another
  model.
- **The rotary pairs across the whole head.** NeoX pairs `(i, i + 32)`, the row on frequencies
  0-15 and the column on 16-31, each section on its own ladder `theta^(-2j / 32)`: `qk_norm_rope`'s
  `axial`, not the `blocks` mode of Gemma 4's tower, which rotates each half on its own.
- **No rotary in the language model.** An image's tokens take their token positions
  (`Item::rope_1d`), and the indexer pools and selects them as text.
- **Without the image rows in the draft head's lookup** the drafts read the placeholder's
  embedding: the text stays the trunk's, only acceptance falls on a prompt with a picture.
- **The canvas is padded, not stretched**, and an image under the budget is not resampled at all:
  its patches are the reference processor's bit for bit (`tests/mm_test.cpp`, every case of
  `tests/glm5_golden.h`, the resampled ones included).

An existing container gains the tower with `rad-convert --reuse <container>` (298 weights,
1.05 GiB, all bf16 as the checkpoint holds them); `--in-place` extends the container itself.

**Measured** against transformers' tower on the CPU (`tests/glm5_vision_reference.py`), the tower
alone, on libref, libavx and one R9700 (libr4d). Three pictures in one pass (1024, 4144 and 792
patches, the last padded): the staged patches are the reference processor's bit for bit; the
merger's rows reach a cosine of 0.9955 to 0.9970 against the f32 reference, a median per-row
relative error of 0.029 to 0.044, where transformers in bf16 reaches 0.9967 to 0.9984 and 0.029 to
0.044; against the bf16 reference, 0.9959 to 0.9983. Through block 11 the engine is nearer f32 than
the bf16 reference (block 0 3.6e-3 against 3.8e-3 relative, q and k after the rotary 3.1e-3
against 3.8e-3); the residual stream then carries a few rows of very large magnitude, and in both a
handful of output rows with a small norm differ by 0.3 to 1.2 relative. A picture's rows are the
same alone and beside two others in its pass. At 16384 patches (a 2048 x 2048 picture at 4096
tokens) the engine reaches 0.9940 to 0.9947 against f32, transformers in bf16 0.9928; at 32000
(8000 tokens), 0.994 against the bf16 reference. On one R9700 a pass of 5960 patches takes 92 ms,
16384 patches 0.49 s and 32000 1.5 s, with an activation arena of 0.56 and 1.10 GiB, beside the
tower's 1.05 GiB of weights on every card.

**MoE decode fusions.** Three, each byte-identical to the launches it replaces, declared
wherever a loaded library serves the fused op (the separate launches otherwise):

| fusion | one launch for | where it applies |
|---|---|---|
| router | `router_sigmoid_topk`, `moe_scatter` and the shared gate's `fill`: `router_sigmoid_topk_scatter` | passes of up to 128 rows (`M * top_k` <= 1024, one workgroup's sort); 84 launches fewer a trunk decode step, 2 a draft pass |
| SwiGLU | `swiglu_clamped` and `quant_act_fp8`: `swiglu_clamped_quant_fp8` | the routed and shared arms and the three dense layers; 87 fewer a trunk step, 2 a draft pass |
| gather inverse | none: the sort writes each slot's row (`moe_scatter_inv`) and the gather reads it (`moe_gather_inv`) instead of searching `sorted` | every MoE pass |

`glm53_moe_decode_fusions_replace_their_launches_where_they_fit` counts the launches. Paired on
the eight R9700s below, greedy text at a fixed draft depth of 3 is identical with and without
them, and eight agent streams decode 456 tok/s against 452, one 144 against 140.

**Decode at a few rows.** A verify pass of one stream carries two to six rows (the draft window
and one), a draft pass one; there a step is a chain of short launches, each bound by its own
latency rather than by bandwidth. Five changes serve those rows, each byte-identical to what it
replaces. Kernel times on one R9700 at the TP8 slices (r4d_selftest's perf modes, `--cold 256`):

| change | where it applies | before / after, us |
|---|---|---|
| `moe_gemm_q` mxfp4a8 takes the small-M form (`moe_mxfp4_small_kernel`): blocks walk the touched experts only, weight codes stay packed in LDS, gate_up stages 512 of K, down reads its rows whole | gate_up to 384 routed rows, down to 32 | gate_up at 1 / 4 / 6 / 16 / 48 tokens: 36.8 / 31.4, 90.0 / 79.7, 120.5 / 101.4, 230.3 / 206.8, 416.7 / 390.0; down at 1 / 2 / 4: 21.2 / 16.4, 26.0 / 22.5, 39.5 / 37.9 |
| `moe_gemm_q` fp8a8f32 takes the same form (`fp8_f32scale_small`) for the MTP layer's gate_up | to 384 routed rows | at 1 / 2 / 8 tokens: 73.6 / 36.7, 83.6 / 65.8, 214.0 / 203.9 |
| `gemm_nt_f32` keeps 16 steps of k in flight a lane (8 from 17 rows at the router's width) | below 64 rows | router (N 288) at 1 / 4 / 8 / 24 rows: 28.1 / 14.4, 36.0 / 14.5, 36.3 / 15.0, 45.4 / 21.6; index head gates (N 32) at 1 / 24: 27.3 / 9.8, 36.2 / 10.0 |
| `logits_gemm` takes `logits_gemm_bf16_cols`, a wave per four vocabulary columns | n_embd 4096 | at 1 / 6 / 8 / 24 / 48 rows: 239.1 / 235.5, 303.2 / 247.5, 378.2 / 250.4, 1133.2 / 635.6, 2354.5 / 1220.1 |
| KDA issues both low-rank down-projections before either up-projection, so a recorded pass orders the four readers of the layer input once and the two up-projections once | every KDA layer | one ordered point fewer a layer |

`r4d_emu` holds both expert forms and the f32 tiles to the host restatement of their per-element
sequence, and `r4d_selftest` compares the head's column form with the block row byte for byte.

**Measured** on eight R9700s (gfx1201, 32 GB; four on PCIe Gen5 x16, four on Gen5 x8) at TP8 with
the flags of `deploy/compose/glm53-flash.yaml` (32768 tokens of context, eight sequences, MTP with
an adaptive window up to 5, reasoning effort low). BetterBench quick, temperature 0.7, cold
prefix cache, and a load generator of agent turns (3.3k-token prompts, 400 tokens out):

| measurement | result |
|---|---|
| prefill, 1511 / 5936 / 11829 / 22138 prompt tokens | 3474 / 3958 / 3951 / 3817 tok/s |
| prefill, one 19861-token prompt | 5.0 s to the first token, 3982 tok/s |
| decode, one stream, weighted over the categories | 166 tok/s, an update every 22.7 ms at p99 |
| decode, 1 / 2 / 4 / 8 streams, aggregate | 145 / 221 / 316 / 426 tok/s |
| agent load, 1 / 4 / 8 streams, aggregate | 146 / 347 / 506 tok/s, MTP acceptance 0.60 / 0.59 / 0.65 |

A second agent run at eight streams after the same restart gave 482. The links count: with three
of the x8 links fallen back to Gen4, the same build prefilled 3368 / 3664 / 3610 / 3512 tok/s at
the four depths above.

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
`arch/gemma4_bf16`'s, and so is the 31B quantised from bf16 at w4a8 or w8a8, which takes the same
DFlash drafter (see "The 31B" below). A z-lab DFlash drafter is merged at convert time
(`--draft-model`); the recipes are `data/recipes/gemma4-26b-a4b-df*.recipe` and
`data/recipes/gemma4-31b-dense-bf16-*-df.recipe`. The reference is transformers'
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

**The draft head in 8 bits.** The drafter's head is the widest read on the draft path: the target's
bf16 table, 352 MiB per rank at tp 4, every draft pass. A container converted with the line
`dflash.draft_head.weight rtn codes=fp8_e4m3 block=128x128 scale=bf16` (the 26B `-df` recipes and
the 31B w4a8 one carry it) stores a block-fp8 copy under that name, and the head reads it at fp8a8
instead: half the bytes, for 176 MiB per rank more. With `codes=i8 group=128 scale=f16 clamp=sym`
(the 31B Q8 recipe) the copy is int8, read at w8a8. Drafts are verified, so the coarser logits can
change what is proposed, never what is accepted; `draft_n_accepted` shows whether they propose
worse. Without the copy the head reads the bf16 table, and the startup note names which head is in
use.

**The drafter's linears on either 8-bit grid.** The seven linears (fc and six per layer) are on the
E4M3 grid (block fp8 or MXFP4, at fp8a8 / mxfp4a8 over an E4M3 activation with a scale per (row,
128)) or on the int8 one (`codes=i8 group=128 scale=f16`, at w8a8 over `quant_act_i8`'s activation
with a scale per row, and `rmsnorm_quant_i8` folding each residual add, norm and quantiser as
`rmsnorm_quant_fp8` does on the other). `fc`'s encoding picks the grid. One quantised activation
feeds several linears, so a linear on the other grid is refused by name. The gate_up|silu|quant
fold (`gemm_nt_q_gated`) is fp8a8 only, so the int8 grid keeps `silu_mul` and the quantiser, one
launch per layer more. Above 64 rows the w8a8 GEMM is the tiled one, which takes N in 256s; the
31B drafter's k and v are 256 rows per rank at tp 4, the narrowest that still fits.

**The 31B.** z-lab's `gemma-4-31B-it-DFlash` is the same block at 5376 (64 heads / 8 kv x 128,
ffn 10752, five layers, taps 1 12 23 35 46 57 of 60, block 16), and `arch/gemma4_bf16` serves it
beside its w4a8 or w8a8 trunk. The taps are copies of the stream each tapped layer's junction
leaves (`rmsnorm_quant_i8` adds the feed-forward and applies the layer scalar, so the copy is
`hidden_states[l + 1]`), declared in the layer they copy for the liveness reason below. The drafter
is declared after the trunk, its layers at 60..64 on a KV group of its own. Its 64 heads over 8 kv
heads are gqa 8 at every rank count, and libr4d's head-128 attention (the one the 26B's gqa-4
drafter runs on, causal over a window or not) takes 4 queries per kv head at a block of 16. So the
cache holds each kv head twice, written by a head-wide copy of k and v after the projection, and
the attention runs at gqa 4. It is exact: query head q reads a copy of kv head q / 8. A container carries the
assistant head or the drafter under `draft.`, not both. The recipes are
`gemma4-31b-dense-bf16-w4a8-df.recipe` (int4 trunk, int8 lm_head, block-fp8 drafter, ~20 GiB)
and `gemma4-31b-dense-bf16-q8-df.recipe` (Q8_0's grid at libr4d's group of 128 for the trunk, its
lm_head and the drafter, ~33 GiB). Both start from bf16, because the drafter reads the trunk's
hidden states and was trained beside the bf16 trunk. Both plugins scale the embedding by
sqrt(hidden) rounded to bf16, as the reference's embedding module does (73.5 at 5376, not 73.32),
and the drafter reads it through the same scale.

**The 31B's prefill overlap.** `arch/gemma4_bf16` cuts a prefill step of 1024 rows or more in two
(slice A whole 128-row tiles, both at least 512) and runs each slice's two all-reduces per layer on
the second lane beside the other slice's compute, as the prefill overlap above does, with one
difference: o_proj and down_proj both write `h`, so the issue order never lets lane 0 write rows of
`h` that lane 1 is reducing, and `h` is the one buffer taken out of the arena's sharing. Attention
runs whole. Both slices are above every row band the sliced kernels have (the int8 GEMMs' skinny
band at 64, since the prefill kernel takes each row's K in order whatever M is, and
r4d_act_threads' last width change at 120 rows for these widths), so on the exact wire the output
equals the whole step's bit for bit. Every junction must be fused, as the w4a8 and w8a8
containers' are; a bf16 trunk runs whole, and so does every step under `RADIANCE_GEMMA4_OVERLAP=0`.

**The 31B's q, k and v as one linear.** Converted with a recipe that names `blk.*.attn_qkv.weight`
(both -df recipes do), each layer's q, k and v (q and k on a global layer, where v is k) are one
weight, each rank holding its slice of every part, and one `gemm_nt_q` reads them where three did:
at decode on the R9700 q, k and v take 17 + 13.5 + 13.5 us and a 4096-row GEMM of the same K 25.
The grid's groups never cross a row, so the codes are those of three separate rules, and on libref
the replies are too. On libr4d the skinny GEMM picks its in-block K split from N (k and v alone
split K in two, the fused GEMM does not), so k and v can round differently in the last bit and a
long greedy reply can part from the old container's late. `qkv_norm_rope` and `kv_store` work on
the row's column slices in place. Attention reads q in the row through `attn_paged_qrow`: a global
layer at every height (the windowed kernel, which serves head_dim 512 under `attn_paged` too), a
sliding layer up to 32 query rows a sequence (libr4d's h256/gqa2 split-KV decode with q at a pitch,
so a verify step included); a taller sliding step (a prefill) copies q out with a `cast` for the
h256/gqa2 prefill kernel, which takes 0.43-0.67 of the windowed kernel's time. Without
`qkv_norm_rope` (a CPU-only engine) q, k and v are copied out for the unfused prologue. A container
that holds the three apart serves as before: the plugin asks the weight's encoding, which
rad-convert answers from the recipe and a container from its entries.

**The 31B's drafter at gqa 8.** z-lab's 31B drafter is 64 heads over 8 kv heads. Where a card
serves `attn_paged_qrow` at head_dim 128, gqa 8 and q_len 16 (libr4d's gqa-4 split-KV decode with
each kv head read by two query groups), the drafter's cache holds each kv head once and attention
runs at gqa 8. The op has no host rows, so the plugin asks it before the drafter's KV group, whose
head count follows the answer. Elsewhere the cache holds each kv head twice and attention runs at
gqa 4, the kv heads copied into place with a `cast` a head and copy (80 launches a draft step at
tp 4).

**The 31B with a DFlash2 drafter.** RedHatAI's `gemma-4-31B-it-speculator.dflash2` is z-lab's
DFlash2 architecture trained with vLLM's speculators library: five Qwen3-shaped layers at 5376 (32
heads / 16 kv x 256, ffn 21504), every one on a 2048-token window and non-causal, the grouped
dynamic convolution and the candidate selector (rank 256, top-k 16), block 8 (seven drafts). The
31B serves it with the block qwen35_fp8 uses for Qwen3.8's (`arch/common/rad_block_dflash2.h`) in
place of z-lab's DFlash, when the container's drafter is a `DFlash2DraftModel`. Its config is in
the speculators layout: the speculator's keys at the top of `draft.` and the layers' under
`draft.transformer_layer_config.` (`drafter_layer_prefix`). Its `aux_hidden_state_layer_ids`
count as vLLM captures them, the residual entering a layer, so 1 17 29 47 59 tap the outputs of
layers 0 16 28 46 58 (`gemma4::dflash2_taps`). It was trained against the target's table as a
plain embedding and the target's head as a plain linear, so its lookup takes no Gemma scale and its
head no soft-cap. The -df2 recipe stores its head copy int8 (`head_i8`, read at w8a8) rather than
the 2-bit grid Qwen3.8's recipes use, so the selector ranks what an 8-bit head proposes. Its
attention is the sliding layers' h256 gqa-2 split-KV decode, non-causal. `run-gemma4-31b.sh`
takes it with `DRAFTER=df2`.

**The 31B's junctions at four ranks.** At two ranks every junction carries the all-reduce in front
of it (`ar_rmsnorm_quant_i8`). At four it does too, at decode heights. libr4d's
`ar_rmsnorm_quant_i8_4rank` is the exact four-rank one-shot with the junction as its epilogue,
byte-identical to `all_reduce` + `rmsnorm_quant_i8`. A one-shot sends three times the message per
rank, so the plugin declares it only up to 4 rows and issues the pair in a taller step. On 4x
R9700 the fold made plain decode 5.6% faster (94 against 89 tok/s) and a DFlash verify (16 rows)
9% slower (276 against 300), where the standalone all-reduce is the exact two-shot. A decode step
at tp 4 then has two standalone reductions (the embedding's and the last feed-forward's) where it
had 121. It gives the pair's greedy text byte for byte. At sixteen rows of 5376 the junction is
55 us against 22 at one row (--profile-ops): its handshakes, not its bytes, set it.
`r4d_selftest --case ar_rmsnorm_quant_i8_4rank` checks it on four cards, and the startup note says
which form a deployment takes.

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

# `gemma3`: Gemma 3 text (TranslateGemma)

`arch/gemma3` serves the language model of a `Gemma3ForConditionalGeneration` checkpoint (the
vision tower and projector are not mapped, so never converted). Written and measured for
TranslateGemma 27B; another Gemma 3 size with the same config keys should declare (4B and 12B have
a 256 head, served by the h256 rows) but is untested, and the 1B, a `Gemma3ForCausalLM`, keeps its
tensors under names this plugin does not map. It is `gemma4_bf16`'s block structure with Gemma 3's differences;
the reference is transformers' `modeling_gemma3` (4.57).

```
TranslateGemma 27B (google/translategemma-27b-it)
  62 layers, hidden 5376, vocab 262208 tied to the embedding, rms_norm_eps 1e-6
  layer_types: five sliding_attention then one full_attention (without it, every
    _sliding_window_pattern-th layer is global: (l + 1) % 6 == 0)
    sliding  32 heads / 16 kv x 128, rope theta 1e4 (rope_local_base_freq), unscaled, window 1024
    global   32 heads / 16 kv x 128, rope theta 1e6, linear scaling factor 8
  scores scaled by query_pre_attn_scalar^-0.5 = 1/sqrt(168); q and k RMS-normed per head, v not
  every norm GemmaRMSNorm, x_hat * (1 + w); sandwich norms around attention and feed-forward
  feed-forward gate|up 2 x 21504 (gelu-tanh) -> down
  embedding * sqrt(5376) held in bf16 (73.5); final norm; tied head; no soft-capping
```

Traps:

- **The rotary scaling is the global class's only.** HF applies `rope_scaling` (linear, 8) to the
  full-attention layers; the sliding ones rotate at `rope_local_base_freq` unscaled. transformers
  5.x reads TranslateGemma's flat `rope_parameters` as unscaled global layers, and llama.cpp's
  current converter misses the factor too; a reference made either way is wrong by KL 0.024.
- **The scale is query_pre_attn_scalar's**, not 1/sqrt(head_dim): 1/sqrt(128) costs KL 0.013.
- **The embedding scale is sqrt(hidden) held in bf16**: 73.5, not 73.32, as HF's buffer is.
- **One int8 scale a row does not fit Gemma 3's activations**: a few channels run hundreds of times
  the rest. `gemma4_bf16`'s w8a8 linears cost KL 0.097 against bf16 on TranslateGemma, w4a8 0.225,
  where int8 weights alone cost 0.003. The quantised linears here scale the activation per (row,
  128).
- **Soft-capping and YaRN are refused**, not ignored: Gemma 2's caps are absent from Gemma 3, and
  these kernels cannot apply them.

Checked against transformers' float32 forward on a tiny random Gemma 3 with TranslateGemma's head
geometry, a 16-token window and the real tokenizer: KL 0.00016 on the host backend and on the cards,
at prefill and at 16-token chunks. The wrong attention scale, the missing linear scaling, a window of
15 or 17 and no window score 80 to 1600 times worse.

**KV groups.** `kv_sliding` (`RAD_KV_WINDOW`, the 1024-token window) and `kv_global`
(`RAD_KV_FULL`), 16 heads of 128 each, 8 a rank at two. The sliding layers' group holds the window,
not the context: at a 16k context a sequence's sliding layers keep a sixteenth of what a full group
would. The cost is the prefix cache: a window group is not reused across requests (see `gemma4`
above), which matters little for translation prompts, whose shared prefix is a line.

**Linears.** Each weight is served as its encoding stores it:

| recipe rule | served | activation |
|---|---|---|
| none (bf16) | `gemm_nt` | bf16 |
| `codes=i4\|i8 group=128 scale=f16` | `gemm_nt_q` w4a8 / w8a8, as `gemma4_bf16` | int8, one scale a row |
| `codes=i8 group=128 scale=bf16` | `gemm_nt_q` i8a8 | int8, an f32 scale per (row, 128) (`quant_act_i8g`) |
| `codes=fp4_e2m1 group=32 scale=e8m0` (MXFP4) | `gemm_nt_q` mxfp4a8 | E4M3, a scale per (row, 128) (`quant_act_fp8`) |
| `codes=u4 zero=u8 group=128 scale=f16` | `gemm_nt_q` w4a8i8_asym | int8, a scale per (row, 128) |

A linear reading the same activation as the one before it (k and v after q, up after gate) reuses
its codes. On every trunk the q and k norms and both rotaries are one `qkv_norm_rope` at `v_norm` 0.
On an MXFP4 trunk each junction (the branch's post-norm, the residual add, the next norm and the
next linear's E4M3 quantiser) is one `ar_rmsnorm_post_quant_fp8` at two ranks on the exact wire,
the branch's all-reduce in the same launch, or a post-norm and `rmsnorm_quant_fp8` otherwise; a
w4a8 / w8a8 trunk takes `gemma4_bf16`'s fused junctions; an i8a8 or asymmetric-int4 trunk runs the
chain. Where a row is missing the plugin declares the chain, and the startup notes say which form
each took. Under calibration no junction is fused (below).

**Calibration.** Serve a calibration container (int8, `translategemma-27b-i8a8.recipe`: KL 0.003
against bf16, and it fits two 32 GB cards) with `RADIANCE_CALIB_DIR` set, push a corpus of the
prompts you will serve through prefill, and each linear's input Gram is written once
`RADIANCE_CALIB_TOKENS` have passed; the junctions run unfused meanwhile, so the tap sees every
linear's bf16 input from any container. `RADIANCE_G3_CALIB_LAYERS=lo:hi` limits the tap to a layer
slice, so the 21504-wide down projections of 62 layers can be taken in passes. At two ranks the
column-split linears (o, down) see their rank's columns only and write `<stem>.part`; libquant
assembles the parts block diagonally (docs/TOOLS.md). Give every prompt a distinct first line, or
prefix-cache hits keep tokens from the tap.

**Measured** on two R9700s, against the bf16 model on 32 held-out translation texts (chunked at
2048 tokens), decode on an 800-token translation:

| container | mean KL | top-1 | decode, 1 / 2 / 4 streams |
|---|---|---|---|
| int8 (calibration) | 0.003 | 98.0% | |
| **MXFP4 + GPTQ** | **0.042** | **93.2%** | **56 / 108 / 205 tok/s** |
| asymmetric int4 + GPTQ | 0.043 | 93.4% | 22 tok/s (the tiled kernel) |
| MXFP4, round to nearest | 0.142 | 87.5% | |
| w4a8, round to nearest | 0.225 | 84.1% | |

**DFlash2.** No public drafter exists for Gemma 3. One trained with NeMo AutoModel's
`TrainDFlash2Recipe` on the target's own translations (a Qwen3-shaped draft config: plain RoPE 1e6
and SiLU; target layers 1, 16, 30, 44, 59; mask token `<unused0>`, 6; block 8; window 2048;
`dflash_config.input_embedding_scale` 73.5) is merged with `--draft-model` and served as
`qwen35_fp8` serves Qwen's: the trunk copies the residual stream after each tapped layer's junction
into `dflash_src`, the probe fixes the depth at block - 1, and `rad_block_dflash2.h` applies the
embedding scale through `embed_lookup`'s `mult` (only when it is not 1, so other drafters' graphs are
unchanged). On 64 held-out translation pieces it accepts 3.3 tokens a verify step: 152 tok/s a
stream and 364 at four, against 56 and 175 without it on the same pieces (the table's 205 at four
is one 800-token translation run four times).

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

# `krea2`: Krea 2 Raw and Turbo, an image generator

`arch/krea2` declares a generator in the core's JOINT_STEP layout: the conditioning pass leaves
each lane's text rows (after the text fusion and `txt_in`) in its store, and every denoising step
runs them with the image, since every row is modulated by the step's timestep and nothing can be
cached as keys. Served on `/v1/images/generations` ([GUIDE §5.6](GUIDE.md)); it takes no reference
pictures, so `/v1/images/edits` answers 501.

```
text      Qwen3-VL-4B's language model over the prompt in Qwen-Image's template, the template's
          system turn dropped; TWELVE taps, hidden_states[2, 5, .., 35] (the outputs of layers 1,
          4, .., 34) before any final norm; the layers past the last tap are not converted
fusion    two blocks attending across a token's twelve taps (twelve consecutive rows), a learned
          Linear(12, 1) over them (mix_rows), two blocks attending across the tokens, txt_in
DiT       28 single-stream blocks over [text | image]; gqa 48 query heads to 12, a sigmoid output
          gate, q and k RMS-normed, three-axis rotary (text at the origin, an image token at
          (0, row, column)); six modulation rows a block = ONE timestep projection shared by all
          blocks + each block's learned table (table_add), gates without tanh
norms     every RMS norm zero-centred: the gain is 1 + w, in f32
VAE       Qwen-Image's (Wan 2.1's causal 3-D convolutions on one frame), 16 channels at 8x; a token
          is 2 x 2 latents (64 values), 16 pixels a side
schedule  Raw: 28 steps, the resolution-shifted schedule of its scheduler config, guidance 4.5 in
          the card's terms against an empty negative prompt; Turbo: 8 steps at a fixed shift
          (mu 1.15), no guidance. Raw or Turbo is read from the pipeline and scheduler config
```

Traps:

- **Guidance is in the card's terms**: `cond + s (cond - uncond)`, which the core runs as
  `uncond + (s + 1)(cond - uncond)` (RadGenDecl::cfg_original). Read as diffusers' usual
  `uncond + s (...)` the picture is under-guided by one.
- **The taps are layer outputs, not inputs.** `hidden_states[i]` is the output of layer i - 1;
  an off-by-one shifts all twelve and gives a plausible, wrong picture.
- **The text rows are a token's twelve taps, then the tokens.** The first two fusion blocks attend
  within a token (twelve rows), the last two across tokens; swapping the order of the rows swaps
  which attention sees what.

Against diffusers' pipeline in bf16 on a tiny random-weight pipeline of the same layout (256x256,
from the same latents, mean absolute difference in 8-bit levels): Turbo 0.6, Raw guided 2.2 in an
all-bf16 container. Block fp8 moves the random model's Raw picture by 8.5 levels, which is the
quantisation's: the container's codes decoded to bf16 on the host give the same picture.

---

# `sdxl`: Stable Diffusion XL and its finetunes, an image generator

`arch/sdxl` serves SDXL 1.0 and every finetune published as one `.safetensors` in Stability's
(sgm) layout: Illustrious, NoobAI, Pony and the civitai checkpoints built on them. rad-convert's
importer (`core/format/checkpoint.cpp`, `open_sgm`) reads the file as it ships; the plugin is a
JOINT_STEP generator whose conditioning rides in each lane's store and is read by the UNet's cross
attention, and whose sampler is the core's discrete-time one (`RadGeneratorDecl::train_steps`).

```
names     the UNet under model.diffusion_model.*, CLIP ViT-L under
          conditioner.embedders.0.transformer.text_model.*, both as the file has them; OpenCLIP
          bigG renamed to transformers' CLIPTextModelWithProjection under
          conditioner.embedders.1.transformer.* (in_proj split into q/k/v_proj, c_fc/c_proj as
          fc1/fc2, ln_* as layer_norm*, text_projection transposed into a linear's weight); the VAE
          decoder in diffusers' names under vae.*. These are the names kohya's lora_unet_*,
          lora_te1_* and lora_te2_* adapters are written against
text      the prompt in CLIP windows: BOS, up to 75 tokens, EOS, padding to 77 (the core cuts them,
          RadGeneratorDecl::chunk_tokens), each window encoded on its own and causally. ViT-L (12
          layers, quick_gelu) and bigG (32 layers, gelu); the context is ViT-L's and bigG's
          PENULTIMATE hidden states side by side (2048 wide, no final norm); the pooled vector is
          bigG's LAST layer through final_layer_norm at the first window's EOS, through
          text_projection
padding   ViT-L's padding is EOS (49407), bigG's is token 0 ("!"), as the two tokenizers pad; the
          padded rows are part of the context the UNet attends to
UNet      320/640/1280 channels at 1, 1/2, 1/4 of the latent; residual blocks (GroupNorm(32) eps
          1e-5, SiLU, 3x3 conv, the time embedding's projection added per channel, a 1x1 skip when
          the width changes); spatial transformers (GroupNorm eps 1e-6, linear proj_in/out, blocks
          of self attention, cross attention over the context and GEGLU, 64-wide heads) 2 deep at
          1/2, 10 deep at 1/4 and in the middle; skip connections concatenated after h
added     the time embedding (sinusoid of the timestep, cos first, 320 wide) plus label_emb of
          [pooled | sinusoids of (height, width), (0, 0), (height, width)] (2816 wide)
VAE       AutoencoderKL, 4 channels at 8x, the latent divided by 0.13025; run bf16
sampler   k-diffusion over 1000 scaled-linear timesteps (betas 0.00085..0.012): f32 latent lanes,
          the UNet's input x / sqrt(sigma^2 + 1), the timestep the nearest trained one (ComfyUI),
          guidance on the denoised estimates; Euler, Euler ancestral, DPM++ 2M on ComfyUI's
          normal, karras, exponential, sgm_uniform and simple schedules
```

Each denoising pass runs the UNet once per entry at that entry's size: a convolution's image size
comes from its operands, so two sizes cannot share a launch. The sampler ops run over every entry.
Every 3x3 convolution is `im2col_nhwc` and a `gemm_nt_bias` over the weight read as
`[cout, cin * 9]` (the checkpoint's order), so it carries a runtime LoRA like a linear; this is 9%
faster than `conv2d_nhwc` over the UNet's shapes at 1024x1024, slower only at 1/1 with 320
channels. conv_in and out.2 (4 channels) are direct convolutions.

Traps:

- **The context is the penultimate layer of both encoders.** hidden_states[-2], before any final
  norm; only the pooled vector goes through bigG's final norm, from its last layer.
- **bigG's padding is not EOS.** Its tokenizer pads with "!"; reading the padding as EOS changes
  the context rows the UNet attends to (it does not change the pooled vector).
- **The size conditioning is (height, width), not (width, height)**, each through its own
  256-wide sinusoid, after the pooled vector.
- **open_clip's text_projection multiplies from the right** (x @ P); a linear's weight is P^T.
- **bf16, not f16.** The release is f16 and the SDXL VAE overflows in f16; bf16 has f32's range.

Against a torch reference -- diffusers' text encoders, UNet and VAE in f32 under the same
k-diffusion loop, schedule and Philox noise -- from the same seeds at 1024x1024 on one R9700 (relative
L2; diffusers' own bf16 run against the same f32 one in brackets): ViT-L's context 0.28% (0.28%),
bigG's 1.8% (2.0%), the pooled vector 0.9% (1.0%), the UNet's first evaluation 0.25% (0.31%); the VAE
on the same latent 45.0 dB. Whole pictures (PSNR against f32; diffusers bf16 in brackets): SDXL 1.0
Euler 31.3 dB (27.4), DPM++ 2M Karras 34.6 dB, Euler ancestral 24.8 dB (18.0); Illustrious-XL v2.0
Euler 30.0 dB, Euler ancestral 22.3 dB (25.7), 832x1216 Euler 33.1 dB (28.3 against a reference
given (width, height) for its size); NoobAI-XL V-Pred 1.0 (v prediction, zero terminal
SNR) Euler 22.4 dB (22.6). A 91-token prompt's two windows against the same windows encoded by
transformers in f32 (Illustrious): ViT-L 0.7% in each, bigG 1.6% and 2.2%, the pooled vector 1.3%.
An image is the same bytes whether its pass carries other requests' images of other sizes or not.

---

# `flux3`: FLUX.3 [klein] and its base, an image generator and editor

`arch/flux3` declares a JOINT_STEP generator over three modes: the text, the reference pictures'
latents and the image. The conditioning pass leaves each lane's text rows (after
`context_embedder`) and reference rows (VAE-encoded by the generator's own encoder and embedded) in
its store; every step runs the three modes' early blocks and then the joint blocks. Served on
`/v1/images/generations` and `/v1/images/edits` with up to ten reference pictures.

```
text      Qwen3-VL-4B-Instruct's language model over the prompt in its chat template, right-padded
          to a multiple of 80 tokens; the padding takes part (each padding token at position 1,
          attending to the prompt alone: attn_cross with valid_cu); EIGHT taps hidden_states[4, 8,
          .., 32] side by side, 20480 values a token, through context_embedder
DiT       parallel blocks, h += gate (attn(n) + mlp(n)), n a LayerNorm without affine times
          (1 + scale) plus shift; 5 early blocks a mode, each mode on its own, then 28 joint blocks
          over [text | pictures | image] an entry; each mode its own modulation (text and pictures
          at t = 0, the image at the step's), separately for the early and the joint blocks
rotary    four axes (t, h, w, l): text along l, the image at (0, h/2, w/2, 0), reference k at
          l = -(1 + k)
flow map  the step-distilled release adds the embedding of the time the step goes TO to every
          mode's timestep embedding (transformer config flow_map)
VAE       FLUX.2's: 32 channels at 8x, GroupNorm(32) and a single-head mid-block attention; a token
          is 2 x 2 latents (128 values), normalised by the VAE's BatchNorm running statistics
          (chan_stats with var 1: sqrt(running_var + eps))
schedule  [klein]: 16 steps, no guidance; base: 50 steps, guidance 3.5 against an empty negative
          prompt; both a constant shift of 6.93
refs      at most ten, each shrunk (never enlarged) to 1024 x 1024 pixels of area
```

Traps:

- **The padding is part of the prompt.** The pipeline pads to a multiple of 80 and the padding
  tokens' states reach the DiT; dropping them, or letting them attend to each other, changes every
  text row the image attends to.
- **GroupNorm over 128 channels in 32 groups** is four channels a group, which group_norm runs at
  4-channel chunks; a kernel that needs 8 refuses the VAE.
- **The prompt upsampler is not run.** diffusers' modular pipeline can rewrite the prompt with a
  separate LLM first; radiance reads the prompt as given, as the pipeline does with that block
  removed.

Against diffusers' modular pipeline in bf16 on a tiny random-weight pipeline of the same layout
(256x256, from the same latents, mean absolute difference in 8-bit levels, GPU): [klein] 1.17, the
base guided 2.30, an edit with one reference picture 1.16.

---

# `anima`: Anima (Base, Aesthetic, Turbo), an image generator

`arch/anima` declares a JOINT_STEP generator, though its DiT is a cross-attention model: the
conditioning pass leaves each lane's 512 context rows in its store, and every denoising step reads
them back (the entry's text mode, `mode_src`) for the cross-attention's keys and values; the text
rows the layout carries are never run through the DiT. Served on `/v1/images/generations`; it
takes no reference pictures.

```
text      Qwen3-0.6B-Base over the prompt as it is (no template); its last hidden state after the
          final norm. An empty prompt is one <|endoftext|> (min_tokens 1), not " "
adapter   the LLM adapter: the prompt's T5 tokens (the container's second vocabulary) as query
          rows, embedded from a 32128-row table; six blocks of RMS-normed self-attention over them,
          cross-attention to the Qwen3 rows and a GELU MLP, 16 heads of 64, q and k RMS-normed and
          rotated (NeoX, theta 1e4) at their own row index on either side; out_proj and an RMS
          norm; zero-padded to 512 rows
DiT       Cosmos-Predict2's MiniTrainDIT at 2B: 28 blocks of 2048 (16 heads of 128); per block a
          self-attention (q/k RMS norms, a NeoX rotary of 22 + 21 + 21 frequency pairs over
          frame, row and column at 10000 x 4^(42/40); the frame is 0), a cross-attention to the
          512 context rows (q/k norms, no rotary) and a GELU MLP, each behind a LayerNorm without
          affine times (1 + scale) plus shift and added back times its gate
modulation each of a block's three: Linear(256, 6144)(Linear(2048, 256)(silu(e))) + the shared
          AdaLN-LoRA term Linear(2048, 6144)(silu(Linear(2048, 2048)(t))), t the sinusoid of the
          step's sigma (a timestep in [0, 1], [cos | sin]) and e its RMS norm
patches   a token is 2 x 2 latents, 64 values channel-major; the patch embedding was trained on 68
          (a padding-mask channel, always zero), read over its first 64 columns (RAD_MAP_COLS);
          the final linear's rows reordered from (row, column, channel) to channel-major
VAE       the Qwen-Image VAE under ComfyUI's Wan 2.1 names, 16 channels at 8x; the latent
          denormalised with Wan 2.1's statistics in f32
schedule  ComfyUI's FLOW sampling (multiplier 1, shift 3) and its `simple` scheduler over 1000
          timesteps (sigma_grid), Euler steps; guidance uncond + s (cond - uncond). Base and
          Aesthetic: 30 steps, s 4, the card's negative prompt; Turbo: 10 steps, unguided. The
          variant is `anima.variant` (rad-convert --set)
```

Traps:

- **T5 tokens, not Qwen3's, are the queries.** The adapter reads the same prompt twice: Qwen3's
  ids for the source, T5's for the queries; T5's normaliser is SentencePiece's precompiled map
  over extended grapheme clusters (core/text/tokenizer_grapheme.h). Reading the queries off Qwen3's
  tokens gives a plausible, wrong picture.
- **The padding rows are attended to.** The context is zero-padded to 512 rows and the DiT's
  cross-attention reads all 512: a zero row is a zero key and value, which still takes its share
  of the softmax.
- **The DiT keys have two spellings.** Base ships `net.blocks.N...`, Aesthetic 1.1 and Turbo 1.1
  `model.diffusion_model.blocks.N...`; the plugin reads the prefix off the checkpoint at convert
  and keeps the file's own names as the name map's sources.
- **The empty negative prompt is not " ".** ComfyUI encodes "" as one padding token for Qwen3 and
  </s> for T5; the core keeps an empty prompt empty for a model with `min_tokens`.

Against ComfyUI in bf16 on an RTX 4090 from the same noise: the DiT's velocity at sigma 1 rel
0.48% (cos 0.99999) from the bf16 container and 1.5% from the block-fp8 one, the adapter's 512 rows
rel 1.9% (Qwen3's output in bf16 against ComfyUI's f32), the VAE on the same latent 46.9 dB; three
1024 x 1024 pictures after 30 guided steps 32.8, 36.0 and 39.0 dB from bf16, 26.7, 26.0 and 28.6 dB
from fp8 (the same picture to the eye).

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

---

# `ltx2`: LTX-2.5, video with sound

`arch/ltx2` declares a video generator (`rad_declare_video_generator`, schedule form 1) on the
passes of `core/gen/video_driver.h`: VCOND once per request (the text encoder over the prompt and
the negative prompt, the connectors into the lane's text rows, the first frame encoded for image to
video), VSTEP once per evaluation, VDECODE and ADECODE once per video, VUPSCALE between the two
stages of the distilled recipe. Served on `/v1/videos` ([GUIDE §5.16](GUIDE.md)), one card, bf16 or
block fp8. The checkpoint is the diffusers `LTX2Pipeline` directory, with `latent_upsampler/`,
`duration_head/` and `diffusion_decoder/` read when they are there.

```
text      Gemma 4's language model over "<bos>{prompt}<eos>" (no template), LEFT-padded to 1024
          rows; every layer's output is kept (the embeddings, each layer's but the last, the final
          norm of the last), each RMS-normed per layer and flattened channel-major per token.
          Gemma 4 scores q.k at scale 1: the q and k norms' gains carry the 1/sqrt(head_dim)
connector per modality a projection of that stack to its width (video 4096, audio 2048), the
          1024 rows past the prompt's tokens filled from 128 learnable registers, eight
          bidirectional blocks with a split rotary over the row index
DiT       48 dual-stream blocks: video 4096 (32 x 128) over 1 x 1 patches of 128 channels (32
          pixels a latent, 8 frames a latent frame + the first), audio 2048 (32 x 64) over 25
          latents a second (8 channels x 16 mel bins); a block's video stream runs self-attention,
          cross-attention to its text, attention to the soundtrack and a feed-forward, the
          soundtrack's the same against the video; every attention gated per head
          (2 sigmoid(logit)), q and k RMS-normed across all heads; per-block tables + the step's
          timestep projection (table_add), the cross-modal ones shift-first
rotary    split over (t, y, x) at the patches' MIDPOINTS for the video, the time midpoint for the
          audio (rope_split, max 20 2048 2048 and 20)
guidance  up to four evaluations a step -- the prompt, the negative prompt, the prompt with block
          28's self-attentions bypassed (STG), the prompt without the cross-modal attentions --
          combined on the clean estimates with a rescale (ltx_guide); base defaults video
          (cfg 3, stg 1, modality 3, rescale 0.7), audio (7, 1, 3, 0.7); distilled: one, unguided
VAE       causal 3-D decoder, four up blocks of residual blocks with depth-to-space and
          PerChannelRMSNorm (pixel_norm div), 4 x 4 pixel patches out; a frame larger than
          960 x 544 in diffusers' tiles (16 latents every 14: tile_grid, tile_crop, a tile store,
          tile_stitch's blends to bytes), a smaller one whole; its encoder takes the first frame
          for image to video
ddec      the diffusion decoder, a request's `decoder` "diffusion": the latent with its last frame
          twice more (NATTEN's border needs them), stages of 4 6 4 2 neighbourhood-attention blocks
          (2048 1024 512 512 wide, windows 3x7x7 3x7x7 3x5x5 3x5x5, pre-norm, a 3-D gptj rotary
          over the voxel coordinates in 8 12 12 pairs, SwiGLU 4x) with pixel-shuffle upsamples
          (1x2x2, 2x1x1, 2x2x2, 2x2x2); the first three over the whole video, then per tile of the
          feature grid (_tile_intervals: 96 cells every 88, 40 every 28) the fourth and one x0 step
          at timestep 1 from the seed's noise (stream 2 + the tile) over 4 x 4 pixel patches: 8
          diffusion blocks of 256 (11x11x11 windows) conditioned on the fourth stage's output, the
          norms modulated by the timestep's shared table plus each block's own, each attending a
          head at a time (one head's q|k|v is a quarter of all four's); tile_stitch blends the
          tiles' overlaps into the frames
audio     a causal 2-D decoder to a stereo log-mel spectrogram at 16 kHz, a BigVGAN vocoder to 16
          kHz, a bandwidth extender (its own vocoder over the waveform's log-mel spectrogram plus a
          sinc-resampled skip, bwe_up) to 48 kHz; the vocoders in f32
length    the duration head (VCOND, when the container has it): each connector's prompt rows
          projected to 256 with a modality embedding, one learned query over all 2048 (four heads
          of 64, no mask), a tanh-GELU MLP to ln(seconds) -- read back by the core, which shapes a
          request that named no length from it (RadVideoGenDecl::duration_out)
schedule  base: 30 steps of FlowMatchEuler, exponential dynamic shift 0.95..2.05 over 1024..4096
          tokens, shift_terminal 0.1; distilled: its 8 trained sigmas (1.0 .. 0.421875), and with
          the upsampler the two-stage recipe -- the 8 at half size, the latent upsampler, both
          latents re-noised to 0.909375 with stream 1 of their seeds (flow_renoise), the stage-2
          sigmas 0.909375, 0.725, 0.421875 at the full size
```

Traps:

- **The noise is radiance's.** The video's latent is stream 0 of the seed in packed (f, y, x)
  order; the soundtrack's the seed with its top bit flipped; the second stage's re-noise is stream
  1 of each. Parity against diffusers means handing it the same tensors: the latents, and for the
  first-frame pipeline and the re-noise its `randn_tensor` patched to return them.
- **A distilled schedule is read as given.** diffusers stretches explicit sigmas to a scheduler
  config's `shift_terminal`; a distilled pipeline whose scheduler config kept the base model's 0.1
  would run different sigmas there, and the plugin runs the trained ones.
- **The vocoder's dilations are nested lists** (`[[1, 3, 5], ...]`); the metadata reader flattens
  them row-major. Dropped, every resnet runs at dilation 1 and the soundtrack is noise.
- **The first frame goes through H.264 at CRF 18 before the encoder**, as the pipeline does for
  LTX-2.5 (33 for the Gemma 3 generations). In the two-stage recipe the first frame of the
  UPSCALED latent is the one kept clean, as diffusers' image pipeline keeps it, not a re-encode
  at the full size.
- **A tile's crop reads the latent after earlier tiles have decoded.** The buffer planner orders
  liveness by declaration, which has one crop before the decoder; the latent is a persistent buffer
  so the decoder's scratch cannot take its memory between tiles. Before that, every tile but the
  first decoded garbage and the first one partly.
- **The MP4 is not a parity instrument.** H.264 at CRF 18 in yuv420p moves a random-weight
  model's frames by about 40 levels on its own; compare the frames before encoding (the op
  oracle's patch_pix8 output).

Against diffusers' pipelines in bf16 on a tiny random-weight pipeline of the same layout (256x256,
17 frames, from the same latents), the first frame's row sums agree to within 0.2% at 2 unguided
and 4 guided steps, for image to video, single-stage distilled and the distilled two-stage recipe
with and without a first frame; diffusers on CUDA and on the CPU differ from each other by up to 0.1%.
On an R9700 (libr4d) the same five renders land within 0.15% of diffusers' CUDA rows, every op
agreeing with its host reference under the op oracle. Block fp8 moves the random model's rows by
0.2 to 0.4% (the card's fp8a8 GEMMs, the CPU's fp8 codes widened to bf16).
