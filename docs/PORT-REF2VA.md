# PORT-REF2VA — reference-media conditioning for MiniMax-H3 (working brief, community PR)

> Scratch planning doc (untracked). Started 2026-10-18 from a ComfyUI-side analysis session
> (`/home/dustin/AI/ComfyUI`). Owner: community contributor on dual Radeon AI PRO R9700
> (RDNA4, gfx1201, 32 GB each). Per the author's call for community PRs, this is the
> implement/test/PR plan. Nothing here is committed yet.

## 1. Goal

Make Radiance's MiniMax-H3 path serve **Ref2VA**: the full official conditioning envelope —
up to **9 reference images, 3 reference videos (each optionally with its own soundtrack),
3 standalone reference audios** — prompt-addressable as `<Picture i>` / `<Video k>` /
`<Audio j>` — producing video + stereo audio, quality-first at ~25 steps, 12–15 s clips,
on **dual-GPU gfx1201** — PLUS the **continuity path**: conditioning on the last ~3 s of
the previous clip (clean latent tail re-injected every step, timeline-continuous) so clips
chain into longer shots; the user's ComfyUI workflows depend on this. Parity reference
implementation: ComfyUI core (`comfy_extras/nodes_minimax_h3.py` +
`comfy/ldm/minimax/model.py`). Out of scope for v1: RIFE frame interpolation (external),
DaSiWa step-cache, tiled upscale. Motivation: an even TP2-vs-two-cards benchmark is only
worth running once Radiance can do what the ComfyUI workflow does.

## 2. Radiance today: what exists, what blocks

| thing | where | state |
|---|---|---|
| H3 T2VA pipeline (text→video+stereo audio) | `arch/minimaxh3/minimaxh3.cpp`, `core/server/videos.cpp` | works; packed sequence is exactly 3 modalities (text 1 / video 0 / audio 2), `minimaxh3.cpp:17,143,307` |
| `/v1/videos` API | `core/server/videos.cpp:135,155,181,189` | **rejects `input_reference` (err_unsupported) and any non-empty `conditions`**; accepted-field list has no media fields |
| container | `minimax-h3-fp8` recipe | reads base transformer only; `transformer_ref/`, `FL2VA/`, `Ref2VA/` layouts **not read** |
| VSA block-sparse attention | `libr4d/r4d_vsa.hip`, `VIDEO_SPARSE_ATTN_H3`, sparsity 0.8, 64-row tiles, prompt+soundtrack sinks | exists; needs ref-row sinks added |
| TP (`--tp 2`) | weights + VAE tiles sharded across ranks; sequence is **not** sharded (`minimaxh3.cpp:60,307,1341-1367`) | ref tokens just extend the sequence identically on every rank — no new sharding logic |
| LoRA | `core/lora/*`, `libr4d/r4d_lora_bf16.hip` | available if the Dasiwa LoRA is wanted later |
| finetune ingestion | `docs/GUIDE.md:406-409` — `model.safetensors.index.json` assembly: changed-tensor shards + links to base shards | the Dasiwa finetune may not need a full re-convert |
| fp4 codes | `spec.md:552-553`: MXFP4 (`fp4_e2m1*e8m0[1x32]`) and NVFP4 (`fp4_e2m1*fp8_e4m3[1x16]*f32`) native; `libr4d/r4d_gemm_mxfp4a8_nt_m64.hip` | available; **not** recommended for DiT weights (no native fp4 MMA on RDNA4 — fp8/int8 WMMA is the floor; fp4 = storage win). Possible lever for the 32B Qwen3-VL TE (cf. shipped `qwen3.8-27b-mxfp4.rad`, 19 GiB) |

## 3. The spec to port (from ComfyUI core, verified line-by-line)

Node: `MiniMaxH3ReferenceToVideo` — `comfy_extras/nodes_minimax_h3.py:243-364`.

**Inputs/limits** (`:264-284`): `ref_image_1..9` (max 9), `ref_video_1..3` (24 fps, ≥5
frames — the "2–15 s" is only the node tooltip and **DaSiWa-side policy**
(`helper_minimax_h3_director.py:121-152`: per-item 2–15 s, per-type totals ≤ 15 s,
≤12 files; the core node itself only truncates to the generation length), so the real
ceiling is the generation length), `ref_video_audio_1..3` (index-paired:
`ref_video_audio_N` belongs to `ref_video_N`), `ref_audios` 1..3 (standalone).
`ref_image_size` ∈ {`match`, `max`}. **The DaSiWa production "audio must encode
≥2.01 s" floor is NOT in any code** (no check in `_encode_ref_audio` or the audio VAE);
it's an empirical/API-contract observation — if Radiance wants it, implement at API
level as policy.

**Presentation order** (`:244-249`, `:294-358`): images → videos (a video's soundtrack gets
its `<Audio j>` label emitted *right before* its `<Video k>`) → standalone audios. Ordinals
1-based per type. Tokenizer sees a multimodal Qwen3-VL presentation (`:360`,
`clip.tokenize(prompt, minimax_ref_items=...)`):
- images as image tokens (downscale: `match` = aspect-preserving scale down to the
  generation's pixel area; `max` = 2048 px short edge; canvas multiple rounding, `:300-309`)
- videos sampled at **2 fps with timestamps** (`i/2.0` s) fed to Qwen (`:337-341`)
- audios as label-only `<Audio j>` entries (`{"type":"audio"}`)

**DiT payload** (`minimax_ref_blocks`, conditioning-set `:362-363`): per ref, kind ∈
{`image`, `video`, `video_audio`, `audio`}: video-VAE latents with `latent_h/w`
(pixel//16 — patch (1,2,2) × VAE /8) and `latent_t`; audio-VAE latents with `ref_audio_t`.
Frame rules (`:326-333`): truncate to generation length; ≥5 frames; trim to `n % 17 == 5`.

**Packing** (`comfy/ldm/minimax/model.py:351-471`, re-verified): segment order is
`[text | cond | cond_audio | per-ref blocks (request order) | audio | video]` —
keyframe/continuation cond rows pack **right after text, ahead of all ref blocks**;
target streams then start at `cursor = text_len + Σ _ref_t_span` (`:369-371`). A
video ref packs its `ref_audio` rows immediately **before** its `ref_img` rows
(`:421-439`). Audio row count is `rt * 2` — stereo is **channel-major stacked**
`[ch0 t0..T-1, ch1 t0..T-1]` (`pack_audio` `:59-62`), NOT frame-interleaved;
`_audio_grid` pins `w` to the grid extremes per channel, `h`=0 (`:117-123`).
`_ref_t_span` per ref (`:105-114`): image = 1.0, audio = `rt`, video =
`max(rt, Σ 5/3·FRAME_PER_TOKEN[k%5])` with `FRAME_PER_TOKEN=(1,4,4,4,4)` (`:30`).
Segment kinds and modulation tags (`:463,:596-599,:711`):
`seg_tag = {text:1, video:0, audio:2, cond:0, ref_img:0, cond_audio:2, ref_audio:2}` —
refs ride the **video/audio** modulation streams (0/2); per-segment timesteps
(`:676-681`): `seg_t = {text/video: t_v, audio: t_a, cond|ref_img: max(t_v,0.999),
cond_audio|ref_audio: max(t_a,1.0)}` — ref/cond rows are **pinned near-clean all run**,
not "following" the stream schedule. Mod-row index = `t_index·3 + tag` (`:711`).

**Continuation / cond rows — MECHANISM CONFIRMED (Q6, 2026-10-18)**: the DaSiWa tail is
a clean **slice of the previous clip's sampled latents** (no VAE round-trip;
`h3_continuity/core.py` ClipStore persists sampled AV latents), injected as a
`minimax_keyframes` entry at `resolved_frame_index: 0`
(`h3_continuity/vendor/continuation_nodes.py:529-535` — the real mechanism is in the
vendored node, not DirectorGuide itself) and converted to `cond`/`cond_audio` rows +
`PackedLayout` in `comfy/model_base.py:2199-2228` (CONDConstant). These rows live
**outside the sampler state** — sampler `x = [video, audio]` is target-only
(`model.py:604-615,648`) and the net returns velocity for the last two segments only
(`:773-790`); every forward rebuilds cond rows from the stored clean latents into the
non-update rows (`:533-580`) — never denoised. Caveat: video-class cond/ref_img rows
get a **deterministic 0.1% noise augmentation** re-seeded from the run seed every
step → identical each step (`:536-545`); audio cond pinned exactly clean.
The denoised target itself contains a **hidden overlap**: the continuation window is
`overlap + extension` latent frames, unmasked and zero-init — the leading overlap rows
are REGENERATED (not replayed) under the identical-position cond rows, then trimmed
after sampling and concatenated behind the untouched previous latent
(`continuation_nodes.py:196-219,632-655`). Time position: `cond_t = cursor +
FRAME_RESCALE*frame_index` (`model.py:377`) with **`FRAME_RESCALE = 5/3`** — a
pixel-frame→audio-latent-unit (1/40 s) conversion, NOT frames-per-latent-frame — so a
tail at frame index 0 shares the target's timeline origin and **overlaps its head** in
RoPE time; continuity comes from the trim, not placement. Grid: legal overlaps snap to
{5,22,39,56,73} px-frames ↔ {2,7,12,17,22} latent (3.0 s tail = 73 frames = **22
latent** video rows + ~121-122 audio rows at 40 Hz, boundary-diff formula
`continuation_nodes.py:180-189`). The old "87 latent frames / 74-frame target = 13 tail
frames" observation fits no legal grid value (74 and 13 are both illegal) — treat it as
an old-version artifact, do NOT port 13 as a constant; the real quantity is
window = overlap + extension.

**Sparse sink rule** (`comfy_extras/nodes_sparse_attention.py`): ref/cond rows must stay
**dense** (`sink_conditioning=exact_kv_and_rows` keeps ref+audio exact); generated
video/audio go sparse (sol-attn, training-free; the checkpoint has no `to_gate_compress`,
so trained VSA is N/A — Radiance's own VSA coarse gate likewise has no trained weights to
use; a training-free pooled-score top-k selection matches what Radiance already does).
Confirmed against the **official HF index** (Q1): `to_gate_compress` appears nowhere —
only FastH3-style distilled releases ship trained gates. Note the vendored ComfyUI
model code is **fully dense** (`optimized_attention(..., mask=None)`, `model.py:200`);
sol-attn/sparse lives entirely in the external node layer, hooking
`transformer_options["minimax_h3_layout"]` (`:667`) — same split Radiance should keep
(pack emits the segment table; sparsity is an attention-backend concern).

**Sampler parity**: guidance-distilled, CFG 1.0; video sigma-shift drives the schedule,
audio shift derived by the DiT (`MiniMaxH3SigmaShift`, `nodes_minimax_h3.py:367+`); 25 steps
preferred (quality at 12–15 s), not the turbo path.

## 4. Implementation phases

- **P0 — API & uploads**: accept reference media on `/v1/videos` (multipart or file refs,
  follow existing `input_reference`/`conditions` field slots at `videos.cpp:135-147`),
  validate ≤9/≤3/≤3, index-pair video soundtracks, stage to disk. Keep T2VA untouched.
  **Q5 RESOLVED (§5.6): fields on the existing endpoint, new `task` value — matches the
  repo's OpenAI/SGLang-schema precedent and the refusal stubs already in place; post the
  §5.6 shape-check issue first.** `RadVideoGenBatch` fields ⇒ ABI bump (23+ after #118
  resolves); author pre-blesses bumps.
  **P0 PARSED/VALIDATED/QUEUED SLICE IMPLEMENTED 2026-10-18 (branch `ref2va`):** file
  parts `ref_image_N`/`ref_video_N`/`ref_video_audio_N`/`ref_audio_N` (N 1-based,
  contiguous, ≤9/≤3/≤3 API ceilings ∧ the model's declaration); pairing, extension and
  empty-file checks all naming the offending part; `task:"ref2va"` accepted only when
  `VideoGenInfo.max_ref_* > 0` (mirror in `engine.cpp` leaves them 0 until the ABI carries
  them — every live model answers the task cleanly refused); refs ride in-memory on
  `VideoRequest::refs` (no staging dir — MP4 path is in-memory too and the transport caps
  bodies at 512 MiB, so "stage to disk" is moot); the generator orders refs by
  (kind, index); the video object echoes `task`/`refs`; `video_driver.cpp` refuses non-empty
  refs until P1 wires them. Tests: `videos_test` +2 cases (order/pairing/reach-the-generator;
  every refusal names its part). The video object does NOT echo the media bytes or sizes —
  only part names, enough to confirm what was accepted.
- **P1 — TE presentation**: build the Qwen3-VL multimodal presentation (images; 2 fps
  timestamped video frames; `<Picture i>`/`<Video k>`/`<Audio j>` tags in fixed order).
  **Q2 RESOLVED — the vision stack already exists, just not wired to H3** (see §5.2):
  reuse `core/mm/` Qwen-VL processor (smart_resize/patchify, vision-token expansion,
  M-RoPE layout, `<T seconds>` markers, 2 fps video default — golden-tested vs HF
  transformers) + the tower in `arch/common/rad_block_vit.h`, patterned on
  `arch/qwenimage21` + `core/gen/driver.cpp:470-560` (tower inline in a generator's
  COND pass). Work = media fields through `videos.cpp`→`VideoRequest`→
  `RadVideoGenBatch`→`video_driver`, encoder declaration in H3, and the ref2va
  re-convert declaring `model.visual.*`. Est: **4–8 days TE presentation; ~1.5–2 wk
  P0+P1 with plumbing.**
  **P1 IMAGE PRESENTATION LANDED 2026-10-18 (branch `ref2va`):**
  (a) *server* (`videos.cpp`): builds the §3 presentation for ref images — head, then per
  image the label `"<Picture i>: "` BPE'd as its own string (`i` = the part's 1-based index,
  which `validate_refs` guarantees gapless), then the item's `expand` tokens appended to
  `VideoRequest::tokens`, then the prompt LAST; `MediaOpts.max_pixels = W*H`, the DaSiWa
  `match` policy (each photo scaled to the output's pixel area); media is built only when
  `deps_.mm->accepts(MediaKind::Image)`, so harnesses without a multimodal backend are
  untouched. (b) *driver* (`video_driver.cpp` `build_cond`): stages `r->media` exactly on
  the image driver's template (`driver.cpp:501-660`) — rope `[3,T]` component-major from
  `pm.rope.at()` into `R_ROPE`, pixels into `R_ENCPIX`, `Item::coords` planes into
  `R_ENCCRD` (stride = total patches), `patch_at` segments into `R_ENCCU` + max segment,
  `mm_rows` from `rope.runs` (`tok+n` per run), and the per-rank `RadBatch` fields
  `rope_pos/rope_mixed/n_mm_rows/mm_rows/enc_n_patch/…`; `minimaxh3.cpp:2084-2124` already
  consumes them (tower step, mm scatter, rope select, deepstack) — **no arch change was
  needed**. (c) *acceptance is phased at the driver, the authority on consumability*: the
  server passes every ref part through; the driver accepts IMAGE parts once media is
  staged and refuses VIDEO/VIDEO_AUDIO/AUDIO naming the part (they reach the model through
  the VAEs — P2), refuses any refs on a tower-less container ("converted without its
  vision tower"), a placeholder run placed past `T`, and `n_patch > max_enc_patches`
  naming both counts. (d) *container*: the tower re-convert
  `--reuse minimax-h3-ref2va-fp8.rad --changed convert-ref2va/changed-visual.txt` (351
  `model.visual.*` tensors) wrote `minimax-h3-ref2va-fp8-tower.rad` (3028 tensors, 61.63
  GiB, 52 s); it is the first container that serves refs — caps verified live over
  `/server_info`: `video_max_text 16384`, `video_max_ref_{images,videos,audios} 9/3/3`
  (the stock containers answer 0/0/0 and the driver refuses refs on them, per §5.7).
  (e) *tests*: `video_gen_test` +2 cases (conditioned VCOND pass field-by-field + VSTEP
  leak checks; every refusal message) and `videos_test` +1 (exact token presentation with
  out-of-order parts — `ref_image_2` posted before `ref_image_1` still presents as
  Picture 1, Picture 2). **Real image decode is still blocked on libav**: a build without
  `RAD_WITH_FFMPEG` refuses every media part at decode, so end-to-end with true files
  awaits the libav dev packages. Audio ordinal sharing between soundtracks and standalone
  audios (one per-type counter) is deliberately deferred to P2 — recorded in the
  `videos.cpp` comment; only images are presented today so nothing can observe it.
- **P2 — ref VAE encoding**: video-VAE **encode** for images/videos + audio-VAE encode
  for soundtracks. **Q3 RESOLVED (§5.5): this is fully net-new — the plugin declares no
  encoder, the containers carry no encoder weights, and both encoders are architecturally
  different networks from the shipped decoders (3-D causal-conv ResNet / DAC strided
  stack, not ViT/BigVGAN); ~2–3 wk with 3 new/extended libr4d kernels, encoder weights
  declared in the P1/P5 re-convert. Audio-only ≈1 wk.** Port the sizing/frame rules
  from §3 exactly (//16 latents, `n%17==5`, ≥5 frames, match/max scaling,
  CANVAS_MULTIPLE; images = t=1 path, mean-only posterior, tiling per §5.5).
  **LANDED (§5.10): the host half — encode sizing, tile plans, constant tables — as
  `arch/minimaxh3/rad_vae_geom.{h,cpp}`, golden-tested against the reference's own CPU
  encoder runs; `docs/NOTE-P2-ENCODE.md` carries the line-referenced extraction. The
  device half (kernels/graphs/declare/value-parity) is what remains.**
- **P2b — continuation path** (**mechanism confirmed, §3+§5.3**): accept a context clip
  (previous clip's tail) or staged latent; seed as `cond`/`cond_audio` rows at
  `resolved_frame_index=0` — clean-latent slice, **outside sampler state**, rebuilt
  every step, RoPE overlapping the target head; window `overlap+extension` sampled
  unmasked then trimmed post-sample. Prefer accepting **sampled latents directly**
  (ClipStore-style per-clip AV-latent store, `latent.safetensors`+`clip.json` pattern)
  so chained takes skip the VAE encode entirely — NOTE-CONTINUITY §3 calls this the
  biggest portable win. Reuses the API upload + VAE encode from P0/P2 — mostly P3 work.
- **P3 — arch pack**: add `ref_img` / `ref_audio` segment kinds in fixed order; per-segment
  3-axis RoPE placement; modulation via existing per-modality rows (refs → tags 0/2);
  timestep spans per ComfyUI `:111-113,:680-681`. TP: nothing new (sequence unsharded).
  **Pack half LANDED 2026-10-19 — §5.9** (`arch/minimaxh3/rad_ref2va_pack`, golden-vector tested);
  first consumption point LANDED too — `positions()` now builds through the pack (§5.11);
  `seg_cu`, `step_dit` row assembly, and the widened mod table are the remaining wiring.
- **P4 — VSA sinks**: extend the prompt/soundtrack sink rule to keep all ref rows dense.
  Expect sparsity wins to shrink with big sinks (9×2048 px images + 3×15 s videos are tens
  of thousands of tokens riding every step) — tune sparsity/tau; log dense-vs-sparse rows.
  **Rule falls out of the layout** — `vsa_geom` makes every pre-video segment a dense prefix, so
  ref rows are sinks once listed in `seg_cu` (§5.9); dense-vs-sparse logging stays open work.
- **P5 — container/weights for a ref2va checkpoint** (**resolved, §5.1**): convert HF
  `MiniMaxAI/MiniMax-H3` `transformer_ref/` via a synthetic input dir (it becomes
  `transformer/`) + `--reuse minimax-h3-fp8.rad --changed <638 transformer.* names>` —
  **zero name-map changes**, and declare `model.visual.*` in the same pass (§5.2).
  Local `-pruned-`/int8 files are NOT convertible sources (folded AdaLN + refused
  quants). Dasiwa finetune via index-shard assembly (`GUIDE.md:406-409`) from its bf16
  release if wanted; LoRA as a MERGED container via `rad-convert --lora` (§5.8 — the
  runtime `core/lora` path is text-generation-only by design; the workflow's ref2va LoRA
  @1.0 is active and ships as `...-tower-lora.rad`).
  **P5 CONVERTED 2026-10-18: `/home/dustin/AI/models/MiniMaxAI/minimax-h3-ref2va-fp8.rad`**
  (60.53 GiB, 2677 tensors — verified a name- and dtype/shape-identical superset of
  `minimax-h3-fp8.rad` except dit.* VALUES now come from `transformer_ref` (zero drift
  elsewhere; the finetune is key-identical as §5.1 predicted)). Gotcha found: the author's
  base was `--in-place`-extended, so its name table sits at EOF — a head scan of it finds
  no names at all; and it carries **322 `ups.*` latent-upscaler tensors** (LBH-123-AI/
  Minimax_h3_latent_Upscaler, Apache-2.0, bf16 691 MB → downloaded to
  `MiniMax-H3/latent_upscaler/` as `diffusion_pytorch_model.safetensors`+`config.json`).
  A container WITHOUT `latent_upscaler/` converts and declares fine (plugin guards
  `if (!w||!bias) return 0` on the upscaler subgraph, `minimaxh3.cpp:1616-1650`) but
  silently loses upscaling — keep the folder in every re-convert. Serve it exactly as the
  base: `radiance --model ... --tp 2 --vision-tower-placement ram` (GUIDE §5.16; without
  the ram placement the towers eat every GiB and the budget resolver stops startup).
  `visual.*` still NOT in this container (plugin declares none yet — lands with P1;
  the P1 re-convert `minimax-h3-ref2va-fp8-tower.rad` carries them — see the P1 bullet
  and §5.7; the uncensored twin's tower re-convert is the same one command:
  `--reuse minimax-h3-ref2va-uncensored-fp8.rad --changed convert-ref2va/changed-visual.txt`).
  Two artifacts written 2026-10-18: `docs/REF2VA-CLIENT.md` (director-agent API brief:
  T2VA today, the ref2va multipart contract, error taxonomy, continuity NOT-yet warning)
  and `convert-ref2va/README.md` (how Deadcode regenerates the container from public HF
  sources, incl. every gotcha above). User's fork remote = `fork`
  (https://codeberg.org/gearwave00001/radiance.git); NOTHING pushes without explicit ask.
  **Push gate (user, 2026-10-18): nothing pushes until full workflow parity** — the whole
  ≤9img/≤3vid/≤3audio ref2va envelope AND the full resolution/duration range generating
  end-to-end (per-setting parity NOT required) — i.e. P1–P4 landed + verified.
  **LORA PARITY LANDED 2026-10-19 (§5.8):** the workflow's active adapter serves as a
  MERGED container — `rad-convert --lora <adapter> --lora-map convert-ref2va/lora-map-h3.json`
  (new `--lora`/`--lora-strength`/`--lora-map` flags; runtime adapter slots stay
  text-generation-only by design, `lora/declare.cpp`). Merge + `--reuse` over the tower
  container yields `minimax-h3-ref2va-fp8-tower-lora.rad`.
  **Uncensored-TE variant BUILT 2026-10-18 (user's own, NOT PR-bound):**
  `minimax-h3-ref2va-uncensored-fp8.rad` (60.53 GiB, 2677 tensors, `--changed
  convert-ref2va/changed-heretic.txt` + `--reuse minimax-h3-ref2va-fp8.rad`, synthetic dir
  `MiniMax-H3-heretic-src`, staged TE `MiniMax-H3/text_encoder_uncensored` via
  `convert-ref2va/tmp-saved/mk-uncensored-te.py`, moved out of `/tmp` 2026-10-19).
**Byte-sweep ground truth: heretic vs official TE — visual
  tower 100% IDENTICAL, language layers 0–49: exactly 9 tensors differ, all
  `self_attn.o_proj.weight` of layers 31–39** (abliteration-style mid-layer surgery; the
  heretic file's own metadata declares `minimax_h3_language_layers: 50`, no lm_head/norm —
  and `minimaxh3.cpp:8` reads only layers 0–49 pre-norm anyway, so the `…generation_tail_50_63`
  companions are irrelevant to H3/Radiance). The container therefore differs from the stock
  ref2va container in exactly those 9 requantised `te.blk.31-39.attn_output.weight`. TE
  checkpoint names carry NO `text_encoder.` prefix in `nm.ckpt` (the transformer/VAE folders
  do) — a `--changed` list with the prefix errors "no weight of this conversion reads".
- **P6 — parity & benchmarks**: §6 protocol.

## 5. Open questions — status

**RESOLVED 2026-10-18 (source-verified): Q1, Q2, Q3, Q4, Q5, Q6** → answers in
§5.1–5.6 below. **No open research questions remain; everything downstream is
execution (§4).**

Follow-ups created by the resolutions: (a) HF `transformer_ref/` download — **DONE +
VERIFIED 2026-10-18**: `/home/dustin/AI/models/MiniMaxAI/MiniMax-H3/transformer_ref`,
14 shards + index + config, 62 GiB on disk; index holds exactly the 638 base keys
(0 diff both directions), shard headers bf16+f32 with the full `[96768,2688]` adaLN
Linear, config carries no vsa/gate key. **P5 convert input is ready**; (b) deepstack
verdict RESOLVED, taps ON (§5.2); (c) the §5.2 presentation spec is now byte-level —
remaining parity work = implement it and diff, no more open research; (d) per-step
timing above 106,016 DiT rows on Radiance's fixed buffers is UNTESTED (§5.3).

### 5.1 Q1 — convert source: HF `MiniMaxAI/MiniMax-H3` → `transformer_ref/`

- `transformer_ref/` is a **complete alternative transformer** (14 shards, 61.7 GiB
  bf16, 638 tensors, 33.12 B) — **key-identical** to base `transformer/` and its
  `config.json` byte-identical; **now verified against the user's local download**
  (§5 status follow-up (a): key sets identical, dtypes bf16+f32, full adaLN).
  Ref2VA = same architecture, different values (a
  finetune): **zero ref-specific tensors anywhere** — ref/cond rows ride the existing
  `proj_in`/`audio_proj_in`/`context_embedder`, so P2b needs no new weights either.
- Names are plain HF-diffusers inside the folder; `checkpoint.cpp:542-559` namespaces
  folder components to `transformer.*`, which is **exactly what `minimaxh3.cpp`
  already reads** (`:729,744-800,845-848,870-880,892,948-953,1000-1018,1115,1140-1143`).
  The recipe (`data/recipes/minimax-h3-fp8.recipe`) carries only quant policy on
  Radiance-side names — the name map is C++ (`rad_arch.h:286-309`). **Verdict: zero
  new name-map entries.**
- Convert command shape (FastH3 pattern, recipe:11-16): synthetic dir with
  `transformer_ref/` renamed/copied to `transformer/` + sibling components symlinked
  from the base snapshot (`checkpoint.cpp:504-506` blesses linked folders), then
  `rad-convert <dir> --recipe minimax-h3-fp8.recipe --reuse minimax-h3-fp8.rad
  --changed <all 638 transformer.* names> -o minimax-h3-ref2va-fp8.rad`.
  **Do NOT point rad-convert at the `Ref2VA/` pipeline-packaging folder** — its
  `video_vae.` names won't match `vae.decoder.*` (`minimaxh3.cpp:1437+`); its
  `model_index.json` is still useful (records `sigma_shift_scales {video:12, audio:3}`).
- **Never use local ComfyUI files as convert sources.** The `*_ref2va_pruned_*` line
  folds AdaLN: per-block `adaln_proj.linear.weight [96768,8]` + `adaln_t_table
  [1025,8]` replace the real `[96768,2688]` Linear (that fold IS the pruning: 33.12 B →
  20.11 B, 13.01 B = 50×260.1 M) — Radiance's K=2688 ada GEMM (`minimaxh3.cpp:879-888`)
  has no folded-table path, plus flat ComfyUI names (fused `qkv_proj`, `mlp.fc1/2`,
  `condition_proj`, `final_layer`…); DaSiWa int8-convrot files are refused by
  rad-convert's importer (`docs/TOOLS.md:183`). The user's active workflow UNET
  (`DasiwaMinimaxH3_dasiwaHybridV4_int8`, 20 G) and the pruned fp8 files share one
  identical core tensor set — FL2VA/REF2VA/T2VA modes are a **conditioning contract,
  not separate weights**. Supporting pruned single-files = optional later feature
  (rename table + folded-adaln GEMM), not a prerequisite.
- `to_gate_compress`: absent from the **official** HF index (also from every local
  file) → official base/ref2va run dense; Radiance's VSA stays training-free (§3).
- Fallback single-file: `Comfy-Org/MiniMax-H3` `minimax_h3_ref2va_bf16.safetensors`
  (66.28 GB, header-verified) — ComfyUI-flattened names would need the §rename-table;
  not recommended.
- **Local pre-converted containers** (`/home/dustin/AI/models/StillDeadcode/`, user's
  HF pull 2026-10-18): `minimax-h3-fp8/minimax-h3-fp8.rad` (60.5 GiB — this IS the
  `--reuse` base above) and `fasth3-8-step-v2-fp8/fasth3-8-step-v2-fp8.rad` (62.3 GiB,
  FastVideo 8-step distilled path). Name-table string-scan of both (2393 tensors each):
  **zero `visual.*`, zero `to_gate_compress`** — declare-driven conversion, as expected;
  even the FastH3 container ships no trained gate, so Radiance's VSA coarse gate has no
  trained weights anywhere (§3 conclusion strengthened). Neither container can serve
  Ref2VA as-is — a container cannot be patched; the HF bf16 `transformer_ref/` download
  is still the required convert INPUT. Bonus find: both containers embed the Qwen3-VL
  chat template incl. the `add_vision_id` `Picture {n}:`/`Video {n}:` naming — reference
  text for the §5.2 risk-3 byte-diff is already local.

### 5.2 Q2 — TE vision path: near-complete, just not wired into H3

Radiance already ships a golden-tested Qwen-VL multimodal stack, one arch already runs
it in a **generation** pipeline:

| piece | where | state |
|---|---|---|
| Qwen-VL processor (smart_resize, patchify, `vision_start/pad*/vision_end` expansion, M-RoPE `RopeLayout`, `<T seconds>` markers, video temporal groups, **fps=2.0 default = Ref2VA's sampling**) | `core/mm/processor.{h,cpp}`, goldens vs transformers in `tests/mm_test.cpp` | EXISTS |
| Qwen3-VL vision tower (patch_embed/blocks/merger/**deepstack taps**, 2-D rotary) + encoder decl + LM-embedding scatter | `arch/common/rad_block_vit.h:176-217,300-306,380-395` | EXISTS |
| image/video decode (libav, mp4) | `core/mm/media.h:41,70`, `core/mm/mp4.cpp` | EXISTS |
| tower inline in a generator COND pass (`mm_embd`/`mm_rows` scatter) | `arch/qwenimage21/qwenimage21.cpp:7-9,1197` + `core/gen/driver.cpp:470-560` | EXISTS — the template to mirror |
| media in H3 (`videos.cpp` fields, `VideoRequest`, `RadVideoGenBatch` — `rad_runtime.h:429-471` ABI) | — | MISSING (P0 plumbing) |
| H3 encoder declaration / `visual.*` weights in `minimax-h3-fp8.rad` | `minimaxh3.cpp:605-614` maps text layers only; rad-convert writes only declared tensors (`rad_convert.cpp:649-651`) | MISSING — re-convert must declare `model.visual.*` |

ComfyUI side confirmed: audios **never** enter the TE as tokens (label-only
`{"type":"audio"}`, `nodes_minimax_h3.py:336,355`) — zero audio-token work for P1.

**TE weights source (user requirement: keep the uncensored TE):** the production
workflows run `qwen3vl_32b_h3_ultra_uncensored_heretic` (llmfan46 fine-tune of
Qwen3-VL-32B-Instruct). The int8-convrot copy CANNOT be ingested (rad-convert refuses
convrot, `docs/TOOLS.md:183`), but the **bf16 source is already on local disk**:
`/home/dustin/AI/ComfyUI/models/text_encoders/civitai-minimax/qwen3vl_32b_h3_ultra_uncensored_heretic_bf16.safetensors`
(48.0 GiB, 902 bf16 tensors, header-verified): `model.embed_tokens` (5120-dim = H3
`text_dim`), exactly 50 language layers, **no `lm_head`, no final norm** (`__metadata__`
documents `minimax_h3_language_layers: 50` / `minimax_h3_final_norm: false` — same
pre-norm layer-50 hidden H3 reads), + 351 `visual.*` tensors **including 3 deepstack
taps** (`visual.deepstack_merger_list.0-2.*`, 18 tensors). So one file supplies BOTH
the heretic `te.*` and the missing `model.visual.*` tower for the ref2va convert.
Name-map check at convert time: tensors are ComfyUI-flat `visual.*` while
`rad_block_vit.h` names are HF-style `model.visual.*` → likely a one-line alias.
TE choice is a **container decision, not code**: develop against the official fp8 TE
already in `minimax-h3-fp8.rad`, then emit a heretic-TE variant with
`--reuse --changed <te.* + visual.*>` from this file when wanted. NOTE-CONTINUITY §4's
A/B (fp8 official vs int8 heretic on `<Subject n>` binding) becomes: official-fp8
container vs heretic-bf16→fp8 container, same code.

Risks — **all three RESOLVED 2026-10-18** (ComfyUI clip trace, paths under
`comfy/text_encoders/` unless noted):

1. **DEEPSTACK: CONSUMED — port with taps ON.** The H3 clip's Qwen3-VL LM injects
   deepstack visual features into the language stream (`qwen3vl.py:94-113` applied at
   `llama.py:1041-1042`, tap indexes [8,16,24] `qwen3vl.py:17`); dropping taps changes
   the hidden state whenever images are present, so parity REQUIRES them — and
   `rad_block_vit.h` already demands opt-in for tapped towers, so the happy path
   matches (no fork). Weights exist (incl. in the heretic bf16 file above).
   Conditioning hidden = **post-layer-50, final_norm=False** (`llama.py:366-374`,
   `sd1_clip.py` `layer="last"`), fp32→bf16 cast (`model_base.py:2183-2188`) —
   exactly what `minimaxh3.cpp:7-9` already does.
2. **kMaxText** → sized in §5.3 (cheap to raise; the DiT row budget is the real wall).
3. **Presentation byte-spec (was risk 3)** — measured from `comfy/text_encoders/
   minimax.py` + bundled tokenizer: image = `<Picture i>: ` (6 tok) + `vision_start` +
   **A/1024 pads** + `vision_end` (patch16×merge2 → 32 px/token, sides mult-32;
   `qwen_vl.py:35-46`, `qwen3vl.py:19`); audio = label-only `<Audio j>: ` (5 tok) —
   audio NEVER enters Qwen (`minimax.py:177-178`); video = `<Video k>: ` (6 tok) + per
   **2-frame blocks** (2 fps, odd tail padded): `<T.T seconds>` (6 tok; 7 for ≥10 s) +
   `vision_start` + A′/1024 pads + `vision_end` (`minimax.py:164-190`); prompt = raw
   BPE, NO chat template. **Parity trap — RESOLVED, NO processor change (2026-10-19,
   minimax.py source re-read):** the earlier "0.2, 0.8, 1.2, 1.8…" sequence here was a
   derivation error. The node builds `timestamps = [i/2.0 for i in range(T)]` and each
   block's stamp is `(ts[i]+ts[i+1])/2` of consecutive frame PAIRS = `k + 0.25` →
   `"%.1f"` (Python round-half-even) → **0.2, 1.2, 2.2, … per block** — byte-identical to
   Radiance's existing golden `tests/mm_golden.h:48` `{ "<0.2 seconds>", "<1.2 seconds>",
   "<2.2 seconds>" }` (glibc `%.1f` matches: 0.25→"0.2", 1.25→"1.2"). Also confirmed from
   source: labels `<Picture %d>: ` / `<Video %d>: ` / `<Audio %d>: ` are BPE'd as SEPARATE
   strings (per-segment `add_text` — mirrors the image path's `render()` marker-split); the
   video label precedes ALL its block stamps; odd frame counts repeat-pad the last frame AND
   duplicate its timestamp; the user prompt rides LAST, after all refs; an all-empty entry
   list falls back to pad token 151643. **Critical for
   P3:** the DiT `text` segment = the FULL TE sequence INCLUDING vision pads
   (`text_len = context.shape[1]`, `model.py:358,659`; pads ride adaLN tag 0 via
   `minimax.py:75-82`→`model.py:720-733`) — each reference is therefore represented
   TWICE (TE pads in the text segment + its own ref rows).

### 5.3 Q4 — token/VRAM budget at 9/3/3 (sized 2026-10-18, two independent passes reconciled)

Two independent sizing passes ran (first trace + a 63-verification recheck with every
constant read from source). They agree on all structural verdicts and differ only in
per-case scenario assumptions, so both scenario sets are kept. Rates: image ≈ 1 TE
token AND 1 DiT row per px²/1024 (both resize to mult-32); video ≈ **1016 TE + ~7190
DiT rows per ref-second** @1344×768 canvas; audio label-only TE + `2·round(s·40)`
DiT rows. DiT design budget: **106,016 rows/step** (`minimaxh3.cpp:128-131,189`);
TE: `kMaxText = 2048`.

| case | TE tokens | vs 2048 | DiT rows/step | vs 106,016 |
|---|---|---|---|---|
| **Worst 9/3/3** (9 img @2048 short-edge + 3×15 s video + 3×15 s audio; recheck assumed 16:9 2048×3648 imgs = 7,296 tok each, first pass square) | **112,977 / 84,189** | 55× / 41× | **594,655 / 485,931** | 5.6× / 4.6× — **infeasible** |
| **DaSiWa production chain** (recheck: 4 photos 1024²→match 608², 3 s tail cond 8,470 rows, 3 s audio, target 12 s 704×544) | **2,977** | **1.45× — BREAKS 2048** | 46,649 | 0.44× ✓ |
| production-min (first pass scenario: 1 match-img + 294 f 704×544) | 1,869 | fits (179 headroom) | 35,748 | 0.34× ✓ |
| **Heavy grid** (first pass: max target + 1 img + 1×15 s video + 15 s audio / recheck: 9 match imgs @1344×768 + max target) | 17,772 / 10,635 | 8.7× / 5.2× | 233,004 / 123,673 | 2.20× / 1.17× ✗ |

**Verdicts (reconciled):**
(i) **kMaxText re-anchor is MANDATORY, not optional** — the DiT `text` segment is the
FULL TE sequence incl. vision pads (§5.2 risk 3), and the production chain with 4
match-mode photos + a realistic brief needs ~2,977 TE tokens > 2,048. The earlier
"production fits untouched" held only for the 1-photo scenario.
(ii) **kMaxText costs were undercounted by the first pass** — each TE token is also a
DiT text-seg row, so ~**398 KiB/rank/token** (284 KiB TE/refiner/lane.text at
`minimaxh3.cpp:617-625,713-739,1925` + 114 KiB DiT activations at `:941-978`; the
first pass's "+308 MB @16 K" counted lane.text only — SUPERSEDED). Baseline occupancy
is ≈23.3 GiB/rank of 32. **→16,384: +5.5 GiB/rank (≈28.8 GiB — fits, RECOMMENDED for
the ref2va re-convert); →32,768: +11.7 → ~35 GiB — does NOT fit.**
(iii) **Hard caps beyond row count**: VSA dispatch `tiles > 3072 → RAD_E_SHAPE`
(`r4d_vsa.hip:1077`; dense-sink ceiling ≈86.8 K rows at 15 s@1344×768 targets) ⇒
absolute row ceiling **196,608**; `ctx->max_tok` is architecture-fixed at build
(probe gen_rows, `engine_bringup.cpp:920-960`; no `--max-num-seqs` analog for video),
and with the sequence unsharded every ref row must be provisioned in persistent
buffers on **both** ranks.
(iv) **Worst 9/3/3 fails every cap simultaneously** (TE ×55 → 30.6 GiB/rank TE-side,
rows ×5.6 → 64.8 GiB/rank buffers, 9,401 tiles > 3,072) → **hard-rejected at the API,
never truncated** — ComfyUI truncates nowhere (tokenizer `max_length=99999999`,
`qwen3vl.py:151`); no knob tier saves it.
(v) **Model-vs-measured per-step cost**: the rows²·dense-fraction model puts the
production chain at ~3.5× baseline per-step, but NOTE-CONTINUITY §3 MEASURED the tail
at only **+31%** — sparsity absorbs video rows and added sinks contribute ≈P²+2PV,
so use the model to rank scenarios, never to predict; P6 measures.

**Recommended defaults:** `ref_image_size=match` (ComfyUI default too; `max` = up to
7,296 tok/rows per 16:9 image, 7.2× a match image, and DiT applies NO area cap on
refs, `model.py:396-411`); per-ref video cap **≤5 s** default (first pass) — the
recheck is stricter: at a 15 s max target even ≤2 video refs blow the <2×/step goal,
cap at **≤3 s @ ≤704×544** and expose `max_ref_rows`/`max_ref_tokens` with
auto-downscale; API validation tiers mirroring node caps (≤9/≤3/≤3+3, ≥5 f, 17k+5,
gen-length truncate) plus an engine guard = min(VSA tiles ≤ 3072, rows ≤ ~145 K to
keep DiT buffers ≤16 GiB/rank, TE ≤ kMaxText); `kMaxText` tiers **2,048** (t2v) →
**16,384** (recommended: 9 match imgs 10,635 ✓, heavy grid TE ✓) → **32,768 rejected**
(does not fit). Untested: per-step timing >106 K rows on fixed buffers, and the
recheck notes official ckpt ships no trained VSA gate → refs ride dense selection
until Radiance adds training-free gating (P4) — P6 benchmark items.

### 5.4 Q6 — continuity tail: cond rows outside the sampler state (port invariants)

Mechanism spec is in §3 ("Continuation / cond rows"). The port checklist:

- Tail = clean **latent slice** of the previous clip's *sampled* AV latent, at
  `resolved_frame_index=0`; never a VAE decode/encode round-trip. Persist sampled
  latents per clip (ClipStore pattern) and cache across takes of one scene.
- Cond/ref rows never in sampler `x` (target streams only, velocity out for the last
  two segments); rebuilt from stored latents every forward.
- Timestep/mod: `t = 1-σ`; per-seg `max(t_stream, pin)`, pins 0.999 (video-class:
  `cond`,`ref_img`) / 1.0 (audio-class: `cond_audio`,`ref_audio`); mod row =
  `t_index·3 + tag` with the §3 seg_tag map; add the deterministic +0.1% noise on
  video-class cond rows (same seed every step).
- Grid math: pixel frames `17k+5` ↔ latent `5k+2`; per-latent-frame coverage
  `(1,4,4,4,4)`; audio 40 Hz, `round(frames·40/24)` with ±1 window tolerance; overlap
  snap {5,22,39,56,73}→{2,7,12,17,22} (3 s tail = 73 px = **22 latent**); extension
  multiple of 17; audio overlap via boundary-diff. **Do not hardcode 13 tail frames or
  FRAME_RESCALE=4** (RoPE time unit = 1/40 s; video advances 5/3 per pixel frame).
- Stereo rows **channel-major stacked**, `2·rt`, `w` = grid low/high per channel.
- Audio flow carry `audio_scale = shift_v/shift_a = 12/3 = 4` — undo before the DiT,
  re-scale velocity after (`model_sampling.py:338-357`, `model.py:605-615,641-644`).
- Window `overlap+extension` sampled unmasked; trim overlap tokens, concat behind the
  byte-identical previous latent → acceptance tests: **zero replayed frames** and
  motion/lighting continuity at the seam (frame-equality with the tail's last frame is
  wrong by construction — the first new frame is its temporal *successor*).
- Replicate rejections fail-clean: prev canvas/dtype/fps-24/grid checks, head-guide
  conflict inside the overlap, audio length vs global boundary
  (`continuation_nodes.py:80-111,444-457,657-662`). Engine-side ref-video rules: ≥5
  frames + `17k+5` trim + truncate to gen length; the 2–15 s / ≥2.01 s items are
  policy-level (API validation), so a ≥3 s tail requirement is met with room to spare —
  actual ceiling = Q4's token/VRAM budget.

### 5.5 Q3 — VAE encode: containers and plugin are DECODE-ONLY; encoder is net-new work

- **Radiance has no encode entry point anywhere.** `minimaxh3.cpp` declares only
  decode networks (video decoder `declare_video :1344`: `vae.pqc`=post_quant_conv,
  `proj_in`, 36 `vae.blk.*` ViT blocks, `proj_out`; audio decoder `:1226-1328`:
  `aud.in_proj`/`conv_pre`/`s0..6`/`act_post`/`conv_post`); there is no
  `rad_declare_encoder` call in the arch, and the generator pass enum is
  VCOND/step/ADECODE/VDECODE — **no VAE-encode pass** (`abi/rad_runtime.h:420-473`).
  Container scan of `minimax-h3-fp8.rad`: all 470 `vae.*` + 780 `aud.*` names are
  decoder-side; **zero encoder weights** — structural: rad-convert writes only
  declared tensors (`rad_convert.cpp:651`) and the plugin declares none. Encoder
  *config* already sits in container meta (`block_out_channels`,
  `spatial_padding_mode=reflect`, `audio_vae.encoder_rates`, …).
- **HF source HAS them** (verified from the checkpoint indexes): video `vae/` 118
  encoder tensors (`encoder.conv_in/down_blocks.0-5.resnets/downsamplers/norm_out/
  conv_out`→48 + `quant_conv`), ~0.36 GB; audio `audio_vae/` 173 (`encoder.block.*`
  DAC stack with **37 weight-norm pairs — fold with existing `map_wnorm`
  `minimaxh3.cpp:1172`**, `pre_block.*` incl. causal 8-head `attn.qkv`, `mean_proj`),
  ~0.30 GB f32. Both declare cleanly in the P1/P5 re-convert pass.
- **Architecture ≠ the shipped decoders** (porting spec): video encoder is a 3-D
  causal-conv ResNet (`EncoderFCN3D`, `comfy/ldm/minimax/vae.py:176-238` — reflect
  spatial pad, **front-only zero temporal pad**, per-frame TemporalIsolatedGroupNorm,
  asymmetric-pad strided `Downsample3D`, 17-f temporal chunks drop-last-3
  `:647-659`, 256-px/64-overlap spatial tiles with latent blend `:564-590`, takes
  **mean only, no sampling** `:775-792`); audio encoder is the DAC strided stack,
  strides (2,4,4,5,5) hop 800→40 Hz + Snake1d + `AttnProjection` causal attn +
  2048→32 pool (`audio_vae.py:155-213,253-269,374-443`). **Images encode through the
  video VAE with t=1 → 1 latent frame** (`nodes_minimax_h3.py:310-311`,
  `vae.py:779-786`) — 1.0 t-span, matches §3.
- **Kernel deltas vs existing libr4d**: extend `r4d_rad_conv3d_ndhwc` (per-axis
  stride + causal + reflect pad + T==1 case; hot path), ~~per-frame-GroupNorm variant
  of `group_norm`~~ **RESOLVED 2026-10-19 — §5.13: zero new op, per-frame launches; the
  only kernel change was the device chunk width for 4-channel groups**, ~~new strided
  even-kernel conv1d~~ **LANDED 2026-10-19 — §5.12**, ~~AttnProjection reduce~~ **RESOLVED
  2026-10-19 — §5.14: zero new kernels, a composition incl. one conversion-time fold**
  (+ encode tiling, adapt `vtile_gather/stitch`).
- **Effort ≈2–3 wk** (kernels 4–6 d w/ libref+selftest parity, encode graphs 4–6 d,
  recipe+re-convert 1–2 d bundled into P5, golden parity 2–3 d); **audio-only ≈1 wk**.
  Encoders are deterministic (mean-only posterior) → golden parity well-defined;
  compute trivial (0.36/0.30 GB, seconds per clip).
- **Sequencing**: **P2b continuity needs NO encode at all** (tail comes from the
  previous clip's sampled latents) — the tail path can ship before P2. Only
  `<Picture>`/`<Video>`/`<Audio>` refs require this section's work.

### 5.6 Q5 — author stance & endpoint shape (Codeberg/HF scan 2026-10-18)

- **Community PRs genuinely land**: 27 community PRs merged of 127 total items; issues
  are open to outsiders; no CONTRIBUTING/ROADMAP exists — the "invitation" is practice,
  not policy. **Merge culture caveat:** the author habitually closes PRs unmerged after
  hand-copying them into his internal tree ("manually merged … closing this PR to mark
  it as done", #38/#47/#75/#76/#82/#88) — closed+unmerged ≠ rejected; authorship
  preserved. Also #75: Codeberg sometimes refuses fork→upstream PRs → be ready with a
  manual fetch handoff.
- **Zero prior discussion of this feature**: keyword sweep of all 127 issue/PR
  titles+bodies — `ref2va`/`minimax`/`continuity`/`input_reference` = 0 hits; none of
  the 10 open PRs touches the video path. Nothing to duplicate, nothing blocks us.
- **Endpoint shape — evidence says FIELDS ON `/v1/videos`, not a new endpoint**: the
  parser already accepts-and-refuses OpenAI's `input_reference`/`task`/`conditions`
  in place (`videos.cpp:135-189`, `supported: ["t2va"]`); the schema is a deliberate
  OpenAI+SGLang superset (header comment `:12-18`) and every modality rode its
  standard OpenAI path (edits, ASR); SGLang expresses H3 conditioning through those
  same fields. Ref2VA slots into the existing refusal stubs (new `task` value /
  non-empty `conditions`). No precedent for a model-specific path in this repo.
- **ABI**: `RAD_ABI_VERSION = 22` at upstream main; adding `RadVideoGenBatch` media
  fields (`abi/rad_runtime.h:429-471`) REQUIRES a bump; the author has explicitly
  blessed bumps when asked (issue #105: "feel free to bump the abi, its the cleaner
  solution"). **Conflict watch:** open PR #118 already proposes 23 — re-check at PR
  time and take the next free number. Test gates: `ctest -L fast`, `-L gpu`, `-L oot`
  (out-of-tree plugin rebuild).
- **HF**: author publishes 20 `.rad` repos, **no ref2va container** (convention: one
  `<model>-<quant>.rad` + recipe + READMEs) → our convert-from-HF plan is the only
  source; offering `minimax-h3-ref2va-fp8.rad` (+ recipe) is a novel contribution.
- **Verdict: ask first, briefly** — the author answers shape questions pre-PR (#105 →
  #107 flow is the template). Proposed issue (adapt as needed, post BEFORE building
  the PR stack):

  > **Ref2VA (reference image/video/audio conditioning + continuity tail) for
  > MiniMax-H3 — shape check before I build.** Planning a community PR making
  > `/v1/videos` serve Ref2VA: accept `input_reference` + non-empty `conditions`
  > (≤9 img / ≤3 vid / ≤3 audio) with a new `task` value on the existing endpoint,
  > threading reference latents through `RadVideoGenBatch` — which bumps
  > `RAD_ABI_VERSION` unless you'd rather I keep it out of the batch. (1) existing
  > `/v1/videos` + OpenAI `conditions`/`task`, or a separate endpoint? (2) should the
  > PR ship the rad-convert recipe, and would you host
  > `minimax-h3-ref2va-fp8.rad`? Happy to build in the #105 shape, and fine with you
  > if it lands via your internal tree.

### 5.7 The tower presence gate — how a container decides it serves refs (RESOLVED 2026-10-18)

One question had no answer in the plan: `vision_config` sits in EVERY MiniMax-H3 container's
meta (the recipe copies it from the checkpoint config), so "meta names a tower" cannot be the
test — the stock `minimax-h3-fp8.rad` must NOT offer a ref2va task it cannot run. The gate is
a WEIGHT probe inside `VisionTower::declare` (`arch/common/rad_block_vit.h`), run first and
before any shape is read: the stem's copy map
`v.patch.weight → model.visual.patch_embed.proj.weight` is registered there (so the stem is
declared *through* it — a COPY map declared twice is refused), and `rad_weight_encoding` is
asked for it. Name maps resolve against the checkpoint at convert and against the container
at load, so both sides answer the probe the same way — there is no way to convert a tower
into a container the loader would then deny, or vice versa. Not found: print
"vision_config names a tower the checkpoint does not ship; declaring without it", declare
nothing, `on` stays false; the dangling map entry is inert (maps are consulted by name at
resolution and are not serialized into the container). H3 chains it as
`m.media = present(meta) && ctx->max_enc_patches > 0; …; m.media = m.tv.on`
(`minimaxh3.cpp:1984-1988`) and declares the 9/3/3 ceilings only when `m.media`
(`:2058-2064`); the engine's auto `mm_max_patches` keys on the `radiance.encoder` meta a
tower convert writes (`engine_bringup.cpp:1012-1027`), so a tower-less container also gets
`max_enc_patches == 0` — the driver's tower refusal and the zeros the server reports are
the same fact seen twice. Verified live: the tower container answers `16384/9/3/3`, the
stock ref2va/uncensored/base containers answer `0/0/0`.

*Test-linking invariant found the hard way (regressed 8 fake-device tests, then fixed):*
`tests/sched_fakes.h` fakes `rad_dev_alloc`/`rad_memcpy_async` as **weak** symbols; linking
ANY core/mm unit that drags the real device layer — anything through `processor.cpp`'s
metadata reader into `dispatch.cpp.o` — silently replaces them with the real HIP backend
against a null stream, uploads fail, and every faked driver returns null passes. This is
why `core/mm/item.cpp` exists ("The batch builder links only this unit from core/mm"):
`PromptMedia::layout()` — pure computation over the placed items — now lives there, not in
`processor.cpp`. Keep device-free mm logic in `item.cpp`.

### 5.8 LoRA on the generator path — why convert-time merge, and the runtime-support ask (LANDED 2026-10-19)

The DaSiWa workflow's active adapter is a video-path LoRA. The engine already loads
adapters at runtime — but deliberately NOT for generators: `lora_declare`
(`core/lora/declare.cpp:79-87`) refuses any pass-driven program ("is an image, audio or
video generator"), because the adapter ops follow the text engine's targeted per-token
GEMMs (`RadBatch::lora_slot`, ABI 18, applies per row of a token batch); a generator's
whole-tensor pass GEMMs have no per-row slot to carry. Making it real is engine surgery
across the pass executor, pools and ABI — a multi-day design decision that belongs to the
author, not to this PR. The PR-correct interim is **merge-based parity**: quantisation is
deterministic, so merging the adapter into the bf16 source before `rad-convert` serves
exactly that adapter at that strength — the same math ComfyUI's `Apply LoRA` performs on
the checkpoint.

Implemented on this branch as three `rad-convert` flags (tests: `convert_test` +5 cases
covering merge/changed-set/reuse, wrapper-prefix + `.alpha` scaling, strength, refusals,
fused-map row windows, and map refusals):

- `--lora FILE|DIR` (repeatable, in order): PEFT dot-form pairs
  (`.lora_A.weight`/`.lora_B.weight`, optional rank-0 `.alpha`; `_down`/`_up` spellings
  read too). `W += strength * (alpha/r where stated, else 1) * B @ A` in f32, cast back
  to the target's dtype. DoRA, quantised-import sources and GGUF sources are refused, not
  silently downgraded. `--lora` + `--in-place` is refused (in-place copies the unmerged
  bytes). Every merged tensor joins `--changed`, so `--reuse` can never copy an unmerged
  weight back — the mark, not the bytes, is what `--reuse` matches on.
- `--lora-strength F`: one strength for all given adapters. (The Hermes gap "no strength
  knob anywhere on the video path" is closed by re-converting per strength; the interim
  halved-alpha copy recipe is subsumed.)
- `--lora-map FILE`: JSON `{"strip": [...], "modules": {"<pat with one *>": ["<target>", …]}}`
  for adapters trained on a module layout the checkpoint does not spell. The `*` captures
  one run (the block index) and fills every target's `*`; several targets mean the module
  is their concatenation, so B's rows split along the targets in listed order by each
  target's own row counts (refused unless they sum to B's rows); a rule whose target the
  checkpoint lacks is a refusal, never a miss. Resolution order: `<module>.weight` direct,
  then strip-prefixed, then first matching rule (rules are matched against the stripped
  name; insertion order wins).

The real H3 adapter (416 tensors = 4 modules × 52 blocks × 2, r=64, no `.alpha`) needs
exactly this: ComfyUI's fused `diffusion_model.blocks.N.attn.qkv_proj` (B [21504,64] =
3×7168) maps to `transformer.transformer_blocks.N.attn.to_{q,k,v}.weight` — note the
`transformer.` **folder prefix**, which is how a pipeline directory registers names — and
`diffusion_model.token_refiner.blocks.N.*` maps to `transformer.token_refiner.refiner_blocks.N.*`
(same four modules; block counts 50+2 match the checkpoint's key families exactly). The
map is `convert-ref2va/lora-map-h3.json`; measured: merge ~2 min CPU (1.2 T MACs, row-buffer
rank-1 form with A's f32 copy hoisted out of the inner loop — the per-element dtype switch
in `rad_load_f32` made the naive form 40+ min of scalar work; do not simplify it back), full
`--reuse` convert ~5 min wall (559 weights written, 2469 copied).

**Built and verified 2026-10-19:** `minimax-h3-ref2va-fp8-tower-lora.rad` and
`minimax-h3-ref2va-uncensored-fp8-tower-lora.rad` — 3028 weights each (312 merged = 6 targets
× 52 blocks; the merged set joins the 351 `visual` sources in `--changed` → 663 read by 559
weights), name sets comm-identical to their tower bases, and both boot and answer
`16384 / 9 / 3 / 3` — the Hermes production gate holds on the merged containers. (Map-rule
order note: `nlohmann::json` sorts object keys; the loader parses `ordered_json` so rule
insertion order is honoured.)

**Runtime generator LoRA — upstream issue draft** (ask-first culture, §5.6; post only if
the PR stack proceeds):

> **Runtime LoRA for generator programs — design question.** `lora_declare` refuses
> pass-driven programs, so today a video/image/audio LoRA means a merged container per
> (adapter, strength). That is correct but coarse: switching costs a re-convert, and
> strength is baked. The text path's mechanism (`lora_slot` per token-row) doesn't map
> onto pass GEMMs; the natural generator shape is a per-GENERATION adapter applied as a
> fixed delta into the pass's weight views (pool-side, not per-row), with `model` on
> `/v1/videos` naming a loaded adapter. Is that a direction you'd take, do you already
> have it in the internal tree, or should generators stay merge-only by design? Happy to
> prototype against whatever you'd bless; meanwhile the convert-time `--lora` path works.

### 5.9 The Ref2VA sequence packing — `arch/minimaxh3/rad_ref2va_pack` (LANDED 2026-10-19)

P3's host half and P4's rule landed as one new device-free TU pair plus a golden-vector test;
`docs/NOTE-P3-PACK.md` carries the line-referenced extraction from the reference `PackedLayout`
that the port follows rule for rule.

- **What it is**: `ref2va_build(text_len, target, keyframes, refs, &pack)` lays out the full
  presentation — `[text | keyframe rows | per-reference blocks | audio | video]` — emitting the
  segment table (kind, rows, update flag), the planar `[3, rows]` f32 rotary coordinates (the
  exact shape `R_ROPE` holds; staging stays a copy), and `ref2va_mod_rows(t_v, t_a)` reproducing
  the reference's per-step `t_row * 3 + tag` modulation numbering, including the collapse when a
  live timestep passes the 0.999/1.0 pins. Pinned rows never enter the sampler: `update` is true
  only for the two target segments, which are always last — the sampler's state is exactly them.
- **Parity**: `tests/data/ref2va/golden.json` is produced by `gen_golden.py`, which AST-extracts
  the reference's constants, grid helpers and `PackedLayout` class from the model file of a
  ComfyUI install and execs them on torch alone — every number is the reference's own float64
  output, not a re-derivation. `tests/ref2va_pack_test.cpp` (compiles the TU directly, no core,
  no device) compares positions **bit-exact** after the f32 cast, plus segments, both latent
  spaces' position/update lists, and mod rows at three timestep probes — across five cases:
  plain t2va, fl2va keyframes at the timeline origin and mid-timeline, all four reference kinds
  with a silent target and a clip's audio packing before its video, a realistic chain signature,
  and an empty prompt. With no references the pack reduces to the engine's historical
  `[text | audio | video]` coordinates row for row (checked against `positions()`); 207 checks,
  8 cases, 0 failures.
- **P4, concretely**: `vsa_geom` already makes every segment before the video grid a dense
  64-row prefix — so ref/keyframe rows become attention sinks for free once the driver lists
  them in `seg_cu` ahead of the targets. No kernel change; `R_SEG`'s four-slot `seg_cu` and the
  mod table's two-timestep columns are the two ceilings the wiring must widen (NOTE-P3-PACK
  "What the engine wiring still owes" lists the four consumption points).
- The plugin CMake grows to two sources; nothing in core changes yet — P2b's tail and the
  driver-side `seg_cu`/step wiring are the next increment, and they are what P1–P4 parity
  gating (§ gate list above) waits on.

### 5.10 The encode geometry — `arch/minimaxh3/rad_vae_geom` (LANDED 2026-10-19, P2's host half)

The half of §5.5 that is pure host arithmetic landed as a second device-free TU pair with a
golden-vector test; `docs/NOTE-P2-ENCODE.md` carries the line-referenced extraction from the
reference VAE sources it follows rule for rule.

- **What it is**: `vae_latent_frames(T)` (the clip plan: T=1 → 1; else `5·ceil(T/17) − 3` —
  padded clips, trailing drop), `vae_clip_latents` (the raw causal chain `ceil(T/4)`),
  `vae_audio_latents(samples)` (right-pad to the hop, `ceil(s/800)`), `vae_split_tiles` (the
  256-px/64-overlap tile plan with the reference's round-robin 16 px slack spread, real clamped
  slice lengths, pixel overlaps), and the verbatim normalization tables (`kVaeLatentsMean/Std`,
  `kVaePixelMean/Std`).
- **Parity**: `tests/data/vae_geom/golden.json` is generated by running the **reference's real
  encoders on CPU** (its `EncoderFCN3D` + wrapper methods bound to a shim self; its DAC audio
  encoder; shims only for the CUDA-only fast paths the CPU guards skip). 408 checks, 7 cases:
  the latent lattice through `encode()` end to end — reproducing the continuity snaps
  {5,22,39,56,73} ↔ {2,7,12,17,22} as the network's own output — tile plans against the
  reference's literal returns, blended tile grids keeping exact size, audio latents at 13
  lengths, constants bit for bit.
- **What remains of §5.5** is the device half: the last kernel delta — the `conv3d_ndhwc`
  extension (per-axis/causal/reflect/T=1; the strided conv1d landed §5.12, per-frame GroupNorm
  resolved §5.13, AttnProjection resolved §5.14) — the
  encode graphs, declaring + re-converting the 118+173 encoder tensors, and value parity —
  the driver's encode pass and the FFmpeg-gated ingest ride on top.

### 5.11 The pack drives staging: `positions()` and `seg_cu` (LANDED 2026-10-19, P3's first two consumption points)

`minimaxh3.cpp`'s `positions()` — the rotary coordinates of the plain `[text | audio | video]`
step — no longer hand-computes them: it calls `ref2va_build(text_len, geom, {}, {}, &pack)` and
copies `pack.pos`, so the reference-media path and the plain generator share one layout
implementation, and the golden already pinning the pack (against the reference's own float64
output) now covers the shipped t2va rope too. Byte-level audit: the hand-rolled grid recovered
the axis step as `((1-ratio)/2 + ratio) - (1-ratio)/2`, which cancels to *exactly* 0 on the grid
row that is mathematically centred, where the reference's `arange*ratio/n` leaves a ~1e-15
float64 residue; sweeping all 1242 legal sizes, exactly 10 sizes × 1 centre row had a
differing f32 coordinate. The swap makes those rows the reference's values — the last bit of
one rope angle, and the direction of travel for every later consumption point. The row/segment
contract the swap leans on (three segments at `[0, n_text, n_text+2·n_audio, rows]`, update
flags 0/1/1, channel-major audio, the 1,4,4,4,4 time pattern) is pinned by
`pack_plain_matches_the_three_segment_order`, and the built plugin was probed directly at the
drift sizes. The second consumption point: a new **optional** plugin export
`rad_arch_video_segs(g, cu, cap)` (`rad_builder.h`, additive — no `RAD_ABI_VERSION` move; loader
→ startup → `VideoDriver::init`), and `stage_geom` stages R_SEG from it — validated against
the geometry, falling back to the driver's own identical table on absence, disagreement, or a
segment count past `R_SEG`'s four int32s. `video_gen_test` drives all three paths. Both points
are byte-neutral for every plain request; they make the pack — not the driver — the author of
the staged sequence once reference rows exist. On-device boot check: the P5 container serves
clean with the new export threaded through loader → startup → driver (`--tp 2
--vision-tower-placement ram`), and a 512×256/2-step job plans, computes its positions through
the pack and queues; the clip then stops at the documented §6.1 blocker — no libavcodec on this
box, the muxer fails at t≈0 and no step runs. Steps stay unverified on-device until the FFmpeg
packages land; the staged-table path itself is driven by `video_gen_test`'s fakes.

### 5.12 `conv1d` learns the strided form (LANDED 2026-10-19, P2's first kernel delta)

The DAC audio encoder's five downsamples (`strides (2,4,4,5,5)`, §5.5) are all PyTorch
`Conv1d(cin, cout, kernel_size=2·stride, stride=stride, padding=ceil(stride/2))` — even kernel,
strided, explicit padding — which the engine's `conv1d` was specced to refuse (odd `k`, no
stride; the transposed `conv_t1d` is the vocoder's upsample and does not apply). The op now
takes two more optionals, `stride` and `pad`, with a deliberately narrow contract: `stride` > 1
requires `pad` explicit and `causal` off (the causal form keeps its own left-only padding; an
explicit `pad` without a stride is refused rather than silently accepted), it selects **bf16
only** (the encoder runs bf16; the f32 vocoder rows refuse it as `RAD_E_DTYPE`), and it is one
sequence — `cu_seqlens` `[0, rows of x]`. Output row `m` reads
`x[m·stride + tap·dil − pad]`, zeros outside the sequence, and that zeroing gather is already
what the kernels do per sequence: the device delta in both bf16 paths (iGEMM and direct) was
one multiplication in the input-row index, and the explicit-pad rule generalises the derived
`dil (k - 1) / 2`. Because `cu` bounds live on the device, the launcher checks the parts it can
read host-side — one sequence and `y` rows exactly `(x_rows + 2·pad − dil·(k−1) − 1)/stride + 1`
(PyTorch's output length; a divisible input gives exactly `x_rows/stride`, which is what the
wrapper's right-pad guarantees) — and libref, which reads `cu` on the CPU, checks the full rule
including `cu`'s ends. All three plugins declare the widened schema (libr4d rows, libref,
libavx; `rad-schemas` diff: 161 shared ops, 0 disagreements), `docs/OPS.md` carries the new
row and spec, and `ref_conv1d_shape` describes the strided operands (`M` stays the input rows;
`y` gets the formula's rows) so the shape-driven test drawers reach the form. Verification:
`ref_test`'s `conv1d_strided_downsamples_one_sequence` pins exact bf16 values (8 rows → 4,
`[20, 40, 60, 44]`) plus five refusals (pad-without-stride, stride-without-pad, two sequences,
a `y` off the formula, stride on f32); `avx_check` draws strides 2/4/5 against libref plus two
hand-built cases; `r4d_selftest` runs the tiled (cin 16, cout 32/128) and direct (cin 5/6)
kernels at all four strides against the host reference on-card, with the same refusals. The
DAC's odd stride 5 included — `pad = ceil(5/2) = 3`, `k = 10` — so every downsample the
encoder graph will emit is expressible today. Nothing existing changed behaviour: with the
optionals absent every code path reduces to the previous expressions, byte for byte.

### 5.13 Per-frame GroupNorm: no new op, one narrower chunk (LANDED 2026-10-19, P2's second kernel delta)

`TemporalIsolatedGroupNorm` (`vae.py:122-130`, built by `group_norm_3d` :133-134: always 32
groups, `eps` 1e-6, affine) computes its statistics per frame. The engine's `group_norm` reduces
over ALL of `y`'s rows — so per-frame means **one launch per frame**, and no contract change is
needed to make that exact: encoder volumes are batch 1, NDHWC frames are contiguous row blocks,
the launcher takes `M` from `y`'s row count and honours both operands' row pitches, so the
driver slices `x` and `y` at frame boundaries, reuses one `stats` scratch across the launches
(each launch rewrites the leading `2·groups` values), and passes its `eps` as the parameter it
already is (the upscaler's 1e-5 and the encoder's 1e-6 are one op). The verification this §5.5
left open — "maybe zero-kernel, fold T into N" — holds: folding T into N is exactly what the
reference itself does (`view(b·t, c, 1, h, w)`).

The one thing that actually blocked the encoder's shapes was the device kernel's thread chunk:
at the encoder's first level, 128 channels in 32 groups is **4 channels a group**, and both GN
kernels give a thread an 8-channel chunk that must sit inside one group — so the launcher
refused the shape, while libref and libavx (which implement the spec's actual words, no %8
constraint anywhere in the schema or OPS.md) handle any channels a group. The device kernels
took the chunk width as a template parameter: a thread's chunk is 8 channels as before, or 4
when a group is exactly that wide (an 8-byte bf16 or 16-byte f32 access under the same row
alignment argument — pitches stay multiples of 8, chunks start on multiples of 4). Nothing
narrower ships or is answerable: 2 or 1 channels a group still refuse, as does the existing
12-channels test case. The `CS == 8` path is `if constexpr`-identical to the previous code, so
every shipped upscaler run is byte-neutral. Verification: `r4d_selftest`'s new
`128 in 32 groups` case (the encoder's level-0 shape, pitched bf16, SiLU) and
`64 in 16 groups, modulated` (f32 output, f32 gain and bias, bf16 modulation — every narrow
vector load runs) pass numerics, statistics, untouched-pitch and rerun-byte-identical checks
on-card; `avx_check` draws the encoder shape at `eps` 1e-6 against libref for both dtypes; the
three plugins' schemas are unchanged and still agree.

### 5.14 The audio posterior head is a composition: AttnProjection needs no new kernel (RESOLVED 2026-10-19)

The DAC encoder's `pre_block` (`audio_vae.py:253-275`, built at :403 as
`AttnProjection(latent_dim, 32, num_heads=8)`) is the last §5.5 item tagged "reduce", implying
a kernel. It is not: every piece is an existing op, and one contraction is *weights*, not code.
Reading the reference exactly — `x = proj(norm3(x)) + attn(norm1(x))`, then `+ mlp(norm2(x))`,
with `CausalAttention` = fused `qkv` Linear (`in → 3·in`, bias the concat of `q_bias`, a zero
`k` bias and `v_bias` — concat at convert time, no runtime cat), causal SDPA over 8 heads of
`in/8`, **mean over heads**, `adaptive_avg_pool1d(head_dim → 32)`, then the attention's own
`proj` Linear(32, 32):

- `layernorm` (×3), `gemm_nt_bias` (qkv with the fused bias; `proj`; `w2`), `attn_dense` with
  `causal` 1 and `scale` the default `head_dim^-0.5`, `add` (×2), and GeGluMlp exactly as
  shipped elsewhere: one `gemm` at `[M, 2·hidden]` from the concatenated `w0|w1` and `gelu_mul`
  (gelu-tanh gate · up — the same form the MoE experts use).
- **Mean over heads**: `attn_dense` takes q/k/v as `[tokens, heads, head_dim]` (OPS.md — the
  fused-projection column-slice form means the qkv gemm's output needs no copy), and its output
  keeps that layout, so the rows are token-major with the heads innermost and `pool_rows(k=8)`
  is literally "one mean row a window of k rows" = the head mean, in f32 as torch takes it.
- **The pool**: `adaptive_avg_pool1d` with an even split (`head_dim = latent_dim / 8` → 32) is
  a block-mean along the feature axis = multiplication by a constant block matrix of `1/block`
  entries — and because pooling and head-mean act on different axes they commute, and because
  both the pool and the attention `proj` are linear the two fold into ONE weight at convert
  time: `W' = W_proj @ M_pool`, bias unchanged. No kernel, one smaller gemm.

The head is therefore graph assembly for the encode-graph step, not a kernel bundle — the same
finding direction as §5.13, and it shrinks §5.5's device half to the one genuinely new kernel:
the `conv3d_ndhwc` extension.

### 5.15 `conv3d_ndhwc` learns the CausalConv3d forms (LANDED 2026-10-19, P2's last kernel delta)

The video encoder's every conv is `CausalConv3d` (vae.py:72-118): spatial **reflect** pad,
temporal **front-only zero** pad, optional per-axis stride and a one-sided (`(0,1,0,1)`) spatial
pad in `Downsample3D`. `conv3d_ndhwc` grew five optional params — `stride` (spatial), `stride_t`,
`causal`, `pad` (`same`|`down`, the `conv2d_nhwc` orientation), `reflect` — declared identically
in all three registries (schema triangle 0-disagree), documented in OPS.md's upscaler section.
The defaults are every pre-existing caller: stride 1, `same` zero pad, no reflect — the kernel
that measured 148 TFLOP/s for the upscaler keeps that exact code.

Contract (OPS.md, the row descriptions): pad first, then one strided pass. Output `(t, i, j)`
reads `x[(t·stride_t + dt − ft, i·stride + kh − pt, j·stride + kw − pl), ci]` over the PyTorch
floor-division lengths `((T + ft + bk − kt)/stride_t + 1)·((H + pt + pb − k)/stride + 1)·((W +
pl + pr − k)/stride + 1)`; `causal` 1 sets `ft = kt − 1, bk = 0` (so T=1 → T=1 at any stride —
the reference's `T == 1` tap truncation is *subsumed*, not special-cased); `down` sets
`pt = pl = 0, pb = pr = 1`; `reflect` mirrors each spatial axis once (torch `reflect`:
`x[−n] = x[n]`, `x[D + n] = x[D − 2 − n]`), which needs each pad below the axis it pads — libref
and avx check that and the output lengths on the host (geom is host memory there); the device
launcher takes the same parameter checks (strides ≥ 1, `causal`/`reflect` ∈ {0,1}, `pad`
`same`|`down`) but cannot see the device geom, so the mirror remap is followed by the ordinary
bounds test and a degenerate axis reads zero rather than running past the volume — every
engine-visible decision still agrees with libref on the tested shapes. All five are 3×3×3-only,
which the device's tap-constant kernel already forced for *any* launch.

Device: the implicit-GEMM kernel gains a `GEN` template flag selected by the launcher (any
non-default ⇒ GEN). GEN keeps the whole block/WMMAs/K-walk machinery — only the gather's base
changes: a staged row is a **grid of the output volume** (T2, H2, W2 computed in-kernel from
geom + params, floored division), holds its base source coordinate (`t·stride_t − ft`, …), and
`at_tap` mirrors spatially before the bounds test. The classic instantiation compiles from
`if constexpr`-elided source — same register allocation, same measured bandwidth — and both
paths share the C64/non-C64 (24-channel `conv_in`) K walks and the tap-major weight relayout.
`Mcap`/`xbytes`/grid logic is untouched; the grid may launch empty blocks past the shorter
stride-2 volume, which return at the raster check.

Tests: `r4d_selftest` conv3d grew a gen field set on `ConvCase` (the row model now carries the
output volume and the same pad/mirror math as libref) and seven device cases — res block,
Downsample3D at both strides, 24-channel `conv_in` (non-C64 walk), **one frame causal** (the T=1
rule), `down` at stride 1, temporal-only stride, and a pitched 128→136 with in-place residual —
plus five launcher refusals (`stride 0`, `stride_t −1`, `causal 2`, `reflect 3`, `pad "mirror"`);
filtered run all ok (small shapes rel_l2 **0**, i.e. bit-equal to the f64 row model). libref got
the hand-value test `conv3d_ndhwc_causal_forms_stride_mirror_and_pad_front_only` (an exact f32
ladder
120.5 / 250.5 / 1710.5 where frame t is `3·x[t,1,1] + x[t−1,0,0] + 130·x[t−2,0,0] + b` through
double-mirrored taps) and six refusals; avx_check_hand got the three drawn-never forms
(downsample, mirror-everywhere, one-frame with residual in place) and five refusal hands, avx
bit-tracking libref within the GEMM tolerance table. Full device selftest and both avx_check
suites green; ref_test green.

**What remains of §5.5**: no kernel deltas. The encode graphs (encoder levels + posterior head
per §5.13/§5.14), declare + re-convert of the 118+173 encoder tensors (`map_wnorm` folds the GN
affines), value parity against ComfyUI CPU encodes, and the driver encode pass (§6.1's FFmpeg
gate aside). The §5.13 recipe (GN one launch a frame from sliced operands) and §5.14's
convert-time folds are the graph's building blocks.

## 6. Benchmark protocol (dual-GPU question)

Claim to test: **one Radiance instance on 2 cards beats two single-GPU ComfyUI instances**
at equal work. For N clips: `N × T_rad_tp2 < T_comfy_1clip` (parallel singles) is the win
condition. **Measured T_comfy baselines (2026-10-18, single R9700, ref2va int8_convrot +
sol-attn tau 1.3, 25 steps):** 12 s / 294 f / 704×544 = **590.4 s** total (dense steps
26.6–27.1 s/it for the first 5, sparse steps 6–25 at **20.0 s/it**, ~56 s fixed non-sampling
overhead); 7 s clip = **315 s**.
**Correction (Hermes round-3, 2026-10-19 — supersedes the attribution above):** Hermes'
runtime logs (2026-10-17, 224/226 lanes, 768×576, 25 steps) identify the 590/315 rows as
**STOCK attention**, pre-sol — sol-attn landed after and is +11–16 %/GPU by the user's own
measurement, with no clean sol rows in the shared log yet. Benchmarking Radiance against
590/315 therefore overstates vs sol-attn lanes; and those wall times INCLUDE queue wait.
Hermes re-derives clean stock-vs-sol pairs (queue-time logged separately) and delivers them
before the P2b router-weighting merge — P6 must use those, keep 590/315 only as the
stock-attention floor, and re-check the 704×544 shape attribution against the raw logs. The sparse-node per-step bar (~25% cut vs dense INT8) is
also the performance target Radiance's VSA should beat, not match. Confounders to control: same seed is impossible cross-engine — compare wall
clock per clip at fixed width/height/length/steps, and note ref-token count dominates
(attention ≈ quadratic in sequence; refs ride **every** step). The user's interim test —
ComfyUI + sol-attn sparse node vs two back-to-back Radiance fp8 T2VA runs — measures
engine throughput only, not ref2va; treat as a baseline, not the claim. Also: do NOT
enable `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` on this box (causes blank-repeat
corruption on gfx1201 — verified elsewhere).

### 6.1 P1 measurements (2026-10-19, 16k container, `--tp 2`, 512×512, seed 7)

**kMaxText re-anchor 2048 → 16 384 VERIFIED end-to-end.** Server boots clean at the new
ceiling (host-pool note: 19.39 GiB placed; the +5.5 GiB/rank fits §5.3). `/v1/models`
reports `video_max_text: 16384` + `video_max_ref_{images,videos,audios}: 0` (model declares
no refs until the P1 plugin work lands — correct). A 7 701-token prompt is ACCEPTED and
queued (was a hard 400 naming the count at 2 048).

**Short-chunk test (kMinChunks 7→4, EXPERIMENT edit still in `minimaxh3.cpp:117`).** API +
geometry accept the sub-5.17 lattice exactly as predicted; soundtrack lengths are the
exact ADECODE readback counts (`n_samples` is deterministic at submit — the readback
consumes precisely that many; new RAD_INFO line in `video_driver.cpp` submit path):

| requested | lattice | frames | soundtrack | ≥2.01 s floor (Hermes) |
|---|---|---|---|---|
| 3.00 s | 4 chunks | 73 px = 3.04 s | **3.05 s** = 97 600 samp @ 32 kHz stereo | clears with margin |
| 3.50 s | 5 chunks | 90 px = 3.75 s | 3.75 s = 120 000 | ✓ |
| 4.00 s | 6 chunks | 107 px = 4.46 s | 4.45 s = 142 400 | ✓ |
| 5.17 s ctl | 8 chunks (125 px rounds UP past 7×17+5=124) | 141 px = 5.88 s | 5.88 s = 188 000 | ✓ |

So the engine's audio track is full-length at every beat — the ≥2.01 s encode floor never
binds down to 3.04 s, and 4-chunk (22 latents) = exactly the P2b tail length.

**BLOCKER FOUND — this machine cannot finish any clip.** The native build has
`RAD_WITH_FFMPEG=AUTO` silently OFF: `ldconfig` shows NO libav* runtime libs, no headers,
no ffmpeg CLI. Consequence, now traced in code: `Server::video_work` opens the muxer BEFORE
consuming events; when `mp4.open` fails it logs `encoding_failed` **at t≈0**, stores
`t.cancel`, and the driver retires the ticket — GPU generation never completes. (The P5
smoke note "generation ran to completion" was wrong — the instant encoding_failed masked
it.) End-to-end evidence for the 4-chunk clips, P2/P2b media decode (`core/mm/mp4.cpp`
needs libavformat to decode uploaded refs at all), and P6 wall-clock parity ALL require
FFmpeg. **User action:** `sudo apt install libavformat-dev libavcodec-dev libavutil-dev
libswscale-dev libswresample-dev`, reconfigure cmake (AUTO picks it up), rebuild; then
re-run the four beats above for full sampler+decode evidence before deciding kMinChunks
7→4 and BEFORE any timing claim.
One more environment-only fast-tier failure: `ops_doc` (`tools/ops_doc_check.sh`) uses gawk's
`gensub` and the box ships only mawk — installing `gawk` clears it; nothing on this branch
touches that script, and CI (gawk present) never sees it.

## 7. Cross-repo file map

Radiance: `core/server/videos.cpp` (API), `arch/minimaxh3/minimaxh3.cpp` (pack/modulation/
VAE/TP; decoders `:1226-1328,1344-1539`, `map_wnorm :1172`), `libr4d/r4d_vsa.hip`
(sparse attn; tiles cap 3072 `:1077`), `libr4d/r4d_video.hip` (`r4d_rad_conv3d_ndhwc`
— to extend for encode), `libr4d/r4d_gemm_mxfp4a8_nt_m64.hip`, `core/lora/`,
`data/recipes/*`, `docs/TOOLS.md` (rad-convert quant codes: MXFP4 =
`codes=fp4_e2m1 group=32 scale=e8m0`; NVFP4 = `group=16` fp8 scales), `docs/OPS.md`
(existing ops: `group_norm :1276`, `conv1d :1031`, `attn_dense causal :678`, `snake`,
`vtile_gather/stitch`), `docs/TP3.md`/`TP4.md`, `spec.md:523,552-553,635-637` (quant
grammar), `engine_bringup.cpp:920-960` (build-time `max_tok`).

ComfyUI (`/home/dustin/AI/ComfyUI`): `comfy_extras/nodes_minimax_h3.py:243-364` (the
port spec), `comfy/ldm/minimax/model.py:406-434,463,596-599,680-681,711` (pack/segments/
seg_tag/timesteps), `comfy/model_detection.py:415` (gate_compress detection),
`comfy_extras/nodes_sparse_attention.py` (sink rules, `minimax_h3_layout`,
`sol_attn_chunked`), `comfy/ldm/modules/attention.py:920-948` (backend dispatch).
Dasiwa pack (`custom_nodes/ComfyUI-DaSiWa-Nodes/nodes/`):
`nodes_minimax_h3_director_guide.py:134-139,190,246` (delegates to the core node — their
9/3/3 capacity IS the core node's; tail seeding goes through
`h3_continuity/core.py:178-217` (overlap snap table :61, layout dict
`{overlap_video_tokens, overlap_audio_tokens}` :199-200, prompt rewrite for the hidden
window :47-66,123-171) → `h3_continuity/nodes.py` ClipStore (`latent.safetensors` +
`clip.json`, :306-338) → the actual mechanism in the vendored
`h3_continuity/vendor/continuation_nodes.py` (window alloc :196-219, guide build
:529-535, append/trim :632-662, rejection rules :80-111,444-457);
`helper_minimax_h3_director.py:22-24,121-152` (the 2–15 s / 15 s / count policy).
ComfyUI glue: `comfy/model_base.py:2199-2228` (`minimax_keyframes`/`minimax_refs` →
`cond_*_latents` + `PackedLayout`, CONDConstant), `comfy/model_sampling.py:338-357`
(audio flow carry ×4). No sampler-side minimax hook exists (`comfy/samplers.py`: 0 hits).
Encode porting spec (§5.5): `comfy/ldm/minimax/vae.py:72-120,122-135,137-152,155-188,
176-238,406-474` (causal conv3d / isolated GroupNorm / downsample / ResnetBlock3D /
EncoderFCN3D + init), `:564-590,647-659,775-792` (tiled/temporal chunk/mean-only
encode), `comfy/ldm/minimax/audio_vae.py:155-213,253-269,374-443` (DAC encoder +
AttnProjection), `nodes_minimax_h3.py:75-83,310-311,344` (encode call sites).

---

## 8. Hermes round-trip outcomes (2026-10-19, contract CLOSED by round-3)

Reply round-2 in `docs/HERMES-COUNTERSPEC-round2.md` (git-excluded). Fleet design settled:
permanent dual-backend lanes (Comfy 222/single-GPU + native TP2 pairs), no cutover, no shim.
Round 3 (`docs/HERMES-REPLY-round3.md`): zero open blockers, design frozen; the document
pair IS the contract — any change to its bolded terms requires a round-note from whoever
initiates. Additions from round 3: (a) **P1 short-chunk test must also log the audio-track
length at 3.04 s** against Hermes' ≥ 2.01 s encode floor — their chaining policy needs that
number; (b) Hermes gate for lane production-eligibility, verbatim: `video_max_text ≥ 16384
AND video_max_ref_images ≥ 9 AND videos/audios ≥ 3`, else protocol-test lane only;
(c) `adapter_required: true` take-kinds stay Comfy-only in their router until task #12
lands; (d) corrected stock-vs-sol baseline pairs + queue-time-separated renders arrive from
Hermes before the P2b weighting merge (§6 correction).

- **Duration floor = model-card policy, not lattice law.** `kMinChunks=7`
  (`minimaxh3.cpp:112`, comment :109-111 "5 to 15 seconds: 7 to 20 chunks") — the official
  5–15 s range. Hermes' 3–5 s beats: 3.04 s = 4 chunks = 73 px = **22 latents** = exactly
  the P2b continuity-tail window (5·4+2), i.e. the stack will already run 22-latent
  sequences on the cond path. P1 experiment: render chunks 4/5/6 (3.04/3.75/4.46 s); clean
  → propose `kMinChunks 7→4` upstream citing DaSiWa production. Until measured: short
  beats route Comfy-only. Server note: `seconds` rounds UP to lattice (videos.cpp:17).
- **Capabilities block implemented** (P0 amend): `Server::video_caps()` in videos.cpp,
  called from `capabilities()` (admin.cpp) → `/v1/models` + `/server_info` now carry
  `video_max_text` + `video_max_ref_{images,videos,audios}` (model decl under API caps =
  what validate_refs applies). videos_test extended (echo + zero cases) — pass; the 2
  failures remain the pre-existing libavcodec ones. Hermes preflight gates on this.
- **LoRA-for-video GAP FOUND** (Q4): the video path applies NO adapters —
  video_driver builds bare `RadBatch{}` (lora_slot null), minimaxh3 plugin has zero lora
  references, `j->model` hardcodes base id (videos.cpp:511). And no strength knob exists
  anywhere: scale = adapter-config `lora_alpha/rank` (lora.h:82-105). DaSiWa production
  runs an adapter @0.5 on every H3 job → parity-relevant. New phase item (see task): wire
  lora_slot through the video batch (ABI per-row slot exists since RAD_ABI_VERSION 18;
  core issues adapter ops whenever the batch carries one, rad_runtime.h:53-63) + DiT
  target-module question; interim exact-0.5 recipe = copy with halved `lora_alpha`.
- **P2b shape fixed by Q1:** uploaded tail REQUIRED (keepers human-picked from ~12 takes,
  often Comfy-era/externally-edited); accept webm + mp4 (webm needs adding to the tail
  ingest exts; audio-only ext list already has ogg-family). job-id = later native→native
  optimization. Their "sidecar" was itself one muxed webm → Radiance muxed mp4 = parity.
- FP8 clarification owed to Hermes: our FP8 = .rad Radiance containers ONLY; no Comfy
  .safetensors FP8 artifacts exist from this work; their INT8→FP8 Comfy template swap is a
  separate Comfy-side conversion.

## cont.12–13 (2026-10-19) — encoder value-parity CLOSED on BOTH families; FL2VA keyframe path SHIPS. DaSiWa functional scope COMPLETE.

### cont.12 — #14 video/image value-parity RESOLVED (commit `8bbde56`, debug scaffolding `7dfc621`)
The video/image encoder ~0.49× under-gain was a **shortcut-AFTER-norm1 ORDERING bug in
`step_venc`**, not a structural spatial-trunk defect (the earlier "DOCUMENT-AND-STOP / needs a
golden box" conclusion was wrong — it *was* structural, but confined to one op-order). The
ResnetBlock3D shortcut conv `E.op_sc` was issued **after** `B.op_n1`, but on a shortcut level
`op_n1` normalises the input `in` **in place** (`nb = E.in`), so the 1×1 `nin_shortcut` residual
was fed `silu(norm1(x))` instead of the raw `x` the reference uses (`ResnetBlock3D.forward`:
`x = nin_shortcut(x)` before any norm). Only shortcut levels (cout≠cin ⇒ levels 1 & 3) were hit,
so the deficit appeared exactly at L1 and cascaded down-trunk. **Fix = issue `op_sc` before
`op_n1` in the `f0` branch** (`minimaxh3.cpp` ~:3737) — an op reorder, not a kernel change; fixes
the shared trunk so image fidelity too. **Cracked via frame-0 per-level bisection:** extended
`R4D_ENC_SEL=all` to dump f32 shadows of every level post-resnet (`v_lv[0..4].a`, decl fields
`dbg_ve_l0..l4`) + `ref_venc_parity.py --evcl0..--evcl4` captures the reference h after each level.
Device std ratio pre→post: L1 .455→1.000, L2 .466→1.000, L3 .380→1.000, L4 .331→1.000,
nout/q48/q2 ~.49→1.000 (all relFro 3–4e-3 = the bf16 floor). Trunk convs resolve to a HOST
kernel here, so the "bf16 implicit-GEMM" theory was moot. **Both encoder families (#14 audio +
video/image) now match the ComfyUI reference to the bf16 floor.**

### cont.13 — Task #8 endpoint keyframe completion: I2VA / L2VA / FL2VA (commit `98ecbeb`)
One keyframe path serves three DaSiWa modes: `first_frame` pins output frame 0, `last_frame`
pins the last frame, the gap is diffused. **A keyframe is pure attention conditioning** — its VAE
latent rides as a pinned (`update=0`) cond row at its frame's time position, faithful to
`comfy/ldm/minimax/model.py` where keyframe latents are `img_update==False` rows. **The sampler is
untouched; the step merely widens to carry the extra cond rows** (this replaces the earlier guess
of a "sampler pin-mask"). Server: `first_frame`/`last_frame` form parts → `VideoRefMedia::FIRST_FRAME`/
`LAST_FRAME`, `kf_task` validation (i2va/fl2va need first, l2va/fl2va need last), keyframes ⊥ refs,
`encoder_image_canvas(cover=LAST)`. Driver: keyframe pass issues one `RAD_ENC_IMAGE` Vencode per
anchor (`slot = ++ref_tab[1]`) + appends `{frame(0|frames-1), video_t=1, audio_t=0}` key triples.
Arch: `ref_parts` emits keyframe cond parts first; step scatter admits `kR2VCond`/`kR2VCondAudio`.
Host green (video_gen_test 553/17, videos_test 255/13, ref2va_pack_test 655/13). **Device-verified**
(TP2, seed 9999, content-consistent differential): FL2VA completes a valid 73-frame MP4+AAC; pins
are *consumed* (fl2va ≠ same-seed t2va) and *force the endpoints* (frame0 vs anchor 0.687 pinned vs
0.460 un-pinned; last 0.719 vs 0.521), visually confirmed. **Fidelity follow-up (not function):**
anchors ride as pins only; the reference also feeds them through the TE as `clip.tokenize` vision
tokens, so a prompt contradicting the anchor lets text win over the soft pin — full endpoint parity
vs a golden needs TE-anchor tokens, in the #10/P6 bucket.

### #22 "Image Inpaint" — investigated → NOT a mask path → DROPPED by user decision
The reference disproves the task's "separate denoise-mask path" premise. `nodes_minimax_h3_director_guide.py:166`
routes the `Image Inpaint` mode through the **same `MiniMaxH3ImageToVideo` node as I2VA**:
`native.execute(clip, vae, prompt, w, h, 5, first_frame=image, last_frame=None)`. The director
(`nodes_minimax_h3_director.py:232`) builds `first_frame`=scaled image, `last_frame=None`, hardcoded
`length=5`; one image at slot 0, video/audio refs rejected. `nodes_minimax_h3_tiled_upscale.py:299`
says it outright: *"Director Image Inpaint is a native five-frame I2V pass… extract the still after
decode."* No mask tensor, no `model_base.set_inpaint`/concat, no `inpaint_post_norm` in the H3 DiT —
crop→mask→I2V→composite→still is entirely client-side `nodes_inpaint.py`. So at the engine level it
**is the shipped I2VA first_frame keyframe path** (frame 0 soft-pinned at ~0.69 SSIM ⇒ the region is
regenerable, which is what an edit needs); the only engine-side delta is `length=5` (2 latents), which
the chunk lattice (`frames = frames_per_chunk·chunks + frames_tail`, `chunks ≥ min_chunks`) cannot
express without dropping `kMinChunks` below the model-card floor — an upstream proposal, not unilateral.
**USER DECISION 2026-10-19: DROP #22** — "never used it… other image models serve inpainting better."
Out of scope by product choice, not a code gap. **Do not implement a mask path or break the length floor
for this.** Task removed from the list.

### cont.18 — Sampler: `res_multistep` (DPM-Solver++2M data-prediction) as a per-request field
The DaSiWa reference workflow samples with `res_multistep`+`simple`, not Euler. Radiance only had
first-order Euler (`flow_x0_step`: `d=x+σ·v`, `x'=(σ'/σ)·x+(1-σ'/σ)·d`). Added a second-order step on
the clean estimate, chosen **closed-form** so it reuses the same per-step scalars the driver already
stages: `x' = r·x + hb1·den + hb2·x0` with `r=σ'/σ`, and on a 2nd-order step `h=ln(σ/σ')`,
`h_last=ln(σ_prev/σ)`, `r_s=h_last/h`, `hb1=(1-r)(1+1/(2·r_s))`, `hb2=-(1-r)/(2·r_s)`. Identity
`hb1+hb2 = 1-r` ⇒ collapses to Euler when `x0==den`. Order 1 (Euler) forced at `i==0`, the first
full-size step after a latent upscale (`i==j.split`, stale history), and the last step (`σ'==0`);
order 2 otherwise. `old_denoised` persists in a new PERSIST buffer (`b_vx0`/`b_ax0`), written each step.
**Proven equal to k-diffusion `sample_dpmpp_2m`** to 5.6e-17 on three sigma grids (pure-python check;
`-(-h).expm1()==1-r`, `sigma_fn(t_next)/sigma_fn(t)==r`, and the 2M `den_d` coefficients match exactly).

**New op `flow_res_multistep`** across all three kernel tiers (must match): `libref` (`ref_flow_res_multistep`,
left-assoc `((r·x+hb1·den)+hb2·x0)`), `libavx` (`AVX_KERNEL`, scratch `5n+64`), `libr4d` (`flow_rm_kernel`,
f32|bf16 velocity, f32 x/x0/scalars). Operands: `v(in) x0(INOUT) σ r hb1 hb2 order x(INOUT)` — multi-INOUT
(x0 read+write, x write) supported by issue.cpp. **AVX fix**: first cut grouped `r·x+(hb1·den+hb2·x0)`
(right-assoc) and failed the tol-0 hand check on the order-2 case only — ref/device are left-assoc;
both the vector loop and scalar tail regrouped left → 451 hand-built checks pass, 0 fail. `rad-avx-check`
816 checked / 0 fail. `r4d_selftest --case flow_res_multistep_f32`: f32|bf16 × order {1,2}, both INOUT
exact with pitch gaps untouched.

**Threading** (ABI 23→24, trailing `int32_t sampler` on `RadVideoGenBatch`): `RAD_SAMPLER_EULER=0`
(default) / `RAD_SAMPLER_RES_MULTISTEP=1` in `rad_runtime.h`. Driver `R_SCHED` region widened 6→16 f32:
`[0..5]` Euler scalars (unchanged), `[6..10]` video coeffs `(σ,r,hb1,hb2,order)`, `[11..15]` audio —
staged in `stage_sched`. `step_dit` issues `op_rmv`/`op_rma` (slice `b_vx0`/`b_ax0`) when
`V->sampler==RES_MULTISTEP`, else the original `op_ev`/`op_ea`. Server `POST /v1/videos` gains a
`sampler` string field (`"euler"`/empty → Euler, `"res_multistep"` → RM, else `invalid_request_error`
with `supported=["euler","res_multistep"]`); echoed on the job object; `capabilities` advertise
`video_samplers`. `VideoRequest.sampler` + `Server::VideoJob.sampler`.

**Device-verified** (TP2, `minimax-h3-ref2va-fp8.rad --vision-tower-placement ram --max-model-len 4096`,
256²/2s/10-step, **seed 1234 identical**): both samplers complete clean (73-frame h264); `sampler` echoed
and advertised; outputs **DIFFER** (md5, size) ⇒ branch genuinely consumed, not silently ignored; a
bad value rejected, field omitted ⇒ euler. Visual: same fox-in-snow composition/pose; res_multistep
renders warmer+higher-contrast (gold snow, deeper shadows, clip_hi 2.3%, mean 192) vs euler cooler/flatter
(clip_hi 0%, mean 184) — coherent 2nd-order trajectory, not degenerate. Side observation for **#45
(brightness)**: at this toy setting res_multistep is *brighter and clips highlights* where euler doesn't —
directionally the opposite of a brightness fix, so res_multistep is a tone lever, NOT the over-exposure cure.
Euler default path byte-for-byte unchanged.

### cont.19 — Example-set diagnosis: the gibberish audio was the PROMPT, not the sampler (no code change)
User eval of the `examples_rm` res_multistep set: (a) **no visible/quality difference vs euler** at same seed
(MAD 12–39, same content/pose), same wall-time ⇒ res_multistep is a *tone* lever only. **Decision: KEEP the
commit, default stays euler** (free, off by default, DaSiWa-doc-recommended; reverting buys nothing). (b)
The jeans/snowmobile clips "look weird and off" and the **audio is gibberish** — root cause is the **prompt
format**, not the sampler and not a code regression (the euler set used the identical hand-written prose
prompt, hence "same issue as before"). The example prompts were freeform prose stuffing SFX into sentences;
the DaSiWa pipeline needs the **structured prompt**: `subject_definitions`/`summary`/`retention_analysis`/
`detailed_description` with `<Picture N>`/`<Subject N>` markup, a `<d>[English] …</d>` **dialogue** tag, an
`overall_soundscape:` block, and a `non_diegetic_music:` block. Standing rule "pass the prompt exactly as in
the example" exists *because* of this. Re-rendered the birdbike FL2VA example (`radiance-fl2va-examples/
…birdbike_FL2VA-2.json` node 2841 widget[1] = the exact 6135-byte prompt; keyframes `first/last-bird-bike-
square.png`) verbatim, 512²/12s/25-step/seed1234, euler+res. Audio signature moved the right way: voice-band
energy 64%→79%, high-band hiss 16–18%→9%. Visual matches the `<Subject>` defs (helmet/goggles/beak/teal top/
khaki shorts/red hardtail), flat cartoon style, bright (the #45 over-exposure still present). euler≈res at
same seed. Deliverables in `build/tmp/ex10/birdbike/`. **Lesson for any future eval: only render with the
exact received structured prompt — prose paraphrase garbles the audio.**
Open threads the user raised here, not sampler: **input-resolution hypothesis** — the cartoon finetune forgives
the low-res keyframe (1389²→512) so it only shows brightness, while STOCK photoreal (the jeans set) *exposes*
distorted faces/hands ⇒ this is #37 (style), and may be partly low-res input; user is supplying a higher-def
REAL first/last example to separate input-quality from generator style.

### cont.20 — The "ghosting regression" was the MODEL CONTAINER (stock-fp8 soft), NOT the sampler (no code change)
User reported the birdbike clips were "a clear regression from the look folder… I could see the tiles… not as
crisp," refined to **ghosting around the beak and throughout**, visible even in a still. Investigation, in order:
1. **Sampler commit EXONERATED, provably.** My fresh new-binary `euler` 26-step fp8 webm decode is **pixel-
   identical (face-crop MAD 0.00, max 0)** to the 17:23 look clip `stock_12s_25.webm` — the same request on the
   same seed. `res_multistep` commit `9d67df3` left the Euler path byte-for-byte unchanged (the differing md5 is
   mux/audio bytes, not the video frames). So the sampler code is NOT the regression.
2. **Container is the whole story.** The user's crisp reference (right VLC window) is `w8a8_12s_25.webm`
   (container `minimax-h3-ref2va-uncensored-w8a8.rad`); the soft/ghosty clips were the **stock** fp8 container
   `minimax-h3-ref2va-fp8.rad` — which is exactly what I had served the birdbike + jeans/snowmobile examples on.
   `stock` and `uncensored` are BOTH fp8/26-grid/seed1234/same-keyframes ⇒ the variable is the **checkpoint, not
   the quant**: the stock checkpoint renders soft/muddy edges with dark halos (the "ghost"); the uncensored
   finetune is crisp. (This is the same fp8-stock-artifact family as #36/#37.) Proof: my fresh **w8a8 euler**
   render is **pixel-identical (MAD 0.00)** to the user's `w8a8_12s_25.webm` — container reproduced exactly.
   Face-crop Laplacian variance: **stock-fp8 4234 (soft) → w8a8-euler 5237 (= ref) → w8a8-res 5629 (crispest)**.
3. **Fix = serve/render eval examples on `uncensored-w8a8`**, not stock fp8. w8a8 (int8) is also the closest
   Radiance container to the DaSiWa reference (`dasiwaHybridV4_int8`, see [[dasiwa-reference-ground-truth]]).
   Re-rendered birdbike FL2VA on w8a8 with `res_multistep` 26-grid: sharp beak edge, crisp eye-rings, clean
   helmet vents — ghost gone, face MAD vs old fp8 = 44.9. Deliverables `build/tmp/ex10/birdbike/
   birdbike_w8a8_{res_multistep,euler}.webm` + `fix_fp8_vs_w8a8.png`. **Standing eval rule: render on
   `uncensored-w8a8`, never stock fp8, and only with the exact structured prompt** (cont.19).
Still open (unchanged by this): #45 brightness, #37 style, and the user's pending higher-def REAL first/last
example (#3) — re-render the other 4 categories on w8a8 too, but each still needs its own structured DaSiWa
prompt or the audio gibbers (cont.19).

### Status after this increment
DaSiWa **functional scope COMPLETE**: T2VA / I2VA / L2VA / FL2VA / REF2VA all device-verified;
continuity chaining drivable via `ref_video` slot 0; both VAE encoders value-parity to the bf16
floor; `res_multistep` sampler added + device-verified + **KEPT (default euler)**; #22 Image Inpaint dropped
by user decision (see above). All branch commits are placeholder-author `radiance-port-pin`, on
`ref2va-merge`, **NOT pushed** (repo has no identity; user re-authors before any push — upstream Codeberg).

### cont.21 (2026-10-20/21) — Curve-form checkpoints (`adaln_t_table`) SHIP in minimaxh3 (#60)
The DaSiWa H3 checkpoints store the timestep modulation as a **precomputed curve**: `adaln_t_table`
F32 [1025,8] (one 8-value row per quantised timestep) and ada weights BF16 [96768,8] / bias [96768]
(96768 = 3 modalities × 6·D/world; rank column-half = 16128), instead of the full-form
[96768,2688] sinusoidal MLP. minimaxh3 now detects the table in the container and switches the
timestep path: `timestep_embed` **lerps table rows** instead of running the MLP; the runtime `b_temb`
becomes BF16 {4·S, 8} (4 blocks = video | audio | video-pin | audio-pin, S = step capacity, each
step writes row `step` of its block); the ada matvec is [96768,8]·[8]. Modulation-row convention
unchanged: table row = t_rank*3 + tag, plain cells [text=t0|tag1 → 1, audio=t1|tag2 → 5, video=t0|tag0 → 0].
`norm_out.linear` is likewise [10752,8].
**Device-verified** on `minimax-h3-dasiwa-w8a8.rad` (TP2 serve, `--vision-tower-placement ram
--max-model-len 4096`): full max-shape request + the eval arms complete clean; a T2VA rerun is
**bit-identical** across serves (determinism intact). Commit `e28cf86` (amended from `24bd297`),
pinned author, NOT pushed.

### cont.22 — The fried look SURVIVES the curve container on every arm (#45/#59 measurement)
Same-seed (1234) renders on the curve dasiwa-w8a8 container, luma mean / clipHi (V≥250):
**T2VA 158.7 / 45.1%, I2VA 119.1 / 28.0%, FL2VA 119.9 / 28.2%** — vs the reference bar (4 DaSiWa
clips) **84–113 / 0–4.4%**. The "finetune just renders punchier" theory (cont.20-era) is **DEAD**:
the T2VA control, which touches no image conditioning, is the WORST arm. The fork is Radiance's
**DiT forward value-parity** (the denoiser emits an over-variance latent), consistent with
[[radiance-fried-look-decode-quant]]'s elimination table. ⇒ build the dump+replay diff (#61/#62).

### cont.23 — `R4D_DIT_SEL`: step-0 DiT dump for CPU-reference replay (#61, commit `e28cf86`)
`R4D_DIT_SEL=<step>` (only 0 implemented) makes minimaxh3 declare f32 **PERSIST** shadows and the
driver read them back on a **plain** request's step 0 into `R4D_ENC_DUMPDIR`:
`dit_h0.f32` (stream entering block 0; shadow capped at **8192 rows** — an uncapped max_tok-sized
shadow ate the arena and starved `vram_kv` to 0, the first launch's abort), `dit_temb.f32` (curve
form only; block pitch = declared step capacity, NOT the request's step count — prep_dump.py slices
it accordingly), `dit_mod.f32` (the full gathered modulation, cross-check only), `dit_vel.f32`
(same-pass velocity head readback — same-pass scratch reads are safe, cross-pass ones are not), plus
driver-side `dit_pos.f32`/`dit_seg_cu.i32`/`dit_info.txt`. Step 0's h0 row count > cap ⇒ clean
step-fail, never a silent truncation. NOTE: `rad_note` lands in the **graph report's notes list**,
not the server log — its absence from the log means nothing (verified by `strings` on the .so).
`build/tmp/look/serve_dasiwa_dump.sh` runs the dump serve. **Verified 2026-10-21**: a 256²/3s/8-step
T2VA request (seed 1234) dumped all seven files (rows 3212 = 1560 text + 244 audio + 1408 video;
driver log `E dit_dbg: dumped step 0 -- rows 3212 h0 176160768 temb 8192 mod 51609600 vel 39481344`).
#62 = `prep_dump.py` → `run_reference.py --td 8 --temb-presilu` (CPU, ~3 min/block) → `compare_vel.py`;
harness gains `--td` (8 = curve). Verdict appended when the replay lands.
