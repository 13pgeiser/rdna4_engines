# Ref2VA / DaSiWa port note — parity, and what this port does not reproduce

DaSiWa is the reference ComfyUI workflow for MiniMax-H3 **Ref2VA**: generate a video *with* a
soundtrack conditioned on reference media — up to **nine images, three videos (each optionally with
its own soundtrack), and three standalone audio clips** — plus **continuity**: feeding a previous
clip's tail back in so a longer generation continues from where it left off. This note states where
the Radiance port (`/v1/videos`, `task=ref2va`, dual gfx1201 / TP2) matches that workflow and, more
importantly, **where it does not** — including the reference-count ceiling that once capped this
build at five images and how it now reaches nine.

Everything below reflects the state on branch `ref2va-merge`. The design derivations live in the
sibling notes (`NOTE-P3-PACK.md`, `NOTE-P2-ENCODE.md`); this one is about capability boundaries.

## What works today

- **Image references — ≤9 photos, device-verified.** A request carrying `ref_image_1 … ref_image_9`
  runs the full conditioned denoise on both cards and returns a video. The conditioning is
  demonstrably *live* and *content-sensitive*, not merely count-sensitive: at a fixed seed, two jobs
  with the **same prompt and the same single reference but different photo content** give distinct
  outputs (a strict content-control), and nine photos complete where the pinned build refused at
  submit. Images steer the output because they feed the text encoder as well as the VAE — the route
  the video / audio references lack (see point 4).
- **Reference-media plumbing and caps.** The container declares 9/3/3 (images/videos/audios); the
  VAE-encode passes, the reference table, and the widened step layout are all in place, and the
  encode path is now unblocked by the host FFmpeg install.
- **Video and audio references — device-verified, live, content-sensitive, deterministic.** The submit
  refusal that used to reject `ref_video_N` / `ref_video_audio_N` / `ref_audio_N` is gone. The server
  decodes each into the encoder's form, the driver stages a `Vencode` / `Aencode` pass and folds it into
  the reference table, and the widened step consumes it (point 4). Proven end-to-end: the host tier is
  green (`videos_test` decodes and speaks every kind; `video_gen_test` clips-pads / zero-pads, slots a
  soundtrack to its video, lays the table, and refuses a non-encoding container by name); on the device
  the complete DaSiWa shape — a picture + a video + that video's soundtrack + a standalone audio, four
  reference parts in one request — runs every encoder pass and the widened denoise to a clean MP4 with no
  engine fault. **And they steer the output:** a warm-server fixed-seed content-control (same prompt,
  same reference *count*, only the reference *content* differing) gives **different decoded streams** for
  a different reference video and for a different reference audio, and a *same*-reference rerun is
  **bit-identical** (deterministic). The earlier "video / audio do not change the output" reading was a
  measurement error — raw-MP4-byte comparison plus a cold-start pair (see point 4); the encoders are
  content-live, not collapsed.
- **Audio-encoder numeric value-parity — measured and now matched.** Against a device-input-anchored
  fp32 reference, the audio latent used to sit at ~0.25–0.39× the reference spread. The cause was a
  **pre_block `norm1`/`norm3` swap** (the projection branch read the wrong LayerNorm — see the
  video-encoder parity section below). After
  the fix the pre_block output and the final latent both track the reference (ratio ~0.97 / ~0.93, R² 0.98–
  0.99, no remaining systematic gain); the ~0.10 residual-after-affine is bounded by low-precision
  execution (a full-bf16 reference itself moves 55% from its fp32). **The video / image encoder is a
  separate, still-open divergence** — it has no pre_block, so this fix does not cover it.
- **Continuity chaining — drivable today through the reference paths.** DaSiWa's scene-to-scene lead-in
  is the previous clip's tail fed back as a `ref_video_N` (+ `ref_video_audio_N`), not a special engine
  mode. Two requests reproduce it: generate clip A, cut its tail, request clip B with `ref_video_1` = the
  tail. A fixed-seed control confirms the tail conditions B (with-tail vs without-tail differ). The
  pack's exact keyframe-pin rows (`kR2VCond`) are a **distinct mechanism, and they have since
  shipped**: I2VA/L2VA/FL2VA arrive as `first_frame`/`last_frame` parts whose pins are
  device-verified (see the keyframe-pin section below), separate from this chaining path.
- **The modulation-cell merge** that lifts the count ceiling (next section).

## The reference-count ceiling: how it went from ≤5 to ≤9

The denoiser's **modulated normalization** reads a *modulation table*: one small vector pair per
"cell", indexed by which segment a row belongs to. The old build sized that table **per reference
segment**. On a dual-32 GB box the table is ≈219 MiB per cell against a 182 MiB KV allowance, so six
to nine photos simply had no headroom and were rejected at submit.

The table does not actually need a cell per reference. **A cell is a pure function of a segment's
kind**: which modality's ada projection it uses (`video` / `text` / `audio`) and which timestep block
it rides (`video-schedule`, `audio-schedule`, `video-conditioned`, `audio-conditioned`). Every
reference photo is `(video-ada, video-conditioned)` — the *same* cell. So:

- **Runs of consecutive same-cell segments collapse to one group.** Group boundaries are a strict
  *subset* of the fine segment boundaries, and because the merge key is the kind alone, the merge is
  **independent of sequence length**.
- The modulated-norm kernel takes its boundaries as an operand and only uses them to index the mod
  row and gate — it has no size specialization. So it can be handed the **coarse group boundaries**
  while attention keeps the **fine segment boundaries**, and the table is pitched by the **group
  count**. Nine photos then occupy **four groups**: *prompt · all-photos · soundtrack · video* — a
  twelve-segment fine layout riding a four-cell table. Buffer sizes (and the arena) are unchanged.

**No regression for ≤5:** before the merge each photo filled its own table column from the *same*
input rows and *same* weights, so every photo cell was already bit-identical to every other; merging
issues that computation once instead of N times and yields the identical per-row value. A five-photo
request is byte-identical before and after the merge — only the previously-rejected sixth through
ninth photos change behavior, by being allowed.

In short: **for photos the modulation table is no longer the limit.** Nine photos are four groups no
matter how many there are.

## Throughput (self-contained — no reference needed)

Wall-clock per job on this dual-gfx1201 / TP2 box, measured as the server's own `completed_at −
created_at` on an **empty queue** (so the figure is the job, not a queue wait). Fixed seed; 25 steps
unless noted. The plain rows are unconditioned; the "full shape" rows are the complete DaSiWa request
— four `ref_image_N` + a `ref_video_1` + its `ref_video_audio_1` + a standalone `ref_audio_1`.

| shape | steps | wall |
|---|---|---|
| plain 256×256, 3 s | 25 | 11 s |
| plain 256×256, 3 s | 50 | 21 s |
| plain 512×512, 3 s | 25 | 38 s |
| plain 256×256, 6 s | 25 | 23 s |
| full-shape ref2va 256×256, 3 s | 25 | 93 s |
| full-shape ref2va 512×512, 3 s | 25 | 121 s |

What they show, and what they deliberately do not:

- **Steps scale ~linearly** — 25 → 50 doubles the wall (11 s → 21 s), as a per-step DiT should.
- **Pixels scale sub-linearly** — 4× the pixels (256² → 512²) costs ~3.5×, not 4×.
- **Duration scales ~linearly** — 3 s → 6 s roughly doubles the target rows and the wall.
- **Conditioning is the dominant cost, and it is the *sequence length*, not the encoders.** The full
  reference set takes 256×256 from 11 s to 93 s (~8×) and 512×512 from 38 s to 121 s (~3×). The extra
  time is the widened denoise — each step now attends over *text + every reference row + the targets*
  instead of the targets alone — so the overhead shrinks in *relative* terms as the target grows (the
  reference rows are a fixed addend on a longer sequence). The VAE encodes are a one-time pre-pass,
  not per-step, which is why the multiplier falls with resolution.

These are **self-contained** numbers — they characterise *this* build's throughput and scaling and are
reproducible here (driver: `build/tmp/bench.sh`). They are **not** a parity figure: a ComfyUI-reference
throughput comparison and the per-tensor *value* parity both need the reference stack and are tracked
as point 6 / point 4. Nothing here claims to match ComfyUI's timings.

## What this port still cannot reproduce (the honest list)

1. **Many *interleaved* mixed-modality references can exceed the 8-group table.** Photos are free
   (each `(video-ada, video-conditioned)`, one merged cell), but every reference *video* contributes
   two cells in order — *video-conditioned* then its *audio-conditioned* soundtrack — and a photo
   placed after it **reopens** the video-conditioned cell, adding a group. Alternating photos with
   video-plus-soundtrack refs can push the merged run past `kMaxSegs = 8` cells, which the driver
   refuses at submit: *"these references change the conditioning's modulation cell N times, but this
   deployment's modulation table holds 8 cells — ask for references that share their modalities and
   target, or fewer of them."* Since reference videos are now wired (point 4) this ceiling **binds for
   real**, not just prospectively. **Workaround:** group references by modality (all photos together,
   then the videos) so same-kind refs are contiguous and merge; the ceiling only bites on heavy
   interleaving. A single video-plus-soundtrack is four to five groups and always fits.

   *Why the ceiling stays at 8:* widening it means growing the shared `b_ada` / `b_nout` / `b_modg`
   buffers on **both** cards — exactly the VRAM a dual-32 GB box does not have. Since the common case
   (photos) already fits, this is a deliberate budget boundary, not an oversight.

2. **The reference-token / row budget is the real binding limit at high resolution.** A conditioned
   step lays out *text + all reference rows + target rows* as one fixed-length sequence, and that
   must fit the server's `--max-model-len`. Nine large photos fit the four-group modulation table
   yet can still exceed the row budget and are refused: *"a conditioned step is N rows and a pass
   carries M; ask for a shorter or smaller video, or fewer references."* This is inherent to a
   fixed-length DiT sequence — the reference ComfyUI workflow pays the same cost; the modulation
   merge removes the *table-per-reference* growth, not the *per-reference token* cost. Raise
   `--max-model-len`, or use smaller / fewer references.

3. **Continuity chaining works through the reference paths; the keyframe-pin mechanism ships
   separately.** The DaSiWa "3-second lead-in" is *not* a special engine mode: per the
   source-verified brief (`docs/NOTE-CONTINUITY.md` §1) the director feeds the previous keeper clip's
   last ~3 s back as a **video + audio reference** (`media_mode: video_audio`, `slot 0`, `start: 0`) —
   *conditioning media, not content*; the next scene simply continues from that tail and never replays
   it. That is exactly the `ref_video_N` (+ `ref_video_audio_N`) path this port drives, so chaining is
   available **today** with two requests: generate clip A, cut its tail, request clip B with
   `ref_video_1` = the tail. A controlled same-seed check confirms the tail genuinely conditions B —
   with-tail and without-tail give different output at an identical seed and prompt. The reference
   engine additionally regenerates a leading overlap window as target rows and then trims it; here the
   reference rows are fixed conditioning held outside the sampler and every output frame is new (no
   overlap to trim), so the net "continues from the tail" semantics match but the seam may differ
   subtly. The pack's *keyframe* rows (`kR2VCond` / `kR2VCondAudio`) are a **distinct exact-pin
   mechanism, and it ships**: I2VA/L2VA/FL2VA reach start/end-frame conditioning through the
   `first_frame`/`last_frame` parts, not through reference images (see the keyframe-pin section
   below for the shipped mechanism).
   **The pack layer was reference-verified first**: `ref2va_build` emits `kR2VCond` / `kR2VCondAudio`
   segments, and the golden `fl2va` (two keyframes) and `chain` (one keyframe) cases assert their
   segment order, sizes, positions, `img_pos`/`audio_pos` and modulation rows **bit-exact** against the
   reference (they re-baked byte-identical here). The engine pieces the first pass left open — the
   driver populating a keyframe row, `step_dit` scattering the cond rows — landed with that section;
   a prior clip's tail needs no bespoke persistence since it rides in as a `ref_video_N`. Seam
   frame-equality is the wrong metric to gate on: the brief states it is imperfect **by
   construction**, because the first new frame is the tail's temporal *successor*, not a replay of
   its last frame.

4. **Reference videos and audios steer the output — through the VAEs, not the text encoder.** A
   `ref_video_N` (mp4/mov), its soundtrack `ref_video_audio_N`, and a standalone `ref_audio_N`
   (wav/mp3/…) are decoded on the server into the encoder's own form (`encoder_video` /
   `encoder_audio` → `enc_pixels` / `enc_samples`), staged into a `Vencode` / `Aencode` pass, folded
   into the pack's reference table, and consumed by the widened step — the same route an image
   reference takes, only via a VAE instead of the language model. A video and its separately-given
   soundtrack are **one** table row: the soundtrack folds into the video's `audio` word and encodes
   at the video's slot (`aenc_slot_row` numbers it by its video). The driver clip-pads a video's short
   final clip by repeating its last frame and zero-pads an audio's tail to a whole hop, because only
   the driver holds the encoder geometry. A request of videos / audios alone carries no `media`: only
   images add soft-token rows the text encoder reads. A container converted **without** its VAE
   encoders (`encodes 0`) refuses such a part *by name* ("… is a reference video, but this container
   was converted without its VAE encoders") rather than drop it into a text-only video.

   **Video and audio references now steer the output.** The earlier "encoder collapse" reading was a
   measurement error and has been corrected: on a *warm* server, a fixed-seed content-control (same
   prompt, same reference count, only the media differing) gives **different** output for a different
   reference **video** and for a different reference **audio**, and a rerun of the *same* reference is
   **bit-identical** — so both VAE encoders are deterministic and content-sensitive on device, not
   collapsed. A reference video (two 256×256 clips, testsrc vs mandelbrot) and a reference audio (440 Hz
   tone vs white noise vs 880 Hz) each change both the output video and audio stream; the image path
   steers as before. (The superseded "byte-identical output" claim came from comparing **raw MP4 bytes**,
   which the muxer stamps, and from a pair that straddled the one-off cold-start request — neither is a
   valid comparator; compare *decoded* streams on a warm server.) An env-gated device dump
   (`R4D_ENC_DEBUG`, reads `b_evc` / `b_enca` after the encode pass) independently confirms the latents
   themselves differ per media and reproduce byte-for-byte on a rerun. The scatter bounds were audited
   and read exactly the slot region the encode pass wrote this job — no stale-neighbor read.

   What remains is **strict numeric value-parity** of the encoder latents against the reference HF
   encoder (a per-tensor golden), not *whether* the encoders are live. **This has now been measured on
   the audio path.** *(The pre-fix numbers this and the next paragraph record — the ~0.37× under-gain —
   have since been **root-caused to a pre_block `norm1`/`norm3` swap and fixed**; see the RESOLVED block
   below. They are kept as the measured history that led there.)* A device-input-anchored check — feed the
   reference HF audio VAE (fp32) the device's
   *own* staged `enc_samples`, then compare its latent to the device's `b_enca`; the reference and the
   container use byte-identical encoder weights and `latents_mean/std`, verified, so only the kernels
   can differ — gives a **relative Frobenius error ~0.79, correlation 0.67**, with the device latent's
   std ~2.5× smaller than the reference's (0.23 vs 0.59). That is far beyond the fp8 floor: fp8
   quantisation is unbiased (it neither collapses std nor anti-correlates channels) and the convert log's
   own audio-quant error is ~1.6e-3. The causes are individually **ruled out** — precision (fp8 would
   keep corr ~0.99), preprocessing (device-input-anchored), reference export (the 97 shared encoder
   tensors are bit-identical to the HF source, and the `latents_mean/std` match to the last digit), and
   normalisation constants (identical). (An earlier reading of this path — "shape error survives scale
   removal, so *not* a single scale" — is **superseded by the affine decomposition in the last paragraph
   below**: fitting one global `a·ref + b` to the two sorted distributions collapses most of this error, so
   the dominant term *is* a single under-gain, and the corr / per-channel anti-correlation lives only in the
   small post-affine residual.) Localising it fully needs
   stage-by-stage activation dumps of the device encoder against the reference module-by-module —
   substantial work on kernels the live image / video paths also share, and deliberately **not**
   blind-fixed. The audio result is measured; the **video / image VAE encoders are a separate kernel
   family** (`conv3d_ndhwc`) and have not yet been measured this way. Functional conditioning is
   unaffected — the DiT is internally consistent with the latents it is fed — but *numerical* fidelity to
   the reference encoder is not achieved on the audio path, so audio-reference output is not guaranteed to
   match ComfyUI latent-for-latent. *(This pre-fix claim is now **superseded**: the audio under-gain was
   root-caused to a `norm1`/`norm3` swap and fixed, and the audio latent is now a near-identity map of the
   reference — see the RESOLVED block below.)* Reproduce: dump with `R4D_ENC_DEBUG=1 R4D_ENC_DUMPDIR=<dir>` over one
   `ref_audio` request, then `ref_aenc_parity.py --enca enca.f32 --encsm encsm_4_1.f32 --kind audio
   --slot 1` (its `--selftest` validates the reshape / slot / metric logic with no device). The audio
   encoder is the cleanest first measurement because it has no spatial preprocessing to confound it.

   The **video / image VAE encoder has now been measured the same way and diverges too** — with the
   *same signature*. A single reference picture is the cleanest video case (the `encode()` T==1 branch,
   one clip, exactly one latent frame, so no temporal keep-first/last ambiguity). The device stores
   `b_evc` as the *patch-folded* normalised latent (24 ch × a 2×2 patch = 96-wide rows), so the harness
   compares the two **sorted** value vectors — the fold is a pure bijection and the latent norm is
   per-channel, so a faithful encoder has an identical multiset regardless of fold order (this sidesteps
   re-deriving the fold's layout, the mistake that once cost a bogus audio reading). Result on a 256×256
   gradient picture: **sorted relative Frobenius 0.65, device latent std 0.44 vs reference 1.15** (a
   ~2.6× under-spread), max-abs deviation 4.26. Note the ratio: the audio path collapsed std 0.59→0.23
   (~2.6×) and the video path 1.15→0.44 (~2.6×) — the *same* qualitative signature across two entirely
   different kernel families (strided conv1d vs `conv3d_ndhwc`). **And for BOTH families the latent
   normalisation constants are byte-identical to the reference** (`kVaeLatentsMean/Std` vs `LATENTS_MEAN/
   STD` max rel-diff 0.0; `kAudioLatents*` likewise verified earlier), so the collapse is *not* the
   normalisation constants. That refines the earlier "suspect the strided conv1d temporal reduction"
   note: the divergence is not audio-specific, so the leading hypothesis is now the encoder *network*
   forward path shared in shape by both (a per-level norm / activation scale or a weight-quantisation
   attenuation), not a single family's conv reduction. Localising it needs stage-by-stage activation
   dumps of a device encode against the reference module-by-module — substantial work on kernels the
   live image / plain paths also run through, and deliberately **not** blind-fixed. Reproduce: dump with
   `R4D_ENC_DEBUG=1 R4D_ENC_DUMPDIR=<dir>` over one single-image `ref2va` request (emits `encpx_1_<slot>`,
   `encg_1_<slot>`, `evc`), then `ref_venc_parity.py --evc evc.f32 --encpx encpx_1_1.f32 --encg
   encg_1_1.f32 --slot 1` (`--selftest` validates the fold/norm/sorted model with no device).

   **RESOLVED (audio): the audio under-gain was a pre_block `norm1`/`norm3` swap — a wiring bug, not a
   norm-kernel scale fault.** An earlier block here attributed both families' ~0.37× to a *shared* final-
   norm / norm-kernel attenuation. That is **wrong**, and stage-by-stage dumps now pin the audio cause
   exactly. Method: dump the audio final chain one stage at a time on device (`R4D_ENC_SEL`) against the
   reference module-by-module, device-input-anchored (same staged waveform both sides). The bf16 `b_ax2`
   readback turned out to **lag the arena** (it dumped ±1e38/NaN while its own `mean_proj` output `b_amb2`
   was finite — impossible for a linear map), so the harness gained `R4D_ENC_SEL=amb` to read `b_amb`
   instead, the **f32 mean_proj *input*** (the pre_block output cast) — the reliable twin. Reading `b_amb`:

   - **pre_block output (`b_amb`): device std 1.65 vs reference 6.62 → ratio 0.25.** The collapse is already
     at the pre_block output, so the fixed `(z-mean)/std` norm and `mean_proj` are both **innocent** (they
     pass the value through: `b_amb2` was likewise 0.27×).
   - The reference `AttnProjection.forward` (`comfy/ldm/minimax/audio_vae.py:267`) is
     `proj(norm3(x)) + attn(norm1(x))` then `+ mlp(norm2(·))`. The **two 2048-wide LayerNorms are not
     interchangeable**: `norm3` has affine |w|mean 0.60, `norm1` 0.27, so `proj` (the dominant branch,
     std 6.6 via `norm3`) dwarfs the attention branch (std 0.4 via `norm1`). **The device had them wired
     backwards** — it staged `norm1` into the projection and `norm3` into the attention (qkv). Feeding the
     small-affine `norm1` to `proj` shrinks it from std 6.6 → 1.8, and the whole block, hence every latent
     downstream, to ~0.25–0.27×. Measured: correct wiring `proj(norm3)+attn(norm1)` std 6.67; swapped
     (device) `proj(norm1)+attn(norm3)` std 1.83 = 0.274×. Exactly the observed collapse.

   **The fix** (`arch/minimaxh3/minimaxh3.cpp`, pre_block issue sequence): stage `norm1` for the attention
   (feeding `p_op_qkv`) and `norm3` for the projection (feeding `p_op_proj`), matching the reference. Both
   LN ops are the *same* `ln(...,2048,b_apb→b_aln3)` kernel — only which consumer reads which norm changed,
   so **no kernel touched, no re-convert** (weights already carry `norm1.*`/`norm3.*` correctly). This also
   **kills the "shared encoder-norm-kernel scale fault" hypothesis**: the audio pre_block now matches the
   reference using that *same* LayerNorm kernel, so the kernel is fine — the loss was pure wiring, and it
   sat **upstream** of the final norm (the "final norm resets the body gain" reasoning was itself wrong:
   `mean_proj` has no LayerNorm after it, and it faithfully passed the collapsed value through).

   **Verified device-side (seed=7 tone, container `minimax-h3-ref2va-fp8`, warm server):**

   | stage | pre-fix | post-fix | reference |
   |---|---|---|---|
   | pre_block out `b_amb` std | 1.65 (ratio 0.25) | **6.43 (ratio 0.97)** | 6.62 |
   | final latent `b_enca` std | 0.23 (ratio 0.39) | **0.546 (ratio 0.93)** | 0.588 |
   | final latent affine `a`,`R²` | 0.385, 0.975 | **0.923, 0.989** | (identity) |
   | final latent relFro → after-affine | 0.79 → 0.062 | **0.302 → 0.099** | — |

   The final latent is now a near-identity map of the reference (slope 0.92, intercept +0.03, R² 0.989) —
   no remaining systematic gain, unlike the pre-fix 0.385×. **The ~0.10 residual-after-affine is
   precision-bounded, not structural:** a *full-model bf16* run of the reference diverges **55%** (relFro
   0.55) from its own fp32 latent, so a low-precision device cannot reach fp32; the device keeps fp32
   accumulation and lands at 0.10 after removing the (now-gone) gain — well inside what the quantised
   container allows. Reproduce: dump `R4D_ENC_SEL=amb` (pre_block out) and unset (final `enca`) over one
   `ref_audio` request, then `ref_aenc_parity.py --stage amb --dev encamb.f32 --encsm encsm_4_1.f32`
   (pre_block) and `--enca enca.f32 --encsm encsm_4_1.f32 --kind audio --slot 1` (final latent).

   **The video / image (`conv3d`) encoder is NOT explained by this and remains open.** It has **no
   `AttnProjection` / pre_block**, so its ~0.369 under-gain is a *different* cause — and since the shared
   LayerNorm kernel is now demonstrably correct on the audio path, the "one norm kernel damps both
   families" reading is dead. Localising the video gain needs its own stage dumps (`norm_out`→`conv_out`→
   `quant_conv`), on kernels the live image / plain paths also run, and stays deliberately **not**
   blind-fixed. Stacked on it is the GroupNorm-extent gap below (#20). Reproduce the video case:
   `ref_venc_parity.py --evc evc.f32 --encpx encpx_1_1.f32 --encg encg_1_1.f32 --slot 1`.

   **Separate real fidelity gap found while localising (its own ticket, not the 0.37):** the device
   GroupNorm reduces statistics over a level's **whole clip volume** (one mean/rstd per group;
   `ref_video.cpp:472-495`, `r4d_upscaler.hip:540-565`), but the reference `TemporalIsolatedGroupNorm`
   normalises **per frame** (time merged into batch; `comfy/ldm/minimax/vae.py:122-130`). This is exactly
   neutral for the single-image parity measurement above (`T=1`), but it **biases real multi-frame video
   reference encodes** — which is what DaSiWa feeds (ref videos are many frames). It changes per-frame
   statistics (shape), not a clean scale, so it is a second, independent divergence stacked on the 0.37 for
   video refs; recorded here, deliberately **not** touched (shared kernel; may be an intentional
   temporal-consistency choice).

   What still differs from the DaSiWa reference (hence not pixel-for-pixel reproducible):
   - **The text encoder only *names* a moving picture; it never sees its pixels.** DaSiWa also shows
     Qwen a strip of timestamped frames (~2 fps) so the language model perceives the motion; we give
     it only the `<Video k>` / `<Audio j>` text label ahead of the prompt. Motion / identity
     conditioning for a video therefore rides **entirely** on its VAE rows — images are the only
     references that also carry soft-token rows. **The 2 fps frame strip is deferred.**
   - **Audio is mono-spread, not kept stereo.** `encoder_audio` returns one mono stream; the server
     duplicates it across the encoder's two channels. A truly stereo source is downmixed, unlike a
     workflow that might preserve both channels.
   - **Frame and slot ceilings.** A reference video is sampled and trimmed to a clip-friendly length
     (`n % 17 == 5`, at least 5 frames, at most `max_enc_frames`), and slots are 1..3 for videos and
     1..3 for audios — at most three videos, three soundtracks and three standalone audios. The row
     budget (point 2) and the interleaving mod-group ceiling (point 1) remain the true binding limits
     at high resolution, and now bind for real, since videos reopen the video-conditioned cell.

5. **References and latent upscaling are mutually exclusive.** Combining the widened reference
   layout with the small-size-then-upscale denoise is not unified; a request asking for both is
   refused at submit (*"reference conditioning does not yet combine with latent upscale"*). Ask for
   the target size directly.

6. **Numeric parity with the reference denoiser for the *conditioned* step is not yet published.**
   A per-tensor / perceptual match to ComfyUI's outputs for the wide step remains an open parity task
   (and, for the encoder values specifically, is now actionable rather than blocked — a `torch` env is
   on the box, point 4 — but has not yet been produced).
   **All three** reference kinds — image, video and audio — are verified *active* and *content-sensitive*
   on-device with a warm-server fixed-seed content-control (see the checks below); each is also
   deterministic (a same-reference rerun is bit-identical). Strict *value* parity (the exact latents
   matching the reference encoder) is a separate, still-open task and has never been the claim. Plain
   (unconditioned) generation retains its existing parity, and the merge is a no-op there: with three
   fine segments the merged group table *is* the fine table.

## Reproducing the checks

- **Host:** `tests/ref2va_pack_test.cpp` — asserts the group invariants: photos collapse to the
  prompt/photos/audio/video quartet; group boundaries are a subset of the fine boundaries; the merge
  is sequence-length independent; a plain request merges to exactly its three fine segments.
- **Device (controlled — a *fixed seed on every job*, or a distinct output proves nothing):** run
  `/v1/videos` with `task=ref2va` and one `seed` value held constant. Two jobs with the **same seed,
  same prompt, same reference count, but different reference content** must give **different**
  `sha256`s — that content-control is what proves the reference is genuinely consumed (a with-ref vs
  no-ref pair is confounded: an image also changes the prompt's token count). For **images** this
  holds for **images** (two same-size photos → distinct output), and now also for **videos and audios**:
  same seed, one reference video (or one reference audio), only the media content differing → **distinct**
  *decoded-stream* md5 (point 4). Compare decoded streams (`ffmpeg -map 0:v` / `-map 0:a -f md5`), never
  raw MP4 bytes (the muxer stamps them), and on a warm server (the first post-boot request is a cold-start
  outlier). Do **not** read a with-video vs no-video difference as content-control — that only shows the
  reference *count* changed the sequence length (a geometry effect); the differing-content, fixed-count,
  fixed-seed pair above is the honest test, and it reports all three kinds live.
  Independently of content, the path must *run*: a `ref_image_1 … ref_image_9` job must complete with
  a clean fault scan (the ceiling lift — the pinned build refused nine at submit), and a single request
  carrying **a picture + a video + that video's `ref_video_audio` soundtrack + a standalone
  `ref_audio`** must complete to an MP4 with a clean fault scan (every encoder pass + the widened step
  execute — this is the complete DaSiWa shape running end-to-end). Sending `seed` with reference parts
  requires `task=ref2va`; reference parts must be real file parts (`curl -F ref_image_1=@pic.png`,
  `-F ref_video_1=@clip.mp4`, `-F ref_audio_1=@tone.wav`) — note a leading `<` or `;` in a text field
  needs `--form-string`, not `-F`.
- **Full-shape content-control (the DaSiWa max request, all kinds live).** One request carrying four
  `ref_image_N`, a `ref_video_1` + its `ref_video_audio_1`, and a standalone `ref_audio_1` is accepted at
  submit (the interleaving group guard passes when the four photos are contiguous) and completes to an MP4
  with a clean fault scan. Holding the seed, the four photos, the video and the soundtrack fixed and changing
  only the *standalone audio* gives **different** decoded video *and* audio streams — so the full-shape
  denoise consumes the audio reference too, not just its count. Reproduction scripts: `refdet.sh`
  (video-ref determinism + content-control), `chain.sh` (tail-as-`ref_video` chaining), `fullshape.sh`
  (this max-shape control) under the build tree.

## Video-encoder value-parity — **RESOLVED** (2026-10-19): shortcut-after-norm1 ordering bug

The video/image VAE encoder was device-live and content-sensitive but its folded latent sat at roughly
**0.49× the reference's scale** with a large structural residual. Root cause found and **fixed**: in
`step_venc`, the shortcut convolution was issued **after** norm1, which normalizes the shortcut level's
input **in place**. So on every shortcut level (a level whose `cout != cin` — levels 1 and 3 here) the
residual was fed `silu(norm1(x))` instead of the **raw** `x` the reference uses
(`ResnetBlock3D.forward`: `x = self.nin_shortcut(x)` before any norm, added to conv2's output). Feeding the
normalized value into the 1×1 shortcut systematically shrank the residual term, giving the ~0.49× structural
deficit that accumulated from the first shortcut level down the trunk. The fix reorders the issue so
`op_sc` reads the raw `in` **before** `op_n1` overwrites it (`arch/minimaxh3/minimaxh3.cpp`, the `f0`
branch of the level/block loop). This is an op *reorder*, not a kernel change, and it corrects the **shared**
trunk — so it fixes the **image** encoder's fidelity too.

**Method that cracked it — a frame-0 per-level bisection.** Extended `R4D_ENC_SEL=all` to dump, in one pass,
frame-0 f32 PERSIST shadows of *every* level's post-resnet output `v_lv[0..4].a` (fields `dbg_ve_l0..l4`)
alongside the existing tail (`trunk/nout/q48/q2`). Frame-0 shrinks fast spatially so all four level shadows
plus the trunk stay ~68 MB in one host slab. The reference side (`ref_level_outs`) captures `h` after each
level's block loop and before its downsample, on the device's **own** staged pixels. The std ratio walked the
deficit to its exact entry point:

| stage | pre-fix ratio | post-fix ratio |
|---|---|---|
| conv_in out (frame0) | 0.999 | 0.999 |
| L0 post-resnet | 0.999 | 0.999 |
| **L1 post-resnet** | **0.455** | **1.000** |
| L2 | 0.466 | 1.000 |
| L3 | 0.380 | 1.000 |
| L4 | 0.331 | 1.000 |
| norm_out / conv_out / quant_conv (raw mu) | ~0.49 | **1.000 / 1.000 / 0.998** |

conv_in and level-0 (no shortcut) stay faithful; the drop appears *exactly* at level 1 (first shortcut level)
— pinpointing the shortcut as the culprit, not the downsample, not a spatial kernel, not bf16 GEMM (the trunk
convs run **host-only** in this config anyway — `conv3d_ndhwc` resolves to a host kernel). Post-fix every
stage matches the reference to the **bf16 precision floor** (`relFro ~3–4e-3`, affine `1.000·ref + 0.000`).

What the earlier pass had already correctly **exonerated** (still true, and why the bisection was the right
cut — the gap was upstream of all of them, at the shortcut): input normalization (byte-identical to
`_normalize_pixels`); `norm_out` (GroupNorm on the device trunk reproduces `nout` to 1.6e-3); GroupNorm extent
(was task #20 — device is **already per-frame**, matching `TemporalIsolatedGroupNorm`; no change); temporal
staging (a causal frame-0 sees only itself — excluded); weight precision (venc not in the convert worst-20).
The "structural spatial trunk / needs a golden box" read from the prior pass was wrong: it *was* structural,
but confined to one op-order, and the frame-0 bisection made it golden-verifiable.

**Verified device-live**: a single-image `ref2va` encode generated cleanly with the fix loaded, and all level
shadows + the full tail now read ratio ~1.0 against the reference. Strict end-to-end numeric parity of the
whole pipeline (benchmarks, multi-frame) remains #10/P6. #20 stays closed (norm already per-frame).

## Engine-throughput vs ComfyUI: Radiance TP2 wall-clock and card-seconds (2026-10-19)

FFmpeg is now present on this box (the §6.1 "no libav → encoding_failed at t≈0" blocker is gone — clips
mux to mp4 and generate to completion). The encoder **value-parity** half of P6 is done (both encoders at the
bf16 floor, above). The comparative half is now **unblocked** — the external Hermes stock-vs-sol-attn baseline
arrived (`radiance-fl2va-examples/solattn-bench-results.md`, single R9700, REF2VA 1-image, 25 step, compute
columns with queue separated). The two are now matched **shape-for-shape** (task=ref2va, ONE image ref, 25
step, seed fixed, `wall = completed_at − created_at` on an empty queue ⇒ wall≈compute).

**Matched battery — Radiance TP2 (`--tp 2 --vision-tower-placement ram --max-model-len 4096`):**

| shape | 3 s | 4 s | 5 s | 6 s | 12 s |
|---|---|---|---|---|---|
| 256×256 · ref2va · 25 step | 13 s | — | 21 s | — | — |
| 512×512 · ref2va · 25 step | 42 s | 62 s | 72 s | 95 s | **207 s** |

**512×512 comparison vs ComfyUI single-card (same shape/config, compute):**

| 512²·s | Rad TP2 | ComfyUI stock | ComfyUI sol | TP2 vs **1** stock | TP2 vs **1** sol | **card-parity**: 2·TP2 vs 2 stock-lanes | 2·TP2 vs 2 sol-lanes |
|---|---|---|---|---|---|---|---|
| 3 s | 42 | 90.2 | 95.2 | **2.15×** | **2.27×** | **1.07× (win)** | **1.13× (win)** |
| 4 s | 62 | 130.7 | 125.3 | **2.11×** | **2.02×** | 1.05× (win) | 1.01× (win) |
| 5 s | 72 | 150.2 | 150.3 | **2.09×** | **2.09×** | 1.04× (win) | 1.04× (win) |
| 6 s | 95 | 195.3 | 175.3 | **2.06×** | **1.85×** | 1.03× (win) | 0.92× (slow) |
| 12 s | 207 | 405.5 | 340.5 | **1.96×** | **1.64×** | 0.98× (≈tie) | 0.82× (slow) |

Two framings, both reported honestly:

- **Per single clip** (one TP2 instance vs one ComfyUI card): Radiance wins at every row — 1.96–2.15× vs
  stock, 1.64–2.27× vs sol. This is a 2-card-vs-1-card comparison (TP2 spreads one clip across both cards)
  and overstates fleet efficiency; quote it only as "a single TP2 request beats a single-card clip."
- **Card-parity fleet** (one TP2 = 2 cards vs a 2-lane ComfyUI fleet = the same 2 cards, equal work = 2
  clips: TP2 serialises them at 2·T, the fleet runs them in parallel at T_1clip): **Radiance beats or ties
  two stock lanes at every row** (1.03–1.07×, and a 0.98× dead-heat at 12 s), and **beats two sol-attn lanes
  at short clips (≤5 s)** because Radiance's native VSA block-sparse attention is always on while sol-attn
  only engages above its ~12 k-token floor. Radiance **trails two sol lanes at 6–12 s** (0.92×, 0.82×) where
  sol's tau-1.3 sparsity fully engages.

**§6 verdict:** the "one TP2 beats two single-card ComfyUI" claim is **MET against stock at every row** (or
dead-heat at 12 s), and **MET against sol-attn below ~12 k tokens (≤5 s @ 512²)** but **NOT met at 6–12 s**,
where sol's sparsity fully engages. The sol-attn gap at the 12 s production row is the honest weak point —
and it is measured under a **215 W per-card power cap (default 300 W), a −75 mV undervolt, a powersave CPU
governor, and a link cap**. The power/undervolt handicap was then A/B'd (see the power-cap subsection below)
and found to be worth only **~5 % at the 12 s row** — so the cap is **not** the cause of the sol gap; the
**link cap** is the heavier constraint. The sol baseline itself carries ±5 s reproducibility. Radiance's wall
also includes ~2.2 s queue+prompt that the ComfyUI compute columns exclude (minor; both are post-submit
processing). Harnesses: `build/tmp/bench.sh` (legacy) and `build/tmp/ex10/bench2.sh` (this matched battery;
results in `build/tmp/ex10/bench/results.tsv`).

**Power-cap A/B — the cap barely moves diffusion (2026-10-20).** The user removed the 215 W cap (→ 300 W
default, confirmed `power1_cap=300 W`) and the −75 mV undervolt (→ `voltage offset 0 mV` in the startup
report); P2P, `iommu=pt` and the gen4 x8 link cap were left unchanged. Re-ran `bench2.sh` (same shapes/config/
seed) into `results_uncapped.tsv`, keeping the capped run as `results_215w.tsv`:

| 512² row | 215 W / −75 mV | 300 W / 0 mV | Δ |
|---|---|---|---|
| 3 s | 42 | 41 | −1 (noise) |
| 4 s | 62 | 61 | −1 (noise) |
| 5 s | 72 | 72 | 0 |
| 6 s | 95 | 93 | −2 (~2 %) |
| **12 s** | 207 | **196** | **−11 (−5.3 %)** |
| 256² · 3 s / 5 s | 13 / 21 | 13 / 20 | 0 / −1 |

The gain is **monotone in duration but small**: ~0 below 6 s, ~2 % at 6 s, **~5 % at 12 s** — the only delta
outside poll noise. `gpu_busy_percent` read **100 % on 519/528 samples**, so the cards are compute-saturated,
yet a +40 % power budget plus the undervolt removal buys only ~5 %. This TP2 diffusion step is therefore **not
SM-clock-bound**: much of that "busy" is the all-reduce waiting on the **gen4 x8 fabric** (and the embedding /
encoder-tower reads streamed over the same link), which does not shrink with core clock. This **corrects the
claim just above** that the cap was a major long-row handicap — the sol-attn gap at 12 s is **not** explained
by the power cap; the link cap is the real ceiling. Practical: leaving the box at 215 W costs ≈5 % worst-case
on diffusion, so the thermal/power saving is nearly free. (`power1_input` isn't exposed on this driver, so
sustained watts weren't logged — `gpu_busy_percent` is the load signal. Uncapped logs:
`build/tmp/ex10/{clock_sampler_300w.log,bench2_uncapped.log}`.)

**Can we dodge the link by forcing the embedding into VRAM? No — it doesn't start (2026-10-20).** The natural
follow-up to the link-bound finding is `--embedding-placement vram`: take the 1.45 GiB/rank embedding read off
the link. At this box's TP2 config it is **not viable**. The table does resolve to VRAM (`placement: token
embedding (1.45 GiB a rank) vram: as asked`), but then the weight-budget resolver leaves the **static
(non-expert) weights filling the entire per-card weight budget**, so *"every expert streams from host memory
and the KV cache gets nothing"*: `vram_kv resolved to 0 MiB`, and Radiance **aborts at startup** (`E startup
stopped: invalid argument`; the resolver's own advice: *"use more cards or a smaller --max-num-batched-tokens"*).
So the shipped `ram` default is **not a naïve inefficiency — it is the trade that leaves VRAM for the KV cache
+ expert residency**. Forcing it to VRAM trades a *small* streaming cost (the embedding) for a *catastrophic*
one (all MoE experts streamed + zero KV cache). The two link-traffic sources (embedding reads, expert streaming)
trade against each other through the VRAM budget, and the balanced point is the default. The process aborted
before any VRAM was claimed (cards stayed at 61 MB), so no benchmark rows were produced; a reduced
`--max-num-batched-tokens` might coax it up, but that changes the batch config and is no longer apples-to-apples
with the 215 W baseline. Log: `build/tmp/ex10/server_vemb.log`.

## The kR2VCond keyframe-pin (FL2VA) — SHIPPED + device-verified (2026-10-19)

**Supersedes the earlier "left optional, not wired" disposition.** The user elevated FL2VA from optional to a
**hard requirement**, so the keyframe path was implemented and shipped (commit `98ecbeb`, placeholder author,
NOT pushed). The `nk=0` gate in `video_driver.cpp` is gone on the keyframe request path; a keyframe request now
stages + VAE-encodes the two anchors and populates the `ref_tab` cond rows.

**Mechanism (correction to the earlier guess).** A keyframe is a **pinned (`update=0`) attention cond row**
carrying its VAE latent at the frame's time position — it is NOT a sampler pin-mask. The **sampler is
untouched**; the denoise step simply widens to carry the cond rows (faithful to `comfy/ldm/minimax/model.py`
`img_update==False` keyframe rows). **One keyframe path serves I2VA / L2VA / FL2VA** (first-frame only,
last-frame only, both). Keyframes are **⊥ refs** — mutually exclusive with `ref_image_N`/`ref_video_N`/etc.

**Device-verified** (seed-9999 differential): pins are **consumed** (fl2va ≠ same-seed t2va) **and force the
endpoints** — frame 0 vs anchor similarity 0.687 pinned vs 0.460 un-pinned; last frame 0.719 vs 0.521.
Endpoint parity is therefore **SOFT / GENERATIVE** (strong with a matching prompt), matching the reference
golden's own loose endpoint honoring (frame0 re-painted, last frame motion-blurred) — not bit-exact. A 73-frame
MP4+AAC renders clean.

**512-output keyframes were broken at ≤256 — found + fixed (2026-10-19).** The `98ecbeb` device-verify above
ran at ≤256; at the production 512² canvas I2VA/L2VA/FL2VA failed with "the video encoder refuses a reference at
512x512 over 1 frames." Root cause: the keyframe is resized to the **target canvas** and its latent rows **share
the TARGET grid** (`rad_ref2va_pack` `ref2va_build` — keyframes carry no own h/w, unlike references), so they
must be VAE-encoded at **output** resolution; but `venc_geom` had a flat `h,w > kEncMaxSide(256)` reject. That
guard is a crude proxy for the real constraint — the encoder holds ONE clip's pixels sized for a
`kVaeClipLength(17)`-frame clip at 256 (`Rmax=17·256²=1.11M` rows); a **one-frame** media (a picture or keyframe)
is a single-frame clip and fits at up to `√Rmax≈1055`/side. Fix = replace the flat side cap with `clip·h·w ≤ 17·256²`
(video refs unchanged: clip=17 forces each side back to 256; single-frame admits full-res). **No buffer/kernel/
arena change** — a 512² keyframe (262 k rows) is a third of the existing budget. A first relax then **faulted**
("op 'cast' n=128 at M=262144 outside band, fatal") because the encoder's **debug shadow** casts (`op_ve_cindbg`,
per-level `op_ve_ldbg`, trunk, nout — write only `b_vcin/b_vl/b_vtrunk/b_vnout`, read only under `R4D_ENC_SEL`)
were issued on **every** encode at a `PF[]` (256-frame-plane) range a 512 plane overflows. Gate them on
`R4D_ENC_DEBUG` (the same env the readback uses) — a no-op for the production latent, and it also stops paying
for debug work on every encode. **Device-verified at 512²:** FL2VA 512×512 completes clean (46.8 s, server
survives, no band fault); host geom/video/ref2va/encoding tests green. Core ops are within their declared bands
(a 1-frame 512 level plane is `0.235×` the 17·256² they're sized for).

**Was a remaining fidelity gap; now CLOSED (2026-10-21).** The pins-only path used to **skip
`clip.tokenize` vision tokens** for the keyframe images (the TE-anchor tokens the reference would also feed),
so a prompt that *contradicts* the pinned image could let the text win. (The over-saturated/posterized
look these renders also showed was a **separate** bug — the velocity head's modulation-table staging,
since fixed; the missing anchors were not its root cause.) This is now
shipped: an endpoint keyframe is fed the DiT on BOTH channels the reference uses — the VAE-latent pin AND a
`<Picture k>` vision-tower anchor block interleaved into the text conditioning. Mechanism: `videos.cpp` now
runs the media/`<Picture N>` block for keyframe requests (first_frame→`<Picture 1>`, last_frame→`<Picture 2>`),
preparing each anchor through the SAME `mm::prepare`/`PromptMedia`/`mm_rows` tower path the reference images
use (a second, independent encode of the keyframe — the tower sees the aspect-kept area-matched image, the
pin sees the stretch/cover exact canvas), IN ADDITION to the `enc_pixels` pin at `videos.cpp:709-730`. The
vision tower is optional: a container converted WITHOUT it still renders keyframes pin-only (`video_driver.cpp`
tower guard now skips FIRST/LAST_FRAME; keyframes still require the VAE `encodes`). Host tests
`fl2va_keyframes_feed_the_tower_and_pin_the_output_ends`, `fl2va_without_a_tower_pins_only` (videos_test) and
`a_keyframe_completion_also_conditions_through_the_tower`, `a_keyframe_completion_needs_no_vision_tower`
(video_gen_test) cover both channels and both tower-present/absent paths. The earlier "no golden engine to
diff against" limit still bounds EXACT value-parity (→ #10/P6): the reference embeds `<Picture N>` inline in
its prompt prose while Radiance prepends the vision block before the prompt (the repo's PORT-REF2VA §3
convention, which the reference tokenizer also uses for the block itself); anchor PRESENCE is what this change
delivers. Chaining also still works independently via the ref_video slot0 path. Device qualitative check (does
the posterize drop) is the remaining validation.

**Endpoint parity re-confirmed at 512 (2026-10-19, post-fix).** Same method (frame-0/last-frame cosine vs the
resized anchor, 256² sampling space, RGB-mean-centred), on the fixed binary at 512²·12 s·25 step·seed 1234:

| request | endpoint vs its anchor | same-seed un-pinned t2va endpoint |
|---|---|---|
| FL2VA frame 0 vs `first` | **0.984** | 0.04 |
| FL2VA last  vs `last`  | **0.985** | −0.573 |
| I2VA  frame 0 vs `first` | **0.985** | 0.04 |
| L2VA  last  vs `last`  | **0.985** | −0.573 |

Every pinned endpoint lands at cos≈0.985 to its anchor; the same-seed un-pinned clip sits at ≈0/−0.6. The pins
**force** the endpoints at output resolution (not just at ≤256), with a much wider pinned/un-pinned gap than the
earlier ≤256 run (0.687 vs 0.460). Still soft/generative (the last frame can carry a little motion blur), not
bit-exact.

**Example-clip inventory** (`build/tmp/ex10/examples/`, `examples.sh` + `examples_kf.sh`; all 512²·12 s·25
step·seed 1234, `first`=standing, `last`=riding; `index.tsv` records status/wall/id):

| category | mp4 | wall | note |
|---|---|---|---|
| T2VA   | `t2va.mp4`   | 203 s | text-only baseline (also the un-pinned endpoint reference) |
| I2VA   | `i2va.mp4`   | 208 s | `first` pinned to frame 0 |
| L2VA   | `l2va.mp4`   | 209 s | `last` pinned to last frame |
| FL2VA  | `fl2va.mp4`  | 214 s | both endpoints pinned (294 frames) |
| REF2VA | `ref2va.mp4` | 214 s | 2 image refs (`last`,`first`), not keyframes |

Each has a matching `.thumb.webp`. These are SEQUENCE clips (single generated run), not loops.

## 2026-10-21 — per-request `shift`, and the "fried" look hunt — **resolved later: the velocity-head modulation table, NOT decode/quant**

The full-resolution frames of a workflow-prompt review batch showed a **blown-highlight / crushed-black
"watercolor" look** (the user's "fried"). Measured with HSV + Laplacian-variance across time on native-512
frames, versus the reference clips (same `dasiwaHybridV4` finetune, via ComfyUI):

| render | black% | clip% (blown) |
|---|---|---|
| reference snow | 1.5–4.7 | **~4.5** |
| Radiance snow T2VA (this batch) | 11–15 | **54–79** |
| Radiance snow T2VA (old ex10) | 13.7 | 51 |

Key facts established by this pass:
1. **It is in the text-only T2VA control too** — so it is NOT the keyframe/vision-tower path (the
   TE-anchor work below), and
   the old ex10 T2VA has the same clip% → **not a regression**. The earlier "T2VA is clean" read was an
   artifact of judging 256² downscaled strips; at native res it was never clean.
2. **Sampler is not the lever**: same-seed euler vs res_multistep gives clip 53–61% vs 56–65% — no
   meaningful change (matches the user's eye).
3. **The container already runs shift 12 / 3** (`/v1/models` `video_shift`=12.0, `video_audio_shift`=3.0)
   — the reference's exact settings. And rendering at `shift=4` leaves clip at ~54%. So shift is not the
   lever either. CFG is not implemented (single conditioning pass = CFG 1), which already matches the
   reference's BasicGuider scale-1.
4. With sampler, shift and CFG all matched, Radiance still blows ~55% where the reference blows ~4.5% on
   the same content ⇒ the gap is downstream of the schedule. *(This pass localised it to "the video VAE
   decode and/or the DiT int8 quant" — **that localisation was wrong.** The cause was the velocity head's
   modulation table: the norm_out shift/scale row was staged at the staging call's narrow pitch instead of
   the declared row width, so the table arrived all zeros at t>0 and both heads ran un-modulated. Two
   lines fixed it and the fried look vanished at every quant; no quantisation was ever implicated, and
   the decode-vs-denoise framing of this item is kept only as measured history.)* What this pass did
   establish stands: not a schedule, sampler, keyframe, downscale or source-compression issue.

**Shipped anyway** (the user asked for it, and it is correct and useful for other checkpoints/workflows):
`shift` (video) and `audio_shift` (soundtrack) are now per-request `/v1/videos` fields, `(0,100]`, each
defaulting to the model's trained shift; echoed in the video object and advertised in the model card as
`video_shift`/`video_audio_shift`. Implementation is **ABI-invisible**: `VideoRequest` carries the two
doubles, `video_schedule()` re-shifts that stream's sigma grid, and the shifted sigmas are staged through
`RadVideoGenBatch::sched` — the plugin reads only sigmas/timesteps, never a shift scalar, so
`RAD_ABI_VERSION` stayed 24 then (it is 27 now) and no plugin rebuilt. Files: `core/gen/video.h`, `core/gen/video_driver.cpp`,
`core/server/iface.h` (`VideoGenInfo` shift), `core/engine.cpp` (populate), `core/server/videos.cpp`
(allowlist/parse/echo/caps); tests in `videos_test.cpp`, `video_gen_test.cpp` (host, green). Device:
`shift=4` accepted+echoed+renders.
