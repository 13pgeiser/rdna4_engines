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
  **pre_block `norm1`/`norm3` swap** (the projection branch read the wrong LayerNorm — see #14 below). After
  the fix the pre_block output and the final latent both track the reference (ratio ~0.97 / ~0.93, R² 0.98–
  0.99, no remaining systematic gain); the ~0.10 residual-after-affine is bounded by low-precision
  execution (a full-bf16 reference itself moves 55% from its fp32). **The video / image encoder is a
  separate, still-open divergence** — it has no pre_block, so this fix does not cover it.
- **Continuity chaining — drivable today through the reference paths.** DaSiWa's scene-to-scene lead-in
  is the previous clip's tail fed back as a `ref_video_N` (+ `ref_video_audio_N`), not a special engine
  mode. Two requests reproduce it: generate clip A, cut its tail, request clip B with `ref_video_1` = the
  tail. A fixed-seed control confirms the tail conditions B (with-tail vs without-tail differ). The
  pack's exact keyframe-pin rows (`kR2VCond`) remain unimplemented — a separate mechanism for FL2VA-style
  start/end-frame pinning, which FL2VA itself reaches with two image references (see point 3).
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

3. **Continuity chaining works through the reference paths; the bespoke keyframe-pin mechanism is
   separate and still pending.** The DaSiWa "3-second lead-in" is *not* a special engine mode: per the
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
   subtly. What is *not* wired is the **engine consumption** of the pack's *keyframe* rows
   (`kR2VCond` / `kR2VCondAudio`) — a distinct exact-pin mechanism for FL2VA-style start/end-frame
   conditioning (which FL2VA itself reaches with two image references — see §2 of the continuity brief).
   **The pack layer is done and reference-verified**: `ref2va_build` emits `kR2VCond` / `kR2VCondAudio`
   segments, and the golden `fl2va` (two keyframes) and `chain` (one keyframe) cases assert their
   segment order, sizes, positions, `img_pos`/`audio_pos` and modulation rows **bit-exact** against the
   reference (they re-baked byte-identical here). What remains is three engine pieces, none touching
   the sampler (cond rows sit ahead of the targets, so the sampler ignores them by construction): the
   driver never populates a keyframe row (its keyframe count stays 0 today), `step_dit` scatters the
   reference rows but not the cond rows, and there is no continuation-tail *input* to persist a prior
   clip's tail. Seam frame-equality is the wrong metric to gate on: the brief states it is imperfect
   **by construction**, because the first new frame is the tail's temporal *successor*, not a replay of
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

## #14 video-encoder value-parity — **RESOLVED** (2026-10-19): shortcut-after-norm1 ordering bug

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

## #10 / P6 — Radiance TP2 engine-throughput measured (2026-10-19)

FFmpeg is now present on this box (the §6.1 "no libav → encoding_failed at t≈0" blocker is gone — clips
mux to mp4 and generate to completion). The encoder **value-parity** half of P6 is done (both encoders at the
bf16 floor, above). What was measured now is Radiance's own **engine wall-clock** at fixed shapes
(`--tp 2 --vision-tower-placement ram --max-model-len 4096`, `wall = completed_at − created_at`, empty queue):

| job | shape | wall |
|---|---|---|
| plain t2v | 256×256 · 3 s · 25 step | **12 s** |
| plain t2v | 256×256 · 3 s · 50 step | 21 s |
| plain t2v | 512×512 · 3 s · 25 step | 38 s |
| plain t2v | 256×256 · 6 s · 25 step | 23 s |
| ref2va full shape | 256×256 · 25 step (4 img + 1 vid + 2 aud) | **93 s** |
| ref2va full shape | 512×512 · 25 step (4 img + 1 vid + 2 aud) | **121 s** |

**These are engine throughput, not the §6 claim.** §6's win condition is `N × T_rad_tp2 < T_comfy_1clip`
(one TP2 instance beating two single-GPU ComfyUI instances at equal work). The documented ComfyUI floors
(590.4 s @ 12 s / 704×544 / 25 step; 315 s @ 7 s) are **stock-attention** (pre-sol) and INCLUDE queue wait —
per the §6 Hermes correction they are only a floor, and the clean stock-vs-sol-attn pairs that the claim must
be measured against are an **external Hermes deliverable** not available on this box (there is no second GPU
pair running ComfyUI here). Against those floors the full-shape 512 clip (121 s) is far below a single
ComfyUI clip, so the claim is *directionally* supported, but **P6's comparative verdict is blocked on the
external sol-attn baseline**, not on anything in this branch. The harness is `build/tmp/bench.sh`.

## #8 kR2VCond keyframe-pin (FL2VA exact-pin) — disposition: left optional, not wired (2026-10-19)

The pack layer is **done and reference-verified** (`ref2va_build` emits `kR2VCond`/`kR2VCondAudio`; golden
fl2va/chain assert segment order/size/position/`img_pos`/`mod` bit-exact; `ref_pack_of`/`rad_arch_video_pack`
already parse `nk` up to 8). What remains is **engine wiring only**: driver populates keyframe rows in
`ref_tab` (`nk` stays 0 today — see `video_driver.cpp` submit/stage_geom "continuation is separate"), `step_dit`
scatters cond rows from the reserved `RAD_ENC_TAIL` slot via `op_vin`/`op_ain`, a continuation-tail *input*
API, and persisting the prior clip's tail latent across requests.

**Not implemented, deliberately.** DaSiWa chaining — the actual capability — already works via the
**ref_video slot0 path** (no new code, device-verified), so the exact-pin is an alternative pinning mechanism,
not a requirement. The wiring lands in the **shared `step_dit`** (serves every conditioned + plain path), and
its *purpose* — an exact frame pin — is a **numeric-fidelity** claim with no golden to check against on this
box (there is no reference FL2VA engine to diff latents against; same external limit as P6's comparative
claim). Building it would be a blind change to shared kernels that could ship a silently-wrong latent, which
the workflow rule forbids. It is left **pending and optional**; a function-level device battery (keyframe
present/absent → deterministic content-sensitive change) is the verification available here, but that
verifies consumption, not the exact pin, and does not justify the shared-path regression risk given chaining
already works. **User action to unblock a faithful implementation:** a reference FL2VA engine (or the Hermes
golden rows) to diff the pinned-clip latents against.
