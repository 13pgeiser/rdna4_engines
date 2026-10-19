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
  runs the full conditioned denoise on both cards and returns a video. Three fixed-seed runs of the
  same prompt — no reference, one reference, nine references — produce three distinct outputs, so
  the conditioning is demonstrably *live* (not inert) and reference-*aware*, and nine photos no
  longer hit the earlier submit-time refusal.
- **Reference-media plumbing and caps.** The container declares 9/3/3 (images/videos/audios); the
  VAE-encode passes, the reference table, and the widened step layout are all in place, and the
  encode path is now unblocked by the host FFmpeg install.
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

## What this port still cannot reproduce (the honest list)

1. **Many *interleaved* mixed-modality references would exceed the 8-group table** *(prospective —
   only reachable once reference videos are wired, see point 4; today every accepted reference is a
   photo, which is four groups and never binds).* Photos are free, but every reference *video* would
   contribute two cells in order — *video-conditioned* then its *audio-conditioned* soundtrack — and
   a photo placed after it **reopens** the video-conditioned cell, adding a group. Alternating photos
   with video-plus-soundtrack refs could push the merged run past `kMaxSegs = 8` cells, which the
   driver refuses at submit: *"these references change the conditioning's modulation cell N times,
   but this deployment's modulation table holds 8 cells — ask for references that share their
   modalities and target, or fewer of them."* **Workaround when videos arrive:** group references by
   modality (all photos together, then the videos) so same-kind refs are contiguous and merge; the
   ceiling only bites on heavy interleaving.

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

3. **Continuity (tail conditioning) is not yet implemented.** The pack places keyframe / tail
   *conditioning rows*, but the driver path that conditions on them **outside the sampler** and trims
   after sampling — what makes clip N+1 continue seamlessly from clip N — is still pending. So
   multi-segment seamless continuation, as the DaSiWa workflow uses it, is not reproduced yet. Single
   generation conditioned on reference *images* is (see point 4 for video / audio).

4. **Reference videos and audios are refused at submit — not yet wired, even though the encoders
   load.** `ref_video_N`, `ref_video_audio_N`, and `ref_audio_N` parts are accepted by the parser,
   capped (3 / 3), and paired correctly, but the request is rejected with *"'ref_video_1' is not
   consumed yet: reference videos and audios reach the model through the video and audio VAEs …
   only ref_image_N parts condition a generation today."* The VAE encoders are declared and loaded,
   and the `Vencode` pass runs for **image** references, but no path decodes a video's frames or an
   audio's samples into the encoder. **Only reference images condition a generation today.** This is
   the single largest remaining DaSiWa gap. The scaffolding is already designed for it — the
   `VideoRefMedia` encoder view is documented as *"IMAGE/VIDEO fill the pixels, AUDIO/VIDEO_AUDIO the
   samples,"* `decode_video` / `decode_audio` exist in the media layer, and the driver's encode
   builder already branches on audio — so enabling them is bounded work, not a redesign:
   (a) server-side, decode a video's frames / an audio's samples into `enc_pixels` / `enc_samples` on
   the encoder grid (mirroring the image loop); (b) driver-side, drop the submit guard and enqueue
   `RAD_ENC_VIDEO` / `RAD_ENC_VAUDIO` / `RAD_ENC_AUDIO` with the right latent grid and slot; (c) a
   per-kind device differential (video, audio, video-with-soundtrack) proving each conditions the
   output. Until then, treat the mixed-modality interleaving ceiling in point 1 as **prospective** —
   today every accepted reference is a photo, which merges to four groups.

5. **References and latent upscaling are mutually exclusive.** Combining the widened reference
   layout with the small-size-then-upscale denoise is not unified; a request asking for both is
   refused at submit (*"reference conditioning does not yet combine with latent upscale"*). Ask for
   the target size directly.

6. **Numeric parity with the reference denoiser for the *conditioned* step is not yet published.**
   Conditioning is verified *active* and *reference-content-sensitive* on-device (see the controlled
   check below), but a per-tensor / perceptual match to ComfyUI's outputs for the wide step is an open
   parity task. Plain (unconditioned) generation retains its existing parity, and the merge is a no-op
   there: with three fine segments the merged group table *is* the fine table.

## Reproducing the checks

- **Host:** `tests/ref2va_pack_test.cpp` — asserts the group invariants: photos collapse to the
  prompt/photos/audio/video quartet; group boundaries are a subset of the fine boundaries; the merge
  is sequence-length independent; a plain request merges to exactly its three fine segments.
- **Device (controlled — a *fixed seed on every job*, or a distinct output proves nothing):** run
  `/v1/videos` with `task=ref2va` and one `seed` value held constant. Two jobs with the **same seed,
  same prompt, same reference count, but different image content** must give **different** `sha256`s —
  that content-control is what proves the reference is genuinely consumed (a with-ref vs no-ref pair
  is confounded: an image also changes the prompt's token count). Add a `ref_image_1 … ref_image_9`
  job: it must *complete* with a clean fault scan — that nine-photo completion is the ceiling lifted
  (the pinned build refused it at submit). Sending `seed` and the reference parts requires `task=
  ref2va`; the reference parts must be real file parts (e.g. `curl -F ref_image_1=@pic.png`).
