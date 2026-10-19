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
- **Video and audio references — wired, host-tested, and running on-device without fault; NOT yet
  content-active.** The submit refusal that used to reject `ref_video_N` / `ref_video_audio_N` /
  `ref_audio_N` is gone. The server decodes each into the encoder's form, the driver stages a
  `Vencode` / `Aencode` pass and folds it into the reference table, and the widened step consumes it
  (point 4). This is proven end-to-end: the host tier is green (`videos_test` decodes and speaks every
  kind; `video_gen_test` clips-pads / zero-pads, slots a soundtrack to its video, and lays the table,
  and refuses a non-encoding container by name), **and on the device the complete DaSiWa shape — a
  picture + a video + that video's soundtrack + a standalone audio, four reference parts in one
  request — runs every encoder pass and the widened denoise to a clean MP4 with no engine fault.**
  What is **not** yet true: a fixed-seed content-control (same prompt, same reference *count*, only the
  reference *content* differing) shows a reference **video** or **audio does not change the output**.
  The reference **image** does (it also rides the text encoder — see point 4). The reason is isolated
  to the VAE **encoders** producing content-invariant latents, not to the decode, staging, upload or
  reference-table scatter (all of which are proven to carry distinct content per media item); it is the
  open **encoder value-parity** task. Until that lands, a video / audio reference is accepted and
  encoded but is, in effect, ignored by the output — treat image references as the only ones that
  currently steer the generation.
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

3. **Continuity (tail conditioning) is not yet implemented.** The pack places keyframe / tail
   *conditioning rows*, but the driver path that conditions on them **outside the sampler** and trims
   after sampling — what makes clip N+1 continue seamlessly from clip N — is still pending. So
   multi-segment seamless continuation, as the DaSiWa workflow uses it, is not reproduced yet. Single
   generation conditioned on reference *images* is (see point 4 for video / audio).

4. **Reference videos and audios are wired and run — through the VAEs, not the text encoder — but do
   not yet steer the output.** A
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

   **The one thing that does not yet work is the encoder's numbers, not its plumbing.** A controlled
   fixed-seed check — same prompt, same reference count, only the media content differing — shows a
   reference video (two 512×512 videos with genuinely different decoded frames) and a reference audio
   (a tone, a chirp, white noise) each producing a **byte-identical** output, while a reference image
   of the same count changes it. The cause is isolated to the encoder by two facts. First, a debug
   hash of the encoder *input* as the driver stages it is distinct for every media item (a tone, a
   chirp and white noise all differ from silence and from one another), so decode, padding and
   host→device upload hand the encoder genuinely different bytes. Second, the step folds those latents
   into the sequence with a **linear projection** (`op_vin` / `op_ain`, each a `dense_bias` GEMM of the
   reference's slot rows in `b_evc` / `b_enca` into `b_h`) — a linear map cannot send two different
   input vectors to the same output, so had the encoder produced different latents, the conditioning
   rows — and therefore the output — would differ. Both point the same way: the encoder is turning all
   sufficiently-nonzero content into the same latent. A third check rules out the tempting shortcut — that the container's
   encoder weights were converted empty or as placeholders (a zero weight maps every input to one bias, exactly this
   signature): the `venc` / `aenc` tensors came from the genuine MiniMax-H3 sources and their recorded quantisation error
   is ordinary bf16 rounding (relative Frobenius ≈ 1.6e-3, which a zero tensor cannot produce). The weights are real; the
   defect is in how the kernels *execute* them. That is a
   numerical-value-parity task (golden-compare the encode chain against the reference encoder, run by
   run), deliberately **not** attempted blind here — a change that made the latents merely *differ*
   could make them *wrong*, and the encoder kernels are shared with the image path (which is verified
   live via the text encoder) and plain generation, both currently green. Until it lands, video /
   audio references are accepted and encoded but inert; image references are the kind that steers.

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
   A per-tensor / perceptual match to ComfyUI's outputs for the wide step is an open parity task.
   **Reference-image** conditioning is verified *active* and *content-sensitive*
   on-device (see the controlled check below); reference **video / audio** conditioning is wired and
   runs to a clean completion but is **not** yet content-sensitive, because the VAE encoders are not
   yet value-correct (point 4). Plain (unconditioned) generation retains its existing parity, and the
   merge is a no-op there: with three fine segments the merged group table *is* the fine table.

## Reproducing the checks

- **Host:** `tests/ref2va_pack_test.cpp` — asserts the group invariants: photos collapse to the
  prompt/photos/audio/video quartet; group boundaries are a subset of the fine boundaries; the merge
  is sequence-length independent; a plain request merges to exactly its three fine segments.
- **Device (controlled — a *fixed seed on every job*, or a distinct output proves nothing):** run
  `/v1/videos` with `task=ref2va` and one `seed` value held constant. Two jobs with the **same seed,
  same prompt, same reference count, but different reference content** must give **different**
  `sha256`s — that content-control is what proves the reference is genuinely consumed (a with-ref vs
  no-ref pair is confounded: an image also changes the prompt's token count). For **images** this
  holds (two same-size photos → distinct output), proving image conditioning is live. For **videos and
  audios it does NOT hold yet**: same seed, one reference video (or one reference audio), only the
  media content differing → **identical** `sha256`, because the VAE encoders are not yet value-correct
  (point 4). Do **not** read a with-video vs no-video difference as video conditioning working — that
  only shows the reference *count* changed the sequence length (a geometry effect, not content); the
  content-control pair above is the only honest test, and it currently reports video / audio inert.
  Independently of content, the path must *run*: a `ref_image_1 … ref_image_9` job must complete with
  a clean fault scan (the ceiling lift — the pinned build refused nine at submit), and a single request
  carrying **a picture + a video + that video's `ref_video_audio` soundtrack + a standalone
  `ref_audio`** must complete to an MP4 with a clean fault scan (every encoder pass + the widened step
  execute — this is the complete DaSiWa shape running end-to-end). Sending `seed` with reference parts
  requires `task=ref2va`; reference parts must be real file parts (`curl -F ref_image_1=@pic.png`,
  `-F ref_video_1=@clip.mp4`, `-F ref_audio_1=@tone.wav`) — note a leading `<` or `;` in a text field
  needs `--form-string`, not `-F`.
