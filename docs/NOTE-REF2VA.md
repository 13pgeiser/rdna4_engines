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
   subtly. What remains unimplemented is the pack's *keyframe* rows (`kR2VCond` / `kR2VCondAudio`) — a
   distinct exact-pin mechanism for FL2VA-style start/end-frame conditioning (which FL2VA itself
   achieves with two image references — see §2 of the continuity brief). Seam frame-equality is the
   wrong metric to gate on: the brief states it is imperfect **by construction**, because the first new
   frame is the tail's temporal *successor*, not a replay of its last frame.

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
   encoder (a per-tensor golden), not *whether* the encoders are live. That golden is **environment-
   blocked here**: this host has no `torch`, and every golden in the tree is a torch-baked `.h`, so no
   value reference for the VAE / audio-VAE encoders can be produced locally. The genuine reference
   sources are present (`MiniMax-H3-ref2va-src`, `MiniMax-H3/vae`, `MiniMax-H3/audio_vae`); generating a
   value golden needs a torch-equipped box. Content-control + determinism + a bit-exact pack (see the
   P3 pack note) is the strongest parity evidence obtainable on this machine; a published per-tensor
   match is left open so that a change is never made merely to make latents *differ* when they could be
   made *wrong*.

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
   (and, for the encoder values specifically, is environment-blocked on a torch-less host — point 4).
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
