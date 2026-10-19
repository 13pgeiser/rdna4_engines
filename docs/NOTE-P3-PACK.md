# P3/P4 port note — the exact Ref2VA packing, from ComfyUI source

Read directly from `/home/dustin/AI/ComfyUI/comfy/ldm/minimax/model.py` (2026-10 install) so the
Radiance port is source-faithful rather than doc-derived. Line numbers below are that file's.
Everything here is **pure host arithmetic** — the port is a new TU (proposed
`arch/minimaxh3/rad_ref2va_pack.{h,cpp}`, device-free, unit-testable) that emits the segment
table + per-row (t,h,w) positions + update flags; consuming them is the P3 engine wiring, and
keeping the non-update rows dense is P4.

## Constants (`:30-33`)

```
FRAME_PER_TOKEN  = (1, 4, 4, 4, 4)     # pixel frames covered by each successive video latent
FRAME_RESCALE    = 5/3                 # pixel frames -> audio-latent units (40/24 fps ratio)
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP  = 1.0
```

## Row shapes

- video rows: patchify (1,2,2) → `latent_t * (h/2) * (w/2)` rows; time grid per latent frame k is
  `origin + FRAME_RESCALE * Σ FRAME_PER_TOKEN[k'%5], k'<k` (`_video_t_grid` `:101-104`).
- audio rows: `pack_audio` `:59-62` — `[C=32, ch=2, T]` → `[2T, 32]` **channel-major**: ch0 all
  T frames then ch1 all T frames. `rt` audio latent frames ⇒ `rt*2` rows.
- spatial frame grid `_frame_grid(h,w)` `:94-99`: `area=sqrt(h*w)`; per axis
  `linspace`-like `(i*(ratio/n) + (1-ratio)/2)*32` with `ratio=dim/area`, `n=dim/2` — area-
  normalized h/w coords of the 2×2 patch rows (h rows major). Audio grid `_audio_grid` `:106-113`:
  `t = cursor + arange(T)` repeated for both channels, `w` pinned per channel to the **target**
  grid's first/last w coord (`target_audio_w = (w_grid[0], w_grid[-1])`), `h=0`.
- `_ref_t_span` `:115-122`: image `1.0`; audio `ref_audio_t`; video/video_audio
  `max(ref_audio_t, Σ_{k<latent_t} FRAME_RESCALE*FRAME_PER_TOKEN[k%5])`.

## Segment order (`PackedLayout.__init__` `:351-471`)

`[text | (keyframe cond | cond_audio)* | per-ref blocks in request order | audio | video]`

- `cursor = text_len + Σ _ref_t_span(all refs)` — the **target timeline origin**; computed in a
  first pass before keyframes (`:364-366`). Keyframes pack right after text, each at
  `cond_t = cursor + FRAME_RESCALE*resolved_frame_index` (frame 0 ⇒ shares the target origin and
  overlaps its head in RoPE time — continuity comes from post-sampling trim, not placement).
- ref block per kind (`:394-441`), all rows **update=False** (pinned, never denoised):
  - `image`: `ref_img` n=`(h/2)*(w/2)` rows, t=cursor, then `cursor += 1.0`
  - `audio`: `ref_audio` `rt*2` rows (only if rt>0) at `_audio_grid(cursor,...)`, then
    `cursor += rt`
  - `video`/`video_audio`: **its `ref_audio` rows pack immediately BEFORE its `ref_img` rows**,
    both at the same cursor origin; ref_audio uses the **ref's own** w-grid extremes
    `(r_w_grid[0], r_w_grid[-1])` (target extremes only for target/cond audio); then
    `cursor += max(rt, Σ video_t_spans(latent_t))` (== `_ref_t_span`, the second pass re-cums).
- targets (`:443-454`): `audio` `audio_t*2` rows then `video` `latent_t*(h/2)*(w/2)` rows, both
  update=True, both at cursor. **Always the last two segments** — the sampler state is exactly
  these (update mask = the sampler's x).

## Timesteps & mod rows (`_forward` `:667-712`)

- `t_v = 1 - sigma_v`, `t_a = 1 - time_shift_sigma(sigma_v, shift_v, shift_a)` (audio sigma
  derived from video sigma via `time_shift_sigma` `:36-39`).
- per-kind timestep labels: `text/video → t_v`, `audio → t_a`,
  `cond|ref_img → max(t_v, 0.999)`, `cond_audio|ref_audio → max(t_a, 1.0)` (pinned near-clean).
- `unique_t` = sorted distinct labels actually present; `t_row[t]` = index.
- mod tags: `{text:1, video:0, audio:2, cond:0, ref_img:0, cond_audio:2, ref_audio:2}`;
  **mod-row index = `t_row(label) * 3 + tag`** — refs ride the video/audio modulation streams.
- denoise masks (`:683-699`): per-row `m` puts a row at `sigma = m*sigma_stream`, label
  `clamp(1-m*sigma, max=pin)`; all-equal collapses to the segment label. Radiance's tail uses no
  partial mask (full rows) — port masks only if partial-row conditioning ever lands.

## Cond-row content + the 0.1% augmentation (`:532-558`)

Cond rows are rebuilt **every step** from stored clean latents (never from sampler state):
`r = aug*z + (1-aug)*randn` with a **CPU RNG re-seeded per step per block-kind**: visual block
seed = run seed, audio block seed = run seed **+ 1**, `aug` = 0.999/1.0. Re-seeded ⇒ identical
noise every step; but it must be the **same stream every step** — port with a seeded CPU PRNG
matching torch's `randn` only if bit-parity is demanded (else document deviation: same-shape
deterministic-noise with Radiance's Philox, parity gate = "cond rows never change across steps").

## Sampler contract (`forward` `:603-643`)

- Sampler x = `[video_latents, audio_latents]` — **targets only** (update rows); the net returns
  velocity for those rows only. Pinned rows are input-only context.
- audio scale-carry (`:605-614`, `:637-641`): sampler carries audio as `(sigma_a/sigma_v)*x_a`;
  undo before the net, re-scale velocity out: `out_a = (1-scale)*(carry*x_a) + (1+(scale-1)*sigma_a)*v_a`.
- guidance-distilled: CFG 1.0; sigma-shift per stream (`MiniMaxH3SigmaShift` node).

## P4 — dense sinks

Sparsity (sol-attn) hooks the layout out-of-band via `transformer_options["minimax_h3_layout"]`
(vendored model code is fully dense, `optimized_attention(..., mask=None)` `:200`); the sparse
node keeps **cond+ref+audio rows** exact (`sink_conditioning=exact_kv_and_rows`) and sparsifies
the update rows only. Radiance parity: the sink rule takes `img_update/audio_update` (P3 output) —
dense set = complement of update set ∪ text — NOT a static prompt-span rule. Keep the split
(pack emits the table; sparsity is the attention backend's concern).

## Landed 2026-10-19 — `arch/minimaxh3/rad_ref2va_pack.{h,cpp}`

The port exists and is tested against the reference's own output. `ref2va_build()` emits the
segment table (start, end, kind, update), the planar `[3, rows]` f32 positions (the exact shape
R_ROPE holds, so staging is a copy), and `ref2va_mod_rows()` does the per-step `t_row * 3 + tag`
numbering (sorted distinct labels, exact-equality collapse when the live timestep passes a pin).
The test binary `tests/ref2va_pack_test.cpp` compiles the TU directly (no rad_core, no device);
`tests/data/ref2va/golden.json` is generated by `gen_golden.py <ComfyUI>/comfy/ldm/minimax/model.py`,
which AST-extracts the reference's constants + grid helpers + `PackedLayout` and execs them on
torch alone — every expectation is the reference's own float64 output. The positions compare
**bit-exact** after the float cast: both sides run identical IEEE double arithmetic, so a ULP
drift means the port reordered its arithmetic, and the test says so. Five golden cases (t2va,
fl2va keyframes at the origin and mid-timeline, all four reference kinds with a silent target, a
realistic chain signature, empty prompt) x three timestep probes (labels apart, collapsed past
both pins, far apart).

Parity with the plain generator: with no refs the pack reduces to `[text | audio | video]` —
same segment order, same row counts, the same planar shape — and `positions()` is now *this
code*: `minimaxh3.cpp` calls `ref2va_build(text_len, geom, {}, {}, &pack)` and copies
`pack.pos` (the first consumption point landed, 2026-10-19). The row contract the hand-rolled
version computed — text counting up, the soundtrack channel-major at the width extremes, the
1,4,4,4,4 time pattern — is pinned in `tests/ref2va_pack_test.cpp`
(`pack_plain_matches_the_three_segment_order`). One honest byte-level footnote: the old
`positions()` recovered the axis step as `((1-ratio)/2 + ratio) - (1-ratio)/2`, and on the grid
row whose coordinate is mathematically 0 the cancellation came out *exactly* 0, where the
reference's `arange*ratio/n` form leaves a float64 residue of ~1e-15. A sweep of all 1242 legal
sizes (multiples of 32, >= 256 a side, area <= 1344x768, aspect <= 4) found exactly 10 axes on
10 sizes where the f32 coordinate differed — one centre row each, the last bit of one rope
angle. The reference's form is authoritative, and the pack implements it verbatim.

## P4 sinks, concretely

`libr4d/r4d_vsa.hip`'s `vsa_geom()` makes **every segment before the video grid a dense 64-row
prefix** (the visibility walk then lets every query tile see all prefix tiles). The pack's order —
keyframes and reference blocks between the prompt and the targets — therefore makes every pinned
row an automatic sink the moment the driver lists it in `seg_cu` ahead of the video segment: no
kernel change. Two footnotes for the wiring: (a) ComfyUI's sparse node sparsifies the *target
audio* rows too, while Radiance's prefix rule keeps them dense — parity watch for P6, cheap to
split if the benchmark shows it matters; (b) `R_SEG`'s staged `seg_cu` is 4 i32 today and a full
presentation needs about two dozen (text, keyframe pairs, up to two segments per reference, the
two targets), so the register's
capacity grows with the wiring, along with `vsa_geom`'s `nd`.

## What the engine wiring still owes (P3 proper)

The pack answers "where do the rows sit"; consuming it touches four places the map found, of
which two have landed:

- **`positions()` now *is* the pack** (`ref2va_build` with no refs, copied straight into R_ROPE's
  buffer — landed 2026-10-19, drift audit above).
- **`seg_cu` stages the pack's table** (landed 2026-10-19): a new *optional* plugin export,
  `rad_arch_video_segs(g, cu, cap)` (documented in `rad_builder.h` — an additive export,
  `RAD_ABI_VERSION` does not move; threaded through `loader.cpp` → `startup.{h,cpp}` →
  `VideoDriver::init`), which for the plain geometry writes exactly `[0, n_text,
  n_text + 2·n_audio, rows]`. `stage_geom` prefers it and **validates**: a plugin that exports
  nothing, a table that disagrees with the geometry, or more segments than `R_SEG`'s four
  int32s — any of those stages the driver's own table instead, byte-identical.
  `video_gen_test`'s `the_plugin_segment_table_is_staged_and_a_disagreeing_one_falls_back`
  drives all three. When refs arrive, `rows` itself comes from the pack, the local table
  disappears, and `R_SEG`'s carve (`carve()`'s `4 * 4` — see P4 footnote (b)) grows with the
  wiring.

What remains: `step_dit`'s row assembly (scatter pinned latents per `img_pos`/`audio_pos` —
rebuild the cond rows from the clean latents each step with the re-seeded 0.1% augmentation);
the modulation table (today `[lane*S, L*3*Ws]` = two timesteps x three tags, and the pack can
present up to four distinct labels x three tags = twelve columns, with the pinned labels
recomputed from `t_v`/`t_a` each step). The sampler already owns exactly the update rows once
they are last — no sampler change.

## Golden test vectors (hand-check against the formulas)

- tail 3.0 s: 73 px frames ⇒ `FRAME_RESCALE*73 = 121.67` audio units ≈ 121-122 rows/channel;
  latent grid legal snaps {5,22,39,56,73}px ↔ {2,7,12,17,22} latent (`continuation_nodes.py:180-189`).
- video latent L frames span `5/3*(1+4(L-1))` audio units: L=22 ⇒ 141.67 (> its 121 audio span ⇒
  the `max()` picks the video span).
- stereo ref audio rt=60 ⇒ 120 rows; rows 0-59 ch0 (w=low), 60-119 ch1 (w=high).
- image ref 1024×1024 target grid h=w=64 patch rows ⇒ n=4096? (h/2·w/2 — the ref's OWN latent_h/w:
  `_frame_grid(latent_h, latent_w)` of the ref, not the target's).

## Landing plan for the rest of P3 (2026-10-19, after the encode device half)

Everything below was scouted against the tree, not from memory; every cited line is as of the
P2-device-half build. Increments are ordered so each one builds+tests green on its own; none
removes the submit-time refusals until the whole path works end-to-end (phased acceptance).

### 13a — ref-aware geometry (host, no behavior change, testable with fakes)
- **ABI trailing block** on `RadVideoGenBatch` (ABI 23 is unreleased — no bump): `const int32_t*
  ref_tab` — NULL when the request carries no VAE references; else self-describing
  `[0]=n_refs, [1]=n_keys`, then `n_refs × 6` words
  `{kind(Ref2VARefKind), slot(1-based within kind), lat_h, lat_w, latent_t, audio_t}`, then
  `n_keys × 3` `{frame, video_t, audio_t}`. HOST-pinned (step_dit reads it host-side exactly as
  it reads `V->geom` by value — no device staging of the table itself).
- **New optional export** `rad_arch_video_pack(g, n_text, ref_tab, seg, cap_seg, pos, cap_pos)` →
  rows (negative = refusal; `seg`/`pos` NULL → count only). Thin wrapper over `ref2va_build` —
  the plugin stays the authority (segs_ precedent). Driver threads it beside `segs_`
  (`loader.cpp` dlsym → `startup.{h,cpp}` → `VideoDriver::init(PackFn, ...)`, defaulted).
- **Carve widen**: `R_SEG` 4→**64** i32 rows (worst pack: text + ≤15 ref segs + 2 targets +
  keyframes headroom). Kernel side needs NOTHING: `vsa_geom` loops `n_seg-1` generically
  (r4d_vsa.hip:66-101) and the launcher takes `n_seg = numel(segs)-1`, cap 4096
  (r4d_vsa.hip:1009-1011). `b_modg`/`b_sel` tile caps: `P = min(P, tiles_cap)` already clamps.
- `stage_geom` stages the pack's full boundary table when `ref_tab` is set (validation flips:
  the local 4-boundary table is only reconstructible for no-refs requests).
- `video_gen_test`: fake `PackFn` cases mirroring the segs_ trio (staged / disagreeing falls
  back / short cap refuses) + `ref2va_pack_test` needs no change (pack unchanged).

### 13b — step_dit consumes the pack (arch; green = compiles + selftest + fakes)
- Rebuild the pack HOST-side inside `step_dit` from `ref_tab` (the TU is in the plugin — the
  driver staged from the same code via 13a's export, so segment starts agree by construction).
- **Row assembly** (minimaxh3.cpp:2939-2944 today): `op_text` already reads `seg[0..1]` from
  `seg_cu`; replace the fixed `nt`/`nt+arows` offsets of `op_ain`/`op_vin` with the pack's audio
  and video segment starts. Then per img-latent segment (cond/ref_img, ≤12 issues) issue
  `op_vin` with `x = brow_slice(b_evc, venc_slot_row(kind,slot), seg_rows, kPatchVals)` — the
  encoder's fold already emits patch-folded rows, so **no gather kernel**: slot slices feed the
  embedder directly. `op_ain` likewise over `b_enca` slices. Target rows unchanged.
- **0.1% re-seeded aug** per pinned img/audio segment, rebuilt from CLEAN slot rows every step
  (never in place into `b_evc`/`b_enca`): draw into scratch (rnd-op form, seed staged per step —
  `build_step` already stages `R_SEED` per pass: bake step in), scale 0.001 via `chan_affine`,
  `add` slot+noise → scratch, embed from scratch. Seeds: visual = seed words 0/1, audio = 2/3
  (already the split in `stage_seed`).
- **Mod table**: `b_ada` per lane-step grows `kSegs`→**12 columns** (≤4 distinct labels × 3
  tags): temb rows 2·S→2·S+2 (two constant pinned timesteps 0.999 video / 1.0 audio —
  `ref2va_t_label` gives the labels; `ref2va_mod_rows` gives each segment's `t_row·3+tag`).
  `step_ada`'s block×segment loop iterates the label cells; `step_dit`'s `mod()`/`modh()` view
  column `(l, mod_row)` instead of `(l, e)`, and per-segment blocks take their mod row from the
  rebuilt pack. TP2 half-split: `cut` moves to the VIDEO segment's middle row and `segh[1]=cu`
  still stands only when the second half is video-only — with refs keep the half inside the
  video segment (`hr0[1] = vseg.start + vrows/2`), and half 0's mod view takes the segment rows
  for its range; `ref2va_mod_rows` order pins which cells it needs.
- Heads/Euler: the two target segments stay LAST, so the heads and both Euler steps keep working
  with the views re-anchored at `aseg.start`/`vseg.start` — sampler state = update rows only.
- `b_sel`/`b_coarse` sizing: prefix tiles P grow with ref rows (each ≤64-row chunk adds a
  tile); declare capacities already take the probe's widened `gen_rows`; re-check
  `tiles_cap` math against a worst-case 9/3/3 presentation.

### 13c — driver media decode + encode scheduling (FFmpeg-gated for real bytes)
- Decode in the driver (bytes already in `VideoRefMedia::data`, in-memory `core/mm/mp4.cpp`
  path): frames → f32 0..255 + clip repeat-pad to entry 23; samples → channel-major f32
  right-padded to entry 0; ref sizes capped by `rad_arch_encode_geom` BEFORE decoding fully
  (probe the export, bound the decode). Images: policy = resize to ≤256 side (the encode
  ceiling) for the VAE view; the TE view keeps `media`'s own pixels.
- New job phase before cond: per ref, one V_ENCODE/A_ENCODE pass (order: tail, images, videos,
  vaudios, audios — tail only for P2b requests); `next()` slot in the pipeline before
  `build_cond`, gated on `decl_.encodes`; refusal removal is HERE (13c), not 13a/13b.
- P2b tail: `RAD_ENC_TAIL` slot 0 rides V_ENCODE from the ClipStore's sampled latents — but the
  tail is ALREADY latents, so the tail path issues no V_ENCODE; it scatters the slice straight
  (see NOTE-CONTINUITY); only its AUDIO tail needs A_ENCODE? — decide when P2b starts.
