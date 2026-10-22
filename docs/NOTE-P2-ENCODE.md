# P2 port note — the VAE encode, from ComfyUI source

Read directly from `/home/main/AI/ComfyUI/comfy/ldm/minimax/vae.py` and `audio_vae.py`
(2026-10 install) so the Radiance port is source-faithful rather than doc-derived. Line numbers
below are those files'. §5.5 of `PORT-REF2VA.md` scoped the whole of P2 (kernels, graphs,
declare, re-convert); this note is the extraction underneath it, and the **host half has
landed**: `arch/minimaxh3/rad_vae_geom.{h,cpp}` + `tests/vae_geom_test.cpp` (bottom section).

## Video encoder — `EncoderFCN3D` (vae.py :176-228) and its blocks

Constructor arguments (`MiniMaxH3VideoVAE.__init__` :409-441): `ch=128`, `ch_mult=(1,2,2,4,4,8)`,
`space_down=(2,2,2,2,1,1)`, `time_down=(1,2,2,1,1,1)`, `num_res_blocks=2`, `z_channels=24`,
`double_z=True` (moments = 2·24 = 48 channels). Derived: `vae_ratio=16`, `vae_ratio_t=8`,
`clip_length=17`, `token_drop=3`, `tile_size=256`, `tile_overlap_min=64` — and the decode-side
`frame_pre_padding=7`, `tokens_chunk_size=3`, `token_overlap=0`, `frame_overlap=0` (the engine's
decode driver already encodes these; not re-ported).

- `CausalConv3d` (:72-118): spatial **reflect** pad `(w,w,h,h)` (or explicit `spatial_pad`),
  temporal **front-only zero pad** `2*pad_t`, stride-s conv with no further padding. The single
  exception: when the input has one frame (`x.shape[2]==1`) the physical pad is skipped
  (`front = 0`) and the temporal taps are truncated instead (`autopad="causal_zero"` :101-103) —
  semantically a front zero pad of `k_t − 1`, so T=1 in ⇒ T=1 out at every layer. The 1×1
  shortcut conv has nothing to fold (:87, pads stay off). The kitchen fused pad/GN paths
  (:41-70, :75-86) are CUDA-only fast paths; the CPU path is exactly pad → conv.
- `TemporalIsolatedGroupNorm` (:122-130): GroupNorm **statistics per frame** (time folded into
  batch), 32 groups, `eps=1e-6`, affine — libr4d's `group_norm` (OPS.md:1276) reduces over ALL
  rows, so the encode variant must fold T into N.
- `Downsample3D` (:137-152): `CausalConv3d` k3 `padding=(1,0,0)` stride `(time_stride, s, s)`,
  called with `spatial_pad=(0,1,0,1)` when s=2 — **right/bottom-only** spatial pad, not centered.
  Temporally it is exactly `L → ceil(L / time_stride)` (front pad 2, k3: `(L+2−3)/t+1`);
  spatially `(L + 0 + 1 − 3)/2 + 1 = ceil(L/2)` too, but the pad asymmetry matters for values.
- `ResnetBlock3D` (:155-173): silu→GN→conv1 k3p1; 1×1 shortcut if channels differ;
  silu→GN→conv2 k3p1 + residual. All stride 1 ⇒ **length-preserving** (taps: k3 p1 front pad 2).
- Graph (:218-228): `conv_in` k3p1 → per level (2 × ResnetBlock, then `Downsample3D` wherever
  `space_down[i]*time_down[i] > 1`) → `conv_out` with `pre_norm=norm_out` (GN+silu folded).
  Lengths: temporal `17 → ceil(17/2)=9 → ceil(9/2)=5` (only two levels time-downsample), then
  levels 3-5 stride 1 in time; spatial 16× at levels 0-3.

## Video encode — chunking, tiling, normalization

- `_normalize_pixels` (:485-486): pixels in [−1,1] → `((x+1)/2 − IMAGENET_MEAN)/IMAGENET_STD`
  with `IMAGENET_MEAN=(0.485,0.456,0.406)`, `IMAGENET_STD=(0.229,0.224,0.225)`.
  `_encode_moments` (:479-480) = `quant_conv(encoder(x))` (1×1 to 48ch).
- `encode` (:775-792): T=1 encodes alone and takes `moments[:,:,-1:]` — one latent; T≥2 goes
  through `encode_temporal` (:647-661): `ceil(T/17)` clips, the short last clip **padded by
  repeating its final frame** to 17, each clip → 5 latents, cat, **drop the last 3** ⇒
  **L(T) = 5·ceil(T/17) − 3**. Mean only — `chunk(moments.float(), 2, dim=1)[0]`, **no sampling**,
  then `(mean − LATENTS_MEAN)/LATENTS_STD` (:20-38, 24 doubles each, transcribed into
  `kVaeLatentsMean/Std`). Deterministic ⇒ golden parity against the reference is well-defined.
- `split_tiles` (:514-536), per axis, always on (per `encode_tiled`'s comment :794-796): if 256 ≥ len, one
  tile; else `N = ceil(len/256)` grown until `256N − 64(N−1) − len ≥ 0`; slack
  `remaining // 16` is spread **+16 px round-robin over the overlaps**; starts cumulate
  `256 − overlap`. Worked 1344: N=7, overlaps [80,80,80,80,64,64], starts
  [0,176,352,528,704,896,1088], last window ends exactly at 1344. Tile slices clamp at the axis
  end (`x[..., pos:pos+256]`), so the last tile can be short.
- `tiled_encode` (:563-590): per-tile `_encode_moments`; latent overlaps = overlap//16; the
  blend (:538-561) is a **linear ramp** `pos/blend_extent` over the overlap (extent clamped to
  both tiles), blending top-of-prev with top-of-current, **cutting** the trailing overlaps from
  each tile before concatenation. Net effect: the moments grid is exactly `len/16` per axis —
  verified end to end by golden runs (320→20, 512→32, 288→18, 192→12 latents).

## Audio encoder — DAC strided stack (audio_vae.py) + posterior head

- `ResidualUnit` (:155-172): Snake → conv k7 dil `d`, symmetric pad `(7−1)d/2` → Snake → conv
  k1; **length-preserving** (pad = exactly the dilation slack), residual add.
  `EncoderBlock` (:174-192): 3 ResidualUnits (dil 1,3,9) → Snake → conv `k=2s, p=ceil(s/2)`,
  stride s — exact division at lengths divisible by s (the wrapper's right pad guarantees it).
- `Encoder` (:195-209): conv k7 p3 in (length-preserving) → blocks at dims 128…2048, strides
  **(2,4,4,5,5)** → Snake → conv k3 p1 to 2048. Hop = 2·4·4·5·5 = **800** samples (40 latent
  frames/s at 32 kHz). Stereo is split to mono streams (`reshape(b*s, 1, −1)`) — the encoder is
  mono; channels rejoin only in the output reshape.
- `encode` (:427-443): **right-pad zeros to a multiple of 800** ⇒ `T = ceil(samples/800)`
  exactly (verified against the real chain at 13 lengths); `pre_block` `AttnProjection`
  (:253-269) = LayerNorm + causal 8-head attention + Linear + GeGluMlp on the **time axis as
  sequence** — no strided op, time-preserving; `mean_proj` 1×1; normalize by the stored
  `latents_mean/std`; out `[B, 32, 2, T]`. Mean only again.
- `Snake1d` (:30-40): `x + sin²(αx)/α`, per-channel α — libr4d already has `snake` (OPS.md).

## Landed 2026-10-19 — `arch/minimaxh3/rad_vae_geom.{h,cpp}` (the host half)

The sizing questions the kernels and the driver will ask, answered as a device-free TU:
`vae_latent_frames` (the clip plan incl. the T=1 case), `vae_clip_latents` (the raw causal
chain — `ceil(T/4)` for T≥2, the nested ceil collapsing), `vae_audio_latents` (the hop rule),
`vae_split_tiles` (the tile plan: starts, real clamped slice lengths, pixel overlaps), and the
verbatim constant tables (`kVaeLatentsMean/Std`, `kVaePixelMean/Std`).

`tests/vae_geom_test.cpp` compiles the TU directly (no core, no device) against
`tests/data/vae_geom/golden.json`, which `gen_golden.py` produces **by running the reference's
real encoders on CPU**: the video `EncoderFCN3D` with the reference constructor arguments plus
the wrapper's own `encode()/encode_temporal()/split_tiles()/tiled_encode()` bound to a shim
self, and the audio DAC encoder. The shims are exactly the pieces the CPU guards keep off the
fast paths (`comfy.quant_ops` without `ck`; `Conv3d.forward` accepting `autopad="causal_zero"`
as the implicit front zero pad). 408 checks, 7 cases: the latent-count lattice through
`encode()` end to end (it reproduces the continuity snaps {5,22,39,56,73} ↔ {2,7,12,17,22} as
the network's own output), the clip chain composing with the encode count, tile plans against
the reference's literal returns, the blended tile grids keeping exact size, audio latents at 13
lengths, and the constant tables bit for bit. The one claim not executed is the audio
posterior head being time-preserving — every op in it is Linear/LayerNorm/attention/MLP over
the time axis, no strided op.

## What P2 still owes (the device half)

1. **Kernels — DONE 2026-10-19** (§5.5 list, four deltas, all landed):
   - **conv3d_ndhwc** grew the `CausalConv3d` forms (PORT-REF2VA §5.15): optional
     `stride`/`stride_t`/`causal`/`pad`(`same`|`down`)/`reflect`, 3×3×3-only, defaults byte-identical
     to the shipped upscaller conv (the `GEN` template path only for non-defaults; the T=1 tap
     truncation is subsumed by the front-only causal pad). Device/libref/avx + tests all landed.
   - **The AttnProjection "reduce" resolved 2026-10-19 as ZERO new kernels** (PORT-REF2VA
     §5.14): `layernorm` ×3, `gemm_nt_bias` for the fused qkv (bias = `q_bias|zeros|v_bias` concat
     at convert), `attn_dense` causal (q/k/v take the `[tokens, heads, head_dim]` fused-slice form),
     head-mean = `pool_rows(k=8)` on the token-major output rows, and the `adaptive_avg_pool1d(256
     → 32)` is a constant block-mean matrix that FOLDS into the attention `proj` at convert
     (`W' = W_proj @ M_pool`); mlp = concat gemm + `gelu_mul` + gemm, residuals = `add`.
   - **The per-frame GroupNorm resolved 2026-10-19 as ZERO new op** (PORT-REF2VA §5.13): launch
     `group_norm` once per frame — slice `x`/`y` at frame boundaries (NDHWC frames are contiguous
     row blocks; the launcher takes `M` from `y`'s rows and honours both pitches), one shared
     `stats` scratch, `eps` 1e-6 as the parameter. The only kernel change: the encoder's level-0
     groups are 4 channels wide (128 ch / 32 groups) and the device GN kernels chunked 8 channels
     a thread — the chunk width is now a template parameter (8, or 4 when the group is 4 wide;
     narrower still refuses; the spec never promised %8, libref/avx always answered).
   - **The strided even-kernel conv1d landed 2026-10-19** (PORT-REF2VA §5.12): `conv1d` gained
     optional `stride`/`pad` (bf16 rows only, one sequence `[0, rows of x)`, any `k`, output row
     `m` reading `x[m·stride + tap·dil − pad]` with the sequence bounds zeroing — PyTorch's
     zero-padded strided conv, `k = 2·stride`/`pad = ceil(stride/2)` giving exactly `L/stride`
     rows, the y length rule checked exactly on the host in all three plugins; `r4d_selftest`
     pins the tiled and direct kernels' numerics and the refusals).
   The graph order is in the sections above; nothing here needs a new math op.
2. **Declare + re-convert**: encoder weights are absent from containers (§5.5) — declare the
   118 video-encoder + 173 audio-encoder tensors (w/norm folds via `map_wnorm`) and re-convert
   the P5 container; the recipe carries the constants above in meta already.
3. **Value parity**: golden *values* against ComfyUI encodes at real shapes on device (shape
   parity is pinned today). Deterministic encoders ⇒ strict comparison, per-tensor tolerances
   only as far as fp8 storage forces.
4. **Driver**: the encode pass enum entry, per-request ref encode, and the tail ingest — the
   last gated on the FFmpeg dev packages (`core/mm/mp4.cpp`, P2b).
