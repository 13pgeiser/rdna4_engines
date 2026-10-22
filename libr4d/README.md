# libr4d

libr4d is radiance's kernel library for RDNA4 (gfx12, developed on gfx1201, the Radeon AI PRO
R9700): specialised HIP kernels behind a plain C ABI ([`r4d.h`](r4d.h)) and the kernel-plugin rows
the engine selects from (`r4d_rows.cpp`). `libr4d/CMakeLists.txt` names every translation unit and
the per-unit flags each needs.

## Naming

An entry point is

```
r4d_<family>_<op>_<the geometry it is compiled for>
```

Every dimension in the name is a compile-time constant, and the entry point rejects a mismatch
instead of running. `attn_decode_h256_gqa6_fp8kv` serves head_dim 256 with 6 queries per KV head and
an fp8-e4m3 paged cache, and nothing else. A model with another head size needs a new
instantiation under its own name, so no existing name changes and no caller can pick up a kernel
that does not fit.

Dimensions that are the same across the whole library stay out of the names: the paged block size
(16), the query dtype (bf16) and the target architecture. The `r4d_*_dims()` accessors in
[`r4d.h`](r4d.h) and the `RadKernelInfo` rows in `r4d_rows.cpp` report them.

The table lists a selection; `r4d_rows.cpp` and `r4d.h` hold the full set.

| entry point | what it is |
| --- | --- |
| `attn_prefill_h256_gqa6_fp8kv` / `..._bf16kv` | paged causal attention, query-tiled; head_dim 256, 6 queries per KV head, fp8-e4m3 or bf16 KV cache |
| `attn_decode_h256_gqa6_fp8kv` / `..._bf16kv` | the same geometry, split-KV, `q_len * gqa` ≤ 64 |
| `attn_decode_h256_gqa6_scratch_bytes` / `attn_decode_h128_gqa4_scratch_bytes` | bytes of split-KV scratch a launch needs; ask before allocating, since the split count is chosen from the geometry |
| `attn_vit_h72_bf16` | dense non-causal vision-encoder attention, head_dim 72 native, varlen over images |
| `gdn_conv_prep_w4_h128_bf16` | the GDN prefill preamble in one kernel: causal conv (width 4, silu, with its state cache), q/k/v split, qk l2norm, gating and the per-chunk gate cumsum |
| `gdn_kkt_solve_k128_c64_bf16` | `A = (I + strict_lower(diag(beta) K K^T e^dg))^-1` per chunk of 64: the gram and its triangular inverse, without the fp32 gram reaching memory |
| `gdn_chunk_scan_k128_v128_c64_bf16` | gated-delta-net chunked scan: WY recompute, state recurrence and output in one kernel; head_k 128, head_v 128, chunk 64 |
| `gdn_conv_update_w4_h128_bf16` | the same convolution for a decode step: a rolling speculative window over the conv state cache, writing q/k/v directly |
| `gdn_recurrent_update_k128_v128_bf16_fp32state` | the decode-time recurrent delta-rule update against the paged fp32 state cache, gating and qk l2norm included |
| `gdn_gated_rmsnorm_h128_bf16` | gated RMS norm over a 128-channel row, `out = rms(x) . w . act(z)` |
| `ar_oneshot_2rank_exact` | one-shot push all-reduce over P2P, exactly 2 ranks, exact bf16/fp16/fp32 |
| `ar_oneshot_2rank_wht6` | the same handshake with a Walsh-Hadamard-rotated 6-bit wire payload (lossy) |
| `gemm_bf16_nt_m16` | the M ≤ 16 band (the MoE router gate), forwarded to `gemm_bf16_nt_m64`; no kernel of its own |
| `gemm_bf16_nt_m64` | skinny bf16 GEMM, `C[M,N] = A[M,K] @ W[N,K]^T`, one 16x16x16 WMMA per row tile per 16 of K, split-K; any M, with `gemm_bf16_nt_tiled` taking M > 64 where N ≥ 128 |
| `gemm_w4a16_nt_m64` | the same GEMM with a 4-bit weight: f16 scale and integer zero per 128 K, pre-permuted offline into WMMA fragment order so the weight path is one `global_load_b128` per lane per four k steps; f16 activations, bf16 out |
| `gemm_w4a8_nt_m64` | the 4-bit weight against an int8 activation through `v_wmma_i32_16x16x16_iu8`, twice the matrix rate of the f16 form. Symmetric scale per 128 K; the skinny decode kernel, M ≤ 64, tunable over `wv`, `sk`, `mb`, `npw`, `nt` |
| `gemm_w4a8_asym_nt_m64` | the same with an asymmetric grid: unsigned B fragment, per-group zero from the scale dword's high half, minus the activation's per-group row sums |
| `gemm_w4a8_tiled` | the large-M sibling: block tile 128x256 by default (a `tile` axis selects other compiled arms), BK = 64 double-buffered in LDS |
| `gemm_w4a8_asym_tiled` | the asymmetric grid at the tiled shape |
| `gemm_w4a8_prefill` | a second large-M W4A8 kernel for the symmetric grid, ranked above `gemm_w4a8_tiled` |
| `gemm_w8a8_nt_m64` | int8 weight against int8 activation, skinny; the promoted (8-bit) linears of a mixed-precision artifact |
| `gemm_w8a8_tiled` | the prefill sibling of the above |
| `gemm_w8a16_nt_m64` | 8-bit weight, f16 activation, skinny |
| `gemm_w2a8_nt` | 2-bit weight against int8 activation: the DFlash2 draft head |
| `gemm_mxfp4a8_nt_m64` | OCP-MXFP4 weight, fp8 activation, skinny |
| `gemm_mxfp4a8_gemv` | the same operands for one to eight rows: the weight streamed straight to registers, no LDS staging; WMMA when K is a multiple of 128, `v_dot4` otherwise, chosen from K alone so a row's bits do not depend on M |
| `gemm_mxfp4a8_gemv_wide` | its WMMA form at 9..64 rows: up to four 16-row tiles a wave over each weight word, two 16-column tiles a block on wide, deep weights |
| `gemm_mxfp4a8_decode` | the same operands for the decode band (M ≤ 128): split-K across workgroups with a last-arriver reduction, the partial in the op's scratch |
| `gemm_mxfp4a8_decode_shortk` | the decode kernel at 33..64 rows of a wide, short weight (N ≥ 8192, K ≤ 2560), above `gemm_mxfp4a8_gemv_wide`, which loses there |
| `gemm_mxfp4a8_tiled` | the prefill sibling: the activation re-tiled into fragment order in the op's scratch and read straight into registers, the weight unpacked once per K slab for a 256-row block |
| `attn_decode_h128_gqa4_fp8kv` / `..._bf16kv` | head_dim 128, 4 queries per KV head, sliding window: a block-diffusion drafter's geometry. Decode only; there is no prefill row at this geometry |
| `rowtopk_bf16` / `rowtopk_bf16_chunks` | exact per-row top-K over a bf16 matrix, two-stage |
| `dflash_conv_t2_g16_bf16` | the drafter's causal conv, two taps, 16 groups |
| `dflash_select_bf16` | the drafter's candidate selector: edge scores and the greedy path walk |
| `dequant_w4_bf16` | unpacks a 4-bit fragment-ordered weight back to bf16 |
| `lora_shrink` / `lora_expand` | runtime LoRA's two halves beside a trunk GEMM, each row's adapter read from a per-row slot on the device (base rows untouched). Tiled rows (preferred): `t = x A[slot]^T` WMMA bf16 over 16-row x 16-j tiles with a K split fixed by K and R (`lora_shrink_wmma`, `lora_shrink_wmma_q` for fp8/i8 codes), `y += scale ⊙ (t B[slot]^T)` over 32-row x 128-column tiles with B[slot] staged in LDS, libref's bytes (`lora_expand_tiled`); a tile of base rows leaves on one ballot, a mixed tile runs one pass a slot. Scalar rows (`_scalar`): one block a row / one thread an output, and the f16/f32-activation shrink. A row's bytes are independent of M in every row |
| `quant_act_i8` / `quant_act_i8_asum` | per-row int8 activation quantiser, in the r4d GEMMs' fragment byte order; the `_asum` form also emits the per-group row sums the asymmetric GEMMs need |
| `had_quant_act_i8` | the same with a block Walsh-Hadamard rotation first, which is what makes one scale per row viable |
| `rmsnorm_had_quant_i8` | residual add, RMS norm, rotate and quantise in one kernel |
| `ar_ln_had_quant_i8` | the all-reduce fused with the norm that follows it: the peer's contribution is a second operand on the norm's load, so the all-reduce's bf16 output never exists. Exact bf16 or rotated 6-bit wire (`wire`) |
| `gated_had_quant_i8` | `silu(a)*b` or `sigmoid(a)*b`, then the same rotation and quantiser |
| `gdn_gated_norm_had_quant_i8` | the GDN gated norm with the rotation and quantiser fused onto it |
| `qk_norm_rope_gate` | q/k RMS norm, rope and the attention gate in one kernel |
| `ar_oneshot_{3,4,8}rank_exact`, `ar_twoshot_{3,4,8}rank_exact`, `ar_twoshot_{3,4}rank_ti8`, `ar_gather_3rank` | the wider-world collectives (TP3/TP4/TP8), exact and int8-wire |
| `ar_twoshot_{3,4,8}rank_exact_lane`, `ar_twoshot_{3,4}rank_ti8_lane` | the same two-shots at `overlap` 1, beside compute: the wide rows' entry points, the same bits; at four ranks a narrow launch (4 blocks of 1024 threads), at three and eight the wide launch unless `R4D_AR_LANE_3_8=1` gives them the narrow one (unmeasured on such cards) |
| `p2p_copy2d`, `p2p_copy2d_kernel`, `p2p_fence` | strided peer device-to-device copy, through the copy engines or a kernel |

## Constraints as data

A geometry is stated in the entry point's name and in the check that rejects a mismatch. A third
copy in each caller would drift, so the constraints are data in the rows the engine reads.

`r4d_rows.cpp` carries one `RadKernelInfo` row per kernel, its geometry written as `RadConstraint`s
as well as in prose, and the engine's selector reads those rows to choose the kernel for a band; a
caller never needs to know a geometry to ask for one. `rad-kbench` tests the rows against the
reference kernels, and `r4d_selftest` calls every launch with a minimal valid input, which catches
a transcription error in a constraint.

The kernels themselves are plain HIP behind the C ABI in [`r4d.h`](r4d.h) and know nothing of the
rows.

## Building

libr4d builds inside radiance's tree (see `build.sh`) or on its own against an installed radiance
(`cmake -S libr4d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`). It is skipped when the build
targets no gfx12 architecture.

Translation units are compiled separately and linked, so per-kernel flags stay scoped: `-mcumode`
applies to `r4d_gdn_chunk_scan_k128_v128_c64_bf16` alone. `-ffp-contract=off` applies to every unit
because the quantised all-reduce requires it: otherwise the two ranks fuse different products and
diverge by about 1 ULP.

## Adding a kernel

1. Write the `.hip`; name the entry point for what it computes and the geometry it is compiled for.
2. Declare it in `r4d.h`, and add a `_dims()` accessor if it has compiled-in geometry a caller
   should be able to read.
3. Name the file after the entry point, geometry included. Only shared machinery that provides no
   kernel (`r4d_common.h`, `r4d_dt16.h`, `r4d_gdn_wmma.h`) is named for what it is.
4. Add the translation unit to `R4D_UNITS` in `libr4d/CMakeLists.txt`, with any per-unit flags.
5. Add a `RadKernelInfo` row in `libr4d/r4d_rows.cpp` whose launch is the
   `(const RadArgs*, RadStream)` entry beside the kernel. The selector reads only these rows, so a
   kernel without one cannot be resolved. Give it the `op` it implements (rows may share one) and
   write its geometry as constraints as well as prose.
6. Add a case to `libr4d/r4d_selftest.cpp` so the row is called at least once; a row without one
   reports `no smoke case`.

## The OCP-MXFP4 GEMM by row count

`gemm_nt_q` at `mxfp4a8` has six rows over one stored weight (the skinny kernel's fragment-order
codes, `[K/32][N]` E8M0 exponents and a per-row reference exponent), so the selector moves between
them by M with no second copy. Each needs N a multiple of 16 and K of 64; of those whose
constraints hold, the highest priority serves:

| row | band | priority |
| --- | --- | --- |
| `gemm_mxfp4a8_decode_shortk` | 33 ≤ M ≤ 64, 8192 ≤ N ≤ 36864, K ≤ 2560 | 59 |
| `gemm_mxfp4a8_gemv` | M ≤ 8 | 58 |
| `gemm_mxfp4a8_gemv_wide` | 9 ≤ M ≤ 64, K a multiple of 128 | 58 |
| `gemm_mxfp4a8_decode` | M ≤ 128, N ≤ 36864 | 55 |
| `gemm_mxfp4a8_nt_m64` | M ≤ 64 | 50 |
| `gemm_mxfp4a8_tiled` | any M | 30 |

One stream's decode (M = 1 + the draft depth) takes `gemv`; several streams with speculation land
in `gemv_wide` (4 × (1 + 7) = 32 rows, 16 × (1 + 3) = 64); `decode` keeps 65..128 rows, and 9..64
where K is not a multiple of 128; `tiled` takes the rest. Unless stated, times are one rank's
weight of Qwen3.8-27B or TranslateGemma 27B at TP2 on an R9700, each row alone (`r4d_selftest
--perfmxd`). Over 16 per-rank shapes of other models and card counts (Qwen3.8-27B at TP4 and TP8,
1B-, 4B- and 8B-class dense at TP1 and TP2, a 31B-class at TP4), M 1..128, the selection below
against the better of the decode and skinny rows before it: median -10.3%, mean -10.6% over 192
cells.

**`gemm_mxfp4a8_gemv`** (M ≤ 8). One stream's decode is a pure weight stream, and staging the
weight through LDS left narrow projections at a third to two thirds of DRAM bandwidth. This row
loads each weight word straight into registers and folds it to e4m3 there: WMMA from register
fragments when K is a multiple of 128, `v_dot4` otherwise. The form and the waves a block depend on
K alone, so a row's bits do not depend on M, and a verify step's first row is the plain decode of
it (`r4d_selftest` checks the first row's bits against M). The scale and reference planes must be
4-byte aligned.

**`gemm_mxfp4a8_gemv_wide`** (9..64 rows, K a multiple of 128). The WMMA form with up to four 16-row
tiles a wave: each weight word of a chunk is loaded and folded once and multiplied with every
tile's activation fragment, so the weight streams once whatever the row count. A row's sum is its
tile's WMMA chain alone, so its bits are `gemv`'s at every M (except in the `decode_shortk` corner,
where the decode kernel serves 33..64 rows). From 25 rows, where N ≥ 4096, K ≥ 3072 and the tile
count is even, a block carries two 16-column tiles: the activation is read per block, and with one
tile a 64-row activation was read N/16 times (356 MB from L2 on a 17408-row gate_up, against a 47 MB
weight). On narrower or shallower weights two tiles lost 4..37 % (fewer blocks than the part has
units, a short K a wave), so they stay at one. Against the decode row that served 9..64 rows
before, in µs: Qwen down 51-80 against 56-105, ssm_out 27-42 against 30-55, q|gate 40-63 against
44-74; TranslateGemma down 61-97 against 66-123, o 22-35 against 25-51; gate_up at 24-48 rows
110-141 against 145-170. With two tiles a block, 17408 × 5120 at 32 rows 125 → 85 µs and at 64
157 → 122, 21504 × 5376 154 → 111 and 183 → 150.

**`gemm_mxfp4a8_decode_shortk`** (33..64 rows, 8192 ≤ N ≤ 36864, K ≤ 2560). The decode row's
kernel, scratch hook and tunables at priority 59, above `gemv_wide`. On a short K a `gemv_wide`
block's three or four row tiles make its cross-wave reduction outweigh its WMMAs, and a wide N
makes many such blocks. Over 24 per-rank shapes (Qwen3.8 and TranslateGemma at TP1, 2, 4 and 8,
and 1B- to 8B-class dense) at 24..64 rows, `gemv_wide` matched or beat the previous decode kernel
everywhere but this corner, where it lost by 7..30 %; there this row wins:

| shape, 64 rows | `gemv_wide` | previous decode kernel | `decode_shortk` |
| --- | --- | --- | --- |
| 19456 × 2560 | 131.8 µs | 104.5 | 93.3 |
| 16384 × 2048 | 92.8 | 76.1 | 68.6 |
| 9728 × 2560 | 74.1 | 67.4 | 52.5 |

Narrower N stays with `gemv_wide` (5120 × 2176 at 40 rows 39.2 µs against 52.8), as do Qwen3.8-27B's
and TranslateGemma's two-rank shapes. It is a row rather than a rule inside the gemv entry so that
the choice stays a lookup over constraints, and `--debug-selection` names it.

**`gemm_mxfp4a8_decode`** (65..128 rows, and 9..64 where K is not a multiple of 128). Past 16 rows
each wave loads its weight words straight into registers, as `gemv` does (a fragment-order word is
the WMMA fragment's register), and only the shared activation is double-buffered in LDS, one
barrier a slab. The fold, k-step, slab order, scale carry and epilogue are unchanged, and the whole
output was compared byte for byte with the LDS-staged kernel's (312 cases, 17..128 rows, both scale
grids, forced splits): 17408 × 5120 at 96 rows 205 → 159 µs, at 128 245 → 214; 5120 × 8704 at 128
169 → 139; 6144 × 5120 at 128 121 → 92. The split and slab rule past 64 rows was then re-tuned for
this kernel: the 128-wide slab always, and a split only while every split keeps 16 slabs, towards a
48-block grid or, at K ≥ 8192, regardless, up to ks·M = 1024 split rows. At 80 / 96 / 128 rows:
17408 × 5120 161 / 187 / 215 → 161 / 175 / 197 µs, 5120 × 3072 71 / 79 / 67 → 55 / 60 / 69, 21504 ×
5376 192 / 207 / 228 → 187 / 200 / 217. A different split sums a row in a different order, so at
65..128 rows the re-tuned rule moves output bits within the selftest's bound: the one change in
this family that is not bit for bit. Up to 64 rows the rule is as it was.

**Tunables.** `gemv` and `gemv_wide` have none, so a `RADIANCE_TUNE` pin on `gemm_nt_q` passes
them by. `decode_shortk` shares the decode row's axes (`ks`, `bk`, `nt`; 0 the rule), so a pin of
those reaches both. A pin applies to every row serving the op, and an axis a serving row lacks
stops startup by name.

## Notes on the kernels

Three properties of the architecture are assumed throughout and not restated per kernel:

* The wave32 WMMA fragment layout is `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`. Both
  A and B want K-contiguous rows, so swapping the operands of a WMMA transposes the result.
* gfx1201 has no `v_cvt_pk_bf16_f32`, no direct-to-LDS and no `ds_read_b64_tr_b16`. Packing bf16 is
  software; the cheapest form is two adds plus a `v_perm_b32`.
* LLVM single-buffers LDS and drains before every WMMA unless told otherwise.
  `__builtin_amdgcn_sched_group_barrier` fixes it, and the granularity of the groups matters more
  than their contents: coarser is better, up to the point where the pattern asks for more
  outstanding loads than the hardware can hold.
