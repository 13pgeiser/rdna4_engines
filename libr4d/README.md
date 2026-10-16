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
| `gemm_mxfp4a8_decode` | the same operands for the decode band (M ≤ 128): split-K across workgroups with a last-arriver reduction, the partial in the op's scratch |
| `gemm_mxfp4a8_tiled` | the prefill sibling: the activation re-tiled into fragment order in the op's scratch and read straight into registers, the weight unpacked once per K slab for a 256-row block |
| `attn_decode_h128_gqa4_fp8kv` / `..._bf16kv` | head_dim 128, 4 queries per KV head, sliding window: a block-diffusion drafter's geometry. Decode only; there is no prefill row at this geometry |
| `rowtopk_bf16` / `rowtopk_bf16_chunks` | exact per-row top-K over a bf16 matrix, two-stage |
| `dflash_conv_t2_g16_bf16` | the drafter's causal conv, two taps, 16 groups |
| `dflash_select_bf16` | the drafter's candidate selector: edge scores and the greedy path walk |
| `dequant_w4_bf16` | unpacks a 4-bit fragment-ordered weight back to bf16 |
| `lora_shrink` / `lora_expand` | runtime LoRA's two halves beside a trunk GEMM, each row's adapter read from a per-row slot on the device (base rows untouched). Tiled rows (preferred): `t = x A[slot]^T` WMMA bf16 over 16-row x 16-j tiles with a K split fixed by K and R (`lora_shrink_wmma`, `lora_shrink_wmma_q` for fp8/i8 codes), `y += scale ⊙ (t B[slot]^T)` over 32-row x 128-column tiles with B[slot] staged in LDS, libref's bytes (`lora_expand_tiled`); a tile of base rows leaves on one ballot, a mixed tile runs one pass a slot. Scalar v1 rows (`_scalar`): one block a row / one thread an output, and the f16/f32-activation shrink. A row's bytes are independent of M in every row |
| `quant_act_i8` / `quant_act_i8_asum` | per-row int8 activation quantiser, in the r4d GEMMs' fragment byte order; the `_asum` form also emits the per-group row sums the asymmetric GEMMs need |
| `had_quant_act_i8` | the same with a block Walsh-Hadamard rotation first, which is what makes one scale per row viable |
| `rmsnorm_had_quant_i8` | residual add, RMS norm, rotate and quantise in one kernel |
| `ar_ln_had_quant_i8` | the all-reduce fused with the norm that follows it: the peer's contribution is a second operand on the norm's load, so the all-reduce's bf16 output never exists. Exact bf16 or rotated 6-bit wire (`wire`) |
| `gated_had_quant_i8` | `silu(a)*b` or `sigmoid(a)*b`, then the same rotation and quantiser |
| `gdn_gated_norm_had_quant_i8` | the GDN gated norm with the rotation and quantiser fused onto it |
| `qk_norm_rope_gate` | q/k RMS norm, rope and the attention gate in one kernel |
| `ar_oneshot_{3,4,8}rank_exact`, `ar_twoshot_{3,4,8}rank_exact`, `ar_twoshot_{3,4}rank_ti8`, `ar_gather_3rank` | the wider-world collectives (TP3/TP4/TP8), exact and int8-wire |
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
