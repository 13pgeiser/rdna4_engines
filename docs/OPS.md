# The conventional op vocabulary

Op names are free strings (`spec.md` §2.3) and the core never interprets one. `libref` can only
implement ops it knows, so its op list is the conventional vocabulary: an architecture plugin that
stays inside it can run somewhere, and an op outside it has no fallback (`spec.md` §17).

This file describes that vocabulary for architecture and kernel plugin authors. The schemas
themselves are the `RadOpSchema` rows the kernel plugins declare (`libref/` for every op it
implements, `libr4d/r4d_rows.cpp` for the rest).

The tables are machine-checked. `tests/CMakeLists.txt` registers `ops_doc`, which dumps every
plugin's schemas with `rad-schemas` and compares each table row with them: op, parameters,
operands, optional markers and weight markers. By hand:

    tools/ops_doc_check.sh docs/OPS.md build/bin/rad-schemas build/radiance_home/kernels/*.so

A row with a fourth column is a proposal, not a description, and is exempt from the check. This is
the only way an op no plugin declares yet may appear here.

Write against this document, not a guess: `qk_norm_rope_gate` and `qk_norm_rope` differ
where a guess goes wrong (one takes a precomputed cos/sin table, the other `theta`).

A table or op marked **device-only** is declared by libr4d alone. It has no host fallback and no
oracle for `rad-kbench`, so an architecture that uses it runs on libr4d or not at all. Every fusion
here is such a choice.

## Conventions

- Operands are positional, in the order given. `→` separates inputs from outputs.
- `w` marks a weight operand and `wt` a weight table (see
  [Weight operands and weight tables](#weight-operands-and-weight-tables));
  `:inout` an operand that is read and written; `?` an optional operand.
- `/opt` marks an optional parameter (a kernel that does not read it ignores it), `/der` a derived
  one ([Parameters the kernel supplies](#parameters-the-kernel-supplies)), `:f64` a float
  parameter and `:str` a string one where the name does not make it obvious.
- `(range)` marks the parameter callers declare as a range (`RAD_RANGE`); it is `M` wherever a token
  count appears. At most one parameter per op is ranged, which keeps `rad_issue` to a single
  integer on the hot path.
- A tensor's leading dimension is tokens unless stated. Everything is contiguous unless an op says
  otherwise; strides are the kernel's business, not the schema's.
- Where an op takes `dtype`, it is a string naming the operand combination, so a weight format and
  an activation format share one key: `bf16`, `f16`/`fp16`, `f32`/`fp32`, `u8`, `fp8_e4m3`,
  `fp8a8`, `fp8a16`, `i8a8`, `w4`, `w4a8`, `w4a8_asym`, `w4a8h`, `w8a8`, `w4a16`, `w8a16`, `w2a8`,
  `mxfp4a8`, `w5a16`, `w4a16_asym`, `w5a8i8`, `w4a8i8_asym`, and the codebook expert forms
  `w4nla8h`, `w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h` (`docs/MOE-W4.md`).
  - `w4a8h` is `w4a8` with a `group`-wide Walsh–Hadamard along K baked into the stored weight
    (`docs/MOE-W4.md`).
  - `w4a8_asym` is the unsigned-nibble-minus-zero-point grid instead of the symmetric one.
  - `fp8a8` and `fp8a16` say whether a block-scaled E4M3 weight meets an fp8 or a bf16 activation.
  - `w5a16` and `w4a16_asym` are ParoQuant's int5 and z-lab's int4 (u5 / u4 codes, an f16 scale and
    a u8 zero point per 128 of K) against an f16 activation; `w5a8i8` and `w4a8i8_asym` are the same
    weights against an int8 activation with an f32 scale per (row, 128 of K).
  - `q_dtype` and `kv_dtype` are the same key split in two where attention reads two formats.
- Every op returns `RAD_OK` or a negative `RAD_E_*`. There is no partial success.
- **Zero rows is not a call.** An op issued over zero rows (zero along its ranged extent, or a
  `cu_seqlens` whose sequences are all empty) returns `RAD_E_SHAPE` and writes nothing. A caller
  with nothing to compute skips the issue, as the architectures' decode and prefill halves do.
  libref holds every op to this, so a host run names an unguarded issue.
- An implementation refuses what it cannot compute: an operand dtype it does not read
  (`RAD_E_DTYPE`) or an optional operand it does not implement (`RAD_E_UNSUPPORTED`). Returning
  `RAD_OK` with the answer to a different function is the one wrong response.
- libref accumulates in f32 (`libref/ref_common.h`). Its double-precision sums are part of the
  op's reference: `seg_mean`, `gram_accum`, `logit_rerank`, the normaliser of `sample_logprobs`,
  `givens_dense_quant`'s rotation, `ple_gate`'s dot, and the sums of squares of the norms in
  `hc_read`, `mtp_enter`, `ple_gate` and `qsa_block_key`.

## How a schema is written

`libref` and `libr4d` implement the same ops, and a schema they disagree on cannot be loaded: the
loader refuses the whole plugin on any one conflict, which makes all its other rows unreachable
and `rad-kbench` unrunnable. Four rules settle every case; they are why `gdn_conv_prep` and
`gdn_recurrent_update` carry long operand lists.

### The four rules

1. **Operands are the union**, and an operand only one implementation needs is optional. A schema
   that lacks an operand cannot express the call at all. An implementation ignores an optional
   operand it does not use; it cannot invent one it needs.
2. **Parameters are the union**, and a parameter only one implementation needs is optional. A
   constraint on an absent key does not hold, so a row that needs a parameter the caller did not
   state does not match.
3. **A float-valued parameter is `RAD_P_F64`**: `eps`, `scale`, `theta`, `softplus_thr`, `l2_eps`,
   `wadd`. A `"%.9g"` string would round-trip an IEEE single but is unreadable in a graph dump.
   Constraints never match on a float: a tolerance is not a predicate, and a kernel that depends on
   `eps` to that precision has a bug.
4. **An enumerated parameter is `RAD_P_STR` with named values.** `act` is `"silu"` or `"sigmoid"`,
   not `0` or `1`, so a graph dump says what runs. A plugin's launch adapter maps the name to
   whatever integer its kernel wants.

The union rule has two deliberate exceptions, both on the gated delta net: `cu` and `num_accepted`
are required ([Linear attention](#linear-attention-gated-delta-net)).

### Parameters the kernel supplies

`attn_paged.block_size`, `gdn_chunk_scan.chunk`, `gdn_kkt_solve.chunk` and `gdn_conv_prep.chunk`
(and the `block_size` of `attn_paged_qrow` and `attn_prefix_self`) are supplied by the kernel, not
the caller (`spec.md` §7.2, `RAD_DERIVED` in `rad_abi.h`). Constraints on them are skipped during
matching, and the winning kernel's `RAD_C_EQ` value is written into the resolved geometry
afterwards. Declared `RAD_REQUIRED` they would be unselectable: a kernel compiled for
`block_size == 16` could never match the query whose answer is to be read off it.

Nothing in `gdn_conv_prep`'s operands carries the chunk length, so `libref`, which has no
`RAD_C_EQ` to supply it, returns `RAD_E_SHAPE` when a caller omits `chunk` rather than treating the
whole sequence as one chunk. libr4d supplies 64.

`all_reduce.hops` is optional: a one-shot/two-shot preference the caller may state and usually
should not. `all_reduce.exact` is required: a caller that does not accept a lossy wire must get no
kernel rather than the quantised one.

### Weight operands and weight tables

- **`RAD_OPD_WEIGHT`** (`w`): the issue names one weight handle, and the kernel receives that
  weight's bytes, dtype and declared shape.
- **`RAD_OPD_WTAB`** (`wt`): the issue names the first handle of a run and the run's length, and
  the kernel receives a device array of pointers, one per entry. `RAD_WTAB(h, n)` in
  `abi/rad_runtime.h` is the issue-side spelling; `rad_decl_weight` hands out consecutive indices,
  so a declaration loop over experts produces such a run.

A table, not a base plus a stride, because a layer's experts are separate movement units
(`spec.md` §4.2): after placement one may be in VRAM, the next streamed over the link and the third
still on disk. `base + e * stride` describes only the case where nothing was offloaded. The core
resolves every handle of the run like any weight (residency table, mover event, use note) and
uploads a pointer array, patching it only when a pointer moves.

What the kernel sees: `RadTensor::data` points at the pointer array, `shape[0]` is the entry count
and `stride[0]` is 0, which marks a table. `dtype` and the trailing extents and strides are one
entry's, and every entry of a run must agree on them; the core refuses the issue by name if they
do not. A kernel may also accept the stacked form (an ordinary rank-3 tensor with a real leading
stride): `rad-kbench` replaying a fixture has nowhere to put a pointer array, and libref and libr4d
both read either form.

How the declared weights are divided: after each fixed weight position takes one weight, the rest
is split evenly across the table positions in declaration order, and where it does not divide the
first positions take one more each. So `moe_gemm_q_mod4`'s first `n_expert % 4` classes hold one
expert more than the rest, and their codes and scales are its first table positions. A list too
short to give every table an entry is refused at declare; a table whose declared run is not
contiguous is refused when the tables are bound, before the first step; entries that disagree on
shape, or an issue naming a run other than the declared one, are refused at issue.

### Weights of several planes

A weight may be declared over several planes of one stored tensor (`RadWeightDecl::planes`,
`spec.md` §4.3) and still be one operand: an fp8 or int8 lm head (codes and scale), Flash-Next's
E4M3 mixing and fc rows (codes and f32 group scales), a zero-point grid's `b_scale` (scale and
zero). A device kernel arranges such a weight in its layout hook. A host kernel reads the joined
form `abi/rad_planes.h` defines: a header naming each plane by role, then the planes as the
container holds them. libref and libavx join the same selections on the same operands and refuse
the others with `RAD_E_DTYPE`, so either may serve any band:

| op | operand | joined planes |
|---|---|---|
| logits_gemm, logit_rerank | `lm_head` / `head` | codes (E4M3 or i8, one a value) and one scale plane (bf16, f16 or f32, any block) |
| hc_read | `mix_down`, `mix_up` | the same (`mix` e4m3) |
| mtp_enter | `fc_h`, `fc_e` | the same (`fc` e4m3) |
| gemm_nt_q | `b_scale` | `scale` (f16) and `zero` (u8, u4 or u5), one each per (row, `group` of K), at the zero-point grids below |

The kernel finds the joined form by its header and its extents from the planes' entries, not the
operand's declared shape. A joined head or matrix is read element by element as the core's affine
decoder gives it (`core/format/encoding.h`): the code times the scale covering it, one f32
rounding, so libref's product over a joined head is bit-identical to its product over that
decoder's f32 head. A single plane is never joined, and the plain forms (a bf16 head, the logical
f32 head a tool describes) reach the same ops unchanged. Where a joined `hc_read` or `mtp_enter`
matrix rounds a gate, it rounds to `h`'s dtype, as a plain matrix rounds to its own.

## Loader checks

Every op a kernel row implements must have a schema declared by some loaded plugin. The check runs
once the whole hierarchy is open, because an override plugin may rely on the schema of a plugin
below it, and the refusal names the kernel, the plugin and the op. Without it a typo and an
unimplemented op give the same diagnostic.

The first plugin in hierarchy order to declare an op fixes its schema. A later plugin declaring the
same op with a different schema (parameter keys, types and requiredness; operand count, roles and
optionality) is refused at load with both plugins named; an identical declaration is expected.
Since arguments are positional, two disagreeing schemas would give one call two meanings depending
on which plugin won selection.

`build/bin/rad-schemas a.so b.so` prints what plugins declare and every disagreement between them
at once. Run it before adding a kernel plugin; the loader names only the first conflict.

---

## Elementwise and norms

| op | params | operands |
|---|---|---|
| `rmsnorm` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt `mult`:f64/opt | `x`, `w`w? → `y` (no gain operand: a gain of one) |
| `rmsnorm_p` | `M`(range) `n` `eps`:f64 `dtype` | `x` → `y` |
| `rmsnorm_add` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `residual`, `w`w → `y`, `residual_out` |
| `hc_enter` | `M`(range) `n` `hc` `dtype` | `x` → `h` |
| `hc_read` | `M`(range) `n` `hc` `lowrank` `eps`:f64 `dtype` `wadd`:f64/opt `inject`/opt `group`/opt `rotate`/opt `mix`/opt | `h`, `w`w, `mix_down`w, `mix_up`w, `inject_w`w? → `x`, `inj`?, `q`?, `scale`?, `rq`?, `rscale`? |
| `hc_write` | `M`(range) `n` `hc` `dtype` | `y`, `inj`, `h`:inout |
| `layernorm` | `M`(range) `n` `eps`:f64 `dtype` | `x`, `w`w, `b`w? → `y` |
| `grid_embed` | `M`(range) `n` `side` `dtype` | `x`:inout, `table`w, `coord` |
| `grid_embed_aa` | `M`(range) `n` `side` `dtype` | `x`:inout, `table`w, `coord` |
| `add` | `M`(range) `n` `dtype` | `a`, `b` → `y` |
| `mul` | `M`(range) `n` `dtype` | `a`, `b` → `y` |
| `silu_mul` | `M`(range) `n` `dtype` | `gate_up` → `y` (splits `[M,2n]` into gate·up) |
| `mul_weight` | `M`(range) `n` `dtype` `scalar`/opt | `x`, `w`w → `y` (w one element or one row) |
| `act_mul` | `M`(range) `n` `act` `dtype` | `a`, `b` → `y` (act(a)·b, two operands) |
| `seg_mean` | `M`(range) `n` `dtype` | `x`, `cu_seqlens` → `y` (one mean row a segment) |
| `pool_rows` | `M`(range) `n` `k` `mult`:f64/opt `dtype` | `x` → `y` (one mean row a window of k rows) |
| `relu` | `M`(range) `n` `dtype` | `x` → `y` |
| `clamp` | `M`(range) `n` `dtype` | `x`, `lo`w, `hi`w → `y` |
| `conv2d` | `M`(range) `width` `cin` `cout` `kh` `kw` `stride` `pad` `cu_div`/opt `valid_stride`/opt `dtype` `t_last`/opt `act`/opt | `x`, `w`w, `cu_seqlens`, `valid`? → `y`; `b`w? |
| `conv2d_unfold` | `M`(range) `width` `cin` `kh` `kw` `stride` `pad` `cu_div`/opt `valid_stride`/opt `t_last`/opt `dtype` | `x`, `cu_seqlens`, `valid`? → `y` |
| `conv1d_dw` | `M`(range) `n` `k` `cu_div`/opt `dtype` | `x`, `w`w, `cu_seqlens` → `y` |
| `rel_sinusoid` | `M`(range) `n` `max_timescale`:f64/opt `dtype` | → `y` |
| `gelu_mul` | `M`(range) `n` `dtype` | `gate_up` → `y` (splits `[M,2n]` into gelu·up, tanh approx) |
| `seg_sinusoid` | `M`(range) `n` `period` `max_timescale`:f64/opt `cu_div`/opt `dtype` | `cu_seqlens` → `y` |
| `gelu` / `silu` | `M`(range) `n` `dtype` | `x` → `y` |
| `cast` | `M`(range) `n` `from` `to` | `x` → `y` |
| `softmax` | `M`(range) `n` `dtype` | `x` → `y` |
| `sigmoid` | `M`(range) `n` `dtype` | `x` → `y` |
| `logit_softcap` | `M`(range) `n` `cap`:f64 `dtype` | `x` → `y` |
| `scale` | `M`(range) `n` `dtype` | `x`, `s`w → `y` |
| `silu` | `M` `n` `dtype` | `x` → `y` |
| `fill` | `M`(range) `n` `dtype` `value`:f64/opt | → `y` |

`gelu` (exact erf) and `silu` share one schema; `silu` has a row of its own because a row names one
op.

**`rmsnorm`** is `y = x * rsqrt(mean(x^2) + eps) * (w + wadd)`; `wadd` is GemmaRMSNorm's `1 +`,
added in f32. `mult` scales the normed row before its one rounding, for a residual branch weighted
by a constant (Gemma 4's audio feed-forward adds half its output). For a power of two this equals
norm-then-scale exactly, since scaling by a power of two commutes with rounding.

**`rmsnorm_p`** is `rmsnorm` with no gain: `y = x * rsqrt(mean(x^2) + eps)`. Gemma's attention
normalises the value with a bare RMS while the key takes the `k_norm` gain, and the checkpoint has
no plane to feed `rmsnorm` as a gain.

**`rmsnorm_add`** is the residual add and the next norm in one pass, as every real implementation
does it. A plugin asks for it first and falls back to `add` + `rmsnorm` (`spec.md` §2.3).

**`gelu_mul`** is `silu_mul` with the tanh GELU: Gemma's gated FFN is trained with
`gelu_pytorch_tanh`, while the unary `gelu` is the exact erf form. Both split the fused `gate_up`
with the gate in the first half.

**`add` and `mul`** broadcast a `b` of exactly `n` elements along the rows of a taller output (a
bias add). Nothing else broadcasts, in these or any other elementwise op.

**`sigmoid`**, with `mul`, gates a shared expert (`ffn_gate_inp_shexp`). Without it the shared
expert runs ungated, which is a different model.

**`mul_weight`** is a learned gain on an activation: `w` is a weight of one element (a scalar for
the whole output, Gemma 4's per-layer `layer_scalar`) or one row broadcast down the rows (a
per-channel layer scale). It is its own op because the schema fixes whether an operand is a weight
or an activation, and `mul`'s `b` is an activation. Kernels read the form off the operand;
`scalar` 1 states the one-element form so a shape description (`rad-kbench`'s) builds the right
weight.

**`act_mul`** is `silu_mul` with its halves in two operands, so the multiplier can be any tensor:
two column slices of a fused `gate_up`, or a gate and a separate per-layer input (EmbeddingGemma 2's
per-layer-embedding block multiplies `gelu_tanh(gate(x))` by one). `act` is `silu`, `gelu` (exact
erf), `gelu_tanh` or `sigmoid` (a GLU: `a` the gate half, `b` the value half). The activation is
rounded to `y`'s dtype before the product, as the unfused pair stores it.

**`seg_mean`** is mean pooling over a ragged batch: row `s` of `y` is the mean of `x`'s rows
`[cu[s], cu[s+1])`, summed in double and rounded once. `M` is the step's token count and bands the
op; the segment count is `cu_seqlens`' length less one. An empty segment is a zero row.

**`pool_rows`** is a vision pooler over windows laid out consecutively: row `r` of `y` is the mean
of `x`'s rows `[r*k, r*k + k)`, summed in f32 and rounded to `y`'s dtype, then times `mult` and
rounded again (the unfused pair stores the mean). `M` is the input row count, a multiple of `k`.

**`fill`** writes `y = value` over the extent (`value` 0 when absent). A rank issues it where it
computes nothing but a downstream reader still expects a defined buffer: an attention-zero rank's
block (`docs/TP3.md`) is one `fill` of its output wire before the all-reduce. A wire shared across
layers cannot be zeroed once a step, and a memset knows neither the row pitch nor the dtype.

**`grid_embed`** adds a learned `side x side` position table resampled to each row's grid. `coord`
is `[4, M]` (row, column, grid height, grid width); sampling is bilinear with the grid's corners on
the table's (align_corners), edge taps clamped, the four taps summed in f32 and rounded to `x`'s
dtype before the add.

**`grid_embed_aa`** is the same addition with SigLIP2's resample, torch's
`F.interpolate(mode="bilinear", align_corners=False, antialias=True)`: per axis a triangle filter on
half-pixel centres, widened by the scale when the grid is smaller than the table, its in-table taps
normalised to sum to one; separable, f32, rounded once. A different function from `grid_embed`'s,
so a different op rather than a mode a library could ignore.

### Hyper-connections

`hc_read` and `hc_write` are the gated residual pair a hyper-connection model has instead of a
pre-norm and a residual add. The stream is `hc` times the model width; every block reads a mix of
the `hc` sub-streams and writes back into all of them, both gated by data:

    hc_read     N      = grouped_rmsnorm(h)          one rsqrt per n-wide stream, one hc*n gain
                g      = sigmoid(mix_up @ silu(mix_down @ N / hc))
                x      = mean over the hc streams of g * N
                inj[s] = 2 * sigmoid(inject_w[s] . N / hc)
    hc_write    h[s]  += inj[s] * y

Three facts a second implementation must match: the mix is a mean over the streams, not a sum;
each of the two gate arguments has its own `/ hc`; and `hc_write` adds into the unnormed `h`. The
normalisation feeds the block input and the gates and never enters the residual. Writing into `N`
instead re-normalises the stream at every connection and still produces fluent text, so the error
is easy to miss.

- `inject_w` absent is the final mixer, which collapses the wide stream to one and writes nothing
  back.
- `group` absent is a read whose consumer wants bf16. Otherwise the fp8 form of `x` (`q`, `scale`)
  comes out of the same pass, quantised from the narrowed `x` rather than the accumulator, so a
  fused kernel and the reference agree bit for bit.
- `rotate` adds `rq`/`rscale`: `had_quant_act_fp8` of the same stored `x`, each group rotated by an
  unnormalised Hadamard before its absmax, for a consumer whose weight carries the rotation (the
  `w4a8h` experts).
- `mix` is the mixing matrices' stored format: `bf16`, or `e4m3` (rows of E4M3 codes with f32
  scales in each row's own tail, one per 128 columns for `mix_down` and one per quarter row for
  `mix_up`). It changes the bytes read, not the arithmetic. The matrices are replicated on every
  rank and read once per connection; on Qwen3.8-Flash-Next (96 connections) `e4m3` halves their
  1.26 GB a step. The scales in the row's tail are libr4d's arrangement; the container holds codes
  and an f32 scale plane (a group of 128 for `mix_down`, a quarter row for `mix_up`), which a host
  kernel reads joined (Weights of several planes).

### Gemma 4's audio tower

- **`relu`** is `max(x, 0)` at each operand's own row pitch.
- **`clamp`** is `min(max(x, lo), hi)` with `lo` and `hi` weights of one element each: a clipped
  linear's input or output range, stated per projection in the checkpoint.
- **`conv2d`** convolves over (time, width) with channels innermost: element `(t, w, c)` of `x` is
  at row `t`, column `w * cin + c`, and `w` is PyTorch's `[cout, cin, kh, kw]`. Each `cu_seqlens`
  segment (rows `[cu[s] / cu_div, cu[s+1] / cu_div)`) is convolved on its own with zero padding
  `pad` at its edges on both axes, so two clips in one pass never see each other, and its output
  rows start at `cu[s] / (cu_div * stride)` (a segment whose start does not divide is refused).
  With `valid`, input row `t` reads as zeros unless `valid[t * valid_stride] != 0`: the frame mask
  the subsampler applies before each convolution. f32 sum, one rounding.
- **`conv1d_dw`** is a causal depthwise convolution over time,
  `y[t, c] = sum_j w[c, 0, j] * x[t - (k - 1) + j, c]`, with rows before the segment's start
  reading as zeros.
- **`rel_sinusoid`** writes the sinusoidal relative-position table, row `d` for distance `d`:
  `sin(d * inv[i])` over the first half and `cos(d * inv[i])` over the second,
  `inv[i] = exp(i * -log(max_timescale) / max(n/2 - 1, 1))`, each step in f32 as the reference's
  torch expressions are, then rounded.

### Gemma 4 fusions (device-only)

| op | params | operands |
|---|---|---|
| `norm_add_norm` | `M`(range) `n` `eps`:f64 `dtype` | `a`, `w_a`w, `x`:inout, `w_y`w → `y` |
| `norm_add_norm_route` | `M`(range) `n` `eps`:f64 `dtype` `mult`:f64 `group` | `a`, `w_a`w, `x`:inout, `w_y`w → `y`; `w_r`w → `r`; `w_e`w → `q`, `qs` |
| `join_add_scale` | `M`(range) `n` `eps`:f64 `dtype` | `d`, `w_d`w, `r`, `w_r`w, `w_j`w, `x`:inout, `s`w, `w_n`w? → `y`? |
| `expert_scale` | `M`(range) `n_expert` `top_k` `dtype` | `ids`, `w_e`w, `ew`:inout |
| `attn_paged_qrow` | `q_len`(range) `head_dim` `gqa` `block_size`/der `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out` |

Gemma's sandwich norms, each one kernel in place of a chain of `rmsnorm`, `add` and `mul_weight`:

- `norm_add_norm` is `t = rmsnorm(a) * w_a; x += t; y = rmsnorm(x) * w_y`: a post-norm, the
  residual add and the next pre-norm.
- `join_add_scale` is `x = (x + rmsnorm(rmsnorm(d) * w_d + rmsnorm(r) * w_r) * w_j) * s`: two arms'
  post-norms, their sum, its norm, the residual add and a one-element layer scalar (Gemma 4's MoE
  feed-forward). Given `w_n` and `y` (both or neither) it also writes `y = rmsnorm(x) * w_n` over
  the new `x`: the next layer's input norm, or the final norm after the last layer.
- `norm_add_norm_route` is `norm_add_norm` plus, from the same norm of the new `x`, a MoE layer's
  two routed inputs: `r = rmsnorm(x) * w_r * mult` (`rmsnorm`'s expression with `mult`), and `q`,
  `qs` = `rmsnorm_quant_fp8` of `x` under `w_e` (E4M3 codes of the bf16-rounded norm, an f32 scale
  per (row, `group` columns) of `amax / 448`, 1 for an all-zero group, the value over it clamped to
  ±448). `n` is whole groups and `group` is 128. In Gemma 4 it replaces the router's `rmsnorm` and
  the experts' `rmsnorm_quant_fp8`.

Every value the chain stores is rounded to the activation dtype at the same point, so the fusions
differ from the chain only in the order a sum of squares is added: with the same order they are
bit-identical, and against a chain summed in another block width a norm factor can differ in its
last bit.

`expert_scale` is `ew[t, k] = ew[t, k] * w_e[ids[t, k]]` over `[M, top_k]`, `w_e` an `[n_expert]`
table and an id outside it reading 0: Gemma 4's `per_expert_scale` on the renormalised routing
weights, in place of an `embed_lookup` and a `mul`. One product rounded once, so the pair's value
bit for bit.

`attn_paged_qrow` is `attn_paged` (same parameters, operands and values) with `q` read at its own
row pitch: the q part of a fused `[q|k|v]` projection, read where it lies instead of copied into a
dense block by a `cast`. It has its own name because other `attn_paged` kernels refuse a pitched `q`
(libr4d's with `RAD_E_STRIDE`, at the first step), and under `attn_paged` the first library in
`--kernels` wins the op, an order the architecture cannot see. Asked for by name, the op resolves to
a library that reads the pitch or to nothing, and the architecture copies `q` for `attn_paged` in
the second case.

libref has no row for any of them, deliberately. An architecture asks for a fusion and declares
the chain where it does not resolve (`rad_op_resolved`); a reference row would make it resolve
everywhere, so a model loaded without the serving library would resolve to the reference and be
refused at serve, instead of running the chain on libr4d or libavx. libr4d declares them with
device rows only, so the host backend runs the chains. With no oracle, its kernels are held bit
for bit to a chain summed in their own block order by `tests/r4d_emu`. No row declares `replaces`:
`rad-kbench`'s fusion oracle runs the replaced chain on the fusing library's own rows and compares
bytes, and libr4d's `rmsnorm` adds its sums of squares in another block order.

**The Whisper-style audio encoder's** (Qwen3-ASR's, `arch/qwen3asr_bf16`), which convolves each
second of a clip apart and attends in windows of eight seconds:

- **`conv2d`'s `t_last`, `b` and `act`.** `t_last` 1 reads the weight as PyTorch's `[cout, cin, kw,
  kh]`, time last: the kernel of a model that convolves a `[mel, time]` spectrogram, whose first
  spatial axis is the one this op calls width. `b`, after `y` so a call without it keeps its five
  operands, is a bias per output channel added to the f32 sum before the one rounding; `act`
  (`none`, `gelu` exact) is then applied to the rounded value and rounded again, as
  `gemm_nt_bias`'s epilogue does. A linear over a `(width, channel)` row whose weight is stored
  channel-major is a `1 x width` convolution with `kh` 1: its `[N, cin * width]` weight is
  PyTorch's `[N, cin, 1, width]`.
- **`conv2d_unfold`** is `conv2d`'s receptive fields as rows, so a convolution with many channels
  is a GEMM. Row `(to, wo)` (output time row `to` in conv2d's numbering, output column `wo`)
  holds, for every input channel and kernel tap, the value conv2d multiplies that tap's weight by,
  and zero where the tap falls in the padding or on a row `valid` masks. Its columns are in the
  weight's order, `(ci, i, j)` for `[cout, cin, kh, kw]` and `(ci, j, i)` with `t_last`, so `y`
  times the weight read as `[cout, cin * kh * kw]` is conv2d's output (`[rows * Wo, cout]` is the
  same bytes as conv2d's `[rows, Wo * cout]`). Segments, `cu_div` and `valid` are conv2d's; `M`
  counts input time rows.
- **`seg_sinusoid`** is `rel_sinusoid`'s table at absolute positions that restart: row `r` of
  segment `s` (rows `[cu[s] / cu_div, cu[s+1] / cu_div)`) takes position `(r - cu[s] / cu_div) mod
  period`. Qwen3-ASR adds it to each second's rows, which start again at position 0. A row no
  segment covers is left untouched.
- **`attn_dense`'s `block` and `cu_div`.** `block` cuts a segment into blocks of that many rows from
  its start, the last one short, each attending only within itself: a chunked encoder's window
  (Qwen3-ASR's 104 rows, eight seconds). `cu_div` reads `cu_seqlens` in units that many rows
  apart, as the other encoder ops do, so a tower can bound its rows by the pass's frame
  boundaries. A kernel that reads neither refuses a call that sets them.

## Quantisation and rotation

| op | params | operands |
|---|---|---|
| `quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale`, `asum`? |
| `quant_act_i8g` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale` |
| `had_quant_act_i8` | `M`(range) `n` `group` `dtype` | `x` → `q`, `scale` |
| `rmsnorm_had_quant_i8` | `M`(range) `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_had_quant_i8` | `M`(range) `n` `group` `dtype` `act`:str/opt `mode`/opt | `a`, `b`? → `q`, `scale` |
| `dequant` | `M`(range) `n` `group` `dtype` | `q`, `scale` → `y` |
| `gram_accum` | `M`(range) `n` `group` `dtype` | `a`, `a_scale`?, `h`:inout |

**`gram_accum`** characterises a quantiser's output:
`h[k1][k2] += sum over rows of dequant(a)[m][k1] * dequant(a)[m][k2]`, the Hessian a GPTQ packer
needs over the operand the consuming GEMM contracts. `h` is f32 `[n, n]` and inout: a caller
zeroes it once and drives a whole calibration corpus through the graph, each launch adding its
block. It is not `gemm_nt` with a transposed operand: that op contracts the last axis of both sides
and this one the first. It is the one op here that no forward pass issues; it runs under a
calibration flag and its result is read at convert time, in another process, through a file.
`dtype` says how `a` is read: `fp8a8` is E4M3 codes with an f32 dequant scale per (row, `group`
columns) in `a_scale`; `bf16` is a plain activation with no `a_scale` and no grouping.

**Activation widths.** Which width a linear's activation takes follows from the weight. Where the
checkpoint ships E4M3 weights and a block scale plane, every linear is `fp8a8`
(`arch/common/rad_fp8.h`). libr4d's `fp8a16` matvec (`M <= 8`) reads bf16 directly, but it assumes
a weight layout the stored WMMA fragment order does not provide, so it is not used. Where the engine
quantises the activation for an integer weight it is int8: `quant_act_i8` for `w4a8`, `w8a8` and
`w2a8`, `quant_act_i8g` for `i8a8`.

**`group` has different meanings in this table.** A kernel that reads it the other way is wrong,
not slow. `libref/ref_quant.cpp` is the definition.

- **`quant_act_i8`**: one f32 scale per row, `max(amax, 1e-8) / 127`, and codes `x * (1 / scale)`
  rounded to nearest even, stored in the step order the int8 WMMA reads an A fragment in: the
  middle two dwords of every 16-byte step exchanged (column `c` at byte `c` with bits 2 and 3
  swapped), so `n` is a multiple of 16. `group` sizes `asum`, each row's code sums per `group`
  columns, group-major `[n/group][ceil(M/16)*16]`, which an asymmetric weight grid needs to remove
  its zero point. Rows past `M` are not written.
- **The `had_*` int8 forms**: `group` is the Walsh–Hadamard rotation width (libr4d's `had`), and
  the int8 scale is per row, carrying the `1/sqrt(group)` normalisation. The rotation is what makes
  one scale per row viable, so its width and the scale granularity cannot be the same number.
- **The `had_*` fp8 forms** (`had_quant_act_fp8`, `gated_had_quant_fp8`): `group` is both. They
  serve a four-bit MoE expert whose scale grid is already one bf16 per (row, 128 of K); a rotation
  of the same 128 lines the two up, so the scale is per group, `[M, n/group]` as for
  `quant_act_fp8`. They apply no normalisation: `<Hw, Ha> = group * <w, a>`, and the whole factor is
  folded into the stored weight scale at convert, where dividing a bf16 by a power of two is exact.
  Splitting it as the int8 forms do would round `amax/7/sqrt(group)` into bf16, a ~0.2% error per
  group. `n` must be a multiple of `group`.
- **`quant_act_i8g`, `had_quant_act_i8g`, `dequant`**: `quant_act_fp8`'s grouping at int8 codes:
  one f32 scale per `group` contiguous elements of a row, so `scale` is `[M, ceil(n/group)]`
  (`group <= 0` or `group >= n` is one per row). The scale is `amax / 127` (1 for a zero group) and
  each code the value over it rounded to nearest even and clamped to ±127, row-major. They feed the
  int8 trunk GEMM (`gemm_nt_q` at `dtype=i8a8`), which folds a scale per K block. The `had_` form
  rotates first. `dequant` inverts `quant_act_i8g`: the scale stream's extent gives the group count
  and `group` each group's width.

The rotation is unnormalised butterflies at ascending strides `1, 2, 4, ... group/2`, pairing `i`
with `i + h` where `(i & h) == 0`. The `1/sqrt(group)` is folded into the scale rather than applied
to the values; dividing every element by a constant cannot change which int8 code it rounds to. A
rotation block may not cross a tensor-parallel shard boundary, so the width must divide the
per-rank `n`.

`gated_had_quant_i8` takes the gate and up as two operands (libr4d's form), or one fused `gate_up`
`[M, 2n]` with the gate first as `a` and `b` absent. `mode` is libr4d's integer spelling of `act`
and is optional.

### The fp8 fusions

| op | params | operands |
|---|---|---|
| `quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `had_quant_act_fp8` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `givens_dense_quant` | `M` `n` `group` `dtype` `parts` `out`:str | `x`, `tdense`w, `rscale`w → `q`, `scale` |
| `givens_rotate` | `M` `n` `group` `dtype` `stages` `parts` | `x`, `pairs`w, `angle`w, `rscale`w → `y` |
| `rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gated_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gemm_nt_q_gated` | `M` `N` `K` `group` `dtype` `act` | `a`, `a_scale`, `b`w, `b_scale`w → `q`, `scale`, `out_bf16`? |

`gemm_nt_q_gated` is `gemm_nt_q` over a `[N, K]` gate_up plane (`N` is the weight's rows, twice the
fused width, gate half first) with `gated_quant_fp8`'s pass in the epilogue, so the `[M, N]` bf16
tensor between them is never written. It is a separate schema, not a flag on `gemm_nt_q`, because
it does not produce that op's `y`. `act` is required here, unlike on the quantiser: an
implementation may compile one gate only, and an optional parameter would let it resolve for
another.

`out_bf16` is part of the function, not an optimisation: the unfused pair leaves a bf16 tensor
behind and takes the codes over it, so a layer with a bf16 consumer of the post-norm value must ask
for it. `residual` present is the `fused_add_rms_norm` form.

### ParoQuant rotations

A weight stored with the `givens` transform (`abi/rad_encoding.h`) is `v = w . D . G_1 ... G_S`: a
channel scale `D` and `S` stages of disjoint pairwise rotations applied to each row before
quantisation (ParoQuant, z-lab). The GEMM contracts `<v, x'>` with `x' = x . D^-1 . G_1 ... G_S`,
so the activation takes the same rotation once a token.

- **`givens_dense_quant`** computes `x / rscale`, the rotation, then a block-scaled quantiser per
  `group` columns that `out` names: `fp8` is `quant_act_fp8`'s (scale `amax / 448`), `i8` is
  `quant_act_i8g`'s (scale `amax / 127`, round to nearest). The rotation is one dense 128 x 128 f16
  matrix per (part, group), `tdense` `[parts * n, 128]`, composed from that group's `stages` rounds
  of `a' = a cos t + b sin t, b' = b cos t - a sin t`. Each pair lies inside one scale group, so the
  matrices belong to the weight: rad-convert composes them once in f64 and stores them as the
  weight's `t.dense` table, which the layout hook (`r4d_rad_layout_gdense`) only moves. Consumers
  are `gemm_nt_q` at `mxfp4a8` (`out` fp8) and at `w4a8i8_asym` / `w5a8i8` (`out` i8).
- **`givens_rotate`** is the same rotation stage by stage with no quantiser, `y` in its own dtype
  (f16), for the 16-bit-activation weights `w5a16` and `w4a16_asym`, whose GEMM serves any `M`
  (rows past 64 run as chunks of 64).

A fused projection (gate|up, qkv|z) was rotated per checkpoint tensor, so the activation has
`parts` rotated copies side by side: `q` is `[M, parts * n]` and `scale` `[M, parts * n / group]`;
`pairs` and `angle` stack each part's `stages` rows and `rscale` has one row per part. `gemm_nt_q`'s
`pb1` and `pb2` are the row bounds of the weight's parts: rows `[0, pb1)`, `[pb1, pb2)` and
`[pb2, N)` meet copies 0, 1 and 2 of the wide `a`/`a_scale`. `pb1` absent is one part; `pb2` absent
ends the second part at `N`. The `mxfp4a8` rows and the asymmetric 4- and 5-bit rows read them;
host rows that do not (libavx) refuse a call that names one rather than contract against the wrong
copy.

### Int8 and gated fp8 fusions (device-only)

| op | params | operands |
|---|---|---|
| `had_quant_act_i8g` | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `gated_had_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate_up` → `q`, `scale`, `out_bf16`? |
| `gate_quant_fp8` | `M` `n` `group` `dtype` `act`:str/opt | `gate`, `x` → `q`, `scale`, `out_bf16`? |
| `rmsnorm_quant_i8` | `M` `n` `eps`:f64 `dtype` `wadd`:f64/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`?; `w_post`w?, `scalar`w? |
| `gelu_mul_quant_i8` | `M` `n` `dtype` | `gate_up` → `q`, `scale`, `out_bf16`? |

`gelu_mul_quant_i8` is `gelu_mul` with `rmsnorm_quant_i8`'s quantiser on its pass, byte-identical
to the pair.

`rmsnorm_quant_i8` is `rmsnorm` with `quant_act_i8`'s quantiser on its pass: step-order int8 codes
and one f32 scale per row, what `gemm_nt_q` reads at `dtype=w4a8` and `w8a8`. With `residual` the
norm is of `residual + x`, stored back; `w_post` normalises `x` before that add and `scalar` (`[1]`)
multiplies the sum after it, together a sandwich-norm model's residual junction (`rmsnorm` → `add`
→ `scale` → `rmsnorm` → `quant_act_i8`). Byte-identical to that chain: every intermediate the chain
stores is rounded to bf16, and both sums of squares take `rmsnorm`'s partition.

### `dequant_w4` (device-only)

| op | params | operands |
|---|---|---|
| `dequant_w4` | `N` `K` `group` `dtype` `twos` | `wq`w, `ws`w → `y` |

libr4d's own op: a packed 4-bit fragment-ordered weight back to row-major bf16. libref declares
`dequant`, which is activation-shaped, and nothing else declares `dequant_w4`.

`twos` selects the symmetric two's-complement nibble over the asymmetric unsigned-nibble-minus-
zero-point grid. The two differ in bit 3, so the wrong one is wrong output, not a refusal.

`wq` and `ws` must be weight operands. As plain inputs they would name the same bytes, but the core
would stop resolving their handles and stop waiting on the mover's event, reading weight bytes that
may not have arrived.

## GEMM

| op | params | operands |
|---|---|---|
| `gemm_nt` | `M`(range) `N` `K` `dtype` | `a`, `b`w → `y` |
| `gemm_nt_bias` | `M`(range) `N` `K` `dtype` `act`/opt | `a`, `b`w, `bias`w, `res`? → `y` |
| `gemm_nt_q` | `M`(range) `N` `K` `group` `dtype` `pb1`/opt `pb2`/opt | `a`, `a_scale`?, `b`w, `b_scale`w → `y`; `a_sum`?, `b_ref`w? |
| `gemm_nt_q_bias` | `M`(range) `N` `K` `group` `dtype` | `a`, `a_scale`, `b`w, `b_scale`w, `bias`w → `y` |

`nt` is A row-major `[M,K]` times B transposed `[N,K]`, giving `[M,N]`: the layout every weight in
a transformer already has. `gemm_nt_q`'s `dtype` names the operand combination, so one op covers
every quantised format and an architecture plugin that changes quantisation changes one string, not
its op names.

`gemm_nt_q` takes the activation scales as an operand because the quantiser produced them; a kernel
that folds dequantisation into its epilogue still receives them. Being a quantiser's amax over its
range, an `a_scale` entry is never negative, and a kernel may treat a block whose scale is zero as
contributing nothing (libr4d's MXFP4 GEMMs carry their accumulators in the last positive scale).
`a_scale` is absent for the f16-activation grids. At `w4a8`, `w4a8_asym`, `w8a8` and `w2a8`, `a` and `a_scale` are
`quant_act_i8`'s codes, in its step order, and per-row scale, and `a_sum` is its `asum`.
At `i8a8` they are `quant_act_i8g`'s row-major codes and per-(row, 128 of K) scales against an int8
weight with a bf16 scale per (row, 128 of K); at `mxfp4a8`, `quant_act_fp8`'s pair against e2m1
codes with an E8M0 exponent per (row, 32 of K).

`a_sum` removes an asymmetric grid's zero point and `b_ref` is the per-row reference exponent an
MXFP4 kernel folds its block exponents against. `b_ref` is derived from `b_scale` and does not
change the product. Both host libraries accept `b_ref` beside an e2m1 weight and ignore it, and
refuse `a_sum` (`RAD_E_UNSUPPORTED`) except at the grids that remove their zero through it.

The zero-point grids read unsigned codes as `b` and their scale and zero joined as `b_scale` (above):
weight element (n, k) is `(code - zero) * scale`, one f16 scale and one integer zero per (row,
`group` of K). libref's reference forms each group's sub-sum on the integer grid and scales it once:

| dtype | codes | `a`, `a_scale` | group sub-sum `sub` | accumulated |
|---|---|---|---|---|
| w2a8, w4a8_asym | u2, u4 | `quant_act_i8`'s step-order codes, per-row scale; `a_sum` required | `sum(a * code) - zero * a_sum[g][m]`, exact | `(sub * a_scale[m]) * scale` |
| w4a8i8_asym, w5a8i8 | u4, u5 | row-major int8, f32 scale per (row, 128 of K) (`givens_dense_quant`) | `sum(a * (code - zero))`, exact | `(sub * a_scale[m][g]) * scale` |
| w4a16_asym, w5a16 | u4, u5 | a 16-bit activation (`givens_rotate`), no `a_scale` | `sum(a * (code - zero))` in f32, ascending k | `sub * scale` |

`a_sum` is `quant_act_i8`'s `asum`, group-major `[K/group][Mpad]` with `Mpad = ceil(M/16)*16`. The
ParoQuant grids (the four 4- and 5-bit ones) take parts (`pb1`, `pb2`); `w2a8` and `w4a8_asym` do
not, and none has a bias form. libr4d's kernels compute the same products in another order (split-K
partials, the activation scale applied at the store); libavx folds the scales into its decoded rows,
each product rounding once where libref rounds each group's sub-sum. A `b_scale` that is not joined
at these grids has no zero to read and is refused (`RAD_E_DTYPE`). libref describes none of them
(`opd_shape` declines: no single tensor states a scale and a zero), so rad-kbench and the op oracle
leave them out and rad-avx-check builds them by hand.

`gemm_nt_bias`'s epilogue is a linear layer's: the biased product rounded to `y`'s dtype, then
`act` of it (`none`, `gelu` exact, `gelu_tanh`) rounded, then `res` added and rounded. Each step is
what a separate tensor would hold, so the fused form equals the unfused chain bit for bit. `res`
may be `y` itself, the residual stream updated in place (the encoder towers' layers do this), so an
implementation reads `res` before it stores the product over it.

### Runtime LoRA: `lora_shrink` and `lora_expand`

| op | params | operands |
|---|---|---|
| `lora_shrink` | `M`(range) `K` `R` `S` `group`/opt `dtype` | `a`, `a_scale`?, `slot`, `lora_a` → `t` |
| `lora_expand` | `M`(range) `N` `R` `S` `dtype` | `t`, `slot`, `lora_b`, `scale`, `y`:inout |

The two halves of `y += scale ⊙ (B · (A · x))` at a trunk GEMM, issued by the core right after the
GEMM in the GEMM's own row slice, with several adapters and no adapter mixed in one step. `slot` is
an i32 per row naming its adapter's pool slot; **a row with `slot < 0 || slot >= S` is a base row
and neither op touches it**: `lora_shrink` writes nothing into its `t` row, and `lora_expand` does
not read or write its `y` row at all — so a base row stays bit for bit what the GEMM wrote, where
`y + 0` would already have turned a -0 into +0.

- `lora_shrink`: `t[m, j] = Σ_k x(m, k) · lora_a[s, j, k]`, `s = slot[m]`, f32 accumulate.
  `a`/`a_scale` are the GEMM's own first two operands, and `x(m, k)` is the value `gemm_nt_q` uses
  for that element: the code times its group's scale, read through the same scale-stream rule
  (`[rows, groups]` or `[rows]`, the group derived from the stream), or a `bf16`/`f16`/`f32`
  element as it stands. An `fp8_e4m3`/`fp8_e5m2`/`i8` `a` without `a_scale` (null data) is refused.
  `group` is a cross-check on the stream's grid -- `ceil(K / group)` groups along K -- and 0 means
  none; `dtype` is `a`'s tensor dtype name
  (`rad_dtype_name`: `bf16`, `f16`, `f32`, `fp8_e4m3`, `i8`, ...), not a combined quant string.
  `lora_a` is `[S, R, K]` bf16, `t` is `[M, R]` f32.
- `lora_expand`: `y[m, n] = y[m, n] + scale[s, n] · Σ_j t[m, j] · lora_b[s, n, j]`, f32
  accumulate, one rounding to `y`'s dtype (`bf16`/`f16`/`f32`, named by `dtype`). `lora_b` is
  `[S, N, R]` bf16 and `scale` `[S, N]` f32 — per output column, so a fused projection whose parts
  carry different `alpha / r` is one launch.

`a`, `a_scale` and `y` may be row and column slices of wider buffers; every row stride is the
tensor's. The pools are ordinary inputs and not weights: the engine owns them as program buffers
and copies an adapter into a slot between steps, so a recorded pass replays the same pointers and
reads whatever adapters the slots hold. A row's result does not depend on `M` or on any other row
(no split of a reduction that moves with the batch), which is what lets a request's bytes be the
same alone and in a mixed batch. Implementations: libref, libavx, libr4d (scalar f32 rows).

## Attention

| op | params | operands |
|---|---|---|
| `attn_paged` | `q_len`(range) `head_dim` `gqa` `block_size`/der `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out` |
| `attn_dense` | `M`(range) `head_dim` `gqa` `causal` `dtype` `scale`:f64/opt `max_seqlen`/opt `window`/opt `cu_div`/opt `block`/opt | `q`, `k`, `v`, `cu_seqlens` → `out` |
| `attn_local_rel` | `M`(range) `head_dim` `n_head` `left` `right` `softcap`:f64 `cu_div`/opt `dtype` | `q`, `k`, `v`, `relk`, `pds`w, `cu_seqlens` → `out` |
| `kv_store` | `M`(range) `head_dim` `n_head_kv` `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `k`, `v`, `slot_mapping`, `kv_cache`:inout |
| `rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `scale`:f64 `mode` `rotary_dim`/opt `sections`/opt `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt `ladder`/opt | `qkv`:inout, `positions` |
| `rope_table` | `M`(range) `rot` `theta`:f64 `scale`:f64 `dtype` `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt `head_dim`/opt `ladder`/opt | `positions` → `cos_sin` |
| `qk_norm_rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `wadd`:f64/opt `v_norm`/opt `sections`/opt `comps`/opt | `qkv`:inout, `positions`, `q_w`w, `k_w`w |
| `qkv_norm_rope` | `M`(range) `head_dim` `n_head` `n_head_kv` `theta`:f64 `eps`:f64 `scale`:f64/opt `mode`/opt `rotary_dim`/opt `ladder`/opt `wadd`:f64/opt `v_from_k`/opt `dtype` | `q`:inout, `k`:inout, `v`:inout, `positions`, `q_w`w, `k_w`w |

**`attn_paged`**'s band is `q_len`, not `M`: one query row at decode, `1 + n_spec` at a speculative
verify, thousands in a prefill chunk, and libr4d has a different kernel for each band.
`block_size` is read off the resolved kernel, not chosen: declare the group, declare the op, then
ask `rad_kv_block_size()` (`spec.md` §7.2).

`cu_seqlens` is optional and makes the batch ragged: `[n_seq+1]` i32, sequence `s` owning query rows
`[cu[s], cu[s+1])` of the dense `q` and `out` blocks. Absent, every sequence has the same query
length and the total query rows must be a multiple of `n_seq`. Pass it: the scheduler builds mixed
steps (one sequence's prefill chunk beside other sequences' decode rows, each with its own verify
window) whenever prefill and decode coexist, and those are not uniform. `q_len` is then the batch
maximum, which selects the band; the per-sequence lengths come from `cu_seqlens` alone.

**`attn_dense`** serves vision towers and encoders: nothing paged, `cu_seqlens` bounding one image
or one input per entry so a whole batch is one launch. `window` is `attn_paged`'s rule over a
segment's own positions (causal: key `j` is visible to query `i` when `i - j < window`; non-causal:
when `|i - j| < window`), and 0 is no window. A bidirectional encoder's sliding layer of radius `r`
(`|i - j| <= r`) is `window = r + 1`. `q`, `k` and `v` may each be `[tokens, heads, head_dim]` or
`[tokens, heads * head_dim]`, and the second form may be a column slice of a fused projection read
at its row pitch, so a tower's `[q|k|v]` row needs no copy.

**`attn_local_rel`** is Gemma 4's audio attention: local, with a relative-position term and a
softcap. Query row `i` of a segment sees key rows `j` of the same segment with
`0 <= i - j <= left` (`right` must be 0). Per head, `qf = q * (head_dim^-0.5 / ln 2) * softplus(pds)`
(softplus rounded to `pds`'s dtype, as the reference evaluates it on the parameter) and
`logit = qf . k * (ln(1 + e) / ln 2) + qf . relk[i - j]`, then `softcap * tanh(logit / softcap)`, a
softmax over the visible keys and the weighted sum of `v`, all in f32 and rounded once. `relk` is
`[left + 1, n_head * head_dim]`, row `d` the projected position for distance `d`. The upstream
reference computes this in 12-query blocks over 24-key windows with a Transformer-XL shift, giving
every key outside the window a `-1e9` logit, so the sliding form is the same function.

**`rope`'s multi-component modes.** `mrope`, `imrope` and `axial` rotate with NeoX pairing and let
several position components drive different frequencies. `positions` is then `[C, M]`,
component-major, and `sections` names how many of the `rotary_dim / 2` frequencies each component
owns:

- `mrope` gives contiguous runs of them to components 0, 1, ... (Qwen2-VL);
- `imrope` interleaves: component 1 takes frequency `i` where `i % 3 == 1` and `i < 3 * s1`,
  component 2 where `i % 3 == 2` and `i < 3 * s2`, component 0 the rest (Qwen3-VL);
- `axial` gives contiguous runs and restarts the frequency ladder in each, `theta^(-2j / (2 s_c))`
  for the run's `j`-th frequency (a vision tower's 2-D rotary over row and column).

With `[M]` positions every component is that one position, so `mrope` and `imrope` over a text batch
are exactly `neox`.

**`ladder`** names the width the base frequency ladder is measured in. Absent or `rotary`, the
exponent divides by `rotary_dim`: a partial rotary is a complete rotation of its own width. `head`
divides by `head_dim` instead (the proportional variant), so the `rotary_dim / 2` frequencies are a
prefix of the full-width table: at `head_dim 512, rotary_dim 128`, frequency `i = 1` is
`theta^(-1 / 256)`, not `theta^(-1 / 64)`. Only the base ladder moves (`axial` keeps its per-section
ladders), and the head past `rotary_dim` is untouched either way. `rope_table` takes `head_dim` for
the same reason: its angles must stay `rope`'s exactly.

**`qk_norm_rope`** RMS-norms each q and k head of a fused `[q | k | v]` row against its gain (plus
`wadd`), rounds, and rotates in place.

- `v_norm` 1 also RMS-norms every v head with a unit gain and does not rotate it: Gemma 4's value
  norm, which has no weight.
- `mode` `blocks` rotates each section as its own block of channels: section `c` is `sections[c]`
  pairs over the next `2 * sections[c]` channels, paired `(start + j, start + sections[c] + j)`
  (NeoX inside the block), turned by position component `comps[c]` (default `c`) at
  `theta^(-j / sections[c])`. The sections fill `rotary_dim` exactly and `positions` is `[C, M]`.
  This is Gemma 4's vision rotary: x over the first half of each head and y over the second, each a
  complete rotary; `rope`'s `axial` pairs across the halves instead.
- `mode` `gptj_blocks` is `blocks` with each section's pairs adjacent, `(start + 2j, start + 2j + 1)`:
  a 3-axis image rotary's complex form.
- `mode` `imrope` is `rope`'s `imrope` over NeoX pairs, reading three position planes (Qwen3-VL's
  text attention).

**`qkv_norm_rope`** (device-only) is the same prologue over separate `q`, `k` and `v` projections
(Gemma 4's text attention, three GEMMs): the q and k norms, the unit-gain value norm and the rotary
in one dispatch. `v_from_k` 1 takes v from the raw k before the k norm (the k == v global class).
The rotary parameters are `rope`'s, `ladder` included. Byte-identical to `rmsnorm_p`, two
`rmsnorm`s and two `rope`s.

| op | params | operands |
|---|---|---|
| `attn_paged_gate_quant` | `q_len`(range) `head_dim` `gqa` `block_size` `causal` `window` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt:capacity `group` `act` | `q`, `kv_cache`, `block_table`, `seqused`, `k_descale`?, `v_descale`?, `cu_seqlens`? → `out`; `gate` → `q8`, `q8_scale` |

**`attn_paged_gate_quant`** (device-only) is `attn_paged` followed by `gate_quant_fp8` over its
output, finished inside the split-KV merge. A merge row is one (token, head) pair of `head_dim`
columns, which is a `gate_quant_fp8` row, so the merged values are gated (`bf16(sigmoid(gate))`,
rounded before the multiply), rounded, written to `out` and quantised to E4M3 with one scale per 128
columns while still in registers. `r4d_selftest` checks it byte for byte against the pair at both
merge forms and over a bf16 and an E4M3 cache. `block_size` is required rather than derived because
the cache's page size is `attn_paged`'s to answer; the caller passes the value it read. The gated
attention block uses it on the QSA sparse path, where every launch is one query row per sequence
and the decode kernel serves the whole step in one launch. `max_seqs` is a capacity
(`RAD_PROLE_CAPACITY`): the block declares it at the step size, and the activation arena's smaller
declares bound it lower.

### The KV cache operand

`attn_paged` and `kv_store` take the group's cache as operand 1, passed as `RAD_KV(group, layer)`.
`RadKVGroupBatch` carries the slot mapping, the block table and the sequence lengths, but the pool
base belongs to the block manager, not the batch.

### `qk_norm_rope_gate` (device-only)

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`? |

It splits an interleaved `[q|gate]` projection per head, RMS-norms q and k over the head, applies
partial NeoX rope and copies the gate: four launches a layer in one. It is not `qk_norm_rope`, which
takes `theta`: this op takes a precomputed `cos_sin` table (`rope_table` builds it) and a different
operand list. `rot` is the rotary width, `cs_f32` and `pos_i64` state the widths of the table and
the position vector, and `gate_out` is optional for a caller that reads the gate at its column
offset inside the projection.

**`n_head` 0 is the K-only form.** The `k` side reads one head per `head_dim` columns at the
operand's own row pitch and writes contiguously to `k_out`; only the q side adds the `[q|gate]`
split. A plain input asks for `n_head` 0 and passes no `q_gate`, `q_w` or `q_out`. The QSA
indexer's query heads have that shape: `heads` of them inside a `heads + 1` projection, a stride no
rank-2 view can express, which would otherwise take `heads` `rmsnorm` launches plus a `rope`.

| op | params | operands |
|---|---|---|
| `qk_norm_rope_gate_kv_store` | `M` `head_dim` `n_head` `n_head_kv` `rot` `q_dtype` `eps`:f64/opt `wadd`:f64/opt `cs_f32`/opt `pos_i64`/opt `block_size` `kv_dtype` `k_scale`:f64/opt `v_scale`:f64/opt | `q_gate`?, `k`, `cos_sin`, `positions`, `q_w`w?, `k_w`w → `q_out`?, `k_out`, `gate_out`?; `v`, `slot_mapping`, `kv_cache`:inout |

**`qk_norm_rope_gate_kv_store`** is the prologue followed by `kv_store` in one launch: each k head's
block already holds its token's rotated k one element a lane, so it also writes that element and
the head's v element into the cache at the token's slot. The store reads `k_out`, so the tail adds
only `v`, `slot_mapping` and `kv_cache`; every parameter is one of the two ops' own. It is
byte-identical to the pair at either cache width: the store converts the rotated k exactly as
`k_out` stores it, a bf16 cache takes the values as they are, and the E4M3 conversion rounds each
element on its own.

Both ops are byte-identical to the kernels they replace, checked in `r4d_selftest` for the
`[q|gate]` form and the K-only form on a strided projection. The check is byte equality, not a
tolerance: a sum of squares folded in another order differs by an ULP, which a relative bound
accepts, but a rotated element is a difference of two products, so where those nearly cancel the
ULP becomes a few percent and the transcript moves within a few dozen steps. The kernel reproduces
the reference's fold exactly: widths above the head dim contribute zeros, and adding 0.0f is exact.

## Image generation

The ops a diffusion image generator declares (`rad_declare_generator`, `abi/rad_builder.h`): a
single-stream diffusion transformer's adaptive norm and gated residual, the timestep embedding,
per-image latent lanes and the flow-matching sampler over them, attention over a cached
conditioning prefix, and an image decoder's convolution, channel norm, resampling shortcuts and
pixel output. Each rounds where the reference model materialises a bf16 tensor; the libref function
says which (`libref/ref_gen.cpp`).

| op | params | operands |
|---|---|---|
| `ln_mod` | `M`(range) `n` `eps`:f64 `dtype` | `x`, `s`, `cu_seqlens` → `y` |
| `gate_add` | `M`(range) `n` `dtype` | `y`, `g`, `cu_seqlens`, `x`:inout |
| `ln_mod_quant_fp8` | `M`(range) `n` `eps`:f64 `group` `gated` `dtype` | `y`?, `g`?, `s`, `cu_seqlens`, `x`:inout → `q`, `scale`, `z`? |
| `timestep_embed` | `M`(range) `n` `scale`:f64 `max_period`:f64 `round`/opt `dtype` | `t` → `y` |
| `gather_rows2` | `M`(range) `n` `dtype` | `a`, `b`, `idx` → `y` |
| `lane_gather` | `M`(range) `n` `stride` `dtype` | `src`, `lane`, `cu_seqlens` → `y` |
| `lane_store` | `M`(range) `n` `stride` `dtype` | `src`, `lane`, `dst`:inout |
| `randn_lanes` | `M`(range) `n` `stride` `dtype` | `flags`, `seed`, `lane`, `cu_seqlens`, `dst`:inout |
| `flow_euler` | `M`(range) `n` `stride` `dtype` | `v`, `sigma`, `sigma_next`, `cfg`, `pair`, `lane`, `flags`, `seed`, `cu_seqlens`, `inp_lat`?, `inp_mask`?, `state`:inout, `preview`:inout? |
| `chan_affine` | `M`(range) `n` `mul` `add` `mode` `dtype` | `x` → `y` |
| `conv2d_nhwc` | `M`(range) `cin` `cout` `k` `stride` `pad` `up` `dtype` | `x`, `w`w, `b`w → `y` |
| `l2norm_chan` | `M`(range) `n` `act` `dtype` | `x`, `gamma`w → `y` |
| `d2s_add` | `M`(range) `cin` `cout` `ft` `dtype` | `x`, `y`:inout |
| `s2d_avg_add` | `M`(range) `cin` `cout` `ft` `dtype` | `x`, `y`:inout |
| `to_rgba8` | `M`(range) `n` `dtype` | `x` → `y` |
| `gelu_tanh` | `M`(range) `n` `dtype` | `x` → `y` |
| `scatter_add_rows` | `M`(range) `n` `dtype` | `v`, `idx`, `x`:inout |
| `attn_prefix_self` | `q_len` `head_dim` `gqa` `block_size`/der `causal` `q_dtype` `kv_dtype` `scale`:f64/opt `max_ctx`/opt `n_head`/opt `max_seqs`/opt `qk`/opt | `q`, `k`, `v`, `kv_cache`, `block_table`, `seqused`, `cu_seqlens`, `k_descale`?, `v_descale`?, `k_i8`?, `k_scale`? → `out` |

**Sequences and lanes.** `cu_seqlens` partitions the rows into entries (`[n_seq+1]` i32, entry `e`
owning `[cu[e], cu[e+1])`; it need not start at 0). `ln_mod` and `gate_add` take each row's
modulation row from `s`/`g`, `[n_seq, >= n]` at its own row pitch (a column slice of one shared
modulation projection). A lane is one image's latent in a per-image state buffer, `stride` rows
apart: `lane_gather` copies entry `e`'s rows out of lane `lane[e]`, and `lane_store` copies one
source into every listed lane (a repainted picture's latent into each of its images).

- `ln_mod`: `y = bf16(bf16(LN(x)) * bf16(1 + s[seq]))`, a layer norm with no gain and no bias.
- `gate_add`: `x = bf16(x + bf16(bf16(tanh(g[seq])) * y))`.
- `ln_mod_quant_fp8`: with `gated` 1, `gate_add` first; then `z = ln_mod(x)` and
  `(q, scale) = quant_act_fp8(z, group)`, `z` written when bound. Each value is rounded as its own
  store would round it, so the bytes are the three ops'.
- `timestep_embed`: `[cos | sin]` of `t' * f_i`, `f_i = exp(-ln(max_period) * i / (n/2))` in f32,
  `t' = scale * t`; with `round` 1, `t' = scale * bf16(bf16(t * scale) / scale)`, the value
  diffusers' pipeline hands its transformer. `t` is f32 `[M]`.
- `gather_rows2`: `y[i] = idx[i] >= 0 ? a[idx[i]] : b[-1 - idx[i]]`, zeros past either source.
- `randn_lanes`: for entries whose `flags` carry 1, the lane's rows become the entry's seed's
  standard-normal stream (`abi/rad_philox.h`), element `(k, c)` the stream's `k * n + c`. `seed` is
  `[n_seq, 2]` u32, low word first.
- `flow_euler`: for every entry with `pair != -2` (an entry that is a negative conditioning moves
  no lane), `v' = pair >= 0 ? bf16(neg + bf16(cfg * bf16(v - neg))) : v`, `neg` the paired entry's
  rows; flag 4 writes `bf16(x - sigma * v')` to the lane's `preview`; then
  `x' = bf16(x + bf16(bf16(sigma_next - sigma) * v'))`; flag 2 restores the kept region,
  `x' = bf16(m x' + (1 - m)((1 - sigma_next) src + sigma_next noise))` with `m` the lane's f32
  `inp_mask` row, `src` its `inp_lat` row and `noise` the seed's stream.

**Images** are `[height, width * channels]`, dense, and the ops read both sizes off the operand
shapes: an image's size belongs to a request, and a parameter is fixed at declare.

- `conv2d_nhwc`: `w` is PyTorch's `[cout, cin, k, k]` (a kernel may store it permuted through its
  layout hook), `b` `[cout]`; `up` 2 reads the input upsampled 2x nearest-neighbour; `pad` `same`
  is `k/2` zeros a side, `down` is none above and left and one below and right
  (`ZeroPad2d((0,1,0,1))` before a stride-2 conv). f32 accumulation, the bias added, one rounding.
- `l2norm_chan`: each pixel's channels to unit L2 (eps 1e-12), times `sqrt(n)`, times `gamma`
  (three roundings), then `act` `none` or `silu`.
- `chan_affine`: `mul` and `add` are space-separated lists of `n` values, each rounded to bf16;
  `fma` is `bf16(bf16(x * mul) + add)`, `inv` is `bf16(bf16(x - add) / mul)`.
- `d2s_add` (Wan's DupUp3D on one frame): `y[2r+i, 2c+j, o] += x[r, c, (o F + off + 2i + j) / rep]`,
  `F = 4 ft`, `off = 4` when `ft` is 2, `rep = cout F / cin`.
- `s2d_avg_add` (AvgDown3D on one frame): `y` gains the mean of each output channel's group of
  `cin F / cout` dealt values, a dealt channel being `(c, t, i, j)` at `c F + 4t + 2i + j`; with
  `ft` 2 the frame `t = 0` is the zero frame the reference pads in, which the mean still counts.
- `to_rgba8`: `round(clamp(bf16(bf16(clamp(x, -1, 1) / 2) + 0.5), 0, 1) * 255)`, u8 out.

**`attn_prefix_self`** attends over two key sources: entry `s`'s first `seqused[s]` keys in the
paged cache, all visible, and the pass's own rows `k`/`v` beside `q` (dense, at their own pitch,
indexed as `q`'s rows), of which row `i` sees `j <= i` with `causal` and all of them without. It is
an image's latent rows against its conditioning, or one segment of a conditioning against the
segments before it. `qk` `"i8"` is the int8 score form and takes `k_i8` `[tokens, kv_heads *
head_dim]` and `k_scale` `[tokens, kv_heads]` f32, the own keys as `quant_act_i8g` writes them at
group `head_dim`. The query and every prefix key are quantised inside the op by the same rule
(scale `amax/127`, 1 for an all-zero row; code `rint(clamp(x * (1/scale), ±127))`), a score is the
exact integer dot product times both scales, and `k` is not read for the own rows.

## Mixture of experts

| op | params | operands |
|---|---|---|
| `router_topk` | `M`(range) `n_expert` `top_k` `norm` `dtype` | `logits` → `expert_ids`, `expert_w` |
| `moe_scatter` | `M`(range) `n_expert` `top_k` `expert_base`/opt `protect`:str/opt | `expert_ids` → `sorted_tok`, `expert_offset`, `expert_count` |
| `router_topk_scatter` | `M`(range) `n_expert` `n_local`/opt `top_k` `norm` `dtype` `expert_base`/opt `protect`:str/opt | `logits` → `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count` |
| `router_gemm_topk_scatter` | `M`(range) `K` `n_expert` `n_local`/opt `top_k` `norm` `dtype` `expert_base`/opt `protect`:str/opt | `a`, `b`w, `gate_b`w? → `logits`, `expert_ids`, `expert_w`, `sorted_tok`, `expert_offset`, `expert_count`, `gate`? |
| `moe_gemm` | `M`(range) `N` `K` `n_expert` `top_k` `a_order` `dtype` | `a`, `w`wt, `sorted_tok`, `expert_offset` → `y` |
| `moe_gather` | `M`(range) `n` `top_k` `dtype` `act`:str/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`? → `y` |
| `row_topk` | `M`(range) `N` `R` `vocab_off`/opt `dtype` | `x` → `idx`, `val`, `pairs`? |
| `row_topk_merge` | `M` `R` `world_size` `dtype` | `gathered` → `idx`, `val` |
| `pair_argmax` | `M`(range) `n` `dtype` | `pairs` → `idx`, `val`? |
| `logit_rerank` | `M`(range) `R` `n_vocab` `n_embd` `vocab_off`/opt `dtype` | `x`, `head`w, `idx_in` → `idx`, `val`, `pairs`? |

**`row_topk` and `row_topk_merge`** are a pair for a vocab-sharded head, whose logits cover only its
own slice. `vocab_off` shifts the reported columns into global ids, and `pairs` writes the same R
entries as `[M, R, 2]` (an int32 id in the first float slot, its value in the second): the plane
`core/sample/sampler.h` holds as `buf_pairs_` and the sampler's vocab-parallel path gathers. An
`all_gather` with `row = R*2` concatenates them into `[M, world_size * R, 2]`, rank r's at
`[r*R, (r+1)*R)`, and `row_topk_merge` reduces that to the global top-R. The merge is exact: a
candidate in the global top-R is in its own shard's top-R, and both stages break ties on the lower
id, so the result equals a top-R over the unsharded plane. A slot a shard could not fill carries id
-1 in `pairs` and is dropped; a sentinel id rather than a -inf value, because a masked logit can be
-inf legitimately.

**`pair_argmax`** is the merge at R = 1, for a vocab-sharded draft head that wants each position's
winner: `idx` gets the id of the largest value among a row's `n` gathered pairs, ties to the lower
id, negative ids skipped, -1 (and -inf in `val`) when nothing is valid, written one element a row at
the output's own pitch (a block drafter's contiguous proposal). It is not `row_topk_merge` at R = 1
because a rank still sends R >= 2 pairs: a row-interleaved `all_gather` moves whole 16-byte words
and a pair is 8 bytes, so `n` is `world_size * R`.

**`logit_rerank`** is the second half of a coarse top-R. A cheap, coarsely quantised draft head
proposes R candidates, and this op rescores those R against the full-precision head the trunk
already carries: a few hundred KB a row instead of a few hundred MB, and an answer that agrees with
the exact argmax on almost every row. A worse draft costs acceptance, never a token, since the
target verifies every proposal. `idx_in` carries global ids as `row_topk` reports them and
`vocab_off` is this rank's first column, so under vocab-parallel drafting each rank rescores the
candidates it owns and the existing `all_gather` + `row_topk_merge` finds the global winner. Ties
break on the lower id, as in the other two.

**`router_topk_scatter`** (device-only) is `router_topk` and `moe_scatter` in one launch. At a
decode shape each is a single 256-thread workgroup ranking about a kilobyte, so each costs the
one-workgroup dispatch floor and almost nothing else. The chain is closed (the sort reads only
`expert_ids`, which the selection writes), so a launch between them buys only a barrier.

- **Two expert counts.** `n_expert` is the router's: it is replicated and every rank picks from
  every expert. `n_local` is this rank's slice, which the sort runs over and which sizes
  `expert_offset` and `expert_count`, as when a calibration run divides Qwen3.8-Flash-Next's experts
  between two ranks. `n_local` absent means the slice is the whole router, as in every deployment
  that gives each rank a slice of every expert.
- **Bounded, with no fallback.** Only `moe_scatter`'s single-workgroup form is fusible
  (`M * top_k <= 1024`); above that the sort is three launches, which keep it stable without an
  atomic deciding the order. That bound covers the decode band (eight sequences at depth 3 and
  top_k 10 is 320 slots). A prefill chunk is past it, where the pair is amortised over real work.
  A caller past the bound gets a refusal, not a silent change of cost.

**`router_gemm_topk_scatter`** (device-only) adds the router's GEMM in front, in the same launch:
`gemm_nt` of `a` `[M, K]` against the router `b` `[n_expert, K]` into `logits`, then
`router_topk_scatter` over those logits. With `gate_b` `[1, K]` and `gate` `[M, 1]` (both or
neither) it is also the shared expert's one-column `gemm_nt` over the same `a`. The selection needs
every logit of a row, so it runs in the workgroup that finishes the GEMM last (a self-resetting
arrival counter per op instance, which libr4d takes in the row's `init`, so it is in the VRAM
budget; a launch without one is refused). It is byte-identical to the launches it replaces: an
element of the skinny bf16 GEMM depends only on its split-K count, which this op takes from the same
law at the same N, and the selection and sort are the same code. Decode only (`M <= 64`, past which
the router is the tiled GEMM), and bounded like the pair. The logits stay an output.

The router GEMM inside it always takes the split the untuned law picks, so a tune-cache entry for
the unfused router's `gemm_nt` does not reach it, and on a tuned install the two forms may differ in
the last bit; no shipped configuration tunes these GEMMs. An architecture asks for this op wherever
it declares the pair, and issues the router and the pair itself where it does not resolve or a pass
is past its rows.

**`moe_scatter`** is a stable counting sort of the (token, slot) pairs by expert. `sorted_tok` holds
the flattened index `token * top_k + slot`. A slot whose expert is outside
`[expert_base, expert_base + n_expert)` is dropped and the tail filled with -1, which is how expert
parallelism is expressed. Its output is what `rad_route_report` hands the heat engine; the histogram
is read back asynchronously and consumed on the next step, so nothing blocks the compute stream
(`spec.md` §5.5).

**`protect`** sorts a few experts last. A layer that keeps some experts at a higher precision serves
them with a second grouped GEMM, and a grouped GEMM takes one contiguous run of the offsets. So
`protect` lists local experts (ascending, at most eight) that the sort places after every other
expert, in the order listed, with the rest closed up: `expert_offset` and `expert_count` are in that
order, the regular experts are `[0, n - p)` of the offsets and the protected ones `[n - p, n)`. A
`moe_gemm` handed the trailing run (offsets that begin above zero) serves those rows and leaves the
ones below untouched.

**`moe_gather`** scatters the sorted expert output back to token order and weights it, summing a
token's slots in ascending slot order. With `shared` and `shared_gate` it also folds the shared
expert in: `y = bf16(shared * act(shared_gate)) + bf16(gathered)`, `scale_rows`' fused
gate-and-add form one op further back, so the bf16 plane between them never exists. Both narrowings
are part of the contract: the unfused pair writes the gather to a bf16 buffer and narrows the
product before the sum, so carrying either in f32 moves the last bit. `act` is `"sigmoid"` or
`"none"` and applies to the scalar, as in `scale_rows`. A shared arm that cannot be width-sharded
runs on one rank, so the op this replaces is a dispatch every other rank waits out inside the
block's all-reduce.

### Quantised experts and the row gate

| op | params | operands |
|---|---|---|
| `moe_gemm_q` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_odd`/opt `K_odd`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_odd`wt, `w_odd_scale`wt → `y` |
| `moe_gemm_q_mod4` | `M`(range) `N` `K` `n_expert` `top_k` `group` `a_order` `dtype` `N_1`/opt `K_1`/opt `N_2`/opt `K_2`/opt `N_3`/opt `K_3`/opt `parts`/opt | `a`, `a_scale`?, `w`wt, `w_scale`wt, `sorted_tok`, `expert_offset`, `w_1`wt, `w_1_scale`wt, `w_2`wt, `w_2_scale`wt, `w_3`wt, `w_3_scale`wt → `y` |
| `scale_rows` | `M`(range) `n` `act` `dtype` | `x`, `s`, `add`? → `y` |

The arithmetic is the same whether a kernel is handed the tables or the stacked form
([Weight operands and weight tables](#weight-operands-and-weight-tables)).

**`moe_gemm_q`'s experts are two tables by parity.** `w`/`w_scale` hold the even-numbered experts at
`[N, K]` and `w_odd`/`w_odd_scale` the odd ones at `[N_odd, K_odd]` (defaulting to `N`, `K`); expert
`e` is entry `e / 2` of its class. Every entry of one table shares a geometry, and a rank that holds
an uneven slice of every expert's width (three 128-column blocks of one parity's experts and two of
the other's, which is how a five-block expert splits evenly across two ranks on average) has two.
`parts` stacked parts run along both the weight rows and the output columns (two for a fused
gate/up); the output is `max(N, N_odd)` wide, and a narrower class fills the leading columns of each
output part and writes zeros after them, so the gated quantiser behind it reads one layout for every
row. The weight list is the even class's codes, its scales, then the odd class's codes and scales.

**`moe_gemm_q_mod4`** is the same GEMM over four classes by the expert's number mod 4. `w`/`w_scale`
hold class 0 at `[N, K]` and `w_q`/`w_q_scale` class `q` at `[N_q, K_q]` (defaulting to `N`, `K`);
expert `e` is entry `e / 4` of class `e % 4`, so class `q` holds `ceil((n_expert - q) / 4)` of them.
Four ranks splitting a five-block expert cannot share the extra block evenly over two classes, but
over four they can: rank `r` takes two blocks of class `r` and one of each other, five of every four
experts on every rank. It is a separate op, not a parameter on `moe_gemm_q`, because the declaration
divides a table op's weights over its table operands by count alone, so the operand list is the
class count. Everything else (`parts`, the zero fill, `a_order`) is `moe_gemm_q`'s, and the weight
list is class 0's codes and scales, then class 1's, 2's and 3's.

**The codebook forms** (`w4nla8h`, `w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h`; `docs/MOE-W4.md`) hold
u4 or u5 codes into an f32 table of 16 or 32 levels, declared as one view over the planes
`codes,table`, and a scale per (row, group of K): bf16 per 128, or E4M3 per 64 or 32 declared with
its fixed f32 level as `scale,scale.1`. Element (n, k) of an expert is `table[code] * scale`, times
`scale.1` where there is one. The encoding's `fwht128` is in the stored weight and in the
activation `had_quant_act_fp8` writes, so the GEMM applies none. On the host, a view of several
planes is stored in the joined form of `abi/rad_planes.h`; libref's and libavx's hooks accept
exactly the planes `dtype` names (code width and table length, scale dtype and group, `fwht128`)
and refuse any other selection at declare with `RAD_E_DTYPE`. libref computes the form as
`gemm_nt_q` does: per weight group the sub-sum of table values times activation codes, then
`(sub * sa) * sb` with `sb` the scale times its level. The logical form the oracle and rad-kbench
pass (table entries for codes, scale products for scales) goes through the same chain, bit for
bit. libr4d sums each 64- or 32-wide group under its scale before the activation's scale per 128,
and libavx folds both scales into the operands before one sum over K; both differ from libref in
the last bits only.

**`moe_gemm_q` is a separate op, not a `dtype` on `moe_gemm`**, for the reason `gemm_nt_q` is
separate from `gemm_nt`: the operand list differs and operands are positional. The group sub-sum is
accumulated from the raw codes and scaled once per group, as `gemm_nt_q` does; the two
references must agree in the last bits or `rad-kbench` blames a kernel for a choice made in
`libref`.

**`moe_gemm`** is what a model served at bf16 runs, and libr4d serves it at `dtype` bf16: one weight
table of the checkpoint's own row-major `[N, K]` planes, read with no layout hook, so a container
stores an expert as the safetensors file holds it. It has no parity classes and no `parts` (those
exist because a quantised expert's width must fall on its scale blocks), and its rows go through
the same run grouping and the same three forms as `moe_gemm_q`'s kernels, byte-identical to each
other.

**`scale_rows`** is the per-token scalar gate: `y[m, i] = x[m, i] * act(s[m])`, one scalar per row
broadcast across it. `mul` broadcasts one row down the rows, and one op with both rules would be
ambiguous when `rows == n`. `act` is `"sigmoid"` or `"none"` and applies to the scalar. Qwen3.5-MoE's
`shared_expert_gate` is a `[1, n_embd]` projection whose one output per token gates the shared
expert's contribution, and this is that multiply. With `add` present it is
`y = bf16(x * act(s)) + add`, the product narrowed before the sum as the unfused pair stores it.

### `a_order`

A routed block runs two `moe_gemm_q` ops, and they disagree about what a row of `a` is, so this is
a parameter, not a convention.

- The **gate_up** projection reads the block's normed input, one row per token. Sorted row `i`
  belongs to the pair `sorted_tok[i] = token * top_k + slot`, so it gathers `a[sorted_tok[i] / top_k]`:
  `a_order = "token"`, the default.
- The **down** projection reads gate_up's output through the gated quantiser, already one row per
  (token, slot) in the scatter's order, so row `i` is row `i`: `a_order = "sorted"`.

`a_scale` has a row for every row `a` has, either way.

It is stated, not derived from the row count (`M` against `M * top_k`), because no harness would
catch a wrong derivation: a kernel and a reference that both gather agree on every case, and
`rad-kbench` sizes each operand from the op's own shape hook, so a hook describing `a` at the extent
the gather wants builds a self-consistent, wrong case. Served, the down projection would hand every
slot of a token the first slot's intermediate, and the model would answer fluently and get facts
wrong.

## Linear attention (gated delta net)

| op | params | operands |
|---|---|---|
| `gdn_conv_prep` | `M`(range) `head_k` `head_v` `chunk`/der `conv_width` `l2_eps`:f64/opt `softplus_thr`:f64/opt | `x`, `w`w, `b`w?, `A_log`w, `dt_bias`w, `conv_state`:inout, `cu`, `a`?, `b_gate`?, `cache_idx`?, `has_init`? → `q`,`k`,`v`,`g`,`beta` |
| `gdn_conv_update` | `q_len`(range) `head_k` `head_v` `conv_width` `max_query_len`/opt | `x`, `w`w, `b`w?, `conv_state`:inout, `state_idx`, `num_accepted`, `cu` → `q`,`k`,`v` |
| `gdn_kkt_solve` | `M`(range) `head_k` `chunk`/der | `k`, `beta`, `g`, `cu` → `A` |
| `gdn_chunk_scan` | `M`(range) `head_k` `head_v` `chunk`/der `scale`:f64/opt `state_fp16`/opt | `q`,`k`,`v`,`A`,`g`,`beta`,`h0`?,`cu` → `o`, `ht`?; `state_idx`? |
| `gdn_recurrent_update` | `q_len`(range) `head_k` `head_v` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `q`,`k`,`v`,`a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`?,`norm_w`w? → `o`,`o_q`?,`o_scale`? |
| `gdn_conv_recurrent_update` | `q_len`(range) `head_k` `head_v` `conv_width` `n_head_v`/opt `scale`:f64/opt `eps`:f64/opt `l2_eps`:f64/opt `softplus_thr`:f64/opt `act`:str/opt `state_fp16`/opt `state_form`:str/opt | `x`, `conv_w`w, `conv_b`w?, `conv_state`:inout, `conv_idx`, `a`,`b`,`A_log`w,`dt_bias`w,`state`:inout,`state_idx`,`num_accepted`,`cu`,`z`,`norm_w`w → `o`,`o_q`?,`o_scale`? |
| `gdn_gated_rmsnorm` | `M`(range) `channels` `eps`:f64 `act`:str | `x`, `z`, `w`w → `o` |

**Speculative rollback is part of the schema** (`spec.md` §10). The recurrent state and the conv
window have already absorbed rejected tokens, so the state cache is a rolling window of
`conv_width-1 + n_spec` entries read at the slot the last accepted token left, and a rejection is a
change of read offset, not a recompute.

**`num_accepted` is required and one-based.** It counts the tokens the previous step committed: the
one it was given plus however many of its drafts survived. 1 is an ordinary decode step that drafted
nothing, and 0 is not a value either kernel can read (`gdn_conv_update` reads its window at
`num_accepted - 1`). A non-speculative decode passes all ones. It is one of the two exceptions to
the union rule: an op that lets the caller omit `num_accepted` has an optional rollback contract.

**`cu` is required by every gdn kernel, decode included**, the other exception. `gdn_conv_update`
and `gdn_recurrent_update` read `cu[n]` and `cu[n+1]` before any null test, and the sequence count
is `cu`'s length minus one; nothing else in the argument list carries it. A caller working from a
schema that made `cu` optional would pass a null and fault rather than be refused.

**`state_idx` is `[n_seq, W]`, and W is the group's, not the op's.** Column 0 is the sequence's
state. A kernel that needs somewhere to put what it cannot commit yet may ask for more columns, and
those are its scratch: the caller stages the row, zeroes every slot it names when the sequence is
admitted, zeroes the scratch again after any step that ran a different path over the sequence (a
chunked prefill commits the state directly and writes no scratch), and otherwise does not look
inside. `gdn_chunk_scan` takes the same row and reads column 0.

**`state_form` says what the state cache means.** `per_candidate`, the default, writes one full
state per candidate token and needs `1 + n_spec` distinct slots in `state_idx`. `anchor` keeps one
committed state per sequence and, beside it, the few hundred floats per head each token's rank-1
update needs; the next step replays the accepted prefix onto the state out of that scratch slot. That
is two slots whatever the draft depth, against `1 + n_spec` full states (1.3 GiB a sequence on
Qwen3.8-27B), and less write traffic. A zeroed scratch block replays as the identity, so "zero it
and pass `num_accepted` 1" is the correct handover out of a prefill.

The two forms produce the same `o` from a different `state`, so an implementation handed the other
form computes a correct output and disagrees with a checker on every element of the cache. The index
row's width does not say which form is meant, so it is declared, not inferred: the engine passes
`anchor` when the deployment speculates and `per_candidate` otherwise (the condition that sizes the
state pool), and an implementation that serves one form refuses the other.

libref serves both, the anchor form in libr4d's cache layout (`libref/ref_gdn.cpp`,
`AnchorLayout`): `state_idx` has six columns, the anchor in 0-2 and the factor block in 3-5. A paged
caller's window may shift by -1, 0 or +1 pages a step, so the advanced anchor is published at column
2 and each factor copy is stamped with its column; the next step reads the stamp at column 4 and the
anchor at the column it implies. The factors are each token's normalised k and its decay exponent
and beta (once per 32-row group of `head_v`, each group's rows replaying from their own copy, as
libr4d reads them), and v, at most 16 tokens a step, with the stamp just past them. A
head whose `[head_v, head_k]` block cannot hold 16 tokens' factors (under 64 x 64) holds as many as
fit, so a 16 x 16 head speculates at most 6 tokens deep. With an f32 state the replayed state is bit
for bit the per-candidate one, so `o` matches the per-candidate form exactly. libavx serves both
forms in the same layout, its replay bit for bit its own per-candidate state (rad-avx-check runs the
two over a four-step schedule). The end-to-end check is that greedy speculation emits the same bytes
as greedy without it.

**`gdn_conv_recurrent_update`** (`gdn_conv_update` then `gdn_recurrent_update`, in one launch)
computes the convolution where the recurrence reads it, so `q`, `k` and `v` are never written. The
operands are the convolution's inputs (`conv_idx` is its `state_idx`) followed by the recurrence's
without those three; the parameters are the union of the two ops' under their own names, and both
ops' rollback and state contracts hold unchanged. `z` and `norm_w` are required, because the fusion
serves the decode shape, where the fused row norm already puts one workgroup on each (sequence,
value head). It is byte-identical to the pair. libr4d rewrites a key head's conv state once every
value head reading it has read it, by an arrival count, so no workgroup waits on another.

**The output gate `z` may be a column slice**, on both `gdn_recurrent_update` and
`gdn_gated_rmsnorm`. The fp8 delta-net block projects `[q|k|v|z]` in one GEMM
(`arch/common/rad_block_gdn_fp8.h`), so the gate is the right-hand `n_head_v * head_v` columns of a
wider row and its `(token, head)` rows are not evenly spaced: heads are `head_v` apart inside a
token and the next token is the slice's own stride away. An implementation that reads one pitch off
`z` walks into the next token after `n_head_v` rows and is wrong without going out of bounds. A
caller may hand over `[tokens, heads, head_v]`, a flat `[tokens, heads*head_v]` buffer of its own,
or that slice; only the slice needs the token stride carried separately.

## Collectives

| op | params | operands |
|---|---|---|
| `all_reduce` | `world_size` `numel`(range) `dtype` `exact` `min_bytes`/opt `hops`/opt `overlap`/opt | `x`:inout, `y`:out? |
| `all_gather` | `world_size` `numel` `dtype` `row`/opt | `x` → `y` |

`exact` separates the exact sum from a lossy wire, and a caller must state it: a request that does
not accept a lossy wire gets no kernel rather than the quantised one.

`hops` (1 or 2) is a caller's request for one-shot or two-shot; libref accepts and ignores it, and
no libr4d row reads it. libr4d chooses by message size where a width has both: its two-shot rows
start above 524288 elements, a band edge of the ranged `numel`. Every all-reduce kernel serves `y`
absent, in place. `overlap` 1 says the caller issues the collective on its second lane, beside
compute; libr4d then picks rows with a narrow launch (8 blocks of 256 threads at two ranks, 4 of
1024 for the four-rank two-shots, exact and tiered-int8) that compute the same bits and leave the
CUs to the kernels running beside them. libref ignores it.

libr4d serves these at 2, 3, 4 and 8 ranks:

- Two ranks have their own kernels (one flag word a block), the rotated 6-bit lossy wire and the
  fused norm-and-reduce forms below.
- The wider widths share one template on the world size. 3, 4 and 8 have both one-shot and
  two-shot (the three-rank two-shot shards unevenly), and 3 and 4 have a tiered int8 lossy two-shot.
  8 has no lossy wire.
- `all_gather` at those widths rides the same instance as the wide `all_reduce` (`ar_gather_3rank`,
  `_4rank`, `_8rank`).

`min_bytes` is the byte floor under which a lossy row serves the message exactly. It is a parameter,
not a library constant, because it is a property of the link: compression pays only once the
transfer is bandwidth-bound, and for a decode step's few residual rows the rotate, pack and unpack
cost more than the bytes they remove. A row selected for `exact=0` therefore decides the row once,
at declare, and the wire per launch. Absent, the plugin uses its own default.

`all_gather.row` is one row of a rank's contribution, in elements. Absent or `0`, `y` is the plain
concatenation, rank `r`'s slice at `r * numel`. Present, the placement is row-interleaved: `x` is
`[rows][row]` and `y` is `[rows][world_size][row]`, so a vocab-parallel candidate merge reads one
row of gathered candidates per sampled position at a single row stride.

**`-ffp-contract=off` is mandatory for any quantised cross-rank op.** Two ranks see the two
dequantised products in opposite order, and contracting either into an FMA makes them disagree by
~1 ULP, which breaks the replicated-state invariant. `cmake/RadAddPlugin.cmake` sets it for all
device code.

### Collectives with a norm (device-only)

| op | params | operands |
|---|---|---|
| `ar_rmsnorm_quant_fp8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `ar_rmsnorm_quant_i8` | `M` `n` `eps`:f64 `world_size` `dtype` `wadd`:f64/opt | `x`:inout, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`?; `w_post`w?, `scalar`w? |
| `ar_hc_write` | `M`(range) `n` `hc` `world_size` `dtype` `wire`/opt | `y`:inout, `inj`, `h`:inout |
| `ar_gather_hc_write` | `M`(range) `n` `hc` `top_k` `world_size` `dtype` `act`/opt `wire`/opt | `y_expert`, `expert_w`, `sorted_tok`, `shared`?, `shared_gate`?, `y`:inout, `inj`, `h`:inout |
| `ar_ln_had_quant_i8` | `M` `n` `eps`:f64 `group` `world_size` `dtype` `wadd`:f64/opt `wire`/opt | `x`, `residual`:inout?, `w`w → `q`, `scale`, `out_bf16`? |
| `gdn_gated_norm_had_quant_i8` | `M` `n` `head_dim` `group` `eps`:f64 `dtype` `zgrp`/opt `zblk`/opt | `x`, `z`, `w`w → `q`, `scale` |

`wire` is the payload width on the link: 0 is exact bf16, 6 the rotated six-bit form. It is a
parameter for the same reason as `min_bytes`. `x` is inout on `ar_rmsnorm_quant_fp8`: it is both
the message and the input, and the reduced sum lands in it.

The norm-carrying forms are two-rank only. `ar_hc_write` and `ar_gather_hc_write` also have an
exact-wire four-rank form. At other widths the arch blocks issue the plain ops.

`ar_hc_write` is `all_reduce` of `y` followed by `hc_write` of the reduced `y`, for a block whose
output feeds a gated residual rather than a norm; `y` is inout as `x` is above. It serves both
wires, one op per wire: a caller declares the rotated form beside the exact one and issues whichever
the step's message goes out on, and `wire` 6 is 6 at every size. A caller states `wire` 0 on the
exact form rather than omitting it: a row that serves one wire says so with a constraint on `wire`,
and a constraint on an absent key does not hold.

`ar_gather_hc_write` is `moe_gather` followed by `ar_hc_write`: the gather's output is the message,
so the gather half writes `y` before it is reduced in place, and the five gather operands and `act`
mean what they mean on `moe_gather`. `wire` is `ar_hc_write`'s. On either wire it is byte-identical
to the three kernels it replaces.

## Vocabulary edges

| op | params | operands |
|---|---|---|
| `embed_lookup` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt `mult`:f64/opt | `tokens`, `wte`w → `x` |
| `embed_lookup_q` | `M`(range) `n_embd` `n_vocab` `dtype` `vocab_offset`/opt `wscale`:f64/opt | `tokens`, `wte`w, `scale`w? → `x`, `ahead`? |
| `logits_gemm` | `M`(range) `n_vocab` `n_embd` `dtype` | `x`, `lm_head`w → `logits` |
| `ngram_ids` | `M`(range) `heads` `ngram` `eos` | `tok`, `state`:inout, `cu`, `mult`w, `vocab_sizes`w, `offsets`w, `cache_idx`?, `has_init`?, `num_accepted`? → `ids`, `ahead_ids`? |
| `ple_gate` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `gate_eps`:f64/opt | `k`, `q`, `v`, `w_key`w, `w_query`w, `w_conv`w → `gv`, `gvn` |
| `ple_conv` | `M`(range) `n` `width` `dilation` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `resid`?, `cache_idx`?, `has_init`?, `num_accepted`? → `y` |
| `causal_conv1d` | `M`(range) `n` `width` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `cache_idx`?, `has_init`?, `num_accepted`? → `y` |

`logits_gemm` is separate from `gemm_nt` because `lm_head` is its own access class
(`RAD_ACCESS_VOCAB`) and because the sampler fuses against it: under the distributed top-k a rank
never materialises a full logits row (`spec.md` §9, §13). `M` is one row per sampled position,
`gather_rows` having run first. A quantised head (block fp8, int8 per row and 128 of K) reaches a
host kernel joined (Weights of several planes, above).

### `gather_rows`

| op | params | operands |
|---|---|---|
| `gather_rows` | `M`(range) `n` `dtype` | `x`, `idx` → `y` |

`y[i, :] = x[idx[i], :]`; a negative index writes a zero row. Without it a plugin computes logits for
every token of a prefill chunk instead of the last one per sequence: `n_tok / n_seq` times the work,
and at `max_tok` 8192 against a 248K vocab the logits buffer alone is 8 GiB.

### `scatter_rows`

| op | params | operands |
|---|---|---|
| `scatter_rows` | `M`(range) `n` `dtype` | `v`, `idx`, `x`:inout |

`x[idx[i], :] = v[i, :]`, the mirror of `gather_rows`. `x` is inout because the write is partial:
rows no index names keep what they held, so it can target a pool other sequences also use. Two
rules differ from the gather, because the destination is live memory:

- **A negative index writes nothing.** `gather_rows` reads a zero row for one (the padding a short
  prefill chunk leaves); here a zero fill would be data loss.
- **Indices must be distinct**, declared with `RAD_OPD_F_IDX_UNIQUE`. Two rows landing on one
  destination have no defined winner on a parallel machine, so the reference and a kernel could
  legitimately disagree.

The caller is the KV tier traffic (`core/mem/kvtier.h`), where a row is a few hundred bytes and `M`
separate transfers would be all dispatch cost. A paged pool is layer-major, so within one layer a
group's blocks are dense rows of `layer_bytes_per_block()`: packing any set of them is a gather and
restoring them a scatter, one launch per (group, layer) however many blocks move.

At `dtype` u8 the rows are opaque bytes, and libr4d supplies that width as its own kernel pair
rather than widening the bf16 one. What moves is whatever the attention kernel wrote. Calling it
bf16 would matter: `libref` is the oracle for every op here and its bf16 path loads through float,
which need not return a NaN payload unchanged. At u8 every value is 0..255, which float carries
exactly, so oracle and kernel agree bit for bit on any payload.

### `embed_lookup`'s `mult`

`mult` (optional, default 1) multiplies the rows that are found, `x[m] = wte[id] * mult`, rounded
once to `x`'s dtype; a missed row (negative padding or another rank's token) writes zero. Gemma
scales its embedding by `sqrt(hidden)` held in a bf16 buffer, so the plugin passes that value
rounded to bf16 (22.625 at a hidden of 512, not 22.6274) and the product is the reference's bit for
bit. Scaling at the lookup leaves the table plain and tied to the head, as the checkpoint stores
it.

`wscale` is a different thing: the scale `embed_lookup_q`'s layout quantises the stored table with,
read back as the dequant. `embed_lookup` reads none, and `embed_lookup_q` takes no `mult`.

### `embed_lookup`'s `vocab_offset`

`vocab_offset` (optional, default 0) is the first vocabulary row this rank holds. With it, `n_vocab`
is this rank's row count (as it is for `logits_gemm`) and the table covers global ids
`[vocab_offset, vocab_offset + n_vocab)`. Token ids stay global: a rank subtracts the offset and
writes a zero row for an id that belongs to another rank, as it does for the negative padding id, so
summing the ranks reconstructs `x`. Every token lands on exactly one rank, so the `all_reduce` that
follows is declared `exact` rather than taking the model's wire setting: it sums disjoint rows, and
a lossy wire would lose precision on values that are already final.

Without it the embedding would be replicated while `lm_head` is row-sharded. The table is bf16
even in an fp8 checkpoint, so replicating a 248320 x 5120 table costs 2.37 GiB on every card that
the split gives to the KV cache instead.

Under a split the kernel cannot tell a garbage id from another rank's token, so "an id at or above
`n_vocab` fails the launch" holds only where the rank holds the whole vocabulary (world size 1, not
merely `vocab_offset` 0: rank 0 of a split has offset 0 and sees ids past its slice on most steps).
Bounding ids is the caller's job, and the tokeniser and sampler are already bounded.

### `embed_lookup_q`, the n-gram gather

A gather from a table quantised with one scale for all of it: `x[m] = wte[id] * scale[0]`, an absent
`scale` being 1. Padding and bounds behave as `embed_lookup`'s, except that an id past the table
fails the launch in the host kernels and writes a zero row in the device one, which cannot read its
ids before it runs.

One scale suffices because E4M3 carries its own exponent: on Qwen3.8-Flash-Next's table a row's
dynamic range is 2.3x and a per-row scale would improve the error by only 2.9%. The FP8 checkpoint
ships it this way, E4M3 rows with a single bf16 `weight_scale`.

It is a separate op, not a `dtype` on `embed_lookup`, because the operand list differs and
operands are positional; an optional scale operand on `embed_lookup` would also move `x` from slot
2 to slot 3 for every caller.

Where it runs is where `--ngram-placement` puts the table: the planner runs an op on the host when
every weight operand of it is `Site::Host`, and takes that from the table's placement. The table is
51.2 GB and is never staged a unit at a time (moving a whole unit to use 5 KB is the wrong shape),
so it is read where it lies:

| placement | the table | the gather |
|---|---|---|
| `disk` | the container's mapping | libavx's host kernel |
| `ram`, libr4d loaded | a pinned copy every card maps at its host address | libr4d's device kernel, over the link |
| `ram`, no device kernel | a locked copy (`core/format/table_copy.h`) | libavx's host kernel |
| `vram` | the card, every rank holding all of it | libr4d's device kernel |

`hipHostRegister` on the container's mapping is not an option: it would force the whole file
resident.

**Host gather.** libavx keeps the rows a gather has touched in a row cache of its own and reads a
miss from the file with `O_DIRECT` (`libavx/avx_ngram.cpp`); the page cache would map a 4 KiB page
per 160-byte row. Reads keep the alignment the filesystem reports for direct I/O, or its block size
where it reports none (as btrfs does); one read of the table proves it before the first row is
served. A filesystem without direct I/O cannot hold a container with this table on `disk`.

**Device gather.** libr4d's `embed_lookup_q_bf16` handles four entries a thread, decodes each E4M3
code exactly, multiplies by the scale in f32 and rounds to nearest even: the value libavx's lookup
table holds, bit for bit. It reads the ids and writes `x` in the host arena, where the architecture
declares them for the host gather, so the step has no host op left in it.

The table must be a weight operand for the host site to be reachable at all: `gather_rows` takes a
plain input and would never be placed there. The host kernel lives in a production plugin (libavx)
because `rad_gate_reference_kernels` refuses to serve on libref.

### `ngram_ids`

Qwen4-Exp's Per-Layer Embedding gives one layer a second embedding table addressed by a hash of the
last `ngram` tokens. Qwen3.8-Flash-Next's has 320,001,536 rows of 160 (102 GB at bf16, 51 of the
model's 180 billion parameters), read sixteen rows a token. `ngram_ids` computes those row ids:

    mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]
    ids[h]   = remainder(mixed(b+1), vocab_sizes[h]) + offsets[h]

where the `heads` are `ngram - 1` equal blocks and block `b` uses the `(b+2)`-gram. `t_s` is the
token `s` positions back, or the EOS token if the shift would cross one, and once a shift falls
back to EOS every deeper shift does too. `mult`, `vocab_sizes` and `offsets` are weights because the
checkpoint carries them; re-deriving them from the reference's seed and prime search would be a
second source.

**`state` is a rolling window of committed token ids**, `ngram - 1 + n_spec` deep, read at
`num_accepted - 1` and rewritten in `ple_conv`'s layouts (below), including the two that
`num_accepted`'s presence chooses between. A cold window is EOS, not zero, since zero is a real
token id. It is a state rather than a plain input because the contract is already paid for: the
dilated convolution downstream carries a window with this rollback contract, so PLE needs
`num_accepted`, `cache_idx` and `has_init` regardless, while nothing in the engine derives "the ids
before this step" (`DeriveInput` carries positions and cu_seqlens and no token ids, and an input
would need a new core derivation and a wider `DeriveFn`). `RAD_KV_CONV` is already "per-sequence, a
rolling window of width-1 + num_spec entries", and `conv_state_index` already derives its read
cursor.

The gather is `embed_lookup_q`, not `gather_rows`, because the table must be a weight operand for
`Site::Host` to be reachable.

**`ahead_ids` and `ahead` fetch the next chunk's rows a step early.** A prefill chunk gathers `heads`
rows a token, and with the table on `disk` and text the row cache has not seen, most are random
reads of the container file while the card idles behind the host-site op. The next chunk of the
prompt is already known (`RadBatch::n_ahead`, stored in `token_ids` after the step's own tokens), so:

- `ngram_ids` with `ahead_ids` present reads that many tokens past the step's in `tok`, as a
  continuation of the last sequence, and writes their ids there: the same ids the step that runs
  those tokens will compute. The window it writes back is the step's own.
- `embed_lookup_q` with `ahead` present may start reading those rows once its own are gathered. No
  output depends on it, so a kernel may ignore it; libavx's row hands them to a reader thread that
  fills its row cache while the step runs.

### `ple_gate`

PLE's mixing coefficient is a dot product between the n-gram embedding and the residual stream, one
scalar per hyper-connection stream per token:

    K[c] = grouped_rmsnorm(k)[c] * (w_key + wadd)      k is key_proj(embedding)
    Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)    q is the incoming hc stream
    s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
    s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))
    gv[c] = sigmoid(s[c]) * v                          v is value_proj(embedding), one row
    gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)

- The signed square root keeps the sign, so a stream whose embedding opposes the residual gets a
  gate below one half rather than zero. The clamp is on the magnitude and the sign comes from the
  original, which matters at one value: torch's `sign(0)` is 0, so a dot product of exactly zero
  stays zero rather than becoming `sqrt(gate_eps)`.
- Both outputs leave the op: the dilated convolution reads the normed copy and the residual add the
  un-normed one, and recomputing either outside would be a second definition of the norm.
- The convolution branch does not see the gate. `gv[c]` is a scalar times one shared `v` row and the
  norm's rms is per `n`-wide stream, so the gate divides back out: `gvn[c][d]` is `v[d]/rms(v)`
  times its gain whatever the gate did. A fused kernel whose `gvn` varies with the gate is wrong. It
  is not a shortcut, though: `gv` is narrowed to its dtype before the third norm reads it, so
  computing `gvn` from `v` directly differs in the last bits.

It is one op rather than six (three grouped norms, a reduction, a scalar nonlinearity and a broadcast
multiply, all over `[M, hc*n]`, 10240 wide on Qwen3.8-Flash-Next), because six memory-bound launches
would each pay the per-kernel floor; `hc_read` is the same argument at the same width.

### `ple_conv`

    y[t] = resid[t] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation] )

The dilation sets the history depth. At Qwen4-Exp's width 4 and dilation 3 the taps are 9, 6, 3 and
0 timesteps back, so a sequence carries `(width-1)*dilation = 9` timesteps, not `width-1 = 3`. A
cache sized for the undilated width reads zeros where the deepest tap belongs, and the model still
generates fluently from a slightly wrong context.

The state is `[n_slots, n, state_len]`, slot `i` holding the value at relative position
`i - (width-1)*dilation` (`gdn_conv_prep`'s convention with the dilation folded in). `has_init` 0
means that window is zeros rather than whatever the slot last held; without it a fresh sequence on a
recycled slot convolves the previous sequence's tail into its first nine tokens.

`num_accepted` shifts the read offset, and its presence picks what the step leaves behind. The
manager sizes the state `(width-1)*dilation + n_spec` deep so a speculative step can read at an
offset inside it. The next step's offset must find the history ending at the last token this step
committed:

- **Present: a decode step**, whose drafts a verify may reject. The window is rewritten as
  `gdn_conv_update` rewrites its own: the history read, shifted down by one, then every input of
  the step, so slot `i` holds position `i + 1` of (window read ++ step). A next step that kept `k`
  of these tokens reads at `k - 1` and finds the history ending at the `k`-th. It needs
  `(width-1)*dilation - 1` slots plus one a token, which a step of `1 + n_spec` tokens gets.
- **Absent: every token commits**, as in a prefill chunk. The read is at offset zero and the last
  `(width-1)*dilation` inputs land at offset zero, where the next step's `num_accepted` of 1 reads
  them.

The two agree for a one-token step, and no single layout serves both at several tokens: at offset
zero one needs the history ending at the step's first token and the other at its last. An
architecture therefore issues the op once over its decode rows and once over its prefill rows, as it
does the two `gdn` convolutions.

`gdn_conv_prep` and `gdn_conv_update` are separate because the work around their convolutions
differs (one splits a fused projection and takes an l2 norm). PLE's convolution is the same
arithmetic at prefill and decode, so one op takes the union of both contracts: `has_init` is
prefill's question and `num_accepted` decode's.

The residual add is folded in because the reference is `gv + silu(conv(gvn))` and both tensors are
`[M, hc*n]`. `resid` absent leaves the plain convolution.

### `causal_conv1d`

    y[t] = sum_j w[c][j] * x[t - (width-1-j)]

LFM2's short convolution: depthwise, causal, undilated, with no activation and no residual (the
gates around it are separate `mul`s in the architecture). The window and its three optional operands
are `ple_conv`'s at dilation one, so an architecture issues it the same way, once over its decode
rows with `num_accepted` and once over its prefill rows without. libr4d, libref and libavx each
serve both ops from one implementation.

## Sampling

Every sampler is a kernel, not a host lift. `rad-kbench` checks the device samplers against a host
reference that follows llama.cpp's chain (`spec.md` §13).

**Per-request settings are operands, not parameters.** `params` is `[M]` rows of `RadSampleParams`
(`abi/rad_sample.h`), uploaded once a step, carrying `temp`, `top_k`, `top_p`, `min_p`,
`typical_p`, the three penalties, the DRY and XTC settings, the seed and position, the history
window and the grammar's `mask_row`. A parameter is geometry: the selector matches on it and freezes
it at declare (`spec.md` §2.2), so two requests in one batch with different `top_p` would need two
resolutions of one op. The one exception is `sample_chain` (below).

| op | params | operands |
|---|---|---|
| `sample_penalties` | `M`(range) `n_vocab` `vocab_off`/opt | `logits`:inout, `params`, `history` |
| `sample_temp` | `M`(range) `n_vocab` | `logits`:inout, `params` |
| `sample_topk` | `M`(range) `n_vocab` `vocab_off`/opt | `logits`, `params` → `cand_idx`, `cand_val`, `pairs`? |
| `sample_topp` | `M`(range) `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_minp` | `M`(range) `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_mask` | `M`(range) `n_vocab` | `logits`:inout, `params`, `bitmask` |
| `sample_pick` | `M`(range) `n_cand` | `cand_idx`, `cand_val`, `params` → `token` |
| `sample_argmax` | `M`(range) `n_vocab` | `logits`, `params` → `token` |
| `sample_merge_topk` | `M`(range) `n_cand` | `gathered`, `params` → `cand_idx`, `cand_val` |
| `sample_logprobs` | `M`(range) `n_vocab` `k` `vocab_off`/opt | `logits`, `params` → `stats` |

`sample_mask` is where a GBNF grammar lands: the state machine stays on the host, and what crosses
is a token bitmask per sequence (~19 KB at a 151K vocab), cheap enough to upload every step. A set
bit means allowed.

`sample_penalties` and `sample_dry` read global token ids from `history` and write this rank's
logits, so under a split vocabulary they take `vocab_off`, as `sample_topk` does, and a token outside
the shard is another rank's to penalise. Their `penalty_last_n` / `dry_penalty_last_n` window is
llama.cpp's: 0 turns the stage off for the row, and a negative value takes the whole history.

`sample_logprobs` serves an OpenAI `logprobs` field or llama-server's `n_probs`. It runs on the raw
logits, before any stage has touched them, and only for a row whose flags carry `RAD_SP_LOGPROBS`;
other rows are not written. `stats` is `[M, 2 + 2k]` f32: the row's maximum, the sum of
`exp(l - max)` over it, and its `k` largest as (id, logit) pairs, descending, ties to the lower id,
the id written as a float (exact below 2^24) and global under a split vocabulary (`vocab_off`
added). A slot past the vocabulary is id -1, logit -infinity. `k` is geometry, the one width every
request's `top_logprobs` fits in: the engine declares 21, OpenAI's limit of 20 plus one so a row is
44 floats, whole 16-byte words, which the vocab-parallel `all_gather` requires. A token's logprob is
`l - max - log(sum)`; under a split vocabulary the host merges the ranks' rows (maxima and sums by
the usual rescale, pairs by value).

### Further sampling stages

| op | params | operands |
|---|---|---|
| `sample_dry` | `M` `n_vocab` `vocab_off`/opt | `logits`:inout, `params`, `history`, `breakers`? |
| `sample_typical` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_xtc` | `M` `n_cand` | `cand_idx`:inout, `cand_val`:inout, `params` |
| `sample_chain` | `M` `n_vocab` `temp`:f64/opt `p`:f64/opt `min_p`:f64/opt `k`/opt | `logits`, `query`?, `u` → `draw` |

`sample_dry` is llama.cpp's DRY: it penalises, once, each token that would extend a repeated suffix
of the history, by its longest such repeat. Sequence breakers are strings, so they are resolved
against the vocabulary on the host: the row's `dry_rep_limit` carries the repeat limit the most
recent breaker imposes, and the optional `breakers` plane (one byte beside every history token) marks
tokens that are whole breakers and are never penalised. Absent, no token is exempt.

`sample_typical` is locally typical sampling. It is not a prefix of the descending candidate order
(that order and the typicality order disagree), so it narrows in place rather than truncating.

`sample_chain` (device-only) is the whole chain in one kernel for a speculative verify: temperature,
softmax, top-k, top-p, min-p, renormalise and draw over an L2-resident row. `query` is the drafted
token per position (absent at the last), `u` is two uniforms per position, and `draw` is
`{u32 tok, u32 tok_excl, f32 p_query}`: a draw, a draw with `query` excluded, and `query`'s filtered
probability, the three values a rejection-sampling verify needs. Its settings are launch-time
parameters, never constraints. libr4d serves it; the engine's sampler issues the decomposed stages
and does not declare it.

### Vocabulary-parallel sampling

When `lm_head` is row-sharded, a rank's logits row covers only its slice of the vocabulary.
`sample_argmax` is then unusable (its answer is a local column index, a different token), so a greedy
request is staged at `top_k = 1` and takes the chain. `sample_topk` writes the optional `pairs`
output: its candidates as (u32 global id, f32 logit), with the rank's `vocab_off` added, which is
what `all_gather` moves and `sample_merge_topk` reads.

`sample_merge_topk` re-applies the row's `top_k`, which is why it takes `params`. Every rank ran its
own top-k before the gather, so the merged set holds up to `world_size * k` candidates and no later
stage narrows again. At `k = 1` that matters: `sample_pick` is a draw, so a greedy row with two live
candidates would return whichever one the seed selects.

`gathered` is `world_size` sorted runs, and the merge relies on it. Each rank's `sample_topk` writes
its `n_cand` pairs in descending (logit, lower id first) order with its holes at the tail, and the
all-gather lays a row's runs side by side. libr4d's merge therefore runs only the merge stages of
its sorting network when the runs are a power of two long and fill the row (11 passes over two runs
of 1024, where a full sort is 66), and the whole network otherwise. Unsorted runs give a wrong
order, not a slow one.

## Sparse attention (QSA)

| op | params | operands |
|---|---|---|
| `qsa_block_key` | `M`(range) `n` `ratio` `rotary_dim` `pos0`/opt `block_size`/opt `eps`:f64 `theta`:f64 `scale`:f64/opt `wadd`:f64/opt `dtype` `mode`/opt `sections`/opt `yarn_factor`:f64/opt `yarn_low`:f64/opt `yarn_high`:f64/opt `yarn_magnitude`:f64/opt | `k`, `w`w, `page`?, `pos`? → `out` |
| `qsa_score` | `M`(range) `n` `heads` `blocks` `seqs` `dtype` | `q`, `bk`, `bt`, `cu`, `nc` → `score` |
| `qsa_select` | `M`(range) `blocks` `topk` `ratio` `seqs` | `score`, `bt`, `nc`, `pos`, `cu` → `sel`, `nsel`, `seqused` |
| `qsa_tail_store` | `M`(range) `n` `ratio` `ring` `seqs` `dtype` | `k`, `pos`, `cu`, `state`:inout, `sidx` |
| `qsa_work` | `M`(range) `n` `ratio` `ring` `seqs` `work` `dtype` | `k`, `pos`, `cu`, `state`, `sidx`, `bt`, `rope`? → `stage`, `page`, `bpos`, `nc` |

Five ops decompose one indexer, which scores blocks of key history against the query and hands
`attn_paged` a shorter block table. `docs/QSA.md` describes the decomposition and
`arch/common/rad_qsa.h` issues all five. libref implements all five (the oracle for libr4d's
kernels), and libavx all five for the host path.

### `qsa_block_key`

The compressed key: one vector per `ratio` tokens of key history, against which the indexer's block
scores are taken.

```
out[b] = rope( rmsnorm( mean over j<ratio of k[b*ratio + j] ),  at position pos0 + b*ratio )
```

The rotation is `rope`'s NeoX ladder over the first `rotary_dim` lanes, with `rope`'s YaRN
coefficients when given, so a compressed key and the keys it summarises are rotated alike. With
YaRN, `scale` must be 1.

- **`M` is blocks and `k` is `ratio` times taller.** Every other op here has one output row per
  input row. A caller that sized `k` at the block count would pool rows of the next block into each
  one, so libr4d's entry point checks it.
- **The mean is accumulated in f32 and rounded to the activation dtype before the norm**, the
  reference's order (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`). Carrying f32 into the
  norm gives a small, systematic difference on every block of every layer.
- **`block_size` equals `ratio` and exists to size a KV group.** `Builder::find_kv_consumer` binds a
  group to the first unclaimed op declared after it that carries the key `block_size`, and the
  block-key store's page is the compress block. libr4d refuses a disagreement.
- **`out` may be a KV cache, and is then rank four.** The block-key store's geometry is a container,
  not an attention shape (`kv_heads` 1, `block_size` the compress ratio, `head_dim` `n/(2*ratio)`),
  so a page holds exactly `n` contiguous values and the manager allocates one per `ratio` tokens.
  libr4d checks the product of the extents rather than the rank, which is the engine's business.
  `qsa_score`'s `bk` is the same object, read the same way.
- **`page` changes where the answer goes, not what it is.** The block-key store is a paged KV group
  (`docs/QSA.md`), so consecutive blocks are not consecutive pages. Absent, the write is contiguous
  (a caller with its own scratch, or a test). Present, `M` comes off `page`, not `out`, because
  `out` is then as tall as the pool. A `-1` entry is a block with no page and is skipped.
- **`pos` turns a launch from a range into a list; `pos0` is the range.** Within one sequence block
  `i`'s first token is at `pos0 + i*ratio`, which describes a prefill chunk. At decode the blocks
  completing in one launch belong to different sequences at different positions, so with `pos`
  given a launch is a set of work items and one launch a layer a step serves any mix.
- **Under a multi-component rotary** (M-RoPE) a block key rotates at its first token's three
  components. `qsa_work` takes the step's `rope` planes (`RadBatch::rope_pos`, `[3, T]`) and writes
  `bpos` as `[work, 3]`, a block's three components side by side: a first token in the step reads its
  own components, and one committed in an earlier step (always text, since a prefill chunk ends on a
  whole block and a prompt ends in text) takes its index plus the shift that sequence's first row in
  the step carries. `qsa_block_key` with `mode` (`mrope`, `imrope`) and `sections` then reads the
  component each frequency's section names, as `rope` does.

A block key is query-independent and permanent once its `ratio` tokens exist. The reference
recomputes it inside its per-query loop; the engine computes it when the block's last token arrives
and caches it, so the indexer costs 64 bytes a token a layer instead of a pass over the whole
history every step.

### `qsa_score`

```
score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
```

- **`m` is a query and `bt` is per sequence; `cu` joins them.** At decode there is one query per
  sequence, but a prefill chunk is many queries over one history, each selecting its own set.
  `cu_seqlens` is the only operand that says which sequence a token belongs to.
- **`nc` bounds the block loop and is required.** `blocks` is a declared width sized for
  `max_ctx / ratio`, so a scorer that trusted it would read the whole declared width of padding on
  a short prompt. `nc[seq]` is the sequence's live complete-block count: an upper bound for every
  query in the chunk, within `T / ratio` of each one's own. `qsa_select` narrows it per query, since
  it has `pos`.
- **The caller narrows `score` to the live reach**, because a grid cannot read `nc`: both ops size
  their launch from the declared width (the live count is device data, the grid a host number). The
  architecture narrows the score operand's column count to
  `(max_ctx_len + n_tok + n_spec + ratio) / ratio + 1`, which the batch carries on the host, and the
  pitch stays the declaration's. Every term is deliberate slack: an underestimate drops blocks from
  the selection silently, and the `+ 1` keeps `n < blocks`, which makes `qsa_select`'s partial page
  `bt[n]` readable.
- **The relu is per (block, head) and the sum over heads**, in that order: independent votes, each
  of which can only argue for a block. Summing first would let one head's large negative cancel
  another's positive, a different model that still ranks blocks plausibly.
- **`bt` holds page ids, not block indices.** The attention KV group is paged at the compress ratio,
  so a block and a page are the same thing: the sequence's block table is already the list to score,
  and the top-k of the output is already the table the attention reads.
- **A negative table entry scores `-inf`, not zero.** A table is padded to its declared width and a
  real score can be zero (every head vetoed the block), so zero would let padding outrank it.

### `qsa_select`

Turns the scores into the table the attention reads: the `topk` highest-scoring blocks as page ids,
with the partial page appended.

- **`nc` is separate from the table's width** because the two differ by the partial page, which the
  op treats specially. `bt[n]`, when not `-1`, is the partial page: the tokens after the last complete
  block, which the reference attends unconditionally and which the query itself lives in. It is
  appended after the selection and never a candidate for it, so `sel` is `topk + 1` wide; sized at
  `topk` it would drop the newest tokens on three queries in four.
- **The bound is the query's**: `n = min(nc[seq(m)], (pos[m] + 1) / ratio)`, because block `b` is
  entirely in this query's past exactly when `b*ratio + ratio - 1 <= pos`. At decode the two
  bounds agree. `cu` is the join, as in `qsa_score`.
- **The partial page exists exactly when `(pos + 1) % ratio != 0`.** This is a causality rule. A
  decode query on a block boundary has `n == nc`, and `bt[nc]` is the page after the sequence's
  last, which the KV manager leaves `-1`, so testing the page would work at decode. Mid-chunk it
  would not: the same query finds later tokens of its own chunk already stored in `bt[n]`, a page
  entirely in its future, which `seqused` would then cover. The model would attend to its own
  future, which shows up only as an implausibly good prefill.
- **The partial page goes last.** Every selected complete page is entirely in the query's past, so
  the only causal masking left is inside the final page, and placing it last lets the attention's
  own `kk <= q` bound do it with no change to the attention kernel. The effective query position
  the caller passes is `(nsel - 1) * ratio + pos % ratio`.
- **Ties break to the lower block index, through a prefix count.** A block score is a sum of relus
  over a shared grid, so equal scores are ordinary, and a top-k placed by whichever atomic landed
  first would not reproduce across runs; the tree checks generated text byte for byte.
- **`seqused` is why QSA needs no new attention kernel.** `attn_paged` takes its block table and
  per-sequence length as ordinary operands, so the sparse attention is that op issued with `sel` in
  place of the sequence's table and this number in place of its length. The KV it walks is `nsel`
  pages of `ratio` with the partial page last, so the query sits at `(nsel-1)*ratio + pos%ratio`
  and the length is one past it. Without a tail, `pos%ratio` is `ratio-1` and this collapses to
  `nsel*ratio`, every selected page fully visible. The op emits it because it is one store off a
  number it already has.

libr4d's kernel takes scratch (sized by the row's scratch hook; it is not an operand). With it, a
launch of at most 32 query rows cuts the block list into up to 16 chunks, takes each chunk's own
top-k and merges them, so a decode step uses more than `1 + n_spec` workgroups. The result is exact,
because a chunk's local top-k contains every global top-k member of that chunk, and identical in
output layout too: the attention sums the selected pages in table order and float addition is not
associative. Without scratch, or above 32 rows, one workgroup per query walks the whole list.

### `qsa_tail_store`

Parks the raw index keys of the tokens that have not completed a block yet:
`state[sidx[s][0]][pos[t] % ring] = k[t]`. A block key is the mean of `ratio` consecutive raw keys,
so at decode most of a block's keys come from earlier steps and something must carry them.

- **The tail is a linear KV state, not a plane indexed by the batch row.** It is read on a later step
  than the one that wrote it, and it has two axes a buffer cannot express: the layer (the keys are
  `index_qk_proj`'s output, a per-layer weight) and the sequence. The batch row is step-local (the
  scheduler builds a step from the running set in priority order and stable-sorts decode first), so a
  request that outlives an earlier one would shift down a row and inherit its keys. A linear group
  gives each sequence a slot for its life, zeroes it when handed out, and saves and restores it with
  the rest of the linear state when the prefix cache checkpoints. `state` is the group's cache for
  one bound layer, `[slots, 1, ring, n]`; `sidx` is the group's `state_index` row, column 0 the
  committed slot.
- **`cu` finds a token's row**; `slot_mapping` is a flat KV slot and carries no sequence. Which state
  that row belongs to is `sidx`, a different question.
- **The ring is `ratio + n_spec`.** A verify step runs `1 + n_spec` positions and the next step
  restarts at the first rejected one, so a position this step has already overwritten may be owed to
  the next. At `ratio` 4 and depth 3, the step covering positions 9..12 writes rows 1, 2, 3, 0, and
  row 0 is what position 8 is still owed, because the block holding 8..11 is rebuilt as soon as the
  following step starts at 10. The widest window a step can still owe is
  `[p0 - ratio + 1, p0 + n_spec]`, so a ring that long overwrites a row only with a position past
  the end of any step that could ask for it. With `n_spec` 0 the ring is the ratio.
- **Only the last `ring` tokens of a sequence's chunk write.** Positions `p` and `p + ring` share a
  row, so writing every token of a prefill chunk would have workgroups racing for one row. With the
  restriction every row has at most one writer, and the result is what a serial last-writer-wins
  loop leaves.
- **`state` is inout, and rows nothing targets keep what they held**: at decode a handful of rows
  are written and the rest are what the next block key needs.

### `qsa_work`

The gather that turns a batch into the flat block list `qsa_block_key` reads. Nothing in a batch
provides that list: the keys of a block completing at decode are mostly rows of the tail state plus
one of this step, those of a block completing inside a prefill chunk are rows of the chunk, and a
block straddling the chunk's start is some of each, while the blocks belong to different sequences
at different positions.

So this op walks the sequences, works out which blocks became complete, copies each one's `ratio`
raw keys into consecutive rows of `stage`, and writes its destination `page` and first-token position
beside it. The block-key kernel never learns that there were two sources. The copy is a kilobyte
per completing block (512 KiB per layer for a 2048-token prefill chunk); the alternative is a
block-key kernel with two sources, a per-key source selector and a tail layout.

- **`M` is tokens and `work` is blocks**: one block per `ratio` tokens plus at most one straddling
  block per sequence. `work` is a host bound on how many blocks can complete, not a count of how many
  did (that is device data); slots past the count get `page = -1`, which `qsa_block_key` skips, so
  the bound costs empty workgroups, never a wrong answer.
- **`bt` is the block-key group's table**, not the attention group's (`docs/QSA.md` explains the
  two).
- **`state` and `sidx` are the tail**, read at row `q % ring` of slot `sidx[s][0]` (see
  `qsa_tail_store`). A sequence with no slot (`sidx[s][0] < 0`) gets `page = -1` on its blocks.
- **`nc` is complete blocks per sequence**, `(last position + 1) / ratio`, which the scorer and the
  selection both need and this op computes on the way. It is not the number of blocks that completed
  this step; the two agree only for a sequence that starts at zero.

## Draft heads

A drafter is an ordinary graph over ordinary ops except at its two ends: what proposes the
candidates, and what enters the head from the trunk.

### DFlash2

| op | params | operands |
|---|---|---|
| `dflash_select` | `M` `steps` `top_k` `rank` `n_vocab` `anchor_stride`/opt `dtype` | `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w → `tokens` |

`dflash_conv` is the drafter's grouped convolution (coefficient add and every tap in one pass):

| op | params | operands |
|---|---|---|
| `dflash_conv` | `T` `hidden_size` `taps` `group_size` `NG` `block_size` `dtype` | `x`, `delta`, `base`w → `out` |

    out[t][h] = sum over taps j with j == 0 or t % block_size >= j of
                (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]

Blocks are `block_size` rows starting at row 0, so a tap never reaches into the previous block.
`delta` is `[T, taps, NG]` read at its own row pitch: the architecture passes a slice of the
`[T, 2, taps, NG]` coefficient projection. `out` must not alias `x`. libr4d's kernel serves
`taps` 2, `group_size` 16 and a power-of-two `block_size`; libref and libavx any.

`dflash_select` is the greedy walk over the candidate lattice: `cand`/`unary` are `row_topk`'s two
output planes at one row per draft position, `hp` the selector's rank-256 query, and `anchor` the
step's own token ids (`anchor_stride` is the block width, because a draft pass carries `block` ids a
sequence and the anchor is the first). `tokens` is the draft, which the engine reads by name rather
than through the sampler. Only the K edges leaving the previous pick are scored, because the walk is
serial.

`hp` absent is a plane of ones, which turns the edge score
`unary + Σ_r pred[prev][r]·hp[l][r]·succ[cand][r]` into the plain low-rank bigram
`⟨pred[prev], succ[cand]⟩`. That is DSpark's Markov head: a rank-256 learned bigram bias added to
the draft logits, decoded greedily, with the predecessor read after the previous step's argmax
inside one workgroup. Written literally it is `steps` full-vocabulary `[rank, n_vocab]` matrix
products; through this operand it is a codebook gather. The only approximation is the top-K
candidate set, which can cost acceptance and nothing else.

### `mtp_enter`

| op | params | operands |
|---|---|---|
| `mtp_enter` | `M`(range) `n` `hc` `eps`:f64 `dtype` `wadd`:f64/opt `ngroup`/opt `fc`/opt | `h`, `e`, `w_h`w, `w_e`w, `fc_h`w, `fc_e`w → `x` |

    ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
    fe[i]    = fc_e[i] . ne
    rs[k]    = rsqrt(mean(h[k*n .. k*n+n)^2) + eps)
    nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
    x[k*n+i] = fc_h[i] . nh[k] + fe[i]                         for each of the hc sub-streams

The MTP head's entry when the residual stream is wide. A pre-norm MTP head (Qwen3.5, Qwen3-Next,
DeepSeek-V3) joins the trunk's hidden state and the next token's embedding with one `fc` over their
concatenation, and `arch/common/rad_block_mtp_fp8.h` runs that as two `rmsnorm`s into the two
halves of one buffer and one `gemm_nt`. Qwen4-Exp's head cannot: its hidden state is the wide
stream, and its checkpoint carries `mtp.fc_hidden` and `mtp.fc_embedding` as two square `[n, n]`
matrices beside a `[hc*n]` gain and an `[n]` one. So the hidden half applies the same matrix to each
sub-stream (the reference's `.unflatten(-1, (hc_count, hidden_size))`), and the embedding half is
one `n`-wide product added into all `hc` of them, the broadcast the trunk enters the stream with
(`hidden_states.repeat(1, 1, hc_count)`).

It is one op because the unfused form cannot be written: every intermediate is a different view of
the same bytes (`nh` is `[M*hc, n]` to the GEMM and `[M, hc*n]` to the norm), and
`core/runtime/issue.cpp` narrows only dim 0 or the last dimension with every stride left alone, so a
row-pitch change is inexpressible. It also saves launches in a draft round, whose latency the head
exists to hide.

`fc` is the two square matrices' stored format: `bf16`, or `e4m3` rows with f32 128-column group
scales in each row's own tail (the geometry of `hc_read`'s `mix_down`). Only the drafts see it (the
head proposes and the trunk verifies), so it can cost acceptance, never a token. A host kernel
reads the codes and f32 scale planes joined (Weights of several planes).

`ngroup` is the one thing the checkpoint does not say. A `[hc*n]` gain fits both a grouped norm
(`hc` rsqrts, one per sub-stream, as `Qwen4ExpTextGatedResidual` does at every connection) and a
flat one (one rsqrt over the whole wide row). Both readings are self-consistent, libref shares
whichever the graph picks, and no per-op oracle can tell them apart. The empirical answer is cheap:
a head wired correctly accepts most of what it proposes, and one wired wrongly proposes noise while
the model's own output stays correct. The head declares the grouped reading (`ngroup = hc`), and the
parameter makes a flat reading (`ngroup = 1`) one declaration away rather than a different kernel.

---

## Adding an op

Add the schema row to `libref/`, write the naive host implementation beside it, and add a row to
this document (`ops_doc` fails on an op a plugin declares that no table lists). No core file
changes. An op added without a ref implementation has no fallback and no oracle; say so in the
commit, because that is a decision, not an oversight.
