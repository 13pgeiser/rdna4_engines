# Flash-Next NPU trunk: CPU preparation

The Qwen4 `npu_trunk` module owns CPU-only model preparation for hipfire 0.4.2;
`pm-npu` owns the numerical reference, panel layout and deployment builder.
Neither preparation path opens hardware or schedules prefill.

`NpuTrunkMode::from_env()` reads `HIPFIRE_QWEN4_NPU_TRUNK` through
`hipfire_config::developer_var`: `off` (default), `auto`, `on`, or `emulate`.
Parsing a mode does not arm a route.

## Shapes and masks

| Shape | Fused family order and output columns | Full padded columns / panels |
| --- | --- | --- |
| `GdnIn` | qkv 10240, z 6144, a 48, b 48 | 16640 / 13 |
| `QsaIn` | q 12288, k 512, v 512, idx 640 | 14080 / 11 |

`FamilyMask` compacts selected families in this order. `ColumnMap::parts`
records each retained family's fused start and width. Padding to 1280-column
G80 panels is zero; padding has no family and its weight scale is zero. An
empty mask produces no panels. Matrix references are supplied in fused family
order; unselected families may be omitted. Retained matrices must have their
family's exact row count and K=2560.

## Numerics and packing

`pack_matrix` accepts MQ6G256V2 rows: each 256-value group is 200 bytes, with
little-endian f16 s0/z0/s1/z1 followed by 192 payload bytes. Decoding uses
separate f32 multiply and add. Row scales are `((absmax as f64)/127.0) as f32`;
W8 codes use f32 division, ties-to-even rounding and clamping to [-127,127].
Zero-scale rows become zero codes. Direct `pack_matrix` calls panic on invalid
length or a K not divisible by 256; shape/trunk packers return descriptive
errors after validating the matrix metadata.

`pack_shape` transposes W8 into panels and invokes `ArrayDesign::pack_b`
directly. `PackedB::ws` contains the padded fused scales.
`pack_trunk` preserves layer order and stores per-layer shape/panels and scales.

The ordinary pack functions select `EpilogueContract::I32Fold`.
`Bf16Direct` rounds each raw weight scale upward with `pm-npu`'s public
`grid15_up`, quantizes with that exact grid scale, and returns padded `(Sm,Se)`
sidecars in `PackedB::scales` as well as embedding them in each packed panel.

## Programs

`build_programs(contract, rows_max)` selects one G80 I32 or direct-BF16/Fast
design per shape, via `pm-npu::kernels::gemm_g80::build_panel_program`.
M is rounded up to 256, N=1280, K=2560; `rows_max` must be 1..=65536.
`build_programs_for_families` uses the same compacted maps as weight packing.
Each shape record reports panel count, padded width, per-submit A/B/C argument
sizes and per-submit host bytes read/written. The designs include PDI and TXN
bytes; panels reuse their shape's design.

`build_rowmajor_programs(rows_max)` builds full-family BF16-direct designs with
col0=0 and fused pitches 16640/14080. Arg2 declares the complete fused arena;
each retained panel command advances its C base by `2*panel*1280` bytes.

## CPU checks

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo test --release -p hipfire-arch-qwen4 -j 12 --test npu_trunk_packing --test npu_trunk_programs --test npu_trunk_simulation
```

The test suite includes independent MQ6/W8 oracles, a copied lab CPU twin,
column-map/mask arithmetic and a functional G80 simulator test. The ignored
`model_sanity` test uses a CPU HFQM reader and prints weight-scale statistics
and pack time for layer 0 GDN qkv and layer 3 QSA q from the pinned
`qwen3.8-flash-next-gptq3.mq4` artifact (recorded MD5 prefix `be007fc3`).
Set `HIPFIRE_NPU_TRUNK_TEST_MODEL` to override its default model path and run
`cargo test --release -p hipfire-arch-qwen4 --test npu_trunk_model_sanity -- --ignored --nocapture`.

## G80 direct BF16 (simulator only)

The epilogue uses no floating operations: 16x16 multiplies construct
`J=I*Sm*Dm` in acc64 by Horner limbs, then an IEF15-derived highest-bit ladder,
sticky-preserving normalization and one `conv_even` SRS emit 32 BF16 values
at a time, descending into C's upper half `[0x5000,0xa000)`.
IEEE ties, carries, gradual underflow, signed underflow and infinity are
included; an exact zero product is positive zero.

Existing A/B DMA rings receive larger chunks, with no new channels or BDs:

* A half: 4096 code bytes + Dm[128] u16 + De[128] i16 = **4608 B**.
  Both E/O halves repeat full token scales for every K64 chunk.
* B tile: 5120 code bytes + Sm[96] u16 + Se[96] i16 = **5504 B**.
  The last 16 feature entries are zero padding for aligned loads.
* Core A slots: `0xa000/0xb200`; B: `0xc400/0xd980`; epilogue scratch
  `[0xef00,0x10000)`. The last input sidecars are captured before returning
  the input locks, so next-wave lookahead cannot overwrite them.
* Memtile C: one whole 20480-B stage per vertical pair at `0x4000/0x9000`.
  Finite row0-drain hands EMPTY to row1, and row1-drain returns it to row0;
  no row-order race or lookahead credit. A slots `0xe000/0x10000`, B
  starts `0x12000`, resident end `0x7d800` at K2560. The low lean-ring
  scratch remains untouched. C tasks are finite for any M-wave parity.
* 18 memtile BDs on injecting columns, 14 otherwise; 5 core BDs, unchanged
  channel counts and bank assignments. B is read once per panel, not per wave.

For W M256 waves and kc K64 chunks, shim A is `4*W*kc*4608`,
B is `8*kc*2*5504`, C is `8*W*4*20480` bytes. At 8192x1280x2560:
read **27,115,520 B**, write **20,971,520 B**, total **48,087,040 B**,
**0.895690918 B/kop** (declared shim bytes, not physical DRAM measurements).
Repeating sidecars is the explicit traffic tradeoff for retaining the existing
input ring machinery; pair-stage serialization makes the resident panel fit.

`npu-gemm M 1280 K --variant G80 --epi bf16` supplies deterministic grid
scales and compares checked outputs to `pm-npu::kernels::w8a8_epi::bf16_direct`.
No silicon claim: new arithmetic, DMA layout and simulator timings require
hardware validation before enabling serving. The int8/I32 PDI/TXN hashes for
M256/M8192, N1280, K2560, Fast/Slow are pinned in `golden_bytes.rs` to the
pre-BF16 images.

At one M256 wave, K2560 (functional simulator, Fast), the executed PDI/TXN
models **113,393 / 140,888 / 209,386 cycles** for int8 / I32 / BF16.
Scaling measured int8 **33.1 TOPS** by those cycle ratios gives **33.1 /
26.6404 / 17.9253 TOPS**, respectively: estimates, not hardware measurements.
The BF16 arithmetic group issues in **223 cycles per32 outputs**. This
integer path meets the host-byte target but predicts a substantial throughput
loss; it does not establish the hardware throughput promotion gate.

Two-wave simulator totals at K2560 are **171,948 / 226,923 / 357,502**
ticks (int8/I32: M512; BF16: M300 padded to the same M512 deployment).
Subtracting the one-wave totals removes startup/tail: incremental wave
costs are **58,555 / 86,035 / 148,116** ticks. Scaling 33.1 TOPS by these
incremental ratios estimates **33.1 / 22.5277 / 13.0855 TOPS**.
These derived warm-wave numbers, not the optimistic cold-submit ratios,
are the relevant throughput warning for a resident multi-wave trunk.

### Fused row-major BF16 drain

`gemm_g80::build_panel_program_rowmajor(rows_max, pitch, col0)` writes
`C[t][col0+n]` at byte `2*(t*pitch+col0+n)`; arg2 is
`ceil(rows_max/256)*256*pitch*2` bytes. Pitch and col0 must be even.
The complete padded panel is written (zero padded tokens), but no other
columns or rows beyond the padded extent are touched. Native-tile BF16
M256/M8192 PDI/TXN hashes are pinned alongside the unchanged int8/I32 pins.

No retile or additional DMA transfer: the memtile MM2S drain uses four
inner-first `(step,wrap)` dimensions in 32-bit words:
`(1,4), (32,10), (4,8), (320,0)`. These read the native 8x8 blocks in
row order. The two shim tile BDs on each C channel then scatter each
128x80 tile with `(1,40), (pitch/2,0)`; transfer length is 5120 words.
Fields are `Bd::memtile_words` words 2..5 (17-bit step-minus-one,
10-bit wraps) and `Bd::shim_words` words 3..5 (20-bit step-minus-one,
10-bit wraps), transcribed from aie-rt
[`xaie2pgbl_params.h`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/global/xaie2pgbl_params.h)
and [`xaie_dma_aieml.c`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/dma/xaie_dma_aieml.c).
Unused outer wrap is zero; length terminates the outer dimension.

At pitches 14080/16640, a 256-row wave step exceeds the shim's 20-bit
stride/iteration field. The TXN therefore SYNCs both C channels per wave
before rewriting/requeueing the same two-tile BD chains with the next row
base. A/B are queued once for the entire submit. This supports all 256
waves without descriptor/queue overflow; firmware command overhead is not
covered by the phase-sum cycle estimate.

Build one design with col0=0 per pitch/row bound. All 11 QsaIn / 13 GdnIn
panels can reuse its PDI and program; advance arg2's base by `2*panel*1280`
through `append_run_body` or `append_lean_run_body`. Retain the full fused
arena behind the offset, and enforce `panel_col0+1280 <= pitch`.
Alternatively the explicit col0 builder changes DDR patch offsets only.
Shim traffic remains A `4*W*kc*4608`, B `8*kc*2*5504`, C
`8*W*4*20480` bytes: the phase-sum estimate is unchanged from native tiles.
The exercised functional simulator reports identical row-major/native ticks:
M200 K64 **102066**, M513 K64 **289334**, M200 K2560 **209386**,
M300 K2560 **357502** (pitches 2624/16640/14080/2624 respectively).
These are modelled timings, not firmware latency or silicon measurements.

Silicon command (not run by this change):

```sh
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 \
  --rowmajor-pitch 16640 --rowmajor-col0 1280 --iters 3 --reuse-ctx
```

The CLI checks every live value against `bf16_direct`, zero padded rows,
and poison bytes outside the panel window. The row-major drain is proven on
Halo at 15.05 TOPS; the serving cutover still requires the parent-run gates below.

## GPU Bf16Direct glue

`rdna_compute::npu_trunk` exposes an explicit `NpuBf16Layout` (padded rows,
K64-compatible K, panel count) and native `NpuTrunkOutput` ranges. S1 maps bind
destinations with `ColumnMap::gpu_output`; no fused output tensor is needed.
The producer writes every packed A byte, including zero padding, directly to a
device tensor or a device-accessible host-registered arena (`unsafe ..._ptr`).
Compact scale sidecars are four bytes per token: LE u16 mantissa then LE i16
exponent. Weight sidecars use the same encoding, indexed by fused column.

`npu_trunk_emulate_bf16` reads exactly that packed A and row-major S1 W8 codes,
uses caller-owned I32 `[rows,n]` scratch, then applies the lab's all-integer
single-rounding BF16 epilogue into native outputs. The consumer reads panel-major
G80 BF16 tiles without rounding; F32 stores are exact `bits << 16`. The producer
requires finite F16 inputs, as does the screened lab contract. Geometry may be
larger than the live token count; it must equal the submitted NPU program.

The signed-IU8 WMMA GEMM and BF16/F32 scatter are certified PM gfx1151 kernels
(`hipfire-isa emit --kernel npu_trunk`, committed image and M7 receipt).
Activation grid `ceilf` and epilogue unsigned CLZ currently use hipcc, with the
two missing ISA-table operations logged in release-0.4.2/PM-GAPS.md. This is an
opt-in primitive API only: no scheduler, forward hook or NPU device open.

Halo GPU byte oracle (do not hide the GPU for this command):

```sh
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo run --release -p rdna-compute --features lab -j 12 \
  --example npu_trunk_glue_check
```

It compares packed bytes/compact scales to the CPU quantizer and G80 packer,
every native output to exact CPU integer dots plus `bf16_direct`, and synthetic
tile scatter including nonfinite/signed-zero bit patterns. Full GDN/QSA maps,
both output dtypes, 512/8192/ragged-4647 tokens and poison tails are checked.
Repeated deterministic input/weight rows cache CPU dots, never GPU answers.
Typed upload shapes count elements (including guard elements), not bytes.
The oracle checks shape/dtype byte counts against every allocation and before
every download, and emits a phase label before launches and comparisons.
It prints GPU-event timings for 8192-token K2560 pack, both full-width consumers
and integer GEMMs; any mismatch exits 1. CPU layout/map tests:
`cargo test --release -p hipfire-arch-qwen4 --lib npu_trunk::gpu::tests -j 12`.

## Stored calibration and deterministic planning (0.4.2)

`Calibration::{load,store,validate,key_matches}` uses schema version 2 JSON.
All fields are required; missing inputs, unknown fields, invalid numeric costs,
unknown versions and inconsistent epilogues are errors. `CalKey` consists of
`device.{arch,pci_bdf,npu_fw_version}`, caller-supplied `build` (the measured
daemon git SHA/binary identity, not the calibration CLI's identity),
`kernel.{uname_release,perfopt_armed}`, `model_sha256` (64 hex digits),
`epilogue` (`bf16` or `i32`), and `trunk_route` (`stock` or
`overlay:<marker>`, e.g. `overlay:gfx1151-zab-v1`). A key mismatch returns the
first mismatching field in that order. The load caller must log `npu-trunk: off (reason)` and
leave the route off; loading never recalibrates or opens hardware.

`plan(&Calibration, FamilyMask, rows) -> Plan` and
`crossover(&Calibration, FamilyMask) -> Option<usize>` port the reference
`npu-trunk/crossover_model.py`, SHA256
`7e7a23afb9df5546358a25141c06006c0bfc20d292ef2274ddb8882612192096`.
The planner uses an 8192-row chunk DP, GPU-only/pair choices, 64-aligned
tail rebalance, even re-banking with boundary tax, and a taxed single-chunk
split. `Step::Gpu { chunk }` and `Step::Pipe { a, b }` carry exact half-open
prompt row ranges, including every pair of a re-banked plan. GPU wins ties;
existing earlier pairing wins ties against rebalance/re-bank. `min_bank_rows`
has the reference's placement (rebalance, single split, and re-bank eligibility).
GPU costs distinguish 36 GDN and 12 QSA layers and prefix-dependent attention;
NPU time rounds fused columns up to `panel_cols`. Contention is a burst term
`(alpha_overlap - 1) * min(G_other, N_this)`, not whole-prefill slowdown.
GPU segments include pack once per moved layer shape and consume on each
selected family's live (not panel-padded) columns. `consume_mode=retile`
charges `rows/8192 * cols * consume_ms_per_col_8192[out_dtype]` per layer;
`direct` charges zero. Consume is GPU work, not NPU time or fill.

`margin` is a fraction in `[0,1)`. Every candidate must cost strictly less
than `(1-margin)` times the GPU-only cost of the rows it replaces, including
tax for splits/re-banking. At nonzero margin every re-bank pair must also pass
its local margin test. Margin zero preserves the Python reference. The crossover
is a **diagnostic**, not a hard-coded activation threshold: it scans
1024..=49152 by 64 and finds the first point whose entire remaining scan
suffix beats GPU-only. Route choice always comes from the prompt's plan.
No live clock, throughput measurement, or previous request time is consulted.

The measured families are grouped as `qkv`, `zab`, `q`, `kvidx`. Whole `zab`
and `kvidx` groups map to the existing family mask; a partially selected group
has no calibrated attribution and stays GPU-only. Calibration column counts
are used verbatim (the legacy lab examples differ from current packing widths).
Never apply a legacy example key to a production model.

`select_families(&Calibration) -> (FamilyMask, SelectionReport)` evaluates all
15 nonempty subsets with `plan` at 16384, 32768, and 65536 rows. The winner
minimizes mean `T_pipe/T_gpu`; exact ties prefer fewer groups, then the sorted
family-name sequence lexicographically. That winner must strictly beat GPU-only
by `margin` at 16384, otherwise the mask is empty (no runner-up substitution).
The report includes all subset times/ratios and the pre-gate winner for logging.
A zero-cost GPU baseline has ratio 1 and cannot pass the strict gate.
Python prints the same table with
`python3 crossover_model.py cal/g80bf16-overlay-v2.json select`;
an optional final `retile|direct` argument overrides consume mode without
changing the stored record.

`load_log_line` takes a `LoadLogKey { current, mode, families }`, crossover,
and plans at 16384/65536. It checks the key, hashes canonical serialized
calibration JSON, and logs the INTERFACES.md §4 fields, with times as
pipe/GPU milliseconds. `request_log_line(rows, &plan)` prints e.g.
`npu-trunk: rows=16384 plan=[pipe 0..8192+8192..16384] predicted=-14.7%`.
The key/hash plus stored calibration identify every deterministic model input.

### Assembling measurements

```sh
hipfire calibrate-npu-trunk --from measurements.json --out cal.json
```

This command assembles and validates an already measured record; it does **not**
perform measurements, probe hardware, fit contention, or silently fill defaults.
The measurements schema is the same as the schema-version-2 output record:

* `schema_version: 2`, `margin` and the complete current `key` described above.
  The collector supplies device/firmware, PerfOpt, measured build, model digest,
  epilogue, and trunk-route identity. The CLI additionally checks `uname -r` against the
  collector's key and refuses a kernel mismatch. The eventual load caller must
  compare **all** key fields to its actual runtime identity.
* `gpu.trunk_in_ms_per_8192`: required `qkv`, `zab`, `q`, `kvidx` millisecond
  totals across the respective 36/12 layers from kernel-trace attribution.
  `gpu.uniform_per_layer_ms`, `gdn_core_per_layer_ms`, `qsa_per_layer_ms`,
  `qsa_growth_per_layer_ms_per_8192_prefix`, `fixed_per_chunk_ms`,
  `boundary_tax_ms`, and `min_bank_rows` are required.
* `families`: those four names, each with `layer` (`gdn` for qkv/zab,
  `qsa` for q/kvidx), positive fused `cols`, and `out_dtype` (`bf16` for qkv,
  `f32` for the other families in today's serving route).
* `glue.pack_ms_8192`, `fold_ms_per_col_8192`, `sync_ms`: measured
  pack/fold/sync costs (fold is charged only for the I32 contract).
  `glue.consume_mode` is `retile|direct`; the required
  `consume_ms_per_col_8192` object contains finite nonnegative `bf16` and `f32`
  costs. Direct mode keeps those measurements but charges zero consume.
* `npu.tops_corun`, `bytes_per_kop`, `epilogue`, `panel_cols`: measured
  co-running rate, measured or declared traffic, chosen output contract, padding.
* `contention.alpha_overlap` (>=1) and `model`: fitted/measured overlap
  slowdown and the provenance/fit description. The planner consumes the alpha;
  it does not reinterpret the descriptive model string.

Collect once per exact device/build/kernel/PerfOpt/model/epilogue/trunk-route tuple.
The lab procedure is in `/home/kaden/qcal/release-0.4.2/npu-trunk/`:
`ptrace.sh` attributes GPU kernel costs by layer/family; `contend.sh` measures
co-run throughput and the contention fit; `tax.sh` measures the extra boundary
tax; `g80bf16.sh` measures the direct-BF16 NPU rate/traffic. Measure GPU glue
separately and record costs on the same tuple. Preserve those source results
alongside the record. Hardware measurement is a separately scheduled gate;
the CLI performs only offline assembly and never claims to have measured.

### Offline golden parity

`tests/fixtures/npu_trunk` preserves all four original calibration JSONs plus
`g80bf16-measured.json` unchanged. `golden.json` captures the actual Python CLI
stdout and full-precision imported-model times/plans for both family sets,
at 8192, 9319, 12288, 16384, 24576, 32768, 57544, 65536 and each crossover:
90 exact-range vectors, 10 crossovers, relative time tolerance 1e-6.
Tests adapt legacy fixture keys/version/margin and Python's absent-panel default
(`panel_cols=1` for V8), not measured costs.
`golden-v2.json` adds all-subset selection and half-bank consume vectors for
stock/overlay × retile/direct, nonzero margins, I32 fold, and a zero-cost GPU
baseline, plus 44 exact-range plans. The same tests run in the library and
the `npu_trunk_plan` integration target.
The migrated lab `g80bf16-{stock,overlay}-v2.json` records preserve measured
costs and add Halo j110 consume rates: BF16 `2.85/16480`, F32 `4.60/16480`
ms per column per 8192 rows. Overlay changes only trunk identity and zab
GPU cost to 154 ms/chunk (`228 - 74` for the measured A4 tier); the NPU
W8 stock-MQ6 numerical contract is unchanged.
Run offline:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo test -p hipfire-arch-qwen4 --lib npu_trunk -j 12
```

## First-light serving route (Linux gfx1151, family B)

`HIPFIRE_QWEN4_NPU_TRUNK=off|emulate|on` is resolved when the forward
owner loads. `off` returns before packing, NPU/GPU allocations, device opens
or trace-file opens. The ordinary chunk executor is unchanged except for
inactive option checks. `auto` is explicitly refused until calibrated admission
is wired; it is not a silent fallback from `on`.

The opt-in route requires Flash-Next K=2560 and stock MQ6G256V2 for all eight
input families. In `attach_forward`, preparation downloads each stock matrix
once and packs the S1 `Bf16Direct` panels and exact row-major W8/Sm/Se twin
**before** applying the pending trunk overlay or developer requantization.
Raw MQ6 host bytes are released layer by layer. Only packed host results cross
the overlay swap, then move into the forward/runtime; no extra stock GPU copies
are retained. This moves the existing download/packing cost earlier without
adding a transfer or a second preparation pass.

Both arms still load the beta `<model>.trunk-overlay`. Covered z/a/b use W8 from
the **stock MQ6**, never the replacement QT44 bytes. The existing load line adds
`trunk_route=overlay(gfx1151-zab-v1)|stock` (only successfully applied overlays
count) and `zab_source=stock-MQ6-DtoH-before-overlay|stock-MQ6-DtoH`, alongside
`download_ms`, `pack_ms`, resident weight bytes and runtime status.
`off` does no download, packing, allocation or new logging; its existing mode
read moves before overlay application, whose payloads and native route are
unchanged. `on` additionally opens two fixed 8192-row row-major G80 programs,
resident 13/11-panel chains, and two banks each with registered A plus separate
GDN/QSA C arenas; any admission/open error fails
the load. Calibration is deliberately bypassed with `first-light-force-pairs`.

Both `on` and `emulate` use the same deterministic, segment-local pair plan.
Only paired banks use BF16-direct numerics: `on` consumes NPU results, while
`emulate` executes the exact IU8 twin sequentially, one bank at a time.
Unpaired chunks/tails, rows below 512, decode and uncovered paths retain the
incumbent GPU projections in **both** modes: z/a/b take the same A4/IU4 overlay
route as `off` wherever that native route applies. No paired input projection
is computed twice. Both covered executors call the mixer only with `Core` and
`OutProj`; GDN's `Whole|InProj` guard encloses the entire `project_trunk` call,
before dtype or `trunk_a4` selection, so it skips A4/IU4 **and** MQ6. A debug
assertion checks every covered mixer dispatch, with a CPU phase-decision test.
Uncovered steps retain `Whole` and the unchanged beta IU4 mask/dispatch.
Under `on` the GPU packs A, then the mixer core directly reads read-only BF16
`Bf16Arena` views of fused C; **there is no consume/retile pass**. Explicit
`GdnTrunkArenas`/`QsaTrunkArenas` arguments select arena launchers only for
`Core`; omitted views retain the incumbent native scratch path. `emulate`
keeps the IU8 twin and its native projection scratch. Out-projections and
GPU recurrence/attention/HC/MoE remain native.

### Bank state and ordering

`on` requires the full 8192-row scratch plan and single-device placement.
It allocates a second complete scratch bank only at load. This deliberately
conservative lab allocation double-buffers:

* token IDs, rotated embeddings, embeddings, **HC residual streams**;
* HC normalized/low/up/mixed/gates, activation rotation and hidden input;
* projection/projection2, GDN a/b/gate/beta/recurrent-output/BF16/z/output;
* QSA index/qgate/k/v/output/selected, attention output;
* PLE staged/rows/key/value/query/gated/normed/output;
* router logits, all routed/shared MoE intermediates, grouped permutation,
  count/offset/tile/gather/down scratch, FFN output, scalar and logits scratch.

The GDN recurrence/convolution, QSA KV/index caches and PLE convolution state
are deliberately shared, not double-buffered. No MoE scratch must survive a
layer boundary, but the complete second bank avoids any scratch alias lifetime
assumption. Opt-in execution uses mapped experts, not donor-layer staging.
The private NPU owner transfers as one exclusive object with `ArchModel`;
none of railgun's internal HIP Rc resources escapes that owner.

With an 8192-row chunk cap, pair whenever at least 8192+512 rows remain:
bank A has 8192 live rows and B has `min(remaining-8192,8192)`.
The GPU numerical twin uses exactly the same 8192-row boundaries. A shorter
unpaired chunk remains sequential. For example, 16200 rows are one
8192+8008 pair, 20000 are a full pair plus 3616 sequential rows, and 40000
are two full pairs plus 7232 sequential rows. For each layer:

```text
prepare(0,A), pack/kick(0,A); prepare(0,B), pack/kick(0,B)
wait(l,A), Core(l,A), retire A, Out/HC/MoE(l,A)
prepare(l+1,A), pack/kick(l+1,A)
wait(l,B), Core(l,B), retire B, Out/HC/MoE(l,B)
prepare(l+1,B), pack/kick(l+1,B)
```

`prepare` includes PLE when present and the attention HC read. `G` is mixer
core/out, attention HC write and MLP HC/MoE/write. The existing typed ops and
output/HC/MoE fusions execute those fragments; the numerical twin uses the
same fragment boundaries. Retirement is recorded after the complete Core,
which encloses every C reader; output projection and the rest of `G` read only
native scratch. `prepare_bank` queues a wait on that retirement event before
packing new A on the same stream. `kick` records its system-release A-ready
event after that wait and pack; the relay queries this event to completion
before submitting the next chain. Thus
`old C reads -> retired -> prepare wait -> pack -> ready -> next NPU C writes`,
even when the next layer changes shape. No arena is overwritten early.
The transport's checked GPU wait remains host-gated; this is not a CPU-free
rejoin and the relay submits NPU chains serially.

The exact live readers (pitch/col0 in BF16 elements) are:

| Arena/window | Core reader |
| --- | --- |
| GDN 16640 / qkv 0..10240, a 16384..16432, b 16432..16480 | `gated_delta_conv_params_batched_arena`, or its `qknorm` variant |
| GDN 16640 / z 10240..16384 | chunk `gated_delta_step_gate_wmma[_qknormed]_arena` (retaining normal/inline-Q8 selection), or non-chunk `gated_delta_gate_batched[_rotate]_arena` |
| QSA 14080 / q 0..12288, k 12288..12800, idx 13312..13952 | out-of-place `indexed_attention_norm_rope_batch_arena` for each window |
| QSA 14080 / idx key tail 13824..13952 | `indexed_attention_index_key_append_batch_arena` to F32 index scratch and raw-key cache |
| QSA 14080 / v 12800..13312 | `indexed_attention_cache_append_batch_arena(key=None)`; K comes from normalized F32 scratch |

No reader includes GDN padding 16480..16640 or QSA padding 13952..14080.
Attention/index selection read transformed F32 scratch, never raw C.

Both NPU programs retain the fixed 8192-row extent. For a partial B, the
producer explicitly zeros every padded A row; kick records B's live count
without patching the program, and all arena readers use only those live rows.
All GPU layer operations, router views, PLE staging, capture/MTP rows and
head work use the bank's real row count. Bank B's output offsets remain
8192, not its own row count.
Before building a partial B program, every bank-B tensor's logical extent is
rebound from the existing `Full` scratch plan for its live rows. This includes
grouped MoE, router and PLE staging/output fields; exact-size kernels never
receive the 8192-row logical tensor for a shorter bank. Allocation capacities
and ownership slots remain unchanged, with no GPU allocation/copy; a later
pair can rebind back to 8192. CPU coverage includes 8151 and 512 rows.

Bank A starts at the request's current position; B starts at position+8192.
Each bank-B QSA view is advanced using bank A's exact `next_lengths()` tuple
(full/raw/pooled/selected/position), rather than independently guessed offsets.
GPU `G(l,A)` precedes `G(l,B)`, so B sees A's recurrent updates and appended
KV. PLE row hashing advances history through A before hashing B. Only after
both banks finish do the head, requested wide-hidden/MTP capture, QSA metadata
and global position/history publish. A boundary at A's end is retained through
fixed-state staging without rewinding the live pair-end state.

The serving entry points are `Qwen4MtpDrafter::mtp_prefill` (default native
MTP), `SpecTarget::spec_advance` (generic speculative target prefill), and
`Qwen4Bundle::prefill_final` (AR, including `--spec off`). The native MTP and
generic target loops now offer up to two chunks to the target instead of
pre-splitting every request at 8192. CLI TTFT bench and fresh singleton HTTP
chat requests both dispatch through these same generate routes.

Only `on` doubles `spec_hidden` to 16384 rows; verify windows and MTP append
scratch keep their original capacities. MTP head append still runs separate
A/B subchunks with the original fill-row boundaries and prompt-ID uploads:
bank B reads hidden rows starting at 8192 under `on`, or 0 after its separate
target forward under `emulate`. Pending hidden names the last row of B.
Only the final bank of a prompt-ending pair runs the target pick; other pairs
capture silently. Each request logs its first scheduled pair once:
`npu-trunk: pair rows=8192+8151 position=0` for a 16343-row fresh prompt.

A pair may cross a session snapshot boundary **only at A's end**. The default
8-GiB cache's periodic 8192-row snapshots therefore permit full pairs: A's fixed
state is staged per layer and published at the same cadence as the sequential
run. Interior A/B boundaries, including message ends, still split the schedule.
The Halo comparison must retain the default cache and exercise a later prefix
hit, not disable the cache. See the staging layout and cost details below.
Generic resumed target advances whose pending snapshot cursor is already behind
their position remain native until that snapshot is reported by its owner.

### Overlap trace

With `HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk.jsonl`, successful pairs
append one JSON object per `(layer,bank)` with `position`, real live `rows`, `seq`,
`pack_issue_ns`, `kick_ns`, `npu_submit_ns`, `npu_done_ns`,
`gpu_wait_enqueue_ns`, `consume_issue_ns`, `mixer_start_ns`, `mixer_end_ns`
and `anchor_uncertainty_ns`. Host/relay timestamps use Linux CLOCK_MONOTONIC.
`consume_issue_ns` is retained for trace compatibility and is always zero.
The mixer interval covers the complete `G` fragment, including HC/MoE.
HIP timing events are resolved only after the pair, relative to a monotonic
anchor calibrated after initialization and **before** pipeline work; its host
bracket width is reported as uncertainty. There is no per-layer GPU sync.
No trace records are emitted for GPU-only tails or `emulate`.

### Parent's Halo gate

No full-model hardware parity or speed claim is established by CPU builds.
Use the same MQ6 family-B model, prompt and greedy settings in fresh daemons;
confirm the tokenized prompt contains at least 8704 rows (a file named `p16k`
is not itself evidence of its token count). Keep chunk admission at 8192.
Do not inherit the build-only `ROCR_VISIBLE_DEVICES=-1`/`HIP_VISIBLE_DEVICES=-1`.

Run transport before either serving arm, with devices visible:

```sh
target/release/npu-trunk-transport --pci 0000:c0:00.0
```

```sh
export HIPFIRE_DAEMON_BIN="$PWD/target/release/daemon"
export HIPFIRE_PREFILL_CHUNK_ROWS=8192
unset HIPFIRE_SESSION_CACHE_BYTES # retain the product-default 8 GiB prefix cache
# MODEL must name the pinned MQ6 family-B artifact, not an MQ4 artifact.
HIPFIRE_QWEN4_NPU_TRUNK=on \
HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk-on.jsonl \
  target/release/hipfire serve "$MODEL" 127.0.0.1 11435 --max-seq 32768
```

For a fresh, non-streaming singleton chat request against that server:

```sh
jq -n --arg model "$MODEL" --rawfile prompt p16k \
  '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:64,stream:false}' \
  | curl -sS http://127.0.0.1:11435/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary @-
```
Save the response, then send a second independent request with the same p16k
prefix and a different suffix (not the first response's generated continuation).
Require a reported prefix-cache hit with reused rows > 0. Save that response too.
Stop the `on` server, then start a fresh `emulate` server with the same settings
and send both identical requests in the same order:

```sh
HIPFIRE_QWEN4_NPU_TRUNK=emulate \
  target/release/hipfire serve "$MODEL" 127.0.0.1 11435 --max-seq 32768
```

Compare generated token IDs/content byte-for-byte separately for the cold request
and cache-hit request (not the nondeterministic response ID/timestamp/timings).
`on` must equal `emulate` on both. Inspect the `on` trace for paired A/B banks
despite the default cache. Separately compare `off` to unchanged beta, not to
BF16-direct `on`; repeat the cold/cache-hit gate with native MTP and `--spec off`.


Native MTP is enabled by default; repeat with `--spec off` for the AR route.
In a separate terminal, then repeat in fresh processes with `emulate` and `off`:

```sh
HIPFIRE_QWEN4_NPU_TRUNK=on \
HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk-bench.jsonl \
  target/release/hipfire bench "$MODEL" --ttft --prompt-file p16k --runs 3

for mode in on emulate off; do
  HIPFIRE_LOCAL=1 HIPFIRE_QWEN4_NPU_TRUNK="$mode" \
    target/release/hipfire run "$MODEL" --temp 0 --repeat-penalty 1 \
    --no-stream --max-seq 32768 --max-tokens 64 "$(cat p16k)" \
    > "/tmp/npu-trunk-$mode.txt"
done
cmp /tmp/npu-trunk-on.txt /tmp/npu-trunk-emulate.txt
# Compare off to the unchanged beta binary's output on the same model/settings,
# not to emulate (emulate intentionally changes the trunk numerical contract).
```

Continuous-batch/lane `prefill_wave` is not paired; its tiles retain the native
GPU route in both arms. A singleton pair may cross a snapshot boundary only
when it is exactly the A/B split. An interior A or B boundary still splits the
schedule; message-end cadence and cache scope are unchanged. EP, compact scratch,
calibrated `auto`, variable-size NPU programs and CPU-free waits remain outside first light.
CPU scheduling/capture/retirement tests live under
`gpu_forward::npu_forward::tests`; existing MTP tests cover head append IDs
and pick boundaries. Full-model greedy comparison and trace overlap
inspection remain the parent-run silicon acceptance gate.

### Default-cache snapshot staging

`state.rs::session_layout/session_parts_at` defines the trunk snapshot:
each GDN layer's recurrent live slot and F32 convolution history; each QSA
layer's complete selection buffer and append-only K/V, raw-index and completed
pooled-index prefixes; optional PLE convolution history; and hyper feedback.
Host metadata is model ID, absolute position, optional two-token PLE context,
and six words per QSA layer (full/raw/pooled/partial/selected lengths, position).
`mtp_gpu.rs::session_parts` adds the head's selection buffer, device selected
length, wide hidden, four append-only streams, model ID, lengths, position,
step index, and draft policy's full-vocabulary hold/margin bits. Request-local
pending scratch is not a session-cache part.

For `on`, `session_pair.rs` allocates one reusable fixed-state mirror before
the first boundary-bearing pair, only with an attached cache. In
`npu_trunk/forward.rs`, GDN recurrent/conv and QSA selection are copied by a
stream-ordered copy kernel immediately after each A layer and before its B
layer. PLE is copied immediately after A's PLE operation, before B's preparation
can overwrite that shared history; unchanged hyper feedback is copied before
the pipeline starts. No staging copy synchronizes the host, including on the
default HIP stream. The ordinary metadata serializer describes A's QSA marks
and token history without rewinding the live B owners. Debug builds reconstruct
the layout through the ordinary restore parser and compare fixed sizes, stream
row widths/counts on the CPU, without draining the stream.

The cache does **not** store QSA KV by lengths alone: `session_cache.rs` copies
append-only row deltas above the parent and pins that parent. A staged snapshot
therefore supplies A's lengths over the still-valid live prefixes, plus staged
fixed bytes. Publication uses the unchanged `SessionCache::at_boundary`:
after the pair and A's MTP head append, before B's head append; AR publishes
after the pair. `emulate` publishes from ordinary sequential live owners after
A, without staging. The next B-end snapshot is ordinary in both arms.

Additional device memory is the sum of trunk fixed-part byte sizes, independent
of prompt length; no KV mirror and no extra MTP mirror are allocated. At pinned
Flash-Next geometry with F32 GDN this is **118,177,936 B (112.70 MiB)**:
36 recurrent/conv pairs, 12 selections of 2051 i32s, nine 10240-wide PLE history
rows, and 10240 feedback floats. Other GDN storage formats use their actual
recurrent byte sizes. Host staging retains one metadata blob and 48 row counts.
Each boundary-bearing pair adds that many logical copy bytes, or about
236.36 MB of read+write traffic, in 50 copy-kernel launches (one per GDN/QSA
layer plus PLE and feedback). A hypothetical effective 100–200 GB/s gives
**2.36–1.18 ms plus launch/serialization cost**; this is a bandwidth estimate,
not a measured Halo timing. Cache publication's ordinary fixed/row copies and
allocation costs remain in addition, just as for sequential capture.

CPU acceptance uses GPUs hidden, `nice -n 19`, the worktree's target directory
and `-j 12`: build `hipfire-daemon`, `hipfire-cli`, and `qwen4_kld --features lab`;
run `cargo test -p hipfire-arch-qwen4 --lib` and
`cargo test -p hipfire-runtime --lib session_cache`. These checks do not certify
snapshot bytes on silicon; the cold/default-cache and branched cache-hit
`on`/`emulate` Halo gate above remains required.


## Direct BF16 arena consumer interfaces

`rdna_compute::tensor_ops::Bf16Arena { tensor, row_pitch_elems, col0_elems }`
is an immutable BF16 view; pitch and column offset are **elements**, not bytes.
The tensor can be a borrowed GPU-visible host-registered buffer and must remain
alive through stream completion. `ColumnMap` is authoritative: GDN pitch 16640,
qkv at 0, z at 10240, **a at 16384, b at 16432**; QSA pitch 14080,
qgate at 0, k at 12288, v at 12800, index at 13312. Qgate is 24 repetitions of
`q[256] | gate[256]`, not a global q/gate split. Index is `query[512] | key[128]`.

All functions below take `gpu` first and return `HipResult<()>`. `p`, `conv`,
`params`, `step`, and `gate` are the corresponding incumbent parameter structs;
all non-projection arguments retain their meaning.

| Variant signature after `gpu` | Projection/output convention | Implementation |
| --- | --- | --- |
| `gated_delta_conv_batched_arena(p, qkv)` | `p.input` ignored; output and history unchanged, including aliasing | hipcc compile-time twin |
| `gated_delta_conv_params_batched_arena(conv, params, qkv, a, b)` | Input/a/b fields ignored; fused convolution and parameter blocks | hipcc compile-time twin |
| `gated_delta_conv_params_qknorm_batched_arena(conv, params, step, keep_qk_conv, qkv, a, b)` | Same fusion; QN/KN go to the incumbent shared scratch | hipcc compile-time twin |
| `gated_delta_params_batched_arena(p, a, b)` | `p.a/p.b` ignored; gate/beta remain contiguous F32 | hipcc |
| `gated_delta_gate_arena(p, z)` | `p.z` ignored; single-row, read-only recurrent input | hipcc |
| `gated_delta_gate_batched_arena(p, z)` | `p.z` ignored; rounded recurrent input updated as before | hipcc |
| `gated_delta_gate_batched_rotate_arena(p, rotated, z)` | Non-chunk prefill: gate plus all FWHT rotation outputs; same prerotation marker | hipcc |
| `gated_delta_step_gate_wmma_arena(step, gate, z)` | `gate.z` ignored; normal state/inline-Q8 selector retained | hipcc compile-time twin |
| `gated_delta_step_gate_wmma_qknormed_arena(step, gate, z)` | Same, with convolution-provided QN/KN | hipcc compile-time twin |
| `gated_delta_step_gate_q8_wmma_arena(step, gate, z)` | Explicit inline-Q8 arm, independent of inline selector | hipcc compile-time twin |
| `indexed_attention_norm_rope_batch_arena(p, src)` | Immutable BF16 input; `p.values` is out-of-place F32 output, at `p.row_stride/p.head_stride` | hipcc |
| `indexed_attention_cache_append_batch_arena(p, key: Option<Bf16Arena>, value)` | `Some(key)` reads BF16; `None` reads normalized contiguous F32 `p.key`; `p.value` ignored | PM for F32 caches; hipcc for Q8/FP8 |
| `indexed_attention_index_key_append_batch_arena(p, index)` | `p.index_rows` is F32 output: query preserved, key tail written; raw keys may be BF16 or F32 | PM |

The out-of-place norm copies untouched gates and index-key tails as it writes
the normalized heads. Thus **attention reads its transformed F32 qgate output**
and **index selection reads its transformed F32 index query**; neither reads a
raw trunk projection, so neither needs an additional arena kernel. Cache append
after K normalization uses `key=None` plus the original BF16 V view. Index append
writes only the output tail and raw-key cache, never the source arena.

`npu_trunk_copy_projection_arena(gpu, src, dst, rows, cols, dst_pitch,
dst_col0, position_start)` is the PM row-copy primitive used for cache stores
and the lab's exact old-path materialization. Its three certified gfx1151
exports cover BF16→F32 (`bits << 16`), BF16→BF16 and F32→F32. The committed
`npu_trunk_arena_pm_gfx1151.{s,co,hxaco,proof.json,m7.json}` carry M7 byte-exact
lifts, empty obligations and zero ambiguous delays; the embedded image is
SHA256-pinned. Hipcc exceptions have one entry per kernel in
`/home/kaden/qcal/release-0.4.2/PM-GAPS.md`.

### Parent's Halo consumer oracle

```sh
cd /home/kaden/ClaudeCode/warpfront/wt-npu-trunk
HIPFIRE_QWEN4_GDN_PREFILL=wy HIPFIRE_QWEN4_F16_WMMA=1 \
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo run --release -p rdna-compute --features lab \
  --example npu_trunk_consumers_check -j 12
```

Run with GPUs visible on Halo, not in the CPU build environment. For 8192 and
4647 rows at each family's pitch, this compares all output/state bytes:
convolution history (including in-place aliasing), normalized QN/KN scratch,
recurrent state (including inline-Q8 codes/scales and diagnostic F32 capture),
non-chunk fused-gate FWHT rotation outputs,
QSA gates/index tails, complete K/V caches and raw index keys with untouched
prefix/suffix guards. It verifies the materializer independently against CPU
bit widening, prints old timings including materialization versus direct reads,
checks immutable arena bytes, and exits 1 on any mismatch. GPU parity/timings
are parent-run silicon evidence, not implied by CPU build/test success.

Replay follows the incumbent position contracts: cache/raw-key stores and
RoPE positions are bound to replay position; compact F32 scratch-tail writes
and the lab copy primitive's concrete destination offset remain position-independent.
