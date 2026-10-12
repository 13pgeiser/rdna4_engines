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
and poison bytes outside the panel window. Row-major DMA silicon validation
remains required even though the native-tile BF16 arithmetic is proven.

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

`Calibration::{load,store,validate,key_matches}` uses schema version 1 JSON.
All fields are required; missing inputs, unknown fields, invalid numeric costs,
unknown versions and inconsistent epilogues are errors. `CalKey` consists of
`device.{arch,pci_bdf,npu_fw_version}`, caller-supplied `build` (the measured
daemon git SHA/binary identity, not the calibration CLI's identity),
`kernel.{uname_release,perfopt_armed}`, `model_sha256` (64 hex digits), and
`epilogue` (`bf16` or `i32`). A key mismatch returns the first mismatching
field in that order. The load caller must log `npu-trunk: off (reason)` and
leave the route off; loading never recalibrates or opens hardware.

`plan(&Calibration, FamilyMask, rows) -> Plan` and
`crossover(&Calibration, FamilyMask) -> Option<usize>` port the reference
`npu-trunk/crossover_model.py`, SHA256
`d50b20c66d1c62b243e88e1b23e956c3e072d31937b3352305b7ba126f4acc4a`.
The planner uses an 8192-row chunk DP, GPU-only/pair choices, 64-aligned
tail rebalance, even re-banking with boundary tax, and a taxed single-chunk
split. `Step::Gpu { chunk }` and `Step::Pipe { a, b }` carry exact half-open
prompt row ranges, including every pair of a re-banked plan. GPU wins ties;
existing earlier pairing wins ties against rebalance/re-bank. `min_bank_rows`
has the reference's placement (rebalance, single split, and re-bank eligibility).
GPU costs distinguish 36 GDN and 12 QSA layers and prefix-dependent attention;
NPU time rounds fused columns up to `panel_cols`. Contention is a burst term
`(alpha_overlap - 1) * min(G_other, N_this)`, not whole-prefill slowdown.

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
The measurements schema is the same as the schema-version-1 output record:

* `schema_version: 1`, `margin` and the complete current `key` described above.
  The collector supplies device/firmware, PerfOpt, measured build, model digest,
  and epilogue identity. The CLI additionally checks `uname -r` against the
  collector's key and refuses a kernel mismatch. The eventual load caller must
  compare **all** key fields to its actual runtime identity.
* `gpu.trunk_in_ms_per_8192`: required `qkv`, `zab`, `q`, `kvidx` millisecond
  totals across the respective 36/12 layers from kernel-trace attribution.
  `gpu.uniform_per_layer_ms`, `gdn_core_per_layer_ms`, `qsa_per_layer_ms`,
  `qsa_growth_per_layer_ms_per_8192_prefix`, `fixed_per_chunk_ms`,
  `boundary_tax_ms`, and `min_bank_rows` are required.
* `families`: those four names, each with `layer` (`gdn` for qkv/zab,
  `qsa` for q/kvidx) and positive fused `cols`.
* `glue.pack_ms_8192`, `fold_ms_per_col_8192`, `sync_ms`: measured
  pack/consume/fold costs (fold is charged only for the I32 contract).
* `npu.tops_corun`, `bytes_per_kop`, `epilogue`, `panel_cols`: measured
  co-running rate, measured or declared traffic, chosen output contract, padding.
* `contention.alpha_overlap` (>=1) and `model`: fitted/measured overlap
  slowdown and the provenance/fit description. The planner consumes the alpha;
  it does not reinterpret the descriptive model string.

Collect once per exact device/build/kernel/PerfOpt/model/epilogue tuple.
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
Run offline:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo test -p hipfire-arch-qwen4 --release npu_trunk -j 12 -- --nocapture
```

## First-light serving route (Linux gfx1151, family B)

`HIPFIRE_QWEN4_NPU_TRUNK=off|emulate|on` is resolved when the forward
owner loads. `off` returns before packing, NPU/GPU allocations, device opens
or trace-file opens. The ordinary chunk executor is unchanged except for
inactive option checks. `auto` is explicitly refused until calibrated admission
is wired; it is not a silent fallback from `on`.

The opt-in route requires Flash-Next K=2560 and MQ6G256V2 for all eight
input families. It packs the S1 `Bf16Direct` contract and uploads the exact
row-major W8 and Sm/Se twin. The bundle retains resident GPU handles, not
source byte owners, so preparation currently downloads each MQ6 matrix once.
The load line records `download_ms`, `pack_ms`, actual resident weight bytes,
mode/families/contract and runtime status. `on` additionally opens the unchanged
8192-row G80 program, resident 13/11-panel chains and two registered A/C banks;
any admission/open error fails the load. Calibration is deliberately bypassed,
with a `first-light-force-pairs` load note.

`emulate`, and every unpaired chunk under `on`, use the producer and exact IU8
BF16-direct GPU twin for rows >=512. Smaller chunks and decode retain the
ordinary projections. Out-projections retain their MQ6 route. QSA output
scratch is F32; GDN qkv is BF16 only when the existing chunk-core route reads
BF16 (otherwise it is exact widened F32); z/a/b are F32.

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
wait/consume(l,A), retire A, G(l,A)
prepare(l+1,A), pack/kick(l+1,A)
wait/consume(l,B), retire B, G(l,B)
prepare(l+1,B), pack/kick(l+1,B)
```

`prepare` includes PLE when present and the attention HC read. `G` is mixer
core/out, attention HC write and MLP HC/MoE/write. The existing typed ops and
output/HC/MoE fusions execute those fragments; the numerical twin uses the
same fragment boundaries. `prepare_bank` queues the previous retirement
event before any new A writes; the relay's A-ready event orders the new submit.
Retirement follows the last C scatter, not the later core's native-tensor reads.
The transport's checked GPU wait remains host-gated; this is not a CPU-free
rejoin and the relay submits NPU chains serially.

Both NPU programs retain the fixed 8192-row extent. For a partial B, the
producer explicitly zeros every padded A row; kick records B's live count
without patching the program, and consume writes only those live rows.
All GPU layer operations, router views, PLE staging, capture/MTP rows and
head work use the bank's real row count. Bank B's output offsets remain
8192, not its own row count.

Bank A starts at the request's current position; B starts at position+8192.
Each bank-B QSA view is advanced using bank A's exact `next_lengths()` tuple
(full/raw/pooled/selected/position), rather than independently guessed offsets.
GPU `G(l,A)` precedes `G(l,B)`, so B sees A's recurrent updates and appended
KV. PLE row hashing advances history through A before hashing B. Only after
both banks finish do the head, requested wide-hidden/MTP capture, QSA metadata
and global position/history publish; existing caller session snapshots remain
outside the pair. Pairing never crosses a caller's session segment boundary.

### Overlap trace

With `HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk.jsonl`, successful pairs
append one JSON object per `(layer,bank)` with `position`, real live `rows`, `seq`,
`pack_issue_ns`, `kick_ns`, `npu_submit_ns`, `npu_done_ns`,
`gpu_wait_enqueue_ns`, `consume_issue_ns`, `mixer_start_ns`, `mixer_end_ns`
and `anchor_uncertainty_ns`. Host/relay timestamps use Linux CLOCK_MONOTONIC.
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

```sh
export HIPFIRE_DAEMON_BIN="$PWD/target/release/daemon"
export HIPFIRE_PREFILL_CHUNK_ROWS=8192
# MODEL must name the pinned MQ6 family-B artifact, not an MQ4 artifact.
HIPFIRE_QWEN4_NPU_TRUNK=on \
HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk-on.jsonl \
  target/release/hipfire serve "$MODEL" 127.0.0.1 11435 --spec off --max-seq 32768
```

In a separate terminal, then repeat in fresh processes with `emulate` and `off`:

```sh
HIPFIRE_QWEN4_NPU_TRUNK=on \
HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk-bench.jsonl \
  target/release/hipfire bench "$MODEL" --ttft --prompt-file p16k --runs 3

for mode in on emulate off; do
  HIPFIRE_LOCAL=1 HIPFIRE_QWEN4_NPU_TRUNK="$mode" \
    target/release/hipfire run "$MODEL" --temp 0 --repeat-penalty 1 \
    --spec off --no-stream --max-seq 32768 --max-tokens 64 "$(cat p16k)" \
    > "/tmp/npu-trunk-$mode.txt"
done
cmp /tmp/npu-trunk-on.txt /tmp/npu-trunk-emulate.txt
# Compare off to the unchanged beta binary's output on the same model/settings,
# not to emulate (emulate intentionally changes the trunk numerical contract).
```

If a caller splits the prompt into individual lane tiles/session segments before
the chunk owner receives it, only pairs within those segments can pipeline.
EP, compact scratch, calibrated `auto`, variable-size NPU programs and CPU-free waits
are outside this first-light route. CPU bookkeeping tests live under
`gpu_forward::npu_forward::tests`; the full-model greedy comparison and trace
overlap inspection remain the parent-run silicon acceptance gate.

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
