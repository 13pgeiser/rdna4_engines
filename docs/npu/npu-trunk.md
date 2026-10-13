# Flash-Next NPU trunk: CPU preparation

The Qwen4 `npu_trunk` module owns CPU-only model preparation for hipfire 0.4.2;
`pm-npu` owns the numerical reference, panel layout and deployment builder.
Neither preparation path opens hardware or schedules prefill.

`NpuTrunkMode::from_env()` borrows the central process snapshot for
`HIPFIRE_QWEN4_NPU_TRUNK`: `off`, `auto` (default), `on`, or `emulate`.
At load, auto resolves silently to off on every arch except gfx1151.

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

The only contract is `EpilogueContract::Bf16Direct` (`as_str` "bf16_direct"; the `I32Fold` contract and its
GPU fold were removed: no int32 partial, combine or fold exists in the product route). The ordinary pack
functions use it: `Bf16Direct` rounds each raw weight scale upward with `pm-npu`'s public `grid15_up`,
quantizes with that exact grid scale, and returns padded `(Sm,Se)` sidecars in `PackedB::scales` as well
as embedding them in each packed panel; `PackedB::ws` holds the grid scale as f32.

## Programs

`build_programs(contract, rows_max)` selects one G80 direct-BF16/Fast
design per shape, via `pm-npu::kernels::gemm_g80::build_panel_program`.
M is rounded up to 256, N=1280, K=2560; `rows_max` must be 1..=65536.
`build_programs_for_families` uses the same compacted maps as weight packing.
Each shape record reports panel count, padded width, per-submit A/B/C argument
sizes and per-submit host bytes read/written. The designs include PDI and TXN
bytes; panels reuse their shape's design.

`build_rowmajor_programs_for_families(rows_max, mask)` builds BF16-direct designs
with col0=0 and the selected map's padded fused pitch. qkv-only GdnIn is exactly
8 panels / pitch 10240. Zero-panel shapes are never submitted or given GPU views.
Arg2 declares the compact fused arena; each retained panel command advances its
C base by `2*panel*1280` bytes.

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
row order. One shim BD per channel/wave scatters both 128x80 parity tiles
with `(1,40), (pitch/2,128), (40,0)`; transfer length is 10240 words.
The outer +40-word step returns to row zero at the adjacent 80-column tile.
Fields are `Bd::memtile_words` words 2..5 (17-bit step-minus-one,
10-bit wraps) and `Bd::shim_words` words 3..5 (20-bit step-minus-one,
10-bit wraps), transcribed from aie-rt
[`xaie2pgbl_params.h`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/global/xaie2pgbl_params.h)
and [`xaie_dma_aieml.c`](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/dma/xaie_dma_aieml.c).
Unused outer wrap is zero; length terminates the outer dimension.

At pitches 14080/16640, a 256-row wave step exceeds the shim's 20-bit
stride/iteration field. The drain instead batches **K=7 waves**, with absolute
DDR addresses for every wave. AIE2P has **16 shared shim BDs per tile**, not
16 per channel (`Bd::address`, four-bit NEXT_BD in `Bd::shim_words`, and the
[vendor register fields](https://github.com/Xilinx/aie-rt/blob/release/main_aig/driver/src/global/xaie2pgbl_params.h)).
B=0 and A=2 remain live. C uses seven BDs per channel:
S2MM0 `[1,4,6,7,8,9,10]`, S2MM1 `[3,5,11,12,13,14,15]`.
All 16 BDs are used on A-injecting columns (15 otherwise). Two BDs per wave
leave room for at most seven complete waves without borrowing A/B descriptors
or changing stream bytes/order.

Each channel gets one finite seven-wave chain/task and one completion token.
Both C SYNCs precede every refresh. The first batch of every full or lean
body writes its C descriptors; later batches refresh **DDR patches only**.
Iteration wrap=1 keeps `current=0`. A partial final batch cuts its last
links with two tail-word writes per column; the next body's initial writes
restore the chain. Ring POLL/SENT/DONE temporarily borrow 4/5/6 outside the
completed GEMM body, so initial C writes are never omitted by same-design
lean bodies. A/B still queue once per submit; the API and panel base offsets
are unchanged. No extra data transfer, channel, stage or memtile/core BD.

For W waves and H=ceil(W/7), C drain TXN operations are
`16*min(W,7) + 16*W + 18*H + 16*(W>7 && W%7!=0)`,
versus the previous `82*W` and native-tile `50` (descriptor writes,
patches, task writes, two aggregate SYNCs). At W=32 this is **730 vs 2624**
C ops (72.2% fewer); SYNCs/tasks fall from 32 to 5 pairs (84.4% fewer)
and descriptor block writes from 1024 to 112 (89.1% fewer).
At W=256: **4890 vs 20992** C ops, 37 vs 256 SYNC/task pairs.
TXN bytes/counts change; the PDI, native BF16 TXN and int8/I32 golden bytes
remain unchanged. Expected firmware overhead is substantially reduced, but
recovery of the measured 9% throughput loss is a silicon gate, not a claim.
Firmware command overhead is not covered by the phase-sum cycle estimate.

Build one design with col0=0 per pitch/row bound. All 11 QsaIn / 13 GdnIn
panels can reuse its PDI and program; advance arg2's base by `2*panel*1280`
through `append_run_body` or `append_lean_run_body`. Retain the full fused
arena behind the offset, and enforce `panel_col0+1280 <= pitch`.
Alternatively the explicit col0 builder changes DDR patch offsets only.
Shim traffic remains A `4*W*kc*4608`, B `8*kc*2*5504`, C
`8*W*4*20480` bytes: the phase-sum estimate is unchanged from native tiles.
Simulator coverage includes 1, 2, K, K+1 and 256 waves at both fused pitches,
full/lean reuse, adjacent panels sharing an offset arena, padded zeros,
outside-window poison and exact shim bytes. Tests compare TXN op counts with
the previous per-wave drain and native tiles; functional ticks are modelled
timings, not firmware latency or silicon measurements.

Silicon commands (not run by this change; repeat at pitch 14080):

```sh
# Every live value, padded zero and outside-window poison checked.
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 \
  --rowmajor-pitch 16640 --rowmajor-col0 1280 --iters 3 --reuse-ctx
# Same exact first/final-poison checks, with busy-region TOPS over 15 seconds.
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 \
  --rowmajor-pitch 16640 --rowmajor-col0 1280 --loop 15
target/release/npu-gemm 8192 1280 2560 --variant G80 --epi bf16 --loop 15
```

The previous per-wave row-major drain measured 15.05 TOPS on Halo versus
16.48 TOPS native; the batched drain needs fresh silicon exactness/performance
validation and the serving cutover still requires the parent-run gates below.

## GPU Bf16Direct glue

`rdna_compute::npu_trunk` exposes an explicit `NpuBf16Layout` (padded rows,
K64-compatible K, panel count) and native `NpuTrunkOutput` ranges. S1 maps bind
destinations with `ColumnMap::gpu_output`; no fused output tensor is needed.
The producer writes every packed A byte, including zero padding, directly to a
device tensor or a device-accessible host-registered arena (`unsafe ..._ptr`).
Compact scale sidecars are four bytes per token: LE u16 mantissa then LE i16
exponent. Weight sidecars use the same encoding, indexed by fused column.

On gfx1151 at K2560, A-pack uses eight wave32 rows per workgroup, 128-bit
activation loads retained in registers, an integer half-magnitude wave max,
and one LDS barrier to coalesce native 8x8 byte tiles into 128-bit stores.
The scale and RNE quantizer DAG is unchanged. Other K/architectures retain the
scalar producer, also exposed as a lab-only old/new oracle. At 8192 rows the
single-read traffic is 65.57 MB including replicated scales and sidecars;
180 GB/s implies **0.364 ms** before arithmetic/LDS/launch overhead, an estimate
pending Halo event timing (not a measured speedup).

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
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-pack/target \
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
It additionally compares old/new complete A arenas and compact sidecars, including
padding and poison guards, at 1/37/512/4647/8151/8192 rows. Zero, maximum finite
F16, RNE half-integer ties and grid-scale boundaries are included even at one row.
It prints old/new pack GPU-event timings and bandwidth for each row count, plus
full-width consumer and integer GEMM timings; any mismatch exits 1. CPU layout/map tests:
`cargo test --release -p hipfire-arch-qwen4 --lib npu_trunk::gpu::tests -j 12`.

## Stored calibration and deterministic planning (0.4.2)

`Calibration::{load,store,validate,key_matches}` uses schema version **4** JSON
(4 removed `glue.fold_ms_per_col_8192` and the I32 contract: `npu.epilogue` and
`key.epilogue` must be `bf16`, BF16-direct only).
Measured fields are mandatory; only an omitted `key.build` defaults to the
release contract. Unknown fields/versions and invalid costs are errors.
Runtime `CalKey` derives:

* `device.arch`: the selected GPU's architecture; `device.pci_bdf`: HIP's PCI
  bus ID, lowercase; `device.npu_fw_version`: trimmed
  `/sys/class/accel/accel0/device/fw_version` (the amdxdna device railgun opens).
* `build`: `hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1`, shared by serving and
  `calibrate-npu-trunk`; no git SHA and no binary hashing.
* `kernel.uname_release`: trimmed `uname -r`. The recorded PerfOpt arming
  signal is `AMD-Vi: PerfOpt armed` in privileged dmesg, not a readable
  sysfs/proc ABI. Schema 3+ therefore drops `perfopt_armed`, rather than inventing
  a boolean from a kernel name. Recollect if boot arming changes under the same
  release: this key cannot distinguish that case.
* `model_index_sha256`: HFQ bytes `[0,data_offset)`, using
  `HfqFile::index_sha256`, exactly the trunk-overlay binding convention.
  The verified local `qwen3.8-flash-next-gptq3.mq4` has `data_offset=14442496`
  and index SHA256 `8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013`.
  No 125-GB full-file hash is computed at load.
* `epilogue`: serving uses `bf16`; `trunk_route`: `stock` or
  `overlay:<marker>` from the **successful overlay outcome**, not its presence.

Mismatch reasons name the first differing field in this order. Loading stored
calibration never measures or recalibrates. The two `g80bf16-{stock,overlay}-v2.json`
lab files and corresponding fixtures retain their historical filenames but now
carry schema 4 identity; measured cost inputs and Python/Rust cost goldens are unchanged.

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
perform measurements, probe hardware, fit contention, or default measured costs.
The measurements schema is the same as the schema-version-4 output record:

* `schema_version: 4`, `margin` and the current `key` described above.
  The collector supplies device/firmware, kernel release, model **index** digest,
  epilogue and trunk route. Omitted `key.build` uses the stable release/NPU
  contract. The CLI checks `uname -r` and refuses a kernel mismatch. Serving
  compares **all** key fields to its runtime identity.
* `gpu.trunk_in_ms_per_8192`: required `qkv`, `zab`, `q`, `kvidx` millisecond
  totals across the respective 36/12 layers from kernel-trace attribution.
  `gpu.uniform_per_layer_ms`, `gdn_core_per_layer_ms`, `qsa_per_layer_ms`,
  `qsa_growth_per_layer_ms_per_8192_prefix`, `fixed_per_chunk_ms`,
  `boundary_tax_ms`, and `min_bank_rows` are required.
* `families`: those four names, each with `layer` (`gdn` for qkv/zab,
  `qsa` for q/kvidx), positive fused `cols`, and `out_dtype` (`bf16` for qkv,
  `f32` for the other families in today's serving route).
* `glue.pack_ms_8192`, `sync_ms`: measured pack/sync costs (schema 4 has no
  fold term: every contract is BF16-direct).
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

`tests/fixtures/npu_trunk` preserves the three BF16 legacy calibration JSONs
(`g80bf16-{est-fast,est-slow,measured}.json`) and the stock/overlay v2 records;
the I32 records (`g80i32-est`, `v8i32-measured`) are gone with the I32 contract.
`golden.json` captures the actual Python CLI
stdout and full-precision imported-model times/plans for both family sets,
at 8192, 9319, 12288, 16384, 24576, 32768, 57544, 65536 and each crossover:
54 exact-range vectors, 6 crossovers, relative time tolerance 1e-6.
Tests adapt legacy fixture keys/version/margin and Python's absent-panel default
(`panel_cols=1` for V8), not measured costs.
`golden-v2.json` adds all-subset selection and half-bank consume vectors for
stock/overlay × retile/direct, nonzero margins and a zero-cost GPU
baseline, plus 40 exact-range plans. `golden-exec.json` pins the executable
projection (`executable_plan`: natural and rebalanced pairs) and the prefill group
schedule (`schedule`: banks and the in-bank snapshot cut, cache off / on / on
without sub-launches) for 8000..65536 rows on the stock/overlay records. The same
tests run in the library and the `npu_trunk_plan` integration target (the
schedule golden runs in the library, where the scheduler is visible).
**All goldens are generated** by
`python3 /home/kaden/qcal/release-0.4.2/npu-trunk/gen_goldens.py
crates/hipfire-arch-qwen4/tests/fixtures/npu_trunk` from `crossover_model.py`
and the fixture calibrations (a pure function; no hardware), never edited by hand.
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

## Calibrated serving route (Linux gfx1151, family B)

`HIPFIRE_QWEN4_NPU_TRUNK=off|auto|on|emulate` is resolved at load.
The default is `auto`, effective only on gfx1151 (Strix Halo). Other arches
resolve auto silently to off before index hashing, calibration/device I/O,
packing, NPU allocations, trace opens or new logging. Explicit on/emulate
refuse non-gfx1151. `off` / `fusions.npu_trunk = off` is the kill switch.

`fusions.npu_trunk_calibration` (alias `npu_trunk_calibration`, env
`HIPFIRE_QWEN4_NPU_TRUNK_CAL`) defaults, when empty, to
`<hipfire home>/npu-trunk/calibration.json`. A file at that path takes precedence.
Only an absent file falls back to compiled-in measured Halo schema-4 records:
`halo-xdna2-overlay-v4.json` (`overlay:gfx1151-zab-v1`) or
`halo-xdna2-stock-v2.json` (`stock`). The stock lab record is schema-compatible,
keyed stock, and measured for `hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1`.
Invalid/unreadable files never fall back. Both sources require identical key
checks; successful arming logs `cal_source=file|builtin`. The reference key is
gfx1151 / `0000:c0:00.0`, firmware `1.1.2.65`, uname `7.3.0-rc7-perfopt+`,
model index `8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013`,
BF16 and the matching stock/overlay route; the builtin does not admit other keys.

Admission runs in this order, before any stock MQ6 download/packing or NPU allocation:

1. Flash-Next family-B K=2560, stock MQ6 shapes for every trunk family,
   single-device/no EP, full scratch, 8192-row maximum chunk.
2. The exact `/dev/accel/accel0` node railgun opens must be an amdxdna PCI
   device with readable, nonempty `fw_version`; no runtime open yet.
3. Load file-or-builtin calibration and match device, firmware, build contract,
   uname, model index sha, epilogue and anticipated trunk route; select families.
4. Download stock MQ6 before the overlay swap, pack only admitted families,
   recheck the successfully applied route and actual scratch plan, then open
   the runtime and allocate the second bank. Any auto preparation/open failure
   frees partial resources and runs GPU-only. Armed logs are emitted only after success.

Every auto refusal on gfx1151 logs exactly one `npu-trunk: off (<reason>)` line;
it never fails a load. After admission, sticky runtime errors still fail the
request closed; GPU fallback must not publish partial NPU results.

The default-on decision uses Halo default serve config with the zab overlay,
three fresh processes per arm: p16k 7,878 → 7,173 ms (−8.9%),
p56k 30,052 → 27,867 ms (−7.3%). Greedy `on` and `emulate` bytes match;
flag-off `serve_harness` battery and chain match beta. Kaden accepted S11 j103
BF16-direct family B: ΔKLD WT2 −0.0005 / code24 +0.0052 (cap +0.015).

| Mode | Calibration unavailable/invalid/key mismatch | Match: all selected | Match: subset/none |
| --- | --- | --- | --- |
| `off` | No I/O | GPU-only | GPU-only |
| `auto` | Route off, one reason line | Admit stored plan | Admit subset plan; none stays native |
| `on` | `calibration=none (forced)`, ALL | Use selected families; force pairs | Use subset; none stays native |
| `emulate` | Same as `on`, GPU twin | Same as `on`, GPU twin | Same as `on`, GPU twin |

An auto key mismatch logs exactly `npu-trunk: off (calibration key mismatch: <field>)`.
Matching records log `load_log_line` with selection, key/hash, crossover and
predicted 16K/64K times. Packing, twin, row-major programs, C arenas and request
planning all use that same selected mask. Without a match, forced modes use ALL.
Auto derives n* once at load. Each request's uncached prompt segment runs
`plan(cal, selected, rows)` and logs `request_log_line` once. Natural planned
8192+B pairs, with B >= `max(512, cal.gpu.min_bank_rows)`, and the planner's
`pipe-rebalanced` last pair (two banks <= 8192 rows, 64-aligned split, each >=
`max(512, min_bank_rows)`, ending the request) execute above n*. Split and
rebanked ranges are projected to ordinary GPU chunks, never force-paired. The
logged executable projection reflects those restrictions; a second snapshot row
inside a pair may additionally shorten a candidate group. Rebalanced pairs and
in-bank snapshot cuts run in the default profile too: partial Core windows route as
their bank (`with_bank_routing`), and banks need at least 512 rows.

For the historical stock-v2/overlay-v2 planner records, executable pair starts/lengths are:
8000: none; 9000: `(0,9000)` (4544+4456); 9319: `(0,9319)` (4672+4647);
16343: `(0,16343)`; 20000: `(0,16384)`; 57544:
`(0,16384),(16384,16384),(32768,16384),(49152,8392)` (4224+4168 tail);
65536: four `16384` pairs. n* is 6848 stock / 8256 overlay. The Python mapping is
`crossover_model.py executable_plan`/`schedule`; Rust is checked against
`golden-exec.json` generated by `gen_goldens.py` (see Fixtures).

### Installing and testing auto on hipx (parent hardware gate)

After CPU acceptance, install the **same** release CLI/daemon and matching
schema-3 overlay calibration. These commands are for the parent, not the CPU worker:

```sh
ssh hipx 'mkdir -p /home/kaden/pm-wave/npu-trunk/bin-auto ~/.hipfire/npu-trunk'
scp target/release/hipfire hipx:/home/kaden/pm-wave/npu-trunk/bin-auto/hipfire
scp target/release/hipfire-daemon hipx:/home/kaden/pm-wave/npu-trunk/bin-auto/hipfire-daemon
scp /home/kaden/qcal/release-0.4.2/npu-trunk/cal/g80bf16-overlay-v2.json \
  hipx:/home/kaden/.hipfire/npu-trunk/calibration.json
scp /home/kaden/qcal/release-0.4.2/npu-trunk/serve_ttft.py \
  hipx:/home/kaden/pm-wave/npu-trunk/bin-auto/serve_ttft.py
```

On hipx, verify `uname -r` is `7.3.0-rc7-perfopt+`, GPU BDF is
`0000:c0:00.0`, and `cat /sys/class/accel/accel0/device/fw_version` matches the
record. Do not rewrite a measured device/kernel key to hide a mismatch.
Use the verified overlay-bound model; for stock instead install the stock
record and set `HIPFIRE_QWEN4_TRUNK_OVERLAY=off`.
Do **not** inherit the CPU gate's visibility masks (`-1`); use the Halo device
selection from the existing hipx campaign. Keep the default session cache.

```sh
export HIPFIRE_DAEMON_BIN=/home/kaden/pm-wave/npu-trunk/bin-auto/hipfire-daemon
export HIPFIRE_QWEN4_NPU_TRUNK_CAL=/home/kaden/.hipfire/npu-trunk/calibration.json
export MODEL=/home/kaden/pm-wave/npu-trunk/models-ov/qwen3.8-flash-next-gptq3.mq4
export ROCR_VISIBLE_DEVICES=1  # recorded Halo campaign ordinal; require gfx1151/c0:00.0
unset HIP_VISIBLE_DEVICES
HIPFIRE_QWEN4_NPU_TRUNK=auto /home/kaden/pm-wave/npu-trunk/bin-auto/hipfire \
  bench "$MODEL" --ttft --prompt-file /home/kaden/pm-wave/halofn/p16k.txt --runs 3
for mode in auto off; do
  for run in 1 2 3; do
    HIPFIRE_QWEN4_NPU_TRUNK="$mode" python3 /home/kaden/pm-wave/npu-trunk/bin-auto/serve_ttft.py \
      --hipfire /home/kaden/pm-wave/npu-trunk/bin-auto/hipfire --model "$MODEL" \
      --prompt /home/kaden/pm-wave/halofn/p16k.txt --measured 1 --greedy \
      --tag "$mode-$run" --out "/tmp/npu-auto-$mode-$run.json"
  done
done
```

Require load `mode=auto`, matching `cal_key`, all families and n*=8256,
not an off/mismatch reason. Require the actual prompt row count and one request
plan line; 16343 rows must select one 8192+8151 pair. Repeat fresh processes
for `auto` and `off` at 16K/57K, and a greedy prefix-sharing second request
through `serve_ttft.py --greedy` with the same model and default cache.
Confirm sub-8256-row prompts (8000) stay GPU-only; 9319 must plan one rebalanced
4672+4647 pair and 57556 must plan 4224+4180 as its final pair, each with
`on == emulate` greedy output including a prefix hit at the 8192 snapshot taken
inside bank B's Core sub-launch (C1 gate). The previously documented
snapshot and quality/TTFT gates still apply; CPU acceptance is not a silicon claim.


The route requires Flash-Next K=2560 and stock MQ6G256V2 for all eight
input families. After metadata/device/calibration admission, `attach_forward`
downloads stock MQ6 and packs auto's selected S1 `Bf16Direct` panels **before**
applying the pending trunk overlay or developer requantization. Only packed
panels cross the overlay swap under auto; raw bytes are released. Forced
on/emulate retain stock host rows until the successfully applied route selects
their calibration, then pack panels (and the exact row-major W8/Sm/Se twin
for emulation). No extra stock GPU copies or second download are retained.

Both arms still load the beta `<model>.trunk-overlay`. Covered z/a/b use W8 from
the **stock MQ6**, never the replacement QT44 bytes. The existing load line adds
`trunk_route=overlay:gfx1151-zab-v1|stock` (only successfully applied overlays
count) and `zab_source=stock-MQ6-DtoH-before-overlay|stock-MQ6-DtoH`, alongside
`download_ms`, `pack_ms`, resident weight bytes and runtime status.
`off` does no download, packing, allocation or new logging; its existing mode
read moves before overlay application, whose payloads and native route are
unchanged. `on` and admitted `auto` open two fixed 8192-row row-major G80 programs,
resident 13/11-panel chains, and two banks each with registered A plus separate
GDN/QSA C arenas. Hardware open errors fail forced on loads; auto frees partial
resources, logs one refusal and remains GPU-only.

Both `on` and `emulate` use the same deterministic, segment-local pair plan.
Only paired banks use BF16-direct numerics: `on`/`auto` consume NPU results, while
`emulate` computes them with the exact IU8 twin on the GPU inside the same pair executor.
Unpaired chunks/tails, rows below 512, decode and uncovered paths retain the
incumbent GPU projections in **both** modes: z/a/b take the same A4/IU4 overlay
route as `off` wherever that native route applies. No paired input projection
is computed twice: `project_trunk_masked` starts covered families as completed,
then runs only uncovered families through the existing MQ6/A4/IU4 dispatch
and native scratch, in each bank's GPU fragment before Core. The debug assertion
is per family, in addition to the Core/OutProj phase assertion.
Shapes with no selected families keep Whole/native execution in both banks.
Under `on` the GPU packs A, then the mixer core directly reads read-only BF16
`Bf16Arena` views of fused C; **there is no consume/retile pass**. Explicit
`GdnTrunkArenas`/`QsaTrunkArenas` arguments select arena launchers only for
`Core`; omitted views retain the incumbent native scratch path. `emulate`
runs the same layer-major pair executor: its `prepare` runs the IU8 twin
projection into scratch where `on` packs and kicks, and its Core reads that
scratch. Bank B therefore owns private copies of the in-projection outputs
(`NpuBankKind::Emulated`, about 1.1 GiB more than `on`). Out-projections and
GPU recurrence/attention/HC/MoE remain native.

Each arena family is `Option<Bf16Arena>`: GDN's fused convolution/parameter
consumer independently reads qkv from its arena or native BF16/F32 projection,
and a/b from their arenas or native F32 scratch. A zero pitch in the opt-in
kernel ABI denotes contiguous F32 scratch, with the incumbent parameter-core
BF16 rounding. Z selects the existing
native gate/chunk kernel when absent. QSA selects native norm/RoPE, index append
and V cache append per absent family. Emulate scatters only covered twin outputs
to native scratch, runs the identical uncovered projection dispatcher, and
executes the incumbent core; arena readers preserve the same BF16 numeric values.
Calibration selection still groups zab and kvidx because costs are measured
per group; dispatch and column maps support independent family views.

### Bank state and ordering

`on` requires the full 8192-row scratch plan and single-device placement.
Bank B is a load-time view of bank A's layer-local buffers, with only the
three persistent fields allocated privately. Bank A's ownership/capacities
and all `off` allocations remain unchanged.

| Bank-B fields | Storage | Why |
| --- | --- | --- |
| `streams` | Private | HC residual rows persist through every layer and final capture/head; B must not overwrite A's rows. |
| `hc_mixed` | Private | Prepared mixer input survives the other bank's prepare/fragment. Uncovered native projections also read it in Core. |
| `ple_rows` | Private | Initialized token-dependent PLE input survives until its layer's prepare. |
| `token_ids`, `embedding_rot`, `embeddings`, `ple_staged` | Shared | Each bank's initialization consumes these completely before the next initialization on the same stream. MTP uploads its own prompt IDs. |
| `hc_normalized`, `hc_low`, `hc_up`, `rotation`, `gdn_bf16` | Shared | Each prepare/fragment/head consumes its own values before returning; no value is handed across their boundaries. |
| `hc_gates` | Shared | Prepare ends before the mixer, so its HC read cannot fuse with the later HC write. That write computes its gates; MLP HC read/write fusions remain entirely inside one fragment. |
| `projection`, `projection2`, `gdn_a`, `gdn_b`, `gdn_gate`, `gdn_beta`, `gdn_recurrent_output`, `gdn_z`, `gdn_output` | Shared | Native uncovered projections, GDN Core and OutProj finish every use inside G(l,bank). |
| `qsa_index`, `qsa_qgate`, `qsa_k`, `qsa_v`, `qsa_output`, `qsa_selected`, `attention_output` | Shared | QSA transform/cache/index/attention and the attention HC write finish inside the fragment. |
| `ple_key`, `ple_value`, `ple_query`, `ple_gated`, `ple_normed`, `ple_output` | Shared | The PLE prepare consumes these before the bank switch. `ple_normed` is retained for ordinary verify rollback, not prompt pairs; pair snapshots copy fixed convolution state instead. |
| `router_logits`, `moe_x_rot`, `moe_gate_up`, `moe_gate`, `moe_up`, `moe_hidden`, `ffn_output`, `moe_shared_output`, `moe_gate_batch`, `moe_up_batch`, `moe_rot_batch`, `moe_topk_indices`, `moe_topk_weights`, `moe_down_expanded`, `moe_scalar`, `moe_expert_token_counts`, `moe_expert_offsets`, `moe_sorted_slot_index`, `moe_expert_tile_ids`, `moe_inverse_perm`, `moe_y_gate_up_grouped`, `moe_y_down_grouped` | Shared | Routing, shared/routed expert work, permutation/gather/combine and the FFN HC write finish before the next fragment. |
| `logits` | Shared | Head scratch is reused only after the preceding bank's head has consumed it. Requested logits/capture rows live in separate caller-owned outputs. |

The same-stream order is the sharing contract, not host launch completion.
`pack(l+1,A)` is queued before `G(l,B)`; its source `hc_mixed` is private.
Rotation's temporary F16 input is consumed by that pack before any subsequent
GPU rotation overwrites it. NPU DMA reads only its separately registered,
bank-private A arena, never the shared GPU layer intermediates.
Capture/MTP copies each bank's private `streams` into its own output row range
after both trunks finish. GDN recurrence/convolution, QSA KV/index caches and
PLE convolution state remain shared model state: B observes A's ordered updates.
Opt-in execution uses mapped experts, not donor-layer staging.
The private NPU owner closes the runtime and frees bank B before bank A;
borrowed fields are never freed by B. None of railgun's internal HIP Rc
resources escapes the exclusive owner.

### Load footprint

Only `emulate` uploads the row-major W8 twin and its weight Sm/Se scales,
or allocates its packed-A and I32 `sums` buffers. Physical `on`/admitted `auto`
retain NPU panels instead; tails, unpaired chunks and uncovered projections
use the native GPU route. Two compact activation sidecar buffers (32 KiB
each at 8192 rows) remain required by the GPU A-pack producer in both modes;
they are not the twin's weight scales.
The load log reports `resident_bytes=npu_panels_bytes+gpu_twin_bytes` for
resident weights, plus `bank_b_bytes` for the new private scratch allocation.

The Flash-Next **full-family baseline**, at 8192 rows (36 GDN, 12 QSA), saves:

| Removed allocation | Bytes |
| --- | ---: |
| Bank-B layer-local scratch (`Full` plan 8,957,998,980 minus private 503,316,480) | 8,454,682,500 |
| GPU W8 twin (`(36*16640+12*14080)*2560`) | 1,966,080,000 |
| Twin weight Sm/Se (`(36*16640+12*14080)*4`) | 3,072,000 |
| Twin I32 sums (`8192*16640*4`) | 545,259,520 |
| Twin packed A (`4*32*40*4608`) | 23,592,960 |
| **Total** | **10,992,686,980 (10.237738 GiB)** |

The twin-only subtotal is 2,538,004,480 bytes (2.363701 GiB); bank-B sharing
saves 7.874037 GiB and leaves 480 MiB private. Subset panel/twin savings are
additional and mask-dependent; the log reflects the actual selected mask.
These are allocation-plan bytes, not a fresh measurement of GTT residency.
CPU tests `npu_bank_sharing_covers_every_field_and_never_crosses_a_bank_switch`
and `npu_private_bank_bytes_are_pinned_at_full_and_partial_rows` enumerate all
58 fields, check produced-value lifetimes against the actual A/B event order,
and pin private bytes for 8192, 8151 and 512 rows.

Sharing is unsafe if a future change spans a prepare/mixer boundary with an
HC fusion, stages projections before the other bank's fragment, adds a
cross-bank stream, or retains a layer-local value for later capture/rollback.
Such a change must update the exhaustive lifetime table and private set.
Partial B changes its view metadata only; A's tensor extents and capacities
stay intact. Numerical operations/order and external capture offsets are
unchanged, but full-model `on`/`emulate` byte parity and GTT savings still
require the parent's Halo hardware gate.

With an 8192-row chunk cap, a natural pair is 8192+B whenever at least 8192+512
rows remain: bank A has 8192 live rows and B has `min(remaining-8192,8192)`. The
planner's rebalanced last pair (two banks of about equal rows, 64-row aligned
split, each at least `max(512, min_bank_rows)`, e.g. 9,319 as 4672+4647 or the
57K tail 8404 as 4224+4180) runs the same executor: both banks bind their live
rows (`bind_bank_rows`; bank A's full extents are restored after the pair) and
bank B's state view advances by A's rows. `emulate` runs the same banks. A shorter unpaired chunk remains sequential. For
example, 16200 rows are one 8192+8008 pair, 20000 are a full pair plus 3616
sequential rows, and 40000 are two full pairs plus 7232 sequential rows. For
each layer:

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
`wait_gpu` now enqueues the per-bank monotonic GEQ terminal-signal wait and
system acquire event immediately, without a host-completion gate; retirement
also records its event without waiting on the host. The following complete `G`
fragment is therefore issued ahead of NPU completion. The relay remains serial.
Bounded host checks at bank reuse protect the ready-event generation, but depend
only on previously issued A production/NPU work, never an unqueued fragment or
the just-enqueued `G`. In `prepare/kick(l+1,A)` the checked chain is NPU(l,A),
whose terminal signal must precede GPU G(l,A): this dependency resolves by
G(l,A)'s start, leaving the host free to enqueue wait(l,B)+G(l,B) while G(l,A)
runs. Retirement itself is a GPU stream dependency, not a host wait for G(l,A).

Every kick receives a terminal signal, including relay errors, a 300 ms
chain-only watchdog (5x the measured ~60 ms chain), relay unwind and shutdown.
The registered signal is terminal rather than success-only; failures are sticky
and stored before releasing the GPU. Retire checks known errors, and pair-end
waits/checks finish before heads/capture or state commits. Failed pair results
are discarded and the request errors; late success cannot clear a failure.
Shutdown releases pending waits before joining the relay, then checks retirement
before unregistering/freeing (or quarantines still-live operands).

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
Before building a partial B program, row-checked bank-B extents are rebound
from the existing `Full` plan for live rows (including grouped MoE, router,
PLE staging/output). Four dedicated native padding buffers—HcMixed, GdnZ,
GdnA and GdnB—retain their original max-chunk logical capacities, exactly as
sequential `off`/`emulate` prefill. Mixer operations still pass live rows:
8192/512 take dense IU4 when admitted; 8151 takes the same padded ragged IU4
route (or the same native fallback when ragged is disabled) in both modes.
The pair and sequential forward share the native prefill-scope gate and
restore it on success/error. Padded input clearing touches private HcMixed;
shared z/a/b outputs are produced and consumed only inside one GPU fragment,
never across prepare. Allocation/ownership stay unchanged, with no extra
allocation/copy. CPU coverage pins decisions and extents at 8192, 8151 and 512.

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

`on` and admitted `auto` double `spec_hidden` to 16384 rows; verify windows and MTP append
scratch keep their original capacities. MTP head append still runs separate
A/B subchunks with the original fill-row boundaries and prompt-ID uploads:
bank B reads hidden rows starting at its bank offset in every mode.
Pending hidden names the last row of B.
Only the final bank of a prompt-ending pair runs the target pick; other pairs
capture silently. Forced modes log their first scheduled pair once:
`npu-trunk: pair rows=8192+8151 position=0` for a 16343-row fresh prompt.

A pair may carry **one interior session snapshot row**, anywhere in the group
(C1, 0.4.2). At the A/B split bank A runs whole and its fixed state is staged
after it, as before. Inside a bank (`CutSite::Bank`), that bank's Core runs as
two launches `[0,r)` and `[r,rows)` (`execute_*_core_rows`, `CoreWindow`):
GDN: each launch ends with the ordinary chunk-end Q8 round trip (frame
`position + lo + rows - 1`) and the conv ring cursor `(start + lo) % 3`; QSA:
the second launch starts from `next_lengths_after(r)`; the PLE grouped depthwise
step of that bank splits at the same row. The staged copies (GDN recurrent +
conv state, QSA selected indices, PLE conv state) are taken after the first
launch and never after the second; the layer's input projections, output
projection, HC and MoE stay bank-wide. A second interior snapshot row ends the
group at the first boundary as an ordinary sequential bank. `emulate` runs the
very same `pair_scoped` (same banks, same Core windows, same staging and
publishing: `launch_geometry` is the single source), with its GPU twin
projection in place of pack + kick, so `on == emulate` holds by construction;
the Halo comparison must retain the default cache and exercise a later prefix
hit at 8192 after a rebalanced pair. Partial windows launch under the routing
of a canonical whole chunk (`with_bank_routing`), so no invariant-profile
setting is needed and banks must have at least 512 rows.
See the staging layout and cost details below.
Generic resumed target advances whose pending snapshot cursor is already behind
their position remain native until that snapshot is reported by its owner.

### Overlap trace

With `HIPFIRE_QWEN4_NPU_TRUNK_TRACE=/tmp/npu-trunk.jsonl`, successful pairs
append one JSON object per `(layer,bank)` with `position`, real live `rows`, `seq`,
`pack_issue_ns`, `kick_ns`, `npu_submit_ns`, `npu_done_ns`,
`gpu_wait_enqueue_ns`, `consume_issue_ns`, `mixer_start_ns`, `mixer_end_ns`
and `anchor_uncertainty_ns`. Host/relay timestamps use Linux CLOCK_MONOTONIC.
Fine fields `prep_begin_ns`, `prep_end_ns`, `pack_end_ns`, `wait_begin_ns`,
`wait_end_ns`, and `signal_write_ns` separate GPU HC/PLE/rotation preparation,
packing, the device wait plus system acquire, and relay terminal publication.
`prep_end_ns` follows rotation and precedes packing; `pack_end_ns` follows the
pack kernel. `signal_write_ns` is sampled after the terminal store and its 64-B
CLFLUSH. All GPU timestamps are after-pair HIP event resolutions, not CPU issue
times. The same-stream pipeline and its bank/snapshot ordering are unchanged.
`consume_issue_ns` is retained for trace compatibility and is always zero.
The mixer interval covers the complete `G` fragment, including HC/MoE.
HIP timing events are resolved only after the pair, relative to a monotonic
anchor calibrated after initialization and **before** pipeline work; its host
bracket width is reported as uncertainty. There is no per-layer GPU sync.
`gpu_wait_enqueue_ns` records the actual enqueue-ahead call and should precede
`npu_done_ns` while work overlaps. Relay submit/done timestamps are harvested at
the next bank reuse, before its generation changes, and at the final pair check;
no host completion lookup is added at rejoin. The transport gate additionally
asserts enqueue-before-done on every exact round and prints NPU-done→GPU-resume
event latency with clock-anchor uncertainty, plus a forced-watchdog release test.
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
calibrated `auto` and variable-size NPU programs remain outside first light.
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
inside `pair()` at the snapshot row. `emulate` runs the same pair executor and
publishes the same way. The next B-end snapshot is ordinary in all modes.

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
alive through stream completion. `ColumnMap` is authoritative for the selected
mask. The ALL baseline has GDN pitch 16640, qkv at 0, z at 10240,
**a at 16384, b at 16432**; QSA pitch 14080, qgate at 0, k at 12288, v at 12800,
index at 13312. Subsets compact these offsets and pad only the fused total.
Qgate is 24 repetitions of
`q[256] | gate[256]`, not a global q/gate split. Index is `query[512] | key[128]`.

All functions below take `gpu` first and return `HipResult<()>`. `p`, `conv`,
`params`, `step`, and `gate` are the corresponding incumbent parameter structs;
all non-projection arguments retain their meaning.

| Variant signature after `gpu` | Projection/output convention | Implementation |
| --- | --- | --- |
| `gated_delta_conv_batched_arena(p, qkv)` | `p.input` ignored; output and history unchanged, including aliasing | hipcc compile-time twin |
| `gated_delta_conv_params_batched_arena(conv, params, qkv, a, b)` | Input/a/b fields ignored; fused convolution and parameter blocks | hipcc compile-time twin |
| `gated_delta_conv_params_qknorm_batched_arena(conv, params, step, keep_qk_conv, qkv, a, b)` | Same fusion; QN/KN go to the incumbent shared scratch | hipcc compile-time twin |
| `gated_delta_conv_params_batched_mixed(conv, params, qkv?, a?, b?)` | Each absent source reads native scratch; present sources use immutable BF16 views | hipcc opt-in twin |
| `gated_delta_conv_params_qknorm_batched_mixed(conv, params, step, keep_qk_conv, qkv?, a?, b?)` | Same mixed sources and QN/KN scratch contract | hipcc opt-in twin |
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

The mixed oracle additionally uses a compact qkv-only pitch of 10240 with
native F32 z/a/b carrying nonzero low mantissa bits. It compares fused
convolution/parameters and both Q/K-normalization keep modes, including
history and QN/KN scratch; native F32 projections retain the incumbent core's
BF16 rounding at parameter consumption, not at projection storage.

Replay follows the incumbent position contracts: cache/raw-key stores and
RoPE positions are bound to replay position; compact F32 scratch-tail writes
and the lab copy primitive's concrete destination offset remain position-independent.
