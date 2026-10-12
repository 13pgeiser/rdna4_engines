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
