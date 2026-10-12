# Flash-Next NPU trunk: CPU preparation

This crate owns the CPU-only S1 preparation for hipfire 0.4.2. It never opens an
NPU or GPU, launches hardware, or schedules prefill. Transport and planning are
separate slices and are intentionally absent.

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

`pack_shape` transposes W8 into panels and invokes pm-npu's canonical G80 B
packing through its public `ArrayDesign::pack_in` API. That API also creates a
packed A buffer, which is discarded during weight preparation; no private
layout is duplicated here. `PackedB::ws` contains the padded fused scales.
`pack_trunk` preserves layer order and stores per-layer shape/panels and scales.

The ordinary pack functions select `EpilogueContract::I32Fold`. The explicit
`*_with_contract` variants and program builders reject `Bf16Direct` with
`Bf16Direct pending probe B0`, before preparing data.

## Programs

`build_programs(contract, rows_max)` makes one G80 I32/Fast design per shape,
M rounded up to 256, N=1280, K=2560. `rows_max` must be 1..=65536.
`build_programs_for_families` uses the same compacted maps as weight packing.
Each shape record reports panel count, padded width, per-submit A/B/C argument
sizes and per-submit host bytes read/written. The designs include PDI and TXN
bytes; panels reuse their shape's design.

## CPU checks

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
CARGO_TARGET_DIR=/home/kaden/ClaudeCode/warpfront/wt-npu-trunk/target \
nice -n 19 cargo test --release -p hipfire-npu-trunk -j 12
```

The test suite includes independent MQ6/W8 oracles, a copied lab CPU twin,
column-map/mask arithmetic and a functional G80 simulator test. The ignored
`model_sanity` test uses a CPU HFQM reader and prints weight-scale statistics
and pack time for layer 0 GDN qkv and layer 3 QSA q from the pinned
`qwen3.8-flash-next-gptq3.mq4` artifact (recorded MD5 prefix `be007fc3`).
Set `HIPFIRE_NPU_TRUNK_TEST_MODEL` to override its default model path and run
`cargo test --release -p hipfire-npu-trunk --test model_sanity -- --ignored --nocapture`.
