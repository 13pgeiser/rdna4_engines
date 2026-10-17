// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G0 GDN recurrent-state precision oracle: native-MTP verify capture and
//! rollback against single-row autoregressive (AR) decode, per state format.
//!
//! Geometry is the Qwen4 tier-C GDN layer: 16 key heads, 48 value heads,
//! key = value = 128, packed q|k|v width 10240. For every window of
//! `rows = 1..=max_rows` consecutive rows the oracle compares
//!
//! * the **reference**: the source state copied to a separate AR state and
//!   advanced by `rows` single-row `gated_delta_step` launches (the production
//!   unfused recurrent output), saving the state bytes after every row, with
//! * the **candidate**: one `gated_delta_step_batched` launch with
//!   `row_states = Some(ring half)` (the verify capture), and
//! * the **rollback**: `gated_delta_rollback_layers` re-running the first
//!   `keep` captured rows from the packed recurrence input
//!   (`[projection rows | gate rows | beta rows]`, one `[input, ring]` pointer
//!   pair in the table).
//!
//! The ring holds `2R` slots (`R = max(max_rows, 8)`, the production capture
//! capacity), the source state lives in slot 0 and is the production
//! `capture_base` rule's "live" slot: the capture writes the half that does not
//! hold the source (`base = R` when `source < R`, else `0`), the candidate view
//! is `rows` slots at `base`, its final state lands in `base + rows - 1`, and
//! after every keep of the window the source advances to that final slot so the
//! half alternates. Windows are consecutive (`>= 2`), each continuing from the
//! previous window's final state at the next absolute positions.
//!
//! Per window the candidate runs **alone first**, twice, and its outputs, final
//! slot and source slot are byte-checked before any rollback can rewrite the
//! final slot (a full-keep rollback would otherwise mask a capture bug). Each
//! of those findings is aggregated into every keep case of the window. Then, for
//! each keep, candidate + rollback run twice (`keep = 0` is the candidate-only
//! result: no rollback, source untouched). The checks per case:
//!
//! * batched outputs equal AR row by row (bitwise);
//! * the final capture slot equals AR's state after the last row;
//! * the rollback target slot `base + keep - 1` equals AR's state after row
//!   `keep - 1`;
//! * the rollback's discarded outputs equal AR's first `keep` rows;
//! * the source slot is byte-identical to its pre-window bytes;
//! * the second identical run reproduces every compared region byte for byte
//!   (same-route determinism);
//! * no output or reference state is non-finite.
//!
//! Output, discard, final slot and rollback target slot are poisoned before each
//! run (outside any graph) so unwritten bytes cannot pass; the source is never
//! reset. With `--graph 1` the candidate-only check uses a candidate-only
//! hipGraph and every keep case a candidate+rollback hipGraph (two kernels
//! captured, instantiated, replayed twice; kernels are warmed eagerly before
//! capture, but every measured candidate and rollback launch is a graph
//! replay, never an eager fallback). The `GraphState` owns one live graph, so
//! each is dropped before the next capture and before any buffer is freed.
//!
//! Route flags set like the production verify forward: `qwen4_scope = true`,
//! `prefill_invariant_scope = false`, `wide_verify_rows = R` for the batched
//! launches and `0` for AR. The eager few-row limit (`max(8, R)`) and the limit
//! inside a graph capture (stays 8) are both printed and written to the JSON.
//!
//! Inputs copy the tier-C generator ranges (`qwen4_tierc/gdn.rs::make_steps`):
//! q|k uniform in [-2, 2], v in [-1, 1] (BF16-rounded), gate in [-0.3, -0.02],
//! beta in [0.1, 0.9]; the initial F32 state is small uniform noise, converted
//! to Q8 by the production `gdn_state_f32_to_q8` kernel (launched exactly like
//! the private `gdn_state_convert` wrapper). The same data feeds every format.
//! `max|state|` per format is taken over the decoded AR snapshots (Q8 decoded
//! by `gdn_state_q8_to_f32`); candidate states are bytewise equal to them
//! whenever the case passes.
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab --example \
//!     gdn_state_precision_oracle -- --formats q8,fp32 --graph 1 \
//!     --max-rows 32 --all-keeps [--seed N] [--windows W] [--json PATH]
//! ```
//!
//! `--formats` takes `q8` and `fp32` (`q8_ef` is refused: not available in this
//! build); `--graph 0|1` (default 0); `--max-rows 1..=32` (default 32, the
//! production wide-verify maximum); `--windows >= 2` (default 3); without
//! `--all-keeps` only keep 0 and the full keep run per window.
//!
//! Exit 0 = every case matched; 1 = at least one mismatch (one stdout line per
//! `(format, graph, rows, keep, window)` with the first differing byte offset
//! and kind/row); 2 = bad usage or an unsupported GPU/graph/setup, named on
//! stderr. Per format a final line prints
//! `oracle: format=<f> graph=<g> cases=<n> mismatches=<m> kernels=<names>`.

use hip_bridge::KernargBlob;
use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
use rdna_compute::tensor_ops::{
    gated_delta_rollback_layers, gated_delta_step, gated_delta_step_batched,
    GatedDeltaRollbackLayers, GatedDeltaStep, GatedDeltaStepBatched, GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::collections::BTreeSet;
use std::path::PathBuf;

const KH: usize = 16;
const VH: usize = 48;
const HD: usize = 128;
const QK: usize = KH * HD;
const VW: usize = VH * HD;
const QKV: usize = 2 * QK + VW;
const STATE_ELEMS: usize = VH * HD * HD;
/// `qwen4_tierc/gdn.rs::START_POSITION`.
const START_POSITION: usize = 5;
/// `qwen4_tierc/gdn.rs` step seed.
const DEFAULT_SEED: u64 = 0x6764_6e04;
const DEFAULT_WINDOWS: usize = 3;
const POISON_OUTPUT: u8 = 0xFF;
const POISON_STATE: u8 = 0xA5;

/// `rdna-compute/src/tensor_ops.rs::TENSOR_OPS_SRC`, same concatenation so the
/// kernel module is the one the production launchers load.
const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);

const USAGE: &str = "usage: gdn_state_precision_oracle [--formats q8,fp32] [--graph 0|1] \
[--max-rows 1..=32] [--all-keeps] [--seed N] [--windows W>=2] [--json PATH]";

type Res<T> = Result<T, String>;

fn ctx<T, E: std::fmt::Display>(result: Result<T, E>, what: &str) -> Res<T> {
    result.map_err(|e| format!("{what}: {e}"))
}

// ═══════════════════════════ CLI ══════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
struct Cli {
    formats: Vec<GdnStateFormat>,
    graph: bool,
    max_rows: usize,
    all_keeps: bool,
    seed: u64,
    windows: usize,
    json: Option<PathBuf>,
}

fn label(format: GdnStateFormat) -> &'static str {
    match format {
        GdnStateFormat::F32 => "fp32",
        GdnStateFormat::Q8 => "q8",
    }
}

fn parse_format(name: &str) -> Res<GdnStateFormat> {
    match name {
        "q8" => Ok(GdnStateFormat::Q8),
        "fp32" | "f32" => Ok(GdnStateFormat::F32),
        "q8_ef" => Err("format q8_ef is not available in this build".to_string()),
        other => Err(format!("unknown format '{other}' (expected q8 or fp32)")),
    }
}

fn parse_number(flag: &str, value: &str) -> Res<u64> {
    let parsed = match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(&hex.replace('_', ""), 16),
        None => value.replace('_', "").parse::<u64>(),
    };
    parsed.map_err(|_| format!("{flag}: '{value}' is not an unsigned integer"))
}

fn parse_args(args: &[String]) -> Res<Cli> {
    let mut formats: Option<Vec<GdnStateFormat>> = None;
    let mut graph = false;
    let mut max_rows = WIDE_VERIFY_MAX_ROWS;
    let mut all_keeps = false;
    let mut seed = DEFAULT_SEED;
    let mut windows = DEFAULT_WINDOWS;
    let mut json = None;
    let mut index = 0;
    while index < args.len() {
        let (flag, inline) = match args[index].split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (args[index].clone(), None),
        };
        index += 1;
        let value = |index: &mut usize| -> Res<String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            let v = args
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))?;
            *index += 1;
            Ok(v)
        };
        match flag.as_str() {
            "--all-keeps" => {
                if inline.is_some() {
                    return Err("--all-keeps takes no value".to_string());
                }
                all_keeps = true;
            }
            "--formats" => {
                let raw = value(&mut index)?;
                let mut parsed = Vec::new();
                for name in raw.split(',') {
                    let format = parse_format(name.trim())?;
                    if parsed.contains(&format) {
                        return Err(format!("format '{}' listed twice", name.trim()));
                    }
                    parsed.push(format);
                }
                formats = Some(parsed);
            }
            "--graph" => {
                graph = match value(&mut index)?.as_str() {
                    "0" => false,
                    "1" => true,
                    other => return Err(format!("--graph must be 0 or 1, got '{other}'")),
                };
            }
            "--max-rows" => {
                let raw = value(&mut index)?;
                max_rows = parse_number("--max-rows", &raw)? as usize;
                if max_rows == 0 || max_rows > WIDE_VERIFY_MAX_ROWS {
                    return Err(format!(
                        "--max-rows {raw} is outside 1..={WIDE_VERIFY_MAX_ROWS} \
                         (the production wide-verify maximum)"
                    ));
                }
            }
            "--seed" => {
                let raw = value(&mut index)?;
                seed = parse_number("--seed", &raw)?;
            }
            "--windows" => {
                let raw = value(&mut index)?;
                windows = parse_number("--windows", &raw)? as usize;
                if windows < 2 {
                    return Err(format!(
                        "--windows {raw} is below 2: the oracle needs consecutive windows"
                    ));
                }
            }
            "--json" => json = Some(PathBuf::from(value(&mut index)?)),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Cli {
        formats: formats.unwrap_or_else(|| vec![GdnStateFormat::Q8, GdnStateFormat::F32]),
        graph,
        max_rows,
        all_keeps,
        seed,
        windows,
        json,
    })
}

// ═══════════════════════════ pure helpers ═════════════════════════════════════

/// `GdnStateOwner::capture_base`: the ring half that does not hold `source`.
fn capture_base(source: usize, ring_rows: usize) -> usize {
    if source < ring_rows {
        ring_rows
    } else {
        0
    }
}

fn keeps_for(rows: usize, all_keeps: bool) -> Vec<usize> {
    if all_keeps {
        (0..=rows).collect()
    } else {
        vec![0, rows]
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Byte offset of the first non-finite F32 in `bytes`.
fn first_nonfinite(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(4)
        .position(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        .map(|i| i * 4)
}

/// Largest finite `|x|` of the F32 `bytes`, and the offset of the first
/// non-finite value.
fn state_stats(bytes: &[u8]) -> (f64, Option<usize>) {
    let mut max = 0.0f64;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if value.is_finite() {
            max = max.max((value as f64).abs());
        }
    }
    (max, first_nonfinite(bytes))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// RNE F32 -> BF16 bits (`qwen4_tierc/common.rs::f32_to_bf16_bits`).
fn f32_to_bf16_bits(value: f32) -> u16 {
    let u = value.to_bits();
    if value.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits((f32_to_bf16_bits(value) as u32) << 16)
}

/// SplitMix64 (`qwen4_tierc/common.rs::Rng`).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
}

struct Steps {
    proj: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

/// `qwen4_tierc/gdn.rs::make_steps` ranges (no zeroed gates); the projection
/// is BF16-rounded like the convolution output the route reads.
fn make_steps(rows: usize, seed: u64) -> Steps {
    let mut rng = Rng::new(seed);
    let mut proj = Vec::with_capacity(rows * QKV);
    for _ in 0..rows {
        proj.extend(rng.vec(2 * QK, -2.0, 2.0));
        proj.extend(rng.vec(VW, -1.0, 1.0));
    }
    for value in &mut proj {
        *value = bf16_round(*value);
    }
    let mut gate = Vec::with_capacity(rows * VH);
    for _ in 0..rows * VH {
        gate.push(rng.range(-0.3, -0.02));
    }
    let beta = rng.vec(rows * VH, 0.1, 0.9);
    Steps { proj, gate, beta }
}

/// Seed of one `(rows, window)` input block, shared by every format.
fn window_seed(seed: u64, rows: usize, window: usize) -> u64 {
    seed ^ ((rows as u64) << 32) ^ ((window as u64 + 1) << 8) ^ DEFAULT_SEED
}

fn state_seed(seed: u64, rows: usize) -> u64 {
    seed.rotate_left(17) ^ ((rows as u64) << 40) ^ 0x5747_5354_4154_4530
}

// ═══════════════════════════ findings ═════════════════════════════════════════

#[derive(Debug, Clone)]
struct Failure {
    kind: String,
    row: Option<usize>,
    byte_offset: usize,
    detail: String,
}

impl Failure {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "kind": self.kind,
            "row": self.row,
            "byte_offset": self.byte_offset,
            "detail": self.detail,
        })
    }
}

/// First difference between `observed` and `expected`, if any.
fn compare(
    kind: &str,
    observed: &[u8],
    expected: &[u8],
    row_bytes: Option<usize>,
) -> Option<Failure> {
    if observed.len() != expected.len() {
        return Some(Failure {
            kind: kind.to_string(),
            row: None,
            byte_offset: observed.len().min(expected.len()),
            detail: format!("length {} != expected {}", observed.len(), expected.len()),
        });
    }
    first_diff(observed, expected).map(|offset| Failure {
        kind: kind.to_string(),
        row: row_bytes.map(|bytes| offset / bytes),
        byte_offset: offset,
        detail: format!(
            "observed 0x{:02x} expected 0x{:02x}",
            observed[offset], expected[offset]
        ),
    })
}

fn nonfinite(kind: &str, bytes: &[u8], row_bytes: usize) -> Option<Failure> {
    first_nonfinite(bytes).map(|offset| Failure {
        kind: kind.to_string(),
        row: Some(offset / row_bytes),
        byte_offset: offset,
        detail: "non-finite F32".to_string(),
    })
}

struct CaseResult {
    rows: usize,
    keep: usize,
    window: usize,
    failures: Vec<Failure>,
}

struct FormatReport {
    format: GdnStateFormat,
    cases: Vec<CaseResult>,
    max_abs_state: f64,
    kernels: BTreeSet<String>,
}

impl FormatReport {
    fn mismatches(&self) -> usize {
        self.cases.iter().filter(|c| !c.failures.is_empty()).count()
    }
}

// ═══════════════════════════ device plumbing ══════════════════════════════════

fn observe(gpu: &Gpu, kernels: &mut BTreeSet<String>) {
    if let Some(name) = gpu.last_launched_kernel() {
        kernels.insert(name.to_string());
    }
}

fn sync(gpu: &Gpu) -> Res<()> {
    ctx(gpu.hip.device_synchronize(), "device synchronize")
}

fn download(gpu: &Gpu, tensor: &GpuTensor) -> Res<Vec<u8>> {
    sync(gpu)?;
    ctx(gpu.download_raw_bytes(tensor), "download")
}

fn fill(gpu: &Gpu, tensor: &GpuTensor, byte: u8) -> Res<()> {
    sync(gpu)?;
    ctx(
        gpu.hip
            .memcpy_htod(&tensor.buf, &vec![byte; tensor.byte_size()]),
        "poison",
    )
}

fn upload(gpu: &Gpu, tensor: &GpuTensor, data: &[f32]) -> Res<()> {
    sync(gpu)?;
    ctx(
        gpu.hip
            .memcpy_htod(&tensor.sub_offset(0, data.len()).buf, &f32_bytes(data)),
        "upload",
    )
}

/// The production `gdn_state_convert` launch (`tensor_ops.rs`): `gdn_state_q8_to_f32`,
/// or `gdn_state_f32_to_q8` with its requantization `frame`; grid
/// `value_heads * 128` value rows, block 128.
fn gdn_state_convert(
    gpu: &mut Gpu,
    kernel: &str,
    source: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    frame: Option<u32>,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    ctx(
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel),
        kernel,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(source);
    args.push_ptr(destination);
    args.push_i32(VH as i32);
    if let Some(frame) = frame {
        args.push_u32(frame);
    }
    args.pad_to(16);
    ctx(
        gpu.launch_blob_recorded(
            kernel,
            [(VH * 128) as u32, 1, 1],
            [128, 1, 1],
            0,
            args.as_mut_slice(),
            Default::default(),
        ),
        kernel,
    )?;
    observe(gpu, kernels);
    Ok(())
}

/// Capture `body` into the `GraphState`'s one live graph (dropping the previous
/// one first). Returns the few-row limit seen while capturing. The launches must
/// have been warmed eagerly: nothing executes during capture.
fn capture_graph(
    gpu: &mut Gpu,
    expected_launches: usize,
    body: impl FnOnce(&mut Gpu) -> Res<()>,
) -> Res<usize> {
    sync(gpu)?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    {
        let stream = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "graph capture: no active stream".to_string())?;
        ctx(
            gpu.graphs
                .begin_graph_capture(&gpu.hip, gpu.device_id, stream),
            "begin graph capture",
        )?;
    }
    let result = body(gpu);
    let limit = gpu.few_row_limit();
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "graph capture: stream vanished".to_string())?;
    if let Err(error) = result {
        gpu.graphs
            .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
        return Err(format!("graph capture body: {error}"));
    }
    ctx(
        gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, stream),
        "end graph capture",
    )?;
    let launches = gpu.graphs.ar_forward_blobs.len();
    if launches != expected_launches || gpu.graphs.graph_exec.is_none() {
        return Err(format!(
            "graph capture recorded {launches} launches, expected {expected_launches} \
             (or no instantiated graph): not a real capture"
        ));
    }
    Ok(limit)
}

fn replay(gpu: &Gpu) -> Res<()> {
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "graph replay: no active stream".to_string())?;
    ctx(
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stream),
        "graph replay",
    )?;
    sync(gpu)
}

// ═══════════════════════════ rig ══════════════════════════════════════════════

struct Rig {
    format: GdnStateFormat,
    /// `R`: slots per ring half.
    ring_rows: usize,
    /// `format.state_units` (dtype units per state slot).
    units: usize,
    slot_bytes: usize,
    ring: GpuTensor,
    ar_state: GpuTensor,
    proj: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    out: GpuTensor,
    ar_out: GpuTensor,
    discard: GpuTensor,
    recurrence: GpuTensor,
    table: GpuTensor,
    /// F32 `[value_head][key][value]` scratch: Q8 init source and decode target.
    dec: GpuTensor,
}

impl Rig {
    fn new(gpu: &mut Gpu, format: GdnStateFormat, ring_rows: usize) -> Res<Self> {
        let units = format.state_units(VH, HD, HD);
        let slot_bytes = format.state_bytes(VH, HD, HD);
        let dtype = format.dtype();
        let specs = [
            (2 * ring_rows * units, dtype),
            (units, dtype),
            (ring_rows * QKV, DType::F32),
            (ring_rows * VH, DType::F32),
            (ring_rows * VH, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * (QKV + 2 * VH), DType::F32),
            (STATE_ELEMS, DType::F32),
        ];
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(specs.len() + 1);
        for (elements, dtype) in specs {
            match gpu.zeros(&[elements], dtype) {
                Ok(tensor) => owned.push(tensor),
                Err(error) => {
                    free_all(gpu, owned);
                    return Err(format!("allocate {elements} x {dtype:?}: {error}"));
                }
            }
        }
        // `[recurrence input, ring]` native-endian u64 pointer pair (one layer).
        let pointers: Vec<u8> = [owned[8].buf.as_ptr(), owned[0].buf.as_ptr()]
            .iter()
            .flat_map(|p| (*p as usize as u64).to_ne_bytes())
            .collect();
        match gpu.upload_raw(&pointers, &[pointers.len()]) {
            Ok(table) => owned.push(table),
            Err(error) => {
                free_all(gpu, owned);
                return Err(format!("upload rollback table: {error}"));
            }
        }
        let mut it = owned.into_iter();
        let mut next = || it.next().expect("rig tensor");
        Ok(Self {
            format,
            ring_rows,
            units,
            slot_bytes,
            ring: next(),
            ar_state: next(),
            proj: next(),
            gate: next(),
            beta: next(),
            out: next(),
            ar_out: next(),
            discard: next(),
            recurrence: next(),
            dec: next(),
            table: next(),
        })
    }

    fn slot(&self, index: usize) -> GpuTensor {
        self.ring.sub_offset(index * self.units, self.units)
    }

    fn free(self, gpu: &mut Gpu) -> Res<()> {
        let mut first = None;
        for tensor in [
            self.ring,
            self.ar_state,
            self.proj,
            self.gate,
            self.beta,
            self.out,
            self.ar_out,
            self.discard,
            self.recurrence,
            self.dec,
            self.table,
        ] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("free: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn free_all(gpu: &mut Gpu, tensors: Vec<GpuTensor>) {
    for tensor in tensors {
        let _ = gpu.free_tensor(tensor);
    }
}

// ═══════════════════════════ launches ═════════════════════════════════════════

fn ar_step(
    gpu: &mut Gpu,
    rig: &Rig,
    row: usize,
    position: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    ctx(
        gated_delta_step(
            gpu,
            &GatedDeltaStep {
                q: &rig.proj.sub_offset(row * QKV, QK),
                k: &rig.proj.sub_offset(row * QKV + QK, QK),
                v: &rig.proj.sub_offset(row * QKV + 2 * QK, VW),
                gate: &rig.gate.sub_offset(row * VH, VH),
                beta: &rig.beta.sub_offset(row * VH, VH),
                state: &rig.ar_state,
                output: &rig.ar_out.sub_offset(row * VW, VW),
                key_heads: KH,
                value_heads: VH,
                key_dim: HD,
                value_dim: HD,
                position,
            },
        ),
        "gated_delta_step",
    )?;
    observe(gpu, kernels);
    Ok(())
}

fn launch_candidate(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    base: usize,
    rows: usize,
    start: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    gpu.set_wide_verify_rows(rig.ring_rows);
    ctx(
        gated_delta_step_batched(
            gpu,
            &GatedDeltaStepBatched {
                projection: &rig.proj.sub_offset(0, rows * QKV),
                gate: &rig.gate.sub_offset(0, rows * VH),
                beta: &rig.beta.sub_offset(0, rows * VH),
                state: &rig.slot(source),
                output: &rig.out.sub_offset(0, rows * VW),
                row_states: Some(&rig.ring.sub_offset(base * rig.units, rows * rig.units)),
                rows,
                qkv_width: QKV,
                key_heads: KH,
                value_heads: VH,
                key_dim: HD,
                value_dim: HD,
                position: start,
            },
        ),
        "gated_delta_step_batched (capture)",
    )?;
    observe(gpu, kernels);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_rollback(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    base: usize,
    rows: usize,
    keep: usize,
    start: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    gpu.set_wide_verify_rows(rig.ring_rows);
    ctx(
        gated_delta_rollback_layers(
            gpu,
            &GatedDeltaRollbackLayers {
                table: &rig.table,
                discard: &rig.discard.sub_offset(0, keep * VW),
                format: rig.format,
                layers: 1,
                rows,
                keep,
                from: source,
                to: base,
                qkv_width: QKV,
                key_heads: KH,
                value_heads: VH,
                position: start + keep - 1,
            },
        ),
        "gated_delta_rollback_layers",
    )?;
    observe(gpu, kernels);
    Ok(())
}

// ═══════════════════════════ reference ════════════════════════════════════════

struct Reference {
    out: Vec<u8>,
    states: Vec<Vec<u8>>,
    max_abs: f64,
    failures: Vec<Failure>,
}

/// AR decode of `rows` rows from the source slot, row `r` at position
/// `start + r`, saving every row's state bytes.
fn run_reference(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    rows: usize,
    start: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<Reference> {
    gpu.set_wide_verify_rows(0);
    fill(gpu, &rig.ar_out, POISON_OUTPUT)?;
    sync(gpu)?;
    ctx(
        gpu.copy_d2d(&rig.slot(source), &rig.ar_state, rig.slot_bytes),
        "copy source to AR state",
    )?;
    let mut states = Vec::with_capacity(rows);
    let mut max_abs = 0.0f64;
    let mut failures = Vec::new();
    for row in 0..rows {
        ar_step(gpu, rig, row, start + row, kernels)?;
        let bytes = download(gpu, &rig.ar_state)?;
        let (max, bad) = match rig.format {
            GdnStateFormat::F32 => state_stats(&bytes),
            GdnStateFormat::Q8 => {
                gdn_state_convert(
                    gpu,
                    "gdn_state_q8_to_f32",
                    rig.ar_state.buf.as_ptr(),
                    rig.dec.buf.as_ptr(),
                    None,
                    kernels,
                )?;
                state_stats(&download(gpu, &rig.dec)?)
            }
        };
        max_abs = max_abs.max(max);
        if let Some(offset) = bad {
            failures.push(Failure {
                kind: "nonfinite_reference_state".to_string(),
                row: Some(row),
                byte_offset: offset,
                detail: "non-finite AR state (decoded F32 offset)".to_string(),
            });
        }
        states.push(bytes);
    }
    let out = download(gpu, &rig.ar_out.sub_offset(0, rows * VW))?;
    if let Some(f) = nonfinite("nonfinite_reference_output", &out, VW * 4) {
        failures.push(f);
    }
    Ok(Reference {
        out,
        states,
        max_abs,
        failures,
    })
}

// ═══════════════════════════ snapshots and checks ═════════════════════════════

struct Snap {
    out: Vec<u8>,
    discard: Vec<u8>,
    final_state: Vec<u8>,
    kept: Vec<u8>,
    source: Vec<u8>,
}

fn snapshot(
    gpu: &Gpu,
    rig: &Rig,
    rows: usize,
    keep: usize,
    source: usize,
    base: usize,
) -> Res<Snap> {
    let out = download(gpu, &rig.out.sub_offset(0, rows * VW))?;
    let discard = if keep > 0 {
        download(gpu, &rig.discard.sub_offset(0, keep * VW))?
    } else {
        Vec::new()
    };
    let final_state = download(gpu, &rig.slot(base + rows - 1))?;
    let kept = if keep > 0 {
        download(gpu, &rig.slot(base + keep - 1))?
    } else {
        Vec::new()
    };
    let source = download(gpu, &rig.slot(source))?;
    Ok(Snap {
        out,
        discard,
        final_state,
        kept,
        source,
    })
}

fn check_determinism(prefix: &str, a: &Snap, b: &Snap) -> Vec<Failure> {
    [
        compare(&format!("{prefix}nondeterministic_output"), &b.out, &a.out, Some(VW * 4)),
        compare(&format!("{prefix}nondeterministic_discard"), &b.discard, &a.discard, Some(VW * 4)),
        compare(&format!("{prefix}nondeterministic_final_state"), &b.final_state, &a.final_state, None),
        compare(&format!("{prefix}nondeterministic_kept_state"), &b.kept, &a.kept, None),
        compare(&format!("{prefix}nondeterministic_source"), &b.source, &a.source, None),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn check_candidate_only(
    snap: &Snap,
    reference: &Reference,
    rows: usize,
    source_bytes: &[u8],
) -> Vec<Failure> {
    [
        compare("candidate_only_output", &snap.out, &reference.out, Some(VW * 4)),
        nonfinite("candidate_only_nonfinite_output", &snap.out, VW * 4),
        compare("candidate_only_final_state", &snap.final_state, &reference.states[rows - 1], None),
        compare("candidate_only_source_modified", &snap.source, source_bytes, None),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn check_case(
    snap: &Snap,
    reference: &Reference,
    rows: usize,
    keep: usize,
    source_bytes: &[u8],
) -> Vec<Failure> {
    [
        compare("candidate_output", &snap.out, &reference.out, Some(VW * 4)),
        nonfinite("nonfinite_output", &snap.out, VW * 4),
        compare("rollback_discard", &snap.discard, &reference.out[..keep * VW * 4], Some(VW * 4)),
        nonfinite("nonfinite_discard", &snap.discard, VW * 4),
        compare("final_state", &snap.final_state, &reference.states[rows - 1], None),
        compare("kept_state", &snap.kept, &reference.states[keep - 1], None),
        compare("source_modified", &snap.source, source_bytes, None),
    ]
    .into_iter()
    .flatten()
    .collect()
}

// ═══════════════════════════ case runners ═════════════════════════════════════

struct Window<'a> {
    graph: bool,
    source: usize,
    base: usize,
    rows: usize,
    start: usize,
    source_bytes: &'a [u8],
    reference: &'a Reference,
}

/// Candidate alone, twice: byte-checked before any rollback can rewrite the
/// final slot.
fn run_candidate_only(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &Window<'_>,
    kernels: &mut BTreeSet<String>,
    capture_limit: &mut Option<usize>,
) -> Res<Vec<Failure>> {
    if w.graph {
        let limit = capture_graph(gpu, 1, |g| {
            launch_candidate(g, rig, w.source, w.base, w.rows, w.start, kernels)
        })?;
        capture_limit.get_or_insert(limit);
    }
    let mut snaps = Vec::with_capacity(2);
    for _ in 0..2 {
        fill(gpu, &rig.out, POISON_OUTPUT)?;
        fill(gpu, &rig.slot(w.base + w.rows - 1), POISON_STATE)?;
        if w.graph {
            replay(gpu)?;
        } else {
            launch_candidate(gpu, rig, w.source, w.base, w.rows, w.start, kernels)?;
            sync(gpu)?;
        }
        snaps.push(snapshot(gpu, rig, w.rows, 0, w.source, w.base)?);
    }
    let mut failures = check_candidate_only(&snaps[0], w.reference, w.rows, w.source_bytes);
    failures.extend(check_determinism("candidate_only_", &snaps[0], &snaps[1]));
    Ok(failures)
}

/// Candidate + rollback for one `keep >= 1`, twice.
fn run_keep(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &Window<'_>,
    keep: usize,
    kernels: &mut BTreeSet<String>,
    capture_limit: &mut Option<usize>,
) -> Res<Vec<Failure>> {
    if w.graph {
        let limit = capture_graph(gpu, 2, |g| {
            launch_candidate(g, rig, w.source, w.base, w.rows, w.start, kernels)?;
            launch_rollback(g, rig, w.source, w.base, w.rows, keep, w.start, kernels)
        })?;
        capture_limit.get_or_insert(limit);
    }
    let mut snaps = Vec::with_capacity(2);
    for _ in 0..2 {
        fill(gpu, &rig.out, POISON_OUTPUT)?;
        fill(gpu, &rig.discard, POISON_OUTPUT)?;
        fill(gpu, &rig.slot(w.base + w.rows - 1), POISON_STATE)?;
        fill(gpu, &rig.slot(w.base + keep - 1), POISON_STATE)?;
        if w.graph {
            replay(gpu)?;
        } else {
            launch_candidate(gpu, rig, w.source, w.base, w.rows, w.start, kernels)?;
            launch_rollback(gpu, rig, w.source, w.base, w.rows, keep, w.start, kernels)?;
            sync(gpu)?;
        }
        snaps.push(snapshot(gpu, rig, w.rows, keep, w.source, w.base)?);
    }
    let mut failures = check_case(&snaps[0], w.reference, w.rows, keep, w.source_bytes);
    failures.extend(check_determinism("", &snaps[0], &snaps[1]));
    Ok(failures)
}

/// Load the AR, capture, rollback and converter kernels before any capture.
fn prewarm(gpu: &mut Gpu, rig: &Rig, kernels: &mut BTreeSet<String>) -> Res<()> {
    gpu.set_wide_verify_rows(0);
    ar_step(gpu, rig, 0, START_POSITION, kernels)?;
    launch_candidate(gpu, rig, 0, rig.ring_rows, 1, START_POSITION, kernels)?;
    launch_rollback(gpu, rig, 0, rig.ring_rows, 1, 1, START_POSITION, kernels)?;
    if rig.format == GdnStateFormat::Q8 {
        gdn_state_convert(
            gpu,
            "gdn_state_f32_to_q8",
            rig.dec.buf.as_ptr(),
            rig.ar_state.buf.as_ptr(),
            Some(START_POSITION as u32),
            kernels,
        )?;
        gdn_state_convert(
            gpu,
            "gdn_state_q8_to_f32",
            rig.ar_state.buf.as_ptr(),
            rig.dec.buf.as_ptr(),
            None,
            kernels,
        )?;
    }
    sync(gpu)
}

/// Source slot 0 := small random F32 state, as Q8 through the production
/// converter for the Q8 format.
fn init_source(
    gpu: &mut Gpu,
    rig: &Rig,
    seed: u64,
    rows: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    let state = Rng::new(state_seed(seed, rows)).vec(STATE_ELEMS, -0.1, 0.1);
    let slot0 = rig.slot(0);
    match rig.format {
        GdnStateFormat::F32 => upload(gpu, &slot0, &state),
        GdnStateFormat::Q8 => {
            upload(gpu, &rig.dec, &state)?;
            gdn_state_convert(
                gpu,
                "gdn_state_f32_to_q8",
                rig.dec.buf.as_ptr(),
                slot0.buf.as_ptr(),
                Some((START_POSITION - 1) as u32),
                kernels,
            )?;
            sync(gpu)
        }
    }
}

fn run_cases(
    gpu: &mut Gpu,
    cli: &Cli,
    rig: &Rig,
    report: &mut FormatReport,
    capture_limit: &mut Option<usize>,
) -> Res<()> {
    let format_label = label(rig.format);
    let graph = u8::from(cli.graph);
    let mut kernels = BTreeSet::new();
    prewarm(gpu, rig, &mut kernels)?;
    for rows in 1..=cli.max_rows {
        init_source(gpu, rig, cli.seed, rows, &mut kernels)?;
        let mut source = 0usize;
        for window in 0..cli.windows {
            let start = START_POSITION + window * rows;
            let steps = make_steps(rows, window_seed(cli.seed, rows, window));
            upload(gpu, &rig.proj, &steps.proj)?;
            upload(gpu, &rig.gate, &steps.gate)?;
            upload(gpu, &rig.beta, &steps.beta)?;
            let packed: Vec<f32> = steps
                .proj
                .iter()
                .chain(&steps.gate)
                .chain(&steps.beta)
                .copied()
                .collect();
            upload(gpu, &rig.recurrence, &packed)?;
            let base = capture_base(source, rig.ring_rows);
            let source_bytes = download(gpu, &rig.slot(source))?;
            let reference = run_reference(gpu, rig, source, rows, start, &mut kernels)?;
            report.max_abs_state = report.max_abs_state.max(reference.max_abs);
            let w = Window {
                graph: cli.graph,
                source,
                base,
                rows,
                start,
                source_bytes: &source_bytes,
                reference: &reference,
            };
            let mut window_failures = reference.failures.clone();
            window_failures.extend(run_candidate_only(
                gpu,
                rig,
                &w,
                &mut kernels,
                capture_limit,
            )?);
            for keep in keeps_for(rows, cli.all_keeps) {
                let mut failures = window_failures.clone();
                if keep > 0 {
                    failures.extend(run_keep(gpu, rig, &w, keep, &mut kernels, capture_limit)?);
                }
                if let Some(first) = failures.first() {
                    println!(
                        "MISMATCH format={format_label} graph={graph} rows={rows} keep={keep} \
                         window={window} first_kind={} byte_offset={} row={} failures={} \
                         kinds={} detail=\"{}\"",
                        first.kind,
                        first.byte_offset,
                        first.row.map_or("-".to_string(), |r| r.to_string()),
                        failures.len(),
                        failures
                            .iter()
                            .map(|f| f.kind.as_str())
                            .collect::<Vec<_>>()
                            .join("|"),
                        first.detail,
                    );
                }
                report.cases.push(CaseResult {
                    rows,
                    keep,
                    window,
                    failures,
                });
            }
            // The committed final slot is the next window's source.
            source = base + rows - 1;
        }
        eprintln!("{format_label} graph={graph} rows={rows}/{} done", cli.max_rows);
    }
    report.kernels.extend(kernels);
    Ok(())
}

fn run_format(
    gpu: &mut Gpu,
    cli: &Cli,
    format: GdnStateFormat,
    ring_rows: usize,
    capture_limit: &mut Option<usize>,
) -> Res<FormatReport> {
    let rig = Rig::new(gpu, format, ring_rows)?;
    let mut report = FormatReport {
        format,
        cases: Vec::new(),
        max_abs_state: 0.0,
        kernels: BTreeSet::new(),
    };
    let result = run_cases(gpu, cli, &rig, &mut report, capture_limit);
    gpu.set_wide_verify_rows(0);
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    let freed = rig.free(gpu);
    result?;
    freed?;
    Ok(report)
}

// ═══════════════════════════ main ═════════════════════════════════════════════

fn run(cli: &Cli) -> Res<i32> {
    let mut gpu = ctx(Gpu::init(), "Gpu::init")?;
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        return Err(format!(
            "arch {} has no gfx11+ SIMT row-capture GDN route (the production verify capture \
             is unavailable here)",
            gpu.arch
        ));
    }
    if cli.graph {
        ctx(gpu.ensure_capture_stream(), "graph route: capture stream")?;
    }
    let arch = gpu.arch.clone();
    // The production verify forward's scope (`gpu_forward.rs`).
    gpu.qwen4_scope = true;
    gpu.prefill_invariant_scope = false;
    let ring_rows = cli.max_rows.max(FEW_ROW_ROUTE_ROWS);
    gpu.set_wide_verify_rows(ring_rows);
    let eager_limit = gpu.few_row_limit();
    gpu.set_wide_verify_rows(0);
    println!(
        "gdn_state_precision_oracle arch={arch} geometry=KH{KH}/VH{VH}/K{HD}/V{HD}/QKV{QKV} \
         graph={} max_rows={} ring_rows={ring_rows} windows={} all_keeps={} seed={} \
         qwen4_scope=true prefill_invariant_scope=false wide_verify_rows={ring_rows} \
         few_row_limit_eager={eager_limit}",
        u8::from(cli.graph),
        cli.max_rows,
        cli.windows,
        cli.all_keeps,
        cli.seed,
    );
    let mut capture_limit: Option<usize> = None;
    let mut reports = Vec::new();
    let mut failed = None;
    for &format in &cli.formats {
        match run_format(&mut gpu, cli, format, ring_rows, &mut capture_limit) {
            Ok(report) => {
                println!(
                    "state format={} max_abs_state={:e} state_units={} state_bytes={}",
                    label(format),
                    report.max_abs_state,
                    format.state_units(VH, HD, HD),
                    format.state_bytes(VH, HD, HD),
                );
                println!(
                    "oracle: format={} graph={} cases={} mismatches={} kernels={}",
                    label(format),
                    u8::from(cli.graph),
                    report.cases.len(),
                    report.mismatches(),
                    report.kernels.iter().cloned().collect::<Vec<_>>().join(","),
                );
                reports.push(report);
            }
            Err(error) => {
                failed = Some(format!("format {}: {error}", label(format)));
                break;
            }
        }
    }
    gpu.qwen4_scope = false;
    if let Some(error) = failed {
        return Err(error);
    }
    if let Some(limit) = capture_limit {
        println!("few_row_limit_in_capture={limit}");
    }
    let mismatches: usize = reports.iter().map(FormatReport::mismatches).sum();
    if let Some(path) = &cli.json {
        let formats: Vec<serde_json::Value> = reports
            .iter()
            .map(|r| {
                let dtype = format!("{:?}", r.format.dtype()).to_ascii_lowercase();
                let kernels: Vec<&String> = r.kernels.iter().collect();
                let results: Vec<serde_json::Value> = r
                    .cases
                    .iter()
                    .map(|c| {
                        let failures: Vec<serde_json::Value> =
                            c.failures.iter().map(Failure::to_json).collect();
                        json!({
                            "rows": c.rows,
                            "keep": c.keep,
                            "window": c.window,
                            "ok": c.failures.is_empty(),
                            "failures": failures,
                        })
                    })
                    .collect();
                let state_bytes = r.format.state_bytes(VH, HD, HD);
                json!({
                    "format": label(r.format),
                    "dtype": dtype,
                    "state_units": r.format.state_units(VH, HD, HD),
                    "state_bytes": state_bytes,
                    "ring_bytes": 2 * ring_rows * state_bytes,
                    "cases": r.cases.len(),
                    "mismatches": r.mismatches(),
                    "max_abs_state": r.max_abs_state,
                    "max_abs_state_source": "decoded AR reference snapshots",
                    "kernels": kernels,
                    "results": results,
                })
            })
            .collect();
        let doc = json!({
            "tool": "gdn_state_precision_oracle",
            "gpu_arch": arch,
            "geometry": {
                "key_heads": KH, "value_heads": VH, "key_dim": HD, "value_dim": HD,
                "qkv_width": QKV, "start_position": START_POSITION,
            },
            "seed": cli.seed,
            "windows": cli.windows,
            "graph": cli.graph,
            "max_rows": cli.max_rows,
            "ring_rows": ring_rows,
            "all_keeps": cli.all_keeps,
            "route_flags": {
                "qwen4_scope": true,
                "prefill_invariant_scope": false,
                "wide_verify_rows_batched": ring_rows,
                "wide_verify_rows_ar": 0,
                "wide_verify_max_rows": WIDE_VERIFY_MAX_ROWS,
                "few_row_limit_eager": eager_limit,
                "few_row_limit_in_capture": capture_limit,
            },
            "total_cases": reports.iter().map(|r| r.cases.len()).sum::<usize>(),
            "total_mismatches": mismatches,
            "formats": formats,
        });
        let text = ctx(serde_json::to_string_pretty(&doc), "serialize json")?;
        ctx(std::fs::write(path, text), &format!("write {}", path.display()))?;
    }
    Ok(i32::from(mismatches > 0))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("gdn_state_precision_oracle: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };
    match run(&cli) {
        Ok(code) => std::process::exit(code),
        Err(message) => {
            eprintln!("gdn_state_precision_oracle: unsupported or failed setup: {message}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn cli_defaults() {
        let cli = parse_args(&[]).unwrap();
        assert_eq!(cli.formats, vec![GdnStateFormat::Q8, GdnStateFormat::F32]);
        assert!(!cli.graph && !cli.all_keeps);
        assert_eq!(cli.max_rows, 32);
        assert_eq!(cli.windows, 3);
        assert_eq!(cli.seed, DEFAULT_SEED);
        assert_eq!(cli.json, None);
    }

    #[test]
    fn cli_parses_full_command_line() {
        let cli = parse_args(&args(
            "--formats fp32,q8 --graph 1 --max-rows=12 --all-keeps --seed 0x10 --windows 4 --json /tmp/o.json",
        ))
        .unwrap();
        assert_eq!(cli.formats, vec![GdnStateFormat::F32, GdnStateFormat::Q8]);
        assert!(cli.graph && cli.all_keeps);
        assert_eq!((cli.max_rows, cli.seed, cli.windows), (12, 16, 4));
        assert_eq!(cli.json, Some(PathBuf::from("/tmp/o.json")));
    }

    #[test]
    fn cli_rejects_q8_ef() {
        let error = parse_args(&args("--formats q8,q8_ef")).unwrap_err();
        assert!(error.contains("q8_ef") && error.contains("not available"), "{error}");
    }

    #[test]
    fn cli_rejects_invalid_values() {
        for bad in [
            "--windows 1",
            "--windows 0",
            "--max-rows 0",
            "--max-rows 33",
            "--graph 2",
            "--formats q8,q8",
            "--formats bf16",
            "--seed nope",
            "--max-rows",
            "--bogus",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn capture_base_alternates_halves() {
        let r = 8;
        let mut source = 0;
        let mut bases = Vec::new();
        for rows in [8, 3, 5, 1, 8] {
            let base = capture_base(source, r);
            bases.push(base);
            source = base + rows - 1;
        }
        assert_eq!(bases, vec![8, 0, 8, 0, 8]);
    }

    #[test]
    fn keeps_selection() {
        assert_eq!(keeps_for(3, true), vec![0, 1, 2, 3]);
        assert_eq!(keeps_for(3, false), vec![0, 3]);
        assert_eq!(keeps_for(1, false), vec![0, 1]);
    }

    #[test]
    fn compare_reports_first_differing_byte_and_row() {
        let a = vec![0u8; 16];
        let mut b = a.clone();
        assert!(compare("k", &a, &b, Some(8)).is_none());
        b[11] = 1;
        b[14] = 2;
        let f = compare("k", &a, &b, Some(8)).unwrap();
        assert_eq!((f.byte_offset, f.row), (11, Some(1)));
        assert_eq!(first_diff(&a, &a[..12]), Some(12));
        assert!(compare("k", &a, &a[..12], None).is_some());
    }

    #[test]
    fn nonfinite_and_state_stats() {
        let bytes = f32_bytes(&[1.0, -3.5, 2.0]);
        assert_eq!(state_stats(&bytes), (3.5, None));
        let bad = f32_bytes(&[1.0, f32::NAN, f32::INFINITY]);
        assert_eq!(first_nonfinite(&bad), Some(4));
        assert_eq!(state_stats(&bad).0, 1.0);
    }

    #[test]
    fn bf16_rounding_is_rne_and_inputs_stay_in_range() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3F80_8000)), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3F81_8000)), f32::from_bits(0x3F82_0000));
        let steps = make_steps(2, window_seed(1, 2, 0));
        assert_eq!((steps.proj.len(), steps.gate.len(), steps.beta.len()), (2 * QKV, 2 * VH, 2 * VH));
        assert!(steps.proj.iter().all(|v| bf16_round(*v) == *v && v.abs() <= 2.0));
        assert!(steps.gate.iter().all(|g| (-0.3..-0.02).contains(g)));
        assert!(steps.beta.iter().all(|b| (0.1..0.9).contains(b)));
    }
}
