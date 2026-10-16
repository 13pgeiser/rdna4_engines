// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exactness oracle of the ordinary BF16 x4 rows kernels and the BF16 HC k4
//! rows kernels for 1..=32 activation rows, including the S2 wide twins
//! `gemv_bf16_xf32_x4_rows_wide_r{16,32}` and `gemv_bf16_xf32_k4_rows_wide_r{16,32}`
//! (`kernels/src/qwen4_gemv_bf16_xf32.hip`).
//!
//! The reference is always the actual single-row decode object, one launch per
//! activation row: `Gpu::gemv_bf16_xf32_x4` (router / selector / shared gate /
//! shared up group) and `Gpu::gemv_bf16_xf32` (single projection) for x4,
//! `Gpu::gemv_bf16_xf32_k4` for HC k4 (gates M=4 and read-down M=320, K=10240).
//! Every output word is compared with `to_bits()` against `[GUARD sentinel]
//! [expected][GUARD sentinel]`; outputs are pre-filled with a NaN payload no
//! kernel produces so an unwritten word cannot pass.
//!
//! Arms (each prints its own summary line):
//!  * `routed`: the public launchers `gemv_bf16_xf32_x4_rows` /
//!    `gemv_bf16_xf32_k4_rows` at rows 1..=32 with the `Bf16` wide-twin switch
//!    on and the verify gate opened (`set_wide_verify_rows(32)`); the launched
//!    symbol is asserted per row count (9..=16 -> `_wide_r16`, 17..=32 ->
//!    `_wide_r32`; x4 with K % 8 != 0 keeps the exact <=8-row group loop, and
//!    the four-column HC gate may report the tiled fallback).
//!  * `direct16` / `direct32`: the wide kernels launched by name for every
//!    R in 1..=16 / 1..=32 (R below the capacity exercises the row masks),
//!    including the K % 8 != 0 scalar domain of x4 that the launcher never
//!    routes to the wide kernel; eager with X contents A, then X contents B
//!    rewritten at the same device address, then a HIP graph captured with the
//!    B contents and replayed after the X buffer is rewritten to A.
//!  * HC handoff: `hyper_write_norm` producing gate quarters into both
//!    alternating slots of a guarded scratch (`quarters_out`), the exact
//!    per-quarter dots against `Gpu::gemv_bf16_xf32` on the quarter slices,
//!    the `(q0 + q1) + (q2 + q3)` fold against the wide k4 gate output, and the
//!    consumer launch with `quarters_in` against the consumer fed the wide k4
//!    result as plain gates.
//!
//! Data: random BF16 weights with +-0, BF16 subnormals, wide exponents and
//! cancelling neighbours; activation rows of very different magnitude with
//! +-0, F32 subnormals, alternating signs, duplicated pairs and all-zero rows.
//! Negative controls (perturbed X, swapped weights, stale quarter slot) must
//! disagree, so a vacuous pass is impossible.
//!
//! `HIPFIRE_QWEN4_WIDE_TWIN_BF16=1 cargo run --release -p rdna-compute --features lab --example bf16_rows_oracle`
//! Exact gfx1151 or gfx1201 only (exit 2 elsewhere); exits nonzero on mismatch.

use hip_bridge::KernargBlob;
use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::kernels::QWEN4_GEMV_BF16_XF32_SRC;
use rdna_compute::tensor_ops::{hyper_write_norm, HyperNextGates, HyperWrite};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;

const GUARD: usize = 64;
/// A quiet NaN with a payload no arithmetic produces.
const SENTINEL: u32 = 0x7fc0_beef;
const MODULE: &str = "qwen4_gemv_bf16_xf32";
const MAX_ROWS: usize = 32;
const HC_K: usize = 10240;
const HC_PART: usize = 2560;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Round-to-nearest-even BF16 bits of a finite F32.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Random BF16 weights `[m * k]`. `exp_span > 0` mixes in +-0, subnormals,
/// exponents 2^-span..2^span and exact cancelling neighbours.
fn gen_weights(m: usize, k: usize, amp: f32, exp_span: i32, rng: &mut Rng) -> Vec<u16> {
    let mut w: Vec<u16> = Vec::with_capacity(m * k);
    for i in 0..m * k {
        let pick = rng.below(100);
        let sign: u16 = if rng.next() & 1 == 0 { 0 } else { 0x8000 };
        let wild = exp_span > 0;
        let bits = if wild && pick < 4 {
            sign
        } else if wild && pick < 7 {
            sign | (1 + rng.below(127) as u16)
        } else if wild && pick < 10 {
            let e = 127 + rng.below(2 * exp_span as u64 + 1) as i32 - exp_span;
            sign | ((e as u16) << 7) | rng.below(128) as u16
        } else if wild && pick < 13 && i > 0 {
            w[i - 1] ^ 0x8000
        } else {
            bf16_bits((rng.unit() * 2.0 - 1.0) * amp)
        };
        w.push(bits);
    }
    w
}

/// Random F32 activations `[rows * k]`; row magnitudes span 2^exp_lo..2^exp_hi.
fn gen_x(rows: usize, k: usize, exp_lo: i32, exp_hi: i32, rng: &mut Rng) -> Vec<f32> {
    let span = (exp_hi - exp_lo + 1) as u64;
    let mut x: Vec<f32> = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 2f32.powi(exp_lo + ((r as u64 * 5) % span) as i32);
        for i in 0..k {
            let v = if r % 11 == 5 {
                0.0
            } else if r % 11 == 8 {
                -0.0
            } else {
                let pick = rng.below(100);
                let sign = if rng.next() & 1 == 0 {
                    0u32
                } else {
                    0x8000_0000
                };
                if pick < 5 {
                    f32::from_bits(sign)
                } else if pick < 8 {
                    f32::from_bits(sign | (1 + rng.below(0x7f_ffff) as u32))
                } else if r % 7 == 3 && i % 2 == 1 {
                    x[r * k + i - 1]
                } else {
                    let mut v = (rng.unit() * 2.0 - 1.0) * scale;
                    if r % 5 == 0 && i % 2 == 1 {
                        v = -v;
                    }
                    v
                }
            };
            x.push(v);
        }
    }
    x
}

fn upload_bf16(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload bf16");
    t.dtype = DType::BF16;
    t.shape = vec![bits.len()];
    t
}

fn sentinel_bytes(words: usize) -> Vec<u8> {
    SENTINEL.to_le_bytes().repeat(words)
}

fn fill_sentinel(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .device_synchronize()
        .expect("sync before sentinel fill");
    gpu.hip
        .memcpy_htod(&t.buf, &sentinel_bytes(t.numel()))
        .expect("sentinel fill");
}

fn write_x(gpu: &Gpu, t: &GpuTensor, host: &[f32]) {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.hip.device_synchronize().expect("sync before X rewrite");
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("X rewrite");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A guarded output allocation `[GUARD][n][GUARD]`, all sentinel.
fn alloc_out(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let total = GUARD + n + GUARD;
    gpu.upload_f32(&vec![f32::from_bits(SENTINEL); total], &[total])
        .expect("alloc out")
}

/// The writable view of a guarded output (at least one word so a zero-M slot
/// still has a valid, never-written pointer).
fn out_view(out: &GpuTensor, n: usize) -> GpuTensor {
    out.sub_offset(GUARD, n.max(1))
}

fn guarded(expected: &[u32]) -> Vec<u32> {
    let mut v = vec![SENTINEL; GUARD];
    v.extend_from_slice(expected);
    v.extend(std::iter::repeat_n(SENTINEL, GUARD));
    v
}

fn mismatches(got: &[u32], want: &[u32]) -> usize {
    assert_eq!(got.len(), want.len(), "guarded length");
    got.iter().zip(want).filter(|(a, b)| a != b).count()
}

type Bits = Vec<u32>;

#[derive(Default)]
struct Stats {
    runs: usize,
    bad: usize,
}

#[derive(Default)]
struct Report {
    rows: BTreeMap<String, Stats>,
    symbols: BTreeMap<String, usize>,
    failures: usize,
}

impl Report {
    fn record(&mut self, label: &str, detail: &str, bad: usize) {
        if bad != 0 {
            eprintln!("MISMATCH {label} {detail}: {bad} words differ");
            self.failures += 1;
        }
        let e = self.rows.entry(label.to_string()).or_default();
        e.runs += 1;
        e.bad += bad;
    }
    fn symbol(&mut self, sym: &str) {
        *self.symbols.entry(sym.to_string()).or_default() += 1;
    }
    fn fail(&mut self, what: &str) {
        eprintln!("FAIL {what}");
        self.failures += 1;
    }
}

// ───────────────────────────── ordinary BF16 x4 ─────────────────────────────

struct Mat {
    m: usize,
    k: usize,
    w: GpuTensor,
}

fn make_mats(gpu: &mut Gpu, ms: &[usize; 4], k: usize, rng: &mut Rng) -> Vec<Mat> {
    let mut mats = Vec::new();
    for &m in ms {
        let bits = if m == 0 {
            vec![0u16; 8]
        } else {
            gen_weights(m, k, 0.25, 40, rng)
        };
        mats.push(Mat {
            m,
            k,
            w: upload_bf16(gpu, &bits),
        });
    }
    mats
}

fn parts_of<'a>(
    mats: &'a [Mat],
    views: &'a [GpuTensor],
) -> [(&'a GpuTensor, &'a GpuTensor, usize); 4] {
    [
        (&mats[0].w, &views[0], mats[0].m),
        (&mats[1].w, &views[1], mats[1].m),
        (&mats[2].w, &views[2], mats[2].m),
        (&mats[3].w, &views[3], mats[3].m),
    ]
}

/// `rows` singleton `gemv_bf16_xf32_x4` launches (the decode object of a
/// router/shared group); per matrix `[rows * m]` bits.
fn ref_x4(gpu: &mut Gpu, mats: &[Mat], x: &GpuTensor, rows: usize) -> Vec<Bits> {
    let k = mats[0].k;
    let outs: Vec<GpuTensor> = mats.iter().map(|m| alloc_out(gpu, rows * m.m)).collect();
    for r in 0..rows {
        let xv = x.sub_offset(r * k, k);
        let views: Vec<GpuTensor> = mats
            .iter()
            .zip(&outs)
            .map(|(m, y)| y.sub_offset(GUARD + r * m.m, m.m.max(1)))
            .collect();
        gpu.gemv_bf16_xf32_x4(parts_of(mats, &views), &xv, k)
            .expect("singleton gemv_bf16_xf32_x4");
    }
    let mut got = Vec::new();
    for (y, m) in outs.iter().zip(mats) {
        got.push(fetch(gpu, y)[GUARD..GUARD + rows * m.m].to_vec());
    }
    for y in outs {
        gpu.free_tensor(y).expect("free ref");
    }
    got
}

/// `rows` singleton `gemv_bf16_xf32` launches (the decode object of a single
/// projection) of one matrix.
fn ref_single(gpu: &mut Gpu, mat: &Mat, x: &GpuTensor, rows: usize) -> Bits {
    let y = alloc_out(gpu, rows * mat.m);
    for r in 0..rows {
        gpu.gemv_bf16_xf32(
            &mat.w,
            &x.sub_offset(r * mat.k, mat.k),
            &y.sub_offset(GUARD + r * mat.m, mat.m),
            mat.m,
            mat.k,
        )
        .expect("singleton gemv_bf16_xf32");
    }
    let bits = fetch(gpu, &y)[GUARD..GUARD + rows * mat.m].to_vec();
    gpu.free_tensor(y).expect("free ref");
    bits
}

#[derive(Clone, Copy)]
enum Arm {
    Routed,
    Direct(usize),
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Routed => "routed".to_string(),
            Arm::Direct(cap) => format!("direct{cap}"),
        }
    }
    fn max_rows(self) -> usize {
        match self {
            Arm::Routed => MAX_ROWS,
            Arm::Direct(cap) => cap,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn direct_x4(
    gpu: &mut Gpu,
    cap: usize,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let func = format!("gemv_bf16_xf32_x4_rows_wide_r{cap}");
    let mut b = KernargBlob::new();
    for p in &parts {
        b.push_ptr(p.0.buf.as_ptr());
    }
    b.push_ptr(x.buf.as_ptr());
    for p in &parts {
        b.push_ptr(p.1.buf.as_ptr());
    }
    for p in &parts {
        b.push_i32(p.2 as i32);
    }
    b.push_i32(k as i32);
    b.push_i32(rows as i32);
    b.pad_to(16);
    let blocks: usize = parts.iter().map(|p| p.2.div_ceil(8)).sum();
    keep.push(b);
    let blob = keep.last_mut().expect("blob").as_mut_slice();
    gpu.launch_kernel_blob(&func, [blocks as u32, 1, 1], [256, 1, 1], 0, blob)
        .map_err(|e| format!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_x4(
    gpu: &mut Gpu,
    arm: Arm,
    parts: [(&GpuTensor, &GpuTensor, usize); 4],
    x: &GpuTensor,
    k: usize,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    match arm {
        Arm::Routed => {
            let launched = if rows == 1 {
                gpu.gemv_bf16_xf32_x4(parts, x, k)
            } else {
                gpu.gemv_bf16_xf32_x4_rows(parts, x, k, rows)
            };
            launched.map_err(|e| format!("{e:?}"))
        }
        Arm::Direct(cap) => direct_x4(gpu, cap, parts, x, k, rows, keep),
    }
}

/// Capture `launch` into a HIP graph, optionally rewrite `swap` (a device
/// buffer and the host contents to write) at its stable address, refill the
/// sentinel and replay.
fn replay_graph(
    gpu: &mut Gpu,
    outs: &[GpuTensor],
    swap: Option<(&GpuTensor, &[f32])>,
    launch: impl FnOnce(&mut Gpu) -> Result<(), String>,
) {
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    let captured = launch(gpu);
    let g = gpu.end_stream_capture().expect("end capture");
    captured.expect("captured launch");
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    if let Some((xt, host)) = swap {
        write_x(gpu, xt, host);
    }
    for y in outs {
        fill_sentinel(gpu, y);
    }
    gpu.launch_graph(&exec).expect("graph replay");
    gpu.hip.device_synchronize().expect("sync");
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
}

fn x4_symbol_ok(sym: &str, rows: usize, k: usize) -> bool {
    match rows {
        1 => sym == "gemv_bf16_xf32_x4",
        2..=8 => sym == format!("gemv_bf16_xf32_x4_rows_r{rows}"),
        // gfx1151 prefers the admitted PeaceMaker twin (`*_pm_gfx1151`, an
        // exact-gfx1151 image, never loaded elsewhere); bytes are checked either way.
        _ if k % 8 == 0 => {
            let wide = if rows <= 16 {
                "gemv_bf16_xf32_x4_rows_wide_r16"
            } else {
                "gemv_bf16_xf32_x4_rows_wide_r32"
            };
            sym == wide || sym == format!("{wide}_pm_gfx1151")
        }
        // K % 8 != 0: the incumbent scalar domain keeps the exact group loop.
        _ => sym == "gemv_bf16_xf32_x4" || sym.starts_with("gemv_bf16_xf32_x4_rows_r"),
    }
}

fn check_x4(got: &[Bits], want: &[Bits], mats: &[Mat], rows: usize) -> usize {
    got.iter()
        .zip(want)
        .zip(mats)
        .map(|((g, w), m)| mismatches(g, &guarded(&w[..rows * m.m])))
        .sum()
}

/// One x4 launch of `mats` over `rows` rows into fresh sentinel outputs.
/// Returns the whole guarded outputs and (eager) the launched symbol.
#[allow(clippy::too_many_arguments)]
fn run_x4(
    gpu: &mut Gpu,
    arm: Arm,
    mats: &[Mat],
    x: &GpuTensor,
    rows: usize,
    graph: Option<&[f32]>,
    keep: &mut Vec<KernargBlob>,
) -> (Vec<Bits>, Option<String>) {
    let k = mats[0].k;
    let outs: Vec<GpuTensor> = mats.iter().map(|m| alloc_out(gpu, rows * m.m)).collect();
    let views: Vec<GpuTensor> = mats
        .iter()
        .zip(&outs)
        .map(|(m, y)| out_view(y, rows * m.m))
        .collect();
    let mut sym = None;
    match graph {
        None => {
            launch_x4(gpu, arm, parts_of(mats, &views), x, k, rows, keep).expect("x4 launch");
            sym = gpu.last_launched_kernel().map(str::to_owned);
        }
        Some(host) => {
            let parts = parts_of(mats, &views);
            replay_graph(gpu, &outs, Some((x, host)), |g| {
                launch_x4(g, arm, parts, x, k, rows, keep)
            });
        }
    }
    let got: Vec<Bits> = outs.iter().map(|y| fetch(gpu, y)).collect();
    for y in outs {
        gpu.free_tensor(y).expect("free out");
    }
    (got, sym)
}

const X4_CASES: [(&str, [usize; 4], usize); 17] = [
    (
        "router|selector|shared_gate|shared_up K=2560",
        [512, 1, 512, 512],
        2560,
    ),
    ("single M=1 K=2560", [1, 0, 0, 0], 2560),
    ("single M=7 K=2560", [7, 0, 0, 0], 2560),
    ("single M=8 K=2560", [8, 0, 0, 0], 2560),
    ("single M=9 K=2560", [9, 0, 0, 0], 2560),
    ("single M=17 K=2560", [17, 0, 0, 0], 2560),
    ("mixed zero-M slots K=2560", [0, 13, 0, 9], 2560),
    ("odd boundaries K=2560", [9, 15, 1, 17], 2560),
    ("one full tile K=256", [9, 8, 0, 17], 256),
    ("partial tile K=320", [9, 8, 0, 17], 320),
    ("one lane in tail tile K=520", [9, 8, 0, 17], 520),
    ("tiny K=72", [9, 8, 0, 17], 72),
    ("K=6144", [16, 1, 0, 33], 6144),
    ("scalar domain K=7", [9, 8, 0, 17], 7),
    ("scalar domain K=100", [9, 8, 0, 17], 100),
    ("scalar domain K=333", [9, 8, 0, 17], 333),
    ("scalar domain K=2561", [9, 8, 0, 17], 2561),
];

fn run_x4_cases(gpu: &mut Gpu, report: &mut Report) {
    let mut keep: Vec<KernargBlob> = Vec::new();
    for (ci, &(name, ms, k)) in X4_CASES.iter().enumerate() {
        let mut rng = Rng((ci as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407));
        let mats = make_mats(gpu, &ms, k, &mut rng);
        let xa = gen_x(MAX_ROWS, k, -8, 8, &mut rng);
        let xb = gen_x(MAX_ROWS, k, -6, 10, &mut rng);
        let x = gpu.upload_f32(&xa, &[xa.len()]).expect("upload x");

        // The decode objects of this group on both X contents; the grouped
        // singleton and the single-projection singleton must agree.
        let ref_a = ref_x4(gpu, &mats, &x, MAX_ROWS);
        let single_a: Vec<Bits> = mats
            .iter()
            .map(|m| {
                if m.m == 0 {
                    Vec::new()
                } else {
                    ref_single(gpu, m, &x, MAX_ROWS)
                }
            })
            .collect();
        for (i, m) in mats.iter().enumerate() {
            if m.m != 0 {
                let bad = mismatches(&ref_a[i], &single_a[i]);
                report.record(
                    &format!("decode objects agree (x4 singleton vs gemv_bf16_xf32) {name}"),
                    &format!("slot {i}"),
                    bad,
                );
            }
        }
        write_x(gpu, &x, &xb);
        let ref_b = ref_x4(gpu, &mats, &x, MAX_ROWS);
        write_x(gpu, &x, &xa);
        if ref_a
            .iter()
            .zip(&ref_b)
            .zip(&mats)
            .any(|((a, b), m)| m.m != 0 && a == b)
        {
            report.fail(&format!(
                "{name}: X contents A and B give identical references (vacuous)"
            ));
        }

        let arms = [Arm::Routed, Arm::Direct(16), Arm::Direct(32)];
        for arm in arms {
            let label = format!("x4 {} {name}", arm.name());
            for rows in 1..=arm.max_rows() {
                // Eager, X contents A.
                let (got, sym) = run_x4(gpu, arm, &mats, &x, rows, None, &mut keep);
                report.record(
                    &label,
                    &format!("rows={rows} eager A"),
                    check_x4(&got, &ref_a, &mats, rows),
                );
                if let (Arm::Routed, Some(sym)) = (arm, sym) {
                    report.symbol(&sym);
                    if !x4_symbol_ok(&sym, rows, k) {
                        report.fail(&format!("{name}: rows={rows} launched {sym}"));
                    }
                }
                // Eager, X contents rewritten at the same address.
                write_x(gpu, &x, &xb);
                let (got, _) = run_x4(gpu, arm, &mats, &x, rows, None, &mut keep);
                report.record(
                    &label,
                    &format!("rows={rows} eager B"),
                    check_x4(&got, &ref_b, &mats, rows),
                );
                // Graph captured with B, X rewritten to A before the replay.
                if let Arm::Direct(_) = arm {
                    let (got, _) = run_x4(gpu, arm, &mats, &x, rows, Some(&xa), &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} graph (B captured, A replayed)"),
                        check_x4(&got, &ref_a, &mats, rows),
                    );
                } else {
                    write_x(gpu, &x, &xa);
                }
                keep.clear();
            }
        }

        // Negative controls on the router group only.
        if ci == 0 {
            let rows = 20;
            let (got, _) = run_x4(gpu, Arm::Direct(32), &mats, &x, rows, None, &mut keep);
            let mut xp = xa.clone();
            for r in 0..MAX_ROWS {
                for j in (0..k).step_by(7) {
                    xp[r * k + j] += 3.0 + (j % 5) as f32;
                }
            }
            let xt = gpu.upload_f32(&xp, &[xp.len()]).expect("upload x'");
            let ref_p = ref_x4(gpu, &mats, &xt, rows);
            let bad: Vec<usize> = got
                .iter()
                .zip(&ref_p)
                .zip(&mats)
                .map(|((g, w), m)| mismatches(g, &guarded(&w[..rows * m.m])))
                .collect();
            println!(
                "bf16_rows_oracle: negative control perturbed-x mismatch words per output {bad:?}"
            );
            if bad.iter().any(|&n| n == 0) {
                report.fail("negative control: a perturbed-X reference agreed with the launch (vacuous oracle)");
            }
            let swapped = mismatches(&got[2], &guarded(&ref_a[3][..rows * mats[3].m]));
            println!("bf16_rows_oracle: negative control swapped shared gate/up mismatch words {swapped}");
            if swapped == 0 {
                report.fail(
                    "negative control: swapped weights agreed with the launch (vacuous oracle)",
                );
            }
            gpu.free_tensor(xt).expect("free x'");
            keep.clear();
        }

        for m in mats {
            gpu.free_tensor(m.w).expect("free weights");
        }
        gpu.free_tensor(x).expect("free x");
    }
}

// ────────────────────────────────── HC k4 ───────────────────────────────────

/// `(m, k)` of the gates / read-down and boundary shapes (all K % 32 == 0).
const K4_SHAPES: [(&str, usize, usize); 10] = [
    ("gates M=4 K=10240", 4, 10240),
    ("read-down M=320 K=10240", 320, 10240),
    ("M=1 K=10240", 1, 10240),
    ("M=2 K=10240", 2, 10240),
    ("M=3 K=10240", 3, 10240),
    ("M=5 K=2048", 5, 2048),
    ("M=7 K=96 (partial lanes)", 7, 96),
    ("M=9 K=1056 (one lane in tail tile)", 9, 1056),
    ("M=33 K=1280", 33, 1280),
    ("M=16 K=1024", 16, 1024),
];

#[allow(clippy::too_many_arguments)]
fn direct_k4(
    gpu: &mut Gpu,
    cap: usize,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    let func = format!("gemv_bf16_xf32_k4_rows_wide_r{cap}");
    let mut b = KernargBlob::new();
    b.push_ptr(w.buf.as_ptr());
    b.push_ptr(x.buf.as_ptr());
    b.push_ptr(y.buf.as_ptr());
    b.push_i32(m as i32);
    b.push_i32(k as i32);
    b.push_f32(act.unwrap_or(1.0));
    b.push_i32(i32::from(act.is_some()));
    b.push_i32(rows as i32);
    b.pad_to(16);
    keep.push(b);
    let blob = keep.last_mut().expect("blob").as_mut_slice();
    gpu.launch_kernel_blob(&func, [m.div_ceil(2) as u32, 1, 1], [256, 1, 1], 0, blob)
        .map_err(|e| format!("{e:?}"))
}

#[allow(clippy::too_many_arguments)]
fn launch_k4(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    keep: &mut Vec<KernargBlob>,
) -> Result<(), String> {
    match arm {
        Arm::Routed => gpu
            .gemv_bf16_xf32_k4_rows(w, x, y, m, k, act, rows)
            .map_err(|e| format!("{e:?}")),
        Arm::Direct(cap) => direct_k4(gpu, cap, w, x, y, m, k, act, rows, keep),
    }
}

fn k4_symbol_ok(sym: &str, rows: usize, m: usize) -> bool {
    match rows {
        1 => sym == "gemv_bf16_xf32_k4",
        2..=8 => sym == format!("gemv_bf16_xf32_k4_rows_r{rows}"),
        _ => {
            let wide = if rows <= 16 {
                "gemv_bf16_xf32_k4_rows_wide_r16"
            } else {
                "gemv_bf16_xf32_k4_rows_wide_r32"
            };
            // The four-column gate (two workgroups) may stay on exact row tiling;
            // gfx1151 prefers the admitted PeaceMaker twin (`*_pm_gfx1151`).
            sym == wide
                || sym == format!("{wide}_pm_gfx1151")
                || (m <= 4
                    && (sym == "gemv_bf16_xf32_k4"
                        || sym.starts_with("gemv_bf16_xf32_k4_rows_tiled_r")))
        }
    }
}

/// `rows` singleton `gemv_bf16_xf32_k4` launches; `[rows * m]` bits.
fn ref_k4(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
) -> Bits {
    let y = alloc_out(gpu, rows * m);
    for r in 0..rows {
        gpu.gemv_bf16_xf32_k4(
            w,
            &x.sub_offset(r * k, k),
            &y.sub_offset(GUARD + r * m, m),
            m,
            k,
            act,
        )
        .expect("singleton gemv_bf16_xf32_k4");
    }
    let bits = fetch(gpu, &y)[GUARD..GUARD + rows * m].to_vec();
    gpu.free_tensor(y).expect("free ref");
    bits
}

#[allow(clippy::too_many_arguments)]
fn run_k4(
    gpu: &mut Gpu,
    arm: Arm,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
    act: Option<f32>,
    rows: usize,
    graph: Option<&[f32]>,
    keep: &mut Vec<KernargBlob>,
) -> (Bits, Option<String>) {
    let out = alloc_out(gpu, rows * m);
    let view = out_view(&out, rows * m);
    let mut sym = None;
    match graph {
        None => {
            launch_k4(gpu, arm, w, x, &view, m, k, act, rows, keep).expect("k4 launch");
            sym = gpu.last_launched_kernel().map(str::to_owned);
        }
        Some(host) => {
            replay_graph(gpu, std::slice::from_ref(&out), Some((x, host)), |g| {
                launch_k4(g, arm, w, x, &view, m, k, act, rows, keep)
            });
        }
    }
    let bits = fetch(gpu, &out);
    gpu.free_tensor(out).expect("free out");
    (bits, sym)
}

fn run_k4_cases(gpu: &mut Gpu, report: &mut Report) {
    let mut keep: Vec<KernargBlob> = Vec::new();
    for (ci, &(name, m, k)) in K4_SHAPES.iter().enumerate() {
        let mut rng = Rng((ci as u64 + 101).wrapping_mul(0xD1B5_4A32_D192_ED03));
        // exp_span 5 keeps products in the HC activation's interesting range.
        let wbits = gen_weights(m, k, 0.05, 5, &mut rng);
        let w = upload_bf16(gpu, &wbits);
        let xa = gen_x(MAX_ROWS, k, -6, 6, &mut rng);
        let xb = gen_x(MAX_ROWS, k, -5, 7, &mut rng);
        let x = gpu.upload_f32(&xa, &[xa.len()]).expect("upload x");
        for act in [None, Some(0.25f32)] {
            let ref_a = ref_k4(gpu, &w, &x, m, k, act, MAX_ROWS);
            write_x(gpu, &x, &xb);
            let ref_b = ref_k4(gpu, &w, &x, m, k, act, MAX_ROWS);
            write_x(gpu, &x, &xa);
            if ref_a == ref_b {
                report.fail(&format!(
                    "{name} act={act:?}: X contents A and B give identical references (vacuous)"
                ));
            }
            for arm in [Arm::Routed, Arm::Direct(16), Arm::Direct(32)] {
                let label = format!("k4 {} {name} act={act:?}", arm.name());
                for rows in 1..=arm.max_rows() {
                    let (got, sym) = run_k4(gpu, arm, &w, &x, m, k, act, rows, None, &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} eager A"),
                        mismatches(&got, &guarded(&ref_a[..rows * m])),
                    );
                    if let (Arm::Routed, Some(sym)) = (arm, sym) {
                        report.symbol(&sym);
                        if !k4_symbol_ok(&sym, rows, m) {
                            report.fail(&format!("{name}: rows={rows} launched {sym}"));
                        }
                    }
                    write_x(gpu, &x, &xb);
                    let (got, _) = run_k4(gpu, arm, &w, &x, m, k, act, rows, None, &mut keep);
                    report.record(
                        &label,
                        &format!("rows={rows} eager B"),
                        mismatches(&got, &guarded(&ref_b[..rows * m])),
                    );
                    if let Arm::Direct(_) = arm {
                        let (got, _) =
                            run_k4(gpu, arm, &w, &x, m, k, act, rows, Some(&xa), &mut keep);
                        report.record(
                            &label,
                            &format!("rows={rows} graph (B captured, A replayed)"),
                            mismatches(&got, &guarded(&ref_a[..rows * m])),
                        );
                    } else {
                        write_x(gpu, &x, &xa);
                    }
                    keep.clear();
                }
            }
        }
        // The activated and raw outputs must differ (the epilogue ran).
        let raw = ref_k4(gpu, &w, &x, m, k, None, 4);
        let act = ref_k4(gpu, &w, &x, m, k, Some(0.25), 4);
        if raw == act {
            report.fail(&format!(
                "{name}: hc_act epilogue is a no-op on this data (vacuous)"
            ));
        }
        gpu.free_tensor(x).expect("free x");
        gpu.free_tensor(w).expect("free weights");
    }
}

// ───────────────────── HC quarters_out -> quarters_in handoff ─────────────────────

fn host_bf16_vec(n: usize, lo: f32, hi: f32, rng: &mut Rng) -> Vec<u16> {
    (0..n)
        .map(|_| bf16_bits(lo + (hi - lo) * rng.unit()))
        .collect()
}

/// One `hyper_write_norm` launch (no `next`, no `clear`): gates from `gates`,
/// or from `q_in` (the `quarters_in` of a previous producer) when present.
/// Returns the whole guarded `output` and `normalized` bits.
fn hc_consume(
    gpu: &mut Gpu,
    streams: &GpuTensor,
    mix: &GpuTensor,
    wn: &GpuTensor,
    gates: &GpuTensor,
    q_in: Option<&GpuTensor>,
    rows: usize,
) -> (Bits, Bits) {
    let o = alloc_out(gpu, rows * HC_K);
    let ov = out_view(&o, rows * HC_K);
    let n = alloc_out(gpu, rows * HC_K);
    let nv = out_view(&n, rows * HC_K);
    hyper_write_norm(
        gpu,
        &HyperWrite {
            input: streams,
            normalized: &nv,
            mixed: mix,
            gates,
            output: &ov,
            branches: 4,
            hidden: HC_PART,
            state_bf16: false,
        },
        wn,
        &nv,
        q_in,
        None,
        None,
    )
    .expect("hyper_write_norm consumer");
    let bits = (fetch(gpu, &o), fetch(gpu, &n));
    gpu.free_tensor(o).expect("free o");
    gpu.free_tensor(n).expect("free n");
    bits
}

fn run_hc_handoff(gpu: &mut Gpu, report: &mut Report) {
    let mut rng = Rng(0x4843_BF16);
    let hidden = HC_PART;
    let wn = upload_bf16(gpu, &host_bf16_vec(HC_K, -0.5, 0.5, &mut rng));
    let inject_bits = gen_weights(4, HC_K, 0.05, 0, &mut rng);
    let inject = upload_bf16(gpu, &inject_bits);
    // The four quarter slices of the gate rows: quarter b holds inject[g][b * 2560 ..].
    let mut qw: Vec<GpuTensor> = Vec::new();
    for b in 0..4 {
        let mut bits = Vec::with_capacity(4 * HC_PART);
        for g in 0..4 {
            bits.extend_from_slice(
                &inject_bits[g * HC_K + b * HC_PART..g * HC_K + (b + 1) * HC_PART],
            );
        }
        qw.push(upload_bf16(gpu, &bits));
    }
    let mut keep: Vec<KernargBlob> = Vec::new();
    for rows in 1..=MAX_ROWS {
        let cap = if rows <= 16 { 16 } else { 32 };
        for slot in 0..2usize {
            let label = format!("hc handoff slot {slot}");
            let streams = gpu
                .upload_f32(&gen_x(rows, HC_K, -1, 1, &mut rng), &[rows * HC_K])
                .expect("streams");
            let mix = gpu
                .upload_f32(&gen_x(rows, hidden, -1, 0, &mut rng), &[rows * hidden])
                .expect("mix");
            let logits = gen_x(rows, 4, 0, 2, &mut rng);
            // Scratch [GUARD][gates R*4][slot A 16R][slot B 16R][GUARD], sentinel except the gates.
            let cap_words = rows * 4 + 2 * rows * 16;
            let mut host = vec![f32::from_bits(SENTINEL); GUARD + cap_words + GUARD];
            host[GUARD..GUARD + rows * 4].copy_from_slice(&logits);
            let scratch = gpu.upload_f32(&host, &[host.len()]).expect("scratch");
            let gates = scratch.sub_offset(GUARD, rows * 4);
            let q_slot = scratch.sub_offset(GUARD + rows * 4 + slot * 16 * rows, 16 * rows);
            let q_other = scratch.sub_offset(GUARD + rows * 4 + (1 - slot) * 16 * rows, 16 * rows);

            // Producer: write the streams and the quarters of the NEXT write's gate rows.
            let out1 = alloc_out(gpu, rows * HC_K);
            let out1v = out_view(&out1, rows * HC_K);
            let norm1 = alloc_out(gpu, rows * HC_K);
            let norm1v = out_view(&norm1, rows * HC_K);
            let next = HyperNextGates {
                norm_weight: &wn,
                inject: &inject,
                quarters: &q_slot,
            };
            hyper_write_norm(
                gpu,
                &HyperWrite {
                    input: &streams,
                    normalized: &norm1v,
                    mixed: &mix,
                    gates: &gates,
                    output: &out1v,
                    branches: 4,
                    hidden,
                    state_bf16: false,
                },
                &wn,
                &norm1v,
                None,
                Some(&next),
                None,
            )
            .expect("hyper_write_norm producer");
            let scratch_bits = fetch(gpu, &scratch);
            let q_start = GUARD + rows * 4 + slot * 16 * rows;
            let q_dev: Vec<u32> = scratch_bits[q_start..q_start + 16 * rows].to_vec();
            // Canaries: guards, the other slot and the live gates are untouched; the slot is fully written.
            let mut canary_bad = 0usize;
            for (i, &w) in scratch_bits.iter().enumerate() {
                let inside_slot = (q_start..q_start + 16 * rows).contains(&i);
                let in_gates = (GUARD..GUARD + rows * 4).contains(&i);
                if in_gates {
                    canary_bad += usize::from(w != host[i].to_bits());
                } else if inside_slot {
                    canary_bad += usize::from(w == SENTINEL);
                } else {
                    canary_bad += usize::from(w != SENTINEL);
                }
            }
            report.record(&label, &format!("rows={rows} scratch canaries"), canary_bad);

            // The normalized rows are the k4 gate GEMV's input (next norm weight == norm weight).
            // Exact quarter dots against gemv_bf16_xf32 on the quarter slices.
            let mut quarter_bad = 0usize;
            for b in 0..4 {
                let yq = alloc_out(gpu, 4 * rows);
                for r in 0..rows {
                    gpu.gemv_bf16_xf32(
                        &qw[b],
                        &norm1v.sub_offset(r * HC_K + b * HC_PART, HC_PART),
                        &yq.sub_offset(GUARD + r * 4, 4),
                        4,
                        HC_PART,
                    )
                    .expect("quarter gemv");
                }
                let yq_bits = fetch(gpu, &yq);
                for r in 0..rows {
                    for g in 0..4 {
                        quarter_bad +=
                            usize::from(q_dev[r * 16 + g * 4 + b] != yq_bits[GUARD + r * 4 + g]);
                    }
                }
                gpu.free_tensor(yq).expect("free yq");
            }
            report.record(
                &label,
                &format!("rows={rows} quarters_out vs per-quarter gemv"),
                quarter_bad,
            );

            // Folded gate: producer quarters, singleton k4, wide k4 (hc_act off).
            let fold: Vec<u32> = (0..rows * 4)
                .map(|i| {
                    let (r, g) = (i / 4, i % 4);
                    let q = |b: usize| f32::from_bits(q_dev[r * 16 + g * 4 + b]);
                    ((q(0) + q(1)) + (q(2) + q(3))).to_bits()
                })
                .collect();
            let single = ref_k4(gpu, &inject, &norm1v, 4, HC_K, None, rows);
            let wide_out = alloc_out(gpu, rows * 4);
            let wide_view = out_view(&wide_out, rows * 4);
            direct_k4(
                gpu, cap, &inject, &norm1v, &wide_view, 4, HC_K, None, rows, &mut keep,
            )
            .expect("wide k4 gates");
            let wide_bits = fetch(gpu, &wide_out);
            report.record(
                &label,
                &format!("rows={rows} fold(quarters_out) vs singleton k4"),
                mismatches(&fold, &single),
            );
            report.record(
                &label,
                &format!("rows={rows} wide k4 gates vs fold(quarters_out)"),
                mismatches(&wide_bits, &guarded(&fold)),
            );

            // Consumer: gates from quarters_in vs the same launch fed the wide k4 result as gates.
            let poison = alloc_out(gpu, rows * 4);
            let poison_view = out_view(&poison, rows * 4);
            let gates_wide = gpu
                .upload_f32(
                    &wide_bits[GUARD..GUARD + rows * 4]
                        .iter()
                        .map(|&b| f32::from_bits(b))
                        .collect::<Vec<f32>>(),
                    &[rows * 4],
                )
                .expect("gates_wide");
            let (o_q, n_q) =
                hc_consume(gpu, &streams, &mix, &wn, &poison_view, Some(&q_slot), rows);
            let (o_g, n_g) = hc_consume(gpu, &streams, &mix, &wn, &gates_wide, None, rows);
            report.record(
                &label,
                &format!("rows={rows} consumer streams quarters_in vs wide gates"),
                mismatches(&o_q, &o_g),
            );
            report.record(
                &label,
                &format!("rows={rows} consumer normalized quarters_in vs wide gates"),
                mismatches(&n_q, &n_g),
            );
            let poison_after = fetch(gpu, &poison);
            report.record(
                &label,
                &format!("rows={rows} poisoned gates untouched"),
                mismatches(&poison_after, &vec![SENTINEL; poison_after.len()]),
            );
            if rows == 9 {
                // Negative control: the other (stale sentinel) slot as quarters_in must not agree.
                let (o_s, _) =
                    hc_consume(gpu, &streams, &mix, &wn, &poison_view, Some(&q_other), rows);
                if mismatches(&o_s, &o_g) == 0 {
                    report.fail("negative control: stale quarters slot agreed with the correct handoff (vacuous)");
                }
            }

            for t in [
                streams, mix, scratch, out1, norm1, wide_out, poison, gates_wide,
            ] {
                gpu.free_tensor(t).expect("free handoff");
            }
            keep.clear();
        }
    }
    for t in qw {
        gpu.free_tensor(t).expect("free qw");
    }
    for t in [wn, inject] {
        gpu.free_tensor(t).expect("free");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    println!("bf16_rows_oracle: arch={arch}");
    if !matches!(arch.as_str(), "gfx1151" | "gfx1201") {
        eprintln!("bf16_rows_oracle: needs exact gfx1151 or gfx1201");
        std::process::exit(2);
    }
    if !WideTwinFamily::Bf16.enabled() {
        eprintln!(
            "bf16_rows_oracle: the routed arm needs {}=1 (wide twin switch is default off)",
            WideTwinFamily::Bf16.env_var()
        );
        std::process::exit(2);
    }
    // The verify-scope gate: wide twins serve 9..=32 rows only inside it (eager launches).
    gpu.set_wide_verify_rows(MAX_ROWS);
    // Load every wide kernel before any graph capture (no compile mid-capture).
    for cap in [16, 32] {
        for func in [
            format!("gemv_bf16_xf32_x4_rows_wide_r{cap}"),
            format!("gemv_bf16_xf32_k4_rows_wide_r{cap}"),
        ] {
            gpu.ensure_kernel_public(MODULE, QWEN4_GEMV_BF16_XF32_SRC, &func)
                .unwrap_or_else(|e| panic!("load {func}: {e:?}"));
        }
    }
    let mut report = Report::default();
    run_x4_cases(&mut gpu, &mut report);
    run_k4_cases(&mut gpu, &mut report);
    run_hc_handoff(&mut gpu, &mut report);

    println!("bf16_rows_oracle: per-case summary (runs, mismatching words)");
    for (label, s) in &report.rows {
        println!("  {label:<84} runs={:<5} mismatch_words={}", s.runs, s.bad);
    }
    println!("bf16_rows_oracle: routed launches by symbol");
    for (sym, n) in &report.symbols {
        println!("  {sym:<44} {n}");
    }
    let runs: usize = report.rows.values().map(|s| s.runs).sum();
    let bad: usize = report.rows.values().map(|s| s.bad).sum();
    if report.failures != 0 || bad != 0 {
        println!(
            "bf16_rows_oracle: FAIL runs={runs} failures={} mismatch_bytes={}",
            report.failures,
            bad * 4
        );
        std::process::exit(1);
    }
    println!("bf16_rows_oracle: PASS runs={runs} mismatch_bytes=0");
}
