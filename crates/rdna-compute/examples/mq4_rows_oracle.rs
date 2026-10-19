// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Column-identity oracle of `Gpu::gemv_mq4g256v2_rows_x4` (MTP / exact-wide
//! verify, 1..=32 activation rows of MQ4G256V2 projections).
//!
//! Every output element of the few-row launch must be bitwise the AR decode
//! GEMV `Gpu::gemv_mq4g256v2` run on that activation row alone (R independent
//! single-row launches of the actual decode-selected object). Weights are
//! random valid MQ4G256V2 (finite normal f16 scale/zero per half group, random
//! nibbles). Activation rows differ wildly: ordinary rows, rows scaled by
//! 2^-60..2^60, alternating-sign cancellation rows, +0/-0/zero rows and
//! subnormal rows. For seeds 1..=3 and rows 1..=32 (every count):
//!
//!  (a) the Qwen3.8-Flash-Next trunk shapes (M, K), each alone in slot 0: GDN
//!      qkv (10240, 2560), z (6144, 2560), a (48, 2560), b (48, 2560), out
//!      (2560, 6144); QSA q (12288, 2560), k (512, 2560), v (512, 2560), indexer
//!      (640, 2560), o (2560, 6144);
//!  (b) the real x4 groupings in one launch: GDN qkv|z|a|b and QSA
//!      q|k|v|indexer (K = 2560), out|o (K = 6144);
//!  (c) synthetic single matrices with M in {1,2,3,7,8,9,15,17,33} (odd and
//!      block-straddling M exposes matrix-local parity mistakes) at K in
//!      {256,512,768,1024,1280,2560,6144} (every four-stream tail length);
//!  (d) synthetic x4 groupings with odd matrix boundaries and zero-M parts in
//!      any slot (K = 256, 1280, 2560);
//!  (e) the same launches with the activation rows permuted (reversed) and
//!      taken from a sub-view at row offset 3.
//!
//! Even and odd output columns (matrix-local) are tallied separately, and the
//! NaN guards on both sides of every y plus the zero-M slots' y are checked.
//! Each eager launch asserts the actual launched symbol
//! (`Gpu::last_launched_kernel`): the narrow `gemv_mq4g256v2_rows_x4_pm` for
//! 1..=8 rows, `..._wide_r16_pm` for 9..=16, `..._wide_r32_pm` for 17..=32
//! when the MQ4 wide twins are on (default here: this oracle sets
//! `HIPFIRE_QWEN4_WIDE_TWIN_MQ4=1` when it is unset; `=0` exercises the exact
//! 8-row group loops instead). Every case runs eagerly and (rows <= 8 and a
//! sample of wider counts, which capture the 8-row group loops: wide verify is
//! eager-only) from an instantiated HIP graph into freshly NaN-filled buffers;
//! a graph replay is also repeated after the activation is overwritten at the
//! same address. Comparison is `to_bits()`. Negative controls (perturbed
//! activation, swapped matrices, even/odd column swap) must disagree, so a
//! vacuous pass is impossible.
//!
//! `HIPFIRE_QWEN4_WIDE_TWIN_MQ4=1 cargo run --release -p rdna-compute --features lab --example mq4_rows_oracle`
//! Exits nonzero on any mismatch.

use rdna_compute::gemv::WideTwinFamily;
use rdna_compute::{Gpu, GpuTensor};
use std::collections::BTreeMap;

const GUARD: usize = 64;
const MAX_ROWS: usize = 32;
const NAN_BITS: u32 = 0x7fc0_0000;
/// Row counts that also replay from a HIP graph (all of 1..=8 plus samples
/// around the wide boundaries).
const GRAPH_ROWS: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32];

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
}

/// Round-to-nearest-even f16 bits of a finite value in the normal f16 range.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let mut m10 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && m10 & 1 == 1) {
        m10 += 1;
    }
    if m10 == 0x400 {
        m10 = 0;
        exp += 1;
    }
    assert!((1..=30).contains(&exp), "{v} is not a normal f16");
    sign | ((exp as u16) << 10) | m10 as u16
}

fn random_scale(rng: &mut Rng) -> u16 {
    let mag = 2f32.powi(-9) + rng.unit() * 0.2;
    f16_bits(if rng.next() & 1 == 0 { mag } else { -mag })
}

fn random_zero(rng: &mut Rng) -> u16 {
    let mut v = (rng.unit() - 0.5) * 4.0;
    if v.abs() < 2f32.powi(-12) {
        v = 0.125;
    }
    f16_bits(v)
}

/// Host packer: `[m][k / 256]` groups of 136 bytes: hA (scale s0, zero z0),
/// hB (s1, z1), then 128 bytes of nibbles (element e at byte e / 2, low
/// nibble = even e).
fn pack_random_mq4g256v2(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 256) * 136);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            out.extend_from_slice(&random_scale(rng).to_le_bytes());
            out.extend_from_slice(&random_zero(rng).to_le_bytes());
        }
        for _ in 0..128 {
            out.push((rng.next() & 0xff) as u8);
        }
    }
    out
}

struct Mat {
    name: String,
    m: usize,
    k: usize,
    w: GpuTensor,
    /// Singleton reference, `[MAX_ROWS][m]` result bits; row r is
    /// `gemv_mq4g256v2` on activation row r. Rows `..R` are the reference of R rows.
    reference: Vec<u32>,
}

const SHAPES: [(&str, usize, usize); 10] = [
    ("gdn.qkv", 10240, 2560),
    ("gdn.z", 6144, 2560),
    ("gdn.a", 48, 2560),
    ("gdn.b", 48, 2560),
    ("gdn.out", 2560, 6144),
    ("qsa.q", 12288, 2560),
    ("qsa.k", 512, 2560),
    ("qsa.v", 512, 2560),
    ("qsa.indexer", 640, 2560),
    ("qsa.o", 2560, 6144),
];

/// The real groupings, by index into `SHAPES`.
const GROUPS: [(&str, &[usize]); 3] = [
    ("group gdn qkv|z|a|b K=2560", &[0, 1, 2, 3]),
    ("group qsa q|k|v|indexer K=2560", &[5, 6, 7, 8]),
    ("group out|o K=6144", &[4, 9]),
];

/// Synthetic single-matrix grid.
const SYN_M: [usize; 9] = [1, 2, 3, 7, 8, 9, 15, 17, 33];
const SYN_K: [usize; 7] = [256, 512, 768, 1024, 1280, 2560, 6144];
/// Synthetic x4 groupings: M per slot (0 = zero-M part, any slot), at these K.
const SYN_GROUPS: [[usize; 4]; 8] = [
    [9, 0, 7, 1],
    [1, 1, 1, 1],
    [3, 0, 0, 13],
    [8, 9, 8, 7],
    [0, 5, 0, 2],
    [17, 16, 15, 33],
    [0, 0, 0, 9],
    [7, 3, 5, 11],
];
const SYN_GROUP_K: [usize; 3] = [256, 1280, 2560];

type Slots<'a> = [Option<&'a Mat>; 4];

fn nan_bytes(words: usize) -> Vec<u8> {
    NAN_BITS.to_le_bytes().repeat(words)
}

fn fill_nan(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip.device_synchronize().expect("sync before NaN fill");
    gpu.hip
        .memcpy_htod(&t.buf, &nan_bytes(t.numel()))
        .expect("NaN fill");
}

fn fetch(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before download");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Activation rows of every character: ordinary, wildly scaled, alternating
/// sign (cancellation), signed zeros, and subnormals. Finite for every weight
/// the packer draws (|w| <= ~5, K <= 6144).
fn host_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAX_ROWS * k);
    for r in 0..MAX_ROWS {
        let e = ((r * 7 + 3) % 13) as i32 * 10 - 60;
        for j in 0..k {
            let base = (rng.unit() + rng.unit() - 1.0) * 1.5;
            let v = match r % 8 {
                0 => base,
                3 => {
                    // Alternating signs of nearly equal magnitude: heavy cancellation.
                    let m = 0.75 + 0.015625 * (rng.unit() - 0.5);
                    if j % 2 == 0 {
                        m
                    } else {
                        -m
                    }
                }
                5 => match j % 3 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => base * 2f32.powi(-100),
                },
                6 => base * 2f32.powi(-135),
                _ => base * 2f32.powi(e),
            };
            x.push(v);
        }
    }
    x
}

fn upload_x(gpu: &mut Gpu, k: usize, rng: &mut Rng) -> GpuTensor {
    let x = host_x(k, rng);
    gpu.upload_f32(&x, &[x.len()]).expect("upload x")
}

/// The AR singleton on every activation row of `x` into its own buffer.
fn singleton_reference(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    m: usize,
    k: usize,
) -> Vec<u32> {
    let y = gpu
        .upload_f32(
            &vec![f32::from_bits(NAN_BITS); MAX_ROWS * m],
            &[MAX_ROWS * m],
        )
        .expect("alloc ref");
    for r in 0..MAX_ROWS {
        gpu.gemv_mq4g256v2(w, &x.sub_offset(r * k, k), &y.sub_offset(r * m, m), m, k)
            .expect("singleton gemv_mq4g256v2");
    }
    let bits = fetch(gpu, &y);
    gpu.free_tensor(y).expect("free ref");
    assert!(
        bits.iter().all(|&b| f32::from_bits(b).is_finite()),
        "singleton reference is not finite: the random weights/activations are invalid"
    );
    bits
}

fn make_mat(gpu: &mut Gpu, name: String, m: usize, k: usize, x: &GpuTensor, rng: &mut Rng) -> Mat {
    let bytes = pack_random_mq4g256v2(m, k, rng);
    let w = gpu
        .upload_raw(&bytes, &[bytes.len()])
        .expect("upload weights");
    let reference = singleton_reference(gpu, &w, x, m, k);
    Mat {
        name,
        m,
        k,
        w,
        reference,
    }
}

/// `[GUARD NaN][expected][GUARD NaN]` split into (even-column, odd-column,
/// guard) mismatching word counts. Column = index % m (matrix-local).
fn tally(got: &[u32], want: &[u32], m: usize) -> (usize, usize, usize) {
    assert_eq!(got.len(), want.len() + 2 * GUARD);
    let guard = got[..GUARD]
        .iter()
        .chain(&got[got.len() - GUARD..])
        .filter(|&&b| b != NAN_BITS)
        .count();
    let (mut even, mut odd) = (0, 0);
    for (i, (&g, &w)) in got[GUARD..got.len() - GUARD].iter().zip(want).enumerate() {
        if g != w {
            if (i % m) % 2 == 0 {
                even += 1;
            } else {
                odd += 1;
            }
        }
    }
    (even, odd, guard)
}

struct Launched {
    /// Full guarded y bits per present slot, in slot order.
    bits: Vec<Vec<u32>>,
    /// `Gpu::last_launched_kernel` after an eager launch.
    symbol: Option<String>,
}

fn present<'a>(slots: &Slots<'a>) -> Vec<&'a Mat> {
    slots.iter().flatten().copied().collect()
}

/// One few-row launch of `slots` (zero-M parts for `None`, in any slot) over
/// `rows` activation rows of `x` into NaN-prefilled guarded outputs.
/// `verify_rows`: the admitted verify width (`Gpu::wide_verify_rows`) of an
/// eager launch. `graph`: capture the launch into a HIP graph, refill NaN, and
/// replay it (capture keeps the few-row limit at 8).
fn launch(gpu: &mut Gpu, slots: &Slots, x: &GpuTensor, rows: usize, graph: bool) -> Launched {
    let mats = present(slots);
    let k = mats[0].k;
    let outs: Vec<GpuTensor> = mats
        .iter()
        .map(|mat| {
            let n = GUARD + rows * mat.m + GUARD;
            gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n])
                .expect("alloc y")
        })
        .collect();
    let views: Vec<GpuTensor> = mats
        .iter()
        .zip(&outs)
        .map(|(mat, y)| y.sub_offset(GUARD, rows * mat.m))
        .collect();
    let mut next = 0usize;
    let parts: Vec<(&GpuTensor, &GpuTensor, usize)> = slots
        .iter()
        .map(|slot| match slot {
            Some(mat) => {
                let p = (&mat.w, &views[next], mat.m);
                next += 1;
                p
            }
            None => (&mats[0].w, &views[0], 0),
        })
        .collect();
    let parts: [(&GpuTensor, &GpuTensor, usize); 4] = parts.try_into().ok().expect("four parts");
    let mut symbol = None;
    if graph {
        gpu.wide_verify_rows = 0;
        gpu.ensure_capture_stream().expect("capture stream");
        gpu.hip.device_synchronize().expect("sync");
        gpu.begin_stream_capture().expect("begin capture");
        let captured = gpu.gemv_mq4g256v2_rows_x4(parts, x, k, rows);
        let g = gpu.end_stream_capture().expect("end capture");
        captured.expect("captured launch");
        let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
        for y in &outs {
            fill_nan(gpu, y);
        }
        gpu.launch_graph(&exec).expect("graph replay");
        gpu.hip.device_synchronize().expect("sync");
        gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
        gpu.hip.graph_destroy(g).expect("destroy graph");
    } else {
        gpu.wide_verify_rows = rows;
        let result = gpu.gemv_mq4g256v2_rows_x4(parts, x, k, rows);
        gpu.wide_verify_rows = 0;
        result.expect("rows x4 launch");
        symbol = gpu.last_launched_kernel().map(str::to_owned);
    }
    let bits = outs.iter().map(|y| fetch(gpu, y)).collect();
    for y in outs {
        gpu.free_tensor(y).expect("free y");
    }
    Launched { bits, symbol }
}

/// Expected y rows (`ids[i]` = the reference row serving launch row i) of
/// every present slot.
fn expected_rows(slots: &Slots, ids: &[usize]) -> Vec<Vec<u32>> {
    present(slots)
        .iter()
        .map(|m| {
            ids.iter()
                .flat_map(|&r| m.reference[r * m.m..(r + 1) * m.m].iter().copied())
                .collect()
        })
        .collect()
}

/// The symbol the launcher must run for `rows` on this arch.
fn expected_symbol(gpu: &mut Gpu, arch: &str, rows: usize, pm: bool) -> Option<String> {
    gpu.wide_verify_rows = rows;
    let cap = gpu.wide_twin_cap(WideTwinFamily::Mq4, rows);
    gpu.wide_verify_rows = 0;
    match cap {
        Some(cap) => Some(format!("gemv_mq4g256v2_rows_x4_wide_r{cap}_pm_{arch}")),
        None if pm => Some(format!("gemv_mq4g256v2_rows_x4_pm_{arch}")),
        None => None,
    }
}

#[derive(Default)]
struct Tally {
    runs: usize,
    even: usize,
    odd: usize,
    guard: usize,
    symbol: usize,
}

struct Ctx {
    arch: String,
    pm: bool,
    cases: usize,
    tallies: BTreeMap<String, Tally>,
}

impl Ctx {
    fn bad_total(&self) -> usize {
        self.tallies
            .values()
            .map(|t| t.even + t.odd + t.guard + t.symbol)
            .sum()
    }

    /// Compare one launch against the reference rows `ids`.
    fn check(
        &mut self,
        gpu: &mut Gpu,
        label: &str,
        tag: &str,
        slots: &Slots,
        x: &GpuTensor,
        ids: &[usize],
        graph: bool,
    ) {
        let rows = ids.len();
        let launched = launch(gpu, slots, x, rows, graph);
        let want = expected_rows(slots, ids);
        let (mut even, mut odd, mut guard, mut symbol) = (0, 0, 0, 0);
        for ((got, want), mat) in launched.bits.iter().zip(&want).zip(present(slots)) {
            let (e, o, g) = tally(got, want, mat.m);
            even += e;
            odd += o;
            guard += g;
        }
        if !graph {
            if let Some(want_symbol) = expected_symbol(gpu, &self.arch, rows, self.pm) {
                if launched.symbol.as_deref() != Some(want_symbol.as_str()) {
                    eprintln!(
                        "SYMBOL {label} rows={rows} {tag}: launched {:?}, expected {want_symbol}",
                        launched.symbol
                    );
                    symbol = 1;
                }
            }
        }
        if even + odd + guard + symbol != 0 {
            eprintln!(
                "MISMATCH {label} rows={rows} {tag} {}: even-column words {even}, odd-column words {odd}, guard words {guard}",
                if graph { "graph" } else { "eager" }
            );
        }
        let entry = self.tallies.entry(label.to_string()).or_default();
        entry.runs += 1;
        entry.even += even;
        entry.odd += odd;
        entry.guard += guard;
        entry.symbol += symbol;
        self.cases += 1;
    }

    /// Every row count 1..=32 eagerly; graph replay for `GRAPH_ROWS`; for the
    /// view cases (`views`) permuted rows and a sub-view at row offset 3.
    fn run(&mut self, gpu: &mut Gpu, label: &str, slots: &Slots, x: &GpuTensor, views: bool) {
        for rows in 1..=MAX_ROWS {
            let ids: Vec<usize> = (0..rows).collect();
            self.check(gpu, label, "", slots, x, &ids, false);
            if GRAPH_ROWS.contains(&rows) {
                self.check(gpu, label, "", slots, x, &ids, true);
            }
        }
        if !views {
            return;
        }
        let k = present(slots)[0].k;
        let host = gpu.download_f32(x).expect("download x");
        for rows in [2usize, 9, 15, 16, 17, 31, 32] {
            // Reversed rows: launch row i is reference row rows-1-i.
            let mut permuted = host.clone();
            for i in 0..rows {
                permuted[i * k..(i + 1) * k]
                    .copy_from_slice(&host[(rows - 1 - i) * k..(rows - i) * k]);
            }
            let xp = gpu
                .upload_f32(&permuted, &[permuted.len()])
                .expect("upload permuted x");
            let ids: Vec<usize> = (0..rows).rev().collect();
            self.check(gpu, label, "permuted", slots, &xp, &ids, false);
            gpu.free_tensor(xp).expect("free permuted x");
            // Sub-view at row offset 3.
            if rows + 3 <= MAX_ROWS {
                let xv = x.sub_offset(3 * k, (MAX_ROWS - 3) * k);
                let ids: Vec<usize> = (3..3 + rows).collect();
                self.check(gpu, label, "subview+3", slots, &xv, &ids, false);
            }
        }
    }
}

/// Graph replay repeated after the activation is overwritten at the same
/// address: capture once, replay with x, overwrite x in place, replay again.
fn graph_changed_x(
    gpu: &mut Gpu,
    label: &str,
    slots: &Slots,
    rows: usize,
    seed: u64,
    ctx: &mut Ctx,
) {
    let mats = present(slots);
    let k = mats[0].k;
    let mut rng = Rng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let first = host_x(k, &mut rng);
    let x = gpu
        .upload_f32(&first, &[first.len()])
        .expect("upload stable x");
    // References of the first activation (a fresh singleton pass per matrix).
    let mut refs: Vec<Vec<u32>> = mats
        .iter()
        .map(|m| singleton_reference(gpu, &m.w, &x, m.m, m.k))
        .collect();
    let outs: Vec<GpuTensor> = mats
        .iter()
        .map(|mat| {
            let n = GUARD + rows * mat.m + GUARD;
            gpu.upload_f32(&vec![f32::from_bits(NAN_BITS); n], &[n])
                .expect("alloc y")
        })
        .collect();
    let views: Vec<GpuTensor> = mats
        .iter()
        .zip(&outs)
        .map(|(mat, y)| y.sub_offset(GUARD, rows * mat.m))
        .collect();
    let mut next = 0usize;
    let parts: Vec<(&GpuTensor, &GpuTensor, usize)> = slots
        .iter()
        .map(|slot| match slot {
            Some(mat) => {
                let p = (&mat.w, &views[next], mat.m);
                next += 1;
                p
            }
            None => (&mats[0].w, &views[0], 0),
        })
        .collect();
    let parts: [(&GpuTensor, &GpuTensor, usize); 4] = parts.try_into().ok().expect("four parts");
    gpu.wide_verify_rows = 0;
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    gpu.begin_stream_capture().expect("begin capture");
    let captured = gpu.gemv_mq4g256v2_rows_x4(parts, &x, k, rows);
    let g = gpu.end_stream_capture().expect("end capture");
    captured.expect("captured launch");
    let exec = gpu.hip.graph_instantiate(&g).expect("instantiate");
    for round in 0..2 {
        if round == 1 {
            let second = host_x(k, &mut rng);
            gpu.hip.device_synchronize().expect("sync");
            let bytes: Vec<u8> = second.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip
                .memcpy_htod(&x.buf, &bytes)
                .expect("overwrite x in place");
            refs = mats
                .iter()
                .map(|m| singleton_reference(gpu, &m.w, &x, m.m, m.k))
                .collect();
        }
        for y in &outs {
            fill_nan(gpu, y);
        }
        gpu.launch_graph(&exec).expect("graph replay");
        gpu.hip.device_synchronize().expect("sync");
        let (mut even, mut odd, mut guard) = (0, 0, 0);
        for ((y, reference), mat) in outs.iter().zip(&refs).zip(&mats) {
            let got = fetch(gpu, y);
            let (e, o, g) = tally(&got, &reference[..rows * mat.m], mat.m);
            even += e;
            odd += o;
            guard += g;
        }
        if even + odd + guard != 0 {
            eprintln!("MISMATCH {label} rows={rows} graph round {round} (changed x): even {even} odd {odd} guard {guard}");
        }
        let entry = ctx
            .tallies
            .entry(format!("{label} graph changed-x"))
            .or_default();
        entry.runs += 1;
        entry.even += even;
        entry.odd += odd;
        entry.guard += guard;
        ctx.cases += 1;
    }
    gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
    gpu.hip.graph_destroy(g).expect("destroy graph");
    for y in outs {
        gpu.free_tensor(y).expect("free y");
    }
    gpu.free_tensor(x).expect("free x");
}

fn main() {
    if std::env::var_os("HIPFIRE_QWEN4_WIDE_TWIN_MQ4").is_none() {
        std::env::set_var("HIPFIRE_QWEN4_WIDE_TWIN_MQ4", "1");
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    let arch = gpu.arch.clone();
    let flag = hipfire_config::developer_bool("HIPFIRE_QWEN4_MQ4_ROWS_PM", true);
    let pm = flag && matches!(arch.as_str(), "gfx1151" | "gfx1201");
    let wide_on = WideTwinFamily::Mq4.enabled() && matches!(arch.as_str(), "gfx1151" | "gfx1201");
    println!("mq4_rows_oracle: arch={arch}");
    println!(
        "mq4_rows_oracle: route={} (HIPFIRE_QWEN4_MQ4_ROWS_PM {}); rows 9..=32 {}",
        if pm {
            "embedded PM image mq4_rows_x4_pm"
        } else {
            "per-row gemv_mq4g256v2 fallback"
        },
        if flag { "on" } else { "0" },
        if wide_on {
            "-> wide twins r16/r32 (HIPFIRE_QWEN4_WIDE_TWIN_MQ4)"
        } else {
            "-> exact 8-row group loops (wide twins OFF)"
        },
    );
    let mut ctx = Ctx {
        arch: arch.clone(),
        pm,
        cases: 0,
        tallies: BTreeMap::new(),
    };

    for seed in 1..=3u64 {
        let mut rng = Rng(seed.wrapping_mul(0xA24B_AED4_963E_E407));
        let mut xs: BTreeMap<usize, GpuTensor> = BTreeMap::new();
        for k in [256usize, 512, 768, 1024, 1280, 2560, 6144] {
            let x = upload_x(&mut gpu, k, &mut rng);
            xs.insert(k, x);
        }

        // (a) trunk shapes, one per launch; (b) the real groupings.
        let mut trunk: Vec<Mat> = Vec::new();
        for &(name, m, k) in &SHAPES {
            trunk.push(make_mat(
                &mut gpu,
                name.to_string(),
                m,
                k,
                &xs[&k],
                &mut rng,
            ));
        }
        for (i, mat) in trunk.iter().enumerate() {
            let label = format!("single {} M={} K={}", mat.name, mat.m, mat.k);
            ctx.run(
                &mut gpu,
                &label,
                &[Some(mat), None, None, None],
                &xs[&mat.k],
                i == 0,
            );
        }
        for &(name, idx) in &GROUPS {
            let mut slots: Slots = [None; 4];
            for (s, &i) in idx.iter().enumerate() {
                slots[s] = Some(&trunk[i]);
            }
            let k = trunk[idx[0]].k;
            ctx.run(&mut gpu, name, &slots, &xs[&k], true);
            if seed == 1 {
                graph_changed_x(&mut gpu, name, &slots, 5, seed, &mut ctx);
                graph_changed_x(&mut gpu, name, &slots, 24, seed, &mut ctx);
            }
        }

        // Negative controls on the GDN grouping at 3 rows (narrow) and 20 rows (wide):
        // a launch must disagree with a reference computed from a perturbed
        // activation, and gdn.a's output must disagree with gdn.b's reference
        // (swapped matrices).
        if seed == 1 {
            let slots: Slots = [
                Some(&trunk[0]),
                Some(&trunk[1]),
                Some(&trunk[2]),
                Some(&trunk[3]),
            ];
            for rows in [3usize, 20] {
                let got = launch(&mut gpu, &slots, &xs[&2560], rows, false).bits;
                let mut host = gpu.download_f32(&xs[&2560]).expect("download x");
                host[0] += 1.0;
                let x_perturbed = gpu.upload_f32(&host, &[host.len()]).expect("upload x'");
                let perturbed_bad: Vec<usize> = slots
                    .iter()
                    .flatten()
                    .zip(&got)
                    .map(|(mat, g)| {
                        let want =
                            &singleton_reference(&mut gpu, &mat.w, &x_perturbed, mat.m, mat.k)
                                [..rows * mat.m];
                        let (e, o, gd) = tally(g, want, mat.m);
                        e + o + gd
                    })
                    .collect();
                gpu.free_tensor(x_perturbed).expect("free x'");
                let (a, b) = (trunk[2].m, trunk[3].m);
                assert_eq!(
                    (trunk[2].name.as_str(), trunk[3].name.as_str(), a, b),
                    ("gdn.a", "gdn.b", 48, 48)
                );
                let (e, o, gd) = tally(&got[2], &trunk[3].reference[..rows * b], a);
                let swapped_bad = e + o + gd;
                // Even/odd column swap: column 2j of the launch against reference column 2j+1.
                let (mut parity_agree, mut parity_total) = (0usize, 0usize);
                let ga = &got[2][GUARD..GUARD + rows * a];
                for r in 0..rows {
                    for c in (0..a).step_by(2) {
                        parity_total += 1;
                        parity_agree +=
                            usize::from(ga[r * a + c] == trunk[2].reference[r * a + c + 1]);
                    }
                }
                println!(
                    "mq4_rows_oracle: negative control rows={rows}: perturbed-x mismatch words per output {perturbed_bad:?}, swapped a/b {swapped_bad}, even/odd-swapped columns agreeing {parity_agree}/{parity_total}"
                );
                if perturbed_bad.iter().any(|&n| n == 0)
                    || swapped_bad == 0
                    || parity_agree * 2 > parity_total
                {
                    eprintln!("negative control FAILED: a wrong reference agreed with the launch (vacuous oracle)");
                    std::process::exit(1);
                }
            }
        }
        for mat in trunk {
            gpu.free_tensor(mat.w).expect("free weights");
        }

        // (c) synthetic single matrices: odd / block-straddling M x every K tail.
        let mut syn = 0u64;
        for &k in &SYN_K {
            for &m in &SYN_M {
                syn += 1;
                let mat = make_mat(
                    &mut gpu,
                    format!("syn M={m} K={k}"),
                    m,
                    k,
                    &xs[&k],
                    &mut rng,
                );
                let label = format!("synthetic single M={m} K={k}");
                // The full 1..=32 sweep for every (M, K); views and graph sample on the small ones.
                ctx.run(
                    &mut gpu,
                    &label,
                    &[Some(&mat), None, None, None],
                    &xs[&k],
                    syn % 5 == 0,
                );
                gpu.free_tensor(mat.w).expect("free weights");
            }
        }

        // (d) synthetic x4 groupings: odd boundaries, zero-M parts in any slot.
        for &k in &SYN_GROUP_K {
            for ms in &SYN_GROUPS {
                let mats: Vec<Option<Mat>> = ms
                    .iter()
                    .enumerate()
                    .map(|(s, &m)| {
                        (m > 0).then(|| {
                            make_mat(
                                &mut gpu,
                                format!("slot{s} M={m} K={k}"),
                                m,
                                k,
                                &xs[&k],
                                &mut rng,
                            )
                        })
                    })
                    .collect();
                let slots: Slots = [
                    mats[0].as_ref(),
                    mats[1].as_ref(),
                    mats[2].as_ref(),
                    mats[3].as_ref(),
                ];
                let label = format!("synthetic x4 M={ms:?} K={k}");
                ctx.run(&mut gpu, &label, &slots, &xs[&k], true);
                if seed == 1 && k == 1280 {
                    graph_changed_x(&mut gpu, &label, &slots, 17, seed, &mut ctx);
                }
                for mat in mats.into_iter().flatten() {
                    gpu.free_tensor(mat.w).expect("free weights");
                }
            }
        }

        for (_, x) in xs {
            gpu.free_tensor(x).expect("free x");
        }
    }

    println!("mq4_rows_oracle: per-case summary (runs = seeds 1..=3 x rows 1..=32 eager + graph samples + views)");
    let mut worst_label = 0;
    for label in ctx.tallies.keys() {
        worst_label = worst_label.max(label.len());
    }
    for (label, t) in &ctx.tallies {
        println!(
            "  {label:<worst_label$} runs={:<4} even_mismatch_words={} odd_mismatch_words={} guard_words={} symbol_errors={}",
            t.runs, t.even, t.odd, t.guard, t.symbol
        );
    }
    let bad = ctx.bad_total();
    if bad != 0 {
        println!(
            "mq4_rows_oracle: FAIL cases={} mismatch_bytes={}",
            ctx.cases,
            bad * 4
        );
        std::process::exit(1);
    }
    println!(
        "mq4_rows_oracle: PASS cases={} mismatch_bytes=0 ({})",
        ctx.cases,
        if wide_on {
            "wide twins r16/r32 exercised for rows 9..=32"
        } else {
            "wide twins OFF: exact 8-row group loops"
        }
    );
}
