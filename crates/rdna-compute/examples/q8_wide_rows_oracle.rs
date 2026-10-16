// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle for the exact wide Q8_0 twins (R = 16 / 32 activation rows):
//! `gemv_q8_0_{k2560_staged,k320_staged}_rows_wide_r{16,32}`,
//! `gemv_q8_0_k2560_staged_rows_x2_wide_r{16,32}`,
//! `gemv_q8_0_k8_rows_wide_r{16,32}` (module `qwen4_gemv_q8_0`) and
//! `gemv_q8_0_wide_rows_wide_r{16,32}` (module `qwen4_gemv_q8_0_wide`).
//!
//! `q8_wide_rows_oracle`: runs on gfx1151 and gfx1201 (the kernels are
//! arch-generic source; production serves Q8 on exact gfx1151 only). The twins are
//! launched raw (`ensure_kernel_public` + `launch_kernel_blob`), so no
//! `gemv.rs` binding is involved. Every actual row count `R = 1..=32` is
//! compared with `to_bits` bytes (the whole output allocation, guards and
//! unwritten rows included, sentinel-prefilled) against `R` independent
//! single-row launches of the object the decode path really selects:
//! `Gpu::gemv_q8_0` for the plain GEMVs (its own per-arch/K selection) and
//! `Gpu::gemv_q8_0_k8` for HC k8. Separate certificates, one recipe each:
//!
//! * `k2560` single and `k2560x2`: one-chain recipe (gemv_q8_0_k2560_staged /
//!   gemv_q8_0 generic), x2 against two single launches.
//! * `k320`: gfx1151's `gemv_q8_0_k320_staged` is the one-chain recipe, so the
//!   `k320` twin is the arm there. On every other arch `gemv_q8_0` picks the
//!   four-chain `gemv_q8_0_wide` at K = 320, so the arm there is the four-chain
//!   twin (the arm names the twin it actually ran).
//! * `k640`: four-chain (gemv_q8_0_wide_k640_staged on gfx1151).
//! * `smallk`: other K <= 1536 with `K % 128 != 0` block tails (and aligned K),
//!   four-chain with unfused tail blocks.
//! * `k8`: HC k8, `hc_act` 0 and 1 (scale 0.25), partial final tiles, plus
//!   per-partition certificates: X masked to one K/8 partition leaves only that
//!   partition's single-chain sum non-zero, so the fixed fold reproduces it and
//!   the comparison vs the k8 singleton checks every one of the eight chains.
//!
//! Per case: forward sweep over R and over capacity (r16 for R <= 16, r32 for
//! all R); a reversed-row-order run (output row r must equal reference row
//! R-1-r); a replay with new X contents at stable addresses; negative controls
//! (swapped weights and changed X must change bytes); and, for K >= 128, a
//! recipe-discrimination control (the other recipe's twin must NOT reproduce the
//! decode object's bytes, proving the data separates one-chain from four-chain).
//! The process exits 0 only if every check holds.
//!
//! Commands (the lead runs each on its own GPU):
//!   cargo run --release -p rdna-compute --example q8_wide_rows_oracle   # gfx1151
//!   cargo run --release -p rdna-compute --example q8_wide_rows_oracle   # gfx1201

use hip_bridge::KernargBlob;
use rdna_compute::kernels::{QWEN4_GEMV_Q8_0_SRC, QWEN4_GEMV_Q8_0_WIDE_SRC};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 32;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x7f;
const HC_SCALE: f32 = 0.25;
/// Row magnitudes (row r uses `AMPS[r % 8]`): `0.0` is the all-zero (+-0) row,
/// `1024` reaches the HC exp guards.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];
const MOD_Q8: &str = "qwen4_gemv_q8_0";
const MOD_Q8_WIDE: &str = "qwen4_gemv_q8_0_wide";

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn fill(&mut self, out: &mut [u8]) {
        let mut chunks = out.chunks_exact_mut(8);
        for c in &mut chunks {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rest = chunks.into_remainder();
        let tail = self.next().to_le_bytes();
        rest.copy_from_slice(&tail[..rest.len()]);
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    /// A finite f16: mostly `2^-lo..2^hi`, sometimes `+-0` or subnormal.
    fn f16(&mut self, lo_exp: u16, hi_exp: u16) -> u16 {
        let sign = (self.below(2) as u16) << 15;
        match self.below(32) {
            0 => sign,
            1 => sign | (1 + self.below(0x3ff) as u16),
            _ => {
                sign | ((15 - lo_exp + self.below(u64::from(lo_exp + hi_exp + 1)) as u16) << 10)
                    | self.below(0x400) as u16
            }
        }
    }
}

/// Q8_0 weights `[m][k/32]` of an f16 scale and 32 int8 (all of -128..=127).
/// `wide_scales` stretches the scale exponents to 2^-12..2^10.
fn gen_q8(m: usize, k: usize, wide_scales: bool, rng: &mut Rng) -> Vec<u8> {
    let (lo, hi) = if wide_scales { (12, 10) } else { (8, 0) };
    let mut bytes = vec![0u8; m * (k / 32) * 34];
    for blk in bytes.chunks_exact_mut(34) {
        rng.fill(&mut blk[2..]);
        blk[..2].copy_from_slice(&rng.f16(lo, hi).to_le_bytes());
    }
    bytes
}

/// `[MAXR][k]` activations, row r at magnitude `AMPS[r % 8]`, with +-0 and F32
/// subnormals; `keep` (if given) zeroes everything outside `[lo, hi)` of each row.
fn gen_x(k: usize, keep: Option<(usize, usize)>, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % 8];
        for i in 0..k {
            let v = match rng.below(64) {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                0 => 0.0,
                1 => -0.0,
                2..=5 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => rng.unit() * amp,
            };
            x.push(match keep {
                Some((lo, hi)) if i < lo || i >= hi => 0.0,
                _ => v,
            });
        }
    }
    x
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// One output allocation: `GUARD | MAXR * m f32 | GUARD`.
struct Out {
    full: GpuTensor,
    m: usize,
}

impl Out {
    fn new(gpu: &mut Gpu, m: usize) -> Result<Self> {
        let full = gpu
            .zeros(&[2 * GUARD + MAXR * m], DType::F32)
            .map_err(err)?;
        Ok(Self { full, m })
    }
    fn view(&self) -> GpuTensor {
        self.full.sub_offset(GUARD, MAXR * self.m)
    }
    fn bytes(&self) -> usize {
        (2 * GUARD + MAXR * self.m) * 4
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.full.buf, i32::from(SENTINEL), self.bytes())
            .map_err(err)
    }
    fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        gpu.hip.device_synchronize().map_err(err)?;
        let mut v = vec![0u8; self.full.buf.size()];
        gpu.hip.memcpy_dtoh(&mut v, &self.full.buf).map_err(err)?;
        v.truncate(self.bytes());
        Ok(v)
    }
    /// Sentinel everywhere except `rows` rows, each `f(r)`'s row of `reference`.
    fn expected(&self, reference: &[u8], rows: usize, f: impl Fn(usize) -> usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let row = self.m * 4;
        for r in 0..rows {
            let src = GUARD * 4 + f(r) * row;
            let dst = GUARD * 4 + r * row;
            e[dst..dst + row].copy_from_slice(&reference[src..src + row]);
        }
        e
    }
    fn locate(&self, idx: usize) -> String {
        let (lo, hi) = (GUARD * 4, GUARD * 4 + MAXR * self.m * 4);
        if idx < lo {
            format!("head guard byte {idx}")
        } else if idx >= hi {
            format!("tail guard byte {}", idx - hi)
        } else {
            let off = idx - lo;
            let (row, rem) = (off / (self.m * 4), off % (self.m * 4));
            format!("row {row} col {} byte {}", rem / 4, rem % 4)
        }
    }
}

/// The decode object on row `c` of X (`x` is `[MAXR][k]`), writing row `c` of every output.
type RefFn = Box<dyn Fn(&mut Gpu, &GpuTensor, usize) -> Result<()>>;
/// The twin at capacity `cap` (16/32), `rows` rows; `swap` uses the alternate weights.
type TwinFn = Box<dyn Fn(&mut Gpu, &GpuTensor, usize, usize, bool) -> Result<()>>;

struct Case {
    cert: &'static str,
    name: String,
    k: usize,
    outs: Vec<Out>,
    x: GpuTensor,
    x_alt: GpuTensor,
    reference: RefFn,
    twin: TwinFn,
    /// The other recipe's twin (same ABI), expected NOT to reproduce the decode bytes.
    other: Option<TwinFn>,
    /// Symbols this case launches (resource gate).
    symbols: Vec<String>,
}

#[derive(Default)]
struct Tally {
    cases: usize,
    launches: usize,
    bad_bytes: usize,
    bad_checks: usize,
    first: Vec<String>,
}

impl Tally {
    fn fail(&mut self, msg: String) {
        self.bad_checks += 1;
        if self.first.len() < 12 {
            self.first.push(msg);
        }
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n + a.len().abs_diff(b.len()), first)
}

fn run_reference(gpu: &mut Gpu, case: &Case, x: &GpuTensor) -> Result<Vec<Vec<u8>>> {
    for o in &case.outs {
        o.reset(gpu)?;
    }
    for c in 0..MAXR {
        (case.reference)(gpu, x, c)?;
    }
    case.outs.iter().map(|o| o.download(gpu)).collect()
}

#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    case: &Case,
    tally: &mut Tally,
    label: &str,
    refs: &[Vec<u8>],
    x: &GpuTensor,
    cap: usize,
    rows: usize,
    perm: &dyn Fn(usize) -> usize,
) -> Result<()> {
    for o in &case.outs {
        o.reset(gpu)?;
    }
    (case.twin)(gpu, x, cap, rows, false)?;
    tally.launches += 1;
    for (oi, o) in case.outs.iter().enumerate() {
        let got = o.download(gpu)?;
        let want = o.expected(&refs[oi], rows, perm);
        let (bad, at) = first_diff(&got, &want);
        if bad != 0 {
            tally.bad_bytes += bad;
            tally.fail(format!(
                "[{}] {} {label} cap {cap} rows {rows} out {oi}: {bad} bad bytes, first at {}",
                case.cert,
                case.name,
                at.map_or("?".to_string(), |i| o.locate(i)),
            ));
        }
    }
    Ok(())
}

fn verify(gpu: &mut Gpu, case: &mut Case, rng: &mut Rng, tally: &mut Tally) -> Result<()> {
    tally.cases += 1;
    let ident = |r: usize| r;
    let x = Rc::new(case.x.sub_offset(0, MAXR * case.k));
    let refs = run_reference(gpu, case, &x)?;
    // Forward sweep: every actual row count, both capacities where legal.
    for rows in 1..=MAXR {
        if rows <= 16 {
            check(gpu, case, tally, "fwd", &refs, &x, 16, rows, &ident)?;
        }
        check(gpu, case, tally, "fwd", &refs, &x, 32, rows, &ident)?;
    }
    // Row permutation: X rows reversed within `rows`; output row r == reference row rows-1-r.
    for rows in [9usize, 16, 17, 32] {
        let mut xh = vec![0f32; MAXR * case.k];
        let src = gpu.download_f32(&x).map_err(err)?;
        for r in 0..MAXR {
            let from = if r < rows { rows - 1 - r } else { r };
            xh[r * case.k..(r + 1) * case.k]
                .copy_from_slice(&src[from * case.k..(from + 1) * case.k]);
        }
        gpu.hip
            .memcpy_htod(&case.x_alt.buf, &f32_bytes(&xh))
            .map_err(err)?;
        let xr = case.x_alt.sub_offset(0, MAXR * case.k);
        let cap = if rows <= 16 { 16 } else { 32 };
        check(gpu, case, tally, "reversed", &refs, &xr, cap, rows, &|r| {
            rows - 1 - r
        })?;
    }
    // Negative controls (each must change bytes).
    {
        let rows = 32;
        let alt_x: Vec<f32> = gen_x(case.k, None, rng);
        gpu.hip
            .memcpy_htod(&case.x_alt.buf, &f32_bytes(&alt_x))
            .map_err(err)?;
        let xa = case.x_alt.sub_offset(0, MAXR * case.k);
        for swap in [true, false] {
            for o in &case.outs {
                o.reset(gpu)?;
            }
            let xin: &GpuTensor = if swap { &*x } else { &xa };
            (case.twin)(gpu, xin, 32, rows, swap)?;
            tally.launches += 1;
            let mut changed = false;
            for (oi, o) in case.outs.iter().enumerate() {
                let got = o.download(gpu)?;
                let want = o.expected(&refs[oi], rows, &ident);
                changed |= first_diff(&got, &want).0 != 0;
            }
            if !changed {
                tally.fail(format!(
                    "[{}] {} negative control ({}) did not change a single byte",
                    case.cert,
                    case.name,
                    if swap { "swapped weights" } else { "changed X" }
                ));
            }
        }
        if let (Some(other), true) = (&case.other, case.k >= 128) {
            for o in &case.outs {
                o.reset(gpu)?;
            }
            other(gpu, &x, 32, rows, false)?;
            tally.launches += 1;
            let mut changed = false;
            for (oi, o) in case.outs.iter().enumerate() {
                let got = o.download(gpu)?;
                let want = o.expected(&refs[oi], rows, &ident);
                changed |= first_diff(&got, &want).0 != 0;
            }
            if !changed {
                tally.fail(format!(
                    "[{}] {} recipe-discrimination control: the other recipe reproduced the decode bytes (data does not separate the chains)",
                    case.cert, case.name
                ));
            }
        }
    }
    // Replay: new X contents at the same address, decode object and twin again.
    let fresh = gen_x(case.k, None, rng);
    gpu.hip
        .memcpy_htod(&case.x.buf, &f32_bytes(&fresh))
        .map_err(err)?;
    let refs2 = run_reference(gpu, case, &x)?;
    for rows in [9usize, 16, 17, 31, 32] {
        let cap = if rows <= 16 { 16 } else { 32 };
        check(gpu, case, tally, "replay", &refs2, &x, cap, rows, &ident)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Case builders.

#[derive(Clone, Copy, PartialEq)]
enum Chain {
    One,
    Four,
}

fn sym_of(tag: &str, cap: usize) -> String {
    format!("gemv_q8_0_{tag}_rows_wide_r{cap}")
}

fn ensure(gpu: &mut Gpu, chain: Chain, tag: &str) -> Result<()> {
    for cap in [16, 32] {
        let sym = sym_of(tag, cap);
        let (module, src) = if chain == Chain::Four && tag == "wide" {
            (MOD_Q8_WIDE, QWEN4_GEMV_Q8_0_WIDE_SRC)
        } else {
            (MOD_Q8, QWEN4_GEMV_Q8_0_SRC)
        };
        gpu.ensure_kernel_public(module, src, &sym).map_err(err)?;
    }
    Ok(())
}

/// (A, X, Y, M, K, R) twin launcher for tag in {k2560_staged, k320_staged, wide}.
fn single_twin(
    tag: &'static str,
    w: Rc<GpuTensor>,
    w_alt: Rc<GpuTensor>,
    y: Rc<GpuTensor>,
    m: usize,
    k: usize,
) -> TwinFn {
    Box::new(move |gpu, x, cap, rows, swap| {
        let mut b = KernargBlob::new();
        b.push_ptr(if swap {
            w_alt.buf.as_ptr()
        } else {
            w.buf.as_ptr()
        });
        b.push_ptr(x.buf.as_ptr());
        b.push_ptr(y.buf.as_ptr());
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(rows as i32);
        gpu.launch_kernel_blob(
            &sym_of(tag, cap),
            [m.div_ceil(8) as u32, 1, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .map_err(err)
    })
}

struct Fixtures {
    w: Rc<GpuTensor>,
    w_alt: Rc<GpuTensor>,
    x: GpuTensor,
    x_alt: GpuTensor,
}

fn fixtures(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    wide_scales: bool,
    keep: Option<(usize, usize)>,
    rng: &mut Rng,
) -> Result<Fixtures> {
    let wb = gen_q8(m, k, wide_scales, rng);
    let wb_alt = gen_q8(m, k, wide_scales, rng);
    let w = Rc::new(gpu.upload_raw(&wb, &[wb.len()]).map_err(err)?);
    let w_alt = Rc::new(gpu.upload_raw(&wb_alt, &[wb_alt.len()]).map_err(err)?);
    let xs = gen_x(k, keep, rng);
    let x = gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?;
    let x_alt = gpu.upload_f32(&xs, &[xs.len()]).map_err(err)?;
    Ok(Fixtures { w, w_alt, x, x_alt })
}

/// One-chain / four-chain single GEMV certificate against `Gpu::gemv_q8_0`.
#[allow(clippy::too_many_arguments)]
fn single_case(
    gpu: &mut Gpu,
    cert: &'static str,
    chain: Chain,
    tag: &'static str,
    other_tag: &'static str,
    m: usize,
    k: usize,
    wide_scales: bool,
    rng: &mut Rng,
) -> Result<Case> {
    ensure(gpu, chain, tag)?;
    ensure(
        gpu,
        if chain == Chain::One {
            Chain::Four
        } else {
            Chain::One
        },
        other_tag,
    )?;
    let f = fixtures(gpu, m, k, wide_scales, None, rng)?;
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let (w1, y1) = (f.w.clone(), y.clone());
    let reference: RefFn = Box::new(move |gpu, x, c| {
        gpu.gemv_q8_0(&w1, &x.sub_offset(c * k, k), &y1.sub_offset(c * m, m), m, k)
            .map_err(err)
    });
    let twin = single_twin(tag, f.w.clone(), f.w_alt.clone(), y.clone(), m, k);
    let other = single_twin(other_tag, f.w.clone(), f.w_alt.clone(), y, m, k);
    Ok(Case {
        cert,
        name: format!(
            "{tag} m={m} k={k}{}",
            if wide_scales { " wide-scales" } else { "" }
        ),
        k,
        outs: vec![out],
        x: f.x,
        x_alt: f.x_alt,
        reference,
        twin,
        other: Some(other),
        symbols: vec![sym_of(tag, 16), sym_of(tag, 32)],
    })
}

/// Two K = 2560 matrices vs two `Gpu::gemv_q8_0` launches.
fn x2_case(gpu: &mut Gpu, m0: usize, m1: usize, rng: &mut Rng) -> Result<Case> {
    const K: usize = 2560;
    for cap in [16, 32] {
        gpu.ensure_kernel_public(
            MOD_Q8,
            QWEN4_GEMV_Q8_0_SRC,
            &format!("gemv_q8_0_k2560_staged_rows_x2_wide_r{cap}"),
        )
        .map_err(err)?;
    }
    let f0 = fixtures(gpu, m0, K, false, None, rng)?;
    let wb1 = gen_q8(m1, K, false, rng);
    let wb1_alt = gen_q8(m1, K, false, rng);
    let w1 = Rc::new(gpu.upload_raw(&wb1, &[wb1.len()]).map_err(err)?);
    let w1_alt = Rc::new(gpu.upload_raw(&wb1_alt, &[wb1_alt.len()]).map_err(err)?);
    let (o0, o1) = (Out::new(gpu, m0)?, Out::new(gpu, m1)?);
    let (y0, y1) = (Rc::new(o0.view()), Rc::new(o1.view()));
    let (a0, a1, ry0, ry1) = (f0.w.clone(), w1.clone(), y0.clone(), y1.clone());
    let reference: RefFn = Box::new(move |gpu, x, c| {
        let xr = x.sub_offset(c * K, K);
        gpu.gemv_q8_0(&a0, &xr, &ry0.sub_offset(c * m0, m0), m0, K)
            .map_err(err)?;
        gpu.gemv_q8_0(&a1, &xr, &ry1.sub_offset(c * m1, m1), m1, K)
            .map_err(err)
    });
    let (b0, b0_alt, b1, b1_alt) = (f0.w.clone(), f0.w_alt.clone(), w1.clone(), w1_alt.clone());
    let twin: TwinFn = Box::new(move |gpu, x, cap, rows, swap| {
        let mut b = KernargBlob::new();
        b.push_ptr(if swap {
            b0_alt.buf.as_ptr()
        } else {
            b0.buf.as_ptr()
        });
        b.push_ptr(if swap {
            b1_alt.buf.as_ptr()
        } else {
            b1.buf.as_ptr()
        });
        b.push_ptr(x.buf.as_ptr());
        b.push_ptr(y0.buf.as_ptr());
        b.push_ptr(y1.buf.as_ptr());
        b.push_i32(m0 as i32);
        b.push_i32(m1 as i32);
        b.push_i32(K as i32);
        b.push_i32(rows as i32);
        let grid = (m0.div_ceil(8) + m1.div_ceil(8)) as u32;
        gpu.launch_kernel_blob(
            &format!("gemv_q8_0_k2560_staged_rows_x2_wide_r{cap}"),
            [grid, 1, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .map_err(err)
    });
    Ok(Case {
        cert: "k2560x2",
        name: format!("x2 m0={m0} m1={m1} k={K}"),
        k: K,
        outs: vec![o0, o1],
        x: f0.x,
        x_alt: f0.x_alt,
        reference,
        twin,
        other: None,
        symbols: vec![
            "gemv_q8_0_k2560_staged_rows_x2_wide_r16".into(),
            "gemv_q8_0_k2560_staged_rows_x2_wide_r32".into(),
        ],
    })
}

/// HC k8 vs `Gpu::gemv_q8_0_k8`; `keep_part` masks X to one K/8 partition.
fn k8_case(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    hc: bool,
    keep_part: Option<usize>,
    rng: &mut Rng,
) -> Result<Case> {
    for cap in [16, 32] {
        gpu.ensure_kernel_public(
            MOD_Q8,
            QWEN4_GEMV_Q8_0_SRC,
            &format!("gemv_q8_0_k8_rows_wide_r{cap}"),
        )
        .map_err(err)?;
    }
    let pk = k / 8;
    let keep = keep_part.map(|p| (p * pk, (p + 1) * pk));
    let f = fixtures(gpu, m, k, false, keep, rng)?;
    let out = Out::new(gpu, m)?;
    let y = Rc::new(out.view());
    let scale = hc.then_some(HC_SCALE);
    let (w1, y1) = (f.w.clone(), y.clone());
    let reference: RefFn = Box::new(move |gpu, x, c| {
        gpu.gemv_q8_0_k8(
            &w1,
            &x.sub_offset(c * k, k),
            &y1.sub_offset(c * m, m),
            m,
            k,
            scale,
        )
        .map_err(err)
    });
    let (w, w_alt) = (f.w.clone(), f.w_alt.clone());
    let twin: TwinFn = Box::new(move |gpu, x, cap, rows, swap| {
        let mut b = KernargBlob::new();
        b.push_ptr(if swap {
            w_alt.buf.as_ptr()
        } else {
            w.buf.as_ptr()
        });
        b.push_ptr(x.buf.as_ptr());
        b.push_ptr(y.buf.as_ptr());
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_f32(scale.unwrap_or(1.0));
        b.push_i32(i32::from(scale.is_some()));
        b.push_i32(rows as i32);
        gpu.launch_kernel_blob(
            &format!("gemv_q8_0_k8_rows_wide_r{cap}"),
            [m.div_ceil(2) as u32, 1, 1],
            [512, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .map_err(err)
    });
    Ok(Case {
        cert: if keep_part.is_some() {
            "k8-partition"
        } else {
            "k8"
        },
        name: format!(
            "k8 m={m} k={k} hc_act={}{}",
            u8::from(hc),
            keep_part.map_or(String::new(), |p| format!(" partition={p}"))
        ),
        k,
        outs: vec![out],
        x: f.x,
        x_alt: f.x_alt,
        reference,
        twin,
        other: None,
        symbols: vec![
            "gemv_q8_0_k8_rows_wide_r16".into(),
            "gemv_q8_0_k8_rows_wide_r32".into(),
        ],
    })
}

fn occupancy_gate(gpu: &mut Gpu, symbols: &[String], block: u32) -> Result<()> {
    // Static LDS is part of the code object, so dynamic shared memory is 0.
    for s in symbols {
        let blocks = gpu
            .occupancy_max_active_blocks(s, [block, 1, 1], 0)
            .map_err(err)?;
        if blocks < 1 {
            return Err(format!(
                "{s}: occupancy_max_active_blocks = {blocks} (block {block})"
            ));
        }
        println!("  {s}: {blocks} resident workgroup(s)/CU of {block} threads");
    }
    Ok(())
}

fn run() -> Result<bool> {
    let mut gpu = Gpu::init().map_err(err)?;
    let arch = gpu.arch.clone();
    if arch != "gfx1151" && arch != "gfx1201" {
        return Err(format!("needs gfx1151 or gfx1201, found {arch}"));
    }
    gpu.lane_rows_scope = false;
    let one_chain_k320 = arch == "gfx1151";
    let mut rng = Rng(0x0858_c8c8_9f3a_7d11);
    let mut tally = Tally::default();
    let mut cases: Vec<Case> = Vec::new();
    println!(
        "arch {arch}; K=320 decode recipe: {}",
        if one_chain_k320 {
            "one-chain (gemv_q8_0_k320_staged)"
        } else {
            "four-chain (gemv_q8_0_wide)"
        }
    );
    // K2560 single, one-chain (gemv_q8_0_k2560_staged on gfx1151, gemv_q8_0 elsewhere): both
    // are the same chain, so the twin is the arm on both archs.
    for (m, wide) in [
        (1, false),
        (7, false),
        (8, false),
        (9, false),
        (17, false),
        (33, true),
        (640, false),
        (2560, false),
    ] {
        cases.push(single_case(
            &mut gpu,
            "k2560",
            Chain::One,
            "k2560_staged",
            "wide",
            m,
            2560,
            wide,
            &mut rng,
        )?);
    }
    for (m0, m1) in [(1, 1), (8, 9), (9, 7), (640, 640), (17, 33)] {
        cases.push(x2_case(&mut gpu, m0, m1, &mut rng)?);
    }
    // K320: arm named by what gemv_q8_0 selects on this arch.
    for m in [1, 7, 8, 9, 33, 512] {
        if one_chain_k320 {
            cases.push(single_case(
                &mut gpu,
                "k320",
                Chain::One,
                "k320_staged",
                "wide",
                m,
                320,
                m == 33,
                &mut rng,
            )?);
        } else {
            cases.push(single_case(
                &mut gpu,
                "k320(four-chain on this arch)",
                Chain::Four,
                "wide",
                "k320_staged",
                m,
                320,
                m == 33,
                &mut rng,
            )?);
        }
    }
    // K640 four-chain (shared-expert down).
    for (m, wide) in [(1, false), (8, false), (9, false), (48, true), (640, false)] {
        cases.push(single_case(
            &mut gpu,
            "k640",
            Chain::Four,
            "wide",
            "k2560_staged",
            m,
            640,
            wide,
            &mut rng,
        )?);
    }
    // Other small K, with and without block tails (K = 32 * nblk; 320/640 have their own arms).
    for nblk in [
        1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 23, 31, 33, 47, 48,
    ] {
        for m in [1, 9, 17] {
            cases.push(single_case(
                &mut gpu,
                "smallk",
                Chain::Four,
                "wide",
                "k2560_staged",
                m,
                nblk * 32,
                nblk % 5 == 0,
                &mut rng,
            )?);
        }
    }
    // HC k8: production K10240, K2048, partial final tiles (pk blocks 3 / 6 / 9 / 20).
    for hc in [false, true] {
        for (m, k) in [
            (1, 10240),
            (2, 10240),
            (3, 10240),
            (9, 10240),
            (320, 10240),
            (5, 2048),
            (3, 768),
            (3, 1536),
            (3, 2304),
            (4, 5120),
        ] {
            cases.push(k8_case(&mut gpu, m, k, hc, None, &mut rng)?);
        }
    }
    // All eight partition chains, one case per partition (hc_act = 0: raw fold).
    for k in [2304usize, 10240] {
        for p in 0..8 {
            cases.push(k8_case(&mut gpu, 9, k, false, Some(p), &mut rng)?);
        }
    }
    println!("resource gate:");
    occupancy_gate(
        &mut gpu,
        &cases.iter().find(|c| c.cert == "k2560").unwrap().symbols,
        256,
    )?;
    occupancy_gate(
        &mut gpu,
        &cases.iter().find(|c| c.cert == "k2560x2").unwrap().symbols,
        256,
    )?;
    occupancy_gate(
        &mut gpu,
        &cases.iter().find(|c| c.cert == "k640").unwrap().symbols,
        256,
    )?;
    occupancy_gate(
        &mut gpu,
        &cases.iter().find(|c| c.cert == "k8").unwrap().symbols,
        512,
    )?;
    if one_chain_k320 {
        occupancy_gate(
            &mut gpu,
            &cases.iter().find(|c| c.cert == "k320").unwrap().symbols,
            256,
        )?;
    }
    let mut per_cert: Vec<(String, usize, usize, usize)> = Vec::new();
    for mut case in cases {
        let (l0, f0) = (tally.launches, tally.bad_checks);
        verify(&mut gpu, &mut case, &mut rng, &mut tally)?;
        let cert = case.cert.to_string();
        match per_cert.iter_mut().find(|e| e.0 == cert) {
            Some(e) => {
                e.1 += 1;
                e.2 += tally.launches - l0;
                e.3 += tally.bad_checks - f0;
            }
            None => per_cert.push((cert, 1, tally.launches - l0, tally.bad_checks - f0)),
        }
    }
    for (cert, n, launches, bad) in &per_cert {
        println!("certificate {cert:<34} {n:>3} cases {launches:>6} twin launches, {bad} failing check(s)");
    }
    println!(
        "{} cases, {} twin launches, {} bad bytes, {} failing checks",
        tally.cases, tally.launches, tally.bad_bytes, tally.bad_checks
    );
    for line in &tally.first {
        eprintln!("FAIL {line}");
    }
    Ok(tally.bad_checks == 0 && tally.bad_bytes == 0)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("q8_wide_rows_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("q8_wide_rows_oracle: {e}");
            std::process::exit(2);
        }
    }
}
