// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-decode byte oracle for the wide MQ6G256V2 verify twins
//! (`gemm_mq6g256v2_f32_rows_wide_r{16,32}`,
//! `gemm_mq6g256v2_f32_rows_x4_wide_r{16,32}`, `kernels/src/gemm_mq6g256v2_f32_rows.hip`),
//! for every actual row count `R` in `1..=32`, on gfx1151 and gfx1201.
//!
//! `mq6_wide_rows_oracle [--quick] [--no-symbol-check]`
//!
//! The reference is `R` independent single-row launches of the actual AR decode
//! objects, never the R8192 prefill reference of `qwen4_mq6_rows`: the one-matrix
//! form is `Gpu::gemv_mq6g256v2` per activation row; the four-matrix form is
//! `Gpu::gemv_mq6g256v2_x4` per activation row (the grouped AR decode launch) and
//! is itself checked bit-for-bit against `gemv_mq6g256v2` per matrix. Whole output
//! allocations (head/tail guards, the unused rows `R..32`, sentinel `0xFF..` NaN
//! bytes everywhere a kernel must not write, including the columns past `M` of the
//! last workgroup) are compared as bytes, so every `to_bits()` word is checked.
//!
//! Per fixture (one-matrix and x4 shapes covering K in 1..=24 groups so every
//! quad/tail stream split, odd/short/zero `M`, `M = 48`, long `K = 10240`, the
//! real trunk shapes, and an "extreme" header/magnitude fixture reaching
//! `+-65504` f16 headers, subnormals, `+-0` and overflow/NaN):
//!
//! 1. direct launches of `wide_r16` for `R = 1..=16` and `wide_r32` for `R = 1..=32`
//!    (the ABI offsets are built here: single `A@0 x@8 y@16 M@24 K@28 R@32`, x4
//!    `A0..A3@0..24 x@32 y0..y3@40..64 M0..M3@72..84 K@88 R@92`, grid
//!    `sum ceil(Mi/8)`, block 256) against the reference;
//! 2. the public launchers `gemm_mq6g256v2_f32_rows[_x4]` for `R = 2..=32` against
//!    the same reference, and the actually launched symbol
//!    (`Gpu::last_launched_kernel`): `_r<R>` for `R <= 8`, `_wide_r16` for
//!    `9..=16`, `_wide_r32` for `17..=32`, or on gfx1151 the admitted lane-rows
//!    PeaceMaker twin of that width (`gemv_mq6g256v2[_x4]_xbatch[_wide_r16|_wide_r32]_pm`)
//!    (`--no-symbol-check` for a tree where the launchers still serve `R > 8` with
//!    the eight-row loop);
//! 3. X with its rows reversed (row permutation) and a second, different X written
//!    into the SAME stable buffers (replay with changed contents), each against a
//!    freshly recomputed reference; the stale reference must then differ;
//! 4. negative control: the same launch with swapped weights must differ from the
//!    original reference.
//!
//! X lives inside a NaN-filled allocation and every launch gets an exact-size view
//! (non-zero element offset), so a kernel reading outside its rows poisons outputs.
//! `--quick` runs `R in {1,2,8,9,15,16,17,24,31,32}` only. Exit 0 only if every
//! comparison, every symbol and every negative control holds.
//!
//! `lane_rows_scope` is forced off: rows `2..=8` are the hipcc rows kernels.

use rdna_compute::{kernels, DType, Gpu, GpuTensor};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 32;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
/// Output prefill: `0xFFFFFFFF` is a NaN no kernel produces.
const SENTINEL: u8 = 0xff;
/// NaN activation rows before and after the `[MAXR, K]` X view.
const PRE_ROWS: usize = 3;
const MODULE: &str = "gemm_mq6g256v2_f32_rows";
/// Per-row magnitudes cycle; `0.0` is the all-zero (+-0) row.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

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
    /// [`Rng::f16`], or in an extreme fixture sometimes a legal extreme header.
    fn header(&mut self, extreme: bool, lo_exp: u16, hi_exp: u16) -> u16 {
        const SPECIAL: [u16; 8] = [
            0x7bff, 0xfbff, 0x0001, 0x8001, 0x03ff, 0x0400, 0x0000, 0x8000,
        ];
        if extreme && self.below(4) == 0 {
            return SPECIAL[self.below(8) as usize];
        }
        self.f16(lo_exp, hi_exp)
    }
}

/// MQ6G256V2 weights: `[m][k/256]` groups of 8 header bytes (`s0 z0 s1 z1`, f16,
/// independent per half) and 192 random packed payload bytes (every six-bit
/// field including the q5 word-boundary crossing).
fn gen_mq6(m: usize, k: usize, extreme: bool, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; m * (k / 256) * 200];
    for g in bytes.chunks_exact_mut(200) {
        rng.fill(&mut g[8..]);
        for half in 0..2 {
            let (s, z) = (rng.header(extreme, 9, 2), rng.header(extreme, 7, 5));
            g[4 * half..4 * half + 2].copy_from_slice(&s.to_le_bytes());
            g[4 * half + 2..4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
    }
    bytes
}

/// `[MAXR][k]` activations: row magnitudes cycle through [`AMPS`]; row kinds
/// add alternating signs (`r % 11 == 3`), +-2^20 cancellation with a small
/// residue (`5`), pure subnormals/zeros (`7`), and in an extreme fixture
/// `1e30`-scale rows (`r % 13 == 9`) that overflow against extreme headers.
fn gen_x(k: usize, extreme: bool, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % AMPS.len()];
        for i in 0..k {
            let v = match r % 11 {
                _ if amp == 0.0 => {
                    if rng.below(2) == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                3 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                5 => match i % 4 {
                    0 => 1_048_576.0,
                    1 => -1_048_576.0,
                    _ => rng.unit() * 1e-3,
                },
                7 => f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32),
                _ => match rng.below(64) {
                    0 => 0.0,
                    1 => -0.0,
                    2..=5 => {
                        f32::from_bits(((rng.below(2) as u32) << 31) | rng.below(1 << 23) as u32)
                    }
                    _ if extreme && r % 13 == 9 => rng.unit() * 1e30,
                    _ => rng.unit() * amp,
                },
            };
            x.push(v);
        }
    }
    x
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
    /// Expected bytes: sentinel everywhere, `reference`'s first `rows` rows in place.
    fn expected(&self, reference: &[u8], rows: usize) -> Vec<u8> {
        let mut e = vec![SENTINEL; self.bytes()];
        let (lo, hi) = (GUARD * 4, GUARD * 4 + rows * self.m * 4);
        e[lo..hi].copy_from_slice(&reference[lo..hi]);
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
            format!("row {row} column {} byte {}", rem / 4, rem % 4)
        }
    }
}

/// One shape: one matrix (`ms = [m, 0, 0, 0]`) or the x4 group `ms`, shared K.
struct Fixture {
    name: String,
    x4: bool,
    ms: [usize; 4],
    k: usize,
    ws: Vec<Rc<GpuTensor>>,
    /// Same shapes, different bytes: the swapped-weights negative control.
    ws_alt: Vec<Rc<GpuTensor>>,
    outs: Vec<Out>,
    /// `[PRE_ROWS NaN rows | MAXR rows | PRE_ROWS NaN rows] x k`.
    x_full: GpuTensor,
    x_host: Vec<f32>,
}

fn upload_weights(
    gpu: &mut Gpu,
    ms: [usize; 4],
    k: usize,
    extreme: bool,
    rng: &mut Rng,
) -> Result<Vec<Rc<GpuTensor>>> {
    let mut ws = Vec::new();
    for &m in &ms {
        // An unused slot still needs a valid pointer.
        let bytes = if m == 0 {
            vec![0u8; 256]
        } else {
            gen_mq6(m, k, extreme, rng)
        };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
    }
    Ok(ws)
}

impl Fixture {
    fn new(
        gpu: &mut Gpu,
        ms: [usize; 4],
        x4: bool,
        k: usize,
        extreme: bool,
        rng: &mut Rng,
    ) -> Result<Self> {
        let ws = upload_weights(gpu, ms, k, extreme, rng)?;
        let ws_alt = upload_weights(gpu, ms, k, extreme, rng)?;
        let mut outs = Vec::new();
        for &m in &ms {
            outs.push(Out::new(gpu, m)?);
        }
        let x_host = gen_x(k, extreme, rng);
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
        let name = if x4 {
            format!(
                "mq6 x4 ms={ms:?} k={k}{}",
                if extreme { " extreme" } else { "" }
            )
        } else {
            format!(
                "mq6 m={} k={k}{}",
                ms[0],
                if extreme { " extreme" } else { "" }
            )
        };
        let fx = Self {
            name,
            x4,
            ms,
            k,
            ws,
            ws_alt,
            outs,
            x_full,
            x_host,
        };
        fx.set_x(gpu, &fx.x_host.clone())?;
        Ok(fx)
    }

    /// Write `rows` (`[MAXR, k]`) into the stable X allocation between NaN rows.
    fn set_x(&self, gpu: &Gpu, rows: &[f32]) -> Result<()> {
        let nan = f32::NAN.to_bits().to_le_bytes();
        let mut bytes = Vec::with_capacity((2 * PRE_ROWS + MAXR) * self.k * 4);
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        for v in rows {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        for _ in 0..PRE_ROWS * self.k {
            bytes.extend_from_slice(&nan);
        }
        gpu.hip.memcpy_htod(&self.x_full.buf, &bytes).map_err(err)
    }

    /// Exact-size view of the first `rows` activation rows.
    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }

    fn reset(&self, gpu: &Gpu) -> Result<()> {
        self.outs.iter().try_for_each(|o| o.reset(gpu))
    }

    fn download(&self, gpu: &Gpu) -> Result<Vec<Vec<u8>>> {
        self.outs.iter().map(|o| o.download(gpu)).collect()
    }

    /// The actual one-matrix AR decode launch, once per matrix and activation row.
    fn ref_single(&self, gpu: &mut Gpu) -> Result<()> {
        let x = self.x_view(MAXR);
        for r in 0..MAXR {
            for i in 0..4 {
                let (m, k) = (self.ms[i], self.k);
                if m > 0 {
                    let y = self.outs[i].view().sub_offset(r * m, m);
                    gpu.gemv_mq6g256v2(&self.ws[i], &x.sub_offset(r * k, k), &y, m, k)
                        .map_err(err)?;
                }
            }
        }
        Ok(())
    }

    /// The actual grouped AR decode launch (`gemv_mq6g256v2_x4`), once per activation row.
    fn ref_x4(&self, gpu: &mut Gpu) -> Result<()> {
        let x = self.x_view(MAXR);
        for r in 0..MAXR {
            let ys: Vec<GpuTensor> = (0..4)
                .map(|i| self.outs[i].view().sub_offset(r * self.ms[i], self.ms[i]))
                .collect();
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
            gpu.gemv_mq6g256v2_x4(parts, &x.sub_offset(r * self.k, self.k), self.k)
                .map_err(err)?;
        }
        Ok(())
    }

    /// The decode-selected reference bytes (guards and unused rows still sentinel):
    /// one-matrix `gemv_mq6g256v2`, or `gemv_mq6g256v2_x4` after proving it equals
    /// `gemv_mq6g256v2` per matrix. Bytes that differ between the two are added
    /// to `tally.ref_disagree`.
    fn reference(
        &self,
        gpu: &mut Gpu,
        tally: &mut Tally,
        notes: &mut Vec<String>,
    ) -> Result<Vec<Vec<u8>>> {
        self.reset(gpu)?;
        self.ref_single(gpu)?;
        let by_single = self.download(gpu)?;
        if !self.x4 {
            return Ok(by_single);
        }
        self.reset(gpu)?;
        self.ref_x4(gpu)?;
        let by_x4 = self.download(gpu)?;
        let (bad, first) = self.compare(&by_x4, &by_single);
        tally.ref_disagree += bad;
        if let Some(first) = first {
            notes.push(format!(
                "decode x4 differs from per-matrix decode at {first}"
            ));
        }
        Ok(by_x4)
    }

    /// Public rows launcher (`rows >= 2`).
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize) -> Result<()> {
        let x = self.x_view(rows);
        if self.x4 {
            let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
            gpu.gemm_mq6g256v2_f32_rows_x4(parts, &x, self.k, rows)
                .map_err(err)
        } else {
            let y = self.outs[0].view();
            gpu.gemm_mq6g256v2_f32_rows(&self.ws[0], &x, &y, self.ms[0], self.k, rows)
                .map_err(err)
        }
    }

    fn symbol(&self, cap: usize) -> String {
        format!(
            "gemm_mq6g256v2_f32_rows{}_wide_r{cap}",
            if self.x4 { "_x4" } else { "" }
        )
    }

    /// Direct launch of the wide entry of capacity `cap` at `rows`, ABI built here.
    fn run_direct(&self, gpu: &mut Gpu, cap: usize, rows: usize, alt: bool) -> Result<()> {
        let name = self.symbol(cap);
        gpu.ensure_kernel_public(MODULE, kernels::GEMM_MQ6G256V2_F32_ROWS_SRC, &name)
            .map_err(err)?;
        let ws = if alt { &self.ws_alt } else { &self.ws };
        let x = self.x_view(rows);
        let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
        let mut kb = hip_bridge::KernargBlob::new();
        let blocks: usize;
        if self.x4 {
            for w in ws.iter() {
                kb.push_ptr(w.buf.as_ptr() as *const std::ffi::c_void);
            }
            kb.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
            for y in &ys {
                kb.push_ptr(y.buf.as_ptr() as *const std::ffi::c_void);
            }
            for m in self.ms {
                kb.push_i32(m as i32);
            }
            kb.push_i32(self.k as i32);
            kb.push_i32(rows as i32);
            blocks = self.ms.iter().map(|m| m.div_ceil(8)).sum();
        } else {
            kb.push_ptr(ws[0].buf.as_ptr() as *const std::ffi::c_void);
            kb.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
            kb.push_ptr(ys[0].buf.as_ptr() as *const std::ffi::c_void);
            kb.push_i32(self.ms[0] as i32);
            kb.push_i32(self.k as i32);
            kb.push_i32(rows as i32);
            blocks = self.ms[0].div_ceil(8);
        }
        gpu.launch_kernel_blob(
            &name,
            [blocks as u32, 1, 1],
            [256, 1, 1],
            0,
            kb.as_mut_slice(),
        )
        .map_err(err)
    }

    /// Diff every output of `got` against `want`; the first difference is described.
    fn compare(&self, got: &[Vec<u8>], want: &[Vec<u8>]) -> (usize, Option<String>) {
        let mut bad = 0;
        let mut first = None;
        for (i, ((g, w), out)) in got.iter().zip(want).zip(&self.outs).enumerate() {
            let (n, at) = diff(g, w);
            bad += n;
            if let (None, Some(at)) = (&first, at) {
                first = Some(format!(
                    "matrix {i} {} (got {:#04x}, want {:#04x})",
                    out.locate(at),
                    g[at],
                    w[at]
                ));
            }
        }
        (bad, first)
    }

    /// Download and compare against the first `rows` rows of `reference`.
    fn check(
        &self,
        gpu: &Gpu,
        reference: &[Vec<u8>],
        rows: usize,
    ) -> Result<(usize, Option<String>)> {
        let expected: Vec<Vec<u8>> = self
            .outs
            .iter()
            .zip(reference)
            .map(|(o, r)| o.expected(r, rows))
            .collect();
        let got = self.download(gpu)?;
        Ok(self.compare(&got, &expected))
    }
}

/// `(mismatching byte count, first mismatching index)`.
fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
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

#[derive(Default)]
struct Tally {
    cases: usize,
    launches: usize,
    bad_bytes: usize,
    ref_disagree: usize,
    bad_symbols: usize,
    negative_control_blind: usize,
}

impl Tally {
    fn failed(&self) -> bool {
        self.bad_bytes + self.ref_disagree + self.bad_symbols + self.negative_control_blind > 0
    }
}

struct Opts {
    quick: bool,
    symbols: bool,
}

fn row_set(max: usize, quick: bool) -> Vec<usize> {
    if quick {
        [1usize, 2, 8, 9, 15, 16, 17, 24, 31, 32]
            .into_iter()
            .filter(|&r| r <= max)
            .collect()
    } else {
        (1..=max).collect()
    }
}

fn expected_symbol(x4: bool, rows: usize) -> String {
    let form = if x4 { "_x4" } else { "" };
    match rows {
        0..=8 => format!("gemm_mq6g256v2_f32_rows{form}_r{rows}"),
        9..=16 => format!("gemm_mq6g256v2_f32_rows{form}_wide_r16"),
        _ => format!("gemm_mq6g256v2_f32_rows{form}_wide_r32"),
    }
}

/// The gfx1151 lane-rows PeaceMaker twin the launcher prefers when admitted
/// (exact-gfx1151 image `qwen4_lane_rows_pm`, never loaded elsewhere); its
/// bytes are checked against the same single-row reference.
fn pm_symbol(x4: bool, rows: usize) -> String {
    let form = if x4 { "_x4" } else { "" };
    match rows {
        0..=8 => format!("gemv_mq6g256v2{form}_xbatch_pm"),
        9..=16 => format!("gemv_mq6g256v2{form}_xbatch_wide_r16_pm"),
        _ => format!("gemv_mq6g256v2{form}_xbatch_wide_r32_pm"),
    }
}

fn note(notes: &mut Vec<String>, text: String) {
    if notes.len() < 6 {
        notes.push(text);
    }
}

/// Every wide direct width and every launcher width against `reference`.
fn sweep(
    gpu: &mut Gpu,
    fx: &Fixture,
    reference: &[Vec<u8>],
    opt: &Opts,
    widths: &[(usize, usize)],
    tag: &str,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    for &(cap, max) in widths {
        for rows in row_set(max, opt.quick) {
            fx.reset(gpu)?;
            fx.run_direct(gpu, cap, rows, false)?;
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(notes, format!("{tag}wide_r{cap} R={rows}: {first}"));
            }
        }
    }
    for rows in row_set(MAXR, opt.quick).into_iter().filter(|&r| r >= 2) {
        fx.reset(gpu)?;
        fx.run_launcher(gpu, rows)?;
        let launched = gpu.last_launched_kernel().unwrap_or("<none>").to_string();
        let (bad, first) = fx.check(gpu, reference, rows)?;
        tally.launches += 1;
        tally.bad_bytes += bad;
        if let Some(first) = first {
            note(
                notes,
                format!("{tag}launcher R={rows} ({launched}): {first}"),
            );
        }
        let want = expected_symbol(fx.x4, rows);
        if opt.symbols && launched != want && launched != pm_symbol(fx.x4, rows) {
            tally.bad_symbols += 1;
            note(
                notes,
                format!("{tag}launcher R={rows} launched {launched}, expected {want}"),
            );
        }
    }
    Ok(())
}

fn verify(gpu: &mut Gpu, fx: &Fixture, opt: &Opts, tally: &mut Tally) -> Result<()> {
    let before = (
        tally.bad_bytes,
        tally.ref_disagree,
        tally.bad_symbols,
        tally.negative_control_blind,
    );
    let mut notes: Vec<String> = Vec::new();
    let widths = [(16usize, 16usize), (32, 32)];

    // 1 + 2: the fixture's own X.
    let reference = fx.reference(gpu, tally, &mut notes)?;
    sweep(gpu, fx, &reference, opt, &widths, "", tally, &mut notes)?;

    // 3a: row permutation (rows reversed) through the same buffers.
    let k = fx.k;
    let reversed: Vec<f32> = (0..MAXR)
        .rev()
        .flat_map(|r| fx.x_host[r * k..(r + 1) * k].to_vec())
        .collect();
    fx.set_x(gpu, &reversed)?;
    let reference_rev = fx.reference(gpu, tally, &mut notes)?;
    let few = Opts {
        quick: true,
        symbols: opt.symbols,
    };
    sweep(
        gpu,
        fx,
        &reference_rev,
        &few,
        &widths,
        "reversed X: ",
        tally,
        &mut notes,
    )?;

    // 3b: replay with changed contents at stable addresses (a fresh X in the same
    // allocation). The stale reference must now differ for a full-width launch.
    let mut rng = Rng(0x00ba_d5ee_d00d_f00d ^ (fx.ms[0] as u64) << 8 ^ k as u64);
    let changed = gen_x(k, false, &mut rng);
    fx.set_x(gpu, &changed)?;
    let reference_new = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference_new,
        &few,
        &widths,
        "changed X: ",
        tally,
        &mut notes,
    )?;
    fx.reset(gpu)?;
    fx.run_direct(gpu, 32, MAXR, false)?;
    let (stale_bad, _) = fx.check(gpu, &reference, MAXR)?;
    tally.launches += 1;
    if stale_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: changed X reproduced the stale reference".to_string(),
        );
    }

    // 4: swapped weights must not reproduce the original-weights reference.
    fx.set_x(gpu, &fx.x_host)?;
    fx.reset(gpu)?;
    fx.run_direct(gpu, 32, MAXR, true)?;
    let (swapped_bad, _) = fx.check(gpu, &reference, MAXR)?;
    tally.launches += 1;
    if swapped_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: swapped weights reproduced the original reference".to_string(),
        );
    }

    tally.cases += 1;
    let after = (
        tally.bad_bytes,
        tally.ref_disagree,
        tally.bad_symbols,
        tally.negative_control_blind,
    );
    let status = if before == after {
        "ok".to_string()
    } else {
        "MISMATCH".to_string()
    };
    println!(
        "{:<52} {status}{}",
        fx.name,
        if notes.is_empty() {
            String::new()
        } else {
            format!(" [{}]", notes.join("; "))
        }
    );
    Ok(())
}

/// Every verification shape of the contract: K 1..=24 groups across quad/tail
/// splits, odd/short/zero M, M = 48, real trunk shapes, long K and extremes.
fn fixtures(gpu: &mut Gpu) -> Result<Vec<Fixture>> {
    let mut seed = 0x6d71_3677_6964_6531u64;
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    let mut cases = Vec::new();
    // One matrix: (M, K).
    for (m, k) in [
        (1, 256),
        (2, 512),
        (3, 768),
        (7, 1024),
        (8, 1280),
        (9, 1536),
        (15, 1792),
        (16, 2048),
        (17, 2304),
        (48, 2560),
        (101, 6144),
        (64, 10240),
        (2560, 6144),
        (4099, 2560),
    ] {
        cases.push(Fixture::new(
            gpu,
            [m, 0, 0, 0],
            false,
            k,
            false,
            &mut next_rng(),
        )?);
    }
    cases.push(Fixture::new(
        gpu,
        [48, 0, 0, 0],
        false,
        2560,
        true,
        &mut next_rng(),
    )?);
    cases.push(Fixture::new(
        gpu,
        [13, 0, 0, 0],
        false,
        768,
        true,
        &mut next_rng(),
    )?);
    // Four matrices at one K (matrix-local M; zero-M slots own no workgroup).
    for (ms, k) in [
        ([10240, 6144, 48, 48], 2560),
        ([12288, 640, 512, 512], 2560),
        ([3, 0, 5, 0], 2560),
        ([1, 2, 3, 7], 2560),
        ([9, 0, 0, 17], 2560),
        ([0, 0, 0, 1], 2560),
        ([0, 5, 0, 0], 2560),
        ([7, 8, 9, 15], 768),
        ([48, 48, 1, 1], 1280),
        ([4099, 640, 512, 9], 2560),
    ] {
        cases.push(Fixture::new(gpu, ms, true, k, false, &mut next_rng())?);
    }
    cases.push(Fixture::new(
        gpu,
        [5, 13, 0, 9],
        true,
        2560,
        true,
        &mut next_rng(),
    )?);
    Ok(cases)
}

fn run() -> Result<bool> {
    let mut opt = Opts {
        quick: false,
        symbols: true,
    };
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--quick" => opt.quick = true,
            "--no-symbol-check" => opt.symbols = false,
            other => return Err(format!("unknown argument {other} (usage: mq6_wide_rows_oracle [--quick] [--no-symbol-check])")),
        }
    }
    let mut gpu = Gpu::init().map_err(err)?;
    gpu.lane_rows_scope = false;
    // The public launchers only admit R > 8 inside an open verify gate (as a
    // verify forward sets it); single-row references are unaffected.
    gpu.set_wide_verify_rows(rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS);
    println!(
        "arch {} (few_row_limit {})",
        gpu.arch.as_str(),
        gpu.few_row_limit()
    );
    for name in [
        "gemm_mq6g256v2_f32_rows_wide_r16",
        "gemm_mq6g256v2_f32_rows_wide_r32",
        "gemm_mq6g256v2_f32_rows_x4_wide_r16",
        "gemm_mq6g256v2_f32_rows_x4_wide_r32",
    ] {
        gpu.ensure_kernel_public(MODULE, kernels::GEMM_MQ6G256V2_F32_ROWS_SRC, name)
            .map_err(err)?;
        let resident = gpu
            .occupancy_max_active_blocks(name, [256, 1, 1], 0)
            .map_err(err)?;
        println!("{name}: occupancy API reports {resident} resident 256-thread workgroups per CU");
    }
    let mut tally = Tally::default();
    for fx in fixtures(&mut gpu)? {
        verify(&mut gpu, &fx, &opt, &mut tally)?;
    }
    println!(
        "{} cases, {} launches: {} bad bytes vs single-row decode, {} reference disagreement bytes, {} wrong symbols, {} blind negative controls",
        tally.cases, tally.launches, tally.bad_bytes, tally.ref_disagree, tally.bad_symbols, tally.negative_control_blind
    );
    Ok(!tally.failed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("mq6_wide_rows_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("mq6_wide_rows_oracle: {e}");
            std::process::exit(2);
        }
    }
}
