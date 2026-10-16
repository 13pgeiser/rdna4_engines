// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1151 byte oracle for the ordinary PeaceMaker BF16 x4 verify twins
//! `gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151` and
//! `gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151` of module
//! `bf16_rows_wide_pm_gfx1151` (`kernels/bf16_rows_wide_pm_gfx1151.hxaco`), for
//! every actual row count `R` the width serves (`1..=16` and `1..=32`).
//!
//! Two families of the same image are covered: the ordinary x4 pair above, and
//! the HC k4 pair `gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151` /
//! `gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151` (ABI48 `W@0 X@8 Y@16 M@24 K@28
//! scale@32 hc_act@36 R@40`, grid `ceil(M/2)`, block 256, static LDS 16384 (all
//! dynamic-shared arguments 0),
//! `K % 32 == 0`). HC fixtures first run a QUARTER GATE: the k4 per-quarter
//! partial sums are not exported, so they are not observed; instead each K/4
//! quarter, packed independently as its own `[M, K/4]` matrix with its own X
//! slices, runs through the ordinary PM x4 twins and is compared with the actual
//! `gemv_bf16_xf32` singleton at local K/4 (certifying the same ordinary
//! per-quarter DAG). The final k4 results are then compared byte for byte with
//! the actual singleton `Gpu::gemv_bf16_xf32_k4` in four modes: plain
//! (scale absent, `hc_act = 0`), kernarg scale .25 with `hc_act = 0` (must be
//! ignored), `hc_act` with scale .25 and `hc_act` with scale 1.
//!
//! ```text
//! cargo run --release -p rdna-compute --example bf16_rows_pm_oracle -- --quick
//! cargo run --release -p rdna-compute --example bf16_rows_pm_oracle
//! cargo run --release -p rdna-compute --example bf16_rows_pm_oracle -- --router512
//! HIPFIRE_QWEN4_WIDE_TWIN_BF16=1 cargo run --release -p rdna-compute \
//!     --example bf16_rows_pm_oracle -- --launcher
//! ```
//!
//! `bf16_rows_pm_oracle [--quick] [--router512] [--hc] [--launcher]`
//!
//! `--hc` adds the real HC k4 shapes (M 4 / 320 / 2560 at K = 10240) to the
//! always-run HC k4 tails; `--launcher` (needs `HIPFIRE_QWEN4_WIDE_TWIN_BF16=1`)
//! also asserts the symbol `gemv_bf16_xf32_x4_rows` / `gemv_bf16_xf32_k4_rows`
//! actually launched is the PM wide twin of the row count (rows 9..=16 `r16`,
//! 17..=32 `r32`; the x4 assertion is skipped for `K % 8 != 0`, the incumbent's
//! scalar domain).
//!
//! The reference is the actual AR decode objects, never a CPU transcription:
//! `Gpu::gemv_bf16_xf32` per matrix and activation row (singleton launches), and
//! `Gpu::gemv_bf16_xf32_x4` per activation row (the grouped decode launch). The
//! grouped launch is itself compared byte for byte with the singletons
//! (`reference disagreement`); the grouped bytes are the reference.
//!
//! Direct launches build the frozen 96-byte ABI here: `A0..A3 @0,8,16,24`,
//! `X @32`, `Y0..Y3 @40,48,56,64`, `M0..M3 @72,76,80,84`, `K @88`, `R @92`;
//! grid `sum ceil(Mi/8)`, block 256. Output `Yi` is `[R, Mi]` row-major.
//!
//! Per fixture (K scalar domain `K % 8 != 0`, partial-256 `K % 8 == 0`, the
//! production `K = 2560`; M slots mixing zero slots and odd/tail `M` 1, 2, 3, 7,
//! 8, 9, 15, 16, 17, 31; optional router shapes `M = 512`; plus wide-range
//! weight fixtures):
//!
//! 1. `r16` for `R = 1..=16` and `r32` for `R = 1..=32` against the reference;
//! 2. X with its rows reversed, and a second X written into the SAME stable
//!    buffers (replay with changed contents), each against a freshly recomputed
//!    reference; the stale reference must then differ;
//! 3. negative control: the same launch with different weights must differ from
//!    the original reference;
//! 4. with `--launcher`, the public `Gpu::gemv_bf16_xf32_x4_rows` for `R = 2..=32`
//!    (set `HIPFIRE_QWEN4_WIDE_TWIN_BF16=1` for the wide route) against the
//!    reference, printing the symbols it actually launched.
//!
//! Weights carry `+-0`, BF16 subnormals, wide and narrow exponents per row and a
//! NaN-poisoned pad after the last row; X rows differ per row (magnitude cycle,
//! alternating signs, `+-2^20` cancellation, F32 subnormals, `-0.0`/`+0.0` rows)
//! and sit inside a NaN-filled allocation with exact-size views. Whole output
//! allocations (head/tail guards, rows `R..32`, zero-M slots, the columns past
//! `M`) are prefilled with `0xFF` bytes and compared as bytes, so every `to_bits`
//! word and every guard is checked. Exit 0 only if every comparison and every
//! negative control holds; exit 2 on a launch/setup error.
//!
//! API: `Gpu::ensure_bf16_rows_wide_pm_entry(&mut self, symbol: &str) -> HipResult<()>`
//! verifies `QWEN4_BF16_ROWS_WIDE_PM_GFX1151_SHA256` and loads `symbol` of the
//! embedded module `bf16_rows_wide_pm_gfx1151` (exact gfx1151 only); the one call
//! site here is [`load_pm`]. Direct launches go through `Gpu::launch_kernel_blob`.

use rdna_compute::gemv::{wide_twin_rcap, WideTwinFamily, WIDE_TWIN_HC_M_BLOCK};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::rc::Rc;

type Result<T, E = String> = std::result::Result<T, E>;

const MAXR: usize = 32;
/// Guard floats on each side of every output allocation.
const GUARD: usize = 64;
/// Output prefill: `0xFFFFFFFF` is a NaN no kernel produces.
const SENTINEL: u8 = 0xff;
/// NaN activation rows before and after the `[MAXR, K]` X view.
const PRE_ROWS: usize = 3;
/// NaN-poisoned BF16 rows after each weight matrix.
const PAD_ROWS: usize = 8;
const ABI_BYTES: usize = 96;
const NAN_BF16: u16 = 0x7fc0;
/// Per-row magnitudes cycle.
const AMPS: [f32; 8] = [1.0, 1.0 / 1024.0, 1.0 / 16.0, 4.0, 64.0, 0.5, 0.0, 1024.0];

/// The two PM wide symbols: `(capacity, symbol)`.
const SYMBOLS: [(usize, &str); 2] = [
    (16, "gemv_bf16_xf32_x4_rows_wide_r16_pm_gfx1151"),
    (32, "gemv_bf16_xf32_x4_rows_wide_r32_pm_gfx1151"),
];

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
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// One BF16 weight matrix `[m, k]` (little-endian u16) followed by [`PAD_ROWS`]
/// NaN rows. Row kinds: ordinary mixed exponents; `+-0` only rows; subnormal-heavy
/// rows; in a `wide` fixture also large and tiny exponent rows.
fn gen_weights(m: usize, k: usize, wide: bool, rng: &mut Rng) -> Vec<u8> {
    let mut words: Vec<u16> = Vec::with_capacity((m + PAD_ROWS) * k);
    for r in 0..m {
        for _ in 0..k {
            let sign = (rng.below(2) as u16) << 15;
            let w = if r % 17 == 3 {
                // signed zeros only
                sign
            } else if r % 13 == 5 {
                // subnormal-heavy
                match rng.below(4) {
                    0 => sign,
                    1 => sign | (0x40 + rng.below(0x40) as u16),
                    _ => sign | (1 + rng.below(0x7f) as u16),
                }
            } else {
                match rng.below(64) {
                    0 => 0x0000,
                    1 => 0x8000,
                    2 => sign | (1 + rng.below(0x7f) as u16),
                    _ => {
                        let (lo, span) = if wide {
                            match r % 3 {
                                0 => (127 - 30, 61),
                                1 => (127 + 10, 20),
                                _ => (127 - 40, 20),
                            }
                        } else {
                            (127 - 12 + (r as u64 % 5), 21)
                        };
                        sign | (((lo + rng.below(span)) as u16) << 7) | rng.below(0x80) as u16
                    }
                }
            };
            words.push(w);
        }
    }
    words.extend(std::iter::repeat(NAN_BF16).take(PAD_ROWS * k));
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `[MAXR][k]` activations. Row magnitude cycles through [`AMPS`]; row kinds add
/// alternating signs (`r % 11 == 3`), `+-2^20` cancellation around a small residue
/// (`5`), F32 subnormals with `+-0` (`7`), all `-0.0` (`9`), alternating `+-0.0` (`10`).
fn gen_x(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = Vec::with_capacity(MAXR * k);
    for r in 0..MAXR {
        let amp = AMPS[r % AMPS.len()];
        for i in 0..k {
            let v = match r % 11 {
                3 => {
                    let m = rng.unit().abs() * amp.max(1.0 / 64.0);
                    if i % 2 == 0 {
                        m
                    } else {
                        -m
                    }
                }
                5 => {
                    if i == k / 3 {
                        5.0
                    } else if i % 2 == 0 {
                        (1u32 << 20) as f32
                    } else {
                        -((1u32 << 20) as f32)
                    }
                }
                7 => match rng.below(8) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => {
                        let sign = (rng.below(2) as u32) << 31;
                        f32::from_bits(sign | (1 + rng.below(0x7f_ffff) as u32))
                    }
                },
                9 => -0.0,
                10 => {
                    if i % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                _ => rng.unit() * amp,
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

/// One x4 shape: four matrices `ms` (zero slots allowed, at least one non-zero), shared K.
struct Fixture {
    name: String,
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
    wide: bool,
    rng: &mut Rng,
) -> Result<Vec<Rc<GpuTensor>>> {
    let mut ws = Vec::new();
    for &m in &ms {
        // An unused slot still needs a valid pointer (NaN-poisoned, never read).
        let bytes = if m == 0 {
            vec![0xffu8; 256]
        } else {
            gen_weights(m, k, wide, rng)
        };
        ws.push(Rc::new(
            gpu.upload_raw(&bytes, &[bytes.len()]).map_err(err)?,
        ));
    }
    Ok(ws)
}

impl Fixture {
    fn new(gpu: &mut Gpu, ms: [usize; 4], k: usize, wide: bool, rng: &mut Rng) -> Result<Self> {
        assert!(ms.iter().any(|&m| m > 0), "a fixture needs one non-zero M");
        let ws = upload_weights(gpu, ms, k, wide, rng)?;
        let ws_alt = upload_weights(gpu, ms, k, wide, rng)?;
        let mut outs = Vec::new();
        for &m in &ms {
            outs.push(Out::new(gpu, m)?);
        }
        let x_host = gen_x(k, rng);
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
        let name = format!(
            "bf16 x4 ms={ms:?} k={k}{}",
            if wide { " wide-range" } else { "" }
        );
        let fx = Self {
            name,
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

    /// The actual singleton decode launch, once per matrix and activation row.
    fn ref_single(&self, gpu: &mut Gpu) -> Result<()> {
        let x = self.x_view(MAXR);
        for r in 0..MAXR {
            for i in 0..4 {
                let (m, k) = (self.ms[i], self.k);
                if m > 0 {
                    let y = self.outs[i].view().sub_offset(r * m, m);
                    gpu.gemv_bf16_xf32(&self.ws[i], &x.sub_offset(r * k, k), &y, m, k)
                        .map_err(err)?;
                }
            }
        }
        Ok(())
    }

    /// The actual grouped decode launch (`gemv_bf16_xf32_x4`), once per activation row.
    fn ref_x4(&self, gpu: &mut Gpu) -> Result<()> {
        let x = self.x_view(MAXR);
        for r in 0..MAXR {
            let ys: Vec<GpuTensor> = (0..4)
                .map(|i| self.outs[i].view().sub_offset(r * self.ms[i], self.ms[i]))
                .collect();
            let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
                std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
            gpu.gemv_bf16_xf32_x4(parts, &x.sub_offset(r * self.k, self.k), self.k)
                .map_err(err)?;
        }
        Ok(())
    }

    /// Reference bytes (guards and unused rows still sentinel): the grouped decode
    /// launch, after comparing it with the singleton launches. Bytes that differ
    /// between the two are added to `tally.ref_disagree`.
    fn reference(
        &self,
        gpu: &mut Gpu,
        tally: &mut Tally,
        notes: &mut Vec<String>,
    ) -> Result<Vec<Vec<u8>>> {
        self.reset(gpu)?;
        self.ref_single(gpu)?;
        let by_single = self.download(gpu)?;
        self.reset(gpu)?;
        self.ref_x4(gpu)?;
        let by_x4 = self.download(gpu)?;
        let (bad, first) = self.compare(&by_x4, &by_single);
        tally.ref_disagree += bad;
        if let Some(first) = first {
            note(
                notes,
                format!("decode x4 differs from singleton decode at {first}"),
            );
        }
        Ok(by_x4)
    }

    /// Public rows launcher (`rows >= 2`).
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize) -> Result<()> {
        let x = self.x_view(rows);
        let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
        let parts: [(&GpuTensor, &GpuTensor, usize); 4] =
            std::array::from_fn(|i| (&*self.ws[i], &ys[i], self.ms[i]));
        gpu.gemv_bf16_xf32_x4_rows(parts, &x, self.k, rows)
            .map_err(err)
    }

    /// Direct launch of the PM wide entry of capacity `cap` at `rows`, ABI built here.
    fn run_direct(&self, gpu: &mut Gpu, cap: usize, rows: usize, alt: bool) -> Result<()> {
        let name = SYMBOLS
            .iter()
            .find(|(c, _)| *c == cap)
            .map(|(_, s)| *s)
            .ok_or("unknown capacity")?;
        load_pm(gpu, name)?;
        let ws = if alt { &self.ws_alt } else { &self.ws };
        let x = self.x_view(rows);
        let ys: Vec<GpuTensor> = self.outs.iter().map(|o| o.view()).collect();
        let mut kb = hip_bridge::KernargBlob::new();
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
        if kb.len() != ABI_BYTES {
            return Err(format!(
                "kernarg blob is {} bytes, the frozen ABI is {ABI_BYTES}",
                kb.len()
            ));
        }
        let blocks: usize = self.ms.iter().map(|m| m.div_ceil(8)).sum();
        gpu.launch_kernel_blob(
            name,
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

/// Load `symbol` of the verified `bf16_rows_wide_pm_gfx1151` module (see the
/// API prerequisite in the file header).
fn load_pm(gpu: &mut Gpu, symbol: &str) -> Result<()> {
    gpu.ensure_bf16_rows_wide_pm_entry(symbol).map_err(err)
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
    fn state(&self) -> (usize, usize, usize, usize) {
        (
            self.bad_bytes,
            self.ref_disagree,
            self.bad_symbols,
            self.negative_control_blind,
        )
    }
    fn failed(&self) -> bool {
        self.bad_bytes + self.ref_disagree + self.bad_symbols + self.negative_control_blind > 0
    }
}

struct Opts {
    quick: bool,
    router512: bool,
    launcher: bool,
    hc: bool,
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

fn note(notes: &mut Vec<String>, text: String) {
    if notes.len() < 6 {
        notes.push(text);
    }
}

/// Every PM wide direct width (and, with `--launcher`, every public launcher width,
/// asserting the launched PM symbol) against `reference`.
fn sweep(
    gpu: &mut Gpu,
    fx: &Fixture,
    reference: &[Vec<u8>],
    quick: bool,
    launcher: bool,
    tag: &str,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    for (cap, symbol) in SYMBOLS {
        for rows in row_set(cap, quick) {
            fx.reset(gpu)?;
            fx.run_direct(gpu, cap, rows, false)?;
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(notes, format!("{tag}{symbol} R={rows}: {first}"));
            }
        }
    }
    if launcher {
        let mut launched_symbols: Vec<String> = Vec::new();
        for rows in row_set(MAXR, quick).into_iter().filter(|&r| r >= 2) {
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
            // K % 8 != 0 is the incumbent's all-scalar domain: it keeps the <=8-row loop.
            if fx.k % 8 == 0 {
                if let Some(want) = pm_symbol(&SYMBOLS, rows) {
                    if launched != want {
                        tally.bad_symbols += 1;
                        note(
                            notes,
                            format!(
                                "{tag}x4 launcher R={rows} launched {launched}, expected {want}"
                            ),
                        );
                    }
                }
            }
            if !launched_symbols.contains(&launched) {
                launched_symbols.push(launched);
            }
        }
        note(
            notes,
            format!("{tag}x4 launcher symbols {launched_symbols:?}"),
        );
    }
    Ok(())
}

fn verify(gpu: &mut Gpu, fx: &Fixture, opt: &Opts, tally: &mut Tally) -> Result<()> {
    let before = tally.state();
    let mut notes: Vec<String> = Vec::new();

    // 1: the fixture's own X.
    let reference = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference,
        opt.quick,
        opt.launcher,
        "",
        tally,
        &mut notes,
    )?;

    // 2a: row permutation (rows reversed) through the same buffers.
    let k = fx.k;
    let reversed: Vec<f32> = (0..MAXR)
        .rev()
        .flat_map(|r| fx.x_host[r * k..(r + 1) * k].to_vec())
        .collect();
    fx.set_x(gpu, &reversed)?;
    let reference_rev = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference_rev,
        true,
        false,
        "reversed X: ",
        tally,
        &mut notes,
    )?;

    // 2b: replay with changed contents at stable addresses (a fresh X in the same
    // allocation). The stale reference must now differ for a full-width launch.
    let mut rng = Rng(0x00ba_d5ee_d00d_f00d
        ^ ((fx.ms[0] as u64) << 8)
        ^ ((fx.ms[3] as u64) << 20)
        ^ k as u64);
    let changed = gen_x(k, &mut rng);
    fx.set_x(gpu, &changed)?;
    let reference_new = fx.reference(gpu, tally, &mut notes)?;
    sweep(
        gpu,
        fx,
        &reference_new,
        true,
        false,
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

    // 3: different weights must not reproduce the original-weights reference.
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
    let status = if before == tally.state() {
        "ok"
    } else {
        "MISMATCH"
    };
    println!(
        "{:<56} {status}{}",
        fx.name,
        if notes.is_empty() {
            String::new()
        } else {
            format!(" [{}]", notes.join("; "))
        }
    );
    Ok(())
}

/// HC k4 PM wide symbols `(capacity, symbol)` of the same module.
const HC_SYMBOLS: [(usize, &str); 2] = [
    (16, "gemv_bf16_xf32_k4_rows_wide_r16_pm_gfx1151"),
    (32, "gemv_bf16_xf32_k4_rows_wide_r32_pm_gfx1151"),
];
/// Frozen HC ABI: `W@0 X@8 Y@16 M@24 K@28 scale@32 hc_act@36 R@40`, last 4 bytes
/// padding. The entry's 16384-byte LDS is static in its descriptor (the ordinary
/// entries carry 8192 static), so every dynamic-shared argument, launch and
/// occupancy alike, is 0.
const HC_ABI_BYTES: usize = 48;

/// One HC launch mode. `scale`/`hc_act` are the kernarg values of the direct PM
/// launch; `api` is the `hc_act_scale` argument of the actual singleton
/// `Gpu::gemv_bf16_xf32_k4` that is the reference (`None` is scale 1.0,
/// `hc_act = 0`; `Some(s)` is `hc_act = 1`).
struct HcMode {
    name: &'static str,
    scale: f32,
    hc_act: i32,
    api: Option<f32>,
}

const HC_MODES: [HcMode; 4] = [
    HcMode {
        name: "plain",
        scale: 1.0,
        hc_act: 0,
        api: None,
    },
    HcMode {
        name: "scale .25, hc_act 0",
        scale: 0.25,
        hc_act: 0,
        api: None,
    },
    HcMode {
        name: "hc_act scale .25",
        scale: 0.25,
        hc_act: 1,
        api: Some(0.25),
    },
    HcMode {
        name: "hc_act scale 1",
        scale: 1.0,
        hc_act: 1,
        api: Some(1.0),
    },
];

/// The HC k4 matrix `[m, k]` (one BF16 weight, `k % 32 == 0`), output `[R, m]`.
struct HcFixture {
    name: String,
    m: usize,
    k: usize,
    w: Rc<GpuTensor>,
    w_alt: Rc<GpuTensor>,
    /// Host bytes of `w` / `w_alt` (`[m, k]` BF16 rows plus the NaN pad), kept for
    /// the independently packed quarter matrices.
    w_bytes: Vec<u8>,
    w_alt_bytes: Vec<u8>,
    out: Out,
    x_full: GpuTensor,
    x_host: Vec<f32>,
}

impl HcFixture {
    fn new(gpu: &mut Gpu, m: usize, k: usize, wide: bool, rng: &mut Rng) -> Result<Self> {
        assert!(m > 0 && k % 32 == 0, "HC k4 needs M > 0 and K % 32 == 0");
        let w_bytes = gen_weights(m, k, wide, rng);
        let w = Rc::new(gpu.upload_raw(&w_bytes, &[w_bytes.len()]).map_err(err)?);
        let w_alt_bytes = gen_weights(m, k, wide, rng);
        let w_alt = Rc::new(
            gpu.upload_raw(&w_alt_bytes, &[w_alt_bytes.len()])
                .map_err(err)?,
        );
        let out = Out::new(gpu, m)?;
        let x_host = gen_x(k, rng);
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
        let name = format!(
            "bf16 k4 hc m={m} k={k}{}",
            if wide { " wide-range" } else { "" }
        );
        let fx = Self {
            name,
            m,
            k,
            w,
            w_alt,
            w_bytes,
            w_alt_bytes,
            out,
            x_full,
            x_host,
        };
        fx.set_x(gpu, &fx.x_host.clone())?;
        Ok(fx)
    }

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

    fn x_view(&self, rows: usize) -> GpuTensor {
        self.x_full.sub_offset(PRE_ROWS * self.k, rows * self.k)
    }

    /// Reference bytes of `mode`: the actual singleton `Gpu::gemv_bf16_xf32_k4`
    /// per activation row (guards and unused rows sentinel).
    fn reference(&self, gpu: &mut Gpu, mode: &HcMode) -> Result<Vec<u8>> {
        self.out.reset(gpu)?;
        let x = self.x_view(MAXR);
        for r in 0..MAXR {
            let y = self.out.view().sub_offset(r * self.m, self.m);
            gpu.gemv_bf16_xf32_k4(
                &self.w,
                &x.sub_offset(r * self.k, self.k),
                &y,
                self.m,
                self.k,
                mode.api,
            )
            .map_err(err)?;
        }
        self.out.download(gpu)
    }

    /// Public rows launcher (`rows >= 2`).
    fn run_launcher(&self, gpu: &mut Gpu, rows: usize, mode: &HcMode) -> Result<()> {
        let x = self.x_view(rows);
        let y = self.out.view();
        gpu.gemv_bf16_xf32_k4_rows(&self.w, &x, &y, self.m, self.k, mode.api, rows)
            .map_err(err)
    }

    /// Direct launch of the HC PM entry of capacity `cap`, ABI48 built here.
    fn run_direct(
        &self,
        gpu: &mut Gpu,
        cap: usize,
        rows: usize,
        mode: &HcMode,
        alt: bool,
    ) -> Result<()> {
        let name = HC_SYMBOLS
            .iter()
            .find(|(c, _)| *c == cap)
            .map(|(_, s)| *s)
            .ok_or("unknown capacity")?;
        load_pm(gpu, name)?;
        let w = if alt { &self.w_alt } else { &self.w };
        let x = self.x_view(rows);
        let y = self.out.view();
        let mut kb = hip_bridge::KernargBlob::new();
        kb.push_ptr(w.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_ptr(x.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_ptr(y.buf.as_ptr() as *const std::ffi::c_void);
        kb.push_i32(self.m as i32);
        kb.push_i32(self.k as i32);
        kb.push_f32(mode.scale);
        kb.push_i32(mode.hc_act);
        kb.push_i32(rows as i32);
        kb.pad_to(8);
        if kb.len() != HC_ABI_BYTES {
            return Err(format!(
                "HC kernarg blob is {} bytes, the frozen ABI is {HC_ABI_BYTES}",
                kb.len()
            ));
        }
        let blocks = self.m.div_ceil(WIDE_TWIN_HC_M_BLOCK);
        gpu.launch_kernel_blob(
            name,
            [blocks as u32, 1, 1],
            [256, 1, 1],
            0,
            kb.as_mut_slice(),
        )
        .map_err(err)
    }

    /// Download and compare against the first `rows` rows of `reference`.
    fn check(&self, gpu: &Gpu, reference: &[u8], rows: usize) -> Result<(usize, Option<String>)> {
        let expected = self.out.expected(reference, rows);
        let got = self.out.download(gpu)?;
        let (bad, at) = diff(&got, &expected);
        Ok((
            bad,
            at.map(|at| {
                format!(
                    "{} (got {:#04x}, want {:#04x})",
                    self.out.locate(at),
                    got[at],
                    expected[at]
                )
            }),
        ))
    }

    /// Quarter `q` as an ordinary one-matrix x4 fixture of width `K / 4`: the
    /// columns `[q*K/4, (q+1)*K/4)` of every weight row packed into their own
    /// matrix (plus the NaN pad), the same columns of every X row packed into a
    /// `[MAXR, K/4]` activation, and the slice of `w_alt` for the negative control.
    fn quarter_fixture(&self, gpu: &mut Gpu, q: usize) -> Result<Fixture> {
        let kq = self.k / 4;
        let pack = |bytes: &[u8]| -> Vec<u8> {
            let mut out = Vec::with_capacity((self.m + PAD_ROWS) * kq * 2);
            for r in 0..self.m {
                let o = (r * self.k + q * kq) * 2;
                out.extend_from_slice(&bytes[o..o + kq * 2]);
            }
            out.extend(
                std::iter::repeat(NAN_BF16.to_le_bytes())
                    .take(PAD_ROWS * kq)
                    .flatten(),
            );
            out
        };
        let x: Vec<f32> = (0..MAXR)
            .flat_map(|r| self.x_host[r * self.k + q * kq..r * self.k + (q + 1) * kq].to_vec())
            .collect();
        Fixture::from_data(
            gpu,
            format!("{} quarter {q}", self.name),
            self.m,
            kq,
            &pack(&self.w_bytes),
            &pack(&self.w_alt_bytes),
            x,
        )
    }
}

impl Fixture {
    /// One-matrix fixture (`ms = [m, 0, 0, 0]`) over caller-packed weights.
    fn from_data(
        gpu: &mut Gpu,
        name: String,
        m: usize,
        k: usize,
        w: &[u8],
        w_alt: &[u8],
        x_host: Vec<f32>,
    ) -> Result<Self> {
        let mut ws = Vec::new();
        let mut ws_alt = Vec::new();
        for slot in 0..4 {
            let (a, b): (&[u8], &[u8]) = if slot == 0 {
                (w, w_alt)
            } else {
                (&[0xffu8; 256], &[0xffu8; 256])
            };
            ws.push(Rc::new(gpu.upload_raw(a, &[a.len()]).map_err(err)?));
            ws_alt.push(Rc::new(gpu.upload_raw(b, &[b.len()]).map_err(err)?));
        }
        let ms = [m, 0, 0, 0];
        let mut outs = Vec::new();
        for &mi in &ms {
            outs.push(Out::new(gpu, mi)?);
        }
        let zeros = vec![0f32; (2 * PRE_ROWS + MAXR) * k];
        let x_full = gpu.upload_f32(&zeros, &[zeros.len()]).map_err(err)?;
        let fx = Self {
            name,
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
}

/// The wide PM symbol of `table` that serves `rows`, `None` at 1..=8 rows.
fn pm_symbol(table: &[(usize, &'static str); 2], rows: usize) -> Option<&'static str> {
    wide_twin_rcap(rows).map(|cap| table[usize::from(cap == 32)].1)
}

/// Every HC PM direct width (and, with `launcher`, every public `k4_rows`
/// launcher width, asserting the launched PM symbol) against `reference`.
#[allow(clippy::too_many_arguments)]
fn sweep_hc(
    gpu: &mut Gpu,
    fx: &HcFixture,
    reference: &[u8],
    mode: &HcMode,
    quick: bool,
    launcher: bool,
    tag: &str,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    for (cap, symbol) in HC_SYMBOLS {
        for rows in row_set(cap, quick) {
            fx.out.reset(gpu)?;
            fx.run_direct(gpu, cap, rows, mode, false)?;
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(
                    notes,
                    format!("{tag}{} {symbol} R={rows}: {first}", mode.name),
                );
            }
        }
    }
    if launcher {
        let mut launched_symbols: Vec<String> = Vec::new();
        for rows in row_set(MAXR, quick).into_iter().filter(|&r| r >= 2) {
            fx.out.reset(gpu)?;
            fx.run_launcher(gpu, rows, mode)?;
            let launched = gpu.last_launched_kernel().unwrap_or("<none>").to_string();
            let (bad, first) = fx.check(gpu, reference, rows)?;
            tally.launches += 1;
            tally.bad_bytes += bad;
            if let Some(first) = first {
                note(
                    notes,
                    format!("{tag}{} launcher R={rows} ({launched}): {first}", mode.name),
                );
            }
            if let Some(want) = pm_symbol(&HC_SYMBOLS, rows) {
                if launched != want {
                    tally.bad_symbols += 1;
                    note(
                        notes,
                        format!("{tag}k4 launcher R={rows} launched {launched}, expected {want}"),
                    );
                }
            }
            if !launched_symbols.contains(&launched) {
                launched_symbols.push(launched);
            }
        }
        note(
            notes,
            format!("{tag}k4 launcher symbols {launched_symbols:?}"),
        );
    }
    Ok(())
}

/// Quarter gate (runs before the final k4 modes). Scope: the k4 internal
/// per-quarter partial sums are not exported, so they are not observed. This
/// certifies the same ordinary per-quarter DAG instead: each of the four K/4
/// quarters, packed independently as its own `[M, K/4]` matrix with its own
/// `[R, K/4]` X, runs through the ordinary PM x4 wide twins (`r16` and `r32`)
/// and must equal the actual singleton `Gpu::gemv_bf16_xf32` (and grouped
/// `gemv_bf16_xf32_x4`) at local `K/4`, byte for byte.
fn quarter_gate(
    gpu: &mut Gpu,
    fx: &HcFixture,
    opt: &Opts,
    tally: &mut Tally,
    notes: &mut Vec<String>,
) -> Result<()> {
    for q in 0..4 {
        let qf = fx.quarter_fixture(gpu, q)?;
        let reference = qf.reference(gpu, tally, notes)?;
        sweep(
            gpu,
            &qf,
            &reference,
            opt.quick,
            false,
            &format!("quarter {q}: "),
            tally,
            notes,
        )?;
    }
    Ok(())
}

fn verify_hc(gpu: &mut Gpu, fx: &HcFixture, opt: &Opts, tally: &mut Tally) -> Result<()> {
    let before = tally.state();
    let mut notes: Vec<String> = Vec::new();
    // Modes whose activation is not saturating carry the replay/control passes:
    // `plain` (a raw dot) and the `.25` HC activation. The negative controls use
    // `plain` only (a saturated activation could reproduce a different dot).
    const CONTROL_MODES: [usize; 2] = [0, 2];

    // 0: quarter results first (ordinary per-quarter DAG, see `quarter_gate`).
    quarter_gate(gpu, fx, opt, tally, &mut notes)?;

    // 1: the fixture's own X, every mode.
    let mut references: Vec<Vec<u8>> = Vec::new();
    for mode in &HC_MODES {
        references.push(fx.reference(gpu, mode)?);
    }
    for (i, (mode, reference)) in HC_MODES.iter().zip(&references).enumerate() {
        sweep_hc(
            gpu,
            fx,
            reference,
            mode,
            opt.quick,
            opt.launcher && CONTROL_MODES.contains(&i),
            "",
            tally,
            &mut notes,
        )?;
    }

    // 2a: rows reversed through the same buffers.
    let k = fx.k;
    let reversed: Vec<f32> = (0..MAXR)
        .rev()
        .flat_map(|r| fx.x_host[r * k..(r + 1) * k].to_vec())
        .collect();
    fx.set_x(gpu, &reversed)?;
    for i in CONTROL_MODES {
        let reference = fx.reference(gpu, &HC_MODES[i])?;
        sweep_hc(
            gpu,
            fx,
            &reference,
            &HC_MODES[i],
            true,
            false,
            "reversed X: ",
            tally,
            &mut notes,
        )?;
    }

    // 2b: a fresh X in the same allocation; the stale reference must now differ.
    let mut rng = Rng(0x00ba_d5ee_d00d_f00d ^ ((fx.m as u64) << 8) ^ ((fx.k as u64) << 24));
    let changed = gen_x(k, &mut rng);
    fx.set_x(gpu, &changed)?;
    for i in CONTROL_MODES {
        let reference = fx.reference(gpu, &HC_MODES[i])?;
        sweep_hc(
            gpu,
            fx,
            &reference,
            &HC_MODES[i],
            true,
            false,
            "changed X: ",
            tally,
            &mut notes,
        )?;
    }
    fx.out.reset(gpu)?;
    fx.run_direct(gpu, 32, MAXR, &HC_MODES[0], false)?;
    let (stale_bad, _) = fx.check(gpu, &references[0], MAXR)?;
    tally.launches += 1;
    if stale_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: changed X reproduced the stale reference".to_string(),
        );
    }

    // 3: different weights must not reproduce the original-weights reference.
    fx.set_x(gpu, &fx.x_host)?;
    fx.out.reset(gpu)?;
    fx.run_direct(gpu, 32, MAXR, &HC_MODES[0], true)?;
    let (swapped_bad, _) = fx.check(gpu, &references[0], MAXR)?;
    tally.launches += 1;
    if swapped_bad == 0 {
        tally.negative_control_blind += 1;
        note(
            &mut notes,
            "negative control blind: swapped weights reproduced the original reference".to_string(),
        );
    }

    tally.cases += 1;
    let status = if before == tally.state() {
        "ok"
    } else {
        "MISMATCH"
    };
    println!(
        "{:<56} {status}{}",
        fx.name,
        if notes.is_empty() {
            String::new()
        } else {
            format!(" [{}]", notes.join("; "))
        }
    );
    Ok(())
}

/// `(M, K, wide-range weights)`: M tails (odd M leaves half a workgroup), small
/// and production K (`K % 32 == 0`); `--hc` adds the real shapes: the M = 4 gates,
/// the M = 320 low-rank HC down and the M = 2560 down at `K = 10240`.
fn hc_specs(hc: bool) -> Vec<(usize, usize, bool)> {
    let mut v = vec![
        (1, 32, false),
        (2, 32, false),
        (3, 96, false),
        (5, 288, false),
        (7, 2560, false),
        (8, 32, false),
        (9, 1056, false),
        (15, 4096, false),
        (17, 10240, false),
        (33, 2560, false),
        (7, 2560, true),
        (9, 1056, true),
    ];
    if hc {
        v.extend([
            (4, 10240, false),
            (320, 10240, false),
            (2560, 10240, false),
            (4, 10240, true),
        ]);
    }
    v
}

fn hc_fixtures(gpu: &mut Gpu, opt: &Opts) -> Result<Vec<HcFixture>> {
    let mut seed = 0x6263_3136_6863_6b34u64;
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    let mut out = Vec::new();
    for (m, k, wide) in hc_specs(opt.hc) {
        out.push(HcFixture::new(gpu, m, k, wide, &mut next_rng())?);
    }
    Ok(out)
}

/// M slot patterns (zero slots, odd and tail sizes).
const PATTERNS: [[usize; 4]; 11] = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 7, 0],
    [7, 8, 9, 0],
    [8, 9, 0, 1],
    [9, 8, 7, 3],
    [1, 2, 3, 7],
    [0, 0, 0, 9],
    [16, 17, 31, 0],
    [8, 8, 8, 8],
    [9, 1, 0, 15],
];

/// `(K, pattern index, wide-range weights)`: scalar `K % 8 != 0`, partial-256
/// multiples of eight, then production `K = 2560` over every pattern.
fn fixture_specs() -> Vec<(usize, usize, bool)> {
    let mut v: Vec<(usize, usize, bool)> = vec![
        // scalar domain
        (1, 3, false),
        (3, 6, false),
        (7, 0, false),
        (7, 5, false),
        (9, 8, false),
        (31, 2, false),
        (33, 4, false),
        (255, 9, false),
        (2559, 10, false),
        // K % 8 == 0, partial 256
        (8, 5, false),
        (8, 7, false),
        (16, 6, false),
        (24, 3, false),
        (264, 1, false),
        (264, 10, false),
        (520, 8, false),
        (1000, 4, false),
        (2056, 2, false),
        (2056, 9, false),
    ];
    // production K
    v.extend((0..PATTERNS.len()).map(|p| (2560, p, false)));
    // wide-range weights
    v.extend([(2560, 5, true), (1000, 6, true), (7, 3, true)]);
    v
}

fn fixtures(gpu: &mut Gpu, opt: &Opts) -> Result<Vec<Fixture>> {
    let mut seed = 0x6266_3136_7077_6964u64;
    let mut next_rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        Rng(seed | 1)
    };
    let mut out = Vec::new();
    for (k, p, wide) in fixture_specs() {
        out.push(Fixture::new(gpu, PATTERNS[p], k, wide, &mut next_rng())?);
    }
    if opt.router512 {
        // Router (512 experts) with the selector and shared gate/up rows at K = 2560.
        for ms in [[512, 1, 0, 0], [512, 1, 1024, 1024], [512, 1, 513, 511]] {
            out.push(Fixture::new(gpu, ms, 2560, false, &mut next_rng())?);
        }
    }
    Ok(out)
}

fn run() -> Result<bool> {
    let mut opt = Opts {
        quick: false,
        router512: false,
        launcher: false,
        hc: false,
    };
    for flag in std::env::args().skip(1) {
        match flag.as_str() {
            "--quick" => opt.quick = true,
            "--router512" => opt.router512 = true,
            "--launcher" => opt.launcher = true,
            "--hc" => opt.hc = true,
            other => {
                return Err(format!(
                    "unknown argument {other} (usage: bf16_rows_pm_oracle [--quick] [--router512] [--hc] [--launcher])"
                ))
            }
        }
    }
    let mut gpu = Gpu::init().map_err(err)?;
    gpu.lane_rows_scope = false;
    println!(
        "arch {} (few_row_limit {})",
        gpu.arch.as_str(),
        gpu.few_row_limit()
    );
    if gpu.arch.as_str() != "gfx1151" {
        return Err(format!(
            "exact gfx1151 only, this device is {}",
            gpu.arch.as_str()
        ));
    }
    if opt.launcher {
        gpu.set_wide_verify_rows(MAXR);
        // The launcher assertions are the point of `--launcher`: fail closed when
        // the wide twin switch is off instead of silently asserting nothing.
        if gpu.wide_twin_cap(WideTwinFamily::Bf16, 9).is_none() {
            return Err(format!(
                "--launcher needs {}=1 (the BF16 wide twin switch)",
                WideTwinFamily::Bf16.env_var()
            ));
        }
    }
    for (_, symbol) in SYMBOLS {
        load_pm(&mut gpu, symbol)?;
        let resident = gpu
            .occupancy_max_active_blocks(symbol, [256, 1, 1], 0)
            .map_err(err)?;
        println!(
            "{symbol}: occupancy API reports {resident} resident 256-thread workgroups per CU"
        );
    }
    for (_, symbol) in HC_SYMBOLS {
        load_pm(&mut gpu, symbol)?;
        let resident = gpu
            .occupancy_max_active_blocks(symbol, [256, 1, 1], 0)
            .map_err(err)?;
        println!(
            "{symbol}: occupancy API reports {resident} resident 256-thread workgroups per CU"
        );
    }
    let mut tally = Tally::default();
    for fx in fixtures(&mut gpu, &opt)? {
        verify(&mut gpu, &fx, &opt, &mut tally)?;
    }
    for fx in hc_fixtures(&mut gpu, &opt)? {
        verify_hc(&mut gpu, &fx, &opt, &mut tally)?;
    }
    println!(
        "{} cases, {} launches: {} bad bytes vs actual decode, {} reference disagreement bytes, {} wrong launcher symbols, {} blind negative controls",
        tally.cases, tally.launches, tally.bad_bytes, tally.ref_disagree, tally.bad_symbols, tally.negative_control_blind
    );
    Ok(!tally.failed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("bf16_rows_pm_oracle: byte mismatch");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("bf16_rows_pm_oracle: {e}");
            std::process::exit(2);
        }
    }
}
