// SPDX-License-Identifier: Apache-2.0
//! LAB-only GPU byte oracle for the A3B native-fp8 FA2-GQA PeaceMaker twin
//! (`fa_prefill_fp8_pm_gfx1201`, H16/KV2/D256, 516-byte E4M3FN K/V token rows). No production
//! dispatch. Separate from the frozen Q8_0 `fa_pm_oracle` (pinned lab source/object untouched).
//!
//! ```text
//! CPU only (no HIP init):
//!   fa_pm_fp8_oracle --plan          cell list, launch configs, allocation census
//!   fa_pm_fp8_oracle --self-check    CPU invariants (ABI, E4M3FN model, cache layout, fixtures)
//! GPU (hold only):
//!   fa_pm_fp8_oracle                 byte oracle (all cells, two position rounds)
//!   fa_pm_fp8_oracle --filter STR    only cells whose name contains STR
//! ```
//!
//! Arms (both read the SAME device cache bytes; Runtime cells write them with the production
//! `kv_cache_write_fp8_e4m3_batched`, Synthetic cells upload host-built rows):
//!
//! * HIP twin: the existing production `attention_q8_0_fa2_gqa.gfx1201.hip` compiled by the repo
//!   JIT with `-DHIPFIRE_FA2_KMODE=8 -DHIPFIRE_FA2_Q0_A3B` (no FA2_FP8): the H16/KV2 f16 Q0
//!   body, symbols `attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201` +
//!   `attention_fp8_e4m3_fa2_gqa_f16_a3b_gfx1201`, block 160, 65,536 B LDS.
//! * PM: the shared Q8-module preconvert (`fa_prefill_pm_preconvert`) + the fp8 direct
//!   (`fa_prefill_pm_direct_codec(Fp8E4M3)`), plus (rows >= 64, one segment) the production wrapper
//!   `attention_fp8_e4m3_fa_prefill_pm_gfx1201`.
//!
//! Gates per cell: Q16 bytes (HIP pre vs PM pre), attention bytes (HIP vs PM), wrapper == PM
//! primitive, inputs unchanged, all guards intact, finite non-trivial output, future-cache poison
//! probes (rows at or below a probe horizon byte-stable when every later key is zeroed or made
//! extreme-finite). NaN-code cells (E4M3FN `0x7f`/`0xff`) are classified separately: NaN positions
//! must agree and every other byte must be identical. `all_byte_identical` covers only finite cells;
//! `all_classifications_pass` includes the NaN cells. A sampled f64 CPU reference over f32 Q and
//! independently decoded, f16-staged K/V omits Q16 conversion and the WMMA/reduction DAG: diagnostic only.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use hip_bridge::KernargBlob;
use rdna_compute::fa_prefill_pm::{self, FaPrefillKvCodec};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, ffi::c_void};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
const ROW: usize = H * D;
/// Bytes of one native-fp8 cache position: 2 heads x 256 codes + 2 f16 scales.
const POS_BYTES: usize = KV * (D + 2);
const SCALE_OFF: usize = KV * D;
const MAX_ROWS: usize = 8192;
const SCALE: f32 = 1.0 / 16.0;
const GUARD: usize = 4096;
const POISON: i32 = 0xFF;
const PRE_BLOCK: u32 = 128;
const DIRECT_BLOCK: u32 = 160;
const DIRECT_LDS: u32 = 65_536;

const HIP_BODY: &str = include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip");
/// Distinct source string (the JIT/module cache keys on it): the H16/KV2 Q0 specialization.
fn hip_source() -> String {
    format!("#define HIPFIRE_FA2_KMODE 8\n#define HIPFIRE_FA2_Q0_A3B 1\n{HIP_BODY}")
}
const HIP_PRE: &str = "attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201";
const HIP_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_f16_a3b_gfx1201";

fn sha_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn bytes<T>(x: &[T]) -> &[u8] {
    // SAFETY: only initialized f32/i32/u8 slices are passed; no padding.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
}

// ------------------------------------------------------- independent E4M3FN / cache model
/// OCP E4M3FN: bias 7, `E = 0` subnormal `M * 2^-9`, `E = 15 && M = 7` NaN (`None`), `0x80` = -0.
fn e4m3fn(code: u8) -> Option<f32> {
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((code >> 3) & 0xf);
    let m = f32::from(code & 7);
    if e == 15 && code & 7 == 7 {
        return None;
    }
    Some(
        sign * if e == 0 {
            m * 2f32.powi(-9)
        } else {
            (1.0 + m / 8.0) * 2f32.powi(e - 7)
        },
    )
}
fn finite_codes() -> Vec<u8> {
    (0..=255u8).filter(|&c| e4m3fn(c).is_some()).collect()
}
fn f16v(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
/// The staged f16 element value (f32 product of the f16 scale and the decoded code, one RNE to f16).
fn staged(cache: &[u8], pos: usize, head: usize, d: usize) -> f32 {
    let at = pos * POS_BYTES + SCALE_OFF + 2 * head;
    let scale = f16v(u16::from_le_bytes([cache[at], cache[at + 1]]));
    match e4m3fn(cache[pos * POS_BYTES + head * D + d]) {
        Some(v) => half::f16::from_f32(scale * v).to_f32(),
        None => f32::NAN,
    }
}
/// Independent f32 -> f16 round-to-nearest-even, copied faithfully from the independent CPU test
/// (`crates/peacemaker-isa/tests/fa_prefill_fp8.rs:44-77`), not from emitted instructions. Finite
/// inputs; overflow rounds to infinity. Bit-level: signed zeros preserved, subnormal RNE with
/// tie-to-even, normal RNE on the low 13 bits with tie-to-even.
fn ieee_f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let exp = e - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = u64::from(m | 0x80_0000);
        let shift = (14 - exp) as u32;
        let mut q = mant >> shift;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut h = ((exp as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}
/// Independent f16 bit decode (subnormal normalization, signed zeros, NaN payload preserved).
fn f16_to_f32_ieee(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let e = u32::from((h >> 10) & 0x1f);
    let m = u32::from(h & 0x3ff);
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            let (mut e2, mut m2) = (113u32, m);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
/// The independent staged f16 element word: decode E4M3FN, multiply by the decoded f16 scale in
/// f32, round once to f16. `None` for a NaN code. Overflow is determined from the rounded f16
/// exponent (as the CPU test does), not from a conservative `abs(product) > 65504` predicate.
fn ieee_stage(cache: &[u8], pos: usize, head: usize, d: usize) -> Option<u16> {
    let at = pos * POS_BYTES + SCALE_OFF + 2 * head;
    let scale = u16::from_le_bytes([cache[at], cache[at + 1]]);
    e4m3fn(cache[pos * POS_BYTES + head * D + d]).map(|v| ieee_f16_rne(f16_to_f32_ieee(scale) * v))
}
/// Single-code staging helper: `None` for NaN codes, `Some(word)` otherwise (possibly infinite
/// when the rounded exponent overflows, which the admission below detects).
fn ieee_stage_one(code: u8, scale: u16) -> Option<u16> {
    e4m3fn(code).map(|v| ieee_f16_rne(f16_to_f32_ieee(scale) * v))
}
/// A finite code admitted at `scale`: the rounded word is finite. Products rounding nonfinite are
/// replaced by the code of 1.0 (`0x38`); every admitted code is independently censused by the
/// frozen-case constructors and the self-check below.
fn ieee_safe_code(scale: u16, code: u8) -> u8 {
    match ieee_stage_one(code, scale) {
        Some(w) if w & 0x7c00 != 0x7c00 => code,
        _ => 0x38,
    }
}
/// Scale boundaries of the f16 staging: smallest subnormal, largest subnormal, smallest normal, 1.0,
/// 1 + ulp, f16 max, two mid-range scales.
const SCALES: [u16; 8] = [
    0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000,
];
/// A finite code whose f16 staged product with `scale` cannot overflow (replaced by 1.0 otherwise).
fn safe_code(scale: u16, code: u8) -> u8 {
    if (f16v(scale) * e4m3fn(code).unwrap()).abs() > 65504.0 {
        0x38
    } else {
        code
    }
}
/// Host-built `[ctx]` rows: every finite code at every scale boundary on both heads (K and V use different
/// rotations), both heads with distinct scales. `nan` injects `0x7f` / `0xff` codes.
fn grid_cache(ctx: usize, salt: usize, nan: bool) -> Vec<u8> {
    let table = finite_codes();
    let mut out = vec![0u8; ctx * POS_BYTES];
    for p in 0..ctx {
        for g in 0..KV {
            let scale = SCALES[(p + 3 * g + salt) % SCALES.len()];
            out[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&scale.to_le_bytes());
            for d in 0..D {
                out[p * POS_BYTES + g * D + d] =
                    safe_code(scale, table[(d + 7 * p + 31 * g + 13 * salt) % table.len()]);
            }
        }
    }
    if nan {
        // A few NaN codes (positions >= 3 so every causal row >= 3 reads them): both heads, both NaN encodings.
        for (p, g, d, code) in [
            (3usize, 0usize, 5usize, 0x7fu8),
            (4, 1, 200, 0xff),
            (5, 0, 0, 0xff),
            (6, 1, 255, 0x7f),
        ] {
            if p < ctx {
                out[p * POS_BYTES + g * D + d] = code;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- fixtures
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fixture {
    Std,
    Reverse8,
    DupPairs8,
    Rotate8,
    SameGroup8,
}
impl Fixture {
    fn name(self) -> &'static str {
        match self {
            Fixture::Std => "std",
            Fixture::Reverse8 => "reverse8",
            Fixture::DupPairs8 => "dup_pairs8",
            Fixture::Rotate8 => "rotate8",
            Fixture::SameGroup8 => "same_group8",
        }
    }
    fn offset(self, r: usize, g: usize, m: usize) -> usize {
        match self {
            Fixture::Std => r,
            Fixture::Reverse8 => m - 1 - r,
            Fixture::DupPairs8 => (r & !1) % m,
            Fixture::Rotate8 => (r * 5 + g) % 8 % m,
            Fixture::SameGroup8 => m - 1,
        }
    }
    fn positions(self, rows: usize, start: usize) -> Vec<i32> {
        (0..rows)
            .map(|row| {
                let (g, r) = (row / 8, row % 8);
                let m = (rows - g * 8).min(8);
                (start + g * 8 + self.offset(r, g, m)) as i32
            })
            .collect()
    }
    fn for_round(self, round: usize) -> Fixture {
        match (round, self) {
            (0, f) => f,
            (_, Fixture::Std) => Fixture::Reverse8,
            _ => Fixture::Std,
        }
    }
}

// ------------------------------------------------------------------ cells
#[derive(Clone, Copy, PartialEq, Debug)]
enum Source {
    /// Random f32 K/V through the production `kv_cache_write_fp8_e4m3_batched`.
    Runtime,
    /// Host-built rows: every finite code x every f16 scale boundary, distinct per-head scales.
    Grid,
    /// `Grid` plus NaN codes (separate classification, not the finite byte gate).
    Nan,
}
impl Source {
    fn name(self) -> &'static str {
        match self {
            Source::Runtime => "runtime",
            Source::Grid => "grid",
            Source::Nan => "nan",
        }
    }
}
#[derive(Clone, Copy)]
struct Cell {
    rows: usize,
    start: usize,
    segment: usize,
    fixture: Fixture,
    source: Source,
}
impl Cell {
    fn ctx(&self) -> usize {
        self.start + self.rows
    }
    fn name(&self) -> String {
        format!(
            "n{}_s{}_seg{}_{}_{}",
            self.rows,
            self.start,
            self.segment,
            self.fixture.name(),
            self.source.name()
        )
    }
    fn segments(&self) -> Vec<(usize, usize)> {
        (0..self.rows)
            .step_by(self.segment)
            .map(|off| (off, (self.rows - off).min(self.segment)))
            .collect()
    }
    /// The production wrapper admits one whole 64..=8192-row segment.
    fn wrapper(&self) -> bool {
        self.segment == self.rows
            && (fa_prefill_pm::MIN_ROWS..=fa_prefill_pm::MAX_ROWS).contains(&self.rows)
    }
    /// Future-cache poison probes and the sampled CPU reference need standard positions.
    fn diagnostics(&self) -> bool {
        self.fixture == Fixture::Std
            && self.source == Source::Runtime
            && self.start == 0
            && self.segment == self.rows
    }
}
// -------------------------------------------- independent native semantic proof (section N)
/// Independent case descriptor: host-built Q16/K/V inputs plus the independently computed
/// expected output words. The original `Source`, names, seeds and ordinary matrix are unchanged;
/// these cases run before the large ordinary arena is constructed.
#[derive(Clone)]
struct IeeeCase {
    name: String,
    qh: Vec<u16>,
    positions: Vec<i32>,
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    expected: IeeeExpected,
}
/// Independent expected output: finite words, or V-NaN / K-NaN with the clean (NaN-free) finite
/// reference retained so unaffected bytes are checked separately from affected NaN positions.
#[derive(Clone)]
enum IeeeExpected {
    Finite(Vec<u32>),
    VNaN {
        clean_expected_words: Vec<u32>,
        kv_head: usize,
        dimension: usize,
    },
    KNaN {
        clean_expected_words: Vec<u32>,
        kv_head: usize,
    },
}
impl IeeeExpected {
    fn kind(&self) -> &'static str {
        match self {
            IeeeExpected::Finite(_) => "finite_byte_gate",
            IeeeExpected::VNaN { .. } => "v_nan_separate",
            IeeeExpected::KNaN { .. } => "k_nan_separate",
        }
    }
    #[allow(dead_code)]
    fn words(&self) -> &[u32] {
        match self {
            IeeeExpected::Finite(w) => w,
            IeeeExpected::VNaN {
                clean_expected_words,
                ..
            } => clean_expected_words,
            IeeeExpected::KNaN {
                clean_expected_words,
                ..
            } => clean_expected_words,
        }
    }
}
/// Finite codes whose value is a multiple of 1/8 (or zero): V operands whose tied sums are exact
/// in f32 for any tie count. Copied from the CPU test (`exact_v_codes`).
fn ieee_exact_v_codes() -> Vec<u8> {
    finite_codes()
        .into_iter()
        .filter(|&c| {
            let v = e4m3fn(c).unwrap();
            v == 0.0 || (v.abs() >= 0.125 && (v * 8.0).fract() == 0.0)
        })
        .collect()
}
/// Independent `winner_case` arithmetic (CPU test `:249-274`): Q16 sign pattern per row/head,
/// K code plane with opposite head signs and opposing scale ramps, V exact-sum codes at 1.0.
/// Returns `(qh, k_cache, v_cache)` with `npos = max(positions) + 1` allocated contexts.
fn ieee_winner_inputs(
    batch: usize,
    positions: &[i32],
    qmag: f32,
    salt: usize,
) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    let npos = *positions.iter().max().unwrap() as usize + 1;
    let qh = (0..batch * ROW)
        .map(|i| {
            let (r, h, d) = (i / ROW, (i / D) % H, i % D);
            let sign = if (r * 5 + h * 3 + salt) % 4 < 2 {
                1.0
            } else {
                -1.0
            };
            ieee_f16_rne(sign * qmag * (1 + d % 3) as f32)
        })
        .collect();
    let mut k = vec![0u8; npos * POS_BYTES];
    for p in 0..npos {
        for g in 0..KV {
            let scale = ieee_f16_rne(if g == 0 {
                0.25 * (1 + p) as f32
            } else {
                0.25 * (200 - p) as f32
            });
            k[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&scale.to_le_bytes());
            for d in 0..D {
                let code = match d % 61 {
                    0 => 0x7e,
                    30 => 0xfe,
                    _ => 0x68 + ((d * 37 + salt) % 4) as u8,
                };
                k[p * POS_BYTES + g * D + d] = code ^ if g == 0 { 0 } else { 0x80 };
            }
        }
    }
    let vcodes = ieee_exact_v_codes();
    let mut v = vec![0u8; npos * POS_BYTES];
    for p in 0..npos {
        for g in 0..KV {
            v[p * POS_BYTES + SCALE_OFF + 2 * g..p * POS_BYTES + SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
            for d in 0..D {
                v[p * POS_BYTES + g * D + d] =
                    vcodes[(p * 29 + d * 7 + g * 61 + salt * 13) % vcodes.len()];
            }
        }
    }
    (qh, k, v)
}
/// Single-live-key caches at `scales` (CPU test `single_key_case`): every finite code admitted by
/// the rounded-exponent overflow rule, rotated by `shift` per head. Returns `(k, v)` single-row
/// caches; the caller reuses them for both K and V single-key sweeps.
fn single_key_caches(scales: [u16; 2], shift: usize) -> (Vec<u8>, Vec<u8>) {
    let table = finite_codes();
    let mut k = vec![0u8; POS_BYTES];
    let mut v = vec![0u8; POS_BYTES];
    for g in 0..KV {
        k[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2].copy_from_slice(&scales[g].to_le_bytes());
        v[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2].copy_from_slice(&scales[g].to_le_bytes());
        for d in 0..D {
            k[g * D + d] =
                ieee_safe_code(scales[g], table[(d + 31 * shift + 128 * g) % table.len()]);
            v[g * D + d] =
                ieee_safe_code(scales[g], table[(d + 31 * shift + 128 * g) % table.len()]);
        }
    }
    (k, v)
}
/// Independent expected output words (CPU test `expected`, `:278-316`): independently staged K/V,
/// f64 dots, f32 score conversion, exact-max ties, minimum 200 gap for nonwinners,
/// exact-representability assertion for tied V sums, ascending accumulation from +0.0, final f32
/// reciprocal multiplication. Single-key output is therefore NOT a V-bit echo: a staged -0
/// normally becomes output +0. NaN codes propagate: a NaN staged K score (or V element) yields
/// NaN outputs in the affected head lanes.
fn ieee_expected(ic: &IeeeCase) -> Result<Vec<u32>> {
    let npos_k = ic.k_cache.len() / POS_BYTES;
    let npos_v = ic.v_cache.len() / POS_BYTES;
    if npos_k == 0 || npos_v == 0 || npos_k != npos_v {
        return Err("independent caches must be non-empty and K/V matched".into());
    }
    let npos = npos_k;
    let batch = ic.positions.len();
    // K staging keeps NaN lanes: a NaN staged K element makes that key's f64 dot NaN, the f32
    // score NaN, the row max NaN, and every output of the affected KV head NaN (the CPU test's
    // K-NaN assertion). Non-NaN lanes stay exact.
    let kst: Vec<Vec<f32>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| match ieee_stage(&ic.k_cache, i % npos, i / npos, d) {
                    Some(w) => f16_to_f32_ieee(w),
                    None => f32::NAN,
                })
                .collect()
        })
        .collect();
    // V staging keeps NaN lanes (V NaN cases): `None` word 0x7e00 is a placeholder; NaN outputs
    // are decided per element below from the raw codes so the clean reference stays exact.
    let v_is_nan: Vec<Vec<bool>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| e4m3fn(ic.v_cache[i % npos * POS_BYTES + (i / npos) * D + d]).is_none())
                .collect()
        })
        .collect();
    let vst: Vec<Vec<f32>> = (0..KV * npos)
        .map(|i| {
            (0..D)
                .map(|d| match ieee_stage(&ic.v_cache, i % npos, i / npos, d) {
                    Some(w) => f16_to_f32_ieee(w),
                    None => f32::NAN,
                })
                .collect()
        })
        .collect();
    let mut out = vec![0u32; batch * ROW];
    for r in 0..batch {
        let pos = ic.positions[r] as usize;
        if pos >= npos {
            return Err(
                format!("independent position {pos} exceeds allocated context {npos}").into(),
            );
        }
        for h in 0..H {
            let g = h / 8;
            let q: Vec<f64> = (0..D)
                .map(|d| f64::from(f16_to_f32_ieee(ic.qh[(r * H + h) * D + d])))
                .collect();
            let scores: Vec<f32> = (0..=pos)
                .map(|j| {
                    let dot: f64 = (0..D).map(|d| q[d] * f64::from(kst[g * npos + j][d])).sum();
                    (dot as f32) * ic.qh_scale()
                })
                .collect();
            // A NaN staged K element makes the dot NaN (`f32::max` would hide it in the fold).
            let nan_score = scores.iter().any(|s| s.is_nan());
            let m = if nan_score {
                f32::NAN
            } else {
                scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };
            let ties: Vec<usize> = (0..=pos).filter(|&j| nan_score || scores[j] == m).collect();
            for j in 0..=pos {
                if !nan_score {
                    assert!(
                        scores[j] == m || m - scores[j] >= 200.0,
                        "ill-posed independent case: row {r} head {h} key {j} gap {}",
                        m - scores[j]
                    );
                }
            }
            let inv = 1.0f32 / ties.len().max(1) as f32;
            for d in 0..D {
                let tied_nan = !nan_score && ties.iter().any(|&j| v_is_nan[g * npos + j][d]);
                if nan_score || tied_nan {
                    out[(r * H + h) * D + d] = f32::NAN.to_bits();
                    continue;
                }
                let (mut sum, mut exact) = (0.0f32, 0.0f64);
                for &j in &ties {
                    let v = vst[g * npos + j][d];
                    sum += v;
                    exact += f64::from(v);
                }
                assert_eq!(
                    f64::from(sum),
                    exact,
                    "inexact V sum: row {r} head {h} dim {d}"
                );
                out[(r * H + h) * D + d] = (sum * inv).to_bits();
            }
        }
    }
    Ok(out)
}
impl IeeeCase {
    fn qh_scale(&self) -> f32 {
        // All frozen independent cases use attention scale 1.0 (stored out-of-band would be a
        // second convention; the CPU test passes it per case and every frozen N case uses 1.0).
        1.0
    }
}
/// Frozen independent cases in contract order (CPU test `:489-511,533-555,660-706`):
/// K NaN `0x7f` then `0xff` (single live key, KV head 1, dim 17); empty-masked-subtile positions
/// `[64,0,65,32,15,16,31,48]`; V NaN tuples `(0x7f,0,5)`, `(0xff,1,200)`, `(0x7f,1,0)`,
/// `(0xff,0,255)`; finite scale sweep over the 12 f16 boundary scales (each head a different
/// scale) plus the swapped-head-scale negative control and the swapped-scale-order pair.
/// One-live-key means causal position 0 with finite cache rows allocated only as needed
/// (single-row caches here); eight query rows do not require an eight-key horizon.
fn ieee_cases() -> Result<Vec<IeeeCase>> {
    let mut v = Vec::new();
    // 1. K NaN, cheap first: single live key, KV head 1 / dim 17, `0x7f` then `0xff`.
    for code in [0x7fu8, 0xff] {
        let (qh, mut k, vc) = ieee_winner_inputs(8, &[0; 8], 1.0, 9);
        let clean = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        k[D + 17] = code;
        v.push(IeeeCase {
            name: format!("ieee_k_nan_{code:#04x}_kvhead1_dim17"),
            qh,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::KNaN {
                clean_expected_words: clean,
                kv_head: 1,
            },
        });
    }
    // 2. Empty-masked-subtile case: low-horizon row keeps its finite answer while later subtiles
    // are causally masked (not an invalid empty-batch launch).
    {
        let positions = vec![64, 0, 65, 32, 15, 16, 31, 48];
        let (qh, k, vc) = ieee_winner_inputs(8, &positions, 1.0, 0);
        let words = ieee_expected(&IeeeCase {
            name: "ieee_empty_subtile".into(),
            qh: qh.clone(),
            positions: positions.clone(),
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        v.push(IeeeCase {
            name: "ieee_empty_subtile".into(),
            qh,
            positions,
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::Finite(words),
        });
    }
    // 3. V NaN tuples from the CPU test.
    for (code, g, d0) in [
        (0x7fu8, 0usize, 5usize),
        (0xff, 1, 200),
        (0x7f, 1, 0),
        (0xff, 0, 255),
    ] {
        let (qh, k, mut vc) = ieee_winner_inputs(8, &[0; 8], 1.0, 7);
        let clean = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        vc[g * D + d0] = code;
        v.push(IeeeCase {
            name: format!("ieee_v_nan_{code:#04x}_kvhead{g}_dim{d0}"),
            qh,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::VNaN {
                clean_expected_words: clean,
                kv_head: g,
                dimension: d0,
            },
        });
    }
    // 4. Finite scale sweep over the 12 f16 boundary scales; each head a different scale.
    const IEEE_SWEEP: [u16; 12] = [
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000, 0xbc01,
        0xfbff,
    ];
    for shift in 0..IEEE_SWEEP.len() {
        let scales = [
            IEEE_SWEEP[shift],
            IEEE_SWEEP[(shift + 3) % IEEE_SWEEP.len()],
        ];
        let (k, vc) = single_key_caches(scales, shift);
        let (qh_base, _, _) = ieee_winner_inputs(8, &[0; 8], 1.0, 3 + shift);
        let words = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh_base.clone(),
            positions: vec![0; 8],
            k_cache: k.clone(),
            v_cache: vc.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        v.push(IeeeCase {
            name: format!("ieee_finite_scale_{scales:#06x?}_shift{shift}"),
            qh: qh_base,
            positions: vec![0; 8],
            k_cache: k,
            v_cache: vc,
            expected: IeeeExpected::Finite(words),
        });
    }
    // Swapped-head-scale negative control: outputs must follow the head, so the swapped pair differs.
    {
        let (ka, va) = single_key_caches([0x3c00, 0x4000], 1);
        let (kb, vb) = single_key_caches([0x4000, 0x3c00], 1);
        let (qh, _, _) = ieee_winner_inputs(8, &[0; 8], 1.0, 4);
        let wa = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: ka.clone(),
            v_cache: va.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        let wb = ieee_expected(&IeeeCase {
            name: String::new(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: kb.clone(),
            v_cache: vb.clone(),
            expected: IeeeExpected::Finite(vec![]),
        })?;
        assert!(
            wa != wb,
            "swapped head scales must change the independent expectation"
        );
        v.push(IeeeCase {
            name: "ieee_head_scales_a".into(),
            qh: qh.clone(),
            positions: vec![0; 8],
            k_cache: ka,
            v_cache: va,
            expected: IeeeExpected::Finite(wa),
        });
        v.push(IeeeCase {
            name: "ieee_head_scales_b".into(),
            qh,
            positions: vec![0; 8],
            k_cache: kb,
            v_cache: vb,
            expected: IeeeExpected::Finite(wb),
        });
    }
    Ok(v)
}
/// Runtime-written long-context cells (rows, start, segment): context exactly 16384, uploaded
/// positions include 16383, whole + segmented equality, wrapper admission where applicable, two
/// changed-position rounds and both production K/V writes (executed by the ordinary path).
const LONG_CTX: usize = 16384;
const LONG_CELLS: [(usize, usize, usize); 3] = [
    (512, LONG_CTX - 512, 512),
    (8192, 8192, 8192),
    (8192, 8192, 512),
];
fn long_cells() -> Vec<Cell> {
    LONG_CELLS
        .iter()
        .map(|&(rows, start, segment)| Cell {
            rows,
            start,
            segment,
            fixture: Fixture::Std,
            source: Source::Runtime,
        })
        .collect()
}
const A0_SHAPES: [(usize, usize); 5] = [(512, 0), (4096, 0), (8192, 0), (512, 3584), (4096, 4096)];
const BOUNDARY_SHAPES: [(usize, usize); 3] = [(64, 0), (65, 0), (8191, 0)];
const UNALIGNED_SHAPES: [(usize, usize); 2] = [(512, 1), (512, 63)];
const FIXTURE_SHAPES: [(usize, usize); 4] = [(512, 0), (512, 3584), (65, 63), (4096, 0)];
const FIXTURES: [Fixture; 4] = [
    Fixture::Reverse8,
    Fixture::DupPairs8,
    Fixture::Rotate8,
    Fixture::SameGroup8,
];
/// Exact primitive tails below the production minimum (1..=63 rows).
const TAIL_ROWS: [usize; 12] = [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63];
const TAIL_STARTS: [usize; 2] = [0, 17];
const GRID_SHAPES: [(usize, usize); 4] = [(64, 0), (65, 63), (512, 0), (512, 3584)];

fn push_modes(v: &mut Vec<Cell>, rows: usize, start: usize, fixture: Fixture, source: Source) {
    v.push(Cell {
        rows,
        start,
        segment: rows,
        fixture,
        source,
    });
    if rows > 512 {
        v.push(Cell {
            rows,
            start,
            segment: 512,
            fixture,
            source,
        });
    }
}
fn oracle_cells() -> Vec<Cell> {
    let mut v = Vec::new();
    for (rows, start) in A0_SHAPES
        .into_iter()
        .chain(BOUNDARY_SHAPES)
        .chain(UNALIGNED_SHAPES)
    {
        push_modes(&mut v, rows, start, Fixture::Std, Source::Runtime);
    }
    for (rows, start) in FIXTURE_SHAPES {
        for f in FIXTURES {
            push_modes(&mut v, rows, start, f, Source::Runtime);
        }
    }
    for rows in TAIL_ROWS {
        for start in TAIL_STARTS {
            v.push(Cell {
                rows,
                start,
                segment: rows,
                fixture: Fixture::Std,
                source: Source::Runtime,
            });
        }
    }
    for (rows, start) in GRID_SHAPES {
        push_modes(&mut v, rows, start, Fixture::Std, Source::Grid);
    }
    v.push(Cell {
        rows: 64,
        start: 0,
        segment: 64,
        fixture: Fixture::Std,
        source: Source::Nan,
    });
    v.push(Cell {
        rows: 65,
        start: 63,
        segment: 65,
        fixture: Fixture::Reverse8,
        source: Source::Nan,
    });
    v
}

// ---------------------------------------------------------- memory census
#[derive(Clone, Copy, Default)]
struct Used {
    q32: usize,
    k32: usize,
    v32: usize,
    allpos: usize,
    kc: usize,
    vc: usize,
    pos: usize,
    q16: usize,
    out: usize,
}
fn used(rows: usize, ctx: usize) -> Used {
    Used {
        q32: rows * ROW * 4,
        k32: ctx * KV * D * 4,
        v32: ctx * KV * D * 4,
        allpos: ctx * 4,
        kc: ctx * POS_BYTES,
        vc: ctx * POS_BYTES,
        pos: rows * 4,
        q16: rows * ROW * 2,
        out: rows * ROW * 4,
    }
}
fn used_of(c: &Cell) -> Used {
    used(c.rows, c.ctx())
}
const BUFFER_NAMES: [&str; 12] = [
    "q32",
    "k32_src",
    "v32_src",
    "allpos",
    "k_cache_fp8",
    "v_cache_fp8",
    "positions",
    "q16_hip",
    "q16_pm",
    "out_hip",
    "out_pm",
    "out_wrapper_or_probe",
];
fn used_list(u: &Used) -> [usize; 12] {
    [
        u.q32, u.k32, u.v32, u.allpos, u.kc, u.vc, u.pos, u.q16, u.q16, u.out, u.out, u.out,
    ]
}
fn arena_sizes(cells: &[Cell]) -> [usize; 12] {
    let mut m = [0usize; 12];
    for c in cells {
        for (slot, b) in m.iter_mut().zip(used_list(&used_of(c))) {
            *slot = (*slot).max(b);
        }
    }
    m
}

// ----------------------------------------------------------- ABI helpers
fn pre_kernargs(q: *mut c_void, q16: *mut c_void, n: usize) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(q);
    b.push_ptr(q16);
    b.push_ptr(std::ptr::null::<c_void>()); // signs1: ignored by the ABI
    b.push_ptr(std::ptr::null::<c_void>()); // signs2: ignored by the ABI
    b.push_i32(n as i32);
    b.push_i32(0); // do_rotate: ignored by the ABI
    b
}
fn direct_kernargs(
    q16: *mut c_void,
    kc: *mut c_void,
    vc: *mut c_void,
    out: *mut c_void,
    pos: *mut c_void,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [q16, kc, vc, out, pos] {
        b.push_ptr(p);
    }
    for x in [H, KV, D, n] {
        b.push_i32(x as i32);
    }
    b.push_f32(SCALE);
    b
}
fn pre_grid(n: usize) -> [u32; 3] {
    [(n * H).div_ceil(4) as u32, 1, 1]
}
fn direct_grid(n: usize) -> [u32; 3] {
    [n.div_ceil(8) as u32, KV as u32, 1]
}

// ------------------------------------------------------------- comparison
#[derive(Clone, Copy)]
enum Elem {
    F32,
    F16,
}
impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 => 2,
        }
    }
    fn at(self, b: &[u8], i: usize) -> (u32, f64) {
        match self {
            Elem::F32 => {
                let bits = u32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
                (bits, f32::from_bits(bits) as f64)
            }
            Elem::F16 => {
                let bits = u16::from_ne_bytes(b[i * 2..i * 2 + 2].try_into().unwrap());
                (bits as u32, half::f16::from_bits(bits).to_f32() as f64)
            }
        }
    }
}
fn coords(idx: usize) -> Value {
    json!({"index":idx,"row":idx / ROW,"head":(idx / D) % H,"dim":idx % D})
}
/// Compare `a` (HIP twin) with `b` (PM) as `elem` arrays of `[row, H, D]`. `nan_class` classifies NaN codes
/// separately: NaN positions must agree and every non-NaN element be bit-identical.
fn compare(a: &[u8], b: &[u8], elem: Elem, nan_class: bool) -> Value {
    if a.len() != b.len() || a.len() % elem.size() != 0 {
        return json!({"pass":false,"byte_identical":false,"error":"length mismatch","len_a":a.len(),"len_b":b.len()});
    }
    let n = a.len() / elem.size();
    let identical = a == b;
    let (mut nf_a, mut nf_b, mut nan_a, mut nan_b) = (0usize, 0usize, 0usize, 0usize);
    let (mut count, mut first) = (0usize, None::<usize>);
    let (mut nan_mismatch, mut value_mismatch) = (0usize, 0usize);
    for i in 0..n {
        let (ba, va) = elem.at(a, i);
        let (bb, vb) = elem.at(b, i);
        nf_a += !va.is_finite() as usize;
        nf_b += !vb.is_finite() as usize;
        nan_a += va.is_nan() as usize;
        nan_b += vb.is_nan() as usize;
        if ba != bb {
            count += 1;
            first.get_or_insert(i);
        }
        if va.is_nan() != vb.is_nan() {
            nan_mismatch += 1;
        } else if !va.is_nan() && ba != bb {
            value_mismatch += 1;
        }
    }
    let finite = nf_a == 0 && nf_b == 0;
    let first_mismatch = first.map(|i| {
        let mut c = coords(i);
        c["hip_bits"] = json!(format!("{:#x}", elem.at(a, i).0));
        c["pm_bits"] = json!(format!("{:#x}", elem.at(b, i).0));
        c
    });
    let pass = if nan_class {
        nan_mismatch == 0 && value_mismatch == 0 && nan_a > 0
    } else {
        identical && finite
    };
    json!({"elements":n,"element_type":match elem {Elem::F32=>"f32",Elem::F16=>"f16"},
        "byte_identical":identical,"mismatched_elements":count,"first_mismatch":first_mismatch,
        "nonfinite_hip":nf_a,"nonfinite_pm":nf_b,"nan_hip":nan_a,"nan_pm":nan_b,
        "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
        "classification":if nan_class {"nan_codes_separate"} else {"finite_byte_gate"},"pass":pass})
}
fn pass(v: &Value) -> bool {
    v["pass"] == true
}

// ------------------------------------------------------------------ rng
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn normal(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
}

// ----------------------------------------------------------- GPU buffers
struct Buf {
    t: GpuTensor,
    es: usize,
    name: &'static str,
}
impl Buf {
    fn new(gpu: &mut Gpu, name: &'static str, bytes: usize, dtype: DType) -> Result<Self> {
        let es = dtype.size();
        assert_eq!(bytes % es, 0);
        let t = gpu.alloc_tensor(&[(bytes + 2 * GUARD) / es], dtype)?;
        Ok(Self { t, es, name })
    }
    fn sub(&self, off: usize, len: usize) -> GpuTensor {
        self.t.sub_offset((GUARD + off) / self.es, len / self.es)
    }
    fn view(&self, len: usize) -> GpuTensor {
        self.sub(0, len)
    }
    fn ptr(&self, off: usize) -> *mut c_void {
        // SAFETY: offset stays inside the allocation (checked by cell sizing).
        unsafe {
            self.t
                .buf
                .as_ptr()
                .cast::<u8>()
                .add(GUARD + off)
                .cast::<c_void>() as *mut c_void
        }
    }
    fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        gpu.hip.memcpy_htod_offset(&self.t.buf, GUARD, data)?;
        Ok(())
    }
    fn download_at(&self, gpu: &Gpu, off: usize, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        gpu.hip.memcpy_dtoh_at(&mut v, &self.t.buf, GUARD + off)?;
        Ok(v)
    }
    fn download(&self, gpu: &Gpu, len: usize) -> Result<Vec<u8>> {
        self.download_at(gpu, 0, len)
    }
    fn poison(&self, gpu: &Gpu, len: usize) -> Result<()> {
        let v = self.view(len);
        gpu.hip.memset(&v.buf, POISON, len)?;
        Ok(())
    }
    fn pattern(&self, side: u8) -> Vec<u8> {
        (0..GUARD)
            .map(|i| (i as u8).wrapping_mul(131) ^ 0xC3 ^ side ^ (self.name.len() as u8))
            .collect()
    }
    fn arm_guards(&self, gpu: &Gpu, used: usize) -> Result<()> {
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, 0, &self.pattern(1))?;
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, GUARD + used, &self.pattern(2))?;
        Ok(())
    }
    fn guards_ok(&self, gpu: &Gpu, used: usize) -> Result<bool> {
        let mut lo = vec![0u8; GUARD];
        let mut hi = vec![0u8; GUARD];
        gpu.hip.memcpy_dtoh_at(&mut lo, &self.t.buf, 0)?;
        gpu.hip.memcpy_dtoh_at(&mut hi, &self.t.buf, GUARD + used)?;
        Ok(lo == self.pattern(1) && hi == self.pattern(2))
    }
}
struct Arena {
    b: Vec<Buf>,
}
impl Arena {
    fn new(gpu: &mut Gpu, cells: &[Cell]) -> Result<Self> {
        let sizes = arena_sizes(cells);
        let dtypes = [
            DType::F32,
            DType::F32,
            DType::F32,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::F32,
            DType::F32,
        ];
        let mut b = Vec::new();
        for i in 0..12 {
            b.push(Buf::new(gpu, BUFFER_NAMES[i], sizes[i], dtypes[i])?);
        }
        Ok(Self { b })
    }
    fn q32(&self) -> &Buf {
        &self.b[0]
    }
    fn k32(&self) -> &Buf {
        &self.b[1]
    }
    fn v32(&self) -> &Buf {
        &self.b[2]
    }
    fn allpos(&self) -> &Buf {
        &self.b[3]
    }
    fn kc(&self) -> &Buf {
        &self.b[4]
    }
    fn vc(&self) -> &Buf {
        &self.b[5]
    }
    fn pos(&self) -> &Buf {
        &self.b[6]
    }
    fn q16_hip(&self) -> &Buf {
        &self.b[7]
    }
    fn q16_pm(&self) -> &Buf {
        &self.b[8]
    }
    fn out_hip(&self) -> &Buf {
        &self.b[9]
    }
    fn out_pm(&self) -> &Buf {
        &self.b[10]
    }
    fn out_extra(&self) -> &Buf {
        &self.b[11]
    }
}

// ----------------------------------------------------------------- arms
fn hip_pre_seg(gpu: &Gpu, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    let mut args = pre_kernargs(a.q32().ptr(off * ROW * 4), q16.ptr(off * ROW * 2), n);
    gpu.launch_kernel_blob(
        HIP_PRE,
        pre_grid(n),
        [PRE_BLOCK, 1, 1],
        0,
        args.as_mut_slice(),
    )?;
    Ok(())
}
fn hip_direct_seg(gpu: &Gpu, a: &Arena, q16: &Buf, out: &Buf, off: usize, n: usize) -> Result<()> {
    let mut args = direct_kernargs(
        q16.ptr(off * ROW * 2),
        a.kc().ptr(0),
        a.vc().ptr(0),
        out.ptr(off * ROW * 4),
        a.pos().ptr(off * 4),
        n,
    );
    gpu.launch_kernel_blob(
        HIP_DIRECT,
        direct_grid(n),
        [DIRECT_BLOCK, 1, 1],
        DIRECT_LDS,
        args.as_mut_slice(),
    )?;
    Ok(())
}
fn pm_pre_seg(gpu: &mut Gpu, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    gpu.fa_prefill_pm_preconvert(
        &a.q32().sub(off * ROW * 4, n * ROW * 4),
        &q16.sub(off * ROW * 2, n * ROW * 2),
        n,
    )?;
    Ok(())
}
fn pm_direct_seg(
    gpu: &mut Gpu,
    a: &Arena,
    c: &Cell,
    q16: &Buf,
    out: &Buf,
    off: usize,
    n: usize,
) -> Result<()> {
    let u = used_of(c);
    gpu.fa_prefill_pm_direct_codec(
        FaPrefillKvCodec::Fp8E4M3,
        &q16.sub(off * ROW * 2, n * ROW * 2),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &out.sub(off * ROW * 4, n * ROW * 4),
        &a.pos().sub(off * 4, n * 4),
        c.ctx(),
        n,
    )?;
    Ok(())
}
fn run_hip(gpu: &Gpu, a: &Arena, c: &Cell) -> Result<()> {
    for (off, n) in c.segments() {
        hip_pre_seg(gpu, a, a.q16_hip(), off, n)?;
        hip_direct_seg(gpu, a, a.q16_hip(), a.out_hip(), off, n)?;
    }
    Ok(())
}
fn run_pm(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<()> {
    for (off, n) in c.segments() {
        pm_pre_seg(gpu, a, a.q16_pm(), off, n)?;
        pm_direct_seg(gpu, a, c, a.q16_pm(), a.out_pm(), off, n)?;
    }
    Ok(())
}
/// Re-run only the PM direct (on the already converted `q16_pm`) into `out`.
fn run_pm_direct(gpu: &mut Gpu, a: &Arena, c: &Cell, out: &Buf) -> Result<()> {
    for (off, n) in c.segments() {
        pm_direct_seg(gpu, a, c, a.q16_pm(), out, off, n)?;
    }
    Ok(())
}

// ------------------------------------------------------------ cell data
struct Prepared {
    qh: Vec<f32>,
    pos: Vec<i32>,
    kc: Vec<u8>,
    vc: Vec<u8>,
    used: Used,
    fixture: Fixture,
    seed: u64,
}
/// Pure host admission, before any launch.
fn validate_cell(rows: usize, positions: &[i32], ctx: usize) -> std::result::Result<(), String> {
    if rows < 1 || rows > MAX_ROWS {
        return Err(format!("rows {rows} outside 1..={MAX_ROWS}"));
    }
    if positions.len() != rows {
        return Err(format!(
            "positions length {} != rows {rows}",
            positions.len()
        ));
    }
    fa_prefill_pm::validate_positions(positions, ctx).map_err(|e| e.to_string())?;
    if rows >= fa_prefill_pm::MIN_ROWS {
        fa_prefill_pm::validate_domain(H, KV, D, rows, ctx).map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn prepare(gpu: &mut Gpu, a: &Arena, c: &Cell, round: usize) -> Result<Prepared> {
    let u = used_of(c);
    let ctx = c.ctx();
    let fixture = c.fixture.for_round(round);
    let pos = fixture.positions(c.rows, c.start);
    validate_cell(c.rows, &pos, ctx)?;
    let seed = fnv(&c.name()) ^ (0x41a3b8f8u64 + (round as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let mut rng = Rng(seed);
    let qh: Vec<f32> = (0..c.rows * ROW).map(|_| rng.normal()).collect();
    for (buf, bytes_used) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, bytes_used)?;
    }
    a.q32().upload(gpu, bytes(&qh))?;
    a.pos().upload(gpu, bytes(&pos))?;
    match c.source {
        Source::Runtime => {
            let kf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal()).collect();
            let vf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal() * 0.7).collect();
            let allpos: Vec<i32> = (0..ctx as i32).collect();
            a.k32().upload(gpu, bytes(&kf))?;
            a.v32().upload(gpu, bytes(&vf))?;
            a.allpos().upload(gpu, bytes(&allpos))?;
            gpu.kv_cache_write_fp8_e4m3_batched(
                &a.kc().view(u.kc),
                &a.k32().view(u.k32),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?;
            gpu.kv_cache_write_fp8_e4m3_batched(
                &a.vc().view(u.vc),
                &a.v32().view(u.v32),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?;
            gpu.hip.device_synchronize()?;
        }
        Source::Grid | Source::Nan => {
            let nan = c.source == Source::Nan;
            a.kc().upload(gpu, &grid_cache(ctx, 0, nan))?;
            a.vc().upload(gpu, &grid_cache(ctx, 5, nan))?;
            gpu.hip.device_synchronize()?;
        }
    }
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for b in [
        a.q16_hip(),
        a.q16_pm(),
        a.out_hip(),
        a.out_pm(),
        a.out_extra(),
    ] {
        let len = if b.es == 4 { u.out } else { u.q16 };
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        qh,
        pos,
        kc,
        vc,
        used: u,
        fixture,
        seed,
    })
}
fn integrity(gpu: &Gpu, a: &Arena, p: &Prepared) -> Result<Value> {
    let u = &p.used;
    let inputs = json!({
        "q32": a.q32().download(gpu, u.q32)? == bytes(&p.qh),
        "k_cache": a.kc().download(gpu, u.kc)? == p.kc,
        "v_cache": a.vc().download(gpu, u.vc)? == p.vc,
        "positions": a.pos().download(gpu, u.pos)? == bytes(&p.pos),
    });
    let mut guards = serde_json::Map::new();
    for (b, n) in a.b.iter().zip(used_list(u)) {
        guards.insert(b.name.into(), json!(b.guards_ok(gpu, n)?));
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    Ok(
        json!({"inputs_unchanged":inputs,"guards_intact":guards,"guard_bytes_each_side":GUARD,"pass":ok}),
    )
}
// --------------------------------------- independent round-trip (after prepare, before integrity)
/// Upload independently constructed host inputs, arm every existing guard, poison every
/// output/Q16 buffer, and return the original host bytes in `Prepared`. Reuses the existing
/// launchers and `integrity`; never uses random `prepare` as the source of expected values.
/// The Q16 input is the frozen case's own Q16 words (f32 host upload of the exact words).
fn prepare_ieee(gpu: &mut Gpu, a: &Arena, ic: &IeeeCase) -> Result<Prepared> {
    let rows = ic.positions.len();
    let npos = ic.k_cache.len() / POS_BYTES;
    validate_cell(rows, &ic.positions, npos)?;
    let u = used(rows, npos);
    for (buf, bytes_used) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, bytes_used)?;
    }
    let qh_f32: Vec<f32> = ic.qh.iter().map(|&w| f16_to_f32_ieee(w)).collect();
    // The hardware preconvert reads f32 Q; upload the exact f32 values of the frozen Q16 words.
    a.q32().upload(gpu, bytes(&qh_f32))?;
    a.pos().upload(gpu, bytes(&ic.positions))?;
    a.kc().upload(gpu, &ic.k_cache)?;
    a.vc().upload(gpu, &ic.v_cache)?;
    gpu.hip.device_synchronize()?;
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for b in [
        a.q16_hip(),
        a.q16_pm(),
        a.out_hip(),
        a.out_pm(),
        a.out_extra(),
    ] {
        let len = if b.es == 4 { u.out } else { u.q16 };
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        qh: qh_f32,
        pos: ic.positions.clone(),
        kc,
        vc,
        used: u,
        fixture: Fixture::Std,
        seed: fnv(&ic.name),
    })
}
/// Independent classifier over raw output bytes: affected NaN positions must be NaN on both
/// arms with HIP/PM NaN-position agreement, and every unaffected word must match the
/// independent clean expectation byte for byte (separately for HIP and PM downloads).
/// Candidate/HIP downloads never construct expected output. Zero-valued independent fixtures
/// are valid: no `out_nonzero` gate here.
fn compare_ieee_words(expected: &IeeeExpected, got_words: &[u32]) -> Value {
    match expected {
        IeeeExpected::Finite(want) => {
            let mut mism = 0usize;
            let mut first = None::<usize>;
            for (i, (&g, &w)) in got_words.iter().zip(want.iter()).enumerate() {
                if g != w {
                    mism += 1;
                    first.get_or_insert(i);
                }
            }
            let pass = mism == 0 && got_words.len() == want.len();
            json!({"classification":"finite_byte_gate","words":got_words.len(),"mismatched_words":mism,
                "first_mismatch":first,"pass":pass})
        }
        IeeeExpected::VNaN {
            clean_expected_words,
            kv_head,
            dimension,
        } => {
            let (mut nan_mismatch, mut value_mismatch, mut affected_nan) = (0usize, 0usize, 0usize);
            let mut first = None::<usize>;
            for (i, (&g, &w)) in got_words
                .iter()
                .zip(clean_expected_words.iter())
                .enumerate()
            {
                let h = (i / D) % H;
                let d = i % D;
                let affected = h / 8 == *kv_head && d == *dimension;
                let gn = f32::from_bits(g).is_nan();
                if affected {
                    affected_nan += gn as usize;
                    if !gn {
                        nan_mismatch += 1;
                        first.get_or_insert(i);
                    }
                } else if g != w {
                    value_mismatch += 1;
                    first.get_or_insert(i);
                }
            }
            let pass = nan_mismatch == 0 && value_mismatch == 0 && affected_nan > 0;
            json!({"classification":"v_nan_separate","words":got_words.len(),"affected_nan":affected_nan,
                "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
                "first_mismatch":first,"pass":pass})
        }
        IeeeExpected::KNaN {
            clean_expected_words,
            kv_head,
        } => {
            let rows = got_words.len() / ROW;
            let (mut affected_rows_with_nan, mut affected_rows) = (0usize, 0usize);
            let (mut nan_mismatch, mut value_mismatch) = (0usize, 0usize);
            let mut first = None::<usize>;
            for r in 0..rows {
                for h in 0..H {
                    let got = &got_words[(r * H + h) * D..(r * H + h + 1) * D];
                    let want = &clean_expected_words[(r * H + h) * D..(r * H + h + 1) * D];
                    if h / 8 == *kv_head {
                        affected_rows += 1;
                        if got.iter().any(|&w| f32::from_bits(w).is_nan()) {
                            affected_rows_with_nan += 1;
                        } else {
                            nan_mismatch += 1;
                            first.get_or_insert((r * H + h) * D);
                        }
                    } else if got != want {
                        value_mismatch +=
                            got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
                        first.get_or_insert((r * H + h) * D);
                    }
                }
            }
            let pass = nan_mismatch == 0
                && value_mismatch == 0
                && affected_rows_with_nan == affected_rows
                && affected_rows > 0;
            json!({"classification":"k_nan_separate","words":got_words.len(),
                "affected_rows_with_nan":affected_rows_with_nan,"affected_rows":affected_rows,
                "nan_position_mismatches":nan_mismatch,"non_nan_mismatches":value_mismatch,
                "first_mismatch":first,"pass":pass})
        }
    }
}
/// Compare independently expected words against one arm's raw bytes.
#[allow(dead_code)]
fn compare_ieee(expected: &IeeeExpected, got_bytes: &[u8]) -> Result<Value> {
    if got_bytes.len() % 4 != 0 {
        return Err("independent output bytes must be whole f32 words".into());
    }
    let words: Vec<u32> = got_bytes
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    Ok(compare_ieee_words(expected, &words))
}
/// One independent round: upload the frozen case, run HIP and PM primitives against the
/// independent expectation separately, then retain the existing mutual HIP/PM classification
/// and byte comparison. Guards/poison/changed-input/original-matrix semantics are preserved
/// by reusing `integrity` and the guarded arena.
fn ieee_round(gpu: &mut Gpu, a: &Arena, ic: &IeeeCase) -> Result<Value> {
    let rows = ic.positions.len();
    let npos = ic.k_cache.len() / POS_BYTES;
    let p = prepare_ieee(gpu, a, ic)?;
    let u = p.used;
    // HIP primitive on the independent buffers.
    hip_pre_seg(gpu, a, a.q16_hip(), 0, rows)?;
    {
        let mut args = direct_kernargs(
            a.q16_hip().ptr(0),
            a.kc().ptr(0),
            a.vc().ptr(0),
            a.out_hip().ptr(0),
            a.pos().ptr(0),
            rows,
        );
        gpu.launch_kernel_blob(
            HIP_DIRECT,
            direct_grid(rows),
            [DIRECT_BLOCK, 1, 1],
            DIRECT_LDS,
            args.as_mut_slice(),
        )?;
    }
    // PM primitive on the independent buffers (whole: independent rows are <= 8, one segment).
    pm_pre_seg(gpu, a, a.q16_pm(), 0, rows)?;
    gpu.fa_prefill_pm_direct_codec(
        FaPrefillKvCodec::Fp8E4M3,
        &a.q16_pm().view(u.q16),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &a.out_pm().view(u.out),
        &a.pos().view(u.pos),
        npos,
        rows,
    )?;
    gpu.hip.device_synchronize()?;
    let out_hip = a.out_hip().download(gpu, u.out)?;
    let out_pm = a.out_pm().download(gpu, u.out)?;
    // Independent expectation per arm, never constructed from candidate downloads: finite cases
    // use the stored words; NaN cases reuse the stored clean reference (the NaN-carrying full
    // expectation equals `ieee_expected` over the frozen caches, verified in self-check).
    let words_hip: Vec<u32> = out_hip
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    let words_pm: Vec<u32> = out_pm
        .chunks_exact(4)
        .map(|w| u32::from_ne_bytes(w.try_into().unwrap()))
        .collect();
    let hip_independent = compare_ieee_words(&ic.expected, &words_hip);
    let pm_independent = compare_ieee_words(&ic.expected, &words_pm);
    // Existing mutual classification/byte comparison retained (NaN-aware for NaN cases).
    let nan_class = !matches!(ic.expected, IeeeExpected::Finite(_));
    let mutual = compare(&out_hip, &out_pm, Elem::F32, nan_class);
    let integ = integrity(gpu, a, &p)?;
    let ok = pass(&hip_independent) && pass(&pm_independent) && pass(&mutual) && pass(&integ);
    Ok(
        json!({"case":ic.name.clone(),"rows":rows,"positions":ic.positions.clone(),"allocated_ctx":npos,
        "assertion":ic.expected.kind(),"hip_vs_independent":hip_independent,"pm_vs_independent":pm_independent,
        "hip_vs_pm":mutual,"integrity":integ,"pass":ok}),
    )
}

// ------------------------------------------------------------ diagnostics
const SAMPLE_HEADS: [usize; 2] = [0, H - 1];
fn sample_rows(n: usize) -> Vec<usize> {
    let mut v = vec![0, n / 2, n - 1];
    v.dedup();
    v
}
/// Probe rows: row 3 sits inside the first 8-row group so the horizon bisects a query group.
fn probe_rows(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [0usize, 3, n / 2].into_iter().filter(|&r| r < n).collect();
    v.dedup();
    v
}
/// f64 causal reference over f32 Q and independently decoded, f16-staged fp8 K/V for sampled rows/heads.
/// It omits Q16 conversion and the kernels' WMMA/reduction DAG: diagnostic only.
fn cpu_reference(p: &Prepared, c: &Cell) -> Vec<f32> {
    let mut expected = Vec::new();
    for row in sample_rows(c.rows) {
        for head in SAMPLE_HEADS {
            let kh = head / (H / KV);
            let end = p.pos[row] as usize + 1;
            let mut scores = Vec::with_capacity(end);
            for pos in 0..end {
                let mut s = 0.0f64;
                for d in 0..D {
                    s += p.qh[(row * H + head) * D + d] as f64 * staged(&p.kc, pos, kh, d) as f64;
                }
                scores.push(s / 16.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max).exp();
                    *s
                })
                .sum();
            for d in 0..D {
                let v: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(pos, s)| s * staged(&p.vc, pos, kh, d) as f64)
                    .sum();
                expected.push((v / denom) as f32);
            }
        }
    }
    expected
}
fn sampled_output(gpu: &Gpu, buf: &Buf, c: &Cell) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    for row in sample_rows(c.rows) {
        let f = f32s(&buf.download_at(gpu, row * ROW * 4, ROW * 4)?);
        for head in SAMPLE_HEADS {
            v.extend_from_slice(&f[head * D..(head + 1) * D]);
        }
    }
    Ok(v)
}
fn ref_compare(expected: &[f32], actual: &[f32]) -> Value {
    let (mut max_abs, mut max_rel, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut nonfinite = 0usize;
    for (&e, &a) in expected.iter().zip(actual) {
        if !(e.is_finite() && a.is_finite()) {
            nonfinite += 1;
            continue;
        }
        let (e64, a64) = (e as f64, a as f64);
        let err = (e64 - a64).abs();
        max_abs = max_abs.max(err);
        max_rel = max_rel.max(err / e64.abs().max(1e-6));
        dot += e64 * a64;
        aa += e64 * e64;
        bb += a64 * a64;
    }
    let cosine = if aa == 0.0 || bb == 0.0 {
        Value::Null
    } else {
        json!(dot / (aa * bb).sqrt())
    };
    json!({"samples":expected.len(),"max_abs":max_abs,"max_rel":max_rel,"cosine":cosine,"nonfinite":nonfinite,
        "gate":false,"note":"f64 exact-decode reference without the shared f16 staging; diagnostic only"})
}
/// Replace every key above `horizon` in a pristine cache: `zero` all-zero rows, `extreme` code 0x7e (448) at scale
/// 1.0 (finite, so `0 * value` stays exact in the masked lanes).
fn poison_future(pristine: &[u8], horizon: usize, kind: &str) -> Vec<u8> {
    let mut v = pristine.to_vec();
    let npos = v.len() / POS_BYTES;
    for p in horizon + 1..npos {
        let row = &mut v[p * POS_BYTES..(p + 1) * POS_BYTES];
        match kind {
            "zero" => row.fill(0),
            _ => {
                row[..SCALE_OFF].fill(0x7e);
                for g in 0..KV {
                    row[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2]
                        .copy_from_slice(&0x3c00u16.to_le_bytes());
                }
            }
        }
    }
    v
}
/// Rows at or below a probe row's causal horizon must stay byte-identical when every later key is poisoned
/// (standard positions). The caches are restored from the pristine host snapshot afterwards.
fn poison_probe(gpu: &mut Gpu, a: &Arena, c: &Cell, p: &Prepared) -> Result<Value> {
    let u = p.used;
    let reference = a.out_pm().download(gpu, u.out)?;
    let mut results = Vec::new();
    let mut all = true;
    for row in probe_rows(c.rows) {
        let horizon = p.pos[row] as usize;
        for kind in ["zero", "extreme"] {
            a.kc().upload(gpu, &poison_future(&p.kc, horizon, kind))?;
            a.vc().upload(gpu, &poison_future(&p.vc, horizon, kind))?;
            gpu.hip.device_synchronize()?;
            a.out_extra().poison(gpu, u.out)?;
            run_pm_direct(gpu, a, c, a.out_extra())?;
            gpu.hip.device_synchronize()?;
            let probe = a.out_extra().download(gpu, u.out)?;
            let live = (row + 1) * ROW * 4;
            let ok = probe[..live] == reference[..live];
            all &= ok;
            results.push(json!({"probe_row":row,"horizon":horizon,"perturbation":kind,"rows_compared":row + 1,"byte_identical":ok}));
        }
    }
    a.kc().upload(gpu, &p.kc)?;
    a.vc().upload(gpu, &p.vc)?;
    gpu.hip.device_synchronize()?;
    Ok(json!({"probes":results,"pass":all}))
}

// ----------------------------------------------------------- oracle round
fn oracle_round(gpu: &mut Gpu, a: &Arena, c: &Cell, round: usize) -> Result<Value> {
    let p = prepare(gpu, a, c, round)?;
    let u = p.used;
    run_hip(gpu, a, c)?;
    run_pm(gpu, a, c)?;
    gpu.hip.device_synchronize()?;
    let nan = c.source == Source::Nan;
    let q16 = compare(
        &a.q16_hip().download(gpu, u.q16)?,
        &a.q16_pm().download(gpu, u.q16)?,
        Elem::F16,
        false,
    );
    let out_hip = a.out_hip().download(gpu, u.out)?;
    let out_pm = a.out_pm().download(gpu, u.out)?;
    let attention = compare(&out_hip, &out_pm, Elem::F32, nan);
    let wrapper = if c.wrapper() {
        a.out_extra().poison(gpu, u.out)?;
        gpu.attention_fp8_e4m3_fa_prefill_pm_gfx1201(
            &a.q32().view(u.q32),
            &a.kc().view(u.kc),
            &a.vc().view(u.vc),
            &a.out_extra().view(u.out),
            &a.pos().view(u.pos),
            H,
            KV,
            D,
            c.ctx(),
            c.rows,
        )?;
        gpu.hip.device_synchronize()?;
        Some(compare(
            &out_pm,
            &a.out_extra().download(gpu, u.out)?,
            Elem::F32,
            nan,
        ))
    } else {
        None
    };
    let q16_bytes = a.q16_pm().download(gpu, u.q16)?;
    let nonzero_q16 = q16_bytes
        .chunks_exact(2)
        .any(|x| u16::from_ne_bytes([x[0], x[1]]) & 0x7FFF != 0);
    let out_nonzero = f32s(&out_pm).iter().any(|&x| x != 0.0);
    let content_ok = nonzero_q16 && out_nonzero;
    let (reference, probes) = if c.diagnostics() && round == 0 && p.fixture == Fixture::Std {
        let r = ref_compare(&cpu_reference(&p, c), &sampled_output(gpu, a.out_pm(), c)?);
        (Some(r), Some(poison_probe(gpu, a, c, &p)?))
    } else {
        (None, None)
    };
    let integ = integrity(gpu, a, &p)?;
    let ok = pass(&q16)
        && pass(&attention)
        && wrapper.as_ref().map_or(true, pass)
        && pass(&integ)
        && content_ok
        && probes.as_ref().map_or(true, pass);
    Ok(
        json!({"round":round,"fixture":p.fixture.name(),"seed":p.seed,"q16":q16,"attention":attention,
        "wrapper_vs_pm_primitive":wrapper,"integrity":integ,"content_sanity_ok":content_ok,
        "future_poison_probes":probes,"cpu_reference_diagnostic":reference,"pass":ok}),
    )
}

// --------------------------------------------------------------- plan
fn plan() {
    let ieee = ieee_cases().expect("independent cases construct on CPU");
    for ic in &ieee {
        println!(
            "{}",
            json!({"ieee_case":ic.name.clone(),"rows":ic.positions.len(),"positions":ic.positions.clone(),
                "allocated_ctx":ic.k_cache.len() / POS_BYTES,"admitted_code_census":"rounded_exponent_overflow",
                "assertion":ic.expected.kind(),
                "hip_vs_pm":"separate","wrapper":false,"launches":"hip_primitive+pm_primitive"})
        );
    }
    let cells = oracle_cells();
    let with_long: Vec<Cell> = cells.iter().copied().chain(long_cells()).collect();
    let sizes = arena_sizes(&with_long);
    let total: usize = sizes.iter().map(|b| b + 2 * GUARD).sum();
    for c in &cells {
        println!(
            "{}",
            json!({"cell":c.name(),"rows":c.rows,"start":c.start,"segment":c.segment,"segments":c.segments().len(),
                "fixture":c.fixture.name(),"source":c.source.name(),"context":c.ctx(),"wrapper":c.wrapper(),
                "pm_pre_grid":pre_grid(c.rows.min(c.segment)),"direct_grid":direct_grid(c.rows.min(c.segment)),
                "launches_per_round":2 * 2 * c.segments().len()})
        );
    }
    for (rows, start, seg) in LONG_CELLS {
        println!(
            "{}",
            json!({"long_cell":format!("n{rows}_s{start}_seg{seg}_std_runtime_long"),"rows":rows,"start":start,"segment":seg,
                "context":LONG_CTX,"includes_last_position_16383":true,"whole_and_segmented":true,"wrapper_where_applicable":true})
        );
    }
    println!(
        "{}",
        json!({"plan":true,"ieee_cases":ieee.len(),"ieee_first":true,"cells":cells.len(),"long_cells":LONG_CELLS.len(),
            "buffers":BUFFER_NAMES,"arena_bytes_each":sizes,"arena_total_bytes":total,
            "hip_symbols":[HIP_PRE,HIP_DIRECT],"hip_defines":["HIPFIRE_FA2_KMODE=8","HIPFIRE_FA2_Q0_A3B=1"],
            "pm_module":fa_prefill_pm::FP8_MODULE,"pm_symbol":fa_prefill_pm::FP8_DIRECT,
            "pm_preconvert_module":fa_prefill_pm::MODULE,"pm_preconvert_symbol":fa_prefill_pm::PRECONVERT,
            "kernarg_bytes":{"pre":40,"direct":60},"direct_block":DIRECT_BLOCK,"direct_lds":DIRECT_LDS})
    );
}

// --------------------------------------------------------- self-check
fn self_check() -> usize {
    let mut n = 0usize;
    let mut ok = |c: bool, what: &str| {
        assert!(c, "self-check failed: {what}");
        n += 1;
    };
    // E4M3FN model.
    ok(
        e4m3fn(0x38) == Some(1.0) && e4m3fn(0x7e) == Some(448.0) && e4m3fn(0xfe) == Some(-448.0),
        "known codes",
    );
    ok(
        e4m3fn(0x01) == Some(2f32.powi(-9)) && e4m3fn(0x08) == Some(2f32.powi(-6)),
        "subnormal / smallest normal",
    );
    ok(
        e4m3fn(0x80).map(f32::to_bits) == Some(0x8000_0000),
        "0x80 is -0, not NaN",
    );
    ok(
        e4m3fn(0x7f).is_none() && e4m3fn(0xff).is_none() && finite_codes().len() == 254,
        "NaN codes",
    );
    ok(
        (0u8..0x7e).all(|c| e4m3fn(c).unwrap() < e4m3fn(c + 1).unwrap()),
        "monotone magnitudes",
    );
    // Row layout: 516-byte rows, head-major codes then two scales; ABI blobs.
    ok(
        POS_BYTES == 516
            && SCALE_OFF == 512
            && POS_BYTES == fa_prefill_pm::FaPrefillKvCodec::Fp8E4M3.cache_row_bytes(),
        "row layout",
    );
    let cache = grid_cache(4, 0, false);
    ok(cache.len() == 4 * POS_BYTES, "grid cache extent");
    let table = finite_codes();
    for p in 0..4 {
        for g in 0..KV {
            let scale = SCALES[(p + 3 * g) % SCALES.len()];
            let at = p * POS_BYTES + SCALE_OFF + 2 * g;
            ok(
                u16::from_le_bytes([cache[at], cache[at + 1]]) == scale,
                "scale slot of each head",
            );
            ok(
                (0..D).all(|d| {
                    cache[p * POS_BYTES + g * D + d]
                        == safe_code(scale, table[(d + 7 * p + 31 * g) % table.len()])
                }),
                "code plane of each head",
            );
            ok(
                staged(&cache, p, g, 3).is_finite(),
                "no overflow-invalid artifact in a grid cache",
            );
        }
    }
    ok(
        grid_cache(8, 0, true)
            .iter()
            .zip(grid_cache(8, 0, false))
            .filter(|(a, b)| a != &b)
            .count()
            >= 3,
        "NaN injection",
    );
    ok(
        pre_kernargs(std::ptr::null_mut(), std::ptr::null_mut(), 8).len()
            == fa_prefill_pm::PRE_KERNARG_BYTES,
        "preconvert ABI size",
    );
    let d = direct_kernargs(
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        8,
    );
    ok(
        d.len() == fa_prefill_pm::DIRECT_KERNARG_BYTES,
        "direct ABI size",
    );
    ok(
        DIRECT_BLOCK == fa_prefill_pm::DIRECT_BLOCK[0]
            && DIRECT_LDS == fa_prefill_pm::DIRECT_LDS
            && PRE_BLOCK == fa_prefill_pm::PRE_BLOCK[0],
        "launch config",
    );
    // Fixtures: every position inside the cell, duplicates / nonmonotone where promised.
    for f in [
        Fixture::Std,
        Fixture::Reverse8,
        Fixture::DupPairs8,
        Fixture::Rotate8,
        Fixture::SameGroup8,
    ] {
        for (rows, start) in [(1usize, 0usize), (7, 3), (64, 0), (65, 63), (512, 1)] {
            let pos = f.positions(rows, start);
            ok(
                pos.len() == rows
                    && pos
                        .iter()
                        .all(|&x| (x as usize) >= start && (x as usize) < start + rows),
                "fixture positions in range",
            );
            ok(
                validate_cell(rows, &pos, start + rows).is_ok(),
                "fixture admitted",
            );
        }
    }
    ok(
        Fixture::DupPairs8
            .positions(8, 0)
            .windows(2)
            .any(|w| w[0] == w[1]),
        "duplicate pairs",
    );
    ok(
        Fixture::Reverse8.positions(8, 0) == [7, 6, 5, 4, 3, 2, 1, 0],
        "reverse8",
    );
    ok(
        validate_cell(0, &[], 1).is_err()
            && validate_cell(1, &[1], 1).is_err()
            && validate_cell(1, &[-1], 4).is_err(),
        "domain refusals",
    );
    // Cells: every cell is admitted, tails are primitive-only, the wrapper cells are production rows.
    let cells = oracle_cells();
    for c in &cells {
        ok(
            validate_cell(c.rows, &c.fixture.positions(c.rows, c.start), c.ctx()).is_ok(),
            "cell admitted",
        );
        ok(
            c.wrapper() == (c.rows >= 64 && c.segment == c.rows),
            "wrapper cells",
        );
        ok(
            c.rows >= 64 || c.source == Source::Runtime,
            "tails are runtime-cache primitive cells",
        );
    }
    ok(
        cells.iter().any(|c| c.rows == 1) && cells.iter().any(|c| c.rows == 63),
        "exact tails 1..=63",
    );
    ok(
        cells.iter().any(|c| c.source == Source::Nan)
            && cells.iter().any(|c| c.source == Source::Grid),
        "synthetic sources",
    );
    ok(
        hip_source().starts_with("#define HIPFIRE_FA2_KMODE 8\n#define HIPFIRE_FA2_Q0_A3B 1\n"),
        "hip twin defines",
    );
    ok(
        !hip_source().contains("#define HIPFIRE_FA2_FP8"),
        "hip twin is not FA2_FP8",
    );
    ok(
        HIP_BODY.contains("HIPFIRE_FA2_Q0_A3B")
            && HIP_BODY.contains(HIP_DIRECT)
            && HIP_BODY.contains("attention_fp8_e4m3_fa2_q_preconvert_f16_a3b_gfx1201"),
        "production hip carries the A3B twin symbols",
    );
    // Comparison semantics (original cells; the independent fixtures use their own classifier).
    let a = [0u8, 0, 0x80, 0x3f];
    ok(pass(&compare(&a, &a, Elem::F32, false)), "identical passes");
    let mut b = a;
    b[0] = 1;
    ok(
        !pass(&compare(&a, &b, Elem::F32, false)),
        "one bit fails the byte gate",
    );
    let nan = f32::NAN.to_ne_bytes();
    ok(
        !pass(&compare(&nan, &nan, Elem::F32, false)),
        "NaN fails the finite gate",
    );
    ok(
        pass(&compare(&nan, &nan, Elem::F32, true)),
        "NaN positions agree under the NaN classification",
    );
    // Independent native semantic proof: IEEE f16 RNE / single-key accumulation / all 256 E4M3
    // encodings / census / overflow / signed zeros / empty-subtile admission / unaffected NaN
    // bytes / negative controls / long-context boundaries. Pure CPU, before any GPU init.
    ok(
        ieee_f16_rne(1.0) == 0x3c00
            && ieee_f16_rne(65504.0) == 0x7bff
            && ieee_f16_rne(65520.0) == 0x7c00,
        "f16 RNE normal/overflow boundaries",
    );
    ok(
        ieee_f16_rne(-0.0) == 0x8000 && ieee_f16_rne(0.0) == 0x0000,
        "f16 RNE signed zero preserved at the bit level",
    );
    ok(
        f16_to_f32_ieee(0x3c00) == 1.0 && f16_to_f32_ieee(0x8000).to_bits() == 0x8000_0000,
        "independent f16 bit decode",
    );
    ok(
        (0..=255u8).all(|code| match e4m3fn(code) {
            Some(_) => ieee_stage_one(code, 0x3c00).is_some(),
            None => code == 0x7f || code == 0xff,
        }),
        "all 256 E4M3 encodings: 254 finite stage, exactly 0x7f/0xff NaN",
    );
    {
        // Staged -0 accumulates from +0 and the independent mean returns +0 (signed-zero canonicalization).
        let mut cache = vec![0u8; POS_BYTES];
        cache[3] = 0x80;
        cache[SCALE_OFF..SCALE_OFF + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for g in 1..KV {
            cache[SCALE_OFF + 2 * g..SCALE_OFF + 2 * g + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let w = ieee_stage(&cache, 0, 0, 3).unwrap();
        let mut acc = 0.0f32;
        acc += f16_to_f32_ieee(w);
        let out = acc;
        ok(
            w == 0x8000 && out.to_bits() == 0x0000_0000,
            "staged -0 accumulates from +0 to output +0",
        );
    }
    ok(
        ieee_cases().map(|v| v.len()).unwrap_or(0) == 21,
        "independent frozen case count: 2 KNaN + 1 empty + 4 VNaN + 14 sweep",
    );
    for ic in ieee_cases().unwrap() {
        let want = ieee_expected(&ic).unwrap();
        ok(
            want.len() == ic.positions.len() * ROW,
            "independent expected extent",
        );
        match &ic.expected {
            IeeeExpected::Finite(_) => ok(true, "finite"),
            IeeeExpected::VNaN {
                clean_expected_words,
                kv_head,
                dimension,
            } => {
                // Precisely the selected KV-head/dimension outputs are NaN; all unaffected words match clean.
                let mut affected_ok = true;
                let mut unaffected_ok = true;
                for (r, _) in ic.positions.iter().enumerate() {
                    for h in 0..H {
                        for d in 0..D {
                            let i = (r * H + h) * D + d;
                            let is_aff = h / 8 == *kv_head && d == *dimension;
                            let word_nan = f32::from_bits(want[i]).is_nan();
                            affected_ok &= !is_aff || word_nan;
                            unaffected_ok &= is_aff || want[i] == clean_expected_words[i];
                        }
                    }
                }
                ok(
                    unaffected_ok,
                    "VNaN unaffected words match clean expectation",
                );
                ok(affected_ok, "VNaN affected words are NaN");
            }
            IeeeExpected::KNaN {
                clean_expected_words,
                kv_head,
            } => {
                // Each query-head row of the affected KV head contains NaN; the other head matches clean.
                let mut affected_ok = true;
                let mut unaffected_ok = true;
                for (r, _) in ic.positions.iter().enumerate() {
                    for h in 0..H {
                        let row = &want[(r * H + h) * D..(r * H + h + 1) * D];
                        let clean = &clean_expected_words[(r * H + h) * D..(r * H + h + 1) * D];
                        if h / 8 == *kv_head {
                            affected_ok &= row.iter().any(|&w| f32::from_bits(w).is_nan());
                        } else {
                            unaffected_ok &= row == clean;
                        }
                    }
                }
                ok(
                    unaffected_ok,
                    "KNaN unaffected words match clean expectation",
                );
                ok(affected_ok, "KNaN affected rows contain NaN");
            }
        }
    }
    {
        // Admitted-code census: recompute it the way the sweep admission does (rounded-exponent
        // overflow) and require every admitted code to be present in the emitted sweep caches.
        const IEEE_SCALES: [u16; 12] = [
            0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x2800, 0x6000, 0xbc01,
            0xfbff,
        ];
        let shifts = [0usize, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 0, 3];
        let mut census_ok = true;
        for shift in shifts {
            let scales = [
                IEEE_SCALES[shift],
                IEEE_SCALES[(shift + 3) % IEEE_SCALES.len()],
            ];
            let (k, v) = single_key_caches(scales, shift);
            for g in 0..KV {
                let admitted: std::collections::BTreeSet<u8> = finite_codes()
                    .into_iter()
                    .filter(|&code| {
                        ieee_stage_one(code, scales[g])
                            .map(|w| w & 0x7c00 != 0x7c00)
                            .unwrap_or(false)
                    })
                    .collect();
                for cache in [&k, &v] {
                    let seen: std::collections::BTreeSet<u8> =
                        cache[g * D..(g + 1) * D].iter().copied().collect();
                    census_ok &= seen == admitted;
                }
            }
        }
        ok(census_ok, "admitted-code census over the sweep caches");
    }
    {
        // Negative controls: a one-bit flip of a finite word, and injecting a NaN code into a
        // finite case, both fail the independent classifier.
        let ic = ieee_cases()
            .unwrap()
            .into_iter()
            .find(|c| matches!(c.expected, IeeeExpected::Finite(_)))
            .unwrap();
        let want = ieee_expected(&ic).unwrap();
        let mut flipped = want.clone();
        flipped[0] ^= 1;
        ok(
            compare_ieee_words(&ic.expected, &flipped)["pass"] == false,
            "one-bit flip fails independent classifier",
        );
        let mut nan_case = ic.clone();
        nan_case.v_cache[5] = 0x7f;
        let nan_got = ieee_expected(&nan_case).unwrap();
        ok(
            compare_ieee_words(&ic.expected, &nan_got)["pass"] == false,
            "NaN injection fails the finite classifier",
        );
    }
    ok(
        LONG_CELLS.iter().all(|&(rows, start, seg)| {
            validate_cell(rows, &Fixture::Std.positions(rows, start), start + rows).is_ok()
                && seg <= rows
        }),
        "long cells admitted",
    );
    ok(
        LONG_CELLS.iter().all(|&(rows, start, _)| {
            start + rows == LONG_CTX
                && Fixture::Std
                    .positions(rows, start)
                    .contains(&(LONG_CTX as i32 - 1))
        }),
        "long cells include last position 16383 at context 16384",
    );
    ok(
        validate_cell(1, &[16383], LONG_CTX).is_ok()
            && validate_cell(1, &[16400], LONG_CTX).is_err(),
        "last position 16383 admitted, 16400 rejected for context 16384",
    );
    n
}

// ------------------------------------------------------------------ main
struct Opts {
    plan: bool,
    check: bool,
    filter: Option<String>,
}
fn parse() -> Result<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        filter: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--filter" => o.filter = Some(it.next().ok_or("--filter needs a substring")?),
            _ => return Err(format!("unknown argument {a}; usage: fa_pm_fp8_oracle [--plan|--self-check] [--filter SUBSTR]").into()),
        }
    }
    Ok(o)
}
/// Same NEW-kernel controlled dynamic-LDS comparison on the actually loaded FP8 direct
/// symbol (`fa_prefill_pm::FP8_DIRECT`, block 160, static LDS fixed 0): actual 32,768 B
/// dynamic vs 65,536 B dynamic control. Not the old PM driver proof. Reports the real
/// `Gpu::occupancy_max_active_blocks` results; call only after the PM module is loaded
/// (a PM direct launch has run), so the loaded-function lookup resolves.
fn occupancy_control(gpu: &Gpu) -> Value {
    let block = [DIRECT_BLOCK, 1, 1];
    let probe = |dynamic_lds: u32| match gpu.occupancy_max_active_blocks(
        fa_prefill_pm::FP8_DIRECT,
        block,
        dynamic_lds,
    ) {
        Ok(v) => json!({"ok": true, "max_active_blocks": v}),
        Err(e) => json!({"ok": false, "error": e.to_string()}),
    };
    json!({"kind": "occupancy_control",
        "kernel": fa_prefill_pm::FP8_DIRECT, "module": fa_prefill_pm::FP8_MODULE,
        "block": block, "static_lds_bytes": 0,
        "actual_dynamic_lds_32768": probe(32_768),
        "control_dynamic_lds_65536": probe(65_536),
        "note": "same NEW-kernel controlled dynamic-LDS comparison; not the old PM driver proof"})
}

fn run() -> Result<bool> {
    let n = self_check();
    let o = parse()?;
    if o.plan {
        plan();
        return Ok(true);
    }
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":n,"gpu_launched":false})
        );
        return Ok(true);
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let src = hip_source();
    let src_sha = sha_hex(src.as_bytes());
    gpu.ensure_kernel_public(HIP_DIRECT, &src, HIP_DIRECT)?;
    gpu.ensure_kernel_public(HIP_DIRECT, &src, HIP_PRE)?;
    println!(
        "{}",
        json!({"kind":"provenance",
            "hip_twin":{"source_sha256":src_sha,"production_hip_sha256":sha_hex(HIP_BODY.as_bytes()),
                "defines":["HIPFIRE_FA2_KMODE=8","HIPFIRE_FA2_Q0_A3B=1"],"symbols":[HIP_PRE,HIP_DIRECT],
                "compiled_by_this_tool":true,"compiler":"Gpu::ensure_kernel_public (repo hipcc JIT)"},
            "pm_fp8":{"module":fa_prefill_pm::FP8_MODULE,"symbol":fa_prefill_pm::FP8_DIRECT,
                "image_bytes":fa_prefill_pm::FP8_IMAGE.len(),"image_sha256":sha_hex(fa_prefill_pm::FP8_IMAGE)},
            "pm_preconvert":{"module":fa_prefill_pm::MODULE,"symbol":fa_prefill_pm::PRECONVERT,
                "image_bytes":fa_prefill_pm::IMAGE.len(),"image_sha256":sha_hex(fa_prefill_pm::IMAGE),
                "note":"shared with the frozen Q8 PM module; its pinned lab object is untouched"}})
    );
    let mut cells: Vec<Cell> = oracle_cells()
        .into_iter()
        .filter(|c| {
            o.filter
                .as_ref()
                .map_or(true, |f| c.name().contains(f.as_str()))
        })
        .collect();
    // Runtime-written long-context cells last: context exactly 16384, uploaded positions include
    // 16383, `Fixture::Std` with two changed-position rounds and both production K/V writes.
    cells.extend(long_cells().into_iter().filter(|c| {
        o.filter.as_ref().map_or(true, |f| {
            f.as_str() == "long" || c.name().contains(f.as_str())
        })
    }));
    let ieee_selected: Vec<IeeeCase> = ieee_cases()?
        .into_iter()
        .filter(|ic| {
            o.filter
                .as_ref()
                .map_or(true, |f| ic.name.contains(f.as_str()))
        })
        .collect();
    if cells.is_empty() && ieee_selected.is_empty() {
        return Err("--filter matched no cell".into());
    }
    // The arena always covers the independent extents (at most 8 rows / 66 contexts): with no
    // ordinary cell selected, a synthetic cover cell sizes it.
    let cover = Cell {
        rows: 8,
        start: 58,
        segment: 8,
        fixture: Fixture::Std,
        source: Source::Runtime,
    };
    let mut sized = cells.clone();
    sized.push(cover);
    let arena = Arena::new(&mut gpu, &sized)?;
    let (mut finite_cells, mut finite_identical, mut nan_cells, mut nan_classified, mut failures) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut ieee_cells, mut ieee_passed) = (0usize, 0usize);
    // Independent native semantic proof first: the first failure returns nonzero before any ordinary
    // or long-context launch.
    let mut ieee_ok = true;
    for ic in &ieee_selected {
        ieee_cells += 1;
        match ieee_round(&mut gpu, &arena, ic) {
            Ok(r) => {
                let ok = r["pass"] == true;
                ieee_passed += ok as usize;
                failures += !ok as usize;
                println!(
                    "{}",
                    json!({"kind":"ieee_case","case":ic.name.clone(),"rows":ic.positions.len(),"positions":ic.positions.clone(),
                        "allocated_ctx":ic.k_cache.len() / POS_BYTES,"assertion":ic.expected.kind(),"rounds":r,"pass":ok})
                );
                if !ok {
                    ieee_ok = false;
                    break;
                }
            }
            Err(e) => {
                failures += 1;
                println!(
                    "{}",
                    json!({"kind":"ieee_case","case":ic.name.clone(),"error":e.to_string(),"pass":false})
                );
                ieee_ok = false;
                break;
            }
        }
    }
    if ieee_ok {
        for c in &cells {
            let mut rounds = Vec::new();
            let mut ok = true;
            for round in 0..2 {
                let r = oracle_round(&mut gpu, &arena, c, round)?;
                ok &= r["pass"] == true;
                rounds.push(r);
            }
            println!(
                "{}",
                json!({"kind":"oracle_cell","cell":c.name(),"rows":c.rows,"start_pos":c.start,"segment_rows":c.segment,
                    "segments":c.segments().len(),"fixture":c.fixture.name(),"source":c.source.name(),
                    "module_key":fa_prefill_pm::FP8_MODULE,"rounds":rounds,
                    "classification":if c.source == Source::Nan {"nan_codes_separate"} else {"finite_byte_gate"},
                    "pass":ok})
            );
            if c.source == Source::Nan {
                nan_cells += 1;
                nan_classified += ok as usize;
            } else {
                finite_cells += 1;
                finite_identical += ok as usize;
            }
            failures += !ok as usize;
            // Fail fast on an actual ordinary-case FAIL: stop immediately so a real GPU
            // divergence is not buried under later cells. NaN expected-classification
            // variation is not a failure here: `ok` already carries the NaN-aware
            // classification gate, and all finite/classification/poison checks are unchanged.
            if !ok {
                break;
            }
        }
    }
    // The ordinary loop above ran at least one PM direct launch, so the actual PM module
    // is loaded and the loaded-function occupancy lookup resolves.
    if ieee_ok && finite_cells + nan_cells > 0 {
        println!("{}", occupancy_control(&gpu));
    }
    println!(
        "{}",
        json!({"summary":true,"mode":"oracle","hip_twin_source_sha256":src_sha,
            "pm_fp8_image_sha256":sha_hex(fa_prefill_pm::FP8_IMAGE),"cells":cells.len(),
            "finite_cells":finite_cells,"finite_byte_identical":finite_identical,
            "nan_cells":nan_cells,"nan_classifications_passed":nan_classified,"failures":failures,
            "ieee_cells":ieee_cells,"ieee_passed":ieee_passed,"independent_failed_before_ordinary":!ieee_ok})
    );
    println!("all_byte_identical {}", finite_cells == finite_identical);
    println!("all_classifications_pass {}", failures == 0);
    Ok(failures == 0)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("fa_pm_fp8_oracle: {e}");
            std::process::exit(1);
        }
    }
}
