// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hardware tests of the split-K QSA decode attention helper
//! `tensor_ops::indexed_attention_attention_splitk_for_test` against the
//! incumbent exact per-head reference `indexed_attention_attention_batch_exact`
//! (F32 K/V, n_heads 24, n_kv_heads 2, head_dim 256).
//!
//! * `qsa_splitk_rows1_matches_exact`: single-row launches at lengths
//!   {1, S-1, S, 64, 1131, 2048, 8195} (for every tuned split count S, up to
//!   48), four compress/budget/capacity formulas, random q/K/V, a pure-valid
//!   selection and a mixed one that carries negative tokens, tokens >= visible
//!   and tokens >= full_capacity. Per head: maxdiff <= 1e-5 * maxabsref + 1e-7
//!   for the head layouts 0/1 at S = 4/8/12/16 and the grouped layout 2 at
//!   S = 8/16/24/32/48 (plus the production defaults, when they differ). A row
//!   with no valid token must be exactly +0, as in the reference.
//! * `qsa_splitk_all_invalid_zero`: all-invalid selections (rows 1 and
//!   batched 1..8) write exactly +0.0 for every split count and layout.
//! * `qsa_splitk_batched_rows_bitwise_individual`: rows 1..8 batched launches
//!   equal, bit for bit, the single-row launches of the same q row, selected row
//!   and position_start + row, for heterogeneous selected_len (compress 8,
//!   budgets around the block boundary, empty slices, S > len, capacity clamp,
//!   an all-invalid row inside every batch of two or more rows). Every output
//!   starts as a NaN sentinel; batched outputs are also compared with the exact
//!   reference.
//! * `qsa_splitk_wide_rows_*_bitwise_individual` (wide verify, design 7B): the
//!   production geometry (compress 4, budget 512, capacity 2051, F32 K/V,
//!   24 heads / 2 KV heads / head_dim 256) at end depths 8192 / 32768 / 131072 /
//!   262144 (full cache capacity = depth), end displacements 0..3 (selected
//!   lengths 2048..2051 across every tail phase). Batched n = 1..=32 rows
//!   (every n at displacement 0, ragged 1/2/3/4/5/8/9/15/16/17/31/32 elsewhere)
//!   equal bit for bit the n single-row launches of the same (splits, layout):
//!   production S32/layout 2, lab S16/S48 layout 2 and layouts 0/1 at S8, plus
//!   the production dispatcher (`indexed_attention_attention_batch`, split-K on
//!   gfx1151, the per-head kernels on gfx1201). Selections mix padded, garbage-
//!   after-ell, invalid-token and all-invalid rows; K/V rows no earlier row may
//!   read (>= its visible count) are set huge and the earlier rows must not
//!   change. The wide gate is set inside `launch` through
//!   `Gpu::set_wide_verify_rows` (32 for a batched launch, 0 for one row), never
//!   by env. The existing batched tests now run rows 1..=32.
//! * `qsa_splitk_timing_table`: hipEvent timing (100 calls per sample, median
//!   of 5, warmup) of the incumbent and every split/layout at rows 1, lengths
//!   1131 and 8188 (the longest length whose incumbent still fits the 64 KiB
//!   dynamic-LDS parallel kernel; 8195 falls to the serial kernel and is not
//!   timed). Each row carries the modelled K/V bytes per call (K and V
//!   separately and total, all tokens valid): 49152 * len for the per-head
//!   layouts 0/1 and the incumbent, 4096 * len for the grouped layout 2. The
//!   best head and best group combos are compared per length. Nothing is
//!   asserted on timings.
//!
//! `#[ignore]`d: needs a working HIP device. Run explicitly, one test thread
//! (the tests share the one GPU):
//!
//!   HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0 cargo test -p rdna-compute --release --features lab --test qsa_decode_splitk \
//!       -- --ignored --nocapture --test-threads=1
//!
//! No environment variable is set by this file. `qsa_splitk_timing_table`
//! requires `HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` in the caller's environment
//! (split-K is on by default on gfx1151), so its production-dispatch row is the
//! per-head incumbent route.

#![cfg(feature = "lab")]

use rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
use rdna_compute::tensor_ops::{
    indexed_attention_attention_batch, indexed_attention_attention_batch_exact,
    indexed_attention_attention_splitk_for_test, IndexedAttentionAttentionBatch, QsaKvFormat,
    QSA_DECODE_SPLITK_SCORING_LAYOUT, QSA_DECODE_SPLITK_SPLITS,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Rows of the heterogeneous batched configurations.
const MAX_BATCH_ROWS: usize = WIDE_VERIFY_MAX_ROWS;

const NH: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
/// KV cache rows: above the longest context (8196 visible) with slack, so
/// tokens in [visible, FULL_CAP) are in-bounds cache rows the kernel must still
/// reject.
const FULL_CAP: usize = 8300;
/// Budget large enough never to bind.
const HUGE: usize = 1 << 20;
const SENTINEL: u32 = 0x7fc0_dead;
const HEAD_SPLITS: [usize; 4] = [4, 8, 12, 16];
const HEAD_LAYOUTS: [usize; 2] = [0, 1];
const GROUP_SPLITS: [usize; 5] = [8, 16, 24, 32, 48];
const GROUP_LAYOUT: usize = 2;
const WARMUP_CALLS: usize = 10;
const CALLS_PER_SAMPLE: usize = 100;
const SAMPLES: usize = 5;

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Deterministic value in [-scale, scale).
    fn value(&mut self, scale: f32) -> f32 {
        (self.next() as u32 & 0xff_ffff) as f32 / (1u32 << 24) as f32 * 2.0 * scale - scale
    }
}

fn fill(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.value(scale)).collect()
}

/// `rows` query rows: per head 256 query values (+/-2) then 256 gate values (+/-3).
fn make_q(rng: &mut Rng, rows: usize) -> Vec<f32> {
    let mut q = Vec::with_capacity(rows * 2 * NH * HD);
    for _ in 0..rows * NH {
        q.extend((0..HD).map(|_| rng.value(2.0)));
        q.extend((0..HD).map(|_| rng.value(3.0)));
    }
    q
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Valid, negative, >= visible and >= FULL_CAP tokens (pure valid when the
    /// selected length is 1).
    Mixed,
    AllValid,
    AllInvalid,
}

fn valid_token(rng: &mut Rng, visible: usize) -> i32 {
    rng.below(visible as u64) as i32
}

/// kind 0: negative, 1: in [visible, FULL_CAP), 2: >= FULL_CAP.
fn invalid_token(rng: &mut Rng, kind: usize, visible: usize) -> i32 {
    match kind % 3 {
        0 => match rng.below(4) {
            0 => -1,
            1 => i32::MIN,
            2 => -1 - rng.below(100_000) as i32,
            _ => -2,
        },
        1 => visible as i32 + rng.below((FULL_CAP - visible) as u64) as i32,
        _ => match rng.below(3) {
            0 => i32::MAX,
            1 => FULL_CAP as i32,
            _ => FULL_CAP as i32 + rng.below(1000) as i32,
        },
    }
}

/// The kernel's active selected length of the row with `visible` keys.
fn selected_len(visible: usize, budget: usize, compress: usize, capacity: usize) -> usize {
    let block_count = visible / compress;
    let chosen = budget.min(block_count);
    (chosen * compress + visible - block_count * compress).min(capacity)
}

/// One `capacity`-wide selected row. Slots at or past `sel_len` hold valid-looking
/// tokens (reading them would change the result) unless the mode is AllInvalid.
fn gen_row(rng: &mut Rng, mode: Mode, visible: usize, sel_len: usize, capacity: usize) -> Vec<i32> {
    let mut row: Vec<i32> = (0..capacity)
        .map(|i| match mode {
            Mode::AllInvalid => {
                let kind = rng.below(3) as usize;
                invalid_token(rng, kind, visible)
            }
            Mode::AllValid => valid_token(rng, visible),
            Mode::Mixed => {
                if i >= sel_len {
                    return valid_token(rng, visible);
                }
                match rng.below(10) {
                    0 => invalid_token(rng, 0, visible),
                    1 => invalid_token(rng, 1, visible),
                    2 => invalid_token(rng, 2, visible),
                    _ => valid_token(rng, visible),
                }
            }
        })
        .collect();
    if mode == Mode::Mixed && sel_len > 0 {
        // One guaranteed valid slot and, as far as the length permits, one
        // guaranteed invalid token of each kind.
        let k = rng.below(sel_len as u64) as usize;
        row[k] = valid_token(rng, visible);
        for t in 1..=3.min(sel_len - 1) {
            row[(k + t) % sel_len] = invalid_token(rng, t - 1, visible);
        }
    }
    row
}

fn valid_count(row: &[i32], sel_len: usize, visible: usize) -> usize {
    row.iter()
        .take(sel_len)
        .filter(|&&t| t >= 0 && (t as usize) < visible && (t as usize) < FULL_CAP)
        .count()
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

struct Dev {
    keys: GpuTensor,
    values: GpuTensor,
    /// Cache rows (`full_capacity` of every launch).
    full_cap: usize,
}

fn make_dev(gpu: &mut Gpu) -> Dev {
    let mut rng = Rng(0x5a5a_0001);
    let n = FULL_CAP * NKV * HD;
    let keys = gpu
        .upload_f32(&fill(&mut rng, n, 1.0), &[n])
        .expect("upload keys");
    let values = gpu
        .upload_f32(&fill(&mut rng, n, 1.0), &[n])
        .expect("upload values");
    Dev {
        keys,
        values,
        full_cap: FULL_CAP,
    }
}

fn sentinel(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let data = vec![f32::from_bits(SENTINEL); n];
    gpu.upload_f32(&data, &[n]).expect("upload sentinel")
}

fn upload_q(gpu: &mut Gpu, q: &[f32]) -> GpuTensor {
    gpu.upload_f32(q, &[q.len()]).expect("upload q")
}

fn upload_selected(gpu: &Gpu, rows: &[Vec<i32>]) -> GpuTensor {
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload selected")
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<f32> {
    gpu.download_f32(t).expect("download")
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    rows: usize,
    position_start: usize,
    budget: usize,
    compress: usize,
    capacity: usize,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    /// `indexed_attention_attention_batch_exact`: the per-head kernels.
    Exact,
    /// `indexed_attention_attention_batch`: the production dispatcher.
    Production,
    /// `indexed_attention_attention_splitk_for_test(splits, scoring_layout)`.
    Split(usize, usize),
}

fn launch(
    gpu: &mut Gpu,
    kind: Kind,
    dev: &Dev,
    q: &GpuTensor,
    selected: &GpuTensor,
    out: &GpuTensor,
    sh: &Shape,
) {
    let p = IndexedAttentionAttentionBatch {
        q_with_gate: q,
        full_keys: &dev.keys,
        full_values: &dev.values,
        selected,
        output: out,
        rows: sh.rows,
        position_start: sh.position_start,
        n_heads: NH,
        n_kv_heads: NKV,
        head_dim: HD,
        budget_blocks: sh.budget,
        compress: sh.compress,
        capacity: sh.capacity,
        full_capacity: dev.full_cap,
        format: QsaKvFormat::F32,
        shape_selected: sh.capacity,
    };
    // A verify forward of two or more rows runs row-exact to the widest rows;
    // a one-row launch is the decode itself.
    gpu.set_wide_verify_rows(if sh.rows >= 2 {
        WIDE_VERIFY_MAX_ROWS
    } else {
        0
    });
    let result = match kind {
        Kind::Exact => indexed_attention_attention_batch_exact(gpu, &p),
        Kind::Production => indexed_attention_attention_batch(gpu, &p),
        Kind::Split(splits, layout) => {
            indexed_attention_attention_splitk_for_test(gpu, &p, splits, layout)
        }
    };
    gpu.set_wide_verify_rows(0);
    if let Err(e) = result {
        panic!("{kind:?} failed for {sh:?}: {e:?}");
    }
}

/// Every (splits, scoring_layout) under test: head layouts 0 and 1 at
/// S = 4/8/12/16, the grouped layout 2 (12 heads share one KV head) at
/// S = 8/16/24/32/48, plus the production defaults when they are not among them.
fn combos() -> Vec<(usize, usize)> {
    let mut combos: Vec<(usize, usize)> = HEAD_LAYOUTS
        .iter()
        .flat_map(|&l| HEAD_SPLITS.iter().map(move |&s| (s, l)))
        .chain(GROUP_SPLITS.iter().map(|&s| (s, GROUP_LAYOUT)))
        .collect();
    let default = (
        QSA_DECODE_SPLITK_SPLITS as usize,
        QSA_DECODE_SPLITK_SCORING_LAYOUT as usize,
    );
    if !combos.contains(&default) {
        combos.push(default);
    }
    combos
}

/// {1, S-1, S, 64, 1131, 2048, 8195} unioned over every tuned S (S <= 48, so
/// every visible length stays well inside FULL_CAP).
fn lengths(combos: &[(usize, usize)]) -> Vec<usize> {
    let mut lens = vec![1, 64, 1131, 2048, 8195];
    for &(s, _) in combos {
        lens.push(s - 1);
        lens.push(s);
    }
    lens.sort_unstable();
    lens.dedup();
    lens
}

// ---------------------------------------------------------------------------
// Comparison against the exact reference
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Worst {
    diff: f64,
    tol: f64,
    ratio: f64,
}

impl Worst {
    fn update(&mut self, diff: f64, tol: f64) {
        let ratio = diff / tol;
        if ratio >= self.ratio {
            *self = Worst { diff, tol, ratio };
        }
    }

    fn merge(&mut self, other: Worst) {
        if other.ratio >= self.ratio {
            *self = other;
        }
    }
}

struct Stats {
    fails: Vec<String>,
    worst: Vec<Worst>,
}

impl Stats {
    fn new(combos: usize) -> Self {
        Stats {
            fails: Vec::new(),
            worst: vec![Worst::default(); combos],
        }
    }

    fn fail(&mut self, msg: String) {
        self.fails.push(msg);
    }

    fn finish(&self, combos: &[(usize, usize)]) {
        println!("--- worst per-head maxdiff / tolerance per (splits, layout) ---");
        for (c, &(s, l)) in combos.iter().enumerate() {
            let w = self.worst[c];
            println!(
                "splits={s:>2} layout={l} maxdiff={:.3e} tol={:.3e} ratio={:.3}",
                w.diff, w.tol, w.ratio
            );
        }
        if !self.fails.is_empty() {
            let shown: Vec<&str> = self.fails.iter().take(40).map(String::as_str).collect();
            panic!(
                "{} failure(s); first {}:\n{}",
                self.fails.len(),
                shown.len(),
                shown.join("\n")
            );
        }
    }
}

/// Launches the exact reference and every combo on one `Shape` (rows `sel.len()`),
/// compares them, and returns each combo's output. A row whose selection holds no
/// valid token inside its selected length must be exactly +0 in the reference and
/// in every combo; any other row is compared per head with the tolerance
/// `1e-5 * maxabsref + 1e-7`, and the reference must be nonzero there.
#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &mut Gpu,
    dev: &Dev,
    sh: &Shape,
    q: &[f32],
    sel: &[Vec<i32>],
    combos: &[(usize, usize)],
    label: &str,
    stats: &mut Stats,
) -> Vec<Vec<f32>> {
    assert_eq!(sel.len(), sh.rows);
    assert_eq!(q.len(), sh.rows * 2 * NH * HD);
    let q_t = upload_q(gpu, q);
    let sel_t = upload_selected(gpu, sel);
    let n_out = sh.rows * NH * HD;
    let valid: Vec<bool> = (0..sh.rows)
        .map(|r| {
            let visible = sh.position_start + r + 1;
            let sel_len = selected_len(visible, sh.budget, sh.compress, sh.capacity);
            valid_count(&sel[r], sel_len, visible) > 0
        })
        .collect();

    let out_ref = sentinel(gpu, n_out);
    launch(gpu, Kind::Exact, dev, &q_t, &sel_t, &out_ref, sh);
    let want = download(gpu, &out_ref);
    let row_len = NH * HD;
    for r in 0..sh.rows {
        let row = &want[r * row_len..(r + 1) * row_len];
        if row.iter().any(|x| !x.is_finite()) {
            stats.fail(format!(
                "{label}: exact reference row {r} not finite/unwritten"
            ));
        } else if valid[r] && row.iter().all(|&x| x == 0.0) {
            stats.fail(format!(
                "{label}: exact reference row {r} is all zero despite valid tokens"
            ));
        } else if !valid[r] && row.iter().any(|x| x.to_bits() != 0) {
            stats.fail(format!(
                "{label}: exact reference row {r} is not +0 without valid tokens"
            ));
        }
    }

    let mut outs = Vec::with_capacity(combos.len());
    let mut fixture = Worst::default();
    for (c, &(s, l)) in combos.iter().enumerate() {
        let out = sentinel(gpu, n_out);
        launch(gpu, Kind::Split(s, l), dev, &q_t, &sel_t, &out, sh);
        let got = download(gpu, &out);
        let mut worst = Worst::default();
        for r in 0..sh.rows {
            if !valid[r] {
                if let Some(i) = (r * row_len..(r + 1) * row_len).find(|&i| got[i].to_bits() != 0) {
                    stats.fail(format!(
                        "{label} splits={s} layout={l}: row {r} has no valid token, expected +0, got bits {:#010x} at {}",
                        got[i].to_bits(),
                        i - r * row_len
                    ));
                }
                continue;
            }
            for h in 0..NH {
                let lo = (r * NH + h) * HD;
                let (g, w) = (&got[lo..lo + HD], &want[lo..lo + HD]);
                let maxabs = w.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
                let tol = 1e-5 * maxabs + 1e-7;
                let mut diff = 0.0f64;
                let mut bad = g.iter().any(|x| !x.is_finite());
                for (a, b) in g.iter().zip(w) {
                    diff = diff.max((*a as f64 - *b as f64).abs());
                }
                if !(diff <= tol) {
                    bad = true;
                }
                if bad {
                    stats.fail(format!(
                        "{label} splits={s} layout={l}: row {r} head {h} maxdiff={diff:.3e} > tol={tol:.3e} (maxabsref={maxabs:.3e})"
                    ));
                }
                worst.update(diff, tol);
            }
        }
        stats.worst[c].merge(worst);
        fixture.merge(worst);
        outs.push(got);
    }
    println!(
        "{label}: valid rows {}/{} worst maxdiff={:.3e} tol={:.3e} ratio={:.3}",
        valid.iter().filter(|&&v| v).count(),
        sh.rows,
        fixture.diff,
        fixture.tol,
        fixture.ratio
    );
    outs
}

// ---------------------------------------------------------------------------
// Single-row configurations
// ---------------------------------------------------------------------------

const ROWS1_CFGS: [&str; 4] = [
    "c1/huge/cap=len",
    "c8/huge/cap=len+5",
    "c8/budget=blocks-1",
    "c8/budget=blocks/3/clamped-cap",
];

/// (compress, budget, capacity) of single-row configuration `cfg` at `len` visible keys.
fn rows1_cfg(cfg: usize, len: usize) -> (usize, usize, usize) {
    let blocks = len / 8;
    match cfg {
        0 => (1, HUGE, len),
        1 => (8, HUGE, len + 5),
        2 => (8, blocks.saturating_sub(1), len),
        _ => {
            let budget = blocks / 3;
            let unclamped = selected_len(len, budget, 8, usize::MAX);
            (8, budget, unclamped.saturating_sub(1).max(1))
        }
    }
}

fn rows1_fixture(
    gpu: &mut Gpu,
    dev: &Dev,
    combos: &[(usize, usize)],
    len: usize,
    cfg: usize,
    mode: Mode,
    stats: &mut Stats,
) {
    let (compress, budget, capacity) = rows1_cfg(cfg, len);
    let sh = Shape {
        rows: 1,
        position_start: len - 1,
        budget,
        compress,
        capacity,
    };
    let mut rng = Rng(0x1000_0000 + (len * 31 + cfg * 7) as u64 * 977 + mode as u64);
    let q = make_q(&mut rng, 1);
    let sel_len = selected_len(len, budget, compress, capacity);
    let row = gen_row(&mut rng, mode, len, sel_len, capacity);
    let label = format!(
        "rows1 len={len:<4} {:<31} {mode:?} sel_len={sel_len}",
        ROWS1_CFGS[cfg]
    );
    check(gpu, dev, &sh, &q, &[row], combos, &label, stats);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_rows1_matches_exact() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "arch={} default splits={} default layout={}",
        gpu.arch, QSA_DECODE_SPLITK_SPLITS, QSA_DECODE_SPLITK_SCORING_LAYOUT
    );
    let dev = make_dev(&mut gpu);
    let combos = combos();
    let mut stats = Stats::new(combos.len());
    for len in lengths(&combos) {
        for cfg in 0..ROWS1_CFGS.len() {
            for mode in [Mode::AllValid, Mode::Mixed] {
                rows1_fixture(&mut gpu, &dev, &combos, len, cfg, mode, &mut stats);
            }
        }
    }
    stats.finish(&combos);
}

// ---------------------------------------------------------------------------
// Batched configurations
// ---------------------------------------------------------------------------

struct BatchCfg {
    name: &'static str,
    compress: usize,
    budget: usize,
    position_start: usize,
    capacity: usize,
}

/// `MAX_BATCH_ROWS`-row windows (`rows` 1..=32 take the prefix) with heterogeneous
/// `selected_len`: S > len, budget at the block boundary (selected_len drops when a
/// block completes), empty slices, capacity clamp, and the long contexts.
const BATCH_CFGS: [BatchCfg; 9] = [
    BatchCfg {
        name: "c8 b2 ps14 cap64",
        compress: 8,
        budget: 2,
        position_start: 14,
        capacity: 64,
    },
    BatchCfg {
        name: "c8 b2 ps20 cap64 (drop at block 3)",
        compress: 8,
        budget: 2,
        position_start: 20,
        capacity: 64,
    },
    BatchCfg {
        name: "c8 huge ps0 cap16 (S>len)",
        compress: 8,
        budget: HUGE,
        position_start: 0,
        capacity: 16,
    },
    BatchCfg {
        name: "c8 b0 ps0 cap16 (empty slice)",
        compress: 8,
        budget: 0,
        position_start: 0,
        capacity: 16,
    },
    BatchCfg {
        name: "c8 b0 ps6 cap16 (empty slice)",
        compress: 8,
        budget: 0,
        position_start: 6,
        capacity: 16,
    },
    BatchCfg {
        name: "c1 b5 ps0 cap8",
        compress: 1,
        budget: 5,
        position_start: 0,
        capacity: 8,
    },
    BatchCfg {
        name: "c8 b140 ps1124 cap1200",
        compress: 8,
        budget: 140,
        position_start: 1124,
        capacity: 1200,
    },
    BatchCfg {
        name: "c8 huge ps8188 cap8200",
        compress: 8,
        budget: HUGE,
        position_start: 8188,
        capacity: 8200,
    },
    BatchCfg {
        name: "c8 huge ps20 cap24 (clamp)",
        compress: 8,
        budget: HUGE,
        position_start: 20,
        capacity: 24,
    },
];

/// Row `r` mode: row 1 is all invalid (so every batch of 2+ rows has one),
/// every third row from 2 is pure valid, the rest mixed.
fn row_mode(r: usize, force: Option<Mode>) -> Mode {
    if let Some(m) = force {
        return m;
    }
    match r {
        1 => Mode::AllInvalid,
        _ if r % 3 == 2 => Mode::AllValid,
        _ => Mode::Mixed,
    }
}

/// Host data of a config's eight rows.
fn batch_host(cfg: &BatchCfg, force: Option<Mode>) -> (Vec<f32>, Vec<Vec<i32>>, Vec<usize>) {
    let mut rng = Rng(0x2000_0000
        + (cfg.position_start * 13 + cfg.compress) as u64 * 7919
        + cfg.budget as u64 % 1009);
    let q = make_q(&mut rng, MAX_BATCH_ROWS);
    let mut sel = Vec::new();
    let mut lens = Vec::new();
    for r in 0..MAX_BATCH_ROWS {
        let visible = cfg.position_start + r + 1;
        let sel_len = selected_len(visible, cfg.budget, cfg.compress, cfg.capacity);
        lens.push(sel_len);
        sel.push(gen_row(
            &mut rng,
            row_mode(r, force),
            visible,
            sel_len,
            cfg.capacity,
        ));
    }
    (q, sel, lens)
}

fn batch_shape(cfg: &BatchCfg, rows: usize) -> Shape {
    Shape {
        rows,
        position_start: cfg.position_start,
        budget: cfg.budget,
        compress: cfg.compress,
        capacity: cfg.capacity,
    }
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_batched_rows_bitwise_individual() {
    let mut gpu = Gpu::init().expect("gpu init");
    let dev = make_dev(&mut gpu);
    let combos = combos();
    let mut stats = Stats::new(combos.len());
    for cfg in &BATCH_CFGS {
        let (q, sel, lens) = batch_host(cfg, None);
        println!("== {} selected_len per row {lens:?}", cfg.name);
        assert!(
            lens.iter().any(|&l| l != lens[0]),
            "{}: selected_len must be heterogeneous: {lens:?}",
            cfg.name
        );
        // The single-row tensors of all MAX_BATCH_ROWS rows.
        let singles: Vec<(GpuTensor, GpuTensor)> = (0..MAX_BATCH_ROWS)
            .map(|r| {
                let q_row = &q[r * 2 * NH * HD..(r + 1) * 2 * NH * HD];
                (
                    upload_q(&mut gpu, q_row),
                    upload_selected(&gpu, &sel[r..r + 1]),
                )
            })
            .collect();
        for rows in 1..=MAX_BATCH_ROWS {
            let sh = batch_shape(cfg, rows);
            let label = format!("batched {} rows={rows}", cfg.name);
            let outs = check(
                &mut gpu,
                &dev,
                &sh,
                &q[..rows * 2 * NH * HD],
                &sel[..rows],
                &combos,
                &label,
                &mut stats,
            );
            let row_len = NH * HD;
            for (c, &(s, l)) in combos.iter().enumerate() {
                for r in 0..rows {
                    let single = Shape {
                        rows: 1,
                        position_start: cfg.position_start + r,
                        ..sh
                    };
                    let out = sentinel(&mut gpu, row_len);
                    launch(
                        &mut gpu,
                        Kind::Split(s, l),
                        &dev,
                        &singles[r].0,
                        &singles[r].1,
                        &out,
                        &single,
                    );
                    let alone = download(&gpu, &out);
                    let batched = &outs[c][r * row_len..(r + 1) * row_len];
                    if alone.iter().any(|x| !x.is_finite()) {
                        stats.fail(format!(
                            "{label} splits={s} layout={l}: single-row launch {r} not finite"
                        ));
                    }
                    if let Some(i) =
                        (0..row_len).find(|&i| alone[i].to_bits() != batched[i].to_bits())
                    {
                        stats.fail(format!(
                            "{label} splits={s} layout={l}: row {r} differs from its single-row launch at {i}: batched {:#010x} single {:#010x}",
                            batched[i].to_bits(),
                            alone[i].to_bits()
                        ));
                    }
                }
            }
        }
    }
    stats.finish(&combos);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_all_invalid_zero() {
    let mut gpu = Gpu::init().expect("gpu init");
    let dev = make_dev(&mut gpu);
    let combos = combos();
    let mut stats = Stats::new(combos.len());
    for len in lengths(&combos) {
        for cfg in 0..ROWS1_CFGS.len() {
            rows1_fixture(
                &mut gpu,
                &dev,
                &combos,
                len,
                cfg,
                Mode::AllInvalid,
                &mut stats,
            );
        }
    }
    for cfg in &BATCH_CFGS {
        let (q, sel, _) = batch_host(cfg, Some(Mode::AllInvalid));
        for rows in 1..=MAX_BATCH_ROWS {
            let sh = batch_shape(cfg, rows);
            let label = format!("all-invalid batched {} rows={rows}", cfg.name);
            check(
                &mut gpu,
                &dev,
                &sh,
                &q[..rows * 2 * NH * HD],
                &sel[..rows],
                &combos,
                &label,
                &mut stats,
            );
        }
    }
    stats.finish(&combos);
}

// ---------------------------------------------------------------------------
// Wide verify: production geometry, n = 1..=32 rows, 8K..256K end depths
// ---------------------------------------------------------------------------

const PROD_COMPRESS: usize = 4;
const PROD_BUDGET: usize = 512;
const PROD_CAPACITY: usize = 2051;
const WIDE_ROWS_RAGGED: [usize; 12] = [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32];
/// (splits, layout) compared bitwise batched vs single-row: the production
/// S32/layout 2, lab S16/S48 of layout 2 and the per-head layouts 0/1 at S8.
const WIDE_COMBOS: [(usize, usize); 5] = [(32, 2), (16, 2), (48, 2), (8, 0), (8, 1)];

/// Row `r`'s selection kind: mostly production-shaped lists (-1 padded), some
/// with valid-looking tokens after the active length, some with invalid tokens
/// inside it, every eleventh with no valid token at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WideRow {
    Padded,
    Garbage,
    Mixed,
    AllInvalid,
}

fn wide_row_kind(r: usize) -> WideRow {
    if r % 11 == 10 {
        return WideRow::AllInvalid;
    }
    match r % 4 {
        1 => WideRow::Garbage,
        2 => WideRow::Mixed,
        _ => WideRow::Padded,
    }
}

/// A token the kernel must reject: negative, in `[visible, full_cap)` or
/// `>= full_cap`.
fn wide_invalid_token(rng: &mut Rng, visible: usize, full_cap: usize) -> i32 {
    match rng.below(3) {
        0 => -1 - rng.below(100) as i32,
        1 if visible < full_cap => {
            (visible + rng.below((full_cap - visible).min(4096) as u64) as usize) as i32
        }
        _ => (full_cap + rng.below(1000) as usize) as i32,
    }
}

/// One production-geometry selected row for `visible` keys: `min(512, blocks)`
/// random blocks expanded in token order, then the causal tail, then padding.
fn gen_wide_row(rng: &mut Rng, kind: WideRow, visible: usize, full_cap: usize) -> Vec<i32> {
    let blocks = visible / PROD_COMPRESS;
    let chosen = PROD_BUDGET.min(blocks);
    let ell = selected_len(visible, PROD_BUDGET, PROD_COMPRESS, PROD_CAPACITY);
    let mut row = vec![-1i32; PROD_CAPACITY];
    let mut i = 0usize;
    for _ in 0..chosen {
        let block = rng.below(blocks as u64) as usize;
        for j in 0..PROD_COMPRESS {
            row[i] = (block * PROD_COMPRESS + j) as i32;
            i += 1;
        }
    }
    for t in blocks * PROD_COMPRESS..visible {
        row[i] = t as i32;
        i += 1;
    }
    assert_eq!(i, ell, "active length of visible={visible}");
    match kind {
        WideRow::Padded => {}
        WideRow::Garbage => {
            for slot in row.iter_mut().skip(ell) {
                *slot = valid_token(rng, visible);
            }
        }
        WideRow::Mixed => {
            for slot in row.iter_mut().take(ell) {
                if rng.below(10) == 0 {
                    *slot = wide_invalid_token(rng, visible, full_cap);
                }
            }
        }
        WideRow::AllInvalid => {
            for slot in row.iter_mut() {
                *slot = wide_invalid_token(rng, visible, full_cap);
            }
        }
    }
    row
}

/// K and V of `full_cap` rows, uploaded in chunks (no host copy of the cache).
fn make_dev_long(gpu: &mut Gpu, full_cap: usize) -> Dev {
    const CHUNK_ROWS: usize = 8192;
    let row = NKV * HD;
    let mut rng = Rng(0x5a5a_1001 ^ full_cap as u64);
    let mut tensors = Vec::new();
    for _ in 0..2 {
        let t = gpu
            .alloc_tensor(&[full_cap * row], DType::F32)
            .expect("alloc long K/V");
        let mut first = 0usize;
        while first < full_cap {
            let rows = CHUNK_ROWS.min(full_cap - first);
            let bytes: Vec<u8> = fill(&mut rng, rows * row, 1.0)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            gpu.hip
                .memcpy_htod_offset(&t.buf, first * row * 4, &bytes)
                .expect("upload K/V chunk");
            first += rows;
        }
        tensors.push(t);
    }
    let values = tensors.pop().unwrap();
    let keys = tensors.pop().unwrap();
    Dev {
        keys,
        values,
        full_cap,
    }
}

/// Sets cache rows `[first_row, first_row + rows)` of K and V huge; returns the
/// saved bytes for [`restore_rows`].
fn perturb_rows(gpu: &mut Gpu, dev: &Dev, first_row: usize, rows: usize) -> [Vec<u8>; 2] {
    let row_bytes = NKV * HD * 4;
    let huge: Vec<u8> = (0..rows * NKV * HD)
        .flat_map(|_| 3.0e18f32.to_le_bytes())
        .collect();
    let mut saved = [vec![0u8; rows * row_bytes], vec![0u8; rows * row_bytes]];
    for (slot, t) in [&dev.keys, &dev.values].into_iter().enumerate() {
        gpu.hip
            .memcpy_dtoh_at(&mut saved[slot], &t.buf, first_row * row_bytes)
            .expect("save rows");
        gpu.hip
            .memcpy_htod_offset(&t.buf, first_row * row_bytes, &huge)
            .expect("perturb rows");
    }
    saved
}

fn restore_rows(gpu: &mut Gpu, dev: &Dev, first_row: usize, saved: &[Vec<u8>; 2]) {
    let row_bytes = NKV * HD * 4;
    for (slot, t) in [&dev.keys, &dev.values].into_iter().enumerate() {
        gpu.hip
            .memcpy_htod_offset(&t.buf, first_row * row_bytes, &saved[slot])
            .expect("restore rows");
    }
}

/// One window `[e_end - n, e_end)`: every combo's batched launch against the n
/// single-row launches (bitwise), no-valid-token rows exactly +0, and the
/// future-cache perturbation. Returns the number of combo cases.
fn wide_splitk_window(
    gpu: &mut Gpu,
    dev: &Dev,
    e_end: usize,
    n: usize,
    fails: &mut Vec<String>,
) -> usize {
    let ps = e_end - n;
    let sh = Shape {
        rows: n,
        position_start: ps,
        budget: PROD_BUDGET,
        compress: PROD_COMPRESS,
        capacity: PROD_CAPACITY,
    };
    let mut rng = Rng(0x3000_0000 + (e_end * 131 + n) as u64 * 7919);
    let q = make_q(&mut rng, n);
    let sel: Vec<Vec<i32>> = (0..n)
        .map(|r| gen_wide_row(&mut rng, wide_row_kind(r), ps + r + 1, dev.full_cap))
        .collect();
    let q_row = 2 * NH * HD;
    let row_len = NH * HD;
    let q_t = upload_q(gpu, &q);
    let sel_t = upload_selected(gpu, &sel);
    let singles: Vec<(GpuTensor, GpuTensor)> = (0..n)
        .map(|r| {
            (
                upload_q(gpu, &q[r * q_row..(r + 1) * q_row]),
                upload_selected(gpu, &sel[r..r + 1]),
            )
        })
        .collect();
    let mut kinds: Vec<Kind> = WIDE_COMBOS
        .iter()
        .map(|&(s, l)| Kind::Split(s, l))
        .collect();
    kinds.push(Kind::Production);
    let mut cases = 0usize;
    let mut base_out: Option<Vec<f32>> = None;
    for kind in kinds {
        let label = format!("wide E={e_end} n={n} {kind:?}");
        let out = sentinel(gpu, n * row_len);
        launch(gpu, kind, dev, &q_t, &sel_t, &out, &sh);
        let got = download(gpu, &out);
        gpu.free_tensor(out).expect("free out");
        for r in 0..n {
            let single = Shape {
                rows: 1,
                position_start: ps + r,
                ..sh
            };
            let o1 = sentinel(gpu, row_len);
            launch(gpu, kind, dev, &singles[r].0, &singles[r].1, &o1, &single);
            let alone = download(gpu, &o1);
            gpu.free_tensor(o1).expect("free single out");
            let batched = &got[r * row_len..(r + 1) * row_len];
            if let Some(i) = (0..row_len).find(|&i| alone[i].to_bits() != batched[i].to_bits()) {
                fails.push(format!(
                    "{label}: row {r} differs from its single-row launch at {i}: batched {:#010x} single {:#010x}",
                    batched[i].to_bits(),
                    alone[i].to_bits()
                ));
            }
            if wide_row_kind(r) == WideRow::AllInvalid {
                if batched.iter().any(|x| x.to_bits() != 0) {
                    fails.push(format!("{label}: all-invalid row {r} is not +0"));
                }
            } else if batched.iter().any(|x| !x.is_finite()) {
                fails.push(format!("{label}: row {r} not finite"));
            }
        }
        if matches!(kind, Kind::Split(32, 2)) {
            base_out = Some(got);
        }
        cases += 1;
    }
    // Cache rows no earlier row may read: rows [0, k) see only cache rows below
    // `ps + k`, so they must not change when rows `ps + k..e_end` become huge.
    if n >= 2 {
        let k = n / 2;
        let first = ps + k;
        let saved = perturb_rows(gpu, dev, first, e_end - first);
        let out = sentinel(gpu, n * row_len);
        launch(gpu, Kind::Split(32, 2), dev, &q_t, &sel_t, &out, &sh);
        let perturbed = download(gpu, &out);
        gpu.free_tensor(out).expect("free out");
        restore_rows(gpu, dev, first, &saved);
        let base = base_out.expect("production split combo ran");
        for r in 0..k {
            let range = r * row_len..(r + 1) * row_len;
            if let Some(i) = range
                .clone()
                .find(|&i| perturbed[i].to_bits() != base[i].to_bits())
            {
                fails.push(format!(
                    "wide E={e_end} n={n}: row {r} changed at {} when only cache rows >= {first} changed",
                    i - r * row_len
                ));
            }
        }
        cases += 1;
    }
    gpu.free_tensor(q_t).expect("free q");
    gpu.free_tensor(sel_t).expect("free selected");
    for (a, b) in singles {
        gpu.free_tensor(a).expect("free single q");
        gpu.free_tensor(b).expect("free single selected");
    }
    cases
}

fn wide_splitk_depth(depth: usize) {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={} depth={depth}", gpu.arch);
    let dev = make_dev_long(&mut gpu, depth);
    let mut fails = Vec::new();
    let (mut windows, mut cases) = (0usize, 0usize);
    for disp in 0..PROD_COMPRESS {
        let e_end = depth - disp;
        let ns: Vec<usize> = if disp == 0 {
            (1..=WIDE_VERIFY_MAX_ROWS).collect()
        } else {
            WIDE_ROWS_RAGGED.to_vec()
        };
        for n in ns {
            cases += wide_splitk_window(&mut gpu, &dev, e_end, n, &mut fails);
            windows += 1;
        }
    }
    println!("depth {depth}: {windows} windows, {cases} combo cases");
    if !fails.is_empty() {
        let shown: Vec<&str> = fails.iter().take(40).map(String::as_str).collect();
        panic!(
            "{} failure(s); first {}:\n{}",
            fails.len(),
            shown.len(),
            shown.join("\n")
        );
    }
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_wide_rows_8k_bitwise_individual() {
    wide_splitk_depth(8192);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_wide_rows_32k_bitwise_individual() {
    wide_splitk_depth(32768);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_wide_rows_128k_bitwise_individual() {
    wide_splitk_depth(131072);
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_wide_rows_256k_bitwise_individual() {
    wide_splitk_depth(262144);
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Median over `SAMPLES` samples of `CALLS_PER_SAMPLE` back-to-back calls timed
/// with hipEvents on the default stream, in microseconds per call. `call` runs
/// `WARMUP_CALLS` times first (JIT compile, scratch allocation, caches).
fn time_us(gpu: &mut Gpu, mut call: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..WARMUP_CALLS {
        call(gpu);
    }
    let start = gpu.hip.event_create().expect("event_create start");
    let stop = gpu.hip.event_create().expect("event_create stop");
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        gpu.hip
            .event_record(&start, None)
            .expect("event_record start");
        for _ in 0..CALLS_PER_SAMPLE {
            call(gpu);
        }
        gpu.hip
            .event_record(&stop, None)
            .expect("event_record stop");
        gpu.hip.event_synchronize(&stop).expect("event_synchronize");
        let ms = gpu
            .hip
            .event_elapsed_ms(&start, &stop)
            .expect("event_elapsed_ms");
        samples.push(ms as f64 * 1000.0 / CALLS_PER_SAMPLE as f64);
    }
    gpu.hip.event_destroy(start).expect("event_destroy start");
    gpu.hip.event_destroy(stop).expect("event_destroy stop");
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[SAMPLES / 2]
}

#[test]
#[ignore = "needs a HIP device"]
fn qsa_splitk_timing_table() {
    assert_eq!(
        std::env::var("HIPFIRE_QWEN4_QSA_DECODE_SPLITK").as_deref(),
        Ok("0"),
        "HIPFIRE_QWEN4_QSA_DECODE_SPLITK must be 0: split-K is the default on gfx1151, and the production-dispatch row is the incumbent route only with it off"
    );
    let mut gpu = Gpu::init().expect("gpu init");
    println!(
        "arch={} FABRIC_JOB={} (timings: {CALLS_PER_SAMPLE} calls/sample, median of {SAMPLES}, \
         warmup {WARMUP_CALLS}; includes host launch cost; GPU may be non-exclusive)",
        gpu.arch,
        std::env::var("FABRIC_JOB").unwrap_or_else(|_| "<unset>".into())
    );
    let hipfire_env: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("HIPFIRE_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    println!("HIPFIRE_* environment: {hipfire_env:?}");
    println!(
        "default splits={} default layout={}",
        QSA_DECODE_SPLITK_SPLITS, QSA_DECODE_SPLITK_SCORING_LAYOUT
    );
    let dev = make_dev(&mut gpu);
    let combos = combos();
    println!("len,variant,splits,layout,us_per_call,vs_exact,k_bytes,v_bytes,kv_bytes");
    // 8188 is the longest context whose exact kernel still fits the 64 KiB
    // dynamic LDS (8 * len + 32 <= 65536) and so runs the parallel kernel. 8195
    // exceeds it and the exact path falls to f32_batched_serial, so it is not
    // timed here (it stays in the correctness tests).
    for len in [1131usize, 8188] {
        let exact_kernel = if 8 * len + 32 <= 64 * 1024 {
            "parallel"
        } else {
            "serial"
        };
        let sh = Shape {
            rows: 1,
            position_start: len - 1,
            budget: HUGE,
            compress: 8,
            capacity: len,
        };
        let mut rng = Rng(0x3000_0000 + len as u64);
        let q = make_q(&mut rng, 1);
        let sel_len = selected_len(len, sh.budget, sh.compress, sh.capacity);
        assert_eq!(
            sel_len, len,
            "timing rows are all valid with selected_len == len"
        );
        let row = gen_row(&mut rng, Mode::AllValid, len, sel_len, sh.capacity);
        let q_t = upload_q(&mut gpu, &q);
        let sel_t = upload_selected(&gpu, &[row]);
        let out = sentinel(&mut gpu, NH * HD);
        // Modelled K/V bytes per call, all tokens valid, excluding Q, output and
        // workspace. The incumbent and the head layouts read K and V once per
        // query head; the grouped layout once per KV head.
        let (hk, hv) = modelled_kv_bytes(0, len);
        let (gk, gv) = modelled_kv_bytes(2, len);
        let exact = time_us(&mut gpu, |g| {
            launch(g, Kind::Exact, &dev, &q_t, &sel_t, &out, &sh)
        });
        println!(
            "{len},exact-{exact_kernel} (incumbent),-,-,{exact:.2},1.00x,{hk},{hv},{}",
            hk + hv
        );
        let production = time_us(&mut gpu, |g| {
            launch(g, Kind::Production, &dev, &q_t, &sel_t, &out, &sh)
        });
        println!(
            "{len},production-dispatch,-,-,{production:.2},{:.2}x,{hk},{hv},{}",
            exact / production,
            hk + hv
        );
        // Best (lowest us) head-layout and grouped-layout combo.
        let mut best_head: Option<(usize, usize, f64)> = None;
        let mut best_group: Option<(usize, usize, f64)> = None;
        for &(s, l) in &combos {
            let us = time_us(&mut gpu, |g| {
                launch(g, Kind::Split(s, l), &dev, &q_t, &sel_t, &out, &sh)
            });
            let (k, v) = modelled_kv_bytes(l, len);
            println!(
                "{len},splitk,{s},{l},{us:.2},{:.2}x,{k},{v},{}",
                exact / us,
                k + v
            );
            let best = if l == 2 {
                &mut best_group
            } else {
                &mut best_head
            };
            if best.map_or(true, |(_, _, b)| us < b) {
                *best = Some((s, l, us));
            }
        }
        println!(
            "len {len} modelled K/V bytes per call: head layouts K={hk} V={hv} total={}; \
             group layout K={gk} V={gv} total={} ({}x fewer)",
            hk + hv,
            gk + gv,
            (hk + hv) / (gk + gv)
        );
        match (best_head, best_group) {
            (Some((hs, hl, hu)), Some((gs, gl, gu))) => println!(
                "len {len} best head: splits={hs} layout={hl} {hu:.2} us; best group: splits={gs} layout={gl} {gu:.2} us; \
                 group/head time ratio {:.3}",
                gu / hu
            ),
            _ => println!("len {len}: head or group combos missing"),
        }
    }
}

/// Modelled K and V bytes read per call (F32, all `len` tokens valid) of scoring
/// `layout`: layouts 0/1 stream K and V once per query head (24 * len * 256 * 4
/// each), the grouped layout 2 once per KV head (2 * len * 256 * 4 each).
fn modelled_kv_bytes(layout: usize, len: usize) -> (usize, usize) {
    let heads = if layout == 2 { NKV } else { NH };
    let each = heads * len * HD * std::mem::size_of::<f32>();
    (each, each)
}
