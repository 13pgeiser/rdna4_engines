// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle, CPU structural proof and timing harness for the two-rows-per-block
//! fused prefill MoE router (`Gpu::moe_router_softmax_topk_k8_multirow_prefill`,
//! block 128 x 2, A3B shape: 256 experts, top-8).
//!
//! Three routes are compared on identical inputs (n_exp 256, k 8):
//! * incumbent pair: `gpu.softmax_f32(&logits)` + `gpu.moe_topk_renorm_k8_batched(..)`.
//! * current fused: `gpu.moe_router_softmax_topk_k8_prefill(.., lanes = 128)`.
//! * new: `gpu.moe_router_softmax_topk_k8_multirow_prefill(..)` (two rows per
//!   block, 128 lanes each, grid `batch / 2`).
//!
//! Modes (one per invocation):
//! * `--plan` (CPU, no GPU): prints the launch geometry, LDS arithmetic, the
//!   admission table, and the exact GPU case grid with an FNV-1a/64 hash of every
//!   deterministic input, so the owner run can be matched to the plan.
//! * `--self-check` (CPU, no GPU): proves the virtualization structure without
//!   libm/GPU exp assumptions. An independent serial reference of the original
//!   256-thread softmax + mode-0 top-8 (strict `>` scan, tree s = 128..1,
//!   replace-on-strictly-greater, sequential renorm, clamp) is compared bit for
//!   bit with a lane-level emulation of the new kernel (128 lanes x 2 row slots,
//!   `for t = tid; t < width; t += 128` virtual-thread loops, per-row separate
//!   storage, `bid = block * 2 + ty`). Both sides use the same CPU `exp`/division,
//!   so only the structure (trees, ties, lane assignment, row layout) is under
//!   test. Also checks: every virtual thread is visited exactly once by lane
//!   `t % 128` in every stage, tree reads/writes are disjoint, the row/block/grid
//!   layout, the LDS byte budget, hand-derived tie orders (all-zero picks
//!   0,128,64,192; {5,133} -> 5,133; {1,2} -> 2,1: tree position, not lowest
//!   index) and the pure admission rule.
//! * default / `--gpu` (wave32 GPU, one process): rows {512, 4096, 8192} x n_exp 256
//!   x norm {false, true} x inputs {allzero, random, tie_heavy, tie_tree_cross,
//!   mixed_rows}. Ids (raw i32 in F32-typed tensors) and weights are compared by
//!   raw bit pattern: new vs current fused AND new vs incumbent (and, for the
//!   record, current fused vs incumbent). Then per rows x norm the event-timed
//!   median over 50 reps (10 warmups, >= 20 required) of each route on the random
//!   input. Every rep first restores the raw logits (softmax is in place) from a
//!   GPU-resident pristine copy outside the event interval.
//!
//! Output (stdout, one JSON object per line): per-comparison verdict lines, timing
//! lines `{"timing":true,"rows","norm","median_us":{"incumbent_pair","current_fused","new"},..}`
//! and a final summary line containing `all_byte_identical` (bool) and
//! `first_mismatch` (null or an object with comparison, rows, variant, norm, row,
//! pick, ids and weight hex bit patterns). Exit code: 0 all pass; 1 mismatch;
//! 2 unusable device/arguments.
//!
//! Commands:
//! * CPU: `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle -- --plan`
//!   `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle -- --self-check`
//! * GPU (owner, wave32 device): `cargo run --release -p rdna-compute --features lab --example moe_router_topk_multirow_oracle`
//!
//! The runtime flag `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW` is intentionally not
//! consulted: the raw launcher is exercised directly.

use rdna_compute::{Gpu, GpuTensor};
use std::time::Instant;

const TOP_K: usize = 8;
const N_EXP: usize = 256;
/// Virtual block of the incumbent softmax (`min(256, n_exp)`) and top-8 kernels.
const VBLOCK: usize = 256;
/// Physical lanes per row slot and rows per block of the new route.
const LANES: usize = 128;
const ROWS_PER_BLOCK: usize = 2;
/// Static LDS of the shared symbol (row slot 0): smem_v + smem_i (1024 each),
/// reduce_v + reduce_i (256 each), picked_idx + picked_w (8 each), 4 bytes each.
const STATIC_LDS_BYTES: usize = 4 * (1024 + 1024 + 256 + 256 + 8 + 8);
/// Dynamic LDS of row slot 1 at n_exp 256: smem_v[n_exp], reduce_v[256],
/// reduce_i[256], picked_idx[8], picked_w[8]; smem_i is mode 0 only.
const DYN_LDS_BYTES: usize = 4 * (N_EXP + 256 + 256 + 8 + 8);

const GPU_ROWS: [usize; 3] = [512, 4096, 8192];
const VARIANTS: [&str; 5] = [
    "allzero",
    "random",
    "tie_heavy",
    "tie_tree_cross",
    "mixed_rows",
];
const MIXED_KINDS: [&str; 4] = ["allzero", "random", "tie_heavy", "tie_tree_cross"];
/// Equal-maximum position sets that cross virtual tree positions (n_exp 256: one
/// element per virtual thread, so the position is the virtual thread).
const TREE_SETS: [&[usize]; 10] = [
    &[0, 128],
    &[1, 2],
    &[5, 133],
    &[127, 255],
    &[64, 192],
    &[31, 32, 159, 160],
    &[0, 64, 128, 192],
    &[3, 67, 131, 195, 7, 71, 135, 199],
    &[255, 254, 127, 126],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 130, 131, 132],
];
/// Poison pattern for unwritten output slots (not a valid raw i32 expert id).
const SENTINEL_BITS: u32 = 0xdead_beef;

const WARMUPS: usize = 10;
const REPS: usize = 50;
const _: () = assert!(REPS >= 20);
const BUDGET_S: f64 = 120.0;
/// Stop starting new timing groups past this elapsed time (seconds).
const TIMING_CUTOFF_S: f64 = 100.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [a] if a == "--plan" => plan(),
        [a] if a == "--self-check" => self_check(),
        [] => gpu_mode(),
        [a] if a == "--gpu" => gpu_mode(),
        _ => {
            eprintln!("usage: moe_router_topk_multirow_oracle [--plan | --self-check | --gpu]");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Admission (pure) shared by --plan, --self-check and the GPU mode.
// ---------------------------------------------------------------------------

fn expected_admission(
    enabled: bool,
    k: usize,
    n_exp: usize,
    batch: usize,
    has_wave32: bool,
) -> bool {
    enabled && k == 8 && n_exp == 256 && has_wave32 && batch >= 2 && batch.is_multiple_of(2)
}

fn check_admission() -> Result<usize, String> {
    let mut checked = 0usize;
    for enabled in [false, true] {
        for k in [1usize, 4, 8, 10] {
            for n_exp in [64usize, 128, 255, 256, 257, 512, 1024] {
                for batch in [0usize, 1, 2, 3, 4, 7, 512, 4095, 4096, 8192, 8193] {
                    for wave32 in [false, true] {
                        let got = Gpu::moe_router_topk_multirow_admitted(
                            enabled, k, n_exp, batch, wave32,
                        );
                        let want = expected_admission(enabled, k, n_exp, batch, wave32);
                        checked += 1;
                        if got != want {
                            return Err(format!(
                                "admission mismatch enabled={enabled} k={k} n_exp={n_exp} \
                                 batch={batch} wave32={wave32}: got {got}, want {want}"
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(checked)
}

// ---------------------------------------------------------------------------
// --plan
// ---------------------------------------------------------------------------

fn plan() {
    println!(
        "{{\"plan\":true,\"geometry\":{{\"lanes\":{LANES},\"rows_per_block\":{ROWS_PER_BLOCK},\
         \"block\":[{LANES},{ROWS_PER_BLOCK},1],\"threads\":{},\"n_exp\":{N_EXP},\"k\":{TOP_K},\
         \"static_lds_bytes\":{STATIC_LDS_BYTES},\"dynamic_lds_bytes\":{DYN_LDS_BYTES},\
         \"total_lds_bytes\":{},\"virtual_block\":{VBLOCK},\"virtual_threads_per_lane\":{}}}}}",
        LANES * ROWS_PER_BLOCK,
        STATIC_LDS_BYTES + DYN_LDS_BYTES,
        VBLOCK / LANES,
    );
    for &rows in &GPU_ROWS {
        println!(
            "{{\"plan_grid\":true,\"rows\":{rows},\"grid\":[{}, 1, 1],\"admitted_wave32\":{}}}",
            rows / ROWS_PER_BLOCK,
            expected_admission(true, TOP_K, N_EXP, rows, true)
        );
    }
    match check_admission() {
        Ok(n) => println!("{{\"plan_admission\":true,\"combinations_checked\":{n},\"pass\":true}}"),
        Err(e) => {
            println!("{{\"plan_admission\":true,\"pass\":false,\"error\":\"{e}\"}}");
            std::process::exit(1);
        }
    }
    let mut cases = 0usize;
    for &rows in &GPU_ROWS {
        for variant in VARIANTS {
            let logits = make_logits(variant, rows);
            let hash = fnv1a(&logits);
            for norm in [false, true] {
                cases += 1;
                println!(
                    "{{\"plan_case\":true,\"rows\":{rows},\"n_exp\":{N_EXP},\"variant\":\"{variant}\",\
                     \"norm\":{norm},\"input_fnv1a64\":\"{hash:016x}\",\
                     \"comparisons\":[\"new_vs_current_fused\",\"new_vs_incumbent\",\"current_fused_vs_incumbent\"]}}"
                );
            }
        }
    }
    println!(
        "{{\"plan_summary\":true,\"cases\":{cases},\"gated_comparisons\":{},\"timing_groups\":{},\
         \"timing_warmups\":{WARMUPS},\"timing_reps\":{REPS},\"budget_s\":{BUDGET_S}}}",
        cases * 2,
        GPU_ROWS.len() * 2
    );
}

// ---------------------------------------------------------------------------
// --self-check: CPU structural proof
// ---------------------------------------------------------------------------

fn self_check() {
    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;
    fn record(name: &str, r: Result<(), String>, failures: &mut Vec<String>) {
        let pass = r.is_ok();
        println!("{{\"self_check\":\"{name}\",\"pass\":{pass}}}");
        if let Err(e) = r {
            eprintln!("FAIL {name}: {e}");
            failures.push(format!("{name}: {e}"));
        }
    }

    record("lds_budget", check_lds(), &mut failures);
    checks += 1;
    record("virtual_thread_coverage", check_coverage(), &mut failures);
    checks += 1;
    record("row_block_layout", check_layout(), &mut failures);
    checks += 1;
    record(
        "admission_rule",
        check_admission().map(|_| ()),
        &mut failures,
    );
    checks += 1;
    record("hand_derived_ties", check_hand_ties(), &mut failures);
    checks += 1;

    for variant in VARIANTS {
        for norm in [false, true] {
            let logits = make_logits(variant, 130);
            let name = format!("emulation_vs_reference/{variant}/norm={norm}");
            record(&name, check_emulation(&logits, 130, norm), &mut failures);
            checks += 1;
        }
    }
    // Identical rows in both slots of every block: slot storage is independent.
    for norm in [false, true] {
        let mut logits = make_logits("tie_tree_cross", 64);
        for pair in 0..32 {
            let (a, b) = (2 * pair * N_EXP, (2 * pair + 1) * N_EXP);
            let (head, tail) = logits.split_at_mut(b);
            tail[..N_EXP].copy_from_slice(&head[a..a + N_EXP]);
        }
        let name = format!("emulation_vs_reference/duplicate_pairs/norm={norm}");
        record(&name, check_emulation(&logits, 64, norm), &mut failures);
        checks += 1;
    }

    println!(
        "{{\"self_check_summary\":true,\"checks\":{checks},\"failures\":{},\"pass\":{}}}",
        failures.len(),
        failures.is_empty()
    );
    if !failures.is_empty() {
        std::process::exit(1);
    }
}

fn check_lds() -> Result<(), String> {
    if STATIC_LDS_BYTES != 10304 {
        return Err(format!("static LDS {STATIC_LDS_BYTES} != 10304"));
    }
    if DYN_LDS_BYTES != 3136 {
        return Err(format!("dynamic LDS {DYN_LDS_BYTES} != 3136"));
    }
    if LANES * ROWS_PER_BLOCK != 256 || !VBLOCK.is_multiple_of(LANES) {
        return Err("block geometry is not 128 x 2 over 256 virtual threads".into());
    }
    Ok(())
}

/// Every stage of the fused kernel loops `for t = tid; t < width; t += 128`.
/// Each virtual thread must be visited exactly once, by lane `t % 128`; tree
/// stages additionally read `[s, 2s)` and write `[0, s)` (disjoint).
fn check_coverage() -> Result<(), String> {
    let mut widths: Vec<(String, usize)> = Vec::new();
    for name in [
        "max_scan",
        "exp_sum_scan",
        "elementwise_divide",
        "top8_scan",
    ] {
        widths.push((name.to_string(), VBLOCK));
    }
    let mut s = VBLOCK >> 1;
    while s > 0 {
        widths.push((format!("tree_s{s}"), s));
        s >>= 1;
    }
    for (name, width) in widths {
        let mut visits = vec![0usize; width];
        for tid in 0..LANES {
            let mut t = tid;
            while t < width {
                if t % LANES != tid {
                    return Err(format!("{name}: virtual thread {t} on lane {tid}"));
                }
                visits[t] += 1;
                t += LANES;
            }
        }
        if let Some(t) = visits.iter().position(|&c| c != 1) {
            return Err(format!(
                "{name}: virtual thread {t} visited {} times",
                visits[t]
            ));
        }
    }
    let mut s = VBLOCK >> 1;
    while s > 0 {
        // write set [0, s), read set [s, 2s): disjoint by construction.
        if s > VBLOCK - s {
            return Err(format!("tree stage {s}: read/write sets overlap"));
        }
        s >>= 1;
    }
    Ok(())
}

/// `bid = blockIdx.x * 2 + threadIdx.y` over grid `batch / 2` covers every row
/// exactly once and every output slot `row * 8 + [0, 8)` exactly once.
fn check_layout() -> Result<(), String> {
    for batch in [2usize, 4, 6, 512, 4096, 8192] {
        let grid = batch / ROWS_PER_BLOCK;
        let mut seen = vec![0u8; batch];
        let mut out = vec![0u8; batch * TOP_K];
        for block in 0..grid {
            for ty in 0..ROWS_PER_BLOCK {
                let bid = block * ROWS_PER_BLOCK + ty;
                if bid >= batch {
                    return Err(format!("batch {batch}: bid {bid} out of range"));
                }
                seen[bid] += 1;
                for tid in 0..TOP_K {
                    out[bid * TOP_K + tid] += 1;
                }
            }
        }
        if seen.iter().any(|&c| c != 1) || out.iter().any(|&c| c != 1) {
            return Err(format!(
                "batch {batch}: rows/outputs not covered exactly once"
            ));
        }
    }
    Ok(())
}

fn check_hand_ties() -> Result<(), String> {
    let cases: [(&str, Vec<f32>, &[i32]); 3] = [
        ("allzero", vec![0.0; N_EXP], &[0, 128, 64, 192][..]),
        ("pair_5_133", pair_row(&[5, 133], 11), &[5, 133][..]),
        ("pair_1_2", pair_row(&[1, 2], 12), &[2, 1][..]),
    ];
    for (name, row, want) in cases {
        for norm in [false, true] {
            let mut logits = row.clone();
            logits.extend_from_slice(&row);
            let (ids, _) = reference_row(&reference_softmax(&logits[..N_EXP]), norm);
            if ids[..want.len()] != *want {
                return Err(format!(
                    "{name} norm={norm}: reference picks {:?}, expected prefix {want:?}",
                    &ids[..want.len()]
                ));
            }
            check_emulation(&logits, 2, norm).map_err(|e| format!("{name} norm={norm}: {e}"))?;
        }
    }
    Ok(())
}

/// Compares the lane-level emulation of the new kernel to the serial reference,
/// bit for bit, for every row of `logits` (`rows` even).
fn check_emulation(logits: &[f32], rows: usize, norm: bool) -> Result<(), String> {
    if !rows.is_multiple_of(ROWS_PER_BLOCK) || logits.len() != rows * N_EXP {
        return Err("rows must be even and logits sized rows * n_exp".into());
    }
    let mut idx = vec![i32::from_ne_bytes(SENTINEL_BITS.to_ne_bytes()); rows * TOP_K];
    let mut w = vec![f32::from_bits(SENTINEL_BITS); rows * TOP_K];
    for block in 0..rows / ROWS_PER_BLOCK {
        emulate_block(logits, block, norm, &mut idx, &mut w);
    }
    for row in 0..rows {
        let probs = reference_softmax(&logits[row * N_EXP..(row + 1) * N_EXP]);
        let (ref_ids, ref_w) = reference_row(&probs, norm);
        for pick in 0..TOP_K {
            let (gi, gw) = (idx[row * TOP_K + pick], w[row * TOP_K + pick]);
            if gi != ref_ids[pick] || gw.to_bits() != ref_w[pick].to_bits() {
                return Err(format!(
                    "row {row} pick {pick}: emulated id {gi} weight {:#010x}, reference id {} \
                     weight {:#010x}",
                    gw.to_bits(),
                    ref_ids[pick],
                    ref_w[pick].to_bits()
                ));
            }
        }
    }
    Ok(())
}

fn fmax(a: f32, b: f32) -> f32 {
    a.max(b)
}

fn fadd(a: f32, b: f32) -> f32 {
    a + b
}

/// Serial emulation of the incumbent `softmax_f32` (block `min(256, n)`):
/// per-thread strided max from -1e30, tree max, per-thread strided sum of exp,
/// tree sum, direct divide.
fn reference_softmax(row: &[f32]) -> Vec<f32> {
    let n = row.len();
    let vb = n.min(256);
    let mut red = vec![0.0f32; vb];
    for (t, slot) in red.iter_mut().enumerate() {
        let mut m = -1e30f32;
        let mut i = t;
        while i < n {
            m = fmax(m, row[i]);
            i += vb;
        }
        *slot = m;
    }
    let mut s = vb >> 1;
    while s > 0 {
        for t in 0..s {
            red[t] = fmax(red[t], red[t + s]);
        }
        s >>= 1;
    }
    let max_val = red[0];
    let mut ex = vec![0.0f32; n];
    for (t, slot) in red.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let mut i = t;
        while i < n {
            let e = (row[i] - max_val).exp();
            ex[i] = e;
            sum += e;
            i += vb;
        }
        *slot = sum;
    }
    let mut s = vb >> 1;
    while s > 0 {
        for t in 0..s {
            red[t] = fadd(red[t], red[t + s]);
        }
        s >>= 1;
    }
    let sum_all = red[0];
    ex.iter().map(|e| e / sum_all).collect()
}

/// Serial emulation of the original mode-0 `moe_topk_renorm_k8_batched`
/// (256 threads, padded smem, strict `>` scan, replace-on-strictly-greater tree,
/// sequential renorm sum, direct division, NaN/OOB clamp).
fn reference_row(probs: &[f32], norm: bool) -> ([i32; TOP_K], [f32; TOP_K]) {
    let n = probs.len();
    let mut smem_v = vec![f32::NEG_INFINITY; 1024];
    let mut smem_i = vec![0i32; 1024];
    for i in 0..1024 {
        smem_v[i] = if i < n { probs[i] } else { f32::NEG_INFINITY };
        smem_i[i] = i as i32;
    }
    let mut picked_idx = [0i32; TOP_K];
    let mut picked_w = [0.0f32; TOP_K];
    for k in 0..TOP_K {
        let mut rv = vec![0.0f32; 256];
        let mut ri = vec![0i32; 256];
        for tid in 0..256 {
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = -1i32;
            let mut i = tid;
            while i < n {
                let v = smem_v[i];
                if v > best_v {
                    best_v = v;
                    best_i = smem_i[i];
                }
                i += 256;
            }
            rv[tid] = best_v;
            ri[tid] = best_i;
        }
        let mut s = 128usize;
        while s > 0 {
            for t in 0..s {
                let (va, vb) = (rv[t], rv[t + s]);
                if vb > va {
                    rv[t] = vb;
                    ri[t] = ri[t + s];
                }
            }
            s >>= 1;
        }
        let winner = ri[0];
        picked_idx[k] = winner;
        picked_w[k] = rv[0];
        if winner >= 0 && (winner as usize) < n {
            smem_v[winner as usize] = f32::NEG_INFINITY;
        }
    }
    let mut s = 0.0f32;
    for w in picked_w.iter() {
        s += *w;
    }
    if norm && s > 0.0 {
        for w in picked_w.iter_mut() {
            *w /= s;
        }
    }
    for k in 0..TOP_K {
        if picked_idx[k] < 0 || picked_idx[k] as usize >= n {
            picked_idx[k] = 0;
            picked_w[k] = 0.0;
        }
    }
    (picked_idx, picked_w)
}

/// Per-row-slot LDS: the original names (smem_v / reduce_v / reduce_i /
/// picked_idx / picked_w) aliased to one row's storage. Slot 0 is the static
/// array set, slot 1 the dynamic block; they never alias each other.
struct RowLds {
    smem_v: Vec<f32>,
    reduce_v: Vec<f32>,
    reduce_i: Vec<i32>,
    picked_idx: [i32; TOP_K],
    picked_w: [f32; TOP_K],
}

impl RowLds {
    fn new() -> Self {
        RowLds {
            smem_v: vec![0.0; N_EXP],
            reduce_v: vec![0.0; VBLOCK],
            reduce_i: vec![0; VBLOCK],
            picked_idx: [0; TOP_K],
            picked_w: [0.0; TOP_K],
        }
    }
}

/// Executes one block (128 lanes x 2 row slots) of the fused path stage by
/// stage. Every `for ty { for tid {..} }` nest is one barrier-delimited stage;
/// lanes run serially, which is exact because within a stage no lane reads
/// anything another lane writes (checked in `check_coverage`). Virtual thread
/// loops are `for t = tid; t < width; t += 128`, as in the kernel.
fn emulate_block(logits: &[f32], block: usize, norm: bool, out_idx: &mut [i32], out_w: &mut [f32]) {
    let mut st = [RowLds::new(), RowLds::new()];
    let probs_of = |ty: usize| {
        let bid = block * ROWS_PER_BLOCK + ty;
        &logits[bid * N_EXP..(bid + 1) * N_EXP]
    };

    // Row max scan.
    for (ty, row) in st.iter_mut().enumerate() {
        let probs = probs_of(ty);
        for tid in 0..LANES {
            for t in (tid..VBLOCK).step_by(LANES) {
                let mut m = -1e30f32;
                let mut i = t;
                while i < N_EXP {
                    m = fmax(m, probs[i]);
                    i += VBLOCK;
                }
                row.reduce_v[t] = m;
            }
        }
    }
    tree(&mut st, fmax);
    let max_val = [st[0].reduce_v[0], st[1].reduce_v[0]];

    // exp(x - max) per element, per-virtual-thread sum.
    for ty in 0..ROWS_PER_BLOCK {
        let probs = probs_of(ty);
        for tid in 0..LANES {
            for t in (tid..VBLOCK).step_by(LANES) {
                let mut sum = 0.0f32;
                let mut i = t;
                while i < N_EXP {
                    let e = (probs[i] - max_val[ty]).exp();
                    st[ty].smem_v[i] = e;
                    sum += e;
                    i += VBLOCK;
                }
                st[ty].reduce_v[t] = sum;
            }
        }
    }
    tree(&mut st, fadd);
    let sum_all = [st[0].reduce_v[0], st[1].reduce_v[0]];

    // Direct divide.
    for ty in 0..ROWS_PER_BLOCK {
        for tid in 0..LANES {
            for i in (tid..N_EXP).step_by(LANES) {
                st[ty].smem_v[i] /= sum_all[ty];
            }
        }
    }

    // Top-8.
    for k in 0..TOP_K {
        for row in st.iter_mut() {
            for tid in 0..LANES {
                for t in (tid..VBLOCK).step_by(LANES) {
                    let mut best_v = f32::NEG_INFINITY;
                    let mut best_i = -1i32;
                    let mut i = t;
                    while i < N_EXP {
                        let v = row.smem_v[i];
                        if v > best_v {
                            best_v = v;
                            best_i = i as i32;
                        }
                        i += VBLOCK;
                    }
                    row.reduce_v[t] = best_v;
                    row.reduce_i[t] = best_i;
                }
            }
        }
        let mut s = VBLOCK >> 1;
        while s > 0 {
            for row in st.iter_mut() {
                for tid in 0..LANES {
                    for t in (tid..s).step_by(LANES) {
                        let (va, vb) = (row.reduce_v[t], row.reduce_v[t + s]);
                        if vb > va {
                            row.reduce_v[t] = vb;
                            row.reduce_i[t] = row.reduce_i[t + s];
                        }
                    }
                }
            }
            s >>= 1;
        }
        for row in st.iter_mut() {
            let winner = row.reduce_i[0];
            let winner_w = row.reduce_v[0];
            // tid == 0 lane.
            row.picked_idx[k] = winner;
            row.picked_w[k] = winner_w;
            if winner >= 0 && (winner as usize) < N_EXP {
                row.smem_v[winner as usize] = f32::NEG_INFINITY;
            }
        }
    }

    // Renorm (tid == 0 lane), then clamp + store (tid < 8 lanes).
    for (ty, row) in st.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for k in 0..TOP_K {
            s += row.picked_w[k];
        }
        if norm && s > 0.0 {
            for k in 0..TOP_K {
                row.picked_w[k] /= s;
            }
        }
        let bid = block * ROWS_PER_BLOCK + ty;
        for tid in 0..TOP_K {
            let mut safe_idx = row.picked_idx[tid];
            let mut safe_w = row.picked_w[tid];
            if safe_idx < 0 || safe_idx as usize >= N_EXP {
                safe_idx = 0;
                safe_w = 0.0;
            }
            out_idx[bid * TOP_K + tid] = safe_idx;
            out_w[bid * TOP_K + tid] = safe_w;
        }
    }
}

/// `for s = 128; s > 0; s >>= 1 { for t = tid; t < s; t += 128 { r[t] = op(r[t], r[t+s]) } }`
/// on both row slots.
fn tree(st: &mut [RowLds; 2], op: fn(f32, f32) -> f32) {
    let mut s = VBLOCK >> 1;
    while s > 0 {
        for row in st.iter_mut() {
            for tid in 0..LANES {
                for t in (tid..s).step_by(LANES) {
                    row.reduce_v[t] = op(row.reduce_v[t], row.reduce_v[t + s]);
                }
            }
        }
        s >>= 1;
    }
}

// ---------------------------------------------------------------------------
// GPU mode
// ---------------------------------------------------------------------------

struct Report {
    comparisons: usize,
    failures: usize,
    first_mismatch: Option<String>,
}

fn gpu_mode() {
    let start = Instant::now();
    let mut gpu = Gpu::init().expect("GPU init");
    if !gpu.arch_caps.is_wave32() {
        eprintln!(
            "REJECT: moe_router_topk_multirow_oracle requires a wave32 device; arch {} is not wave32",
            gpu.arch
        );
        std::process::exit(2);
    }
    for &rows in &GPU_ROWS {
        if !Gpu::moe_router_topk_multirow_admitted(true, TOP_K, N_EXP, rows, true) {
            eprintln!(
                "REJECT: multirow router not admitted for k={TOP_K} n_exp={N_EXP} rows={rows}"
            );
            std::process::exit(2);
        }
    }

    let mut report = Report {
        comparisons: 0,
        failures: 0,
        first_mismatch: None,
    };
    let mut cases = 0usize;
    for &rows in &GPU_ROWS {
        for variant in VARIANTS {
            let logits = make_logits(variant, rows);
            let hash = fnv1a(&logits);
            for norm in [false, true] {
                cases += 1;
                run_case(&mut gpu, &logits, rows, variant, norm, hash, &mut report);
            }
        }
    }

    let mut timings: Vec<String> = Vec::new();
    let mut timing_truncated = false;
    for &rows in &GPU_ROWS {
        for norm in [false, true] {
            if start.elapsed().as_secs_f64() > TIMING_CUTOFF_S {
                timing_truncated = true;
                continue;
            }
            let line = run_timing(&mut gpu, rows, norm);
            println!("{line}");
            timings.push(line);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let all_byte_identical = report.failures == 0;
    let first = report.first_mismatch.as_deref().unwrap_or("null");
    println!(
        "{{\"summary\":true,\"all_byte_identical\":{all_byte_identical},\"cases\":{cases},\
         \"comparisons\":{},\"failures\":{},\"first_mismatch\":{first},\
         \"timing_groups\":{},\"timing_truncated\":{timing_truncated},\"timing_reps\":{REPS},\
         \"elapsed_s\":{elapsed:.1},\"within_{}s\":{}}}",
        report.comparisons,
        report.failures,
        timings.len(),
        BUDGET_S as u32,
        elapsed <= BUDGET_S,
    );
    if !all_byte_identical || timing_truncated || elapsed > BUDGET_S {
        std::process::exit(1);
    }
}

fn upload_sentinel(gpu: &mut Gpu, rows: usize) -> (GpuTensor, GpuTensor) {
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); rows * TOP_K];
    let ids = gpu.upload_f32(&sentinel, &[rows, TOP_K]).expect("ids");
    let w = gpu.upload_f32(&sentinel, &[rows, TOP_K]).expect("weights");
    (ids, w)
}

fn run_case(
    gpu: &mut Gpu,
    logits: &[f32],
    rows: usize,
    variant: &str,
    norm: bool,
    hash: u64,
    report: &mut Report,
) {
    // Incumbent pair (softmax is in place).
    let l_inc = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload incumbent logits");
    let (i_inc, w_inc) = upload_sentinel(gpu, rows);
    gpu.softmax_f32(&l_inc).expect("incumbent softmax");
    gpu.moe_topk_renorm_k8_batched(&l_inc, &i_inc, &w_inc, N_EXP, norm, rows)
        .expect("incumbent batched top-k");
    let inc_idx = gpu.download_f32(&i_inc).expect("download incumbent ids");
    let inc_w = gpu
        .download_f32(&w_inc)
        .expect("download incumbent weights");
    for t in [l_inc, i_inc, w_inc] {
        gpu.free_tensor(t).expect("free");
    }

    // Current fused, 128 lanes.
    let l_fus = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload fused logits");
    let (i_fus, w_fus) = upload_sentinel(gpu, rows);
    gpu.moe_router_softmax_topk_k8_prefill(&l_fus, &i_fus, &w_fus, N_EXP, norm, rows, LANES)
        .expect("current fused router");
    let fus_idx = gpu.download_f32(&i_fus).expect("download fused ids");
    let fus_w = gpu.download_f32(&w_fus).expect("download fused weights");
    for t in [l_fus, i_fus, w_fus] {
        gpu.free_tensor(t).expect("free");
    }

    // New multirow route.
    let l_new = gpu
        .upload_f32(logits, &[rows, N_EXP])
        .expect("upload new logits");
    let (i_new, w_new) = upload_sentinel(gpu, rows);
    gpu.moe_router_softmax_topk_k8_multirow_prefill(&l_new, &i_new, &w_new, N_EXP, norm, rows)
        .expect("multirow router");
    let new_idx = gpu.download_f32(&i_new).expect("download new ids");
    let new_w = gpu.download_f32(&w_new).expect("download new weights");
    for t in [l_new, i_new, w_new] {
        gpu.free_tensor(t).expect("free");
    }

    let ctx = CaseCtx {
        rows,
        variant,
        norm,
    };
    // (comparison name, reference ids, reference weights, got ids, got weights)
    type Comparison<'a> = (&'a str, &'a [f32], &'a [f32], &'a [f32], &'a [f32]);
    let pairs: [Comparison<'_>; 3] = [
        (
            "new_vs_current_fused",
            fus_idx.as_slice(),
            fus_w.as_slice(),
            new_idx.as_slice(),
            new_w.as_slice(),
        ),
        (
            "new_vs_incumbent",
            inc_idx.as_slice(),
            inc_w.as_slice(),
            new_idx.as_slice(),
            new_w.as_slice(),
        ),
        (
            "current_fused_vs_incumbent",
            inc_idx.as_slice(),
            inc_w.as_slice(),
            fus_idx.as_slice(),
            fus_w.as_slice(),
        ),
    ];
    for (name, ref_idx, ref_w, got_idx, got_w) in pairs {
        let mismatch = first_mismatch(ref_idx, ref_w, got_idx, got_w);
        let pass = mismatch.is_none();
        println!(
            "{{\"rows\":{rows},\"n_exp\":{N_EXP},\"variant\":\"{variant}\",\"norm\":{norm},\
             \"input_fnv1a64\":\"{hash:016x}\",\"comparison\":\"{name}\",\"pass\":{pass}}}"
        );
        // The third pair is informational (existing fused route vs incumbent);
        // a failure there is still a failure of the byte oracle.
        report.comparisons += 1;
        if let Some(m) = mismatch {
            report.failures += 1;
            let json = ctx.mismatch_json(name, &m);
            eprintln!("MISMATCH {json}");
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some(json);
            }
        }
    }
}

struct CaseCtx<'a> {
    rows: usize,
    variant: &'a str,
    norm: bool,
}

struct Mismatch {
    row: usize,
    pick: usize,
    ref_id: u32,
    got_id: u32,
    ref_w: u32,
    got_w: u32,
}

impl CaseCtx<'_> {
    fn mismatch_json(&self, comparison: &str, m: &Mismatch) -> String {
        format!(
            "{{\"comparison\":\"{comparison}\",\"rows\":{},\"n_exp\":{N_EXP},\"variant\":\"{}\",\
             \"norm\":{},\"row\":{},\"pick\":{},\"ref_id\":{},\"got_id\":{},\
             \"ref_id_hex\":\"{:#010x}\",\"got_id_hex\":\"{:#010x}\",\
             \"ref_weight_bits\":\"{:#010x}\",\"got_weight_bits\":\"{:#010x}\"}}",
            self.rows,
            self.variant,
            self.norm,
            m.row,
            m.pick,
            m.ref_id as i32,
            m.got_id as i32,
            m.ref_id,
            m.got_id,
            m.ref_w,
            m.got_w,
        )
    }
}

/// First raw-bit difference (ids or weights), `None` when byte-identical.
fn first_mismatch(
    ref_idx: &[f32],
    ref_w: &[f32],
    got_idx: &[f32],
    got_w: &[f32],
) -> Option<Mismatch> {
    if ref_idx.len() != got_idx.len() || ref_w.len() != got_w.len() || ref_idx.len() != ref_w.len()
    {
        return Some(Mismatch {
            row: usize::MAX,
            pick: usize::MAX,
            ref_id: ref_idx.len() as u32,
            got_id: got_idx.len() as u32,
            ref_w: ref_w.len() as u32,
            got_w: got_w.len() as u32,
        });
    }
    (0..ref_idx.len()).find_map(|i| {
        let differs = ref_idx[i].to_bits() != got_idx[i].to_bits()
            || ref_w[i].to_bits() != got_w[i].to_bits();
        differs.then(|| Mismatch {
            row: i / TOP_K,
            pick: i % TOP_K,
            ref_id: ref_idx[i].to_bits(),
            got_id: got_idx[i].to_bits(),
            ref_w: ref_w[i].to_bits(),
            got_w: got_w[i].to_bits(),
        })
    })
}

/// Times the three routes on the random input. Returns one JSON line.
fn run_timing(gpu: &mut Gpu, rows: usize, norm: bool) -> String {
    let logits = make_logits("random", rows);
    let numel = rows * N_EXP;
    let src = gpu.upload_f32(&logits, &[rows, N_EXP]).expect("upload src");
    let work = gpu
        .upload_f32(&logits, &[rows, N_EXP])
        .expect("upload work");
    let (idx, w) = upload_sentinel(gpu, rows);
    gpu.hip.device_synchronize().expect("sync uploads");

    let inc = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.softmax_f32(&work).expect("timing incumbent softmax");
        gpu.moe_topk_renorm_k8_batched(&work, &idx, &w, N_EXP, norm, rows)
            .expect("timing incumbent top-k");
    });
    let fus = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.moe_router_softmax_topk_k8_prefill(&work, &idx, &w, N_EXP, norm, rows, LANES)
            .expect("timing current fused router");
    });
    let new = measure(gpu, &src, &work, numel, &mut |gpu: &mut Gpu| {
        gpu.moe_router_softmax_topk_k8_multirow_prefill(&work, &idx, &w, N_EXP, norm, rows)
            .expect("timing multirow router");
    });
    for t in [src, work, idx, w] {
        gpu.free_tensor(t).expect("free timing tensor");
    }
    let (inc_m, inc_lo, inc_hi) = stats(inc);
    let (fus_m, fus_lo, fus_hi) = stats(fus);
    let (new_m, new_lo, new_hi) = stats(new);
    format!(
        "{{\"timing\":true,\"rows\":{rows},\"n_exp\":{N_EXP},\"norm\":{norm},\
         \"median_us\":{{\"incumbent_pair\":{inc_m:.3},\"current_fused\":{fus_m:.3},\"new\":{new_m:.3}}},\
         \"p10_us\":{{\"incumbent_pair\":{inc_lo:.3},\"current_fused\":{fus_lo:.3},\"new\":{new_lo:.3}}},\
         \"p90_us\":{{\"incumbent_pair\":{inc_hi:.3},\"current_fused\":{fus_hi:.3},\"new\":{new_hi:.3}}},\
         \"warmups\":{WARMUPS},\"reps\":{REPS}}}"
    )
}

/// WARMUPS + REPS reps of `run`; returns the REPS timed durations in
/// microseconds. Each rep restores the raw logits from `src` into `work` and
/// synchronizes outside the event-bracketed interval.
fn measure(
    gpu: &mut Gpu,
    src: &GpuTensor,
    work: &GpuTensor,
    numel: usize,
    run: &mut dyn FnMut(&mut Gpu),
) -> Vec<f64> {
    let start = gpu.hip.event_create().expect("create start event");
    let stop = gpu.hip.event_create().expect("create stop event");
    let mut samples = Vec::with_capacity(REPS);
    for rep in 0..WARMUPS + REPS {
        gpu.copy_f32_buffer(work, src, numel)
            .expect("restore raw logits");
        gpu.hip.device_synchronize().expect("sync restore");

        gpu.hip
            .event_record(&start, gpu.active_stream.as_ref())
            .expect("record start");
        run(gpu);
        gpu.hip
            .event_record(&stop, gpu.active_stream.as_ref())
            .expect("record stop");
        gpu.hip.event_synchronize(&stop).expect("sync stop");
        let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
        if rep >= WARMUPS {
            samples.push(f64::from(ms) * 1_000.0);
        }
    }
    gpu.hip.event_destroy(start).expect("destroy start event");
    gpu.hip.event_destroy(stop).expect("destroy stop event");
    samples
}

/// (median = mean of the two middle samples, p10, p90) of REPS samples.
fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    assert_eq!(samples.len(), REPS);
    samples.sort_by(f64::total_cmp);
    let median = (samples[REPS / 2 - 1] + samples[REPS / 2]) / 2.0;
    let p10 = samples[(0.1 * (REPS - 1) as f64).round() as usize];
    let p90 = samples[(0.9 * (REPS - 1) as f64).round() as usize];
    (median, p10, p90)
}

// ---------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------

fn fnv1a(values: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct Rng(u64);

impl Rng {
    fn new(rows: usize, tag: usize) -> Self {
        Rng(0x9e37_79b9_7f4a_7c15 ^ ((rows as u64) << 40) ^ ((N_EXP as u64) << 20) ^ tag as u64)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// `count` distinct positions below `N_EXP`.
    fn positions(&mut self, count: usize) -> Vec<usize> {
        let mut out: Vec<usize> = Vec::with_capacity(count);
        while out.len() < count.min(N_EXP) {
            let p = self.below(N_EXP);
            if !out.contains(&p) {
                out.push(p);
            }
        }
        out
    }
}

/// Row of random lows in [-6, -1) with the given positions set to the shared
/// maximum 7.0.
fn pair_row(positions: &[usize], seed: usize) -> Vec<f32> {
    let mut rng = Rng::new(2, 1000 + seed);
    let mut v: Vec<f32> = (0..N_EXP).map(|_| rng.range(-6.0, -1.0)).collect();
    for &p in positions {
        v[p] = 7.0;
    }
    v
}

fn make_logits(variant: &str, rows: usize) -> Vec<f32> {
    let tag = VARIANTS
        .iter()
        .position(|v| *v == variant)
        .unwrap_or(VARIANTS.len());
    let mut rng = Rng::new(rows, tag);
    let mut out = Vec::with_capacity(rows * N_EXP);
    for row in 0..rows {
        let kind = if variant == "mixed_rows" {
            MIXED_KINDS[row % MIXED_KINDS.len()]
        } else {
            variant
        };
        out.extend(make_row(kind, row, &mut rng));
    }
    out
}

fn make_row(kind: &str, row: usize, rng: &mut Rng) -> Vec<f32> {
    match kind {
        "allzero" => vec![0.0; N_EXP],
        "random" => (0..N_EXP).map(|_| rng.range(-12.0, 12.0)).collect(),
        "tie_heavy" => match row % 3 {
            // Seven coarse levels: ~2/7 of the experts tie at the maximum.
            0 => {
                const LEVELS: [f32; 7] = [3.0, 3.0, 2.0, 2.0, 1.0, 0.0, -1.0];
                (0..N_EXP)
                    .map(|_| LEVELS[rng.below(LEVELS.len())])
                    .collect()
            }
            // Two levels, ~25% at the top.
            1 => (0..N_EXP)
                .map(|_| if rng.below(4) == 0 { 1.0 } else { 0.0 })
                .collect(),
            // 9..=24 maxima over a flat floor: the k = 8 cut falls inside the tie.
            _ => {
                let mut v = vec![-2.0f32; N_EXP];
                let count = 9 + rng.below(16);
                for p in rng.positions(count) {
                    v[p] = 5.0;
                }
                v
            }
        },
        "tie_tree_cross" => {
            // Equal maxima at fixed virtual-tree positions (same lane across the
            // two virtual halves, adjacent, mirrored, wave-boundary and sibling
            // subtrees); later passes over the set list shift the positions.
            let set = TREE_SETS[row % TREE_SETS.len()];
            const SHIFTS: [usize; 5] = [0, 0, 1, 32, 127];
            let shift = SHIFTS[(row / TREE_SETS.len()) % SHIFTS.len()];
            let mut v: Vec<f32> = (0..N_EXP).map(|_| rng.range(-6.0, -1.0)).collect();
            for &p in set {
                v[(p + shift) % N_EXP] = 7.0;
            }
            v
        }
        other => panic!("unknown variant {other}"),
    }
}
