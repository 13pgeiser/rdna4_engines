// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exactness + cost oracle for `hipfire_runtime::batch_sampler::sample_batch`.
//
// Contract under test: for every lane, `sample_batch` returns the SAME token,
// the SAME advanced RNG and leaves the SAME logits bytes (blocked tokens and
// the penalty prepass write in place) as a serial `sampler::sample` call with
// a `repeat_buf` of `repeat_cap` tokens, for any lane order. It also counts
// the host transfers of both paths and (with `--timing`) measures what an
// active penalty costs each path.
//
// Needs a live GPU; no model. Run under the card lock:
//   cargo build --release -p hipfire-runtime --example sample_batch_exact_oracle --features lab
//   sample_batch_exact_oracle --lanes 1,2,4,8 --trials 40 --out /abs/oracle.json [--timing]
//
// Flags:
//   --lanes 1,2,4,8     lane counts K (default 1,2,4,8)
//   --trials N          randomized batches per K (default 40; each runs 3 lane orders)
//   --out ABS.json      JSON report (required)
//   --vocab V           vocab size (default 248320)
//   --repeat-cap C      serial repeat_buf capacity in tokens (default 2048)
//   --seed S            host RNG seed
//   --timing            K=8 penalty-cost section (>=200 timed iterations per config)
//   --iters N           timing iterations (default 200, floor 200)
//
// Exit status is nonzero when mismatches > 0, or when the run never exercised
// a fast-reducer sentinel fallback (the exactness claim would be untested).
//
// `fallbacks_exercised` counts sample_batch calls that issued the second D2H
// (one or more lanes ran the exact-reducer sentinel fallback): the batch's D2H
// count per call is 1, or 2 when a fallback ran. It reads 0 when
// HIPFIRE_SAMPLE_PARALLEL=0 (the batch then runs the serial sampler).

use hip_bridge::memory_effects;
use hipfire_runtime::batch_sampler::{
    sample_batch, BatchSampleJob, BatchSamplerArena,
};
use hipfire_runtime::sampler::{self, SamplerConfig};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (((self.next() >> 40) as f32) + 0.5) * (1.0 / 16_777_216.0)
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f32::consts::PI * b).cos()
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

struct PoolRow {
    kind: &'static str,
    logits: Vec<f32>,
    /// Token ids by descending logit (ties by id), first 32.
    top: Vec<u32>,
}

fn top_of(logits: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(32);
    idx
}

fn normal_row(rng: &mut Rng, vocab: usize, scale: f32) -> Vec<f32> {
    (0..vocab).map(|_| rng.normal() * scale).collect()
}

fn build_pool(rng: &mut Rng, vocab: usize) -> Vec<PoolRow> {
    let mut pool = Vec::new();
    let mut add = |kind: &'static str, logits: Vec<f32>| {
        let top = top_of(&logits);
        pool.push(PoolRow { kind, logits, top });
    };
    add("normal", normal_row(rng, vocab, 2.0));
    add("normal", normal_row(rng, vocab, 3.0));
    // Exact ties at top-1.
    {
        let mut r = normal_row(rng, vocab, 2.0);
        let max = r.iter().cloned().fold(f32::MIN, f32::max) + 0.5;
        let (a, b) = (rng.below(vocab), rng.below(vocab));
        r[a] = max;
        r[b] = max;
        add("tie_top1", r);
    }
    // Exact ties straddling the top-k boundary (ranks 18..=24 equal the
    // rank-19 value, so every requested k in {5,20,40,64} sees a tie cut
    // for at least the 20/21 candidate widths).
    {
        let mut r = normal_row(rng, vocab, 2.0);
        let top = top_of(&r);
        let v = r[top[19] as usize];
        let mut ranked: Vec<u32> = (0..vocab as u32).collect();
        ranked.sort_unstable_by(|&a, &b| {
            r[b as usize].partial_cmp(&r[a as usize]).unwrap().then(a.cmp(&b))
        });
        for &t in &ranked[18..=24] {
            r[t as usize] = v;
        }
        add("tie_topk20", r);
    }
    {
        let mut r = normal_row(rng, vocab, 2.0);
        let mut ranked: Vec<u32> = (0..vocab as u32).collect();
        ranked.sort_unstable_by(|&a, &b| {
            r[b as usize].partial_cmp(&r[a as usize]).unwrap().then(a.cmp(&b))
        });
        let v = r[ranked[4] as usize];
        for &t in &ranked[3..=7] {
            r[t as usize] = v;
        }
        add("tie_topk5", r);
    }
    // Six equal tokens carry ~all the mass, so the top-p cut falls inside a tie.
    {
        let mut r: Vec<f32> = normal_row(rng, vocab, 0.1).iter().map(|x| x - 20.0).collect();
        for _ in 0..6 {
            let t = rng.below(vocab);
            r[t] = 5.0;
        }
        add("tie_topp", r);
    }
    // All -inf except one.
    {
        let mut r = vec![f32::NEG_INFINITY; vocab];
        r[rng.below(vocab)] = 1.0;
        add("one_hot", r);
    }
    // Very peaked.
    {
        let mut r = normal_row(rng, vocab, 1.0);
        r[rng.below(vocab)] += 30.0;
        add("peaked", r);
    }
    // Flat (every token tied) and flat-with-three-way top tie.
    add("flat", vec![0.0; vocab]);
    {
        let mut r = normal_row(rng, vocab, 1.0);
        let max = r.iter().cloned().fold(f32::MIN, f32::max) + 1.0;
        for _ in 0..3 {
            let t = rng.below(vocab);
            r[t] = max;
        }
        add("tie_top3", r);
    }
    pool
}

#[derive(Clone)]
struct Lane {
    row: usize,
    cfg: SamplerConfig,
    history: Vec<u32>,
    rng: u32,
}

fn random_lane(rng: &mut Rng, pool: &[PoolRow], vocab: usize, row: usize) -> Lane {
    let temperature = rng.pick(&[0.0f32, 0.7, 0.7, 0.7, 1.0]);
    let top_p = if temperature == 0.0 { 1.0 } else { rng.pick(&[0.95f32, 0.95, 1.0, 0.5]) };
    let repeat_penalty = rng.pick(&[1.0f32, 1.05, 1.3]);
    let presence_penalty = rng.pick(&[0.0f32, 0.5]);
    let frequency_penalty = rng.pick(&[0.0f32, 0.5]);
    let repeat_window = rng.pick(&[0usize, 1, 128, 2048]);
    let top_k = rng.pick(&[None, Some(5u32), Some(20), Some(40), Some(64)]);
    let min_p = rng.pick(&[None, Some(0.05f32)]);
    let mut blocked_tokens = Vec::new();
    match rng.below(4) {
        0 => {}
        1 => blocked_tokens.push(pool[row].top[0]),
        2 => {
            blocked_tokens.push(pool[row].top[rng.below(8)]);
            blocked_tokens.push(rng.below(vocab) as u32);
        }
        _ => {
            blocked_tokens.push(pool[row].top[1]);
            blocked_tokens.push(vocab as u32);
            blocked_tokens.push(vocab as u32 + 7);
            blocked_tokens.push(u32::MAX);
        }
    }
    let hist_len = match rng.below(6) {
        0 => 0,
        1 => 3,
        2 => repeat_window.saturating_sub(1),
        3 => repeat_window,
        4 => repeat_window + 50,
        _ => 3000,
    };
    // Duplicates on purpose: half the ids come from the row's own top-32 (so
    // the penalties actually move candidates), the rest from a tiny alphabet.
    let history: Vec<u32> = (0..hist_len)
        .map(|_| {
            if rng.below(2) == 0 {
                pool[row].top[rng.below(32)]
            } else {
                rng.below(24) as u32
            }
        })
        .collect();
    Lane {
        row,
        cfg: SamplerConfig {
            temperature,
            top_p,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            blocked_tokens,
            top_k,
            min_p,
        },
        history,
        rng: rng.next() as u32,
    }
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 { xs[n / 2] } else { 0.5 * (xs[n / 2 - 1] + xs[n / 2]) }
}

/// Bootstrap 95% CI of median(active) - median(neutral).
fn bootstrap_delta(rng: &mut Rng, active: &[f64], neutral: &[f64]) -> (f64, f64, f64) {
    let point = median(&mut active.to_vec()) - median(&mut neutral.to_vec());
    let mut diffs = Vec::with_capacity(2000);
    let mut a = vec![0.0; active.len()];
    let mut n = vec![0.0; neutral.len()];
    for _ in 0..2000 {
        for x in a.iter_mut() {
            *x = active[rng.below(active.len())];
        }
        for x in n.iter_mut() {
            *x = neutral[rng.below(neutral.len())];
        }
        diffs.push(median(&mut a) - median(&mut n));
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (point, diffs[(diffs.len() as f64 * 0.025) as usize], diffs[(diffs.len() as f64 * 0.975) as usize - 1])
}

fn upload_rows(gpu: &Gpu, buf: &GpuTensor, pos: usize, vocab: usize, row: &[f32]) {
    gpu.hip
        .memcpy_htod_offset(&buf.buf, pos * vocab * 4, f32_bytes(row))
        .expect("row upload");
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lanes_list: Vec<usize> = arg(&args, "--lanes")
        .unwrap_or_else(|| "1,2,4,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("--lanes"))
        .collect();
    let trials: usize = arg(&args, "--trials").map(|s| s.parse().expect("--trials")).unwrap_or(40);
    let vocab: usize = arg(&args, "--vocab").map(|s| s.parse().expect("--vocab")).unwrap_or(248_320);
    let repeat_cap: usize =
        arg(&args, "--repeat-cap").map(|s| s.parse().expect("--repeat-cap")).unwrap_or(2048);
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse().expect("--seed")).unwrap_or(0x5eed_5a3b);
    let out_path = arg(&args, "--out").expect("--out ABS.json is required");
    let timing = args.iter().any(|a| a == "--timing");
    let iters: usize =
        arg(&args, "--iters").map(|s| s.parse().expect("--iters")).unwrap_or(200).max(200);
    let max_k = *lanes_list.iter().max().expect("--lanes");

    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(seed | 1);
    let pool = build_pool(&mut rng, vocab);
    let mut arena = BatchSamplerArena::new(&mut gpu, max_k.max(8), repeat_cap).expect("arena");
    let sample_buf = gpu.alloc_tensor(&[2], DType::F32).expect("sample_buf");
    let repeat_buf = gpu.alloc_tensor(&[repeat_cap], DType::F32).expect("repeat_buf");

    let mut per_k = Vec::new();
    let mut total_cases = 0usize;
    let mut total_lane_cases = 0usize;
    let mut total_mismatches = 0usize;
    let mut total_fallbacks = 0usize;
    let mut mismatch_log: Vec<String> = Vec::new();

    for &k in &lanes_list {
        let a_rows = gpu.alloc_tensor(&[k * vocab], DType::F32).expect("rows A");
        let b_rows = gpu.alloc_tensor(&[k * vocab], DType::F32).expect("rows B");
        let (mut s_h2d, mut s_d2h, mut b_h2d, mut b_d2h) = (0u64, 0u64, 0u64, 0u64);
        let (mut cases, mut lane_cases, mut mism, mut fallbacks) = (0usize, 0usize, 0usize, 0usize);
        let mut kinds_seen = std::collections::BTreeMap::<&'static str, usize>::new();

        for trial in 0..trials {
            let lanes: Vec<Lane> = (0..k)
                .map(|i| {
                    // The first trials walk the whole pool through lane 0 so
                    // every adversarial row kind is covered for every K.
                    let row = if i == 0 && trial < pool.len() { trial } else { rng.below(pool.len()) };
                    random_lane(&mut rng, &pool, vocab, row)
                })
                .collect();
            let identity: Vec<usize> = (0..k).collect();
            let reversed: Vec<usize> = (0..k).rev().collect();
            let mut shuffled = identity.clone();
            for i in (1..k).rev() {
                shuffled.swap(i, rng.below(i + 1));
            }
            for order in [identity, reversed, shuffled] {
                for (pos, &li) in order.iter().enumerate() {
                    let row = &pool[lanes[li].row].logits;
                    upload_rows(&gpu, &a_rows, pos, vocab, row);
                    upload_rows(&gpu, &b_rows, pos, vocab, row);
                    *kinds_seen.entry(pool[lanes[li].row].kind).or_default() += 1;
                }
                gpu.hip.device_synchronize().expect("sync");

                // Serial reference.
                let before = memory_effects::snapshot();
                let mut serial = Vec::with_capacity(k);
                for (pos, &li) in order.iter().enumerate() {
                    let view = a_rows.sub_offset(pos * vocab, vocab);
                    let mut r = lanes[li].rng;
                    let tok = sampler::sample(
                        &mut gpu,
                        &view,
                        &sample_buf,
                        &repeat_buf,
                        vocab,
                        &lanes[li].history,
                        &lanes[li].cfg,
                        &mut r,
                    );
                    serial.push((tok, r));
                }
                let mid = memory_effects::snapshot();

                // Batched.
                let views: Vec<GpuTensor> =
                    (0..k).map(|pos| b_rows.sub_offset(pos * vocab, vocab)).collect();
                let jobs: Vec<BatchSampleJob<'_>> = order
                    .iter()
                    .enumerate()
                    .map(|(pos, &li)| BatchSampleJob {
                        logits: &views[pos],
                        history: &lanes[li].history,
                        cfg: &lanes[li].cfg,
                        rng_state: lanes[li].rng,
                    })
                    .collect();
                let picks: Vec<(u32, u32)> = sample_batch(&mut gpu, &mut arena, &jobs)
                    .expect("sample_batch")
                    .iter()
                    .map(|p| (p.token_id, p.advanced_rng))
                    .collect();
                let after = memory_effects::snapshot();
                let sd = mid.since(before);
                let bd = after.since(mid);
                s_h2d += sd.htod;
                s_d2h += sd.dtoh;
                b_h2d += bd.htod;
                b_d2h += bd.dtoh;
                if bd.dtoh == 2 {
                    fallbacks += 1;
                }

                let a_after = gpu.download_f32(&a_rows).expect("download A");
                let b_after = gpu.download_f32(&b_rows).expect("download B");
                cases += 1;
                for pos in 0..k {
                    lane_cases += 1;
                    let li = order[pos];
                    let rows_eq = bits_eq(
                        &a_after[pos * vocab..(pos + 1) * vocab],
                        &b_after[pos * vocab..(pos + 1) * vocab],
                    );
                    if serial[pos] != picks[pos] || !rows_eq {
                        mism += 1;
                        if mismatch_log.len() < 20 {
                            mismatch_log.push(format!(
                                "K={k} trial={trial} pos={pos} row={} serial={:?} batch={:?} logits_bytes_equal={rows_eq} cfg=T{} p{} k{:?} mp{:?} rp{} pp{} fp{} w{} hist={} blocked={:?}",
                                pool[lanes[li].row].kind,
                                serial[pos],
                                picks[pos],
                                lanes[li].cfg.temperature,
                                lanes[li].cfg.top_p,
                                lanes[li].cfg.top_k,
                                lanes[li].cfg.min_p,
                                lanes[li].cfg.repeat_penalty,
                                lanes[li].cfg.presence_penalty,
                                lanes[li].cfg.frequency_penalty,
                                lanes[li].cfg.repeat_window,
                                lanes[li].history.len(),
                                lanes[li].cfg.blocked_tokens,
                            ));
                        }
                    }
                }
            }
        }
        println!(
            "K={k}: cases={cases} lane_cases={lane_cases} mismatches={mism} fallback_batches={fallbacks} \
             transfers serial h2d={s_h2d} d2h={s_d2h} | batch h2d={b_h2d} d2h={b_d2h}"
        );
        total_cases += cases;
        total_lane_cases += lane_cases;
        total_mismatches += mism;
        total_fallbacks += fallbacks;
        per_k.push(json!({
            "k": k,
            "cases": cases,
            "lane_cases": lane_cases,
            "mismatches": mism,
            "fallback_batches": fallbacks,
            "rows_by_kind": kinds_seen,
            "transfers": {
                "serial_h2d": s_h2d, "serial_d2h": s_d2h,
                "batch_h2d": b_h2d, "batch_d2h": b_d2h,
            },
        }));
        gpu.free_tensor(a_rows).ok();
        gpu.free_tensor(b_rows).ok();
    }

    let mut timing_json = serde_json::Value::Null;
    if timing {
        timing_json = run_timing(&mut gpu, &mut arena, &sample_buf, &repeat_buf, &pool, vocab, iters, &mut rng);
    }

    let report = json!({
        "vocab": vocab,
        "repeat_cap": repeat_cap,
        "seed": seed,
        "trials": trials,
        "cases": total_cases,
        "lane_cases": total_lane_cases,
        "mismatches": total_mismatches,
        "fallbacks_exercised": total_fallbacks,
        "per_k": per_k,
        "mismatch_log": mismatch_log,
        "timing": timing_json,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap()).expect("write --out");
    println!(
        "TOTAL cases={total_cases} lane_cases={total_lane_cases} mismatches={total_mismatches} \
         fallbacks_exercised={total_fallbacks} -> {out_path}"
    );
    for m in &mismatch_log {
        eprintln!("MISMATCH {m}");
    }

    arena.free_gpu(&mut gpu);
    gpu.free_tensor(sample_buf).ok();
    gpu.free_tensor(repeat_buf).ok();

    if total_mismatches > 0 {
        eprintln!("FAIL: {total_mismatches} mismatching lanes");
        std::process::exit(1);
    }
    if total_fallbacks == 0 {
        eprintln!(
            "FAIL: no sentinel fallback exercised (HIPFIRE_SAMPLE_PARALLEL=0 or HIPFIRE_SAMPLE_FAST=0 set, or too few trials)"
        );
        std::process::exit(2);
    }
}

/// K=8 penalty cost: identical rows and histories on every lane, so the lanes
/// differ from the neutral baseline only by the penalty stage. Each timed
/// iteration restores the logits rows from a pristine device copy OUTSIDE the
/// timer; the timer is bracketed by `device_synchronize`.
#[allow(clippy::too_many_arguments)]
fn run_timing(
    gpu: &mut Gpu,
    arena: &mut BatchSamplerArena,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    pool: &[PoolRow],
    vocab: usize,
    iters: usize,
    rng: &mut Rng,
) -> serde_json::Value {
    const K: usize = 8;
    const WARMUP: usize = 20;
    let row = &pool[0];
    let pristine = gpu.alloc_tensor(&[K * vocab], DType::F32).expect("pristine");
    let work = gpu.alloc_tensor(&[K * vocab], DType::F32).expect("work");
    for pos in 0..K {
        upload_rows(gpu, &pristine, pos, vocab, &row.logits);
    }
    let views: Vec<GpuTensor> = (0..K).map(|p| work.sub_offset(p * vocab, vocab)).collect();
    let history: Vec<u32> = (0..3000)
        .map(|_| if rng.below(2) == 0 { row.top[rng.below(32)] } else { rng.below(24) as u32 })
        .collect();

    // (name, repeat, presence, frequency)
    let modes: [(&str, f32, f32, f32); 5] = [
        ("neutral", 1.0, 0.0, 0.0),
        ("repeat_only", 1.05, 0.0, 0.0),
        ("presence_only", 1.0, 0.5, 0.0),
        ("frequency_only", 1.0, 0.0, 0.5),
        ("combined", 1.05, 0.5, 0.5),
    ];
    let mut out = Vec::new();
    for window in [128usize, 2048] {
        // series[path][mode] -> ms per iteration
        let mut series: Vec<[Vec<f64>; 2]> = Vec::new();
        for &(_, rp, pp, fp) in &modes {
            let cfg = SamplerConfig {
                temperature: 0.7,
                top_p: 0.95,
                repeat_penalty: rp,
                repeat_window: window,
                presence_penalty: pp,
                frequency_penalty: fp,
                blocked_tokens: Vec::new(),
                top_k: Some(20),
                min_p: None,
            };
            let mut batch_ms = Vec::with_capacity(iters);
            let mut serial_ms = Vec::with_capacity(iters);
            for it in 0..WARMUP + iters {
                // Batch.
                gpu.memcpy_dtod_at_auto(&work.buf, 0, &pristine.buf, 0, K * vocab * 4).expect("restore");
                gpu.hip.device_synchronize().expect("sync");
                let jobs: Vec<BatchSampleJob<'_>> = views
                    .iter()
                    .map(|v| BatchSampleJob { logits: v, history: &history, cfg: &cfg, rng_state: 12345 })
                    .collect();
                let t = Instant::now();
                sample_batch(gpu, arena, &jobs).expect("sample_batch");
                gpu.hip.device_synchronize().expect("sync");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if it >= WARMUP {
                    batch_ms.push(ms);
                }
                // Serial.
                gpu.memcpy_dtod_at_auto(&work.buf, 0, &pristine.buf, 0, K * vocab * 4).expect("restore");
                gpu.hip.device_synchronize().expect("sync");
                let t = Instant::now();
                for v in &views {
                    let mut r = 12345u32;
                    sampler::sample(gpu, v, sample_buf, repeat_buf, vocab, &history, &cfg, &mut r);
                }
                gpu.hip.device_synchronize().expect("sync");
                let ms = t.elapsed().as_secs_f64() * 1e3;
                if it >= WARMUP {
                    serial_ms.push(ms);
                }
            }
            series.push([batch_ms, serial_ms]);
        }
        for (path_idx, path) in ["batch", "serial"].iter().enumerate() {
            let neutral = series[0][path_idx].clone();
            let neutral_med = median(&mut neutral.clone());
            for (mi, &(name, ..)) in modes.iter().enumerate().skip(1) {
                let active = &series[mi][path_idx];
                let (delta, lo, hi) = bootstrap_delta(rng, active, &neutral);
                println!(
                    "timing K={K} window={window} {path:>6} {name:>14}: neutral {:.4} ms active {:.4} ms delta {:+.4} ms [95% CI {:+.4}, {:+.4}]",
                    neutral_med,
                    median(&mut active.clone()),
                    delta,
                    lo,
                    hi
                );
                out.push(json!({
                    "window": window,
                    "path": path,
                    "mode": name,
                    "iterations": active.len(),
                    "neutral_median_ms": neutral_med,
                    "active_median_ms": median(&mut active.clone()),
                    "delta_median_ms": delta,
                    "delta_ci95_ms": [lo, hi],
                }));
            }
        }
    }
    gpu.free_tensor(pristine).ok();
    gpu.free_tensor(work).ok();
    json!({ "k": K, "history_len": history.len(), "results": out })
}
