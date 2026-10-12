// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone exactness + timing oracle for the FN sampled-plumbing S1
//! primitives (`rdna_compute::spec_sampling_cpu_ar`): resident target
//! distribution rows, resident draft-candidate rows, and the ordered sparse
//! accept. Needs no model: synthetic logit rows exercise the law's edge
//! cases; `--rows-f32` adds real rows.
//!
//! fn_sampled_plumbing_oracle [--vocab N] [--seed N] [--windows N]
//!     [--replays N] [--timing-iters N] [--timing-windows N] [--ks 3,7]
//!     [--rows-f32 FILE --rows-vocab N] [--expect-arch gfx1201]
//!     [--expect-uuid GPU-...] [--expect-bdf 0000:c3:00.0] [--out receipt.json]
//!     [--skip-timing]
//!
//! What it proves, against the CPU `SparseDist` / `accept_sampled_prefix`
//! reference compiled from the same source the request path uses:
//!
//! * target rows: count, token ids (in `SparseDist::entries()` order) and
//!   every f32 probability bit equal, across 16 row shapes (ties, non-finite
//!   logits, all-non-finite argmax fallback, ascending/descending scans,
//!   sparse support, overflow scale), 15 policies and several vocabulary
//!   sizes;
//! * candidate rows: the eight-candidate draft `q` row bits and the sampled
//!   id equal `build_from_candidates` + `sample(u)`; all-non-finite
//!   candidates report the invalid pick and an empty row;
//! * accept: result words (accepted, terminal) equal the host verdict for
//!   directed boundary windows (k=0, rejection first/middle/last, p=q,
//!   zero residual, accepted EOS first/last, replacement EOS, bonus EOS,
//!   n-gram point-mass q) and randomized windows. The RNG draws consumed by
//!   the host are measured by stepping a cloned stream to the post-state and
//!   must equal the contract rule applied to the device words;
//! * capture/replay: the enqueue sequence is captured into a hipGraph (real
//!   kernel nodes, counted from the captured kernarg blobs), replayed with
//!   the resident buffers mutated between replays, and compared bit for bit
//!   with the host every replay;
//! * transfer/sync counts come from the HIP wrapper's own per-call counters
//!   (`hip_bridge::memory_effects`, `launch_counters`), not from this file's
//!   expectations; any transfer/sync inside an enqueue call fails the run;
//! * timing: host row-copy + host law (the existing route) versus the device
//!   route (direct enqueue, replayed graph, replayed graph with device
//!   candidate q), raw distributions only. Nothing here claims a speedup.
//!
//! Exit status: 0 all exact; 1 any bit/verdict/coverage mismatch or hidden
//! transfer (stops at the first one); 2 setup, identity or capture failure.
//! A JSON receipt (identity, counts, timings) goes to stdout and `--out`.

use hip_bridge::{launch_counters, memory_effects, Graph, GraphExec};
use hipfire_runtime::spec_sampling::{accept_sampled_prefix, SampleSpec, SparseDist, SpecRng};
use rdna_compute::spec_sampling_cpu_ar as sc;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const EOS_NONE: u32 = u32::MAX;
const CAP: usize = 64;
const MAX_K: usize = 8;
const SENT: u32 = 0xFFFF_FFFF;
const BUDGET_US: f64 = 200.0;

// ── failure / errors ─────────────────────────────────────────────────

struct Fail {
    code: i32,
    msg: String,
}
type R<T> = Result<T, Fail>;

fn mismatch<T>(msg: String) -> R<T> {
    Err(Fail { code: 1, msg })
}
fn setup<T>(msg: String) -> R<T> {
    Err(Fail { code: 2, msg })
}
fn hip<T>(r: HipResult<T>, what: &str) -> R<T> {
    r.map_err(|e| Fail {
        code: 2,
        msg: format!("{what}: {e:?}"),
    })
}

// ── raw transfers (counted per call) ─────────────────────────────────

static W_H2D_CALLS: AtomicU64 = AtomicU64::new(0);
static W_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static W_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
static W_D2H_BYTES: AtomicU64 = AtomicU64::new(0);

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Blocking H2D, only outside captured regions.
fn h2d(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) -> R<()> {
    W_H2D_CALLS.fetch_add(1, Ordering::Relaxed);
    W_H2D_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
    hip(gpu.memcpy_htod_auto(&t.buf, bytes), "h2d")
}
/// Blocking D2H of the first `n` bytes.
fn d2h(gpu: &Gpu, t: &GpuTensor, n: usize) -> R<Vec<u8>> {
    W_D2H_CALLS.fetch_add(1, Ordering::Relaxed);
    W_D2H_BYTES.fetch_add(n as u64, Ordering::Relaxed);
    let mut v = vec![0u8; n];
    hip(gpu.memcpy_dtoh_auto(&mut v, &t.buf), "d2h")?;
    Ok(v)
}
fn stream_of(gpu: &Gpu) -> R<&hip_bridge::Stream> {
    gpu.active_stream.as_ref().ok_or_else(|| Fail {
        code: 2,
        msg: "no active stream".into(),
    })
}
fn h2d_async(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) -> R<()> {
    W_H2D_CALLS.fetch_add(1, Ordering::Relaxed);
    W_H2D_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
    hip(
        gpu.hip.memcpy_htod_async(&t.buf, bytes, stream_of(gpu)?),
        "h2d_async",
    )
}
fn d2h_async(gpu: &Gpu, t: &GpuTensor, out: &mut [u8]) -> R<()> {
    W_D2H_CALLS.fetch_add(1, Ordering::Relaxed);
    W_D2H_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
    hip(
        gpu.hip.memcpy_dtoh_async(out, &t.buf, stream_of(gpu)?),
        "d2h_async",
    )
}
fn sync(gpu: &Gpu) -> R<()> {
    hip(gpu.hip.stream_synchronize(stream_of(gpu)?), "stream_synchronize")
}
/// Blocking D2H of one logit row out of a resident logits tensor, as the
/// existing request path's `download_f32_into`-per-visited-row does.
fn d2h_row_at(gpu: &Gpu, t: &GpuTensor, off_bytes: usize, out: &mut [f32]) -> R<()> {
    W_D2H_CALLS.fetch_add(1, Ordering::Relaxed);
    W_D2H_BYTES.fetch_add((out.len() * 4) as u64, Ordering::Relaxed);
    // SAFETY: `out` is a live &mut [f32]; any bit pattern is a valid f32.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    hip(gpu.hip.memcpy_dtoh_at(bytes, &t.buf, off_bytes), "d2h_row")
}

#[derive(Clone, Copy, Default)]
struct Tally {
    htod: u64,
    dtod: u64,
    dtoh: u64,
    memset: u64,
    kernel_launches: u64,
    stream_sync: u64,
    event_sync: u64,
    device_sync: u64,
    graph_launch: u64,
    w_h2d_calls: u64,
    w_h2d_bytes: u64,
    w_d2h_calls: u64,
    w_d2h_bytes: u64,
}
fn tally() -> Tally {
    let e = memory_effects::snapshot();
    Tally {
        htod: e.htod,
        dtod: e.dtod,
        dtoh: e.dtoh,
        memset: e.memset,
        kernel_launches: launch_counters::launch_kernel::count(),
        stream_sync: launch_counters::stream_sync::count(),
        event_sync: launch_counters::event_sync::count(),
        device_sync: launch_counters::device_sync::count(),
        graph_launch: launch_counters::graph_launch::count(),
        w_h2d_calls: W_H2D_CALLS.load(Ordering::Relaxed),
        w_h2d_bytes: W_H2D_BYTES.load(Ordering::Relaxed),
        w_d2h_calls: W_D2H_CALLS.load(Ordering::Relaxed),
        w_d2h_bytes: W_D2H_BYTES.load(Ordering::Relaxed),
    }
}
impl Tally {
    fn since(self, e: Tally) -> Tally {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Tally {
            htod: d(self.htod, e.htod),
            dtod: d(self.dtod, e.dtod),
            dtoh: d(self.dtoh, e.dtoh),
            memset: d(self.memset, e.memset),
            kernel_launches: d(self.kernel_launches, e.kernel_launches),
            stream_sync: d(self.stream_sync, e.stream_sync),
            event_sync: d(self.event_sync, e.event_sync),
            device_sync: d(self.device_sync, e.device_sync),
            graph_launch: d(self.graph_launch, e.graph_launch),
            w_h2d_calls: d(self.w_h2d_calls, e.w_h2d_calls),
            w_h2d_bytes: d(self.w_h2d_bytes, e.w_h2d_bytes),
            w_d2h_calls: d(self.w_d2h_calls, e.w_d2h_calls),
            w_d2h_bytes: d(self.w_d2h_bytes, e.w_d2h_bytes),
        }
    }
    fn add(&mut self, o: Tally) {
        self.htod += o.htod;
        self.dtod += o.dtod;
        self.dtoh += o.dtoh;
        self.memset += o.memset;
        self.kernel_launches += o.kernel_launches;
        self.stream_sync += o.stream_sync;
        self.event_sync += o.event_sync;
        self.device_sync += o.device_sync;
        self.graph_launch += o.graph_launch;
        self.w_h2d_calls += o.w_h2d_calls;
        self.w_h2d_bytes += o.w_h2d_bytes;
        self.w_d2h_calls += o.w_d2h_calls;
        self.w_d2h_bytes += o.w_d2h_bytes;
    }
    /// Any transfer, memset or synchronization at all.
    fn any_host_traffic(&self) -> bool {
        self.htod
            + self.dtod
            + self.dtoh
            + self.memset
            + self.stream_sync
            + self.event_sync
            + self.device_sync
            > 0
    }
    fn json(&self) -> Value {
        json!({
            "hip_h2d_calls": self.htod, "hip_d2h_calls": self.dtoh,
            "hip_d2d_calls": self.dtod, "hip_memset_calls": self.memset,
            "stream_sync": self.stream_sync, "event_sync": self.event_sync,
            "device_sync": self.device_sync, "graph_launch": self.graph_launch,
            "kernel_launches": self.kernel_launches,
            "oracle_h2d_calls": self.w_h2d_calls, "oracle_h2d_bytes": self.w_h2d_bytes,
            "oracle_d2h_calls": self.w_d2h_calls, "oracle_d2h_bytes": self.w_d2h_bytes,
        })
    }
}

// ── args ─────────────────────────────────────────────────────────────

struct Args {
    vocab: usize,
    seed: u64,
    windows: usize,
    replays: usize,
    timing_iters: usize,
    timing_windows: usize,
    ks: Vec<usize>,
    rows_f32: Option<String>,
    rows_vocab: usize,
    expect_arch: Option<String>,
    expect_uuid: Option<String>,
    expect_bdf: Option<String>,
    skip_timing: bool,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        vocab: 248_320,
        seed: 0x5eed_0c1e,
        windows: 1500,
        replays: 24,
        timing_iters: 60,
        timing_windows: 8,
        ks: vec![3, 7],
        rows_f32: None,
        rows_vocab: 0,
        expect_arch: None,
        expect_uuid: None,
        expect_bdf: None,
        skip_timing: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |s: &str, f: &str| -> R<u64> {
        s.parse::<u64>().map_err(|e| Fail {
            code: 2,
            msg: format!("{f}: {e}"),
        })
    };
    while i < argv.len() {
        let f = argv[i].as_str();
        if f == "--skip-timing" {
            a.skip_timing = true;
            i += 1;
            continue;
        }
        let v = argv
            .get(i + 1)
            .ok_or_else(|| Fail {
                code: 2,
                msg: format!("{f} needs a value"),
            })?
            .clone();
        match f {
            "--vocab" => a.vocab = num(&v, f)? as usize,
            "--seed" => a.seed = num(&v, f)?,
            "--windows" => a.windows = num(&v, f)? as usize,
            "--replays" => a.replays = num(&v, f)? as usize,
            "--timing-iters" => a.timing_iters = num(&v, f)? as usize,
            "--timing-windows" => a.timing_windows = num(&v, f)? as usize,
            "--ks" => {
                a.ks = v
                    .split(',')
                    .map(|s| num(s, f).map(|x| x as usize))
                    .collect::<R<Vec<_>>>()?
            }
            "--rows-f32" => a.rows_f32 = Some(v),
            "--rows-vocab" => a.rows_vocab = num(&v, f)? as usize,
            "--expect-arch" => a.expect_arch = Some(v),
            "--expect-uuid" => a.expect_uuid = Some(v),
            "--expect-bdf" => a.expect_bdf = Some(v),
            "--out" => {} // consumed by main()
            _ => return setup(format!("unknown flag {f}")),
        }
        i += 2;
    }
    if a.vocab < 64 || a.ks.iter().any(|&k| k == 0 || k > MAX_K) {
        return setup("--vocab >= 64 and --ks values in 1..=8".into());
    }
    if a.rows_f32.is_some() && a.rows_vocab == 0 {
        return setup("--rows-f32 needs --rows-vocab".into());
    }
    if a.timing_windows == 0 || a.timing_iters == 0 {
        return setup("--timing-windows/--timing-iters must be >= 1".into());
    }
    Ok(a)
}

// ── test-data rng ────────────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn gauss(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 3.4641
    }
}

// ── row shapes, policies ─────────────────────────────────────────────

const N_KINDS: usize = 16;

fn gen_row(kind: usize, v: usize, r: &mut Rng) -> Vec<f32> {
    let mut row: Vec<f32> = match kind {
        0 | 2 | 12 => (0..v).map(|_| r.gauss()).collect(),
        1 => (0..v).map(|_| r.gauss() * 4.0).collect(),
        3 => vec![0.0; v],
        4 => (0..v).map(|_| (r.gauss() * 1.5).round()).collect(),
        5 | 8 | 15 => vec![f32::NEG_INFINITY; v],
        6 => (0..v).map(|_| r.gauss() * 2.0).collect(),
        7 => (0..v)
            .map(|_| [f32::NAN, f32::NEG_INFINITY, f32::INFINITY][r.below(3)])
            .collect(),
        9 => (0..v).map(|i| i as f32 * 1e-3).collect(),
        10 => (0..v).map(|i| -(i as f32) * 1e-3).collect(),
        11 => (0..v)
            .map(|_| match r.below(4) {
                0 => 1e30,
                1 => -1e30,
                _ => 0.0,
            })
            .collect(),
        13 => (0..v).map(|_| r.gauss() / (r.unit() + 0.02)).collect(),
        14 => (0..v).map(|i| 1.0 + (i % 7) as f32 * 1e-7).collect(),
        _ => unreachable!(),
    };
    match kind {
        2 => row[r.below(v)] = 40.0,
        5 => {
            for _ in 0..3 {
                row[r.below(v)] = r.gauss() * 3.0;
            }
        }
        6 => {
            for _ in 0..(v / 40).max(3) {
                row[r.below(v)] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][r.below(3)];
            }
        }
        12 => {
            let (a, b) = (r.below(v), r.below(v));
            row[a] = 20.0;
            row[b] = 20.0;
        }
        15 => row[r.below(v)] = 1.5,
        _ => {}
    }
    row
}

fn policies() -> Vec<(SampleSpec, String)> {
    let raw: [(f32, f32, Option<u32>, f32); 15] = [
        (1.0, 1.0, None, 0.0),
        (0.7, 0.95, None, 0.0),
        (0.6, 0.9, Some(20), 0.05),
        (1.0, 0.8, Some(5), 0.0),
        (0.8, 1.0, Some(64), 0.1),
        (1.3, 0.99, Some(0), 0.02),
        (0.3, 0.5, Some(40), 0.0),
        (2.0, 0.95, Some(100), 0.5),
        (1.0, 0.0, None, 0.0),
        (1.0, 1.0, Some(1), 0.0),
        (1e-3, 0.9, None, 0.0),
        (50.0, 1.0, Some(0), 0.0),
        (1e-30, 1.0, None, 0.0),
        (1.0, 1.5, None, f32::NAN),
        (0.9, 0.9, Some(21), 1.0),
    ];
    raw.iter()
        .map(|&(t, p, k, m)| {
            (
                SampleSpec::cpu_ar(t, p, k, m),
                format!("t={t} top_p={p} top_k={k:?} min_p={m}"),
            )
        })
        .collect()
}
/// top_p = 0: p is a deterministic point mass.
const POLICY_POINT_MASS: usize = 8;

fn policy_of(s: SampleSpec) -> sc::SpecCpuArPolicy {
    sc::SpecCpuArPolicy::new(s.temperature, s.top_p, s.pool as u32, s.cap as u32, s.min_p)
}

struct Corpus {
    name: String,
    vocab: usize,
    rows: Vec<Vec<f32>>,
}

// ── device resources ─────────────────────────────────────────────────

struct Res {
    logits: GpuTensor,
    tgt: sc::SpecDistRows,
    scratch: sc::SpecDistScratch,
    drf: sc::SpecDistRows,
    cand: Vec<GpuTensor>,
    picked: Vec<GpuTensor>,
    props: GpuTensor,
    tape: GpuTensor,
    result: GpuTensor,
    cap_k: usize,
    cap_vocab: usize,
}

fn alloc_res(gpu: &mut Gpu, cap_k: usize, cap_vocab: usize) -> R<Res> {
    let logits = hip(
        gpu.alloc_tensor(&[(cap_k + 1) * cap_vocab], DType::F32),
        "alloc logits",
    )?;
    let tgt = hip(sc::SpecDistRows::new(gpu, cap_k + 1), "SpecDistRows tgt")?;
    let scratch = hip(sc::SpecDistScratch::new(gpu, cap_k + 1), "SpecDistScratch")?;
    let drf = hip(sc::SpecDistRows::new(gpu, cap_k.max(1)), "SpecDistRows drf")?;
    let mut cand = Vec::new();
    let mut picked = Vec::new();
    for _ in 0..cap_k.max(1) {
        cand.push(hip(sc::spec_cpu_ar_alloc_u32(gpu, 16), "alloc cand")?);
        picked.push(hip(sc::spec_cpu_ar_alloc_u32(gpu, 1), "alloc picked")?);
    }
    Ok(Res {
        logits,
        tgt,
        scratch,
        drf,
        cand,
        picked,
        props: hip(sc::spec_cpu_ar_alloc_u32(gpu, cap_k.max(1)), "alloc props")?,
        tape: hip(sc::spec_cpu_ar_alloc_f32(gpu, 2 * cap_k + 1), "alloc tape")?,
        result: hip(sc::spec_cpu_ar_alloc_u32(gpu, 2), "alloc result")?,
        cap_k,
        cap_vocab,
    })
}

fn free_res(gpu: &mut Gpu, res: Res) {
    let _ = res.tgt.free(gpu);
    let _ = res.drf.free(gpu);
    let _ = res.scratch.free(gpu);
    let _ = gpu.free_tensor(res.logits);
    for t in res.cand.into_iter().chain(res.picked) {
        let _ = gpu.free_tensor(t);
    }
    let _ = gpu.free_tensor(res.props);
    let _ = gpu.free_tensor(res.tape);
    let _ = gpu.free_tensor(res.result);
}

// ── windows ──────────────────────────────────────────────────────────

#[derive(Clone)]
enum QSrc {
    /// Host-built q uploaded into the draft rows (the S2 first-cut route).
    Host(SparseDist),
    /// Eight candidates; q is built on the device by the candidates enqueue.
    Cand { ids: [u32; 8], logits: [f32; 8] },
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum DKind {
    /// q = p exactly; draft from p. Accepts for every u in [0, 1).
    AcceptEq,
    /// n-gram point mass at p's top token.
    AcceptPoint,
    /// n-gram point mass outside p's support: certain rejection.
    RejectPointOut,
    /// q = p, draft outside both supports: rejection with zero residual mass.
    ZeroResidual,
    /// Device-built q from candidates, draft mostly sampled from q.
    Cand,
    /// Host-built q from candidates, uploaded.
    HostCand,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum EosMode {
    None,
    /// EOS = the token the no-EOS verdict ends the window with: replacement
    /// EOS after a rejection, bonus EOS after full acceptance.
    Terminal,
    /// EOS = the draft token at this position.
    Draft(usize),
    /// EOS = row 0's top token.
    Top0,
    Fixed(u32),
}

struct Win {
    name: String,
    corpus: usize,
    vocab: usize,
    policy: usize,
    k: usize,
    rows: Vec<usize>,
    ps: Vec<SparseDist>,
    drafts: Vec<u32>,
    qsrc: Vec<QSrc>,
    u_draft: Vec<f32>,
    eos: Option<u32>,
    rng_start: SpecRng,
    tape: Vec<f32>,
}

fn dist_bits(d: &SparseDist) -> Vec<(u32, u32)> {
    d.entries().iter().map(|&(t, p)| (t, p.to_bits())).collect()
}

fn outside_support(p: &SparseDist, vocab: usize, r: &mut Rng) -> Option<u32> {
    for _ in 0..64 {
        let t = r.below(vocab) as u32;
        if !p.entries().iter().any(|&(x, _)| x == t) {
            return Some(t);
        }
    }
    (0..vocab as u32).find(|t| !p.entries().iter().any(|&(x, _)| x == *t))
}

fn gen_cands(p: &SparseDist, vocab: usize, eos_hint: Option<u32>, r: &mut Rng) -> ([u32; 8], [f32; 8]) {
    let mut ids = [0u32; 8];
    let mut n = 0usize;
    fn put(t: u32, ids: &mut [u32; 8], n: &mut usize, vocab: usize) {
        if *n < 8 && !ids[..*n].contains(&t) && (t as usize) < vocab {
            ids[*n] = t;
            *n += 1;
        }
    }
    if let Some(e) = eos_hint {
        put(e, &mut ids, &mut n, vocab);
    }
    for &(t, _) in p.entries().iter().take(4) {
        put(t, &mut ids, &mut n, vocab);
    }
    let mut guard = 0;
    while n < 8 && guard < 4096 {
        put(r.below(vocab) as u32, &mut ids, &mut n, vocab);
        guard += 1;
    }
    let mut logits = [0f32; 8];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = r.gauss() * 3.0 + if i < 4 { 1.0 } else { 0.0 };
        if r.below(25) == 0 {
            *l = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][r.below(3)];
        }
        if r.below(12) == 0 {
            *l = r.below(3) as f32; // tie group
        }
    }
    (ids, logits)
}

fn cand_dist(ids: &[u32; 8], logits: &[f32; 8], spec: SampleSpec) -> Option<SparseDist> {
    let mut v: Vec<(u32, f32)> = ids.iter().copied().zip(logits.iter().copied()).collect();
    let mut d = SparseDist::default();
    d.build_from_candidates(&mut v, spec).ok()?;
    Some(d)
}

fn measured_consumption(before: SpecRng, after: SpecRng, max: usize) -> Option<usize> {
    let target = format!("{after:?}");
    let mut r = before;
    for n in 0..=max {
        if format!("{r:?}") == target {
            return Some(n);
        }
        r.next_u64();
    }
    None
}

struct RefOut {
    committed: Vec<u32>,
    accepted: usize,
    hit_eos: bool,
    words: [u32; 2],
    consumed: usize,
}

fn qs_host(win: &Win, spec: SampleSpec) -> Vec<SparseDist> {
    win.qsrc
        .iter()
        .map(|q| match q {
            QSrc::Host(d) => d.clone(),
            QSrc::Cand { ids, logits } => cand_dist(ids, logits, spec).expect("validated at build"),
        })
        .collect()
}

fn host_reference(win: &Win, qs: &[SparseDist], eos: Option<u32>) -> R<RefOut> {
    let mut rng = win.rng_start;
    let mut tgt = SparseDist::default();
    let ps = &win.ps;
    let acc = accept_sampled_prefix(&win.drafts, qs, eos, &mut rng, &mut tgt, |row, out| {
        *out = ps[row].clone();
        Ok(())
    })
    .map_err(|e| Fail {
        code: 2,
        msg: format!("host accept: {e}"),
    })?;
    let accepted_eos = acc.hit_eos && acc.committed.len() == acc.accepted;
    let last = acc.committed.last().copied();
    let words = [
        acc.accepted as u32,
        if accepted_eos {
            EOS_NONE
        } else {
            match last {
                Some(t) => t,
                None => return setup("empty committed".into()),
            }
        },
    ];
    let consumed = match measured_consumption(win.rng_start, rng, 2 * win.k + 2) {
        Some(c) => c,
        None => return setup("host RNG consumption not measurable".into()),
    };
    Ok(RefOut {
        committed: acc.committed,
        accepted: acc.accepted,
        hit_eos: acc.hit_eos,
        words,
        consumed,
    })
}

/// Frozen-contract RNG consumption implied by the device result words.
fn rule_consumed(words: [u32; 2], k: usize) -> usize {
    if words[1] == EOS_NONE {
        words[0] as usize
    } else if (words[0] as usize) < k {
        words[0] as usize + 2
    } else {
        k + 1
    }
}

struct Spec {
    k: usize,
    kinds: Vec<DKind>,
    eos: EosMode,
    policy: usize,
    corpus: usize,
    rows: Option<Vec<usize>>,
    seed: u64,
    /// Candidate uniforms baked by a captured graph (no draws consumed).
    fixed_u: Option<Vec<f32>>,
}

fn build_window(
    corpora: &[Corpus],
    pols: &[(SampleSpec, String)],
    sp: &Spec,
    dr: &mut Rng,
    name: String,
) -> R<Win> {
    let c = &corpora[sp.corpus];
    let spec = pols[sp.policy].0;
    let k = sp.k;
    let rows: Vec<usize> = match &sp.rows {
        Some(r) => r.clone(),
        None => (0..=k).map(|_| dr.below(c.rows.len())).collect(),
    };
    let mut scr = Vec::new();
    let mut ps = Vec::with_capacity(k + 1);
    for &ri in &rows {
        let mut d = SparseDist::default();
        d.build_from_logits(&c.rows[ri], spec, &mut scr)
            .map_err(|e| Fail {
                code: 2,
                msg: format!("host p: {e}"),
            })?;
        if d.entries().is_empty() {
            return setup(format!("{name}: host target row {ri} has no entries"));
        }
        ps.push(d);
    }
    let mut rng = SpecRng::new(sp.seed);
    let eos_hint = match sp.eos {
        EosMode::Fixed(e) => Some(e),
        _ => None,
    };
    let mut drafts = Vec::new();
    let mut qsrc = Vec::new();
    let mut u_draft = Vec::new();
    for j in 0..k {
        let p = &ps[j];
        let u = match &sp.fixed_u {
            Some(f) => f[j],
            None => rng.next_f32(),
        };
        u_draft.push(u);
        let mut kind = sp.kinds[j];
        // A support that covers the whole vocabulary leaves no outside token.
        if matches!(kind, DKind::RejectPointOut | DKind::ZeroResidual)
            && outside_support(p, c.vocab, dr).is_none()
        {
            kind = DKind::AcceptEq;
        }
        let top = p.entries()[0].0;
        let (q, x) = match kind {
            DKind::AcceptEq => {
                let e = p.entries();
                let x = e[dr.below(e.len())].0;
                (QSrc::Host(p.clone()), x)
            }
            DKind::AcceptPoint => {
                let mut q = SparseDist::default();
                q.set_point_mass(top);
                (QSrc::Host(q), top)
            }
            DKind::RejectPointOut => {
                let x = match outside_support(p, c.vocab, dr) {
                    Some(x) => x,
                    None => return setup("no outside token".into()),
                };
                let mut q = SparseDist::default();
                q.set_point_mass(x);
                (QSrc::Host(q), x)
            }
            DKind::ZeroResidual => {
                let x = match outside_support(p, c.vocab, dr) {
                    Some(x) => x,
                    None => return setup("no outside token".into()),
                };
                (QSrc::Host(p.clone()), x)
            }
            DKind::Cand | DKind::HostCand => {
                let mut found = None;
                for _ in 0..32 {
                    let (ids, logits) = gen_cands(p, c.vocab, eos_hint, dr);
                    if let Some(q) = cand_dist(&ids, &logits, spec) {
                        found = Some((ids, logits, q));
                        break;
                    }
                }
                let (ids, logits, q) = match found {
                    Some(f) => f,
                    None => return setup("no finite candidate set after 32 draws".into()),
                };
                let x = match dr.below(20) {
                    0..=14 => q.sample(u),
                    15 | 16 => top,
                    17 | 18 => outside_support(p, c.vocab, dr).unwrap_or(top),
                    _ => dr.below(c.vocab) as u32,
                };
                if kind == DKind::HostCand {
                    (QSrc::Host(q), x)
                } else {
                    (QSrc::Cand { ids, logits }, x)
                }
            }
        };
        qsrc.push(q);
        drafts.push(x);
    }
    let rng_start = rng;
    let mut tr = rng;
    let tape: Vec<f32> = (0..2 * k + 1).map(|_| tr.next_f32()).collect();
    let mut win = Win {
        name,
        corpus: sp.corpus,
        vocab: c.vocab,
        policy: sp.policy,
        k,
        rows,
        ps,
        drafts,
        qsrc,
        u_draft,
        eos: None,
        rng_start,
        tape,
    };
    win.eos = match sp.eos {
        EosMode::None => None,
        EosMode::Fixed(e) => Some(e),
        EosMode::Top0 => Some(win.ps[0].entries()[0].0),
        EosMode::Draft(j) => win.drafts.get(j).copied(),
        EosMode::Terminal => {
            let qs = qs_host(&win, spec);
            let r = host_reference(&win, &qs, None)?;
            r.committed.last().copied()
        }
    };
    Ok(win)
}

// ── device execution of a window ─────────────────────────────────────

fn upload_window(gpu: &Gpu, res: &Res, c: &Corpus, win: &Win, qs: &[SparseDist]) -> R<()> {
    let v = win.vocab;
    if win.k > res.cap_k || v > res.cap_vocab {
        return setup("window exceeds resource capacity".into());
    }
    let mut flat = Vec::with_capacity((win.k + 1) * v);
    for &ri in &win.rows {
        flat.extend_from_slice(&c.rows[ri]);
    }
    h2d(gpu, &res.logits, &f32_bytes(&flat))?;
    // Sentinels prove the device wrote every output it is responsible for.
    h2d(gpu, &res.tgt.counts, &u32_bytes(&vec![SENT; res.tgt.capacity_rows]))?;
    let dc = res.drf.capacity_rows;
    let mut counts = vec![SENT; dc];
    let mut ids = vec![0u32; dc * CAP];
    let mut probs = vec![0f32; dc * CAP];
    for j in 0..win.k {
        match &win.qsrc[j] {
            QSrc::Host(_) => {
                let e = qs[j].entries();
                counts[j] = e.len() as u32;
                for (i, &(t, p)) in e.iter().enumerate() {
                    ids[j * CAP + i] = t;
                    probs[j * CAP + i] = p;
                }
            }
            QSrc::Cand { ids: cids, logits } => {
                let mut w = [0u32; 16];
                w[..8].copy_from_slice(cids);
                for i in 0..8 {
                    w[8 + i] = logits[i].to_bits();
                }
                h2d(gpu, &res.cand[j], &u32_bytes(&w))?;
                h2d(gpu, &res.picked[j], &u32_bytes(&[SENT]))?;
            }
        }
    }
    h2d(gpu, &res.drf.counts, &u32_bytes(&counts))?;
    h2d(gpu, &res.drf.ids, &u32_bytes(&ids))?;
    h2d(gpu, &res.drf.probabilities, &f32_bytes(&probs))?;
    let mut props = vec![0u32; res.cap_k.max(1)];
    props[..win.k].copy_from_slice(&win.drafts);
    h2d(gpu, &res.props, &u32_bytes(&props))?;
    h2d(gpu, &res.tape, &f32_bytes(&win.tape))?;
    h2d(gpu, &res.result, &u32_bytes(&[0xDEAD_BEEF, 0xDEAD_BEEF]))?;
    Ok(())
}

/// Per enqueue call: (name, capture-blob delta). Blob deltas are meaningful
/// only while a capture is open.
type CallNodes = Vec<(String, usize)>;

/// The resident enqueue sequence under test. `cand_us[j] = Some(u)` enqueues
/// the device candidate q for draft row `j`.
fn enqueue_window(
    gpu: &mut Gpu,
    res: &Res,
    vocab: usize,
    k: usize,
    policy: sc::SpecCpuArPolicy,
    eos: u32,
    cand_us: &[Option<f32>],
) -> R<CallNodes> {
    let mut nodes: CallNodes = Vec::new();
    let b0 = gpu.graphs.capture_blobs.len();
    hip(
        gpu.spec_cpu_ar_rows_enqueue(&res.logits, &res.tgt, &res.scratch, vocab, k + 1, policy),
        "spec_cpu_ar_rows_enqueue",
    )?;
    let mut last = gpu.graphs.capture_blobs.len();
    nodes.push(("rows".into(), last - b0));
    for j in 0..k {
        if let Some(u) = cand_us[j] {
            hip(
                gpu.spec_cpu_ar_candidates_enqueue(
                    &res.cand[j],
                    &res.drf,
                    j,
                    vocab,
                    policy,
                    u,
                    &res.picked[j],
                ),
                "spec_cpu_ar_candidates_enqueue",
            )?;
            let now = gpu.graphs.capture_blobs.len();
            nodes.push((format!("candidates[{j}]"), now - last));
            last = now;
        }
    }
    hip(
        gpu.spec_accept_ordered_enqueue(&res.tgt, &res.drf, &res.props, &res.tape, &res.result, k, eos),
        "spec_accept_ordered_enqueue",
    )?;
    let now = gpu.graphs.capture_blobs.len();
    nodes.push(("accept".into(), now - last));
    Ok(nodes)
}

fn eos_word(e: Option<u32>) -> u32 {
    e.unwrap_or(EOS_NONE)
}

fn cand_us_of(win: &Win) -> Vec<Option<f32>> {
    (0..win.k)
        .map(|j| match win.qsrc[j] {
            QSrc::Cand { .. } => Some(win.u_draft[j]),
            QSrc::Host(_) => None,
        })
        .collect()
}

#[derive(Default)]
struct Stats {
    windows: u64,
    rows_compared: u64,
    prob_bits_compared: u64,
    id_words_compared: u64,
    cand_rows_compared: u64,
    result_words_compared: u64,
    rng_checks: u64,
    enq_rows: u64,
    enq_cand: u64,
    enq_accept: u64,
    enq_traffic: Tally,
    cov: BTreeMap<String, u64>,
}
impl Stats {
    fn bump(&mut self, key: &str) {
        *self.cov.entry(key.to_string()).or_insert(0) += 1;
    }
}

fn check_rows(
    label: &str,
    counts: &[u32],
    ids: &[u32],
    probs: &[u32],
    row0: usize,
    dists: &[&SparseDist],
    st: &mut Stats,
) -> R<()> {
    for (r, d) in dists.iter().enumerate() {
        let row = row0 + r;
        let e = d.entries();
        if counts[row] as usize != e.len() {
            return mismatch(format!(
                "{label}: row {row} count device={} host={} (host entries {:?})",
                counts[row],
                e.len(),
                dist_bits(d)
            ));
        }
        for (i, &(t, p)) in e.iter().enumerate() {
            let (dt, dp) = (ids[row * CAP + i], probs[row * CAP + i]);
            if dt != t || dp != p.to_bits() {
                return mismatch(format!(
                    "{label}: row {row} entry {i} device=(id {dt}, bits {dp:#010x}={}) host=(id {t}, bits {:#010x}={})",
                    f32::from_bits(dp),
                    p.to_bits(),
                    p
                ));
            }
        }
        st.rows_compared += 1;
        st.id_words_compared += e.len() as u64;
        st.prob_bits_compared += e.len() as u64;
    }
    Ok(())
}

struct DevRead {
    t_counts: Vec<u32>,
    t_ids: Vec<u32>,
    t_probs: Vec<u32>,
    d_counts: Vec<u32>,
    d_ids: Vec<u32>,
    d_probs: Vec<u32>,
    picked: Vec<u32>,
    result: [u32; 2],
}

fn read_back(gpu: &Gpu, res: &Res, k: usize) -> R<DevRead> {
    sync(gpu)?;
    let tc = res.tgt.capacity_rows;
    let dc = res.drf.capacity_rows;
    let mut picked = Vec::new();
    for j in 0..k {
        picked.push(bytes_u32(&d2h(gpu, &res.picked[j], 4)?)[0]);
    }
    let r = bytes_u32(&d2h(gpu, &res.result, 8)?);
    Ok(DevRead {
        t_counts: bytes_u32(&d2h(gpu, &res.tgt.counts, tc * 4)?),
        t_ids: bytes_u32(&d2h(gpu, &res.tgt.ids, tc * CAP * 4)?),
        t_probs: bytes_u32(&d2h(gpu, &res.tgt.probabilities, tc * CAP * 4)?),
        d_counts: bytes_u32(&d2h(gpu, &res.drf.counts, dc * 4)?),
        d_ids: bytes_u32(&d2h(gpu, &res.drf.ids, dc * CAP * 4)?),
        d_probs: bytes_u32(&d2h(gpu, &res.drf.probabilities, dc * CAP * 4)?),
        picked,
        result: [r[0], r[1]],
    })
}

fn classify(win: &Win, qs: &[SparseDist], rf: &RefOut, st: &mut Stats) {
    let k = win.k;
    let accepted_eos = rf.hit_eos && rf.committed.len() == rf.accepted;
    let rej = !accepted_eos && rf.accepted < k;
    st.bump("windows");
    if k == 0 {
        st.bump("k0");
        if rf.hit_eos {
            st.bump("k0_bonus_eos");
        }
    }
    if k > 0 && !rej && !accepted_eos {
        st.bump("all_accepted_bonus");
    }
    if rej {
        if rf.accepted == 0 {
            st.bump("reject_first");
        }
        if rf.accepted + 1 == k && k >= 2 {
            st.bump("reject_last");
        }
        if rf.accepted > 0 && rf.accepted + 1 < k {
            st.bump("reject_mid");
        }
        if rf.hit_eos {
            st.bump("replacement_eos");
        }
    }
    if accepted_eos {
        st.bump("accepted_eos");
        if rf.accepted == 1 {
            st.bump("accepted_eos_first");
        }
        if rf.accepted == k {
            st.bump("accepted_eos_last");
        }
    }
    if !rej && !accepted_eos && rf.accepted == k && rf.hit_eos {
        st.bump("bonus_eos");
    }
    let nverd = if rej { rf.accepted + 1 } else { rf.accepted };
    for j in 0..nverd.min(k) {
        let (p, q) = (&win.ps[j], &qs[j]);
        let accepted_here = j < rf.accepted;
        if matches!(win.qsrc[j], QSrc::Cand { .. }) {
            st.bump("visited_device_cand_q");
        }
        if dist_bits(p) == dist_bits(q) && accepted_here {
            st.bump("p_eq_q_accept");
        }
        if q.entries().len() == 1 {
            st.bump(if accepted_here {
                "point_mass_q_accept"
            } else {
                "point_mass_q_reject"
            });
        }
        if !accepted_here {
            let mass: f32 = p
                .entries()
                .iter()
                .map(|&(t, pt)| (pt - q.prob(t)).max(0.0))
                .sum();
            st.bump(if !(mass > 0.0) {
                "zero_residual_reject"
            } else {
                "positive_residual_reject"
            });
            if q.prob(win.drafts[j]) == 0.0 {
                st.bump("reject_draft_outside_q");
            }
            if p.prob(win.drafts[j]) == 0.0 {
                st.bump("reject_px_zero");
            }
        }
    }
}

/// Compare one executed window (direct or replayed) against the host.
fn check_window(
    gpu: &Gpu,
    res: &Res,
    win: &Win,
    spec: SampleSpec,
    label: &str,
    st: &mut Stats,
) -> R<RefOut> {
    let qs = qs_host(win, spec);
    let rf = host_reference(win, &qs, win.eos)?;
    let dev = read_back(gpu, res, win.k)?;
    let k = win.k;
    let ctx = format!("{label} [{}]", win.name);
    check_rows(
        &format!("{ctx} target"),
        &dev.t_counts,
        &dev.t_ids,
        &dev.t_probs,
        0,
        &win.ps.iter().collect::<Vec<_>>(),
        st,
    )?;
    check_rows(
        &format!("{ctx} draft"),
        &dev.d_counts,
        &dev.d_ids,
        &dev.d_probs,
        0,
        &qs.iter().collect::<Vec<_>>(),
        st,
    )?;
    for j in 0..k {
        if let QSrc::Cand { .. } = win.qsrc[j] {
            let want = qs[j].sample(win.u_draft[j]);
            if dev.picked[j] != want {
                return mismatch(format!(
                    "{ctx}: candidate pick row {j} device={} host={want} (u={})",
                    dev.picked[j], win.u_draft[j]
                ));
            }
            st.cand_rows_compared += 1;
        }
    }
    if dev.result != rf.words {
        return mismatch(format!(
            "{ctx}: accept result device={:?} host={:?} (k={k} eos={:?} drafts={:?} committed={:?} accepted={})",
            dev.result, rf.words, win.eos, win.drafts, rf.committed, rf.accepted
        ));
    }
    st.result_words_compared += 2;
    let rule = rule_consumed(dev.result, k);
    if rule != rf.consumed {
        return mismatch(format!(
            "{ctx}: RNG consumption contract rule={rule} host measured={} (words {:?})",
            rf.consumed, dev.result
        ));
    }
    st.rng_checks += 1;
    // Host-side reconstruction from the two words, as the integration will do.
    let mut rebuilt: Vec<u32> = win.drafts[..dev.result[0] as usize].to_vec();
    let hit = if dev.result[1] == EOS_NONE {
        true
    } else {
        rebuilt.push(dev.result[1]);
        win.eos == Some(dev.result[1])
    };
    if rebuilt != rf.committed || hit != rf.hit_eos {
        return mismatch(format!(
            "{ctx}: committed reconstruction {rebuilt:?}/{hit} != host {:?}/{}",
            rf.committed, rf.hit_eos
        ));
    }
    classify(win, &qs, &rf, st);
    Ok(rf)
}

fn run_window(
    gpu: &mut Gpu,
    res: &Res,
    corpora: &[Corpus],
    pols: &[(SampleSpec, String)],
    win: &Win,
    label: &str,
    st: &mut Stats,
) -> R<()> {
    let spec = pols[win.policy].0;
    let qs = qs_host(win, spec);
    upload_window(gpu, res, &corpora[win.corpus], win, &qs)?;
    sync(gpu)?;
    let cu = cand_us_of(win);
    let before = tally();
    enqueue_window(gpu, res, win.vocab, win.k, policy_of(spec), eos_word(win.eos), &cu)?;
    let during = tally().since(before);
    if during.any_host_traffic() {
        return mismatch(format!(
            "{label} [{}]: transfer/sync issued inside enqueue calls: {}",
            win.name,
            during.json()
        ));
    }
    st.enq_traffic.add(during);
    st.enq_rows += 1;
    st.enq_cand += cu.iter().flatten().count() as u64;
    st.enq_accept += 1;
    check_window(gpu, res, win, spec, label, st)?;
    st.windows += 1;
    Ok(())
}

// ── distribution oracle ──────────────────────────────────────────────

fn run_distribution(
    gpu: &mut Gpu,
    corpora_v: &[(usize, Vec<Vec<f32>>)],
    pols: &[(SampleSpec, String)],
    st: &mut Stats,
) -> R<Value> {
    let max_rows = corpora_v
        .iter()
        .map(|(_, r)| r.len())
        .max()
        .unwrap_or(1)
        .clamp(1, 32);
    let cap_v = corpora_v.iter().map(|(v, _)| *v).max().unwrap_or(1);
    let res = alloc_res(gpu, max_rows - 1, cap_v)?;
    let mut cases = 0u64;
    for (vocab, rows) in corpora_v {
        for chunk in rows.chunks(max_rows) {
            for (pi, (spec, pname)) in pols.iter().enumerate() {
                let mut flat = Vec::with_capacity(chunk.len() * vocab);
                for r in chunk {
                    flat.extend_from_slice(r);
                }
                h2d(gpu, &res.logits, &f32_bytes(&flat))?;
                h2d(gpu, &res.tgt.counts, &u32_bytes(&vec![SENT; res.tgt.capacity_rows]))?;
                sync(gpu)?;
                let before = tally();
                hip(
                    gpu.spec_cpu_ar_rows_enqueue(
                        &res.logits,
                        &res.tgt,
                        &res.scratch,
                        *vocab,
                        chunk.len(),
                        policy_of(*spec),
                    ),
                    "rows_enqueue",
                )?;
                let during = tally().since(before);
                if during.any_host_traffic() {
                    return mismatch(format!(
                        "distribution: transfer/sync inside rows enqueue: {}",
                        during.json()
                    ));
                }
                st.enq_traffic.add(during);
                st.enq_rows += 1;
                sync(gpu)?;
                let tc = res.tgt.capacity_rows;
                let counts = bytes_u32(&d2h(gpu, &res.tgt.counts, tc * 4)?);
                let ids = bytes_u32(&d2h(gpu, &res.tgt.ids, tc * CAP * 4)?);
                let probs = bytes_u32(&d2h(gpu, &res.tgt.probabilities, tc * CAP * 4)?);
                let mut scr = Vec::new();
                for (ri, r) in chunk.iter().enumerate() {
                    let mut d = SparseDist::default();
                    d.build_from_logits(r, *spec, &mut scr)
                        .map_err(|e| Fail { code: 2, msg: e })?;
                    check_rows(
                        &format!("distribution vocab={vocab} policy[{pi}]({pname}) row {ri}"),
                        &counts,
                        &ids,
                        &probs,
                        ri,
                        &[&d],
                        st,
                    )?;
                }
                cases += 1;
            }
        }
    }
    free_res(gpu, res);
    Ok(json!({
        "cases": cases,
        "vocabs": corpora_v.iter().map(|(v, r)| json!({"vocab": v, "rows": r.len()})).collect::<Vec<_>>(),
    }))
}

// ── directed + random windows ────────────────────────────────────────

fn directed_specs() -> Vec<(String, usize, Vec<DKind>, EosMode)> {
    use DKind::*;
    let mut v: Vec<(String, usize, Vec<DKind>, EosMode)> = Vec::new();
    let mut push = |n: &str, ks: Vec<DKind>, e: EosMode| {
        v.push((n.to_string(), ks.len(), ks, e));
    };
    push("k0", vec![], EosMode::None);
    push("k0_bonus_eos", vec![], EosMode::Terminal);
    push("k1_accept_eq", vec![AcceptEq], EosMode::None);
    push("k1_accepted_eos_last", vec![AcceptEq], EosMode::Draft(0));
    push("k1_bonus_eos", vec![AcceptEq], EosMode::Terminal);
    push("k1_reject_point_out", vec![RejectPointOut], EosMode::None);
    push("k1_replacement_eos", vec![RejectPointOut], EosMode::Terminal);
    push("k1_zero_residual", vec![ZeroResidual], EosMode::None);
    push("k1_zero_residual_eos", vec![ZeroResidual], EosMode::Terminal);
    push("k1_ngram_point", vec![AcceptPoint], EosMode::None);
    push("k1_cand", vec![Cand], EosMode::None);
    push("k3_reject_first", vec![RejectPointOut, AcceptEq, AcceptEq], EosMode::None);
    push("k3_reject_mid", vec![AcceptEq, RejectPointOut, AcceptEq], EosMode::None);
    push("k3_reject_last", vec![AcceptEq, AcceptEq, RejectPointOut], EosMode::None);
    push("k3_reject_last_eos", vec![AcceptEq, AcceptEq, RejectPointOut], EosMode::Terminal);
    push("k3_all_accept", vec![AcceptEq, AcceptEq, AcceptEq], EosMode::None);
    push("k3_all_accept_bonus_eos", vec![AcceptEq, AcceptEq, AcceptEq], EosMode::Terminal);
    push("k3_accepted_eos_first", vec![AcceptEq, AcceptEq, AcceptEq], EosMode::Draft(0));
    push("k3_accepted_eos_last", vec![AcceptEq, AcceptEq, AcceptEq], EosMode::Draft(2));
    push("k3_zero_residual_mid", vec![AcceptEq, ZeroResidual, AcceptEq], EosMode::None);
    push("k3_zero_residual_last", vec![AcceptEq, AcceptEq, ZeroResidual], EosMode::None);
    push("k3_ngram_chain", vec![AcceptPoint, AcceptPoint, AcceptPoint], EosMode::None);
    push("k3_ngram_chain_eos", vec![AcceptPoint, AcceptPoint, AcceptPoint], EosMode::Terminal);
    push("k3_ngram_reject", vec![AcceptPoint, RejectPointOut, AcceptPoint], EosMode::None);
    push("k3_cand_mix", vec![Cand, HostCand, Cand], EosMode::None);
    push("k3_cand_eos", vec![Cand, Cand, Cand], EosMode::Terminal);
    push("k8_all_accept", vec![AcceptEq; 8], EosMode::None);
    push(
        "k8_reject_last",
        vec![AcceptEq, AcceptEq, AcceptEq, AcceptEq, AcceptEq, AcceptEq, AcceptEq, RejectPointOut],
        EosMode::None,
    );
    push("k8_cand", vec![Cand; 8], EosMode::None);
    push("k8_accepted_eos_last", vec![AcceptEq; 8], EosMode::Draft(7));
    v
}

fn run_windows(
    gpu: &mut Gpu,
    res: &Res,
    corpora: &[Corpus],
    pols: &[(SampleSpec, String)],
    args: &Args,
    st: &mut Stats,
) -> R<Value> {
    let mut dr = Rng(args.seed ^ 0xA11CE);
    let mut seed_ctr = args.seed;
    let mut directed = 0u64;
    // Directed windows under four policies, including the point-mass policy
    // that makes p deterministic.
    let dir_pols = [1usize, 2, POLICY_POINT_MASS, 5];
    for (name, k, kinds, eos) in directed_specs() {
        for (ci, c) in corpora.iter().enumerate() {
            if c.vocab < 64 && !(name.starts_with("k3") || name.starts_with("k1")) {
                continue;
            }
            for &pi in &dir_pols {
                seed_ctr = seed_ctr.wrapping_add(0x1234_5677);
                let sp = Spec {
                    k,
                    kinds: kinds.clone(),
                    eos,
                    policy: pi,
                    corpus: ci,
                    rows: None,
                    seed: seed_ctr,
                    fixed_u: None,
                };
                let win = build_window(corpora, pols, &sp, &mut dr, format!("directed:{name}:{}:p{pi}", c.name))?;
                run_window(gpu, res, corpora, pols, &win, "directed", st)?;
                directed += 1;
            }
        }
    }
    let mut randomized = 0u64;
    for w in 0..args.windows {
        let k = dr.below(MAX_K + 1);
        let ci = if corpora.len() > 1 && dr.below(8) == 0 {
            1 + dr.below(corpora.len() - 1)
        } else {
            0
        };
        let kinds: Vec<DKind> = (0..k)
            .map(|_| match dr.below(100) {
                0..=19 => DKind::AcceptEq,
                20..=31 => DKind::AcceptPoint,
                32..=41 => DKind::RejectPointOut,
                42..=49 => DKind::ZeroResidual,
                50..=89 => DKind::Cand,
                _ => DKind::HostCand,
            })
            .collect();
        let eos = match dr.below(100) {
            0..=59 => EosMode::None,
            60..=74 => EosMode::Terminal,
            75..=89 if k > 0 => EosMode::Draft(dr.below(k)),
            _ => EosMode::Top0,
        };
        let pi = dr.below(pols.len());
        seed_ctr = seed_ctr.wrapping_add(0x1234_5677);
        let sp = Spec {
            k,
            kinds,
            eos,
            policy: pi,
            corpus: ci,
            rows: None,
            seed: seed_ctr,
            fixed_u: None,
        };
        let win = build_window(corpora, pols, &sp, &mut dr, format!("random#{w}"))?;
        run_window(gpu, res, corpora, pols, &win, "random", st)?;
        randomized += 1;
    }
    Ok(json!({"directed": directed, "randomized": randomized}))
}

const REQUIRED_COVERAGE: &[&str] = &[
    "k0",
    "k0_bonus_eos",
    "all_accepted_bonus",
    "reject_first",
    "reject_mid",
    "reject_last",
    "accepted_eos",
    "accepted_eos_first",
    "accepted_eos_last",
    "replacement_eos",
    "bonus_eos",
    "p_eq_q_accept",
    "zero_residual_reject",
    "positive_residual_reject",
    "point_mass_q_accept",
    "point_mass_q_reject",
    "reject_draft_outside_q",
    "reject_px_zero",
    "visited_device_cand_q",
];

// ── candidates: invalid case ─────────────────────────────────────────

/// All eight candidates non-finite: the host build errors; the device must
/// report the invalid pick and an empty row.
fn run_candidate_invalid(gpu: &mut Gpu, res: &Res, pols: &[(SampleSpec, String)], vocab: usize) -> R<Value> {
    let spec = pols[0].0;
    let ids: [u32; 8] = [3, 5, 8, 13, 21, 34, 55, 89];
    let nan = f32::NAN;
    let (pi, ni) = (f32::INFINITY, f32::NEG_INFINITY);
    let logits = [nan, pi, ni, nan, nan, pi, ni, nan];
    if cand_dist(&ids, &logits, spec).is_some() {
        return setup("host unexpectedly built q from all-non-finite candidates".into());
    }
    let mut w = [0u32; 16];
    w[..8].copy_from_slice(&ids);
    for i in 0..8 {
        w[8 + i] = logits[i].to_bits();
    }
    h2d(gpu, &res.cand[0], &u32_bytes(&w))?;
    h2d(gpu, &res.picked[0], &u32_bytes(&[0x1234_5678]))?;
    h2d(gpu, &res.drf.counts, &u32_bytes(&vec![0x7777u32; res.drf.capacity_rows]))?;
    sync(gpu)?;
    hip(
        gpu.spec_cpu_ar_candidates_enqueue(&res.cand[0], &res.drf, 0, vocab, policy_of(spec), 0.5, &res.picked[0]),
        "candidates_enqueue(invalid)",
    )?;
    sync(gpu)?;
    let picked = bytes_u32(&d2h(gpu, &res.picked[0], 4)?)[0];
    let counts = bytes_u32(&d2h(gpu, &res.drf.counts, res.drf.capacity_rows * 4)?);
    if picked != u32::MAX || counts[0] != 0 {
        return mismatch(format!(
            "invalid candidates: device picked={picked:#x} counts[0]={} (want u32::MAX, 0)",
            counts[0]
        ));
    }
    Ok(json!({"checked": true, "picked": picked, "count": counts[0]}))
}

// ── capture / replay ─────────────────────────────────────────────────

struct Captured {
    graph: Option<Graph>,
    exec: Option<GraphExec>,
    blobs: Vec<Vec<u8>>,
    calls: CallNodes,
    capture_kernel_launches: u64,
}

fn capture(
    gpu: &mut Gpu,
    res: &Res,
    vocab: usize,
    k: usize,
    policy: sc::SpecCpuArPolicy,
    eos: u32,
    cu: &[Option<f32>],
) -> R<Captured> {
    // Direct warm run first: every kernel is compiled and module-loaded
    // before the stream enters capture.
    enqueue_window(gpu, res, vocab, k, policy, eos, cu)?;
    sync(gpu)?;
    let before = tally();
    {
        let s = match gpu.active_stream.as_ref() {
            Some(s) => s,
            None => return setup("no active stream".into()),
        };
        hip(
            gpu.graphs.begin_graph_capture(&gpu.hip, gpu.device_id, s),
            "begin_graph_capture",
        )?;
    }
    let calls = match enqueue_window(gpu, res, vocab, k, policy, eos, cu) {
        Ok(c) => c,
        Err(e) => {
            if let Some(s) = gpu.active_stream.as_ref() {
                gpu.graphs.abort_graph_capture(&gpu.hip, gpu.device_id, s);
            }
            return Err(e);
        }
    };
    {
        let s = match gpu.active_stream.as_ref() {
            Some(s) => s,
            None => return setup("no active stream".into()),
        };
        hip(
            gpu.graphs.end_graph_capture(&gpu.hip, gpu.device_id, s),
            "end_graph_capture",
        )?;
    }
    let during = tally().since(before);
    if during.any_host_traffic() {
        return mismatch(format!(
            "capture window contained a transfer/sync: {}",
            during.json()
        ));
    }
    let graph = gpu.graphs.captured_graph.take();
    let exec = gpu.graphs.graph_exec.take();
    let blobs = std::mem::take(&mut gpu.graphs.ar_forward_blobs);
    if graph.is_none() || exec.is_none() {
        return setup("capture produced no graph".into());
    }
    let cap = Captured {
        graph,
        exec,
        blobs,
        calls,
        capture_kernel_launches: during.kernel_launches,
    };
    for (name, n) in &cap.calls {
        if *n == 0 {
            let msg = format!("enqueue `{name}` recorded no kernel node during capture");
            free_captured(gpu, cap);
            return setup(msg);
        }
    }
    let total: usize = cap.calls.iter().map(|c| c.1).sum();
    if total != cap.blobs.len() {
        let msg = format!(
            "capture node accounting: per-call sum {total} != retained blobs {}",
            cap.blobs.len()
        );
        free_captured(gpu, cap);
        return setup(msg);
    }
    Ok(cap)
}

fn free_captured(gpu: &Gpu, mut c: Captured) {
    if let Some(e) = c.exec.take() {
        let _ = gpu.hip.graph_exec_destroy(e);
    }
    if let Some(g) = c.graph.take() {
        let _ = gpu.hip.graph_destroy(g);
    }
    drop(c.blobs);
}

fn replay(gpu: &Gpu, c: &Captured) -> R<()> {
    let exec = c.exec.as_ref().ok_or_else(|| Fail {
        code: 2,
        msg: "no exec".into(),
    })?;
    hip(gpu.hip.graph_launch(exec, stream_of(gpu)?), "graph_launch")
}

fn run_graph_oracle(
    gpu: &mut Gpu,
    corpora: &[Corpus],
    pols: &[(SampleSpec, String)],
    args: &Args,
    st: &mut Stats,
) -> R<Value> {
    let eos_tok: u32 = 7;
    // (k, policy, eos)
    let configs: [(usize, usize, Option<u32>); 6] = [
        (3, 1, None),
        (3, 2, Some(eos_tok)),
        (7, 0, None),
        (1, POLICY_POINT_MASS, Some(eos_tok)),
        (8, 5, None),
        (2, 4, Some(eos_tok)),
    ];
    let mut out = Vec::new();
    let mut dr = Rng(args.seed ^ 0x6A9);
    for (gi, &(k, pi, eos)) in configs.iter().enumerate() {
        let spec = pols[pi].0;
        let policy = policy_of(spec);
        let vocab = corpora[0].vocab;
        let res = alloc_res(gpu, k, vocab)?;
        let fixed_u: Vec<f32> = (0..k).map(|_| dr.unit()).collect();
        let cu: Vec<Option<f32>> = fixed_u.iter().map(|&u| Some(u)).collect();
        let eos_mode = eos.map_or(EosMode::None, EosMode::Fixed);
        // Seed the buffers with a valid window so the warm run is defined.
        let first = Spec {
            k,
            kinds: vec![DKind::Cand; k],
            eos: eos_mode,
            policy: pi,
            corpus: 0,
            rows: None,
            seed: args.seed ^ gi as u64,
            fixed_u: Some(fixed_u.clone()),
        };
        let w0 = build_window(corpora, pols, &first, &mut dr, format!("graph{gi}:seed"))?;
        let qs0 = qs_host(&w0, spec);
        upload_window(gpu, &res, &corpora[0], &w0, &qs0)?;
        sync(gpu)?;
        let cap = capture(gpu, &res, vocab, k, policy, eos_word(eos), &cu)?;
        let phase0 = tally();
        let mut exact = 0u64;
        let mut accepted_hist = [0u64; MAX_K + 2];
        let mut graph_acc_eos = 0u64;
        for rp in 0..args.replays {
            let sp = Spec {
                k,
                kinds: vec![DKind::Cand; k],
                eos: eos_mode,
                policy: pi,
                corpus: 0,
                rows: Some((0..=k).map(|_| dr.below(corpora[0].rows.len())).collect()),
                seed: dr.next(),
                fixed_u: Some(fixed_u.clone()),
            };
            let win = build_window(corpora, pols, &sp, &mut dr, format!("graph{gi}:replay{rp}"))?;
            // Mutate every resident input, then replay the captured graph.
            let qs = qs_host(&win, spec);
            upload_window(gpu, &res, &corpora[0], &win, &qs)?;
            sync(gpu)?;
            let before = tally();
            replay(gpu, &cap)?;
            let d = tally().since(before);
            if d.graph_launch != 1 || d.dtoh + d.dtod + d.memset + d.htod != 0 {
                return mismatch(format!("graph{gi} replay {rp}: unexpected host traffic {}", d.json()));
            }
            let rf = check_window(gpu, &res, &win, spec, &format!("graph{gi} replay"), st)?;
            accepted_hist[rf.accepted] += 1;
            if rf.hit_eos && rf.committed.len() == rf.accepted {
                graph_acc_eos += 1;
            }
            exact += 1;
            st.windows += 1;
        }
        let total = tally().since(phase0);
        out.push(json!({
            "graph": gi, "k": k, "policy": pols[pi].1, "eos": eos,
            "nodes": cap.blobs.len(),
            "per_call_nodes": cap.calls.iter().map(|(n, c)| json!({"call": n, "nodes": c})).collect::<Vec<_>>(),
            "capture_kernel_launch_counter": cap.capture_kernel_launches,
            "capture_window_host_traffic": "none (htod/dtod/dtoh/memset/stream/event/device sync all zero)",
            "replays": args.replays, "exact_replays": exact,
            "replay_accepted_histogram": accepted_hist.to_vec(),
            "replay_accepted_eos": graph_acc_eos,
            "replay_phase_tally_including_uploads_and_readbacks": total.json(),
        }));
        free_captured(gpu, cap);
        free_res(gpu, res);
    }
    Ok(json!(out))
}

// ── timing ───────────────────────────────────────────────────────────

fn pct(v: &[f64], p: f64) -> f64 {
    v[((v.len() - 1) as f64 * p).round() as usize]
}
fn stat_json(mut v: Vec<f64>) -> Value {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    json!({"n": v.len(), "min_us": v[0], "p10_us": pct(&v, 0.1), "p50_us": pct(&v, 0.5),
           "mean_us": mean, "p90_us": pct(&v, 0.9), "max_us": v[v.len() - 1]})
}

struct TW {
    res: Res,
    win: Win,
    qs: Vec<SparseDist>,
    words: [u32; 2],
    visited: usize,
    q_counts: Vec<u8>,
    q_ids: Vec<u8>,
    q_probs: Vec<u8>,
    props_b: Vec<u8>,
    tape_b: Vec<u8>,
    g_host_q: Option<Captured>,
    g_cand: Option<Captured>,
}

fn run_timing(
    gpu: &mut Gpu,
    corpora: &[Corpus],
    pols: &[(SampleSpec, String)],
    args: &Args,
) -> R<Value> {
    // Real rows when supplied, else the main synthetic corpus.
    let tci = if corpora.len() > 2 { corpora.len() - 1 } else { 0 };
    let c = &corpora[tci];
    let vocab = c.vocab;
    let pi = 1usize; // 0.7 / 0.95 / pool 20: the AR default family
    let spec = pols[pi].0;
    let policy = policy_of(spec);
    let mut dr = Rng(args.seed ^ 0x7131);
    let mut out = Vec::new();
    let ev0 = hip(gpu.hip.event_create(), "event_create")?;
    let ev1 = hip(gpu.hip.event_create(), "event_create")?;
    for &k in &args.ks {
        let mut tws: Vec<TW> = Vec::new();
        for w in 0..args.timing_windows {
            let res = alloc_res(gpu, k, vocab)?;
            let fixed_u: Vec<f32> = (0..k).map(|_| dr.unit()).collect();
            let sp = Spec {
                k,
                kinds: vec![DKind::Cand; k],
                eos: EosMode::None,
                policy: pi,
                corpus: tci,
                rows: None,
                seed: dr.next(),
                fixed_u: Some(fixed_u),
            };
            let win = build_window(corpora, pols, &sp, &mut dr, format!("timing:k{k}:w{w}"))?;
            let qs = qs_host(&win, spec);
            let rf = host_reference(&win, &qs, win.eos)?;
            let accepted_eos = rf.hit_eos && rf.committed.len() == rf.accepted;
            let visited = if accepted_eos {
                rf.accepted
            } else if rf.accepted < k {
                rf.accepted + 1
            } else {
                k + 1
            };
            let mut counts = vec![0u32; k];
            let mut ids = vec![0u32; k * CAP];
            let mut probs = vec![0f32; k * CAP];
            for j in 0..k {
                let e = qs[j].entries();
                counts[j] = e.len() as u32;
                for (i, &(t, p)) in e.iter().enumerate() {
                    ids[j * CAP + i] = t;
                    probs[j * CAP + i] = p;
                }
            }
            let cu = cand_us_of(&win);
            let tw = TW {
                res,
                qs,
                words: rf.words,
                visited,
                q_counts: u32_bytes(&counts),
                q_ids: u32_bytes(&ids),
                q_probs: f32_bytes(&probs),
                props_b: u32_bytes(&win.drafts),
                tape_b: f32_bytes(&win.tape),
                win,
                g_host_q: None,
                g_cand: None,
            };
            // Resident inputs: logits and candidate tensors are uploaded once
            // (they are produced on device by verify / draft in production).
            upload_window(gpu, &tw.res, c, &tw.win, &tw.qs)?;
            sync(gpu)?;
            let none: Vec<Option<f32>> = vec![None; k];
            let g1 = capture(gpu, &tw.res, vocab, k, policy, EOS_NONE, &none)?;
            let g2 = capture(gpu, &tw.res, vocab, k, policy, EOS_NONE, &cu)?;
            tws.push(TW {
                g_host_q: Some(g1),
                g_cand: Some(g2),
                ..tw
            });
        }
        let iters = args.timing_iters;
        let warm = 3usize;
        let nw = tws.len();
        let nwin = (iters * nw) as f64;
        let mut variants = Map::new();

        // ── existing route: D2H visited rows + host law ──
        {
            let mut buf = vec![0f32; vocab];
            let mut total = Vec::new();
            let mut dl = Vec::new();
            let mut law = Vec::new();
            let mut t0 = tally();
            for it in 0..(warm + iters) {
                if it == warm {
                    t0 = tally();
                    total.clear();
                    dl.clear();
                    law.clear();
                }
                for tw in tws.iter() {
                    let mut dl_acc = 0f64;
                    let mut law_acc = 0f64;
                    let mut rng = tw.win.rng_start;
                    let mut tgt = SparseDist::default();
                    let mut scr = Vec::new();
                    let mut err: Option<Fail> = None;
                    let t = Instant::now();
                    let acc = accept_sampled_prefix(
                        &tw.win.drafts,
                        &tw.qs,
                        tw.win.eos,
                        &mut rng,
                        &mut tgt,
                        |row, o| {
                            let a = Instant::now();
                            if let Err(e) = d2h_row_at(gpu, &tw.res.logits, row * vocab * 4, &mut buf) {
                                err = Some(e);
                                return Err("d2h".to_string());
                            }
                            dl_acc += a.elapsed().as_secs_f64() * 1e6;
                            let b = Instant::now();
                            let r = o.build_from_logits(&buf, spec, &mut scr);
                            law_acc += b.elapsed().as_secs_f64() * 1e6;
                            r
                        },
                    );
                    let wall = t.elapsed().as_secs_f64() * 1e6;
                    if let Some(e) = err {
                        return Err(e);
                    }
                    let acc = acc.map_err(|e| Fail { code: 2, msg: e })?;
                    if acc.accepted as u32 != tw.words[0] {
                        return mismatch(format!("{}: host timing route verdict changed", tw.win.name));
                    }
                    total.push(wall);
                    dl.push(dl_acc);
                    law.push(law_acc);
                }
            }
            let d = tally().since(t0);
            variants.insert(
                "host_row_copy_plus_law".into(),
                json!({
                    "per_window": stat_json(total),
                    "row_d2h_part": stat_json(dl),
                    "host_law_part": stat_json(law),
                    "visited_rows_per_window": tws.iter().map(|t| t.visited).collect::<Vec<_>>(),
                    "per_window_avg": {
                        "hip_d2h_calls": d.dtoh as f64 / nwin, "d2h_bytes": d.w_d2h_bytes as f64 / nwin,
                        "stream_sync": d.stream_sync as f64 / nwin,
                    },
                }),
            );
        }

        // ── device variants ──
        for (vname, mode) in [
            ("device_direct_enqueue", 0u8),
            ("device_graph_replay_host_q", 1),
            ("device_graph_replay_device_cand_q", 2),
        ] {
            let mut total = Vec::new();
            let mut result_bytes = [0u8; 8];
            let mut t0 = tally();
            let mut enq_traffic = Tally::default();
            for it in 0..(warm + iters) {
                if it == warm {
                    t0 = tally();
                    total.clear();
                    enq_traffic = Tally::default();
                }
                for tw in tws.iter() {
                    let t = Instant::now();
                    if mode != 2 {
                        h2d_async(gpu, &tw.res.drf.counts, &tw.q_counts)?;
                        h2d_async(gpu, &tw.res.drf.ids, &tw.q_ids)?;
                        h2d_async(gpu, &tw.res.drf.probabilities, &tw.q_probs)?;
                    }
                    h2d_async(gpu, &tw.res.props, &tw.props_b)?;
                    h2d_async(gpu, &tw.res.tape, &tw.tape_b)?;
                    match mode {
                        0 => {
                            let b = tally();
                            let none: Vec<Option<f32>> = vec![None; k];
                            enqueue_window(gpu, &tw.res, vocab, k, policy, EOS_NONE, &none)?;
                            enq_traffic.add(tally().since(b));
                        }
                        1 => replay(gpu, tw.g_host_q.as_ref().ok_or_else(|| Fail { code: 2, msg: "graph".into() })?)?,
                        _ => replay(gpu, tw.g_cand.as_ref().ok_or_else(|| Fail { code: 2, msg: "graph".into() })?)?,
                    }
                    d2h_async(gpu, &tw.res.result, &mut result_bytes)?;
                    sync(gpu)?;
                    let wall = t.elapsed().as_secs_f64() * 1e6;
                    let words = bytes_u32(&result_bytes);
                    if words != tw.words {
                        return mismatch(format!(
                            "{vname} {}: timed result {words:?} != host {:?}; timings invalid",
                            tw.win.name, tw.words
                        ));
                    }
                    total.push(wall);
                }
            }
            let d = tally().since(t0);
            let mut m = json!({
                "per_window": stat_json(total),
                "per_window_avg": {
                    "hip_h2d_calls": d.htod as f64 / nwin, "hip_d2h_calls": d.dtoh as f64 / nwin,
                    "h2d_bytes": d.w_h2d_bytes as f64 / nwin, "d2h_bytes": d.w_d2h_bytes as f64 / nwin,
                    "stream_sync": d.stream_sync as f64 / nwin, "device_sync": d.device_sync as f64 / nwin,
                    "event_sync": d.event_sync as f64 / nwin,
                    "graph_launch": d.graph_launch as f64 / nwin,
                    "kernel_launches": d.kernel_launches as f64 / nwin,
                },
                "results_checked_bit_exact_each_window": true,
            });
            if mode == 0 {
                if enq_traffic.any_host_traffic() {
                    return mismatch(format!("timed enqueue calls issued traffic: {}", enq_traffic.json()));
                }
                m["traffic_inside_enqueue_calls"] = enq_traffic.json();
            }
            // GPU-side graph time, from events bracketing the replay.
            if mode != 0 {
                let mut ev = Vec::new();
                for _ in 0..iters.min(20) {
                    for tw in tws.iter() {
                        if mode != 2 {
                            h2d_async(gpu, &tw.res.drf.counts, &tw.q_counts)?;
                            h2d_async(gpu, &tw.res.drf.ids, &tw.q_ids)?;
                            h2d_async(gpu, &tw.res.drf.probabilities, &tw.q_probs)?;
                        }
                        h2d_async(gpu, &tw.res.props, &tw.props_b)?;
                        h2d_async(gpu, &tw.res.tape, &tw.tape_b)?;
                        hip(gpu.hip.event_record(&ev0, Some(stream_of(gpu)?)), "event_record")?;
                        let g = if mode == 1 { &tw.g_host_q } else { &tw.g_cand };
                        replay(gpu, g.as_ref().ok_or_else(|| Fail { code: 2, msg: "graph".into() })?)?;
                        hip(gpu.hip.event_record(&ev1, Some(stream_of(gpu)?)), "event_record")?;
                        sync(gpu)?;
                        let ms = hip(gpu.hip.event_elapsed_ms(&ev0, &ev1), "event_elapsed")?;
                        ev.push(ms as f64 * 1e3);
                    }
                }
                m["graph_event_bracket"] = stat_json(ev);
            }
            variants.insert(vname.into(), m);
        }
        let mut within = Map::new();
        for vn in [
            "device_direct_enqueue",
            "device_graph_replay_host_q",
            "device_graph_replay_device_cand_q",
        ] {
            let p50 = variants[vn]["per_window"]["p50_us"].as_f64().unwrap_or(f64::NAN);
            within.insert(vn.into(), json!({"p50_us": p50, "within_0_2ms_budget": p50 <= BUDGET_US}));
        }
        out.push(json!({
            "k": k, "vocab": vocab, "corpus": c.name, "policy": pols[pi].1,
            "windows": nw, "iters": iters, "warmup_iters": warm,
            "variants": Value::Object(variants),
            "s1_budget_us": BUDGET_US, "budget_check": Value::Object(within),
            "note": "raw per-window wall times; no speedup is claimed or computed",
        }));
        for tw in tws {
            if let Some(g) = tw.g_host_q {
                free_captured(gpu, g);
            }
            if let Some(g) = tw.g_cand {
                free_captured(gpu, g);
            }
            free_res(gpu, tw.res);
        }
    }
    let _ = gpu.hip.event_destroy(ev0);
    let _ = gpu.hip.event_destroy(ev1);
    Ok(json!(out))
}

// ── identity / main ──────────────────────────────────────────────────

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn identity(gpu: &Gpu, args: &Args) -> R<Value> {
    let uuid = hip(gpu.hip.device_uuid(gpu.device_id), "device_uuid")?;
    let bdf = hip(gpu.hip.device_pci_bus_id(gpu.device_id), "device_pci_bus_id")?;
    if let Some(a) = &args.expect_arch {
        if &gpu.arch != a {
            return setup(format!("arch {} != expected {a}", gpu.arch));
        }
    }
    if let Some(u) = &args.expect_uuid {
        if !uuid.eq_ignore_ascii_case(u) {
            return setup(format!("uuid {uuid} != expected {u}"));
        }
    }
    if let Some(b) = &args.expect_bdf {
        if !bdf.eq_ignore_ascii_case(b) {
            return setup(format!("bdf {bdf} != expected {b}"));
        }
    }
    let exe = std::env::current_exe().ok();
    let exe_sha = exe
        .as_ref()
        .and_then(|p| std::fs::read(p).ok())
        .map(|b| sha_hex(&b));
    let kernel_src = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels/src/spec_sampling_cpu_ar.hip");
    let kernel_sha = std::fs::read(kernel_src).ok().map(|b| sha_hex(&b));
    let mut env_map = BTreeMap::new();
    for (k, v) in std::env::vars() {
        if k.starts_with("HIPFIRE_") || k == "HIP_VISIBLE_DEVICES" || k == "ROCR_VISIBLE_DEVICES" {
            env_map.insert(k, v);
        }
    }
    Ok(json!({
        "arch": gpu.arch, "device_id": gpu.device_id, "uuid": uuid, "pci_bdf": bdf,
        "asserted": {"arch": args.expect_arch, "uuid": args.expect_uuid, "bdf": args.expect_bdf},
        "hostname": std::fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()),
        "exe": exe.map(|p| p.display().to_string()), "exe_sha256": exe_sha,
        "kernel_source": kernel_src, "kernel_source_sha256": kernel_sha,
        "env": env_map,
    }))
}

fn fill_stats(rec: &mut Map<String, Value>, st: &Stats) {
    rec.insert(
        "exactness_compared_exact".into(),
        json!({
            "rows": st.rows_compared, "id_words": st.id_words_compared,
            "probability_bits": st.prob_bits_compared,
            "candidate_picks": st.cand_rows_compared,
            "accept_result_words": st.result_words_compared,
            "rng_consumption_checks": st.rng_checks, "windows": st.windows,
        }),
    );
    rec.insert("coverage".into(), json!(st.cov));
    rec.insert(
        "enqueue_calls".into(),
        json!({
            "rows_enqueue": st.enq_rows, "candidates_enqueue": st.enq_cand,
            "accept_enqueue": st.enq_accept,
            "transfers_or_syncs_inside_enqueue_calls": st.enq_traffic.json(),
        }),
    );
}

/// Record a stage failure; the receipt is still emitted by `main`.
fn finish_fail(rec: &mut Map<String, Value>, stage: &str, e: Fail, st: &Stats) -> R<bool> {
    fill_stats(rec, st);
    rec.insert("pass".into(), json!(false));
    rec.insert("speedup_claimed".into(), json!(false));
    rec.insert("failure".into(), json!({"stage": stage, "code": e.code, "message": e.msg}));
    Ok(false)
}

fn real_main(rec: &mut Map<String, Value>) -> R<bool> {
    let args = parse_args()?;
    rec.insert(
        "args".into(),
        json!({"vocab": args.vocab, "seed": args.seed, "windows": args.windows,
               "replays": args.replays, "timing_iters": args.timing_iters,
               "timing_windows": args.timing_windows, "ks": args.ks,
               "rows_f32": args.rows_f32, "rows_vocab": args.rows_vocab,
               "skip_timing": args.skip_timing}),
    );
    let mut gpu = hip(Gpu::init(), "Gpu::init")?;
    let stream = hip(gpu.hip.stream_create(), "stream_create")?;
    gpu.active_stream = Some(stream);
    let ident = identity(&gpu, &args)?;
    eprintln!(
        "[fn_sampled_plumbing_oracle] arch={} uuid={} bdf={}",
        ident["arch"], ident["uuid"], ident["pci_bdf"]
    );
    rec.insert("identity".into(), ident);
    hip(gpu.spec_cpu_ar_precompile(), "spec_cpu_ar_precompile")?;

    let pols = policies();
    let mut rng = Rng(args.seed);
    // Synthetic corpora: main vocabulary and a small odd one.
    let mut main_rows = Vec::new();
    for _ in 0..3 {
        for kind in 0..N_KINDS {
            main_rows.push(gen_row(kind, args.vocab, &mut rng));
        }
    }
    let mut small_rows = Vec::new();
    for _ in 0..2 {
        for kind in 0..N_KINDS {
            small_rows.push(gen_row(kind, 37, &mut rng));
        }
    }
    let mut corpora = vec![
        Corpus {
            name: format!("synthetic_v{}", args.vocab),
            vocab: args.vocab,
            rows: main_rows,
        },
        Corpus {
            name: "synthetic_v37".into(),
            vocab: 37,
            rows: small_rows,
        },
    ];
    if let Some(path) = &args.rows_f32 {
        let b = std::fs::read(path).map_err(|e| Fail {
            code: 2,
            msg: format!("{path}: {e}"),
        })?;
        let row_bytes = args.rows_vocab * 4;
        if b.is_empty() || b.len() % row_bytes != 0 {
            return setup(format!("{path}: size {} not a multiple of vocab*4", b.len()));
        }
        let rows: Vec<Vec<f32>> = b
            .chunks_exact(row_bytes)
            .map(|r| {
                r.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect();
        rec.insert(
            "real_rows".into(),
            json!({"path": path, "sha256": sha_hex(&b), "rows": rows.len(), "vocab": args.rows_vocab}),
        );
        corpora.push(Corpus {
            name: "real_rows".into(),
            vocab: args.rows_vocab,
            rows,
        });
    }
    rec.insert(
        "corpora".into(),
        json!(corpora
            .iter()
            .map(|c| json!({"name": c.name, "vocab": c.vocab, "rows": c.rows.len()}))
            .collect::<Vec<_>>()),
    );

    let mut st = Stats::default();
    let t_all = tally();

    // 1. Target-row distributions over every corpus plus tiny vocabularies.
    let mut dist_corpora: Vec<(usize, Vec<Vec<f32>>)> =
        corpora.iter().map(|c| (c.vocab, c.rows.clone())).collect();
    for v in [10usize, 64] {
        let rows: Vec<Vec<f32>> = (0..N_KINDS).map(|kind| gen_row(kind, v, &mut rng)).collect();
        dist_corpora.push((v, rows));
    }
    match run_distribution(&mut gpu, &dist_corpora, &pols, &mut st) {
        Ok(v) => {
            rec.insert("distribution".into(), v);
        }
        Err(e) => return finish_fail(rec, "distribution", e, &st),
    }

    // 2. Windows: candidates + accept + RNG, direct enqueue.
    let cap_v = corpora.iter().map(|c| c.vocab).max().unwrap_or(1);
    let res = alloc_res(&mut gpu, MAX_K, cap_v)?;
    match run_candidate_invalid(&mut gpu, &res, &pols, 37) {
        Ok(v) => {
            rec.insert("candidates_invalid".into(), v);
        }
        Err(e) => return finish_fail(rec, "candidates_invalid", e, &st),
    }
    match run_windows(&mut gpu, &res, &corpora, &pols, &args, &mut st) {
        Ok(v) => {
            rec.insert("windows".into(), v);
        }
        Err(e) => return finish_fail(rec, "windows", e, &st),
    }
    free_res(&mut gpu, res);
    let missing: Vec<&&str> = REQUIRED_COVERAGE
        .iter()
        .filter(|k| st.cov.get(**k).copied().unwrap_or(0) == 0)
        .collect();
    if !missing.is_empty() {
        let e = Fail {
            code: 1,
            msg: format!("boundary classes never exercised: {missing:?}"),
        };
        return finish_fail(rec, "coverage", e, &st);
    }

    // 3. Real capture + replay.
    match run_graph_oracle(&mut gpu, &corpora, &pols, &args, &mut st) {
        Ok(v) => {
            rec.insert("capture_replay".into(), v);
        }
        Err(e) => return finish_fail(rec, "capture_replay", e, &st),
    }

    // 4. Timing (only reached when everything above was bit-exact).
    if !args.skip_timing {
        match run_timing(&mut gpu, &corpora, &pols, &args) {
            Ok(v) => {
                rec.insert("timing".into(), v);
            }
            Err(e) => return finish_fail(rec, "timing", e, &st),
        }
    }
    rec.insert("whole_run_tally".into(), tally().since(t_all).json());
    fill_stats(rec, &st);
    rec.insert("pass".into(), json!(true));
    rec.insert("speedup_claimed".into(), json!(false));
    Ok(true)
}

fn main() {
    let mut rec = Map::new();
    rec.insert("oracle".into(), json!("fn_sampled_plumbing_oracle"));
    let (pass, code) = match real_main(&mut rec) {
        Ok(true) => (true, 0),
        Ok(false) => (
            false,
            rec.get("failure")
                .and_then(|f| f["code"].as_i64())
                .unwrap_or(1) as i32,
        ),
        Err(f) => {
            rec.insert("pass".into(), json!(false));
            rec.insert(
                "failure".into(),
                json!({"stage": "setup", "code": f.code, "message": f.msg}),
            );
            (false, f.code)
        }
    };
    let text = serde_json::to_string_pretty(&Value::Object(rec.clone())).unwrap_or_default();
    let argv: Vec<String> = std::env::args().collect();
    if let Some(i) = argv.iter().position(|a| a == "--out") {
        if let Some(p) = argv.get(i + 1) {
            if let Err(e) = std::fs::write(p, &text) {
                eprintln!("receipt write {p}: {e}");
            }
        }
    }
    println!("{text}");
    if !pass {
        eprintln!(
            "[fn_sampled_plumbing_oracle] FAIL: {}",
            rec.get("failure").map(|v| v.to_string()).unwrap_or_default()
        );
        std::process::exit(code.max(1));
    }
    eprintln!("[fn_sampled_plumbing_oracle] PASS (exact; no speedup claimed)");
}
