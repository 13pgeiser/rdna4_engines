// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Bitwise oracle + timing for the gfx1151 QT53 grouped top-10 MoE down
//! variants, against the production incumbent:
//!
//!   incumbent  `gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151`
//!              grid [ceil(M/8),  ceil(grouped_rows/16), 1]
//!   twin       `gemm_mq4g128v2_moe_grouped_top10_o8_r8_gfx1151`
//!              grid [ceil(M/8),  ceil(grouped_rows/16), 2]   (z = half tile)
//!   o16r16     `gemm_mq4g128v2_moe_grouped_top10_o16_r16_gfx1151`
//!              grid [ceil(M/16), ceil(grouped_rows/16), 1]
//!
//! All three are block = [32,1,1], take the production 11-argument signature,
//! and are compiled directly from the `.hip` sources (`include_str!`) through
//! `ensure_kernel_public`; no dispatch/production code is touched.
//!
//! ## What is verified
//! Every f32 bit of `Y_grouped` (INCLUDING padding / invalid slots, which must
//! be +0.0, and a 16-row guard band after `grouped_rows * M` that must stay
//! untouched) of each candidate equals the incumbent's. Every cell of the
//! `grouped_rows * M` region must additionally be written (no NaN-payload
//! sentinel may survive in any variant).
//!
//! ## Routing / layout (production scatter, not a CPU imitation)
//! The grouped layout (`sorted_slot_index`, `expert_tile_ids`) is produced by
//! the production launcher `Gpu::moe_scatter_fused_top10` (block_m = 16)
//! from deterministic seeded top-10-without-replacement routing. Its
//! `grouped_rows` argument is the production worst-case padded bound
//! `(slots + 15 * min(slots, E)).next_multiple_of(16)` (the same expression the
//! dispatch scratch sizing uses and the launch geometry the incumbent comment
//! quotes: 2910 slots, E = 512 -> 10592), NOT `rows * 10`; tiles beyond the
//! live padded count are `-1` sentinels. The live padded count
//! (`expert_offsets[E]`) is printed too. The scatter's within-expert slot order
//! uses LDS atomics (non-deterministic across runs) but is computed once per
//! case and shared by all variants. The scatter output is structurally
//! validated on the host (permutation, padding, tile ids) before any GEMM runs.
//! `x_src` has `rows * 10` rows with `x_row_div = 1` unless the case says
//! otherwise.
//!
//! ## Weights / X
//! Seeded random QT53 experts: per 128-K group, 68 bytes = random finite
//! signed FP16 scale, random finite signed FP16 zero, 64 random payload bytes
//! (padded K nibbles are random too, so the zero-X padding contract is
//! exercised). X is seeded random f32 in [-2, 2).
//!
//! ## Timing
//! Each variant: warmup launches, then SAMPLES individually timed launches
//! (HIP event pair around ONE launch each), launch order rotated through all
//! six permutations of the three variants between samples. Reported: median
//! and min per-launch time.
//!
//! ## Bandwidth denominator (effective GB/s, 1 GB = 1e9 B)
//!   bytes = W + X + Y
//!   W = (# experts that own >= 1 tile with >= 1 in-range slot) * M * ceil(K/128) * 68
//!       (each touched expert counted ONCE, however many tiles it has)
//!   X = (# distinct in-range x_src rows referenced) * K * 4   (each row ONCE)
//!   Y = grouped_rows * M * 4                                  (every cell is written)
//! GB/s = bytes / median_time. It is an "ideal unique traffic" rate, NOT
//! measured DRAM traffic: reuse of W across tiles/waves and X across row
//! blocks is credited as free, so a kernel re-reading more than this shows up
//! as a lower number.
//!
//! ## Sweep mode (`--sweep`)
//! Runs the same production scatter, weight and X generators over uniform
//! E512 M2560 K640 `x_row_div = 1` cases at tokens 145..=497 step 32 plus the 529
//! control, comparing ONLY incumbent vs O8R8 (O16 is neither compiled nor run).
//! Weights are built once and reused across cases. Every case still checks every
//! output bit (padding, guard band, unwritten cells) against the incumbent, with
//! WARMUP warmup launches and the median of SAMPLES event-timed launches, order
//! alternating (inc,twin / twin,inc) per sample. Each case is seeded by its token
//! count, so `--tokens=` reproduces the same routing/X for shared counts.
//! `--tokens=a,b,c` (requires `--sweep`) replaces the token list, e.g. a dense
//! pass around the crossover. The summary lists one row per case: tokens, live /
//! launched tiles, incumbent and O8R8 median us and effective GB/s, the
//! difference (inc - o8r8, positive = O8R8 faster) and the ratio. A positional
//! argument still substring-filters case names.
//!
//! GFX1151 ONLY: refuses any other arch (exit 2).
//!
//! Run:
//!   cargo run --release -p rdna-compute --example test_moe_down_twin_gfx1151 [case-name-filter]
//!   cargo run --release -p rdna-compute --example test_moe_down_twin_gfx1151 -- --sweep [--tokens=225,241,257] [filter]

use hip_bridge::KernargBlob;
use std::borrow::Cow;
use rdna_compute::{Gpu, GpuTensor};

const SRC_INCUMBENT: &str = include_str!(
    "../../../kernels/src/gemm_mq4g128v2_moe_grouped_top10_o8_r16.gfx1151.hip"
);
const SRC_TWIN: &str =
    include_str!("../../../kernels/src/gemm_mq4g128v2_moe_grouped_top10_o8_r8.gfx1151.hip");
const SRC_O16: &str =
    include_str!("../../../kernels/src/gemm_mq4g128v2_moe_grouped_top10_o16_r16.gfx1151.hip");

#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    func: &'static str,
    src: &'static str,
    rows_per_block: usize,
    grid_z: u32,
}

const VARIANTS: [Variant; 3] = [
    Variant {
        label: "incumbent o8r16",
        func: "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151",
        src: SRC_INCUMBENT,
        rows_per_block: 8,
        grid_z: 1,
    },
    Variant {
        label: "twin o8r8 z2",
        func: "gemm_mq4g128v2_moe_grouped_top10_o8_r8_gfx1151",
        src: SRC_TWIN,
        rows_per_block: 8,
        grid_z: 2,
    },
    Variant {
        label: "o16r16",
        func: "gemm_mq4g128v2_moe_grouped_top10_o16_r16_gfx1151",
        src: SRC_O16,
        rows_per_block: 16,
        grid_z: 1,
    },
];

const K_TOP: usize = 10;
const BLOCK_M: usize = 16;
const GROUP: usize = 128;
const GROUP_BYTES: usize = 68;
const SEED: u64 = 0x5154_3533_4D6F_4531; // "QT53MoE1"
const WARMUP: usize = 6;
const SAMPLES: usize = 31; // >= 20 individually timed launches per variant
const GUARD_ROWS: usize = 16;
const SENTINEL_BITS: u32 = 0x7fc0_1234;
const PERMS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
    [0, 2, 1],
    [1, 0, 2],
];

// ── RNG ────────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 11) as u128 * n as u128 >> 53) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A finite, normal, signed FP16 with biased exponent in `lo..=hi`.
    fn half_bits(&mut self, lo: u64, hi: u64) -> u16 {
        let r = self.next_u64();
        let sign = (r & 1) as u16;
        let exp = lo + (r >> 1) % (hi - lo + 1);
        let mant = ((r >> 20) & 0x3ff) as u16;
        (sign << 15) | ((exp as u16) << 10) | mant
    }
}

// ── weights ────────────────────────────────────────────────────────────────

struct Weights {
    e: usize,
    m: usize,
    k: usize,
    tensors: Vec<GpuTensor>,
    ptr_t: GpuTensor,
}

impl Weights {
    fn build(gpu: &mut Gpu, e: usize, m: usize, k: usize, seed: u64) -> Weights {
        let groups = k.div_ceil(GROUP);
        let mut bytes = vec![0u8; m * groups * GROUP_BYTES];
        let mut tensors = Vec::with_capacity(e);
        let mut ptrs = Vec::with_capacity(e * 8);
        for expert in 0..e {
            let mut rng = Rng(seed ^ (expert as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            for chunk in bytes.chunks_exact_mut(GROUP_BYTES) {
                let scale = rng.half_bits(5, 11);
                let zero = rng.half_bits(8, 13);
                chunk[0..2].copy_from_slice(&scale.to_le_bytes());
                chunk[2..4].copy_from_slice(&zero.to_le_bytes());
                for w in chunk[4..].chunks_exact_mut(8) {
                    w.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
            }
            let t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("expert upload");
            ptrs.extend_from_slice(&(t.buf.as_ptr() as u64).to_le_bytes());
            tensors.push(t);
        }
        let ptr_t = gpu.upload_raw(&ptrs, &[ptrs.len()]).expect("ptr table upload");
        Weights { e, m, k, tensors, ptr_t }
    }
    fn free(self, gpu: &mut Gpu) {
        for t in self.tensors {
            gpu.free_tensor(t).expect("free expert");
        }
        gpu.free_tensor(self.ptr_t).expect("free ptr table");
    }
}

// ── routing ────────────────────────────────────────────────────────────────

/// Uniform top-10 WITHOUT replacement per token (partial Fisher-Yates).
fn route_uniform(rng: &mut Rng, e: usize, tokens: usize) -> Vec<i32> {
    let mut perm: Vec<i32> = (0..e as i32).collect();
    let mut out = Vec::with_capacity(tokens * K_TOP);
    for _ in 0..tokens {
        for j in 0..K_TOP {
            let r = j + rng.below(e - j);
            perm.swap(j, r);
            out.push(perm[j]);
        }
    }
    out
}

/// Skewed top-10 WITHOUT replacement: Zipf(1.1) weights over a random expert
/// ranking, sequential weighted draws with removal.
fn route_skewed(rng: &mut Rng, e: usize, tokens: usize) -> Vec<i32> {
    let mut order: Vec<i32> = (0..e as i32).collect();
    for i in (1..e).rev() {
        let j = rng.below(i + 1);
        order.swap(i, j);
    }
    let base: Vec<f64> = (0..e).map(|r| 1.0 / ((r + 1) as f64).powf(1.1)).collect();
    let mut out = Vec::with_capacity(tokens * K_TOP);
    for _ in 0..tokens {
        let mut w = base.clone();
        for _ in 0..K_TOP {
            let total: f64 = w.iter().sum();
            let mut u = rng.unit() * total;
            let mut pick = e - 1;
            for (r, &x) in w.iter().enumerate() {
                if x > 0.0 && u < x {
                    pick = r;
                    break;
                }
                u -= x;
            }
            if w[pick] == 0.0 {
                pick = w.iter().position(|&x| x > 0.0).expect("weights exhausted");
            }
            w[pick] = 0.0;
            out.push(order[pick]);
        }
    }
    out
}

// ── cases ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Routing {
    Uniform,
    Skewed,
}

#[derive(Clone, Copy, PartialEq)]
enum XMode {
    /// x_src_rows = rows*10, x_row_div = 1.
    PerSlot,
    /// x_src_rows = ceil(rows*10/2), x_row_div = 2.
    Div2,
    /// x_row_div = 1 but only half the rows exist: upper flats are out of range.
    Truncated,
}

struct Spec {
    name: Cow<'static, str>,
    e: usize,
    m: usize,
    k: usize,
    tokens: usize,
    routing: Routing,
    xmode: XMode,
    /// Pass a `grouped_rows` that is not a multiple of 16 (partial final tile).
    partial_grouped: bool,
    /// Inject invalid expert ids into the router output (scatter must skip them).
    invalid_topk: bool,
    /// Corrupt the scatter output afterwards: sentinel/out-of-range tiles and
    /// slots (negative, beyond x_src_rows).
    mutate: bool,
}

const fn spec(name: &'static str, e: usize, m: usize, k: usize, tokens: usize) -> Spec {
    Spec {
        name: Cow::Borrowed(name),
        e,
        m,
        k,
        tokens,
        routing: Routing::Uniform,
        xmode: XMode::PerSlot,
        partial_grouped: false,
        invalid_topk: false,
        mutate: false,
    }
}

fn specs() -> Vec<Spec> {
    let mut v = vec![
        spec("u145_E512_M2560_K640", 512, 2560, 640, 145),
        spec("u273_E512_M2560_K640", 512, 2560, 640, 273),
        spec("u401_E512_M2560_K640", 512, 2560, 640, 401),
        spec("u529_E512_M2560_K640", 512, 2560, 640, 529),
        Spec { routing: Routing::Skewed, ..spec("skew401_E512_M2560_K640", 512, 2560, 640, 401) },
    ];
    // Tiny tail / bounds cases (E = 16 so top-10 without replacement exists).
    v.push(spec("tail_K639_M40", 16, 40, 639, 13));
    v.push(spec("tail_K641_M40", 16, 40, 641, 13));
    v.push(spec("tail_M17_K640", 16, 17, 640, 13));
    v.push(spec("tail_M17_K639", 16, 17, 639, 13));
    v.push(spec("tail_M17_K641", 16, 17, 641, 13));
    v.push(Spec { xmode: XMode::Div2, ..spec("div2_M40_K640", 16, 40, 640, 13) });
    v.push(Spec { xmode: XMode::Div2, ..spec("div2_M17_K641", 16, 17, 641, 13) });
    v.push(Spec { xmode: XMode::Truncated, ..spec("xbound_M40_K640", 16, 40, 640, 13) });
    v.push(Spec { partial_grouped: true, ..spec("partial_grouped_M40_K640", 16, 40, 640, 13) });
    v.push(Spec {
        invalid_topk: true,
        mutate: true,
        ..spec("invalid_sentinel_M40_K639", 16, 40, 639, 13)
    });
    v.push(Spec {
        invalid_topk: true,
        mutate: true,
        partial_grouped: true,
        ..spec("invalid_sentinel_partial_M17_K640", 16, 17, 640, 13)
    });
    v.push(Spec {
        invalid_topk: true,
        mutate: true,
        xmode: XMode::Div2,
        ..spec("invalid_sentinel_div2_M2560_K640", 16, 2560, 640, 13)
    });
    v
}

// ── host helpers ───────────────────────────────────────────────────────────

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn zeros_raw(gpu: &Gpu, bytes: usize) -> GpuTensor {
    gpu.upload_raw(&vec![0u8; bytes], &[bytes]).expect("zero upload")
}

/// Structural check of the production scatter against the router output.
fn validate_scatter(
    topk: &[i32],
    e: usize,
    offsets: &[i32],
    sorted: &[i32],
    tiles: &[i32],
    counts_dev: &[i32],
) -> usize {
    let mut flats: Vec<Vec<i32>> = vec![Vec::new(); e];
    for (flat, &x) in topk.iter().enumerate() {
        if x >= 0 && (x as usize) < e {
            flats[x as usize].push(flat as i32);
        }
    }
    let mut off = 0usize;
    for ex in 0..e {
        let padded = flats[ex].len().next_multiple_of(BLOCK_M);
        assert_eq!(offsets[ex] as usize, off, "scatter offset of expert {ex}");
        assert_eq!(counts_dev[ex] as usize, padded, "scatter padded count of expert {ex}");
        let region = &sorted[off..off + padded];
        let mut got: Vec<i32> = region.iter().copied().filter(|&s| s >= 0).collect();
        assert!(region.iter().all(|&s| s >= -1), "bad slot value");
        got.sort_unstable();
        assert_eq!(got, flats[ex], "scatter slots of expert {ex}");
        for t in off / BLOCK_M..(off + padded) / BLOCK_M {
            assert_eq!(tiles[t], ex as i32, "tile {t} expert");
        }
        off += padded;
    }
    assert_eq!(offsets[e] as usize, off, "scatter total");
    assert!(sorted[off..].iter().all(|&s| s == -1), "slot tail must be -1");
    assert!(tiles[off / BLOCK_M..].iter().all(|&t| t == -1), "tile tail must be -1");
    off
}

struct Traffic {
    touched_experts: usize,
    unique_x_rows: usize,
    live_slots: usize,
    w_bytes: u64,
    x_bytes: u64,
    y_bytes: u64,
}

/// Mirrors the kernel's own validity predicates.
#[allow(clippy::too_many_arguments)]
fn traffic(
    sorted: &[i32],
    tiles: &[i32],
    e: usize,
    m: usize,
    k: usize,
    div: usize,
    gr: usize,
    x_rows: usize,
) -> Traffic {
    let mut touched = vec![false; e];
    let mut seen_x = vec![false; x_rows];
    let mut live_slots = 0usize;
    for t in 0..gr.div_ceil(16) {
        let ex = tiles[t];
        if ex < 0 || ex as usize >= e {
            continue;
        }
        for slot in 0..16 {
            let gs = t * 16 + slot;
            if gs >= gr {
                continue;
            }
            let flat = sorted[gs];
            if flat < 0 {
                continue;
            }
            let xr = flat as usize / div;
            if xr < x_rows {
                touched[ex as usize] = true;
                seen_x[xr] = true;
                live_slots += 1;
            }
        }
    }
    let touched_experts = touched.iter().filter(|&&b| b).count();
    let unique_x_rows = seen_x.iter().filter(|&&b| b).count();
    let groups = k.div_ceil(GROUP);
    Traffic {
        touched_experts,
        unique_x_rows,
        live_slots,
        w_bytes: (touched_experts * m * groups * GROUP_BYTES) as u64,
        x_bytes: (unique_x_rows * k * 4) as u64,
        y_bytes: (gr * m * 4) as u64,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid byte patterns; length in bytes is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ── one case ───────────────────────────────────────────────────────────────

struct CaseOutcome {
    name: String,
    tokens: usize,
    live_tiles: usize,
    launched_tiles: usize,
    ok: bool,
    median_us: [f64; 3],
    gbps: [f64; 3],
}

/// `active` lists the variant indices to run; `active[0]` MUST be 0 (the incumbent
/// reference every other variant is compared against bit-for-bit).
fn run_case(
    gpu: &mut Gpu,
    spec: &Spec,
    w: &Weights,
    case_idx: usize,
    active: &[usize],
) -> CaseOutcome {
    assert_eq!(active.first(), Some(&0), "incumbent must be the reference variant");
    let (e, m, k) = (spec.e, spec.m, spec.k);
    let mut rng = Rng(SEED ^ ((case_idx as u64 + 1) << 32) ^ 0xA5A5);
    let slots = spec.tokens * K_TOP;
    let prod_gr = (slots + 15 * slots.min(e)).next_multiple_of(BLOCK_M);

    // Routing.
    let mut topk = match spec.routing {
        Routing::Uniform => route_uniform(&mut rng, e, spec.tokens),
        Routing::Skewed => route_skewed(&mut rng, e, spec.tokens),
    };
    if spec.invalid_topk {
        for (i, bad) in [(3usize, -1i32), (17, e as i32), (40, -9), (41, e as i32 + 100)] {
            if i < topk.len() {
                topk[i] = bad;
            }
        }
    }

    // Production scatter.
    let topk_t = gpu.upload_raw(&i32_bytes(&topk), &[slots * 4]).expect("topk");
    let counts_t = zeros_raw(gpu, e * 4);
    let offsets_t = zeros_raw(gpu, (e + 1) * 4);
    let sorted_t = zeros_raw(gpu, prod_gr * 4);
    let tiles_t = zeros_raw(gpu, prod_gr / BLOCK_M * 4);
    let inverse_t = zeros_raw(gpu, slots * 4);
    gpu.moe_scatter_fused_top10(
        &topk_t, &counts_t, &offsets_t, &sorted_t, &tiles_t, &inverse_t, slots, e, prod_gr,
        BLOCK_M,
    )
    .expect("production scatter");
    gpu.hip.device_synchronize().expect("scatter sync");
    let offsets = i32s(&gpu.download_raw_bytes(&offsets_t).expect("offsets"));
    let counts = i32s(&gpu.download_raw_bytes(&counts_t).expect("counts"));
    let mut sorted = i32s(&gpu.download_raw_bytes(&sorted_t).expect("sorted"));
    let mut tiles = i32s(&gpu.download_raw_bytes(&tiles_t).expect("tiles"));
    let live_padded = validate_scatter(&topk, e, &offsets, &sorted, &tiles, &counts);

    // Optional corruption: sentinel / out-of-range tiles and bad slots.
    let x_rows = match spec.xmode {
        XMode::PerSlot => slots,
        XMode::Div2 => slots.div_ceil(2),
        XMode::Truncated => slots / 2,
    };
    let x_div = if spec.xmode == XMode::Div2 { 2usize } else { 1 };
    if spec.mutate {
        let live_tiles = live_padded / BLOCK_M;
        assert!(live_tiles >= 6, "mutation case needs >= 6 live tiles");
        tiles[1] = -1; // sentinel tile that still has live slots
        tiles[2] = e as i32; // out-of-range expert
        tiles[3] = -7;
        for (t, slot, val) in [
            (4usize, 3usize, -1i32),
            (4, 5, i32::MAX),
            (5, 0, -5),
            (5, 9, (x_rows * x_div) as i32), // first flat whose x row is out of range
            (0, 2, -1),
        ] {
            sorted[t * 16 + slot] = val;
        }
        gpu.hip.memcpy_htod(&sorted_t.buf, &i32_bytes(&sorted)).expect("mutate sorted");
        gpu.hip.memcpy_htod(&tiles_t.buf, &i32_bytes(&tiles)).expect("mutate tiles");
    }
    let gr = if spec.partial_grouped && live_padded > 2 * BLOCK_M {
        live_padded - 5
    } else {
        prod_gr
    };
    assert!(gr > 0 && gr <= prod_gr);

    // X.
    let xs: Vec<f32> =
        (0..x_rows * k).map(|_| (rng.unit() * 4.0 - 2.0) as f32).collect();
    let x_t = gpu.upload_f32(&xs, &[xs.len()]).expect("x upload");

    // Y with a guard band; every variant starts from the NaN-payload sentinel.
    let y_floats = (gr + GUARD_ROWS) * m;
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); y_floats];
    let y_t = gpu.upload_f32(&sentinel, &[y_floats]).expect("y upload");

    let tr = traffic(&sorted, &tiles, e, m, k, x_div, gr, x_rows);
    let total_bytes = tr.w_bytes + tr.x_bytes + tr.y_bytes;
    let tiles_launched = gr.div_ceil(16);
    println!(
        "\n=== {} ===\n  shape E={e} M={m} K={k} tokens={} slots={slots} routing={} \
         x_src_rows={x_rows} x_row_div={x_div}\n  grouped_rows(arg)={gr} (production bound {prod_gr}, \
         live padded {live_padded}, live tiles {}, launched tiles {tiles_launched})\n  \
         touched experts={} live slots={} unique X rows={} | bytes: W={:.2} MB X={:.2} MB Y={:.2} MB \
         total={:.2} MB",
        spec.name,
        spec.tokens,
        if spec.routing == Routing::Skewed { "skewed-zipf1.1" } else { "uniform" },
        live_padded / BLOCK_M,
        tr.touched_experts,
        tr.live_slots,
        tr.unique_x_rows,
        tr.w_bytes as f64 / 1e6,
        tr.x_bytes as f64 / 1e6,
        tr.y_bytes as f64 / 1e6,
        total_bytes as f64 / 1e6,
    );

    // Kernarg blobs (production argument order and types).
    let blob = |_: &Variant| -> Vec<u8> {
        let mut b = KernargBlob::new();
        b.push_u64(w.ptr_t.buf.as_ptr() as u64);
        b.push_u64(tiles_t.buf.as_ptr() as u64);
        b.push_u64(sorted_t.buf.as_ptr() as u64);
        b.push_u64(x_t.buf.as_ptr() as u64);
        b.push_u64(y_t.buf.as_ptr() as u64);
        b.push_i32(m as i32);
        b.push_i32(k as i32);
        b.push_i32(x_div as i32);
        b.push_i32(gr as i32);
        b.push_i32(x_rows as i32);
        b.push_i32(e as i32);
        b.into_vec()
    };
    let mut blobs: Vec<Vec<u8>> = VARIANTS.iter().map(blob).collect();
    let grid = |v: &Variant| [m.div_ceil(v.rows_per_block) as u32, tiles_launched as u32, v.grid_z];
    let block = [32u32, 1, 1];

    // Correctness: one fresh-sentinel run per variant.
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for &i in active {
        let v = &VARIANTS[i];
        gpu.hip.memcpy_htod(&y_t.buf, f32_bytes(&sentinel)).expect("reset y");
        gpu.launch_kernel_blob(v.func, grid(v), block, 0, &mut blobs[i]).expect("launch");
        gpu.hip.device_synchronize().expect("sync");
        outputs.push(gpu.download_f32(&y_t).expect("download y"));
    }
    let mut ok = true;
    let main_cells = gr * m;
    let mut verdicts: Vec<String> = Vec::new();
    for (pos, &vi) in active.iter().enumerate() {
        let v = &VARIANTS[vi];
        let i = pos;
        let out = &outputs[i];
        let unwritten = out[..main_cells].iter().filter(|x| x.to_bits() == SENTINEL_BITS).count();
        let guard_bad =
            out[main_cells..].iter().filter(|x| x.to_bits() != SENTINEL_BITS).count();
        let nonfinite = out[..main_cells].iter().filter(|x| !x.is_finite()).count();
        let nonzero = out[..main_cells].iter().filter(|x| x.to_bits() << 1 != 0).count();
        let mut line = format!(
            "  [{}] unwritten={unwritten} guard_clobbered={guard_bad} nonfinite={nonfinite} \
             nonzero_cells={nonzero}",
            v.label
        );
        let mut v_ok = unwritten == 0 && guard_bad == 0 && nonfinite == 0;
        if tr.live_slots > 0 && nonzero == 0 {
            v_ok = false;
            line.push_str(" ALL-ZERO");
        }
        if i == 0 {
            line.push_str(if v_ok { " (reference sane)" } else { " REFERENCE-BAD" });
        } else {
            let diff: Vec<usize> = (0..out.len())
                .filter(|&j| out[j].to_bits() != outputs[0][j].to_bits())
                .collect();
            if diff.is_empty() {
                line.push_str(" | BITWISE-IDENTICAL to incumbent");
            } else {
                v_ok = false;
                let j = diff[0];
                line.push_str(&format!(
                    " | MISMATCH {} cells; first at grouped_slot {} row {}: {:#010x} vs incumbent {:#010x}",
                    diff.len(),
                    j / m,
                    j % m,
                    out[j].to_bits(),
                    outputs[0][j].to_bits()
                ));
            }
        }
        ok &= v_ok;
        verdicts.push(line);
    }
    for l in &verdicts {
        println!("{l}");
    }
    drop(outputs);

    // Timing: individually event-timed launches, rotating order.
    let ev0 = gpu.hip.event_create().expect("event");
    let ev1 = gpu.hip.event_create().expect("event");
    let time_one = |gpu: &mut Gpu, i: usize, blobs: &mut Vec<Vec<u8>>| -> f64 {
        gpu.hip.event_record(&ev0, None).expect("rec");
        gpu.launch_kernel_blob(VARIANTS[i].func, grid(&VARIANTS[i]), block, 0, &mut blobs[i])
            .expect("launch");
        gpu.hip.event_record(&ev1, None).expect("rec");
        gpu.hip.event_synchronize(&ev1).expect("evsync");
        gpu.hip.event_elapsed_ms(&ev0, &ev1).expect("elapsed") as f64 * 1000.0
    };
    let perms: Vec<Vec<usize>> = match active {
        [a, b] => vec![vec![*a, *b], vec![*b, *a]],
        _ => PERMS.iter().map(|p| p.to_vec()).collect(),
    };
    for w_i in 0..WARMUP {
        for &i in &perms[w_i % perms.len()] {
            time_one(gpu, i, &mut blobs);
        }
    }
    let mut samples: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for s in 0..SAMPLES {
        for &i in &perms[s % perms.len()] {
            let t = time_one(gpu, i, &mut blobs);
            samples[i].push(t);
        }
    }
    gpu.hip.event_destroy(ev0).expect("destroy");
    gpu.hip.event_destroy(ev1).expect("destroy");

    let mut median_us = [0f64; 3];
    let mut gbps = [0f64; 3];
    println!(
        "  timing: {SAMPLES} individual launches/variant after {WARMUP} warmup, order rotated over {} perms",
        perms.len()
    );
    for &i in active {
        let v = &VARIANTS[i];
        let min = samples[i].iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut samples[i]);
        median_us[i] = med;
        gbps[i] = total_bytes as f64 / 1e9 / (med * 1e-6);
        println!(
            "  {:<16} median {:>9.2} us  min {:>9.2} us  eff {:>7.1} GB/s  ({:.3}x incumbent median)",
            v.label,
            med,
            min,
            gbps[i],
            median_us[0] / med
        );
    }
    println!("  VERDICT {}: {}", spec.name, if ok { "PASS (bitwise)" } else { "FAIL" });

    for t in [topk_t, counts_t, offsets_t, sorted_t, tiles_t, inverse_t, x_t, y_t] {
        gpu.free_tensor(t).expect("free");
    }
    CaseOutcome {
        name: spec.name.to_string(),
        tokens: spec.tokens,
        live_tiles: live_padded / BLOCK_M,
        launched_tiles: tiles_launched,
        ok,
        median_us,
        gbps,
    }
}

/// Production-crossover sweep token counts (E512 M2560 K640, uniform, xdiv 1):
/// 145..=497 step 32, plus the 529 control.
const SWEEP_TOKENS: [usize; 13] =
    [145, 177, 209, 241, 273, 305, 337, 369, 401, 433, 465, 497, 529];
/// Sweep compares only the incumbent (variant 0) and the O8R8 twin (variant 1).
const SWEEP_ACTIVE: [usize; 2] = [0, 1];

fn sweep_specs(tokens: &[usize]) -> Vec<Spec> {
    tokens
        .iter()
        .map(|&t| Spec {
            name: Cow::Owned(format!("sweep_t{t}_E512_M2560_K640")),
            ..spec("", 512, 2560, 640, t)
        })
        .collect()
}

fn main() {
    let mut sweep = false;
    let mut tokens_override: Option<Vec<usize>> = None;
    let mut filter: Option<String> = None;
    for a in std::env::args().skip(1) {
        if a == "--sweep" {
            sweep = true;
        } else if let Some(list) = a.strip_prefix("--tokens=") {
            let v: Vec<usize> = list
                .split(',')
                .map(|s| s.trim().parse().unwrap_or_else(|_| panic!("bad --tokens entry {s:?}")))
                .collect();
            assert!(!v.is_empty() && v.iter().all(|&t| t > 0), "--tokens needs positive counts");
            tokens_override = Some(v);
        } else if a.starts_with("--") {
            eprintln!("unknown flag {a} (supported: --sweep, --tokens=a,b,c with --sweep)");
            std::process::exit(2);
        } else {
            filter = Some(a);
        }
    }
    if tokens_override.is_some() && !sweep {
        eprintln!("--tokens requires --sweep");
        std::process::exit(2);
    }

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch = {}", gpu.arch);
    if !(gpu.arch_caps.is_gfx1151() && gpu.arch.starts_with("gfx1151")) {
        eprintln!(
            "REFUSING: test_moe_down_twin_gfx1151 is a gfx1151-only oracle (found {}).",
            gpu.arch
        );
        std::process::exit(2);
    }
    let active: Vec<usize> =
        if sweep { SWEEP_ACTIVE.to_vec() } else { (0..VARIANTS.len()).collect() };
    for &i in &active {
        let v = &VARIANTS[i];
        gpu.ensure_kernel_public(v.func, v.src, v.func)
            .unwrap_or_else(|e| panic!("compile {}: {e:?}", v.func));
    }

    println!(
        "bytes denominator: W (unique touched experts, once each) + X (unique referenced rows, \
         once each) + Y (grouped_rows*M*4); GB/s = bytes/1e9/median_s"
    );

    // Sweep cases are seeded by token count (not list position), so a dense
    // follow-up run reproduces the same routing/X for any shared token count.
    let all_specs: Vec<(u64, Spec)> = if sweep {
        let toks = tokens_override.clone().unwrap_or_else(|| SWEEP_TOKENS.to_vec());
        sweep_specs(&toks).into_iter().map(|s| (s.tokens as u64, s)).collect()
    } else {
        specs().into_iter().enumerate().map(|(i, s)| (i as u64, s)).collect()
    };

    let mut cache: Option<Weights> = None;
    let mut outcomes = Vec::new();
    for (idx, sp) in all_specs.iter() {
        if filter.as_deref().is_some_and(|f| !sp.name.contains(f)) {
            continue;
        }
        let rebuild = cache.as_ref().map_or(true, |w| (w.e, w.m, w.k) != (sp.e, sp.m, sp.k));
        if rebuild {
            if let Some(old) = cache.take() {
                old.free(&mut gpu);
            }
            println!(
                "\n[building {} experts of {}x{} (QT53, {:.1} MB each)]",
                sp.e,
                sp.m,
                sp.k,
                (sp.m * sp.k.div_ceil(GROUP) * GROUP_BYTES) as f64 / 1e6
            );
            cache = Some(Weights::build(
                &mut gpu,
                sp.e,
                sp.m,
                sp.k,
                SEED ^ ((sp.e * 1_000_003 + sp.m * 131 + sp.k) as u64),
            ));
        }
        let w = cache.as_ref().unwrap();
        outcomes.push(run_case(&mut gpu, sp, w, *idx as usize, &active));
    }
    if let Some(old) = cache.take() {
        old.free(&mut gpu);
    }

    let mut all_ok = !outcomes.is_empty();
    if sweep {
        println!(
            "\n================ SWEEP SUMMARY (incumbent o8r16 vs twin o8r8 z2; median us | eff GB/s) ================"
        );
        println!(
            "{:>6} {:<26} {:>7} {:>6} {:>6} | {:>9} {:>7} | {:>9} {:>7} | {:>9} {:>7}",
            "tokens", "case", "verdict", "live_t", "launch", "inc us", "GB/s", "o8r8 us", "GB/s",
            "inc-o8r8", "inc/o8r8"
        );
        for o in &outcomes {
            all_ok &= o.ok;
            println!(
                "{:>6} {:<26} {:>7} {:>6} {:>6} | {:>9.2} {:>7.1} | {:>9.2} {:>7.1} | {:>+9.2} {:>7.3}x",
                o.tokens,
                o.name,
                if o.ok { "PASS" } else { "FAIL" },
                o.live_tiles,
                o.launched_tiles,
                o.median_us[0],
                o.gbps[0],
                o.median_us[1],
                o.gbps[1],
                o.median_us[0] - o.median_us[1],
                o.median_us[0] / o.median_us[1],
            );
        }
    } else {
        println!(
            "\n================ SUMMARY (median us | eff GB/s; ratio vs incumbent) ================"
        );
        println!(
            "{:<36} {:>8} | {:>9} {:>7} | {:>9} {:>7} {:>6} | {:>9} {:>7} {:>6}",
            "case", "verdict", "inc us", "GB/s", "twin us", "GB/s", "x", "o16 us", "GB/s", "x"
        );
        for o in &outcomes {
            all_ok &= o.ok;
            println!(
                "{:<36} {:>8} | {:>9.2} {:>7.1} | {:>9.2} {:>7.1} {:>6.3} | {:>9.2} {:>7.1} {:>6.3}",
                o.name,
                if o.ok { "PASS" } else { "FAIL" },
                o.median_us[0],
                o.gbps[0],
                o.median_us[1],
                o.gbps[1],
                o.median_us[0] / o.median_us[1],
                o.median_us[2],
                o.gbps[2],
                o.median_us[0] / o.median_us[2],
            );
        }
    }
    println!("OVERALL: {}", if all_ok { "PASS (all bitwise identical)" } else { "FAIL" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
