// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C gate for the gfx942 qt=55 `FP8E4M3B128` kernels:
//! `dequant_fp8b128_to_bf16_gfx942` and `gemm_fp8b128_moe_grouped_bf16mfma`
//! (F32-x and BF16-x entries).
//!
//! 1. Decode identity: every non-NaN code x a scale sweep, BF16 bits vs the CPU
//!    teacher twin `rdna_compute::fp8_e4m3::dequant_b128_block`; with
//!    `--fixtures DIR` also vs bits written by the Python teacher
//!    (`fp8_teacher.dequant_e4m3_block`) for 256 codes x the checkpoint's scales
//!    and for one real gate_up/down expert (`mkfix.py` layout).
//! 2. Grouped GEMM vs an F64 reference on the CPU-decoded W and BF16 x (Qwen4
//!    gate_up r=1280 k=2560 x_row_div=10, down r=2560 k=640 x_row_div=1, top-10
//!    scatter over 12 experts, padding rows, a skipped tile); bound
//!    |err| <= k * 2^-24 * sum|w x| (F32 accumulation of exact products).
//! 3. F32-x and BF16-x entries byte-identical; compact tiles (nrows < 16)
//!    byte-identical to the padded layout and nothing past live rows written.
//! 4. Batch invariance: bytes(M=1) == row r of bytes(M=n), n in {2,7,8,16,17,128},
//!    every row; every row of the grouped launch == its single-row launch; and
//!    tile-position independence (a row at each of the 16 tile positions, tile
//!    index 0 and 5, == bytes(M=1)).
//!
//! Run: `cargo run --release -p rdna-compute --example test_fp8_moe_gfx942 -- [--fixtures DIR]`

use rdna_compute::fp8_e4m3::{b128_block_stride, b128_scale_count, bf16_to_f32, dequant_b128_block};
use rdna_compute::{DType, Gpu, GpuTensor};

const POISON: f32 = 12345.625;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// One qt=55 block: codes || s_inv LE || zero pad to the stride.
fn block_bytes(codes: &[u8], scales: &[u16], r: usize, k: usize) -> Vec<u8> {
    let stride = b128_block_stride(r, k).unwrap();
    let mut b = Vec::with_capacity(stride);
    b.extend_from_slice(codes);
    for s in scales {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b.resize(stride, 0);
    b
}

/// Realistic positive block scale (the checkpoint's s_inv live in ~2^-14..2^-7).
fn rand_scale(st: &mut u64) -> u16 {
    let e = 113 + lcg(st) % 8; // 2^-14 .. 2^-7
    ((e << 7) | (lcg(st) & 0x7F)) as u16
}

fn rand_codes(n: usize, st: &mut u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let b = (lcg(st) >> 8) as u8;
            if b & 0x7F == 0x7F { b ^ 1 } else { b }
        })
        .collect()
}

fn upload_bytes(gpu: &Gpu, b: &[u8]) -> GpuTensor {
    gpu.upload_raw(b, &[b.len()]).expect("upload bytes")
}

fn upload_i32(gpu: &Gpu, v: &[i32]) -> GpuTensor {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut t = upload_bytes(gpu, &b);
    t.shape = vec![v.len()];
    t.dtype = DType::F32; // 4-byte carrier so byte_size() covers the payload
    t
}

struct Fail(Vec<String>);
impl Fail {
    fn check(&mut self, ok: bool, what: String) {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.0.push(what);
        }
    }
}

/// GPU dequant of `e` blocks (`r` x `k`) -> BF16 bits.
fn gpu_dequant(gpu: &mut Gpu, blocks: &[u8], e: usize, r: usize, k: usize) -> Vec<u16> {
    let d_blk = upload_bytes(gpu, blocks);
    let d_out = upload_bytes(gpu, &vec![0xA5u8; e * r * k * 2]);
    gpu.try_gfx942()
        .expect("exact gfx942")
        .dequant_fp8b128_to_bf16(&d_blk, &d_out, e, r, k)
        .expect("dequant_fp8b128_to_bf16");
    gpu.hip.device_synchronize().expect("sync");
    let bytes = gpu.download_raw_bytes(&d_out).expect("download");
    gpu.free_tensor(d_out).ok();
    gpu.free_tensor(d_blk).ok();
    bytes.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect()
}

struct Experts {
    r: usize,
    k: usize,
    n: usize,
    payload: Vec<u8>,
    w: Vec<Vec<u16>>, // CPU-decoded BF16 per expert
}

impl Experts {
    fn synthetic(r: usize, k: usize, n: usize, seed: u64) -> Self {
        let mut st = seed;
        let mut payload = Vec::new();
        let mut w = Vec::new();
        for _ in 0..n {
            let codes = rand_codes(r * k, &mut st);
            let scales: Vec<u16> = (0..b128_scale_count(r, k).unwrap()).map(|_| rand_scale(&mut st)).collect();
            w.push(dequant_b128_block(&codes, &scales, r, k).expect("cpu decode"));
            payload.extend(block_bytes(&codes, &scales, r, k));
        }
        Experts { r, k, n, payload, w }
    }
}

struct Launch<'a> {
    ex: &'a Experts,
    d_payload: &'a GpuTensor,
    x_row_div: usize,
    x_rows: usize,
    x_f32: &'a GpuTensor,
    x_bf16: &'a GpuTensor,
}

type Tile = [i32; 4]; // {expert, row0, nrows, 0}

impl Launch<'_> {
    /// Run one grouped launch over `rows` (sorted row ids); returns Y
    /// [rows.len(), r] (POISON where the kernel wrote nothing).
    fn run(&self, gpu: &mut Gpu, tiles: &[Tile], rows: &[i32], bf16_x: bool) -> Vec<f32> {
        let stride = b128_block_stride(self.ex.r, self.ex.k).unwrap();
        let flat: Vec<i32> = tiles.iter().flatten().copied().collect();
        let d_tiles = upload_i32(gpu, &flat);
        let d_rows = upload_i32(gpu, rows);
        let n = rows.len();
        let y = gpu.upload_f32(&vec![POISON; n * self.ex.r], &[n * self.ex.r]).expect("y");
        gpu.try_gfx942()
            .expect("exact gfx942")
            .gemm_fp8b128_moe_grouped_bf16mfma(
                self.d_payload,
                stride,
                self.ex.n,
                &d_tiles,
                tiles.len(),
                &d_rows,
                if bf16_x { self.x_bf16 } else { self.x_f32 },
                self.x_row_div,
                self.x_rows,
                &y,
                n,
                self.ex.r,
                self.ex.k,
            )
            .expect("gemm_fp8b128_moe_grouped_bf16mfma");
        gpu.hip.device_synchronize().expect("sync");
        let out = gpu.download_f32(&y).expect("download y");
        for t in [y, d_rows, d_tiles] {
            gpu.free_tensor(t).ok();
        }
        out
    }

    /// One row `flat` alone: a single tile with nrows = 1.
    fn single(&self, gpu: &mut Gpu, expert: i32, flat: i32) -> Vec<f32> {
        self.run(gpu, &[[expert, 0, 1, 0]], &[flat], false)
    }
}

/// Group `slot_expert[flat] = e` into tiles, expert ascending, flat ascending.
/// `padded`: MQ4 layout (every tile 16 rows, -1 padding); otherwise compact
/// (an expert's last tile has nrows < 16, no padding rows).
fn scatter(slot_expert: &[usize], n_exp: usize, padded: bool) -> (Vec<Tile>, Vec<i32>) {
    let mut tiles = Vec::new();
    let mut rows = Vec::new();
    for e in 0..n_exp {
        let mine: Vec<i32> = (0..slot_expert.len()).filter(|&i| slot_expert[i] == e).map(|i| i as i32).collect();
        for chunk in mine.chunks(16) {
            let row0 = rows.len() as i32;
            rows.extend_from_slice(chunk);
            let nrows = if padded {
                rows.extend(std::iter::repeat_n(-1, 16 - chunk.len()));
                16
            } else {
                chunk.len() as i32
            };
            tiles.push([e as i32, row0, nrows, 0]);
        }
    }
    (tiles, rows)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn gemm_case(gpu: &mut Gpu, fail: &mut Fail, name: &str, ex: &Experts, x_row_div: usize, tokens: usize) {
    let (r, k) = (ex.r, ex.k);
    // x: BF16-exact values, including signed zeros. At least 128 rows so the
    // M-invariance sweep can reach n = 128.
    let x_rows = (if x_row_div == 1 { tokens * 10 } else { tokens }).max(128);
    let mut st = 0x5eed_0f00_u64 ^ (r as u64) << 20;
    let xb: Vec<u16> = (0..x_rows * k)
        .map(|i| match i % 97 {
            0 => 0x0000,
            1 => 0x8000,
            _ => f32_to_bf16(((lcg(&mut st) as f32 / u32::MAX as f32) - 0.5) * 4.0),
        })
        .collect();
    let xf: Vec<f32> = xb.iter().map(|&b| bf16_to_f32(b)).collect();
    let d_payload = upload_bytes(gpu, &ex.payload);
    let x_f32 = gpu.upload_f32(&xf, &[x_rows * k]).expect("x f32");
    let mut x_bf16 = upload_bytes(gpu, &xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    x_bf16.shape = vec![x_rows * k];
    x_bf16.dtype = DType::BF16;
    let l = Launch { ex, d_payload: &d_payload, x_row_div, x_rows, x_f32: &x_f32, x_bf16: &x_bf16 };

    // Top-10 distinct experts per token (deterministic, skewed).
    let mut slot_expert = Vec::with_capacity(tokens * 10);
    for t in 0..tokens {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < 10 {
            let e = (lcg(&mut st) as usize % (ex.n * 2)).min(ex.n - 1 - (t % 2));
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        slot_expert.extend(chosen);
    }
    let live = slot_expert.len();

    // Padded (MQ4) layout + one skipped tile (expert -1) over 16 extra rows that
    // must stay POISON.
    let (mut tiles, mut rows) = scatter(&slot_expert, ex.n, true);
    tiles.push([-1, rows.len() as i32, 16, 0]);
    rows.extend(std::iter::repeat_n(0, 16));
    let y = l.run(gpu, &tiles, &rows, false);
    let yb = l.run(gpu, &tiles, &rows, true);
    fail.check(bits_eq(&y, &yb), format!("{name}: F32-x and BF16-x entries byte-identical ({} values)", y.len()));

    // GPU tile-desc builder from padded-16 tile ids == the host descriptors.
    {
        let ids: Vec<i32> = tiles.iter().map(|t| t[0]).collect();
        let d_ids = upload_i32(gpu, &ids);
        let d_desc = upload_bytes(gpu, &vec![0xEEu8; ids.len() * 16]);
        gpu.try_gfx942()
            .expect("exact gfx942")
            .fp8b128_moe_tile_desc_pad16(&d_ids, &d_desc, ids.len())
            .expect("fp8b128_moe_tile_desc_pad16");
        gpu.hip.device_synchronize().expect("sync");
        let got: Vec<i32> = gpu
            .download_raw_bytes(&d_desc)
            .expect("download desc")
            .chunks_exact(4)
            .map(|p| i32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();
        let want: Vec<i32> = tiles.iter().flatten().copied().collect();
        fail.check(got == want, format!("{name}: fp8b128_moe_tile_desc_pad16 == host descriptors ({} tiles)", ids.len()));
        for t in [d_desc, d_ids] {
            gpu.free_tensor(t).ok();
        }
    }

    let bound_k = k as f64 * (2f64).powi(-24);
    let (mut worst, mut worst_abs, mut maxref) = (0f64, 0f64, 0f64);
    let (mut bad, mut pad_ok, mut skip_ok) = (0usize, true, true);
    // flat slot -> Y row bytes of the padded launch, for the layout checks below.
    let mut by_flat: Vec<Option<Vec<f32>>> = vec![None; live];
    for t in &tiles {
        for i in 0..16usize {
            let g = t[1] as usize + i;
            let yrow = &y[g * r..(g + 1) * r];
            if t[0] < 0 {
                skip_ok &= yrow.iter().all(|v| v.to_bits() == POISON.to_bits());
                continue;
            }
            let flat = rows[g];
            if flat < 0 {
                pad_ok &= yrow.iter().all(|v| v.to_bits() == 0);
                continue;
            }
            by_flat[flat as usize] = Some(yrow.to_vec());
            let xr = &xf[(flat as usize / x_row_div) * k..][..k];
            let w = &ex.w[t[0] as usize];
            for o in 0..r {
                let (mut acc, mut aa) = (0f64, 0f64);
                for (wk, xk) in w[o * k..(o + 1) * k].iter().zip(xr) {
                    let p = bf16_to_f32(*wk) as f64 * *xk as f64;
                    acc += p;
                    aa += p.abs();
                }
                let err = (yrow[o] as f64 - acc).abs();
                maxref = maxref.max(acc.abs());
                worst_abs = worst_abs.max(err);
                if aa > 0.0 {
                    worst = worst.max(err / aa);
                }
                bad += (err > bound_k * aa) as usize;
            }
        }
    }
    fail.check(
        bad == 0,
        format!(
            "{name}: grouped GEMM vs F64 ref, {live} live rows, max err/sum|wx| = {worst:.3e} (bound {bound_k:.3e}), max err/max|ref| = {:.3e}, violations {bad}",
            worst_abs / maxref.max(1e-30)
        ),
    );
    fail.check(pad_ok, format!("{name}: padding rows (-1) of live tiles written as +0"));
    fail.check(skip_ok, format!("{name}: skipped tile (expert -1) left untouched"));

    // Compact layout (nrows < 16 tails, no padding): same bytes per flat slot,
    // and nothing beyond the live rows written.
    let (ct, cr) = scatter(&slot_expert, ex.n, false);
    let mut cr_guard = cr.clone();
    cr_guard.extend(std::iter::repeat_n(0, 16)); // guard rows no tile covers
    let yc = l.run(gpu, &ct, &cr_guard, false);
    let compact_ok = cr.iter().enumerate().all(|(g, &flat)| bits_eq(&yc[g * r..(g + 1) * r], by_flat[flat as usize].as_ref().unwrap()));
    let guard_ok = yc[cr.len() * r..].iter().all(|v| v.to_bits() == POISON.to_bits());
    fail.check(compact_ok && guard_ok, format!("{name}: compact tiles (nrows<16) byte-identical to padded layout, rows past the tiles untouched"));

    // Every grouped row == its single-row launch (same expert, same X row).
    let mut mismatched = 0usize;
    for (flat, e) in slot_expert.iter().enumerate() {
        let y1 = l.single(gpu, *e as i32, flat as i32);
        mismatched += !bits_eq(&y1, by_flat[flat].as_ref().unwrap()) as usize;
    }
    fail.check(mismatched == 0, format!("{name}: every grouped row byte-identical to its M=1 launch ({live} rows, {mismatched} mismatched)"));

    // bytes(M=1) == row of bytes(M=n), every row, single expert 0, x_row_div 1 view.
    let l1 = Launch { x_row_div: 1, ..l };
    let singles: Vec<Vec<f32>> = (0..128).map(|row| l1.single(gpu, 0, row)).collect();
    for n in [2usize, 7, 8, 16, 17, 128] {
        let (t, s) = scatter(&vec![0usize; n], 1, true);
        let yn = l1.run(gpu, &t, &s, false);
        let bad_rows = (0..n).filter(|&row| !bits_eq(&yn[row * r..(row + 1) * r], &singles[row])).count();
        fail.check(bad_rows == 0, format!("{name}: bytes(M=1) == row of bytes(M={n}) for all {n} rows ({bad_rows} mismatched)"));
    }

    // Tile-position independence: X row q placed at every position p of a full
    // 16-row tile (other positions hold other rows), with the tile at index 0 and
    // at index 5 of a launch; every placement must reproduce bytes(M=1).
    let mut bad_pos = 0usize;
    let checked_rows = [0i32, 37, 127];
    for &q in &checked_rows {
        for p in 0..16usize {
            let mut tile_rows: Vec<i32> = (0..16).map(|i| ((q as usize + 1 + i * 7) % 128) as i32).collect();
            tile_rows[p] = q;
            for tile_index in [0usize, 5] {
                let mut rows_l = Vec::new();
                let mut tiles_l = Vec::new();
                for t in 0..=tile_index {
                    tiles_l.push([0, rows_l.len() as i32, 16, 0]);
                    if t == tile_index {
                        rows_l.extend_from_slice(&tile_rows);
                    } else {
                        rows_l.extend((0..16).map(|i| ((t * 16 + i) % 128) as i32));
                    }
                }
                let yy = l1.run(gpu, &tiles_l, &rows_l, false);
                let g = tile_index * 16 + p;
                bad_pos += !bits_eq(&yy[g * r..(g + 1) * r], &singles[q as usize]) as usize;
            }
        }
    }
    fail.check(
        bad_pos == 0,
        format!("{name}: tile-position independence, rows {checked_rows:?} x 16 positions x tile index {{0,5}} == bytes(M=1) ({bad_pos} mismatched)"),
    );

    for t in [x_bf16, x_f32, d_payload] {
        gpu.free_tensor(t).ok();
    }
}

fn read(dir: &str, f: &str) -> Vec<u8> {
    std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("read {dir}/{f}: {e}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fixtures = args.iter().position(|a| a == "--fixtures").map(|i| args[i + 1].clone());
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("detected arch: {}", gpu.arch);
    assert_eq!(gpu.arch, "gfx942", "this Tier C gate requires gfx942");
    let mut fail = Fail(Vec::new());

    // 1a. All 256 codes x scale sweep vs CPU teacher twin. One [128, 256] block
    // per scale pair: every 128x128 scale block holds each code 64 times.
    {
        let (r, k) = (128usize, 256usize);
        let mut st = 0xdec0_de42_u64;
        let mut sweep: Vec<u16> = vec![
            0x3F80, 0x3F81, 0x3FFF, 0x3C00, 0x3800, 0x3A83, 0x3B7F, 0x3901, 0x39FF, 0x4000, 0x4321, 0x3E55,
            0x3D2B, 0x2F80, 0x2FFF, 0xBA83, 0xBF80, 0x0D80, 0x7000,
        ];
        sweep.extend((0..45).map(|_| rand_scale(&mut st)));
        if sweep.len() % 2 == 1 {
            sweep.push(0x3B00);
        }
        let codes: Vec<u8> = (0..r * k)
            .map(|i| {
                let b = (i % 256) as u8;
                if b & 0x7F == 0x7F { 0x80 } else { b }
            })
            .collect();
        let e = sweep.len() / 2;
        let mut payload = Vec::new();
        let mut want = Vec::new();
        for p in sweep.chunks(2) {
            payload.extend(block_bytes(&codes, p, r, k));
            want.extend(dequant_b128_block(&codes, p, r, k).expect("cpu twin refused a sweep scale"));
        }
        let got = gpu_dequant(&mut gpu, &payload, e, r, k);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        fail.check(
            bad == 0,
            format!("decode: 254 finite codes x {} scales, GPU dequant BF16 bits == CPU teacher twin ({} values, {bad} mismatched)", sweep.len(), got.len()),
        );
    }

    // 1b. Python-teacher bits (fixtures from mkfix.py), if given.
    if let Some(dir) = fixtures.as_deref() {
        let scales: Vec<u16> = read(dir, "codes256.scales").chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        let refs: Vec<u16> = read(dir, "codes256.ref").chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        let (r, k) = (1usize, 256usize);
        let codes: Vec<u8> = (0..256).map(|b| if b & 0x7F == 0x7F { 0 } else { b as u8 }).collect();
        let payload: Vec<u8> = scales.iter().flat_map(|&s| block_bytes(&codes, &[s, s], r, k)).collect();
        let got = gpu_dequant(&mut gpu, &payload, scales.len(), r, k);
        let (mut cmp, mut bad) = (0usize, 0usize);
        for (i, (&g, &w)) in got.iter().zip(&refs).enumerate() {
            if (i % 256) & 0x7F == 0x7F {
                continue;
            }
            cmp += 1;
            bad += (g != w) as usize;
        }
        fail.check(bad == 0, format!("decode: 256 codes x {} scales vs Python teacher bits ({cmp} compared, {bad} mismatched)", scales.len()));
        for (name, r, k) in [("gateup", 1280usize, 2560usize), ("down", 2560, 640)] {
            let blk = read(dir, &format!("{name}.blk"));
            let want: Vec<u16> = read(dir, &format!("{name}.ref")).chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
            let got = gpu_dequant(&mut gpu, &blk, 1, r, k);
            fail.check(got == want, format!("decode: real checkpoint {name} expert [{r},{k}] GPU dequant == Python teacher bits"));
            let cpu = rdna_compute::fp8_e4m3::dequant_b128_expert_bytes(&blk, r, k).expect("cpu decode real");
            fail.check(cpu == want, format!("decode: real checkpoint {name} expert CPU twin == Python teacher bits"));
        }
        // Real expert in the grouped GEMM (single expert, x_row_div 1).
        for (name, r, k) in [("gateup", 1280usize, 2560usize), ("down", 2560, 640)] {
            let blk = read(dir, &format!("{name}.blk"));
            let w: Vec<u16> = read(dir, &format!("{name}.ref")).chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
            let ex = Experts { r, k, n: 1, payload: blk, w: vec![w] };
            // Reuse the synthetic driver with a single-expert pool: tokens=1 slot row per
            // flat index, every slot on expert 0.
            real_gemm(&mut gpu, &mut fail, name, &ex);
        }
    }

    // 2-4. Grouped GEMM Tier C at the Qwen4 shapes.
    let gate_up = Experts::synthetic(1280, 2560, 12, 0x6a7e_0001);
    gemm_case(&mut gpu, &mut fail, "gate_up r=1280 k=2560", &gate_up, 10, 37);
    drop(gate_up);
    let down = Experts::synthetic(2560, 640, 12, 0xd0e1_0002);
    gemm_case(&mut gpu, &mut fail, "down r=2560 k=640", &down, 1, 13);

    if fail.0.is_empty() {
        println!("ALL PASS test_fp8_moe_gfx942 (arch {})", gpu.arch);
    } else {
        println!("FAILED {} checks:", fail.0.len());
        for f in &fail.0 {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}

/// Real checkpoint expert: 64 BF16 x rows through one expert vs F64 on the
/// Python-teacher W.
fn real_gemm(gpu: &mut Gpu, fail: &mut Fail, name: &str, ex: &Experts) {
    let (r, k) = (ex.r, ex.k);
    let rows = 64usize;
    let mut st = 0x7ea1_u64;
    let xb: Vec<u16> = (0..rows * k).map(|_| f32_to_bf16(((lcg(&mut st) as f32 / u32::MAX as f32) - 0.5) * 4.0)).collect();
    let xf: Vec<f32> = xb.iter().map(|&b| bf16_to_f32(b)).collect();
    let d_payload = upload_bytes(gpu, &ex.payload);
    let x_f32 = gpu.upload_f32(&xf, &[rows * k]).expect("x");
    let mut x_bf16 = upload_bytes(gpu, &xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    x_bf16.shape = vec![rows * k];
    x_bf16.dtype = DType::BF16;
    let l = Launch { ex, d_payload: &d_payload, x_row_div: 1, x_rows: rows, x_f32: &x_f32, x_bf16: &x_bf16 };
    let (t, s) = scatter(&vec![0usize; rows], 1, true);
    let y = l.run(gpu, &t, &s, false);
    let bound_k = k as f64 * (2f64).powi(-24);
    let (mut worst, mut bad) = (0f64, 0usize);
    for row in 0..rows {
        for o in 0..r {
            let (mut acc, mut aa) = (0f64, 0f64);
            for c in 0..k {
                let p = bf16_to_f32(ex.w[0][o * k + c]) as f64 * xf[row * k + c] as f64;
                acc += p;
                aa += p.abs();
            }
            let err = (y[row * r + o] as f64 - acc).abs();
            if aa > 0.0 {
                worst = worst.max(err / aa);
            }
            bad += (err > bound_k * aa) as usize;
        }
    }
    fail.check(bad == 0, format!("real {name}: GEMM vs F64 on teacher W, {rows} rows, max err/sum|wx| = {worst:.3e}, violations {bad}"));
    for t in [x_bf16, x_f32, d_payload] {
        gpu.free_tensor(t).ok();
    }
}
