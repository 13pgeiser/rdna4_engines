// SPDX-License-Identifier: Apache-2.0
//! K2 G1 oracle for the MQ3-I4 NT1 gate/up grouped IU4 GEMM (raw PM module launch; never runtime dispatch).
//! Build: cargo build -p hipfire-runtime --release --features lab --example qwen4_moe_sym_w3_g1_oracle
//! Run (on a locked card):
//!   qwen4_moe_sym_w3_g1_oracle --w3 kernels/qwen4_moe_iu4_sym_w3_pm_gfx1201.hxaco \
//!       --mq4 kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco [--seed N] [--expect-bdf 0000:..]
//!
//! Per scenario (rows 1/2/7/8/15/16/17, pad slots, sentinel tile, mixed experts, x_row_div 1 and 10):
//!  (1) the `_acc` entry's F32 accumulators == `mq_i4_reference::iu4_gemm_f32_parts`, bit for bit
//!      (pad slots +0), every output including unwritten-poison checks;
//!  (2) the shipped entry's BF16 H bytes == the unchanged MQ4 NT1 entry run on the bit-equivalent QT44
//!      tensors (signed nibble `2c-7` stored as `2c+1`, same f16 scale): the certified MQ4 epilogue is the
//!      byte anchor for identical F32 inputs.
//! Experts sit in one slab buffer at 256-byte offsets with 0xFF sentinel bytes between and after them; the
//! `sentinel_*` scenarios put 0xFF/0x00 low bytes into the scale plane's first scale (the bytes a last-row
//! chunk's unaligned dword read spills into).
use hip_bridge::{Function, HipRuntime};
use hipfire_runtime::{mq_i4::{self, MqI4Codec as Codec, MqI4Profile as Profile, MqI4Rotation as Rotation}, mq_i4_reference::{self, A4Block}};
use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String { e.to_string() }

const M: usize = 1280;
const K: usize = 2560;
const HALF: usize = M / 2;
const GROUPS: usize = K / 128;
const BLK: usize = 72;
const SLAB_ALIGN: usize = 256;

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 17; self.0 ^= self.0 << 5; self.0 }
    fn below(&mut self, n: u32) -> u32 { self.next() % n }
}

struct Expert { codes: Vec<u8>, scales: Vec<u16>, w3: Vec<u8>, qt44: Vec<u8> }

fn profile() -> Profile { Profile { codec: Codec::W3, rotation: Rotation::R256 } }

/// QT44 (136 B per K256: two `[f16 sc, f16 zp]` headers, then two 64-byte nibble halves, low nibble first)
/// holding the same signed values: stored nibble `2c-7+8 = 2c+1`.
fn qt44(codes: &[u8], scales: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(M * (K / 256) * 136);
    for r in 0..M {
        for g2 in 0..K / 256 {
            let mut block = [0u8; 136];
            for h in 0..2 {
                let sc = scales[r * GROUPS + 2 * g2 + h];
                let zp: u16 = if (sc >> 10) & 0x1f != 0 && ((sc >> 10) & 0x1f) <= 27 { (sc + (3 << 10)) | 0x8000 } else { 0 };
                block[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                block[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
                for j in 0..128 {
                    let c = codes[r * K + (2 * g2 + h) * 128 + j];
                    let u = 2 * c + 1;
                    block[8 + 64 * h + j / 2] |= u << (4 * (j % 2));
                }
            }
            out.extend_from_slice(&block);
        }
    }
    out
}

#[derive(Clone, Copy)]
enum Kind { Random, Extreme, EveryCode, Sentinel(u16) }
fn expert(kind: Kind, seed: u32) -> Result<Expert> {
    let mut rng = Rng(seed | 1);
    let mid = profile().codec.midpoint();
    let mut codes: Vec<u8> = (0..M * K).map(|_| rng.below(8) as u8).collect();
    let pool = [0x3000u16, 0x3800, 0x3c00, 0x4000, 0x2c00, 0x3555];
    let mut scales: Vec<u16> = (0..M * GROUPS).map(|_| pool[rng.below(pool.len() as u32) as usize]).collect();
    match kind {
        Kind::Random => {}
        Kind::Extreme => {
            for (i, c) in codes.iter_mut().enumerate() { *c = if (i / 3) % 2 == 0 { 0 } else { 7 }; }
            for (i, s) in scales.iter_mut().enumerate() { *s = [0x7bff, 0x0001, 0x0400, 0x3c00][i % 4]; }
            // zero scale groups hold only the midpoint code
            for r in (0..M).step_by(97) { for g in 0..GROUPS { if g % 5 == 0 { scales[r * GROUPS + g] = 0; for j in 0..128 { codes[r * K + g * 128 + j] = mid; } } } }
        }
        Kind::EveryCode => { for (i, c) in codes.iter_mut().enumerate() { *c = ((i + i / K) % 8) as u8; } }
        Kind::Sentinel(first) => { scales[0] = first; }
    }
    let w3 = mq_i4::encode_tensor(profile(), M, K, &codes, &scales).map_err(err)?;
    let qt = qt44(&codes, &scales);
    Ok(Expert { codes, scales, w3, qt44: qt })
}

/// 72-byte A4 blocks for `rows` rows, epoch-major (`[g][row]`); also the token-major copy for the twin.
fn make_x(rows: usize, seed: u32) -> (Vec<u8>, Vec<A4Block>) {
    let mut rng = Rng(seed | 1);
    let ds = [1.0f32, 0.5, 1.0 / 64.0, 1.0e-3, 3.0, 0.0, 65536.0, 1.0e-20];
    let mut tok: Vec<A4Block> = Vec::new();
    for r in 0..rows {
        for g in 0..GROUPS {
            let mut qs = [0u8; 64];
            let mut s = 0i32;
            for j in 0..128 {
                let v: i8 = match (r + g) % 9 { 0 => -8, 1 => 7, 2 => if j % 2 == 0 { -8 } else { 7 }, _ => rng.below(16) as i8 - 8 };
                qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
                s += i32::from(v);
            }
            tok.push(A4Block { d: ds[(r * 3 + g) % ds.len()], s, qs });
        }
    }
    let mut epoch_major = vec![0u8; rows * GROUPS * BLK];
    for g in 0..GROUPS { for r in 0..rows {
        let b = &tok[r * GROUPS + g];
        let o = (g * rows + r) * BLK;
        epoch_major[o..o + 4].copy_from_slice(&b.d.to_le_bytes());
        epoch_major[o + 4..o + 8].copy_from_slice(&b.s.to_le_bytes());
        epoch_major[o + 8..o + BLK].copy_from_slice(&b.qs);
    } }
    (epoch_major, tok)
}

struct Scenario { name: String, div: usize, rows: usize, tiles: Vec<(i32, [i32; 16])>, experts: Vec<Expert> }

fn tile(expert: i32, slots: &[i32]) -> (i32, [i32; 16]) {
    let mut t = [-1i32; 16];
    t[..slots.len()].copy_from_slice(slots);
    (expert, t)
}

/// `n` slots of one expert, in tiles of 16 (pad -1).
fn tiles_for(expert: i32, slots: Vec<i32>) -> Vec<(i32, [i32; 16])> { slots.chunks(16).map(|c| tile(expert, c)).collect() }

fn scenarios(seed: u32, neg: u8) -> Result<Vec<Scenario>> {
    let mk = |kinds: &[Kind]| -> Result<Vec<Expert>> { kinds.iter().enumerate().map(|(i, k)| expert(*k, seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 7919))).collect() };
    let mut out = Vec::new();
    for (div, rows_per) in [(1usize, 20usize), (10, 4)] {
        for n in [1usize, 2, 7, 8, 15, 16, 17, 33] {
            let mut slots: Vec<i32> = (0..n).map(|i| (((i * 7 + 3) % (rows_per * div)) as i32)).collect();
            slots.dedup();
            let experts = mk(&[Kind::Random, Kind::EveryCode, Kind::Extreme])?;
            let e = (n % 3) as i32;
            let mut tiles = tiles_for(e, slots);
            tiles.push((-1, [0; 16])); // sentinel tile: must store nothing
            out.push(Scenario { name: format!("rows{n}_div{div}"), div, rows: rows_per, tiles, experts });
        }
    }
    // mixed experts and a non-monotone slot order
    let experts = mk(&[Kind::Random, Kind::Extreme, Kind::EveryCode])?;
    let mut tiles = tiles_for(2, vec![5, 3, 19, 0, 1, 2, 11, 12]);
    tiles.extend(tiles_for(0, (0..17).map(|i| (i * 3 % 20) as i32).collect()));
    tiles.extend(tiles_for(1, vec![7]));
    tiles.push(tile(0, &[-1, 4, -1, 9]));
    out.push(Scenario { name: "mixed_experts_unsorted_pad".into(), div: 1, rows: 20, tiles, experts });
    for (tag, first) in [("sentinel_ff", 0x3cffu16), ("sentinel_00", 0x3c00u16)] {
        let experts = mk(&[Kind::Sentinel(first), Kind::Sentinel(first), Kind::Random])?;
        let tiles = tiles_for(1, (0..16).collect());
        out.push(Scenario { name: format!("{tag}_last_row_chunk"), div: 1, rows: 16, tiles, experts });
    }
    // Negative controls (must fail): 1 flips a twin-side code, 2 flips an anchor-side (QT44) nibble of expert 0.
    for sc in &mut out {
        match neg { 1 => sc.experts[0].codes[12345] ^= 1, 2 => sc.experts[0].qt44[1000] ^= 0x10, _ => {} }
    }
    Ok(out)
}

fn align(n: usize) -> usize { n.div_ceil(SLAB_ALIGN) * SLAB_ALIGN }

/// One slab buffer: each expert's tensor at a 256-byte aligned offset, 0xFF sentinels between and after.
fn slab(tensors: &[&[u8]]) -> (Vec<u8>, Vec<usize>) {
    let mut buf = Vec::new();
    let mut offs = Vec::new();
    for t in tensors {
        offs.push(buf.len());
        buf.extend_from_slice(t);
        buf.resize(align(buf.len()) + SLAB_ALIGN, 0xff);
    }
    (buf, offs)
}

fn run_scenario(hip: &HipRuntime, f_acc: &Function, f_w3: &Function, f_mq4: &Function, sc: &Scenario, seed: u32) -> Result<Value> {
    let rows = sc.rows;
    let (xq, x_tok) = make_x(rows, seed ^ 0x51ed);
    let m_total = sc.tiles.len() * 16;
    let tile_ids: Vec<i32> = sc.tiles.iter().map(|t| t.0).collect();
    let sorted: Vec<i32> = sc.tiles.iter().flat_map(|t| t.1).collect();
    let (w3_slab, w3_offs) = slab(&sc.experts.iter().map(|e| e.w3.as_slice()).collect::<Vec<_>>());
    let (qt_slab, qt_offs) = slab(&sc.experts.iter().map(|e| e.qt44.as_slice()).collect::<Vec<_>>());
    let up = |bytes: &[u8]| -> Result<hip_bridge::DeviceBuffer> { let b = hip.malloc(bytes.len().max(8)).map_err(err)?; hip.memcpy_htod(&b, bytes).map_err(err)?; Ok(b) };
    let w3_dev = up(&w3_slab)?;
    let qt_dev = up(&qt_slab)?;
    let ptrs = |base: u64, offs: &[usize]| -> Vec<u8> { offs.iter().flat_map(|o| (base + *o as u64).to_le_bytes()).collect() };
    let w3_ptrs = up(&ptrs(w3_dev.as_ptr() as u64, &w3_offs))?;
    let qt_ptrs = up(&ptrs(qt_dev.as_ptr() as u64, &qt_offs))?;
    let tiles_dev = up(&tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let sorted_dev = up(&sorted.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let x_dev = up(&xq)?;
    let acc_bytes = m_total * M * 4;
    let h_bytes = m_total * HALF * 2;
    let y_acc = hip.malloc(acc_bytes).map_err(err)?;
    let y_w3 = hip.malloc(h_bytes).map_err(err)?;
    let y_mq4 = hip.malloc(h_bytes).map_err(err)?;
    let launch = |f: &Function, ptrs: &hip_bridge::DeviceBuffer, y: &hip_bridge::DeviceBuffer| -> Result<()> {
        let mut a = [0u8; 60];
        for (off, p) in [(0usize, hip_ptr(ptrs)), (8, hip_ptr(&tiles_dev)), (16, hip_ptr(&sorted_dev)), (24, hip_ptr(&x_dev)), (32, hip_ptr(y))] { a[off..off + 8].copy_from_slice(&p.to_le_bytes()); }
        for (off, v) in [(40usize, M as u32), (44, K as u32), (48, sc.div as u32), (52, m_total as u32), (56, rows as u32)] { a[off..off + 4].copy_from_slice(&v.to_le_bytes()); }
        // SAFETY: exact 60-byte ABI; every buffer outlives the synchronize.
        unsafe { hip.launch_kernel_blob(f, [(M / 64) as u32, sc.tiles.len() as u32, 1], [64, 1, 1], 0, None, &mut a) }.map_err(err)?;
        hip.device_synchronize().map_err(err)
    };
    hip.memset(&y_acc, 0xa5, acc_bytes).map_err(err)?;
    hip.memset(&y_w3, 0xa5, h_bytes).map_err(err)?;
    hip.memset(&y_mq4, 0xa5, h_bytes).map_err(err)?;
    launch(f_acc, &w3_ptrs, &y_acc)?;
    launch(f_w3, &w3_ptrs, &y_w3)?;
    launch(f_mq4, &qt_ptrs, &y_mq4)?;
    let read = |b: &hip_bridge::DeviceBuffer, n: usize| -> Result<Vec<u8>> { let mut v = vec![0u8; n]; hip.memcpy_dtoh(&mut v, b).map_err(err)?; Ok(v) };
    let (got_acc, got_w3, got_mq4) = (read(&y_acc, acc_bytes)?, read(&y_w3, h_bytes)?, read(&y_mq4, h_bytes)?);
    // Twin F32 accumulators per used expert over every X row.
    let mut dense: Vec<Option<Vec<f32>>> = vec![None; sc.experts.len()];
    for t in &sc.tiles { if t.0 >= 0 && dense[t.0 as usize].is_none() {
        let e = &sc.experts[t.0 as usize];
        dense[t.0 as usize] = Some(mq_i4_reference::iu4_gemm_f32_parts(profile(), M, K, &e.codes, &e.scales, &x_tok));
    } }
    let (mut acc_bad, mut first_acc, mut checked) = (0usize, None, 0usize);
    for (ti, t) in sc.tiles.iter().enumerate() { for lane in 0..16 {
        let p = ti * 16 + lane;
        let live = t.0 >= 0 && t.1[lane] >= 0;
        for r in 0..M {
            let bits = u32::from_le_bytes(got_acc[(p * M + r) * 4..(p * M + r) * 4 + 4].try_into().unwrap());
            let want = if t.0 < 0 { 0xa5a5_a5a5 } else if live { dense[t.0 as usize].as_ref().unwrap()[(t.1[lane] as usize / sc.div) * M + r].to_bits() } else { 0 };
            checked += 1;
            if bits != want { acc_bad += 1; if first_acc.is_none() { first_acc = Some(json!({"p": p, "row": r, "gpu": format!("{bits:#010x}"), "twin": format!("{want:#010x}")})); } }
        }
    } }
    // BF16 H: W3 entry vs the unchanged MQ4 entry on the equivalent QT44 tensors; sentinel tile untouched in both.
    let (mut h_bad, mut first_h) = (0usize, None);
    for i in 0..h_bytes / 2 {
        let (a, b) = (&got_w3[2 * i..2 * i + 2], &got_mq4[2 * i..2 * i + 2]);
        if a != b { h_bad += 1; if first_h.is_none() { first_h = Some(json!({"index": i, "w3": format!("{:#06x}", u16::from_le_bytes([a[0], a[1]])), "mq4": format!("{:#06x}", u16::from_le_bytes([b[0], b[1]]))})); } }
    }
    Ok(json!({"scenario": sc.name, "tiles": sc.tiles.len(), "acc_outputs_checked": checked, "acc_mismatches": acc_bad, "acc_first": first_acc, "h_mismatches": h_bad, "h_first": first_h}))
}

fn hip_ptr(b: &hip_bridge::DeviceBuffer) -> u64 { b.as_ptr() as u64 }

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn run() -> Result<()> {
    let (mut w3, mut mq4, mut seed, mut expect_bdf, mut neg) = (None, None, 1u32, None, 0u8);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().ok_or("flag needs a value")?;
        match a.as_str() { "--w3" => w3 = Some(v), "--mq4" => mq4 = Some(v), "--seed" => seed = v.parse().map_err(err)?, "--expect-bdf" => expect_bdf = Some(v.to_ascii_lowercase()), "--negative-control" => neg = v.parse().map_err(err)?, _ => return Err(format!("unknown flag {a}")) }
    }
    let (w3, mq4) = (w3.ok_or("--w3 required")?, mq4.ok_or("--mq4 required")?);
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let (uuid, bdf, arch) = (hip.device_uuid(id).map_err(err)?, hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase(), hip.get_arch(id).map_err(err)?);
    let device = json!({"uuid": uuid, "bdf": bdf, "arch": arch});
    if arch != "gfx1201" { return Err(format!("expected gfx1201, got {arch}")); }
    if let Some(b) = &expect_bdf { if *b != bdf { return Err(format!("expected BDF {b}, got {bdf}")); } }
    let (w3_bytes, mq4_bytes) = (std::fs::read(&w3).map_err(err)?, std::fs::read(&mq4).map_err(err)?);
    let (m_w3, m_mq4) = (hip.module_load_data(&w3_bytes).map_err(err)?, hip.module_load_data(&mq4_bytes).map_err(err)?);
    let sym = "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1201";
    let f_acc = hip.module_get_function(&m_w3, &format!("{sym}_acc")).map_err(err)?;
    let f_w3 = hip.module_get_function(&m_w3, sym).map_err(err)?;
    let f_mq4 = hip.module_get_function(&m_mq4, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201").map_err(err)?;
    let (mut rows, mut bad) = (Vec::new(), 0usize);
    for sc in scenarios(seed, neg)? {
        let r = run_scenario(&hip, &f_acc, &f_w3, &f_mq4, &sc, seed)?;
        bad += r["acc_mismatches"].as_u64().unwrap() as usize + r["h_mismatches"].as_u64().unwrap() as usize;
        rows.push(r);
    }
    println!("{}", json!({"schema": "qwen4-moe-sym-w3-g1-oracle-v1", "device": device, "w3_module": w3, "w3_module_sha256": sha256_hex(&w3_bytes),
        "mq4_module": mq4, "mq4_module_sha256": sha256_hex(&mq4_bytes), "seed": seed, "negative_control": neg, "scenarios": rows, "mismatch_total": bad, "pass": bad == 0}));
    if bad != 0 { return Err("K2 W3 G1 mismatch".into()); }
    Ok(())
}
fn main() { if let Err(e) = run() { eprintln!("{e}"); std::process::exit(1); } }
