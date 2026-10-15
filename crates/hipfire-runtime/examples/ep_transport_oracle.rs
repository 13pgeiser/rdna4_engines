// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synthetic ONE-device oracle for the Flash-Next expert-parallel transport
//! launchers at N = 2..=8 ranks. No peer access and no model: every "rank" is
//! a set of buffers on the same physical card, so the launchers' routing,
//! bounds and bit-exactness are checked against a host reference.
//!
//! ```sh
//! cargo build --release -p hipfire-runtime --features lab --example ep_transport_oracle
//! HIPFIRE_DEVICES=0 target/release/examples/ep_transport_oracle \
//!     [--ranks 2,3,6] [--seed N] [--slots 37] [--device 0] \
//!     [--suites rows,tile] [--json receipt.json]
//! ```
//!
//! Exit status: 0 only when every case passed; 1 on any mismatch or launcher
//! error; 2 on usage errors / no device.
//!
//! For every `n` in `--ranks` and every `self_rank` in `0..n`:
//!
//! * `rows_by_owner` / `rows_by_slot_owner` (`Gpu::ep_gather_rows_by_owner` /
//!   `ep_gather_rows_by_slot_owner`, `src: [usize; 8]`): `n` source row buffers
//!   (rank `r`'s buffer = seeded random bytes with NaN payloads, -0.0,
//!   denormals, infinities in every third word) and a destination with its own
//!   seeded bytes. Slots carry random expert ids (including negative ids) or
//!   explicit owners (including `-1`). Expected destination: row `s` equals
//!   `src[owner]` row `s` when `owner` is a real peer, else the destination's
//!   ORIGINAL bytes (owned rows, negative ids / owners are never written).
//!   Two row widths (`n16` = 640 and 257 uint4) cover the grid tail.
//! * `tile_mask` (`Gpu::ep_tile_mask`): 1000 tile ids over 512 experts with
//!   `-1` sentinels and a padding tail; owner table `e % n`. Expected
//!   `owned[t] = ids[t]` when the tile's expert is owned by `self_rank`, else
//!   `-1`; the source ids must be unchanged.
//! * `tile_pull` (`Gpu::ep_gather_grouped_rows_by_tile_owner`,
//!   `sources: [usize; 8]`, owner table `e % n`): `n` grouped-row buffers of
//!   40 tiles x 16 rows (two row widths); expected destination: every live
//!   tile whose expert another rank owns is the owner's tile bytes, every other
//!   tile (owned, `-1` sentinel, padding tail) is the destination's original
//!   bytes.
//!
//! Every destination byte is compared. Cases print `PASS <case>` or
//! `FAIL <case>: <first difference>`; `--json` writes a receipt.

use rdna_compute::moe::{ep_tile_pm_arch, EP_TILE_ROWS};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};

type R<T> = Result<T, String>;

const EXPERTS: usize = 512;
const MASK_TILES: usize = 1000;
const PULL_TILES: usize = 40;
const MASK_POISON: i32 = 0x5A5A_5A5A;

const SPECIAL: [u32; 16] = [
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x7fc0_0000, // quiet NaN
    0x7f80_0001, // signalling NaN
    0xffc1_2345, // negative quiet NaN with payload
    0xff80_0001, // negative signalling NaN
    0x7fa5_5a5a, // signalling NaN, wide payload
    0x0000_0001, // smallest denormal
    0x007f_ffff, // largest denormal
    0x8000_0001, // negative smallest denormal
    0x807f_ffff, // negative largest denormal
    0x7f7f_ffff, // max finite
    0xff7f_ffff, // -max finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x3f80_0000, // 1.0
];

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Deterministic stream of u64 from a seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = splitmix(self.0);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `len` bytes of seeded random words; every third word is a special bit
/// pattern (rotating through [`SPECIAL`] by seed).
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for (i, w) in out.chunks_exact_mut(4).enumerate() {
        let v = if i % 3 == 0 {
            SPECIAL[(i / 3 + seed as usize) % SPECIAL.len()]
        } else {
            splitmix(seed.wrapping_mul(0x1_0000_0001).wrapping_add(i as u64)) as u32
        };
        w.copy_from_slice(&v.to_le_bytes());
    }
    out
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// First differing byte of two slices, as a report string.
fn first_diff(label: &str, expected: &[u8], got: &[u8], row_bytes: usize) -> Option<String> {
    if expected.len() != got.len() {
        return Some(format!("{label}: length {} vs {}", expected.len(), got.len()));
    }
    let offset = expected.iter().zip(got).position(|(a, b)| a != b)?;
    let bad = expected.iter().zip(got).filter(|(a, b)| a != b).count();
    let word = offset & !3;
    let end = (word + 4).min(expected.len());
    Some(format!(
        "{label}: {bad} differing byte(s); first at byte {offset} (row {}, word@{word} expected {} got {})",
        offset / row_bytes.max(1),
        hex(&expected[word..end]),
        hex(&got[word..end]),
    ))
}

fn ctx<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> R<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

struct Rig {
    gpu: Gpu,
}

impl Rig {
    fn upload(&self, bytes: &[u8], what: &str) -> R<GpuTensor> {
        ctx(self.gpu.upload_raw(bytes, &[bytes.len()]), what)
    }
    fn sync(&self) -> R<()> {
        ctx(self.gpu.bind_thread(), "bind_thread")?;
        ctx(self.gpu.hip.device_synchronize(), "device_synchronize")
    }
    fn download(&self, t: &GpuTensor, what: &str) -> R<Vec<u8>> {
        self.sync()?;
        ctx(self.gpu.download_raw_bytes(t), what)
    }
}

fn addr(t: &GpuTensor) -> usize {
    t.buf.as_ptr() as usize
}

fn sources8(tensors: &[GpuTensor]) -> [usize; 8] {
    let mut out = [0usize; 8];
    for (slot, t) in out.iter_mut().zip(tensors) {
        *slot = addr(t);
    }
    out
}

// ───────────────────────────── row gathers ─────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum RowKind {
    ByOwner,
    BySlotOwner,
}

fn row_case(rig: &mut Rig, kind: RowKind, n: usize, self_rank: usize, slots: usize, n16: usize, seed: u64) -> R<Option<String>> {
    let row_bytes = n16 * 16;
    let len = slots * row_bytes;
    let case_seed = splitmix(seed ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40) ^ ((n16 as u64) << 8) ^ kind as u64);
    // Slot -> expert id (modulo kernel) or explicit owner (slot-owner kernel).
    let mut rng = Rng(case_seed);
    let ids: Vec<i32> = (0..slots)
        .map(|s| match kind {
            RowKind::ByOwner => {
                if s % 7 == 3 {
                    -1 - rng.below(5) as i32
                } else {
                    rng.below(EXPERTS) as i32
                }
            }
            RowKind::BySlotOwner => {
                if s % 7 == 3 {
                    -1
                } else {
                    rng.below(n) as i32
                }
            }
        })
        .collect();
    let owner_of = |id: i32| -> Option<usize> {
        match kind {
            RowKind::ByOwner => (id >= 0).then(|| id as usize % n),
            RowKind::BySlotOwner => (id >= 0 && (id as usize) < n).then_some(id as usize),
        }
    };
    let src_host: Vec<Vec<u8>> = (0..n).map(|r| random_bytes(case_seed ^ splitmix(r as u64 + 1), len)).collect();
    let dst_host = random_bytes(case_seed ^ 0xD57D_57D5, len);
    let mut expected = dst_host.clone();
    for (s, id) in ids.iter().enumerate() {
        if let Some(owner) = owner_of(*id) {
            if owner != self_rank {
                expected[s * row_bytes..(s + 1) * row_bytes].copy_from_slice(&src_host[owner][s * row_bytes..(s + 1) * row_bytes]);
            }
        }
    }
    let mut src = Vec::with_capacity(n);
    for (r, host) in src_host.iter().enumerate() {
        src.push(rig.upload(host, &format!("upload source {r}"))?);
    }
    let dst = rig.upload(&dst_host, "upload destination")?;
    let id_tensor = rig.upload(&i32_bytes(&ids), "upload slot ids")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    let sources = sources8(&src);
    let launch = match kind {
        RowKind::ByOwner => rig.gpu.ep_gather_rows_by_owner(&id_tensor, sources, addr(&dst), slots, n, self_rank, n16),
        RowKind::BySlotOwner => rig.gpu.ep_gather_rows_by_slot_owner(&id_tensor, sources, addr(&dst), slots, n, self_rank, n16),
    };
    ctx(launch, "launch")?;
    let got = rig.download(&dst, "download destination")?;
    let mut problem = first_diff("destination", &expected, &got, row_bytes);
    if problem.is_none() {
        // Sources and ids are read-only inputs.
        for (r, host) in src_host.iter().enumerate() {
            let after = rig.download(&src[r], "download source")?;
            if let Some(p) = first_diff(&format!("source {r} (read-only)"), host, &after, row_bytes) {
                problem = Some(p);
                break;
            }
        }
    }
    Ok(problem)
}

// ───────────────────────────── tile mask / pull ─────────────────────────────

/// Tile ids over `EXPERTS` with ~20% `-1` sentinels and a `-1` padding tail.
fn tile_ids(seed: u64, tiles: usize) -> Vec<i32> {
    let mut rng = Rng(seed);
    let live = tiles - tiles / 5;
    (0..tiles)
        .map(|t| if t >= live || t % 5 == 2 { -1 } else { rng.below(EXPERTS) as i32 })
        .collect()
}

fn owner_table(n: usize) -> Vec<i32> {
    (0..EXPERTS).map(|e| (e % n) as i32).collect()
}

fn mask_case(rig: &mut Rig, n: usize, self_rank: usize, seed: u64) -> R<Option<String>> {
    let case_seed = splitmix(seed ^ 0x3A5C ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40));
    let ids = tile_ids(case_seed, MASK_TILES);
    let table = owner_table(n);
    let expected: Vec<i32> = ids
        .iter()
        .map(|&id| if id >= 0 && table[id as usize] as usize == self_rank { id } else { -1 })
        .collect();
    let ids_t = rig.upload(&i32_bytes(&ids), "upload tile ids")?;
    let owned_t = rig.upload(&i32_bytes(&vec![MASK_POISON; MASK_TILES]), "upload owned ids")?;
    let table_t = rig.upload(&i32_bytes(&table), "upload owner table")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    ctx(rig.gpu.ep_tile_mask(&ids_t, &owned_t, &table_t, MASK_TILES, EXPERTS, self_rank), "launch")?;
    let got = rig.download(&owned_t, "download owned ids")?;
    if let Some(p) = first_diff("tile_ids_owned", &i32_bytes(&expected), &got, 4) {
        return Ok(Some(p));
    }
    let ids_after = rig.download(&ids_t, "download tile ids")?;
    Ok(first_diff("tile_ids (read-only)", &i32_bytes(&ids), &ids_after, 4))
}

fn pull_case(rig: &mut Rig, n: usize, self_rank: usize, row_u4: usize, seed: u64) -> R<Option<String>> {
    let tile_bytes = EP_TILE_ROWS * row_u4 * 16;
    let len = PULL_TILES * tile_bytes;
    let case_seed = splitmix(seed ^ 0x9111 ^ ((n as u64) << 48) ^ ((self_rank as u64) << 40) ^ ((row_u4 as u64) << 8));
    let ids = tile_ids(case_seed, PULL_TILES);
    let table = owner_table(n);
    let src_host: Vec<Vec<u8>> = (0..n).map(|r| random_bytes(case_seed ^ splitmix(r as u64 + 1), len)).collect();
    let dst_host = random_bytes(case_seed ^ 0xD57D_57D5, len);
    let mut expected = dst_host.clone();
    for (t, &id) in ids.iter().enumerate() {
        if id < 0 {
            continue;
        }
        let owner = table[id as usize] as usize;
        if owner != self_rank {
            expected[t * tile_bytes..(t + 1) * tile_bytes].copy_from_slice(&src_host[owner][t * tile_bytes..(t + 1) * tile_bytes]);
        }
    }
    let mut src = Vec::with_capacity(n);
    for (r, host) in src_host.iter().enumerate() {
        src.push(rig.upload(host, &format!("upload source {r}"))?);
    }
    let dst = rig.upload(&dst_host, "upload destination")?;
    let ids_t = rig.upload(&i32_bytes(&ids), "upload tile ids")?;
    let table_t = rig.upload(&i32_bytes(&table), "upload owner table")?;
    ctx(rig.gpu.bind_thread(), "bind_thread")?;
    ctx(
        rig.gpu.ep_gather_grouped_rows_by_tile_owner(
            &ids_t,
            &table_t,
            sources8(&src),
            &dst,
            PULL_TILES,
            EXPERTS,
            n,
            self_rank,
            row_u4,
        ),
        "launch",
    )?;
    let got = rig.download(&dst, "download destination")?;
    let mut problem = first_diff("destination", &expected, &got, tile_bytes);
    if problem.is_none() {
        for (r, host) in src_host.iter().enumerate() {
            let after = rig.download(&src[r], "download source")?;
            if let Some(p) = first_diff(&format!("source {r} (read-only)"), host, &after, tile_bytes) {
                problem = Some(p);
                break;
            }
        }
    }
    Ok(problem)
}

// ───────────────────────────── driver ─────────────────────────────

struct Args {
    ranks: Vec<usize>,
    seed: u64,
    slots: usize,
    device: i32,
    rows: bool,
    tile: bool,
    json: Option<String>,
}

fn parse_args() -> R<Args> {
    let mut a = Args { ranks: vec![2, 3, 6], seed: 0x5EED_0E9A, slots: 37, device: 0, rows: true, tile: true, json: None };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--ranks" => {
                a.ranks = value("--ranks")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().map_err(|e| format!("--ranks {s:?}: {e}")))
                    .collect::<R<_>>()?;
            }
            "--seed" => a.seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--slots" => a.slots = value("--slots")?.parse().map_err(|e| format!("--slots: {e}"))?,
            "--device" => a.device = value("--device")?.parse().map_err(|e| format!("--device: {e}"))?,
            "--json" => a.json = Some(value("--json")?),
            "--suites" => {
                let v = value("--suites")?;
                if let Some(bad) = v.split(',').find(|s| !matches!(*s, "rows" | "tile")) {
                    return Err(format!("--suites takes rows,tile; got {bad:?}"));
                }
                a.rows = v.split(',').any(|s| s == "rows");
                a.tile = v.split(',').any(|s| s == "tile");
            }
            "-h" | "--help" => return Err("usage: ep_transport_oracle [--ranks 2,3,6] [--seed N] [--slots 37] [--device 0] [--suites rows,tile] [--json FILE]".into()),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if a.ranks.is_empty() || a.ranks.iter().any(|n| !(2..=8).contains(n)) {
        return Err("--ranks entries must be within 2..=8".into());
    }
    if a.slots == 0 {
        return Err("--slots must be non-zero".into());
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let gpu = match Gpu::init_with_device(args.device) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device: {e}");
            std::process::exit(2);
        }
    };
    let arch = gpu.arch.clone();
    println!(
        "ep_transport_oracle device={} arch={arch} tile_pm_arch={} ranks={:?} seed={:#x}",
        args.device,
        ep_tile_pm_arch(&arch),
        args.ranks,
        args.seed
    );
    let mut rig = Rig { gpu };
    let (mut pass, mut fail) = (0usize, 0usize);
    let mut cases: Vec<Value> = Vec::new();
    let mut record = |name: String, outcome: R<Option<String>>| {
        let (ok, detail) = match outcome {
            Ok(None) => (true, None),
            Ok(Some(p)) => (false, Some(p)),
            Err(e) => (false, Some(format!("error: {e}"))),
        };
        match &detail {
            None => println!("PASS {name}"),
            Some(d) => println!("FAIL {name}: {d}"),
        }
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
        cases.push(json!({"case": name, "pass": ok, "detail": detail}));
    };
    for &n in &args.ranks {
        for self_rank in 0..n {
            if args.rows {
                for (kind, label) in [(RowKind::ByOwner, "rows_by_owner"), (RowKind::BySlotOwner, "rows_by_slot_owner")] {
                    for n16 in [640usize, 257] {
                        let name = format!("{label} n={n} self={self_rank} slots={} n16={n16}", args.slots);
                        let outcome = row_case(&mut rig, kind, n, self_rank, args.slots, n16, args.seed);
                        record(name, outcome);
                    }
                }
            }
            if args.tile {
                record(format!("tile_mask n={n} self={self_rank} tiles={MASK_TILES}"), mask_case(&mut rig, n, self_rank, args.seed));
                for row_u4 in [64usize, 128] {
                    record(
                        format!("tile_pull n={n} self={self_rank} tiles={PULL_TILES} row_u4={row_u4}"),
                        pull_case(&mut rig, n, self_rank, row_u4, args.seed),
                    );
                }
            }
        }
    }
    println!("RESULT: {} ({pass} passed, {fail} failed)", if fail == 0 { "PASS" } else { "FAIL" });
    if let Some(path) = &args.json {
        let receipt = json!({
            "arch": arch,
            "device": args.device,
            "tile_pm_arch": ep_tile_pm_arch(&arch),
            "ranks": args.ranks,
            "seed": args.seed,
            "slots": args.slots,
            "passed": pass,
            "failed": fail,
            "cases": cases,
        });
        if let Err(e) = std::fs::write(path, serde_json::to_string_pretty(&receipt).unwrap_or_default()) {
            eprintln!("--json {path}: {e}");
            std::process::exit(2);
        }
    }
    std::process::exit(if fail == 0 && pass > 0 { 0 } else { 1 });
}
