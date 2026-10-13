// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next owner-routed chunk MoE peer tile pull builder module (gfx1201):
//! `ep_gather_grouped_rows_by_tile_owner_pm_gfx1201`. The emitted program runs on the
//! PeaceMaker CPU interpreter (256-thread workgroups, one per tile) against a host
//! reference: tiles of experts another rank owns are byte copies of the peer's rows, every
//! other byte of the destination (owned tiles, the -1 padding tail, bytes past the grid) is
//! untouched, and the peer buffer is never written. The "peer" is a distinct mapped region at
//! a high virtual address. No GPU.
use hipfire_isa::Arch;
use hipfire_isa::kernels::ep_gather_grouped_rows_by_tile_owner as pull;
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use std::sync::LazyLock;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCH: Arch = Arch::Gfx1201;
const BLOCK: u32 = pull::BLOCK;
/// Production tile: 16 grouped rows x 2560 BF16 = 5120 uint4.
const PRODUCTION_TILE_U4: usize = 16 * 2560 * 2 / 16;

#[test]
fn emission_is_deterministic_and_targets_gfx1201_only() {
    let (a, b) = (pull::emit(ARCH).unwrap(), pull::emit(ARCH).unwrap());
    assert_eq!(a.s_text, b.s_text);
    assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
    assert!(a.s_text.contains(&format!("{}:", pull::symbol(ARCH))));
    assert_eq!(a.shape.barriers, 0);
    assert!(a.shape.next_free_vgpr <= 24 && a.shape.next_free_sgpr <= 104, "{:?}", a.shape);
    let (emitted, text, _) = pull::emit_module(ARCH).unwrap();
    assert_eq!(emitted.len(), 1);
    assert!(text.contains(&format!("{}:", pull::symbol(ARCH))));
    assert_eq!(pull::module(ARCH), "ep_gather_grouped_rows_by_tile_owner_pm_gfx1201");
    assert_eq!(pull::symbol(ARCH), "ep_gather_grouped_rows_by_tile_owner_pm_gfx1201");
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(pull::emit(arch).is_err(), "{arch:?}");
        assert!(pull::emit_module(arch).is_err(), "{arch:?}");
    }
}

/// Kernarg ABI of `Gpu::ep_gather_grouped_rows_by_tile_owner`: four pointers, four i32 (48 bytes);
/// 256-thread workgroup, no LDS, no atomics, no scratch, no flat / buffer / ds access; the peer
/// loads are `buffer_load_b128` through a raw descriptor whose base is the peer's address.
#[test]
fn kernarg_layout_matches_the_launcher() {
    let text = pull::emit_module(ARCH).unwrap().1;
    let metadata = &text[text.find("amdhsa.kernels:").unwrap()..];
    let offsets: Vec<u32> = metadata.lines()
        .filter_map(|l| l.trim().strip_prefix(".offset:").map(|v| v.trim().parse::<u32>().unwrap())).collect();
    assert_eq!(offsets, [0, 8, 16, 24, 32, 36, 40, 44]);
    assert!(text.contains(".amdhsa_kernarg_size 48\n"), "kernarg size");
    assert!(text.contains(".amdhsa_group_segment_fixed_size 0\n"), "LDS");
    assert!(text.contains(".max_flat_workgroup_size: 256"), "workgroup size");
    assert!(!text.contains("atomic"));
    for l in text.lines().map(str::trim_start) {
        assert!(!(l.starts_with("scratch_") || l.starts_with("flat_") || l.starts_with("global_") || l.starts_with("ds_")), "{l}");
    }
    let loads = text.lines().filter(|l| l.trim_start().starts_with("buffer_load_b128")).count();
    let stores = text.lines().filter(|l| l.trim_start().starts_with("buffer_store_b128")).count();
    assert_eq!((loads, stores), (pull::UNROLL as usize, pull::UNROLL as usize));
}

#[test]
fn committed_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = pull::module(ARCH);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = pull::emit_module(ARCH).unwrap().1;
    assert!(text_section(&link(&text, ARCH.name(), &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle");
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("hipfire-isa-ep-tile-pull-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (dir.join(format!("{stem}.s")), dir.join(format!("{stem}.o")), dir.join(format!("{stem}.co")));
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc).args(["-triple=amdgcn-amd-amdhsa", &format!("-mcpu={arch}"), "-filetype=obj"]).arg(&s).arg("-o").arg(&o).status().unwrap().success());
    assert!(Command::new(linker).arg("-shared").arg(&o).arg("-o").arg(&co).status().unwrap().success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum).find_map(|i| {
        let name = &e[names + u32_(sh(i))..];
        name.starts_with(b".text\0").then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
    }).unwrap()
}

// ---- CPU interpreter vs the host reference ------------------------------------

static PROGRAM: LazyLock<peacemaker_ir::Program> = LazyLock::new(|| {
    let text = pull::emit_module(ARCH).unwrap().1;
    let elf = hipfire_isa::native::assemble(&text, ARCH).unwrap();
    peacemaker_lift::lift_object(&elf, peacemaker_lift::Options { frontend: peacemaker_ir::inst::Frontend::Builder }).unwrap().program
});

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

/// Bytes past the grid in the destination and the sentinel an unpulled tile keeps.
const TAIL: usize = 4096;
const SENTINEL: u8 = 0xC3;

/// One full grid (`grid_tiles` workgroups, which may exceed `tile_ids.len()`). Returns the destination.
/// The peer buffer's bytes are checked unmodified.
fn run(tile_ids: &[i32], owner_table: &[i32], rank: i32, tile_u4: usize, grid_tiles: usize, peer: &[u8]) -> Vec<u8> {
    let tiles = tile_ids.len();
    let tile_bytes = tile_u4 * 16;
    assert_eq!(peer.len(), tiles * tile_bytes);
    let emu = Emulator::new(&PROGRAM, &pull::symbol(ARCH)).unwrap();
    let (kp, ip, tp) = (0x1_0000u64, 0x10_0000u64, 0x20_0000u64);
    let (peer_va, dst_va) = (0x7f00_0000_0000u64, 0x1000_0000u64);
    let words = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let mut kern = Vec::new();
    for p in [ip, tp, peer_va, dst_va] { kern.extend(p.to_le_bytes()); }
    for x in [owner_table.len() as i32, rank, tile_u4 as i32, tiles as i32] { kern.extend(x.to_le_bytes()); }
    assert_eq!(kern.len(), pull::KERNARG_BYTES as usize);
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kp, kern).unwrap();
    mem.map("tile_ids".into(), ip, words(tile_ids)).unwrap();
    mem.map("table".into(), tp, words(owner_table)).unwrap();
    mem.map("peer".into(), peer_va, peer.to_vec()).unwrap();
    mem.map("dst".into(), dst_va, vec![SENTINEL; peer.len() + TAIL]).unwrap();
    for wg in 0..grid_tiles {
        emu.run_workgroup(&Launch { kernarg_ptr: kp, workgroup: [wg as u32, 0, 0], block: [BLOCK, 1, 1], dynamic_lds: 0 }, &mut mem)
            .unwrap_or_else(|e| panic!("tiles {tiles} rank {rank} tile_u4 {tile_u4} wg {wg}: {e}"));
    }
    let region = |name: &str| mem.regions.iter().find(|r| r.name == name).unwrap().bytes.clone();
    assert_eq!(region("peer"), peer, "peer buffer modified");
    assert_eq!(region("tile_ids"), words(tile_ids), "tile ids modified");
    region("dst")
}

/// Peer bytes: every byte pattern class (NaN payloads, +-0, denormals, all-ones) among random bytes.
fn peer_bytes(rng: &mut Rng, len: usize) -> Vec<u8> {
    let specials: [u32; 8] = [0x7fc0_0001, 0xffc0_0000, 0x8000_0000, 0x0000_0001, 0x0080_0000, 0x7f80_0000, 0xffff_ffff, 0];
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let w = if rng.below(4) == 0 { specials[rng.below(8)] } else { rng.next() as u32 };
        out.extend(w.to_le_bytes());
    }
    out.truncate(len);
    out
}

fn check(name: &str, rng: &mut Rng, tile_ids: &[i32], owner_table: &[i32], rank: i32, tile_u4: usize, grid_extra: usize) {
    let tiles = tile_ids.len();
    let tile_bytes = tile_u4 * 16;
    let peer = peer_bytes(rng, tiles * tile_bytes);
    let dst = run(tile_ids, owner_table, rank, tile_u4, tiles + grid_extra, &peer);
    let mut want = vec![SENTINEL; peer.len() + TAIL];
    for (t, &id) in tile_ids.iter().enumerate() {
        let live = id >= 0 && (id as usize) < owner_table.len();
        if live && owner_table[id as usize] != rank {
            want[t * tile_bytes..(t + 1) * tile_bytes].copy_from_slice(&peer[t * tile_bytes..(t + 1) * tile_bytes]);
        }
    }
    if dst != want {
        let first = dst.iter().zip(&want).position(|(a, b)| a != b).unwrap();
        panic!("{name}: tiles {tiles} rank {rank} tile_u4 {tile_u4}: first differing byte {first} (tile {}) ids {tile_ids:?}", first / tile_bytes);
    }
}

fn table(experts: usize, ranks: usize) -> Vec<i32> { (0..experts).map(|e| (e % ranks) as i32).collect() }

/// Mixed ownership and the -1 tail on small tiles (1024, 2048 and 3072 uint4: one to three trips)
/// for both ranks of a two-rank split.
#[test]
fn emulated_launches_equal_the_host_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let owner = table(512, 2);
    let mut cases = 0usize;
    for &tile_u4 in &[1024usize, 2048, 3072] {
        for &tiles in &[1usize, 2, 7, 24] {
            for trial in 0..3 {
                let live = tiles - rng.below(tiles / 3 + 1);
                let mut ids: Vec<i32> = (0..live).map(|_| rng.below(512) as i32).collect();
                if trial == 0 { for (i, id) in ids.iter_mut().enumerate() { *id = (i % 2) as i32 + 2 * (i as i32 / 3); } }
                ids.resize(tiles, -1);
                for rank in 0..2 {
                    check(&format!("tile_u4 {tile_u4} trial {trial}"), &mut rng, &ids, &owner, rank, tile_u4, 0);
                    cases += 1;
                }
            }
        }
    }
    println!("emulated_launches_equal_the_host_reference: {cases} launches");
}

/// The production tile (16 rows x 2560 BF16 = 5120 uint4, five 1024-uint4 trips of the unrolled
/// walk) with a handful of tiles of both owners and sentinels.
#[test]
fn emulated_production_tile_equals_the_host_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    assert_eq!(PRODUCTION_TILE_U4 % pull::TRIP_U4 as usize, 0);
    let owner = table(512, 2);
    let ids = [0, 1, 1, 2, 3, 5, 6, -1, -1];
    for rank in 0..2 {
        check("production tile", &mut rng, &ids, &owner, rank, PRODUCTION_TILE_U4, 0);
    }
}

/// Grid larger than the tile count (the workgroups past `tiles` return), ids outside the table (never
/// pulled), a table owner no rank holds is pulled (it is "not this rank"), an all-owned grid and an
/// all-peer grid.
#[test]
fn emulated_edge_launches_equal_the_host_reference() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let owner = table(512, 2);
    for rank in 0..2 {
        check("grid past tiles", &mut rng, &[3, 4, -1], &owner, rank, 1024, 5);
        check("odd ids", &mut rng, &[-1, i32::MIN, i32::MAX, 512, 513, 0x4000_0000, 511, 0, 1], &owner, rank, 1024, 0);
        check("one tile", &mut rng, &[9], &owner, rank, 1024, 0);
        check("all sentinel", &mut rng, &[-1; 6], &owner, rank, 1024, 0);
    }
    check("all owned", &mut rng, &[0, 2, 4, 6, 8, 10], &owner, 0, 1024, 0);
    check("all peer", &mut rng, &[0, 2, 4, 6, 8, 10], &owner, 1, 1024, 0);
    let arbitrary = [0, 2, -1, 1, 7, 0, 1, 2];
    for rank in 0..3 {
        check("arbitrary owners", &mut rng, &[0, 1, 2, 3, 4, 5, 6, 7, 8, -1], &arbitrary, rank, 1024, 0);
    }
}

/// Random shapes: tiles 1..=40, tile widths 1024..=4096, any rank of 2, ids uniform / runs / odd.
#[test]
fn emulated_random_shapes_equal_the_host_reference() {
    let mut rng = Rng(0x2F0B_4A32_7C15_91E3);
    let cases = 40usize;
    for case in 0..cases {
        let tiles = 1 + rng.below(40);
        let tile_u4 = 1024 * (1 + rng.below(4));
        let rank = rng.below(2) as i32;
        let owner = table(512, 2);
        let ids: Vec<i32> = (0..tiles).map(|_| match rng.below(8) {
            0 => -1,
            1 => [i32::MIN, 512, 700, 0x4000_0000][rng.below(4)],
            _ => rng.below(512) as i32,
        }).collect();
        let extra = rng.below(3);
        check(&format!("shape #{case}"), &mut rng, &ids, &owner, rank, tile_u4, extra);
    }
    println!("emulated_random_shapes_equal_the_host_reference: {cases} launches");
}

/// Every instruction of the module has interpreter semantics.
#[test]
fn interpreter_covers_the_module() {
    let mut missing = std::collections::BTreeSet::new();
    for k in &PROGRAM.kernels {
        for id in &k.body.layout {
            let i = k.body.insts.get(*id).unwrap();
            let row = peacemaker_ir::isa::lookup(PROGRAM.target.arch, i.op, i.form).unwrap();
            if peacemaker_ir::emu::semantic_class(row.name).is_none() { missing.insert(row.name); }
        }
    }
    assert!(missing.is_empty(), "no interpreter semantics: {missing:?}");
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{ledger_replay, pm_check, toolchain::{build, certify, Toolchain}};

    /// The symbol certifies: parse-back, independent wait replay, the committed shape contract,
    /// the static LDS bound, and M7's lift identity plus wait/hazard/barrier/window analyses of the
    /// linked ELF, obligation-free.
    #[test]
    fn module_passes_certification() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let (emitted, text, _) = pull::emit_module(ARCH).unwrap();
        for e in &emitted { ledger_replay::replay_waits(&e.s_text, ARCH).unwrap(); }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-ep-tile-pull-cert-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("module.hsaco"), ARCH.name()).unwrap();
        let symbol = pull::symbol(ARCH);
        let m7 = pm_check::m7(&build.elf, ARCH.name(), &symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
        assert_eq!((m7["lift"].as_str(), &m7["obligations"]), (Some("byte-exact"), &serde_json::json!({})), "{symbol}: {m7}");
        let path = format!("{}/kernels/ep_gather_grouped_rows_by_tile_owner_pm.{}.contract.json", env!("CARGO_MANIFEST_DIR"), ARCH.name());
        let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(json, pull::contract(ARCH, &emitted[0]), "{path} is stale");
        let contract = serde_json::from_slice(&raw).unwrap();
        certify(&toolchain, &build, &s, ARCH.name(), &dir.join("manifest.json"), Some(&contract), "test", "test")
            .unwrap_or_else(|e| panic!("{symbol}: {e}"));
    }
}
