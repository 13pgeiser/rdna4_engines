// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 tile pull of the Flash-Next
//! owner-routed chunk MoE (`HIPFIRE_EP_OWNER_ROUTED_PREFILL=1`):
//! `ep_gather_grouped_rows_by_tile_owner_pm_gfx1201`. For every live tile whose
//! expert another rank owns, copy the tile's sixteen grouped rows of the PEER's
//! `y_down_grouped` (peer-mapped device memory) onto the same rows of this
//! rank's buffer. Both ranks run the same stable grouping, so a tile has the
//! same row offset on both. Rows of tiles this rank owns, and of the padding
//! tail, are never written. Pure `buffer_load_b128` / `buffer_store_b128`: the
//! copy is bit-exact for every pattern (+-0, NaN payloads, denormals).
//!
//! ABI (eight arguments, 48 bytes): tile_ids `*i32` @0, owner_table `*i32` @8
//! (one owner rank per expert), peer_src `*u8` @16 (the peer's `y_down_grouped`
//! base), dst `*u8` @24 (this rank's), n_experts i32 @32, rank i32 @36, tile_u4
//! i32 @40 (uint4 per tile = 16 rows x row_bytes / 16), tiles i32 @44. Grid
//! `[tiles, 1, 1]` (the padded tile capacity), block 256, no LDS, no barrier,
//! no scratch. Contract (the caller's checks): `tile_u4 % 1024 == 0`, `tiles *
//! tile_u4 * 16 <= 2^32 - 1`, `tile_ids` holds `tiles` words, `owner_table`
//! holds `n_experts` words, the peer base is mapped and the tile range is inside
//! both buffers.
//!
//! Law (the same predicate as the mask kernel's complement):
//! ```text
//!   pull(t) = 0 <= tile_ids[t] < n_experts && owner_table[tile_ids[t]] != rank
//!   pull(t) => dst[t * 16 rows ..+ 16 rows] = peer_src[same bytes]
//! ```
//!
//! Schedule: workgroup `t` reads its id with an SMEM load and returns
//! (workgroup-uniform) when it is the padding sentinel / out of range or the
//! owner is this rank, before any VMEM access. A pulling workgroup walks the
//! tile `1024 * 16` bytes per trip through two raw buffer descriptors (the
//! peer's virtual address is just a descriptor base): each lane keeps four
//! `buffer_load_b128` in flight at byte `16 tid + {0, 4096, 8192, 12288}` of the
//! trip, then stores them to the same offsets; the two bases advance by 16384
//! bytes per trip (`tile_u4 / 1024` trips).
use super::common::{add64, add64_imm, bload, bstore_b128, op, s, smem, sop, srd_tail, v};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, insn::Instruction, plan::Access, reg::{Kind, Live}};
use peacemaker_author::{End, Gfx1201, Scc, WgUniform, Workgroup};

type Wg<'b> = Workgroup<'b, Gfx1201, Builder>;
type R = Result<(), String>;

/// Threads per workgroup.
pub const BLOCK: u32 = 256;
/// Loads in flight per lane and trip.
pub const UNROLL: u32 = 4;
/// uint4 moved per workgroup trip; `tile_u4` is a multiple of it.
pub const TRIP_U4: u32 = BLOCK * UNROLL;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 48;

pub fn symbol(arch: Arch) -> String { format!("ep_gather_grouped_rows_by_tile_owner_pm_{}", arch.name()) }
pub fn module(arch: Arch) -> String { format!("ep_gather_grouped_rows_by_tile_owner_pm_{}", arch.name()) }

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg words 0..8 -> s8..s15 (tile_ids 8:9, table 10:11, peer 12:13, dst 14:15), words 8..12 -> s16..s19.
const TIDS: u8 = 8; const TABLE: u8 = 10; const PEER: u8 = 12; const DST: u8 = 14;
const NEXP: u8 = 16; const RANK: u8 = 17; const TU4: u8 = 18; const TILES: u8 = 19;
const WGX: u8 = 20; const ID: u8 = 21; const OWN: u8 = 22; const T0: u8 = 23; const T64: u8 = 24;
const TRIPS: u8 = 26; const BYTES: u8 = 27;
// Source / destination raw buffer descriptors of the trip loop (4-aligned SGPR quads).
const SRD_SRC: u8 = 32; const SRD_OUT: u8 = 36;
const S_END: u8 = 40;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0; const VOFF: u8 = 1;
/// Four 16-byte loads per lane and trip (each quad inside one 8-VGPR plan block).
const DATA: u8 = 8;
const V_END: u8 = 24;

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("tile_ids", 0, Access::ReadOnly)
        .pointer_access("owner_table", 8, Access::ReadOnly)
        .pointer_access("peer_src", 16, Access::ReadOnly)
        .pointer_access("dst", 24, Access::WriteOnly)
        .hidden("n_experts", 32, 4, "by_value")
        .hidden("rank", 36, 4, "by_value")
        .hidden("tile_u4", 40, 4, "by_value")
        .hidden("tiles", 44, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) { p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?; }
    for base in (0u8..V_END).step_by(8) { p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?; }
    Ok(p)
}

/// The builder-emitted kernel (gfx1201 only; every other target is refused).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 { return Err(format!("ep_gather_grouped_rows_by_tile_owner_pm: built for gfx1201 only, not {}", arch.name())) }
    let kspec = KernelSpec {
        kernel_id: "ep_gather_grouped_rows_by_tile_owner_pm".into(), variant: "pull".into(), arch, symbol: symbol(arch), kernargs: kernargs(),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: BLOCK as u16, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The single-symbol module `ep_gather_grouped_rows_by_tile_owner_pm_<arch>`.
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn cmp_wg(wg: &mut Wg, text: String, uses: &[crate::reg::RegRef]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}

// ---- kernel -----------------------------------------------------------------
fn kernel(wg: &mut Wg) -> R {
    let end = wg.exit(".Lpull_end")?;
    prologue(wg, &end)?;
    copy_tile(wg)?;
    wg.end(end)
}

/// Kernarg loads and the three workgroup-uniform exits: tile out of range, tile
/// without a live expert, tile owned by this rank. Leaves the running source and
/// destination pairs at the tile's first byte and `TRIPS` = `tile_u4 / 1024`.
fn prologue(wg: &mut Wg, end: &End) -> R {
    let b = wg.isa();
    smem(b, TIDS, 8, KARG, 0x00)?;
    smem(b, NEXP, 4, KARG, 0x20)?;
    // Workgroup id x (gfx12: ttmp9) = the tile.
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    let outside = cmp_wg(wg, format!("s_cmp_ge_u32 s{WGX}, s{TILES}"), &[s(WGX), s(TILES)])?;
    wg.exit_if(outside, end)?;
    // id = tile_ids[tile]; the unsigned compare also rejects the -1 sentinel.
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGX}, 2"), &[T0], &[WGX])?;
    add64(b, T64, TIDS, T0)?;
    smem(b, ID, 1, T64, 0)?;
    let dead = cmp_wg(wg, format!("s_cmp_ge_u32 s{ID}, s{NEXP}"), &[s(ID), s(NEXP)])?;
    wg.exit_if(dead, end)?;
    // owner = owner_table[id].
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{ID}, 2"), &[T0], &[ID])?;
    add64(b, T64, TABLE, T0)?;
    smem(b, OWN, 1, T64, 0)?;
    let mine = cmp_wg(wg, format!("s_cmp_eq_u32 s{OWN}, s{RANK}"), &[s(OWN), s(RANK)])?;
    wg.exit_if(mine, end)?;
    // Tile byte offset = tile * tile_u4 * 16 (fits 32 bits by contract); trips; the
    // source / destination raw descriptors (base = pointer + tile offset, num_records =
    // the tile's bytes).
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{BYTES}, s{TU4}, 4"), &[BYTES], &[TU4])?;
    sop(b, format!("s_mul_i32 s{T0}, s{WGX}, s{BYTES}"), &[T0], &[WGX, BYTES])?;
    add64(b, SRD_SRC, PEER, T0)?;
    srd_tail(b, SRD_SRC, Some(BYTES))?;
    add64(b, SRD_OUT, DST, T0)?;
    srd_tail(b, SRD_OUT, Some(BYTES))?;
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{TU4}, 10"), &[TRIPS], &[TU4])?;
    op(b, format!("v_lshlrev_b32_e32 v{VOFF}, 4, v{TID}"), &[v(VOFF)], &[v(TID)])
}

/// One trip: four peer loads in flight, four local stores, advance the two descriptor bases.
fn copy_tile(wg: &mut Wg) -> R {
    wg.loop_carried(".Lpull_trip", (), |wg, ()| {
        let b = wg.isa();
        for k in 0..UNROLL as u8 {
            bload(b, 4, DATA + 4 * k, VOFF, SRD_SRC, 4096 * u32::from(k))?;
        }
        for k in 0..UNROLL as u8 {
            bstore_b128(b, DATA + 4 * k, VOFF, SRD_OUT, 4096 * u32::from(k))?;
        }
        add64_imm(b, SRD_SRC, 4096 * UNROLL)?;
        add64_imm(b, SRD_OUT, 4096 * UNROLL)?;
        sop(b, format!("{} s{TRIPS}, s{TRIPS}, -1", super::common::s_add_i32(b.spec.arch)), &[TRIPS], &[TRIPS])?;
        let more = cmp_wg(wg, format!("s_cmp_gt_i32 s{TRIPS}, 0"), &[s(TRIPS)])?;
        Ok(((), more))
    })?;
    wg.isa().wait_all()
}

/// Shape contract of the emitted kernel (the `--shape` input of
/// `hipfire-isa emit --co`): exact counts of the memory and control mnemonics
/// the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in ["s_load_b32", "s_load_b128", "s_load_b256", "buffer_load_b128", "buffer_store_b128", "s_barrier_signal", "s_barrier_wait", "s_endpgm"] {
        let n = e.s_text.lines().filter(|l| l.trim().split_whitespace().next() == Some(name)).count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
