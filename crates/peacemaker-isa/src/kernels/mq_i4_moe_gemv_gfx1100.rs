// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Consumer24 B2 K1: the R5 exact F4 read-once C8 lane-union decode GEMV over pointer-addressed
//! expert tensors (`kernels::mq_i4_moe_gemv`, gfx1201), ported to gfx1100 (RDNA3, wave32) with the
//! checked ISA builder. Four symbols `mq_i4_moe_gemv_{mq2i4,mq4sym}_{r256_gate_up,r128_down}_pm_gfx1100`
//! in the module `mq_i4_moe_gemv_pm_gfx1100`. The gfx1201 emitter is untouched and remains the
//! authority for every arithmetic step; this file changes only what the architecture forces and
//! swaps the `Mq3I4` codec for `Mq2I4`.
//!
//! This module is not registered in `kernels`; it is a standalone emitter (see
//! `examples/mq_i4_moe_gemv_gfx1100_emit.rs`, which includes it with `#[path]`). It names its
//! builder dependencies through `crate::` / `super::`, so a facade module tree that re-exports
//! them (the example) and the library crate itself (a later registry entry) both compile it.
//!
//! # Frozen 96-byte kernarg ABI (R5, DESIGN §4.2 plus the trailing `n_experts` and a reserved zero word, all four symbols)
//! u64 pointers `expert_weight_ptrs@0` (table indexed by expert id; each entry is the projection
//! tensor's code-plane base of that expert, never `expert_id * stride`), `expert_ids@8`
//! (`u32[U]`, ascending), `expert_offsets@16` (`u32[U+1]`), `slot_ids@24` (`u32[10*C]`, original
//! slots `lane*10+rank`), `Xrot@32` (F32 rotated input), `Y0@40`, `Y1@48`; u32 `M@56`, `K@60`,
//! `U@64`, `C@68`, `x_row_div@72`, `x_src_rows@76`, `x_row_stride@80`, `y_slot_stride@84`,
//! `n_experts@88` (the extent of the `expert_weight_ptrs` table, a nonnegative signed-i32 count) and
//! `reserved@92` (zero, pads the block to 8 bytes).
//! Strides are F32 elements. The ABI, the slot semantics (`n = offsets[u+1]-offsets[u]`, `1..=8`,
//! slot `s` reads X row `s / x_row_div` through a raw buffer descriptor of
//! `x_src_rows * x_row_stride * 4` bytes and writes its F32 result at original slot `s`), the Y0/Y1
//! gate split at `M/2`, the early exits and the domain are exactly those of the gfx1201 emitter
//! (`mq_i4_moe_gemv.rs`); no route weight is applied and there are no atomics. A workgroup whose expert id is
//! `>= n_experts` (one unsigned compare, so a negative id too, made before `expert_id * 8` and the table lookup)
//! returns without a load or a store; `n_experts` is read into `s4`, which no other instruction of the entry
//! touches, so every other register assignment is unchanged.
//! Launch: grid `[M, U, 1]` (x = output row, y = distinct expert), block `[32, 1, 1]`, one wave32
//! per workgroup, no LDS (the only cross-lane traffic is `ds_bpermute_b32`).
//!
//! # Host-mapped meaning
//! Every `expert_weight_ptrs[e]` is just a 64-bit address the kernel loads from with
//! `global_load_*`. It may point into VRAM or at a device-visible alias of pinned host memory (a
//! host-mapped expert spilled over PCIe); the kernel cannot tell them apart and has no residency
//! flag. The tensor bytes are read exactly once per (row, expert) with plain global loads; there
//! is no staging, no atomics and no ordering dependence on the memory's location. The X rows and
//! the slot tables are always device memory in practice but are likewise only addresses.
//!
//! # x4 caveat
//! Every speed or bandwidth figure taken from this kernel on the hipx XTX carries this caveat:
//! that card sits behind a **PCIe 4.0 x4 root port (`00:03.1`)**, so host-mapped reads are
//! link-limited and raw kernel timings are not an end-to-end decode-throughput claim. Nothing in
//! the emitted code depends on the link width.
//!
//! # Exactness (unchanged from R5)
//! Each (slot, output row) value is the frozen F4 tree. Lane `l` owns the contiguous elements
//! below; per group `Mq2I4`: `p = RN(w0*x0)` (`v_mul_f32`) then left-associated
//! `p = fma(w_j, x_j, p)` (`v_fmac_f32`); `Mq4Sym`: see below. The group partial is added
//! separately into the chain accumulator; accumulators start at `+0`; lane value gate
//! `RN(RN(a0+a1)+RN(a2+a3))` over four chains `acc[g&3]`, down the single chain; then the
//! snapshot shuffle-down at offsets 16, 8, 4, 2, 1 with out-of-range lanes unchanged
//! (`v_add_f32` into scratch, `v_cndmask_b32` mask select), and lane 0 stores.
//!  * `Mq2I4` (planar qt133-class 2-bit tensors, the W2 arithmetic of `mq_i4_rows`): codes are 2
//!    bits, LSB first (code `j` at bits `[2j, 2j+2)`), one f16 scale per 128-column row group,
//!    levels `L = 3`, `w_i = RN(float(d) * float(2c-3))`. A row is `K/4` bytes, the code plane is
//!    `M*K/4` bytes, the scale plane starts at `base + M*K/4`, and the f16 of row `r` group `g`
//!    is at `2*(r*K/128+g)` inside it. A 128-column group is 32 code bytes; lane `l` owns the
//!    four contiguous codes `g*128 + 4l ..+3`, which are exactly byte `l` of the group (one
//!    `global_load_u8`, codes `v_bfe_u32(byte, 2i, 2)`).
//!  * `Mq4Sym` (pointer-addressed MQ4, the existing MQ4 arithmetic and the production
//!    contraction): `w = fma(d, float(q), z)` with the header's f16 `d` and `z` (`z = -8*d` for
//!    symmetric tensors; never an odd grid), `q` the unsigned nibble, element `j` the
//!    low-nibble-first nibble `j`. Gate (legacy G256 group, 136 B: f16 `d,z` for columns 0..127 at
//!    `+0`, 128..255 at `+4`, payload at `+8`): lane `l` owns the eight elements
//!    `g*256 + 8l ..+7`, header half `l >= 16`, chain `acc[g&3]`, terms `p = RN(w1*x1)`,
//!    `p = fma(w0,x0,p)`, then `fma(w_j,x_j,p)` for `j = 2..7`. Down (G128 group, 68 B: f16 `d,z`
//!    at `+0`, payload at `+4`, row stride `K/128*68`): lane `l` owns `g*128 + 4l ..+3`, the u16
//!    at payload byte `2l`, one chain, the same order over four terms.
//!
//! # What differs from gfx1201 (and only that)
//!  * Workgroup ids are **system SGPRs**, not `ttmp`: with `user_sgpr_count = 2` (kernarg pointer)
//!    and `system_sgpr_workgroup_id_y` set, x is `s2` and y is `s3` (`Arch::workgroup_ids`). Both
//!    are declared in the register plan; `s3` is the whole y id (gfx1201's `ttmp7[15:0]` mask does
//!    not exist here).
//!  * Scalar add/sub spellings are the gfx11 ones (`s_add_u32`/`s_addc_u32`/`s_add_i32`/`s_sub_i32`),
//!    SMEM zero offset is `null`, and the raw buffer load soffset is `0`; all come from
//!    `kernels::common`, which is arch-switched.
//!  * Memory waits are gfx11 `s_waitcnt vmcnt/lgkmcnt` and the gfx1100 `VALUTransUseHazard`
//!    `s_waitcnt_depctr depctr_va_vdst(0)` guard (the `v_rcp_iflag_f32` of the slot-index divide
//!    is the only transcendental). The builder inserts both from its wait ledger and hazard model
//!    and records them in the proof; this emitter never hand-places a wait. They are required by
//!    the architecture, so the shape contract must not forbid them (it counts them instead).
//!  * Raw buffer descriptor word 3 is `common::SRD_WORD3` (GPU-proven on gfx1100).
//!  * `Mq3I4` and the optimized packet network are not ported: `Mq2I4` replaces `Mq3I4`
//!    (a one-byte packet, no shift, no second load).
//!
//! Read-once C8 structure is unchanged: weights are loaded and dequantized once per group, then
//! the up-to-eight slots run their own chains on the shared weights, skipped by scalar branches
//! (`i >= n`), never by re-reading weights.

use crate::arch::WgIdSource;
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    add64, bload, gather_offset, lit, op, s, s_add_i32, smem, sop, sr, srd_tail, v, GatherTemps,
};
use crate::plan::Access;
use crate::reg::{Kind as RKind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size.
pub const KERNARG_BYTES: u32 = 96;
/// Workgroup size (threads): one wave32.
pub const WORKGROUP_SIZE: u32 = 32;
/// Most distinct original slots one expert workgroup serves (C8: at most one expert row per lane).
pub const MAX_SLOTS: u32 = 8;
/// Planar scale / activation group width in columns.
pub const GROUP: u32 = 128;
/// User SGPRs (the kernarg segment pointer); the system workgroup-id SGPRs follow them.
pub const USER_SGPRS: u8 = 2;

/// Weight codec of the tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// Planar 2-bit codes, `2c - 3`, one f16 scale per row-group, plane offset `M*K/4`.
    Mq2I4,
    /// Pointer-addressed MQ4 (qt44 gate group 136 B / qt53 down group 68 B), `d*q + z`.
    Mq4Sym,
}
impl Codec {
    pub const ALL: [Codec; 2] = [Codec::Mq2I4, Codec::Mq4Sym];
    pub fn tag(self) -> &'static str {
        match self {
            Self::Mq2I4 => "mq2i4",
            Self::Mq4Sym => "mq4sym",
        }
    }
}

/// Projection kind; selects the reduction tree (the profile's rotation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Gate/up (`K2560`, rotation 256): four group chains `acc[g&3]`, fold `(a0+a1)+(a2+a3)`, Y0/Y1 split at `M/2`.
    GateUp,
    /// Down (`K640`, rotation 128): one chain, Y0 only.
    Down,
}
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    pub fn tag(self) -> &'static str {
        match self {
            Self::GateUp => "r256_gate_up",
            Self::Down => "r128_down",
        }
    }
    /// Accumulator chains per slot.
    pub fn chains(self) -> u8 {
        match self {
            Self::GateUp => 4,
            Self::Down => 1,
        }
    }
}

/// One entry: `(arch, codec, kind)`; gfx1100 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub codec: Codec,
    pub kind: Kind,
}
impl Spec {
    /// The exported symbol, `mq_i4_moe_gemv_<codec>_<kind>_pm_<arch>`.
    pub fn symbol(&self) -> String {
        format!("mq_i4_moe_gemv_{}_pm_{}", self.tag(), self.arch.name())
    }
    /// `<codec>_<kind>`: the label prefix, the kernel variant and the contract file tag.
    pub fn tag(&self) -> String {
        format!("{}_{}", self.codec.tag(), self.kind.tag())
    }
    pub fn module(arch: Arch) -> String {
        format!("mq_i4_moe_gemv_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1100 {
            return Err("mq_i4_moe_gemv_gfx1100 builder is exact-gfx1100 only".into());
        }
        Ok(())
    }
    /// Weights (and X elements) one lane owns per group: eight for the `Mq4Sym` gate G256 group, four otherwise.
    pub fn lane_elems(&self) -> u8 {
        if self.codec == Codec::Mq4Sym && self.kind == Kind::GateUp {
            8
        } else {
            4
        }
    }
    /// Group instances in the emitted body: gate four loop groups plus three tail groups, down one.
    pub fn group_instances(&self) -> u32 {
        match self.kind {
            Kind::GateUp => 7,
            Kind::Down => 1,
        }
    }
}

/// Every entry of one architecture in module order: Mq2I4 gate, Mq2I4 down, Mq4Sym gate, Mq4Sym down.
pub fn module_specs(arch: Arch) -> Vec<Spec> {
    Codec::ALL
        .into_iter()
        .flat_map(|codec| {
            Kind::ALL
                .into_iter()
                .map(move |kind| Spec { arch, codec, kind })
        })
        .collect()
}

// ----------------------------------------------------------------------- SGPRs
// s0:1 kernarg pointer; s2 / s3 workgroup id x / y (system SGPRs after the two user SGPRs).
const KARG: u8 = 0;
const WGX: u8 = USER_SGPRS;
const WGY: u8 = USER_SGPRS + 1;
/// Kernargs (s_load_b256 @0, b256 @0x20, b128 @0x40, b64 @0x50).
const WPTRS: u8 = 8;
const EIDS: u8 = 10;
const OFFS: u8 = 12;
const SLOTS: u8 = 14;
const XROT: u8 = 16;
const Y0: u8 = 18;
const Y1: u8 = 20;
const SM: u8 = 22;
const SK: u8 = 23;
const SU: u8 = 24;
const SC: u8 = 25;
const SDIV: u8 = 26;
const SXSRC: u8 = 27;
const SXSTR: u8 = 28;
const SYSTR: u8 = 29;
const ROW: u8 = 30;
const UIDX: u8 = 31;
const OFF0: u8 = 32;
const OFF1: u8 = 33;
const EID: u8 = 34;
const NS: u8 = 35;
const TPTR: u8 = 36;
const WPTR: u8 = 38;
const SRDX: u8 = 40;
const ROW_PTR: u8 = 44;
const SCALE_PTR: u8 = 46;
const GROUPS: u8 = 48;
const ROW_BYTES: u8 = 49;
const CODES_BYTES: u8 = 50;
const TMP: u8 = 51;
const TMP2: u8 = 52;
const GI: u8 = 53;
const QUADS: u8 = 54;
const G_CODE: u8 = 55;
const G_X: u8 = 56;
const G_SCALE: u8 = 57;
const YSEL: u8 = 58;
const ROWIN4: u8 = 60;
const SXS4: u8 = 61;
const SYS4: u8 = 62;
const NSM1: u8 = 63;
const LIVE: u8 = 64;
const MASK: u8 = 66;
const SHM: u8 = 68;
const TL: u8 = 73;
const MI: u8 = 77;
const ROWM: u8 = 78;
const ROWIN: u8 = 79;
/// SGPRs the plan declares (the honest ceiling, `sgpr_max` of the contract).
const SGPRS: u8 = 80;
/// `n_experts@88`, the extent of the `expert_weight_ptrs` table: `s4`, the first SGPR after the system workgroup ids,
/// which no other instruction of the entry touches (the plan's SGPR ceiling and the physical assignment of every
/// other register are unchanged). An `expert_ids` entry at or past it is never used to index the table.
const SNEXP: u8 = 4;

// ----------------------------------------------------------------------- VGPRs
const LANE: u8 = 0;

/// VGPR layout of one entry, assigned sequentially so no range can overlap.
#[derive(Clone, Copy, Debug)]
struct Lay {
    /// Weights / X elements per lane per group, accumulator chains per slot.
    nw: u8,
    chains: u8,
    caddr: u8,
    saddr: u8,
    xl: u8,
    cl: u8,
    hl: u8,
    xb: u8,
    xa: u8,
    pkt: u8,
    d: u8,
    z: u8,
    dot: u8,
    exch: u8,
    shuf: u8,
    w: u8,
    x: u8,
    acc: u8,
    xo: u8,
    ya: u8,
    end: u8,
}
impl Lay {
    fn new(spec: Spec) -> Self {
        let (nw, chains) = (spec.lane_elems(), spec.kind.chains());
        let mut n = 1u8;
        let mut take = |len: u8, align: u8| {
            n = n.div_ceil(align) * align;
            let base = n;
            n += len;
            base
        };
        let caddr = take(1, 1);
        let saddr = take(1, 1);
        let xl = take(1, 1);
        let cl = take(1, 1);
        let hl = take(1, 1);
        let xb = take(1, 1);
        let xa = take(1, 1);
        let pkt = take(1, 1);
        let d = take(1, 1);
        let z = take(1, 1);
        let dot = take(2, 1);
        let exch = take(1, 1);
        let shuf = take(5, 1);
        let w = take(nw, 4);
        let x = take(MAX_SLOTS as u8 * nw, 4);
        let acc = take(MAX_SLOTS as u8 * chains, 1);
        let xo = take(MAX_SLOTS as u8, 1);
        let ya = take(MAX_SLOTS as u8, 1);
        let end = n.div_ceil(8) * 8;
        Self {
            nw,
            chains,
            caddr,
            saddr,
            xl,
            cl,
            hl,
            xb,
            xa,
            pkt,
            d,
            z,
            dot,
            exch,
            shuf,
            w,
            x,
            acc,
            xo,
            ya,
            end,
        }
    }
    /// First X register of slot `i` (`nw` consecutive registers).
    fn xr(&self, i: u32) -> u8 {
        self.x + i as u8 * self.nw
    }
    /// Accumulator register of slot `i`, chain `c`.
    fn ac(&self, i: u32, c: u8) -> u8 {
        self.acc + i as u8 * self.chains + c
    }
}

fn lbl(spec: Spec, name: &str) -> String {
    format!(".Lmqi4moe_{}_{name}", spec.tag())
}

/// Declare `len` registers from `base` as the legal ranges the plan allows (widths 1, 2, 4 or 8, SGPR
/// ranges aligned to their width): greedy largest-first granules of the same name, so every multi-register
/// access (kernarg loads, pointer pairs, 128-bit X loads) lies inside one declared range.
fn declare(regs: &mut RegPlan, name: &str, kind: RKind, base: u8, len: u8) -> R {
    let (mut at, end) = (u16::from(base), u16::from(base) + u16::from(len));
    while at < end {
        let width = [8u16, 4, 2, 1]
            .into_iter()
            .find(|&w| at + w <= end && (kind == RKind::V || at % w == 0))
            .unwrap_or(1);
        regs.add_range(name, kind, at as u8, width as u8, Live::Whole)?;
        at += width;
    }
    Ok(())
}
fn plan(l: &Lay) -> Result<RegPlan, String> {
    let mut regs = RegPlan::new(u16::from(l.end), u16::from(SGPRS))?;
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("workgroup_id_x", WGX, 1),
        ("workgroup_id_y", WGY, 1),
        ("kernel_arguments_ptrs", WPTRS, 8),
        ("kernel_arguments_x_y_m_k", XROT, 8),
        ("kernel_arguments_u_c_div_src", SU, 4),
        ("kernel_arguments_strides", SXSTR, 2),
        ("kernel_arguments_n_experts", SNEXP, 1),
        ("row", ROW, 1),
        ("expert_index", UIDX, 1),
        ("expert_offsets_pair", OFF0, 2),
        ("expert_id", EID, 1),
        ("slot_count", NS, 1),
        ("table_pointer", TPTR, 2),
        ("expert_pointer", WPTR, 2),
        ("x_descriptor", SRDX, 4),
        ("row_pointer", ROW_PTR, 2),
        ("scale_pointer", SCALE_PTR, 2),
        ("groups", GROUPS, 1),
        ("row_bytes", ROW_BYTES, 1),
        ("codes_bytes", CODES_BYTES, 1),
        ("temp", TMP, 1),
        ("temp2", TMP2, 1),
        ("group_index", GI, 1),
        ("quad_end", QUADS, 1),
        ("group_code_offset", G_CODE, 1),
        ("group_x_offset", G_X, 1),
        ("group_scale_offset", G_SCALE, 1),
        ("y_select", YSEL, 2),
        ("row_in_y_bytes", ROWIN4, 1),
        ("x_stride_bytes", SXS4, 1),
        ("y_stride_bytes", SYS4, 1),
        ("slots_minus_one", NSM1, 1),
        ("gather_live", LIVE, 2),
        ("gather_mask", MASK, 2),
        ("shuffle_masks", SHM, 5),
        ("lane_temps", TL, 4),
        ("half_rows", MI, 1),
        ("row_minus_half", ROWM, 1),
        ("row_in_y", ROWIN, 1),
    ] {
        declare(&mut regs, name, RKind::S, base, len)?;
    }
    for (name, base, len) in [
        ("lane", LANE, 1),
        ("code_address", l.caddr, 1),
        ("scale_address", l.saddr, 1),
        ("x_lane_offset", l.xl, 1),
        ("code_lane_offset", l.cl, 1),
        ("header_lane_offset", l.hl, 1),
        ("x_base", l.xb, 1),
        ("x_address", l.xa, 1),
        ("packet", l.pkt, 1),
        ("scale", l.d, 1),
        ("zero_point", l.z, 1),
        ("dots", l.dot, 2),
        ("exchange", l.exch, 1),
        ("shuffle_addresses", l.shuf, 5),
        ("weights", l.w, l.nw),
        ("x_values", l.x, MAX_SLOTS as u8 * l.nw),
        ("accumulators", l.acc, MAX_SLOTS as u8 * l.chains),
        ("slot_x_offsets", l.xo, MAX_SLOTS as u8),
        ("slot_y_addresses", l.ya, MAX_SLOTS as u8),
    ] {
        declare(&mut regs, name, RKind::V, base, len)?;
    }
    let top = l.xo.max(l.ya) + MAX_SLOTS as u8;
    if l.end > top {
        declare(&mut regs, "granule_pad", RKind::V, top, l.end - top)?;
    }
    Ok(regs)
}

/// Emit one entry: a complete single-kernel `.s` with its builder proof.
pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let l = Lay::new(spec);
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("expert_weight_ptrs", 0, Access::ReadOnly)
        .pointer_access("expert_ids", 8, Access::ReadOnly)
        .pointer_access("expert_offsets", 16, Access::ReadOnly)
        .pointer_access("slot_ids", 24, Access::ReadOnly)
        .pointer_access("Xrot", 32, Access::ReadOnly)
        .pointer_access("Y0", 40, Access::WriteOnly)
        .pointer_access("Y1", 48, Access::WriteOnly)
        .hidden("M", 56, 4, "by_value")
        .hidden("K", 60, 4, "by_value")
        .hidden("U", 64, 4, "by_value")
        .hidden("C", 68, 4, "by_value")
        .hidden("x_row_div", 72, 4, "by_value")
        .hidden("x_src_rows", 76, 4, "by_value")
        .hidden("x_row_stride", 80, 4, "by_value")
        .hidden("y_slot_stride", 84, 4, "by_value")
        .hidden("n_experts", 88, 4, "by_value")
        .hidden("reserved", 92, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq_i4_moe_gemv".into(),
        variant: spec.tag(),
        arch: spec.arch,
        symbol: spec.symbol(),
        kernargs,
        user_sgpr_count: USER_SGPRS,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: WORKGROUP_SIZE as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&l)?);
    b.enable_delay_alu();
    prologue(&mut b, spec, &l)?;
    b.wait_all()?;
    match spec.kind {
        Kind::GateUp => {
            // Groups strictly ascending; group g feeds chain g & 3: whole quads in the
            // loop, then up to three tail groups on chains 0, 1, 2.
            sop(
                &mut b,
                format!("s_and_b32 s{QUADS}, s{GROUPS}, -4"),
                &[QUADS],
                &[GROUPS],
            )?;
            sop(&mut b, format!("s_cmp_eq_u32 s{QUADS}, 0"), &[], &[QUADS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "tail")))?;
            b.loop_(&lbl(spec, "quad"), |b| {
                for chain in 0..4 {
                    group(b, spec, &l, chain, &format!("q{chain}"))?;
                }
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{QUADS}"),
                    &[],
                    &[GI, QUADS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "quad")))
            })?;
            b.label(&lbl(spec, "tail"))?;
            for chain in 0..3 {
                sop(
                    &mut b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(&mut b, &format!("s_cbranch_scc0 {}", lbl(spec, "reduce")))?;
                group(&mut b, spec, &l, chain, &format!("t{chain}"))?;
            }
        }
        Kind::Down => {
            sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
            branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "reduce")))?;
            b.loop_(&lbl(spec, "group"), |b| {
                group(b, spec, &l, 0, "g")?;
                sop(
                    b,
                    format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                    &[],
                    &[GI, GROUPS],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "group")))
            })?;
        }
    }
    b.label(&lbl(spec, "reduce"))?;
    epilogue(&mut b, spec, &l)?;
    b.finish()
}

/// The four entries as one code object source (`Spec::module`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = module_specs(arch)
        .into_iter()
        .map(emit)
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// Kernarg loads, the guards, the expert's pointer and the row/slot bookkeeping, per-lane
/// address constants, zeroed accumulators and the per-slot X offsets / Y addresses.
fn prologue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    let a = b.spec.arch;
    let exit = lbl(spec, "exit");
    smem(b, WPTRS, 8, KARG, 0)?;
    smem(b, XROT, 8, KARG, 0x20)?;
    smem(b, SU, 4, KARG, 0x40)?;
    smem(b, SXSTR, 2, KARG, 0x50)?;
    smem(b, SNEXP, 1, KARG, 0x58)?;
    // gfx11 workgroup ids are system SGPRs (after the user SGPRs), not ttmp registers.
    let ids = a.workgroup_ids(USER_SGPRS)?;
    if (ids.x, ids.y) != (WgIdSource::Sgpr(WGX), WgIdSource::Sgpr(WGY)) {
        return Err(format!("unexpected workgroup id ABI {ids:?}"));
    }
    sop(b, format!("s_mov_b32 s{ROW}, s{WGX}"), &[ROW], &[WGX])?;
    sop(b, format!("s_mov_b32 s{UIDX}, s{WGY}"), &[UIDX], &[WGY])?;
    sop(b, format!("s_cmp_ge_u32 s{ROW}, s{SM}"), &[], &[ROW, SM])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_ge_u32 s{UIDX}, s{SU}"), &[], &[UIDX, SU])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // offsets[u], offsets[u+1] and the expert id; the expert's tensor pointer from the table.
    sop(b, format!("s_lshl_b32 s{TMP}, s{UIDX}, 2"), &[TMP], &[UIDX])?;
    add64(b, TPTR, OFFS, TMP)?;
    smem(b, OFF0, 2, TPTR, 0)?;
    add64(b, TPTR, EIDS, TMP)?;
    smem(b, EID, 1, TPTR, 0)?;
    sop(
        b,
        format!("s_sub_i32 s{NS}, s{OFF1}, s{OFF0}"),
        &[NS],
        &[OFF1, OFF0],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{NS}, 0"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_cmp_gt_u32 s{NS}, {MAX_SLOTS}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_mul_i32 s{TMP}, s{SC}, 10"), &[TMP], &[SC])?;
    sop(
        b,
        format!("s_cmp_gt_u32 s{OFF1}, s{TMP}"),
        &[],
        &[OFF1, TMP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    // Expert id range check before the table index (and before the expert pointer lookup): an id at or past
    // `n_experts` (unsigned, so a negative id too) exits without a table load or a store.
    sop(
        b,
        format!("s_cmp_ge_u32 s{EID}, s{SNEXP}"),
        &[],
        &[EID, SNEXP],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{EID}, 3"), &[TMP], &[EID])?;
    add64(b, TPTR, WPTRS, TMP)?;
    smem(b, WPTR, 2, TPTR, 0)?;
    // Group count, row bytes and the row's code (and, planar, scale) pointers.
    match (spec.codec, spec.kind) {
        (Codec::Mq2I4, _) => {
            // Row bytes K*2/8 = K/4; the code plane is M*K/4 bytes and the scale plane follows it.
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, 7"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_lshr_b32 s{ROW_BYTES}, s{SK}, 2"),
                &[ROW_BYTES],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{CODES_BYTES}, s{SM}, s{ROW_BYTES}"),
                &[CODES_BYTES],
                &[SM, ROW_BYTES],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
                &[TMP],
                &[ROW, ROW_BYTES],
            )?;
            add64(b, ROW_PTR, WPTR, TMP)?;
            sop(
                b,
                format!("s_mul_i32 s{TMP2}, s{ROW}, s{GROUPS}"),
                &[TMP2],
                &[ROW, GROUPS],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{TMP2}, s{TMP2}, 1"),
                &[TMP2],
                &[TMP2],
            )?;
            sop(
                b,
                format!("{} s{TMP2}, s{TMP2}, s{CODES_BYTES}", s_add_i32(a)),
                &[TMP2],
                &[TMP2, CODES_BYTES],
            )?;
            add64(b, SCALE_PTR, WPTR, TMP2)?;
        }
        (Codec::Mq4Sym, kind) => {
            let (shift, stride) = if kind == Kind::GateUp {
                (8, 136)
            } else {
                (7, 68)
            };
            sop(
                b,
                format!("s_lshr_b32 s{GROUPS}, s{SK}, {shift}"),
                &[GROUPS],
                &[SK],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{ROW_BYTES}, s{GROUPS}, {}", lit(stride)),
                &[ROW_BYTES],
                &[GROUPS],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{ROW}, s{ROW_BYTES}"),
                &[TMP],
                &[ROW, ROW_BYTES],
            )?;
            add64(b, ROW_PTR, WPTR, TMP)?;
        }
    }
    // Y pointer and the row's byte offset inside a slot's output row.
    match spec.kind {
        Kind::GateUp => {
            sop(b, format!("s_lshr_b32 s{MI}, s{SM}, 1"), &[MI], &[SM])?;
            sop(
                b,
                format!("s_sub_i32 s{ROWM}, s{ROW}, s{MI}"),
                &[ROWM],
                &[ROW, MI],
            )?;
            sop(b, format!("s_cmp_lt_u32 s{ROW}, s{MI}"), &[], &[ROW, MI])?;
            op(
                b,
                format!(
                    "s_cselect_b64 s[{YSEL}:{}], s[{Y0}:{}], s[{Y1}:{}]",
                    YSEL + 1,
                    Y0 + 1,
                    Y1 + 1
                ),
                &[sr(YSEL, 2)],
                &[sr(Y0, 2), sr(Y1, 2)],
            )?;
            sop(
                b,
                format!("s_cselect_b32 s{ROWIN}, s{ROW}, s{ROWM}"),
                &[ROWIN],
                &[ROW, ROWM],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROWIN}, 2"),
                &[ROWIN4],
                &[ROWIN],
            )?;
        }
        Kind::Down => {
            sop(b, format!("s_mov_b32 s{YSEL}, s{Y0}"), &[YSEL], &[Y0])?;
            sop(
                b,
                format!("s_mov_b32 s{}, s{}", YSEL + 1, Y0 + 1),
                &[YSEL + 1],
                &[Y0 + 1],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{ROWIN4}, s{ROW}, 2"),
                &[ROWIN4],
                &[ROW],
            )?;
        }
    }
    sop(
        b,
        format!("s_lshl_b32 s{SXS4}, s{SXSTR}, 2"),
        &[SXS4],
        &[SXSTR],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SYS4}, s{SYSTR}, 2"),
        &[SYS4],
        &[SYSTR],
    )?;
    sop(
        b,
        format!("{} s{NSM1}, s{NS}, -1", s_add_i32(a)),
        &[NSM1],
        &[NS],
    )?;
    // X descriptor: base Xrot, x_src_rows*x_row_stride*4 records (rows past it read 0).
    sop(b, format!("s_mov_b32 s{SRDX}, s{XROT}"), &[SRDX], &[XROT])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRDX + 1, XROT + 1),
        &[SRDX + 1],
        &[XROT + 1],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{SXSRC}, s{SXS4}"),
        &[TMP],
        &[SXSRC, SXS4],
    )?;
    srd_tail(b, SRDX, Some(TMP))?;
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    // Per-lane constants: x byte offset within a group (16*l, or 32*l for the 8-element lane),
    // the code byte offset of the lane's packet within a group and its extras.
    let xshift = if l.nw == 8 { 5 } else { 4 };
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{}, {xshift}, v{LANE}", l.xl),
        &[l.xl],
        &[LANE],
        &[],
    )?;
    match (spec.codec, spec.kind) {
        // Mq2I4: the lane's four codes are group byte l.
        (Codec::Mq2I4, _) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, v{LANE}", l.cl),
            &[l.cl],
            &[LANE],
            &[],
        )?,
        (Codec::Mq4Sym, Kind::GateUp) => {
            // Payload word 4*l; header half (bytes 0 or 4) 4*(l >> 4).
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{}, 2, v{LANE}", l.cl),
                &[l.cl],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{}, 4, v{LANE}", l.hl),
                &[l.hl],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{0}, 2, v{0}", l.hl),
                &[l.hl],
                &[l.hl],
                &[],
            )?;
        }
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 1, v{LANE}", l.cl),
            &[l.cl],
            &[LANE],
            &[],
        )?,
    }
    for k in 0..MAX_SLOTS as u8 * l.chains {
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, 0", l.acc + k),
            &[l.acc + k],
            &[],
            &[],
        )?;
    }
    // Slot lanes: lane i < 8 loads slot position min(i, n-1) (never past the expert's slots),
    // derives the X-row byte offset (slot / x_row_div * x_row_stride * 4) and the Y byte
    // offset (slot * y_slot_stride * 4 + row-in-Y * 4); each is then broadcast into per-slot VGPRs.
    let (sid, xq, t0, yq, p) = (l.x, l.x + 1, l.x + 2, l.x + 6, l.x + 7);
    valu(
        b,
        &format!("v_min_u32_e32 v{p}, s{NSM1}, v{LANE}"),
        &[p],
        &[LANE],
        &[NSM1],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{p}, s{OFF0}, v{p}"),
        &[p],
        &[p],
        &[OFF0],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{p}, 2, v{p}"),
        &[p],
        &[p],
        &[],
    )?;
    gload(b, "global_load_b32", sid, p, SLOTS, "")?;
    gather_offset(
        b,
        xq,
        sid,
        Some(SDIV),
        1,
        LIVE,
        GatherTemps {
            v: [t0, t0 + 1, t0 + 2, t0 + 3],
            mask: MASK,
        },
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{xq}, v{xq}, s{SXS4}"),
        &[xq],
        &[xq],
        &[SXS4],
    )?;
    valu(
        b,
        &format!("v_mul_lo_u32 v{yq}, v{sid}, s{SYS4}"),
        &[yq],
        &[sid],
        &[SYS4],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{yq}, s{ROWIN4}, v{yq}"),
        &[yq],
        &[yq],
        &[ROWIN4],
    )?;
    for i in 0..MAX_SLOTS as u8 {
        let tl = TL + i % 4;
        op(
            b,
            format!("v_readlane_b32 s{tl}, v{xq}, {i}"),
            &[s(tl)],
            &[v(xq)],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{tl}", l.xo + i),
            &[l.xo + i],
            &[],
            &[tl],
        )?;
        op(
            b,
            format!("v_readlane_b32 s{tl}, v{yq}, {i}"),
            &[s(tl)],
            &[v(yq)],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{tl}", l.ya + i),
            &[l.ya + i],
            &[],
            &[tl],
        )?;
    }
    Ok(())
}

/// `if n <= i skip to target` (scalar, uniform): SCC = (n > i), branch when clear.
fn slot_gate(b: &mut Builder, i: u32, target: &str) -> R {
    sop(b, format!("s_cmp_gt_u32 s{NS}, {i}"), &[], &[NS])?;
    branch(b, &format!("s_cbranch_scc0 {target}"))
}

/// One group of the row, loaded and dequantized once, then applied to every live slot.
/// Chain `chain` (always 0 for down) takes the group partial of every slot.
fn group(b: &mut Builder, spec: Spec, l: &Lay, chain: u8, inst: &str) -> R {
    let arch = b.spec.arch;
    let wide = l.nw == 8;
    // Scalar group offsets, then the lane addresses, then the weight loads (all before
    // any slot branch: slot branches never re-read weights).
    match spec.codec {
        Codec::Mq2I4 => {
            // 128 columns * 2 bits = 32 code bytes per group; 512 X bytes; one f16 scale.
            sop(
                b,
                format!("s_mul_i32 s{G_CODE}, s{GI}, 32"),
                &[G_CODE],
                &[GI],
            )?;
            sop(b, format!("s_lshl_b32 s{G_X}, s{GI}, 9"), &[G_X], &[GI])?;
            sop(
                b,
                format!("s_lshl_b32 s{G_SCALE}, s{GI}, 1"),
                &[G_SCALE],
                &[GI],
            )?;
        }
        Codec::Mq4Sym => {
            sop(
                b,
                format!(
                    "s_mul_i32 s{G_CODE}, s{GI}, {}",
                    lit(if wide { 136 } else { 68 })
                ),
                &[G_CODE],
                &[GI],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{G_X}, s{GI}, {}", if wide { 10 } else { 9 }),
                &[G_X],
                &[GI],
            )?;
        }
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", l.caddr, l.cl),
        &[l.caddr],
        &[l.cl],
        &[G_CODE],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq2I4, _) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_SCALE}", l.saddr),
            &[l.saddr],
            &[],
            &[G_SCALE],
        )?,
        (Codec::Mq4Sym, Kind::GateUp) => valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, s{G_CODE}, v{}", l.saddr, l.hl),
            &[l.saddr],
            &[l.hl],
            &[G_CODE],
        )?,
        (Codec::Mq4Sym, Kind::Down) => valu(
            b,
            &format!("v_mov_b32_e32 v{}, s{G_CODE}", l.saddr),
            &[l.saddr],
            &[],
            &[G_CODE],
        )?,
    }
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{}, s{G_X}, v{}", l.xb, l.xl),
        &[l.xb],
        &[l.xl],
        &[G_X],
    )?;
    match (spec.codec, spec.kind) {
        (Codec::Mq2I4, _) => {
            gload(b, "global_load_u8", l.pkt, l.caddr, ROW_PTR, "")?;
            gload(b, "global_load_u16", l.d, l.saddr, SCALE_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::GateUp) => {
            gload(b, "global_load_b32", l.pkt, l.caddr, ROW_PTR, " offset:8")?;
            gload(b, "global_load_b32", l.z, l.saddr, ROW_PTR, "")?;
        }
        (Codec::Mq4Sym, Kind::Down) => {
            gload(b, "global_load_u16", l.pkt, l.caddr, ROW_PTR, " offset:4")?;
            gload(b, "global_load_b32", l.z, l.saddr, ROW_PTR, "")?;
        }
    }
    // X loads of the live slots (slot 0 always: n >= 1), each slot's own X row.
    let xdone = lbl(spec, &format!("{inst}_xdone"));
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &xdone)?
        }
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{}, v{}, v{}", l.xa, l.xo + i as u8, l.xb),
            &[l.xa],
            &[l.xo + i as u8, l.xb],
            &[],
        )?;
        bload(b, 4, l.xr(i), l.xa, SRDX, 0)?;
        if wide {
            bload(b, 4, l.xr(i) + 4, l.xa, SRDX, 16)?
        }
    }
    b.label(&xdone)?;
    b.wait_all()?;
    unpack(b, spec, l)?;
    // Per-slot chains on the shared weights.
    let done = lbl(spec, &format!("{inst}_done"));
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &done)?
        }
        let dot = l.dot + (i % 2) as u8;
        let x = l.xr(i);
        // Mq2I4: p = RN(w0*x0), then fma(w_j, x_j, p) for j = 1..; Mq4Sym (the production contraction of the
        // old MQ4 kernels): p = RN(w1*x1), then fma(w0, x0, p), then fma(w_j, x_j, p) for j = 2.. ascending.
        let first = if spec.codec == Codec::Mq4Sym { 1 } else { 0 };
        valu(
            b,
            &format!("v_mul_f32_e32 v{dot}, v{}, v{}", l.w + first, x + first),
            &[dot],
            &[l.w + first, x + first],
            &[],
        )?;
        if first == 1 {
            valu(
                b,
                &format!("v_fmac_f32_e32 v{dot}, v{}, v{x}", l.w),
                &[dot],
                &[dot, l.w, x],
                &[],
            )?
        }
        for j in 2 - u8::from(first == 0)..l.nw {
            valu(
                b,
                &format!("v_fmac_f32_e32 v{dot}, v{}, v{}", l.w + j, x + j),
                &[dot],
                &[dot, l.w + j, x + j],
                &[],
            )?;
        }
        let acc = l.ac(i, chain);
        valu(
            b,
            &format!("v_add_f32_e32 v{acc}, v{acc}, v{dot}"),
            &[acc],
            &[acc, dot],
            &[],
        )?;
    }
    b.label(&done)?;
    sop(
        b,
        format!("{} s{GI}, s{GI}, 1", s_add_i32(arch)),
        &[GI],
        &[GI],
    )?;
    b.wait_all()
}

/// The exact unpack: the lane's packet to `nw` dequantized weights `l.w ..`.
///  * `Mq2I4`: `d = float(f16)`, per code `w = RN(d * float(2*bfe(byte, 2i, 2) - 3))`.
///  * `Mq4Sym`: `d`, `z` the header's f16 halves converted exactly, per nibble
///    `w = fma(d, float(bfe(packet, 4i, 4)), z)`.
fn unpack(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    match spec.codec {
        Codec::Mq2I4 => {
            valu(
                b,
                &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.d),
                &[l.d],
                &[l.d],
                &[],
            )?;
            for i in 0..4u8 {
                let w = l.w + i;
                valu(
                    b,
                    &format!("v_bfe_u32 v{w}, v{}, {}, 2", l.pkt, u32::from(i) * 2),
                    &[w],
                    &[l.pkt],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_lshlrev_b32_e32 v{w}, 1, v{w}"),
                    &[w],
                    &[w],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_add_nc_u32_e32 v{w}, -3, v{w}"),
                    &[w],
                    &[w],
                    &[],
                )?;
                valu(b, &format!("v_cvt_f32_i32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
                valu(
                    b,
                    &format!("v_mul_f32_e32 v{w}, v{}, v{w}", l.d),
                    &[w],
                    &[l.d, w],
                    &[],
                )?;
            }
        }
        Codec::Mq4Sym => {
            valu(
                b,
                &format!("v_cvt_f32_f16_e64 v{}, v{}.l", l.d, l.z),
                &[l.d],
                &[l.z],
                &[],
            )?;
            valu(
                b,
                &format!("v_lshrrev_b32_e32 v{0}, 16, v{0}", l.z),
                &[l.z],
                &[l.z],
                &[],
            )?;
            valu(
                b,
                &format!("v_cvt_f32_f16_e64 v{0}, v{0}.l", l.z),
                &[l.z],
                &[l.z],
                &[],
            )?;
            for i in 0..l.nw {
                let w = l.w + i;
                valu(
                    b,
                    &format!("v_bfe_u32 v{w}, v{}, {}, 4", l.pkt, u32::from(i) * 4),
                    &[w],
                    &[l.pkt],
                    &[],
                )?;
                valu(b, &format!("v_cvt_f32_u32_e32 v{w}, v{w}"), &[w], &[w], &[])?;
                valu(
                    b,
                    &format!("v_fma_f32 v{w}, v{}, v{w}, v{}", l.d, l.z),
                    &[w],
                    &[l.d, w, l.z],
                    &[],
                )?;
            }
        }
    }
    Ok(())
}

/// Shuffle tables, the per-slot fold + snapshot shuffle-down tree, then the lane-0 stores.
fn epilogue(b: &mut Builder, spec: Spec, l: &Lay) -> R {
    // Shuffle step k (offset 16 >> k): mask lane < 32 - off, address (lane + off or lane) * 4.
    let offsets = [16u32, 8, 4, 2, 1];
    for (k, off) in offsets.into_iter().enumerate() {
        let (m, sh) = (SHM + k as u8, l.shuf + k as u8);
        op(
            b,
            format!("v_cmp_gt_u32_e64 s{m}, {}, v{LANE}", 32 - off),
            &[s(m)],
            &[v(LANE)],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{sh}, 0, {off}, s{m}"),
            &[sh],
            &[],
            &[m],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{sh}, v{sh}, v{LANE}, 2"),
            &[sh],
            &[sh, LANE],
            &[],
        )?;
    }
    let red_done = lbl(spec, "red_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &red_done)?
        }
        let a0 = l.ac(i, 0);
        if spec.kind == Kind::GateUp {
            let (a1, a2, a3) = (l.ac(i, 1), l.ac(i, 2), l.ac(i, 3));
            valu(
                b,
                &format!("v_add_f32_e32 v{a0}, v{a0}, v{a1}"),
                &[a0],
                &[a0, a1],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{a2}, v{a2}, v{a3}"),
                &[a2],
                &[a2, a3],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_f32_e32 v{a0}, v{a0}, v{a2}"),
                &[a0],
                &[a0, a2],
                &[],
            )?;
        }
        for k in 0..5u8 {
            let (m, sh) = (SHM + k, l.shuf + k);
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{}, v{sh}, v{a0}", l.exch),
                    vec![vreg(l.exch, 1)],
                    vec![vreg(sh, 1), vreg(a0, 1)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            // sum = acc + acc[lane + off]; lanes with lane + off > 31 (mask clear) keep acc unchanged.
            valu(
                b,
                &format!("v_add_f32_e32 v{}, v{a0}, v{}", l.exch, l.exch),
                &[l.exch],
                &[a0, l.exch],
                &[],
            )?;
            valu(
                b,
                &format!("v_cndmask_b32_e64 v{a0}, v{a0}, v{}, s{m}", l.exch),
                &[a0],
                &[a0, l.exch],
                &[m],
            )?;
        }
    }
    b.label(&red_done)?;
    // Lane 0 stores each live slot's value at its original slot's row.
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    let st_done = lbl(spec, "st_done");
    for i in 0..MAX_SLOTS {
        if i > 0 {
            slot_gate(b, i, &st_done)?
        }
        let (ya, a0) = (l.ya + i as u8, l.ac(i, 0));
        b.push(
            Instruction::new(
                format!("global_store_b32 v{ya}, v{a0}, s[{YSEL}:{}]", YSEL + 1),
                vec![],
                vec![vreg(ya, 1), vreg(a0, 1), sreg(YSEL, 2)],
            )
            .memory(MemoryClass::VmemStore),
        )?;
    }
    b.label(&st_done)?;
    b.wait_all()?;
    b.label(&lbl(spec, "exit"))?;
    branch(b, "s_endpgm")
}

/// The shape contract of one emitted entry: counts of its characteristic instructions (including the
/// builder-inserted gfx11 `s_waitcnt`, `s_waitcnt_depctr` and `s_delay_alu`, so a certified object cannot
/// silently change its wait or hazard placement), the forbidden families, honest register ceilings of the
/// C8 slot state (accumulators, X registers and per-slot offsets/addresses are inside `vgpr_max`; the scalar
/// bookkeeping is `sgpr_max`), wave32, no spills, no private or LDS memory.
///
/// The forbidden list is the gfx12-only spelling set (a gfx11 object containing any of them is wrong) plus
/// the families this kernel never uses. `s_waitcnt*` is deliberately **not** forbidden: on gfx11 the
/// counters are `s_waitcnt vmcnt/lgkmcnt` and every load needs one. Committed (by the parent) as
/// `kernels/mq_i4_moe_gemv.gfx1100.<tag>.contract.json`.
pub fn contract(spec: Spec, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    serde_json::json!({
        "symbol": spec.symbol(), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "ds_load*", "ds_store*", "ds_swizzle*", "ds_read*", "ds_write*", "s_barrier*",
                      "v_writelane*", "v_wmma*", "v_swmmac*", "v_fma_mix*", "s_setpc*", "s_swappc*",
                      "s_wait_loadcnt*", "s_wait_dscnt*", "s_wait_storecnt*", "s_wait_kmcnt*", "s_wait_alu",
                      "s_add_co_*", "s_sub_co_*"],
        "vgpr_max": e.shape.next_free_vgpr, "sgpr_max": e.shape.next_free_sgpr,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0, "group_segment_fixed_bytes": 0,
    })
}
const COUNTED: [&str; 31] = [
    "s_load_b256",
    "s_load_b128",
    "s_load_b64",
    "s_load_b32",
    "global_load_u8",
    "global_load_u16",
    "global_load_b32",
    "buffer_load_b128",
    "global_store_b32",
    "ds_bpermute_b32",
    "v_readlane_b32",
    "v_cvt_f32_f16_e64",
    "v_cvt_f32_i32_e32",
    "v_cvt_f32_u32_e32",
    "v_bfe_u32",
    "v_mul_f32_e32",
    "v_fma_f32",
    "v_fmac_f32_e32",
    "v_add_f32_e32",
    "v_cmpx_eq_u32_e32",
    "v_cndmask_b32_e64",
    "v_add_lshl_u32",
    "v_mul_lo_u32",
    "v_min_u32_e32",
    "s_cbranch_scc0",
    "s_cbranch_scc1",
    "s_cselect_b64",
    "s_endpgm",
    "s_waitcnt",
    "s_waitcnt_depctr",
    "s_delay_alu",
];

fn vreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: RKind::V,
        base,
        len,
    }
}
fn sreg(base: u8, len: u8) -> RegRef {
    RegRef {
        kind: RKind::S,
        base,
        len,
    }
}
fn refs(kind: RKind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(RKind::V, uses);
    inputs.extend(refs(RKind::S, scalar));
    b.push(Instruction::new(text, refs(RKind::V, defs), inputs))
}
/// `mnemonic v{dst}, v{voff}, s[{base}:{base+1}]{suffix}` of one dword (or sub-dword) load.
fn gload(b: &mut Builder, mnemonic: &str, dst: u8, voff: u8, base: u8, suffix: &str) -> R {
    b.push(
        Instruction::new(
            format!("{mnemonic} v{dst}, v{voff}, s[{base}:{}]{suffix}", base + 1),
            vec![vreg(dst, 1)],
            vec![vreg(voff, 1), sreg(base, 2)],
        )
        .memory(MemoryClass::VmemLoad),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
