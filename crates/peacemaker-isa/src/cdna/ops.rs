// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Instruction constructors. VALU helpers choose the 32-bit encoding when
//! the operands allow it (src1 a VGPR, no modifiers, VCC as the implicit
//! scalar operand) and the VOP3 form otherwise, as the assembler does.
//!
//! SCAFFOLD CONTRACT (owner: child `IsaOps`). Signatures are frozen.
use peacemaker_cdna::enc::{
    ds::DsOp, flat::FlatOp, mubuf::MubufOp, smem::SmemOp,
    sop::{Sop1Op, Sop2Op, SopcOp, SopkOp, SoppOp},
    vop::{DppCtrl, SdwaSel, Vop1Op, Vop2Op, VopcOp}, vop3::{Vop3Op, Vop3OnlyOp}, vop3p::{MaiOp, Vop3pOp},
};
use peacemaker_cdna::{Inst, Src};

/// VOP3 modifiers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods { pub abs: u8, pub neg: u8, pub clamp: bool, pub omod: u8, pub op_sel: u8 }

pub fn s1(op: Sop1Op, dst: Src, src: Src) -> Inst { let _ = (op, dst, src); todo!("IsaOps") }
pub fn s2(op: Sop2Op, dst: Src, a: Src, b: Src) -> Inst { let _ = (op, dst, a, b); todo!("IsaOps") }
pub fn sk(op: SopkOp, dst: Src, imm: u16) -> Inst { let _ = (op, dst, imm); todo!("IsaOps") }
pub fn sc(op: SopcOp, a: Src, b: Src) -> Inst { let _ = (op, a, b); todo!("IsaOps") }
/// SOPP other than branches (`s_nop`, `s_sleep`, `s_setprio`, ...).
pub fn sp(op: SoppOp, imm: u16) -> Inst { let _ = (op, imm); todo!("IsaOps") }
/// `s_load_dword*` / `s_buffer_load_dword*` with an immediate byte offset.
pub fn s_load(op: SmemOp, dst: Src, base: Src, offset: i32) -> Inst { let _ = (op, dst, base, offset); todo!("IsaOps") }
pub fn v1(op: Vop1Op, dst: Src, src: Src) -> Inst { let _ = (op, dst, src); todo!("IsaOps") }
pub fn v2(op: Vop2Op, dst: Src, a: Src, b: Src) -> Inst { let _ = (op, dst, a, b); todo!("IsaOps") }
/// Carry-out VOP2 (`v_add_co_u32` ...): `sdst` VCC → e32 when legal, else VOP3B.
pub fn v2c(op: Vop2Op, dst: Src, sdst: Src, a: Src, b: Src) -> Inst { let _ = (op, dst, sdst, a, b); todo!("IsaOps") }
/// Carry-in VOP2 (`v_addc_co_u32`, `v_cndmask_b32`): `cin` VCC → e32 when legal.
pub fn v2ci(op: Vop2Op, dst: Src, sdst: Option<Src>, a: Src, b: Src, cin: Src) -> Inst { let _ = (op, dst, sdst, a, b, cin); todo!("IsaOps") }
/// Compare into `sdst` (VCC → e32 when legal).
pub fn vc(op: VopcOp, sdst: Src, a: Src, b: Src) -> Inst { let _ = (op, sdst, a, b); todo!("IsaOps") }
/// VOP3-only op without modifiers.
pub fn v3(op: Vop3OnlyOp, dst: Src, srcs: &[Src]) -> Inst { let _ = (op, dst, srcs); todo!("IsaOps") }
/// VOP3B op (`v_div_scale_f32`, `v_mad_u64_u32`, ...).
pub fn v3b(op: Vop3OnlyOp, dst: Src, sdst: Src, srcs: &[Src]) -> Inst { let _ = (op, dst, sdst, srcs); todo!("IsaOps") }
/// Any VOP3 op with modifiers.
pub fn v3m(op: Vop3Op, dst: Src, srcs: &[Src], mods: Mods) -> Inst { let _ = (op, dst, srcs, mods); todo!("IsaOps") }
/// VOP1 with DPP on `src` (a VGPR).
pub fn dpp1(op: Vop1Op, dst: Src, src: Src, ctrl: DppCtrl, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> Inst { let _ = (op, dst, src, ctrl, row_mask, bank_mask, bound_ctrl); todo!("IsaOps") }
/// VOP2 with DPP on `a` (a VGPR).
pub fn dpp2(op: Vop2Op, dst: Src, a: Src, b: Src, ctrl: DppCtrl, row_mask: u8, bank_mask: u8, bound_ctrl: bool) -> Inst { let _ = (op, dst, a, b, ctrl, row_mask, bank_mask, bound_ctrl); todo!("IsaOps") }
/// VOP1 with SDWA source select (`v_cvt_f32_fp8` byte, `v_cvt_pk_f32_fp8` word).
pub fn sdwa1(op: Vop1Op, dst: Src, src: Src, src_sel: SdwaSel) -> Inst { let _ = (op, dst, src, src_sel); todo!("IsaOps") }
pub fn vp(op: Vop3pOp, dst: Src, srcs: &[Src], op_sel: u8, op_sel_hi: u8) -> Inst { let _ = (op, dst, srcs, op_sel, op_sel_hi); todo!("IsaOps") }
/// MFMA `dst = c + a·b` (dst/c AGPRs or VGPRs, `c` may be inline 0).
pub fn mfma(op: MaiOp, dst: Src, a: Src, b: Src, c: Src) -> Inst { let _ = (op, dst, a, b, c); todo!("IsaOps") }
pub fn acc_read(dst: Src, src: Src) -> Inst { let _ = (dst, src); todo!("IsaOps") }
pub fn acc_write(dst: Src, src: Src) -> Inst { let _ = (dst, src); todo!("IsaOps") }
/// `global_load_*`: `saddr` `None` → 64-bit VGPR address in `vaddr`.
pub fn global_load(op: FlatOp, dst: Src, vaddr: Src, saddr: Option<Src>, offset: i16) -> Inst { let _ = (op, dst, vaddr, saddr, offset); todo!("IsaOps") }
pub fn global_store(op: FlatOp, vaddr: Src, data: Src, saddr: Option<Src>, offset: i16) -> Inst { let _ = (op, vaddr, data, saddr, offset); todo!("IsaOps") }
pub fn buffer_load(op: MubufOp, dst: Src, vaddr: Option<Src>, srsrc: Src, soffset: Src, offset: u16, offen: bool) -> Inst { let _ = (op, dst, vaddr, srsrc, soffset, offset, offen); todo!("IsaOps") }
pub fn buffer_store(op: MubufOp, data: Src, vaddr: Option<Src>, srsrc: Src, soffset: Src, offset: u16, offen: bool) -> Inst { let _ = (op, data, vaddr, srsrc, soffset, offset, offen); todo!("IsaOps") }
pub fn ds_read(op: DsOp, dst: Src, addr: Src, offset: u16) -> Inst { let _ = (op, dst, addr, offset); todo!("IsaOps") }
pub fn ds_write(op: DsOp, addr: Src, data: Src, offset: u16) -> Inst { let _ = (op, addr, data, offset); todo!("IsaOps") }
pub fn ds_read2(op: DsOp, dst: Src, addr: Src, offset0: u8, offset1: u8) -> Inst { let _ = (op, dst, addr, offset0, offset1); todo!("IsaOps") }
pub fn ds_write2(op: DsOp, addr: Src, data0: Src, data1: Src, offset0: u8, offset1: u8) -> Inst { let _ = (op, addr, data0, data1, offset0, offset1); todo!("IsaOps") }
/// `ds_swizzle_b32` (`addr` ignored: pass the data VGPR's lane-free form), `ds_permute_b32`, `ds_bpermute_b32`.
pub fn ds_xlane(op: DsOp, dst: Src, addr: Src, data: Src, offset: u16) -> Inst { let _ = (op, dst, addr, data, offset); todo!("IsaOps") }
/// Raw promoted-VOP3 form of a 32-bit op (for tests and for forms `v1`/`v2` would not pick).
pub fn vop3_of(op: Vop3Op, dst: Src, srcs: &[Src]) -> Inst { v3m(op, dst, srcs, Mods::default()) }
