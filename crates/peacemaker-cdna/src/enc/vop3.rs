// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 64-bit vector formats VOP3A / VOP3B (CDNA3 ISA §13.3.5-13.3.6, §12.7-12.10).
//!
//! On gfx9 the VOP3 opcode space embeds the 32-bit formats: VOPC at
//! `0x000 + op`, VOP2 at `0x100 + op`, VOP1 at `0x140 + op`, VOP3-only ops at
//! their own numbers (CDNA3 ISA §12.7.1 "VOP2 using VOP3 encoding" and
//! siblings). VOP3 on gfx9 has no literal constant.
//!
//! SCAFFOLD CONTRACT (owner: child `EncVop3`). Types and the op table are
//! frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::enc::vop::{Vop1Op, Vop2Op, VopcOp};
use crate::inst::Footprint;
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(Vop3OnlyOp {
    MadI32I24, MadU32U24, FmaF32, BfeU32, BfeI32, BfiB32, AlignbitB32, AlignbyteB32,
    Min3F32, Min3I32, Min3U32, Max3F32, Max3I32, Max3U32, Med3F32, Med3I32, Med3U32,
    DivScaleF32, DivFmasF32, DivFixupF32, MadU64U32, MadI64I32,
    LshlAddU32, AddLshlU32, Add3U32, LshlOrB32, AndOrB32, Or3B32, XadU32, PermB32,
    MulLoU32, MulHiU32, MulHiI32, LdexpF32,
    CvtPkFp8F32, CvtPkBf8F32, CvtSrFp8F32, CvtSrBf8F32,
    CvtPkrtzF16F32, CvtPkU16U32, CvtPkI16I32,
    LshlrevB64, LshrrevB64, AshrrevI64,
    ReadlaneB32, WritelaneB32, BcntU32B32, MbcntLoU32B32, MbcntHiU32B32,
    PackB32F16, BfmB32, AddF64, MulF64, FmaF64,
});

/// A VOP3-encoded opcode: a promoted 32-bit opcode or a VOP3-only one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Vop3Op { C(VopcOp), V2(Vop2Op), V1(Vop1Op), Only(Vop3OnlyOp) }

impl Vop3OnlyOp {
}

impl Vop3Op {
    /// Every encodable VOP3 opcode: promoted VOPC, VOP2, VOP1 (except those
    /// with no VOP3 form on `arch`, e.g. `v_nop`/`v_readfirstlane_b32` if llvm
    /// rejects them), then VOP3-only.
    pub fn all(arch: Arch) -> Vec<Vop3Op> { let _ = arch; todo!("EncVop3: all") }
    pub fn from_opcode(arch: Arch, opcode: u16) -> Option<Self> { let _ = (arch, opcode); todo!("EncVop3: from_opcode") }
    /// VOP3B (SDST field in bits 14:8 instead of ABS/OP_SEL): carry ops,
    /// `v_div_scale_f32`, `v_mad_u64_u32`, `v_mad_i64_i32`.
    pub fn is_vop3b(self) -> bool { todo!("EncVop3: is_vop3b") }
    /// Dword widths of (dst, src0, src1, src2); 0 where absent.
    pub fn widths(self) -> (u8, u8, u8, u8) { todo!("EncVop3: widths") }
}

/// `dst = op(src0, src1, src2)` in the 64-bit encoding. `dst` is a VGPR, or
/// an SGPR/VCC for promoted VOPC compares and `v_readlane_b32`. `sdst` is
/// the VOP3B scalar destination (carry-out / `v_div_scale` VCC), `None` for
/// VOP3A. `abs`/`neg` are 3-bit per-source masks, `op_sel` the 4-bit OP_SEL
/// field (bit 3 = destination half for the FP8 converts, CDNA3 ISA §7.2 Table 31).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop3 {
    pub op: Vop3Op,
    pub dst: Src,
    pub sdst: Option<Src>,
    pub src: [Option<Src>; 3],
    pub abs: u8,
    pub neg: u8,
    pub clamp: bool,
    pub omod: u8,
    pub op_sel: u8,
}

impl Vop3 {
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncVop3: encode") }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncVop3: decode") }
    pub fn footprint(&self) -> Footprint { todo!("EncVop3: footprint") }
}

impl fmt::Display for Vop3 {
    /// `llvm-objdump --mcpu=gfx942` text, `_e64` suffix where llvm prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncVop3: display") }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP3 / VOP3_SDST_ENC); proven by the
// llvm-mc round-trip tests.
impl Vop3OnlyOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u16> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MadI32I24 => 450,
                Self::MadU32U24 => 451,
                Self::FmaF32 => 459,
                Self::BfeU32 => 456,
                Self::BfeI32 => 457,
                Self::BfiB32 => 458,
                Self::AlignbitB32 => 462,
                Self::AlignbyteB32 => 463,
                Self::Min3F32 => 464,
                Self::Min3I32 => 465,
                Self::Min3U32 => 466,
                Self::Max3F32 => 467,
                Self::Max3I32 => 468,
                Self::Max3U32 => 469,
                Self::Med3F32 => 470,
                Self::Med3I32 => 471,
                Self::Med3U32 => 472,
                Self::DivScaleF32 => 480,
                Self::DivFmasF32 => 482,
                Self::DivFixupF32 => 478,
                Self::MadU64U32 => 488,
                Self::MadI64I32 => 489,
                Self::LshlAddU32 => 509,
                Self::AddLshlU32 => 510,
                Self::Add3U32 => 511,
                Self::LshlOrB32 => 512,
                Self::AndOrB32 => 513,
                Self::Or3B32 => 514,
                Self::XadU32 => 499,
                Self::PermB32 => 493,
                Self::MulLoU32 => 645,
                Self::MulHiU32 => 646,
                Self::MulHiI32 => 647,
                Self::LdexpF32 => 648,
                Self::CvtPkFp8F32 => 674,
                Self::CvtPkBf8F32 => 675,
                Self::CvtSrFp8F32 => 676,
                Self::CvtSrBf8F32 => 677,
                Self::CvtPkrtzF16F32 => 662,
                Self::CvtPkU16U32 => 663,
                Self::CvtPkI16I32 => 664,
                Self::LshlrevB64 => 655,
                Self::LshrrevB64 => 656,
                Self::AshrrevI64 => 657,
                Self::ReadlaneB32 => 649,
                Self::WritelaneB32 => 650,
                Self::BcntU32B32 => 651,
                Self::MbcntLoU32B32 => 652,
                Self::MbcntHiU32B32 => 653,
                Self::PackB32F16 => 672,
                Self::BfmB32 => 659,
                Self::AddF64 => 640,
                Self::MulF64 => 641,
                Self::FmaF64 => 460,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MadI32I24 => "v_mad_i32_i24",
            Self::MadU32U24 => "v_mad_u32_u24",
            Self::FmaF32 => "v_fma_f32",
            Self::BfeU32 => "v_bfe_u32",
            Self::BfeI32 => "v_bfe_i32",
            Self::BfiB32 => "v_bfi_b32",
            Self::AlignbitB32 => "v_alignbit_b32",
            Self::AlignbyteB32 => "v_alignbyte_b32",
            Self::Min3F32 => "v_min3_f32",
            Self::Min3I32 => "v_min3_i32",
            Self::Min3U32 => "v_min3_u32",
            Self::Max3F32 => "v_max3_f32",
            Self::Max3I32 => "v_max3_i32",
            Self::Max3U32 => "v_max3_u32",
            Self::Med3F32 => "v_med3_f32",
            Self::Med3I32 => "v_med3_i32",
            Self::Med3U32 => "v_med3_u32",
            Self::DivScaleF32 => "v_div_scale_f32",
            Self::DivFmasF32 => "v_div_fmas_f32",
            Self::DivFixupF32 => "v_div_fixup_f32",
            Self::MadU64U32 => "v_mad_u64_u32",
            Self::MadI64I32 => "v_mad_i64_i32",
            Self::LshlAddU32 => "v_lshl_add_u32",
            Self::AddLshlU32 => "v_add_lshl_u32",
            Self::Add3U32 => "v_add3_u32",
            Self::LshlOrB32 => "v_lshl_or_b32",
            Self::AndOrB32 => "v_and_or_b32",
            Self::Or3B32 => "v_or3_b32",
            Self::XadU32 => "v_xad_u32",
            Self::PermB32 => "v_perm_b32",
            Self::MulLoU32 => "v_mul_lo_u32",
            Self::MulHiU32 => "v_mul_hi_u32",
            Self::MulHiI32 => "v_mul_hi_i32",
            Self::LdexpF32 => "v_ldexp_f32",
            Self::CvtPkFp8F32 => "v_cvt_pk_fp8_f32",
            Self::CvtPkBf8F32 => "v_cvt_pk_bf8_f32",
            Self::CvtSrFp8F32 => "v_cvt_sr_fp8_f32",
            Self::CvtSrBf8F32 => "v_cvt_sr_bf8_f32",
            Self::CvtPkrtzF16F32 => "v_cvt_pkrtz_f16_f32",
            Self::CvtPkU16U32 => "v_cvt_pk_u16_u32",
            Self::CvtPkI16I32 => "v_cvt_pk_i16_i32",
            Self::LshlrevB64 => "v_lshlrev_b64",
            Self::LshrrevB64 => "v_lshrrev_b64",
            Self::AshrrevI64 => "v_ashrrev_i64",
            Self::ReadlaneB32 => "v_readlane_b32",
            Self::WritelaneB32 => "v_writelane_b32",
            Self::BcntU32B32 => "v_bcnt_u32_b32",
            Self::MbcntLoU32B32 => "v_mbcnt_lo_u32_b32",
            Self::MbcntHiU32B32 => "v_mbcnt_hi_u32_b32",
            Self::PackB32F16 => "v_pack_b32_f16",
            Self::BfmB32 => "v_bfm_b32",
            Self::AddF64 => "v_add_f64",
            Self::MulF64 => "v_mul_f64",
            Self::FmaF64 => "v_fma_f64",
        }
    }
}
