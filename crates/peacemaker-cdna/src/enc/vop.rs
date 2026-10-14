// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 32-bit vector formats VOP1, VOP2, VOPC and their DPP / SDWA extension
//! dwords (CDNA3 ISA §6, §13.3; DPP §13.3.4, SDWA §13.3.3).
//!
//! SCAFFOLD CONTRACT (owner: child `EncVop`). Types and op tables are frozen;
//! every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::Footprint;
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(Vop1Op {
    Nop, MovB32, ReadfirstlaneB32,
    CvtF32I32, CvtF32U32, CvtU32F32, CvtI32F32, CvtF16F32, CvtF32F16, CvtRpiI32F32, CvtFlrI32F32,
    CvtF32Ubyte0, CvtF32Ubyte1, CvtF32Ubyte2, CvtF32Ubyte3,
    FractF32, TruncF32, CeilF32, RndneF32, FloorF32, ExpF32, LogF32, RcpF32, RcpIflagF32, RsqF32, SqrtF32,
    NotB32, BfrevB32, FfbhU32, FfblB32, FfbhI32, FrexpExpI32F32, FrexpMantF32,
    MovB64, AccvgprMovB32,
    CvtF32Fp8, CvtF32Bf8, CvtPkF32Fp8, CvtPkF32Bf8,
    SwapB32, CvtF64F32, CvtF32F64,
});

op_table!(Vop2Op {
    CndmaskB32, AddF32, SubF32, SubrevF32, MulF32, MulI32I24, MulHiI32I24, MulU32U24, MulHiU32U24,
    MinF32, MaxF32, MinI32, MaxI32, MinU32, MaxU32,
    LshrrevB32, AshrrevI32, LshlrevB32, AndB32, OrB32, XorB32, XnorB32, FmacF32,
    AddCoU32, SubCoU32, SubrevCoU32, AddcCoU32, SubbCoU32, SubbrevCoU32,
    AddU32, SubU32, SubrevU32,
    AddF16, SubF16, MulF16, MaxF16, MinF16, AddU16, SubU16, LshlrevB16, MulLoU16,
});

op_table!(VopcOp {
    CmpClassF32, CmpLtF32, CmpEqF32, CmpLeF32, CmpGtF32, CmpLgF32, CmpGeF32, CmpOF32, CmpUF32, CmpNeqF32,
    CmpLtI32, CmpEqI32, CmpLeI32, CmpGtI32, CmpNeI32, CmpGeI32,
    CmpLtU32, CmpEqU32, CmpLeU32, CmpGtU32, CmpNeU32, CmpGeU32,
    CmpxEqU32, CmpxNeU32, CmpxGtU32, CmpxLtU32, CmpxGtI32, CmpxLtI32, CmpxGtF32, CmpxLtF32,
    CmpEqU64, CmpNeU64,
});

impl Vop1Op {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Dword widths of (vdst, src0); 0 where absent (`v_nop`).
    pub fn widths(self) -> (u8, u8) { todo!("EncVop: widths") }
}
impl Vop2Op {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Reads VCC as carry-in / select (`v_cndmask_b32`, `v_addc_co_u32`, ...).
    pub fn reads_vcc(self) -> bool { todo!("EncVop: reads_vcc") }
    /// Writes VCC as carry-out in the e32 form (`v_add_co_u32`, ...).
    pub fn writes_vcc(self) -> bool { todo!("EncVop: writes_vcc") }
}
impl VopcOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Source dword width (2 for the `_u64` compares).
    pub fn src_dwords(self) -> u8 { todo!("EncVop: src_dwords") }
    /// `v_cmpx_*`: writes EXEC (CDNA3 ISA §6: and VCC in the e32 form).
    pub fn is_cmpx(self) -> bool { todo!("EncVop: is_cmpx") }
}

/// DPP_CTRL (CDNA3 ISA §13.3.4 Table "DPP_CTRL enumeration").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DppCtrl {
    /// `quad_perm:[a,b,c,d]`, each 0..=3.
    QuadPerm([u8; 4]),
    /// `row_shl:n`, `row_shr:n`, `row_ror:n`, n in 1..=15.
    RowShl(u8), RowShr(u8), RowRor(u8),
    WaveShl1, WaveRol1, WaveShr1, WaveRor1,
    RowMirror, RowHalfMirror,
    RowBcast15, RowBcast31,
}

/// The DPP dword: `src0` of the owning instruction is the DPP source VGPR
/// (the src0 field holds 250 = DPP).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dpp {
    pub ctrl: DppCtrl,
    pub row_mask: u8,
    pub bank_mask: u8,
    /// BOUND_CTRL bit (llvm prints `bound_ctrl:0` when set on gfx9).
    pub bound_ctrl: bool,
    pub src0_neg: bool, pub src0_abs: bool, pub src1_neg: bool, pub src1_abs: bool,
}

/// SDWA operand / destination select.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaSel { Byte0, Byte1, Byte2, Byte3, Word0, Word1, Dword }
/// SDWA DST_UNUSED.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaUnused { Pad, Sext, Preserve }

/// The SDWA dword (src0 field holds 249 = SDWA). `src0`/`src1` of the owning
/// instruction may be SGPRs (the S0/S1 bits). VOPC: `sdst` is the SDWA SD
/// destination (`None` = VCC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sdwa {
    pub dst_sel: SdwaSel,
    pub dst_unused: SdwaUnused,
    pub clamp: bool,
    pub omod: u8,
    pub src0_sel: SdwaSel, pub src0_sext: bool, pub src0_neg: bool, pub src0_abs: bool,
    pub src1_sel: SdwaSel, pub src1_sext: bool, pub src1_neg: bool, pub src1_abs: bool,
    pub sdst: Option<Src>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VopExt { None, Dpp(Dpp), Sdwa(Sdwa) }

/// `vdst = op(src0)`. `v_readfirstlane_b32`: `vdst` is an SGPR. `v_nop`:
/// `vdst = Vgpr(0)`, `src0 = Sgpr(0)` (zero fields).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop1 { pub op: Vop1Op, pub vdst: Src, pub src0: Src, pub ext: VopExt }
/// `vdst = op(src0, src1)`; `src1` is a VGPR except under SDWA (S1 bit).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop2 { pub op: Vop2Op, pub vdst: Src, pub src0: Src, pub src1: Src, pub ext: VopExt }
/// `VCC = op(src0, src1)` (EXEC too for `v_cmpx_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct Vopc { pub op: VopcOp, pub src0: Src, pub src1: Src, pub ext: VopExt }

macro_rules! format_stub {
    ($t:ty) => {
        impl $t {
            pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncVop: encode") }
            pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncVop: decode") }
            pub fn footprint(&self) -> Footprint { todo!("EncVop: footprint") }
        }
        impl fmt::Display for $t {
            /// `llvm-objdump --mcpu=gfx942` text: `_e32` / `_dpp` / `_sdwa` suffix as llvm prints it.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncVop: display") }
        }
    };
}
format_stub!(Vop1);
format_stub!(Vop2);
format_stub!(Vopc);

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP1); proven by the
// llvm-mc round-trip tests.
impl Vop1Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::Nop => 0,
                Self::MovB32 => 1,
                Self::ReadfirstlaneB32 => 2,
                Self::CvtF32I32 => 5,
                Self::CvtF32U32 => 6,
                Self::CvtU32F32 => 7,
                Self::CvtI32F32 => 8,
                Self::CvtF16F32 => 10,
                Self::CvtF32F16 => 11,
                Self::CvtRpiI32F32 => 12,
                Self::CvtFlrI32F32 => 13,
                Self::CvtF32Ubyte0 => 17,
                Self::CvtF32Ubyte1 => 18,
                Self::CvtF32Ubyte2 => 19,
                Self::CvtF32Ubyte3 => 20,
                Self::FractF32 => 27,
                Self::TruncF32 => 28,
                Self::CeilF32 => 29,
                Self::RndneF32 => 30,
                Self::FloorF32 => 31,
                Self::ExpF32 => 32,
                Self::LogF32 => 33,
                Self::RcpF32 => 34,
                Self::RcpIflagF32 => 35,
                Self::RsqF32 => 36,
                Self::SqrtF32 => 39,
                Self::NotB32 => 43,
                Self::BfrevB32 => 44,
                Self::FfbhU32 => 45,
                Self::FfblB32 => 46,
                Self::FfbhI32 => 47,
                Self::FrexpExpI32F32 => 51,
                Self::FrexpMantF32 => 52,
                Self::MovB64 => 56,
                Self::AccvgprMovB32 => 82,
                Self::CvtF32Fp8 => 84,
                Self::CvtF32Bf8 => 85,
                Self::CvtPkF32Fp8 => 86,
                Self::CvtPkF32Bf8 => 87,
                Self::SwapB32 => 81,
                Self::CvtF64F32 => 16,
                Self::CvtF32F64 => 15,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::Nop => "v_nop",
            Self::MovB32 => "v_mov_b32",
            Self::ReadfirstlaneB32 => "v_readfirstlane_b32",
            Self::CvtF32I32 => "v_cvt_f32_i32",
            Self::CvtF32U32 => "v_cvt_f32_u32",
            Self::CvtU32F32 => "v_cvt_u32_f32",
            Self::CvtI32F32 => "v_cvt_i32_f32",
            Self::CvtF16F32 => "v_cvt_f16_f32",
            Self::CvtF32F16 => "v_cvt_f32_f16",
            Self::CvtRpiI32F32 => "v_cvt_rpi_i32_f32",
            Self::CvtFlrI32F32 => "v_cvt_flr_i32_f32",
            Self::CvtF32Ubyte0 => "v_cvt_f32_ubyte0",
            Self::CvtF32Ubyte1 => "v_cvt_f32_ubyte1",
            Self::CvtF32Ubyte2 => "v_cvt_f32_ubyte2",
            Self::CvtF32Ubyte3 => "v_cvt_f32_ubyte3",
            Self::FractF32 => "v_fract_f32",
            Self::TruncF32 => "v_trunc_f32",
            Self::CeilF32 => "v_ceil_f32",
            Self::RndneF32 => "v_rndne_f32",
            Self::FloorF32 => "v_floor_f32",
            Self::ExpF32 => "v_exp_f32",
            Self::LogF32 => "v_log_f32",
            Self::RcpF32 => "v_rcp_f32",
            Self::RcpIflagF32 => "v_rcp_iflag_f32",
            Self::RsqF32 => "v_rsq_f32",
            Self::SqrtF32 => "v_sqrt_f32",
            Self::NotB32 => "v_not_b32",
            Self::BfrevB32 => "v_bfrev_b32",
            Self::FfbhU32 => "v_ffbh_u32",
            Self::FfblB32 => "v_ffbl_b32",
            Self::FfbhI32 => "v_ffbh_i32",
            Self::FrexpExpI32F32 => "v_frexp_exp_i32_f32",
            Self::FrexpMantF32 => "v_frexp_mant_f32",
            Self::MovB64 => "v_mov_b64",
            Self::AccvgprMovB32 => "v_accvgpr_mov_b32",
            Self::CvtF32Fp8 => "v_cvt_f32_fp8",
            Self::CvtF32Bf8 => "v_cvt_f32_bf8",
            Self::CvtPkF32Fp8 => "v_cvt_pk_f32_fp8",
            Self::CvtPkF32Bf8 => "v_cvt_pk_f32_bf8",
            Self::SwapB32 => "v_swap_b32",
            Self::CvtF64F32 => "v_cvt_f64_f32",
            Self::CvtF32F64 => "v_cvt_f32_f64",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP2); proven by the
// llvm-mc round-trip tests.
impl Vop2Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CndmaskB32 => 0,
                Self::AddF32 => 1,
                Self::SubF32 => 2,
                Self::SubrevF32 => 3,
                Self::MulF32 => 5,
                Self::MulI32I24 => 6,
                Self::MulHiI32I24 => 7,
                Self::MulU32U24 => 8,
                Self::MulHiU32U24 => 9,
                Self::MinF32 => 10,
                Self::MaxF32 => 11,
                Self::MinI32 => 12,
                Self::MaxI32 => 13,
                Self::MinU32 => 14,
                Self::MaxU32 => 15,
                Self::LshrrevB32 => 16,
                Self::AshrrevI32 => 17,
                Self::LshlrevB32 => 18,
                Self::AndB32 => 19,
                Self::OrB32 => 20,
                Self::XorB32 => 21,
                Self::XnorB32 => 61,
                Self::FmacF32 => 59,
                Self::AddCoU32 => 25,
                Self::SubCoU32 => 26,
                Self::SubrevCoU32 => 27,
                Self::AddcCoU32 => 28,
                Self::SubbCoU32 => 29,
                Self::SubbrevCoU32 => 30,
                Self::AddU32 => 52,
                Self::SubU32 => 53,
                Self::SubrevU32 => 54,
                Self::AddF16 => 31,
                Self::SubF16 => 32,
                Self::MulF16 => 34,
                Self::MaxF16 => 45,
                Self::MinF16 => 46,
                Self::AddU16 => 38,
                Self::SubU16 => 39,
                Self::LshlrevB16 => 42,
                Self::MulLoU16 => 41,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CndmaskB32 => "v_cndmask_b32",
            Self::AddF32 => "v_add_f32",
            Self::SubF32 => "v_sub_f32",
            Self::SubrevF32 => "v_subrev_f32",
            Self::MulF32 => "v_mul_f32",
            Self::MulI32I24 => "v_mul_i32_i24",
            Self::MulHiI32I24 => "v_mul_hi_i32_i24",
            Self::MulU32U24 => "v_mul_u32_u24",
            Self::MulHiU32U24 => "v_mul_hi_u32_u24",
            Self::MinF32 => "v_min_f32",
            Self::MaxF32 => "v_max_f32",
            Self::MinI32 => "v_min_i32",
            Self::MaxI32 => "v_max_i32",
            Self::MinU32 => "v_min_u32",
            Self::MaxU32 => "v_max_u32",
            Self::LshrrevB32 => "v_lshrrev_b32",
            Self::AshrrevI32 => "v_ashrrev_i32",
            Self::LshlrevB32 => "v_lshlrev_b32",
            Self::AndB32 => "v_and_b32",
            Self::OrB32 => "v_or_b32",
            Self::XorB32 => "v_xor_b32",
            Self::XnorB32 => "v_xnor_b32",
            Self::FmacF32 => "v_fmac_f32",
            Self::AddCoU32 => "v_add_co_u32",
            Self::SubCoU32 => "v_sub_co_u32",
            Self::SubrevCoU32 => "v_subrev_co_u32",
            Self::AddcCoU32 => "v_addc_co_u32",
            Self::SubbCoU32 => "v_subb_co_u32",
            Self::SubbrevCoU32 => "v_subbrev_co_u32",
            Self::AddU32 => "v_add_u32",
            Self::SubU32 => "v_sub_u32",
            Self::SubrevU32 => "v_subrev_u32",
            Self::AddF16 => "v_add_f16",
            Self::SubF16 => "v_sub_f16",
            Self::MulF16 => "v_mul_f16",
            Self::MaxF16 => "v_max_f16",
            Self::MinF16 => "v_min_f16",
            Self::AddU16 => "v_add_u16",
            Self::SubU16 => "v_sub_u16",
            Self::LshlrevB16 => "v_lshlrev_b16",
            Self::MulLoU16 => "v_mul_lo_u16",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOPC); proven by the
// llvm-mc round-trip tests.
impl VopcOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CmpClassF32 => 16,
                Self::CmpLtF32 => 65,
                Self::CmpEqF32 => 66,
                Self::CmpLeF32 => 67,
                Self::CmpGtF32 => 68,
                Self::CmpLgF32 => 69,
                Self::CmpGeF32 => 70,
                Self::CmpOF32 => 71,
                Self::CmpUF32 => 72,
                Self::CmpNeqF32 => 77,
                Self::CmpLtI32 => 193,
                Self::CmpEqI32 => 194,
                Self::CmpLeI32 => 195,
                Self::CmpGtI32 => 196,
                Self::CmpNeI32 => 197,
                Self::CmpGeI32 => 198,
                Self::CmpLtU32 => 201,
                Self::CmpEqU32 => 202,
                Self::CmpLeU32 => 203,
                Self::CmpGtU32 => 204,
                Self::CmpNeU32 => 205,
                Self::CmpGeU32 => 206,
                Self::CmpxEqU32 => 218,
                Self::CmpxNeU32 => 221,
                Self::CmpxGtU32 => 220,
                Self::CmpxLtU32 => 217,
                Self::CmpxGtI32 => 212,
                Self::CmpxLtI32 => 209,
                Self::CmpxGtF32 => 84,
                Self::CmpxLtF32 => 81,
                Self::CmpEqU64 => 234,
                Self::CmpNeU64 => 237,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CmpClassF32 => "v_cmp_class_f32",
            Self::CmpLtF32 => "v_cmp_lt_f32",
            Self::CmpEqF32 => "v_cmp_eq_f32",
            Self::CmpLeF32 => "v_cmp_le_f32",
            Self::CmpGtF32 => "v_cmp_gt_f32",
            Self::CmpLgF32 => "v_cmp_lg_f32",
            Self::CmpGeF32 => "v_cmp_ge_f32",
            Self::CmpOF32 => "v_cmp_o_f32",
            Self::CmpUF32 => "v_cmp_u_f32",
            Self::CmpNeqF32 => "v_cmp_neq_f32",
            Self::CmpLtI32 => "v_cmp_lt_i32",
            Self::CmpEqI32 => "v_cmp_eq_i32",
            Self::CmpLeI32 => "v_cmp_le_i32",
            Self::CmpGtI32 => "v_cmp_gt_i32",
            Self::CmpNeI32 => "v_cmp_ne_i32",
            Self::CmpGeI32 => "v_cmp_ge_i32",
            Self::CmpLtU32 => "v_cmp_lt_u32",
            Self::CmpEqU32 => "v_cmp_eq_u32",
            Self::CmpLeU32 => "v_cmp_le_u32",
            Self::CmpGtU32 => "v_cmp_gt_u32",
            Self::CmpNeU32 => "v_cmp_ne_u32",
            Self::CmpGeU32 => "v_cmp_ge_u32",
            Self::CmpxEqU32 => "v_cmpx_eq_u32",
            Self::CmpxNeU32 => "v_cmpx_ne_u32",
            Self::CmpxGtU32 => "v_cmpx_gt_u32",
            Self::CmpxLtU32 => "v_cmpx_lt_u32",
            Self::CmpxGtI32 => "v_cmpx_gt_i32",
            Self::CmpxLtI32 => "v_cmpx_lt_i32",
            Self::CmpxGtF32 => "v_cmpx_gt_f32",
            Self::CmpxLtF32 => "v_cmpx_lt_f32",
            Self::CmpEqU64 => "v_cmp_eq_u64",
            Self::CmpNeU64 => "v_cmp_ne_u64",
        }
    }
}
