// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP3P packed math and the VOP3P-MAI matrix format (MFMA, accvgpr moves)
//! (CDNA3 ISA §6.7, §7, §13.3.7-13.3.8).
//!
//! SCAFFOLD CONTRACT (owner: child `EncVop3p`). Types, op tables and
//! [`MaiOp::shape`] are frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::Footprint;
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(Vop3pOp {
    PkFmaF16, PkAddF16, PkMulF16, PkMinF16, PkMaxF16,
    PkAddU16, PkSubU16, PkMulLoU16, PkLshlrevB16, PkLshrrevB16, PkMaxI16, PkMinI16,
    FmaMixF32, FmaMixloF16, FmaMixhiF16,
    Dot2F32F16, Dot2I32I16, Dot2U32U16, Dot4I32I8, Dot4U32U8, Dot8I32I4, Dot8U32U4,
    PkFmaF32, PkMulF32, PkAddF32, PkMovB32,
});

op_table!(MaiOp {
    AccvgprReadB32, AccvgprWriteB32,
    MfmaF32_16x16x16Bf16, MfmaF32_32x32x8Bf16,
    MfmaF32_16x16x16F16, MfmaF32_32x32x8F16,
    MfmaF32_16x16x32Fp8Fp8, MfmaF32_16x16x32Fp8Bf8, MfmaF32_16x16x32Bf8Fp8, MfmaF32_16x16x32Bf8Bf8,
    MfmaF32_32x32x16Fp8Fp8, MfmaF32_32x32x16Fp8Bf8, MfmaF32_32x32x16Bf8Fp8, MfmaF32_32x32x16Bf8Bf8,
    MfmaI32_16x16x32I8, MfmaI32_32x32x16I8,
    MfmaF32_16x16x4F32, MfmaF32_32x32x2F32,
    MfmaF64_16x16x4F64,
});

impl Vop3pOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Dword widths of (dst, src0, src1, src2); 0 where absent.
    pub fn widths(self) -> (u8, u8, u8, u8) { todo!("EncVop3p: widths") }
}

/// Element type of an MFMA operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MfmaType { F32, F16, Bf16, Fp8, Bf8, I8, I32, F64 }

/// `D[b,i,j] = C[b,i,j] + Σ_k A[b,i,k]·B[b,k,j]` over `blocks` blocks of
/// M×N×K (CDNA3 ISA §7.1.4 p.46-48), with the per-lane dword counts of the
/// A, B and C/D register ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MfmaShape {
    pub m: u8, pub n: u8, pub k: u8, pub blocks: u8,
    pub a: MfmaType, pub b: MfmaType, pub cd: MfmaType,
    pub a_dwords: u8, pub b_dwords: u8, pub cd_dwords: u8,
}

impl MaiOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// The matrix shape; `None` for the accvgpr moves.
    pub fn shape(self) -> Option<MfmaShape> {
        use MfmaType::*;
        let s = |m, n, k, blocks, a, b, cd, a_dwords, cd_dwords| Some(MfmaShape { m, n, k, blocks, a, b, cd, a_dwords, b_dwords: a_dwords, cd_dwords });
        match self {
            Self::AccvgprReadB32 | Self::AccvgprWriteB32 => None,
            Self::MfmaF32_16x16x16Bf16 => s(16, 16, 16, 1, Bf16, Bf16, F32, 2, 4),
            Self::MfmaF32_32x32x8Bf16 => s(32, 32, 8, 1, Bf16, Bf16, F32, 2, 16),
            Self::MfmaF32_16x16x16F16 => s(16, 16, 16, 1, F16, F16, F32, 2, 4),
            Self::MfmaF32_32x32x8F16 => s(32, 32, 8, 1, F16, F16, F32, 2, 16),
            Self::MfmaF32_16x16x32Fp8Fp8 => s(16, 16, 32, 1, Fp8, Fp8, F32, 2, 4),
            Self::MfmaF32_16x16x32Fp8Bf8 => s(16, 16, 32, 1, Fp8, Bf8, F32, 2, 4),
            Self::MfmaF32_16x16x32Bf8Fp8 => s(16, 16, 32, 1, Bf8, Fp8, F32, 2, 4),
            Self::MfmaF32_16x16x32Bf8Bf8 => s(16, 16, 32, 1, Bf8, Bf8, F32, 2, 4),
            Self::MfmaF32_32x32x16Fp8Fp8 => s(32, 32, 16, 1, Fp8, Fp8, F32, 2, 16),
            Self::MfmaF32_32x32x16Fp8Bf8 => s(32, 32, 16, 1, Fp8, Bf8, F32, 2, 16),
            Self::MfmaF32_32x32x16Bf8Fp8 => s(32, 32, 16, 1, Bf8, Fp8, F32, 2, 16),
            Self::MfmaF32_32x32x16Bf8Bf8 => s(32, 32, 16, 1, Bf8, Bf8, F32, 2, 16),
            Self::MfmaI32_16x16x32I8 => s(16, 16, 32, 1, I8, I8, I32, 2, 4),
            Self::MfmaI32_32x32x16I8 => s(32, 32, 16, 1, I8, I8, I32, 2, 16),
            Self::MfmaF32_16x16x4F32 => s(16, 16, 4, 1, F32, F32, F32, 1, 4),
            Self::MfmaF32_32x32x2F32 => s(32, 32, 2, 1, F32, F32, F32, 1, 16),
            Self::MfmaF64_16x16x4F64 => s(16, 16, 4, 1, F64, F64, F64, 2, 8),
        }
    }
}

/// Packed math: `dst = op(src0, src1, src2)` with per-half OP_SEL /
/// OP_SEL_HI / NEG_LO / NEG_HI masks (3 bits each, one per source).
#[derive(Clone, Debug, PartialEq)]
pub struct Vop3p {
    pub op: Vop3pOp,
    pub dst: Src,
    pub src: [Option<Src>; 3],
    pub op_sel: u8,
    pub op_sel_hi: u8,
    pub neg_lo: u8,
    pub neg_hi: u8,
    pub clamp: bool,
}

/// VOP3P-MAI. `dst`/`src_c` are `Agpr` when ACC_CD is set (they must agree;
/// `src_c` may also be an inline constant); `src_a`/`src_b` are `Vgpr` or
/// `Agpr` (the ACC bits). `v_accvgpr_read_b32 vD, aS`: `dst = Vgpr`,
/// `src_a = Agpr`. `v_accvgpr_write_b32 aD, src`: `dst = Agpr`,
/// `src_a` = VGPR or inline constant. `src_b`/`src_c` are `None` for the moves.
#[derive(Clone, Debug, PartialEq)]
pub struct Mai {
    pub op: MaiOp,
    pub dst: Src,
    pub src_a: Src,
    pub src_b: Option<Src>,
    pub src_c: Option<Src>,
    pub cbsz: u8,
    pub abid: u8,
    pub blgp: u8,
}

macro_rules! format_stub {
    ($t:ty) => {
        impl $t {
            pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncVop3p: encode") }
            pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncVop3p: decode") }
            pub fn footprint(&self) -> Footprint { todo!("EncVop3p: footprint") }
        }
        impl fmt::Display for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncVop3p: display") }
        }
    };
}
format_stub!(Vop3p);
format_stub!(Mai);

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_VOP3P); proven by the
// llvm-mc round-trip tests.
impl Vop3pOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::PkFmaF16 => 14,
                Self::PkAddF16 => 15,
                Self::PkMulF16 => 16,
                Self::PkMinF16 => 17,
                Self::PkMaxF16 => 18,
                Self::PkAddU16 => 10,
                Self::PkSubU16 => 11,
                Self::PkMulLoU16 => 1,
                Self::PkLshlrevB16 => 4,
                Self::PkLshrrevB16 => 5,
                Self::PkMaxI16 => 7,
                Self::PkMinI16 => 8,
                Self::FmaMixF32 => 32,
                Self::FmaMixloF16 => 33,
                Self::FmaMixhiF16 => 34,
                Self::Dot2F32F16 => 35,
                Self::Dot2I32I16 => 38,
                Self::Dot2U32U16 => 39,
                Self::Dot4I32I8 => 40,
                Self::Dot4U32U8 => 41,
                Self::Dot8I32I4 => 42,
                Self::Dot8U32U4 => 43,
                Self::PkFmaF32 => 48,
                Self::PkMulF32 => 49,
                Self::PkAddF32 => 50,
                Self::PkMovB32 => 51,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::PkFmaF16 => "v_pk_fma_f16",
            Self::PkAddF16 => "v_pk_add_f16",
            Self::PkMulF16 => "v_pk_mul_f16",
            Self::PkMinF16 => "v_pk_min_f16",
            Self::PkMaxF16 => "v_pk_max_f16",
            Self::PkAddU16 => "v_pk_add_u16",
            Self::PkSubU16 => "v_pk_sub_u16",
            Self::PkMulLoU16 => "v_pk_mul_lo_u16",
            Self::PkLshlrevB16 => "v_pk_lshlrev_b16",
            Self::PkLshrrevB16 => "v_pk_lshrrev_b16",
            Self::PkMaxI16 => "v_pk_max_i16",
            Self::PkMinI16 => "v_pk_min_i16",
            Self::FmaMixF32 => "v_fma_mix_f32",
            Self::FmaMixloF16 => "v_fma_mixlo_f16",
            Self::FmaMixhiF16 => "v_fma_mixhi_f16",
            Self::Dot2F32F16 => "v_dot2_f32_f16",
            Self::Dot2I32I16 => "v_dot2_i32_i16",
            Self::Dot2U32U16 => "v_dot2_u32_u16",
            Self::Dot4I32I8 => "v_dot4_i32_i8",
            Self::Dot4U32U8 => "v_dot4_u32_u8",
            Self::Dot8I32I4 => "v_dot8_i32_i4",
            Self::Dot8U32U4 => "v_dot8_u32_u4",
            Self::PkFmaF32 => "v_pk_fma_f32",
            Self::PkMulF32 => "v_pk_mul_f32",
            Self::PkAddF32 => "v_pk_add_f32",
            Self::PkMovB32 => "v_pk_mov_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding VOP3P_MFMA / ENC_VOP3P); proven by the
// llvm-mc round-trip tests.
impl MaiOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::AccvgprReadB32 => 88,
                Self::AccvgprWriteB32 => 89,
                Self::MfmaF32_16x16x16Bf16 => 97,
                Self::MfmaF32_32x32x8Bf16 => 96,
                Self::MfmaF32_16x16x16F16 => 77,
                Self::MfmaF32_32x32x8F16 => 76,
                Self::MfmaF32_16x16x32Fp8Fp8 => 115,
                Self::MfmaF32_16x16x32Fp8Bf8 => 114,
                Self::MfmaF32_16x16x32Bf8Fp8 => 113,
                Self::MfmaF32_16x16x32Bf8Bf8 => 112,
                Self::MfmaF32_32x32x16Fp8Fp8 => 119,
                Self::MfmaF32_32x32x16Fp8Bf8 => 118,
                Self::MfmaF32_32x32x16Bf8Fp8 => 117,
                Self::MfmaF32_32x32x16Bf8Bf8 => 116,
                Self::MfmaI32_16x16x32I8 => 87,
                Self::MfmaI32_32x32x16I8 => 86,
                Self::MfmaF32_16x16x4F32 => 69,
                Self::MfmaF32_32x32x2F32 => 68,
                Self::MfmaF64_16x16x4F64 => 110,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::AccvgprReadB32 => "v_accvgpr_read_b32",
            Self::AccvgprWriteB32 => "v_accvgpr_write_b32",
            Self::MfmaF32_16x16x16Bf16 => "v_mfma_f32_16x16x16_bf16",
            Self::MfmaF32_32x32x8Bf16 => "v_mfma_f32_32x32x8_bf16",
            Self::MfmaF32_16x16x16F16 => "v_mfma_f32_16x16x16_f16",
            Self::MfmaF32_32x32x8F16 => "v_mfma_f32_32x32x8_f16",
            Self::MfmaF32_16x16x32Fp8Fp8 => "v_mfma_f32_16x16x32_fp8_fp8",
            Self::MfmaF32_16x16x32Fp8Bf8 => "v_mfma_f32_16x16x32_fp8_bf8",
            Self::MfmaF32_16x16x32Bf8Fp8 => "v_mfma_f32_16x16x32_bf8_fp8",
            Self::MfmaF32_16x16x32Bf8Bf8 => "v_mfma_f32_16x16x32_bf8_bf8",
            Self::MfmaF32_32x32x16Fp8Fp8 => "v_mfma_f32_32x32x16_fp8_fp8",
            Self::MfmaF32_32x32x16Fp8Bf8 => "v_mfma_f32_32x32x16_fp8_bf8",
            Self::MfmaF32_32x32x16Bf8Fp8 => "v_mfma_f32_32x32x16_bf8_fp8",
            Self::MfmaF32_32x32x16Bf8Bf8 => "v_mfma_f32_32x32x16_bf8_bf8",
            Self::MfmaI32_16x16x32I8 => "v_mfma_i32_16x16x32_i8",
            Self::MfmaI32_32x32x16I8 => "v_mfma_i32_32x32x16_i8",
            Self::MfmaF32_16x16x4F32 => "v_mfma_f32_16x16x4_f32",
            Self::MfmaF32_32x32x2F32 => "v_mfma_f32_32x32x2_f32",
            Self::MfmaF64_16x16x4F64 => "v_mfma_f64_16x16x4_f64",
        }
    }
}
