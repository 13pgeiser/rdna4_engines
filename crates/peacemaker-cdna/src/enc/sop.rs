// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar ALU formats SOP1, SOP2, SOPK, SOPC, SOPP (CDNA3 ISA §13.1, §12.1-12.5).
//!
//! SCAFFOLD CONTRACT (owner: child `EncSop`). Types and op tables are frozen;
//! every `todo!()` is the child's to replace. Opcode numbers come from the
//! CDNA3 ISA / machine-readable XML and are proven against
//! `llvm-mc -arch=amdgcn -mcpu=gfx942 -show-encoding`.
use crate::arch::Arch;
use crate::inst::Footprint;
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(Sop2Op {
    AddU32, SubU32, AddI32, SubI32, AddcU32, SubbU32, MinI32, MinU32, MaxI32, MaxU32,
    CselectB32, CselectB64, AndB32, AndB64, OrB32, OrB64, XorB32, XorB64,
    Andn2B32, Andn2B64, Orn2B32, Orn2B64, LshlB32, LshlB64, LshrB32, LshrB64, AshrI32, AshrI64,
    BfmB32, MulI32, BfeU32, BfeI32, MulHiU32, MulHiI32,
    Lshl1AddU32, Lshl2AddU32, Lshl3AddU32, Lshl4AddU32,
    PackLlB32B16, PackLhB32B16, PackHhB32B16, AbsdiffI32,
});

op_table!(Sop1Op {
    MovB32, MovB64, CmovB32, CmovB64, NotB32, NotB64, BrevB32,
    Bcnt1I32B32, Bcnt1I32B64, Ff1I32B32, Ff1I32B64, FlbitI32B32, SextI32I8, SextI32I16,
    AndSaveexecB64, OrSaveexecB64, XorSaveexecB64, Andn2SaveexecB64,
    GetpcB64, SetpcB64, AbsI32, Bitset0B32, Bitset1B32,
});

op_table!(SopkOp {
    MovkI32, CmovkI32,
    CmpkEqI32, CmpkLgI32, CmpkGtI32, CmpkGeI32, CmpkLtI32, CmpkLeI32,
    CmpkEqU32, CmpkLgU32, CmpkGtU32, CmpkGeU32, CmpkLtU32, CmpkLeU32,
    AddkI32, MulkI32, GetregB32, SetregB32,
});

op_table!(SopcOp {
    CmpEqI32, CmpLgI32, CmpGtI32, CmpGeI32, CmpLtI32, CmpLeI32,
    CmpEqU32, CmpLgU32, CmpGtU32, CmpGeU32, CmpLtU32, CmpLeU32,
    Bitcmp0B32, Bitcmp1B32, CmpEqU64, CmpLgU64,
});

op_table!(SoppOp {
    Nop, Endpgm, Branch, CbranchScc0, CbranchScc1, CbranchVccz, CbranchVccnz, CbranchExecz, CbranchExecnz,
    Barrier, Waitcnt, Sleep, Setprio, Sendmsg, Trap, IcacheInv,
});

macro_rules! op_meta {
    ($t:ty, $w:ty) => {
        impl $t {
            /// Inverse of [`Self::opcode`].
            pub fn from_opcode(arch: Arch, opcode: $w) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
        }
    };
}
op_meta!(Sop2Op, u8);
op_meta!(Sop1Op, u8);
op_meta!(SopkOp, u8);
op_meta!(SopcOp, u8);
op_meta!(SoppOp, u8);

impl Sop2Op {
    /// Dword widths of (sdst, ssrc0, ssrc1).
    pub fn widths(self) -> (u8, u8, u8) { todo!("EncSop: widths") }
}
impl Sop1Op {
    /// Dword widths of (sdst, ssrc0); 0 where the operand is absent
    /// (`s_getpc_b64` has no source, `s_setpc_b64` no destination).
    pub fn widths(self) -> (u8, u8) { todo!("EncSop: widths") }
}
impl SopcOp {
    /// Dword width of both sources.
    pub fn width(self) -> u8 { todo!("EncSop: width") }
}

/// `sdst = op(ssrc0, ssrc1)`; at most one source may be a literal.
#[derive(Clone, Debug, PartialEq)]
pub struct Sop2 { pub op: Sop2Op, pub sdst: Src, pub ssrc0: Src, pub ssrc1: Src }
/// `sdst = op(ssrc0)`; `None` where the opcode has no such operand (the
/// field encodes as 0 and decodes back to `None`).
#[derive(Clone, Debug, PartialEq)]
pub struct Sop1 { pub op: Sop1Op, pub sdst: Option<Src>, pub ssrc0: Option<Src> }
/// 16-bit immediate forms. `s_getreg_b32`/`s_setreg_b32`: `sdst` is the SGPR
/// read or written and `simm16` the hwreg(id, offset, size) field.
#[derive(Clone, Debug, PartialEq)]
pub struct Sopk { pub op: SopkOp, pub sdst: Src, pub simm16: u16 }
/// `SCC = op(ssrc0, ssrc1)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sopc { pub op: SopcOp, pub ssrc0: Src, pub ssrc1: Src }
/// Program control. Branches carry the signed dword offset from the next
/// instruction in `simm16`; `s_waitcnt` carries [`waitcnt_imm`].
#[derive(Clone, Debug, PartialEq)]
pub struct Sopp { pub op: SoppOp, pub simm16: u16 }

impl Sopp {
    /// `Some(simm16 as i16)` for `s_branch` / `s_cbranch_*`.
    pub fn branch_offset(&self) -> Option<i16> {
        matches!(self.op, SoppOp::Branch | SoppOp::CbranchScc0 | SoppOp::CbranchScc1 | SoppOp::CbranchVccz
            | SoppOp::CbranchVccnz | SoppOp::CbranchExecz | SoppOp::CbranchExecnz).then_some(self.simm16 as i16)
    }
}

/// The `s_waitcnt` SIMM16 for the given counter thresholds (gfx9 layout:
/// VM_CNT[3:0] in bits 3:0 and VM_CNT[5:4] in bits 15:14, EXP_CNT bits 6:4,
/// LGKM_CNT bits 11:8; CDNA3 ISA §12.5 S_WAITCNT). Errors past the field maxima
/// of `arch` ([`crate::Caps`]).
pub fn waitcnt_imm(arch: Arch, vm: u8, exp: u8, lgkm: u8) -> Result<u16> { let _ = (arch, vm, exp, lgkm); todo!("EncSop: waitcnt_imm") }
/// Inverse of [`waitcnt_imm`]: `(vm, exp, lgkm)`.
pub fn waitcnt_fields(arch: Arch, simm16: u16) -> (u8, u8, u8) { let _ = (arch, simm16); todo!("EncSop: waitcnt_fields") }

macro_rules! format_stub {
    ($t:ty) => {
        impl $t {
            /// Append this instruction's dwords (and literal) to `out`.
            pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncSop: encode") }
            /// Decode from `words[0..]`; returns the dword count consumed.
            pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncSop: decode") }
            /// Registers read and written, including SCC/EXEC/VCC/M0 implicits.
            pub fn footprint(&self) -> Footprint { todo!("EncSop: footprint") }
        }
        impl fmt::Display for $t {
            /// Exactly `llvm-objdump --mcpu=gfx942` text (single spaces).
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncSop: display") }
        }
    };
}
format_stub!(Sop2);
format_stub!(Sop1);
format_stub!(Sopk);
format_stub!(Sopc);
format_stub!(Sopp);

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOP2); proven by the
// llvm-mc round-trip tests.
impl Sop2Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::AddU32 => 0,
                Self::SubU32 => 1,
                Self::AddI32 => 2,
                Self::SubI32 => 3,
                Self::AddcU32 => 4,
                Self::SubbU32 => 5,
                Self::MinI32 => 6,
                Self::MinU32 => 7,
                Self::MaxI32 => 8,
                Self::MaxU32 => 9,
                Self::CselectB32 => 10,
                Self::CselectB64 => 11,
                Self::AndB32 => 12,
                Self::AndB64 => 13,
                Self::OrB32 => 14,
                Self::OrB64 => 15,
                Self::XorB32 => 16,
                Self::XorB64 => 17,
                Self::Andn2B32 => 18,
                Self::Andn2B64 => 19,
                Self::Orn2B32 => 20,
                Self::Orn2B64 => 21,
                Self::LshlB32 => 28,
                Self::LshlB64 => 29,
                Self::LshrB32 => 30,
                Self::LshrB64 => 31,
                Self::AshrI32 => 32,
                Self::AshrI64 => 33,
                Self::BfmB32 => 34,
                Self::MulI32 => 36,
                Self::BfeU32 => 37,
                Self::BfeI32 => 38,
                Self::MulHiU32 => 44,
                Self::MulHiI32 => 45,
                Self::Lshl1AddU32 => 46,
                Self::Lshl2AddU32 => 47,
                Self::Lshl3AddU32 => 48,
                Self::Lshl4AddU32 => 49,
                Self::PackLlB32B16 => 50,
                Self::PackLhB32B16 => 51,
                Self::PackHhB32B16 => 52,
                Self::AbsdiffI32 => 42,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::AddU32 => "s_add_u32",
            Self::SubU32 => "s_sub_u32",
            Self::AddI32 => "s_add_i32",
            Self::SubI32 => "s_sub_i32",
            Self::AddcU32 => "s_addc_u32",
            Self::SubbU32 => "s_subb_u32",
            Self::MinI32 => "s_min_i32",
            Self::MinU32 => "s_min_u32",
            Self::MaxI32 => "s_max_i32",
            Self::MaxU32 => "s_max_u32",
            Self::CselectB32 => "s_cselect_b32",
            Self::CselectB64 => "s_cselect_b64",
            Self::AndB32 => "s_and_b32",
            Self::AndB64 => "s_and_b64",
            Self::OrB32 => "s_or_b32",
            Self::OrB64 => "s_or_b64",
            Self::XorB32 => "s_xor_b32",
            Self::XorB64 => "s_xor_b64",
            Self::Andn2B32 => "s_andn2_b32",
            Self::Andn2B64 => "s_andn2_b64",
            Self::Orn2B32 => "s_orn2_b32",
            Self::Orn2B64 => "s_orn2_b64",
            Self::LshlB32 => "s_lshl_b32",
            Self::LshlB64 => "s_lshl_b64",
            Self::LshrB32 => "s_lshr_b32",
            Self::LshrB64 => "s_lshr_b64",
            Self::AshrI32 => "s_ashr_i32",
            Self::AshrI64 => "s_ashr_i64",
            Self::BfmB32 => "s_bfm_b32",
            Self::MulI32 => "s_mul_i32",
            Self::BfeU32 => "s_bfe_u32",
            Self::BfeI32 => "s_bfe_i32",
            Self::MulHiU32 => "s_mul_hi_u32",
            Self::MulHiI32 => "s_mul_hi_i32",
            Self::Lshl1AddU32 => "s_lshl1_add_u32",
            Self::Lshl2AddU32 => "s_lshl2_add_u32",
            Self::Lshl3AddU32 => "s_lshl3_add_u32",
            Self::Lshl4AddU32 => "s_lshl4_add_u32",
            Self::PackLlB32B16 => "s_pack_ll_b32_b16",
            Self::PackLhB32B16 => "s_pack_lh_b32_b16",
            Self::PackHhB32B16 => "s_pack_hh_b32_b16",
            Self::AbsdiffI32 => "s_absdiff_i32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOP1); proven by the
// llvm-mc round-trip tests.
impl Sop1Op {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MovB32 => 0,
                Self::MovB64 => 1,
                Self::CmovB32 => 2,
                Self::CmovB64 => 3,
                Self::NotB32 => 4,
                Self::NotB64 => 5,
                Self::BrevB32 => 8,
                Self::Bcnt1I32B32 => 12,
                Self::Bcnt1I32B64 => 13,
                Self::Ff1I32B32 => 16,
                Self::Ff1I32B64 => 17,
                Self::FlbitI32B32 => 18,
                Self::SextI32I8 => 22,
                Self::SextI32I16 => 23,
                Self::AndSaveexecB64 => 32,
                Self::OrSaveexecB64 => 33,
                Self::XorSaveexecB64 => 34,
                Self::Andn2SaveexecB64 => 35,
                Self::GetpcB64 => 28,
                Self::SetpcB64 => 29,
                Self::AbsI32 => 48,
                Self::Bitset0B32 => 24,
                Self::Bitset1B32 => 26,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MovB32 => "s_mov_b32",
            Self::MovB64 => "s_mov_b64",
            Self::CmovB32 => "s_cmov_b32",
            Self::CmovB64 => "s_cmov_b64",
            Self::NotB32 => "s_not_b32",
            Self::NotB64 => "s_not_b64",
            Self::BrevB32 => "s_brev_b32",
            Self::Bcnt1I32B32 => "s_bcnt1_i32_b32",
            Self::Bcnt1I32B64 => "s_bcnt1_i32_b64",
            Self::Ff1I32B32 => "s_ff1_i32_b32",
            Self::Ff1I32B64 => "s_ff1_i32_b64",
            Self::FlbitI32B32 => "s_flbit_i32_b32",
            Self::SextI32I8 => "s_sext_i32_i8",
            Self::SextI32I16 => "s_sext_i32_i16",
            Self::AndSaveexecB64 => "s_and_saveexec_b64",
            Self::OrSaveexecB64 => "s_or_saveexec_b64",
            Self::XorSaveexecB64 => "s_xor_saveexec_b64",
            Self::Andn2SaveexecB64 => "s_andn2_saveexec_b64",
            Self::GetpcB64 => "s_getpc_b64",
            Self::SetpcB64 => "s_setpc_b64",
            Self::AbsI32 => "s_abs_i32",
            Self::Bitset0B32 => "s_bitset0_b32",
            Self::Bitset1B32 => "s_bitset1_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPK); proven by the
// llvm-mc round-trip tests.
impl SopkOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::MovkI32 => 0,
                Self::CmovkI32 => 1,
                Self::CmpkEqI32 => 2,
                Self::CmpkLgI32 => 3,
                Self::CmpkGtI32 => 4,
                Self::CmpkGeI32 => 5,
                Self::CmpkLtI32 => 6,
                Self::CmpkLeI32 => 7,
                Self::CmpkEqU32 => 8,
                Self::CmpkLgU32 => 9,
                Self::CmpkGtU32 => 10,
                Self::CmpkGeU32 => 11,
                Self::CmpkLtU32 => 12,
                Self::CmpkLeU32 => 13,
                Self::AddkI32 => 14,
                Self::MulkI32 => 15,
                Self::GetregB32 => 17,
                Self::SetregB32 => 18,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::MovkI32 => "s_movk_i32",
            Self::CmovkI32 => "s_cmovk_i32",
            Self::CmpkEqI32 => "s_cmpk_eq_i32",
            Self::CmpkLgI32 => "s_cmpk_lg_i32",
            Self::CmpkGtI32 => "s_cmpk_gt_i32",
            Self::CmpkGeI32 => "s_cmpk_ge_i32",
            Self::CmpkLtI32 => "s_cmpk_lt_i32",
            Self::CmpkLeI32 => "s_cmpk_le_i32",
            Self::CmpkEqU32 => "s_cmpk_eq_u32",
            Self::CmpkLgU32 => "s_cmpk_lg_u32",
            Self::CmpkGtU32 => "s_cmpk_gt_u32",
            Self::CmpkGeU32 => "s_cmpk_ge_u32",
            Self::CmpkLtU32 => "s_cmpk_lt_u32",
            Self::CmpkLeU32 => "s_cmpk_le_u32",
            Self::AddkI32 => "s_addk_i32",
            Self::MulkI32 => "s_mulk_i32",
            Self::GetregB32 => "s_getreg_b32",
            Self::SetregB32 => "s_setreg_b32",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPC); proven by the
// llvm-mc round-trip tests.
impl SopcOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::CmpEqI32 => 0,
                Self::CmpLgI32 => 1,
                Self::CmpGtI32 => 2,
                Self::CmpGeI32 => 3,
                Self::CmpLtI32 => 4,
                Self::CmpLeI32 => 5,
                Self::CmpEqU32 => 6,
                Self::CmpLgU32 => 7,
                Self::CmpGtU32 => 8,
                Self::CmpGeU32 => 9,
                Self::CmpLtU32 => 10,
                Self::CmpLeU32 => 11,
                Self::Bitcmp0B32 => 12,
                Self::Bitcmp1B32 => 13,
                Self::CmpEqU64 => 18,
                Self::CmpLgU64 => 19,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::CmpEqI32 => "s_cmp_eq_i32",
            Self::CmpLgI32 => "s_cmp_lg_i32",
            Self::CmpGtI32 => "s_cmp_gt_i32",
            Self::CmpGeI32 => "s_cmp_ge_i32",
            Self::CmpLtI32 => "s_cmp_lt_i32",
            Self::CmpLeI32 => "s_cmp_le_i32",
            Self::CmpEqU32 => "s_cmp_eq_u32",
            Self::CmpLgU32 => "s_cmp_lg_u32",
            Self::CmpGtU32 => "s_cmp_gt_u32",
            Self::CmpGeU32 => "s_cmp_ge_u32",
            Self::CmpLtU32 => "s_cmp_lt_u32",
            Self::CmpLeU32 => "s_cmp_le_u32",
            Self::Bitcmp0B32 => "s_bitcmp0_b32",
            Self::Bitcmp1B32 => "s_bitcmp1_b32",
            Self::CmpEqU64 => "s_cmp_eq_u64",
            Self::CmpLgU64 => "s_cmp_lg_u64",
        }
    }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SOPP); proven by the
// llvm-mc round-trip tests.
impl SoppOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::Nop => 0,
                Self::Endpgm => 1,
                Self::Branch => 2,
                Self::CbranchScc0 => 4,
                Self::CbranchScc1 => 5,
                Self::CbranchVccz => 6,
                Self::CbranchVccnz => 7,
                Self::CbranchExecz => 8,
                Self::CbranchExecnz => 9,
                Self::Barrier => 10,
                Self::Waitcnt => 12,
                Self::Sleep => 14,
                Self::Setprio => 15,
                Self::Sendmsg => 16,
                Self::Trap => 18,
                Self::IcacheInv => 19,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::Nop => "s_nop",
            Self::Endpgm => "s_endpgm",
            Self::Branch => "s_branch",
            Self::CbranchScc0 => "s_cbranch_scc0",
            Self::CbranchScc1 => "s_cbranch_scc1",
            Self::CbranchVccz => "s_cbranch_vccz",
            Self::CbranchVccnz => "s_cbranch_vccnz",
            Self::CbranchExecz => "s_cbranch_execz",
            Self::CbranchExecnz => "s_cbranch_execnz",
            Self::Barrier => "s_barrier",
            Self::Waitcnt => "s_waitcnt",
            Self::Sleep => "s_sleep",
            Self::Setprio => "s_setprio",
            Self::Sendmsg => "s_sendmsg",
            Self::Trap => "s_trap",
            Self::IcacheInv => "s_icache_inv",
        }
    }
}
