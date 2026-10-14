// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Data share format DS (CDNA3 ISA §11, §13.5). gfx9 spellings: `ds_read_*` /
//! `ds_write_*`.
//!
//! SCAFFOLD CONTRACT (owner: child `EncDs`). Types and the op table are
//! frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(DsOp {
    ReadB32, ReadB64, ReadB96, ReadB128, Read2B32, Read2B64, Read2st64B32, Read2st64B64,
    ReadU8, ReadI8, ReadU16, ReadI16,
    WriteB8, WriteB16, WriteB32, WriteB64, WriteB96, WriteB128, Write2B32, Write2B64, Write2st64B32, Write2st64B64,
    AddU32, AddRtnU32, MaxU32, MaxI32, MaxF32, MinF32, AddF32,
    SwizzleB32, PermuteB32, BpermuteB32, Nop,
});

impl DsOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Dword widths of (vdst, data0, data1); 0 where absent.
    pub fn widths(self) -> (u8, u8, u8) { todo!("EncDs: widths") }
    /// Two-address forms (`ds_read2*`, `ds_write2*`): `offset0`/`offset1`
    /// are separate 8-bit offsets in units of the element size (×64 for
    /// `st64`); otherwise one 16-bit byte offset `offset1:offset0`.
    pub fn two_addr(self) -> bool { todo!("EncDs: two_addr") }
}

/// `addr` is the VGPR address (`None` for `ds_nop`), `data0`/`data1` the
/// data VGPRs, `vdst` the returned VGPRs.
#[derive(Clone, Debug, PartialEq)]
pub struct Ds {
    pub op: DsOp,
    pub vdst: Option<Src>,
    pub addr: Option<Src>,
    pub data0: Option<Src>,
    pub data1: Option<Src>,
    pub offset0: u8,
    pub offset1: u8,
    pub gds: bool,
}

impl Ds {
    /// The 16-bit byte offset of single-address forms.
    pub fn offset(&self) -> u16 { u16::from(self.offset1) << 8 | u16::from(self.offset0) }
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncDs: encode") }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncDs: decode") }
    pub fn footprint(&self) -> Footprint { todo!("EncDs: footprint") }
    pub fn mem_kind(&self) -> MemKind { todo!("EncDs: mem_kind") }
}

impl fmt::Display for Ds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncDs: display") }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_DS); proven by the
// llvm-mc round-trip tests.
impl DsOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::ReadB32 => 54,
                Self::ReadB64 => 118,
                Self::ReadB96 => 254,
                Self::ReadB128 => 255,
                Self::Read2B32 => 55,
                Self::Read2B64 => 119,
                Self::Read2st64B32 => 56,
                Self::Read2st64B64 => 120,
                Self::ReadU8 => 58,
                Self::ReadI8 => 57,
                Self::ReadU16 => 60,
                Self::ReadI16 => 59,
                Self::WriteB8 => 30,
                Self::WriteB16 => 31,
                Self::WriteB32 => 13,
                Self::WriteB64 => 77,
                Self::WriteB96 => 222,
                Self::WriteB128 => 223,
                Self::Write2B32 => 14,
                Self::Write2B64 => 78,
                Self::Write2st64B32 => 15,
                Self::Write2st64B64 => 79,
                Self::AddU32 => 0,
                Self::AddRtnU32 => 32,
                Self::MaxU32 => 8,
                Self::MaxI32 => 6,
                Self::MaxF32 => 19,
                Self::MinF32 => 18,
                Self::AddF32 => 21,
                Self::SwizzleB32 => 61,
                Self::PermuteB32 => 62,
                Self::BpermuteB32 => 63,
                Self::Nop => 20,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadB32 => "ds_read_b32",
            Self::ReadB64 => "ds_read_b64",
            Self::ReadB96 => "ds_read_b96",
            Self::ReadB128 => "ds_read_b128",
            Self::Read2B32 => "ds_read2_b32",
            Self::Read2B64 => "ds_read2_b64",
            Self::Read2st64B32 => "ds_read2st64_b32",
            Self::Read2st64B64 => "ds_read2st64_b64",
            Self::ReadU8 => "ds_read_u8",
            Self::ReadI8 => "ds_read_i8",
            Self::ReadU16 => "ds_read_u16",
            Self::ReadI16 => "ds_read_i16",
            Self::WriteB8 => "ds_write_b8",
            Self::WriteB16 => "ds_write_b16",
            Self::WriteB32 => "ds_write_b32",
            Self::WriteB64 => "ds_write_b64",
            Self::WriteB96 => "ds_write_b96",
            Self::WriteB128 => "ds_write_b128",
            Self::Write2B32 => "ds_write2_b32",
            Self::Write2B64 => "ds_write2_b64",
            Self::Write2st64B32 => "ds_write2st64_b32",
            Self::Write2st64B64 => "ds_write2st64_b64",
            Self::AddU32 => "ds_add_u32",
            Self::AddRtnU32 => "ds_add_rtn_u32",
            Self::MaxU32 => "ds_max_u32",
            Self::MaxI32 => "ds_max_i32",
            Self::MaxF32 => "ds_max_f32",
            Self::MinF32 => "ds_min_f32",
            Self::AddF32 => "ds_add_f32",
            Self::SwizzleB32 => "ds_swizzle_b32",
            Self::PermuteB32 => "ds_permute_b32",
            Self::BpermuteB32 => "ds_bpermute_b32",
            Self::Nop => "ds_nop",
        }
    }
}
