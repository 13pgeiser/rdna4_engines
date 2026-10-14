// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar memory format SMEM (CDNA3 ISA §9, §13.2).
//!
//! SCAFFOLD CONTRACT (owner: child `EncSmem`). Types and the op table are
//! frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(SmemOp {
    LoadDword, LoadDwordx2, LoadDwordx4, LoadDwordx8, LoadDwordx16,
    BufferLoadDword, BufferLoadDwordx2, BufferLoadDwordx4, BufferLoadDwordx8, BufferLoadDwordx16,
    Memtime, Memrealtime, DcacheInv,
});

impl SmemOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Dwords written to `sdata` (0 for `s_dcache_inv`).
    pub fn data_dwords(self) -> u8 { todo!("EncSmem: data_dwords") }
    /// `s_buffer_load_*` reads a 4-SGPR buffer resource at `sbase`; the
    /// others a 2-SGPR address.
    pub fn buffer(self) -> bool { matches!(self, Self::BufferLoadDword | Self::BufferLoadDwordx2 | Self::BufferLoadDwordx4 | Self::BufferLoadDwordx8 | Self::BufferLoadDwordx16) }
}

/// The address offset: a 21-bit signed byte immediate, an SGPR, or both
/// (`soffset_en`; CDNA3 ISA §13.2 SMEM fields IMM / SOFFSET_EN / OFFSET / SOFFSET).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmemOffset { Imm(i32), Sgpr(Src), SgprImm(Src, i32) }

/// `sdata = load(sbase + offset)`. `sdata` and `sbase` are `None` for the
/// opcodes without them (`s_dcache_inv`; `s_memtime` has `sdata` only).
#[derive(Clone, Debug, PartialEq)]
pub struct Smem { pub op: SmemOp, pub sdata: Option<Src>, pub sbase: Option<Src>, pub offset: SmemOffset, pub glc: bool }

impl Smem {
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncSmem: encode") }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncSmem: decode") }
    pub fn footprint(&self) -> Footprint { todo!("EncSmem: footprint") }
    /// `SmemLoad` for every opcode here (all count on LGKM_CNT).
    pub fn mem_kind(&self) -> Option<MemKind> { Some(MemKind::SmemLoad) }
}

impl fmt::Display for Smem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncSmem: display") }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_SMEM); proven by the
// llvm-mc round-trip tests.
impl SmemOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::LoadDword => 0,
                Self::LoadDwordx2 => 1,
                Self::LoadDwordx4 => 2,
                Self::LoadDwordx8 => 3,
                Self::LoadDwordx16 => 4,
                Self::BufferLoadDword => 8,
                Self::BufferLoadDwordx2 => 9,
                Self::BufferLoadDwordx4 => 10,
                Self::BufferLoadDwordx8 => 11,
                Self::BufferLoadDwordx16 => 12,
                Self::Memtime => 36,
                Self::Memrealtime => 37,
                Self::DcacheInv => 32,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::LoadDword => "s_load_dword",
            Self::LoadDwordx2 => "s_load_dwordx2",
            Self::LoadDwordx4 => "s_load_dwordx4",
            Self::LoadDwordx8 => "s_load_dwordx8",
            Self::LoadDwordx16 => "s_load_dwordx16",
            Self::BufferLoadDword => "s_buffer_load_dword",
            Self::BufferLoadDwordx2 => "s_buffer_load_dwordx2",
            Self::BufferLoadDwordx4 => "s_buffer_load_dwordx4",
            Self::BufferLoadDwordx8 => "s_buffer_load_dwordx8",
            Self::BufferLoadDwordx16 => "s_buffer_load_dwordx16",
            Self::Memtime => "s_memtime",
            Self::Memrealtime => "s_memrealtime",
            Self::DcacheInv => "s_dcache_inv",
        }
    }
}
