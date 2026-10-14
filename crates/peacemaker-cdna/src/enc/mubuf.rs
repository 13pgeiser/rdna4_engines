// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Untyped buffer format MUBUF (CDNA3 ISA §8, §13.4.1).
//!
//! SCAFFOLD CONTRACT (owner: child `EncMubuf`). Types and the op table are
//! frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(MubufOp {
    LoadUbyte, LoadSbyte, LoadUshort, LoadSshort, LoadDword, LoadDwordx2, LoadDwordx3, LoadDwordx4,
    StoreByte, StoreShort, StoreDword, StoreDwordx2, StoreDwordx3, StoreDwordx4,
});

impl MubufOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    pub fn data_dwords(self) -> u8 { todo!("EncMubuf: data_dwords") }
    pub fn is_store(self) -> bool { todo!("EncMubuf: is_store") }
}

/// `vdata` is the loaded/stored VGPRs; `vaddr` the VGPR(s) for `offen`
/// and/or `idxen` (`None` when neither); `srsrc` the first of 4 SGPRs of the
/// buffer resource (multiple of 4); `soffset` an SGPR or inline constant;
/// `offset` the unsigned 12-bit immediate.
#[derive(Clone, Debug, PartialEq)]
pub struct Mubuf {
    pub op: MubufOp,
    pub vdata: Src,
    pub vaddr: Option<Src>,
    pub srsrc: Src,
    pub soffset: Src,
    pub offset: u16,
    pub offen: bool,
    pub idxen: bool,
    pub sc0: bool,
    pub sc1: bool,
    pub nt: bool,
}

impl Mubuf {
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncMubuf: encode") }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncMubuf: decode") }
    pub fn footprint(&self) -> Footprint { todo!("EncMubuf: footprint") }
    pub fn mem_kind(&self) -> MemKind { if self.op.is_store() { MemKind::VmemStore } else { MemKind::VmemLoad } }
}

impl fmt::Display for Mubuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncMubuf: display") }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_MUBUF); proven by the
// llvm-mc round-trip tests.
impl MubufOp {
    /// The opcode field on `arch`; `None` if the variant lacks the instruction.
    pub fn opcode(self, arch: Arch) -> Option<u8> {
        match arch {
            Arch::Gfx942 => Some(match self {
                Self::LoadUbyte => 16,
                Self::LoadSbyte => 17,
                Self::LoadUshort => 18,
                Self::LoadSshort => 19,
                Self::LoadDword => 20,
                Self::LoadDwordx2 => 21,
                Self::LoadDwordx3 => 22,
                Self::LoadDwordx4 => 23,
                Self::StoreByte => 24,
                Self::StoreShort => 26,
                Self::StoreDword => 28,
                Self::StoreDwordx2 => 29,
                Self::StoreDwordx3 => 30,
                Self::StoreDwordx4 => 31,
            }),
        }
    }
    /// llvm-mc mnemonic (no encoding suffix).
    pub fn name(self) -> &'static str {
        match self {
            Self::LoadUbyte => "buffer_load_ubyte",
            Self::LoadSbyte => "buffer_load_sbyte",
            Self::LoadUshort => "buffer_load_ushort",
            Self::LoadSshort => "buffer_load_sshort",
            Self::LoadDword => "buffer_load_dword",
            Self::LoadDwordx2 => "buffer_load_dwordx2",
            Self::LoadDwordx3 => "buffer_load_dwordx3",
            Self::LoadDwordx4 => "buffer_load_dwordx4",
            Self::StoreByte => "buffer_store_byte",
            Self::StoreShort => "buffer_store_short",
            Self::StoreDword => "buffer_store_dword",
            Self::StoreDwordx2 => "buffer_store_dwordx2",
            Self::StoreDwordx3 => "buffer_store_dwordx3",
            Self::StoreDwordx4 => "buffer_store_dwordx4",
        }
    }
}
