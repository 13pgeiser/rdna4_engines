// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLAT / GLOBAL / SCRATCH format (CDNA3 ISA §10, §13.6). gfx940+ cache
//! policy bits are `sc0`, `sc1`, `nt` (replacing glc/slc).
//!
//! SCAFFOLD CONTRACT (owner: child `EncFlat`). Types and the op table are
//! frozen; every `todo!()` is the child's to replace.
use crate::arch::Arch;
use crate::inst::{Footprint, MemKind};
use crate::operand::Src;
use crate::Result;
use std::fmt;

op_table!(FlatOp {
    LoadUbyte, LoadSbyte, LoadUshort, LoadSshort, LoadDword, LoadDwordx2, LoadDwordx3, LoadDwordx4,
    StoreByte, StoreShort, StoreDword, StoreDwordx2, StoreDwordx3, StoreDwordx4,
    AtomicAdd, AtomicAddF32,
});

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlatSeg { Flat, Scratch, Global }

impl FlatOp {
    pub fn from_opcode(arch: Arch, opcode: u8) -> Option<Self> { Self::ALL.iter().copied().find(|op| op.opcode(arch) == Some(opcode)) }
    /// Dword widths of (vdst, data).
    pub fn widths(self) -> (u8, u8) { todo!("EncFlat: widths") }
}

/// `addr` is a 64-bit VGPR address when `saddr` is `None` (`off`), else a
/// 32-bit VGPR offset added to the SGPR-pair base `saddr`. `offset` is the
/// signed 13-bit immediate (global/scratch) or unsigned 12-bit (flat). Atomics
/// return the old value into `vdst` when `sc0` is set.
#[derive(Clone, Debug, PartialEq)]
pub struct Flat {
    pub op: FlatOp,
    pub seg: FlatSeg,
    pub vdst: Option<Src>,
    pub addr: Option<Src>,
    pub data: Option<Src>,
    pub saddr: Option<Src>,
    pub offset: i16,
    pub sc0: bool,
    pub sc1: bool,
    pub nt: bool,
}

impl Flat {
    pub fn encode(&self, arch: Arch, out: &mut Vec<u32>) -> Result<()> { let _ = (arch, out); todo!("EncFlat: encode") }
    pub fn decode(arch: Arch, words: &[u32]) -> Result<(Self, usize)> { let _ = (arch, words); todo!("EncFlat: decode") }
    pub fn footprint(&self) -> Footprint { todo!("EncFlat: footprint") }
    /// Global/scratch count on VM_CNT only; plain FLAT on VM_CNT and LGKM_CNT
    /// (CDNA3 ISA §4.4 p.19): the kind reports the VMEM side.
    pub fn mem_kind(&self) -> MemKind { todo!("EncFlat: mem_kind") }
}

impl fmt::Display for Flat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { let _ = f; todo!("EncFlat: display") }
}

// Generated from the AMD machine-readable CDNA3 ISA (`amdgpu_isa_cdna3.xml`, GPUOpen
// machine-readable ISA zip sha256 82404f11…, encoding ENC_FLAT_GLBL); proven by the
// llvm-mc round-trip tests.
impl FlatOp {
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
                Self::AtomicAdd => 66,
                Self::AtomicAddF32 => 77,
            }),
        }
    }
    /// Mnemonic for `seg`; `None` where the segment lacks the instruction (scratch atomics).
    pub fn name(self, seg: FlatSeg) -> Option<&'static str> {
        Some(match (seg, self) {
            (FlatSeg::Global, Self::LoadUbyte) => "global_load_ubyte",
            (FlatSeg::Flat, Self::LoadUbyte) => "flat_load_ubyte",
            (FlatSeg::Scratch, Self::LoadUbyte) => "scratch_load_ubyte",
            (FlatSeg::Global, Self::LoadSbyte) => "global_load_sbyte",
            (FlatSeg::Flat, Self::LoadSbyte) => "flat_load_sbyte",
            (FlatSeg::Scratch, Self::LoadSbyte) => "scratch_load_sbyte",
            (FlatSeg::Global, Self::LoadUshort) => "global_load_ushort",
            (FlatSeg::Flat, Self::LoadUshort) => "flat_load_ushort",
            (FlatSeg::Scratch, Self::LoadUshort) => "scratch_load_ushort",
            (FlatSeg::Global, Self::LoadSshort) => "global_load_sshort",
            (FlatSeg::Flat, Self::LoadSshort) => "flat_load_sshort",
            (FlatSeg::Scratch, Self::LoadSshort) => "scratch_load_sshort",
            (FlatSeg::Global, Self::LoadDword) => "global_load_dword",
            (FlatSeg::Flat, Self::LoadDword) => "flat_load_dword",
            (FlatSeg::Scratch, Self::LoadDword) => "scratch_load_dword",
            (FlatSeg::Global, Self::LoadDwordx2) => "global_load_dwordx2",
            (FlatSeg::Flat, Self::LoadDwordx2) => "flat_load_dwordx2",
            (FlatSeg::Scratch, Self::LoadDwordx2) => "scratch_load_dwordx2",
            (FlatSeg::Global, Self::LoadDwordx3) => "global_load_dwordx3",
            (FlatSeg::Flat, Self::LoadDwordx3) => "flat_load_dwordx3",
            (FlatSeg::Scratch, Self::LoadDwordx3) => "scratch_load_dwordx3",
            (FlatSeg::Global, Self::LoadDwordx4) => "global_load_dwordx4",
            (FlatSeg::Flat, Self::LoadDwordx4) => "flat_load_dwordx4",
            (FlatSeg::Scratch, Self::LoadDwordx4) => "scratch_load_dwordx4",
            (FlatSeg::Global, Self::StoreByte) => "global_store_byte",
            (FlatSeg::Flat, Self::StoreByte) => "flat_store_byte",
            (FlatSeg::Scratch, Self::StoreByte) => "scratch_store_byte",
            (FlatSeg::Global, Self::StoreShort) => "global_store_short",
            (FlatSeg::Flat, Self::StoreShort) => "flat_store_short",
            (FlatSeg::Scratch, Self::StoreShort) => "scratch_store_short",
            (FlatSeg::Global, Self::StoreDword) => "global_store_dword",
            (FlatSeg::Flat, Self::StoreDword) => "flat_store_dword",
            (FlatSeg::Scratch, Self::StoreDword) => "scratch_store_dword",
            (FlatSeg::Global, Self::StoreDwordx2) => "global_store_dwordx2",
            (FlatSeg::Flat, Self::StoreDwordx2) => "flat_store_dwordx2",
            (FlatSeg::Scratch, Self::StoreDwordx2) => "scratch_store_dwordx2",
            (FlatSeg::Global, Self::StoreDwordx3) => "global_store_dwordx3",
            (FlatSeg::Flat, Self::StoreDwordx3) => "flat_store_dwordx3",
            (FlatSeg::Scratch, Self::StoreDwordx3) => "scratch_store_dwordx3",
            (FlatSeg::Global, Self::StoreDwordx4) => "global_store_dwordx4",
            (FlatSeg::Flat, Self::StoreDwordx4) => "flat_store_dwordx4",
            (FlatSeg::Scratch, Self::StoreDwordx4) => "scratch_store_dwordx4",
            (FlatSeg::Global, Self::AtomicAdd) => "global_atomic_add",
            (FlatSeg::Flat, Self::AtomicAdd) => "flat_atomic_add",
            (FlatSeg::Global, Self::AtomicAddF32) => "global_atomic_add_f32",
            (FlatSeg::Flat, Self::AtomicAddF32) => "flat_atomic_add_f32",
            _ => return None,
        })
    }
}
