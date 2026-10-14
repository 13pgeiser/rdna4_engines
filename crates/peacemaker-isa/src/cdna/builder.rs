// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The checked wave64 builder.
//!
//! Every pushed instruction goes through the shared wait ledger
//! (`crate::ledger::Ledger` with `crate::Arch::Gfx942` rules): the builder
//! inserts the `s_waitcnt` (one combined `vmcnt/lgkmcnt` instruction when
//! both are needed) the ledger requires, recording a `WaitProof`. Branches
//! carry the ledger: forward branches join at their label
//! (`Ledger::join`), a backward branch must arrive with the same pending
//! shape (`Ledger::shape`) its label had (loop invariant), else an error.
//! `barrier()` drains LGKM_CNT (DS loads and stores: the LDS RAW and WAR
//! rules) before `s_barrier`. `finish()` emits `s_endpgm`, runs
//! [`super::hazard::resolve`], resolves branch offsets, encodes, and returns
//! a [`super::Kernel`].
//!
//! Register allocation is a bump allocator per file with alignment; the
//! system registers come first: s[0:1] kernarg pointer, then the enabled
//! workgroup-ID SGPRs (request them with [`Builder::workgroup_id`] before any
//! [`Builder::sgprs`]), v0 the packed work-item IDs.
//!
//! SCAFFOLD CONTRACT (owner: child `IsaBuilder`). Signatures are frozen.
use super::{Item, Kernel};
use crate::Arch;
use peacemaker_cdna::enc::sop::SoppOp;
use peacemaker_cdna::{Inst, Src};

/// An LDS allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lds { pub offset: u32, pub bytes: u32 }

pub struct Builder { _private: () }

impl Builder {
    /// `arch` must be a CDNA arch (`Arch::Gfx942`); `workgroup_size` is the
    /// flat work-item count (multiple of 64, ≤ 1024).
    pub fn new(arch: Arch, name: &str, workgroup_size: u32) -> Result<Self, String> { let _ = (arch, name, workgroup_size); todo!("IsaBuilder: new") }
    /// Declare a global-pointer kernel argument (8 bytes, 8-aligned); returns its byte offset.
    pub fn arg_ptr(&mut self, name: &str, readonly: bool) -> Result<u32, String> { let _ = (name, readonly); todo!("IsaBuilder: arg_ptr") }
    /// Declare a 4-byte by-value kernel argument; returns its byte offset.
    pub fn arg_u32(&mut self, name: &str) -> Result<u32, String> { let _ = name; todo!("IsaBuilder: arg_u32") }
    /// s[0:1]: the kernarg segment pointer.
    pub fn kernarg_base(&self) -> Src { Src::Sgpr(0) }
    /// The SGPR holding workgroup ID `dim` (0..=2); enables it.
    pub fn workgroup_id(&mut self, dim: u8) -> Result<Src, String> { let _ = dim; todo!("IsaBuilder: workgroup_id") }
    /// v0, the packed work-item IDs (X[9:0], Y[19:10], Z[29:20]); enables `dims` (1..=3) dimensions.
    pub fn workitem_ids(&mut self, dims: u8) -> Result<Src, String> { let _ = dims; todo!("IsaBuilder: workitem_ids") }
    /// `count` consecutive SGPRs starting at a multiple of `align`.
    pub fn sgprs(&mut self, count: u8, align: u8) -> Result<Src, String> { let _ = (count, align); todo!("IsaBuilder: sgprs") }
    /// `count` consecutive VGPRs starting at a multiple of `align`.
    pub fn vgprs(&mut self, count: u8, align: u8) -> Result<Src, String> { let _ = (count, align); todo!("IsaBuilder: vgprs") }
    /// `count` consecutive AccVGPRs starting at a multiple of `align`.
    pub fn agprs(&mut self, count: u8, align: u8) -> Result<Src, String> { let _ = (count, align); todo!("IsaBuilder: agprs") }
    /// Static LDS bytes (aligned); total ≤ the arch's LDS size.
    pub fn lds(&mut self, bytes: u32, align: u32) -> Result<Lds, String> { let _ = (bytes, align); todo!("IsaBuilder: lds") }
    /// MODE.FP16_OVFL for the kernel (descriptor bit).
    pub fn set_fp16_ovfl(&mut self, on: bool) { let _ = on; todo!("IsaBuilder: set_fp16_ovfl") }
    /// Any instruction except `s_waitcnt`, `s_barrier`, `s_endpgm` and
    /// branches; waits are inserted before it as the ledger requires.
    pub fn push(&mut self, inst: Inst) -> Result<(), String> { let _ = inst; todo!("IsaBuilder: push") }
    /// Place a branch target.
    pub fn label(&mut self, name: &str) -> Result<(), String> { let _ = name; todo!("IsaBuilder: label") }
    /// `s_branch` / `s_cbranch_*` to `target`.
    pub fn branch(&mut self, op: SoppOp, target: &str) -> Result<(), String> { let _ = (op, target); todo!("IsaBuilder: branch") }
    /// Drain LGKM_CNT, then `s_barrier`.
    pub fn barrier(&mut self) -> Result<(), String> { todo!("IsaBuilder: barrier") }
    /// Drain every counter (`s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` as needed).
    pub fn wait_all(&mut self) -> Result<(), String> { todo!("IsaBuilder: wait_all") }
    /// The items so far (read-only).
    pub fn items(&self) -> &[Item] { todo!("IsaBuilder: items") }
    /// `s_endpgm`, hazard pass, branch resolution, encoding.
    pub fn finish(self) -> Result<Kernel, String> { todo!("IsaBuilder: finish") }
}
