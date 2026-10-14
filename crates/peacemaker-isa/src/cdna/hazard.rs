// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CDNA3 software wait states: the dependencies the hardware does not
//! interlock (CDNA3 ISA §4.5 Table 11 p.20-21; §7.5 "Dependency Resolution:
//! Required Independent Instructions" p.55-57 for MFMA/accvgpr; §7.2 p.51 for
//! the FP8 converts writing halves of one VGPR), cross-checked against LLVM
//! `GCNHazardRecognizer` for gfx940. `resolve` inserts the minimal `s_nop`s
//! (an `s_nop N` provides N+1 wait states) and records why; it is sound
//! across branches (state joined at labels, loop back-edges to a fixpoint).
//!
//! SCAFFOLD CONTRACT (owner: child `IsaHazard`). Signatures are frozen.
use super::Item;
use crate::hazard::HazardProof;
use peacemaker_cdna::{Arch, Inst};

/// Insert the required wait states; `HazardProof::pc_index` indexes the
/// returned items.
pub fn resolve(arch: Arch, items: Vec<Item>) -> Result<(Vec<Item>, Vec<HazardProof>), String> { let _ = (arch, items); todo!("IsaHazard: resolve") }

/// Every violated rule in a final straight-line instruction list (branches
/// as decoded), as `"index: rule"` strings; empty when hazard-free.
pub fn check(arch: Arch, insts: &[Inst]) -> Vec<String> { let _ = (arch, insts); todo!("IsaHazard: check") }
