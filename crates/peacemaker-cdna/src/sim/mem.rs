// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Memory semantics and the wait-counter model: SMEM, DS, FLAT/GLOBAL and
//! MUBUF issue, retire, `s_waitcnt` and program-end drain (CDNA3 ISA §4.4
//! p.19-20, §8-11). The counter discipline is described in `sim/mod.rs`.
//!
//! SCAFFOLD CONTRACT (owner: child `SimMem`).
use super::{Flow, Memory, SimResult, Wave};
use crate::inst::Inst;

/// Issue a memory instruction: read addresses (and store data) from
/// registers now, queue a [`super::Pending`] whose `retire` closure does the
/// memory access later, mark load destinations busy.
pub fn issue(w: &mut Wave, inst: &Inst, lds: &mut [u8], mem: &mut Memory) -> SimResult<Flow> { let _ = (w, inst, lds, mem); todo!("SimMem: issue") }

/// `s_waitcnt vmcnt(vm) lgkmcnt(lgkm) expcnt(exp)`: retire oldest operations
/// until each counter is at or below its threshold.
pub fn wait(w: &mut Wave, lds: &mut [u8], mem: &mut Memory, vm: u8, lgkm: u8, exp: u8) -> SimResult<()> { let _ = (w, lds, mem, vm, lgkm, exp); todo!("SimMem: wait") }

/// Retire everything (at `s_endpgm`).
pub fn drain(w: &mut Wave, lds: &mut [u8], mem: &mut Memory) -> SimResult<()> { let _ = (w, lds, mem); todo!("SimMem: drain") }
