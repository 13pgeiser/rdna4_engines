// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar ALU semantics: SOP1, SOP2, SOPK, SOPC and SOPP other than
//! `s_waitcnt` / `s_barrier` / `s_endpgm` (handled in `sim/mod.rs`)
//! (CDNA3 ISA §5, §12.1-12.5).
//!
//! SCAFFOLD CONTRACT (owner: child `SimSalu`).
use super::{Flow, SimResult, Wave};
use crate::inst::Inst;

/// Execute one scalar instruction. Branches return `Flow::Branch(offset)`
/// when taken, `Flow::Next` otherwise. Every opcode of the SOP tables is
/// implemented or returns `SimError::Unsupported` naming it.
pub fn exec(w: &mut Wave, inst: &Inst) -> SimResult<Flow> { let _ = (w, inst); todo!("SimSalu: exec") }
