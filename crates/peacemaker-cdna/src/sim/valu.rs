// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vector ALU semantics: VOP1, VOP2, VOPC and VOP3 (promoted and VOP3-only),
//! including DPP/SDWA operand forms via [`super::xlane`] and the FP8/BF8
//! conversions via [`crate::float`] (CDNA3 ISA §6, §7.2, §12.6-12.10).
//!
//! SCAFFOLD CONTRACT (owner: child `SimValu`).
use super::{Flow, SimResult, Wave};
use crate::inst::Inst;

/// Execute one vector-ALU instruction under EXEC (VOPC/`v_cmpx` and
/// `v_readlane`/`v_readfirstlane` per their scalar-result rules).
pub fn exec(w: &mut Wave, inst: &Inst) -> SimResult<Flow> { let _ = (w, inst); todo!("SimValu: exec") }
