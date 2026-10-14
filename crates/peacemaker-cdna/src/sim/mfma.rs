// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Matrix and packed-math semantics: VOP3P-MAI (MFMA, `v_accvgpr_*`) and
//! VOP3P packed math / dot products (CDNA3 ISA §6.7, §7.1, §7.3).
//!
//! SCAFFOLD CONTRACT (owner: child `SimMfma`).
use super::{Config, Flow, SimResult, Wave};
use crate::inst::Inst;

/// How an F32-accumulating MFMA sums `C + Σ_k A·B` for one output element.
/// The hardware order inside one MFMA is not documented in the CDNA3 ISA
/// guide (§7.1.4 gives only the mathematical definition); the default is the
/// documented model of this emulator, `[UNVERIFIED]` against MI300X until a
/// hardware probe confirms it. I8 → I32 is exact under every model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AccumModel {
    /// One rounding: `RNE(C + Σ_k A_k·B_k)` with the sum exact.
    #[default]
    Exact,
    /// `D = C; for k ascending: D = fma(A_k, B_k, D)`, RNE each step.
    Serial,
    /// Exact sum within each group of `g` consecutive k, each group's sum
    /// added to the accumulator with one RNE, groups in ascending k.
    Grouped(u8),
}

/// Execute one VOP3P or MAI instruction.
pub fn exec(w: &mut Wave, inst: &Inst, cfg: &Config) -> SimResult<Flow> { let _ = (w, inst, cfg); todo!("SimMfma: exec") }
