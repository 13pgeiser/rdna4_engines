// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-lane data movement: DPP and SDWA operand semantics for the VALU,
//! and the `ds_swizzle_b32` / `ds_permute_b32` / `ds_bpermute_b32` lane
//! functions for the DS unit (CDNA3 ISA §13.3.3-13.3.4, §11 DS_SWIZZLE /
//! DS_PERMUTE / DS_BPERMUTE).
//!
//! SCAFFOLD CONTRACT (owner: child `SimXlane`).
use super::{SimResult, Wave};
use crate::enc::vop::{Dpp, SdwaSel, SdwaUnused};
use crate::WAVE;

/// The DPP source value lane `lane` reads from VGPR `vgpr` (or AGPR when
/// `agpr`): `Some(value)`, or `None` when the lane must not write (source
/// lane out of range or inactive without BOUND_CTRL, or row/bank masked off).
pub fn dpp_read(w: &Wave, dpp: &Dpp, vgpr: u8, agpr: bool, lane: usize) -> SimResult<Option<u32>> { let _ = (w, dpp, vgpr, agpr, lane); todo!("SimXlane: dpp_read") }

/// Lane `lane` writes under this DPP control's ROW_MASK / BANK_MASK.
pub fn dpp_lane_enabled(dpp: &Dpp, lane: usize) -> bool { let _ = (dpp, lane); todo!("SimXlane: dpp_lane_enabled") }

/// An SDWA source: select `sel` of `raw`, zero- or sign-extended (`sext`).
pub fn sdwa_src(raw: u32, sel: SdwaSel, sext: bool) -> u32 { let _ = (raw, sel, sext); todo!("SimXlane: sdwa_src") }

/// An SDWA destination: place `result` at `sel` of the old value per
/// DST_UNUSED.
pub fn sdwa_dst(old: u32, result: u32, sel: SdwaSel, unused: SdwaUnused) -> u32 { let _ = (old, result, sel, unused); todo!("SimXlane: sdwa_dst") }

/// `ds_swizzle_b32 offset:pattern`: per-lane result (lanes not in `exec`
/// unchanged by the caller; reads from inactive lanes give 0).
pub fn ds_swizzle(pattern: u16, data: &[u32; WAVE], exec: u64) -> SimResult<[u32; WAVE]> { let _ = (pattern, data, exec); todo!("SimXlane: ds_swizzle") }

/// `ds_bpermute_b32`: lane i reads `data[(addr[i] + offset) / 4 % 64]`.
pub fn ds_bpermute(addr: &[u32; WAVE], offset: u16, data: &[u32; WAVE], exec: u64) -> [u32; WAVE] { let _ = (addr, offset, data, exec); todo!("SimXlane: ds_bpermute") }

/// `ds_permute_b32`: lane i writes `data[i]` to lane `(addr[i] + offset) / 4 % 64`.
pub fn ds_permute(addr: &[u32; WAVE], offset: u16, data: &[u32; WAVE], exec: u64) -> [u32; WAVE] { let _ = (addr, offset, data, exec); todo!("SimXlane: ds_permute") }
