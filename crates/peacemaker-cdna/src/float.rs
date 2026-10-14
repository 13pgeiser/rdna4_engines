// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Small-float conversions, bit-exact: FP8 (E4M3) and BF8 (E5M2) in the FNUZ
//! encoding of CDNA3 (bias 8 / 16, `0x80` = NaN, no -0, no E4M3 infinity;
//! CDNA3 ISA §7.2 Table 30 p.50) and the OCP encoding (E4M3FN bias 7, max 448,
//! NaN `0x7f`/`0xff`; E5M2 bias 15 with infinities), F16 and BF16.
//!
//! F32 → FP8/BF8 follows CDNA3 ISA §7.2 p.50-51: round to nearest even
//! (`v_cvt_pk_*`) or stochastic (`v_cvt_sr_*`: add the random dword's
//! low bits below the result's LSB, then truncate), clamp/omod ignored, and
//! the FP16_OVFL table (p.51): with FP16_OVFL=1 ±Inf and out-of-range
//! values saturate to ±max (E5M2: ±max), with FP16_OVFL=0 they become NaN
//! (E4M3) / ±Inf (E5M2; FNUZ has no Inf → NaN `0x80`). NaN → NaN.
//!
//! SCAFFOLD CONTRACT (owner: child `Float`). Signatures are frozen; every
//! `todo!()` is the child's to replace.
use crate::arch::Fp8Encoding;

/// FP8 (E4M3) or BF8 (E5M2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F8 { Fp8, Bf8 }

/// The value of `code` in `enc` (NaN codes → a quiet f32 NaN).
pub fn f8_to_f32(enc: Fp8Encoding, kind: F8, code: u8) -> f32 { let _ = (enc, kind, code); todo!("Float: f8_to_f32") }

/// `v_cvt_pk_{fp8,bf8}_f32` on one element: RNE with the FP16_OVFL rule.
pub fn f32_to_f8_rne(enc: Fp8Encoding, kind: F8, x: f32, fp16_ovfl: bool) -> u8 { let _ = (enc, kind, x, fp16_ovfl); todo!("Float: f32_to_f8_rne") }

/// `v_cvt_sr_{fp8,bf8}_f32`: stochastic rounding with random dword `rnd`
/// (CDNA3 ISA §7.2 p.51; the exact bit-use of `rnd` is `[UNVERIFIED]` until
/// confirmed on MI300X and must be documented as such).
pub fn f32_to_f8_sr(enc: Fp8Encoding, kind: F8, x: f32, rnd: u32, fp16_ovfl: bool) -> u8 { let _ = (enc, kind, x, rnd, fp16_ovfl); todo!("Float: f32_to_f8_sr") }

/// OCP E4M3FN byte → its FNUZ E4M3 image with value `ocp/2`: the same byte
/// for every finite code except `0x80` (OCP -0) → `0x00`; `None` for the OCP
/// NaN codes `0x7f`/`0xff` (GFX942-FP8-FN-PLAN §3.2).
pub fn ocp_e4m3_to_fnuz_half(code: u8) -> Option<u8> { let _ = code; todo!("Float: ocp_e4m3_to_fnuz_half") }

/// IEEE binary16 bits → f32 (exact).
pub fn f16_to_f32(bits: u16) -> f32 { let _ = bits; todo!("Float: f16_to_f32") }
/// f32 → binary16, round to nearest even (`v_cvt_f16_f32`, MODE RNE).
pub fn f32_to_f16_rne(x: f32) -> u16 { let _ = x; todo!("Float: f32_to_f16_rne") }
/// f32 → binary16, round toward zero (`v_cvt_pkrtz_f16_f32`).
pub fn f32_to_f16_rtz(x: f32) -> u16 { let _ = x; todo!("Float: f32_to_f16_rtz") }
/// BF16 bits → f32 (exact).
pub fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits(u32::from(bits) << 16) }
/// f32 → BF16, round to nearest even; NaN stays NaN (quiet).
pub fn f32_to_bf16_rne(x: f32) -> u16 { let _ = x; todo!("Float: f32_to_bf16_rne") }
