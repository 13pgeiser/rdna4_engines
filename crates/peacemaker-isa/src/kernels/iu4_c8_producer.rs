// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201 exact C8 IU4 producers of the A3B F3 prefill route, emitted with the checked ISA builder.
//! One module `iu4_c8_producer_pm_gfx1201` (bundle `kernels/iu4_c8_producer_pm_gfx1201.hxaco`) with two
//! exports that reproduce, bit for bit, the incumbent `cand_mode == 8` HIP producers
//! (`kernels/src/mq_rotate_x_i4.hip`, `fused_rmsnorm_mq_rotate.hip`, shared recipe
//! `block_i4_128_quant.hip`, `emit_iu4_sidecar_from_producer8<true>`):
//!
//!  * `iu4_c8_rotate_pm_gfx1201` - FWHT rotate of an f32 `[rows, K]` activation (K 2048 or 4096), optional AWQ
//!    divide, f32 rotated side output, C8 `block_i4_128` AoS sidecar. Block 32, grid `rows * K / 256`.
//!  * `iu4_c8_rmsnorm_rotate_pm_gfx1201` - the same behind the fused RMSNorm (K 2048). Block 256, grid `rows`.
//!
//! ABI (72-byte kernarg, frozen): `x`@0 f32, `rotated`@8 f32 (always written, non-null), `sidecar`@16 u8,
//! `signs1`@24 f32, `signs2`@32 f32, `weight`@40 f32 (norm only), `awq`@48 f32 (null = absent), `rows` i32@56,
//! `K` i32@60, `eps` f32@64, padding@68. Launches outside `rows` 1..=8192 (and K outside the export's set) exit
//! before any access. The sign tables are the incumbent `+-1.0` tables (the incumbent's first butterfly
//! contracts the sign multiply into an FMA, which is bit-identical to the separate product only because the
//! product with `+-1` is exact).
//!
//! # Arithmetic (every rounding point equals the incumbent objects)
//!
//! Lane `p` of a wave owns the eight consecutive elements `8p..8p+7` of a 256-element FWHT group.
//!  * **Rotate**: `v = x * signs1` (AWQ: `(x / awq) * signs1`, full IEEE `v_div_scale` / `v_rcp` / Newton /
//!    `v_div_fmas` / `v_div_fixup` division); in-lane butterflies strides 1, 2, 4 (`lo' = lo + hi`,
//!    `hi' = lo - hi`); lane butterflies at lane XOR 1, 2, 4, 8, 16 (`v = fma(+-1, v, partner)`, `-1` on lanes
//!    with the stride bit, i.e. `partner - v` there and `v + partner` elsewhere: one rounding each);
//!    `f = (v * 0.0625) * signs2`. `f` is stored as f32.
//!  * **Norm**: per-thread sequential `fma(x[i], x[i], acc)` over `i = tid + 256 j` (`j` ascending, first term
//!    the exact square), wave XOR tree 16, 8, 4, 2, 1 (every lane ends with the incumbent's `shfl_down` lane 0),
//!    eight wave sums through LDS combined `((r0+r4)+(r2+r6)) + ((r1+r5)+(r3+r7))` (the incumbent's
//!    `shfl_down` 4, 2, 1; every wave recomputes it, so one barrier); `mean = sum / 2048.0f` (full division),
//!    `eps + mean`, then the incumbent's **vector** `v_rsq_f32` with its denormal guard (`x < 0x00800000` ->
//!    `x * 2^24`, result `* 2^12`); `y = ((x * w) * rms)` (AWQ: `/ awq`), then `* signs1` and the rotate above.
//!  * **C8 quantizer** (`quantize_block_i4_128_wave<false>` through the lane remap, evaluated in producer
//!    ownership: each 16-lane DPP row is one 128-block, `h = lane >> 4`): `amax` = max `|f|` (order free);
//!    `base = RN(RN(amax / 7) * 0.5)`; candidate `j = 0..7` in order `dj = RN(base * RN(1 + RN(j / 7)))`
//!    (the incumbent computes `1.0f + (float)j / 7.0f` per candidate with an IEEE divide: constants below are
//!    those exact roundings, never `j * (1/7) + 1`); `q = clamp(rint(f / dj), -8, 7)` with a full division;
//!    `err = fma(-q, dj, f)`; per remap lane `mse` = four dependent FMAs from `+0` in element order (lane
//!    elements `0..3` and `4..7` are the two remap lanes of one producer lane), the remap lane tree XOR
//!    16, 8, 4, 2, 1 equals DPP `row_xmask` 8, 4, 2, 1 on each half followed by their sum (the equality the
//!    incumbent's one-pass C2 emit proves); strict `<` against the `1e30f` sentinel keeps the first candidate
//!    on ties; `best_d` starts `1.0f`; `amax == 0` never searches and forces zero codes. Final codes are
//!    recomputed from `best_d` and their exact integer sum is the record's `s`.
//!  * **Record** (72 B, `(2 * group + h) * rows + row`): `d`@0, `s`@4, codes low-nibble first @8; lane `j`
//!    of half `h` writes the dword at `8 + 4j`.
//!
//! The clamp is `v_max_num_f32` against `-8` then `v_max3_num_f32(-m, -7, -7) = -min(m, 7)`: the result is
//! `-q`, which is exactly the `-q` of `fma(-q, dj, f)` (signed zeros included). Zero blocks substitute
//! `base = 1.0f` so no lane forms a NaN quotient; the search result of such blocks is masked off anyway.
//!
//! Resources: 64 VGPRs, wave32, no scratch; norm uses one 32-byte LDS slot and one split barrier.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, bstore_b128, lit, mem, op, s, smem, sop, sr, srd_tail, v, vr};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// CLI kernel id.
pub const KERNEL_ID: &str = "iu4_c8_producer_pm";
/// Module / cache key.
pub const MODULE: &str = "iu4_c8_producer_pm_gfx1201";
/// Rotate export.
pub const ROTATE_SYMBOL: &str = "iu4_c8_rotate_pm_gfx1201";
/// RMSNorm + rotate export.
pub const NORM_SYMBOL: &str = "iu4_c8_rmsnorm_rotate_pm_gfx1201";
/// Kernarg segment bytes.
pub const KERNARG_BYTES: u32 = 72;
/// Rotate block (one wave32 per 256-element group).
pub const ROTATE_BLOCK: u32 = 32;
/// Norm block (eight waves, one group each).
pub const NORM_BLOCK: u32 = 256;
/// FWHT group width.
pub const GROUP: u32 = 256;
/// Largest supported row count.
pub const MAX_ROWS: u32 = 8192;
/// Bytes of one `block_i4_128` record.
pub const RECORD_BYTES: u32 = 72;
/// The norm K.
pub const NORM_K: u32 = 2048;
/// `RN(1 + RN(j / 7))` bits, `j = 0..7`: the incumbent's `1.0f + (float)j / 7.0f`.
pub fn candidate_multiplier(j: u32) -> f32 {
    1.0f32 + (j as f32) / 7.0f32
}

/// One of the two exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Export {
    Rotate,
    Norm,
}
impl Export {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Rotate => ROTATE_SYMBOL,
            Self::Norm => NORM_SYMBOL,
        }
    }
    /// Contract file tag.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Rotate => "rotate",
            Self::Norm => "norm",
        }
    }
    pub fn block(self) -> u32 {
        match self {
            Self::Rotate => ROTATE_BLOCK,
            Self::Norm => NORM_BLOCK,
        }
    }
    fn exit(self) -> &'static str {
        match self {
            Self::Rotate => ".Liu4c8r_exit",
            Self::Norm => ".Liu4c8n_exit",
        }
    }
    fn skip_awq(self) -> &'static str {
        match self {
            Self::Rotate => ".Liu4c8r_noawq",
            Self::Norm => ".Liu4c8n_noawq",
        }
    }
}

/// The module's exports in order.
pub fn module_exports() -> [Export; 2] {
    [Export::Rotate, Export::Norm]
}

/// Pure launch validation (host side): rotate K 2048 / 4096, norm K 2048, rows `1..=8192`.
pub fn validate(export: Export, k: usize, rows: usize) -> Result<(), String> {
    if rows == 0 || rows > MAX_ROWS as usize {
        return Err(format!("rows {rows} outside 1..={MAX_ROWS}"));
    }
    let ok = match export {
        Export::Rotate => k == 2048 || k == 4096,
        Export::Norm => k == NORM_K as usize,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{} does not serve K {k}", export.symbol()))
    }
}

fn admit(arch: Arch) -> R {
    if arch != Arch::Gfx1201 {
        return Err("iu4_c8_producer builder is exact-gfx1201 only".into());
    }
    Ok(())
}

// ------------------------------------------------------------------ SGPR map
const KARG: u8 = 0;
const ARG_X: u8 = 8;
const ARG_ROT: u8 = 10;
const ARG_SC: u8 = 12;
const ARG_S1: u8 = 14;
const ARG_S2: u8 = 16;
const ARG_W: u8 = 18;
const ARG_AWQ: u8 = 20;
const ARG_ROWS: u8 = 22;
const ARG_K: u8 = 23;
const ARG_EPS: u8 = 24;
const WGX: u8 = 25;
const ROW: u8 = 26;
const GRP: u8 = 27;
const GS: u8 = 28;
const T0: u8 = 29;
const T1: u8 = 30;
const T2: u8 = 31;
const XP: u8 = 32;
const SCP: u8 = 34;
const WP: u8 = 36;
const AP: u8 = 38;
const RSRD: u8 = 40;
const C7: u8 = 44;
const CM8: u8 = 45;
const CN7: u8 = 46;
const PSEL: u8 = 47;
const KDIV: u8 = 48;
const HS: u8 = 49;
const MHI: u8 = 50;
const ZERO: u8 = 51;
const NZ: u8 = 52;
const HAVE: u8 = 53;
const WAVE: u8 = 54;
const LT: u8 = 55;
const TAKE: u8 = 56;
const EL: u8 = 57;
/// The norm's phase-1a row base (x row, no group offset).
const XP0_ROW: u8 = 58;

// ------------------------------------------------------------------ VGPR map
const TID: u8 = 0;
const LANE: u8 = 1;
const VOFF: u8 = 2; // lane * 32: byte offset of the lane's eight f32
const VQS: u8 = 3; // qs dword store offset
const HOFF: u8 = 4; // half * rows * 72
const VJ: u8 = 5; // lane & 15
const VT: u8 = 6;
const VT2: u8 = 7;
const X: u8 = 8; // eight values (x, y, then the butterfly values)
const S1: u8 = 16; // signs1, then lane-butterfly partners
const S2: u8 = 24; // signs2
const AW: u8 = 32; // awq
const WT: u8 = 40; // norm weight, then the final f values
const SGN: u8 = 48; // lane-butterfly signs (5)
const PART: u8 = 56; // butterfly spares / divide temporaries / |f|
const DIVT: u8 = 56; // five divide temporaries
const F: u8 = WT;
// Quantizer scratch (dead X / S1 / S2 blocks).
const NQ: u8 = 8;
const ERR: u8 = 9;
const DJ: u8 = 10;
const BASE: u8 = 11;
const AMAX: u8 = 12;
const BESTD: u8 = 13;
const BESTM: u8 = 14;
const ACC0: u8 = 15;
const ACC1: u8 = 16;
const TMP: u8 = 17;
const MSE: u8 = 18;
const WORD: u8 = 19;
const SUMS: u8 = 20;
const CI: u8 = 24; // eight codes
const VGPRS: u16 = 64;
// Norm temporaries that live in the lane block and the three spare lane-sign registers.
const NV_SUM: u8 = VT;
const NV_TMP: u8 = VT2;
const NV_T4: u8 = VQS; // tid * 4 (phase-1a offset), then the LDS zero address
const NV_WAVEV: u8 = HOFF; // tid >> 5
const NV_LDSA: u8 = VJ; // wave * 4
const TA: u8 = 53;
const TB: u8 = 54;
const TC: u8 = 55;

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPRS, 104)?;
    for (name, kind, base, len) in [
        ("kernarg", Kind::S, KARG, 2),
        ("args0", Kind::S, ARG_X, 8),
        ("args1", Kind::S, ARG_S2, 8),
        ("eps", Kind::S, ARG_EPS, 1),
        ("workgroup", Kind::S, WGX, 1),
        ("row", Kind::S, ROW, 1),
        ("group", Kind::S, GRP, 1),
        ("group_shift", Kind::S, GS, 1),
        ("temp0", Kind::S, T0, 1),
        ("temp1", Kind::S, T1, 1),
        ("temp2", Kind::S, T2, 1),
        ("x_pointer", Kind::S, XP, 2),
        ("sidecar_pointer", Kind::S, SCP, 2),
        ("weight_pointer", Kind::S, WP, 2),
        ("awq_pointer", Kind::S, AP, 2),
        ("rotated_descriptor", Kind::S, RSRD, 4),
        ("const_7", Kind::S, C7, 1),
        ("const_m8", Kind::S, CM8, 1),
        ("const_mn7", Kind::S, CN7, 1),
        ("permlane_select", Kind::S, PSEL, 1),
        ("k_float", Kind::S, KDIV, 1),
        ("half_stride", Kind::S, HS, 1),
        ("half_mask", Kind::S, MHI, 1),
        ("zero_mask", Kind::S, ZERO, 1),
        ("nonzero_mask", Kind::S, NZ, 1),
        ("have_awq", Kind::S, HAVE, 1),
        ("wave", Kind::S, WAVE, 1),
        ("lt_mask", Kind::S, LT, 1),
        ("take_mask", Kind::S, TAKE, 1),
        ("element_offset", Kind::S, EL, 1),
        ("row_pointer", Kind::S, XP0_ROW, 2),
        ("lane_block", Kind::V, TID, 8),
        ("x", Kind::V, X, 8),
        ("signs1_partners", Kind::V, S1, 8),
        ("signs2", Kind::V, S2, 8),
        ("awq", Kind::V, AW, 8),
        ("weight_final", Kind::V, WT, 8),
        ("lane_signs", Kind::V, SGN, 8),
        ("spares", Kind::V, PART, 8),
    ] {
        p.add_range(name, kind, base, len, Live::Whole)?;
    }
    Ok(p)
}

// ------------------------------------------------------------------ small helpers
/// Float source operand as llvm-objdump prints it: the inline constants `0.5`, `1.0`, `2.0`, `4.0` by name,
/// everything else as a hex literal.
fn float_operand(x: f32) -> String {
    match x {
        _ if x == 0.5 => "0.5".into(),
        _ if x == 1.0 => "1.0".into(),
        _ if x == 2.0 => "2.0".into(),
        _ if x == 4.0 => "4.0".into(),
        _ => lit(x.to_bits()),
    }
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}

fn gload(b: &mut Builder, dst: u8, width: u8, voff: u8, sbase: u8, imm: u32) -> R {
    let kind = match width {
        1 => "b32",
        4 => "b128",
        _ => return Err("load width".into()),
    };
    let d = if width == 1 { v(dst) } else { vr(dst, width) };
    let off = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    mem(
        b,
        format!(
            "global_load_{kind} {d}, v{voff}, s[{sbase}:{}]{off}",
            sbase + 1
        ),
        &[d],
        &[v(voff), sr(sbase, 2)],
        MemoryClass::VmemLoad,
    )
}

fn gstore_b32(b: &mut Builder, voff: u8, data: u8, sbase: u8, imm: u32) -> R {
    let off = if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    };
    mem(
        b,
        format!(
            "global_store_b32 v{voff}, v{data}, s[{sbase}:{}]{off}",
            sbase + 1
        ),
        &[],
        &[v(voff), v(data), sr(sbase, 2)],
        MemoryClass::VmemStore,
    )
}

/// `dst = src` of lane `l ^ m` for `m` in 1, 2, 4, 8 (DPP `row_xmask`) and 16 (`v_permlanex16_b32`).
fn xch(b: &mut Builder, dst: u8, src: u8, m: u32) -> R {
    if m < 16 {
        op(
            b,
            format!("v_mov_b32_dpp v{dst}, v{src} row_xmask:{m} row_mask:0xf bank_mask:0xf"),
            &[v(dst)],
            &[v(src)],
        )
    } else {
        op(
            b,
            format!("v_permlanex16_b32 v{dst}, v{src}, s{PSEL}, 0xfedcba98"),
            &[v(dst)],
            &[v(src), s(PSEL)],
        )
    }
}

#[derive(Clone, Copy)]
enum Opd {
    V(u8),
    S(u8),
}
impl Opd {
    fn text(self) -> String {
        match self {
            Self::V(n) => format!("v{n}"),
            Self::S(n) => format!("s{n}"),
        }
    }
    fn reg(self) -> RegRef {
        match self {
            Self::V(n) => v(n),
            Self::S(n) => s(n),
        }
    }
}

/// IEEE f32 `dst = num / den`: the hipcc `v_div_scale` / `v_rcp` / fma refinement / `v_div_fmas` /
/// `v_div_fixup` expansion (op for op the incumbent objects'), scaled-numerator carry in VCC. Uses
/// `DIVT..DIVT+5`; `dst` may alias `num`.
fn fdiv(b: &mut Builder, dst: u8, num: u8, den: Opd) -> R {
    let t = |i: u8| DIVT + i;
    let d = den.text();
    op(
        b,
        format!("v_div_scale_f32 v{}, null, {d}, {d}, v{num}", t(0)),
        &[v(t(0))],
        &[den.reg(), v(num)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{}, vcc_lo, v{num}, {d}, v{num}", t(1)),
        &[v(t(1))],
        &[den.reg(), v(num)],
    )?;
    op(
        b,
        format!("v_rcp_f32_e32 v{}, v{}", t(2), t(0)),
        &[v(t(2))],
        &[v(t(0))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(3), t(0), t(2)),
        &[v(t(3))],
        &[v(t(0)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(2), t(3)),
        &[v(t(2))],
        &[v(t(2)), v(t(3))],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{}, v{}, v{}", t(3), t(1), t(2)),
        &[v(t(3))],
        &[v(t(1)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(4), t(0), t(3), t(1)),
        &[v(t(4))],
        &[v(t(0)), v(t(3)), v(t(1))],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{}, v{}, v{}", t(3), t(4), t(2)),
        &[v(t(3))],
        &[v(t(3)), v(t(4)), v(t(2))],
    )?;
    op(
        b,
        format!("v_fma_f32 v{}, -v{}, v{}, v{}", t(4), t(0), t(3), t(1)),
        &[v(t(4))],
        &[v(t(0)), v(t(3)), v(t(1))],
    )?;
    op(
        b,
        format!("v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}", t(4), t(2), t(3)),
        &[v(t(4))],
        &[v(t(4)), v(t(2)), v(t(3))],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{dst}, v{}, {d}, v{num}", t(4)),
        &[v(dst)],
        &[v(t(4)), den.reg(), v(num)],
    )
}

// ------------------------------------------------------------------ prologues
/// Kernarg loads, launch validation, the (row, group) of the workgroup, pointers and per-lane constants.
/// Leaves `ROW`, `GRP` (norm: `GRP` = wave, set after `v_readfirstlane`) valid and every pointer built.
fn prologue(b: &mut Builder, export: Export) -> R {
    let a = b.spec.arch;
    let exit = export.exit();
    smem(b, ARG_X, 8, KARG, 0)?;
    smem(b, ARG_S2, 8, KARG, 32)?;
    if export == Export::Norm {
        smem(b, ARG_EPS, 1, KARG, 64)?;
    }
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    // rows in 1..=8192 (unsigned: rows - 1 < 8192).
    sop(
        b,
        format!("{} s{T0}, s{ARG_ROWS}, -1", super::common::s_add_i32(a)),
        &[T0],
        &[ARG_ROWS],
    )?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{T0}, {}", lit(MAX_ROWS)),
        &[],
        &[T0],
    )?;
    branch(b, &format!("s_cbranch_scc1 {exit}"))?;
    match export {
        Export::Rotate => {
            // K 2048 -> shift 3, K 4096 -> shift 4, otherwise exit.
            sop(
                b,
                format!("s_cmp_eq_u32 s{ARG_K}, {}", lit(2048)),
                &[],
                &[ARG_K],
            )?;
            sop(b, format!("s_cselect_b32 s{GS}, 3, 0"), &[GS], &[])?;
            sop(
                b,
                format!("s_cmp_eq_u32 s{ARG_K}, {}", lit(4096)),
                &[],
                &[ARG_K],
            )?;
            sop(b, format!("s_cselect_b32 s{GS}, 4, s{GS}"), &[GS], &[GS])?;
            sop(b, format!("s_cmp_eq_u32 s{GS}, 0"), &[], &[GS])?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
            sop(
                b,
                format!("s_lshl_b32 s{T0}, s{ARG_ROWS}, s{GS}"),
                &[T0],
                &[ARG_ROWS, GS],
            )?;
            sop(b, format!("s_cmp_ge_u32 s{WGX}, s{T0}"), &[], &[WGX, T0])?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
            sop(
                b,
                format!("s_lshr_b32 s{ROW}, s{WGX}, s{GS}"),
                &[ROW],
                &[WGX, GS],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{T0}, s{ROW}, s{GS}"),
                &[T0],
                &[ROW, GS],
            )?;
            sop(
                b,
                format!("s_sub_co_i32 s{GRP}, s{WGX}, s{T0}"),
                &[GRP],
                &[WGX, T0],
            )?;
        }
        Export::Norm => {
            sop(
                b,
                format!("s_cmp_lg_u32 s{ARG_K}, {}", lit(NORM_K)),
                &[],
                &[ARG_K],
            )?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
            sop(
                b,
                format!("s_cmp_ge_u32 s{WGX}, s{ARG_ROWS}"),
                &[],
                &[WGX, ARG_ROWS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {exit}"))?;
            sop(b, format!("s_mov_b32 s{ROW}, s{WGX}"), &[ROW], &[WGX])?;
            op(
                b,
                format!("v_lshrrev_b32_e32 v{NV_WAVEV}, 5, v{TID}"),
                &[v(NV_WAVEV)],
                &[v(TID)],
            )?;
            op(
                b,
                format!("v_readfirstlane_b32 s{WAVE}, v{NV_WAVEV}"),
                &[s(WAVE)],
                &[v(NV_WAVEV)],
            )?;
            sop(b, format!("s_mov_b32 s{GRP}, s{WAVE}"), &[GRP], &[WAVE])?;
            sop(b, format!("s_lshl_b32 s{T0}, s{ROW}, 13"), &[T0], &[ROW])?;
            add64(b, XP0_ROW, ARG_X, T0)?;
        }
    }
    // Element byte offset of the (row, group) 256-element run: (row * K + group * 256) * 4.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ROW}, s{ARG_K}"),
        &[T0],
        &[ROW, ARG_K],
    )?;
    sop(b, format!("s_lshl_b32 s{T1}, s{GRP}, 8"), &[T1], &[GRP])?;
    sop(
        b,
        format!("{} s{T0}, s{T0}, s{T1}", super::common::s_add_i32(a)),
        &[T0],
        &[T0, T1],
    )?;
    sop(b, format!("s_lshl_b32 s{EL}, s{T0}, 2"), &[EL], &[T0])?;
    add64(b, XP, ARG_X, EL)?;
    add64(b, RSRD, ARG_ROT, EL)?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", RSRD + 1, RSRD + 1),
        &[RSRD + 1],
        &[RSRD + 1],
    )?;
    srd_tail(b, RSRD, None)?;
    // Sidecar: record base ((2 * group) * rows + row) * 72, half stride rows * 72.
    sop(b, format!("s_lshl_b32 s{T0}, s{GRP}, 1"), &[T0], &[GRP])?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{T0}, s{ARG_ROWS}"),
        &[T0],
        &[T0, ARG_ROWS],
    )?;
    sop(
        b,
        format!("{} s{T0}, s{T0}, s{ROW}", super::common::s_add_i32(a)),
        &[T0],
        &[T0, ROW],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{T0}, {}", lit(RECORD_BYTES)),
        &[T0],
        &[T0],
    )?;
    add64(b, SCP, ARG_SC, T0)?;
    sop(
        b,
        format!("s_mul_i32 s{HS}, s{ARG_ROWS}, {}", lit(RECORD_BYTES)),
        &[HS],
        &[ARG_ROWS],
    )?;
    // AWQ: scales are group-indexed (not row-indexed); an absent table loads from the x run instead.
    sop(
        b,
        format!("s_or_b32 s{T2}, s{ARG_AWQ}, s{}", ARG_AWQ + 1),
        &[T2],
        &[ARG_AWQ, ARG_AWQ + 1],
    )?;
    sop(b, format!("s_cmp_lg_u32 s{T2}, 0"), &[], &[T2])?;
    sop(b, format!("s_cselect_b32 s{HAVE}, 1, 0"), &[HAVE], &[])?;
    sop(b, format!("s_lshl_b32 s{T1}, s{GRP}, 10"), &[T1], &[GRP])?;
    add64(b, AP, ARG_AWQ, T1)?;
    sop(b, format!("s_cmp_eq_u32 s{HAVE}, 0"), &[], &[HAVE])?;
    op(
        b,
        format!(
            "s_cselect_b64 s[{AP}:{}], s[{XP}:{}], s[{AP}:{}]",
            AP + 1,
            XP + 1,
            AP + 1
        ),
        &[sr(AP, 2)],
        &[sr(XP, 2), sr(AP, 2)],
    )?;
    if export == Export::Norm {
        add64(b, WP, ARG_W, T1)?;
    }
    // Constants.
    sop(
        b,
        format!("s_mov_b32 s{C7}, {}", lit(7.0f32.to_bits())),
        &[C7],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{CM8}, {}", lit((-8.0f32).to_bits())),
        &[CM8],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{CN7}, {}", lit((-7.0f32).to_bits())),
        &[CN7],
        &[],
    )?;
    sop(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[PSEL], &[])?;
    sop(b, format!("s_mov_b32 s{MHI}, 0xffff0000"), &[MHI], &[])?;
    if export == Export::Norm {
        sop(
            b,
            format!("s_mov_b32 s{KDIV}, {}", lit((NORM_K as f32).to_bits())),
            &[KDIV],
            &[],
        )?;
    }
    // Lane constants: LANE, the 32-byte element offset, the lane-butterfly signs (-1.0 where the stride bit is set).
    op(
        b,
        format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[v(LANE)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF}, 5, v{LANE}"),
        &[v(VOFF)],
        &[v(LANE)],
    )?;
    for i in 0..5u8 {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{VT2}, {}, v{LANE}", 31 - i),
            &[v(VT2)],
            &[v(LANE)],
        )?;
        op(
            b,
            format!("v_and_b32_e32 v{VT2}, {}, v{VT2}", lit(0x8000_0000)),
            &[v(VT2)],
            &[v(VT2)],
        )?;
        op(
            b,
            format!("v_or_b32_e32 v{}, 1.0, v{VT2}", SGN + i),
            &[v(SGN + i)],
            &[v(VT2)],
        )?;
    }
    Ok(())
}

/// The lane-owned record offsets of the final stores (VQS, HOFF, VJ), computed late so the norm's early
/// temporaries can share their registers.
fn store_offsets(b: &mut Builder) -> R {
    op(
        b,
        format!("v_and_b32_e32 v{VJ}, 15, v{LANE}"),
        &[v(VJ)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{HOFF}, 0, s{HS}, s{MHI}"),
        &[v(HOFF)],
        &[s(HS), s(MHI)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{VQS}, v{VJ}, 2, v{HOFF}"),
        &[v(VQS)],
        &[v(VJ), v(HOFF)],
    )
}

// ------------------------------------------------------------------ the two bodies
/// Rotate: x, signs1, signs2 and awq loads, optional `x / awq`, `v = x * signs1`.
fn rotate_front(b: &mut Builder) -> R {
    for h in 0..2u32 {
        gload(b, X + 4 * h as u8, 4, VOFF, XP, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, S1 + 4 * h as u8, 4, VOFF, ARG_S1, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, S2 + 4 * h as u8, 4, VOFF, ARG_S2, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, AW + 4 * h as u8, 4, VOFF, AP, 16 * h)?;
    }
    awq_and_signs(b, Export::Rotate)
}

/// `x[k] / awq[k]` when a table is present (a scalar skip, with every load drained first so the builder's wait
/// ledger agrees on both paths), then `x[k] * signs1[k]`.
fn awq_and_signs(b: &mut Builder, export: Export) -> R {
    b.wait_all()?;
    sop(b, format!("s_cmp_eq_u32 s{HAVE}, 0"), &[], &[HAVE])?;
    branch(b, &format!("s_cbranch_scc1 {}", export.skip_awq()))?;
    for k in 0..8u8 {
        fdiv(b, X + k, X + k, Opd::V(AW + k))?;
    }
    b.wait_all()?;
    b.label(export.skip_awq())?;
    for k in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{0}, v{1}", X + k, S1 + k),
            &[v(X + k)],
            &[v(X + k), v(S1 + k)],
        )?;
    }
    Ok(())
}

fn ds_store_b32(addr: u8, data: u8) -> Instruction {
    Instruction::new(
        format!("ds_store_b32 v{addr}, v{data}"),
        vec![],
        vec![v(addr), v(data)],
    )
    .memory(MemoryClass::DsStore)
}
fn ds_load_b128(dst: u8, addr: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_load_b128 v[{dst}:{}], v{addr}{o}", dst + 3),
        vec![vr(dst, 4)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}

/// Norm: phase 1a sum of squares, wave tree, LDS, the eight-root tree, `rsqrt`, then `y = (x * w) * rms`
/// (AWQ `/ awq`), `y * signs1`.
fn norm_front(b: &mut Builder) -> R {
    // Phase 1a: 8 dword loads of the row at tid + 256 j (into the spare block), then the phase-2 prefetch.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{NV_T4}, 2, v{TID}"),
        &[v(NV_T4)],
        &[v(TID)],
    )?;
    for j in 0..8u32 {
        gload(b, PART + j as u8, 1, NV_T4, XP0_ROW, 1024 * j)?;
    }
    for h in 0..2u32 {
        gload(b, X + 4 * h as u8, 4, VOFF, XP, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, WT + 4 * h as u8, 4, VOFF, WP, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, AW + 4 * h as u8, 4, VOFF, AP, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, S1 + 4 * h as u8, 4, VOFF, ARG_S1, 16 * h)?;
    }
    for h in 0..2u32 {
        gload(b, S2 + 4 * h as u8, 4, VOFF, ARG_S2, 16 * h)?;
    }
    // Sequential per-thread fma chain from +0 (the first term is the exact square).
    op(
        b,
        format!("v_mul_f32_e32 v{NV_SUM}, v{PART}, v{PART}"),
        &[v(NV_SUM)],
        &[v(PART)],
    )?;
    for j in 1..8u8 {
        op(
            b,
            format!("v_fmac_f32_e32 v{NV_SUM}, v{0}, v{0}", PART + j),
            &[v(NV_SUM)],
            &[v(NV_SUM), v(PART + j)],
        )?;
    }
    // Wave tree 16, 8, 4, 2, 1.
    for m in [16u32, 8, 4, 2, 1] {
        xch(b, NV_TMP, NV_SUM, m)?;
        op(
            b,
            format!("v_add_f32_e32 v{NV_SUM}, v{NV_SUM}, v{NV_TMP}"),
            &[v(NV_SUM)],
            &[v(NV_SUM), v(NV_TMP)],
        )?;
    }
    // Every lane of the wave stores the same wave sum at wave * 4; one barrier publishes all eight.
    // (The masks are the form `pm_check::lds_bounds` proves: wave * 4 <= 28 for the store, address 0 for the loads.)
    op(
        b,
        format!("v_lshlrev_b32_e32 v{NV_LDSA}, 2, v{NV_WAVEV}"),
        &[v(NV_LDSA)],
        &[v(NV_WAVEV)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{NV_LDSA}, {}, v{NV_LDSA}", lit(0x1c)),
        &[v(NV_LDSA)],
        &[v(NV_LDSA)],
    )?;
    b.ds_store(0, ds_store_b32(NV_LDSA, NV_SUM))?;
    b.barrier(&[Transition::Ready(0)])?;
    op(
        b,
        format!("v_and_b32_e32 v{NV_T4}, 0, v{LANE}"),
        &[v(NV_T4)],
        &[v(LANE)],
    )?;
    b.ds_load(0, ds_load_b128(PART, NV_T4, 0))?;
    b.ds_load(0, ds_load_b128(PART + 4, NV_T4, 16))?;
    let r = |i: u8| PART + i;
    // ((r0 + r4) + (r2 + r6)) + ((r1 + r5) + (r3 + r7)).
    for (d, x, y) in [(TA, r(0), r(4)), (TB, r(2), r(6))] {
        op(
            b,
            format!("v_add_f32_e32 v{d}, v{x}, v{y}"),
            &[v(d)],
            &[v(x), v(y)],
        )?;
    }
    op(
        b,
        format!("v_add_f32_e32 v{TA}, v{TA}, v{TB}"),
        &[v(TA)],
        &[v(TA), v(TB)],
    )?;
    for (d, x, y) in [(TB, r(1), r(5)), (TC, r(3), r(7))] {
        op(
            b,
            format!("v_add_f32_e32 v{d}, v{x}, v{y}"),
            &[v(d)],
            &[v(x), v(y)],
        )?;
    }
    op(
        b,
        format!("v_add_f32_e32 v{TB}, v{TB}, v{TC}"),
        &[v(TB)],
        &[v(TB), v(TC)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{NV_SUM}, v{TA}, v{TB}"),
        &[v(NV_SUM)],
        &[v(TA), v(TB)],
    )?;
    // mean = sum / 2048.0f; eps + mean; the incumbent's vector rsqrt with its denormal guard.
    fdiv(b, NV_TMP, NV_SUM, Opd::S(KDIV))?;
    op(
        b,
        format!("v_add_f32_e32 v{NV_TMP}, s{ARG_EPS}, v{NV_TMP}"),
        &[v(NV_TMP)],
        &[s(ARG_EPS), v(NV_TMP)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{NV_SUM}, {}, v{NV_TMP}", lit(0x4b80_0000)),
        &[v(NV_SUM)],
        &[v(NV_TMP)],
    )?;
    op(
        b,
        format!("v_cmp_gt_f32_e64 s{LT}, {}, v{NV_TMP}", lit(0x0080_0000)),
        &[s(LT)],
        &[v(NV_TMP)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{NV_TMP}, v{NV_TMP}, v{NV_SUM}, s{LT}"),
        &[v(NV_TMP)],
        &[v(NV_TMP), v(NV_SUM), s(LT)],
    )?;
    op(
        b,
        format!("v_rsq_f32_e32 v{NV_TMP}, v{NV_TMP}"),
        &[v(NV_TMP)],
        &[v(NV_TMP)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{NV_SUM}, {}, v{NV_TMP}", lit(0x4580_0000)),
        &[v(NV_SUM)],
        &[v(NV_TMP)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{NV_SUM}, v{NV_TMP}, v{NV_SUM}, s{LT}"),
        &[v(NV_SUM)],
        &[v(NV_TMP), v(NV_SUM), s(LT)],
    )?;
    // y = (x * weight) * rms, two roundings in that order.
    for k in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{0}, v{1}", X + k, WT + k),
            &[v(X + k)],
            &[v(X + k), v(WT + k)],
        )?;
    }
    for k in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{0}, v{NV_SUM}", X + k),
            &[v(X + k)],
            &[v(X + k), v(NV_SUM)],
        )?;
    }
    awq_and_signs(b, Export::Norm)
}

// ------------------------------------------------------------------ FWHT, scale, store, quantizer
fn fwht_scale_store(b: &mut Builder) -> R {
    let mut e: [u8; 8] = core::array::from_fn(|k| X + k as u8);
    let mut spare: Vec<u8> = (PART..PART + 8).collect();
    // In-lane butterflies, strides 1, 2, 4: lo' = lo + hi, hi' = lo - hi (the difference lands in a spare).
    for stride in [1usize, 2, 4] {
        for lo in (0..8).filter(|i| i & stride == 0) {
            let (hi, t) = (lo + stride, spare.pop().ok_or("butterfly spares")?);
            op(
                b,
                format!("v_sub_f32_e32 v{t}, v{}, v{}", e[lo], e[hi]),
                &[v(t)],
                &[v(e[lo]), v(e[hi])],
            )?;
            op(
                b,
                format!("v_add_f32_e32 v{0}, v{0}, v{1}", e[lo], e[hi]),
                &[v(e[lo])],
                &[v(e[lo]), v(e[hi])],
            )?;
            spare.push(e[hi]);
            e[hi] = t;
        }
    }
    // Lane butterflies, strides 1..16: v = fma(sign, v, partner).
    for (i, m) in [1u32, 2, 4, 8, 16].into_iter().enumerate() {
        for k in 0..8u8 {
            xch(b, S1 + k, e[usize::from(k)], m)?;
        }
        let sign = SGN + i as u8;
        for k in 0..8u8 {
            let x = e[usize::from(k)];
            op(
                b,
                format!("v_fma_f32 v{x}, v{sign}, v{x}, v{}", S1 + k),
                &[v(x)],
                &[v(sign), v(x), v(S1 + k)],
            )?;
        }
    }
    // f = (v * 0.0625) * signs2.
    for k in 0..8u8 {
        op(
            b,
            format!(
                "v_mul_f32_e32 v{}, {}, v{}",
                F + k,
                lit(0.0625f32.to_bits()),
                e[usize::from(k)]
            ),
            &[v(F + k)],
            &[v(e[usize::from(k)])],
        )?;
    }
    for k in 0..8u8 {
        op(
            b,
            format!("v_mul_f32_e32 v{0}, v{0}, v{1}", F + k, S2 + k),
            &[v(F + k)],
            &[v(F + k), v(S2 + k)],
        )?;
    }
    bstore_b128(b, F, VOFF, RSRD, 0)?;
    bstore_b128(b, F + 4, VOFF, RSRD, 16)
}

/// `q = -clamp(rint(f / d), -8, 7)` into `dst`: `v_max_num(q, -8)` then `max(-m, -7) = -min(m, 7)`.
fn neg_code(b: &mut Builder, dst: u8, f: u8, d: u8) -> R {
    fdiv(b, dst, f, Opd::V(d))?;
    op(
        b,
        format!("v_rndne_f32_e32 v{0}, v{0}", dst),
        &[v(dst)],
        &[v(dst)],
    )?;
    op(
        b,
        format!("v_max_num_f32_e32 v{0}, s{CM8}, v{0}", dst),
        &[v(dst)],
        &[s(CM8), v(dst)],
    )?;
    op(
        b,
        format!("v_max3_num_f32 v{0}, -v{0}, s{CN7}, s{CN7}", dst),
        &[v(dst)],
        &[v(dst), s(CN7)],
    )
}

fn quantize(b: &mut Builder) -> R {
    // amax over the lane's |f| and the 16-lane row.
    for k in 0..8u8 {
        op(
            b,
            format!(
                "v_and_b32_e32 v{}, {}, v{}",
                PART + k,
                lit(0x7fff_ffff),
                F + k
            ),
            &[v(PART + k)],
            &[v(F + k)],
        )?;
    }
    let ab = |k: u8| PART + k;
    op(
        b,
        format!("v_max3_num_f32 v{AMAX}, v{}, v{}, v{}", ab(0), ab(1), ab(2)),
        &[v(AMAX)],
        &[v(ab(0)), v(ab(1)), v(ab(2))],
    )?;
    op(
        b,
        format!("v_max3_num_f32 v{AMAX}, v{AMAX}, v{}, v{}", ab(3), ab(4)),
        &[v(AMAX)],
        &[v(AMAX), v(ab(3)), v(ab(4))],
    )?;
    op(
        b,
        format!("v_max3_num_f32 v{AMAX}, v{AMAX}, v{}, v{}", ab(5), ab(6)),
        &[v(AMAX)],
        &[v(AMAX), v(ab(5)), v(ab(6))],
    )?;
    op(
        b,
        format!("v_max_num_f32_e32 v{AMAX}, v{}, v{AMAX}", ab(7)),
        &[v(AMAX)],
        &[v(ab(7)), v(AMAX)],
    )?;
    for m in [8u32, 4, 2, 1] {
        xch(b, TMP, AMAX, m)?;
        op(
            b,
            format!("v_max_num_f32_e32 v{AMAX}, v{TMP}, v{AMAX}"),
            &[v(AMAX)],
            &[v(TMP), v(AMAX)],
        )?;
    }
    op(
        b,
        format!("v_cmp_eq_f32_e64 s{ZERO}, 0, v{AMAX}"),
        &[s(ZERO)],
        &[v(AMAX)],
    )?;
    sop(b, format!("s_not_b32 s{NZ}, s{ZERO}"), &[NZ], &[ZERO])?;
    // base = (amax / 7) * 0.5 (1.0 for a zero block: its search is masked off).
    fdiv(b, BASE, AMAX, Opd::S(C7))?;
    op(
        b,
        format!("v_mul_f32_e32 v{BASE}, 0.5, v{BASE}"),
        &[v(BASE)],
        &[v(BASE)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{BASE}, v{BASE}, 1.0, s{ZERO}"),
        &[v(BASE)],
        &[v(BASE), s(ZERO)],
    )?;
    op(b, format!("v_mov_b32_e32 v{BESTD}, 1.0"), &[v(BESTD)], &[])?;
    op(
        b,
        format!("v_mov_b32_e32 v{BESTM}, {}", lit(1.0e30f32.to_bits())),
        &[v(BESTM)],
        &[],
    )?;
    for j in 0..8u32 {
        op(
            b,
            format!(
                "v_mul_f32_e32 v{DJ}, {}, v{BASE}",
                float_operand(candidate_multiplier(j))
            ),
            &[v(DJ)],
            &[v(BASE)],
        )?;
        for k in 0..8u8 {
            neg_code(b, NQ, F + k, DJ)?;
            // err = fma(-q, dj, f): NQ already is -q.
            op(
                b,
                format!("v_fma_f32 v{ERR}, v{NQ}, v{DJ}, v{}", F + k),
                &[v(ERR)],
                &[v(NQ), v(DJ), v(F + k)],
            )?;
            let acc = if k < 4 { ACC0 } else { ACC1 };
            if k % 4 == 0 {
                op(
                    b,
                    format!("v_mul_f32_e32 v{acc}, v{ERR}, v{ERR}"),
                    &[v(acc)],
                    &[v(ERR)],
                )?;
            } else {
                op(
                    b,
                    format!("v_fmac_f32_e32 v{acc}, v{ERR}, v{ERR}"),
                    &[v(acc)],
                    &[v(acc), v(ERR)],
                )?;
            }
        }
        for m in [8u32, 4, 2, 1] {
            xch(b, TMP, ACC0, m)?;
            op(
                b,
                format!("v_add_f32_e32 v{ACC0}, v{ACC0}, v{TMP}"),
                &[v(ACC0)],
                &[v(ACC0), v(TMP)],
            )?;
            xch(b, TMP, ACC1, m)?;
            op(
                b,
                format!("v_add_f32_e32 v{ACC1}, v{ACC1}, v{TMP}"),
                &[v(ACC1)],
                &[v(ACC1), v(TMP)],
            )?;
        }
        op(
            b,
            format!("v_add_f32_e32 v{MSE}, v{ACC0}, v{ACC1}"),
            &[v(MSE)],
            &[v(ACC0), v(ACC1)],
        )?;
        op(
            b,
            format!("v_cmp_lt_f32_e64 s{LT}, v{MSE}, v{BESTM}"),
            &[s(LT)],
            &[v(MSE), v(BESTM)],
        )?;
        sop(
            b,
            format!("s_and_b32 s{TAKE}, s{LT}, s{NZ}"),
            &[TAKE],
            &[LT, NZ],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{BESTM}, v{BESTM}, v{MSE}, s{TAKE}"),
            &[v(BESTM)],
            &[v(BESTM), v(MSE), s(TAKE)],
        )?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{BESTD}, v{BESTD}, v{DJ}, s{TAKE}"),
            &[v(BESTD)],
            &[v(BESTD), v(DJ), s(TAKE)],
        )?;
    }
    // Final codes from best_d (zero blocks force 0): int code = -(-q).
    for k in 0..8u8 {
        neg_code(b, CI + k, F + k, BESTD)?;
        op(
            b,
            format!("v_cndmask_b32_e64 v{0}, v{0}, 0, s{ZERO}", CI + k),
            &[v(CI + k)],
            &[v(CI + k), s(ZERO)],
        )?;
        op(
            b,
            format!("v_cvt_i32_f32_e32 v{0}, v{0}", CI + k),
            &[v(CI + k)],
            &[v(CI + k)],
        )?;
        op(
            b,
            format!("v_sub_nc_u32_e32 v{0}, 0, v{0}", CI + k),
            &[v(CI + k)],
            &[v(CI + k)],
        )?;
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v{SUMS}, v{CI}, v{}", CI + 1),
        &[v(SUMS)],
        &[v(CI), v(CI + 1)],
    )?;
    for k in 2..8u8 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{SUMS}, v{}, v{SUMS}", CI + k),
            &[v(SUMS)],
            &[v(CI + k), v(SUMS)],
        )?;
    }
    for m in [8u32, 4, 2, 1] {
        xch(b, TMP, SUMS, m)?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{SUMS}, v{SUMS}, v{TMP}"),
            &[v(SUMS)],
            &[v(SUMS), v(TMP)],
        )?;
    }
    // Packed dword: nibble k = code k, low nibble first.
    op(
        b,
        format!("v_and_b32_e32 v{WORD}, 15, v{CI}"),
        &[v(WORD)],
        &[v(CI)],
    )?;
    for k in 1..8u8 {
        op(
            b,
            format!("v_and_b32_e32 v{TMP}, 15, v{}", CI + k),
            &[v(TMP)],
            &[v(CI + k)],
        )?;
        op(
            b,
            format!("v_lshl_or_b32 v{WORD}, v{TMP}, {}, v{WORD}", 4 * k),
            &[v(WORD)],
            &[v(TMP), v(WORD)],
        )?;
    }
    Ok(())
}

/// Every lane stores its qs dword at record byte `8 + 4j`; lane `j == 0` of each half stores `d` and `s`.
fn store_record(b: &mut Builder, export: Export) -> R {
    store_offsets(b)?;
    gstore_b32(b, VQS, WORD, SCP, 8)?;
    op(b, format!("v_cmpx_eq_u32_e32 0, v{VJ}"), &[], &[v(VJ)])?;
    gstore_b32(b, HOFF, BESTD, SCP, 0)?;
    gstore_b32(b, HOFF, SUMS, SCP, 4)?;
    b.wait_all()?;
    b.label(export.exit())?;
    branch(b, "s_endpgm")
}

// ------------------------------------------------------------------ emission
pub fn emit(arch: Arch, export: Export) -> Result<Emitted, String> {
    admit(arch)?;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("x", 0, Access::ReadOnly)
        .pointer_access("rotated", 8, Access::WriteOnly)
        .pointer_access("sidecar", 16, Access::WriteOnly)
        .pointer_access("signs1", 24, Access::ReadOnly)
        .pointer_access("signs2", 32, Access::ReadOnly)
        .pointer_access("weight", 40, Access::ReadOnly)
        .pointer_access("awq", 48, Access::ReadOnly)
        .hidden("rows", 56, 4, "by_value")
        .hidden("k", 60, 4, "by_value")
        .hidden("eps", 64, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: KERNEL_ID.into(),
        variant: export.tag().into(),
        arch,
        symbol: export.symbol().into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: export.block() as u16,
        group_segment_fixed_size: if export == Export::Norm { 32 } else { 0 },
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    if export == Export::Norm && b.lds_slot("rms_partials", 0, 32)? != 0 {
        return Err("LDS slot order".into());
    }
    prologue(&mut b, export)?;
    match export {
        Export::Rotate => rotate_front(&mut b)?,
        Export::Norm => norm_front(&mut b)?,
    }
    fwht_scale_store(&mut b)?;
    quantize(&mut b)?;
    store_record(&mut b, export)?;
    b.finish()
}

/// The module as one code object (`MODULE`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    admit(arch)?;
    let emitted = module_exports()
        .into_iter()
        .map(|e| emit(arch, e))
        .collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

// ------------------------------------------------------------------ contract
const COUNTED: [&str; 31] = [
    "s_load_b256",
    "s_load_b32",
    "global_load_b32",
    "global_load_b128",
    "global_store_b32",
    "buffer_store_b128",
    "ds_store_b32",
    "ds_load_b128",
    "s_barrier_signal",
    "s_barrier_wait",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_rcp_f32_e32",
    "v_rsq_f32_e32",
    "v_rndne_f32_e32",
    "v_max_num_f32_e32",
    "v_max3_num_f32",
    "v_fma_f32",
    "v_fmac_f32_e32",
    "v_mov_b32_dpp",
    "v_permlanex16_b32",
    "v_cvt_i32_f32_e32",
    "v_cndmask_b32_e64",
    "v_cmp_lt_f32_e64",
    "v_cmp_eq_f32_e64",
    "v_cmp_gt_f32_e64",
    "v_cmpx_eq_u32_e32",
    "s_cselect_b64",
    "s_cbranch_scc1",
    "s_endpgm",
];

/// Shape contract of one export (`toolchain::IsaShapeContract`): counts derived from the emission, the
/// forbidden families, register ceilings, wave32, no spills, no private memory. Rotate has no DS or barrier;
/// the norm has exactly one 32-byte LDS slot, `ds_store_b32` / `ds_load_b128` and one split barrier.
pub fn contract(arch: Arch, e: &Emitted, export: Export) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in COUNTED {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.split_whitespace().next() == Some(name))
            .count();
        if n > 0 {
            counts.insert(name.into(), n.into());
        }
    }
    let _ = arch;
    let mut forbidden = vec![
        "scratch_*",
        "flat_*",
        "global_atomic*",
        "buffer_load*",
        "buffer_atomic*",
        "v_readlane*",
        "v_writelane*",
        "v_wmma*",
        "v_swmmac*",
        "v_fma_mix*",
        "s_waitcnt*",
        "s_setpc*",
        "s_swappc*",
        "s_branch*",
        "s_cbranch_vcc*",
        "s_cbranch_execz*",
        "s_cbranch_execnz*",
        "s_cbranch_scc0*",
        "ds_bpermute*",
        "ds_swizzle*",
        "v_s_rsq*",
        "s_add_f32",
        "s_mul_f32",
    ];
    if export == Export::Rotate {
        forbidden.extend(["ds_*", "s_barrier*"]);
    } else {
        forbidden.extend([
            "ds_load_b32",
            "ds_load_b64",
            "ds_store_b64",
            "ds_store_b128",
            "ds_add*",
        ]);
    }
    serde_json::json!({
        "symbol": export.symbol(), "counts": counts, "forbidden": forbidden,
        "vgpr_max": VGPRS, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": if export == Export::Norm { 32 } else { 0 },
    })
}
