// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151 multi-column row kernels of the Flash-Next lane step, MQ6 half:
//! `gemv_mq6g256v2_xbatch_pm` and `gemv_mq6g256v2_x4_xbatch_pm`.
//!
//! Every column `c` of `y[c*M + row]` is byte-identical to the 1-row
//! incumbent `gemv_mq6g256v2` on `x[c*K..]` (and to the incumbent rows kernel's
//! column `c`); `fn-batch/s7/DAG.md` §1 is the arithmetic spec. Each
//! workgroup (one wave32) owns four weight rows and keeps one register
//! ring of activation loads that all four rows share, which is what bounds
//! the incumbents (they re-read the activations per weight row).
//!
//! ABI (frozen by `fn-batch/S7-CONTRACT.md`), x is F32 `[B][K]`, y is F32
//! `[B][M]`, `B` in 1..=8, `K % 256 == 0`, block 32, no LDS, no scratch:
//! - `gemv_mq6g256v2_xbatch_pm`: `A@0 x@8 y@16 M@24 K@28 B@32` (36 B), grid
//!   `ceil(M/4)`; workgroup `w` owns rows `4w..4w+3` (loads clamp to `M-1`,
//!   only valid rows are stored).
//! - `gemv_mq6g256v2_x4_xbatch_pm`: `A0..A3@0..24 x@32 y0..y3@40..64
//!   M0..M3@72..84 K@88 B@92` (96 B), grid `sum ceil(Mi/4)`; workgroup `w`
//!   maps to matrix `i` and local row `4*(w - sum_{j<i} ceil(Mj/4))`, then runs
//!   the same body (`Mi <= 0` owns no workgroups).
//!
//! Per-matrix byte sizes must stay below 4 GiB: each row's byte offset is a
//! 32-bit buffer soffset against a raw, unbounded buffer descriptor.
//!
//! Arithmetic per (row, column) (`DAG.md` §1):
//! - Group `g` (200 B: half headers `h0` for lanes 0-15 / `h1` for lanes 16-31
//!   in bytes 0..7, then six payload bytes per lane at `8 + 6*lane`): lane
//!   `l` covers weights `8l..8l+7`. The payload is read as a dword at `8+6l`
//!   plus a zero-extended word at `12+6l` (`p` = bytes b0..b3, `u` = bytes
//!   b4,b5), and `q0 = p & 63`, `q1..q4 = bfe(p, 6/12/18/24, 6)`,
//!   `q5 = alignbit(u, p, 30) & 63`, `q6 = bfe(u, 4, 6)`, `q7 = bfe(u, 10, 6)`
//!   are the unsigned values of `DAG.md` §1 (`b0&63`, `(b0>>6)|((b1&15)<<2)`,
//!   ...); `w_i = fma_mix(sc_f16, f32(q_i), zp_f16)` with `op_sel:[0,0,1]
//!   op_sel_hi:[1,0,1]` (one rounding), `f32(q_i)` from `v_cvt_f32_ubyte0`.
//! - The group dot is `d = x1*w1` (`v_mul`) and then `fma(x0,w0,d)`,
//!   `fma(x2,w2,d)` .. `fma(x7,w7,d)` in that fixed order (`v_fmac`, one
//!   rounding each); multiplicand order is `w, x` for every column.
//! - Group `g` adds into stream `g % 4` as `a_s = a_s + d` in increasing
//!   `g` (quads in a loop, then up to three tail groups into streams
//!   0..2); the four streams start at +0 and fold as `(a0 + a1) + (a3 + a2)`.
//! - Lane tree: `ds_swizzle_b32` offset 16 (`"1pppp"`: lanes 16..31 read
//!   themselves) then `ds_bpermute_b32` at lane + 8/4/2/1 (own lane when
//!   out of range), `own + other` at every level; lane 0 stores.
//!
//! Instruction selection: columns are paired two at a time so the two halves
//! of a `v_dual_*` packet read different weight rows (src0 banks differ) and
//! different activation slots (src1 banks differ: ring slots sit 9 VGPRs
//! apart), which is what gfx11's VOPD rules require; a lone odd column runs
//! plain VALU. Every VOPD pair is validated by `vopd::validate_pair` while
//! the kernel is built.
//!
//! Wide twins (`build_wide_gfx1151`, module `wide`): `gemv_mq6g256v2_xbatch_wide_r{16,32}_pm`
//! and `gemv_mq6g256v2_x4_xbatch_wide_r{16,32}_pm`, appended to the module after
//! every narrow export (the narrow exports above are unchanged). They keep the
//! narrow ABIs byte for byte (36 B `A@0 x@8 y@16 M@24 K@28 R@32`; 96 B x4 with
//! `R@92`), but `R` is the actual row count `1..=cap` (any other value returns
//! before the first barrier and stores nothing). Grid `ceil(M/8)` /
//! `sum ceil(Mi/8)`, block 256 (eight wave32, `cu_mode = false`), static LDS
//! 8192 B. Wave `v` owns matrix-local output column `8*(w - base) + v` for all
//! `R` rows: it loads/dequantizes its weight group once, keeps it in VGPRs for
//! every 8-row X tile of the group and never reloads it inside the row tiles.
//! X is staged through LDS in 8-row x K256 tiles with two uniform workgroup
//! barriers per tile; waves past `M` still produce X and run every barrier and
//! skip only the weights, arithmetic and stores. The per-(row, column)
//! arithmetic is exactly the DAG above (term order x1 mul, x0, x2..x7; a
//! separate stream add; `(a0+a1)+(a3+a2)`; swizzle-16 then bpermute 8/4/2/1,
//! own + other); only the schedule differs. See `wide` for the register
//! ceilings (128 / 192 VGPRs) and the checked high-water arithmetic.
use super::common::smem_off;
use crate::insn::{Instruction, MemoryClass};
use crate::plan::Access;
use crate::reg::{Kind, Live, RegRef};
use crate::vopd::{Operand, VopdF32, VopdOp};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

pub const SYMBOL: &str = "gemv_mq6g256v2_xbatch_pm";
pub const SYMBOL_X4: &str = "gemv_mq6g256v2_x4_xbatch_pm";
pub const MAX_COLUMNS: u8 = 8;

/// MQ6 group bytes and the four rows per workgroup.
const GROUP: u32 = 200;
const VGPRS: u16 = 256;
const SGPRS: u16 = 88;
/// Group-dot term order: x1 first (`v_mul`), then x0, x2 .. x7.
const TERMS: [u8; 8] = [1, 0, 2, 3, 4, 5, 6, 7];

// VGPRs.
const LANE: u8 = 0;
/// `(lane >> 4) * 4`: the lane's half-header byte offset.
const HDR_OFF: u8 = 1;
/// `lane * 6`: the lane's payload byte offset (plus 8 immediate).
const PK_OFF: u8 = 2;
/// `lane * 32`: the lane's eight activations.
const X_OFF: u8 = 3;
const SHUF: u8 = 4;
/// Raw loads of (buffer, row): header, payload dword, payload u16 (3 each;
/// two buffers alternate between the groups of a body).
fn raw(buf: u8, r: u8, k: u8) -> u8 {
    8 + 12 * buf + 3 * r + k
}
/// Dequantized weights of row r (bases differ by 2 mod 4 within a row pair).
const W: [u8; 4] = [32, 42, 52, 62];
/// Group-dot registers (column slot, row): even rows even, odd rows odd, and
/// `DOT[s][r] % 4 == r`, which makes every VOPD pairing of the add phase legal.
const DOT: [[u8; 4]; 2] = [[40, 41, 50, 51], [60, 61, 70, 71]];
const ACC0: u16 = 72;
/// Stream accumulator (column c, stream s, row r): parity `r % 2`, bank `r % 4`.
fn acc(c: u8, s: u8, r: u8) -> u8 {
    (ACC0 + 16 * u16::from(c) + 4 * u16::from(s) + u16::from(r)) as u8
}
fn ring_base(nb: u8) -> u16 {
    ACC0 + 16 * u16::from(nb)
}
/// Activation ring slots (8 VGPRs at a 9-VGPR pitch) of the `nb`-column
/// path: every free register after the accumulators, at most eight. The
/// pitch makes consecutive items' slots differ in bank (mod 4) even across
/// the wrap, so any two consecutive items form a legal VOPD src1 pair.
fn slots(nb: u8) -> u16 {
    ((VGPRS - ring_base(nb) - 8) / 9 + 1).min(8)
}
fn ring(nb: u8, item: u16) -> u8 {
    (ring_base(nb) + 9 * (item % slots(nb))) as u8
}
/// One past the highest VGPR any column count touches.
fn vgpr_high() -> u16 {
    (1..=MAX_COLUMNS)
        .map(|nb| ring_base(nb) + 9 * (slots(nb) - 1) + 8)
        .max()
        .unwrap_or(0)
}
// Epilogue registers (raw loads, weights and the ring are dead by then).
fn red_tmp(c: u8, r: u8) -> u8 {
    8 + 4 * c + r
}
fn y_addr(c: u8) -> u8 {
    40 + c
}

// SGPRs. s0:1 kernarg, s2 workgroup id x (`user_sgpr_count = 2`).
const S_WG: u8 = 2;
const S_B: u8 = 3;
const S_A: u8 = 4;
const S_X: u8 = 6;
const S_Y: u8 = 8;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_NV: u8 = 13;
const S_GROUPS: u8 = 14;
const S_STRIDE: u8 = 15;
/// Weight soffset of row r (row offset plus the current quad base).
fn s_row(r: u8) -> u8 {
    16 + r
}
const S_ASRD: u8 = 20;
const S_XSRD: u8 = 24;
/// Activation soffset of column c (column offset plus the current quad base).
fn s_xcol(c: u8) -> u8 {
    28 + c
}
const S_QUADS: u8 = 36;
const S_TAIL: u8 = 37;
const S_KX4: u8 = 38;
const S_MX4: u8 = 39;
const S_T0: u8 = 40;
const S_T1: u8 = 41;
const S_ROW0X4: u8 = 42;
// x4 routing (kernargs of the four matrices, loaded whole).
const R_A: u8 = 48;
const R_X: u8 = 56;
const R_Y: u8 = 64;
const R_M: u8 = 72;
const R_K: u8 = 76;
const R_B: u8 = 77;
/// Cumulative block ends of matrices 0..3 (s78..s81) and a temporary.
const R_END: u8 = 78;
const R_TMP: u8 = 82;

/// Both exports, in contract order.
pub fn build_gfx1151() -> Result<Vec<Emitted>, String> {
    Ok(vec![kernel(false)?, kernel(true)?])
}

/// Declare registers `0..end` in aligned 8/4/2/1-register pieces.
fn declare(regs: &mut RegPlan, kind: Kind, end: u16) -> Result<(), String> {
    let (prefix, mut at) = (if kind == Kind::V { "v" } else { "s" }, 0u16);
    while at < end {
        let piece = [8u16, 4, 2, 1]
            .into_iter()
            .find(|p| at % p == 0 && at + p <= end)
            .unwrap_or(1);
        regs.add_range(
            &format!("{prefix}{at}"),
            kind,
            at as u8,
            piece as u8,
            Live::Whole,
        )?;
        at += piece;
    }
    Ok(())
}

fn kernel(x4: bool) -> Result<Emitted, String> {
    let mut regs = RegPlan::new(VGPRS, SGPRS)?;
    declare(&mut regs, Kind::V, vgpr_high())?;
    declare(
        &mut regs,
        Kind::S,
        if x4 {
            u16::from(R_TMP) + 1
        } else {
            u16::from(S_ROW0X4) + 1
        },
    )?;
    let (symbol, variant, prefix, kernargs) = if x4 {
        let mut k = KernargLayout::new(96);
        for (i, name) in ["A0", "A1", "A2", "A3"].into_iter().enumerate() {
            k = k.pointer_access(name, 8 * i as u32, Access::ReadOnly);
        }
        k = k.pointer_access("x", 32, Access::ReadOnly);
        for (i, name) in ["y0", "y1", "y2", "y3"].into_iter().enumerate() {
            k = k.pointer_access(name, 40 + 8 * i as u32, Access::WriteOnly);
        }
        for (i, name) in ["M0", "M1", "M2", "M3"].into_iter().enumerate() {
            k = k.hidden(name, 72 + 4 * i as u32, 4, "by_value");
        }
        (
            SYMBOL_X4,
            "x4_xbatch_r4_b8",
            "mq6x4",
            k.hidden("K", 88, 4, "by_value")
                .hidden("B", 92, 4, "by_value"),
        )
    } else {
        (
            SYMBOL,
            "xbatch_r4_b8",
            "mq6",
            KernargLayout::new(36)
                .pointer_access("A", 0, Access::ReadOnly)
                .pointer_access("x", 8, Access::ReadOnly)
                .pointer_access("y", 16, Access::WriteOnly)
                .hidden("M", 24, 4, "by_value")
                .hidden("K", 28, 4, "by_value")
                .hidden("B", 32, 4, "by_value"),
        )
    };
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "qwen4_lane_rows_mq6".into(),
            variant: variant.into(),
            arch: Arch::Gfx1151,
            symbol: symbol.into(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 32,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    if x4 {
        route(&mut b, prefix)?;
    } else {
        select(&mut b, prefix)?;
    }
    setup(&mut b)?;
    // One specialized path per column count; each section ends in `s_endpgm`,
    // so every branch stays inside one section.
    for nb in 1..=MAX_COLUMNS {
        salu(&mut b, &format!("s_cmp_eq_u32 s{S_B}, {nb}"), &[], &[S_B])?;
        branch(&mut b, &format!("s_cbranch_scc0 .L{prefix}_skip{nb}"))?;
        section(&mut b, prefix, nb)?;
        b.label(&format!(".L{prefix}_skip{nb}"))?;
    }
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// Single-matrix workgroup row block and the early exit.
fn select(b: &mut Builder, prefix: &str) -> Result<(), String> {
    smem_load(b, S_M, 2, 0x18)?;
    salu(
        b,
        &format!("s_lshl_b32 s{S_ROW0}, s{S_WG}, 2"),
        &[S_ROW0],
        &[S_WG],
    )?;
    b.wait_all()?;
    salu(
        b,
        &format!("s_cmp_ge_i32 s{S_ROW0}, s{S_M}"),
        &[],
        &[S_ROW0, S_M],
    )?;
    exit_unless(b, prefix, "go")?;
    smem_load(b, S_A, 4, 0)?;
    smem_load(b, S_Y, 2, 0x10)?;
    smem_load(b, S_B, 1, 0x20)
}

/// `s_cbranch_scc0` over a lone `s_endpgm`: leaves the wave when SCC is set.
fn exit_unless(b: &mut Builder, prefix: &str, name: &str) -> Result<(), String> {
    branch(b, &format!("s_cbranch_scc0 .L{prefix}_{name}"))?;
    branch(b, "s_endpgm")?;
    b.label(&format!(".L{prefix}_{name}"))
}

/// x4: workgroup `w` -> matrix `i` (the smallest `i` with `w < end_i`, `end_i`
/// the prefix sum of `ceil(max(Mi,0)/4)`), its local row block and its A/y/M.
fn route(b: &mut Builder, prefix: &str) -> Result<(), String> {
    smem_load(b, R_M, 4, 0x48)?;
    smem_load(b, R_K, 2, 0x58)?;
    smem_load(b, R_A, 8, 0)?;
    smem_load(b, R_X, 2, 0x20)?;
    smem_load(b, R_Y, 8, 0x28)?;
    b.wait_all()?;
    for i in 0..4u8 {
        let (m, end) = (R_M + i, R_END + i);
        salu(b, &format!("s_max_i32 s{end}, s{m}, 0"), &[end], &[m])?;
        salu(b, &format!("s_add_i32 s{end}, s{end}, 3"), &[end], &[end])?;
        salu(b, &format!("s_lshr_b32 s{end}, s{end}, 2"), &[end], &[end])?;
        if i > 0 {
            salu(
                b,
                &format!("s_add_i32 s{end}, s{end}, s{}", end - 1),
                &[end],
                &[end, end - 1],
            )?;
        }
    }
    salu(
        b,
        &format!("s_cmp_ge_u32 s{S_WG}, s{}", R_END + 3),
        &[],
        &[S_WG, R_END + 3],
    )?;
    exit_unless(b, prefix, "route")?;
    // Matrix 3, then matrices 2, 1, 0 override it while `w < end_i`.
    pair(b, "s_mov_b64", S_A, R_A + 6)?;
    pair(b, "s_mov_b64", S_Y, R_Y + 6)?;
    salu(
        b,
        &format!("s_mov_b32 s{S_M}, s{}", R_M + 3),
        &[S_M],
        &[R_M + 3],
    )?;
    salu(
        b,
        &format!("s_sub_i32 s{S_ROW0}, s{S_WG}, s{}", R_END + 2),
        &[S_ROW0],
        &[S_WG, R_END + 2],
    )?;
    for i in (0..3u8).rev() {
        let local = if i == 0 {
            S_WG
        } else {
            salu(
                b,
                &format!("s_sub_i32 s{R_TMP}, s{S_WG}, s{}", R_END + i - 1),
                &[R_TMP],
                &[S_WG, R_END + i - 1],
            )?;
            R_TMP
        };
        salu(
            b,
            &format!("s_cmp_lt_u32 s{S_WG}, s{}", R_END + i),
            &[],
            &[S_WG, R_END + i],
        )?;
        pair(b, "s_cselect_b64", S_A, R_A + 2 * i)?;
        pair(b, "s_cselect_b64", S_Y, R_Y + 2 * i)?;
        salu(
            b,
            &format!("s_cselect_b32 s{S_M}, s{}, s{S_M}", R_M + i),
            &[S_M],
            &[R_M + i, S_M],
        )?;
        salu(
            b,
            &format!("s_cselect_b32 s{S_ROW0}, s{local}, s{S_ROW0}"),
            &[S_ROW0],
            &[local, S_ROW0],
        )?;
    }
    salu(
        b,
        &format!("s_lshl_b32 s{S_ROW0}, s{S_ROW0}, 2"),
        &[S_ROW0],
        &[S_ROW0],
    )?;
    pair(b, "s_mov_b64", S_X, R_X)?;
    salu(b, &format!("s_mov_b32 s{S_K}, s{R_K}"), &[S_K], &[R_K])?;
    salu(b, &format!("s_mov_b32 s{S_B}, s{R_B}"), &[S_B], &[R_B])
}

/// `op dst_pair, src_pair` over 64-bit SGPR pairs; `s_cselect_b64` also keeps
/// the pair it does not select.
fn pair(b: &mut Builder, op: &str, dst: u8, src: u8) -> Result<(), String> {
    let text = if op == "s_cselect_b64" {
        format!(
            "{op} s[{dst}:{}], s[{src}:{}], s[{dst}:{}]",
            dst + 1,
            src + 1,
            dst + 1
        )
    } else {
        format!("{op} s[{dst}:{}], s[{src}:{}]", dst + 1, src + 1)
    };
    let uses: &[u8] = if op == "s_cselect_b64" {
        &[src, src + 1, dst, dst + 1]
    } else {
        &[src, src + 1]
    };
    salu(b, &text, &[dst, dst + 1], uses)
}

/// Shared scalar/vector setup: valid-row count, row soffsets, lane offsets
/// and the two raw buffer resources.
fn setup(b: &mut Builder) -> Result<(), String> {
    salu(
        b,
        &format!("s_sub_i32 s{S_NV}, s{S_M}, s{S_ROW0}"),
        &[S_NV],
        &[S_M, S_ROW0],
    )?;
    salu(
        b,
        &format!("s_min_i32 s{S_NV}, s{S_NV}, 4"),
        &[S_NV],
        &[S_NV],
    )?;
    salu(
        b,
        &format!("s_add_i32 s{S_T0}, s{S_M}, -1"),
        &[S_T0],
        &[S_M],
    )?;
    salu(
        b,
        &format!("s_lshr_b32 s{S_GROUPS}, s{S_K}, 8"),
        &[S_GROUPS],
        &[S_K],
    )?;
    salu(
        b,
        &format!("s_mul_i32 s{S_STRIDE}, s{S_GROUPS}, {GROUP:#x}"),
        &[S_STRIDE],
        &[S_GROUPS],
    )?;
    for r in 0..4u8 {
        salu(
            b,
            &format!("s_add_i32 s{S_T1}, s{S_ROW0}, {r}"),
            &[S_T1],
            &[S_ROW0],
        )?;
        salu(
            b,
            &format!("s_min_i32 s{S_T1}, s{S_T1}, s{S_T0}"),
            &[S_T1],
            &[S_T1, S_T0],
        )?;
        salu(
            b,
            &format!("s_mul_i32 s{}, s{S_T1}, s{S_STRIDE}", s_row(r)),
            &[s_row(r)],
            &[S_T1, S_STRIDE],
        )?;
    }
    salu(
        b,
        &format!("s_lshl_b32 s{S_KX4}, s{S_K}, 2"),
        &[S_KX4],
        &[S_K],
    )?;
    salu(
        b,
        &format!("s_lshl_b32 s{S_MX4}, s{S_M}, 2"),
        &[S_MX4],
        &[S_M],
    )?;
    salu(
        b,
        &format!("s_lshl_b32 s{S_ROW0X4}, s{S_ROW0}, 2"),
        &[S_ROW0X4],
        &[S_ROW0],
    )?;
    for c in 0..MAX_COLUMNS {
        salu(
            b,
            &format!("s_mul_i32 s{}, s{S_KX4}, {c}", s_xcol(c)),
            &[s_xcol(c)],
            &[S_KX4],
        )?;
    }
    valu(
        b,
        &format!("v_lshrrev_b32_e32 v{HDR_OFF}, 4, v{LANE}"),
        &[HDR_OFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HDR_OFF}, 2, v{HDR_OFF}"),
        &[HDR_OFF],
        &[HDR_OFF],
        &[],
    )?;
    valu(
        b,
        &format!("v_mul_u32_u24_e32 v{PK_OFF}, 6, v{LANE}"),
        &[PK_OFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{X_OFF}, 5, v{LANE}"),
        &[X_OFF],
        &[LANE],
        &[],
    )?;
    b.wait_all()?;
    // Raw buffer resources (stride 0, unbounded records) over A and x.
    for (res, ptr) in [(S_ASRD, S_A), (S_XSRD, S_X)] {
        salu(b, &format!("s_mov_b32 s{res}, s{ptr}"), &[res], &[ptr])?;
        salu(
            b,
            &format!("s_and_b32 s{}, s{}, 0xffff", res + 1, ptr + 1),
            &[res + 1],
            &[ptr + 1],
        )?;
        salu(b, &format!("s_mov_b32 s{}, -1", res + 2), &[res + 2], &[])?;
        salu(
            b,
            &format!("s_mov_b32 s{}, {:#x}", res + 3, super::common::SRD_WORD3),
            &[res + 3],
            &[],
        )?;
    }
    Ok(())
}

/// The complete kernel body for `nb` columns (ends in `s_endpgm`).
fn section(b: &mut Builder, prefix: &str, nb: u8) -> Result<(), String> {
    for c in 0..nb {
        for s in 0..4 {
            for r in 0..4 {
                let a = acc(c, s, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    salu(
        b,
        &format!("s_lshr_b32 s{S_QUADS}, s{S_GROUPS}, 2"),
        &[S_QUADS],
        &[S_GROUPS],
    )?;
    salu(
        b,
        &format!("s_and_b32 s{S_TAIL}, s{S_GROUPS}, 3"),
        &[S_TAIL],
        &[S_GROUPS],
    )?;
    salu(b, &format!("s_cmp_eq_u32 s{S_QUADS}, 0"), &[], &[S_QUADS])?;
    branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_tail"))?;
    b.loop_(&format!(".L{prefix}_b{nb}_quad"), |b| {
        body(b, nb, &[0, 1, 2, 3])?;
        for r in 0..4 {
            salu(
                b,
                &format!("s_add_i32 s{0}, s{0}, {1:#x}", s_row(r), 4 * GROUP),
                &[s_row(r)],
                &[s_row(r)],
            )?;
        }
        for c in 0..nb {
            salu(
                b,
                &format!("s_add_i32 s{0}, s{0}, 0x1000", s_xcol(c)),
                &[s_xcol(c)],
                &[s_xcol(c)],
            )?;
        }
        salu(
            b,
            &format!("s_add_i32 s{S_QUADS}, s{S_QUADS}, -1"),
            &[S_QUADS],
            &[S_QUADS],
        )?;
        salu(b, &format!("s_cmp_lg_u32 s{S_QUADS}, 0"), &[], &[S_QUADS])?;
        branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_quad"))
    })?;
    b.label(&format!(".L{prefix}_b{nb}_tail"))?;
    // Tail group t (at the post-quad bases) accumulates into stream t.
    for t in 0..3u8 {
        salu(b, &format!("s_cmp_gt_u32 s{S_TAIL}, {t}"), &[], &[S_TAIL])?;
        branch(b, &format!("s_cbranch_scc0 .L{prefix}_b{nb}_fold"))?;
        body(b, nb, &[t])?;
    }
    b.label(&format!(".L{prefix}_b{nb}_fold"))?;
    fold(b, nb)?;
    reduce(b, nb)?;
    store(b, prefix, nb)
}

/// Groups `js` (slot = stream = immediate index) of the four rows against
/// every column. The first two groups' weights and the first activations
/// issue up front; the weights of group `i + 2` reuse the raw registers
/// group `i` just dequantized, and activations of later items stream
/// through the ring as earlier items retire.
fn body(b: &mut Builder, nb: u8, js: &[u8]) -> Result<(), String> {
    let items = js.len() as u16 * u16::from(nb);
    let ring_slots = slots(nb);
    weight_loads(b, js[0], 0)?;
    let mut issued = 0u16;
    while issued < ring_slots.min(items) {
        x_load(b, nb, js, issued)?;
        issued += 1;
    }
    if js.len() > 1 {
        weight_loads(b, js[1], 1)?;
    }
    for (index, &j) in js.iter().enumerate() {
        let buf = (index % 2) as u8;
        dequant(b, buf)?;
        if let Some(&next) = js.get(index + 2) {
            weight_loads(b, next, buf)?;
        }
        let mut c = 0u8;
        while c < nb {
            let item = index as u16 * u16::from(nb) + u16::from(c);
            while issued < (item + ring_slots).min(items) {
                x_load(b, nb, js, issued)?;
                issued += 1;
            }
            let cols: Vec<(u8, u8)> = (c..nb.min(c + 2))
                .map(|col| (col, ring(nb, item + u16::from(col - c))))
                .collect();
            columns(b, j, &cols)?;
            c += 2;
        }
    }
    b.wait_all()
}

/// Payload dwords, payload u16s and lane-selected half headers of group
/// slot `j`, all four rows, into raw buffer `buf`.
fn weight_loads(b: &mut Builder, j: u8, buf: u8) -> Result<(), String> {
    let off = u32::from(j) * GROUP;
    let srd = [S_ASRD, S_ASRD + 1, S_ASRD + 2, S_ASRD + 3];
    for r in 0..4u8 {
        let p = raw(buf, r, 1);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{p}, v{PK_OFF}, s[{S_ASRD}:{}], s{} offen offset:{}",
                S_ASRD + 3,
                s_row(r),
                off + 8
            ),
            &[p],
            &[PK_OFF],
            &scalar,
            false,
        )?;
    }
    for r in 0..4u8 {
        let u = raw(buf, r, 2);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        vmem(
            b,
            &format!(
                "buffer_load_u16 v{u}, v{PK_OFF}, s[{S_ASRD}:{}], s{} offen offset:{}",
                S_ASRD + 3,
                s_row(r),
                off + 12
            ),
            &[u],
            &[PK_OFF],
            &scalar,
            false,
        )?;
    }
    for r in 0..4u8 {
        let h = raw(buf, r, 0);
        let mut scalar = srd.to_vec();
        scalar.push(s_row(r));
        let suffix = if off == 0 {
            String::new()
        } else {
            format!(" offset:{off}")
        };
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{h}, v{HDR_OFF}, s[{S_ASRD}:{}], s{} offen{suffix}",
                S_ASRD + 3,
                s_row(r)
            ),
            &[h],
            &[HDR_OFF],
            &scalar,
            false,
        )?;
    }
    Ok(())
}

/// The eight activations of item `item` (group `js[item / nb]`, column
/// `item % nb`) into its ring slot.
fn x_load(b: &mut Builder, nb: u8, js: &[u8], item: u16) -> Result<(), String> {
    let j = js[usize::from(item / u16::from(nb))];
    let c = (item % u16::from(nb)) as u8;
    let x = ring(nb, item);
    for half in 0..2u8 {
        let off = u32::from(j) * 1024 + u32::from(half) * 16;
        let suffix = if off == 0 {
            String::new()
        } else {
            format!(" offset:{off}")
        };
        let d = x + 4 * half;
        vmem(
            b,
            &format!(
                "buffer_load_b128 v[{d}:{}], v{X_OFF}, s[{S_XSRD}:{}], s{} offen{suffix}",
                d + 3,
                S_XSRD + 3,
                s_xcol(c)
            ),
            &[d, d + 1, d + 2, d + 3],
            &[X_OFF],
            &[S_XSRD, S_XSRD + 1, S_XSRD + 2, S_XSRD + 3, s_xcol(c)],
            false,
        )?;
    }
    Ok(())
}

/// `w[r][n] = fma_mix(sc, f32(q_n), zp)` for the four rows of the group in
/// raw buffer `buf`, extracting `q0..q7` from the payload dword `p` and word
/// `u` exactly as `DAG.md` §1 (unsigned 6-bit fields; `q5` straddles `p` and `u`).
fn dequant(b: &mut Builder, buf: u8) -> Result<(), String> {
    let rows = |f: &mut dyn FnMut(u8, u8, u8, u8) -> Result<(), String>| -> Result<(), String> {
        for r in 0..4u8 {
            f(
                W[usize::from(r)],
                raw(buf, r, 0),
                raw(buf, r, 1),
                raw(buf, r, 2),
            )?;
        }
        Ok(())
    };
    rows(&mut |w, _, p, _| valu(b, &format!("v_and_b32_e32 v{w}, 63, v{p}"), &[w], &[p], &[]))?;
    for (n, lsb) in [(1u8, 6u32), (2, 12), (3, 18), (4, 24)] {
        rows(&mut |w, _, p, _| {
            valu(
                b,
                &format!("v_bfe_u32 v{}, v{p}, {lsb}, 6", w + n),
                &[w + n],
                &[p],
                &[],
            )
        })?;
    }
    rows(&mut |w, _, p, u| {
        valu(
            b,
            &format!("v_alignbit_b32 v{}, v{u}, v{p}, 30", w + 5),
            &[w + 5],
            &[u, p],
            &[],
        )
    })?;
    rows(&mut |w, _, _, _| {
        valu(
            b,
            &format!("v_and_b32_e32 v{0}, 63, v{0}", w + 5),
            &[w + 5],
            &[w + 5],
            &[],
        )
    })?;
    rows(&mut |w, _, _, u| {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{u}, 4, 6", w + 6),
            &[w + 6],
            &[u],
            &[],
        )
    })?;
    rows(&mut |w, _, _, u| {
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{u}, 10, 6", w + 7),
            &[w + 7],
            &[u],
            &[],
        )
    })?;
    for n in 0..8u8 {
        rows(&mut |w, _, _, _| {
            valu(
                b,
                &format!("v_cvt_f32_ubyte0_e32 v{0}, v{0}", w + n),
                &[w + n],
                &[w + n],
                &[],
            )
        })?;
    }
    for n in 0..8u8 {
        rows(&mut |w, h, _, _| {
            let d = w + n;
            valu(
                b,
                &format!("v_fma_mix_f32 v{d}, v{h}, v{d}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[d],
                &[h, d],
                &[],
            )
        })?;
    }
    Ok(())
}

/// Dot the four dequantized rows against one or two columns (each `(column,
/// ring slot)`) and add the dots into stream `s`. Terms run x1, x0, x2..x7.
/// Two columns: each packet pairs (row rx, column 0) with (row ry, column 1)
/// for `rx != ry` of one row pair, so the src0 banks (weights) differ by
/// construction, the src1 banks (ring slots) differ by the slot pitch and the
/// destinations alternate parity by row.
fn columns(b: &mut Builder, s: u8, cols: &[(u8, u8)]) -> Result<(), String> {
    let row_pairs = [(0u8, 1u8), (2, 3)];
    for (term, n) in TERMS.into_iter().enumerate() {
        let op = if term == 0 {
            VopdF32::Mul
        } else {
            VopdF32::Fmac
        };
        match cols {
            [(_, x0), (_, x1)] => {
                for (ra, rb) in row_pairs {
                    for (rx, ry) in [(ra, rb), (rb, ra)] {
                        let (wx, wy) = (W[usize::from(rx)] + n, W[usize::from(ry)] + n);
                        b.vopd(
                            VopdOp {
                                op,
                                dst: DOT[0][usize::from(rx)],
                                src0: Operand::V(wx),
                                src1: x0 + n,
                            },
                            VopdOp {
                                op,
                                dst: DOT[1][usize::from(ry)],
                                src0: Operand::V(wy),
                                src1: x1 + n,
                            },
                        )?;
                    }
                }
            }
            [(_, x)] => {
                for r in 0..4u8 {
                    let (d, w) = (DOT[0][usize::from(r)], W[usize::from(r)] + n);
                    if term == 0 {
                        valu(
                            b,
                            &format!("v_mul_f32_e32 v{d}, v{w}, v{}", x + n),
                            &[d],
                            &[w, x + n],
                            &[],
                        )?;
                    } else {
                        valu(
                            b,
                            &format!("v_fmac_f32_e32 v{d}, v{w}, v{}", x + n),
                            &[d],
                            &[d, w, x + n],
                            &[],
                        )?;
                    }
                }
            }
            _ => return Err("columns takes one or two columns".into()),
        }
    }
    match cols {
        [(c0, _), (c1, _)] => {
            for (ra, rb) in row_pairs {
                for (rx, ry) in [(ra, rb), (rb, ra)] {
                    let (ax, ay) = (acc(*c0, s, rx), acc(*c1, s, ry));
                    b.vopd(
                        VopdOp {
                            op: VopdF32::Add,
                            dst: ax,
                            src0: Operand::V(ax),
                            src1: DOT[0][usize::from(rx)],
                        },
                        VopdOp {
                            op: VopdF32::Add,
                            dst: ay,
                            src0: Operand::V(ay),
                            src1: DOT[1][usize::from(ry)],
                        },
                    )?;
                }
            }
        }
        [(c, _)] => {
            for r in 0..4u8 {
                let (a, d) = (acc(*c, s, r), DOT[0][usize::from(r)]);
                valu(
                    b,
                    &format!("v_add_f32_e32 v{a}, v{a}, v{d}"),
                    &[a],
                    &[a, d],
                    &[],
                )?;
            }
        }
        _ => unreachable!("checked above"),
    }
    Ok(())
}

/// `(a0 + a1) + (a3 + a2)` per column and row, into stream 0.
fn fold(b: &mut Builder, nb: u8) -> Result<(), String> {
    for (dst, other) in [(0u8, 1u8), (3, 2)] {
        for c in 0..nb {
            for r in 0..4u8 {
                let (x, y) = (acc(c, dst, r), acc(c, other, r));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{x}, v{x}, v{y}"),
                    &[x],
                    &[x, y],
                    &[],
                )?;
            }
        }
    }
    for c in 0..nb {
        for r in 0..4u8 {
            let (x, y) = (acc(c, 0, r), acc(c, 3, r));
            valu(
                b,
                &format!("v_add_f32_e32 v{x}, v{x}, v{y}"),
                &[x],
                &[x, y],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, nb: u8) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..nb)
        .flat_map(|c| (0..4u8).map(move |r| (c, r)))
        .collect();
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        b.ds_crosslane(
            Instruction::new(
                format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
                refs(Kind::V, &[t]),
                refs(Kind::V, &[v]),
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        valu(
            b,
            &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
            &[v],
            &[v, t],
            &[],
        )?;
    }
    for offset in [8u8, 4, 2, 1] {
        // Lanes whose partner is out of range read themselves.
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - offset),
            &[],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{SHUF}, 0, {offset}, vcc_lo"),
            &[SHUF],
            &[],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"),
            &[SHUF],
            &[SHUF, LANE],
            &[],
        )?;
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            b.ds_crosslane(
                Instruction::new(
                    format!("ds_bpermute_b32 v{t}, v{SHUF}, v{v}"),
                    refs(Kind::V, &[t]),
                    refs(Kind::V, &[SHUF, v]),
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            valu(
                b,
                &format!("v_add_f32_e32 v{v}, v{v}, v{t}"),
                &[v],
                &[v, t],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Lane 0 stores `y[c*M + row0 + r]` for every column and each valid row.
fn store(b: &mut Builder, prefix: &str, nb: u8) -> Result<(), String> {
    for c in 0..nb {
        let a = y_addr(c);
        if c == 0 {
            valu(
                b,
                &format!("v_mov_b32_e32 v{a}, s{S_ROW0X4}"),
                &[a],
                &[],
                &[S_ROW0X4],
            )?;
        } else {
            salu(
                b,
                &format!("s_mul_i32 s{S_T0}, s{S_MX4}, {c}"),
                &[S_T0],
                &[S_MX4],
            )?;
            salu(
                b,
                &format!("s_add_i32 s{S_T0}, s{S_T0}, s{S_ROW0X4}"),
                &[S_T0],
                &[S_T0, S_ROW0X4],
            )?;
            valu(
                b,
                &format!("v_mov_b32_e32 v{a}, s{S_T0}"),
                &[a],
                &[],
                &[S_T0],
            )?;
        }
    }
    valu(
        b,
        &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
        &[],
        &[LANE],
        &[],
    )?;
    for r in 0..4u8 {
        if r > 0 {
            salu(b, &format!("s_cmp_le_i32 s{S_NV}, {r}"), &[], &[S_NV])?;
            branch(b, &format!("s_cbranch_scc1 .L{prefix}_b{nb}_stored"))?;
        }
        for c in 0..nb {
            let (a, v) = (y_addr(c), acc(c, 0, r));
            let suffix = if r == 0 {
                String::new()
            } else {
                format!(" offset:{}", 4 * r)
            };
            vmem(
                b,
                &format!("global_store_b32 v{a}, v{v}, s[{S_Y}:{}]{suffix}", S_Y + 1),
                &[],
                &[a, v],
                &[S_Y, S_Y + 1],
                true,
            )?;
        }
    }
    b.label(&format!(".L{prefix}_b{nb}_stored"))?;
    b.wait_all()?;
    branch(b, "s_endpgm")
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<RegRef> {
    indices
        .iter()
        .map(|&base| RegRef { kind, base, len: 1 })
        .collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(
        text,
        refs(Kind::S, defs),
        refs(Kind::S, uses),
    ))
}
fn valu(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses))
}
/// `s_load_b{32,64,128,256}` of `len` dwords at kernarg byte `off`.
fn smem_load(b: &mut Builder, dst: u8, len: u8, off: u32) -> Result<(), String> {
    let name = match len {
        1 => "s_load_b32",
        2 => "s_load_b64",
        4 => "s_load_b128",
        8 => "s_load_b256",
        _ => return Err("SMEM width".into()),
    };
    let target = if len == 1 {
        format!("s{dst}")
    } else {
        format!("s[{dst}:{}]", dst + len - 1)
    };
    let defs: Vec<u8> = (dst..dst + len).collect();
    let offset = smem_off(b.spec.arch, off);
    b.push(
        Instruction::new(
            format!("{name} {target}, s[0:1], {offset}"),
            refs(Kind::S, &defs),
            refs(Kind::S, &[0, 1]),
        )
        .memory(MemoryClass::SmemLoad),
    )
}
fn vmem(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
    store: bool,
) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), uses).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> {
    b.control(Instruction::new(text, vec![], vec![]))
}

/// The wide exports, appended after every narrow export: single r16, single
/// r32, x4 r16, x4 r32 (`wide::SYMBOLS` order).
pub fn build_wide_gfx1151() -> Result<Vec<Emitted>, String> {
    let mut out = Vec::new();
    for x4 in [false, true] {
        for cap in wide::CAPS {
            out.push(wide::emit(wide::Spec { x4, cap })?);
        }
    }
    Ok(out)
}

/// Row-exact wide verify twins of the MQ6G256V2 lane-step GEMV (design
/// `WideTwinGemvSol.md` section 2.2). Symbols
/// `gemv_mq6g256v2_xbatch_wide_r{16,32}_pm` and
/// `gemv_mq6g256v2_x4_xbatch_wide_r{16,32}_pm`, exact gfx1151 only.
///
/// Geometry: grid `ceil(M/8)` (x4: `sum_i ceil(max(Mi,0)/8)`, workgroup `w`
/// routed to the first matrix `i` with `w < P0+..+Pi`, never spanning two),
/// block 256 (wave `v = tid >> 5`), static LDS 8192 B (8 rows x 256 K x 4 B),
/// `cu_mode = false`. Wave `v` owns matrix-LOCAL column `m = 8*(w - base_i) +
/// v` for every row `r < R` (`R` in `1..=cap`; anything else exits uniformly
/// before the first barrier): `4*cap` accumulators (4 streams per row) stay in
/// its VGPRs for the whole K sweep.
///
/// Schedule per K group `g` (ascending, unrolled by four so the stream
/// `g % 4` is a compile-time register choice): each valid wave loads and
/// dequantizes its 200-byte group ONCE (`v_fma_mix_f32` weights, q5 word
/// boundary as narrow); then for row tiles `rb = 0, 8, 16, 24` (`rb < R`):
/// producer wave `v` loads X row `rb + v` (`g*256 + 8*lane + j`, two b128
/// loads) and stores the unchanged f32 bits to LDS word `(v*8 + j)*32 + lane`;
/// uniform barrier; every valid wave consumes the tile's rows (LDS reads, the
/// 8-term chain x1 mul, x0, x2..x7 with the weights held in registers, then
/// `a_s = a_s + d`); uniform barrier before the tile is overwritten. There is
/// no weight load inside the row loop. Waves with `m >= M_i` produce X and
/// execute every barrier and skip the weight loads, arithmetic and stores;
/// they leave only after the last barrier. Every barrier decision depends on
/// `R`, `K` and the workgroup id alone. Rows `>= R` are never loaded,
/// consumed or stored (their accumulators stay +0 and are folded along).
///
/// Epilogue per row: `(a0 + a1) + (a3 + a2)`, `ds_swizzle` offset 16
/// (`"1pppp"`) then `ds_bpermute` +8/+4/+2/+1 (own lane when out of range),
/// `own + other` each level, lane 0 stores `y[r*M + m]`.
///
/// Register ceilings (rejected above the budget and above 256, checked in
/// `usize`): VGPR `ACC0 + 4*cap` = 104 of 128 (r16) / 168 of 192 (r32); SGPR 60
/// of 80. The LDS address VGPRs are derived in the entry block from the
/// work-item id and never redefined (`pm_check::lds_bounds` evaluates them).
pub mod wide {
    use super::{branch, valu, vmem, GROUP, TERMS};
    use crate::insn::{Instruction, MemoryClass};
    use crate::kernels::common::{lit, op, s, smem, sop, sr, v, SRD_WORD3};
    use crate::lds::Transition;
    use crate::ledger::Counter;
    use crate::plan::Access;
    use crate::reg::{Kind, Live};
    use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

    type R = Result<(), String>;

    /// Compiled row capacities.
    pub const CAPS: [u8; 2] = [16, 32];
    /// Output columns (waves) per workgroup.
    pub const M_BLOCK: u32 = 8;
    /// Threads per workgroup.
    pub const BLOCK: u16 = 256;
    /// Static LDS bytes: 8 rows x 256 K x 4 B.
    pub const LDS_BYTES: u32 = 8192;
    /// Kernarg bytes of the single and x4 exports (the narrow ABIs).
    pub const SINGLE_KERNARG_BYTES: u32 = 36;
    pub const X4_KERNARG_BYTES: u32 = 96;
    /// The four exports in `build_wide_gfx1151` order.
    pub const SYMBOLS: [&str; 4] = [
        "gemv_mq6g256v2_xbatch_wide_r16_pm",
        "gemv_mq6g256v2_xbatch_wide_r32_pm",
        "gemv_mq6g256v2_x4_xbatch_wide_r16_pm",
        "gemv_mq6g256v2_x4_xbatch_wide_r32_pm",
    ];

    /// One wide export: single or x4, row capacity `cap` (16 or 32).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Spec {
        pub x4: bool,
        pub cap: u8,
    }
    impl Spec {
        pub fn symbol(&self) -> String {
            format!(
                "gemv_mq6g256v2_{}xbatch_wide_r{}_pm",
                if self.x4 { "x4_" } else { "" },
                self.cap
            )
        }
        pub fn validate(&self) -> R {
            if !CAPS.contains(&self.cap) {
                return Err(format!("mq6 wide capacity {} is not 16 or 32", self.cap));
            }
            Ok(())
        }
        /// Declared VGPR budget (design ceilings).
        pub fn vgpr_budget(&self) -> u16 {
            if self.cap == 16 {
                128
            } else {
                192
            }
        }
        fn tiles(&self) -> u32 {
            u32::from(self.cap) / 8
        }
        fn tag(&self) -> String {
            format!("mq6{}w{}", if self.x4 { "x4" } else { "" }, self.cap)
        }
    }

    // SGPRs. s0:1 kernarg, s2 workgroup id x (`user_sgpr_count = 2`).
    const KARG: u8 = 0;
    const SW: u8 = 2;
    const SRD: u8 = 4;
    const PA: u8 = 8;
    const PXY: u8 = 16;
    const XP: u8 = 16;
    const Y0: u8 = 18;
    const Y1: u8 = 20;
    const Y2: u8 = 22;
    const Y3: u8 = 24;
    const M0: u8 = 26;
    const M1: u8 = 27;
    const M2: u8 = 28;
    const M3: u8 = 29;
    const SK: u8 = 30;
    const SR: u8 = 31;
    const SY: u8 = 32;
    const SM: u8 = 34;
    const SBASE: u8 = 35;
    const CP0: u8 = 36;
    const CP1: u8 = 37;
    const CP2: u8 = 38;
    const CP3: u8 = 39;
    const WAVE: u8 = 40;
    const ROW0: u8 = 41;
    const GROUPS: u8 = 42;
    const STRIDE: u8 = 43;
    const TMP: u8 = 44;
    const RB: u8 = 45;
    const GOFF: u8 = 46;
    const XG: u8 = 47;
    const GI: u8 = 48;
    const KB: u8 = 49;
    const MB: u8 = 50;
    const STMP: u8 = 51;
    const SVAL: u8 = 52;
    const R0B: u8 = 53;
    /// `XT[t] = (8t + wave) * K * 4`: byte offset of tile `t`'s producer row.
    const XT: u8 = 56;
    const SGPR_BUDGET: u16 = 80;

    // VGPRs.
    const TID: u8 = 0;
    const HOFF: u8 = 1;
    const POFF: u8 = 2;
    const XOFF: u8 = 3;
    const SHUF: u8 = 4;
    const LANE: u8 = 5;
    const HDR: u8 = 6;
    const PACK: u8 = 7;
    const UW: u8 = 8;
    const DOT: u8 = 9;
    /// LDS producer store address `wave*1024 + 4*lane` and consumer read base `4*lane`.
    const LDST: u8 = 10;
    const LDC: u8 = 11;
    const VT: u8 = 12;
    /// `(lane >> 4) * 4`: half-header byte offset; `lane * 6`: payload byte offset.
    const HALF: u8 = 13;
    const PKB: u8 = 14;
    /// Dequantized weight slots W0..W7.
    const WS: u8 = 16;
    /// X slots X0..X7 (producer staging and consumer row vector).
    const XS: u8 = 24;
    /// Reduction temporaries / store addresses (16 live) are `XS..XS+16`.
    const TS: u8 = 32;
    const ACC0: u16 = 40;

    /// High-water VGPR index + 1 for `cap` rows, in `usize` arithmetic.
    fn vgpr_high_water(cap: u8) -> usize {
        usize::from(ACC0) + 4 * usize::from(cap)
    }
    /// High-water SGPR index + 1.
    fn sgpr_high_water() -> usize {
        usize::from(XT) + 4
    }
    fn acc(row: u32, stream: u32) -> u8 {
        let n = usize::from(ACC0) + 4 * row as usize + stream as usize;
        u8::try_from(n).expect("accumulator index validated by layout()")
    }
    /// The high-water marks must fit the budgets and the architectural limits.
    fn layout(spec: Spec) -> R {
        let hw = vgpr_high_water(spec.cap);
        if hw > 256 {
            return Err(format!("mq6 wide r{} needs {hw} VGPRs (> 256)", spec.cap));
        }
        if hw > usize::from(spec.vgpr_budget()) {
            return Err(format!(
                "mq6 wide r{} needs {hw} VGPRs (> budget {})",
                spec.cap,
                spec.vgpr_budget()
            ));
        }
        let sw = sgpr_high_water();
        if sw > usize::from(SGPR_BUDGET) {
            return Err(format!(
                "mq6 wide needs {sw} SGPRs (> budget {SGPR_BUDGET})"
            ));
        }
        Ok(())
    }

    fn lbl(spec: Spec, name: &str) -> String {
        format!(".L{}_{name}", spec.tag())
    }
    fn glbl(spec: Spec, stream: u32, tile: u32, name: &str) -> String {
        format!(".L{}_s{stream}t{tile}_{name}", spec.tag())
    }
    fn off(imm: u32) -> String {
        if imm == 0 {
            String::new()
        } else {
            format!(" offset:{imm}")
        }
    }

    fn plan(spec: Spec) -> Result<RegPlan, String> {
        layout(spec)?;
        let mut regs = RegPlan::new(spec.vgpr_budget(), SGPR_BUDGET)?;
        let mut s_ranges: Vec<(&str, u8, u8)> = vec![
            ("kernarg", KARG, 2),
            ("workgroup_id", SW, 1),
            ("weight_resource", SRD, 4),
        ];
        if spec.x4 {
            s_ranges.extend([
                ("weight_pointers", PA, 8),
                ("x_y_pointers", PXY, 8),
                ("y3_m01", Y3, 4),
                ("m23_k_r", M2, 4),
                ("blocks_0", CP0, 1),
                ("blocks_01", CP1, 1),
                ("blocks_012", CP2, 1),
            ]);
        } else {
            s_ranges.extend([("x_pointer", XP, 2), ("k", SK, 1), ("r", SR, 1)]);
        }
        s_ranges.extend([
            ("y_selected", SY, 2),
            ("m_selected", SM, 1),
            ("matrix_base", SBASE, 1),
            ("blocks_total", CP3, 1),
            ("wave", WAVE, 1),
            ("column", ROW0, 1),
            ("groups", GROUPS, 1),
            ("stride", STRIDE, 1),
            ("temp", TMP, 1),
            ("weight_row_offset", RB, 1),
            ("group_offset", GOFF, 1),
            ("x_group_offset", XG, 1),
            ("group_index", GI, 1),
            ("k_bytes", KB, 1),
            ("m_bytes", MB, 1),
            ("store_temp", STMP, 1),
            ("column_valid", SVAL, 1),
            ("column_bytes", R0B, 1),
            ("x_tile_offsets", XT, 4),
        ]);
        for (name, base, len) in s_ranges {
            regs.add_range(name, Kind::S, base, len, Live::Whole)?;
        }
        for (name, base, len) in [
            ("tid", TID, 1),
            ("header_offset", HOFF, 1),
            ("packed_offset", POFF, 1),
            ("x_offset", XOFF, 1),
            ("shuffle_addr", SHUF, 1),
            ("lane", LANE, 1),
            ("header", HDR, 1),
            ("packed", PACK, 1),
            ("packed_hi", UW, 1),
            ("dot", DOT, 1),
            ("lds_store", LDST, 1),
            ("lds_read", LDC, 1),
            ("temp", VT, 1),
            ("half_offset", HALF, 1),
            ("payload_offset", PKB, 1),
            ("weights", WS, 8),
            ("x_slots", XS, 8),
            ("temps", TS, 8),
        ] {
            regs.add_range(name, Kind::V, base, len, Live::Whole)?;
        }
        for block in 0..(4 * usize::from(spec.cap) / 8) {
            let base = u8::try_from(usize::from(ACC0) + 8 * block)
                .map_err(|_| "accumulator block past v255".to_string())?;
            regs.add_range(&format!("acc_block{block}"), Kind::V, base, 8, Live::Whole)?;
        }
        Ok(regs)
    }

    pub fn emit(spec: Spec) -> Result<Emitted, String> {
        spec.validate()?;
        let (variant, kernargs) = if spec.x4 {
            let mut k = KernargLayout::new(X4_KERNARG_BYTES);
            for (i, name) in ["A0", "A1", "A2", "A3"].into_iter().enumerate() {
                k = k.pointer_access(name, 8 * i as u32, Access::ReadOnly);
            }
            k = k.pointer_access("x", 32, Access::ReadOnly);
            for (i, name) in ["y0", "y1", "y2", "y3"].into_iter().enumerate() {
                k = k.pointer_access(name, 40 + 8 * i as u32, Access::WriteOnly);
            }
            for (i, name) in ["M0", "M1", "M2", "M3"].into_iter().enumerate() {
                k = k.hidden(name, 72 + 4 * i as u32, 4, "by_value");
            }
            (
                format!("x4_xbatch_wide_r{}_m8w8", spec.cap),
                k.hidden("K", 88, 4, "by_value")
                    .hidden("R", 92, 4, "by_value"),
            )
        } else {
            (
                format!("xbatch_wide_r{}_m8w8", spec.cap),
                KernargLayout::new(SINGLE_KERNARG_BYTES)
                    .pointer_access("A", 0, Access::ReadOnly)
                    .pointer_access("x", 8, Access::ReadOnly)
                    .pointer_access("y", 16, Access::WriteOnly)
                    .hidden("M", 24, 4, "by_value")
                    .hidden("K", 28, 4, "by_value")
                    .hidden("R", 32, 4, "by_value"),
            )
        };
        let kspec = KernelSpec {
            kernel_id: "qwen4_lane_rows_mq6".into(),
            variant,
            arch: Arch::Gfx1151,
            symbol: spec.symbol(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: BLOCK,
            group_segment_fixed_size: LDS_BYTES,
            wave32: true,
            cu_mode: false,
        };
        let mut b = Builder::new(kspec, plan(spec)?);
        b.enable_delay_alu();
        let slot = b.lds_slot("x_tile", 0, LDS_BYTES)?;
        prologue(&mut b, spec)?;
        // Zero the 4*cap accumulators (+0, as the incumbent's streams).
        for row in 0..u32::from(spec.cap) {
            for st in 0..4 {
                valu(
                    &mut b,
                    &format!("v_mov_b32_e32 v{}, 0", acc(row, st)),
                    &[acc(row, st)],
                    &[],
                    &[],
                )?;
            }
        }
        sop(&mut b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
        sop(&mut b, format!("s_cmp_eq_u32 s{GROUPS}, 0"), &[], &[GROUPS])?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "fold")))?;
        b.loop_(&lbl(spec, "group"), |b| {
            for st in 0..4u32 {
                if st > 0 {
                    sop(
                        b,
                        format!("s_cmp_ge_u32 s{GI}, s{GROUPS}"),
                        &[],
                        &[GI, GROUPS],
                    )?;
                    branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "fold")))?;
                }
                group(b, spec, slot, st)?;
                sop(b, format!("s_add_i32 s{GI}, s{GI}, 1"), &[GI], &[GI])?;
            }
            sop(
                b,
                format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                &[],
                &[GI, GROUPS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "group")))
        })?;
        b.label(&lbl(spec, "fold"))?;
        // Waves past M never reach another barrier: they leave here.
        sop(&mut b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
        epilogue(&mut b, spec)?;
        b.label(&lbl(spec, "end"))?;
        branch(&mut b, "s_endpgm")?;
        b.finish()
    }

    /// `dst = prev + ceil(max(m, 0) / 8)` (`prev` absent: `dst = ceil(...)`).
    fn blocks(b: &mut Builder, dst: u8, prev: Option<u8>, m: u8) -> R {
        let t = if prev.is_some() { TMP } else { dst };
        sop(b, format!("s_max_i32 s{t}, s{m}, 0"), &[t], &[m])?;
        sop(b, format!("s_add_i32 s{t}, s{t}, 7"), &[t], &[t])?;
        sop(b, format!("s_lshr_b32 s{t}, s{t}, 3"), &[t], &[t])?;
        if let Some(p) = prev {
            sop(b, format!("s_add_i32 s{dst}, s{p}, s{t}"), &[dst], &[p, t])?;
        }
        Ok(())
    }

    fn prologue(b: &mut Builder, spec: Spec) -> R {
        if spec.x4 {
            smem(b, PA, 8, KARG, 0)?;
            smem(b, PXY, 8, KARG, 32)?;
            smem(b, Y3, 4, KARG, 64)?;
            smem(b, M2, 4, KARG, 80)?;
        } else {
            smem(b, SRD, 2, KARG, 0)?;
            smem(b, XP, 2, KARG, 8)?;
            smem(b, SY, 2, KARG, 16)?;
            smem(b, SM, 1, KARG, 24)?;
            smem(b, SK, 1, KARG, 28)?;
            smem(b, SR, 1, KARG, 32)?;
        }
        // Entry block (before the first branch): the LDS address vectors and
        // the wave id derive from the work-item id alone. LDS word of (row v,
        // element j, lane l) is (v*8 + j)*32 + l: the producer stores at
        // v*1024 + 4l (+128 j), the consumer reads rows at 4l + 1024 r + 128 j.
        valu(
            b,
            &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
            &[LANE],
            &[TID],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
            &[LDC],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
            &[VT],
            &[TID],
            &[],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{WAVE}, v{VT}"),
            &[s(WAVE)],
            &[v(VT)],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{VT}, 10, v{VT}"),
            &[VT],
            &[VT],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDST}, v{VT}, v{LDC}"),
            &[LDST],
            &[VT, LDC],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{HALF}, 4, v{LANE}"),
            &[HALF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{HALF}, 2, v{HALF}"),
            &[HALF],
            &[HALF],
            &[],
        )?;
        valu(
            b,
            &format!("v_mul_u32_u24_e32 v{PKB}, 6, v{LANE}"),
            &[PKB],
            &[LANE],
            &[],
        )?;
        b.wait_all()?;
        if spec.x4 {
            // Cumulative block counts Pi = ceil(max(Mi, 0) / 8).
            blocks(b, CP0, None, M0)?;
            blocks(b, CP1, Some(CP0), M1)?;
            blocks(b, CP2, Some(CP1), M2)?;
            blocks(b, CP3, Some(CP2), M3)?;
        } else {
            blocks(b, CP3, None, SM)?;
        }
        // Workgroups past the sum return; so does an R outside 1..=cap (all
        // uniform, before the first barrier).
        sop(b, format!("s_cmp_ge_u32 s{SW}, s{CP3}"), &[], &[SW, CP3])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
        sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
        sop(b, format!("s_cmp_gt_u32 s{SR}, {}", spec.cap), &[], &[SR])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
        if spec.x4 {
            route(b)?;
        } else {
            sop(b, format!("s_mov_b32 s{SBASE}, 0"), &[SBASE], &[])?;
        }
        // This wave's matrix-local column m = 8*(w - base) + wave, and whether it exists.
        sop(
            b,
            format!("s_sub_i32 s{ROW0}, s{SW}, s{SBASE}"),
            &[ROW0],
            &[SW, SBASE],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{ROW0}, s{ROW0}, 3"),
            &[ROW0],
            &[ROW0],
        )?;
        sop(
            b,
            format!("s_add_i32 s{ROW0}, s{ROW0}, s{WAVE}"),
            &[ROW0],
            &[ROW0, WAVE],
        )?;
        sop(b, format!("s_cmp_lt_i32 s{ROW0}, s{SM}"), &[], &[ROW0, SM])?;
        sop(b, format!("s_cselect_b32 s{SVAL}, 1, 0"), &[SVAL], &[])?;
        sop(
            b,
            format!("s_lshr_b32 s{GROUPS}, s{SK}, 8"),
            &[GROUPS],
            &[SK],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{STRIDE}, s{GROUPS}, {}", lit(GROUP)),
            &[STRIDE],
            &[GROUPS],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{RB}, s{ROW0}, s{STRIDE}"),
            &[RB],
            &[ROW0, STRIDE],
        )?;
        // Raw weight buffer resource (unbounded records) over the selected matrix.
        sop(
            b,
            format!("s_and_b32 s{}, s{}, 0xffff", SRD + 1, SRD + 1),
            &[SRD + 1],
            &[SRD + 1],
        )?;
        sop(b, format!("s_mov_b32 s{}, -1", SRD + 2), &[SRD + 2], &[])?;
        sop(
            b,
            format!("s_mov_b32 s{}, {}", SRD + 3, lit(SRD_WORD3)),
            &[SRD + 3],
            &[],
        )?;
        sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
        sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
        sop(b, format!("s_lshl_b32 s{R0B}, s{ROW0}, 2"), &[R0B], &[ROW0])?;
        // Producer X row offsets of every tile: (8t + wave) * K * 4.
        for t in 0..spec.tiles() {
            sop(
                b,
                format!("s_add_i32 s{TMP}, s{WAVE}, {}", 8 * t),
                &[TMP],
                &[WAVE],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{}, s{TMP}, s{KB}", XT + t as u8),
                &[XT + t as u8],
                &[TMP, KB],
            )?;
        }
        Ok(())
    }

    /// x4: workgroup `w` -> the first matrix `i` with `w < end_i` (prefix sums
    /// of `ceil(max(Mi,0)/8)`): its A/y/M and the base workgroup of the matrix.
    fn route(b: &mut Builder) -> R {
        let pair = |n: u8| sr(n, 2);
        // Matrix 3, then matrices 2, 1, 0 override it while `w < end_i`.
        sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP2}"), &[], &[SW, CP2])?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{}:{}]",
                SRD + 1,
                PA + 4,
                PA + 5,
                PA + 6,
                PA + 7
            ),
            &[pair(SRD)],
            &[pair(PA + 4), pair(PA + 6)],
        )?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SY}:{}], s[{Y2}:{}], s[{Y3}:{}]",
                SY + 1,
                Y2 + 1,
                Y3 + 1
            ),
            &[pair(SY)],
            &[pair(Y2), pair(Y3)],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SM}, s{M2}, s{M3}"),
            &[SM],
            &[M2, M3],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SBASE}, s{CP1}, s{CP2}"),
            &[SBASE],
            &[CP1, CP2],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP1}"), &[], &[SW, CP1])?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SRD}:{}], s[{}:{}], s[{SRD}:{}]",
                SRD + 1,
                PA + 2,
                PA + 3,
                SRD + 1
            ),
            &[pair(SRD)],
            &[pair(PA + 2), pair(SRD)],
        )?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SY}:{}], s[{Y1}:{}], s[{SY}:{}]",
                SY + 1,
                Y1 + 1,
                SY + 1
            ),
            &[pair(SY)],
            &[pair(Y1), pair(SY)],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SM}, s{M1}, s{SM}"),
            &[SM],
            &[M1, SM],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SBASE}, s{CP0}, s{SBASE}"),
            &[SBASE],
            &[CP0, SBASE],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{SW}, s{CP0}"), &[], &[SW, CP0])?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SRD}:{}], s[{PA}:{}], s[{SRD}:{}]",
                SRD + 1,
                PA + 1,
                SRD + 1
            ),
            &[pair(SRD)],
            &[pair(PA), pair(SRD)],
        )?;
        op(
            b,
            format!(
                "s_cselect_b64 s[{SY}:{}], s[{Y0}:{}], s[{SY}:{}]",
                SY + 1,
                Y0 + 1,
                SY + 1
            ),
            &[pair(SY)],
            &[pair(Y0), pair(SY)],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SM}, s{M0}, s{SM}"),
            &[SM],
            &[M0, SM],
        )?;
        sop(
            b,
            format!("s_cselect_b32 s{SBASE}, 0, s{SBASE}"),
            &[SBASE],
            &[SBASE],
        )
    }

    /// One K group with the compile-time stream `st` (`g % 4`).
    fn group(b: &mut Builder, spec: Spec, slot: usize, st: u32) -> R {
        sop(
            b,
            format!("s_mul_i32 s{GOFF}, s{GI}, {}", lit(GROUP)),
            &[GOFF],
            &[GI],
        )?;
        sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 10"), &[XG], &[GI])?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{XOFF}, 5, v{LANE}"),
            &[XOFF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{XOFF}, s{XG}, v{XOFF}"),
            &[XOFF],
            &[XOFF],
            &[XG],
        )?;
        // Valid waves: the weight group, loaded and dequantized ONCE.
        let wdone = glbl(spec, st, 9, "wdone");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
        weight_loads(b)?;
        b.wait(Counter::Vm, 0)?;
        dequant(b)?;
        b.label(&wdone)?;
        for t in 0..spec.tiles() {
            tile(b, spec, slot, st, t)?;
        }
        Ok(())
    }

    /// Header dword (lane-selected half), payload dword at `8 + 6*lane` and
    /// the zero-extended u16 at `12 + 6*lane`, exactly as the narrow kernel.
    fn weight_loads(b: &mut Builder) -> R {
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HALF}"),
            &[HOFF],
            &[HALF],
            &[GOFF],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{POFF}, s{GOFF}, v{PKB}"),
            &[POFF],
            &[PKB],
            &[GOFF],
        )?;
        let mut scalar: Vec<u8> = (SRD..SRD + 4).collect();
        scalar.push(RB);
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{PACK}, v{POFF}, s[{SRD}:{}], s{RB} offen offset:8",
                SRD + 3
            ),
            &[PACK],
            &[POFF],
            &scalar,
            false,
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_u16 v{UW}, v{POFF}, s[{SRD}:{}], s{RB} offen offset:12",
                SRD + 3
            ),
            &[UW],
            &[POFF],
            &scalar,
            false,
        )?;
        vmem(
            b,
            &format!(
                "buffer_load_b32 v{HDR}, v{HOFF}, s[{SRD}:{}], s{RB} offen",
                SRD + 3
            ),
            &[HDR],
            &[HOFF],
            &scalar,
            false,
        )
    }

    /// `w[n] = fma_mix(sc, f32(q_n), zp)`, extracting `q0..q7` from the payload
    /// dword `p` and word `u` exactly as the narrow `dequant` (q5 straddles).
    fn dequant(b: &mut Builder) -> R {
        let (p, u, h) = (PACK, UW, HDR);
        valu(
            b,
            &format!("v_and_b32_e32 v{WS}, 63, v{p}"),
            &[WS],
            &[p],
            &[],
        )?;
        for (n, lsb) in [(1u8, 6u32), (2, 12), (3, 18), (4, 24)] {
            valu(
                b,
                &format!("v_bfe_u32 v{}, v{p}, {lsb}, 6", WS + n),
                &[WS + n],
                &[p],
                &[],
            )?;
        }
        valu(
            b,
            &format!("v_alignbit_b32 v{}, v{u}, v{p}, 30", WS + 5),
            &[WS + 5],
            &[u, p],
            &[],
        )?;
        valu(
            b,
            &format!("v_and_b32_e32 v{0}, 63, v{0}", WS + 5),
            &[WS + 5],
            &[WS + 5],
            &[],
        )?;
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{u}, 4, 6", WS + 6),
            &[WS + 6],
            &[u],
            &[],
        )?;
        valu(
            b,
            &format!("v_bfe_u32 v{}, v{u}, 10, 6", WS + 7),
            &[WS + 7],
            &[u],
            &[],
        )?;
        for n in 0..8u8 {
            valu(
                b,
                &format!("v_cvt_f32_ubyte0_e32 v{0}, v{0}", WS + n),
                &[WS + n],
                &[WS + n],
                &[],
            )?;
        }
        for n in 0..8u8 {
            let d = WS + n;
            valu(
                b,
                &format!("v_fma_mix_f32 v{d}, v{h}, v{d}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
                &[d],
                &[h, d],
                &[],
            )?;
        }
        Ok(())
    }

    /// Row tile `t` (rows `8t..8t+7`) of one group: fill, barrier, consume, barrier.
    fn tile(b: &mut Builder, spec: Spec, slot: usize, st: u32, t: u32) -> R {
        let rb = 8 * t;
        let done = glbl(spec, st, t, "tile_done");
        // The whole tile is past R: skipped by every wave of the workgroup.
        if t > 0 {
            sop(b, format!("s_cmp_le_u32 s{SR}, {rb}"), &[], &[SR])?;
            branch(b, &format!("s_cbranch_scc1 {done}"))?;
        }
        // Producer wave `v` stages X row rb+v when it exists.
        let nofill = glbl(spec, st, t, "nofill");
        sop(
            b,
            format!("s_add_i32 s{TMP}, s{WAVE}, {rb}"),
            &[TMP],
            &[WAVE],
        )?;
        sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SR}"), &[], &[TMP, SR])?;
        branch(b, &format!("s_cbranch_scc1 {nofill}"))?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{VT}, s{}, v{XOFF}", XT + t as u8),
            &[VT],
            &[XOFF],
            &[XT + t as u8],
        )?;
        for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
            vmem(
                b,
                &format!(
                    "global_load_b128 v[{}:{}], v{VT}, s[{XP}:{}]{suffix}",
                    XS + j,
                    XS + j + 3,
                    XP + 1
                ),
                &[XS + j, XS + j + 1, XS + j + 2, XS + j + 3],
                &[VT],
                &[XP, XP + 1],
                false,
            )?;
        }
        for j in 0..8u32 {
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{}{}", XS + j as u8, off(128 * j)),
                    vec![],
                    vec![v(LDST), v(XS + j as u8)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.wait_all()?;
        b.label(&nofill)?;
        b.barrier(&[Transition::Ready(slot)])?;
        // Valid waves consume rows rb.. in ascending order while row < R.
        let cdone = glbl(spec, st, t, "cdone");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
        for r in 0..8u32 {
            if r > 0 {
                sop(b, format!("s_cmp_le_u32 s{SR}, {}", rb + r), &[], &[SR])?;
                branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
            }
            consume_row(b, slot, st, rb + r, 1024 * r)?;
        }
        b.label(&cdone)?;
        b.barrier(&[Transition::Retire(slot)])?;
        b.label(&done)
    }

    /// One row's group dot (x1 mul, then x0, x2..x7 `v_fmac`, multiplicand order
    /// `w, x`) and its separate stream add.
    fn consume_row(b: &mut Builder, slot: usize, st: u32, row: u32, row_off: u32) -> R {
        for n in TERMS {
            let x = XS + n;
            b.ds_load(
                slot,
                Instruction::new(
                    format!(
                        "ds_load_b32 v{x}, v{LDC}{}",
                        off(row_off + 128 * u32::from(n))
                    ),
                    vec![v(x)],
                    vec![v(LDC)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
        }
        for (term, n) in TERMS.into_iter().enumerate() {
            let (w, x) = (WS + n, XS + n);
            if term == 0 {
                valu(
                    b,
                    &format!("v_mul_f32_e32 v{DOT}, v{w}, v{x}"),
                    &[DOT],
                    &[w, x],
                    &[],
                )?;
            } else {
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{DOT}, v{w}, v{x}"),
                    &[DOT],
                    &[DOT, w, x],
                    &[],
                )?;
            }
        }
        let a = acc(row, st);
        valu(
            b,
            &format!("v_add_f32_e32 v{a}, v{a}, v{DOT}"),
            &[a],
            &[a, DOT],
            &[],
        )
    }

    fn epilogue(b: &mut Builder, spec: Spec) -> R {
        let cap = u32::from(spec.cap);
        // Fold (a0 + a1) + (a3 + a2) for every row (rows >= R fold zeros).
        for r in 0..cap {
            for (l, h) in [(0, 1), (3, 2), (0, 3)] {
                let (d, x) = (acc(r, l), acc(r, h));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
                    &[d],
                    &[d, x],
                    &[],
                )?;
            }
        }
        // 16/8/4/2/1 own+other tree, 16 rows at a time on the dead x/temp VGPRs.
        for batch in 0..cap.div_ceil(16) {
            let rows: Vec<u32> = (16 * batch..cap.min(16 * batch + 16)).collect();
            let tmp = |i: usize| XS + i as u8;
            for (i, &r) in rows.iter().enumerate() {
                let (val, t) = (acc(r, 0), tmp(i));
                b.ds_crosslane(
                    Instruction::new(
                        format!(
                            "ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"
                        ),
                        vec![v(t)],
                        vec![v(val)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            for (i, &r) in rows.iter().enumerate() {
                let (val, t) = (acc(r, 0), tmp(i));
                valu(
                    b,
                    &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
                    &[val],
                    &[val, t],
                    &[],
                )?;
            }
            for offs in [8u32, 4, 2, 1] {
                // Lanes whose partner is out of range read themselves.
                valu(
                    b,
                    &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - offs),
                    &[],
                    &[LANE],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_cndmask_b32_e64 v{SHUF}, 0, {offs}, vcc_lo"),
                    &[SHUF],
                    &[],
                    &[],
                )?;
                valu(
                    b,
                    &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"),
                    &[SHUF],
                    &[SHUF, LANE],
                    &[],
                )?;
                for (i, &r) in rows.iter().enumerate() {
                    let (val, t) = (acc(r, 0), tmp(i));
                    b.ds_crosslane(
                        Instruction::new(
                            format!("ds_bpermute_b32 v{t}, v{SHUF}, v{val}"),
                            vec![v(t)],
                            vec![v(SHUF), v(val)],
                        )
                        .memory(MemoryClass::DsLoad),
                    )?;
                }
                for (i, &r) in rows.iter().enumerate() {
                    let (val, t) = (acc(r, 0), tmp(i));
                    valu(
                        b,
                        &format!("v_add_f32_e32 v{val}, v{val}, v{t}"),
                        &[val],
                        &[val, t],
                        &[],
                    )?;
                }
            }
        }
        // Lane 0 stores y[r*M + m] for every r < R (16 address VGPRs per batch).
        valu(
            b,
            &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
            &[],
            &[LANE],
            &[],
        )?;
        let ys = [SY, SY + 1];
        for batch in 0..cap.div_ceil(16) {
            let rows: Vec<u32> = (16 * batch..cap.min(16 * batch + 16)).collect();
            for (i, &r) in rows.iter().enumerate() {
                sop(b, format!("s_mul_i32 s{STMP}, s{MB}, {r}"), &[STMP], &[MB])?;
                sop(
                    b,
                    format!("s_add_i32 s{STMP}, s{STMP}, s{R0B}"),
                    &[STMP],
                    &[STMP, R0B],
                )?;
                valu(
                    b,
                    &format!("v_mov_b32_e32 v{}, s{STMP}", XS + i as u8),
                    &[XS + i as u8],
                    &[],
                    &[STMP],
                )?;
            }
            let bdone = lbl(spec, &format!("stored{batch}"));
            for (i, &r) in rows.iter().enumerate() {
                sop(b, format!("s_cmp_le_u32 s{SR}, {r}"), &[], &[SR])?;
                branch(b, &format!("s_cbranch_scc1 {bdone}"))?;
                vmem(
                    b,
                    &format!(
                        "global_store_b32 v{}, v{}, s[{SY}:{}]",
                        XS + i as u8,
                        acc(r, 0),
                        SY + 1
                    ),
                    &[],
                    &[XS + i as u8, acc(r, 0)],
                    &ys,
                    true,
                )?;
            }
            b.label(&bdone)?;
            b.wait_all()?;
            if 16 * (batch + 1) < cap {
                sop(
                    b,
                    format!("s_cmp_le_u32 s{SR}, {}", 16 * (batch + 1)),
                    &[],
                    &[SR],
                )?;
                branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn register_ceilings_hold_in_wide_arithmetic() {
            assert_eq!(vgpr_high_water(16), 104);
            assert_eq!(vgpr_high_water(32), 168);
            for cap in CAPS {
                let spec = Spec { x4: false, cap };
                layout(spec).expect("fits");
                assert!(vgpr_high_water(cap) <= usize::from(spec.vgpr_budget()));
                // The last accumulator is addressable as u8 and the plan accepts the layout.
                assert_eq!(
                    usize::from(acc(u32::from(cap) - 1, 3)) + 1,
                    vgpr_high_water(cap)
                );
                plan(spec).expect("plan");
            }
            assert!(sgpr_high_water() <= usize::from(SGPR_BUDGET));
            // A capacity past the ceilings is refused, never wrapped.
            assert!(Spec { x4: false, cap: 64 }.validate().is_err());
            assert!(vgpr_high_water(64) > usize::from(u8::MAX));
            // LDS address registers, weights, X slots, temps and accumulators are disjoint.
            assert_eq!(usize::from(TS) + 8, usize::from(ACC0));
            assert_eq!(usize::from(XS) + 16, usize::from(ACC0));
            assert_eq!(WS + 8, XS);
            assert!(PKB < WS);
        }

        #[test]
        fn builds_four_wide_exports_in_contract_order() {
            let emitted = super::super::build_wide_gfx1151().expect("mq6 wide");
            assert_eq!(emitted.len(), 4);
            for (e, symbol) in emitted.iter().zip(SYMBOLS) {
                assert!(e.s_text.contains(&format!("{symbol}:")), "{symbol}");
                assert!(
                    e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"),
                    "{symbol}"
                );
                assert!(
                    e.s_text.contains("v_fma_mix_f32")
                        && e.s_text.contains("ds_store_b32")
                        && e.s_text.contains("ds_load_b32"),
                    "{symbol}"
                );
                assert!(e.s_text.contains("s_barrier"), "{symbol}");
                assert!(
                    !e.s_text.contains("v_dual_")
                        && !e.s_text.contains("ttmp")
                        && !e.s_text.contains("s_wait_alu"),
                    "{symbol}"
                );
            }
            assert_eq!(
                [(false, 16), (false, 32), (true, 16), (true, 32)].map(|(x4, cap)| Spec {
                    x4,
                    cap
                }
                .symbol()),
                SYMBOLS.map(String::from)
            );
            // The narrow exports are a separate builder, untouched by the wide one.
            assert_eq!(super::super::build_gfx1151().expect("narrow").len(), 2);
        }

        #[cfg(feature = "toolchain")]
        #[test]
        fn wide_lds_bounds_and_m7() {
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-lane-rows-mq6-wide-{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            for (emitted, symbol) in super::super::build_wide_gfx1151()
                .expect("mq6 wide")
                .into_iter()
                .zip(SYMBOLS)
            {
                let end = crate::pm_check::lds_bounds(
                    &emitted.s_text,
                    symbol,
                    u32::from(BLOCK) / 32,
                    LDS_BYTES,
                )
                .expect("LDS bounds");
                assert_eq!(end, LDS_BYTES, "{symbol}");
                let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1151)
                    .expect("native assemble");
                std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
                let co = dir.join(format!("{symbol}.test.co"));
                std::fs::write(&co, &elf).unwrap();
                let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
                println!("{symbol}: {report}");
                assert_eq!(
                    report["obligations"],
                    serde_json::json!({}),
                    "{symbol}: {report}"
                );
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plan_fits() {
        for nb in 1..=MAX_COLUMNS {
            let k = slots(nb);
            assert!(k >= 2, "nb={nb}");
            // Consecutive items (including the wrap) sit in different banks.
            for item in 0..2 * k {
                assert_ne!(
                    ring(nb, item) % 4,
                    ring(nb, item + 1) % 4,
                    "nb={nb} item={item}"
                );
            }
            // Slots stay disjoint, above the accumulators and inside the file.
            assert_eq!(ring_base(nb), u16::from(acc(nb - 1, 3, 3)) + 1, "nb={nb}");
            assert!(ring_base(nb) + 9 * (k - 1) + 8 <= VGPRS, "nb={nb}");
            // Outstanding VMEM loads (two weight groups + the full ring) fit the 6-bit counter.
            assert!(2 * 12 + 2 * k <= 63, "nb={nb}");
            // Epilogue scratch lives in dead raw/weight registers below the accumulators.
            assert!(
                u16::from(red_tmp(nb - 1, 3)) < u16::from(y_addr(0)),
                "nb={nb}"
            );
            assert!(u16::from(y_addr(nb - 1)) < ACC0, "nb={nb}");
        }
        assert!(vgpr_high() <= VGPRS);
        // Raw buffers, weights and dot registers tile v8..v71 without overlap.
        assert_eq!(raw(1, 3, 2), 31);
        assert_eq!(W[0] + 8, DOT[0][0]);
        assert_eq!(u16::from(DOT[1][3]) + 1, ACC0);
    }

    #[test]
    fn dot_registers_follow_the_vopd_rules() {
        for s in 0..2 {
            for r in 0..4usize {
                assert_eq!(usize::from(DOT[s][r]) % 4, r);
                assert_eq!(usize::from(DOT[s][r]) % 2, r % 2);
            }
        }
        for (a, b) in [(0usize, 1usize), (2, 3)] {
            assert_ne!(W[a] % 4, W[b] % 4);
        }
    }

    #[test]
    fn builds_both_exports_for_every_column_count() {
        let emitted = build_gfx1151().expect("mq6 lane rows");
        assert_eq!(emitted.len(), 2);
        for (e, symbol) in emitted.iter().zip([SYMBOL, SYMBOL_X4]) {
            assert!(e.s_text.contains(&format!("{symbol}:")), "{symbol}");
            assert!(e.s_text.contains("v_dual_fmac_f32"), "{symbol}");
            assert!(
                e.s_text.contains("ds_swizzle_b32") && e.s_text.contains("ds_bpermute_b32"),
                "{symbol}"
            );
            assert!(
                !e.s_text.contains("ttmp") && !e.s_text.contains("s_wait_alu"),
                "{symbol}"
            );
        }
    }

    #[cfg(feature = "toolchain")]
    #[test]
    fn lane_rows_mq6_m7() {
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-lane-rows-mq6-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (emitted, symbol) in build_gfx1151()
            .expect("mq6 lane rows")
            .into_iter()
            .zip([SYMBOL, SYMBOL_X4])
        {
            // The two columns of the first packet: x1 mul against rows (0, 1) of slots 0 and 1.
            let packet = format!(
                "v_dual_mul_f32 v{}, v{}, v{} :: v_dual_mul_f32 v{}, v{}, v{}",
                DOT[0][0],
                W[0] + 1,
                ring(2, 0) + 1,
                DOT[1][1],
                W[1] + 1,
                ring(2, 1) + 1
            );
            assert!(emitted.s_text.contains(&packet), "{symbol}: {packet}");
            let elf =
                crate::native::assemble(&emitted.s_text, Arch::Gfx1151).expect("native assemble");
            std::fs::write(dir.join(format!("{symbol}.test.s")), &emitted.s_text).unwrap();
            let co = dir.join(format!("{symbol}.test.co"));
            std::fs::write(&co, &elf).unwrap();
            let report = crate::pm_check::m7(&co, "gfx1151", symbol).expect("M7");
            println!("{symbol}: {report}");
            assert_eq!(
                report["obligations"],
                serde_json::json!({}),
                "{symbol}: {report}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
