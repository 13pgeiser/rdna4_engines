// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exact WIDE verify twins of the few-row MQ4G256V2 GEMV
//! (`mq4_rows_x4`): `gemv_mq4g256v2_rows_x4_wide_r{16,32}_pm_<arch>` for
//! gfx1151 and gfx1201, linked into the same module as the narrow entry
//! (`mq4_rows_x4_pm_<arch>`). The narrow `gemv_mq4g256v2_rows_x4_pm_<arch>`
//! (B = 1..8, wave per row pair) is untouched.
//!
//! ABI: the narrow entry's 96 bytes, unchanged: `A0@0 A1@8 A2@16 A3@24`,
//! `x@32`, `y0@40 y1@48 y2@56 y3@64`, `M0@72 M1@76 M2@80 M3@84 K@88`, and
//! `R@92` (the actual row count, `1..=cap`; any other `R` returns before any
//! store). Unused matrices have `M = 0`.
//!
//! Geometry: grid `[sum_i ceil(M_i/8), 1, 1]`, block `[256,1,1]` (eight
//! wave32), static LDS 8192 B (one 8-row x K256 X tile), `cu_mode = false`.
//! Workgroup `w` routes to the first matrix `i` with `w < P0+..+Pi`
//! (`Pi = ceil(Mi/8)`); no workgroup spans two matrices. Wave `v`
//! (`threadIdx.x >> 5`) owns ONE output column `m = 8*(w - base_i) + v`
//! (matrix-local) for every row `r < R`: `cap` rows' independent accumulators
//! (4 streams each, `4*cap` VGPRs) live in that wave for the whole K sweep.
//!
//! Exactness: each `(row r, column m)` is bitwise the 1-row decode GEMV's
//! (`gemv_mq4g256v2` = `gemv_mq4g256v2_multirow_r2`, the narrow entry's
//! reference) for activation row `r`:
//!  * lane `l` covers elements `g*256 + 8l .. 8l+7` of group `g`; the header
//!    dword is hA for `l < 16` else hB; the packed dword is group byte `8+4l`;
//!  * `w_j = v_fma_mix_f32(sc_f16, f32(nibble_j), zp_f16)`;
//!  * MATRIX-LOCAL parity (`m & 1` = `v & 1`, the block bases being multiples
//!    of 8): even `m` starts `d = x0*w0` (`v_mul_f32`), `d = fma(x1,w1,d)`;
//!    odd `m` starts `d = x1*w1`, `d = fma(x0,w0,d)`; then fma terms 2..7;
//!  * stream `s = g % 4`: `acc[s] = acc[s] + d` (a separate FP32 add, never a
//!    seeded dot), groups strictly ascending; fold `(a0+a1)+(a2+a3)`; the
//!    16/8/4/2/1 `ds_swizzle`/`ds_bpermute` own+other tree; lane 0 stores.
//!
//! Parity without branches: the instruction stream is the even chain
//! `mul(S0,X0); fma(S1,X1); fma(S2..S7,X2..X7)` over weight slots `S` and X
//! slots `X`. Even waves fill slot 0/1 with nibble 0/1 and read X element
//! 0/1; odd waves fill slot 0/1 with nibble 1/0 (`v_bfe_u32` offset taken
//! from SGPRs) and read X element 1/0 (LDS read bases `va`/`vb` differ by
//! parity). Slot `S0 = w_start`, `X0 = x_start` hold exactly the odd start
//! pair `(w1, x1)` then `(w0, x0)`, so the chain is the incumbent's, term for
//! term. Slots 2..7 are parity independent.
//!
//! Schedule (design `WideTwinGemvSol.md` §2.2): per K group `g` (ascending,
//! unrolled by four so the stream is a compile-time register choice) each
//! valid wave loads and dequantizes its weight group ONCE (buffer loads,
//! retired before use); then for row tiles `rb = 0,8,16,24` (`rb < R`):
//! producer wave `v` loads X row `rb+v` (`g*256 + 8*lane + j`, two b128 loads)
//! and stores the unchanged f32 bits to LDS word `(v*8 + j)*32 + lane`;
//! workgroup barrier; every valid wave consumes the tile's rows (LDS reads of
//! `x[row][8*lane+j]`, the 8-term chain, the group add) with the weights held
//! in registers; workgroup barrier before the tile is overwritten. A wave with
//! `m >= M_i` still produces X and executes every barrier; it skips the weight
//! loads, the arithmetic and the stores. All barrier decisions depend only on
//! `R`, `K` and the workgroup id, so they are uniform. Rows `>= R` are neither
//! loaded, consumed nor stored (accumulators of the unused rows stay +0 and
//! are folded and reduced along, never stored).
//!
//! Branch discipline of the checked builder: its wait ledger is linear, so no
//! conditional region leaves a memory operation pending and none is entered
//! with one pending (every region drains with explicit zero-count waits).
//!
//! Register ceilings (checked in u16/usize arithmetic, rejected above the
//! budget and above 256): VGPR `ACC0 + 4*cap` (R16: 104 of 128, R32: 168 of
//! 192); SGPR 64 of 80.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{lit, op, s, s_add_i32, smem, sop, sr, v, SRD_WORD3};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size (the narrow entry's).
pub const KERNARG_BYTES: u32 = 96;
/// Output columns (waves) per workgroup.
pub const M_BLOCK: u32 = 8;
/// Threads per workgroup.
pub const BLOCK: u16 = 256;
/// Static LDS bytes: 8 rows x 256 K x 4 B.
pub const LDS_BYTES: u32 = 8192;
/// Compiled row capacities.
pub const CAPS: [u8; 2] = [16, 32];

/// One wide entry: row capacity `cap` (16 or 32), per arch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub cap: u8,
}
impl Spec {
    pub fn symbol(&self) -> String {
        format!(
            "gemv_mq4g256v2_rows_x4_wide_r{}_pm_{}",
            self.cap,
            self.arch.name()
        )
    }
    pub fn validate(&self) -> R {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) {
            return Err("mq4_rows_x4 wide builder is exact-gfx1151 / gfx1201 only".into());
        }
        if !CAPS.contains(&self.cap) {
            return Err(format!(
                "mq4_rows_x4 wide capacity {} is not 16 or 32",
                self.cap
            ));
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
        format!("r{}", self.cap)
    }
}

// ----------------------------------------------------------------------- SGPRs
const KARG: u8 = 0;
const WGIN: u8 = 2;
const SRD: u8 = 4;
const PA: u8 = 8;
const PXY: u8 = 16;
const Y3M: u8 = 24;
const M2KB: u8 = 28;
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
const SW: u8 = 40;
const WAVE: u8 = 41;
const ROW0: u8 = 42;
const GROUPS: u8 = 43;
const STRIDE: u8 = 44;
const TMP: u8 = 45;
const RB: u8 = 46;
const SPAR: u8 = 47;
const SBO0: u8 = 48;
const SBO1: u8 = 49;
const GOFF: u8 = 50;
const XG: u8 = 51;
const GI: u8 = 52;
const KB: u8 = 53;
const MB: u8 = 54;
const STMP: u8 = 55;
const SVAL: u8 = 56;
const R0B: u8 = 57;
const SXOR: u8 = 58;
/// `XT[t] = (8t + wave) * K * 4`: the byte offset of tile `t`'s producer row.
const XT: u8 = 60;
const SGPR_BUDGET: u16 = 80;

// ----------------------------------------------------------------------- VGPRs
const TID: u8 = 0;
const HOFF: u8 = 1;
const POFF: u8 = 2;
const XOFF: u8 = 3;
const SHUF: u8 = 4;
const LANE: u8 = 5;
const HDR: u8 = 6;
const PACK: u8 = 7;
const DOT: u8 = 8;
/// LDS producer store address, and the three consumer read bases (element 0
/// slot, element 1 slot, elements 2..7).
const LDST: u8 = 9;
const LDA: u8 = 10;
const LDB: u8 = 11;
const LDC: u8 = 12;
const VT: u8 = 13;
const HALF: u8 = 14;
/// Dequantized weight slots S0..S7.
const WS: u8 = 16;
/// X slots X0..X7 (the producer's staging and the consumer's row vector).
const XS: u8 = 24;
/// Reduction temporaries / store addresses (16 live), with `XS`.
const TS: u8 = 32;
const ACC0: u16 = 40;

/// High-water VGPR index + 1 for `cap` rows, in u16 arithmetic.
pub fn vgpr_high_water(cap: u8) -> usize {
    usize::from(ACC0) + 4 * usize::from(cap)
}
fn acc(row: u32, stream: u32) -> u8 {
    let n = usize::from(ACC0) + 4 * row as usize + stream as usize;
    u8::try_from(n).expect("accumulator index validated by layout()")
}
/// Validate the register layout of `spec`: the high-water mark must fit the
/// budget and the architectural 256.
fn layout(spec: Spec) -> Result<(), String> {
    let hw = vgpr_high_water(spec.cap);
    if hw > 256 {
        return Err(format!("wide r{} needs {hw} VGPRs (> 256)", spec.cap));
    }
    if hw > usize::from(spec.vgpr_budget()) {
        return Err(format!(
            "wide r{} needs {hw} VGPRs (> budget {})",
            spec.cap,
            spec.vgpr_budget()
        ));
    }
    Ok(())
}

fn plan(spec: Spec) -> Result<RegPlan, String> {
    layout(spec)?;
    let mut regs = RegPlan::new(spec.vgpr_budget(), SGPR_BUDGET)?;
    let mut s_ranges: Vec<(&str, u8, u8)> = vec![
        ("kernarg", KARG, 2),
        ("weight_resource", SRD, 4),
        ("weight_pointers", PA, 8),
        ("x_y_pointers", PXY, 8),
        ("y3_m01", Y3M, 4),
        ("m23_k_r", M2KB, 4),
        ("y_selected", SY, 2),
        ("m_selected", SM, 1),
        ("pair_base", SBASE, 1),
        ("blocks_0", CP0, 1),
        ("blocks_01", CP1, 1),
        ("blocks_012", CP2, 1),
        ("blocks_0123", CP3, 1),
        ("workgroup", SW, 1),
        ("wave", WAVE, 1),
        ("column", ROW0, 1),
        ("groups", GROUPS, 1),
        ("stride", STRIDE, 1),
        ("temp", TMP, 1),
        ("weight_row_offset", RB, 1),
        ("parity", SPAR, 1),
        ("bfe_offset0", SBO0, 1),
        ("bfe_offset1", SBO1, 1),
        ("group_offset", GOFF, 1),
        ("x_group_offset", XG, 1),
        ("group_index", GI, 1),
        ("k_bytes", KB, 1),
        ("m_bytes", MB, 1),
        ("store_temp", STMP, 1),
        ("column_valid", SVAL, 1),
        ("column_bytes", R0B, 1),
        ("parity_xor", SXOR, 1),
        ("x_tile_offsets", XT, 4),
    ];
    if !spec.arch.gfx12() {
        s_ranges.push(("workgroup_id", WGIN, 1));
    }
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
        ("dot", DOT, 1),
        ("lds_store", LDST, 1),
        ("lds_read_a", LDA, 1),
        ("lds_read_b", LDB, 1),
        ("lds_read_c", LDC, 1),
        ("temp", VT, 1),
        ("half_offset", HALF, 1),
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

fn load_counter(arch: Arch) -> Counter {
    if arch.gfx12() {
        Counter::Load
    } else {
        Counter::Vm
    }
}
fn s_sub_i32(arch: Arch) -> &'static str {
    if arch.gfx12() {
        "s_sub_co_i32"
    } else {
        "s_sub_i32"
    }
}
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn lbl(spec: Spec, name: &str) -> String {
    format!(".Lmq4w{}_{name}", spec.cap)
}
fn glbl(spec: Spec, stream: u32, tile: u32, name: &str) -> String {
    format!(".Lmq4w{}_s{stream}t{tile}_{name}", spec.cap)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let arch = spec.arch;
    let kernargs = KernargLayout::new(KERNARG_BYTES)
        .pointer_access("A0", 0, Access::ReadOnly)
        .pointer_access("A1", 8, Access::ReadOnly)
        .pointer_access("A2", 16, Access::ReadOnly)
        .pointer_access("A3", 24, Access::ReadOnly)
        .pointer_access("x", 32, Access::ReadOnly)
        .pointer_access("y0", 40, Access::WriteOnly)
        .pointer_access("y1", 48, Access::WriteOnly)
        .pointer_access("y2", 56, Access::WriteOnly)
        .pointer_access("y3", 64, Access::WriteOnly)
        .hidden("M0", 72, 4, "by_value")
        .hidden("M1", 76, 4, "by_value")
        .hidden("M2", 80, 4, "by_value")
        .hidden("M3", 84, 4, "by_value")
        .hidden("K", 88, 4, "by_value")
        .hidden("R", 92, 4, "by_value");
    let kspec = KernelSpec {
        kernel_id: "mq4_rows_x4".into(),
        variant: format!("wide_{}_m8w8_x4", spec.tag()),
        arch,
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
    // Zero the 4*cap accumulators (+0, as the incumbent's `0.f`).
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
            sop(
                b,
                format!("{} s{GI}, s{GI}, 1", s_add_i32(spec.arch)),
                &[GI],
                &[GI],
            )?;
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

fn prologue(b: &mut Builder, spec: Spec) -> R {
    let a = b.spec.arch;
    smem(b, PA, 8, KARG, 0)?;
    smem(b, PXY, 8, KARG, 32)?;
    smem(b, Y3M, 4, KARG, 64)?;
    smem(b, M2KB, 4, KARG, 80)?;
    // Wave id, lane; the workgroup id.
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
        &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[LANE],
        &[TID],
        &[],
    )?;
    if a.gfx12() {
        sop(b, format!("s_mov_b32 s{SW}, ttmp9"), &[SW], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{SW}, s{WGIN}"), &[SW], &[WGIN])?;
    }
    b.wait_all()?;
    // Cumulative block counts: Pi = ceil(Mi / 8).
    sop(
        b,
        format!("{} s{CP0}, s{M0}, 7", s_add_i32(a)),
        &[CP0],
        &[M0],
    )?;
    sop(b, format!("s_lshr_b32 s{CP0}, s{CP0}, 3"), &[CP0], &[CP0])?;
    for (dst, prev, m) in [(CP1, CP0, M1), (CP2, CP1, M2), (CP3, CP2, M3)] {
        sop(b, format!("{} s{TMP}, s{m}, 7", s_add_i32(a)), &[TMP], &[m])?;
        sop(b, format!("s_lshr_b32 s{TMP}, s{TMP}, 3"), &[TMP], &[TMP])?;
        sop(
            b,
            format!("{} s{dst}, s{prev}, s{TMP}", s_add_i32(a)),
            &[dst],
            &[prev, TMP],
        )?;
    }
    // Workgroups past the sum return; so does an R outside 1..=cap (all
    // uniform, before the first barrier).
    sop(b, format!("s_cmp_ge_u32 s{SW}, s{CP3}"), &[], &[SW, CP3])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
    sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
    sop(b, format!("s_cmp_gt_u32 s{SR}, {}", spec.cap), &[], &[SR])?;
    branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "end")))?;
    // First matrix i with w < P0+..+Pi (matrix 2 or 3, then 1, then 0).
    let pair = |n: u8| sr(n, 2);
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
    )?;
    // This wave's matrix-local column m = 8*(w - base) + wave, and whether it exists.
    sop(
        b,
        format!("{} s{ROW0}, s{SW}, s{SBASE}", s_sub_i32(a)),
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
        format!("{} s{ROW0}, s{ROW0}, s{WAVE}", s_add_i32(a)),
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
        format!("s_mul_i32 s{STRIDE}, s{GROUPS}, {}", lit(0x88)),
        &[STRIDE],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{RB}, s{ROW0}, s{STRIDE}"),
        &[RB],
        &[ROW0, STRIDE],
    )?;
    // Weight buffer resource over the selected matrix.
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
    // Parity bookkeeping: wave & 1 equals the matrix-local column parity.
    sop(
        b,
        format!("s_and_b32 s{SPAR}, s{WAVE}, 1"),
        &[SPAR],
        &[WAVE],
    )?;
    sop(
        b,
        format!("s_xor_b32 s{SXOR}, s{SPAR}, 1"),
        &[SXOR],
        &[SPAR],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SBO0}, s{SPAR}, 2"),
        &[SBO0],
        &[SPAR],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{SBO1}, s{SXOR}, 2"),
        &[SBO1],
        &[SXOR],
    )?;
    // Producer X row offsets of every tile: (8t + wave) * K * 4.
    for t in 0..spec.tiles() {
        sop(
            b,
            format!("{} s{TMP}, s{WAVE}, {}", s_add_i32(a), 8 * t),
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
    // Lane-indexed vectors. LDS word of (row v, element j, lane l) is
    // (v*8 + j)*32 + l: the producer stores at v*1024 + 4l (+128 j); the
    // consumer reads elements 0/1 at 4l + 128*{par, 1-par} and 2..7 at 4l.
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
        &[LDC],
        &[LANE],
        &[],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{SPAR}, 7"), &[TMP], &[SPAR])?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{LDA}, s{TMP}, v{LDC}"),
        &[LDA],
        &[LDC],
        &[TMP],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{SXOR}, 7"), &[TMP], &[SXOR])?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{LDB}, s{TMP}, v{LDC}"),
        &[LDB],
        &[LDC],
        &[TMP],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TMP}, s{WAVE}, 10"),
        &[TMP],
        &[WAVE],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{LDST}, s{TMP}, v{LDC}"),
        &[LDST],
        &[LDC],
        &[TMP],
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
    b.wait_all()
}

/// One K group with the compile-time stream `st` (`g % 4`).
fn group(b: &mut Builder, spec: Spec, slot: usize, st: u32) -> R {
    sop(
        b,
        format!("s_mul_i32 s{GOFF}, s{GI}, {}", lit(0x88)),
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
    // Valid waves: the weight group, loaded and dequantized once.
    let wdone = glbl(spec, st, 9, "wdone");
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
    weight_loads(b)?;
    b.wait(load_counter(b.spec.arch), 0)?;
    dequant(b)?;
    b.label(&wdone)?;
    for t in 0..spec.tiles() {
        tile(b, spec, slot, st, t)?;
    }
    Ok(())
}

fn weight_loads(b: &mut Builder) -> R {
    let a = b.spec.arch;
    let policy = if a.gfx12() { " scope:SCOPE_DEV" } else { "" };
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HALF}"),
        &[HOFF],
        &[HALF],
        &[GOFF],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{POFF}, s{GOFF}, v{LDC}"),
        &[POFF],
        &[LDC],
        &[GOFF],
    )?;
    let mut scalar: Vec<u8> = (SRD..SRD + 4).collect();
    scalar.push(RB);
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{HDR}, v{HOFF}, s[{SRD}:{}], s{RB} offen{policy}",
            SRD + 3
        ),
        &[HDR],
        &[HOFF],
        &scalar,
        false,
    )?;
    vmem(
        b,
        &format!(
            "buffer_load_b32 v{PACK}, v{POFF}, s[{SRD}:{}], s{RB} offen offset:8{policy}",
            SRD + 3
        ),
        &[PACK],
        &[POFF],
        &scalar,
        false,
    )
}

/// Eight weights, one f32 rounding each: slot 0/1 take nibble `par` / `1-par`
/// (SGPR bit-field offsets), slots 2..7 nibbles 2..7.
fn dequant(b: &mut Builder) -> R {
    for n in 0..8u8 {
        let w = WS + n;
        let field = match n {
            0 => format!("s{SBO0}"),
            1 => format!("s{SBO1}"),
            _ => format!("{}", n * 4),
        };
        let sc: &[u8] = match n {
            0 => &[SBO0],
            1 => &[SBO1],
            _ => &[],
        };
        valu(
            b,
            &format!("v_bfe_u32 v{w}, v{PACK}, {field}, 4"),
            &[w],
            &[PACK],
            sc,
        )?;
        valu(
            b,
            &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"),
            &[w],
            &[w],
            &[],
        )?;
        valu(
            b,
            &format!("v_fma_mix_f32 v{w}, v{HDR}, v{w}, v{HDR} op_sel:[0,0,1] op_sel_hi:[1,0,1]"),
            &[w],
            &[HDR, w],
            &[],
        )?;
    }
    Ok(())
}

/// Row tile `t` (rows `8t..8t+7`) of one group: fill, barrier, consume, barrier.
fn tile(b: &mut Builder, spec: Spec, slot: usize, st: u32, t: u32) -> R {
    let a = b.spec.arch;
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
        format!("{} s{TMP}, s{WAVE}, {rb}", s_add_i32(a)),
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

/// One row's group dot and its stream add. LDS reads: element slots 0 and 1
/// through the parity bases, 2..7 through the lane base.
fn consume_row(b: &mut Builder, slot: usize, st: u32, row: u32, row_off: u32) -> R {
    for j in 0..8u32 {
        let (addr, imm) = match j {
            0 => (LDA, row_off),
            1 => (LDB, row_off),
            _ => (LDC, row_off + 128 * j),
        };
        let x = XS + j as u8;
        b.ds_load(
            slot,
            Instruction::new(
                format!("ds_load_b32 v{x}, v{addr}{}", off(imm)),
                vec![v(x)],
                vec![v(addr)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    valu(
        b,
        &format!("v_mul_f32_e32 v{DOT}, v{WS}, v{XS}"),
        &[DOT],
        &[WS, XS],
        &[],
    )?;
    for j in 1..8u8 {
        valu(
            b,
            &format!("v_fmac_f32_e32 v{DOT}, v{}, v{}", WS + j, XS + j),
            &[DOT],
            &[DOT, WS + j, XS + j],
            &[],
        )?;
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
    let a = b.spec.arch;
    let cap = u32::from(spec.cap);
    // Fold (a0+a1)+(a2+a3) for every row (rows >= R fold zeros).
    for r in 0..cap {
        for (l, h) in [(0, 1), (2, 3), (0, 2)] {
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
                    format!("ds_swizzle_b32 v{t}, v{val} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
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
            valu(
                b,
                &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - offs),
                &[],
                &[LANE],
                &[],
            )?;
            if a.gfx12() {
                b.push(Instruction::new(
                    "s_wait_alu depctr_va_vcc(0)",
                    vec![],
                    vec![],
                ))?;
            }
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
                format!("{} s{STMP}, s{STMP}, s{R0B}", s_add_i32(a)),
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

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
