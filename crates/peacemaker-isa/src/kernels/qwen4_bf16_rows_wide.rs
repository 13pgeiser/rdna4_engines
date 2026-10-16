// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exact WIDE verify twins of the ordinary BF16 GEMV (`gemv_bf16_xf32_x4`,
//! its row extension `gemv_bf16_xf32_x4_rows_r<R>`): the PM entries
//! `gemv_bf16_xf32_x4_rows_wide_r{16,32}_pm_gfx1151`, module
//! `bf16_rows_wide_pm_gfx1151`. Native gfx1151 only: `Spec { arch, cap }`
//! is accepted but any other `arch` is refused by [`Spec::validate`].
//!
//! CONTRACT STATUS: this is a schedule/layout transcription of the frozen
//! design (`WideTwinGemvSol.md` §2, §3 BF16) onto the checked builder, with the
//! multiply-add operand orientation taken from the extracted gfx1151 decode
//! object (`v_fmac_f32 acc, X, W`; fold `(a1+a0)`, `(a3+a2)`, `(a3+a2)+(a1+a0)`).
//! NO numerical certification exists yet: nothing here has been run, and byte
//! equality with R independent single-row launches is established only by the
//! oracle the integration owner runs (CPU emitter build first, then the GPU
//! row oracle). Scope is exactly the ordinary BF16 x4 / single projection
//! (router, selector, shared gate/up, one matrix) and the HC k4 projection
//! (`gemv_bf16_xf32_k4_rows_wide_r{16,32}_pm_gfx1151`, kernarg 48 B, static LDS
//! 16384 B, see the `hc` module below). The HC value is the exact scalar
//! `(q0+q1)+(q2+q3)` fold followed by the incumbent activation (BF16 round
//! trips, scale, `v / (1 + expf(-v))`) from `qwen4_bf16_hc_epilogue`, with the
//! exp and the divide emitted exactly, no shortcut or approximation. The HIP
//! twins in `qwen4_gemv_bf16_xf32.hip` are retained as the fallback. Q8, MQ4
//! and MQ6 are other files.
//!
//! ABI: the narrow MQ4 PM entry's 96 bytes, unchanged: `A0@0 A1@8 A2@16 A3@24`
//! (raw BF16 weight rows, `[Mi, K]` u16), `x@32` (`[R, K]` f32), `y0@40 y1@48
//! y2@56 y3@64` (`[R, Mi]` f32), `M0@72 M1@76 M2@80 M3@84 K@88`, `R@92` (the
//! actual row count `1..=cap`; any other `R` returns before any barrier and
//! before any store). Unused matrices have `M = 0`.
//!
//! Geometry: grid `[sum_i ceil(M_i/8), 1, 1]`, block `[256,1,1]` (eight wave32),
//! static LDS 8192 B (one 8-row x K256 X tile), `cu_mode = false` (WGP mode).
//! Workgroup `w` routes to the first matrix `i` with `w < P0+..+Pi`
//! (`Pi = ceil(Mi/8)`); no workgroup spans two matrices. Wave `v`
//! (`threadIdx.x >> 5`) owns ONE output column `m = 8*(w - base_i) + v`
//! (matrix-local) for every row `r < R`: `cap` rows' independent accumulators
//! (4 streams each, `4*cap` VGPRs) live in that wave for the whole K sweep.
//!
//! Arithmetic (the decode object `gemv_bf16_xf32_row`, per `(row, column)`):
//!  * `K % 8 == 0` (production): lane `l` owns elements `g*256 + 8l + j`,
//!    `j = 0..7`. Weights: one `buffer_load_b128` (8 u16) per lane per group;
//!    widening is bitwise (`u16 << 16` for the low half, `& 0xffff0000` for the
//!    high half of each dword). Per group and row, `a[j & 3] = fma(X_j, W_j,
//!    a[j & 3])` for `j = 0..7` increasing (`v_fmac_f32 acc, X, W`), i.e. chain
//!    `a0: j0 then j4`, `a1: j1 then j5`, `a2: j2 then j6`, `a3: j3 then j7`.
//!    There is no group dot and no separate add. Groups strictly ascending.
//!    PARTIAL256: when `g*256 + 256 > K` the lane is active only if
//!    `g*256 + 8l + 8 <= K`; inactive lanes skip the weight load, the X
//!    producer load/store and every consuming FMA (EXEC mask, never zero
//!    padding). The mask is restored to full EXEC before every barrier and no
//!    memory operation is pending when it changes.
//!  * `K % 8 != 0` (scalar domain, exact incumbent all-scalar path): lane `l`
//!    owns `k = l + 32 t`, `t` ascending, ONE chain `a0 = fma(X, W, a0)`; `a1..a3`
//!    stay `+0` and are still folded. The tile is K32 (one weight u16 and one X
//!    f32 per lane, `X0`/`W0` only), same 8-row/two-barrier schedule; lanes with
//!    `l + 32 t >= K` are masked identically to the above.
//!  * Fold per row, source operand order as in the extracted object:
//!    `a1 = a1 + a0`, `a3 = a3 + a2`, `a3 = a3 + a1` (finite-equal to
//!    `(a0+a1)+(a2+a3)`; the emitted orientation is kept), then the
//!    16/8/4/2/1 `ds_swizzle`/`ds_bpermute` own+other tree (out-of-range lanes
//!    add their own value, `shfl_down` semantics); lane 0 stores.
//!
//! Schedule (design §2.2): per K tile (ascending) each valid wave loads and
//! widens its weights ONCE (retired before use); then for row tiles `rb = 0, 8,
//! 16, 24` (`rb < R`): producer wave `v` loads X row `rb+v` and stores the
//! unchanged f32 bits to LDS word `(v*8 + j)*32 + lane`; workgroup barrier;
//! every valid wave consumes the tile's rows (LDS read, the FMAs, no weight
//! reload); workgroup barrier before the tile is overwritten. A wave with
//! `m >= M_i` still produces X and executes every barrier; it skips the weight
//! loads, the arithmetic and the stores and leaves at the epilogue entry. All
//! barrier decisions depend only on `R`, `K` and the workgroup id, so they are
//! uniform. Rows `>= R` are neither loaded, consumed nor stored.
//!
//! Branch discipline of the checked builder: its wait ledger is linear, so no
//! conditional region leaves a memory operation pending and none is entered
//! with one pending (every region drains with explicit zero-count waits).
//!
//! Resource high-water (checked in u16/usize arithmetic, rejected above the
//! budget and above 256): VGPR `ACC0 + 4*cap` (R16: 104 of 128, R32: 168 of
//! 192); SGPR 64 of 80; LDS 8192 B; block 256, WGP mode.
//!
//! The same module also holds the HC k4 twins
//! `gemv_bf16_xf32_k4_rows_wide_r{16,32}_pm_gfx1151` ([`emit_hc`], schedule and
//! register map in the `hc` module below; VGPR as above, SGPR 80 of 80, LDS
//! 16384 B). Module symbol order: x4 R16, x4 R32, k4 R16, k4 R32.
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, sr, v, SRD_WORD3,
};
use crate::lds::Transition;
use crate::ledger::Counter;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

/// The kernarg segment size (the narrow MQ4 entry's).
pub const KERNARG_BYTES: u32 = 96;
/// Output columns (waves) per workgroup.
pub const M_BLOCK: u32 = 8;
/// Threads per workgroup.
pub const BLOCK: u16 = 256;
/// Static LDS bytes: 8 rows x 256 K x 4 B.
pub const LDS_BYTES: u32 = 8192;
/// Compiled row capacities.
pub const CAPS: [u8; 2] = [16, 32];
/// HC k4 kernarg segment: `W@0 X@8 Y@16 M@24 K@28 scale@32 hc_act@36 R@40`
/// (44 logical bytes, padded to 48 for the metadata's 8-byte alignment).
pub const HC_KERNARG_BYTES: u32 = 48;
/// HC k4 static LDS bytes: 4 rows x 4 quarters x 256 K x 4 B.
pub const HC_LDS_BYTES: u32 = 16384;

/// One wide entry: row capacity `cap` (16 or 32). gfx1151 only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub arch: Arch,
    pub cap: u8,
}
impl Spec {
    pub fn symbol(&self) -> String {
        format!(
            "gemv_bf16_xf32_x4_rows_wide_r{}_pm_{}",
            self.cap,
            self.arch.name()
        )
    }
    pub fn symbol_hc(&self) -> String {
        format!(
            "gemv_bf16_xf32_k4_rows_wide_r{}_pm_{}",
            self.cap,
            self.arch.name()
        )
    }
    /// The module `bf16_rows_wide_pm_<arch>` (frozen name; the source kernel id stays `qwen4_bf16_rows_wide`).
    pub fn module(arch: Arch) -> String {
        format!("bf16_rows_wide_pm_{}", arch.name())
    }
    pub fn validate(&self) -> R {
        if self.arch != Arch::Gfx1151 {
            return Err("qwen4_bf16_rows_wide builder is exact-gfx1151 only".into());
        }
        if !CAPS.contains(&self.cap) {
            return Err(format!(
                "qwen4_bf16_rows_wide capacity {} is not 16 or 32",
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
const NT32: u8 = 44;
const TMP: u8 = 45;
/// Weight row byte offset (64-bit: low / high), added to the descriptor base.
const WRL: u8 = 46;
const WRH: u8 = 47;
/// Active-lane EXEC mask of the current tile, and the K remaining at its start.
const EM: u8 = 48;
const KREM: u8 = 49;
const GOFF: u8 = 50;
const XG: u8 = 51;
const GI: u8 = 52;
const KB: u8 = 53;
const MB: u8 = 54;
const STMP: u8 = 55;
const SVAL: u8 = 56;
const R0B: u8 = 57;
/// `K & 7`: nonzero selects the scalar domain. And `2*K`, the weight row bytes.
const SPATH: u8 = 58;
const KB2: u8 = 59;
/// `XT[t] = (8t + wave) * K * 4`: the byte offset of tile `t`'s producer row.
const XT: u8 = 60;
/// High-water SGPR index + 1.
const SGPR_HIGH: u16 = XT as u16 + 4;
const SGPR_BUDGET: u16 = 80;

// ----------------------------------------------------------------------- VGPRs
const TID: u8 = 0;
const HOFF: u8 = 1;
const XOFF: u8 = 3;
const SHUF: u8 = 4;
const LANE: u8 = 5;
/// LDS producer store address and the consumer read base.
const LDST: u8 = 9;
const LDC: u8 = 12;
const VT: u8 = 13;
/// Widened weight slots W0..W7.
const WS: u8 = 16;
/// X slots X0..X7 (the producer's staging and the consumer's row vector).
const XS: u8 = 24;
/// Reduction temporaries / store addresses (16 live), with `XS`.
const TS: u8 = 32;
const ACC0: u16 = 40;

/// High-water VGPR index + 1 for `cap` rows, in u16 arithmetic.
fn vgpr_high_water(cap: u8) -> usize {
    usize::from(ACC0) + 4 * usize::from(cap)
}
fn acc(row: u32, stream: u32) -> u8 {
    let n = usize::from(ACC0) + 4 * row as usize + stream as usize;
    u8::try_from(n).expect("accumulator index validated by layout()")
}
/// Validate the register layout of `spec`: the high-water marks must fit the
/// budgets and the architectural limits.
fn layout(spec: Spec) -> Result<(), String> {
    let hw = vgpr_high_water(spec.cap);
    if hw > 256 {
        return Err(format!("bf16 wide r{} needs {hw} VGPRs (> 256)", spec.cap));
    }
    if hw > usize::from(spec.vgpr_budget()) {
        return Err(format!(
            "bf16 wide r{} needs {hw} VGPRs (> budget {})",
            spec.cap,
            spec.vgpr_budget()
        ));
    }
    if SGPR_HIGH > SGPR_BUDGET {
        return Err(format!(
            "bf16 wide needs {SGPR_HIGH} SGPRs (> budget {SGPR_BUDGET})"
        ));
    }
    Ok(())
}

fn end_label(spec: Spec) -> String {
    format!(".Lbf16w{}_end", spec.cap)
}

fn plan(spec: Spec) -> Result<RegPlan, String> {
    layout(spec)?;
    let mut regs = RegPlan::new(spec.vgpr_budget(), SGPR_BUDGET)?;
    for (name, base, len) in [
        ("kernarg", KARG, 2),
        ("workgroup_id", WGIN, 1),
        ("weight_resource", SRD, 4),
        ("weight_pointers", PA, 8),
        ("x_y_pointers", PXY, 8),
        ("y3_m01", Y3M, 4),
        ("m23_k_r", M2KB, 4),
        ("y_selected", SY, 2),
        ("m_selected", SM, 1),
        ("matrix_base", SBASE, 1),
        ("blocks_0", CP0, 1),
        ("blocks_01", CP1, 1),
        ("blocks_012", CP2, 1),
        ("blocks_0123", CP3, 1),
        ("workgroup", SW, 1),
        ("wave", WAVE, 1),
        ("column", ROW0, 1),
        ("groups", GROUPS, 1),
        ("tiles32", NT32, 1),
        ("temp", TMP, 1),
        ("weight_row_lo", WRL, 1),
        ("weight_row_hi", WRH, 1),
        ("exec_mask", EM, 1),
        ("k_remaining", KREM, 1),
        ("group_offset", GOFF, 1),
        ("x_group_offset", XG, 1),
        ("group_index", GI, 1),
        ("k_bytes", KB, 1),
        ("m_bytes", MB, 1),
        ("store_temp", STMP, 1),
        ("column_valid", SVAL, 1),
        ("column_bytes", R0B, 1),
        ("scalar_domain", SPATH, 1),
        ("weight_row_bytes", KB2, 1),
        ("x_tile_offsets", XT, 4),
    ] {
        regs.add_range(name, Kind::S, base, len, Live::Whole)?;
    }
    for (name, base, len) in [
        ("tid", TID, 1),
        ("weight_offset", HOFF, 1),
        ("x_offset", XOFF, 1),
        ("shuffle_addr", SHUF, 1),
        ("lane", LANE, 1),
        ("lds_store", LDST, 1),
        ("lds_read", LDC, 1),
        ("temp", VT, 1),
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

fn load_counter() -> Counter {
    Counter::Vm
}
fn s_sub_i32() -> &'static str {
    "s_sub_i32"
}
fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn lbl(spec: Spec, name: &str) -> String {
    format!(".Lbf16w{}_{name}", spec.cap)
}
fn glbl(spec: Spec, tag: &str, tile: u32, name: &str) -> String {
    format!(".Lbf16w{}_{tag}t{tile}_{name}", spec.cap)
}

/// Which K schedule a tile belongs to: the vector domain with compile-time
/// stream `g % 4` (unused by the arithmetic, which streams by `j & 3`; kept as
/// the unroll slot), or the K32 scalar domain.
#[derive(Clone, Copy)]
enum Mode {
    Fast(u32),
    Scalar,
}
impl Mode {
    fn tag(self) -> String {
        match self {
            Mode::Fast(st) => format!("s{st}"),
            Mode::Scalar => "sc".into(),
        }
    }
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
        kernel_id: "qwen4_bf16_rows_wide".into(),
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
    // Zero the 4*cap accumulators (+0, as the incumbent's `0.0f`).
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
    sop(&mut b, format!("s_cmp_lg_u32 s{SPATH}, 0"), &[], &[SPATH])?;
    branch(&mut b, &format!("s_cbranch_scc1 {}", lbl(spec, "scalar")))?;
    // Vector domain (K % 8 == 0): K256 groups, unrolled by four.
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
            fast_group(b, spec, slot, st)?;
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
    branch(&mut b, &format!("s_branch {}", lbl(spec, "fold")))?;
    // Scalar domain (K % 8 != 0, K >= 1): K32 tiles, one chain.
    b.label(&lbl(spec, "scalar"))?;
    b.loop_(&lbl(spec, "tile32"), |b| {
        scalar_step(b, spec, slot)?;
        sop(
            b,
            format!("{} s{GI}, s{GI}, 1", s_add_i32(spec.arch)),
            &[GI],
            &[GI],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{GI}, s{NT32}"), &[], &[GI, NT32])?;
        branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "tile32")))
    })?;
    b.label(&lbl(spec, "fold"))?;
    // Waves past M never reach another barrier: they leave here.
    sop(&mut b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(&mut b, &format!("s_cbranch_scc1 {}", end_label(spec)))?;
    epilogue(&mut b, spec)?;
    b.label(&end_label(spec))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// The module `bf16_rows_wide_pm_<arch>`: the ordinary x4 entries
/// (R16, R32), then the HC k4 entries (R16, R32). The shared module combiner
/// keeps exactly one end-of-text guard (`emit::CODE_END_TAIL`).
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let mut emitted = Vec::new();
    for cap in CAPS {
        emitted.push(emit(Spec { arch, cap })?);
    }
    for cap in CAPS {
        emitted.push(emit_hc(Spec { arch, cap })?);
    }
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}

/// The HC k4 twin `gemv_bf16_xf32_k4_rows_wide_r<cap>_pm_gfx1151`.
pub fn emit_hc(spec: Spec) -> Result<Emitted, String> {
    hc::emit(spec)
}

fn exec_set(b: &mut Builder) -> R {
    op(b, format!("s_mov_b32 exec_lo, s{EM}"), &[], &[s(EM)])
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
fn exec_lane0(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}

/// bf16 -> f32 is bitwise: low half `<< 16`, high half `& 0xffff0000`. The
/// dwords sit in `WS..WS+3` (one `buffer_load_b128`); widen in place, dword 3
/// first, so no dword is overwritten before it is read.
fn widen_weights(b: &mut Builder) -> R {
    let w = |i: u8| WS + i;
    for d in (0..4u8).rev() {
        valu(
            b,
            &format!("v_and_b32_e32 v{}, 0xffff0000, v{}", w(2 * d + 1), w(d)),
            &[w(2 * d + 1)],
            &[w(d)],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{}, 16, v{}", w(2 * d), w(d)),
            &[w(2 * d)],
            &[w(d)],
            &[],
        )?;
    }
    Ok(())
}

fn prologue(b: &mut Builder, spec: Spec) -> R {
    let a = b.spec.arch;
    let end = end_label(spec);
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
    sop(b, format!("s_mov_b32 s{SW}, s{WGIN}"), &[SW], &[WGIN])?;
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
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
    sop(b, format!("s_cmp_gt_u32 s{SR}, {}", spec.cap), &[], &[SR])?;
    branch(b, &format!("s_cbranch_scc1 {end}"))?;
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
        format!("{} s{ROW0}, s{SW}, s{SBASE}", s_sub_i32()),
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
    // K domain: K256 groups for K % 8 == 0, K32 tiles for the scalar domain.
    sop(b, format!("s_and_b32 s{SPATH}, s{SK}, 7"), &[SPATH], &[SK])?;
    sop(
        b,
        format!("{} s{GROUPS}, s{SK}, {}", s_add_i32(a), lit(255)),
        &[GROUPS],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{GROUPS}, s{GROUPS}, 8"),
        &[GROUPS],
        &[GROUPS],
    )?;
    sop(
        b,
        format!("{} s{NT32}, s{SK}, 31", s_add_i32(a)),
        &[NT32],
        &[SK],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{NT32}, s{NT32}, 5"),
        &[NT32],
        &[NT32],
    )?;
    // Weight buffer resource over the selected matrix, advanced by this wave's
    // row (64-bit: ROW0 * 2K); a column past M is never loaded from.
    sop(b, format!("s_lshl_b32 s{KB2}, s{SK}, 1"), &[KB2], &[SK])?;
    sop(
        b,
        format!("s_mul_i32 s{WRL}, s{ROW0}, s{KB2}"),
        &[WRL],
        &[ROW0, KB2],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{WRH}, s{ROW0}, s{KB2}"),
        &[WRH],
        &[ROW0, KB2],
    )?;
    sop(
        b,
        format!("{} s{SRD}, s{SRD}, s{WRL}", s_add_u32(a)),
        &[SRD],
        &[SRD, WRL],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, s{WRH}", s_addc_u32(a), SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1, WRH],
    )?;
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
    // consumer reads row r element j at 1024 r + 4l + 128 j.
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
        &[LDC],
        &[LANE],
        &[],
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
    b.wait_all()
}

/// Active-lane mask of the current tile from `KREM` (K left at its start):
/// the low `KREM >> shift` lanes, or all of them when `KREM >= full`.
/// A shift/invert forms the low-lane mask; the select handles a full wave.
fn tile_mask(b: &mut Builder, shift: u32, full: u32) -> R {
    sop(
        b,
        format!(
            "s_lshl_b32 s{KREM}, s{GI}, {}",
            if shift == 3 { 8 } else { 5 }
        ),
        &[KREM],
        &[GI],
    )?;
    sop(
        b,
        format!("{} s{KREM}, s{SK}, s{KREM}", s_sub_i32()),
        &[KREM],
        &[SK, KREM],
    )?;
    if shift == 3 {
        sop(b, format!("s_lshr_b32 s{EM}, s{KREM}, 3"), &[EM], &[KREM])?;
    } else {
        sop(b, format!("s_mov_b32 s{EM}, s{KREM}"), &[EM], &[KREM])?;
    }
    sop(b, format!("s_lshl_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
    sop(b, format!("s_not_b32 s{EM}, s{EM}"), &[EM], &[EM])?;
    sop(
        b,
        format!("s_cmp_ge_u32 s{KREM}, {}", lit(full)),
        &[],
        &[KREM],
    )?;
    sop(b, format!("s_cselect_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])
}

/// One K256 group (vector domain), compile-time unroll slot `st`.
fn fast_group(b: &mut Builder, spec: Spec, slot: usize, st: u32) -> R {
    let mode = Mode::Fast(st);
    sop(b, format!("s_lshl_b32 s{GOFF}, s{GI}, 9"), &[GOFF], &[GI])?;
    sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 10"), &[XG], &[GI])?;
    tile_mask(b, 3, 256)?;
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
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HOFF}, 4, v{LANE}"),
        &[HOFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
        &[HOFF],
        &[HOFF],
        &[GOFF],
    )?;
    // Valid waves: the weight group, loaded and widened once, active lanes only.
    let wdone = glbl(spec, &mode.tag(), 9, "wdone");
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
    exec_set(b)?;
    let w = |i: u8| WS + i;
    vmem(
        b,
        &format!(
            "buffer_load_b128 v[{}:{}], v{HOFF}, s[{SRD}:{}], 0 offen",
            WS,
            WS + 3,
            SRD + 3
        ),
        &[w(0), w(1), w(2), w(3)],
        &[HOFF],
        &[SRD, SRD + 1, SRD + 2, SRD + 3],
        false,
    )?;
    b.wait(load_counter(), 0)?;
    // bf16 -> f32 is bitwise: low half `<< 16`, high half `& 0xffff0000`. Done
    // in place, dword 3 first, so no dword is overwritten before it is read.
    widen_weights(b)?;
    exec_all(b)?;
    b.label(&wdone)?;
    for t in 0..spec.tiles() {
        tile(b, spec, slot, mode, t)?;
    }
    Ok(())
}

/// One K32 tile (scalar domain): one weight u16 and one X f32 per lane.
fn scalar_step(b: &mut Builder, spec: Spec, slot: usize) -> R {
    let mode = Mode::Scalar;
    sop(b, format!("s_lshl_b32 s{GOFF}, s{GI}, 6"), &[GOFF], &[GI])?;
    sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 7"), &[XG], &[GI])?;
    tile_mask(b, 0, 32)?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{XOFF}, 2, v{LANE}"),
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
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{HOFF}, 1, v{LANE}"),
        &[HOFF],
        &[LANE],
        &[],
    )?;
    valu(
        b,
        &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
        &[HOFF],
        &[HOFF],
        &[GOFF],
    )?;
    let wdone = glbl(spec, &mode.tag(), 9, "wdone");
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
    exec_set(b)?;
    vmem(
        b,
        &format!(
            "buffer_load_u16 v{WS}, v{HOFF}, s[{SRD}:{}], 0 offen",
            SRD + 3
        ),
        &[WS],
        &[HOFF],
        &[SRD, SRD + 1, SRD + 2, SRD + 3],
        false,
    )?;
    b.wait(load_counter(), 0)?;
    valu(
        b,
        &format!("v_lshlrev_b32_e32 v{WS}, 16, v{WS}"),
        &[WS],
        &[WS],
        &[],
    )?;
    exec_all(b)?;
    b.label(&wdone)?;
    for t in 0..spec.tiles() {
        tile(b, spec, slot, mode, t)?;
    }
    Ok(())
}

/// Row tile `t` (rows `8t..8t+7`) of one tile: fill, barrier, consume, barrier.
/// Every EXEC mask is restored before a barrier and nothing is pending then.
fn tile(b: &mut Builder, spec: Spec, slot: usize, mode: Mode, t: u32) -> R {
    let a = b.spec.arch;
    let rb = 8 * t;
    let tag = mode.tag();
    let done = glbl(spec, &tag, t, "tile_done");
    // The whole tile is past R: skipped by every wave of the workgroup.
    if t > 0 {
        sop(b, format!("s_cmp_le_u32 s{SR}, {rb}"), &[], &[SR])?;
        branch(b, &format!("s_cbranch_scc1 {done}"))?;
    }
    // Producer wave `v` stages X row rb+v when it exists, active lanes only.
    let nofill = glbl(spec, &tag, t, "nofill");
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
    exec_set(b)?;
    match mode {
        Mode::Fast(_) => {
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
        }
        Mode::Scalar => {
            vmem(
                b,
                &format!("global_load_b32 v{XS}, v{VT}, s[{XP}:{}]", XP + 1),
                &[XS],
                &[VT],
                &[XP, XP + 1],
                false,
            )?;
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{XS}"),
                    vec![],
                    vec![v(LDST), v(XS)],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
    }
    b.wait_all()?;
    exec_all(b)?;
    b.label(&nofill)?;
    b.barrier(&[Transition::Ready(slot)])?;
    // Valid waves consume rows rb.. in ascending order while row < R.
    let cdone = glbl(spec, &tag, t, "cdone");
    sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
    branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
    exec_set(b)?;
    for r in 0..8u32 {
        if r > 0 {
            sop(b, format!("s_cmp_le_u32 s{SR}, {}", rb + r), &[], &[SR])?;
            branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
        }
        consume_row(b, slot, mode, rb + r, 1024 * r, LDC)?;
    }
    b.label(&cdone)?;
    exec_all(b)?;
    b.barrier(&[Transition::Retire(slot)])?;
    b.label(&done)
}

/// One row's chains advanced by one tile. Vector domain: `a[j&3] =
/// fma(X_j, W_j, a[j&3])`, `j = 0..7` (the extracted object's `v_fmac_f32 acc,
/// X, W`). Scalar domain: `a0 = fma(X0, W0, a0)`.
fn consume_row(b: &mut Builder, slot: usize, mode: Mode, row: u32, row_off: u32, base: u8) -> R {
    let n = match mode {
        Mode::Fast(_) => 8u32,
        Mode::Scalar => 1,
    };
    for j in 0..n {
        let x = XS + j as u8;
        b.ds_load(
            slot,
            Instruction::new(
                format!("ds_load_b32 v{x}, v{base}{}", off(row_off + 128 * j)),
                vec![v(x)],
                vec![v(base)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
    }
    for j in 0..n {
        let a = acc(row, j & 3);
        valu(
            b,
            &format!("v_fmac_f32_e32 v{a}, v{}, v{}", XS + j as u8, WS + j as u8),
            &[a],
            &[a, XS + j as u8, WS + j as u8],
            &[],
        )?;
    }
    Ok(())
}

/// Per-row fold and wave reduction of the first `cap` rows' accumulators; the
/// result of row `r` is left in `acc(r, 3)` of every lane (lane 0 is the sum).
fn reduce_rows(b: &mut Builder, cap: u32) -> R {
    // Fold in the extracted object's source order: a1 = a1 + a0, a3 = a3 + a2,
    // a3 = a3 + a1 (rows >= R fold zeros). The folded value is stream 3.
    for r in 0..cap {
        for (d, x) in [(1, 0), (3, 2), (3, 1)] {
            let (d, x) = (acc(r, d), acc(r, x));
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
            let (val, t) = (acc(r, 3), tmp(i));
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
            let (val, t) = (acc(r, 3), tmp(i));
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
                let (val, t) = (acc(r, 3), tmp(i));
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
                let (val, t) = (acc(r, 3), tmp(i));
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
    Ok(())
}

fn epilogue(b: &mut Builder, spec: Spec) -> R {
    let a = b.spec.arch;
    let end = end_label(spec);
    let cap = u32::from(spec.cap);
    reduce_rows(b, cap)?;
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
                    acc(r, 3),
                    SY + 1
                ),
                &[],
                &[XS + i as u8, acc(r, 3)],
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
            branch(b, &format!("s_cbranch_scc1 {end}"))?;
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
pub(super) fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
pub(super) fn vmem(
    b: &mut Builder,
    text: &str,
    defs: &[u8],
    uses: &[u8],
    scalar: &[u8],
    store: bool,
) -> R {
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
pub(super) fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}

/// HC k4 wide twin (`gemv_bf16_xf32_k4_rows_wide_r{16,32}_pm_gfx1151`).
///
/// CONTRACT STATUS: same as the ordinary entries: a schedule/layout
/// transcription of design §5 with the extracted decode object's operand
/// orientation; not run, no numerical certification. The activation epilogue
/// is `qwen4_bf16_hc_epilogue::emit_value` (its own contract).
///
/// ABI: `W@0 X@8 Y@16 M@24 K@28 scale@32 hc_act@36 R@40` (44 logical bytes,
/// segment 48). `W` is `[M, K]` raw BF16, `X` `[R, K]` f32, `Y` `[R, M]` f32.
/// Grid `[ceil(M/2), 1, 1]`, block 256, static LDS 16384 B, WGP mode. Any
/// workgroup with `2*wg >= M`, any `R` outside `1..=cap` and any `K % 32 != 0`
/// returns before the first barrier and before any store (the host enforces
/// `K % 32 == 0`: the quarter `K/4` is then a multiple of 8, so only the
/// vector domain exists here).
///
/// Wave `w` owns output column `m = 2*wg + (w >> 2)` and quarter `q = w & 3`
/// (`part = K/4`, elements `q*part .. (q+1)*part`) for every row `r < R`:
/// the same per-chain DAG as the ordinary kernel on local `K = part` (lane
/// `l`, `part`-local element `256 t + 8 l + j`, `a[j&3] = fma(X_j, W_j, a[j&3])`,
/// partial256 lanes with `8 l + 8 > part - 256 t` masked), then the ordinary
/// fold/shuffle per `(row, quarter)`, which is the incumbent quarter sum.
///
/// Schedule: per local tile, each valid wave loads/widens its quarter weight
/// vector once; X is staged four rows x four quarters (16 KiB) per stage:
/// vector `v = 4*rl + qq` is row `rb + rl`, quarter `qq`, LDS bytes
/// `v*1024 + 128 j + 4 lane`; producer wave `w` stages vectors `w` and `w + 8`
/// (rows `rb + (w>>2)`, `rb + (w>>2) + 2`, quarter `w & 3`). Barrier; every
/// valid wave consumes rows `rb..rb+3` from its own quarter's vectors
/// (`4096 rl + 1024 q + ...`); barrier. After the final drain barrier the LDS
/// holds `partial[mloc][r][q]` f32 at byte `((mloc*cap + r)*4 + q)*4`
/// (lane 0 of every valid wave writes its rows' quarter sums); barrier; item
/// `n = w + 8 k` (`mloc = n / cap`, `r = n % cap`) is folded by wave `n % 8`:
/// `ds_load_b128` of the four quarters, `v_readfirstlane` into
/// `s64..s67`, `emit_value` (`(q0+q1)+(q2+q3)` and the activation, result in
/// `s68`), one lane-0 store to `Y[r*M + m]`. Invalid column waves still
/// produce X and execute every barrier. All barrier decisions are uniform
/// functions of `R`, `K`, `M` and the workgroup id.
///
/// Resource high-water: VGPR `ACC0 + 4*cap` (R16 104/128, R32 168/192);
/// SGPR 80/80 (`s0..s50` own, `s64..s79` the epilogue block); LDS 16384 B.
mod hc {
    use super::{
        acc, branch, consume_row, exec_all, exec_lane0, layout, load_counter, off, reduce_rows,
        s_sub_i32, valu, vmem, widen_weights, Mode, Spec, BLOCK, HC_KERNARG_BYTES, HC_LDS_BYTES,
        HOFF, LANE, LDC, LDST, R, SGPR_BUDGET, SHUF, TID, TS, VT, WS, XOFF, XS,
    };
    use super::{lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, sr, v, SRD_WORD3};
    use crate::insn::{Instruction, MemoryClass};
    use crate::kernels::qwen4_bf16_hc_epilogue::{self as epi, Regs};
    use crate::lds::Transition;
    use crate::plan::Access;
    use crate::reg::{Kind, Live};
    use crate::{Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

    // SGPRs.
    const KARG: u8 = 0;
    const WGIN: u8 = 2;
    const SRD: u8 = 4;
    /// Kernarg words 0..8: `W@8:9 X@10:11 Y@12:13 M@14 K@15`; words 8..12: `scale@16 hc_act@17 R@18`.
    const KA: u8 = 8;
    const KB_: u8 = 16;
    const PW: u8 = 8;
    const PX: u8 = 10;
    const PY: u8 = 12;
    const SM: u8 = 14;
    const SK: u8 = 15;
    const SCALE: u8 = 16;
    const ACT: u8 = 17;
    const SR: u8 = 18;
    const SW: u8 = 20;
    const WAVE: u8 = 21;
    const MLOC: u8 = 22;
    const QI: u8 = 23;
    const COL: u8 = 24;
    const SVAL: u8 = 25;
    const PART: u8 = 26;
    const GROUPS: u8 = 27;
    const TMP: u8 = 28;
    const TMP2: u8 = 29;
    const WRL: u8 = 30;
    const WRH: u8 = 31;
    const EM: u8 = 32;
    const KREM: u8 = 33;
    const GOFF: u8 = 34;
    const XG: u8 = 35;
    const GI: u8 = 36;
    const KB: u8 = 37;
    const MB: u8 = 38;
    const K2B: u8 = 39;
    const XWB: u8 = 40;
    const WRB: u8 = 41;
    const PBASE: u8 = 42;
    const MBASE: u8 = 44;
    const IR: u8 = 45;
    const YB: u8 = 46;
    const QOFF: u8 = 47;
    const PARTB: u8 = 48;
    /// The epilogue helper's block: quarters, result, temporaries.
    const HQ: u8 = 64;
    const HOUT: u8 = 68;
    const HTMP: u8 = 69;
    // VGPRs beyond the shared map: the consumer read base and the partial store address; the
    // item registers (dead accumulator space after the partial stores).
    const LDR: u8 = 10;
    const VPART: u8 = 15;
    const VITA: u8 = 40;
    const VITQ: u8 = 41;
    const VYA: u8 = 45;
    const VYD: u8 = 46;

    fn lbl(spec: Spec, name: &str) -> String {
        format!(".Lbf16hc{}_{name}", spec.cap)
    }
    fn exec_set(b: &mut Builder) -> R {
        op(b, format!("s_mov_b32 exec_lo, s{EM}"), &[], &[s(EM)])
    }

    fn plan(spec: Spec) -> Result<RegPlan, String> {
        layout(spec)?;
        let mut regs = RegPlan::new(spec.vgpr_budget(), SGPR_BUDGET)?;
        for (name, base, len) in [
            ("kernarg", KARG, 2),
            ("workgroup_id", WGIN, 1),
            ("weight_resource", SRD, 4),
            ("kernarg_ptrs_m_k", KA, 8),
            ("kernarg_scale_act_r", KB_, 4),
            ("workgroup", SW, 1),
            ("wave", WAVE, 1),
            ("column_local", MLOC, 1),
            ("quarter", QI, 1),
            ("column", COL, 1),
            ("column_valid", SVAL, 1),
            ("part", PART, 1),
            ("groups", GROUPS, 1),
            ("temp", TMP, 1),
            ("temp2", TMP2, 1),
            ("weight_row_lo", WRL, 1),
            ("weight_row_hi", WRH, 1),
            ("exec_mask", EM, 1),
            ("k_remaining", KREM, 1),
            ("group_offset", GOFF, 1),
            ("x_group_offset", XG, 1),
            ("group_index", GI, 1),
            ("k_bytes", KB, 1),
            ("m_bytes", MB, 1),
            ("k2_bytes", K2B, 1),
            ("x_wave_base", XWB, 1),
            ("weight_row_bytes", WRB, 1),
            ("partial_base", PBASE, 1),
            ("m_base", MBASE, 1),
            ("item_row", IR, 1),
            ("y_bytes", YB, 1),
            ("quarter_offset", QOFF, 1),
            ("part_bytes", PARTB, 1),
            ("epilogue_q", HQ, 4),
            ("epilogue_out", HOUT, 1),
            ("epilogue_tmp0", HTMP, 1),
            ("epilogue_tmp1", HTMP + 1, 2),
            ("epilogue_tmp2", HTMP + 3, 8),
        ] {
            regs.add_range(name, Kind::S, base, len, Live::Whole)?;
        }
        for (name, base, len) in [
            ("tid", TID, 1),
            ("weight_offset", HOFF, 1),
            ("x_offset", XOFF, 1),
            ("shuffle_addr", SHUF, 1),
            ("lane", LANE, 1),
            ("lds_store", LDST, 1),
            ("lds_read_quarter", LDR, 1),
            ("lds_lane", LDC, 1),
            ("temp", VT, 1),
            ("partial_addr", VPART, 1),
            ("weights", WS, 8),
            ("x_slots", XS, 8),
            ("temps", TS, 8),
        ] {
            regs.add_range(name, Kind::V, base, len, Live::Whole)?;
        }
        for block in 0..(4 * usize::from(spec.cap) / 8) {
            let base = u8::try_from(usize::from(super::ACC0) + 8 * block)
                .map_err(|_| "accumulator block past v255".to_string())?;
            regs.add_range(&format!("acc_block{block}"), Kind::V, base, 8, Live::Whole)?;
        }
        Ok(regs)
    }

    pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
        spec.validate()?;
        let arch = spec.arch;
        let kernargs = KernargLayout::new(HC_KERNARG_BYTES)
            .pointer_access("W", 0, Access::ReadOnly)
            .pointer_access("X", 8, Access::ReadOnly)
            .pointer_access("Y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("scale", 32, 4, "by_value")
            .hidden("hc_act", 36, 4, "by_value")
            .hidden("R", 40, 4, "by_value");
        let kspec = KernelSpec {
            kernel_id: "qwen4_bf16_rows_wide_hc".into(),
            variant: format!("wide_hc_{}_m2w8_k4", spec.tag()),
            arch,
            symbol: spec.symbol_hc(),
            kernargs,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: BLOCK,
            group_segment_fixed_size: HC_LDS_BYTES,
            wave32: true,
            cu_mode: false,
        };
        let mut b = Builder::new(kspec, plan(spec)?);
        b.enable_delay_alu();
        let slot = b.lds_slot("x_tile", 0, HC_LDS_BYTES)?;
        prologue(&mut b, spec)?;
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
        b.loop_(&lbl(spec, "tile"), |b| {
            group(b, spec, slot)?;
            sop(
                b,
                format!("{} s{GI}, s{GI}, 1", s_add_i32(spec.arch)),
                &[GI],
                &[GI],
            )?;
            sop(
                b,
                format!("s_cmp_lt_u32 s{GI}, s{GROUPS}"),
                &[],
                &[GI, GROUPS],
            )?;
            branch(b, &format!("s_cbranch_scc1 {}", lbl(spec, "tile")))
        })?;
        b.label(&lbl(spec, "fold"))?;
        partials(&mut b, spec, slot)?;
        items(&mut b, spec, slot)?;
        b.wait_all()?;
        b.barrier(&[Transition::Retire(slot)])?;
        b.label(&lbl(spec, "end"))?;
        branch(&mut b, "s_endpgm")?;
        b.finish()
    }

    fn prologue(b: &mut Builder, spec: Spec) -> R {
        let a = b.spec.arch;
        let end = lbl(spec, "end");
        smem(b, KA, 8, KARG, 0)?;
        smem(b, KB_, 4, KARG, 32)?;
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
        sop(b, format!("s_mov_b32 s{SW}, s{WGIN}"), &[SW], &[WGIN])?;
        b.wait_all()?;
        // Uniform early returns, before any barrier and any store.
        sop(b, format!("s_lshl_b32 s{MBASE}, s{SW}, 1"), &[MBASE], &[SW])?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{MBASE}, s{SM}"),
            &[],
            &[MBASE, SM],
        )?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_eq_u32 s{SR}, 0"), &[], &[SR])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_cmp_gt_u32 s{SR}, {}", spec.cap), &[], &[SR])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        sop(b, format!("s_and_b32 s{TMP}, s{SK}, 31"), &[TMP], &[SK])?;
        sop(b, format!("s_cmp_lg_u32 s{TMP}, 0"), &[], &[TMP])?;
        branch(b, &format!("s_cbranch_scc1 {end}"))?;
        // Column m = 2 wg + (wave >> 2), quarter wave & 3.
        sop(
            b,
            format!("s_lshr_b32 s{MLOC}, s{WAVE}, 2"),
            &[MLOC],
            &[WAVE],
        )?;
        sop(b, format!("s_and_b32 s{QI}, s{WAVE}, 3"), &[QI], &[WAVE])?;
        sop(
            b,
            format!("{} s{COL}, s{MBASE}, s{MLOC}", s_add_i32(a)),
            &[COL],
            &[MBASE, MLOC],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{COL}, s{SM}"), &[], &[COL, SM])?;
        sop(b, format!("s_cselect_b32 s{SVAL}, 1, 0"), &[SVAL], &[])?;
        sop(b, format!("s_lshr_b32 s{PART}, s{SK}, 2"), &[PART], &[SK])?;
        sop(
            b,
            format!("{} s{GROUPS}, s{PART}, {}", s_add_i32(a), lit(255)),
            &[GROUPS],
            &[PART],
        )?;
        sop(
            b,
            format!("s_lshr_b32 s{GROUPS}, s{GROUPS}, 8"),
            &[GROUPS],
            &[GROUPS],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{PARTB}, s{PART}, 2"),
            &[PARTB],
            &[PART],
        )?;
        // Weight descriptor: W + (m*K + q*part) * 2 (64-bit); a column past M is never loaded from.
        sop(b, format!("s_lshl_b32 s{WRB}, s{SK}, 1"), &[WRB], &[SK])?;
        sop(
            b,
            format!("s_mul_i32 s{WRL}, s{COL}, s{WRB}"),
            &[WRL],
            &[COL, WRB],
        )?;
        sop(
            b,
            format!("s_mul_hi_u32 s{WRH}, s{COL}, s{WRB}"),
            &[WRH],
            &[COL, WRB],
        )?;
        op(
            b,
            format!("s_mov_b64 s[{SRD}:{}], s[{PW}:{}]", SRD + 1, PW + 1),
            &[sr(SRD, 2)],
            &[sr(PW, 2)],
        )?;
        sop(
            b,
            format!("{} s{SRD}, s{SRD}, s{WRL}", s_add_u32(a)),
            &[SRD],
            &[SRD, WRL],
        )?;
        sop(
            b,
            format!("{} s{}, s{}, s{WRH}", s_addc_u32(a), SRD + 1, SRD + 1),
            &[SRD + 1],
            &[SRD + 1, WRH],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{QOFF}, s{QI}, s{PART}"),
            &[QOFF],
            &[QI, PART],
        )?;
        sop(
            b,
            format!("s_lshl_b32 s{QOFF}, s{QOFF}, 1"),
            &[QOFF],
            &[QOFF],
        )?;
        sop(
            b,
            format!("{} s{SRD}, s{SRD}, s{QOFF}", s_add_u32(a)),
            &[SRD],
            &[SRD, QOFF],
        )?;
        sop(
            b,
            format!("{} s{}, s{}, 0", s_addc_u32(a), SRD + 1, SRD + 1),
            &[SRD + 1],
            &[SRD + 1],
        )?;
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
        sop(b, format!("s_lshl_b32 s{K2B}, s{KB}, 1"), &[K2B], &[KB])?;
        // Producer X base of this wave's vector: (wave>>2) * K * 4 + (wave&3) * part * 4.
        sop(
            b,
            format!("s_mul_i32 s{TMP}, s{MLOC}, s{KB}"),
            &[TMP],
            &[MLOC, KB],
        )?;
        sop(
            b,
            format!("s_mul_i32 s{TMP2}, s{QI}, s{PARTB}"),
            &[TMP2],
            &[QI, PARTB],
        )?;
        sop(
            b,
            format!("{} s{XWB}, s{TMP}, s{TMP2}", s_add_i32(a)),
            &[XWB],
            &[TMP, TMP2],
        )?;
        // Partial slot base: (mloc * cap) * 16 + q * 4.
        sop(
            b,
            format!(
                "s_mul_i32 s{PBASE}, s{MLOC}, {}",
                lit(16 * u32::from(spec.cap))
            ),
            &[PBASE],
            &[MLOC],
        )?;
        sop(b, format!("s_lshl_b32 s{TMP}, s{QI}, 2"), &[TMP], &[QI])?;
        sop(
            b,
            format!("{} s{PBASE}, s{PBASE}, s{TMP}", s_add_i32(a)),
            &[PBASE],
            &[PBASE, TMP],
        )?;
        // Lane vectors: producer store (wave*1024 + 4 lane), consumer read (q*1024 + 4 lane).
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{LDC}, 2, v{LANE}"),
            &[LDC],
            &[LANE],
            &[],
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
        sop(b, format!("s_lshl_b32 s{TMP}, s{QI}, 10"), &[TMP], &[QI])?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{LDR}, s{TMP}, v{LDC}"),
            &[LDR],
            &[LDC],
            &[TMP],
        )?;
        valu(
            b,
            &format!("v_mov_b32_e32 v{VPART}, s{PBASE}"),
            &[VPART],
            &[],
            &[PBASE],
        )?;
        b.wait_all()
    }

    /// One local K256 tile: weights once, then every four-row stage.
    fn group(b: &mut Builder, spec: Spec, slot: usize) -> R {
        sop(b, format!("s_lshl_b32 s{GOFF}, s{GI}, 9"), &[GOFF], &[GI])?;
        sop(b, format!("s_lshl_b32 s{XG}, s{GI}, 10"), &[XG], &[GI])?;
        // Active lanes: 8 l + 8 <= part - 256 GI (part is a multiple of 8).
        sop(b, format!("s_lshl_b32 s{KREM}, s{GI}, 8"), &[KREM], &[GI])?;
        sop(
            b,
            format!("{} s{KREM}, s{PART}, s{KREM}", s_sub_i32()),
            &[KREM],
            &[PART, KREM],
        )?;
        sop(b, format!("s_lshr_b32 s{EM}, s{KREM}, 3"), &[EM], &[KREM])?;
        sop(b, format!("s_lshl_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
        sop(b, format!("s_not_b32 s{EM}, s{EM}"), &[EM], &[EM])?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{KREM}, {}", lit(256)),
            &[],
            &[KREM],
        )?;
        sop(b, format!("s_cselect_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
        sop(
            b,
            format!("{} s{TMP}, s{XG}, s{XWB}", s_add_i32(spec.arch)),
            &[TMP],
            &[XG, XWB],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{XOFF}, 5, v{LANE}"),
            &[XOFF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{XOFF}, s{TMP}, v{XOFF}"),
            &[XOFF],
            &[XOFF],
            &[TMP],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{HOFF}, 4, v{LANE}"),
            &[HOFF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
            &[HOFF],
            &[HOFF],
            &[GOFF],
        )?;
        let wdone = lbl(spec, "wdone");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {wdone}"))?;
        exec_set(b)?;
        let w = |i: u8| WS + i;
        vmem(
            b,
            &format!(
                "buffer_load_b128 v[{}:{}], v{HOFF}, s[{SRD}:{}], 0 offen",
                WS,
                WS + 3,
                SRD + 3
            ),
            &[w(0), w(1), w(2), w(3)],
            &[HOFF],
            &[SRD, SRD + 1, SRD + 2, SRD + 3],
            false,
        )?;
        b.wait(load_counter(), 0)?;
        widen_weights(b)?;
        exec_all(b)?;
        b.label(&wdone)?;
        for st in 0..u32::from(spec.cap) / 4 {
            stage(b, spec, slot, st)?;
        }
        Ok(())
    }

    /// Rows `4 st .. 4 st + 3`: fill (two vectors per producer wave), barrier, consume, barrier.
    fn stage(b: &mut Builder, spec: Spec, slot: usize, st: u32) -> R {
        let a = b.spec.arch;
        let rb = 4 * st;
        let done = lbl(spec, &format!("st{st}_done"));
        if st > 0 {
            sop(b, format!("s_cmp_le_u32 s{SR}, {rb}"), &[], &[SR])?;
            branch(b, &format!("s_cbranch_scc1 {done}"))?;
        }
        for h in 0..2u32 {
            // Vector wave + 8h: row rb + (wave >> 2) + 2h, quarter wave & 3.
            let nofill = lbl(spec, &format!("st{st}_h{h}_nofill"));
            sop(b, format!("s_lshr_b32 s{TMP}, s{WAVE}, 2"), &[TMP], &[WAVE])?;
            sop(
                b,
                format!("{} s{TMP}, s{TMP}, {}", s_add_i32(a), rb + 2 * h),
                &[TMP],
                &[TMP],
            )?;
            sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SR}"), &[], &[TMP, SR])?;
            branch(b, &format!("s_cbranch_scc1 {nofill}"))?;
            sop(b, format!("s_mul_i32 s{TMP2}, s{KB}, {rb}"), &[TMP2], &[KB])?;
            if h == 1 {
                sop(
                    b,
                    format!("{} s{TMP2}, s{TMP2}, s{K2B}", s_add_i32(a)),
                    &[TMP2],
                    &[TMP2, K2B],
                )?;
            }
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{VT}, s{TMP2}, v{XOFF}"),
                &[VT],
                &[XOFF],
                &[TMP2],
            )?;
            exec_set(b)?;
            for (j, suffix) in [(0u8, ""), (4, " offset:16")] {
                vmem(
                    b,
                    &format!(
                        "global_load_b128 v[{}:{}], v{VT}, s[{PX}:{}]{suffix}",
                        XS + j,
                        XS + j + 3,
                        PX + 1
                    ),
                    &[XS + j, XS + j + 1, XS + j + 2, XS + j + 3],
                    &[VT],
                    &[PX, PX + 1],
                    false,
                )?;
            }
            for j in 0..8u32 {
                b.ds_store(
                    slot,
                    Instruction::new(
                        format!(
                            "ds_store_b32 v{LDST}, v{}{}",
                            XS + j as u8,
                            off(8192 * h + 128 * j)
                        ),
                        vec![],
                        vec![v(LDST), v(XS + j as u8)],
                    )
                    .memory(MemoryClass::DsStore),
                )?;
            }
            b.wait_all()?;
            exec_all(b)?;
            b.label(&nofill)?;
        }
        b.barrier(&[Transition::Ready(slot)])?;
        let cdone = lbl(spec, &format!("st{st}_cdone"));
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
        exec_set(b)?;
        for rl in 0..4u32 {
            if rb + rl > 0 {
                sop(b, format!("s_cmp_le_u32 s{SR}, {}", rb + rl), &[], &[SR])?;
                branch(b, &format!("s_cbranch_scc1 {cdone}"))?;
            }
            consume_row(b, slot, Mode::Fast(0), rb + rl, 4096 * rl, LDR)?;
        }
        b.label(&cdone)?;
        exec_all(b)?;
        b.barrier(&[Transition::Retire(slot)])?;
        b.label(&done)
    }

    /// Quarter sums of every real row into LDS (lane 0 of each valid wave), then the barrier.
    fn partials(b: &mut Builder, spec: Spec, slot: usize) -> R {
        let pdone = lbl(spec, "pdone");
        let stored = lbl(spec, "pstored");
        sop(b, format!("s_cmp_eq_u32 s{SVAL}, 0"), &[], &[SVAL])?;
        branch(b, &format!("s_cbranch_scc1 {pdone}"))?;
        reduce_rows(b, u32::from(spec.cap))?;
        valu(
            b,
            &format!("v_cmpx_eq_u32_e32 0, v{LANE}"),
            &[],
            &[LANE],
            &[],
        )?;
        for r in 0..u32::from(spec.cap) {
            if r > 0 {
                sop(b, format!("s_cmp_le_u32 s{SR}, {r}"), &[], &[SR])?;
                branch(b, &format!("s_cbranch_scc1 {stored}"))?;
            }
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{VPART}, v{}{}", acc(r, 3), off(16 * r)),
                    vec![],
                    vec![v(VPART), v(acc(r, 3))],
                )
                .memory(MemoryClass::DsStore),
            )?;
        }
        b.label(&stored)?;
        b.wait_all()?;
        exec_all(b)?;
        b.label(&pdone)?;
        b.barrier(&[Transition::Ready(slot)])
    }

    /// Items `n = wave + 8 k` (`mloc = n / cap`, `r = n % cap`): fold, activate, store.
    fn items(b: &mut Builder, spec: Spec, slot: usize) -> R {
        let a = b.spec.arch;
        let cap = u32::from(spec.cap);
        let mut s_tmp = [0u8; 11];
        for (i, t) in s_tmp.iter_mut().enumerate() {
            *t = HTMP + i as u8;
        }
        let mut v_tmp = [0u8; 16];
        for (i, t) in v_tmp.iter_mut().enumerate() {
            *t = XS + i as u8;
        }
        let regs = Regs {
            q: [HQ, HQ + 1, HQ + 2, HQ + 3],
            scale: SCALE,
            act: ACT,
            out: HOUT,
            s_tmp,
            v_tmp,
        };
        for k in 0..2 * cap / 8 {
            let mloc = u32::from(8 * k >= cap);
            let rbase = 8 * k - cap * mloc;
            let skip = lbl(spec, &format!("iskip{k}"));
            sop(
                b,
                format!("{} s{IR}, s{WAVE}, {rbase}", s_add_i32(a)),
                &[IR],
                &[WAVE],
            )?;
            sop(b, format!("s_cmp_ge_u32 s{IR}, s{SR}"), &[], &[IR, SR])?;
            branch(b, &format!("s_cbranch_scc1 {skip}"))?;
            if mloc == 1 {
                sop(
                    b,
                    format!("{} s{TMP}, s{MBASE}, 1", s_add_i32(a)),
                    &[TMP],
                    &[MBASE],
                )?;
                sop(b, format!("s_cmp_ge_u32 s{TMP}, s{SM}"), &[], &[TMP, SM])?;
                branch(b, &format!("s_cbranch_scc1 {skip}"))?;
            }
            sop(b, format!("s_lshl_b32 s{TMP}, s{IR}, 4"), &[TMP], &[IR])?;
            if mloc == 1 {
                sop(
                    b,
                    format!("{} s{TMP}, s{TMP}, {}", s_add_i32(a), lit(16 * cap)),
                    &[TMP],
                    &[TMP],
                )?;
            }
            valu(
                b,
                &format!("v_mov_b32_e32 v{VITA}, s{TMP}"),
                &[VITA],
                &[],
                &[TMP],
            )?;
            b.ds_load(
                slot,
                Instruction::new(
                    format!("ds_load_b128 v[{}:{}], v{VITA}", VITQ, VITQ + 3),
                    (0..4).map(|i| v(VITQ + i)).collect(),
                    vec![v(VITA)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            for i in 0..4u8 {
                op(
                    b,
                    format!("v_readfirstlane_b32 s{}, v{}", HQ + i, VITQ + i),
                    &[s(HQ + i)],
                    &[v(VITQ + i)],
                )?;
            }
            epi::emit_value(b, &lbl(spec, &format!("item{k}")), &regs)?;
            // Y[r*M + m] with m = m_base + mloc.
            sop(
                b,
                format!("s_mul_i32 s{YB}, s{IR}, s{MB}"),
                &[YB],
                &[IR, MB],
            )?;
            sop(
                b,
                format!("{} s{TMP}, s{MBASE}, {mloc}", s_add_i32(a)),
                &[TMP],
                &[MBASE],
            )?;
            sop(b, format!("s_lshl_b32 s{TMP}, s{TMP}, 2"), &[TMP], &[TMP])?;
            sop(
                b,
                format!("{} s{YB}, s{YB}, s{TMP}", s_add_i32(a)),
                &[YB],
                &[YB, TMP],
            )?;
            valu(
                b,
                &format!("v_mov_b32_e32 v{VYA}, s{YB}"),
                &[VYA],
                &[],
                &[YB],
            )?;
            valu(
                b,
                &format!("v_mov_b32_e32 v{VYD}, s{HOUT}"),
                &[VYD],
                &[],
                &[HOUT],
            )?;
            exec_lane0(b)?;
            vmem(
                b,
                &format!("global_store_b32 v{VYA}, v{VYD}, s[{PY}:{}]", PY + 1),
                &[],
                &[VYA, VYD],
                &[PY, PY + 1],
                true,
            )?;
            b.wait_all()?;
            exec_all(b)?;
            b.label(&skip)?;
        }
        Ok(())
    }
}
