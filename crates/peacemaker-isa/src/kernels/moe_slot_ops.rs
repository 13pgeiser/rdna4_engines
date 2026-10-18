// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Builder-emitted (PeaceMaker) gfx1201 BF16-input top-k combine of the symmetric
//! IU4 grouped MoE (`HIPFIRE_MOE_SYM_IU4`, prefill Path2): the twin of
//! `moe_down_combine_grouped_k8` for the BF16 grouped down rows the symmetric
//! down GEMM writes, `moe_down_combine_grouped_bf16in_pm_gfx1201` in the module
//! `moe_slot_ops_pm_gfx1201`. gfx1201 only (gfx1100 / gfx1151 are refused).
//!
//! ABI (eight arguments, 48 bytes): `y_grouped_bf16` `*u16` @0 (BF16 bits,
//! `[grouped_rows, dim]`), `inverse_perm` `*i32` @8 (`[n_tokens * k_top]`, flat
//! slot -> grouped row, negative = unrouted), `topk_weights` `*f32` @16
//! (`[n_tokens * k_top]`), `x_residual` `*f32` @24 (`[n_tokens, dim]`, updated in
//! place), `dim` u32 @32, `k_top` u32 @36, `grouped_rows` u32 @40, `n_tokens` u32
//! @44. Grid `[n_tokens, ceil(dim / 256), 1]`, block `[256, 1, 1]`, wave32, no
//! LDS, no barrier, no scratch, no private memory, no atomics. Contract (the
//! caller's checks): `k_top` in `1..=8`, every byte offset below `2^32`,
//! `inverse_perm` and `topk_weights` hold `n_tokens * k_top` words.
//!
//! Law (per `t` = workgroup id x, `m` = 256 * workgroup id y + tid):
//! ```text
//!   return if t >= n_tokens or m >= dim                       (no write)
//!   acc = +0.0f32
//!   for k in 0..k_top ascending:
//!       slot = inverse_perm[t * k_top + k]                    (unsigned)
//!       skip if slot >= grouped_rows                          (negative = unrouted)
//!       w = topk_weights[t * k_top + k]
//!       v = f32::from_bits(y[slot * dim + m] << 16)           (BF16 widening, exact)
//!       acc = fma(w, v, acc)                                  (one fused rounding)
//!   x_residual[t * dim + m] = RN(x_residual[t * dim + m] + acc)
//! ```
//! Outside `k_top` in `1..=8` the whole workgroup returns before any VMEM access
//! (a defensive domain guard; the host never launches there), so a bad `k_top`
//! can neither read past the slot tables nor write.
//!
//! Schedule: a runtime ascending-`k` loop of `k_top` trips (one static
//! `v_fma_f32`). The slot is workgroup-uniform (it depends on `t` and `k` only), so
//! each trip loads it with SMEM and tests `slot >=u grouped_rows` on the scalar
//! unit; a skipped slot branches over the whole rest of the trip: no weight load,
//! no `y` load, no fma. A live slot loads its weight (SMEM), forms the row byte
//! offset `slot * dim * 2` (the raw-buffer `soffset`), loads the lane's BF16
//! (`buffer_load_u16`, exec-masked to `m < dim`), widens it with a 16-bit left
//! shift and runs `v_fma_f32 acc, w, v, acc` in place. After the loop one
//! `v_add_f32` adds the accumulator to the residual (loaded there) and the sum is
//! stored. There is no mask select, no clamped or dummy load.
use super::common::{add64, add64_imm, mem, op, s, s_add_i32, smem, sop, sr, srd_tail, v};
use crate::{
    insn::{Instruction, MemoryClass},
    plan::Access,
    reg::{Kind, Live, RegRef},
    Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan,
};
use peacemaker_author::{Gfx1201, Scc, Target, WgUniform, Workgroup};

type R = Result<(), String>;

/// Threads (output columns) per workgroup.
pub const BLOCK: u32 = 256;
/// Kernarg bytes.
pub const KERNARG_BYTES: u32 = 48;
/// Largest `k_top` the kernel accepts (a workgroup with a larger `k_top` returns).
pub const MAX_K_TOP: u32 = 8;

/// The kernel-spec id (`peacemaker-isa emit --kernel`); the symbol is [`symbol`].
pub const KERNEL_ID: &str = "moe_slot_ops_pm";

pub fn symbol(arch: Arch) -> String {
    format!("moe_down_combine_grouped_bf16in_pm_{}", arch.name())
}
pub fn module(arch: Arch) -> String {
    format!("moe_slot_ops_pm_{}", arch.name())
}

// ---- SGPRs ------------------------------------------------------------------
const KARG: u8 = 0;
// kernarg words 0..8 -> s8..s15 (y 8:9, inverse_perm 10:11, topk_weights 12:13, x_residual 14:15);
// words 8..12 -> s16..s19 (dim, k_top, grouped_rows, n_tokens).
const Y: u8 = 8;
const INV: u8 = 10;
const WGT: u8 = 12;
const XR: u8 = 14;
const DIM: u8 = 16;
const KTOP: u8 = 17;
const GROWS: u8 = 18;
const NTOK: u8 = 19;
const WGX: u8 = 20;
const WGY: u8 = 21;
/// Trips left.
const KLEFT: u8 = 22;
/// `t * k_top * 4`: the token's first slot byte offset.
const TK4: u8 = 23;
/// `dim * 2`: BF16 row bytes.
const DIM2: u8 = 24;
/// `t * dim * 4`: the token's residual row byte offset.
const XSOFF: u8 = 25;
const BYTES: u8 = 26;
const T0: u8 = 27;
/// Running `inverse_perm + 4 (t k_top + k)` (28:29) and `topk_weights + 4 (t k_top + k)` (30:31).
const INVP: u8 = 28;
const WP: u8 = 30;
// Raw buffer descriptors (4-aligned quads).
const SRD_Y: u8 = 32;
const SRD_X: u8 = 36;
/// The trip's slot, weight and BF16 row byte offset.
const SLOT: u8 = 40;
const WK: u8 = 41;
const YOFF: u8 = 42;
/// The `m < dim` lane mask pair (the high word is never written in wave32 and held at zero).
const LV: u8 = 44;
const S_END: u8 = 48;

// ---- VGPRs ------------------------------------------------------------------
const TID: u8 = 0;
const VM: u8 = 1;
const VOFF2: u8 = 2;
const VOFF4: u8 = 3;
const ACC: u8 = 4;
const XV: u8 = 5;
const YV: u8 = 6;
const V_END: u8 = 8;

fn kernargs() -> KernargLayout {
    KernargLayout::new(KERNARG_BYTES)
        .pointer_access("y_grouped_bf16", 0, Access::ReadOnly)
        .pointer_access("inverse_perm", 8, Access::ReadOnly)
        .pointer_access("topk_weights", 16, Access::ReadOnly)
        .pointer_access("x_residual", 24, Access::ReadWrite)
        .hidden("dim", 32, 4, "by_value")
        .hidden("k_top", 36, 4, "by_value")
        .hidden("grouped_rows", 40, 4, "by_value")
        .hidden("n_tokens", 44, 4, "by_value")
}

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(V_END), 104)?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    for base in (8u8..S_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..V_END).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

/// The builder-emitted kernel (gfx1201 only; every other target is refused before any builder exists).
pub fn emit(arch: Arch) -> Result<Emitted, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!(
            "{KERNEL_ID}: built for gfx1201 only, not {}",
            arch.name()
        ));
    }
    let kspec = KernelSpec {
        kernel_id: KERNEL_ID.into(),
        variant: "combine".into(),
        arch,
        symbol: symbol(arch),
        kernargs: kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan()?);
    b.enable_delay_alu();
    kernel(&mut Workgroup::<Gfx1201, Builder>::new(&mut b)?)?;
    b.finish()
}

/// The single-symbol module `moe_slot_ops_pm_<arch>`.
pub fn emit_module(
    arch: Arch,
) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = vec![emit(arch)?];
    let (text, proof) = super::iu4_gemm::module(&emitted, &module(arch))?;
    Ok((emitted, text, proof))
}

// ---- helpers ----------------------------------------------------------------
fn m(x: u8) -> RegRef {
    sr(x, 2)
}
fn cmp_wg<T: Target>(
    wg: &mut Workgroup<'_, T, Builder>,
    text: String,
    uses: &[RegRef],
) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(text, vec![], uses.to_vec()))
}
/// `s{dst}:{dst+3}` raw buffer descriptor of the pointer pair at `s{ptr}` with `num_records` from `s{records}`.
fn srd(b: &mut Builder, dst: u8, ptr: u8, records: u8) -> R {
    sop(b, format!("s_mov_b32 s{dst}, s{ptr}"), &[dst], &[ptr])?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", dst + 1, ptr + 1),
        &[dst + 1],
        &[ptr + 1],
    )?;
    srd_tail(b, dst, Some(records))
}
/// Raw buffer load (`buffer_load_u16` / `buffer_load_b32`) at `v{voff} + s{soff}`.
fn bload_s(b: &mut Builder, name: &str, dst: u8, voff: u8, desc: u8, soff: u8) -> R {
    mem(
        b,
        format!(
            "{name} v{dst}, v{voff}, s[{desc}:{}], s{soff} offen",
            desc + 3
        ),
        &[v(dst)],
        &[v(voff), sr(desc, 4), s(soff)],
        MemoryClass::VmemLoad,
    )
}
/// Raw buffer store of one dword at `v{voff} + s{soff}`.
fn bstore_s(b: &mut Builder, data: u8, voff: u8, desc: u8, soff: u8) -> R {
    mem(
        b,
        format!(
            "buffer_store_b32 v{data}, v{voff}, s[{desc}:{}], s{soff} offen",
            desc + 3
        ),
        &[],
        &[v(voff), v(data), sr(desc, 4), s(soff)],
        MemoryClass::VmemStore,
    )
}

// ---- kernel -----------------------------------------------------------------
fn kernel<T: Target>(wg: &mut Workgroup<'_, T, Builder>) -> R {
    let end = wg.exit(".Lcombine_end")?;
    let b = wg.isa();
    smem(b, Y, 8, KARG, 0x00)?;
    smem(b, DIM, 4, KARG, 0x20)?;
    // Workgroup ids: gfx12 `ttmp9` (x) and `ttmp7[15:0]` (y).
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    // t >= n_tokens: the workgroup returns before any access.
    let outside = cmp_wg(
        wg,
        format!("s_cmp_ge_u32 s{WGX}, s{NTOK}"),
        &[s(WGX), s(NTOK)],
    )?;
    wg.exit_if(outside, &end)?;
    // k_top outside 1..=8 (`k_top - 1 >u 7`, which also rejects 0).
    let b = wg.isa();
    sop(
        b,
        format!("{} s{T0}, s{KTOP}, -1", s_add_i32(Arch::Gfx1201)),
        &[T0],
        &[KTOP],
    )?;
    let bad_k = cmp_wg(
        wg,
        format!("s_cmp_gt_u32 s{T0}, {}", MAX_K_TOP - 1),
        &[s(T0)],
    )?;
    wg.exit_if(bad_k, &end)?;
    // 256 wg_y >= dim: no lane of this workgroup has a column.
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 8"), &[T0], &[WGY])?;
    let no_cols = cmp_wg(wg, format!("s_cmp_ge_u32 s{T0}, s{DIM}"), &[s(T0), s(DIM)])?;
    wg.exit_if(no_cols, &end)?;

    let b = wg.isa();
    // Uniform byte offsets, the running slot / weight pointers and the two raw descriptors.
    sop(b, format!("s_mov_b32 s{KLEFT}, s{KTOP}"), &[KLEFT], &[KTOP])?;
    sop(
        b,
        format!("s_mul_i32 s{TK4}, s{WGX}, s{KTOP}"),
        &[TK4],
        &[WGX, KTOP],
    )?;
    sop(b, format!("s_lshl_b32 s{TK4}, s{TK4}, 2"), &[TK4], &[TK4])?;
    add64(b, INVP, INV, TK4)?;
    add64(b, WP, WGT, TK4)?;
    sop(b, format!("s_lshl_b32 s{DIM2}, s{DIM}, 1"), &[DIM2], &[DIM])?;
    sop(
        b,
        format!("s_mul_i32 s{XSOFF}, s{WGX}, s{DIM}"),
        &[XSOFF],
        &[WGX, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{XSOFF}, s{XSOFF}, 2"),
        &[XSOFF],
        &[XSOFF],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{GROWS}, s{DIM2}"),
        &[BYTES],
        &[GROWS, DIM2],
    )?;
    srd(b, SRD_Y, Y, BYTES)?;
    sop(
        b,
        format!("s_mul_i32 s{BYTES}, s{NTOK}, s{DIM}"),
        &[BYTES],
        &[NTOK, DIM],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{BYTES}, s{BYTES}, 2"),
        &[BYTES],
        &[BYTES],
    )?;
    srd(b, SRD_X, XR, BYTES)?;
    // The lane-mask pair's high word (wave32 never writes it; VOP3 reads the pair).
    sop(b, format!("s_mov_b32 s{}, 0", LV + 1), &[LV + 1], &[])?;
    // Lane column m = 256 wg_y + tid and its byte offsets; lanes with m >= dim leave exec.
    op(
        b,
        format!("v_lshl_add_u32 v{VM}, s{WGY}, 8, v{TID}"),
        &[v(VM)],
        &[s(WGY), v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF2}, 1, v{VM}"),
        &[v(VOFF2)],
        &[v(VM)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VOFF4}, 2, v{VM}"),
        &[v(VOFF4)],
        &[v(VM)],
    )?;
    op(
        b,
        format!("v_cmp_gt_u32_e64 s{LV}, s{DIM}, v{VM}"),
        &[m(LV)],
        &[s(DIM), v(VM)],
    )?;
    op(
        b,
        format!("s_and_b32 exec_lo, exec_lo, s{LV}"),
        &[],
        &[m(LV)],
    )?;
    // acc = +0.
    op(b, format!("v_mov_b32_e32 v{ACC}, 0"), &[v(ACC)], &[])?;
    b.wait_all()?;

    // k = 0..k_top ascending (k_top >= 1: a do-while). Per trip the slot word is always in range of the token's k_top words.
    wg.loop_carried(".Lcombine_trip", (), |wg, ()| {
        let b = wg.isa();
        smem(b, SLOT, 1, INVP, 0)?;
        // Unrouted (negative) or out-of-range slot: skip the weight, the row and the fma.
        let dead = cmp_wg(
            wg,
            format!("s_cmp_ge_u32 s{SLOT}, s{GROWS}"),
            &[s(SLOT), s(GROWS)],
        )?;
        wg.wg_skip_if(dead, ".Lcombine_skip", (), |wg, ()| {
            let b = wg.isa();
            smem(b, WK, 1, WP, 0)?;
            sop(
                b,
                format!("s_mul_i32 s{YOFF}, s{SLOT}, s{DIM2}"),
                &[YOFF],
                &[SLOT, DIM2],
            )?;
            bload_s(b, "buffer_load_u16", YV, VOFF2, SRD_Y, YOFF)?;
            op(
                b,
                format!("v_lshlrev_b32_e32 v{YV}, 16, v{YV}"),
                &[v(YV)],
                &[v(YV)],
            )?;
            op(
                b,
                format!("v_fma_f32 v{ACC}, s{WK}, v{YV}, v{ACC}"),
                &[v(ACC)],
                &[s(WK), v(YV), v(ACC)],
            )?;
            b.wait_all()
        })?;
        let b = wg.isa();
        add64_imm(b, INVP, 4)?;
        add64_imm(b, WP, 4)?;
        sop(
            b,
            format!("{} s{KLEFT}, s{KLEFT}, -1", s_add_i32(Arch::Gfx1201)),
            &[KLEFT],
            &[KLEFT],
        )?;
        let more = cmp_wg(wg, format!("s_cmp_gt_i32 s{KLEFT}, 0"), &[s(KLEFT)])?;
        Ok(((), more))
    })?;

    // x_residual = RN(x_residual + acc).
    let b = wg.isa();
    bload_s(b, "buffer_load_b32", XV, VOFF4, SRD_X, XSOFF)?;
    op(
        b,
        format!("v_add_f32_e32 v{XV}, v{XV}, v{ACC}"),
        &[v(XV)],
        &[v(XV), v(ACC)],
    )?;
    bstore_s(b, XV, VOFF4, SRD_X, XSOFF)?;
    b.wait_all()?;
    wg.end(end)
}

/// Shape contract of the emitted kernel (the `--shape` input of
/// `peacemaker-isa emit --co`): exact counts of the memory, arithmetic and control
/// mnemonics the kernel is allowed, the forbidden families and the resource limits.
pub fn contract(arch: Arch, e: &Emitted) -> serde_json::Value {
    let mut counts = serde_json::Map::new();
    for name in [
        "s_load_b32",
        "s_load_b128",
        "s_load_b256",
        "buffer_load_u16",
        "buffer_load_b32",
        "buffer_store_b32",
        "v_fma_f32",
        "v_add_f32_e32",
        "s_barrier_signal",
        "s_barrier_wait",
        "s_endpgm",
    ] {
        let n = e
            .s_text
            .lines()
            .filter(|l| l.trim().split_whitespace().next() == Some(name))
            .count();
        counts.insert(name.into(), n.into());
    }
    serde_json::json!({
        "symbol": symbol(arch), "counts": counts,
        "forbidden": ["scratch_*", "flat_*", "global_*", "buffer_atomic_*", "ds_*", "v_writelane*", "s_setpc*", "s_swappc*",
            "v_mul_f32*", "v_fmac_f32*", "v_mac_f32*", "v_mad_f32*", "v_dual_*", "v_pk_*", "v_cndmask_*"],
        "vgpr_max": V_END, "sgpr_max": 104,
        "require_wave32": true, "require_zero_spills": true, "require_zero_private": true,
        "launch_dynamic_lds_bytes": 0,
        "group_segment_fixed_bytes": 0,
    })
}
