// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 one-epoch K loop (B4-B baseline).
//!
//! Per iteration: scalar epoch offsets from the epoch index, bounded loads of
//! exactly this epoch, compute/fold ([`super::fold::compute`]), then a full
//! drain of the wait ledger before the back edge. No prefetch, no double
//! buffering, no load past the last epoch. The descriptors never move: every
//! load is `voffset + immediate` against a descriptor whose `num_records` is
//! the exact tensor extent, so each address is bounds-checked by the hardware
//! as well as proven by the admission check.
//!
//! The index `e` runs `0..E` ascending (`E = K/128`). `Negative::ReversedEpoch`
//! runs `E-1..0` with the matching operands and scales, which changes only the
//! F32 fold order.
use super::*;

fn cmp(wg: &mut Wg<'_>, text: String, uses: &[u8]) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        text,
        vec![],
        uses.iter().map(|&n| s(n)).collect(),
    ))
}

/// Scalar byte offsets of epoch `EIDX` (code, scale, activation) and the
/// three per-epoch VGPR offsets they feed.
pub(super) fn epoch_offsets(b: &mut Builder, g: &Gen) -> Result<(), String> {
    match (g.spec.codec, g.spec.kind) {
        (Codec::W3, _) => {
            sop(
                b,
                format!("s_mul_i32 s{COFF}, s{EIDX}, 48"),
                &[COFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{SOFF}, s{EIDX}, 1"),
                &[SOFF],
                &[EIDX],
            )?;
        }
        (Codec::W2 | Codec::E8p, _) => {
            sop(
                b,
                format!("s_lshl_b32 s{COFF}, s{EIDX}, 5"),
                &[COFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_lshl_b32 s{SOFF}, s{EIDX}, 1"),
                &[SOFF],
                &[EIDX],
            )?;
        }
        (Codec::Mq4, Kind::Down) => {
            // QT53 group `[f16 sc, f16 zp][64 nibble bytes]`: codes e*68+4, scale e*68.
            sop(
                b,
                format!("s_mul_i32 s{SOFF}, s{EIDX}, 0x44"),
                &[SOFF],
                &[EIDX],
            )?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{SOFF}, 4"),
                &[COFF],
                &[SOFF],
            )?;
        }
        (Codec::Mq4, Kind::GateUp) => {
            // QT44 group `[sc0, zp0, sc1, zp1][128 nibble bytes]` per two epochs:
            // codes (e>>1)*136 + 8 + (e&1)*64, scale (e>>1)*136 + (e&1)*4.
            sop(b, format!("s_lshr_b32 s{T0}, s{EIDX}, 1"), &[T0], &[EIDX])?;
            sop(b, format!("s_mul_i32 s{T0}, s{T0}, 0x88"), &[T0], &[T0])?;
            sop(b, format!("s_and_b32 s{T1}, s{EIDX}, 1"), &[T1], &[EIDX])?;
            sop(b, format!("s_lshl_b32 s{COFF}, s{T1}, 6"), &[COFF], &[T1])?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{COFF}, s{T0}"),
                &[COFF],
                &[COFF, T0],
            )?;
            sop(
                b,
                format!("s_add_i32 s{COFF}, s{COFF}, 8"),
                &[COFF],
                &[COFF],
            )?;
            sop(b, format!("s_lshl_b32 s{SOFF}, s{T1}, 2"), &[SOFF], &[T1])?;
            sop(
                b,
                format!("s_add_i32 s{SOFF}, s{SOFF}, s{T0}"),
                &[SOFF],
                &[SOFF, T0],
            )?;
        }
    }
    sop(
        b,
        format!("s_mul_i32 s{XEO}, s{EIDX}, s{XSTR}"),
        &[XEO],
        &[EIDX, XSTR],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{WE}, s{COFF}, v{WOFF}"),
        &[v(WE)],
        &[s(COFF), v(WOFF)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{HE}, s{SOFF}, v{HOFF}"),
        &[v(HE)],
        &[s(SOFF), v(HOFF)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{XE}, s{XEO}, v{XOFF}"),
        &[v(XE)],
        &[s(XEO), v(XOFF)],
    )?;
    // A padding slot stays out of range at every epoch (no wrap of OOB + epoch).
    op(
        b,
        format!(
            "v_cndmask_b32_e64 v{XE}, {}, v{XE}, s{LIVE}",
            lit(GATHER_OOB)
        ),
        &[v(XE)],
        &[v(XE), s(LIVE)],
    )
}

/// The 16-bit scale load: `buffer_load_u16` zero-extends into the low half.
pub(super) fn scale_load(b: &mut Builder, dst: u8, srd: u8) -> Result<(), String> {
    mem(
        b,
        format!(
            "buffer_load_u16 v{dst}, v{HE}, s[{srd}:{}], 0 offen",
            srd + 3
        ),
        &[v(dst)],
        &[v(HE), sr(srd, 4)],
        MemoryClass::VmemLoad,
    )
}

pub(super) fn weight_loads(b: &mut Builder, g: &Gen, t: u8) -> Result<(), String> {
    let mut dst = g.a(t);
    for (off, width) in code_loads(g.spec.codec) {
        bload(b, width, dst, WE, g.code_srd(t), off)?;
        dst += width;
    }
    Ok(())
}

/// Issue exactly epoch `EIDX`'s loads: scales and the first tensor's codes,
/// the second tensor's codes (GateUp), then the activation block.
fn load_epoch(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.clause(|b| {
        for t in 0..g.tn() {
            scale_load(b, g.hs(t), g.scale_srd(t))?;
        }
        weight_loads(b, g, 0)
    })?;
    if g.tn() == 2 {
        b.clause(|b| weight_loads(b, g, 1))?;
    }
    b.clause(|b| {
        for q in 0..4u8 {
            bload(b, 4, g.x() + 4 * q, XE, SRD_X, 8 + 16 * u32::from(q))?;
        }
        bload(b, 1, g.d(), XE, SRD_X, 0)
    })
}

/// The epoch loop. Places `k_begin`; leaves the builder at the fall-through
/// into the caller's `epilogue` label with an empty wait ledger.
pub(super) fn emit(wg: &mut Wg<'_>, g: &Gen) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    wg.isa().wait_all()?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger().shape();
    let none = cmp(wg, format!("s_cmp_eq_u32 s{ECNT}, 0"), &[ECNT])?;
    wg.wg_skip_if(none, &done, (), |wg, ()| {
        wg.loop_carried(&head, (), |wg, ()| {
            let b = wg.isa();
            epoch_offsets(b, g)?;
            load_epoch(b, g)?;
            fold::compute(b, g)?;
            // Drain before the back edge: the next iteration (and the
            // epilogue aliases) start from an empty ledger.
            b.wait_all()?;
            if g.spec.negative == Negative::ReversedEpoch {
                sop(
                    b,
                    format!("s_add_i32 s{EIDX}, s{EIDX}, -1"),
                    &[EIDX],
                    &[EIDX],
                )?;
                Ok(((), cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, -1"), &[EIDX])?))
            } else {
                sop(
                    b,
                    format!("s_add_i32 s{EIDX}, s{EIDX}, 1"),
                    &[EIDX],
                    &[EIDX],
                )?;
                Ok((
                    (),
                    cmp(wg, format!("s_cmp_lg_u32 s{EIDX}, s{ECNT}"), &[EIDX, ECNT])?,
                ))
            }
        })?;
        if wg.isa().ledger().shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(())
    })?;
    wg.isa().wait_all()
}
