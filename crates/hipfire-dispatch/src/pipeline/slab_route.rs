// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab route (`MoeRouteCapability::SlabCodecGrouped`) stage
//! helpers shared by the decode and prefill lowerings.
//!
//! The layer's experts live in one slab arena and carry a codec each: MQ4
//! symmetric plus at most one cold codec per layer (MQ3-I4 W3 on gfx1201 and gfx1151,
//! or E8P-I4 on gfx1201 and gfx1100; W2 nowhere). Two kernel families serve them:
//!
//! * `route_rows <= Gpu::few_row_limit()` (decode, MTP verify, a one-row forward): the K1 lane-union
//!   GEMV over codec buckets ([`rdna_compute::slab_moe`]). The layer sits in the
//!   path-1 stage shape (gate/up F32 rows, SwiGLU activation + 128-wide
//!   rotation, unweighted F32 down rows, top-10 combine); only the two GEMVs
//!   are K1 launches.
//! * longer prefill: the K2 grouped IU4 kernels, MQ4 and the cold codec, over
//!   the same grouped buffers with codec-masked tile ids (disjoint tiles).
//!
//! A layer whose experts all share one codec carries no codec table; a mixed
//! layer's table is the layer's dtype-tag tensor: `[512]` u8 codec bytes
//! (cold-codec membership only: `0` = MQ4, `1` = cold) followed by the same
//! codes as `[512]` i32 (the table-driven tile mask's owner table). The
//! tensor's logical shape is the u8 table; its allocation is
//! [`CODEC_TAG_ALLOC_BYTES`]. Buckets: cold = 0, MQ4 = 1.

use super::layer_ops::hip;
use crate::families::moe::MoeDtypes;
use crate::types::DispatchError;
use rdna_compute::slab_moe::{
    k1_rows8_cold_codec, SlabBuckets, SLAB_EXPERTS, SLAB_NARROW_MAX_LANES, SLAB_TOP_K,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Lanes of one unchanged narrow K1 launch; wider admitted calls either select
/// an opted-in C16/C32 twin or retain exact groups of this size.
pub const K1_MAX_ROWS: usize = SLAB_NARROW_MAX_LANES;
/// Bytes of the u8 codec table.
pub const CODEC_U8_BYTES: usize = SLAB_EXPERTS;
/// Bytes of the whole codec-tag allocation: u8 table, then i32 table.
pub const CODEC_TAG_ALLOC_BYTES: usize = CODEC_U8_BYTES + 4 * SLAB_EXPERTS;

fn refuse(what: &str) -> DispatchError {
    DispatchError::Hip(format!("sealed expert-slab route: {what}"))
}

/// Whether `arch` hosts the sealed expert-slab route (the arch admits MQ4).
pub(crate) fn arch_supported(arch: &str) -> bool {
    rdna_compute::qwen4_slab_codec_admitted(arch, 0)
}

/// A layer's K1 / K2 codec layout: the bucket map and the raw codec id of its
/// cold bucket (the `cold_codec` the runtime entries take).
#[derive(Clone, Copy)]
pub(crate) struct SlabLayer<'a> {
    pub buckets: SlabBuckets<'a>,
    pub cold_codec: u8,
}

/// The codec layout of a layer on `arch`: its codec table when mixed, else the
/// one codec its (uniform) routed dtype names, which `arch` must admit. Two
/// buckets: cold = 0, MQ4 = 1. The cold codec is the policy's declaration,
/// validated against the dtype tables by route selection; nothing here scans
/// a table.
pub(crate) fn layer<'a>(
    arch: &str,
    policy: Option<&crate::families::moe::MoeRoutePolicy>,
    dtypes: &MoeDtypes<'_>,
    tags: Option<&'a GpuTensor>,
) -> Result<SlabLayer<'a>, DispatchError> {
    let policy = policy
        .filter(|policy| policy.capability.is_slab())
        .ok_or_else(|| refuse("the layer declares no slab route"))?;
    let cold_codec = crate::families::moe::slab_launch_cold_codec(arch, policy.slab_cold_codec)
        .ok_or_else(|| {
            refuse(&format!(
                "{arch} does not admit the layer's cold codec {:?}",
                policy.slab_cold_codec
            ))
        })?;
    let buckets = match tags {
        Some(table) => {
            if table.buf.size() < CODEC_TAG_ALLOC_BYTES {
                return Err(refuse(
                    "codec tag allocation is smaller than the u8 + i32 tables",
                ));
            }
            SlabBuckets::Table(table)
        }
        None => crate::families::moe::slab_bucket_of_pair(
            arch,
            dtypes.routed_gate_up,
            dtypes.routed_down,
        )
        .map(SlabBuckets::Uniform)
        .ok_or_else(|| {
            refuse(&format!(
                "uniform layer dtypes {:?}/{:?} are not a slab codec of {arch}",
                dtypes.routed_gate_up, dtypes.routed_down
            ))
        })?,
    };
    Ok(SlabLayer {
        buckets,
        cold_codec,
    })
}

/// The `[512]` i32 codec table of a mixed layer's tag tensor (the table-driven
/// tile mask's owner table).
pub(crate) fn codec_i32(tags: &GpuTensor) -> Result<GpuTensor, DispatchError> {
    if tags.buf.size() < CODEC_TAG_ALLOC_BYTES {
        return Err(refuse(
            "codec tag allocation is smaller than the u8 + i32 tables",
        ));
    }
    Ok(GpuTensor {
        buf: tags.buf.byte_view(CODEC_U8_BYTES, 4 * SLAB_EXPERTS),
        shape: vec![SLAB_EXPERTS],
        dtype: DType::F32,
    })
}

/// One K1 launch group of a wide call: `n` lanes (`<= K1_MAX_ROWS`) starting at
/// lane `first`.
fn k1_groups(lanes: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..lanes)
        .step_by(K1_MAX_ROWS)
        .map(move |first| (first, K1_MAX_ROWS.min(lanes - first)))
}

fn k1_check_lanes(gpu: &Gpu, what: &str, lanes: usize) -> Result<(), DispatchError> {
    let limit = gpu.few_row_limit();
    if lanes == 0 || lanes > limit {
        return Err(refuse(&format!("{what}: lanes must be 1..={limit}")));
    }
    Ok(())
}

/// The `[n * 10]` id view of lanes `first..first + n` of `topk` (slot = lane * 10
/// + rank, i32 words).
fn topk_group(topk: &GpuTensor, first: usize, n: usize) -> GpuTensor {
    GpuTensor {
        buf: topk
            .buf
            .byte_view(first * SLAB_TOP_K * 4, n * SLAB_TOP_K * 4),
        shape: vec![n * SLAB_TOP_K],
        dtype: topk.dtype,
    }
}

/// K1 gate/up of `lanes` lanes (up to the shared per-architecture exact cap):
/// union of `topk` and launches every non-empty bucket. `xr` `[lanes, 2560]`
/// F32 rotated activations; `gate_out`/`up_out` `[lanes * 10, 640]` F32.
///
/// The K1 opt-in selects one C16/C32 union and launch for wider admitted calls.
/// Otherwise consecutive <=8-lane groups retain S1's exact fallback, with a
/// separate union plan immediately before each launch. The scratch then holds
/// only the last group's plan on return (see [`k1_down`]).
///
/// One lane of an MQ4 or W3 layer on gfx1151 (unless `HIPFIRE_QWEN4_SLAB_K1_C1=0`) takes K1-C1
/// instead: one launch for both codecs straight from `topk`, no union plan (bitwise the K1 result).
/// E8P layers keep the K1 bucket entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k1_gate_up(
    gpu: &mut Gpu,
    layer: SlabLayer<'_>,
    gate_up_ptrs: &GpuTensor,
    topk: &GpuTensor,
    xr: &GpuTensor,
    gate_out: &GpuTensor,
    up_out: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    k1_check_lanes(gpu, "k1_gate_up", lanes)?;
    if lanes == 1 && gpu.slab_k1_c1() && k1_rows8_cold_codec(layer.cold_codec) {
        return hip(gpu.mq_i4_moe_gemv_slab_c1(
            false,
            layer.buckets,
            gate_up_ptrs,
            topk,
            xr,
            gate_out,
            Some(up_out),
            layer.cold_codec,
        ));
    }
    if lanes <= K1_MAX_ROWS || gpu.qwen4_slab_wide_twin(lanes, layer.cold_codec) {
        hip(gpu.qwen4_slab_union_plan(topk, layer.buckets, lanes))?;
        return hip(gpu.mq_i4_moe_gemv_slab(
            false,
            layer.buckets,
            gate_up_ptrs,
            xr,
            gate_out,
            Some(up_out),
            lanes,
            layer.cold_codec,
        ));
    }
    const GATE_UP_K: usize = rdna_compute::slab_moe::SLAB_GATE_UP_K;
    const GATE_UP_HALF: usize = rdna_compute::slab_moe::SLAB_GATE_UP_M / 2;
    for (first, n) in k1_groups(lanes) {
        let ids = topk_group(topk, first, n);
        let slots = (first * SLAB_TOP_K, n * SLAB_TOP_K);
        let x = xr.sub_offset(first * GATE_UP_K, n * GATE_UP_K);
        let gate = gate_out.sub_offset(slots.0 * GATE_UP_HALF, slots.1 * GATE_UP_HALF);
        let up = up_out.sub_offset(slots.0 * GATE_UP_HALF, slots.1 * GATE_UP_HALF);
        hip(gpu.qwen4_slab_union_plan(&ids, layer.buckets, n))?;
        hip(gpu.mq_i4_moe_gemv_slab(
            false,
            layer.buckets,
            gate_up_ptrs,
            &x,
            &gate,
            Some(&up),
            n,
            layer.cold_codec,
        ))?;
    }
    Ok(())
}

/// K1 down of `lanes` lanes (up to the shared per-architecture exact cap): `xrot`
/// `[lanes * 10, 640]` F32, `y0` `[lanes * 10, 2560]` F32 unweighted.
///
/// Narrow and opted-in C16/C32 calls use the union gate/up planned (K1-C1 at one lane reads
/// `topk` instead and needs no union).
/// Otherwise a wider call runs 8-lane groups like [`k1_gate_up`]; the union
/// scratch holds only the last gate/up group's plan (the SwiGLU stage sits
/// between the two calls), so each group re-plans its own `topk` ids
/// immediately before its launch, which makes every group's down read exactly
/// the union its gate/up read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k1_down(
    gpu: &mut Gpu,
    layer: SlabLayer<'_>,
    topk: &GpuTensor,
    down_ptrs: &GpuTensor,
    xrot: &GpuTensor,
    y0: &GpuTensor,
    lanes: usize,
) -> Result<(), DispatchError> {
    k1_check_lanes(gpu, "k1_down", lanes)?;
    if lanes == 1 && gpu.slab_k1_c1() && k1_rows8_cold_codec(layer.cold_codec) {
        return hip(gpu.mq_i4_moe_gemv_slab_c1(true, layer.buckets, down_ptrs, topk, xrot, y0, None, layer.cold_codec));
    }
    if lanes <= K1_MAX_ROWS || gpu.qwen4_slab_wide_twin(lanes, layer.cold_codec) {
        return hip(gpu.mq_i4_moe_gemv_slab(
            true,
            layer.buckets,
            down_ptrs,
            xrot,
            y0,
            None,
            lanes,
            layer.cold_codec,
        ));
    }
    const DOWN_K: usize = rdna_compute::slab_moe::SLAB_DOWN_K;
    const DOWN_M: usize = rdna_compute::slab_moe::SLAB_DOWN_M;
    for (first, n) in k1_groups(lanes) {
        let ids = topk_group(topk, first, n);
        let (slot0, slots) = (first * SLAB_TOP_K, n * SLAB_TOP_K);
        let x = xrot.sub_offset(slot0 * DOWN_K, slots * DOWN_K);
        let y = y0.sub_offset(slot0 * DOWN_M, slots * DOWN_M);
        hip(gpu.qwen4_slab_union_plan(&ids, layer.buckets, n))?;
        hip(gpu.mq_i4_moe_gemv_slab(
            true,
            layer.buckets,
            down_ptrs,
            &x,
            &y,
            None,
            n,
            layer.cold_codec,
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::slab_moe_decode::{self as k1, BucketKey, Codec, Kind};
    use super::{k1_groups, K1_MAX_ROWS};
    use rdna_compute::slab_moe as gpu;

    #[test]
    fn capacity_does_not_widen_narrow_fallback_groups() {
        assert_eq!(gpu::SLAB_MAX_LANES, 32);
        assert_eq!(K1_MAX_ROWS, 8);
        for lanes in 1..=gpu::SLAB_MAX_LANES {
            let mut next = 0;
            for (first, n) in k1_groups(lanes) {
                assert_eq!(first, next);
                assert!((1..=8).contains(&n));
                next += n;
            }
            assert_eq!(next, lanes);
        }
    }

    /// The launcher's constants are the certified K1 contract's: symbols in
    /// bucket order (W3 first), geometry, lanes, and the codec byte mapping.
    #[test]
    fn launcher_contract_is_the_k1_contract() {
        for (bucket, codec) in [(0usize, Codec::Mq3I4), (1usize, Codec::Mq4Sym)] {
            assert_eq!(Codec::ORDER[bucket], codec);
            for (down, kind) in [(0usize, Kind::GateUp), (1usize, Kind::Down)] {
                assert_eq!(
                    gpu::MQ_I4_MOE_GEMV_SYMBOLS[bucket][down],
                    BucketKey::new(codec, kind).symbol()
                );
            }
        }
        for (cap_index, capacity) in [16, 32].into_iter().enumerate() {
            for bucket in 0..2 {
                for kind in 0..2 {
                    assert_eq!(
                        gpu::MQ_I4_MOE_GEMV_WIDE_SYMBOLS[cap_index][bucket][kind],
                        format!("{}_c{capacity}", gpu::MQ_I4_MOE_GEMV_SYMBOLS[bucket][kind])
                    );
                }
            }
        }
        assert_eq!(gpu::SLAB_MAX_LANES, k1::MAX_LANES);
        assert_eq!(gpu::SLAB_TOP_K, k1::TOP_K);
        assert_eq!(gpu::SLAB_EXPERTS, k1::MAX_EXPERTS);
        assert_eq!(
            (gpu::SLAB_GATE_UP_M, gpu::SLAB_GATE_UP_K),
            (k1::GATE_UP_M, k1::GATE_UP_K)
        );
        assert_eq!(
            (gpu::SLAB_DOWN_M, gpu::SLAB_DOWN_K),
            (k1::DOWN_M, k1::DOWN_K)
        );
        // Codec bytes of the per-layer table: 1 = W3 -> bucket 0, 0 = MQ4 -> bucket 1.
        assert_eq!(gpu::slab_bucket_of_codec(1), Codec::Mq3I4.order_index());
        assert_eq!(gpu::slab_bucket_of_codec(0), Codec::Mq4Sym.order_index());
    }

    /// The K1-C1 launcher's constants: its two symbols, grid `[M / 8, 10, 1]`, block
    /// `[256, 1, 1]`, the 80-byte bounded kernarg (`n_experts@72`, zero `reserved@76`),
    /// and the shapes it shares with K1.
    #[test]
    fn c1_launcher_contract() {
        assert_eq!(
            gpu::MQ_I4_MOE_GEMV_C1_SYMBOLS,
            ["mq_i4_moe_gemv_c1_r256_gate_up", "mq_i4_moe_gemv_c1_r128_down"]
        );
        assert_eq!(gpu::MQ_I4_MOE_GEMV_C1_KERNARG_BYTES, 80);
        assert_eq!((gpu::MQ_I4_MOE_GEMV_C1_ROWS_PER_WG, gpu::MQ_I4_MOE_GEMV_C1_BLOCK), (8, 256));
        let gate = gpu::slab_c1_launch(false);
        assert_eq!(gate.grid, [1280 / 8, 10, 1]);
        assert_eq!(gate.block, [256, 1, 1]);
        assert_eq!(
            (gate.m as usize, gate.k as usize, gate.x_row_div as usize, gate.y_slot_stride as usize),
            (k1::GATE_UP_M, k1::GATE_UP_K, k1::TOP_K, k1::GATE_UP_M / 2)
        );
        let down = gpu::slab_c1_launch(true);
        assert_eq!(down.grid, [2560 / 8, 10, 1]);
        assert_eq!(down.block, [256, 1, 1]);
        assert_eq!(
            (down.m as usize, down.k as usize, down.x_row_div as usize, down.y_slot_stride as usize),
            (k1::DOWN_M, k1::DOWN_K, 1, k1::DOWN_M)
        );
    }

    /// The planner's fixed-capacity layout is `6 S + 2` words per call.
    #[test]
    fn union_scratch_words_match_the_descriptor_layout() {
        for lanes in 1..=gpu::SLAB_MAX_LANES {
            let s = 10 * lanes;
            assert_eq!(
                rdna_compute::scratch::qwen4_slab_union_words(lanes),
                2 * s + 2 * (s + 1) + 2 * s
            );
        }
    }
}
