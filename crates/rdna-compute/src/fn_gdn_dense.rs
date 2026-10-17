// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next's gfx1201 prefill GDN on the dense C64 chunk scan.
//!
//! The fused producer preserves the lab route's serial BF16 Q/K norm,
//! F16 packing and chunk-local gate cumsum. The consumer reads the scan's
//! BF16 output directly. The Q8 state aliases the scan's codes/scales. A plain Q8 state's EF
//! enters as zero (thread-local scratch, cleared by the producer) and is discarded on exit
//! (FN keeps none). A Q8Ef state scans against its live per-layer residual (`p.residual`,
//! F16 `[head][row][key]`): the producer's clear is disabled (`ef_words == 0`), the scan
//! rebuilds `code * scale + ef` on entry and stores the new residual on exit, so the residual
//! persists across prefill launches. Exit Q8 uses
//! round-to-nearest-even, not the pipe128 recurrence's stochastic rounding.
//! Default on for admitted steps; `HIPFIRE_FN_GDN_DENSE_SCAN=0` opts out.

use std::cell::RefCell;

use hip_bridge::{DeviceBuffer, HipResult, KernargBlob};

use crate::tensor_ops::{
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched,
    GatedDeltaStepBatched, GdnStateFormat,
};
use crate::{DType, Gpu, GpuTensor};

const EF_BYTES: usize = 48 * 128 * 128 * 2;
/// Bytes of one A row of the KKT solve: 48 value heads x 64 BF16 columns.
const A_ROW_BYTES: usize = 48 * 64 * 2;
/// Admission floor of [`applies`]: below this row count the route never runs.
const MIN_ROWS: usize = 64;

/// Bytes of the thread-local A scratch for a `rows`-row step: `round64(rows)`
/// rows of [`A_ROW_BYTES`] (KKT writes whole 64-row blocks). 0 below the
/// [`applies`] row floor, where the route never allocates; `None` on overflow.
pub fn workspace_a_bytes(rows: usize) -> Option<usize> {
    if rows < MIN_ROWS {
        return Some(0);
    }
    rows.div_ceil(64).checked_mul(64)?.checked_mul(A_ROW_BYTES)
}

/// Bytes of the thread-local zero-entry EF scratch of a plain Q8 state (one per allocation,
/// rows-independent); 0 below the [`applies`] row floor. A Q8Ef state scans against its live
/// residual instead and never allocates this ([`workspace_ef_bytes_for`]).
pub fn workspace_ef_bytes(rows: usize) -> usize {
    if rows < MIN_ROWS { 0 } else { EF_BYTES }
}

/// A scratch bytes the dense scan charges for a state stored in `format`: the
/// [`workspace_a_bytes`] of an admitted Q8 / Q8Ef state, 0 for F32 (the route is unreachable).
pub fn workspace_a_bytes_for(format: GdnStateFormat, rows: usize) -> Option<usize> {
    match format {
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => workspace_a_bytes(rows),
        GdnStateFormat::F32 => Some(0),
    }
}

/// Thread-local EF scratch bytes the dense scan charges for `format`: [`workspace_ef_bytes`]
/// for plain Q8 only. A Q8Ef state's persistent residual belongs to the state owner and is
/// not charged here; F32 never reaches the route.
pub fn workspace_ef_bytes_for(format: GdnStateFormat, rows: usize) -> usize {
    match format {
        GdnStateFormat::Q8 => workspace_ef_bytes(rows),
        GdnStateFormat::Q8Ef | GdnStateFormat::F32 => 0,
    }
}

/// Total thread-local workspace (A + EF) of the dense scan for `rows` rows of a plain Q8
/// state, `None` on overflow. Excludes the aliased Q/K/V/G/scan-output views.
pub fn workspace_bytes(rows: usize) -> Option<usize> {
    workspace_bytes_for(GdnStateFormat::Q8, rows)
}

/// [`workspace_bytes`] for a state stored in `format`.
pub fn workspace_bytes_for(format: GdnStateFormat, rows: usize) -> Option<usize> {
    workspace_a_bytes_for(format, rows)?.checked_add(workspace_ef_bytes_for(format, rows))
}

struct Scratch {
    rows: usize,
    a: GpuTensor,
    /// Zero-entry EF plane of a plain Q8 state; allocated on first plain-Q8 use and never for
    /// Q8Ef (which scans against the owner's live residual).
    ef: Option<GpuTensor>,
}


thread_local! {
    /// One scratch per device id: a single submission thread can drive several
    /// devices (pipeline-parallel bands), and a buffer allocated on one device
    /// must never be handed to a kernel running on another.
    static SCRATCH: RefCell<Vec<(i32, Scratch)>> = const { RefCell::new(Vec::new()) };
}

fn raw(t: &GpuTensor, off: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: unsafe { DeviceBuffer::from_raw((t.buf.as_ptr() as *mut u8).add(off) as *mut _, len) },
        shape: vec![len],
        dtype: DType::Raw,
    }
}

/// Pure admission by architecture, state format and live residual: exact gfx1201
/// ([`crate::scratch::qwen4_dense_gdn_admits`]), a plain Q8 state (scratch EF, no residual
/// tensor) or a Q8Ef state with its live residual. F32 and every other architecture retain
/// the incumbent recurrence.
fn admits_state(arch: &str, format: GdnStateFormat, has_residual: bool) -> bool {
    crate::scratch::qwen4_dense_gdn_admits(arch, format) && has_residual == format.has_residual()
}

/// Exact gfx1201 FN geometry, Q8 or Q8Ef state (with its live residual), no
/// capture/recording, and the mseg segment geometry. Other steps retain the
/// incumbent recurrence.
pub fn applies(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    let n = p.rows;
    gpu.flags.fn_gdn_dense_scan
        && admits_state(&gpu.arch, p.format, p.residual.is_some())
        && p.row_states.is_none()
        && p.key_heads == 16
        && p.value_heads == 48
        && p.key_dim == 128
        && p.value_dim == 128
        && p.qkv_width == 10240
        && p.projection.dtype == DType::F32
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && n >= MIN_ROWS
        && (n % 512 == 0 || (64..512).contains(&(n % 512)))
}

fn launch(gpu: &mut Gpu, sym: &str, grid: [u32; 3], block: u32, args: &mut KernargBlob) -> HipResult<()> {
    // Uses the registered module/pack lookup, with the same compiler options
    // as tensor_ops (no chunk-scan-only floating-point flags).
    gpu.ensure_kernel("fn_gdn_dense", crate::kernels::FN_GDN_DENSE_SRC, sym)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        sym, grid, [block, 1, 1], 0, args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Fused convolution/parameters/packing, then KKT and mseg. Caller checked
/// [`applies`]; the input projection and convolution destination are distinct.
pub fn run(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    p: &GatedDeltaStepBatched<'_>,
) -> HipResult<()> {
    // Refuse before any launch: only an admitted format with its exact residual is run.
    if !admits_state(&gpu.arch, p.format, p.residual.is_some()) {
        return Err(hip_bridge::HipError::new(
            0,
            "fn_gdn_dense: GDN state format/residual not admitted by the dense scan",
        ));
    }
    crate::tensor_ops::validate_gdn_state(
        p.format, p.state, p.residual, None, p.rows, p.value_heads, p.key_dim, p.value_dim,
    )?;
    let n = p.rows;
    let n64 = n.div_ceil(64) * 64;
    // KKT clamps K loads to the last real row and guards G/beta; mseg clamps
    // Q/K/V loads, guards its controls, and only stores output for tok<rows.
    // Neither reads Q/K/V/G/beta pad rows. KKT DOES write a full 64-row A
    // block, so only A needs padded scratch. Q/K/V alias the dead F32 conv
    // destination (n64*10240*2 <= n*10240*4 for admitted n>=64); the producer
    // also zeros their pad rows. G aliases gate, beta stays in its original
    // plane, and BF16 scan output aliases the F32 recurrent destination.
    let q_bytes = n64 * 2048 * 2;
    let q = raw(p.projection, 0, q_bytes);
    let k = raw(p.projection, q_bytes, q_bytes);
    let v = raw(p.projection, q_bytes * 2, n64 * 6144 * 2);
    let out = raw(p.output, 0, n * 6144 * 2);
    let mut slot = SCRATCH.with(|s| {
        let mut slots = s.borrow_mut();
        let index = slots
            .iter()
            .position(|(device, _)| *device == gpu.device_id)?;
        Some(slots.swap_remove(index).1)
    });
    if slot.as_ref().is_some_and(|s| s.rows < n64) {
        let s = slot.take().unwrap();
        gpu.free_tensor(s.a)?;
        if let Some(ef) = s.ef {
            gpu.free_tensor(ef)?;
        }
    }
    let mut s = match slot {
        Some(s) => s,
        None => Scratch {
            rows: n64,
            a: gpu.zeros(
                &[workspace_a_bytes(n).ok_or_else(|| {
                    hip_bridge::HipError::new(0, "fn_gdn_dense: A scratch extent overflow")
                })?],
                DType::Raw,
            )?,
            ef: None,
        },
    };
    if p.residual.is_none() && s.ef.is_none() {
        match gpu.zeros(&[EF_BYTES], DType::Raw) {
            Ok(ef) => s.ef = Some(ef),
            Err(error) => {
                SCRATCH.with(|c| c.borrow_mut().push((gpu.device_id, s)));
                return Err(error);
            }
        }
    }
    let result = (|| -> HipResult<()> {
        // Plain Q8: the cleared thread-local scratch (`ef_words` words zeroed by the
        // producer's block x=81). Q8Ef: the owner's live residual, which the producer must
        // not clear (`ef_words == 0`) and the scan reads on entry and rewrites on exit.
        let (ef_plane, ef_words): (&GpuTensor, usize) = match (p.residual, s.ef.as_ref()) {
            (Some(residual), _) => (residual, 0),
            (None, Some(ef)) => (ef, EF_BYTES / 4),
            (None, None) => {
                return Err(hip_bridge::HipError::new(0, "fn_gdn_dense: missing EF plane"));
            }
        };
        let mut args = KernargBlob::new();
        for t in [
            conv.input, conv.kernel, conv.history, conv.next_history,
            &q, &k, &v, p.gate, p.beta,
            params.a, params.b, params.a_log, params.dt_bias,
        ] {
            args.push_ptr(t.buf.as_ptr());
        }
        args.push_ptr(ef_plane.buf.as_ptr());
        args.push_i32(n as i32);
        args.push_i32(n64 as i32);
        args.push_i32(conv.start_cursor as i32);
        args.push_i32(i32::from(conv.input.dtype == DType::BF16));
        args.push_f32((p.key_dim as f32).sqrt().recip());
        args.push_i32(ef_words as i32);
        launch(gpu, "fn_gdn_dense_producer", [82, (n64 / 32) as u32, 1], 256, &mut args)?;
        gpu.gdn_chunk_kkt_solve_batched(&k, p.gate, p.beta, &s.a, n)?;
        let codes = raw(p.state, 0, 48 * 128 * 128);
        let scales = raw(p.state, 48 * 128 * 128, 48 * 128 * 4);
        gpu.gdn_chunk_scan_layer_mseg(
            &q, &k, &v, &s.a, p.gate, p.beta,
            &codes, &scales, ef_plane, &out, n,
        )
    })();
    SCRATCH.with(|c| c.borrow_mut().push((gpu.device_id, s)));
    result
}

/// Gate the packed BF16 scan output, optionally also producing the output
/// projection's Aligned256 rotation. No later reader needs recurrent_output.
pub fn gate(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: Option<&GpuTensor>,
) -> HipResult<()> {
    let elements = p.rows * p.value_heads * p.value_dim;
    let mut args = KernargBlob::new();
    for t in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(t.buf.as_ptr());
    }
    if let Some(rotated) = rotated {
        gpu.ensure_mq_signs()?;
        args.push_ptr(rotated.buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
        launch(gpu, "fn_gdn_dense_gate_rotate_bf16in", [(elements / 256) as u32, 1, 1], 64, &mut args)?;
        gpu.scratch.prerotated = Some((
            p.output.buf.as_ptr() as usize,
            rotated.buf.as_ptr() as usize,
            elements,
        ));
        Ok(())
    } else {
        args.push_i32(p.rows as i32);
        args.push_i32(p.value_heads as i32);
        args.push_i32(p.value_dim as i32);
        launch(gpu, "fn_gdn_dense_gate_bf16in", [(elements / 128) as u32, 1, 1], 32, &mut args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_bytes_match_flash_next_4096() {
        assert_eq!(workspace_a_bytes(4096), Some(25_165_824));
        assert_eq!(workspace_ef_bytes(4096), 1_572_864);
        assert_eq!(workspace_bytes(4096), Some(26_738_688));
        // A rounds up to whole 64-row blocks; the route needs n >= 64.
        assert_eq!(workspace_a_bytes(65), workspace_a_bytes(128));
        assert_eq!(workspace_bytes(63), Some(0));
        assert_eq!(workspace_a_bytes(usize::MAX), None);
    }

    /// Dense A is charged for every admitted state; the 1.5 MiB thread-local EF scratch
    /// only for plain Q8 (Q8Ef scans against the owner's live residual, which this scratch
    /// census never charges); F32 never reaches the route.
    #[test]
    fn fn_gdn_dense_workspace_charges_by_format() {
        const A_4096: usize = 25_165_824;
        const EF: usize = 1_572_864;
        assert_eq!(EF_BYTES, EF);
        for (format, a, ef) in [
            (GdnStateFormat::Q8, A_4096, EF),
            (GdnStateFormat::Q8Ef, A_4096, 0),
            (GdnStateFormat::F32, 0, 0),
        ] {
            assert_eq!(workspace_a_bytes_for(format, 4096), Some(a), "{format:?} A");
            assert_eq!(workspace_ef_bytes_for(format, 4096), ef, "{format:?} EF");
            assert_eq!(workspace_bytes_for(format, 4096), Some(a + ef), "{format:?} total");
            // Below the row floor the route never allocates, whatever the format.
            assert_eq!(workspace_bytes_for(format, MIN_ROWS - 1), Some(0), "{format:?} floor");
        }
        assert_eq!(workspace_bytes(4096), workspace_bytes_for(GdnStateFormat::Q8, 4096));
        // The residual tensor Q8Ef scans against is exactly the scratch plane's extent.
        assert_eq!(GdnStateFormat::Q8Ef.residual_bytes(48, 128, 128), EF_BYTES);
        for format in [GdnStateFormat::Q8, GdnStateFormat::Q8Ef, GdnStateFormat::F32] {
            assert_eq!(workspace_a_bytes_for(format, usize::MAX), if format == GdnStateFormat::F32 { Some(0) } else { None });
        }
    }

    /// `applies`' format/architecture/residual predicate: exact gfx1201, Q8 without a
    /// residual tensor, Q8Ef with one; never F32, never another architecture (gfx942 is
    /// F32-only), never a residual/format mismatch.
    #[test]
    fn fn_gdn_dense_admission_table() {
        use GdnStateFormat::{Q8, Q8Ef, F32};
        let arches = ["gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx1200", ""];
        for arch in arches {
            for (format, residual, want) in [
                (Q8, false, true),
                (Q8, true, false),
                (Q8Ef, true, true),
                (Q8Ef, false, false),
                (F32, false, false),
                (F32, true, false),
            ] {
                assert_eq!(
                    admits_state(arch, format, residual),
                    arch == "gfx1201" && want,
                    "{arch} {format:?} residual={residual}"
                );
            }
        }
    }
}
