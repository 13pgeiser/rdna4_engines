// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PeaceMaker twins of the exact (`cand_mode = 8`) gfx1201 IU4 C8 producers that feed the A3B
//! F3 prefill route (`HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`), behind
//! `HIPFIRE_PREFILL_IU4_C8_PM` ([`PrefillRoute::PrefillIu4C8Pm`], qualified on exact gfx1201:
//! unset follows the qualification, `1` opts in, `0` kills).
//!
//! * [`Gpu::mq_rotate_x_i4_c8_pm`]: twin of `Gpu::rotate_x_mq_i4_gfx12_exact_batched`
//!   (`iu4_c8_rotate_pm_gfx1201`, block 32, grid `rows * K / 256`, K 2048 or 4096).
//! * [`Gpu::fused_rmsnorm_mq_rotate_i4_c8_pm`]: twin of
//!   `Gpu::fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched` (`iu4_c8_rmsnorm_rotate_pm_gfx1201`,
//!   block 256, grid `rows`, K 2048).
//!
//! Both are standalone low-level launchers with the exact signature of their HIP twin: callable
//! regardless of any flag (the byte oracle `iu4_c8_pm_oracle` calls them directly), eager only,
//! no HIP fallback, and they always store the f32 rotated side output (`x_out` / `x_rot` must be
//! `Some`). The production exact wrappers in `gemv.rs` intercept into them only where
//! [`Gpu::iu4_c8_pm_admitted`] holds (both flags, raw inventory, eager, supported shape and
//! operands) and otherwise run the unchanged HIP exact path. The shared-down SwiGLU producer and
//! the sigmoid transform keep the incumbent.
//!
//! The kernarg ABI is the frozen 72-byte block of the PeaceMaker family
//! (`peacemaker_isa::kernels::iu4_c8_producer`): x f32 `@0`, rotated f32 output `@8`, sidecar u8
//! `@16`, signs1 f32 `@24`, signs2 f32 `@32`, norm weight f32 `@40` (null for rotate), AWQ f32
//! `@48` (null when absent), rows i32 `@56`, K i32 `@60`, eps f32 `@64`, padding `@68`.

use crate::prefill_route_capabilities::{raw_inventory, PrefillRoute};
use crate::scratch::{Int4MmqPrepared, Int4MmqReservation};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

pub const MODULE: &str = "iu4_c8_producer_pm_gfx1201";
pub const ROTATE_SYMBOL: &str = "iu4_c8_rotate_pm_gfx1201";
pub const NORM_SYMBOL: &str = "iu4_c8_rmsnorm_rotate_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/iu4_c8_producer_pm_gfx1201.hxaco");
pub const KERNARG_BYTES: usize = 72;
pub const ROTATE_BLOCK: [u32; 3] = [32, 1, 1];
pub const NORM_BLOCK: [u32; 3] = [256, 1, 1];
/// Elements of one `block_i4_128` pair (one 256-element FWHT group per wave).
pub const GROUP: usize = 256;
pub const MAX_ROWS: usize = 8192;
/// K served by the rotate export.
pub const ROTATE_KS: [usize; 2] = [2048, 4096];
/// The only K served by the RMSNorm export.
pub const NORM_K: usize = 2048;

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("iu4_c8_pm: {}", message.as_ref()))
}

/// CPU-only shape support of the two exports: `rows` in `1..=8192`; rotate K 2048 or 4096, norm
/// K 2048 (`is_norm`). Never touches HIP.
pub fn c8_pm_supported_shape(is_norm: bool, k: usize, rows: usize) -> bool {
    (1..=MAX_ROWS).contains(&rows)
        && if is_norm {
            k == NORM_K
        } else {
            ROTATE_KS.contains(&k)
        }
}

/// Why a production exact C8 producer keeps the HIP incumbent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum C8PmRefusal {
    /// `HIPFIRE_PREFILL_IU4_C8_PM` unset or `0`.
    Disabled,
    /// The resolved `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` flag is off (the route nests beneath it).
    F3Disabled,
    /// Raw exact-gfx1201 inventory of the route or of F3 is absent.
    InventoryUnavailable,
    /// Recording or capture is active.
    NotEager,
    /// Rows or K outside the exports' domain.
    Shape,
    /// Missing f32 side output, non-f32 or undersized operand, or reservation mismatch.
    Operands,
}

impl C8PmRefusal {
    fn log_once(self) {
        let index = match self {
            // An unset/0 route flag and an F3 kill are the caller's choices: silent.
            Self::Disabled | Self::F3Disabled => return,
            Self::InventoryUnavailable => 0,
            Self::NotEager => 1,
            Self::Shape => 2,
            Self::Operands => 3,
        };
        static REFUSALS: [std::sync::Once; 4] = [
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
        ];
        REFUSALS[index].call_once(|| {
            eprintln!(
                "[iu4-c8-pm] HIPFIRE_PREFILL_IU4_C8_PM producer refused: {self:?}; \
                 keeping the HIP exact C8 producer"
            );
        });
    }
}

/// Pure admission of the PM C8 producers inside the production exact wrappers: the resolved
/// route flag (`c8_flag`: explicit `1`, unset and `0` are off), the resolved F3 flag it nests
/// beneath, the raw exact-gfx1201 inventory of both routes, eager execution, the supported shape
/// and valid operands.
#[allow(clippy::too_many_arguments)]
pub fn c8_pm_admission(
    c8_flag: bool,
    f3_flag: bool,
    arch: &str,
    eager: bool,
    is_norm: bool,
    k: usize,
    rows: usize,
    operands_ok: bool,
) -> Result<(), C8PmRefusal> {
    use C8PmRefusal::*;
    if !c8_flag {
        Err(Disabled)
    } else if !f3_flag {
        Err(F3Disabled)
    } else if !raw_inventory(arch, PrefillRoute::PrefillIu4C8Pm)
        || !raw_inventory(arch, PrefillRoute::PrefillIu4ProducerFused)
    {
        Err(InventoryUnavailable)
    } else if !eager {
        Err(NotEager)
    } else if !c8_pm_supported_shape(is_norm, k, rows) {
        Err(Shape)
    } else if !operands_ok {
        Err(Operands)
    } else {
        Ok(())
    }
}

/// The frozen 72-byte kernarg block.
#[allow(clippy::too_many_arguments)]
fn kernargs(
    x: *mut c_void,
    rotated: *mut c_void,
    sidecar: *mut c_void,
    signs1: *mut c_void,
    signs2: *mut c_void,
    weight: *mut c_void,
    awq: *mut c_void,
    rows: usize,
    k: usize,
    eps: f32,
) -> [u8; KERNARG_BYTES] {
    let mut args = [0u8; KERNARG_BYTES];
    for (offset, ptr) in [
        (0, x),
        (8, rotated),
        (16, sidecar),
        (24, signs1),
        (32, signs2),
        (40, weight),
        (48, awq),
    ] {
        args[offset..offset + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
    }
    args[56..60].copy_from_slice(&(rows as i32).to_le_bytes());
    args[60..64].copy_from_slice(&(k as i32).to_le_bytes());
    args[64..68].copy_from_slice(&eps.to_le_bytes());
    args
}

fn f32_span(tensor: &GpuTensor, floats: usize) -> bool {
    tensor.dtype == DType::F32 && tensor.buf.size() >= floats * std::mem::size_of::<f32>()
}

/// Operand validity shared by the standalone launchers and the production admission.
#[allow(clippy::too_many_arguments)]
fn operands_valid(
    is_norm: bool,
    x: &GpuTensor,
    weight: Option<&GpuTensor>,
    awq: Option<&GpuTensor>,
    out: Option<&GpuTensor>,
    reservation: &Int4MmqReservation,
    k: usize,
    rows: usize,
) -> bool {
    // The shape check bounds rows * k, so the products cannot overflow.
    if !c8_pm_supported_shape(is_norm, k, rows) {
        return false;
    }
    reservation.k() == k
        && reservation.n() == rows
        && f32_span(x, rows * k)
        && out.is_some_and(|t| f32_span(t, rows * k))
        && awq.is_none_or(|t| f32_span(t, k))
        && match (is_norm, weight) {
            (true, Some(w)) => f32_span(w, k),
            (true, None) => false,
            (false, _) => true,
        }
}

impl Gpu {
    /// Whether no replay recording and no HIP stream capture is active.
    fn iu4_c8_pm_eager(&self) -> bool {
        !self.replay.is_recording() && !self.graphs.capture_mode
    }

    /// Current-device admission used by the production exact wrappers: the resolved flags of
    /// this device (`flags.prefill_iu4_c8_pm`, `flags.prefill_iu4_producer_fused`), the pure
    /// [`c8_pm_admission`] and the operands the standalone launchers would validate. Enabled
    /// refusals log once per reason; a refusal means the caller runs the HIP exact producer.
    #[allow(clippy::too_many_arguments)]
    pub fn iu4_c8_pm_admitted(
        &self,
        is_norm: bool,
        x: &GpuTensor,
        weight: Option<&GpuTensor>,
        awq: Option<&GpuTensor>,
        out: Option<&GpuTensor>,
        reservation: &Int4MmqReservation,
        k: usize,
        rows: usize,
    ) -> bool {
        match c8_pm_admission(
            self.flags.prefill_iu4_c8_pm,
            self.flags.prefill_iu4_producer_fused,
            &self.arch,
            self.iu4_c8_pm_eager(),
            is_norm,
            k,
            rows,
            operands_valid(is_norm, x, weight, awq, out, reservation, k, rows),
        ) {
            Ok(()) => true,
            Err(reason) => {
                reason.log_once();
                false
            }
        }
    }

    fn iu4_c8_pm_require_eager(&self) -> HipResult<()> {
        if !self.iu4_c8_pm_eager() {
            return Err(err(
                "eager only: capture and replay recording are unsupported",
            ));
        }
        if self.arch != "gfx1201" {
            return Err(err("requires gfx1201; no HIP fallback"));
        }
        Ok(())
    }

    /// PeaceMaker twin of [`Gpu::rotate_x_mq_i4_gfx12_exact_batched`]: FWHT rotate of `x_in`
    /// `[rows, k]` into the f32 `x_out` (always written; `None` is rejected) and the C8
    /// `block_i4_128` AoS sidecar sealed into `reservation`. `awq = Some` divides by the AWQ
    /// scales. K 2048 or 4096, rows `1..=8192`. Eager gfx1201 only, no HIP fallback; reads no
    /// flag.
    pub fn mq_rotate_x_i4_c8_pm(
        &mut self,
        x_in: &GpuTensor,
        awq: Option<&GpuTensor>,
        x_out: Option<&GpuTensor>,
        reservation: Int4MmqReservation,
        k: usize,
        batch_size: usize,
    ) -> HipResult<Int4MmqPrepared> {
        self.launch_iu4_c8_pm(
            false,
            x_in,
            None,
            awq,
            x_out,
            reservation,
            k,
            0.0,
            batch_size,
        )
    }

    /// PeaceMaker twin of [`Gpu::fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched`]: RMSNorm of
    /// `x` by the f32 `weight`, FWHT rotate into the f32 `x_rot` (always written; `None` is
    /// rejected) and the C8 `block_i4_128` AoS sidecar sealed into `reservation`. K 2048, rows
    /// `1..=8192`. Eager gfx1201 only, no HIP fallback; reads no flag.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_rmsnorm_mq_rotate_i4_c8_pm(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        awq: Option<&GpuTensor>,
        x_rot: Option<&GpuTensor>,
        reservation: Int4MmqReservation,
        k: usize,
        eps: f32,
        batch_size: usize,
    ) -> HipResult<Int4MmqPrepared> {
        self.launch_iu4_c8_pm(
            true,
            x,
            Some(weight),
            awq,
            x_rot,
            reservation,
            k,
            eps,
            batch_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_iu4_c8_pm(
        &mut self,
        is_norm: bool,
        x: &GpuTensor,
        weight: Option<&GpuTensor>,
        awq: Option<&GpuTensor>,
        out: Option<&GpuTensor>,
        reservation: Int4MmqReservation,
        k: usize,
        eps: f32,
        rows: usize,
    ) -> HipResult<Int4MmqPrepared> {
        self.bind_thread()?;
        self.iu4_c8_pm_require_eager()?;
        let Some(out) = out else {
            return Err(err(
                "the f32 rotated side output is always written: x_out/x_rot must be Some",
            ));
        };
        if !operands_valid(is_norm, x, weight, awq, Some(out), &reservation, k, rows) {
            return Err(err(format!(
                "unsupported or invalid operands: norm={is_norm} k={k} rows={rows} (rotate K 2048|4096, \
                 norm K 2048, rows 1..=8192, F32 operands sized for rows*k, reservation (k,n) match)"
            )));
        }
        let symbol = if is_norm { NORM_SYMBOL } else { ROTATE_SYMBOL };
        self.ensure_mq_signs()?;
        self.ensure_embedded_kernel(MODULE, IMAGE, symbol)?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let mut args = kernargs(
            x.buf.as_ptr(),
            out_ptr,
            reservation.ptr(),
            s1,
            s2,
            weight.map_or(std::ptr::null_mut(), |w| w.buf.as_ptr()),
            awq.map_or(std::ptr::null_mut(), |a| a.buf.as_ptr()),
            rows,
            k,
            eps,
        );
        let (grid, block) = if is_norm {
            ([rows as u32, 1, 1], NORM_BLOCK)
        } else {
            ([((k / GROUP) * rows) as u32, 1, 1], ROTATE_BLOCK)
        };
        let bytes = (k * 4 * if is_norm { 3 } else { 2 } + (k / 128) * 72) * rows;
        let timer = crate::profile::begin_timer(
            &self.hip,
            if is_norm { "fused" } else { "fwht" },
            symbol,
            bytes,
        );
        let result = self.launch_blob_recorded(
            symbol,
            grid,
            block,
            0,
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(out_ptr);
        result?;
        Ok(Int4MmqPrepared::from_reservation(reservation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCHES: [&str; 6] = [
        "gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx906", "gfx1030",
    ];

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        !needle.is_empty()
            && haystack
                .windows(needle.len())
                .any(|window| window == needle)
    }

    #[test]
    fn launcher_constants_match_the_family_authoring_contract() {
        use peacemaker_isa::kernels::iu4_c8_producer as author;
        assert_eq!(MODULE, author::MODULE);
        assert_eq!(ROTATE_SYMBOL, author::ROTATE_SYMBOL);
        assert_eq!(NORM_SYMBOL, author::NORM_SYMBOL);
        assert_eq!(KERNARG_BYTES as u32, author::KERNARG_BYTES);
        assert_eq!(ROTATE_BLOCK[0], author::ROTATE_BLOCK);
        assert_eq!(NORM_BLOCK[0], author::NORM_BLOCK);
        assert_eq!(GROUP as u32, author::GROUP);
    }

    #[test]
    fn embedded_image_carries_both_exports() {
        for symbol in [ROTATE_SYMBOL, NORM_SYMBOL] {
            assert!(contains(IMAGE, symbol.as_bytes()), "{symbol} missing");
            assert!(
                contains(IMAGE, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
        }
    }

    #[test]
    fn frozen_kernarg_abi() {
        let args = kernargs(
            0x1000usize as *mut c_void,
            0x2000usize as *mut c_void,
            0x3000usize as *mut c_void,
            0x4000usize as *mut c_void,
            0x5000usize as *mut c_void,
            0x6000usize as *mut c_void,
            std::ptr::null_mut(),
            8191,
            2048,
            1e-6,
        );
        assert_eq!(args.len(), 72);
        let ptr = |at: usize| u64::from_le_bytes(args[at..at + 8].try_into().unwrap());
        for (at, want) in [
            (0, 0x1000u64),
            (8, 0x2000),
            (16, 0x3000),
            (24, 0x4000),
            (32, 0x5000),
            (40, 0x6000),
            (48, 0),
        ] {
            assert_eq!(ptr(at), want, "@{at}");
        }
        assert_eq!(i32::from_le_bytes(args[56..60].try_into().unwrap()), 8191);
        assert_eq!(i32::from_le_bytes(args[60..64].try_into().unwrap()), 2048);
        assert_eq!(f32::from_le_bytes(args[64..68].try_into().unwrap()), 1e-6);
        assert_eq!(&args[68..72], &[0; 4], "padding stays zero");
    }

    #[test]
    fn supported_shape_table() {
        for rows in [1, 7, 64, 512, 4096, 8192] {
            assert!(
                c8_pm_supported_shape(false, 2048, rows),
                "rotate 2048 x {rows}"
            );
            assert!(
                c8_pm_supported_shape(false, 4096, rows),
                "rotate 4096 x {rows}"
            );
            assert!(
                c8_pm_supported_shape(true, 2048, rows),
                "norm 2048 x {rows}"
            );
            assert!(
                !c8_pm_supported_shape(true, 4096, rows),
                "norm 4096 x {rows}"
            );
        }
        for rows in [0, 8193, usize::MAX] {
            assert!(!c8_pm_supported_shape(false, 2048, rows), "rows {rows}");
            assert!(!c8_pm_supported_shape(true, 2048, rows), "rows {rows}");
        }
        for k in [0, 128, 256, 512, 1024, 2047, 2049, 3072, 6144, 8192] {
            assert!(!c8_pm_supported_shape(false, k, 64), "rotate k {k}");
            assert!(!c8_pm_supported_shape(true, k, 64), "norm k {k}");
        }
    }

    #[test]
    fn admission_requires_every_gate_in_order() {
        use C8PmRefusal as R;
        let ok = |c8: bool,
                  f3: bool,
                  arch: &str,
                  eager: bool,
                  norm: bool,
                  k: usize,
                  rows: usize,
                  operands: bool| {
            c8_pm_admission(c8, f3, arch, eager, norm, k, rows, operands)
        };
        assert_eq!(
            ok(true, true, "gfx1201", true, false, 4096, 512, true),
            Ok(())
        );
        assert_eq!(
            ok(true, true, "gfx1201", true, true, 2048, 8192, true),
            Ok(())
        );
        assert_eq!(
            ok(false, true, "gfx1201", true, true, 2048, 512, true),
            Err(R::Disabled)
        );
        assert_eq!(
            ok(true, false, "gfx1201", true, true, 2048, 512, true),
            Err(R::F3Disabled)
        );
        for arch in [
            "gfx1200", "gfx1151", "gfx1100", "gfx942", "gfx906", "unknown",
        ] {
            assert_eq!(
                ok(true, true, arch, true, true, 2048, 512, true),
                Err(R::InventoryUnavailable),
                "{arch}"
            );
        }
        assert_eq!(
            ok(true, true, "gfx1201", false, true, 2048, 512, true),
            Err(R::NotEager)
        );
        assert_eq!(
            ok(true, true, "gfx1201", true, true, 4096, 512, true),
            Err(R::Shape)
        );
        assert_eq!(
            ok(true, true, "gfx1201", true, false, 2048, 0, true),
            Err(R::Shape)
        );
        assert_eq!(
            ok(true, true, "gfx1201", true, false, 2048, 8193, true),
            Err(R::Shape)
        );
        assert_eq!(
            ok(true, true, "gfx1201", true, false, 2048, 512, false),
            Err(R::Operands)
        );
    }

    #[test]
    fn route_follows_its_qualification_and_nests_beneath_f3() {
        use crate::prefill_route_capabilities::enabled;
        // Resolve both flags exactly as `FeatureFlags` does, then admit.
        let admit = |arch: &str, f3: Option<bool>, c8: Option<bool>| {
            c8_pm_admission(
                enabled(c8, arch, PrefillRoute::PrefillIu4C8Pm),
                enabled(f3, arch, PrefillRoute::PrefillIu4ProducerFused),
                arch,
                true,
                true,
                2048,
                512,
                true,
            )
        };
        use C8PmRefusal as R;
        for arch in ARCHES {
            let gfx1201 = arch == "gfx1201";
            // Unset: the route follows its exact-gfx1201 qualification beneath the resolved F3.
            let unset = if gfx1201 { Ok(()) } else { Err(R::Disabled) };
            assert_eq!(admit(arch, None, None), unset, "{arch}");
            assert_eq!(admit(arch, Some(true), None), unset, "{arch}");
            // Explicit 0 kills.
            assert_eq!(admit(arch, None, Some(false)), Err(R::Disabled), "{arch}");
            assert_eq!(
                admit(arch, Some(true), Some(false)),
                Err(R::Disabled),
                "{arch}"
            );
            // Explicit 1 follows F3 and the exact-gfx1201 inventory.
            assert_eq!(
                admit(arch, None, Some(true)),
                if gfx1201 { Ok(()) } else { Err(R::F3Disabled) },
                "{arch}"
            );
            assert_eq!(
                admit(arch, Some(false), Some(true)),
                Err(R::F3Disabled),
                "{arch}"
            );
            assert_eq!(
                admit(arch, Some(true), Some(true)),
                if gfx1201 {
                    Ok(())
                } else {
                    Err(R::InventoryUnavailable)
                },
                "{arch}"
            );
        }
    }
}
