// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measured host-to-device link class and the policies that read it.
//!
//! Host-resident routed experts reach the GPU over the PCIe link, so the best
//! long-prefill route depends on how wide that link really is, which no arch
//! string says (a gfx1100 in a Gen4 x16 slot and in a Gen3 x4 riser are the
//! same card).  [`host_link`] therefore *measures* it once per device at load
//! and every consumer reads the cached value: the chunk default
//! ([`crate::gpu_forward::qwen4_prefill_chunk_requested`]), the G2 stage
//! switch ([`expert_stage_default`], used by both the load admission charge
//! and `Qwen4ExpertStage::new`) and the stage `min_rows` default.  One value
//! per device per process means admission and the constructor can never
//! disagree.
//!
//! # Probe semantics
//!
//! * **Unified memory** ([`rdna_compute::Gpu::is_uma`], the live
//!   `hipDeviceAttributeIntegrated` query, which answers "unified" when the
//!   query fails) is classified [`HostLink::Unified`] without any copy: there
//!   is no link to measure and the discrete-link policies never apply.
//! * **Discrete**: one 64 MiB `hipHostMalloc` portable pinned buffer (zero
//!   initialised, so every page is resident) is copied to a 64 MiB device
//!   buffer with `hipMemcpyAsync` on a private non-blocking stream.  One
//!   untimed warm-up copy is followed by 5 copies, each bracketed by a
//!   `hipEvent` pair; the figure is `PROBE_BYTES / min(elapsed)` in
//!   GB/s (1 GB = 10^9 bytes), the *best* sample, so contention from other
//!   processes lowers it only if every sample was slowed.  The buffers,
//!   events and stream are released before [`host_link`] returns, so the
//!   probe leaves free device memory unchanged for the admission that
//!   follows.
//! * **Fail closed**: any allocation, copy or timing failure, or a
//!   non-finite or non-positive figure, is [`HostLink::Unmeasured`].  An
//!   unmeasured link is never narrow, so no default that depends on a narrow
//!   link switches on; the result is cached like a measurement, so a later
//!   call cannot flip the decision between admission and construction.
//!
//! "Narrow" is a discrete link measuring below [`NARROW_LINK_GBPS`] (16 GB/s):
//! a Gen3 x16, Gen4 x8 or x4 slot or riser.  A Gen4 x16 link (about
//! 26 GB/s) and an R9700 (gfx1201) in a Gen5 x16 slot (27.3 GB/s measured
//! through the mapped-read probe; the link itself has about 50 GB/s of
//! headroom) are above it and keep their existing behavior.  gfx1201 does
//! not even consult the measurement: its G2 default is the arch rule that
//! predates the probe ([`expert_stage_default`]), so it behaves identically
//! whatever the probe returns.  Unified-memory devices (gfx1151 and other
//! APUs) are unaffected: they are never narrow and never probed.

use rdna_compute::Gpu;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// A discrete link measuring below this many GB/s (10^9 bytes/s) is narrow.
pub const NARROW_LINK_GBPS: f64 = 16.0;

/// Bytes of one probe copy.
const PROBE_BYTES: usize = 64 << 20;
/// Untimed copies before the timed ones (first-touch, DMA queue warm-up).
const PROBE_WARMUP_COPIES: usize = 1;
/// Timed copies; the fastest sets the figure.
const PROBE_TIMED_COPIES: usize = 5;

/// What the host-to-device path of one device looks like.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HostLink {
    /// Unified-memory device (APU): device memory is host memory, there is no
    /// link, and the discrete-link policies never apply.
    Unified,
    /// Discrete device whose measured pinned H2D copy rate is `gbps` GB/s.
    Discrete { gbps: f64 },
    /// Probe failed or returned an unusable figure. Treated as not narrow.
    Unmeasured,
}

impl HostLink {
    /// Classify a probe result. `measured_gbps` is `None` when the probe
    /// failed; non-finite and non-positive figures are failures too.
    pub fn classify(unified: bool, measured_gbps: Option<f64>) -> Self {
        if unified {
            return Self::Unified;
        }
        match measured_gbps {
            Some(gbps) if gbps.is_finite() && gbps > 0.0 => Self::Discrete { gbps },
            _ => Self::Unmeasured,
        }
    }

    /// A discrete device whose measured link is below [`NARROW_LINK_GBPS`].
    /// False for unified-memory devices and for an unmeasured link.
    pub fn is_narrow_discrete(self) -> bool {
        matches!(self, Self::Discrete { gbps } if gbps < NARROW_LINK_GBPS)
    }

    /// Short text for the load log.
    pub fn describe(self) -> String {
        match self {
            Self::Unified => "unified memory (no link)".to_string(),
            Self::Discrete { gbps } => format!(
                "discrete, {gbps:.1} GB/s H2D ({})",
                if self.is_narrow_discrete() {
                    "narrow"
                } else {
                    "not narrow"
                }
            ),
            Self::Unmeasured => "unmeasured (treated as not narrow)".to_string(),
        }
    }
}

/// Default of the G2 long-prefill expert stage switch when
/// `HIPFIRE_QWEN4_EXPERT_STAGE` is unset: on for gfx1201 (the card it was
/// measured on, whatever its link) and on every discrete device whose
/// measured link is narrow. Off for unified memory and for an unmeasured
/// link. `link` is only evaluated off gfx1201, so gfx1201 never probes.
pub fn expert_stage_default(arch: &str, link: impl FnOnce() -> HostLink) -> bool {
    arch == "gfx1201" || link().is_narrow_discrete()
}

/// [`HostLink`] of `gpu`, measured on first use and cached per device
/// ordinal for the life of the process (see the module docs for the probe).
pub fn host_link(gpu: &Gpu) -> HostLink {
    cached(gpu.device_id, || measure(gpu))
}

static CACHE: LazyLock<Mutex<HashMap<i32, HostLink>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cached lookup: the first value stored for `device` wins and is returned to
/// every caller, even when two probes race. The lock is not held while
/// measuring.
fn cached(device: i32, measure: impl FnOnce() -> HostLink) -> HostLink {
    let lock = || {
        CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    if let Some(link) = lock().get(&device).copied() {
        return link;
    }
    let link = measure();
    *lock().entry(device).or_insert(link)
}

fn measure(gpu: &Gpu) -> HostLink {
    if gpu.is_uma() {
        return HostLink::classify(true, None);
    }
    let measured = match probe_gbps(gpu) {
        Ok(gbps) => Some(gbps),
        Err(error) => {
            eprintln!(
                "  qwen4 host link: H2D probe failed on device {} ({error}); \
                 treating the link as unmeasured",
                gpu.device_id
            );
            None
        }
    };
    let link = HostLink::classify(false, measured);
    eprintln!(
        "  qwen4 host link: device {} {}",
        gpu.device_id,
        link.describe()
    );
    link
}

/// Best-of-N pinned H2D copy rate of `gpu` in GB/s, with every resource
/// released before returning.
fn probe_gbps(gpu: &Gpu) -> Result<f64, String> {
    let hip = &gpu.hip;
    gpu.bind_thread().map_err(|error| error.to_string())?;
    let mut host = hip
        .host_buffer(PROBE_BYTES)
        .map_err(|error| format!("pinned host buffer: {error}"))?;
    let device = hip
        .malloc(PROBE_BYTES)
        .map_err(|error| format!("device buffer: {error}"))?;
    let stream = match hip.stream_create_non_blocking() {
        Ok(stream) => stream,
        Err(error) => {
            let _ = hip.free(device);
            return Err(format!("stream: {error}"));
        }
    };
    let timed = (|| -> Result<f64, String> {
        let src = host.as_bytes_mut();
        let start = hip.event_create().map_err(|error| error.to_string())?;
        let stop = match hip.event_create() {
            Ok(stop) => stop,
            Err(error) => {
                let _ = hip.event_destroy(start);
                return Err(error.to_string());
            }
        };
        let samples = (|| -> Result<f64, String> {
            for _ in 0..PROBE_WARMUP_COPIES {
                hip.memcpy_htod_async(&device, src, &stream)
                    .map_err(|error| format!("warm-up copy: {error}"))?;
            }
            let mut best_ms = f32::INFINITY;
            for _ in 0..PROBE_TIMED_COPIES {
                hip.event_record(&start, Some(&stream))
                    .map_err(|error| error.to_string())?;
                hip.memcpy_htod_async(&device, src, &stream)
                    .map_err(|error| format!("timed copy: {error}"))?;
                hip.event_record(&stop, Some(&stream))
                    .map_err(|error| error.to_string())?;
                hip.event_synchronize(&stop)
                    .map_err(|error| error.to_string())?;
                let ms = hip
                    .event_elapsed_ms(&start, &stop)
                    .map_err(|error| error.to_string())?;
                best_ms = best_ms.min(ms);
            }
            Ok(gbps_from_ms(PROBE_BYTES, best_ms))
        })();
        // Copies may still be queued after an error: drain before the events
        // and the buffers go away.
        let drained = hip
            .stream_synchronize(&stream)
            .map_err(|error| format!("stream drain: {error}"));
        let _ = hip.event_destroy(start);
        let _ = hip.event_destroy(stop);
        let gbps = samples?;
        drained?;
        Ok(gbps)
    })();
    let drained = hip.stream_synchronize(&stream);
    let _ = hip.stream_destroy(stream);
    let _ = hip.free(device);
    // `host` drops last: the stream is drained, so no copy still reads it.
    drained.map_err(|error| format!("stream drain: {error}"))?;
    timed
}

/// GB/s (10^9 bytes/s) of `bytes` moved in `ms` milliseconds; 0 for a
/// non-finite or non-positive time, which [`HostLink::classify`] rejects.
fn gbps_from_ms(bytes: usize, ms: f32) -> f64 {
    if !ms.is_finite() || ms <= 0.0 {
        return 0.0;
    }
    bytes as f64 / (f64::from(ms) * 1.0e-3) / 1.0e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_fail_closed() {
        // Unified memory never carries a discrete figure, measured or not.
        assert_eq!(HostLink::classify(true, None), HostLink::Unified);
        assert_eq!(HostLink::classify(true, Some(3.0)), HostLink::Unified);
        assert_eq!(
            HostLink::classify(false, Some(27.3)),
            HostLink::Discrete { gbps: 27.3 }
        );
        for bad in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(
                HostLink::classify(false, bad),
                HostLink::Unmeasured,
                "{bad:?}"
            );
        }
    }

    #[test]
    fn narrow_means_a_measured_discrete_link_below_sixteen() {
        let narrow = |unified, gbps| HostLink::classify(unified, gbps).is_narrow_discrete();
        assert!(narrow(false, Some(3.1)));
        assert!(narrow(false, Some(15.99)));
        // The threshold itself and anything above are not narrow: Gen4 x16
        // (~26 GB/s) and the Gen5 x16 R9700 (27.3 GB/s through the mapped
        // probe) stay on their existing behavior.
        assert!(!narrow(false, Some(16.0)));
        assert!(!narrow(false, Some(27.3)));
        assert!(!narrow(false, Some(52.0)));
        // Unified and unmeasured are never narrow.
        assert!(!narrow(true, Some(3.1)));
        assert!(!narrow(true, None));
        assert!(!narrow(false, None));
    }

    #[test]
    fn stage_default_is_gfx1201_or_narrow_discrete() {
        let link = |unified, gbps| HostLink::classify(unified, gbps);
        // gfx1201 is on whatever the link, including unmeasured and unified,
        // and never evaluates the link (no probe).
        for l in [
            link(false, Some(27.3)),
            link(false, Some(8.0)),
            link(false, None),
            link(true, None),
        ] {
            assert!(expert_stage_default("gfx1201", || l), "{l:?}");
        }
        assert!(expert_stage_default("gfx1201", || unreachable!(
            "gfx1201 probed"
        )));
        // Other arches switch on only for a measured narrow discrete link.
        for arch in ["gfx1100", "gfx1030", "gfx1151", "gfx942"] {
            assert!(
                expert_stage_default(arch, || link(false, Some(12.0))),
                "{arch}"
            );
            assert!(
                !expert_stage_default(arch, || link(false, Some(27.3))),
                "{arch}"
            );
            assert!(!expert_stage_default(arch, || link(false, None)), "{arch}");
            assert!(!expert_stage_default(arch, || link(true, None)), "{arch}");
        }
    }

    #[test]
    fn gbps_arithmetic_and_bad_times() {
        // 64 MiB in 4.194304 ms is exactly 16 GB/s (decimal): the boundary.
        let ms = PROBE_BYTES as f32 / 16.0e9 * 1.0e3;
        assert!((gbps_from_ms(PROBE_BYTES, ms) - 16.0).abs() < 0.01);
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(gbps_from_ms(PROBE_BYTES, bad), 0.0, "{bad}");
        }
    }

    #[test]
    fn cache_measures_once_per_device_and_first_value_wins() {
        // Device ordinals no real GPU uses, so parallel tests cannot collide.
        let (a, b) = (-7001, -7002);
        let calls = std::cell::Cell::new(0);
        let first = cached(a, || {
            calls.set(calls.get() + 1);
            HostLink::Discrete { gbps: 9.0 }
        });
        let again = cached(a, || {
            calls.set(calls.get() + 1);
            HostLink::Discrete { gbps: 40.0 }
        });
        assert_eq!(first, HostLink::Discrete { gbps: 9.0 });
        assert_eq!(again, first);
        assert_eq!(calls.get(), 1);
        // Another device is measured on its own; an unmeasured result is
        // cached too (no retry flips a decision after admission).
        assert_eq!(cached(b, || HostLink::Unmeasured), HostLink::Unmeasured);
        assert_eq!(
            cached(b, || HostLink::Discrete { gbps: 1.0 }),
            HostLink::Unmeasured
        );
    }
}
