// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-load default of the D2 CPU split (decode and `<=4`-row verify of
//! host-placed routed experts on the host).
//!
//! The split is requested explicitly by `memory.offload_exec=cpu` for hot-set
//! tier layers (which also arms the whole-layer D1 splice on un-tiered layers
//! and flips the process-wide `hipfire_dispatch::cpu_exec_enabled()` mode).
//! This module is the *only* other way to request it, and it touches none of
//! that: the verdict is a plain `bool` for one load, consumed by the D2 layer
//! admission alone (`CpuSplitAdmission::enabled`). `memory.offload_exec` is not
//! rewritten, `cpu_exec_enabled()` and `passback_enabled()` keep their values,
//! D1 never arms from it, and no process-global state is latched (a second
//! model load decides for itself).
//!
//! # What it arms
//!
//! Every Flash-Next trunk layer that places routed experts in host memory and
//! the split can serve: a hot-set tier layer with at least one host-mapped
//! expert, or an un-tiered aggregate layer whose two expert allocations are
//! host-mapped QT44/QT53 (the shipped placement; the split reads those pinned
//! bytes where they are, so nothing is copied or allocated twice and placement
//! is whatever the loader resolved).
//!
//! # Selector
//!
//! In order, the first match wins:
//!
//! 1. `memory.offload_exec` explicitly set (any spelling, `pcie` included):
//!    the default does not apply. `cpu` keeps today's explicit behavior,
//!    `pcie`/`passback` keep D2 off.
//! 2. `developer.qwen4_d2_default = false` (`HIPFIRE_QWEN4_D2_DEFAULT=0`): off.
//! 3. `HIPFIRE_MOE_CPU_EXPERTS=0` (the developer kill-switch): off.
//! 4. No qualifying host split layer: off.
//! 5. Unified memory (`Gpu::is_uma`): off, and never probed (there is no link).
//! 6. `developer.qwen4_d2_default = true` (`HIPFIRE_QWEN4_D2_DEFAULT=1`): on,
//!    skipping only the rate comparison (lab and receipt runs).
//! 7. The measured rates of the **existing** passback calibration
//!    (`hipfire_dispatch::offload_split::probe_synthetic_on`): the CPU
//!    GEMV on the splice's own pinned physical-core pool against the GPU GEMV
//!    reading the same host-mapped bytes over the link, on the layers' real
//!    gate/up format. On only when the CPU is strictly faster. A failed or
//!    unusable measurement is off, and so is a tie.
//!
//! There is no arch rule: a gfx1201 card on a link the CPU beats arms, and a
//! gfx1100 card on a link the CPU does not beat does not.
//!
//! Both rates are the calibration's *alone-and-dense* figures (the module docs
//! of `offload_split`): upper bounds of what an expert decode of a few rows
//! reaches on either engine, so the comparison is a heuristic selector, not a
//! prediction of decode tokens per second.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

use hipfire_config::memory::OffloadExec;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::offload_split::{probe_synthetic_on, CpuProbePool};
use hipfire_runtime::offload_calibrate::{DEFAULT_BUFFER_MB, DEFAULT_REPS};
use rdna_compute::{DType, Gpu};

use crate::config::Qwen4Config;
use crate::weights::Qwen4Weights;

/// One device's measured engine rates over the same host-mapped bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostDecodeRates {
    /// CPU GEMV on the splice pool, bytes of weight per second.
    pub cpu_bytes_per_s: f64,
    /// GPU GEMV reading the host-mapped weight over the link (zero-copy),
    /// bytes of weight per second.
    pub link_bytes_per_s: f64,
}

impl HostDecodeRates {
    /// `None` for a non-finite or non-positive figure: an unusable
    /// measurement is no measurement.
    pub fn classify(cpu_bytes_per_s: f64, link_bytes_per_s: f64) -> Option<Self> {
        let usable = |v: f64| v.is_finite() && v > 0.0;
        (usable(cpu_bytes_per_s) && usable(link_bytes_per_s)).then_some(Self {
            cpu_bytes_per_s,
            link_bytes_per_s,
        })
    }

    /// The CPU engine strictly beats the link engine.
    pub fn cpu_faster(self) -> bool {
        self.cpu_bytes_per_s > self.link_bytes_per_s
    }
}

/// Why the D2 default is on or off for one load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum D2Default {
    /// `memory.offload_exec` is set: the default does not apply.
    Explicit(OffloadExec),
    /// `developer.qwen4_d2_default = false`.
    Disabled,
    /// `HIPFIRE_MOE_CPU_EXPERTS=0`.
    KillSwitch,
    /// No layer places experts in host memory where the split can serve them.
    NoHostSplitLayers,
    /// Unified-memory device: no link, never probed.
    Unified,
    /// `developer.qwen4_d2_default = true`: on without the rate comparison.
    Forced,
    /// The calibration failed or returned an unusable figure.
    Unmeasured,
    /// Measured: the CPU engine is faster than the link engine.
    CpuFaster(HostDecodeRates),
    /// Measured: the link engine is at least as fast as the CPU engine.
    LinkFaster(HostDecodeRates),
}

impl D2Default {
    /// Whether the default requests the D2 split for this load. An explicit
    /// setting is not the default: it is honored by the existing admission.
    pub fn selects_d2(self) -> bool {
        matches!(self, Self::Forced | Self::CpuFaster(_))
    }

    /// Load-log text; `None` for the silent cases (nothing to decide, or the
    /// user decided).
    pub fn describe(self) -> Option<String> {
        let gb = |v: f64| v / 1e9;
        match self {
            Self::Explicit(_) | Self::NoHostSplitLayers => None,
            Self::Disabled => Some("off (developer.qwen4_d2_default=false)".to_owned()),
            Self::KillSwitch => Some("off (HIPFIRE_MOE_CPU_EXPERTS=0)".to_owned()),
            Self::Unified => Some("off (unified memory: no link to beat)".to_owned()),
            Self::Forced => Some("FORCED ON (developer.qwen4_d2_default=true)".to_owned()),
            Self::Unmeasured => {
                Some("off (engine-rate calibration failed or was unusable)".to_owned())
            }
            Self::CpuFaster(r) => Some(format!(
                "ON (CPU {:.1} GB/s beats the host link {:.1} GB/s on the host layers' gate/up \
                 format; alone-and-dense calibration; set memory.offload_exec=pcie to opt out)",
                gb(r.cpu_bytes_per_s),
                gb(r.link_bytes_per_s)
            )),
            Self::LinkFaster(r) => Some(format!(
                "off (host link {:.1} GB/s is at least as fast as the CPU {:.1} GB/s on the \
                 host layers' gate/up format)",
                gb(r.link_bytes_per_s),
                gb(r.cpu_bytes_per_s)
            )),
        }
    }
}

/// The selector truth table. `rates` is only evaluated when every earlier
/// rule passed, so an explicit setting, the typed `false`, the kill-switch, a
/// load without host split layers, a unified-memory device and a typed `true`
/// never run the calibration.
pub fn decide(
    explicit: Option<OffloadExec>,
    forced: Option<bool>,
    kill_switch: bool,
    host_split_layers: usize,
    unified: bool,
    rates: impl FnOnce() -> Option<HostDecodeRates>,
) -> D2Default {
    if let Some(mode) = explicit {
        return D2Default::Explicit(mode);
    }
    if forced == Some(false) {
        return D2Default::Disabled;
    }
    if kill_switch {
        return D2Default::KillSwitch;
    }
    if host_split_layers == 0 {
        return D2Default::NoHostSplitLayers;
    }
    if unified {
        return D2Default::Unified;
    }
    if forced == Some(true) {
        return D2Default::Forced;
    }
    match rates() {
        None => D2Default::Unmeasured,
        Some(r) if r.cpu_faster() => D2Default::CpuFaster(r),
        Some(r) => D2Default::LinkFaster(r),
    }
}

/// What the loader resolved for one trunk layer's routed experts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LayerHostFacts {
    /// `Some(any expert is host-mapped)` for a hot-set tier layer, `None` for
    /// an un-tiered aggregate layer.
    tier_any_host: Option<bool>,
    /// Un-tiered layer: both aggregate allocations are host-mapped.
    aggregate_host: bool,
    /// Un-tiered layer: gate/up QT44 and down QT53, the CPU-executed pair.
    aggregate_qt44_qt53: bool,
    /// `qwen4.expert_keep` payload (compact, original-id owners cannot index it).
    pruned: bool,
}

impl LayerHostFacts {
    fn qualifies(self) -> bool {
        !self.pruned
            && match self.tier_any_host {
                Some(any_host) => any_host,
                None => self.aggregate_host && self.aggregate_qt44_qt53,
            }
    }
}

/// Trunk layers the split could serve that place experts in host memory, and
/// the gate/up format of the first one (the calibration's shape). Single
/// device, unpruned Flash-Next only: EP, pipeline bands, MTP-head and
/// expert-slab (arena) layers never arm here.
fn host_split_layers(
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    single_scope: bool,
) -> (usize, Option<DType>) {
    if !single_scope || weights.ep_placement().is_some() || !config.profile.is_flash_next() {
        return (0, None);
    }
    let mut count = 0;
    let mut dtype = None;
    for layer in &weights.layer_refs {
        let Some(moe) = layer.ffn.moe() else { continue };
        if layer.layer >= config.num_hidden_layers
            || moe.expert_arena.is_some()
            || moe.gate.name.starts_with("mtp.")
        {
            continue;
        }
        let pair = weights
            .resident(&moe.experts_gate_up)
            .ok()
            .zip(weights.resident(&moe.experts_down).ok());
        let facts = LayerHostFacts {
            tier_any_host: weights
                .expert_tier_layer(layer.layer)
                .map(|view| view.any_host()),
            aggregate_host: pair
                .is_some_and(|(gu, dn)| gu.buf.is_host_mapped() && dn.buf.is_host_mapped()),
            aggregate_qt44_qt53: pair.is_some_and(|(gu, dn)| {
                gu.dtype == DType::MQ4G256V2 && dn.dtype == DType::MQ4G128V2
            }),
            pruned: config.kept_expert_ids(layer.layer, false).is_some(),
        };
        if facts.qualifies() {
            count += 1;
            // An un-tiered layer's real gate/up format; a tier layer's entries
            // are MQ4 XTS by construction (QT44 gate/up).
            dtype.get_or_insert(pair.map_or(DType::MQ4G256V2, |(gu, _)| gu.dtype));
        }
    }
    (count, dtype)
}

/// What one measurement is valid for: the device ordinal and the probe shape
/// (gate/up format and `k`). Another model's shape never reuses it.
type RateKey = (i32, DType, usize);

static RATES: LazyLock<Mutex<BTreeMap<RateKey, Option<HostDecodeRates>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// `measure()` once per distinct key; the first stored value wins and the lock
/// is not held while measuring (the same discipline as
/// [`crate::host_link::host_link`]).
fn cached_rates(
    cache: &Mutex<BTreeMap<RateKey, Option<HostDecodeRates>>>,
    key: RateKey,
    measure: impl FnOnce() -> Option<HostDecodeRates>,
) -> Option<HostDecodeRates> {
    let lock = || {
        cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    if let Some(cached) = lock().get(&key).copied() {
        return cached;
    }
    let measured = measure();
    *lock().entry(key).or_insert(measured)
}

/// The rates for this device and probe shape, measured once per process per
/// `(device ordinal, gate/up dtype, k)`.
fn measured_rates(gpu: &mut Gpu, config: &Qwen4Config, dtype: DType) -> Option<HostDecodeRates> {
    let k = config.hidden_size;
    let key = (gpu.device_id, dtype, k);
    cached_rates(&RATES, key, || probe_rates(gpu, dtype, k))
}

/// The exact measurement the default's selector takes for one shape: the CPU
/// GEMV on the splice's pinned physical-core pool against the GPU GEMV over
/// the same host-mapped bytes, [`DEFAULT_BUFFER_MB`] of weight, the median of
/// [`DEFAULT_REPS`] passes each. `None` when the format has no CPU decoder, the
/// shape is invalid, the measurement fails or a figure is unusable. The shape
/// is the host layers' gate/up format at `k` = hidden size (Flash-Next:
/// `MQ4G256V2`, 2560); `hipfire calibrate` runs this same function so a stored
/// result and a load-time measurement cannot drift.
pub fn probe_rates(gpu: &mut Gpu, dtype: DType, k: usize) -> Option<HostDecodeRates> {
    let row_bytes = hipfire_dispatch::row_bytes_for(dtype, k)?;
    if row_bytes == 0 {
        return None;
    }
    let m = (DEFAULT_BUFFER_MB << 20).div_ceil(row_bytes);
    let ctx = DispatchCtx::new(gpu);
    match probe_synthetic_on(gpu, &ctx, dtype, m, k, DEFAULT_REPS, CpuProbePool::Splice) {
        Ok(cal) => HostDecodeRates::classify(cal.cpu_bytes_per_s, cal.gpu_bytes_per_s),
        Err(error) => {
            eprintln!(
                "  qwen4 D2 default: engine-rate calibration failed on device {}: {error:?}",
                gpu.device_id
            );
            None
        }
    }
}

/// Resolve the D2 default for one load and log it. Call once, before the MoE
/// layer runtimes are built (the layer admission reads the result), and while
/// no stage or scratch buffer competes for the calibration's pinned host
/// buffer. Returns whether the default requests D2.
pub(crate) fn resolve(
    gpu: &mut Gpu,
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    single_scope: bool,
) -> bool {
    let (layers, dtype) = host_split_layers(weights, config, single_scope);
    let unified = gpu.is_uma();
    let verdict = decide(
        hipfire_config::memory::offload_exec_explicit(),
        hipfire_config::memory::qwen4_d2_default(),
        hipfire_config::developer_var("HIPFIRE_MOE_CPU_EXPERTS")
            .ok()
            .as_deref()
            == Some("0"),
        layers,
        unified,
        || dtype.and_then(|dtype| measured_rates(gpu, config, dtype)),
    );
    if let Some(line) = verdict.describe() {
        eprintln!("  qwen4 D2 default ({layers} host split layers): {line}");
    }
    verdict.selects_d2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn rates(cpu: f64, link: f64) -> Option<HostDecodeRates> {
        HostDecodeRates::classify(cpu, link)
    }

    /// Run [`decide`] and report whether the calibration closure ran.
    fn run(
        explicit: Option<OffloadExec>,
        forced: Option<bool>,
        kill_switch: bool,
        layers: usize,
        unified: bool,
        measured: Option<HostDecodeRates>,
    ) -> (D2Default, bool) {
        let probed = Cell::new(false);
        let verdict = decide(explicit, forced, kill_switch, layers, unified, || {
            probed.set(true);
            measured
        });
        (verdict, probed.get())
    }

    #[test]
    fn explicit_setting_wins_both_ways_and_never_probes() {
        let fast_cpu = rates(2.0, 1.0);
        let fast_link = rates(1.0, 2.0);
        for mode in [OffloadExec::Pcie, OffloadExec::Cpu, OffloadExec::Passback] {
            for forced in [None, Some(true), Some(false)] {
                for measured in [fast_cpu, fast_link, None] {
                    let (verdict, probed) = run(Some(mode), forced, false, 40, false, measured);
                    assert_eq!(verdict, D2Default::Explicit(mode));
                    assert!(
                        !verdict.selects_d2(),
                        "{mode:?}: explicit is not the default"
                    );
                    assert!(!probed, "{mode:?} must not calibrate");
                }
            }
        }
    }

    #[test]
    fn unset_selects_d2_only_when_every_gate_passes_and_the_cpu_is_strictly_faster() {
        // (kill-switch, host split layers, unified, measured) -> (selects, probed)
        let cpu_wins = rates(2.0, 1.0);
        let tie = rates(1.0, 1.0);
        let link_wins = rates(1.0, 2.0);
        let cases = [
            // The only ON row.
            (false, 40, false, cpu_wins, true, true),
            // Each gate alone turns it off, and the calibration never runs.
            (true, 40, false, cpu_wins, false, false),
            (false, 0, false, cpu_wins, false, false),
            (false, 40, true, cpu_wins, false, false),
            // A measured loss, a tie and no measurement are all off.
            (false, 40, false, link_wins, false, true),
            (false, 40, false, tie, false, true),
            (false, 40, false, None, false, true),
        ];
        for (kill, layers, unified, measured, selects, probed) in cases {
            let (verdict, did_probe) = run(None, None, kill, layers, unified, measured);
            assert_eq!(
                verdict.selects_d2(),
                selects,
                "kill={kill} layers={layers} unified={unified} {measured:?}: {verdict:?}"
            );
            assert_eq!(did_probe, probed, "{verdict:?}");
        }
    }

    /// The typed developer key: `false` is never, `true` skips only the rate
    /// comparison (every other gate still applies) and never probes.
    #[test]
    fn typed_key_forces_or_vetoes_but_keeps_every_other_gate() {
        let loses = rates(1.0, 2.0);
        // false: off even where the CPU measurably wins.
        let (verdict, probed) = run(None, Some(false), false, 40, false, rates(2.0, 1.0));
        assert_eq!(verdict, D2Default::Disabled);
        assert!(!verdict.selects_d2() && !probed);
        // true: on even where the link wins, without calibrating.
        let (verdict, probed) = run(None, Some(true), false, 40, false, loses);
        assert_eq!(verdict, D2Default::Forced);
        assert!(verdict.selects_d2() && !probed);
        // true does not override the kill-switch, an empty layer set or UMA.
        assert_eq!(
            run(None, Some(true), true, 40, false, loses).0,
            D2Default::KillSwitch
        );
        assert_eq!(
            run(None, Some(true), false, 0, false, loses).0,
            D2Default::NoHostSplitLayers
        );
        assert_eq!(
            run(None, Some(true), false, 40, true, loses).0,
            D2Default::Unified
        );
        // Both vetoes are off; the typed key is the one reported.
        assert_eq!(
            run(None, Some(false), true, 40, false, loses).0,
            D2Default::Disabled
        );
    }

    #[test]
    fn verdict_variants_match_their_gate() {
        assert_eq!(
            run(None, None, true, 5, false, None).0,
            D2Default::KillSwitch
        );
        assert_eq!(
            run(None, None, false, 0, false, None).0,
            D2Default::NoHostSplitLayers
        );
        assert_eq!(run(None, None, false, 5, true, None).0, D2Default::Unified);
        assert_eq!(
            run(None, None, false, 5, false, None).0,
            D2Default::Unmeasured
        );
        let win = rates(3.0, 1.0).unwrap();
        assert_eq!(
            run(None, None, false, 5, false, Some(win)).0,
            D2Default::CpuFaster(win)
        );
        let loss = rates(1.0, 3.0).unwrap();
        assert_eq!(
            run(None, None, false, 5, false, Some(loss)).0,
            D2Default::LinkFaster(loss)
        );
    }

    /// One measurement serves exactly one `(device, dtype, k)`: another model's
    /// shape on the same device, or the same shape on another device, measures
    /// again, and a repeat of the same key reuses the stored value.
    #[test]
    fn rate_cache_key_never_crosses_device_or_shape() {
        let cache = Mutex::new(BTreeMap::new());
        let calls = Cell::new(0);
        let measure = |cpu: f64| {
            calls.set(calls.get() + 1);
            HostDecodeRates::classify(cpu, 1.0)
        };
        let a = cached_rates(&cache, (0, DType::MQ4G256V2, 2560), || measure(5.0));
        assert_eq!(calls.get(), 1);
        // Same key: cached, the closure does not run, the first value stays.
        assert_eq!(
            cached_rates(&cache, (0, DType::MQ4G256V2, 2560), || measure(9.0)),
            a
        );
        assert_eq!(calls.get(), 1);
        // Another k, another dtype and another device each measure afresh.
        let other_k = cached_rates(&cache, (0, DType::MQ4G256V2, 5120), || measure(7.0));
        let other_dtype = cached_rates(&cache, (0, DType::MQ6G256V2, 2560), || measure(8.0));
        let other_device = cached_rates(&cache, (1, DType::MQ4G256V2, 2560), || measure(6.0));
        assert_eq!(calls.get(), 4);
        assert_eq!(other_k.unwrap().cpu_bytes_per_s, 7.0);
        assert_eq!(other_dtype.unwrap().cpu_bytes_per_s, 8.0);
        assert_eq!(other_device.unwrap().cpu_bytes_per_s, 6.0);
        assert_eq!(a.unwrap().cpu_bytes_per_s, 5.0);
        // A failed measurement is cached too (no re-probe per layer).
        assert_eq!(
            cached_rates(&cache, (2, DType::MQ4G256V2, 2560), || None),
            None
        );
        assert_eq!(
            cached_rates(&cache, (2, DType::MQ4G256V2, 2560), || measure(1.0)),
            None
        );
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn unusable_figures_are_no_measurement() {
        for (cpu, link) in [
            (0.0, 1.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert_eq!(HostDecodeRates::classify(cpu, link), None, "{cpu} {link}");
        }
        assert!(HostDecodeRates::classify(1.0, 1.0).is_some());
    }

    /// The measured product fixture the selector must agree with in direction:
    /// the R9700 (gfx1201, Gen5 link) decodes at 34.45 tok/s on the CPU
    /// engine against 44.05 tok/s over PCIe, so a device whose CPU is the
    /// slower engine stays on PCIe. These are product decode rates quoted for
    /// the ordering, not the calibration's bytes/s.
    #[test]
    fn slower_cpu_engine_keeps_pcie_like_the_r9700_fixture() {
        let r9700 = HostDecodeRates::classify(34.45, 44.05).unwrap();
        assert!(!r9700.cpu_faster());
        let (verdict, _) = run(None, None, false, 40, false, Some(r9700));
        assert_eq!(verdict, D2Default::LinkFaster(r9700));
        assert!(!verdict.selects_d2());
    }

    /// Which layers count as host split layers: a tier layer needs a
    /// host-mapped expert; an un-tiered layer needs both aggregates
    /// host-mapped and the QT44/QT53 pair; a pruned layer never counts.
    #[test]
    fn host_split_layer_truth_table() {
        let tier = |any_host| LayerHostFacts {
            tier_any_host: Some(any_host),
            ..LayerHostFacts::default()
        };
        let aggregate = |host, qt| LayerHostFacts {
            aggregate_host: host,
            aggregate_qt44_qt53: qt,
            ..LayerHostFacts::default()
        };
        assert!(tier(true).qualifies());
        assert!(
            !tier(false).qualifies(),
            "an all-resident tier layer has nothing to do"
        );
        assert!(aggregate(true, true).qualifies());
        assert!(
            !aggregate(false, true).qualifies(),
            "a resident aggregate stays on the GPU"
        );
        assert!(
            !aggregate(true, false).qualifies(),
            "no CPU executor for the format pair"
        );
        // A tier layer ignores the aggregate fields (its allocations are the tier owners').
        assert!(
            tier(true).qualifies()
                && LayerHostFacts {
                    aggregate_host: false,
                    aggregate_qt44_qt53: false,
                    ..tier(true)
                }
                .qualifies()
        );
        for facts in [tier(true), aggregate(true, true)] {
            assert!(
                !LayerHostFacts {
                    pruned: true,
                    ..facts
                }
                .qualifies(),
                "pruned payloads are compact: the original-id owners cannot index them"
            );
        }
    }

    /// The default is a per-load `bool`: deciding it, even ON, does not change
    /// the process-wide host-exec mode, so D1, the step seam, the fusion
    /// selector and passback keep reading what they read before.
    #[test]
    fn selecting_d2_does_not_set_the_process_host_exec_mode() {
        if hipfire_config::memory::offload_exec_explicit().is_some() {
            return; // the user's explicit mode is the process mode; nothing to compare
        }
        let before = (
            hipfire_dispatch::cpu_exec_enabled(),
            hipfire_dispatch::passback_enabled(),
            hipfire_dispatch::moe_cpu_experts_enabled(),
        );
        let (verdict, _) = run(None, None, false, 40, false, rates(2.0, 1.0));
        assert!(verdict.selects_d2());
        let (forced, _) = run(None, Some(true), false, 40, false, None);
        assert!(forced.selects_d2());
        let after = (
            hipfire_dispatch::cpu_exec_enabled(),
            hipfire_dispatch::passback_enabled(),
            hipfire_dispatch::moe_cpu_experts_enabled(),
        );
        assert_eq!(before, after);
        assert_eq!(before, (false, false, false), "unset mode is pcie");
        assert_eq!(hipfire_config::memory::offload_exec(), OffloadExec::Pcie);
    }
}
