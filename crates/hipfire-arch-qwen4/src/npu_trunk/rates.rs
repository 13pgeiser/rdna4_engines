// SPDX-License-Identifier: Apache-2.0
//! Live rate scales for the `auto` NPU trunk plan. The built-in calibration is the prior; measured pairs
//! learn a GPU-window scale and an NPU-chain scale that multiply the prior's terms (`scaled_calibration`).
//! Forced modes (`on`, `emulate`) never use the estimator: they plan from the static calibration only.
use super::calibration::FAMILY_NAMES;
use super::plan::{bank_segments_moved, contention_extra, gpu_chunk, pair_time_moved, plan_executable, subset_mask};
use super::{Calibration, FamilyMask, Moved};

/// Weight of a new sample in the scale EWMAs.
pub const EWMA_ALPHA: f64 = 0.25;
/// Raw ratios and scales are clamped to `SCALE_MIN..=SCALE_MAX`.
pub const SCALE_MIN: f64 = 0.5;
pub const SCALE_MAX: f64 = 2.0;
/// Samples whose modelled window or NPU time is below this many ms are ignored.
pub const MIN_MODEL_MS: f64 = 50.0;
/// Relative EWMA/snapshot difference above which an unfrozen snapshot takes the EWMA values.
pub const APPLY_EPS: f64 = 0.01;
/// Valid samples needed before the snapshot may freeze.
pub const FREEZE_AFTER: u32 = 6;
/// Consecutive valid samples that must each move both EWMAs by less than `FREEZE_EPS` (relative) to freeze.
pub const FREEZE_WINDOW: u32 = 4;
pub const FREEZE_EPS: f64 = 0.01;
/// A frozen snapshot reopens after `REOPEN_CONFIRM` consecutive valid samples whose raw ratio deviates from the
/// frozen scale by more than `REOPEN_EPS` (relative) in the same direction.
pub const REOPEN_EPS: f64 = 0.05;
pub const REOPEN_CONFIRM: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind { Gdn, Qsa }
impl LayerKind {
    fn index(self) -> usize { match self { LayerKind::Gdn => 0, LayerKind::Qsa => 1 } }
}

/// Modelled times of one executed pair, comparable with the measured NPU chain and GPU window sums.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairModel { pub npu_ms: f64, pub window_ms: f64, pub gpu_only_ms: f64, pub savings_ms: f64 }

/// Model of one executed pair. `kinds[l]` is the kind of layer `l`; `rows`/`prefix` describe banks A, B.
/// `npu_ms`: the sum of the in-proj NPU chain (`npu[k]` of `bank_segments_moved`) over every (layer, bank) whose
/// kind has a nonzero chain under `moved`. `window_ms`: the sum over the same chains except (0, A) (its window is
/// bank-B initialization) of the GPU window hiding the chain, plus its contention: chain (l, A), l >= 1, is hidden
/// by bank B's segment of layer l-1; chain (l, B) by bank A's segment of layer l.
/// `gpu_only_ms`: both banks run GPU-only. `savings_ms`: what pairing saves the GPU before any NPU wait: the moved
/// families' GPU projection time minus the pair's GPU glue (pack, consume, visit) and its DRAM contention. The pair
/// loses to GPU-only once the GPU's measured NPU waits exceed it; unlike `gpu_only_ms` it carries no prefix-dependent
/// term, so the model's error at depth does not move it.
pub fn pair_model(cal: &Calibration, moved: Moved, rows: [usize; 2], prefix: [usize; 2], kinds: &[LayerKind]) -> PairModel {
    let a = bank_segments_moved(cal, moved, rows[0], prefix[0]);
    let b = bank_segments_moved(cal, moved, rows[1], prefix[1]);
    let (mut npu_ms, mut window_ms) = (0.0, 0.0);
    for (l, kind) in kinds.iter().enumerate() {
        let k = kind.index();
        if a.npu[k] != 0.0 {
            npu_ms += a.npu[k];
            if l >= 1 {
                let window = b.gpu[kinds[l - 1].index()];
                window_ms += window + contention_extra(cal, window, a.npu[k], a.bytes[k]);
            }
        }
        if b.npu[k] != 0.0 {
            npu_ms += b.npu[k];
            let window = a.gpu[k];
            window_ms += window + contention_extra(cal, window, b.npu[k], b.bytes[k]);
        }
    }
    let mut savings_ms = -pair_time_moved(cal, moved, rows[0], prefix[0], rows[1], prefix[1]).extra;
    for (s, (r, p)) in [&a, &b].into_iter().zip([(rows[0], prefix[0]), (rows[1], prefix[1])]) {
        let chunk = gpu_chunk(cal, r, p);
        let moved_trunk: f64 = (0..4).filter(|&i| moved.inproj[i]).map(|i| chunk.trunk[i]).sum();
        let segments: f64 = kinds.iter().map(|kind| s.gpu[kind.index()]).sum();
        let glue = segments - (chunk.total - cal.gpu.fixed_per_chunk_ms - moved_trunk);
        savings_ms += moved_trunk - glue;
    }
    PairModel { npu_ms, window_ms, savings_ms,
        gpu_only_ms: gpu_chunk(cal, rows[0], prefix[0]).total + gpu_chunk(cal, rows[1], prefix[1]).total }
}

/// One measured pair against its model: summed NPU chain time and summed GPU window time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateSample { pub npu_measured_ms: f64, pub npu_model_ms: f64, pub window_measured_ms: f64, pub window_model_ms: f64 }

/// The estimator's current calibration (`scaled_calibration(prior, gpu_scale, npu_scale)`). `generation` advances
/// whenever it changes; `frozen` means it is fixed until a reopen.
#[derive(Clone, Debug)]
pub struct RateSnapshot { pub calibration: Calibration, pub generation: u64, pub gpu_scale: f64, pub npu_scale: f64, pub samples: u32, pub frozen: bool }

/// Prior calibration scaled by learned GPU-window and NPU-chain rates (EWMA of measured/model, clamped), frozen once
/// settled. Owned by one process; the forced modes never create one.
pub struct RateEstimator {
    prior: Calibration,
    snapshot: RateSnapshot,
    ewma: Option<(f64, f64)>,
    /// Consecutive valid samples that moved both EWMAs by less than `FREEZE_EPS`.
    calm: u32,
    /// Frozen only: direction of the current run of deviating samples and its length.
    deviating: Option<((i8, i8), u32)>,
}

fn relative(value: f64, base: f64) -> f64 { (value - base).abs() / base }

impl RateEstimator {
    pub fn new(prior: Calibration) -> Self {
        let snapshot = RateSnapshot { calibration: prior.clone(), generation: 0, gpu_scale: 1.0, npu_scale: 1.0, samples: 0, frozen: false };
        Self { prior, snapshot, ewma: None, calm: 0, deviating: None }
    }
    pub fn prior(&self) -> &Calibration { &self.prior }
    pub fn snapshot(&self) -> &RateSnapshot { &self.snapshot }

    fn publish(&mut self, gpu_scale: f64, npu_scale: f64) {
        self.snapshot.calibration = scaled_calibration(&self.prior, gpu_scale, npu_scale);
        self.snapshot.gpu_scale = gpu_scale;
        self.snapshot.npu_scale = npu_scale;
        self.snapshot.generation += 1;
    }

    /// Feeds one measured pair. Invalid samples (non-finite, model below `MIN_MODEL_MS`, measured not positive)
    /// are ignored. Returns true iff the snapshot changed (its generation advanced).
    pub fn observe(&mut self, sample: &RateSample) -> bool {
        let values = [sample.npu_measured_ms, sample.npu_model_ms, sample.window_measured_ms, sample.window_model_ms];
        if !values.iter().all(|v| v.is_finite()) || sample.npu_model_ms < MIN_MODEL_MS || sample.window_model_ms < MIN_MODEL_MS
            || sample.npu_measured_ms <= 0.0 || sample.window_measured_ms <= 0.0 { return false; }
        let raw_gpu = (sample.window_measured_ms / sample.window_model_ms).clamp(SCALE_MIN, SCALE_MAX);
        let raw_npu = (sample.npu_measured_ms / sample.npu_model_ms).clamp(SCALE_MIN, SCALE_MAX);
        self.snapshot.samples = self.snapshot.samples.saturating_add(1);
        let (gpu, npu) = match self.ewma {
            None => { self.calm = 0; (raw_gpu, raw_npu) }
            Some((g, n)) => {
                let next = (g + EWMA_ALPHA * (raw_gpu - g), n + EWMA_ALPHA * (raw_npu - n));
                self.calm = if relative(next.0, g) < FREEZE_EPS && relative(next.1, n) < FREEZE_EPS { self.calm + 1 } else { 0 };
                next
            }
        };
        self.ewma = Some((gpu, npu));
        if self.snapshot.frozen {
            let direction = |raw: f64, base: f64| -> i8 {
                let d = (raw - base) / base;
                if d > REOPEN_EPS { 1 } else if d < -REOPEN_EPS { -1 } else { 0 }
            };
            let dir = (direction(raw_gpu, self.snapshot.gpu_scale), direction(raw_npu, self.snapshot.npu_scale));
            if dir == (0, 0) {
                self.deviating = None;
                return false;
            }
            let run = match self.deviating { Some((d, n)) if d == dir => n + 1, _ => 1 };
            self.deviating = Some((dir, run));
            if run < REOPEN_CONFIRM { return false; }
            self.snapshot.frozen = false;
            self.deviating = None;
            self.calm = 0;
            self.publish(gpu, npu);
            return true;
        }
        let changed = relative(gpu, self.snapshot.gpu_scale) > APPLY_EPS || relative(npu, self.snapshot.npu_scale) > APPLY_EPS;
        if changed { self.publish(gpu, npu); }
        if self.snapshot.samples >= FREEZE_AFTER && self.calm >= FREEZE_WINDOW {
            self.snapshot.frozen = true;
            self.deviating = None;
        }
        changed
    }
}

/// `prior` with its GPU terms multiplied by `gpu_scale` and its NPU terms by `npu_scale` (rates divide).
/// GPU: every `gpu.*` ms term (not `min_bank_rows`), glue `pack_ms_8192`, `visit_ms_8192` and
/// `consume_ms_per_col_8192`, out-family `gpu_ms_per_8192` and `qsa_order`, all x `gpu_scale`.
/// NPU: `npu.tops_corun`, `npu.tops_alone` and out-family `tops_alone` / `npu_scale`; `glue.sync_ms` x `npu_scale`.
/// Nothing else changes; scales of 1.0 reproduce `prior` exactly.
pub fn scaled_calibration(prior: &Calibration, gpu_scale: f64, npu_scale: f64) -> Calibration {
    let mut cal = prior.clone();
    let g = &mut cal.gpu;
    for v in g.trunk_in_ms_per_8192.values_mut() { *v *= gpu_scale; }
    for v in [&mut g.uniform_per_layer_ms, &mut g.gdn_core_per_layer_ms, &mut g.qsa_per_layer_ms,
              &mut g.qsa_growth_per_layer_ms_per_8192_prefix, &mut g.fixed_per_chunk_ms, &mut g.boundary_tax_ms] { *v *= gpu_scale; }
    cal.glue.pack_ms_8192 *= gpu_scale;
    cal.glue.visit_ms_8192 *= gpu_scale;
    for v in cal.glue.consume_ms_per_col_8192.values_mut() { *v *= gpu_scale; }
    cal.glue.sync_ms *= npu_scale;
    cal.npu.tops_corun /= npu_scale;
    if let Some(t) = &mut cal.npu.tops_alone { *t /= npu_scale; }
    for out in cal.families.values_mut().filter_map(|f| f.out.as_mut()) {
        out.gpu_ms_per_8192 *= gpu_scale;
        out.tops_alone /= npu_scale;
    }
    if let Some(q) = &mut cal.qsa_order {
        for v in [&mut q.core_a_ms, &mut q.core_b_ms, &mut q.post_ms, &mut q.inproj_chain_ms] { *v *= gpu_scale; }
    }
    cal
}

/// Fastest group subset of `allowed` for a `rows`-row request: the minimum `plan_executable(cal, subset, rows).t_pipe_ms` over
/// the 15 `select_families` subsets that lie within `allowed` and whose families are `selectable`. Ties go to fewer
/// families, then `select_families`' name order. `FamilyMask::NONE` when no subset's `t_pipe_ms` is below its `t_gpu_ms`.
pub fn choose_families(cal: &Calibration, allowed: FamilyMask, rows: usize) -> FamilyMask {
    let names = |bits: u8| -> Vec<&'static str> {
        ["kvidx", "q", "qkv", "zab"].into_iter()
            .filter(|n| FAMILY_NAMES.iter().position(|f| f == n).is_some_and(|i| bits & (1 << i) != 0)).collect()
    };
    let mut best: Option<(f64, usize, Vec<&'static str>, FamilyMask)> = None;
    for bits in 1u8..16 {
        let subset = subset_mask(bits);
        if subset.bits() & !allowed.bits() != 0 { continue; }
        if !FAMILY_NAMES.iter().enumerate().all(|(i, name)| bits & (1 << i) == 0 || cal.families[*name].selectable) { continue; }
        let p = plan_executable(cal, subset, rows);
        if p.t_pipe_ms >= p.t_gpu_ms { continue; }
        let key = (p.t_pipe_ms, bits.count_ones() as usize, names(bits));
        let better = best.as_ref().is_none_or(|(t, count, order, _)| {
            key.0.total_cmp(t).then_with(|| key.1.cmp(count)).then_with(|| key.2.cmp(order)).is_lt()
        });
        if better { best = Some((key.0, key.1, key.2, subset)); }
    }
    best.map_or(FamilyMask::NONE, |(.., subset)| subset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Family;
    use super::super::plan::{executable_plan_for, plan, Step};

    const ROUTE: &str = "overlay:gfx1151-zab-v1";
    fn builtin() -> Calibration {
        let (cal, source) = Calibration::load_with_builtin("/nonexistent/x.json", ROUTE).unwrap();
        assert_eq!(source, "builtin");
        cal
    }
    fn flash_next_kinds() -> Vec<LayerKind> {
        let kinds: Vec<_> = (0..48).map(|l| if l % 4 == 3 { LayerKind::Qsa } else { LayerKind::Gdn }).collect();
        assert_eq!(kinds.iter().filter(|k| **k == LayerKind::Gdn).count(), 36);
        assert_eq!(kinds.iter().filter(|k| **k == LayerKind::Qsa).count(), 12);
        kinds
    }
    fn json(cal: &Calibration) -> serde_json::Value { serde_json::to_value(cal).unwrap() }
    fn sample(window_ratio: f64, npu_ratio: f64) -> RateSample {
        RateSample { npu_measured_ms: 100.0 * npu_ratio, npu_model_ms: 100.0, window_measured_ms: 100.0 * window_ratio, window_model_ms: 100.0 }
    }

    #[test]
    fn new_snapshot_is_the_prior() {
        let prior = builtin();
        let e = RateEstimator::new(prior.clone());
        let s = e.snapshot();
        assert_eq!(json(&s.calibration), json(&prior));
        assert_eq!(json(e.prior()), json(&prior));
        assert_eq!((s.generation, s.gpu_scale, s.npu_scale, s.samples, s.frozen), (0, 1.0, 1.0, 0, false));
    }

    #[test]
    fn unit_scales_reproduce_the_prior_and_scaled_output_validates() {
        for route in [ROUTE, "stock"] {
            let (prior, _) = Calibration::load_with_builtin("/nonexistent/x.json", route).unwrap();
            assert_eq!(json(&scaled_calibration(&prior, 1.0, 1.0)), json(&prior));
            for (g, n) in [(0.5, 2.0), (2.0, 0.5), (1.3, 0.8), (SCALE_MIN, SCALE_MIN), (SCALE_MAX, SCALE_MAX)] {
                let scaled = scaled_calibration(&prior, g, n);
                scaled.validate().unwrap();
                assert_eq!(scaled.key, prior.key);
                assert_eq!(scaled.margin, prior.margin);
            }
        }
        let prior = builtin();
        let scaled = scaled_calibration(&prior, 2.0, 0.5);
        assert_eq!(scaled.gpu.uniform_per_layer_ms, prior.gpu.uniform_per_layer_ms * 2.0);
        assert_eq!(scaled.gpu.min_bank_rows, prior.gpu.min_bank_rows);
        assert_eq!(scaled.glue.sync_ms, prior.glue.sync_ms * 0.5);
        assert_eq!(scaled.npu.tops_corun, prior.npu.tops_corun * 2.0);
        assert_eq!(scaled.npu.bytes_per_kop, prior.npu.bytes_per_kop);
        assert_eq!(scaled.contention.alpha_overlap, prior.contention.alpha_overlap);
    }

    #[test]
    fn pair_model_matches_plan_and_segments() {
        let cal = builtin();
        let kinds = flash_next_kinds();
        let model = pair_model(&cal, Moved::from(FamilyMask::ALL), [8192, 8192], [0, 8192], &kinds);
        let p = plan(&cal, FamilyMask::ALL, 16384);
        assert!((model.gpu_only_ms - p.t_gpu_ms).abs() < 1e-6, "{} vs {}", model.gpu_only_ms, p.t_gpu_ms);
        let a = bank_segments_moved(&cal, Moved::from(FamilyMask::ALL), 8192, 0);
        let b = bank_segments_moved(&cal, Moved::from(FamilyMask::ALL), 8192, 8192);
        assert!(a.npu[0] > 0.0 && a.npu[1] > 0.0);
        let expected_npu = 36.0 * (a.npu[0] + b.npu[0]) + 12.0 * (a.npu[1] + b.npu[1]);
        assert!((model.npu_ms - expected_npu).abs() < 1e-6, "{} vs {expected_npu}", model.npu_ms);
        // Window: every chain but (0, A); (l, A) l >= 1 is hidden by bank B of layer l-1, (l, B) by bank A of layer l.
        let mut expected_window = 0.0;
        for (l, kind) in kinds.iter().enumerate() {
            let k = kind.index();
            if l >= 1 {
                let w = b.gpu[kinds[l - 1].index()];
                expected_window += w + contention_extra(&cal, w, a.npu[k], a.bytes[k]);
            }
            let w = a.gpu[k];
            expected_window += w + contention_extra(&cal, w, b.npu[k], b.bytes[k]);
        }
        assert!((model.window_ms - expected_window).abs() < 1e-6, "{} vs {expected_window}", model.window_ms);
        assert!(model.window_ms > 0.0);
        // Nothing moved: no chains, no window.
        let none = pair_model(&cal, Moved::from(FamilyMask::NONE), [8192, 8192], [0, 8192], &kinds);
        assert_eq!((none.npu_ms, none.window_ms), (0.0, 0.0));
        assert!((none.gpu_only_ms - p.t_gpu_ms).abs() < 1e-6);
        assert!(none.savings_ms.abs() < 1e-9);
        // savings is exactly what the planner's pair time credits the pipe: t = gpu_only - savings + stall + fill.
        let pr = pair_time_moved(&cal, Moved::from(FamilyMask::ALL), 8192, 0, 8192, 8192);
        assert!((pr.t - (model.gpu_only_ms - model.savings_ms + pr.stall + pr.fill)).abs() < 1e-6, "{pr:?} {model:?}");
        // At the built-in rates the modelled NPU waits (stall + fill) stay far below the saving: no guard trip.
        assert!(pr.stall + pr.fill < 0.5 * model.savings_ms, "{pr:?} {model:?}");
        // Only GDN in-proj moved: QSA layers contribute no chain.
        let qkv = pair_model(&cal, Moved::from(FamilyMask::from(Family::GdnQkv)), [8192, 8192], [0, 8192], &kinds);
        let qa = bank_segments_moved(&cal, Moved::from(FamilyMask::from(Family::GdnQkv)), 8192, 0);
        assert_eq!(qa.npu[1], 0.0);
        assert!(qkv.npu_ms > 0.0 && qkv.npu_ms < model.npu_ms);
    }

    #[test]
    fn constant_ratios_converge_and_freeze() {
        let mut e = RateEstimator::new(builtin());
        assert!(e.observe(&sample(1.3, 0.8)));
        assert_eq!(e.snapshot().generation, 1);
        for _ in 0..12 { e.observe(&sample(1.3, 0.8)); }
        let s = e.snapshot();
        assert!((s.gpu_scale / 1.3 - 1.0).abs() < 0.01 && (s.npu_scale / 0.8 - 1.0).abs() < 0.01, "{s:?}");
        assert!(s.frozen);
        assert_eq!(s.samples, 13);
        assert_eq!(json(&s.calibration), json(&scaled_calibration(e.prior(), s.gpu_scale, s.npu_scale)));
        let generation = s.generation;
        // A frozen snapshot does not move on samples inside the reopen band.
        assert!(!e.observe(&sample(1.33, 0.8)));
        assert_eq!(e.snapshot().generation, generation);
    }

    #[test]
    fn npu_thirty_percent_slower_than_the_prior_converges_and_drops_zab() {
        // A stock-kernel host whose NPU leg runs x1.3 the built-in model at depth (fivetide's 57K), after short
        // requests that matched the prior: from fresh, and from a snapshot already frozen at the prior.
        let zab = |m: FamilyMask| m.contains(Family::GdnZ);
        for frozen_first in [false, true] {
            let mut e = RateEstimator::new(builtin());
            for _ in 0..if frozen_first { FREEZE_AFTER } else { 2 } { e.observe(&sample(1.0, 1.0)); }
            assert_eq!(e.snapshot().frozen, frozen_first);
            for _ in 0..20 { e.observe(&sample(1.0, 1.3)); }
            let s = e.snapshot();
            assert!((s.npu_scale / 1.3 - 1.0).abs() < 0.02 && (s.gpu_scale - 1.0).abs() < 0.01, "{s:?}");
            let chosen = choose_families(&s.calibration, FamilyMask::ALL, 57_552);
            assert!(!zab(chosen) && chosen.contains(Family::GdnQkv) && chosen.contains(Family::QsaQ), "{chosen:?}");
            let p = plan_executable(&s.calibration, chosen, 57_552);
            assert!(p.t_pipe_ms < p.t_gpu_ms);
        }
    }

    #[test]
    fn unit_ratios_freeze_without_changing_the_snapshot() {
        let mut e = RateEstimator::new(builtin());
        for i in 1..=FREEZE_AFTER {
            assert!(!e.observe(&sample(1.0, 1.0)));
            assert_eq!(e.snapshot().frozen, i >= FREEZE_AFTER);
        }
        assert_eq!(e.snapshot().generation, 0);
    }

    #[test]
    fn ratios_are_clamped() {
        let mut e = RateEstimator::new(builtin());
        assert!(e.observe(&sample(10.0, 0.01)));
        assert_eq!((e.snapshot().gpu_scale, e.snapshot().npu_scale), (SCALE_MAX, SCALE_MIN));
        e.observe(&sample(0.01, 10.0));
        assert!(e.snapshot().gpu_scale <= SCALE_MAX && e.snapshot().gpu_scale >= SCALE_MIN);
        assert!(e.snapshot().npu_scale <= SCALE_MAX && e.snapshot().npu_scale >= SCALE_MIN);
        e.snapshot().calibration.validate().unwrap();
    }

    #[test]
    fn invalid_samples_are_ignored() {
        let mut e = RateEstimator::new(builtin());
        let good = sample(1.3, 0.8);
        let bad = [
            RateSample { npu_measured_ms: f64::NAN, ..good },
            RateSample { npu_model_ms: f64::INFINITY, ..good },
            RateSample { window_measured_ms: f64::NEG_INFINITY, ..good },
            RateSample { window_model_ms: f64::NAN, ..good },
            RateSample { npu_model_ms: MIN_MODEL_MS - 0.1, ..good },
            RateSample { window_model_ms: 0.0, ..good },
            RateSample { npu_measured_ms: 0.0, ..good },
            RateSample { window_measured_ms: -5.0, ..good },
        ];
        for s in bad {
            assert!(!e.observe(&s), "{s:?}");
        }
        let snap = e.snapshot();
        assert_eq!((snap.samples, snap.generation, snap.frozen, snap.gpu_scale, snap.npu_scale), (0, 0, false, 1.0, 1.0));
        assert!(e.observe(&RateSample { npu_model_ms: MIN_MODEL_MS, window_model_ms: MIN_MODEL_MS, npu_measured_ms: 40.0, window_measured_ms: 60.0 }));
        assert_eq!(e.snapshot().samples, 1);
    }

    #[test]
    fn frozen_snapshot_reopens_after_consistent_deviations() {
        let mut e = RateEstimator::new(builtin());
        for _ in 0..10 { e.observe(&sample(1.3, 0.8)); }
        assert!(e.snapshot().frozen);
        let generation = e.snapshot().generation;
        // Alternating directions never confirm.
        for i in 0..6 {
            assert!(!e.observe(&sample(if i % 2 == 0 { 1.7 } else { 0.9 }, 0.8)));
        }
        assert!(e.snapshot().frozen);
        assert!(!e.observe(&sample(1.3, 0.8)));
        for _ in 0..REOPEN_CONFIRM - 1 { assert!(!e.observe(&sample(1.7, 0.8))); }
        assert!(e.snapshot().frozen && e.snapshot().generation == generation);
        assert!(e.observe(&sample(1.7, 0.8)));
        let s = e.snapshot();
        assert!(!s.frozen);
        assert_eq!(s.generation, generation + 1);
        assert!(s.gpu_scale > 1.3);
        s.calibration.validate().unwrap();
        // Unfrozen again, it follows the new rate and settles.
        for _ in 0..40 { e.observe(&sample(1.7, 0.8)); }
        let s = e.snapshot();
        // Freezing is judged on per-sample EWMA movement, so the settled scale sits within a few percent of the rate.
        assert!((s.gpu_scale / 1.7 - 1.0).abs() < 0.05 && s.frozen, "{s:?}");
    }

    #[test]
    fn slow_npu_drops_zab_and_nominal_rates_keep_all() {
        let prior = builtin();
        let all = FamilyMask::ALL;
        assert_eq!(choose_families(&prior, all, 16384), all);
        let slow = scaled_calibration(&prior, 1.0, 1.25);
        let chosen = choose_families(&slow, all, 16384);
        for f in [Family::GdnZ, Family::GdnA, Family::GdnB] { assert!(!chosen.contains(f), "{chosen:?}"); }
        for f in [Family::GdnQkv, Family::QsaQ, Family::QsaK, Family::QsaV, Family::QsaIdx] { assert!(chosen.contains(f), "{chosen:?}"); }
        // Restricted to a subset, the choice stays inside it; nothing allowed -> NONE.
        let qsa_only = choose_families(&prior, FamilyMask::QSA, 16384);
        assert_eq!(qsa_only.bits() & !FamilyMask::QSA.bits(), 0);
        assert_eq!(choose_families(&prior, FamilyMask::NONE, 16384), FamilyMask::NONE);
    }

    #[test]
    fn scaled_requests_plan_executable_pairs_where_the_rebanked_model_wins() {
        // Halo gate, cand-auto r1: after one 16K pair the scales were gpu x0.981 / npu x0.892, and the full model
        // preferred the re-banked plan for 57,552 rows, which the executor projects to GPU chunks: the request ran
        // GPU-only. Request planning uses `plan_executable`, which keeps the natural and rebalanced pairs.
        let cal = scaled_calibration(&builtin(), 0.981, 0.892);
        let rows = 57_552;
        let pipes = |p: &super::super::Plan| p.steps.iter().filter(|s| matches!(s, Step::Pipe { .. })).count();
        let full = executable_plan_for(&cal, FamilyMask::ALL, &plan(&cal, FamilyMask::ALL, rows), rows, Some(0));
        assert_eq!(pipes(&full), 0, "the re-banked model plan has no executable pair");
        let natural = plan_executable(&cal, FamilyMask::ALL, rows);
        let executable = executable_plan_for(&cal, FamilyMask::ALL, &natural, rows, Some(0));
        assert_eq!(pipes(&executable), 4);
        assert!(executable.t_pipe_ms < executable.t_gpu_ms);
        // The unscaled prior plans the same pairs either way.
        let prior = builtin();
        assert_eq!(plan(&prior, FamilyMask::ALL, rows), plan_executable(&prior, FamilyMask::ALL, rows));
    }
}
