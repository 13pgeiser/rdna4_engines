// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! ABBA hipEvent timing core shared by `pm_decode_twins time` and `pm_loop`:
//! the allocation-free ABBA schedule, sample statistics and the separability
//! rule. `Incumbent` is arm A (the shipped/baseline kernel), `Candidate` is
//! arm B. Included with `#[path = "support/abba.rs"] mod abba;`.
#![allow(dead_code)]

use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn positive_count(s: &str) -> Result<usize> {
    let n = s.parse::<usize>().map_err(err)?;
    if n == 0 {
        return Err("counts must be positive".into());
    }
    Ok(n)
}

pub fn timing_counts(n: usize, rounds: usize) -> Result<(usize, usize)> {
    if n == 0 || rounds == 0 {
        return Err("counts must be positive".into());
    }
    let per_arm = n
        .checked_mul(rounds)
        .and_then(|v| v.checked_mul(2))
        .ok_or("timing call count overflow")?;
    let total = per_arm.checked_mul(2).ok_or("timing call count overflow")?;
    per_arm
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("sample storage overflow")?;
    Ok((per_arm, total))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingArm {
    Incumbent,
    Candidate,
}

/// Pure, allocation-free ABBA iterator; construction checks all count arithmetic.
pub fn timing_schedule(n: usize, rounds: usize) -> Result<impl Iterator<Item = TimingArm>> {
    let (_, total) = timing_counts(n, rounds)?;
    Ok((0..total).map(move |i| match (i / n) % 4 {
        0 | 3 => TimingArm::Incumbent,
        _ => TimingArm::Candidate,
    }))
}

#[derive(Debug)]
pub struct TimingStats {
    pub count: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p90: f64,
    pub mean: f64,
}

impl TimingStats {
    pub fn spread(&self) -> f64 {
        self.max - self.min
    }
    pub fn json(&self) -> Value {
        json!({"call_count": self.count, "median_ns": self.median, "min_ns": self.min,
            "max_ns": self.max, "p90_ns": self.p90, "mean_ns": self.mean,
            "spread_ns": self.spread()})
    }
}

pub fn timing_stats(samples: &[f64]) -> Result<TimingStats> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("timing samples must be nonempty, positive and finite".into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n.is_multiple_of(2) {
        sorted[n / 2 - 1] / 2.0 + sorted[n / 2] / 2.0
    } else {
        sorted[n / 2]
    };
    // ceil(0.9*n), without overflowing integer multiplication.
    let rank = n - n / 10;
    let mean = sorted.iter().map(|v| v / n as f64).sum::<f64>();
    if !mean.is_finite() {
        return Err("nonfinite timing mean".into());
    }
    Ok(TimingStats {
        count: n,
        median,
        min: sorted[0],
        max: sorted[n - 1],
        p90: sorted[rank - 1],
        mean,
    })
}

pub fn timing_relation(a: &TimingStats, b: &TimingStats) -> (f64, f64, f64, bool) {
    let spread = a.spread().max(b.spread());
    let delta = b.median - a.median;
    (
        b.median / a.median,
        spread,
        delta,
        delta.abs() > spread && (a.max < b.min || b.max < a.min),
    )
}

// ---------------------------------------------------------------------------
// Block-median rule (pm_loop). The legacy rule above (`timing_relation`, kept
// unchanged for `pm_decode_twins`) compares full sample ranges, so one hiccup
// launch (a 7 ms outlier among 300 us samples) vetoes separability and more
// samples only make that worse. This rule works on per-block medians instead.

/// Medians of consecutive `n`-launch blocks, in schedule order. Each arm's
/// samples are pushed in schedule order, so every consecutive `n` samples are
/// exactly one ABBA block of that arm.
pub fn block_medians(samples: &[f64], n: usize) -> Result<Vec<f64>> {
    if n == 0 || samples.is_empty() || samples.len() % n != 0 {
        return Err(format!(
            "{} samples are not a whole number of {n}-launch blocks",
            samples.len()
        ));
    }
    samples
        .chunks_exact(n)
        .map(|c| timing_stats(c).map(|s| s.median))
        .collect()
}

fn min_max(v: &[f64]) -> (f64, f64) {
    (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Exact two-sided sign-test p-value of `k` "candidate faster" rounds out of
/// `n` under the null that either arm is equally likely to win a round.
pub fn sign_test_p(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let tail = k.min(n - k);
    // P(X <= tail), X ~ Binomial(n, 1/2), built from C(n, i) / 2^n iteratively in f64.
    let mut term = 0.5f64.powi(n as i32);
    let mut sum = term;
    for i in 0..tail {
        term *= (n - i) as f64 / (i + 1) as f64;
        sum += term;
    }
    (2.0 * sum).min(1.0)
}

/// Outcome of [`block_rule`]. Ratios are candidate / baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockRule {
    /// Every candidate block median is below every baseline block median, or
    /// every one is above.
    pub separable: bool,
    /// `median(candidate block medians) / median(baseline block medians)`.
    pub block_ratio: f64,
    /// Per ABBA round, the candidate round median over the baseline round
    /// median (a round is the 2n launches each arm ran in one ABBA cycle);
    /// min and max over rounds.
    pub round_ratio_min: f64,
    pub round_ratio_max: f64,
    /// Rounds in which the candidate round median beat the baseline's (ratio < 1).
    /// Informational only: it does not enter `separable`.
    pub rounds_candidate_faster: usize,
    pub baseline_block_medians: Vec<f64>,
    pub candidate_block_medians: Vec<f64>,
}

/// Block-median separability of two arms' samples (nanoseconds) taken with
/// `n`-launch ABBA blocks: whole rounds only (`2n` samples per arm per round).
pub fn block_rule(base: &[f64], cand: &[f64], n: usize) -> Result<BlockRule> {
    if base.len() != cand.len() || base.len() % (2 * n.max(1)) != 0 {
        return Err("block rule needs equal whole ABBA rounds of both arms".into());
    }
    let bm = block_medians(base, n)?;
    let cm = block_medians(cand, n)?;
    let med = |v: &[f64]| timing_stats(v).map(|s| s.median);
    let (lo, hi) = min_max(&bm);
    let (clo, chi) = min_max(&cm);
    let separable = chi < lo || clo > hi;
    let mut rr = Vec::new();
    for (b, c) in base.chunks_exact(2 * n).zip(cand.chunks_exact(2 * n)) {
        rr.push(med(c)? / med(b)?);
    }
    Ok(BlockRule {
        separable,
        block_ratio: med(&cm)? / med(&bm)?,
        round_ratio_min: min_max(&rr).0,
        round_ratio_max: min_max(&rr).1,
        rounds_candidate_faster: rr.iter().filter(|r| **r < 1.0).count(),
        baseline_block_medians: bm,
        candidate_block_medians: cm,
    })
}

/// Why auto-extension stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtendStop {
    Separable,
    MaxRounds,
    TimeBudget,
}

impl ExtendStop {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Separable => "separable",
            Self::MaxRounds => "max_rounds",
            Self::TimeBudget => "time_budget",
        }
    }
}

/// How many more ABBA rounds to run after `done` rounds took `elapsed_s`
/// seconds: `Ok(extra)` doubles the run (`extra == done`) capped by
/// `max_rounds` and by what still fits the budget at the observed per-round
/// time; `Err(why)` stops. `extra` is always at least 1.
pub fn extend_rounds(
    done: usize,
    max_rounds: usize,
    elapsed_s: f64,
    budget_s: f64,
    separable: bool,
) -> std::result::Result<usize, ExtendStop> {
    if separable {
        return Err(ExtendStop::Separable);
    }
    if done >= max_rounds {
        return Err(ExtendStop::MaxRounds);
    }
    let per_round = elapsed_s / done.max(1) as f64;
    let fit = if per_round > 0.0 {
        ((budget_s - elapsed_s) / per_round).floor()
    } else {
        f64::INFINITY
    };
    if fit < 1.0 {
        return Err(ExtendStop::TimeBudget);
    }
    let want = done.min(max_rounds - done);
    Ok(want.min(fit as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_abba_two_rounds() {
        use TimingArm::{Candidate as B, Incumbent as A};
        let got: Vec<_> = timing_schedule(2, 2).unwrap().collect();
        assert_eq!(got, vec![A, A, B, B, B, B, A, A, A, A, B, B, B, B, A, A]);
        assert_eq!(got.iter().filter(|&&v| v == A).count(), 8);
        assert_eq!(got.iter().filter(|&&v| v == B).count(), 8);
    }
    #[test]
    fn identical_not_separable() {
        let s = timing_stats(&[3.0, 3.0]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&s, &s);
        assert_eq!(ratio, 1.0);
        assert!(!separable);
    }
    #[test]
    fn constant_double_separable() {
        let a = timing_stats(&[2.0; 4]).unwrap();
        let b = timing_stats(&[4.0; 4]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&a, &b);
        assert_eq!(ratio, 2.0);
        assert!(separable);
    }
    #[test]
    fn noisy_overlap_not_separable() {
        let a = timing_stats(&[1.0, 5.0, 9.0]).unwrap();
        let b = timing_stats(&[8.0, 10.0, 12.0]).unwrap();
        assert!(!timing_relation(&a, &b).3);
    }
    #[test]
    fn statistics_centers_and_rank() {
        let s = timing_stats(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(
            (s.count, s.median, s.min, s.max, s.p90, s.mean, s.spread()),
            (4, 2.5, 1.0, 4.0, 4.0, 2.5, 3.0)
        );
        assert_eq!(timing_stats(&[3.0, 1.0, 2.0]).unwrap().median, 2.0);
        assert_eq!(
            timing_stats(&(1..=10).map(f64::from).collect::<Vec<_>>())
                .unwrap()
                .p90,
            9.0
        );
    }
    #[test]
    fn bad_samples_and_counts() {
        for values in [&[][..], &[0.0], &[-1.0], &[f64::NAN], &[f64::INFINITY]] {
            assert!(timing_stats(values).is_err());
        }
        assert!(positive_count("0").is_err());
        assert!(positive_count("-1").is_err());
        assert!(timing_counts(usize::MAX, 1).is_err());
        assert!(timing_counts(1, usize::MAX).is_err());
        assert!(timing_schedule(0, 1).is_err());
    }
    #[test]
    fn block_medians_are_per_block() {
        let s = [1.0, 3.0, 2.0, 10.0, 30.0, 20.0];
        assert_eq!(block_medians(&s, 3).unwrap(), vec![2.0, 20.0]);
        assert!(block_medians(&s, 4).is_err());
        assert!(block_medians(&[], 1).is_err());
    }

    /// One 7 ms hiccup among ~300 us samples: the legacy full-range rule is
    /// vetoed, the block-median rule is not.
    #[test]
    fn block_rule_survives_one_outlier_legacy_rule_does_not() {
        let n = 4;
        let mut base: Vec<f64> = (0..16).map(|i| 300.0 + f64::from(i % 3)).collect();
        let cand: Vec<f64> = (0..16).map(|i| 290.0 + f64::from(i % 3)).collect();
        base[5] = 7003.0;
        let r = block_rule(&base, &cand, n).unwrap();
        assert!(r.separable, "{r:?}");
        assert!(r.block_ratio < 1.0 && r.round_ratio_max < 1.0);
        let (a, b) = (timing_stats(&base).unwrap(), timing_stats(&cand).unwrap());
        assert!(!timing_relation(&a, &b).3, "legacy rule sees the hiccup");
    }

    #[test]
    fn block_rule_needs_every_block_on_one_side() {
        let n = 2;
        // Candidate blocks (rounds 1: B B = 2 blocks) 100/110 vs baseline blocks 105/115: overlap.
        let base = [105.0, 105.0, 115.0, 115.0];
        let cand = [100.0, 100.0, 110.0, 110.0];
        let r = block_rule(&base, &cand, n).unwrap();
        assert!(!r.separable);
        assert_eq!(r.baseline_block_medians, vec![105.0, 115.0]);
        assert_eq!(r.candidate_block_medians, vec![100.0, 110.0]);
        assert!((r.block_ratio - 105.0 / 110.0).abs() < 1e-12);
        // Strictly above every baseline block: separable (slower candidate).
        let slow = [200.0, 200.0, 210.0, 210.0];
        let r = block_rule(&base, &slow, n).unwrap();
        assert!(r.separable && r.round_ratio_min > 1.0);
        // Equal arms are never separable (strict inequality).
        assert!(!block_rule(&base, &base, n).unwrap().separable);
    }

    #[test]
    fn block_rule_rejects_partial_rounds() {
        assert!(block_rule(&[1.0; 6], &[1.0; 6], 2).is_err());
        assert!(block_rule(&[1.0; 8], &[1.0; 4], 2).is_err());
    }

    #[test]
    fn round_ratios_are_paired_per_round() {
        let n = 1;
        // round 1: base 10,10 cand 5,5 -> 0.5; round 2: base 10,10 cand 20,20 -> 2.0
        let base = [10.0, 10.0, 10.0, 10.0];
        let cand = [5.0, 5.0, 20.0, 20.0];
        let r = block_rule(&base, &cand, n).unwrap();
        assert_eq!((r.round_ratio_min, r.round_ratio_max), (0.5, 2.0));
        assert_eq!(r.rounds_candidate_faster, 1);
        assert!(!r.separable);
    }

    #[test]
    fn sign_test_matches_exact_binomial() {
        assert_eq!(sign_test_p(0, 0), 1.0);
        assert_eq!(sign_test_p(3, 6), 1.0);
        // 6/6: 2 * (1/64).
        assert!((sign_test_p(6, 6) - 2.0 / 64.0).abs() < 1e-15);
        assert!((sign_test_p(0, 6) - 2.0 / 64.0).abs() < 1e-15);
        // 1/6 or 5/6: 2 * (1 + 6) / 64.
        assert!((sign_test_p(5, 6) - 14.0 / 64.0).abs() < 1e-15);
        assert!((sign_test_p(1, 6) - 14.0 / 64.0).abs() < 1e-15);
        assert!(sign_test_p(63, 64) < 1e-15);
        assert!(sign_test_p(34, 64) > 0.5);
    }

    #[test]
    fn extension_doubles_until_a_stop() {
        use ExtendStop::*;
        assert_eq!(extend_rounds(6, 64, 10.0, 240.0, true), Err(Separable));
        assert_eq!(extend_rounds(6, 64, 10.0, 240.0, false), Ok(6));
        assert_eq!(extend_rounds(12, 64, 20.0, 240.0, false), Ok(12));
        // Capped by max_rounds.
        assert_eq!(extend_rounds(48, 64, 80.0, 240.0, false), Ok(16));
        assert_eq!(extend_rounds(64, 64, 80.0, 240.0, false), Err(MaxRounds));
        // Capped by what fits the budget at the observed per-round time (2 s/round).
        assert_eq!(extend_rounds(6, 64, 12.0, 20.0, false), Ok(4));
        assert_eq!(extend_rounds(6, 64, 12.0, 13.0, false), Err(TimeBudget));
        assert_eq!(extend_rounds(6, 64, 300.0, 240.0, false), Err(TimeBudget));
    }

    #[test]
    fn extension_schedule_terminates_within_bounds() {
        // Simulate a never-separable run: rounds 6 -> 12 -> 24 -> 48 -> 64.
        let (mut done, mut t) = (6usize, 6.0f64);
        let mut steps = vec![done];
        while let Ok(extra) = extend_rounds(done, 64, t, 1e9, false) {
            done += extra;
            t += extra as f64;
            steps.push(done);
        }
        assert_eq!(steps, vec![6, 12, 24, 48, 64]);
    }
}
