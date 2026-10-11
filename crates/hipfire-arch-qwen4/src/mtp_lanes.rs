// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native MTP windows on the exact lanes of the Flash-Next batch route.
//!
//! Every window of a lane is byte-identical to one singleton window on that
//! lane's own states (`fn-batch/S5-CONTRACT.md`, exactness model):
//!
//! | [`LaneRoute`] | singleton twin |
//! |---|---|
//! | `Batched { k_max }` | `mtp_step`'s batched route at depth `k_max` |
//! | `ArSynced` | `mtp_ar_step(calibrate = true)` without the timing |
//! | `ArRetired` | `mtp_ar_step(calibrate = false)` |
//! | `Ar` | non-spec lane: the S3 AR row |
//!
//! The phases the executor ([`crate::lane_exec`]) runs around its one trunk
//! forward:
//!
//! - [`draft_lanes`]: route every lane and run the step-synchronous batched
//!   draft (every still-drafting lane enqueues its draft step, then one D2H
//!   reads all of them).
//! - [`finish_lane_window`]: a Verify lane's greedy accept and the singleton's
//!   window tail ([`crate::mtp_spec::finish_native_window`]).
//! - [`ar_synced_head`]: the head append of an `ArSynced` lane.
//! - [`prefill_tile`]: a spec lane's prompt-fill chunk (trunk tile with wide
//!   hidden capture plus the head append of every row).
//!
//! The per-lane floor ([`LaneSpecPolicy`]) keeps the singleton's
//! [`MtpFloor`] and agreement, fed in AR-row units from the store-level
//! [`LaneCostModel`] instead of wall clocks (see its docs).

use crate::bundle::{BundleError, Qwen4Bundle};
use crate::lane::{LabRoutes, LaneSpec, LaneSpecConfig, Qwen4LaneState};
use crate::lane_exec::{LaneDraft, LaneRoute};
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuStateSnapshot};
use crate::mtp_spec::{
    accept_native_greedy, batched_depth, draft_accept_estimate, draft_pairing,
    finish_native_window, mtp_batched_fill_enabled, mtp_prefill_chunk,
    mtp_prefill_chunk_needs_pick, mtp_reuse_prefill_ids_enabled,
    mtp_skip_intermediate_pick_enabled, observe_agreement, Agreement, DraftPairing, FloorRoute,
    MtpFloor, NativeFinish, NativeWindow, DRAFT_ACCEPT_MAX, MTP_AGREEMENT_PRIOR, MTP_MAX_DEPTH,
    MTP_ROW_WORTH,
};
use crate::lane::Qwen4LaneStore;
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::{GreedyAccept, MtpWindow};
use rdna_compute::{Gpu, GpuTensor};

// ---------------------------------------------------------------------------
// Lane cost model
// ---------------------------------------------------------------------------

/// Prior `α` (µs, the pure-AR cost of a step) for 1..=8 decoding lanes
/// (measured, gfx1151; `fn-batch/S5-CONTRACT.md`, lane floor).
const COST_ALPHA_PRIOR_US: [f64; 8] = [
    30_240.0, 43_180.0, 54_080.0, 64_980.0, 79_700.0, 94_430.0, 109_160.0, 123_890.0,
];
/// Prior growth of `α` per lane above 8.
const COST_ALPHA_STEP_US: f64 = 15_490.0;
/// Prior `m` (µs): the marginal cost of one verified draft row.
const COST_M_PRIOR_US: f64 = 14_700.0;
/// Weight kept per later observation.
const COST_DECAY: f64 = 0.95;
/// The prior's second pseudo-observation sits at this many verified drafts
/// (the first at zero).
const COST_PRIOR_ROWS: f64 = 4.0;
/// `α` never fits below this.
const COST_ALPHA_FLOOR_US: f64 = 1_000.0;

fn alpha_prior(lanes: usize) -> f64 {
    match lanes {
        0 | 1 => COST_ALPHA_PRIOR_US[0],
        2..=8 => COST_ALPHA_PRIOR_US[lanes - 1],
        _ => COST_ALPHA_PRIOR_US[7] + COST_ALPHA_STEP_US * (lanes - 8) as f64,
    }
}

/// Decayed sums of `(1, e, e², W, e·W)` over the observations of one lane
/// count.
#[derive(Clone, Copy, Debug)]
struct Moments {
    w: f64,
    e: f64,
    ee: f64,
    y: f64,
    ey: f64,
}

impl Moments {
    fn empty() -> Self {
        Self {
            w: 0.0,
            e: 0.0,
            ee: 0.0,
            y: 0.0,
            ey: 0.0,
        }
    }

    /// The prior as two pseudo-observations: `W = α` at `e = 0` and
    /// `W = α + 4m` at `e = 4`, weight one each.
    fn prior(lanes: usize) -> Self {
        let alpha = alpha_prior(lanes);
        let mut moments = Self::empty();
        moments.add(0.0, alpha);
        moments.add(COST_PRIOR_ROWS, alpha + COST_PRIOR_ROWS * COST_M_PRIOR_US);
        moments
    }

    fn add(&mut self, e: f64, y: f64) {
        self.w += 1.0;
        self.e += e;
        self.ee += e * e;
        self.y += y;
        self.ey += e * y;
    }

    /// One more observation: everything seen so far decays first.
    fn observe(&mut self, e: f64, y: f64) {
        self.w *= COST_DECAY;
        self.e *= COST_DECAY;
        self.ee *= COST_DECAY;
        self.y *= COST_DECAY;
        self.ey *= COST_DECAY;
        self.add(e, y);
    }

    /// Least-squares `(α, m)` of `W = α + m·e`: `m ≥ 0`, then `α ≥ 1000`,
    /// then `m ≤ 4α`. All verified counts equal (no spread) keeps the prior
    /// `m`.
    fn fit(&self) -> (f64, f64) {
        let det = self.w * self.ee - self.e * self.e;
        let mut m = if det > 1e-9 * self.w * self.ee {
            (self.w * self.ey - self.e * self.y) / det
        } else {
            COST_M_PRIOR_US
        };
        m = m.max(0.0);
        let alpha = ((self.y - m * self.e) / self.w).max(COST_ALPHA_FLOOR_US);
        (alpha, m.min(4.0 * alpha))
    }
}

/// Store-level model of a pure-decode step's wall time: per decoding-lane
/// count `C`, a decayed regression `W = α_C + m_C · e` of the step wall `W`
/// (µs, draft start to accept end, synced) on `e`, the verified drafts the
/// step carries. A lane's AR row costs its proportional share `a = α_C / C`
/// (the unit of every lane floor); a depth-`d` window costs
/// `1 + d · m_C / a` units.
#[derive(Clone, Debug)]
pub(crate) struct LaneCostModel {
    per_lanes: Vec<Moments>,
}

impl LaneCostModel {
    pub(crate) fn new(max_lanes: usize) -> Self {
        Self {
            per_lanes: (1..=max_lanes.max(1)).map(Moments::prior).collect(),
        }
    }

    fn index(&self, lanes: usize) -> usize {
        lanes.clamp(1, self.per_lanes.len()) - 1
    }

    /// One finished pure-decode step of `lanes` decoding lanes carrying
    /// `verified` drafts in all, `wall_us` long. An unusable wall is dropped.
    pub(crate) fn observe(&mut self, lanes: usize, verified: usize, wall_us: f32) {
        if !(wall_us.is_finite() && wall_us > 0.0) {
            return;
        }
        let index = self.index(lanes);
        self.per_lanes[index].observe(verified as f64, f64::from(wall_us));
    }

    /// `(α_C, m_C)` in µs.
    #[cfg(test)]
    pub(crate) fn fit(&self, lanes: usize) -> (f32, f32) {
        let (alpha, m) = self.per_lanes[self.index(lanes)].fit();
        (alpha as f32, m as f32)
    }

    /// Cost of a window with `drafts` verified drafts in a step of `lanes`
    /// decoding lanes, in AR-row units (`1 + d · m / (α / C)`).
    pub(crate) fn window_units(&self, lanes: usize, drafts: usize) -> f32 {
        let (alpha, m) = self.per_lanes[self.index(lanes)].fit();
        let share = alpha / lanes.max(1) as f64;
        (1.0 + drafts as f64 * m / share) as f32
    }
}

// ---------------------------------------------------------------------------
// Lane floor policy
// ---------------------------------------------------------------------------

/// `HIPFIRE_MTP_AR_FLOOR` (`0` opts out), read the way the drafter's
/// `configure_request` reads it.
fn floor_env_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_MTP_AR_FLOOR").as_deref() != Ok("0")
}

/// One lane's window router: the singleton's agreement and AR floor, priced
/// in AR-row units (calibrated once at `ar_us = 1.0`; batched windows
/// observed at [`LaneCostModel::window_units`], `ArSynced` windows as
/// interleaved windows of one unit).
#[derive(Clone, Debug)]
pub(crate) struct LaneSpecPolicy {
    agreement: Agreement,
    floor: MtpFloor,
    floor_enabled: bool,
    max_k: usize,
    calibrated: bool,
    /// Windows routed under a lab route mode (the cycle's position).
    lab_windows: usize,
    /// A promoted singleton whose floor already retired it: the head lags
    /// the trunk, so the lane is head-free AR for good (every route mode).
    promoted_retired: bool,
}

impl LaneSpecPolicy {
    /// A request's fresh policy for drafts of at most `k`.
    pub(crate) fn new(k: usize) -> Self {
        Self::with_floor(k, floor_env_enabled())
    }

    pub(crate) fn with_floor(k: usize, floor_enabled: bool) -> Self {
        Self {
            agreement: [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH],
            floor: MtpFloor::new(),
            floor_enabled,
            max_k: k.min(MTP_MAX_DEPTH),
            calibrated: false,
            lab_windows: 0,
            promoted_retired: false,
        }
    }

    /// The policy of a promoted singleton: the drafter's per-depth
    /// `agreement` seeds the lane (performance only); `retired` (the
    /// singleton's floor retired it) keeps the lane `ArRetired`, otherwise
    /// the lane floor starts fresh like any admitted lane.
    pub(crate) fn promoted(k: usize, agreement: Agreement, retired: bool) -> Self {
        Self::promoted_with_floor(k, agreement, retired, floor_env_enabled())
    }

    fn promoted_with_floor(
        k: usize,
        agreement: Agreement,
        retired: bool,
        floor_enabled: bool,
    ) -> Self {
        Self {
            agreement,
            promoted_retired: retired,
            ..Self::with_floor(k, floor_enabled)
        }
    }

    /// Most drafts a window of this lane may carry.
    pub(crate) fn max_k(&self) -> usize {
        self.max_k
    }

    /// The floor retired this lane to head-free AR (sticky).
    pub(crate) fn retired(&self) -> bool {
        self.promoted_retired || (self.floor_enabled && self.floor.retired())
    }

    /// Route the lane's next window; `budget` is the most drafts it may
    /// carry (0 = AR). Retired lanes stay `ArRetired`.
    pub(crate) fn route(&mut self, budget: usize, lab: LabRoutes) -> LaneRoute {
        if self.promoted_retired {
            return LaneRoute::ArRetired;
        }
        if lab != LabRoutes::Floor {
            return self.lab_route(budget, lab);
        }
        if !self.calibrated {
            self.floor.observe_calibration(1.0, &self.agreement);
            self.calibrated = true;
        }
        if self.retired() {
            return LaneRoute::ArRetired;
        }
        if budget == 0 {
            return LaneRoute::ArSynced;
        }
        let k = budget.min(self.max_k);
        let provisional = batched_depth(&self.agreement, k);
        let batched = |depth: usize| match depth.min(k) {
            0 => LaneRoute::ArSynced,
            k_max => LaneRoute::Batched { k_max },
        };
        if !self.floor_enabled {
            return batched(provisional);
        }
        match self.floor.route(&self.agreement, k, provisional) {
            FloorRoute::Ar => LaneRoute::ArRetired,
            FloorRoute::Interleaved => LaneRoute::ArSynced,
            FloorRoute::Batched(depth) => batched(depth),
        }
    }

    /// Lab route modes bypass the floor: `Batched` drafts the whole budget;
    /// `Cycle` walks `Batched{K}`, `Batched{1}`, `ArSynced`, `Batched{2}`
    /// three times, then `ArRetired` for good (a zero budget is `ArSynced`).
    fn lab_route(&mut self, budget: usize, lab: LabRoutes) -> LaneRoute {
        let k = budget.min(self.max_k);
        let batched = |depth: usize| match depth.min(k) {
            0 => LaneRoute::ArSynced,
            k_max => LaneRoute::Batched { k_max },
        };
        let route = match lab {
            LabRoutes::Floor => unreachable!("lab_route is not the floor"),
            LabRoutes::Batched => batched(k),
            LabRoutes::Cycle => match self.lab_windows {
                n if n >= 12 => LaneRoute::ArRetired,
                n => match n % 4 {
                    0 => batched(self.max_k),
                    1 => batched(1),
                    2 => LaneRoute::ArSynced,
                    _ => batched(2),
                },
            },
        };
        self.lab_windows += 1;
        route
    }

    /// A finished batched window of budget `k` priced at `units` AR rows.
    pub(crate) fn observe_batched(&mut self, window: &MtpWindow, units: f32, k: usize) {
        observe_agreement(&mut self.agreement, window);
        if self.floor_enabled {
            self.floor
                .observe_native(NativeWindow::Batched, window, units, k, &self.agreement);
        }
    }

    /// A finished `ArSynced` window of budget `k`: one AR row, observed as an
    /// interleaved window of one unit. A zero budget observes nothing.
    pub(crate) fn observe_ar_synced(&mut self, window: &MtpWindow, k: usize) {
        if k == 0 {
            return;
        }
        observe_agreement(&mut self.agreement, window);
        if self.floor_enabled {
            self.floor
                .observe_native(NativeWindow::Interleaved, window, 1.0, k, &self.agreement);
        }
    }
}

// ---------------------------------------------------------------------------
// Pending window
// ---------------------------------------------------------------------------

/// One routed window between `draft_step` and `commit_step`.
pub(crate) struct PendingWindow {
    pub route: LaneRoute,
    /// The lane's `state.position` when the window was routed (the seed's).
    pub position: usize,
    pub seed: u32,
    /// The kept drafts (`Batched` only).
    pub drafts: Vec<u32>,
    /// Draft steps the head ran (one more than `drafts` when the last step
    /// drafted a dropped token).
    pub steps: usize,
    /// The drafts budget (clamped to the lane's `K`) the floor observes.
    pub budget: usize,
    /// The head ticket taken before the draft steps (`Batched`).
    pub head_ticket: Option<MtpGpuStateSnapshot>,
    /// The target ticket taken before the verify forward (`Batched`).
    pub target_ticket: Option<Qwen4StateSnapshot>,
    /// Set by the forward once the window is finished; consumed by commit.
    pub outcome: Option<MtpWindow>,
}

// ---------------------------------------------------------------------------
// Phases
// ---------------------------------------------------------------------------

/// Flatten `with_lane`'s `Result<Result<R, _>, _>`.
pub(crate) fn lane_call<R>(
    result: Result<Result<R, BundleError>, BundleError>,
) -> Result<R, String> {
    result.and_then(|inner| inner).map_err(|error| error.to_string())
}

/// One lane's drafting state inside the step-synchronous loop.
struct Drafting {
    lane: usize,
    position: usize,
    k_max: usize,
    input: u32,
    /// Probability the whole draft prefix is accepted.
    prefix: f32,
    drafts: Vec<u32>,
    steps: usize,
    done: bool,
}

/// Route the next window of every listed Decoding lane and run the batched
/// draft of the `Batched` ones. See `Qwen4LaneExecutor::draft_step`.
pub(crate) fn draft_lanes(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    requests: &[(RequestEpoch, usize)],
) -> Result<Vec<LaneDraft>, String> {
    let epochs: Vec<RequestEpoch> = requests.iter().map(|(epoch, _)| *epoch).collect();
    let (mut lanes, mut spec_store) = store.draft_parts(&epochs)?;
    let routes = spec_store.as_ref().map(|s| s.routes);
    let mut out: Vec<LaneDraft> = Vec::with_capacity(requests.len());
    let mut drafting: Vec<Drafting> = Vec::new();

    for (index, ((epoch, budget), lane)) in requests.iter().zip(lanes.iter_mut()).enumerate() {
        let Qwen4LaneState { host, state, spec } = &mut **lane;
        if host.status != crate::lane::LaneStatus::Decoding {
            return Err(format!(
                "draft_step: lane {epoch:?} is {:?}, not Decoding",
                host.status
            ));
        }
        let seed = host
            .pending_seed
            .ok_or_else(|| format!("draft_step: lane {epoch:?} has no pending seed"))?;
        let Some(spec) = spec.as_mut() else {
            out.push(LaneDraft {
                epoch: *epoch,
                route: LaneRoute::Ar,
                drafts: Vec::new(),
            });
            continue;
        };
        if spec.window.is_some() {
            return Err(format!(
                "draft_step: lane {epoch:?} has an unconsumed window (commit or abort the step first)"
            ));
        }
        let route = spec.policy.route(*budget, routes.unwrap_or_default());
        let position = state.position;
        spec.window = Some(PendingWindow {
            route,
            position,
            seed,
            drafts: Vec::new(),
            steps: 0,
            budget: (*budget).min(spec.policy.max_k()),
            head_ticket: None,
            target_ticket: None,
            outcome: None,
        });
        out.push(LaneDraft {
            epoch: *epoch,
            route,
            drafts: Vec::new(),
        });
        let LaneRoute::Batched { k_max } = route else {
            continue;
        };
        // I1: the head sits at the lane's frontier.
        if state.position != spec.head.position() {
            return Err(format!(
                "draft_step: lane {epoch:?} I1: trunk position {} != head position {}",
                state.position,
                spec.head.position()
            ));
        }
        // The head's last draft step and its full-accept append reach
        // `position + k_max`; the verify's rows end at the same bound.
        let end = position
            .checked_add(k_max + 1)
            .ok_or_else(|| "draft_step: position overflow".to_string())?;
        spec.head
            .ensure_mapped_capacity(gpu, end)
            .map_err(|e| format!("draft_step: map head arenas of {epoch:?}: {e}"))?;
        state
            .ensure_mapped_capacity(gpu, end)
            .map_err(|e| format!("draft_step: map trunk arenas of {epoch:?}: {e}"))?;
        let ticket = lane_call(bundle.with_lane(
            state,
            Some((&mut spec.head, &mut spec.draft)),
            |b| b.mtp_snapshot(gpu),
        ))
        .map_err(|e| format!("draft_step: head snapshot of {epoch:?}: {e}"))?;
        if let Some(window) = spec.window.as_mut() {
            window.head_ticket = Some(ticket);
        }
        drafting.push(Drafting {
            lane: index,
            position,
            k_max,
            input: seed,
            prefix: 1.0,
            drafts: Vec::with_capacity(k_max),
            steps: 0,
            done: false,
        });
    }

    let slot_bytes = Gpu::TOPK8_OUT_BYTES;
    let max_steps = drafting.iter().map(|d| d.k_max).max().unwrap_or(0);
    for j in 0..max_steps {
        let active: Vec<usize> = (0..drafting.len()).filter(|&d| !drafting[d].done).collect();
        if active.is_empty() {
            break;
        }
        let store = spec_store
            .as_mut()
            .ok_or_else(|| "draft_step: batched window in a store without MTP lanes".to_string())?;
        // Every still-drafting lane enqueues its draft step, in slot order.
        for (slot, &d) in active.iter().enumerate() {
            let st = &drafting[d];
            let Qwen4LaneState { state, spec, .. } = &mut *lanes[st.lane];
            let spec = spec
                .as_mut()
                .ok_or_else(|| "draft_step: drafting lane lost its MTP state".to_string())?;
            let LaneSpec {
                head,
                draft,
                pending_hidden,
                ..
            } = spec;
            let top1 = store
                .draft_top1
                .sub_offset(slot * slot_bytes, slot_bytes);
            let hidden = (j == 0).then_some(&*pending_hidden);
            let (input, position) = (st.input, st.position + j);
            lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                b.mtp_forward_token_deferred(gpu, input, hidden, position, j == 0, &top1)
            }))
            .map_err(|e| format!("draft_step: draft step {j}: {e}"))?;
        }
        // One D2H for every active lane.
        let bytes = active.len() * slot_bytes;
        gpu.hip
            .memcpy_dtoh(&mut store.draft_top1_host[..bytes], &store.draft_top1.buf)
            .map_err(|e| format!("draft_step: draft download: {e}"))?;
        for (slot, &d) in active.iter().enumerate() {
            let top = &store.draft_top1_host[slot * slot_bytes..(slot + 1) * slot_bytes];
            let st = &mut drafting[d];
            let Qwen4LaneState { state, spec, .. } = &mut *lanes[st.lane];
            let spec = spec
                .as_mut()
                .ok_or_else(|| "draft_step: drafting lane lost its MTP state".to_string())?;
            let LaneSpec { head, draft, .. } = spec;
            let (token, margin) = lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
                let token = b.mtp_draft_finish(top)?;
                Ok::<_, BundleError>((token, b.mtp_draft_margin()))
            }))
            .map_err(|e| format!("draft_step: draft finish {j}: {e}"))?;
            // The singleton's early stop, statement for statement.
            st.input = token;
            st.steps += 1;
            st.prefix *= draft_accept_estimate(margin);
            if j > 0 && st.prefix < MTP_ROW_WORTH {
                st.done = true;
                continue;
            }
            st.drafts.push(token);
            if st.prefix * DRAFT_ACCEPT_MAX < MTP_ROW_WORTH || j + 1 >= st.k_max {
                st.done = true;
            }
        }
    }

    for st in drafting {
        let lane = &mut *lanes[st.lane];
        let window = lane
            .spec
            .as_mut()
            .and_then(|spec| spec.window.as_mut())
            .ok_or_else(|| "draft_step: window vanished".to_string())?;
        window.steps = st.steps;
        window.drafts = st.drafts.clone();
        out[st.lane].drafts = st.drafts;
    }
    Ok(out)
}

/// Finish a Verify lane after the trunk forward: greedy accept over the
/// lane's picks, then the singleton's window tail on the lane's own states.
/// `hidden` is the store's wide-hidden capture, the lane's rows starting at
/// `row_begin`; `picks` are the lane's `1 + d` greedy picks. Returns the
/// window (the lane's `window.outcome` is the caller's to set).
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_lane_window(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    config: LaneSpecConfig,
    hidden: &GpuTensor,
    row_begin: usize,
    picks: &[u32],
    skip_rollback: bool,
) -> Result<MtpWindow, String> {
    let wide = bundle
        .config
        .hc_count
        .checked_mul(bundle.config.hidden_size)
        .ok_or_else(|| "finish_lane_window: hidden width overflow".to_string())?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "finish_lane_window: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        window,
        ..
    } = spec;
    let window = window
        .as_ref()
        .ok_or_else(|| "finish_lane_window: lane has no pending window".to_string())?;
    if !matches!(window.route, LaneRoute::Batched { .. }) {
        return Err(format!(
            "finish_lane_window: window routed {:?} has no verify",
            window.route
        ));
    }
    let rows = window.drafts.len() + 1;
    if picks.len() != rows {
        return Err(format!(
            "finish_lane_window: {} picks for a {rows}-row verify",
            picks.len()
        ));
    }
    // The ring must cover the verify: the replay branch of the singleton is
    // unreachable on lanes.
    if state.row_capture_rows() < rows {
        return Err(format!(
            "finish_lane_window: the lane ring holds {} rows, the verify has {rows}",
            state.row_capture_rows()
        ));
    }
    // I1 at finish: the verify published `rows` rows, the head ran `steps`
    // draft steps from the window's frontier.
    if state.position != window.position + rows {
        return Err(format!(
            "finish_lane_window: I1: trunk position {} != {} after the verify",
            state.position,
            window.position + rows
        ));
    }
    if head.position() != window.position + window.steps {
        return Err(format!(
            "finish_lane_window: I1: head position {} != {} after {} draft steps",
            head.position(),
            window.position + window.steps,
            window.steps
        ));
    }
    let target_snapshot = window
        .target_ticket
        .ok_or_else(|| "finish_lane_window: no target ticket".to_string())?;
    let mtp_ticket = window
        .head_ticket
        .ok_or_else(|| "finish_lane_window: no head ticket".to_string())?;
    let eos = config.end_of_turn.unwrap_or(bundle.config.eos_token_id);
    let drafts = window.drafts.clone();
    let acceptance = accept_native_greedy(&drafts, picks, Some(eos))?;
    let mut block = Vec::with_capacity(rows);
    block.push(window.seed);
    block.extend_from_slice(&drafts);
    let (position, steps) = (window.position, window.steps);
    let verify_hidden = hidden.sub_offset(row_begin * wide, rows * wide);
    // Taken before `with_lane`: it names the forward scratch.
    let ple_normed = bundle
        .lane_ple_normed_rows(row_begin, rows)
        .map_err(|e| format!("finish_lane_window: PLE rows: {e}"))?;
    let result = MtpWindow {
        committed: acceptance.committed.clone(),
        accepted: acceptance.accepted,
        drafts_generated: drafts.len(),
    };
    // Lab negative control: a rejected verify finishes as a full accept, so
    // the lane keeps every verified row (the rejected tail included) while
    // the stream still follows the true acceptance.
    let finish_acceptance = if skip_rollback && acceptance.accepted < drafts.len() {
        GreedyAccept {
            committed: drafts
                .iter()
                .copied()
                .chain(std::iter::once(picks[drafts.len()]))
                .collect(),
            accepted: drafts.len(),
            hit_eos: false,
        }
    } else {
        acceptance
    };
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        finish_native_window(
            b,
            gpu,
            NativeFinish {
                position,
                block: &block,
                steps,
                acceptance: finish_acceptance,
                target_snapshot,
                mtp_ticket,
                pending_hidden,
                verify_hidden: &verify_hidden,
                ple_normed: &ple_normed,
            },
            None,
        )
        .map_err(BundleError::Forward)
    }))
    .map_err(|e| format!("finish_lane_window: {e}"))?;
    Ok(result)
}

/// The head work of an `ArSynced` lane after its AR row: append the seed
/// with the `DraftPairing` conditioning, then keep the AR row's wide hidden
/// as the pending hidden. `dec_row` is the lane's row in `hidden`.
pub(crate) fn ar_synced_head(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    hidden: &GpuTensor,
    dec_row: usize,
) -> Result<(), String> {
    let wide = bundle
        .config
        .hc_count
        .checked_mul(bundle.config.hidden_size)
        .ok_or_else(|| "ar_synced_head: hidden width overflow".to_string())?;
    let Qwen4LaneState { state, spec, .. } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "ar_synced_head: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        window,
        ..
    } = spec;
    let window = window
        .as_ref()
        .ok_or_else(|| "ar_synced_head: lane has no pending window".to_string())?;
    let (position, seed) = (window.position, window.seed);
    if head.position() != position {
        return Err(format!(
            "ar_synced_head: I1: head position {} != lane position {position}",
            head.position()
        ));
    }
    if state.position != position + 1 {
        return Err(format!(
            "ar_synced_head: lane position {} != {} after the AR row",
            state.position,
            position + 1
        ));
    }
    let row = hidden.sub_offset(dec_row * wide, wide);
    let conditioning = match draft_pairing() {
        DraftPairing::HeadState => None,
        DraftPairing::AlignedHead | DraftPairing::AlignedTarget => Some(&row),
    };
    lane_call(bundle.with_lane(state, Some((&mut *head, draft)), |b| {
        b.mtp_append_token(gpu, seed, conditioning, position)
    }))
    .map_err(|e| format!("ar_synced_head: head append: {e}"))?;
    if head.position() != position + 1 {
        return Err(format!(
            "ar_synced_head: head ended at {}, expected {}",
            head.position(),
            position + 1
        ));
    }
    gpu.copy_d2d(&row, pending_hidden, pending_hidden.byte_size())
        .map_err(|e| format!("ar_synced_head: pending hidden copy: {e}"))
}

/// One prompt tile of a spec lane: the singleton `mtp_prefill` chunk body on
/// the lane's own states. Returns the trunk pick when the tile asks for one
/// (always on the prompt-completing tile).
pub(crate) fn prefill_tile(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    lane: &mut Qwen4LaneState,
    tokens: &[u32],
    append_scratch: Option<&mut MtpAppendScratch>,
) -> Result<Option<u32>, String> {
    let Qwen4LaneState { host, state, spec } = lane;
    let spec = spec
        .as_mut()
        .ok_or_else(|| "prefill_tile: lane has no MTP state".to_string())?;
    let LaneSpec {
        head,
        draft,
        pending_hidden,
        ..
    } = spec;
    let start = host.prefilled;
    if state.position != start || head.position() != start {
        return Err(format!(
            "prefill_tile: I1: trunk position {}, head position {}, prefilled {start}",
            state.position,
            head.position()
        ));
    }
    let end = start + tokens.len();
    let needs_pick = mtp_prefill_chunk_needs_pick(
        end,
        host.prompt.len(),
        mtp_skip_intermediate_pick_enabled(),
    );
    let scratch = append_scratch.filter(|_| mtp_batched_fill_enabled());
    let reuse = mtp_reuse_prefill_ids_enabled();
    lane_call(bundle.with_lane(state, Some((head, draft)), |b| {
        mtp_prefill_chunk(
            b,
            gpu,
            tokens,
            start,
            needs_pick,
            pending_hidden,
            scratch,
            reuse,
            &|| false,
        )
        .map_err(BundleError::Forward)
    }))
    .map_err(|e| format!("prefill_tile: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(committed: usize, accepted: usize, drafts: usize) -> MtpWindow {
        MtpWindow {
            committed: vec![7; committed],
            accepted,
            drafts_generated: drafts,
        }
    }

    /// Feed `observations` `(lanes, e, W)` `rounds` times.
    fn feed(model: &mut LaneCostModel, lanes: usize, alpha: f64, m: f64, rounds: usize) {
        for round in 0..rounds {
            let e = round % 6;
            model.observe(lanes, e, (alpha + m * e as f64) as f32);
        }
    }

    #[test]
    fn a_fresh_model_reproduces_its_priors() {
        let model = LaneCostModel::new(12);
        for (lanes, alpha) in [(1, 30_240.0), (2, 43_180.0), (8, 123_890.0), (9, 139_380.0), (12, 185_850.0)] {
            let (a, m) = model.fit(lanes);
            assert!((f64::from(a) - alpha).abs() < alpha * 1e-4, "alpha {lanes}: {a}");
            assert!((m - 14_700.0).abs() < 14_700.0 * 1e-4, "m {lanes}: {m}");
        }
        // A window of d drafts costs 1 + d m C / alpha units.
        let units = model.window_units(2, 3);
        let want = 1.0 + 3.0 * 14_700.0 / (43_180.0 / 2.0);
        assert!((f64::from(units) - want).abs() < 1e-3, "{units} vs {want}");
        assert!((model.window_units(4, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn regression_recovers_alpha_and_m_from_synthetic_steps() {
        let mut model = LaneCostModel::new(8);
        feed(&mut model, 3, 52_000.0, 9_000.0, 400);
        let (alpha, m) = model.fit(3);
        assert!((f64::from(alpha) - 52_000.0).abs() < 520.0, "alpha {alpha}");
        assert!((f64::from(m) - 9_000.0).abs() < 90.0, "m {m}");
        // Other lane counts keep their priors.
        let (alpha4, m4) = model.fit(4);
        assert!((f64::from(alpha4) - 64_980.0).abs() < 10.0);
        assert!((f64::from(m4) - 14_700.0).abs() < 10.0);
        // Units follow the fit: share a = alpha / C.
        let units = model.window_units(3, 2);
        let want = 1.0 + 2.0 * 9_000.0 / (52_000.0 / 3.0);
        assert!((f64::from(units) - want).abs() < 0.02, "{units} vs {want}");
    }

    #[test]
    fn the_fit_tracks_a_drifting_cost_through_the_decay() {
        let mut model = LaneCostModel::new(2);
        feed(&mut model, 1, 30_000.0, 14_000.0, 200);
        feed(&mut model, 1, 45_000.0, 6_000.0, 300);
        let (alpha, m) = model.fit(1);
        assert!((f64::from(alpha) - 45_000.0).abs() < 450.0, "alpha {alpha}");
        assert!((f64::from(m) - 6_000.0).abs() < 60.0, "m {m}");
    }

    #[test]
    fn the_fit_respects_the_clamps() {
        // W falls with e: the slope clamps at zero, alpha is the mean.
        let mut model = LaneCostModel::new(2);
        for round in 0..300 {
            let e = round % 6;
            model.observe(2, e, (60_000.0 - 5_000.0 * e as f64) as f32);
        }
        let (alpha, m) = model.fit(2);
        assert_eq!(m, 0.0);
        assert!(alpha >= 1_000.0 && alpha < 60_000.0, "alpha {alpha}");
        // A near-zero wall cannot take alpha below the floor.
        let mut tiny = LaneCostModel::new(1);
        for round in 0..400 {
            tiny.observe(1, round % 6, 10.0);
        }
        let (alpha, m) = tiny.fit(1);
        assert_eq!(alpha, 1_000.0);
        assert!(m <= 4.0 * alpha + 1e-3, "m {m} above 4 alpha");
        // A steep slope clamps at 4 alpha.
        let mut steep = LaneCostModel::new(1);
        for round in 0..400 {
            let e = round % 6;
            steep.observe(1, e, (2_000.0 + 40_000.0 * e as f64) as f32);
        }
        let (alpha, m) = steep.fit(1);
        assert!(m <= 4.0 * alpha + 1e-3, "m {m} alpha {alpha}");
    }

    #[test]
    fn unusable_walls_and_a_lone_verified_count_do_not_break_the_fit() {
        let mut model = LaneCostModel::new(2);
        let before = model.fit(1);
        for wall in [f32::NAN, f32::INFINITY, -4.0, 0.0] {
            model.observe(1, 2, wall);
        }
        assert_eq!(model.fit(1), before);
        // Every observation at one verified count: the slope keeps the prior.
        for _ in 0..2_000 {
            model.observe(1, 0, 31_000.0);
        }
        let (alpha, m) = model.fit(1);
        assert!(alpha.is_finite() && m.is_finite() && m >= 0.0);
        assert!((f64::from(alpha) - 31_000.0).abs() < 400.0, "alpha {alpha}");
    }

    #[test]
    fn a_zero_budget_routes_ar_synced() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(policy.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
        let mut off = LaneSpecPolicy::with_floor(3, false);
        assert_eq!(off.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
    }

    #[test]
    fn floor_off_routes_the_agreement_depth() {
        let mut policy = LaneSpecPolicy::with_floor(3, false);
        let depth = batched_depth(&policy.agreement, 3);
        let want = match depth {
            0 => LaneRoute::ArSynced,
            d => LaneRoute::Batched { k_max: d },
        };
        assert_eq!(policy.route(3, LabRoutes::Floor), want);
        // A smaller budget caps the depth.
        let capped = batched_depth(&policy.agreement, 1);
        let want = match capped {
            0 => LaneRoute::ArSynced,
            d => LaneRoute::Batched { k_max: d },
        };
        assert_eq!(policy.route(1, LabRoutes::Floor), want);
        // Floor off never retires, whatever it observes.
        for _ in 0..8 {
            policy.observe_batched(&window(1, 0, 1), 50.0, 3);
        }
        assert!(!policy.retired());
        assert_ne!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
    }

    #[test]
    fn a_fresh_floor_probes_batched_then_retires_stickily() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        let first = policy.route(3, LabRoutes::Floor);
        let depth = batched_depth(&policy.agreement, 3);
        assert_eq!(
            first,
            match depth {
                0 => LaneRoute::ArSynced,
                d => LaneRoute::Batched { k_max: d },
            }
        );
        // Three probe windows far costlier than their emitted tokens.
        for _ in 0..3 {
            policy.observe_batched(&window(1, 0, 1), 6.0, 3);
        }
        assert!(policy.retired());
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        // Sticky: a later cheap window changes nothing, nor does budget 0.
        policy.observe_batched(&window(4, 3, 3), 0.5, 3);
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        assert_eq!(policy.route(0, LabRoutes::Floor), LaneRoute::ArRetired);
    }

    #[test]
    fn a_low_agreement_probe_routes_ar_synced() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        policy.agreement = [(0.0, 2.0); MTP_MAX_DEPTH];
        assert_eq!(batched_depth(&policy.agreement, 3), 0);
        // The floor's probe budget routes the provisional depth: none is
        // interleaved, which a lane runs as one synced AR row.
        assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArSynced);
        policy.observe_ar_synced(&window(1, 0, 0), 3);
        assert!(!policy.retired());
        // A zero budget observes nothing.
        policy.observe_ar_synced(&window(1, 0, 0), 0);
    }

    #[test]
    fn a_promoted_retired_policy_stays_ar_retired_in_every_mode() {
        for floor_enabled in [true, false] {
            let mut policy = LaneSpecPolicy::promoted_with_floor(
                3,
                [(0.9, 1.0); MTP_MAX_DEPTH],
                true,
                floor_enabled,
            );
            assert!(policy.retired());
            for lab in [LabRoutes::Floor, LabRoutes::Batched, LabRoutes::Cycle] {
                for budget in [0, 1, 3] {
                    assert_eq!(policy.route(budget, lab), LaneRoute::ArRetired);
                }
            }
            policy.observe_batched(&window(4, 3, 3), 0.5, 3);
            policy.observe_ar_synced(&window(1, 0, 0), 3);
            assert!(policy.retired());
            assert_eq!(policy.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        }
    }

    #[test]
    fn a_promoted_live_policy_routes_like_a_fresh_one_with_the_given_agreement() {
        let low = [(0.0, 2.0); MTP_MAX_DEPTH];
        let mut promoted = LaneSpecPolicy::promoted_with_floor(3, low, false, true);
        assert!(!promoted.retired());
        assert_eq!(promoted.agreement, low);
        assert_eq!(batched_depth(&promoted.agreement, 3), 0);
        let mut fresh = LaneSpecPolicy::with_floor(3, true);
        fresh.agreement = low;
        assert_eq!(
            promoted.route(3, LabRoutes::Floor),
            fresh.route(3, LabRoutes::Floor)
        );
        assert_eq!(promoted.route(0, LabRoutes::Floor), LaneRoute::ArSynced);
        // The floor starts fresh: it still retires on costly probe windows.
        let mut probing = LaneSpecPolicy::promoted_with_floor(
            3,
            [MTP_AGREEMENT_PRIOR; MTP_MAX_DEPTH],
            false,
            true,
        );
        assert_ne!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        for _ in 0..3 {
            probing.observe_batched(&window(1, 0, 1), 6.0, 3);
        }
        assert!(probing.retired());
        assert_eq!(probing.route(3, LabRoutes::Floor), LaneRoute::ArRetired);
        // Lab modes still bypass the floor on a live promoted lane.
        let mut lab = LaneSpecPolicy::promoted_with_floor(3, low, false, true);
        assert_eq!(
            lab.route(2, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 2 }
        );
    }

    #[test]
    fn lab_route_modes_bypass_the_floor() {
        let mut policy = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(
            policy.route(2, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 2 }
        );
        assert_eq!(
            policy.route(9, LabRoutes::Batched),
            LaneRoute::Batched { k_max: 3 }
        );
        assert_eq!(policy.route(0, LabRoutes::Batched), LaneRoute::ArSynced);

        let mut cycle = LaneSpecPolicy::with_floor(3, true);
        let mut routes = Vec::new();
        for _ in 0..14 {
            routes.push(cycle.route(3, LabRoutes::Cycle));
        }
        let pattern = [
            LaneRoute::Batched { k_max: 3 },
            LaneRoute::Batched { k_max: 1 },
            LaneRoute::ArSynced,
            LaneRoute::Batched { k_max: 2 },
        ];
        for (n, route) in routes.iter().take(12).enumerate() {
            assert_eq!(*route, pattern[n % 4], "window {n}");
        }
        assert_eq!(routes[12], LaneRoute::ArRetired);
        assert_eq!(routes[13], LaneRoute::ArRetired);
        // The budget caps the cycle's depths.
        let mut capped = LaneSpecPolicy::with_floor(3, true);
        assert_eq!(
            capped.route(1, LabRoutes::Cycle),
            LaneRoute::Batched { k_max: 1 }
        );
    }
}
