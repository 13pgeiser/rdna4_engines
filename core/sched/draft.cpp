/* draft.cpp -- the static draft-depth policy, and the adaptive one `--draft-window adaptive` turns
 * on. See draft.h for why the first is the default and what the second changes.
 */
#include "draft.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rad {

namespace {

/* THE STEP-COST PRIOR, a shape and not a measurement: every draft adds six percent to a step --
 * one round of a serial head and one more verify row against a weight-bound step. It prices a
 * depth nothing has measured yet, and only its RATIOS are read once anything has been measured:
 * an unmeasured depth is the nearest measured one scaled by prior(d) / prior(d'). */
constexpr double kCostSlope = 0.06;
double cost_prior(int d) { return 1.0 + kCostSlope * (double)d; }

/* A STEP TIME'S MEAN IS A RUNNING MEAN FOR ITS FIRST STEPS AND THEN A MOVING AVERAGE WITH A
 * HALF-LIFE OF 32 STEPS. Long enough that the scatter of one step against the next -- a tier
 * copy, a recording, a prefix-cache trim -- averages out; short enough to follow a context that
 * grows by a few thousand tokens over a reply. */
constexpr double kCostHalfLife = 32.0;
/* AND ONE STEP COUNTS FOR AT MOST FOUR TIMES THE KEY'S MEAN, once the mean is over four steps. A
 * step that waited on something other than its own work -- a first recording, an allocation, a
 * host stall -- says nothing about what its depth costs, and one such step would otherwise move a
 * short mean by more than any depth. A mean over fewer steps is not yet a reference to clip
 * against: its first step may itself be the odd one. */
constexpr double kCostOutlier = 4.0;
constexpr int64_t kCostClipAfter = 4;

/* THE PRIOR WEIGHS AS MUCH AS THIS MANY VERIFIES. The (n+1)-th real one moves an estimate by
 * 1 / (n + k + 1) -- a running mean over the prior and the evidence -- until that step falls below
 * the half-life's, and from then on it is the moving average it is meant to be. Without it a
 * sequence would spend its first ~16 verifies at whatever window the prior chose, which is most of
 * a short answer. */
constexpr double kPriorWeight = 4.0;


}  /* namespace */

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;

    adapt_ = p.adapt && p.max_depth > 0;
    alpha_ = 1.0 - std::exp2(-1.0 / kHalfLife);
    step_d_ = 0;
    step_w_ = 0;
    step_held_ = 0;
    prior_n_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) prior_[j] = 0.0f;
    for (int d = 0; d <= RAD_SCHED_MAX_SPEC; ++d) {
        by_depth_[d] = 0;
        for (int c = 0; c < kCostClasses; ++c) { cost_ms_[d][c] = 0.0; cost_n_[d][c] = 0; }
    }
    if (adapt_)
        RAD_INFO("draft depth: ADAPTIVE (--draft-window adaptive), %d..%d -- the step's head depth "
                 "maximises the tokens its sequences are expected to emit over the step's measured "
                 "cost, and a sequence stops at the first draft emitted less than %.0f%% of the "
                 "time; moves at +%.0f%% and at most every %d steps",
                 kMinDepth, p_.max_depth, 100.0 * kDraftMinChance, 100.0 * kHyst, kHold);
    return RAD_OK;
}

void DraftController::observe(int drafted, int accepted) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;

    /* Position j was EVALUATED only if every position before it was accepted, so this verify
     * produces exactly: 0..accepted-1 evaluated and accepted, and -- when the prefix was not the
     * whole draft -- position `accepted` evaluated and rejected. Nothing past that was reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
    if (adapt_) ++by_depth_[drafted];
}

/* ------------------------------------------------------------------ the step's cost */
int DraftController::cost_class(int n_rows) {
    int c = 0;
    while (c + 1 < kCostClasses && n_rows >= (2 << c)) ++c;
    return c;
}

void DraftController::step_time(int depth, int n_rows, double ms) {
    if (!adapt_ || depth < 1 || depth > p_.max_depth || n_rows < 1 || !(ms > 0.0) ||
        !std::isfinite(ms))
        return;
    const int c = cost_class(n_rows);
    double&  m = cost_ms_[depth][c];
    int64_t& n = cost_n_[depth][c];
    if (n >= kCostClipAfter && ms > kCostOutlier * m) ms = kCostOutlier * m;
    const double w = std::fmax(1.0 - std::exp2(-1.0 / kCostHalfLife), 1.0 / (double)(n + 1));
    m = n ? m + w * (ms - m) : ms;
    ++n;
}

int64_t DraftController::cost_samples(int depth, int n_rows) const {
    if (depth < 1 || depth > RAD_SCHED_MAX_SPEC || n_rows < 1) return 0;
    return cost_n_[depth][cost_class(n_rows)];
}

double DraftController::step_cost(int depth, int n_rows) const {
    if (depth < 1 || depth > RAD_SCHED_MAX_SPEC) return 0.0;
    const int c = n_rows >= 1 ? cost_class(n_rows) : 0;
    if (cost_n_[depth][c] > 0) return cost_ms_[depth][c];
    /* THE NEAREST MEASURED KEY: in this class first, the nearest depth and the shallower on a
     * tie; then the nearest class. Only its ratio to the prior is borrowed, so a class whose steps
     * are twice as dear prices every depth twice as dear, and the comparison D makes -- depths at
     * ONE class -- reads the prior's shape until its own depths are measured. */
    for (int dc = 0; dc < kCostClasses; ++dc) {
        for (int sgn = 0; sgn < 2; ++sgn) {
            const int k = sgn ? c + dc : c - dc;
            if (k < 0 || k >= kCostClasses || (dc == 0 && sgn)) continue;
            int best = -1;
            for (int d = 1; d <= p_.max_depth; ++d) {
                if (cost_n_[d][k] <= 0) continue;
                if (best < 0 || std::abs(d - depth) < std::abs(best - depth)) best = d;
            }
            if (best > 0) return cost_ms_[best][k] * cost_prior(depth) / cost_prior(best);
        }
    }
    return cost_prior(depth);
}

/* ------------------------------------------------------------------ one sequence */
int DraftController::stop_of(const float* a) const {
    double p = 1.0;
    int d = 0;
    while (d < p_.max_depth) {
        p *= (double)a[d];
        if (p < kDraftMinChance) break;
        ++d;
    }
    return d < kMinDepth ? kMinDepth : d;
}

double DraftController::prior_at(int j) const {
    if (j < 0 || j >= RAD_SCHED_MAX_SPEC) return 0.0;
    return prior_n_ >= kPriorMinSeqs ? (double)prior_[j] : kPrior;
}

void DraftController::start(DraftAdapt& s) const {
    /* Each position from the sequences that finished before it once there are enough of them
     * (adapt_retire); otherwise from the fixed prior, which is optimistic on purpose (kPrior). */
    for (int j = 0; j < p_.max_depth; ++j) {
        s.a[j] = (float)prior_at(j);
        s.n[j] = 0;
    }
    s.depth = stop_of(s.a);
    s.held  = 0;
}

void DraftController::adapt_retire(const DraftAdapt& s) {
    if (!adapt_ || s.depth <= 0 || (double)s.n[0] < kHalfLife) return;
    const double w = std::fmax(1.0 - std::exp2(-1.0 / kPriorHalfLifeSeqs),
                               1.0 / (double)(prior_n_ + 1));
    for (int j = 0; j < p_.max_depth; ++j)
        prior_[j] = prior_n_ ? (float)((double)prior_[j] + w * ((double)s.a[j] - (double)prior_[j]))
                             : s.a[j];
    ++prior_n_;
}

double DraftController::yield(const DraftAdapt& s, int d) const {
    if (d > p_.max_depth) d = p_.max_depth;
    double e = 1.0, run = 1.0;
    for (int k = 1; k <= d; ++k) { run *= (double)s.a[k - 1]; e += run; }
    return e;
}

int DraftController::window_of(const DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) return 0;
    const int D = step_d_ > 0 ? step_d_ : p_.max_depth;
    return s.depth < D ? s.depth : D;
}

int DraftController::window_for(DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    return window_of(s);
}

int DraftController::adapt_observe(DraftAdapt& s, int drafted, int accepted) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    if (drafted <= 0) return s.depth;
    const int D = p_.max_depth;
    if (drafted > D) drafted = D;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    /* THE POSITIONS THIS VERIFY EVALUATED, by observe()'s rule: every one before the first
     * rejection, taken, and the rejected one, refused. A position behind the rejection was drafted
     * and never looked at, and says nothing either way. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        const double w = std::fmax(alpha_, 1.0 / ((double)s.n[j] + kPriorWeight + 1.0));
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* AND THE POSITIONS THE WINDOW DID NOT REACH DRIFT TOWARD THE DEEPEST ONE IT DID. Nothing is
     * learned about them while the window is shorter -- and a sequence held at three by prose that
     * turns into code would otherwise keep pricing positions four and up at whatever they read
     * before, and never draft deep enough to find out. The guess is that acceptance decays with
     * position rather than falling off a cliff, so the window's edge stands in for what lies past
     * it. Where that is wrong it costs a move out, the verifies that refute it, and a move back. */
    const double edge = (double)s.a[drafted - 1];
    for (int j = drafted; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));

    /* THE STOP MOVES ONLY PAST THE HOLD: it is this sequence's share of the step's shape. */
    if (s.held < INT32_MAX) ++s.held;
    if (s.held < kHold) return s.depth;
    const int d = stop_of(s.a);
    if (d != s.depth) {
        s.depth = d;
        s.held  = 0;
    }
    return s.depth;
}

/* ------------------------------------------------------------------ the step */
int DraftController::plan_step(DraftAdapt* const* seq, int n_seq, int n_rows) {
    if (!adapt_ || n_seq <= 0) return step_d_;
    if (n_rows < n_seq) n_rows = n_seq;
    const int M = p_.max_depth;

    /* THE STEP'S EXPECTED TOKENS AT EVERY HEAD DEPTH: sum_i E_i(min(D, stop_i)), plus one for
     * every row that does not draft. A depth past every stop is the deepest stop's step -- no
     * sequence drafts further and the head runs no further -- so the candidates end there. */
    double tok[RAD_SCHED_MAX_SPEC + 1] = {};
    int top = kMinDepth;
    for (int i = 0; i < n_seq; ++i) {
        DraftAdapt& s = *seq[i];
        if (s.depth <= 0) start(s);
        if (s.depth > top) top = s.depth;
        double e = 1.0, run = 1.0;
        for (int d = 1; d <= M; ++d) {
            if (d <= s.depth) { run *= (double)s.a[d - 1]; e += run; }
            tok[d] += e;
        }
    }
    const double plain = (double)(n_rows - n_seq);
    if (top > M) top = M;

    int best = kMinDepth;
    double best_g = -1.0;
    for (int d = kMinDepth; d <= top; ++d) {
        const double g = (tok[d] + plain) / step_cost(d, n_rows);
        if (g > best_g) { best_g = g; best = d; }
    }

    /* THE MOVE, rate-limited twice: not within kHold steps of the last one, and only for a depth
     * worth kHyst more than the one in force. The first step that drafts takes the best outright. */
    if (step_d_ <= 0) {
        step_d_ = best;
        step_held_ = 0;
    } else {
        if (step_held_ < INT32_MAX) ++step_held_;
        const int cur = step_d_ < top ? step_d_ : top;
        const double cur_g = (tok[cur] + plain) / step_cost(cur, n_rows);
        if (best != cur && step_held_ >= kHold && best_g > cur_g * (1.0 + kHyst)) {
            RAD_DEBUG("draft window: head depth %d -> %d over %d drafting of %d rows (%.3f -> "
                      "%.3f tokens per unit of step cost)", step_d_, best, n_seq, n_rows, cur_g,
                      best_g);
            step_d_ = best;
            step_held_ = 0;
        }
    }
    step_w_ = step_d_ < top ? step_d_ : top;
    return step_d_;
}

}  /* namespace rad */
