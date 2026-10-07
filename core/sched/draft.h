/* draft.h -- the speculative draft-depth controller.
 *
 * BY DEFAULT IT IS STATIC. The depth is `--num-speculative-tokens` and nothing moves it.
 *
 * No dynamic policy sits there, and the reason is structural rather than a tuning failure. The
 * dynamic policy this engine is otherwise modelled on -- a confidence-product gate against a
 * decayed per-position acceptance estimate, plus a batch-size ceiling schedule -- gates on the
 * DRAFTER's own top-1 confidences, per slot, on device, after the drafter has run. The scheduler
 * decides the depth BEFORE the drafter runs and cannot see them; the only substitute available
 * here is measured acceptance, which is the right quantity in expectation but arrives one step
 * late and pooled across sequences. Against a draft head whose cost is nearly flat in depth (one
 * extra verify row on a step that is weight-bound, not compute-bound) that estimate has nothing to
 * buy: the best depth is a property of the model pair, it is known before the serve starts, and it
 * is a flag. A controller that can only pick a worse depth than the constant contributes only the
 * risk that it picks one.
 *
 * What survives is the accounting, because the acceptance rate is a reported metric (spec §14)
 * and because a verify that accepts nothing has to be visible.
 *
 * UNDER `--draft-window adaptive` THE PREMISE ABOVE IS WHAT DOES NOT HOLD. On a model whose
 * weights stream -- a MoE that reads experts per verify row, a serial head that runs one pass a
 * round -- a step is NOT flat in depth: each draft is one more round of the head and one more
 * verify row. And the best depth is not a property of the model pair but of the TEXT:
 * per-position acceptance stays high deep into the window on code and JSON and falls off after a
 * position or two on prose and summaries. No constant is right for all of them, and the one that
 * is right for one costs the others either rounds nobody accepts or tokens nobody drafted.
 *
 * So the adaptive window is a policy on the host, in two halves:
 *
 *   PER SEQUENCE (DraftAdapt): a smoothed estimate of each draft position's conditional
 *   acceptance, fed by the same verify outcomes observe() counts, and the sequence's STOP -- the
 *   deepest draft whose chance of being emitted still pays for the verify row it adds
 *   (kDraftMinChance).
 *
 *   PER STEP (plan_step): the HEAD DEPTH D. A serial head runs as many rounds as the deepest
 *   window in the step, so depth is a property of the step and not of one sequence: D maximises
 *   the tokens the whole step is expected to emit, sum_i E_i(min(D, stop_i)), over what a step
 *   at that depth and that many sequences costs, T(D, n) -- with E(d) = 1 + sum_k prod_{j<=k} a_j
 *   and T measured online (step_time). Every sequence then drafts min(D, stop_i).
 *
 * Both move only past a hold, and D also past a hysteresis, because a window is a pass shape and
 * a new shape is a new recording before it plays (runtime/ctx.h). `max_depth` stays the ceiling,
 * and everything sized for it stays sized for it.
 *
 * IT IS THE SERIAL DRAFTER'S POLICY. A block drafter proposes its whole block in one pass, and
 * prompt lookup runs no pass at all, so neither pays the rounds the policy prices; the scheduler
 * turns it on only for a serial drafter (Scheduler::init) and says so when it does not.
 *
 * IT CHANGES HOW MANY DRAFTS A STEP VERIFIES AND NOTHING ELSE. Acceptance is the rule in
 * core/sample/accept.h, so what a step emits is the target's own token at every position whatever
 * window it verified. At temperature 0 that is the same text at every window, to exactly the
 * degree a speculative run matches a plain one at all: exact given the same logits, and a verify
 * row's logits are computed at an M that now moves with the window (accept.h says what that does
 * and does not change).
 */
#pragma once
#include "rad_core.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows precisely because a verify step falls in that
 * band (spec §10), so 64 is the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* ONE SEQUENCE'S HALF OF THE ADAPTIVE POLICY: what the drafter has done on this sequence's own
 * text, and how deep that makes it worth drafting. It lives in the scheduler's per-request state
 * and the controller does the arithmetic, so neither has to look the other up.
 *
 * `depth` 0 is "not started": the first ask fills the estimate from the prior and sets the stop.
 * It belongs to the REQUEST, not to the cache it holds -- a preemption keeps it, because the text
 * it describes is still the text the request is writing -- and a recycled slot starts it over. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the sequence's stop; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int  max_depth = 0;       /* the deployment's window; 0 disables speculation entirely */
        /* THE ADAPTIVE POLICY (see the top of this file): `--draft-window adaptive` with a serial
         * drafter, which Scheduler::init decides. Off, the depth is max_depth, as it always was. */
        bool adapt = false;
    };

    /* ---------------------------------------------------------- the policy's constants
     * Named rather than configurable: each is a property of how the policy works, not of a
     * deployment, and the one property that IS a deployment's -- what a step costs -- is
     * measured (step_time). */

    /* A WINDOW MOVES ONLY FOR TWO PERCENT MORE YIELD. One draft more or less moves E/T by several
     * percent on any drafter worth speculating with, while a smoothed estimate wobbles by about
     * a percent verify to verify; below two percent a move would flip D between two near-equal
     * neighbours and pay a recording each time. */
    static constexpr double kHyst = 0.02;
    /* AND NOT WITHIN EIGHT VERIFIES (steps, for D) OF THE LAST MOVE. A pass shape is recorded the
     * second time its key is seen and played from the third (runtime/ctx.h), so a shape held for
     * eight steps plays from the tape for most of them; and eight verifies at the half-life below
     * move an estimate by about a third of the way to a new rate, which is a trend rather than
     * one rejection. */
    static constexpr int kHold = 8;
    /* EACH POSITION'S ESTIMATE HAS A HALF-LIFE OF SIXTEEN VERIFIES THAT EVALUATE IT. At a few
     * tokens a verify that is some fifty tokens of text: short enough to follow a reply from prose
     * into a code block and back, long enough that a single rejection moves an estimate by four
     * percent rather than resetting it. */
    static constexpr double kHalfLife = 16.0;
    /* A POSITION NOTHING HAS MEASURED STARTS AT 0.8, optimistic on purpose: a window that starts
     * deep evaluates every position and learns all of them, one that starts shallow learns nothing
     * about the positions it never drafts to. 0.8 sits between what draft heads measure on prose
     * and on code at the first positions, so a wrong start is corrected within a hold. */
    static constexpr double kPrior = 0.8;
    /* ONCE FOUR SEQUENCES HAVE FINISHED, A NEW ONE STARTS FROM WHAT THEY ENDED ON (adapt_retire):
     * the mean of their estimates, a running mean and then a moving average with a half-life of 32
     * sequences, so a deployment that moves from one kind of traffic to another follows it within
     * a few dozen requests. Four, because a mean over fewer is one text's. A sequence counts only
     * once it has had a half-life of verifies: before that its estimate is mostly the prior it
     * started from, and folding it back would feed the prior to itself. */
    static constexpr int    kPriorMinSeqs = 4;
    static constexpr double kPriorHalfLifeSeqs = 32.0;
    /* NEVER FEWER THAN ONE DRAFT. A sequence at zero would evaluate no position, so nothing could
     * ever move its estimate off zero again. */
    static constexpr int kMinDepth = 1;
    /* A SEQUENCE STOPS AT THE FIRST DRAFT WHOSE CHANCE OF BEING EMITTED -- every draft before it
     * and itself accepted, prod_{j<=d} a_j -- IS UNDER ONE IN TEN. The draft adds a verify row to
     * the step whatever D is, and that row is the only cost a sequence adds once the head runs D
     * rounds for its deepest neighbour. At one sequence the step-cost prior (1 + 0.06 d) already
     * breaks even near that chance -- one more draft is worth it while p_d / E exceeds the 5-6 %
     * it adds to the step, with E around two -- and a verify row alone costs less than a round
     * plus a row, so the threshold is never stricter than D's own trade. A constant rather than a
     * measurement: a row's share of the step is not separable from the head's rounds in step
     * times keyed by depth, and the measured T(D, n) already prices the rounds. */
    static constexpr double kDraftMinChance = 0.1;
    /* STEP TIMES ARE KEPT IN THESE RUNNING-SEQUENCE CLASSES: 1, 2-3, 4-7, 8-15, 16 and up. A
     * step's cost moves with how many rows share its weight reads, and a class a power of two
     * wide keeps every key fed often enough to mean something. */
    static constexpr int kCostClasses = 5;

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth for the next step. 0 means "do not speculate". Constant by construction:
     * under the adaptive policy it is the CEILING, and window_for() is the window. */
    int  next_depth() const { return p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }
    /* The window this sequence may draft into: min(D, its stop), starting its estimate on the
     * first ask; max_depth when the policy is off. Before any step has chosen D, D is the
     * ceiling. */
    int  window_for(DraftAdapt& s) const;
    /* The same for a sequence whose estimate has started, without starting one: 0 when it has
     * not. For reporting. */
    int  window_of(const DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Moves its
     * estimate, and its stop past the hold. Returns the stop after. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* A SEQUENCE THAT HAS FINISHED: its estimate joins the prior the next sequences start from.
     *
     * PER SEQUENCE, NOT PER VERIFY. The deployment's own counts -- accepted(j) / reached(j) over
     * every verify -- are what a fixed window would want, and under this policy they are biased:
     * only a sequence whose drafts keep being accepted is given a window that reaches the deep
     * positions, so those positions are measured almost only on the texts that predict well, and
     * a new sequence would start as if every text did. Each finished sequence counts once here, at
     * every position, and at a position it never reached its estimate is the extrapolation from
     * its own window's edge (adapt_observe) -- a shallow text pulls the deep positions down as
     * much as a deep one holds them up. */
    void adapt_retire(const DraftAdapt& s);
    /* The prior a sequence starting now is given at position `j`. For tests and logs. */
    double prior_at(int j) const;

    /* D FOR THE STEP THAT IS OPENING. `seq` are the estimates of the `n_seq` sequences that will
     * draft in it, `n_rows` the decode rows it carries in all (the rest emit one token each and
     * draft nothing). Started here if they have not been. Moves D past the hold and the
     * hysteresis; with no sequence to draft, D stays. Returns D. */
    int  plan_step(DraftAdapt* const* seq, int n_seq, int n_rows);
    /* The step depth in force: 0 before the first step chose one. */
    int  step_depth() const { return step_d_; }
    /* The deepest window a step at that depth carries: D, but no deeper than the deepest stop it
     * was chosen over, since no head-drafting sequence drafts past its stop; the ceiling before
     * any step chose one, and with the policy off. What a proposal from outside the head (prompt
     * lookup) is held to, so it adds no deeper pass shape than the head's own. */
    int  step_window() const { return adapt_ && step_w_ > 0 ? step_w_ : p_.max_depth; }

    /* T(depth, n): a decode step whose deepest window was `depth`, with `n_rows` decode rows,
     * took `ms` from the end of the step before it. Fed by Scheduler::step_finished; only the
     * adaptive policy reads it. */
    void step_time(int depth, int n_rows, double ms);
    /* What a step at `depth` with `n_rows` rows is expected to cost, in ms when anything has
     * been measured: the key's own mean when it has samples; the nearest measured key scaled by
     * the prior's ratio (1 + 0.06 d) when it does not; the prior itself when nothing has been
     * measured. */
    double step_cost(int depth, int n_rows) const;
    /* Samples behind a key: for tests and logs. */
    int64_t cost_samples(int depth, int n_rows) const;

    /* E(d) for `s`'s estimate as it stands: the tokens a verify of d drafts is expected to emit,
     * the verified token included. For tests and logs. */
    double yield(const DraftAdapt& s, int d) const;
    /* Verifies by the window they carried, counted only under the policy: the histogram of the
     * depths it actually chose, which the mean gauge on its own cannot show. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* One verify's outcome: `drafted` tokens were proposed and a prefix of `accepted` of them was
     * taken. Feeds the reported metrics and nothing else. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* PER-POSITION ACCEPTANCE, and it is a diagnostic rather than a policy input.
     *
     * The pooled ratio above cannot tell a drafter that decays from one that falls off a cliff,
     * and the difference is the difference between "this is what the head is worth" and "round
     * three is broken". A drafter whose accepted-tokens-per-step stops rising past some depth has
     * a hazard that goes to zero there; pooled over the whole window that reads as a uniformly
     * mediocre drafter, which is a different diagnosis and a different fix.
     *
     * `reached` is "every earlier position was accepted, so this one was evaluated"; `hits` is
     * "and it was accepted too". Their ratio is the conditional hazard. A position the draft
     * never reached is charged nothing, which is why a rejection at position 0 does not libel
     * every position behind it. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has ever reached; how much of the two arrays is meaningful. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  alpha_ = 0.0;                         /* the smoothing step from kHalfLife */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t step_d_ = 0;                          /* D; 0 until a step chose one */
    int32_t step_w_ = 0;                          /* min(D, the deepest stop it saw) */
    int32_t step_held_ = 0;                       /* steps since D last moved */
    /* The prior from finished sequences (adapt_retire), and how many are behind it. */
    float   prior_[RAD_SCHED_MAX_SPEC] = {};
    int64_t prior_n_ = 0;
    /* T(d, class): the mean step time of each key, and the steps behind it. */
    double  cost_ms_[RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};
    int64_t cost_n_ [RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};

    /* Fill a sequence's estimate from the prior and set its first stop. */
    void    start(DraftAdapt& s) const;
    /* The deepest draft whose chance of being emitted is at least kDraftMinChance. */
    int     stop_of(const float* a) const;
    static int cost_class(int n_rows);
};

}  /* namespace rad */
