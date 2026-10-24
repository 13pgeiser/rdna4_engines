/* tests/spechub_accept_test.cpp -- the coupling's acceptance against the greedy drafter's.
 *
 * spechub_couple_test pins the output law. This file pins the acceptance rate, against the rule the
 * engine applies without the coupling, over the same candidate stream and uniforms:
 *
 *   greedy    the head proposes its argmax and the draft survives when it equals the target's own
 *             sampled token. The rate is P(argmax Q).
 *   coupled   the head proposes x ~ Q and rad_mtp_couple_decide keeps it with probability
 *             min(1, P(x)/Q(x)). The rate is sum_i min(P_i, Q_i).
 *
 * Neither rate dominates. A flat draft over a spread target favours the coupling; a peaked target
 * whose argmax the head shares favours the greedy draft, because Q is the head's law at temperature
 * 1 whatever the request's temperature is. Both cases are checked against their closed forms.
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

std::vector<double> softmax(const std::vector<float>& v) {
    const float top = *std::max_element(v.begin(), v.end());
    std::vector<double> p(v.size());
    double sum = 0;
    for (size_t i = 0; i < v.size(); ++i) { p[i] = std::exp((double)v[i] - top); sum += p[i]; }
    for (double& x : p) x /= sum;
    return p;
}

/* Inverse CDF over a law with a uniform in [0, 1). */
int32_t pick(const std::vector<double>& p, double u) {
    double c = u;
    for (size_t i = 0; i < p.size(); ++i) { c -= p[i]; if (c < 0) return (int32_t)i; }
    return (int32_t)p.size() - 1;
}

struct Rates { double greedy, coupled, want_greedy, want_coupled; int64_t n; };

/* Both rules over `trials` positions of one (target, head) pair; ids are 0..7 in both. */
Rates measure(const std::vector<float>& pval, const std::vector<float>& qval, int trials) {
    std::vector<int32_t> ids;
    for (int i = 0; i < 8; ++i) ids.push_back(i);
    const std::vector<double> P = softmax(pval), Q = softmax(qval);
    const int32_t argmax_q = (int32_t)(std::max_element(qval.begin(), qval.end()) - qval.begin());
    const uint64_t seed = 90210;
    int64_t greedy_ok = 0, coupled_ok = 0;
    for (int k = 0; k < trials; ++k) {
        const uint64_t pos = (uint64_t)k;
        /* The target's own draw at this position, independent of the head. */
        if (pick(P, rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_PICK)) == argmax_q) ++greedy_ok;

        RadMtpCoupleState s{};
        s.draw.seed = seed;
        s.draw.base = pos;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        float qv[8];
        for (int i = 0; i < 8; ++i) qv[i] = qval[(size_t)i];
        CHECK(rad_mtp_couple_build_q(ids.data(), qv, 8, &s));
        s.x = s.ids[rad_mtp_couple_pick(
            s, rad_spechub_uniform(seed, pos, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), pval.data(), 8, 8,
                                    rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), &out));
        if (out == s.x) ++coupled_ok;
    }
    Rates r{};
    r.n = trials;
    r.greedy = (double)greedy_ok / trials;
    r.coupled = (double)coupled_ok / trials;
    r.want_greedy = P[(size_t)argmax_q];
    for (size_t i = 0; i < P.size(); ++i) r.want_coupled += std::min(P[i], Q[i]);
    std::fprintf(stderr, "  acceptance over %d positions: greedy %.4f (P(argmax Q) %.4f), "
                         "coupled %.4f (sum min(P, Q) %.4f)\n",
                 trials, r.greedy, r.want_greedy, r.coupled, r.want_coupled);
    return r;
}

}  /* namespace */

/* A spread target and a flatter head: the coupling accepts more than the greedy draft. */
TEST(the_coupling_accepts_more_than_the_greedy_draft_on_a_spread_target) {
    const Rates r = measure({ 3.10f, 2.95f, 2.40f, 1.80f, 1.10f, 0.40f, -0.60f, -2.00f },
                            { 2.60f, 2.55f, 2.30f, 2.00f, 1.60f, 1.20f, 0.60f, -0.40f }, 20000);
    const double sig = std::sqrt(0.25 / (double)r.n);
    CHECK_NEAR(r.greedy, r.want_greedy, 5.0 * sig);
    CHECK_NEAR(r.coupled, r.want_coupled, 5.0 * sig);
    CHECK(r.want_coupled > r.want_greedy + 10.0 * sig);
    CHECK(r.coupled > r.greedy);
}

/* A peaked target (a low temperature) whose argmax the head shares: the greedy draft is accepted
 * almost always, while x ~ Q at temperature 1 often proposes a token the target rarely draws. */
TEST(the_greedy_draft_accepts_more_on_a_peaked_target) {
    const Rates r = measure({ 6.00f, 2.00f, 1.50f, 1.00f, 0.50f, 0.00f, -1.00f, -2.00f },
                            { 2.00f, 1.20f, 1.00f, 0.80f, 0.40f, 0.00f, -0.50f, -1.00f }, 20000);
    const double sig = std::sqrt(0.25 / (double)r.n);
    CHECK_NEAR(r.greedy, r.want_greedy, 5.0 * sig);
    CHECK_NEAR(r.coupled, r.want_coupled, 5.0 * sig);
    CHECK(r.want_greedy > r.want_coupled + 10.0 * sig);
    CHECK(r.greedy > r.coupled);
}

/* The proposal is a draw from Q, the FP64 softmax of the shortlist, not its argmax: over many
 * rounds every candidate with mass appears at its frequency under Q. */
TEST(the_drafted_token_is_sampled_from_the_heads_own_law) {
    const std::vector<float> qval = { 1.0f, 0.4f, -0.3f, -1.2f };
    int32_t qidx[8];
    float   qv[8];
    for (int i = 0; i < 8; ++i) { qidx[i] = i < 4 ? i : -1; qv[i] = i < 4 ? qval[(size_t)i] : -INFINITY; }

    double sum = 0, Q[4] = {};
    for (int i = 0; i < 4; ++i) { Q[i] = std::exp((double)qval[(size_t)i] - 1.0); sum += Q[i]; }
    for (int i = 0; i < 4; ++i) Q[i] /= sum;

    const int trials = 20000;
    int64_t seen[4] = {};
    for (int k = 0; k < trials; ++k) {
        RadMtpCoupleState s{};
        s.draw.seed = 7;
        s.draw.base = (uint64_t)k;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        CHECK(rad_mtp_couple_build_q(qidx, qv, 8, &s));
        const int32_t x = s.ids[rad_mtp_couple_pick(
            s, rad_spechub_uniform(7, (uint64_t)k, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        CHECK(x >= 0 && x < 4);
        ++seen[x];
    }
    for (int i = 0; i < 4; ++i) {
        const double got = (double)seen[i] / (double)trials;
        const double sigma = std::sqrt(Q[i] * (1.0 - Q[i]) / (double)trials);
        CHECK_NEAR(got, Q[i], 5.0 * sigma + 1e-9);
    }
}

RAD_TEST_MAIN()
