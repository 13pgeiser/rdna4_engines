/* tests/spechub_dflash_couple_test.cpp -- DFlash2's coupled walk and the verify it feeds, checked
 * against the target's own law.
 *
 * The rule under test (abi/rad_spechub.h): the drafter samples x ~ Q from a law it states exactly,
 * the verify keeps x with probability min(1, P(x)/Q(x)) on its own uniform and otherwise draws from
 * norm((P - Q)+), and the bonus position is the ordinary pick. Then every emitted token is
 * distributed as P, the distribution the non-speculative sampler draws from.
 *
 *   1. NAMED (P, Q) PAIRS. The emitted token against P by chi-square, and the acceptance
 *      rate against sum min(P, Q), for the shapes that break a wrong rule: Q = P, Q on fewer tokens
 *      than P, Q mostly outside P's support, and P cut by top-k, top-p and a temperature the way the
 *      sampler's candidate plane is cut (index -1, value -inf).
 *
 *   Each chi-square runs over three independent seeds: every p-value above 0.001 and their Fisher
 *   combination above 0.01 (fisher_ok).
 *
 *   2. THE WALK'S Q. rad_dflash_couple_walk's state q[] is exactly the softmax of its edge scores
 *      over the best eight (recomputed here independently), and the drafted x follows it.
 *
 *   3. THE WHOLE LOOP. A synthetic block drafter (unary scores from the committed text only, the
 *      pairwise term from the walk's own previous token) drafts three positions a step through
 *      rad_dflash_couple_walk; the verify runs at n_out + k with the bound state, the first
 *      rejection ends the step, the bonus row is the ordinary pick. The joint law of the first
 *      three emitted tokens must be the target's autoregressive law, also when some positions fall
 *      back to the ordinary verify (status -1) and when Q is sharpened (q_temp < 1).
 *
 *   4. A greedy request's row keeps the selector's tokens (status 0).
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace {

/* Upper tail of a chi-square with k degrees of freedom (Wilson-Hilferty). */
double chi2_tail(double x, int k) {
    if (k <= 0) return 1.0;
    const double z = (std::cbrt(x / k) - (1 - 2.0 / (9 * k))) / std::sqrt(2.0 / (9 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

/* P as the verify computes it: the FP64 softmax over live ids at finite values, by token id. */
std::vector<double> target_law(const std::vector<int32_t>& ids, const std::vector<float>& v,
                               int vocab) {
    std::vector<double> P((size_t)vocab, 0.0);
    float top = -INFINITY;
    for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] >= 0 && v[j] > top) top = v[j];
    double s = 0;
    for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] >= 0 && v[j] != -INFINITY) {
            const double w = std::exp((double)v[j] - top);
            P[(size_t)ids[j]] += w;
            s += w;
        }
    for (double& p : P) p /= s;
    return P;
}

/* The sampler's candidate plane over logits `l`: temperature, top-k, then top-p (f32 cumulative in
 * descending order, keeping at least one), dropped entries as (-1, -inf), the plane sorted. */
void cut_plane(const std::vector<float>& l, float temp, int top_k, float top_p,
               std::vector<int32_t>* ids, std::vector<float>* v) {
    const int n = (int)l.size();
    std::vector<int> ord((size_t)n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
    const int k = top_k > 0 && top_k < n ? top_k : n;
    ids->assign((size_t)n, -1);
    v->assign((size_t)n, -INFINITY);
    for (int i = 0; i < k; ++i) { (*ids)[(size_t)i] = ord[(size_t)i]; (*v)[(size_t)i] = l[ord[(size_t)i]] / temp; }
    if (top_p < 1.0f) {
        const float mx = (*v)[0];
        float sum = 0;
        for (int i = 0; i < k; ++i) sum += std::exp((*v)[(size_t)i] - mx);
        float cum = 0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((*v)[(size_t)i] - mx) / sum;
            if (cum >= top_p) { cut = i + 1; break; }
        }
        for (int i = cut; i < k; ++i) { (*ids)[(size_t)i] = -1; (*v)[(size_t)i] = -INFINITY; }
    }
}

/* sample_pick's rule (the bonus row and an uncoupled row): an f32 inverse CDF at the seed's own
 * stream. */
int32_t f32_pick(const std::vector<int32_t>& ids, const std::vector<float>& v, uint64_t seed,
                 uint64_t pos) {
    float mx = -INFINITY, sum = 0;
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] >= 0) mx = std::max(mx, v[i]);
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] >= 0) sum += std::exp(v[i] - mx);
    const float u = rad_sample_uniform01(seed ^ RAD_SAMPLE_KEY_PICK, pos);
    float cum = 0;
    int32_t last = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0) continue;
        if (v[i] > -INFINITY) last = ids[i];
        cum += std::exp(v[i] - mx) / sum;
        if (u < cum) return ids[i];
    }
    return last;
}

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

/* Chi-square of counts against a law; dof counts the cells with mass. A cell without mass must be
 * empty. */
double chi_tail(const std::vector<int64_t>& cnt, const std::vector<double>& P, int64_t n) {
    double chi = 0;
    int dof = -1;
    for (size_t i = 0; i < P.size(); ++i) {
        if (P[i] == 0) { CHECK_EQ(cnt[i], (int64_t)0); continue; }
        const double e = P[i] * (double)n;
        chi += ((double)cnt[i] - e) * ((double)cnt[i] - e) / e;
        ++dof;
    }
    return chi2_tail(chi, dof);
}

/* Independent repeats of one check: every p-value above 0.001, and their Fisher combination,
 * -2 sum ln p ~ chi2(2n), above 0.01. A single fixed seed would pass or fail one chi-square draw;
 * several, combined, separate a biased rule from an unlucky seed. */
bool fisher_ok(const std::vector<double>& p, double* combined) {
    double x = 0;
    bool each = true;
    for (double v : p) { x += -2.0 * std::log(std::max(v, 1e-300)); each = each && v > 0.001; }
    *combined = chi2_tail(x, 2 * (int)p.size());
    return each && *combined > 0.01;
}

/* One named case: the target plane (ids, v) and the draft's eight (id, score) slots. Every trial
 * draws x ~ Q at its own coordinate, then couples. */
void run_case(const char* name, const std::vector<int32_t>& ids, const std::vector<float>& v,
              int vocab, const int32_t* qid, const float* qv, int draws = 100000) {
    std::vector<double> pv;
    for (uint64_t rep = 0; rep < 3; ++rep) {
    const std::vector<double> P = target_law(ids, v, vocab);
    RadMtpCoupleState base{};
    base.draw.seed = rad_sample_splitmix64(0xC0FFEEull + (uint64_t)vocab * 7919 + rep);
    base.draw.mode = 1;
    base.status = 1;
    base.round = 1;
    REQUIRE(rad_mtp_couple_build_q(qid, qv, vocab, &base));
    double want_acc = 0;
    {
        std::vector<double> Q((size_t)vocab, 0.0);
        for (int i = 0; i < base.count; ++i) Q[(size_t)base.ids[i]] = base.q[i];
        for (int i = 0; i < vocab; ++i) want_acc += std::min(P[(size_t)i], Q[(size_t)i]);
    }
    std::vector<int64_t> cnt((size_t)vocab, 0);
    int64_t acc = 0;
    const uint64_t seed = base.draw.seed;
    for (int t = 0; t < draws; ++t) {
        RadMtpCoupleState s = base;
        s.draw.base = (uint64_t)t;
        s.x = s.ids[rad_mtp_couple_pick(s, rad_spechub_uniform(seed, s.draw.base,
                                                                RAD_MTP_COUPLE_KEY_DRAFT + 1))];
        int32_t out = -1;
        REQUIRE(rad_mtp_couple_decide(s, ids.data(), v.data(), (int64_t)ids.size(), vocab,
                                      rad_spechub_uniform(seed, (uint64_t)t, RAD_SAMPLE_KEY_ACCEPT),
                                      rad_spechub_uniform(seed, (uint64_t)t, RAD_SAMPLE_KEY_RESID),
                                      &out));
        REQUIRE(out >= 0 && out < vocab);
        ++cnt[(size_t)out];
        acc += out == s.x;
    }
    const double tail = chi_tail(cnt, P, draws);
    const double rate = (double)acc / draws;
    std::fprintf(stderr, "  %-34s chi2 p %.3f  acceptance %.4f (sum min(P,Q) %.4f)\n", name, tail,
                 rate, want_acc);
    pv.push_back(tail);
    /* Q = P sums to one only to rounding, so the variance is clamped at zero. */
    const double sd = std::sqrt(std::max(0.0, want_acc * (1 - want_acc)) / draws) + 1e-9;
    CHECK_NEAR(rate, want_acc, 5.0 * sd);
    }
    double comb = 0;
    const bool ok = fisher_ok(pv, &comb);
    std::fprintf(stderr, "  %-34s combined p %.3f over %zu seeds\n", name, comb, pv.size());
    CHECK(ok);
}

/* Eight slots from up to eight (id, score) pairs. */
void slots(std::initializer_list<std::pair<int, float>> in, int32_t* id, float* v) {
    int n = 0;
    for (const auto& p : in) { id[n] = p.first; v[n] = p.second; ++n; }
    for (; n < 8; ++n) { id[n] = -1; v[n] = -INFINITY; }
}

}  /* namespace */

TEST(the_coupled_token_follows_the_target_for_named_pairs) {
    const int V = 24;
    std::vector<float> l((size_t)V);
    for (int i = 0; i < V; ++i) l[(size_t)i] = 2.0f - 0.35f * (float)i + 0.2f * std::sin((float)i);
    std::vector<int32_t> ids;
    std::vector<float> v;
    int32_t qid[8];
    float qv[8];

    /* Q = P: P on eight tokens (top-k 8) and Q the same scores. Acceptance must be 1. */
    cut_plane(l, 1.0f, 8, 1.0f, &ids, &v);
    slots({{0, l[0]}, {1, l[1]}, {2, l[2]}, {3, l[3]}, {4, l[4]}, {5, l[5]}, {6, l[6]}, {7, l[7]}},
          qid, qv);
    run_case("Q = P", ids, v, V, qid, qv);

    /* Q on two tokens, P on all 24. */
    cut_plane(l, 1.0f, 0, 1.0f, &ids, &v);
    slots({{0, 1.0f}, {3, 0.2f}}, qid, qv);
    run_case("Q support smaller than P", ids, v, V, qid, qv);

    /* Q mostly outside P's support: P is the top 5, Q sits on 4..11. */
    cut_plane(l, 1.0f, 5, 1.0f, &ids, &v);
    slots({{4, 0.0f}, {5, 0.4f}, {6, 0.9f}, {7, 0.1f}, {8, 0.3f}, {9, 0.0f}, {10, 0.2f},
           {11, 0.5f}}, qid, qv);
    run_case("Q mostly outside P (disjoint-ish)", ids, v, V, qid, qv);

    /* The service's settings over a flatter target: top-k 20, top-p 0.95, T 1. Q the first eight at
     * a sharper temperature, as a draft head would be. */
    std::vector<float> flat((size_t)V);
    for (int i = 0; i < V; ++i) flat[(size_t)i] = -0.12f * (float)i;
    cut_plane(flat, 1.0f, 20, 0.95f, &ids, &v);
    slots({{0, 0.0f}, {1, -0.2f}, {2, -0.4f}, {3, -0.6f}, {4, -0.8f}, {5, -1.0f}, {6, -1.2f},
           {7, -1.4f}}, qid, qv);
    run_case("top_k 20 + top_p 0.95, sharp Q", ids, v, V, qid, qv);

    /* top-p alone cutting deep into Q's support. */
    cut_plane(l, 1.0f, 0, 0.6f, &ids, &v);
    slots({{0, l[0]}, {1, l[1]}, {2, l[2]}, {3, l[3]}, {4, l[4]}, {5, l[5]}, {6, l[6]}, {7, l[7]}},
          qid, qv);
    run_case("top_p 0.6 inside Q's support", ids, v, V, qid, qv);

    /* top-k 3 with a temperature, Q flatter than P. */
    cut_plane(l, 0.7f, 3, 1.0f, &ids, &v);
    slots({{0, 0.3f}, {1, 0.2f}, {2, 0.1f}, {3, 0.0f}, {12, 0.0f}}, qid, qv);
    run_case("top_k 3 at T 0.7, flat Q", ids, v, V, qid, qv);

    /* A peaked Q on P's second token. */
    cut_plane(l, 1.0f, 0, 1.0f, &ids, &v);
    slots({{1, 6.0f}, {0, 0.0f}}, qid, qv);
    run_case("peaked Q on P's runner-up", ids, v, V, qid, qv);
}

namespace {

/* A synthetic block drafter over kV tokens. Its candidate lattice for a step depends only on the
 * committed text (a block drafter sees nothing of its own picks); the pairwise codebooks make the
 * walk's Q at position l depend on its own x_{l-1}. */
constexpr int kV = 6, kK = 4, kR = 16, kSteps = 3;

uint64_t ctx_hash(const std::vector<int32_t>& ctx, uint64_t salt) {
    uint64_t h = salt;
    for (int32_t t : ctx) h = rad_sample_splitmix64(h ^ (uint64_t)(t + 1) * 0x100000001b3ull);
    return h;
}

float unit(uint64_t h) { return (float)((double)(h >> 40) / 16777216.0 * 2.0 - 1.0); }

/* The target: hash logits of the context, top 4 at T 0.8 (so the drafter's candidates often lie
 * outside P's support). */
void target_plane(const std::vector<int32_t>& ctx, std::vector<int32_t>* ids,
                  std::vector<float>* v) {
    const uint64_t h = ctx_hash(ctx, 0xabcdefull);
    std::vector<float> l((size_t)kV);
    for (int i = 0; i < kV; ++i) l[(size_t)i] = 2.0f * unit(rad_sample_splitmix64(h + (uint64_t)i));
    cut_plane(l, 0.8f, 4, 1.0f, ids, v);
}

struct Lattice {
    int32_t  cand[kSteps * kK];
    float    unary[kSteps * kK];
    uint16_t hp[kSteps * kR];
};

/* Position l's candidates: kK distinct ids from a hash of the committed text and l, with scores. */
void draft_lattice(const std::vector<int32_t>& committed, Lattice* L) {
    for (int l = 0; l < kSteps; ++l) {
        const uint64_t h = ctx_hash(committed, 0x5151ull + (uint64_t)l);
        int ord[kV];
        for (int i = 0; i < kV; ++i) ord[i] = i;
        for (int i = kV - 1; i > 0; --i)
            std::swap(ord[i], ord[rad_sample_splitmix64(h + 77 + (uint64_t)i) % (uint64_t)(i + 1)]);
        for (int c = 0; c < kK; ++c) {
            L->cand[l * kK + c] = ord[c];
            L->unary[l * kK + c] = 1.5f * unit(rad_sample_splitmix64(h + 1000 + (uint64_t)c));
        }
        for (int r = 0; r < kR; ++r)
            L->hp[l * kR + r] = f2bf(unit(rad_sample_splitmix64(h + 5000 + (uint64_t)r)));
    }
}

struct Codebooks {
    uint16_t pred[kV * kR], succ[kV * kR];
    Codebooks() {
        for (int i = 0; i < kV * kR; ++i) {
            pred[i] = f2bf(unit(rad_sample_splitmix64(0x9000ull + (uint64_t)i)));
            succ[i] = f2bf(unit(rad_sample_splitmix64(0xA000ull + (uint64_t)i)));
        }
    }
};

/* The target's own autoregressive law of the first `n` tokens, by sequence. */
double ar_prob(const std::vector<int32_t>& seq) {
    double p = 1;
    std::vector<int32_t> ctx;
    for (int32_t t : seq) {
        std::vector<int32_t> ids;
        std::vector<float> v;
        target_plane(ctx, &ids, &v);
        p *= target_law(ids, v, kV)[(size_t)t];
        ctx.push_back(t);
    }
    return p;
}

/* The engine's loop over whole steps. `fallback_every` > 0 refuses the draw's card every that many
 * steps (status -1: the walk's argmax and the ordinary verify). Returns the acceptance rate. */
double run_loop(int trials, float q_temp, int fallback_every, uint64_t salt, double* p_value) {
    const int L = 3;
    static const Codebooks cb;
    std::map<std::vector<int32_t>, int64_t> cnt;
    int64_t drafted = 0, accepted = 0;
    for (int t = 0; t < trials; ++t) {
        const uint64_t seed = rad_sample_splitmix64(salt + (uint64_t)t);
        std::vector<int32_t> out;
        int step = 0;
        while ((int)out.size() < L) {
            const uint64_t n_out = out.size();
            Lattice lat;
            draft_lattice(out, &lat);
            RadMtpCoupleDraw d{seed, n_out, 1, 0, 1u};
            const bool card_ok = !(fallback_every > 0 && step % fallback_every == fallback_every - 1);
            int32_t tok[kSteps] = {0, 0, 0};
            RadMtpCoupleState st[kSteps];
            const int32_t anchor = out.empty() ? 0 : out.back();
            rad_dflash_couple_walk(lat.cand, lat.unary, kK, kK, lat.hp, kR, anchor, cb.pred,
                                   cb.succ, kK, kR, kV, kSteps, q_temp, d, card_ok, tok,
                                   reinterpret_cast<uint32_t*>(st), RAD_MTP_COUPLE_STATE_WORDS);
            /* The engine's bind: the verify position after the commit. */
            for (int k = 0; k < kSteps; ++k) st[k].pos = rad_mtp_couple_verify_pos(n_out, k + 1);
            std::vector<int32_t> ctx = out;
            for (int k = 0; k <= kSteps && (int)out.size() < L; ++k) {
                std::vector<int32_t> ids;
                std::vector<float> v;
                target_plane(ctx, &ids, &v);
                const uint64_t pos = n_out + (uint64_t)k;
                int32_t y = -1;
                bool more = false;
                if (k < kSteps) {
                    ++drafted;
                    if (rad_mtp_couple_binds(st[k], 1, seed, n_out, k, tok[k], kV)) {
                        const bool ok = rad_mtp_couple_decide(
                            st[k], ids.data(), v.data(), (int64_t)ids.size(), kV,
                            rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                            rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), &y);
                        CHECK(ok);
                        if (!ok) return 0.0;
                    } else {
                        CHECK(!card_ok);
                        CHECK_EQ(st[k].status, -1);
                        y = f32_pick(ids, v, seed, pos);
                    }
                    more = y == tok[k];
                    accepted += more;
                } else {
                    y = f32_pick(ids, v, seed, pos);
                }
                out.push_back(y);
                ctx.push_back(y);
                if (!more) break;
            }
            ++step;
        }
        out.resize((size_t)L);
        ++cnt[out];
    }
    std::vector<int64_t> c;
    std::vector<double> P;
    for (int code = 0; code < kV * kV * kV; ++code) {
        std::vector<int32_t> seq((size_t)L);
        for (int i = 0, x = code; i < L; ++i, x /= kV) seq[(size_t)i] = x % kV;
        P.push_back(ar_prob(seq));
        const auto it = cnt.find(seq);
        c.push_back(it == cnt.end() ? 0 : it->second);
    }
    *p_value = chi_tail(c, P, trials);
    return drafted ? (double)accepted / (double)drafted : 0.0;
}

}  /* namespace */

/* THE WALK'S Q: the stored law is the softmax of the walk's own scores over its best eight, and x
 * follows it. Scores are recomputed here from the lattice in FP64 (the product of two bf16 values
 * is exact in f32 and in f64), so the comparison is to rounding. */
TEST(the_walk_stores_and_samples_the_softmax_of_its_edge_scores) {
    static const Codebooks cb;
    Lattice lat;
    draft_lattice({2, 5, 1}, &lat);
    const float q_temp = 0.8f;
    const int32_t anchor = 1;
    std::vector<int64_t> cnt((size_t)kV, 0);
    RadMtpCoupleState first{};
    const int draws = 100000;
    for (int t = 0; t < draws; ++t) {
        RadMtpCoupleDraw d{77, (uint64_t)t, 1, 0, 1u};
        int32_t tok[kSteps] = {0, 0, 0};
        RadMtpCoupleState st[kSteps];
        rad_dflash_couple_walk(lat.cand, lat.unary, kK, kK, lat.hp, kR, anchor, cb.pred, cb.succ,
                               kK, kR, kV, kSteps, q_temp, d, true, tok,
                               reinterpret_cast<uint32_t*>(st), RAD_MTP_COUPLE_STATE_WORDS);
        for (int k = 0; k < kSteps; ++k) {
            CHECK_EQ(st[k].status, 1);
            CHECK_EQ(st[k].round, k + 1);
            CHECK_EQ(st[k].x, tok[k]);
            CHECK(rad_mtp_couple_state_valid(st[k], kV));
        }
        if (t == 0) first = st[0];
        ++cnt[(size_t)tok[0]];
    }
    /* Position 0's predecessor is the anchor, so its Q is the same on every trial. */
    double sc[kK], top = -1e300;
    for (int c = 0; c < kK; ++c) {
        double e = 0;
        const int32_t id = lat.cand[c];
        for (int r = 0; r < kR; ++r)
            e += (double)rad_dflash_bf2f(cb.pred[anchor * kR + r]) * rad_dflash_bf2f(lat.hp[r]) *
                 rad_dflash_bf2f(cb.succ[id * kR + r]);
        sc[c] = ((double)lat.unary[c] + e) / q_temp;
        top = std::max(top, sc[c]);
    }
    double z = 0;
    for (int c = 0; c < kK; ++c) z += std::exp(sc[c] - top);
    std::vector<double> Q((size_t)kV, 0.0);
    for (int c = 0; c < kK; ++c) Q[(size_t)lat.cand[c]] = std::exp(sc[c] - top) / z;
    CHECK_EQ(first.count, kK);
    for (int i = 0; i < first.count; ++i)
        CHECK_NEAR(first.q[i], Q[(size_t)first.ids[i]], 1e-5);
    const double tail = chi_tail(cnt, Q, draws);
    std::fprintf(stderr, "  position 0's drafts against the stated Q: chi2 p %.3f\n", tail);
    CHECK(tail > 0.01);
}

namespace {
void loop_case(const char* name, float q_temp, int fallback_every) {
    std::vector<double> pv;
    for (uint64_t rep = 0; rep < 3; ++rep) {
        double p = 0;
        const double acc = run_loop(150000, q_temp, fallback_every,
                                    0x5eed0000ull + rep * 0x100000000ull, &p);
        std::fprintf(stderr, "  joint law of 3 tokens, %s: chi2 p %.3f, acceptance %.3f\n", name,
                     p, acc);
        pv.push_back(p);
    }
    double comb = 0;
    const bool ok = fisher_ok(pv, &comb);
    std::fprintf(stderr, "  joint law of 3 tokens, %s: combined p %.3f over %zu seeds\n", name,
                 comb, pv.size());
    CHECK(ok);
}
}  /* namespace */

TEST(the_coupled_walk_emits_the_target_law_over_steps) {
    loop_case("q_temp 1", 1.0f, 0);
}

TEST(the_coupled_walk_stays_exact_with_a_sharpened_q_and_ordinary_fallback_rows) {
    loop_case("q_temp 0.5", 0.5f, 0);
    loop_case("every 2nd step status -1", 1.0f, 2);
}

TEST(a_greedy_row_keeps_the_selectors_walk) {
    static const Codebooks cb;
    Lattice lat;
    draft_lattice({3}, &lat);
    RadMtpCoupleDraw d{5, 9, 1, 0, 0u};
    int32_t tok[kSteps] = {4, 0, 2};
    RadMtpCoupleState st[kSteps];
    rad_dflash_couple_walk(lat.cand, lat.unary, kK, kK, lat.hp, kR, 3, cb.pred, cb.succ, kK, kR,
                           kV, kSteps, 1.0f, d, true, tok, reinterpret_cast<uint32_t*>(st),
                           RAD_MTP_COUPLE_STATE_WORDS);
    const int32_t want[kSteps] = {4, 0, 2};
    for (int k = 0; k < kSteps; ++k) {
        CHECK_EQ(tok[k], want[k]);
        CHECK_EQ(st[k].status, 0);
        CHECK_EQ(st[k].x, want[k]);
        CHECK_EQ(st[k].round, k + 1);
        CHECK_EQ(st[k].count, 0);
        CHECK(std::memcmp(&st[k].draw, &d, sizeof d) == 0);
    }
}

/* A draw staged for another card (status -1) takes the walk's argmax over its own scores, which is
 * what dflash_select picks over the same lattice up to rounding. */
TEST(a_refused_draw_takes_the_walks_argmax) {
    static const Codebooks cb;
    Lattice lat;
    draft_lattice({0, 4}, &lat);
    RadMtpCoupleDraw d{5, 9, 1, 3, 1u};
    int32_t tok[kSteps] = {-7, -7, -7};
    RadMtpCoupleState st[kSteps];
    rad_dflash_couple_walk(lat.cand, lat.unary, kK, kK, lat.hp, kR, 4, cb.pred, cb.succ, kK, kR,
                           kV, kSteps, 1.0f, d, false, tok, reinterpret_cast<uint32_t*>(st),
                           RAD_MTP_COUPLE_STATE_WORDS);
    int32_t pid = 4;
    for (int k = 0; k < kSteps; ++k) {
        CHECK_EQ(st[k].status, -1);
        CHECK_EQ(st[k].count, 0);
        double best = -1e300;
        int32_t want = -1;
        for (int c = 0; c < kK; ++c) {
            double e = 0;
            const int32_t id = lat.cand[k * kK + c];
            for (int r = 0; r < kR; ++r)
                e += (double)rad_dflash_bf2f(cb.pred[pid * kR + r]) *
                     rad_dflash_bf2f(lat.hp[k * kR + r]) * rad_dflash_bf2f(cb.succ[id * kR + r]);
            const double s = (double)lat.unary[k * kK + c] + e;
            if (s > best + 1e-5) { best = s; want = id; }
        }
        CHECK_EQ(tok[k], want);
        CHECK_EQ(st[k].x, tok[k]);
        pid = tok[k];
    }
}

RAD_TEST_MAIN()
