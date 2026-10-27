/* plugin/spechub/spechub.hip -- the device launches for the coupling ops.
 *
 * The propose kernel is one thread a row over abi/rad_spechub.h's helpers. The verify is one
 * workgroup a row: the masses are computed in parallel, and the sums and walks run on one thread
 * in ascending candidate order, the order rad_mtp_couple_decide uses, so the token matches the
 * host reference bit for bit.
 *
 * Strict FP64: fast-math or FMA contraction would reorder those sums and change the answer, hence
 * the #error below and -ffp-contract=off in CMakeLists.txt.
 */
#ifdef __FAST_MATH__
#error "the MTP coupling requires strict FP64 arithmetic: build without -ffast-math"
#endif

#include <hip/hip_runtime.h>

#include "spechub.h"
#include "../../libr4d/r4d_emit.h"

namespace {

/* One thread a row: both ops are row-local and the M is the number of sampled positions. */
__global__ void propose(RadMtpCoupleProposeCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_propose_row(c, row);
}

/* One workgroup a row, 128 threads. The weight and residual planes are in LDS: 16 KiB of FP64 at
 * n_cand 1024. */
constexpr int kThreads = 128;
constexpr int64_t kMaxCand = 1024;

__global__ void couple_verify(const int32_t* idx, const float* values, const uint32_t* params,
                              const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                              int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                              int64_t ld4) {
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    /* Above kMaxCand the LDS planes would overflow, so one thread runs the serial helper. */
    if (n_cand > kMaxCand) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row(c, row);
        return;
    }
    /* `weight` is P's unnormalised FP64 mass per candidate and `mass` its residual (P-Q)+. The
     * workgroup writes them; one thread reads them in ascending candidate order. */
    __shared__ double weight[kMaxCand], mass[kMaxCand], pq[8], sum;
    __shared__ float  top;
    __shared__ int    eligible, first[8], ix, dominated;
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float*   val = c.val + row * c.ld[1];
    /* The same eligibility gate as rad_mtp_couple_verify_row: an unbound row keeps the ordinary
     * token. */
    if (threadIdx.x == 0) {
        eligible = 0;
        if (s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
            s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab)) {
            float maximum = -INFINITY;
            bool  valid   = true;
            for (int64_t j = 0; j < c.n_cand; ++j) {
                if (ids[j] < 0) continue;
                if (ids[j] >= c.vocab || val[j] != val[j] || val[j] == INFINITY) {
                    valid = false;
                    break;
                }
                if (val[j] > maximum) maximum = val[j];
            }
            if (valid && maximum != -INFINITY) { top = maximum; eligible = 1; }
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* Each thread writes exp(v - max) for a strided slice; no cross-thread reduction, so the sum
     * below matches the serial order bit for bit. */
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x)
        if (ids[j] >= 0 && val[j] != -INFINITY) weight[j] = ::exp((double)val[j] - top);
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] < 0 || val[j] == -INFINITY) continue;
            const double w = weight[j];
            sum += w;
            for (int k = 0; k < s.count; ++k)
                if (ids[j] == s.ids[k]) {
                    if (first[k] < 0) first[k] = (int)j;
                    pq[k] += w;
                    break;
                }
        }
        if (!(sum > 0) || !rad_spechub_finite(sum)) eligible = 0;
        if (eligible) {
            ix = -1;
            dominated = true;
            for (int k = 0; k < s.count; ++k) {
                pq[k] /= sum;
                dominated = dominated && pq[k] >= s.q[k];
                if (s.ids[k] == s.x) ix = k;
            }
            if (ix < 0) eligible = 0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* Residual per slot: a Q token contributes max(0, P-Q) once, at its first occurrence
     * (`first[k]`); every other candidate contributes its own P. */
    for (int64_t j = threadIdx.x; j < c.n_cand; j += blockDim.x) {
        double m = 0;
        if (ids[j] >= 0 && val[j] != -INFINITY) {
            bool in_q = false;
            for (int k = 0; k < s.count; ++k)
                if (s.ids[k] == ids[j]) {
                    in_q = true;
                    if (first[k] == (int)j) m = rad_spechub_positive(pq[k] - s.q[k]);
                    break;
                }
            if (!in_q) m = weight[j] / sum;
        }
        mass[j] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    /* One thread, ascending candidate index: parallelising these sums would reorder them and
     * break bit-exactness with the host reference. */
    double resid = 0;
    for (int64_t j = 0; j < c.n_cand; ++j) resid += mass[j];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int64_t j = 0; j < c.n_cand; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = weight[j];
                if (w > 0) { out = ids[j]; cdf -= w; if (cdf < 0) break; }
            }
    } else if (pq[ix] >= s.q[ix] ||
               rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT) * s.q[ix] < pq[ix]) {
        out = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int64_t j = 0; j < c.n_cand; ++j) {
            if (ids[j] == s.x) continue;
            const double m = mass[j];
            if (m > 0) { out = ids[j]; cdf -= m; if (cdf < 0) break; }
        }
    }
    c.token[row * c.ld[4]] = out;
}

/* DFlash2's coupled walk: one workgroup a sequence. Per step the workgroup forms the
 * candidate-independent half `sa` of the edge, then thread (c, j) computes part j of candidate c's
 * edge (rad_dflash_edge_part, the host's serial fmaf chain over the same slice), and thread 0 adds
 * the parts in order, takes the step (rad_dflash_couple_step) and publishes the token as the next
 * predecessor. The arithmetic and its order are the host reference's, so tokens and states match
 * rad_ref_dflash_couple bit for bit. */
constexpr int kDfThreads = RAD_DFLASH_COUPLE_MAX_K * RAD_DFLASH_EDGE_PARTS;   /* 256 */

__global__ __launch_bounds__(kDfThreads) void dflash_couple(
    const int32_t* cand, const float* unary, const uint16_t* hp, const int32_t* anchor,
    const uint16_t* pred, const uint16_t* succ, const int32_t* card, const uint32_t* draw,
    int32_t* tokens, uint32_t* state, int64_t steps, int64_t K, int64_t R, int64_t vocab,
    int64_t astr, int64_t ld_cand, int64_t ld_unary, int64_t ld_hp, int64_t ld_draw,
    int64_t ld_tok, int64_t st_round, int64_t st_row, float q_temp) {
    __shared__ float   sa[RAD_DFLASH_COUPLE_MAX_R];
    __shared__ float   part[kDfThreads];
    __shared__ float   sc[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t sid[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t spid;
    /* The step's Q in the making: its live (id, score / q_temp) in selection order, their count
     * (-1: no Q), the top score, the FP64 weights, their sum and the normalised q. */
    __shared__ int32_t lid[8], slot_id[8];
    __shared__ float   lv[8], slot_v[8];
    __shared__ int     sn;
    __shared__ float   stop;
    __shared__ double  w[8], wq[8], ssum;
    __shared__ RadMtpCoupleState sst;
    const int64_t s   = blockIdx.x;
    const int     tid = threadIdx.x;
    const RadMtpCoupleDraw d = *reinterpret_cast<const RadMtpCoupleDraw*>(draw + s * ld_draw);
    int32_t* tok = tokens + s * ld_tok;
    uint32_t* st0 = state + s * st_row;
    if (d.mode != 1) {
        /* A greedy request: dflash_select's walk stands, and each round records it (status 0). */
        if (tid == 0)
            for (int64_t l = 0; l < steps; ++l)
                (void)rad_dflash_couple_step(sc, sid, (int)K, q_temp, d, false, (int)l + 1, vocab,
                                             tok[l],
                                             reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round));
        return;
    }
    const bool card_ok = d.card == card[s];
    if (tid == 0) spid = anchor[s * astr];
    __syncthreads();
    const int c = tid / RAD_DFLASH_EDGE_PARTS, j = tid % RAD_DFLASH_EDGE_PARTS;
    for (int64_t l = 0; l < steps; ++l) {
        const int64_t row = s * steps + l;
        /* What does not depend on the predecessor is loaded first, so its latency overlaps the
         * predecessor's codebook row: the candidate, its successor slice and its unary score. */
        const int32_t id = c < K ? cand[row * ld_cand + c] : -1;
        const bool    id_ok = id >= 0 && id < vocab;
        const bool    wide = R == 32 * RAD_DFLASH_EDGE_PARTS;
        uint4 v[4] = {};
        if (id_ok && wide) {
            /* DFlash2's rank 256: the slice's 32 values in four 16-byte loads issued together. A
             * load inside the fmaf chain would wait out one memory latency per element. */
            const uint4* src = reinterpret_cast<const uint4*>(succ + (int64_t)id * R + j * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = src[i];
        }
        const float un = tid < K ? unary[row * ld_unary + tid] : 0.0f;
        const int32_t pid = spid;
        const bool ok = pid >= 0 && pid < vocab;
        for (int64_t r = tid; r < R; r += kDfThreads)
            sa[r] = ok ? rad_dflash_bf2f(pred[(int64_t)pid * R + r]) *
                             (hp ? rad_dflash_bf2f(hp[row * ld_hp + r]) : 1.0f)
                       : 0.0f;
        __syncthreads();
        if (c < K) {
            float pv = 0.0f;
            if (id_ok) {
                if (wide) {
                    /* rad_dflash_edge_part's fmaf chain over the slice, in its order. */
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(v);
                    const float* a = sa + j * 32;
#pragma unroll
                    for (int r = 0; r < 32; ++r) pv = fmaf(a[r], rad_dflash_bf2f(h[r]), pv);
                } else {
                    pv = rad_dflash_edge_part(sa, succ + (int64_t)id * R, R, j);
                }
            }
            part[tid] = pv;
            if (j == 0) sid[c] = id;
        }
        __syncthreads();
        /* Each candidate's score from its parts, in the host's order. */
        if (tid < K) {
            float acc = 0.0f;
            if (sid[tid] >= 0 && sid[tid] < vocab)
                for (int q = 0; q < RAD_DFLASH_EDGE_PARTS; ++q)
                    acc += part[tid * RAD_DFLASH_EDGE_PARTS + q];
            sc[tid] = un + acc;
        }
        if (tid < 8) slot_id[tid] = -2;
        __syncthreads();
        RadMtpCoupleState* so = reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round);
        /* rad_dflash_couple_step, spread over the workgroup: one lane walking the selection and
         * build_q's FP64 work serially waits out an LDS or FP64 latency per element. Each operation
         * and its order are the helper's, so the bits are too.
         *
         * The eight: candidate k's place is the number of eligible candidates ahead of it (a higher
         * score, or an equal one at a lower index), which is exactly the order the helper's repeated
         * first-maximum scan takes them in. */
        if (tid < K) {
            const int32_t id = sid[tid];
            const float   v  = sc[tid];
            if (id >= 0 && id < vocab && v == v) {
                int rank = 0;
                for (int k = 0; k < K; ++k) {
                    const int32_t ik = sid[k];
                    const float   vk = sc[k];
                    rank += (ik >= 0 && ik < vocab && vk == vk && (vk > v || (vk == v && k < tid)));
                }
                if (rank < 8) { slot_id[rank] = id; slot_v[rank] = v / q_temp; }
            }
        }
        __syncthreads();
        if (tid == 0) {
            sn = -1;
            if (card_ok) {
                int live = 0;
                bool bad = false;
                for (int n = 0; n < 8 && slot_id[n] != -2; ++n) {
                    const float v = slot_v[n];
                    /* build_q: +inf refuses the whole Q, -inf carries no mass, a repeat refuses. */
                    if (v == INFINITY) { bad = true; break; }
                    if (v == -INFINITY) continue;
                    for (int i = 0; i < live; ++i) bad = bad || lid[i] == slot_id[n];
                    if (bad) break;
                    lid[live] = slot_id[n];
                    lv[live++] = v;
                }
                if (!bad && live > 0) {
                    float top = lv[0];
                    for (int i = 1; i < live; ++i)
                        if (lv[i] > top) top = lv[i];
                    stop = top;
                    sn = live;
                }
            }
        }
        __syncthreads();
        if (tid < 8 && tid < sn) w[tid] = ::exp((double)lv[tid] - stop);
        __syncthreads();
        if (tid == 0 && sn > 0) {
            double sum = 0;
            for (int i = 0; i < sn; ++i) sum += w[i];
            ssum = sum;
        }
        __syncthreads();
        if (tid < 8 && tid < sn) wq[tid] = w[tid] / ssum;
        __syncthreads();
        /* The state is built in LDS: a private struct indexed by a runtime count lives in scratch,
         * and every access to it is a memory round trip. */
        if (tid == 0) {
            RadMtpCoupleState& s = sst;
            s.draw = d;
            s.pos = 0;
            s.round = (int)l + 1;
            s.status = -1;
            s.count = 0;
            for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
            for (int i = 0; i < sn; ++i)
                if (wq[i] > 0) { s.ids[s.count] = lid[i]; s.q[s.count++] = wq[i]; }
            if (s.count > 0) {
                const double u = rad_spechub_uniform(d.seed, d.base,
                                                     RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)(l + 1));
                s.x = s.ids[rad_mtp_couple_pick(s, u)];
                s.status = 1;
            } else {
                s.count = 0;
                for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
                int   best = 0;
                float bv = -INFINITY;
                for (int k = 0; k < K; ++k)
                    if (sc[k] > bv) { bv = sc[k]; best = k; }
                s.x = sid[best];
            }
            tok[l] = s.x;
            spid = s.x;
        }
        __syncthreads();
        /* One word a lane out to the state plane. */
        if (tid < (int)RAD_MTP_COUPLE_STATE_WORDS)
            reinterpret_cast<uint32_t*>(so)[tid] = reinterpret_cast<const uint32_t*>(&sst)[tid];
        /* The next step reads spid and overwrites sa, part and the Q scratch (the copy above reads
         * sst, which the next step writes only after its own barriers). */
    }
}

}  /* namespace */

int rad_mine_dflash_couple_launch(const RadArgs* a, RadStream stream) {
    RadDflashCoupleCall c{};
    const int rc = rad_dflash_couple_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, dflash_couple, dim3((unsigned)c.M), dim3(kDfThreads), 0,
                    c.cand, c.unary, c.hp, c.anchor, c.pred, c.succ, c.card, c.draw, c.tokens,
                    c.state, c.steps, c.K, c.R, c.vocab, c.anchor_stride, c.ld_cand, c.ld_unary,
                    c.ld_hp, c.ld_draw, c.ld_tok, c.st_round, c.st_row, c.q_temp);
}

int rad_mine_mtp_couple_propose_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, propose, dim3((unsigned)((c.M + 63) / 64)), dim3(64), 0, c);
}

int rad_mine_mtp_couple_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    /* Separate pointer arguments let graph replay rebind each operand address independently: a
     * struct on the host would freeze the addresses it was built with. */
    return R4D_EMIT(stream, couple_verify, dim3((unsigned)c.M), dim3(kThreads), 0,
                    c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                    c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
}
