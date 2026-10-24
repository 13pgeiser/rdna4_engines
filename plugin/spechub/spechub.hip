/* plugin/spechub/spechub.hip -- the device launches for the two coupling ops.
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

}  /* namespace */

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
