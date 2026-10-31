/* r3d_qsa_tail_store.hip -- qsa_tail_store: the QSA indexer's tail, raw index keys of tokens that
 * have not yet completed a block, into a per-layer linear KV state.
 *
 * The tail is a per-layer linear KV state owned by the sequence (allocated zeroed, kept for the
 * sequence's life), not a plane indexed by the step-local batch row. `state` is one layer's cache,
 * [slots, 1, ring, n]; `sidx` names the sequence's slot; position p lands at row p % ring.
 *
 * The ring is ratio + n_spec, not ratio: the next step restarts at the first rejected position, so
 * a step may still owe positions back to p0 - ratio + 1 while writing up to p0 + n_spec. A token's
 * sequence comes from cu_seqlens (a linear scan suffices for a handful of sequences). Only the last
 * `ring` tokens of a sequence's chunk write: positions p and p + ring share a row, so writing all
 * of a prefill chunk would race.
 *
 *   k [T][n] bf16, pos [T] i32, cu [seqs+1] i32, state [slots,1,ring,n] bf16 (inout), sidx [seqs]
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void qsa_tail_store_kernel(
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ sidx,
    uint16_t* __restrict__ state,
    int T, int n, int ring, int seqs, int k_ld, int s_slot, int s_row, int sidx_ld) {
    const int t = (int)blockIdx.x;
    if (t >= T) return;

    // This token's sequence, by a linear scan in every thread (cheaper than LDS plus a barrier).
    int s = -1;
    for (int i = 0; i < seqs; ++i)
        if (t >= cu[i] && t < cu[i + 1]) { s = i; break; }
    if (s < 0) return;

    // Only the last `ring` tokens of this sequence's chunk.
    if (t < cu[s + 1] - ring) return;

    // Column 0 is the committed slot; -1 marks a sequence with no state.
    const int si = sidx[(size_t)s * sidx_ld];
    if (si < 0) return;

    const int row = pos[t] % ring;
    const size_t src = (size_t)t * k_ld;
    const size_t dst = (size_t)si * s_slot + (size_t)row * s_row;
    for (int d = (int)threadIdx.x; d < n; d += kThreads) state[dst + d] = k[src + d];
}

}  // namespace

extern "C" {

int r3d_rad_qsa_tail_store(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* k = r3d_opt(a, 0), *po = r3d_opt(a, 1), *cu = r3d_opt(a, 2),
                    *st = r3d_opt(a, 3), *sx = r3d_opt(a, 4);
    if (!r3d_p(k) || !r3d_p(po) || !r3d_p(cu) || !r3d_wp(st) || !r3d_p(sx))
        return R3D_NO(RAD_E_INVAL);
    if (k->rank != 2) return R3D_NO(RAD_E_SHAPE);
    // The state arrives as the group declared it: [slots, heads, ring, n], heads 1.
    if (st->rank != 4 || st->shape[1] != 1) return R3D_NO(RAD_E_SHAPE);
    if (st->stride[3] != 1) return R3D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || st->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long ratio = r3d_geti_or(a, "ratio", 0);
    const long long ring = r3d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R3D_NO(RAD_E_INVAL);
    const long long T = k->shape[0];
    if (T < 1 || k->shape[1] != n) return R3D_NO(RAD_E_SHAPE);
    if (st->shape[2] < ring || st->shape[3] != n) return R3D_NO(RAD_E_SHAPE);
    const long long seqs = r3d_numel(cu) - 1;
    if (seqs < 1 || r3d_numel(po) < T) return R3D_NO(RAD_E_SHAPE);
    const long long sidx_ld = sx->rank >= 2 ? sx->shape[1] : 1;
    if (sidx_ld < 1 || (sx->rank >= 1 && sx->shape[0] < seqs)) return R3D_NO(RAD_E_SHAPE);
    const long long kp = k->stride[1] ? k->stride[0] / k->stride[1] : k->shape[1];
    if (kp < n) return R3D_NO(RAD_E_STRIDE);

    R3D_KLAUNCH(qsa_tail_store_kernel, dim3((unsigned)T), dim3(kThreads), (hipStream_t)s,
                (const uint16_t*)r3d_p(k), (const int32_t*)r3d_p(po), (const int32_t*)r3d_p(cu),
                (const int32_t*)r3d_p(sx), (uint16_t*)r3d_wp(st), (int)T, (int)n, (int)ring,
                (int)seqs, (int)kp, (int)st->stride[0], (int)st->stride[2], (int)sidx_ld);
    return (int)hipGetLastError();
}

}  // extern "C"
