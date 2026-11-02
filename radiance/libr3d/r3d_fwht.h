/* r3d_fwht.h -- a wave-local fast Walsh-Hadamard transform in registers, no LDS, no barrier.
 *
 * One wave holds a HAD-wide block, VPL = HAD/32 contiguous values per lane, so lane l holds
 * indices l*VPL .. l*VPL+VPL-1. A butterfly at stride h pairs index i with i+h:
 *
 *   h < VPL    both indices in the same lane     -- plain register arithmetic
 *   h >= VPL   the partner is lane l + h/VPL     -- one lane exchange with mask h/VPL
 *
 * HAD/VPL is always 32, so the cross-lane masks are 1, 2, 4, 8, 16 at any HAD: five exchanges per
 * register and no barrier. The lane holding the low index keeps (v + partner), the high one
 * (partner - v). The caller still needs one barrier before a whole-row reduction.
 *
 * Unnormalised: the caller folds 1/sqrt(HAD) into whatever scale it already applies.
 */
#pragma once
#include <hip/hip_runtime.h>
#include "r3d_common.h"

/* `hi ? (o - v) : (v + o)` as `o + (hi ? -v : v)`, the sign flip one XOR against a stage-uniform
 * mask, so there is no per-element select. Bit-identical: negation is exact, addition commutes. */
template <int VPL, int MASK>
__device__ __forceinline__ void r3d_fwht_xstage(float* v, int lane)
{
    const unsigned int sm = (lane & MASK) ? 0x80000000u : 0u;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
        const float o = r3d_xor_lane<MASK>(v[j]);
        v[j] = __int_as_float(__float_as_uint(v[j]) ^ sm) + o;
    }
}

/* In-register FWHT of a HAD-wide block held VPL = HAD/32 values per lane, lane-major. */
template <int HAD>
__device__ __forceinline__ void r3d_fwht_wave(float* v, int lane)
{
    constexpr int VPL = HAD / 32;
    // in-lane stages: stride 1 .. VPL/2, entirely within this lane's registers
#pragma unroll
    for (int h = 1; h < VPL; h <<= 1) {
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            if ((j & h) == 0) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    // cross-lane stages: stride VPL*mask for mask = 1, 2, 4, 8, 16
    r3d_fwht_xstage<VPL, 1>(v, lane);
    r3d_fwht_xstage<VPL, 2>(v, lane);
    r3d_fwht_xstage<VPL, 4>(v, lane);
    r3d_fwht_xstage<VPL, 8>(v, lane);
    r3d_fwht_xstage<VPL, 16>(v, lane);
}
