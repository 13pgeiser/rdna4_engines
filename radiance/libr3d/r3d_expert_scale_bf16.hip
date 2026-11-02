/* r3d_expert_scale_bf16.hip -- expert_scale: each routing weight times its expert's own scale,
 * in place.
 *
 * The router renormalises its top-k, then multiplies each weight by its expert's per-expert
 * scale. Unfused that is two launches per MoE layer (an embed lookup of the [n_expert, 1] scale
 * table by expert id, then a mul), each a few dozen elements at decode. Fused:
 *
 *     ew[t, k] = bf16( ew[t, k] * scale[ids[t, k]] )
 *
 * Bit-identical to the pair: one f32 product of two bf16 values, rounded once to nearest even.
 * An id outside the table reads a zero scale, so the product's sign matches too. One thread per
 * element; no cross-lane traffic, no wmma.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"

namespace {

__global__ __launch_bounds__(256) void escale_kernel(const int32_t* __restrict__ ids, long long ids_ld,
                                                     const unsigned short* __restrict__ sc,
                                                     unsigned short* __restrict__ ew, long long ew_ld,
                                                     long long rows, int k, int n_expert)
{
    const long long e = (long long)blockIdx.x * 256 + threadIdx.x;
    if (e >= rows * k) return;
    const long long r = e / k;
    const int c = (int)(e - r * k);
    const int id = ids[r * ids_ld + c];
    const float s = (id >= 0 && id < n_expert) ? bf16_to_f32(sc[id]) : 0.0f;
    uint16_t* w = (uint16_t*)(ew + r * ew_ld + c);
    *w = (uint16_t)f32_to_bf16(bf16_to_f32(*w) * s);
}

// [rows, k] with a dense last axis: its rows and row pitch, or false.
static bool rows_k(const RadTensor* t, int dt, long long k, long long* rows, long long* ld)
{
    if (!t || t->dtype != dt || t->rank < 1) return false;
    if (t->shape[t->rank - 1] != k || t->stride[t->rank - 1] != 1) return false;
    *rows = 1;
    for (int i = 0; i + 1 < t->rank; ++i) *rows *= t->shape[i];
    *ld = r3d_row_pitch(t);
    return *ld >= k || *rows <= 1;
}

}  // namespace

extern "C" {

/* operands: ids, w_e (the [n_expert] scale table), ew (inout) */
int r3d_rad_expert_scale_bf16(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const long long k = r3d_geti_or(a, "top_k", 0), ne = r3d_geti_or(a, "n_expert", 0);
    if (k <= 0 || k > 64 || ne <= 0 || ne > (1LL << 20)) return R3D_NO(RAD_E_SHAPE);
    const RadTensor* I = r3d_opt(a, 0), *S = r3d_opt(a, 1), *W = r3d_opt(a, 2);
    if (!r3d_p(I) || !r3d_p(S) || !r3d_wp(W)) return R3D_NO(RAD_E_INVAL);
    if (S->dtype != RAD_BF16 || r3d_numel(S) < ne) return R3D_NO(RAD_E_DTYPE);
    long long ri = 0, li = 0, rw = 0, lw = 0;
    if (!rows_k(I, RAD_I32, k, &ri, &li) || !rows_k(W, RAD_BF16, k, &rw, &lw))
        return R3D_NO(RAD_E_SHAPE);
    const long long M = rw;
    if (M <= 0) return 0;
    if (ri < M || M * k > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    const unsigned blocks = (unsigned)((M * k + 255) / 256);
    R3D_KLAUNCH(escale_kernel, dim3(blocks), dim3(256), (hipStream_t)s,
                (const int32_t*)r3d_p(I), li, (const unsigned short*)r3d_p(S),
                (unsigned short*)r3d_wp(W), lw, M, (int)k, (int)ne);
    return (int)hipGetLastError();
}

}  // extern "C"
