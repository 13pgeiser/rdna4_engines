/* r3d_kv_len_trim.hip -- kv_len_trim: a batch's per-sequence context lengths less `trim`.
 *
 * The prefill overlap runs attention per slice of a one-sequence step: slice A's rows are the step's
 * first rows, so its attention must see the context ending at its own last row, not the step's (rows
 * past it are not yet in the cache). `trim` is the rows after the slice, a runtime count carried by
 * the extent of operand `tail`, whose values are never read.
 *
 *   in [n] i32, tail (extent = trim) -> out [n] i32, out[i] = max(in[i] - trim, 0)
 *
 * n <= 64 (a batch), one workgroup.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

__global__ __launch_bounds__(64) void kv_len_trim_kernel(const int32_t* __restrict__ in,
                                                         int32_t* __restrict__ out, int n,
                                                         int trim) {
    const int i = (int)threadIdx.x;
    if (i < n) {
        const int v = in[i] - trim;
        out[i] = v > 0 ? v : 0;
    }
}

}  // namespace

extern "C" {

int r3d_rad_kv_len_trim(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* in = r3d_opt(a, 0), *tail = r3d_opt(a, 1), *out = r3d_opt(a, 2);
    if (!r3d_p(in) || !r3d_wp(out)) return R3D_NO(RAD_E_INVAL);
    if (in->dtype != RAD_I32 || out->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_numel(in);
    if (n < 1 || n > 64 || r3d_numel(out) < n) return R3D_NO(RAD_E_SHAPE);
    const long long trim = tail ? r3d_numel(tail) : 0;
    if (trim < 0 || trim > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);

    R3D_KLAUNCH(kv_len_trim_kernel, dim3(1), dim3(64), (hipStream_t)s, (const int32_t*)r3d_p(in),
                (int32_t*)r3d_wp(out), (int)n, (int)trim);
    return (int)hipGetLastError();
}

}  // extern "C"
