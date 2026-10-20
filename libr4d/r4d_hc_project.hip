// r4d_hc_project.hip -- the experimental speed projection on the gated residual's wide stream.
//
// The operation is llama.cpp's `--cvec-mode project`, which the Qwen3.8-Flash-Next control-vector
// package publishes and Strata ports (data/experimental-speed-projection): for each of the hc
// streams of a token's residual row,
//
//     h[s] -= scale * (h[s] . v) * v
//
// with `v` that layer's UNIT direction and `scale` the norm the file stored for it. The dot is
// taken per stream, not on the stream mean: llama.cpp's patch applies `build_cvec` to the
// [n_embd, hc, tokens] residual view, where `mul_mat(v, cur)` reduces over dim 0 alone.
//
// The op is issued after a layer's feed-forward write into the stream, which is where the patch's
// `build_cvec(res_hc, il)` sits in the `qwen4exp` graph.
//
// ============================== the per-row switch ==============================
//
// `on` is an int32 per row (core derive.cpp's `projection_on_rows`: the request's own choice,
// expanded across its tokens once a step). A zero row is skipped before any load: no read, no
// write. That is what makes a request without the vector byte-identical to a deployment with no
// vector loaded -- not merely close -- and it costs one load per row per layer.
//
// ============================== shape ==============================
//
// One workgroup per (token row, stream): it holds the whole of `v`'s stream in no memory at all
// (it streams `h` and `v` together), reduces the dot in LDS, and then rewrites the row. Grid
// (rows, hc). At the model's shape (n 2560, hc 4) that is 2560 f32 of `v` re-read per (row,
// stream): 40 KB per token per layer, against 40 KB of `h` -- the same traffic as the write it
// follows, within a fraction of the step's expert traffic.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r4d.h"
#include "r4d_args.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxWave  = 64;

__device__ __forceinline__ float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One workgroup per (row, stream). `h` is the row's stream at `h + row*h_ld + stream*n`, `v` the
 * layer's unit direction (f32, `v_ld` may exceed n for a slice of a wider plane). */
__global__ __launch_bounds__(kThreads) void hc_project_kernel(uint16_t* __restrict__ h,
                                                              const float* __restrict__ v,
                                                              const int32_t* __restrict__ on,
                                                              uint32_t n, uint32_t h_ld,
                                                              uint32_t v_ld, float scale) {
    const uint32_t row = blockIdx.x;
    if (on[row] == 0) return;   // this token's request did not ask for the projection

    uint16_t* __restrict__ hr = h + (size_t) row * h_ld + (size_t) blockIdx.y * n;
    const float* __restrict__ vr = v + (size_t) blockIdx.y * v_ld;

    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) acc += bf2f(hr[i]) * vr[i];

    __shared__ float red[kMaxWave];
    const uint32_t lane = threadIdx.x % kMaxWave;
    const uint32_t wave = threadIdx.x / kMaxWave;
    for (uint32_t off = kMaxWave / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, kMaxWave);
    }
    if (lane == 0) red[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        const uint32_t nw = (blockDim.x + kMaxWave - 1) / kMaxWave;
        for (uint32_t w = 0; w < nw; ++w) total += red[w];
        red[0] = total * scale;
    }
    __syncthreads();

    const float g = red[0];
    for (uint32_t i = threadIdx.x; i < n; i += kThreads)
        hr[i] = f2bf(bf2f(hr[i]) - g * vr[i]);
}

}  // namespace

/* Host form: rows and the stream count come from the caller (the entry point below, which reads
 * them from the operands). */
extern "C" int r4d_hc_project_bf16(long h, long v, long on, int rows, int n, int hc, int h_ld,
                                   int v_ld, float scale, long stream) {
    if (!h || !v || !on) return -1;
    if (rows <= 0 || n <= 0 || hc <= 0) return -2;
    if (h_ld <= 0) h_ld = hc * n;
    if (v_ld <= 0) v_ld = n;
    hipLaunchKernelGGL(hc_project_kernel, dim3((uint32_t) rows, (uint32_t) hc), dim3(kThreads), 0,
                       (hipStream_t) stream, (uint16_t*) h, (const float*) v,
                       (const int32_t*) on, (uint32_t) n, (uint32_t) h_ld, (uint32_t) v_ld,
                       scale);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ============================== the entry point ==============================
 *
 * Operands: h (inout, bf16, [rows, hc*n]), v (f32, [n]), on (int32, [rows]). Parameters: n, hc,
 * scale, dtype. `rows` is derived from h's extent, as every per-row op here does: the kernel owns
 * whole rows. */
extern "C" int r4d_rad_hc_project(const RadArgs* a, RadStream s) {
    const RadTensor* h  = r4d_opt(a, 0);
    const RadTensor* v  = r4d_opt(a, 1);
    const RadTensor* on = r4d_opt(a, 2);
    if (!h || !v || !on) return R4D_NO(RAD_E_INVAL);
    if (h->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (v->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (on->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(v) < n) return R4D_NO(RAD_E_SHAPE);

    long long rows = 0, row_n = 0;
    if (r4d_rows_of(h, &rows, &row_n) < 0) return R4D_NO(RAD_E_SHAPE);
    if (row_n != hc * n || rows <= 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(on) < rows) return R4D_NO(RAD_E_SHAPE);

    /* No pitch handling: the architecture passes the model's own wide stream, contiguous rows of
     * hc*n. A strided view is refused rather than silently read as contiguous. */
    const long long h_ld = r4d_stride(h, (int) h->rank - 2);
    const long long v_ld = r4d_stride(v, (int) v->rank - 1);
    if (h_ld != hc * n || v_ld != 1) return R4D_NO(RAD_E_STRIDE);

    const float scale = (float) r4d_getf(a, "scale", 1.0);
    return r4d_hc_project_bf16(r4d_addr(h), r4d_addr(v), r4d_addr(on), (int) rows, (int) n,
                               (int) hc, (int) h_ld, 0, scale, (long) (uintptr_t) s) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
