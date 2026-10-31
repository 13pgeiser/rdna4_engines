/* r3d_moe_gemm_w4a8.hip -- moe_gemm_w4a8 (Phase 2: the GEMM core + the gather + per-expert weights).
 *
 *   y[T,N] = a[gather(row), :] @ w[e(row), :]^T, with a per-(row, 128-of-K) scale fold.
 *
 * The routed rows are in SORTED order (one row per (token, expert slot)); row i reads its
 * activation from `a[sorted[i] / top_k]` and its weight from expert e(i)'s table, where e(i) is the
 * expert owning sorted row i (the one whose [off, off') span contains i). `a` is int8 with one f32
 * scale per (row, 128 of K); each expert's weight is a 4-bit two's-complement nibble (-8..7) packed
 * two per byte, with one f32 scale per (row, 128 of K). RDNA3's native int8 wave-matrix
 * (`v_wmma_i32_16x16x16_iu8`, exact i32 accumulate) takes the int4 weight widened to int8, so no f16
 * rounding. Each 128-wide block of K is accumulated raw in i32, then folded by
 * scale_a[src][b] * scale_w[e][n][b].
 *
 * STATUS: WIP. Phases 1-3 are verified bit-exact against a CPU reference (the i32 accumulate is
 * exact, so the random cases are exact, not just within tolerance):
 *   * Phase 1: the GEMM core (int8 x widened-int4, the per-(row, 128-of-K) scale fold).
 *   * Phase 2: the gather (row i reads a[sorted[i]/top_k]) and the per-expert weight tables.
 *   * Phase 3: the run structure -- the block walks the runs in its 16 rows, so a block spanning
 *     several experts (short runs, decode) is correct. Verified for runs of 1, 2, 4 and 16 rows.
 *
 * The `w4a8` form (a two's-complement nibble, -8..7) is served. Remaining (Phase 4): the other
 * weight forms (w4nl64a8h / w4nl32a8h / w5nl64a8h, whose nibbles decode through a magnitude table
 * and a per-64/32 E4M3 scale), the prefill run map, and the SK/UW/RS optimisations. Not yet a row:
 * no r3d_rows.cpp entry, no selftest.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kGroup   = 128;           // a scale block: one k tile, eight wmma steps
constexpr int kPitch   = 33;            // dwords a staged a row: 33 mod 32 = 1, bank-free


__device__ __forceinline__ int32_t nib_i8(uint32_t nib)
{
    return (nib & 8u) ? (int32_t)nib - 16 : (int32_t)(nib & 7u);
}

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void moe_w4a8_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    const uint8_t* __restrict__ w, const float* __restrict__ w_scale,
    float* __restrict__ y, int T, int N, int K, int a_ld, int w_ld_b, int y_ld,
    int n_expert, int top_k)
{
    __shared__ __align__(16) uint32_t s_a[16 * kPitch];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[16], s_re[16], s_e[16];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;

    const uint32_t r0 = (uint32_t)blockIdx.y * 16;   // the block's sorted rows
    const uint32_t ncol = ((uint32_t)blockIdx.x * kWaves + wave) * 16;   // the wave's 16 columns
    const uint32_t r1 = (r0 + 16 < (uint32_t)T) ? r0 + 16 : (uint32_t)T;

    // The block's rows' source tokens, resolved once (the gather). -1 marks a dropped slot.
    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (f / top_k);
    }
    // The runs in [r0, r1): each is a contiguous set of rows for one expert. The block walks them.
    if (tid == 0) {
        int nruns = 0, i = (int)r0;
        while (i < (int)r1) {
            int32_t ee = -1;
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x;
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;
            ++nruns; i = re;
        }
        s_nruns = nruns;
    }
    group_barrier();
    const int nruns = s_nruns;

    const uint32_t nc = ncol + r;
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    // One GEMM per run: the run's rows read the gathered activation and expert e's weight. Rows the
    // run does not own are padding (they read the run's first row) and their output is dropped.
    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const uint8_t* wrow = w + (size_t)e * (size_t)N * w_ld_b + (size_t)n_real * w_ld_b;
        const float* wsb = w_scale + (size_t)e * (size_t)N * (K / kGroup);

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 i8 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < 16 * (kGroup / 4); i += kThreads) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                const uint2 wp = *(const uint2*)(wrow + (kt + k) / 2);   // 8 bytes = 16 nibbles
                int32_t v[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int byt = j / 2;
                    const bool high = (j & 1);
                    const uint32_t byte = (byt < 4) ? ((wp.x >> (8 * byt)) & 0xFFu)
                                                    : ((wp.y >> (8 * (byt - 4))) & 0xFFu);
                    v[j] = nib_i8(high ? (byte >> 4) : (byte & 0xFu));
                }
                v4int bf{ (int)((uint32_t)(v[0] & 0xFF) | (uint32_t)(v[1] & 0xFF) << 8 |
                            (uint32_t)(v[2] & 0xFF) << 16 | (uint32_t)(v[3] & 0xFF) << 24),
                          (int)((uint32_t)(v[4] & 0xFF) | (uint32_t)(v[5] & 0xFF) << 8 |
                            (uint32_t)(v[6] & 0xFF) << 16 | (uint32_t)(v[7] & 0xFF) << 24),
                          (int)((uint32_t)(v[8] & 0xFF) | (uint32_t)(v[9] & 0xFF) << 8 |
                            (uint32_t)(v[10] & 0xFF) << 16 | (uint32_t)(v[11] & 0xFF) << 24),
                          (int)((uint32_t)(v[12] & 0xFF) | (uint32_t)(v[13] & 0xFF) << 8 |
                            (uint32_t)(v[14] & 0xFF) << 16 | (uint32_t)(v[15] & 0xFF) << 24) };
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w }, bf, sub);
            }

            // Fold the 128-block's scales: scale_a[src][b] * scale_w[e][n][b].
            const float sw = wsb[(size_t)n_real * (K / kGroup) + b];
#pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                const int row = (int)(g2 * 2 + (lane >> 4));
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                 : s_tok[rs - (int)r0];
                const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * (K / kGroup) + b];
                acc[g2] += (float)sub[g2] * sa * sw;
            }
            group_barrier();
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int row = (int)(g2 * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = acc[g2];
        }
    }
}

}  // namespace

extern "C" {

int r3d_rad_moe_gemm_w4a8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 7) return R3D_NO(RAD_E_INVAL);
    const RadTensor* at = r3d_opt(a, 0), *asc = r3d_opt(a, 1), *st = r3d_opt(a, 2),
                    *oft = r3d_opt(a, 3), *wt = r3d_opt(a, 4), *wsc = r3d_opt(a, 5),
                    *yt = r3d_opt(a, 6);
    if (!r3d_p(at) || !r3d_p(asc) || !r3d_p(st) || !r3d_p(oft) || !r3d_p(wt) || !r3d_p(wsc) ||
        !r3d_wp(yt))
        return R3D_NO(RAD_E_INVAL);
    const long long ne = r3d_geti_or(a, "n_expert", 0), tk = r3d_geti_or(a, "top_k", 0);
    long long M = 0, K = 0, T = 0, N = 0;
    if (r3d_rows_of(at, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    if (r3d_rows_of(yt, &T, &N) != 0) return R3D_NO(RAD_E_INVAL);
    if (ne <= 0 || tk <= 0 || M <= 0 || N <= 0 || K <= 0 || K % 128) return R3D_NO(RAD_E_SHAPE);
    const int a_ld = (int)(at->rank >= 2 ? at->shape[1] : at->stride[0]);
    const int w_ld_b = (int)(wt->rank >= 3 ? wt->shape[2] : wt->stride[2]);   // bytes/row = K/2
    const int y_ld = (int)(yt->rank >= 2 ? yt->shape[1] : yt->stride[0]);

    dim3 grid((unsigned)((N + 127) / 128), (unsigned)((T + 15) / 16)), blk(kThreads);
    R3D_KLAUNCH(moe_w4a8_kernel, grid, blk, (hipStream_t)s, (const int8_t*)r3d_p(at),
                (const float*)r3d_p(asc), (const int32_t*)r3d_p(st), (const int32_t*)r3d_p(oft),
                (const uint8_t*)r3d_p(wt), (const float*)r3d_p(wsc), (float*)r3d_wp(yt), (int)T,
                (int)N, (int)K, a_ld, w_ld_b, y_ld, (int)ne, (int)tk);
    return (int)hipGetLastError();
}

}  // extern "C"
