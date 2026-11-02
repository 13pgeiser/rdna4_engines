/* r3d_moe_gemm_q.hip -- moe_gemm_q (Phase 1: the GEMM core, dense rows, no gather).
 *
 *   y[M,N] = a[M,K] @ w[N,K]^T, both operands E4M3, with a per-(row, g-of-K) scale fold.
 *
 * `a` and `w` are E4M3 (fp8) with one f32 scale per (row, g of K). RDNA3 has no fp8 matrix unit, so
 * each operand is staged to LDS compactly (8-bit) and upcast to f16 in registers as the wave-matrix
 * fragments load -- the FeatherOps `fp8e4m3x4_to_half2x2` bit trick (a V_PERM_B32 + shift-add, no
 * table, no extra pass), then the f16 wave-matrix (`v_wmma_f32_16x16x16_f16`, f32 accumulate). Each
 * g-wide block of K is accumulated raw, then folded by scale_a[m][b] * scale_w[n][b].
 *
 * STATUS: WIP. Phases 1-2 are verified against a CPU reference (maxrel ~1e-4, the f16 rounding
 * floor):
 *   * Phase 1: the GEMM core (E4M3 x E4M3 -> f16 wave-matrix, the per-(row, 128-of-K) scale fold).
 *   * Phase 2: the gather (row i reads a[sorted[i]/top_k]), the per-expert weight tables, and the
 *     run structure (the block walks the runs in its 16 rows, so a block spanning several experts
 *     is correct).
 *
 *   * Phase 3: the even/odd expert split (expert e's weight is in `w` for even e, `w_odd` for odd
 *     e, at index e/2) and the dtype-selected scale grid (`grid_rows` = 1 for w4a8's per-row scale,
 *     128 for fp8a8's per-128x128 tile).
 *
 * Verified against a CPU reference for runs of 1, 2, 4 and 16 rows, the even/odd split, and both
 * scale grids (maxrel ~1e-4, the f16 rounding floor). A row: `moe_gemm_q_e4m3` in r3d_rows.cpp, a
 * selftest case in r3d_selftest.cpp.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_fp8_dequant.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kNBlock  = 128;           // a block is 16 rows x 128 cols: eight waves x 16 cols
constexpr int kGroup   = 128;           // a scale block: one k tile, eight wmma steps
constexpr int kPitch   = 33;            // dwords a staged row: 33 mod 32 = 1, bank-free


__device__ __forceinline__ _Float16 h16(uint32_t w) { return __builtin_bit_cast(_Float16, (uint16_t)w); }

template <int ALU>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void moe_q_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    const uint8_t* __restrict__ w, const float* __restrict__ w_scale,
    const uint8_t* __restrict__ w_odd, const float* __restrict__ w_odd_scale,
    float* __restrict__ y, int T, int N, int K, int a_ld, int w_ld_b, int y_ld,
    int n_expert, int top_k, int grid_rows)
{
    __shared__ __align__(16) uint32_t s_a[16 * kPitch];
    __shared__ __align__(16) uint32_t s_w[kNBlock * kPitch];
    __shared__ __align__(16) uint16_t s_tbl[R3D_E4M3_TBL];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[16], s_re[16], s_e[16];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wv = tid / kWave;          // the wave index (0..7): owns cols wv*16..wv*16+16
    const uint32_t r = lane & 15;

    if (!ALU) r3d_e4m3_f16_init(s_tbl, (int)tid, kThreads);   // the E4M3 -> f16 table (ALU=0)

    const uint32_t r0 = (uint32_t)blockIdx.y * 16;   // the block's sorted rows
    const uint32_t n0 = (uint32_t)blockIdx.x * kNBlock;
    const uint32_t r1 = (r0 + 16 < (uint32_t)T) ? r0 + 16 : (uint32_t)T;

    // The block's rows' source tokens (the gather). -1 marks a dropped slot.
    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (f / top_k);
    }
    // The runs in [r0, r1): each a contiguous set of rows for one expert. The block walks them.
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

    const uint32_t nc = n0 + wv * 16 + r;   // this wave's output column
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    // One GEMM per run: the run's rows read the gathered activation and expert e's weight. Rows the
    // run does not own are padding (reading the run's first row) and their output is dropped. Expert
    // e's weight is in `w` (even e) or `w_odd` (odd e) at index e/2.
    for (int run = 0; run < nruns; ++run) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const int e2 = e >> 1;
        const uint8_t* wbase = (e & 1) ? w_odd + (size_t)e2 * (size_t)N * w_ld_b
                                       : w + (size_t)e2 * (size_t)N * w_ld_b;
        const float* wsb = (e & 1) ? w_odd_scale + (size_t)e2 * (size_t)(N / grid_rows) * (K / kGroup)
                                   : w_scale + (size_t)e2 * (size_t)(N / grid_rows) * (K / kGroup);

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 E4M3 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < 16 * (kGroup / 4); i += kThreads) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            // Stage 128 cols x 128 E4M3 of the expert's weight (the block's columns).
            for (int i = (int)tid; i < kNBlock * (kGroup / 4); i += kThreads) {
                const int col = i / (kGroup / 4), d = i % (kGroup / 4);
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
                s_w[(size_t)col * kPitch + d] =
                    *(const uint32_t*)(wbase + (size_t)nn * w_ld_b + kt + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            const uint32_t* wp = s_w + (size_t)(wv * 16 + r) * kPitch;   // this wave's 16 cols
            v8float blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                uint32_t ain[4], af[8], bin[4], bf[8];
#pragma unroll
                for (int g = 0; g < 4; ++g) ain[g] = ap[k / 4 + g];
                r3d_e4m3x16_decode<ALU>(s_tbl, ain, af);
#pragma unroll
                for (int g = 0; g < 4; ++g) bin[g] = wp[k / 4 + g];
                r3d_e4m3x16_decode<ALU>(s_tbl, bin, bf);
                v16half fa, fb;
#pragma unroll
                for (int g = 0; g < 8; ++g) { fa[2 * g] = h16(af[g] & 0xffffu); fa[2 * g + 1] = h16(af[g] >> 16); }
#pragma unroll
                for (int g = 0; g < 8; ++g) { fb[2 * g] = h16(bf[g] & 0xffffu); fb[2 * g + 1] = h16(bf[g] >> 16); }
                blk = wmma_f16(fa, fb, blk);
            }
            group_barrier();

            // Fold the g-block's scales. acc[g] is at row g*2 + lane/16 (varies with g), column r.
            // The weight scale is per (grid_rows columns, 128 of K).
            const float sw = wsb[(size_t)(n_real / grid_rows) * (K / kGroup) + b];
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int row = (int)(g * 2 + (lane >> 4));
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                 : s_tok[rs - (int)r0];
                const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * (K / kGroup) + b];
                acc[g] += blk[g] * sa * sw;
            }
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = (int)(g * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = acc[g];
        }
    }
}

}  // namespace

extern "C" {

static int r3d_moe_gemm_q_dispatch(const RadArgs* a, RadStream s, int alu)
{
    if (!a || a->n_t < 9) return R3D_NO(RAD_E_INVAL);
    const RadTensor* at = r3d_opt(a, 0), *asc = r3d_opt(a, 1), *st = r3d_opt(a, 2),
                    *ot = r3d_opt(a, 3), *wt = r3d_opt(a, 4), *wsc = r3d_opt(a, 5),
                    *wot = r3d_opt(a, 6), *wosc = r3d_opt(a, 7), *yt = r3d_opt(a, 8);
    if (!r3d_p(at) || !r3d_p(asc) || !r3d_p(st) || !r3d_p(ot) || !r3d_p(wt) || !r3d_p(wsc) ||
        !r3d_wp(yt))
        return R3D_NO(RAD_E_INVAL);
    long long M = 0, K = 0;
    if (r3d_rows_of(at, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    const long long T = (st->rank >= 1) ? st->shape[0] : 0;
    const long long N = (wt->rank >= 3) ? wt->shape[1] : 0;
    if (M <= 0 || T <= 0 || N <= 0 || K <= 0 || K % 128) return R3D_NO(RAD_E_SHAPE);   // N tail clamps
    const int n_expert = (int)r3d_geti_or(a, "n_expert", 1);
    const int top_k = (int)r3d_geti_or(a, "top_k", 1);
    const int grid_rows = (int)r3d_geti_or(a, "grid_rows", 1);  // 1 (w4a8) or 128 (fp8a8)
    const int a_ld = (int)at->stride[0];
    const int w_ld_b = (int)wt->shape[2];            // the weight's K in bytes
    const int y_ld = (int)yt->stride[0];

    dim3 grid((unsigned)((N + kNBlock - 1) / kNBlock), (unsigned)((T + 15) / 16)), blk(kThreads);
    const uint8_t* pa = (const uint8_t*)r3d_p(at);
    const float* psa = (const float*)r3d_p(asc);
    const int32_t* ps = (const int32_t*)r3d_p(st);
    const int32_t* po = (const int32_t*)r3d_p(ot);
    const uint8_t* pw = (const uint8_t*)r3d_p(wt);
    const float* pws = (const float*)r3d_p(wsc);
    const uint8_t* pwo = (const uint8_t*)r3d_p(wot);
    const float* pwos = (const float*)r3d_p(wosc);
    float* py = (float*)r3d_wp(yt);
    if (alu)
        moe_q_kernel<1><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, pw, pws, pwo, pwos, py,
                                                          (int)T, (int)N, (int)K, a_ld, w_ld_b, y_ld,
                                                          n_expert, top_k, grid_rows);
    else
        moe_q_kernel<0><<<grid, blk, 0, (hipStream_t)s>>>(pa, psa, ps, po, pw, pws, pwo, pwos, py,
                                                          (int)T, (int)N, (int)K, a_ld, w_ld_b, y_ld,
                                                          n_expert, top_k, grid_rows);
    return (int)hipGetLastError();
}

/* The two rows share the dispatch: the table decode (ALU=0) and the ALU decode (ALU=1). The tune
 * picks the faster per geometry. */
extern "C" int r3d_rad_moe_gemm_q(const RadArgs* a, RadStream s)
{ return r3d_moe_gemm_q_dispatch(a, s, 0); }
extern "C" int r3d_rad_moe_gemm_q_alu(const RadArgs* a, RadStream s)
{ return r3d_moe_gemm_q_dispatch(a, s, 1); }

}  // extern "C"
