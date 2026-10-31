/* r3d_gemm_bf16_nt_m64_wmma.hip -- the wave-matrix bf16 GEMM for the M = 17..127 band.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate, narrowed once by the store.
 *
 * The m16 arm (r3d_gemm_bf16_nt_m16_wmma) owns one 16-row tile a wave and stages that tile in
 * LDS; above M = 16 a 128-row tiled block wastes half its rows, so this arm takes MT 16-row
 * tiles a block and walks them with MT accumulators a wave. The B-fragment (the weight column)
 * is loaded once per 16-k step and reused over the MT row tiles, so the weight stream is read
 * once per 16-column tile no matter how many row tiles the block carries -- the activation, not
 * the weight, is the traffic that grows with MT, and it is read from global (no LDS) so a block
 * stays small enough to keep the waves on the SIMD.
 *
 * ---- the tile map --------------------------------------------------------------------------
 *
 * One wave owns a 16-column output tile across MT 16-row tiles and walks k in 16-wide steps, one
 * wmma a step per row tile. The ISA's lane map (wave32, the `_w32` builtins), the m16 arm's:
 *
 *   A fragment: lane L holds row L % 16 of a row tile, 16 bf16 in eight packed VGPRs
 *   B fragment: lane L holds column L % 16 of the tile, 16 bf16 in eight packed VGPRs
 *   C/D:        lane L, VGPR g holds D[g * 2 + L / 16][L % 16], eight f32 a lane
 *
 * ---- split-K --------------------------------------------------------------------------------
 *
 * The m16 arm's slabs and f32 partial plane, with the same law: a slab is whole 16-k wmma steps,
 * so a split is legal only when K % (sk * 16) == 0, and the reducer adds the sk partials in
 * ascending split order. sk = 1 serves a ragged K.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;           // eight waves, the default
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kUnit         = 16;            // k a wmma step consumes: a slab's law


#define R3D_U16(w) ((short)((w) & 0xffffu)), ((short)((w) >> 16))

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint16_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

/* y[M,N] = a[M,K] @ b[N,K]^T, MT 16-row tiles a block, one 16-column tile a wave.
 *
 * `kslab` is the k range a blockIdx.y owns; with sk = 1 it is all of K and the store is the
 * narrowed bf16 the caller asked for. With sk > 1 every split writes its own f32 partial, dense
 * [sk][M][N], and the reducer adds them in split order. */
template <bool NT, int MT, bool SPLIT, bool F16ACC>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m64_wmma_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Part,
    int M, int N, int K, int a_ld, int y_ld, int w_ld, int kslab)
{
    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t waves = blockDim.x / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = ((uint32_t)blockIdx.x * waves + wave) * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const uint32_t mt0 = (uint32_t)blockIdx.y * MT * 16;   // this block's first row

    const int kbeg = (int)blockIdx.y * 0;             // the split is blockIdx.z; see the launch
    (void)kbeg;

    const uint16_t* w = B + (size_t)n_real * w_ld;
    /* The accumulator: f32 for the f32-acc arm, v16short for the bf16-acc arm. The bf16-acc
     * builtin's 8 live elements sit in the even indices (opsel = false); the epilogue reads
     * only those, the odd half is dead.
     *
     * PRECISION: the bf16-acc arm is a documented low-precision mode -- the hardware quantises
     * the running sum at bf16 precision per wave-matrix step, and the loss grows with the number
     * of steps (measured on gfx1102, random data in [-1,1]: ~0.78 abs after 16 k, ~51 after 256
     * k, ~622 after 1024 k; integer-valued data stays exact). It measures no throughput gain
     * over the f32 arm. It is a register-pressure option for future occupancy work, not a speed
     * one; do not select it for accuracy-sensitive shapes. See the README's 16-bit
     * wave-matrix accumulate section. */
    using AccT = std::conditional_t<F16ACC, v16short, v8float>;
    AccT acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        if constexpr (F16ACC)
            acc[i] = v16short{ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 };
        else
            acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    }

    // The split's k range: blockIdx.z owns [z*kslab, z*kslab + kslab).
    const int kbeg2 = (int)blockIdx.z * kslab;
    const int kend = kbeg2 + kslab < K ? kbeg2 + kslab : K;

    /* Double-buffered k-walk: the next step's A and B fragments are issued into registers while
     * the current step's wmmas run, so the global load latency overlaps the multiply. The first
     * step is prefetched before the loop; each iteration consumes the current fragments, issues
     * the next step's loads, then swaps. The ragged end (kl < kUnit) keeps the zero-fill. */
    const uint16_t* bp = w + kbeg2;
    const r3d_u32x4 b0 = ldw<NT>(bp);
    const r3d_u32x4 b1 = ldw<NT>(bp + 8);
    v16short fb = v16short{ R3D_U16(b0.x), R3D_U16(b0.y), R3D_U16(b0.z),
                            R3D_U16(b0.w), R3D_U16(b1.x), R3D_U16(b1.y),
                            R3D_U16(b1.z), R3D_U16(b1.w) };
    v16short fa[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int mrow = (int)mt0 + i * 16 + (int)r;
        const uint16_t* ap = A + (size_t)(mrow < M ? mrow : M - 1) * a_ld + kbeg2;
        const r3d_u32x4 a0 = ldw<NT>(ap);
        const r3d_u32x4 a1 = ldw<NT>(ap + 8);
        fa[i] = v16short{ R3D_U16(a0.x), R3D_U16(a0.y), R3D_U16(a0.z),
                          R3D_U16(a0.w), R3D_U16(a1.x), R3D_U16(a1.y),
                          R3D_U16(a1.z), R3D_U16(a1.w) };
    }
    for (int kt = kbeg2; kt < kend; kt += kUnit) {
        const int kl = std::min(kUnit, kend - kt);
        const int ktn = kt + kUnit;
        const bool more = (ktn < kend);
        // Issue the next step's fragment loads now, while this step's wmmas run below.
        r3d_u32x4 b0n, b1n;
        v16short fbn;
        v16short fan[MT];
        if (more) {
            b0n = ldw<NT>(w + ktn);
            b1n = ldw<NT>(w + ktn + 8);
            fbn = v16short{ R3D_U16(b0n.x), R3D_U16(b0n.y), R3D_U16(b0n.z),
                            R3D_U16(b0n.w), R3D_U16(b1n.x), R3D_U16(b1n.y),
                            R3D_U16(b1n.z), R3D_U16(b1n.w) };
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int mrow = (int)mt0 + i * 16 + (int)r;
                const uint16_t* apn = A + (size_t)(mrow < M ? mrow : M - 1) * a_ld + ktn;
                const r3d_u32x4 a0n = ldw<NT>(apn);
                const r3d_u32x4 a1n = ldw<NT>(apn + 8);
                fan[i] = v16short{ R3D_U16(a0n.x), R3D_U16(a0n.y), R3D_U16(a0n.z),
                                   R3D_U16(a0n.w), R3D_U16(a1n.x), R3D_U16(a1n.y),
                                   R3D_U16(a1n.z), R3D_U16(a1n.w) };
            }
        }
        // Consume the current step's fragments.
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            if (kl == kUnit) {
                if constexpr (F16ACC) acc[i] = wmma_bf16acc<false>(fa[i], fb, acc[i]);
                else acc[i] = wmma_bf16(fa[i], fb, acc[i]);
            } else {
                v16short fa0 = fa[i];
                for (int e = kl; e < 16; ++e) fa0[e] = (short)0;
                if constexpr (F16ACC) acc[i] = wmma_bf16acc<false>(fa0, fb, acc[i]);
                else acc[i] = wmma_bf16(fa0, fb, acc[i]);
            }
        }
        // Advance: the prefetched next step becomes the current step.
        if (more) {
            fb = fbn;
#pragma unroll
            for (int i = 0; i < MT; ++i) fa[i] = fan[i];
        }
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        for (int g = 0; g < 8; ++g) {
            const int m = (int)mt0 + i * 16 + g * 2 + (int)(lane >> 4);
            if (m >= M) continue;
            if constexpr (F16ACC) {
                // The bf16-acc arm: the 8 live elements are the even indices (opsel = false).
                // Widen the bf16 short to f32 for the split partial; the store re-narrows.
                const uint32_t u = ((uint32_t)(uint16_t)acc[i][2 * g]) << 16;
                float v; memcpy(&v, &u, 4);
                if constexpr (SPLIT)
                    Part[((size_t)blockIdx.z * M + m) * N + n0] = v;
                else
                    Y[(size_t)m * y_ld + n0] = (uint16_t)((uint16_t)acc[i][2 * g]);
            } else {
                if constexpr (SPLIT)
                    Part[((size_t)blockIdx.z * M + m) * N + n0] = acc[i][g];
                else
                    Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[i][g]);
            }
        }
    }
}
#undef R3D_U16

/* The split reduction: v = ((p0 + p1) + p2) + ... in ascending split order, then one
 * narrowing. The m16 arm's, copied: the two arms must not share a symbol through a header. */
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m64_wmma_split_reduce(
    const float* __restrict__ Part, uint16_t* __restrict__ Y, int M, int N, int y_ld, int sk)
{
    const int mn = M * N;
    const int stride = gridDim.x * kBlockThreads;
    for (int i = blockIdx.x * kBlockThreads + threadIdx.x; i < mn; i += stride) {
        const int m = i / N, n = i - m * N;
        const float* p = Part + (size_t)m * N + n;
        float v = p[0];
        for (int s = 1; s < sk; ++s) v = v + p[(size_t)s * M * N];
        Y[(size_t)m * y_ld + n] = f32_to_bf16(v);
    }
}

}  // namespace

/* y[M,N] = a[M,K] @ b[N,K]^T, bf16 in and out, f32 accumulate, the M = 17..127 band on the
 * wave-matrix. `sk` splits K (0 = the largest legal split), `wv` waves per block (0 = 8), `nt`
 * the weight load's lowering, `scratch` the f32 partial arena, sk * M * N * 4 bytes, required
 * when the split is on. Returns 0, -1 a null pointer, -2 a shape it does not serve, -4 a launch
 * failure. */
extern "C" int r3d_gemm_nt_bf16_m64_wmma(long a, long b, long y, int M, int N, int K, int a_ld,
                                         int y_ld, int sk, int wv, int nt, long scratch,
                                         long stream, int acc)
{
    if (acc < 0) acc = 0;
    if (acc > 1) acc = 1;
    if (!a || !b || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    const int w_ld = K;
    if (sk == 0) {
        static const int kLadder[] = { 8, 4, 2, 1 };
        for (int s : kLadder)
            if (s > 1 ? K % ((long long)s * kUnit) == 0 : true) { sk = s; break; }
    }
    if (sk < 1) return -2;
    if (sk > 1 && K % ((long long)sk * kUnit)) return -2;
    if (sk > 1 && !scratch) return -1;
    if (wv <= 0) wv = kWaves;
    if (wv != 1 && wv != 2 && wv != 4 && wv != kWaves) return -2;

    const int kslab = K / sk;
    const int ntiles = (N + 15) / 16;
    const int ncolblk = (ntiles + wv - 1) / wv;

#define R3D_WMT(NTV, MT, ACC)                                                                           \
    do {                                                                                          \
        const int mrowblk = (M + MT * 16 - 1) / (MT * 16);                                                    \
        const dim3 grid(ncolblk, (unsigned)mrowblk, (unsigned)sk), block((unsigned)(wv * kWave)); \
        if (sk > 1) {                                                                             \
            R3D_KLAUNCH(R3D_KNAME((gemm_m64_wmma_kernel<NTV, MT, true, ACC>)), grid, block,           \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,             \
                        (uint16_t*)y, (float*)scratch, M, N, K, a_ld, y_ld, w_ld, kslab);         \
            const int mn = M * N;                                                                 \
            const int nb = std::min((mn + kBlockThreads - 1) / kBlockThreads, 4096);              \
            R3D_KLAUNCH(gemm_m64_wmma_split_reduce, dim3(nb), dim3(kBlockThreads),                \
                        (hipStream_t)stream, (const float*)scratch, (uint16_t*)y, M, N, y_ld,     \
                        sk);                                                                      \
        } else {                                                                                  \
            R3D_KLAUNCH(R3D_KNAME((gemm_m64_wmma_kernel<NTV, MT, false, ACC>)), grid, block,          \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,             \
                        (uint16_t*)y, (float*)nullptr, M, N, K, a_ld, y_ld, w_ld, kslab);         \
        }                                                                                         \
    } while (0)

#define R3D_WNT(NTV)                                                                            \
    do {                                                                                         \
        if (M <= 16) { if (acc) R3D_WMT(NTV, 1, true); else R3D_WMT(NTV, 1, false); }           \
        else if (M <= 32) { if (acc) R3D_WMT(NTV, 2, true); else R3D_WMT(NTV, 2, false); }      \
        else { if (acc) R3D_WMT(NTV, 4, true); else R3D_WMT(NTV, 4, false); }                   \
    } while (0)
    if (nt) R3D_WNT(true);
    else R3D_WNT(false);
#undef R3D_WNT
#undef R3D_WMT
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The RadArgs dispatch: the row's entry. The M = 17..127 band, sk = 1 (the N tiles carry the
 * parallelism at prefill widths, so no split and no scratch), wv = 8, nt = 0. */
extern "C" int r3d_rad_gemm_nt_bf16_m64_wmma(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* at = &a->t[0], *wt = &a->t[1], *yt = &a->t[2];
    if (!r3d_p(at) || !r3d_p(wt) || !r3d_wp(yt)) return R3D_NO(RAD_E_INVAL);
    if (at->dtype != RAD_BF16 || wt->dtype != RAD_BF16 || yt->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    long long M = 0, K = 0;
    if (r3d_rows_of(at, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    const long long N = (wt->rank >= 2) ? wt->shape[0] : 0;
    if (M <= 0 || N <= 0 || K <= 0)
        return R3D_NO(RAD_E_SHAPE);
    const int a_ld = (int)at->stride[0];
    const int y_ld = (int)yt->stride[0];
    return r3d_gemm_nt_bf16_m64_wmma((long)(uintptr_t)r3d_p(at), (long)(uintptr_t)r3d_p(wt),
                                     (long)(uintptr_t)r3d_wp(yt), (int)M, (int)N, (int)K, a_ld,
                                     y_ld, 1, 8, 0, 0, (long)(uintptr_t)s, 0);
}
