/* r3d_gemm_bf16_nt_m16_wmma.hip -- the wave-matrix arm of the skinny bf16 GEMM, M <= 16.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate, narrowed once by the store.
 *
 * The dot2 arm (r3d_gemm_bf16_nt_m16.hip) owns one output element a lane and folds k in
 * ascending pairs; this arm issues V_WMMA_F32_16X16X16_BF16 over a 16x16x16 tile, a wave a
 * tile, and the two arms are two points of the row's `impl` axis. The tune cache picks the
 * point per shape per machine; an untuned install runs the dot2 arm, which is the axis's
 * default.
 *
 * ---- the tile map --------------------------------------------------------------------------
 *
 * One wave owns a 16-row by 16-column output tile and walks k in 16-wide steps, one wmma a
 * step. The ISA's lane map (wave32, the `_w32` builtins):
 *
 *   A fragment: lane L holds row L % 16 of the tile, 16 bf16 in eight packed VGPRs
 *   B fragment: lane L holds column L % 16 of the tile, 16 bf16 in eight packed VGPRs
 *   C/D:        lane L, VGPR g holds D[g * 2 + L / 16][L % 16], eight f32 a lane
 *
 * and lanes 0-15 and 16-31 must hold identical A and B data (the replication the ISA
 * requires; only L % 16 selects the row or column). The accumulator chain is D = C, so a
 * dependent wmma needs no V_NOP between steps: the hazard the ISA names is A/B overlapping
 * the next step's D, and the fragments are fresh registers each step.
 *
 * The k order is the dot2 arm's contract at the tile boundary: tiles are summed in ascending
 * k, and within a tile the sixteen products run in the hardware's fixed order. That is the
 * arm's exactness statement: sk = 1 agrees with the reference to the last few bits, the way
 * the dot2 arm's splits do, and the row's tolerance gate prices it, not a bit-exact claim.
 *
 * ---- where the bytes come from --------------------------------------------------------------
 *
 * The activation is staged in LDS in bf16, the checkpoint's own bytes: the bf16 builtin
 * takes the tile's bits directly, so the widening the dot2 arm pays once per element
 * (bf16x2_to_f16x2 on the way in) is not here at all. One tile is 16 rows x 128 k of bf16,
 * the dot2 arm's k tile, 4.25 KiB with the pitch, and carries eight wmma steps. The pitch is
 * the dot2 arm's 68 dwords: 68 mod 32 = 4, so the lanes of a ds_read phase land four banks
 * apart and the two 16-byte fragment reads a lane issues per step are conflict-free in
 * their phases.
 *
 * The weight streams from global, one 16-byte pair a lane per step (column L % 16 of the
 * tile, the step's 16 k), the NT axis choosing the lowering the dot2 arm's header describes.
 * Every weight byte is read once per output tile, the same stream the dot2 arm streams; what
 * the arm changes is the issue: one wmma where the dot2 arm issues eight dot4 folds.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * Eight accumulator VGPRs, eight A, eight B and the pointers: about 34 a lane. Eight waves
 * a SIMD asks 272 of the 1024 the gfx11 SIMD keeps, so the block default stands and `wv`
 * walks the same ladder as the dot2 arm.
 *
 * ---- split-K --------------------------------------------------------------------------------
 *
 * The dot2 arm's slabs and partial plane, with one change: a slab is whole wmma steps, so a
 * split is legal only when K % (sk * 16) == 0, and the ragged end is not a scalar tail but a
 * step whose fragments zero-fill past K -- the products with zero are the identity and the
 * k loop stays uniform. sk = 1 serves a ragged K the dot2 arm serves with its scalar tail.
 * The reducer is the dot2 arm's, copied here: the two arms must not share a symbol through a
 * header, and twelve lines are cheaper than the coupling.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;           // eight waves, the default
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kKTile        = 128;           // k a staged tile covers, eight wmma steps
constexpr int kAPitch       = kKTile / 2 + 4;   // dwords a staged a row, the dot2 arm's 68
constexpr int kUnit         = 16;            // k a wmma step consumes: a slab's law



/* One dword of two bf16 into the fragment's two shorts: the low half first, the high half
 * after -- the packed order the builtin's VGPR map expects. A vector element is not a
 * modifiable lvalue, so the unpack is a constructor argument, not a reference target. */
#define R3D_U16(w) ((short)((w) & 0xffffu)), ((short)((w) >> 16))

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint16_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

/* y[M,N] = a[M,K] @ b[N,K]^T, one 16x16 tile a wave.
 *
 * `kslab` is the k range a blockIdx.y owns; with sk = 1 it is all of K and the store is the
 * narrowed bf16 the caller asked for. With sk > 1 every split writes its own f32 partial,
 * dense [sk][M][N], and the reducer adds them in split order. */
template <bool NT, bool SPLIT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m16_wmma_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, float* __restrict__ Part,
    int M, int N, int K, int a_ld, int y_ld, int w_ld, int kslab)
{
    /* Staged activation in the checkpoint's own bytes, a dword array at the dot2 arm's pitch. */
    __shared__ uint32_t s_a[16 * kAPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t waves = blockDim.x / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = ((uint32_t)blockIdx.x * waves + wave) * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));

    const int kbeg = (int)blockIdx.y * kslab;
    const int kend = kbeg + kslab < K ? kbeg + kslab : K;

    const uint16_t* w = B + (size_t)n_real * w_ld;
    v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int kt = kbeg; kt < kend; kt += kKTile) {
        const int kl = std::min(kKTile, kend - kt);

        /* Stage 16 rows x kl bf16, eight elements a unit, 256 threads. A row past M re-reads
         * the last real row, whose products the store drops; a unit past kl zero-fills. */
        const int nu = (kl + 7) / 8;
        for (int i = (int)tid; i < 16 * nu; i += (int)blockDim.x) {
            const int row = i / nu, c = (i % nu) * 8;
            const int rr = row < M ? row : M - 1;
            const uint16_t* src = A + (size_t)rr * a_ld + kt + c;
            uint32_t* dst = s_a + (size_t)row * kAPitch + c / 2;
            if (c + 8 <= kl) {
                const r3d_u32x4 q = *(const __attribute__((address_space(1))) r3d_u32x4*)src;
                *(r3d_u32x4*)dst = q;
            } else {
                uint32_t p[4] = { 0, 0, 0, 0 };
                for (int e = 0; e < kl - c; e += 2) {
                    const uint32_t lo = ldg_u16(src + e);
                    const uint32_t hi = (e + 1 < kl - c) ? ldg_u16(src + e + 1) : 0u;
                    p[e / 2] = lo | (hi << 16);
                }
                for (int e = 0; e < (kl - c + 1) / 2; ++e) dst[e] = p[e];
            }
        }
        group_barrier();

        const uint32_t* ap = s_a + (size_t)r * kAPitch;
        for (int k = 0; k < kl; k += 16) {
            /* Eight wmma steps per tile: two 16-byte reads a lane, the row's k pair at
             * dword k/2 and k/2 + 4, and the column's from global. */
            const r3d_u32x4 a0 = *(const r3d_u32x4*)(ap + k / 2);
            const r3d_u32x4 a1 = *(const r3d_u32x4*)(ap + k / 2 + 4);
            const r3d_u32x4 b0 = ldw<NT>(w + kt + k);
            const r3d_u32x4 b1 = ldw<NT>(w + kt + k + 8);
            const v16short fa = v16short{ R3D_U16(a0.x), R3D_U16(a0.y), R3D_U16(a0.z),
                                          R3D_U16(a0.w), R3D_U16(a1.x), R3D_U16(a1.y),
                                          R3D_U16(a1.z), R3D_U16(a1.w) };
            const v16short fb = v16short{ R3D_U16(b0.x), R3D_U16(b0.y), R3D_U16(b0.z),
                                          R3D_U16(b0.w), R3D_U16(b1.x), R3D_U16(b1.y),
                                          R3D_U16(b1.z), R3D_U16(b1.w) };
            acc = wmma_bf16(fa, fb, acc);
        }
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
    for (int g = 0; g < 8; ++g) {
        const int m = g * 2 + (int)(lane >> 4);
        if (m >= M) continue;
        if constexpr (SPLIT)
            Part[((size_t)blockIdx.y * M + m) * N + n0] = acc[g];
        else
            Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[g]);
    }
}
#undef R3D_U16

/* The split reduction: v = ((p0 + p1) + p2) + ... in ascending split order, then one
 * narrowing. The dot2 arm's, copied: the slab sums bracket the products differently than
 * one chain does, and the difference is in the last bits, the split's, not a bug in it. */
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_m16_wmma_split_reduce(
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

/* y[M,N] = a[M,K] @ b[N,K]^T, bf16 in and out, f32 accumulate, M <= 16, the wave-matrix arm.
 *
 * The dot2 arm's contract, with the slab law at 16 k. `sk` splits K (0 = the largest legal
 * split), `wv` waves per block (0 = 8), `nt` the weight load's lowering, `scratch` the f32
 * partial arena, sk * M * N * 4 bytes, required when the split is on. Returns 0, -1 a null
 * pointer, -2 a shape it does not serve, -3 a pitch it cannot read, -4 the HIP launch failed. */
extern "C" int r3d_gemm_nt_bf16_m16_wmma(long a, long b, long y, int M, int N, int K, int a_ld,
                                         int y_ld, int sk, int wv, int nt, long scratch,
                                         long stream)
{
    if (!a || !b || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > 16) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (sk == 0) {
        static const int kLadder[] = { 16, 8, 4, 2, 1 };
        for (int s : kLadder)
            if (s > 1 ? K % ((long long)s * kUnit) == 0 : true) { sk = s; break; }
    }
    if (sk < 1) return -2;
    /* The slab is kUnit deep: a ragged tail (K % kUnit) would stage a partial unit
     * the tile loop reads whole (stale dwords) and load B fragments past the weight,
     * so the shape is refused, as the dot2 arm's pick_sk already does. */
    if (K % ((long long)sk * kUnit)) return -2;
    if (sk > 1 && !scratch) return -1;
    if (wv <= 0) wv = kWaves;
    if (wv != 1 && wv != 2 && wv != 4 && wv != kWaves) return -2;

    const int kslab = K / sk;

#define R3D_WNT(NTV)                                                                           \
    do {                                                                                        \
        const int nt_ = (N + 15) / 16;                                                          \
        const dim3 grid((nt_ + wv - 1) / wv, (unsigned)sk), block((unsigned)(wv * kWave));      \
        if (sk > 1) {                                                                           \
            R3D_KLAUNCH(R3D_KNAME((gemm_m16_wmma_kernel<NTV, true>)), grid, block,              \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,           \
                        (uint16_t*)y, (float*)scratch, M, N, K, a_ld, y_ld, K, kslab);          \
            const int mn = M * N;                                                               \
            const int nb = std::min((mn + kBlockThreads - 1) / kBlockThreads, 4096);            \
            R3D_KLAUNCH(gemm_m16_wmma_split_reduce, dim3(nb), dim3(kBlockThreads),              \
                        (hipStream_t)stream, (const float*)scratch, (uint16_t*)y, M, N, y_ld,   \
                        sk);                                                                    \
        } else {                                                                                \
            R3D_KLAUNCH(R3D_KNAME((gemm_m16_wmma_kernel<NTV, false>)), grid, block,             \
                        (hipStream_t)stream, (const uint16_t*)a, (const uint16_t*)b,           \
                        (uint16_t*)y, (float*)nullptr, M, N, K, a_ld, y_ld, K, kslab);          \
        }                                                                                       \
    } while (0)
    if (nt) R3D_WNT(true);
    else R3D_WNT(false);
#undef R3D_WNT
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
