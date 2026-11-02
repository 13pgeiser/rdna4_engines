/* r3d_vocab_bf16_wmma.hip -- the prefill tiled GEMM on the wave-matrix: the WMMA twin of the
 * ALU fdot2 kernel in r3d_vocab_bf16.hip (the `gemm_nt_bf16_tiled` row's compute-dual).
 *
 *   y[M,N] = a[M,K] @ w[N,K]^T, bf16 in, f32 accumulate, bf16 out (one narrowing at the store).
 *
 * The ALU arm owns 64 f32 accumulators a lane (an 8x8 square) and runs v_fdot2; this arm runs the
 * same 128x128 tile on the wave-matrix, which is the matrix core at 512 FLOPS/clk/CU against the
 * ALU's ~256, so it is the faster of the pair at prefill heights. The tile is split eight ways
 * across the eight waves by column: wave w owns cols [w*16, w*16+16) and all 128 rows, so it
 * accumulates eight 16x16 wave-matrix tiles (one per 16-row block) -- 64 f32 accumulators a wave,
 * the same count the ALU arm keeps a lane. Every wave does unique work (the 16x128 wave-filling
 * retile), so no wave re-runs a tile.
 *
 * K is walked in 32-deep chunks (two 16-wide wave-matrix steps); the operands stage to LDS as bf16
 * (16 dwords a row, +1 bank-free) and load straight into the bf16 wave-matrix fragments.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kTM      = 128;           // activation rows a block
constexpr int kTN      = 128;           // weight cols a block
constexpr int kKC      = 32;            // k codes a chunk: two 16-wide wave-matrix steps
constexpr int kDW      = kKC / 2;       // dwords a row: 32 bf16 are 16 dwords (two bf16 a dword)
constexpr int kPitch   = kDW + 1;       // dwords a staged row: +1, bank-free

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void vocab_bf16_wmma_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ w,
    uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int w_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t sA[kTM * kPitch];   // 128 rows x 16 dwords
    __shared__ __align__(16) uint32_t sB[kTN * kPitch];   // 128 cols x 16 dwords

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wv = tid / kWave;          // the wave index (0..7): owns cols wv*16..wv*16+16
    const uint32_t r = lane & 15;

    const uint32_t m0 = (uint32_t)blockIdx.y * kTM;
    const uint32_t n0 = (uint32_t)blockIdx.x * kTN;

    // acc[8]: one 16x16 wave-matrix tile per 16-row block (128 rows / 16).
    v8float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int k0 = 0; k0 < K; k0 += kKC) {
        // Stage 128 activation rows and 128 weight cols (16 dwords each), clamped at the edge.
        for (int i = (int)tid; i < kTM * kDW; i += kThreads) {
            const int row = i / kDW, d = i % kDW;
            const int mm = (int)(m0 + row < (uint32_t)M ? m0 + row : (uint32_t)M - 1);
            sA[row * kPitch + d] = *(const uint32_t*)(a + (size_t)mm * a_ld + k0 + 2 * d);
        }
        for (int i = (int)tid; i < kTN * kDW; i += kThreads) {
            const int col = i / kDW, d = i % kDW;
            const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
            sB[col * kPitch + d] = *(const uint32_t*)(w + (size_t)nn * w_ld + k0 + 2 * d);
        }
        group_barrier();

        // Wave wv owns cols wv*16..wv*16+16, all eight 16-row blocks.
        const uint32_t* wp = sB + (size_t)(wv * 16 + r) * kPitch;
        #pragma unroll
        for (int rb = 0; rb < 8; ++rb) {
            const uint32_t* ap = sA + (size_t)(rb * 16 + r) * kPitch;
            #pragma unroll
            for (int k = 0; k < kKC; k += 16) {
                v16short fa, fb;
                #pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const uint32_t dw = ap[k / 2 + g];
                    fa[2 * g] = (short)(dw & 0xffffu); fa[2 * g + 1] = (short)(dw >> 16);
                    const uint32_t dwb = wp[k / 2 + g];
                    fb[2 * g] = (short)(dwb & 0xffffu); fb[2 * g + 1] = (short)(dwb >> 16);
                }
                acc[rb] = wmma_bf16(fa, fb, acc[rb]);
            }
        }
        group_barrier();
    }

    // Store: C-frag lane l, VGPR g -> row 2*(l>>4)+g, col l&15; offset by the row block and wave.
    #pragma unroll
    for (int rb = 0; rb < 8; ++rb)
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int m = m0 + rb * 16 + 2 * g + (lane >> 4);
            const int n = n0 + wv * 16 + r;
            if (m < M && n < N) y[m * y_ld + n] = f32_to_bf16(acc[rb][g]);
        }
}

int r3d_vocab_bf16_wmma(long av, long wv, long yv, int M, int N, int K, int a_ld, int w_ld,
                        int y_ld, long stream)
{
    if (!av || !wv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (K % kKC) return -2;   // the tile's k-chunk; the M/N tails clamp and are dropped at the store
    if (!a_ld) a_ld = K;
    if (!w_ld) w_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K || w_ld < K || y_ld < N) return -3;
    if ((av & 15) || (wv & 15) || (yv & 15)) return -5;

    const uint16_t* a = (const uint16_t*)(uintptr_t)av;
    const uint16_t* w = (const uint16_t*)(uintptr_t)wv;
    uint16_t* y = (uint16_t*)(uintptr_t)yv;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;

    dim3 grid((unsigned)(N + kTN - 1) / kTN, (unsigned)(M + kTM - 1) / kTM), blk(kThreads);
    vocab_bf16_wmma_kernel<<<grid, blk, 0, s>>>(a, w, y, M, N, K, a_ld, w_ld, y_ld);
    return (int)hipGetLastError();
}

}  // namespace

extern "C" {

int r3d_gemm_nt_bf16_tiled_wmma(long a, long b, long y, int M, int N, int K, int a_ld, int w_ld,
                                int y_ld, long stream)
{ return r3d_vocab_bf16_wmma(a, b, y, M, N, K, a_ld, w_ld, y_ld, stream); }

int r3d_rad_gemm_nt_bf16_tiled_wmma(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* at = &a->t[0], *wt = &a->t[1], *yt = &a->t[2];
    if (!r3d_p(at) || !r3d_p(wt) || !r3d_wp(yt)) return R3D_NO(RAD_E_INVAL);
    if (at->dtype != RAD_BF16 || wt->dtype != RAD_BF16 || yt->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);   // the plain bf16 head; the quantised form is the ALU arm's
    long long M = 0, K = 0;
    if (r3d_rows_of(at, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    const long long N = (wt->rank >= 2) ? wt->shape[0] : 0;
    if (M <= 0 || N <= 0 || K <= 0 || (K & 31))
        return R3D_NO(RAD_E_SHAPE);   // the 128x128 tile, 32-deep k chunks; the M/N tails clamp
    const int a_ld = (int)at->stride[0];
    const int w_ld = (int)wt->stride[0];
    const int y_ld = (int)yt->stride[0];
    return r3d_vocab_bf16_wmma((long)(uintptr_t)r3d_p(at), (long)(uintptr_t)r3d_p(wt),
                               (long)(uintptr_t)r3d_wp(yt), (int)M, (int)N, (int)K, a_ld, w_ld,
                               y_ld, (long)(uintptr_t)s);
}

}  // extern "C"
