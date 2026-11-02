/* r3d_gemm_inta8_wmma.hip -- the wave-matrix arm of the w8a8 grid at decode heights.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, int8 a against signed int8 codes, M <= 16.
 *
 * The dot2 arm (r3d_gemm_inta8.hip) owns one output element a lane and folds each 128 group's
 * products through f16 pairs; this arm issues V_WMMA_I32_16X16X16_IU8 over a 16x16x16 tile, a
 * wave a tile, and the two arms are two points of the row's `impl` axis. The tune cache picks
 * the point per shape per machine; an untuned install runs the dot2 arm, which is the axis's
 * default.
 *
 * ---- the group law --------------------------------------------------------------------------
 *
 * The dot2 arm's exactness is the arm's: a group's 128 products are |a * code| <= 127 * 127,
 * and their sum 2,064,512 is an exact integer far inside i32, so the eight wmma steps of a
 * group chain an i32 sub-sum with no rounding at all, and the group term is the reference's
 *
 *   acc += (sub * a_scale[m]) * b_scale[n, g]
 *
 * in f32, in ascending group order -- the same order the dot2 arm's shared-memory walk forms.
 * What the arm changes is the issue: eight wmma where the dot2 arm issues its decoded dot4
 * folds and a pair fold, and the f16 widening the dot2 arm pays on the codes is not here at
 * all, the i8 bytes going into the fragment directly.
 *
 * ---- where the bytes come from --------------------------------------------------------------
 *
 * The activation is staged in LDS as plain i8, the step order unpermuting on the way in: the
 * container's dword at logical position d sits at byte offset dword_pos(d) (the dot2 arm's
 * map, its own inverse), so the staging reads the swapped position and writes the plain one. One k tile is 16 rows x 128 i8, one scale group, 2.1 KiB with the pitch, and
 * carries eight wmma steps. The pitch is 33 dwords: 33 mod 32 = 1, so the lanes of a
 * ds_read phase land one bank apart and the lane's single 16-byte fragment read is
 * conflict-free in its phase.
 *
 * The weight streams from global, one 16-byte load a lane per step (column L % 16 of the
 * tile, the step's 16 k), the NT axis choosing the lowering the bf16 arm's header describes.
 * Every weight byte is read once per output tile, the stream the dot2 arm streams.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * Eight f32 accumulators, eight i32 sub-sums, four A and four B fragment VGPRs and the
 * pointers: about 30 a lane. Eight waves a SIMD asks 240 of the 1024 the gfx11 SIMD keeps,
 * so the quantised rows' fixed eight-wave block stands.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kWaves        = 8;
constexpr int kBlockThreads = kWaves * kWave;
constexpr int kGroup        = 128;            /* the scale group: one k tile, eight wmma steps */
constexpr int kPitch        = 33;             /* dwords a staged a row: 33 mod 32 = 1, bank-free */


/* Signed i8 in, signed i32 out: the builtin's sign selectors take the operands' signedness as
 * compile-time constants, both signed for this grid. */

/* Column c sits at byte step_pos(c), bits 2 and 3 of c swapped, the map its own inverse --
 * the dot2 arm's, so the staging's read position is the plain position run through it. */
__device__ __forceinline__ int64_t dword_pos(int64_t d)
{
    return 4 * ((d & ~(int64_t)3) | ((d & 1) << 1) | ((d & 2) >> 1));
}

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint8_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

/* y[M,N] = a[M,K] @ b[N,K]^T, one 16x16 tile a wave, one scale group a k tile.
 *
 * The dot2 arm's contract: K a multiple of the group, the group term the reference's, in
 * ascending group order. */
template <bool NT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_w8a8_wmma_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t s_a[16 * kPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16 + wave * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));

    /* The lane's eight output rows, and their activation scales, hoisted: a_scale is a row
     * property, constant across the group walk. */
    float as[8];
    #pragma unroll
    for (int g2 = 0; g2 < 8; ++g2) {
        const int m = g2 * 2 + (int)(lane >> 4);
        as[g2] = sa[m < M ? m : M - 1];
    }

    const uint8_t* w = B + (size_t)n_real * b_rb;
    v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int g = 0; g < K / kGroup; ++g) {
        const int kt = g * kGroup;

        /* Stage 16 rows x 128 i8, the step order unpermuting on the way in. A row past M
         * re-reads the last real row, whose products the store drops. */
        for (int i = (int)tid; i < 16 * (kGroup / 4); i += kBlockThreads) {
            const int row = i / (kGroup / 4), d = i % (kGroup / 4);
            const int rr = row < M ? row : M - 1;
            /* The dot2 arm's read: a_ld is in elements (bytes, for i8) and dword_pos is already
             * the byte offset, the map's own inverse; kt walks the group along the row. */
            const uint32_t src =
                *(const uint32_t*)(a + (size_t)rr * a_ld + kt + dword_pos(d));
            s_a[(size_t)row * kPitch + d] = src;
        }
        group_barrier();

        const uint32_t* ap = s_a + (size_t)r * kPitch;
        v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int k = 0; k < kGroup; k += 16) {
            const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
            const r3d_u32x4 bq = ldw<NT>(w + kt + k);
            sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w },
                          v4int{ (int)bq.x, (int)bq.y, (int)bq.z, (int)bq.w }, sub);
        }
        /* The group term, the reference's parentheses, in ascending group order. The b_scale
         * load, its drain, its convert, and the eight fmacs run as one asm block: the load
         * goes straight from global (staging it through LDS lets the scheduler move the
         * store past the barrier, so a group reads its predecessor's scale), the drain is a
         * vmcnt this asm owns, and the scale's register is written only by this block's
         * global_load, so no fragment load can clobber it mid-group. v20 is named in the
         * clobber because this LLVM aliases a constrained asm output with a dead input
         * register; the clobber keeps every t/acc/sub register out of v20 across the block.
         * The hardware f16->f32 is exact for normals; subnormal halves flush to 0, below any
         * scale this kernel sees and far below the quantisation noise. */
        float t[8];
        #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2)
            t[g2] = (float)sub[g2] * as[g2];
        {
            const uint64_t ga = (uint64_t)(sb + (size_t)n_real * sb_rb + g);
            asm volatile(
                "global_load_u16 v20, %[ga], off\n\t"
                "s_waitcnt vmcnt(0)\n\t"
                "v_cvt_f32_f16_e32 v20, v20.l\n\t"
                "v_fmac_f32_e32 %[a0], %[t0], v20\n\t"
                "v_fmac_f32_e32 %[a1], %[t1], v20\n\t"
                "v_fmac_f32_e32 %[a2], %[t2], v20\n\t"
                "v_fmac_f32_e32 %[a3], %[t3], v20\n\t"
                "v_fmac_f32_e32 %[a4], %[t4], v20\n\t"
                "v_fmac_f32_e32 %[a5], %[t5], v20\n\t"
                "v_fmac_f32_e32 %[a6], %[t6], v20\n\t"
                "v_fmac_f32_e32 %[a7], %[t7], v20"
                : [a0] "+v"(acc[0]), [a1] "+v"(acc[1]), [a2] "+v"(acc[2]), [a3] "+v"(acc[3]),
                  [a4] "+v"(acc[4]), [a5] "+v"(acc[5]), [a6] "+v"(acc[6]), [a7] "+v"(acc[7])
                : [ga] "v"(ga),
                  [t0] "v"(t[0]), [t1] "v"(t[1]), [t2] "v"(t[2]), [t3] "v"(t[3]),
                  [t4] "v"(t[4]), [t5] "v"(t[5]), [t6] "v"(t[6]), [t7] "v"(t[7])
                : "v20");
        }
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
    #pragma unroll
    for (int g2 = 0; g2 < 8; ++g2) {
        const int m = g2 * 2 + (int)(lane >> 4);
        if (m < M) Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[g2]);
    }
}

}  // namespace

/* y[M,N] = a[M,K] @ b[N,K]^T over the w8a8 grid, the wave-matrix arm.
 *
 * The dot2 arm's contract: K a multiple of the group, the group term the reference's. `nt`
 * picks the weight load's lowering. Returns 0, -1 a null pointer, -2 a shape it does not
 * serve, -3 a pitch it cannot read, -4 the HIP launch failed, -5 an alignment it cannot
 * read. */
extern "C" int r3d_gemm_nt_q_w8a8_m16_wmma(long a, long a_scale, long b, long b_scale, long y,
                                           int M, int N, int K, int a_ld, int y_ld,
                                           long long b_rb, long long sb_rb, int nt, long stream)
{
    if (!a || !a_scale || !b || !b_scale || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > 16) return -2;
    if (K % kGroup) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld % 4) return -3;
    if (y_ld < N) return -3;
    if (!b_rb) b_rb = K;
    if (b_rb < K) return -3;
    if (b_rb & 15) return -5;
    if (!sb_rb) sb_rb = K / kGroup;
    if (sb_rb < K / kGroup) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)b & 15u)) return -5;

    const dim3 grid((N + kWaves * 16 - 1) / (kWaves * 16)), block(kBlockThreads);
    if (nt)
        R3D_KLAUNCH(gemm_w8a8_wmma_kernel<true>, grid, block, (hipStream_t)stream,
                    (const int8_t*)a, (const float*)a_scale, (const uint8_t*)b, (int)b_rb,
                    (const uint16_t*)b_scale, (int)sb_rb, (uint16_t*)y, M, N, K, a_ld, y_ld);
    else
        R3D_KLAUNCH(gemm_w8a8_wmma_kernel<false>, grid, block, (hipStream_t)stream,
                    (const int8_t*)a, (const float*)a_scale, (const uint8_t*)b, (int)b_rb,
                    (const uint16_t*)b_scale, (int)sb_rb, (uint16_t*)y, M, N, K, a_ld, y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
