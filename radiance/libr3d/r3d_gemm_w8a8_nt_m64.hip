/* r3d_gemm_w8a8_nt_m64.hip -- the wave-matrix arm of the w8a8 grid at the M = 17..127 band.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, int8 a against signed int8 codes, the scale group 128.
 *
 * The m16 arm (r3d_gemm_inta8_wmma) owns one 16-row tile a wave; above M = 16 this arm takes
 * MT = 2 16-row tiles a wave and walks them with MT accumulators. The B-fragment (the weight
 * column) is loaded once per 16-k step and reused over the MT row tiles, so the weight stream
 * is read once per 16-column tile no matter how many row tiles the wave carries -- the
 * activation, not the weight, is the traffic that grows with MT.
 *
 * ---- the group law --------------------------------------------------------------------------
 *
 * The m16 arm's exactness: a group's 128 products are |a * code| <= 127 * 127, their sum
 * 2,064,512 an exact integer far inside i32, so the eight wmma steps of a group chain an i32
 * sub-sum with no rounding, and the group term is the reference's
 *
 *   acc += (sub * a_scale[m]) * b_scale[n, g]
 *
 * in f32, in ascending group order -- the same order the dot2 arm's shared-memory walk forms.
 *
 * ---- the B pointer --------------------------------------------------------------------------
 *
 * The weight row base `w = B + n_real * b_rb` carries NO group offset; the group's k-base
 * enters the fragment read exactly once, as `w + kt + k` with kt = g * 128. Adding the group
 * offset to the row base AND the step read doubles it and reads a foreign weight slice -- the
 * per-n blow-up the prior draft hit. The A is staged per group into LDS, so its fragment read
 * is the group-local step offset; both operands land on the same logical k.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * MT = 2: two f32 accumulators, two i32 sub-sums, sixteen A and sixteen B fragment VGPRs, the
 * scales and the pointers, about 66 a lane. Eight waves a SIMD ask 528 of the 1024 the gfx11
 * SIMD keeps, so the eight-wave block stands. (A four-tile draft reached ~130 a lane, 1040 for
 * eight waves, and the overflow miscompiled the sub-sum chain.)
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
constexpr int MT            = 2;              /* 16-row tiles a wave: 32 rows a block */


/* Signed i8 in, signed i32 out: the builtin's sign selectors take the operands' signedness as
 * compile-time constants, both signed for this grid. */

/* Column c sits at byte step_pos(c), bits 2 and 3 of c swapped, the map its own inverse. */
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

/* y[M,N] = a[M,K] @ b[N,K]^T, MT 16-row tiles a wave, one 16-column tile a wave, one scale
 * group a k tile. The dot2 arm's contract: K a multiple of the group, the group term the
 * reference's, in ascending group order. */
template <bool NT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_w8a8_wmma_m64_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t s_a[MT * 16 * kPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16 + wave * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const int mbase = (int)blockIdx.y * MT * 16;    // this block's first row

    /* The lane's eight output rows per tile, and their activation scales, hoisted: a_scale is a
     * row property, constant across the group walk. */
    float as[MT][8];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
    #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
            as[i][g2] = sa[m < M ? m : M - 1];
        }

    const uint8_t* w = B + (size_t)n_real * b_rb;   // the row base: NO group offset
    v8float acc[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int g = 0; g < K / kGroup; ++g) {
        const int kt = g * kGroup;

        /* Stage MT*16 rows x 128 i8, the step order unpermuting on the way in. A row past M
         * re-reads the last real row, whose products the store drops. */
        for (int i = (int)tid; i < MT * 16 * (kGroup / 4); i += kBlockThreads) {
            const int row = i / (kGroup / 4), d = i % (kGroup / 4);
            const int rr = mbase + row;
            const uint32_t src =
                *(const uint32_t*)(a + (size_t)(rr < M ? rr : M - 1) * a_ld + kt + dword_pos(d));
            s_a[(size_t)row * kPitch + d] = src;
        }
        group_barrier();

        #pragma unroll
        for (int i = 0; i < MT; ++i) {
            const uint32_t* ap = s_a + (size_t)(i * 16 + r) * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                // The group's k-base enters once (kt); the row base carries none.
                const r3d_u32x4 bq = ldw<NT>(w + kt + k);
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w },
                              v4int{ (int)bq.x, (int)bq.y, (int)bq.z, (int)bq.w }, sub);
            }
            /* The group term, the reference's parentheses. The b_scale load, its drain, its
             * convert, and the eight fmacs run as one asm block: the load goes straight from
             * global, the drain is a vmcnt this asm owns, and the scale's register is written
             * only by this block's global_load, so no fragment load can clobber it mid-group.
             * v20 is named in the clobber because this LLVM aliases a constrained asm output
             * with a dead input register; the clobber keeps every t/acc/sub register out of
             * v20 across the block. The hardware f16->f32 is exact for normals. */
            float t[8];
            #pragma unroll
            for (int g2 = 0; g2 < 8; ++g2)
                t[g2] = (float)sub[g2] * as[i][g2];
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
                    : [a0] "+v"(acc[i][0]), [a1] "+v"(acc[i][1]), [a2] "+v"(acc[i][2]),
                      [a3] "+v"(acc[i][3]), [a4] "+v"(acc[i][4]), [a5] "+v"(acc[i][5]),
                      [a6] "+v"(acc[i][6]), [a7] "+v"(acc[i][7])
                    : [ga] "v"(ga),
                      [t0] "v"(t[0]), [t1] "v"(t[1]), [t2] "v"(t[2]), [t3] "v"(t[3]),
                      [t4] "v"(t[4]), [t5] "v"(t[5]), [t6] "v"(t[6]), [t7] "v"(t[7])
                    : "v20");
            }
        }
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
    #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
            if (m < M) Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[i][g2]);
        }
}

}  // namespace

/* y[M,N] = a[M,K] @ b[N,K]^T over the w8a8 grid, the wave-matrix arm at the M = 17..127 band.
 *
 * The dot2 arm's contract: K a multiple of the group, the group term the reference's. `nt`
 * picks the weight load's lowering. Returns 0, -1 a null pointer, -2 a shape it does not
 * serve, -3 a pitch it cannot read, -4 the HIP launch failed, -5 an alignment it cannot
 * read. */
extern "C" int r3d_gemm_nt_q_w8a8_m64_wmma(long a, long a_scale, long b, long b_scale, long y,
                                           int M, int N, int K, int a_ld, int y_ld,
                                           long long b_rb, long long sb_rb, int nt, long stream)
{
    if (!a || !a_scale || !b || !b_scale || !y) return -1;
    if (M < 17 || N < 1 || K < 1) return -2;
    if (M > 127) return -2;
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

    const int mrowblk = (M + MT * 16 - 1) / (MT * 16);
    const dim3 grid((N + kWaves * 16 - 1) / (kWaves * 16), (unsigned)mrowblk), block(kBlockThreads);
    if (nt)
        R3D_KLAUNCH(gemm_w8a8_wmma_m64_kernel<true>, grid, block, (hipStream_t)stream,
                    (const int8_t*)a, (const float*)a_scale, (const uint8_t*)b, (int)b_rb,
                    (const uint16_t*)b_scale, (int)sb_rb, (uint16_t*)y, M, N, K, a_ld, y_ld);
    else
        R3D_KLAUNCH(gemm_w8a8_wmma_m64_kernel<false>, grid, block, (hipStream_t)stream,
                    (const int8_t*)a, (const float*)a_scale, (const uint8_t*)b, (int)b_rb,
                    (const uint16_t*)b_scale, (int)sb_rb, (uint16_t*)y, M, N, K, a_ld, y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ==================================================================================================
 * The RadArgs shim. Schema positions: a 0, a_scale 1, b 2, b_scale 3, y 4, a_sum 5, b_ref 6. The
 * m64 band is 17 <= M <= 127; M <= 16 is the m16 arm's, so the two rows never compete.
 * ================================================================================================== */
extern "C" int r3d_rad_gemm_nt_q_w8a8_m64(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "w8a8") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_i8(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);
    if (BS->dtype != RAD_F16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 17 || M > 127) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);

    const int rc = r3d_gemm_nt_q_w8a8_m64_wmma(r3d_addr(A), r3d_addr(AS), r3d_addr(B),
                                               r3d_addr(BS), r3d_addr(Y), (int)M, (int)N,
                                               (int)K, (int)ap, (int)yp,
                                               (long long)r3d_row_pitch(B),
                                               (long long)r3d_row_pitch(BS), c.nt, (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
