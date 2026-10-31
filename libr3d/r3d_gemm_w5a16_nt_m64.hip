/* r3d_gemm_w5a16_nt_m64.hip -- the 5-bit-weight skinny GEMM, M up to 64.
 *
 * The w4a16 row with a fifth bit per weight, for ParoQuant's int5 (asymmetric: an f16 scale and
 * a 5-bit integer zero per 128 of K). The weight arrives as three planes behind the one
 * b_scale operand, joined rad.planes, the codes as the B operand in the container's order:
 *
 *     codes   [N][K/2]   u4, the w4a16 fragment order (element e is k = e)
 *     scale   [N][K/128] f16
 *     zero    [N][K/128] u8, 0..31
 *     hi      [N/16][K/2] u32: word (t, kb, lane) holds the fifth bits of tile t's k block
 *               kb, lane's 32 weights: byte s is k step s of the 64-wide block, bit e is
 *               fragment element e
 *
 * The dequant is the w4a16's integer code minus the zero point, the code widened with its fifth
 * bit (0..31, exact in f16 after the subtract: the range is [-31, 31]). The rest of the
 * kernel is the w4a16 row's, and its header notes apply: the block-level split, the
 * block-cooperative LDS stage, the runtime trip counts, the pinned barriers, the 128-VGPR
 * ceiling's npw==4 && mb>1 refusal.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"
#include "rad_planes.h"

#include <algorithm>
#include <cstring>

namespace {

constexpr int kMaxM    = 64;
constexpr int kGroup   = 128;
constexpr int kStep    = 16;
constexpr int kWave32  = 32;
constexpr int kAPitch  = 68;    // dwords per staged A row (128 + 8 f16, the m16 pitch)


/* One dword of two f16 into the fragment's two halves, low first: the packed order the
 * builtin's VGPR map wants, a bit pattern not a value. */
#define R3D_W5_F16(w) __builtin_bit_cast(_Float16, (uint16_t)((w) & 0xffffu)), \
                      __builtin_bit_cast(_Float16, (uint16_t)((w) >> 16))

/* Two dwords of sixteen 4-bit codes plus the two bytes of their fifth bits -> one 16x16x16
 * f16 B fragment, minus the zero point. Element e is the code for k = e, the k-contiguous
 * map the card's tile wants: nibble 2e (e < 4), 2(e-4)+1 (e < 8), then w1's 2(e-8),
 * 2(e-12)+1.
 *
 * The fifth bits: a byte covers eight k, the even ones in its low four bits (bit l/2), the
 * odd in its high four (bit 4 + l/2), l the k within the eight -- the reference
 * quantiser's spread16 form. The step's sixteen k are two bytes: hb0 is k 0..7, hb1 is
 * k 8..15.
 *
 * The value is code - zero as an integer, converted once per element: the range is
 * [-31, 31], exact in f16. No offset form, no half arithmetic: the w4a16 row's probe
 * history is the reason (the packed-half construction lost registers across the unrolled
 * WMMA chain). */
__device__ __forceinline__ v16half dequant5(unsigned w0, unsigned w1, unsigned hb0,
                                            unsigned hb1, int z)
{
    /* The bits are extracted inline, not materialised: sixteen named bit registers push
     * the kernel over the 128-VGPR ceiling and the A fragment loads collapse.
     *
     * The nibbles are k-contiguous (w0 = kk 0..7, w1 = kk 8..15) but the fifth-bit
     * words are the reference layout's lh split: h0 holds kk 0,1,2,3,8,9,10,11 at
     * bits 0,2,4,6,1,3,5,7 and h1 holds kk 4,5,6,7,12,13,14,15 at the same bit
     * order. Element e is kk e, so the bit source crosses the e 8 boundary: e 4..7
     * and 12..15 take h1, e 8..11 take h0 (probe C7: a k-dependent A exposes the
     * mispairing a per-row-constant A erases). */
    const unsigned h0 = hb0 & 0xFFu, h1 = hb1 & 0xFFu;
    const int c0 = (int)((w0 & 0xFu) | (((h0 >> 0) & 1u) << 4)) - z;
    const int c1 = (int)(((w0 >> 4) & 0xFu) | (((h0 >> 2) & 1u) << 4)) - z;
    const int c2 = (int)(((w0 >> 8) & 0xFu) | (((h0 >> 4) & 1u) << 4)) - z;
    const int c3 = (int)(((w0 >> 12) & 0xFu) | (((h0 >> 6) & 1u) << 4)) - z;
    const int c4 = (int)(((w0 >> 16) & 0xFu) | (((h1 >> 0) & 1u) << 4)) - z;
    const int c5 = (int)(((w0 >> 20) & 0xFu) | (((h1 >> 2) & 1u) << 4)) - z;
    const int c6 = (int)(((w0 >> 24) & 0xFu) | (((h1 >> 4) & 1u) << 4)) - z;
    const int c7 = (int)(((w0 >> 28) & 0xFu) | (((h1 >> 6) & 1u) << 4)) - z;
    const int c8 = (int)((w1 & 0xFu) | (((h0 >> 1) & 1u) << 4)) - z;
    const int c9 = (int)(((w1 >> 4) & 0xFu) | (((h0 >> 3) & 1u) << 4)) - z;
    const int c10 = (int)(((w1 >> 8) & 0xFu) | (((h0 >> 5) & 1u) << 4)) - z;
    const int c11 = (int)(((w1 >> 12) & 0xFu) | (((h0 >> 7) & 1u) << 4)) - z;
    const int c12 = (int)(((w1 >> 16) & 0xFu) | (((h1 >> 1) & 1u) << 4)) - z;
    const int c13 = (int)(((w1 >> 20) & 0xFu) | (((h1 >> 3) & 1u) << 4)) - z;
    const int c14 = (int)(((w1 >> 24) & 0xFu) | (((h1 >> 5) & 1u) << 4)) - z;
    const int c15 = (int)(((w1 >> 28) & 0xFu) | (((h1 >> 7) & 1u) << 4)) - z;
    return v16half{ __float2half((float)c0),  __float2half((float)c1),
                    __float2half((float)c2),  __float2half((float)c3),
                    __float2half((float)c4),  __float2half((float)c5),
                    __float2half((float)c6),  __float2half((float)c7),
                    __float2half((float)c8),  __float2half((float)c9),
                    __float2half((float)c10), __float2half((float)c11),
                    __float2half((float)c12), __float2half((float)c13),
                    __float2half((float)c14), __float2half((float)c15) };
}

/* A block barrier the backend cannot move; see the w4a16 row's note. */
#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

/* y[M,N] = a[M,K] @ dequant(wq, hi)[N,K]^T. The w4a16 row's kernel with the fifth-bit
 * plane: per group the two 64-wide blocks' words, one a lane, then the step's byte. */
template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024, 1) void r3d_w5a16_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Bsz, size_t off_scale, size_t off_zero, size_t off_hi,
    __hip_bfloat16* __restrict__ C, float* __restrict__ Scr,
    int M, int K, int N, int a_ld, int y_ld, int WV, int SK, int MTb)
{
    __shared__ uint32_t s_a[16 * kAPitch];

    const int tid = threadIdx.x, wave = tid / kWave32, lane = tid % kWave32;
    const int col_local = wave;
    const int ksplit = blockIdx.y % SK;
    const int nt0 = (blockIdx.x * WV + col_local) * NPW;
    const int n0 = nt0 * 16;
    const int mt0 = (blockIdx.y / SK) * MT;
    const int Ks = K / SK;
    const int steps = Ks / kStep;
    const int kbase = ksplit * Ks;
    const int G = K / kGroup;

    v8float acc[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++) acc[j][i] = (v8float)(0.f);

    if (n0 < N) {
        /* Weight: the lane's column, eight bytes a step; the fifth bits, one word a
         * 64-wide block; the planes indexed by the absolute group. */
        const int ntlast = N / 16 - 1;
        const unsigned char* pw[NPW];
        const unsigned short* ps[NPW];
        const unsigned char* pz[NPW];
        size_t hibase[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
            const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
            const size_t row = (size_t)t * 16 + (lane & 15);
            pw[j] = Wq + row * (K / 2) + kbase / 2;
            ps[j] = (const unsigned short*)(Bsz + off_scale) + row * G;
            pz[j] = (Bsz + off_zero) + row * G;
            /* The hi plane's byte base for this tile: [N/16][K/2] u32, the tile's
             * words are (K/2) u32 = K/2*4 bytes. */
            hibase[j] = (size_t)t * (K / 2) * 4;
        }

        const int spg = kGroup / kStep;
        const int spg_rt = (K % kGroup) + kGroup / kStep;
        const int ngroups = steps / spg;

        for (int i = 0; i < MT; i++) {
            const int mrow_base = (mt0 + i) * 16;
            for (int g = 0; g < ngroups; g++) {
                /* Stage 16 rows x 128 f16 of the group's slab, eight f16 a unit; the
                 * w4a16 row's guarded wave loop, its notes apply. */
                const int kslab = kbase + g * kGroup;
                const int nu = (K % kGroup) + kGroup / 8;
                const int nunits = 16 * nu;
                const int nwaves = (int)blockDim.x / kWave32;
                const int nstride = (int)blockDim.x;
                const int niter = (nunits + nstride - 1) / nstride;
#pragma unroll 1
                for (int w = 0; w < nwaves; w++) {
                    const int wtid = w * kWave32 + lane;
#pragma unroll 1
                    for (int k = 0; k < niter; k++) {
                        const int ui = wtid + k * nstride;
                        if (ui < nunits && (w * kWave32) <= tid) {
                            const int row = ui / nu, c = (ui % nu) * 8;
                            const int mrow = mrow_base + row;
                            const int mr = mrow < M ? mrow : M - 1;
                            const unsigned short* src = A + (size_t)(mr * a_ld + kslab + c);
                            uint32_t* dst = s_a + (size_t)row * kAPitch + c / 2;
                            *(r3d_u32x4*)dst = *(const r3d_u32x4*)src;
                        }
                    }
                }
                R3D_BARRIER();

                float sc[NPW];
                int z[NPW];
                const int g_abs = kbase / kGroup + g;
#pragma unroll
                for (int j = 0; j < NPW; j++) {
                    sc[j] = f16_to_f32((uint16_t)ps[j][(size_t)g_abs]);
                    z[j] = (int)pz[j][(size_t)g_abs];
                }

                /* The group's fifth-bit bytes, one a lane: the group is 128 wide, eight
                 * steps: steps 0..3 are 64-block 2g, steps 4..7 are block 2g+1. The
                 * lane's words are (t, kb, column) and (t, kb, 16 + column), the
                 * reference layout's lh split; the byte pointer is the word's first
                 * byte. */
                const int kb0 = kbase / (kGroup / 2) + 2 * g;

                /* The step's two fifth-bit bytes, loaded from global memory as bytes at
                 * plain runtime addresses -- the form the W loads use and the card
                 * honours. Every select shape tried before this is miscompiled: a
                 * runtime shift by the step's amount drops the bytes, an if/else chain
                 * leaks a branch into the other steps, a stack-array index collapses to
                 * 0, a mask-OR select leaks a value (probe C6). Two blocks of four
                 * steps; the block's byte pointer is set once per block. */
                v8float g_acc[NPW];
#pragma unroll
                for (int j = 0; j < NPW; j++) g_acc[j] = (v8float)(0.f);
                for (int b2 = 0; b2 < 2; b2++) {
                    for (int j = 0; j < NPW; j++) {
                        const unsigned char* p0 = (const unsigned char*)(Bsz + off_hi)
                                + hibase[j]
                                + (size_t)(kb0 + b2) * kWave32 * 4 + (lane & 15) * 4;
                        const unsigned char* p1 = p0 + 16 * 4;
                        for (int sb = 0; sb < 4; sb++) {
                            const int s = 4 * b2 + sb;
                            const int u = g * spg + s;
                            const unsigned long long wbj =
                                NT ? __builtin_nontemporal_load((const unsigned long long*)(pw[j] + 8 * u))
                                   : *(const unsigned long long*)(pw[j] + 8 * u);
                            const unsigned hb0 = (unsigned)*(p0 + sb);
                            const unsigned hb1 = (unsigned)*(p1 + sb);
                            const v16half bfrag =
                                dequant5((unsigned)wbj, (unsigned)(wbj >> 32), hb0, hb1, z[j]);
                            const uint32_t* ap = s_a + (size_t)(lane & 15) * kAPitch + s * 8;
                            const r3d_u32x4 a0 = *(const r3d_u32x4*)ap;
                            const r3d_u32x4 a1 = *(const r3d_u32x4*)(ap + 4);
                            const v16half afrag = v16half{ R3D_W5_F16(a0.x), R3D_W5_F16(a0.y),
                                                           R3D_W5_F16(a0.z), R3D_W5_F16(a0.w),
                                                           R3D_W5_F16(a1.x), R3D_W5_F16(a1.y),
                                                           R3D_W5_F16(a1.z), R3D_W5_F16(a1.w) };
                            g_acc[j] = wmma_f16(afrag, bfrag, g_acc[j]);
                        }
                    }
                }
                for (int j = 0; j < NPW; j++)
                    for (int e = 0; e < kWave32 / 4; e++)
                        acc[j][i][e] = fmaf(g_acc[j][e], sc[j], acc[j][i][e]);
                R3D_BARRIER();  // the next slab's stage must not race these reads
            }
        }
    }

    /* Epilogue: the w4a16 row's D map and store; the notes apply. */
    const int ncols = WV * NPW;
    for (int i = 0; i < MT; i++) {
        const int m0 = (mt0 + i) * 16;
        for (int j = 0; j < NPW; j++) {
            const int n0 = (nt0 + j) * 16;
            const int col_g = blockIdx.x * ncols + (col_local * NPW + j);
            for (int e = 0; e < kWave32 / 4; e++) {
                const int m = m0 + 2 * e + (lane >> 4);
                const int n = n0 + (lane & 15);
                if (SK == 1) {
                    if (m < M && n < N)
                        ((uint16_t*)C)[(size_t)m * y_ld + n] = f32_to_bf16(acc[j][i][e]);
                } else {
                    /* The pool is sized for ntiles; the grid's last column group strides
                     * past it when ntiles % ncols != 0. */
                    if (col_g < N / 16) {
                        const size_t o =
                            (((size_t)col_g * SK + ksplit) * MTb + i) * 256 + (e * kWave32 + lane);
                        Scr[o] = acc[j][i][e];
                    }
                }
            }
        }
    }
}

/* The split-K reduce: the w4a16 row's, the D map inverted. */
__global__ void r3d_w5a16_split_reduce(const float* __restrict__ Scr,
                                       __hip_bfloat16* __restrict__ C,
                                       int M, int N, int y_ld, int SK, int MTb, int ntiles)
{
    const int total = ntiles * MTb * 256;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const int tile = idx / 256, off = idx % 256;
        const int i = tile % MTb, col_g = tile / MTb;
        float v = 0.f;
        for (int s = 0; s < SK; s++)
            v += Scr[(((size_t)col_g * SK + s) * MTb + i) * 256 + off];
        const int l = off & 31, e = off >> 5;
        const int m = i * 16 + 2 * e + (l >> 4), n = col_g * 16 + (l & 15);
        if (m < M && n < N) ((uint16_t*)C)[(size_t)m * y_ld + n] = f32_to_bf16(v);
    }
}

/* ==================================================================== the launcher

 * y[M,N] = a[M,K] @ dequant(wq, hi)[N,K]^T:
 *
 *   a      [M][K]  f16
 *   wq     [N][K/2]  u4 codes, the fragment order
 *   bsz    the joined (scale, zero, hi) operand, rad.planes
 *   y      [M][N]  bf16, row-major
 *
 * `wv` waves per block (0 = 8), `sk` the K split (0 = 1), `mb` row tiles per block (0 = 1),
 * `npw` column tiles a wave (0 = 1; 1 or 4), `nt` the weight load's lowering. Returns 0,
 * -1 a null pointer, -2 a shape it does not serve, -4 the HIP launch failed. */
static float* s_scratch_pool = nullptr;
static size_t s_scratch_sz = 0;

extern "C" int r3d_gemm_nt_q_w5a16_m64(long a, long wq, long bsz, size_t off_scale,
                                       size_t off_zero, size_t off_hi, long y, int M, int N,
                                       int K, int a_ld, int y_ld,
                                       int wv, int sk, int mb, int npw, int nt,
                                       long stream)
{
    if (!a || !wq || !bsz || !y) return -1;
    if (M < 1 || M > kMaxM) return -2;
    if (N < 16 || N % 16) return -2;
    if (K < kGroup || K % kGroup) return -2;
    if (wv <= 0) wv = 8;
    if (sk <= 0) sk = 1;
    if (mb <= 0) mb = 1;
    if (npw <= 0) npw = 1;
    if (wv > 32 || sk > 32) return -2;
    if (mb < 1 || mb > 4) return -2;
    if (npw != 1 && npw != 4) return -2;
    if (K % (sk * kGroup)) return -2;
    /* npw=4 with mb>1 is the shape the backend cannot lower at the 128-VGPR ceiling; the
     * w4a16 row's probe and refusal apply. */
    if (npw == 4 && mb > 1) return -2;

    const int ntiles = N / 16, mtiles = (M + 15) / 16;
    const int ncols = wv * npw;
    const dim3 grid((ntiles + ncols - 1) / ncols, ((mtiles + mb - 1) / mb) * sk),
                block(wv * kWave32);

    float* scr = nullptr;
    if (sk > 1) {
        const size_t bytes = (size_t)ntiles * sk * mb * 256 * sizeof(float);
        if (bytes > s_scratch_sz) {
            if (s_scratch_pool) hipFree(s_scratch_pool);
            if (hipMalloc(&s_scratch_pool, bytes) != hipSuccess) return -4;
            s_scratch_sz = bytes;
        }
        scr = s_scratch_pool;
    }

#define R3D_W5L(MT, NP, TT) \
    R3D_KLAUNCH_S(R3D_KNAME((r3d_w5a16_m64_kernel<MT, NP, TT>)), grid, block, 0, \
                  (hipStream_t)stream, (const unsigned short*)a, (const unsigned char*)wq, \
                  (const unsigned char*)bsz, off_scale, off_zero, off_hi, (__hip_bfloat16*)y, \
                  scr, M, K, N, a_ld, y_ld, wv, sk, mb)
#define R3D_W5L_MB(NP, TT) \
    do { switch (mb) { \
        case 1:  R3D_W5L(1, NP, TT); break; \
        case 2:  R3D_W5L(2, NP, TT); break; \
        case 3:  R3D_W5L(3, NP, TT); break; \
        default: R3D_W5L(4, NP, TT); break; } } while (0)
    if (npw == 1) { if (nt) R3D_W5L_MB(1, true);  else R3D_W5L_MB(1, false); }
    else          { if (nt) R3D_W5L_MB(4, true);  else R3D_W5L_MB(4, false); }
#undef R3D_W5L_MB
#undef R3D_W5L
    if (hipGetLastError() != hipSuccess) return -4;

    if (sk > 1) {
        const int total = ntiles * mb * 256;
        const int rblock = 256;
        const int rgrid = (total + rblock - 1) / rblock;
        R3D_KLAUNCH_S(R3D_KNAME((r3d_w5a16_split_reduce)), dim3(rgrid), rblock, 0,
                      (hipStream_t)stream, scr, (__hip_bfloat16*)y, M, N, y_ld, sk, mb, ntiles);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

}  // namespace

/* ==================================================================== the plugin ABI */

enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y, Q_ASUM, Q_BREF };

extern "C" int r3d_rad_gemm_nt_q_w5a16_m64(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "w5a16") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A  = r3d_opt(a, Q_A);
    const RadTensor* AS = r3d_opt(a, Q_ASCALE);
    const RadTensor* B  = r3d_opt(a, Q_B);
    const RadTensor* BS = r3d_opt(a, Q_BSCALE);
    const RadTensor* Y  = r3d_opt(a, Q_Y);
    const RadTensor* AU = r3d_opt(a, Q_ASUM);
    const RadTensor* BR = r3d_opt(a, Q_BREF);
    if (!r3d_p(A) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    if (r3d_has(AS) || r3d_has(AU) || r3d_has(BR)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f16(A)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_U4) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (N % 16) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMaxM) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);

    /* The joined form: the scale and zero planes, the hi plane, and their byte offsets. */
    const RadPlanesEntry* es = nullptr, *ez = nullptr, *eh = nullptr;
    const uint8_t* sc_b = (const uint8_t*)rad_planes_find(r3d_p(BS), "scale", &es);
    const uint8_t* z_b = (const uint8_t*)rad_planes_find(r3d_p(BS), "zero", &ez);
    const uint8_t* h_b = (const uint8_t*)rad_planes_find(r3d_p(BS), "hi", &eh);
    if (!sc_b || !z_b || !h_b || es->dtype != RAD_F16 || ez->dtype != RAD_U8 ||
        eh->dtype != RAD_U32)
        return R3D_NO(RAD_E_DTYPE);
    if (es->rows < N || ez->rows < N || es->cols < K / kGroup || ez->cols < K / kGroup)
        return R3D_NO(RAD_E_SHAPE);
    if (eh->rows < N / 16) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N || yrows < M) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 16) || !r3d_aligned(r3d_p(B), 8) ||
        !r3d_aligned(h_b, 4))
        return R3D_NO(RAD_E_ALIGN);

    const int wv = (int)r3d_geti_or(a, "wv", 8);
    const int sk = (int)r3d_geti_or(a, "sk", 1);
    const int mb = (int)r3d_geti_or(a, "mb", 1);
    const int npw = (int)r3d_geti_or(a, "npw", 1);
    const int nt = (int)r3d_geti_or(a, "nt", 1);

    const int e = r3d_gemm_nt_q_w5a16_m64(r3d_addr(A), r3d_addr(B), r3d_addr(BS),
                                          (size_t)(sc_b - (const uint8_t*)r3d_p(BS)),
                                          (size_t)(z_b - (const uint8_t*)r3d_p(BS)),
                                          (size_t)(h_b - (const uint8_t*)r3d_p(BS)),
                                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                          wv, sk, mb, npw, nt, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE : RAD_E_DEVICE);
}

/* ==================================================================== the layout hooks
 *
 * The w4a16 row's scoped exception with a third plane: the codes are read where they lie
 * (the B operand, the container's order), and the scale, zero and hi planes join behind
 * the b_scale operand. NOTHING HERE MOVES A BYTE.
 */

extern "C" int r3d_layout_gemm_nt_q_w5a16(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out)
{
    if (n < 2 || operand != Q_BSCALE) return RAD_E_UNSUPPORTED;
    const char* d = r3d_pgets(p, n_p, "dtype");
    if (!d || std::strcmp(d, "w5a16") != 0) return RAD_E_DTYPE;
    const long long group = r3d_pgeti_or(p, n_p, "group", kGroup);
    if (group <= 0) return RAD_E_DTYPE;
    if (!enc || !rad_enc_is(enc, "affine")) return RAD_E_DTYPE;
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0)
        return RAD_E_DTYPE;
    if (enc->n_planes < 3) return RAD_E_DTYPE;
    if (n != 3) return RAD_E_DTYPE;
    const char* want[3] = { "scale", "zero", "hi" };
    for (int i = 0; i < 3; ++i) {
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_DTYPE;
        if (std::strncmp(enc->plane[sel[i]].role, want[i], RAD_ENC_STR) != 0)
            return RAD_E_DTYPE;
    }
    /* The codes plane is the one this operand does not take: U4, block {1,1}. */
    int codes = -1;
    for (int i = 0; i < enc->n_planes; ++i)
        if (std::strncmp(enc->plane[i].role, "codes", RAD_ENC_STR) == 0) { codes = i; break; }
    if (codes < 0 || enc->plane[codes].dtype != RAD_U4 ||
        enc->plane[codes].block[0] != 1 || enc->plane[codes].block[1] != 1)
        return RAD_E_DTYPE;
    for (int i = 0; i < 3; ++i) {
        const RadEncPlane* q = &enc->plane[sel[i]];
        if (q->kind != RAD_PLANE_TILED) return RAD_E_DTYPE;
        if (i < 2 && (q->block[0] != 1 || q->block[1] != group)) return RAD_E_DTYPE;
        if (i == 0 && q->dtype != RAD_F16) return RAD_E_DTYPE;
        if (i == 1 && q->dtype != RAD_U8) return RAD_E_DTYPE;
        if (i == 2 && q->dtype != RAD_U32) return RAD_E_DTYPE;
    }
    /* The hi plane's extent: one word a (tile, 64-wide k block, lane). */
    const long long N = r3d_pgeti_or(p, n_p, "N", 0), K = r3d_pgeti_or(p, n_p, "K", 0);
    if (N > 0 && K > 0) {
        const RadTensor* h = &pl[sel[2]];
        if (h->rank < 2 || h->shape[0] < N / 16 || h->shape[1] < K / 2)
            return RAD_E_SHAPE;
    }
    return rad_planes_layout(enc, sel, pl, n, out);
}

/* The writer for the joined operand: the planes copied, the header and entries filled. */
extern "C" int r3d_relayout_gemm_nt_q_w5a16(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, void* dst,
                                            int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != Q_BSCALE) return RAD_E_UNSUPPORTED;
    return rad_planes_relayout(enc, sel, pl, n, dst, dst_bytes);
}
