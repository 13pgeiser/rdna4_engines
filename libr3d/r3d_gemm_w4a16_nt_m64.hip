/* r3d_gemm_w4a16_nt_m64.hip -- the 4-bit-weight, f16-activation GEMM, M up to 64.
 *
 *   y[M,N] = a[M,K] @ dequant(wq)[N,K]^T, f32 accumulate, narrowed once by the store.
 *
 * The band is speculative verify and the small prefill chunks, where M is 8 to 64 and the
 * weight is still the cost: a wave owns NPW column tiles by MT row tiles and streams the
 * weight once. This is the RDNA3 form of the row's RDNA4 sibling. The instruction is the
 * same 16x16x16 f16 tile the card's own f16 row runs, but the tile's fragment map is not
 * the RDNA4 one, and the port is what the map implies, not a transcription of the other
 * card's kernel:
 *
 *   A fragment:  lane L holds row L % 16, the tile's sixteen k, replicated across L / 16
 *   B fragment:  lane L holds column L % 16, the tile's sixteen k, replicated across L / 16
 *   C / D:       lane L, VGPR g, holds D[g * 2 + L / 16][L % 16]
 *
 * (the map the f16 m16 row states, probed green on this card). RDNA4's tile scatters eight
 * k a lane "two runs of four" across the half-wave; this one is k-contiguous. The
 * consequence is that the tile's natural load is a row's own bytes, and the weight is read
 * in the container's stored order -- no permutation, no relayout, the library's own rule:
 * a lane's step load is its column's eight codes, a 64-bit read eight bytes down the row a
 * step, and a row's 128-byte cache line serves sixteen of its steps, so the line is fully
 * consumed before it goes.
 *
 * The grid is the asymmetric one: unsigned 4-bit codes, an f16 scale and an integer zero
 * per 128 of K, an f16 activation. It is the form the quantisers write (compressed-tensors'
 * W4A16_ASYM and AutoRound's) and the one the reference library serves, which is what the
 * selftest compares against. The symmetric grid (two's-complement nibbles, the zero fixed at
 * 8) is the same kernel with one XOR per dword and a fixed zero in the packer.
 *
 * ---- the dequant --------------------------------------------------------------------------
 *
 * The codes run the f16 tile on an offset: `0x6400 | n` is exactly the f16 for 1024 + n
 * (1024 = 2^10, n < 16 sits in the f16's low mantissa bits), so a nibble becomes a tile
 * element with one or, and the zero point is added as a packed f16 constant, -(1024 + zero).
 * The subtraction is exact (integers below 2048 in f16's mantissa), and the group scale is
 * applied to the f32 accumulator at the group's end. The scale is NOT folded into the
 * fragment: fma(1024 + n, s, -(1024 + z) * s) cancels ~1024*s against a result of ~4*s, an
 * error of ~12%.
 *
 * A is f16 and is staged through LDS one 16-row tile at a time: the block copies the tile's
 * 16 x (group/8) 16-byte units (the m16 row's pitch, 128 + 8 f16), a barrier, and the
 * fragment reads are two 16-byte LDS loads a lane a step. The stage loop is a count loop
 * with a per-iteration guard, not a strided `tid; <; += blockDim` -- this backend
 * wave-specialises the strided form at small blocks and the upper wave's iterations vanish
 * (probed: the wave-1 rows came back NaN, three loop forms, byte-identical). C is bf16,
 * narrowed once at the store.
 *
 * ---- split-K and the reduction -------------------------------------------------------------
 *
 * `sk` splits K into contiguous slabs of whole 128-wide groups, one per *block*: grid.y is
 * (m-tile group, k-split), the block's columns and row tiles come from its own y index, and
 * the partial tiles land in a global scratch plane (a grow-only device pool, no per-call
 * sync) that a separate reduce kernel sums and narrows. The block-level form is what the
 * wave-matrix m16 row runs; a wave-level split on a block-cooperative A stage is a race in
 * one LDS tile (probed: the split wave's row-0 fragment held its sibling's k slab). The
 * slab sums bracket the products differently than one chain does; the difference is in the
 * last bits and is the split's, not a bug in it.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * NPW*MT accumulators of eight f32, one A fragment of sixteen f16, one B fragment: at the
 * 128-VGPR ceiling of a 1024-thread workgroup the backend cannot carry four column tiles a
 * wave with more than one row tile (probed: the wmma expansion's A-fragment loads collapse
 * and half the tile's rows come back NaN), so the launcher declines `npw == 4 && mb > 1`
 * and the tuning path selects around it. `wv` and `sk` are bounded at 32 each: the block
 * is WV*32 threads and the split is grid, not workgroup.
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
#include <cstdint>
#include <cstring>

namespace {

constexpr int kMaxM    = 64;    // the band's upper bound
constexpr int kGroup   = 128;   // K per (scale, zero)
constexpr int kStep    = 16;    // k a wmma step (a B fragment) covers
constexpr int kWave32  = 32;
constexpr int kAPitch  = 68;    // dwords per staged A row (128 + 8 f16, the m16 pitch)


/* One dword of two f16 into the fragment's two halves, low first: the packed order the
 * builtin's VGPR map wants, a bit pattern not a value (the f16 m16 row's R3D_F16). */
#define R3D_W4_F16(w) __builtin_bit_cast(_Float16, (uint16_t)((w) & 0xffffu)), \
                      __builtin_bit_cast(_Float16, (uint16_t)((w) >> 16))

/* Two dwords of sixteen 4-bit codes (k 0..7 in w0, k 8..15 in w1) -> one 16x16x16 f16 B
 * fragment, minus the zero point. Element e is the code for k = e, the k-contiguous map the
 * card's tile wants.
 *
 * The value is code - zero, computed as an integer and converted once per element: the range
 * is [-255, 15], exact in f16. The offset form (1024 + n) + (-(1024 + zero)) looks cheaper
 * but is not this card: built through packed half pairs it loses registers across the
 * unrolled WMMA chain -- probed, the zero point survived in one half of the first pair and
 * the code vanished from the load. Integer math plus a constructor list has no union views
 * and no half arithmetic to alias. */
__device__ __forceinline__ v16half dequant(unsigned w0, unsigned w1, int z)
{
    const int c0  =  (int)(w0        & 0xFu) - z, c1  =  (int)((w0 >> 4)  & 0xFu) - z;
    const int c2  =  (int)((w0 >> 8)  & 0xFu) - z, c3  =  (int)((w0 >> 12) & 0xFu) - z;
    const int c4  =  (int)((w0 >> 16) & 0xFu) - z, c5  =  (int)((w0 >> 20) & 0xFu) - z;
    const int c6  =  (int)((w0 >> 24) & 0xFu) - z, c7  =  (int)((w0 >> 28) & 0xFu) - z;
    const int c8  =  (int)(w1        & 0xFu) - z, c9  =  (int)((w1 >> 4)  & 0xFu) - z;
    const int c10 =  (int)((w1 >> 8)  & 0xFu) - z, c11 =  (int)((w1 >> 12) & 0xFu) - z;
    const int c12 =  (int)((w1 >> 16) & 0xFu) - z, c13 =  (int)((w1 >> 20) & 0xFu) - z;
    const int c14 =  (int)((w1 >> 24) & 0xFu) - z, c15 =  (int)((w1 >> 28) & 0xFu) - z;
    return v16half{ __float2half((float)c0),  __float2half((float)c1),
                    __float2half((float)c2),  __float2half((float)c3),
                    __float2half((float)c4),  __float2half((float)c5),
                    __float2half((float)c6),  __float2half((float)c7),
                    __float2half((float)c8),  __float2half((float)c9),
                    __float2half((float)c10), __float2half((float)c11),
                    __float2half((float)c12), __float2half((float)c13),
                    __float2half((float)c14), __float2half((float)c15) };
}

/* A block barrier the backend cannot move. The bare __syncthreads() is not enough on this
 * toolchain: in unrolled straight-line regions it reorders LDS traffic across the barrier
 * (probed: the stage's stores rotated to after the fragment reads; the reduction's fill
 * crossed its barrier and the MB > 1 and split-K cases read stale slots). The empty
 * memory-clobbered asm is an opaque side effect no pass moves memory ops across. */
#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

/* y[M,N] = a[M,K] @ dequant(wq)[N,K]^T. MT row tiles a block, NPW column tiles a wave, SK
 * split-K slabs a wave group; the slab partials reduce in the block's dynamic LDS.
 *
 * Wq is the codes in the container's order, [N][K/2] bytes, row n's k contiguous. Bsz is
 * the joined (scale, zero) operand, rad.planes: off_scale and off_zero are the byte
 * offsets of the scale plane, [N][K/128] f16, and the zero plane, [N][K/128] u8. */
/* minBlocks 1: the bare 1024 bound caps the register file at ~16 VGPRs a thread (32 waves a
 * CUA), which spills every accumulator to the stack and the spill traffic corrupts the LDS
 * fill (probed: the split-K plane came back half-filled and the MB > 1 tiles mispaired). */
template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024, 1) void r3d_w4a16_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wq,
    const unsigned char* __restrict__ Bsz, size_t off_scale, size_t off_zero,
    __hip_bfloat16* __restrict__ C, float* __restrict__ Scr,
    int M, int K, int N, int a_ld, int y_ld, int WV, int SK, int MTb)
{
    /* Staged activation, the f16 m16 row's formulation, one 16-row tile: the row tiles run
     * as an outer loop over this single stage (the 16*MT stage miscompiles, probed).
     * Function scope: a block-scoped __shared__ here makes the backend's LDS alias analysis
     * drop most of the stage's stores (probed: the fragment reads came back stale and
     * row-independent). Pitch 136 f16 (68 dwords) = 128 + 8, the m16 pitch. */
    __shared__ uint32_t s_a[16 * kAPitch];

    /* The split is a BLOCK split, the f16 m16 row's formulation: blockIdx.y = (m tile,
     * k split). A wave-level split cannot share the block-cooperative A stage -- two split
     * waves staging their own slabs into the one LDS tile race and each wave's fragments
     * come back row-mixed (probed: a split wave's row-0 fragment held the other split's
     * k slab). */
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
        /* Weight: the lane's column, eight bytes a step, eight bytes down the row a step.
         * The half-wave above replicates the same column. */
        const int ntlast = N / 16 - 1;
        const unsigned char* pw[NPW];
        const unsigned short* ps[NPW];
        const unsigned char* pz[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
            const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
            const size_t row = (size_t)t * 16 + (lane & 15);
            pw[j] = Wq + row * (K / 2) + kbase / 2;
            ps[j] = (const unsigned short*)(Bsz + off_scale) + row * G;
            pz[j] = (Bsz + off_zero) + row * G;
        }

        /* Activation staged in LDS, the f16 m16 row's formulation. Streaming the fragments
         * straight from global miscompiles on this toolchain: the unrolled chain's per-step
         * loads fold the runtime row stride to the tile's width, so every lane reads the
         * first rows and the products come out row-mixed (probed flat across all rows).
         * Through LDS the fragment address is a constexpr pitch -- no runtime stride in the
         * read the wmma chain consumes.
         *
         * The stage is ONE 16-row tile and the row tiles run as an outer loop: a
         * 16*MT-row stage with the tile index in the fragment read is a miscompile trap
         * (probed: the MT > 1 fragments came back e-row mixed, half the tile NaN). One
         * tile at a time is the shape the MT == 1 path proves, and it costs the W codes a
         * re-read per tile, which L2 serves. */
        const int spg = kGroup / kStep;      // steps per group
        const int spg_rt = (K % kGroup) + kGroup / kStep;
        const int ngroups = steps / spg;

        for (int i = 0; i < MT; i++) {
            const int mrow_base = (mt0 + i) * 16;
            for (int g = 0; g < ngroups; g++) {
                /* Stage 16 rows x 128 f16 of the group's slab, eight f16 a unit. A row past
                 * M re-reads the last real row, whose products the store drops.
                 *
                 * The unit count is carried as a runtime value the way the f16 m16 row does:
                 * with a compile-time divisor the backend's i/16 strength reduction drops
                 * the pitch multiply from the 64-bit LDS offset (probed: the stage stored
                 * every unit at dword i/16 and the fragments came back stale). K % kGroup
                 * is 0 by the launcher's contract, so this is 16. */
                const int kslab = kbase + g * kGroup;
                const int nu = (K % kGroup) + kGroup / 8;
                const int nunits = 16 * nu;
                /* The stage runs an explicit wave loop: a strided block loop with more than
                 * one iteration a thread is a trap at small blocks -- the backend
                 * wave-specialises it and the upper wave's iterations vanish behind its
                 * exec mask (probed at 64 threads: the wave-1 rows staged nothing and came
                 * back NaN, identically across three loop forms). With the wave spelled out
                 * the per-wave work is a distinct loop nest and nothing to CSE away; the
                 * cost is that every thread walks every wave's units and the guard keeps
                 * the non-owning threads from storing. */
                const int nwaves = (int)blockDim.x / kWave32;
                const int nstride = (int)blockDim.x;
                const int niter = (nunits + nstride - 1) / nstride;
#pragma unroll 1
                for (int w = 0; w < nwaves; w++) {
                    const int wtid = w * kWave32 + lane;
#pragma unroll 1
                    for (int k = 0; k < niter; k++) {
                        const int ui = wtid + k * nstride;
                        /* No per-wave ownership: a wave whose column group is past N skips
                         * the whole block, so with an ownership guard the tile's tail rows
                         * never stage and the D rows they feed come out corrupt (probed:
                         * N = 64, an eight-wave block, four live column groups, every row
                         * off by the missing A rows). The tile is 16 x 128 f16 and the
                         * extra stores are same-value duplicates, ordered by the barrier. */
                        if (ui < nunits) {
                            const int row = ui / nu, c = (ui % nu) * 8;
                            const int mrow = mrow_base + row;
                            const int mr = mrow < M ? mrow : M - 1;
                            /* The offset in one 32-bit sum, one 64-bit add: a 64-bit
                             * add *chain* (mr*K + kslab + c, each term promoted) is a
                             * trap here -- the kslab term dropped from the lowering
                             * (probed: a split block's A slab read from kbase 0 while
                             * its W codes read from the right base). */
                            const unsigned short* src = A + (size_t)(mr * a_ld + kslab + c);
                            uint32_t* dst = s_a + (size_t)row * kAPitch + c / 2;
                            *(r3d_u32x4*)dst = *(const r3d_u32x4*)src;
                        }
                    }
                }
                R3D_BARRIER();

                float sc[NPW];
                int z[NPW];
                /* The planes are indexed by the absolute group; g is the block's local
                 * group, so a split block must offset by its base (probed: split block 1
                 * dequantised with group 0's scale and zero and the partials summed wrong). */
                const int g_abs = kbase / kGroup + g;
#pragma unroll
                for (int j = 0; j < NPW; j++) {
                    /* The codec's integer conversion, not __half2float: this toolchain's
                     * builtin returns the raw half bits as a float (probed: 0x3C00 came
                     * back as 15360.0), which silently scales every group by the bits. */
                    sc[j] = f16_to_f32((uint16_t)ps[j][(size_t)g_abs]);
                    z[j] = (int)pz[j][(size_t)g_abs];
                }

                /* The first WMMA of a group takes an inline zero for C, so no clear is
                 * needed. The step count is a runtime value, the way the f16 m16 row keeps
                 * its k loop: with a compile-time count the steps unroll to a
                 * straight-line block and the backend rotates the group loop, moving the
                 * stage past the barrier to after the reads. */
                v8float g_acc[NPW];
                for (int s = 0; s < spg_rt; s++) {
                    const int u = g * spg + s;
                    unsigned long long wb[NPW];
#pragma unroll
                    for (int j = 0; j < NPW; j++) {
                        /* Eight bytes a step: sixteen 4-bit codes, eight of them a dword. */
                        wb[j] = NT ? __builtin_nontemporal_load((const unsigned long long*)(pw[j] + 8 * u))
                                   : *(const unsigned long long*)(pw[j] + 8 * u);
                    }
#pragma unroll
                    for (int j = 0; j < NPW; j++) {
                        const v16half bfrag =
                            dequant((unsigned)wb[j], (unsigned)(wb[j] >> 32), z[j]);
                        const uint32_t* ap = s_a + (size_t)(lane & 15) * kAPitch + s * 8;
                        const r3d_u32x4 a0 = *(const r3d_u32x4*)ap;
                        const r3d_u32x4 a1 = *(const r3d_u32x4*)(ap + 4);
                        const v16half afrag = v16half{ R3D_W4_F16(a0.x), R3D_W4_F16(a0.y),
                                                       R3D_W4_F16(a0.z), R3D_W4_F16(a0.w),
                                                       R3D_W4_F16(a1.x), R3D_W4_F16(a1.y),
                                                       R3D_W4_F16(a1.z), R3D_W4_F16(a1.w) };
                        g_acc[j] = s == 0 ? wmma_f16(afrag, bfrag, (v8float)(0.f))
                                          : wmma_f16(afrag, bfrag, g_acc[j]);
                    }
                }
                /* All-runtime loops. This toolchain's unrolled index math is a miscompile
                 * trap (probed across builds: an unrolled fold mispaired e == 0 with the
                 * previous tile's e == 7 register). Runtime trip counts keep the register
                 * handling honest; the loop is tiny. */
                for (int j = 0; j < NPW; j++)
                    for (int e = 0; e < kWave32 / 4; e++)
                        acc[j][i][e] = fmaf(g_acc[j][e], sc[j], acc[j][i][e]);
                R3D_BARRIER();  // the next slab's stage must not race these reads
            }
        }
    }

    /* Epilogue. The D map: acc element e is output row 2e + lane/16, column lane%16.
     *
     * Store straight from the lane's registers, to the output (SK == 1) or to the global
     * split-partial arena (SK > 1, summed by r3d_w4a16_split_reduce). The dynamic-LDS
     * partial plane this used to run through is a miscompile trap on this toolchain,
     * probed across many builds: the fill fused with the read loop, an e * 32 stride
     * collapsed to e * 8, a lane * 8 index lost its lane / 16 bit. Global stores have
     * none of that surface. */
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
                    /* Integer-narrowed store: a bf16-typed store on this toolchain lowers
                     * through a numeric f32->bf16->integer conversion that replaces the
                     * half bits with the rounded value's bits; the codec's bit manipulation
                     * returns the bits straight. */
                    if (m < M && n < N)
                        ((uint16_t*)C)[(size_t)m * y_ld + n] = f32_to_bf16(acc[j][i][e]);
                } else {
                    /* The partial plane is fully sized: clamped rows store too, the reduce
                     * drops them. The pool is sized for ntiles, and the grid's last column
                     * group strides past it when ntiles % ncols != 0. */
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

/* The f16-accumulator arm is shelved, and the reason is the card's, probed on this one:
 * the f16-acc instruction takes a compile-time column selector and computes a 16-wide
 * column window of the 32-column space, and its 16x16 D tile carries only 15 rows --
 * rows 1..15 of the A tile, row 0 absent (the 256 fragment slots hold 15 rows x 16
 * columns and 16 pass-through halves). A 16-row M tile loses a row on every tile
 * boundary, which no staging shift fixes without a 17-row overlap the block-cooperative
 * stage does not have. The row runs the f32 arm; `acc` stays a tuning axis with the f32
 * point, the f16 point joins when a row-overlapping stage is measured to pay.
 */

/* The split-K reduce: y = the SK partial tiles summed, narrowed once. One thread per
 * (column tile, row tile, tile element); the element's (m, n) is the D map inverted. */
__global__ void r3d_w4a16_split_reduce(const float* __restrict__ Scr,
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

}  // namespace

/* y[M,N] = a[M,K] @ dequant(wq)[N,K]^T, f16 a, bf16 y, M <= 64.
 *
 *   a    [M][K]  f16, k contiguous
 *   wq   [N][K/2] the codes in the container's order
 *   bsz  the joined (scale, zero) operand: the f16 scale plane, [N][K/128], then the u8
 *        zero plane, [N][K/128], at off_scale and off_zero
 *   y    [M][N]  bf16, row-major
 *
 * `wv` waves per block (0 = 8), `sk` the K split (0 = 1), `mb` row tiles per block (0 = 1),
 * `npw` column tiles a wave (0 = 1; 1 or 4), `nt` the weight load's lowering, `acc` the
 * accumulator (0 = the f32 arm, the arm served; 1 = the f16 arm, declined, see the
 * shelved-arm note; negative-for-absent = 0). Returns 0, -1 a null pointer, -2 a shape it
 * does not serve, -4 the HIP launch failed. */
/* The split-partial pool: file scope, grown only, so a call never syncs the stream. */
static float* s_scratch_pool = nullptr;
static size_t s_scratch_sz = 0;

extern "C" int r3d_gemm_nt_q_w4a16_m64(long a, long wq, long bsz, size_t off_scale,
                                       size_t off_zero, long y, int M, int N, int K,
                                       int a_ld, int y_ld,
                                       int wv, int sk, int mb, int npw, int nt, int acc,
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
    if (acc < 0) acc = 0;
    if (acc > 1) return -2;
    if (wv > 32 || sk > 32) return -2;
    if (mb < 1 || mb > 4) return -2;
    if (npw != 1 && npw != 4) return -2;
    if (K % (sk * kGroup)) return -2;
    /* npw=4 with mb>1 is the shape the backend cannot lower: at the 128-VGPR ceiling the
     * wmma expansion's A-fragment loads collapse (probed: the 64-thread <2,4> tile came
     * back with half its rows NaN). The tuning path selects around it: npw=4 with mb=1,
     * or npw=1 with the mb it wants. The f16 arm is shelved (its tile drops a row, probed):
     * the axis carries no f16 point, the launcher declines it. */
    if (npw == 4 && mb > 1) return -2;
    if (acc == 1) return -2;

    const int ntiles = N / 16, mtiles = (M + 15) / 16;
    const int ncols = wv * npw;
    const dim3 grid((ntiles + ncols - 1) / ncols, ((mtiles + mb - 1) / mb) * sk),
                block(wv * kWave32);

    /* Split partials live in a grow-only device pool: a per-call hipMalloc/Free would
     * synchronise the stream. */
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

#define R3D_W4L(MT, NP, TT) \
    R3D_KLAUNCH_S(R3D_KNAME((r3d_w4a16_m64_kernel<MT, NP, TT>)), grid, block, 0, \
                  (hipStream_t)stream, (const unsigned short*)a, (const unsigned char*)wq, \
                  (const unsigned char*)bsz, off_scale, off_zero, (__hip_bfloat16*)y, scr, \
                  M, K, N, a_ld, y_ld, wv, sk, mb)
#define R3D_W4L_MB(NP, TT) \
    do { switch (mb) { \
        case 1:  R3D_W4L(1, NP, TT); break; \
        case 2:  R3D_W4L(2, NP, TT); break; \
        case 3:  R3D_W4L(3, NP, TT); break; \
        default: R3D_W4L(4, NP, TT); break; } } while (0)
    if (npw == 1) { if (nt) R3D_W4L_MB(1, true);  else R3D_W4L_MB(1, false); }
    else          { if (nt) R3D_W4L_MB(4, true);  else R3D_W4L_MB(4, false); }
#undef R3D_W4L_MB
#undef R3D_W4L
    if (hipGetLastError() != hipSuccess) return -4;

    if (sk > 1) {
        const int total = ntiles * mb * 256;
        const int rblock = 256;
        const int rgrid = (total + rblock - 1) / rblock;
        R3D_KLAUNCH_S(R3D_KNAME((r3d_w4a16_split_reduce)), dim3(rgrid), rblock, 0,
                      (hipStream_t)stream, scr, (__hip_bfloat16*)y, M, N, y_ld, sk, mb, ntiles);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}



/* ==================================================================== the plugin ABI entry point
 *
 * gemm_nt_q's operand list is [a, a_scale, b, b_scale, y, a_sum, b_ref]; this row takes a,
 * b, b_scale and y, and declines the rest. b is the codes plane in the container's order;
 * b_scale is the (scale, zero) pair in rad.planes' joined form. */
enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y, Q_ASUM, Q_BREF };

extern "C" int r3d_rad_gemm_nt_q_w4a16_m64(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "w4a16_asym") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A  = r3d_opt(a, Q_A);
    const RadTensor* AS = r3d_opt(a, Q_ASCALE);
    const RadTensor* B  = r3d_opt(a, Q_B);
    const RadTensor* BS = r3d_opt(a, Q_BSCALE);
    const RadTensor* Y  = r3d_opt(a, Q_Y);
    const RadTensor* AU = r3d_opt(a, Q_ASUM);
    const RadTensor* BR = r3d_opt(a, Q_BREF);
    if (!r3d_p(A) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    /* Any other operand would read plausible but wrong numbers. */
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
    if (r3d_numel(B) < (N * K) / 2) return R3D_NO(RAD_E_SHAPE);

    /* The joined form, read at launch: the scale and zero planes, and their byte offsets. */
    const RadPlanesEntry* es = nullptr, *ez = nullptr;
    const uint8_t* sc_b = (const uint8_t*)rad_planes_find(r3d_p(BS), "scale", &es);
    const uint8_t* z_b = (const uint8_t*)rad_planes_find(r3d_p(BS), "zero", &ez);
    if (!sc_b || !z_b || es->dtype != RAD_F16 || ez->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);
    if (es->rows < N || ez->rows < N) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N || yrows < M) return R3D_NO(RAD_E_STRIDE);
    /* 16-byte A fragment reads and 8-byte weight reads, both at aligned bases. */
    if (!r3d_aligned(r3d_p(A), 16) || !r3d_aligned(r3d_p(B), 8)) return R3D_NO(RAD_E_ALIGN);

    const int wv = (int)r3d_geti_or(a, "wv", 8);
    const int sk = (int)r3d_geti_or(a, "sk", 1);
    const int mb = (int)r3d_geti_or(a, "mb", 1);
    const int npw = (int)r3d_geti_or(a, "npw", 1);
    const int nt = (int)r3d_geti_or(a, "nt", 1);
    const int acc = (int)r3d_geti_or(a, "acc", 0);

    const int e = r3d_gemm_nt_q_w4a16_m64(r3d_addr(A), r3d_addr(B), r3d_addr(BS),
                                          (size_t)(sc_b - (const uint8_t*)r3d_p(BS)),
                                          (size_t)(z_b - (const uint8_t*)r3d_p(BS)),
                                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                          wv, sk, mb, npw, nt, acc, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE : RAD_E_DEVICE);
}

/* ==================================================================== the (scale, zero) pair
 *
 * The weight's scale and zero are two planes behind the one b_scale operand, and the mapped
 * tier places a weight only when a loaded library can read it in place. This is the scoped
 * exception to the library's "no layout hooks, no repacking" rule, transcribed from the
 * reference's own hook:
 *
 *     if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
 *     the grid's dtype is "w4a16_asym";
 *     the affine encoding: U4 codes block {1,1}, f16 scale block {1, group}, u8 zero block
 *     {1, group}; the transform empty or beginning "givens";
 *     sel names "scale" and "zero", in order;
 *     return rad_planes_layout(enc, sel, planes, n, out);
 *
 * NOTHING HERE MOVES A BYTE: the writer is rad_planes_relayout, and the codes plane is read
 * where it lies -- the kernel's load is the container's order, which is the point.
 */

extern "C" int r3d_layout_gemm_nt_q_w4a16(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out)
{
    if (n < 2 || operand != Q_BSCALE) return RAD_E_UNSUPPORTED;
    const char* d = r3d_pgets(p, n_p, "dtype");
    if (!d || std::strcmp(d, "w4a16_asym") != 0) return RAD_E_DTYPE;
    const long long group = r3d_pgeti_or(p, n_p, "group", kGroup);
    if (group <= 0) return RAD_E_DTYPE;
    if (!enc || !rad_enc_is(enc, "affine")) return RAD_E_DTYPE;
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0)
        return RAD_E_DTYPE;
    if (enc->n_planes < 3) return RAD_E_DTYPE;
    if (n != 2) return RAD_E_DTYPE;
    for (int i = 0; i < 2; ++i) {
        const char* want = i == 0 ? "scale" : "zero";
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return RAD_E_DTYPE;
        if (std::strncmp(enc->plane[sel[i]].role, want, RAD_ENC_STR) != 0) return RAD_E_DTYPE;
    }
    /* The codes plane is the one this operand does not take: U4, block {1,1}. */
    int codes = -1;
    for (int i = 0; i < enc->n_planes; ++i)
        if (std::strncmp(enc->plane[i].role, "codes", RAD_ENC_STR) == 0) { codes = i; break; }
    if (codes < 0 || enc->plane[codes].dtype != RAD_U4 ||
        enc->plane[codes].block[0] != 1 || enc->plane[codes].block[1] != 1)
        return RAD_E_DTYPE;
    for (int i = 0; i < 2; ++i) {
        const RadEncPlane* q = &enc->plane[sel[i]];
        if (q->kind != RAD_PLANE_TILED || q->block[0] != 1 || q->block[1] != group)
            return RAD_E_DTYPE;
        if (i == 0 && q->dtype != RAD_F16) return RAD_E_DTYPE;
        if (i == 1 && q->dtype != RAD_U8) return RAD_E_DTYPE;
    }
    return rad_planes_layout(enc, sel, pl, n, out);
}

/* The writer for the joined operand this hook names: scale and zero as the plan's one
 * rad.planes weight, assembled by rad_planes_relayout. The ABI's contract is that a query
 * answering RAD_OK is a promise the loader keeps on the row's word (rad_abi.h). */
extern "C" int r3d_relayout_gemm_nt_q_w4a16(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, void* dst,
                                            int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != Q_BSCALE) return RAD_E_UNSUPPORTED;
    return rad_planes_relayout(enc, sel, pl, n, dst, dst_bytes);
}
