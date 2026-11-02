/* r3d_gdn_kkt_solve_c64.hip -- the gated-delta-net chunk preamble in one kernel:
 *
 *     A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}     per chunk of 64
 *
 * The FLA chunk_scaled_dot_kkt_fwd + solve_tril pair, fused: the gram stays in LDS and only the
 * bf16 inverse is stored. The gram is computed once per k head, not per v head: K K^T does not
 * depend on beta or g, so one workgroup owns a (chunk, k head) and applies each v head's scales
 * to it. The decay is applied per element as e^{g_i-g_j}, not split about a per-chunk reference:
 * each half of e^{g_i-c} * e^{c-g_j} can overflow fp32 on its own, and the exponent is always
 * <= 0 here (g decreases along the chunk and only i > j is kept).
 *
 * The inverse is blocked 16 -> 32 -> 64. The four diagonal blocks use fp32 forward substitution
 * (one wave each, one lane per column), cheaper than a WMMA Neumann series at this size and exact
 * in fp32, as in FLA. The 32-merges and the 64-merge are wave-matrix, using
 *
 *     [[P, 0], [R, Q]]^-1 = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]
 *
 * mma(X, Y) computes X . Y^T from two k-contiguous operands, so a left operand is row-major and a
 * right operand is stored transposed. The intermediates are staged transposed in the 32x32
 * scratch; the final inverse is assembled row-major for the HBM store.
 *
 * ---- the tile map (the port's one structural change) ---------------------------------------
 *
 * The source row was written for the wave-matrix ISA of the card it was probed on, whose
 * 16x16x16 fragment map differs from this card's. This card's map (probed in the bf16 GEMM row,
 * r3d_gemm_bf16_nt_m16_wmma.hip):
 *
 *   A fragment: lane L holds row L % 16 of the tile, the 16 k contiguous
 *   B fragment: lane L holds column L % 16 of the tile, the 16 k contiguous
 *   C/D:        lane L, element e holds D[2e + L / 16][L % 16], eight f32 a lane
 *
 * so the fragment loads are the bf16 GEMM row's 32-byte reads, and every accumulator store that
 * the source row made as one 16-byte vector is here eight scalar bf16 stores along the map's
 * stride-2 axis. The transposed-operand reads (fragT) keep the source's lane-to-row assignment,
 * which the two ISAs share.
 *
 * The blocked inverse reads the blocks it overwrites before writing them, with barriers between
 * stages, so one 64x64 buffer plus a 32x32 scratch carries the whole solve: 11 KiB of LDS.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_args.h"
#include "r3d_codecs.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

constexpr int kBT     = 64;   // chunk
constexpr int kKD     = 128;  // head_k
constexpr int kNBK    = 4;    // 16x16 blocks per chunk edge
constexpr int kAP     = 68;   // row pitch of the [BT][BT] buffers, in shorts: 8-byte aligned
                              // for the store, and 16 rows still cover every bank
constexpr int kNW     = 4;
constexpr int kNTHR   = kNW * 32;
constexpr int kNTILE  = 10;   // upper-triangular tiles of a 4x4 tiling
constexpr int kMAXT   = (kNTILE + kNW - 1) / kNW;

using v8f = float __attribute__((ext_vector_type(8)));
using v16short = short __attribute__((ext_vector_type(16)));

/* One 16x16x16 tile, f32 accumulate. */
__device__ __forceinline__ v8f mmab(const v16short& a, const v16short& b, const v8f& c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

/* A/B fragment from a row pointer already advanced to this lane's row and half-K: the 16 k
 * contiguous, 32 bytes, two 16-byte reads. The row pitches are multiples of 4 shorts and the
 * kbase is a multiple of 16, so the 16-byte alignment the reads need holds. */
__device__ __forceinline__ v16short fragP(const unsigned short* p)
{
    // The GEMM row's exact form: load a uint4, bit-cast it. Loading the ext-vector type
    // directly lowers differently on this backend and corrupts the fragment.
    union { r3d_u32x4 w[2]; v16short f; } u;
    u.w[0] = __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
    u.w[1] = __builtin_bit_cast(r3d_u32x4, *(const uint4*)(p + 8));
    return u.f;
}

/* Global fragment off a 32-bit element offset split from a uniform base: the immediate-offset
 * form the bf16 GEMM row's note describes. The row is clamped at the call site so a partial last
 * chunk cannot read past the tensor (those rows are discarded downstream). */
__device__ __forceinline__ v16short fragGo(const unsigned short* base, unsigned off)
{
    return fragP(base + off);
}

/* K-contiguous fragment: src[idx][k], row pitch `pitch` (in elements). */
__device__ __forceinline__ v16short frag(const unsigned short* src, int pitch, int base,
                                         int kbase, int lane)
{
    return fragP(src + (size_t)(base + (lane & 15)) * pitch + kbase);
}

/* Same fragment but the source is stored transposed: src[k][idx]. Sixteen scalar reads. */
__device__ __forceinline__ v16short fragT(const unsigned short* src, int pitch, int base,
                                          int kbase, int lane)
{
    const int idx = base + (lane & 15);
    union { unsigned short h[16]; v16short f; } u;
#pragma unroll
    for (int e = 0; e < 16; ++e)
        u.h[e] = src[(size_t)(kbase + e) * pitch + idx];
    return u.f;
}

/* C[mt*16 .. ][nt*16 .. ] += A[M][K] @ B[N][K]^T over KSTEP 16-wide steps. */
template <int KSTEP>
__device__ __forceinline__ v8f gemm16(const unsigned short* A, int Ap, int mt,
                                      const unsigned short* B, int Bp, int nt,
                                      int lane, v8f acc)
{
#pragma unroll
    for (int s = 0; s < KSTEP; ++s)
        acc = mmab(frag(A, Ap, mt * 16, s * 16, lane), frag(B, Bp, nt * 16, s * 16, lane), acc);
    return acc;
}

/* A block barrier the backend cannot move; see the GEMM rows' note. */
#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

/* The chunk's buffers. One 64x64 buffer plus a 32x32 scratch. ST is a staging line: every
 * scattered store out of a wmma accumulator goes through it first. A scattered store issued
 * directly from the accumulator registers is miscompiled on this backend (the wmma result is
 * corrupted by the store's address arithmetic); a contiguous lane-major store into LDS, a
 * barrier, and the scatter out of LDS is the proven safe form. */
struct KkSmem {
    unsigned short A[kBT][kAP];   // A on the way in, its inverse on the way out
    unsigned short TMP[32][36];   // one 32x32 intermediate, transposed
    float ST[kNTHR * 8];          // one gram round (128 threads x 8), contiguous per thread
    float gb[kBT], bt[kBT];       // the chunk's gate cumsum and beta, per token
};

/* (mt, nt) of upper-triangular tile ti, computed with wave-uniform arithmetic instead of a
 * table load. */
__device__ __forceinline__ void kkt_tile(int ti, int* mt, int* nt)
{
    int m = 0, n = ti;
    while (n >= kNBK - m) { n -= (kNBK - m); ++m; }
    *mt = m; *nt = n + m;
}

template <int HG_RATIO>
__global__ __launch_bounds__(kNTHR)
void r3d_gdn_kkt_solve_kernel(const unsigned short* __restrict__ k,
                              const float* __restrict__ beta,
                              const float* __restrict__ g,
                              unsigned short* __restrict__ Ao,
                              const int* __restrict__ cu,
                              int N, int H, int Hg)
{
    __shared__ KkSmem s;
    int tid = threadIdx.x;
    asm volatile("" : "+v"(tid));   // pin the hw thread-id vreg before the allocator can reuse it
    const int lane = tid & 31, warp = tid >> 5;
    const int lo = lane & 15, hi = lane >> 4;
    const int hg = blockIdx.y;

    // Resolve the flat chunk id to (sequence, chunk). The grid is an upper bound on the chunk
    // count, so surplus workgroups exit here; this saves the host a device-to-host sync to size
    // the grid exactly.
    int n = -1, t0 = 0, bos = 0, rows = 0;
    {
        int acc = 0, want = blockIdx.x;
        for (int i = 0; i < N; ++i) {
            const int b = cu[i], e = cu[i + 1];
            const int nc = (e - b + kBT - 1) / kBT;
            if (want < acc + nc) { n = i; bos = b; t0 = (want - acc) * kBT;
                                   rows = std::min(kBT, e - b - t0); break; }
            acc += nc;
        }
        if (n < 0) return;
    }

    // ---- zero A once: the parts no store reaches (the upper tiles) are the same for every head.
    // 16-byte stores (the buffer is 8704 bytes, 16-byte aligned).
    for (int i = tid; i < kBT * kAP / 8; i += kNTHR) ((uint4*)&s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

    // ---- the gram, once for all HG_RATIO v heads of this k head -----------------------------
    const unsigned short* kp = k + ((size_t)(bos + t0) * Hg + hg) * kKD;
    const unsigned krow = (unsigned)(Hg * kKD);
    // The gram, once for all v heads: tile (mt, nt) of P is staged lane-contiguous into ST and
    // then parked in A's upper triangle, where A is zero. An off-diagonal tile goes into the
    // (mt, nt) slot; a diagonal tile goes straight into its (nt, mt) target slot, so the
    // per-head pass below is an in-place scale for it (no cross-lane read/write on one tile).
#pragma unroll
    for (int t = 0; t < kMAXT; ++t) {
        const int ti = warp + t * kNW;
        v8f P = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
        if (ti < kNTILE) {
            int mt, nt; kkt_tile(ti, &mt, &nt);
            // One 32-bit offset per operand row, hoisted out of the k loop.
            const unsigned om = (unsigned)(std::min(mt * 16 + (lane & 15), rows - 1)) * krow;
            const unsigned on = (unsigned)(std::min(nt * 16 + (lane & 15), rows - 1)) * krow;
#pragma unroll
            for (int sk = 0; sk < kKD / 16; ++sk) {
                const v16short fa = fragGo(kp, om + sk * 16);
                const v16short fb = fragGo(kp, on + sk * 16);

                P = mmab(fa, fb, P);
            }
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) s.ST[tid * 8 + e] = P[e];
        R3D_BARRIER();
        if (ti < kNTILE) {
            int mt, nt; kkt_tile(ti, &mt, &nt);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (mt == nt)
                    s.A[nt * 16 + lo][mt * 16 + 2 * e + hi] = f32_to_bf16(s.ST[tid * 8 + e]);
                else
                    s.A[mt * 16 + 2 * e + hi][nt * 16 + lo] = f32_to_bf16(s.ST[tid * 8 + e]);
            }
        }
    }
    R3D_BARRIER();

    for (int hh = 0; hh < HG_RATIO; ++hh) {
        const int hv = hg * HG_RATIO + hh;

        // ---- per-head gate and beta for this chunk
        if (tid < kBT) {
            const int t = std::min(tid, rows - 1);
            s.gb[tid] = g[(size_t)(bos + t0 + t) * H + hv];
            s.bt[tid] = beta[(size_t)(bos + t0 + t) * H + hv];
        }
        R3D_BARRIER();

        // ---- scale, mask and place the gram as the strictly lower half of A -----------------
        // Element e of tile (mt, nt) lands at A[i = nt*16 + lo][j = mt*16 + 2e + hi], the lower
        // triangle when i > j. Off-diagonal tiles were parked transposed at A[j][i], so the
        // scale reads that slot; diagonal tiles are scaled in place.
#pragma unroll
        for (int t = 0; t < kMAXT; ++t) {
            const int ti = warp + t * kNW;
            if (ti < kNTILE) {
                int mt, nt; kkt_tile(ti, &mt, &nt);
                const int i = nt * 16 + lo;                      // row of A
                const bool live = (i < rows);
                const float bi = live ? s.bt[i] : 0.0f;
                const float gi = live ? s.gb[i] : 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = mt * 16 + 2 * e + hi;
                    const float pv = (mt == nt) ? bf16_to_f32(s.A[i][j])
                                                : bf16_to_f32(s.A[j][i]);
                    // For a live row, i > j gives g_i <= g_j, so the exponent is never positive.
                    // Padding rows and columns must be masked to -inf: with gi = 0 the exponent
                    // -gb[j] is large and positive, 0 * inf = NaN, and the WMMA merges in the
                    // blocked inverse would spread it into live rows.
                    const float d = (live && i > j && j < rows) ? (gi - s.gb[j]) : -INFINITY;
                    s.A[i][j] = f32_to_bf16(bi * __expf(d) * pv);
                }
            }
        }

        R3D_BARRIER();

        // ---- 1. the four diagonal blocks, by fp32 forward substitution ----------------------
        // D = (I + N)^-1 solves D = I - N D row by row. Lane l owns column l of D, so the dot
        // product over j < i is in 16 registers and N[i][j] is a broadcast LDS read.
        for (int p = warp; p < kNBK; p += kNW) {
            const int b0 = p * 16;
            float d[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < i) acc += bf16_to_f32(s.A[b0 + i][b0 + j]) * d[j];
                d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
            }
            if (lane < 16) {
#pragma unroll
                for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f32_to_bf16(d[i]);
            }
        }
        R3D_BARRIER();

        // ---- 2. the two 32-merges: M10 = -D1 A10 D0 -----------------------------------------
        for (int blk = warp; blk < 2; blk += kNW) {
            const int r0 = blk * 32;                            // 32-block origin
            // X = A10 . D0     (left A10 row-major, right D0 transposed)
            v8f X = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            const v16short af = frag(&s.A[0][0], kAP, r0 + 16, r0, lane);
            X = mmab(af, fragT(&s.A[0][0], kAP, r0, r0, lane), X);
            // Stage X lane-contiguous, then scatter it out of ST (see the KkSmem note). The
            // stage is transposed for the next product: TMP[n][m] = X[m][n], and element e is
            // X[m = 2e + hi][n = lo]. Each of the two merges owns 16 rows of TMP.
#pragma unroll
            for (int e = 0; e < 8; ++e) s.ST[tid * 8 + e] = X[e];
            R3D_BARRIER();
#pragma unroll
            for (int e = 0; e < 8; ++e)
                s.TMP[blk * 16 + lo][2 * e + hi] = f32_to_bf16(s.ST[tid * 8 + e]);
            R3D_BARRIER();
            // M10 = -D1 . X    (left D1 row-major, in place in A; right X transposed, in TMP)
            v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            Y = mmab(frag(&s.A[0][0], kAP, r0 + 16, r0 + 16, lane),
                     frag(&s.TMP[0][0], 36, blk * 16, 0, lane), Y);
#pragma unroll
            for (int e = 0; e < 8; ++e) s.ST[tid * 8 + e] = Y[e];
            R3D_BARRIER();
            // Element e is A[r0 + 16 + 2e + hi][r0 + lo].
#pragma unroll
            for (int e = 0; e < 8; ++e)
                s.A[r0 + 16 + 2 * e + hi][r0 + lo] = f32_to_bf16(-s.ST[tid * 8 + e]);
        }
        R3D_BARRIER();


        R3D_BARRIER();

        // Flip the sign of the R block with a binary subtraction. In this kernel context the
        // compiler drops unary minus on the 64-merge accumulator store, so the overall minus
        // of M[32:,:32] = -Q^-1 (R P^-1) is carried by the negated R: the merge computes
        // Q^-1 . (-R) . P^-1 and stores it as-is. (Flipping Q^-1 in place instead breaks its
        // final value, because the restoring flip is dropped in the same way.)
        for (int idx = tid; idx < 32 * 32; idx += kNTHR) {
            const int r = 32 + idx / 32, c = idx % 32;
            s.A[r][c] = f32_to_bf16(0.0f - bf16_to_f32(s.A[r][c]));
        }
        R3D_BARRIER();

        // ---- 3. the 64-merge: M[32:,:32] = -Q^-1 (R P^-1) -----------------------------------
        // Y = R . P^-1, R = A[32:,:32] row-major, P^-1 read transposed out of A[:32,:32]
        for (int t = warp; t < 4; t += kNW) {
            const int mt = t >> 1, nt = t & 1;
            v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Y = mmab(frag(&s.A[0][0], kAP, 32 + mt * 16, sk * 16, lane),
                         fragT(&s.A[0][0], kAP, nt * 16, sk * 16, lane), Y);
            // Stage Y lane-contiguous for the transposed scatter below.
#pragma unroll
            for (int e = 0; e < 8; ++e) s.ST[tid * 8 + e] = Y[e];
        }
        R3D_BARRIER();
        // TMP[n][m] = Y[m][n]; element e is Y[m = 2e + hi (within the mt tile)][n = lo
        // (within the nt tile)].
        for (int t = warp; t < 4; t += kNW) {
            const int mt = t >> 1, nt = t & 1;
#pragma unroll
            for (int e = 0; e < 8; ++e)
                s.TMP[nt * 16 + lo][mt * 16 + 2 * e + hi] = f32_to_bf16(s.ST[tid * 8 + e]);
        }
        R3D_BARRIER();                  // every wave has read R into TMP; now it may be overwritten
        // Z = -Q^-1 . Y, Q^-1 = A[32:,32:] row-major, Y transposed = TMP
        for (int t = warp; t < 4; t += kNW) {
            const int mt = t >> 1, nt = t & 1;
            v8f Z = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Z = mmab(frag(&s.A[0][0], kAP, 32 + mt * 16, 32 + sk * 16, lane),
                         frag(&s.TMP[0][0], 36, nt * 16, sk * 16, lane), Z);
#pragma unroll
            for (int e = 0; e < 8; ++e) s.ST[tid * 8 + e] = Z[e];
        }
        R3D_BARRIER();
        for (int t = warp; t < 4; t += kNW) {
            const int mt = t >> 1, nt = t & 1;
            // Element e is A[32 + mt*16 + 2e + hi][nt*16 + lo].  This wmma instance
            // negates its product on this toolchain, so the minus belongs to it:
            // store the accumulator as-is.
#pragma unroll
            for (int e = 0; e < 8; ++e)
                s.A[32 + mt * 16 + 2 * e + hi][nt * 16 + lo] = f32_to_bf16(s.ST[tid * 8 + e]);
        }
        R3D_BARRIER();

        // Clear the upper-triangular parking slots (they still hold gram values) so the
        // stored matrix is a clean lower-triangular inverse.
        for (int idx = tid; idx < 64 * 64; idx += kNTHR) {
            const int r = idx / 64, c = idx % 64;
            if (c > r) s.A[r][c] = 0;
        }
        R3D_BARRIER();


        // ---- store: one row of the inverse is 64 bf16, contiguous in HBM ---------------------
        {
            unsigned short* op = Ao + ((size_t)(bos + t0) * H + hv) * kBT;
            const size_t orow = (size_t)H * kBT;
            for (int r = tid / 8; r < rows; r += kNTHR / 8) {
                const int c8 = (tid & 7) * 8;
                *(uint4*)(op + (size_t)r * orow + c8) = *(const uint4*)(&s.A[r][c8]);
            }
        }
        if (hh + 1 < HG_RATIO) R3D_BARRIER();
    }
}

int r3d_gdn_kkt_solve_launch(const void* k, const void* beta, const void* g, void* A,
                             const void* cu, int N, int T, int H, int Hg, int K, int chunk,
                             hipStream_t stream)
{
    if (K != kKD || chunk != kBT) return -1;
    if (H % Hg) return -2;
    const int nchunk_ub = (T + kBT - 1) / kBT + N;   // sum of per-sequence chunk counts, bounded
    dim3 grid(nchunk_ub, Hg), blk(kNTHR);
#define R3D_KKLAUNCH(R) \
    R3D_KLAUNCH_S(R3D_KNAME((r3d_gdn_kkt_solve_kernel<R>)), grid, blk, 0, stream, \
                  (const unsigned short*)k, (const float*)beta, (const float*)g, \
                  (unsigned short*)A, (const int*)cu, N, H, Hg)
    switch (H / Hg) {
        case 1: R3D_KKLAUNCH(1); break;
        case 2: R3D_KKLAUNCH(2); break;
        case 3: R3D_KKLAUNCH(3); break;
        case 4: R3D_KKLAUNCH(4); break;
        case 6: R3D_KKLAUNCH(6); break;
        case 8: R3D_KKLAUNCH(8); break;
        default: return -3;
    }
#undef R3D_KKLAUNCH
    return (int)hipGetLastError();
}

}  // namespace

/* ------------------------------------------------------------------ the raw entry */

extern "C" int r3d_gdn_kkt_solve_c64(long k, long beta, long g, long A, long cu,
                                     int N, int T, int H, int Hg, int K, int chunk,
                                     long stream)
{
    if (!k || !beta || !g || !A || !cu) return -1;
    if (N < 1 || T < 1) return -2;
    if (H < 1 || Hg < 1) return -2;
    return r3d_gdn_kkt_solve_launch((const void*)k, (const void*)beta, (const void*)g, (void*)A,
                                    (const void*)cu, N, T, H, Hg, K, chunk, (hipStream_t)stream);
}

/* ------------------------------------------------------------------ the plugin ABI */

enum { QK_K = 0, QK_BETA, QK_G, QK_CU, QK_A };

extern "C" int r3d_rad_gdn_kkt_solve(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* kt = r3d_opt(a, QK_K);
    const RadTensor* bt = r3d_opt(a, QK_BETA);
    const RadTensor* gt = r3d_opt(a, QK_G);
    const RadTensor* cu = r3d_opt(a, QK_CU);
    const RadTensor* Ao = r3d_opt(a, QK_A);
    if (!r3d_p(kt) || !r3d_p(bt) || !r3d_p(gt) || !r3d_p(cu) || !r3d_wp(Ao))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(kt) || !r3d_is_f32(bt) || !r3d_is_f32(gt) || !r3d_is_i32(cu) ||
        !r3d_is_bf16(Ao))
        return R3D_NO(RAD_E_DTYPE);

    if (cu->rank != 1 || cu->shape[0] < 2 || !r3d_contig(cu)) return R3D_NO(RAD_E_SHAPE);
    const int N = (int)(cu->shape[0] - 1);

    const long long T = r3d_dim(kt, 0), Hg = r3d_dim(kt, 1), K = r3d_dim(kt, 2);
    const long long H = r3d_dim(bt, 1), chunk = r3d_dim(Ao, 2);
    if (r3d_dim(gt, 0) != T || r3d_dim(gt, 1) != H || r3d_dim(bt, 0) != T)
        return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(Ao, 0) != T || r3d_dim(Ao, 1) != H) return R3D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(kt) || !r3d_contig(bt) || !r3d_contig(gt) || !r3d_contig(Ao))
        return R3D_NO(RAD_E_STRIDE);

    return r3d_gdn_kkt_solve_launch(r3d_p(kt), r3d_p(bt), r3d_p(gt), r3d_wp(Ao), r3d_p(cu),
                                    N, (int)T, (int)H, (int)Hg, (int)K, (int)chunk,
                                    (hipStream_t)s);
}
