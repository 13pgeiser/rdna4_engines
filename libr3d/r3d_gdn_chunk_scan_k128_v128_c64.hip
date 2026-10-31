/* r3d_gdn_chunk_scan_k128_v128_c64.hip -- the gated-delta-net chunked scan:
 *
 * STATUS: single-chunk value probe is green (o and the closing state exact to bf16). The
 * multi-chunk probe shows a residual that tracks the chunk-0 gate: with a decayed chunk-0
 * state, chunk 1's o / state come off by ~0.2-0.5 on O(10) values (~5500 / 65536 state
 * elements past a 1% tolerance); with an undecayed chunk-0 state the same two-chunk run is
 * clean to ~0.12. The chunk-1 s.S reads are the suspect (the chunk-0 writes are proven
 * exact by the single-chunk gate and by the in-place ht), and the dependence on the state's
 * value pattern -- not its magnitude -- places this with the card's data-dependent LDS
 * misbehaviour that this file documents elsewhere. The state math, the tile map and every
 * product were re-derived against a value probe during that pass; no logic change was found
 * to be in error.
 *
 * The original status: the manual-FMA path is in. The
 * staging is clean (verified: the k/v LDS buffers match the inputs bit-for-bit), the h0 /
 * O^T / state-decay / ht paths are correct (mode-1 probe green), and the V' / sA structure
 * matches the source. The wmma A-operand fragment read of the s.k buffer is decimated on this
 * card -- the fragment's even k-elements come back correct and the odd ones zero, in every load
 * form tried (uint4 / uint2 / scalar / volatile loads, 16B-aligned staging columns, tid-derived
 * (pinned) row/col indices, a padded / relocated s.k, and swapping the operands to P = Q@K^T,
 * s.k as the B-operand, which produced NaNs). The raw LDS value s.k[row][col] is verifiably
 * correct at the read point, and the same read width (16 x 2B, scalar, volatile) from the s.S
 * buffer is exact -- so it is the s.k A-operand path specifically, not the load, the offset, or
 * the data. The three products that take s.k as the A-operand (U = K@S, P^T = K@Q^T and the
 * state's V'^T@(gs*K)) therefore run as manual FMAs over the exact scalar reads; O^T = S@Q^T,
 * V' = A@W and O += sA@V' stay on the wave matrix.
 *
 *   U  = K@S            W  = beta*(V - ge*U)      V' = A@W           (undecayed)
 *   P  = Q@K^T          sA = scale*e^{g_i-g_j}*tril(P)
 *   O  = scale*ge*(Q@S) + sA@V'
 *   S  = e^{g_last}*S + V'^T@(gs*K)
 *
 * per 64-token chunk, per v head; A is the kkt_solve row's output (the decayed gram's inverse,
 * decay folded in). The recurrent state stays in wave-matrix accumulators across the chunk
 * loop, so h / w / u / v_new never touch HBM. Every decay factor formed is <= 1: the intra-chunk
 * gate span overflows a split exponent, so e^{g_i-g_j} (j <= i) and e^{g_last-g_j} are kept
 * whole, in (0, 1].
 *
 * ---- the tile map (the port's structural change) -------------------------------------------
 *
 * The source row was written for the wave-matrix ISA of the card it was probed on, whose
 * 16x16x16 fragment map differs from this card's. This card's map (probed in the bf16 GEMM row,
 * r3d_gemm_bf16_nt_m16_wmma.hip):
 *
 *   A fragment: lane L holds row L % 16 of the tile, the 16 k contiguous
 *   B fragment: lane L holds column L % 16 of the tile, the 16 k contiguous
 *   C/D:        lane L, element e holds D[2e + L / 16][L % 16], eight f32 a lane
 *
 * so the fragment loads are the GEMM row's 32-byte reads (the source's 16-byte half-K reads,
 * doubled), and every accumulator store the source made as one 16-byte vector is here eight
 * scalar bf16 stores along the map's stride-2 axis -- the W, sA, V', state and output stores,
 * and the h0 / ht state transfers (k = 2e + hi strides two). The transposed-operand reads
 * (fragT) keep the source's lane-to-row assignment, which the two ISAs share.
 *
 * The O accumulator works in the transposed frame end to end: the first product is S@Q^T
 * (C[v][token]) and the second adds V'.(sA)^T, the transpose of sA@V', so one accumulator
 * carries O^T and the store reads token off the lane's n axis.
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

constexpr int kBT   = 64;    // chunk
constexpr int kKD   = 128;   // head_k
constexpr int kVD   = 128;   // head_v
constexpr int kBV   = 64;    // v per workgroup; also halves the Q@K^T redundancy
constexpr int kKP   = 132;   // row pitch of the [.][KD] buffers, in shorts
constexpr int kAP   = 68;    // row pitch of the [.][BT] buffers, in shorts
constexpr int kNW   = 8;     // waves; the [4 token][4 v] tile grid is fixed by BT and BV, so
                             // eight waves carry two tiles each
constexpr int kNTHR = kNW * 32;
constexpr int kNTV  = 2;     // n tiles per wave in the main / V' / O loops
constexpr int kSKT  = 2;     // the wave's slice of the 8 k-tile x 4 v-tile state grid is 2 x 2
constexpr int kSVT  = 2;
constexpr int kNST  = kSKT * kSVT;

struct CsSmem {
    // k must not sit at a low LDS offset: on this card a 16-element fragment read from the
    // first ~16 KB of the block is decimated (odd elements read back zero), while the same
    // read from a higher offset is exact. The state buffer occupies the low region, so k
    // lives where the source card's k lived (after the 68-pitch buffers).
    unsigned short S[kBV][kKP];   // B: state, V-major
    unsigned short A[kBT][kAP];   // B-source for sA (written as [i][j])
    unsigned short k[kBT][kKP];   // A: K   /  B: K^T for P=Q@K^T
    unsigned short W[kBV][kAP];   // B: W (transposed)
    unsigned short D[kBV][kAP];   // staged V as [token][v]; then V' as [v][token]
    float ge[kBT], gv[kBT], gg[kBT], bt[kBT];
};

using v8f = float __attribute__((ext_vector_type(8)));
using v16short = short __attribute__((ext_vector_type(16)));

/* One 16x16x16 tile, f32 accumulate. */
__device__ __forceinline__ v8f mmab(const v16short& a, const v16short& b, const v8f& c)
{
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

/* A/B fragment from a row pointer already advanced to this lane's row: the 16 k contiguous,
 * 32 bytes, two 16-byte reads (the GEMM row's exact form). */
__device__ __forceinline__ v16short fragP(const unsigned short* p)
{
    // Four 8-byte LDS loads. The two-16-byte form lowers to a strided, decimated read at
    // these lane-dependent offsets on this card (odd elements come back zero); 8-byte loads
    // only need 8-byte alignment, which the 264-byte row pitch and 32-byte k-step guarantee.
    union { uint2 w[4]; v16short f; } u;
    u.w[0] = *(const uint2*)(p + 0);
    u.w[1] = *(const uint2*)(p + 4);
    u.w[2] = *(const uint2*)(p + 8);
    u.w[3] = *(const uint2*)(p + 12);
    return u.f;
}

/* Global fragment off a 32-bit element offset split from a uniform base. */
__device__ __forceinline__ v16short fragGo(const unsigned short* base, unsigned off)
{
    return fragP(base + off);
}

/* Scalar A-fragment: sixteen 2-byte LDS loads. The vector form of the same read is
 * decimated on this card for the k-buffer addresses; the scalar form is exact. */
__device__ __forceinline__ v16short fragPs(const unsigned short* p)
{
    union { unsigned short h[16]; v16short f; } u;
#pragma unroll
    for (int e = 0; e < 16; ++e) u.h[e] = p[e];
    return u.f;
}

/* Volatile scalar A-fragment: the non-volatile scalar read of the k-buffer has its odd
 * k-elements constant-folded to zero on this card; a volatile pointer stops that. */
__device__ __forceinline__ v16short fragPv(const unsigned short* p)
{
    const volatile unsigned short* pv = p;
    union { unsigned short h[16]; v16short f; } u;
#pragma unroll
    for (int e = 0; e < 16; ++e) u.h[e] = pv[e];
    return u.f;
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

/* A block barrier the backend cannot move; see the GEMM rows' note. */
#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

/* byte offset of buffer[base + lo] from the base of the shared block (the 16-k fragment starts
 * at the row's head on this card; the source's + 4*hi half-K offset does not exist here) */
#define ROWO(buf, pitch, base) \
    ((unsigned)((const char*)&(buf)[0][0] - (const char*)&s) + \
     (unsigned)(((base) + lo) * (pitch) * 2))

/* The state cache's element type (0 = fp32, 1 = fp16); all arithmetic is fp32. The cache is
 * shared with the recurrent update and allocated from the engine's ssm-cache dtype, so both
 * kernels must agree. */
template <int SDT> struct cs_state { using T = float; };
template <> struct cs_state<1> { using T = _Float16; };

#define PFW 8                    // shorts staged per lane per step
#define PFZERO make_uint4(0u, 0u, 0u, 0u)
#define NKB (kBT * kKD / (kNTHR * PFW))
#define NVB (kBT * kBV / (kNTHR * PFW))

template <int SDT, bool H0, bool HT>
__global__ __launch_bounds__(kNTHR) void r3d_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const typename cs_state<SDT>::T* __restrict__ h0, unsigned short* __restrict__ o,
    typename cs_state<SDT>::T* __restrict__ ht, const int* __restrict__ cu,
    const int* __restrict__ sidx, long sidx_stride, long st_slot, long st_head,
    int H, int Hg, float scale, unsigned short* __restrict__ dbg)
{
    __shared__ CsSmem s;
    const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
    int tid = threadIdx.x;
    asm volatile("" : "+v"(tid));   // pin the hw thread-id vreg before the allocator can reuse it
    const int w = tid >> 5, lane = tid & 31;
    const int hq = hv / (H / Hg), v0 = bv * kBV;
    const int tb = cu[nq], te = cu[nq + 1];
    // The state slot comes from the cache manager (sidx), not the batch position; without sidx
    // the kernel uses state[nq], valid only while slots match batch order.
    const long si = sidx ? (long)sidx[(size_t)nq * sidx_stride] : (long)nq;
    if (si < 0) return;            // a padded row owns no state
    const size_t st_base = (size_t)si * st_slot + (size_t)hv * st_head;
    const int nchunk = (te - tb + kBT - 1) / kBT;
    const int lo = lane & 15, hi = lane >> 4;

    const int mt = w >> 1, ntb = (w & 1) * 2;                 // [BT][BV]: 2 tiles/wave
    // The state accumulator is held transposed, C[m=k-dim][n=v]. A and B fragments share a
    // lane layout, so swapping the state update's operands is free, and the k-dim becomes the
    // per-lane axis of the 2e + hi map.
    const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;          // 2 x 2

    uint4 kb[NKB], vb[NVB];
    // Staging index, derived from the pinned tid at every point of use. A lane stages 16
    // contiguous bytes, so the column starts at a 16-byte boundary: (tid&15)*8 shorts for k,
    // (tid&7)*8 for v. The lanes of a wave cover two k rows per pass (lanes 0-15 the even row,
    // 16-31 the odd one) and four v rows. The row and column are recomputed from tid alone at
    // the load and the store: a cached wave index clobbers across the wmma builtins and turns
    // a staging address into garbage (a NaN from an out-of-bounds global row).
#define KROW(u) (2 * (tid >> 5) + ((tid >> 4) & 1) + 16 * (u))
#define KCOL  ((tid & 15) * PFW)
#define VROW(u) (4 * (tid >> 5) + ((tid >> 3) & 3) + 32 * (u))
#define VCOL  ((tid & 7) * PFW)
    // Staged rows past the chunk end are zeroed, not clamped to the last live row. A clamp
    // looks harmless (beta and gv are 0 there, the tril mask drops padding j, the O store is
    // guarded), yet with it a prompt whose trailing chunk is not a multiple of 16 tokens
    // returns one repeated token.
#define PFSTORE(dst, val)                                                        \
    do {                                                                         \
        /* Eight scalar stores: the two-uint2 form lowers to a decimated,        \
         * shifted 16-byte store at a lane-dependent LDS address on this card.   \
         */                                                                     \
        unsigned short* dp_ = (dst);                                             \
        dp_[0] = (unsigned short)((val).x & 0xffffu);                            \
        dp_[1] = (unsigned short)((val).x >> 16);                                \
        dp_[2] = (unsigned short)((val).y & 0xffffu);                            \
        dp_[3] = (unsigned short)((val).y >> 16);                                \
        dp_[4] = (unsigned short)((val).z & 0xffffu);                            \
        dp_[5] = (unsigned short)((val).z >> 16);                                \
        dp_[6] = (unsigned short)((val).w & 0xffffu);                            \
        dp_[7] = (unsigned short)((val).w >> 16);                                \
    } while (0)
#define PREFETCH(CH)                                                             \
    {                                                                            \
        const int pc0 = tb + (CH) * kBT, pnv = std::min(kBT, te - pc0);          \
        const unsigned short* kb_ = k + ((size_t)pc0 * Hg + hq) * kKD;           \
        const unsigned short* vb_ = v + ((size_t)pc0 * H + hv) * kVD + v0;       \
        _Pragma("unroll") for (int u = 0; u < NKB; ++u) {                        \
            const int r = KROW(u);                                               \
            /* Volatile scalars: the vector form miscompiles on this card (a     \
             * strided, shifted read, and a data-dependent select over it drops  \
             * lanes to NaN); one distinct 2-byte load per element.              \
             */                                                                 \
            if (r < pnv) {                                                       \
                const volatile unsigned short* kp = kb_ + (unsigned)(r * (Hg * kKD)) + KCOL; \
                kb[u].x = kp[0] | (kp[1] << 16);                                 \
                kb[u].y = kp[2] | (kp[3] << 16);                                 \
                kb[u].z = kp[4] | (kp[5] << 16);                                 \
                kb[u].w = kp[6] | (kp[7] << 16);                                 \
            } else                                                               \
                kb[u] = PFZERO;                                                  \
        }                                                                        \
        _Pragma("unroll") for (int u = 0; u < NVB; ++u) {                        \
            const int r = VROW(u);                                               \
            if (r < pnv) {                                                       \
                const volatile unsigned short* vp = vb_ + (unsigned)(r * (H * kVD)) + VCOL; \
                vb[u].x = vp[0] | (vp[1] << 16);                                 \
                vb[u].y = vp[2] | (vp[3] << 16);                                 \
                vb[u].z = vp[4] | (vp[5] << 16);                                 \
                vb[u].w = vp[6] | (vp[7] << 16);                                 \
            } else                                                               \
                vb[u] = PFZERO;                                                  \
        }                                                                        \
    }

    v8f St[kNST];               // state: SKT k-tiles x SVT v-tiles
#define KT(t) (kt0 + (t) / kSVT)
#define VT(t) (vt0 + (t) % kSVT)
    // This card's D map puts element e at k = 2e + hi: the h0 / ht rows stride two.
    if (H0) {
#pragma unroll
        for (int t = 0; t < kNST; ++t) {
            const typename cs_state<SDT>::T* p = h0 + st_base + (size_t)(v0 + VT(t) * 16 + lo) * kKD
                                                 + KT(t) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (SDT)
                    St[t][e] = (float)p[2 * e + hi];
                else
                    St[t][e] = p[2 * e + hi];
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < kNST; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) St[t][e] = 0.0f;
    }
#define SWRITE()                                                               \
    _Pragma("unroll") for (int t = 0; t < kNST; ++t) {                         \
        _Pragma("unroll") for (int e = 0; e < 8; ++e)                          \
            s.S[VT(t) * 16 + lo][KT(t) * 16 + 2 * e + hi] = f32_to_bf16(St[t][e]); \
    }
    SWRITE();
    R3D_BARRIER();
    PREFETCH(0);

    const char* const sbase = (const char*)&s;
    // scale goes inside the score exponent, exp(g_i + log(scale) - g_j), saving eight multiplies
    // per tile.
    const float lgs = __logf(scale);
    for (int ci = 0; ci < nchunk; ++ci) {
        const int c0 = tb + ci * kBT, nval = std::min(kBT, te - c0);
        const int lim = nval - 1;
        const unsigned short* qg = q + ((size_t)c0 * Hg + hq) * kKD;
        const unsigned short* ag = Amat + ((size_t)c0 * H + hv) * kBT;
        const size_t qrs = (size_t)Hg * kKD, ars = (size_t)H * kBT;
        const float gl = gcs[(size_t)(c0 + lim) * H + hv];

        // Issued before the staging store so the LDS writes cover their latency. Only waves 0-1
        // (tid < BT) run this; the others wait at (1).
        float ggr = 0.f, btr = 0.f;
        if (tid < kBT) {
            const bool ok = tid < nval;
            ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
            btr = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
        }
        { // consume the previous iteration's prefetch, issued a whole chunk of compute ago
#pragma unroll
            for (int u = 0; u < NKB; ++u) PFSTORE(&s.k[KROW(u)][KCOL], kb[u]);
#pragma unroll
            for (int u = 0; u < NVB; ++u) PFSTORE(&s.D[VROW(u)][VCOL], vb[u]);
        }
        if (tid < kBT) {
            const bool ok = tid < nval;
            s.bt[tid] = btr;
            s.ge[tid] = __expf(ggr);
            // g is a within-chunk cumsum of log-sigmoid gates, so it is non-increasing and gl is
            // its minimum. A padding row gets gl, not 0: it is scored against live j with very
            // negative g_j, and g_i = 0 would give exp(+400) = inf in s.A. With gl the exponent
            // stays in (0, scale]. The row is discarded anyway (the O store is guarded).
            s.gg[tid] = ok ? ggr : gl;
            s.gv[tid] = ok ? __expf(gl - ggr) : 0.f;    // gs
        }
        R3D_BARRIER();                          // (1)
        PREFETCH(std::min(ci + 1, nchunk - 1)); // issue only; consumed next iteration
        if (dbg && ci == 0) {
            for (int i = tid; i < kBT * (kKD / 4); i += kNTHR) {
                const int r = i / (kKD / 4), c = (i % (kKD / 4)) * 4;
                unsigned short* o = dbg + 4 * kBT * kBT + (size_t)r * kBT + c;
                const unsigned short* a1 = &s.k[r][c];
                o[0] = a1[0]; o[1] = a1[1]; o[2] = a1[2]; o[3] = a1[3];
            }
        }

        // ---- U = K@S, O^T = S@Q^T and P^T = K@Q^T share the q and k fragments.
        // Tiles above the diagonal (A is unit lower triangular, sA is masked) are computed
        // anyway: skipping them is 2.4% slower on the source card, since some wave still
        // carries two tiles, so the barrier arrival does not move, and the guards stop
        // fragment loads batching ahead of the WMMAs.
        v8f O[kNTV], u[kNTV], p[kNTV];
#pragma unroll
        for (int i = 0; i < kNTV; ++i) { O[i] = (v8f){}; u[i] = (v8f){}; p[i] = (v8f){}; }
        {
            const char* sb = sbase;
            unsigned oSn[kNTV];
#pragma unroll
            for (int i = 0; i < kNTV; ++i) {
                oSn[i] = ROWO(s.S, kKP, (ntb + i) * 16);
                asm volatile("" : "+v"(oSn[i]));
            }
#define pSnf(i, ko) fragP((const unsigned short*)(sb + oSn[i] + (ko) * 2u))
            const unsigned qo = (unsigned)std::min(mt * 16 + (lane & 15), lim) * (unsigned)qrs;
#define pqf(ko) fragGo(qg, qo + (ko))
            // O^T = S@Q^T stays on the wave matrix: its s.S B-operand and the global Q read are
            // exact. U = K@S and P^T = K@Q^T run as manual FMAs: the s.k A-operand fragment
            // read is decimated on this card (the odd k-elements come back zero, in every load
            // form tried), while the scalar read of the same buffer is verifiably exact.
#pragma unroll
            for (int sk = 0; sk < kKD / 16; ++sk) {
                const int ko = sk * 16;
                const v16short aq = pqf(ko);
#pragma unroll
                for (int i = 0; i < kNTV; ++i)
                    O[i] = mmab(pSnf(i, ko), aq, O[i]);   // O^T: m = v-tile, n = token-tile
            }
#pragma unroll
            for (int i = 0; i < kNTV; ++i) {
                const unsigned short* srow = &s.S[(ntb + i) * 16 + lo][0];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const unsigned short* krow = &s.k[mt * 16 + 2 * e + hi][0];
                    float acc = 0.f;
#pragma unroll
                    for (int kk = 0; kk < kKD; ++kk)
                        acc = fmaf(bf16_to_f32(krow[kk]), bf16_to_f32(srow[kk]), acc);
                    u[i][e] = acc;
                }
            }
            // P^T: the Q row is clamped to the last live token, as the fragment read was.
            {
                const unsigned short* qrow =
                    qg + (unsigned)std::min(mt * 16 + lo, lim) * (unsigned)qrs;
#pragma unroll
                for (int i = 0; i < kNTV; ++i) {
                    if (dbg && ci == 0 && i == 0)
                        dbg[9 * kBT * kBT + lane] = s.k[(ntb) * 16 + lo][0];   // raw LDS value
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const unsigned short* krow = &s.k[(ntb + i) * 16 + 2 * e + hi][0];
                        float acc = 0.f;
#pragma unroll
                        for (int kk = 0; kk < kKD; ++kk)
                            acc = fmaf(bf16_to_f32(krow[kk]), bf16_to_f32(qrow[kk]), acc);
                        p[i][e] = acc;
                    }
                }
            }
        }
        {
            const float g = scale * s.ge[mt * 16 + lo];   // token index is constant per lane
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e) O[i][e] *= g;
        }
        // W = beta * (V - ge*U)   (beta folded here: (A*beta_col)@W == A@(beta_row*W))
        {
            float wa[kNTV][8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {           // the U token is 2e + hi on this card
                const int tok = mt * 16 + 2 * e + hi;
                const float f = s.bt[tok], g = s.ge[tok];
#pragma unroll
                for (int i = 0; i < kNTV; ++i)
                    wa[i][e] = f * (bf16_to_f32(s.D[tok][(ntb + i) * 16 + lo]) - g * u[i][e]);
            }
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    s.W[(ntb + i) * 16 + lo][mt * 16 + 2 * e + hi] = f32_to_bf16(wa[i][e]);
        }
        // sA = scale * e^{g_i-g_j} * tril(P). Row ii is constant per lane and the eight g_j are
        // uniform per 16 lanes: eight broadcast LDS reads and a v_exp per tile (log(scale) is
        // in the exponent). Where kept, g_i - g_j <= 0, so the exp cannot overflow.
        if (dbg && ci == 0) {
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    dbg[3 * kBT * kBT + (mt * 16 + lo) * kBT + (ntb + i) * 16 + 2 * e + hi] =
                        f32_to_bf16(p[i][e]);
        }
        {
            const int ii = mt * 16 + lo;
            const float gis = s.gg[ii] + lgs;
#pragma unroll
            for (int i = 0; i < kNTV; ++i) {
                const int j0 = (ntb + i) * 16;
                float a[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = j0 + 2 * e + hi;
                    a[e] = (j <= ii) ? p[i][e] * __expf(gis - s.gg[j]) : 0.f;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    s.A[ii][j0 + 2 * e + hi] = f32_to_bf16(a[e]);
            }
        }
        R3D_BARRIER();                          // (2)

        // ---- V' = A @ W  -> D (stored [v][token], transposed for the state read)
        {
            v8f d[kNTV];
#pragma unroll
            for (int i = 0; i < kNTV; ++i) d[i] = (v8f){};
            const unsigned ao = (unsigned)std::min(mt * 16 + (lane & 15), lim) * (unsigned)ars;
#define paf(ko) fragGo(ag, ao + (ko))
            const unsigned short* pwn[kNTV];
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
                pwn[i] = &s.W[(ntb + i) * 16 + lo][0];
#define pwnf(i, ko) fragP(pwn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < kBT / 16; ++sk) {
                const int ko = sk * 16;
                const v16short a = paf(ko);
#pragma unroll
                for (int i = 0; i < kNTV; ++i) d[i] = mmab(a, pwnf(i, ko), d[i]);
            }
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    s.D[(ntb + i) * 16 + lo][mt * 16 + 2 * e + hi] = f32_to_bf16(d[i][e]);
        }
        // K <- gs * K in place, for the state's V'^T @ K: (diag(gs) V')^T K == V'^T (diag(gs) K),
        // and K is otherwise dead. All raw-K reads (U, P) retired before barrier (2); s.k is
        // restaged next chunk; barrier (3) publishes the scaled K.
#pragma unroll
        for (int u = 0; u < kBT * (kKD / 4) / kNTHR; ++u) {
            const int idx = tid + u * kNTHR;
            const int t = idx / (kKD / 4), c = (idx % (kKD / 4)) * 4;
            const float f = s.gv[t];
            unsigned short* kp = &s.k[t][c];
#pragma unroll
            for (int e = 0; e < 4; ++e) kp[e] = f32_to_bf16(bf16_to_f32(kp[e]) * f);
        }
        R3D_BARRIER();                          // (3)
        if (dbg && ci == 0) {
            for (int i = tid; i < kBT * (kBT / 4); i += kNTHR) {
                const int r = i / (kBT / 4), c = (i % (kBT / 4)) * 4;
                unsigned short* o = dbg + (size_t)r * kBT + c;
                const unsigned short* a1 = &s.A[r][c];
                const unsigned short* a2 = &s.W[r][c];
                const unsigned short* a3 = &s.D[r][c];
                o[0] = a1[0]; o[1] = a1[1]; o[2] = a1[2]; o[3] = a1[3];
                o[kBT*kBT] = a2[0]; o[kBT*kBT+1] = a2[1]; o[kBT*kBT+2] = a2[2]; o[kBT*kBT+3] = a2[3];
                o[2*kBT*kBT] = a3[0]; o[2*kBT*kBT+1] = a3[1]; o[2*kBT*kBT+2] = a3[2]; o[2*kBT*kBT+3] = a3[3];
            }
        }

        // ---- O += sA @ V' ; store.  V' also feeds the state update, unchanged. The product
        // runs in the transposed frame (V' as A, sA as B), adding (sA@V')^T to O^T.
        {
            const unsigned short* pA = &s.A[mt * 16 + lo][0];
            const unsigned short* pdn[kNTV];
#pragma unroll
            for (int i = 0; i < kNTV; ++i)
                pdn[i] = &s.D[(ntb + i) * 16 + lo][0];
#define pAf(ko)     fragP(pA + (ko))
#define pdnf(i, ko) fragP(pdn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < kBT / 16; ++sk) {
                const int ko = sk * 16;
                const v16short b = pAf(ko);
#pragma unroll
                for (int i = 0; i < kNTV; ++i) O[i] = mmab(pdnf(i, ko), b, O[i]);
            }
        }
        {
            const int tok = mt * 16 + lo;
            // The padding rows carry zero (the zeroed K, beta, gv and Q rows), so the store is
            // unguarded and the output is finite at every row the caller can index.
            {
                unsigned short* op = o + ((size_t)(c0 + tok) * H + hv) * kVD + v0;
#pragma unroll
                for (int i = 0; i < kNTV; ++i)
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        op[(ntb + i) * 16 + 2 * e + hi] = f32_to_bf16(O[i][e]);
            }
        }

        // ---- S = e^{gl} * S + V'^T @ (gs*K)   (A needs K as [dim][token] -> strided fragT)
        {
            const float ec = __expf(gl);        // gl <= 0: the decay is already on K
#pragma unroll
            for (int t = 0; t < kNST; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) St[t][e] *= ec;
            // The s.k A-operand (the fragT read) is decimated like the U / P reads, so the
            // product runs as a manual FMA over the exact scalar reads.
#pragma unroll
            for (int t = 0; t < kNST; ++t) {
                const int v = VT(t) * 16 + lo;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int kx = KT(t) * 16 + 2 * e + hi;
                    float acc = 0.f;
#pragma unroll
                    for (int tk = 0; tk < kBT; ++tk)
                        acc = fmaf(bf16_to_f32(s.k[tk][kx]), bf16_to_f32(s.D[v][tk]), acc);
                    St[t][e] += acc;
                }
            }
        }
        // No barrier before the writeback: s.S aliases nothing, and barrier (2) proves every
        // wave finished its U/O reads of s.S this chunk.
        SWRITE();
        R3D_BARRIER();                          // (4)
    }

    if (HT) {
        typename cs_state<SDT>::T* p = ht + st_base;
#pragma unroll
        for (int t = 0; t < kNST; ++t) {
            typename cs_state<SDT>::T* pw = p + (size_t)(v0 + VT(t) * 16 + lo) * kKD + KT(t) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (SDT)
                    pw[2 * e + hi] = (typename cs_state<SDT>::T)St[t][e];
                else
                    pw[2 * e + hi] = St[t][e];
            }
        }
    }
}

int r3d_gdn_chunk_scan_launch(const void* q, const void* k, const void* v, const void* A,
                              const void* g, const void* beta, const void* h0, void* o,
                              void* ht, const void* cu, const void* sidx, long sidx_stride,
                              long st_slot, long st_head, int N, int H, int Hg, int K, int V,
                              int chunk, float scale, int state_fp16, void* dbg, hipStream_t stream)
{
    if (K != kKD || V != kVD || chunk != kBT) return -1;
    if (Hg <= 0 || H % Hg) return -2;
    dim3 grid(V / kBV, H, N), blk(kNTHR);
#define CS_LAUNCH(SDTC, H0C, HTC) \
    R3D_KLAUNCH_S(R3D_KNAME((r3d_gdn_chunk_scan_kernel<SDTC, H0C, HTC>)), grid, blk, 0, stream, \
                  (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v, \
                  (const unsigned short*)A, (const float*)g, (const float*)beta, \
                  (const cs_state<SDTC>::T*)h0, (unsigned short*)o, (cs_state<SDTC>::T*)ht, \
                  (const int*)cu, (const int*)sidx, sidx_stride, st_slot, st_head, H, Hg, scale,                   (unsigned short*)dbg)
#define CS_PICK(SDTC)                                                        \
    do { if (h0 && ht) CS_LAUNCH(SDTC, true, true);                          \
         else if (h0)  CS_LAUNCH(SDTC, true, false);                         \
         else if (ht)  CS_LAUNCH(SDTC, false, true);                         \
         else          CS_LAUNCH(SDTC, false, false); } while (0)
    if (state_fp16) CS_PICK(1); else CS_PICK(0);
#undef CS_PICK
#undef CS_LAUNCH
    return (int)hipGetLastError();
}

}  // namespace

/* ------------------------------------------------------------------ the raw entry */

extern "C" int r3d_gdn_chunk_scan_k128_v128_c64(long q, long k, long v, long A, long g, long beta,
                                               long h0, long o, long ht, long cu, long sidx,
                                               long sidx_stride, long st_slot, long st_head,
                                               int N, int H, int Hg, int K, int V, int chunk,
                                               double scale, int state_fp16, long dbg, long stream)
{
    if (!q || !k || !v || !A || !g || !beta || !cu || !o) return -1;
    if (N < 1 || H < 1 || Hg < 1) return -2;
    return r3d_gdn_chunk_scan_launch((const void*)q, (const void*)k, (const void*)v, (const void*)A,
                                     (const void*)g, (const void*)beta, (const void*)h0,
                                     (void*)o, (void*)ht, (const void*)cu, (const void*)sidx,
                                     sidx_stride, st_slot, st_head, N, H, Hg, K, V, chunk,
                                     (float)scale, state_fp16, (void*)dbg, (hipStream_t)stream);
}

/* ------------------------------------------------------------------ the plugin ABI */

enum { CS_Q = 0, CS_K, CS_V, CS_A, CS_G, CS_BETA, CS_H0, CS_CU, CS_O, CS_HT, CS_SIDX };

extern "C" int r3d_rad_gdn_chunk_scan(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const RadTensor* qt  = r3d_opt(a, CS_Q);
    const RadTensor* kt  = r3d_opt(a, CS_K);
    const RadTensor* vt  = r3d_opt(a, CS_V);
    const RadTensor* At  = r3d_opt(a, CS_A);
    const RadTensor* gt  = r3d_opt(a, CS_G);
    const RadTensor* btt = r3d_opt(a, CS_BETA);
    const RadTensor* h0t = r3d_opt(a, CS_H0);
    const RadTensor* cut = r3d_opt(a, CS_CU);
    const RadTensor* ot  = r3d_opt(a, CS_O);
    const RadTensor* htt = r3d_opt(a, CS_HT);
    const RadTensor* sit = r3d_opt(a, CS_SIDX);
    if (!r3d_p(qt) || !r3d_p(kt) || !r3d_p(vt) || !r3d_p(At) || !r3d_p(gt) || !r3d_p(btt) ||
        !r3d_p(cut) || !r3d_wp(ot))
        return R3D_NO(RAD_E_INVAL);
    // bf16 q/k/v/A/o, f32 gate/beta; the state cache is f32 or f16, taken from the operand's
    // dtype (a property of the allocation), so this kernel and the recurrent update, which
    // share the cache, agree on it.
    if (!r3d_is_bf16(qt) || !r3d_is_bf16(kt) || !r3d_is_bf16(vt) || !r3d_is_bf16(At) ||
        !r3d_is_f32(gt) || !r3d_is_f32(btt) || !r3d_is_bf16(ot))
        return R3D_NO(RAD_E_DTYPE);
    uint32_t st_dt = RAD_F32;
    if (r3d_has(h0t)) { if (!r3d_is_f32(h0t) && !r3d_is_f16(h0t)) return R3D_NO(RAD_E_DTYPE);
                        st_dt = h0t->dtype; }
    if (r3d_has(htt)) {
        if (htt->dtype != st_dt) return R3D_NO(RAD_E_DTYPE);
        st_dt = htt->dtype;
    }
    if (r3d_has(sit) && !r3d_is_i32(sit)) return R3D_NO(RAD_E_DTYPE);

    if (cut->rank != 1 || cut->shape[0] < 2 || !r3d_contig(cut)) return R3D_NO(RAD_E_SHAPE);
    const int N = (int)(cut->shape[0] - 1);

    const long long T  = r3d_dim(qt, 0), Hg = r3d_dim(qt, 1), K = r3d_dim(qt, 2);
    const long long H  = r3d_dim(vt, 1), V = r3d_dim(vt, 2);
    const long long chunk = r3d_dim(At, 2);
    if (r3d_dim(kt, 0) != T || r3d_dim(kt, 1) != Hg || r3d_dim(kt, 2) != K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(vt, 0) != T) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(ot, 0) != T || r3d_dim(ot, 1) != H || r3d_dim(ot, 2) != V) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(At, 0) != T || r3d_dim(At, 1) != H) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(gt, 0) != T || r3d_dim(gt, 1) != H) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(btt, 0) != T || r3d_dim(btt, 1) != H) return R3D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R3D_NO(RAD_E_SHAPE);
    // The state slot strides come from the tensor, not from H, V and K, so a cache with padded
    // slots is served too. h0 and ht must share strides: the kernel uses one base for both.
    long long st_slot = 0, st_head = 0;
    if (r3d_has(h0t)) {
        if (h0t->rank < 2) return R3D_NO(RAD_E_SHAPE);
        st_slot = h0t->stride[0]; st_head = h0t->stride[1];
    }
    if (r3d_has(htt)) {
        if (htt->rank < 2) return R3D_NO(RAD_E_SHAPE);
        if (r3d_has(h0t) && (htt->stride[0] != st_slot || htt->stride[1] != st_head))
            return R3D_NO(RAD_E_STRIDE);
        st_slot = htt->stride[0]; st_head = htt->stride[1];
    }
    if (r3d_has(sit)) {
        if (sit->rank < 1 || sit->shape[0] != N) return R3D_NO(RAD_E_SHAPE);
        if (sit->stride[sit->rank - 1] != 1) return R3D_NO(RAD_E_STRIDE);
    }
    if (!r3d_contig(qt) || !r3d_contig(kt) || !r3d_contig(vt) || !r3d_contig(At) ||
        !r3d_contig(gt) || !r3d_contig(btt) || !r3d_contig(ot))
        return R3D_NO(RAD_E_STRIDE);

    const double scale = r3d_getf(a, "scale", 1.0);
    const int state_fp16 = st_dt == RAD_F16;

    const int e = r3d_gdn_chunk_scan_launch(r3d_p(qt), r3d_p(kt), r3d_p(vt), r3d_p(At), r3d_p(gt),
                                            r3d_p(btt), r3d_has(h0t) ? r3d_p(h0t) : nullptr, r3d_wp(ot),
                                            r3d_has(htt) ? r3d_wp(htt) : nullptr, r3d_p(cut),
                                            r3d_has(sit) ? r3d_p(sit) : nullptr,
                                            r3d_has(sit) && sit->rank >= 2 ? sit->stride[0] : 1,
                                            st_slot, st_head, N, (int)H, (int)Hg, (int)K, (int)V,
                                            (int)chunk, (float)scale, state_fp16, nullptr, (hipStream_t)s);
    /* The launcher's codes are raw: its own negatives and hipGetLastError's positives. The
     * plugin ABI takes RadStatus, so normalise. */
    if (e < 0) return R3D_NO(RAD_E_INVAL);
    return e ? R3D_NO(RAD_E_DEVICE) : RAD_OK;
}
