// Block-scaled FP8 weight, fp8 activation. Two entry points share one K loop with two tile shapes:
//
//   r4d_gemm_fp8a8_nt_m16   M <= 16 (decode and speculative verify). One 16-wide token tile,
//                           split-K over the caller's scratch.
//   r4d_gemm_fp8a8_tiled    any M (prefill). 128x128 tile, eight waves.
//
// Reads block-scaled FP8 checkpoints as shipped (DeepSeek-style, e.g. Qwen3.8-27B-FP8):
// `<linear>.weight` is F8_E4M3 [N][K] and `<linear>.weight_scale_inv` is a BF16 scale per
// 128x128 block, [ceil(N/128)][ceil(K/128)]. No repacking and no scale conversion, so the
// checkpoint never has to be dequantised to bf16 (2x the weight bytes).
//
// Properties of the K loop:
//   1. The bf16 scale decodes as `__uint_as_float(s << 16)` (r4d_fp8_wmma.h).
//   2. Operand roles are a template parameter (`AW`): with the weight on A the kernel computes
//      Y[rows][nb], i.e. C^T of `C[M,N] = A[M,K] @ W[N,K]^T`.
//   3. Split-K uses the caller's scratch (nothing on a launch path allocates), sized by
//      r4d_gemm_fp8a8_nt_m16_scratch(). The summation order is fixed.
//   4. C is bf16, like every GEMM in this library.
//
// Load-bearing: the padded LDS row pitch (r4d_fp8_wmma.h `lds_pitch`), the one-K-tile-ahead
// register prefetch, the two accumulator sets, and the staging assignment (`kTPR`). Reasons are at
// each.
//
// Entry points return a negative code rather than throwing: they are reachable from a kernel
// plugin, and an exception must not cross the .so boundary.

#include "r4d.h"
#include "r4d_fp8_args.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <type_traits>

namespace {

using namespace r4d_fp8;

constexpr uint32_t kWave = 32;

// The scale block on both axes. The checkpoint's scale grid, the activation quantiser's group
// (r4d_quant_act_fp8) and this fold must agree, so it is one constant, and not a build flag: it is
// a property of the checkpoint.
constexpr uint32_t kBlock = 128;

// Prefill tile: kFN 128, 8 waves. A block re-reads its weight row band once per M tile, but that is
// not the limit: the passes are absorbed by the 64 MB MALL (the grid is x-fastest, so one N band's
// M tiles dispatch together), and doubling FM to halve the traffic is slower. The fragment read's
// LDS bank conflicts are fixed by the padded pitch (r4d_fp8_wmma.h `lds_pitch`). The remaining time
// is spread over the per-128-column fold, the global fetch, the B fragment reads, the activation
// scale gather (4-13% each) and the loop itself (barrier, staging stores, loop arithmetic); with all
// per-iteration costs removed the loop reaches about 65% of the WMMA issue rate.
constexpr uint32_t kFN = 128, kFK = 64;   // kFK is BYTES per LDS row
// Narrow-tile K width: one gfx1201 cacheline, 256 B. `kTPR` is FK/32, so this puts eight threads
// on a row, which makes the weight stream coalesce (64 B would put two and cost a third of the
// bandwidth; 128 B is also full speed; 512 B needs 64 VGPRs of prefetch and does not build). The
// prefill tile stays at 64: it is not fetch-bound, and a 256 B row of its 128-row LDS tile would be
// 32 KiB of A alone.
constexpr uint32_t kFKNarrow = 256;
constexpr uint32_t kFWaves = 8;
// Activation bands per raster group of the prefill tile; see the kernel's grouping.
constexpr uint32_t kGroupM = 16;
// Prefill register tile: 1 M tile x 8 N tiles per wave, so the 8 waves cover 8 distinct row tiles.
// It is a template parameter so the narrow instantiation keeps 1x1. Eight accumulator tiles is a
// hard cap: each is two f32 accumulator sets, so MT*NT > 8 spills (2x8 runs at a third the rate).
// 1x8 and 2x4 tie. Measured and rejected:
//   - 1x4 (FN = 64, four resident blocks instead of three): ~20% slower;
//   - 2x4 on a second wave axis (4x2 waves, same 128x128 block, a third fewer fragment reads):
//     a wash, since the fragment reads are not the limit;
//   - a whole 128-column scale block per iteration (compile-time fold position): costs 44 VGPRs
//     and two waves per SIMD, slower; zero-C first WMMA instead of clearing `raw`, explicit
//     sched_group_barrier interleaving, and a manual fragment-read pipeline: neutral or worse;
//   - lds_barrier() instead of __syncthreads(): a few percent slower here (the vector-cache
//     invalidate it skips is worth having in this loop);
//   - a transposed 1xMW wave grid (e.g. 64x256 blocks for a SwiGLU epilogue): about 4x slower even
//     at the same 128x128 tile. The AW=false fold keeps one activation scale per row the wave owns
//     (`as[MT][8]`), so a grid that raises MT pays in registers; the MWx1 grid gives a wave one row
//     tile, and the weight's scale (one scalar per 128 columns) tolerates being spread. Hence the
//     GATED epilogue exists only for the decode tile.
constexpr uint32_t kFMTiles = 1, kFNTiles = 8;

// Blocks the split aims for. A fixed tuning default, not derived from the CU count: `ksplit` sets
// the order K partials are summed in, so deriving it would make output depend on the card.
//
// `gx * mw` is N/16, so the rule depends on N alone. A target of 512 leaves the attention
// projections (4096-5120 rows) at ~60% of their split speed; 256 leaves the 2048-3072-row
// projections 3-12% short. Above 8192 rows splitting costs, and the guard below stops it there.
// 128 holds across the whole M band: the split's cost (an M*N f32 partial plane) grows with M while
// its benefit does not, so a wider band wants a lower target, and 128 is already below every trunk
// grid. Do not re-sweep it.
constexpr uint32_t kSplitTarget = 128u;
/* A lower target at M > 8 would split N = 512 (the attention k/v projections) three ways instead
 * of four and save ~0.1 ms per step in the engine, but a split repartitions K, which changes the
 * emitted text at M > 8 (short prefill chunks). Not worth the output change.
 *
 * Any such rule would also have to fix the scratch query: it is asked once at declare with the
 * band's upper bound and is safe only because `ksplit * M * N` falls with M. A larger split at
 * M <= 8 can exceed the bound's answer (e.g. 3*9 < 4*8), and the launch would silently fall back
 * to ksplit 1. */
// Split cap when the output is thin (`M*N < 8192`), where the reduce could be thinner than the GEMM
// the split widened. The reduce keeps eight loads in flight, which lets the cap sit this high.
constexpr uint32_t kSplitThinCap = 64u;

// ---- fragment-order weight ---------------------------------------------------------------------
//
// A 16x16 fragment from a row-major [N][K] plane is 8 bytes every `w_ld`, so the weight must go
// global -> LDS -> fragment (a staging pass, a barrier per K tile, 36 KiB of LDS). In fragment
// order (r4d_fp8_frag.h, applied by the layout hook at convert) the bytes are contiguous and none
// of that is needed. The order is a container property, so the prefill tile and matvec serve it
// too. The win comes from the request width: 8 bytes per lane (a 256 B request) loses to staging;
// 32 bytes per lane (1 KiB per wave) wins everywhere, by ~13% on gate_up's decode shape.
// ---- the body ----------------------------------------------------------------------------------
//
// fp8 x fp8 -> f32 with no conversion in the loop: both operands are E4M3 bytes, staged as raw
// bytes (no 16-bit WMMA, no in-loop v_cvt).
//
// Both operands carry a scale per 128 along K, constant together over a 128-column block, so they
// are applied once per block: `raw` accumulates 128 columns unscaled and then folds into `out`
// (64 extra VGPRs, no per-element arithmetic in the inner loop).
//
// Both operands are row-major [row][k] byte planes, so staging is a byte copy of 32 consecutive k.
//
// ---- AW: which operand is the weight -----------------------------------------------------------
//
// The scales differ in shape (weight: one per 128 rows; activation: one per token), and the
// accumulator is column-distributed (a lane owns one B row and eight A rows), so a per-B-row scale
// is one scalar per lane and a per-A-row scale eight.
//
//   AW = true    A is the weight. D is [N][M] and the epilogue transposes on the way out (sixteen
//                lanes write addresses `N` apart). Used only at M <= 16, where the output is
//                negligible against the weight read. The small dimension lands on B (one 16-wide
//                N tile) and the grid runs over the weight rows.
//
//   AW = false   A is the activation (prefill). D is [M][N] and the store is coalesced. Correct
//                only when FN divides 128, so the weight's block scale is one scalar per block
//                (static_assert'd below).
// ---- GATED: SwiGLU and the quantiser folded into the epilogue ---------------------------------
//
// Replaces `gated_quant_fp8` re-reading this kernel's [M, 2n] bf16 output and writing [M, n] E4M3
// plus a scale per (token, 128 columns). It fits the narrow tile (and cannot fit the prefill tile;
// see the 8-accumulator cap): a block owns FM = 256 weight rows, 16 per wave. The first half of the
// waves takes gate rows `[c0, c0+128)`, the second half up rows `[n + c0, n + c0 + 128)`, so:
//
//   * a block emits exactly 128 output columns, one scale group, and the amax closes in the block;
//   * waves w and w + MW/2 hold the same 16 output columns for the same tokens in the same lane
//     (the accumulator mapping does not depend on the weight rows), so the gate stash is a
//     straight per-lane copy through LDS with no bank conflict;
//   * `acc_row(lane, t) = (lane >> 4) * 8 + t` is eight consecutive output columns for one token,
//     so the E4M3 store is one 8-byte access;
//   * block count and total WMMA are unchanged.
//
// The stash reuses `s_b`, dead after the K loop (16 KiB at NT = 4 of the 16.5 KiB it holds); only
// the 2 KiB amax plane is new.
//
// Byte-identical to the unfused pair: the accumulator is rounded to bf16 first and silu and the
// multiply run on the round-tripped values. The amax is an fmaxf reduction, exact and order-free.
//
// The narrow path's weight loads carry no nontemporal hint: the activation every block re-reads
// was just written by the previous kernel and is resident, so protecting it does not move a step.

/* I8 is the same kernel at int8 codes (gemm_i8a8_*): i8 operands in the same staging and fragment
 * order, v_wmma_i32_16x16x16_iu8, and a weight scale per row per 128 K (bf16, tile-major
 * [N/16][K/128][16], r4d_fp8_layout.cpp) instead of per 128x128 tile. Only the fold differs: one
 * weight scale per accumulator row on the narrow path, one per lane's column on the prefill tile. */
/* Waves per SIMD for each tile shape (R4D_WAVES): pins the register budget each shape was tuned at
 * (the narrow tile's NT accumulators and K-step unroll are sized against it), so a compiler release
 * cannot change occupancy. Shapes with one or two waves per block are LDS-bound and use that
 * count. */
constexpr uint32_t f8_waves(bool full, bool aw, uint32_t nt, uint32_t mw, uint32_t fk, bool i8) {
    if (!aw) return 6u;
    if (mw == 1u) return nt == 1u ? 8u : nt == 2u ? 4u : nt == 3u ? 3u : 2u;
    if (mw == 2u) return nt == 1u ? 12u : nt == 2u ? 8u : nt == 3u ? 5u : 4u;
    if (!i8) {
        if (fk == 128u)
            return nt == 1u ? 16u : nt == 2u ? (full ? 16u : 12u) : nt == 3u ? 10u
                 : (full && mw >= 8u) ? 10u : 9u;
        return nt == 1u ? 12u : nt == 2u ? 10u : nt == 3u ? 9u : mw == 4u ? 7u : 8u;
    }
    if (fk == 128u)
        return nt == 1u ? 16u : nt == 2u ? (full ? 10u : 12u) : nt == 3u ? (full ? 9u : 10u)
             : full ? 7u : 9u;
    if (nt == 1u) return 12u;
    if (nt == 2u) return full ? 9u : mw == 4u ? 9u : 10u;
    if (nt == 3u) return full ? (mw == 16u ? 8u : 7u) : (mw == 4u ? 8u : 9u);
    return full ? (mw == 4u ? 5u : 6u) : (mw == 4u ? 7u : 8u);
}

template <bool Full, bool AW, uint32_t NT, uint32_t MT, uint32_t MW, uint32_t FK,
          bool GATED = false, bool I8 = false>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(f8_waves(Full, AW, NT, MW, FK, I8))
void gemm_fp8a8_kernel(
    const uint8_t* __restrict__ Ap, const void* __restrict__ Sa,
    const uint8_t* __restrict__ Bp, const void* __restrict__ Sb,
    void* __restrict__ Y, void* __restrict__ C, unsigned int* __restrict__ Ctr,
    uint32_t rows, uint32_t cols, uint32_t nb,
    uint32_t a_ld, uint32_t b_ld, uint32_t sa_ld, uint32_t sb_ld, uint32_t ck,
    uint8_t* __restrict__ Q = nullptr, float* __restrict__ Sc = nullptr,
    uint16_t* __restrict__ Yb = nullptr,
    uint32_t q_ld = 0, uint32_t s_ld = 0, uint32_t y_ld = 0) {
    constexpr uint32_t FN = NT * 16;              // B rows a block
    constexpr uint32_t FM = MW * MT * 16;   // A rows a block

    // The AW=false fold reads the weight's scale as one scalar for the block's B rows, so they must
    // sit in one 128-row scale row: FN divides 128 and n0 is FN-aligned.
    static_assert(AW || kBlock % FN == 0, "AW=false folds the weight scale as one scalar per block");
    // With AW the A-side (weight) scale is folded as one scalar per wave, indexed at
    // `(m0 + wm) / kBlock`, so a wave's MT*16 rows (not the block's FM) must sit in one 128-row
    // scale row. Requiring it of the block would cap MW at 8.
    static_assert(!AW || (MT * 16 <= kBlock && kBlock % (MT * 16) == 0),
                  "a wave's A rows must sit in one scale row");
    // ...and m0 is FM-aligned with FM a multiple of MT*16, so a wave's rows never straddle either.
    static_assert(!AW || FM % (MT * 16) == 0, "FM must be a whole number of wave row tiles");

    // Double-buffered on the prefill tile only: one buffer needs two barriers per K tile (finish
    // reading, then publish) around just 32 WMMA; two buffers leave only the publish barrier (~5%
    // at gate_up's prefill shape, despite 3 resident blocks per WGP instead of 4). The narrow tile
    // keeps one buffer (two would be 74 KiB, over the 64 KiB limit; decode waits on HBM anyway) and
    // the two-barrier structure is kept by the `else` at the loop foot.
    constexpr uint32_t kBufs = AW ? 1u : 2u;
    __shared__ uint8_t s_b[kBufs][FN * lds_pitch<FK>()];
    // No LDS for A when A is the (fragment-order) weight: a lane loads its fragment bytes straight
    // from global, so the narrow tile needs 4 KiB of LDS instead of 38. One byte, since a
    // zero-length array is not a type.
    __shared__ uint8_t s_a[kBufs][AW ? 1u : FM * lds_pitch<FK>()];
    uint32_t buf = 0;

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    // Split-K. `ck` is a multiple of both 128 (the scale block) and FK, and the launcher derives
    // ksplit from it, so every slice has work and starts on a scale boundary.
    const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
    const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
    if (kbeg >= kend) return;

    // The prefill grid runs in groups of kGroupM activation bands. Blocks dispatch in flattened-id
    // order; with the plain x-fastest grid the activation is read once per 128 weight rows, which
    // halves the rate once it exceeds the 64 MB MALL. Within a group the bands stay cached across
    // the N sweep and the weight is read once per group. With kGroupM bands or fewer the plain order
    // is used and the division skipped.
    uint32_t bm = blockIdx.x, bn = blockIdx.y;
    if (!AW && gridDim.x > kGroupM) {
        const uint32_t per = kGroupM * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kGroupM, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kGroupM);
        bm = first + r % gm;
        bn = r / gm;
    }
    const uint32_t m0 = bm * FM;
    const uint32_t n0 = bn * FN;
    // Wave grid MW(M) x 1(N): a wave owns MT*16 A rows and all FN B rows (see kFMTiles for why
    // there is no second wave axis).
    const uint32_t wm = wave * (MT * 16), wn = 0;
    /* GATED changes which weight rows a wave owns before the epilogue: the gate and up halves are
     * `rows/2` apart, so later uses of `m0 + wm` (fragment tile, liveness, A-side scale) go through
     * wrow0. A wave still owns 16 consecutive rows, so they stay in one scale row. */
    constexpr uint32_t kGHalf = GATED ? (MW / 2u) : MW;
    const uint32_t wrow0 =
        GATED ? ((wave < kGHalf ? 0u : rows / 2u) + blockIdx.x * (FM / 2u)
                     + (wave % kGHalf) * 16u)
              : (m0 + wm);
    static_assert(!GATED || (AW && MT == 1u && (MW % 2u) == 0u),
                  "the gated fold is the narrow tile's epilogue: A is the weight, one M tile a "
                  "wave, and the waves split in half");
    static_assert(!(GATED && I8), "the gated epilogue quantises to E4M3 and has no int8 form");

    f8v raw[MT][NT], out[MT][NT];
#pragma unroll
    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) { raw[i][j] = acc_zero(); out[i][j] = acc_zero(); }

    // ---- staging assignment (a bandwidth decision) ------------------------------------------------
    //
    // A thread stages one 32-byte granule pair, and the threads sharing a row are consecutive, so
    // `kTPR` threads cover a row and one wave load asks for `32 * 32 / kTPR` rows of `kTPR * 32`
    // contiguous bytes. Streaming 5120-byte rows on gfx1201:
    //
    //     threads per row    2         4         8        16        32
    //     GB/s             422       626       633       635       636      (plain stream: 638)
    //
    // Two threads per row leaves a third of the bandwidth unused; four recover it. kTPR = FK/32: 2 on
    // the prefill tile (not fetch-bound), 8 on the 256-byte narrow tile (fetch-bound). A pass covers
    // blockDim/kTPR rows; a thread holds kAP granule pairs at one column across kAP rows.
    constexpr uint32_t kSeg = 32;                       // bytes a thread stages, contiguous
    constexpr uint32_t kTPR = FK / kSeg;                // threads covering one row
    constexpr uint32_t kRPP = (MW * kWave) / kTPR;      // rows one pass covers
    constexpr uint32_t kAP  = AW ? 0u : (FM + kRPP - 1) / kRPP;   // passes over the A tile
    // Fragment-order supertiles per staged K tile when A is the weight: one 32-byte load per lane,
    // four fragments each (see r4d_fp8_frag.h).
    constexpr uint32_t kAF  = AW ? FK / r4d_fp8::kSuperK : 0u;
    constexpr uint32_t kBP  = AW ? (FN + kRPP - 1) / kRPP : 0u;   // ... and over the B tile
    static_assert(kTPR >= 1 && kRPP >= 1, "FK must be at least 32 B and no wider than the block");

    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;
    const uint32_t am = sn;   // A staging, identical assignment

    // The B rows this lane owns, one per N tile (column `frag_c` of each of its NT accumulators,
    // 16 apart). A per-B-row scale must be looked up for each, not just the first.
    const uint32_t brow0 = n0 + wn + frag_c(lane);

    // ---- B from the fragment-order plane, when B is the weight -----------------------------------
    //
    // A wave reads one (16 row, 64 k) tile per pass: 32 lanes x 32 contiguous bytes, the full 1 KiB.
    // It is de-permuted in the LDS store, so LDS holds the usual row-major padded tile.
    constexpr uint32_t kBSK = FK / r4d_fp8::kSuperK;              // k supertiles a staged tile
    constexpr uint32_t kBTL = (FN / 16u) * (kBSK ? kBSK : 1u);    // 16-row tiles in it
    constexpr uint32_t kBFP = AW ? 0u : (kBTL * 32u + MW * kWave - 1u) / (MW * kWave);
    const uint32_t btile = tid / 32u, blane = tid % 32u;
    const uint8_t* bsrc = Bp + (size_t)(n0 + sn) * b_ld + sk;
    const uint8_t* asrc = Ap + (size_t)(m0 + am) * a_ld + sk;
    // A as fragments: a wave owns one 16-row weight tile (MT is 1 on the narrow path), so its A
    // operand for 64 k is this lane's 32 bytes. `ksuper` is the plane's tile stride; `arow` is out
    // of range only when N is not a multiple of FM.
    const uint32_t ksuper = cols / r4d_fp8::kSuperK;
    const uint32_t arow   = wrow0 / 16u;
    const bool     alive  = !AW || wrow0 < rows;
    const uint8_t* afrag  = Ap + (size_t)lane * r4d_fp8::kLaneBytes;

    // One K tile ahead in registers: the load for tile c0+FK is issued before tile c0's WMMA and
    // consumed after it, hiding memory latency under arithmetic. `sk` is the same column for every
    // pass, so only row existence varies per pass.
    const bool k_live = Full || (kbeg + sk + kSeg <= cols);
    // `pa` stages A row-major into LDS; `pf` is A read as fragments. Exactly one is live; the other
    // is a zero-length array.
    /* `pf` is read straight from registers by the mma loop, so its prefetch goes to `pfn` and the
     * loop foot rotates them; prefetching into `pf` would overwrite the current tile (a bug hidden
     * whenever split-K gives a block only one K iteration). */
    Stage8 pb[kBP ? kBP : 1u], pa[kAP ? kAP : 1u], pw[kBFP ? kBFP : 1u];
    Stage8 pf[kAF ? kAF : 1u], pfn[kAF ? kAF : 1u];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + kbeg,
                            k_live && r < FN && (Full || (n0 + r) < nb));
    }
#pragma unroll
    for (uint32_t p = 0; p < kBFP; ++p) {
        const uint32_t ti = btile + p * ((MW * kWave) / 32u);
        const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
        const uint32_t st = kbeg / r4d_fp8::kSuperK + ks;
        pw[p] = stage8_load(Bp + (size_t)((n0 / 16u + rt) * ksuper + st) * r4d_fp8::kTileBytes
                                + (size_t)blane * r4d_fp8::kLaneBytes,
                            ti < kBTL && (n0 + rt * 16u) < nb && st < ksuper);
    }
#pragma unroll
    for (uint32_t p = 0; p < kAP; ++p) {
        const uint32_t r = am + p * kRPP;
        pa[p] = stage8_load(asrc + (size_t)p * kRPP * a_ld + kbeg,
                            k_live && r < FM && (Full || (m0 + r) < rows));
    }
#pragma unroll
    for (uint32_t p = 0; p < kAF; ++p) {
        const uint32_t st = kbeg / r4d_fp8::kSuperK + p;
        pf[p] = stage8_load(
            afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes, alive && st < ksuper);
    }

    for (uint32_t c0 = kbeg; c0 < kend; c0 += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b[buf], r, sk, pb[p]);
        }
#pragma unroll
        for (uint32_t p = 0; p < kBFP; ++p) {
            const uint32_t ti = btile + p * ((MW * kWave) / 32u);
            if (ti >= kBTL) continue;
            const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
            stage8_store_frag<FK>(s_b[buf], rt * 16u + (blane & 15u),
                                  ks * r4d_fp8::kSuperK + (blane >> 4) * 8u, pw[p]);
        }
#pragma unroll
        for (uint32_t p = 0; p < kAP; ++p) {
            const uint32_t r = am + p * kRPP;
            if (r < FM) stage8_store<FK>(s_a[buf], r, sk, pa[p]);
        }
        __syncthreads();
        const uint32_t c1 = c0 + FK;
        if (c1 < kend) {
            const bool live = Full || (c1 + sk + kSeg <= cols);
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r < FN)
                    pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + c1,
                                        live && (Full || (n0 + r) < nb));
            }
#pragma unroll
            for (uint32_t p = 0; p < kBFP; ++p) {
                const uint32_t ti = btile + p * ((MW * kWave) / 32u);
                const uint32_t rt = ti / (kBSK ? kBSK : 1u), ks = ti % (kBSK ? kBSK : 1u);
                const uint32_t st = c1 / r4d_fp8::kSuperK + ks;
                pw[p] = stage8_load(Bp + (size_t)((n0 / 16u + rt) * ksuper + st) * r4d_fp8::kTileBytes
                                        + (size_t)blane * r4d_fp8::kLaneBytes,
                                    ti < kBTL && (n0 + rt * 16u) < nb && st < ksuper);
            }
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = am + p * kRPP;
                if (r < FM)
                    pa[p] = stage8_load(asrc + (size_t)p * kRPP * a_ld + c1,
                                        live && (Full || (m0 + r) < rows));
            }
#pragma unroll
            for (uint32_t p = 0; p < kAF; ++p) {
                const uint32_t st = c1 / r4d_fp8::kSuperK + p;
                pfn[p] = stage8_load(
                    afrag + (size_t)(arow * ksuper + st) * r4d_fp8::kTileBytes,
                    alive && st < ksuper);
            }
        }

        /* fp8 prefill tile: the fold's activation scales (eight per lane) are read at the top of a
         * staged tile that ends in a fold, so its WMMA hide them. At FK <= kBlock the fold fires
         * only at a tile's last step, so that tile is known here. */
        constexpr bool kAsAhead = !AW && !I8 && FK <= kBlock;
        [[maybe_unused]] float asn[kAsAhead ? MT : 1][8];
        if constexpr (kAsAhead) {
            if ((c0 + FK) % kBlock == 0 || c0 + FK >= kend) {
                const uint32_t kb = (c0 + FK - 1) / kBlock;
#pragma unroll
                for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) {
                        const uint32_t r = m0 + wm + i * 16 + acc_row(lane, t);
                        asn[i][t] = (Full || r < rows) ? ((const float*)Sa)[(size_t)r * sa_ld + kb]
                                                       : 0.0f;
                    }
            }
        }
#pragma unroll
        for (uint32_t ks = 0; ks < FK; ks += kK) {
            i2v af[MT], bf[NT];
            // When A is the weight it comes from registers: the 32 bytes this lane loaded for the
            // 64-k supertile are its four fragments (+0, +8, +16, +24). `ks` is constant after
            // unrolling, so the selection folds away.
            if (AW) {
                af[0] = frag_of(pf[ks / r4d_fp8::kSuperK], (ks % r4d_fp8::kSuperK) / 16u);
            } else {
#pragma unroll
                for (uint32_t i = 0; i < MT; ++i)
                    af[i] = frag<FK>(s_a[buf], wm + i * 16, ks, lane);
            }
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j)       bf[j] = frag<FK>(s_b[buf], wn + j * 16, ks, lane);
#pragma unroll
            for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[i][j] = mma8<I8>(af[i], bf[j], raw[i][j]);

            // End of a 128-column scale block: fold both scales in and reset. Inside the ks loop
            // because at FK = 256 one staged tile spans two scale blocks.
            //
            // `c0` is a runtime value, so the compiler if-converts this test at every unrolled ks
            // (libr4d/isa/loop.sh: ~150 of ~190 instructions per eight WMMA on the prefill tile).
            // Gating it at the compile-time-known positions is legal and byte-identical, cuts the
            // loop from 847 to 310 instructions, but raises the kernel from 203 to 253 VGPRs and is
            // ~16% slower on the prefill tile (narrow tiles unchanged). The fold's register live
            // range, not the instruction count, is the cost; see libr4d/isa/README.md.
            if ((c0 + ks + kK) % kBlock == 0 || c0 + ks + kK >= kend) {
                const uint32_t kb = (c0 + ks) / kBlock;
                if (AW && I8) {
                    // A is the int8 weight with a bf16 scale per row. A lane's eight accumulator
                    // slots are eight consecutive rows of one 16-row tile, adjacent in the
                    // tile-major [N/16][K/128][16] plane: one 16-byte load. B is the activation.
                    const uint32_t nkb8 = (cols + kBlock - 1) / kBlock;
                    float wsr[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i) {
                        const uint32_t tr = wrow0 / 16u + i;
                        uint4 w8 = make_uint4(0u, 0u, 0u, 0u);
                        if (Full || tr * 16u < rows)
                            w8 = *(const uint4*)((const uint16_t*)Sa +
                                                 ((size_t)tr * nkb8 + kb) * 16u + (lane >> 4) * 8u);
                        const uint32_t d[4] = { w8.x, w8.y, w8.z, w8.w };
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t)
                            wsr[i][t] = bscale((uint16_t)(d[t / 2] >> (16 * (t % 2))));
                    }
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t tok = brow0 + j * 16;
                        const float s = (Full || tok < nb)
                                            ? ((const float*)Sb)[(size_t)tok * sb_ld + kb] : 0.0f;
#pragma unroll
                        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += acc_val<I8>(raw[i][j][t]) * (wsr[i][t] * s);
                                raw[i][j][t] = 0.0f;
                            }
                    }
                } else if (I8) {
                    // B is the int8 weight with a bf16 scale per row; a lane owns one weight row per
                    // tile (frag_c), so one scale. A is the activation, f32 scale per (token,
                    // 128 K), eight per lane per tile.
                    const uint32_t nkb8 = (cols + kBlock - 1) / kBlock;
                    float wsc[NT];
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t col = n0 + wn + j * 16 + frag_c(lane);
                        wsc[j] = (Full || col < nb)
                                     ? bscale(((const uint16_t*)Sb)[((size_t)(col / 16u) * nkb8 + kb)
                                                                        * 16u + col % 16u])
                                     : 0.0f;
                    }
                    float as[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            const uint32_t r = m0 + wm + i * 16 + acc_row(lane, t);
                            as[i][t] = (Full || r < rows)
                                           ? ((const float*)Sa)[(size_t)r * sa_ld + kb] : 0.0f;
                        }
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += acc_val<I8>(raw[i][j][t]) * (as[i][t] * wsc[j]);
                                raw[i][j][t] = 0.0f;
                            }
                } else if (AW) {
                    // A is the weight: one bf16 scale for this wave's MT*16 rows, which sit in one
                    // scale row (see the static_assert). B is the activation: one f32 scale per
                    // (token, 128-column block), one per lane per tile.
                    const float ws =
                        bscale(((const uint16_t*)Sa)[(size_t)(wrow0 / kBlock) * sa_ld + kb]);
#pragma unroll
                    for (uint32_t j = 0; j < NT; ++j) {
                        const uint32_t tok = brow0 + j * 16;
                        const float s = ws * ((Full || tok < nb)
                                                  ? ((const float*)Sb)[(size_t)tok * sb_ld + kb]
                                                  : 0.0f);
#pragma unroll
                        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += raw[i][j][t] * s;
                                raw[i][j][t] = 0.0f;
                            }
                    }
                } else {
                    // B is the weight: one bf16 scale for the block's B rows. A is the activation:
                    // one f32 scale per token, eight per lane (acc_row).
                    const float ws = bscale(((const uint16_t*)Sb)[(size_t)(n0 / kBlock) * sb_ld + kb]);
                    float as[MT][8];
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            const uint32_t r = m0 + wm + i * 16 + acc_row(lane, t);
                            if constexpr (kAsAhead)
                                as[i][t] = (Full || r < rows) ? ws * asn[i][t] : 0.0f;
                            else
                                as[i][t] = (Full || r < rows)
                                               ? ws * ((const float*)Sa)[(size_t)r * sa_ld + kb]
                                               : 0.0f;
                        }
#pragma unroll
                    for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                out[i][j][t] += raw[i][j][t] * as[i][t];
                                raw[i][j][t] = 0.0f;
                            }
                }
            }
        }
        // Two buffers: the next store lands in the other one. One buffer: wait until every wave has
        // finished reading before it is overwritten. The prefetched fragments become current.
#pragma unroll
        for (uint32_t p = 0; p < kAF; ++p) pf[p] = pfn[p];
        if (kBufs > 1u) buf ^= 1u; else __syncthreads();
    }

    if constexpr (GATED) {
        constexpr uint32_t kStash = kGHalf * kWave * NT * 8u;          // bf16 gate values
        static_assert(kStash * 2u <= FN * lds_pitch<FK>(),
                      "the gate stash must fit in the staging buffer it reuses");
        uint16_t* const gst = (uint16_t*)&s_b[0][0];
        __shared__ float s_amx[kGHalf][FN];
        const uint32_t half = wave / kGHalf, w2 = wave % kGHalf;
        const uint32_t c = frag_c(lane);

        // s_b is still being read by the last K tile's fragments when this is reached.
        __syncthreads();
        if (half == 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (!Full && n0 + wn + j * 16u >= nb) continue;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    gst[((w2 * kWave + lane) * NT + j) * 8u + t] = f32_to_bf16(out[0][j][t]);
            }
        }
        __syncthreads();

        // Same arithmetic and order as r4d_fused_quant_fp8.hip's gated_quant_fp8_kernel: gate and up
        // both round-tripped through bf16, silu in f32, product rounded to bf16, then quantised.
        float v[NT][8];
        if (half != 0u) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                /* Skip a token tile wholly past the end (wave-uniform test); the per-lane guard in
                 * the store is divergent and saves nothing. */
                if (!Full && n0 + wn + j * 16u >= nb) continue;
                float m = 0.0f;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const float g = bf16_to_f32(gst[((w2 * kWave + lane) * NT + j) * 8u + t]);
                    const float a = g / (1.0f + expf(-g));
                    const uint16_t h =
                        f32_to_bf16(a * bf16_to_f32(f32_to_bf16(out[0][j][t])));
                    v[j][t] = bf16_to_f32(h);
                    m = fmaxf(m, fabsf(v[j][t]));
                }
                // A wave's 16 output columns for one token are in lanes `c` and `c ^ 16`; the other
                // 112 columns of the group are in the other waves. fmaxf is exact and order-free.
                m = fmaxf(m, r4d_xor_lane<16>(m));
                if (lane < 16u) s_amx[w2][j * 16u + lane] = m;
            }
        }
        __syncthreads();
        if (half != 0u) {

            const uint32_t col0 = blockIdx.x * (FM / 2u) + w2 * 16u + (lane >> 4) * 8u;
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const uint32_t tok = n0 + wn + j * 16u + c;
                if (!Full && tok >= nb) continue;
                float mm = 0.0f;
#pragma unroll
                for (uint32_t u = 0; u < kGHalf; ++u) mm = fmaxf(mm, s_amx[u][j * 16u + c]);
                const float sc = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
                if (w2 == 0u && (lane >> 4) == 0u) Sc[(size_t)tok * s_ld + blockIdx.x] = sc;
                const float inv = 1.0f / sc;
                // The clamp is required (see r4d_q8.h): the builtin returns NaN out of range, and
                // `amax * (1/scale)` can land an ulp above 448.
                float qc[8];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    qc[t] = fminf(fmaxf(v[j][t] * inv, -448.0f), 448.0f);
                int pk[2] = { 0, 0 };
#pragma unroll
                for (uint32_t d = 0; d < 2; ++d) {
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 0], qc[d * 4 + 1], pk[d],
                                                            false);
                    pk[d] = __builtin_amdgcn_cvt_pk_fp8_f32(qc[d * 4 + 2], qc[d * 4 + 3], pk[d],
                                                            true);
                }
                // Eight consecutive columns: one 8-byte store, aligned since `q_ld` is a multiple of
                // 128 and col0 of 8.
                *(uint2*)(Q + (size_t)tok * q_ld + col0) =
                    make_uint2((uint32_t)pk[0], (uint32_t)pk[1]);
                if (Yb) {
                    /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits); a uint16_t[8]
                     * would be assembled from register halves. */
                    uint16_t* const yr = Yb + (size_t)tok * y_ld + col0;
                    *(uint4*)yr = make_uint4(bf16x2_rne(v[j][0], v[j][1]), bf16x2_rne(v[j][2], v[j][3]),
                                             bf16x2_rne(v[j][4], v[j][5]), bf16x2_rne(v[j][6], v[j][7]));
                }

            }
        }
        return;
    }

    // Split-K writes an f32 partial plane; without it the store is the final bf16 C.
    float* const part = ck ? (float*)Y + (size_t)blockIdx.z * rows * nb : nullptr;
    uint16_t* const c16 = (uint16_t*)Y;

    const uint32_t c = frag_c(lane);
    /* ---- eight consecutive elements in one store ---------------------------------------------
     *
     * `acc_row(lane, t)` is `(lane>>4)*8 + t`, so with AW (output index `col * rows + r`) a lane's
     * eight slots are eight consecutive addresses in one 16-byte piece; one store instead of eight
     * removes the epilogue's M-proportional cost. Bit-identical. Requires AW (otherwise the eight
     * are `nb` apart) and `rows` % 8 == 0 for 16-byte alignment (`r` is already a multiple of 8). */
    const bool vec8 = AW && (rows & 7u) == 0u;
    /* The split-K and vec8 tests are hoisted out of the store loop as template arguments;
     * otherwise the compiler keeps a compare and branch at every store and assembles the vec8
     * words from register halves. */
    auto store = [&](auto split, auto v8) {
        constexpr bool CK = decltype(split)::value;
        constexpr bool V8 = decltype(v8)::value;
#pragma unroll
        for (uint32_t i = 0; i < MT; ++i)
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const uint32_t col = n0 + wn + j * 16 + c;
                if (!Full && col >= nb) continue;
                const uint32_t rb = m0 + wm + i * 16;
                const uint32_t r0 = rb + acc_row(lane, 0);
                if (V8 && (Full || r0 + 8u <= rows)) {
                    const size_t o = (size_t)col * rows + r0;
                    if constexpr (CK) {
                        __attribute__((aligned(16))) float pv[8];
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) pv[t] = out[i][j][t];
                        *(uint4*)(part + o)     = *(const uint4*)pv;
                        *(uint4*)(part + o + 4) = *(const uint4*)(pv + 4);
                    } else {
                        /* Packed in whole registers (bf16x2_rne, f32_to_bf16's bits); a
                         * uint16_t[8] would be assembled from register halves. */
                        const f8v& ov = out[i][j];
                        *(uint4*)(c16 + o) = make_uint4(bf16x2_rne(ov[0], ov[1]), bf16x2_rne(ov[2], ov[3]),
                                                        bf16x2_rne(ov[4], ov[5]), bf16x2_rne(ov[6], ov[7]));
                    }
                    continue;
                }
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = rb + acc_row(lane, t);
                    if (!Full && r >= rows) continue;
                    // AW: A held the weight, so [r][col] is [N][M] and C is [M][N].
                    const size_t o = AW ? (size_t)col * rows + r : (size_t)r * nb + col;
                    if constexpr (CK) part[o] = out[i][j][t];
                    else              c16[o]  = f32_to_bf16(out[i][j][t]);
                }
            }
    };
    if (vec8) {
        if (ck) store(std::true_type{}, std::true_type{});
        else    store(std::false_type{}, std::true_type{});
    } else {
        if (ck) store(std::true_type{}, std::false_type{});
        else    store(std::false_type{}, std::false_type{});
    }

    // ---- split finisher ---------------------------------------------------------------------------
    //
    // Dependent dispatches cost ~2.8 us each on this part (2.3 in a hipGraph), so a separate reduce
    // launch per split GEMM is expensive. Instead the last z block of a column reduces it:
    // `ctr[blockIdx.x]` counts arrivals, and the block that reaches gridDim.z resets it and sums.
    // The z loop runs in fixed ascending order with no atomic on an accumulator, so the result is
    // bit-identical to the separate reduce and deterministic.
    //
    // The counters are a small allocation keyed on (stream, output) (`split_ctr`), not the caller's
    // scratch, which other ops overwrite. The self-reset keeps them zero, which is sound only while
    // launches sharing a counter block never overlap; see split_ctr.
    if (ck && Ctr) {
        // Every thread's stores must be visible device-wide before the arrival is published: a
        // per-thread fence, then the barrier.
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&Ctr[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gridDim.z) {
            if (tid == 0)
                __hip_atomic_store(&Ctr[blockIdx.x], 0u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();                       // acquire the other slices' partials
            // This block owns A rows [m0, m0+FM) and all nb output columns, exactly what the stores
            // above wrote.
            const uint32_t rlo = m0, rhi = min(m0 + FM, rows);
            const float* const w0 = (const float*)Y;
            uint16_t* const cy = (uint16_t*)C;
            for (uint32_t idx = tid; idx < (rhi - rlo) * nb; idx += MW * kWave) {
                const uint32_t r = rlo + idx / nb, col = idx % nb;
                const size_t o = AW ? (size_t)col * rows + r : (size_t)r * nb + col;
                float acc = 0.0f;
                for (uint32_t z = 0; z < gridDim.z; ++z) acc += w0[(size_t)z * rows * nb + o];
                cy[o] = f32_to_bf16(acc);
            }
        }
    }
}

// Sum the split-K partials into bf16 C, in fixed z order with no atomics. Latency bound, so eight
// loads are issued before their adds; the order of the adds is unchanged.
__global__ void splitk_reduce_kernel(const float* __restrict__ ws, uint16_t* __restrict__ y,
                                     uint32_t n_out, uint32_t ksplit) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t)n_out;
    if (i >= n) return;
    const float* __restrict__ p = ws + i;
    constexpr uint32_t U = 8;
    float acc = 0.0f;
    uint32_t z = 0;
    for (; z + U <= ksplit; z += U) {
        float v[U];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) v[u] = p[(size_t)(z + u) * n];
#pragma unroll
        for (uint32_t u = 0; u < U; ++u) acc += v[u];
    }
    for (; z < ksplit; ++z) acc += p[(size_t)z * n];
    y[i] = f32_to_bf16(acc);
}

// The narrow tile's K width is kFKNarrow (256 B); 128 B loses at every wave count (see `kTPR`), and
// is used only via narrow_fk for a K that 256 does not divide.

// Double-buffering the narrow tile is ~10% slower even where it fits (MW = 2): the tile is short
// of LDS, not waiting on the barrier.

// The finisher's arrival counters, one per grid.x column (see the finisher). Zeroed once and kept
// zero by the finisher. If allocation fails, the caller falls back to the separate reduce kernel,
// which computes the same bytes.
constexpr uint32_t kMaxCtr = 8192;

/* One counter block per (stream, output), so concurrent launches never share one (sharing would
 * mean a hang or a wrong answer). Launches overlap across streams (`rad_lane`) or on one stream
 * (rad_kernel_concurrent); overlapping launches on one stream write disjoint outputs, and launches
 * sharing an output on one stream are ordered by their dependency. A step has a few dozen distinct
 * outputs at fixed arena addresses, so this is a few dozen 32 KiB blocks, allocated once. A block
 * is never freed: a recorded pass replays the pointer. */
unsigned int* split_ctr(RadStream s, const void* out) {
    static std::mutex m;
    static std::map<std::pair<RadStream, const void*>, unsigned int*> blocks;
    std::lock_guard<std::mutex> lk(m);
    const auto key = std::make_pair(s, out);
    if (auto it = blocks.find(key); it != blocks.end()) return it->second;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)kMaxCtr * sizeof(unsigned int)) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, (size_t)kMaxCtr * sizeof(unsigned int)) != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    blocks.emplace(key, (unsigned int*)p);
    return (unsigned int*)p;
}
// The narrow tile's wave count MW, which is its M tile: the wave grid is MW x 1, so a block covers
// MW*16 weight rows and the grid over N is N/(MW*16). The rule is fitted together with
// kSplitTarget (which shapes split decides whether narrow or wide blocks win) on the shapes the
// engine actually issues (R4D_F8_MWLOG; n_ff is already per rank).
//
//   - M <= 16: MW = 4, best or within a microsecond on every decode shape.
//   - M > 16: MW = 8. The weight is streamed once whatever MW is, but every block stages the whole
//     [M, K] activation, so a call moves N*K + M*K*N/(MW*16) bytes; the second term matches the
//     weight at M = 64 (served from cache, but it is the drop from 586 to 460 GB/s on gate_up).
//     MW = 8 halves it at the same N/16 waves. The crossover is where NT leaves 1.
//   - M > 32 and N >= 2048: MW = 16 (FM = 256, legal because the AW fold reads the weight scale
//     per wave). Halves the re-read again; narrower N would leave too few blocks for 32 WGPs.
//
// Token tiles NT = ceil(M/16), including 3: rounding 33..48 up to a 64-token tile would spend
// three-eighths of the work at M = 40 on padding. NT = 3 (FN = 48) is legal only here, where AW =
// true; the prefill tile requires FN to divide 128.
//
// Neither MW nor NT changes the output bits: `narrow_ksplit` depends on `gx * mw` = N/16 (the rule
// requires `n % 128` so that holds), so every element is one wave's accumulation over the same K
// tiles in the same order. The selftest checks this with the pin_mw / pin_nt hooks.
//
// This part shows occasional correlated +5 us outliers across repetitions; a cell that breaks its
// row's monotonicity needs re-measuring.
/* Test hook behind r4d_gemm_fp8a8_pin_mw: zero is the rule, anything else overrides every call.
 * Separate from the environment pin, which is read once per process, so a bit check can compare
 * both settings in one process. */
std::atomic<uint32_t> g_mw_force{0u};

uint32_t narrow_mw(uint32_t n, uint32_t m) {
    uint32_t mw = 4u;
    if (m > 16u && (n % 128u) == 0u) mw = 8u;
    if (m > 32u && n >= 2048u && (n % 256u) == 0u) mw = 16u;
    /* Regrouping the same waves into more or fewer blocks (e.g. N = 2560: 40 blocks at MW 4) is
     * slower both ways: the cost is per-block work (activation re-read, scale plane, prologue), not
     * scheduling balance. */
    if (const uint32_t f = g_mw_force.load(std::memory_order_relaxed)) mw = f;
    /* `R4D_F8_MWLOG` prints each distinct (N, M) once, up to 64 (n_ff in a container is already
     * per rank, so the shapes must be read, not derived). */
    static const bool mwlog = [] { const char* v = std::getenv("R4D_F8_MWLOG");
                                   return v && std::strtoul(v, nullptr, 10) != 0; }();
    if (mwlog) {
        static std::mutex lm;
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)n << 32) | m;
        std::lock_guard<std::mutex> lk(lm);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr, "[mwlog] N=%u M=%u -> mw=%u\n", n, m, mw);
        }
    }
    return mw;
}

// Raising the split target so N = 2560 (160 waves, 40 blocks) splits does not change its time, so
// the narrow grid is not parallelism-starved at 128. A split shows in grid.z, not grid.x.

/* The narrow tile's K width: 256, or 128 when K is a multiple of 128 but not 256 (e.g. K = 640, a
 * shared expert's down projection), which would otherwise fall to the prefill kernel. At 128 the
 * fold takes the `FK <= kBlock` branch. One function, because the scratch query and the launch
 * must agree. */
inline uint32_t narrow_fk(uint32_t K) { return (K % kFKNarrow == 0u) ? kFKNarrow : 128u; }

// The split, given the narrow grid. Shared by the scratch query and the launch so they agree on the
// number of partial planes.
uint32_t narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck_out) {
    *ck_out = 0;
    const uint32_t mw = narrow_mw(N, M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = (N + fm - 1) / fm;              // the M grid runs over N here: AW = true
    if (K < 512u || gx * mw >= kSplitTarget) return 1;
    const uint32_t want = (kSplitTarget + gx * mw - 1) / (gx * mw);
    uint32_t ckw = ((K / want) + kBlock - 1u) / kBlock * kBlock;
    if (ckw < kBlock) ckw = kBlock;
    uint32_t k2 = (K + ckw - 1) / ckw;
    if (k2 > kSplitThinCap && (uint64_t)N * M < 8192) k2 = kSplitThinCap;
    if (k2 < 2) return 1;
    // Round the slice to the staged tile, not the 128-column scale block: a slice that is only a
    // multiple of 128 would run one 256-wide tile past its end into the next slice.

    const uint32_t fk = narrow_fk(K);
    uint32_t ck = ((K + k2 - 1) / k2 + fk - 1u) / fk * fk;
    k2 = (K + ck - 1) / ck;
    if (k2 < 2) return 1;
    *ck_out = ck;
    return k2;
}

}  // namespace

// ================================================================== the entry points

extern "C" int r4d_gemm_fp8a8_block(void)          { return (int)kBlock; }
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, RadStream stream,
                                  bool i8 = false);

extern "C" int r4d_gemm_fp8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                     int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                     long scratch, long scratch_bytes, long stream) {
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, (RadStream)(uintptr_t)stream);
}

extern "C" int r4d_gemm_fp8a8_nt_m16_max_m(void)   { return 64; }

/* See r4d.h. Only instantiated wave counts are accepted; anything else clears the override rather
 * than falling through to the dispatch's last arm. */
extern "C" void r4d_gemm_fp8a8_pin_mw(int mw) {
    const uint32_t v = (mw == 1 || mw == 2 || mw == 4 || mw == 8 || mw == 16) ? (uint32_t)mw : 0u;
    g_mw_force.store(v, std::memory_order_relaxed);
}

/* The same hook for the token-tile count. `nt` may be pinned up but never below ceil(M/16): NT*16
 * is how many token columns the epilogue writes, so a lower pin would drop rows. */
std::atomic<uint32_t> g_nt_force{0u};

extern "C" void r4d_gemm_fp8a8_pin_nt(int nt) {
    const uint32_t v = (nt == 1 || nt == 2 || nt == 3 || nt == 4) ? (uint32_t)nt : 0u;
    g_nt_force.store(v, std::memory_order_relaxed);
}

// Bytes of f32 scratch one r4d_gemm_fp8a8_nt_m16 call needs, or 0 when it will not split. The
// caller may pass more, or none (the launch then runs unsplit: correct but slow, and unreported).
// The bound is max_m() so the query and the band cannot drift apart.
extern "C" long r4d_gemm_fp8a8_nt_m16_scratch(int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0 || M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    uint32_t ck = 0;
    const uint32_t k2 = narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck);
    return k2 > 1 ? (long)k2 * M * N * (long)sizeof(float) : 0;
}

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, M <= 64.
//
//   a       [M][K]                     E4M3, row-major, `a_ld` bytes a row (0 = K)
//   ascale  [M][ceil(K/128)]           f32,  `sa_ld` elements a row (0 = ceil(K/128))
//   wq      [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws      [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c       [M][N]                     bf16
//
// Returns 0, or a negative code for a shape it does not serve. `scratch` may be 0.
static int gemm_fp8a8_nt_m16_emit(long a, long ascale, long wq, long ws, long c,
                                  int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                  long scratch, long scratch_bytes, RadStream stream, bool i8) {
    int emit_rc = RAD_OK;
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // K must be a multiple of 128; the staged tile is 256 bytes where K allows and 128 otherwise
    // (narrow_fk). The declared constraint must match this guard (cGemmNtFp8A8M16 in r4d_rows.cpp):
    // the resolver offers a call by what a row claims, not by what the entry point accepts.
    if (K % 128) return -4;
    // A wave owns 16 weight rows and the transposing store writes `N`-strided columns.
    if (N % 16) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    const uint32_t mw = narrow_mw((uint32_t)N, (uint32_t)M);
    const uint32_t fm = mw * 16u;   // the narrow tile is one M tile a wave
    const uint32_t gx = ((uint32_t)N + fm - 1) / fm;

    uint32_t ck = 0;
    uint32_t ksplit = narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck);
    const long need = ksplit > 1 ? (long)ksplit * M * N * (long)sizeof(float) : 0;
    if (need && (!scratch || scratch_bytes < need)) { ksplit = 1; ck = 0; }

    const dim3 grid(gx, 1, ksplit);
    void* const dst = ksplit > 1 ? (void*)scratch : (void*)c;
    /* Token tiles per wave: the activation is the B operand, so NT tiles cover NT*16 tokens with
     * one weight read. NT = ceil(M/16), including 3 (see the note above narrow_mw). The gate stash
     * scales with NT on both sides (4096*NT bytes against FN * (256+8) = 4224*NT), so it fits. */
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    if (const uint32_t f = g_nt_force.load(std::memory_order_relaxed)) { if ((uint32_t)M <= f * 16u) nt = f; }
    /* `Full` drops the column guard, so it must mean all NT tiles are live, not just M % 16 == 0.
     * Otherwise (e.g. M = 48, NT = 4) the epilogue writes past a split-K scratch sized for M, over
     * the arrival counter, and the finisher hangs. */
    const bool full = ((uint32_t)N % fm == 0) && ((uint32_t)M % (nt * 16u) == 0) &&
                      (K % (int)kBlock == 0);
    // The finisher needs one counter per column; beyond kMaxCtr or on allocation failure, the
    // separate reduce kernel computes the same bytes.
    unsigned int* const ctr =
        (ksplit > 1 && gx <= kMaxCtr) ? split_ctr(stream, (const void*)c) : nullptr;

#define R4D_F8D(f, mw_, nt_, fk_, i8_)                                                         \
    emit_rc = R4D_EMIT(stream, (gemm_fp8a8_kernel<f, true, nt_, 1u, mw_, fk_, false, i8_>),   \
                       grid, dim3(mw_ * kWave),                                                \
                       0, (const uint8_t*)wq, (const void*)ws,                                 \
                       (const uint8_t*)a, (const void*)ascale, dst, (void*)c, ctr,              \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, ck,                                                     \
                       /* The six defaulted parameters, spelled out: r4d_emit passes            \
                        * kernelParams as an array, which cannot fill defaults, and asserts the  \
                        * arity. */                                                              \
                       (uint8_t*)nullptr, (float*)nullptr, (uint16_t*)nullptr,                  \
                       (uint32_t)0, (uint32_t)0, (uint32_t)0)
#define R4D_F8D_NT(f, mw_, fk_, i8_)                                                           \
    do { if (nt == 4u)      R4D_F8D(f, mw_, 4u, fk_, i8_);                                     \
         else if (nt == 3u) R4D_F8D(f, mw_, 3u, fk_, i8_);                                     \
         else if (nt == 2u) R4D_F8D(f, mw_, 2u, fk_, i8_);                                     \
         else               R4D_F8D(f, mw_, 1u, fk_, i8_); } while (0)
#define R4D_F8D_MW(fk_, i8_)                                                                   \
    do { if (full) {                                                                           \
             if (mw == 4u)      R4D_F8D_NT(true, 4u, fk_, i8_);                                \
             else if (mw == 8u) R4D_F8D_NT(true, 8u, fk_, i8_);                                \
             else               R4D_F8D_NT(true, 16u, fk_, i8_);                               \
         } else {                                                                              \
             if (mw == 4u)      R4D_F8D_NT(false, 4u, fk_, i8_);                               \
             else if (mw == 8u) R4D_F8D_NT(false, 8u, fk_, i8_);                               \
             else               R4D_F8D_NT(false, 16u, fk_, i8_);                              \
         } } while (0)
    const uint32_t fk = narrow_fk((uint32_t)K);
    if (mw == 1u || mw == 2u) {
        /* Only the selftest's pin reaches these (narrow_mw answers 4, 8 or 16), and only the fp8
         * set at the 256-wide K step builds them, to keep the object size down. */
        if (i8 || fk != kFKNarrow) return -4;
        if (full) { if (mw == 1u) R4D_F8D_NT(true, 1u, kFKNarrow, false);
                    else          R4D_F8D_NT(true, 2u, kFKNarrow, false); }
        else      { if (mw == 1u) R4D_F8D_NT(false, 1u, kFKNarrow, false);
                    else          R4D_F8D_NT(false, 2u, kFKNarrow, false); }
    } else if (fk == kFKNarrow) {
        if (i8) R4D_F8D_MW(kFKNarrow, true); else R4D_F8D_MW(kFKNarrow, false);
    } else {
        if (i8) R4D_F8D_MW(128u, true); else R4D_F8D_MW(128u, false);
    }
#undef R4D_F8D_MW
#undef R4D_F8D_NT
#undef R4D_F8D
    if (emit_rc != RAD_OK) return -6;
    if (hipGetLastError() != hipSuccess) return -6;

    if (ksplit > 1 && !ctr) {
        const uint32_t n = (uint32_t)M * (uint32_t)N;
        if (R4D_EMIT(stream, splitk_reduce_kernel, dim3((n + 255) / 256), dim3(256), 0,
                     (const float*)scratch, (uint16_t*)c, n, ksplit) != RAD_OK) return -7;
    }
    return 0;
}

// The same GEMM with gated_quant_fp8's pass folded into its epilogue (see `GATED` on the kernel).
// Every unmet shape condition is a refusal: a gated call running ungated would write the wrong
// buffer.
//
//   q       [M][nout]                  E4M3, `q_ld` bytes a row (0 = nout)
//   s       [M][ceil(nout/128)]        f32,  `s_ld` elements a row
//   y       [M][nout] or 0             bf16, the silu product the unfused pair also wrote
//
// The weight is the gate_up plane [2*nout][K], gate first, as the unfused GEMM reads it.
extern "C" int r4d_gemm_fp8a8_gated_nt_m16_max_m(void) { return 64; }

extern "C" int r4d_gemm_fp8a8_gated_nt_m16(long a, long ascale, long wq, long ws,
                                           long q, long s, long y, int act,
                                           int M, int K, int nout,
                                           int a_ld, int sa_ld, int w_ld, int sw_ld,
                                           int q_ld, int s_ld, int y_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !q || !s) return -1;
    if (M <= 0 || K <= 0 || nout <= 0) return -2;
    if (M > r4d_gemm_fp8a8_gated_nt_m16_max_m()) return -3;
    if (K % (int)kFKNarrow) return -4;
    // A block emits 128 fused columns, one scale group, so nout must be a multiple of 128.
    if (nout % 128) return -5;
    // Only silu: the sigmoid form (gated attention) goes through `gate_quant_fp8` instead.
    if (act != 0) return -8;
    const int N = 2 * nout;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;
    if (!q_ld)  q_ld  = nout;
    if (!s_ld)  s_ld  = nout / 128;
    if (!y_ld)  y_ld  = nout;

    // The fold cannot split K, so a shape the split rule would split is refused (gate_up's shape,
    // N/16 = 1088, never splits).
    uint32_t ck = 0;
    if (narrow_ksplit((uint32_t)N, (uint32_t)K, (uint32_t)M, &ck) != 1u) return -9;

    constexpr uint32_t kGmw = 16u;                  // FM = 256: 128 gate rows and 128 up rows
    /* Same NT rule as the narrow dispatch: a tile past M is skipped in the epilogue but still costs
     * every K iteration. */
    uint32_t nt = (M > 48) ? 4u : ((M > 32) ? 3u : ((M > 16) ? 2u : 1u));
    if (const uint32_t f = g_nt_force.load(std::memory_order_relaxed)) { if ((uint32_t)M <= f * 16u) nt = f; }
    const bool full = ((uint32_t)M % (nt * 16u) == 0) && (K % (int)kBlock == 0);
    const dim3 grid((uint32_t)(nout / 128), 1, 1);

#define R4D_F8G_NT(f, nt_)                                                                     \
    hipLaunchKernelGGL((gemm_fp8a8_kernel<f, true, nt_, 1u, kGmw, kFKNarrow, true>), grid,      \
                       dim3(kGmw * kWave),                                                     \
                       0, (hipStream_t)stream, (const uint8_t*)wq, (const void*)ws,            \
                       (const uint8_t*)a, (const void*)ascale, nullptr, nullptr, nullptr,       \
                       (uint32_t)N, (uint32_t)K,                                                \
                       (uint32_t)M, (uint32_t)w_ld, (uint32_t)a_ld, (uint32_t)sw_ld,            \
                       (uint32_t)sa_ld, 0u, (uint8_t*)q, (float*)s, (uint16_t*)y,               \
                       (uint32_t)q_ld, (uint32_t)s_ld, (uint32_t)y_ld)
#define R4D_F8G(f)                                                                             \
    do { if (nt == 4u)      R4D_F8G_NT(f, 4u);                                                 \
         else if (nt == 3u) R4D_F8G_NT(f, 3u);                                                 \
         else if (nt == 2u) R4D_F8G_NT(f, 2u);                                                 \
         else               R4D_F8G_NT(f, 1u); } while (0)
    if (full) R4D_F8G(true); else R4D_F8G(false);
#undef R4D_F8G
#undef R4D_F8G_NT
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// The same GEMM for prefill row counts: same operands and layouts, but a 128x128 tile with the
// activation on A, so the store is coalesced. No split-K: the grid is ceil(M/128) x ceil(N/128),
// hundreds of blocks at any prefill shape.
template <bool I8>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream);

extern "C" int r4d_gemm_fp8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long stream) {
    return gemm_8a8_tiled<false>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

// The int8 twins: i8 codes for both operands and a bf16 weight scale per row per 128 K (ws is
// [N][ceil(K/128)]); otherwise as the fp8 entry points.
extern "C" int r4d_gemm_i8a8_tiled(long a, long ascale, long wq, long ws, long c,
                                   int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                   long stream) {
    return gemm_8a8_tiled<true>(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld, stream);
}

extern "C" int r4d_gemm_i8a8_nt_m16(long a, long ascale, long wq, long ws, long c,
                                    int M, int K, int N, int a_ld, int sa_ld, int w_ld, int sw_ld,
                                    long scratch, long scratch_bytes, long stream) {
    return gemm_fp8a8_nt_m16_emit(a, ascale, wq, ws, c, M, K, N, a_ld, sa_ld, w_ld, sw_ld,
                                  scratch, scratch_bytes, (RadStream)(uintptr_t)stream, true);
}

template <bool I8>
static int gemm_8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int a_ld, int sa_ld, int w_ld, int sw_ld, long stream) {
    if (!a || !ascale || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    // Required: the fold assumes whole 128-column scale blocks, and the AW=false branch reads the
    // weight's scale as one scalar for 128 B rows.
    if (K % (int)kBlock) return -4;
    if (N % (int)kBlock) return -5;
    if (!a_ld)  a_ld  = K;
    if (!w_ld)  w_ld  = K;
    const int nkb = (K + (int)kBlock - 1) / (int)kBlock;
    if (!sa_ld) sa_ld = nkb;
    if (!sw_ld) sw_ld = nkb;

    constexpr uint32_t FM = kFWaves * kFMTiles * 16;
    const dim3 grid(((uint32_t)M + FM - 1) / FM, ((uint32_t)N + kFN - 1) / kFN, 1);
    const bool full = ((uint32_t)M % FM == 0);         // N % 128 and K % 128 are already required
    if (full)
        hipLaunchKernelGGL((gemm_fp8a8_kernel<true, false, kFNTiles, kFMTiles, kFWaves, kFK, false, I8>), grid,
                           dim3(kFWaves * kWave), 0, (hipStream_t)stream, (const uint8_t*)a,
                           (const void*)ascale, (const uint8_t*)wq, (const void*)ws, (void*)c, (void*)c, nullptr,
                           (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)w_ld,
                           (uint32_t)sa_ld, (uint32_t)sw_ld, 0u);
    else
        hipLaunchKernelGGL((gemm_fp8a8_kernel<false, false, kFNTiles, kFMTiles, kFWaves, kFK, false, I8>), grid,
                           dim3(kFWaves * kWave), 0, (hipStream_t)stream, (const uint8_t*)a,
                           (const void*)ascale, (const uint8_t*)wq, (const void*)ws, (void*)c, (void*)c, nullptr,
                           (uint32_t)M, (uint32_t)K, (uint32_t)N, (uint32_t)a_ld, (uint32_t)w_ld,
                           (uint32_t)sa_ld, (uint32_t)sw_ld, 0u);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h; see libr4d/r4d_args.h.
extern "C" int64_t r4d_rad_gemm_fp8a8_m16_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0),
                    K = r4d_geti_or(a, "K", 0);
    if (M < 1 || N < 1 || K < 1) return 0;
    if (M > r4d_gemm_fp8a8_nt_m16_max_m()) return 0;
    return (int64_t)r4d_gemm_fp8a8_nt_m16_scratch((int)M, (int)K, (int)N);
}

extern "C" int r4d_rad_gemm_fp8a8_m16(const RadArgs* a, RadStream stream) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    /* A short scratch is not an error: the launch runs unsplit (correct, slower). The length is
     * passed so the launch never reads past an arena sized for a different band. */
    const int e = gemm_fp8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, stream);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* The gated GEMM's shim. Not fp8_ops: that derives M and N from operand 4, which here is the
 * quantiser's [M, nout] output rather than the GEMM's [M, 2*nout]. Operand indices are
 * `gemm_nt_q_gated`'s (oGemmNtQGated in r4d_rows.cpp). */
enum { GG_A = 0, GG_AS = 1, GG_B = 2, GG_BS = 3, GG_Q = 4, GG_S = 5, GG_Y = 6 };

extern "C" int r4d_rad_gemm_fp8a8_gated_m16(const RadArgs* a, RadStream s) {
    const RadTensor* x   = r4d_opt(a, GG_A);
    const RadTensor* xs  = r4d_opt(a, GG_AS);
    const RadTensor* w   = r4d_opt(a, GG_B);
    const RadTensor* wsc = r4d_opt(a, GG_BS);
    const RadTensor* q   = r4d_opt(a, GG_Q);
    const RadTensor* sc  = r4d_opt(a, GG_S);
    const RadTensor* y   = r4d_opt(a, GG_Y);
    if (!x || !xs || !w || !wsc || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (w->rank != 2 || wsc->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (x->rank < 2 || xs->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);

    const long long blk = r4d_gemm_fp8a8_block();
    if (blk != R4D_FP8_BLOCK) return R4D_NO(RAD_E_SHAPE);
    if (r4d_geti_or(a, "group", blk) != blk) return R4D_NO(RAD_E_SHAPE);

    /* nout is the fused (quantiser) width, taken from `q`, the buffer written; the weight is twice
     * it. */
    const long long M = q->shape[0], nout = q->shape[1], K = w->shape[1];
    if (M < 1 || nout < 1 || K < 1) return R4D_NO(RAD_E_SHAPE);
    if (M > r4d_gemm_fp8a8_gated_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (w->shape[0] != 2 * nout) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + blk - 1) / blk;
    if (wsc->shape[0] != (2 * nout + blk - 1) / blk || wsc->shape[1] != nkb)
        return R4D_NO(RAD_E_SHAPE);
    if (x->shape[0] != M || r4d_numel(x) != M * K) return R4D_NO(RAD_E_SHAPE);
    if (xs->shape[0] != M || xs->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != (nout + blk - 1) / blk) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != nout)) return R4D_NO(RAD_E_SHAPE);

    const long long ap = r4d_row_pitch(x), asp = r4d_row_pitch(xs);
    const long long wp = r4d_row_pitch(w), wsp = r4d_row_pitch(wsc);
    const long long qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    if (ap < 0 || asp < 0 || wp < 0 || wsp < 0 || qp < 0 || sp < 0 || yp < 0)
        return R4D_NO(RAD_E_STRIDE);

    /* `act` is required by this schema (unlike gated_quant_fp8's); only silu is compiled. */
    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "silu")) return R4D_NO(RAD_E_UNSUPPORTED);

    const int e = r4d_gemm_fp8a8_gated_nt_m16(
        r4d_addr(x), r4d_addr(xs), r4d_addr(w), r4d_addr(wsc),
        r4d_addr(q), r4d_addr(sc), y ? r4d_addr(y) : 0, 0,
        (int)M, (int)K, (int)nout, (int)ap, (int)asp, (int)wp, (int)wsp,
        (int)qp, (int)sp, (int)yp, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* ---- the int8 twins' shims: the same operands with i8 codes and the weight's scale plane per
 * row, which fp8_ops checks at [N][K/128] when asked (FP8_OPS_ROW_SCALE). */
extern "C" int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs* a) {
    return r4d_rad_gemm_fp8a8_m16_scratch(a);
}

extern "C" int r4d_rad_gemm_i8a8_m16(const RadArgs* a, RadStream stream) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a8_nt_m16_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = gemm_fp8a8_nt_m16_emit(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, stream, true);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_i8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1, FP8_OPS_ROW_SCALE);
    if (rc != RAD_OK) return rc;
    if (g.a->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const int e = r4d_gemm_i8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gemm_fp8a8_tiled(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    const int e = r4d_gemm_fp8a8_tiled(
        r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
        (int)g.M, (int)g.K, (int)g.N, (int)g.a_ld, (int)g.sa_ld, (int)g.w_ld, (int)g.sw_ld,
        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

