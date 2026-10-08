// ParoQuant's activation side: channel scale, Givens rotations, then the per-(row, 128-column)
// symmetric E4M3 quantiser of r4d_quant_act_fp8.hip. The weight it pairs with is stored as
// v = w . D . G_1 ... G_S (abi/rad_encoding.h, "givens"), so the product <v, x'> with
// x' = x . D^-1 . G_1 ... G_S is <w, x>: the rotation is the activation's to do, once a token, and
// the weight is read by the plain MXFP4 GEMM unchanged.
//
// WHY THE ROTATION IS A GROUP AT A TIME. ParoQuant's pairs never leave a group of 128 columns --
// that is how the checkpoint is trained, and the converter and the loader (r4d_rad_relayout_givens_pairs)
// refuse one that does -- so a group is a closed system: its 128 values are rotated by S stages of 64 disjoint pairs and then quantised
// with ONE scale. A wave owns a group. The values sit in LDS (128 floats a wave, no barrier
// needed: a wave's LDS accesses are in order, and the pairs of a stage are disjoint so no two lanes
// touch one address), each lane takes two pairs a stage, and the absmax is a wave reduction. The
// quantiser after it is quant_act_fp8's, byte for byte.
//
// PARTS. A fused projection (gate|up, q|k|v) was rotated per piece, so a row yields `parts` rotated
// copies, side by side: q is [M, parts * K] and scale [M, parts * K/128]. A (part, group) pair is
// one unit of work, one wave's (a row is several blocks, grid.y). The tables stack the parts
// as the "givens" encoding does -- stages rows each of pairs and angle, one row of channel scales.
//
// THE CHANNEL SCALE IS A DIVIDE, because the encoding's t.scale is the factor the stored row was
// multiplied by (D), and the activation takes its inverse. A true IEEE divide on both sides: the
// reference (libref) divides in f32 too, so the two agree on this step exactly.
//
// COS AND SIN ARE COMPUTED HERE, in f32, from the stored angle: sincosf is accurate to an ulp or
// two, which is also what libref's libm is. The rotation is a handful of FMAs on top of that; the
// quantiser's rounding to E4M3 then swallows nearly all of the difference, and rad-kbench-style
// comparisons against the reference allow one code of disagreement per element.
//
// givens_quant_act_i8g is the same producer with an int8 output (scale amax / 127, round to nearest) for
// the int8 prefill GEMMs: after the rotation a group is flat, and 127 uniform levels beat e4m3's three
// mantissa bits for nearly every element.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_fp8_wmma.h"
#include "r4d_q8.h"
#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;
constexpr uint32_t kPairsPerLane = kGroup / 2 / kWave;      // 2

// kPre stages' tables are loaded and their angles' sine and cosine taken BEFORE the first of them is
// applied, so the dependent chain a stage would otherwise be -- table load, LDS read, rotate, LDS
// write -- overlaps its neighbours' loads. At decode (M of a handful of rows) the kernel is latency
// and not bandwidth: a row is 40 groups of 128 and each group is eight serial stages.
constexpr uint32_t kPre = 8;

__device__ __forceinline__ void wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// One wave rotates and quantises one (part, group) of one row: grid.x is the row, and grid.y
// spreads a row's `parts * nkb` groups over blocks of kWaves waves -- one group a wave, so a row is
// several blocks and a decode step of a few rows still fills the part. A wave that has no group
// returns at once; nothing in the kernel is a block barrier.
// Rotate is givens_rotate: the same rotation, no quantiser, the rotated row written as f16 into
// `q` (read as a uint16 plane, y_ld elements a row) for a GEMM that reads a 16-bit activation.
// Fp8 and I8 are the two quantisers (r4d_q8.h: E4M3 with scale amax/448, int8 with amax/127).
enum class GvMode { Rotate, Fp8, I8 };

template <GvMode MODE>
__global__ __launch_bounds__(kThreads) void givens_quant_act_fp8_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ pairs,
    const float* __restrict__ angle, const float* __restrict__ rscale, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t K, uint32_t nkb, uint32_t parts, uint32_t stages,
    uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    constexpr bool I8 = MODE == GvMode::I8;
    __shared__ float sbuf[kWaves][kGroup];
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint32_t item = blockIdx.y * kWaves + wave;
    if (item >= parts * nkb) return;
    const uint32_t part = item / nkb, g = item % nkb;
    const uint32_t c0 = g * kGroup;
    const size_t xrow = (size_t)row * x_ld;
    float* buf = sbuf[wave];

    // channel scale: v = x / D, four consecutive columns a lane
    {
        const float4 d = *(const float4*)(rscale + (size_t)part * K + c0 + lane * 4u);
        const uint32_t xc = c0 + lane * 4u;
        const float v0 = __uint_as_float((uint32_t)x[xrow + xc + 0] << 16) / d.x;
        const float v1 = __uint_as_float((uint32_t)x[xrow + xc + 1] << 16) / d.y;
        const float v2 = __uint_as_float((uint32_t)x[xrow + xc + 2] << 16) / d.z;
        const float v3 = __uint_as_float((uint32_t)x[xrow + xc + 3] << 16) / d.w;
        *(float4*)(buf + lane * 4u) = make_float4(v0, v1, v2, v3);
    }

    for (uint32_t st0 = 0; st0 < stages; st0 += kPre) {
        const uint32_t nst = (stages - st0) < kPre ? (stages - st0) : kPre;
        int2  ab[kPre][kPairsPerLane];
        float th[kPre][kPairsPerLane], cs[kPre][kPairsPerLane], sn[kPre][kPairsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst) {
                const size_t trow = (size_t)part * stages + st0 + k;
                const int32_t* pr = pairs + trow * K + c0;
                const float* an = angle + trow * (K / 2u) + g * (kGroup / 2u);
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) {
                    const uint32_t j = lane + u * kWave;
                    ab[k][u] = *(const int2*)(pr + 2u * j);
                    th[k][u] = an[j];
                }
            }
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst)
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) sincosf(th[k][u], &sn[k][u], &cs[k][u]);
        wave_sync();            // the scaled row (or the previous chunk's last stage) is in LDS
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst) {
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) {
                    const int ia = ab[k][u].x - (int)c0, ib = ab[k][u].y - (int)c0;
                    const float va = buf[ia], vb = buf[ib];
                    buf[ia] = fmaf(cs[k][u], va, sn[k][u] * vb);
                    buf[ib] = fmaf(cs[k][u], vb, -sn[k][u] * va);
                }
                wave_sync();
            }
    }

    const float4 r = *(const float4*)(buf + lane * 4u);
    float v[4] = { r.x, r.y, r.z, r.w };
    if constexpr (MODE == GvMode::Rotate) {
        // f16 holds +-65504; a rotated post-norm row is orders below that, and the clamp keeps an
        // outlier finite instead of inf
        union { __half h[4]; uint2 u; } o;
#pragma unroll
        for (int j = 0; j < 4; ++j) o.h[j] = __float2half(fminf(fmaxf(v[j], -65504.0f), 65504.0f));
        *(uint2*)((uint16_t*)q + (size_t)row * q_ld + (size_t)part * K + c0 + lane * 4u) = o.u;
        return;
    }
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
    m = r4d_wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    if (lane == 0) s[(size_t)row * s_ld + (size_t)part * nkb + g] = sc;
    *(uint32_t*)(q + (size_t)row * q_ld + (size_t)part * K + c0 + lane * 4u) =
        q8_pack4<I8>(v[0], v[1], v[2], v[3], 1.0f / sc);
}

}  // namespace

//   x       [M][K]               bf16, `x_ld` elements a row (0 = K)
//   pairs   [parts*stages][K]    int32 column indices; (pairs[2j], pairs[2j+1]) is pair j
//   angle   [parts*stages][K/2]  f32
//   rscale  [parts][K]           f32, the factor the stored row was multiplied by
//   q       [M][parts*K]         E4M3
//   s       [M][parts*K/128]     f32
static int givens_quant_launch(long x, long pairs, long angle, long rscale, long q, long s, int M,
                               int K, int parts, int stages, int x_ld, int q_ld, int s_ld,
                               const R4dSink& sk, GvMode mode = GvMode::Fp8) {
    if (!x || !pairs || !angle || !rscale || !q || (mode != GvMode::Rotate && !s)) return -1;
    if (M <= 0 || K <= 0 || parts <= 0 || stages <= 0) return -2;
    if (K % (int)kGroup) return -3;       // a rotation group cannot straddle the end of a row
    if (!x_ld) x_ld = K;
    const int nkb = K / (int)kGroup;
    if (!q_ld) q_ld = parts * K;
    if (!s_ld) s_ld = parts * nkb;
    const uint32_t gy = ((uint32_t)(parts * nkb) + kWaves - 1) / kWaves;
#define GV_EMIT(M_) R4D_EMIT(sk, (givens_quant_act_fp8_kernel<M_>), dim3((uint32_t)M, gy), dim3(kThreads), 0, \
        (const uint16_t*)x, (const int32_t*)pairs, (const float*)angle, (const float*)rscale, \
        (uint8_t*)q, (float*)s, (uint32_t)K, (uint32_t)nkb, (uint32_t)parts, (uint32_t)stages, \
        (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld)
    const int rc = mode == GvMode::I8 ? GV_EMIT(GvMode::I8)
                 : mode == GvMode::Fp8 ? GV_EMIT(GvMode::Fp8) : GV_EMIT(GvMode::Rotate);
#undef GV_EMIT
    if (rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_givens_quant_act_fp8_group(void) { return (int)kGroup; }

// ============================================================== the plugin ABI entry point
enum { GV_X = 0, GV_PAIRS = 1, GV_ANGLE = 2, GV_RSCALE = 3, GV_Q = 4, GV_S = 5 };

static int givens_body(const RadArgs* a, const R4dSink& sk, GvMode mode) {
    const bool quant = mode != GvMode::Rotate;
    const RadTensor* x  = r4d_opt(a, GV_X);
    const RadTensor* pr = r4d_opt(a, GV_PAIRS);
    const RadTensor* an = r4d_opt(a, GV_ANGLE);
    const RadTensor* rs = r4d_opt(a, GV_RSCALE);
    const RadTensor* q  = r4d_opt(a, GV_Q);
    const RadTensor* sc = quant ? r4d_opt(a, GV_S) : nullptr;
    if (!x || !pr || !an || !rs || !q || (quant && !sc)) return R4D_NO(RAD_E_INVAL);
    if (x->rank < 2 || q->rank != 2 || (quant && sc->rank != 2) || pr->rank != 2 || an->rank != 2 ||
        rs->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (mode == GvMode::Rotate && q->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (mode == GvMode::I8 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (pr->dtype != RAD_I32 || an->dtype != RAD_F32 || rs->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    // rows are dim 0 and everything after it is the row, as quant_act_fp8 reads them
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long grp = (long long)kGroup, nkb = K / grp;
    const long long parts = r4d_geti_or(a, "parts", 1), stages = r4d_geti_or(a, "stages", 0);
    if (r4d_geti_or(a, "group", grp) != grp || K % grp || parts < 1 || stages < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != parts * K) return R4D_NO(RAD_E_SHAPE);
    if (quant && (sc->shape[0] != M || sc->shape[1] != parts * nkb)) return R4D_NO(RAD_E_SHAPE);
    // the tables are the weight's, contiguous: parts * stages rows of K pairs and K/2 angles
    if (pr->shape[0] != parts * stages || pr->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (an->shape[0] != parts * stages || an->shape[1] != K / 2) return R4D_NO(RAD_E_SHAPE);
    if (rs->shape[0] != parts || rs->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(rs->data, 16) || !r4d_aligned(pr->data, 8) || !r4d_aligned(q->data, 4))
        return R4D_NO(RAD_E_ALIGN);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = quant ? r4d_row_pitch(sc) : 0;
    if (xp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = givens_quant_launch(r4d_addr(x), r4d_addr(pr), r4d_addr(an), r4d_addr(rs),
                                      r4d_addr(q), r4d_addr(sc), (int)M, (int)K, (int)parts,
                                      (int)stages, (int)xp, (int)qp, (int)sp, sk, mode);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
static int rad_givens_quant_act_fp8_body(const RadArgs* a, const R4dSink& sk) {
    return givens_body(a, sk, GvMode::Fp8);
}
static int rad_givens_quant_act_i8g_body(const RadArgs* a, const R4dSink& sk) {
    return givens_body(a, sk, GvMode::I8);
}
static int rad_givens_rotate_body(const RadArgs* a, const R4dSink& sk) {
    return givens_body(a, sk, GvMode::Rotate);
}
R4D_ENTRY(r4d_rad_givens_quant_act_fp8, rad_givens_quant_act_fp8_body)
R4D_ENTRY(r4d_rad_givens_rotate, rad_givens_rotate_body)
R4D_ENTRY(r4d_rad_givens_quant_act_i8g, rad_givens_quant_act_i8g_body)
