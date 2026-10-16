// Per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation, the input to the fp8a8
// GEMMs. The checkpoint's activation_scheme is "dynamic" (no stored activation scale), and a GEMM
// cannot compute one without a reduction across every M block.
//
// Output is plain row-major, unlike r4d_quant_act_i8 (which writes gemm_w4a8_nt_m64's register
// fragment order): the fp8a8 GEMMs stage both operands through the same `stage8` copy and fragment
// read, and row-major lets the activation be the B operand of the narrow instantiation.
//
// The group is fixed at 128, the checkpoint's weight-scale block width (unlike
// R4D_GEMM_W4A8_GROUP); another group would make the GEMM's fold read the wrong scale. No LDS: the
// input is already row-major per token.
//
// Not obvious:
//   - The scale is `amax / 448` (E4M3's largest finite value), a dequant multiplier like the
//     checkpoint's `weight_scale_inv`, so the GEMM's fold only multiplies.
//   - A zero block gets scale 1 and quantises to zeros.
//   - The clamp is required: `amax * (1/scale)` can land an ulp above 448, and v_cvt_pk_fp8_f32
//     returns NaN for out-of-range input instead of saturating, so every block's maximum would
//     become NaN.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;

// One block per row; a wave owns one 128-column group, a lane four consecutive columns (one dword
// store, 128 contiguous bytes per wave, two v_cvt_pk_fp8_f32).
//
// ROT applies r4d_fwht_wave<128> in registers between the load and the absmax; the mapping (four
// values per lane) is the one it expects, so no LDS or barrier is needed, and the absmax sees the
// rotated values. The sqrt(128) factor of the unnormalised transform is not removed here:
// <Hw, Ha> = 128 <w, a>, and r4d_layout.cpp folds that into the stored weight scale at convert.
//
// I8 is the same quantiser at int8 codes (quant_act_i8g, had_quant_act_i8g): scale amax/127,
// round-to-nearest-even into -127..127, same row-major layout and per-(row, 128) scales.
template <bool ROT, bool I8 = false>
__global__ __launch_bounds__(kThreads) void quant_act_fp8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t nkb, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += kWaves) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4];
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = c0 + (uint32_t)j;
            v[j] = c < K ? __uint_as_float((uint32_t)x[xrow + c] << 16) : 0.0f;
        }
        // The rotated launcher requires K % 128 == 0: a partial block would rotate real values
        // with padding zeros, and the weight has no matching block.
        if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v, (int)lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = r4d_wave_max(m);
        constexpr float kTop = I8 ? 127.0f : 448.0f;
        const float sc = m > 0.0f ? m * (1.0f / kTop) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
        int pk = 0;
        if constexpr (I8) {
            uint32_t b = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = (int)rintf(fminf(fmaxf(v[j] * inv, -127.0f), 127.0f));
                b |= ((uint32_t)c & 0xFFu) << (8 * j);
            }
            pk = (int)b;
        } else {
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                                 fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk,
                                                 false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                                 fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk,
                                                 true);
        }
        // A ragged last group is written zero-padded; the GEMM's bounds guard keeps the padding out
        // of the accumulators. The caller must own K rounded up to 4 bytes of `q`.
        if (c0 < K) *(uint32_t*)(q + qrow + c0) = (uint32_t)pk;
    }
}

}  // namespace

extern "C" int r4d_quant_act_fp8_group(void) { return (int)kGroup; }

//   a  [M][K]                bf16, `x_ld` elements a row (0 = K)
//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), rounded up to 4
//   s  [M][ceil(K/128)]      f32,  `s_ld` elements a row (0 = ceil(K/128))
//
// `rot` runs the 128-wide Hadamard first and additionally requires K % 128 == 0 (see the kernel).
static int quant_act_fp8_launch(long a, long q, long s, int M, int K,
                                int x_ld, int q_ld, int s_ld, RadStream stream, bool rot,
                                bool i8 = false) {
    int emit_rc = RAD_OK;
    if (!a || !q || !s) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;              // a lane owns four columns; see the kernel
    if (rot && (K % (int)kGroup)) return -5;
    if (!x_ld) x_ld = K;
    if (!q_ld) q_ld = K;
    const int nkb = (K + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;
#define R4D_QA(ROT_, I8_)                                                                     \
    R4D_EMIT(stream, (quant_act_fp8_kernel<ROT_, I8_>), dim3((uint32_t)M), dim3(kThreads), 0,   \
             (const uint16_t*)a, (uint8_t*)q, (float*)s, (uint32_t)K, (uint32_t)nkb,             \
             (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld)
    if (i8) emit_rc = rot ? R4D_QA(true, true) : R4D_QA(false, true);
    else    emit_rc = rot ? R4D_QA(true, false) : R4D_QA(false, false);
#undef R4D_QA
    if (emit_rc != RAD_OK) return -4;
    return (hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_quant_act_fp8(long a, long q, long s, int M, int K,
                                 int x_ld, int q_ld, int s_ld, long stream) {
    return quant_act_fp8_launch(a, q, s, M, K, x_ld, q_ld, s_ld,
                                (RadStream)(uintptr_t)stream, false);
}

extern "C" int r4d_had_quant_act_fp8(long a, long q, long s, int M, int K,
                                     int x_ld, int q_ld, int s_ld, long stream) {
    return quant_act_fp8_launch(a, q, s, M, K, x_ld, q_ld, s_ld,
                                (RadStream)(uintptr_t)stream, true);
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h; see libr4d/r4d_args.h.
/* ================================================================== the fp8 activation quantiser */

static int rad_quant_act_fp8(const RadArgs* a, RadStream stream, bool rot, bool i8 = false) {
    const RadTensor* x  = r4d_opt(a, QF_X);
    const RadTensor* q  = r4d_opt(a, QF_Q);
    const RadTensor* sc = r4d_opt(a, QF_S);
    if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* Rows are dim 0 and everything after is the row, as r4d_row_pitch reads it. A GDN block's
     * out-projection input is [tokens, heads, width] (the scan kernels require that rank) and its
     * K is heads*width; one row per head would put scales on a grid the GEMM does not read.
     * `q` and `sc` are written for the GEMM and stay rank 2. */
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long grp = r4d_quant_act_fp8_group();
    const long long nkb = (K + grp - 1) / grp;
    /* The group is fixed by the checkpoint's scale grid; a different declared group is refused. */
    if (r4d_geti_or(a, "group", grp) != grp) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = quant_act_fp8_launch(r4d_addr(x), r4d_addr(q), r4d_addr(sc), (int)M, (int)K,
                                       (int)xp, (int)qp, (int)sp, stream, rot, i8);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_quant_act_fp8(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, false);
}

/* The rotated form is a separate op, not a parameter: it writes different numbers, so a swapped
 * graph would still run as a different model. The op name pairs with the weight's layout tag in
 * arch/common/rad_block_moe_fp8.h. It pairs only with the `w4a8h` expert plane; any consumer not
 * rotated at convert reads it as noise. */
extern "C" int r4d_rad_had_quant_act_fp8(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, true);
}

/* The int8 pair, for the int8 trunk GEMMs (gemm_i8a8_*). Separate ops for the same reason as the
 * rotated form: different numbers in a buffer of the same shape. */
extern "C" int r4d_rad_quant_act_i8g(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, false, true);
}

extern "C" int r4d_rad_had_quant_act_i8g(const RadArgs* a, RadStream stream) {
    return rad_quant_act_fp8(a, stream, true, true);
}
