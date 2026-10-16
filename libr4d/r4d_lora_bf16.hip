/* r4d_lora_bf16.hip -- runtime LoRA's two halves, lora_shrink and lora_expand, per-row slot.
 *
 *     lora_shrink   t[m, j]  = sum_k x(m, k) * A[s, j, k]                    s = slot[m]
 *     lora_expand   y[m, n]  = y[m, n] + scale[s, n] * sum_j t[m, j] * B[s, n, j]
 *
 * The core issues the pair right after a targeted trunk GEMM, in the GEMM's own row slice, with
 * `a`/`a_scale` the GEMM's activation operands and `y` its output (docs/OPS.md). A row whose slot
 * is outside [0, S) is a BASE row and both kernels leave it before touching anything: shrink
 * writes no `t`, expand neither reads nor writes `y` -- `y + 0` would already turn a -0 into +0,
 * and the engine's promise is that a base row is bit for bit what the GEMM wrote.
 *
 * V1, SCALAR f32 AND CORRECT FIRST. The adapter term is a rank-R correction (R <= 64 a module)
 * next to an N x K GEMM, so it is small whatever it costs per FLOP; the WMMA form -- a 16-row tile
 * per distinct slot, other slots' rows masked out of the fragment -- is a later speed pass. What
 * this form already guarantees, and the later one must keep:
 *
 *   * NO SPLIT THAT MOVES WITH M. A row's shrink is one wave per column j, lane l summing the k
 *     that are l mod 32 in ascending order, then a fixed xor butterfly; expand is one thread per
 *     (m, n) summing j ascending. Neither depends on the batch a row arrives in, which is what
 *     lets the engine claim "a request alone == the same request in a mixed batch" for the LoRA
 *     term.
 *   * THE GRID IS M AND THE DECLARED PARAMETERS. Which adapters are active is read on the device,
 *     from `slot` and the pools, so a recorded pass replays the same packets whatever the slots
 *     hold.
 *   * NO HOST READS, NO SCRATCH, NO LDS: the reduction is cross-lane only, so a block never waits
 *     on another wave and a base row leaves before any cross-lane op.
 *
 * x(m, k) IS gemm_nt_q's ACTIVATION VALUE: a code times its group's scale (one f32 product), or a
 * bf16/f16/f32 element as it stands, with the scale stream's grid read the way libref's
 * make_scales reads it. The decodes below are the ABI's conversions (rad_plugin.h) restated for
 * the device, bit for bit, so a code decodes here exactly as it does in the oracle. gfx1201 has no
 * bf16 convert; the narrowing is the ABI's RTNE with its quiet-NaN rule. Every multiply-add rounds
 * twice (-ffp-contract=off, RadAddPlugin.cmake), so expand computes libref's bytes and shrink
 * differs from them only by its summation order.
 *
 * Kernels are written against r4d_launch.h so tests/r4d_emu runs this source on the host.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_lora {

/* What `a` holds; a template argument so the decode is chosen once per launch, not per element. */
enum { kBF16 = 0, kF16 = 1, kF32 = 2, kE4M3 = 3, kE5M2 = 4, kI8 = 5 };

struct ShrinkArgs {
    const void*     a;    long long a_rs, a_cs;        /* element strides of `a` */
    const float*    as;   long long s_rs, s_cs;        /* null: no scale stream */
    const int32_t*  slot; long long slot_st;
    const uint16_t* la;   long long la_s0, la_s1, la_s2;
    float*          t;    long long t_rs, t_cs;
    int             K, group, R, S;                  /* K < 2^31: 32-bit k and k / group */
};

struct ExpandArgs {
    const float*    t;    long long t_rs, t_cs;
    const int32_t*  slot; long long slot_st;
    const uint16_t* lb;   long long lb_s0, lb_s1, lb_s2;
    const float*    sc;   long long sc_rs, sc_cs;
    void*           y;    long long y_rs, y_cs;
    long long       N, ntiles;
    int             R, S;
};

/* ------------------------------------------------------- the ABI's conversions, on the device */
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {           /* rad_f32_to_bf16 */
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ inline float f16_to_f32(uint16_t h) {             /* rad_f16_to_f32 */
    const unsigned s = (unsigned)(h >> 15) << 31;
    const unsigned e = ((unsigned)h >> 10) & 0x1fu;
    unsigned m = (unsigned)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((unsigned)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

__device__ inline uint16_t f32_to_f16(float f) {             /* rad_f32_to_f16 */
    const unsigned u = __float_as_uint(f);
    const unsigned s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(s | 0x7c00u);
    if (a < 0x33000000u) return (uint16_t)s;
    const int e = (int)(a >> 23) - 127;
    unsigned m = a & 0x7fffffu;
    if (e < -14) {
        const int sh = -14 - e + 13;
        m |= 0x800000u;
        const unsigned lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        unsigned q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint16_t)(s | q);
    }
    unsigned q = m >> 13, he = (unsigned)(e + 15);
    const unsigned lo = m & 0x1fffu;
    if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
    if (q == 0x400u) { q = 0; ++he; }
    if (he >= 0x1fu) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | (he << 10) | q);
}

__device__ inline float e4m3_to_f32(uint8_t b) {             /* rad_fp8e4m3_to_f32 */
    const unsigned s = (unsigned)(b >> 7) << 31;
    const unsigned e = ((unsigned)b >> 3) & 0xfu;
    unsigned m = (unsigned)b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((unsigned)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

template <int ADT>
__device__ __forceinline__ float ld_act(const void* p, long long i) {
    if (ADT == kBF16) return bf2f(((const uint16_t*)p)[i]);
    if (ADT == kF16)  return f16_to_f32(((const uint16_t*)p)[i]);
    if (ADT == kF32)  return ((const float*)p)[i];
    if (ADT == kE4M3) return e4m3_to_f32(((const uint8_t*)p)[i]);
    if (ADT == kE5M2) return f16_to_f32((uint16_t)((unsigned)((const uint8_t*)p)[i] << 8));
    return (float)((const int8_t*)p)[i];
}

/* The sum of a wave's 32 values, the same on every lane and in a fixed order: an xor butterfly over
 * ds_bpermute. Each step adds the same two values on both partners (a + b == b + a), so the lanes
 * never drift apart and no lane's answer depends on the schedule. */
__device__ __forceinline__ float wave_sum(float v, int lane) {
    for (int off = 16; off >= 1; off >>= 1) {
        const int o = __builtin_amdgcn_ds_bpermute((lane ^ off) << 2, __float_as_int(v));
        v = v + __int_as_float(o);
    }
    return v;
}

/* ---------------------------------------------------------------- the kernels */

/* One block a row, eight waves; wave w owns columns j = w, w + 8, ... The x row is re-read per
 * column from cache rather than staged: K reaches 17408, past what LDS holds as f32. */
template <int ADT>
__global__ __launch_bounds__(256) void lora_shrink_kernel(ShrinkArgs A) {
    const long long m = blockIdx.x;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: no t written */
    const int lane = (int)(threadIdx.x & 31u), wave = (int)(threadIdx.x >> 5);
    const long long x0 = m * A.a_rs, s0 = m * A.s_rs;
    for (int j = wave; j < A.R; j += 8) {
        const uint16_t* arow = A.la + (long long)s * A.la_s0 + (long long)j * A.la_s1;
        float acc = 0.0f;
        for (int k = lane; k < A.K; k += 32) {
            float x = ld_act<ADT>(A.a, x0 + (long long)k * A.a_cs);
            if (A.as) x = x * A.as[s0 + (long long)(k / A.group) * A.s_cs];
            acc += x * bf2f(arow[(long long)k * A.la_s2]);
        }
        acc = wave_sum(acc, lane);
        if (lane == 0) A.t[m * A.t_rs + (long long)j * A.t_cs] = acc;
    }
}

/* One thread an output element over a flat grid of M row-strips of 256 columns. */
template <int YDT>
__global__ __launch_bounds__(256) void lora_expand_kernel(ExpandArgs A) {
    const long long b = blockIdx.x;
    const long long m = b / A.ntiles;
    const long long n = (b - m * A.ntiles) * 256 + threadIdx.x;
    if (n >= A.N) return;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: y not read, not written */
    const float* trow = A.t + m * A.t_rs;
    const uint16_t* brow = A.lb + (long long)s * A.lb_s0 + n * A.lb_s1;
    float acc = 0.0f;
    for (int j = 0; j < A.R; ++j)
        acc += trow[(long long)j * A.t_cs] * bf2f(brow[(long long)j * A.lb_s2]);
    const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc;
    const long long o = m * A.y_rs + n * A.y_cs;
    if (YDT == kBF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f2bf(bf2f(*y) + d);
    } else if (YDT == kF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f32_to_f16(f16_to_f32(*y) + d);
    } else {
        float* y = (float*)A.y + o;
        *y = *y + d;
    }
}

/* ---------------------------------------------------------------- operand reading (host) */

/* `t` read as [rows, cols]: the trailing axes that span `cols` must be one dense run, the leading
 * ones fold to rows, and the row stride is the axis just outside the run -- libref/ref_gemm.h's
 * row_stride, so a rank-3 [tokens, heads, width] activation and a column slice of a fused plane are
 * read the way the oracle reads them. False when no single (rows, row stride, column stride)
 * describes the tensor. */
static bool as_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* rs, int64_t* cs) {
    if (!t || t->rank < 1 || cols <= 0) return false;
    const int r = (int)t->rank;
    int i = r - 1;
    int64_t span = t->shape[i];
    while (span < cols && i > 0) {
        if (t->stride[i - 1] != t->stride[i] * t->shape[i]) return false;
        --i;
        span *= t->shape[i];
    }
    if (span != cols) return false;
    int64_t n = 1;
    for (int d = 0; d < i; ++d) n *= t->shape[d];
    for (int d = 0; d + 1 < i; ++d)
        if (t->shape[d + 1] > 1 && t->stride[d] != t->stride[d + 1] * t->shape[d + 1]) return false;
    *rows = n;
    *rs = i > 0 ? t->stride[i - 1] : cols;
    *cs = t->stride[r - 1];
    return true;
}

/* The per-row slot array: [M], or [M, 1] -- its length and element stride. */
static bool as_vec(const RadTensor* t, int64_t* n, int64_t* st) {
    if (!t) return false;
    if (t->rank == 1) { *n = t->shape[0]; *st = t->stride[0]; return true; }
    int64_t cs = 0;
    return as_rows(t, 1, n, st, &cs);
}

/* A pool [S, rows, cols]: its three strides; anything but rank 3 is read dense. */
static void pool3(const RadTensor* t, int64_t rows, int64_t cols, long long* s0, long long* s1,
                  long long* s2) {
    if (t->rank == 3) { *s0 = t->stride[0]; *s1 = t->stride[1]; *s2 = t->stride[2]; return; }
    *s0 = rows * cols; *s1 = cols; *s2 = 1;
}

/* libref/ref_gemm.h's grid_block: the power of two whose ceiling division gives `blocks`. */
static int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* The dtype parameter is the tensor's own name (rad_dtype_name); a row resolved for one dtype and
 * handed another would compute the wrong decode, so a disagreement is refused, not followed. */
static bool dtype_agrees(const RadArgs* a, uint32_t dt) {
    const char* d = rad_args_gets(a, "dtype");
    return !d || rad_dtype_parse(d) == dt;
}

/* The launches, one per dtype instantiation: the grid is M blocks for shrink and M strips of 256
 * columns for expand, 256 threads each, whatever the slots hold. */
template <int ADT>
static void launch_shrink(const ShrinkArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;                                 /* the host emulator's launch takes none */
    R4D_KLAUNCH(R4D_KNAME(lora_shrink_kernel<ADT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}
template <int YDT>
static void launch_expand(const ExpandArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;
    R4D_KLAUNCH(R4D_KNAME(lora_expand_kernel<YDT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}

}  /* namespace r4d_lora */

extern "C" {

/* operands: a, a_scale?, slot, lora_a, t */
int r4d_rad_lora_shrink(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *X = rad_arg_in(a, 0), *XS = rad_arg_in(a, 1), *SL = rad_arg_in(a, 2);
    const RadTensor *LA = rad_arg_in(a, 3), *T = rad_arg_in(a, 4);
    if (!X || !SL || !LA || !T) return RAD_E_INVAL;
    if (XS && XS->dtype == RAD_DT_INVALID) XS = nullptr;     /* absent, as libref */
    const int64_t K = rad_args_geti_or(a, "K", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (K <= 0 || K > 0x7fffffff || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20))
        return RAD_E_SHAPE;
    int adt;
    switch (X->dtype) {
        case RAD_BF16:   adt = kBF16; break;
        case RAD_F16:    adt = kF16;  break;
        case RAD_F32:    adt = kF32;  break;
        case RAD_F8E4M3: adt = kE4M3; break;
        case RAD_F8E5M2: adt = kE5M2; break;
        case RAD_I8:     adt = kI8;   break;
        default:         return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, X->dtype)) return RAD_E_DTYPE;
    if (adt >= kE4M3 && !XS) return RAD_E_INVAL;     /* a code without its scale has no value */
    if (LA->dtype != RAD_BF16 || T->dtype != RAD_F32 || SL->dtype != RAD_I32) return RAD_E_DTYPE;

    int64_t M = 0, t_rs = 0, t_cs = 0, xr = 0, x_rs = 0, x_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(T, R, &M, &t_rs, &t_cs)) return RAD_E_SHAPE;     /* the step's rows: t's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(X, K, &xr, &x_rs, &x_cs) || xr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LA) < S * R * K) return RAD_E_SHAPE;
    if (M > 0x7fffffff) return RAD_E_SHAPE;

    ShrinkArgs P{};
    P.a = X->data; P.a_rs = x_rs; P.a_cs = x_cs;
    if (XS) {
        /* gemm_nt_q's reading of the stream: [rows, groups] with its own strides, or a flat
         * [rows * groups]; the group is what the stream's extent implies. A grid with fewer rows
         * than the activation is a weight's block grid, never an activation's. */
        if (XS->dtype != RAD_F32) return RAD_E_DTYPE;
        int64_t groups = 0;
        if (XS->rank >= 2) {
            if (XS->shape[XS->rank - 2] < M) return RAD_E_SHAPE;
            groups = XS->shape[XS->rank - 1];
            P.s_rs = XS->stride[XS->rank - 2];
            P.s_cs = XS->stride[XS->rank - 1];
        } else {
            const int64_t nel = rad_tensor_numel(XS);
            if (nel < M) return RAD_E_SHAPE;
            groups = nel / M;
            P.s_rs = groups;
            P.s_cs = 1;
        }
        if (groups <= 0) return RAD_E_SHAPE;
        const int64_t group = grid_block(K, groups);
        const int64_t gp = rad_args_geti_or(a, "group", 0);         /* the grid, as libref */
        if (gp > 0 && (K + gp - 1) / gp != (K + group - 1) / group) return RAD_E_SHAPE;
        P.group = (int)group;
        P.as = (const float*)XS->data;
    }
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.la = (const uint16_t*)LA->data;
    pool3(LA, R, K, &P.la_s0, &P.la_s1, &P.la_s2);
    P.t = (float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.K = (int)K; P.R = (int)R; P.S = (int)S;
    if (!P.as) P.group = (int)K;

    switch (adt) {
        case kBF16: launch_shrink<kBF16>(P, M, stream); break;
        case kF16:  launch_shrink<kF16>(P, M, stream);  break;
        case kF32:  launch_shrink<kF32>(P, M, stream);  break;
        case kE4M3: launch_shrink<kE4M3>(P, M, stream); break;
        case kE5M2: launch_shrink<kE5M2>(P, M, stream); break;
        default:    launch_shrink<kI8>(P, M, stream);   break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* operands: t, slot, lora_b, scale, y (inout) */
int r4d_rad_lora_expand(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    if (!a || a->n_t < 5 || !rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *T = rad_arg_in(a, 0), *SL = rad_arg_in(a, 1), *LB = rad_arg_in(a, 2);
    const RadTensor *SC = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    const int64_t N = rad_args_geti_or(a, "N", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (N <= 0 || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20)) return RAD_E_SHAPE;
    int ydt;
    switch (Y->dtype) {
        case RAD_BF16: ydt = kBF16; break;
        case RAD_F16:  ydt = kF16;  break;
        case RAD_F32:  ydt = kF32;  break;
        default:       return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, Y->dtype)) return RAD_E_DTYPE;
    if (T->dtype != RAD_F32 || SC->dtype != RAD_F32 || LB->dtype != RAD_BF16 ||
        SL->dtype != RAD_I32)
        return RAD_E_DTYPE;

    int64_t M = 0, y_rs = 0, y_cs = 0, tr = 0, t_rs = 0, t_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(Y, N, &M, &y_rs, &y_cs)) return RAD_E_SHAPE;     /* the step's rows: y's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(T, R, &tr, &t_rs, &t_cs) || tr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LB) < S * N * R || rad_tensor_numel(SC) < S * N) return RAD_E_SHAPE;
    const int64_t ntiles = (N + 255) / 256;
    if (M * ntiles > 0x7fffffff) return RAD_E_SHAPE;

    ExpandArgs P{};
    P.t = (const float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.lb = (const uint16_t*)LB->data;
    pool3(LB, N, R, &P.lb_s0, &P.lb_s1, &P.lb_s2);
    P.sc = (const float*)SC->data;
    P.sc_rs = SC->rank >= 2 ? SC->stride[SC->rank - 2] : N;
    P.sc_cs = SC->rank >= 2 ? SC->stride[SC->rank - 1] : 1;
    P.y = Y->data; P.y_rs = y_rs; P.y_cs = y_cs;
    P.N = N; P.ntiles = ntiles; P.R = (int)R; P.S = (int)S;

    switch (ydt) {
        case kBF16: launch_expand<kBF16>(P, M * ntiles, stream); break;
        case kF16:  launch_expand<kF16>(P, M * ntiles, stream);  break;
        default:    launch_expand<kF32>(P, M * ntiles, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
