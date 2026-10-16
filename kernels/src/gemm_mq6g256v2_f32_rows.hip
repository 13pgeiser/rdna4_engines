// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// One warp per weight row, R activation rows per warp. x is pre-rotated F32
// [R,K], y is F32 [R,M]. Each group's header and packed bytes are read and
// decoded once; each activation row keeps the scalar GEMV's own four sums.
template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_group(
    const char* gp, int boff, int tid, const float* x, int K, int base,
    float (&acc)[R]
) {
    const unsigned int hA = *(const unsigned int*)(gp);
    const unsigned int hB = *(const unsigned int*)(gp + 4);
    const unsigned int hs = (tid < 16) ? hA : hB;
    float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
    float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
    const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
    unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2];
    unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5];
    int q0 = b0 & 63;
    int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
    int q2 = (b1 >> 4) | ((b2 & 3) << 4);
    int q3 = b2 >> 2;
    int q4 = b3 & 63;
    int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
    int q6 = (b4 >> 4) | ((b5 & 3) << 4);
    int q7 = b5 >> 2;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float* xr = x + (long long)r * K;
        acc[r] += (sc * (float)q0 + zp) * xr[base]
                + (sc * (float)q1 + zp) * xr[base + 1]
                + (sc * (float)q2 + zp) * xr[base + 2]
                + (sc * (float)q3 + zp) * xr[base + 3]
                + (sc * (float)q4 + zp) * xr[base + 4]
                + (sc * (float)q5 + zp) * xr[base + 5]
                + (sc * (float)q6 + zp) * xr[base + 6]
                + (sc * (float)q7 + zp) * xr[base + 7];
    }
}

template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_row(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row
) {
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;
    float acc0[R], acc1[R], acc2[R], acc3[R];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const int base = g * 256 + tid * 8;
        mq6g256v2_f32_rows_group<R>(gp0,       boff, tid, x, K, base,       acc0);
        mq6g256v2_f32_rows_group<R>(gp0 + 200, boff, tid, x, K, base + 256, acc1);
        mq6g256v2_f32_rows_group<R>(gp0 + 400, boff, tid, x, K, base + 512, acc2);
        mq6g256v2_f32_rows_group<R>(gp0 + 600, boff, tid, x, K, base + 768, acc3);
    }
    const int g = quads << 2;
    const char* gp = row_ptr + g * 200;
    const int base = g * 256 + tid * 8;
    if (tail >= 1)
        mq6g256v2_f32_rows_group<R>(gp,       boff, tid, x, K, base,       acc0);
    if (tail >= 2)
        mq6g256v2_f32_rows_group<R>(gp + 200, boff, tid, x, K, base + 256, acc1);
    if (tail >= 3)
        mq6g256v2_f32_rows_group<R>(gp + 400, boff, tid, x, K, base + 512, acc2);

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[(long long)r * M + row] = acc;
    }
}

template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_body(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    mq6g256v2_f32_rows_row<R>(A, x, y, M, K, row);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r2(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<2>(A, x, y, M, K); }

__launch_bounds__(32, 12)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r3(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<3>(A, x, y, M, K); }

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r4(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<4>(A, x, y, M, K); }

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r5(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<5>(A, x, y, M, K); }

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r6(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<6>(A, x, y, M, K); }

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r7(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<7>(A, x, y, M, K); }

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r8(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<8>(A, x, y, M, K); }

// Four MQ6G256V2 matrices reading the same R rotated x rows (same K) in one
// launch, blocks routed as gemv_mq6g256v2_x4 routes them; each output row is
// the one-matrix rows kernel's (bitwise). Unused matrices have M = 0.
#define MQ6_ROWS_X4(R, OCC)                                                          \
    __launch_bounds__(32, OCC) extern "C" __global__ void gemm_mq6g256v2_f32_rows_x4_r##R( \
        const char* A0, const char* A1, const char* A2, const char* A3, const float* x, \
        float* y0, float* y1, float* y2, float* y3, int M0, int M1, int M2, int M3, int K) { \
        int row = blockIdx.x;                                                        \
        const char* A = A0; float* y = y0; int M = M0;                               \
        if (row >= M0) {                                                             \
            row -= M0; A = A1; y = y1; M = M1;                                       \
            if (row >= M1) {                                                         \
                row -= M1; A = A2; y = y2; M = M2;                                   \
                if (row >= M2) {                                                     \
                    row -= M2; A = A3; y = y3; M = M3;                               \
                    if (row >= M3) return;                                           \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        mq6g256v2_f32_rows_row<R>(A, x, y, M, K, row);                               \
    }
MQ6_ROWS_X4(2, 16)
MQ6_ROWS_X4(3, 12)
MQ6_ROWS_X4(4, 8)
MQ6_ROWS_X4(5, 6)
MQ6_ROWS_X4(6, 6)
MQ6_ROWS_X4(7, 4)
MQ6_ROWS_X4(8, 4)
#undef MQ6_ROWS_X4

// ─────────────────────────────────────────────────────────────────────────
// Wide verify twins (production R = 9..32; capacities 16 and 32, any 1 <= R <= RCAP).
//
// Geometry (design WideTwinGemvSol §2): one wave32 per output column, eight
// columns per workgroup (block 256), every activation row's four stream sums
// live in the owning wave, so each weight group is read and dequantized once
// for the whole launch. X is staged through 8 KiB of LDS (8 rows x one K=256
// group) and shared by the eight column waves; two workgroup barriers per
// (group, 8-row tile). Out-of-range column waves still produce X and take
// every barrier; they only skip weight loads, arithmetic and stores.
//
// Arithmetic: every (row, column) chain is the operation sequence the
// SINGLE-ROW DECODE objects (`gemv_mq6g256v2`, `gemv_mq6g256v2_x4`; hipcc,
// default -ffp-contract, identical on gfx1151 and gfx1201 — receipt in the
// slice notes) execute, spelled out with explicit operations so no
// contraction decision is left to the compiler:
//   w_i = fma(sc, f32(q_i), zp)                   (one rounding)
//   d   = RN(w_1 * x_1); d = fma(w_0, x_0, d); d = fma(w_i, x_i, d), i=2..7
//   acc[g & 3] = acc[g & 3] + d                   (separate add)
//   y   = lane-tree shfl_down 16/8/4/2/1 of (a0 + a1) + (a3 + a2), own + other
// The <= 8 row kernels above are unchanged.
#define MQ6_WIDE_BLOCK 256
#define MQ6_WIDE_COLS 8

static __device__ __forceinline__ int mq6_wide_blocks(int m) {
    return m > 0 ? (m + (MQ6_WIDE_COLS - 1)) / MQ6_WIDE_COLS : 0;
}

// One K=256 group (stream S) for every row of the launch. `gp` is this wave's
// weight group, only dereferenced when `valid` (a column past M still stages X
// and takes every barrier). x must be 16-byte aligned (float4 stage loads) and
// K a multiple of 256.
template<int RCAP, int S>
__device__ __forceinline__ void mq6g256v2_wide_group(
    const char* gp, const bool valid, const int lane, const int wave, const float* x,
    const int K, const int R, const int g, float* lds, float (&acc)[4][RCAP]
) {
#pragma clang fp contract(off)
    float w[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = 0.0f;
    if (valid) {
        const unsigned int hA = *(const unsigned int*)(gp);
        const unsigned int hB = *(const unsigned int*)(gp + 4);
        const unsigned int hs = (lane < 16) ? hA : hB;
        const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp + 8 + lane * 6);
        const unsigned char b0 = d[0], b1 = d[1], b2 = d[2], b3 = d[3], b4 = d[4], b5 = d[5];
        const int q0 = b0 & 63;
        const int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        const int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        const int q3 = b2 >> 2;
        const int q4 = b3 & 63;
        const int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        const int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        const int q7 = b5 >> 2;
        w[0] = __builtin_fmaf(sc, (float)q0, zp);
        w[1] = __builtin_fmaf(sc, (float)q1, zp);
        w[2] = __builtin_fmaf(sc, (float)q2, zp);
        w[3] = __builtin_fmaf(sc, (float)q3, zp);
        w[4] = __builtin_fmaf(sc, (float)q4, zp);
        w[5] = __builtin_fmaf(sc, (float)q5, zp);
        w[6] = __builtin_fmaf(sc, (float)q6, zp);
        w[7] = __builtin_fmaf(sc, (float)q7, zp);
    }
    const int xoff = g * 256 + lane * 8;
    #pragma unroll
    for (int t = 0; t < RCAP / 8; ++t) {
        const int rb = t * 8;
        if (rb >= R) break;                      // uniform: R is a kernel argument
        // Producer wave `wave` stages activation row rb + wave: lane l holds
        // elements g*256 + 8l + j at LDS [wave][j][l] (unchanged bits).
        if (rb + wave < R) {
            const float4* xp = (const float4*)(x + (long long)(rb + wave) * K + xoff);
            const float4 v0 = xp[0];
            const float4 v1 = xp[1];
            float* dst = lds + wave * 256 + lane;
            dst[0 * 32] = v0.x; dst[1 * 32] = v0.y; dst[2 * 32] = v0.z; dst[3 * 32] = v0.w;
            dst[4 * 32] = v1.x; dst[5 * 32] = v1.y; dst[6 * 32] = v1.z; dst[7 * 32] = v1.w;
        }
        __syncthreads();
        if (valid) {
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) {
                const int r = rb + rr;
                if (r < R) {
                    const float* xs = lds + rr * 256 + lane;
                    const float x0 = xs[0 * 32], x1 = xs[1 * 32], x2 = xs[2 * 32], x3 = xs[3 * 32];
                    const float x4 = xs[4 * 32], x5 = xs[5 * 32], x6 = xs[6 * 32], x7 = xs[7 * 32];
                    float dot = w[1] * x1;
                    dot = __builtin_fmaf(w[0], x0, dot);
                    dot = __builtin_fmaf(w[2], x2, dot);
                    dot = __builtin_fmaf(w[3], x3, dot);
                    dot = __builtin_fmaf(w[4], x4, dot);
                    dot = __builtin_fmaf(w[5], x5, dot);
                    dot = __builtin_fmaf(w[6], x6, dot);
                    dot = __builtin_fmaf(w[7], x7, dot);
                    acc[S][r] = acc[S][r] + dot;
                }
            }
        }
        __syncthreads();                         // X tile is rewritten next
    }
}

// One workgroup: eight columns `m_base .. m_base + 7` of one matrix.
template<int RCAP>
__device__ __forceinline__ void mq6g256v2_wide_body(
    const char* __restrict__ A, const float* x, float* __restrict__ y,
    const int M, const int K, const int R, const int m_base
) {
#pragma clang fp contract(off)
    __shared__ float lds[MQ6_WIDE_COLS * 8 * 32];   // [row-in-tile][j][lane], 8 KiB
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int m = m_base + wave;
    const bool valid = m < M;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (valid ? (long long)m * groups_per_row * 200 : 0LL);

    float acc[4][RCAP];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        #pragma unroll
        for (int r = 0; r < RCAP; ++r) acc[s][r] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp = row_ptr + g * 200;
        mq6g256v2_wide_group<RCAP, 0>(gp,       valid, lane, wave, x, K, R, g,     lds, acc);
        mq6g256v2_wide_group<RCAP, 1>(gp + 200, valid, lane, wave, x, K, R, g + 1, lds, acc);
        mq6g256v2_wide_group<RCAP, 2>(gp + 400, valid, lane, wave, x, K, R, g + 2, lds, acc);
        mq6g256v2_wide_group<RCAP, 3>(gp + 600, valid, lane, wave, x, K, R, g + 3, lds, acc);
    }
    {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 200;
        if (tail >= 1) mq6g256v2_wide_group<RCAP, 0>(gp,       valid, lane, wave, x, K, R, g,     lds, acc);
        if (tail >= 2) mq6g256v2_wide_group<RCAP, 1>(gp + 200, valid, lane, wave, x, K, R, g + 1, lds, acc);
        if (tail >= 3) mq6g256v2_wide_group<RCAP, 2>(gp + 400, valid, lane, wave, x, K, R, g + 2, lds, acc);
    }

    if (valid) {
        #pragma unroll
        for (int r = 0; r < RCAP; ++r) {
            if (r < R) {
                float a = (acc[0][r] + acc[1][r]) + (acc[3][r] + acc[2][r]);
                for (int offset = 16; offset > 0; offset >>= 1)
                    a = a + __shfl_down(a, offset);
                if (lane == 0) y[(long long)r * M + m] = a;
            }
        }
    }
}

#define MQ6_ROWS_WIDE(RCAP, WPE)                                                     \
    __launch_bounds__(MQ6_WIDE_BLOCK) __attribute__((amdgpu_waves_per_eu(WPE)))     \
    extern "C" __global__ void gemm_mq6g256v2_f32_rows_wide_r##RCAP(                 \
        const char* A, const float* x, float* y, int M, int K, int R) {              \
        const int m_base = blockIdx.x * MQ6_WIDE_COLS;                               \
        if (R < 1 || m_base >= M) return;                                            \
        mq6g256v2_wide_body<RCAP>(A, x, y, M, K, R, m_base);                         \
    }

// Four matrices, blocks routed like gemm_mq6g256v2_f32_rows_x4 but eight
// columns per block (a block never spans two matrices; M0..M3 are
// matrix-local, an unused matrix has M = 0 and owns no block).
#define MQ6_ROWS_X4_WIDE(RCAP, WPE)                                                  \
    __launch_bounds__(MQ6_WIDE_BLOCK) __attribute__((amdgpu_waves_per_eu(WPE)))     \
    extern "C" __global__ void gemm_mq6g256v2_f32_rows_x4_wide_r##RCAP(              \
        const char* A0, const char* A1, const char* A2, const char* A3, const float* x, \
        float* y0, float* y1, float* y2, float* y3, int M0, int M1, int M2, int M3,  \
        int K, int R) {                                                              \
        int blk = blockIdx.x;                                                        \
        const char* A = A0; float* y = y0; int M = M0;                               \
        const int n0 = mq6_wide_blocks(M0);                                          \
        if (blk >= n0) {                                                             \
            blk -= n0; A = A1; y = y1; M = M1;                                       \
            const int n1 = mq6_wide_blocks(M1);                                      \
            if (blk >= n1) {                                                         \
                blk -= n1; A = A2; y = y2; M = M2;                                   \
                const int n2 = mq6_wide_blocks(M2);                                  \
                if (blk >= n2) {                                                     \
                    blk -= n2; A = A3; y = y3; M = M3;                               \
                    if (blk >= mq6_wide_blocks(M3)) return;                          \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        if (R < 1) return;                                                           \
        mq6g256v2_wide_body<RCAP>(A, x, y, M, K, R, blk * MQ6_WIDE_COLS);            \
    }

MQ6_ROWS_WIDE(16, 12)
MQ6_ROWS_WIDE(32, 8)
MQ6_ROWS_X4_WIDE(16, 12)
MQ6_ROWS_X4_WIDE(32, 8)
#undef MQ6_ROWS_WIDE
#undef MQ6_ROWS_X4_WIDE
