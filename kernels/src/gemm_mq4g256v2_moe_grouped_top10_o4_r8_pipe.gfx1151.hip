// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT44 grouped top-10 gate/up O4×R8, pipelined twin.
//
// Bitwise twin of `gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4` (same file
// family, `.._o4_r8_x4.gfx1151.hip`) for the exact Qwen4 shape M = 1280,
// K = 2560 (ten MQ4G256V2 groups per row).  Every output element is computed
// by one wave with the incumbent's per-element arithmetic:
//   * the same MQ4G256V2 decode (fma(sc, (float)nibble, zp), f16 scale/zero
//     halves selected by lane < 16), the same lane -> K mapping (lane t owns
//     the eight values t*8 .. t*8 + 7 of every 256-value group);
//   * group g accumulates into static chain g & 3, groups visited in
//     increasing order, with the incumbent's 8-term expression
//     w0*x0 + ... + w7*x7 as its compiled code evaluates it (see PIPE_ROW);
//   * the fold (acc0 + acc1) + (acc2 + acc3) and the lane-0 tree
//     (hf_lane0_tree_sum2, offsets 16, 8, 4, 2, 1);
//   * dead slots skip accumulation and store +0.0; a subtile without a live
//     slot stores zeros; rows past M read row M - 1 and are never stored; a
//     negative expert id leaves the output untouched; slots past grouped_rows
//     are never stored.
// Only scheduling changes, which cannot change results:
//   * W waves per block (4 * W output rows) share each group's staged X, so
//     X is fetched once per 4 * W rows instead of once per 16;
//   * D groups of weight and X loads are in flight per wave (the incumbent
//     issues one group ahead), from a static register ring;
//   * optionally the staged X is double-buffered in LDS (one barrier per
//     group instead of two) and the barriers are LDS-only;
//   * the ten-group walk is fully unrolled (K = 2560 is a launch contract).
//
// Layouts are the live grouped gate/up contract:
//   expert_weight_ptrs: [E] device pointers to [M, K] MQ4G256V2 rows
//   expert_tile_ids:   [ceil(grouped_rows / 16)] expert ID per 16-slot tile
//   sorted_slot_index: [grouped_rows] flat token/top-k slot, -1 for padding
//   X_src:             [N, K] F32 rotated token rows
//   Y_grouped:         [grouped_rows, M] F32 grouped output
// Grid: [ceil(M / (4 * W)), ceil(grouped_rows / 8), 1], block 32 * W.
// K must be 2560 (the launcher only selects this module for M = 1280,
// K = 2560); M must be a multiple of 4.
// HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Two lane-0 trees in one pass (verbatim from the incumbent): lanes 0-15
// carry a, lanes 16-31 carry b.  After the cross-row step lane i holds
// a[i] + a[i+16] and lane 16+i holds b[16+i] + b[i] (the same sum, IEEE add
// commutes); the in-row DPP steps then reduce both rows with the reference
// offsets, so a ends in lane 0 and b in lane 16 bit-identical to lane 0 of the
// `v += __shfl_down(v, off, 32)` tree (off = 16..1) applied to each.  a is
// returned in lane 0; b is broadcast (readlane 16).
static __device__ __forceinline__ void hf_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}
// x & 0x0F0F0F0F, opaque to the optimizer so each nibble converts with one
// v_cvt_f32_ubyteN instead of being re-canonicalized into v_bfe + cvt.
static __device__ __forceinline__ unsigned int hf_nibble_bytes(unsigned int x) {
    unsigned int r;
    asm("v_and_b32 %0, 0x0f0f0f0f, %1" : "=v"(r) : "v"(x));
    return r;
}

// LDS-only workgroup barrier (the attention_verify_wmma.gfx1151 idiom): keeps
// the in-flight global prefetches in flight, where __syncthreads() would wait
// for every outstanding load (vmcnt(0)) before the barrier.  Each wave waits
// for its own LDS stores (lgkmcnt(0); gfx11 s_waitcnt 0xFC07 leaves vmcnt and
// expcnt open) before it signals: in WGP mode a workgroup's waves span both
// CUs and their LDS queues are not ordered against each other.  X and the
// weights are read-only and Y is written only after the walk, so no global
// ordering is needed.
#define HF_LDS_BARRIER()                                                  \
    do {                                                                  \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  \
        __builtin_amdgcn_s_waitcnt(0xFC07);                               \
        __builtin_amdgcn_s_barrier();                                     \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");  \
    } while (0)

namespace {

constexpr int kSlots = 8;    // R: slots per subtile
constexpr int kGroups = 10;  // K / 256 for K = 2560
constexpr int kGroupBytes = 136;

// Register ring entry: one group's weight words for four rows plus this
// thread's share of the staged X chunk (S float4 of the 8 x 64).
template <int S>
struct Stage {
    unsigned int nh[4];
    unsigned int npk[4];
    float4 st[S];
};

// F bits: 1 = double-buffered LDS, one barrier per group (else one buffer, two
// barriers like the incumbent); 2 = LDS-only barrier (else __syncthreads).
template <bool LdsBar>
__device__ __forceinline__ void hf_barrier() {
    if constexpr (LdsBar) HF_LDS_BARRIER();
    else __syncthreads();
}

template <int W, int D, int F>
struct Walk {
    static constexpr bool kDbuf = F & 1, kLdsBar = F & 2;
    static constexpr int S = kSlots * 64 / (32 * W);
    static_assert(S >= 1 && S * 32 * W == kSlots * 64, "X chunk split");
    static_assert(D >= 1 && D <= kGroups, "prefetch depth");

    __amdgpu_buffer_rsrc_t wrsrc, xrsrc;
    int row_soff[4];
    int stage_soff[S];
    int hoff, boff, stage_voff;
    bool live[kSlots];
    float acc[4][4][kSlots];  // [chain][row][slot]
    Stage<S> ring[D];

    __device__ __forceinline__ void load(Stage<S>& p, int g) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            p.nh[r] = __builtin_amdgcn_raw_buffer_load_b32(
                wrsrc, hoff, row_soff[r] + g * kGroupBytes, 0);
            p.npk[r] = __builtin_amdgcn_raw_buffer_load_b32(
                wrsrc, 8 + boff, row_soff[r] + g * kGroupBytes, 0);
        }
#pragma unroll
        for (int j = 0; j < S; ++j)
            p.st[j] = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128(
                xrsrc, stage_voff, stage_soff[j] + g * 1024, 0));
    }

    template <int G>
    __device__ __forceinline__ void group(float4* __restrict__ xs) {
        constexpr int C = G & 3;
        constexpr int P = G % D;
        float4* __restrict__ xg = xs + (kDbuf ? (G & 1) * (kSlots * 64) : 0);
        unsigned int ch[4], cpk[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { ch[r] = ring[P].nh[r]; cpk[r] = ring[P].npk[r]; }
#pragma unroll
        for (int j = 0; j < S; ++j) xg[j * (32 * W) + threadIdx.x] = ring[P].st[j];
        // Double-buffered: one barrier per group (buffer G & 1 was last read
        // in group G - 2, and every wave passed this group's predecessor
        // barrier after that).  Single buffer: a second barrier after the
        // compute, as in the incumbent.
        hf_barrier<kLdsBar>();
        if constexpr (G + D < kGroups) load(ring[P], G + D);

#define PIPE_DECODE(R) \
        float w##R[8]; \
        { \
            const float sc = __half2float(__ushort_as_half( \
                (unsigned short)(ch[R] & 0xFFFFu))); \
            const float zp = __half2float(__ushort_as_half( \
                (unsigned short)(ch[R] >> 16))); \
            const unsigned int lo = hf_nibble_bytes(cpk[R]); \
            const unsigned int hi = hf_nibble_bytes(cpk[R] >> 4); \
            w##R[0] = __builtin_fmaf(sc, (float)(lo & 0xFFu), zp); \
            w##R[1] = __builtin_fmaf(sc, (float)(hi & 0xFFu), zp); \
            w##R[2] = __builtin_fmaf(sc, (float)((lo >> 8) & 0xFFu), zp); \
            w##R[3] = __builtin_fmaf(sc, (float)((hi >> 8) & 0xFFu), zp); \
            w##R[4] = __builtin_fmaf(sc, (float)((lo >> 16) & 0xFFu), zp); \
            w##R[5] = __builtin_fmaf(sc, (float)((hi >> 16) & 0xFFu), zp); \
            w##R[6] = __builtin_fmaf(sc, (float)(lo >> 24), zp); \
            w##R[7] = __builtin_fmaf(sc, (float)(hi >> 24), zp); \
        }
// The incumbent writes `acc += w0*x0 + w1*x1 + ... + w7*x7` and leaves the
// fusion to the backend's FP contraction (HIP default fast-honor-pragmas).
// Its gfx1151 code object evaluates every instance (all 256 in the module) as
// t = w1*x1; t = fma(w0, x0, t); t = fma(wk, xk, t) for k = 2..7; acc += t
// (DAGCombiner folds the left product of the first add when both products
// have one use).  Spelled out here with explicit fmas so a different
// surrounding schedule cannot flip that choice: the same source text in this
// file contracted some instances as fma(w1, x1, w0*x0) instead.
#define PIPE_ROW(R) do { \
            float t = w##R[1] * x1; \
            t = __builtin_fmaf(w##R[0], x0, t); \
            t = __builtin_fmaf(w##R[2], x2, t); \
            t = __builtin_fmaf(w##R[3], x3, t); \
            t = __builtin_fmaf(w##R[4], x4, t); \
            t = __builtin_fmaf(w##R[5], x5, t); \
            t = __builtin_fmaf(w##R[6], x6, t); \
            t = __builtin_fmaf(w##R[7], x7, t); \
            acc[C][R][slot] += t; \
        } while (0)
        PIPE_DECODE(0) PIPE_DECODE(1) PIPE_DECODE(2) PIPE_DECODE(3)
        const int tid = threadIdx.x & 31;
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            if (!live[slot]) continue;
            const float4 xa = xg[slot * 64 + tid * 2];
            const float4 xb = xg[slot * 64 + tid * 2 + 1];
            const float x0 = xa.x, x1 = xa.y, x2 = xa.z, x3 = xa.w;
            const float x4 = xb.x, x5 = xb.y, x6 = xb.z, x7 = xb.w;
            PIPE_ROW(0); PIPE_ROW(1); PIPE_ROW(2); PIPE_ROW(3);
        }
#undef PIPE_ROW
#undef PIPE_DECODE
        // ALU results carry no memory chain; pin them before the next
        // group's LDS store and barrier.
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot)
            asm volatile("" : "+v"(acc[C][0][slot]), "+v"(acc[C][1][slot]),
                         "+v"(acc[C][2][slot]), "+v"(acc[C][3][slot]));
        if constexpr (!kDbuf) hf_barrier<kLdsBar>();
    }
};

template <int W, int D, int F>
__device__ __forceinline__ void grouped_top10_o4_r8_pipe_body(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int x_row_div,
    int grouped_rows,
    float4* __restrict__ xs
) {
    constexpr int R = kSlots;
    using WalkT = Walk<W, D, F>;
    constexpr int S = WalkT::S;
    // readfirstlane: the compiler cannot see that threadIdx.x >> 5 is
    // wave-uniform, and a VGPR row offset turns every buffer load into a
    // waterfall loop.
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int tid = threadIdx.x & 31;
    const int row0 = (blockIdx.x * W + wave) * 4;
    const int slot_start = blockIdx.y * R;
    // Block-uniform exits only: every wave takes part in the barriers.
    if (slot_start >= grouped_rows) return;

    int flat_raw[R];
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_raw[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }
    const int expert_id = expert_tile_ids[slot_start >> 4];
#pragma unroll
    for (int slot = 0; slot < R; ++slot) asm volatile("" :: "s"(flat_raw[slot]));
    if (expert_id < 0) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    WalkT wk;
    // Rows past M read row M - 1 and are never stored.
    bool row_valid[4];
    constexpr int row_stride = kGroups * kGroupBytes;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        row_valid[r] = row0 + r < M;
        wk.row_soff[r] = (row_valid[r] ? row0 + r : M - 1) * row_stride;
    }

    int x_soff[R];
    bool any_live = false;
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        const int grouped_slot = slot_start + slot;
        const int flat = (grouped_slot < grouped_rows) ? flat_raw[slot] : -1;
        wk.live[slot] = flat >= 0;
        const int x_row = !wk.live[slot] ? 0
            : (x_row_div > 1) ? (flat / x_row_div) : flat;
        x_soff[slot] = x_row * (kGroups * 256) * 4;
        any_live |= wk.live[slot];
    }

#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int slot = 0; slot < R; ++slot) wk.acc[c][r][slot] = 0.0f;

    // A subtile with no live slot accumulates nothing; the epilogue stores
    // the zeros the full walk would have stored.
    if (any_live) {
        wk.wrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
        wk.xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)X_src, 0, 0x7FFFFFFF, 0x31004000);
        wk.boff = tid * 4;
        wk.hoff = (tid < 16) ? 0 : 4;
        // Staging: float4 v = j * 32W + threadIdx.x of the 8 x 64 chunk
        // belongs to slot j * W/2 + (wave >> 1) (wave-uniform), float4
        // (wave & 1) * 32 + tid within it.
        wk.stage_voff = ((wave & 1) * 32 + tid) * 16;
        const int half = wave >> 1;
#pragma unroll
        for (int j = 0; j < S; ++j) {
            int s = x_soff[j * (W / 2)];
#pragma unroll
            for (int q = 1; q < W / 2; ++q)
                s = (half == q) ? x_soff[j * (W / 2) + q] : s;
            wk.stage_soff[j] = s;
        }
#pragma unroll
        for (int d = 0; d < D; ++d) wk.load(wk.ring[d], d);
        wk.template group<0>(xs);
        wk.template group<1>(xs);
        wk.template group<2>(xs);
        wk.template group<3>(xs);
        wk.template group<4>(xs);
        wk.template group<5>(xs);
        wk.template group<6>(xs);
        wk.template group<7>(xs);
        wk.template group<8>(xs);
        wk.template group<9>(xs);
    }

    float sum[4][R];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int slot = 0; slot < R; ++slot)
            sum[r][slot] = (wk.acc[0][r][slot] + wk.acc[1][r][slot])
                + (wk.acc[2][r][slot] + wk.acc[3][r][slot]);
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        hf_lane0_tree_sum2(sum[0][slot], sum[1][slot]);
        hf_lane0_tree_sum2(sum[2][slot], sum[3][slot]);
    }
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < R; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
#pragma unroll
            for (int r = 0; r < 4; ++r)
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] = sum[r][slot];
        }
    }
}

}  // namespace

#define HF_PIPE_KERNEL(NAME, W, D, F) \
__launch_bounds__(32 * (W), 1) \
extern "C" __global__ void NAME( \
    const unsigned long long* __restrict__ expert_weight_ptrs, \
    const int* __restrict__ expert_tile_ids, \
    const int* __restrict__ sorted_slot_index, \
    const float* __restrict__ X_src, \
    float* __restrict__ Y_grouped, \
    int M, \
    int K, \
    int x_row_div, \
    int grouped_rows \
) { \
    (void)K; \
    __shared__ float4 xs[(((F) & 1) ? 2 : 1) * kSlots * 64]; \
    grouped_top10_o4_r8_pipe_body<(W), (D), (F)>(expert_weight_ptrs, expert_tile_ids, \
        sorted_slot_index, X_src, Y_grouped, M, x_row_div, grouped_rows, xs); \
}

HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4_d1_f0, 4, 1, 0)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4_d1_f2, 4, 1, 2)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4_d1_f3, 4, 1, 3)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4_d2_f2, 4, 2, 2)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4_d2_f3, 4, 2, 3)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x8_d1_f3, 8, 1, 3)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x8_d2_f2, 8, 2, 2)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x8_d2_f3, 8, 2, 3)
HF_PIPE_KERNEL(gemm_mq4g256v2_moe_grouped_top10_o4_r8_x8_d3_f3, 8, 3, 3)
