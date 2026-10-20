// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched MoE top-K + renorm. Companion to moe_topk_renorm_k8 for the
// prefill path.
//
// One workgroup per token. blockIdx.x selects the token row in
// [N × n_exp] input. Same algorithm as the single-token kernel, just
// strided by bid * n_exp / bid * K_TOP.
//
// fuse_softmax == 0 (original behavior, block 256): `probs_in` holds rows that
// were already softmaxed by a separate softmax_f32 launch.
//
// fuse_softmax != 0 (prefill fusion, block = blockDim.x physical lanes):
// `probs_in` holds raw router logits and the kernel computes the softmax
// in-block, then the same top-8 + renorm. The result is byte-identical to the
// unfused softmax_f32 -> mode-0 pair because each virtual 256-thread block of
// the incumbent kernels is emulated stage for stage by the physical lanes
// (virtual thread t handled by lane t % blockDim.x via
// `t = tid; t < N; t += blockDim.x`), with a barrier after every reduction
// stage so the exact tree order, the exact strided accumulation order, and the
// exact (non-index-ascending) tie resolution of the 256-thread tree are
// preserved. No reciprocal, no shuffle/DPP reassociation, no approximation.
// The fused path is correct for any blockDim.x >= 8 (the final clamp/store
// uses lanes 0..K_TOP-1) and <= BLOCK_SIZE; the host launcher narrows the
// valid set to 32/64/128/256 and rejects everything else. The block size is
// launch geometry only (not a kernel argument).
//
// rows_per_block (runtime scalar, appended last): every existing launch passes
// 1 with dynamic shared bytes 0 and blockDim.y == 1, which is the original
// one-row-per-block kernel. The opt-in multirow fused route passes 2 with
// blockDim = (128, 2) and 3136 dynamic shared bytes (n_exp = 256). Row slot
// threadIdx.y == 0 uses the original static arrays; slot 1 uses the same
// arrays laid out in dynamic shared memory, so the reduction bodies below are
// the single original text for every slot. Only `bid` (block row base +
// threadIdx.y) differs per slot; `tid` stays threadIdx.x, so each row sees
// the identical lane assignment, virtual trees, and operation order. Slots
// share only the block barriers. Mode 0 always runs rows_per_block == 1
// (smem_i has no dynamic twin); the multirow route needs an even batch so
// no slot is ever past the last row.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      8

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_k8_batched(
    const float* __restrict__ probs_in,    // [N × n_exp] probs (mode 0) / logits (mode 1)
    int*         __restrict__ topk_idx,    // [N × 8]
    float*       __restrict__ topk_w,      // [N × 8]
    int n_exp,
    int norm_topk,
    int fuse_softmax,
    int rows_per_block
) {
    // Row slot 0: the original static arrays.
    __shared__ float smem_v_slot0[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];     // mode 0 only (rows_per_block == 1)
    __shared__ float reduce_v_slot0[BLOCK_SIZE];
    __shared__ int   reduce_i_slot0[BLOCK_SIZE];
    __shared__ int   picked_idx_slot0[K_TOP];
    __shared__ float picked_w_slot0[K_TOP];
    // Row slot 1 (rows_per_block == 2 only): dynamic shared memory, laid out
    // smem_v[n_exp], reduce_v[256], reduce_i[256], picked_idx[8], picked_w[8]
    // = 4 * (n_exp + 528) bytes.
    extern __shared__ float dyn_slot_smem[];

    // The names the (unchanged) row algorithm below uses.
    float* smem_v     = smem_v_slot0;
    float* reduce_v   = reduce_v_slot0;
    int*   reduce_i   = reduce_i_slot0;
    int*   picked_idx = picked_idx_slot0;
    float* picked_w   = picked_w_slot0;
    if (threadIdx.y != 0) {
        smem_v     = dyn_slot_smem;
        reduce_v   = dyn_slot_smem + n_exp;
        reduce_i   = reinterpret_cast<int*>(dyn_slot_smem + n_exp + BLOCK_SIZE);
        picked_idx = reduce_i + BLOCK_SIZE;
        picked_w   = reinterpret_cast<float*>(picked_idx + K_TOP);
    }

    const int bid = blockIdx.x * rows_per_block + threadIdx.y;
    const int tid = threadIdx.x;
    const float* __restrict__ row_probs = probs_in + (long long)bid * n_exp;
    int*   __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w   = topk_w   + bid * K_TOP;

    if (fuse_softmax) {
        // ---- Fused prefill path: block = blockDim.x lanes, emulating virtual 256 ----
        //
        // Softmax: bit-for-bit the incumbent kernels/src/softmax.hip
        // (softmax_f32 launched with block = min(256, n_exp)). `vblock` is that
        // virtual block size.
        const int vblock = n_exp < 256 ? n_exp : 256;

        // Row max. Each virtual thread folds i = t, t+vblock, ... starting
        // from -1e30f; reduced by the incumbent tree s = vblock/2 .. 1.
        for (int t = tid; t < vblock; t += blockDim.x) {
            float m = -1e30f;
            for (int i = t; i < n_exp; i += vblock) {
                m = fmaxf(m, row_probs[i]);
            }
            reduce_v[t] = m;
        }
        __syncthreads();
        for (int s = vblock >> 1; s > 0; s >>= 1) {
            for (int t = tid; t < s; t += blockDim.x) {
                reduce_v[t] = fmaxf(reduce_v[t], reduce_v[t + s]);
            }
            __syncthreads();
        }
        const float max_val = reduce_v[0];
        __syncthreads();

        // exp(x - max) per element, accumulated per virtual thread in the
        // incumbent strided increasing-index order from 0.0f, then the same
        // tree. The incumbent stores expf() back to the row and re-reads it
        // (an exact fp32 round trip); here it is kept in smem_v.
        for (int t = tid; t < vblock; t += blockDim.x) {
            float sum = 0.0f;
            for (int i = t; i < n_exp; i += vblock) {
                const float e = expf(row_probs[i] - max_val);
                smem_v[i] = e;
                sum += e;
            }
            reduce_v[t] = sum;
        }
        __syncthreads();
        for (int s = vblock >> 1; s > 0; s >>= 1) {
            for (int t = tid; t < s; t += blockDim.x) {
                reduce_v[t] += reduce_v[t + s];
            }
            __syncthreads();
        }
        const float sum_all = reduce_v[0];
        __syncthreads();

        // Direct divide (no reciprocal multiply), as softmax_f32 does.
        for (int i = tid; i < n_exp; i += blockDim.x) {
            smem_v[i] = smem_v[i] / sum_all;
        }
        __syncthreads();

        // Top-8: the incumbent mode-0 algorithm with its 256 virtual threads
        // (always 256 here, independent of n_exp). Local scan is strict `>`
        // over i = t, t+256, ...; the reduction tree s = 128..1 replaces only
        // on strictly greater (`vb > va`). Equal values therefore resolve by
        // tree position, NOT by lowest index (all-equal picks 0,128,64,192,..),
        // and that tree is reproduced exactly.
        for (int k = 0; k < K_TOP; k++) {
            for (int t = tid; t < BLOCK_SIZE; t += blockDim.x) {
                float best_v = -INFINITY;
                int   best_i = -1;
                for (int i = t; i < n_exp; i += BLOCK_SIZE) {
                    const float v = smem_v[i];
                    if (v > best_v) { best_v = v; best_i = i; }
                }
                reduce_v[t] = best_v;
                reduce_i[t] = best_i;
            }
            __syncthreads();
            for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
                for (int t = tid; t < s; t += blockDim.x) {
                    const float va = reduce_v[t];
                    const float vb = reduce_v[t + s];
                    if (vb > va) {
                        reduce_v[t] = vb;
                        reduce_i[t] = reduce_i[t + s];
                    }
                }
                __syncthreads();
            }
            const int   winner   = reduce_i[0];
            const float winner_w = reduce_v[0];
            if (tid == 0) {
                picked_idx[k] = winner;
                picked_w[k]   = winner_w;
                if (winner >= 0 && winner < n_exp) {
                    smem_v[winner] = -INFINITY;
                }
            }
            __syncthreads();
        }

        // Renorm: sequential sum of the eight picks from 0.0f, direct divide.
        if (tid == 0) {
            float s = 0.0f;
            for (int k = 0; k < K_TOP; k++) s += picked_w[k];
            if (norm_topk && s > 0.0f) {
                for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
            }
        }
        __syncthreads();

        // Same NaN/OOB clamp as the mode-0 path.
        if (tid < K_TOP) {
            int   safe_idx = picked_idx[tid];
            float safe_w   = picked_w[tid];
            if (safe_idx < 0 || safe_idx >= n_exp) {
                safe_idx = 0;
                safe_w   = 0.0f;
            }
            row_idx[tid] = safe_idx;
            row_w[tid]   = safe_w;
        }
        return;
    }

    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? row_probs[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float va = reduce_v[tid];
                float vb = reduce_v[tid + s];
                if (vb > va) {
                    reduce_v[tid] = vb;
                    reduce_i[tid] = reduce_i[tid + s];
                }
            }
            __syncthreads();
        }
        const int   winner = reduce_i[0];
        const float winner_w = reduce_v[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k]   = winner_w;
        }
        if (winner >= 0 && winner < n_exp) {
            if ((winner & (BLOCK_SIZE - 1)) == tid) {
                smem_v[winner] = -INFINITY;
            }
        }
        __syncthreads();
    }

    // Renorm via direct division (matches CPU reference; mul-by-reciprocal
    // adds 1 ULP/element which the MoE layer stack amplifies).
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
        }
    }
    __syncthreads();

    // Defense-in-depth NaN/OOB clamp (issue #152 sibling).
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        row_idx[tid] = safe_idx;
        row_w[tid]   = safe_w;
    }
}
