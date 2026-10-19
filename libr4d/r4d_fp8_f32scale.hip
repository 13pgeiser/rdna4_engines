/* Canonical E4M3 weights with exact F32 block scales: native FP8 WMMA and F32 scale folding, no
 * checkpoint scale narrowing. The grouped form and the legacy dense form compute one 16x64 output
 * tile a block; the dense decode and prefill forms below them produce the same bytes faster. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_launch.h"
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
struct Meta {
    long ar, asr, yr, wr, sr, wstep[2], sstep[2];
    int T, N, K, experts, topk, sorted_a, tile;
};
__device__ const uint8_t* entry(const void* p, long step, int e) {
    return step ? (const uint8_t*)p + step * e : ((const uint8_t* const*)p)[e];
}
template <bool Grouped>
__global__ void fp8_f32scale(const uint8_t* a, const float* as, const void* w0, const void* s0,
                             const void* w1, const void* s1, const int* sorted, const int* offsets,
                             uint16_t* y, Meta m) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, n0 = blockIdx.x * 64;
    __shared__ uint8_t qa[16 * 136], wb[64 * 136];
    __shared__ int source[16];
    int first = (int)blockIdx.y * (Grouped ? m.tile : 16);
    const int limit = min(first + (Grouped ? m.tile : 16), m.T);
    while (first < limit) {
        int expert = 0, end = limit;
        if constexpr (Grouped) {
            // Tile sorted rows rather than launching every expert, most of which are empty
            // during native MTP decode. Split a tile at each expert boundary.
            if (sorted[first] < 0 || offsets[m.experts] <= first) {
                for (int c = tid; c < 64; c += 128)
                    if (n0 + c < m.N) y[(long)first * m.yr + n0 + c] = 0;
                ++first;
                continue;
            }
            int lo = 0, hi = m.experts;
            while (lo + 1 < hi) {
                const int mid = (lo + hi) / 2;
                if (offsets[mid] <= first)
                    lo = mid;
                else
                    hi = mid;
            }
            expert = lo;
            end = min(limit, offsets[expert + 1]);
            if (end <= first) return;
        }
        const int parity = Grouped ? expert & 1 : 0, e = Grouped ? expert / 2 : 0;
        const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e);
        const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e);
        if (tid < 16) {
            const int r = first + tid;
            const int flat = Grouped && r < end ? sorted[r] : r;
            source[tid] = r < end && flat >= 0 ? (Grouped && !m.sorted_a ? flat / m.topk : r) : -1;
        }
        __syncthreads();
        auto acc = r4d_fp8::acc_zero();
        for (int k0 = 0; k0 < m.K; k0 += 128) {
            for (int j = tid; j < 16 * 32; j += 128) {
                const int r = j / 32, c = (j % 32) * 4, src = source[r];
                uint32_t v = 0;
                if (src >= 0) {
                    const auto* p = a + (long)src * m.ar + k0 + c;
                    v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                        ((uint32_t)p[3] << 24);
                }
                *(uint32_t*)(qa + r * 136 + c) = v;
            }
            for (int j = tid; j < 64 * 32; j += 128) {
                const int r = j / 32, c = (j % 32) * 4;
                const auto* p = w + (long)(n0 + r) * m.wr + k0 + c;
                *(uint32_t*)(wb + r * 136 + c) = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            }
            __syncthreads();
            auto raw = r4d_fp8::acc_zero();
#pragma unroll
            for (int k = 0; k < 128; k += 16)
                raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, 0, k, lane),
                                   r4d_fp8::frag<128>(wb, wave * 16, k, lane), raw);
            const float weight_scale = ws[(long)(n0 / 128) * m.sr + k0 / 128];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int src = source[(lane >> 4) * 8 + j];
                const float scale = src >= 0 ? as[(long)src * m.asr + k0 / 128] * weight_scale : 0;
                acc[j] += raw[j] * scale;
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = first + (lane >> 4) * 8 + j, c = n0 + wave * 16 + (lane & 15);
            if (r < end) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
        }
        __syncthreads();
        first = end;
    }
}
/* The grouped form expert-major: a block per (column span, expert) takes every row of its
 * expert in 16-row tiles, so an expert slice is read from memory once however its rows fall in
 * the sorted order (the row tiles above re-read it once per tile its run spans). Each element
 * keeps the sequence above: per 128 of K in order a fresh chain of eight WMMAs, folded as raw *
 * (activation scale * weight scale). A block walks ColTiles column tiles of 64, each over K in
 * stages of StageK, as one stage sequence with the next stage loaded into registers while the
 * current one computes from LDS. Block row `experts` zeroes the dropped rows past
 * offsets[experts]. */
typedef unsigned f8_u4 __attribute__((ext_vector_type(4)));
template <int StageK>
struct F8Stage {
    static constexpr int KB = StageK / 128;
    f8_u4 a[StageK / 128], w[StageK / 32];
    float wsc[KB], sa;
};
template <int StageK, int ColTiles>
__global__ __launch_bounds__(128) void fp8_f32scale_experts(const uint8_t* a, const float* as,
                                                            const void* w0, const void* s0,
                                                            const void* w1, const void* s1,
                                                            const int* sorted, const int* offsets,
                                                            uint16_t* y, Meta m) {
    using St = F8Stage<StageK>;
    constexpr int KB = St::KB, P = StageK + 8;
    __shared__ __align__(16) uint8_t qa[16 * P], wb[64 * P];
    __shared__ float sa[16 * KB];
    __shared__ int source[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, e = blockIdx.y,
              colb = blockIdx.x * 64 * ColTiles;
    if (e == m.experts) {
        const int r0 = offsets[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * 64 * ColTiles; i += 128) {
            const int c = colb + (int)(i % (64 * ColTiles));
            if (c < m.N) y[(long)(r0 + i / (64 * ColTiles)) * m.yr + c] = 0;
        }
        return;
    }
    const int lo = offsets[e], hi = offsets[e + 1];
    if (lo >= hi) return;
    const int parity = e & 1;
    const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int ksn = (m.K + StageK - 1) / StageK, tiles = min(ColTiles, (m.N - colb + 63) / 64),
              S = tiles * ksn;
    const int rowbase = (lane >> 4) * 8;
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            source[tid] = r < hi && flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = source[rowbase + j] >= 0;
        auto load = [&](int s, St& R) {
            const int col0 = colb + (s / ksn) * 64, k0 = (s % ksn) * StageK;
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16,
                          src = source[r];
                R.a[i] = f8_u4{ 0, 0, 0, 0 };
                if (src >= 0 && k0 + c < m.K)
                    __builtin_memcpy(&R.a[i], a + (long)src * m.ar + k0 + c, 16);
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                R.w[i] = f8_u4{ 0, 0, 0, 0 };
                if (k0 + c < m.K)
                    __builtin_memcpy(&R.w[i], w + (long)(col0 + r) * m.wr + k0 + c, 16);
            }
#pragma unroll
            for (int kb = 0; kb < KB; ++kb)
                R.wsc[kb] =
                    k0 + kb * 128 < m.K ? ws[(long)(col0 / 128) * m.sr + (k0 + kb * 128) / 128] : 0;
            if (tid < 16 * KB) {
                const int src = source[tid / KB], kb = k0 + (tid % KB) * 128;
                R.sa = src >= 0 && kb < m.K ? as[(long)src * m.asr + kb / 128] : 0;
            }
        };
        auto store = [&](const St& R) {
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(qa + r * P + c) = R.a[i];
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(wb + r * P + c) = R.w[i];
            }
            if (tid < 16 * KB) sa[tid] = R.sa;
        };
        auto acc = r4d_fp8::acc_zero();
        St R;
        load(0, R);
        for (int s = 0; s < S; ++s) {
            store(R);
            float wsc[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) wsc[kb] = R.wsc[kb];
            __syncthreads();
            if (s + 1 < S) load(s + 1, R);
            const int k0 = (s % ksn) * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= m.K) break;
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int k = kb * 128; k < (kb + 1) * 128; k += 16)
                    raw = r4d_fp8::mma(r4d_fp8::frag<StageK>(qa, 0, k, lane),
                                       r4d_fp8::frag<StageK>(wb, wave * 16, k, lane), raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scale = rv[j] ? sa[(rowbase + j) * KB + kb] * wsc[kb] : 0;
                    acc[j] += raw[j] * scale;
                }
            }
            if (s % ksn == ksn - 1) {
                const int c = colb + (s / ksn) * 64 + wave * 16 + (lane & 15);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int r = r0 + rowbase + j;
                    if (r < hi) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
                }
                acc = r4d_fp8::acc_zero();
            }
            __syncthreads();
        }
    }
}
// Both forms below keep fp8_f32scale's arithmetic for every output: per 128-K block the same
// eight-mma chain from zero over the same fragments (the bytes at k + 8*(lane/16) of the block's
// row segment, which the old kernel copied through LDS unchanged), then acc += raw * (a_scale *
// w_scale) in ascending block order. A WMMA output depends only on its own A row, B column and
// C, so which rows and columns share a tile does not change it.

// Decode: one wave C adjacent 16-column tiles, S waves of a block on S consecutive K blocks at
// once, the fragments loaded straight from global memory; R row tiles share each weight fragment
// and C column tiles each activation fragment. Each wave scales its own partial products (the
// a_scale x w_scale product and the multiply are the old fold's own two roundings) into LDS; then
// every wave folds a share of the outputs, adding the round's products in block order to the
// output's running sum, which lives in LDS between rounds. The block's activation scales are
// staged in LDS when they fit (`stage`).
template <int R, int C>
__global__ void fp8_f32scale_decode(const uint8_t* a, const float* as, const uint8_t* w,
                                    const float* ws, uint16_t* y, Meta m, int stage) {
    constexpr int SL = R * C * 8;
    R4D_DYN_LDS(float, lds);
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5, S = blockDim.x >> 5, h = lane >> 4;
    const int c0 = blockIdx.x * 16 * C, first = (int)blockIdx.y * R * 16, KB = m.K / 128;
    float* part = lds;
    float* sums = part + S * SL * 32;
    float* scales = sums + SL * 32;
    if (stage)
        for (int i = threadIdx.x; i < R * 16 * KB; i += blockDim.x) {
            const int r = first + i / KB;
            scales[i] = r < m.T ? as[(long)r * m.asr + i % KB] : 0;
        }
    for (int i = threadIdx.x; i < SL * 32; i += blockDim.x) sums[i] = 0;
    const uint8_t* arow[R];
    bool alive[R];
#pragma unroll
    for (int t = 0; t < R; ++t) {
        const int r = first + t * 16 + (lane & 15);
        alive[t] = r < m.T;
        arow[t] = a + (long)(alive[t] ? r : 0) * m.ar + h * 8;
    }
    __syncthreads();
    for (int k0 = 0; k0 < KB; k0 += S) {
        const int kb = k0 + s;
        if (kb < KB) {
            r4d_fp8::i2v b[C][8];
            float weight_scale[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint8_t* wrow =
                    w + (long)(c0 + c * 16 + (lane & 15)) * m.wr + h * 8 + (long)kb * 128;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint2 v = *(const uint2*)(wrow + q * 16);
                    b[c][q][0] = (int)v.x;
                    b[c][q][1] = (int)v.y;
                }
                weight_scale[c] = ws[(long)((c0 + c * 16) / 128) * m.sr + kb];
            }
#pragma unroll
            for (int t = 0; t < R; ++t) {
                r4d_fp8::i2v f[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    uint2 v = make_uint2(0, 0);
                    if (alive[t]) v = *(const uint2*)(arow[t] + (long)kb * 128 + q * 16);
                    f[q][0] = (int)v.x;
                    f[q][1] = (int)v.y;
                }
                float act[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = t * 16 + h * 8 + j, r = first + rr;
                    act[j] =
                        stage ? scales[rr * KB + kb] : (r < m.T ? as[(long)r * m.asr + kb] : 0);
                }
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int q = 0; q < 8; ++q) raw = r4d_fp8::mma(f[q], b[c][q], raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int r = first + t * 16 + h * 8 + j;
                        const float scale = r < m.T ? act[j] * weight_scale[c] : 0;
                        part[((long)s * SL + (t * C + c) * 8 + j) * 32 + lane] = raw[j] * scale;
                    }
                }
            }
        }
        __syncthreads();
        for (int slot = s; slot < SL; slot += S) {
            float acc = sums[slot * 32 + lane];
            for (int q = 0; q < S && k0 + q < KB; ++q)
                acc += part[((long)q * SL + slot) * 32 + lane];
            sums[slot * 32 + lane] = acc;
        }
        __syncthreads();
    }
    for (int slot = s; slot < SL; slot += S) {
        const int t = slot / (C * 8), c = slot / 8 % C, j = slot % 8,
                  r = first + t * 16 + h * 8 + j;
        if (r < m.T)
            y[(long)r * m.yr + c0 + c * 16 + (lane & 15)] = f32_to_bf16(sums[slot * 32 + lane]);
    }
}

// Prefill: the old kernel's 64-column block and LDS layout over R row tiles, so a staged weight
// block serves R*16 rows instead of 16, staged with 16-byte loads and the next K block loaded
// into registers while the current one is multiplied.
template <int R>
__global__ void __launch_bounds__(128)
    fp8_f32scale_rows(const uint8_t* a, const float* as, const uint8_t* w, const float* ws,
                      uint16_t* y, Meta m) {
    constexpr int P = 136, AV = R * 16 * 8 / 128, WV = 64 * 8 / 128;
    __shared__ uint8_t qa[R * 16 * P], wb[64 * P];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, h = lane >> 4,
              n0 = blockIdx.x * 64;
    const int first = (int)blockIdx.y * R * 16, KB = m.K / 128;
    uint4 av[AV], wv[WV];
    auto load = [&](int kb) {
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * 128, r = j / 8, c = (j % 8) * 16;
            av[i] = first + r < m.T
                        ? *(const uint4*)(a + (long)(first + r) * m.ar + (long)kb * 128 + c)
                        : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * 128, r = j / 8, c = (j % 8) * 16;
            wv[i] = *(const uint4*)(w + (long)(n0 + r) * m.wr + (long)kb * 128 + c);
        }
    };
    r4d_fp8::f8v acc[R];
#pragma unroll
    for (int t = 0; t < R; ++t) acc[t] = r4d_fp8::acc_zero();
    load(0);
    for (int kb = 0; kb < KB; ++kb) {
        // The 136-byte pitch keeps 8-byte alignment only, so each 16 bytes go in as two halves.
        auto put = [](uint8_t* p, const uint4& v) {
            ((uint2*)p)[0] = make_uint2(v.x, v.y);
            ((uint2*)p)[1] = make_uint2(v.z, v.w);
        };
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * 128;
            put(qa + (j / 8) * P + (j % 8) * 16, av[i]);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * 128;
            put(wb + (j / 8) * P + (j % 8) * 16, wv[i]);
        }
        __syncthreads();
        if (kb + 1 < KB) load(kb + 1);
        const float weight_scale = ws[(long)(n0 / 128) * m.sr + kb];
#pragma unroll
        for (int t = 0; t < R; ++t) {
            auto raw = r4d_fp8::acc_zero();
#pragma unroll
            for (int k = 0; k < 128; k += 16)
                raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, t * 16, k, lane),
                                   r4d_fp8::frag<128>(wb, wave * 16, k, lane), raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = first + t * 16 + h * 8 + j;
                const float scale = r < m.T ? as[(long)r * m.asr + kb] * weight_scale : 0;
                acc[t][j] += raw[j] * scale;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < R; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = first + t * 16 + h * 8 + j, c = n0 + wave * 16 + (lane & 15);
            if (r < m.T) y[(long)r * m.yr + c] = f32_to_bf16(acc[t][j]);
        }
}

// The dense form. RADIANCE_FP8F32_FORM names "legacy", "rows1"/"rows2"/"rows4" (row tiles a
// block) or "decode", optionally "decodeRxS" or "decodeRxSxC" (R row tiles, S waves a block, C
// column tiles a wave); otherwise the rule at the end of dense_form. The forms are byte-identical;
// the choice is speed only.
struct Form {
    int kind = 0, r = 0, s = 0, c = 1;
};   // kind 0 legacy, 1 decode, 2 rows
constexpr int kLdsCap = 64 * 1024;
long decode_lds(int R, int S, int C, long KB, bool stage) {
    return ((long)S * R * C * 8 * 32 + (long)R * C * 8 * 32 + (stage ? (long)R * 16 * KB : 0)) * 4;
}
// Row tiles a decode block shares a weight fragment over (at most three: four need 192 registers
// and spill), then waves a block: doubled while the grid stays under 1024 waves, the K blocks last
// and the exchange fits the LDS.
Form decode_plan(long N, long K, long T) {
    Form f;
    f.kind = 1;
    const long tiles = (T + 15) / 16, kb = K / 128;
    f.r = (int)(tiles >= 3 ? 3 : tiles);
    const long blocks = (N / 16) * ((tiles + f.r - 1) / f.r);
    f.s = 1;
    while (f.s < 32 && f.s * 2 <= kb && blocks * f.s < 1024 &&
           decode_lds(f.r, f.s * 2, 1, kb, false) <= kLdsCap)
        f.s *= 2;
    return f;
}
Form dense_form(long N, long K, long T) {
    if (const char* e = std::getenv("RADIANCE_FP8F32_FORM")) {
        if (!std::strcmp(e, "legacy")) return Form{};
        if (!std::strncmp(e, "rows", 4)) {
            Form f;
            f.kind = 2;
            f.r = e[4] == '1' ? 1 : e[4] == '2' ? 2 : 4;
            return f;
        }
        if (!std::strncmp(e, "decode", 6)) {
            Form f = decode_plan(N, K, T);
            int r = 0, s = 0, c = 1;
            const int n = std::sscanf(e + 6, "%dx%dx%d", &r, &s, &c);
            if (n >= 2 && r >= 1 && r <= 3 && s >= 1 && s <= 32 && (s & (s - 1)) == 0 &&
                (c == 1 || c == 2 || c == 4) && !(r == 3 && c == 4)) {
                f.r = r;
                f.s = s;
                f.c = c;
            }
            return f;
        }
    }
    // Measured on gfx1201 (r4d_selftest --perfq, cold) at GLM's six shapes: decode1x8 up to 16
    // rows; from 17 to 64 two column tiles a wave (decode1x8x2) lead by 5 to 15 %, except at N 512
    // (too few column tiles, 1x8 stays) and N 4096 over K 1536, where the 16-row tile wins; from 65
    // to 128 rows decode1x8 holds at K 4096 and the row tiles win at K 2048 and below. Among the
    // row tiles (and past 128 rows) the 16-row tile beats the 64-row tile by a quarter; the 32-row
    // tile wins at N 4096.
    const long kb = K / 128;
    Form f;
    const bool rows =
        T > 128 || (T > 64 && K <= 2048) || (T > 16 && T <= 64 && K <= 1536 && N >= 4096);
    if (!rows) {
        f.kind = 1;
        f.r = 1;
        f.c = T > 16 && T <= 64 && N > 512 ? 2 : 1;
        f.s = 8;
        while (f.s > 1 && f.s > kb) f.s /= 2;
        return f;
    }
    f.kind = 2;
    f.r = T > 64 && N >= 4096 ? 2 : 1;
    return f;
}
template <int R>
int emit_rows(RadStream stream, const uint8_t* xa, const float* xsa, const uint8_t* wa,
              const float* wsa, uint16_t* ya, const Meta& m) {
    return R4D_EMIT(stream, fp8_f32scale_rows<R>,
                    dim3(m.N / 64, (unsigned)((m.T + R * 16 - 1) / (R * 16))), dim3(128), 0, xa,
                    xsa, wa, wsa, ya, m);
}
int dense_fast(RadStream stream, const RadTensor& x, const RadTensor& xs, const RadTensor& w,
               const RadTensor& ws, const RadTensor& y, const Meta& m, bool* done) {
    *done = false;
    const Form f = dense_form(m.N, m.K, m.T);
    if (!f.kind || m.wr % 16 || m.ar % 16 ||
        (reinterpret_cast<uintptr_t>(x.data) | reinterpret_cast<uintptr_t>(w.data)) % 16)
        return RAD_OK;
    const uint8_t *xa = (const uint8_t*)x.data, *wa = (const uint8_t*)w.data;
    const float *xsa = (const float*)xs.data, *wsa = (const float*)ws.data;
    uint16_t* ya = (uint16_t*)y.data;
    if (f.kind == 1) {
        const long kb = m.K / 128;
        const int stage = decode_lds(f.r, f.s, f.c, kb, true) <= kLdsCap;
        const long lds = decode_lds(f.r, f.s, f.c, kb, stage);
        if (lds > kLdsCap || m.N % (16 * f.c)) return RAD_OK;
        *done = true;
        const dim3 grid(m.N / (16 * f.c), (unsigned)(((m.T + 15) / 16 + f.r - 1) / f.r)),
            block(32 * f.s);
#define F32S_DECODE(R_, C_)                                                                        \
    if (f.r == R_ && f.c == C_)                                                                    \
        return R4D_EMIT(stream, (fp8_f32scale_decode<R_, C_>), grid, block, (uint32_t)lds, xa,     \
                        xsa, wa, wsa, ya, m, stage);
        F32S_DECODE(1, 1)
        F32S_DECODE(2, 1) F32S_DECODE(3, 1) F32S_DECODE(1, 2) F32S_DECODE(2, 2) F32S_DECODE(3, 2)
            F32S_DECODE(1, 4) F32S_DECODE(2, 4)
#undef F32S_DECODE
                * done = false;
        return RAD_OK;
    }
    *done = true;
    if (f.r == 1) return emit_rows<1>(stream, xa, xsa, wa, wsa, ya, m);
    if (f.r == 2) return emit_rows<2>(stream, xa, xsa, wa, wsa, ya, m);
    return emit_rows<4>(stream, xa, xsa, wa, wsa, ya, m);
}
int body(const RadArgs* a, RadStream stream, bool grouped) {
    if (!a || !a->t || a->n_t < (grouped ? 9 : 5)) return RAD_E_INVAL;
    const auto &x = a->t[0], &xs = a->t[1], &w = a->t[2], &ws = a->t[3], &y = a->t[grouped ? 8 : 4];
    const int64_t N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N < 128 || N % 128 || K < 128 || K % 128 || N > INT_MAX || K > INT_MAX ||
        r4d_geti_or(a, "group", 0) != 128 || r4d_geti_or(a, "pb1", 0) > 0)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_F8E4M3 || xs.dtype != RAD_F32 || w.dtype != RAD_F8E4M3 ||
        ws.dtype != RAD_F32 || y.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (!x.data || !xs.data || !w.data || !ws.data || !y.data || x.rank != 2 || xs.rank != 2 ||
        y.rank != 2 || x.shape[1] != K || xs.shape[0] != x.shape[0] || xs.shape[1] != K / 128 ||
        y.shape[1] != N || x.stride[1] != 1 || xs.stride[1] != 1 || y.stride[1] != 1 ||
        y.shape[0] < 1 || y.shape[0] > INT_MAX)
        return RAD_E_SHAPE;
    const int rank = grouped ? 3 : 2;
    if (w.rank != rank || ws.rank != rank || w.shape[rank - 2] != N || w.shape[rank - 1] != K ||
        ws.shape[rank - 2] != N / 128 || ws.shape[rank - 1] != K / 128 || w.stride[rank - 1] != 1 ||
        ws.stride[rank - 1] != 1)
        return RAD_E_SHAPE;
    Meta m{};
    m.ar = x.stride[0];
    m.asr = xs.stride[0];
    m.yr = y.stride[0];
    m.wr = w.stride[rank - 2];
    m.sr = ws.stride[rank - 2];
    m.T = (int)y.shape[0];
    m.N = N;
    m.K = K;
    const void *w1 = nullptr, *s1 = nullptr;
    const int *sorted = nullptr, *offsets = nullptr;
    if (grouped) {
        m.experts = r4d_geti_or(a, "n_expert", 0);
        m.topk = r4d_geti_or(a, "top_k", 0);
        const char* order = r4d_gets(a, "a_order");
        m.sorted_a = order && !std::strcmp(order, "sorted");
        const auto &wo = a->t[6], &so = a->t[7], &st = a->t[4], &off = a->t[5];
        if (m.experts < 2 || m.topk < 1 || r4d_geti_or(a, "N_odd", N) != N ||
            r4d_geti_or(a, "K_odd", K) != K || wo.dtype != RAD_F8E4M3 || so.dtype != RAD_F32 ||
            wo.rank != 3 || so.rank != 3 || wo.shape[1] != N || wo.shape[2] != K ||
            so.shape[1] != N / 128 || so.shape[2] != K / 128 || wo.stride[1] != m.wr ||
            so.stride[1] != m.sr || wo.stride[2] != 1 || so.stride[2] != 1 || st.dtype != RAD_I32 ||
            off.dtype != RAD_I32 || st.rank != 1 || off.rank != 1 || st.stride[0] != 1 ||
            off.stride[0] != 1 || st.shape[0] < m.T || off.shape[0] < m.experts + 1 || !wo.data ||
            !so.data || !st.data || !off.data)
            return RAD_E_SHAPE;
        m.wstep[0] = w.stride[0];
        m.wstep[1] = wo.stride[0];
        m.sstep[0] = ws.stride[0] * 4;
        m.sstep[1] = so.stride[0] * 4;
        if (w.shape[0] < (m.experts + 1) / 2 || wo.shape[0] < m.experts / 2) return RAD_E_SHAPE;
        w1 = wo.data;
        s1 = so.data;
        sorted = (const int*)st.data;
        offsets = (const int*)off.data;
    } else {
        if (x.shape[0] != m.T) return RAD_E_SHAPE;
        m.wstep[0] = N * m.wr;
        m.sstep[0] = (N / 128) * m.sr * 4;
        bool done = false;
        const int rc = dense_fast(stream, x, xs, w, ws, y, m, &done);
        if (done || rc < 0) return rc;
    }
    m.tile = m.T < 32 ? 1 : m.T < 128 ? 4 : 16;
    if (grouped) {
        /* The expert-major form at every routed-row count unless the environment bounds or
         * refuses it (RADIANCE_FP8F32_DECODE_EXPERTS 0 or 1, ..._MAX_T). Measured at GLM's TP8
         * slices: a long K (gate_up's 4096) wants the 256 stage, 25 to 40 % ahead of 128, and
         * under 128 routed rows four tiles a block on top; at down's K of 256 every variant
         * reads within a few percent of the bandwidth, so it keeps 128 over four tiles. Stage
         * and tiles can be pinned, for the microbenchmark. */
        auto env_long = [](const char* name, long dflt) {
            const char* v = std::getenv(name);
            return v && *v ? std::strtol(v, nullptr, 10) : dflt;
        };
        const char* force = std::getenv("RADIANCE_FP8F32_DECODE_EXPERTS");
        const long max_t = env_long("RADIANCE_FP8F32_DECODE_EXPERTS_MAX_T", -1);
        if (force ? force[0] == '1' : max_t < 0 || m.T <= max_t) {
            const long stage = env_long("RADIANCE_FP8F32_EXPERTS_STAGE_K", K <= 256 ? 128 : 256);
            const long tiles =
                env_long("RADIANCE_FP8F32_EXPERTS_COLTILES", K <= 256 || m.T < 128 ? 4 : 1);
#define F8_EXPERTS(SK, C)                                                                          \
    R4D_EMIT(stream, (fp8_f32scale_experts<SK, C>),                                                \
             dim3((N + 64 * C - 1) / (64 * C), m.experts + 1), dim3(128), 0,                       \
             (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, w1, s1, sorted,       \
             offsets, (uint16_t*)y.data, m)
            if (stage == 256) return tiles == 4 ? F8_EXPERTS(256, 4) : F8_EXPERTS(256, 1);
            return tiles == 4 ? F8_EXPERTS(128, 4) : F8_EXPERTS(128, 1);
#undef F8_EXPERTS
        }
    }
    if (grouped)
        return R4D_EMIT(stream, fp8_f32scale<true>, dim3(N / 64, (m.T + m.tile - 1) / m.tile),
                        dim3(128), 0, (const uint8_t*)x.data, (const float*)xs.data, w.data,
                        ws.data, w1, s1, sorted, offsets, (uint16_t*)y.data, m);
    return R4D_EMIT(stream, fp8_f32scale<false>, dim3(N / 64, (m.T + 15) / 16), dim3(128), 0,
                    (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, w1, s1, sorted,
                    offsets, (uint16_t*)y.data, m);
}
int dense(const RadArgs* a, RadStream stream) { return body(a, stream, false); }
int grouped(const RadArgs* a, RadStream stream) { return body(a, stream, true); }
}
extern "C" int r4d_rad_fp8_f32scale(const RadArgs* a, RadStream stream) { return dense(a, stream); }
extern "C" int r4d_rad_moe_fp8_f32scale(const RadArgs* a, RadStream stream) {
    return grouped(a, stream);
}
extern "C" int r4d_layout_fp8_f32scale(const RadParam*, int, int operand, const RadEncoding* e,
                                       const int* sel, const RadTensor*, int n, RadLayout*) {
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7) return RAD_E_UNSUPPORTED;
    if (!e || !sel || n != 1 || !rad_enc_is(e, "affine") || e->n_planes != 2 ||
        e->plane[0].dtype != RAD_F8E4M3 || e->plane[1].dtype != RAD_F32 ||
        e->plane[1].block[0] != 128 || e->plane[1].block[1] != 128 ||
        sel[0] != (operand == 3 || operand == 7 ? 1 : 0))
        return RAD_E_DTYPE;
    return RAD_E_UNSUPPORTED; // Both canonical planes are read without a relayout.
}
