// R4D attention over a paged prefix and the pass's own rows (attn_prefix_self), and the causal
// dense attention a text encoder's prompt pass needs (attn_dense, causal 1, head_dim 128).
//
// THE KERNEL IS THE DENSE ONE (r4d_attn_dense_bf16.hip) WITH A SECOND KEY SOURCE. S^T = K.Q^T, so a
// lane owns one query row and its softmax state, the C fragment of S^T is the B fragment the P.V
// matmul wants, and P never goes through LDS. A workgroup owns one KV head and every query head
// that reads it. What is new:
//
//   THE PREFIX. A sequence's first `used[s]` keys are in the paged cache -- [blocks, kv_heads, 16,
//   2 * HD], K then V in a slot -- and its own rows' keys are dense, beside the queries. The key
//   loop walks the prefix tiles first and then the own tiles; a tile is one source or the other,
//   and its rows are fetched through that source's address rule. Every prefix key is visible.
//
//   CAUSAL OWN KEYS. With `causal` a query sees its own rows up to itself: a text segment of a
//   conditioning sequence, and the text encoder's prompt. A tile wholly below every query of the
//   workgroup pays no test.
//
// bf16 operands (DT16<0>): a diffusion transformer's values carry outliers an f16 operand could
// overflow, and the reference takes P to bf16 for its P.V product too (rounded there, truncated
// here -- see the softmax).
//
// THE INT8 Q.K FORM (`I8`, the op's `k_i8` operand present). The score product runs on
// v_wmma_i32_16x16x16_iu8, which issues in half the cycles of the bf16 one; P.V stays bf16. Every
// code is quant_act_i8g's: a scale per (row, head) of amax / 127 (1 for an all-zero row) and a code
// rint(x * (1 / scale)) clamped to +-127. The own rows' keys arrive quantised that way (k_i8 and
// k_scale, one op ahead of this one), the query is quantised here once a workgroup, and a prefix
// tile is quantised as it is staged -- the prefix is a picture's conditioning, a few hundred rows
// against the thousands of own rows. A score is (q . k) * qs * ks, the integer dot product exact.
// At a DiT's 16384 rows this takes the kernel from 38 to about 32 ms; against the bf16 form the
// picture moves by less than swapping between exact attention implementations moves it.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#include <cmath>
#include <type_traits>

typedef int v2i_ps __attribute__((ext_vector_type(2)));
typedef int v8i_ps __attribute__((ext_vector_type(8)));

/* quant_act_i8g's scale and codes for eight values against `inv` = 1 / scale. */
__device__ __forceinline__ uint32_t ps_q8x4(float a, float b, float c, float d, float inv) {
    auto q = [inv](float x) {
        return (uint32_t)((int)rintf(fminf(fmaxf(x * inv, -127.0f), 127.0f)) & 0xff);
    };
    return q(a) | (q(b) << 8) | (q(c) << 16) | (q(d) << 24);
}
__device__ __forceinline__ float ps_lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float ps_hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xffff0000u); }
__device__ __forceinline__ float ps_amax8(uint4 u) {
    return fmaxf(fmaxf(fmaxf(fabsf(ps_lo(u.x)), fabsf(ps_hi(u.x))),
                       fmaxf(fabsf(ps_lo(u.y)), fabsf(ps_hi(u.y)))),
                 fmaxf(fmaxf(fabsf(ps_lo(u.z)), fabsf(ps_hi(u.z))),
                       fmaxf(fabsf(ps_lo(u.w)), fabsf(ps_hi(u.w)))));
}
__device__ __forceinline__ uint2 ps_q8x8(uint4 u, float inv) {
    return make_uint2(ps_q8x4(ps_lo(u.x), ps_hi(u.x), ps_lo(u.y), ps_hi(u.y), inv),
                      ps_q8x4(ps_lo(u.z), ps_hi(u.z), ps_lo(u.w), ps_hi(u.w), inv));
}

template<int HD, int GQA, int NQB, int TILE, bool I8>
__global__ __launch_bounds__(NQB * GQA * 32)
void r4d_attn_ps_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                        const int8_t* __restrict__ gk8, const float* __restrict__ gks,
                        const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                        const int* __restrict__ cu, int num_seqs, const int* __restrict__ used,
                        const int* __restrict__ bt, int bt_pitch,
                        const uint16_t* __restrict__ kvc, int kv_heads,
                        float scale, int causal, int q_ld, int k_ld, int k8_ld, int ks_ld,
                        int v_ld, int o_ld)
{
    typedef DT16<0> D;
    typedef D::frag frag16;
    typedef typename std::conditional<I8, v2i_ps, frag16>::type qfrag;

    constexpr int NWARPS = NQB * GQA;
    constexpr int NTHR   = NWARPS * 32;
    constexpr int BQ     = NQB * 16;
    constexpr int NKS    = HD / 16;
    constexpr int MT     = TILE / 16;
    constexpr int KSTR   = HD + 8;
    constexpr int VSTR   = TILE + 8;
    constexpr int DG     = HD / 8;
    constexpr int KGRP   = TILE / 8;
    constexpr int KITEMS = TILE * DG;
    constexpr int KP     = (KITEMS + NTHR - 1) / NTHR;
    constexpr int NTR    = KGRP * DG;
    constexpr int LGRP   = NTHR / 8;
    constexpr int VP     = (NTR + LGRP - 1) / LGRP;
    constexpr float GROW = D::GROW;
    constexpr int BS     = 16;                 // the paged block, libr4d's (block_size == 16)
    static_assert(HD % 16 == 0 && TILE % 16 == 0, "geometry");
    // THE INT8 KEY TILE: a key's row is HD bytes, padded by 8 so the sixteen rows a fragment read
    // touches land on distinct bank pairs (34 dwords a row), and written as 8-byte halves because
    // the padding leaves odd rows 8-byte aligned. A prefix tile's key is DG lanes of 8 dims each,
    // which is why those lanes must sit in one half-wave for the row's absolute maximum.
    constexpr int KSTR8  = HD + 8;
    constexpr int K8G    = HD / 16;            // 16-byte pieces of an own key's row
    constexpr int K8P    = (TILE * K8G + NTHR - 1) / NTHR;
    static_assert(!I8 || (DG == 16 && NTHR % 32 == 0), "an int8 prefix key is one half-wave");

    __shared__ __align__(16) uint16_t sK[I8 ? 8 : TILE * KSTR];
    __shared__ __align__(16) int8_t   sK8[I8 ? TILE * KSTR8 : 16];
    __shared__ __align__(16) float    sKs[I8 ? TILE : 4];
    __shared__ __align__(16) uint16_t sV[HD * VSTR];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int g   = warp % GQA;
    const int qbi = warp / GQA;
    const int kvh = blockIdx.y;
    const int hq  = kvh * GQA + g;

    // THE GRID IS FLAT OVER THE SEQUENCES' QUERY BLOCKS: block b is the b-th BQ-row block counting
    // through every sequence in order, so a pass of many equal sequences launches what it needs
    // rather than (longest sequence) x (sequences). A handful of sequences, so a walk.
    int seq = -1, q0 = 0;
    for (int s = 0, b = blockIdx.x; s < num_seqs; ++s) {
        const int nb = (cu[s + 1] - cu[s] + BQ - 1) / BQ;
        if (b < nb) { seq = s; q0 = b * BQ; break; }
        b -= nb;
    }
    if (seq < 0) return;
    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || q0 >= slen) return;
    const int qlast = min(q0 + BQ, slen) - 1;
    const int P = used ? used[seq] : 0;
    const int ke = causal ? qlast + 1 : slen;     // own keys any query here can see

    const float mul = scale * 1.44269504089f;
    const int  qpos = q0 + qbi * 16 + c;
    const bool live = qpos < slen;

    // The query: bf16 fragments pre-multiplied by the softmax scale, or int8 codes against its
    // row's absolute maximum -- a lane holds half the head's dims, the lane 16 away the other half
    // -- with the scale carried in `qs` and applied to the integer score.
    qfrag qf[NKS];
    float qs = mul;
    {
        const int qrow = live ? qpos : slen - 1;
        const uint16_t* qp = gq + (long)(s0 + qrow) * q_ld + (long)hq * HD;
        if constexpr (I8) {
            uint4 qu[NKS];
            float am = 0.0f;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                qu[t] = *(const uint4*)(qp + 16 * t + 8 * h);
                am = fmaxf(am, ps_amax8(qu[t]));
            }
            am = fmaxf(am, swap16(am));
            const float sq = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sq;
            qs = sq * mul;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                const uint2 w = ps_q8x8(qu[t], inv);
                qf[t] = (v2i_ps){ (int)w.x, (int)w.y };
            }
        } else {
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
                auto cv = [&](uint32_t w) {
                    return D::pk(__builtin_bit_cast(float, w << 16) * mul,
                                 __builtin_bit_cast(float, w & 0xffff0000u) * mul);
                };
                qf[t] = D::mk(make_uint2(cv(u.x), cv(u.y)), make_uint2(cv(u.z), cv(u.w)));
            }
        }
    }

    float m_ref = -3.0e38f, soff = 3.0e38f, l_i = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    const int vj = tid & 7, lgi = tid >> 3;
    const uint16_t* kbase = gk + (long)kvh * HD;
    const uint16_t* vbase = gv + (long)kvh * HD;
    const int* btrow = bt ? bt + (long)seq * bt_pitch : nullptr;

    // The K row of key `kk` of the current source; its V row is `vrow`.
    auto krow = [&](bool pre, int kk) -> const uint16_t* {
        if (!pre) return kbase + (long)(s0 + kk) * k_ld;
        const int blk = btrow[kk / BS];
        return kvc + ((long)(blk * kv_heads + kvh) * BS + (kk % BS)) * (2 * HD);
    };
    auto vrow = [&](bool pre, int kk) -> const uint16_t* {
        if (!pre) return vbase + (long)(s0 + kk) * v_ld;
        const int blk = btrow[kk / BS];
        return kvc + ((long)(blk * kv_heads + kvh) * BS + (kk % BS)) * (2 * HD) + HD;
    };

    const int ntp = (P + TILE - 1) / TILE, nts = (ke + TILE - 1) / TILE;
    for (int ti = 0; ti < ntp + nts; ++ti) {
        const bool pre = ti < ntp;
        const int k0 = pre ? ti * TILE : (ti - ntp) * TILE;
        const int klim = pre ? P : ke;
        const bool mask = (k0 + TILE > klim) || (!pre && causal && k0 + TILE - 1 > q0);

        // An own tile in the int8 form is K8P 16-byte pieces a thread and a scale for the first
        // TILE threads; anything else is the bf16 rows (a prefix tile quantised below).
        const bool own8 = I8 && !pre;
        uint4 kr[KP], vr[VP];
        float ksr = 0.0f;
        if (own8) {
            #pragma unroll
            for (int p = 0; p < K8P; ++p) {
                const int i = tid + p * NTHR;
                if ((TILE * K8G) % NTHR == 0 || i < TILE * K8G) {
                    const int key = i / K8G, gg = i % K8G;
                    const int kk = (k0 + key < klim) ? k0 + key : klim - 1;
                    kr[p] = *(const uint4*)(gk8 + (long)(s0 + kk) * k8_ld + (long)kvh * HD + gg * 16);
                }
            }
            if (tid < TILE) {
                const int kk = (k0 + tid < klim) ? k0 + tid : klim - 1;
                ksr = gks[(long)(s0 + kk) * ks_ld + kvh];
            }
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int i = tid + p * NTHR;
                if (KITEMS % NTHR == 0 || i < KITEMS) {
                    const int key = i / DG, gg = i % DG;
                    const int kk = (k0 + key < klim) ? k0 + key : klim - 1;
                    kr[p] = *(const uint4*)(krow(pre, kk) + gg * 8);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                const int kk0 = k0 + kg * 8 + vj;
                const int kk = (kk0 < klim) ? kk0 : klim - 1;
                vr[r] = __builtin_bit_cast(uint4, load_tr_b128(vrow(pre, kk) + dg * 8));
            }
        }
        // A prefix key in the int8 form: its row's maximum across the DG lanes that hold it (one
        // half-wave, so a DPP butterfly), then its eight codes a lane.
        uint2 kq[KP];
        float kqs[KP];
        if (I8 && pre) {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                float am = ps_amax8(kr[p]);
                am = fmaxf(am, r4d_xor_lane<8>(am));
                am = fmaxf(am, r4d_xor_lane<4>(am));
                am = fmaxf(am, r4d_xor_lane<2>(am));
                am = fmaxf(am, r4d_xor_lane<1>(am));
                kqs[p] = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
                kq[p] = ps_q8x8(kr[p], 1.0f / kqs[p]);
            }
        }

        lds_barrier();
        #pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] = vr[r];
            }
        }
        if constexpr (I8) {
            if (!pre) {
                #pragma unroll
                for (int p = 0; p < K8P; ++p) {
                    const int i = tid + p * NTHR;
                    if ((TILE * K8G) % NTHR == 0 || i < TILE * K8G) {
                        int8_t* d = &sK8[(i / K8G) * KSTR8 + (i % K8G) * 16];
                        *(uint2*)d       = make_uint2(kr[p].x, kr[p].y);
                        *(uint2*)(d + 8) = make_uint2(kr[p].z, kr[p].w);
                    }
                }
                if (tid < TILE) sKs[tid] = ksr;
            } else {
                #pragma unroll
                for (int p = 0; p < KP; ++p) {
                    const int i = tid + p * NTHR;
                    if (KITEMS % NTHR == 0 || i < KITEMS) {
                        *(uint2*)&sK8[(i / DG) * KSTR8 + (i % DG) * 8] = kq[p];
                        if (i % DG == 0) sKs[i / DG] = kqs[p];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int i = tid + p * NTHR;
                if (KITEMS % NTHR == 0 || i < KITEMS)
                    *(uint4*)&sK[(i / DG) * KSTR + (i % DG) * 8] = kr[p];
            }
        }
        lds_barrier();

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            // The scores of 16 keys, in log2 units once times `qs` (a no-op 1 for bf16, whose
            // query carries the scale already).
            float sc[8];
            if constexpr (I8) {
                v8i_ps si = (v8i_ps){0, 0, 0, 0, 0, 0, 0, 0};
                const int8_t* kr2 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    v2i_ps kf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint2 u = *(const uint2*)(kr2 + 16 * (t0 + j));
                            kf[j] = (v2i_ps){ (int)u.x, (int)u.y };
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS)
                            si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kf[j], true,
                                                                                 qf[t0 + j], si, false);
                }
                const float4 ka = *(const float4*)&sKs[mt * 16 + 8 * h];
                const float4 kb = *(const float4*)&sKs[mt * 16 + 8 * h + 4];
                const float ks[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = (float)si[e] * ks[e];
            } else {
                v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                const uint16_t* kr2 = &sK[(mt * 16 + c) * KSTR] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 kf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(kr2 + 16 * (t0 + j));
                            kf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) s = D::wmma(kf[j], qf[t0 + j], s);
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = s[e];
            }
            if (mask) {
                const int kq0 = k0 + mt * 16 + 8 * h;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int key = kq0 + e;
                    const bool ok = key < klim && (pre || !causal || key <= qpos);
                    sc[e] = ok ? sc[e] : -INFINITY;
                }
            }

            // `qs` is positive, so the maximum of the scaled scores is the scaled maximum.
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (I8) smax *= qs;
            if (wave_any(smax > m_ref + GROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                soff  = -mnew;
            }

            // P TO BF16 BY TRUNCATION: one v_perm a pair where gfx1201's software round-to-nearest
            // is about six VALU a value, on a path where the WMMA and the VALU share the SIMD
            // (4-7% of the kernel at a DiT's 16384 rows). Its error is the rounding's in size, and
            // its downward bias cancels: the row sum below is taken off the same truncated values.
            uint32_t pw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const float a0 = I8 ? __builtin_fmaf(sc[e], qs, soff) : sc[e] + soff;
                const float a1 = I8 ? __builtin_fmaf(sc[e + 1], qs, soff) : sc[e + 1] + soff;
                const uint32_t lo = __builtin_bit_cast(uint32_t, ex2(a0));
                const uint32_t hi = __builtin_bit_cast(uint32_t, ex2(a1));
                pw[e >> 1] = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
            }
            // The row sum off the packed P, so the denominator is the values P.V used.
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i += __builtin_bit_cast(float, pw[e] << 16) +
                       __builtin_bit_cast(float, pw[e] & 0xffff0000u);
            const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));

            {
                const uint16_t* vr2 = &sV[c * VSTR + mt * 16] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 vf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(vr2 + (t0 + j) * 16 * VSTR);
                            vf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) acc[t0 + j] = D::wmma(vf[j], p16, acc[t0 + j]);
                }
            }
        }
    }

    if (!live) return;
    const float ls = l_i + swap16(l_i);
    const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
    uint16_t* op = go + (long)(s0 + qpos) * o_ld + (long)hq * HD;
    #pragma unroll
    for (int t = 0; t < NKS; ++t) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        *(uint4*)(op + t * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

template<int HD, int GQA, int NQB, int TILE, bool I8>
static int r4d_attn_ps_launch(const void* q, const void* k, const void* k8, const float* ks,
                              const void* v, void* o, const int* cu,
                              const int* used, const int* bt, int bt_pitch, const void* kvc,
                              int num_seqs, int max_q, int n_head_kv, float scale, int causal,
                              int q_ld, int k_ld, int k8_ld, int ks_ld, int v_ld, int o_ld,
                              hipStream_t s)
{
    constexpr int BQ = NQB * 16;
    // Every sequence's blocks: at most total / BQ, plus one partial block a sequence.
    dim3 grid((max_q + BQ - 1) / BQ + num_seqs, n_head_kv, 1);
    hipLaunchKernelGGL((r4d_attn_ps_kernel<HD, GQA, NQB, TILE, I8>), grid, dim3(NQB * GQA * 32), 0, s,
                       (const uint16_t*)q, (const uint16_t*)k, (const int8_t*)k8, ks,
                       (const uint16_t*)v, (uint16_t*)o, cu, num_seqs, used, bt, bt_pitch,
                       (const uint16_t*)kvc, n_head_kv, scale, causal, q_ld, k_ld, k8_ld, ks_ld,
                       v_ld, o_ld);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}

/* head_dim 128 at gqa 1, 2, 4 and 8. `used`/`bt`/`kvc` null is no prefix: the dense form. `max_q`
 * is the query rows of every sequence together, which bounds the flat grid. `k8` non-null is the
 * int8 Q.K form -- the own keys as int8 codes (`k8_ld` bytes a row) with a scale per (row, head)
 * in `ks` (`ks_ld` floats a row) -- at gqa 1, the only ratio a DiT has. */
static int r4d_attn_ps(const void* q, const void* k, const void* k8, const float* ks,
                       const void* v, void* o, const int* cu,
                       const int* used, const int* bt, int bt_pitch, const void* kvc,
                       int num_seqs, int max_q, int n_head, int n_head_kv, int head_dim,
                       float scale, int causal, int q_ld, int k_ld, int k8_ld, int ks_ld,
                       int v_ld, int o_ld, hipStream_t s)
{
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv) return -2;
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return -3;
    if (k8 && ((k8_ld & 15) || ((uintptr_t)k8 & 15) || !ks)) return -3;
    if (num_seqs <= 0 || max_q <= 0) return 0;
    const int gqa = n_head / n_head_kv;
    if (head_dim != 128) return -1;
#define R4D_PS(G, NQB, T, I8)                                                                     \
    return r4d_attn_ps_launch<128, G, NQB, T, I8>(q, k, k8, ks, v, o, cu, used, bt, bt_pitch, kvc, \
                                                  num_seqs, max_q, n_head_kv, scale, causal,      \
                                                  q_ld, k_ld, k8_ld, ks_ld, v_ld, o_ld, s)
    /* SIXTEEN QUERY TILES A WORKGROUP at gqa 1: a workgroup streams its head's whole K and V once,
     * so the rows it serves divide that traffic. At a DiT's 16384 rows a head's K and V are 8 MB
     * and fall out of L2; 128 rows a workgroup ran at 66 TFLOP/s, 256 at 111, 512 at 102. */
    if (k8) {
        if (gqa != 1) return -1;
        R4D_PS(1, 16, 64, true);
    }
    switch (gqa) {
        case 1: R4D_PS(1, 16, 64, false);
        case 2: R4D_PS(2, 4, 64, false);
        case 4: R4D_PS(4, 2, 64, false);
        case 8: R4D_PS(8, 1, 64, false);
        default: return -1;
    }
#undef R4D_PS
}

// ============================================================== the plugin ABI entry points
/* [tokens, heads * head_dim] (or rank 3) at the operand's own row pitch. */
static bool ps_rows(const RadTensor* t, long long head_dim, long long* n_tok, long long* n_head,
                    long long* ld) {
    long long width = 1;
    for (uint32_t i = 1; i < t->rank; ++i) width *= t->shape[i];
    if (t->rank < 2 || width % head_dim) return false;
    *n_head = width / head_dim;
    *ld = r4d_rows_pitch(t, width, n_tok);
    return *ld >= 0;
}

enum { PS_Q = 0, PS_K, PS_V, PS_KV, PS_BT, PS_USED, PS_CU, PS_KDES, PS_VDES, PS_K8, PS_KS,
       PS_OUT };

extern "C" int r4d_rad_attn_prefix_self(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, PS_Q);
    const RadTensor* k  = r4d_opt(a, PS_K);
    const RadTensor* v  = r4d_opt(a, PS_V);
    const RadTensor* kv = r4d_opt(a, PS_KV);
    const RadTensor* bt = r4d_opt(a, PS_BT);
    const RadTensor* su = r4d_opt(a, PS_USED);
    const RadTensor* cu = r4d_opt(a, PS_CU);
    const RadTensor* o  = r4d_opt(a, PS_OUT);
    if (!q || !k || !v || !kv || !bt || !su || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (r4d_opt(a, PS_KDES) || r4d_opt(a, PS_VDES)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (kv->dtype != RAD_BF16 || q->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if (hd != 128 || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    if (kv->rank != 4 || kv->shape[2] != 16 || kv->shape[3] != 2 * hd) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu) || !r4d_contig(su) || cu->rank != 1 || cu->shape[0] < 2)
        return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    if (r4d_numel(su) < num_seqs) return R4D_NO(RAD_E_SHAPE);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!ps_rows(q, hd, &total, &nh, &q_ld) || !ps_rows(k, hd, &tk, &hk, &k_ld) ||
        !ps_rows(v, hd, &tv, &hv, &v_ld) || !ps_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh || hk != kv->shape[1]) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    const long long bt_pitch = bt->rank >= 2 ? bt->shape[1] : r4d_numel(bt) / num_seqs;
    if (bt->rank >= 2 && bt->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long mq = total;
    /* THE INT8 Q.K FORM (`qk` "i8"): the own keys' codes [tokens, kv_heads * head_dim] and their
     * scales [tokens, kv_heads], as quant_act_i8g writes them at group head_dim. */
    const RadTensor* k8 = r4d_opt(a, PS_K8);
    const RadTensor* ks = r4d_opt(a, PS_KS);
    const char* qk = r4d_gets(a, "qk");
    const bool i8 = qk && !std::strcmp(qk, "i8");
    if (qk && !i8 && std::strcmp(qk, "bf16")) return R4D_NO(RAD_E_INVAL);
    if (i8 != (k8 != nullptr) || i8 != (ks != nullptr)) return R4D_NO(RAD_E_INVAL);
    long long k8_ld = 0, ks_ld = 0;
    if (k8) {
        if (k8->dtype != RAD_I8 || ks->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
        long long t8 = 0, h8 = 0, ts = 0;
        if (!ps_rows(k8, hd, &t8, &h8, &k8_ld)) return R4D_NO(RAD_E_STRIDE);
        ks_ld = r4d_rows_pitch(ks, hk, &ts);
        if (ks_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (h8 != hk || t8 < total || ts < total) return R4D_NO(RAD_E_SHAPE);
    }
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const int e = r4d_attn_ps(q->data, k->data, k8 ? k8->data : nullptr,
                              ks ? (const float*)ks->data : nullptr, v->data, o->data,
                              (const int*)cu->data, (const int*)su->data, (const int*)bt->data,
                              (int)bt_pitch, kv->data, num_seqs, (int)mq, (int)nh, (int)hk, (int)hd,
                              scale, (int)r4d_geti_or(a, "causal", 0), (int)q_ld, (int)k_ld,
                              (int)k8_ld, (int)ks_ld, (int)v_ld, (int)o_ld, (hipStream_t)s);
    if (e == -1 || e == -2) return R4D_NO(RAD_E_SHAPE);
    return e == 0 ? RAD_OK : R4D_NO(e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}

/* attn_dense at head_dim 128, causal or not: the same kernel with no prefix. */
extern "C" int r4d_rad_attn_dense_h128(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, 0);
    const RadTensor* k  = r4d_opt(a, 1);
    const RadTensor* v  = r4d_opt(a, 2);
    const RadTensor* cu = r4d_opt(a, 3);
    const RadTensor* o  = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu) || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    if (hd != 128 || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_geti_or(a, "window", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!ps_rows(q, hd, &total, &nh, &q_ld) || !ps_rows(k, hd, &tk, &hk, &k_ld) ||
        !ps_rows(v, hd, &tv, &hv, &v_ld) || !ps_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const int e = r4d_attn_ps(q->data, k->data, nullptr, nullptr, v->data, o->data,
                              (const int*)cu->data, nullptr, nullptr, 0, nullptr, num_seqs,
                              (int)total, (int)nh, (int)hk, (int)hd, scale,
                              (int)r4d_geti_or(a, "causal", 0), (int)q_ld, (int)k_ld, 0, 0,
                              (int)v_ld, (int)o_ld, (hipStream_t)s);
    if (e == -1 || e == -2) return R4D_NO(RAD_E_SHAPE);
    return e == 0 ? RAD_OK : R4D_NO(e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}

// ============================================================== wide heads (attn_dense, h768 / h1152)
// AN IMAGE DECODER'S MID-BLOCK ATTENTION: one head 1152 wide over every latent pixel -- 16384 rows
// at 2048 x 2048. A lane cannot hold a 1152-wide output row (576 accumulator registers), so the
// head is split across the workgroup's eight waves instead of its rows: a wave owns HD/8 of the
// dims for the whole 16-query block.
//
//   S  = Q.K^T is contracted in eight pieces, a wave's dims each, and the partial score tiles are
//        summed through LDS in wave order -- a fixed order, so the sum is the same every run.
//   P  is formed once, by sixteen threads (a query row each), rounded to bf16, and its row sum is
//        taken off the rounded values, the denominator P.V uses.
//   O  += P.V is a wave's own dims: V^T fragments straight from global with the transposing load
//        (no LDS), P^T from LDS.
//
// S^T = K.Q^T as in the dense kernel above, so a partial tile's C layout is (query = lane % 16,
// key = 8 (lane / 16) + e) and both the reduction and the P^T fragment index it the same way.
template<int HD>
__global__ __launch_bounds__(256)
void r4d_attn_wide_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                          const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                          const int* __restrict__ cu, int num_seqs, float scale, int q_ld,
                          int k_ld, int v_ld, int o_ld)
{
    typedef DT16<0> D;
    constexpr int DW = HD / 8;          // dims a wave
    constexpr int NS = DW / 16;         // its 16-wide steps
    constexpr int KT = 64;              // keys a tile
    constexpr int PP = KT + 8;          // P's LDS row pitch, in bf16
    static_assert(HD % 128 == 0, "eight waves of whole 16-wide steps");

    __shared__ float sPart[8 * KT * 16];
    __shared__ float sS[KT * 16];
    __shared__ __align__(16) uint16_t sP[16 * PP];
    __shared__ float sAlpha[16], sL[16];

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int head = blockIdx.y;

    int seq = -1, q0 = 0;
    for (int s = 0, b = blockIdx.x; s < num_seqs; ++s) {
        const int nb = (cu[s + 1] - cu[s] + 15) / 16;
        if (b < nb) { seq = s; q0 = b * 16; break; }
        b -= nb;
    }
    if (seq < 0) return;
    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    if (slen <= 0 || q0 >= slen) return;

    const int d0 = wave * DW;
    const uint16_t* qb = gq + (long)head * HD + d0;
    const uint16_t* kb = gk + (long)head * HD + d0;
    const uint16_t* vb = gv + (long)head * HD + d0;
    const float mul = scale * 1.44269504089f;

    D::frag qf[NS];
    {
        const int qr = min(q0 + c, slen - 1);
        const uint16_t* p = qb + (long)(s0 + qr) * q_ld + 4 * h;
        #pragma unroll
        for (int t = 0; t < NS; ++t) {
            const uint2 lo = *(const uint2*)(p + 16 * t), hi = *(const uint2*)(p + 16 * t + 8);
            qf[t] = D::mk(lo, hi);
        }
    }
    v8f acc[NS];
    #pragma unroll
    for (int t = 0; t < NS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run = -INFINITY, l_run = 0.0f;

    for (int k0 = 0; k0 < slen; k0 += KT) {
        // ---- this wave's share of S^T for the tile's four 16-key blocks
        #pragma unroll
        for (int j = 0; j < KT / 16; ++j) {
            v8f sp = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            const int kr = min(k0 + 16 * j + c, slen - 1);
            const uint16_t* p = kb + (long)(s0 + kr) * k_ld + 4 * h;
            #pragma unroll
            for (int t = 0; t < NS; ++t) {
                const uint2 lo = *(const uint2*)(p + 16 * t), hi = *(const uint2*)(p + 16 * t + 8);
                sp = D::wmma(D::mk(lo, hi), qf[t], sp);
            }
            #pragma unroll
            for (int e = 0; e < 8; ++e) sPart[wave * KT * 16 + (16 * j + 8 * h + e) * 16 + c] = sp[e];
        }
        __syncthreads();
        for (int u = tid; u < KT * 16; u += 256) {
            float t = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) t += sPart[w * KT * 16 + u];
            sS[u] = t;
        }
        __syncthreads();
        // ---- the softmax, a thread a query row
        if (tid < 16) {
            const int nk = min(KT, slen - k0);
            float mx = m_run;
            for (int key = 0; key < nk; ++key) mx = fmaxf(mx, sS[key * 16 + tid] * mul);
            const float alpha = ex2(m_run - mx);
            float lsum = 0.0f;
            for (int key = 0; key < KT; ++key) {
                const uint16_t pb = key < nk ? f32_to_bf16(ex2(sS[key * 16 + tid] * mul - mx)) : 0;
                sP[tid * PP + key] = pb;
                lsum += __builtin_bit_cast(float, (uint32_t)pb << 16);
            }
            l_run = l_run * alpha + lsum;
            m_run = mx;
            sAlpha[tid] = alpha;
        }
        __syncthreads();
        // ---- O^T += V^T . P^T over the wave's dims
        const float al = sAlpha[c];
        #pragma unroll
        for (int t = 0; t < NS; ++t)
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] *= al;
        #pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
            const uint4 pu = *(const uint4*)&sP[c * PP + 16 * ks + 8 * h];
            const D::frag pf = D::mk(make_uint2(pu.x, pu.y), make_uint2(pu.z, pu.w));
            const int kr = min(k0 + 16 * ks + 8 * h + (lane & 7), slen - 1);
            const uint16_t* vp = vb + (long)(s0 + kr) * v_ld + 8 * ((lane >> 3) & 1);
            #pragma unroll
            for (int t = 0; t < NS; ++t) {
                const uint4 u = __builtin_bit_cast(uint4, load_tr_b128(vp + 16 * t));
                acc[t] = D::wmma(D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w)), pf, acc[t]);
            }
        }
    }
    if (tid < 16) sL[tid] = l_run;
    __syncthreads();
    if (q0 + c >= slen) return;
    const float inv = sL[c] > 0.0f ? 1.0f / sL[c] : 0.0f;
    uint16_t* op = go + (long)(s0 + q0 + c) * o_ld + (long)head * HD + d0;
    #pragma unroll
    for (int t = 0; t < NS; ++t) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        *(uint4*)(op + 16 * t + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

/* attn_dense at head_dim 768 or 1152 (an image encoder's and decoder's mid block), one query head a
 * KV head, bidirectional. */
extern "C" int r4d_rad_attn_dense_wide(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, 0);
    const RadTensor* k  = r4d_opt(a, 1);
    const RadTensor* v  = r4d_opt(a, 2);
    const RadTensor* cu = r4d_opt(a, 3);
    const RadTensor* o  = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu) || cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0);
    if ((hd != 768 && hd != 1152) || r4d_geti_or(a, "gqa", 1) != 1 || r4d_geti_or(a, "causal", 0) != 0 ||
        r4d_geti_or(a, "window", 0) != 0)
        return R4D_NO(RAD_E_UNSUPPORTED);
    long long total = 0, nh = 0, q_ld = 0, tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0;
    long long to = 0, ho = 0, o_ld = 0;
    if (!ps_rows(q, hd, &total, &nh, &q_ld) || !ps_rows(k, hd, &tk, &hk, &k_ld) ||
        !ps_rows(v, hd, &tv, &hv, &v_ld) || !ps_rows(o, hd, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != nh || hv != nh || ho != nh) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return R4D_NO(RAD_E_STRIDE);
    const int num_seqs = (int)(cu->shape[0] - 1);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    dim3 grid((unsigned)((total + 15) / 16 + num_seqs), (unsigned)nh);
#define R4D_WIDE(HD)                                                                             \
    hipLaunchKernelGGL((r4d_attn_wide_kernel<HD>), grid, dim3(256), 0, (hipStream_t)s,           \
                       (const uint16_t*)q->data, (const uint16_t*)k->data,                        \
                       (const uint16_t*)v->data, (uint16_t*)o->data, (const int*)cu->data,        \
                       num_seqs, scale, (int)q_ld, (int)k_ld, (int)v_ld, (int)o_ld)
    if (hd == 768) R4D_WIDE(768);
    else           R4D_WIDE(1152);
#undef R4D_WIDE
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
