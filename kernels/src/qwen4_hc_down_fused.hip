// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// gfx1201 Qwen4 HC read down projection with the activation in the epilogue:
// ONE launch for what the default prefill chain runs as
//   qwen4_wmma_lds_160_80_32_80_k32_p_s4   (4 K-split partials -> scratch)
//   qwen4_wmma_lds_reduce                  (P0 + P1 + P2 + P3, ascending)
//   hc_activation_fused_f32                (bf16 -> * scale -> bf16 -> SiLU -> bf16)
// (gemm_f16_x_f16_wmma_lds_splitk.hip, tensor_ops.hip).  Y[b, m] = sum_k A[m, k] * X[b, k]
// with M = 320, K = 10240, and the packed BF16 activation lands in `out[b, m]`, the
// operand hyper_read_up_wmma_bf16_gfx1201_t128 reads.  The F32 `low` the chain also
// left behind is never read by the HC read tail, so it is not written.
//
// BYTE-IDENTICAL to that chain, per output:
//   * the four split partials are four independent accumulator chains, each started at
//     zero and fed the 16-wide K substeps of its quarter [z * K / 4, (z + 1) * K / 4)
//     in ascending order, with the same fragment operands (A: row m, X: row b,
//     K elements [8 * (l >> 4), +8) of each substep per lane half) as the split-K tile;
//   * the partials are summed s = P0; s += P1; s += P2; s += P3, which is the reducer's
//     order, in F32 (no contraction: there is no multiply to fuse);
//   * the activation is hc_activation_fused_f32's expression with `bf16_out`.
// One wave owns an 80 (M) x 32 (B) patch (5 x 2 WMMA fragments, 80 accumulator VGPRs)
// and runs the four splits back to back: after split z the finished accumulator is
// added into the running sum (a second 80 VGPRs) instead of leaving through memory.
// The workgroup is the whole M = 320 (four M waves) by BN = 32 * NWN rows of B, so the X
// operand is read from DRAM once and the weight shadow stays L2-resident (every
// workgroup walks the same K quarter at about the same time).
//
// Main loop: K staged KS = 32 per barrier, LDS double buffered (one barrier per stage),
// the global loads for stage s + 2 issued before stage s's WMMA and written to LDS before
// the WMMA of stage s - 1 + 1... see the loop below.
//
// PRECONDITIONS (host-checked): M == 320, K % (4 * KS * 2) == 0, out non-null.
// Grid:  [ceil(B / BN)]    Block: 4 * (BN / 32) * 32

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#if !defined(__gfx1201__)
#error "qwen4_hc_down_fused.hip: gfx1201 only"
#endif

typedef _Float16 __attribute__((ext_vector_type(16))) hd_half16_t;
typedef _Float16 __attribute__((ext_vector_type(8)))  hd_half8_t;
typedef float    __attribute__((ext_vector_type(8)))  hd_float8_t;

// hc_activation_fused_f32 (tensor_ops.hip) with `bf16_out`, same expression, same casts.
#ifdef HD_X_NOACT
static __device__ __forceinline__ unsigned short hd_hc_act(float v, float scale) {
    return static_cast<hip_bfloat16>(v * scale).data;
}
#else
static __device__ __forceinline__ unsigned short hd_hc_act(float v, float scale) {
    float value = static_cast<float>(static_cast<hip_bfloat16>(v));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    return static_cast<hip_bfloat16>(value).data;
}
#endif

constexpr int HD_M  = 320;   // HC low rank: the whole M in one workgroup
constexpr int HD_WP = 5;     // A fragments per wave (80 rows)
constexpr int HD_WQ = 2;     // X fragments per wave (32 rows of B)
constexpr int HD_NWM = 4;    // M waves
constexpr int HD_SPLIT = 4;
constexpr int HD_KFIX = 10240;   // 4 x hidden (2560): the only K the route admits
#ifndef HD_PADROWS
#define HD_PADROWS 2
#endif

// Split LDS-only workgroup barrier (same idiom as gdn_chunk_scan.gfx1201.hip GS_BAR, split
// into signal and wait): the "local" fences keep in-flight global prefetches in flight, and
// each wave waits for its own LDS ops (s_wait_dscnt 0, which the waitcnt inserter cannot
// erase, llvm/llvm-project#220357) before it signals.
extern "C" __device__ void hd_wait_dscnt(unsigned short) __asm("llvm.amdgcn.s.wait.dscnt");
#ifdef HD_X_NOBAR
#define HD_BAR_SIGNAL() do { } while (0)
#define HD_BAR_WAIT()   do { } while (0)
#else
#define HD_BAR_SIGNAL() do { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
                             hd_wait_dscnt(0); __builtin_amdgcn_s_barrier_signal(-1); } while (0)
#define HD_BAR_WAIT()   do { __builtin_amdgcn_s_barrier_wait(-1); \
                             __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); } while (0)
#endif
#ifdef HD_X_NOWMMA
#define HD_WMMA(a, b, c) ((c) + ((float)(a)[0] + (float)(b)[0] + (float)(a)[5] + (float)(b)[7]))
#else
#define HD_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
#endif
#define HD_SCHED() __builtin_amdgcn_sched_barrier(0)

// Staging is 16-byte chunks: chunk c of a tile is row c / (2 * KT), K elements
// [(c % (2 * KT)) * 8, +8) of the stage; its LDS slot is the tile-major [kt][row][16]
// one the fragment reads expect.
template <int NWN, int KS>
static __device__ __forceinline__ void hd_tile(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ X,
    unsigned short* __restrict__ out,
    int K_arg, int B, float scale,
    _Float16* a_lds, _Float16* x_lds
) {
    constexpr int K      = HD_KFIX;   // host guarantees K_arg == HD_KFIX
    (void)K_arg;
    constexpr int BN     = 32 * NWN;
    constexpr int KT     = KS / 16;
    constexpr int CPR    = 2 * KT;            // 16-byte chunks per row per stage
    constexpr int NT     = HD_NWM * NWN * 32;
    constexpr int A_CH   = HD_M * CPR;
    constexpr int X_CH   = BN * CPR;
    constexpr int A_PASS = (A_CH + NT - 1) / NT;   // last pass partial when A_CH % NT != 0
    constexpr int X_PASS = X_CH / NT;
    constexpr int A_PS   = HD_M + HD_PADROWS;  // rows per A plane (padded against bank conflicts)
    constexpr int X_PS   = BN + HD_PADROWS;
    // LDS tile layout: one plane per (kt, kh) 16-byte chunk column, [plane][row][8 halves], so
    // the eight lanes of a fragment read (rows ml..ml+7 of one plane) are 128 contiguous bytes
    // and a stage store's eight lanes (two rows x four planes) are bank-disjoint with the
    // plane pitch = 2 (mod 8) rows (HD_PADROWS = 2).
    constexpr int A_BUF  = CPR * A_PS * 8;   // halves per A buffer
    constexpr int X_BUF  = CPR * X_PS * 8;

    static_assert(KS == 32, "KS must be 32");
    static_assert(X_CH % NT == 0, "X chunks must divide over the block");
    constexpr int A_LAST = A_CH - (A_PASS - 1) * NT;   // threads active in the last A pass

    const int n_block = blockIdx.x * BN;
    if (n_block >= B) return;

    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave_id = tid >> 5;
    const int ml      = lane & 15;
    const int kh      = lane >> 4;

    const int wave_m = wave_id % HD_NWM;
    const int wave_n = wave_id / HD_NWM;
    const int n_wave = n_block + wave_n * (16 * HD_WQ);

    constexpr int KLEN = K / HD_SPLIT;
    constexpr int SPS  = KLEN / KS;          // stages per split, even

    // Per-thread chunk byte offsets of the thread's first chunk; chunk p is NT / CPR rows
    // further (global K pitch, LDS row pitch), folded into the instruction immediates.
    constexpr int ROWS_PER_PASS = NT / CPR;
    const int a_row = tid / CPR, a_part = tid % CPR;
#ifdef HD_X_AHOT
    const unsigned a_off0 = (unsigned)((a_row & 3) * K + a_part * 8) * 2u;
#else
    const unsigned a_off0 = (unsigned)(a_row * K + a_part * 8) * 2u;
#endif
    const unsigned a_dst0 = (unsigned)(a_part * A_PS + a_row) * 16u;
    const int x_row = tid / CPR, x_part = tid % CPR;
#ifdef HD_X_XHOT
    const int xr = (n_block + x_row) & 63;
#else
    const int xr = n_block + x_row;
#endif
    const unsigned x_off0 = (unsigned)(((xr < B) ? xr : (B - 1)) * K + x_part * 8) * 2u;
    const unsigned x_dst0 = (unsigned)(x_part * X_PS + x_row) * 16u;
    const char* a_base = (const char*)A;
    const char* x_base = (const char*)X;
    char* a_lds_b = (char*)a_lds;
    char* x_lds_b = (char*)x_lds;

    hd_half8_t a_reg[A_PASS];
    hd_half8_t x_reg[X_PASS];

    auto load_regs = [&](int k0) {
        const char* ak = a_base + (long long)k0 * 2;
        const char* xk = x_base + (long long)k0 * 2;
#ifndef HD_X_NOLOADA
        #pragma unroll
        for (int p = 0; p < A_PASS; p++) {
            if (p == A_PASS - 1 && A_LAST != NT && tid >= A_LAST) continue;
            a_reg[p] = *(const hd_half8_t*)(ak + a_off0 + (unsigned)(p * ROWS_PER_PASS) * (unsigned)K * 2u);
        }
#endif
#ifndef HD_X_NOLOADX
        #pragma unroll
        for (int p = 0; p < X_PASS; p++) x_reg[p] = *(const hd_half8_t*)(xk + x_off0);
#endif
    };
    auto store_regs = [&](int buf) {
        #pragma unroll
        for (int p = 0; p < A_PASS; p++) {
            if (p == A_PASS - 1 && A_LAST != NT && tid >= A_LAST) continue;
            *(hd_half8_t*)(a_lds_b + buf * (A_BUF * 2) + a_dst0 + p * (ROWS_PER_PASS * 16)) = a_reg[p];
        }
        #pragma unroll
        for (int p = 0; p < X_PASS; p++)
            *(hd_half8_t*)(x_lds_b + buf * (X_BUF * 2) + x_dst0) = x_reg[p];
    };

    // Fragment lane offsets (halves) inside a buffer.
    const int a_frag = (kh * A_PS + wave_m * (16 * HD_WP) + ml) * 8;
    const int x_frag = (kh * X_PS + wave_n * (16 * HD_WQ) + ml) * 8;

    hd_float8_t acc[HD_WP * HD_WQ];
    hd_float8_t run[HD_WP * HD_WQ];
    #pragma unroll
    for (int i = 0; i < HD_WP * HD_WQ; i++) {
        acc[i] = hd_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // Software-pipelined stage loop.  Per stage s (LDS buffer `buf`; stage s + 1 is already
    // complete in the other buffer, stage s + 2 sits in the staging registers):
    //   reads(s, kt1)  ->  WMMA(s, kt0)  ->  signal  ->  WMMA(s, kt1, j = 0)  ->  wait
    //   ->  store stage s + 2 into `buf`, load stage s + 3  ->  reads(s + 1, kt0)
    //   ->  WMMA(s, kt1, j = 1)
    // The signal comes after this wave's last read of `buf` (and its earlier stores), the
    // wait after half the second substep's WMMAs, so barrier skew and the next stage's first
    // fragment reads hide under tensor work.  Fragments for kt0 of the next stage are loop
    // carried in fa0 / fb0.
    hd_half8_t fa0[HD_WP], fb0[HD_WQ];
    auto read_kt0 = [&](int buf) {
        const _Float16* al = a_lds + buf * A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * X_BUF + x_frag;
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb0[j] = *(const hd_half8_t*)(xl + j * 16 * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa0[i] = *(const hd_half8_t*)(al + i * 16 * 8);
    };
    auto stage = [&](int buf, int k_next) {
        const _Float16* al = a_lds + buf * A_BUF + a_frag;
        const _Float16* xl = x_lds + buf * X_BUF + x_frag;
        hd_half8_t fa1[HD_WP], fb1[HD_WQ];
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) fb1[j] = *(const hd_half8_t*)(xl + (2 * X_PS + j * 16) * 8);
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) fa1[i] = *(const hd_half8_t*)(al + (2 * A_PS + i * 16) * 8);
        HD_SCHED();
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) {
            #pragma unroll
            for (int i = 0; i < HD_WP; i++) {
                acc[i * HD_WQ + j] = HD_WMMA(
                    fa0[i], fb0[j], acc[i * HD_WQ + j]);
            }
        }
        HD_SCHED();
        HD_BAR_SIGNAL();
        HD_SCHED();
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            acc[i * HD_WQ + 0] = HD_WMMA(
                fa1[i], fb1[0], acc[i * HD_WQ + 0]);
        }
        HD_SCHED();
        HD_BAR_WAIT();
        HD_SCHED();
#ifdef HD_ORDER_RSL
        read_kt0(buf ^ 1);                   // stage s + 1, substep 0
        HD_SCHED();
#ifndef HD_X_NOSTORE
        store_regs(buf);                     // stage s + 2 -> buf
#endif
#ifndef HD_X_NOLOAD
        load_regs(k_next);                   // stage s + 3 in flight
#endif
        HD_SCHED();
#else
#ifndef HD_X_NOSTORE
        store_regs(buf);                     // stage s + 2 -> buf
#endif
#ifndef HD_X_NOLOAD
        load_regs(k_next);                   // stage s + 3 in flight
#endif
        HD_SCHED();
        read_kt0(buf ^ 1);                   // stage s + 1, substep 0
        HD_SCHED();
#endif
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            acc[i * HD_WQ + 1] = HD_WMMA(
                fa1[i], fb1[1], acc[i * HD_WQ + 1]);
        }
        HD_SCHED();
    };

    const int wid_s = __builtin_amdgcn_readfirstlane(wave_id);

    // Prologue: stages 0 and 1 into buffers 0 and 1, stage 2 in flight in the registers.
    load_regs(0);
    store_regs(0);
    load_regs(KS);
    store_regs(1);
    load_regs(2 * KS);
    HD_BAR_SIGNAL();
    HD_BAR_WAIT();
    read_kt0(0);

    #pragma unroll 1
    for (int z = 0; z < HD_SPLIT; z++) {
        const int base = z * KLEN;
        #pragma unroll 1
        for (int t = 0; t < SPS; t += 2) {
            // Entering: stage (z, t) in buffer 0 with its kt0 fragments loaded.  The next
            // load's K offset (stage t + 3 / t + 4, into the next split at the edge) is clamped
            // so the last prefetches past the end of the last split re-read the final stage
            // instead of branching; their data is never consumed.
#ifdef HD_PF
            {
                // L2 prefetch of the X rows' next 512-byte blocks HD_PF stages ahead.
                const int ip = (z * SPS + t) >> 1;
                const int g  = ip & 3;
                const int kb = ((ip >> 2) * 8 + HD_PF) * KS;
                if (kb < K) {
                    const int r0 = g * (BN / 4) + wid_s * 2;
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        int r = n_block + r0 + u;
                        r = (r < B) ? r : (B - 1);
                        __builtin_amdgcn_s_prefetch_data(X + (long long)r * K + kb, HD_PFLEN);
                    }
                }
            }
#endif
            int k0 = base + (t + 3) * KS;
            k0 = (k0 < K) ? k0 : (K - KS);
            stage(0, k0);
            int k1 = base + (t + 4) * KS;
            k1 = (k1 < K) ? k1 : (K - KS);
            stage(1, k1);
        }
    // Split z done: fold its accumulator into the running sum in the reducer's order.
        if (z == 0) {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = acc[i];
        } else {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = run[i] + acc[i];
        }
        #pragma unroll
        for (int i = 0; i < HD_WP * HD_WQ; i++) {
            acc[i] = hd_float8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
    }

    // Epilogue: lane (ml, kh) holds Y[n_wave + j * 16 + ml][wave_m * 80 + i * 16 + 8 * kh + e].
    #pragma unroll
    for (int j = 0; j < HD_WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        unsigned short* orow = out + (long long)out_b * HD_M;
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            const int m0 = wave_m * (16 * HD_WP) + i * 16 + 8 * kh;
            const hd_float8_t v = run[i * HD_WQ + j];
            unsigned int w[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int lo = hd_hc_act(v[2 * e], scale);
                const unsigned int hi = hd_hc_act(v[2 * e + 1], scale);
                w[e] = lo | (hi << 16);
            }
            *(uint4*)(orow + m0) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}


// Direct variant: both operands' fragments straight from global memory into registers, no LDS,
// no barrier; waves are independent.  Fragment ring of RING k16 steps.
#ifndef HD_RING
#define HD_RING 2
#endif
template <int NWN>
static __device__ __forceinline__ void hd_direct(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ X,
    unsigned short* __restrict__ out,
    int K, int B, float scale
) {
    constexpr int BN = 32 * NWN;
    constexpr int RING = HD_RING;
    const int n_block = blockIdx.x * BN;
    if (n_block >= B) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave_id = tid >> 5;
    const int ml = lane & 15, kh = lane >> 4;
    const int wave_m = wave_id % HD_NWM;
    const int wave_n = wave_id / HD_NWM;
    const int n_wave = n_block + wave_n * (16 * HD_WQ);
    const int KLEN = K / HD_SPLIT;
    const int NSTEP = KLEN / 16;                 // k16 steps per split (multiple of RING)

    const unsigned a_off0 = (unsigned)((wave_m * (16 * HD_WP) + ml) * K + kh * 8) * 2u;
    int xr = n_wave + ml;
    xr = (xr < B) ? xr : (B - 1);
    int xr1 = n_wave + 16 + ml;
    xr1 = (xr1 < B) ? xr1 : (B - 1);
    const unsigned x_off0 = (unsigned)(xr * K + kh * 8) * 2u;
    const unsigned x_off1 = (unsigned)(xr1 * K + kh * 8) * 2u;
    const char* a_base = (const char*)A;
    const char* x_base = (const char*)X;

    hd_float8_t acc[HD_WP * HD_WQ];
    hd_float8_t run[HD_WP * HD_WQ];
    #pragma unroll
    for (int i = 0; i < HD_WP * HD_WQ; i++) acc[i] = hd_float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    hd_half8_t fa[RING][HD_WP], fb[RING][HD_WQ];
    auto ld = [&](int r, int k) {
        const char* ak = a_base + (long long)k * 2;
        const char* xk = x_base + (long long)k * 2;
        #pragma unroll
        for (int i = 0; i < HD_WP; i++)
            fa[r][i] = *(const hd_half8_t*)(ak + a_off0 + (unsigned)(i * 16) * (unsigned)K * 2u);
        fb[r][0] = *(const hd_half8_t*)(xk + x_off0);
        fb[r][1] = *(const hd_half8_t*)(xk + x_off1);
    };
    auto mma = [&](int r) {
        #pragma unroll
        for (int j = 0; j < HD_WQ; j++) {
            #pragma unroll
            for (int i = 0; i < HD_WP; i++)
                acc[i * HD_WQ + j] = HD_WMMA(fa[r][i], fb[r][j], acc[i * HD_WQ + j]);
        }
    };
    // Prologue: steps 0 .. RING-2 in flight.
    #pragma unroll
    for (int r = 0; r < RING - 1; r++) ld(r, r * 16);
    int step = 0;                                   // global k16 step
    const int NTOT = NSTEP * HD_SPLIT;
    #pragma unroll 1
    for (int z = 0; z < HD_SPLIT; z++) {
        #pragma unroll 1
        for (int t = 0; t < NSTEP; t += RING) {
            #pragma unroll
            for (int r = 0; r < RING; r++) {
                int kn = (step + r + RING - 1) * 16;
                kn = (kn < K) ? kn : (K - 16);
                ld((r + RING - 1) % RING, kn);
                HD_SCHED();
                mma(r);
                HD_SCHED();
            }
            step += RING;
        }
        if (z == 0) {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = acc[i];
        } else {
            #pragma unroll
            for (int i = 0; i < HD_WP * HD_WQ; i++) run[i] = run[i] + acc[i];
        }
        #pragma unroll
        for (int i = 0; i < HD_WP * HD_WQ; i++) acc[i] = hd_float8_t{0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    }
    (void)NTOT;
    #pragma unroll
    for (int j = 0; j < HD_WQ; j++) {
        const int out_b = n_wave + j * 16 + ml;
        if (out_b >= B) continue;
        unsigned short* orow = out + (long long)out_b * HD_M;
        #pragma unroll
        for (int i = 0; i < HD_WP; i++) {
            const int m0 = wave_m * (16 * HD_WP) + i * 16 + 8 * kh;
            const hd_float8_t v = run[i * HD_WQ + j];
            unsigned int w[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int lo = hd_hc_act(v[2 * e], scale);
                const unsigned int hi = hd_hc_act(v[2 * e + 1], scale);
                w[e] = lo | (hi << 16);
            }
            *(uint4*)(orow + m0) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}
#define HD_DIRECT_KERNEL(NAME, NWN)                                                    \
    __launch_bounds__(HD_NWM * (NWN) * 32, 1)                                          \
    extern "C" __global__ void NAME(                                                   \
        const _Float16* __restrict__ A, const _Float16* __restrict__ X,                \
        unsigned short* __restrict__ out, int K, int B, float scale) {                 \
        hd_direct<NWN>(A, X, out, K, B, scale);                                        \
    }
HD_DIRECT_KERNEL(qwen4_hc_down_direct_b64, 2)
HD_DIRECT_KERNEL(qwen4_hc_down_direct_b32, 1)

#define HD_KERNEL(NAME, NWN, KS)                                                       \
    __launch_bounds__(HD_NWM * (NWN) * 32, 1)                                          \
    extern "C" __global__ void NAME(                                                   \
        const _Float16* __restrict__ A,                                                \
        const _Float16* __restrict__ X,                                                \
        unsigned short* __restrict__ out,                                              \
        int K, int B, float scale                                                      \
    ) {                                                                                \
        __shared__ __align__(32) _Float16 a_lds[2 * 2 * (KS / 16) * (HD_M + HD_PADROWS) * 8]; \
        __shared__ __align__(32) _Float16 x_lds[2 * 2 * (KS / 16) * (32 * (NWN) + HD_PADROWS) * 8]; \
        hd_tile<NWN, KS>(A, X, out, K, B, scale, a_lds, x_lds);                        \
    }

HD_KERNEL(qwen4_hc_down_fused_b64_k32, 2, 32)
HD_KERNEL(qwen4_hc_down_fused_b32_k32, 1, 32)
HD_KERNEL(qwen4_hc_down_fused_b128_k32, 4, 32)
