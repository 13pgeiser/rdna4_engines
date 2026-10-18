/* tests/r4d_emu: SIMT emulator shim that runs libr4d kernel source on the host. Each GPU thread
 * is a fiber on one OS thread; __syncthreads and cross-lane ops are barriers the scheduler
 * enforces, a barrier some threads never reach is reported as a deadlock, and lanes of a wave
 * that meet at different cross-lane ops as a divergence. Test-only. */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(n)

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
typedef void* hipStream_t;
typedef int hipError_t;
enum { hipSuccess = 0 };
inline hipError_t hipGetLastError() { return 0; }
/* one device, with the R9700's 64 KiB of LDS a workgroup, which launchers size dynamic LDS by */
struct hipDeviceProp_t { size_t sharedMemPerBlock; };
inline hipError_t hipGetDevice(int* d) { *d = 0; return hipSuccess; }
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int) {
    *p = hipDeviceProp_t{};
    p->sharedMemPerBlock = 64 * 1024;
    return hipSuccess;
}

struct EmuIdx { unsigned x, y, z; };
EmuIdx& emu_thread_idx();
EmuIdx& emu_block_idx();
#define threadIdx (emu_thread_idx())
#define blockIdx (emu_block_idx())
void __syncthreads();
int emu_bpermute(int addr, int v);
int emu_readlane(int v, int lane);
#define __builtin_amdgcn_ds_bpermute(a, v) emu_bpermute((a), (v))
#define __builtin_amdgcn_readlane(v, l) emu_readlane((v), (l))
unsigned emu_ballot(bool p);
#define __builtin_amdgcn_ballot_w32(p) emu_ballot((p))
/* HIP's __ballot: the wave32 mask, zero-extended */
inline unsigned long long __ballot(int p) { return emu_ballot(p != 0); }
inline int __popcll(unsigned long long v) { return __builtin_popcountll(v); }
/* v_readfirstlane_b32: a cross-lane op is a wave barrier here, so every lane is active, lane 0 first */
#define __builtin_amdgcn_readfirstlane(v) emu_readlane((v), 0)
/* DPP16 row_shl, row_shr and row_xmask with every row and bank written; a source outside the row
 * reads 0 under bound_ctrl, else leaves `old`. Other controls and masks are reported. */
int emu_update_dpp(int old, int src, int ctrl, int row_mask, int bank_mask, bool bound_ctrl);
#define __builtin_amdgcn_update_dpp(o, s, c, rm, bm, bc) emu_update_dpp((o), (s), (c), (rm), (bm), (bc))
/* v_permlanex16_b32: lane L reads the lane of the other row of sixteen that L's selector names */
int emu_permlanex16(int old, int src, unsigned sel_lo, unsigned sel_hi, bool fi, bool bc);
#define __builtin_amdgcn_permlanex16(o, s, lo, hi, fi, bc) emu_permlanex16((o), (s), (lo), (hi), (fi), (bc))
/* v_perm_b32: result byte i is the byte of {a, b} (b low) that selector byte i names; 8-11 a sign
 * byte (bit 15 or 31 of b, then of a), 12 zero, 13 and up 0xff */
inline unsigned emu_perm(unsigned a, unsigned b, unsigned sel) {
    const unsigned long long d = ((unsigned long long)a << 32) | b;
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (8 * i)) & 0xffu;
        unsigned v = s >= 13 ? 0xffu : s == 12 ? 0u : s >= 8 ? ((d >> (16 * (s - 8) + 15)) & 1u) * 0xffu
                                                           : (unsigned)(d >> (8 * s)) & 0xffu;
        r |= v << (8 * i);
    }
    return r;
}
#define __builtin_amdgcn_perm(a, b, s) emu_perm((a), (b), (s))
/* s_barrier is the workgroup barrier. Fibers share one memory and switch only at barriers, so a
 * fence or a scheduling barrier orders nothing more. */
#define __builtin_amdgcn_s_barrier() __syncthreads()
#define __builtin_amdgcn_fence(...) ((void)0)
#define __builtin_amdgcn_sched_barrier(m) ((void)0)
/* LDS atomics: fibers switch only at barriers, so a plain read-modify-write is indivisible */
template <class T>
inline T atomicAdd(T* p, T v) { const T o = *p; *p = o + v; return o; }
/* v_wmma_f32_16x16x16_bf16 (gfx12, wave32): A[M][K] lane M + 16 * (K / 8) element K % 8; B[K][N]
 * lane N + 16 * (K / 8) element K % 8; D[M][N] lane N + 16 * (M / 8) element M % 8 */
typedef short emu_v8s __attribute__((ext_vector_type(8)));
typedef float emu_v8f __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) emu_wmma_bf16((a), (b), (c))
typedef _Float16 emu_v8h __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) emu_wmma_f16((a), (b), (c))
/* global_load_tr_b128: not emulated; reported when reached */
emu_v8s emu_load_tr_b128(const void* p);
#define __builtin_amdgcn_global_load_tr_b128_v8i16(p) emu_load_tr_b128((const void*)(p))
/* v_cvt_pk_fp8_f32 (gfx12, OCP E4M3): converts a and b into the low half of `old` (hi false) or
 * its high half, keeping the other half; round to nearest even. Matches the saturating encoder
 * rad_f32_to_fp8e4m3 on inputs a kernel has already clamped to +-448. */
int emu_cvt_pk_fp8(float a, float b, int old, bool hi);
#define __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi) emu_cvt_pk_fp8((a), (b), (old), (hi))
extern int emu_last_block;     /* threads a block of the last grid had: which kernel ran */

/* Blocks run one after another on one OS thread, so a fence orders nothing that is not already in
 * order and an atomic is a plain read-modify-write. Enough for the split-K arrival pattern
 * (r4d_lora_bf16.hip): the last block of a group in launch order is the one that reduces. */
inline void __threadfence() {}
#define __HIP_MEMORY_SCOPE_AGENT 3
#define __hip_atomic_fetch_add(p, v, order, scope) emu_atomic_fetch_add((p), (v))
template <typename T> inline T emu_atomic_fetch_add(T* p, T v) { const T o = *p; *p = o + v; return o; }
/* The device allocations a unit makes for itself (counter blocks): host memory here. */
inline hipError_t hipMalloc(void** p, size_t n) { *p = std::calloc(n ? n : 1, 1); return *p ? 0 : 2; }
inline hipError_t hipMemsetAsync(void* p, int v, size_t n, hipStream_t) { std::memset(p, v, n); return 0; }

struct alignas(8)  uint2  { unsigned x, y; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(16) int4   { int x, y, z, w; };
struct alignas(16) float4 { float x, y, z, w; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return uint2{a, b}; }
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }
inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }   /* v_rsq_f32 is within an ulp of it */
inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int __float_as_int(float f) { int u; std::memcpy(&u, &f, 4); return u; }

/* one block at a time, every thread a fiber; `dyn_lds` bytes of dynamic LDS */
void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx, size_t dyn_lds = 0);
template <typename K, typename A>
struct EmuCall { K k; A a; static void run(void* p) { auto* c = (EmuCall*)p; c->k(c->a); } };
template <typename A>
void emu_launch(void (*k)(A), dim3 g, dim3 b, A a) {
    EmuCall<void (*)(A), A> c{k, a};
    emu_run_grid(g, b, &EmuCall<void (*)(A), A>::run, &c);
}
#define R4D_KLAUNCH(kernel, grid, block, stream, ...) emu_launch(&kernel, grid, block, __VA_ARGS__)

/* R4D_EMIT (r4d_emit.h): every argument converted to its parameter's type, as r4d_emit does, and
 * `lds` bytes of dynamic LDS; the kernel's name is kept in emu_last_kernel. Returns RAD_OK. */
extern const char* emu_last_kernel;
template <class... P, class... A>
int emu_emit(const char* name, void (*k)(P...), dim3 g, dim3 b, size_t lds, const A&... a) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    struct Call {
        void (*k)(P...);
        std::tuple<P...> v;
        static void run(void* p) { Call* c = (Call*)p; std::apply(c->k, c->v); }
    };
    Call c{k, std::tuple<P...>(static_cast<P>(a)...)};
    emu_last_kernel = name;
    emu_run_grid(g, b, &Call::run, &c, lds);
    return 0;
}
#define R4D_EMIT(stream, kernel, grid, block, lds, ...) \
    emu_emit(#kernel, (kernel), (grid), (block), (size_t)(lds), __VA_ARGS__)
/* R4D_DYN_LDS (r4d_launch.h): the launch's dynamic LDS, poisoned at each block's start and ending
 * at an inaccessible page (its start 16-byte aligned), so a read before any write shows and one
 * past the end faults */
void* emu_dyn_lds();
#define R4D_DYN_LDS(type, name) type* const name = (type*)emu_dyn_lds()
