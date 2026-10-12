/* tests/g4k_emu: SIMT EMULATOR SHIM -- runs gemma4k's kernel source on the host. Every GPU thread is a fiber on
 * one OS thread; __syncthreads and every cross-lane op are barriers the scheduler enforces, and a
 * barrier some threads never reach is reported as a deadlock rather than hung. Test-only. */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
/* v_wmma_f32_16x16x16_bf16 (gfx12, wave32): A[M][K] lane M + 16 * (K / 8) element K % 8; B[K][N]
 * lane N + 16 * (K / 8) element K % 8; D[M][N] lane N + 16 * (M / 8) element M % 8 */
typedef short emu_v8s __attribute__((ext_vector_type(8)));
typedef float emu_v8f __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_bf16(emu_v8s a, emu_v8s b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c) emu_wmma_bf16((a), (b), (c))
typedef _Float16 emu_v8h __attribute__((ext_vector_type(8)));
emu_v8f emu_wmma_f16(emu_v8h a, emu_v8h b, emu_v8f c);
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c) emu_wmma_f16((a), (b), (c))
/* v_cvt_pk_fp8_f32 (gfx12, OCP E4M3): a and b as two codes into the low half of `old` (hi false)
 * or its high half, the other half kept; round to nearest even. The project's saturating encoder,
 * rad_f32_to_fp8e4m3, is the same function on the +-448 a kernel clamps to first. */
int emu_cvt_pk_fp8(float a, float b, int old, bool hi);
#define __builtin_amdgcn_cvt_pk_fp8_f32(a, b, old, hi) emu_cvt_pk_fp8((a), (b), (old), (hi))
extern int emu_last_block;     /* threads a block of the last grid had: which kernel ran */

struct alignas(8)  uint2  { unsigned x, y; };
struct alignas(16) uint4  { unsigned x, y, z, w; };
struct alignas(16) float4 { float x, y, z, w; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { return uint4{a, b, c, d}; }
inline uint2 make_uint2(unsigned a, unsigned b) { return uint2{a, b}; }
inline float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }
inline float __uint_as_float(unsigned u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }   /* v_rsq_f32 is within an ulp of it */
inline unsigned __float_as_uint(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }
inline float __int_as_float(int u) { float f; std::memcpy(&f, &u, 4); return f; }
inline int __float_as_int(float f) { int u; std::memcpy(&u, &f, 4); return u; }

/* one block at a time, every thread a fiber */
void emu_run_grid(dim3 grid, dim3 block, void (*body)(void*), void* ctx);
template <typename K, typename A>
struct EmuCall { K k; A a; static void run(void* p) { auto* c = (EmuCall*)p; c->k(c->a); } };
template <typename A>
void emu_launch(void (*k)(A), dim3 g, dim3 b, A a) {
    EmuCall<void (*)(A), A> c{k, a};
    emu_run_grid(g, b, &EmuCall<void (*)(A), A>::run, &c);
}
#define G4K_LAUNCH(kernel, grid, block, stream, ...) emu_launch(&kernel, grid, block, __VA_ARGS__)
