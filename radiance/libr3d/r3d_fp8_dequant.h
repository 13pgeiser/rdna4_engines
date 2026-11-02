/* r3d_fp8_dequant.h -- E4M3 -> f16 decode through a 256-entry LDS table.
 *
 * gfx11 has no E4M3-to-f16 instruction: `__builtin_amdgcn_cvt_pk_f16_fp8` compiles on gfx1102 but
 * emits no `v_cvt_pk_f16_fp8`, only the same software exponent work as `e4m3_to_f32`. The codec in
 * r3d_codecs.h costs about 25 instructions a pair, 100 an eight-code group. On the M=1 vocabulary
 * head (r3d_vocab_bf16.hip, gfx1102, 32 CU) that makes the row ALU-bound at about 8x its DRAM floor.
 *
 * Every one of E4M3's 256 bit patterns has an exact f16 value (f16's exponent range covers E4M3's,
 * subnormals included, and NaN maps to NaN), so the decoder is a 512-byte table and eight LDS reads
 * replace about a hundred ALU instructions. The table lives in LDS, not __constant__: each lane
 * decodes a different code, and divergent constant indexes serialise.
 *
 * Cost: 512 bytes of LDS a block and a 256-store fill plus a barrier at block start. Reads are
 * 16-bit loads at data-dependent addresses; random codes put about two lanes on each bank word, for
 * about two cycles a read.
 *
 * The table is filled by calling the codec once per entry, so results are bit-identical to the
 * codec. Invariant: the table must be filled before any lane reads it; r3d_e4m3_f16_init contains
 * the barrier.
 *
 * int8 codes need no table: `v_cvt_f32_i8` is a hardware instruction (r3d_codecs.h: i8x2_to_f16x2).
 *
 * Include in a .hip after r3d_codecs.h.
 */
#ifndef R3D_FP8_DEQUANT_H
#define R3D_FP8_DEQUANT_H

#include "r3d_codecs.h"

#define R3D_E4M3_TBL 256          /* every E4M3 bit pattern, in order, as the f16 it means */
#define R3D_E4M3_TBL_BYTES (R3D_E4M3_TBL * 2)

/* Fill the block's table: `tbl` is __shared__ uint16_t[R3D_E4M3_TBL]. Ends with the barrier that
 * orders the fill before any read. */
__device__ __forceinline__ void r3d_e4m3_f16_init(uint16_t* tbl, int tid, int nthr)
{
    for (int i = tid; i < R3D_E4M3_TBL; i += nthr)
        tbl[i] = f32_to_f16(e4m3_to_f32((uint8_t)i));
    __syncthreads();
}

/* Two codes (the low 16 bits of a dword) as one f16 pair in dot2_f16x2 layout, the first code in
 * the low half. Same result as e4m3x2_to_f16x2. */
__device__ __forceinline__ uint32_t r3d_e4m3x2_ld(const uint16_t* tbl, uint32_t pair)
{
    const uint16_t lo = tbl[pair & 0xffu];
    const uint16_t hi = tbl[(pair >> 8) & 0xffu];
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

/* Eight codes (two dwords) as four f16 pairs; the fp8 counterpart of codes8_to_f16x4 in
 * r3d_vocab_bf16.hip. */
__device__ __forceinline__ void r3d_e4m3x8_ld(const uint16_t* tbl, uint2 c, uint32_t* w)
{
    const uint32_t d0 = c.x, d1 = c.y;
    w[0] = r3d_e4m3x2_ld(tbl, d0 & 0xffffu);
    w[1] = r3d_e4m3x2_ld(tbl, d0 >> 16);
    w[2] = r3d_e4m3x2_ld(tbl, d1 & 0xffffu);
    w[3] = r3d_e4m3x2_ld(tbl, d1 >> 16);
}

/* One code, for rows that decode a scalar per step (an activation tile staged a byte a lane). */
__device__ __forceinline__ uint16_t r3d_e4m3_ld(const uint16_t* tbl, uint8_t code)
{
    return tbl[code];
}

/* --- The pure-ALU E4M3 -> f16 decode (no LDS table). ---
 *
 * Two E4M3 bytes (a halfword) -> two f16 (a half2 pair, [hi:lo]). E4M3 [s][e:4][m:3] -> f16
 * [s][e:4 + 8 bias][m:3][0:7], so f16 = (x<<7) + ((x&0x80)<<7) + 0x2000 per byte. Addition carries
 * the +8 bias when bit 13 is already set (E4M3 exp >= 8); OR would drop it. Denormals/zero map to
 * tiny f16 (the fold's 2^-11 floor dwarfs the difference from exact zero).
 *
 * Shared by the fp8a8_tiled, moe_gemm_q, and mxfp4a8 kernels as the ALU=1 variant; the ALU=0
 * variant is the table above. The two rows per op let the tune pick the faster per geometry. */
__device__ __forceinline__ uint32_t r3d_e4m3x2_to_f16x2_alu(const uint32_t pair)
{
    const uint32_t lo = ((pair & 0xffu) << 7) + ((pair & 0x80u) << 7) + 0x2000u;
    const uint32_t hi = (((pair >> 8) & 0xffu) << 7) + (((pair >> 8) & 0x80u) << 7) + 0x2000u;
    return lo | (hi << 16);
}

/* Eight codes (two dwords) as four f16 pairs, pure ALU (no table). */
__device__ __forceinline__ void r3d_e4m3x8_to_f16x8_alu(const uint2 c, uint32_t* w)
{
    w[0] = r3d_e4m3x2_to_f16x2_alu(c.x & 0xffffu);
    w[1] = r3d_e4m3x2_to_f16x2_alu(c.x >> 16);
    w[2] = r3d_e4m3x2_to_f16x2_alu(c.y & 0xffffu);
    w[3] = r3d_e4m3x2_to_f16x2_alu(c.y >> 16);
}

/* --- The ALU-branching decode: one call site per kernel, the branch lives here. ---
 *
 * ALU=1: the per-byte formula above (no LDS table). ALU=0: the 256-entry table (tbl). The two
 * forms are bit-identical for normal-range E4M3 codes; denormals/zero differ by a tiny f16 the
 * scale fold's 2^-11 floor dwarfs. Kernels keep an `ALU` template param and call these; they do
 * not branch on ALU themselves, so a new decode form is one change here, not per kernel. */

/* Eight codes (two dwords) as four f16 pairs. */
template <int ALU>
__device__ __forceinline__ void r3d_e4m3x8_decode(const uint16_t* tbl, const uint2 c, uint32_t* w)
{
    if (ALU) r3d_e4m3x8_to_f16x8_alu(c, w);
    else r3d_e4m3x8_ld(tbl, c, w);
}

/* Four codes (one dword) as two f16 pairs. */
template <int ALU>
__device__ __forceinline__ void r3d_e4m3x4_decode(const uint16_t* tbl, const uint32_t d,
                                                  uint32_t* o)
{
    if (ALU) {
        o[0] = r3d_e4m3x2_to_f16x2_alu(d & 0xffffu);
        o[1] = r3d_e4m3x2_to_f16x2_alu(d >> 16);
    } else {
        o[0] = r3d_e4m3x2_ld(tbl, d & 0xffffu);
        o[1] = r3d_e4m3x2_ld(tbl, d >> 16);
    }
}

/* Sixteen codes (four dwords) as eight f16 pairs. */
template <int ALU>
__device__ __forceinline__ void r3d_e4m3x16_decode(const uint16_t* tbl, const uint32_t d[4],
                                                   uint32_t* w)
{
    r3d_e4m3x8_decode<ALU>(tbl, uint2{ d[0], d[1] }, w);
    r3d_e4m3x8_decode<ALU>(tbl, uint2{ d[2], d[3] }, w + 4);
}

/* --- Shared scale-fold and code-load helpers (the quantised GEMM family). --- */

/* code * scale, in f16 arithmetic: one v_mul_f16 a half, round to nearest, subnormals kept. Not
 * f16x2_pack, which truncates. Shared by the fp8a8 and mxfp4a8 tiles. */
__device__ __forceinline__ uint32_t r3d_fold_pair(uint32_t w, float sc)
{
    const uint16_t lo16 = (uint16_t)(w & 0xffffu), hi16 = (uint16_t)(w >> 16);
    const _Float16 a = __builtin_bit_cast(_Float16, lo16);
    const _Float16 b = __builtin_bit_cast(_Float16, hi16);
    const _Float16 s = (_Float16)sc;
    const _Float16 rl = a * s, rh = b * s;
    return (uint32_t)__builtin_bit_cast(uint16_t, rl) |
           ((uint32_t)__builtin_bit_cast(uint16_t, rh) << 16);
}

/* Eight codes (8 bytes) as a uint2, a nontemporal load when NT, a memcpy otherwise. Shared by the
 * fp8a8, mxfp4a8, and vocab tiles. */
template <int NT>
__device__ __forceinline__ uint2 r3d_loadc8(const uint8_t* p)
{
    typedef uint32_t __attribute__((ext_vector_type(2))) u8v;
    u8v v;
    if (NT) v = __builtin_nontemporal_load((const u8v*)p);
    else __builtin_memcpy(&v, p, 8);
    uint2 r;
    __builtin_memcpy(&r, &v, 8);
    return r;
}

#endif /* R3D_FP8_DEQUANT_H */
