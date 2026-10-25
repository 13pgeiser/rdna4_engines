// Shared device code for the M2XFP4 GEMMs (r4d_gemm_m2xfp4a8_decode, r4d_gemm_m2xfp4a8_tiled):
// the folded-magnitude table and the eight-element unpack that reads it. The M2 analogue of
// r4d_mxfp4.h: the same e2m1 codes and E8M0 exponents, plus one more per-group byte of weight-side
// Sg-EM metadata (arXiv 2601.19213 §4.4.2), which folds the same way -- through a table, with a
// v_perm_b32 per half-word.
//
// The container ("m2xfp4", the quantiser's scheme) stores, per group of 32 K, an E8M0 exponent byte
// whose adaptive bias b in {-1,0,1} is already folded in, and a meta byte holding one two-bit
// subgroup multiplier k_s per eight columns, s = columns [8s, 8s+8) of the group: the stored byte's
// bits [2s, 2s+2). The exact weight is
//
//     w = e2m1mag[c & 7] * sign(c) * 2^(scale - 127) * (1 + k_s / 4)
//
// with e2m1mag = {0, .5, 1, 1.5, 2, 3, 4, 6} and k -> {1.0, 1.25, 1.5, 1.75}. The fold of
// r4d_gemm_mxfp4a8_nt_m64.hip applies to it unchanged: d = Wref[n] - E8M0[n][blk] goes through the
// table, 2^(Wref[n]-127) once in the epilogue. Wref[n] is still the row's largest exponent byte,
// so d >= 0 holds for every block; what the meta adds is bounded the same way -- the largest
// element a table cell holds is 6 * 1.75 * 2^-d = 10.5 * 2^-d, an E4M3 normal for every d >= 0, so
// no cell saturates and the fold is exact in its exponent at every block.
#pragma once

#include "r4d_mxfp4.h"   // the fold clamp r4d_mxfp4_fold_d, the carried-scale reader, GROUP

// kM2xfp4Mag[d][k] is the eight e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by
// (1 + k/4) * 2^-d, as packed e4m3 bytes (low word, high word), indexed by the 3-bit magnitude
// through v_perm_b32 -- kMxfp4Mag's table with a second index. Entries are the round-to-nearest-even
// e4m3 of the product, matching the hardware conversion.
//
// Exactness, per magnitude and k, in the normal range of e4m3 (every entry here down to its
// subnormal floor behaves exactly like r4d_mxfp4.h's table one binade earlier, where the 2^-9 step
// is coarse; the entries flush the same way and the sums they miss are invisible next to the
// accumulator):
//   - k = 0 is r4d_mxfp4.h's kMxfp4Mag row, byte for byte.
//   - k = 1, 2: exact. The products' mantissas (1.25, 1.5 and 1.75 times a magnitude whose own
//     mantissa is 1.0 or 1.1) take at most the three e4m3 mantissa bits, at every d while the
//     value is normal.
//   - k = 3 (x1.75): the even-mantissa magnitudes {0, .5, 1, 2, 4} stay exact (1.11 x 1.0); the odd
//     ones {1.5, 3, 6} land on 1.0101, four mantissa bits, and round to the tie's even side -- 2.5,
//     5 and 10 scaled by 2^-d. That is 2^-4.4 relative error on those elements and nothing on the
//     rest, accepted here: the CPU decode (and the reference) stay exact, and 6 * 1.75 = 10.5 at
//     d = 0 is what the row-maximum bound above has to fit, which it does.
//
// Static: each kernel's code object carries its own copy.
static __device__ __constant__ unsigned int kM2xfp4Mag[16][4][2] = {
  {{0x3c383000u, 0x4c484440u}, {0x3f3a3200u, 0x4f4a4742u}, {0x413c3400u, 0x514c4944u},
    {0x423e3600u, 0x524e4a46u}},
  {{0x34302800u, 0x44403c38u}, {0x37322a00u, 0x47423f3au}, {0x39342c00u, 0x4944413cu},
    {0x3a362e00u, 0x4a46423eu}},
  {{0x2c282000u, 0x3c383430u}, {0x2f2a2200u, 0x3f3a3732u}, {0x312c2400u, 0x413c3934u},
    {0x322e2600u, 0x423e3a36u}},
  {{0x24201800u, 0x34302c28u}, {0x27221a00u, 0x37322f2au}, {0x29241c00u, 0x3934312cu},
    {0x2a261e00u, 0x3a36322eu}},
  {{0x1c181000u, 0x2c282420u}, {0x1f1a1200u, 0x2f2a2722u}, {0x211c1400u, 0x312c2924u},
    {0x221e1600u, 0x322e2a26u}},
  {{0x14100800u, 0x24201c18u}, {0x17120a00u, 0x27221f1au}, {0x19140c00u, 0x2924211cu},
    {0x1a160e00u, 0x2a26221eu}},
  {{0x0c080400u, 0x1c181410u}, {0x0f0a0500u, 0x1f1a1712u}, {0x110c0600u, 0x211c1914u},
    {0x120e0700u, 0x221e1a16u}},
  {{0x06040200u, 0x14100c08u}, {0x08050200u, 0x17120f0au}, {0x09060300u, 0x1914110cu},
    {0x0a070400u, 0x1a16120eu}},
  {{0x03020100u, 0x0c080604u}, {0x04020100u, 0x0f0a0805u}, {0x04030200u, 0x110c0906u},
    {0x05040200u, 0x120e0a07u}},
  {{0x02010000u, 0x06040302u}, {0x02010100u, 0x08050402u}, {0x02020100u, 0x09060403u},
    {0x03020100u, 0x0a070504u}},
  {{0x01000000u, 0x03020201u}, {0x01010000u, 0x04020201u}, {0x01010000u, 0x04030202u},
    {0x01010000u, 0x05040302u}},
  {{0x00000000u, 0x02010100u}, {0x00000000u, 0x02010101u}, {0x01000000u, 0x02020101u},
    {0x01000000u, 0x03020101u}},
  {{0x00000000u, 0x01000000u}, {0x00000000u, 0x01010000u}, {0x00000000u, 0x01010100u},
    {0x00000000u, 0x01010100u}},
  {{0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x01000000u},
    {0x00000000u, 0x01000000u}},
  {{0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u}},
  {{0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u}},
};

// The two-bit field of a meta byte that covers `koff`, an element offset inside a 32-column group
// that is a multiple of 8 (which the kernels' staging offsets all are): the subgroup s = koff / 8,
// bits [2s, 2s+2). A thread whose eight elements start at koff takes this shift once and one meta
// byte per (row, block) alongside the exponent byte.
static __device__ __forceinline__ int r4d_m2xfp4_kshift(int koff) {
  return 2 * ((koff >> 3) & 3);
}

// Eight e2m1 codes (four packed bytes, element 2j in the low nibble of byte j) -> eight e4m3 bytes
// in k order (word 0 = elements 0..3, word 1 = 4..7), with the exponent difference `d` and the
// subgroup multiplier `k` folded in. r4d_mxfp4_unpack8_t with the [d][k] table row; see there. A
// caller that already holds the row, e.g. in an LDS copy (the tiled kernel's sMag, which keeps the
// lookup off the global load counter the weight and activation loads wait on), calls
// r4d_mxfp4_unpack8_t with it directly.
static __device__ __forceinline__ void r4d_m2xfp4_unpack8(unsigned int wv, int d, int k,
                                                          unsigned int* lo, unsigned int* hi) {
  r4d_mxfp4_unpack8_t(wv, kM2xfp4Mag[d][k][0], kM2xfp4Mag[d][k][1], lo, hi);
}

// ---- The raw entry points, kept beside the table rather than in r4d.h so the M2XFP4 kernels'
// whole surface is in their own files (the plugin ABI's entries are r4d_plugin.h's).

extern "C" {

// M2XFP4 weight, fp8 activation: the MXFP4 rows at the paper's weight-side Sg-EM (arXiv 2601.19213
// §4.4.2). Beside each E8M0 exponent per 32 K (whose adaptive bias the quantiser already folded
// in) the weight carries one meta byte per (row, 32 of K): four two-bit subgroup multipliers, the
// subgroup s covering columns [8s, 8s+8) of the group, value k_s in 0..3. The exact weight is
// e2m1mag[c&7] * sign(c) * 2^(scale-127) * (1 + k_s/4); r4d_m2xfp4.h folds it into the same e4m3
// staging bytes through a second table index, so the fold, the Wref epilogue and the activation
// scale grids are the MXFP4 kernels'. `wm` is the [K/32][N] meta plane, `ldws` its row pitch (0 =
// N); the k=3 multipliers on odd-mantissa codes round as r4d_m2xfp4.h documents, up to 2^-4.4
// relative.
void r4d_gemm_m2xfp4a8_tiled(long at, long ascale, long wq, long ws, long wm, long wref, long c,
                             int M, int K, int N, int TN, int as_group, int as_rs, long stream,
                             int ldws = 0, int ldc = 0);
void r4d_m2xfp4a8_tile_a(long a, long at, int M, int K, long stream, int lda = 0);
long r4d_m2xfp4a8_tile_a_bytes(int M, int K);
int  r4d_gemm_m2xfp4a8_tiled_group(void);
// The decode band (M <= 128). The plugin ABI's r4d_rad_gemm_m2xfp4a8_decode serves it in the
// engine, the split partial in the op's scratch as for mxfp4a8; this raw entry takes that scratch
// from the caller: KS * M * N * 4 bytes when KS > 1 (a KS of 2 runs as 4 where a 128-wide slab
// would run past K), 16 * M * N * 4 when KS is 0 (the widest split the rule picks), null only at
// KS 1. The arrival counters are the unit's own, one block per output buffer.
void r4d_gemm_m2xfp4a8_decode(long a, long ascale, long wq, long ws, long wm, long wref, long c,
                              int M, int K, int N, int KS, int BK, int NT, int as_group,
                              int as_rs, long scratch, long scratch_bytes, long stream,
                              int lda = 0, int ldws = 0, int ldc = 0);
int  r4d_gemm_m2xfp4a8_decode_max_m(void);

}  // extern "C"
