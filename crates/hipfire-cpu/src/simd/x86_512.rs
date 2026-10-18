// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AVX-512F row dots for the two MQ4 V2 formats the D2 host-slot expert path
//! reads: qt 44 `Mq4G256V2` (gate/up, `k = 2560`) and qt 53 `Mq4G128V2` (down,
//! `k = 640`).
//!
//! This tier is **bit-identical** to the AVX2 kernels in [`super::x86`], not
//! merely close to them — it is a wider execution of the *same* arithmetic, so
//! swapping tiers never changes a logit. [`super::x86::codes_dot_sum`] keeps two
//! independent 8-lane FMA chains: `dot0` takes the even chunks of a range, `dot1`
//! the odd ones, and `hsum256(dot0 + dot1)` joins them. A chunk pair's sixteen
//! activations are contiguous, so one `zmm` carries both chains at once:
//! lanes 0‥7 *are* `dot0` (the even chunk) and lanes 8‥15 *are* `dot1` (the odd
//! chunk). Every lane then executes exactly the FMA sequence its AVX2 chain
//! executes, in the same order, on the same operands (an FMA is the same fused,
//! single-rounded operation at either width). An odd tail chunk belongs to
//! chain 0 only, so it is applied to lanes 0‥7 under a mask. The join takes the
//! two `ymm` halves of the accumulator, adds them lane-wise (what
//! `_mm256_add_ps(dot0, dot1)` does), and reuses the same `hsum256` tree; the
//! affine association above it is the AVX2 kernels' own.
//!
//! Nothing here reassociates, reorders a reduction, widens a precision or
//! changes an operand: the only new thing is how many lanes one instruction
//! carries. Only AVX-512F is used (no BW/VL/DQ): the nibble planes are split
//! and interleaved with SSE2 byte ops and widened with `vpmovzxbd`.
//!
//! The `Σx` side follows suit: `!CACHED` runs the two `Σx` chains in a second
//! `zmm` with the identical lane mapping, and the cached variants take the same
//! per-128 sums [`super::x86::activation_sums_v2`] prepares for the AVX2 tier.

use super::x86::{f16_pair, f16_quad, hsum256};
use core::arch::x86_64::*;

/// Mask selecting lanes 0‥7 — chain 0's half of the accumulator.
const CHAIN0: __mmask16 = 0x00ff;

/// Low and high nibble planes of `bytes`, interleaved into element order:
/// byte `j` contributes its low nibble as code `2j` and its high nibble as code
/// `2j + 1` (the qt 13/44/53 nibble order [`super::x86`] decodes). Returns
/// `(codes 0‥15, codes 16‥31)` as bytes `0‥15`.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn nibble_planes(bytes: __m128i) -> (__m128i, __m128i) {
    let mask = _mm_set1_epi8(0x0f);
    let lo = _mm_and_si128(bytes, mask);
    let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    (_mm_unpacklo_epi8(lo, hi), _mm_unpackhi_epi8(lo, hi))
}

/// Sixteen byte codes widened to `f32` lanes.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn widen16(codes: __m128i) -> __m512 {
    _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes))
}

/// The two `Σ` chains' join: `hsum256(chain0 + chain1)` with chain 0 in lanes
/// 0‥7 and chain 1 in lanes 8‥15 — the AVX2 kernels' exact reduction tree.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn join_chains(acc: __m512) -> f32 {
    let chain0 = _mm512_castps512_ps256(acc);
    let chain1 = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(acc)));
    hsum256(_mm256_add_ps(chain0, chain1))
}

/// `(Σ c·x, Σ x)` over the `first..last` 8-lane chunks of 4-bit codes that
/// start `PAYLOAD` bytes into the group: [`super::x86::codes_dot_sum`]`::<4,
/// PAYLOAD, CACHED>` bit for bit, for any `first..last`.
///
/// Two chunk pairs (four chunks, 16 payload bytes, 32 activations) are decoded
/// per step; each pair is one FMA into the shared accumulator, in chunk order.
///
/// # Safety
///
/// Requires AVX-512F (+ AVX2/FMA/F16C), `4` readable payload bytes per chunk of
/// `first..last` at `gptr + PAYLOAD`, and `8` readable `f32` per chunk at `xg`.
/// Memory is touched only within those chunks (the odd tail uses a 4-byte
/// payload read and a masked activation load).
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn codes_dot_sum512<const PAYLOAD: usize, const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    first: usize,
    last: usize,
    cached_sum: f32,
) -> (f32, f32) {
    let mut dot = _mm512_setzero_ps();
    let mut sum = _mm512_setzero_ps();
    let mut idx = first;
    while idx + 3 < last {
        let bytes = _mm_loadu_si128(gptr.add(PAYLOAD + idx * 4) as *const __m128i);
        let (lo, hi) = nibble_planes(bytes);
        let (c0, c1) = (widen16(lo), widen16(hi));
        let x0 = _mm512_loadu_ps(xg.add(idx * 8));
        let x1 = _mm512_loadu_ps(xg.add(idx * 8 + 16));
        dot = _mm512_fmadd_ps(c0, x0, dot);
        dot = _mm512_fmadd_ps(c1, x1, dot);
        if !CACHED {
            sum = _mm512_add_ps(sum, x0);
            sum = _mm512_add_ps(sum, x1);
        }
        idx += 4;
    }
    if idx + 1 < last {
        let bytes = _mm_loadl_epi64(gptr.add(PAYLOAD + idx * 4) as *const __m128i);
        let c = widen16(nibble_planes(bytes).0);
        let x = _mm512_loadu_ps(xg.add(idx * 8));
        dot = _mm512_fmadd_ps(c, x, dot);
        if !CACHED {
            sum = _mm512_add_ps(sum, x);
        }
        idx += 2;
    }
    if idx < last {
        // Odd tail: chain 0 only, i.e. lanes 0‥7. Lanes 8‥15 keep their value.
        let word = (gptr.add(PAYLOAD + idx * 4) as *const i32).read_unaligned();
        let c = widen16(nibble_planes(_mm_cvtsi32_si128(word)).0);
        let x = _mm512_maskz_loadu_ps(CHAIN0, xg.add(idx * 8));
        dot = _mm512_mask3_fmadd_ps(c, x, dot, CHAIN0);
        if !CACHED {
            sum = _mm512_mask_add_ps(sum, CHAIN0, sum, x);
        }
    }
    let dot = join_chains(dot);
    let sum = if CACHED { cached_sum } else { join_chains(sum) };
    (dot, sum)
}

/// One 256-element `Mq4G256V2` group (`[s0 z0 s1 z1]` `fp16`, then 128 B of
/// nibbles): [`super::x86`]'s `v2_group_dot::<4>` bit for bit. `sums` is read
/// (`sums[0]`, `sums[1]`) only when `CACHED`.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn v2_group_dot512<const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    sums: *const f32,
) -> f32 {
    let h = f16_quad(gptr);
    let (c0, c1) = if CACHED {
        (*sums, *sums.add(1))
    } else {
        (0.0, 0.0)
    };
    let (d0, s0) = codes_dot_sum512::<8, CACHED>(gptr, xg, 0, 16, c0);
    let (d1, s1) = codes_dot_sum512::<8, CACHED>(gptr, xg, 16, 32, c1);
    (h[0] * d0 + h[1] * s0) + (h[2] * d1 + h[3] * s1)
}

/// One 128-element `Mq4G128V2` group (`[scale zero]` `fp16`, then 64 B of
/// nibbles): [`super::x86`]'s `uniform_group_dot::<4, 4, 16, _>` bit for bit.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn g128_group_dot512<const CACHED: bool>(
    gptr: *const u8,
    xg: *const f32,
    cached_sum: f32,
) -> f32 {
    let (scale, zero) = f16_pair(gptr);
    let (dot, sum) = codes_dot_sum512::<4, CACHED>(gptr, xg, 0, 16, cached_sum);
    scale * dot + zero * sum
}

/// qt 44 — `Mq4G256V2` row dot, 256 elements / 136 B per group. Bit-equal to
/// [`super::x86::mq4g256v2_row_dot`].
///
/// # Safety
///
/// Requires AVX-512F, AVX2, FMA and F16C; `k % 256 == 0`; `(k / 256) * 136`
/// readable bytes at `row`; `k` readable `f32` at `x`.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += v2_group_dot512::<false>(row.add(g * 136), x.add(g * 256), core::ptr::null());
    }
    acc
}

/// qt 44 with the per-128 `Σx` from [`super::x86::activation_sums_v2`]
/// (`sums[2g]`, `sums[2g + 1]` for group `g`). Bit-equal to
/// [`mq4g256v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g256v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 256 {
        acc += v2_group_dot512::<true>(row.add(g * 136), x.add(g * 256), sums.add(2 * g));
    }
    acc
}

/// qt 53 — `Mq4G128V2` row dot, 128 elements / 68 B per group. Bit-equal to
/// [`super::x86::mq4g128v2_row_dot`].
///
/// # Safety
///
/// Requires AVX-512F, AVX2, FMA and F16C; `k % 128 == 0`; `(k / 128) * 68`
/// readable bytes at `row`; `k` readable `f32` at `x`.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot(row: *const u8, k: usize, x: *const f32) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 128 {
        acc += g128_group_dot512::<false>(row.add(g * 68), x.add(g * 128), 0.0);
    }
    acc
}

/// qt 53 with the per-128 `Σx` from [`super::x86::activation_sums_v2`]
/// (`sums[g]` for group `g`). Bit-equal to [`mq4g128v2_row_dot`].
///
/// # Safety
///
/// As [`mq4g128v2_row_dot`], plus `k / 128` readable `f32` at `sums`.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot_cached(
    row: *const u8,
    k: usize,
    x: *const f32,
    sums: *const f32,
) -> f32 {
    let mut acc = 0.0f32;
    for g in 0..k / 128 {
        acc += g128_group_dot512::<true>(row.add(g * 68), x.add(g * 128), *sums.add(g));
    }
    acc
}

/// Widest token group the AVX-512 multi-token kernels carry: one `zmm` dot
/// accumulator per token (both chains share it) plus the two shared code
/// vectors and an activation temporary, far inside the 32 `zmm` registers.
pub(super) const MULTI_TOKENS_AVX512: usize = 8;

/// The sixteen 8-lane chunks (128 activations) of one V2 half / `Mq4G128V2`
/// group at payload `p`, dotted with `T` activation rows `x[t] + off`, each
/// four-chunk payload decoded **once** for all `T` tokens.
///
/// Per token this is [`codes_dot_sum512`]'s `CACHED` body over `first..last =
/// 0..16` verbatim (two FMAs per four-chunk step into the token's own `zmm`,
/// then [`join_chains`]), so every result is bit-identical to the single-token
/// kernel's; tokens never mix.
///
/// # Safety
///
/// Requires AVX-512F (+ AVX2/FMA/F16C), `64` readable payload bytes at `p` and
/// `128` readable `f32` at `x[t].add(off)` for every `t < T`.
#[inline]
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn block_dots_multi512<const T: usize>(
    p: *const u8,
    x: &[*const f32; T],
    off: usize,
) -> [f32; T] {
    let mut dot = [_mm512_setzero_ps(); T];
    let mut idx = 0;
    while idx < 16 {
        let bytes = _mm_loadu_si128(p.add(idx * 4) as *const __m128i);
        let (lo, hi) = nibble_planes(bytes);
        let (c0, c1) = (widen16(lo), widen16(hi));
        for t in 0..T {
            let xb = x[t].add(off + idx * 8);
            dot[t] = _mm512_fmadd_ps(c0, _mm512_loadu_ps(xb), dot[t]);
            dot[t] = _mm512_fmadd_ps(c1, _mm512_loadu_ps(xb.add(16)), dot[t]);
        }
        idx += 4;
    }
    core::array::from_fn(|t| join_chains(dot[t]))
}

/// [`mq4g256v2_row_dot_cached`] for `T` activation rows at once (`x[t]`,
/// `sums[t]`), each weight chunk decoded once. Result `t` is bit-equal to
/// `mq4g256v2_row_dot_cached(row, k, x[t], sums[t])`.
///
/// # Safety
///
/// As [`mq4g256v2_row_dot_cached`] for every token.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g256v2_row_dot_cached_multi<const T: usize>(
    row: *const u8,
    k: usize,
    x: &[*const f32; T],
    sums: &[*const f32; T],
) -> [f32; T] {
    let mut acc = [0.0f32; T];
    for g in 0..k / 256 {
        let gptr = row.add(g * 136);
        let h = f16_quad(gptr);
        let d0 = block_dots_multi512::<T>(gptr.add(8), x, g * 256);
        let d1 = block_dots_multi512::<T>(gptr.add(8 + 64), x, g * 256 + 128);
        for t in 0..T {
            let (s0, s1) = (*sums[t].add(2 * g), *sums[t].add(2 * g + 1));
            acc[t] += (h[0] * d0[t] + h[1] * s0) + (h[2] * d1[t] + h[3] * s1);
        }
    }
    acc
}

/// [`mq4g128v2_row_dot_cached`] for `T` activation rows at once. Result `t` is
/// bit-equal to `mq4g128v2_row_dot_cached(row, k, x[t], sums[t])`.
///
/// # Safety
///
/// As [`mq4g128v2_row_dot_cached`] for every token.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
pub(super) unsafe fn mq4g128v2_row_dot_cached_multi<const T: usize>(
    row: *const u8,
    k: usize,
    x: &[*const f32; T],
    sums: &[*const f32; T],
) -> [f32; T] {
    let mut acc = [0.0f32; T];
    for g in 0..k / 128 {
        let gptr = row.add(g * 68);
        let (scale, zero) = f16_pair(gptr);
        let d = block_dots_multi512::<T>(gptr.add(4), x, g * 128);
        for t in 0..T {
            acc[t] += scale * d[t] + zero * *sums[t].add(g);
        }
    }
    acc
}
