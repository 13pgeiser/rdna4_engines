// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 QSA (sparse/indexed attention, 12 layers) Tier C cases: GPU kernel vs
//! CPU twin at the Qwen4 shapes (24 q heads / 2 KV heads / head_dim 256,
//! indexer 4 heads x 1 KV x 128, compress 4, budget 2048 = 512 blocks,
//! selected capacity 2051, partial rotary 64, RoPE theta 1e7).
//!
//! Census of record: `local://census-qsa.md` (G942 route audit).
//!
//! IMPORTANT (gfx942): every parallel QSA kernel (`select_*_batched`, rows8 /
//! rows16 scores, `select_from_scores`, `attention_*_batched{,_hg4,_hg12}`) is
//! launched only when `arch_caps.has_gfx11_plus_simt()` (RDNA3/RDNA4) holds
//! (`tensor_ops.rs` `parallel`/`hg12` predicates). On gfx942 the public
//! launchers therefore run the 1-thread-per-row `*_serial` selectors and the
//! `indexed_attention_attention_f32_batched_serial` attention. The `route`
//! cases below call the public launchers (whatever branch the arch takes); the
//! `qwen4_qsa_probe_*` cases JIT the same `tensor_ops.hip` source under a
//! private module name and launch the unreachable-on-gfx942 kernels directly,
//! so a future gate widening has a ready wave64 verdict. A failing probe is a
//! finding about that kernel, NOT a failure of the gfx942 route.

use super::common::*;
use super::Q4Case;
use rdna_compute::rope::{Rope, RopeScaling, RopeYarn, Yarn};
use rdna_compute::tensor_ops::{
    indexed_attention_append_prologue, indexed_attention_attention,
    indexed_attention_attention_batch_with_arena, indexed_attention_cache_append,
    indexed_attention_cache_append_batch, indexed_attention_decode_prologue,
    indexed_attention_index_key_append_batch, indexed_attention_norm_rope,
    indexed_attention_norm_rope_batch, indexed_attention_pool_rope,
    indexed_attention_pool_rope_incremental, indexed_attention_reuse_selection,
    indexed_attention_select_batch_mirrored, IndexedAttentionAppendPrologue,
    IndexedAttentionAttention, IndexedAttentionAttentionBatch, IndexedAttentionCacheAppend,
    IndexedAttentionCacheAppendBatch, IndexedAttentionDecodePrologue,
    IndexedAttentionIndexKeyAppendBatch, IndexedAttentionNormRope,
    IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope, IndexedAttentionReuseSelection,
    IndexedAttentionSelectBatch, QsaKvFormat, QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const CASES: &[Q4Case] = &[
    Q4Case { name: "qwen4_qsa_gfx942_gates_and_kv_format_refusal", timeout_s: 60, run: case_gates },
    Q4Case { name: "qwen4_indexed_attention_norm_rope_f32_batched", timeout_s: 120, run: case_norm_rope_batched },
    Q4Case { name: "qwen4_indexed_attention_norm_rope_f32_single", timeout_s: 60, run: case_norm_rope_single },
    Q4Case { name: "qwen4_indexed_attention_cache_append_f32_batched", timeout_s: 60, run: case_cache_append_batched },
    Q4Case { name: "qwen4_indexed_attention_cache_append_f32_single", timeout_s: 60, run: case_cache_append_single },
    Q4Case { name: "qwen4_indexed_attention_index_key_append_bf16_batched", timeout_s: 60, run: case_index_key_append },
    Q4Case { name: "qwen4_bf16_round_trip_f32_strided_qsa_index_k", timeout_s: 60, run: case_round_trip_strided },
    Q4Case { name: "qwen4_copy_rows_strided_f32_qsa_raw_index_keys", timeout_s: 60, run: case_copy_rows_strided },
    Q4Case { name: "qwen4_indexed_attention_decode_prologue_f32", timeout_s: 120, run: case_decode_prologue },
    Q4Case { name: "qwen4_indexed_attention_pool_rope_f32", timeout_s: 120, run: case_pool_rope_f32 },
    Q4Case { name: "qwen4_indexed_attention_pool_rope_bf16", timeout_s: 120, run: case_pool_rope_bf16 },
    Q4Case { name: "qwen4_indexed_attention_select_batch_f32_route", timeout_s: 600, run: case_select_f32 },
    Q4Case { name: "qwen4_indexed_attention_select_batch_bf16_route", timeout_s: 600, run: case_select_bf16 },
    Q4Case { name: "qwen4_indexed_attention_attention_batch_f32_route", timeout_s: 600, run: case_attention_batch },
    Q4Case { name: "qwen4_indexed_attention_reuse_selection", timeout_s: 60, run: case_reuse_selection },
    Q4Case { name: "qwen4_indexed_attention_attention_f32_single", timeout_s: 120, run: case_attention_single },
    Q4Case { name: "qwen4_qsa_probe_select_scores_rows8_f32", timeout_s: 600, run: probe_rows8 },
    Q4Case { name: "qwen4_qsa_probe_select_scores_rows16_f32", timeout_s: 600, run: probe_rows16 },
    Q4Case { name: "qwen4_qsa_probe_select_f32_batched", timeout_s: 600, run: probe_select_parallel },
    Q4Case { name: "qwen4_qsa_probe_attention_f32_batched", timeout_s: 600, run: probe_attention_batched },
    Q4Case { name: "qwen4_qsa_probe_attention_f32_batched_hg4", timeout_s: 600, run: probe_attention_hg4 },
    Q4Case { name: "qwen4_qsa_probe_attention_f32_batched_hg12", timeout_s: 600, run: probe_attention_hg12 },
];

// ── Geometry ──────────────────────────────────────────────────────────────────
const HD: usize = QSA_HEAD_DIM; // 256
const IDX_DIM: usize = QSA_INDEX_DIM; // 128
const IDX_Q_W: usize = QSA_INDEX_HEADS * QSA_INDEX_DIM; // 512
const IDX_W: usize = IDX_Q_W + IDX_DIM; // 640: [index q | index k]
const KV_W: usize = QSA_KV_HEADS * QSA_HEAD_DIM; // 512
const Q_W: usize = QSA_Q_HEADS * QSA_HEAD_DIM; // 6144
const QG_W: usize = 2 * Q_W; // 12288: [heads][q | gate]
const ROTARY: usize = 64; // head_dim * partial_rotary_factor 0.25
const BUDGET_BLOCKS: usize = QSA_BUDGET / QSA_COMPRESS; // 512
const SEL_CAP: usize = QSA_BUDGET + QSA_COMPRESS - 1; // 2051
const POOLED_CAP: usize = 1024; // shape bound handed to the selector

// ── Small helpers ─────────────────────────────────────────────────────────────
/// Byte-addressed `DType::Raw` i32 tensor with `numel == bytes`, the shape the
/// QSA launchers check (`selected.numel() >= capacity * 4`).
fn raw_i32(gpu: &mut Gpu, data: &[i32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.upload_raw(&bytes, &[bytes.len()]).map_err(e)
}

fn bf16_vec(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
    up_bf16(gpu, data)
}

/// NaN-sentinel F32 tensor, shaped `[n]`.
fn nan_f32(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    sentinel_f32(gpu, n)
}

/// RNE round of a finite F32 to BF16 precision, in place on the F32 bits
/// (`bf16_round_trip_f32_strided` / the prologue's index-key round trip).
fn bf16_trunc_rne(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000)
}

fn dyadic(rng: &mut Rng, n: usize, denom: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.below(17) as f32 - 8.0) / denom).collect()
}

/// Compare `got` to `want`: elements where `exact_at(i)` bit-exactly, the rest
/// within `tol`.
fn cmp_split(
    label: &str,
    got: &[f32],
    want: &[f32],
    tol: Tol,
    exact_at: impl Fn(usize) -> bool,
) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let (mut ge, mut we, mut gc, mut wc) = (vec![], vec![], vec![], vec![]);
    for i in 0..got.len() {
        if exact_at(i) {
            ge.push(got[i]);
            we.push(want[i]);
        } else {
            gc.push(got[i]);
            wc.push(want[i]);
        }
    }
    all(vec![
        exact_f32(&format!("{label}[copy]"), &ge, &we),
        close_f32(&format!("{label}[math]"), &gc, &wc, tol),
    ])
}

fn yarns() -> [(&'static str, RopeYarn); 2] {
    let rope = Rope {
        theta: 1.0e7,
        scaling: RopeScaling::Yarn(Yarn::new(4.0, 4096, None, None, None, None)),
    };
    [("plain", RopeYarn::OFF), ("yarn4", rope.yarn(ROTARY))]
}

// ── CPU twins ─────────────────────────────────────────────────────────────────
/// `indexed_attention_norm_rope_f32{,_batched}` / `qsa_norm_rope_head`
/// (`kernels/src/tensor_ops.hip:2881-2921`, `3176-3215`): serial i-order
/// sum of squares, `inv = rsqrt(sum/dim + 1e-6)`, `(1 + norm)` scale, HalfSplit
/// RoPE over the first `rotary_dim` channels with
/// `angle = position * powf(1e7, -2*base/rotary_dim)` and the YaRN `sin_cos`
/// host twin. The arch-qwen4 CPU twin is `ops.rs:464 rope_prefix_halfsplit`
/// (same angle/yarn math; it has no RMS/learned-norm stage).
fn norm_rope_head(x: &[f32], norm: &[f32], position: usize, rotary_dim: usize, yarn: RopeYarn) -> Vec<f32> {
    let head_dim = x.len();
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let inv = 1.0f32 / (sum / head_dim as f32 + 1.0e-6f32).sqrt();
    let half = rotary_dim / 2;
    let mut out = vec![0.0f32; head_dim];
    for c in 0..head_dim {
        let mut value = x[c] * inv * (1.0 + norm[c]);
        if c < rotary_dim {
            let partner = if c < half { c + half } else { c - half };
            let other = x[partner] * inv * (1.0 + norm[partner]);
            let base = if c < half { c } else { partner };
            let angle =
                position as f32 * 10_000_000.0f32.powf(-((2 * base) as f32) / rotary_dim as f32);
            let (sine, cosine) = yarn.sin_cos(angle, base);
            value = if c < half {
                value * cosine - other * sine
            } else {
                other * sine + value * cosine
            };
        }
        out[c] = value;
    }
    out
}

/// One pooled index block, `qsa_pool_rope` (`tensor_ops.hip:3471-3542`): mean of
/// `compress` raw keys (sequential `/compress`), BF16 round, BF16-rounded
/// square, the kernel's 256-slot LDS tree sum (strides 128..1), learned-norm
/// scale rounded to BF16, partner RoPE at `block * compress`, final BF16
/// round. (`ops.rs:490 qsa_pool_block` / `reference_forward.rs:946` are the
/// model-level twins without the BF16 intermediates.)
fn pool_rope_block(raw: &[f32], norm: &[f32], block: usize, yarn: RopeYarn) -> Vec<f32> {
    let dim = IDX_DIM;
    let mean_bf16: Vec<f32> = (0..dim)
        .map(|c| {
            let mut mean = 0.0f32;
            for row in 0..QSA_COMPRESS {
                mean += raw[row * dim + c] / QSA_COMPRESS as f32;
            }
            bf16_round(mean)
        })
        .collect();
    let mut partial = vec![0.0f32; 256];
    for c in 0..dim {
        partial[c] = bf16_round(mean_bf16[c] * mean_bf16[c]);
    }
    let mut stride = 128;
    while stride > 0 {
        for t in 0..stride {
            partial[t] += partial[t + stride];
        }
        stride >>= 1;
    }
    let inv = 1.0f32 / (partial[0] / dim as f32 + 1.0e-6f32).sqrt();
    let half = ROTARY / 2;
    (0..dim)
        .map(|c| {
            let mut value = bf16_round(mean_bf16[c] * inv * (1.0 + norm[c]));
            if c < ROTARY {
                let partner = if c < half { c + half } else { c - half };
                let other = bf16_round(mean_bf16[partner] * inv * (1.0 + norm[partner]));
                let pair = if c < half { c } else { partner };
                let angle = (block * QSA_COMPRESS) as f32
                    * 10_000_000.0f32.powf(-((2 * pair) as f32) / ROTARY as f32);
                let (sine, cosine) = yarn.sin_cos(angle, pair);
                value = if c < half {
                    value * cosine - other * sine
                } else {
                    other * sine + value * cosine
                };
            }
            bf16_round(value)
        })
        .collect()
}

/// `indexed_attention_select_score_f32` (`tensor_ops.hip:3570-3591`): per head
/// a serial d-order dot, `relu(dot / sqrt(128))`, heads summed in order.
/// Exact for the dyadic test data (every product and partial sum is exactly
/// representable, so FMA contraction / association cannot change a bit).
/// Model-level twin: `ops.rs:517 qsa_select_blocks`,
/// `reference_forward.rs:963 select_index_tokens`.
fn index_score(q: &[f32], key: &[f32]) -> f32 {
    let mut score = 0.0f32;
    for h in 0..QSA_INDEX_HEADS {
        let mut dot = 0.0f32;
        for d in 0..IDX_DIM {
            dot += q[h * IDX_DIM + d] * key[d];
        }
        score += (dot / (IDX_DIM as f32).sqrt()).max(0.0);
    }
    score
}

/// Blocks ordered (score desc, index asc): the stable lowest-index tie rule.
fn rank_order(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&x, &y| {
        scores[y]
            .partial_cmp(&scores[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });
    idx
}

/// `[rows][SEL_CAP]` selection of `indexed_attention_select_*_batched*`
/// (`tensor_ops.hip:3642-3706` serial twin): top `min(budget, blocks)` blocks
/// by (score desc, index asc) at slot `rank * compress + r`, then the causal
/// tail tokens, `-1` elsewhere. Returns `(selected, scores[row][block])`.
fn select_rows_cpu(
    query: &[f32],
    pooled: &[f32],
    rows: usize,
    blocks: usize,
    position_start: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut selected = vec![-1i32; rows * SEL_CAP];
    let mut all_scores = vec![0.0f32; rows * blocks];
    for r in 0..rows {
        let visible = position_start + r + 1;
        let rbc = (visible / QSA_COMPRESS).min(blocks);
        let q = &query[r * IDX_W..r * IDX_W + IDX_Q_W];
        let scores: Vec<f32> = (0..rbc)
            .map(|b| index_score(q, &pooled[b * IDX_DIM..(b + 1) * IDX_DIM]))
            .collect();
        all_scores[r * blocks..r * blocks + rbc].copy_from_slice(&scores);
        let chosen = BUDGET_BLOCKS.min(rbc);
        let order = rank_order(&scores);
        let row = &mut selected[r * SEL_CAP..(r + 1) * SEL_CAP];
        for (slot, &block) in order.iter().take(chosen).enumerate() {
            for k in 0..QSA_COMPRESS {
                let index = slot * QSA_COMPRESS + k;
                let token = block * QSA_COMPRESS + k;
                if index < SEL_CAP && token < visible {
                    row[index] = token as i32;
                }
            }
        }
        let mut offset = chosen * QSA_COMPRESS;
        for token in rbc * QSA_COMPRESS..visible {
            if offset >= SEL_CAP {
                break;
            }
            row[offset] = token as i32;
            offset += 1;
        }
    }
    (selected, all_scores)
}

/// Dyadic select inputs with abundant exact ties: a palette of 24 repeated
/// pooled rows, all-zero rows (relu → score 0) and fresh rows; the last row's
/// top-`chosen` boundary is forced to sit inside a tie group when blocks
/// exceed the budget. Returns `(query[rows*640], pooled[blocks*128])`.
fn gen_select_data(seed: u64, rows: usize, blocks: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut rng = Rng::new(seed);
    let palette: Vec<Vec<f32>> = (0..24).map(|_| dyadic(&mut rng, IDX_DIM, 32.0)).collect();
    let mut pooled = Vec::with_capacity(blocks * IDX_DIM);
    for b in 0..blocks {
        if b % 5 == 0 {
            pooled.extend(dyadic(&mut rng, IDX_DIM, 32.0));
        } else if b % 11 == 0 {
            pooled.extend(vec![0.0f32; IDX_DIM]);
        } else {
            pooled.extend_from_slice(&palette[rng.below(24)]);
        }
    }
    let mut query = Vec::with_capacity(rows * IDX_W);
    for _ in 0..rows {
        query.extend(dyadic(&mut rng, IDX_Q_W, 16.0));
        query.extend(rng.vec(IDX_DIM, -1.0, 1.0)); // index k half: ignored by select
    }
    let chosen = BUDGET_BLOCKS.min(blocks);
    if chosen < blocks {
        let last_q = &query[(rows - 1) * IDX_W..(rows - 1) * IDX_W + IDX_Q_W];
        let scores_of = |pooled: &[f32]| -> Vec<f32> {
            (0..blocks)
                .map(|b| index_score(last_q, &pooled[b * IDX_DIM..(b + 1) * IDX_DIM]))
                .collect()
        };
        let mut scores = scores_of(&pooled);
        let order = rank_order(&scores);
        let (a, b) = (order[chosen - 1], order[chosen]);
        if scores[a] != scores[b] {
            let row = pooled[a * IDX_DIM..(a + 1) * IDX_DIM].to_vec();
            pooled[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&row);
            scores = scores_of(&pooled);
        }
        let order = rank_order(&scores);
        if scores[order[chosen - 1]] != scores[order[chosen]] {
            return Err("select data: failed to straddle the budget boundary with a tie".into());
        }
    }
    Ok((query, pooled))
}

/// One query row of `qsa_attention_batched` / `_serial`
/// (`tensor_ops.hip:4338-4417`, `4419-4605`; model-level twin
/// `ops.rs:582 qsa_attention`, which multiplies by `sigmoid(gate)` per term
/// instead of dividing the finished sum). Invalid selected entries
/// (`< 0`, `>= full_capacity`, `>= visible`) contribute nothing.
fn attention_row_cpu(
    q_row: &[f32],
    keys: &[f32],
    values: &[f32],
    sel: &[i32],
    visible: usize,
    full_capacity: usize,
) -> Vec<f32> {
    let group = QSA_Q_HEADS / QSA_KV_HEADS;
    let mut out = vec![0.0f32; Q_W];
    for head in 0..QSA_Q_HEADS {
        let kv_head = head / group;
        let q = &q_row[head * 2 * HD..head * 2 * HD + HD];
        let gate = &q_row[head * 2 * HD + HD..head * 2 * HD + 2 * HD];
        let mut toks: Vec<usize> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut max_score = -3.402_823_5e38f32;
        for &t in sel {
            if t < 0 || t as usize >= full_capacity || t as usize >= visible {
                continue;
            }
            let t = t as usize;
            let key = &keys[t * KV_W + kv_head * HD..t * KV_W + kv_head * HD + HD];
            let mut s = 0.0f32;
            for d in 0..HD {
                s += q[d] * key[d];
            }
            let s = s / (HD as f32).sqrt();
            max_score = max_score.max(s);
            toks.push(t);
            scores.push(s);
        }
        let mut den = 0.0f32;
        let mut num = vec![0.0f32; HD];
        for (i, &t) in toks.iter().enumerate() {
            let w = (scores[i] - max_score).exp();
            den += w;
            let v = &values[t * KV_W + kv_head * HD..t * KV_W + kv_head * HD + HD];
            for c in 0..HD {
                num[c] += w * v[c];
            }
        }
        for c in 0..HD {
            out[head * HD + c] = if den > 0.0 {
                num[c] / den / (1.0 + (-gate[c]).exp())
            } else {
                0.0
            };
        }
    }
    out
}

struct AttnInputs {
    end: usize,
    position_start: usize,
    q: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    selected: Vec<i32>,
    want: Vec<f32>,
}

/// Attention inputs at `end` cached tokens: selection from the CPU select twin
/// (`#blocks > 512` ⇒ blocks are dropped), plus injected invalid entries on row
/// 0 (`-1` and a not-yet-visible token) the kernels must skip.
fn attn_inputs(seed: u64, rows: usize, end: usize) -> Result<AttnInputs, String> {
    let blocks = end / QSA_COMPRESS;
    let position_start = end - rows;
    let (query, pooled) = gen_select_data(seed ^ 0x5151, rows, blocks)?;
    let (mut selected, _) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
    let visible0 = position_start + 1;
    selected[3] = -1;
    selected[5] = (visible0 + 1) as i32;
    let mut rng = Rng::new(seed);
    let q = rng.vec(rows * QG_W, -3.0, 3.0);
    let keys = rng.vec(end * KV_W, -1.0, 1.0);
    let values = rng.vec(end * KV_W, -1.0, 1.0);
    let mut want = Vec::with_capacity(rows * Q_W);
    for r in 0..rows {
        let visible = position_start + r + 1;
        let block_count = visible / QSA_COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        let selected_len =
            (chosen * QSA_COMPRESS + visible - block_count * QSA_COMPRESS).min(SEL_CAP);
        want.extend(attention_row_cpu(
            &q[r * QG_W..(r + 1) * QG_W],
            &keys,
            &values,
            &selected[r * SEL_CAP..r * SEL_CAP + selected_len],
            visible,
            end,
        ));
    }
    Ok(AttnInputs { end, position_start, q, keys, values, selected, want })
}

// ── norm + RoPE ───────────────────────────────────────────────────────────────
struct NrConfig {
    name: &'static str,
    heads: usize,
    head_dim: usize,
    head_stride: usize,
    row_stride: usize,
}
const NR_CONFIGS: [NrConfig; 3] = [
    NrConfig { name: "index_q", heads: QSA_INDEX_HEADS, head_dim: IDX_DIM, head_stride: IDX_DIM, row_stride: IDX_W },
    NrConfig { name: "q", heads: QSA_Q_HEADS, head_dim: HD, head_stride: 2 * HD, row_stride: QG_W },
    NrConfig { name: "k", heads: QSA_KV_HEADS, head_dim: HD, head_stride: HD, row_stride: KV_W },
];

fn nr_want(cfg: &NrConfig, input: &[f32], norm: &[f32], rows: usize, pos0: usize, yarn: RopeYarn) -> (Vec<f32>, Vec<bool>) {
    let mut want = input.to_vec();
    let mut touched = vec![false; input.len()];
    for r in 0..rows {
        for h in 0..cfg.heads {
            let start = r * cfg.row_stride + h * cfg.head_stride;
            let out = norm_rope_head(&input[start..start + cfg.head_dim], norm, pos0 + r, ROTARY, yarn);
            want[start..start + cfg.head_dim].copy_from_slice(&out);
            for t in &mut touched[start..start + cfg.head_dim] {
                *t = true;
            }
        }
    }
    (want, touched)
}

#[allow(clippy::too_many_arguments)]
fn norm_rope_launch(
    gpu: &mut Gpu,
    values: &GpuTensor,
    norm: &GpuTensor,
    rows: usize,
    position: usize,
    yarn: RopeYarn,
    heads: usize,
    head_dim: usize,
    head_stride: usize,
    row_stride: usize,
) -> Result<(), String> {
    indexed_attention_norm_rope_batch(
        gpu,
        &IndexedAttentionNormRopeBatch {
            values,
            norm,
            rows,
            row_stride,
            heads,
            head_dim,
            head_stride,
            position_start: position,
            rotary_dim: ROTARY,
            yarn,
        },
    )
    .map_err(e)
}

/// `indexed_attention_norm_rope_batch` (the route's norm+RoPE for rows > 8 in
/// decode and for every prefill chunk on gfx942, where `route_rows == 8192`).
/// Positions stay below 40: the CPU `powf`/`sin`/`cos` differ from the device
/// ocml functions by ~1 ulp of the angle, which the frozen F32_ACCUM tol only
/// absorbs at small angles.
fn case_norm_rope_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for cfg in &NR_CONFIGS {
        let mut rng = Rng::new(0x7101 + cfg.heads as u64);
        let norm = rng.vec_bf16_exact(cfg.head_dim, -0.5, 0.5);
        let norm_t = bf16_vec(gpu, &norm)?;
        for (yname, yarn) in yarns() {
            for rows in ROW_COUNTS {
                let pos0 = 3;
                let input = rng.vec(rows * cfg.row_stride, -2.0, 2.0);
                let values = up_f32(gpu, &input)?;
                norm_rope_launch(
                    gpu, &values, &norm_t, rows, pos0, yarn, cfg.heads, cfg.head_dim, cfg.head_stride,
                    cfg.row_stride,
                )?;
                sync(gpu)?;
                let got = down_f32(gpu, &values)?;
                let (want, touched) = nr_want(cfg, &input, &norm, rows, pos0, yarn);
                checks.push(cmp_split(
                    &format!("{}/{yname}/rows{rows}", cfg.name),
                    &got,
                    &want,
                    F32_ACCUM,
                    |i| !touched[i],
                ));
                free_all(gpu, vec![values])?;
            }
        }
        free_all(gpu, vec![norm_t])?;
    }
    all(checks)
}

/// `indexed_attention_norm_rope` (single row; used by parity harnesses, not by
/// the C1 route): index-q and k configs at position 5.
fn case_norm_rope_single(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for cfg in [&NR_CONFIGS[0], &NR_CONFIGS[2]] {
        let mut rng = Rng::new(0x7201 + cfg.heads as u64);
        let norm = rng.vec_bf16_exact(cfg.head_dim, -0.5, 0.5);
        let norm_t = bf16_vec(gpu, &norm)?;
        let input = rng.vec(cfg.row_stride, -2.0, 2.0);
        let values = up_f32(gpu, &input)?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &values,
                norm: &norm_t,
                heads: cfg.heads,
                head_dim: cfg.head_dim,
                head_stride: cfg.head_stride,
                position: 5,
                rotary_dim: ROTARY,
                yarn: RopeYarn::OFF,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &values)?;
        let (want, touched) = nr_want(cfg, &input, &norm, 1, 5, RopeYarn::OFF);
        checks.push(cmp_split(cfg.name, &got, &want, F32_ACCUM, |i| !touched[i]));
        free_all(gpu, vec![values, norm_t])?;
    }
    all(checks)
}

// ── cache append / index key append ──────────────────────────────────────────
/// `indexed_attention_cache_append_f32_batched` (`tensor_ops.hip:3377`): pure
/// copy to `position_start + row`; bit-exact incl. untouched rows. 600 rows
/// exceeds every row threshold the route sees (512).
fn case_cache_append_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in [1usize, 7, 33, 600] {
        let mut rng = Rng::new(0x7301 + rows as u64);
        let key = rng.vec(rows * KV_W, -2.0, 2.0);
        let value = rng.vec(rows * KV_W, -2.0, 2.0);
        let cap = (pos0 + rows) * KV_W;
        let (k_t, v_t) = (up_f32(gpu, &key)?, up_f32(gpu, &value)?);
        let (fk, fv) = (nan_f32(gpu, cap)?, nan_f32(gpu, cap)?);
        indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k_t,
                value: &v_t,
                full_keys: &fk,
                full_values: &fv,
                rows,
                position_start: pos0,
                kv_heads: QSA_KV_HEADS,
                head_dim: HD,
                format: QsaKvFormat::F32,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let sentinel = f32::from_bits(0x7FC0_DEAD);
        let mut want_k = vec![sentinel; cap];
        let mut want_v = vec![sentinel; cap];
        want_k[pos0 * KV_W..].copy_from_slice(&key);
        want_v[pos0 * KV_W..].copy_from_slice(&value);
        checks.push(exact_f32(&format!("rows{rows}/keys"), &down_f32(gpu, &fk)?, &want_k));
        checks.push(exact_f32(&format!("rows{rows}/values"), &down_f32(gpu, &fv)?, &want_v));
        free_all(gpu, vec![k_t, v_t, fk, fv])?;
    }
    all(checks)
}

/// `indexed_attention_cache_append_f32` (single row; parity harnesses only).
fn case_cache_append_single(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x7401);
    let key = rng.vec(KV_W, -2.0, 2.0);
    let value = rng.vec(KV_W, -2.0, 2.0);
    let position = 7;
    let cap = (position + 1) * KV_W;
    let (k_t, v_t) = (up_f32(gpu, &key)?, up_f32(gpu, &value)?);
    let (fk, fv) = (nan_f32(gpu, cap)?, nan_f32(gpu, cap)?);
    indexed_attention_cache_append(
        gpu,
        &IndexedAttentionCacheAppend {
            key: &k_t,
            value: &v_t,
            full_keys: &fk,
            full_values: &fv,
            position,
            kv_width: KV_W,
        },
    )
    .map_err(e)?;
    sync(gpu)?;
    let sentinel = f32::from_bits(0x7FC0_DEAD);
    let mut want_k = vec![sentinel; cap];
    let mut want_v = vec![sentinel; cap];
    want_k[position * KV_W..].copy_from_slice(&key);
    want_v[position * KV_W..].copy_from_slice(&value);
    let r = all(vec![
        exact_f32("keys", &down_f32(gpu, &fk)?, &want_k),
        exact_f32("values", &down_f32(gpu, &fv)?, &want_v),
    ]);
    free_all(gpu, vec![k_t, v_t, fk, fv])?;
    r
}

/// Index-key specials: RNE ties both ways, just below/above a tie, ±Inf (kept
/// unrounded in place), signed zeros.
fn specials(include_nan: bool) -> Vec<f32> {
    let mut v = vec![
        f32::from_bits(0x3F80_8000),
        f32::from_bits(0x3F81_8000),
        f32::from_bits(0x3F80_7FFF),
        f32::from_bits(0x3F80_8001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
    ];
    if include_nan {
        v.push(f32::from_bits(0x7FC0_1234));
    }
    v
}

/// `indexed_attention_index_key_append_bf16_batched` (`tensor_ops.hip:3441`).
/// NOT on the gfx942 route: only a BF16 raw-key arena (`QsaKvFormat::{Fp8,Q8}`,
/// refused on gfx942) takes it; F32 state uses the round-trip + row-copy pair
/// (cases below). Exercised here because the launcher/kernel are portable.
fn case_index_key_append(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in [1usize, 7, 33, 600] {
        let mut rng = Rng::new(0x7501 + rows as u64);
        let mut input = rng.vec(rows * IDX_W, -3.0, 3.0);
        for (i, s) in specials(false).into_iter().enumerate() {
            input[IDX_Q_W + i] = s; // row 0's index-k head
        }
        let rows_t = up_f32(gpu, &input)?;
        let raw = bf16_vec(gpu, &vec![f32::NAN; (pos0 + rows) * IDX_DIM])?;
        indexed_attention_index_key_append_batch(
            gpu,
            &IndexedAttentionIndexKeyAppendBatch {
                index_rows: &rows_t,
                raw_index_keys: &raw,
                rows,
                index_q_width: IDX_Q_W,
                index_kv_width: IDX_DIM,
                position_start: pos0,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let mut want_rows = input.clone();
        let mut want_raw = vec![f32_to_bf16_bits(f32::NAN); (pos0 + rows) * IDX_DIM];
        for r in 0..rows {
            for i in 0..IDX_DIM {
                let idx = r * IDX_W + IDX_Q_W + i;
                let v = input[idx];
                let rounded = if v.is_finite() { bf16_trunc_rne(v) } else { v };
                want_rows[idx] = rounded;
                want_raw[(pos0 + r) * IDX_DIM + i] = f32_to_bf16_bits(rounded);
            }
        }
        checks.push(exact_f32(&format!("rows{rows}/index_rows"), &down_f32(gpu, &rows_t)?, &want_rows));
        checks.push(exact_bf16(&format!("rows{rows}/raw"), &down_bf16(gpu, &raw)?, &want_raw));
        free_all(gpu, vec![rows_t, raw])?;
    }
    all(checks)
}

/// `bf16_round_trip_f32_strided` (`qwen4_bf16_round_trip.hip`), the route's F32
/// index-k boundary (offset 512, width 128, stride 640). Non-finite values
/// pass through unchanged.
fn case_round_trip_strided(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for rows in ROW_COUNTS {
        let mut rng = Rng::new(0x7601 + rows as u64);
        let mut input = rng.vec(rows * IDX_W, -3.0, 3.0);
        for (i, s) in specials(true).into_iter().enumerate() {
            input[IDX_Q_W + i] = s;
            input[i] = s; // outside the rounded window: must stay untouched
        }
        let x = up_f32(gpu, &input)?;
        gpu.bf16_round_trip_f32_strided(&x, rows, IDX_Q_W, IDX_W, IDX_DIM).map_err(e)?;
        sync(gpu)?;
        let mut want = input.clone();
        for r in 0..rows {
            for i in 0..IDX_DIM {
                let idx = r * IDX_W + IDX_Q_W + i;
                if want[idx].is_finite() {
                    want[idx] = bf16_trunc_rne(want[idx]);
                }
            }
        }
        checks.push(exact_f32(&format!("rows{rows}"), &down_f32(gpu, &x)?, &want));
        free_all(gpu, vec![x])?;
    }
    all(checks)
}

/// `copy_rows_strided_f32` as the route calls it to append the index key rows
/// to the F32 raw-key arena (`layer_ops.rs` `execute_indexed_attention_hc_impl`): src = a view at
/// offset 512 of the `[rows][640]` index batch, dst column offset
/// `position * 128`, vec4 fast path (everything 16-byte aligned).
fn case_copy_rows_strided(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in ROW_COUNTS {
        let mut rng = Rng::new(0x7701 + rows as u64);
        let input = rng.vec(rows * IDX_W, -3.0, 3.0);
        let batch = up_f32(gpu, &input)?;
        let keys = batch.sub_offset(IDX_Q_W, rows * IDX_W - IDX_Q_W);
        let raw = nan_f32(gpu, (pos0 + rows) * IDX_DIM)?;
        gpu.copy_rows_strided_f32(&keys, &raw, rows, IDX_DIM, IDX_W, IDX_DIM, pos0 * IDX_DIM, Some(IDX_DIM))
            .map_err(e)?;
        sync(gpu)?;
        let mut want = vec![f32::from_bits(0x7FC0_DEAD); (pos0 + rows) * IDX_DIM];
        for r in 0..rows {
            let src = r * IDX_W + IDX_Q_W;
            want[(pos0 + r) * IDX_DIM..(pos0 + r + 1) * IDX_DIM].copy_from_slice(&input[src..src + IDX_DIM]);
        }
        checks.push(exact_f32(&format!("rows{rows}"), &down_f32(gpu, &raw)?, &want));
        free_all(gpu, vec![batch, raw])?; // `keys` is a view: not freed
    }
    all(checks)
}

// ── decode prologue ───────────────────────────────────────────────────────────
struct PrologueWant {
    index_row: Vec<f32>,
    qgate: Vec<f32>,
    keys: Vec<f32>,
    full_keys: Vec<f32>,
    full_values: Vec<f32>,
    raw: Vec<f32>,
}

/// `qsa_decode_prologue<QSA_KV_F32>` (`tensor_ops.hip:3252-3346`): the three
/// norm+RoPE stages, K/V cache append at `position + row`, index-k BF16 round
/// trip and raw-key copy.
#[allow(clippy::too_many_arguments)]
fn prologue_want(
    index_row: &[f32],
    qgate: &[f32],
    keys: &[f32],
    values: &[f32],
    iq_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    rows: usize,
    position: usize,
    yarn: RopeYarn,
) -> PrologueWant {
    let mut w = PrologueWant {
        index_row: index_row.to_vec(),
        qgate: qgate.to_vec(),
        keys: keys.to_vec(),
        full_keys: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * KV_W],
        full_values: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * KV_W],
        raw: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * IDX_DIM],
    };
    for r in 0..rows {
        let pos = position + r;
        for h in 0..QSA_INDEX_HEADS {
            let s = r * IDX_W + h * IDX_DIM;
            let o = norm_rope_head(&index_row[s..s + IDX_DIM], iq_norm, pos, ROTARY, yarn);
            w.index_row[s..s + IDX_DIM].copy_from_slice(&o);
        }
        for h in 0..QSA_Q_HEADS {
            let s = r * QG_W + h * 2 * HD;
            let o = norm_rope_head(&qgate[s..s + HD], q_norm, pos, ROTARY, yarn);
            w.qgate[s..s + HD].copy_from_slice(&o);
        }
        for h in 0..QSA_KV_HEADS {
            let s = r * KV_W + h * HD;
            let o = norm_rope_head(&keys[s..s + HD], k_norm, pos, ROTARY, yarn);
            w.keys[s..s + HD].copy_from_slice(&o);
        }
        let rolled: Vec<f32> = w.keys[r * KV_W..(r + 1) * KV_W].to_vec();
        w.full_keys[pos * KV_W..(pos + 1) * KV_W].copy_from_slice(&rolled);
        w.full_values[pos * KV_W..(pos + 1) * KV_W].copy_from_slice(&values[r * KV_W..(r + 1) * KV_W]);
        for i in 0..IDX_DIM {
            let idx = r * IDX_W + IDX_Q_W + i;
            let v = index_row[idx];
            let v = if v.is_finite() { bf16_trunc_rne(v) } else { v };
            w.index_row[idx] = v;
            w.raw[pos * IDX_DIM + i] = v;
        }
    }
    w
}

struct PrologueGpu {
    index_row: GpuTensor,
    qgate: GpuTensor,
    keys: GpuTensor,
    values: GpuTensor,
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw: GpuTensor,
}

impl PrologueGpu {
    fn new(gpu: &mut Gpu, i: &[Vec<f32>; 4], rows: usize, position: usize) -> Result<Self, String> {
        Ok(Self {
            index_row: up_f32(gpu, &i[0])?,
            qgate: up_f32(gpu, &i[1])?,
            keys: up_f32(gpu, &i[2])?,
            values: up_f32(gpu, &i[3])?,
            full_keys: nan_f32(gpu, (position + rows) * KV_W)?,
            full_values: nan_f32(gpu, (position + rows) * KV_W)?,
            raw: nan_f32(gpu, (position + rows) * IDX_DIM)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        free_all(
            gpu,
            vec![self.index_row, self.qgate, self.keys, self.values, self.full_keys, self.full_values, self.raw],
        )
    }
}

/// `indexed_attention_decode_prologue` (route: decode / verify rows <= 8, via
/// `layer_ops.rs` `execute_indexed_attention_hc_impl`; rows 33 runs the same kernel past the route's
/// `route_rows <= 8` gate, which on gfx942 prefill is always false because the
/// invariant profile pins `route_rows == 8192`), the append prologue
/// (`heads == 0`, MTP) and the bitwise claim "prologue == the six launches it
/// replaces" (norm_rope_batch x3 + round trip + copy_rows + cache_append_batch,
/// i.e. the prefill chain).
fn case_decode_prologue(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let position = 3;
    for (yname, yarn) in yarns() {
        for rows in ROW_COUNTS {
            let mut rng = Rng::new(0x7801 + rows as u64);
            let iq_norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
            let q_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let k_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let inputs = [
                rng.vec(rows * IDX_W, -2.0, 2.0),
                rng.vec(rows * QG_W, -2.0, 2.0),
                rng.vec(rows * KV_W, -2.0, 2.0),
                rng.vec(rows * KV_W, -2.0, 2.0),
            ];
            let (iq_t, q_t, k_t) =
                (bf16_vec(gpu, &iq_norm)?, bf16_vec(gpu, &q_norm)?, bf16_vec(gpu, &k_norm)?);
            let label = format!("{yname}/rows{rows}");

            // Fused decode prologue.
            let p = PrologueGpu::new(gpu, &inputs, rows, position)?;
            indexed_attention_decode_prologue(
                gpu,
                &IndexedAttentionDecodePrologue {
                    index_row: &p.index_row,
                    qgate: &p.qgate,
                    keys: &p.keys,
                    values: &p.values,
                    full_keys: &p.full_keys,
                    full_values: &p.full_values,
                    raw_index_keys: &p.raw,
                    index_q_norm: &iq_t,
                    q_norm: &q_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    heads: QSA_Q_HEADS,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: QsaKvFormat::F32,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let got_index = down_f32(gpu, &p.index_row)?;
            let got_qgate = down_f32(gpu, &p.qgate)?;
            let got_keys = down_f32(gpu, &p.keys)?;
            let got_fk = down_f32(gpu, &p.full_keys)?;
            let got_fv = down_f32(gpu, &p.full_values)?;
            let got_raw = down_f32(gpu, &p.raw)?;

            // CPU twin.
            let w = prologue_want(
                &inputs[0], &inputs[1], &inputs[2], &inputs[3], &iq_norm, &q_norm, &k_norm, rows,
                position, yarn,
            );
            checks.push(cmp_split(&format!("{label}/index_row"), &got_index, &w.index_row, F32_ACCUM, |i| i % IDX_W >= IDX_Q_W));
            checks.push(cmp_split(&format!("{label}/qgate"), &got_qgate, &w.qgate, F32_ACCUM, |i| i % (2 * HD) >= HD));
            checks.push(close_f32(&format!("{label}/keys"), &got_keys, &w.keys, F32_ACCUM));
            checks.push(cmp_split(&format!("{label}/full_keys"), &got_fk, &w.full_keys, F32_ACCUM, |i| i < position * KV_W));
            checks.push(exact_f32(&format!("{label}/full_values"), &got_fv, &w.full_values));
            checks.push(exact_f32(&format!("{label}/raw_index_keys"), &got_raw, &w.raw));

            // The prefill chain the prologue replaces (bitwise).
            let c = PrologueGpu::new(gpu, &inputs, rows, position)?;
            norm_rope_launch(gpu, &c.index_row, &iq_t, rows, position, yarn, QSA_INDEX_HEADS, IDX_DIM, IDX_DIM, IDX_W)?;
            gpu.bf16_round_trip_f32_strided(&c.index_row, rows, IDX_Q_W, IDX_W, IDX_DIM).map_err(e)?;
            let keys_view = c.index_row.sub_offset(IDX_Q_W, rows * IDX_W - IDX_Q_W);
            gpu.copy_rows_strided_f32(&keys_view, &c.raw, rows, IDX_DIM, IDX_W, IDX_DIM, position * IDX_DIM, Some(IDX_DIM))
                .map_err(e)?;
            norm_rope_launch(gpu, &c.qgate, &q_t, rows, position, yarn, QSA_Q_HEADS, HD, 2 * HD, QG_W)?;
            norm_rope_launch(gpu, &c.keys, &k_t, rows, position, yarn, QSA_KV_HEADS, HD, HD, KV_W)?;
            indexed_attention_cache_append_batch(
                gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &c.keys,
                    value: &c.values,
                    full_keys: &c.full_keys,
                    full_values: &c.full_values,
                    rows,
                    position_start: position,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    format: QsaKvFormat::F32,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(exact_f32(&format!("{label}/chain==prologue index_row"), &down_f32(gpu, &c.index_row)?, &got_index));
            checks.push(exact_f32(&format!("{label}/chain==prologue qgate"), &down_f32(gpu, &c.qgate)?, &got_qgate));
            checks.push(exact_f32(&format!("{label}/chain==prologue keys"), &down_f32(gpu, &c.keys)?, &got_keys));
            checks.push(exact_f32(&format!("{label}/chain==prologue full_keys"), &down_f32(gpu, &c.full_keys)?, &got_fk));
            checks.push(exact_f32(&format!("{label}/chain==prologue full_values"), &down_f32(gpu, &c.full_values)?, &got_fv));
            checks.push(exact_f32(&format!("{label}/chain==prologue raw"), &down_f32(gpu, &c.raw)?, &got_raw));
            c.free(gpu)?;

            // Append prologue (heads == 0): same kernel, same bytes.
            let a = PrologueGpu::new(gpu, &inputs, rows, position)?;
            indexed_attention_append_prologue(
                gpu,
                &IndexedAttentionAppendPrologue {
                    index_row: &a.index_row,
                    keys: &a.keys,
                    values: &a.values,
                    full_keys: &a.full_keys,
                    full_values: &a.full_values,
                    raw_index_keys: &a.raw,
                    index_q_norm: &iq_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: QsaKvFormat::F32,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(exact_f32(&format!("{label}/append==decode index_row"), &down_f32(gpu, &a.index_row)?, &got_index));
            checks.push(exact_f32(&format!("{label}/append==decode keys"), &down_f32(gpu, &a.keys)?, &got_keys));
            checks.push(exact_f32(&format!("{label}/append==decode full_keys"), &down_f32(gpu, &a.full_keys)?, &got_fk));
            checks.push(exact_f32(&format!("{label}/append==decode full_values"), &down_f32(gpu, &a.full_values)?, &got_fv));
            checks.push(exact_f32(&format!("{label}/append==decode raw"), &down_f32(gpu, &a.raw)?, &got_raw));
            a.free(gpu)?;
            p.free(gpu)?;
            free_all(gpu, vec![iq_t, q_t, k_t])?;
        }
    }
    all(checks)
}

// ── pool + RoPE ───────────────────────────────────────────────────────────────
/// Shared body of the two `pool_rope` cases. Incremental launches (the route:
/// `execute_indexed_attention_hc_impl`) pool only blocks `[position_start/4, (position_start +
/// rows)/4)` and must leave every other pooled block untouched; the full
/// (non-incremental) launch pools all 700 blocks. Context 2800 tokens ⇒ 700
/// blocks > 512 budget blocks. Output is BF16-quantised, so the frozen
/// BF16_INPUT_F32_ACCUM tol applies (it also absorbs the device `rsqrtf` /
/// `powf` vs CPU differences at positions up to 2800).
fn pool_rope_case(gpu: &mut Gpu, bf16_arena: bool) -> Result<String, String> {
    let end = 2800usize;
    let blocks = end / QSA_COMPRESS;
    let mut rng = Rng::new(if bf16_arena { 0x7A02 } else { 0x7A01 });
    let raw = rng.vec_bf16_exact(end * IDX_DIM, -2.0, 2.0);
    let norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
    let norm_t = bf16_vec(gpu, &norm)?;
    let raw_t = if bf16_arena { bf16_vec(gpu, &raw)? } else { up_f32(gpu, &raw)? };
    let sentinel = if bf16_arena { f32::from_bits(0x7FC0_0000) } else { f32::from_bits(0x7FC0_DEAD) };
    let new_pooled = |gpu: &mut Gpu| -> Result<GpuTensor, String> {
        if bf16_arena {
            bf16_vec(gpu, &vec![f32::NAN; POOLED_CAP * IDX_DIM])
        } else {
            nan_f32(gpu, POOLED_CAP * IDX_DIM)
        }
    };
    let download = |gpu: &Gpu, t: &GpuTensor| -> Result<Vec<f32>, String> {
        if bf16_arena {
            Ok(down_bf16(gpu, t)?.into_iter().map(bf16_bits_to_f32).collect())
        } else {
            down_f32(gpu, t)
        }
    };
    let mut checks = Vec::new();
    for (yname, yarn) in yarns() {
        let block_row = |b: usize| {
            pool_rope_block(
                &raw[b * QSA_COMPRESS * IDX_DIM..(b + 1) * QSA_COMPRESS * IDX_DIM],
                &norm,
                b,
                yarn,
            )
        };
        for rows in ROW_COUNTS {
            let position_start = end - rows;
            let first = position_start / QSA_COMPRESS;
            let pooled = new_pooled(gpu)?;
            indexed_attention_pool_rope_incremental(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &raw_t,
                    pooled: &pooled,
                    norm: Some(&norm_t),
                    block_count: blocks,
                    compress: QSA_COMPRESS,
                    index_dim: IDX_DIM,
                    position: Some(QsaPositionBinding { position_start, rows }),
                    grid_bound: POOLED_CAP,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let mut want = vec![sentinel; POOLED_CAP * IDX_DIM];
            for b in first..blocks {
                want[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&block_row(b));
            }
            let got = download(gpu, &pooled)?;
            let (lo, hi) = (first * IDX_DIM, blocks * IDX_DIM);
            checks.push(cmp_split(
                &format!("{yname}/incremental rows{rows} (blocks {first}..{blocks})"),
                &got,
                &want,
                BF16_INPUT_F32_ACCUM,
                |i| i < lo || i >= hi,
            ));
            free_all(gpu, vec![pooled])?;
        }
        let pooled = new_pooled(gpu)?;
        indexed_attention_pool_rope(
            gpu,
            &IndexedAttentionPoolRope {
                raw_keys: &raw_t,
                pooled: &pooled,
                norm: Some(&norm_t),
                block_count: blocks,
                compress: QSA_COMPRESS,
                index_dim: IDX_DIM,
                position: None,
                grid_bound: blocks,
                yarn,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let mut want = vec![sentinel; POOLED_CAP * IDX_DIM];
        for b in 0..blocks {
            want[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&block_row(b));
        }
        let got = download(gpu, &pooled)?;
        let hi = blocks * IDX_DIM;
        checks.push(cmp_split(
            &format!("{yname}/full {blocks} blocks"),
            &got,
            &want,
            BF16_INPUT_F32_ACCUM,
            |i| i >= hi,
        ));
        if !bf16_arena {
            // F32 arena values are BF16-exact by construction.
            let bad = got[..hi].iter().filter(|&&v| v != bf16_round(v)).count();
            checks.push(if bad == 0 {
                Ok("F32 pooled values are BF16-exact".to_string())
            } else {
                Err(format!("{bad} pooled F32 values are not BF16-exact"))
            });
        }
        free_all(gpu, vec![pooled])?;
    }
    free_all(gpu, vec![raw_t, norm_t])?;
    all(checks)
}

/// `indexed_attention_pool_rope_f32` — the F32 arena the gfx942 route uses.
fn case_pool_rope_f32(gpu: &mut Gpu) -> Result<String, String> {
    pool_rope_case(gpu, false)
}

/// `indexed_attention_pool_rope_bf16` — BF16 arena (quantised-KV states only;
/// not reachable on gfx942).
fn case_pool_rope_bf16(gpu: &mut Gpu) -> Result<String, String> {
    pool_rope_case(gpu, true)
}

// ── select (route) ────────────────────────────────────────────────────────────
/// Public `indexed_attention_select_batch_mirrored`, exactly as
/// `execute_indexed_attention_hc_impl` calls it. On gfx942 that is
/// `indexed_attention_select_{f32,bf16}_batched_serial` (1 thread per row);
/// elsewhere the launcher's own arch-dependent kernels. Contexts: 2802 tokens
/// (700 blocks > 512 budget; last row's boundary sits in a tie group; partial
/// 2-token tail) and 1602 tokens (400 blocks: every block selected). Selected
/// ids and the final-row mirror are compared bit-exactly with the stable
/// (score desc, block asc) twin; scores come from dyadic data so the twin is
/// exact regardless of FMA contraction.
fn select_route(gpu: &mut Gpu, bf16: bool) -> Result<String, String> {
    let mut checks = Vec::new();
    for end in [2802usize, 1602] {
        let blocks = end / QSA_COMPRESS;
        for rows in ROW_COUNTS {
            let position_start = end - rows;
            let (query, pooled) = gen_select_data(0x7B00 + end as u64 + rows as u64, rows, blocks)?;
            let (want, _) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
            let query_t = up_f32(gpu, &query)?;
            let pooled_t = if bf16 { bf16_vec(gpu, &pooled)? } else { up_f32(gpu, &pooled)? };
            let selected = raw_i32(gpu, &vec![-7777; rows * SEL_CAP])?;
            let mirror = raw_i32(gpu, &vec![-7777; SEL_CAP])?;
            let persisted = indexed_attention_select_batch_mirrored(
                gpu,
                &IndexedAttentionSelectBatch {
                    query: &query_t,
                    pooled: &pooled_t,
                    selected: &selected,
                    rows,
                    query_row_stride: IDX_W,
                    block_count: blocks,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    budget_blocks: BUDGET_BLOCKS,
                    compress: QSA_COMPRESS,
                    position_start,
                    capacity: SEL_CAP,
                    shape_blocks: POOLED_CAP,
                },
                &mirror,
            )
            .map_err(e)?;
            sync(gpu)?;
            let label = format!("end{end}/rows{rows}");
            checks.push(exact_ints(&format!("{label}/selected"), &down_i32(gpu, &selected)?, &want));
            let want_mirror = if persisted {
                want[(rows - 1) * SEL_CAP..].to_vec()
            } else {
                vec![-7777; SEL_CAP]
            };
            checks.push(exact_ints(
                &format!("{label}/mirror(persisted={persisted})"),
                &down_i32(gpu, &mirror)?,
                &want_mirror,
            ));
            free_all(gpu, vec![query_t, pooled_t, selected, mirror])?;
        }
    }
    all(checks)
}

fn case_select_f32(gpu: &mut Gpu) -> Result<String, String> {
    select_route(gpu, false)
}

/// BF16 pooled arena (quantised-KV states): not on the gfx942 route.
fn case_select_bf16(gpu: &mut Gpu) -> Result<String, String> {
    select_route(gpu, true)
}

// ── attention (route) ─────────────────────────────────────────────────────────
fn attn_compare(gpu: &Gpu, out: &GpuTensor, inp: &AttnInputs, label: &str) -> Result<String, String> {
    close_f32(label, &down_f32(gpu, out)?, &inp.want, F32_ACCUM)
}

/// Public `indexed_attention_attention_batch_with_arena(.., None)` as the route
/// calls it (`execute_indexed_attention_hc_impl`): on gfx942
/// `indexed_attention_attention_f32_batched_serial` (no hg12/hg4/parallel, no
/// WMMA, no arena). 2802 ctx ⇒ 700 blocks, 512 chosen + tail ⇒ selected_len
/// 2048..2050 (> any 256-row hg12 tile); 402 ctx keeps every block. Row 0
/// carries injected `-1` / not-yet-visible entries the kernel must skip.
/// F32_ACCUM: the CPU twin sums in the kernel's d / selected-row order but
/// `expf` and FMA contraction differ in the last ulps.
fn case_attention_batch(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for rows in ROW_COUNTS {
            let inp = attn_inputs(0x7C00 + end as u64 + rows as u64, rows, end)?;
            let q = up_f32(gpu, &inp.q)?;
            let (k, v) = (up_f32(gpu, &inp.keys)?, up_f32(gpu, &inp.values)?);
            let sel = raw_i32(gpu, &inp.selected)?;
            let out = nan_f32(gpu, rows * Q_W)?;
            indexed_attention_attention_batch_with_arena(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &q,
                    full_keys: &k,
                    full_values: &v,
                    selected: &sel,
                    output: &out,
                    rows,
                    position_start: inp.position_start,
                    n_heads: QSA_Q_HEADS,
                    n_kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    budget_blocks: BUDGET_BLOCKS,
                    compress: QSA_COMPRESS,
                    capacity: SEL_CAP,
                    full_capacity: end,
                    format: QsaKvFormat::F32,
                    shape_selected: SEL_CAP,
                },
                None,
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(attn_compare(gpu, &out, &inp, &format!("end{end}/rows{rows}")));
            free_all(gpu, vec![q, k, v, sel, out])?;
        }
    }
    all(checks)
}

// ── reuse selection + single-row attention (MTP route) ───────────────────────
struct ReuseScenario {
    name: &'static str,
    prior: Vec<i32>,
    selected_len: usize,
    position: usize,
}

fn reuse_scenarios() -> Vec<ReuseScenario> {
    let mut rng = Rng::new(0x7D01);
    let position = 3000usize;
    let mut a: Vec<i32> = (0..600).map(|_| rng.below(position + 10) as i32 - 1).collect();
    a.retain(|&t| t != position as i32);
    a[5] = position as i32 + 4; // above position: dropped
    let a_len = a.len();
    let mut b: Vec<i32> = (0..500).map(|_| rng.below(position - 1) as i32).collect();
    b[123] = position as i32; // already present: not appended
    b[7] = -1;
    let b_len = b.len();
    let c: Vec<i32> = (0..SEL_CAP).map(|i| (i as i32 * 3) % position as i32).collect();
    vec![
        ReuseScenario { name: "append_current", prior: a, selected_len: a_len, position },
        ReuseScenario { name: "current_present", prior: b, selected_len: b_len, position },
        ReuseScenario { name: "full_capacity_no_room", prior: c, selected_len: SEL_CAP, position },
        ReuseScenario { name: "empty_prior", prior: vec![], selected_len: 0, position },
    ]
}

fn reuse_want(s: &ReuseScenario) -> (Vec<i32>, i32) {
    let n = s.selected_len.min(SEL_CAP);
    let mut kept: Vec<i32> = s.prior[..n]
        .iter()
        .copied()
        .filter(|&t| t >= 0 && t <= s.position as i32)
        .collect();
    if !kept.contains(&(s.position as i32)) && kept.len() < SEL_CAP {
        kept.push(s.position as i32);
    }
    let count = kept.len() as i32;
    kept.resize(SEL_CAP, -1);
    (kept, count)
}

/// `indexed_attention_reuse_selection` (`tensor_ops.hip:4280`): stable
/// in-place compaction (keep `0 <= t <= position`, append `position` unless
/// present or full), `-1` fill, length scalar. Width-32 `__shfl_up` scan inside
/// 8 software waves of a 256-thread block.
fn case_reuse_selection(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for s in reuse_scenarios() {
        let mut buf = s.prior.clone();
        buf.resize(SEL_CAP, 12345); // beyond selected_len: garbage the kernel must overwrite
        let selected = raw_i32(gpu, &buf)?;
        let len_out = raw_i32(gpu, &[-9999])?;
        indexed_attention_reuse_selection(
            gpu,
            &IndexedAttentionReuseSelection {
                selected: &selected,
                selected_len: s.selected_len,
                position: s.position,
                capacity: SEL_CAP,
                selected_len_out: &len_out,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let (want, want_len) = reuse_want(&s);
        checks.push(exact_ints(&format!("{}/selected", s.name), &down_i32(gpu, &selected)?, &want));
        checks.push(exact_ints(&format!("{}/len", s.name), &down_i32(gpu, &len_out)?, &[want_len][..]));
        free_all(gpu, vec![selected, len_out])?;
    }
    all(checks)
}

/// `indexed_attention_attention` (single query row, explicit `selected_len`;
/// the MTP reuse route in `execute_indexed_attention_hc_impl`): on gfx942
/// `indexed_attention_attention_f32_serial`. Selection = scenario A of the
/// reuse case after compaction (arbitrary token order, `-1` fill beyond
/// `selected_len`), cache of 3011 rows.
fn case_attention_single(gpu: &mut Gpu) -> Result<String, String> {
    let scenarios = reuse_scenarios();
    let s = &scenarios[0];
    let (sel, count) = reuse_want(s);
    let full_capacity = s.position + 11;
    let mut rng = Rng::new(0x7E01);
    let q = rng.vec(QG_W, -3.0, 3.0);
    let keys = rng.vec(full_capacity * KV_W, -1.0, 1.0);
    let values = rng.vec(full_capacity * KV_W, -1.0, 1.0);
    let want = attention_row_cpu(&q, &keys, &values, &sel[..count as usize], full_capacity, full_capacity);
    let (q_t, k_t, v_t) = (up_f32(gpu, &q)?, up_f32(gpu, &keys)?, up_f32(gpu, &values)?);
    let sel_t = raw_i32(gpu, &sel)?;
    let out = nan_f32(gpu, Q_W)?;
    indexed_attention_attention(
        gpu,
        &IndexedAttentionAttention {
            q_with_gate: &q_t,
            full_keys: &k_t,
            full_values: &v_t,
            selected: &sel_t,
            output: &out,
            n_heads: QSA_Q_HEADS,
            n_kv_heads: QSA_KV_HEADS,
            head_dim: HD,
            selected_len: count as usize,
            full_capacity,
            format: QsaKvFormat::F32,
        },
    )
    .map_err(e)?;
    sync(gpu)?;
    let r = close_f32("single-row attention", &down_f32(gpu, &out)?, &want, F32_ACCUM);
    free_all(gpu, vec![q_t, k_t, v_t, sel_t, out])?;
    r
}

// ── gates / unreachable-kernel proofs ────────────────────────────────────────
/// Pure + arch-conditional evidence that the WMMA / quantised-KV / arena routes
/// are unreachable on gfx942: `kv_mode::resolve_qwen4` (fp8/q8 refused, auto/bf16
/// = F32 state), `has_gfx11_plus_simt` / `has_wmma_w32` / `is_gfx1151` false,
/// the F16-arena and gathered-WMMA predicates false, and the fp8/q8 K/V
/// launchers returning `Err` (`check_qsa_format`) before any launch.
fn case_gates(gpu: &mut Gpu) -> Result<String, String> {
    use hipfire_runtime::kv_mode::{resolve_qwen4, KvMode};
    for raw in ["", "auto", "bf16"] {
        let r = resolve_qwen4(raw, "gfx942")?;
        if !matches!(r.mode, KvMode::Bf16) {
            return Err(format!("resolve_qwen4({raw:?}, gfx942) is not the F32 (Bf16) state"));
        }
    }
    for raw in ["fp8", "q8"] {
        if resolve_qwen4(raw, "gfx942").is_ok() {
            return Err(format!("resolve_qwen4({raw:?}, gfx942) must be refused"));
        }
    }
    let mut notes = vec!["resolve_qwen4(gfx942): auto/bf16 -> F32 state, fp8/q8 refused".to_string()];
    if gpu.arch != "gfx942" {
        notes.push(format!("arch {} != gfx942: arch-conditional checks skipped", gpu.arch));
        return Ok(notes.join("; "));
    }
    {
        let caps = &gpu.arch_caps;
        if caps.has_gfx11_plus_simt() || caps.has_wmma_w32() || caps.is_gfx1151() {
            return Err("gfx942 reports gfx11 SIMT / WMMA / gfx1151 capability".into());
        }
    }
    for fmt in [QsaKvFormat::F32, QsaKvFormat::Q8, QsaKvFormat::Fp8] {
        if rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(gpu, fmt) {
            return Err(format!("gathered WMMA enabled for {fmt:?} on gfx942"));
        }
    }
    if rdna_compute::tensor_ops::qsa_f16_arena_enabled(gpu, QsaKvFormat::F32, HD) {
        return Err("QSA F16 arena enabled on gfx942".into());
    }
    notes.push(
        "has_gfx11_plus_simt/has_wmma_w32/is_gfx1151 false; gathered-WMMA + F16-arena predicates false"
            .into(),
    );
    for fmt in [QsaKvFormat::Fp8, QsaKvFormat::Q8] {
        let row_units = fmt.kv_row_units(QSA_KV_HEADS, HD);
        let cache_k = gpu.zeros(&[row_units], DType::Raw).map_err(e)?;
        let cache_v = gpu.zeros(&[row_units], DType::Raw).map_err(e)?;
        let key = up_f32(gpu, &vec![0.5; KV_W])?;
        let value = up_f32(gpu, &vec![0.5; KV_W])?;
        let append = indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &key,
                value: &value,
                full_keys: &cache_k,
                full_values: &cache_v,
                rows: 1,
                position_start: 0,
                kv_heads: QSA_KV_HEADS,
                head_dim: HD,
                format: fmt,
            },
        );
        free_all(gpu, vec![cache_k, cache_v, key, value])?;
        match append {
            Ok(()) => return Err(format!("cache_append_batch accepted {fmt:?} K/V on gfx942")),
            Err(err) => notes.push(format!("cache_append_batch {fmt:?} refused: {err}")),
        }
    }
    Ok(notes.join("; "))
}

// ── probes: kernels the gfx942 route does NOT launch ─────────────────────────
// A private module name makes the JIT compile the shipped `tensor_ops.hip`
// source (same concatenation as `tensor_ops.rs` `TENSOR_OPS_SRC`) for gfx942
// even though the launchers' `has_gfx11_plus_simt()` gates never reach these
// kernels. Kernarg layouts follow the launchers (`tensor_ops.rs`).
const PROBE_SRC: &str = concat!(
    include_str!("../../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../../kernels/src/rope_yarn.h"),
    include_str!("../../../../kernels/src/tensor_ops.hip")
);
const PROBE_MODULE: &str = "qwen4_tierc_qsa_probe_tensor_ops";

fn probe_load(gpu: &mut Gpu, sym: &str) -> Result<(), String> {
    gpu.ensure_kernel_public(PROBE_MODULE, PROBE_SRC, sym).map_err(e)
}

fn probe_launch(
    gpu: &mut Gpu,
    sym: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    mut args: hip_bridge::KernargBlob,
) -> Result<(), String> {
    args.pad_to(16);
    gpu.launch_kernel_blob(sym, grid, block, shared, args.as_mut_slice()).map_err(e)
}

/// rows8 / rows16 score kernels (`tensor_ops.hip:3919-3996`) then
/// `indexed_attention_select_from_scores` (`:4004`) — the prefill/decode pair
/// the launcher would run on gfx11/12. Scores are compared bitwise (dyadic
/// data), selected ids bitwise against the stable-tie twin. Context 2802 ⇒ 700
/// blocks; the last row's budget boundary is inside a tie group, exercising the
/// radix select's `__shfl_up(.., 32)` equal-count scan.
fn probe_rows(gpu: &mut Gpu, group_rows: usize, score_sym: &str) -> Result<String, String> {
    let select_sym = "indexed_attention_select_from_scores";
    probe_load(gpu, score_sym)?;
    probe_load(gpu, select_sym)?;
    let end = 2802usize;
    let blocks = end / QSA_COMPRESS;
    let mut checks = Vec::new();
    for rows in ROW_COUNTS {
        let position_start = end - rows;
        let (query, pooled) = gen_select_data(0x7F00 + rows as u64, rows, blocks)?;
        let (want_sel, want_scores) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
        let query_t = up_f32(gpu, &query)?;
        let pooled_t = up_f32(gpu, &pooled)?;
        let scores_t = nan_f32(gpu, rows * blocks)?;
        let selected = raw_i32(gpu, &vec![-7777; rows * SEL_CAP])?;
        let mut a = hip_bridge::KernargBlob::new();
        a.push_ptr(query_t.buf.as_ptr());
        a.push_ptr(pooled_t.buf.as_ptr());
        a.push_ptr(scores_t.buf.as_ptr());
        for v in [rows, IDX_W, blocks, IDX_DIM, QSA_COMPRESS, position_start, blocks] {
            a.push_i32(v as i32);
        }
        probe_launch(
            gpu,
            score_sym,
            [blocks.div_ceil(256) as u32, rows.div_ceil(group_rows) as u32, 1],
            [256, 1, 1],
            0,
            a,
        )?;
        let mut s = hip_bridge::KernargBlob::new();
        s.push_ptr(scores_t.buf.as_ptr());
        s.push_i32(blocks as i32); // score_stride
        s.push_ptr(selected.buf.as_ptr());
        for v in [rows, blocks, BUDGET_BLOCKS, QSA_COMPRESS, position_start, SEL_CAP] {
            s.push_i32(v as i32);
        }
        s.push_ptr(std::ptr::null::<std::ffi::c_void>());
        probe_launch(gpu, select_sym, [rows as u32, 1, 1], [256, 1, 1], 0, s)?;
        sync(gpu)?;
        // Unwritten score slots (blocks past a row's visible count) stay NaN.
        let mut want_scores_full = vec![f32::from_bits(0x7FC0_DEAD); rows * blocks];
        for r in 0..rows {
            let rbc = ((position_start + r + 1) / QSA_COMPRESS).min(blocks);
            want_scores_full[r * blocks..r * blocks + rbc]
                .copy_from_slice(&want_scores[r * blocks..r * blocks + rbc]);
        }
        checks.push(exact_f32(&format!("rows{rows}/scores"), &down_f32(gpu, &scores_t)?, &want_scores_full));
        checks.push(exact_ints(&format!("rows{rows}/selected"), &down_i32(gpu, &selected)?, &want_sel));
        free_all(gpu, vec![query_t, pooled_t, scores_t, selected])?;
    }
    all(checks)
}

fn probe_rows8(gpu: &mut Gpu) -> Result<String, String> {
    probe_rows(gpu, 8, "indexed_attention_select_scores_rows8_f32")
}

fn probe_rows16(gpu: &mut Gpu) -> Result<String, String> {
    probe_rows(gpu, 16, "indexed_attention_select_scores_rows16_f32")
}

/// `indexed_attention_select_f32_batched` (the rank-parallel fused selector,
/// `tensor_ops.hip:3714-3909`, dynamic LDS = `shape_blocks * 4`) with a
/// non-null mirror, atomics and the past-budget radix path.
fn probe_select_parallel(gpu: &mut Gpu) -> Result<String, String> {
    let sym = "indexed_attention_select_f32_batched";
    probe_load(gpu, sym)?;
    let end = 2802usize;
    let blocks = end / QSA_COMPRESS;
    let mut checks = Vec::new();
    for rows in ROW_COUNTS {
        let position_start = end - rows;
        let (query, pooled) = gen_select_data(0x7F80 + rows as u64, rows, blocks)?;
        let (want, _) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
        let query_t = up_f32(gpu, &query)?;
        let pooled_t = up_f32(gpu, &pooled)?;
        let selected = raw_i32(gpu, &vec![-7777; rows * SEL_CAP])?;
        let mirror = raw_i32(gpu, &vec![-7777; SEL_CAP])?;
        let mut a = hip_bridge::KernargBlob::new();
        a.push_ptr(query_t.buf.as_ptr());
        a.push_ptr(pooled_t.buf.as_ptr());
        a.push_ptr(selected.buf.as_ptr());
        for v in [rows, IDX_W, blocks, QSA_INDEX_HEADS, IDX_DIM, BUDGET_BLOCKS, QSA_COMPRESS, position_start, SEL_CAP] {
            a.push_i32(v as i32);
        }
        a.push_ptr(mirror.buf.as_ptr());
        a.push_ptr(std::ptr::null::<std::ffi::c_void>()); // scores_global: scores in LDS
        a.push_i32(0);
        probe_launch(gpu, sym, [rows as u32, 1, 1], [256, 1, 1], (POOLED_CAP * 4) as u32, a)?;
        sync(gpu)?;
        checks.push(exact_ints(&format!("rows{rows}/selected"), &down_i32(gpu, &selected)?, &want));
        checks.push(exact_ints(&format!("rows{rows}/mirror"), &down_i32(gpu, &mirror)?, &want[(rows - 1) * SEL_CAP..]));
        free_all(gpu, vec![query_t, pooled_t, selected, mirror])?;
    }
    all(checks)
}

/// Direct launch of one batched F32 attention kernel over the shared
/// `attn_inputs` (same grid/LDS as `indexed_attention_attention_batch_arena`).
fn probe_attention(gpu: &mut Gpu, sym: &str, grid_x: u32, shared: u32) -> Result<String, String> {
    probe_load(gpu, sym)?;
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for rows in ROW_COUNTS {
            let inp = attn_inputs(0x7C00 + end as u64 + rows as u64, rows, end)?;
            let q = up_f32(gpu, &inp.q)?;
            let (k, v) = (up_f32(gpu, &inp.keys)?, up_f32(gpu, &inp.values)?);
            let sel = raw_i32(gpu, &inp.selected)?;
            let out = nan_f32(gpu, rows * Q_W)?;
            let mut a = hip_bridge::KernargBlob::new();
            for t in [&q, &k, &v, &sel, &out] {
                a.push_ptr(t.buf.as_ptr());
            }
            for x in [
                rows,
                inp.position_start,
                QSA_Q_HEADS,
                QSA_KV_HEADS,
                HD,
                BUDGET_BLOCKS,
                QSA_COMPRESS,
                SEL_CAP,
                inp.end,
            ] {
                a.push_i32(x as i32);
            }
            probe_launch(gpu, sym, [grid_x, 1, rows as u32], [256, 1, 1], shared, a)?;
            sync(gpu)?;
            checks.push(attn_compare(gpu, &out, &inp, &format!("end{end}/rows{rows}")));
            free_all(gpu, vec![q, k, v, sel, out])?;
        }
    }
    all(checks)
}

/// `indexed_attention_attention_f32_batched`: one 256-thread block per head,
/// scores/tokens in dynamic LDS (8 B per selected row), width-32 `__shfl_xor`.
fn probe_attention_batched(gpu: &mut Gpu) -> Result<String, String> {
    probe_attention(gpu, "indexed_attention_attention_f32_batched", QSA_Q_HEADS as u32, (SEL_CAP * 8) as u32)
}

/// `indexed_attention_attention_f32_batched_hg4`: four heads per block; uses a
/// gfx11-format `__builtin_amdgcn_make_buffer_rsrc` descriptor word
/// (`0x31004000`) for the value loads — the case's likeliest failure on CDNA.
fn probe_attention_hg4(gpu: &mut Gpu) -> Result<String, String> {
    let lds = SEL_CAP * (4 * 4 + 4) + 4 * (4 * 256 + 4);
    probe_attention(gpu, "indexed_attention_attention_f32_batched_hg4", (QSA_Q_HEADS / 4) as u32, lds as u32)
}

/// `indexed_attention_attention_f32_batched_hg12`: one block per KV group (12
/// heads), scores in registers (QSA_T = 9 tiles of 256 rows), same gfx11-format
/// buffer descriptor for value loads.
fn probe_attention_hg12(gpu: &mut Gpu) -> Result<String, String> {
    probe_attention(gpu, "indexed_attention_attention_f32_batched_hg12", QSA_KV_HEADS as u32, 0)
}
