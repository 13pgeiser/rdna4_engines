// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Resident (device) twins of the host CPU-AR speculative sampling law in
//! `hipfire-runtime::spec_sampling` (`kernels/src/spec_sampling_cpu_ar.hip`):
//!
//! * [`Gpu::spec_cpu_ar_rows_enqueue`] — `SparseDist::build_from_logits` for
//!   `row_count` contiguous logit rows (target `p`).
//! * [`Gpu::spec_cpu_ar_candidates_enqueue`] — `SparseDist::build_from_candidates`
//!   for one draft row of at most eight candidates, plus the inverse-CDF draw.
//! * [`Gpu::spec_accept_ordered_enqueue`] — `accept_sampled_prefix` over the
//!   resident rows with the host's pre-drawn uniform tape.
//!
//! The kernels are a serial, faithful reference of the host arithmetic (same
//! operation order, no FMA contraction, plain HIP `expf`). Bit equality with
//! the host, including `expf` versus `f32::exp`, is NOT established: it awaits
//! the hardware bit-comparison gate. Launches are stream-ordered, atomic-free
//! and recorded through `launch_maybe_blob`; no enqueue method reads anything
//! back. Allocate
//! [`SpecDistRows`]/[`SpecDistScratch`]/tapes and call
//! [`Gpu::spec_cpu_ar_precompile`] before capture or the timed window.
//!
//! The crate has no integer `DType`: `u32` buffers are `F32`-typed tensors of
//! the same byte size (the convention of `sampled_rows_compact`); read them
//! back as raw bytes.
//!
//! Self-contained via `include_str!` (no shared-registry edits).

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for every launcher in this module.
pub const SPEC_SAMPLING_CPU_AR_SRC: &str =
    include_str!("../../../kernels/src/spec_sampling_cpu_ar.hip");
/// Module key.
pub const SPEC_SAMPLING_CPU_AR_MODULE: &str = "spec_sampling_cpu_ar";
/// Device symbol of [`Gpu::spec_cpu_ar_rows_enqueue`].
pub const SPEC_CPU_AR_ROWS_SYMBOL: &str = "spec_cpu_ar_rows_f32";
/// Device symbol of [`Gpu::spec_cpu_ar_candidates_enqueue`].
pub const SPEC_CPU_AR_CANDIDATES_SYMBOL: &str = "spec_cpu_ar_candidates_f32";
/// Device symbol of [`Gpu::spec_accept_ordered_enqueue`].
pub const SPEC_ACCEPT_ORDERED_SYMBOL: &str = "spec_accept_ordered_u32";

/// Fixed entry capacity of one dist row (`SampleSpec::MAX_POOL`).
pub const SPEC_CPU_AR_CAP: usize = 64;
/// Candidates a draft row is built from.
pub const SPEC_CPU_AR_MAX_CANDIDATES: usize = 8;
/// `eos` value meaning "no EOS" and the accept `terminal` of an accepted EOS.
pub const SPEC_CPU_AR_EOS_NONE: u32 = u32::MAX;
/// `picked` value written when the draft row has no finite candidate/mass
/// (the host's `Err`).
pub const SPEC_CPU_AR_PICK_INVALID: u32 = u32::MAX;

/// Words of scratch per row: 64 pool token ids then 64 pool logits.
const SCRATCH_ROW_WORDS: usize = 128;

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("spec_sampling_cpu_ar: {}", msg.as_ref()))
}

/// A request's truncation, resolved in Rust from the host `SampleSpec`
/// (`SampleSpec::cpu_ar`); fields are passed to the device unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecCpuArPolicy {
    pub temperature: f32,
    /// Already clamped to `[0, 1]`.
    pub top_p: f32,
    /// Candidate pool, in `1..=64`.
    pub pool: u32,
    /// Candidates kept, in `1..=pool`.
    pub cap: u32,
    /// `0` disables the cut; else in `(0, 1]`.
    pub min_p: f32,
}

impl SpecCpuArPolicy {
    /// Plain field constructor (no clamping: pass `SampleSpec` fields as-is).
    pub fn new(temperature: f32, top_p: f32, pool: u32, cap: u32, min_p: f32) -> Self {
        Self { temperature, top_p, pool, cap, min_p }
    }

    fn validate(&self, what: &str) -> HipResult<()> {
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return Err(err(format!(
                "{what}: needs a positive finite temperature, got {}",
                self.temperature
            )));
        }
        if self.pool == 0 || self.pool as usize > SPEC_CPU_AR_CAP {
            return Err(err(format!("{what}: pool {} outside 1..=64", self.pool)));
        }
        if self.cap == 0 || self.cap > self.pool {
            return Err(err(format!("{what}: cap {} outside 1..=pool {}", self.cap, self.pool)));
        }
        Ok(())
    }
}

/// Device distribution rows in `SparseDist::entries()` order (descending
/// probability, not ascending token): row `r` is
/// `ids/probabilities[r*64 .. r*64 + counts[r]]`, slots past the count are
/// zero. `counts`/`ids` are `u32` data in `F32`-typed tensors.
pub struct SpecDistRows {
    /// `u32[capacity_rows]`.
    pub counts: GpuTensor,
    /// `u32[capacity_rows * 64]`.
    pub ids: GpuTensor,
    /// `f32[capacity_rows * 64]`.
    pub probabilities: GpuTensor,
    pub capacity_rows: usize,
}

impl SpecDistRows {
    /// Zero-initialised rows (every count 0).
    pub fn new(gpu: &mut Gpu, capacity_rows: usize) -> HipResult<Self> {
        if capacity_rows == 0 {
            return Err(err("SpecDistRows::new: capacity_rows must be > 0"));
        }
        Ok(Self {
            counts: spec_cpu_ar_alloc_u32(gpu, capacity_rows)?,
            ids: spec_cpu_ar_alloc_u32(gpu, capacity_rows * SPEC_CPU_AR_CAP)?,
            probabilities: spec_cpu_ar_alloc_f32(gpu, capacity_rows * SPEC_CPU_AR_CAP)?,
            capacity_rows,
        })
    }

    /// Release the device buffers.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.free_tensor(self.counts)?;
        gpu.free_tensor(self.ids)?;
        gpu.free_tensor(self.probabilities)
    }
}

/// Reusable per-row working storage of the rows kernel (the pool slots). Sized for `capacity_rows` rows.
pub struct SpecDistScratch {
    pub words: GpuTensor,
    pub capacity_rows: usize,
}

impl SpecDistScratch {
    pub fn new(gpu: &mut Gpu, capacity_rows: usize) -> HipResult<Self> {
        if capacity_rows == 0 {
            return Err(err("SpecDistScratch::new: capacity_rows must be > 0"));
        }
        Ok(Self {
            words: spec_cpu_ar_alloc_f32(gpu, capacity_rows * SCRATCH_ROW_WORDS)?,
            capacity_rows,
        })
    }

    /// Release the device buffer.
    pub fn free(self, gpu: &mut Gpu) -> HipResult<()> {
        gpu.free_tensor(self.words)
    }
}

/// Zeroed `n`-word device buffer holding `u32` data (F32-typed tensor): proposal
/// ids, accept results, `picked`, candidates.
pub fn spec_cpu_ar_alloc_u32(gpu: &mut Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.zeros(&[n.max(1)], DType::F32)
}

/// Zeroed `n`-element f32 device buffer: the uniform tape.
pub fn spec_cpu_ar_alloc_f32(gpu: &mut Gpu, n: usize) -> HipResult<GpuTensor> {
    gpu.zeros(&[n.max(1)], DType::F32)
}

fn check_bytes(name: &str, t: &GpuTensor, need: usize) -> HipResult<()> {
    if t.buf.size() < need {
        return Err(err(format!(
            "{name} buffer too small (have {} need {need} bytes)",
            t.buf.size()
        )));
    }
    Ok(())
}

impl Gpu {
    /// Load the module and bind all three kernels (call before graph capture
    /// and before the timed window).
    pub fn spec_cpu_ar_precompile(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        for symbol in [
            SPEC_CPU_AR_ROWS_SYMBOL,
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
            SPEC_ACCEPT_ORDERED_SYMBOL,
        ] {
            self.ensure_kernel(SPEC_SAMPLING_CPU_AR_MODULE, SPEC_SAMPLING_CPU_AR_SRC, symbol)?;
        }
        Ok(())
    }

    /// Build the target distribution of `row_count` contiguous logit rows
    /// (`logits`: f32 `[row_count * vocab]`, row index = token id) into rows
    /// `0..row_count` of `rows`, exactly as `SparseDist::build_from_logits`.
    /// `scratch` must cover `row_count` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_cpu_ar_rows_enqueue(
        &mut self,
        logits: &GpuTensor,
        rows: &SpecDistRows,
        scratch: &SpecDistScratch,
        vocab: usize,
        row_count: usize,
        policy: SpecCpuArPolicy,
    ) -> HipResult<()> {
        policy.validate("spec_cpu_ar_rows_enqueue")?;
        if row_count == 0 || vocab == 0 {
            return Err(err(format!("rows: row_count {row_count} and vocab {vocab} must be > 0")));
        }
        if vocab > i32::MAX as usize
            || row_count
                .checked_mul(vocab)
                .is_none_or(|n| n.checked_mul(4).is_none())
        {
            return Err(err("rows: row_count * vocab exceeds the addressable range"));
        }
        if row_count > rows.capacity_rows || row_count > scratch.capacity_rows {
            return Err(err(format!(
                "rows: row_count {row_count} exceeds rows capacity {} / scratch capacity {}",
                rows.capacity_rows, scratch.capacity_rows
            )));
        }
        if logits.dtype != DType::F32 {
            return Err(err(format!("rows: logits must be F32, got {:?}", logits.dtype)));
        }
        check_bytes("logits", logits, row_count * vocab * 4)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_CPU_AR_ROWS_SYMBOL,
        )?;
        let lp = logits.buf.as_ptr();
        let cp = rows.counts.buf.as_ptr();
        let ip = rows.ids.buf.as_ptr();
        let pp = rows.probabilities.buf.as_ptr();
        let sp = scratch.words.buf.as_ptr();
        let vocab_i = vocab as i32;
        let pool_i = policy.pool as i32;
        let cap_i = policy.cap as i32;
        let temp = policy.temperature;
        let top_p = policy.top_p;
        let min_p = policy.min_p;
        let mut params: [*mut c_void; 11] = [
            &lp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &vocab_i as *const _ as *mut c_void,
            &pool_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
            &temp as *const _ as *mut c_void,
            &top_p as *const _ as *mut c_void,
            &min_p as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SPEC_CPU_AR_ROWS_SYMBOL,
            [row_count as u32, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(pp);
                b.push_ptr(sp);
                b.push_i32(vocab_i);
                b.push_i32(pool_i);
                b.push_i32(cap_i);
                b.push_f32(temp);
                b.push_f32(top_p);
                b.push_f32(min_p);
                b
            },
        )
    }

    /// Build draft row `row` of `rows` from up to eight candidates exactly as
    /// `SparseDist::build_from_candidates`, then draw from it with `uniform`
    /// as `SparseDist::sample`.
    ///
    /// `candidates`: 16 `u32` words, `[0, 8)` token ids and `[8, 16)` f32 logit
    /// bits; slots with a non-finite logit are dropped (pad with `-inf`).
    /// `picked`: one `u32` word that receives the drawn token, or
    /// [`SPEC_CPU_AR_PICK_INVALID`] (row count 0) when no candidate is finite
    /// or the mass is not positive finite: this is the host `Err`, which a
    /// kernel cannot raise, so the caller MUST treat it as an error (never as
    /// a distribution) when it reads `picked` back.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_cpu_ar_candidates_enqueue(
        &mut self,
        candidates: &GpuTensor,
        rows: &SpecDistRows,
        row: usize,
        vocab: usize,
        policy: SpecCpuArPolicy,
        uniform: f32,
        picked: &GpuTensor,
    ) -> HipResult<()> {
        policy.validate("spec_cpu_ar_candidates_enqueue")?;
        if vocab == 0 || vocab > u32::MAX as usize {
            return Err(err(format!("candidates: vocab {vocab} out of range")));
        }
        if row >= rows.capacity_rows {
            return Err(err(format!(
                "candidates: row {row} >= rows capacity {}",
                rows.capacity_rows
            )));
        }
        check_bytes("candidates", candidates, 16 * 4)?;
        check_bytes("picked", picked, 4)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
        )?;
        let kp = candidates.buf.as_ptr();
        let cp = rows.counts.buf.as_ptr();
        let ip = rows.ids.buf.as_ptr();
        let pp = rows.probabilities.buf.as_ptr();
        let pk = picked.buf.as_ptr();
        let row_i = row as i32;
        let cap_i = policy.cap as i32;
        let temp = policy.temperature;
        let top_p = policy.top_p;
        let min_p = policy.min_p;
        let mut params: [*mut c_void; 11] = [
            &kp as *const _ as *mut c_void,
            &cp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &pp as *const _ as *mut c_void,
            &pk as *const _ as *mut c_void,
            &row_i as *const _ as *mut c_void,
            &cap_i as *const _ as *mut c_void,
            &temp as *const _ as *mut c_void,
            &top_p as *const _ as *mut c_void,
            &min_p as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SPEC_CPU_AR_CANDIDATES_SYMBOL,
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(kp);
                b.push_ptr(cp);
                b.push_ptr(ip);
                b.push_ptr(pp);
                b.push_ptr(pk);
                b.push_i32(row_i);
                b.push_i32(cap_i);
                b.push_f32(temp);
                b.push_f32(top_p);
                b.push_f32(min_p);
                b.push_f32(uniform);
                b
            },
        )
    }

    /// `accept_sampled_prefix` over resident rows. `target` rows `0..=k`,
    /// `draft` rows `0..k`; `proposal_ids`: `u32[k]`; `uniforms`: f32 tape of
    /// at least `k + 1` entries (row `j` verdict `uniforms[j]`, its residual
    /// `uniforms[j + 1]`, bonus `uniforms[k]`); `result`: `u32[2]` =
    /// `[accepted, terminal]`, terminal = [`SPEC_CPU_AR_EOS_NONE`] when the last
    /// accepted draft was `eos` (no bonus). `eos = SPEC_CPU_AR_EOS_NONE`
    /// disables the EOS stop. The host's consumed draws are `accepted` after an
    /// accepted EOS, `accepted + 2` after a rejection, `k + 1` when all
    /// accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_accept_ordered_enqueue(
        &mut self,
        target: &SpecDistRows,
        draft: &SpecDistRows,
        proposal_ids: &GpuTensor,
        uniforms: &GpuTensor,
        result: &GpuTensor,
        k: usize,
        eos: u32,
    ) -> HipResult<()> {
        if k >= target.capacity_rows || k > draft.capacity_rows || k > i32::MAX as usize {
            return Err(err(format!(
                "accept: k {k} needs {} target rows (have {}) and {k} draft rows (have {})",
                k + 1,
                target.capacity_rows,
                draft.capacity_rows
            )));
        }
        check_bytes("proposal_ids", proposal_ids, k * 4)?;
        check_bytes("uniforms", uniforms, (k + 1) * 4)?;
        check_bytes("result", result, 8)?;
        self.bind_thread()?;
        self.ensure_kernel(
            SPEC_SAMPLING_CPU_AR_MODULE,
            SPEC_SAMPLING_CPU_AR_SRC,
            SPEC_ACCEPT_ORDERED_SYMBOL,
        )?;
        let tc = target.counts.buf.as_ptr();
        let ti = target.ids.buf.as_ptr();
        let tp = target.probabilities.buf.as_ptr();
        let dc = draft.counts.buf.as_ptr();
        let di = draft.ids.buf.as_ptr();
        let dp = draft.probabilities.buf.as_ptr();
        let pr = proposal_ids.buf.as_ptr();
        let un = uniforms.buf.as_ptr();
        let rs = result.buf.as_ptr();
        let k_i = k as i32;
        let mut params: [*mut c_void; 11] = [
            &tc as *const _ as *mut c_void,
            &ti as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &dc as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
            &dp as *const _ as *mut c_void,
            &pr as *const _ as *mut c_void,
            &un as *const _ as *mut c_void,
            &rs as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
            &eos as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            SPEC_ACCEPT_ORDERED_SYMBOL,
            [1, 1, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(tc);
                b.push_ptr(ti);
                b.push_ptr(tp);
                b.push_ptr(dc);
                b.push_ptr(di);
                b.push_ptr(dp);
                b.push_ptr(pr);
                b.push_ptr(un);
                b.push_ptr(rs);
                b.push_i32(k_i);
                b.push_u32(eos);
                b
            },
        )
    }
}
