// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun design A slice A8: the GPU-facing [`WindowBuilder`] the window
//! owner (`dflash_railgun`) installs when `HIPFIRE_DFLASH_RAILGUN_WINDOW=1`.
//!
//! One window, in the order the owner drives it:
//!
//! 1. [`WindowBuilder::prepare`] — on the first window of a topology, plan the
//!    A4 draft and the A5 verify under the plan sink
//!    ([`plan_dflash_draft`], [`plan_dflash_verify_full`]), author them into
//!    one railgun program with exact-artifact facts from the receipt corpus
//!    ([`build_window`]), plan its PM4 boundaries, lower it through railgun's
//!    own transport (`Gpu::lower_railgun_window`) and wrap it in a
//!    [`PreparedWindow`] over a `RailgunWindowQueue`. Later windows of the
//!    same topology find the skeleton in the owner's 32-entry cache and only
//!    re-bind. Either way `prepare` returns this window's
//!    [`PreparedInputs`]; the GDN frame words carry the plan's authored frame
//!    (the HIP-graph verify bakes it at capture, a replay reserves none; see
//!    the comment in `prepare`).
//! 2. [`WindowBuilder::stage`] / [`WindowBuilder::await_staged`] — pack the
//!    window inputs into persistent pinned storage, one async H2D into the
//!    device staging buffer on the owner stream and one fence wait. The
//!    staging buffer is the A4 draft's input section; the verify inputs
//!    (`pbs.tokens[0]`, `pbs.positions`) reach their buffers through two typed
//!    transport copies at the head of the window program.
//! 3. [`WindowBuilder::read_results`] — after the queue is quiescent: the
//!    device proposal block (and its status word) and, for `Whole16`, the 16
//!    verify argmax picks.
//!
//! ## Topology identity
//!
//! The owner's cache is keyed by `railgun::window::TopologyKey`, which is only
//! known once a program exists. The builder therefore keeps a cheap
//! *fingerprint* → topology map: the shape family, execution shape, tape and
//! selector presence and the `(address, extent, generation)` of every owner
//! allocation a window reads or writes (draft scratch, verify scratch, hidden
//! ring, DeltaNet state, and the KV tensors with their VMM generation). A
//! fingerprint hit goes straight to the cached skeleton without re-planning;
//! a miss plans. Allocations are re-fingerprinted every window, so growing a
//! VMM KV mapping, replacing a scratch buffer or a capacity change is a new
//! topology before anything is submitted. Weights are immutable for the
//! speculator's life and enter through the lowering's own program resources.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use hip_bridge::{DeviceBuffer, Event, HipError, HipResult, PinnedHostBuffer, Stream};
use hipfire_runtime::dflash::{
    PinnedWindowInputs, SelectorDeviceCodebooks, SelectorGreedyBlock,
};
use railgun::index_bind::{
    PositionsIdentity, PositionsStage, PositionsTransport, SealedPositions, TransportFailure,
    WriterExtent, POSITION_BYTES, POSITION_ROWS,
};
use railgun::kernel::{Arg, KernelProgram, NodeFacts, WordRole, WITNESS_INDEXED_BIND};
use railgun::plan::{plan, CacheTable, G4Table, RowPolicy};
use railgun::window::{
    AllocationIdentity, BoundWindow, NodeImage, PreparedWindow, QueueFailure, Quiescence, Refusal,
    ResolvedStep, TopologyKey, WindowBindingKey, WindowBindings, WindowCache, WindowExecution,
    WindowInputs, WindowPlan, WindowQueue, WindowShape,
};
use railgun::{Binding, ResourceId, WordId};
use rdna_compute::railgun_window::{RailgunLoweringContext, RailgunTransportDigests, WindowCorpus};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::dflash_draft_emit::{
    draft_family_key, draft_staged_words, draft_state_tensors, plan_dflash_draft, DraftEmitArgs,
    DraftStagingLayout, DRAFT_ROWS, SCALAR_DRAFT_SEED,
};
use crate::dflash_railgun::{PreparedInputs, WindowBuilder, WindowRequest, WindowResults};
use crate::dflash_verify_emit::plan_dflash_verify_full;
use crate::dflash_window_build::{
    build_window, window_sequence, BuildInput, SealedRetarget, SequenceNode, TransportCopy,
    TransportDigests, WindowFactsSource, SCALAR_RING_WRITTEN,
};

/// Directory of the railgun JIT receipt corpus (required for any window).
pub const CORPUS_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_CORPUS";
/// Optional whole16 window manifest (`railgun-cert whole16`): when set, a
/// `Whole16` plan must be admitted by it and only then reports
/// `same_as_hip_graph`.
pub const MANIFEST_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_MANIFEST";
/// Optional path: the first `Whole16` plan's node sequence is written there as
/// the JSONL `railgun-cert whole16-manifest --sequence` reads (phase, symbol,
/// loaded module per node), so the manifest is generated from the very plan
/// that will run.
pub const SEQUENCE_OUT_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_SEQUENCE_OUT";
/// Developer arm (default off): one stderr line per railgun window, before
/// submit, describing the bound window (shape, skeleton cache key and
/// hit/miss, patched words, bound pointer range per resource, image digest).
pub const DEBUG_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_DEBUG";
/// Developer arm (default off): execute each bound window's node images through
/// ordinary HIP launches (exact bound kernarg bytes, grids, copy/fill
/// parameters) instead of the PM4 lowering. A fault that also happens here is
/// a bad binding; one that only happens on PM4 is the lowering or transport.
pub const TWIN_VAR: &str = "HIPFIRE_DFLASH_RAILGUN_TWIN";
/// Optional G4 probe cache table (`cache-table.<arch>.json`), shared with the
/// M1 shadow: receipted boundary rows use their own rung, the rest
/// `WaitIdle + System` (`RowPolicy::UnreceiptedSystem`).
const CACHE_TABLE_VAR: &str = "HIPFIRE_RAILGUN_CACHE_TABLE";
/// Minimum G4 probe trials per rung (the M1 shadow's bar).
const G4_MIN_TRIALS: u64 = 1 << 20;

/// Rows of the fixed chain window.
const ROWS: usize = DRAFT_ROWS;

// Staging layout (byte offsets; u32 words): seed, noise ids, Q positions,
// verify positions,
// then the K positions (live suffix + the B noise positions), which are last
// because their length varies with the context.
const SEED_OFF: usize = 0;
const IDS_OFF: usize = 4;
const Q_OFF: usize = IDS_OFF + ROWS * 4;
const VERIFY_OFF: usize = Q_OFF + ROWS * 4;
const K_OFF: usize = VERIFY_OFF + ROWS * 4;

const LAYOUT: DraftStagingLayout = DraftStagingLayout {
    seed_off: SEED_OFF,
    ids_off: IDS_OFF,
    q_off: Q_OFF,
    k_off: K_OFF,
};

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn refuse(msg: impl Into<String>) -> Refusal {
    Refusal::Malformed(msg.into())
}

// ------------------------------------------------------------ pure parts

/// The staging sections of one window, in layout order. `inputs.k_position_suffix`
/// followed by the Q positions is exactly the `positions_k[pos_base..]` the
/// eager draft uploads ([`draft_staged_words`]).
fn staging_sections(inputs: &WindowInputs) -> [Vec<u32>; 5] {
    let ids: Vec<u32> = std::iter::once(inputs.seed)
        .chain(std::iter::repeat(inputs.mask_id).take(ROWS - 1))
        .collect();
    let mut k = inputs.k_position_suffix.clone();
    k.extend_from_slice(&inputs.q_positions);
    [
        vec![inputs.seed],
        ids,
        inputs.q_positions.clone(),
        inputs.verify_positions.clone(),
        k,
    ]
}

/// Words the staging buffer must hold for a draft window of `full_w` rows.
fn staging_words(full_w: usize) -> usize {
    K_OFF / 4 + full_w + ROWS
}

/// This window's `WindowInputs` (everything but the GDN frames).
#[allow(clippy::too_many_arguments)]
fn window_inputs(
    abs_positions: &[i32],
    position: usize,
    compact_offset: i32,
    seed: u32,
    mask_token: u32,
    swa_w: usize,
    ring_written: usize,
    scalars_required: &[String],
) -> Result<WindowInputs, String> {
    let words = draft_staged_words(
        abs_positions,
        position,
        compact_offset,
        seed,
        mask_token,
        ROWS,
        swa_w,
    );
    let suffix_len = words
        .k_positions
        .len()
        .checked_sub(ROWS)
        .ok_or("K positions shorter than the noise block")?;
    let mut scalars = std::collections::BTreeMap::new();
    for name in scalars_required {
        let value = match name.as_str() {
            SCALAR_DRAFT_SEED => u64::from(seed),
            SCALAR_RING_WRITTEN => ring_written as u64,
            other => return Err(format!("unknown required window scalar {other:?}")),
        };
        scalars.insert(name.clone(), value);
    }
    Ok(WindowInputs {
        seed,
        mask_id: mask_token,
        q_positions: words.q_positions,
        k_position_suffix: words.k_positions[..suffix_len].to_vec(),
        verify_positions: (position..position + ROWS).map(|p| p as u32).collect(),
        scalars,
        gdn_frames: std::collections::BTreeMap::new(),
    })
}

/// The `GdnFrame` words of `program`, in node order, with their frame counts.
fn gdn_words(program: &KernelProgram) -> Vec<(WordId, u32)> {
    program
        .words
        .iter()
        .enumerate()
        .filter_map(|(i, w)| match w.role {
            WordRole::GdnFrame { frames } => Some((WordId(i as u32), frames)),
            WordRole::Window => None,
        })
        .collect()
}

// ------------------------------------------------------------- allocations

/// One owner allocation: base, extent and the VMM owner generation (0 for a
/// plain device allocation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Alloc {
    base: u64,
    bytes: u64,
    generation: u64,
}

fn alloc_of(gpu: &Gpu, t: &GpuTensor) -> Alloc {
    Alloc {
        base: t.buf.as_ptr() as usize as u64,
        bytes: t.buf.size() as u64,
        generation: if t.buf.is_vmm_owner() {
            gpu.vmm_owner_generation(t).unwrap_or(0)
        } else {
            0
        },
    }
}

/// The KV tensors that are VMM owners: their mapped extent and generation come
/// from the owner, because `hipMemGetAddressRange` is not an extent oracle for
/// a partially mapped reservation.
fn kv_vmm_allocs(gpu: &Gpu, kv: &saddle_core::kv::KvCache) -> Vec<Alloc> {
    kv.k_gpu
        .iter()
        .chain(&kv.v_gpu)
        .chain(&kv.k_scales)
        .chain(&kv.v_scales)
        .filter(|t| t.buf.is_vmm_owner() && t.buf.size() > 0)
        .map(|t| alloc_of(gpu, t))
        .collect()
}

/// Resources of `program` whose planned allocation (`hipMemGetAddressRange`)
/// is gone, moved or resized in the live HIP runtime. VMM-backed resources
/// (KV) may report differently from their reservation; this is a diagnostic,
/// not an admission rule.
fn stale_resources(gpu: &Gpu, program: &KernelProgram) -> Vec<String> {
    program
        .resources
        .iter()
        .enumerate()
        .filter_map(|(i, r)| match gpu.railgun_probe_allocation(r.base) {
            Err(_) => Some(format!("r{i}[{:#x}+{}] is not a live HIP allocation", r.base, r.bytes)),
            Ok(b) if b.base != r.base => Some(format!(
                "r{i}[{:#x}+{}] now lies inside {:#x}+{}",
                r.base, r.bytes, b.base, b.bytes
            )),
            Ok(b) if b.bytes != r.bytes => Some(format!(
                "r{i}[{:#x}+{}] is now {} bytes",
                r.base, r.bytes, b.bytes
            )),
            Ok(_) => None,
        })
        .collect()
}

/// `(name, base, bytes)` of the owner tensors and `Gpu` scratch slots a window
/// binds pointers into, for the debug resource table. Best effort: weights,
/// KV and anything not listed stay unnamed.
fn named_allocations(gpu: &Gpu, request: &WindowRequest<'_>) -> Vec<(String, u64, u64)> {
    let mut v: Vec<(String, u64, u64)> = Vec::new();
    let mut add = |name: String, t: &GpuTensor| {
        v.push((name, t.buf.as_ptr() as usize as u64, t.buf.size() as u64));
    };
    for (n, t) in draft_state_tensors(&*request.draft_scratch, request.verify_scratch) {
        add(format!("draft.{n}"), t);
    }
    add("verify.final_hidden".into(), &request.verify_scratch.final_hidden);
    for (i, t) in request.hidden_rb.layer_bufs.iter().enumerate() {
        add(format!("ring[{i}]"), t);
    }
    for (i, t) in request.hidden_rb.staging_bufs.iter().enumerate() {
        add(format!("ring_staging[{i}]"), t);
    }
    let dn = &request.target.dn_state;
    for (label, list) in [
        ("dn_s", &dn.s_matrices),
        ("dn_scale", &dn.s_scales),
        ("dn_conv", &dn.conv_states),
        ("dn_ef", &dn.s_ef_residual),
    ] {
        for (i, t) in list.iter().enumerate() {
            add(format!("{label}[{i}]"), t);
        }
    }
    if let Some(p) = request.verify_scratch.prefill_batch.as_ref() {
        macro_rules! pbs {
            ($($f:ident),* $(,)?) => {$( add(concat!("pbs.", stringify!($f)).into(), &p.$f); )*};
        }
        pbs!(x_batch, x_rot_batch, x_norm_batch, dn_qkv_batch, dn_z_batch, dn_z_fold_batch,
            dn_alpha_batch, dn_beta_batch, dn_q_raw_batch, dn_k_raw_batch, dn_v_batch,
            dn_q_batch, dn_k_batch, dn_attn_out_batch, dn_normed_batch, gate_ffn_batch,
            up_batch, ffn_hidden_batch, dn_normed_rot_batch, positions, rope_positions, pos3,
            tokens, fa_q_full_batch, fa_q_batch, fa_gate_batch, fa_k_batch, fa_v_batch,
            fa_attn_out_batch, fa_attn_out_rot_batch, x_rot_f16_batch, dn_normed_rot_f16_batch,
            ffn_hidden_f16_batch, fa_attn_out_rot_f16_batch);
    }
    let sc = &request.target.scratch;
    macro_rules! target {
        ($($f:ident),* $(,)?) => {$( add(concat!("target.", stringify!($f)).into(), &sc.$f); )*};
    }
    target!(x, tmp, flash_partials, logits, x_rot);
    if let Some(tape) = request.gdn_tape.as_ref() {
        for (i, t) in tape.qkv_bufs.iter().enumerate() {
            add(format!("tape.qkv[{i}]"), t);
        }
        for (i, t) in tape.alpha_bufs.iter().enumerate() {
            add(format!("tape.alpha[{i}]"), t);
        }
        for (i, t) in tape.beta_bufs.iter().enumerate() {
            add(format!("tape.beta[{i}]"), t);
        }
    }
    for (n, b, bytes) in gpu.railgun_scratch_slots() {
        v.push((format!("gpu.scratch.{n}"), b, bytes));
    }
    v
}

/// `(base, extent, generation)` of every allocation a window touches through
/// owner state (see the module docs).
fn fingerprint(
    gpu: &Gpu,
    request: &WindowRequest<'_>,
    vmm: &[Alloc],
    positions: Option<PositionsIdentity>,
) -> Vec<Alloc> {
    let mut v: Vec<Alloc> = Vec::new();
    let mut push = |t: &GpuTensor| v.push(alloc_of(gpu, t));
    for (_, t) in draft_state_tensors(&*request.draft_scratch, request.verify_scratch) {
        push(t);
    }
    for t in request
        .hidden_rb
        .layer_bufs
        .iter()
        .chain(&request.hidden_rb.staging_bufs)
    {
        push(t);
    }
    let dn = &request.target.dn_state;
    for t in dn
        .s_matrices
        .iter()
        .chain(&dn.s_scales)
        .chain(&dn.conv_states)
        .chain(&dn.s_ef_residual)
    {
        push(t);
    }
    if let Some(pbs) = request.verify_scratch.prefill_batch.as_ref() {
        push(&pbs.tokens);
        push(&pbs.positions);
    }
    v.extend_from_slice(vmm);
    if let Some(p) = positions {
        v.push(Alloc {
            base: p.base,
            bytes: p.bytes,
            generation: p.generation,
        });
    }
    v
}

/// Fingerprint of a topology (the owner cache key is only known after a
/// program exists).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BuilderKey {
    execution: WindowExecution,
    fc_rows: Vec<u32>,
    concat_segments: usize,
    tape: bool,
    selector: bool,
    allocations: Vec<Alloc>,
    /// `rdna_compute::scratch::scratch_growth_epoch()`: a scratch slot that
    /// grew freed its old buffer, so every skeleton holding a pointer into
    /// scratch is stale (its images would write through freed memory, whose
    /// address range the allocator may already have handed out again).
    scratch_epoch: u64,
}

#[derive(Clone)]
struct KnownTopology {
    topology: TopologyKey,
    scalars_required: Vec<String>,
    /// `logical_ctx_rows` over which the planned attention geometry (block
    /// size, tile bounds) holds: a window outside it plans its own skeleton.
    l_domain: std::ops::RangeInclusive<u64>,
}

// ------------------------------------------------------------------ owner

/// Long-lived device/host resources of the builder, created on the first
/// window (the builder is installed before a `Gpu` is at hand).
struct Resources {
    books: Option<SelectorDeviceCodebooks>,
    block: Option<SelectorGreedyBlock>,
    staging: GpuTensor,
    staging_words: usize,
    pinned: PinnedWindowInputs,
    transport: RailgunTransportDigests,
    positions: PositionsDevice,
}

/// Bytes requested for the dedicated sealed-positions allocation: one whole
/// page, so the `hipMalloc` is the positions' alone (§A9.4: ≥ 64 B, never
/// shared with tokens, counters, scratch or packed descriptors).
const POSITIONS_ALLOC_BYTES: usize = 4096;
/// A reported allocation larger than this is a shared pool block, not a
/// dedicated allocation.
const POSITIONS_ALLOC_MAX: u64 = 1 << 16;

/// The dedicated device-only positions allocation of `Whole16` and its
/// lifecycle owner. `buf` is read only by the window's conditional writers
/// (and any pure read consumer); the only writer is the exclusive H2D of
/// [`RailgunWindowBuilder::seal_positions`].
struct PositionsDevice {
    buf: DeviceBuffer,
    /// Pinned upload source; exclusively used by the sealed upload.
    host: PinnedHostBuffer,
    fence: Event,
    sealed: SealedPositions,
    stage: PositionsStage,
}

impl PositionsDevice {
    fn create(gpu: &Gpu) -> Result<Self, String> {
        let buf = gpu
            .hip
            .malloc(POSITIONS_ALLOC_BYTES)
            .map_err(|e| format!("sealed positions allocation: {e}"))?;
        let base = buf.as_ptr() as usize as u64;
        let fail = |gpu: &Gpu, buf: DeviceBuffer, m: String| {
            let _ = gpu.hip.free(buf);
            Err(m)
        };
        let probe = match gpu.railgun_probe_allocation(base) {
            Ok(p) => p,
            Err(e) => return fail(gpu, buf, format!("sealed positions probe: {e}")),
        };
        if probe.base != base || probe.bytes < POSITION_BYTES || probe.bytes > POSITIONS_ALLOC_MAX {
            let m = format!(
                "sealed positions {base:#x} is reported inside allocation {:#x}+{}: not dedicated",
                probe.base, probe.bytes
            );
            return fail(gpu, buf, m);
        }
        let sealed = match SealedPositions::new(PositionsIdentity {
            base: probe.base,
            bytes: probe.bytes,
            generation: 1,
        }) {
            Ok(s) => s,
            Err(e) => return fail(gpu, buf, e.to_string()),
        };
        let host = match gpu.hip.host_buffer(256) {
            Ok(h) => h,
            Err(e) => return fail(gpu, buf, format!("sealed positions pinned stage: {e}")),
        };
        let fence = match gpu.hip.event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING) {
            Ok(f) => f,
            Err(e) => return fail(gpu, buf, format!("sealed positions fence: {e}")),
        };
        Ok(Self {
            buf,
            host,
            fence,
            sealed,
            stage: PositionsStage::new(),
        })
    }

    fn base(&self) -> u64 {
        self.buf.as_ptr() as usize as u64
    }

    fn binding(&self) -> Binding {
        let id = self.sealed.identity();
        Binding {
            base: id.base,
            bytes: id.bytes,
        }
    }

    fn free(self, gpu: &Gpu) {
        // Quarantined or in-flight positions must not be freed under a copy:
        // only a retired allocation is returned.
        use railgun::index_bind::PositionsState;
        if self.sealed.state() == PositionsState::Retired {
            let _ = gpu.hip.event_destroy(self.fence);
            let _ = gpu.hip.free(self.buf);
        } else {
            eprintln!(
                "[dflash] railgun window: sealed positions are {:?} at teardown; storage retained",
                self.sealed.state()
            );
        }
    }
}

/// The trusted H2D + visibility operations of the sealed upload (§A9.4).
struct DeviceTransport<'a> {
    gpu: &'a Gpu,
    stream: &'a Stream,
    dev: &'a DeviceBuffer,
    host: &'a mut PinnedHostBuffer,
    fence: &'a Event,
}

fn unknown_after_enqueue(e: HipError) -> TransportFailure {
    // The copy may be queued or running: nothing proves the stream idle.
    TransportFailure {
        message: e.to_string(),
        quiescence: Quiescence::Unknown,
    }
}

impl PositionsTransport for DeviceTransport<'_> {
    fn begin_h2d(
        &mut self,
        device: u64,
        bytes: &[u8; POSITION_BYTES as usize],
    ) -> Result<(), TransportFailure> {
        if self.dev.as_ptr() as usize as u64 != device {
            return Err(TransportFailure {
                message: "upload targets a different allocation than the sealed positions".into(),
                quiescence: Quiescence::Proven,
            });
        }
        let n = POSITION_BYTES as usize;
        self.host.as_bytes_mut()[..n].copy_from_slice(bytes);
        self.gpu
            .hip
            .memcpy_htod_async(self.dev, &self.host.as_bytes_mut()[..n], self.stream)
            .map_err(unknown_after_enqueue)?;
        self.gpu
            .hip
            .event_record(self.fence, Some(self.stream))
            .map_err(unknown_after_enqueue)
    }

    fn wait_h2d(&mut self) -> Result<(), TransportFailure> {
        self.gpu
            .hip
            .event_synchronize(self.fence)
            .map_err(unknown_after_enqueue)
    }

    /// The railgun entry/system acquire of the existing transport/cache
    /// receipts is boundary 0 of the window's single PM4 segment
    /// ([`railgun::plan::ENTRY`]); the IB is queued only after the host
    /// observed the H2D fence, so that acquire follows the completed upload.
    fn acquire(&mut self) -> Result<(), TransportFailure> {
        if railgun::plan::ENTRY.acquire == railgun::plan::Rung::System {
            Ok(())
        } else {
            Err(TransportFailure {
                message: "the window entry boundary is not a system acquire".into(),
                quiescence: Quiescence::Proven,
            })
        }
    }
}

/// Non-owning views of what [`WindowBuilder::read_results`] downloads; they
/// are re-cut every `prepare` (the owner keeps the allocations alive).
struct ReadViews {
    /// `verify_scratch.argmax[0..16]`.
    argmax: DeviceBuffer,
    /// `pbs.tokens[0..16]`: `[seed, d1..d15]`, the proposals when the draft has
    /// no selector block.
    tokens: DeviceBuffer,
    selector: bool,
}

/// What `prepare` knows about the skeleton behind one window (trace only).
#[derive(Clone, Debug)]
pub(crate) struct PrepNote {
    /// The skeleton came from the owner's cache without re-planning.
    pub(crate) hit: bool,
    pub(crate) key_hash: u64,
    pub(crate) fc_rows: Vec<u32>,
    pub(crate) concat_segments: usize,
    pub(crate) l_domain: std::ops::RangeInclusive<u64>,
}

fn topology_hash(t: &TopologyKey) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// The GPU-facing window builder.
pub struct RailgunWindowBuilder {
    corpus_dir: Option<PathBuf>,
    manifest: Option<PathBuf>,
    sequence_out: Option<PathBuf>,
    /// `HIPFIRE_DFLASH_RAILGUN_DEBUG`.
    debug: bool,
    /// `HIPFIRE_DFLASH_RAILGUN_TWIN`.
    twin: bool,
    /// Cases the manifest admitted for this window (conditional plans): the
    /// bind witness must discharge exactly these before the window is
    /// certified exact.
    admitted: Option<Vec<(usize, String)>>,
    /// This window was certified exact (manifest admission + witnesses): its
    /// receipt will report `same_as_hip_graph`, which lets the owner keep a
    /// full accept's live state without the DeltaNet repair replay.
    certified: bool,
    /// What `prepare` learned about this window's skeleton, for the trace.
    note: Option<PrepNote>,
    corpus: Option<WindowCorpus>,
    /// One HSA runtime/device, kernarg pools and code-object loads shared by
    /// every skeleton this builder lowers (created on the first window).
    lowering_ctx: Option<RailgunLoweringContext>,
    table: Option<CacheTable>,
    resources: Option<Resources>,
    known: HashMap<BuilderKey, Vec<KnownTopology>>,
    reads: Option<ReadViews>,
}

// SAFETY: `RailgunWindowBuilder` is owned by one `DflashSpeculator` and only
// ever driven through `&mut self` from the thread that holds the `Gpu`. Its raw
// handles are device pointers (`DeviceBuffer`, which is `Send`), pinned host
// storage and an event (`Send`), and the corpus. Nothing is shared.
unsafe impl Send for RailgunWindowBuilder {}

impl RailgunWindowBuilder {
    /// A builder reading its corpus / manifest from the environment at first
    /// use ([`CORPUS_VAR`], [`MANIFEST_VAR`]).
    pub fn from_env() -> Self {
        let var = |name: &str| {
            hipfire_config::developer_var(name)
                .ok()
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        let flag = |name: &str| hipfire_config::developer_var(name).ok().as_deref() == Some("1");
        Self {
            corpus_dir: var(CORPUS_VAR),
            manifest: var(MANIFEST_VAR),
            sequence_out: var(SEQUENCE_OUT_VAR),
            debug: flag(DEBUG_VAR),
            twin: flag(TWIN_VAR),
            admitted: None,
            certified: false,
            note: None,
            corpus: None,
            lowering_ctx: None,
            table: None,
            resources: None,
            known: HashMap::new(),
            reads: None,
        }
    }

    /// Free the builder's device resources (the owner's `release`).
    pub fn free(self, gpu: &mut Gpu) {
        let RailgunWindowBuilder { resources, .. } = self;
        if let Some(r) = resources {
            let Resources {
                books,
                block,
                staging,
                pinned,
                positions,
                ..
            } = r;
            positions.free(gpu);
            if let Err(e) = pinned.destroy(gpu) {
                eprintln!("[dflash] railgun window: staging fence release failed: {e}");
            }
            let _ = gpu.hip.free(staging.buf);
            if let Some(b) = block {
                b.free(gpu);
            }
            if let Some(b) = books {
                b.free(gpu);
            }
        }
    }

    fn ensure_ready(&mut self, request: &mut WindowRequest<'_>) -> Result<(), Refusal> {
        if request.gpu.arch != "gfx1201" {
            return Err(refuse(format!(
                "railgun window: arch {} is not gfx1201",
                request.gpu.arch
            )));
        }
        if self.corpus.is_none() {
            let dir = self.corpus_dir.as_ref().ok_or_else(|| {
                refuse(format!("{CORPUS_VAR} is not set: no receipt corpus to take facts from"))
            })?;
            self.corpus =
                Some(WindowCorpus::open(request.gpu, dir).map_err(refuse)?);
        }
        if self.lowering_ctx.is_none() && !self.twin {
            self.lowering_ctx = Some(request.gpu.railgun_lowering_context().map_err(refuse)?);
        }
        if self.table.is_none() {
            self.table = Some(cache_table(&request.gpu.arch).map_err(refuse)?);
        }
        if self.resources.is_none() {
            self.resources = Some(Self::create_resources(request)?);
        }
        Ok(())
    }

    fn create_resources(request: &mut WindowRequest<'_>) -> Result<Resources, Refusal> {
        let gpu = &mut *request.gpu;
        let transport = gpu.railgun_transport_digests().map_err(refuse)?;
        let words = staging_words(request.shape.full_w as usize);
        let staging = gpu
            .alloc_tensor_exact(&[words], DType::F32)
            .map_err(|e| refuse(format!("staging allocation: {e}")))?;
        let pinned = match PinnedWindowInputs::new(gpu, words * 4) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.hip.free(staging.buf);
                return Err(refuse(format!("pinned staging: {e}")));
            }
        };
        let positions = match PositionsDevice::create(gpu) {
            Ok(p) => p,
            Err(e) => {
                let _ = pinned.destroy(gpu);
                let _ = gpu.hip.free(staging.buf);
                return Err(refuse(e));
            }
        };
        let (books, block) = if request.draft_weights.has_candidate_selector() {
            let books = SelectorDeviceCodebooks::upload(gpu, request.draft_weights)
                .map_err(|e| refuse(format!("selector codebooks: {e}")))?;
            let block = SelectorGreedyBlock::new(gpu, request.draft_weights, ROWS - 1)
                .map_err(|e| refuse(format!("selector block: {e}")))?;
            (Some(books), Some(block))
        } else {
            (None, None)
        };
        Ok(Resources {
            books,
            block,
            staging,
            staging_words: words,
            pinned,
            transport,
            positions,
        })
    }

    /// Plan, author, lower and wrap one window.
    fn build(
        &mut self,
        request: &mut WindowRequest<'_>,
        vmm: &[Alloc],
        scratch_epoch: u64,
    ) -> Result<Built, Refusal> {
        let execution = request.execution;
        let whole16 = execution == WindowExecution::Whole16;
        let position = request.shape.position as usize;
        let tape_populated = request.gdn_tape.is_some();
        let compact_offset = request.target.kv_cache.compact_offset as i32;
        let pbs_ptrs = {
            let pbs = request
                .verify_scratch
                .prefill_batch
                .as_ref()
                .ok_or_else(|| refuse("verify scratch has no PrefillBatchScratch"))?;
            (
                pbs.tokens.buf.as_ptr() as usize as u64,
                pbs.positions.buf.as_ptr() as usize as u64,
            )
        };
        let argmax_ptr = request.verify_scratch.argmax.buf.as_ptr() as usize as u64;
        let (corpus, resources) = (
            self.corpus.as_mut().expect("ensure_ready"),
            self.resources.as_ref().expect("ensure_ready"),
        );
        let transport = TransportDigests {
            copy: resources.transport.copy,
            fill: resources.transport.fill,
        };
        let staging_base = resources.staging.buf.as_ptr() as usize as u64;

        // ---- plan the draft under the sink --------------------------------
        let selector = match (&resources.books, &resources.block) {
            (Some(b), Some(k)) => Some((b, k)),
            _ => None,
        };
        let draft = plan_dflash_draft(
            &mut *request.gpu,
            DraftEmitArgs {
                target_weights: &request.target.weights,
                draft_weights: request.draft_weights,
                draft_cfg: request.draft_cfg,
                draft_scratch: &mut *request.draft_scratch,
                verify_scratch: request.verify_scratch,
                selector,
                staging: &resources.staging.buf,
                layout: LAYOUT,
                verify_tokens: &request
                    .verify_scratch
                    .prefill_batch
                    .as_ref()
                    .expect("checked above")
                    .tokens
                    .buf,
                position,
                compact_offset,
                seed_token: request.seed,
                mask_token: request.mask_token,
                max_position: request.shape.max_position,
            },
        )
        .map_err(|e| refuse(format!("draft plan: {e}")))?;

        // ---- plan the verify (Whole16) -------------------------------------
        let verify = if whole16 {
            let tokens = vec![request.seed; ROWS];
            Some(
                plan_dflash_verify_full(
                    &mut *request.gpu,
                    &mut *request.target,
                    &tokens,
                    position,
                    &*request.hidden_rb,
                    request.gdn_tape.as_mut().map(|t| &mut **t),
                    request.verify_scratch,
                )
                .map_err(|e| refuse(format!("verify plan: {e}")))?,
            )
        } else {
            None
        };

        // ---- author ----------------------------------------------------------
        let prefix = if whole16 {
            vec![
                TransportCopy {
                    src: staging_base + SEED_OFF as u64,
                    dst: pbs_ptrs.0,
                    bytes: 4,
                },
                TransportCopy {
                    src: staging_base + VERIFY_OFF as u64,
                    dst: pbs_ptrs.1,
                    bytes: (ROWS * 4) as u64,
                },
            ]
        } else {
            Vec::new()
        };
        let completion: Vec<u64> = if whole16 { vec![argmax_ptr] } else { Vec::new() };
        let mut facts = CorpusFacts {
            corpus,
            transport,
        };
        let gpu_ref: &Gpu = &*request.gpu;
        let mut probe = |addr: u64| -> Result<Binding, String> {
            if let Some(a) = vmm
                .iter()
                .find(|a| addr >= a.base && addr - a.base < a.bytes.max(1))
            {
                return Ok(Binding {
                    base: a.base,
                    bytes: a.bytes,
                });
            }
            gpu_ref.railgun_probe_allocation(addr)
        };
        let build_input = BuildInput {
            arch: &request.gpu.arch,
            execution,
            prefix: &prefix,
            draft: &draft,
            verify: verify.as_ref(),
            ring_capacity: request.hidden_rb.max_positions as u64,
            completion_addrs: &completion,
            certified: false,
            tape_populated,
            // §A9.4: Whole16's conditional KV writers read the dedicated
            // sealed positions instead of `pbs.positions` (the same
            // `[i32; 16]` bytes; the prefix copy still feeds every other
            // reader of `pbs.positions`).
            sealed_positions: whole16.then(|| SealedRetarget {
                from: pbs_ptrs.1,
                to: resources.positions.base(),
                binding: resources.positions.binding(),
            }),
        };
        // The node sequence needs no facts: dump it first so the corpus and the
        // manifest can be built from a window whose objects have no record yet.
        if whole16 {
            if let Some(path) = &self.sequence_out {
                let sequence = window_sequence(&build_input).map_err(|e| refuse(e.to_string()))?;
                write_sequence(path, &sequence).map_err(refuse)?;
            }
        }
        let built = build_window(&build_input, &mut facts, &mut probe)
            .map_err(|e| refuse(e.to_string()))?;
        // Manifest admission is per window, host-only and BEFORE anything is
        // mutated, in `WindowBuilder::admit` (right after bind): it needs no
        // build-time decision, so a skeleton the manifest refuses is cached
        // like any other instead of being re-planned every window.
        let scalars_required = built.scalars_required.clone();


        // ---- PM4 plan and lowering --------------------------------------------
        let table = self.table.as_ref().expect("ensure_ready");
        let pm4 = plan(&built.parts.program, table, RowPolicy::UnreceiptedSystem);
        let window_plan = Arc::new(WindowPlan::new(built.parts, &pm4)?);
        // Name the program's resources for the debug arm (and list the `Gpu`
        // scratch slots among them: those are replaced when they grow).
        let resource_names: Vec<String> = if self.debug {
            let named = named_allocations(&*request.gpu, request);
            let names: Vec<String> = window_plan
                .program()
                .resources
                .iter()
                .map(|r| {
                    named
                        .iter()
                        .find(|(_, b, bytes)| *b <= r.base && r.base < b + (*bytes).max(1))
                        .map(|(n, ..)| n.clone())
                        .unwrap_or_default()
                })
                .collect();
            let scratch: Vec<String> = names
                .iter()
                .enumerate()
                .zip(&window_plan.program().resources)
                .filter(|((_, n), _)| n.starts_with("gpu.scratch."))
                .map(|((i, n), r)| format!("r{i}={n}[{:#x}+{}]", r.base, r.bytes))
                .collect();
            let unnamed = names.iter().filter(|n| n.is_empty()).count();
            eprintln!(
                "[dflash-railgun] skeleton resources: {} total, {unnamed} unnamed (weights, KV, other); \
                 Gpu scratch slots (replaced on growth, epoch {scratch_epoch}): {}",
                names.len(),
                if scratch.is_empty() { "none".to_owned() } else { scratch.join(" ") }
            );
            names
        } else {
            Vec::new()
        };
        // Twin mode runs the bound images through HIP: no PM4 lowering, and a
        // queue that does nothing (the twin itself proves quiescence).
        let queue: Box<dyn WindowQueue> = if self.twin {
            Box::new(TwinQueue)
        } else {
            let ctx = self.lowering_ctx.as_mut().expect("ensure_ready");
            let mut queue = request
                .gpu
                .lower_railgun_window(ctx, window_plan.program(), &pm4)
                .map_err(|e| refuse(format!("lowering: {e}")))?;
            // HIPFIRE_DFLASH_RAILGUN_DEBUG: read the lowered state back and
            // compare it with the bound images before every replay.
            queue.set_self_check(self.debug);
            queue.set_resource_names(resource_names);
            Box::new(queue)
        };
        let key = binding_key(
            &request.gpu,
            request,
            Some(resources.positions.sealed.identity()),
            window_plan.program(),
            facts.corpus.toolchain_pin(),
            vmm,
            selector.is_some(),
            tape_populated,
            execution,
            &draft.l_domain,
            scratch_epoch,
        );
        let topology = TopologyKey {
            binding: key.clone(),
            family: window_plan.family().clone(),
        };
        let prepared = PreparedWindow::new(window_plan, key, queue)?;
        Ok(Built {
            prepared,
            topology,
            scalars_required,
            l_domain: draft.l_domain.clone(),
        })
    }
}

struct Built {
    prepared: PreparedWindow,
    topology: TopologyKey,
    scalars_required: Vec<String>,
    l_domain: std::ops::RangeInclusive<u64>,
}

/// Facts adapter over the receipt corpus and the loaded transport objects.
struct CorpusFacts<'a> {
    corpus: &'a mut WindowCorpus,
    transport: TransportDigests,
}

impl WindowFactsSource for CorpusFacts<'_> {
    fn facts(&mut self, artifact_sha256_hex: &str, symbol: &str) -> Result<Arc<NodeFacts>, String> {
        self.corpus.facts(artifact_sha256_hex, symbol)
    }

    fn transport(&mut self) -> Result<TransportDigests, String> {
        Ok(self.transport)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// gfx12 cache table, with the optional G4 receipts attached.
fn cache_table(arch: &str) -> Result<CacheTable, String> {
    let mut table = CacheTable::for_arch(arch)
        .ok_or_else(|| format!("railgun has no cache table for {arch}"))?;
    if let Some(path) = hipfire_config::developer_var(CACHE_TABLE_VAR)
        .ok()
        .filter(|v| !v.is_empty())
    {
        let bytes = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        let probe: G4Table = serde_json::from_slice(&bytes).map_err(|e| format!("{path}: {e}"))?;
        use sha2::{Digest, Sha256};
        let source = format!("{path} sha256={}", hex(&Sha256::digest(&bytes)));
        let attached = table.attach_receipts(&probe, &source, G4_MIN_TRIALS)?;
        eprintln!(
            "[dflash] railgun window: G4 receipts {attached}/{} rows backed by {source}",
            table.rows.len()
        );
    }
    Ok(table)
}

/// The JSONL `railgun-cert whole16-manifest --sequence` reads. `path` is a
/// file (overwritten: the last plan wins) or a directory, which receives one
/// `<fnv64 of the sequence>.seq.jsonl` per DISTINCT sequence, so every shape
/// family of a run can get its own manifest.
fn write_sequence(path: &std::path::Path, sequence: &[SequenceNode]) -> Result<(), String> {
    let mut text = String::new();
    for n in sequence {
        text.push_str(
            &serde_json::json!({
                "phase": n.phase,
                "symbol": n.symbol,
                "route": "railgun",
                "module": n.module,
            })
            .to_string(),
        );
        text.push('\n');
    }
    let target = if path.is_dir() {
        path.join(format!("{:016x}.seq.jsonl", fnv64(&[text.as_bytes()])))
    } else {
        path.to_owned()
    };
    std::fs::write(&target, text).map_err(|e| format!("{}: {e}", target.display()))
}

/// Retire sealed positions of a window refused before anything was queued, to
/// `Retired` so the next window can host-check and upload afresh. Only states
/// that can exist with nothing queued are retired; anything else (an upload or
/// a submit that may still touch the positions, a quarantine) is reported and
/// left alone.
fn retire_unsubmitted(sealed: &SealedPositions) -> Result<(), String> {
    use railgun::index_bind::PositionsState as S;
    match sealed.state() {
        S::Retired => Ok(()),
        S::HostChecked | S::SealedReady => sealed
            .release_unsubmitted()
            .map_err(|e| format!("release_unsubmitted: {e}")),
        other => Err(format!("positions are {other:?}: not retirable without a completion")),
    }
}

/// The binding key's context mode. It carries the plan's `l_domain` (the
/// `logical_ctx_rows` span its declared patch domains cover, which A4 derives
/// from the attention block): two skeletons that differ only in that span are
/// different topologies, so a shape outside a cached plan's domain is a cache
/// miss that builds its own skeleton instead of fetching the old one (whose
/// patch domains would refuse the bind forever).
fn context_mode(swa_w: u64, full_w: u64, l_domain: &std::ops::RangeInclusive<u64>) -> String {
    format!(
        "windowed w={swa_w} w_full={full_w} l_domain={}..={}",
        l_domain.start(),
        l_domain.end()
    )
}

#[allow(clippy::too_many_arguments)]
fn binding_key(
    gpu: &Gpu,
    request: &WindowRequest<'_>,
    positions: Option<PositionsIdentity>,
    program: &KernelProgram,
    toolchain_pin: &str,
    vmm: &[Alloc],
    selector: bool,
    tape: bool,
    execution: WindowExecution,
    l_domain: &std::ops::RangeInclusive<u64>,
    scratch_epoch: u64,
) -> WindowBindingKey {
    let cfg = &request.target.config;
    let kv = &request.target.kv_cache;
    let dn = &request.target.dn_state;
    WindowBindingKey {
        model_id: format!(
            "qwen35 dim={} layers={} vocab={} kv_heads={}",
            cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.n_kv_heads
        ),
        draft_id: format!(
            "dflash hidden={} layers={} extract={:?} block={}",
            request.draft_cfg.hidden,
            request.draft_cfg.num_extract(),
            request.hidden_rb.extract_layers,
            request.draft_cfg.block_size
        ),
        object_digests: program
            .nodes
            .iter()
            .map(|n| n.kref.as_ref().map(|k| k.artifact_sha256.clone()).unwrap_or_default())
            .collect(),
        toolchain_digest: toolchain_pin.to_owned(),
        device: gpu.arch.clone(),
        route_flags: u64::from(selector)
            | u64::from(tape) << 1
            | u64::from(execution == WindowExecution::Whole16) << 2,
        allocations: program
            .resources
            .iter()
            .enumerate()
            .map(|(i, b)| AllocationIdentity {
                resource: ResourceId(i as u32),
                base: b.base,
                bytes: b.bytes,
                generation: vmm
                    .iter()
                    .find(|a| a.base == b.base)
                    .map(|a| a.generation)
                    .or_else(|| positions.filter(|p| p.base == b.base).map(|p| p.generation))
                    .unwrap_or(0),
            })
            .collect(),
        kv_mapping_generation: vmm
            .iter()
            .fold(0u64, |acc, a| acc.wrapping_mul(1_000_003).wrapping_add(a.generation)),
        scratch_capacities: vec![
            kv.physical_cap as u64,
            request.verify_scratch.max_n as u64,
            request
                .verify_scratch
                .prefill_batch
                .as_ref()
                .map_or(0, |p| p.max_batch as u64),
            request.hidden_rb.max_positions as u64,
            request.hidden_rb.max_batch as u64,
            // Scratch-slot growth epoch: a skeleton planned before a growth is a
            // different topology (its scratch pointers are stale), never a hit.
            scratch_epoch,
        ],
        context_mode: context_mode(request.shape.swa_w, request.shape.full_w, l_domain),
        gdn_mode: format!("{:?} ef={}", dn.quant, !dn.s_ef_residual.is_empty()),
        extraction_layers: request
            .hidden_rb
            .extract_layers
            .iter()
            .map(|l| *l as u32)
            .collect(),
    }
}

/// §A9.4 per-writer extents: for each conditional node, the intended KV
/// tensor slice (`physical_cap` rows of `R = H*258` bytes of the writer's own
/// K or V tensor, never the arena), its actually mapped interval and the
/// mapping generation the binding key holds.
fn writer_extents(
    gpu: &Gpu,
    kv: &saddle_core::kv::KvCache,
    prepared: &PreparedWindow,
) -> HipResult<Vec<WriterExtent>> {
    let program = prepared.plan().program();
    let mut out = Vec::new();
    for node in program.conditional_nodes() {
        let image = prepared
            .kernarg_image(node)
            .ok_or_else(|| err(format!("no kernarg image for node {node}")))?;
        let word = |at: usize| -> HipResult<u64> {
            image
                .get(at..at + 8)
                .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
                .ok_or_else(|| err(format!("node {node}: short kernarg image")))
        };
        let dst = word(railgun::index_bind::abi::DST as usize)?;
        let h = u64::from(
            image
                .get(24..28)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
                .ok_or_else(|| err(format!("node {node}: short kernarg image")))?,
        );
        let row = h
            .checked_mul(258)
            .ok_or_else(|| err(format!("node {node}: R overflows")))?;
        let tensor = kv
            .k_gpu
            .iter()
            .chain(&kv.v_gpu)
            .find(|t| {
                let base = t.buf.as_ptr() as usize as u64;
                dst >= base && dst - base < t.buf.size() as u64
            })
            .ok_or_else(|| err(format!("node {node}: dst {dst:#x} is not a KV K/V tensor")))?;
        let a = alloc_of(gpu, tensor);
        let reserved = gpu
            .vmm_reserved_bytes(tensor)
            .map_or(a.bytes, |v| v as u64);
        let slice_end = (kv.physical_cap as u64)
            .checked_mul(row)
            .filter(|end| *end <= reserved)
            .ok_or_else(|| {
                err(format!(
                    "node {node}: {} rows of {row} bytes exceed the tensor's {reserved}-byte reservation",
                    kv.physical_cap
                ))
            })?;
        out.push(WriterExtent {
            node,
            dst_slice: 0..slice_end,
            mapped: vec![0..a.bytes],
            mapping_generation: a.generation,
        });
    }
    Ok(out)
}

/// `(node, case digest)` of every conditional node: what a bind of this plan
/// discharges, the only witnesses the manifest admission is shown.
pub(crate) fn assumed_cases(program: &KernelProgram) -> Vec<(usize, String)> {
    program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| n.indexed.as_ref().map(|c| (i, c.case_digest.clone())))
        .collect()
}

/// A queue that does nothing: twin mode executes the bound images through HIP
/// before `submit`, so the window state machine only needs a queue that
/// accepts and retires.
struct TwinQueue;

impl WindowQueue for TwinQueue {
    fn submit(&mut self, _: &[NodeImage<'_>], steps: &[ResolvedStep]) -> Result<(), QueueFailure> {
        if steps.is_empty() {
            Ok(())
        } else {
            Err(QueueFailure {
                message: "twin queue: typed steps are program nodes on the railgun route".into(),
                quiescence: Quiescence::Proven,
            })
        }
    }

    fn wait(&mut self) -> Result<(), QueueFailure> {
        Ok(())
    }
}

fn le_u32(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
}

fn le_u64(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 8).map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
}

/// The bound window as ordinary HIP operations, in node order: every kernel
/// node as a launch of its exact bound explicit kernarg bytes (zero-padded to
/// 16, the HIP `extra` blob the incumbent launches use) and bound grid; the
/// transport nodes as the raw D2D copy / word fill their bound parameters
/// describe (the A5 twin's mapping), so a bad pointer in a copy or fill node
/// faults here too.
pub(crate) fn twin_ops(
    program: &KernelProgram,
    images: &[&[u8]],
    grids: &[[u32; 3]],
) -> Result<Vec<rdna_compute::plan_sink::PlannedOp>, String> {
    use rdna_compute::plan_sink::{PlannedKernel, PlannedOp};
    if images.len() != program.nodes.len() || grids.len() != program.nodes.len() {
        return Err(format!(
            "{} images and {} grids for {} nodes",
            images.len(),
            grids.len(),
            program.nodes.len()
        ));
    }
    let mut ops = Vec::with_capacity(program.nodes.len());
    for (i, n) in program.nodes.iter().enumerate() {
        let image = images[i];
        let want = |v: Option<u64>, what: &str| v.ok_or_else(|| format!("node {i} {}: {what} outside its image", n.symbol));
        let want32 = |v: Option<u32>, what: &str| v.ok_or_else(|| format!("node {i} {}: {what} outside its image", n.symbol));
        match n.symbol.as_str() {
            railgun::copy::COPY_SYMBOL => ops.push(PlannedOp::Copy {
                src: want(le_u64(image, 0), "src")?,
                dst: want(le_u64(image, 8), "dst")?,
                bytes: u64::from(want32(le_u32(image, 16), "bytes")?),
            }),
            railgun::copy::FILL_SYMBOL => ops.push(PlannedOp::Fill {
                dst: want(le_u64(image, 0), "dst")?,
                value: want32(le_u32(image, 8), "value")?,
                words: want32(le_u32(image, 12), "words")?,
            }),
            _ => {
                let explicit = n.explicit_bytes as usize;
                if image.len() < explicit {
                    return Err(format!(
                        "node {i} {}: image of {} bytes is shorter than its {explicit} explicit bytes",
                        n.symbol,
                        image.len()
                    ));
                }
                let mut kernarg = vec![0u8; explicit.next_multiple_of(16)];
                kernarg[..explicit].copy_from_slice(&image[..explicit]);
                ops.push(PlannedOp::Kernel(PlannedKernel {
                    module: n.kref.as_ref().map(|k| k.module.clone()).unwrap_or_default(),
                    symbol: n.symbol.clone(),
                    code_object_digest: [0; 32],
                    grid: grids[i],
                    block: n.block,
                    dynamic_lds: n.dynamic_lds,
                    kernarg,
                    patches: Vec::new(),
                    label: "railgun.twin",
                }));
            }
        }
    }
    Ok(ops)
}

fn fnv64(chunks: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in chunks {
        for &b in *c {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The `HIPFIRE_DFLASH_RAILGUN_DEBUG` line of one bound window.
pub(crate) fn trace_line(
    prepared: &PreparedWindow,
    shape: &WindowShape,
    note: &PrepNote,
    manifest: bool,
    certified: bool,
) -> String {
    use railgun::kernel::PatchKind;
    let plan = prepared.plan();
    let program = plan.program();
    let images: Vec<&[u8]> = (0..program.nodes.len())
        .map(|i| prepared.kernarg_image(i).unwrap_or(&[]))
        .collect();
    let (mut scalar, mut interior, mut grid) = (0usize, 0usize, 0usize);
    for p in plan.patches() {
        match p {
            PatchKind::Scalar32 { .. } => scalar += 1,
            PatchKind::Interior64 { .. } => interior += 1,
            PatchKind::GridAxis { .. } => grid += 1,
        }
    }
    // Per resource: the min/max pointer any node's bound image holds for it,
    // flagged when outside the bound allocation.
    let allocations = &prepared.key().allocations;
    let mut range: Vec<Option<(u64, u64)>> = vec![None; allocations.len()];
    for (i, n) in program.nodes.iter().enumerate() {
        for (offset, arg) in &n.args {
            let Arg::Resource { resource, .. } = arg else { continue };
            let Some(ptr) = le_u64(images[i], *offset as usize) else { continue };
            if let Some(slot) = range.get_mut(resource.0 as usize) {
                *slot = Some(match *slot {
                    None => (ptr, ptr),
                    Some((lo, hi)) => (lo.min(ptr), hi.max(ptr)),
                });
            }
        }
    }
    let resources: Vec<String> = range
        .iter()
        .zip(allocations)
        .enumerate()
        .filter_map(|(r, (rng, a))| {
            rng.map(|(lo, hi)| {
                let bad = lo < a.base || hi > a.base + a.bytes;
                format!(
                    "r{r}[{:#x}+{}]={lo:#x}..{hi:#x}{}",
                    a.base,
                    a.bytes,
                    if bad { " OOB" } else { "" }
                )
            })
        })
        .collect();
    let l = shape.logical_ctx_rows;
    let delta = l - shape.projected_rows.max(l.saturating_sub(shape.swa_w));
    format!(
        "[dflash-railgun] window gen={} pos={} delta={delta} ctx={l} projected={} ring_head={} \
         family=fc{:?}/cat{} l_domain={}..={} key={:016x} cache={} patched_words={} \
         (scalar32={scalar} interior64={interior} grid={grid}) nodes={} image_digest={:016x} \
         manifest={} certified={certified} resources: {}",
        shape.generation,
        shape.position,
        shape.projected_rows,
        shape.ring_head,
        note.fc_rows,
        note.concat_segments,
        note.l_domain.start(),
        note.l_domain.end(),
        note.key_hash,
        if note.hit { "hit" } else { "miss" },
        scalar + interior + grid,
        program.nodes.len(),
        fnv64(&images),
        if manifest { "on" } else { "off" },
        resources.join(" ")
    )
}

impl WindowBuilder for RailgunWindowBuilder {
    fn prepare<'c>(
        &mut self,
        cache: &'c mut WindowCache,
        request: &mut WindowRequest<'_>,
    ) -> Result<(&'c mut PreparedWindow, PreparedInputs), Refusal> {
        self.ensure_ready(request)?;
        let execution = request.execution;
        let position = request.shape.position as usize;
        if execution == WindowExecution::Whole16 {
            // The same host/mapping step the HIP verify performs before the
            // graph: the KV mapping must cover this window's rows.
            request
                .target
                .kv_cache
                .ensure_mapped_capacity(&mut *request.gpu, position + ROWS)
                .map_err(|e| refuse(format!("KV mapped capacity: {e}")))?;
        }
        let family = draft_family_key(&request.shape).map_err(refuse)?;
        let scratch_epoch = rdna_compute::scratch::scratch_growth_epoch();
        let vmm = kv_vmm_allocs(request.gpu, &request.target.kv_cache);
        let selector = request.draft_weights.has_candidate_selector();
        let builder_key = BuilderKey {
            execution,
            fc_rows: family.fc_rows.clone(),
            concat_segments: family.concat_segments,
            tape: request.gdn_tape.is_some(),
            selector,
            scratch_epoch,
            allocations: fingerprint(
                request.gpu,
                request,
                &vmm,
                self.resources.as_ref().map(|r| r.positions.sealed.identity()),
            ),
        };

        let (prepared, scalars_required, note): (&'c mut PreparedWindow, Vec<String>, PrepNote) =
            match self
                .known
                .get(&builder_key)
                .and_then(|v| v.iter().find(|k| k.l_domain.contains(&request.shape.logical_ctx_rows)))
                .cloned()
            {
                Some(known) => {
                    let mut rebuilt: Option<TopologyKey> = None;
                    let p = cache.get_or_prepare(&known.topology, || {
                        let b = self.build(request, &vmm, scratch_epoch)?;
                        rebuilt = Some(b.topology);
                        Ok(b.prepared)
                    })?;
                    if let Some(t) = &rebuilt {
                        if *t != known.topology {
                            return Err(refuse(
                                "a re-planned window has a different topology than the cached one",
                            ));
                        }
                    }
                    let note = PrepNote {
                        hit: rebuilt.is_none(),
                        key_hash: topology_hash(&known.topology),
                        fc_rows: family.fc_rows.clone(),
                        concat_segments: family.concat_segments,
                        l_domain: known.l_domain.clone(),
                    };
                    (p, known.scalars_required, note)
                }
                None => {
                    let Built {
                        prepared,
                        topology,
                        scalars_required,
                        l_domain,
                    } = self.build(request, &vmm, scratch_epoch)?;
                    self.known.entry(builder_key).or_default().push(KnownTopology {
                        topology: topology.clone(),
                        scalars_required: scalars_required.clone(),
                        l_domain: l_domain.clone(),
                    });
                    let p = cache.get_or_prepare(&topology, move || Ok(prepared))?;
                    let note = PrepNote {
                        hit: false,
                        key_hash: topology_hash(&topology),
                        fc_rows: family.fc_rows.clone(),
                        concat_segments: family.concat_segments,
                        l_domain,
                    };
                    (p, scalars_required, note)
                }
            };

        // Debug arm: a skeleton found in the cache binds pointers planned
        // earlier. Probe every resource against the HIP runtime NOW: one that is
        // gone, moved or resized is a stale pointer the cache key missed.
        if self.debug && note.hit {
            let stale = stale_resources(&*request.gpu, prepared.plan().program());
            if !stale.is_empty() {
                eprintln!(
                    "[dflash-railgun] STALE skeleton (key {:016x}, scratch epoch {scratch_epoch}): {} of {} resources no longer match the live allocation: {}",
                    note.key_hash,
                    stale.len(),
                    prepared.plan().program().resources.len(),
                    stale.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
                );
            }
        }
        self.note = Some(note);

        // ---- this window's inputs ------------------------------------------
        let mut inputs = window_inputs(
            request.draft_scratch.thlog.abs_positions(),
            position,
            request.target.kv_cache.compact_offset as i32,
            request.seed,
            request.mask_token,
            request.shape.swa_w as usize,
            request.hidden_rb.written,
            &scalars_required,
        )
        .map_err(refuse)?;
        // GDN requant frames. The incumbent HIP-graph verify bakes the frame at
        // capture and a replay reserves none, so the railgun route supplies the
        // plan's authored frame and reserves nothing (the global counter stays
        // where HIP leaves it). That is byte-exact only where the frame cannot
        // matter, which `admit_window` guarantees for `Whole16`
        // (`gdn_frame_independent`): with error feedback on, the Q8 requant is
        // deterministic round-to-nearest and never reads `frame`
        // (`kernels/src/gated_delta_net_q8.hip:158-163`,
        // `gated_delta_net_q8_fast.hip:533-535`, `..._lanes_varrows.hip`: `frame`
        // appears only in the `else` stochastic seed); FP32 and Q4 state take no
        // frame at all. EF off is refused as `gdn_frame_dependent`.
        for (id, _frames) in gdn_words(prepared.plan().program()) {
            let authored = prepared.plan().program().words[id.0 as usize].authored;
            inputs.gdn_frames.insert(id, authored);
        }

        // ---- read-back views ---------------------------------------------------
        let pbs = request
            .verify_scratch
            .prefill_batch
            .as_ref()
            .ok_or_else(|| refuse("verify scratch has no PrefillBatchScratch"))?;
        self.reads = Some(ReadViews {
            argmax: request.verify_scratch.argmax.buf.byte_view(0, ROWS * 4),
            tokens: pbs.tokens.buf.byte_view(0, ROWS * 4),
            selector,
        });

        let mut shape: WindowShape = request.shape;
        // No declared patch reads `max_tiles` (verify attention is the graph
        // body at physical capacity): the partial-buffer geometry is covered by
        // the binding key's scratch capacities.
        shape.max_tiles = 0;
        let bindings = WindowBindings {
            key: prepared.key().clone(),
        };
        Ok((
            prepared,
            PreparedInputs {
                shape,
                bindings,
                inputs,
            },
        ))
    }

    fn admit(&mut self, prepared: &PreparedWindow, bound: &mut BoundWindow) -> Result<(), String> {
        self.admitted = None;
        self.certified = false;
        let Some(path) = self.manifest.clone() else {
            return Ok(());
        };
        // Only a Whole16 plan has a manifest; a HostSplice draft segment is
        // never certified exact.
        if prepared.plan().execution() != WindowExecution::Whole16 {
            return Ok(());
        }
        let corpus = self
            .corpus
            .as_mut()
            .ok_or("manifest admission before the corpus is open")?;
        let program = prepared.plan().program();
        let nodes: Vec<(String, String)> = program
            .nodes
            .iter()
            .map(|n| {
                (
                    n.symbol.clone(),
                    n.kref.as_ref().map(|k| k.artifact_sha256.clone()).unwrap_or_default(),
                )
            })
            .collect();
        // The cases the bind of this plan will discharge are a function of the
        // plan alone (each conditional node's receipt case digest); the real
        // bind is checked against them after the witness is attached.
        let assumed = assumed_cases(program);
        let cases: Vec<(usize, &str, &str)> = assumed
            .iter()
            .map(|(node, case)| (*node, case.as_str(), WITNESS_INDEXED_BIND))
            .collect();
        corpus.admit_manifest_witnessed(&path, &nodes, &cases)?;
        if assumed.is_empty() {
            prepared
                .certify_exact(bound)
                .map_err(|e| format!("certify exact: {e}"))?;
            self.certified = true;
        } else {
            self.admitted = Some(assumed);
        }
        Ok(())
    }

    fn trace_window(&mut self, prepared: &PreparedWindow, shape: &WindowShape) {
        if !self.debug {
            return;
        }
        let Some(note) = self.note.as_ref() else {
            return;
        };
        eprintln!("{}", trace_line(prepared, shape, note, self.manifest.is_some(), self.certified));
    }

    fn twin_run(&mut self, gpu: &mut Gpu, prepared: &PreparedWindow) -> HipResult<bool> {
        if !self.twin {
            return Ok(false);
        }
        let program = prepared.plan().program();
        let images: Vec<&[u8]> = (0..program.nodes.len())
            .map(|i| prepared.kernarg_image(i).ok_or_else(|| err(format!("no bound image for node {i}"))))
            .collect::<HipResult<_>>()?;
        let grids: Vec<[u32; 3]> = (0..program.nodes.len())
            .map(|i| prepared.grid(i).ok_or_else(|| err(format!("no bound grid for node {i}"))))
            .collect::<HipResult<_>>()?;
        let ops = twin_ops(program, &images, &grids).map_err(err)?;
        gpu.run_planned_ops(&ops)?;
        // The queue is a no-op in twin mode: this is the quiescence proof.
        if let Some(stream) = gpu.active_stream.as_ref() {
            gpu.hip.stream_synchronize(stream)?;
        }
        Ok(true)
    }

    fn stage(&mut self, gpu: &Gpu, inputs: &WindowInputs, stream: &Stream) -> HipResult<()> {
        let res = self
            .resources
            .as_mut()
            .ok_or_else(|| err("railgun window: staging before prepare"))?;
        let k_words = inputs.k_position_suffix.len() + inputs.q_positions.len();
        if K_OFF / 4 + k_words > res.staging_words {
            return Err(err(format!(
                "railgun window: {k_words} K positions exceed the staging buffer ({} words)",
                res.staging_words
            )));
        }
        // A previous window's refused/failed attempt may have left its staging
        // upload marked in flight (its fence wait failed): the owner proved the
        // stream idle, so retire it before repacking.
        res.pinned.complete(gpu)?;
        let sections = staging_sections(inputs);
        let refs: Vec<&[u32]> = sections.iter().map(Vec::as_slice).collect();
        let offsets = res.pinned.pack(&refs)?;
        if offsets[..4] != [SEED_OFF, IDS_OFF, Q_OFF, VERIFY_OFF] || offsets[4] != K_OFF {
            return Err(err(format!(
                "railgun window: staging layout {offsets:?} is not the planned one"
            )));
        }
        res.pinned.submit(gpu, &res.staging.buf, stream)
    }

    fn await_staged(&mut self, gpu: &Gpu) -> HipResult<()> {
        self.resources
            .as_mut()
            .ok_or_else(|| err("railgun window: await before prepare"))?
            .pinned
            .complete(gpu)
    }

    fn seal_positions(
        &mut self,
        request: &mut WindowRequest<'_>,
        prepared: &PreparedWindow,
        bound: &mut BoundWindow,
        inputs: &WindowInputs,
    ) -> HipResult<()> {
        let plan = prepared.plan();
        if plan.indexed_positions().is_none() {
            return Ok(());
        }
        let ierr = |e: railgun::index_bind::IndexBindError| err(format!("sealed positions: {e}"));
        // The bytes to upload are the verify positions the staging pack hands
        // `pbs.positions` (`inputs.verify_positions`); the producer they must
        // equal is the physical-position vector of the HIP verify at this
        // window's position (`start_pos + i`), computed independently.
        let positions: Vec<i32> = inputs
            .verify_positions
            .iter()
            .map(|p| i32::try_from(*p).map_err(|_| err(format!("verify position {p} exceeds i32"))))
            .collect::<HipResult<_>>()?;
        let producer: Vec<i32> = (0..POSITION_ROWS as u64)
            .map(|i| {
                i32::try_from(request.shape.position + i)
                    .map_err(|_| err("window position exceeds i32"))
            })
            .collect::<HipResult<_>>()?;
        let array: [i32; POSITION_ROWS] = positions
            .as_slice()
            .try_into()
            .map_err(|_| err(format!("{} verify positions, need {POSITION_ROWS}", positions.len())))?;
        let extents = writer_extents(&*request.gpu, &request.target.kv_cache, prepared)?;

        let res = self
            .resources
            .as_mut()
            .ok_or_else(|| err("railgun window: sealing before prepare"))?;
        let pos = &mut res.positions;
        if plan.indexed_positions().map(|r| prepared.key().allocations[r.0 as usize].base)
            != Some(pos.base())
        {
            return Err(err("the plan's conditional writers do not read the sealed positions"));
        }
        let checked = pos.sealed.host_check(&positions, &producer).map_err(ierr)?;
        pos.stage.fill(&array).map_err(ierr)?;
        let gpu: &Gpu = &*request.gpu;
        let stream = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| err("owner stream missing"))?;
        let mut transport = DeviceTransport {
            gpu,
            stream,
            dev: &pos.buf,
            host: &mut pos.host,
            fence: &pos.fence,
        };
        let token = pos
            .sealed
            .begin_upload(checked, &pos.stage, &mut transport)
            .map_err(ierr)?;
        pos.sealed.complete_upload(token, &mut transport).map_err(ierr)?;
        prepared
            .attach_indexed_witness(bound, &pos.sealed, &extents)
            .map_err(|e| err(format!("sealed positions bind: {e}")))?;

        // Certified exact: the manifest admitted this plan in `admit` assuming
        // the cases this bind now actually discharged. They must be exactly
        // those (a bind that discharges anything else is refused), and only
        // then is the window certified exact.
        if let Some(assumed) = self.admitted.take() {
            let mut got: Vec<(usize, String)> = bound
                .indexed_witness()
                .map(|w| {
                    w.discharged_cases()
                        .into_iter()
                        .map(|(node, case, _)| (node, case.to_owned()))
                        .collect()
                })
                .unwrap_or_default();
            got.sort();
            let mut want = assumed;
            want.sort();
            if got != want {
                return Err(err(format!(
                    "sealed positions: the bind discharged {got:?}, the manifest admitted {want:?}"
                )));
            }
            prepared
                .certify_exact(bound)
                .map_err(|e| err(format!("certify exact: {e}")))?;
            self.certified = true;
        }
        Ok(())
    }

    fn abandon_positions(&mut self) {
        self.admitted = None;
        self.certified = false;
        if let Some(r) = self.resources.as_ref() {
            // Nothing was queued: a sealed or host-checked window retires.
            if let Err(e) = retire_unsubmitted(&r.positions.sealed) {
                eprintln!("[dflash] railgun window: sealed positions not retired after a refusal: {e}");
            }
        }
    }

    fn read_results(&mut self, gpu: &Gpu, execution: WindowExecution) -> HipResult<WindowResults> {
        let reads = self
            .reads
            .as_ref()
            .ok_or_else(|| err("railgun window: read_results before prepare"))?;
        let block = if reads.selector {
            let blk = self
                .resources
                .as_ref()
                .and_then(|r| r.block.as_ref())
                .ok_or_else(|| err("railgun window: no selector block"))?;
            let rb = blk.read_back(gpu)?;
            if rb.status != 0 {
                return Err(err(format!(
                    "railgun window: selector status {:#x} (malformed input)",
                    rb.status
                )));
            }
            rb.block
        } else {
            let mut bytes = vec![0u8; ROWS * 4];
            gpu.hip.memcpy_dtoh(&mut bytes, &reads.tokens)?;
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let argmax_per_pos = match execution {
            WindowExecution::Whole16 => {
                let mut bytes = vec![0u8; ROWS * 4];
                gpu.hip.memcpy_dtoh(&mut bytes, &reads.argmax)?;
                Some(
                    bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            WindowExecution::HostSplice => None,
        };
        Ok(WindowResults {
            block,
            argmax_per_pos,
        })
    }

    fn invalidate(&mut self) {
        self.known.clear();
        self.reads = None;
    }

    fn release(self: Box<Self>, gpu: &mut Gpu) {
        (*self).free(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(position: usize, abs: &[i32], compact: i32) -> WindowInputs {
        window_inputs(abs, position, compact, 1234, 99, 8, 100, &[
            SCALAR_DRAFT_SEED.to_owned(),
            SCALAR_RING_WRITTEN.to_owned(),
        ])
        .expect("inputs")
    }

    #[test]
    fn staging_layout_offsets_are_the_planned_ones() {
        assert_eq!((SEED_OFF, IDS_OFF, Q_OFF, VERIFY_OFF, K_OFF), (0, 4, 68, 132, 196));
        assert_eq!(LAYOUT.seed_off, 0);
        assert_eq!(LAYOUT.ids_off, 4);
        assert_eq!(LAYOUT.q_off, 68);
        assert_eq!(LAYOUT.k_off, 196);
        assert_eq!(staging_words(2048), 49 + 2048 + 16);
    }

    #[test]
    fn staged_words_equal_the_eager_draft_uploads() {
        // 12 live rows, W = 8: the suffix is the last 8 absolute positions,
        // compaction offset included (never 0..L).
        let abs: Vec<i32> = (0..12).map(|i| 1000 + i * 3).collect();
        let position = 12usize;
        let compact = 7i32;
        let w = inputs(position, &abs, compact);
        let eager = draft_staged_words(&abs, position, compact, 1234, 99, ROWS, 8);
        let [seed, ids, q, verify, k] = staging_sections(&w);
        assert_eq!(seed, vec![eager.seed]);
        assert_eq!(ids, eager.noise_ids);
        assert_eq!(q, eager.q_positions);
        assert_eq!(k, eager.k_positions);
        assert_eq!(k.len(), 8 + ROWS);
        assert_eq!(&k[..8], &[1012, 1015, 1018, 1021, 1024, 1027, 1030, 1033]);
        assert_eq!(verify, (12u32..28).collect::<Vec<_>>());
        assert_eq!(w.q_positions[0], 12 + 7);
        // Scalars the plan declares.
        assert_eq!(w.scalars.get(SCALAR_DRAFT_SEED), Some(&1234));
        assert_eq!(w.scalars.get(SCALAR_RING_WRITTEN), Some(&100));
        // Sections concatenate to exactly the pinned layout.
        let mut words = 0usize;
        for (section, off) in [(&seed, SEED_OFF), (&ids, IDS_OFF), (&q, Q_OFF), (&verify, VERIFY_OFF)] {
            assert_eq!(off / 4, words, "section starts where the previous one ended");
            words += section.len();
        }
        assert_eq!(words, K_OFF / 4);
    }

    #[test]
    fn skeletons_with_different_l_domains_have_different_topology_keys() {
        // Same family, same allocations: only the attention-block domain differs.
        let a = context_mode(2048, 2048, &(1..=112));
        let b = context_mode(2048, 2048, &(113..=240));
        assert_ne!(a, b);
        assert_eq!(a, context_mode(2048, 2048, &(1..=112)));
    }

    struct MockTransport {
        begin: Result<(), Quiescence>,
    }

    impl PositionsTransport for MockTransport {
        fn begin_h2d(&mut self, _: u64, _: &[u8; POSITION_BYTES as usize]) -> Result<(), TransportFailure> {
            self.begin.map_err(|quiescence| TransportFailure {
                message: "mock".into(),
                quiescence,
            })
        }
        fn wait_h2d(&mut self) -> Result<(), TransportFailure> {
            Ok(())
        }
        fn acquire(&mut self) -> Result<(), TransportFailure> {
            Ok(())
        }
    }

    fn sealed() -> SealedPositions {
        SealedPositions::new(PositionsIdentity {
            base: 0x1000,
            bytes: 4096,
            generation: 1,
        })
        .expect("sealed positions")
    }

    fn rows(base: i32) -> Vec<i32> {
        (0..POSITION_ROWS as i32).map(|i| base + i).collect()
    }

    /// A window refused after `seal_positions` (or after only the host check)
    /// must leave the positions `Retired` so the HIP fallback and the next
    /// Whole16 window start from a clean lease, and the next window must be
    /// able to host-check and seal afresh.
    #[test]
    fn a_window_refused_after_the_seal_retires_the_positions_for_the_next_one() {
        use railgun::index_bind::PositionsState as S;
        let sp = sealed();
        let stage = PositionsStage::new();
        let mut t = MockTransport { begin: Ok(()) };
        let seal = |sp: &SealedPositions, base: i32, t: &mut MockTransport| {
            let r = rows(base);
            let checked = sp.host_check(&r, &r).expect("host check");
            stage.fill(&r.clone().try_into().unwrap()).unwrap();
            let token = sp.begin_upload(checked, &stage, t).expect("begin");
            sp.complete_upload(token, t).expect("complete");
        };
        // Sealed, then refused (e.g. the manifest or the witness refuses).
        seal(&sp, 251, &mut t);
        assert_eq!(sp.state(), S::SealedReady);
        retire_unsubmitted(&sp).expect("a sealed, unsubmitted window retires");
        assert_eq!(sp.state(), S::Retired);
        assert!(!stage.is_in_flight(), "the stage is released after the fence");
        // Refused after only the host check.
        let r = rows(265);
        sp.host_check(&r, &r).expect("host check");
        assert_eq!(sp.state(), S::HostChecked);
        retire_unsubmitted(&sp).expect("a host-checked window retires");
        assert_eq!(sp.state(), S::Retired);
        retire_unsubmitted(&sp).expect("idempotent");
        // The next window seals afresh with its own positions.
        seal(&sp, 280, &mut t);
        assert_eq!(sp.state(), S::SealedReady);
        retire_unsubmitted(&sp).unwrap();
        // A transport that may still touch P is not retired: the error is
        // reported and the quarantine is left alone.
        let r = rows(300);
        let checked = sp.host_check(&r, &r).unwrap();
        stage.fill(&r.clone().try_into().unwrap()).unwrap();
        let mut bad = MockTransport { begin: Err(Quiescence::Unknown) };
        assert!(sp.begin_upload(checked, &stage, &mut bad).is_err());
        assert_eq!(sp.state(), S::Quarantined);
        assert!(retire_unsubmitted(&sp).is_err());
        assert_eq!(sp.state(), S::Quarantined);
    }

    #[test]
    fn a_sequence_directory_gets_one_file_per_distinct_sequence() {
        let dir = std::env::temp_dir().join(format!("rg-seq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let node = |s: &str| SequenceNode {
            phase: "p".into(),
            symbol: s.into(),
            module: "m".into(),
        };
        let a = vec![node("a"), node("b")];
        let b = vec![node("a"), node("b"), node("c")];
        write_sequence(&dir, &a).unwrap();
        write_sequence(&dir, &b).unwrap();
        write_sequence(&dir, &a).unwrap();
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(files.len(), 2, "a repeated sequence rewrites its own file");
        files.sort_by_key(|f| f.metadata().unwrap().len());
        assert!(files[0].file_name().to_string_lossy().ends_with(".seq.jsonl"));
        // A plain file path keeps the last plan.
        let file = dir.join("last.jsonl");
        write_sequence(&file, &a).unwrap();
        write_sequence(&file, &b).unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 3);
    }

    /// A scratch slot that grew freed its old buffer: a skeleton planned before
    /// it must never be fetched afterwards (its bound pointers are stale, and
    /// the freed range may be reused, e.g. by the kernarg pool). The epoch is
    /// part of both cache keys.
    #[test]
    fn a_scratch_growth_epoch_change_is_a_cache_miss() {
        let key = |scratch_epoch| BuilderKey {
            execution: WindowExecution::Whole16,
            fc_rows: vec![9],
            concat_segments: 1,
            tape: true,
            selector: true,
            allocations: vec![Alloc { base: 0x1000, bytes: 64, generation: 0 }],
            scratch_epoch,
        };
        assert_eq!(key(3), key(3));
        assert_ne!(key(3), key(4));
        let mut known: HashMap<BuilderKey, u8> = HashMap::new();
        known.insert(key(3), 1);
        assert!(known.get(&key(4)).is_none(), "a grown scratch slot finds no skeleton");
        let before = rdna_compute::scratch::scratch_growth_epoch();
        // `note_scratch_growth` is crate-private to rdna-compute: its effect is
        // covered there; here the public epoch is simply monotone.
        assert!(rdna_compute::scratch::scratch_growth_epoch() >= before);
    }

    #[test]
    fn unknown_required_scalars_are_refused() {
        assert!(window_inputs(&[], 0, 0, 1, 2, 8, 0, &["mystery".to_owned()]).is_err());
    }

    #[test]
    fn gdn_words_are_found_in_node_order() {
        use railgun::kernel::ProgramWord;
        let program = KernelProgram {
            arch: "gfx1201".into(),
            resources: vec![],
            words: vec![
                ProgramWord {
                    node: 1,
                    offset: 16,
                    role: WordRole::Window,
                    authored: 0,
                },
                ProgramWord {
                    node: 3,
                    offset: 92,
                    role: WordRole::GdnFrame { frames: 2 },
                    authored: 5,
                },
                ProgramWord {
                    node: 9,
                    offset: 92,
                    role: WordRole::GdnFrame { frames: 1 },
                    authored: 7,
                },
            ],
            nodes: vec![],
        };
        assert_eq!(gdn_words(&program), [(WordId(1), 2), (WordId(2), 1)]);
    }
}
