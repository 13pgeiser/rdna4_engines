// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 dense tensor parallelism on the generic mesh executor (#927 item 3).
//!
//! Every TP rank holds the whole model with each dense FFN sharded along the
//! intermediate ([`crate::weights::Qwen4Weights::shard_dense_ffn`]). The split
//! plan ([`qwen4_tp_plan`]) runs every other op replicated and reduces each
//! dense layer's partial FFN output ([`QWEN4_FFN_OUTPUT_RESOURCE`]) with one
//! rank-ordered `AllReduce` before its `HyperWrite`, so a rank's program runs
//! in one segment per collective ([`Qwen4TpBindings::run_ops`]).

use crate::config::{Qwen4Config, Qwen4FfnConfig};
use crate::gpu_forward::Qwen4MeshHead;
use crate::mesh_bindings::Qwen4RankBand;
use crate::program::{qwen4_step_graph_with_output, QWEN4_FFN_OUTPUT_RESOURCE};
use crate::weights::{Qwen4BandScope, Qwen4Manifest};
use hipfire_dispatch::pipeline::step_graph::{OpId, ResourceClass};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{split_program, CollectiveKind, PlannedCollective, SplitPlan};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::EdgeDescriptor;
use hipfire_runtime::weight_manifest::plan_manifest;
use rdna_compute::Gpu;

type Result<T, E = String> = std::result::Result<T, E>;

/// The TP split plan of the trunk program (`head`/`top1` as in
/// [`qwen4_step_graph_with_output`]) on a pure TP `mesh`. The splitter plans
/// only the graph's declared weights (the dense FFN); the manifest's other TP
/// declarations (attention, GDN, HC) are not run by this slice, so they are
/// left out of the plan rather than admitted as if split.
pub fn qwen4_tp_plan(
    config: &Qwen4Config,
    mesh: &DeviceMesh,
    head: bool,
    top1: bool,
) -> Result<SplitPlan> {
    if mesh.size_of(DimKind::Tp) != mesh.n_devices() {
        return Err("Qwen4 TP plan needs a pure TP mesh".into());
    }
    let graph = qwen4_step_graph_with_output(config, MeshRoute::Prefill, head, top1)?;
    let declared = |name: &str, layer: Option<usize>| {
        graph.resources().iter().any(|decl| {
            matches!(&decl.class, ResourceClass::Weight { name: n, layer: l, .. } if n == name && *l == layer)
        })
    };
    let weights: Vec<_> = Qwen4Manifest::build(config)
        .map_err(|error| error.to_string())?
        .weights
        .into_iter()
        .filter(|entry| declared(&entry.name, entry.layer))
        .collect();
    let layers = config.num_hidden_layers;
    let placement = BandPlacement::uniform(mesh, layers)?;
    let plan = plan_manifest(&weights, &[], mesh, layers, &placement)?;
    split_program(&graph, &plan, mesh).map_err(|error| format!("Qwen4 TP split: {error}"))
}

/// One epoch adapter over the TP ranks (rank `i` on device `i`): borrows each
/// rank's whole-model bundle and forward, never a GPU. Session, checkpoint
/// and prefix-repair lifecycles are not admitted on TP yet and are refused.
pub struct Qwen4TpBindings<'a> {
    ranks: Vec<Qwen4RankBand<'a>>,
    tokens: &'a [u32],
    /// One head per rank (every rank computes the replicated head).
    heads: Vec<Option<Qwen4MeshHead<'a>>>,
    tp_epoch: u64,
    prepared: Option<ForwardEpoch>,
    /// Per rank, the first op id its next segment must start at.
    next_op: Vec<u32>,
}

impl<'a> Qwen4TpBindings<'a> {
    /// Validate before any submission: two or more whole-model ranks of one
    /// config, no EP or lanes, and every dense FFN layer sharded into exactly
    /// `ranks.len()` parts. `tp_epoch` is `Gpus::mesh_tp_epoch()`.
    pub fn new(ranks: Vec<Qwen4RankBand<'a>>, tp_epoch: u64) -> Result<Self> {
        let parts = ranks.len();
        let config = &ranks
            .first()
            .ok_or("Qwen4 TP mesh has no ranks")?
            .bundle
            .config;
        if parts < 2 || tp_epoch == 0 {
            return Err("Qwen4 TP needs two or more ranks and a provisioned TP group".into());
        }
        let intermediate = match config.ffn {
            Qwen4FfnConfig::Dense {
                intermediate_size, ..
            } => Some(intermediate_size),
            Qwen4FfnConfig::Moe { .. } => None,
        };
        for rank in &ranks {
            let bundle = &*rank.bundle;
            if &bundle.config != config
                || *bundle.weights.band_scope() != Qwen4BandScope::single(config)
                || *rank.forward.band_scope() != Qwen4BandScope::single(config)
                || bundle.ep.is_some()
                || bundle.lanes.is_some()
            {
                return Err(
                    "Qwen4 TP ranks must each hold the whole model of one config, without EP or lanes"
                        .into(),
                );
            }
            let sharded = bundle.weights.layer_refs.iter().all(|layer| {
                match (layer.ffn.dense(), intermediate) {
                    (Some(dense), Some(full)) => dense.gate.shape[0] * parts == full,
                    (None, None) => true,
                    _ => false,
                }
            });
            if !sharded {
                return Err(format!(
                    "Qwen4 TP rank dense FFN is not sharded into {parts} parts"
                ));
            }
        }
        Ok(Self {
            next_op: vec![0; parts],
            heads: vec![None; parts],
            ranks,
            tokens: &[],
            tp_epoch,
            prepared: None,
        })
    }

    /// Host token storage stays immutable until submission is drained.
    pub fn set_tokens(&mut self, tokens: &'a [u32]) {
        self.tokens = tokens;
        self.prepared = None;
    }

    /// One head per rank, or `None` everywhere to commit the trunk only.
    pub fn set_heads(&mut self, heads: Vec<Option<Qwen4MeshHead<'a>>>) -> Result<()> {
        if heads.len() != self.ranks.len()
            || heads
                .iter()
                .any(|head| head.is_some() != heads[0].is_some())
        {
            return Err("Qwen4 TP needs one head per rank, or none".into());
        }
        self.heads = heads;
        self.prepared = None;
        Ok(())
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Qwen4 TP epoch has not been prepared".into());
        }
        Ok(())
    }

    fn full_range(&self, owner: usize) -> std::ops::Range<u32> {
        let head = self.heads[owner].as_ref();
        self.ranks[owner]
            .forward
            .mesh_op_range(head.is_some(), head.is_some_and(|head| head.top1.is_some()))
    }

    fn unsupported(what: &str) -> String {
        format!("Qwen4 TP does not admit {what} yet")
    }
}

impl MeshStepBindings for Qwen4TpBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        if epoch.actual_rows == 0
            || self.tokens.len() != epoch.actual_rows
            || (epoch.route == MeshRoute::Decode && epoch.actual_rows != 1)
        {
            return Err("invalid Qwen4 TP rows or tokens".into());
        }
        let end = epoch
            .start_position
            .checked_add(epoch.actual_rows)
            .ok_or("Qwen4 TP position overflows")?;
        for rank in &self.ranks {
            let state = &rank.bundle.state;
            if state.position != epoch.start_position
                || end > state.max_seq_len
                || epoch.actual_rows > rank.forward.scratch.max_chunk
                || state.row_capture_armed
            {
                return Err("Qwen4 TP rank position, capacity or row capture disagrees".into());
            }
        }
        self.next_op.fill(0);
        self.prepared = Some(*epoch);
        Ok(())
    }

    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()> {
        self.require_prepared(epoch)?;
        let start = *self.next_op.get(owner).ok_or("unknown Qwen4 TP rank")?;
        let end = start + ops.len() as u32;
        if !ops.iter().map(|op| op.0).eq(start..end) {
            return Err("Qwen4 TP segment does not continue its rank program in order".into());
        }
        let head = self.heads[owner];
        let rank = &mut self.ranks[owner];
        rank.forward
            .forward_mesh_chunk(
                rank.bundle,
                gpu,
                self.tokens,
                epoch.route_rows,
                epoch.route,
                head,
                None,
                start..end,
            )
            .map_err(|error| error.to_string())?;
        self.next_op[owner] = end;
        Ok(())
    }

    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        self.require_prepared(epoch)?;
        if collective.kind != CollectiveKind::AllReduce
            || collective.operand != QWEN4_FFN_OUTPUT_RESOURCE
        {
            return Err("Qwen4 TP only reduces the dense FFN output".into());
        }
        let rank = self.ranks.get(owner).ok_or("unknown Qwen4 TP rank")?;
        let bytes = epoch
            .actual_rows
            .checked_mul(rank.bundle.config.hidden_size * std::mem::size_of::<f32>())
            .ok_or("Qwen4 TP FFN output bytes overflow")?;
        Ok(BoundaryIo {
            view: &rank.forward.scratch.ffn_output,
            into: None,
            descriptor: EdgeDescriptor {
                epoch: self.tp_epoch,
                op: epoch.op_id,
                rows: epoch.actual_rows,
                dtype_bytes: 4,
                bytes,
            },
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        let full = self.full_range(owner);
        let state = &self.ranks[owner].bundle.state;
        if self.next_op[owner] != full.end
            || state.position != epoch.start_position + epoch.actual_rows
            || state.qsa.iter().any(|qsa| qsa.position != state.position)
        {
            return Err("Qwen4 TP rank did not run its whole program and commit its marks".into());
        }
        Ok(())
    }

    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen4 TP rank")?;
        rank.bundle.reset(gpu).map_err(|error| error.to_string())
    }

    fn repair_prefix_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _keep: usize,
        _tokens: &[u32],
    ) -> Result<()> {
        Err(Self::unsupported("prefix repair"))
    }

    fn capture_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("session capture"))
    }

    fn probe_session_rank(
        &mut self,
        _owner: usize,
        _epoch: &ForwardEpoch,
        _piece: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn restore_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _checkpoint: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn checkpoint_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("checkpoints"))
    }

    fn restore_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    fn validate_finalize_rank(
        &mut self,
        _owner: usize,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    /// Unreachable: `validate_finalize_rank` refuses every ticket first.
    fn finalize_rank(&mut self, _: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_dispatch::pipeline::step_graph::OpKind;
    use hipfire_runtime::mesh_split::RankItem;

    /// The tiny developer dense profile (intermediate 8, 4 layers, PLE, MTP).
    fn dense_config() -> Qwen4Config {
        let ffn = Qwen4FfnConfig::Dense {
            intermediate_size: 8,
        };
        crate::config::synthetic_test_config(ffn, 16, 4, true, true)
    }

    /// Every rank runs the whole program; each dense layer's FFN output is
    /// reduced once, right after its `GatedMlp` and before its `HyperWrite`.
    #[test]
    fn tp_plan_reduces_each_dense_ffn_before_its_hyper_write() {
        let config = dense_config();
        let graph = qwen4_step_graph_with_output(&config, MeshRoute::Prefill, true, true).unwrap();
        for parts in [2, 4] {
            let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap();
            let plan = qwen4_tp_plan(&config, &mesh, true, true).unwrap();
            assert_eq!(plan.collectives.len(), config.num_hidden_layers);
            for (layer, c) in plan.collectives.iter().enumerate() {
                assert_eq!(c.kind, CollectiveKind::AllReduce);
                assert_eq!(c.operand, QWEN4_FFN_OUTPUT_RESOURCE);
                assert_eq!(c.group, (0..parts).collect::<Vec<_>>());
                let producer = &graph.ops()[c.producer.0 as usize];
                assert_eq!(
                    (producer.kind, producer.layer),
                    (OpKind::GatedMlp, Some(layer))
                );
            }
            let expected: Vec<RankItem> = graph
                .ops()
                .iter()
                .enumerate()
                .flat_map(|(index, op)| {
                    let item = RankItem::Op(OpId(index as u32));
                    let reduce = plan
                        .collectives
                        .iter()
                        .find(|c| c.producer.0 as usize == index)
                        .map(|c| RankItem::Collective(c.id));
                    std::iter::once(item).chain(reduce.filter(|_| op.kind == OpKind::GatedMlp))
                })
                .collect();
            for rank in &plan.ranks {
                assert_eq!(rank.items, expected, "rank {}", rank.device);
            }
        }
    }

    /// The tiny FFN (intermediate 8) splits two and four ways; three parts
    /// leave a remainder and are refused by the manifest.
    #[test]
    fn tp_plan_refuses_an_indivisible_intermediate() {
        let config = dense_config();
        for parts in [2, 4] {
            let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap();
            assert!(qwen4_tp_plan(&config, &mesh, true, false).is_ok());
        }
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 3)]).unwrap();
        let error = qwen4_tp_plan(&config, &mesh, true, false).unwrap_err();
        assert!(error.contains("not divisible by Tp=3"), "{error}");
    }
}

/// Emulated TP forward of a developer artifact through the generic executor,
/// against a single-device run of the same rank shards that sums their FFN
/// partials in rank order, and against the plain single card.
#[cfg(test)]
mod hw_tests {
    use super::*;
    use crate::bundle::Qwen4Bundle;
    use crate::gpu_forward::{Qwen4GpuForward, Qwen4OutputRows};
    use crate::{admit_hfqm_artifact, Qwen4KvBackend};
    use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
    use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank};
    use hipfire_runtime::mesh_split::RankItem;
    use hipfire_runtime::model_source::SourcePayload;
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
    use rdna_compute::{DType, GpuTensor};

    const ARTIFACT_ENV: &str = "HIPFIRE_QWEN4_DEV_ARTIFACT";
    const MAX_SEQ: usize = 64;
    const PROMPT: usize = 12;
    const GREEDY: usize = 16;

    /// The developer artifact assembled on `gpu` (as `dev_profiles_hw`).
    fn load(gpu: &mut Gpu) -> Qwen4Bundle {
        let path = std::env::var(ARTIFACT_ENV).expect("HIPFIRE_QWEN4_DEV_ARTIFACT");
        let mut hfq = HfqFile::open(std::path::Path::new(&path)).expect("open artifact");
        let receipt = admit_hfqm_artifact(&hfq).expect("artifact admission");
        crate::bundle::admit_executable_profile(&receipt.config).expect("executor admission");
        let max_seq = MAX_SEQ.min(receipt.config.max_position_embeddings);
        if gpu.is_uma() {
            hfq.drop_mmap();
        }
        let mesh = DeviceMesh::single().unwrap();
        let expected = WeightOrigin::for_single(&mesh, gpu);
        let source = HfqModelSource::from_hfq(hfq);
        let transaction = fulfill_manifest_from_payloads(
            &receipt.manifest.weights,
            &mesh,
            receipt.config.num_hidden_layers,
            gpu,
            expected,
            |entry| {
                source
                    .tensor_range(&entry.name)
                    .map_err(|e| e.to_string())?
                    .map(SourcePayload::Range)
                    .ok_or_else(|| format!("missing tensor '{}'", entry.name))
            },
        )
        .expect("weights");
        let state_format = crate::resolve_state_format(
            &hipfire_runtime::config::get().kv_mode,
            "",
            gpu,
            &receipt.config,
        )
        .expect("state format");
        let backend = Qwen4KvBackend::automatic(gpu);
        let mut bundle = Qwen4Bundle::assemble_with_metadata(
            receipt.config,
            &receipt.manifest.identity.recipe,
            transaction,
            &receipt.placements,
            gpu,
            max_seq,
            receipt.ple,
            state_format,
            backend,
        )
        .expect("assemble");
        bundle
            .attach_forward(
                gpu,
                max_seq,
                crate::gpu_forward::Qwen4ForwardScratchMode::Full,
            )
            .expect("attach_forward");
        bundle
    }

    fn argmax(row: &[f32]) -> u32 {
        let mut best = 0;
        for (index, value) in row.iter().enumerate() {
            if *value > row[best] {
                best = index;
            }
        }
        best as u32
    }

    fn fnv(rows: &[Vec<f32>]) -> u64 {
        rows.iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    /// Prompt logits plus `GREEDY` decode logits; token `i + 1` is the host
    /// argmax of logits `i`. `step(tokens, route)` runs one forward.
    fn greedy(
        prompt: &[u32],
        mut step: impl FnMut(&[u32], MeshRoute) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let mut rows = vec![step(prompt, MeshRoute::Prefill)];
        for _ in 0..GREEDY {
            let token = argmax(rows.last().unwrap());
            rows.push(step(&[token], MeshRoute::Decode));
        }
        rows
    }

    /// The op spans of rank 0's program, each with whether a collective
    /// follows it.
    fn segments(plan: &SplitPlan) -> Vec<(std::ops::Range<u32>, bool)> {
        let mut out = Vec::new();
        let (mut start, mut end) = (0, 0);
        for item in &plan.ranks[0].items {
            match *item {
                RankItem::Op(op) => end = op.0 + 1,
                RankItem::Collective(_) => {
                    out.push((start..end, true));
                    start = end;
                }
            }
        }
        if end > start {
            out.push((start..end, false));
        }
        out
    }

    fn take_forwards(bundles: &mut [Qwen4Bundle]) -> Vec<Qwen4GpuForward> {
        bundles
            .iter_mut()
            .map(|bundle| bundle.execution.take().expect("forward attached"))
            .collect()
    }

    fn restore_forwards(bundles: &mut [Qwen4Bundle], forwards: Vec<Qwen4GpuForward>) {
        for (bundle, forward) in bundles.iter_mut().zip(forwards) {
            bundle.execution = Some(forward);
        }
    }

    /// Map one rank's context through `rows` more tokens; returns the routing
    /// rows of the epoch.
    fn prepare_rank(
        bundle: &mut Qwen4Bundle,
        forward: &mut Qwen4GpuForward,
        gpu: &mut Gpu,
        rows: usize,
        route: MeshRoute,
    ) -> usize {
        gpu.bind_thread().unwrap();
        let layout = forward.mesh_layout(bundle, gpu, route, rows).unwrap();
        let end = bundle.state.position + rows;
        forward
            .prepare_mesh_capacity(bundle, gpu, end, layout.route_rows)
            .unwrap();
        layout.route_rows
    }

    fn head(logits: &GpuTensor) -> Option<Qwen4MeshHead<'_>> {
        Some(Qwen4MeshHead {
            logits,
            top1: None,
            rows: Qwen4OutputRows::Final,
        })
    }

    #[test]
    #[ignore = "GPU: HIPFIRE_QWEN4_DEV_ARTIFACT=<tiny-dense-dev .hfq> with HIPFIRE_EMULATE_GPUS=2 or =4"]
    fn tp_forward_matches_rank_order_oracle() {
        let parts = hipfire_runtime::config::get()
            .emulate_gpus
            .expect("set HIPFIRE_EMULATE_GPUS");

        // Plain single card, unsharded, on the ordinary forward.
        let mut gpu = Gpu::init().unwrap();
        let mut plain = load(&mut gpu);
        let config = plain.config.clone();
        let vocab = config.vocab_size;
        let prompt: Vec<u32> = (0..PROMPT).map(|i| ((i * 7 + 3) % vocab) as u32).collect();
        let logits = gpu.zeros(&[vocab], DType::F32).unwrap();
        let plain_rows = greedy(&prompt, |tokens, route| {
            match route {
                MeshRoute::Prefill => plain.forward_chunk_final(&mut gpu, tokens, &logits, None),
                _ => plain.forward_token(&mut gpu, tokens[0], &logits, None),
            }
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            gpu.download_f32(&logits).unwrap()
        });

        // The rank shards on one device, FFN partials summed in rank order.
        gpu.ensure_capture_stream().unwrap();
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap();
        let plan = qwen4_tp_plan(&config, &mesh, true, false).unwrap();
        let mut shards: Vec<Qwen4Bundle> = (0..parts)
            .map(|rank| {
                let mut bundle = load(&mut gpu);
                bundle
                    .weights
                    .shard_dense_ffn(&mut gpu, rank, parts)
                    .unwrap();
                bundle
            })
            .collect();
        let heads: Vec<GpuTensor> = (0..parts)
            .map(|_| gpu.zeros(&[vocab], DType::F32).unwrap())
            .collect();
        let sum = gpu
            .zeros(&[MAX_SEQ * config.hidden_size], DType::F32)
            .unwrap();
        let oracle_rows = greedy(&prompt, |tokens, route| {
            let mut forwards = take_forwards(&mut shards);
            let mut route_rows = 0;
            for (bundle, forward) in shards.iter_mut().zip(&mut forwards) {
                route_rows = prepare_rank(bundle, forward, &mut gpu, tokens.len(), route);
            }
            let count = tokens.len() * config.hidden_size;
            let bytes = count * std::mem::size_of::<f32>();
            let acc = sum.sub_offset(0, count);
            let partial = |f: &Qwen4GpuForward| f.scratch.ffn_output.sub_offset(0, count);
            for (ops, reduce) in segments(&plan) {
                for (rank, (bundle, forward)) in shards.iter_mut().zip(&mut forwards).enumerate() {
                    forward
                        .forward_mesh_chunk(
                            bundle,
                            &mut gpu,
                            tokens,
                            route_rows,
                            route,
                            head(&heads[rank]),
                            None,
                            ops.clone(),
                        )
                        .unwrap();
                }
                if !reduce {
                    continue;
                }
                // Every copy and add enqueues on the forwards' stream, in order.
                let copy = |gpu: &Gpu, dst: &GpuTensor, src: &GpuTensor| {
                    let stream = gpu.active_stream.as_ref().unwrap();
                    gpu.hip
                        .memcpy_dtod_async_at(&dst.buf, 0, &src.buf, 0, bytes, stream)
                        .unwrap();
                };
                copy(&gpu, &acc, &partial(&forwards[0]));
                for forward in &forwards[1..] {
                    gpu.add_f32(&acc, &partial(forward), &acc).unwrap();
                }
                for forward in &forwards {
                    copy(&gpu, &partial(forward), &acc);
                }
            }
            restore_forwards(&mut shards, forwards);
            gpu.hip.device_synchronize().unwrap();
            let rows: Vec<Vec<f32>> = heads.iter().map(|t| gpu.download_f32(t).unwrap()).collect();
            assert!(
                rows.iter().all(|row| row == &rows[0]),
                "oracle ranks disagree"
            );
            rows[0].clone()
        });

        // The same shards, one per emulated device, through the executor.
        let mut gpus = Gpus::init_tp(parts, 1).unwrap();
        for gpu in &mut gpus.devices {
            gpu.ensure_capture_stream().unwrap();
        }
        let tp_mesh = gpus.mesh.clone();
        let admitted = AdmittedMesh {
            mesh: tp_mesh.clone(),
            ranks: gpus
                .devices
                .iter()
                .enumerate()
                .map(|(logical, gpu)| AdmittedRank {
                    logical,
                    device_ordinal: gpu.device_id,
                    pci_bus_id: None,
                    arch: gpu.arch.clone(),
                })
                .collect(),
            edges: Vec::new(),
            family: "qwen4-tp".into(),
            numerical_policy: "f32".into(),
        };
        gpus.provision_mesh_edges(
            &admitted,
            &BandPlacement::uniform(&tp_mesh, 1).unwrap(),
            MAX_SEQ * config.hidden_size * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let tp_plan = qwen4_tp_plan(&config, &tp_mesh, true, false).unwrap();
        let mut ranks: Vec<Qwen4Bundle> = Vec::with_capacity(parts);
        let mut rank_logits: Vec<GpuTensor> = Vec::with_capacity(parts);
        for (rank, gpu) in gpus.devices.iter_mut().enumerate() {
            gpu.bind_thread().unwrap();
            let mut bundle = load(gpu);
            bundle.weights.shard_dense_ffn(gpu, rank, parts).unwrap();
            ranks.push(bundle);
            rank_logits.push(gpu.zeros(&[vocab], DType::F32).unwrap());
        }
        let tp_epoch = gpus.mesh_tp_epoch().unwrap();
        let mut op_id = 0;
        let tp_rows = greedy(&prompt, |tokens, route| {
            let mut forwards = take_forwards(&mut ranks);
            let mut route_rows = 0;
            for (rank, (bundle, forward)) in ranks.iter_mut().zip(&mut forwards).enumerate() {
                let gpu = &mut gpus.devices[rank];
                route_rows = prepare_rank(bundle, forward, gpu, tokens.len(), route);
            }
            op_id += 1;
            let epoch = ForwardEpoch {
                mesh_epoch: tp_mesh.epoch(),
                load_id: 1,
                request_id: 1,
                op_id,
                route,
                start_position: ranks[0].state.position,
                actual_rows: tokens.len(),
                route_rows,
                policy_id: 1,
            };
            let bands = ranks
                .iter_mut()
                .zip(&mut forwards)
                .map(|(bundle, forward)| Qwen4RankBand { bundle, forward })
                .collect();
            let mut bindings = Qwen4TpBindings::new(bands, tp_epoch).unwrap();
            bindings.set_tokens(tokens);
            bindings
                .set_heads(rank_logits.iter().map(head).collect())
                .unwrap();
            let ticket = gpus
                .submit_mesh_epoch(&tp_plan, &epoch, &mut bindings)
                .unwrap();
            drop(bindings);
            gpus.drain_mesh_epoch(ticket).unwrap();
            restore_forwards(&mut ranks, forwards);
            let rows: Vec<Vec<f32>> = gpus
                .devices
                .iter()
                .zip(&rank_logits)
                .map(|(gpu, t)| {
                    gpu.bind_thread().unwrap();
                    gpu.download_f32(t).unwrap()
                })
                .collect();
            assert!(rows.iter().all(|row| row == &rows[0]), "TP ranks disagree");
            rows[0].clone()
        });

        let ids = |rows: &[Vec<f32>]| rows.iter().map(|row| argmax(row)).collect::<Vec<_>>();
        let max_diff = tp_rows
            .iter()
            .flatten()
            .zip(plain_rows.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let same = ids(&tp_rows)
            .iter()
            .zip(ids(&plain_rows))
            .filter(|(a, b)| **a == *b)
            .count();
        eprintln!(
            "TP{parts}: {} collectives/forward; tp tokens {:?}; oracle fnv64 {:016x}, tp fnv64 {:016x}; \
             plain single card: tokens {:?} ({same}/{} equal), max |logit diff| {max_diff:e}, fnv64 {:016x}",
            tp_plan.collectives.len(),
            ids(&tp_rows),
            fnv(&oracle_rows),
            fnv(&tp_rows),
            ids(&plain_rows),
            tp_rows.len(),
            fnv(&plain_rows),
        );
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&tp_rows),
            bits(&oracle_rows),
            "TP{parts} vs rank-order oracle"
        );
        gpus.drain_mesh_edges().unwrap();
    }
}
