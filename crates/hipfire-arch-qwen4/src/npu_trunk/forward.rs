// SPDX-License-Identifier: Apache-2.0
//! Opt-in first-light scheduler. The ordinary forward never enters this module when off.
use super::*;
use crate::npu_trunk::{self as trunk, Family, FamilyMask, Shape, NpuTrunkMode, EpilogueContract};
use rdna_compute::npu_trunk::{NpuBf16Layout, NpuTrunkOutput};
use railgun::npu::trunk::{NpuTrunkRuntime, LayerPanels, PanelProgram, Seq};
use std::{ffi::c_void, io::Write};

pub(super) struct NpuForward {
    mode: NpuTrunkMode,
    layers: Vec<NpuLayer>,
    a: GpuTensor,
    scales: [GpuTensor; 2],
    sums: GpuTensor,
    bank_b: Option<Qwen4GpuForwardScratch>,
    runtime: Option<NpuTrunkRuntime>,
    trace: Option<std::io::BufWriter<std::fs::File>>,
    host_tokens: Vec<u8>,
    host_ple: Vec<u8>,
}

// SAFETY: this private owner exclusively owns the entire railgun HIP Rc graph;
// none of its HIP wrappers/Rc clones escape it. ArchModel moves the owner, never
// shares it, and every execution uses the destination thread's selected Gpu.
// The relay only sees immutable entry points and raw events, not those Rcs.
unsafe impl Send for NpuForward {}
struct NpuLayer { shape: Shape, map: trunk::ColumnMap, w: GpuTensor, scales: GpuTensor }
fn fail(e: impl std::fmt::Display) -> Qwen4GpuForwardError { invalid(format!("npu-trunk: {e}")) }
fn stream(gpu: &Gpu) -> *mut c_void { gpu.active_stream.as_ref().map_or(std::ptr::null_mut(), |s| s.as_raw()) }
fn layout(shape: Shape, rows: usize) -> NpuBf16Layout {
    NpuBf16Layout { padded_rows: rows.div_ceil(256) * 256, k: 2560, panels: shape.column_map(FamilyMask::ALL).panel_count() }
}
fn transport_shape(shape: Shape) -> railgun::npu::trunk::Shape {
    match shape { Shape::GdnIn => railgun::npu::trunk::Shape::GdnIn, Shape::QsaIn => railgun::npu::trunk::Shape::QsaIn }
}

impl NpuForward {
    pub(super) fn load(gpu: &mut Gpu, bundle: &Qwen4Bundle, rows: usize, scratch_mode: Qwen4ForwardScratchMode) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let mode = NpuTrunkMode::from_env().map_err(fail)?;
        if mode.is_off() { return Ok(None); }
        if mode == NpuTrunkMode::Auto { return Err(fail("auto requires calibrated admission; first light accepts on or emulate")); }
        if gpu.arch != "gfx1151" || !bundle.config.profile.is_flash_next() || bundle.config.hidden_size != 2560 {
            return Err(fail("requires Flash-Next family B, K=2560, gfx1151"));
        }
        if mode == NpuTrunkMode::On && (rows != 8192 || scratch_mode != Qwen4ForwardScratchMode::Full || bundle.weights.ep_placement().is_some()) {
            return Err(fail("on requires two full 8192-row banks, full scratch and single-device placement"));
        }
        let started = Instant::now();
        // Qwen4Weights retains resident handles, not source bytes; no host/source
        // owner is retained by the bundle. One load-only DtoH copy per matrix.
        let mut raw = Vec::new();
        for l in &bundle.weights.layer_refs {
            let (shape, refs) = if let Some(g) = &l.gdn {
                (Shape::GdnIn, vec![&g.qkv, &g.z, &g.in_proj_a, &g.in_proj_b])
            } else if let Some(q) = &l.attention {
                (Shape::QsaIn, vec![&q.q, &q.k, &q.v, &q.indexer_qk])
            } else { return Err(fail("layer has no trunk projections")); };
            let mut mats = Vec::new();
            for reference in refs {
                let w = bundle.weights.resident(reference)?;
                if w.dtype != DType::MQ6G256V2 { return Err(fail(format!("{} must be MQ6G256V2, got {:?}", reference.name, w.dtype))); }
                let m = reference.shape[0];
                let mut bytes = vec![0; m * 2560 / 256 * 200];
                gpu.hip.memcpy_dtoh(&mut bytes, &w.buf)?;
                mats.push((m, bytes));
            }
            raw.push((shape, mats));
        }
        let download_ms=started.elapsed().as_secs_f64()*1000.0;
        let refs: Vec<_> = raw.iter().map(|(shape, mats)| trunk::LayerTrunkRefs {
            shape: *shape, matrices: shape.column_map(FamilyMask::ALL).parts.iter().zip(mats).map(|(&(family, _, _), (m, bytes))| trunk::MatrixRef { family, mq6: bytes, m: *m, k: 2560 }).collect()
        }).collect();
        let packed = trunk::pack_trunk_with_contract(EpilogueContract::Bf16Direct, &refs, FamilyMask::ALL).map_err(fail)?;
        let mut allocations = Vec::new();
        let result = (|| {
            let mut layers = Vec::new();
            let mut resident = Vec::new();
            let mut resident_bytes = 0usize;
            for (index, entry) in packed.per_layer.into_iter().enumerate() {
                let (shape, b) = entry.ok_or_else(|| fail("missing packed layer"))?;
                let mut codes = vec![0u8; b.map.padded_cols() * 2560];
                for (&(_, col, _), (m, bytes)) in b.map.parts.iter().zip(&raw[index].1) {
                    trunk::pack_matrix_direct(bytes,*m,2560,&mut codes[col as usize*2560..(col as usize+m)*2560]);
                }
                let sidecars: Vec<_> = b.scales.iter().flat_map(|(m,e)| m.to_le_bytes().into_iter().chain(e.to_le_bytes())).collect();
                let w = gpu.upload_raw(&codes, &[codes.len()])?; allocations.push(w.sub_offset(0, w.numel()));
                let scales = gpu.upload_raw(&sidecars, &[sidecars.len()])?; allocations.push(scales.sub_offset(0, scales.numel()));
                resident_bytes += codes.len() + sidecars.len()
                    + if mode==NpuTrunkMode::On {b.panels.iter().map(Vec::len).sum::<usize>()}else{0};
                resident.push(LayerPanels { layer: index, shape: transport_shape(shape), panels: b.panels });
                layers.push(NpuLayer { shape, map: b.map, w, scales });
            }
            let a = gpu.zeros(&[layout(Shape::GdnIn, rows).a_bytes()], DType::Raw)?; allocations.push(a.sub_offset(0, a.numel()));
            let d0 = gpu.zeros(&[rows * 4], DType::Raw)?; allocations.push(d0.sub_offset(0, d0.numel()));
            let d1 = gpu.zeros(&[rows * 4], DType::Raw)?; allocations.push(d1.sub_offset(0, d1.numel()));
            let sums = gpu.zeros(&[rows * 16640 * 4], DType::Raw)?; allocations.push(sums.sub_offset(0, sums.numel()));
            let trace = hipfire_config::developer_var("HIPFIRE_QWEN4_NPU_TRUNK_TRACE").ok().filter(|s| !s.is_empty()).map(|path| {
                std::fs::OpenOptions::new().create(true).append(true).open(path).map(std::io::BufWriter::new).map_err(fail)
            }).transpose()?;
            let pack_ms=started.elapsed().as_secs_f64()*1000.0-download_ms;
            let runtime = if mode == NpuTrunkMode::On {
                let design = pm_npu::kernels::gemm_g80::build_panel_program(8192, pm_npu::kernels::gemm_core::Epilogue::Bf16).map_err(fail)?;
                let pci = gpu.hip.device_pci_bus_id(gpu.device_id)?;
                Some(NpuTrunkRuntime::open(PanelProgram { pdi: design.pdi, txn: design.insts, padded_rows: 8192 }, resident, 2, 8192, &pci).map_err(fail)?)
            } else { None };
            let (bank_b,host_tokens,host_ple) = if mode == NpuTrunkMode::On {
                let (scratch,ids,ple)=Qwen4GpuForwardScratch::new(gpu,&bundle.config,rows,scratch_mode)?;
                (Some(scratch),ids,ple)
            } else { (None,Vec::new(),Vec::new()) };
            eprintln!("npu-trunk: mode={mode} families=all contract=bf16_direct pack_ms={pack_ms:.3} download_ms={download_ms:.3} resident_bytes={resident_bytes} runtime={} raw=DtoH(no-retained-host-copy) calibration=first-light-force-full-pairs", if runtime.is_some() { "open" } else { "not-opened" });
            Ok(Some(Self { mode, layers, a, scales: [d0,d1], sums, bank_b, runtime, trace,host_tokens,host_ple }))
        })();
        if result.is_err() { for tensor in allocations { let _ = gpu.free_tensor(tensor); } }
        result
    }
    pub(super) fn pipelined(&self) -> bool { self.mode == NpuTrunkMode::On }
    pub(super) fn free(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let mut first = None;
        if let Some(rt) = self.runtime { if let Err(e) = rt.close() { first = Some(hip_bridge::HipError::new(1, &e)); } }
        if let Some(b) = self.bank_b { if let Some(e) = b.free_gpu(gpu) { first.get_or_insert(e); } }
        for t in self.layers.into_iter().flat_map(|l| [l.w,l.scales]).chain([self.a,self.sums]).chain(self.scales) {
            if let Err(e) = gpu.free_tensor(t) { first.get_or_insert(e); }
        }
        first.map_or(Ok(()), Err)
    }
    fn qkv_view(&self, gpu: &Gpu, step: &Step<'_>) -> Result<GpuTensor, Qwen4GpuForwardError> {
        match step {
            Step::GatedDeltaNet(op) => {
                let mut qkv = op.projection.sub_offset(0, op.rows * 10240);
                let dims = rdna_compute::tensor_ops::GatedDeltaStepBatched {
                    projection: op.projection2, gate: op.gate, beta: op.beta,
                    state: op.recurrent, output: op.recurrent_output, row_states: None,
                    rows: op.rows, qkv_width: 10240, key_heads: op.key_heads,
                    value_heads: op.value_heads, key_dim: op.key_dim,
                    value_dim: op.value_dim, position: op.start_position,
                };
                // Match the core's native projection dtype, including chunk-off.
                if op.key_dim==128 && op.value_dim==128 && op.conv_kernel==4
                    && rdna_compute::tensor_ops::gated_delta_chunk_route(gpu,&dims)
                    && gpu.gemm_mq6g256v2_xf16_applies(2560,op.rows)
                { qkv.dtype=DType::BF16; }
                Ok(qkv)
            }
            Step::IndexedAttention(op) => Ok(op.qgate_scratch.sub_offset(0, op.rows * 12288)),
            _ => Err(fail("expected mixer step")),
        }
    }
    fn destinations<'a>(&self, step: &'a Step<'_>, qkv: &'a GpuTensor) -> Result<SmallVec<[NpuTrunkOutput<'a>; 4]>, Qwen4GpuForwardError> {
        let (shape, targets): (Shape, [(Family, &GpuTensor); 4]) = match step {
            Step::GatedDeltaNet(op) => (Shape::GdnIn, [(Family::GdnQkv,qkv),(Family::GdnZ,op.z_output),(Family::GdnA,op.a),(Family::GdnB,op.b)]),
            Step::IndexedAttention(op) => (Shape::QsaIn, [(Family::QsaQ,op.qgate_scratch),(Family::QsaK,op.k_scratch),(Family::QsaV,op.v_scratch),(Family::QsaIdx,op.index_scratch)]),
            _ => return Err(fail("expected mixer")),
        };
        let map = shape.column_map(FamilyMask::ALL);
        targets.into_iter().map(|(f,t)| map.gpu_output(f,t).map_err(Into::into)).collect()
    }
    fn pack(&self, gpu: &mut Gpu, step: &Step<'_>, bank: usize, pipe: bool) -> Result<u64, Qwen4GpuForwardError> {
        let (input, rows, shape) = match step { Step::GatedDeltaNet(op) => (op.input,op.rows,Shape::GdnIn), Step::IndexedAttention(op) => (op.input,op.rows,Shape::QsaIn), _ => return Err(fail("pack expected mixer")) };
        let x = gpu.rotate_x_mq_batched_f16(input,2560,rows)?;
        let pack_issue_ns=if pipe && self.trace.is_some() {monotonic_ns()}else{0};
        if pipe {
            let rt = self.runtime.as_ref().ok_or_else(|| fail("runtime missing"))?;
            unsafe { gpu.npu_trunk_pack_a_bf16_ptr(&x,rows,layout(shape,8192),rt.a_arena(bank).map_err(fail)?.device as *mut c_void,&self.scales[bank])?; }
        } else { gpu.npu_trunk_pack_a_bf16(&x,rows,layout(shape,rows),&self.a,&self.scales[bank])?; }
        Ok(pack_issue_ns)
    }
    fn emulate(&self, gpu: &mut Gpu, layer: usize, step: &Step<'_>) -> Result<(), Qwen4GpuForwardError> {
        self.pack(gpu, step, 0, false)?;
        let qkv = self.qkv_view(gpu,step)?;
        let shape = self.layers[layer].shape;
        let rows = match step { Step::GatedDeltaNet(op)=>op.rows, Step::IndexedAttention(op)=>op.rows, _=>unreachable!() };
        let outs = self.destinations(step,&qkv)?;
        let l = &self.layers[layer];
        gpu.npu_trunk_emulate_bf16(&self.a,&l.w,&l.scales,&self.scales[0],&self.sums,layout(shape,rows),rows,l.map.padded_cols(),&outs)?;
        Ok(())
    }
}

fn execute(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), Qwen4GpuForwardError> {
    execute_validated_steps(gpu,ctx,steps).map_err(fail)
}
fn phase(step: &mut Step<'_>, value: MixerPhase) {
    match step { Step::GatedDeltaNet(op) => op.phase=value, Step::IndexedAttention(op) => op.phase=value, _=>unreachable!() }
}
impl NpuForward {
    pub(super) fn execute_emulate(&self, gpu: &mut Gpu, ctx: &DispatchCtx, steps: &mut [Step<'_>], mut layer: usize) -> Result<(), Qwen4GpuForwardError> {
        let mut start = 0;
        for index in 0..steps.len() {
            if !matches!(steps[index],Step::GatedDeltaNet(_) | Step::IndexedAttention(_)) { continue; }
            execute(gpu,ctx,&steps[start..index])?;
            self.emulate(gpu,layer,&steps[index])?;
            phase(&mut steps[index],MixerPhase::Core);
            execute(gpu,ctx,&steps[index..=index])?;
            phase(&mut steps[index],MixerPhase::OutProj);
            // Use the same layer boundaries as the pipelined route, preserving
            // output/HC/MoE fusions but never fusing across a bank boundary.
            let next=(index+1..steps.len()).find(|&j|matches!(steps[j],Step::GatedDeltaNet(_) | Step::IndexedAttention(_)));
            let end=next.map_or(steps.len(),|j| {
                let read=j-1;
                if read>0 && matches!(steps[read-1],Step::GroupedDepthwise(_)) {read-1}else{read}
            });
            execute(gpu,ctx,&steps[index..end])?;
            phase(&mut steps[index],MixerPhase::Whole);
            start=end;
            layer+=1;
        }
        execute(gpu,ctx,&steps[start..])
    }
}

#[path = "layers.rs"]
mod layers;
use railgun::npu::trunk::monotonic_ns;

struct TraceRow {
    layer: usize, bank: usize, position: usize, seq: u32,
    pack_issue_ns: u64, kick_ns: u64, submit_ns: u64, done_ns: u64,
    wait_enqueue_ns: u64, consume_issue_ns: u64,
    start: hip_bridge::Event, end: hip_bridge::Event,
}
fn mixer_index(program: &layers::BankProgram<'_>, layer: usize) -> usize {
    program.ranges[layer].clone().find(|&i| matches!(program.steps[i],Step::GatedDeltaNet(_) | Step::IndexedAttention(_))).expect("typed layer has a mixer")
}
fn commit_qsa(bundle: &mut Qwen4Bundle, commits: &[(usize,usize,usize,usize,usize)]) {
    for (slot,&(full,raw,pooled,selected,position)) in commits.iter().enumerate() {
        let state=bundle.state.qsa_mut(slot).expect("validated QSA slot");
        state.full_len=full; state.raw_len=raw; state.pooled_len=pooled;
        state.selected_len=selected; state.position=position; state.clamp_f16_valid();
    }
}

impl NpuForward {
    fn prepare(&self, gpu: &mut Gpu, ctx: &DispatchCtx, program: &layers::BankProgram<'_>, layer: usize, bank: usize, records: &mut [TraceRow]) -> Result<Seq,Qwen4GpuForwardError> {
        let index=mixer_index(program,layer);
        let rt=self.runtime.as_ref().expect("on runtime");
        unsafe { rt.prepare_bank(bank,stream(gpu)).map_err(fail)?; }
        // Includes PLE (when present) and the attention HC read. Neither reads
        // recurrent/QSA state; bank B's input can be prepared ahead of bank A.
        execute(gpu,ctx,&program.steps[program.ranges[layer].start..index])?;
        let pack_issue_ns=self.pack(gpu,&program.steps[index],bank,true)?;
        let kick_ns=if records.is_empty(){0}else{monotonic_ns()};
        let seq=unsafe { rt.kick(layer,transport_shape(self.layers[layer].shape),bank,8192,stream(gpu)).map_err(fail)? };
        if !records.is_empty() {
            let r=&mut records[layer*2+bank];
            r.pack_issue_ns=pack_issue_ns; r.kick_ns=kick_ns; r.seq=seq.value;
        }
        Ok(seq)
    }
    fn run_bank(&self, gpu: &mut Gpu, ctx: &DispatchCtx, program: &mut layers::BankProgram<'_>, layer: usize, bank: usize, seq: Seq, records: &mut [TraceRow]) -> Result<(),Qwen4GpuForwardError> {
        let index=mixer_index(program,layer);
        let rt=self.runtime.as_ref().expect("on runtime");
        let completion=unsafe { rt.wait_gpu(seq,stream(gpu)).map_err(fail)? };
        let wait_enqueue_ns=monotonic_ns();
        let qkv=self.qkv_view(gpu,&program.steps[index])?;
        let outs=self.destinations(&program.steps[index],&qkv)?;
        let consume_issue_ns=monotonic_ns();
        unsafe {
            gpu.npu_trunk_consume_bf16_fast_ptr(rt.c_arena(bank).map_err(fail)?.device as *mut c_void,layout(self.layers[layer].shape,8192),&outs,8192)?;
            // C's last consumer is the scatter, not the subsequent mixer.
            rt.retire_bank(seq,stream(gpu)).map_err(fail)?;
        }
        drop(outs);
        if !records.is_empty() {
            let r=&mut records[layer*2+bank];
            r.submit_ns=completion.submit_ns; r.done_ns=completion.done_ns;
            r.wait_enqueue_ns=wait_enqueue_ns; r.consume_issue_ns=consume_issue_ns;
            gpu.hip.event_record(&r.start,gpu.active_stream.as_ref())?;
        }
        phase(&mut program.steps[index],MixerPhase::Core);
        execute(gpu,ctx,&program.steps[index..=index])?;
        phase(&mut program.steps[index],MixerPhase::OutProj);
        execute(gpu,ctx,&program.steps[index..program.ranges[layer].end])?;
        if !records.is_empty() { gpu.hip.event_record(&records[layer*2+bank].end,gpu.active_stream.as_ref())?; }
        Ok(())
    }
    pub(super) fn pair(&mut self, owner: &mut Qwen4GpuForward, bundle: &mut Qwen4Bundle, gpu: &mut Gpu, tokens: &[u32], logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
        if tokens.len()!=16384 { return Err(fail("pair requires exactly 16384 rows")); }
        let old_scope=std::mem::replace(&mut gpu.qwen4_scope,true);
        let invariant=owner.prefill_scope && rdna_compute::prefill_invariant::prefill_pm_invariant_enabled();
        let old_invariant=std::mem::replace(&mut gpu.prefill_invariant_scope,invariant);
        let result=self.pair_scoped(owner,bundle,gpu,tokens,logits,top1,capture,output);
        gpu.qwen4_scope=old_scope; gpu.prefill_invariant_scope=old_invariant;
        if result.is_err() { bundle.state.invalidate_f16_arenas(); }
        result
    }
    fn pair_scoped(&mut self, owner: &mut Qwen4GpuForward, bundle: &mut Qwen4Bundle, gpu: &mut Gpu, tokens: &[u32], logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
        owner.restore_expert_stage(gpu)?;
        let position=bundle.state.position;
        let end=position.checked_add(16384).ok_or_else(||fail("position overflow"))?;
        bundle.release_session_for_growth(gpu,end);
        bundle.state.ensure_mapped_capacity(gpu,end).map_err(fail)?;
        if let Some(qsa)=bundle.state.qsa.first().filter(|q|q.f16_arena.is_none()) {
            rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(gpu,qsa.format,bundle.config.num_key_value_heads,end)?;
        }
        // No tape can contain relay/host operations. Decode retains its old route.
        gpu.replay.set_forward_eligible(false);
        owner.uploaded_id_rows=0;
        drop(owner.ple_ahead.take());
        let b=self.bank_b.as_ref().expect("on bank B");
        let ra=matrix_view(&owner.scratch.router_logits,8192,router_columns(program_dims(&bundle.config)))?;
        let rb=matrix_view(&b.router_logits,8192,router_columns(program_dims(&bundle.config)))?;
        let ctx=DispatchCtx::new(gpu);
        let sa=layer_scratch(&owner.scratch,&ra);
        let sb=layer_scratch(b,&rb);
        let mut programs=[
            layers::build_program(owner,&owner.scratch,bundle,gpu,&ctx,&sa,0)?,
            layers::build_program(owner,b,bundle,gpu,&ctx,&sb,1)?,
        ];
        let mut history=bundle.state.ple_history;
        for bank in 0..2 {
            let scratch=if bank==0 { &owner.scratch } else { b };
            initialize(gpu,bundle,scratch,&tokens[bank*8192..(bank+1)*8192],&history,programs[bank].ple_rows_f16,&mut self.host_tokens,&mut self.host_ple)?;
            for &token in &tokens[bank*8192..(bank+1)*8192] { history.push(token); }
        }
        let commits: SmallVec<[_; QWEN4_QSA_INLINE_CAPACITY]>=programs[1].steps.iter().filter_map(|step|match step {
            Step::IndexedAttention(op)=>Some(op.next_lengths().map_err(fail)), _=>None,
        }).collect::<Result<_,_>>()?;
        let anchor=if self.trace.is_some() { Some(gpu.hip.event_create()?) } else { None };
        let mut anchor_ns=0;
        let mut anchor_uncertainty_ns=0;
        let mut records=Vec::new();
        if let Some(anchor)=&anchor {
            // Drain initialization before calibrating the HIP clock. Both waits
            // are before the first pipeline pack, never between its layers.
            gpu.hip.event_record(anchor,gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(anchor)?;
            let before=monotonic_ns();
            gpu.hip.event_record(anchor,gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(anchor)?;
            let after=monotonic_ns();
            anchor_ns=before+(after-before)/2; anchor_uncertainty_ns=after-before;
            for layer in 0..bundle.config.num_hidden_layers { for bank in 0..2 {
                records.push(TraceRow { layer,bank,position:position+bank*8192,seq:0,pack_issue_ns:0,kick_ns:0,submit_ns:0,done_ns:0,wait_enqueue_ns:0,consume_issue_ns:0,start:gpu.hip.event_create()?,end:gpu.hip.event_create()? });
            }}
        }
        let attempt: Result<(),Qwen4GpuForwardError>=(|| {
            let mut seq=[
                self.prepare(gpu,&ctx,&programs[0],0,0,&mut records)?,
                self.prepare(gpu,&ctx,&programs[1],0,1,&mut records)?,
            ];
            for (layer,bank) in bank_visits(bundle.config.num_hidden_layers) {
                self.run_bank(gpu,&ctx,&mut programs[bank],layer,bank,seq[bank],&mut records)?;
                if layer+1<bundle.config.num_hidden_layers {
                    seq[bank]=self.prepare(gpu,&ctx,&programs[bank],layer+1,bank,&mut records)?;
                }
            }
            // Both trunks finish before prompt-end head/capture work.
            finish_bank(gpu,bundle,&owner.scratch,logits,top1,capture,0,output)?;
            finish_bank(gpu,bundle,b,logits,top1,capture,1,output)?;
            Ok(())
        })();
        drop(programs);
        let attempt = attempt.and_then(|_| {
            if let (Some(out),Some(anchor))=(&mut self.trace,&anchor) {
                if let Some(last)=records.last() { gpu.hip.event_synchronize(&last.end)?; }
                for r in &records {
                    let start_ns=anchor_ns+(gpu.hip.event_elapsed_ms(anchor,&r.start)? as f64*1e6) as u64;
                    let end_ns=anchor_ns+(gpu.hip.event_elapsed_ms(anchor,&r.end)? as f64*1e6) as u64;
                    writeln!(out,"{{\"layer\":{},\"bank\":\"{}\",\"position\":{},\"rows\":8192,\"seq\":{},\"pack_issue_ns\":{},\"kick_ns\":{},\"npu_submit_ns\":{},\"npu_done_ns\":{},\"gpu_wait_enqueue_ns\":{},\"consume_issue_ns\":{},\"mixer_start_ns\":{},\"mixer_end_ns\":{},\"anchor_uncertainty_ns\":{}}}",r.layer,if r.bank==0{"A"}else{"B"},r.position,r.seq,r.pack_issue_ns,r.kick_ns,r.submit_ns,r.done_ns,r.wait_enqueue_ns,r.consume_issue_ns,start_ns,end_ns,anchor_uncertainty_ns).map_err(fail)?;
                }
                out.flush().map_err(fail)?;
            }
            Ok(())
        });
        for r in records { let _=gpu.hip.event_destroy(r.start); let _=gpu.hip.event_destroy(r.end); }
        if let Some(anchor)=anchor { let _=gpu.hip.event_destroy(anchor); }
        attempt?;
        commit_qsa(bundle,&commits);
        bundle.state.ple_history=history;
        bundle.state.commit_row_capture(16384);
        bundle.state.position=end;
        Ok(())
    }
}

fn initialize(gpu: &mut Gpu, bundle: &Qwen4Bundle, scratch: &Qwen4GpuForwardScratch, tokens: &[u32], history: &crate::ple::PleHistory, ple_f16: bool, ids: &mut [u8], ple_bytes: &mut [u8]) -> Result<(),Qwen4GpuForwardError> {
    let n=tokens.len(); let config=&bundle.config;
    for (dst,&token) in ids[..n*4].chunks_exact_mut(4).zip(tokens) { dst.copy_from_slice(&(token as i32).to_ne_bytes()); }
    gpu.memcpy_htod_auto(&scratch.token_ids.buf,&ids[..n*4])?;
    let rotated=view(&scratch.embedding_rot,0,n*config.hidden_size);
    let embeddings=view(&scratch.embeddings,0,n*config.hidden_size);
    execute_embedding(gpu,&EmbeddingOp { table:bundle.weights.resident(&bundle.weights.root.embedding)?, rotated:&rotated, token_ids:&scratch.token_ids, output:&embeddings, rows:n,dim:config.hidden_size }).map_err(fail)?;
    if gpu.qwen4_bf16_streams(n) {
        gpu.hc_streams_init_from_embed_batched_bf16(&embeddings,&scratch.streams,config.hidden_size as i32,config.hc_count as i32,n as i32)?;
    } else {
        gpu.hc_streams_init_from_embed_batched(&embeddings,&scratch.streams,config.hidden_size as i32,config.hc_count as i32,n as i32)?;
    }
    if let Some(ple)=&bundle.ple {
        let mut fetch=RowFetch::begin_or_adopt(&ple.rows,None,history.row_ids(&ple.metadata,tokens)).map_err(fail)?;
        let lease=fetch.wait().map_err(fail)?;
        let bytes=&mut ple_bytes[..lease.as_bytes().map_err(fail)?.len()];
        lease.stage_into(bytes).map_err(fail)?;
        gpu.memcpy_htod_auto(&scratch.ple_staged.buf,bytes)?;
        lease.validate_after_upload().map_err(fail)?;
        let geometry=config.ple.as_ref().expect("PLE owner has config");
        if ple_f16 { gpu.grouped_gather_convert_bf16_f16(&scratch.ple_staged,&scratch.ple_rows,n,geometry.head_count,geometry.row_width)?; }
        else { gpu.grouped_gather_convert_bf16(&scratch.ple_staged,&scratch.ple_rows,n,geometry.head_count,geometry.row_width)?; }
        fetch.complete();
    }
    Ok(())
}

fn finish_bank(gpu: &mut Gpu, bundle: &Qwen4Bundle, scratch: &Qwen4GpuForwardScratch, logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, bank: usize, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
    let config=&bundle.config; let dims=program_dims(config); let n=8192;
    let bf16=gpu.qwen4_bf16_streams(n);
    if let Some(capture)=capture {
        let dst=f32_view(capture,bank*n*dims.wide(),n*dims.wide());
        let src=view(&scratch.streams,0,n*dims.wide());
        if bf16 { rdna_compute::tensor_ops::hc_state_bf16_to_f32(gpu,&src,&dst,n*dims.wide())?; }
        else { gpu.copy_d2d(&src,&dst,src.byte_size())?; }
    }
    let rows=match output {
        Qwen4OutputPolicy::Rows(Qwen4OutputRows::All)=>n,
        Qwen4OutputPolicy::Rows(Qwen4OutputRows::Final) if bank==1=>1,
        _=>return Ok(()),
    };
    let scratch_desc=layer_scratch(scratch,&scratch.router_logits);
    let hyper=hyper_read_desc(&bundle.weights,&bundle.weights.root.final_hyper)?;
    let lm=dense_ref(&bundle.weights,&bundle.weights.root.lm_head)?;
    execute_final_hyper(gpu,dims,&hyper,&scratch.streams,&scratch_desc,n,bf16).map_err(fail)?;
    let hidden=view(&scratch.hc_mixed,0,n*config.hidden_size);
    let offset=if rows==n {bank*n*config.vocab_size}else{0};
    let logits=logits.ok_or_else(||fail("head requires logits"))?.sub_offset(offset,rows*config.vocab_size);
    execute_lm_head(gpu,&lm,&hidden,&logits,n,rows,Some(&scratch.rotation)).map_err(fail)?;
    if let Some(top1)=top1 {
        let indices=top1.sub_offset(if rows==n{bank*n*4}else{0},rows*4);
        argmax_f32(gpu,&ArgmaxF32 { logits:&logits,indices:&indices,rows,vocab:config.vocab_size })?;
    }
    Ok(())
}

/// Only complete consecutive 8192-row chunks are paired; the tail keeps the
/// same boundaries as the GPU numerical twin.
pub(super) fn pair_rows(remaining: usize, chunk: usize) -> Option<usize> {
    (chunk==8192 && remaining>=16384).then_some(16384)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn pairs_cover_exact_ranges_and_keep_tail() {
        for rows in [0,511,8192,16383,16384,16385,32768,40960] {
            let mut at=0; let mut ranges=Vec::new();
            while at<rows {
                let n=pair_rows(rows-at,8192).unwrap_or((rows-at).min(8192));
                ranges.push(at..at+n); at+=n;
            }
            assert_eq!(at,rows);
            for pair in ranges.windows(2) { assert_eq!(pair[0].end,pair[1].start); }
            for r in ranges { if r.len()==16384 { assert_eq!((r.start..r.start+8192).end,(r.start+8192..r.end).start); } }
        }
        assert_eq!(pair_rows(32768,4096),None);
    }
    #[test] fn arena_retirement_precedes_next_pack_and_state_order() {
        let mut operations=vec![(0,0,"pack"),(0,1,"pack")];
        for (layer,bank) in bank_visits(3) {
            operations.extend([(layer,bank,"consume"),(layer,bank,"retire"),(layer,bank,"core")]);
            if layer+1<3 { operations.push((layer+1,bank,"pack")); }
        }
        for layer in 0..3 {
            let index=|bank,op|operations.iter().position(|x|*x==(layer,bank,op)).unwrap();
            assert!(index(0,"core")<index(1,"core"));
            for bank in 0..2 { assert!(index(bank,"consume")<index(bank,"retire")); }
            if layer+1<3 {
                for bank in 0..2 {
                    let next=operations.iter().position(|x|*x==(layer+1,bank,"pack")).unwrap();
                    assert!(index(bank,"retire")<next);
                    if bank==0 { assert!(next<index(1,"core")); }
                }
            }
        }
    }
}

pub(super) fn run_pair_request(owner: &mut Qwen4GpuForward, bundle: &mut Qwen4Bundle, gpu: &mut Gpu, tokens: &[u32], offset: usize, logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
    let all=matches!(output,Qwen4OutputPolicy::Rows(Qwen4OutputRows::All));
    let logits=logits.map(|t|if all { t.sub_offset(offset*bundle.config.vocab_size,16384*bundle.config.vocab_size) } else { t.sub_offset(0,t.numel()) });
    let top1=top1.map(|t|if all {t.sub_offset(offset*4,16384*4)}else{t.sub_offset(0,t.numel())});
    let capture=capture.map(|t|t.sub_offset(offset*program_dims(&bundle.config).wide(),16384*program_dims(&bundle.config).wide()));
    let mut npu=owner.npu_trunk.take().expect("on owner");
    let result=npu.pair(owner,bundle,gpu,tokens,logits.as_ref(),top1.as_ref(),capture.as_ref(),output);
    owner.npu_trunk=Some(npu);
    result
}

fn bank_visits(layers: usize) -> impl Iterator<Item=(usize,usize)> {
    (0..layers).flat_map(|layer|[(layer,0),(layer,1)])
}
