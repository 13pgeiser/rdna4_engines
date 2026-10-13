// SPDX-License-Identifier: Apache-2.0
//! Calibrated Halo scheduler. Effective off performs no new I/O or allocation.
use super::*;
use crate::npu_trunk::{self as trunk, Family, FamilyMask, Shape, NpuTrunkMode, EpilogueContract};
use rdna_compute::npu_trunk::{NpuBf16Layout, NpuTrunkOutput};
use railgun::npu::trunk::{NpuTrunkRuntime, LayerPanels, PanelProgram, Seq};
use rdna_compute::tensor_ops::Bf16Arena;
use hipfire_dispatch::pipeline::{
    execute_gated_delta_net_arena, execute_indexed_attention_arena, GdnTrunkArenas, QsaTrunkArenas,
    execute_gated_delta_net_uncovered, execute_indexed_attention_uncovered, gdn_qkv_projection_bf16,
};
use std::{ffi::c_void, io::Write};

pub(super) struct NpuForward {
    mode: NpuTrunkMode,
    families: FamilyMask,
    layers: Vec<NpuLayer>,
    emulation: Option<EmulationScratch>,
    // The pack producer writes compact activation sidecars in both modes.
    scales: [GpuTensor; 2],
    bank_b: Option<Qwen4GpuForwardScratch>,
    runtime: Option<NpuTrunkRuntime>,
    arenas: Option<[[Option<GpuTensor>; 2]; 2]>,
    trace: Option<std::io::BufWriter<std::fs::File>>,
    host_tokens: Vec<u8>,
    host_ple: Vec<u8>,
    pair_logged: bool,
    calibration: Option<trunk::Calibration>,
    n_star: Option<usize>,
    request_plan: Option<trunk::Plan>,
    request_origin: usize,
}

// SAFETY: this private owner exclusively owns the entire railgun HIP Rc graph;
// none of its HIP wrappers/Rc clones escape it. ArchModel moves the owner, never
// shares it, and every execution uses the destination thread's selected Gpu.
// The relay only sees immutable entry points and raw events, not those Rcs.
unsafe impl Send for NpuForward {}
struct NpuLayer { shape: Shape, map: trunk::ColumnMap, twin: Option<GpuTwin> }
struct GpuTwin { w: GpuTensor, scales: GpuTensor }
struct EmulationScratch { a: GpuTensor, sums: GpuTensor }
fn fail(e: impl std::fmt::Display) -> Qwen4GpuForwardError { invalid(format!("npu-trunk: {e}")) }
fn stream(gpu: &Gpu) -> *mut c_void { gpu.active_stream.as_ref().map_or(std::ptr::null_mut(), |s| s.as_raw()) }
fn layout(map: &trunk::ColumnMap, rows: usize) -> NpuBf16Layout {
    NpuBf16Layout { padded_rows: rows.div_ceil(256) * 256, k: 2560, panels: map.panel_count() }
}
fn family_view<'a>(map: &trunk::ColumnMap, tensor: &'a GpuTensor, family: Family) -> Option<Bf16Arena<'a>> {
    map.parts.iter().find(|&&(f, _, _)| f == family).map(|&(_, col, _)| Bf16Arena {
        tensor, row_pitch_elems: map.padded_cols(), col0_elems: col as usize,
    })
}
fn transport_shape(shape: Shape) -> railgun::npu::trunk::Shape {
    match shape { Shape::GdnIn => railgun::npu::trunk::Shape::GdnIn, Shape::QsaIn => railgun::npu::trunk::Shape::QsaIn }
}

fn registered_c_view(rt: &NpuTrunkRuntime, bank: usize, shape: Shape, families: FamilyMask) -> Result<Option<GpuTensor>, Qwen4GpuForwardError> {
    let map = shape.column_map(families);
    if map.panel_count() == 0 { return Ok(None); }
    let arena = rt.c_arena(bank, transport_shape(shape)).map_err(fail)?;
    // Borrowed metadata only: runtime owns the registration/pages until all
    // bank retirement events complete. Never free this as a GPU allocation.
    Ok(Some(GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(arena.device as *mut c_void, arena.len) },
        shape: vec![8192, map.padded_cols()], dtype: DType::BF16,
    }))
}

/// Rebind logical extents only; the dedicated bank's allocation capacity and
/// ownership slots remain unchanged, so a later pair can grow back to 8192.
fn bind_bank_extents(tensors: &mut [&mut GpuTensor], slots: &[ScratchSlot]) -> Result<(), Qwen4GpuForwardError> {
    if tensors.len() != ScratchField::COUNT || slots.len() != tensors.len() {
        return Err(fail("bank scratch field count mismatch"));
    }
    for (tensor, slot) in tensors.iter().zip(slots) {
        if tensor.shape.is_empty() || tensor.dtype != slot.dtype || slot.bytes()? > tensor.buf.size() {
            return Err(fail("bank live view exceeds its dedicated scratch allocation"));
        }
    }
    for (tensor, slot) in tensors.iter_mut().zip(slots) {
        tensor.shape.truncate(1);
        tensor.shape[0] = slot.elements;
    }
    Ok(())
}

fn preserve_native_projection_capacity(live: &mut [ScratchSlot], capacity: &[ScratchSlot]) {
    // The native ragged producer pads only these four dedicated buffers.
    // Eligibility must see the same max_chunk capacity as sequential prefill;
    // mixer/kernel row counts stay the live rows, never the padded capacity.
    for field in [ScratchField::HcMixed, ScratchField::GdnZ, ScratchField::GdnA, ScratchField::GdnB] {
        live[field as usize].elements = capacity[field as usize].elements;
    }
}

fn bind_bank_rows(s: &mut Qwen4GpuForwardScratch, config: &Qwen4Config, rows: usize) -> Result<(), Qwen4GpuForwardError> {
    if s.mode != Qwen4ForwardScratchMode::Full || rows == 0 || rows > s.max_chunk {
        return Err(fail("bank live views require full scratch within its row capacity"));
    }
    let (mut slots, _) = Qwen4GpuForwardScratch::plan(config, rows, Qwen4ForwardScratchMode::Full)?;
    preserve_native_projection_capacity(&mut slots, &s.slots);
    // Other row-checked bindings (including PLE and grouped MoE) use live rows.
    // ScratchField order is identical to the existing allocation/free tables.
    bind_bank_extents(&mut [
        &mut s.token_ids, &mut s.embedding_rot, &mut s.embeddings, &mut s.streams,
        &mut s.hc_normalized, &mut s.hc_low, &mut s.hc_up, &mut s.hc_mixed, &mut s.hc_gates,
        &mut s.rotation, &mut s.projection, &mut s.projection2,
        &mut s.gdn_a, &mut s.gdn_b, &mut s.gdn_gate, &mut s.gdn_beta,
        &mut s.gdn_recurrent_output, &mut s.gdn_bf16, &mut s.gdn_z, &mut s.gdn_output,
        &mut s.qsa_index, &mut s.qsa_qgate, &mut s.qsa_k, &mut s.qsa_v,
        &mut s.qsa_output, &mut s.qsa_selected, &mut s.attention_output,
        &mut s.ple_staged, &mut s.ple_rows, &mut s.ple_key, &mut s.ple_value,
        &mut s.ple_query, &mut s.ple_gated, &mut s.ple_normed, &mut s.ple_output,
        &mut s.router_logits, &mut s.moe_x_rot, &mut s.moe_gate_up,
        &mut s.moe_gate, &mut s.moe_up, &mut s.moe_hidden, &mut s.ffn_output,
        &mut s.moe_shared_output, &mut s.moe_gate_batch, &mut s.moe_up_batch, &mut s.moe_rot_batch,
        &mut s.moe_topk_indices, &mut s.moe_topk_weights, &mut s.moe_down_expanded, &mut s.moe_scalar,
        &mut s.moe_expert_token_counts, &mut s.moe_expert_offsets, &mut s.moe_sorted_slot_index,
        &mut s.moe_expert_tile_ids, &mut s.moe_inverse_perm,
        &mut s.moe_y_gate_up_grouped, &mut s.moe_y_down_grouped, &mut s.logits,
    ], &slots)
}

/// Auto panels are packed before the overlay swap; forced modes retain stock
/// bytes until the successfully applied route determines their calibration.
enum PreparedLayers {
    Stock(Vec<(Shape, Vec<(usize, Vec<u8>)>)>),
    Packed(Vec<(Shape, trunk::PackedB, Vec<u8>)>),
}
pub(crate) struct PreparedTrunk {
    mode: NpuTrunkMode,
    layers: PreparedLayers,
    download_ms: f64,
    pack_ms: f64,
    pre_overlay: bool,
    overlay_route: Option<rdna_compute::TrunkIu4Default>,
    matched: Option<(trunk::Calibration, trunk::CalKey, &'static str)>,
    families: FamilyMask,
}

impl PreparedTrunk {
    pub(crate) fn set_overlay_route(&mut self, route: Option<rdna_compute::TrunkIu4Default>) {
        self.overlay_route = route;
    }
}

fn match_calibration(gpu: &Gpu, bundle: &Qwen4Bundle, route: &str, mode: NpuTrunkMode, firmware: Option<&str>) -> Result<(trunk::Calibration, trunk::CalKey, &'static str, FamilyMask), String> {
    let (cal, source) = trunk::Calibration::load_with_builtin(hipfire_config::npu_trunk_calibration_path(), route)
        .map_err(|e| format!("calibration unavailable: {e}"))?;
    let index = bundle.npu_model_index.as_ref().ok_or("model index identity unavailable")?.as_ref().map_err(Clone::clone)?;
    let pci = gpu.hip.device_pci_bus_id(gpu.device_id).map_err(|e| e.to_string())?;
    let key = match firmware {
        Some(firmware) => trunk::runtime_key_with_firmware(&gpu.arch, &pci, firmware, index, route)?,
        None => trunk::runtime_calibration_key(&gpu.arch, &pci, index, route)?,
    };
    let families = if mode == NpuTrunkMode::Auto { trunk::auto_admission(&cal, &key)? } else {
        cal.key_matches(&key)?;
        trunk::select_families(&cal).0
    };
    Ok((cal, key, source, families))
}

fn pack_stock(mode: NpuTrunkMode, raw: Vec<(Shape, Vec<(usize, Vec<u8>)>)>, families: FamilyMask) -> Result<Vec<(Shape, trunk::PackedB, Vec<u8>)>, Qwen4GpuForwardError> {
    let refs: Vec<_> = raw.iter().map(|(shape, mats)| trunk::LayerTrunkRefs {
        shape: *shape, matrices: shape.column_map(FamilyMask::ALL).parts.iter().zip(mats).map(|(&(family, _, _), (m, bytes))|
            trunk::MatrixRef { family, mq6: bytes, m: *m, k: 2560 }).collect(),
    }).collect();
    let packed_trunk = trunk::pack_trunk_with_contract(EpilogueContract::Bf16Direct, &refs, families).map_err(fail)?;
    drop(refs);
    let mut packed = Vec::with_capacity(raw.len());
    for (entry, (shape, mats)) in packed_trunk.per_layer.into_iter().zip(raw) {
        let b = entry.map(|(_, b)| b).unwrap_or_else(|| trunk::PackedB {
            panels: Vec::new(), ws: Vec::new(), scales: Vec::new(), map: shape.column_map(families),
        });
        let mut codes = if mode == NpuTrunkMode::Emulate { vec![0u8; b.map.padded_cols() * 2560] } else { Vec::new() };
        if mode == NpuTrunkMode::Emulate {
            let full = shape.column_map(FamilyMask::ALL);
            for &(family, col, width) in &b.map.parts {
                let i = full.parts.iter().position(|&(f, _, _)| f == family).expect("shape family");
                let (m, bytes) = &mats[i];
                debug_assert_eq!(*m, width as usize);
                trunk::pack_matrix_direct(bytes, *m, 2560, &mut codes[col as usize * 2560..(col as usize + m) * 2560]);
            }
        }
        packed.push((shape, b, codes));
    }
    Ok(packed)
}

impl NpuForward {
    pub(crate) fn prepare_stock(gpu: &mut Gpu, bundle: &Qwen4Bundle, pre_overlay: bool, rows: usize, scratch_mode: Qwen4ForwardScratchMode) -> Result<Option<PreparedTrunk>, Qwen4GpuForwardError> {
        let mode = NpuTrunkMode::from_env().map_err(fail)?.resolve(&gpu.arch);
        if mode.is_off() { return Ok(None); }
        let result = trunk::admission::prepare_after_checks(
            || {
                if gpu.arch != "gfx1151" || !bundle.config.profile.is_flash_next() || bundle.config.hidden_size != 2560 {
                    return Err(fail("requires Flash-Next family B, K=2560, gfx1151"));
                }
                if mode != NpuTrunkMode::Emulate && (rows != 8192 || scratch_mode != Qwen4ForwardScratchMode::Full || bundle.weights.ep_placement().is_some()) {
                    return Err(fail("requires full 8192-row scratch and single-device/no-EP placement"));
                }
                // Validate every source before any download, packing or NPU allocation.
                for l in &bundle.weights.layer_refs {
                    let (refs, widths) = if let Some(g) = &l.gdn {
                        ([&g.qkv, &g.z, &g.in_proj_a, &g.in_proj_b], [10240, 6144, 48, 48])
                    } else if let Some(q) = &l.attention {
                        ([&q.q, &q.k, &q.v, &q.indexer_qk], [12288, 512, 512, 640])
                    } else { return Err(fail("layer has no trunk projections")); };
                    for (reference, width) in refs.into_iter().zip(widths) {
                        let w = bundle.weights.resident(reference)?;
                        if w.dtype != DType::MQ6G256V2 || reference.shape.as_slice() != [width, 2560] {
                            return Err(fail(format!("{} requires stock MQ6G256V2 [{width},2560]", reference.name)));
                        }
                    }
                }
                if mode == NpuTrunkMode::Auto {
                    // Reject the memory rung downgrade before downloading stock rows.
                    let (free, _) = gpu.device_mem_info()?;
                    let required = qwen4_forward_device_bytes(&bundle.config, rows, scratch_mode)
                        .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES));
                    if required.is_none_or(|bytes| bytes > free as u64) {
                        return Err(fail("8192-row full scratch does not fit the single-device memory rung"));
                    }
                }
                Ok(())
            },
            || if mode == NpuTrunkMode::Auto { trunk::npu_firmware().map(Some).map_err(fail) } else { Ok(None) },
            |firmware| {
                let route = bundle.trunk_overlay.as_ref().map(|overlay| overlay.default).or(bundle.config.trunk_iu4_default);
                if mode != NpuTrunkMode::Auto { return Ok((None, FamilyMask::ALL, route)); }
                let trunk_route = route.map_or_else(|| "stock".to_owned(), |default| format!("overlay:{}", default.marker()));
                let (cal, key, source, families) = match_calibration(gpu, bundle, &trunk_route, mode, firmware.as_deref()).map_err(fail)?;
                Ok((Some((cal, key, source)), families, route))
            },
            |(matched, families, overlay_route)| {
                let started = Instant::now();
                let mut raw = Vec::with_capacity(bundle.weights.layer_refs.len());
                for l in &bundle.weights.layer_refs {
                    let (shape, refs) = if let Some(g) = &l.gdn {
                        (Shape::GdnIn, [&g.qkv, &g.z, &g.in_proj_a, &g.in_proj_b])
                    } else if let Some(q) = &l.attention {
                        (Shape::QsaIn, [&q.q, &q.k, &q.v, &q.indexer_qk])
                    } else { unreachable!("eligibility checked"); };
                    let mut mats = Vec::with_capacity(4);
                    for (reference, &family) in refs.into_iter().zip(shape.families()) {
                        let m = reference.shape[0];
                        let bytes = if mode != NpuTrunkMode::Auto || families.contains(family) {
                            let w = bundle.weights.resident(reference)?;
                            let mut bytes = vec![0; m * 2560 / 256 * 200];
                            gpu.hip.memcpy_dtoh(&mut bytes, &w.buf)?;
                            bytes
                        } else { Vec::new() };
                        mats.push((m, bytes));
                    }
                    raw.push((shape, mats));
                }
                let download_ms = started.elapsed().as_secs_f64() * 1000.0;
                let packing_started = Instant::now();
                let layers = if mode == NpuTrunkMode::Auto {
                    PreparedLayers::Packed(pack_stock(mode, raw, families)?)
                } else { PreparedLayers::Stock(raw) };
                let pack_ms = packing_started.elapsed().as_secs_f64() * 1000.0;
                Ok(Some(PreparedTrunk { mode, layers, download_ms, pack_ms, pre_overlay, overlay_route, matched, families }))
            },
        );
        trunk::admission::finish(mode, result)
    }

    pub(super) fn load(gpu: &mut Gpu, bundle: &Qwen4Bundle, scratch: &Qwen4GpuForwardScratch, rows: usize, scratch_mode: Qwen4ForwardScratchMode, prepared: Option<Option<PreparedTrunk>>) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let mode = NpuTrunkMode::from_env().map_err(fail)?.resolve(&gpu.arch);
        if mode.is_off() { return Ok(None); }
        trunk::admission::finish(mode, Self::load_admitted(gpu, bundle, scratch, rows, scratch_mode, prepared))
    }

    fn load_admitted(gpu: &mut Gpu, bundle: &Qwen4Bundle, scratch: &Qwen4GpuForwardScratch, rows: usize, scratch_mode: Qwen4ForwardScratchMode, prepared: Option<Option<PreparedTrunk>>) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let Some(prepared) = (match prepared {
            Some(prepared) => prepared,
            None => Self::prepare_stock(gpu, bundle, false, rows, scratch_mode)?,
        }) else { return Ok(None); };
        let PreparedTrunk { mode, layers, download_ms, mut pack_ms, pre_overlay, overlay_route, matched, mut families } = prepared;
        let trunk_route = overlay_route.map_or_else(|| "stock".to_owned(), |default| format!("overlay:{}", default.marker()));
        let matched = if mode == NpuTrunkMode::Auto { matched } else {
            match match_calibration(gpu, bundle, &trunk_route, mode, None) {
                Ok((cal, key, source, selected)) => { families = selected; Some((cal, key, source)) }
                Err(reason) => {
                    eprintln!("npu-trunk: mode={mode} calibration=none (forced) reason={reason}");
                    None
                }
            }
        };
        let (calibration, n_star, cal_key, cal_source) = match matched {
            Some((cal, mut key, source)) => {
                key.trunk_route.clone_from(&trunk_route);
                cal.key_matches(&key).map_err(fail)?;
                let n_star = trunk::crossover(&cal, families);
                (Some(cal), n_star, Some(key), source)
            }
            None => (None, None, None, "none"),
        };
        if families.is_empty() { return Ok(None); }
        let physical = mode != NpuTrunkMode::Emulate;
        if physical && (rows != 8192 || scratch_mode != Qwen4ForwardScratchMode::Full || bundle.weights.ep_placement().is_some()) {
            return Err(fail("requires two full 8192-row banks, full scratch and single-device placement"));
        }
        let packing_started = Instant::now();
        let packed = match layers {
            PreparedLayers::Packed(packed) => packed,
            PreparedLayers::Stock(raw) => pack_stock(mode, raw, families)?,
        };
        pack_ms += packing_started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let mut allocations = Vec::new();
        let result = (|| {
            let mut layers = Vec::new();
            let mut resident = Vec::new();
            let mut npu_panels_bytes = 0usize;
            let mut gpu_twin_bytes = 0usize;
            for (index, (shape, b, codes)) in packed.into_iter().enumerate() {
                let twin = if !physical && !b.map.parts.is_empty() {
                    let sidecars: Vec<_> = b.scales.iter().flat_map(|(m,e)| m.to_le_bytes().into_iter().chain(e.to_le_bytes())).collect();
                    let w = gpu.upload_raw(&codes, &[codes.len()])?; allocations.push(w.sub_offset(0, w.numel()));
                    let scales = gpu.upload_raw(&sidecars, &[sidecars.len()])?; allocations.push(scales.sub_offset(0, scales.numel()));
                    gpu_twin_bytes += codes.len() + sidecars.len();
                    Some(GpuTwin { w, scales })
                } else { None };
                if physical && !b.panels.is_empty() {
                    npu_panels_bytes += b.panels.iter().map(Vec::len).sum::<usize>();
                    resident.push(LayerPanels { layer: index, shape: transport_shape(shape), panels: b.panels });
                }
                layers.push(NpuLayer { shape, map: b.map, twin });
            }
            let emulation = if physical { None } else {
                // A geometry depends only on rows and K, not selected columns.
                let a_bytes = NpuBf16Layout { padded_rows: rows.div_ceil(256) * 256, k: 2560, panels: 1 }.a_bytes();
                let a = gpu.zeros(&[a_bytes], DType::Raw)?; allocations.push(a.sub_offset(0, a.numel()));
                let cols = layers.iter().map(|layer| layer.map.padded_cols()).max().unwrap_or(0).max(1);
                let sums = gpu.zeros(&[rows * cols * 4], DType::Raw)?; allocations.push(sums.sub_offset(0, sums.numel()));
                Some(EmulationScratch { a, sums })
            };
            let d0 = gpu.zeros(&[rows * 4], DType::Raw)?; allocations.push(d0.sub_offset(0, d0.numel()));
            let d1 = gpu.zeros(&[rows * 4], DType::Raw)?; allocations.push(d1.sub_offset(0, d1.numel()));
            let resident_bytes = npu_panels_bytes + gpu_twin_bytes;
            let trace = hipfire_config::developer_var("HIPFIRE_QWEN4_NPU_TRUNK_TRACE").ok().filter(|s| !s.is_empty()).map(|path| {
                std::fs::OpenOptions::new().create(true).append(true).open(path).map(std::io::BufWriter::new).map_err(fail)
            }).transpose()?;
            pack_ms += started.elapsed().as_secs_f64() * 1000.0;
            let runtime = if physical {
                let programs = trunk::build_rowmajor_programs_for_families(8192, families).map_err(fail)?;
                let programs = programs.per_shape.into_iter().filter(|p| p.panel_count != 0).map(|p| PanelProgram {
                    shape: transport_shape(p.shape), pdi: p.design.pdi, txn: p.design.insts, padded_rows: 8192, pitch: p.padded_cols,
                }).collect();
                let pci = gpu.hip.device_pci_bus_id(gpu.device_id)?;
                Some(NpuTrunkRuntime::open(programs, resident, 2, 8192, &pci).map_err(fail)?)
            } else { None };
            let arenas = runtime.as_ref().map(|rt| -> Result<_, Qwen4GpuForwardError> {
                Ok([
                    [registered_c_view(rt, 0, Shape::GdnIn, families)?, registered_c_view(rt, 0, Shape::QsaIn, families)?],
                    [registered_c_view(rt, 1, Shape::GdnIn, families)?, registered_c_view(rt, 1, Shape::QsaIn, families)?],
                ])
            }).transpose()?;
            let (bank_b,host_tokens,host_ple) = if physical {
                let (scratch,ids,ple)=Qwen4GpuForwardScratch::new_npu_bank(gpu,&bundle.config,scratch)?;
                (Some(scratch),ids,ple)
            } else { (None,Vec::new(),Vec::new()) };
            let zab_source = if pre_overlay { "stock-MQ6-DtoH-before-overlay" } else { "stock-MQ6-DtoH" };
            let bank_b_bytes = match bank_b.as_ref() {
                Some(bank) => Qwen4GpuForwardScratch::npu_bank_plan_bytes(&bank.slots)?,
                None => 0,
            };
            if let (Some(cal), Some(key)) = (&calibration, &cal_key) {
                eprintln!("{} cal_source={cal_source}", trunk::load_log_line(cal, &trunk::LoadLogKey {
                    current: key, mode: mode.as_str(), families,
                }, n_star, &trunk::plan(cal, families, 16384), &trunk::plan(cal, families, 65536)));
            }
            eprintln!("npu-trunk: mode={mode} families={:#x} contract=bf16_direct trunk_route={trunk_route} zab_source={zab_source} pack_ms={pack_ms:.3} download_ms={download_ms:.3} resident_bytes={resident_bytes} npu_panels_bytes={npu_panels_bytes} gpu_twin_bytes={gpu_twin_bytes} bank_b_bytes={bank_b_bytes} runtime={} raw=DtoH(no-retained-host-copy)", families.bits(), if runtime.is_some() { "open" } else { "not-opened" });
            Ok(Some(Self { mode, families, layers, emulation, scales: [d0,d1], bank_b, runtime, arenas, trace,host_tokens,host_ple,pair_logged:false,
                calibration, n_star, request_plan: None, request_origin: 0 }))
        })();
        if result.is_err() { for tensor in allocations { let _ = gpu.free_tensor(tensor); } }
        result
    }
    pub(super) fn pipelined(&self) -> bool { self.mode != NpuTrunkMode::Emulate }
    pub(super) fn begin_request(&mut self) { self.pair_logged=false; self.request_plan=None; }
    pub(super) fn plan_request(&mut self, rows: usize, position: usize) {
        if self.request_plan.is_some() || self.mode != NpuTrunkMode::Auto { return; }
        let cal = self.calibration.as_ref().expect("auto requires matched calibration");
        let planned = trunk::plan(cal, self.families, rows);
        let executable = trunk::executable_plan(cal, &planned, rows, self.n_star);
        eprintln!("{}", trunk::request_log_line(rows, &executable));
        self.request_origin = position;
        self.request_plan = Some(executable);
    }
    pub(super) fn selected_pair_rows(&self, position: usize, remaining: usize, chunk: usize) -> Option<usize> {
        if self.mode != NpuTrunkMode::Auto { return pair_rows(remaining, chunk); }
        if chunk != trunk::CHUNK { return None; }
        let rows = trunk::planned_pair_rows(self.request_plan.as_ref()?, position.checked_sub(self.request_origin)?)?;
        (rows <= remaining).then_some(rows)
    }
    pub(super) fn note_pair(&mut self, rows: usize, position: usize) {
        if !self.pair_logged && self.mode != NpuTrunkMode::Auto {
            eprintln!("npu-trunk: pair rows=8192+{} position={position}",rows-8192);
            self.pair_logged=true;
        }
    }
    pub(super) fn free(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let mut first = None;
        if let Some(rt) = self.runtime { if let Err(e) = rt.close() { first = Some(hip_bridge::HipError::new(1, &e)); } }
        if let Some(b) = self.bank_b { if let Some(e) = b.free_gpu(gpu) { first.get_or_insert(e); } }
        let twins = self.layers.into_iter().filter_map(|layer| layer.twin).flat_map(|twin| [twin.w, twin.scales]);
        let emulation = self.emulation.into_iter().flat_map(|scratch| [scratch.a, scratch.sums]);
        for t in twins.chain(emulation).chain(self.scales) {
            if let Err(e) = gpu.free_tensor(t) { first.get_or_insert(e); }
        }
        first.map_or(Ok(()), Err)
    }
    fn qkv_view(&self, gpu: &Gpu, step: &Step<'_>) -> Result<GpuTensor, Qwen4GpuForwardError> {
        match step {
            Step::GatedDeltaNet(op) => {
                let mut qkv = op.projection.sub_offset(0, op.rows * 10240);
                // Match the core's native projection dtype, including chunk-off.
                if gdn_qkv_projection_bf16(gpu, op) { qkv.dtype=DType::BF16; }
                Ok(qkv)
            }
            Step::IndexedAttention(op) => Ok(op.qgate_scratch.sub_offset(0, op.rows * 12288)),
            _ => Err(fail("expected mixer step")),
        }
    }
    fn shape_map(&self, shape: Shape) -> &trunk::ColumnMap {
        &self.layers.iter().find(|l| l.shape == shape).expect("Flash-Next shape").map
    }
    fn destinations<'a>(&self, step: &'a Step<'_>, qkv: &'a GpuTensor) -> Result<SmallVec<[NpuTrunkOutput<'a>; 4]>, Qwen4GpuForwardError> {
        let (shape, targets): (Shape, [(Family, &GpuTensor); 4]) = match step {
            Step::GatedDeltaNet(op) => (Shape::GdnIn, [(Family::GdnQkv,qkv),(Family::GdnZ,op.z_output),(Family::GdnA,op.a),(Family::GdnB,op.b)]),
            Step::IndexedAttention(op) => (Shape::QsaIn, [(Family::QsaQ,op.qgate_scratch),(Family::QsaK,op.k_scratch),(Family::QsaV,op.v_scratch),(Family::QsaIdx,op.index_scratch)]),
            _ => return Err(fail("expected mixer")),
        };
        let map = self.shape_map(shape);
        targets.into_iter().filter(|(f,_)| self.families.contains(*f)).map(|(f,t)| map.gpu_output(f,t).map_err(Into::into)).collect()
    }
    fn pack(&self, gpu: &mut Gpu, step: &Step<'_>, bank: usize, pipe: bool, prep_end: Option<&hip_bridge::Event>) -> Result<u64, Qwen4GpuForwardError> {
        let (input, rows, shape) = match step { Step::GatedDeltaNet(op) => (op.input,op.rows,Shape::GdnIn), Step::IndexedAttention(op) => (op.input,op.rows,Shape::QsaIn), _ => return Err(fail("pack expected mixer")) };
        let map = self.shape_map(shape);
        let x = gpu.rotate_x_mq_batched_f16(input,2560,rows)?;
        if let Some(event) = prep_end { gpu.hip.event_record(event, gpu.active_stream.as_ref())?; }
        let pack_issue_ns=if pipe && self.trace.is_some() {monotonic_ns()}else{0};
        if pipe {
            let rt = self.runtime.as_ref().ok_or_else(|| fail("runtime missing"))?;
            unsafe { gpu.npu_trunk_pack_a_bf16_ptr(&x,rows,layout(map,8192),rt.a_arena(bank).map_err(fail)?.device as *mut c_void,&self.scales[bank])?; }
        } else { gpu.npu_trunk_pack_a_bf16(&x,rows,layout(map,rows),&self.emulation.as_ref().expect("emulate scratch").a,&self.scales[bank])?; }
        Ok(pack_issue_ns)
    }
    fn emulate(&self, gpu: &mut Gpu, layer: usize, step: &Step<'_>) -> Result<(), Qwen4GpuForwardError> {
        self.pack(gpu, step, 0, false, None)?;
        let qkv = self.qkv_view(gpu,step)?;
        let rows = match step { Step::GatedDeltaNet(op)=>op.rows, Step::IndexedAttention(op)=>op.rows, _=>unreachable!() };
        let outs = self.destinations(step,&qkv)?;
        let l = &self.layers[layer];
        let scratch = self.emulation.as_ref().expect("emulate scratch");
        let twin = l.twin.as_ref().expect("emulate twin");
        gpu.npu_trunk_emulate_bf16(&scratch.a,&twin.w,&twin.scales,&self.scales[0],&scratch.sums,layout(&l.map,rows),rows,l.map.padded_cols(),&outs)?;
        Ok(())
    }
    fn uncovered(&self, gpu: &mut Gpu, step: &Step<'_>) -> Result<(), Qwen4GpuForwardError> {
        let covered = |families: [Family; 4]| families.map(|f| self.families.contains(f));
        match step {
            Step::GatedDeltaNet(op) => execute_gated_delta_net_uncovered(gpu, op,
                covered([Family::GdnQkv, Family::GdnA, Family::GdnB, Family::GdnZ])).map_err(fail),
            Step::IndexedAttention(op) => execute_indexed_attention_uncovered(gpu, op,
                covered([Family::QsaIdx, Family::QsaQ, Family::QsaK, Family::QsaV])).map_err(fail),
            _ => unreachable!("typed mixer"),
        }
    }
}

fn skips_input_projections(phase: MixerPhase) -> bool {
    matches!(phase, MixerPhase::Core | MixerPhase::OutProj)
}
fn execute(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> Result<(), Qwen4GpuForwardError> {
    // The dispatch guards project_trunk (MQ6 AND A4/IU4) with Whole|InProj.
    // Every mixer reaching this covered executor must therefore omit both.
    debug_assert!(steps.iter().all(|step| match step {
        Step::GatedDeltaNet(op) => skips_input_projections(op.phase),
        Step::IndexedAttention(op) => skips_input_projections(op.phase),
        _ => true,
    }), "covered NPU/twin mixer must not launch native input projections");
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
            let covered = self.layers[layer].map.panel_count() != 0;
            if covered {
                self.emulate(gpu,layer,&steps[index])?;
                self.uncovered(gpu,&steps[index])?;
                phase(&mut steps[index],MixerPhase::Core);
                execute(gpu,ctx,&steps[index..=index])?;
                phase(&mut steps[index],MixerPhase::OutProj);
            }
            // Use the same layer boundaries as the pipelined route, preserving
            // output/HC/MoE fusions but never fusing across a bank boundary.
            let next=(index+1..steps.len()).find(|&j|matches!(steps[j],Step::GatedDeltaNet(_) | Step::IndexedAttention(_)));
            let end=next.map_or(steps.len(),|j| {
                let read=j-1;
                if read>0 && matches!(steps[read-1],Step::GroupedDepthwise(_)) {read-1}else{read}
            });
            if covered { execute(gpu,ctx,&steps[index..end])?; }
            else { execute_validated_steps(gpu,ctx,&steps[index..end]).map_err(fail)?; }
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
    layer: usize, bank: usize, position: usize, rows: usize, seq: u32,
    pack_issue_ns: u64, kick_ns: u64, submit_ns: u64, done_ns: u64, signal_write_ns: u64,
    wait_enqueue_ns: u64, consume_issue_ns: u64,
    start: hip_bridge::Event, end: hip_bridge::Event,
    prep_begin: hip_bridge::Event,
    prep_end: hip_bridge::Event, pack_end: hip_bridge::Event,
    wait_begin: hip_bridge::Event, wait_end: hip_bridge::Event,
}
/// Scalar JSONL schema, independent of HIP handles so CPU tests exercise the
/// same serializer as the after-pair event resolver. Existing keys stay ordered.
#[derive(Default, serde::Serialize)]
struct TraceRecord {
    layer: usize, bank: &'static str, position: usize, rows: usize, seq: u32,
    pack_issue_ns: u64, kick_ns: u64, npu_submit_ns: u64, npu_done_ns: u64,
    gpu_wait_enqueue_ns: u64, consume_issue_ns: u64,
    mixer_start_ns: u64, mixer_end_ns: u64,
    anchor_uncertainty_ns: u64,
    prep_begin_ns: u64, prep_end_ns: u64, pack_end_ns: u64,
    wait_begin_ns: u64, wait_end_ns: u64, signal_write_ns: u64,
}
impl TraceRecord {
    fn write(&self, out: &mut impl Write) -> std::io::Result<()> {
        serde_json::to_writer(&mut *out, self)?;
        writeln!(out)
    }
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
    fn prepare(&self, gpu: &mut Gpu, ctx: &DispatchCtx, program: &layers::BankProgram<'_>, layer: usize, bank: usize, records: &mut [TraceRow], stage: Option<&crate::bundle::PairSessionStage>) -> Result<Option<Seq>,Qwen4GpuForwardError> {
        let index=mixer_index(program,layer);
        let rt=self.runtime.as_ref().expect("on runtime");
        let covered = self.layers[layer].map.panel_count() != 0;
        if let Some(r) = records.get(layer*2+bank) {
            gpu.hip.event_record(&r.prep_begin, gpu.active_stream.as_ref())?;
        }
        if covered {
            unsafe { rt.prepare_bank(bank,stream(gpu)).map_err(fail)?; }
            if let Some(r) = records.iter_mut().take(layer*2).rev().find(|r| r.bank == bank && r.seq != 0 && r.done_ns == 0) {
                let completion=rt.wait_host(Seq { bank, value:r.seq }).map_err(fail)?;
                r.submit_ns=completion.submit_ns; r.done_ns=completion.done_ns; r.signal_write_ns=completion.signal_write_ns;
            }
        }
        // Includes PLE (when present) and the attention HC read. Neither reads
        // recurrent/QSA state; bank B's input can be prepared ahead of bank A.
        execute(gpu,ctx,&program.steps[program.ranges[layer].start..index])?;
        if bank == 0 {
            if let Some(stage) = stage {
                for step in &program.steps[program.ranges[layer].start..index] {
                    if let Step::GroupedDepthwise(op) = step {
                        stage.copy(gpu, stage.fixed.len() - 2, op.state).map_err(fail)?;
                    }
                }
            }
        }
        let prep_end = records.get(layer*2+bank).map(|r| &r.prep_end);
        if !covered {
            if let Some(r) = records.get(layer*2+bank) {
                for event in [&r.prep_end, &r.pack_end] {
                    gpu.hip.event_record(event, gpu.active_stream.as_ref())?;
                }
            }
            return Ok(None);
        }
        let pack_issue_ns=self.pack(gpu,&program.steps[index],bank,true,prep_end)?;
        if let Some(r) = records.get(layer*2+bank) {
            gpu.hip.event_record(&r.pack_end, gpu.active_stream.as_ref())?;
        }
        let kick_ns=if records.is_empty(){0}else{monotonic_ns()};
        let seq=unsafe { rt.kick(layer,transport_shape(self.layers[layer].shape),bank,program.rows,stream(gpu)).map_err(fail)? };
        if !records.is_empty() {
            let r=&mut records[layer*2+bank];
            r.pack_issue_ns=pack_issue_ns; r.kick_ns=kick_ns; r.seq=seq.value;
        }
        Ok(Some(seq))
    }
    fn run_bank(&self, gpu: &mut Gpu, ctx: &DispatchCtx, program: &mut layers::BankProgram<'_>, layer: usize, bank: usize, seq: Option<Seq>, records: &mut [TraceRow]) -> Result<(),Qwen4GpuForwardError> {
        let index=mixer_index(program,layer);
        let rt=self.runtime.as_ref().expect("on runtime");
        if let Some(r) = records.get(layer*2+bank) {
            gpu.hip.event_record(&r.wait_begin, gpu.active_stream.as_ref())?;
        }
        if let Some(seq) = seq { unsafe { rt.wait_gpu(seq,stream(gpu)).map_err(fail)?; } }
        let wait_enqueue_ns=monotonic_ns();
        if !records.is_empty() {
            let r=&mut records[layer*2+bank];
            r.wait_enqueue_ns=wait_enqueue_ns; r.consume_issue_ns=0;
            gpu.hip.event_record(&r.wait_end,gpu.active_stream.as_ref())?;
            gpu.hip.event_record(&r.start,gpu.active_stream.as_ref())?;
        }
        if seq.is_none() {
            execute_validated_steps(gpu,ctx,&program.steps[index..program.ranges[layer].end]).map_err(fail)?;
            if !records.is_empty() { gpu.hip.event_record(&records[layer*2+bank].end,gpu.active_stream.as_ref())?; }
            return Ok(());
        }
        self.uncovered(gpu,&program.steps[index])?;
        phase(&mut program.steps[index],MixerPhase::Core);
        let shape = self.layers[layer].shape;
        let tensor = self.arenas.as_ref().expect("on arena views")[bank][shape.index()].as_ref().expect("selected shape arena");
        let map = &self.layers[layer].map;
        let family = |family| family_view(map, tensor, family);
        match &program.steps[index] {
            Step::GatedDeltaNet(op) => execute_gated_delta_net_arena(gpu, op, GdnTrunkArenas {
                qkv: family(Family::GdnQkv), z: family(Family::GdnZ),
                a: family(Family::GdnA), b: family(Family::GdnB),
            }).map_err(fail)?,
            Step::IndexedAttention(op) => execute_indexed_attention_arena(gpu, op, QsaTrunkArenas {
                q: family(Family::QsaQ), k: family(Family::QsaK),
                v: family(Family::QsaV), idx: family(Family::QsaIdx),
            }).map_err(fail)?,
            _ => unreachable!("typed layer mixer"),
        }
        // Core queues every arena reader (including the GDN Z gate and QSA
        // cache/index appends). OutProj and the remaining layer read only scratch.
        // prepare_bank waits this event before packing A; kick records ready
        // after that wait/pack, and the relay waits ready before its next C write.
        unsafe { rt.retire_bank(seq.expect("covered bank sequence"),stream(gpu)).map_err(fail)?; }
        phase(&mut program.steps[index],MixerPhase::OutProj);
        execute(gpu,ctx,&program.steps[index..program.ranges[layer].end])?;
        if !records.is_empty() { gpu.hip.event_record(&records[layer*2+bank].end,gpu.active_stream.as_ref())?; }
        Ok(())
    }
    pub(super) fn pair(&mut self, owner: &mut Qwen4GpuForward, bundle: &mut Qwen4Bundle, gpu: &mut Gpu, tokens: &[u32], logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
        if pair_rows(tokens.len(),8192)!=Some(tokens.len()) {
            return Err(fail("pair requires bank A=8192 and bank B=512..8192 live rows"));
        }
        let old_scope=std::mem::replace(&mut gpu.qwen4_scope,true);
        let invariant=owner.prefill_scope && rdna_compute::prefill_invariant::prefill_pm_invariant_enabled();
        let old_invariant=std::mem::replace(&mut gpu.prefill_invariant_scope,invariant);
        let ragged_scope = native_trunk_prefill_scope(
            owner.prefill_scope, gpu.qwen4_trunk_iu4, &gpu.arch, owner.scratch.mode,
        );
        let old_ragged = std::mem::replace(&mut gpu.qwen4_trunk_iu4_prefill_scope, ragged_scope);
        let result=self.pair_scoped(owner,bundle,gpu,tokens,logits,top1,capture,output);
        gpu.qwen4_scope=old_scope; gpu.prefill_invariant_scope=old_invariant;
        gpu.qwen4_trunk_iu4_prefill_scope = old_ragged;
        if result.is_err() { bundle.state.invalidate_f16_arenas(); }
        result
    }
    fn pair_scoped(&mut self, owner: &mut Qwen4GpuForward, bundle: &mut Qwen4Bundle, gpu: &mut Gpu, tokens: &[u32], logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
        owner.restore_expert_stage(gpu)?;
        let rows=[8192,tokens.len()-8192];
        let position=bundle.state.position;
        let end=position.checked_add(tokens.len()).ok_or_else(||fail("position overflow"))?;
        bundle.release_session_for_growth(gpu,end);
        bundle.state.ensure_mapped_capacity(gpu,end).map_err(fail)?;
        if let Some(qsa)=bundle.state.qsa.first().filter(|q|q.f16_arena.is_none()) {
            rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(gpu,qsa.format,bundle.config.num_key_value_heads,end)?;
        }
        let mut stage = bundle.prepare_pair_session_stage(gpu, position + rows[0]).map_err(fail)?;
        self.note_pair(tokens.len(),position);
        // No tape can contain relay/host operations. Decode retains its old route.
        gpu.replay.set_forward_eligible(false);
        owner.uploaded_id_rows=0;
        drop(owner.ple_ahead.take());
        bind_bank_rows(self.bank_b.as_mut().expect("on bank B"), &bundle.config, rows[1])?;
        let b=self.bank_b.as_ref().expect("on bank B");
        let ra=matrix_view(&owner.scratch.router_logits,8192,router_columns(program_dims(&bundle.config)))?;
        let rb=matrix_view(&b.router_logits,rows[1],router_columns(program_dims(&bundle.config)))?;
        let ctx=DispatchCtx::new(gpu);
        let sa=layer_scratch(&owner.scratch,&ra);
        let sb=layer_scratch(b,&rb);
        let mut programs=[
            layers::build_program(owner,&owner.scratch,bundle,gpu,&ctx,&sa,0,rows[0])?,
            layers::build_program(owner,b,bundle,gpu,&ctx,&sb,1,rows[1])?,
        ];
        let mut history=bundle.state.ple_history;
        // Bank A only: bank B initializes after kick(0, A) (L1 fill reorder),
        // so NPU(0, A) does not wait behind bank B's embedding/PLE staging.
        initialize(gpu,bundle,&owner.scratch,&tokens[..rows[0]],&history,programs[0].ple_rows_f16,&mut self.host_tokens,&mut self.host_ple)?;
        for &token in &tokens[..rows[0]] { history.push(token); }
        let boundary_history = history;
        if let Some(stage) = &mut stage {
            let commits: SmallVec<[_; QWEN4_QSA_INLINE_CAPACITY]> = programs[0].steps.iter().filter_map(|step| match step {
                Step::IndexedAttention(op) => Some(op.next_lengths().map_err(fail)), _ => None,
            }).collect::<Result<_, _>>()?;
            stage.configure(&bundle.state, boundary_history, &commits).map_err(fail)?;
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
                records.push(TraceRow { layer,bank,position:position+bank*8192,rows:rows[bank],seq:0,pack_issue_ns:0,kick_ns:0,submit_ns:0,done_ns:0,signal_write_ns:0,wait_enqueue_ns:0,consume_issue_ns:0,prep_begin:gpu.hip.event_create()?,prep_end:gpu.hip.event_create()?,pack_end:gpu.hip.event_create()?,wait_begin:gpu.hip.event_create()?,wait_end:gpu.hip.event_create()?,start:gpu.hip.event_create()?,end:gpu.hip.event_create()? });
            }}
        }
        // The closure borrows `self` for prepare/run_bank; bank B's
        // initialization borrows the host upload buffers, so they move out.
        let mut host_tokens=std::mem::take(&mut self.host_tokens);
        let mut host_ple=std::mem::take(&mut self.host_ple);
        let attempt: Result<(),Qwen4GpuForwardError>=(|| {
            let mut seq=[self.prepare(gpu,&ctx,&programs[0],0,0,&mut records,stage.as_ref())?, None];
            // Same stream: prepare(0, A) is enqueued before bank B's
            // initialization overwrites the shared init buffers (Footprint table).
            let bank_b_tokens=&tokens[8192..8192+rows[1]];
            initialize(gpu,bundle,b,bank_b_tokens,&history,programs[1].ple_rows_f16,&mut host_tokens,&mut host_ple)?;
            for &token in bank_b_tokens { history.push(token); }
            seq[1]=self.prepare(gpu,&ctx,&programs[1],0,1,&mut records,None)?;
            let mut last_seq = seq;
            let mut gdn_slot = 0;
            let mut qsa_slot = 0;
            for (layer,bank) in bank_visits(bundle.config.num_hidden_layers) {
                self.run_bank(gpu,&ctx,&mut programs[bank],layer,bank,seq[bank],&mut records)?;
                if bank == 0 {
                    if let Some(stage) = &stage {
                        match &programs[0].steps[mixer_index(&programs[0], layer)] {
                            Step::GatedDeltaNet(op) => {
                                stage.copy_parts(gpu, 2 * gdn_slot, [op.recurrent, op.conv_state]).map_err(fail)?;
                                gdn_slot += 1;
                            }
                            Step::IndexedAttention(op) => {
                                stage.copy(gpu, 2 * bundle.state.gdn.len() + qsa_slot, op.state.selected_indices).map_err(fail)?;
                                qsa_slot += 1;
                            }
                            _ => unreachable!("typed mixer"),
                        }
                    }
                }
                if layer+1<bundle.config.num_hidden_layers {
                    seq[bank]=self.prepare(gpu,&ctx,&programs[bank],layer+1,bank,&mut records,if bank == 0 { stage.as_ref() } else { None })?;
                    if seq[bank].is_some() { last_seq[bank] = seq[bank]; }
                }
            }
            // Only already-enqueued chains are checked here. Failed terminal
            // signals may have admitted speculative fragment work, but no pair
            // output/head/capture or request state is published on that path.
            let rt=self.runtime.as_ref().expect("on runtime");
            for bank in 0..2 {
                if let Some(seq) = last_seq[bank] {
                    let completion=rt.wait_host(seq).map_err(fail)?;
                    if let Some(r) = records.iter_mut().rev().find(|r| r.bank == bank && r.seq == seq.value) {
                        r.submit_ns=completion.submit_ns; r.done_ns=completion.done_ns; r.signal_write_ns=completion.signal_write_ns;
                    }
                }
            }
            rt.check_error().map_err(fail)?;
            // Both trunks finish before prompt-end head/capture work.
            finish_bank(gpu,bundle,&owner.scratch,logits,top1,capture,0,rows[0],output)?;
            finish_bank(gpu,bundle,b,logits,top1,capture,1,rows[1],output)?;
            Ok(())
        })();
        self.host_tokens=host_tokens;
        self.host_ple=host_ple;
        drop(programs);
        let attempt = attempt.and_then(|_| {
            if let (Some(out),Some(anchor))=(&mut self.trace,&anchor) {
                if let Some(last)=records.last() { gpu.hip.event_synchronize(&last.end)?; }
                for r in &records {
                    let start_ns=anchor_ns+(gpu.hip.event_elapsed_ms(anchor,&r.start)? as f64*1e6) as u64;
                    let end_ns=anchor_ns+(gpu.hip.event_elapsed_ms(anchor,&r.end)? as f64*1e6) as u64;
                    let stamp = |event| -> Result<u64, hip_bridge::HipError> {
                        Ok(anchor_ns+(gpu.hip.event_elapsed_ms(anchor,event)? as f64*1e6) as u64)
                    };
                    TraceRecord {
                        layer:r.layer, bank:if r.bank==0{"A"}else{"B"}, position:r.position, rows:r.rows, seq:r.seq,
                        pack_issue_ns:r.pack_issue_ns, kick_ns:r.kick_ns, npu_submit_ns:r.submit_ns, npu_done_ns:r.done_ns,
                        gpu_wait_enqueue_ns:r.wait_enqueue_ns, consume_issue_ns:r.consume_issue_ns,
                        mixer_start_ns:start_ns, mixer_end_ns:end_ns, anchor_uncertainty_ns,
                        prep_begin_ns:stamp(&r.prep_begin)?, prep_end_ns:stamp(&r.prep_end)?, pack_end_ns:stamp(&r.pack_end)?,
                        wait_begin_ns:stamp(&r.wait_begin)?, wait_end_ns:stamp(&r.wait_end)?, signal_write_ns:r.signal_write_ns,
                    }.write(out).map_err(fail)?;
                }
                out.flush().map_err(fail)?;
            }
            Ok(())
        });
        for r in records {
            for event in [r.prep_begin,r.prep_end,r.pack_end,r.wait_begin,r.wait_end,r.start,r.end] { let _=gpu.hip.event_destroy(event); }
        }
        if let Some(anchor)=anchor { let _=gpu.hip.event_destroy(anchor); }
        bundle.pair_session_stage = stage;
        attempt?;
        commit_qsa(bundle,&commits);
        bundle.state.ple_history=history;
        bundle.state.commit_row_capture(tokens.len());
        bundle.state.position=end;
        if bundle.pair_boundary_is_ar() {
            bundle.publish_pair_session_boundary(gpu, position + rows[0]).map_err(fail)?;
        }
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

fn finish_bank(gpu: &mut Gpu, bundle: &Qwen4Bundle, scratch: &Qwen4GpuForwardScratch, logits: Option<&GpuTensor>, top1: Option<&GpuTensor>, capture: Option<&GpuTensor>, bank: usize, n: usize, output: Qwen4OutputPolicy) -> Result<(),Qwen4GpuForwardError> {
    let config=&bundle.config; let dims=program_dims(config);
    let bf16=gpu.qwen4_bf16_streams(n);
    if let Some(capture)=capture {
        let dst=f32_view(capture,bank*8192*dims.wide(),n*dims.wide());
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
    let offset=if rows==n {bank*8192*config.vocab_size}else{0};
    let logits=logits.ok_or_else(||fail("head requires logits"))?.sub_offset(offset,rows*config.vocab_size);
    execute_lm_head(gpu,&lm,&hidden,&logits,n,rows,Some(&scratch.rotation)).map_err(fail)?;
    if let Some(top1)=top1 {
        let indices=top1.sub_offset(if rows==n{bank*8192*4}else{0},rows*4);
        argmax_f32(gpu,&ArgmaxF32 { logits:&logits,indices:&indices,rows,vocab:config.vocab_size })?;
    }
    Ok(())
}

/// Keep the numerical twin's 8192-row chunk boundary; bank B may be partial.
pub(super) fn pair_rows(remaining: usize, chunk: usize) -> Option<usize> {
    let rows=qwen4_prefill_segment_rows(0,remaining,chunk,None,true);
    (rows>chunk).then_some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn trace_jsonl_preserves_schema_and_adds_fine_timestamps() {
        let record = TraceRecord { layer:3, bank:"B", rows:512, seq:7, prep_begin_ns:11,
            prep_end_ns:12, pack_end_ns:13, wait_begin_ns:14, wait_end_ns:17,
            signal_write_ns:16, ..Default::default() };
        let mut bytes = Vec::new();
        record.write(&mut bytes).unwrap();
        assert_eq!(bytes.iter().filter(|&&b| b==b'\n').count(), 1);
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let fields = ["layer","bank","position","rows","seq","pack_issue_ns","kick_ns",
            "npu_submit_ns","npu_done_ns","gpu_wait_enqueue_ns","consume_issue_ns",
            "mixer_start_ns","mixer_end_ns","anchor_uncertainty_ns",
            "prep_begin_ns","prep_end_ns","pack_end_ns","wait_begin_ns","wait_end_ns","signal_write_ns"];
        assert_eq!(json.as_object().unwrap().len(), fields.len());
        for key in fields { assert!(json.get(key).is_some(), "{key}"); }
        assert_eq!(json["bank"], "B");
        assert_eq!(json["prep_begin_ns"], 11);
        assert_eq!(json["wait_end_ns"], 17);
        assert_eq!(json["signal_write_ns"], 16);
    }
    #[test] fn every_mask_selects_only_covered_family_views() {
        for bits in 0..=u8::MAX {
            let mask = FamilyMask::from_bits(bits);
            for shape in Shape::ALL {
                let map = shape.column_map(mask);
                let tensor = GpuTensor {
                    buf: unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::dangling_mut(), 256 * map.padded_cols() * 2) },
                    shape: vec![256, map.padded_cols()], dtype: DType::BF16,
                };
                let mut col = 0;
                for family in Family::ALL.into_iter().filter(|f| f.shape() == shape) {
                    let view = family_view(&map, &tensor, family);
                    assert_eq!(view.is_some(), mask.contains(family));
                    if let Some(view) = view {
                        assert_eq!((view.col0_elems, view.row_pitch_elems), (col, map.padded_cols()));
                        col += family.cols();
                    }
                }
                assert_eq!(map.panel_count(), col.div_ceil(1280));
                assert_eq!(map.padded_cols(), col.div_ceil(1280) * 1280);
            }
        }
    }
    #[test] fn partial_bank_views_use_live_rows_and_keep_allocation_capacity() {
        let config = crate::config::compact_test_config();
        let dims = program_dims(&config);
        let (capacity, _) = Qwen4GpuForwardScratch::plan(&config, 8192, Qwen4ForwardScratchMode::Full).unwrap();
        let mut bank: Vec<_> = capacity.iter().map(|slot| GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(std::ptr::dangling_mut(), slot.bytes().unwrap()) },
            shape: vec![slot.elements], dtype: slot.dtype,
        }).collect();
        for rows in [8151, 512, 8192] {
            let (mut live, _) = Qwen4GpuForwardScratch::plan(&config, rows, Qwen4ForwardScratchMode::Full).unwrap();
            preserve_native_projection_capacity(&mut live, &capacity);
            bind_bank_extents(&mut bank.iter_mut().collect::<Vec<_>>(), &live).unwrap();
            for (field, (tensor, slot)) in bank.iter().zip(&live).enumerate() {
                assert_eq!(tensor.numel(), slot.elements, "field {field} rows {rows}");
                assert_eq!(tensor.buf.size(), capacity[field].bytes().unwrap());
            }
            let at = |field: ScratchField| &bank[field as usize];
            assert_eq!(at(ScratchField::PleStaged).numel(), rows * 2560);
            assert_eq!(at(ScratchField::PleRows).numel(), rows * 2560);
            assert_eq!(matrix_view(at(ScratchField::RouterLogits), rows, router_columns(dims)).unwrap().numel(), rows * router_columns(dims));
            assert_eq!(at(ScratchField::MoeXRot).numel(), rows * dims.hidden);
            assert_eq!(at(ScratchField::MoeTopkIndices).numel(), rows * config.num_experts_per_tok * 4);
            assert_eq!(at(ScratchField::MoeYGateUpGrouped).numel(), live[ScratchField::MoeYGateUpGrouped as usize].elements);
            assert_eq!(f32_view(at(ScratchField::Streams), 0, rows * dims.wide()).numel(), rows * dims.wide());
            for field in [ScratchField::HcMixed, ScratchField::GdnZ, ScratchField::GdnA, ScratchField::GdnB] {
                assert_eq!(at(field).numel(), capacity[field as usize].elements);
            }
        }
    }
    #[test] fn native_trunk_bank_decisions_use_live_rows_and_sequential_capacity() {
        let config = crate::config::compact_test_config();
        let (capacity, _) = Qwen4GpuForwardScratch::plan(&config, 8192, Qwen4ForwardScratchMode::Full).unwrap();
        assert!(native_trunk_prefill_scope(true, true, "gfx1151", Qwen4ForwardScratchMode::Full));
        assert!(!native_trunk_prefill_scope(false, true, "gfx1151", Qwen4ForwardScratchMode::Full));
        assert!(!native_trunk_prefill_scope(true, true, "gfx1151", Qwen4ForwardScratchMode::FnSymGfx1201));
        for rows in [8192, 8151, 512] {
            let (mut live, _) = Qwen4GpuForwardScratch::plan(&config, rows, Qwen4ForwardScratchMode::Full).unwrap();
            preserve_native_projection_capacity(&mut live, &capacity);
            let padded = rdna_compute::gemm::qwen4_trunk_iu4_ragged_rows(rows);
            assert_eq!(padded, (rows == 8151).then_some(8192));
            for (field, width) in [(ScratchField::HcMixed, config.hidden_size),
                (ScratchField::GdnZ, config.linear_num_value_heads * config.linear_value_head_dim),
                (ScratchField::GdnA, config.linear_num_value_heads), (ScratchField::GdnB, config.linear_num_value_heads)] {
                let padded = padded.unwrap_or(rows);
                assert!(live[field as usize].elements >= padded * width);
                assert_eq!(live[field as usize].elements, capacity[field as usize].elements);
            }
        }
    }
    #[test] fn covered_phases_skip_all_native_projection_routes() {
        // The phase gate precedes project_trunk's dtype/IU4 mask selection:
        // these decisions are identical for stock MQ6 and overlay A4.
        assert!(!skips_input_projections(MixerPhase::Whole));
        assert!(!skips_input_projections(MixerPhase::InProj));
        assert!(skips_input_projections(MixerPhase::Core));
        assert!(skips_input_projections(MixerPhase::OutProj));
    }
    #[test] fn arena_family_windows_never_include_padding() {
        for (shape, live, pitch) in [(Shape::GdnIn, 16480, 16640), (Shape::QsaIn, 13952, 14080)] {
            let map = shape.column_map(FamilyMask::ALL);
            assert_eq!(map.padded_cols(), pitch);
            assert_eq!(map.parts.last().map(|&(_, col, width)| col + width), Some(live));
            for &(_, col, width) in &map.parts {
                for rows in [512, 4647, 8192] {
                    let last_read = (rows - 1) * pitch + col as usize + width as usize - 1;
                    assert!(last_read < (rows - 1) * pitch + live as usize);
                }
            }
        }
    }
    #[test] fn pairs_cover_exact_ranges_and_keep_tail() {
        for rows in [0,511,8192,8703,8704,16200,16383,16384,16385,20000,32768,40000,40960] {
            let mut at=0; let mut ranges=Vec::new(); let mut banks=Vec::new();
            while at<rows {
                let paired=pair_rows(rows-at,8192);
                let n=paired.unwrap_or((rows-at).min(8192));
                ranges.push(at..at+n);
                if paired.is_some() {
                    banks.extend([at..at+8192,at+8192..at+n]);
                    assert!((512..=8192).contains(&(n-8192)));
                } else { banks.push(at..at+n); }
                at+=n;
            }
            assert_eq!(at,rows);
            for pair in ranges.windows(2) { assert_eq!(pair[0].end,pair[1].start); }
            // Emulate splits at the same 8192 boundary, including ragged B.
            let sequential:Vec<_>=(0..rows).step_by(8192).map(|start|start..(start+8192).min(rows)).collect();
            assert_eq!(banks,sequential);
            match rows {
                16200=>assert_eq!(ranges,vec![0..16200]),
                16384=>assert_eq!(ranges,vec![0..16384]),
                20000=>assert_eq!(ranges,vec![0..16384,16384..20000]),
                40000=>assert_eq!(ranges,vec![0..16384,16384..32768,32768..40000]),
                _=>{}
            }
        }
        assert_eq!(pair_rows(8703,8192),None);
        assert_eq!(pair_rows(8704,8192),Some(8704));
        assert_eq!(pair_rows(32768,4096),None);
    }
    #[test] fn segments_preserve_snapshots_and_native_tails() {
        // Both arms allow the A/B split itself as a snapshot boundary.
        let schedule=|end:usize,boundaries:&[usize],planned:bool| {
            let mut pos=0; let mut groups=Vec::new(); let mut banks=Vec::new();
            while pos<end {
                let boundary=boundaries.iter().copied().find(|b|*b>pos);
                let mut rows=qwen4_prefill_segment_rows(pos,end,8192,boundary,planned);
                if rows > 8192 && boundary == Some(pos + 8192)
                    && boundaries.iter().any(|&b| b > pos + 8192 && b < pos + rows)
                {
                    rows = 8192;
                }
                assert!(rows>0);
                let covered=rows>8192;
                groups.push(pos..pos+rows);
                for at in (pos..pos+rows).step_by(8192) {
                    banks.push((at..(at+8192).min(pos+rows),covered));
                }
                assert!(boundaries.iter().all(|&b| b <= pos || b >= pos+rows || b == pos+8192));
                pos+=rows;
            }
            (groups,banks)
        };
        assert_eq!(schedule(16343,&[],true).0,vec![0..16343]);
        assert_eq!(schedule(20000,&[],true).1,vec![
            (0..8192,true),(8192..16384,true),(16384..20000,false),
        ]);
        assert_eq!(schedule(8703,&[],true).1,vec![(0..8192,false),(8192..8703,false)]);
        assert_eq!(schedule(16343,&[8192,16343],true).0,vec![0..16343]);
        assert_eq!(schedule(16384,&[8192,16384],true).0,vec![0..16384]);
        assert_eq!(schedule(16384,&[8192,10000,16384],true).0,
            vec![0..8192,8192..10000,10000..16384]);
        assert_eq!(schedule(20000,&[12000,20000],true).0,
            vec![0..8192,8192..12000,12000..20000]);
        assert_eq!(schedule(20000,&[4000,13000,20000],true).0,
            vec![0..4000,4000..12192,12192..13000,13000..20000]);
        assert_eq!(qwen4_prefill_segment_rows(8192,40000,8192,Some(8192),true),8192);
        assert!(schedule(40000,&[],false).1.iter().all(|(_,covered)|!*covered));
        for end in [16343,20000,40000] {
            let (groups,banks)=schedule(end,&[],true);
            let emulated:Vec<_>=groups.into_iter().flat_map(|group| {
                let covered=group.len()>8192;
                (group.start..group.end).step_by(8192)
                    .map(move |at|(at..(at+8192).min(group.end),covered))
            }).collect();
            assert_eq!(banks,emulated);
        }
    }
    #[test] fn capture_capacity_only_doubles_for_actual_pipeline() {
        use crate::bundle::spec_hidden_capacity_rows;
        assert_eq!(spec_hidden_capacity_rows(8192,8192,true),Some(16384));
        assert_eq!(spec_hidden_capacity_rows(8192,8192,false),Some(8192));
        assert_eq!(spec_hidden_capacity_rows(20000,8192,true),Some(20000));
        assert_eq!(spec_hidden_capacity_rows(8192,usize::MAX,true),None);
    }
    #[test] fn arena_retirement_precedes_next_pack_and_state_order() {
        let mut operations=vec![(0,0,"pack"),(0,1,"pack")];
        for (layer,bank) in bank_visits(3) {
            operations.extend([(layer,bank,"core"),(layer,bank,"retire")]);
            if layer+1<3 { operations.push((layer+1,bank,"pack")); }
        }
        for layer in 0..3 {
            let index=|bank,op|operations.iter().position(|x|*x==(layer,bank,op)).unwrap();
            assert!(index(0,"core")<index(1,"core"));
            for bank in 0..2 { assert!(index(bank,"core")<index(bank,"retire")); }
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
    let logits=logits.map(|t|if all { t.sub_offset(offset*bundle.config.vocab_size,tokens.len()*bundle.config.vocab_size) } else { t.sub_offset(0,t.numel()) });
    let top1=top1.map(|t|if all {t.sub_offset(offset*4,tokens.len()*4)}else{t.sub_offset(0,t.numel())});
    let capture=capture.map(|t|t.sub_offset(offset*program_dims(&bundle.config).wide(),tokens.len()*program_dims(&bundle.config).wide()));
    let mut npu=owner.npu_trunk.take().expect("on owner");
    let result=npu.pair(owner,bundle,gpu,tokens,logits.as_ref(),top1.as_ref(),capture.as_ref(),output);
    owner.npu_trunk=Some(npu);
    result
}

fn bank_visits(layers: usize) -> impl Iterator<Item=(usize,usize)> {
    (0..layers).flat_map(|layer|[(layer,0),(layer,1)])
}
