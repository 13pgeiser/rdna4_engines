// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Long-prefill expert staging for sealed expert-slab layers.
//!
//! The legacy stage (`Qwen4ExpertStage`) borrows the two last resident trunk
//! layers as donors and copies a host layer's aggregate gate/up and down
//! tensors over them. A slab layer is one arena whose per-expert extents
//! (MQ4 or W3, by codec) differ from layer to layer, so a donor layer cannot
//! hold an arbitrary cold arena. RouteOwner3 therefore refused staging for
//! slab bundles, and every cold layer's grouped K2 GEMMs read host memory in
//! place over PCIe (measured: cold-layer expert kernels 4.8-6.3x the staged
//! legacy ones, rcpt/speed/prefill.attrib.json).
//!
//! Slab staging uses two dedicated VRAM stage buffers (one per layer parity,
//! each the largest cold arena plus its zeroed tail pad) instead of donors:
//! nothing is overwritten, so there is nothing to restore. Each cold layer's
//! pointer tables name the layer's own arena offsets rebased onto its parity's
//! stage buffer; the layer's codec tag tensor (VRAM, per layer) is untouched.
//! The copy moves the arena plus its tail pad, so the kernels read the same
//! bytes (including the W3 down over-read) they would read in place.
//! Forwards below `min_rows` rows, decode and MTP verify keep reading in place.

use super::*;
use crate::expert_arena::ARENA_TAIL_PAD_BYTES;

/// Host sources of the staged layers.
pub(super) struct SlabStageMeta {
    /// Per host layer from `Qwen4ExpertStage::first_host`: the host address of
    /// the layer's arena and the bytes to copy (arena plus tail pad).
    sources: Vec<(usize, usize)>,
}

/// Bytes of one stage buffer: the largest cold arena plus its tail pad.
pub(super) fn stage_buffer_bytes(arena_bytes: impl IntoIterator<Item = usize>) -> Option<usize> {
    arena_bytes
        .into_iter()
        .max()
        .and_then(|largest| largest.checked_add(ARENA_TAIL_PAD_BYTES))
}

/// A layer's per-expert pointer table moved from the layer's arena to
/// `stage_base`: every entry must lie inside `[arena_base, arena_base +
/// arena_bytes)`; the result is the little-endian device pointer table.
pub(super) fn rebase_stage_entries(
    entries: &[usize],
    arena_base: usize,
    arena_bytes: usize,
    stage_base: usize,
) -> Result<Vec<u8>, String> {
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    for (expert, &entry) in entries.iter().enumerate() {
        let offset = entry
            .checked_sub(arena_base)
            .filter(|offset| *offset < arena_bytes)
            .ok_or_else(|| {
                format!(
                    "expert {expert} entry {entry:#x} is outside the arena [{arena_base:#x}, +{arena_bytes})"
                )
            })?;
        let rebased = stage_base
            .checked_add(offset)
            .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?;
        table.extend_from_slice(&rebased.to_ne_bytes());
    }
    Ok(table)
}

impl Qwen4ExpertStage {
    /// Slab-arena stage, or `None` (printed) when this bundle cannot stage.
    pub(super) fn new_slab(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Qwen4MoeLayerRuntime],
        first_host: usize,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            Ok(None)
        };
        if bundle.weights.ep_placement().is_some() {
            return off("slab arenas have no EP route");
        }
        if !moe[first_host..].iter().all(RoutedExpertWeights::host_mapped) {
            return off("host-mapped slab layers are not a suffix");
        }
        let config = &bundle.config;
        let mut sources = Vec::with_capacity(moe.len() - first_host);
        for (layer, runtime) in moe.iter().enumerate().skip(first_host) {
            let Some(slab) = runtime.slab.as_ref() else {
                return off("a host layer is not a slab arena");
            };
            let arena_ref = bundle.weights.layer_refs[layer]
                .ffn
                .moe()
                .and_then(|weights| weights.expert_arena.as_ref())
                .ok_or_else(|| invalid("slab stage layer has no arena reference"))?;
            let owner = bundle.weights.resident(arena_ref)?;
            let Some(host) = gpu.host_bytes(owner) else {
                return off("a host slab arena has no pinned host bytes");
            };
            if host.len() < slab.arena_bytes {
                return off("a host slab arena is shorter than its plan");
            }
            // Host-mapped allocations carry a zeroed tail pad past `len()`
            // (`HOST_TAIL_PAD_BYTES`, 1 MiB >= `ARENA_TAIL_PAD_BYTES`).
            sources.push((host.as_ptr() as usize, slab.arena_bytes + ARENA_TAIL_PAD_BYTES));
        }
        let Some(stage_bytes) = stage_buffer_bytes(
            moe[first_host..]
                .iter()
                .filter_map(|runtime| runtime.slab.as_ref().map(|slab| slab.arena_bytes)),
        ) else {
            return off("no slab host layers");
        };
        // Staging is an optimisation: stage buffers that do not fit VRAM leave
        // every host layer reading its arena in place.
        let mut buffers = Vec::with_capacity(2);
        for _ in 0..2 {
            match gpu.alloc_tensor(&[stage_bytes], DType::Raw) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        let _ = gpu.free_tensor(buffer);
                    }
                    return off(&format!(
                        "two {} MiB stage buffers do not fit VRAM: {error}",
                        stage_bytes >> 20
                    ));
                }
            }
        }
        gpu.ensure_capture_stream()?;
        let mut stage = Self {
            min_rows: qwen4_expert_stage_min_rows(),
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors: [usize::MAX; 2],
            donor_views: Vec::new(),
            mirrors: Vec::with_capacity(2),
            tables: Vec::with_capacity(2 * sources.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            placement: None,
            bytes: [0, 0],
            copy_stream: gpu.hip.stream_create_non_blocking()?,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: Some(SlabStageMeta { sources }),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            stage.mirrors.extend(buffers);
            for (layer, runtime) in moe.iter().enumerate().skip(first_host) {
                let slab = runtime.slab.as_ref().expect("checked above");
                let stage_base = stage.mirrors[layer % 2].buf.as_ptr() as usize;
                for entries in [&runtime.expert_gate_up_entries, &runtime.expert_down_entries] {
                    let table = rebase_stage_entries(entries, slab.arena_base, slab.arena_bytes, stage_base)
                        .map_err(|reason| invalid(format!("layer {layer} slab stage table: {reason}")))?;
                    let tensor = gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?;
                    stage.tables.push(tensor);
                    gpu.memcpy_htod_auto(&stage.tables.last().expect("table just allocated").buf, &table)?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage slab host layers {first_host}..{} through two {} MiB VRAM stage buffers (nothing borrowed, nothing to restore; wide-hidden capture forwards {})",
            stage.min_rows,
            moe.len(),
            stage_bytes >> 20,
            if stage.capture { "stage" } else { "mapped" }
        );
        Ok(Some(stage))
    }

    /// Start of a staged forward: the first two host layers (one per parity)
    /// copy while the resident layers compute.
    pub(super) fn slab_prime(&self, gpu: &Gpu, layers: usize) -> Result<(), Qwen4GpuForwardError> {
        if self.slab.is_none() {
            return Ok(());
        }
        for layer in self.first_host..(self.first_host + 2).min(layers) {
            self.slab_enqueue(gpu, layer)?;
        }
        Ok(())
    }

    /// Copy host `layer`'s arena over its parity's stage buffer once the
    /// previous occupant's down released it, recording `ready`.
    fn slab_enqueue(&self, gpu: &Gpu, layer: usize) -> Result<(), Qwen4GpuForwardError> {
        let meta = self.slab.as_ref().ok_or_else(|| invalid("not a slab stage"))?;
        let &(address, bytes) = meta
            .sources
            .get(layer - self.first_host)
            .ok_or_else(|| invalid("slab stage layer out of range"))?;
        let parity = layer % 2;
        gpu.hip.stream_wait_event(&self.copy_stream, &self.free[parity])?;
        // SAFETY: `address` is the registered host allocation of the layer's
        // arena (immutable for the bundle's lifetime) and `bytes` is the arena
        // plus `ARENA_TAIL_PAD_BYTES` of its zeroed tail slack (see `new_slab`).
        let source = unsafe { std::slice::from_raw_parts(address as *const u8, bytes) };
        gpu.hip
            .memcpy_htod_async(&self.mirrors[parity].buf, source, &self.copy_stream)?;
        gpu.hip.event_record(&self.ready[parity], Some(&self.copy_stream))?;
        Ok(())
    }

    /// After a host layer's MoE (its down recorded `free`): refill its parity
    /// for host layer L+2.
    pub(super) fn slab_after_moe(
        &self,
        gpu: &Gpu,
        moe: &[Qwen4MoeLayerRuntime],
        layer: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        if !moe[layer].host_mapped() {
            return Ok(());
        }
        match moe.get(layer + 2) {
            Some(next) if next.host_mapped() => self.slab_enqueue(gpu, layer + 2),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_moves_every_entry_by_the_same_delta() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        let entries = [arena, arena + 1280, arena + 99_999];
        let table = rebase_stage_entries(&entries, arena, 100_000, stage).unwrap();
        let decoded: Vec<usize> = table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![stage, stage + 1280, stage + 99_999]);
    }

    #[test]
    fn rebase_refuses_entries_outside_the_arena_after_an_arena_swap() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        // Below the base, at the end, and a table of a different (swapped) arena.
        assert!(rebase_stage_entries(&[arena - 8], arena, 100_000, stage).is_err());
        assert!(rebase_stage_entries(&[arena + 100_000], arena, 100_000, stage).is_err());
        let other_arena = 0x9000_0000usize;
        assert!(rebase_stage_entries(&[other_arena + 16], arena, 100_000, stage).is_err());
        assert!(rebase_stage_entries(&[arena, usize::MAX], arena, 100_000, usize::MAX - 8).is_err());
    }

    #[test]
    fn stage_buffer_holds_the_largest_arena_and_its_tail_pad() {
        assert_eq!(stage_buffer_bytes([10, 30, 20]), Some(30 + ARENA_TAIL_PAD_BYTES));
        assert_eq!(stage_buffer_bytes(std::iter::empty()), None);
        assert_eq!(stage_buffer_bytes([usize::MAX]), None);
    }
}
