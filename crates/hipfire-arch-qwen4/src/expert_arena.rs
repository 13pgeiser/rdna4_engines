// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer expert slab arenas (subq4 K4 loader).
//!
//! A version-1 expert-slab artifact stores the 512 experts of a trunk layer
//! back to back (`gate_up`, `down` per expert, 256 B aligned), so a layer's
//! experts are one contiguous byte range of the file. The loader moves that
//! range to the device (or to pinned host memory) as ONE tensor, the *arena*,
//! and addresses every expert by its offset inside it. This module is the pure
//! plan: which bytes each layer's arena covers and where each expert's two
//! payloads sit in it. Nothing here touches a GPU.
//!
//! The plan is part of the recipe identity ([`crate::weights::Qwen4TierRecipe::FlashNextSlab`]),
//! so the arena extents the manifest declares are exactly the bytes the table
//! validated.

use crate::expert_slab_table::ExpertSlabTable;
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};

/// Suffix of the synthetic per-layer arena entry name
/// (`<prefix>.layers.{L}.mlp.experts.slab_arena`).
pub const ARENA_SUFFIX: &str = ".mlp.experts.slab_arena";

/// Codec id of an MQ4 symmetric (qt44/qt53) expert in the per-layer codec table.
pub const CODEC_MQ4: u8 = 0;
/// Codec id of an MQ3-I4 (qt131/qt132) expert in the per-layer codec table.
pub const CODEC_W3: u8 = 1;

/// One expert inside its layer's arena. Offsets are relative to the arena base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaExpert {
    pub format: ExpertFormat,
    pub gate_up_offset: usize,
    pub down_offset: usize,
}

impl ArenaExpert {
    /// The per-layer codec table byte of this expert's format.
    pub fn codec(self) -> Result<u8, ExpertArenaError> {
        match self.format {
            ExpertFormat::Mq4Xts => Ok(CODEC_MQ4),
            ExpertFormat::Mq3I4 => Ok(CODEC_W3),
            ExpertFormat::Mq2I4 => Err(ExpertArenaError::UnsupportedFormat(ExpertFormat::Mq2I4)),
        }
    }
}

/// One trunk layer's arena: the absolute file range and its 512 experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertArenaLayer {
    /// Absolute `data_offset` of the arena's first byte in the artifact.
    pub file_offset: usize,
    /// Physical arena bytes: exactly the sum of the experts' payload bytes.
    pub bytes: usize,
    pub experts: Vec<ArenaExpert>,
}

impl ExpertArenaLayer {
    /// Per-expert codec bytes (`0` = MQ4, `1` = W3). Fails closed on a W2 expert.
    pub fn codec_table(&self) -> Result<Vec<u8>, ExpertArenaError> {
        self.experts.iter().map(|expert| expert.codec()).collect()
    }

    /// `(mq4, w3)` expert counts.
    pub fn codec_counts(&self) -> (usize, usize) {
        let w3 = self
            .experts
            .iter()
            .filter(|expert| expert.format == ExpertFormat::Mq3I4)
            .count();
        (self.experts.len() - w3, w3)
    }
}

/// The 48 trunk layer arenas of one artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertArenaPlan {
    layers: Vec<ExpertArenaLayer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertArenaError {
    /// Only MQ4 symmetric and MQ3-I4 experts have kernels.
    UnsupportedFormat(ExpertFormat),
    /// The span table is not contiguous at `(layer, expert)`.
    Noncontiguous { layer: usize, expert: usize },
    /// A layer arena is larger than the 32-bit offsets of the device tables.
    TooLarge { layer: usize, bytes: usize },
}

impl std::fmt::Display for ExpertArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => write!(
                f,
                "expert format {format:?} has no gfx1201 kernels; only MQ4-XTS (44,53) and MQ3-I4 (131,132) experts are admitted"
            ),
            Self::Noncontiguous { layer, expert } => write!(
                f,
                "layer {layer} expert {expert} is not contiguous with the previous payload"
            ),
            Self::TooLarge { layer, bytes } => {
                write!(f, "layer {layer} arena of {bytes} bytes exceeds 4 GiB")
            }
        }
    }
}

impl std::error::Error for ExpertArenaError {}

impl ExpertArenaPlan {
    /// Derive the arenas from a validated span table. The table's builder
    /// guarantees back-to-back slabs; this re-checks it per layer because the
    /// arena is one read of `[first.gate_up.offset, last.down.end)`.
    pub fn from_table(table: &ExpertSlabTable) -> Result<Self, ExpertArenaError> {
        let mut layers = Vec::with_capacity(TRUNK_LAYERS as usize);
        for layer in 0..TRUNK_LAYERS as usize {
            let first = table
                .slab(layer, 0)
                .ok_or(ExpertArenaError::Noncontiguous { layer, expert: 0 })?;
            let base = first.gate_up.offset;
            let mut cursor = base;
            let mut experts = Vec::with_capacity(EXPERTS_PER_LAYER as usize);
            for expert in 0..EXPERTS_PER_LAYER as usize {
                let slab = table
                    .slab(layer, expert)
                    .ok_or(ExpertArenaError::Noncontiguous { layer, expert })?;
                if slab.gate_up.offset != cursor
                    || slab.gate_up.end() != Some(slab.down.offset)
                {
                    return Err(ExpertArenaError::Noncontiguous { layer, expert });
                }
                experts.push(ArenaExpert {
                    format: slab.format,
                    gate_up_offset: slab.gate_up.offset - base,
                    down_offset: slab.down.offset - base,
                });
                cursor = slab
                    .down
                    .end()
                    .ok_or(ExpertArenaError::Noncontiguous { layer, expert })?;
            }
            let bytes = cursor - base;
            if bytes > u32::MAX as usize {
                return Err(ExpertArenaError::TooLarge { layer, bytes });
            }
            layers.push(ExpertArenaLayer {
                file_offset: base,
                bytes,
                experts,
            });
        }
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &[ExpertArenaLayer] {
        &self.layers
    }

    pub fn layer(&self, layer: usize) -> Option<&ExpertArenaLayer> {
        self.layers.get(layer)
    }

    /// Total physical arena bytes (the routed-expert bytes admission must use).
    pub fn physical_bytes(&self) -> u64 {
        self.layers.iter().map(|layer| layer.bytes as u64).sum()
    }

    /// Synthetic manifest/entry name of `layer`'s arena under `layer_prefix`
    /// (`model.language_model.layers.{L}`).
    pub fn arena_name(layer_prefix: &str) -> String {
        format!("{layer_prefix}{ARENA_SUFFIX}")
    }

    /// Refuse anything the gfx1201 sealed route cannot execute.
    pub fn check_executable(&self) -> Result<(), ExpertArenaError> {
        for layer in &self.layers {
            layer.codec_table()?;
        }
        Ok(())
    }
}

/// The fulfillment payload of a manifest entry that is a layer slab arena, or
/// `None` when `entry` is an ordinary index tensor (the caller then reads it by
/// name as before).
///
/// Resident arenas stream through a checked source range (no host copy). A
/// host-mapped arena needs the bytes in hand, so it is served as an owned copy
/// of the mapped span; the mapping must therefore still be alive.
pub fn arena_payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    plan: &ExpertArenaPlan,
    entry: &hipfire_runtime::weight_manifest::WeightEntry,
) -> Option<Result<hipfire_runtime::model_source::SourcePayload<'a>, String>> {
    use hipfire_runtime::model_source::{SourcePayload, TensorInfo};
    use hipfire_runtime::weight_manifest::WeightResidency;
    if !entry.name.ends_with(ARENA_SUFFIX) {
        return None;
    }
    let result = (|| {
        let layer = entry
            .layer
            .and_then(|layer| plan.layer(layer))
            .ok_or_else(|| format!("slab arena entry '{}' names no trunk layer", entry.name))?;
        if entry.logical_shape.as_slice() != [layer.bytes] {
            return Err(format!(
                "slab arena entry '{}' declares {:?}, plan has {} bytes",
                entry.name, entry.logical_shape, layer.bytes
            ));
        }
        if entry.residency == WeightResidency::HostMapped {
            let bytes = source
                .span_bytes(layer.file_offset, layer.bytes)
                .ok_or_else(|| {
                    format!(
                        "slab arena '{}': the container mapping is unavailable",
                        entry.name
                    )
                })?
                .to_vec();
            return Ok(SourcePayload::Owned {
                info: TensorInfo {
                    name: entry.name.clone(),
                    dtype: "RAW".to_string(),
                    shape: vec![layer.bytes],
                    quant_type: 0xFF,
                    data_offset: layer.file_offset,
                    data_size: layer.bytes,
                },
                bytes,
            });
        }
        source
            .span_range(layer.file_offset, layer.bytes, "RAW", vec![layer.bytes])
            .map(SourcePayload::Range)
            .map_err(|error| error.to_string())
    })();
    Some(result)
}
