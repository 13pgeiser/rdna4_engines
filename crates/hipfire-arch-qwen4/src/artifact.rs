// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked Qwen4 HFQM artifact admission.
//!
//! This is the one source/index validator for Qwen4.  Loader, evaluator, and
//! any future carrier must consume [`admit_hfqm_artifact`] rather than parsing
//! the HFQM index or re-implementing quant/PLE geometry locally.  The receipt
//! is value-only: it owns validated declarations and a source seal, but never
//! allocates GPU memory or starts a reader worker.

use crate::config::{Qwen4Config, ARCH_ID};
use crate::ple::PleHashMetadata;
use crate::weights::{
    is_ple_shard_name, Qwen4Manifest, Qwen4Placement, PLE_ROW_WIDTH, PLE_SHARD_COUNT,
    PLE_SHARD_ROWS,
};
use hipfire_runtime::external_rows::RowEncoding;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::model_source::{SourceFormat, SourceIdentity, SourceRangeDescriptor};
use hipfire_runtime::ple_hot_pack::PackOrder;
use hipfire_runtime::ple_hot_store::HotRowStore;
use hipfire_runtime::ple_q4::{lm4_codebook_bits, lm4_scale_grid_bits};
use hipfire_runtime::ple_sidecar::ple_shard_name;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::external_row_stride;
use rdna_compute::DType;
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

const QWEN4_FORMAT_VERSION: u32 = 1;
const QWEN4_QT_BF16: u8 = 16;
const QWEN4_QT_I64: u8 = 54;
const QWEN4_QT_MQ4G256V2: u8 = 44;
const QWEN4_QT_MQ4G128V2: u8 = 53;
/// MQ6G256V2 (qt=47): aligned-K 256 groups, 200 B/group (6-bit payload).
const QWEN4_QT_MQ6G256V2: u8 = 47;
/// Q8F16 (qt=3): f16 scale + 32 int8 per block, 34 bytes per 32 weights.  Not
/// FWHT-rotated.  The MoE fixed classes (embed, lm_head) ship at this tier.
const QWEN4_QT_Q8F16: u8 = 3;
/// MFP4G32E8SOA (qt=35): E8-lattice SoA rows, 4.25 bpw, 32-weight blocks with a
/// 16-byte-aligned padded scale block.
const QWEN4_QT_MFP4G32E8SOA: u8 = 35;
/// One FWHT-256 segment: the unit the encoder rotates and the decode GEMV/GEMM
/// assert (`assert!(k % 256 == 0)`).  A payload that only satisfied the
/// 32-weight block size would pass every byte count here and then either panic
/// or read unrotated activations.
const QWEN4_MFP4G32E8SOA_K_ALIGNMENT: usize = 256;

const PLE_MULTIPLIERS_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
const PLE_OFFSETS_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";
const PLE_VOCAB_SIZES_NAME: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes";
/// PLE metadata is deliberately tiny.  Refuse an unexpectedly large record
/// before allocating a host read buffer, even if a corrupt index advertises a
/// matching shape with a huge byte extent.
const MAX_PLE_METADATA_BYTES: usize = 2048;

/// Where an admitted artifact's PLE n-gram rows come from.
///
/// The rows are the only part of a Qwen4 model that can live outside the HFQM
/// file the loader opens: [`Self::Sidecar`] is a `<model>.ple-q4` /
/// `<model>.ple-q8` file attached with
/// `hipfire_runtime::ple_sidecar::resolve_and_attach_ple_rows`.
#[derive(Clone, Debug)]
pub enum Qwen4PleRows {
    /// The 128 shard tensors of the admitted HFQM file itself (the
    /// monolithic artifact, read exactly as before sidecars existed).
    Embedded,
    /// An attached, validated sidecar rows file.
    Sidecar {
        path: PathBuf,
        encoding: RowEncoding,
        rows: SidecarRows,
    },
}

/// How a sidecar's rows are served.
#[derive(Clone)]
pub enum SidecarRows {
    /// Row-order table: the 128 ordered shard ranges under one sealed
    /// single-file identity, read through the bounded row store like the
    /// embedded shards.
    Descriptors(Vec<SourceRangeDescriptor>),
    /// Rank-order table: the rows are stored hottest-first with an embedded
    /// rank, so the file is its own hot-row pack; the store serves rows
    /// directly and the row store is built on top of it.
    Table(Arc<HotRowStore>),
}

impl fmt::Debug for SidecarRows {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptors(descriptors) => f
                .debug_tuple("Descriptors")
                .field(&descriptors.len())
                .finish(),
            Self::Table(_) => f.write_str("Table"),
        }
    }
}

impl Qwen4PleRows {
    /// The rank-order table store, when the rows come from one.
    pub fn rank_table(&self) -> Option<&Arc<HotRowStore>> {
        match self {
            Self::Sidecar {
                rows: SidecarRows::Table(store),
                ..
            } => Some(store),
            _ => None,
        }
    }
}

/// All declarations needed by the executable carrier and source evaluator.
///
/// The receipt contains no GPU handle and no mutable source reader.  The
/// `source_identity` seal is retained so an evaluator can bind any later
/// range reads to the exact artifact that was admitted.
#[derive(Clone, Debug)]
pub struct Qwen4HfqmArtifact {
    pub config: Qwen4Config,
    /// The weight manifest.  With a sidecar the 128 PLE shard entries are
    /// removed ([`Qwen4Manifest::without_ple_shards`]): their rows are read
    /// from the sidecar, never fulfilled from the HFQM file.
    pub manifest: Qwen4Manifest,
    pub ple: PleHashMetadata,
    pub placements: Vec<Qwen4Placement>,
    pub source_tensor_count: usize,
    pub source_identity: Arc<SourceIdentity>,
    /// The PLE rows source the bundle must be assembled with
    /// (`Qwen4Bundle::assemble_with_ple_rows`).
    pub ple_rows: Qwen4PleRows,
}

/// A checked artifact boundary failure.  Keeping the error owned and typed
/// lets loader/evaluator callers preserve one diagnostic convention without
/// exposing private parser helpers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ArtifactError {
    message: String,
}

impl Qwen4ArtifactError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Qwen4ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Qwen4ArtifactError {}

/// Escape hatch for deliberately loading an artifact whose manifest target is
/// quantized while its bytes are BF16 (the unpacked decode path).
fn unpacked_quantized_target_allowed() -> bool {
    hipfire_config::developer_var("HIPFIRE_ALLOW_BF16_QUANTIZED_TARGET")
        .is_ok_and(|value| value == "1")
}

fn source_dtype_name(quant_type: u8) -> Option<&'static str> {
    match quant_type {
        QWEN4_QT_BF16 => Some("BF16"),
        QWEN4_QT_I64 => Some("I64"),
        QWEN4_QT_MQ4G256V2 => Some("MQ4G256V2"),
        QWEN4_QT_MQ4G128V2 => Some("MQ4G128V2"),
        QWEN4_QT_MQ6G256V2 => Some("MQ6G256V2"),
        // The effective-source seal is composed by the runtime from the artifact's
        // own index through `quant_type_to_dtype`, which spells qt=3 "Q8_0". This
        // table is compared against that seal, so the spelling has to match it.
        QWEN4_QT_Q8F16 => Some("Q8_0"),
        QWEN4_QT_MFP4G32E8SOA => Some("MFP4G32E8SOA"),
        _ => None,
    }
}

fn shape_elements(shape: &[u32], name: &str) -> Result<usize, Qwen4ArtifactError> {
    if shape.is_empty() || shape.iter().any(|&dimension| dimension == 0) {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} has an empty or zero-sized shape {shape:?}"
        )));
    }
    shape.iter().try_fold(1usize, |elements, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| {
            Qwen4ArtifactError::new(format!("qwen4: source tensor {name} shape overflows usize"))
        })?;
        elements.checked_mul(dimension).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {name} element count overflows usize"
            ))
        })
    })
}

/// Row stride and byte extent of a packed `[rows..., K]` tensor, refusing a
/// `K` that is not a multiple of `k_alignment` (1 = any `K`).
fn quantized_extent(
    shape: &[u32],
    name: &str,
    dtype: DType,
    k_alignment: usize,
) -> Result<(usize, usize), Qwen4ArtifactError> {
    if shape.len() < 2 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: quantized source tensor {name} shape {shape:?} has no row/K dimensions"
        )));
    }
    let k = usize::try_from(*shape.last().expect("shape length checked")).map_err(|_| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} K dimension overflows usize"
        ))
    })?;
    if k == 0 || k % k_alignment != 0 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: {dtype:?} tensor {name} needs a nonzero K multiple of {k_alignment}, got {k}"
        )));
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &dimension| {
            rows.checked_mul(usize::try_from(dimension).ok()?)
        })
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {name} row count overflows usize"
            ))
        })?;
    let row_stride = dtype.row_bytes(k).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} has no {dtype:?} row layout for K={k}"
        ))
    })?;
    let extent = rows.checked_mul(row_stride).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {name} data extent overflows usize"
        ))
    })?;
    Ok((row_stride, extent))
}

fn validate_weight_geometry(
    entry: &WeightEntry,
    info: &HfqTensorInfo,
) -> Result<(), Qwen4ArtifactError> {
    let expected_shape = entry
        .logical_shape
        .iter()
        .copied()
        .map(|dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: manifest dimension for {} does not fit HFQ shape",
                    entry.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if info.shape != expected_shape {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source tensor {} shape {:?} does not match manifest {:?}",
            entry.name, info.shape, entry.logical_shape
        )));
    }

    let (source_dtype, expected_extent, row_stride) = match info.quant_type {
        QWEN4_QT_BF16 => {
            if info.group_size != 0 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: BF16 tensor {} has group_size={}, expected 0",
                    entry.name, info.group_size
                )));
            }
            let elements = shape_elements(&info.shape, &entry.name)?;
            let extent = elements.checked_mul(2).ok_or_else(|| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: BF16 tensor {} data extent overflows usize",
                    entry.name
                ))
            })?;
            (Some(DType::BF16), extent, 0usize)
        }
        QWEN4_QT_MQ4G256V2 => {
            if info.group_size != 256 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt44 group_size={}, expected 256",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ4G256V2, 256)?;
            (Some(DType::MQ4G256V2), extent, row_stride)
        }
        QWEN4_QT_MQ6G256V2 => {
            if info.group_size != 256 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt47 group_size={}, expected 256",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ6G256V2, 256)?;
            (Some(DType::MQ6G256V2), extent, row_stride)
        }
        QWEN4_QT_MFP4G32E8SOA => {
            if info.group_size != 32 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt35 group_size={}, expected 32",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) = quantized_extent(
                &info.shape,
                &entry.name,
                DType::MFP4G32E8SOA,
                QWEN4_MFP4G32E8SOA_K_ALIGNMENT,
            )?;
            (Some(DType::MFP4G32E8SOA), extent, row_stride)
        }
        QWEN4_QT_Q8F16 => {
            if info.group_size != 32 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt3 group_size={}, expected 32",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) = quantized_extent(&info.shape, &entry.name, DType::Q8_0, 1)?;
            (Some(DType::Q8_0), extent, row_stride)
        }
        QWEN4_QT_MQ4G128V2 => {
            if info.group_size != 128 {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} qt53 group_size={}, expected 128",
                    entry.name, info.group_size
                )));
            }
            let (row_stride, extent) =
                quantized_extent(&info.shape, &entry.name, DType::MQ4G128V2, 1)?;
            (Some(DType::MQ4G128V2), extent, row_stride)
        }
        other => {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: tensor {} has unsupported HFQM quant_type={other}",
                entry.name
            )));
        }
    };

    match entry.dtype {
        DType::MQ4G256V2
        | DType::MQ4G128V2
        | DType::MQ6G256V2
        | DType::MFP4G32E8SOA
        | DType::Q8_0 => {
            // An artifact whose manifest target is quantized but whose bytes are
            // BF16 decodes that matrix through the unpacked path.  Supporting
            // that silently is how a packed trunk regresses to BF16 without
            // anyone noticing, so it takes an explicit opt-in — *unless* the
            // entry's own source contract admits BF16, which is the case for a
            // table that ships in two tiers and whose artifact declares which
            // one it carries (Qwen4 PLE rows: Q8F16 or BF16).
            if source_dtype == Some(DType::BF16)
                && !entry.dtype_constraint.accepts(DType::BF16)
                && !unpacked_quantized_target_allowed()
            {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} is declared {:?} by the manifest but the artifact \
                     carries BF16 bytes; this artifact would decode it unpacked. \
                     Re-publish with the packed trunk, or set \
                     HIPFIRE_ALLOW_BF16_QUANTIZED_TARGET=1 to load it anyway",
                    entry.name, entry.dtype
                )));
            }
            // A converted artifact may retain BF16 source bytes for a matrix
            // whose manifest target is quantized.  When it is already packed,
            // the tag must name a representation the entry's *own* source
            // contract admits — not the target this build happens to declare.
            // The trunk is the class where those differ: its contract names
            // both MQ6G256V2 (qt=47) and Q8F16 (qt=3) because the artifact
            // chooses the tier, and every extent above was computed from the
            // tag the artifact actually carries.  This is the same question the
            // runtime's weight store asks at fulfillment, so the two layers
            // admit the same artifacts.
            if source_dtype != Some(DType::BF16)
                && !entry
                    .dtype_constraint
                    .accepts(source_dtype.expect("every admitted tag names a dtype"))
            {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: tensor {} source quant tag {} is not an admitted source for manifest dtype {:?}",
                    entry.name, info.quant_type, entry.dtype
                )));
            }
        }
        DType::BF16 => {
            if source_dtype != Some(DType::BF16) {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: native BF16 tensor {} must use HFQM quant_type=16, got {}",
                    entry.name, info.quant_type
                )));
            }
        }
        dtype => {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest tensor {} has unsupported target dtype {dtype:?}",
                entry.name
            )));
        }
    }

    if let WeightResidency::ExternalRows {
        row_bytes,
        valid_rows,
    } = entry.residency
    {
        let physical_rows = usize::try_from(*info.shape.first().expect("shape is non-empty"))
            .map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: external tensor {} row count overflows usize",
                    entry.name
                ))
            })?;
        let width = info.shape[1..]
            .iter()
            .try_fold(1usize, |width, &dimension| {
                let dimension = usize::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: external tensor {} width overflows usize",
                        entry.name
                    ))
                })?;
                width.checked_mul(dimension).ok_or_else(|| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: external tensor {} row width overflows usize",
                        entry.name
                    ))
                })
            })?;
        // A row-addressed table declares its tier through the artifact: the
        // shard's own quant tag names the representation, its own dtype-derived
        // stride must be the one the manifest declared, and the entry's source
        // contract has to admit that tier. Qwen4 PLE rows ship as BF16 and as
        // Q8F16, so both are answered here rather than one being privileged.
        let declared = match info.quant_type {
            QWEN4_QT_BF16 => DType::BF16,
            QWEN4_QT_Q8F16 => DType::Q8_0,
            other => {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: external tensor {} has quant_type={other}, which is not a \
                     row-addressed tier",
                    entry.name
                )));
            }
        };
        external_row_stride(declared, width).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} width {width} is not addressable as {declared:?} rows",
                entry.name
            ))
        })?;
        // `row_bytes` is the stride the declaration defaults to. A table may
        // admit more than one row tier (Qwen4 PLE rows ship as BF16 and Q8F16),
        // so the declaration only has to name a stride that is coherent for this
        // width; the artifact's own tag and extent below pin the real geometry.
        let declared_stride_is_a_tier = [DType::BF16, DType::Q8_0]
            .into_iter()
            .any(|tier| external_row_stride(tier, width) == Some(row_bytes));
        if !declared_stride_is_a_tier {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} declares row_stride={}, which is not a row tier \
                 for width {width}",
                entry.name, row_bytes
            )));
        }
        if !entry.dtype_constraint.accepts(declared) {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} carries {declared:?} rows, which its manifest \
                 declaration does not admit",
                entry.name
            )));
        }
        if valid_rows == 0 || valid_rows > physical_rows {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} valid_rows={} outside 1..={physical_rows}",
                entry.name, valid_rows
            )));
        }
        let expected_group = if declared == DType::Q8_0 { 32 } else { 0 };
        if info.group_size != expected_group {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: external tensor {} has quant_type={}, group_size={}, expected \
                 {declared:?} rows",
                entry.name, info.quant_type, info.group_size
            )));
        }
        // The extent is pinned by the tag and shape above and compared against
        // the index below; the declaration's stride only has to be a coherent
        // tier for this width, because a two-tier table cannot declare one
        // stride for both.
    } else if row_stride != 0 && info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: tensor {} row_stride={} data_size={} expected {}",
            entry.name, row_stride, info.data_size, expected_extent
        )));
    }
    if info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: tensor {} data_size={} expected {}",
            entry.name, info.data_size, expected_extent
        )));
    }
    Ok(())
}

fn validate_metadata_geometry(
    metadata: &crate::weights::MetadataTensor,
    info: &HfqTensorInfo,
) -> Result<(), Qwen4ArtifactError> {
    let expected_shape = metadata
        .shape
        .iter()
        .copied()
        .map(|dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: metadata dimension for {} does not fit HFQ shape",
                    metadata.name
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if info.shape != expected_shape {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} shape {:?} does not match {:?}",
            metadata.name, info.shape, metadata.shape
        )));
    }
    if metadata.source_dtype != "I64" || info.quant_type != QWEN4_QT_I64 || info.group_size != 0 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} must use HFQM I64 (qt54/group0), got qt{}/group{}",
            metadata.name, info.quant_type, info.group_size
        )));
    }
    let expected_extent = shape_elements(&info.shape, &metadata.name)?
        .checked_mul(8)
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: metadata tensor {} data extent overflows usize",
                metadata.name
            ))
        })?;
    if info.data_size != expected_extent {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} data_size={} expected {} (I64)",
            metadata.name, info.data_size, expected_extent
        )));
    }
    if expected_extent > MAX_PLE_METADATA_BYTES {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: metadata tensor {} exceeds bounded PLE record size {}",
            metadata.name, MAX_PLE_METADATA_BYTES
        )));
    }
    Ok(())
}

fn is_vision_tensor(name: &str) -> bool {
    [
        "model.visual.",
        "model.vision_tower.",
        "model.vision_projection.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "visual.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn validate_source_identity(hfq: &HfqFile) -> Result<Arc<SourceIdentity>, Qwen4ArtifactError> {
    let effective = hfq.source_identity().map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: source identity seal failed: {error}"))
    })?;
    if effective.format != SourceFormat::Hfq {
        return Err(Qwen4ArtifactError::new("qwen4: source identity is not HFQ"));
    }
    if effective.files.len() != 1 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: exact HFQM source requires one backing file, got {}",
            effective.files.len()
        )));
    }
    let loaded = hfq.load_identity().map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: source identity stat failed: {error}"))
    })?;
    if loaded.arch_id != ARCH_ID
        || effective.canonical_path != loaded.canonical_path
        || effective.metadata_json != loaded.metadata_json
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source identity metadata/path does not match opened HFQM",
        ));
    }
    let file = &effective.files[0];
    if file.canonical_path != loaded.canonical_path
        || file.dev != loaded.dev
        || file.ino != loaded.ino
        || file.len != loaded.len
        || file.mtime_secs != loaded.mtime_secs
        || file.mtime_nanos != loaded.mtime_nanos
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source file identity changed while admitting HFQM",
        ));
    }
    if effective.manifest.len() != hfq.tensors().len()
        || loaded.manifest.len() != hfq.tensors().len()
    {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source identity manifest length does not match HFQM index",
        ));
    }
    for (index, info) in hfq.tensors().iter().enumerate() {
        let range_identity = &effective.manifest[index];
        let loaded_identity = &loaded.manifest[index];
        let expected_dtype = source_dtype_name(info.quant_type).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source identity contains unknown HFQM quant_type={} for {}",
                info.quant_type, info.name
            ))
        })?;
        let logical_shape = info
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        if range_identity.name != info.name
            || range_identity.file_index != 0
            || range_identity.offset != info.data_offset as u64
            || range_identity.length != info.data_size as u64
            || range_identity.dtype != expected_dtype
            || range_identity.logical_shape != logical_shape
            || loaded_identity.name != info.name
            || loaded_identity.quant_type != info.quant_type
            || loaded_identity.shape != info.shape
            || loaded_identity.group_size != info.group_size
            || loaded_identity.data_offset != info.data_offset
            || loaded_identity.data_size != info.data_size
        {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source identity geometry for {} does not match HFQM index",
                info.name
            )));
        }
        let range = hfq
            .tensor_range(&info.name)
            .map_err(|error| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: source range seal failed for {}: {error}",
                    info.name
                ))
            })?
            .ok_or_else(|| {
                Qwen4ArtifactError::new(format!("qwen4: source range missing for {}", info.name))
            })?;
        if range.source_identity() != effective.as_ref()
            || range.offset != info.data_offset as u64
            || range.length != info.data_size as u64
            || range.dtype() != expected_dtype
            || range.logical_shape() != logical_shape
        {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source range identity for {} does not match HFQM geometry",
                info.name
            )));
        }
    }
    Ok(effective)
}

/// The base file's PLE shard tensors against what the rows source allows.
///
/// Returns the number of PLE shard tensors the index carries (and that the
/// manifest therefore must not declare).  `sidecar` is whether a PLE rows
/// sidecar is attached: then the base is either a split base (marker present,
/// no shards) or the monolithic file (all shards, shadowed by the sidecar and
/// never read).  Without a sidecar the manifest declares the shards, so a
/// split base fails here with the actionable message instead of a bare
/// missing-tensor error.
fn validate_ple_shard_presence(hfq: &HfqFile, sidecar: bool) -> Result<usize, Qwen4ArtifactError> {
    let present = hfq
        .tensors()
        .iter()
        .filter(|tensor| is_ple_shard_name(&tensor.name))
        .count();
    let split = present == 0 && hfq.ple_rows_marker().is_some();
    if !sidecar {
        if split {
            return Err(Qwen4ArtifactError::new(
                "qwen4: this is a split base (its PLE n-gram rows live in a sidecar) but no PLE \
                 rows file is attached; put <model>.ple-q4 or <model>.ple-q8 next to it \
                 (`hipfire pull <model> --ple q4|q8`) and select the format with \
                 memory.ple_format",
            ));
        }
        return Ok(0);
    }
    if present != 0 && present != PLE_SHARD_COUNT {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source carries {present} of {PLE_SHARD_COUNT} PLE shard tensors; a PLE rows \
             sidecar needs all of them (monolithic) or none (split base)"
        )));
    }
    if present == 0 && !split {
        return Err(Qwen4ArtifactError::new(
            "qwen4: source has no PLE shard tensors and no hipfire_ple_rows split-base marker; \
             a PLE rows sidecar cannot bind to it",
        ));
    }
    Ok(present)
}

fn validate_manifest_inventory(
    hfq: &HfqFile,
    manifest: &Qwen4Manifest,
    shadowed_ple_shards: usize,
) -> Result<usize, Qwen4ArtifactError> {
    let mut expected = BTreeMap::<String, Vec<u32>>::new();
    for entry in &manifest.weights {
        if matches!(&entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let shape = entry
            .logical_shape
            .iter()
            .copied()
            .map(|dimension| {
                u32::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: manifest dimension for {} does not fit HFQ shape",
                        entry.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if expected.insert(entry.name.clone(), shape).is_some() {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest contains duplicate source tensor {}",
                entry.name
            )));
        }
    }
    for entry in &manifest.metadata {
        let shape = entry
            .shape
            .iter()
            .copied()
            .map(|dimension| {
                u32::try_from(dimension).map_err(|_| {
                    Qwen4ArtifactError::new(format!(
                        "qwen4: metadata dimension for {} does not fit HFQ shape",
                        entry.name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if expected.insert(entry.name.clone(), shape).is_some() {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: manifest metadata duplicates source tensor {}",
                entry.name
            )));
        }
    }
    let manifest_count = manifest.source_tensor_count();
    if expected.len() != manifest_count {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: manifest source inventory has {} descriptors but contract declares {manifest_count}",
            expected.len()
        )));
    }
    let hfq_declared = hfq.tensors().len() - shadowed_ple_shards;
    if hfq_declared != manifest_count {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: HFQ source inventory has {hfq_declared} tensors but manifest declares {manifest_count}",
        )));
    }
    for tensor in hfq.tensors() {
        if is_vision_tensor(&tensor.name) {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: vision tensor {} is not admitted by the text-only source boundary",
                tensor.name
            )));
        }
        if shadowed_ple_shards != 0 && is_ple_shard_name(&tensor.name) {
            // Shadowed by the attached sidecar: `HfqFile::attach_ple_rows`
            // already bound their geometry to the sidecar header.
            continue;
        }
        let Some(shape) = expected.remove(&tensor.name) else {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} is not declared by the manifest",
                tensor.name
            )));
        };
        if tensor.shape != shape {
            return Err(Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} has shape {:?}, expected manifest shape {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
    }
    if let Some((name, _)) = expected.into_iter().next() {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source inventory is missing manifest tensor {name}"
        )));
    }
    for entry in &manifest.weights {
        if matches!(&entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let info = hfq.find_tensor_info(&entry.name).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: source tensor {} disappeared while validating geometry",
                entry.name
            ))
        })?;
        validate_weight_geometry(entry, info)?;
    }
    for metadata in &manifest.metadata {
        let info = hfq.find_tensor_info(&metadata.name).ok_or_else(|| {
            Qwen4ArtifactError::new(format!(
                "qwen4: metadata tensor {} disappeared while validating geometry",
                metadata.name
            ))
        })?;
        validate_metadata_geometry(metadata, info)?;
    }
    Ok(manifest_count)
}

/// Validate one exact Qwen4 HFQM artifact before any GPU allocation.
///
/// The check covers the immutable container/index identity, the complete
/// manifest inventory, quantized row geometry, PLE external rows, and the
/// three signed-I64 PLE metadata records.  It intentionally accepts only the
/// executable text HFQM contract; safetensors, overlays, vision records, and
/// unknown quant tags fail closed.
pub fn admit_hfqm_artifact(hfq: &HfqFile) -> Result<Qwen4HfqmArtifact, Qwen4ArtifactError> {
    if hfq.arch_id != ARCH_ID {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: source arch_id={} does not match reserved arch_id={ARCH_ID}",
            hfq.arch_id
        )));
    }
    if hfq.format_version != QWEN4_FORMAT_VERSION {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: unsupported HFQM format version {} (expected {})",
            hfq.format_version, QWEN4_FORMAT_VERSION
        )));
    }
    if hfq.has_overlay() {
        return Err(Qwen4ArtifactError::new(
            "qwen4: REAP/head overlays are not admitted for exact HFQM inventory",
        ));
    }
    let config = Qwen4Config::from_metadata_json(&hfq.metadata_json).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: config admission failed: {error}"))
    })?;
    let mut manifest = Qwen4Manifest::build(&config).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: manifest admission failed: {error}"))
    })?;
    let sidecar = hfq.ple_rows().is_some();
    let shadowed_ple_shards = validate_ple_shard_presence(hfq, sidecar)?;
    if sidecar {
        manifest = manifest.without_ple_shards();
    }
    let source_tensor_count = validate_manifest_inventory(hfq, &manifest, shadowed_ple_shards)?;
    let ple = PleHashMetadata::from_json(&hfq.metadata_json).map_err(|error| {
        Qwen4ArtifactError::new(format!("qwen4: PLE metadata admission failed: {error}"))
    })?;
    let canonical_ple = PleHashMetadata::qwen4();
    if ple != canonical_ple {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE metadata does not match the pinned manifest contract",
        ));
    }

    // Seal every descriptor before reading the three tiny records.  This
    // keeps the bounded host reads tied to the same source identity as all
    // later external-row reads.
    let source_identity = validate_source_identity(hfq)?;
    validate_ple_records(hfq, &canonical_ple)?;
    let ple_rows = admit_ple_rows(hfq, &canonical_ple)?;

    let placements = manifest
        .resident_keys()
        .map(|(name, layer)| Qwen4Placement {
            name: name.to_string(),
            layer,
            device: 0,
        })
        .collect();
    Ok(Qwen4HfqmArtifact {
        config,
        manifest,
        ple,
        placements,
        source_tensor_count,
        source_identity,
        ple_rows,
    })
}

/// The geometry facts of a PLE table header that admission checks against the
/// canonical contract (a plain value so the checks are testable without
/// building a container).
#[derive(Clone, Debug, PartialEq, Eq)]
struct SidecarGeometry {
    rows_per_shard: u64,
    row_width: u64,
    padded_rows: u64,
    encoding: RowEncoding,
    encoded_row_bytes: u64,
    codebook_f32_bits: Option<[u32; 16]>,
    scale_grid_f32_bits: Option<[u32; 9]>,
}

impl SidecarGeometry {
    fn from_header(
        header: &hipfire_runtime::ple_hot_pack::PleHotPackHeader,
    ) -> Result<Self, Qwen4ArtifactError> {
        let table = header.table.as_ref().ok_or_else(|| {
            Qwen4ArtifactError::new(
                "qwen4: PLE rows sidecar geometry: the file is a hot-row pack, not a PLE table",
            )
        })?;
        Ok(Self {
            rows_per_shard: header.rows_per_shard,
            row_width: header.row_width,
            padded_rows: header.padded_rows,
            encoding: header.row_encoding,
            encoded_row_bytes: header.encoded_row_bytes,
            codebook_f32_bits: table.codebook_f32_bits,
            scale_grid_f32_bits: table.scale_grid_f32_bits,
        })
    }
}

/// The sidecar's geometry against the canonical PLE contract, and the row
/// encoding it declares.
///
/// A sidecar is accepted only at the exact geometry the pinned metadata
/// hashes into: [`PLE_SHARD_COUNT`] shards of [`PLE_SHARD_ROWS`] rows of
/// [`PLE_ROW_WIDTH`] values, summing to the metadata's padded rows.  Q4LM
/// must carry the exact LM4 codebook and scale grid the decoder is compiled
/// with (the decoder has no way to read another), and Q8F16 carries neither.
fn validate_sidecar_geometry(
    geometry: &SidecarGeometry,
    canonical: &PleHashMetadata,
) -> Result<RowEncoding, Qwen4ArtifactError> {
    let fail =
        |what: String| Qwen4ArtifactError::new(format!("qwen4: PLE rows sidecar geometry: {what}"));
    if geometry.rows_per_shard != PLE_SHARD_ROWS as u64
        || geometry.row_width != PLE_ROW_WIDTH as u64
    {
        return Err(fail(format!(
            "{} rows per shard x {} values, expected {PLE_SHARD_ROWS} x {PLE_ROW_WIDTH}",
            geometry.rows_per_shard, geometry.row_width
        )));
    }
    // Canonical rows per shard times `PLE_SHARD_COUNT` is the metadata's
    // padded row count, so equality here pins the shard count too.
    if geometry.padded_rows != canonical.padded_rows()
        || geometry.padded_rows != (PLE_SHARD_ROWS * PLE_SHARD_COUNT) as u64
    {
        return Err(fail(format!(
            "{} padded rows, PLE metadata has {} ({PLE_SHARD_COUNT} shards of {PLE_SHARD_ROWS})",
            geometry.padded_rows,
            canonical.padded_rows()
        )));
    }
    let encoding = geometry.encoding;
    if !matches!(encoding, RowEncoding::Q8F16 | RowEncoding::Q4Lm) {
        return Err(fail(format!(
            "row encoding {} is not Q8F16 or Q4LM",
            encoding.dtype_name()
        )));
    }
    let row_bytes = encoding.encoded_row_bytes(PLE_ROW_WIDTH) as u64;
    if geometry.encoded_row_bytes != row_bytes {
        return Err(fail(format!(
            "{} rows are {} bytes, expected {row_bytes}",
            encoding.dtype_name(),
            geometry.encoded_row_bytes
        )));
    }
    let (book, grid) = match encoding {
        RowEncoding::Q4Lm => (Some(lm4_codebook_bits()), Some(lm4_scale_grid_bits())),
        _ => (None, None),
    };
    if geometry.codebook_f32_bits != book || geometry.scale_grid_f32_bits != grid {
        return Err(fail(format!(
            "{} codebook/scale grid do not match this build's Lloyd-Max tables",
            encoding.dtype_name()
        )));
    }
    Ok(encoding)
}

/// The PLE rows source of `hfq`: its attached sidecar (validated) or, with
/// none attached, its own embedded shard tensors.
fn admit_ple_rows(
    hfq: &HfqFile,
    canonical: &PleHashMetadata,
) -> Result<Qwen4PleRows, Qwen4ArtifactError> {
    let Some(rows) = hfq.ple_rows() else {
        return Ok(Qwen4PleRows::Embedded);
    };
    let path = rows.path();
    let geometry = SidecarGeometry::from_header(rows.header())?;
    let encoding = validate_sidecar_geometry(&geometry, canonical)?;
    let sidecar_rows = match rows.order() {
        PackOrder::Row => {
            let descriptors = rows.shard_descriptors().map_err(|error| {
                Qwen4ArtifactError::new(format!(
                    "qwen4: PLE rows sidecar {} descriptors could not be sealed: {error}",
                    path.display()
                ))
            })?;
            if descriptors.len() != PLE_SHARD_COUNT {
                return Err(Qwen4ArtifactError::new(format!(
                    "qwen4: PLE rows sidecar {} yields {} shard descriptors, expected \
                     {PLE_SHARD_COUNT}",
                    path.display(),
                    descriptors.len()
                )));
            }
            SidecarRows::Descriptors(descriptors)
        }
        PackOrder::Rank => SidecarRows::Table(rows.open_table_store().map_err(|error| {
            Qwen4ArtifactError::new(format!(
                "qwen4: PLE rows sidecar {} could not be opened: {error}",
                path.display()
            ))
        })?),
    };
    Ok(Qwen4PleRows::Sidecar {
        path: path.to_path_buf(),
        encoding,
        rows: sidecar_rows,
    })
}

impl Qwen4HfqmArtifact {
    /// The [`PLE_SHARD_COUNT`] ordered shard descriptors of the PLE rows
    /// source: the row-order sidecar's, or the artifact's own shard tensors
    /// (`hfq` must be the file this receipt was admitted from).  A
    /// rank-order table has none: it is its own hot-row pack and is read
    /// through its store ([`Qwen4PleRows::rank_table`]).
    pub fn ple_row_descriptors(
        &self,
        hfq: &HfqFile,
    ) -> Result<Vec<SourceRangeDescriptor>, Qwen4ArtifactError> {
        match &self.ple_rows {
            Qwen4PleRows::Sidecar {
                rows: SidecarRows::Descriptors(descriptors),
                ..
            } => Ok(descriptors.clone()),
            Qwen4PleRows::Sidecar {
                path,
                rows: SidecarRows::Table(_),
                ..
            } => Err(Qwen4ArtifactError::new(format!(
                "qwen4: PLE rows {} are a rank-order table; it has no row-order shard ranges",
                path.display()
            ))),
            Qwen4PleRows::Embedded => (0..PLE_SHARD_COUNT)
                .map(|shard| {
                    let name = ple_shard_name(shard);
                    hfq.tensor_range(&name)
                        .map_err(|error| {
                            Qwen4ArtifactError::new(format!(
                                "qwen4: PLE shard range {name} could not be sealed: {error}"
                            ))
                        })?
                        .ok_or_else(|| {
                            Qwen4ArtifactError::new(format!("qwen4: PLE shard {name} is missing"))
                        })
                })
                .collect(),
        }
    }

    /// The file the PLE rows are read from and its length: the sidecar, or
    /// the model file itself (path as opened).  A hot-row pack is cut from
    /// and verified against exactly this file.
    pub fn ple_rows_source(&self, hfq: &HfqFile) -> Result<(PathBuf, u64), Qwen4ArtifactError> {
        match &self.ple_rows {
            Qwen4PleRows::Embedded => {
                let file = self.source_identity.files.first().ok_or_else(|| {
                    Qwen4ArtifactError::new("qwen4: source identity has no backing file")
                })?;
                Ok((hfq.path().to_path_buf(), file.len))
            }
            Qwen4PleRows::Sidecar { path, rows, .. } => {
                let bytes = match rows {
                    SidecarRows::Descriptors(descriptors) => descriptors
                        .first()
                        .and_then(|descriptor| descriptor.source_identity().files.first())
                        .map(|file| file.len)
                        .ok_or_else(|| {
                            Qwen4ArtifactError::new("qwen4: PLE rows sidecar has no backing file")
                        })?,
                    SidecarRows::Table(_) => std::fs::metadata(path)
                        .map_err(|error| {
                            Qwen4ArtifactError::new(format!(
                                "qwen4: stat PLE rows {}: {error}",
                                path.display()
                            ))
                        })?
                        .len(),
                };
                Ok((path.clone(), bytes))
            }
        }
    }
}

fn read_i64_record(
    hfq: &HfqFile,
    name: &str,
    expected_values: usize,
) -> Result<Vec<i64>, Qwen4ArtifactError> {
    let expected_bytes = expected_values.checked_mul(8).ok_or_else(|| {
        Qwen4ArtifactError::new(format!(
            "qwen4: PLE record {name} byte count overflows usize"
        ))
    })?;
    if expected_bytes > MAX_PLE_METADATA_BYTES {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE record {name} exceeds bounded read size {MAX_PLE_METADATA_BYTES}"
        )));
    }
    let info = hfq.find_tensor_info(name).ok_or_else(|| {
        Qwen4ArtifactError::new(format!("qwen4: PLE metadata record {name} is missing"))
    })?;
    if info.quant_type != QWEN4_QT_I64 || info.group_size != 0 || info.data_size != expected_bytes {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE metadata record {name} has qt={}, group={}, bytes={}, expected qt54/group0/{}",
            info.quant_type, info.group_size, info.data_size, expected_bytes
        )));
    }
    let range = hfq
        .tensor_range(name)
        .map_err(|error| {
            Qwen4ArtifactError::new(format!(
                "qwen4: PLE metadata range {name} could not be sealed: {error}"
            ))
        })?
        .ok_or_else(|| {
            Qwen4ArtifactError::new(format!("qwen4: PLE metadata range {name} is missing"))
        })?;
    if range.length != expected_bytes as u64 {
        return Err(Qwen4ArtifactError::new(format!(
            "qwen4: PLE metadata range {name} length={} expected {expected_bytes}",
            range.length
        )));
    }
    let mut bytes = vec![0u8; expected_bytes];
    range.read_exact(&mut bytes).map_err(|error| {
        Qwen4ArtifactError::new(format!(
            "qwen4: bounded PLE metadata read {name} failed: {error}"
        ))
    })?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)")))
        .collect())
}

fn validate_ple_records(
    hfq: &HfqFile,
    canonical: &PleHashMetadata,
) -> Result<(), Qwen4ArtifactError> {
    let multipliers = read_i64_record(hfq, PLE_MULTIPLIERS_NAME, 3)?;
    if multipliers.as_slice() != canonical.multipliers().as_slice() {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 multipliers payload does not match canonical metadata",
        ));
    }
    let offsets = read_i64_record(hfq, PLE_OFFSETS_NAME, 16)?;
    let expected_offsets = canonical
        .head_offsets()
        .iter()
        .map(|&value| i64::try_from(value).expect("canonical PLE offset fits i64"))
        .collect::<Vec<_>>();
    if offsets != expected_offsets {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 head_offsets payload does not match canonical metadata",
        ));
    }
    let vocab_sizes = read_i64_record(hfq, PLE_VOCAB_SIZES_NAME, 16)?;
    let expected_vocab_sizes = canonical
        .head_vocab_sizes()
        .iter()
        .map(|&value| i64::try_from(value).expect("canonical PLE vocab size fits i64"))
        .collect::<Vec<_>>();
    if vocab_sizes != expected_vocab_sizes {
        return Err(Qwen4ArtifactError::new(
            "qwen4: PLE I64 head_vocab_sizes payload does not match canonical metadata",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
    use serde_json::json;
    use std::path::{Path, PathBuf};

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn metadata_json() -> String {
        let canonical = PleHashMetadata::qwen4();
        json!({
            "qwen4_ple": {
                "version": 1,
                "multipliers": canonical.multipliers(),
                "head_vocab_sizes": canonical.head_vocab_sizes(),
                "head_offsets": canonical.head_offsets(),
                "padded_rows": canonical.padded_rows(),
            }
        })
        .to_string()
    }

    fn fixture_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-qwen4-artifact-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("artifact.hfq")
    }

    fn write_ple_fixture(path: &Path, multipliers: &[i64]) {
        let canonical = PleHashMetadata::qwen4();
        let tensors = vec![
            HfqMemTensor {
                name: PLE_MULTIPLIERS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![3],
                group_size: 0,
                data: i64_bytes(multipliers),
            },
            HfqMemTensor {
                name: PLE_OFFSETS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64_bytes(
                    &canonical
                        .head_offsets()
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                ),
            },
            HfqMemTensor {
                name: PLE_VOCAB_SIZES_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64_bytes(
                    &canonical
                        .head_vocab_sizes()
                        .iter()
                        .map(|&value| value as i64)
                        .collect::<Vec<_>>(),
                ),
            },
        ];
        write_hfqm_package_mem(path, ARCH_ID, &metadata_json(), &tensors).unwrap();
    }

    #[test]
    fn bounded_ple_i64_validation_rejects_payload_mismatch() {
        let path = fixture_path("mismatch");
        let mut multipliers = PleHashMetadata::qwen4().multipliers().to_vec();
        multipliers[1] += 1;
        write_ple_fixture(&path, &multipliers);
        let hfq = HfqFile::open(&path).unwrap();
        let error = validate_ple_records(&hfq, &PleHashMetadata::qwen4()).unwrap_err();
        assert!(
            error.to_string().contains("multipliers"),
            "payload mismatch must identify the signed I64 record: {error}"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    #[test]
    fn bounded_ple_i64_validation_accepts_canonical_payloads() {
        let path = fixture_path("canonical");
        write_ple_fixture(&path, PleHashMetadata::qwen4().multipliers());
        let hfq = HfqFile::open(&path).unwrap();
        validate_ple_records(&hfq, &PleHashMetadata::qwen4()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    /// A routed expert block is rank-3 on disk and rows-by-K in the format, so
    /// the loader's extent must flatten the leading dimensions exactly as the
    /// producer's row count does — and refuse a K the decode kernels assert on.
    #[test]
    fn mfp4e8soa_extent_flattens_expert_blocks_and_requires_fwht_alignment() {
        let soa = |shape: &[u32], name: &str| {
            quantized_extent(
                shape,
                name,
                DType::MFP4G32E8SOA,
                QWEN4_MFP4G32E8SOA_K_ALIGNMENT,
            )
        };
        let (stride, extent) = soa(&[512, 1280, 2560], "gate_up").unwrap();
        assert_eq!(stride, 16 + 80 + 80 * 16);
        assert_eq!(extent, 512 * 1280 * stride);

        // The same rows as a flat rank-2 matrix describe the same payload: the
        // flattening is exactly the leading-dimension product.
        let (_flat_stride, flat) = soa(&[512 * 1280, 2560], "gate_up_flat").unwrap();
        assert_eq!(flat, extent);

        // `moe_intermediate_size = 640` is not one FWHT-256 segment, so the
        // routed down projection cannot be carried by this format at all: the
        // boundary must say so by name instead of admitting a payload whose
        // decode would drop the unaligned tail.
        let error = soa(&[512, 2560, 640], "down").unwrap_err();
        assert!(
            error.to_string().contains("multiple of 256"),
            "a K below one FWHT-256 segment must be refused by name, got: {error}"
        );
        let error = soa(&[512, 1280, 128], "gate_up").unwrap_err();
        assert!(
            error.to_string().contains("multiple of 256"),
            "a K below one FWHT-256 segment must be refused by name, got: {error}"
        );
    }

    /// GPU-free admission probe over a real artifact.
    ///
    ///   HIPFIRE_PROBE_MODEL=<artifact.hfq> cargo test -p hipfire-arch-qwen4 --lib \
    ///     admits_a_sealed_artifact_with_its_declared_ple_tier --ignored --nocapture
    ///
    /// Reads headers and the index only. Prints the admitted PLE tier, its row
    /// stride and the external byte count, so a sealed artifact written before
    /// the Q8F16 tier and one written with it can both be checked without a GPU.
    #[test]
    #[ignore = "artifact admission probe; set HIPFIRE_PROBE_MODEL to an HFQ artifact"]
    fn admits_a_sealed_artifact_with_its_declared_ple_tier() {
        let Ok(model) = hipfire_config::developer_var("HIPFIRE_PROBE_MODEL") else {
            println!("admit-probe: skipped, HIPFIRE_PROBE_MODEL is unset");
            return;
        };
        let hfq = hipfire_runtime::hfq::HfqFile::open(Path::new(&model)).expect("open artifact");
        let admitted = admit_hfqm_artifact(&hfq).expect("artifact must be admitted");
        let mut shards = 0usize;
        let mut stride = 0usize;
        let mut external_bytes = 0u64;
        let mut tier = None;
        for entry in admitted.manifest.weights.iter() {
            if !entry.name.contains(".ngram_embedding.shard_") {
                continue;
            }
            shards += 1;
            if let hipfire_runtime::weight_manifest::WeightResidency::ExternalRows {
                row_bytes,
                ..
            } = entry.residency
            {
                stride = row_bytes;
            }
            tier = Some(entry.dtype);
            let info = hfq
                .find_tensor_info(&entry.name)
                .expect("PLE shard index entry");
            external_bytes += info.data_size as u64;
        }
        println!(
            "admit-probe {model}: admitted, {shards} PLE shards, declared {tier:?} tier, \
             stride={stride}, external_bytes={external_bytes}"
        );
        assert_eq!(shards, 128, "every PLE shard must be admitted");
        assert!(stride == 320 || stride == 170, "declared stride {stride}");
    }

    #[test]
    fn external_row_geometry_admits_each_declared_ple_tier() {
        // A PLE shard declaration admits BF16 rows and Q8F16 rows; the artifact
        // decides which one it carries, and the geometry gate must answer both
        // without privileging either.
        let mut entry = WeightEntry::layer(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            1,
            vec![4, 160],
            DType::Q8_0,
            ShardPolicy::Replicate,
        )
        .external_rows(170, 4);
        entry.dtype_constraint =
            hipfire_runtime::weight_manifest::DTypeConstraint::source_from_sources(vec![
                DType::Q8_0,
                DType::BF16,
            ]);

        let info =
            |quant_type: u8, group_size: u32, data_size: u64| hipfire_runtime::hfq::HfqTensorInfo {
                name: entry.name.clone(),
                quant_type,
                shape: vec![4, 160],
                group_size,
                data_offset: 0,
                data_size: data_size.try_into().unwrap(),
            };

        validate_weight_geometry(&entry, &info(QWEN4_QT_BF16, 0, 4 * 320))
            .expect("a sealed BF16 shard must be admitted");
        validate_weight_geometry(&entry, &info(QWEN4_QT_Q8F16, 32, 4 * 170))
            .expect("a Q8F16 shard must be admitted");

        // A stride that is no tier for this width, a mismatched index extent,
        // and a tag the declaration excludes are each refused by name.
        let mut nonsense = entry.clone();
        nonsense.residency =
            hipfire_runtime::weight_manifest::WeightResidency::external_rows(640, 4);
        let error = validate_weight_geometry(&nonsense, &info(QWEN4_QT_BF16, 0, 4 * 320))
            .expect_err("a stride that is no row tier must be refused");
        assert!(error.to_string().contains("not a row tier"), "{error}");

        let error = validate_weight_geometry(&entry, &info(QWEN4_QT_BF16, 0, 4 * 320 - 1))
            .expect_err("an extent that disagrees with the index must be refused");
        assert!(error.to_string().contains("data_size"), "{error}");

        let mut bf16_only = entry.clone();
        bf16_only.dtype_constraint =
            hipfire_runtime::weight_manifest::DTypeConstraint::source_exact(DType::BF16);
        let error = validate_weight_geometry(&bf16_only, &info(QWEN4_QT_Q8F16, 32, 4 * 170))
            .expect_err("a tier the declaration excludes must be refused");
        assert!(
            error.to_string().contains("not an admitted source"),
            "{error}"
        );
    }

    fn metadata_tensors() -> Vec<crate::weights::MetadataTensor> {
        [
            (PLE_MULTIPLIERS_NAME, 3),
            (PLE_OFFSETS_NAME, 16),
            (PLE_VOCAB_SIZES_NAME, 16),
        ]
        .into_iter()
        .map(|(name, values)| crate::weights::MetadataTensor {
            name: name.to_string(),
            shape: vec![values],
            source_dtype: "I64".to_string(),
        })
        .collect()
    }

    /// A manifest that declares only the three PLE metadata records, i.e.
    /// what a sidecar-fed manifest reduces to once the 128 shards are trimmed
    /// from a base whose other tensors are not under test.
    fn trimmed_manifest() -> Qwen4Manifest {
        Qwen4Manifest {
            weights: Vec::new(),
            state: Vec::new(),
            metadata: metadata_tensors(),
        }
    }

    /// Metadata JSON with (`split`) or without the split-base marker.
    fn marked_metadata_json(split: bool) -> String {
        let mut value: serde_json::Value = serde_json::from_str(&metadata_json()).unwrap();
        if split {
            value[hipfire_runtime::ple_sidecar::PLE_BASE_MARKER_KEY] = json!({
                "version": 1,
                "location": "sidecar",
                "shard_count": PLE_SHARD_COUNT,
                "rows_per_shard": PLE_SHARD_ROWS,
                "row_width": PLE_ROW_WIDTH,
                "q8_rows_sha256": "0".repeat(64),
            });
        }
        value.to_string()
    }

    /// The three metadata records plus `shards` tiny Q8F16 shard tensors under
    /// the canonical shard names (their geometry is not what the sidecar
    /// seams under test look at).
    fn write_base_fixture(path: &Path, split: bool, shards: usize) {
        let canonical = PleHashMetadata::qwen4();
        let i64s = |values: Vec<i64>| i64_bytes(&values);
        let mut tensors = (0..shards)
            .map(|shard| HfqMemTensor {
                name: ple_shard_name(shard),
                quant_type: QWEN4_QT_Q8F16,
                shape: vec![1, 32],
                group_size: 0,
                data: vec![0u8; 34],
            })
            .collect::<Vec<_>>();
        tensors.extend([
            HfqMemTensor {
                name: PLE_MULTIPLIERS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![3],
                group_size: 0,
                data: i64s(canonical.multipliers().to_vec()),
            },
            HfqMemTensor {
                name: PLE_OFFSETS_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64s(canonical.head_offsets().iter().map(|&v| v as i64).collect()),
            },
            HfqMemTensor {
                name: PLE_VOCAB_SIZES_NAME.to_string(),
                quant_type: QWEN4_QT_I64,
                shape: vec![16],
                group_size: 0,
                data: i64s(
                    canonical
                        .head_vocab_sizes()
                        .iter()
                        .map(|&v| v as i64)
                        .collect(),
                ),
            },
        ]);
        write_hfqm_package_mem(path, ARCH_ID, &marked_metadata_json(split), &tensors).unwrap();
    }

    fn with_base_fixture<T>(
        name: &str,
        split: bool,
        shards: usize,
        check: impl FnOnce(&HfqFile) -> T,
    ) -> T {
        let path = fixture_path(name);
        write_base_fixture(&path, split, shards);
        let hfq = HfqFile::open(&path).unwrap();
        let result = check(&hfq);
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
        result
    }

    #[test]
    fn split_base_without_sidecar_names_the_sidecar_and_the_setting() {
        with_base_fixture("split-bare", true, 0, |hfq| {
            let error = validate_ple_shard_presence(hfq, false)
                .unwrap_err()
                .to_string();
            for needle in [".ple-q4", ".ple-q8", "memory.ple_format", "split base"] {
                assert!(error.contains(needle), "{needle} missing from: {error}");
            }
        });
    }

    #[test]
    fn plain_base_without_shards_keeps_the_missing_tensor_diagnostic() {
        // No marker, no shards, no sidecar: not a split base; the inventory
        // check (not this one) refuses it as before.
        with_base_fixture("plain-bare", false, 0, |hfq| {
            assert_eq!(validate_ple_shard_presence(hfq, false).unwrap(), 0);
        });
    }

    #[test]
    fn sidecar_requires_all_shards_or_a_split_marker() {
        with_base_fixture("split-sidecar", true, 0, |hfq| {
            assert_eq!(validate_ple_shard_presence(hfq, true).unwrap(), 0);
        });
        with_base_fixture("mono-sidecar", false, PLE_SHARD_COUNT, |hfq| {
            assert_eq!(
                validate_ple_shard_presence(hfq, true).unwrap(),
                PLE_SHARD_COUNT
            );
            // Without a sidecar the same file is the grandfathered monolith.
            assert_eq!(validate_ple_shard_presence(hfq, false).unwrap(), 0);
        });
        with_base_fixture("partial-sidecar", false, 5, |hfq| {
            let error = validate_ple_shard_presence(hfq, true)
                .unwrap_err()
                .to_string();
            assert!(error.contains("5 of 128"), "{error}");
        });
        with_base_fixture("bare-sidecar", false, 0, |hfq| {
            let error = validate_ple_shard_presence(hfq, true)
                .unwrap_err()
                .to_string();
            assert!(error.contains("hipfire_ple_rows"), "{error}");
        });
    }

    #[test]
    fn inventory_accepts_a_split_base_and_a_shadowed_monolith_under_a_sidecar() {
        let manifest = trimmed_manifest();
        assert_eq!(manifest.source_tensor_count(), 3);
        // Split base: the index is exactly the trimmed manifest.
        with_base_fixture("inv-split", true, 0, |hfq| {
            let shadowed = validate_ple_shard_presence(hfq, true).unwrap();
            assert_eq!(
                validate_manifest_inventory(hfq, &manifest, shadowed).unwrap(),
                3
            );
        });
        // Monolith + sidecar: the 128 shard tensors are in the index, shadowed
        // and ignored.
        with_base_fixture("inv-mono", false, PLE_SHARD_COUNT, |hfq| {
            let shadowed = validate_ple_shard_presence(hfq, true).unwrap();
            assert_eq!(shadowed, PLE_SHARD_COUNT);
            assert_eq!(
                validate_manifest_inventory(hfq, &manifest, shadowed).unwrap(),
                3
            );
            // The same index against the trimmed manifest without the shadow
            // allowance is an inventory mismatch, never a silent pass.
            let error = validate_manifest_inventory(hfq, &manifest, 0).unwrap_err();
            assert!(error.to_string().contains("131 tensors"), "{error}");
        });
    }

    #[test]
    fn inventory_still_refuses_an_undeclared_tensor_next_to_a_sidecar() {
        let path = fixture_path("inv-foreign");
        let canonical = PleHashMetadata::qwen4();
        let record = |name: &str, shape: u32, values: Vec<i64>| HfqMemTensor {
            name: name.to_string(),
            quant_type: QWEN4_QT_I64,
            shape: vec![shape],
            group_size: 0,
            data: i64_bytes(&values),
        };
        let tensors = vec![
            HfqMemTensor {
                name: "model.unknown.weight".to_string(),
                quant_type: QWEN4_QT_BF16,
                shape: vec![1],
                group_size: 0,
                data: vec![0, 0],
            },
            record(PLE_MULTIPLIERS_NAME, 3, canonical.multipliers().to_vec()),
            record(
                PLE_OFFSETS_NAME,
                16,
                canonical.head_offsets().iter().map(|&v| v as i64).collect(),
            ),
            record(
                PLE_VOCAB_SIZES_NAME,
                16,
                canonical
                    .head_vocab_sizes()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            ),
        ];
        write_hfqm_package_mem(&path, ARCH_ID, &marked_metadata_json(true), &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        let error = validate_manifest_inventory(&hfq, &trimmed_manifest(), 0).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("4 tensors but manifest declares 3"),
            "{error}"
        );
        drop(hfq);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }

    fn sidecar_geometry(encoding: RowEncoding) -> SidecarGeometry {
        let q4 = encoding == RowEncoding::Q4Lm;
        SidecarGeometry {
            rows_per_shard: PLE_SHARD_ROWS as u64,
            row_width: PLE_ROW_WIDTH as u64,
            padded_rows: PleHashMetadata::qwen4().padded_rows(),
            encoding,
            encoded_row_bytes: encoding.encoded_row_bytes(PLE_ROW_WIDTH) as u64,
            codebook_f32_bits: q4.then(lm4_codebook_bits),
            scale_grid_f32_bits: q4.then(lm4_scale_grid_bits),
        }
    }

    #[test]
    fn sidecar_geometry_admits_the_canonical_q8_and_q4_tables() {
        let canonical = PleHashMetadata::qwen4();
        for encoding in [RowEncoding::Q8F16, RowEncoding::Q4Lm] {
            assert_eq!(
                validate_sidecar_geometry(&sidecar_geometry(encoding), &canonical).unwrap(),
                encoding
            );
        }
        assert_eq!(RowEncoding::Q4Lm.encoded_row_bytes(PLE_ROW_WIDTH), 90);
    }

    #[test]
    fn sidecar_geometry_refuses_each_deviation() {
        let canonical = PleHashMetadata::qwen4();
        let refused = |geometry: SidecarGeometry, needle: &str| {
            let error = validate_sidecar_geometry(&geometry, &canonical)
                .expect_err(needle)
                .to_string();
            assert!(error.contains(needle), "{needle} missing from: {error}");
        };
        let q8 = || sidecar_geometry(RowEncoding::Q8F16);
        let q4 = || sidecar_geometry(RowEncoding::Q4Lm);
        let mut geometry = q8();
        geometry.rows_per_shard += 1;
        refused(geometry, "rows per shard");
        let mut geometry = q8();
        geometry.row_width = 128;
        refused(geometry, "rows per shard");
        let mut geometry = q8();
        geometry.padded_rows += 1;
        refused(geometry, "padded rows");
        let mut geometry = q8();
        geometry.padded_rows /= 2;
        refused(geometry, "padded rows");
        let mut geometry = q8();
        geometry.encoding = RowEncoding::Bf16;
        refused(geometry, "not Q8F16 or Q4LM");
        let mut geometry = q8();
        geometry.encoded_row_bytes = 171;
        refused(geometry, "bytes, expected 170");
        let mut geometry = q4();
        geometry.encoded_row_bytes = 170;
        refused(geometry, "bytes, expected 90");
        let mut geometry = q4();
        geometry.codebook_f32_bits = None;
        refused(geometry, "Lloyd-Max tables");
        let mut geometry = q4();
        geometry.codebook_f32_bits.as_mut().unwrap()[3] ^= 1;
        refused(geometry, "Lloyd-Max tables");
        let mut geometry = q4();
        geometry.scale_grid_f32_bits.as_mut().unwrap()[0] ^= 1;
        refused(geometry, "Lloyd-Max tables");
        // A Q8F16 table must not carry a codebook.
        let mut geometry = q8();
        geometry.codebook_f32_bits = Some(lm4_codebook_bits());
        refused(geometry, "Lloyd-Max tables");
    }

    #[test]
    fn ple_fingerprint_helper_matches_the_metadata_fingerprint() {
        let expected = PleHashMetadata::qwen4()
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hipfire_runtime::ple_sidecar::ple_fingerprint_from_metadata(&metadata_json()).unwrap(),
            expected
        );
    }
}
