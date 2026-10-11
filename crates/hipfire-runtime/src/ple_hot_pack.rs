// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE hot-row pack (`<model>.plehotrows`): the hot rows of the `.plehot` rank
//! sidecar re-laid out into a separate file so the runtime can read them
//! through the kernel page cache (clean, file-backed, never swapped).
//!
//! Rows are copied *verbatim* from the model file's PLE shards, in the exact
//! order `ple_hot_store::HotRowStore` keeps them in RAM, so the bytes a request
//! returns never change.  This module is the file FORMAT: the header reader
//! and the writer (driven offline by `hipfire-quantize ple-hot-pack`).
//!
//! # Layout (all integers little-endian)
//!
//! ```text
//! 0..8     magic b"HFPLEHP1"
//! 8..16    u64 header_len
//! 16..     header: UTF-8 JSON, header_len bytes
//! pad      zero bytes up to data_offset (data_offset = round_up(16 + header_len, 4096))
//! data     tier 0 rows, tier 1 rows, ..., tier T-1 rows; each row `encoded_row_bytes` verbatim
//! EOF      file length == data_offset + data_bytes
//! ```
//!
//! * `kept_buckets` = K: buckets `0..K` of the sidecar are packed;
//!   `tier_group` = G (default `ple_hot_store::DEFAULT_TIER_BUCKETS`).
//! * Tier `t` holds every padded row `r` with `bucket[r] / G == t` and
//!   `bucket[r] < K`, in ascending global row order.  `T = ceil(K / G)`.
//!   Tier byte base = `sum(tier_rows[..t]) * encoded_row_bytes`.
//! * Global row `r = shard * rows_per_shard + local_row`; its model-file bytes
//!   are at `shard_offsets[shard] + local_row * encoded_row_bytes` (padding rows included).
//! * Pack offset of a kept row = `data_offset + tier_base[t] + slot(r) * encoded_row_bytes`,
//!   `slot(r)` = number of rows of tier `t` with a smaller global row index
//!   (exactly `HotRowStore::slot`).
//!
//! Header JSON fields (all required): `version` (2), `padded_rows`,
//! `rows_per_shard`, `row_encoding` (the artifact dtype name of the row encoding,
//! `RowEncoding::dtype_name()`, parsed back with `RowEncoding::from_dtype`: `"Q8_0"`
//! for 170-byte PLE rows, `"BF16"`, ...), `row_width` (values per row; PLE is 160),
//! `encoded_row_bytes` (must equal `row_encoding.encoded_row_bytes(row_width)`; the
//! pack layout never assumes a particular size), `ple_fingerprint` (64 lowercase
//! hex), `rank_payload_sha256` (the sidecar's `payload_sha256`), `tier_group`,
//! `kept_buckets`, `tier_rows` (u64 per tier), `data_offset`, `data_bytes`
//! (= sum(tier_rows) * `encoded_row_bytes`), `shard_offsets` (model-file offset of each PLE
//! shard), `model_file_bytes` (size of the model the pack was cut from),
//! `generator` (`"hipfire-quantize ple-hot-pack"`).
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PACK_MAGIC: &[u8; 8]; PACK_VERSION: u64
//! PleHotPackHeader { padded_rows, rows_per_shard, row_encoding, row_width, encoded_row_bytes,
//!                    ple_fingerprint, rank_payload_sha256, tier_group, kept_buckets, tier_rows,
//!                    data_offset, data_bytes, shard_offsets, model_file_bytes }
//! read_ple_hot_pack_header(path: &Path) -> Result<PleHotPackHeader, String>
//! checked_row_bytes(encoding: RowEncoding, width: usize) -> Result<usize, String>
//! write_ple_hot_pack(model_path: &Path, shard_offsets: &[u64], rows_per_shard: usize,
//!                    row_encoding: RowEncoding, row_width: usize,
//!                    fingerprint_hex: &str, rank: &PleHotRank, kept_buckets: usize,
//!                    tier_group: usize, out: &Path) -> Result<PleHotPackHeader, String>
//! write_ple_hot_pack_with_progress(.., out: &Path, progress: &mut dyn FnMut(u64, u64))
//!     -> Result<PleHotPackHeader, String>      // progress(rows_done, rows_total)
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::external_rows::RowEncoding;
use crate::ple_hot_store::{
    open_direct, AlignedBuf, PleHotRank, ShardRead, BUCKETS, READ_ALIGN, READ_CHUNK,
};

pub const PACK_MAGIC: &[u8; 8] = b"HFPLEHP1";
pub const PACK_VERSION: u64 = 2;

/// Sanity bound on the JSON header.
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// The data section starts on this boundary.
const DATA_ALIGN: u64 = 4096;
/// A tier's output buffer is written out when it reaches this size.
const FLUSH_BYTES: usize = 8 << 20;
/// `tier_of` value of a bucket that is not kept.
const NO_TIER: u8 = u8::MAX;
const GENERATOR: &str = "hipfire-quantize ple-hot-pack";

/// A validated pack header (see the module docs for the field meanings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHotPackHeader {
    pub padded_rows: u64,
    pub rows_per_shard: u64,
    /// Row encoding of every packed row (the model's PLE shard encoding).
    pub row_encoding: RowEncoding,
    /// Values per row.
    pub row_width: u64,
    /// Bytes per row; equals `row_encoding.encoded_row_bytes(row_width)`.
    pub encoded_row_bytes: u64,
    pub ple_fingerprint: String,
    pub rank_payload_sha256: String,
    pub tier_group: usize,
    pub kept_buckets: usize,
    pub tier_rows: Vec<u64>,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub shard_offsets: Vec<u64>,
    pub model_file_bytes: u64,
}

fn is_hex64_lower(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Bytes of one `width`-value row of `encoding`, or an error for a width the
/// encoding cannot hold (zero, or a partial quantisation block).
pub fn checked_row_bytes(encoding: RowEncoding, width: usize) -> Result<usize, String> {
    crate::weight_store::external_row_stride(encoding.dtype(), width)
        .filter(|&bytes| width > 0 && bytes > 0)
        .ok_or_else(|| {
            format!(
                "row width {width} is not a valid {} row width",
                encoding.dtype_name()
            )
        })
}

fn round_up_data(n: u64) -> u64 {
    n.next_multiple_of(DATA_ALIGN)
}

fn tier_count(kept_buckets: usize, tier_group: usize) -> usize {
    kept_buckets.div_ceil(tier_group)
}

fn json_header(h: &PleHotPackHeader) -> serde_json::Value {
    serde_json::json!({
        "version": PACK_VERSION,
        "padded_rows": h.padded_rows,
        "rows_per_shard": h.rows_per_shard,
        "row_encoding": h.row_encoding.dtype_name(),
        "row_width": h.row_width,
        "encoded_row_bytes": h.encoded_row_bytes,
        "ple_fingerprint": h.ple_fingerprint,
        "rank_payload_sha256": h.rank_payload_sha256,
        "tier_group": h.tier_group,
        "kept_buckets": h.kept_buckets,
        "tier_rows": h.tier_rows,
        "data_offset": h.data_offset,
        "data_bytes": h.data_bytes,
        "shard_offsets": h.shard_offsets,
        "model_file_bytes": h.model_file_bytes,
        "generator": GENERATOR,
    })
}

/// Serialise `header`, setting its `data_offset` to the fixed point of
/// `round_up(16 + len(json(data_offset)), 4096)`.
fn encode_header(header: &mut PleHotPackHeader) -> Result<Vec<u8>, String> {
    let mut candidate = DATA_ALIGN;
    loop {
        header.data_offset = candidate;
        let bytes = serde_json::to_vec(&json_header(header))
            .map_err(|e| format!("ple hot pack header: {e}"))?;
        if bytes.len() as u64 > MAX_HEADER_BYTES {
            return Err(format!("ple hot pack header is {} bytes", bytes.len()));
        }
        let needed = round_up_data(16 + bytes.len() as u64);
        if needed == candidate {
            return Ok(bytes);
        }
        candidate = needed;
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Read and validate a `.plehotrows` header: magic, version, header bound,
/// field presence and types, row size, tier count, `data_bytes`, `data_offset`
/// alignment rule, group/kept ranges, and the exact file length.  The data
/// section is not read.
pub fn read_ple_hot_pack_header(path: &Path) -> Result<PleHotPackHeader, String> {
    let bad = |what: String| format!("ple hot pack {}: {what}", path.display());
    let io_err = |what: &str, e: io::Error| format!("ple hot pack {}: {what}: {e}", path.display());
    let mut file = File::open(path).map_err(|e| io_err("open", e))?;
    let file_len = file.metadata().map_err(|e| io_err("stat", e))?.len();
    let mut prefix = [0u8; 16];
    file.read_exact(&mut prefix)
        .map_err(|e| io_err("read preamble", e))?;
    if &prefix[..8] != PACK_MAGIC {
        return Err(bad("bad magic".to_string()));
    }
    let header_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
    if header_len == 0 || header_len > MAX_HEADER_BYTES || 16 + header_len > file_len {
        return Err(bad(format!("implausible header length {header_len}")));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|e| io_err("read header", e))?;
    let json: serde_json::Value =
        serde_json::from_slice(&header_bytes).map_err(|e| bad(format!("parse header: {e}")))?;

    let field = |name: &str| {
        json.get(name)
            .ok_or_else(|| bad(format!("header lacks `{name}`")))
    };
    let field_u64 = |name: &str| {
        field(name)?
            .as_u64()
            .ok_or_else(|| bad(format!("header `{name}` is not an unsigned integer")))
    };
    let field_usize = |name: &str| {
        usize::try_from(field_u64(name)?).map_err(|_| bad(format!("header `{name}` overflows")))
    };
    let field_hex = |name: &str| {
        let text = field(name)?
            .as_str()
            .ok_or_else(|| bad(format!("header `{name}` is not a string")))?;
        if is_hex64_lower(text) {
            Ok(text.to_string())
        } else {
            Err(bad(format!(
                "header `{name}` is not 64 lowercase hex digits"
            )))
        }
    };
    let field_u64_array = |name: &str| -> Result<Vec<u64>, String> {
        field(name)?
            .as_array()
            .ok_or_else(|| bad(format!("header `{name}` is not an array")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| bad(format!("header `{name}` holds a non-integer")))
            })
            .collect()
    };

    let version = field_u64("version")?;
    if version != PACK_VERSION {
        return Err(bad(format!("version {version}, expected {PACK_VERSION}")));
    }
    if field("generator")?.as_str().is_none() {
        return Err(bad("header `generator` is not a string".to_string()));
    }
    let header = PleHotPackHeader {
        padded_rows: field_u64("padded_rows")?,
        rows_per_shard: field_u64("rows_per_shard")?,
        row_encoding: {
            let text = field("row_encoding")?
                .as_str()
                .ok_or_else(|| bad("header `row_encoding` is not a string".to_string()))?;
            RowEncoding::from_dtype(text).ok_or_else(|| {
                bad(format!(
                    "header `row_encoding` `{text}` is not a known row encoding"
                ))
            })?
        },
        row_width: field_u64("row_width")?,
        encoded_row_bytes: field_u64("encoded_row_bytes")?,
        ple_fingerprint: field_hex("ple_fingerprint")?,
        rank_payload_sha256: field_hex("rank_payload_sha256")?,
        tier_group: field_usize("tier_group")?,
        kept_buckets: field_usize("kept_buckets")?,
        tier_rows: field_u64_array("tier_rows")?,
        data_offset: field_u64("data_offset")?,
        data_bytes: field_u64("data_bytes")?,
        shard_offsets: field_u64_array("shard_offsets")?,
        model_file_bytes: field_u64("model_file_bytes")?,
    };

    let declared_bytes = usize::try_from(header.row_width)
        .map_err(|_| bad("header `row_width` overflows".to_string()))
        .and_then(|width| checked_row_bytes(header.row_encoding, width).map_err(&bad))?;
    if header.encoded_row_bytes != declared_bytes as u64 {
        return Err(bad(format!(
            "encoded_row_bytes {}, but {} rows of {} values are {declared_bytes} bytes",
            header.encoded_row_bytes,
            header.row_encoding.dtype_name(),
            header.row_width
        )));
    }
    if !(2..=BUCKETS).contains(&header.tier_group) {
        return Err(bad(format!(
            "tier_group {} outside 2..={BUCKETS}",
            header.tier_group
        )));
    }
    if !(1..=BUCKETS).contains(&header.kept_buckets) {
        return Err(bad(format!(
            "kept_buckets {} outside 1..={BUCKETS}",
            header.kept_buckets
        )));
    }
    let tiers = tier_count(header.kept_buckets, header.tier_group);
    if header.tier_rows.len() != tiers {
        return Err(bad(format!(
            "tier_rows has {} entries, expected {tiers}",
            header.tier_rows.len()
        )));
    }
    let total_rows = header
        .tier_rows
        .iter()
        .try_fold(0u64, |sum, &n| sum.checked_add(n))
        .ok_or_else(|| bad("tier_rows overflow".to_string()))?;
    let expected_bytes = total_rows
        .checked_mul(header.encoded_row_bytes)
        .ok_or_else(|| bad("data_bytes overflow".to_string()))?;
    if header.data_bytes != expected_bytes {
        return Err(bad(format!(
            "data_bytes {} != sum(tier_rows) * {} = {expected_bytes}",
            header.data_bytes, header.encoded_row_bytes
        )));
    }
    if total_rows > header.padded_rows {
        return Err(bad(format!(
            "tier_rows sum {total_rows} exceeds padded_rows {}",
            header.padded_rows
        )));
    }
    if header.rows_per_shard == 0
        || header.shard_offsets.len() as u64 * header.rows_per_shard != header.padded_rows
    {
        return Err(bad(format!(
            "{} shards of {} rows do not cover padded_rows {}",
            header.shard_offsets.len(),
            header.rows_per_shard,
            header.padded_rows
        )));
    }
    let expected_offset = round_up_data(16 + header_len);
    if header.data_offset != expected_offset {
        return Err(bad(format!(
            "data_offset {} != round_up(16 + header_len, {DATA_ALIGN}) = {expected_offset}",
            header.data_offset
        )));
    }
    let expected_len = header
        .data_offset
        .checked_add(header.data_bytes)
        .ok_or_else(|| bad("file length overflow".to_string()))?;
    if file_len != expected_len {
        return Err(bad(format!(
            "file is {file_len} bytes, expected {expected_len}"
        )));
    }
    Ok(header)
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Model file read through O_DIRECT (or buffered where the filesystem refuses
/// the flag).
struct DirectRead(File);

impl ShardRead for DirectRead {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.0.read_at(dst, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                other => return other,
            }
        }
    }
}

/// Cut the pack `out` from the model file `model_path` (O_DIRECT reads).
/// See [`write_ple_hot_pack_with_progress`].
#[allow(clippy::too_many_arguments)]
pub fn write_ple_hot_pack(
    model_path: &Path,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
) -> Result<PleHotPackHeader, String> {
    write_ple_hot_pack_with_progress(
        model_path,
        shard_offsets,
        rows_per_shard,
        row_encoding,
        row_width,
        fingerprint_hex,
        rank,
        kept_buckets,
        tier_group,
        out,
        &mut |_, _| {},
    )
}

/// [`write_ple_hot_pack`] reporting `progress(rows_done, rows_total)` after
/// every model window read.  The pack is written to `<out>.tmp.<pid>` and
/// renamed into place after `sync_all`; the tmp file is removed on error.
#[allow(clippy::too_many_arguments)]
pub fn write_ple_hot_pack_with_progress(
    model_path: &Path,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PleHotPackHeader, String> {
    let file = open_direct(model_path)
        .map_err(|e| format!("ple hot pack: open {}: {e}", model_path.display()))?;
    let model_file_bytes = file
        .metadata()
        .map_err(|e| format!("ple hot pack: stat {}: {e}", model_path.display()))?
        .len();
    write_pack_with(
        &DirectRead(file),
        model_file_bytes,
        shard_offsets,
        rows_per_shard,
        row_encoding,
        row_width,
        fingerprint_hex,
        rank,
        kept_buckets,
        tier_group,
        out,
        progress,
    )
}

/// Writer core over any positional model reader (tests drive it with an
/// in-memory model).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_pack_with(
    reader: &dyn ShardRead,
    model_file_bytes: u64,
    shard_offsets: &[u64],
    rows_per_shard: usize,
    row_encoding: RowEncoding,
    row_width: usize,
    fingerprint_hex: &str,
    rank: &PleHotRank,
    kept_buckets: usize,
    tier_group: usize,
    out: &Path,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PleHotPackHeader, String> {
    let fail = |what: String| format!("ple hot pack: {what}");
    let row_bytes = checked_row_bytes(row_encoding, row_width).map_err(&fail)? as u64;
    if !is_hex64_lower(fingerprint_hex) {
        return Err(fail(format!(
            "fingerprint `{fingerprint_hex}` is not 64 lowercase hex digits"
        )));
    }
    if !(2..=BUCKETS).contains(&tier_group) {
        return Err(fail(format!(
            "tier group {tier_group} outside 2..={BUCKETS}"
        )));
    }
    if !(1..=BUCKETS).contains(&kept_buckets) {
        return Err(fail(format!(
            "kept buckets {kept_buckets} outside 1..={BUCKETS}"
        )));
    }
    let rows_per_shard_u64 = rows_per_shard as u64;
    let padded_rows = shard_offsets.len() as u64 * rows_per_shard_u64;
    if rows_per_shard == 0 || shard_offsets.is_empty() || rank.padded_rows() != padded_rows {
        return Err(fail(format!(
            "rank covers {} rows but {} shards of {rows_per_shard} rows cover {padded_rows}",
            rank.padded_rows(),
            shard_offsets.len()
        )));
    }
    let buckets = rank.buckets();
    if buckets.len() as u64 != padded_rows {
        return Err(fail(format!(
            "rank holds {} buckets for {padded_rows} rows",
            buckets.len()
        )));
    }
    let rank_sha = rank
        .header()
        .get("payload_sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::to_ascii_lowercase)
        .filter(|sha| is_hex64_lower(sha))
        .ok_or_else(|| fail("rank header lacks a 64-hex payload_sha256".to_string()))?;
    for (shard, &offset) in shard_offsets.iter().enumerate() {
        let end = rows_per_shard_u64
            .checked_mul(row_bytes)
            .and_then(|len| offset.checked_add(len));
        if end.is_none_or(|end| end > model_file_bytes) {
            return Err(fail(format!(
                "shard {shard} at {offset} is not inside the {model_file_bytes}-byte model file"
            )));
        }
    }

    // Tier of every bucket, rows per tier, tier byte bases.
    let tiers = tier_count(kept_buckets, tier_group);
    let mut tier_of = [NO_TIER; BUCKETS];
    for (bucket, slot) in tier_of.iter_mut().enumerate().take(kept_buckets) {
        *slot = (bucket / tier_group) as u8;
    }
    let mut tier_rows = vec![0u64; tiers];
    for &bucket in buckets {
        let tier = tier_of[usize::from(bucket)];
        if tier != NO_TIER {
            tier_rows[usize::from(tier)] += 1;
        }
    }
    let mut tier_base = Vec::with_capacity(tiers);
    let mut total_rows = 0u64;
    for &rows in &tier_rows {
        tier_base.push(total_rows * row_bytes);
        total_rows += rows;
    }
    let data_bytes = total_rows * row_bytes;

    let mut header = PleHotPackHeader {
        padded_rows,
        rows_per_shard: rows_per_shard_u64,
        row_encoding,
        row_width: row_width as u64,
        encoded_row_bytes: row_bytes,
        ple_fingerprint: fingerprint_hex.to_string(),
        rank_payload_sha256: rank_sha,
        tier_group,
        kept_buckets,
        tier_rows,
        data_offset: 0,
        data_bytes,
        shard_offsets: shard_offsets.to_vec(),
        model_file_bytes,
    };
    let header_bytes = encode_header(&mut header)?;

    let mut tmp_name = out.as_os_str().to_owned();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp_name);
    let result = write_tmp(
        reader,
        &tmp,
        &header,
        &header_bytes,
        &tier_of,
        &tier_base,
        buckets,
        progress,
    )
    .and_then(|()| {
        std::fs::rename(&tmp, out).map_err(|e| {
            fail(format!(
                "rename {} -> {}: {e}",
                tmp.display(),
                out.display()
            ))
        })
    });
    match result {
        Ok(()) => Ok(header),
        Err(error) => {
            let _ = std::fs::remove_file(&tmp);
            Err(error)
        }
    }
}

/// One tier's pending output.
struct TierOut {
    buf: Vec<u8>,
    /// Absolute file offset the next flush writes at.
    cursor: u64,
    /// Rows appended so far (flushed or buffered).
    rows: u64,
}

#[allow(clippy::too_many_arguments)]
fn write_tmp(
    reader: &dyn ShardRead,
    tmp: &Path,
    header: &PleHotPackHeader,
    header_bytes: &[u8],
    tier_of: &[u8; BUCKETS],
    tier_base: &[u64],
    buckets: &[u8],
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), String> {
    let tmp_err = |what: &str, e: io::Error| format!("ple hot pack: {what} {}: {e}", tmp.display());
    let output = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(tmp)
        .map_err(|e| tmp_err("create", e))?;
    output
        .set_len(header.data_offset + header.data_bytes)
        .map_err(|e| tmp_err("set_len", e))?;
    let mut preamble = Vec::with_capacity(16 + header_bytes.len());
    preamble.extend_from_slice(PACK_MAGIC);
    preamble.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    preamble.extend_from_slice(header_bytes);
    output
        .write_all_at(&preamble, 0)
        .map_err(|e| tmp_err("write header", e))?;

    let row_bytes = header.encoded_row_bytes as usize;
    let rows_per_shard = header.rows_per_shard;
    let mut tiers: Vec<TierOut> = header
        .tier_rows
        .iter()
        .zip(tier_base)
        .map(|(&rows, &base)| TierOut {
            buf: Vec::with_capacity(((rows * row_bytes as u64).min(FLUSH_BYTES as u64)) as usize),
            cursor: header.data_offset + base,
            rows: 0,
        })
        .collect();
    let flush = |tier: &mut TierOut| -> Result<(), String> {
        if !tier.buf.is_empty() {
            output
                .write_all_at(&tier.buf, tier.cursor)
                .map_err(|e| tmp_err("write rows to", e))?;
            tier.cursor += tier.buf.len() as u64;
            tier.buf.clear();
        }
        Ok(())
    };

    let mut window = AlignedBuf::new(READ_CHUNK);
    for (shard, &file_offset) in header.shard_offsets.iter().enumerate() {
        let first_global = shard as u64 * rows_per_shard;
        let shard_end = file_offset + rows_per_shard * row_bytes as u64;
        let mut next = 0u64;
        while next < rows_per_shard {
            // Aligned window starting at or before the first unread row.
            let start = (file_offset + next * row_bytes as u64) & !(READ_ALIGN - 1);
            let end = (start + READ_CHUNK as u64).min(shard_end.next_multiple_of(READ_ALIGN));
            let want = (end - start) as usize;
            let dst = &mut window.as_mut_slice()[..want];
            let mut filled = 0usize;
            while filled < want {
                let n = reader
                    .read_at(start + filled as u64, &mut dst[filled..])
                    .map_err(|e| {
                        format!("ple hot pack: read model at {}: {e}", start + filled as u64)
                    })?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            let available_end = start + filled as u64;
            let rows_end =
                (available_end.saturating_sub(file_offset) / row_bytes as u64).min(rows_per_shard);
            if rows_end <= next {
                return Err(format!(
                    "ple hot pack: unexpected end of model file at byte {available_end} \
                     (shard {shard} rows {next}..{rows_per_shard})"
                ));
            }
            for row in next..rows_end {
                let tier = tier_of[usize::from(buckets[(first_global + row) as usize])];
                if tier == NO_TIER {
                    continue;
                }
                let tier = &mut tiers[usize::from(tier)];
                if tier.buf.len() + row_bytes > FLUSH_BYTES {
                    flush(tier)?;
                }
                let at = (file_offset + row * row_bytes as u64 - start) as usize;
                tier.buf.extend_from_slice(&dst[at..at + row_bytes]);
                tier.rows += 1;
            }
            next = rows_end;
            progress(first_global + next, header.padded_rows);
        }
    }
    for (index, tier) in tiers.iter_mut().enumerate() {
        flush(tier)?;
        if tier.rows != header.tier_rows[index] {
            return Err(format!(
                "ple hot pack: tier {index} got {} rows, expected {}",
                tier.rows, header.tier_rows[index]
            ));
        }
        let expected_end = header.data_offset + tier_base[index] + tier.rows * row_bytes as u64;
        if tier.cursor != expected_end {
            return Err(format!(
                "ple hot pack: tier {index} cursor {} != {expected_end}",
                tier.cursor
            ));
        }
    }
    output.sync_all().map_err(|e| tmp_err("sync", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ple_hot_store::test_support::*;

    struct Fixture {
        model: Vec<u8>,
        offsets: Vec<u64>,
        rows_per_shard: usize,
        encoding: RowEncoding,
        row_width: usize,
        row_bytes: usize,
        buckets: Vec<u8>,
        rank: PleHotRank,
    }

    /// PLE-shaped fixture: Q8F16 rows of 160 values (170 B).
    fn fixture(shards: usize, rows_per_shard: usize, seed: u64) -> Fixture {
        fixture_with(
            synthetic_shards(shards, rows_per_shard),
            rows_per_shard,
            seed,
            RowEncoding::Q8F16,
            TEST_ROW_WIDTH,
        )
    }

    /// Fixture whose shards hold deterministic rows of any encoding/width.
    fn encoded_fixture(
        shards: usize,
        rows_per_shard: usize,
        seed: u64,
        encoding: RowEncoding,
        row_width: usize,
    ) -> Fixture {
        let row_bytes = checked_row_bytes(encoding, row_width).unwrap();
        let data = (0..shards)
            .map(|s| {
                (0..rows_per_shard * row_bytes)
                    .map(|i| {
                        let global = (s * rows_per_shard + i / row_bytes) as u64;
                        (global
                            .wrapping_mul(0x9E37_79B9)
                            .wrapping_add((i % row_bytes) as u64 * 31)
                            >> 3) as u8
                    })
                    .collect()
            })
            .collect();
        fixture_with(data, rows_per_shard, seed, encoding, row_width)
    }

    fn fixture_with(
        shard_data: Vec<Vec<u8>>,
        rows_per_shard: usize,
        seed: u64,
        encoding: RowEncoding,
        row_width: usize,
    ) -> Fixture {
        let shards = shard_data.len();
        let (model, offsets) = layout_shards(&shard_data);
        let rows = shards * rows_per_shard;
        let buckets = synthetic_buckets(rows, seed);
        let rank = PleHotRank::from_buckets(rows as u64, FINGERPRINT, buckets.clone());
        Fixture {
            model,
            offsets,
            rows_per_shard,
            encoding,
            row_width,
            row_bytes: checked_row_bytes(encoding, row_width).unwrap(),
            buckets,
            rank,
        }
    }

    fn pack(
        fx: &Fixture,
        kept: usize,
        group: usize,
        out: &Path,
    ) -> Result<PleHotPackHeader, String> {
        let mut last = (0u64, 0u64);
        let header = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.model.len() as u64,
            &fx.offsets,
            fx.rows_per_shard,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            kept,
            group,
            out,
            &mut |done, total| {
                assert!(done >= last.0 && done <= total);
                last = (done, total);
            },
        )?;
        assert_eq!(
            last,
            (header.padded_rows, header.padded_rows),
            "progress reaches 100%"
        );
        Ok(header)
    }

    /// Every kept row lands at its slot, byte for byte; the file length is exact.
    fn verify(fx: &Fixture, header: &PleHotPackHeader, out: &Path) {
        let bytes = std::fs::read(out).unwrap();
        assert_eq!(bytes.len() as u64, header.data_offset + header.data_bytes);
        assert_eq!(&bytes[..8], PACK_MAGIC);
        let kept = header.kept_buckets;
        let group = header.tier_group;
        let tiers = tier_count(kept, group);
        let mut base = vec![0u64; tiers];
        for t in 1..tiers {
            base[t] = base[t - 1] + header.tier_rows[t - 1] * fx.row_bytes as u64;
        }
        let mut seen = vec![0u64; tiers];
        let mut padding_zero = true;
        for &b in &bytes[16 + u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize
            ..header.data_offset as usize]
        {
            padding_zero &= b == 0;
        }
        assert!(padding_zero, "padding between header and data is zero");
        for (row, &bucket) in fx.buckets.iter().enumerate() {
            if usize::from(bucket) >= kept {
                continue;
            }
            let tier = usize::from(bucket) / group;
            let slot = seen[tier];
            seen[tier] += 1;
            let shard = row / fx.rows_per_shard;
            let local = row % fx.rows_per_shard;
            let model_at = fx.offsets[shard] as usize + local * fx.row_bytes;
            let pack_at =
                header.data_offset as usize + base[tier] as usize + slot as usize * fx.row_bytes;
            assert_eq!(
                &bytes[pack_at..pack_at + fx.row_bytes],
                &fx.model[model_at..model_at + fx.row_bytes],
                "row {row} (tier {tier}, slot {slot})"
            );
        }
        assert_eq!(seen, header.tier_rows, "tier row counts");
    }

    #[test]
    fn round_trip_group_5_and_2() {
        let fx = fixture(3, 4096, 7);
        assert_ne!(fx.offsets[0] % 4096, 0, "shard start is not sector aligned");
        let dir = tempfile::tempdir().unwrap();
        for (kept, group) in [(100usize, 5usize), (100, 2), (7, 5), (1, 2)] {
            let out = dir.path().join(format!("pack-{kept}-{group}.plehotrows"));
            let written = pack(&fx, kept, group, &out).unwrap();
            assert_eq!(written.tier_rows.len(), tier_count(kept, group));
            assert_eq!(written.data_offset % 4096, 0);
            assert_eq!(written.model_file_bytes, fx.model.len() as u64);
            assert_eq!(written.shard_offsets, fx.offsets);
            assert_eq!(written.rank_payload_sha256.len(), 64);
            let read = read_ple_hot_pack_header(&out).unwrap();
            assert_eq!(read, written, "header reads back equal");
            verify(&fx, &read, &out);
        }
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "tmp files left: {leftovers:?}");
    }

    #[test]
    fn keep_all_buckets_packs_every_row() {
        let fx = fixture(3, 4096, 11);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("all.plehotrows");
        let written = pack(&fx, 256, 5, &out).unwrap();
        assert_eq!(
            written.data_bytes,
            written.padded_rows * fx.row_bytes as u64
        );
        assert_eq!(written.tier_rows.iter().sum::<u64>(), written.padded_rows);
        assert_eq!(written.tier_rows.len(), 52);
        verify(&fx, &read_ple_hot_pack_header(&out).unwrap(), &out);
    }

    /// Shards larger than one 16 MiB read window, at unaligned file offsets.
    #[test]
    fn multi_window_unaligned_shards() {
        let fx = fixture(2, 100_000, 3);
        const { assert!(100_000 * 170 > READ_CHUNK) };
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("big.plehotrows");
        let written = pack(&fx, 64, 4, &out).unwrap();
        verify(&fx, &read_ple_hot_pack_header(&out).unwrap(), &out);
        assert!(written.data_bytes > 0);
    }

    #[test]
    fn writer_rejects_bad_inputs() {
        let fx = fixture(2, 1024, 5);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("x.plehotrows");
        assert!(pack(&fx, 0, 5, &out).is_err());
        assert!(pack(&fx, 257, 5, &out).is_err());
        assert!(pack(&fx, 10, 1, &out).is_err());
        assert!(pack(&fx, 10, 257, &out).is_err());
        let err = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.model.len() as u64,
            &fx.offsets,
            1000,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            10,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("cover"), "{err}");
        let err = write_pack_with(
            &SliceReader(fx.model.clone()),
            fx.offsets[1] + (1024 * fx.row_bytes) as u64 - 1,
            &fx.offsets,
            1024,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            10,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("not inside"), "{err}");
        assert!(!out.exists());
    }

    #[test]
    fn writer_cleans_tmp_on_read_failure() {
        let fx = fixture(2, 1024, 5);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("short.plehotrows");
        // The reader's file is shorter than the model_file_bytes claims.
        let mut truncated = fx.model.clone();
        truncated.truncate(fx.offsets[1] as usize + 1000);
        let err = write_pack_with(
            &SliceReader(truncated),
            fx.model.len() as u64,
            &fx.offsets,
            fx.rows_per_shard,
            fx.encoding,
            fx.row_width,
            FINGERPRINT,
            &fx.rank,
            100,
            5,
            &out,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.contains("unexpected end"), "{err}");
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "tmp removed"
        );
    }

    /// The pack layout is row-size agnostic: BF16 rows of small widths (16 B,
    /// 48 B, an odd 34 B) round-trip through the writer and the header reader,
    /// and the writer refuses a width its encoding cannot hold.
    #[test]
    fn non_q8_row_encodings_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        for (encoding, width, row_bytes) in [
            (RowEncoding::Bf16, 8usize, 16usize),
            (RowEncoding::Bf16, 24, 48),
            (RowEncoding::Bf16, 17, 34),
            (RowEncoding::Q8F16, 32, 34),
        ] {
            let fx = encoded_fixture(3, 2048, 13, encoding, width);
            assert_eq!(fx.row_bytes, row_bytes);
            assert_ne!(row_bytes, 170);
            let out = dir
                .path()
                .join(format!("{}-{width}.plehotrows", encoding.dtype_name()));
            let written = pack(&fx, 100, 5, &out).unwrap();
            assert_eq!(written.row_encoding, encoding);
            assert_eq!(written.row_width, width as u64);
            assert_eq!(written.encoded_row_bytes, row_bytes as u64);
            assert_eq!(
                written.data_bytes,
                written.tier_rows.iter().sum::<u64>() * row_bytes as u64
            );
            let read = read_ple_hot_pack_header(&out).unwrap();
            assert_eq!(read, written, "header reads back equal");
            verify(&fx, &read, &out);
        }

        let fx = encoded_fixture(2, 1024, 5, RowEncoding::Bf16, 8);
        let out = dir.path().join("bad-width.plehotrows");
        for (encoding, width) in [
            (RowEncoding::Q8F16, 33usize),
            (RowEncoding::Q8F16, 0),
            (RowEncoding::Bf16, 0),
        ] {
            let err = write_pack_with(
                &SliceReader(fx.model.clone()),
                fx.model.len() as u64,
                &fx.offsets,
                fx.rows_per_shard,
                encoding,
                width,
                FINGERPRINT,
                &fx.rank,
                10,
                5,
                &out,
                &mut |_, _| {},
            )
            .unwrap_err();
            assert!(err.contains("row width"), "{err}");
        }
        assert!(!out.exists());
    }

    /// Rewrite a good pack's header through `mutate`, recomputing a consistent
    /// `data_offset`, and keep the original data bytes.
    fn forge(good: &Path, bad: &Path, mutate: impl Fn(&mut serde_json::Value)) {
        let bytes = std::fs::read(good).unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let old_offset = round_up_data(16 + header_len as u64) as usize;
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        mutate(&mut json);
        let mut offset = 4096u64;
        let new_header = loop {
            json["data_offset"] = serde_json::json!(offset);
            let text = serde_json::to_vec(&json).unwrap();
            let needed = round_up_data(16 + text.len() as u64);
            if needed == offset {
                break text;
            }
            offset = needed;
        };
        let mut forged = Vec::new();
        forged.extend_from_slice(PACK_MAGIC);
        forged.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
        forged.extend_from_slice(&new_header);
        forged.resize(offset as usize, 0);
        forged.extend_from_slice(&bytes[old_offset..]);
        std::fs::write(bad, forged).unwrap();
    }

    #[test]
    fn reader_rejects_malformed_packs() {
        let fx = fixture(2, 1024, 9);
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.plehotrows");
        pack(&fx, 100, 5, &good).unwrap();
        read_ple_hot_pack_header(&good).unwrap();
        let bad = dir.path().join("bad.plehotrows");
        let expect = |what: &str| {
            let err = read_ple_hot_pack_header(&bad).unwrap_err();
            assert!(err.contains(what), "expected `{what}` in: {err}");
        };

        // forge round-trips unchanged
        forge(&good, &bad, |_| {});
        read_ple_hot_pack_header(&bad).unwrap();

        let mut bytes = std::fs::read(&good).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&bad, &bytes).unwrap();
        expect("bad magic");

        let bytes = std::fs::read(&good).unwrap();
        std::fs::write(&bad, &bytes[..bytes.len() - 1]).unwrap();
        expect("file is");
        std::fs::write(&bad, &bytes[..10]).unwrap();
        assert!(read_ple_hot_pack_header(&bad).is_err());
        std::fs::write(&bad, &bytes[..20]).unwrap();
        expect("header length");
        let mut extended = bytes.clone();
        extended.push(0);
        std::fs::write(&bad, &extended).unwrap();
        expect("file is");

        forge(&good, &bad, |j| {
            j["data_bytes"] = serde_json::json!(j["data_bytes"].as_u64().unwrap() + 170)
        });
        expect("data_bytes");
        forge(&good, &bad, |j| {
            j["tier_rows"].as_array_mut().unwrap().pop();
        });
        expect("tier_rows has");
        forge(&good, &bad, |j| j["version"] = serde_json::json!(1));
        expect("version 1");
        forge(&good, &bad, |j| {
            j["encoded_row_bytes"] = serde_json::json!(34)
        });
        expect("encoded_row_bytes");
        // Encoding and size must agree: 170 B rows are not BF16 rows of 160 values.
        forge(&good, &bad, |j| {
            j["row_encoding"] = serde_json::json!("BF16")
        });
        expect("encoded_row_bytes");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(96));
        expect("encoded_row_bytes");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(0));
        expect("row width");
        forge(&good, &bad, |j| j["row_width"] = serde_json::json!(161));
        expect("row width");
        forge(&good, &bad, |j| {
            j["row_encoding"] = serde_json::json!("Q3_K")
        });
        expect("row_encoding");
        forge(&good, &bad, |j| j["row_encoding"] = serde_json::json!(8));
        expect("row_encoding");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("row_encoding");
        });
        expect("row_encoding");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("row_width");
        });
        expect("row_width");
        forge(&good, &bad, |j| j["tier_group"] = serde_json::json!(1));
        expect("tier_group");
        forge(&good, &bad, |j| j["kept_buckets"] = serde_json::json!(0));
        expect("kept_buckets");
        forge(&good, &bad, |j| j["kept_buckets"] = serde_json::json!(257));
        expect("kept_buckets");
        forge(&good, &bad, |j| {
            j["ple_fingerprint"] = serde_json::json!("ABC")
        });
        expect("ple_fingerprint");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("generator");
        });
        expect("generator");
        forge(&good, &bad, |j| {
            j.as_object_mut().unwrap().remove("model_file_bytes");
        });
        expect("model_file_bytes");
        forge(&good, &bad, |j| {
            j["padded_rows"] = serde_json::json!("many")
        });
        expect("padded_rows");

        // data_offset that is not round_up(16 + header_len, 4096)
        let bytes = std::fs::read(&good).unwrap();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut json: serde_json::Value =
            serde_json::from_slice(&bytes[16..16 + header_len]).unwrap();
        json["data_offset"] = serde_json::json!(8192);
        let text = serde_json::to_vec(&json).unwrap();
        assert_eq!(text.len(), header_len, "same-width edit");
        let mut forged = bytes.clone();
        forged[16..16 + header_len].copy_from_slice(&text);
        std::fs::write(&bad, &forged).unwrap();
        expect("data_offset");
    }
}
