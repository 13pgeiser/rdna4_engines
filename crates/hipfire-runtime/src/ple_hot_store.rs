// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE hot-row store: the most frequently used rows of the external PLE row
//! table, read from a rank-ordered copy of them (the hot-row pack) through the
//! kernel page cache, so on unified-memory GPUs most rows a request needs come
//! from RAM without SSD reads, and the memory they use stays reclaimable.
//!
//! The Qwen4 PLE table is ~320M padded rows of 170 B (Q8F16) stored in shards
//! inside the model file.  A frequency rank of the rows (the `.plehot`
//! sidecar, one bucket byte per row, bucket 0 = hottest) selects which rows
//! are hot.  `hipfire-quantize ple-hot-pack` copies the rows of the hottest
//! buckets *verbatim from the model file* into `<model>.plehotrows`
//! ([`crate::ple_hot_pack`]), grouped in tiers of buckets with each tier's
//! rows in ascending row order, so the hottest rows share OS pages (24 rows
//! per 4 KiB page instead of 24 random-rank rows).  Rows are decoded by the
//! same `RowEncoding::decode_row` the model-file path uses, so the store never
//! changes the bytes a request returns, only where they come from.
//!
//! The pack is ordinary file data: page-cache pages of it are clean, so under
//! memory pressure the kernel drops them (and reads them again from the pack
//! when needed) instead of writing anything to swap.  After the model is
//! ready, a prewarm reads the pack's hottest prefix sequentially while no
//! request is being served: every PLE fetch holds `busy`, and the prewarm
//! resumes `BUSY_QUIET` after the last one.
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PleHotRank::open(path: &Path, expected_padded_rows: u64, expected_fingerprint_hex: &str)
//!     -> Result<PleHotRank, String>          // everything validated except the payload sha
//! PleHotRank::from_buckets(padded_rows: u64, fingerprint_hex: &str, buckets: Vec<u8>) -> PleHotRank
//! PleHotRank::{padded_rows() -> u64, buckets() -> &[u8], header() -> &serde_json::Value,
//!              verify() -> Result<(), String>}
//! write_ple_hot_rank(path: &Path, padded_rows: u64, fingerprint_hex: &str, buckets: &[u8],
//!                    rank_meta: serde_json::Value) -> Result<(), String>
//! HotRowStore::open(rank: PleHotRank, pack_path: &Path, rows_per_shard: usize,
//!                   encoding: RowEncoding, encoded_row_bytes: usize, shard_offsets: &[u64],
//!                   model_file_bytes: u64)
//!     -> Result<Arc<HotRowStore>, String>
//! HotRowStore::spawn_worker(self: &Arc<Self>, model_path: &Path, shard_offsets: Vec<u64>)
//!     -> JoinHandle<()>                      // verify, then prewarm once a budget is set
//! HotRowStore::{set_prewarm_budget(bytes: u64), stop(), is_serving() -> bool,
//!               verified() -> Option<bool>, pack_offset(row: u64) -> Option<u64>,
//!               read_pack(offset: u64, dst: &mut [u8]) -> Result<(), SourceError>,
//!               tier_count(), kept_buckets(), data_bytes(), prewarmed_bytes(),
//!               prewarm_complete(), hot_rows(), hot_pages(), hot_reads()}
//! mem_available_bytes() -> Option<u64>
//! ```
//!
//! # Sidecar format (`<model>.plehot`)
//!
//! ```text
//! 0..8    magic b"HFPLEHT1"
//! 8..16   u64 LE header_len
//! header  UTF-8 JSON {version:1, padded_rows, buckets:256, encoded_row_bytes:170,
//!                     ple_fingerprint, payload_sha256, rank:{..}}
//! payload padded_rows bytes, one bucket per global PLE row (0 = hottest)
//! ```

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::external_rows::RowEncoding;
use crate::model_source::SourceError;
use crate::ple_hot_pack::read_ple_hot_pack_header;

const MAGIC: &[u8; 8] = b"HFPLEHT1";
const VERSION: u64 = 1;
/// Frequency buckets in a sidecar.
pub const BUCKETS: usize = 256;
/// Encoded bytes of one Q8F16 PLE row (160 values = 5 blocks of 34 B): the
/// informational `encoded_row_bytes` a sidecar records and the row size of the
/// test fixtures.  Neither the sidecar reader nor the pack depends on it.
pub const ENCODED_ROW_BYTES: usize = 170;
/// Buckets per tier of a hot-row pack.
pub const DEFAULT_TIER_BUCKETS: usize = 5;
/// Rows per prefix-count block of the slot index.
const BLOCK_ROWS: usize = 512;
/// One aligned O_DIRECT read of the pack writer.
pub(crate) const READ_CHUNK: usize = 16 << 20;
pub(crate) const READ_ALIGN: u64 = 4096;
/// The prewarm resumes once no PLE fetch ran for this long: longer than the
/// compute between two prefill chunks' fetches (~4 s for an 8192-row chunk on
/// Halo), so the prewarm only reads while no request is being served.
const BUSY_QUIET: Duration = Duration::from_secs(5);
/// Bytes per prewarm read.
const PREWARM_CHUNK: usize = 8 << 20;
/// Kept rows spot-checked against the model file before the store serves.
const VERIFY_SAMPLES: usize = 4096;
/// Threads of the spot check (its reads are random 170-B reads).
const VERIFY_THREADS: usize = 16;
/// Sanity bound on a sidecar header.
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// `tier_of` value of a bucket that is not kept.
const NO_TIER: u8 = u8::MAX;
/// `prewarm_budget` before [`HotRowStore::set_prewarm_budget`].
const BUDGET_UNSET: u64 = u64::MAX;

const STATE_VERIFYING: u8 = 0;
const STATE_SERVING: u8 = 1;
const STATE_FAILED: u8 = 2;

const TAG: &str = "qwen4 ple hot store";

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 15)] as char);
    }
    out
}

fn is_hex64(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// ---------------------------------------------------------------------------
// Sidecar: reader + writer
// ---------------------------------------------------------------------------

/// A validated `.plehot` rank: one bucket byte per padded PLE row.
pub struct PleHotRank {
    header: serde_json::Value,
    padded_rows: u64,
    payload_sha256: String,
    buckets: Vec<u8>,
}

impl PleHotRank {
    /// Read a sidecar and validate magic, version, `padded_rows`, bucket
    /// count, row size, fingerprint, and payload length.  The payload sha256
    /// is NOT checked here (hashing ~320 MB belongs on the fill thread): call
    /// [`PleHotRank::verify`], which `HotRowStore::spawn_fill` does itself.
    pub fn open(
        path: &Path,
        expected_padded_rows: u64,
        expected_fingerprint_hex: &str,
    ) -> Result<PleHotRank, String> {
        let describe = |what: &str, error: &dyn std::fmt::Display| {
            format!("ple hot rank {}: {what}: {error}", path.display())
        };
        let mut file = File::open(path).map_err(|e| describe("open", &e))?;
        let file_len = file.metadata().map_err(|e| describe("stat", &e))?.len();
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix)
            .map_err(|e| describe("read preamble", &e))?;
        if &prefix[..8] != MAGIC {
            return Err(format!("ple hot rank {}: bad magic", path.display()));
        }
        let header_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(format!(
                "ple hot rank {}: implausible header length {header_len}",
                path.display()
            ));
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| describe("read header", &e))?;
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).map_err(|e| describe("parse header", &e))?;

        let field_u64 = |name: &str| {
            header
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("ple hot rank {}: header lacks `{name}`", path.display()))
        };
        let version = field_u64("version")?;
        if version != VERSION {
            return Err(format!(
                "ple hot rank {}: version {version}, expected {VERSION}",
                path.display()
            ));
        }
        let padded_rows = field_u64("padded_rows")?;
        if padded_rows != expected_padded_rows {
            return Err(format!(
                "ple hot rank {}: padded_rows {padded_rows} != model's {expected_padded_rows}",
                path.display()
            ));
        }
        let buckets = field_u64("buckets")?;
        if buckets != BUCKETS as u64 {
            return Err(format!(
                "ple hot rank {}: {buckets} buckets, expected {BUCKETS}",
                path.display()
            ));
        }
        let row_bytes = field_u64("encoded_row_bytes")?;
        if row_bytes == 0 {
            return Err(format!(
                "ple hot rank {}: encoded_row_bytes is zero",
                path.display()
            ));
        }
        let fingerprint = header
            .get("ple_fingerprint")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("ple hot rank {}: header lacks fingerprint", path.display()))?;
        if !fingerprint.eq_ignore_ascii_case(expected_fingerprint_hex) {
            return Err(format!(
                "ple hot rank {}: PLE fingerprint {fingerprint} does not match the model's \
                 {expected_fingerprint_hex}",
                path.display()
            ));
        }
        let payload_sha256 = header
            .get("payload_sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|sha| is_hex64(sha))
            .ok_or_else(|| {
                format!(
                    "ple hot rank {}: header lacks a 64-hex payload_sha256",
                    path.display()
                )
            })?
            .to_ascii_lowercase();
        let expected_len = 16u64
            .checked_add(header_len)
            .and_then(|n| n.checked_add(padded_rows))
            .ok_or_else(|| format!("ple hot rank {}: length overflow", path.display()))?;
        if file_len != expected_len {
            return Err(format!(
                "ple hot rank {}: file is {file_len} bytes, expected {expected_len}",
                path.display()
            ));
        }
        let rows = usize::try_from(padded_rows)
            .map_err(|_| format!("ple hot rank {}: padded_rows overflow", path.display()))?;
        let mut payload = vec![0u8; rows];
        file.read_exact(&mut payload)
            .map_err(|e| describe("read payload", &e))?;
        Ok(PleHotRank {
            header,
            padded_rows,
            payload_sha256,
            buckets: payload,
        })
    }

    /// An in-memory rank (tests, tools).  Its payload hash is computed here,
    /// so [`PleHotRank::verify`] passes.
    pub fn from_buckets(padded_rows: u64, fingerprint_hex: &str, buckets: Vec<u8>) -> PleHotRank {
        assert_eq!(buckets.len() as u64, padded_rows, "one bucket per row");
        let payload_sha256 = hex(&Sha256::digest(&buckets));
        let header = serde_json::json!({
            "version": VERSION,
            "padded_rows": padded_rows,
            "buckets": BUCKETS,
            "encoded_row_bytes": ENCODED_ROW_BYTES,
            "ple_fingerprint": fingerprint_hex,
            "payload_sha256": payload_sha256,
        });
        PleHotRank {
            header,
            padded_rows,
            payload_sha256,
            buckets,
        }
    }

    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    /// One bucket per padded row (0 = hottest).
    pub fn buckets(&self) -> &[u8] {
        &self.buckets
    }

    /// The parsed sidecar header.
    pub fn header(&self) -> &serde_json::Value {
        &self.header
    }

    /// Check the payload against the header's `payload_sha256`.
    pub fn verify(&self) -> Result<(), String> {
        let actual = hex(&Sha256::digest(&self.buckets));
        if actual == self.payload_sha256 {
            Ok(())
        } else {
            Err(format!(
                "ple hot rank payload sha256 {actual} != header {}",
                self.payload_sha256
            ))
        }
    }
}

/// Write a `.plehot` sidecar atomically (tmp file + rename).  `buckets` holds
/// one byte per padded row; `rank_meta` is stored verbatim as the header's
/// `rank` object.
pub fn write_ple_hot_rank(
    path: &Path,
    padded_rows: u64,
    fingerprint_hex: &str,
    buckets: &[u8],
    rank_meta: serde_json::Value,
) -> Result<(), String> {
    if buckets.len() as u64 != padded_rows {
        return Err(format!(
            "ple hot rank: {} buckets for {padded_rows} padded rows",
            buckets.len()
        ));
    }
    if !is_hex64(fingerprint_hex) {
        return Err(format!(
            "ple hot rank: fingerprint `{fingerprint_hex}` is not 64 hex digits"
        ));
    }
    let header = serde_json::json!({
        "version": VERSION,
        "padded_rows": padded_rows,
        "buckets": BUCKETS,
        "encoded_row_bytes": ENCODED_ROW_BYTES,
        "ple_fingerprint": fingerprint_hex.to_ascii_lowercase(),
        "payload_sha256": hex(&Sha256::digest(buckets)),
        "rank": rank_meta,
    });
    let header_bytes =
        serde_json::to_vec(&header).map_err(|e| format!("ple hot rank header: {e}"))?;
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = PathBuf::from(tmp_name);
    let write = || -> io::Result<()> {
        let mut file = File::create(&tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&header_bytes)?;
        file.write_all(buckets)?;
        file.flush()?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("ple hot rank {}: write: {e}", path.display())
    })
}

// ---------------------------------------------------------------------------
// O_DIRECT model reads (the pack writer)
// ---------------------------------------------------------------------------

/// A sector-aligned read buffer for O_DIRECT.
pub(crate) struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

impl AlignedBuf {
    pub(crate) fn new(len: usize) -> AlignedBuf {
        let layout = Layout::from_size_align(len, READ_ALIGN as usize).expect("read buffer layout");
        // SAFETY: non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        AlignedBuf {
            ptr: NonNull::new(ptr).expect("read buffer allocation"),
            len,
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: owned, initialised (zeroed) allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe {
            dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.len, READ_ALIGN as usize),
            )
        }
    }
}

// SAFETY: exclusively owned bytes.
unsafe impl Send for AlignedBuf {}

/// Positional reader of the model file (the file's own offsets).
pub(crate) trait ShardRead: Sync {
    /// Read up to `dst.len()` bytes at absolute `offset`; 0 at EOF.
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize>;
}

/// Open `path` for O_DIRECT reads, falling back to buffered reads on a
/// filesystem that refuses the flag.
pub(crate) fn open_direct(path: &Path) -> io::Result<File> {
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {
            eprintln!(
                "{TAG}: O_DIRECT unsupported on {}; using buffered reads",
                path.display()
            );
            File::open(path)
        }
        other => other,
    }
}

/// `/proc/meminfo` `MemAvailable` in bytes (page cache counts as available,
/// so a prewarmed pack does not lower it).
pub fn mem_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: u64 = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kib * 1024)
}

/// Read exactly `dst.len()` bytes at `offset`, retrying interrupted and short
/// reads; a read that hits end of file is a `ShortRead`.
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
    let mut done = 0;
    while done < dst.len() {
        let at = offset + done as u64;
        match file.read_at(&mut dst[done..], at) {
            Ok(0) => {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: done,
                })
            }
            Ok(n) => done += n,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(SourceError::Io { offset: at, source }),
        }
    }
    Ok(())
}

fn fadvise(file: &File, advice: libc::c_int) {
    // SAFETY: a valid open descriptor; advice only.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, advice) };
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// The hot rows of the PLE table: where each kept row sits in the hot-row
/// pack, and the open pack.
///
/// A row is kept when its sidecar bucket is below the pack's `kept_buckets`;
/// its tier is `bucket / tier_group`, and its pack offset is the tier's base
/// plus its slot (the tier's rows with a smaller global row index) times the
/// row size.  Nothing is served until the worker thread has checked the
/// sidecar payload hash and spot-checked [`VERIFY_SAMPLES`] kept rows against
/// the model file.
pub struct HotRowStore {
    padded_rows: u64,
    rows_per_shard: usize,
    encoding: RowEncoding,
    encoded_row_bytes: usize,
    /// Buckets per tier.
    group: usize,
    /// Kept buckets `K`: buckets `0..K` are in the pack.
    kept: usize,
    tier_count: usize,
    /// Bucket -> tier (`NO_TIER` when not kept).
    tier_of: [u8; BUCKETS],
    /// The sidecar payload.
    bucket: Box<[u8]>,
    /// `prefix[t][block]` = rows of tier `t` in rows `[0, block * 512)`.
    prefix: Vec<Box<[u32]>>,
    /// Absolute pack offset of each tier's first row.
    tier_base: Vec<u64>,
    /// The pack, opened for random reads (no readahead).
    pack: File,
    pack_path: PathBuf,
    data_offset: u64,
    data_bytes: u64,
    expected_sha: String,
    state: AtomicU8,
    stop: AtomicBool,
    hot_rows: AtomicU64,
    hot_pages: AtomicU64,
    hot_reads: AtomicU64,
    prewarm_budget: AtomicU64,
    prewarmed: AtomicU64,
    prewarm_done: AtomicBool,
    /// Time base of `busy_at_ms`.
    created: Instant,
    /// Milliseconds after `created` (plus one; 0 = never) at which the last
    /// PLE fetch finished (see [`HotRowStore::busy`]).
    busy_at_ms: AtomicU64,
    /// PLE fetches in flight.
    busy_active: AtomicUsize,
}

/// A PLE fetch in flight (from [`HotRowStore::busy`]); the prewarm stays
/// paused until [`BUSY_QUIET`] after the last guard drops.
pub(crate) struct BusyGuard<'a>(&'a HotRowStore);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        let store = self.0;
        let at = store.created.elapsed().as_millis() as u64 + 1;
        store.busy_at_ms.fetch_max(at, Ordering::AcqRel);
        store.busy_active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl HotRowStore {
    /// Open the hot-row pack at `pack_path` for `rank` and build the row ->
    /// pack-offset index.  Errors when the pack was not cut from this model
    /// (PLE fingerprint, shard offsets, model file size), from this rank
    /// (sidecar payload hash, per-tier row counts), or for this geometry.
    /// Serves nothing until [`HotRowStore::spawn_worker`] has verified it.
    pub fn open(
        rank: PleHotRank,
        pack_path: &Path,
        rows_per_shard: usize,
        encoding: RowEncoding,
        encoded_row_bytes: usize,
        shard_offsets: &[u64],
        model_file_bytes: u64,
    ) -> Result<std::sync::Arc<HotRowStore>, String> {
        let fail = |what: String| format!("ple hot pack {}: {what}", pack_path.display());
        let header = read_ple_hot_pack_header(pack_path)?;
        let PleHotRank {
            header: rank_header,
            padded_rows,
            payload_sha256,
            buckets,
        } = rank;
        if encoded_row_bytes == 0
            || header.row_encoding != encoding
            || header.encoded_row_bytes != encoded_row_bytes as u64
        {
            return Err(fail(format!(
                "rows are {} B {} in the pack and {encoded_row_bytes} B {} in the row store",
                header.encoded_row_bytes,
                header.row_encoding.dtype_name(),
                encoding.dtype_name()
            )));
        }
        if rows_per_shard == 0
            || header.rows_per_shard != rows_per_shard as u64
            || header.padded_rows != padded_rows
            || padded_rows != shard_offsets.len() as u64 * rows_per_shard as u64
        {
            return Err(fail(format!(
                "geometry {} rows in shards of {}, model {padded_rows} rows in {} shards of \
                 {rows_per_shard}",
                header.padded_rows,
                header.rows_per_shard,
                shard_offsets.len()
            )));
        }
        if padded_rows > u64::from(u32::MAX) {
            return Err(fail(format!("{padded_rows} rows exceed the index range")));
        }
        let rank_fingerprint = rank_header
            .get("ple_fingerprint")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !header
            .ple_fingerprint
            .eq_ignore_ascii_case(rank_fingerprint)
        {
            return Err(fail(format!(
                "PLE fingerprint {} is not the model's {rank_fingerprint}",
                header.ple_fingerprint
            )));
        }
        if header.rank_payload_sha256 != payload_sha256 {
            return Err(fail(format!(
                "cut from rank {} but the sidecar is {payload_sha256}",
                header.rank_payload_sha256
            )));
        }
        if header.shard_offsets != shard_offsets {
            return Err(fail(
                "PLE shard offsets differ from the model's".to_string(),
            ));
        }
        if header.model_file_bytes != model_file_bytes {
            return Err(fail(format!(
                "cut from a {} B model file, this one is {model_file_bytes} B",
                header.model_file_bytes
            )));
        }

        let group = header.tier_group;
        let kept = header.kept_buckets;
        let tier_count = kept.div_ceil(group);
        let mut tier_of = [NO_TIER; BUCKETS];
        for (bucket, slot) in tier_of.iter_mut().enumerate().take(kept) {
            *slot = (bucket / group) as u8;
        }
        let blocks = (padded_rows as usize).div_ceil(BLOCK_ROWS);
        let mut prefix: Vec<Vec<u32>> = (0..tier_count)
            .map(|_| Vec::with_capacity(blocks))
            .collect();
        let mut counts = [0u32; BUCKETS];
        for block in buckets.chunks(BLOCK_ROWS) {
            for (tier, column) in prefix.iter_mut().enumerate() {
                column.push(counts[tier]);
            }
            for &b in block {
                counts[usize::from(tier_of[usize::from(b)])] += 1;
            }
        }
        let tier_rows: Vec<u64> = counts[..tier_count].iter().map(|&n| u64::from(n)).collect();
        if tier_rows != header.tier_rows {
            return Err(fail(
                "per-tier row counts differ from the sidecar's".to_string(),
            ));
        }
        let mut tier_base = Vec::with_capacity(tier_count);
        let mut base = header.data_offset;
        for &rows in &tier_rows {
            tier_base.push(base);
            base += rows * encoded_row_bytes as u64;
        }
        let pack = File::open(pack_path).map_err(|e| fail(format!("open: {e}")))?;
        fadvise(&pack, libc::POSIX_FADV_RANDOM);
        let index_bytes =
            buckets.len() as u64 + prefix.iter().map(|c| c.len() as u64 * 4).sum::<u64>();
        eprintln!(
            "{TAG}: pack {}: kept {kept}/{BUCKETS} buckets in {tier_count} tiers ({:.2} GiB, index \
             {:.0} MiB)",
            pack_path.display(),
            gib(header.data_bytes),
            mib(index_bytes),
        );
        Ok(std::sync::Arc::new(HotRowStore {
            padded_rows,
            rows_per_shard,
            encoding,
            encoded_row_bytes,
            group,
            kept,
            tier_count,
            tier_of,
            bucket: buckets.into_boxed_slice(),
            prefix: prefix.into_iter().map(Vec::into_boxed_slice).collect(),
            tier_base,
            pack,
            pack_path: pack_path.to_path_buf(),
            data_offset: header.data_offset,
            data_bytes: header.data_bytes,
            expected_sha: payload_sha256,
            state: AtomicU8::new(STATE_VERIFYING),
            stop: AtomicBool::new(false),
            hot_rows: AtomicU64::new(0),
            hot_pages: AtomicU64::new(0),
            hot_reads: AtomicU64::new(0),
            prewarm_budget: AtomicU64::new(BUDGET_UNSET),
            prewarmed: AtomicU64::new(0),
            prewarm_done: AtomicBool::new(false),
            created: Instant::now(),
            busy_at_ms: AtomicU64::new(0),
            busy_active: AtomicUsize::new(0),
        }))
    }

    /// Tiers in the pack.
    pub fn tier_count(&self) -> usize {
        self.tier_count
    }

    /// Buckets (of 256) in the pack.
    pub fn kept_buckets(&self) -> usize {
        self.kept
    }

    pub fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    pub fn rows_per_shard(&self) -> usize {
        self.rows_per_shard
    }

    pub fn encoded_row_bytes(&self) -> usize {
        self.encoded_row_bytes
    }

    /// Row encoding of the pack (equal to the row store's).
    pub fn encoding(&self) -> RowEncoding {
        self.encoding
    }

    /// Row bytes in the pack.
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// The pack passed verification and serves rows.
    pub fn is_serving(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_SERVING
    }

    /// `None` while the worker is still verifying, then whether it passed.
    pub fn verified(&self) -> Option<bool> {
        match self.state.load(Ordering::Acquire) {
            STATE_SERVING => Some(true),
            STATE_FAILED => Some(false),
            _ => None,
        }
    }

    /// Bytes of the pack read by the prewarm so far.
    pub fn prewarmed_bytes(&self) -> u64 {
        self.prewarmed.load(Ordering::Relaxed)
    }

    /// The prewarm read its whole budget.
    pub fn prewarm_complete(&self) -> bool {
        self.prewarm_done.load(Ordering::Acquire)
    }

    /// Rows served from the pack.
    pub fn hot_rows(&self) -> u64 {
        self.hot_rows.load(Ordering::Relaxed)
    }

    /// Pages whose every requested row the pack served.
    pub fn hot_pages(&self) -> u64 {
        self.hot_pages.load(Ordering::Relaxed)
    }

    /// Reads issued against the pack (a read covers a run of consecutive
    /// pack rows).
    pub fn hot_reads(&self) -> u64 {
        self.hot_reads.load(Ordering::Relaxed)
    }

    pub(crate) fn note_served(&self, rows: u64, pages: u64, reads: u64) {
        self.hot_rows.fetch_add(rows, Ordering::Relaxed);
        self.hot_pages.fetch_add(pages, Ordering::Relaxed);
        self.hot_reads.fetch_add(reads, Ordering::Relaxed);
    }

    /// The row store is serving a PLE fetch (a prefill chunk or a decode
    /// step).  The prewarm pauses while the returned guard lives and for
    /// [`BUSY_QUIET`] after the last one drops, so it never competes with a
    /// request for the SSD.
    pub(crate) fn busy(&self) -> BusyGuard<'_> {
        self.busy_active.fetch_add(1, Ordering::AcqRel);
        BusyGuard(self)
    }

    /// Sleep while a PLE fetch is in flight or finished less than
    /// [`BUSY_QUIET`] ago; `false` when the store was stopped meanwhile.
    fn wait_until_idle(&self) -> bool {
        loop {
            if self.stopped() {
                return false;
            }
            if self.busy_active.load(Ordering::Acquire) > 0 {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            let last = self.busy_at_ms.load(Ordering::Acquire);
            if last == 0 {
                return true;
            }
            let since = Duration::from_millis(
                (self.created.elapsed().as_millis() as u64 + 1).saturating_sub(last),
            );
            if since >= BUSY_QUIET {
                return true;
            }
            std::thread::sleep((BUSY_QUIET - since).min(Duration::from_millis(20)));
        }
    }

    /// Ask the worker to stop at its next read; pair with joining the handle
    /// [`HotRowStore::spawn_worker`] returned.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// Absolute pack offset of global `row`'s encoded bytes when the pack
    /// holds it and serves.
    pub fn pack_offset(&self, row: u64) -> Option<u64> {
        if row >= self.padded_rows || !self.is_serving() {
            return None;
        }
        let tier = self.tier_of[usize::from(self.bucket[row as usize])];
        if tier == NO_TIER {
            return None;
        }
        let tier = usize::from(tier);
        Some(self.tier_base[tier] + self.slot(row, tier) * self.encoded_row_bytes as u64)
    }

    /// Read `dst.len()` bytes of the pack at absolute `offset`.
    pub fn read_pack(&self, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or(SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
        if offset < self.data_offset || end > self.data_offset + self.data_bytes {
            return Err(SourceError::RangeOutOfBounds {
                offset,
                length: dst.len() as u64,
                source_len: self.data_offset + self.data_bytes,
            });
        }
        read_exact_at(&self.pack, offset, dst)
    }

    /// Index of `row` inside tier `tier` (rows of the tier in ascending global
    /// row order): the prefix count of its 512-row block plus the tier's rows
    /// earlier in the block.
    pub(crate) fn slot(&self, row: u64, tier: usize) -> u64 {
        let row = row as usize;
        let block = row / BLOCK_ROWS;
        let lo = (tier * self.group) as u16;
        let hi = ((tier + 1) * self.group).min(self.kept) as u16;
        let before = self.bucket[block * BLOCK_ROWS..row]
            .iter()
            .filter(|&&b| (lo..hi).contains(&u16::from(b)))
            .count();
        u64::from(self.prefix[tier][block]) + before as u64
    }

    /// Bytes of the pack to prewarm once the model is ready (capped at the
    /// pack's size; 0 = none).  The worker starts reading at once if it has
    /// verified the pack, else right after.
    pub fn set_prewarm_budget(&self, bytes: u64) {
        self.prewarm_budget
            .store(bytes.min(self.data_bytes), Ordering::Release);
    }

    /// Start the worker: check the sidecar payload hash and spot-check
    /// [`VERIFY_SAMPLES`] kept rows of the pack against the model file
    /// (`shard_offsets[s]` = file offset of PLE shard `s`), then serve; once
    /// [`HotRowStore::set_prewarm_budget`] was called, read that many bytes of
    /// the pack's hottest tiers sequentially into the page cache while no
    /// request is being served.  Call [`HotRowStore::stop`] then join it
    /// before unloading the row store.
    pub fn spawn_worker(
        self: &std::sync::Arc<Self>,
        model_path: &Path,
        shard_offsets: Vec<u64>,
    ) -> JoinHandle<()> {
        let store = self.clone();
        let model_path = model_path.to_path_buf();
        std::thread::Builder::new()
            .name("ple-hot-pack".to_string())
            .spawn(move || store.run_worker(&model_path, &shard_offsets))
            .expect("spawn ple hot pack thread")
    }

    fn run_worker(&self, model_path: &Path, shard_offsets: &[u64]) {
        match self.verify(model_path, shard_offsets) {
            Ok(summary) => {
                eprintln!("{TAG}: {summary}; serving");
                self.state.store(STATE_SERVING, Ordering::Release);
            }
            Err(error) => {
                eprintln!("{TAG}: {error}; the pack is not used");
                self.state.store(STATE_FAILED, Ordering::Release);
                return;
            }
        }
        let budget = loop {
            if self.stopped() {
                return;
            }
            match self.prewarm_budget.load(Ordering::Acquire) {
                BUDGET_UNSET => std::thread::sleep(Duration::from_millis(50)),
                budget => break budget,
            }
        };
        self.prewarm(budget);
    }

    /// Sidecar payload hash, then [`VERIFY_SAMPLES`] kept rows compared byte
    /// for byte between the pack and the model file.
    fn verify(&self, model_path: &Path, shard_offsets: &[u64]) -> Result<String, String> {
        let started = Instant::now();
        let actual = hex(&Sha256::digest(&self.bucket));
        if actual != self.expected_sha {
            return Err(format!(
                "sidecar payload sha256 {actual} != {}",
                self.expected_sha
            ));
        }
        let model = File::open(model_path).map_err(|e| {
            format!(
                "opening {} for the spot check failed: {e}",
                model_path.display()
            )
        })?;
        fadvise(&model, libc::POSIX_FADV_RANDOM);
        // Deterministic sample: pseudo-random rows, kept ones only.
        let mut rows = Vec::with_capacity(VERIFY_SAMPLES);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..VERIFY_SAMPLES * 64 {
            if rows.len() == VERIFY_SAMPLES {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let row = state % self.padded_rows;
            if self.tier_of[usize::from(self.bucket[row as usize])] != NO_TIER {
                rows.push(row);
            }
        }
        let row_bytes = self.encoded_row_bytes;
        let per_thread = rows.len().div_ceil(VERIFY_THREADS).max(1);
        let mismatch = std::thread::scope(|scope| {
            let handles: Vec<_> = rows
                .chunks(per_thread)
                .map(|chunk| {
                    let model = &model;
                    scope.spawn(move || -> Result<(), String> {
                        let mut from_pack = vec![0u8; row_bytes];
                        let mut from_model = vec![0u8; row_bytes];
                        for &row in chunk {
                            let tier =
                                usize::from(self.tier_of[usize::from(self.bucket[row as usize])]);
                            let offset =
                                self.tier_base[tier] + self.slot(row, tier) * row_bytes as u64;
                            read_exact_at(&self.pack, offset, &mut from_pack)
                                .map_err(|e| format!("pack read of row {row}: {e}"))?;
                            let shard = (row / self.rows_per_shard as u64) as usize;
                            let local = row % self.rows_per_shard as u64;
                            read_exact_at(
                                model,
                                shard_offsets[shard] + local * row_bytes as u64,
                                &mut from_model,
                            )
                            .map_err(|e| format!("model read of row {row}: {e}"))?;
                            if from_pack != from_model {
                                return Err(format!(
                                    "row {row} differs between the pack and the model"
                                ));
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("ple hot pack verify thread panicked"))
                .collect::<Result<Vec<()>, String>>()
        });
        mismatch?;
        Ok(format!(
            "sidecar sha256 ok, {} pack rows match the model ({:.0} ms)",
            rows.len(),
            started.elapsed().as_secs_f64() * 1e3
        ))
    }

    /// Read the first `budget` bytes of the pack's rows (the hottest tiers)
    /// sequentially, so the page cache holds them, pausing while requests run.
    fn prewarm(&self, budget: u64) {
        if budget == 0 {
            self.prewarm_done.store(true, Ordering::Release);
            return;
        }
        let file = match File::open(&self.pack_path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("{TAG}: prewarm: opening the pack failed: {error}");
                return;
            }
        };
        fadvise(&file, libc::POSIX_FADV_SEQUENTIAL);
        let started = Instant::now();
        let mut paused = Duration::ZERO;
        let mut buffer = vec![0u8; PREWARM_CHUNK];
        let end = self.data_offset + budget;
        let mut offset = self.data_offset;
        while offset < end {
            let idle_from = Instant::now();
            if !self.wait_until_idle() {
                return;
            }
            paused += idle_from.elapsed();
            let len = ((end - offset) as usize).min(PREWARM_CHUNK);
            if let Err(error) = read_exact_at(&file, offset, &mut buffer[..len]) {
                eprintln!(
                    "{TAG}: prewarm stopped at {:.2} GiB: {error}",
                    gib(offset - self.data_offset)
                );
                return;
            }
            offset += len as u64;
            self.prewarmed
                .store(offset - self.data_offset, Ordering::Relaxed);
        }
        self.prewarm_done.store(true, Ordering::Release);
        let reading = started.elapsed().saturating_sub(paused).as_secs_f64();
        eprintln!(
            "{TAG}: prewarm complete: {:.2} GiB of {:.2} GiB in {:.1} s ({:.2} GB/s reading, {:.1} s \
             paused for requests), MemAvailable {:.1} GiB",
            gib(budget),
            gib(self.data_bytes),
            started.elapsed().as_secs_f64(),
            budget as f64 / 1e9 / reading.max(1e-9),
            paused.as_secs_f64(),
            mem_available_bytes().map_or(f64::NAN, gib),
        );
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::Arc;

    /// A model "file" in memory.
    pub(crate) struct SliceReader(pub Vec<u8>);

    impl ShardRead for SliceReader {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
            let begin = usize::try_from(offset).unwrap().min(self.0.len());
            let n = dst.len().min(self.0.len() - begin);
            dst[..n].copy_from_slice(&self.0[begin..begin + n]);
            Ok(n)
        }
    }

    /// Concatenate shards behind a 13-byte prefix and 7 bytes of padding
    /// between shards, so shard ranges are not sector aligned.
    pub(crate) fn layout_shards(shards: &[Vec<u8>]) -> (Vec<u8>, Vec<u64>) {
        let mut file = vec![0xEEu8; 13];
        let mut offsets = Vec::new();
        for shard in shards {
            offsets.push(file.len() as u64);
            file.extend_from_slice(shard);
            file.extend_from_slice(&[0xEE; 7]);
        }
        (file, offsets)
    }

    /// Deterministic encoded rows: every byte derives from the global row.
    pub(crate) fn synthetic_shards(shards: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        (0..shards)
            .map(|s| {
                let mut bytes = Vec::with_capacity(rows_per_shard * ENCODED_ROW_BYTES);
                for r in 0..rows_per_shard {
                    let global = (s * rows_per_shard + r) as u64;
                    for i in 0..ENCODED_ROW_BYTES as u64 {
                        bytes.push(
                            (global.wrapping_mul(0x9E37_79B9).wrapping_add(i * 31) >> 3) as u8,
                        );
                    }
                }
                bytes
            })
            .collect()
    }

    /// Buckets of a pseudo-random ranking: `bucket = rank * 256 / rows`.
    pub(crate) fn synthetic_buckets(rows: usize, seed: u64) -> Vec<u8> {
        let mut order: Vec<u32> = (0..rows as u32).collect();
        let mut state = seed | 1;
        for i in (1..rows).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            order.swap(i, (state % (i as u64 + 1)) as usize);
        }
        let mut buckets = vec![0u8; rows];
        for (rank, &row) in order.iter().enumerate() {
            buckets[row as usize] = (rank as u64 * BUCKETS as u64 / rows as u64) as u8;
        }
        buckets
    }

    /// Values per row of the test fixtures' Q8F16 rows ([`ENCODED_ROW_BYTES`] bytes).
    pub(crate) const TEST_ROW_WIDTH: usize = 160;

    pub(crate) const FINGERPRINT: &str =
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    /// A model file and its hot-row pack on disk, and the opened store.
    pub(crate) struct PackFixture {
        pub dir: tempfile::TempDir,
        pub model_path: PathBuf,
        pub pack_path: PathBuf,
        pub offsets: Vec<u64>,
        pub buckets: Vec<u8>,
        pub store: Arc<HotRowStore>,
        pub worker: Option<JoinHandle<()>>,
    }

    impl PackFixture {
        /// Write `shards` as a model file, pack its rows of buckets
        /// `0..kept` (tiers of `group`) under a pseudo-random rank, open the
        /// store and wait until its worker verified the pack.
        pub(crate) fn new(
            shards: &[Vec<u8>],
            rows_per_shard: usize,
            kept: usize,
            group: usize,
            seed: u64,
        ) -> Self {
            Self::with_encoding(
                shards,
                rows_per_shard,
                kept,
                group,
                seed,
                RowEncoding::Q8F16,
                TEST_ROW_WIDTH,
            )
        }

        /// [`PackFixture::new`] for shards of `encoding` rows of `row_width`
        /// values (the shard bytes are opaque to the pack).
        pub(crate) fn with_encoding(
            shards: &[Vec<u8>],
            rows_per_shard: usize,
            kept: usize,
            group: usize,
            seed: u64,
            encoding: RowEncoding,
            row_width: usize,
        ) -> Self {
            let row_bytes = crate::ple_hot_pack::checked_row_bytes(encoding, row_width).unwrap();
            let dir = tempfile::tempdir().unwrap();
            let (file, offsets) = layout_shards(shards);
            let model_path = dir.path().join("model.mq4");
            std::fs::write(&model_path, &file).unwrap();
            let rows = shards.len() * rows_per_shard;
            let buckets = synthetic_buckets(rows, seed);
            let rank = PleHotRank::from_buckets(rows as u64, FINGERPRINT, buckets.clone());
            let pack_path = dir.path().join("model.mq4.plehotrows");
            crate::ple_hot_pack::write_pack_with(
                &SliceReader(file.clone()),
                file.len() as u64,
                &offsets,
                rows_per_shard,
                encoding,
                row_width,
                FINGERPRINT,
                &rank,
                kept,
                group,
                &pack_path,
                &mut |_, _| {},
            )
            .unwrap();
            let store = HotRowStore::open(
                rank,
                &pack_path,
                rows_per_shard,
                encoding,
                row_bytes,
                &offsets,
                file.len() as u64,
            )
            .unwrap();
            let worker = store.spawn_worker(&model_path, offsets.clone());
            while store.verified().is_none() {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(store.verified(), Some(true));
            PackFixture {
                dir,
                model_path,
                pack_path,
                offsets,
                buckets,
                store,
                worker: Some(worker),
            }
        }
    }

    impl Drop for PackFixture {
        fn drop(&mut self) {
            self.store.stop();
            if let Some(worker) = self.worker.take() {
                worker.join().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    const ROWS_PER_SHARD: usize = 1000;
    const SHARDS: usize = 7;
    const ROWS: usize = ROWS_PER_SHARD * SHARDS;

    fn model_row(shards: &[Vec<u8>], row: u64) -> &[u8] {
        let shard = row as usize / ROWS_PER_SHARD;
        let local = row as usize % ROWS_PER_SHARD;
        &shards[shard][local * ENCODED_ROW_BYTES..(local + 1) * ENCODED_ROW_BYTES]
    }

    /// Every kept row's pack bytes are the model's bytes; rows of buckets the
    /// pack does not keep have no pack offset.
    #[test]
    fn pack_offsets_return_the_model_rows() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        for (kept, group) in [(100, 5), (256, 4), (7, 2)] {
            let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, kept, group, 41);
            let store = &fixture.store;
            assert_eq!(store.kept_buckets(), kept);
            assert_eq!(store.tier_count(), kept.div_ceil(group));
            let mut held = 0u64;
            let mut bytes = vec![0u8; ENCODED_ROW_BYTES];
            for row in 0..ROWS as u64 {
                match store.pack_offset(row) {
                    Some(offset) => {
                        assert!(usize::from(fixture.buckets[row as usize]) < kept);
                        store.read_pack(offset, &mut bytes).unwrap();
                        assert_eq!(bytes, model_row(&shards, row), "row {row}");
                        held += 1;
                    }
                    None => assert!(usize::from(fixture.buckets[row as usize]) >= kept),
                }
            }
            assert_eq!(held * ENCODED_ROW_BYTES as u64, store.data_bytes());
            assert_eq!(store.pack_offset(ROWS as u64), None);
        }
    }

    /// Reads outside the pack's row region are refused.
    #[test]
    fn read_pack_is_bounded_to_the_rows() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let store = &fixture.store;
        let mut bytes = vec![0u8; ENCODED_ROW_BYTES];
        assert!(
            store.read_pack(0, &mut bytes).is_err(),
            "header bytes are not rows"
        );
        let end = store.data_offset + store.data_bytes();
        assert!(store.read_pack(end - 10, &mut bytes).is_err());
        assert!(store
            .read_pack(end - ENCODED_ROW_BYTES as u64, &mut bytes)
            .is_ok());
    }

    /// A pack cut from another rank, another model size, or other shard
    /// offsets is refused at open.
    #[test]
    fn open_refuses_a_pack_from_another_rank_or_model() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let other_rank =
            || PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, synthetic_buckets(ROWS, 99));
        let same_rank =
            || PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone());
        let open = |rank, offsets: &[u64], bytes| {
            HotRowStore::open(
                rank,
                &fixture.pack_path,
                ROWS_PER_SHARD,
                RowEncoding::Q8F16,
                ENCODED_ROW_BYTES,
                offsets,
                bytes,
            )
        };
        assert!(open(same_rank(), &fixture.offsets, model_bytes).is_ok());
        assert!(open(other_rank(), &fixture.offsets, model_bytes).is_err());
        assert!(open(same_rank(), &fixture.offsets, model_bytes + 1).is_err());
        let mut moved = fixture.offsets.clone();
        moved[1] += 1;
        assert!(open(same_rank(), &moved, model_bytes).is_err());
    }

    /// A pack whose row encoding or row size differs from the row store's is
    /// refused at open.
    #[test]
    fn open_refuses_a_pack_of_another_row_encoding_or_size() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 64, 5, 3);
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let open = |encoding, row_bytes| {
            HotRowStore::open(
                PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone()),
                &fixture.pack_path,
                ROWS_PER_SHARD,
                encoding,
                row_bytes,
                &fixture.offsets,
                model_bytes,
            )
        };
        let store = open(RowEncoding::Q8F16, ENCODED_ROW_BYTES).unwrap();
        assert_eq!(store.encoding(), RowEncoding::Q8F16);
        assert_eq!(store.encoded_row_bytes(), ENCODED_ROW_BYTES);
        // Right size, wrong encoding.
        let err = open(RowEncoding::Bf16, ENCODED_ROW_BYTES).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        // Right encoding, wrong size.
        let err = open(RowEncoding::Q8F16, 160).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        let err = open(RowEncoding::Q8F16, 0).err().unwrap();
        assert!(err.contains("row store"), "{err}");
        // A BF16 row store of 160 values is 320 B rows: not this pack either.
        assert!(open(RowEncoding::Bf16, 2 * TEST_ROW_WIDTH).is_err());
    }

    /// A pack whose rows were altered after it was written fails the spot
    /// check and never serves.
    #[test]
    fn a_corrupted_pack_fails_verification_and_serves_nothing() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 256, 4, 5);
        // Flip one byte in every row of the pack.
        let header = read_ple_hot_pack_header(&fixture.pack_path).unwrap();
        let mut bytes = std::fs::read(&fixture.pack_path).unwrap();
        for row in 0..(header.data_bytes / ENCODED_ROW_BYTES as u64) {
            bytes[(header.data_offset + row * ENCODED_ROW_BYTES as u64) as usize] ^= 0x5A;
        }
        std::fs::write(&fixture.pack_path, &bytes).unwrap();
        let rank = PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, fixture.buckets.clone());
        let model_bytes = std::fs::metadata(&fixture.model_path).unwrap().len();
        let store = HotRowStore::open(
            rank,
            &fixture.pack_path,
            ROWS_PER_SHARD,
            RowEncoding::Q8F16,
            ENCODED_ROW_BYTES,
            &fixture.offsets,
            model_bytes,
        )
        .unwrap();
        assert!(!store.is_serving());
        assert_eq!(
            store.pack_offset(0),
            None,
            "nothing is served before verification"
        );
        store
            .spawn_worker(&fixture.model_path, fixture.offsets.clone())
            .join()
            .unwrap();
        assert_eq!(store.verified(), Some(false));
        assert!((0..ROWS as u64).all(|row| store.pack_offset(row).is_none()));
    }

    /// The prewarm waits for a budget, reads exactly that many bytes, and
    /// pauses while a fetch is in flight.
    #[test]
    fn prewarm_reads_its_budget_while_idle() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 200, 5, 8);
        let store = &fixture.store;
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(store.prewarmed_bytes(), 0, "no budget, no prewarm");
        let guard = store.busy();
        let budget = store.data_bytes() / 2;
        store.set_prewarm_budget(budget);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            store.prewarmed_bytes(),
            0,
            "a fetch in flight pauses the prewarm"
        );
        drop(guard);
        // Resumes BUSY_QUIET after the fetch; wait it out.
        let deadline = Instant::now() + BUSY_QUIET + Duration::from_secs(10);
        while !store.prewarm_complete() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(store.prewarm_complete());
        assert_eq!(store.prewarmed_bytes(), budget);
    }

    /// `stop` ends a worker that is still waiting for its budget.
    #[test]
    fn stop_ends_the_worker() {
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let mut fixture = PackFixture::new(&shards, ROWS_PER_SHARD, 32, 4, 2);
        fixture.store.stop();
        fixture.worker.take().unwrap().join().unwrap();
        assert!(!fixture.store.prewarm_complete());
    }

    #[test]
    fn sidecar_round_trip_and_rejections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.plehot");
        let buckets = synthetic_buckets(ROWS, 7);
        write_ple_hot_rank(
            &path,
            ROWS as u64,
            FINGERPRINT,
            &buckets,
            serde_json::json!({"k": 1}),
        )
        .unwrap();
        let rank = PleHotRank::open(&path, ROWS as u64, FINGERPRINT).unwrap();
        assert_eq!(rank.buckets(), buckets.as_slice());
        assert_eq!(rank.header()["rank"]["k"], 1);
        rank.verify().unwrap();
        assert!(PleHotRank::open(&path, ROWS as u64 + 1, FINGERPRINT).is_err());
        let other = FINGERPRINT.replace('0', "f");
        assert!(PleHotRank::open(&path, ROWS as u64, &other).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let tampered = PleHotRank::open(&path, ROWS as u64, FINGERPRINT).unwrap();
        assert!(tampered.verify().is_err());
        bytes.pop();
        std::fs::write(&path, &bytes).unwrap();
        assert!(PleHotRank::open(&path, ROWS as u64, FINGERPRINT).is_err());
    }
}
