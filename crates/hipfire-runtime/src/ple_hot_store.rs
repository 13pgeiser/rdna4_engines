// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE hot-row store: raw encoded PLE rows kept in host RAM on unified-memory
//! GPUs, so the most frequently used rows of the external row table are served
//! without SSD reads.
//!
//! The Qwen4 PLE table is ~320M padded rows of 170 B (Q8F16) stored in shards
//! inside the model file.  A frequency rank of the rows (the `.plehot`
//! sidecar, one bucket byte per row, bucket 0 = hottest) selects which rows to
//! keep.  Rows are copied *verbatim from the model file* and decoded by the
//! same `RowEncoding::decode_row` the SSD path uses, so the store never changes
//! the bytes a request returns, only where they come from.  The fill runs only
//! while no request is being served: every PLE fetch holds `busy`, and the
//! fill resumes `BUSY_QUIET` after the last one.
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
//! HotRowStore::new(rank: PleHotRank, budget_bytes: u64, rows_per_shard: usize,
//!                  encoded_row_bytes: usize) -> Result<Arc<HotRowStore>, String>
//! HotRowStore::new_with_group(.., tier_group: usize)   // tier = `tier_group` buckets (default 5)
//! HotRowStore::spawn_fill(self: &Arc<Self>, model_path: &Path, shard_file_offsets: Vec<u64>,
//!                         floor_bytes: u64, throttle: FillThrottle) -> JoinHandle<()>
//! FillThrottle { load_bytes_per_s: u64, min_bytes_per_s: u64, adaptive: bool }
//! HotRowStore::{stop(), tier_count(), kept_buckets(), resident_bytes() -> u64,
//!               filled_bytes() -> u64, release_throttle(), release_lowest() -> u64,
//!               released_tiers(), fill_complete(),
//!               sha_verified() -> Option<bool>, hot_rows() -> u64, hot_pages() -> u64,
//!               read() -> HotGuard, get(&HotGuard, row: u64) -> Option<&[u8]>}
//! impl session_cache::YieldFirst for HotRowStore
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
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::session_cache::YieldFirst;

const MAGIC: &[u8; 8] = b"HFPLEHT1";
const VERSION: u64 = 1;
/// Frequency buckets in a sidecar.
pub const BUCKETS: usize = 256;
/// Encoded bytes of one Q8F16 PLE row (160 values = 5 blocks of 34 B).
pub const ENCODED_ROW_BYTES: usize = 170;
/// Buckets per tier (tier = release unit).
pub const DEFAULT_TIER_BUCKETS: usize = 5;
/// Rows per prefix-count block of the slot index.
const BLOCK_ROWS: usize = 512;
/// One aligned read of the fill.
const READ_CHUNK: usize = 16 << 20;
const READ_ALIGN: u64 = 4096;
const FILL_THREADS: usize = 4;
/// Copied bytes between two host-availability checks of one fill thread.
const PRESSURE_INTERVAL_BYTES: u64 = 256 << 20;
/// The fill resumes once no PLE fetch ran for this long: longer than the
/// compute between two prefill chunks' fetches (~4 s for an 8192-row chunk on
/// Halo), so the fill only runs while no request is being served.
const BUSY_QUIET: Duration = Duration::from_secs(5);
/// Sanity bound on a sidecar header.
const MAX_HEADER_BYTES: u64 = 16 << 20;
/// `tier_of` value of a bucket that is not kept.
const NO_TIER: u8 = u8::MAX;

const SHA_UNKNOWN: u8 = 0;
const SHA_OK: u8 = 1;
const SHA_FAILED: u8 = 2;

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
        if row_bytes != ENCODED_ROW_BYTES as u64 {
            return Err(format!(
                "ple hot rank {}: encoded_row_bytes {row_bytes}, expected {ENCODED_ROW_BYTES}",
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
    let header_bytes = serde_json::to_vec(&header).map_err(|e| format!("ple hot rank header: {e}"))?;
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
// Tier storage
// ---------------------------------------------------------------------------

/// One tier's rows: a zero-initialised allocation committed lazily by the
/// kernel (`calloc` -> anonymous mmap for these sizes; never pre-touched).
///
/// It is a raw allocation rather than a `Box<[u8]>` because the fill thread
/// writes rows of not-yet-ready shards while readers hold `&` access to rows
/// of ready shards: only row-sized slices are ever materialised (see
/// [`TierBuf::row`] / [`TierBuf::write_row`]), never a whole-tier `&[u8]`.
struct TierBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer is plain bytes owned by this value; concurrent access is
// disciplined by the `ready` protocol documented at `HotRowStore`.
unsafe impl Send for TierBuf {}
unsafe impl Sync for TierBuf {}

impl TierBuf {
    /// Align 16 keeps `alloc_zeroed` on the `calloc` path (a larger alignment
    /// falls back to allocate-then-memset, which would touch every page).
    const ALIGN: usize = 16;

    fn zeroed(len: usize) -> Option<TierBuf> {
        if len == 0 {
            return Some(TierBuf {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, Self::ALIGN).ok()?;
        // SAFETY: `layout` has non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })?;
        Some(TierBuf { ptr, len })
    }

    /// Copy `src` to byte `offset`.
    ///
    /// The caller guarantees no other thread reads or writes `offset..+len`
    /// concurrently: the fill writes only rows of shards that are not yet
    /// `ready`, readers read only rows of ready shards, and each row has
    /// exactly one writer (its shard's fill thread).
    fn write_row(&self, offset: usize, src: &[u8]) {
        assert!(
            offset.checked_add(src.len()).is_some_and(|end| end <= self.len),
            "hot row write outside its tier"
        );
        // SAFETY: bounds checked above; exclusivity per the contract above.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr().add(offset), src.len()) }
    }

    /// Borrow `len` bytes at `offset`.  Only called for rows of ready shards,
    /// which the fill never writes again.
    fn row(&self, offset: usize, len: usize) -> &[u8] {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "hot row read outside its tier"
        );
        // SAFETY: in bounds; the row is immutable once its shard is ready
        // (published with Release, observed with Acquire by the caller).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(offset), len) }
    }
}

impl Drop for TierBuf {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: allocated in `zeroed` with this exact layout.
            unsafe {
                dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.len, Self::ALIGN),
                )
            }
        }
    }
}

/// A sector-aligned read buffer for O_DIRECT.
struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> AlignedBuf {
        let layout = Layout::from_size_align(len, READ_ALIGN as usize).expect("read buffer layout");
        // SAFETY: non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        AlignedBuf {
            ptr: NonNull::new(ptr).expect("read buffer allocation"),
            len,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
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

struct FileRead(File);

impl ShardRead for FileRead {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.0.read_at(dst, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                other => return other,
            }
        }
    }
}

/// Pacing of a fill that overlaps the model load (see
/// [`HotRowStore::spawn_fill`]).  The four fill threads share one token bucket
/// (a read of up to 16 MiB takes that many tokens first), so the fill's read
/// rate is bounded however many threads it uses, until
/// [`HotRowStore::release_throttle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillThrottle {
    /// Fill rate ceiling while the load runs, bytes/s.  `0` = do not read at
    /// all until [`HotRowStore::release_throttle`] (post-load fill).
    pub load_bytes_per_s: u64,
    /// Floor of the adaptive rate, bytes/s.  Also the starting rate when
    /// `adaptive` (the load gets a clean read-rate baseline first, then the
    /// fill ramps up by 25% per second).
    pub min_bytes_per_s: u64,
    /// Back the rate off when the load's own read rate drops, ramp up
    /// otherwise.  `false` = a fixed `load_bytes_per_s`.
    pub adaptive: bool,
}

/// Outcome of asking the bucket for tokens.
#[derive(Debug, PartialEq)]
enum Take {
    Granted,
    /// Seconds until enough tokens exist at the current rate.
    Wait(f64),
    /// Rate is 0: nothing is granted until the throttle is released.
    Blocked,
}

/// Token-bucket arithmetic over an explicit clock (seconds), so it can be
/// tested without sleeping.
struct BucketState {
    /// Bytes per second.
    rate: f64,
    tokens: f64,
    last_s: f64,
    /// Largest balance (the burst).
    cap: f64,
}

impl BucketState {
    fn new(rate: f64, cap: f64) -> Self {
        BucketState {
            rate,
            tokens: 0.0,
            last_s: 0.0,
            cap,
        }
    }

    fn refill(&mut self, now_s: f64) {
        let elapsed = (now_s - self.last_s).max(0.0);
        self.tokens = (self.tokens + elapsed * self.rate).min(self.cap);
        self.last_s = self.last_s.max(now_s);
    }

    fn set_rate(&mut self, now_s: f64, rate: f64) {
        self.refill(now_s);
        self.rate = rate;
    }

    fn try_take(&mut self, now_s: f64, bytes: f64) -> Take {
        self.refill(now_s);
        if self.rate <= 0.0 {
            return Take::Blocked;
        }
        let bytes = bytes.min(self.cap);
        // Tolerance: refill by f64 products can land a hair under `bytes`,
        // and a wait of that hair is below the clock's resolution.
        if self.tokens >= bytes - 1e-6 {
            self.tokens -= bytes;
            Take::Granted
        } else {
            Take::Wait((bytes - self.tokens) / self.rate)
        }
    }
}

/// The fill's shared token bucket.  Unthrottled until
/// [`TokenBucket::configure`]; [`TokenBucket::release`] ends the throttle.
struct TokenBucket {
    /// A throttle is in force (configured and not yet released).
    active: AtomicBool,
    epoch: Instant,
    state: Mutex<BucketState>,
    /// The configured throttle and when it was configured.
    meta: Mutex<Option<(FillThrottle, Instant)>>,
}

impl TokenBucket {
    fn new() -> Self {
        TokenBucket {
            active: AtomicBool::new(false),
            epoch: Instant::now(),
            state: Mutex::new(BucketState::new(0.0, (FILL_THREADS * READ_CHUNK) as f64)),
            meta: Mutex::new(None),
        }
    }

    fn now_s(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    fn state(&self) -> std::sync::MutexGuard<'_, BucketState> {
        self.state.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    /// Start rate: the floor when adaptive, the fixed rate otherwise.
    fn configure(&self, throttle: FillThrottle) {
        let ceiling = throttle.load_bytes_per_s;
        let start = if throttle.adaptive {
            throttle.min_bytes_per_s.min(ceiling)
        } else {
            ceiling
        };
        self.state().set_rate(self.now_s(), start as f64);
        *self.meta.lock().unwrap_or_else(|poison| poison.into_inner()) =
            Some((throttle, Instant::now()));
        self.active.store(true, Ordering::Release);
    }

    fn throttle(&self) -> Option<FillThrottle> {
        let meta = *self.meta.lock().unwrap_or_else(|poison| poison.into_inner());
        meta.map(|(throttle, _)| throttle)
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    fn set_rate(&self, rate: f64) {
        self.state().set_rate(self.now_s(), rate);
    }

    fn rate(&self) -> f64 {
        self.state().rate
    }

    fn try_take(&self, bytes: u64) -> Take {
        if !self.is_active() {
            return Take::Granted;
        }
        self.state().try_take(self.now_s(), bytes as f64)
    }

    /// End the throttle.  `Some(configured_at)` on the call that ended it.
    fn release(&self) -> Option<Instant> {
        if !self.active.swap(false, Ordering::AcqRel) {
            return None;
        }
        let meta = *self.meta.lock().unwrap_or_else(|poison| poison.into_inner());
        meta.map(|(_, configured_at)| configured_at)
    }
}

/// Load-read rate (bytes/s) below which the sweep is considered not reading
/// (GPU upload phases): the fill rate is kept.
const SWEEP_IDLE_BYTES_PER_S: f64 = 100e6;
/// A load rate under this fraction of its recent peak means the fill slows it.
const SLOWDOWN_RATIO: f64 = 0.85;
/// Per-sample decay of the load-rate peak.
const PEAK_DECAY: f64 = 0.97;
/// Multiplicative ramp-up of the fill rate per unslowed sample.
const RAMP_UP: f64 = 1.25;

/// The adaptive rate rule, one [`AdaptiveRate::step`] per second.
struct AdaptiveRate {
    rate: f64,
    min: f64,
    max: f64,
    peak: f64,
}

impl AdaptiveRate {
    fn new(start: f64, min: f64, max: f64) -> Self {
        AdaptiveRate {
            rate: start.clamp(min, max),
            min,
            max,
            peak: 0.0,
        }
    }

    /// Feed the load's read rate over the last interval; returns the new fill
    /// rate.  Reading but under [`SLOWDOWN_RATIO`] of the peak: halve (not
    /// below `min`); reading at full pace: +25% (not above `max`); not
    /// reading: unchanged.
    fn step(&mut self, load_bytes_per_s: f64) -> f64 {
        self.peak = (self.peak * PEAK_DECAY).max(load_bytes_per_s);
        if load_bytes_per_s > SWEEP_IDLE_BYTES_PER_S {
            self.rate = if load_bytes_per_s < SLOWDOWN_RATIO * self.peak {
                (self.rate * 0.5).max(self.min)
            } else {
                (self.rate * RAMP_UP).min(self.max)
            };
        }
        self.rate
    }
}

/// Bytes this process has read from storage (`/proc/self/io` `read_bytes`).
fn proc_io_read_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/io").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("read_bytes:"))
        .and_then(|value| value.trim().parse().ok())
}

fn debug_enabled() -> bool {
    hipfire_config::memory::ple_hot_debug()
}

/// Parameters of one fill run (production values via `spawn_fill`).
pub(crate) struct FillParams<'a> {
    /// Yield tiers while host availability is below this.
    pub floor_bytes: u64,
    /// Copied bytes between availability checks (per fill thread).
    pub pressure_interval_bytes: u64,
    /// Host availability probe.
    pub avail: &'a (dyn Fn() -> Option<u64> + Sync),
    /// Process storage-read counter probe (the adaptive throttle's input).
    pub io_read_bytes: &'a (dyn Fn() -> Option<u64> + Sync),
    /// Time between adaptive-rate samples.
    pub adapt_interval: Duration,
}

// ---------------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------------

/// Read access to the tiers.  Hold it only for the copy loop (no I/O under
/// it): [`HotRowStore::release_lowest`] waits for outstanding guards.
pub struct HotGuard<'a>(RwLockReadGuard<'a, Vec<Option<TierBuf>>>);

/// Raw encoded PLE rows of the hottest buckets, resident in host RAM.
///
/// # Concurrency protocol
///
/// `ready[s]` (one flag per shard) is set with `Release` by the fill thread
/// after every kept row of shard `s` has been copied, and read with `Acquire`
/// by [`HotRowStore::get`].  Fill threads write only rows of shards that are
/// not ready; readers return only rows of ready shards, and a ready shard is
/// never written again.  A tier is dropped under the write lock; fill and
/// readers hold the read lock only around their copy loops.
pub struct HotRowStore {
    padded_rows: u64,
    rows_per_shard: usize,
    encoded_row_bytes: usize,
    /// Buckets per tier.
    group: usize,
    /// Kept buckets `K`: buckets `0..K` are stored.
    kept: usize,
    tier_count: usize,
    /// Bucket -> tier (`NO_TIER` when not kept).
    tier_of: [u8; BUCKETS],
    /// The sidecar payload (empty when nothing is kept).
    bucket: Box<[u8]>,
    /// `prefix[t][block]` = rows of tier `t` in rows `[0, block * 512)`.
    prefix: Vec<Box<[u32]>>,
    /// Rows per tier (read by the tests' slot/fill checks).
    #[cfg_attr(not(test), allow(dead_code))]
    tier_rows: Vec<u64>,
    /// Lowest-ranked present tier is released first; released tiers are
    /// `None` and never re-filled.
    tiers: RwLock<Vec<Option<TierBuf>>>,
    ready: Box<[AtomicBool]>,
    stop: AtomicBool,
    expected_sha: String,
    sha_state: AtomicU8,
    fill_done: AtomicBool,
    released: AtomicUsize,
    hot_rows: AtomicU64,
    hot_pages: AtomicU64,
    /// Throttle shared by the fill threads (unthrottled until `spawn_fill`).
    throttle_bucket: TokenBucket,
    /// Bytes the fill read from the model file.
    bytes_read: AtomicU64,
    /// Bytes of rows copied into tiers still present.
    filled: AtomicU64,
    /// Per-tier share of `filled`.
    tier_filled: Box<[AtomicU64]>,
    /// Time base of `busy_at_ms`.
    created: Instant,
    /// Milliseconds after `created` (plus one; 0 = never) at which the last
    /// PLE fetch finished (see [`HotRowStore::busy`]).
    busy_at_ms: AtomicU64,
    /// PLE fetches in flight.
    busy_active: AtomicUsize,
}

/// A PLE fetch in flight (from [`HotRowStore::busy`]); the fill stays paused
/// until [`BUSY_QUIET`] after the last guard drops.
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
    /// Build the index for `rank` under `budget_bytes` (tiers of
    /// [`DEFAULT_TIER_BUCKETS`] buckets).  A budget too small for one bucket
    /// yields a store with no tiers, which is a no-op everywhere.  Errors on
    /// a geometry mismatch only.  Allocates tier memory lazily; copies
    /// nothing until [`HotRowStore::spawn_fill`].
    pub fn new(
        rank: PleHotRank,
        budget_bytes: u64,
        rows_per_shard: usize,
        encoded_row_bytes: usize,
    ) -> Result<std::sync::Arc<HotRowStore>, String> {
        Self::new_with_group(
            rank,
            budget_bytes,
            rows_per_shard,
            encoded_row_bytes,
            DEFAULT_TIER_BUCKETS,
        )
    }

    /// [`HotRowStore::new`] with an explicit tier size in buckets (>= 2).
    pub fn new_with_group(
        rank: PleHotRank,
        budget_bytes: u64,
        rows_per_shard: usize,
        encoded_row_bytes: usize,
        tier_group: usize,
    ) -> Result<std::sync::Arc<HotRowStore>, String> {
        let PleHotRank {
            padded_rows,
            payload_sha256,
            buckets,
            ..
        } = rank;
        if encoded_row_bytes != ENCODED_ROW_BYTES {
            return Err(format!(
                "ple hot store: row store rows are {encoded_row_bytes} B, sidecar rows are \
                 {ENCODED_ROW_BYTES} B"
            ));
        }
        if rows_per_shard == 0 || padded_rows == 0 || padded_rows % rows_per_shard as u64 != 0 {
            return Err(format!(
                "ple hot store: {padded_rows} padded rows is not a whole number of \
                 {rows_per_shard}-row shards"
            ));
        }
        if padded_rows > u64::from(u32::MAX) {
            return Err(format!("ple hot store: {padded_rows} rows exceed the index range"));
        }
        if !(2..=BUCKETS).contains(&tier_group) {
            return Err(format!("ple hot store: tier group {tier_group} outside 2..=256"));
        }
        let shards = (padded_rows / rows_per_shard as u64) as usize;
        let ready: Box<[AtomicBool]> = (0..shards).map(|_| AtomicBool::new(false)).collect();

        let kept = if budget_bytes == 0 {
            0
        } else {
            let rows_fit = u128::from(budget_bytes) * BUCKETS as u128
                / (encoded_row_bytes as u128 * u128::from(padded_rows));
            rows_fit.min(BUCKETS as u128) as usize
        };
        let tier_count = kept.div_ceil(tier_group);

        let mut tier_of = [NO_TIER; BUCKETS];
        for (bucket, slot) in tier_of.iter_mut().enumerate().take(kept) {
            *slot = (bucket / tier_group) as u8;
        }

        let (bucket, prefix, tier_rows) = if kept == 0 {
            (Box::default(), Vec::new(), Vec::new())
        } else {
            let blocks = (padded_rows as usize).div_ceil(BLOCK_ROWS);
            let mut prefix: Vec<Vec<u32>> = (0..tier_count).map(|_| Vec::with_capacity(blocks)).collect();
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
            (
                buckets.into_boxed_slice(),
                prefix.into_iter().map(Vec::into_boxed_slice).collect(),
                tier_rows,
            )
        };

        let mut tiers: Vec<Option<TierBuf>> = Vec::with_capacity(tier_count);
        let mut logical = 0u64;
        for &rows in &tier_rows {
            let bytes = rows * encoded_row_bytes as u64;
            let tier = usize::try_from(bytes).ok().and_then(TierBuf::zeroed);
            if tier.is_some() {
                logical += bytes;
            } else {
                eprintln!("{TAG}: allocating a {:.0} MiB tier failed; tier disabled", mib(bytes));
            }
            tiers.push(tier);
        }
        let index_bytes = bucket.len() as u64
            + prefix.iter().map(|column| column.len() as u64 * 4).sum::<u64>();
        eprintln!(
            "{TAG}: budget {:.2} GiB -> kept {kept}/{BUCKETS} buckets in {tier_count} tiers \
             ({:.2} GiB logical, index {:.0} MiB), {padded_rows} rows in {shards} shards",
            gib(budget_bytes),
            gib(logical),
            mib(index_bytes),
        );

        Ok(std::sync::Arc::new(HotRowStore {
            padded_rows,
            rows_per_shard,
            encoded_row_bytes,
            group: tier_group,
            kept,
            tier_count,
            tier_of,
            bucket,
            prefix,
            tier_rows,
            tiers: RwLock::new(tiers),
            ready,
            stop: AtomicBool::new(false),
            expected_sha: payload_sha256,
            sha_state: AtomicU8::new(SHA_UNKNOWN),
            fill_done: AtomicBool::new(false),
            released: AtomicUsize::new(0),
            hot_rows: AtomicU64::new(0),
            hot_pages: AtomicU64::new(0),
            throttle_bucket: TokenBucket::new(),
            bytes_read: AtomicU64::new(0),
            filled: AtomicU64::new(0),
            tier_filled: (0..tier_count).map(|_| AtomicU64::new(0)).collect(),
            created: Instant::now(),
            busy_at_ms: AtomicU64::new(0),
            busy_active: AtomicUsize::new(0),
        }))
    }

    /// Tiers this store was built with (0 = no-op store).
    pub fn tier_count(&self) -> usize {
        self.tier_count
    }

    /// Buckets (of 256) kept under the budget.
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

    /// Logical bytes of the tiers still present (includes pages the fill has
    /// not touched yet).
    pub fn resident_bytes(&self) -> u64 {
        self.read_tiers()
            .iter()
            .flatten()
            .map(|tier| tier.len as u64)
            .sum()
    }

    /// Bytes of rows copied so far into tiers that are still present
    /// (released tiers subtract what they held).  Unlike
    /// [`HotRowStore::resident_bytes`] this counts only what the fill wrote.
    pub fn filled_bytes(&self) -> u64 {
        self.filled.load(Ordering::Acquire)
    }

    /// Put the throttle in force (the fill threads of `spawn_fill`).
    pub(crate) fn configure_throttle(&self, throttle: FillThrottle) {
        self.throttle_bucket.configure(throttle);
    }

    /// The model load is done: from now on the fill runs unthrottled (a
    /// `load_bytes_per_s == 0` fill starts reading).  Logs once.
    pub fn release_throttle(&self) {
        let Some(configured_at) = self.throttle_bucket.release() else {
            return;
        };
        let seconds = configured_at.elapsed().as_secs_f64();
        eprintln!(
            "{TAG}: throttle released after {seconds:.1} s ({:.2} GiB filled, {:.2} GB/s \
             average while throttled)",
            gib(self.filled_bytes()),
            self.bytes_read.load(Ordering::Relaxed) as f64 / 1e9 / seconds.max(1e-9),
        );
    }

    /// Tiers dropped so far.
    pub fn released_tiers(&self) -> usize {
        self.released.load(Ordering::Acquire)
    }

    /// Every shard was read and its rows copied.
    pub fn fill_complete(&self) -> bool {
        self.fill_done.load(Ordering::Acquire)
    }

    /// Sidecar payload hash result: `None` until the fill thread checked it.
    pub fn sha_verified(&self) -> Option<bool> {
        match self.sha_state.load(Ordering::Acquire) {
            SHA_OK => Some(true),
            SHA_FAILED => Some(false),
            _ => None,
        }
    }

    /// Rows served from the store.
    pub fn hot_rows(&self) -> u64 {
        self.hot_rows.load(Ordering::Relaxed)
    }

    /// Pages served entirely from the store.
    pub fn hot_pages(&self) -> u64 {
        self.hot_pages.load(Ordering::Relaxed)
    }

    pub(crate) fn note_served(&self, rows: u64, pages: u64) {
        self.hot_rows.fetch_add(rows, Ordering::Relaxed);
        self.hot_pages.fetch_add(pages, Ordering::Relaxed);
    }

    /// The row store is serving a PLE fetch (a prefill chunk or a decode
    /// step).  The fill pauses while the returned guard lives and for
    /// [`BUSY_QUIET`] after the last one drops, so it never competes with a
    /// request for the SSD, CPU or memory.
    pub(crate) fn busy(&self) -> BusyGuard<'_> {
        self.busy_active.fetch_add(1, Ordering::AcqRel);
        BusyGuard(self)
    }

    /// Sleep while a PLE fetch is in flight or finished less than
    /// [`BUSY_QUIET`] ago; `false` when the fill was stopped meanwhile.
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

    /// Ask a running fill to stop at its next read; pair with joining the
    /// handle `spawn_fill` returned.  Dropping the last `Arc` does not stop
    /// the fill (the fill thread holds one).
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    fn read_tiers(&self) -> RwLockReadGuard<'_, Vec<Option<TierBuf>>> {
        self.tiers.read().unwrap_or_else(|poison| poison.into_inner())
    }

    /// Read access for a run of [`HotRowStore::get`] calls.
    pub fn read(&self) -> HotGuard<'_> {
        HotGuard(self.read_tiers())
    }

    /// The `encoded_row_bytes` encoded bytes of global `row`, when its shard
    /// is ready and its tier present.  Never returns padding-garbage: an
    /// unready shard or released tier is `None`.
    pub fn get<'g>(&self, guard: &'g HotGuard<'_>, row: u64) -> Option<&'g [u8]> {
        if row >= self.padded_rows || self.kept == 0 {
            return None;
        }
        let shard = (row / self.rows_per_shard as u64) as usize;
        if !self.ready[shard].load(Ordering::Acquire) {
            return None;
        }
        let tier = self.tier_of[usize::from(self.bucket[row as usize])];
        if tier == NO_TIER {
            return None;
        }
        let tier = usize::from(tier);
        let buffer = guard.0[tier].as_ref()?;
        let offset = self.slot(row, tier) as usize * self.encoded_row_bytes;
        Some(buffer.row(offset, self.encoded_row_bytes))
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

    /// Tier of `row` when its bucket is kept.
    #[cfg(test)]
    pub(crate) fn tier_of_row(&self, row: u64) -> Option<usize> {
        match self.tier_of[usize::from(*self.bucket.get(row as usize)?)] {
            NO_TIER => None,
            tier => Some(usize::from(tier)),
        }
    }

    /// Rows of each tier that precede global `row` (the fill cursors for a
    /// shard that starts at `row`).
    fn cursors_at(&self, row: u64) -> Vec<u64> {
        let row = row as usize;
        let block = row / BLOCK_ROWS;
        let mut cursors: Vec<u64> = self
            .prefix
            .iter()
            .map(|column| u64::from(column[block]))
            .collect();
        for &b in &self.bucket[block * BLOCK_ROWS..row] {
            let tier = self.tier_of[usize::from(b)];
            if tier != NO_TIER {
                cursors[usize::from(tier)] += 1;
            }
        }
        cursors
    }

    /// Drop the lowest-ranked present tier and return its bytes (0 when none
    /// is left).  Waits for readers to leave their copy loop; the memory is
    /// back with the kernel when this returns.
    pub fn release_lowest(&self) -> u64 {
        let started = Instant::now();
        let released = {
            let mut tiers = self.tiers.write().unwrap_or_else(|poison| poison.into_inner());
            let found = tiers
                .iter_mut()
                .enumerate()
                .rev()
                .find_map(|(index, slot)| match slot.take() {
                    // An empty tier frees nothing and would read as "none left".
                    Some(tier) if tier.len > 0 => Some((index, tier)),
                    _ => None,
                });
            // The fill adds to these only under the read lock, so under the
            // write lock the tier's count is final.
            if let Some((index, _)) = &found {
                let copied = self.tier_filled[*index].swap(0, Ordering::AcqRel);
                self.filled.fetch_sub(copied, Ordering::AcqRel);
            }
            found
        };
        let Some((index, tier)) = released else {
            return 0;
        };
        let bytes = tier.len as u64;
        // Unmapping a touched ~1 GiB tier takes milliseconds: outside the lock.
        drop(tier);
        self.released.fetch_add(1, Ordering::AcqRel);
        eprintln!(
            "{TAG}: released tier {index} ({:.0} MiB) in {:.1} ms",
            mib(bytes),
            started.elapsed().as_secs_f64() * 1e3
        );
        bytes
    }

    fn release_all(&self) {
        while self.release_lowest() > 0 {}
    }

    /// Force a shard's readiness (tests: simulate a fill that has not reached it).
    #[cfg(test)]
    pub(crate) fn set_shard_ready_for_test(&self, shard: usize, ready: bool) {
        self.ready[shard].store(ready, Ordering::Release);
    }

    /// Verify the sidecar payload, then read the PLE shards of `model_path`
    /// (sequentially, O_DIRECT, 4 threads) and keep every row of a present
    /// tier.  `shard_file_offsets[s]` is the absolute file offset of shard `s`
    /// (`rows_per_shard` rows of `encoded_row_bytes`; padding rows included).
    /// While host availability is below `floor_bytes` the lowest tiers are
    /// released.  The fill is paced by `throttle` (shared token bucket, see
    /// [`FillThrottle`]) until [`HotRowStore::release_throttle`].  Returns at
    /// once; the work runs on the returned thread.
    /// Call [`HotRowStore::stop`] then join it before unloading the row store.
    pub fn spawn_fill(
        self: &std::sync::Arc<Self>,
        model_path: &Path,
        shard_file_offsets: Vec<u64>,
        floor_bytes: u64,
        throttle: FillThrottle,
    ) -> JoinHandle<()> {
        let store = self.clone();
        let path = model_path.to_path_buf();
        // Before the thread exists, so a release right after this call
        // cannot be overtaken by the configuration.
        self.configure_throttle(throttle);
        std::thread::Builder::new()
            .name("ple-hot-fill".to_string())
            .spawn(move || store.run_fill(&path, &shard_file_offsets, floor_bytes))
            .expect("spawn ple hot fill thread")
    }

    fn run_fill(&self, path: &Path, offsets: &[u64], floor_bytes: u64) {
        if self.tier_count == 0 {
            return;
        }
        if !self.verify_sidecar() {
            self.release_all();
            return;
        }
        let file = match open_direct(path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("{TAG}: opening {} failed: {error}; releasing the store", path.display());
                self.release_all();
                return;
            }
        };
        self.fill_with(
            &FileRead(file),
            offsets,
            &FillParams {
                floor_bytes,
                pressure_interval_bytes: PRESSURE_INTERVAL_BYTES,
                avail: &rdna_compute::uma_memory::host_available_bytes,
                io_read_bytes: &proc_io_read_bytes,
                adapt_interval: Duration::from_secs(1),
            },
        );
    }

    /// Hash the retained sidecar payload; `false` (and a log) on mismatch.
    fn verify_sidecar(&self) -> bool {
        let started = Instant::now();
        let actual = hex(&Sha256::digest(&self.bucket));
        let ok = actual == self.expected_sha;
        self.sha_state
            .store(if ok { SHA_OK } else { SHA_FAILED }, Ordering::Release);
        if ok {
            eprintln!(
                "{TAG}: sidecar sha256 ok ({:.0} ms)",
                started.elapsed().as_secs_f64() * 1e3
            );
        } else {
            eprintln!(
                "{TAG}: sidecar sha256 FAILED (payload {actual}, header {}); releasing the store",
                self.expected_sha
            );
        }
        ok
    }

    /// The fill proper, over any positional reader (production: the O_DIRECT
    /// model file; tests: memory).
    pub(crate) fn fill_with(&self, reader: &dyn ShardRead, offsets: &[u64], params: &FillParams<'_>) {
        let shards = self.ready.len();
        if offsets.len() != shards {
            eprintln!(
                "{TAG}: {} shard offsets for {shards} shards; releasing the store",
                offsets.len()
            );
            self.release_all();
            return;
        }
        let started = Instant::now();
        let read_at_start = self.bytes_read.load(Ordering::Relaxed);
        let rows_copied = AtomicU64::new(0);
        let failed = AtomicBool::new(false);
        let pressure = Mutex::new(());
        let workers = FILL_THREADS.min(shards);
        let running = AtomicUsize::new(workers);
        std::thread::scope(|scope| {
            for worker in 0..workers {
                let (rows_copied, failed, pressure, running) =
                    (&rows_copied, &failed, &pressure, &running);
                scope.spawn(move || {
                    let work = || {
                        let mut buffer = AlignedBuf::new(READ_CHUNK);
                        let mut since_check = 0u64;
                        let mut shard = worker;
                        while shard < shards {
                            if self.stopped() || failed.load(Ordering::Acquire) {
                                return;
                            }
                            let outcome = self.fill_shard(
                                shard,
                                offsets[shard],
                                reader,
                                &mut buffer,
                                params,
                                &mut since_check,
                                pressure,
                            );
                            match outcome {
                                Ok(Some(copied)) => {
                                    rows_copied.fetch_add(copied, Ordering::Relaxed);
                                    self.ready[shard].store(true, Ordering::Release);
                                }
                                Ok(None) => return,
                                Err(error) => {
                                    eprintln!("{TAG}: fill of shard {shard} failed: {error}");
                                    failed.store(true, Ordering::Release);
                                    return;
                                }
                            }
                            shard += FILL_THREADS;
                        }
                    };
                    work();
                    running.fetch_sub(1, Ordering::AcqRel);
                });
            }
            scope.spawn(|| self.run_rate_controller(params, &running));
        });
        let seconds = started.elapsed().as_secs_f64();
        let complete = !failed.load(Ordering::Acquire)
            && !self.stopped()
            && self.ready.iter().all(|flag| flag.load(Ordering::Acquire));
        if complete {
            self.fill_done.store(true, Ordering::Release);
        }
        let bytes = self.bytes_read.load(Ordering::Relaxed) - read_at_start;
        eprintln!(
            "{TAG}: fill {}: {} tiers present, {:.2} GiB resident, {} rows, {seconds:.1} s \
             ({:.2} GB/s read), {} tiers released",
            if complete { "complete" } else { "stopped early" },
            self.tier_count - self.released_tiers().min(self.tier_count),
            gib(self.resident_bytes()),
            rows_copied.load(Ordering::Relaxed),
            bytes as f64 / 1e9 / seconds.max(1e-9),
            self.released_tiers(),
        );
    }

    /// Block until the throttle grants `bytes` of tokens; `false` when the
    /// fill was stopped meanwhile.
    fn acquire_tokens(&self, bytes: u64) -> bool {
        loop {
            if self.stopped() {
                return false;
            }
            let nap = match self.throttle_bucket.try_take(bytes) {
                Take::Granted => return true,
                // Short naps so a rate change, release or stop is noticed.
                Take::Wait(seconds) => Duration::from_secs_f64(seconds)
                    .clamp(Duration::from_millis(1), Duration::from_millis(50)),
                Take::Blocked => Duration::from_millis(20),
            };
            std::thread::sleep(nap);
        }
    }

    /// Once per `adapt_interval`: the load's read rate (process storage reads
    /// minus the fill's own) drives [`AdaptiveRate`], whose rate goes to the
    /// bucket.  Returns when the fill threads are done, stopped, or the
    /// throttle was released.
    fn run_rate_controller(&self, params: &FillParams<'_>, running: &AtomicUsize) {
        let Some(throttle) = self.throttle_bucket.throttle() else {
            return;
        };
        if !throttle.adaptive || throttle.load_bytes_per_s == 0 {
            return;
        }
        let max = throttle.load_bytes_per_s as f64;
        let min = (throttle.min_bytes_per_s as f64).min(max);
        let mut rate = AdaptiveRate::new(self.throttle_bucket.rate(), min, max);
        let debug = debug_enabled();
        let mut last_io = (params.io_read_bytes)();
        let mut last_fill = self.bytes_read.load(Ordering::Relaxed);
        let mut last_at = Instant::now();
        while running.load(Ordering::Acquire) > 0 && !self.stopped() && self.throttle_bucket.is_active() {
            std::thread::sleep(Duration::from_millis(10).min(params.adapt_interval));
            if last_at.elapsed() < params.adapt_interval {
                continue;
            }
            let io = (params.io_read_bytes)();
            let fill = self.bytes_read.load(Ordering::Relaxed);
            let seconds = last_at.elapsed().as_secs_f64();
            last_at = Instant::now();
            if let (Some(io), Some(before)) = (io, last_io) {
                let load_bytes = io
                    .saturating_sub(before)
                    .saturating_sub(fill.saturating_sub(last_fill));
                let load_rate = load_bytes as f64 / seconds.max(1e-9);
                let previous = rate.rate;
                let next = rate.step(load_rate);
                if next != previous {
                    self.throttle_bucket.set_rate(next);
                    if debug {
                        eprintln!(
                            "{TAG}: fill rate {:.0} -> {:.0} MB/s (load reads {:.0} MB/s, peak \
                             {:.0} MB/s)",
                            previous / 1e6,
                            next / 1e6,
                            load_rate / 1e6,
                            rate.peak / 1e6,
                        );
                    }
                }
            }
            last_io = io;
            last_fill = fill;
        }
    }

    /// Fill one shard.  `Ok(Some(rows))` when every row was visited,
    /// `Ok(None)` when stopped or nothing is left to fill.
    #[allow(clippy::too_many_arguments)]
    fn fill_shard(
        &self,
        shard: usize,
        file_offset: u64,
        reader: &dyn ShardRead,
        buffer: &mut AlignedBuf,
        params: &FillParams<'_>,
        since_check: &mut u64,
        pressure: &Mutex<()>,
    ) -> Result<Option<u64>, String> {
        let row_bytes = self.encoded_row_bytes as u64;
        let rows = self.rows_per_shard as u64;
        let first_global = shard as u64 * rows;
        let shard_end = file_offset + rows * row_bytes;
        let mut cursors = self.cursors_at(first_global);
        let mut copied_per_tier = vec![0u64; self.tier_count];
        let mut next = 0u64;
        let mut copied_total = 0u64;
        while next < rows {
            if self.stopped() {
                return Ok(None);
            }
            // Aligned window starting at or before the first unread row.
            let start = (file_offset + next * row_bytes) & !(READ_ALIGN - 1);
            let end = (start + READ_CHUNK as u64).min(shard_end.next_multiple_of(READ_ALIGN));
            let want = (end - start) as usize;
            if !self.wait_until_idle() || !self.acquire_tokens(want as u64) {
                return Ok(None);
            }
            let dst = &mut buffer.as_mut_slice()[..want];
            let mut filled = 0usize;
            while filled < want {
                let n = reader
                    .read_at(start + filled as u64, &mut dst[filled..])
                    .map_err(|e| format!("read at {}: {e}", start + filled as u64))?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            self.bytes_read.fetch_add(filled as u64, Ordering::Relaxed);
            let available_end = start + filled as u64;
            let rows_end = (available_end.saturating_sub(file_offset) / row_bytes).min(rows);
            if rows_end <= next {
                return Err(format!(
                    "unexpected end of file at byte {available_end} (shard rows {next}..{rows})"
                ));
            }
            let mut copied = 0u64;
            {
                let guard = self.read_tiers();
                if guard.iter().all(Option::is_none) {
                    return Ok(None);
                }
                for row in next..rows_end {
                    let global = first_global + row;
                    let tier = self.tier_of[usize::from(self.bucket[global as usize])];
                    if tier == NO_TIER {
                        continue;
                    }
                    let tier = usize::from(tier);
                    debug_assert_eq!(cursors[tier], self.slot(global, tier), "cursor == slot");
                    if let Some(buf) = &guard[tier] {
                        let at = (file_offset + row * row_bytes - start) as usize;
                        buf.write_row(
                            cursors[tier] as usize * self.encoded_row_bytes,
                            &dst[at..at + self.encoded_row_bytes],
                        );
                        copied += 1;
                        copied_per_tier[tier] += row_bytes;
                    }
                    cursors[tier] += 1;
                }
                // Under the read lock: `release_lowest` reads the tier's
                // count only under the write lock.
                for (tier, bytes) in copied_per_tier.iter_mut().enumerate() {
                    if *bytes > 0 {
                        self.tier_filled[tier].fetch_add(*bytes, Ordering::AcqRel);
                        self.filled.fetch_add(*bytes, Ordering::AcqRel);
                        *bytes = 0;
                    }
                }
            }
            copied_total += copied;
            next = rows_end;
            *since_check += copied * row_bytes;
            if *since_check >= params.pressure_interval_bytes {
                *since_check = 0;
                self.yield_under_pressure(params, pressure);
            }
        }
        Ok(Some(copied_total))
    }

    /// Release the lowest tiers while host memory is below the floor.
    fn yield_under_pressure(&self, params: &FillParams<'_>, pressure: &Mutex<()>) {
        // One thread at a time: the next one re-reads availability after the
        // previous release instead of releasing again for the same shortfall.
        let _serial = pressure.lock().unwrap_or_else(|poison| poison.into_inner());
        while (params.avail)().is_some_and(|available| available < params.floor_bytes) {
            if self.release_lowest() == 0 {
                break;
            }
        }
    }
}

impl YieldFirst for HotRowStore {
    fn release_one(&self) -> u64 {
        self.release_lowest()
    }
}

/// Open `path` for O_DIRECT reads, falling back to buffered reads on a
/// filesystem that refuses the flag.
fn open_direct(path: &Path) -> io::Result<File> {
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {
            eprintln!("{TAG}: O_DIRECT unsupported on {}; using buffered reads", path.display());
            File::open(path)
        }
        other => other,
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

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

    /// Fill `store` from per-shard encoded bytes laid out at unaligned file
    /// offsets, with a floor that never triggers.
    pub(crate) fn fill_from_shards(store: &HotRowStore, shards: &[Vec<u8>]) -> Vec<u64> {
        let (file, offsets) = layout_shards(shards);
        store.fill_with(
            &SliceReader(file),
            &offsets,
            &FillParams {
                floor_bytes: 0,
                pressure_interval_bytes: u64::MAX,
                avail: &|| None,
                io_read_bytes: &|| None,
                adapt_interval: Duration::from_millis(20),
            },
        );
        offsets
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
                        bytes.push((global.wrapping_mul(0x9E37_79B9).wrapping_add(i * 31) >> 3) as u8);
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

    pub(crate) const FINGERPRINT: &str =
        "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use std::sync::Arc;

    const ROWS_PER_SHARD: usize = 1000;
    const SHARDS: usize = 7;
    const ROWS: usize = ROWS_PER_SHARD * SHARDS;

    /// A throttle that never limits in practice (adaptive on, so the rate
    /// controller runs and must end with the fill).
    fn fast_throttle() -> FillThrottle {
        FillThrottle {
            load_bytes_per_s: 1 << 40,
            min_bytes_per_s: 1 << 30,
            adaptive: true,
        }
    }

    fn store(budget_fraction_256: u64, group: usize, seed: u64) -> Arc<HotRowStore> {
        let rank = PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, synthetic_buckets(ROWS, seed));
        // Budget for `budget_fraction_256` buckets of the table.
        let budget = ROWS as u64 * ENCODED_ROW_BYTES as u64 * budget_fraction_256 / 256 + 1;
        HotRowStore::new_with_group(rank, budget, ROWS_PER_SHARD, ENCODED_ROW_BYTES, group).unwrap()
    }

    /// The fill's per-tier cursor equals `slot(row)` for every kept row (the
    /// fill itself debug-asserts it per copy), slots are dense and unique per
    /// tier, and every served row is the model file's bytes.
    #[test]
    fn slot_matches_fill_cursor_and_rows_are_the_file_bytes() {
        let hot = store(200, 3, 7);
        assert!(hot.tier_count() > 2);
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        fill_from_shards(&hot, &shards);
        assert!(hot.fill_complete());

        let mut seen: Vec<Vec<bool>> = hot.tier_rows.iter().map(|&n| vec![false; n as usize]).collect();
        let guard = hot.read();
        let mut served = 0usize;
        for row in 0..ROWS as u64 {
            let bucket = hot.bucket[row as usize] as usize;
            let want = &shards[row as usize / ROWS_PER_SHARD]
                [(row as usize % ROWS_PER_SHARD) * ENCODED_ROW_BYTES..][..ENCODED_ROW_BYTES];
            if bucket < hot.kept_buckets() {
                let tier = hot.tier_of_row(row).unwrap();
                assert_eq!(tier, bucket / 3);
                let slot = hot.slot(row, tier) as usize;
                assert!(!seen[tier][slot], "slot {slot} of tier {tier} assigned twice");
                seen[tier][slot] = true;
                assert_eq!(hot.get(&guard, row), Some(want), "row {row}");
                served += 1;
            } else {
                assert_eq!(hot.get(&guard, row), None, "cold row {row}");
            }
        }
        assert!(served > 0);
        assert!(seen.iter().flatten().all(|&used| used), "tier slots are dense");
        // Rows are in ascending global order inside a tier.
        for tier in 0..hot.tier_count() {
            let mut previous = None;
            for row in (0..ROWS as u64).filter(|&r| hot.tier_of_row(r) == Some(tier)) {
                let slot = hot.slot(row, tier);
                assert!(previous.map_or(true, |p| slot == p + 1));
                previous = Some(slot);
            }
        }
    }

    /// A row of an unready shard is never served, a ready one is.
    #[test]
    fn unready_shards_and_released_tiers_serve_nothing() {
        let hot = store(256, 4, 3);
        let guard = hot.read();
        assert!((0..ROWS as u64).all(|row| hot.get(&guard, row).is_none()));
        drop(guard);
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        fill_from_shards(&hot, &shards);
        let top = hot.tier_count() - 1;
        let victim = (0..ROWS as u64).find(|&row| hot.tier_of_row(row) == Some(top)).unwrap();
        let other = (0..ROWS as u64).find(|&row| hot.tier_of_row(row) == Some(0)).unwrap();
        assert!(hot.get(&hot.read(), victim).is_some());
        let before = hot.resident_bytes();
        let freed = hot.release_lowest();
        assert_eq!(freed, hot.tier_rows[top] * ENCODED_ROW_BYTES as u64);
        assert_eq!(hot.resident_bytes(), before - freed);
        assert_eq!(hot.released_tiers(), 1);
        let guard = hot.read();
        assert_eq!(hot.get(&guard, victim), None);
        assert!(hot.get(&guard, other).is_some());
        drop(guard);
        let mut total = freed;
        loop {
            let bytes = hot.release_one();
            if bytes == 0 {
                break;
            }
            total += bytes;
        }
        assert_eq!(hot.resident_bytes(), 0);
        assert_eq!(total, before);
        assert_eq!(hot.release_lowest(), 0);
    }

    /// Budget arithmetic: K = floor(B/170*256/rows), tiers = ceil(K/G), and a
    /// budget below one bucket is a 0-tier no-op store.
    #[test]
    fn budget_selects_kept_buckets_and_tiers() {
        let rank = |seed| PleHotRank::from_buckets(ROWS as u64, FINGERPRINT, synthetic_buckets(ROWS, seed));
        let row = ENCODED_ROW_BYTES as u64;
        let small = HotRowStore::new(rank(1), row * ROWS as u64 / 256 - 1, ROWS_PER_SHARD, 170).unwrap();
        assert_eq!((small.kept_buckets(), small.tier_count()), (0, 0));
        assert_eq!(small.resident_bytes(), 0);
        assert!(small.get(&small.read(), 0).is_none());
        let full = HotRowStore::new(rank(1), u64::MAX / 2, ROWS_PER_SHARD, 170).unwrap();
        assert_eq!(full.kept_buckets(), 256);
        assert_eq!(full.tier_count(), 256usize.div_ceil(DEFAULT_TIER_BUCKETS));
        assert_eq!(full.resident_bytes(), ROWS as u64 * row);
        let geometry = HotRowStore::new(rank(1), 1 << 30, 999, 170);
        assert!(geometry.is_err());
        let row_size = HotRowStore::new(rank(1), 1 << 30, ROWS_PER_SHARD, 320);
        assert!(row_size.is_err());
    }

    /// Host-memory pressure during the fill releases the lowest tiers.
    #[test]
    fn fill_yields_tiers_under_memory_pressure() {
        let hot = store(256, 4, 11);
        let tiers = hot.tier_count();
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let (file, offsets) = layout_shards(&shards);
        hot.fill_with(
            &SliceReader(file),
            &offsets,
            &FillParams {
                floor_bytes: 1 << 30,
                pressure_interval_bytes: 1,
                avail: &|| Some(0),
                io_read_bytes: &|| None,
                adapt_interval: Duration::from_millis(20),
            },
        );
        assert_eq!(hot.released_tiers(), tiers);
        assert_eq!(hot.resident_bytes(), 0);
        assert!(!hot.fill_complete());
        assert_eq!(hot.filled_bytes(), 0, "released tiers subtract what they held");
    }

    /// `stop` ends a fill without marking it complete.
    #[test]
    fn stop_halts_the_fill() {
        let hot = store(128, 4, 5);
        hot.stop();
        fill_from_shards(&hot, &synthetic_shards(SHARDS, ROWS_PER_SHARD));
        assert!(!hot.fill_complete());
        let guard = hot.read();
        assert!((0..ROWS as u64).all(|row| hot.get(&guard, row).is_none()));
    }

    /// Before any fetch the fill runs at once; a fetch pauses it while its
    /// guard lives and for `BUSY_QUIET` after; a stop ends the pause.
    #[test]
    fn fill_waits_for_idle() {
        let hot = store(128, 2, 17);
        let started = Instant::now();
        assert!(hot.wait_until_idle());
        assert!(started.elapsed() < Duration::from_millis(100));

        let held = Duration::from_millis(200);
        let guard = hot.busy();
        let started = Instant::now();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                std::thread::sleep(held);
                drop(guard);
            });
            assert!(hot.wait_until_idle());
        });
        assert!(started.elapsed() >= held + BUSY_QUIET - Duration::from_millis(5));

        let _guard = hot.busy();
        hot.stop();
        assert!(!hot.wait_until_idle());
    }

    /// `filled_bytes` is the copied bytes of the tiers still present: equal to
    /// the resident bytes after a full fill, and reduced by each release.
    #[test]
    fn filled_bytes_tracks_the_fill_and_releases() {
        let hot = store(256, 4, 13);
        assert_eq!(hot.filled_bytes(), 0);
        fill_from_shards(&hot, &synthetic_shards(SHARDS, ROWS_PER_SHARD));
        assert!(hot.fill_complete());
        assert_eq!(hot.filled_bytes(), ROWS as u64 * ENCODED_ROW_BYTES as u64);
        assert_eq!(hot.filled_bytes(), hot.resident_bytes());
        let freed = hot.release_lowest();
        assert!(freed > 0);
        assert_eq!(hot.filled_bytes(), hot.resident_bytes());
        assert_eq!(hot.filled_bytes(), ROWS as u64 * ENCODED_ROW_BYTES as u64 - freed);
        while hot.release_one() > 0 {}
        assert_eq!(hot.filled_bytes(), 0);

        let partial = store(100, 3, 17);
        fill_from_shards(&partial, &synthetic_shards(SHARDS, ROWS_PER_SHARD));
        assert_eq!(partial.filled_bytes(), partial.resident_bytes());
        assert!(partial.filled_bytes() > 0);
    }

    /// The bucket grants no more than `rate` bytes per second (fake clock):
    /// 5000 B at 1000 B/s take 5 s, the burst is capped, a rate change and a
    /// zero rate apply at once.
    #[test]
    fn token_bucket_limits_the_rate() {
        let mut bucket = BucketState::new(1000.0, 400.0);
        let mut now = 0.0;
        for _ in 0..50 {
            loop {
                match bucket.try_take(now, 100.0) {
                    Take::Granted => break,
                    Take::Wait(seconds) => {
                        assert!(seconds > 0.0 && seconds <= 0.1 + 1e-9, "{seconds}");
                        now += seconds;
                    }
                    Take::Blocked => panic!("blocked at a positive rate"),
                }
            }
        }
        assert!((now - 5.0).abs() < 1e-6, "5000 B at 1000 B/s took {now} s");

        // Idle for a long time: the balance is capped at the burst.
        now += 100.0;
        assert_eq!(bucket.try_take(now, 400.0), Take::Granted);
        assert_eq!(bucket.try_take(now, 100.0), Take::Wait(0.1));
        // A request above the burst is clamped to it instead of waiting forever.
        now += 100.0;
        assert_eq!(bucket.try_take(now, 10_000.0), Take::Granted);

        // Rate change: a larger rate shortens the wait; zero blocks.
        bucket.set_rate(now, 2000.0);
        assert_eq!(bucket.try_take(now, 100.0), Take::Wait(0.05));
        bucket.set_rate(now, 0.0);
        assert_eq!(bucket.try_take(now + 50.0, 100.0), Take::Blocked);
    }

    /// The adaptive rule: ramp +25% while the load reads at pace, halve (down
    /// to the floor) when it falls under 85% of its peak, hold while it is
    /// not reading.
    #[test]
    fn adaptive_rate_ramps_halves_and_holds() {
        let (min, max) = (200e6, 800e6);
        let mut rate = AdaptiveRate::new(min, min, max);
        assert_eq!(rate.step(1.4e9), 250e6);
        let mut previous = 250e6;
        for _ in 0..20 {
            let next = rate.step(1.4e9);
            assert!(next >= previous && next <= max);
            previous = next;
        }
        assert_eq!(previous, max, "ramp stops at the ceiling");
        // Slowed but still reading: halve.
        assert_eq!(rate.step(1.0e9), 400e6);
        // Within 85% of the peak: ramp again.
        assert_eq!(rate.step(1.3e9), 500e6);
        // Not reading (GPU upload phase): hold, however low.
        assert_eq!(rate.step(50e6), 500e6);
        assert_eq!(rate.step(0.0), 500e6);
        // Persistently slowed: halve to the floor, never below.
        for _ in 0..6 {
            rate.step(0.5e9);
        }
        assert_eq!(rate.rate, min);
    }

    /// A throttled fill reads no faster than its rate.
    #[test]
    fn throttled_fill_respects_the_rate() {
        let hot = store(256, 4, 19);
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let rate = 4_000_000u64;
        hot.configure_throttle(FillThrottle {
            load_bytes_per_s: rate,
            min_bytes_per_s: rate,
            adaptive: false,
        });
        let started = Instant::now();
        let (file, offsets) = layout_shards(&shards);
        let none = || -> Option<u64> { None };
        hot.fill_with(
            &SliceReader(file),
            &offsets,
            &FillParams {
                floor_bytes: 0,
                pressure_interval_bytes: u64::MAX,
                avail: &none,
                io_read_bytes: &none,
                adapt_interval: Duration::from_millis(20),
            },
        );
        let elapsed = started.elapsed().as_secs_f64();
        let bytes = hot.bytes_read.load(Ordering::Relaxed) as f64;
        assert!(hot.fill_complete());
        assert!(bytes > 1.0e6, "{bytes}");
        // The bucket starts empty: every byte read was granted by the rate.
        assert!(elapsed >= 0.95 * bytes / rate as f64, "{bytes} B in {elapsed} s at {rate} B/s");
        assert_eq!(hot.filled_bytes(), hot.resident_bytes());
    }

    /// A 0-rate fill reads nothing until `release_throttle`, then completes;
    /// releasing twice is harmless.
    #[test]
    fn release_throttle_unblocks_a_zero_rate_fill() {
        let hot = store(256, 4, 23);
        hot.configure_throttle(FillThrottle {
            load_bytes_per_s: 0,
            min_bytes_per_s: 200_000_000,
            adaptive: true,
        });
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let (file, offsets) = layout_shards(&shards);
        let reader = SliceReader(file);
        let none = || -> Option<u64> { None };
        let params = FillParams {
            floor_bytes: 0,
            pressure_interval_bytes: u64::MAX,
            avail: &none,
            io_read_bytes: &none,
            adapt_interval: Duration::from_millis(20),
        };
        std::thread::scope(|scope| {
            let fill = scope.spawn(|| hot.fill_with(&reader, &offsets, &params));
            std::thread::sleep(Duration::from_millis(200));
            assert_eq!(hot.bytes_read.load(Ordering::Relaxed), 0, "no read before release");
            assert_eq!(hot.filled_bytes(), 0);
            assert!(!hot.fill_complete());
            hot.release_throttle();
            fill.join().unwrap();
        });
        hot.release_throttle();
        assert!(hot.fill_complete());
        assert_eq!(hot.filled_bytes(), ROWS as u64 * ENCODED_ROW_BYTES as u64);
    }

    fn write_rank(path: &Path, buckets: &[u8]) {
        write_ple_hot_rank(
            path,
            buckets.len() as u64,
            FINGERPRINT,
            buckets,
            serde_json::json!({"window_tokens": 8192, "generator": "test"}),
        )
        .unwrap();
    }

    /// Sidecar round trip, plus rejection of a bad fingerprint, row count,
    /// magic, truncation, and a payload whose sha does not match.
    #[test]
    fn sidecar_round_trip_and_rejections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.mq4.plehot");
        let buckets = synthetic_buckets(ROWS, 9);
        write_rank(&path, &buckets);
        assert!(!dir.path().read_dir().unwrap().any(|e| {
            e.unwrap().file_name().to_string_lossy().contains(".tmp.")
        }));

        let rank = PleHotRank::open(&path, ROWS as u64, FINGERPRINT).unwrap();
        assert_eq!(rank.padded_rows(), ROWS as u64);
        assert_eq!(rank.buckets(), buckets.as_slice());
        assert_eq!(rank.header()["rank"]["generator"], "test");
        assert_eq!(rank.header()["version"], 1);
        rank.verify().unwrap();
        assert!(PleHotRank::open(&path, ROWS as u64, &FINGERPRINT.to_ascii_uppercase()).is_ok());

        let other = "ff".repeat(32);
        let error = PleHotRank::open(&path, ROWS as u64, &other).err().unwrap();
        assert!(error.contains("fingerprint"), "{error}");
        let error = PleHotRank::open(&path, ROWS as u64 + 1, FINGERPRINT).err().unwrap();
        assert!(error.contains("padded_rows"), "{error}");

        let raw = std::fs::read(&path).unwrap();
        let tampered = dir.path().join("tampered.plehot");
        let mut bad = raw.clone();
        *bad.last_mut().unwrap() ^= 1;
        std::fs::write(&tampered, &bad).unwrap();
        let rank = PleHotRank::open(&tampered, ROWS as u64, FINGERPRINT).unwrap();
        assert!(rank.verify().unwrap_err().contains("sha256"));

        let magic = dir.path().join("magic.plehot");
        let mut bad = raw.clone();
        bad[0] = b'X';
        std::fs::write(&magic, &bad).unwrap();
        assert!(PleHotRank::open(&magic, ROWS as u64, FINGERPRINT).err().unwrap().contains("magic"));

        let short = dir.path().join("short.plehot");
        std::fs::write(&short, &raw[..raw.len() - 1]).unwrap();
        assert!(PleHotRank::open(&short, ROWS as u64, FINGERPRINT).is_err());

        assert!(write_ple_hot_rank(&path, ROWS as u64 + 1, FINGERPRINT, &buckets, serde_json::json!({})).is_err());
        assert!(write_ple_hot_rank(&path, ROWS as u64, "xyz", &buckets, serde_json::json!({})).is_err());
    }

    /// Through the real thread + file path: a tampered sidecar releases the
    /// whole store, a good one fills it with the model file's bytes.
    #[test]
    fn spawn_fill_verifies_sha_and_reads_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let buckets = synthetic_buckets(ROWS, 21);
        let shards = synthetic_shards(SHARDS, ROWS_PER_SHARD);
        let (file_bytes, offsets) = layout_shards(&shards);
        let model = dir.path().join("model.bin");
        std::fs::write(&model, &file_bytes).unwrap();
        let budget = ROWS as u64 * ENCODED_ROW_BYTES as u64 * 100 / 256;

        let good = dir.path().join("good.plehot");
        write_rank(&good, &buckets);
        let rank = PleHotRank::open(&good, ROWS as u64, FINGERPRINT).unwrap();
        let hot = HotRowStore::new_with_group(rank, budget, ROWS_PER_SHARD, 170, 4).unwrap();
        hot.spawn_fill(&model, offsets.clone(), 0, fast_throttle()).join().unwrap();
        assert_eq!(hot.sha_verified(), Some(true));
        assert!(hot.fill_complete());
        let guard = hot.read();
        let mut served = 0;
        for row in 0..ROWS as u64 {
            if let Some(bytes) = hot.get(&guard, row) {
                let r = row as usize;
                let want = &shards[r / ROWS_PER_SHARD][(r % ROWS_PER_SHARD) * 170..][..170];
                assert_eq!(bytes, want);
                served += 1;
            }
        }
        drop(guard);
        assert!(served > 0);

        let bad = dir.path().join("bad.plehot");
        let mut raw = std::fs::read(&good).unwrap();
        *raw.last_mut().unwrap() ^= 0x40;
        std::fs::write(&bad, &raw).unwrap();
        let rank = PleHotRank::open(&bad, ROWS as u64, FINGERPRINT).unwrap();
        let hot = HotRowStore::new_with_group(rank, budget, ROWS_PER_SHARD, 170, 4).unwrap();
        assert!(hot.tier_count() > 0);
        hot.spawn_fill(&model, offsets, 0, fast_throttle()).join().unwrap();
        assert_eq!(hot.sha_verified(), Some(false));
        assert!(!hot.fill_complete());
        assert_eq!(hot.resident_bytes(), 0);
        assert!((0..ROWS as u64).all(|row| hot.get(&hot.read(), row).is_none()));
    }
}
