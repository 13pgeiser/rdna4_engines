// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PLE prewarm of a row-order PLE sidecar file, as part of model load.
//!
//! When the PLE rows live in a `.ple-q4` / `.ple-q8` sidecar (and no hot-row
//! pack serves them), reading the sidecar's data region sequentially once puts
//! it in the kernel page cache, so the scattered row reads of a request hit RAM
//! instead of the SSD.  The pages are ordinary clean file pages: under memory
//! pressure the kernel drops them rather than swapping.
//!
//! [`PleRowsPrewarm::spawn`] starts with the model load and reads
//! `[data_offset, data_offset + min(budget, data_bytes))` in 8 MiB chunks with
//! plain buffered `pread`s into one reused buffer (the data is discarded),
//! concurrently with the weight upload, at no more than `bytes_per_sec` when
//! that is nonzero.  The upload streams the model through the same page cache
//! and can evict what was read, so once the loader calls
//! [`PleRowsPrewarm::finish_load`] (the weights are uploaded) the worker
//! re-reads every evicted page of the budget (`mincore`) and only then is the
//! prewarm complete.  The loader declares the model ready after
//! [`PleRowsPrewarm::wait`]: nothing of it runs while requests are served.
//! Dropping the handle stops and joins the worker.
//!
//! # Public API (exact signatures)
//!
//! ```text
//! PleRowsPrewarm::spawn(path: &Path, data_offset: u64, data_bytes: u64, budget: u64,
//!                       bytes_per_sec: u64) -> std::io::Result<PleRowsPrewarm>
//! PleRowsPrewarm::{finish_load(), wait() -> bool, stop(), prewarmed_bytes() -> u64,
//!                  complete() -> bool, failed() -> bool}
//! ```
//!
//! Log lines (stderr): `  qwen4 ple prewarm complete: <MiB> MiB ...`,
//! `  qwen4 ple prewarm stopped, ...`, `  qwen4 ple prewarm failed, ...`.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::ple_hot_store::Residency;

/// Bytes per prewarm read.
const PREWARM_CHUNK: usize = 8 << 20;
const TAG: &str = "  qwen4 ple prewarm";

fn mib(bytes: u64) -> u64 {
    bytes >> 20
}

struct Shared {
    data_offset: u64,
    budget: u64,
    bytes_per_sec: u64,
    prewarmed: AtomicU64,
    upload_done: AtomicBool,
    done: AtomicBool,
    failed: AtomicBool,
    ended: AtomicBool,
    stop: AtomicBool,
}

/// Load-time reader that warms the page cache with a sidecar's rows.  Stops
/// and joins its worker on drop.
pub struct PleRowsPrewarm {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl PleRowsPrewarm {
    /// Open `path` and start reading the first `budget` bytes (capped at
    /// `data_bytes`) of its data region `[data_offset, data_offset +
    /// data_bytes)` at once, at no more than `bytes_per_sec` (0 =
    /// unthrottled).
    pub fn spawn(
        path: &Path,
        data_offset: u64,
        data_bytes: u64,
        budget: u64,
        bytes_per_sec: u64,
    ) -> io::Result<PleRowsPrewarm> {
        data_offset.checked_add(data_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "prewarm range overflows u64")
        })?;
        let file = File::open(path)?;
        // Plain buffered reads; the hint only makes the kernel read ahead.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
        let shared = Arc::new(Shared {
            data_offset,
            budget: budget.min(data_bytes),
            bytes_per_sec,
            prewarmed: AtomicU64::new(0),
            upload_done: AtomicBool::new(false),
            done: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("ple-rows-prewarm".to_string())
            .spawn(move || {
                worker_shared.run(&file);
                worker_shared.ended.store(true, Ordering::Release);
            })?;
        Ok(PleRowsPrewarm {
            shared,
            worker: Some(worker),
        })
    }

    /// The weights are uploaded: re-read what the upload evicted, then end.
    pub fn finish_load(&self) {
        self.shared.upload_done.store(true, Ordering::Release);
    }

    /// Block until the worker ended (after [`PleRowsPrewarm::finish_load`]);
    /// `true` when the whole budget is resident.
    pub fn wait(&self) -> bool {
        while !self.shared.ended.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(5));
        }
        self.complete()
    }

    /// Ask the worker to stop at its next chunk.  Dropping the handle does this
    /// and joins.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
    }

    /// Bytes of the data region read so far.
    pub fn prewarmed_bytes(&self) -> u64 {
        self.shared.prewarmed.load(Ordering::Relaxed)
    }

    /// The whole budget was read and, after the upload, found resident.
    pub fn complete(&self) -> bool {
        self.shared.done.load(Ordering::Acquire)
    }

    /// A read failed and the worker gave up.
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Acquire)
    }
}

impl Drop for PleRowsPrewarm {
    fn drop(&mut self) {
        self.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Shared {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// Read `[at, at + len)` (absolute); `false` after logging a failure.
    fn read(&self, file: &File, buffer: &mut [u8], at: u64, len: usize) -> bool {
        if let Err(error) = file.read_exact_at(&mut buffer[..len], at) {
            self.failed.store(true, Ordering::Release);
            eprintln!(
                "{TAG} failed, read error at {} MiB of {} MiB: {error}",
                mib(at - self.data_offset),
                mib(self.budget)
            );
            return false;
        }
        true
    }

    fn run(&self, file: &File) {
        let budget = self.budget;
        if budget == 0 {
            self.done.store(true, Ordering::Release);
            return;
        }
        let started = Instant::now();
        let mut buffer = vec![0u8; PREWARM_CHUNK];
        let begin = self.data_offset;
        let end = begin + budget;
        let mut offset = begin;
        while offset < end {
            if self.stopped() {
                eprintln!(
                    "{TAG} stopped, {} MiB of {} MiB in {:.1} s",
                    mib(offset - begin),
                    mib(budget),
                    started.elapsed().as_secs_f64()
                );
                return;
            }
            if self.bytes_per_sec > 0 {
                let due = Duration::from_secs_f64((offset - begin) as f64 / self.bytes_per_sec as f64);
                let elapsed = started.elapsed();
                if due > elapsed {
                    std::thread::sleep((due - elapsed).min(Duration::from_millis(50)));
                    continue;
                }
            }
            let len = ((end - offset) as usize).min(PREWARM_CHUNK);
            if !self.read(file, &mut buffer, offset, len) {
                return;
            }
            offset += len as u64;
            self.prewarmed.store(offset - begin, Ordering::Relaxed);
        }
        let read_secs = started.elapsed().as_secs_f64();
        while !self.upload_done.load(Ordering::Acquire) {
            if self.stopped() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // Re-read the pages of the budget the upload evicted, hottest
        // (lowest offset) first, until a pass finds none.
        let mut repaired = 0u64;
        match Residency::map(file, end) {
            Ok(mut view) => {
                let mut from = begin;
                loop {
                    if self.stopped() {
                        return;
                    }
                    let hole = match view.first_hole(from, end) {
                        Ok(hole) => hole,
                        Err(error) => {
                            eprintln!("{TAG}: residency scan failed ({error}); not re-reading evicted pages");
                            break;
                        }
                    };
                    if hole >= end {
                        break;
                    }
                    let len = ((end - hole) as usize).min(PREWARM_CHUNK);
                    if !self.read(file, &mut buffer, hole, len) {
                        return;
                    }
                    repaired += len as u64;
                    if !matches!(view.first_hole(hole, hole + 1), Ok(next) if next > hole) {
                        eprintln!("{TAG}: the residency view is blind; not re-reading evicted pages");
                        break;
                    }
                    from = hole + len as u64;
                }
            }
            Err(error) => {
                eprintln!("{TAG}: no residency view ({error}); not re-reading evicted pages")
            }
        }
        self.done.store(true, Ordering::Release);
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "{TAG} complete: {} MiB resident in {elapsed:.1} s ({} MiB read in {read_secs:.1} s, \
             {:.2} GB/s; {} MiB re-reading evicted pages after the upload)",
            mib(budget),
            mib(budget),
            budget as f64 / 1e9 / read_secs.max(1e-9),
            mib(repaired),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const OFFSET: u64 = 4096;
    /// Two full chunks and a partial one.
    const DATA: u64 = 2 * PREWARM_CHUNK as u64 + (4 << 20);

    fn data_file(len: u64) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.ple-q4");
        let mut file = File::create(&path).unwrap();
        file.write_all(&vec![0xA5u8; len as usize]).unwrap();
        (dir, path)
    }

    fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn spawn(path: &Path, budget: u64) -> PleRowsPrewarm {
        PleRowsPrewarm::spawn(path, OFFSET, DATA, budget, 0).unwrap()
    }

    /// The prewarm reads its budget (capped at the data) at once, during the
    /// load, but completes only after the upload is done.
    #[test]
    fn reads_exactly_its_budget_and_completes_after_the_upload() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let budget = PREWARM_CHUNK as u64 + (1 << 20);
        let prewarm = spawn(&path, budget);
        wait_until("a partial budget", || prewarm.prewarmed_bytes() == budget);
        std::thread::sleep(Duration::from_millis(50));
        assert!(!prewarm.complete(), "not complete before the upload is done");
        prewarm.finish_load();
        assert!(prewarm.wait());
        assert_eq!(prewarm.prewarmed_bytes(), budget);
        assert!(!prewarm.failed());

        let all = spawn(&path, u64::MAX - 1);
        all.finish_load();
        assert!(all.wait());
        assert_eq!(all.prewarmed_bytes(), DATA);

        let none = spawn(&path, 0);
        assert!(none.wait(), "a zero budget is complete at once");
        assert_eq!(none.prewarmed_bytes(), 0);
    }

    /// A throttled prewarm outlasting the upload holds `wait` (ready) back
    /// until its whole budget is read.
    #[test]
    fn wait_holds_ready_until_a_throttled_budget_is_read() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let rate = 40_000_000;
        let prewarm = PleRowsPrewarm::spawn(&path, OFFSET, DATA, DATA, rate).unwrap();
        let load_start = Instant::now();
        prewarm.finish_load();
        assert!(prewarm.wait());
        let floor = Duration::from_secs_f64((DATA - PREWARM_CHUNK as u64) as f64 / rate as f64);
        assert!(load_start.elapsed() >= floor);
        assert_eq!(prewarm.prewarmed_bytes(), DATA);
    }

    /// Dropping the handle ends a worker that waits for the upload.
    #[test]
    fn drop_stops_and_joins() {
        let (_dir, path) = data_file(OFFSET + DATA);
        let waiting = spawn(&path, DATA);
        wait_until("the budget", || waiting.prewarmed_bytes() == DATA);
        let started = Instant::now();
        drop(waiting);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    /// A file shorter than the declared data region fails the worker instead of
    /// reporting completion.
    #[test]
    fn a_short_file_fails() {
        let (_dir, path) = data_file(OFFSET + PREWARM_CHUNK as u64 + 12345);
        let prewarm = spawn(&path, DATA);
        prewarm.finish_load();
        assert!(!prewarm.wait());
        assert!(prewarm.failed());
        assert_eq!(prewarm.prewarmed_bytes(), PREWARM_CHUNK as u64);
    }

    #[test]
    fn spawn_reports_a_missing_file_and_a_bad_range() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent");
        assert!(PleRowsPrewarm::spawn(&missing, 0, 1, 1, 0).is_err());
        let (_dir, path) = data_file(16);
        assert!(PleRowsPrewarm::spawn(&path, u64::MAX, 2, 1, 0).is_err());
    }
}
