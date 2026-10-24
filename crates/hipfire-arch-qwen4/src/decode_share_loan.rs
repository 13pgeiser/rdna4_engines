// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The donor-tail loan controller of decode-share (`fixed` / `on` modes).
//!
//! While a long prefill runs, the forward scratch lives on the donor tail (a
//! contiguous run of cold resident experts' VRAM). The controller is the one
//! state machine that decides when those bytes may be overwritten and when
//! they must be whole again:
//!
//! ```text
//!   Resident --acquire--> Dirty --finish_prompt / require_resident--> Restoring --wait--> Resident
//!        any state --(error)--> recovery (device drain + blocking copy) --> Resident | Poisoned
//! ```
//!
//! It is pure: every device effect goes through [`LoanOps`], so each
//! transition is unit-tested on the CPU with a recording fake. The forward's
//! `GpuLoanOps` (gpu_forward.rs) is the real implementation: the compute
//! stream, a dedicated restore stream, two events and the pinned mirror.
//!
//! Invariants:
//! * No byte of the donor tail is written before the state is `Dirty`.
//! * `Resident` is entered only after the restore copy completed (waited, or
//!   a blocking recovery copy), never on a promise.
//! * A failed recovery, or any need to restore under a stream capture,
//!   poisons the loan: every later call fails and names the reason.

use std::time::Instant;

/// Device effects of the loan protocol. Every method is one real operation;
/// errors are plain text (the forward wraps them).
pub(crate) trait LoanOps {
    /// Whether the compute stream is being captured (a capture can neither
    /// run a restore nor host-wait on one).
    fn capturing(&mut self) -> Result<bool, String>;
    /// Enqueue a zero fill of `bytes` at `offset` into the donor tail on the
    /// compute stream.
    fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String>;
    /// Enqueue the restore of the first `bytes` of the tail from its mirror:
    /// event A on the compute stream, the restore stream waits A, async H2D
    /// copy, `done` recorded on the restore stream, and the compute stream
    /// waits `done`.
    fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String>;
    /// Host-wait for the enqueued restore's `done` event and check the async
    /// error state.
    fn wait_restore(&mut self) -> Result<(), String>;
    /// Drain every queue of the device.
    fn device_sync(&mut self) -> Result<(), String>;
    /// Blocking H2D copy of the first `bytes` of the tail from its mirror.
    fn blocking_restore(&mut self, bytes: u64) -> Result<(), String>;
    /// The `HIPFIRE_QWEN4_DECODE_SHARE_VERIFY` oracle over the first `bytes`
    /// (a no-op unless the knob is set).
    fn verify(&mut self, bytes: u64) -> Result<(), String>;
}

/// Where the donor tail stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LoanState {
    /// The tail holds its experts.
    Resident,
    /// Scratch may have overwritten the first `dirty_bytes`; the first
    /// `zeroed_bytes` were zero-filled for this epoch.
    Dirty {
        epoch: u64,
        dirty_bytes: u64,
        zeroed_bytes: u64,
    },
    /// The restore of `bytes` is enqueued (the `done` event lives in the ops).
    Restoring {
        epoch: u64,
        bytes: u64,
        zeroed_bytes: u64,
    },
    /// The tail's contents are unknown; nothing may run.
    Poisoned(String),
}

/// Receipt counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoanCounters {
    pub epochs: u64,
    pub tiles: u64,
    pub restores: u64,
    pub restored_bytes: u64,
    pub recoveries: u64,
    pub zeroed_bytes: u64,
    /// Time actually blocked in the host wait for a restore.
    pub wait_ns: u64,
}

pub(crate) struct LoanController {
    state: LoanState,
    epoch: u64,
    epoch_tiles: u64,
    /// `HIPFIRE_QWEN4_DECODE_SHARE_SYNC`: host-wait right after each enqueue.
    sync_restore: bool,
    /// `HIPFIRE_QWEN4_DECODE_SHARE_LOG`: one line per epoch.
    log_epochs: bool,
    counters: LoanCounters,
}

const MIB: u64 = 1 << 20;

impl LoanController {
    pub(crate) fn new(sync_restore: bool, log_epochs: bool) -> Self {
        Self {
            state: LoanState::Resident,
            epoch: 0,
            epoch_tiles: 0,
            sync_restore,
            log_epochs,
            counters: LoanCounters::default(),
        }
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> &LoanState {
        &self.state
    }

    pub(crate) fn is_resident(&self) -> bool {
        self.state == LoanState::Resident
    }

    #[cfg(test)]
    pub(crate) fn counters(&self) -> LoanCounters {
        self.counters
    }

    fn poison_error(reason: &str) -> String {
        format!("decode-share donor tail is poisoned ({reason}); reload the model")
    }

    /// `Err` naming the reason when poisoned.
    pub(crate) fn check(&self) -> Result<(), String> {
        match &self.state {
            LoanState::Poisoned(reason) => Err(Self::poison_error(reason)),
            _ => Ok(()),
        }
    }

    fn poison(&mut self, reason: String) -> String {
        let error = Self::poison_error(&reason);
        self.state = LoanState::Poisoned(reason);
        error
    }

    /// Bytes a recovery must copy back from the current state.
    fn pending_bytes(&self) -> u64 {
        match self.state {
            LoanState::Dirty { dirty_bytes, .. } => dirty_bytes,
            LoanState::Restoring { bytes, .. } => bytes,
            _ => 0,
        }
    }

    /// Recovery: drain every queue, copy `bytes` back with a blocking copy,
    /// drain again, and only then mark the tail `Resident`. Under a capture,
    /// or if any step fails, the loan is poisoned. Returns `original` with the
    /// recovery failure appended, if any.
    fn recover(&mut self, ops: &mut dyn LoanOps, bytes: u64, original: String) -> String {
        self.counters.recoveries += 1;
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) => {
                let why = self.poison(format!(
                    "{original}; the donor tail needs a restore that a stream capture cannot run"
                ));
                return format!("{original}; {why}");
            }
            Err(error) => {
                let why = self.poison(format!("{original}; capture query failed: {error}"));
                return format!("{original}; {why}");
            }
        }
        let outcome = ops
            .device_sync()
            .and_then(|()| ops.blocking_restore(bytes))
            .and_then(|()| ops.device_sync());
        match outcome {
            Ok(()) => {
                self.counters.restored_bytes += bytes;
                self.state = LoanState::Resident;
                original
            }
            Err(error) => {
                let why = self.poison(format!("{original}; recovery failed: {error}"));
                format!("{original}; {why}")
            }
        }
    }

    /// Enqueue the restore of `bytes` and move to `Restoring`.
    fn enqueue(
        &mut self,
        ops: &mut dyn LoanOps,
        bytes: u64,
        zeroed_bytes: u64,
    ) -> Result<(), String> {
        match ops.enqueue_restore(bytes) {
            Ok(()) => {
                self.counters.restores += 1;
                self.counters.restored_bytes += bytes;
                self.state = LoanState::Restoring {
                    epoch: self.epoch,
                    bytes,
                    zeroed_bytes,
                };
                Ok(())
            }
            Err(error) => Err(self.recover(ops, bytes, format!("enqueue donor restore: {error}"))),
        }
    }

    /// Host-wait the enqueued restore: `Restoring` -> `Resident`.
    fn wait(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        let LoanState::Restoring {
            epoch,
            bytes,
            zeroed_bytes,
        } = self.state
        else {
            return Ok(());
        };
        let started = Instant::now();
        let waited = ops.wait_restore();
        let wait_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.counters.wait_ns = self.counters.wait_ns.saturating_add(wait_ns);
        if let Err(error) = waited {
            return Err(self.recover(ops, bytes, format!("wait donor restore: {error}")));
        }
        if let Err(error) = ops.verify(bytes) {
            let reason = format!("donor tail verification after restore: {error}");
            return Err(self.poison(reason));
        }
        self.state = LoanState::Resident;
        if self.log_epochs {
            eprintln!(
                "  decode-share epoch {epoch}: tiles {}, class span {} MiB, zeroed {} MiB, restore {} MiB, wait {:.3} ms",
                self.epoch_tiles,
                bytes / MIB,
                zeroed_bytes / MIB,
                bytes / MIB,
                wait_ns as f64 / 1.0e6
            );
        }
        Ok(())
    }

    /// A borrowed prompt tile whose scratch spans `span` bytes of the tail is
    /// about to run. Writes nothing unless the state is (made) `Dirty` first;
    /// zero-fills only the growth beyond what this epoch already zeroed.
    pub(crate) fn acquire(&mut self, ops: &mut dyn LoanOps, span: u64) -> Result<(), String> {
        self.check()?;
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) => {
                return Err(
                    "a decode-share prompt tile cannot run inside a stream capture (its scratch \
                     overwrites donor experts that need a restore)"
                        .to_string(),
                )
            }
            Err(error) => return Err(format!("decode-share capture query: {error}")),
        }
        if matches!(self.state, LoanState::Restoring { .. }) {
            self.wait(ops)?;
        }
        if self.state == LoanState::Resident {
            self.epoch += 1;
            self.epoch_tiles = 0;
            self.counters.epochs += 1;
            self.state = LoanState::Dirty {
                epoch: self.epoch,
                dirty_bytes: 0,
                zeroed_bytes: 0,
            };
        }
        let LoanState::Dirty {
            epoch,
            dirty_bytes,
            zeroed_bytes,
        } = self.state
        else {
            return Err(format!("decode-share acquire in state {:?}", self.state));
        };
        // The write may land as soon as it is enqueued: record the extent first.
        let dirty_bytes = dirty_bytes.max(span);
        self.state = LoanState::Dirty {
            epoch,
            dirty_bytes,
            zeroed_bytes,
        };
        if span > zeroed_bytes {
            // Donor bytes are arbitrary bit patterns (NaNs among them): padded
            // rows of a scratch must read zeros, as in a zero-initialized allocation.
            if let Err(error) = ops.memset_zero(zeroed_bytes, span - zeroed_bytes) {
                return Err(self.recover(ops, dirty_bytes, format!("zero donor scratch: {error}")));
            }
            self.counters.zeroed_bytes += span - zeroed_bytes;
            self.state = LoanState::Dirty {
                epoch,
                dirty_bytes,
                zeroed_bytes: span,
            };
        }
        self.epoch_tiles += 1;
        self.counters.tiles += 1;
        Ok(())
    }

    /// A call that runs on the resident bank and the Resident runtimes
    /// (decode, verify, lanes, a tile below 512 rows) is about to start: the
    /// donors must be whole. Enqueues the restore if dirty, then host-waits.
    pub(crate) fn require_resident(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        self.check()?;
        if let LoanState::Dirty {
            dirty_bytes,
            zeroed_bytes,
            ..
        } = self.state
        {
            match ops.capturing() {
                Ok(false) => {}
                Ok(true) | Err(_) => {
                    return Err(self.poison(
                        "a resident call under a stream capture found the donor tail dirty"
                            .to_string(),
                    ))
                }
            }
            self.enqueue(ops, dirty_bytes, zeroed_bytes)?;
        }
        self.wait(ops)
    }

    /// After every tile with the resident bank back in place: an `Err` tile
    /// must not leave dirty weights behind even if the caller abandons the
    /// prompt. Returns `Err` only if the recovery itself failed.
    pub(crate) fn after_tile(&mut self, ops: &mut dyn LoanOps, ok: bool) -> Result<(), String> {
        if ok {
            return Ok(());
        }
        let bytes = self.pending_bytes();
        if matches!(
            self.state,
            LoanState::Dirty { .. } | LoanState::Restoring { .. }
        ) {
            let text = self.recover(ops, bytes, "decode-share tile failed".to_string());
            if let LoanState::Poisoned(_) = self.state {
                return Err(text);
            }
        }
        Ok(())
    }

    /// The outermost whole-prompt entry ended (`ok`: it returned `Ok`). A dirty
    /// tail is restored asynchronously (the host wait is deferred to the next
    /// call that needs the donors, so it overlaps sampling and emission); an
    /// abandoned prompt takes the blocking recovery.
    pub(crate) fn finish_prompt(&mut self, ops: &mut dyn LoanOps, ok: bool) -> Result<(), String> {
        let LoanState::Dirty {
            dirty_bytes,
            zeroed_bytes,
            ..
        } = self.state
        else {
            return Ok(());
        };
        if !ok {
            let text = self.recover(ops, dirty_bytes, "decode-share prompt failed".to_string());
            return match self.state {
                LoanState::Poisoned(_) => Err(text),
                _ => Ok(()),
            };
        }
        match ops.capturing() {
            Ok(false) => {}
            Ok(true) | Err(_) => {
                return Err(self.poison(
                    "a decode-share prompt ended under a stream capture with a dirty donor tail"
                        .to_string(),
                ))
            }
        }
        self.enqueue(ops, dirty_bytes, zeroed_bytes)?;
        if self.sync_restore {
            self.wait(ops)?;
        }
        Ok(())
    }

    /// Teardown: the donors must be whole before anything the lease owns is
    /// freed. `Err` means the restore could not be proven complete and the
    /// caller must leak the lease and the restore resources, not free them
    /// under a possibly pending DMA.
    pub(crate) fn teardown(&mut self, ops: &mut dyn LoanOps) -> Result<(), String> {
        let outcome = self.require_resident(ops);
        // A recovered tail is whole (a blocking copy completed after a drain):
        // safe to release even though the fast path failed.
        if self.is_resident() {
            Ok(())
        } else {
            outcome.and(Err(format!(
                "decode-share teardown left the donor tail in state {:?}",
                self.state
            )))
        }
    }

    /// One summary line for the free log.
    pub(crate) fn summary(&self) -> String {
        let counters = self.counters;
        format!(
            "{} epochs, {} prompt tiles, {} restores ({} MiB), {} recoveries, {} MiB zeroed, host wait {:.3} ms",
            counters.epochs,
            counters.tiles,
            counters.restores,
            counters.restored_bytes / MIB,
            counters.recoveries,
            counters.zeroed_bytes / MIB,
            counters.wait_ns as f64 / 1.0e6
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recording fake: every call is logged; one named failure and the
    /// capture flag are injected.
    #[derive(Default)]
    struct Fake {
        log: Vec<String>,
        fail: Option<&'static str>,
        capturing: bool,
        /// A failure that stays after the first hit.
        sticky: bool,
    }

    impl Fake {
        fn hit(&mut self, name: &'static str, text: String) -> Result<(), String> {
            self.log.push(text);
            if self.fail == Some(name) {
                if !self.sticky {
                    self.fail = None;
                }
                Err(format!("{name} failed"))
            } else {
                Ok(())
            }
        }
    }

    impl LoanOps for Fake {
        fn capturing(&mut self) -> Result<bool, String> {
            self.log.push("capturing".to_string());
            if self.fail == Some("capturing") {
                return Err("capturing failed".to_string());
            }
            Ok(self.capturing)
        }
        fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String> {
            self.hit("memset", format!("memset {offset}+{bytes}"))
        }
        fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("enqueue", format!("enqueue {bytes}"))
        }
        fn wait_restore(&mut self) -> Result<(), String> {
            self.hit("wait", "wait".to_string())
        }
        fn device_sync(&mut self) -> Result<(), String> {
            self.hit("sync", "sync".to_string())
        }
        fn blocking_restore(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("blocking", format!("blocking {bytes}"))
        }
        fn verify(&mut self, bytes: u64) -> Result<(), String> {
            self.hit("verify", format!("verify {bytes}"))
        }
    }

    fn writes(fake: &Fake) -> Vec<&String> {
        fake.log
            .iter()
            .filter(|line| line.starts_with("memset") || line.starts_with("enqueue"))
            .collect()
    }

    fn controller() -> LoanController {
        LoanController::new(false, false)
    }

    #[test]
    fn acquire_from_resident_enters_dirty_before_the_first_write_and_zeroes_the_span() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 1000,
                zeroed_bytes: 1000
            }
        );
        assert_eq!(fake.log, vec!["capturing", "memset 0+1000"]);
        assert_eq!(ctl.counters().epochs, 1);
        assert_eq!(ctl.counters().zeroed_bytes, 1000);
        assert_eq!(ctl.counters().tiles, 1);
    }

    #[test]
    fn class_growth_zeroes_only_the_delta_and_a_smaller_class_nothing() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 1000).unwrap();
        ctl.acquire(&mut fake, 2500).unwrap();
        ctl.acquire(&mut fake, 400).unwrap();
        let memsets: Vec<_> = fake
            .log
            .iter()
            .filter(|line| line.starts_with("memset"))
            .collect();
        assert_eq!(memsets, ["memset 0+1000", "memset 1000+1500"]);
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 1,
                dirty_bytes: 2500,
                zeroed_bytes: 2500
            }
        );
        assert_eq!(ctl.counters().tiles, 3);
        assert_eq!(ctl.counters().zeroed_bytes, 2500);
        assert_eq!(ctl.counters().epochs, 1, "one epoch for three tiles");
    }

    #[test]
    fn finish_prompt_enqueues_the_dirty_prefix_and_defers_the_host_wait() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 4096).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert_eq!(
            *ctl.state(),
            LoanState::Restoring {
                epoch: 1,
                bytes: 4096,
                zeroed_bytes: 4096
            }
        );
        assert!(!fake.log.contains(&"wait".to_string()), "no host wait yet");
        assert_eq!(ctl.counters().restores, 1);
        assert_eq!(ctl.counters().restored_bytes, 4096);
        // The next resident call waits, then runs.
        ctl.require_resident(&mut fake).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(fake.log.last().map(String::as_str), Some("verify 4096"));
        assert!(fake.log.contains(&"wait".to_string()));
    }

    #[test]
    fn sync_knob_waits_right_after_the_enqueue() {
        let mut ctl = LoanController::new(true, false);
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 64).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert!(ctl.is_resident());
        let enqueue = fake.log.iter().position(|l| l == "enqueue 64").unwrap();
        let wait = fake.log.iter().position(|l| l == "wait").unwrap();
        assert!(enqueue < wait);
    }

    #[test]
    fn require_resident_from_each_state() {
        // Resident: nothing.
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.require_resident(&mut fake).unwrap();
        assert!(fake.log.is_empty());
        // Dirty: enqueue then wait.
        ctl.acquire(&mut fake, 512).unwrap();
        fake.log.clear();
        ctl.require_resident(&mut fake).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(fake.log, vec!["capturing", "enqueue 512", "wait", "verify 512"]);
        // Restoring: wait only.
        ctl.acquire(&mut fake, 512).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.log.clear();
        ctl.require_resident(&mut fake).unwrap();
        assert_eq!(fake.log, vec!["wait", "verify 512"]);
        assert!(ctl.is_resident());
    }

    #[test]
    fn acquire_from_restoring_waits_first_and_starts_a_new_epoch() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 100).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.log.clear();
        ctl.acquire(&mut fake, 100).unwrap();
        assert_eq!(
            fake.log,
            vec!["capturing", "wait", "verify 100", "memset 0+100"]
        );
        assert_eq!(
            *ctl.state(),
            LoanState::Dirty {
                epoch: 2,
                dirty_bytes: 100,
                zeroed_bytes: 100
            }
        );
        assert_eq!(ctl.counters().epochs, 2);
    }

    #[test]
    fn capture_refuses_a_borrowed_tile_before_any_write() {
        let mut ctl = controller();
        let mut fake = Fake {
            capturing: true,
            ..Fake::default()
        };
        let error = ctl.acquire(&mut fake, 100).unwrap_err();
        assert!(error.contains("stream capture"), "{error}");
        assert!(writes(&fake).is_empty());
        assert!(ctl.is_resident(), "no state change either");
        assert_eq!(ctl.counters().epochs, 0);
    }

    #[test]
    fn capture_with_a_dirty_tail_poisons_instead_of_restoring() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 100).unwrap();
        fake.capturing = true;
        fake.log.clear();
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
        assert!(writes(&fake).is_empty());
    }

    #[test]
    fn tile_error_recovers_with_a_blocking_copy_and_returns_resident() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.log.clear();
        ctl.after_tile(&mut fake, false).unwrap();
        assert!(ctl.is_resident());
        assert_eq!(
            fake.log,
            vec!["capturing", "sync", "blocking 2048", "sync"],
            "drain, blocking copy, drain, then Resident"
        );
        assert_eq!(ctl.counters().recoveries, 1);
        // A successful tile never recovers.
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.log.clear();
        ctl.after_tile(&mut fake, true).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn failed_recovery_poisons_and_everything_then_refuses() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 2048).unwrap();
        fake.fail = Some("blocking");
        let error = ctl.after_tile(&mut fake, false).unwrap_err();
        assert!(error.contains("recovery failed"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
        fake.log.clear();
        for result in [
            ctl.acquire(&mut fake, 10),
            ctl.require_resident(&mut fake),
            ctl.check(),
        ] {
            let error = result.unwrap_err();
            assert!(error.contains("poisoned") && error.contains("recovery failed"), "{error}");
        }
        assert!(fake.log.is_empty(), "a poisoned loan touches nothing");
        // finish_prompt and teardown of a poisoned loan do not restore either.
        ctl.finish_prompt(&mut fake, true).unwrap();
        assert!(ctl.teardown(&mut fake).is_err());
        assert!(fake.log.is_empty());
    }

    #[test]
    fn recovery_under_a_capture_poisons() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 64).unwrap();
        fake.capturing = true;
        let error = ctl.after_tile(&mut fake, false).unwrap_err();
        assert!(error.contains("capture"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn zero_fill_failure_recovers_the_extent_it_may_have_dirtied() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        fake.fail = Some("memset");
        let error = ctl.acquire(&mut fake, 300).unwrap_err();
        assert!(error.contains("zero donor scratch"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn enqueue_failure_recovers_and_keeps_the_original_error() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 300).unwrap();
        fake.fail = Some("enqueue");
        let error = ctl.finish_prompt(&mut fake, true).unwrap_err();
        assert!(error.contains("enqueue donor restore"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn wait_failure_recovers_the_restoring_extent() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 300).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.fail = Some("wait");
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("wait donor restore"), "{error}");
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 300".to_string()));
    }

    #[test]
    fn verify_failure_after_a_restore_poisons() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 128).unwrap();
        ctl.finish_prompt(&mut fake, true).unwrap();
        fake.fail = Some("verify");
        let error = ctl.require_resident(&mut fake).unwrap_err();
        assert!(error.contains("verification"), "{error}");
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn an_abandoned_prompt_recovers_instead_of_enqueuing() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 640).unwrap();
        fake.log.clear();
        ctl.finish_prompt(&mut fake, false).unwrap();
        assert!(ctl.is_resident());
        assert!(fake.log.contains(&"blocking 640".to_string()));
        assert!(!fake.log.iter().any(|l| l.starts_with("enqueue")));
        // Nothing dirty: finish is a no-op either way.
        fake.log.clear();
        ctl.finish_prompt(&mut fake, true).unwrap();
        ctl.finish_prompt(&mut fake, false).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn teardown_restores_before_release_and_reports_failure() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        ctl.acquire(&mut fake, 256).unwrap();
        fake.log.clear();
        ctl.teardown(&mut fake).unwrap();
        assert!(ctl.is_resident());
        let enqueue = fake.log.iter().position(|l| l == "enqueue 256").unwrap();
        let wait = fake.log.iter().position(|l| l == "wait").unwrap();
        assert!(enqueue < wait);
        // An enqueue that fails at teardown falls back to the blocking recovery
        // copy: the tail is whole, so teardown succeeds.
        let mut ctl = controller();
        ctl.acquire(&mut fake, 256).unwrap();
        fake.fail = Some("enqueue");
        fake.sticky = true;
        fake.capturing = false;
        ctl.teardown(&mut fake).unwrap();
        assert!(ctl.is_resident());
        // A restore that is needed under a stream capture cannot run: the loan
        // poisons and teardown is an error (the caller leaks).
        let mut ctl = controller();
        fake.fail = None;
        fake.sticky = false;
        ctl.acquire(&mut fake, 256).unwrap();
        fake.capturing = true;
        assert!(ctl.teardown(&mut fake).is_err());
        assert!(matches!(ctl.state(), LoanState::Poisoned(_)));
    }

    #[test]
    fn epoch_numbers_and_counters_accumulate_across_epochs() {
        let mut ctl = controller();
        let mut fake = Fake::default();
        for epoch in 1..=3u64 {
            ctl.acquire(&mut fake, 1 << 20).unwrap();
            ctl.acquire(&mut fake, 2 << 20).unwrap();
            assert_eq!(
                *ctl.state(),
                LoanState::Dirty {
                    epoch,
                    dirty_bytes: 2 << 20,
                    zeroed_bytes: 2 << 20
                }
            );
            ctl.finish_prompt(&mut fake, true).unwrap();
            ctl.require_resident(&mut fake).unwrap();
        }
        let counters = ctl.counters();
        assert_eq!(counters.epochs, 3);
        assert_eq!(counters.tiles, 6);
        assert_eq!(counters.restores, 3);
        assert_eq!(counters.restored_bytes, 3 * (2 << 20));
        assert_eq!(counters.zeroed_bytes, 3 * (2 << 20));
        assert_eq!(counters.recoveries, 0);
        assert!(ctl.summary().contains("3 epochs"));
        assert!(ctl.summary().contains("6 prompt tiles"));
    }
}
