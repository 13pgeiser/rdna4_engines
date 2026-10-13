// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Fixed-size G80 BF16 panel transport. See docs/npu/railgun-npu.md.
//! The relay owns all DRM objects; HIP resources remain on the calling thread.
use super::{Bo, Chain, Device, HwCtx, Prepared, ERT_STATE_COMPLETED};
use super::hip_runtime::{HipEvent, HipRuntime, HostRegistration, RelayApi};
use core::ffi::c_void;
use std::{cell::RefCell, sync::{Arc, mpsc}, thread, time::{Duration, Instant}};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};

pub const WAIT_TIMEOUT: Duration = Duration::from_secs(3);
/// Five times the measured ~60 ms full G80 chain envelope. GPU-ready polling
/// has its separate WAIT_TIMEOUT budget; queued producer work is not NPU time.
pub const EXPECTED_CHAIN_TIME: Duration = Duration::from_millis(60);
pub const CHAIN_WATCHDOG: Duration = Duration::from_millis(300);
pub const PANEL_N: usize = 1280;
pub const PANEL_K: usize = 2560;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape { GdnIn, QsaIn }
impl Shape {
    pub fn panels(self) -> usize { match self { Self::GdnIn => 13, Self::QsaIn => 11 } }
    pub fn pitch(self) -> usize { self.panels() * PANEL_N }
    fn index(self) -> usize { match self { Self::GdnIn => 0, Self::QsaIn => 1 } }
}

/// One row-major design per shape, with col0=0; every panel reuses it.
/// Supply build_panel_program_rowmajor(rows_max, pitch, 0)'s bytes.
pub struct PanelProgram { pub shape: Shape, pub pdi: Vec<u8>, pub txn: Vec<u8>, pub padded_rows: usize, pub pitch: usize }
/// Every entry is already pack_b_bf16 encoded, including Sm/Se sidecars.
pub struct LayerPanels { pub layer: usize, pub shape: Shape, pub panels: Vec<Vec<u8>> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout { pub padded_rows: usize, pub a_bytes: usize, pub b_bytes: usize, pitches: [usize; 2] }
impl ArenaLayout {
    pub fn new(rows_max: usize) -> Result<Self, String> {
        if !(1..=65536).contains(&rows_max) { return Err("rows_max outside 1..=65536".into()); }
        let padded_rows = rows_max.div_ceil(256) * 256;
        Ok(Self { padded_rows, a_bytes: padded_rows * 2880, b_bytes: 8 * 40 * 2 * 5504,
            pitches: [Shape::GdnIn.pitch(), Shape::QsaIn.pitch()] })
    }
    pub fn c_bytes(self, shape: Shape) -> usize { self.padded_rows * self.pitches[shape.index()] * 2 }
    pub fn c_offset(self, shape: Shape, panel: usize) -> Result<usize, String> {
        if panel >= self.pitches[shape.index()] / PANEL_N { Err("panel index outside C arena".into()) } else { Ok(panel * PANEL_N * 2) }
    }
    /// Addressed span, not the full arg2 declaration: offset panels write only
    /// their 1280 columns on each row, including the final padded row.
    pub fn c_span(self, shape: Shape) -> usize {
        ((self.padded_rows - 1) * self.pitches[shape.index()] + PANEL_N) * 2
    }
}

/// A view, not ownership: valid only until runtime teardown. The BO itself is
/// relay-owned; handle/address expose its NPU binding without transferring it.
#[derive(Clone, Copy, Debug)]
pub struct HostRegistered { pub npu_handle: u32, pub npu_address: u64, pub host: *mut u8, pub device: u64, pub len: usize }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seq { pub bank: usize, pub value: u32 }
#[derive(Clone, Copy, Debug)]
pub struct Completion {
    pub kick_to_done: Duration,
    pub npu_busy: Duration,
    pub submit_ns: u64,
    pub done_ns: u64,
    /// CLOCK_MONOTONIC immediately after publishing/flushing the terminal word.
    pub signal_write_ns: u64,
}

/// Linux CLOCK_MONOTONIC timestamp shared by relay and forward trace.
pub fn monotonic_ns() -> u64 {
    let mut time = [0i64; 2];
    // CLOCK_MONOTONIC is always available on the Linux-only NPU backend.
    assert_eq!(unsafe { super::sys::clock_gettime(1, &mut time) }, 0);
    time[0] as u64 * 1_000_000_000 + time[1] as u64
}

struct Pages { ptr: *mut u8, len: usize }
impl Pages {
    fn new(bytes: usize) -> Result<Self, String> {
        let len = bytes.div_ceil(4096) * 4096;
        let ptr = unsafe { super::sys::mmap(core::ptr::null_mut(), len, super::sys::PROT_RW,
            super::sys::MAP_PRIVATE | super::sys::MAP_ANON, -1, 0) };
        if ptr == super::sys::MAP_FAILED { return Err(format!("trunk mmap: errno {}", super::errno())); }
        Ok(Self { ptr: ptr.cast(), len })
    }
}
impl Drop for Pages { fn drop(&mut self) { unsafe { super::sys::munmap(self.ptr.cast(), self.len); } } }
// Field order enforces HIP unregister before pages are unmapped, including open errors.
struct RegisteredPages<R, P> { registration: R, pages: P }
type Arena = RegisteredPages<HostRegistration, Pages>;
impl Arena {
    fn new(hip: &HipRuntime, bytes: usize) -> Result<Self, String> {
        let pages = Pages::new(bytes)?;
        let registration = hip.register_host(pages.ptr, pages.len)?;
        Ok(Self { registration, pages })
    }
}
struct Bank { a: Arena, c: [Arena; 2], control: Arena, ready: HipEvent, acquire: HipEvent, retired: HipEvent }
#[derive(Clone, Copy)]
struct Binding { handle: u32, addr: u64 }
#[derive(Clone, Copy)]
struct BankPages { a: usize, a_len: usize, c: [usize; 2], c_len: [usize; 2] }
struct Job { seq: Seq, layer: usize, shape: Shape, event: usize, start: Instant }
struct Pending { value: u32, result: Option<Result<Completion, String>> }
struct Status { pending: Vec<Pending>, failure: Option<String>, stopping: bool }
struct Shared { status: Mutex<Status>, changed: Condvar, signals: Vec<usize> }
impl Shared {
    fn validate(&self, seq: Seq) -> Result<(), String> {
        let status = self.status.lock();
        let p = status.pending.get(seq.bank).ok_or("invalid sequence bank")?;
        if seq.value == 0 || p.value != seq.value { return Err("unknown/stale sequence".into()); }
        Ok(())
    }
    fn check_error(&self) -> Result<(), String> {
        match &self.status.lock().failure { Some(e) => Err(e.clone()), None => Ok(()) }
    }
    fn wait(&self, seq: Seq, timeout: Duration) -> Result<Completion, String> {
        let deadline = Instant::now() + timeout;
        let mut status = self.status.lock();
        loop {
            let p = status.pending.get(seq.bank).ok_or("invalid sequence bank")?;
            if seq.value == 0 || p.value != seq.value { return Err("unknown/stale sequence".into()); }
            if let Some(result) = &p.result { return result.clone(); }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { return Err(format!("trunk host wait timed out: {seq:?}")); }
            self.changed.wait_for(&mut status, left);
        }
    }
    // First terminal result wins: shutdown cannot be overwritten by a late
    // successful timeline observation. Record failure BEFORE releasing the GPU.
    fn finish(&self, seq: Seq, mut result: Result<Completion, String>) {
        let mut status = self.status.lock();
        let Some(p) = status.pending.get(seq.bank) else { return; };
        if p.value != seq.value || p.result.is_some() { return; }
        if let Err(e) = &result {
            if status.failure.is_none() { status.failure = Some(format!("trunk {seq:?}: {e}")); }
        }
        let signal_write_ns = self.publish(seq);
        if let Ok(completion) = &mut result { completion.signal_write_ns = signal_write_ns; }
        status.pending[seq.bank].result = Some(result);
        self.changed.notify_all();
    }
    fn publish(&self, seq: Seq) -> u64 {
        // No generation wrap or skipped reuse: both success and failure are
        // terminal for this generation. The registered coherent line is alive
        // until the relay has joined and all queued GPU consumers have retired.
        let done = self.signals[seq.bank] as *mut AtomicU32;
        unsafe { (*done).store(seq.value, Ordering::Release); super::clflush(done.cast(), 64); }
        monotonic_ns()
    }
    fn stop(&self, reason: &str) {
        let mut status = self.status.lock();
        status.stopping = true;
        for bank in 0..status.pending.len() {
            let p = &mut status.pending[bank];
            if p.value == 0 || p.result.is_some() { continue; }
            let seq = Seq { bank, value: p.value };
            p.result = Some(Err(reason.into()));
            if status.failure.is_none() { status.failure = Some(format!("trunk {seq:?}: {reason}")); }
            self.publish(seq);
        }
        self.changed.notify_all();
    }
}
struct Watchdog { deadline: Instant }
impl Watchdog {
    fn new(start: Instant, timeout: Duration) -> Self { Self { deadline: start + timeout } }
    fn remaining(&self, now: Instant) -> Result<Duration, String> {
        let remaining = self.deadline.saturating_duration_since(now);
        if remaining.is_zero() { Err("NPU chain watchdog timed out".into()) } else { Ok(remaining) }
    }
}
struct BankState { generation: u32, kicked: bool, retired: bool, prepared: bool }

/// Explicit diagnostic access and chain watchdog; production never touches
/// A/C from the CPU after open. CPU oracle users must opt into arena invalidation.
#[derive(Clone, Copy, Debug)]
pub struct OpenOptions {
    pub watchdog: Duration,
    pub cpu_reads: bool,
}
impl Default for OpenOptions {
    fn default() -> Self { Self { watchdog: CHAIN_WATCHDOG, cpu_reads: false } }
}
impl OpenOptions {
    fn flush_for_cpu_reads(self, flush: impl FnOnce()) {
        if self.cpu_reads { flush(); }
    }
}

/// Async kick and enqueue-ahead GPU rejoin, with terminal failure signalling.
/// No rows/addresses/commands are patched per kick; padded A MUST be zeroed.
pub struct NpuTrunkRuntime {
    hip: HipRuntime,
    api: RelayApi,
    banks: Vec<Bank>,
    bindings: Vec<[Binding; 3]>,
    layout: ArenaLayout,
    rows_max: usize,
    layers: Vec<(usize, Shape)>,
    state: RefCell<Vec<BankState>>,
    shared: Arc<Shared>,
    tx: Option<mpsc::SyncSender<Job>>,
    relay: Option<thread::JoinHandle<()>>,
}
impl NpuTrunkRuntime {
    /// Uses the existing dlopen HIP backend, without a link-time ROCm dependency
    /// or a second GPU runtime crate. pci identifies the same GPU as producer.
    pub fn open(programs: Vec<PanelProgram>, layers: Vec<LayerPanels>, banks: usize, rows_max: usize, pci: &str) -> Result<Self, String> {
        Self::open_with_options(programs, layers, banks, rows_max, pci, OpenOptions::default())
    }
    /// CPU oracle access requires `cpu_reads`; a 1 ns watchdog forces the real
    /// timeout path in the transport gate without a production environment flag.
    pub fn open_with_options(programs: Vec<PanelProgram>, layers: Vec<LayerPanels>, banks: usize, rows_max: usize, pci: &str, options: OpenOptions) -> Result<Self, String> {
        if options.watchdog.is_zero() || options.watchdog > WAIT_TIMEOUT { return Err("chain watchdog outside (0, WAIT_TIMEOUT]".into()); }
        let mut layout = ArenaLayout::new(rows_max)?;
        layout.pitches = [0; 2];
        if banks == 0 { return Err("trunk needs at least one bank".into()); }
        let mut shapes = Vec::new();
        for program in &programs {
            if shapes.contains(&program.shape) { return Err("duplicate shape program".into()); }
            if program.pdi.is_empty() || program.txn.is_empty() { return Err("empty panel program".into()); }
            if program.padded_rows != layout.padded_rows { return Err("panel program padded_rows mismatch".into()); }
            if program.pitch == 0 || program.pitch % PANEL_N != 0 || program.pitch > program.shape.pitch() {
                return Err("invalid compacted panel pitch".into());
            }
            layout.pitches[program.shape.index()] = program.pitch;
            if programs[0].pdi != program.pdi { return Err("shape programs must share the G80 PDI".into()); }
            shapes.push(program.shape);
        }
        let mut keys = Vec::new();
        for l in &layers {
            if !shapes.contains(&l.shape) { return Err("missing shape program".into()); }
            if keys.contains(&(l.layer, l.shape)) { return Err("duplicate layer/shape".into()); }
            if l.panels.len() != layout.pitches[l.shape.index()] / PANEL_N || l.panels.iter().any(|b| b.len() != layout.b_bytes) {
                return Err(format!("layer {} {:?}: invalid BF16 B panel count/size", l.layer, l.shape));
            }
            keys.push((l.layer, l.shape));
        }
        if keys.is_empty() { return Err("no resident B panels".into()); }
        let hip = HipRuntime::load_for_pci(pci)?;
        if hip.device_arch()? != "gfx1151" { return Err("trunk transport currently requires gfx1151".into()); }
        let api = hip.relay_api()?;
        let mut storage = Vec::with_capacity(banks);
        for _ in 0..banks {
            storage.push(Bank { a: Arena::new(&hip, layout.a_bytes)?,
                // An absent shape keeps only one registration page for the fixed
                // binding table, never a full output arena.
                c: [Arena::new(&hip, layout.c_bytes(Shape::GdnIn).max(4096))?, Arena::new(&hip, layout.c_bytes(Shape::QsaIn).max(4096))?],
                control: Arena::new(&hip, 4096)?, ready: hip.system_event()?, acquire: hip.system_event()?, retired: hip.system_event()? });
        }
        let pages = storage.iter().map(|b| BankPages { a: b.a.pages.ptr as usize, a_len: b.a.pages.len,
            c: [b.c[0].pages.ptr as usize, b.c[1].pages.ptr as usize],
            c_len: [b.c[0].pages.len, b.c[1].pages.len] }).collect();
        let shared = Arc::new(Shared {
            status: Mutex::new(Status { pending: (0..banks).map(|_| Pending { value: 0, result: None }).collect(), failure: None, stopping: false }),
            changed: Condvar::new(), signals: storage.iter().map(|b| b.control.pages.ptr as usize).collect(),
        });
        let (tx, rx) = mpsc::sync_channel(banks);
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        let relay_shared = Arc::clone(&shared);
        let relay = thread::Builder::new().name("npu-trunk-relay".into()).spawn(move || {
            relay_main(programs, layers, pages, api, layout, rx, init_tx, relay_shared, options);
        }).map_err(|e| format!("spawn trunk relay: {e}"))?;
        // Construct the teardown guard before receiving: all early returns join
        // the relay (contexts/BOs) before unregistering/unmapping caller pages.
        let mut rt = Self { hip, api, banks: storage, bindings: Vec::new(), layout, rows_max, layers: keys,
            state: RefCell::new((0..banks).map(|_| BankState { generation: 0, kicked: false, retired: false, prepared: false }).collect()),
            shared, tx: Some(tx), relay: Some(relay) };
        rt.bindings = init_rx.recv().map_err(|_| "trunk relay exited during open")??;
        Ok(rt)
    }
    pub fn layout(&self) -> ArenaLayout { self.layout }
    pub fn hip(&self) -> &HipRuntime { &self.hip }
    fn view(&self, bank: usize, shape: Option<Shape>) -> Result<HostRegistered, String> {
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let arena = shape.map_or(&b.a, |s| &b.c[s.index()]);
        let binding = self.bindings[bank][shape.map_or(0, |s| 1 + s.index())];
        Ok(HostRegistered { npu_handle: binding.handle, npu_address: binding.addr, host: arena.pages.ptr,
            device: arena.registration.ptr(), len: shape.map_or(self.layout.a_bytes, |s| self.layout.c_bytes(s)) })
    }
    pub fn a_arena(&self, bank: usize) -> Result<HostRegistered, String> { self.view(bank, None) }
    pub fn c_arena(&self, bank: usize, shape: Shape) -> Result<HostRegistered, String> { self.view(bank, Some(shape)) }
    pub fn done_word(&self, bank: usize) -> Result<u64, String> {
        Ok(self.banks.get(bank).ok_or("invalid trunk bank")?.control.registration.ptr())
    }

    /// Nonblocking sticky error check. A terminal failure releases its GPU wait,
    /// but never licenses using that pair's outputs or committing request state.
    pub fn check_error(&self) -> Result<(), String> { self.shared.check_error() }
    /// Enqueue retirement before writing the next A. Must precede producer pack.
    /// # Safety
    /// `stream` is live, belongs to this GPU, and carries the following producer.
    pub unsafe fn prepare_bank(&self, bank: usize, stream: *mut c_void) -> Result<(), String> {
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let mut states = self.state.borrow_mut();
        let state = &mut states[bank];
        if state.kicked {
            if !state.retired { return Err("bank consumer not retired".into()); }
            self.wait_host(Seq { bank, value: state.generation })?;
            unsafe { self.hip.wait_raw(&b.retired, stream)?; }
        }
        state.prepared = true;
        Ok(())
    }
    /// Records a system-release event after GPU A writes on stream, then queues
    /// relay work. Caller must not touch A/C until completion/consumer retirement.
    /// `stream` must be live and on the selected GPU; null selects its default stream.
    pub unsafe fn kick(&self, layer: usize, shape: Shape, bank: usize, rows: usize, stream: *mut c_void) -> Result<Seq, String> {
        self.check_error()?;
        if rows == 0 || rows > self.rows_max { return Err("rows outside admitted extent".into()); }
        if !self.layers.contains(&(layer, shape)) { return Err("unknown layer/shape".into()); }
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let mut states = self.state.borrow_mut();
        let state = &mut states[bank];
        if state.kicked {
            if !state.retired || (!state.prepared && !self.api.query(b.retired.raw())?) { return Err("bank consumer not retired".into()); }
            self.wait_host(Seq { bank, value: state.generation })?;
        }
        // signed-difference GEQ implementations must never see generation wrap.
        if state.generation == i32::MAX as u32 { return Err("bank sequence exhausted; reopen runtime".into()); }
        let seq = Seq { bank, value: state.generation + 1 };
        let start = Instant::now();
        {
            let mut status = self.shared.status.lock();
            if status.stopping { return Err("trunk runtime stopping".into()); }
            status.pending[bank] = Pending { value: seq.value, result: None };
        }
        // Mark ownership before any fallible enqueue. Even on HIP/relay failure,
        // GPU A writes may already be queued by the caller: teardown must not
        // unmap that operand just because no job reached the relay.
        state.generation = seq.value;
        state.kicked = true;
        state.retired = false;
        state.prepared = false;
        let enqueue = (|| {
            unsafe { self.hip.record_raw(&b.ready, stream)?; }
            self.tx.as_ref().ok_or("runtime closed")?.try_send(Job { seq, layer, shape, event: b.ready.raw(), start })
                .map_err(|e| format!("trunk relay queue: {e}"))
        })();
        if let Err(e) = enqueue {
            self.shared.finish(seq, Err(e.clone()));
            return Err(e);
        }
        Ok(seq)
    }
    pub fn wait_host(&self, seq: Seq) -> Result<Completion, String> {
        self.shared.wait(seq, WAIT_TIMEOUT)
    }
    /// Enqueue the terminal GEQ signal wait immediately, without waiting for
    /// host completion, then a system-fenced event before GPU C loads. Failures
    /// also release this wait; callers MUST check errors before using pair outputs.
    pub unsafe fn wait_gpu(&self, seq: Seq, stream: *mut c_void) -> Result<(), String> {
        self.shared.validate(seq)?;
        let bank = &self.banks[seq.bank];
        unsafe {
            self.api.wait_value(stream, self.done_word(seq.bank)?, seq.value)?;
            self.hip.record_raw(&bank.acquire, stream)?;
        }
        Ok(())
    }
    /// Call after enqueuing the LAST C consumer. Required before bank reuse or
    /// teardown. The event bounds retirement without synchronizing unrelated streams.
    pub unsafe fn retire_bank(&self, seq: Seq, stream: *mut c_void) -> Result<(), String> {
        self.shared.validate(seq)?;
        unsafe { self.hip.record_raw(&self.banks[seq.bank].retired, stream)?; }
        self.state.borrow_mut()[seq.bank].retired = true;
        self.check_error()
    }
    /// Checked teardown. If a GPU consumer does not retire within 3 s, keeps
    /// registrations/pages alive rather than unmapping memory still in use.
    pub fn close(mut self) -> Result<(), String> { self.shutdown() }
    fn shutdown(&mut self) -> Result<(), String> {
        // Release every outstanding GPU wait BEFORE joining/destructing the
        // NPU context. A late completion cannot replace the cancellation result.
        self.shared.stop("runtime shutdown before NPU completion");
        self.tx.take();
        let mut failure = None;
        if let Some(relay) = self.relay.take() {
            if relay.join().is_err() { failure = Some("trunk relay panicked".to_string()); }
        }
        let deadline = Instant::now() + WAIT_TIMEOUT;
        for (b, state) in self.banks.iter().zip(self.state.borrow().iter()) {
            if !state.kicked { continue; }
            if !state.retired { failure = Some("GPU bank not retired before teardown".to_string()); break; }
            loop {
                match self.api.query(b.retired.raw()) {
                    Ok(true) => break,
                    Ok(false) if Instant::now() < deadline => thread::sleep(Duration::from_micros(50)),
                    Ok(false) => { failure = Some("GPU retirement timed out".into()); break; }
                    Err(e) => { failure = Some(e); break; }
                }
            }
        }
        if let Some(e) = failure {
            // Exceptional safe quarantine: cannot destroy events, unregister or
            // unmap a live GPU operand. OS cleanup happens at process exit.
            std::mem::forget(std::mem::take(&mut self.banks));
            return Err(e);
        }
        self.banks.clear();
        self.check_error()?;
        Ok(())
    }
}
impl Drop for NpuTrunkRuntime {
    fn drop(&mut self) {
        // close() already joined and either freed or quarantined every bank.
        if self.banks.is_empty() && self.relay.is_none() { return; }
        if let Err(e) = self.shutdown() { eprintln!("npu-trunk teardown: {e}"); }
    }
}

// The order is intentional and tested below: ctx, chains, commands, operands,
// instruction/PDI BOs. The enclosing Device drops only after this owner.
struct Resources<C, Ch = Chain, P = Prepared, B = Bo> {
    ctx: C, chains: Vec<Ch>, commands: Vec<P>, b: Vec<B>, a: Vec<B>, c: Vec<B>,
    insts: Vec<B>, pdi: B,
}
fn prepared_panel(dev: &Device, insts: &Bo, a: &Bo, b: &Bo, c: &Bo, c_offset: usize, c_bytes: usize) -> Result<Prepared, String> {
    if c_offset.checked_add(c_bytes).is_none_or(|end| end > c.len) { return Err("C panel outside arena".into()); }
    let mut command = Prepared::new(dev, insts, &[a, b, c])?;
    // Initialization only, not a hot-path patch: retains the real C handle for
    // pinning while binding a panel's fixed offset in that BO.
    command.addresses[2] = c.dev_addr().checked_add(c_offset as u64).ok_or("C address overflow")?;
    let mut words = [0; 17];
    super::encode_start_raw(&mut words, insts.xdna, insts.len, &command.addresses[..3]);
    super::write_words(&command.cmd, &words);
    Ok(command)
}
fn relay_main(programs: Vec<PanelProgram>, layers: Vec<LayerPanels>, pages: Vec<BankPages>, api: RelayApi, layout: ArenaLayout,
    rx: mpsc::Receiver<Job>, init: mpsc::SyncSender<Result<Vec<[Binding; 3]>, String>>, shared: Arc<Shared>, options: OpenOptions) {
    let dev = match Device::open().and_then(|mut d| { d.map_heap(64 << 20)?; Ok(d) }) {
        Ok(d) => d, Err(e) => { let _ = init.send(Err(e)); return; }
    };
    let setup = (|| {
        api.select()?;
        let pdi = dev.dev_bo(&programs[0].pdi)?;
        let ctx = HwCtx::create(&dev, 32, 1)?;
        let mut r = Resources { ctx, chains: Vec::new(), commands: Vec::new(), b: Vec::new(), a: Vec::new(), c: Vec::new(), insts: Vec::new(), pdi };
        r.ctx.config_cu(&r.pdi)?;
        for program in &programs { r.insts.push(dev.dev_bo(&program.txn)?); }
        let mut bindings = Vec::with_capacity(pages.len());
        for p in &pages {
            let a = dev.userptr_bo(p.a as *mut u8, p.a_len)?;
            let c0 = dev.userptr_bo(p.c[0] as *mut u8, p.c_len[0])?;
            let c1 = dev.userptr_bo(p.c[1] as *mut u8, p.c_len[1])?;
            bindings.push([Binding { handle: a.handle, addr: a.dev_addr() },
                Binding { handle: c0.handle, addr: c0.dev_addr() }, Binding { handle: c1.handle, addr: c1.dev_addr() }]);
            r.a.push(a); r.c.extend([c0, c1]);
        }
        for l in &layers {
            let b_start = r.b.len();
            let program = programs.iter().position(|p| p.shape == l.shape).ok_or("missing relay shape program")?;
            for bytes in &l.panels {
                let mut b = dev.shmem_bo(bytes.len())?;
                b.as_mut_slice().copy_from_slice(bytes); b.flush(); r.b.push(b);
            }
            for bank in 0..pages.len() {
                let start = r.commands.len();
                for panel in 0..l.panels.len() {
                    r.commands.push(prepared_panel(&dev, &r.insts[program], &r.a[bank], &r.b[b_start + panel],
                        &r.c[bank * 2 + l.shape.index()], layout.c_offset(l.shape, panel)?, layout.c_span(l.shape))?);
                }
                let refs: Vec<_> = r.commands[start..].iter().collect();
                r.chains.push(Chain::new(&dev, &refs)?);
            }
        }
        // Anonymous pages start zeroed; registrations/userptr setup is complete.
        // Evict any initialization lines once, before handing arenas to devices.
        // Production CPU code never accesses A/C again.
        for arena in r.a.iter().chain(&r.c) { arena.flush(); }
        Ok::<_, String>((r, bindings))
    })();
    let (resources, bindings) = match setup { Ok(r) => r, Err(e) => { let _ = init.send(Err(e)); return; } };
    if init.send(Ok(bindings)).is_err() { return; }
    // Declared after resources, so unwinding releases waits before context/BO
    // destruction (which may itself wait for a device job to be cancelled).
    struct RelayExit<'a>(&'a Shared);
    impl Drop for RelayExit<'_> { fn drop(&mut self) { self.0.stop("trunk relay exited before completion"); } }
    let _exit = RelayExit(&shared);
    let mut fatal: Option<String> = None;
    for job in rx {
        let result = (|| {
            if let Some(e) = &fatal { return Err(format!("trunk relay poisoned: {e}")); }
            if shared.status.lock().stopping { return Err("runtime shutdown before NPU completion".into()); }
            let deadline = job.start + WAIT_TIMEOUT;
            loop {
                if Instant::now() >= deadline { return Err("GPU A-ready event timed out".into()); }
                if shared.status.lock().stopping { return Err("runtime shutdown before GPU A-ready".into()); }
                if api.query(job.event)? { break; }
                thread::sleep(Duration::from_micros(20));
            }
            // GPU system release makes A writes visible to NPU DMA in DRAM.
            // Production has no CPU A lines after the open-time flush; only the
            // diagnostic oracle can have cached aliases. Never SYNC_BO.
            options.flush_for_cpu_reads(|| resources.a[job.seq.bank].flush());
            let layer = layers.iter().position(|l| l.layer == job.layer && l.shape == job.shape).ok_or("unknown relay layer")?;
            let chain = &resources.chains[layer * pages.len() + job.seq.bank];
            let start = Instant::now();
            let watchdog = options.watchdog;
            let chain_watchdog = Watchdog::new(start, watchdog);
            let submit_ns = monotonic_ns();
            let point = resources.ctx.submit_chain(chain)?;
            let left = chain_watchdog.remaining(Instant::now())?;
            let state = resources.ctx.wait_chain(chain, point, left.as_millis().max(1) as u64)
                .map_err(|e| format!("NPU chain completion failed (watchdog {watchdog:?}): {e}"))?;
            let done_ns = monotonic_ns();
            chain_watchdog.remaining(Instant::now())?;
            let npu_busy = start.elapsed();
            if state != ERT_STATE_COMPLETED { return Err(format!("NPU chain state {state}, error index {}", chain.error_index())); }
            // The transaction tail synchronizes all NPU C DMA writes to DRAM.
            // GPU consumers wait on the terminal word then system-acquire; CPU
            // caches hold no C lines in production after the open-time flush.
            options.flush_for_cpu_reads(|| resources.c[job.seq.bank * 2 + job.shape.index()].flush());
            Ok(Completion { kick_to_done: job.start.elapsed(), npu_busy, submit_ns, done_ns, signal_write_ns: 0 })
        })();
        if let Err(e) = &result { fatal = Some(e.clone()); }
        shared.finish(job.seq, result);
    }
    drop(resources); // destroys context before any operand BO, even on timeout
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn arena_flush_policy_keeps_production_device_only() {
        for cpu_reads in [false, true] {
            let options = OpenOptions { cpu_reads, ..OpenOptions::default() };
            let mut flushes = 0;
            for _ in 0..20 {
                options.flush_for_cpu_reads(|| flushes += 1); // A before submit
                options.flush_for_cpu_reads(|| flushes += 1); // C after completion
            }
            assert_eq!(flushes, if cpu_reads { 40 } else { 0 });
        }
        assert!(!OpenOptions::default().cpu_reads);
    }
    #[test] fn compacted_qkv_arena_has_eight_panel_windows() {
        let mut l = ArenaLayout::new(8192).unwrap();
        l.pitches = [10240, 0];
        assert_eq!(l.c_bytes(Shape::GdnIn), 8192 * 10240 * 2);
        assert_eq!(l.c_bytes(Shape::QsaIn), 0);
        for panel in 0..8 {
            assert_eq!(l.c_offset(Shape::GdnIn, panel).unwrap(), panel * 1280 * 2);
            assert!(l.c_offset(Shape::GdnIn, panel).unwrap() + l.c_span(Shape::GdnIn) <= l.c_bytes(Shape::GdnIn));
        }
        assert!(l.c_offset(Shape::GdnIn, 8).is_err());
        assert!(l.c_offset(Shape::QsaIn, 0).is_err());
    }
    #[test] fn production_arena_sizes_and_offsets() {
        let l = ArenaLayout::new(8192).unwrap();
        assert_eq!(l.a_bytes, 23_592_960);
        assert_eq!(l.b_bytes, 3_522_560);
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            assert_eq!(l.c_bytes(shape), 8192 * shape.pitch() * 2);
            for p in 0..shape.panels() {
                let offset = l.c_offset(shape, p).unwrap();
                assert_eq!(offset, 2 * p * PANEL_N);
                assert!(offset + l.c_span(shape) <= l.c_bytes(shape));
            }
            assert!(l.c_offset(shape, shape.panels()).is_err());
            assert_eq!(l.c_offset(shape, shape.panels() - 1).unwrap() + l.c_span(shape), l.c_bytes(shape));
        }
        assert_eq!(l.c_bytes(Shape::GdnIn), 272_629_760);
        assert_eq!(l.c_bytes(Shape::QsaIn), 230_686_720);
        assert_eq!(ArenaLayout::new(257).unwrap().padded_rows, 512);
        assert!(ArenaLayout::new(0).is_err()); assert!(ArenaLayout::new(65537).is_err());
    }
    #[test] fn shape_chains_preserve_panel_order() {
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            let handles: Vec<u32> = (100..100 + shape.panels() as u32).collect();
            let words = super::super::encode_chain(&handles).unwrap();
            assert_eq!(words[1] as usize, shape.panels());
            for (p, h) in handles.iter().enumerate() { assert_eq!(words[7 + 2 * p], *h); }
            assert_eq!((words[0] >> 23) & 31, 19);
        }
    }
    #[repr(align(64))]
    struct Signal(AtomicU32);
    fn shared(signals: &[Signal], value: u32) -> Shared {
        Shared { status: Mutex::new(Status {
            pending: signals.iter().map(|_| Pending { value, result: None }).collect(),
            failure: None, stopping: false,
        }), changed: Condvar::new(), signals: signals.iter().map(|s| &s.0 as *const AtomicU32 as usize).collect() }
    }
    fn completion() -> Completion {
        Completion { kick_to_done: Duration::ZERO, npu_busy: Duration::ZERO, submit_ns: 1, done_ns: 2, signal_write_ns: 0 }
    }
    #[test] fn invalid_timeout_and_failed_sequences_are_bounded() {
        let signals = [Signal(AtomicU32::new(2))];
        let pending = shared(&signals, 3);
        let seq = Seq { bank: 0, value: 3 };
        assert!(pending.wait(Seq { bank: 0, value: 999 }, Duration::ZERO).unwrap_err().contains("unknown"));
        assert!(pending.wait(Seq { bank: 1, value: 3 }, Duration::ZERO).is_err());
        assert!(pending.wait(seq, Duration::from_millis(1)).unwrap_err().contains("timed out"));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.finish(seq, Err("NPU timeout".into()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 3);
        assert_eq!(pending.wait(seq, Duration::ZERO).unwrap_err(), "NPU timeout");
        assert!(pending.check_error().unwrap_err().contains("NPU timeout"));
        pending.finish(seq, Ok(completion()));
        assert_eq!(pending.wait(seq, Duration::ZERO).unwrap_err(), "NPU timeout");
    }
    #[test] fn terminal_signals_are_monotonic_and_stale_completions_are_ignored() {
        let signals = [Signal(AtomicU32::new(0))];
        let pending = shared(&signals, 1);
        let seq = Seq { bank: 0, value: 1 };
        pending.validate(seq).unwrap(); // enqueue-ahead validation needs no completion
        assert_eq!(signals[0].0.load(Ordering::Acquire), 0);
        pending.finish(seq, Ok(completion()));
        assert_eq!(pending.wait(seq, Duration::ZERO).unwrap().done_ns, 2);
        assert!(pending.wait(seq, Duration::ZERO).unwrap().signal_write_ns >= 2);
        pending.status.lock().pending[0] = Pending { value: 2, result: None };
        pending.finish(seq, Err("stale".into()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 1);
        pending.check_error().unwrap();
        pending.finish(Seq { bank: 0, value: 2 }, Ok(completion()));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.stop("shutdown");
        assert_eq!(signals[0].0.load(Ordering::Acquire), 2);
        pending.check_error().unwrap();
    }
    #[test] fn shutdown_releases_every_bank_and_late_success_cannot_clear_failure() {
        let signals = [Signal(AtomicU32::new(6)), Signal(AtomicU32::new(6))];
        let pending = shared(&signals, 7);
        pending.stop("runtime dropped");
        for (bank, signal) in signals.iter().enumerate() {
            let seq = Seq { bank, value: 7 };
            assert_eq!(signal.0.load(Ordering::Acquire), 7);
            pending.finish(seq, Ok(completion()));
            assert_eq!(pending.wait(seq, Duration::ZERO).unwrap_err(), "runtime dropped");
        }
        assert!(pending.status.lock().stopping);
        assert!(pending.check_error().unwrap_err().contains("runtime dropped"));
    }
    #[test] fn watchdog_deadline_is_chain_only_and_timeout_releases_wait() {
        let start = Instant::now();
        let watchdog = Watchdog::new(start, CHAIN_WATCHDOG);
        assert_eq!(CHAIN_WATCHDOG, EXPECTED_CHAIN_TIME * 5);
        assert_eq!(watchdog.remaining(start).unwrap(), CHAIN_WATCHDOG);
        assert_eq!(watchdog.remaining(start + CHAIN_WATCHDOG - Duration::from_nanos(1)).unwrap(), Duration::from_nanos(1));
        let signals = [Signal(AtomicU32::new(0))];
        let pending = shared(&signals, 1);
        let error = watchdog.remaining(start + CHAIN_WATCHDOG).unwrap_err();
        pending.finish(Seq { bank: 0, value: 1 }, Err(error));
        assert_eq!(signals[0].0.load(Ordering::Acquire), 1);
        assert!(pending.check_error().unwrap_err().contains("watchdog timed out"));
        assert!(watchdog.remaining(start + WAIT_TIMEOUT).is_err());
    }
    #[test] fn owning_fields_teardown_in_declared_order() {
        use std::rc::Rc;
        struct Trace(&'static str, Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Trace { fn drop(&mut self) { self.1.borrow_mut().push(self.0); } }
        let log = Rc::new(RefCell::new(Vec::new()));
        let t = |name| Trace(name, Rc::clone(&log));
        // Instantiate the actual production ownership aggregates with tracing
        // resources, exercising their field order without opening either device.
        let npu = Resources { ctx: t("context"), chains: vec![t("chain")], commands: vec![t("command")],
            b: vec![t("bo")], a: vec![], c: vec![], insts: vec![t("insts")], pdi: t("pdi") };
        let arena = RegisteredPages { registration: t("unregister"), pages: t("unmap") };
        drop((npu, arena));
        assert_eq!(&*log.borrow(), &["context", "chain", "command", "bo", "insts", "pdi", "unregister", "unmap"]);
    }
    #[test] fn panel_commands_bind_fixed_arena_offsets() {
        let layout = ArenaLayout::new(8192).unwrap();
        for panel in 0..13 {
            let c = 0x1234_0000_0000u64 + layout.c_offset(Shape::GdnIn, panel).unwrap() as u64;
            let mut words = [0; 17];
            super::super::encode_start_raw(&mut words, 0x1000, 4096, &[0x2000, 0x3000 + panel as u64 * layout.b_bytes as u64, c]);
            assert_eq!(u64::from(words[11]) | (u64::from(words[12]) << 32), c);
            assert_eq!(words[7], 0x2000);
            assert_eq!(words[6], 4096);
        }
    }
}
