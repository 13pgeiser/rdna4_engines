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
pub const PANEL_N: usize = 1280;
pub const PANEL_K: usize = 2560;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape { GdnIn, QsaIn }
impl Shape { pub fn panels(self) -> usize { match self { Self::GdnIn => 13, Self::QsaIn => 11 } } }

/// A shared panel design: both shapes use exactly the same PDI/TXN/context.
/// Supply build_panel_program(rows_max, Epilogue::Bf16)'s bytes unchanged.
pub struct PanelProgram { pub pdi: Vec<u8>, pub txn: Vec<u8>, pub padded_rows: usize }
/// Every entry is already pack_b_bf16 encoded, including Sm/Se sidecars.
pub struct LayerPanels { pub layer: usize, pub shape: Shape, pub panels: Vec<Vec<u8>> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout { pub padded_rows: usize, pub a_bytes: usize, pub b_bytes: usize, pub c_panel_bytes: usize, pub c_bytes: usize }
impl ArenaLayout {
    pub fn new(rows_max: usize) -> Result<Self, String> {
        if !(1..=65536).contains(&rows_max) { return Err("rows_max outside 1..=65536".into()); }
        let padded_rows = rows_max.div_ceil(256) * 256;
        Ok(Self { padded_rows, a_bytes: padded_rows * 2880, b_bytes: 8 * 40 * 2 * 5504,
            c_panel_bytes: padded_rows * PANEL_N * 2, c_bytes: padded_rows * PANEL_N * 2 * 13 })
    }
    pub fn c_offset(self, panel: usize) -> Result<usize, String> {
        if panel >= 13 { Err("panel index outside C arena".into()) } else { Ok(panel * self.c_panel_bytes) }
    }
}

/// A view, not ownership: valid only until runtime teardown. The BO itself is
/// relay-owned; handle/address expose its NPU binding without transferring it.
#[derive(Clone, Copy, Debug)]
pub struct HostRegistered { pub npu_handle: u32, pub npu_address: u64, pub host: *mut u8, pub device: u64, pub len: usize }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seq { pub bank: usize, pub value: u32 }
#[derive(Clone, Copy, Debug)]
pub struct Completion { pub kick_to_done: Duration, pub npu_busy: Duration }

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
struct Bank { a: Arena, c: Arena, control: Arena, ready: HipEvent, acquire: HipEvent, retired: HipEvent }
#[derive(Clone, Copy)]
struct Binding { handle: u32, addr: u64 }
#[derive(Clone, Copy)]
struct BankPages { a: usize, a_len: usize, c: usize, c_len: usize, done: usize }
struct Job { seq: Seq, layer: usize, shape: Shape, event: usize, start: Instant }
enum Work { Run(Job), InvalidTimeline(mpsc::SyncSender<Result<(), String>>) }
struct Pending { value: u32, result: Option<Result<Completion, String>> }
struct Shared { pending: Mutex<Vec<Pending>>, changed: Condvar }
impl Shared {
    fn wait(&self, seq: Seq, timeout: Duration) -> Result<Completion, String> {
        let deadline = Instant::now() + timeout;
        let mut pending = self.pending.lock();
        loop {
            let p = pending.get(seq.bank).ok_or("invalid sequence bank")?;
            if seq.value == 0 || p.value != seq.value { return Err("unknown/stale sequence".into()); }
            if let Some(result) = &p.result { return result.clone(); }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { return Err(format!("trunk host wait timed out: {seq:?}")); }
            self.changed.wait_for(&mut pending, left);
        }
    }
}
struct BankState { generation: u32, kicked: bool, retired: bool }

/// Async kick, bounded host relay, checked GPU rejoin. No rows/addresses/commands
/// are patched per kick; padded A rows MUST contain zero codes and zero scales.
pub struct NpuTrunkRuntime {
    hip: HipRuntime,
    api: RelayApi,
    banks: Vec<Bank>,
    bindings: Vec<[Binding; 2]>,
    layout: ArenaLayout,
    rows_max: usize,
    layers: Vec<(usize, Shape)>,
    state: RefCell<Vec<BankState>>,
    shared: Arc<Shared>,
    tx: Option<mpsc::SyncSender<Work>>,
    relay: Option<thread::JoinHandle<()>>,
}
impl NpuTrunkRuntime {
    /// Uses the existing dlopen HIP backend, without a link-time ROCm dependency
    /// or a second GPU runtime crate. pci identifies the same GPU as producer.
    pub fn open(program: PanelProgram, layers: Vec<LayerPanels>, banks: usize, rows_max: usize, pci: &str) -> Result<Self, String> {
        let layout = ArenaLayout::new(rows_max)?;
        if banks == 0 { return Err("trunk needs at least one bank".into()); }
        if program.pdi.is_empty() || program.txn.is_empty() { return Err("empty panel program".into()); }
        if program.padded_rows != layout.padded_rows { return Err("panel program padded_rows mismatch".into()); }
        let mut keys = Vec::new();
        for l in &layers {
            if keys.contains(&(l.layer, l.shape)) { return Err("duplicate layer/shape".into()); }
            if l.panels.len() != l.shape.panels() || l.panels.iter().any(|b| b.len() != layout.b_bytes) {
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
            storage.push(Bank { a: Arena::new(&hip, layout.a_bytes)?, c: Arena::new(&hip, layout.c_bytes)?,
                control: Arena::new(&hip, 4096)?, ready: hip.system_event()?, acquire: hip.system_event()?, retired: hip.system_event()? });
        }
        let pages = storage.iter().map(|b| BankPages { a: b.a.pages.ptr as usize, a_len: b.a.pages.len,
            c: b.c.pages.ptr as usize, c_len: b.c.pages.len, done: b.control.pages.ptr as usize }).collect();
        let shared = Arc::new(Shared { pending: Mutex::new((0..banks).map(|_| Pending { value: 0, result: None }).collect()), changed: Condvar::new() });
        let (tx, rx) = mpsc::sync_channel(banks);
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        let relay_shared = Arc::clone(&shared);
        let relay = thread::Builder::new().name("npu-trunk-relay".into()).spawn(move || {
            relay_main(program, layers, pages, api, layout, rx, init_tx, relay_shared);
        }).map_err(|e| format!("spawn trunk relay: {e}"))?;
        // Construct the teardown guard before receiving: all early returns join
        // the relay (contexts/BOs) before unregistering/unmapping caller pages.
        let mut rt = Self { hip, api, banks: storage, bindings: Vec::new(), layout, rows_max, layers: keys,
            state: RefCell::new((0..banks).map(|_| BankState { generation: 0, kicked: false, retired: false }).collect()),
            shared, tx: Some(tx), relay: Some(relay) };
        rt.bindings = init_rx.recv().map_err(|_| "trunk relay exited during open")??;
        Ok(rt)
    }
    pub fn layout(&self) -> ArenaLayout { self.layout }
    pub fn hip(&self) -> &HipRuntime { &self.hip }
    fn view(&self, bank: usize, c: bool) -> Result<HostRegistered, String> {
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let arena = if c { &b.c } else { &b.a };
        let binding = self.bindings[bank][usize::from(c)];
        Ok(HostRegistered { npu_handle: binding.handle, npu_address: binding.addr, host: arena.pages.ptr,
            device: arena.registration.ptr(), len: if c { self.layout.c_bytes } else { self.layout.a_bytes } })
    }
    pub fn a_arena(&self, bank: usize) -> Result<HostRegistered, String> { self.view(bank, false) }
    pub fn c_arena(&self, bank: usize) -> Result<HostRegistered, String> { self.view(bank, true) }
    pub fn done_word(&self, bank: usize) -> Result<u64, String> {
        Ok(self.banks.get(bank).ok_or("invalid trunk bank")?.control.registration.ptr())
    }
    /// Records a system-release event after GPU A writes on stream, then queues
    /// relay work. Caller must not touch A/C until completion/consumer retirement.
    /// `stream` must be live and on the selected GPU; null selects its default stream.
    pub unsafe fn kick(&self, layer: usize, shape: Shape, bank: usize, rows: usize, stream: *mut c_void) -> Result<Seq, String> {
        if rows == 0 || rows > self.rows_max { return Err("rows outside admitted extent".into()); }
        if !self.layers.contains(&(layer, shape)) { return Err("unknown layer/shape".into()); }
        let b = self.banks.get(bank).ok_or("invalid trunk bank")?;
        let mut states = self.state.borrow_mut();
        let state = &mut states[bank];
        if state.kicked {
            if !state.retired || !self.api.query(b.retired.raw())? { return Err("bank consumer not retired".into()); }
            self.wait_host(Seq { bank, value: state.generation })?;
        }
        // signed-difference GEQ implementations must never see generation wrap.
        if state.generation == i32::MAX as u32 { return Err("bank sequence exhausted; reopen runtime".into()); }
        let seq = Seq { bank, value: state.generation + 1 };
        let start = Instant::now();
        {
            let mut pending = self.shared.pending.lock();
            pending[bank] = Pending { value: seq.value, result: None };
        }
        // Mark ownership before any fallible enqueue. Even on HIP/relay failure,
        // GPU A writes may already be queued by the caller: teardown must not
        // unmap that operand just because no job reached the relay.
        state.generation = seq.value;
        state.kicked = true;
        state.retired = false;
        let enqueue = (|| {
            unsafe { self.hip.record_raw(&b.ready, stream)?; }
            self.tx.as_ref().ok_or("runtime closed")?.try_send(Work::Run(Job { seq, layer, shape, event: b.ready.raw(), start }))
                .map_err(|e| format!("trunk relay queue: {e}"))
        })();
        if let Err(e) = enqueue {
            finish(&self.shared, seq, Err(e.clone()));
            return Err(e);
        }
        Ok(seq)
    }
    pub fn wait_host(&self, seq: Seq) -> Result<Completion, String> {
        self.shared.wait(seq, WAIT_TIMEOUT)
    }
    /// Hardware negative control, before the first kick only: wait for an
    /// impossible timeline point without submitting or mutating any command.
    /// A correct driver returns an error within the 3 s bound.
    pub fn diagnostic_invalid_timeline(&self) -> Result<(), String> {
        if self.state.borrow().iter().any(|s| s.kicked) { return Err("timeline probe requires fresh runtime".into()); }
        let (tx, rx) = mpsc::sync_channel(1);
        self.tx.as_ref().ok_or("runtime closed")?.try_send(Work::InvalidTimeline(tx))
            .map_err(|e| format!("timeline probe queue: {e}"))?;
        rx.recv_timeout(WAIT_TIMEOUT).map_err(|e| format!("timeline probe bounded receive: {e}"))?
    }
    /// Error-gated rejoin: first bounded host completion, then stream GEQ wait
    /// and a system-fenced event (cache writeback/invalidate) before GPU C loads.
    /// Do not queue a raw done wait yourself: a failed NPU job never publishes success.
    pub unsafe fn wait_gpu(&self, seq: Seq, stream: *mut c_void) -> Result<Completion, String> {
        let completion = self.wait_host(seq)?;
        let bank = &self.banks[seq.bank];
        unsafe {
            self.api.wait_value(stream, self.done_word(seq.bank)?, seq.value)?;
            self.hip.record_raw(&bank.acquire, stream)?;
        }
        Ok(completion)
    }
    /// Call after enqueuing the LAST C consumer. Required before bank reuse or
    /// teardown. The event bounds retirement without synchronizing unrelated streams.
    pub unsafe fn retire_bank(&self, seq: Seq, stream: *mut c_void) -> Result<(), String> {
        self.wait_host(seq)?;
        unsafe { self.hip.record_raw(&self.banks[seq.bank].retired, stream)?; }
        self.state.borrow_mut()[seq.bank].retired = true;
        Ok(())
    }
    /// Checked teardown. If a GPU consumer does not retire within 3 s, keeps
    /// registrations/pages alive rather than unmapping memory still in use.
    pub fn close(mut self) -> Result<(), String> { self.shutdown() }
    fn shutdown(&mut self) -> Result<(), String> {
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
        Ok(())
    }
}
impl Drop for NpuTrunkRuntime { fn drop(&mut self) { if let Err(e) = self.shutdown() { eprintln!("npu-trunk teardown: {e}"); } } }

// The order is intentional and tested below: ctx, chains, commands, operands,
// instruction/PDI BOs. The enclosing Device drops only after this owner.
struct Resources<C, Ch = Chain, P = Prepared, B = Bo> {
    ctx: C, chains: Vec<Ch>, commands: Vec<P>, b: Vec<B>, a: Vec<B>, c: Vec<B>,
    insts: Option<B>, pdi: B,
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
fn finish(shared: &Shared, seq: Seq, result: Result<Completion, String>) {
    let mut pending = shared.pending.lock();
    pending[seq.bank].result = Some(result);
    shared.changed.notify_all();
}
fn relay_main(program: PanelProgram, layers: Vec<LayerPanels>, pages: Vec<BankPages>, api: RelayApi, layout: ArenaLayout,
    rx: mpsc::Receiver<Work>, init: mpsc::SyncSender<Result<Vec<[Binding; 2]>, String>>, shared: Arc<Shared>) {
    let dev = match Device::open().and_then(|mut d| { d.map_heap(64 << 20)?; Ok(d) }) {
        Ok(d) => d, Err(e) => { let _ = init.send(Err(e)); return; }
    };
    let setup = (|| {
        api.select()?;
        let pdi = dev.dev_bo(&program.pdi)?;
        let ctx = HwCtx::create(&dev, 32, 1)?;
        let mut r = Resources { ctx, chains: Vec::new(), commands: Vec::new(), b: Vec::new(), a: Vec::new(), c: Vec::new(), insts: None, pdi };
        r.ctx.config_cu(&r.pdi)?;
        r.insts = Some(dev.dev_bo(&program.txn)?);
        let mut bindings = Vec::with_capacity(pages.len());
        for p in &pages {
            let a = dev.userptr_bo(p.a as *mut u8, p.a_len)?;
            let c = dev.userptr_bo(p.c as *mut u8, p.c_len)?;
            bindings.push([Binding { handle: a.handle, addr: a.dev_addr() }, Binding { handle: c.handle, addr: c.dev_addr() }]);
            r.a.push(a); r.c.push(c);
        }
        for l in &layers {
            let b_start = r.b.len();
            for bytes in &l.panels {
                let mut b = dev.shmem_bo(bytes.len())?;
                b.as_mut_slice().copy_from_slice(bytes); b.flush(); r.b.push(b);
            }
            for bank in 0..pages.len() {
                let start = r.commands.len();
                for panel in 0..l.shape.panels() {
                    r.commands.push(prepared_panel(&dev, r.insts.as_ref().unwrap(), &r.a[bank], &r.b[b_start + panel],
                        &r.c[bank], layout.c_offset(panel)?, layout.c_panel_bytes)?);
                }
                let refs: Vec<_> = r.commands[start..].iter().collect();
                r.chains.push(Chain::new(&dev, &refs)?);
            }
        }
        Ok::<_, String>((r, bindings))
    })();
    let (resources, bindings) = match setup { Ok(r) => r, Err(e) => { let _ = init.send(Err(e)); return; } };
    if init.send(Ok(bindings)).is_err() { return; }
    let mut fatal: Option<String> = None;
    for work in rx {
        let job = match work {
            Work::Run(job) => job,
            Work::InvalidTimeline(reply) => {
                // Leave 10 ms for forwarding the driver's bounded failure.
                let result = resources.ctx.wait_chain(&resources.chains[0], u64::MAX, 2990).map(|_| ());
                let _ = reply.send(result);
                continue;
            }
        };
        let result = (|| {
            if let Some(e) = &fatal { return Err(format!("trunk relay poisoned: {e}")); }
            let deadline = job.start + WAIT_TIMEOUT;
            loop {
                if Instant::now() >= deadline { return Err("GPU A-ready event timed out".into()); }
                if api.query(job.event)? { break; }
                thread::sleep(Duration::from_micros(20));
            }
            // HIP system event released GPU writes. Invalidate CPU aliases
            // before DMA; no SYNC_BO (especially not on imported BOs).
            resources.a[job.seq.bank].flush();
            let layer = layers.iter().position(|l| l.layer == job.layer && l.shape == job.shape).ok_or("unknown relay layer")?;
            let chain = &resources.chains[layer * pages.len() + job.seq.bank];
            let start = Instant::now();
            let point = resources.ctx.submit_chain(chain)?;
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { return Err("NPU deadline exhausted after submit".into()); }
            let state = resources.ctx.wait_chain(chain, point, left.as_millis().max(1) as u64)?;
            let npu_busy = start.elapsed();
            if state != ERT_STATE_COMPLETED { return Err(format!("NPU chain state {state}, error index {}", chain.error_index())); }
            // Transaction tail synchronizes every C channel before timeline done.
            // Flush CPU aliases, publish success with release ordering, flush the
            // control line so the GPU's uncached wait observes it.
            resources.c[job.seq.bank].flush();
            let done = pages[job.seq.bank].done as *mut AtomicU32;
            unsafe { (*done).store(job.seq.value, Ordering::Release); super::clflush(done.cast(), 64); }
            Ok(Completion { kick_to_done: job.start.elapsed(), npu_busy })
        })();
        if let Err(e) = &result { fatal = Some(e.clone()); }
        finish(&shared, job.seq, result);
    }
    drop(resources); // destroys context before any operand BO, even on timeout
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn production_arena_sizes_and_offsets() {
        let l = ArenaLayout::new(8192).unwrap();
        assert_eq!(l.a_bytes, 23_592_960);
        assert_eq!(l.b_bytes, 3_522_560);
        assert_eq!(l.c_panel_bytes, 20_971_520);
        assert_eq!(l.c_bytes, 272_629_760);
        for p in 0..13 { assert_eq!(l.c_offset(p).unwrap(), p * l.c_panel_bytes); }
        assert!(l.c_offset(13).is_err());
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
    #[test] fn invalid_timeout_and_failed_sequences_are_bounded() {
        let pending = Shared { pending: Mutex::new(vec![Pending { value: 3, result: None }]), changed: Condvar::new() };
        let seq = Seq { bank: 0, value: 3 };
        assert!(pending.wait(Seq { bank: 0, value: 999 }, Duration::ZERO).unwrap_err().contains("unknown"));
        assert!(pending.wait(Seq { bank: 1, value: 3 }, Duration::ZERO).is_err());
        assert!(pending.wait(seq, Duration::from_millis(1)).unwrap_err().contains("timed out"));
        finish(&pending, seq, Err("NPU timeout".into()));
        assert_eq!(pending.wait(seq, Duration::ZERO).unwrap_err(), "NPU timeout");
        finish(&pending, seq, Ok(Completion { kick_to_done: Duration::ZERO, npu_busy: Duration::ZERO }));
        assert!(pending.wait(seq, Duration::ZERO).is_ok());
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
            b: vec![t("bo")], a: vec![], c: vec![], insts: Some(t("insts")), pdi: t("pdi") };
        let arena = RegisteredPages { registration: t("unregister"), pages: t("unmap") };
        drop((npu, arena));
        assert_eq!(&*log.borrow(), &["context", "chain", "command", "bo", "insts", "pdi", "unregister", "unmap"]);
    }
    #[test] fn panel_commands_bind_fixed_arena_offsets() {
        let layout = ArenaLayout::new(8192).unwrap();
        for panel in 0..13 {
            let c = 0x1234_0000_0000u64 + layout.c_offset(panel).unwrap() as u64;
            let mut words = [0; 17];
            super::super::encode_start_raw(&mut words, 0x1000, 4096, &[0x2000, 0x3000 + panel as u64 * layout.b_bytes as u64, c]);
            assert_eq!(u64::from(words[11]) | (u64::from(words[12]) << 32), c);
            assert_eq!(words[7], 0x2000);
            assert_eq!(words[6], 4096);
        }
    }
}
