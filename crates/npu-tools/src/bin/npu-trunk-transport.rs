// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Production-size two-bank transport gate. No HIP VMM export, no SYNC_BO.
use npu_tools::coop_gpu;
use pm_npu::kernels::{gemm_g80, w8a8_epi::bf16_direct};
use railgun::npu::{clflush, trunk::{monotonic_ns, NpuTrunkRuntime, OpenOptions, LayerPanels, PanelProgram, Shape, Seq, WAIT_TIMEOUT, CHAIN_WATCHDOG}};
use std::time::{Duration, Instant};
const ROWS: usize = 8192;
const K: usize = 2560;
const N: usize = 1280;
const CLASSES: usize = 16;
const ROUNDS: usize = 20;
const POISON: u8 = 0xa5;
fn a_code(row: usize, k: usize, round: usize) -> i8 { ((k * 5 + (row % CLASSES) * 7 + round * 11) % 23) as i8 - 11 }
fn b_code(panel: usize, k: usize, col: usize) -> i8 { ((k * 7 + (col % CLASSES) * 3 + panel * 5) % 19) as i8 - 9 }
fn dm(row: usize, round: usize) -> (u16, i16) {
    if row % CLASSES == 0 { (0, 0) } else { (16384 + (row % CLASSES) as u16 * 127 + round as u16 * 17, -20 + (row % 3) as i16) }
}
fn sm(panel: usize, col: usize) -> (u16, i16) { (16384 + (col % CLASSES) as u16 * 151 + panel as u16 * 31, -19 + (panel % 3) as i16) }
fn percentile(values: &[f64], p: f64) -> f64 { values[((values.len() - 1) as f64 * p).ceil() as usize] }
fn at<T>(round: Option<usize>, stage: &str, result: Result<T, String>) -> Result<T, String> {
    result.map_err(|e| format!("stage={stage} round={}: {e}", round.map_or_else(|| "none".into(), |r| r.to_string())))
}
fn resume_latency_ns(anchor_ns: u64, elapsed_ms: f32, done_ns: u64) -> i128 {
    i128::from(anchor_ns) + (f64::from(elapsed_ms) * 1e6) as i128 - i128::from(done_ns)
}
fn consumer_columns(shape: Shape) -> [std::ops::Range<usize>; 4] {
    match shape {
        Shape::GdnIn => [0..10240, 10240..16384, 16384..16432, 16432..16480],
        Shape::QsaIn => [0..12288, 12288..12800, 12800..13312, 13312..13952],
    }
}

fn run(pci: &str, shape: Shape) -> Result<(), String> {
    let panels_count = shape.panels();
    let columns = consumer_columns(shape);
    let live = columns[3].end;
    // The consumer launch geometry uses precisely these live windows, not a
    // panel-rounded width. Diagnostic full-C downloads below also check padding.
    if columns[0].start != 0 || columns.windows(2).any(|w| w[0].end != w[1].start)
        || columns.iter().any(|r| r.is_empty() || r.end > live) || live >= shape.pitch()
    {
        return at(None, "consumer-padding-bounds", Err("consumer reads padded columns".into()));
    }
    let design = at(None, "build-panel", gemm_g80::build_panel_program_rowmajor(ROWS, shape.pitch(), 0))?;
    let mut panels = Vec::with_capacity(panels_count);
    for panel in 0..panels_count {
        let b: Vec<i8> = (0..K).flat_map(|k| (0..N).map(move |n|
            if panel * N + n < live { b_code(panel, k, n) } else { 0 })).collect();
        let scales: Vec<_> = (0..N).map(|n| if panel * N + n < live { sm(panel, n) } else { (0, 0) }).collect();
        panels.push(gemm_g80::pack_b_bf16(&design, &b, &scales));
    }
    let program = PanelProgram { shape, pdi: design.pdi.clone(), txn: design.insts.clone(), padded_rows: ROWS, pitch: shape.pitch() };
    let runtime = at(None, "open-transport", NpuTrunkRuntime::open_with_options(vec![program], vec![LayerPanels { layer: 0, shape, panels }], 2, ROWS, pci, OpenOptions { cpu_reads: true, ..OpenOptions::default() }))?;
    let outcome = (|| {
        let hip = runtime.hip();
        let layout = runtime.layout();
        let c_bytes = layout.c_bytes(shape);
        let module = at(None, "load-copy-module", hip.load_module(coop_gpu::CODE_OBJECT))?;
        let copy = at(None, "get-copy-kernel", module.function(coop_gpu::COPY))?;
        let stream = at(None, "create-stream", hip.stream())?;
        let end = at(None, "create-system-event", hip.system_event())?;
        let anchor = at(None, "create-timing-anchor", hip.event())?;
        let resumed = at(None, "create-resume-event", hip.event())?;
        // Map the HIP event clock into the relay's CLOCK_MONOTONIC domain.
        // Report the bracket width rather than pretending these are exact clocks.
        at(None, "warm-anchor", hip.record_on(&anchor, Some(&stream)))?;
        at(None, "warm-anchor-completion", hip.wait_event_timeout(&anchor, WAIT_TIMEOUT))?;
        let anchor_before = monotonic_ns();
        at(None, "record-anchor", hip.record_on(&anchor, Some(&stream)))?;
        at(None, "anchor-completion", hip.wait_event_timeout(&anchor, WAIT_TIMEOUT))?;
        let anchor_after = monotonic_ns();
        let anchor_ns = anchor_before + (anchor_after - anchor_before) / 2;
        let anchor_uncertainty_ns = anchor_after - anchor_before;
        let input = at(None, "allocate-input", hip.allocate(layout.a_bytes, None))?;
        let output = at(None, "allocate-output", hip.allocate(c_bytes, None))?;
        let poison = at(None, "allocate-poison", hip.allocate(c_bytes, None))?;
        at(None, "initialize-poison", hip.memset(&poison, POISON))?;
        let blocks = at(None, "query-compute-units", hip.compute_units())? * 8;
        let gpu_copy = |dst: u64, src: u64, bytes: usize| -> Result<(), String> {
            if bytes % 16 != 0 { return Err("copy size not aligned".into()); }
            let mut args = coop_gpu::copy_args(dst, src, (bytes / 16) as u64, u64::from(blocks * coop_gpu::COPY_BLOCK));
            hip.launch_on(Some(&stream), &copy, blocks, coop_gpu::COPY_BLOCK, &mut args)
        };
        let drain = || -> Result<(), String> {
            hip.record_on(&end, Some(&stream))?;
            hip.wait_event_timeout(&end, WAIT_TIMEOUT)
        };
        let mut observed = vec![0u8; c_bytes];
        let mut a = vec![0i8; ROWS * K];
        let mut scales = vec![(0u16, 0i16); ROWS];
        let mut reference = vec![0u16; panels_count * CLASSES * CLASSES * 3];
        let mut latencies = Vec::with_capacity(ROUNDS);
        let mut resumes = Vec::with_capacity(ROUNDS);
        let mut busy = Duration::ZERO;
        let mut last_seq = [None; 2];
        let wall = Instant::now();
        println!("transport shape={shape:?} layout=rowmajor pitch={} live={live} banks=2 panels={panels_count} rounds={ROUNDS} A={} C={c_bytes} bytes/bank ready_timeout={WAIT_TIMEOUT:?} chain_watchdog={CHAIN_WATCHDOG:?} shim_axi=default consumer_windows={columns:?} anchor_uncertainty_ns={anchor_uncertainty_ns}", shape.pitch(), layout.a_bytes);
        for round in 0..ROUNDS {
            let bank = round % 2;
            for row in 0..ROWS {
                scales[row] = dm(row, round);
                for k in 0..K { a[row * K + k] = a_code(row, k, round); }
            }
            let packed = gemm_g80::pack_a_bf16(&design, &a, &scales);
            if packed.len() != layout.a_bytes { return at(Some(round), "pack-A", Err("A sizing mismatch".into())); }
            at(Some(round), "upload-packed-A", hip.upload(&input, &packed))?;
            // Independent CPU dot products for EVERY class combination. The
            // operand is periodic in rows/columns, but comparisons cover EVERY
            // byte of the fused row-major C arena, including zero padding.
            for panel in 0..panels_count { for r in 0..CLASSES { for n in 0..CLASSES {
                let dot: i32 = (0..K).map(|k| i32::from(a_code(r, k, round)) * i32::from(b_code(panel, k, n))).sum();
                let (sm, se) = sm(panel, n);
                for rem in 0..3 {
                    let (dm, _) = dm(r, round);
                    reference[((panel * CLASSES + r) * CLASSES + n) * 3 + rem] = bf16_direct(dot, sm, se, dm, -20 + rem as i16);
                }
            } } }
            let av = at(Some(round), "A-arena", runtime.a_arena(bank))?;
            let cv = at(Some(round), "C-arena", runtime.c_arena(bank, shape))?;
            at(Some(round), "prepare-bank", unsafe { runtime.prepare_bank(bank, stream.raw()) })?;
            at(Some(round), "poison-C", gpu_copy(cv.device, poison.ptr(), c_bytes))?;
            at(Some(round), "GPU-write-A", gpu_copy(av.device, input.ptr(), layout.a_bytes))?;
            let seq = at(Some(round), "kick", unsafe { runtime.kick(0, shape, bank, ROWS, stream.raw()) })?;
            at(Some(round), "wait-GPU-enqueue-ahead", unsafe { runtime.wait_gpu(seq, stream.raw()) })?;
            let wait_enqueue_ns = monotonic_ns();
            at(Some(round), "record-GPU-resume", hip.record_on(&resumed, Some(&stream)))?;
            at(Some(round), "GPU-read-C", gpu_copy(output.ptr(), cv.device, c_bytes))?;
            at(Some(round), "retire-bank", unsafe { runtime.retire_bank(seq, stream.raw()) })?;
            last_seq[bank] = Some(seq);
            at(Some(round), "consumer-completion", drain())?;
            let completion = at(Some(round), "check-terminal-completion", runtime.wait_host(seq))?;
            at(Some(round), "check-runtime-error", runtime.check_error())?;
            if wait_enqueue_ns >= completion.done_ns {
                return at(Some(round), "enqueue-ahead-order", Err("GPU wait was not enqueued before NPU completion".into()));
            }
            let resume_ns = resume_latency_ns(anchor_ns, at(Some(round), "GPU-resume-time", hip.elapsed_ms(&anchor, &resumed))?, completion.done_ns);
            resumes.push(resume_ns as f64 / 1000.0);
            at(Some(round), "download-C", hip.download(&output, &mut observed))?;
            // The oracle intentionally reads C's CPU alias; open opted into
            // per-completion invalidation before terminal publication.
            if unsafe { core::slice::from_raw_parts(cv.host, cv.len) } != &observed[..] {
                return at(Some(round), "compare-C-alias", Err(format!("bank {bank}: GPU C/userptr alias mismatch")));
            }
            // Also prove the entire GPU-written A aliases the anonymous pages.
            unsafe { clflush(av.host, av.len); }
            if unsafe { core::slice::from_raw_parts(av.host, av.len) } != &packed[..] {
                return at(Some(round), "compare-A-alias", Err(format!("bank {bank}: GPU A/userptr alias mismatch")));
            }
            for panel in 0..panels_count { for row in 0..ROWS { for col in 0..N {
                let fused_col = panel * N + col;
                let offset = 2 * (row * shape.pitch() + fused_col);
                let actual = u16::from_le_bytes([observed[offset], observed[offset + 1]]);
                let expected = if fused_col < live {
                    reference[((panel * CLASSES + row % CLASSES) * CLASSES + col % CLASSES) * 3 + row % 3]
                } else { 0 };
                if actual != expected {
                    return at(Some(round), "compare-C", Err(format!("bank {bank} panel {panel} row {row} col {col}: {actual:04x} != {expected:04x}")));
                }
            } } }
            let ms = completion.kick_to_done.as_secs_f64() * 1000.0;
            busy += completion.npu_busy;
            latencies.push(ms);
            println!("round={round:02} bank={bank} seq={} exact enqueue_ahead=true gpu_wait_enqueue_ns={wait_enqueue_ns} npu_done_ns={} kick->done={ms:.3} ms npu_submit_wait={:.3} ms npu_done->gpu_resume={:.3} us", seq.value, completion.done_ns, completion.npu_busy.as_secs_f64() * 1000.0, resume_ns as f64 / 1000.0);
        }
        latencies.sort_by(f64::total_cmp);
        println!("EXACT rounds={ROUNDS} p50={:.3} p99={:.3} max={:.3} ms npu_busy_envelope={:.3} ms wall={:.3} ms (busy=submit-to-timeline host envelope, wall includes CPU oracle/copies)",
            percentile(&latencies, 0.5), percentile(&latencies, 0.99), latencies[ROUNDS - 1], busy.as_secs_f64() * 1000.0, wall.elapsed().as_secs_f64() * 1000.0);
        resumes.sort_by(f64::total_cmp);
        println!("RESUME npu_done->GPU_event p50={:.3} p99={:.3} max={:.3} us anchor_uncertainty_ns={anchor_uncertainty_ns}", percentile(&resumes, 0.5), percentile(&resumes, 0.99), resumes[ROUNDS-1]);
        // Negative controls run only AFTER every exact round has reported.
        // This generation was never submitted; Shared::wait rejects it in
        // process, without a DRM wait, relay request or GPU wait.
        let invalid_start = Instant::now();
        let unsubmitted = Seq { bank: 0, value: last_seq[0].unwrap().value + 1 };
        match runtime.wait_host(unsubmitted) {
            Ok(_) => return at(None, "negative-unsubmitted-seq", Err("unsubmitted sequence unexpectedly succeeded".into())),
            Err(e) if e == "unknown/stale sequence" => println!("negative unsubmitted-seq: error ({e}), {:.3} ms", invalid_start.elapsed().as_secs_f64() * 1000.0),
            Err(e) => return at(None, "negative-unsubmitted-seq", Err(format!("unexpected error: {e}"))),
        }
        // Deterministic missing-wait negative: read freshly poisoned C without
        // submitting a replacement NPU job. A racing read could finish after
        // NPU completion and accidentally pass, so it is not used here.
        let cv = at(None, "negative-C-arena", runtime.c_arena(0, shape))?;
        at(None, "negative-poison-C", gpu_copy(cv.device, poison.ptr(), c_bytes))?;
        at(None, "negative-GPU-read-C", gpu_copy(output.ptr(), cv.device, c_bytes))?;
        at(None, "negative-retire-bank", unsafe { runtime.retire_bank(last_seq[0].unwrap(), stream.raw()) })?;
        at(None, "negative-consumer-completion", drain())?;
        at(None, "negative-download-C", hip.download(&output, &mut observed))?;
        if observed.iter().any(|&b| b != POISON) {
            return at(None, "negative-missing-wait", Err("poison negative failed".into()));
        }
        if reference.iter().all(|&x| x == u16::from_le_bytes([POISON; 2])) {
            return at(None, "negative-missing-wait", Err("negative not distinguishable from reference".into()));
        }
        println!("negative missing-wait: detected poisoned C without replacement submission (all bytes)");
        println!("EXACT_CONTROLS shape={shape:?} rounds={ROUNDS} negatives=2 consumer_padding_bounds=checked");
        Ok(())
    })();
    let teardown = at(None, "teardown", runtime.close());
    outcome.and(teardown)?;
    // A separate runtime is intentional: a real timeout permanently poisons the
    // relay, and can never be followed by success on that same transport.
    watchdog_negative(pci, shape, design.pdi, design.insts)?;
    println!("PASS shape={shape:?} rounds={ROUNDS} negatives=3 consumer_padding_bounds=checked");
    Ok(())
}
fn watchdog_negative(pci: &str, shape: Shape, pdi: Vec<u8>, txn: Vec<u8>) -> Result<(), String> {
    let layout = railgun::npu::trunk::ArenaLayout::new(ROWS)?;
    // Valid zero operands/program, but a deliberately impossible chain budget:
    // EXEC_CMD must return before its 1 ns deadline can be checked. This exercises
    // production watchdog cancellation, not an environment-dependent test shim.
    let panels = (0..shape.panels()).map(|_| vec![0; layout.b_bytes]).collect();
    let runtime = at(None, "negative-timeout-open", NpuTrunkRuntime::open_with_options(
        vec![PanelProgram { shape, pdi, txn, padded_rows: ROWS, pitch: shape.pitch() }],
        vec![LayerPanels { layer: 0, shape, panels }], 1, ROWS, pci,
        OpenOptions { watchdog: Duration::from_nanos(1), cpu_reads: true }))?;
    let outcome = (|| {
        let hip = runtime.hip();
        let stream = at(None, "negative-timeout-stream", hip.stream())?;
        let resumed = at(None, "negative-timeout-resume-event", hip.system_event())?;
        let start = Instant::now();
        let seq = at(None, "negative-timeout-kick", unsafe { runtime.kick(0, shape, 0, ROWS, stream.raw()) })?;
        at(None, "negative-timeout-wait-enqueue", unsafe { runtime.wait_gpu(seq, stream.raw()) })?;
        at(None, "negative-timeout-resume-record", hip.record_on(&resumed, Some(&stream)))?;
        // Record retirement even when a very fast timeout is already visible.
        let retirement = unsafe { runtime.retire_bank(seq, stream.raw()) };
        at(None, "negative-timeout-GPU-released", hip.wait_event_timeout(&resumed, CHAIN_WATCHDOG))?;
        let elapsed = start.elapsed();
        if elapsed > CHAIN_WATCHDOG { return Err(format!("timeout GPU release exceeded watchdog: {elapsed:?}")); }
        let error = runtime.wait_host(seq).err().ok_or("forced watchdog unexpectedly succeeded")?;
        if !error.contains("watchdog") { return Err(format!("unexpected forced-timeout error: {error}")); }
        let sticky = runtime.check_error().err().ok_or("timeout did not poison runtime")?;
        if !sticky.contains("watchdog") { return Err(format!("missing sticky watchdog error: {sticky}")); }
        if let Err(e) = retirement {
            if !e.contains("watchdog") { return Err(format!("unexpected retirement error: {e}")); }
        }
        let retire_error = unsafe { runtime.retire_bank(seq, stream.raw()) }.err()
            .ok_or("retirement did not surface terminal timeout")?;
        if !retire_error.contains("watchdog") { return Err(format!("missing retirement watchdog error: {retire_error}")); }
        println!("negative forced-NPU-watchdog: GPU resumed in {:.3} ms (limit {:.3} ms); surfaced error ({error}); failed C never read", elapsed.as_secs_f64()*1000.0, CHAIN_WATCHDOG.as_secs_f64()*1000.0);
        Ok(())
    })();
    // A failed runtime reports the sticky job error at checked teardown, too.
    let teardown = runtime.close();
    outcome?;
    match teardown {
        Err(e) if e.contains("watchdog") => Ok(()),
        Err(e) => Err(format!("negative-timeout teardown: {e}")),
        Ok(()) => Err("negative-timeout teardown lost sticky error".into()),
    }
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let pci = match args.as_slice() {
        [] => "0000:c0:00.0",
        [flag, pci] if flag == "--pci" => pci,
        [flag] if flag == "--help" => { println!("npu-trunk-transport [--pci 0000:c0:00.0]\n8192 rows, row-major GdnIn + QsaIn (13/11 panels), 2 banks, 20 enqueue-ahead exact rounds/shape FIRST; negatives AFTER include forced chain timeout and bounded GPU release. Prints npu_done->GPU resume event latency; 3 s ready waits, 300 ms chain watchdog, default shim AXI."); return; }
        _ => { eprintln!("FAIL: stage=arguments round=none: usage: npu-trunk-transport [--pci PCI]"); std::process::exit(1); }
    };
    for shape in [Shape::GdnIn, Shape::QsaIn] {
        if let Err(e) = run(pci, shape) { eprintln!("FAIL shape={shape:?}: {e}"); std::process::exit(1); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_event_clock_conversion_preserves_signed_latency() {
        assert_eq!(resume_latency_ns(1_000_000, 2.0, 2_900_000), 100_000);
        assert_eq!(resume_latency_ns(1_000_000, 2.0, 3_100_000), -100_000);
    }

    #[test]
    fn consumer_windows_exclude_panel_padding() {
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            let columns = consumer_columns(shape);
            assert_eq!(columns[0].start, 0);
            for pair in columns.windows(2) { assert_eq!(pair[0].end, pair[1].start); }
            let live = columns[3].end;
            assert!(live < shape.pitch());
            for window in columns {
                for row in [0, ROWS - 1] {
                    let end = 2 * (row * shape.pitch() + window.end);
                    assert!(end <= 2 * (row * shape.pitch() + live));
                }
            }
        }
    }
}
