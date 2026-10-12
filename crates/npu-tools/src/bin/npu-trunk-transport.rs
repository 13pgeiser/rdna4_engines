// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Production-size two-bank transport gate. No HIP VMM export, no SYNC_BO.
use npu_tools::coop_gpu;
use pm_npu::kernels::{gemm_core::Epilogue, gemm_g80, w8a8_epi::bf16_direct};
use railgun::npu::{clflush, trunk::{NpuTrunkRuntime, LayerPanels, PanelProgram, Shape, Seq, WAIT_TIMEOUT}};
use std::time::{Duration, Instant};
const ROWS: usize = 8192;
const K: usize = 2560;
const N: usize = 1280;
const PANELS: usize = 13;
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
fn run(pci: &str) -> Result<(), String> {
    let design = at(None, "build-panel", gemm_g80::build_panel_program(ROWS, Epilogue::Bf16))?;
    let mut panels = Vec::with_capacity(PANELS);
    for panel in 0..PANELS {
        let b: Vec<i8> = (0..K).flat_map(|k| (0..N).map(move |n| b_code(panel, k, n))).collect();
        let scales: Vec<_> = (0..N).map(|n| sm(panel, n)).collect();
        panels.push(gemm_g80::pack_b_bf16(&design, &b, &scales));
    }
    let program = PanelProgram { pdi: design.pdi.clone(), txn: design.insts.clone(), padded_rows: ROWS };
    let runtime = at(None, "open-transport", NpuTrunkRuntime::open(program, vec![LayerPanels { layer: 0, shape: Shape::GdnIn, panels }], 2, ROWS, pci))?;
    let outcome = (|| {
        let hip = runtime.hip();
        let layout = runtime.layout();
        let module = at(None, "load-copy-module", hip.load_module(coop_gpu::CODE_OBJECT))?;
        let copy = at(None, "get-copy-kernel", module.function(coop_gpu::COPY))?;
        let stream = at(None, "create-stream", hip.stream())?;
        let end = at(None, "create-system-event", hip.system_event())?;
        let input = at(None, "allocate-input", hip.allocate(layout.a_bytes, None))?;
        let output = at(None, "allocate-output", hip.allocate(layout.c_bytes, None))?;
        let poison = at(None, "allocate-poison", hip.allocate(layout.c_bytes, None))?;
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
        let mut observed = vec![0u8; layout.c_bytes];
        let mut a = vec![0i8; ROWS * K];
        let mut scales = vec![(0u16, 0i16); ROWS];
        let mut reference = vec![0u16; PANELS * CLASSES * CLASSES * 3];
        let mut latencies = Vec::with_capacity(ROUNDS);
        let mut busy = Duration::ZERO;
        let mut last_seq = [None; 2];
        let wall = Instant::now();
        println!("transport rows={ROWS} banks=2 panels={PANELS} rounds={ROUNDS} A={} C={} bytes/bank timeout=3s shim_axi=default", layout.a_bytes, layout.c_bytes);
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
            // byte of all 13 production-size C panels, not a sampled subset.
            for panel in 0..PANELS { for r in 0..CLASSES { for n in 0..CLASSES {
                let dot: i32 = (0..K).map(|k| i32::from(a_code(r, k, round)) * i32::from(b_code(panel, k, n))).sum();
                let (sm, se) = sm(panel, n);
                for rem in 0..3 {
                    let (dm, _) = dm(r, round);
                    reference[((panel * CLASSES + r) * CLASSES + n) * 3 + rem] = bf16_direct(dot, sm, se, dm, -20 + rem as i16);
                }
            } } }
            let av = at(Some(round), "A-arena", runtime.a_arena(bank))?;
            let cv = at(Some(round), "C-arena", runtime.c_arena(bank))?;
            at(Some(round), "poison-C", gpu_copy(cv.device, poison.ptr(), layout.c_bytes))?;
            at(Some(round), "GPU-write-A", gpu_copy(av.device, input.ptr(), layout.a_bytes))?;
            let seq = at(Some(round), "kick", unsafe { runtime.kick(0, Shape::GdnIn, bank, ROWS, stream.raw()) })?;
            let completion = at(Some(round), "wait-GPU", unsafe { runtime.wait_gpu(seq, stream.raw()) })?;
            at(Some(round), "GPU-read-C", gpu_copy(output.ptr(), cv.device, layout.c_bytes))?;
            at(Some(round), "retire-bank", unsafe { runtime.retire_bank(seq, stream.raw()) })?;
            last_seq[bank] = Some(seq);
            at(Some(round), "consumer-completion", drain())?;
            at(Some(round), "download-C", hip.download(&output, &mut observed))?;
            // Also prove the entire GPU-written A aliases the anonymous pages.
            unsafe { clflush(av.host, av.len); }
            if unsafe { core::slice::from_raw_parts(av.host, av.len) } != &packed[..] {
                return at(Some(round), "compare-A-alias", Err(format!("bank {bank}: GPU A/userptr alias mismatch")));
            }
            for panel in 0..PANELS { for row in 0..ROWS { for col in 0..N {
                let offset = panel * layout.c_panel_bytes + gemm_g80::bf16_output_offset(ROWS, row, col);
                let actual = u16::from_le_bytes([observed[offset], observed[offset + 1]]);
                let expected = reference[((panel * CLASSES + row % CLASSES) * CLASSES + col % CLASSES) * 3 + row % 3];
                if actual != expected {
                    return at(Some(round), "compare-C", Err(format!("bank {bank} panel {panel} row {row} col {col}: {actual:04x} != {expected:04x}")));
                }
            } } }
            let ms = completion.kick_to_done.as_secs_f64() * 1000.0;
            busy += completion.npu_busy;
            latencies.push(ms);
            println!("round={round:02} bank={bank} seq={} exact kick->done={ms:.3} ms npu_submit_wait={:.3} ms", seq.value, completion.npu_busy.as_secs_f64() * 1000.0);
        }
        latencies.sort_by(f64::total_cmp);
        println!("EXACT rounds={ROUNDS} p50={:.3} p99={:.3} max={:.3} ms npu_busy_envelope={:.3} ms wall={:.3} ms (busy=submit-to-timeline host envelope, wall includes CPU oracle/copies)",
            percentile(&latencies, 0.5), percentile(&latencies, 0.99), latencies[ROUNDS - 1], busy.as_secs_f64() * 1000.0, wall.elapsed().as_secs_f64() * 1000.0);
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
        let cv = at(None, "negative-C-arena", runtime.c_arena(0))?;
        at(None, "negative-poison-C", gpu_copy(cv.device, poison.ptr(), layout.c_bytes))?;
        at(None, "negative-GPU-read-C", gpu_copy(output.ptr(), cv.device, layout.c_bytes))?;
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
        println!("PASS rounds={ROUNDS} negatives=2");
        Ok(())
    })();
    let teardown = at(None, "teardown", runtime.close());
    outcome.and(teardown)
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let pci = match args.as_slice() {
        [] => "0000:c0:00.0",
        [flag, pci] if flag == "--pci" => pci,
        [flag] if flag == "--help" => { println!("npu-trunk-transport [--pci 0000:c0:00.0]\n8192 rows, 13 BF16 G80 panels, 2 banks, 20 rounds FIRST, negative controls AFTER, 3 s waits, default shim AXI."); return; }
        _ => { eprintln!("FAIL: stage=arguments round=none: usage: npu-trunk-transport [--pci PCI]"); std::process::exit(1); }
    };
    if let Err(e) = run(pci) { eprintln!("FAIL: {e}"); std::process::exit(1); }
}
