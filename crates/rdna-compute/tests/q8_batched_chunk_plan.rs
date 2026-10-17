// SPDX-License-Identifier: Apache-2.0
// #732: the scalar `gemm_q8_0_batched` kernel takes at most 64 activation rows
// and asserts above that. Every caller that can see more rows goes through
// `Gpu::gemm_q8_0_batched_f32_chunked`, whose split is `q8_0_batched_chunks`.
// CPU-only: no HIP, no GPU.
use rdna_compute::gemm::{q8_0_batched_chunks, Q8_0_BATCHED_MAX_BATCH};
use std::path::{Path, PathBuf};

const BOUNDARY_ROWS: [usize; 5] = [63, 64, 65, 70, 129];

/// Every launch is within the kernel cap; launches are consecutive, start at
/// row 0 and end at `n`, so each row is computed exactly once. Also checks the
/// element offsets the wrapper derives (`off*k` into x, `off*m` into y) stay in
/// the `[n, k]` / `[n, m]` buffers.
fn assert_plan(n: usize, expect: &[(usize, usize)]) {
    let plan: Vec<(usize, usize)> = q8_0_batched_chunks(n).collect();
    assert_eq!(plan, expect, "plan for n={n}");
    let mut next = 0;
    for &(off, take) in &plan {
        assert_eq!(off, next, "n={n}: launch must start where the last ended");
        assert!(
            (1..=Q8_0_BATCHED_MAX_BATCH).contains(&take),
            "n={n}: launch of {take} rows breaks the kernel cap"
        );
        next = off + take;
    }
    assert_eq!(next, n, "n={n}: rows covered");
    let (k, m) = (2560, 248_320);
    for &(off, take) in &plan {
        assert!((off + take) * k <= n * k, "n={n}: x window past the input");
        assert!((off + take) * m <= n * m, "n={n}: y window past the output");
    }
}

#[test]
fn kernel_cap_is_64() {
    assert_eq!(Q8_0_BATCHED_MAX_BATCH, 64);
}

#[test]
fn boundary_row_counts_plan() {
    assert_plan(63, &[(0, 63)]);
    assert_plan(64, &[(0, 64)]);
    assert_plan(65, &[(0, 64), (64, 1)]);
    assert_plan(70, &[(0, 64), (64, 6)]);
    assert_plan(129, &[(0, 64), (64, 64), (128, 1)]);
    for n in BOUNDARY_ROWS {
        assert!(q8_0_batched_chunks(n).all(|(_, t)| t <= 64));
    }
}

#[test]
fn small_and_empty_row_counts_plan() {
    assert_plan(0, &[]);
    assert_plan(1, &[(0, 1)]);
    assert_plan(128, &[(0, 64), (64, 64)]);
    assert_plan(256, &[(0, 64), (64, 64), (128, 64), (192, 64)]);
}

#[test]
fn every_row_count_to_1024_is_covered_once() {
    for n in 0..=1024usize {
        let mut seen = vec![0u8; n];
        for (off, take) in q8_0_batched_chunks(n) {
            assert!((1..=64).contains(&take));
            for row in &mut seen[off..off + take] {
                *row += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "n={n}");
    }
}

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_files(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// A bare `gemm_q8_0_batched(` call asserts above 64 rows. The only callers
/// allowed are the wrapper that does the splitting, the EAGLE-strict sandwich
/// loop (driven by `q8_0_batched_chunks`) and the bare-kernel channel example,
/// which splits by hand. Any other production caller must use
/// `gemm_q8_0_batched_f32_chunked` (or `_chunked`).
#[test]
fn no_unchunked_production_caller_of_the_bare_q8_kernel() {
    let crates = crates_dir();
    let allowed = [
        "rdna-compute/src/gemm.rs",
        "hipfire-dispatch/src/pipeline/sandwich.rs",
        "rdna-compute/examples/channel_q8_mmq_gfx1030.rs",
        "rdna-compute/tests/q8_batched_chunk_plan.rs",
    ];
    let mut files = Vec::new();
    rust_files(&crates, &mut files);
    assert!(files.len() > 100, "crate scan found {} files", files.len());
    let mut offenders = Vec::new();
    for f in files {
        let rel = f.strip_prefix(&crates).unwrap().to_string_lossy().replace('\\', "/");
        if allowed.contains(&rel.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("gemm_q8_0_batched(") {
                offenders.push(format!("{rel}:{}", i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "bare gemm_q8_0_batched caller(s) without a 64-row split: {offenders:?}"
    );
}

#[test]
fn bare_kernel_users_split_with_the_shared_plan() {
    let crates = crates_dir();
    let sandwich =
        std::fs::read_to_string(crates.join("hipfire-dispatch/src/pipeline/sandwich.rs")).unwrap();
    assert!(sandwich.contains("q8_0_batched_chunks(rows)"));

    // The wrapper itself splits with the shared plan.
    let gemm = std::fs::read_to_string(crates.join("rdna-compute/src/gemm.rs")).unwrap();
    assert!(gemm.contains("for (off, take) in q8_0_batched_chunks(n)"));

    // The `GemmQ8_0Batched` dispatch key runs the chunked wrapper.
    let family =
        std::fs::read_to_string(crates.join("hipfire-dispatch/src/families/gemm.rs")).unwrap();
    let arm = family
        .split("K::GemmQ8_0Batched =>")
        .nth(1)
        .expect("GemmQ8_0Batched arm");
    let arm = arm.split("K::GemmHfq4G256BatchedLmhead").next().unwrap();
    assert!(arm.contains("gemm_q8_0_batched_f32_chunked"), "arm: {arm}");

    // The two #732 call sites route through the chunked wrapper.
    let batch =
        std::fs::read_to_string(crates.join("hipfire-arch-qwen35/src/qwen35/batch.rs")).unwrap();
    let lm = batch.split("fn lm_head_batched").nth(1).unwrap();
    let lm = lm.split("DType::HFQ4G256").next().unwrap();
    assert!(lm.contains("gemm_q8_0_batched_f32_chunked"), "qwen35 lm_head_batched");
    let gemma =
        std::fs::read_to_string(crates.join("hipfire-arch-gemma4/src/forward.rs")).unwrap();
    assert!(gemma.contains("let lm_result = gpu.gemm_q8_0_batched_f32_chunked("));
}
