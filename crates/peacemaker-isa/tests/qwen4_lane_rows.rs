// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 lane-step row module `qwen4_lane_rows_pm`: frozen exports, the
//! kernarg ABI and launch bounds of every export, determinism, exact-arch
//! gating, the committed bundle's identity with fresh native emission, and
//! (feature `toolchain`) M7 on every export of the linked object.
use peacemaker_isa::kernels::qwen4_lane_rows::{
    self, KERNARGS, KERNARG_BYTES, MODULE, SYMBOLS, WORKGROUP_SIZE,
};
use peacemaker_isa::Arch;

const ARCH: Arch = Arch::Gfx1151;

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

#[test]
fn module_exports_keep_narrow_prefix_and_append_wide_twins() {
    assert_eq!(MODULE, "qwen4_lane_rows_pm");
    assert_eq!(
        SYMBOLS[..4],
        [
            "gemv_mq6g256v2_xbatch_pm",
            "gemv_mq6g256v2_x4_xbatch_pm",
            "gemv_q8_0_k8_xbatch_pm",
            "gemv_q8_0_k320_xbatch_pm"
        ]
    );
    assert_eq!(SYMBOLS.len(), 18);
    assert_eq!(
        qwen4_lane_rows::bundle_name(),
        "qwen4_lane_rows_pm_gfx1151.hxaco"
    );
    let (emitted, text, proof) = qwen4_lane_rows::emit_module().unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, ARCH));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
}

#[test]
fn emission_is_deterministic() {
    let (_, ta, pa) = qwen4_lane_rows::emit_module().unwrap();
    let (_, tb, pb) = qwen4_lane_rows::emit_module().unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
}

/// Every export is built for exact gfx1151 and wave32 without scratch.
#[test]
fn every_export_is_exact_gfx1151_wave32_without_scratch() {
    let (emitted, text, _) = qwen4_lane_rows::emit_module().unwrap();
    for (e, symbol) in emitted.iter().zip(SYMBOLS) {
        assert_eq!(e.proof.arch, ARCH, "{symbol}");
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".private_segment_fixed_size: 0\n"),
            "{symbol}: private segment"
        );
        assert!(
            e.s_text.contains(".sgpr_spill_count: 0\n")
                && e.s_text.contains(".vgpr_spill_count: 0\n"),
            "{symbol}: spills"
        );
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));
    assert_eq!(
        text.matches(".fill 96, 4, 3214868480").count(),
        1,
        "one module s_code_end guard"
    );
}

/// The kernarg block, per-argument offsets/sizes and the launch bound of every
/// export are the frozen contract's.
#[test]
fn kernarg_abi_and_launch_bounds_match_the_contract() {
    let (emitted, _, _) = qwen4_lane_rows::emit_module().unwrap();
    for (i, (e, symbol)) in emitted.iter().zip(SYMBOLS).enumerate() {
        let size = KERNARG_BYTES[i];
        assert!(
            e.s_text
                .contains(&format!(".kernarg_segment_size: {size}\n")),
            "{symbol}: kernarg size {size}"
        );
        let want: Vec<u32> = KERNARGS[i].iter().map(|a| a.0).collect();
        assert_eq!(
            field(&e.s_text, ".offset:"),
            want,
            "{symbol}: kernarg offsets"
        );
        let want: Vec<u32> = KERNARGS[i].iter().map(|a| a.1).collect();
        assert_eq!(field(&e.s_text, ".size:"), want, "{symbol}: kernarg sizes");
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [WORKGROUP_SIZE[i]],
            "{symbol}: workgroup size"
        );
        let (last_off, last_size) = *KERNARGS[i].last().unwrap();
        assert_eq!(
            last_off + last_size,
            size,
            "{symbol}: arguments cover the kernarg block"
        );
    }
    assert_eq!(KERNARG_BYTES[..4], [36, 96, 44, 32]);
    assert_eq!(WORKGROUP_SIZE[..4], [32, 32, 256, 64]);
}

/// Historical emitted-text hashes, captured before adding any wide export.
#[test]
fn narrow_emitted_text_is_unchanged() {
    let want = [
        "5859a235e4ca897446ac65f257679b0088d0b7d904db9a3b995452858665f16f",
        "a3bf1c1dae4e5a912b1a3081cb7c7bf9f3213a1b8f77a0980e5625ca18568e70",
        "3e9b52d257e8f97c1295d059efcfef2e1f5013b74a41f5dd0548f97bade39fce",
        "a53772ff34a7dbcffdbeedf77ffc83384f9ce7b984f88a1c71b55dcad76520ad",
    ];
    let (_, _, proof) = qwen4_lane_rows::emit_module().unwrap();
    for (kernel, hash) in proof.kernels.iter().take(4).zip(want) {
        assert_eq!(kernel.s_text_sha256, hash, "{}", kernel.variant);
    }
}

#[test]
fn wide_entries_keep_resource_and_barrier_contract() {
    let (emitted, _, _) = qwen4_lane_rows::emit_module().unwrap();
    for (i, e) in emitted.iter().enumerate().skip(4) {
        let symbol = SYMBOLS[i];
        let hc = symbol.starts_with("gemv_q8_0_k8_");
        assert_eq!(
            field(&e.s_text, ".group_segment_fixed_size:"),
            [if hc { 16384 } else { 8192 }],
            "{symbol}"
        );
        assert!(
            e.s_text.contains(".amdhsa_workgroup_processor_mode 1"),
            "{symbol}: WGP mode"
        );
        assert!(
            e.s_text.contains("s_barrier"),
            "{symbol}: cooperative X staging"
        );
        let ceiling = if symbol.contains("mq6") || symbol.contains("gemv_q8_0_wide_") {
            if symbol.contains("_r16_") {
                128
            } else {
                192
            }
        } else {
            80
        };
        let vgpr = field(&e.s_text, ".vgpr_count:");
        assert_eq!(vgpr.len(), 1, "{symbol}");
        assert!(vgpr[0] <= ceiling, "{symbol}: {} VGPR > {ceiling}", vgpr[0]);
    }
}

/// The native writer assembles the module and bundles it for the gfx1151 host entry.
#[test]
fn module_assembles_natively() {
    let (_, text, _) = qwen4_lane_rows::emit_module().unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let bundle =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(bundle.len() > elf.len());
}

/// The runtime embeds `kernels/qwen4_lane_rows_pm_gfx1151.hxaco`: it must be
/// exactly what the builders emit today (native writer, no ROCm needed).
#[test]
fn committed_bundle_equals_fresh_emission() {
    let path = format!(
        "{}/../../kernels/{}",
        env!("CARGO_MANIFEST_DIR"),
        qwen4_lane_rows::bundle_name()
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel qwen4_lane_rows --arch gfx1151 ...)"));
    let (_, text, _) = qwen4_lane_rows::emit_module().unwrap();
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    let fresh =
        peacemaker_isa::native::bundle(&elf, ARCH, peacemaker_isa::native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check};

    /// Every export's waits replay independently, and M7 lifts the linked
    /// object byte-exactly with no outstanding obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = qwen4_lane_rows::emit_module().unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, ARCH).unwrap();
        }
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-lane-rows-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, peacemaker_isa::native::assemble(&text, ARCH).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 =
                pm_check::m7(&co, ARCH.name(), symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
    }
}
