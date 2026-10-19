// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 symmetric IU4 MoE builder family (gfx1151 + gfx1201).
use peacemaker_isa::kernels::qwen4_moe_sym::{self, Kind, Spec};
use peacemaker_isa::Arch;
#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;

const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

fn specs(arch: Arch) -> impl Iterator<Item = Spec> {
    qwen4_moe_sym::module_specs(arch).into_iter()
}

#[test]
fn entries_are_deterministic_and_target_only_their_arches() {
    for arch in ARCHES {
        for spec in specs(arch) {
            let (a, b) = (
                qwen4_moe_sym::emit(spec).unwrap(),
                qwen4_moe_sym::emit(spec).unwrap(),
            );
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
            // Only the expert-run gate/up hands its gate rows over LDS: one
            // barrier on each of the gate and up paths.
            let handoff = spec.kind == Kind::GateUp && spec.nt > 1;
            assert_eq!(a.shape.barriers, if handoff { 2 } else { 0 }, "{spec:?}");
            assert_eq!(
                a.s_text.contains("ds_store") || a.s_text.contains("ds_load"),
                handoff,
                "{spec:?}"
            );
        }
    }
    assert!(qwen4_moe_sym::emit(Spec {
        arch: Arch::Gfx1100,
        kind: Kind::Down,
        nt: 1,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym
    })
    .is_err());
    // gfx11 NT8 does not fit the VGPR file.
    assert!(qwen4_moe_sym::emit(Spec {
        arch: Arch::Gfx1151,
        kind: Kind::GateUp,
        nt: 8,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym
    })
    .is_err());
}

/// Row repeats exist only as 2 or 4 blocks of the gfx1151 down NT4 entry.
#[test]
fn row_repeat_shapes_are_validated() {
    let spec = |arch, kind, nt, rr| Spec {
        arch,
        kind,
        nt,
        rr,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
    };
    for rr in [0, 3, 8] {
        assert!(
            qwen4_moe_sym::emit(spec(Arch::Gfx1151, Kind::Down, 4, rr)).is_err(),
            "rr {rr}"
        );
    }
    for bad in [
        spec(Arch::Gfx1151, Kind::GateUp, 4, 2),
        spec(Arch::Gfx1151, Kind::Down, 1, 2),
        spec(Arch::Gfx1151, Kind::Down, 2, 2),
        spec(Arch::Gfx1201, Kind::Down, 4, 2),
        spec(Arch::Gfx1201, Kind::Down, 8, 4),
    ] {
        assert!(qwen4_moe_sym::emit(bad).is_err(), "{bad:?}");
    }
}

/// The runtime embeds one certified module per arch: it must be exactly
/// what the builder emits today.
#[test]
fn committed_bundles_equal_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    for arch in ARCHES {
        let module = Spec::module(arch);
        let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
        let text = qwen4_moe_sym::emit_module(arch).unwrap().1;
        assert!(
            text_section(&link(&text, arch.name(), &module)) == text_section(&committed),
            "{module}: fresh emission differs from the committed bundle"
        );
    }
}

fn w3_specs_for(arch: Arch) -> Vec<Spec> {
    qwen4_moe_sym::module_specs_codec(arch, qwen4_moe_sym::WeightCodec::Mq3I4)
}

/// The MQ3-I4 codec is built for the gfx1151 and gfx1201 NT1 gate/up and down entries only,
/// emits deterministically and never changes the MQ4 module (its symbols are not in it).
#[test]
fn w3_entry_is_deterministic_and_isolated_from_the_mq4_module() {
    use qwen4_moe_sym::WeightCodec::Mq3I4;
    for arch in ARCHES {
        let name = arch.name();
        let specs = w3_specs_for(arch);
        assert_eq!(specs.len(), 2);
        let spec = specs[0];
        assert_eq!(
            spec.symbol(),
            format!("qwen4_moe_gate_up_silu_iu4_sym_w3_pm_{name}")
        );
        assert_eq!(
            specs[1].symbol(),
            format!("qwen4_moe_down_iu4_sym_w3_pm_{name}")
        );
        assert_eq!(
            Spec::module_codec(arch, Mq3I4),
            format!("qwen4_moe_iu4_sym_w3_pm_{name}")
        );
        for spec in specs {
            let (a, b) = (
                qwen4_moe_sym::emit(spec).unwrap(),
                qwen4_moe_sym::emit(spec).unwrap(),
            );
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
            assert_eq!(
                (
                    a.shape.barriers,
                    a.s_text.contains("ds_store") || a.s_text.contains("ds_load")
                ),
                (0, false)
            );
            let acc = peacemaker_isa::kernels::qwen4_moe_sym::w3::emit_acc(spec).unwrap();
            assert!(acc.s_text.contains(&format!("{}_acc", spec.symbol())));
        }
        for bad in [
            Spec {
                arch: Arch::Gfx1100,
                ..spec
            },
            Spec { nt: 4, ..spec },
            Spec { rr: 2, ..spec },
        ] {
            assert!(qwen4_moe_sym::emit(bad).is_err(), "{bad:?}");
        }
        let mq4 = qwen4_moe_sym::emit_module(arch).unwrap().1;
        assert!(
            !mq4.contains("_w3"),
            "the MQ4 module must not mention the W3 entries"
        );
    }
    assert!(w3_specs_for(Arch::Gfx1100).is_empty());
}

/// The runtime embeds the W3 module: it must be exactly what the builder emits.
#[test]
fn committed_w3_bundle_equals_fresh_emission() {
    let _ = require_rocm_tool!("llvm-mc");
    let _ = require_rocm_tool!("ld.lld");
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let module = Spec::module_codec(Arch::Gfx1201, qwen4_moe_sym::WeightCodec::Mq3I4);
    let committed = std::fs::read(format!("{root}/{module}.hxaco")).unwrap();
    let text = qwen4_moe_sym::emit_module_codec(Arch::Gfx1201, qwen4_moe_sym::WeightCodec::Mq3I4)
        .unwrap()
        .1;
    assert!(
        text_section(&link(&text, "gfx1201", &module)) == text_section(&committed),
        "{module}: fresh emission differs from the committed bundle"
    );
}

/// The whole shipped offload bundle (host entry `host-x86_64-unknown-linux-gnu`, device
/// `hipv4-amdgcn-amd-amdhsa--gfx1151`, `.comment` `hipfire peacemaker native-emit 0.4.1`) equals the fresh native
/// emission byte for byte (native writer, no ROCm).
fn assert_committed_gfx1151_bundle_is_fresh(module: &str, text: &str) {
    let path = format!(
        "{}/../../kernels/{module}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let elf = peacemaker_isa::native::assemble(text, Arch::Gfx1151).unwrap();
    let fresh = peacemaker_isa::native::bundle(
        &elf,
        Arch::Gfx1151,
        peacemaker_isa::native::DEFAULT_HOST_TARGET,
    );
    assert!(
        committed == fresh,
        "{module}: fresh emission differs from the committed bundle {path}"
    );
}

/// `kernels/qwen4_moe_iu4_sym_w3_pm_gfx1151.hxaco` (the MQ3-I4 NT1 gate/up and down entries and their F32 `_acc`
/// twins) is exactly what the builder emits today.
#[test]
fn committed_gfx1151_w3_bundle_equals_fresh_emission() {
    let arch = Arch::Gfx1151;
    let module = Spec::module_codec(arch, qwen4_moe_sym::WeightCodec::Mq3I4);
    let text = qwen4_moe_sym::emit_module_codec(arch, qwen4_moe_sym::WeightCodec::Mq3I4)
        .unwrap()
        .1;
    assert_committed_gfx1151_bundle_is_fresh(&module, &text);
}

/// gfx1151 MQ3-I4 NT4 entries: module name, symbols in module order, bit-deterministic emission (with the
/// `_acc` twins of the `rr == 1` entries), the module's symbol/variant order, and refusal of every other spec.
/// gfx1201 and gfx1100 have no NT4 entries.
#[test]
fn w3_nt4_gfx1151_entries_module_order_and_invalid_specs() {
    use peacemaker_isa::kernels::qwen4_moe_sym::{
        w3::nt4,
        WeightCodec::{Mq3I4, Mq4Sym},
    };
    let arch = Arch::Gfx1151;
    assert_eq!(
        nt4::module_name(arch),
        "qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151"
    );
    let specs = nt4::module_specs(arch);
    let spec = |kind, rr| Spec {
        arch,
        kind,
        nt: 4,
        rr,
        codec: Mq3I4,
    };
    assert_eq!(
        specs,
        [
            spec(Kind::GateUp, 1),
            spec(Kind::Down, 1),
            spec(Kind::Down, 2),
            spec(Kind::Down, 4)
        ]
    );
    assert_eq!(
        specs.iter().map(|s| s.symbol()).collect::<Vec<_>>(),
        [
            "qwen4_moe_gate_up_silu_iu4_sym_w3_pm_gfx1151_nt4",
            "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4",
            "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4r2",
            "qwen4_moe_down_iu4_sym_w3_pm_gfx1151_nt4r4",
        ]
    );
    for other in [Arch::Gfx1201, Arch::Gfx1100] {
        assert!(nt4::module_specs(other).is_empty(), "{other:?}");
        assert!(nt4::emit_module(other).is_err(), "{other:?}");
        for s in &specs {
            let s = Spec { arch: other, ..*s };
            assert!(nt4::emit(s).is_err() && nt4::emit_acc(s).is_err(), "{s:?}");
        }
    }
    for s in &specs {
        let (a, b) = (nt4::emit(*s).unwrap(), nt4::emit(*s).unwrap());
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{s:?}");
        assert_eq!(a.s_text, b.s_text, "{s:?}");
        assert!(a.s_text.contains(&s.symbol()), "{s:?}");
        if s.rr == 1 {
            let (a, b) = (nt4::emit_acc(*s).unwrap(), nt4::emit_acc(*s).unwrap());
            assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{s:?} acc");
            assert!(a.s_text.contains(&format!("{}_acc", s.symbol())), "{s:?}");
        } else {
            assert!(
                nt4::emit_acc(*s).is_err(),
                "{s:?}: no F32 twin for the row repeats"
            );
        }
    }
    let (emitted, text, proof) = nt4::emit_module(arch).unwrap();
    assert_eq!(proof.module, "qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151");
    let variants: Vec<_> = emitted.iter().map(|e| e.proof.variant.clone()).collect();
    assert_eq!(
        variants,
        [
            "gate_up_nt4_w3",
            "down_nt4_w3",
            "down_nt4r2_w3",
            "down_nt4r4_w3",
            "gate_up_nt4_w3_acc",
            "down_nt4_w3_acc"
        ]
    );
    for s in &specs {
        assert!(
            text.contains(&format!(".amdhsa_kernel {}\n", s.symbol())),
            "{s:?}"
        );
        if s.rr == 1 {
            assert!(
                text.contains(&format!(".amdhsa_kernel {}_acc\n", s.symbol())),
                "{s:?} acc"
            );
        }
    }
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 6);
    let (_, again, _) = nt4::emit_module(arch).unwrap();
    assert_eq!(text, again);
    // The NT4 module is its own: the NT1 W3 and MQ4 modules never mention an `_nt4` W3 entry.
    assert!(!qwen4_moe_sym::emit_module_codec(arch, Mq3I4)
        .unwrap()
        .1
        .contains("_w3_pm_gfx1151_nt4"));
    assert!(!qwen4_moe_sym::emit_module(arch).unwrap().1.contains("_w3"));
    // Invalid specs: the wrong tile width, codec, row repeat, kind/row-repeat pairing and architecture.
    let down = spec(Kind::Down, 1);
    for bad in [
        Spec { nt: 1, ..down },
        Spec { nt: 2, ..down },
        Spec { nt: 8, ..down },
        Spec {
            codec: Mq4Sym,
            ..down
        },
        Spec {
            codec: Mq4Sym,
            ..spec(Kind::GateUp, 1)
        },
        Spec { rr: 0, ..down },
        Spec { rr: 3, ..down },
        Spec { rr: 8, ..down },
        spec(Kind::GateUp, 2),
        spec(Kind::GateUp, 4),
    ] {
        assert!(
            nt4::emit(bad).is_err() && nt4::emit_acc(bad).is_err(),
            "{bad:?}"
        );
    }
}

/// `kernels/qwen4_moe_iu4_sym_w3_nt4_pm_gfx1151.hxaco` (gate/up, down, the down `nt4r2`/`nt4r4` row repeats, and the
/// `_acc` twins of gate/up and down) is exactly what the builder emits today.
#[test]
fn committed_gfx1151_w3_nt4_bundle_equals_fresh_emission() {
    use peacemaker_isa::kernels::qwen4_moe_sym::w3::nt4;
    let arch = Arch::Gfx1151;
    assert_committed_gfx1151_bundle_is_fresh(
        &nt4::module_name(arch),
        &nt4::emit_module(arch).unwrap().1,
    );
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("peacemaker-isa-q4s-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    let llvm_mc = rocm::rocm_tool("llvm-mc").expect("ROCm llvm-mc");
    let linker = rocm::rocm_tool("ld.lld").expect("ROCm ld.lld");
    assert!(Command::new(llvm_mc)
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(linker)
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
    std::fs::read(co).unwrap()
}

/// `.text` of an ELF, or of the device ELF inside a clang offload bundle.
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let start = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[start..];
    let u16_ = |o: usize| u16::from_le_bytes([e[o], e[o + 1]]) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(0x28), u16_(0x3a), u16_(0x3c), u16_(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{
        ledger_replay, pm_check,
        toolchain::{build, certify, Toolchain},
    };

    /// Every symbol of both modules certifies: parse-back, independent wait
    /// replay, its committed shape contract, and M7's lift identity plus
    /// wait/hazard/barrier/window analyses of the linked ELF, obligation-free.
    #[test]
    fn modules_pass_certification_for_every_symbol() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) = qwen4_moe_sym::emit_module(arch).unwrap();
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-q4s-cert-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let toolchain = Toolchain::oracle();
            let build = build(&toolchain, &s, &dir.join("module.hsaco"), arch.name()).unwrap();
            for spec in specs(arch) {
                let symbol = spec.symbol();
                let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                    .unwrap_or_else(|e| panic!("{symbol}: {e}"));
                assert_eq!(
                    (m7["lift"].as_str(), &m7["obligations"]),
                    (Some("byte-exact"), &serde_json::json!({})),
                    "{symbol}"
                );
                let tag = spec.variant();
                let path = format!(
                    "{}/kernels/qwen4_moe_sym.{}.{tag}.contract.json",
                    env!("CARGO_MANIFEST_DIR"),
                    arch.name()
                );
                let contract = serde_json::from_slice(
                    &std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
                )
                .unwrap();
                certify(
                    &toolchain,
                    &build,
                    &s,
                    arch.name(),
                    &dir.join(format!("{tag}.manifest.json")),
                    Some(&contract),
                    "test",
                    "test",
                )
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            }
        }
    }

    /// Both MQ3-I4 symbols (the shipped entry and its F32 qualification entry)
    /// lift byte-exact with no M7 obligations and replay their wait ledger.
    #[test]
    fn w3_module_passes_m7_for_both_symbols() {
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        for arch in ARCHES {
            let (emitted, text, _) =
                qwen4_moe_sym::emit_module_codec(arch, qwen4_moe_sym::WeightCodec::Mq3I4).unwrap();
            assert_eq!(emitted.len(), 4);
            for e in &emitted {
                ledger_replay::replay_waits(&e.s_text, arch).unwrap();
            }
            let dir = std::env::temp_dir().join(format!(
                "peacemaker-isa-q4s-w3-{}-{}",
                arch.name(),
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let s = dir.join("module.s");
            std::fs::write(&s, &text).unwrap();
            let build = build(
                &Toolchain::oracle(),
                &s,
                &dir.join("module.hsaco"),
                arch.name(),
            )
            .unwrap();
            for spec in w3_specs_for(arch) {
                let base = spec.symbol();
                for symbol in [base.clone(), format!("{base}_acc")] {
                    let m7 = pm_check::m7(&build.elf, arch.name(), &symbol)
                        .unwrap_or_else(|e| panic!("{symbol}: {e}"));
                    assert_eq!(
                        (m7["lift"].as_str(), &m7["obligations"]),
                        (Some("byte-exact"), &serde_json::json!({})),
                        "{symbol}"
                    );
                }
            }
        }
    }

    /// Every symbol of the gfx1151 MQ3-I4 NT4 module (gate/up, down, the down `nt4r2`/`nt4r4` row repeats and the
    /// `_acc` twins of the two `rr == 1` entries) replays its wait ledger independently of the builder (source and
    /// module text) and lifts byte-exact under M7 with no obligations, on the ROCm-linked module.
    #[test]
    fn w3_nt4_gfx1151_module_passes_replay_and_m7_for_every_symbol() {
        use peacemaker_isa::kernels::qwen4_moe_sym::w3::nt4;
        let _ = require_rocm_tool!("llvm-mc");
        let _ = require_rocm_tool!("ld.lld");
        let _ = require_rocm_tool!("clang-offload-bundler");
        let _ = require_rocm_tool!("llvm-objdump");
        let _ = require_rocm_tool!("llvm-readobj");
        let arch = Arch::Gfx1151;
        let nt4_specs = nt4::module_specs(arch);
        let symbols: Vec<String> = nt4_specs
            .iter()
            .map(|s| s.symbol())
            .chain(
                nt4_specs
                    .iter()
                    .filter(|s| s.rr == 1)
                    .map(|s| format!("{}_acc", s.symbol())),
            )
            .collect();
        assert_eq!(symbols.len(), 6);
        let (emitted, text, _) = nt4::emit_module(arch).unwrap();
        assert_eq!(emitted.len(), symbols.len());
        for (e, symbol) in emitted.iter().zip(&symbols) {
            assert!(e.s_text.contains(symbol.as_str()), "{symbol}");
            ledger_replay::replay_waits(&e.s_text, arch)
                .unwrap_or_else(|err| panic!("{symbol}: {err}"));
        }
        ledger_replay::replay_waits(&text, arch).unwrap_or_else(|err| panic!("module: {err}"));
        let dir = std::env::temp_dir().join(format!(
            "peacemaker-isa-q4s-w3nt4-{}-{}",
            arch.name(),
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("module.s");
        std::fs::write(&s, &text).unwrap();
        let build = build(
            &Toolchain::oracle(),
            &s,
            &dir.join("module.hsaco"),
            arch.name(),
        )
        .unwrap();
        for symbol in &symbols {
            let m7 = pm_check::m7(&build.elf, arch.name(), symbol)
                .unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
