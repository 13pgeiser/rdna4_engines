// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The committed `.rip` twin of the gfx1201 MQ4 down NT4/NT8 entries of `qwen4_moe_sym`
//! (`kernels/qwen4_moe_down_gp.rip`, the K-loop composed from the `qdown::*` natives) re-emits the
//! Rust generators' bytes: `qwen4_moe_sym::emit` for the flat-slot entries and `emit_down_gp` for
//! the grouped-position twins.

use peacemaker_isa::kernels::qwen4_moe_sym::{self as moe, Kind, Spec, WeightCodec};
use peacemaker_isa::Arch;

const TWIN: &str = include_str!("../../peacemaker-isa/src/kernels/qwen4_moe_down_gp.rip");

fn spec(nt: u8) -> Spec {
    Spec {
        arch: Arch::Gfx1201,
        kind: Kind::Down,
        nt,
        rr: 1,
        codec: WeightCodec::Mq4Sym,
    }
}

/// `(variant, spec, gp)` for every entry the twin declares.
fn entries() -> Vec<(String, Spec, bool)> {
    let mut out = Vec::new();
    for nt in [4u8, 8] {
        out.push((format!("down_nt{nt}"), spec(nt), false));
        out.push((format!("down_gp_nt{nt}"), spec(nt), true));
    }
    out
}

fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: rust `{x}` vs rip `{y}`", i + 1);
        }
    }
    format!(
        "lengths {} vs {} lines",
        a.lines().count(),
        b.lines().count()
    )
}

/// `s_text` and the builder proof are byte-identical for every entry.
#[test]
fn qwen4_moe_down_twin_matches_the_rust_emit() {
    for (name, spec, gp) in entries() {
        let rust = if gp {
            moe::emit_down_gp(spec)
        } else {
            moe::emit(spec)
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = peacemaker_rip::compile_named(TWIN, "qwen4_moe_down_gp.rip", spec.arch, &name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            rust.s_text == rip.s_text,
            "{name}: {}",
            first_difference(&rust.s_text, &rip.s_text)
        );
        assert_eq!(
            serde_json::to_value(&rust.proof).unwrap(),
            serde_json::to_value(&rip.proof).unwrap(),
            "{name}: proof"
        );
    }
}

#[test]
fn the_twin_declares_exactly_its_entries() {
    let mut declared = peacemaker_rip::variants(TWIN, "qwen4_moe_down_gp.rip").unwrap();
    let mut want: Vec<String> = entries().into_iter().map(|e| e.0).collect();
    declared.sort();
    want.sort();
    assert_eq!(declared, want);
}

#[test]
fn a_malformed_variant_is_refused_with_its_grammar() {
    for (v, want) in [
        ("down", "expected down[_gp]_ntN"),
        ("gate_up_nt4", "expected down[_gp]_ntN"),
        ("down_nt1", "unknown tile width `nt1`"),
        ("down_gp_nt16", "unknown tile width `nt16`"),
    ] {
        let e = peacemaker_rip::compile_named(TWIN, "qwen4_moe_down_gp.rip", Arch::Gfx1201, v)
            .err()
            .unwrap_or_else(|| panic!("{v} built"));
        assert!(e.contains(want), "{v}: {e}");
    }
}

#[test]
fn the_twin_builds_on_gfx1201_only() {
    let e = peacemaker_rip::compile_named(TWIN, "qwen4_moe_down_gp.rip", Arch::Gfx1151, "down_nt4")
        .expect_err("gfx1151 must be refused");
    assert!(e.contains("gfx1201"), "{e}");
}

/// The loop is the script's: swapping the descriptor step and the second weight load changes the
/// program (or the builder refuses it); it never emits the same bytes.
#[test]
fn reordering_the_kloop_changes_the_program() {
    let edited = TWIN.replace(
        "            qdown::advance(wg)?;\n            qdown::load_w(wg, g, 0)?;\n",
        "            qdown::load_w(wg, g, 0)?;\n            qdown::advance(wg)?;\n",
    );
    assert_ne!(edited, TWIN, "the K-loop statements must exist in the twin");
    let name = "down_gp_nt4";
    let base = peacemaker_rip::compile_named(TWIN, "t.rip", Arch::Gfx1201, name).unwrap();
    if let Ok(moved) = peacemaker_rip::compile_named(&edited, "t.rip", Arch::Gfx1201, name) {
        assert_ne!(base.s_text, moved.s_text);
    }
}
