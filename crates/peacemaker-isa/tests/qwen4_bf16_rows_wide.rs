// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Structural tests of the BF16 wide verify GEMV twins, module `bf16_rows_wide_pm_gfx1151` (gfx1151 only):
//! the ordinary x4 entries `gemv_bf16_xf32_x4_rows_wide_r{16,32}_pm_gfx1151` and the k4 HC entries
//! `gemv_bf16_xf32_k4_rows_wide_r{16,32}_pm_gfx1151`. Emitter metadata, ABI, resources, proof obligations
//! (barrier/EXEC discipline, row masking, early returns), the per-row source DAG, instruction counts, weight-load
//! reuse, and the committed contracts / bundle.
//!
//! These tests establish structure only. Nothing here certifies numerical equality with R independent
//! single-row launches; that is the GPU row oracle's job.
use peacemaker_isa::kernels::qwen4_bf16_rows_wide::{self, Spec};
use peacemaker_isa::Arch;

const ARCH: Arch = Arch::Gfx1151;
const CAPS: [u8; 2] = [16, 32];
const MODULE: &str = "bf16_rows_wide_pm_gfx1151";

/// Ordinary ABI: `A0..A3 x y0..y3` pointers, then `M0 M1 M2 M3 K R` i32.
const X4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 8),
    (32, 8),
    (40, 8),
    (48, 8),
    (56, 8),
    (64, 8),
    (72, 4),
    (76, 4),
    (80, 4),
    (84, 4),
    (88, 4),
    (92, 4),
];
/// HC k4 ABI: `W X Y` pointers, then `M K scale hc_act R` 4-byte values.
const K4_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 4),
    (28, 4),
    (32, 4),
    (36, 4),
    (40, 4),
];

fn spec(cap: u8) -> Spec {
    Spec { arch: ARCH, cap }
}
fn x4(cap: u8) -> peacemaker_isa::Emitted {
    qwen4_bf16_rows_wide::emit(spec(cap)).unwrap()
}
fn k4(cap: u8) -> peacemaker_isa::Emitted {
    qwen4_bf16_rows_wide::emit_hc(spec(cap)).unwrap()
}
/// Row tiles of eight rows (ordinary) per capacity.
fn tiles(cap: u8) -> usize {
    usize::from(cap) / 8
}
/// Four-row stages (HC) per capacity.
fn stages(cap: u8) -> usize {
    usize::from(cap) / 4
}

/// Instruction lines of the emitted assembly (everything before the `.rodata`), labels and directives removed.
fn instructions(text: &str) -> Vec<&str> {
    let body = &text[..text.find(".section .rodata").unwrap()];
    body.lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && !l.starts_with('.') && !l.starts_with(';') && !l.ends_with(':')
        })
        .collect()
}

fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(lines: &[&str], m: &str) -> usize {
    lines.iter().filter(|l| mnemonic(l) == m).count()
}
fn count_exact(lines: &[&str], line: &str) -> usize {
    lines.iter().filter(|l| **l == line).count()
}
fn operands(line: &str) -> Vec<&str> {
    line.strip_prefix(mnemonic(line))
        .unwrap()
        .split(',')
        .map(str::trim)
        .collect()
}
/// `v40` -> 40.
fn vreg(op: &str) -> u32 {
    op.split_whitespace()
        .next()
        .unwrap()
        .strip_prefix('v')
        .unwrap_or_else(|| panic!("not a VGPR: {op}"))
        .parse()
        .unwrap()
}
/// The first instruction after `i`, ignoring scheduling hints.
fn after<'a>(lines: &[&'a str], i: usize) -> &'a str {
    lines[i + 1..]
        .iter()
        .copied()
        .find(|l| mnemonic(l) != "s_delay_alu" && mnemonic(l) != "s_nop")
        .unwrap()
}
fn position(lines: &[&str], pred: impl Fn(&str) -> bool) -> usize {
    lines
        .iter()
        .position(|l| pred(l))
        .expect("instruction present")
}
fn kernel_field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}

/// `(index, [acc, x, w])` of every `v_fmac_f32_e32` with three VGPR operands, in program order.
fn fmacs(lines: &[&str]) -> Vec<(usize, [u32; 3])> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| mnemonic(l) == "v_fmac_f32_e32")
        .map(|(i, l)| {
            let o = operands(l);
            (i, [vreg(o[0]), vreg(o[1]), vreg(o[2])])
        })
        .collect()
}

/// The ABI/geometry fields every entry's descriptor and metadata carry.
fn check_abi(
    t: &str,
    what: &str,
    args: &[(u32, u32)],
    kernarg: u32,
    reads: usize,
    writes: usize,
    lds: u32,
) {
    assert!(
        t.contains(&format!(".kernarg_segment_size: {kernarg}\n"))
            && t.contains(&format!(".amdhsa_kernarg_size {kernarg}\n")),
        "{what}: kernarg size"
    );
    assert_eq!(
        kernel_field(t, ".offset:"),
        args.iter().map(|a| a.0).collect::<Vec<_>>(),
        "{what}: offsets"
    );
    assert_eq!(
        kernel_field(t, ".size:"),
        args.iter().map(|a| a.1).collect::<Vec<_>>(),
        "{what}: sizes"
    );
    assert_eq!(
        t.matches(".actual_access: read_only\n").count(),
        reads,
        "{what}: read-only pointers"
    );
    assert_eq!(
        t.matches(".actual_access: write_only\n").count(),
        writes,
        "{what}: write-only pointers"
    );
    assert!(
        t.contains(".max_flat_workgroup_size: 256\n"),
        "{what}: block 256"
    );
    assert!(
        t.contains(&format!(".group_segment_fixed_size: {lds}\n"))
            && t.contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")),
        "{what}: LDS {lds}"
    );
    assert!(
        t.contains(".workgroup_processor_mode: 1\n"),
        "{what}: WGP mode (cu_mode false)"
    );
    assert!(
        t.contains(".wavefront_size: 32\n") && t.contains(".amdhsa_wavefront_size32 1\n"),
        "{what}: wave32"
    );
    assert!(
        t.contains(".private_segment_fixed_size: 0\n")
            && t.contains(".vgpr_spill_count: 0\n")
            && t.contains(".sgpr_spill_count: 0\n"),
        "{what}: no scratch"
    );
    assert!(
        t.contains(".amdhsa_system_sgpr_workgroup_id_x 1\n")
            && t.contains(".amdhsa_system_sgpr_workgroup_id_y 0\n"),
        "{what}: 1-D grid"
    );
}

/// Mnemonic families no entry of the module may contain.
const FORBIDDEN: [&str; 11] = [
    "scratch_",
    "flat_",
    "buffer_atomic_",
    "global_atomic_",
    "v_readlane",
    "v_writelane",
    "s_setpc",
    "s_swappc",
    "v_wmma",
    "v_cvt_",
    "v_fma_mix",
];

/// The last EXEC write before every `s_barrier` (program order) is the all-ones restore.
fn assert_barriers_run_with_full_exec(lines: &[&str], what: &str, want: usize) {
    let mut last_exec: Option<&str> = None;
    let mut barriers = 0;
    for &l in lines {
        if mnemonic(l) == "s_barrier" {
            assert_eq!(
                last_exec,
                Some("s_mov_b32 exec_lo, -1"),
                "{what}: barrier #{barriers} with a narrowed EXEC"
            );
            barriers += 1;
        } else if l.contains("exec_lo") || mnemonic(l) == "v_cmpx_eq_u32_e32" {
            last_exec = Some(l);
        }
    }
    assert_eq!(barriers, want, "{what}: barrier count");
}

/// Every vector load is drained with `vmcnt(0)` before the next barrier.
fn assert_x_loads_drain_before_barriers(lines: &[&str], what: &str) {
    for (i, _) in lines
        .iter()
        .enumerate()
        .filter(|(_, l)| mnemonic(l) == "global_load_b128" || mnemonic(l) == "global_load_b32")
    {
        let wait = lines
            .iter()
            .enumerate()
            .skip(i + 1)
            .find(|(_, w)| w.starts_with("s_waitcnt") && w.contains("vmcnt(0)"))
            .map(|(j, _)| j)
            .expect("vmcnt(0) after the X load");
        let next_barrier = lines
            .iter()
            .enumerate()
            .skip(i + 1)
            .find(|(_, b)| mnemonic(b) == "s_barrier")
            .map(|(j, _)| j)
            .unwrap();
        assert!(
            wait < next_barrier,
            "{what}: X load not drained before the barrier"
        );
    }
}

#[test]
fn names_modules_and_gating() {
    assert_eq!(qwen4_bf16_rows_wide::CAPS, CAPS);
    for cap in CAPS {
        assert_eq!(
            spec(cap).symbol(),
            format!("gemv_bf16_xf32_x4_rows_wide_r{cap}_pm_gfx1151")
        );
        assert_eq!(
            spec(cap).symbol_hc(),
            format!("gemv_bf16_xf32_k4_rows_wide_r{cap}_pm_gfx1151")
        );
    }
    assert_eq!(Spec::module(ARCH), MODULE);

    // The module holds exactly the four entries, ordinary r16, r32 then k4 r16, r32.
    let (emitted, text, proof) = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap();
    assert_eq!(proof.module, MODULE);
    assert_eq!((emitted.len(), proof.kernels.len()), (4, 4));
    let symbols: Vec<String> = CAPS
        .iter()
        .map(|&c| spec(c).symbol())
        .chain(CAPS.iter().map(|&c| spec(c).symbol_hc()))
        .collect();
    let mut at = 0;
    for symbol in &symbols {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}"
        );
        assert_eq!(
            text.matches(&format!("{symbol}:\n")).count(),
            1,
            "{symbol}: symbol label"
        );
        let here = text.find(&format!("{symbol}:\n")).unwrap();
        assert!(here >= at, "{symbol}: module order");
        at = here;
    }
    assert_eq!(text.matches(".amdhsa_kernel ").count(), 4);
    assert_eq!(
        text.matches(".amdgpu_metadata\n").count(),
        1,
        "one metadata document"
    );
    // Single entries are the module's kernels, byte for byte.
    for (e, symbol) in emitted.iter().zip(&symbols) {
        assert!(e.s_text.contains(&format!("{symbol}:\n")), "{symbol}");
    }
    assert_eq!(emitted[0].s_text, x4(16).s_text);
    assert_eq!(emitted[1].s_text, x4(32).s_text);
    assert_eq!(emitted[2].s_text, k4(16).s_text);
    assert_eq!(emitted[3].s_text, k4(32).s_text);

    // Exact-gfx1151 only, and only the two compiled capacities exist.
    for arch in [Arch::Gfx1100, Arch::Gfx1201] {
        for cap in CAPS {
            assert!(
                qwen4_bf16_rows_wide::emit(Spec { arch, cap }).is_err(),
                "{arch:?} x4 r{cap}"
            );
            assert!(
                qwen4_bf16_rows_wide::emit_hc(Spec { arch, cap }).is_err(),
                "{arch:?} k4 r{cap}"
            );
        }
        assert!(qwen4_bf16_rows_wide::emit_module(arch).is_err(), "{arch:?}");
    }
    for cap in [0u8, 8, 24, 48, 64] {
        assert!(
            qwen4_bf16_rows_wide::emit(spec(cap)).is_err(),
            "x4 cap {cap}"
        );
        assert!(
            qwen4_bf16_rows_wide::emit_hc(spec(cap)).is_err(),
            "k4 cap {cap}"
        );
    }
}

#[test]
fn emission_is_deterministic() {
    for cap in CAPS {
        for (name, a, b) in [("x4", x4(cap), x4(cap)), ("k4", k4(cap), k4(cap))] {
            assert_eq!(a.s_text, b.s_text, "{name} r{cap}");
            assert_eq!(
                a.proof.s_text_sha256, b.proof.s_text_sha256,
                "{name} r{cap}"
            );
            assert_eq!(
                serde_json::to_value(&a.proof).unwrap(),
                serde_json::to_value(&b.proof).unwrap(),
                "{name} r{cap}"
            );
        }
    }
    let (ea, ta, pa) = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap();
    let (eb, tb, pb) = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap();
    assert_eq!((ea.len(), &ta), (eb.len(), &tb));
    assert_eq!(
        serde_json::to_value(&pa).unwrap(),
        serde_json::to_value(&pb).unwrap(),
        "ModuleProof"
    );
}

/// Ordinary entries: the frozen 96-byte ABI of the narrow PM entry. HC entries: `W X Y M K scale hc_act R`, 48 bytes.
#[test]
fn kernarg_abi_and_launch_geometry() {
    assert_eq!(
        (
            qwen4_bf16_rows_wide::KERNARG_BYTES,
            qwen4_bf16_rows_wide::HC_KERNARG_BYTES
        ),
        (96, 48)
    );
    assert_eq!(
        (qwen4_bf16_rows_wide::M_BLOCK, qwen4_bf16_rows_wide::BLOCK),
        (8, 256)
    );
    assert_eq!(
        (
            qwen4_bf16_rows_wide::LDS_BYTES,
            qwen4_bf16_rows_wide::HC_LDS_BYTES
        ),
        (8192, 16384)
    );
    for cap in CAPS {
        check_abi(
            &x4(cap).s_text,
            &format!("x4 r{cap}"),
            X4_ARGS,
            96,
            5,
            4,
            8192,
        );
        check_abi(
            &k4(cap).s_text,
            &format!("k4 r{cap}"),
            K4_ARGS,
            48,
            2,
            1,
            16384,
        );
    }
}

/// Declared resource ceilings: 4 streams x cap accumulators past the working set, 128 / 192 VGPR budgets;
/// the ordinary entries use 64 SGPRs, the HC entries all 80.
#[test]
fn resources_stay_inside_the_budgets() {
    for (cap, budget) in [(16u8, 128u16), (32, 192)] {
        assert_eq!(spec(cap).vgpr_budget(), budget);
        let high_water = 40 + 4 * u16::from(cap);
        for (name, e, sgpr) in [("x4", x4(cap), 64u16), ("k4", k4(cap), 80)] {
            assert!(
                e.shape.next_free_vgpr >= high_water && e.shape.next_free_vgpr <= budget,
                "{name} r{cap}: vgpr {}",
                e.shape.next_free_vgpr
            );
            assert_eq!(e.shape.next_free_sgpr, sgpr, "{name} r{cap}: sgpr");
            assert_eq!(
                (e.proof.next_free_vgpr, e.proof.next_free_sgpr),
                (e.shape.next_free_vgpr, e.shape.next_free_sgpr)
            );
            assert!(e
                .s_text
                .contains(&format!(".vgpr_count: {}\n", e.shape.next_free_vgpr)));
            assert!(e.s_text.contains(&format!(
                ".amdhsa_next_free_vgpr {}\n",
                e.shape.next_free_vgpr
            )));
        }
    }
}

/// Exact instruction counts of the ordinary entries. `T` = row tiles of eight (`cap / 8`) and `C` = `cap`:
/// the 8-term vector chain runs once per row per K256 slot (4 unrolled slots) and the scalar K32 step once per
/// row; fold is 3 adds per row, plus a 5-level own+other tree per row.
#[test]
fn x4_instruction_counts() {
    for cap in CAPS {
        let (t, c) = (tiles(cap), usize::from(cap));
        let e = x4(cap);
        let lines = instructions(&e.s_text);
        let case = |what: &str, got: usize, want: usize| assert_eq!(got, want, "x4 r{cap}: {what}");
        case("v_fmac_f32_e32", count(&lines, "v_fmac_f32_e32"), 264 * t);
        case("ds_load_b32", count(&lines, "ds_load_b32"), 264 * t);
        case("ds_store_b32", count(&lines, "ds_store_b32"), 33 * t);
        case(
            "global_load_b128 (X, two per slot tile)",
            count(&lines, "global_load_b128"),
            8 * t,
        );
        case(
            "global_load_b32 (X, scalar step)",
            count(&lines, "global_load_b32"),
            t,
        );
        case(
            "buffer_load_b128 (weights, one per slot)",
            count(&lines, "buffer_load_b128"),
            4,
        );
        case(
            "buffer_load_u16 (weights, scalar step)",
            count(&lines, "buffer_load_u16"),
            1,
        );
        case(
            "s_barrier (two per tile)",
            count(&lines, "s_barrier"),
            10 * t,
        );
        case("shape.barriers", e.shape.barriers, 10 * t);
        case("ds_swizzle_b32", count(&lines, "ds_swizzle_b32"), c);
        case("ds_bpermute_b32", count(&lines, "ds_bpermute_b32"), 4 * c);
        case(
            "v_add_f32_e32 (fold + tree)",
            count(&lines, "v_add_f32_e32"),
            8 * c,
        );
        case("global_store_b32", count(&lines, "global_store_b32"), c);
        case("v_cmpx_eq_u32_e32", count(&lines, "v_cmpx_eq_u32_e32"), 1);
        case("s_endpgm", count(&lines, "s_endpgm"), 1);
        // Weight widening is bitwise: four `& 0xffff0000` and four `<< 16` per slot, one `<< 16` in the scalar step.
        case(
            "high-half widen",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_and_b32_e32" && l.contains("0xffff0000"))
                .count(),
            16,
        );
        case(
            "low-half widen",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_lshlrev_b32_e32" && operands(l)[1] == "16")
                .count(),
            17,
        );
        // One +0 initialization per accumulator, 4 streams per row.
        case(
            "accumulator zeroing",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_mov_b32_e32" && operands(l)[1] == "0")
                .count(),
            4 * c,
        );
        // EXEC is narrowed and restored symmetrically (weights of 5 slots + fill + consume of 5*T tile calls).
        case(
            "exec restores",
            count_exact(&lines, "s_mov_b32 exec_lo, -1"),
            5 + 10 * t,
        );
        case(
            "exec narrows",
            lines
                .iter()
                .filter(|l| l.starts_with("s_mov_b32 exec_lo, s"))
                .count(),
            5 + 10 * t,
        );
        case(
            "active-lane masks (-1 << n, not)",
            count(&lines, "s_not_b32"),
            5,
        );
        for forbidden in FORBIDDEN
            .iter()
            .chain(["v_mul_f", "v_fma_f", "v_div_", "v_exp_", "v_rcp_"].iter())
        {
            assert!(
                !lines.iter().any(|l| l.starts_with(forbidden)),
                "x4 r{cap}: forbidden {forbidden}*"
            );
        }
        assert!(
            !lines
                .iter()
                .any(|l| mnemonic(l).starts_with("s_") && mnemonic(l).ends_with("_f32")),
            "x4 r{cap}: no scalar FP"
        );
    }
}

/// Exact instruction counts of the HC entries. `S` = four-row stages (`cap / 4`), `C` = `cap`; one local K256
/// tile body (not unrolled), so the 8-term chain runs once per row; then `S` epilogue items, each the decoded
/// `hc_act` sequence (fold, 3 BF16 round trips, `exp`, the full division expansion).
#[test]
fn k4_instruction_counts() {
    for cap in CAPS {
        let (s, c) = (stages(cap), usize::from(cap));
        let e = k4(cap);
        let lines = instructions(&e.s_text);
        let case = |what: &str, got: usize, want: usize| assert_eq!(got, want, "k4 r{cap}: {what}");
        case(
            "v_fmac_f32_e32 (8C chain + 3 per item)",
            count(&lines, "v_fmac_f32_e32"),
            8 * c + 3 * s,
        );
        case("ds_load_b32", count(&lines, "ds_load_b32"), 32 * s);
        case(
            "ds_load_b128 (item partials)",
            count(&lines, "ds_load_b128"),
            s,
        );
        case(
            "ds_store_b32 (X stage + partials)",
            count(&lines, "ds_store_b32"),
            16 * s + c,
        );
        case("global_load_b128", count(&lines, "global_load_b128"), 4 * s);
        case("global_load_b32", count(&lines, "global_load_b32"), 0);
        case(
            "buffer_load_b128 (weights, one per tile)",
            count(&lines, "buffer_load_b128"),
            1,
        );
        case("buffer_load_u16", count(&lines, "buffer_load_u16"), 0);
        case(
            "s_barrier (2 per stage, partials, retire)",
            count(&lines, "s_barrier"),
            2 * s + 2,
        );
        case("shape.barriers", e.shape.barriers, 2 * s + 2);
        case("ds_swizzle_b32", count(&lines, "ds_swizzle_b32"), c);
        case("ds_bpermute_b32", count(&lines, "ds_bpermute_b32"), 4 * c);
        case(
            "v_add_f32_e32 (fold + tree + 1 + e per item)",
            count(&lines, "v_add_f32_e32"),
            8 * c + s,
        );
        case(
            "global_store_b32 (one per item)",
            count(&lines, "global_store_b32"),
            s,
        );
        case("v_cmpx_eq_u32_e32", count(&lines, "v_cmpx_eq_u32_e32"), 1);
        case("s_endpgm", count(&lines, "s_endpgm"), 1);
        case(
            "high-half widen",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_and_b32_e32" && l.contains("0xffff0000"))
                .count(),
            4,
        );
        case(
            "low-half widen",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_lshlrev_b32_e32" && operands(l)[1] == "16")
                .count(),
            4,
        );
        case(
            "accumulator zeroing",
            lines
                .iter()
                .filter(|l| mnemonic(l) == "v_mov_b32_e32" && operands(l)[1] == "0")
                .count(),
            4 * c,
        );
        case(
            "exec narrows",
            lines
                .iter()
                .filter(|l| l.starts_with("s_mov_b32 exec_lo, s"))
                .count(),
            3 * s + 1,
        );
        case(
            "exec restores",
            count_exact(&lines, "s_mov_b32 exec_lo, -1"),
            4 * s + 2,
        );
        case(
            "lane-0 exec (item stores)",
            count_exact(&lines, "s_mov_b32 exec_lo, 1"),
            s,
        );
        case(
            "active-lane masks (-1 << n, not)",
            count(&lines, "s_not_b32"),
            1,
        );
        // The decoded activation, per item: scalar FP, exp, and the full division expansion.
        for (m, per_item) in [
            ("s_add_f32", 4),
            ("s_mul_f32", 2),
            ("s_sub_f32", 1),
            ("s_rndne_f32", 1),
            ("s_fmamk_f32", 2),
            ("s_cmp_ngt_f32", 1),
            ("s_cmp_nlt_f32", 1),
            ("s_cvt_i32_f32", 1),
            ("v_exp_f32_e32", 1),
            ("v_ldexp_f32", 1),
            ("v_rcp_f32_e32", 1),
            ("v_div_scale_f32", 2),
            ("v_div_fmas_f32", 1),
            ("v_div_fixup_f32", 1),
            ("v_fma_f32", 2),
            ("v_mul_f32_e32", 1),
            ("v_cndmask_b32_e32", 2),
        ] {
            case(m, count(&lines, m), per_item * s);
        }
        // Three BF16 round trips (x, scale*x, result) per item.
        case("round-trip bfe", count(&lines, "s_bfe_u32"), 3 * s);
        case("round-trip addk", count(&lines, "s_addk_i32"), 3 * s);
        for forbidden in FORBIDDEN {
            assert!(
                !lines.iter().any(|l| l.starts_with(forbidden)),
                "k4 r{cap}: forbidden {forbidden}*"
            );
        }
    }
}

/// The wide entries double the arithmetic and X traffic when `cap` doubles and never touch the weights again:
/// weight loads are per K group, not per row tile.
#[test]
fn x4_weight_loads_are_independent_of_rows() {
    let (a, b) = (x4(16), x4(32));
    let (la, lb) = (instructions(&a.s_text), instructions(&b.s_text));
    for m in ["buffer_load_b128", "buffer_load_u16"] {
        assert_eq!(
            count(&la, m),
            count(&lb, m),
            "{m}: the weight loads do not scale with the row capacity"
        );
    }
    assert_eq!(
        2 * count(&la, "v_fmac_f32_e32"),
        count(&lb, "v_fmac_f32_e32")
    );
    assert_eq!(
        2 * count(&la, "global_load_b128"),
        count(&lb, "global_load_b128")
    );

    for cap in CAPS {
        let t = tiles(cap);
        let e = x4(cap);
        let lines = instructions(&e.s_text);
        let loads: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "buffer_load_b128")
            .map(|(i, _)| i)
            .collect();
        let u16_load = position(&lines, |l| mnemonic(l) == "buffer_load_u16");
        assert_eq!(loads.len(), 4);
        // Between consecutive weight loads every row tile of the previous slot is consumed once: T tiles x 8 rows
        // x 8 terms, two barriers per tile, two b128 X loads per tile, the weights untouched meanwhile.
        let mut bounds = loads.clone();
        bounds.push(u16_load);
        for w in bounds.windows(2) {
            let span = &lines[w[0] + 1..w[1]];
            assert_eq!(
                count(span, "v_fmac_f32_e32"),
                64 * t,
                "r{cap}: fmac between weight loads"
            );
            assert_eq!(
                count(span, "s_barrier"),
                2 * t,
                "r{cap}: barriers between weight loads"
            );
            assert_eq!(
                count(span, "global_load_b128"),
                2 * t,
                "r{cap}: X loads between weight loads"
            );
            assert_eq!(
                count(span, "ds_load_b32"),
                64 * t,
                "r{cap}: LDS reads between weight loads"
            );
            assert_eq!(
                count(span, "buffer_load_b128") + count(span, "buffer_load_u16"),
                0,
                "r{cap}: weights reloaded"
            );
        }
        // The scalar step's single weight load feeds the last T tiles: 8 rows x 1 term per tile.
        let tail = &lines[u16_load + 1..];
        assert_eq!(
            count(tail, "v_fmac_f32_e32"),
            8 * t,
            "r{cap}: scalar-step fmac"
        );
        assert_eq!(
            count(tail, "s_barrier"),
            2 * t,
            "r{cap}: scalar-step barriers"
        );
        // Each weight load retires (vmcnt(0)) before the widening that consumes it.
        for &load in loads.iter().chain([&u16_load]) {
            let wait = lines
                .iter()
                .enumerate()
                .skip(load + 1)
                .find(|(_, l)| l.starts_with("s_waitcnt") && l.contains("vmcnt(0)"))
                .map(|(i, _)| i)
                .expect("vmcnt(0) after weight load");
            let widen = lines
                .iter()
                .enumerate()
                .skip(load + 1)
                .find(|(_, l)| {
                    (mnemonic(l) == "v_and_b32_e32" && l.contains("0xffff0000"))
                        || (mnemonic(l) == "v_lshlrev_b32_e32" && operands(l)[1] == "16")
                })
                .map(|(i, _)| i)
                .unwrap();
            assert!(
                wait < widen,
                "r{cap}: widening before the weight load retires"
            );
        }
    }
}

/// HC: one weight load per local K256 tile, regardless of `cap`; all `4 cap/4 = cap` rows' chains consume it.
#[test]
fn k4_weight_load_is_independent_of_rows() {
    let (a, b) = (k4(16), k4(32));
    let (la, lb) = (instructions(&a.s_text), instructions(&b.s_text));
    assert_eq!(
        (
            count(&la, "buffer_load_b128"),
            count(&lb, "buffer_load_b128")
        ),
        (1, 1)
    );
    for cap in CAPS {
        let (s, c) = (stages(cap), usize::from(cap));
        let emitted = k4(cap);
        let lines = instructions(&emitted.s_text);
        let load = position(&lines, |l| mnemonic(l) == "buffer_load_b128");
        let first_swizzle = position(&lines, |l| mnemonic(l) == "ds_swizzle_b32");
        let span = &lines[load + 1..first_swizzle];
        assert_eq!(
            count(span, "v_fmac_f32_e32"),
            8 * c,
            "r{cap}: every row's 8-term chain between the weight load and the reduction"
        );
        assert_eq!(count(span, "global_load_b128"), 4 * s, "r{cap}: X traffic");
        assert_eq!(count(span, "s_barrier"), 2 * s, "r{cap}: stage barriers");
        assert_eq!(count(span, "ds_load_b32"), 32 * s, "r{cap}: LDS reads");
        assert_eq!(
            count(span, "buffer_load_b128"),
            0,
            "r{cap}: weights reloaded"
        );
        let wait = lines
            .iter()
            .enumerate()
            .skip(load + 1)
            .find(|(_, l)| l.starts_with("s_waitcnt") && l.contains("vmcnt(0)"))
            .map(|(i, _)| i)
            .expect("vmcnt(0) after the weight load");
        let widen = position(&lines, |l| {
            mnemonic(l) == "v_and_b32_e32" && l.contains("0xffff0000")
        });
        assert!(
            load < wait && wait < widen,
            "r{cap}: widening before the weight load retires"
        );
    }
}

/// The per-row source DAG of the ordinary entries: row `8t + r` of slot `s` accumulates `a[j & 3] =
/// fma(X_j, W_j, a[j & 3])` for `j = 0..7` ascending (`v_fmac_f32 acc, X, W`), rows' accumulators disjoint; the
/// scalar step is one chain `a0`. Then fold `a1 += a0`, `a3 += a2`, `a3 += a1` and a 5-level tree, lane 0 stores
/// stream 3 of each row.
#[test]
fn x4_per_row_dag_orientation_and_fold() {
    for cap in CAPS {
        let (t, c) = (tiles(cap), usize::from(cap));
        let e = x4(cap);
        let lines = instructions(&e.s_text);
        let fm = fmacs(&lines);
        assert_eq!(fm.len(), 264 * t);
        let [acc0, x0, w0] = fm[0].1;
        // Accumulators sit above the working set; X (first source) and weight (second source) slots are two
        // disjoint eight-register windows below them.
        assert!(acc0 >= 32 && x0 + 8 <= acc0 && w0 + 8 <= acc0 && (x0 + 8 <= w0 || w0 + 8 <= x0));
        for slot in 0..4 {
            for tile in 0..t {
                for r in 0..8 {
                    for j in 0..8 {
                        let at = slot * 64 * t + tile * 64 + r * 8 + j;
                        let row = (8 * tile + r) as u32;
                        assert_eq!(
                            fm[at].1,
                            [
                                acc0 + 4 * row + (j as u32 % 4),
                                x0 + j as u32,
                                w0 + j as u32
                            ],
                            "r{cap}: slot {slot} tile {tile} row {r} term {j}"
                        );
                    }
                }
            }
        }
        for tile in 0..t {
            for r in 0..8 {
                let row = (8 * tile + r) as u32;
                assert_eq!(
                    fm[256 * t + tile * 8 + r].1,
                    [acc0 + 4 * row, x0, w0],
                    "r{cap}: scalar step tile {tile} row {r}"
                );
            }
        }
        // All 4*cap accumulators are used, none outside.
        let mut dests: Vec<u32> = fm.iter().map(|f| f.1[0]).collect();
        dests.sort_unstable();
        dests.dedup();
        assert_eq!(
            dests,
            (acc0..acc0 + 4 * c as u32).collect::<Vec<_>>(),
            "r{cap}: accumulator set"
        );

        // Fold, in the extracted decode object's source order, row 0 first; no row is skipped.
        let last_fmac = fm.last().unwrap().0;
        let adds: Vec<[u32; 3]> = lines[last_fmac + 1..]
            .iter()
            .filter(|l| mnemonic(l) == "v_add_f32_e32")
            .take(3 * c)
            .map(|l| {
                let o = operands(l);
                [vreg(o[0]), vreg(o[1]), vreg(o[2])]
            })
            .collect();
        for r in 0..c as u32 {
            let a = acc0 + 4 * r;
            assert_eq!(
                adds[3 * r as usize..3 * r as usize + 3],
                [
                    [a + 1, a + 1, a],
                    [a + 3, a + 3, a + 2],
                    [a + 3, a + 3, a + 1]
                ],
                "r{cap}: fold of row {r}"
            );
        }
        // Stores: lane 0 only, stream 3 of each row in ascending row order, after the last barrier.
        let last_barrier = lines
            .iter()
            .rposition(|l| mnemonic(l) == "s_barrier")
            .unwrap();
        let cmpx = position(&lines, |l| mnemonic(l) == "v_cmpx_eq_u32_e32");
        assert_eq!(operands(lines[cmpx])[0], "0", "r{cap}: lane == 0 mask");
        let stores: Vec<(usize, u32)> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "global_store_b32")
            .map(|(i, l)| (i, vreg(operands(l)[1])))
            .collect();
        assert_eq!(stores.len(), c);
        assert!(
            last_barrier < cmpx && cmpx < stores[0].0,
            "r{cap}: lane-0 store mask after the last barrier"
        );
        assert_eq!(
            stores.iter().map(|s| s.1).collect::<Vec<_>>(),
            (0..c as u32).map(|r| acc0 + 4 * r + 3).collect::<Vec<_>>(),
            "r{cap}: stored streams"
        );
    }
}

/// The HC per-(row, quarter) chain is the ordinary chain on the quarter: row `r` consumes `a[j & 3] =
/// fma(X_j, W_j, a[j & 3])`, `j = 0..7`, one tile body, stages of four rows in ascending row order; fold and tree
/// are the ordinary ones; lane 0 writes each row's quarter sum to LDS `partial[mloc][r][q]`; the item epilogue folds
/// `(q0 + q1) + (q2 + q3)` in the decoded order and branches over the activation on `hc_act == 0`.
#[test]
fn k4_per_row_dag_partials_and_item_fold() {
    for cap in CAPS {
        let (s, c) = (stages(cap), usize::from(cap));
        let e = k4(cap);
        let lines = instructions(&e.s_text);
        let fm = fmacs(&lines);
        assert_eq!(fm.len(), 8 * c + 3 * s);
        let [acc0, x0, w0] = fm[0].1;
        assert!(acc0 >= 32 && x0 + 8 <= acc0 && w0 + 8 <= acc0 && (x0 + 8 <= w0 || w0 + 8 <= x0));
        for row in 0..c {
            for j in 0..8 {
                assert_eq!(
                    fm[row * 8 + j].1,
                    [
                        acc0 + 4 * row as u32 + (j as u32 % 4),
                        x0 + j as u32,
                        w0 + j as u32
                    ],
                    "r{cap}: row {row} term {j}"
                );
            }
        }
        // The DAG's accumulators are exactly 4*cap, disjoint per row.
        let mut dests: Vec<u32> = fm[..8 * c].iter().map(|f| f.1[0]).collect();
        dests.sort_unstable();
        dests.dedup();
        assert_eq!(
            dests,
            (acc0..acc0 + 4 * c as u32).collect::<Vec<_>>(),
            "r{cap}: accumulator set"
        );

        let last_chain = fm[8 * c - 1].0;
        let adds: Vec<[u32; 3]> = lines[last_chain + 1..]
            .iter()
            .filter(|l| mnemonic(l) == "v_add_f32_e32")
            .take(3 * c)
            .map(|l| {
                let o = operands(l);
                [vreg(o[0]), vreg(o[1]), vreg(o[2])]
            })
            .collect();
        for r in 0..c as u32 {
            let a = acc0 + 4 * r;
            assert_eq!(
                adds[3 * r as usize..3 * r as usize + 3],
                [
                    [a + 1, a + 1, a],
                    [a + 3, a + 3, a + 2],
                    [a + 3, a + 3, a + 1]
                ],
                "r{cap}: fold of row {r}"
            );
        }

        // Barrier layout: 2 per stage, then the partials barrier, then the final retire barrier.
        let barriers: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "s_barrier")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(barriers.len(), 2 * s + 2);
        let (partials_barrier, retire_barrier) = (barriers[2 * s], barriers[2 * s + 1]);
        assert!(
            last_chain < barriers[2 * s - 1],
            "r{cap}: the chains end before the last stage barrier"
        );
        // Partial stores: ds_store_b32 whose data is a row's stream 3, at LDS offset 16 r, lane 0 only, before the
        // partials barrier and after the last stage barrier.
        let cmpx = position(&lines, |l| mnemonic(l) == "v_cmpx_eq_u32_e32");
        assert_eq!(operands(lines[cmpx])[0], "0", "r{cap}: lane == 0 mask");
        let partial_stores: Vec<(usize, u32, u32)> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "ds_store_b32" && vreg(operands(l)[1]) >= acc0)
            .map(|(i, l)| {
                (
                    i,
                    vreg(operands(l)[1]),
                    l.split("offset:")
                        .nth(1)
                        .map_or(0, |o| o.trim().parse().unwrap()),
                )
            })
            .collect();
        assert_eq!(partial_stores.len(), c);
        for (r, st) in partial_stores.iter().enumerate() {
            assert_eq!(
                (st.1, st.2),
                (acc0 + 4 * r as u32 + 3, 16 * r as u32),
                "r{cap}: partial store of row {r}"
            );
            assert!(
                barriers[2 * s - 1] < cmpx && cmpx < st.0 && st.0 < partials_barrier,
                "r{cap}: partial store {r} placement"
            );
        }
        // Items: one ds_load_b128 of the four quarters each, after the partials barrier and before the retire barrier.
        let item_loads: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "ds_load_b128")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(item_loads.len(), s);
        assert!(
            item_loads
                .iter()
                .all(|&i| partials_barrier < i && i < retire_barrier),
            "r{cap}: items between the partials and retire barriers"
        );
        for &i in &item_loads {
            // Four v_readfirstlane (the quarters), consecutive SGPRs.
            let reads: Vec<u32> = lines[i + 1..]
                .iter()
                .filter(|l| mnemonic(l) == "v_readfirstlane_b32")
                .take(4)
                .map(|l| operands(l)[0].strip_prefix('s').unwrap().parse().unwrap())
                .collect();
            assert_eq!(
                reads,
                (reads[0]..reads[0] + 4).collect::<Vec<_>>(),
                "r{cap}: quarters read into consecutive SGPRs"
            );
            // Fold (q0 + q1) + (q2 + q3): the first three scalar adds after the reads.
            let first_add = lines[i + 1..]
                .iter()
                .position(|l| mnemonic(l) == "s_add_f32")
                .unwrap()
                + i
                + 1;
            let adds: Vec<usize> = (first_add..lines.len())
                .filter(|&j| mnemonic(lines[j]) == "s_add_f32")
                .take(3)
                .collect();
            let fold: Vec<Vec<u32>> = adds
                .iter()
                .map(|&j| {
                    operands(lines[j])
                        .iter()
                        .map(|o| o.strip_prefix('s').unwrap().parse().unwrap())
                        .collect()
                })
                .collect();
            let q = reads[0];
            assert_eq!(fold[0][1..], [q, q + 1], "r{cap}: q0 + q1");
            assert_eq!(fold[1][1..], [q + 2, q + 3], "r{cap}: q2 + q3");
            assert_eq!(
                fold[2][1..],
                [fold[0][0], fold[1][0]],
                "r{cap}: (q0+q1) + (q2+q3)"
            );
            // `hc_act == 0` leaves the raw fold: the compare and branch directly follow the final add.
            let cmp = adds[2]
                + 1
                + lines[adds[2] + 1..]
                    .iter()
                    .position(|l| mnemonic(l) == "s_cmp_eq_u32")
                    .unwrap();
            assert_eq!(operands(lines[cmp])[1], "0", "r{cap}: act compare");
            let branch = after(&lines, cmp);
            assert!(
                branch.starts_with("s_cbranch_scc1 .Lbf16hc") && branch.ends_with("_done"),
                "r{cap}: act branch"
            );
        }
        // Every item stores from lane 0 only: the last EXEC write before each global store is `exec_lo = 1`.
        let mut last_exec = "";
        let mut stores = 0;
        for &l in &lines {
            if mnemonic(l) == "global_store_b32" {
                assert_eq!(last_exec, "s_mov_b32 exec_lo, 1", "r{cap}: item store mask");
                stores += 1;
            } else if l.contains("exec_lo") || mnemonic(l) == "v_cmpx_eq_u32_e32" {
                last_exec = l;
            }
        }
        assert_eq!(stores, s);
    }
}

/// Row masking (ordinary): `R` outside `1..=cap` returns before any barrier or store; a tile wholly past `R` is
/// skipped by every wave; each row's consumption and its store are guarded by `R > row`.
#[test]
fn x4_row_masking_and_early_returns() {
    for cap in CAPS {
        let e = x4(cap);
        let lines = instructions(&e.s_text);
        let end = format!("s_cbranch_scc1 .Lbf16w{cap}_end");
        let first_barrier = position(&lines, |l| mnemonic(l) == "s_barrier");
        let gt = position(&lines, |l| {
            l.starts_with("s_cmp_gt_u32 ") && l.ends_with(&format!(", {cap}"))
        });
        let rr = operands(lines[gt])[0].to_owned(); // the R SGPR
        let eq0 = position(&lines, |l| l == format!("s_cmp_eq_u32 {rr}, 0"));
        assert!(
            gt < first_barrier && eq0 < first_barrier,
            "r{cap}: R guards precede every barrier"
        );
        assert_eq!(
            (after(&lines, gt), after(&lines, eq0)),
            (end.as_str(), end.as_str()),
            "r{cap}: R guards leave the kernel"
        );
        assert_eq!(count_exact(&lines, &format!("s_cmp_eq_u32 {rr}, 0")), 1);
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("s_cmp_gt_u32 ") && l.contains(&format!("{rr},")))
                .count(),
            1
        );
        assert!(position(&lines, |l| mnemonic(l) == "global_store_b32") > first_barrier);
        assert!(position(&lines, |l| mnemonic(l) == "global_load_b128") > gt);
        assert!(position(&lines, |l| mnemonic(l) == "buffer_load_b128") > gt);

        // Every row's store is guarded, after the last barrier, in ascending row order: the guard precedes its own
        // store and its branch follows it.
        let last_barrier = lines
            .iter()
            .rposition(|l| mnemonic(l) == "s_barrier")
            .unwrap();
        let stores: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| mnemonic(l) == "global_store_b32")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(stores.len(), usize::from(cap));
        let mut from = last_barrier;
        for r in 0..u32::from(cap) {
            let guard_line = format!("s_cmp_le_u32 {rr}, {r}");
            let exit = format!("s_cbranch_scc1 .Lbf16w{cap}_stored{}", r / 16);
            let at = lines
                .iter()
                .enumerate()
                .skip(from)
                .find(|(i, l)| **l == guard_line && after(&lines, *i) == exit)
                .map(|(i, _)| i)
                .unwrap_or_else(|| panic!("r{cap}: no store guard for row {r}"));
            let store = stores[r as usize];
            assert!(at < store, "r{cap}: row {r} guard precedes its store");
            from = store;
        }
        // Tiles past R are skipped: tile guards `R <= 8t` for t = 1..T in every slot and the scalar step.
        for tile in 1..tiles(cap) {
            let n = count_exact(&lines, &format!("s_cmp_le_u32 {rr}, {}", 8 * tile));
            assert!(n >= 5, "r{cap}: tile {tile} skip guards ({n})");
        }
        // Producers: wave + 8t >= R skips the fill, once per tile call.
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("s_cmp_ge_u32 ") && l.ends_with(&format!(", {rr}")))
                .count(),
            5 * tiles(cap),
            "r{cap}: producer guards"
        );
        // Consume guards: rows 1..7 of every tile.
        for row in 1..8u32 {
            assert!(
                count_exact(&lines, &format!("s_cmp_le_u32 {rr}, {row}")) >= 5,
                "r{cap}: consume guard of row {row}"
            );
        }
        // No guard exists for rows beyond the capacity.
        assert_eq!(
            count_exact(&lines, &format!("s_cmp_le_u32 {rr}, {}", 8 * tiles(cap))),
            0,
            "r{cap}: guard past cap"
        );
    }
}

/// Row masking (HC): the uniform early returns (`2*wg >= M`, `R` outside `1..=cap`, `K % 32 != 0`) precede every
/// barrier and store; stage, consume, partial and item guards follow `R`.
#[test]
fn k4_row_masking_and_early_returns() {
    for cap in CAPS {
        let (s, c) = (stages(cap), usize::from(cap));
        let emitted = k4(cap);
        let lines = instructions(&emitted.s_text);
        let end = format!("s_cbranch_scc1 .Lbf16hc{cap}_end");
        let first_barrier = position(&lines, |l| mnemonic(l) == "s_barrier");
        let gt = position(&lines, |l| {
            l.starts_with("s_cmp_gt_u32 ") && l.ends_with(&format!(", {cap}"))
        });
        let rr = operands(lines[gt])[0].to_owned();
        let eq0 = position(&lines, |l| l == format!("s_cmp_eq_u32 {rr}, 0"));
        let kmask = position(&lines, |l| {
            l.starts_with("s_and_b32 ") && l.ends_with(", 31")
        });
        let kcmp = position(&lines, |l| {
            l.starts_with("s_cmp_lg_u32 ") && l.ends_with(", 0")
        });
        let mguard = position(&lines, |l| {
            l.starts_with("s_cmp_ge_u32 ") && !l.ends_with(&format!(", {rr}"))
        });
        for at in [gt, eq0, kcmp, mguard] {
            assert!(
                at < first_barrier,
                "r{cap}: early return {at} before the first barrier"
            );
            assert_eq!(
                after(&lines, at),
                end,
                "r{cap}: early return leaves the kernel"
            );
        }
        assert!(kmask < kcmp, "r{cap}: K % 32 test");
        assert!(position(&lines, |l| mnemonic(l) == "global_store_b32") > first_barrier);
        assert!(
            position(&lines, |l| mnemonic(l) == "buffer_load_b128")
                > gt.max(eq0).max(kcmp).max(mguard)
        );
        assert!(
            position(&lines, |l| mnemonic(l) == "global_load_b128")
                > gt.max(eq0).max(kcmp).max(mguard)
        );
        assert_eq!(count_exact(&lines, &format!("s_cmp_eq_u32 {rr}, 0")), 1);
        // `R <= n` guards, rows 1..cap: the consume chain and the partial store of every row, plus the stage guard
        // when `n` starts a stage; none for row 0 or for rows past the capacity.
        assert_eq!(count_exact(&lines, &format!("s_cmp_le_u32 {rr}, 0")), 0);
        for n in 1..c {
            let want = if n % 4 == 0 { 3 } else { 2 };
            assert_eq!(
                count_exact(&lines, &format!("s_cmp_le_u32 {rr}, {n}")),
                want,
                "r{cap}: guards of row {n}"
            );
        }
        assert_eq!(
            count_exact(&lines, &format!("s_cmp_le_u32 {rr}, {c}")),
            0,
            "r{cap}: guard past cap"
        );
        // Producers (two vectors per wave per stage) and items skip on `row >= R`.
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("s_cmp_ge_u32 ") && l.ends_with(&format!(", {rr}")))
                .count(),
            3 * s,
            "r{cap}: producer and item guards"
        );
    }
}

/// Barrier and EXEC discipline: the full mask is restored before every barrier (the last EXEC write before each
/// `s_barrier` in program order is the all-ones restore), X loads are drained before barriers, and the barrier
/// proofs list every barrier.
#[test]
fn barriers_run_with_full_exec() {
    for cap in CAPS {
        let e = x4(cap);
        let lines = instructions(&e.s_text);
        assert_barriers_run_with_full_exec(&lines, &format!("x4 r{cap}"), 10 * tiles(cap));
        assert_x_loads_drain_before_barriers(&lines, &format!("x4 r{cap}"));
        assert_eq!(
            e.proof.barriers.len(),
            10 * tiles(cap),
            "x4 r{cap}: barrier proofs"
        );
        assert!(
            e.proof.barriers.iter().all(|b| !b.transitions.is_empty()),
            "x4 r{cap}: every barrier carries LDS transitions"
        );

        let e = k4(cap);
        let lines = instructions(&e.s_text);
        assert_barriers_run_with_full_exec(&lines, &format!("k4 r{cap}"), 2 * stages(cap) + 2);
        assert_x_loads_drain_before_barriers(&lines, &format!("k4 r{cap}"));
        assert_eq!(
            e.proof.barriers.len(),
            2 * stages(cap) + 2,
            "k4 r{cap}: barrier proofs"
        );
        assert!(
            e.proof.barriers.iter().all(|b| !b.transitions.is_empty()),
            "k4 r{cap}: every barrier carries LDS transitions"
        );
    }
}

fn offsets(lines: &[&str], m: &str, keep: impl Fn(&str) -> bool) -> Vec<u32> {
    lines
        .iter()
        .filter(|l| mnemonic(l) == m && keep(l))
        .map(|l| {
            l.split("offset:")
                .nth(1)
                .map_or(0, |o| o.trim().parse().unwrap())
        })
        .collect()
}

/// The ordinary LDS tile: eight producer rows x K256 f32. Producers store `128 j` apart, consumers read
/// `1024 r + 128 j` for `r, j < 8`, so every access stays under the 8192-byte slot.
#[test]
fn x4_lds_offsets_stay_inside_the_tile() {
    for cap in CAPS {
        let emitted = x4(cap);
        let lines = instructions(&emitted.s_text);
        let (stores, loads) = (
            offsets(&lines, "ds_store_b32", |_| true),
            offsets(&lines, "ds_load_b32", |_| true),
        );
        let mut s = stores.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(
            s,
            (0..8).map(|j| 128 * j).collect::<Vec<_>>(),
            "r{cap}: producer word offsets"
        );
        let mut l = loads.clone();
        l.sort_unstable();
        l.dedup();
        assert_eq!(
            l,
            (0..8)
                .flat_map(|r| (0..8).map(move |j| 1024 * r + 128 * j))
                .collect::<Vec<_>>(),
            "r{cap}: consumer offsets"
        );
        // Highest byte touched: max offset + 31 lanes * 4 B + 4, producer base (wave * 1024) included.
        assert!(loads.iter().max().unwrap() + 31 * 4 + 4 <= 8192);
        assert!(7 * 1024 + stores.iter().max().unwrap() + 31 * 4 + 4 <= 8192);
    }
}

/// The HC LDS: 4 rows x 4 quarters x K256 f32 per stage (16 KiB), then the partial slots. Producers store at
/// `8192 h + 128 j` (two vectors per wave), consumers read `4096 rl + 128 j` from their quarter's base, the
/// partials land at `16 r` inside the first `2 cap * 16` bytes.
#[test]
fn k4_lds_offsets_stay_inside_the_tile() {
    for cap in CAPS {
        let c = u32::from(cap);
        let emitted = k4(cap);
        let lines = instructions(&emitted.s_text);
        let first_acc = fmacs(&lines)[0].1[0];
        let is_partial = |l: &str| vreg(operands(l)[1]) >= first_acc;
        let producer = offsets(&lines, "ds_store_b32", |l| !is_partial(l));
        let mut p = producer.clone();
        p.sort_unstable();
        p.dedup();
        assert_eq!(
            p,
            [0u32, 8192]
                .iter()
                .flat_map(|h| (0..8).map(move |j| h + 128 * j))
                .collect::<Vec<_>>(),
            "r{cap}: producer word offsets"
        );
        let consumer = offsets(&lines, "ds_load_b32", |_| true);
        let mut l = consumer.clone();
        l.sort_unstable();
        l.dedup();
        assert_eq!(
            l,
            (0..4)
                .flat_map(|rl| (0..8).map(move |j| 4096 * rl + 128 * j))
                .collect::<Vec<_>>(),
            "r{cap}: consumer offsets"
        );
        let partials = offsets(&lines, "ds_store_b32", is_partial);
        assert_eq!(
            partials,
            (0..c).map(|r| 16 * r).collect::<Vec<_>>(),
            "r{cap}: partial slots"
        );
        // Highest byte: producer wave base 7*1024 + 8192 + 7*128 + 31*4 + 4; consumer quarter base 3*1024 + 3*4096 + ...
        assert!(7 * 1024 + producer.iter().max().unwrap() + 31 * 4 + 4 <= 16384);
        assert!(3 * 1024 + consumer.iter().max().unwrap() + 31 * 4 + 4 <= 16384);
        // Partials (mloc*cap + r)*16 + 4q plus the item loads' 16-byte quarters stay in the tile.
        assert!((2 * c) * 16 <= 16384);
    }
}

/// The committed shape contracts describe the fresh emission: symbol, resource ceilings, forbidden families,
/// every counted mnemonic.
#[test]
fn committed_contracts_match_fresh_emission() {
    for (tag, e, lds) in [
        ("r16", x4(16), 8192u64),
        ("r32", x4(32), 8192),
        ("k4_r16", k4(16), 16384),
        ("k4_r32", k4(32), 16384),
    ] {
        let path = format!(
            "{}/kernels/qwen4_bf16_rows_wide.gfx1151.{tag}.contract.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let contract: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")))
                .unwrap();
        let cap: u8 = tag
            .trim_start_matches("k4_r")
            .trim_start_matches('r')
            .parse()
            .unwrap();
        let symbol = if tag.starts_with("k4") {
            spec(cap).symbol_hc()
        } else {
            spec(cap).symbol()
        };
        let lines = instructions(&e.s_text);
        assert_eq!(contract["symbol"], symbol.as_str(), "{path}");
        for (name, want) in contract["counts"].as_object().unwrap() {
            assert_eq!(
                count(&lines, name) as u64,
                want.as_u64().unwrap(),
                "{path}: {name}"
            );
        }
        for pattern in contract["forbidden"].as_array().unwrap() {
            let prefix = pattern.as_str().unwrap().trim_end_matches('*');
            assert!(
                !lines.iter().any(|l| mnemonic(l).starts_with(prefix)),
                "{path}: emission contains forbidden {prefix}*"
            );
        }
        assert!(
            u64::from(e.shape.next_free_vgpr) <= contract["vgpr_max"].as_u64().unwrap(),
            "{path}: vgpr_max"
        );
        assert!(
            u64::from(e.shape.next_free_sgpr) <= contract["sgpr_max"].as_u64().unwrap(),
            "{path}: sgpr_max"
        );
        assert_eq!(contract["group_segment_fixed_bytes"], lds, "{path}");
        assert_eq!(contract["launch_dynamic_lds_bytes"], 0, "{path}");
        for flag in [
            "require_wave32",
            "require_zero_spills",
            "require_zero_private",
        ] {
            assert_eq!(contract[flag], true, "{path}: {flag}");
        }
    }
}

/// The runtime embeds one certified module: its `.text` must be exactly what the builder emits today
/// (`peacemaker-isa emit --kernel qwen4_bf16_rows_wide --epi all --arch gfx1151 ... --bundle`).
#[test]
fn committed_bundle_equals_native_emission() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let path = format!("{root}/{MODULE}.hxaco");
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = qwen4_bf16_rows_wide::emit_module(ARCH).unwrap().1;
    let elf = peacemaker_isa::native::assemble(&text, ARCH).unwrap();
    assert!(
        text_section(&elf) == text_section(&committed),
        "{MODULE}: native emission differs from the committed bundle"
    );
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
