// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 fixed-eight-row BF16 tests; total R is exhaustively 1..=64.
//!
//! Numerical scopes are deliberately separate: ordinary and raw HC execute the PM
//! object against an independent transcription of PLAN §13's gfx1100 singleton
//! DAG. PM R=1 is an additional scheduling oracle, never the actual-object oracle.
//! Activated HC has no CPU byte claim: gfx1100 exp/rcp semantics are unqualified
//! in peacemaker-ir, unlike gfx1151/gfx1201. The activated test requires the exact
//! interpreter refusal at every R, not a host exp2 approximation or an error skip.
//! The six basic emitted opcodes (s_sub_u32/s_sub_co_u32, s_cmp_le_i32, s_bfe_u32,
//! v_cmp_gt_u32_e32, v_mul_f32_e64, v_cndmask_b32_e32) own decode rows, ALU
//! dispatch arms and semantics.tbl interpreter rows, so Emulator::new builds
//! both PM programs. The construction test pins that supported state; every R
//! then still owes mandatory hardware G1 for actual-singleton bytes.
//!
//! Actual singleton evidence uses runtime fixture files, not machine-local
//! include_bytes!: QWEN4_BF16_GFX1100_FIXTURE_DIR must contain bf16-scalar.co and
//! bf16-hc.co with the pins below (default: tests/fixtures/gfx1100_singletons/bf16).
//! Missing files, wrong hashes and lift errors are fatal evidence refusals. The
//! current actual objects contain unmodeled VOPD integer instructions, including
//! v_dual_mov_b32; the actual-object test records that precise missing CPU claim.
//! Neither a refusal assertion nor a PM/DAG comparison is singleton-byte proof.
//!
//! Mandatory gfx1100 hardware G1: actual loaded singleton versus every R=1..64,
//! ordinary scalar/vector tails and x4 masks, raw/activated HC, all quarter tails,
//! realistic and extreme scales, signed-zero, denormal, infinity and NaN domains.
//! Record compiler/descriptor/object identity and poison/input immutability. CPU
//! tests do not qualify exceptional inputs, intrinsic keys, hazards or throughput.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_ir::Program;
use peacemaker_isa::kernels::qwen4_bf16_rows_gfx1100;
use peacemaker_isa::{Arch, Emitted};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const ARCH: Arch = Arch::Gfx1100;
const IR_ARCH: peacemaker_ir::inst::Arch = peacemaker_ir::inst::Arch::Gfx1100;
const ORDINARY: &str = "gemv_bf16_xf32_rows_x4_t8_pm_gfx1100";
const HC: &str = "gemv_bf16_xf32_k4_rows_t8_pm_gfx1100";
const KARG: u64 = 0x1_0000;
const X: u64 = 0x1_0000_0000;
const GUARD: usize = 64;
const POISON: u32 = 0x7fc0_a5a5;
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
const HC_ARGS: &[(u32, u32)] = &[
    (0, 8),
    (8, 8),
    (16, 8),
    (24, 4),
    (28, 4),
    (32, 4),
    (36, 4),
    (40, 4),
];

// PLAN §13.2: extracted target ELFs, not rebuilt PM stand-ins. Cache stems are
// under ~/.hipfire_kernels/gfx1100; each original radiowave receipt owns compiler
// provenance. These source pins are not proof that a GPU process loaded them.
const FIXTURES: [(&str, &str, &str, &str, &str); 2] = [
    (
        "bf16-scalar.co",
        "gemv_bf16_xf32",
        "b217ac560c63db3027eda1d83711c8d6399cb23df692ffd6deb0637acbd7bd64",
        "gemv_bf16_xf32.1e42614df293a4a4",
        "80ca9c63f6d74d061213fa4013f6a1da6f87c024ba60d145113f00d6c17bef42",
    ),
    (
        "bf16-hc.co",
        "gemv_bf16_xf32_k4",
        "3b7b89d5f8ff79609a04c949cd9cb7deec3e8d7ef758fce492cc303a3279c9bb",
        "qwen4_gemv_bf16_xf32.7eaf30c5c824f70c",
        "995b0bb6355dedc7edca0660c95cf9a39ede8441d2d31855f39f5a7614faed25",
    ),
];

fn emit(hc: bool) -> Emitted {
    qwen4_bf16_rows_gfx1100::emit_rowtiles(hc).unwrap()
}

fn program(hc: bool) -> Program {
    let elf = peacemaker_isa::native::assemble(&emit(hc).s_text, ARCH).unwrap();
    peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program
}

fn instructions(text: &str) -> Vec<&str> {
    text[..text.find(".section .rodata").unwrap()]
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && !l.starts_with('.') && !l.starts_with(';') && !l.ends_with(':')
        })
        .collect()
}

fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap()
}
fn count(lines: &[&str], op: &str) -> usize {
    lines.iter().filter(|l| mnemonic(l) == op).count()
}
fn fields(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn put32(args: &mut [u8], at: usize, value: u32) {
    args[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn put64(args: &mut [u8], at: usize, value: u64) {
    args[at..at + 8].copy_from_slice(&value.to_le_bytes());
}
fn wbase(i: usize) -> u64 {
    0x2_0000_0000 + i as u64 * 0x100_0000
}
fn ybase(i: usize) -> u64 {
    0x3_0000_0000 + i as u64 * 0x100_0000
}
fn poison(words: usize) -> Vec<u8> {
    (0..words).flat_map(|_| POISON.to_le_bytes()).collect()
}

#[test]
fn frozen_symbols_abi_resources_geometry_and_determinism() {
    for hc in [false, true] {
        let e = emit(hc);
        let again = emit(hc);
        let t = &e.s_text;
        let symbol = if hc { HC } else { ORDINARY };
        let (args, bytes, block, lds, reads, writes) = if hc {
            (HC_ARGS, 48, 128, 128, 2, 1)
        } else {
            (X4_ARGS, 96, 32, 0, 5, 4)
        };
        assert_eq!(t.matches(&format!("{symbol}:\n")).count(), 1);
        assert_eq!(t.matches(".amdhsa_kernel ").count(), 1);
        for field in [
            format!(".amdhsa_kernel {symbol}\n"),
            format!(".amdhsa_kernarg_size {bytes}\n"),
            format!(".kernarg_segment_size: {bytes}\n"),
            format!(".max_flat_workgroup_size: {block}\n"),
            format!(".group_segment_fixed_size: {lds}\n"),
            format!(".amdhsa_group_segment_fixed_size {lds}\n"),
        ] {
            assert!(t.contains(&field), "{symbol}: {field}");
        }
        assert_eq!(
            fields(t, ".offset:"),
            args.iter().map(|a| a.0).collect::<Vec<_>>()
        );
        assert_eq!(
            fields(t, ".size:"),
            args.iter().map(|a| a.1).collect::<Vec<_>>()
        );
        assert_eq!(
            args.last().map(|&(o, n)| o + n),
            Some(if hc { 44 } else { 96 })
        );
        for field in [
            ".amdhsa_system_sgpr_workgroup_id_x 1\n",
            ".amdhsa_system_sgpr_workgroup_id_y 1\n",
            ".amdhsa_wavefront_size32 1\n",
            ".wavefront_size: 32\n",
            ".workgroup_processor_mode: 1\n",
            ".private_segment_fixed_size: 0\n",
            ".vgpr_spill_count: 0\n",
            ".sgpr_spill_count: 0\n",
            ".amdhsa_float_round_mode_32 0\n",
            ".amdhsa_float_round_mode_16_64 0\n",
            ".amdhsa_float_denorm_mode_32 3\n",
            ".amdhsa_float_denorm_mode_16_64 3\n",
            ".amdhsa_dx10_clamp 1\n",
            ".amdhsa_ieee_mode 1\n",
            ".amdhsa_fp16_overflow 0\n",
            ".amdhsa_workgroup_processor_mode 1\n",
        ] {
            assert!(t.contains(field), "{symbol}: {field}");
        }
        assert_eq!(t.matches(".actual_access: read_only\n").count(), reads);
        assert_eq!(t.matches(".actual_access: write_only\n").count(), writes);
        assert_eq!((e.shape.next_free_vgpr, e.shape.next_free_sgpr), (80, 80));
        assert_eq!((e.proof.next_free_vgpr, e.proof.next_free_sgpr), (80, 80));
        assert_eq!(e.shape.barriers as usize, usize::from(hc));
        assert_eq!(e.proof.barriers.len(), usize::from(hc));
        assert_eq!(t, &again.s_text);
        assert_eq!(e.proof.s_text_sha256, again.proof.s_text_sha256);
        // The object really targets gfx1100; do not assemble under a borrowed arch.
        assert_eq!(program(hc).target.arch, IR_ARCH);
    }
}

#[test]
fn fixed_eight_row_dag_and_no_scalar_float_hc_retagging() {
    for hc in [false, true] {
        let e = emit(hc);
        let lines = instructions(&e.s_text);
        let fmas: Vec<_> = lines
            .iter()
            .filter(|l| mnemonic(l) == "v_fmac_f32_e32")
            .copied()
            .collect();
        for row in 0..8 {
            for j in 0..8 {
                assert_eq!(
                    fmas[8 * row + j],
                    format!(
                        "v_fmac_f32_e32 v{}, v{}, v{}",
                        40 + 4 * row + (j & 3),
                        24 + j,
                        16 + j,
                    )
                );
            }
            assert_eq!(
                fmas[64 + row],
                format!("v_fmac_f32_e32 v{}, v24, v16", 40 + 4 * row)
            );
            let a = 40 + 4 * row;
            for fold in [
                format!("v_add_f32_e32 v{}, v{}, v{}", a + 1, a + 1, a),
                format!("v_add_f32_e32 v{}, v{}, v{}", a + 3, a + 3, a + 2),
                format!("v_add_f32_e32 v{}, v{}, v{}", a + 3, a + 3, a + 1),
            ] {
                assert_eq!(lines.iter().filter(|&&l| l == fold).count(), 1, "{fold}");
            }
        }
        for (op, n) in [
            ("buffer_load_b128", 1),
            ("buffer_load_u16", 1),
            ("global_load_b32", 72),
            ("global_load_b128", 0),
            ("ds_swizzle_b32", 8),
            ("ds_bpermute_b32", 32),
            ("global_store_b32", 8),
            ("ds_load_b32", 0),
        ] {
            assert_eq!(count(&lines, op), n, "hc={hc}: {op}");
        }
        for forbidden in [
            "scratch_",
            "flat_",
            "global_atomic",
            "buffer_atomic",
            "v_wmma",
            "v_mfma",
            "v_dot",
            "v_fma_mix",
            "v_cvt_pk_bf16",
            "s_add_f32",
            "s_mul_f32",
            "s_rndne_f32",
            "s_fmamk_f32",
            "s_cvt_i32_f32",
        ] {
            assert!(
                !lines.iter().any(|l| l.starts_with(forbidden)),
                "{forbidden}"
            );
        }
        assert!(lines.contains(&"s_lshl_b32 s41, s3, 3"));
        assert!(lines.contains(&"s_cmp_gt_u32 s31, 64"));
        assert!(lines.contains(&"s_sub_u32 s45, s31, s41"));
        assert!(lines.contains(&"s_min_u32 s40, s45, 8"));
        // COUNT (s40) keeps exactly ONE definition: the domain min above.
        assert_eq!(count(&lines, "s_min_u32"), 1);
        assert_eq!(
            lines
                .iter()
                .filter(|l| {
                    let m = mnemonic(l);
                    !(m.starts_with("s_cmp")
                        || m.starts_with("s_cbranch")
                        || m.starts_with("s_branch"))
                        && l.split_whitespace().nth(1) == Some("s40,")
                })
                .count(),
            1,
            "COUNT s40 must have exactly one definition",
        );
        assert_eq!(count(&lines, "s_barrier"), usize::from(hc));
        assert_eq!(count(&lines, "ds_store_b32"), if hc { 8 } else { 0 });
        assert_eq!(count(&lines, "ds_load_b128"), if hc { 8 } else { 0 });
        if hc {
            assert!(lines.contains(&"s_and_b32 s45, s30, 31"));
            assert!(lines.contains(&"s_lshr_b32 s42, s30, 2"));
            assert!(lines.contains(&"s_mul_i32 s57, s43, s42"));
            for row in 0..8 {
                let offset = if row == 0 {
                    String::new()
                } else {
                    format!(" offset:{}", 16 * row)
                };
                assert!(lines
                    .iter()
                    .any(|l| mnemonic(l) == "ds_store_b32" && l.ends_with(&offset)));
            }
            for (row, want) in [(5, "0x50"), (6, "0x60"), (7, "0x70")] {
                let expected = format!("v_mov_b32_e32 v9, {want}");
                assert!(lines.contains(&expected.as_str()), "row {row}: {want}");
            }
            assert!(!lines.contains(&"v_mov_b32_e32 v9, 80"));
            let barrier = lines
                .iter()
                .position(|l| mnemonic(l) == "s_barrier")
                .unwrap();
            let exec = lines[..barrier]
                .iter()
                .rev()
                .find(|l| l.starts_with("s_mov_b32 exec_lo,"))
                .unwrap();
            assert_eq!(*exec, "s_mov_b32 exec_lo, -1");
            assert!(lines[..barrier]
                .iter()
                .all(|l| mnemonic(l) != "ds_load_b128"));
            assert!(lines[barrier + 1..]
                .iter()
                .all(|l| mnemonic(l) != "ds_store_b32"));
            for (op, n) in [
                ("v_exp_f32_e32", 8),
                ("v_ldexp_f32", 8),
                ("v_rcp_f32_e32", 8),
                ("v_div_scale_f32", 16),
                ("v_div_fmas_f32", 8),
                ("v_div_fixup_f32", 8),
            ] {
                assert_eq!(count(&lines, op), n, "{op}: complete VALU activation");
            }
        }
    }
}

// Finite normal values and both zero encodings; varied rows, K terms, matrices
// and columns expose stride, chain-assignment and fold errors. No NaN proof.
fn sample(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    if index % 29 == 0 {
        return f32::from_bits(x & 0x8000_0000);
    }
    f32::from_bits((x & 0x807f_ffff) | ((119 + (x >> 25) % 12) << 23))
}

/// Independent source/object-derived DAG, not an emitter helper or an ordinary
/// mathematical dot product. The masked tail contributes no padded +0 FMAs.
fn dot(x: &[f32], weights: &[u16]) -> f32 {
    assert_eq!(x.len(), weights.len());
    let mut lane = [0.0f32; 32];
    for (l, result) in lane.iter_mut().enumerate() {
        let mut a = [0.0f32; 4];
        if x.len() % 8 == 0 {
            for group in (0..x.len()).step_by(256) {
                for j in 0..8 {
                    let k = group + 8 * l + j;
                    if k < x.len() {
                        a[j & 3] =
                            x[k].mul_add(f32::from_bits(u32::from(weights[k]) << 16), a[j & 3]);
                    }
                }
            }
        } else {
            for k in (l..x.len()).step_by(32) {
                a[0] = x[k].mul_add(f32::from_bits(u32::from(weights[k]) << 16), a[0]);
            }
        }
        *result = (a[3] + a[2]) + (a[1] + a[0]);
    }
    let before = lane;
    for l in 0..32 {
        lane[l] += before[l | 16];
    }
    for distance in [8, 4, 2, 1] {
        let before = lane;
        for l in 0..32 {
            lane[l] += before[if l + distance < 32 { l + distance } else { l }];
        }
    }
    lane[0]
}

fn hc_raw(x: &[f32], weights: &[u16]) -> u32 {
    assert_eq!(x.len() % 32, 0);
    let n = x.len() / 4;
    let p: [f32; 4] =
        std::array::from_fn(|q| dot(&x[q * n..(q + 1) * n], &weights[q * n..(q + 1) * n]));
    ((p[0] + p[1]) + (p[2] + p[3])).to_bits()
}

fn launch(hc: bool, gx: u32, gy: u32) -> Launch {
    Launch {
        kernarg_ptr: KARG,
        workgroup: [gx, gy, 0],
        block: [if hc { 128 } else { 32 }, 1, 1],
        dynamic_lds: 0,
    }
}

fn check_readonly(mem: &Memory, before: &[Vec<u8>]) {
    for (region, original) in mem.regions.iter().zip(before) {
        if !region.name.starts_with('y') {
            assert_eq!(&region.bytes, original, "{} mutated", region.name);
            assert_eq!(region.written.count(), 0, "{} written", region.name);
        }
    }
}

fn check_output(
    mem: &Memory,
    name: &str,
    original: &[u8],
    rows: usize,
    m: usize,
    expected: &[u32],
    tag: &str,
) {
    let region = mem.regions.iter().find(|r| r.name == name).unwrap();
    let end = GUARD + 4 * rows * m;
    assert_eq!(expected.len(), rows * m);
    assert_eq!(
        region.written.count(),
        4 * rows * m,
        "{tag}: {name} write count"
    );
    for (byte, &old) in original.iter().enumerate() {
        let live = (GUARD..end).contains(&byte);
        assert_eq!(
            region.written.contains(byte),
            live,
            "{tag}: {name} byte={byte} ownership"
        );
        if !live {
            assert_eq!(region.bytes[byte], old, "{tag}: {name} byte={byte} poison");
        }
    }
    for (word, (&want, got)) in expected
        .iter()
        .zip(region.bytes[GUARD..end].chunks_exact(4))
        .enumerate()
    {
        assert_eq!(
            u32::from_le_bytes(got.try_into().unwrap()),
            want,
            "{tag}: {name} row={} col={}",
            word / m,
            word % m
        );
    }
}

fn ordinary_case(emu: &Emulator<'_>, rows: usize, k: usize, ms: [usize; 4], salt: u32) -> Memory {
    let mut mem = Memory::default();
    let mut args = vec![0u8; 96];
    let x: Vec<_> = (0..rows * k).map(|i| sample(i, salt)).collect();
    // No mapped input tail: masked K lanes and inactive rows must not load.
    mem.map(
        "x".into(),
        X,
        x.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    put64(&mut args, 32, X);
    let mut expected: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());
    for i in 0..4 {
        put32(&mut args, 72 + 4 * i, ms[i] as u32);
        if ms[i] == 0 {
            continue;
        } // Empty matrices have null W/Y pointers.
        let w: Vec<u16> = (0..ms[i] * k)
            .map(|j| (sample(j, salt.wrapping_add(17 + i as u32 * 97)).to_bits() >> 16) as u16)
            .collect();
        mem.map(
            format!("w{i}"),
            wbase(i),
            w.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap();
        mem.map(
            format!("y{i}"),
            ybase(i),
            poison(2 * GUARD / 4 + (rows.div_ceil(8) * 8 + 8) * ms[i]),
        )
        .unwrap();
        put64(&mut args, 8 * i, wbase(i));
        put64(&mut args, 40 + 8 * i, ybase(i) + GUARD as u64);
        for row in 0..rows {
            for col in 0..ms[i] {
                expected[i]
                    .push(dot(&x[row * k..(row + 1) * k], &w[col * k..(col + 1) * k]).to_bits());
            }
        }
    }
    put32(&mut args, 88, k as u32);
    put32(&mut args, 92, rows as u32);
    mem.map("kernarg".into(), KARG, args).unwrap();
    let before: Vec<_> = mem.regions.iter().map(|r| r.bytes.clone()).collect();
    let tag = format!("ordinary R={rows} K={k} M={ms:?} seed={salt:#x}");
    for gy in 0..=rows.div_ceil(8) as u32 {
        for gx in 0..=ms.iter().sum::<usize>() as u32 {
            emu.run_workgroup(&launch(false, gx, gy), &mut mem)
                .unwrap_or_else(|err| panic!("{tag} wg=({gx},{gy}): {err}"));
        }
    }
    check_readonly(&mem, &before);
    for i in 0..4 {
        if ms[i] == 0 {
            continue;
        }
        let region = mem
            .regions
            .iter()
            .position(|r| r.name == format!("y{i}"))
            .unwrap();
        check_output(
            &mem,
            &format!("y{i}"),
            &before[region],
            rows,
            ms[i],
            &expected[i],
            &tag,
        );
    }
    mem
}

#[test]
fn ordinary_every_r_scalar_k_all_residues_and_divisible8_lane_tails() {
    let p = program(false);
    let emu = Emulator::new(&p, ORDINARY).unwrap();
    for rows in 1..=64 {
        for k in [
            1, 2, 3, 4, 5, 6, 7, 9, 17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513,
        ] {
            ordinary_case(&emu, rows, k, [1, 0, 0, 0], 0x53a9);
        }
        // All 32 legal K256 lane masks, both before and after a complete group.
        for tail in 1..=32 {
            for full in [0, 256] {
                ordinary_case(&emu, rows, full + 8 * tail, [0, 1, 0, 0], 0x7bd3);
            }
        }
        for k in [768, 1024, 1032] {
            ordinary_case(&emu, rows, k, [0, 0, 1, 0], 0xb517);
        }
    }
}

#[test]
fn ordinary_every_r_x4_differing_heights_and_all_sixteen_empty_live_masks() {
    let p = program(false);
    let emu = Emulator::new(&p, ORDINARY).unwrap();
    for rows in 1..=64 {
        for mask in 0..16 {
            let ms = std::array::from_fn(|i| {
                if mask & (1 << i) != 0 {
                    [1, 3, 7, 9][i]
                } else {
                    0
                }
            });
            for k in [33, 264] {
                ordinary_case(&emu, rows, k, ms, 0x1d35 + mask);
            }
        }
        ordinary_case(&emu, rows, 520, [2, 8, 3, 1], 0x2a79);
    }
}

fn hc_memory(
    rows: usize,
    k: usize,
    m: usize,
    salt: u32,
    act: u32,
    scale: f32,
    negative: bool,
) -> (Memory, Vec<u32>) {
    let mut mem = Memory::default();
    let mut args = vec![0u8; 48];
    let x: Vec<_> = (0..rows * k)
        .map(|i| if negative { -1.0 } else { sample(i, salt) })
        .collect();
    let w: Vec<u16> = (0..m * k)
        .map(|i| {
            (if negative {
                1.0f32
            } else {
                sample(i, salt.wrapping_add(17))
            })
            .to_bits()
            .wrapping_shr(16) as u16
        })
        .collect();
    mem.map(
        "x".into(),
        X,
        x.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map(
        "w".into(),
        wbase(0),
        w.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap();
    mem.map(
        "y".into(),
        ybase(0),
        poison(2 * GUARD / 4 + (rows.div_ceil(8) * 8 + 8) * m),
    )
    .unwrap();
    put64(&mut args, 0, wbase(0));
    put64(&mut args, 8, X);
    put64(&mut args, 16, ybase(0) + GUARD as u64);
    put32(&mut args, 24, m as u32);
    put32(&mut args, 28, k as u32);
    put32(&mut args, 32, scale.to_bits());
    put32(&mut args, 36, act);
    put32(&mut args, 40, rows as u32);
    mem.map("kernarg".into(), KARG, args).unwrap();
    let mut expected = Vec::with_capacity(rows * m);
    for row in 0..rows {
        for col in 0..m {
            expected.push(hc_raw(
                &x[row * k..(row + 1) * k],
                &w[col * k..(col + 1) * k],
            ));
        }
    }
    (mem, expected)
}

fn raw_hc_case(
    emu: &Emulator<'_>,
    rows: usize,
    k: usize,
    m: usize,
    salt: u32,
    scale: f32,
) -> Memory {
    let (mut mem, expected) = hc_memory(rows, k, m, salt, 0, scale, false);
    let before: Vec<_> = mem.regions.iter().map(|r| r.bytes.clone()).collect();
    let tag = format!("raw HC R={rows} K={k} M={m} scale={:#x}", scale.to_bits());
    for gy in 0..=rows.div_ceil(8) as u32 {
        for gx in 0..=m as u32 {
            emu.run_workgroup(&launch(true, gx, gy), &mut mem)
                .unwrap_or_else(|err| panic!("{tag} wg=({gx},{gy}): {err}"));
        }
    }
    check_readonly(&mem, &before);
    check_output(&mem, "y", &before[2], rows, m, &expected, &tag);
    mem
}

#[test]
fn hc_raw_every_r_all_quarter_lane_tails_and_ragged_m() {
    let p = program(true);
    let emu = Emulator::new(&p, HC).unwrap();
    for rows in 1..=64 {
        // K/4=8..256 and 264..512: all quarter lane masks and loop transitions.
        for lanes in 1..=32 {
            for full in [0, 1024] {
                raw_hc_case(&emu, rows, full + 32 * lanes, 1, 0x419d, 0.125);
            }
        }
        for m in [1, 2, 3, 7, 8, 9] {
            raw_hc_case(&emu, rows, 1056, m, 0xa783, -3.25);
        }
        // act=0 bypasses ALL BF16 rounding, scale and intrinsic arithmetic.
        for scale in [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc1_2345),
        ] {
            raw_hc_case(&emu, rows, 32, 1, 0x617b, scale);
        }
    }
}

/// Both PM twins construct: the six basic interpreter rows
/// (`s_sub_u32`, `s_sub_co_u32`, `s_cmp_le_i32`, `s_bfe_u32`,
/// `v_cmp_gt_u32_e32`, `v_cndmask_b32_e32`, `v_mul_f32_e64`) exist and the
/// matching dispatch arms execute them. The activated HC twin still has no
/// CPU byte claim: gfx1100 exp/rcp semantics are unqualified, and the
/// activated test below requires that exact interpreter refusal at every R.
/// Actual-singleton VOPD bytes remain hardware-owned (see the
/// pinned-object test); refusal assertions are not singleton-byte proof.
#[test]
fn pm_emulator_construction_reports_exact_missing_semantics() {
    for (hc, symbol) in [(false, ORDINARY), (true, HC)] {
        let p = program(hc);
        let kernel = p
            .kernels
            .iter()
            .find(|k| k.symbol.0 == symbol)
            .expect("PM symbol");
        let mut missing = BTreeSet::new();
        for id in &kernel.body.layout {
            let inst = kernel.body.insts.get(*id).unwrap();
            let row =
                peacemaker_ir::isa::lookup(IR_ARCH, inst.op, inst.form).expect("decoded opcode");
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                missing.insert(row.name.to_string());
            }
        }
        assert!(
            missing.is_empty(),
            "hc={hc}: unexpected missing interpreter rows: {missing:?}"
        );
        Emulator::new(&p, symbol)
            .unwrap_or_else(|err| panic!("hc={hc}: construction failed: {err}"));
    }
}

#[test]
fn hc_activated_every_r_precisely_refuses_unqualified_gfx1100_exp_bytes() {
    let p = program(true);
    let emu = Emulator::new(&p, HC).unwrap();
    for rows in 1..=64 {
        for act in [1, u32::MAX] {
            // n=-32*0.125=-4 makes rndne's input positive; this reaches exp
            // instead of the independently unqualified negative-to-zero case.
            let (mut mem, _) = hc_memory(rows, 32, 1, 0, act, 0.125, true);
            let before: Vec<_> = mem.regions.iter().map(|r| r.bytes.clone()).collect();
            for gy in 0..rows.div_ceil(8) as u32 {
                let err = emu.run_workgroup(&launch(true, 0, gy), &mut mem)
                    .expect_err("gfx1100 activation became executable: require actual intrinsic-key and singleton byte oracle, not this gap assertion");
                assert!(
                    err.contains(" v_exp_f32_e32: unqualified numerical architecture Gfx1100"),
                    "R={rows} act={act} gy={gy}: unexpected refusal: {err}"
                );
            }
            check_readonly(&mem, &before);
            let y = mem.regions.iter().find(|r| r.name == "y").unwrap();
            assert_eq!(y.bytes, before[2], "refusal must precede any output store");
            assert_eq!(y.written.count(), 0);
        }
    }
    eprintln!("ABSENT CPU CLAIM: activated HC bytes at every R=1..64; gfx1100 v_exp_f32/v_rcp_f32 observations, rndne signed-zero and scaled division domains are unqualified. Mandatory actual-singleton hardware G1; no host exp2/divide fallback.");
}

#[test]
fn invalid_domains_return_before_accessing_unmapped_tensors() {
    for hc in [false, true] {
        let p = program(hc);
        let emu = Emulator::new(&p, if hc { HC } else { ORDINARY }).unwrap();
        let mut cases = vec![
            (0, 32, 1),
            (65, 32, 1),
            (u32::MAX, 32, 1),
            (1, 0, 1),
            (64, u32::MAX, 1),
            (1, 32, 0),
        ];
        if hc {
            for k in 1..64 {
                if k % 32 != 0 {
                    cases.push((1, k, 1));
                }
            }
        }
        for (rows, k, m) in cases {
            let mut args = vec![0u8; if hc { 48 } else { 96 }];
            for (offset, value) in if hc {
                [(24, m), (28, k), (40, rows)]
            } else {
                [(72, m), (88, k), (92, rows)]
            } {
                put32(&mut args, offset, value);
            }
            let original = args.clone();
            let mut mem = Memory::default();
            mem.map("kernarg".into(), KARG, args).unwrap();
            for gy in [0, 1, 8, 9] {
                for gx in [0, 1] {
                    emu.run_workgroup(&launch(hc, gx, gy), &mut mem)
                        .unwrap_or_else(|err| panic!("hc={hc} R={rows} K={k} M={m}: {err}"));
                }
            }
            assert_eq!(mem.regions[0].bytes, original);
            assert_eq!(mem.regions[0].written.count(), 0);
        }
    }
}

#[test]
fn every_r_grid_has_unique_outputs_and_row_local_quarter_partial_owners() {
    for rows in 1usize..=64 {
        for mask in 0..16 {
            let ms: [usize; 4] = std::array::from_fn(|i| {
                if mask & (1 << i) == 0 {
                    0
                } else {
                    [1, 3, 7, 9][i]
                }
            });
            let mut owners: [Vec<u8>; 4] = std::array::from_fn(|i| vec![0; rows * ms[i]]);
            for gy in 0..rows.div_ceil(8) {
                let base = 8 * gy;
                let live = (rows - base).min(8);
                for gx in 0..ms.iter().sum::<usize>() {
                    let mut col = gx;
                    let matrix = (0..4)
                        .find(|&i| {
                            if col < ms[i] {
                                true
                            } else {
                                col -= ms[i];
                                false
                            }
                        })
                        .unwrap();
                    for row in base..base + live {
                        owners[matrix][row * ms[matrix] + col] += 1;
                    }
                }
                let mut partials = [0u8; 128];
                for quarter in 0..4 {
                    for row in 0..live {
                        let at = 16 * row + 4 * quarter;
                        for byte in &mut partials[at..at + 4] {
                            *byte += 1;
                        }
                    }
                }
                assert!(partials[..16 * live].iter().all(|&n| n == 1));
                assert!(partials[16 * live..].iter().all(|&n| n == 0));
            }
            assert!(
                owners.iter().flatten().all(|&n| n == 1),
                "R={rows} M={ms:?}"
            );
        }
        for k in [32, 96, 1024, 1056, 2048] {
            let mut terms = vec![0u8; k];
            for q in 0..4 {
                for term in &mut terms[q * (k / 4)..(q + 1) * (k / 4)] {
                    *term += 1;
                }
            }
            assert!(terms.iter().all(|&n| n == 1));
        }
    }
}

#[test]
fn pinned_actual_singleton_objects_report_exact_missing_cpu_semantics() {
    let dir = std::env::var_os("QWEN4_BF16_GFX1100_FIXTURE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gfx1100_singletons/bf16")
        });
    for (name, symbol, sha, cache, source_sha) in FIXTURES {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!(
            "ACTUAL-OBJECT CPU EVIDENCE REFUSED: {}: {err}; supply QWEN4_BF16_GFX1100_FIXTURE_DIR with pinned extracted objects; never substitute PM/DAG or another arch", path.display(),
        ));
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            sha,
            "{}: singleton pin",
            path.display()
        );
        // File offsets of the frozen AMDHSA KD RSRC1 words, provided by the
        // object inspection in PLAN §13. Pins above make these offsets stable.
        let (offset, expected) = if name == "bf16-scalar.co" {
            (0x6b0, 0xe0af_0003u32)
        } else {
            (0x90f0, 0xe0af_000du32)
        };
        let rsrc1 = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(rsrc1, expected, "{symbol}: pinned descriptor");
        assert_eq!((rsrc1 >> 12) & 15, 0, "round32 and round16_64 RN");
        assert_eq!((rsrc1 >> 16) & 15, 15, "all denormals preserved");
        assert_eq!((rsrc1 >> 21) & 1, 1, "DX10 clamp");
        assert_eq!((rsrc1 >> 23) & 1, 1, "IEEE");
        assert_eq!((rsrc1 >> 26) & 1, 0, "FP16 overflow");
        assert_eq!((rsrc1 >> 29) & 1, 1, "WGP");
        let p = peacemaker_lift::lift_object(&bytes, peacemaker_lift::Options { frontend: Frontend::Hipcc })
            .unwrap_or_else(|err| panic!("ACTUAL-OBJECT CPU EVIDENCE REFUSED: {name} lift: {err}; missing claim requires hardware G1")).program;
        assert_eq!(p.target.arch, IR_ARCH);
        let kernel = p
            .kernels
            .iter()
            .find(|k| k.symbol.0 == symbol)
            .expect("pinned singleton export");
        let mut absent = BTreeSet::new();
        let mut first = None;
        for id in &kernel.body.layout {
            let inst = kernel.body.insts.get(*id).unwrap();
            let row =
                peacemaker_ir::isa::lookup(IR_ARCH, inst.op, inst.form).expect("decoded opcode");
            if peacemaker_ir::emu::semantic_class(row.name).is_none() {
                absent.insert(row.name.to_string());
                if first.is_none() {
                    first = Some(row.name.to_string());
                }
            }
        }
        // The blocker set shrinks as the interpreter gains rows (the VOPD `v_dual_*` rows are modeled now);
        // the singleton stays CPU-unclaimed exactly while any row is absent and `Emulator::new` refuses it.
        assert!(!absent.is_empty(), "missing actual-object blocker changed: require complete singleton-byte comparisons rather than treating this gap test as proof: {absent:?}");
        let err = match Emulator::new(&p, symbol) {
            Ok(_) => panic!("actual singleton became executable: replace refusal with mandatory all-R byte comparisons"),
            Err(err) => err,
        };
        assert_eq!(err, format!("no semantics for {}", first.unwrap()));
        eprintln!("ABSENT ACTUAL-OBJECT CPU BYTE CLAIM: {symbol} object={name} sha256={sha} cache={cache} source_sha256={source_sha} missing={absent:?}; first={err}; every R=1..64 ordinary/raw/activated HC requires actual-singleton hardware G1, not PM C1 or DAG fallback");
    }
}

/// Separate R=1 launches use each full-window row's actual pointer. Neither
/// emitter helpers nor the independent DAG can substitute for this comparison.
fn compare_pm_c1(
    emu: &Emulator<'_>,
    hc: bool,
    tiled: &Memory,
    rows: usize,
    k: usize,
    ms: [usize; 4],
) {
    let mut singleton = Memory::default();
    for region in &tiled.regions {
        let bytes = if region.name.starts_with('y') {
            poison(region.bytes.len() / 4)
        } else {
            region.bytes.clone()
        };
        singleton
            .map(region.name.clone(), region.base, bytes)
            .unwrap();
    }
    let arg = singleton
        .regions
        .iter()
        .position(|r| r.name == "kernarg")
        .unwrap();
    for row in 0..rows {
        let args = &mut singleton.regions[arg].bytes;
        put64(args, if hc { 8 } else { 32 }, X + (4 * row * k) as u64);
        put32(args, if hc { 40 } else { 92 }, 1);
        if hc {
            put64(args, 16, ybase(0) + (GUARD + 4 * row * ms[0]) as u64);
        } else {
            for (i, &m) in ms.iter().enumerate() {
                if m != 0 {
                    put64(args, 40 + 8 * i, ybase(i) + (GUARD + 4 * row * m) as u64);
                }
            }
        }
        for gx in 0..if hc { ms[0] } else { ms.iter().sum() } {
            emu.run_workgroup(&launch(hc, gx as u32, 0), &mut singleton)
                .unwrap_or_else(|err| {
                    panic!("PM C1 hc={hc} R={rows} row={row} K={k} M={ms:?}: {err}")
                });
        }
    }
    for (tile, single) in tiled.regions.iter().zip(&singleton.regions) {
        if tile.name == "kernarg" {
            assert_eq!(single.written.count(), 0);
            continue;
        }
        assert_eq!(
            tile.bytes, single.bytes,
            "PM C1 hc={hc} R={rows} K={k}: {}",
            tile.name
        );
        assert_eq!(
            tile.written.count(),
            single.written.count(),
            "{} write extents",
            tile.name
        );
        for byte in 0..tile.bytes.len() {
            assert_eq!(
                tile.written.contains(byte),
                single.written.contains(byte),
                "{} byte={byte}",
                tile.name
            );
        }
    }
}

#[test]
fn ordinary_and_raw_hc_every_r_also_equal_independent_pm_c1_launches() {
    let ordinary = program(false);
    let hc = program(true);
    let ordinary_emu = Emulator::new(&ordinary, ORDINARY).unwrap();
    let hc_emu = Emulator::new(&hc, HC).unwrap();
    for rows in 1..=64 {
        for k in [33, 264] {
            let ms = [1, 2, 3, 9];
            let tiled = ordinary_case(&ordinary_emu, rows, k, ms, 0x91b7);
            compare_pm_c1(&ordinary_emu, false, &tiled, rows, k, ms);
        }
        let tiled = raw_hc_case(&hc_emu, rows, 1056, 3, 0xb41d, 0.125);
        compare_pm_c1(&hc_emu, true, &tiled, rows, 1056, [3, 0, 0, 0]);
    }
}

#[test]
fn hc_valu_partition_fold_scale_order_and_divide_carry_are_row_local() {
    let e = emit(true);
    let lines = instructions(&e.s_text);
    // hc_value moves the v16 quarter fold into v24 (XS) before each scalar
    // read, so every readfirstlane names v24, never the WS fold register.
    for exact in [
        "v_add_f32_e32 v16, v16, v17",
        "v_add_f32_e32 v18, v18, v19",
        "v_add_f32_e32 v16, v16, v18",
        "v_mul_f32_e64 v24, s44, s64",
    ] {
        assert_eq!(
            lines.iter().filter(|&&line| line == exact).count(),
            8,
            "{exact}"
        );
    }
    // Three scalar reads per row: rounding input, scale input, activation input.
    assert_eq!(
        lines
            .iter()
            .filter(|&&line| line == "v_readfirstlane_b32 s64, v24")
            .count(),
        24,
        "v24 readfirstlane"
    );
    let mut carries = 0;
    for (start, line) in lines.iter().enumerate() {
        if mnemonic(line) != "v_div_scale_f32" || !line.contains(", vcc_lo,") {
            continue;
        }
        carries += 1;
        let end = start
            + 1
            + lines[start + 1..]
                .iter()
                .position(|l| mnemonic(l) == "v_div_fmas_f32")
                .expect("same row consumes division carry");
        for between in &lines[start + 1..end] {
            assert!(
                !between.starts_with("v_cmp")
                    && !between.contains("vcc_lo")
                    && !between.contains("vcc_hi"),
                "division carry clobbered before div_fmas: {between}"
            );
            assert!(
                !between.starts_with("s_branch") && !between.starts_with("s_cbranch"),
                "another row must not interleave the divide carry lifetime: {between}"
            );
        }
    }
    assert_eq!(carries, 8);
}

#[test]
fn source_pair_provenance_remains_the_recorded_gfx1100_reference() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src");
    for (source, expected) in [
        ("gemv_bf16_xf32.hip", FIXTURES[0].4),
        ("qwen4_gemv_bf16_xf32.hip", FIXTURES[1].4),
    ] {
        let bytes = std::fs::read(root.join(source)).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected,
            "{source}: reference source changed; re-acquire/re-pin actual singleton before numerical qualification");
    }
}
