// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The gfx1151 K1-Cn multi-lane (2..8 decode lanes) slab decode GEMV module `mq_i4_moe_gemv_cn_pm_gfx1151`
//! (`mq_i4_moe_gemv_cn_{mq3i4,mq4sym}_{r256_gate_up,r128_down}`, one codec per symbol): symbols, K1's frozen 96-byte kernarg
//! ABI (`n_experts@88`, `reserved@92`), the unsigned `expert_ids[u] >= n_experts` exit (emitted order and, on the peacemaker CPU
//! interpreter, invalid ids `-1`, `i32::MIN`, `n`, `n + 1` against valid `n - 1`), the fixed LDS (8 slots x one column phase),
//! exact-arch gating, the code-end guard, the schedule census (every weight
//! load precedes the first barrier and none follows it, no weight load sits in a slot branch, the X staging and `vmcnt`
//! ladder of the two-phase gate), the per-slot instruction census of K1's numerical tree (dequantization once per group,
//! the partial/chain add per live slot, the fold and the bpermute reduction per live slot), the derived shape contract, the
//! committed bundle and contract files, and (feature `toolchain`) wait replay and M7.
use peacemaker_isa::emit::CODE_END_TAIL;
use peacemaker_isa::kernels::mq_i4_moe_gemv_cn::{self as cn, Codec, Kind, Spec};
use peacemaker_isa::{native, Arch};

const SYMBOLS: [&str; 4] = [
    "mq_i4_moe_gemv_cn_mq3i4_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq3i4_r128_down",
    "mq_i4_moe_gemv_cn_mq4sym_r256_gate_up",
    "mq_i4_moe_gemv_cn_mq4sym_r128_down",
];
const MODULE: &str = "mq_i4_moe_gemv_cn_pm_gfx1151";
/// K1's frozen ABI: `(offset, size, name)` in kernarg order.
const ARGS: [(u32, u32, &str); 17] = [
    (0, 8, "expert_weight_ptrs"),
    (8, 8, "expert_ids"),
    (16, 8, "expert_offsets"),
    (24, 8, "slot_ids"),
    (32, 8, "Xrot"),
    (40, 8, "Y0"),
    (48, 8, "Y1"),
    (56, 4, "M"),
    (60, 4, "K"),
    (64, 4, "U"),
    (68, 4, "C"),
    (72, 4, "x_row_div"),
    (76, 4, "x_src_rows"),
    (80, 4, "x_row_stride"),
    (84, 4, "y_slot_stride"),
    (88, 4, "n_experts"),
    (92, 4, "reserved"),
];
const S_CODE_END: [u8; 4] = 0xbf9f_0000u32.to_le_bytes();

fn field(text: &str, key: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|l| l.trim().strip_prefix(key))
        .map(|v| v.trim().parse().unwrap())
        .collect()
}
fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}
fn count(lines: &[&str], mnemonic_name: &str) -> usize {
    lines
        .iter()
        .filter(|l| mnemonic(l) == mnemonic_name)
        .count()
}
/// Operand `n` (0-based) of an instruction line, split on `", "`.
fn operand(line: &str, n: usize) -> String {
    line.trim()
        .split_once(' ')
        .map(|(_, rest)| rest.split(", ").nth(n).unwrap_or("").trim().to_owned())
        .unwrap_or_default()
}
fn vreg(token: &str) -> u32 {
    token
        .trim_start_matches('v')
        .parse()
        .unwrap_or_else(|_| panic!("not a VGPR: {token}"))
}
fn imm_offset(line: &str) -> u32 {
    line.split("offset:")
        .nth(1)
        .map_or(0, |v| v.split_whitespace().next().unwrap().parse().unwrap())
}
fn is_weight_load(line: &str) -> bool {
    matches!(
        mnemonic(line),
        "global_load_b64" | "global_load_b32" | "global_load_u16"
    )
}
fn arch_specs() -> Vec<Spec> {
    cn::module_specs()
}
fn gate(spec: Spec) -> bool {
    spec.kind == Kind::GateUp
}
fn mq4(spec: Spec) -> bool {
    spec.codec == Codec::Mq4Sym
}

/// The trimmed lines of the kernel body (after `.Lcn_<tag>_body:` up to `_exit:`).
fn kernel_lines(text: &str, spec: Spec) -> Vec<&str> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let at = |name: &str| {
        lines
            .iter()
            .position(|l| *l == format!(".Lcn_{}_{name}:", spec.tag()))
            .unwrap_or_else(|| panic!("label {name}"))
    };
    let (body, exit) = (at("body"), at("exit"));
    assert!(body < exit);
    lines[body + 1..exit].to_vec()
}
/// The lines of `body` strictly between the labels `from:` and `to:` (full label names without the colon).
fn between<'a>(body: &[&'a str], spec: Spec, from: &str, to: &str) -> Vec<&'a str> {
    let at = |name: &str| {
        body.iter()
            .position(|l| *l == format!(".Lcn_{}_{name}:", spec.tag()))
            .unwrap_or_else(|| panic!("label {name}"))
    };
    body[at(from) + 1..at(to)].to_vec()
}

#[test]
fn module_exports_are_the_four_frozen_symbols_in_order() {
    let specs = cn::module_specs();
    assert_eq!(specs.iter().map(Spec::symbol).collect::<Vec<_>>(), SYMBOLS);
    assert_eq!(cn::SYMBOLS, SYMBOLS);
    assert_eq!(
        specs.iter().map(Spec::tag).collect::<Vec<_>>(),
        [
            "mq3i4_r256_gate_up",
            "mq3i4_r128_down",
            "mq4sym_r256_gate_up",
            "mq4sym_r128_down"
        ]
    );
    assert_eq!(Spec::module(), MODULE);
    let (emitted, text, proof) = cn::emit_module().unwrap();
    assert_eq!(emitted.len(), SYMBOLS.len());
    assert_eq!((proof.module.as_str(), proof.arch), (MODULE, Arch::Gfx1151));
    assert_eq!(proof.kernels.len(), SYMBOLS.len());
    assert_eq!(text.matches(".amdhsa_kernel ").count(), SYMBOLS.len());
    for symbol in SYMBOLS {
        assert!(
            text.contains(&format!(".amdhsa_kernel {symbol}\n")),
            "{symbol}: missing from module source"
        );
        assert!(
            text.contains(&format!(".globl {symbol}\n")),
            "{symbol}: not global"
        );
        assert!(
            text.contains(&format!(".name: {symbol}\n")),
            "{symbol}: missing from module metadata"
        );
    }
    assert!(text.contains("amdgcn-amd-amdhsa--gfx1151"));
}

/// Emission is bit-deterministic, and every architecture but gfx1151 is refused (no implicit alias).
#[test]
fn emission_is_deterministic_and_refuses_other_arches() {
    for spec in arch_specs() {
        let (a, b) = (cn::emit(spec).unwrap(), cn::emit(spec).unwrap());
        assert_eq!(a.s_text, b.s_text, "{spec:?}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{spec:?}");
    }
    let (_, ta, pa) = cn::emit_module().unwrap();
    let (_, tb, pb) = cn::emit_module().unwrap();
    assert_eq!((ta, pa.s_text_sha256), (tb, pb.s_text_sha256));
    for arch in [Arch::Gfx1100, Arch::Gfx1201, Arch::Gfx942] {
        for codec in Codec::ALL {
            for kind in Kind::ALL {
                let spec = Spec { arch, codec, kind };
                assert!(
                    spec.validate().is_err() && cn::emit(spec).is_err(),
                    "{arch:?} {codec:?} {kind:?}"
                );
            }
        }
    }
}

/// Wave32, eight waves per workgroup, grid y = the expert, fixed LDS of 8 slots x one column phase (40,960 B gate = 1,280
/// F32 per slot, 20,480 B down = 640), no scratch/spills/private, and the register budget the schedule targets (<= 128 VGPRs).
#[test]
fn launch_resources_match_the_frozen_geometry() {
    assert_eq!(
        (
            cn::KERNARG_BYTES,
            cn::WORKGROUP_SIZE,
            cn::ROWS_PER_WORKGROUP,
            cn::MAX_SLOTS
        ),
        (96, 256, 8, 8)
    );
    for spec in arch_specs() {
        let lds = if gate(spec) { 40960u32 } else { 20480 };
        let e = cn::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert_eq!(spec.kind.lds_bytes(), lds);
        assert!(
            e.s_text
                .contains(&format!(".amdhsa_group_segment_fixed_size {lds}\n")),
            "{symbol}: descriptor LDS"
        );
        assert!(
            e.s_text
                .contains(&format!(".group_segment_fixed_size: {lds}\n")),
            "{symbol}: metadata LDS"
        );
        assert_eq!(
            field(&e.s_text, ".max_flat_workgroup_size:"),
            [256],
            "{symbol}: block 256"
        );
        assert!(
            e.s_text.contains(".amdhsa_wavefront_size32 1"),
            "{symbol}: wave32"
        );
        assert!(
            e.s_text.contains(".amdhsa_system_sgpr_workgroup_id_y 1"),
            "{symbol}: grid y is the expert"
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
        let vgprs = field(&e.s_text, ".amdhsa_next_free_vgpr");
        assert_eq!(vgprs.len(), 1, "{symbol}");
        assert!(vgprs[0] <= 128, "{symbol}: {} VGPRs", vgprs[0]);
        assert_eq!(u32::from(e.shape.next_free_vgpr), vgprs[0], "{symbol}");
        assert!(
            e.s_text.contains(&format!(".vgpr_count: {}\n", vgprs[0])),
            "{symbol}: metadata VGPR count"
        );
        assert!(
            !e.s_text
                .lines()
                .any(|l| mnemonic(l).starts_with("scratch_") || mnemonic(l).starts_with("flat_")),
            "{symbol}: scratch/flat"
        );
        assert_eq!(
            e.shape.barriers,
            if gate(spec) { 3 } else { 1 },
            "{symbol}: gate publish/retire/publish, down publish"
        );
    }
}

/// K1's 96-byte kernarg block: per-argument offsets/sizes/names, access, the launch bound, the first loads.
#[test]
fn kernarg_abi_matches_the_frozen_96_byte_layout() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let symbol = spec.symbol();
        assert!(
            e.s_text.contains(".kernarg_segment_size: 96\n"),
            "{symbol}: kernarg size"
        );
        assert!(
            e.s_text.contains(".amdhsa_kernarg_size 96\n"),
            "{symbol}: descriptor kernarg size"
        );
        assert_eq!(
            field(&e.s_text, ".offset:"),
            ARGS.iter().map(|a| a.0).collect::<Vec<_>>(),
            "{symbol}: kernarg offsets"
        );
        assert_eq!(
            field(&e.s_text, ".size:"),
            ARGS.iter().map(|a| a.1).collect::<Vec<_>>(),
            "{symbol}: kernarg sizes"
        );
        let names: Vec<String> = e
            .s_text
            .lines()
            .filter_map(|l| l.trim().strip_prefix(".name: "))
            .take(ARGS.len())
            .map(str::to_owned)
            .collect();
        assert_eq!(
            names,
            ARGS.iter().map(|a| a.2.to_owned()).collect::<Vec<_>>(),
            "{symbol}: kernarg names"
        );
        let (last_off, last_size, _) = *ARGS.last().unwrap();
        assert_eq!(
            last_off + last_size,
            cn::KERNARG_BYTES,
            "{symbol}: arguments cover the kernarg block"
        );
        for (name, access) in [
            ("expert_weight_ptrs", "read_only"),
            ("expert_ids", "read_only"),
            ("expert_offsets", "read_only"),
            ("slot_ids", "read_only"),
            ("Xrot", "read_only"),
            ("Y0", "write_only"),
            ("Y1", "write_only"),
        ] {
            assert!(e.s_text.contains(&format!("      - .actual_access: {access}\n        .address_space: global\n        .name: {name}\n")), "{symbol}: {name} access {access}");
        }
        // The kernarg is loaded by five scalar loads covering offsets 0..92 with no over-read (`reserved@92` is never read);
        // `n_experts` lands in its own SGPR (s100, the declared ceiling is 101 SGPRs, `sgpr_max` of the contract).
        for load in [
            "s_load_b256 s[8:15], s[0:1], null",
            "s_load_b256 s[16:23], s[0:1], 0x20",
            "s_load_b128 s[24:27], s[0:1], 0x40",
            "s_load_b64 s[28:29], s[0:1], 0x50",
            "s_load_b32 s100, s[0:1], 0x58",
        ] {
            assert!(e.s_text.contains(load), "{symbol}: kernarg load {load}");
        }
        assert!(
            !e.s_text.contains("s[0:1], 0x5c"),
            "{symbol}: reserved@92 is never read"
        );
        assert_eq!(
            field(&e.s_text, ".sgpr_count:"),
            [101],
            "{symbol}: sgpr count"
        );
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_n_experts"),
            "{symbol}: typed n_experts ownership"
        );
        assert!(
            e.proof
                .reg_plan
                .iter()
                .any(|r| r.name.as_ref() == "kernel_arguments_ptrs"),
            "{symbol}: typed kernarg ownership"
        );
    }
}

fn u16_at(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap()) as usize
}
fn u32_at(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize
}
fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}
/// The bytes of the named section.
fn section(elf: &[u8], name: &str) -> Vec<u8> {
    let (shoff, entsize, count, strndx) = (
        u64_at(elf, 40),
        u16_at(elf, 58),
        u16_at(elf, 60),
        u16_at(elf, 62),
    );
    let header = |i: usize| shoff + i * entsize;
    let names = u64_at(elf, header(strndx) + 24);
    for i in 0..count {
        let at = names + u32_at(elf, header(i));
        let end = at + elf[at..].iter().position(|&c| c == 0).unwrap();
        if &elf[at..end] == name.as_bytes() {
            let (offset, size) = (u64_at(elf, header(i) + 24), u64_at(elf, header(i) + 32));
            return elf[offset..offset + size].to_vec();
        }
    }
    panic!("no section {name}")
}
fn function_sizes(elf: &[u8]) -> Vec<usize> {
    section(elf, ".symtab")
        .chunks(24)
        .filter(|s| s[4] & 0xf == 2)
        .map(|s| u64_at(s, 16))
        .collect()
}

/// The module (and each entry) carries the gfx11 `s_code_end` guard exactly once at the end, the native writer lays it
/// out after the last kernel (128-byte alignment fill then 96 `s_code_end` dwords) without moving any kernel, symbol
/// size or descriptor.
#[test]
fn code_end_tail_is_present_once_and_changes_nothing_else() {
    let (emitted, text, _) = cn::emit_module().unwrap();
    for t in emitted
        .iter()
        .map(|e| e.s_text.as_str())
        .chain([text.as_str()])
    {
        assert!(t.ends_with(CODE_END_TAIL));
        assert_eq!(t.matches(".p2alignl").count(), 1);
        assert_eq!(t.matches(".fill").count(), 1);
    }
    let bare = text
        .strip_suffix(CODE_END_TAIL)
        .expect("module ends with the guard");
    let guarded = native::assemble(&text, Arch::Gfx1151).unwrap();
    let plain = native::assemble(bare, Arch::Gfx1151).unwrap();
    let (guarded_text, bare_text) = (section(&guarded, ".text"), section(&plain, ".text"));
    assert_eq!(
        guarded_text[..bare_text.len()],
        bare_text[..],
        "the guard only appends"
    );
    assert!(
        guarded_text[bare_text.len()..]
            .chunks(4)
            .all(|w| w == S_CODE_END),
        "alignment fill and guard are s_code_end"
    );
    assert!(
        guarded_text.len() >= bare_text.len() + 96 * 4 && guarded_text.len() % 128 == 0,
        "128-byte aligned code end plus 96 guard dwords"
    );
    assert!(
        guarded_text
            .chunks(4)
            .rev()
            .take(96)
            .all(|w| w == S_CODE_END),
        "the last 96 dwords are s_code_end"
    );
    assert_eq!(function_sizes(&guarded), function_sizes(&plain));
    assert_eq!(section(&guarded, ".rodata"), section(&plain, ".rodata"));
    assert_eq!(function_sizes(&guarded).len(), SYMBOLS.len());
}

/// Weight loads of the row: every one precedes the first `s_barrier` and none follows it; none sits inside a slot branch
/// (they come after the last prologue guard and before the first compute label, and the body between the first and the last
/// weight load holds no branch or label); the phase-0 X loads precede every weight load; the first explicit
/// `vmcnt(<weight loads>)` separates the last weight load from the first `ds_store_b128`; the post-barrier `vmcnt` ladder
/// is the exact per-group descending sequence for phase 0 (the loads of a group are the oldest not yet consumed) and the
/// single `vmcnt(0)` of the gate's second phase.
#[test]
fn every_weight_load_precedes_the_barrier_and_the_vmcnt_ladder_is_exact() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        let weights = if g { 20 } else { 10 };
        let x_phase = if g { 10 } else { 5 };
        let phases = if g { 2 } else { 1 };
        let idx = |pred: &dyn Fn(&str) -> bool| -> Vec<usize> {
            body.iter()
                .enumerate()
                .filter(|(_, l)| pred(l))
                .map(|(i, _)| i)
                .collect()
        };
        let weight_ix = idx(&is_weight_load);
        let x_ix = idx(&|l| mnemonic(l) == "global_load_b128");
        let barrier_ix = idx(&|l| mnemonic(l) == "s_barrier");
        let store_ix = idx(&|l| mnemonic(l) == "ds_store_b128");
        assert_eq!(weight_ix.len(), weights, "{ctx}: weight loads");
        assert_eq!(x_ix.len(), x_phase * phases, "{ctx}: X staging loads");
        assert_eq!(store_ix.len(), x_phase * phases, "{ctx}: X stores");
        assert_eq!(barrier_ix.len(), if g { 3 } else { 1 }, "{ctx}: barriers");
        assert!(
            x_ix[..x_phase]
                .iter()
                .all(|x| weight_ix.iter().all(|w| x < w)),
            "{ctx}: phase-0 X loads precede every weight load"
        );
        assert!(
            weight_ix.iter().all(|w| *w < barrier_ix[0]),
            "{ctx}: every weight load precedes the first barrier"
        );
        assert!(
            body[barrier_ix[0]..].iter().all(|l| !is_weight_load(l)),
            "{ctx}: no weight load after the first barrier"
        );
        // not inside a slot branch: the weight loads sit between the prologue and the first compute label with no branch or label between them
        let (first, last) = (weight_ix[0], *weight_ix.last().unwrap());
        assert!(
            body[first..=last]
                .iter()
                .all(|l| !l.starts_with("s_cbranch") && !l.starts_with(".L") && *l != "s_branch"),
            "{ctx}: weight loads are straight-line"
        );
        assert!(
            store_ix[..x_phase]
                .iter()
                .all(|s| *s < barrier_ix[0] && *s > last),
            "{ctx}: phase-0 X stores sit between the loads and the first barrier"
        );
        // the explicit first wait
        let first_wait = idx(&|l| *l == format!("s_waitcnt vmcnt({weights})"));
        assert_eq!(
            first_wait.len(),
            1,
            "{ctx}: the explicit wait for the X loads"
        );
        assert!(
            first_wait[0] > last && first_wait[0] < store_ix[0],
            "{ctx}: explicit wait placement"
        );
        // the waits after the first barrier, in order
        let waits: Vec<u32> = body[barrier_ix[0]..]
            .iter()
            .filter_map(|l| l.strip_prefix("s_waitcnt vmcnt("))
            .filter_map(|r| r.split(')').next())
            .map(|n| n.parse().unwrap())
            .collect();
        let expect: Vec<u32> = if g {
            // phase 0: groups 0..9 (W3) / 0..4 (MQ4 gate, two loads per group) consume the oldest loads; then vmcnt(0) for the phase-1 X loads
            let mut v: Vec<u32> = (10..20).rev().collect();
            v.push(0);
            v
        } else if q {
            (0..10).rev().collect()
        } else {
            vec![8, 6, 4, 2, 0]
        };
        assert_eq!(waits, expect, "{ctx}: vmcnt ladder");
        if g {
            // phase 1: the retire barrier sits right after the second set of X loads, `vmcnt(0)` after it, the stores after that
            assert!(
                x_ix[x_phase..]
                    .iter()
                    .all(|x| *x > barrier_ix[0] && *x < barrier_ix[1]),
                "{ctx}: phase-1 X loads between the publish and the retire barrier"
            );
            let zero_wait = idx(&|l| l == "s_waitcnt vmcnt(0)");
            assert_eq!(zero_wait.len(), 1, "{ctx}");
            assert!(
                zero_wait[0] > barrier_ix[1]
                    && store_ix[x_phase..]
                        .iter()
                        .all(|s| *s > zero_wait[0] && *s < barrier_ix[2]),
                "{ctx}: phase-1 stores after vmcnt(0), before the publish barrier"
            );
        }
        // one lane-0 store per live slot region; no other memory ops; one endpgm (the exit label's, outside the body)
        assert_eq!(
            count(&body, "global_store_b32"),
            8,
            "{ctx}: one Y store per slot region"
        );
        assert_eq!(count(&body, "v_cmpx_eq_u32_e32"), 1, "{ctx}");
        assert!(
            !body.iter().any(|l| mnemonic(l).starts_with("flat_")
                || mnemonic(l).starts_with("scratch_")
                || mnemonic(l).starts_with("global_atomic")),
            "{ctx}: no other memory ops"
        );
    }
}

/// X staging addresses (immediates and offset registers of the `global_load_b128` and LDS stores) and the weight-load
/// addresses (group stride, packet / header / payload immediates) follow the formats; every `ds_load_b128` offset follows
/// `slot*slot_bytes + 512*g_local` (W3 and MQ4 down) / `+ 1024*g_local` and `+16` (MQ4 gate) in group-major slot order.
#[test]
fn addresses_follow_the_formats() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        // X loads: per phase `j = 0..` immediates 512 j (gate j >= 8: second register, 512 (j - 8))
        let x: Vec<&&str> = body
            .iter()
            .filter(|l| mnemonic(l) == "global_load_b128")
            .collect();
        let phase_expect: Vec<u32> = if g {
            (0..8).map(|j| 512 * j).chain([0, 512]).collect()
        } else {
            (0..5).map(|j| 512 * j).collect()
        };
        for ph in x.chunks(phase_expect.len()) {
            assert_eq!(
                ph.iter().map(|l| imm_offset(l)).collect::<Vec<_>>(),
                phase_expect,
                "{ctx}: X load immediates"
            );
            if g {
                let regs: Vec<String> = ph.iter().map(|l| operand(l, 1)).collect();
                assert!(
                    regs[..8].iter().all(|r| *r == regs[0])
                        && regs[8] == regs[9]
                        && regs[8] != regs[0],
                    "{ctx}: two X offset registers (+4096 for j >= 8)"
                );
            }
        }
        let stores: Vec<u32> = body
            .iter()
            .filter(|l| mnemonic(l) == "ds_store_b128")
            .map(|l| imm_offset(l))
            .collect();
        let per_phase: Vec<u32> = (0..phase_expect.len() as u32).map(|j| 512 * j).collect();
        assert_eq!(
            stores,
            per_phase
                .iter()
                .chain(if g { per_phase.iter() } else { [].iter() })
                .copied()
                .collect::<Vec<_>>(),
            "{ctx}: LDS stores"
        );
        // weight loads
        let groups = spec.groups();
        if !q {
            let b64: Vec<u32> = body
                .iter()
                .filter(|l| mnemonic(l) == "global_load_b64")
                .map(|l| imm_offset(l))
                .collect();
            assert_eq!(
                b64,
                (0..groups).map(|k| 48 * k).collect::<Vec<_>>(),
                "{ctx}: W3 packet windows"
            );
            let scales: Vec<u32> = body
                .iter()
                .filter(|l| mnemonic(l) == "global_load_u16")
                .map(|l| imm_offset(l))
                .collect();
            assert_eq!(
                scales,
                if g {
                    vec![]
                } else {
                    (0..groups).map(|k| 2 * k).collect::<Vec<_>>()
                },
                "{ctx}: W3 scales"
            );
            if g {
                // the row's 40 scale bytes at +40*row: ten SGPRs
                assert!(
                    body.contains(&"s_load_b256 s[88:95], s[50:51], null")
                        && body.contains(&"s_load_b64 s[96:97], s[50:51], 0x20"),
                    "{ctx}: gate scale loads"
                );
            }
        } else {
            let loads: Vec<&&str> = body.iter().filter(|l| is_weight_load(l)).collect();
            let (hdr, pay): (Vec<u32>, Vec<u32>) = (
                loads.iter().step_by(2).map(|l| imm_offset(l)).collect(),
                loads
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .map(|l| imm_offset(l))
                    .collect(),
            );
            let (stride, pay_off) = if g { (136, 8) } else { (68, 4) };
            assert_eq!(
                hdr,
                (0..groups).map(|k| stride * k).collect::<Vec<_>>(),
                "{ctx}: MQ4 headers"
            );
            assert_eq!(
                pay,
                (0..groups)
                    .map(|k| stride * k + pay_off)
                    .collect::<Vec<_>>(),
                "{ctx}: MQ4 payloads"
            );
            let payload_mnemonic = if g {
                "global_load_b32"
            } else {
                "global_load_u16"
            };
            assert_eq!(
                body.iter()
                    .filter(|l| mnemonic(l) == payload_mnemonic)
                    .count(),
                if g {
                    2 * groups as usize
                } else {
                    groups as usize
                },
                "{ctx}: payload-class loads"
            );
        }
        // LDS X reads, in program order: group-major, slot ascending
        let slot_bytes = spec.kind.slot_bytes();
        let lds: Vec<u32> = body
            .iter()
            .filter(|l| mnemonic(l) == "ds_load_b128")
            .map(|l| imm_offset(l))
            .collect();
        let per_phase_groups = groups / spec.kind.phases();
        let mut expect = Vec::new();
        for k in 0..groups {
            let gl = k % per_phase_groups;
            for i in 0..8u32 {
                if q && g {
                    expect.extend([i * slot_bytes + 1024 * gl, i * slot_bytes + 1024 * gl + 16])
                } else {
                    expect.push(i * slot_bytes + 512 * gl)
                }
            }
        }
        assert_eq!(lds, expect, "{ctx}: LDS X reads");
    }
}

/// The ops of one slot region of a group (between slot gates) as `(mul, fmac, add)` plus the operand check of K1's DAG.
/// Returns the destination accumulator register of its chain add.
fn partial_region(region: &[&str], spec: Spec, ctx: &str) -> u32 {
    let (mul, fmac, add): (Vec<&&str>, Vec<&&str>, Vec<&&str>) = (
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_mul_f32_e32")
            .collect(),
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_fmac_f32_e32")
            .collect(),
        region
            .iter()
            .filter(|l| mnemonic(l) == "v_add_f32_e32")
            .collect(),
    );
    let nw = if mq4(spec) && gate(spec) { 8 } else { 4 };
    assert_eq!(
        (mul.len(), fmac.len(), add.len()),
        (1, nw - 1, 1),
        "{ctx}: per-slot value ops equal C1's per-group partial"
    );
    let dot = operand(mul[0], 0);
    // W3: p = w0*x0, fmac(w_j, x_j) j = 1..; MQ4: p = w1*x1, fmac(w0, x0), fmac(w_j, x_j) j = 2..
    let (w0, x0) = (
        vreg(&operand(if mq4(spec) { fmac[0] } else { mul[0] }, 1)),
        vreg(&operand(if mq4(spec) { fmac[0] } else { mul[0] }, 2)),
    );
    let first = usize::from(mq4(spec));
    assert_eq!(
        (vreg(&operand(mul[0], 1)), vreg(&operand(mul[0], 2))),
        (w0 + first as u32, x0 + first as u32),
        "{ctx}: first product"
    );
    let rest: Vec<&&str> = if mq4(spec) {
        fmac[1..].to_vec()
    } else {
        fmac.clone()
    };
    for (n, f) in rest.iter().enumerate() {
        let j = if mq4(spec) { n + 2 } else { n + 1 } as u32;
        assert_eq!(
            (operand(f, 0), vreg(&operand(f, 1)), vreg(&operand(f, 2))),
            (dot.clone(), w0 + j, x0 + j),
            "{ctx}: fmac term {j}"
        );
    }
    assert_eq!(
        operand(add[0], 0),
        operand(add[0], 1),
        "{ctx}: acc = acc + p (acc is src0)"
    );
    assert_eq!(operand(add[0], 2), dot, "{ctx}: the partial is src1");
    vreg(&operand(add[0], 0))
}

/// Per-group census: the dequantization appears ONCE per group (the same counts as C1's per-group unpack) and every live-slot
/// region carries exactly one partial (C1's per-group mul/fmac/chain-add counts) into `acc[slot][g & 3]` (down: chain 0), with
/// scalar slot gates between regions; no weight load or LDS write inside any group.
#[test]
fn per_group_census_matches_the_frozen_dag() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let (g, q) = (gate(spec), mq4(spec));
        let chains = spec.kind.chains() as u32;
        let nw = if q && g { 8 } else { 4 };
        // the accumulators: the first block of `8*chains` consecutive v_mov ..., 0 after the first weight load
        let first_load = body.iter().position(|l| is_weight_load(l)).unwrap();
        let acc: Vec<u32> = body[first_load..]
            .iter()
            .filter(|l| l.starts_with("v_mov_b32_e32 v") && l.ends_with(", 0"))
            .map(|l| vreg(&operand(l, 0)))
            .take(8 * chains as usize)
            .collect();
        assert_eq!(
            acc.len(),
            8 * chains as usize,
            "{ctx}: accumulators start at +0"
        );
        assert!(
            acc.windows(2).all(|w| w[1] == w[0] + 1),
            "{ctx}: consecutive accumulators"
        );
        let batches = if q && g { 2 } else { 1 };
        let bsz = 8 / batches;
        for k in 0..spec.groups() {
            let chain = if g { k & 3 } else { 0 };
            for batch in 0..batches {
                let (lw, ld) = (format!("g{k}_b{batch}_lw"), format!("g{k}_b{batch}_ld"));
                let section = between(&body, spec, &lw, &ld);
                // dequantization: before the explicit lgkmcnt(0), once per group (batch 0 only)
                let wait = section
                    .iter()
                    .position(|l| *l == "s_waitcnt lgkmcnt(0)")
                    .unwrap_or_else(|| panic!("{ctx} g{k}: explicit lgkmcnt(0)"));
                let (deq, compute) = section.split_at(wait);
                if batch == 0 {
                    if q {
                        for (m, n) in [
                            ("v_bfe_u32", nw),
                            ("v_cvt_f32_u32_e32", nw),
                            ("v_fma_f32", nw),
                            ("v_cvt_f32_f16_e64", 2),
                            ("v_lshrrev_b32_e32", 1),
                        ] {
                            assert_eq!(count(deq, m), n, "{ctx} g{k}: dequant {m}");
                        }
                    } else {
                        for (m, n) in [
                            ("v_lshrrev_b64", 1),
                            ("v_bfe_u32", 4),
                            ("v_lshl_add_u32", 4),
                            ("v_cvt_f32_i32_e32", 4),
                            ("v_cvt_f32_f16_e64", 1),
                            ("v_mul_f32_e32", 4),
                        ] {
                            assert_eq!(count(deq, m), n, "{ctx} g{k}: dequant {m}");
                        }
                    }
                } else {
                    assert!(deq.iter().all(|l| !l.starts_with("v_")), "{ctx} g{k} b{batch}: no dequantization in later batches (weights are dequantized once)");
                }
                assert!(
                    compute.iter().all(|l| !is_weight_load(l)
                        && mnemonic(l) != "ds_store_b128"
                        && mnemonic(l) != "global_load_b128"),
                    "{ctx} g{k}: no weight/LDS-write inside a slot region"
                );
                // slot regions: split on the scalar slot gates
                let regions: Vec<&[&str]> = compute[1..]
                    .split(|l| mnemonic(l) == "s_cbranch_scc0")
                    .collect();
                assert_eq!(
                    regions.len(),
                    bsz,
                    "{ctx} g{k} b{batch}: one region per slot of the batch"
                );
                for (r, region) in regions.iter().enumerate() {
                    let slot = (batch * bsz) as usize + r;
                    let dst = partial_region(region, spec, &format!("{ctx} g{k} slot{slot}"));
                    assert_eq!(
                        dst,
                        acc[0] + slot as u32 * chains + chain,
                        "{ctx} g{k} slot{slot}: chain add into acc[slot][{chain}]"
                    );
                }
            }
        }
    }
}

/// The epilogue per live slot region: gate fold `(a0+a1)+(a2+a3)` (acc as src0), the five snapshot shuffle steps
/// (`ds_bpermute` of a0, `exch = a0 + exch`, `a0 = cndmask(a0, exch, mask_k)`), one lane-0 store of a0; the shuffle tables
/// once; and nothing else on the value path (no packed math, dot, fma_mix, mad, VOPD, exp).
#[test]
fn epilogue_census_matches_the_frozen_dag() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        let g = gate(spec);
        let chains = spec.kind.chains() as u32;
        let first_load = body.iter().position(|l| is_weight_load(l)).unwrap();
        let acc0 = body[first_load..]
            .iter()
            .filter(|l| l.starts_with("v_mov_b32_e32 v") && l.ends_with(", 0"))
            .map(|l| vreg(&operand(l, 0)))
            .next()
            .unwrap();
        let epi = between(&body, spec, "epi", "red_done");
        assert_eq!(
            count(&epi, "v_cmp_gt_u32_e64"),
            5,
            "{ctx}: five shuffle masks"
        );
        let regions: Vec<&[&str]> = epi.split(|l| mnemonic(l) == "s_cbranch_scc0").collect();
        assert_eq!(regions.len(), 8, "{ctx}: one reduction region per slot");
        for (slot, region) in regions.iter().enumerate() {
            let a0 = acc0 + slot as u32 * chains;
            let adds: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "v_add_f32_e32")
                .collect();
            let fold = if g { 3 } else { 0 };
            assert_eq!(
                adds.len(),
                fold + 5,
                "{ctx} slot{slot}: fold + 5 shuffle adds"
            );
            if g {
                let (f1, f2, f3) = (adds[0], adds[1], adds[2]);
                assert_eq!(
                    (vreg(&operand(f1, 0)), operand(f1, 1), vreg(&operand(f1, 2))),
                    (a0, format!("v{a0}"), a0 + 1),
                    "{ctx} slot{slot}: a0 = a0 + a1"
                );
                assert_eq!(
                    (vreg(&operand(f2, 0)), operand(f2, 1), vreg(&operand(f2, 2))),
                    (a0 + 2, format!("v{}", a0 + 2), a0 + 3),
                    "{ctx} slot{slot}: a2 = a2 + a3"
                );
                assert_eq!(
                    (vreg(&operand(f3, 0)), operand(f3, 1), vreg(&operand(f3, 2))),
                    (a0, format!("v{a0}"), a0 + 2),
                    "{ctx} slot{slot}: a0 = a0 + a2"
                );
            }
            let perm: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "ds_bpermute_b32")
                .collect();
            let selects: Vec<&&str> = region
                .iter()
                .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 0) == format!("v{a0}"))
                .collect();
            assert_eq!(
                (perm.len(), selects.len()),
                (5, 5),
                "{ctx} slot{slot}: five shuffle steps"
            );
            let exch = operand(perm[0], 0);
            for k in 0..5 {
                assert_eq!(
                    (operand(perm[k], 0), operand(perm[k], 2)),
                    (exch.clone(), format!("v{a0}")),
                    "{ctx} slot{slot}: bpermute {k} reads a0"
                );
                let step = adds[fold + k];
                assert_eq!(
                    (operand(step, 0), operand(step, 1), operand(step, 2)),
                    (exch.clone(), format!("v{a0}"), exch.clone()),
                    "{ctx} slot{slot}: exch = a0 + exch ({k})"
                );
                assert_eq!(
                    (
                        operand(selects[k], 0),
                        operand(selects[k], 1),
                        operand(selects[k], 2)
                    ),
                    (format!("v{a0}"), format!("v{a0}"), exch.clone()),
                    "{ctx} slot{slot}: a0 = cndmask(a0, exch, mask) ({k})"
                );
            }
            // the five masks are used in order 16, 8, 4, 2, 1 (the same SGPRs for every slot)
            let masks: Vec<String> = selects.iter().map(|l| operand(l, 3)).collect();
            let expect: Vec<String> = (72..77).map(|m| format!("s{m}")).collect();
            assert_eq!(masks, expect, "{ctx} slot{slot}: shuffle masks");
        }
        // the shuffle tables: offsets 16, 8, 4, 2, 1 (mask lane < 32 - off, address (lane + off) * 4)
        let tables: Vec<&&str> = epi
            .iter()
            .filter(|l| mnemonic(l) == "v_cndmask_b32_e64" && operand(l, 1) == "0")
            .collect();
        assert_eq!(
            tables.iter().map(|l| operand(l, 2)).collect::<Vec<_>>(),
            ["16", "8", "4", "2", "1"],
            "{ctx}: shuffle offsets"
        );
        // the stores: lane 0 only, one per live slot region, the slot's final a0, into the selected Y pointer
        let stores = between(&body, spec, "red_done", "st_done");
        assert_eq!(count(&stores, "v_cmpx_eq_u32_e32"), 1, "{ctx}: lane 0 only");
        let store_lines: Vec<&&str> = stores
            .iter()
            .filter(|l| mnemonic(l) == "global_store_b32")
            .collect();
        assert_eq!(store_lines.len(), 8, "{ctx}: one store per slot region");
        for (slot, s) in store_lines.iter().enumerate() {
            assert_eq!(
                operand(s, 1),
                format!("v{}", acc0 + slot as u32 * chains),
                "{ctx} slot{slot}: stores a0 of its slot"
            );
            assert_eq!(operand(s, 2), "s[52:53]", "{ctx}: selected Y pointer");
        }
        assert!(
            !body.iter().any(|l| {
                let m = mnemonic(l);
                m.starts_with("v_pk_")
                    || m.starts_with("v_dot")
                    || m.starts_with("v_wmma")
                    || m.starts_with("v_swmmac")
                    || m.starts_with("v_fma_mix")
                    || m.starts_with("v_mad")
                    || m.starts_with("v_fmamk")
                    || m.starts_with("v_fmaak")
                    || m.starts_with("v_dual")
                    || m.starts_with("v_exp")
            }),
            "{ctx}: forbidden value-path instruction"
        );
    }
}

/// Unpack ties of one group of each codec: exact operand patterns of K1's dequantization.
#[test]
fn unpack_matches_k1() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let body = kernel_lines(&e.s_text, spec);
        let ctx = spec.tag();
        if mq4(spec) {
            let fma = body
                .iter()
                .find(|l| mnemonic(l) == "v_fma_f32" && operand(l, 0) == operand(l, 2))
                .unwrap_or_else(|| panic!("{ctx}: fma"));
            assert_eq!(
                operand(fma, 0),
                operand(fma, 2),
                "{ctx}: w = fma(d, q, z) in place"
            );
        } else {
            let at = body
                .iter()
                .position(|l| mnemonic(l) == "v_lshrrev_b64")
                .unwrap();
            let pkt = format!(
                "v{}",
                operand(body[at], 0)
                    .trim_start_matches("v[")
                    .split(':')
                    .next()
                    .unwrap()
            );
            for (i, bfe) in body[at + 1..]
                .iter()
                .filter(|l| mnemonic(l) == "v_bfe_u32")
                .take(4)
                .enumerate()
            {
                assert_eq!(
                    (operand(bfe, 1), operand(bfe, 2), operand(bfe, 3)),
                    (pkt.clone(), (3 * i).to_string(), "3".into()),
                    "{ctx}: W3 code {i}: bfe(pkt, 3i, 3)"
                );
            }
            let lsa = body
                .iter()
                .find(|l| mnemonic(l) == "v_lshl_add_u32")
                .unwrap();
            assert_eq!(
                (operand(lsa, 2), operand(lsa, 3)),
                ("1".into(), "-7".into()),
                "{ctx}: W3 grid: (c << 1) + -7"
            );
        }
    }
}

/// The shape contract derives deterministically from emission, covers all four entries, forbids scratch/flat/WMMA/mix/dot/pk
/// and the gfx12 wait family (but not LDS), pins the register ceilings to the plan and the fixed LDS.
#[test]
fn shape_contract_is_derived_from_the_emission() {
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let c = cn::contract(spec, &e);
        assert_eq!(c, cn::contract(spec, &cn::emit(spec).unwrap()));
        assert_eq!(c["symbol"].as_str(), Some(spec.symbol().as_str()));
        assert_eq!(
            (c["vgpr_max"].as_u64(), c["sgpr_max"].as_u64()),
            (
                Some(u64::from(e.shape.next_free_vgpr)),
                Some(u64::from(e.shape.next_free_sgpr))
            )
        );
        assert!(c["vgpr_max"].as_u64().unwrap() <= 128, "{}", spec.symbol());
        assert_eq!(
            c["group_segment_fixed_bytes"].as_u64(),
            Some(u64::from(spec.kind.lds_bytes()))
        );
        assert_eq!(c["launch_dynamic_lds_bytes"].as_u64(), Some(0));
        for flag in [
            "require_wave32",
            "require_zero_spills",
            "require_zero_private",
        ] {
            assert_eq!(c[flag], true, "{flag}");
        }
        let forbidden: Vec<&str> = c["forbidden"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_str().unwrap())
            .collect();
        for f in [
            "scratch_*",
            "flat_*",
            "v_wmma*",
            "v_swmmac*",
            "v_fma_mix*",
            "v_dot*",
            "v_pk_*",
            "s_wait_loadcnt*",
            "s_wait_alu*",
        ] {
            assert!(forbidden.contains(&f), "{f}");
        }
        assert!(
            !forbidden
                .iter()
                .any(|f| f.starts_with("ds_") || f.starts_with("s_barrier")),
            "this kernel stages X in LDS"
        );
        for (name, n) in c["counts"].as_object().unwrap() {
            let have = e.s_text.lines().filter(|l| mnemonic(l) == name).count() as u64;
            assert_eq!(n.as_u64(), Some(have), "{} counts {name}", spec.symbol());
        }
        assert_eq!(
            c["counts"]["s_barrier"].as_u64(),
            Some(if gate(spec) { 3 } else { 1 }),
            "{}",
            spec.symbol()
        );
    }
}

/// The shipped `kernels/mq_i4_moe_gemv_cn_pm_gfx1151.hxaco` is exactly what the builder emits today (the whole offload
/// bundle, byte for byte, native writer, no ROCm), and the committed contract files are the derived contracts.
#[test]
fn committed_bundle_and_contracts_equal_fresh_emission() {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let path = format!("{root}/kernels/{MODULE}.hxaco");
    let committed = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel mq_i4_moe_gemv_cn --arch gfx1151 --epi all ... --bundle)"));
    let (emitted, text, _) = cn::emit_module().unwrap();
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    let fresh = native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET);
    assert!(
        committed == fresh,
        "{MODULE}: fresh emission differs from the committed bundle {path}"
    );
    for (spec, e) in arch_specs().into_iter().zip(&emitted) {
        let file = format!(
            "{}/kernels/mq_i4_moe_gemv_cn.gfx1151.{}.contract.json",
            env!("CARGO_MANIFEST_DIR"),
            spec.tag()
        );
        let committed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).unwrap_or_else(|e| panic!("{file}: {e}")))
                .unwrap();
        assert_eq!(
            committed,
            cn::contract(spec, e),
            "{file}: committed contract differs from the derived contract"
        );
    }
}

/// The module and every entry assemble with the native writer (no ROCm), and the bundle wraps the object.
#[test]
fn native_writer_assembles_every_entry_and_the_module() {
    let (emitted, text, _) = cn::emit_module().unwrap();
    for e in &emitted {
        let elf = native::assemble(&e.s_text, Arch::Gfx1151)
            .unwrap_or_else(|err| panic!("{}: {err}", e.proof.kernel_id));
        assert!(elf.starts_with(b"\x7fELF"));
    }
    let elf = native::assemble(&text, Arch::Gfx1151).unwrap();
    assert!(native::bundle(&elf, Arch::Gfx1151, native::DEFAULT_HOST_TARGET).len() > elf.len());
}

/// The expert-id range check is the FIRST use of the loaded id (after the `n_experts` kernarg load) and is an UNSIGNED
/// `s_cmp_ge_u32` immediately followed by the uniform branch to the exit label; before it the kernel has issued no
/// barrier, no LDS access, no global access and no scalar load but the kernarg loads, the `offsets[u]` pair load and the
/// `expert_ids[u]` load itself; the id shift, the weight-pointer table load, the slot-id loads and every X / weight load
/// follow it; the exit is shared with the `wg_y`, `M`, `K` and slot-count guards.
#[test]
fn expert_id_guard_is_the_first_use_of_the_id_before_any_table_read_prefetch_or_barrier() {
    const EID: &str = "s34";
    const NEXP: &str = "s100";
    for spec in arch_specs() {
        let e = cn::emit(spec).unwrap();
        let (tag, symbol) = (spec.tag(), spec.symbol());
        let lines: Vec<&str> = e.s_text.lines().map(str::trim).collect();
        let exit_branch = format!("s_cbranch_scc1 .Lcn_{tag}_exit");
        let pos = |pred: &dyn Fn(&str) -> bool| {
            lines
                .iter()
                .position(|l| pred(*l))
                .unwrap_or_else(|| panic!("{symbol}: line not found"))
        };
        let uses_eid = |l: &str| {
            l.split_whitespace()
                .skip(1)
                .any(|t| t.trim_end_matches(',') == EID)
        };
        let nexp_load = pos(&|l: &str| l == "s_load_b32 s100, s[0:1], 0x58");
        let eid_load = pos(&|l: &str| l.starts_with("s_load_b32 s34, "));
        let cmp = (eid_load + 1..lines.len())
            .find(|&i| uses_eid(lines[i]))
            .unwrap();
        assert_eq!(
            lines[cmp],
            format!("s_cmp_ge_u32 {EID}, {NEXP}"),
            "{symbol}: the first use of the id is the unsigned n_experts compare"
        );
        assert!(
            nexp_load < eid_load && eid_load < cmp,
            "{symbol}: n_experts load, id load, compare order"
        );
        assert_eq!(
            lines[cmp + 1],
            exit_branch,
            "{symbol}: uniform exit right after the compare"
        );
        assert!(
            !lines[..cmp]
                .iter()
                .any(|l| uses_eid(l) && !l.starts_with("s_load_b32 s34, ")),
            "{symbol}: id used before the compare"
        );
        // Nothing before the compare touches memory but the kernarg loads, the offsets pair and the expert_ids[u] load.
        for l in &lines[..cmp] {
            let m = mnemonic(l);
            assert!(
                !(m == "s_barrier"
                    || m.starts_with("global_")
                    || m.starts_with("ds_load")
                    || m.starts_with("ds_store")
                    || m.starts_with("flat_")
                    || m.starts_with("scratch_")),
                "{symbol}: `{l}` before the id guard"
            );
            if m.starts_with("s_load_") {
                let base = l.split(", ").nth(1).unwrap_or_default();
                assert!(
                    base == "s[0:1]" || base == "s[36:37]",
                    "{symbol}: `{l}` is neither a kernarg load nor the offsets / expert_ids load"
                );
            }
        }
        // The id shift, the weight-pointer table load, the slot-id loads and every X / weight load follow the guard.
        let shift = pos(&|l: &str| l == "s_lshl_b32 s41, s34, 3");
        assert!(cmp < shift, "{symbol}: id shift after the guard");
        let table_loads: Vec<usize> = (0..lines.len())
            .filter(|&i| lines[i].starts_with("s_load_b64 s[38:39], "))
            .collect();
        assert_eq!(
            table_loads.len(),
            1,
            "{symbol}: one weight-pointer table load"
        );
        assert!(
            cmp < table_loads[0],
            "{symbol}: weight-pointer table load after the guard"
        );
        for i in (0..lines.len())
            .filter(|&i| (57..=65).any(|r| lines[i].starts_with(&format!("s_load_b32 s{r}, "))))
        {
            assert!(cmp < i, "{symbol}: slot-id load before the guard")
        }
        for i in (0..lines.len()).filter(|&i| mnemonic(lines[i]).starts_with("global_load")) {
            assert!(cmp < i, "{symbol}: X / weight load before the guard")
        }
        // Shared exit: wg_y, M, K, slot-count (two), offsets and expert-id guards all branch to it before the body.
        let body = pos(&|l: &str| l == format!(".Lcn_{tag}_body:"));
        assert_eq!(
            lines[..body].iter().filter(|l| **l == exit_branch).count(),
            7,
            "{symbol}: wg_y, M, K, n == 0, n > 8, offsets, expert-id exits"
        );
        assert!(cmp < body, "{symbol}: guard in the prologue");
        assert!(
            lines[..body].contains(&"s_cmp_ge_u32 s3, s24"),
            "{symbol}: wg_y guard unchanged"
        );
    }
}

// ------------------------------------------------------------------------------------------------ CPU interpreter
// The real emitted code objects, lifted and run on the peacemaker CPU interpreter: invalid expert ids exit with no store
// and no dereference, valid ids are untouched by the guard and equal, bit for bit, the incumbent K1 narrow entry (the
// kernel Cn replaces, `mq_i4_moe_gemv.rs`, itself guarded) on the same descriptors. Tensor data are random full-mantissa as
// in K1's wide-emulator test; only experts 0 and `E_N - 1` are mapped, and the weight-pointer table is exactly `E_N`
// entries, so a table read at or past `E_N` (or an expert dereference through a null table slot) faults.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_moe_gemv as k1;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;
const E_KERNARG: u64 = 0x1_0000;
const E_KERNARG_REF: u64 = 0x2_0000;
const E_TABLE: u64 = 0x10_0000;
const E_IDS: u64 = 0x18_0000;
const E_OFFS: u64 = 0x20_0000;
const E_SLOTS: u64 = 0x28_0000;
const E_EXPERTS: u64 = 0x1000_0000;
const E_STRIDE: u64 = 0x40_0000;
const E_X: u64 = 0x8000_0000;
const E_Y0: u64 = 0x9000_0000;
const E_Y1: u64 = 0xa000_0000;
const E_R0: u64 = 0xb000_0000;
const E_R1: u64 = 0xc000_0000;
/// The extent of the weight-pointer table (`n_experts`) of every case.
const E_N: u32 = 8;
/// Two decode lanes (`C`): twenty original slots; the one expert serves the original slots below (its `n = 3`).
const LANES: u32 = 2;
const SLOT_IDS: [u32; 3] = [3, 7, 12];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Normal f16 scale with exponent field 8..=20: `-8 * sc` is then exactly the symmetric MQ4 zero.
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

/// Planar W3 tensor: per row `K*3/8` code bytes (128 3-bit codes per 48-byte group), then the f16 scale plane.
fn w3_tensor(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let (groups, row_bytes) = (k / 128, k * 3 / 8);
    let mut out = vec![0u8; m * row_bytes + 2 * m * groups];
    for r in 0..m {
        for g in 0..groups {
            let base = r * row_bytes + g * 48;
            for j in 0..128 {
                let code = rng.below(8) as u32;
                for b in 0..3 {
                    if (code >> b) & 1 == 1 {
                        out[base + (3 * j + b) / 8] |= 1 << ((3 * j + b) % 8);
                    }
                }
            }
            let off = m * row_bytes + 2 * (r * groups + g);
            out[off..off + 2].copy_from_slice(&scale_bits(rng).to_le_bytes());
        }
    }
    out
}

/// MQ4: gate G256 groups of 136 B (`[sc0 zp0 sc1 zp1]`, 128 payload bytes at +8), down G128 groups of 68 B (`[sc zp]`, 64 at +4).
fn mq4_tensor(rng: &mut Rng, kind: Kind, m: usize, k: usize) -> Vec<u8> {
    let (cols, stride, halves, payload_at, payload) = match kind {
        Kind::GateUp => (256, 136, 2, 8, 128),
        Kind::Down => (128, 68, 1, 4, 64),
    };
    let groups = k / cols;
    let mut out = vec![0u8; m * groups * stride];
    for r in 0..m {
        for g in 0..groups {
            let base = (r * groups + g) * stride;
            for h in 0..halves {
                let sc = scale_bits(rng);
                out[base + 4 * h..base + 4 * h + 2].copy_from_slice(&sc.to_le_bytes());
                out[base + 4 * h + 2..base + 4 * h + 4]
                    .copy_from_slice(&sym_zero_bits(sc).to_le_bytes());
            }
            for byte in &mut out[base + payload_at..base + payload_at + payload] {
                *byte = rng.next() as u8;
            }
        }
    }
    out
}

/// Full-mantissa activations over a wide exponent band, random signs.
fn x_rows(rng: &mut Rng, rows: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * k * 4);
    for _ in 0..rows * k {
        let (s, e, m) = (
            (rng.next() & 1) as u32,
            118 + rng.below(19) as u32,
            (rng.next() & 0x7f_ffff) as u32,
        );
        out.extend(((s << 31) | (e << 23) | m).to_le_bytes());
    }
    out
}

struct Rig {
    cn: peacemaker_ir::Program,
    k1: peacemaker_ir::Program,
}
impl Rig {
    fn new() -> Self {
        let lift = |text: &str| {
            let elf = native::assemble(text, Arch::Gfx1151).unwrap();
            peacemaker_lift::lift_object(
                &elf,
                peacemaker_lift::Options {
                    frontend: Frontend::Builder,
                },
            )
            .unwrap()
            .program
        };
        let (_, cn_text, _) = cn::emit_module().unwrap();
        let (_, k1_text, _) = k1::emit_module(Arch::Gfx1151).unwrap();
        Rig {
            cn: lift(&cn_text),
            k1: lift(&k1_text),
        }
    }
}

/// One entry's deterministic data: the X rows and one expert tensor of the entry's codec.
struct Data {
    spec: Spec,
    m: usize,
    k: usize,
    xdiv: u32,
    xsrc: u32,
    ys: usize,
    x: Vec<u8>,
    tensor: Vec<u8>,
}
fn data(spec: Spec) -> Data {
    // gate: `C` X rows (slot / 10), Y rows `M/2` per slot; down: `10*C` X rows (slot / 1), Y rows `M` per slot.
    let (m, xdiv, xrows, ys) = match spec.kind {
        Kind::GateUp => (1280usize, 10u32, LANES as usize, 640usize),
        Kind::Down => (2560, 1, 10 * LANES as usize, 2560),
    };
    let k = spec.kind.k() as usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ k as u64 ^ ((spec.codec as u64) << 40));
    let tensor = match spec.codec {
        Codec::Mq3I4 => w3_tensor(&mut rng, m, k),
        Codec::Mq4Sym => mq4_tensor(&mut rng, spec.kind, m, k),
    };
    let x = x_rows(&mut rng, xrows, k);
    Data {
        spec,
        m,
        k,
        xdiv,
        xsrc: xrows as u32,
        ys,
        x,
        tensor,
    }
}

/// The Y words and the number of Y bytes the launches wrote.
struct Out {
    y: Vec<u8>,
    written: usize,
}

/// Everything but the kernarg blocks: experts `0` and `E_N - 1` mapped (the same tensor), the exactly-`E_N`-entry weight table,
/// the one-expert descriptors (`expert_ids = [id]`, `offsets = [0, 3]`, `slot_ids = SLOT_IDS`), X, and the Cn (`y0`/`y1`)
/// and K1 reference (`r0`/`r1`) outputs, all sentinel-filled with guard words on both sides.
fn memory(d: &Data, id: i32) -> Result<Memory, String> {
    let mut mem = Memory::default();
    let mut table = vec![0u8; E_N as usize * 8];
    for expert in [0u32, E_N - 1] {
        let base = E_EXPERTS + u64::from(expert) * E_STRIDE;
        assert!((d.tensor.len() as u64) < E_STRIDE);
        table[expert as usize * 8..expert as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
        mem.map(format!("e{expert}"), base, d.tensor.clone())?;
    }
    mem.map("table".into(), E_TABLE, table)?;
    mem.map("ids".into(), E_IDS, id.to_le_bytes().to_vec())?;
    mem.map(
        "offs".into(),
        E_OFFS,
        [0u32, SLOT_IDS.len() as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
    )?;
    mem.map(
        "slots".into(),
        E_SLOTS,
        SLOT_IDS.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    mem.map("x".into(), E_X, d.x.clone())?;
    let ybytes = 10 * LANES as usize * d.ys * 4 + 2 * GUARD;
    let gate = d.spec.kind == Kind::GateUp;
    for (name, base, live) in [
        ("y0", E_Y0, true),
        ("y1", E_Y1, gate),
        ("r0", E_R0, true),
        ("r1", E_R1, gate),
    ] {
        if live {
            mem.map(name.into(), base, vec![SENTINEL; ybytes])?;
        }
    }
    Ok(mem)
}

/// The 96-byte K1 / Cn kernarg block (identical ABI): `n_experts@88`, zero `reserved@92`.
fn kernarg(d: &Data, y0: u64, y1: u64, n_experts: u32) -> Vec<u8> {
    let mut karg = Vec::with_capacity(cn::KERNARG_BYTES as usize);
    for ptr in [
        E_TABLE,
        E_IDS,
        E_OFFS,
        E_SLOTS,
        E_X,
        y0 + GUARD as u64,
        if d.spec.kind == Kind::GateUp {
            y1 + GUARD as u64
        } else {
            0
        },
    ] {
        karg.extend(ptr.to_le_bytes());
    }
    // M, K, U, C, x_row_div, x_src_rows, x_row_stride, y_slot_stride, n_experts, reserved.
    for v in [
        d.m as u32,
        d.k as u32,
        1,
        LANES,
        d.xdiv,
        d.xsrc,
        d.k as u32,
        d.ys as u32,
        n_experts,
        0,
    ] {
        karg.extend(v.to_le_bytes());
    }
    assert_eq!(karg.len(), cn::KERNARG_BYTES as usize);
    karg
}

fn collect(mem: &Memory, names: [&str; 2]) -> Out {
    let (mut y, mut written) = (Vec::new(), 0);
    for r in mem
        .regions
        .iter()
        .filter(|r| r.name == names[0] || r.name == names[1])
    {
        y.extend_from_slice(&r.bytes);
        written += r.written.count();
    }
    Out { y, written }
}

/// Cn: workgroups `(0, 0)` and `(M/8 - 1, 0)` (first and last row block; for the gate the second lands in `Y1`) of the one
/// expert `expert_ids[0] = id`, `n_experts` as given.
fn run_cn(rig: &Rig, d: &Data, id: i32, n_experts: u32) -> Result<Out, String> {
    let mut mem = memory(d, id)?;
    mem.map(
        "kernarg".into(),
        E_KERNARG,
        kernarg(d, E_Y0, E_Y1, n_experts),
    )?;
    let emu = Emulator::new(&rig.cn, &d.spec.symbol())?;
    for wg in [[0u32, 0, 0], [d.m as u32 / 8 - 1, 0, 0]] {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG,
                workgroup: wg,
                block: [256, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    Ok(collect(&mem, ["y0", "y1"]))
}

/// The incumbent K1 narrow entry of the same codec and kind on the same descriptors (valid ids only), writing `r0`/`r1`: one
/// wave32 workgroup per output row, the sixteen rows Cn's two workgroups cover.
fn run_k1(rig: &Rig, d: &Data, id: i32) -> Result<Out, String> {
    let mut mem = memory(d, id)?;
    mem.map("kernarg".into(), E_KERNARG_REF, kernarg(d, E_R0, E_R1, E_N))?;
    let codec = match d.spec.codec {
        Codec::Mq3I4 => k1::Codec::Mq3I4,
        Codec::Mq4Sym => k1::Codec::Mq4Sym,
    };
    let kind = match d.spec.kind {
        Kind::GateUp => k1::Kind::GateUp,
        Kind::Down => k1::Kind::Down,
    };
    let emu = Emulator::new(
        &rig.k1,
        &k1::Spec::narrow(Arch::Gfx1151, codec, kind).symbol(),
    )?;
    for row in (0..8).chain(d.m as u32 - 8..d.m as u32) {
        emu.run_workgroup(
            &Launch {
                kernarg_ptr: E_KERNARG_REF,
                workgroup: [row, 0, 0],
                block: [32, 1, 1],
                dynamic_lds: 0,
            },
            &mut mem,
        )?;
    }
    Ok(collect(&mem, ["r0", "r1"]))
}

/// Invalid ids `-1`, `i32::MIN` (whose `<< 3` is 0: unguarded it would alias table entry 0 and run the expert through it),
/// `n` and `n + 1` (one past a table with exactly `n` entries, whose weight-pointer read would fault) exit with no store; the
/// valid id `n - 1` produces exactly the bytes the permissive `n_experts = u32::MAX` run produces AND the bytes of the
/// incumbent K1 narrow entry (an independent reference), and writes the 48 lane-0 words of the two workgroups (8 waves x 3
/// slots each). All four contracts (both codecs x gate / down).
#[test]
fn out_of_range_expert_ids_exit_with_no_store_or_dereference_and_valid_ids_are_unchanged() {
    let rig = Rig::new();
    let n = E_N as i32;
    for spec in arch_specs() {
        let d = data(spec);
        let case = spec.tag();
        for id in [-1, i32::MIN, n, n + 1] {
            let out = run_cn(&rig, &d, id, E_N).unwrap_or_else(|e| panic!("{case}: id {id}: interpreter fault (an expert or table read escaped the guard): {e}"));
            assert_eq!(out.written, 0, "{case}: id {id} stored");
            assert!(
                out.y.iter().all(|&b| b == SENTINEL),
                "{case}: id {id} changed Y"
            );
        }
        let valid = run_cn(&rig, &d, n - 1, E_N)
            .unwrap_or_else(|e| panic!("{case}: valid id {}: {e}", n - 1));
        assert_eq!(
            valid.written,
            2 * 8 * SLOT_IDS.len() * 4,
            "{case}: valid id writes one lane-0 word per wave and live slot of both workgroups"
        );
        for w in valid
            .y
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .filter(|&w| w != 0xa5a5_a5a5)
        {
            assert!(
                f32::from_bits(w).is_finite(),
                "{case}: non-finite output word {w:#x}"
            );
        }
        let open = run_cn(&rig, &d, n - 1, u32::MAX).unwrap();
        assert!(
            valid.y == open.y && valid.written == open.written,
            "{case}: the guard changed a valid launch"
        );
        let reference =
            run_k1(&rig, &d, n - 1).unwrap_or_else(|e| panic!("{case}: K1 reference: {e}"));
        assert_eq!(
            reference.written, valid.written,
            "{case}: K1 reference written bytes"
        );
        assert!(
            valid.y == reference.y,
            "{case}: Cn differs from the incumbent K1 on the same valid launch"
        );
    }
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use peacemaker_isa::{ledger_replay, pm_check, toolchain::IsaShapeContract};

    /// Every symbol's waits replay independently of the builder's ledger (entry and module text), the contract
    /// parses as an `IsaShapeContract`, and M7 lifts the natively linked object byte-exactly with no obligations.
    #[test]
    fn every_export_passes_wait_replay_and_m7() {
        let (emitted, text, _) = cn::emit_module().unwrap();
        for (spec, e) in arch_specs().into_iter().zip(&emitted) {
            ledger_replay::replay_waits(&e.s_text, Arch::Gfx1151)
                .unwrap_or_else(|err| panic!("{}: {err}", e.proof.variant));
            assert!(e.s_text.contains(".kernarg_segment_size: 96\n"));
            serde_json::from_value::<IsaShapeContract>(cn::contract(spec, e)).unwrap();
        }
        ledger_replay::replay_waits(&text, Arch::Gfx1151).unwrap();
        let dir =
            std::env::temp_dir().join(format!("peacemaker-isa-mqi4cn-m7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let co = dir.join("module.co");
        std::fs::write(&co, native::assemble(&text, Arch::Gfx1151).unwrap()).unwrap();
        for symbol in SYMBOLS {
            let m7 =
                pm_check::m7(&co, "gfx1151", symbol).unwrap_or_else(|e| panic!("{symbol}: {e}"));
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{symbol}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
