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

fn geometry(
    experts: u32,
    top_k: u32,
    hidden: u32,
    intermediate: u32,
) -> qwen4_moe_sym::ExpertGeometry {
    qwen4_moe_sym::ExpertGeometry {
        experts,
        top_k,
        hidden,
        intermediate,
    }
}

/// GU `M = 2I, K = H, epochs = H/128, x_row_div = top_k`; down `M = H, K = I, epochs = I/128` (slot rows).
#[test]
fn geometry_plan_derives_gate_up_and_down_shapes() {
    let mq4 = qwen4_moe_sym::WeightCodec::Mq4Sym;
    // (experts, top_k, H, I, GU epochs, down epochs)
    for (e, t, h, i, gu_epochs, down_epochs) in
        [(512, 10, 2560, 640, 20, 5), (256, 8, 2048, 512, 16, 4)]
    {
        for arch in ARCHES {
            let plan = geometry(e, t, h, i).plan(arch, mq4).unwrap();
            assert_eq!(
                (plan.arch, plan.codec, plan.geometry),
                (arch, mq4, geometry(e, t, h, i))
            );
            assert_eq!(plan.tile_widths, qwen4_moe_sym::tile_widths(arch));
            assert_eq!(plan.descriptor_table_bytes, e * 8);
            let (gu, down) = (plan.gate_up, plan.down);
            assert_eq!(
                (gu.kind, gu.m, gu.k, gu.epochs, gu.x_row_div, gu.grid_x),
                (Kind::GateUp, 2 * i, h, gu_epochs, t, 2 * i / 64)
            );
            assert_eq!(
                (
                    down.kind,
                    down.m,
                    down.k,
                    down.epochs,
                    down.x_row_div,
                    down.grid_x
                ),
                (Kind::Down, h, i, down_epochs, 1, h / 64)
            );
            assert_eq!(
                (gu.row_bytes, down.row_bytes),
                (gu_epochs * 68, down_epochs * 68)
            );
            for &nt in plan.tile_widths {
                for kind in Kind::ALL {
                    assert_eq!(
                        plan.spec(kind, nt).unwrap(),
                        Spec {
                            arch,
                            kind,
                            nt,
                            rr: 1,
                            codec: mq4
                        }
                    );
                }
            }
            assert!(plan.spec(Kind::Down, 2).is_err());
        }
    }
    // The down tail is odd exactly when I/128 is odd.
    assert_eq!(
        geometry(512, 10, 2560, 640)
            .plan(Arch::Gfx1201, mq4)
            .unwrap()
            .down
            .epochs
            % 2,
        1
    );
    assert_eq!(
        geometry(256, 8, 2048, 512)
            .plan(Arch::Gfx1201, mq4)
            .unwrap()
            .down
            .epochs
            % 2,
        0
    );
}

#[test]
fn geometry_plan_refuses_invalid_shapes() {
    let mq4 = qwen4_moe_sym::WeightCodec::Mq4Sym;
    let bad = |g: qwen4_moe_sym::ExpertGeometry, what: &str| {
        for arch in ARCHES {
            assert!(g.plan(arch, mq4).is_err(), "{what}: {g:?} {arch:?}");
        }
    };
    let ok = geometry(256, 8, 2048, 512);
    // Gate/up K = H % 256 (128 is a multiple of the down K step only).
    bad(geometry(256, 8, 128, 512), "H % 256");
    bad(geometry(256, 8, 2048 + 128, 512), "H % 256");
    bad(geometry(256, 8, 0, 512), "H == 0");
    // Down K = I % 128 and M = H % 64; gate/up M = 2I % 64.
    bad(geometry(256, 8, 2048, 64), "I % 128");
    bad(geometry(256, 8, 2048, 512 + 32), "I % 128");
    bad(geometry(256, 8, 2048, 0), "I == 0");
    bad(geometry(256, 8, 2048, u32::MAX / 2 + 1), "2I overflows u32");
    // top_k 1..=16.
    bad(geometry(256, 0, 2048, 512), "top_k 0");
    bad(geometry(256, 17, 2048, 512), "top_k 17");
    for top_k in [1, 16] {
        for arch in ARCHES {
            assert_eq!(
                geometry(256, top_k, 2048, 512)
                    .plan(arch, mq4)
                    .unwrap()
                    .gate_up
                    .x_row_div,
                top_k
            );
        }
    }
    // experts >= 1 and the descriptor table (experts * 8 bytes) fits u32.
    bad(geometry(0, 8, 2048, 512), "experts 0");
    bad(geometry(u32::MAX / 8 + 1, 8, 2048, 512), "descriptor table");
    for arch in ARCHES {
        assert!(geometry(u32::MAX / 8, 8, 2048, 512).plan(arch, mq4).is_ok());
        assert!(geometry(1, 1, 2048, 512).plan(arch, mq4).is_ok());
    }
    // The 32-bit row arithmetic of the kernel: 24-bit row bytes, 32-bit expert weight row base.
    bad(geometry(256, 8, 128 * 250_000, 512), "row bytes >= 2^24");
    bad(
        geometry(256, 8, 2048, 128 * 100_000),
        "expert matrix bytes >= 2^32",
    );
    // Only the certified MQ4 objects on gfx1151/gfx1201.
    for arch in [Arch::Gfx1100, Arch::Gfx942] {
        assert!(ok.plan(arch, mq4).is_err(), "{arch:?}");
    }
    for codec in [
        qwen4_moe_sym::WeightCodec::Mq3I4,
        qwen4_moe_sym::WeightCodec::E8pI4,
    ] {
        for arch in ARCHES {
            assert!(ok.plan(arch, codec).is_err(), "{codec:?} {arch:?}");
        }
    }
}

/// Every planned entry (NT1/NT4 gfx1151, NT1/NT4/NT8 gfx1201) reads `K` and `n_experts` from the kernarg at run time
/// (so one object serves every planned `H`/`I`/expert count), exits on an out-of-range expert id before the
/// pointer-table load, derives its row bytes and trips from the runtime `K`, and only the down entry has the odd
/// K128 epoch tail.
#[test]
fn planned_entries_take_k_and_experts_at_run_time_and_down_has_the_odd_tail() {
    let plan = |arch| {
        geometry(256, 8, 2048, 512)
            .plan(arch, qwen4_moe_sym::WeightCodec::Mq4Sym)
            .unwrap()
    };
    for arch in ARCHES {
        let plan = plan(arch);
        let mut census = Vec::new();
        for &nt in plan.tile_widths {
            for kind in Kind::ALL {
                let spec = plan.spec(kind, nt).unwrap();
                let text = qwen4_moe_sym::emit(spec).unwrap().s_text;
                let has = |needle: &str| text.contains(needle);
                // Runtime kernarg K (+ M) and the expert-count guard.
                assert!(
                    has("s_load_b64 s[18:19], s[0:1], 0x28"),
                    "{spec:?}: M/K kernarg"
                );
                assert!(
                    has("s_load_b32 s5, s[0:1], 0x3c"),
                    "{spec:?}: n_experts kernarg"
                );
                assert!(
                    has("s_cmp_ge_u32 s23, s5"),
                    "{spec:?}: unsigned expert id guard"
                );
                // Trips = (K/128 - 1) / 2 from the runtime K.
                assert!(
                    has("s_lshr_b32 s44, s19, 7") && has("s_lshr_b32 s44, s44, 1"),
                    "{spec:?}: trips"
                );
                let odd_tail = has("s_bitcmp1_b32 s19, 7");
                match kind {
                    Kind::GateUp => {
                        assert!(
                            has("s_lshr_b32 s46, s19, 8") && has("s_mulk_i32 s46, 0x88"),
                            "{spec:?}: QT44 row bytes"
                        );
                        assert!(
                            !odd_tail && !has("tail_odd"),
                            "{spec:?}: gate/up K % 256 == 0 has no odd tail"
                        );
                    }
                    Kind::Down => {
                        assert!(
                            has("s_lshr_b32 s46, s19, 7") && has("s_mulk_i32 s46, 0x44"),
                            "{spec:?}: QT53 row bytes"
                        );
                        assert!(odd_tail && has("tail_odd"), "{spec:?}: down odd epoch tail");
                    }
                }
                census.push((nt, kind));
            }
        }
        let nts: Vec<u8> = census.iter().step_by(2).map(|c| c.0).collect();
        assert_eq!(
            nts,
            if arch == Arch::Gfx1201 {
                vec![1, 4, 8]
            } else {
                vec![1, 4]
            },
            "{arch:?}"
        );
    }
}

/// The shipped gfx1201 down-NT8 two-slot stream: only two activation tiles
/// resident (d in v154..155, X in v160..175), SUM72..135 and both weight sets
/// W136..153 unchanged. The live top is at most v175, granule-aligned to the
/// 176-VGPR descriptor; the kernel-independent descriptors keep the 64-byte
/// ABI, wave32, no LDS and no spills/private traffic.
#[test]
fn down_nt8_stream_keeps_two_slots_resident_in_176_vgprs() {
    let spec = Spec {
        arch: Arch::Gfx1201,
        kind: Kind::Down,
        nt: 8,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
    };
    let e = qwen4_moe_sym::emit(spec).unwrap();
    assert_eq!(e.proof.arch, Arch::Gfx1201);
    // `Builder::finish` checks every range's lifetime ordering and the
    // 256-VGPR budget; emission succeeding is the validity exercise, and the
    // live-top assertion below pins the streamed shape.
    let vgprs: Vec<_> = e
        .proof
        .reg_plan
        .iter()
        .filter(|r| r.kind == peacemaker_isa::reg::Kind::V)
        .collect();
    let top = vgprs
        .iter()
        .map(|r| u16::from(r.base) + u16::from(r.len))
        .max()
        .unwrap();
    assert!(top <= 176, "down NT8 stream: live VGPR top {top}");
    assert_eq!(top, 176, "down NT8 stream: streamed plan tops at X v160..175");
    assert_eq!(e.shape.next_free_vgpr, 176, "down NT8 stream: VGPR plan");
    assert!(
        e.shape.next_free_sgpr <= 104,
        "down NT8 stream: SGPR plan {}",
        e.shape.next_free_sgpr
    );
    // Only the d fragments (one dword each) and the X fragments (four dword
    // pairs each) of slots 0 and 1 are kernel-live; slots 2..7 stream
    // through the same physical pair in turn (`G::plan`: `p.v::<2>("x",
    // self.x(j) + 2 * c)` for `c` in 0..4).
    let d: Vec<_> = e
        .proof
        .reg_plan
        .iter()
        .filter(|r| r.kind == peacemaker_isa::reg::Kind::V && r.name.as_ref() == "d")
        .collect();
    assert_eq!(
        d.iter().map(|r| (r.base, r.len)).collect::<Vec<_>>(),
        [(154, 1), (155, 1)],
        "down NT8 stream: d fragments"
    );
    let mut x: Vec<_> = e
        .proof
        .reg_plan
        .iter()
        .filter(|r| r.kind == peacemaker_isa::reg::Kind::V && r.name.as_ref() == "x")
        .map(|r| (r.base, r.len))
        .collect();
    x.sort();
    assert_eq!(
        x,
        [
            (160, 2),
            (162, 2),
            (164, 2),
            (166, 2),
            (168, 2),
            (170, 2),
            (172, 2),
            (174, 2)
        ],
        "down NT8 stream: resident X pair"
    );
    // Descriptor and kernel-independent shape: the granule-aligned 176-VGPR
    // reservation, the unchanged 104-SGPR plan, wave32, no LDS, no spills
    // and no private traffic.
    assert_eq!(e.shape.barriers, 0, "down NT8 stream: no barrier");
    assert_eq!(spec.lds_bytes(), 0, "down NT8 stream: no static LDS");
    let text = &e.s_text;
    let field = |key: &str| -> Vec<u32> {
        text.lines()
            .filter_map(|l| l.trim().strip_prefix(key))
            .map(|v| v.trim().parse().unwrap())
            .collect()
    };
    assert_eq!(
        field(".amdhsa_next_free_vgpr"),
        [176],
        "down NT8 stream: VGPR descriptor"
    );
    assert_eq!(
        u32::from(e.shape.next_free_vgpr),
        field(".amdhsa_next_free_vgpr")[0],
        "down NT8 stream: shape/descriptor VGPRs"
    );
    assert!(
        text.contains(".vgpr_count: 176\n"),
        "down NT8 stream: metadata VGPRs"
    );
    let sgprs = field(".amdhsa_next_free_sgpr");
    assert_eq!(sgprs.len(), 1, "down NT8 stream: one SGPR descriptor");
    assert!(
        sgprs[0] <= 104 && u32::from(e.shape.next_free_sgpr) == sgprs[0],
        "down NT8 stream: {} SGPRs",
        sgprs[0]
    );
    assert_eq!(spec.threads(), 128, "down NT8 stream: block is four waves");
    for needle in [
        ".amdhsa_wavefront_size32 1\n",
        ".wavefront_size: 32\n",
        ".group_segment_fixed_size: 0\n",
        ".amdhsa_group_segment_fixed_size 0\n",
        ".private_segment_fixed_size: 0\n",
        ".sgpr_spill_count: 0\n",
        ".vgpr_spill_count: 0\n",
        ".kernarg_segment_size: 64\n",
        ".max_flat_workgroup_size: 128\n",
    ] {
        assert!(text.contains(needle), "down NT8 stream: {needle:?}");
    }
    for forbidden in [
        "scratch_",
        "flat_",
        "global_",
        "ds_load",
        "ds_store",
        "s_barrier",
        "s_waitcnt",
    ] {
        assert!(
            !text.lines().any(|l| {
                let code = l.split("//").next().unwrap_or("");
                code.split_whitespace().next().is_some_and(|t| t.starts_with(forbidden))
            }),
            "down NT8 stream: no {forbidden} traffic"
        );
    }
    // Same 64-byte ABI as every other entry: the M/K load, the n_experts
    // guard and the 60-byte-by-value layout.
    assert!(
        text.contains("s_load_b64 s[18:19], s[0:1], 0x28"),
        "down NT8 stream: M/K kernarg"
    );
    assert!(
        text.contains("s_load_b32 s5, s[0:1], 0x3c"),
        "down NT8 stream: n_experts kernarg"
    );
    let down_nt4 = Spec {
        arch: Arch::Gfx1151,
        kind: Kind::Down,
        nt: 4,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
    };
    // Same 64-byte ABI value layout as the rest of the family: size 64 and
    // the identical address-space/value-kind entries (pointer, hidden M/K,
    // x_row_div, m_total, x_src_rows, n_experts).
    let abi = |s: Spec| {
        (
            s.kernargs().size,
            s.kernargs().args.iter().map(|a| (a.offset, a.size)).collect::<Vec<_>>(),
        )
    };
    assert_eq!(abi(spec), abi(down_nt4), "down NT8 stream: ABI layout");
    assert_eq!(spec.kernargs().size, 64, "down NT8 stream: ABI bytes");
}

/// The stream folds and refills per fold: the per-fold WMMA/FMA census, the
/// streamed tile-load census (b32 52, b64 168), per-tile waits beside the
/// paired folds, no extra math ops, and the split epoch advance
/// (`s[24:25] += 136` then X0 before compute1, X1 advanced after it).
fn down_nt8_stream_fold_census_and_split_epoch_advance() {
    let spec = Spec {
        arch: Arch::Gfx1201,
        kind: Kind::Down,
        nt: 8,
        rr: 1,
        codec: qwen4_moe_sym::WeightCodec::Mq4Sym,
    };
    let e = qwen4_moe_sym::emit(spec).unwrap();
    let text = &e.s_text;
    let mnemonic = |l: &str| {
        l.split("//")
            .next()
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string()
    };
    // Static census: the streamed schedule issues 52 b32 / 168 b64 tile
    // loads (the contract's counts), with math identical to the resident
    // schedule.
    for (name, n) in [
        ("buffer_load_b32", 52),
        ("buffer_load_b64", 168),
        ("buffer_store_b128", 8),
        ("ds_swizzle_b32", 40),
        ("s_endpgm", 1),
        ("v_dual_fmac_f32", 160),
        ("v_dual_mul_f32", 320),
        ("v_wmma_i32_16x16x32_iu4", 160),
    ] {
        assert_eq!(count(name), n, "down NT8 stream: {name}");
    }
    // No extra math relative to the shipped down-NT8 contract: every counted
    // math op matches the committed census byte for byte.
    let committed: serde_json::Value = serde_json::from_str(include_str!(
        "../kernels/qwen4_moe_sym.gfx1201.down_nt8.contract.json"
    ))
    .unwrap();
    for math in [
        "v_wmma_i32_16x16x32_iu4",
        "v_dual_fmac_f32",
        "v_dual_mul_f32",
    ] {
        assert_eq!(
            committed["counts"][math].as_u64(),
            Some(count(math)),
            "down NT8 stream: committed {math} census"
        );
    }
    // Per-row WMMA in ascending tile order: each fold's four K32 WMMA steps
    // read the fold tile's resident X dword pair (even tiles v160..167, odd
    // tiles v168..175). Every compute pass folds tiles 0..7 in order, so
    // fold f reads tile f%8; within the WMMA steps, only odd-tile folds
    // address `v[17..]` (the v170+ pairs).
    let wmma: Vec<&str> = text
        .lines()
        .filter(|l| mnemonic(l) == "v_wmma_i32_16x16x32_iu4")
        .collect();
    assert_eq!(wmma.len(), 160, "down NT8 stream: 40 folds x 4 steps");
    assert!(
        wmma.chunks(4).enumerate().all(|(f, chunk)| {
            let odd = f % 8 % 2 == 1;
            chunk.iter().any(|l| l.contains("v[17")) == odd
        }),
        "down NT8 stream: tile order with the resident pair"
    );
    // Per-row scale products (`fold_pass`: `t_j = d * sc_j`): each of the 40
    // folds contributes eight dual-mul packets, all reading the fold's own d
    // slot (v154 even tiles, v155 odd tiles); together a fold's packets
    // cover every row scale v64..71.
    let mul: Vec<&str> = text
        .lines()
        .filter(|l| mnemonic(l) == "v_dual_mul_f32")
        .collect();
    assert_eq!(mul.len(), 320, "down NT8 stream: 40 folds x 8 products");
    assert!(
        mul.iter().all(|l| l.contains("v154") || l.contains("v155")),
        "down NT8 stream: every scale product reads a d slot"
    );
    assert!(
        mul.chunks(8).all(|chunk| (64..72u8).all(|s| {
            let needle = format!("v{s}");
            chunk.iter().any(|l| l.contains(&needle))
        })),
        "down NT8 stream: each fold covers every row scale"
    );
    // Per-row FMA (`fold_pass`: `sum_j = fma(t_j, C_j, sum_j)`): each of the
    // 40 folds contributes four dual-FMAC packets accumulating the sums
    // v72..135 from the fold products v56..63 (TPROD) and the chain v48..55
    // (CACC) — never the d slot, which only the scale products above read.
    let fmac: Vec<&str> = text
        .lines()
        .filter(|l| mnemonic(l) == "v_dual_fmac_f32")
        .collect();
    assert_eq!(fmac.len(), 160, "down NT8 stream: 40 folds x 4 packets");
    assert!(
        fmac.iter().all(|l| !l.contains("v154") && !l.contains("v155")),
        "down NT8 stream: no FMA reads the d slots"
    );
    assert!(
        fmac.chunks(4).all(|chunk| (56..64u8).all(|p| {
            let needle = format!("v{p}");
            chunk.iter().any(|l| l.contains(&needle))
        }) && (48..56u8).all(|c| {
            let needle = format!("v{c}");
            chunk.iter().any(|l| l.contains(&needle))
        }) && (72..136u8).all(|s| {
            let needle = format!("v{s}");
            chunk.iter().any(|l| l.contains(&needle))
        })),
        "down NT8 stream: each fold reads TPROD and CACC and writes the sums"
    );
    // Per-tile waits ride with the paired folds: each even tile's wait
    // retires its pair's loads before either slot is overwritten. Check the
    // pair-relevant window: the first fold has its wait before it, and every
    // even tile's first WMMA has a load wait after the previous fold's WMMA
    // and before itself — no assumption about epilogue waits after the
    // last fold.
    let lines: Vec<&str> = text.lines().collect();
    let waits: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let m = mnemonic(l);
            m == "s_wait_loadcnt_dscnt" || m == "s_wait_loadcnt"
        })
        .map(|(i, _)| i)
        .collect();
    let wmma_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| mnemonic(l) == "v_wmma_i32_16x16x32_iu4")
        .map(|(i, _)| i)
        .collect();
    assert!(!waits.is_empty(), "down NT8 stream: pair waits exist");
    assert!(
        waits[0] < wmma_lines[0],
        "down NT8 stream: first pair wait guards the first fold"
    );
    assert!(
        wmma_lines
            .chunks(4)
            .enumerate()
            .filter(|(f, _)| f % 8 % 2 == 0)
            .all(|(_, chunk)| {
                let start = chunk[0];
                let prev_end = wmma_lines
                    .iter()
                    .filter(|&&w| w < start)
                    .max()
                    .copied()
                    .unwrap_or(0);
                waits.iter().any(|&w| prev_end < w && w < start)
            }),
        "down NT8 stream: every even tile fold has its pair wait in-window"
    );
    // Split epoch advance (`kloop` stream arm): the weight descriptor steps
    // one epoch (136, `0x88`) first, then X0 follows it, both before the
    // second-half compute, while X1 still addresses the old epoch and
    // advances after it. Each appears exactly once (the K loop body is
    // emitted once).
    let at = |needle: &str| {
        assert_eq!(
            lines.iter().filter(|l| l.contains(needle)).count(),
            1,
            "down NT8 stream: one {needle}"
        );
        lines.iter().position(|l| l.contains(needle)).unwrap()
    };
    let (w, x0, x1) = (
        at("s_add_co_u32 s24, s24, 0x88"),
        at("s_add_co_u32 s32, s32, s45"),
        at("s_add_co_u32 s36, s36, s45"),
    );
    assert!(
        w < x0 && x0 < x1,
        "down NT8 stream: split advance W+X0 then X1"
    );
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
