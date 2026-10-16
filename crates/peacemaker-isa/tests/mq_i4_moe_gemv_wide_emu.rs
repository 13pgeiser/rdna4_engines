// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Row-exact wide K1 on the peacemaker CPU interpreter (gfx1151 and gfx1201 instruction tables, real
//! workgroup barriers, LDS, `ds_bpermute_b32`): every output word of the `_c16` / `_c32` twins, launched as
//! one lane-union over `C` token lanes, must equal bit for bit the word of the narrow (C8) entry launched once
//! per token lane (`C` singleton `C = 1` launches, original slot ids `0..10`, X / Y re-based per lane).
//!
//! The data are full-mantissa random F32 activations and random headers / codes, so a reordered chain, a
//! mis-staged X element, a wrong slot, a wrong row of a partial M block or a clobbered neighbour changes bits.
//! Shapes cover K with and without quad-loop and tail groups, M below / at / above eight rows with a ragged last
//! block, unions with one slot per expert up to every token lane on one expert (tiles of 1, 2, 3 and 4 staged
//! slot groups), padded X / Y strides, and `U` capacity tails (`n == 0` descriptors) as the production planner
//! writes them. Y guard words around the outputs and every word a launch must not write are checked against the
//! sentinel. `K1_WIDE_EMU_FULL=1` widens the matrix.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq_i4_moe_gemv::{self, Cap, Codec, Kind, Spec};
use peacemaker_isa::Arch;
use std::collections::BTreeMap;

const SENTINEL: u8 = 0xa5;
const GUARD: usize = 64;
const KERNARG: u64 = 0x1_0000;
const TABLE: u64 = 0x10_0000;
const IDS: u64 = 0x20_0000;
const OFFS: u64 = 0x30_0000;
const SLOTS: u64 = 0x40_0000;
const REF_IDS: u64 = 0x50_0000;
const REF_OFFS: u64 = 0x60_0000;
const REF_SLOTS: u64 = 0x70_0000;
const EXPERTS: u64 = 0x1000_0000;
const EXPERT_STRIDE: u64 = 0x10_0000;
const XBASE: u64 = 0x8000_0000;
const Y0: u64 = 0x9000_0000;
const Y1: u64 = 0xa000_0000;
const R0: u64 = 0xb000_0000;
const R1: u64 = 0xc000_0000;
const N_EXPERTS: usize = 512;

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
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

/// Normal f16 scale with exponent field 8..=20: `-8 * sc` is then exactly the symmetric MQ4 zero.
fn scale_bits(rng: &mut Rng) -> u16 {
    (((8 + rng.below(13)) as u16) << 10) | (rng.next() as u16 & 0x3ff)
}
fn sym_zero_bits(sc: u16) -> u16 {
    0x8000 | (sc + (3 << 10))
}

/// Planar W3 tensor: per row `K*3/8` code bytes (128 3-bit codes per 48-byte group, little-endian bit stream),
/// then the f16 scale plane `2 * (row * K/128 + g)` at `M*K*3/8`.
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

/// Pointer-addressed MQ4: gate G256 groups of 136 B (`[sc0 zp0 sc1 zp1]`, 128 payload bytes at +8), down G128
/// groups of 68 B (`[sc zp]`, 64 payload bytes at +4).
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

/// Full-mantissa activations over a wide exponent band, random signs; the stride padding is a NaN.
fn x_rows(rng: &mut Rng, rows: usize, k: usize, stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * stride * 4);
    for _ in 0..rows {
        for i in 0..stride {
            let bits = if i < k {
                let (s, e, m) = (
                    (rng.next() & 1) as u32,
                    118 + rng.below(19) as u32,
                    (rng.next() & 0x7f_ffff) as u32,
                );
                (s << 31) | (e << 23) | m
            } else {
                0x7fc0_0000
            };
            out.extend(bits.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Routing {
    /// Each lane draws ten distinct experts from a pool of this size.
    Pool(usize),
    /// Every lane routes to the same ten experts (each expert has one slot per lane).
    Identical,
    /// Ten fresh experts per lane while ids last (`10 * C` distinct, capped at the table).
    Disjoint,
}

struct Case {
    c: usize,
    m: usize,
    k: usize,
    routing: Routing,
    /// Extra padding words of the X row stride and the Y slot stride.
    pad: bool,
    seed: u64,
}

fn route(rng: &mut Rng, case: &Case) -> Vec<u32> {
    let mut out = Vec::with_capacity(10 * case.c);
    match case.routing {
        Routing::Pool(n) => {
            let mut pool: Vec<u32> = (0..n).map(|i| ((i * 37 + 11) % N_EXPERTS) as u32).collect();
            pool.sort();
            pool.dedup();
            assert!(pool.len() >= 10);
            for _ in 0..case.c {
                let mut p = pool.clone();
                rng.shuffle(&mut p);
                out.extend_from_slice(&p[..10]);
            }
        }
        Routing::Identical => {
            let mut lane: Vec<u32> = (0..10).map(|i| (i * 53 + 7) as u32).collect();
            rng.shuffle(&mut lane);
            for _ in 0..case.c {
                out.extend_from_slice(&lane);
            }
        }
        Routing::Disjoint => {
            let mut ids: Vec<u32> = (0..N_EXPERTS as u32).collect();
            rng.shuffle(&mut ids);
            for l in 0..case.c {
                out.extend_from_slice(
                    &ids[10 * (l % (N_EXPERTS / 10))..10 * (l % (N_EXPERTS / 10)) + 10],
                );
            }
        }
    }
    out
}

/// One codec bucket: ascending expert ids, original slots per expert in increasing order.
struct Bucket {
    ids: Vec<u32>,
    offs: Vec<u32>,
    slots: Vec<u32>,
}
fn bucket(route: &[u32]) -> Bucket {
    let mut ids: Vec<u32> = route.to_vec();
    ids.sort();
    ids.dedup();
    let mut b = Bucket {
        ids: ids.clone(),
        offs: vec![],
        slots: vec![],
    };
    for id in ids {
        b.offs.push(b.slots.len() as u32);
        for (s, &e) in route.iter().enumerate() {
            if e == id {
                b.slots.push(s as u32);
            }
        }
    }
    b.offs.push(b.slots.len() as u32);
    b
}

fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn kernarg(
    ptrs: [u64; 7],
    m: usize,
    k: usize,
    u: usize,
    c: usize,
    xdiv: usize,
    xsrc: usize,
    xrs: usize,
    ys: usize,
) -> Vec<u8> {
    let mut a = Vec::with_capacity(88);
    for p in ptrs {
        a.extend(p.to_le_bytes());
    }
    for v in [m, k, u, c, xdiv, xsrc, xrs, ys] {
        a.extend((v as u32).to_le_bytes());
    }
    assert_eq!(a.len(), 88);
    a
}

struct Rig {
    arch: Arch,
    program: peacemaker_ir::Program,
}
impl Rig {
    fn new(arch: Arch) -> Self {
        let (_, text, _) = mq_i4_moe_gemv::emit_module(arch).unwrap();
        let elf = peacemaker_isa::native::assemble(&text, arch).unwrap();
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap()
        .program;
        Rig { arch, program }
    }
    fn emu(&self, spec: Spec) -> Emulator<'_> {
        Emulator::new(&self.program, &spec.symbol()).unwrap()
    }
}

fn set_region(mem: &mut Memory, name: &str, bytes: Vec<u8>) {
    mem.regions
        .iter_mut()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("region {name}"))
        .bytes = bytes;
}
fn region<'a>(mem: &'a Memory, name: &str) -> &'a [u8] {
    &mem.regions.iter().find(|r| r.name == name).unwrap().bytes
}

fn launch(emu: &Emulator, mem: &mut Memory, tag: &str, grid: [u32; 3], block: u32) {
    for gy in 0..grid[1] {
        for gx in 0..grid[0] {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx, gy, 0],
                    block: [block, 1, 1],
                    dynamic_lds: 0,
                },
                mem,
            )
            .unwrap_or_else(|e| panic!("{tag} wg({gx},{gy}): {e}"));
        }
    }
}

/// One wide launch over the whole route against `C` narrow singleton launches; returns the number of compared words.
fn run_case(rig: &Rig, codec: Codec, kind: Kind, wide: Cap, case: &Case) -> usize {
    let arch = rig.arch;
    let mut rng = Rng(case.seed | 1);
    let route = route(&mut rng, case);
    let (c, m, k) = (case.c, case.m, case.k);
    let out_rows = if kind == Kind::GateUp { m / 2 } else { m };
    let (xrows, xdiv) = if kind == Kind::GateUp {
        (c, 10)
    } else {
        (10 * c, 1)
    };
    let (xrs, ys) = if case.pad {
        (k + 4, out_rows + 3)
    } else {
        (k, out_rows)
    };
    let slots = 10 * c;

    let mut mem = Memory::default();
    let mut table = vec![0u8; N_EXPERTS * 8];
    let distinct = bucket(&route).ids;
    for &id in &distinct {
        let bytes = match codec {
            Codec::Mq3I4 => w3_tensor(&mut rng, m, k),
            Codec::Mq4Sym => mq4_tensor(&mut rng, kind, m, k),
        };
        let base = EXPERTS + u64::from(id) * EXPERT_STRIDE;
        assert!((bytes.len() as u64) < EXPERT_STRIDE);
        table[id as usize * 8..id as usize * 8 + 8].copy_from_slice(&base.to_le_bytes());
        mem.map(format!("e{id}"), base, bytes).unwrap();
    }
    mem.map("table".into(), TABLE, table).unwrap();
    mem.map("x".into(), XBASE, x_rows(&mut rng, xrows, k, xrs))
        .unwrap();
    let ybytes = slots * ys * 4;
    for (name, base) in [("y0", Y0), ("y1", Y1), ("r0", R0), ("r1", R1)] {
        mem.map(name.into(), base, vec![SENTINEL; ybytes + 2 * GUARD])
            .unwrap();
    }
    // Wide descriptors at the production fixed capacity S = 10*C: live experts first, `n == 0` tail after.
    let b = bucket(&route);
    let cap = slots;
    let mut ids = b.ids.clone();
    ids.resize(cap, 0);
    let mut offs = b.offs.clone();
    offs.resize(cap + 1, *b.offs.last().unwrap());
    let mut slot_words = b.slots.clone();
    slot_words.resize(slots, 0);
    mem.map("ids".into(), IDS, u32s(&ids)).unwrap();
    mem.map("offs".into(), OFFS, u32s(&offs)).unwrap();
    mem.map("slots".into(), SLOTS, u32s(&slot_words)).unwrap();
    mem.map("ref_ids".into(), REF_IDS, vec![0; 4 * 16]).unwrap();
    mem.map("ref_offs".into(), REF_OFFS, vec![0; 4 * 16])
        .unwrap();
    mem.map("ref_slots".into(), REF_SLOTS, vec![0; 4 * 16])
        .unwrap();
    mem.map("kernarg".into(), KERNARG, vec![0; 88]).unwrap();

    let xsrc = if kind == Kind::GateUp { c } else { 10 * c };
    let (y1w, y1r) = if kind == Kind::GateUp {
        (Y1 + GUARD as u64, R1 + GUARD as u64)
    } else {
        (0, 0)
    };

    // The wide launch.
    let wspec = Spec {
        arch,
        codec,
        kind,
        cap: wide,
    };
    set_region(
        &mut mem,
        "kernarg",
        kernarg(
            [TABLE, IDS, OFFS, SLOTS, XBASE, Y0 + GUARD as u64, y1w],
            m,
            k,
            cap,
            c,
            xdiv,
            xsrc,
            xrs,
            ys,
        ),
    );
    launch(
        &rig.emu(wspec),
        &mut mem,
        &wspec.symbol(),
        [m.div_ceil(8) as u32, cap as u32, 1],
        256,
    );

    // C singleton narrow launches: lane `l` with slots 0..10 and X / Y re-based to the lane.
    let nspec = Spec::narrow(arch, codec, kind);
    let nemu = rig.emu(nspec);
    for l in 0..c {
        let lane = &route[10 * l..10 * l + 10];
        let lb = bucket(lane);
        set_region(&mut mem, "ref_ids", u32s(&lb.ids));
        set_region(&mut mem, "ref_offs", u32s(&lb.offs));
        set_region(&mut mem, "ref_slots", u32s(&lb.slots));
        let (xoff, xsrc1) = if kind == Kind::GateUp {
            (l * xrs * 4, 1)
        } else {
            (l * 10 * xrs * 4, 10)
        };
        let yoff = l * 10 * ys * 4;
        let y1l = if kind == Kind::GateUp {
            R1 + GUARD as u64 + yoff as u64
        } else {
            0
        };
        set_region(
            &mut mem,
            "kernarg",
            kernarg(
                [
                    TABLE,
                    REF_IDS,
                    REF_OFFS,
                    REF_SLOTS,
                    XBASE + xoff as u64,
                    R0 + GUARD as u64 + yoff as u64,
                    y1l,
                ],
                m,
                k,
                lb.ids.len(),
                1,
                xdiv,
                xsrc1,
                xrs,
                ys,
            ),
        );
        launch(
            &nemu,
            &mut mem,
            &format!("{} lane {l}", nspec.symbol()),
            [m as u32, lb.ids.len() as u32, 1],
            32,
        );
    }

    // Compare every word, and the guards.
    let mut compared = 0;
    let parts: &[(&str, &str)] = if kind == Kind::GateUp {
        &[("y0", "r0"), ("y1", "r1")]
    } else {
        &[("y0", "r0")]
    };
    for (wide_name, ref_name) in parts {
        let (w, r) = (region(&mem, wide_name), region(&mem, ref_name));
        assert!(
            w[..GUARD]
                .iter()
                .chain(&w[GUARD + ybytes..])
                .all(|&b| b == SENTINEL),
            "{} {wide_name}: guard written",
            wspec.symbol()
        );
        for (i, (a, b)) in w[GUARD..GUARD + ybytes]
            .chunks(4)
            .zip(r[GUARD..GUARD + ybytes].chunks(4))
            .enumerate()
        {
            let (a, b) = (
                u32::from_le_bytes(a.try_into().unwrap()),
                u32::from_le_bytes(b.try_into().unwrap()),
            );
            let (slot, row) = (i / ys, i % ys);
            assert_eq!(a, b, "{arch:?} {} C={c} M={m} K={k} {wide_name}[slot {slot}][row {row}]: wide {a:#010x} vs singleton {b:#010x}", wspec.symbol());
            if row < out_rows {
                assert_ne!(
                    b,
                    u32::from_le_bytes([SENTINEL; 4]),
                    "{} slot {slot} row {row} not written by the singleton",
                    wspec.symbol()
                );
            }
            compared += 1;
        }
    }
    compared
}

fn k_set(codec: Codec, kind: Kind) -> [usize; 3] {
    match (codec, kind) {
        (Codec::Mq4Sym, Kind::GateUp) => [256, 1280, 2816],
        _ => [128, 640, 1408],
    }
}

fn matrix(full: bool) -> Vec<Case> {
    let mut v = vec![
        // n = 1 everywhere (one tile, one live producer wave); M below eight rows.
        Case {
            c: 1,
            m: 6,
            k: 0,
            routing: Routing::Identical,
            pad: false,
            seed: 1,
        },
        // C9: a second tile with one slot; pool unions of mixed n.
        Case {
            c: 9,
            m: 12,
            k: 0,
            routing: Routing::Identical,
            pad: true,
            seed: 2,
        },
        Case {
            c: 9,
            m: 10,
            k: 0,
            routing: Routing::Pool(14),
            pad: false,
            seed: 3,
        },
        // C16: every token on every expert (n = 16 = two full tiles).
        Case {
            c: 16,
            m: 10,
            k: 0,
            routing: Routing::Identical,
            pad: true,
            seed: 4,
        },
        // C17 / C31 / C32: three and four tiles, ragged n.
        Case {
            c: 17,
            m: 8,
            k: 0,
            routing: Routing::Pool(12),
            pad: false,
            seed: 5,
        },
        Case {
            c: 32,
            m: 6,
            k: 0,
            routing: Routing::Identical,
            pad: false,
            seed: 6,
        },
        Case {
            c: 31,
            m: 12,
            k: 0,
            routing: Routing::Pool(16),
            pad: true,
            seed: 7,
        },
    ];
    if full {
        v.extend([
            Case {
                c: 5,
                m: 20,
                k: 0,
                routing: Routing::Disjoint,
                pad: false,
                seed: 8,
            },
            Case {
                c: 9,
                m: 16,
                k: 0,
                routing: Routing::Disjoint,
                pad: true,
                seed: 9,
            },
            Case {
                c: 16,
                m: 24,
                k: 0,
                routing: Routing::Pool(40),
                pad: false,
                seed: 10,
            },
            Case {
                c: 32,
                m: 20,
                k: 0,
                routing: Routing::Pool(24),
                pad: true,
                seed: 11,
            },
            Case {
                c: 24,
                m: 10,
                k: 0,
                routing: Routing::Identical,
                pad: false,
                seed: 12,
            },
        ]);
    }
    v
}

fn run_arch(arch: Arch) {
    let full = std::env::var("K1_WIDE_EMU_FULL").is_ok_and(|v| v == "1");
    let rig = Rig::new(arch);
    let mut compared: BTreeMap<String, usize> = BTreeMap::new();
    for codec in Codec::ALL {
        for kind in Kind::ALL {
            let ks = k_set(codec, kind);
            for (ci, case) in matrix(full).into_iter().enumerate() {
                // K rotates through the three shapes: one group, quad + one tail, two quads + three tails.
                let k = if full {
                    ks[ci % 3]
                } else {
                    ks[(ci + (codec as usize) + (kind as usize)) % 3]
                };
                let case = Case { k, ..case };
                for wide in [Cap::C16, Cap::C32] {
                    if wide == Cap::C16 && case.c > 16 {
                        continue;
                    }
                    let n = run_case(&rig, codec, kind, wide, &case);
                    *compared
                        .entry(
                            Spec {
                                arch,
                                codec,
                                kind,
                                cap: wide,
                            }
                            .symbol(),
                        )
                        .or_default() += n;
                }
            }
        }
    }
    // Every wide symbol compared a non-trivial number of words.
    assert_eq!(compared.len(), 8, "{arch:?}: {compared:?}");
    assert!(
        compared.values().all(|&n| n > 1000),
        "{arch:?}: {compared:?}"
    );
}

#[test]
fn wide_k1_matches_c_singleton_narrow_launches_on_gfx1151() {
    run_arch(Arch::Gfx1151)
}

#[test]
fn wide_k1_matches_c_singleton_narrow_launches_on_gfx1201() {
    run_arch(Arch::Gfx1201)
}
