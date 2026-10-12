// SPDX-License-Identifier: Apache-2.0
//! G80 I32 programs in the whole-array simulator, fed with panels from OUR pack pipeline (`pack_shape`).
//!
//! * `sim_*_all_families_k128`: for each shape, EVERY panel (GdnIn 13, QsaIn 11 — each family boundary falls inside a panel
//!   or on one) is submitted as one `M = 256` (one wave), `K = 128` (two K chunks: the smallest K that still crosses a chunk
//!   boundary) G80 `Epilogue::I32` / `Control::Fast` program on ONE reused `Config`. The B argument is the pack_shape K = 2560
//!   panel cut down to its first two K chunks per column segment (chunk-major layout, see `truncate_panel`), A is the
//!   canonical `pack_in` of random a8. All 256 x 1280 int32 outputs of every panel are compared with an independent CPU dot
//!   product, scattered into each family's own output through `ColumnMap::locate` and compared with the plain
//!   `a8 . w8(family)^T` of that family. Padded fused columns must come back exactly 0.
//! * `sim_masked_programs_k2560`: the REAL program of `build_programs_for_families` (`K = 2560`, one wave) with the small
//!   masks GdnA+GdnB (GdnIn) and QsaIdx (QsaIn): the pack_shape panel is submitted as is, full-K CPU dot.
//! * `pack_shape_panels_equal_canonical_b_*`: pack_shape panels (K = 2560, ALL and skip-a-family masks, family
//!   compaction across panel boundaries) equal `design_g80(..).pack_in` of a B matrix built here from `pack_matrix` w8 with
//!   the fused order spelled out independently; no simulator.
//!
//! Reference w8 is `pack_matrix` (its own oracle lives in tests/packing.rs); packing the panel and placing columns is what is
//! being proven. Functional ticks are simulator ticks, never silicon cycles. Budget: a fixed number of submits of one wave,
//! `tick_limit` as in the pm-npu harness (`Config` default, per wait, so a stall is an `Err`, not a hang).
use hipfire_npu_trunk::{
    build_programs_for_families, pack_matrix, pack_shape, EpilogueContract, Family, FamilyMask, MatrixRef, PackedB, Shape,
    PANEL_COLS, TRUNK_K,
};
use pm_npu::kernels::gemm_array::ArrayDesign;
use pm_npu::kernels::gemm_core::{Control, Epilogue};
use pm_npu::kernels::gemm_g80::design_g80;
use pm_npu::sim::config::Config;

const WAVE_M: usize = 256;
/// K of the all-families simulator runs: two 64-wide chunks.
const K_SIM: usize = 128;
const K_CHUNK: usize = 64;
/// Bytes of one K chunk of one core column's B segment (two 80-wide N tiles x 64 K).
const B_CHUNK: usize = 10240;
const B_COLS: usize = 8;
const POISON: u8 = 0xa5;
/// Same value as `Config::new()` (the pm-npu harness never changes it); written out so the budget is explicit.
const TICK_LIMIT: usize = 10_000_000;
const MQ6_GROUP_BYTES: usize = 200;

// ---------------------------------------------------------------- deterministic data
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random MQ6G256V2 matrix `[m][2560]`: per 256-group `[f16 s0, f16 z0, f16 s1, f16 z1, 192 B random payload]`. Rows 3 and
/// m-1 have all-zero scales and zero points (absmax 0: ws == 0 and all codes 0); row 5 has huge-amplitude scales next to tiny
/// ones so its codes saturate at +-127.
fn random_mq6(m: usize, seed: u64) -> Vec<u8> {
    let groups = TRUNK_K / 256;
    let mut rng = Rng::new(seed);
    let mut out = vec![0u8; m * groups * MQ6_GROUP_BYTES];
    for row in 0..m {
        for g in 0..groups {
            let at = (row * groups + g) * MQ6_GROUP_BYTES;
            let blank = row == 3 || row + 1 == m;
            for h in 0..2 {
                let (s, z) = if blank {
                    (0.0, 0.0)
                } else if row == 5 && h == 1 {
                    (0.001, 0.0)
                } else {
                    let r = rng.next();
                    let s = ((r & 0xff) as f32 + 1.0) / 2048.0 * if row == 5 { 64.0 } else { 1.0 };
                    let z = (((r >> 8) & 0xff) as f32 - 128.0) / 256.0;
                    (s, z)
                };
                out[at + 4 * h..at + 4 * h + 2].copy_from_slice(&half::f16::from_f32(s).to_bits().to_le_bytes());
                out[at + 4 * h + 2..at + 4 * h + 4].copy_from_slice(&half::f16::from_f32(z).to_bits().to_le_bytes());
            }
            for b in out[at + 8..at + MQ6_GROUP_BYTES].chunks_mut(8) {
                let r = rng.next().to_le_bytes();
                b.copy_from_slice(&r[..b.len()]);
            }
        }
    }
    out
}

struct Fam {
    family: Family,
    mq6: Vec<u8>,
    /// `[family.cols()][keep_k]` row-major (first `keep_k` of the 2560 w8 codes of each output row).
    w8: Vec<i8>,
    ws: Vec<f32>,
    keep_k: usize,
}
impl Fam {
    fn row(&self, c: usize) -> &[i8] {
        &self.w8[c * self.keep_k..(c + 1) * self.keep_k]
    }
}

/// The selected families of `shape` (fused order), their MQ6 blobs and the reference w8 (first `keep_k` codes per row).
fn fixture(shape: Shape, which: FamilyMask, keep_k: usize) -> Vec<Fam> {
    shape
        .families()
        .iter()
        .filter(|&&f| which.contains(f))
        .map(|&family| {
            let m = family.cols();
            let mq6 = random_mq6(m, 0x5eed_0000 + family.index() as u64);
            let (full, ws) = pack_matrix(&mq6, m, TRUNK_K);
            assert_eq!((full.len(), ws.len()), (m * TRUNK_K, m));
            let mut w8 = Vec::with_capacity(m * keep_k);
            for c in 0..m {
                w8.extend_from_slice(&full[c * TRUNK_K..c * TRUNK_K + keep_k]);
            }
            Fam { family, mq6, w8, ws, keep_k }
        })
        .collect()
}

fn refs(fams: &[Fam]) -> Vec<MatrixRef<'_>> {
    fams.iter().map(|f| MatrixRef { family: f.family, mq6: &f.mq6, m: f.family.cols(), k: TRUNK_K }).collect()
}

/// a8 `[WAVE_M][k]` in -127..=127 with edge rows: all +127, all -127, all 0, alternating +-127.
fn random_a8(k: usize, seed: u64) -> Vec<i8> {
    let mut rng = Rng::new(seed);
    let mut a: Vec<i8> = (0..WAVE_M * k).map(|_| (((rng.next() >> 33) % 255) as i32 - 127) as i8).collect();
    for kk in 0..k {
        a[kk] = 127;
        a[k + kk] = -127;
        a[2 * k + kk] = 0;
        a[3 * k + kk] = if kk % 2 == 0 { 127 } else { -127 };
    }
    a
}

fn dot(a: &[i8], w: &[i8]) -> i32 {
    a.iter().zip(w).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Fused layout of the selected families, spelled out here: `(family, first fused col, cols)` in fused order, compacted.
fn expected_parts(shape: Shape, mask: FamilyMask) -> Vec<(Family, usize, usize)> {
    let mut at = 0;
    let mut parts = Vec::new();
    for &f in shape.families() {
        if mask.contains(f) {
            parts.push((f, at, f.cols()));
            at += f.cols();
        }
    }
    parts
}

fn assert_map(shape: Shape, mask: FamilyMask) -> hipfire_npu_trunk::ColumnMap {
    let map = shape.column_map(mask);
    let got: Vec<_> = map.parts.iter().map(|&(f, c0, n)| (f, c0 as usize, n as usize)).collect();
    assert_eq!(got, expected_parts(shape, mask), "{shape:?} column map");
    let useful: usize = got.iter().map(|p| p.2).sum();
    assert_eq!((map.useful_cols(), map.panel_count(), map.padded_cols()), (useful, useful.div_ceil(PANEL_COLS), useful.div_ceil(PANEL_COLS) * PANEL_COLS));
    map
}

/// B `[k][1280]` of one fused panel: column `j` is the w8 row of the family column at fused col `panel*1280+j` (0 in padding).
fn fused_b(shape: Shape, mask: FamilyMask, fams: &[Fam], panel: usize, k: usize) -> Vec<i8> {
    let mut b = vec![0i8; k * PANEL_COLS];
    for (f, col0, cols) in expected_parts(shape, mask) {
        let fam = fams.iter().find(|x| x.family == f).expect("fixture family");
        for c in 0..cols {
            let fused = col0 + c;
            if fused / PANEL_COLS != panel {
                continue;
            }
            let j = fused % PANEL_COLS;
            for (kk, &w) in fam.row(c)[..k].iter().enumerate() {
                b[kk * PANEL_COLS + j] = w;
            }
        }
    }
    b
}

fn pack(shape: Shape, fams: &[Fam], mask: FamilyMask) -> PackedB {
    pack_shape(shape, &refs(fams), mask).unwrap_or_else(|e| panic!("{shape:?} pack_shape: {e}"))
}

/// First `kc` K chunks of every column segment of a K = 2560 panel: the G80 B stream is `[col 0..8][chunk][even, odd tile]`
/// with `B_CHUNK` bytes per (column, chunk), so a K-prefix is a per-segment prefix. Equality with the canonical `pack_in` of the
/// K-prefix matrix is asserted by every caller.
fn truncate_panel(panel: &[u8], kc: usize) -> Vec<u8> {
    let kc_full = TRUNK_K / K_CHUNK;
    assert_eq!(panel.len(), B_COLS * kc_full * B_CHUNK);
    let mut out = Vec::with_capacity(B_COLS * kc * B_CHUNK);
    for col in 0..B_COLS {
        out.extend_from_slice(&panel[col * kc_full * B_CHUNK..(col * kc_full + kc) * B_CHUNK]);
    }
    out
}

fn g80(rows: usize, k: usize) -> ArrayDesign {
    design_g80(rows, PANEL_COLS, k, Epilogue::I32, Control::Fast)
}

// ---------------------------------------------------------------- one checked submit
/// One submit on a (possibly reused) `Config`: returns the row-major `WAVE_M x 1280` int32 C. Every word of the C arena must
/// have been overwritten (poison never survives) and be bounded by `k * 127 * 127`; the packed inputs must be unchanged.
fn submit(sim: &mut Config, d: &ArrayDesign, a_host: &[u8], b_host: &[u8], k: usize, what: &str) -> Vec<i32> {
    assert_eq!((a_host.len(), b_host.len()), (d.args[0].bytes, d.args[1].bytes), "{what}: packed argument sizes");
    let mut args = vec![a_host.to_vec(), b_host.to_vec(), vec![POISON; d.args[2].bytes]];
    let ticks0 = sim.ticks;
    sim.submit(&d.insts, &mut args).unwrap_or_else(|e| panic!("{what}: submit failed: {e}"));
    let poison = i32::from_le_bytes([POISON; 4]) as i64;
    let bound = (k * 127 * 127) as i64;
    assert!(poison.abs() > bound, "{what}: poison must lie outside the accumulator bound");
    for (i, w) in args[2].chunks_exact(4).enumerate() {
        let v = i32::from_le_bytes(w.try_into().unwrap()) as i64;
        assert!(v != poison, "{what}: C arena word {i} never written");
        assert!(v.abs() <= bound, "{what}: C arena word {i} = {v} exceeds {bound}");
    }
    assert!(args[0] == a_host && args[1] == b_host, "{what}: packed inputs changed");
    eprintln!("PASS {what}: {} functional ticks (simulator ticks, not hardware cycles)", sim.ticks - ticks0);
    d.unpack_out(&args[2])
}

fn fresh(d: &ArrayDesign) -> Config {
    let mut sim = Config::from_pdi(&d.pdi).expect("PDI loads");
    sim.tick_limit = TICK_LIMIT;
    sim
}

// ---------------------------------------------------------------- all families, every panel, K = 128
fn sim_all_families(shape: Shape) {
    let mask = FamilyMask::ALL;
    let fams = fixture(shape, mask, K_SIM);
    let packed = pack(shape, &fams, mask);
    let map = assert_map(shape, mask);
    assert_eq!(packed.panels.len(), map.panel_count());
    assert_eq!(packed.map.parts, map.parts, "PackedB carries the column map");

    let d = g80(WAVE_M, K_SIM);
    let a8 = random_a8(K_SIM, 0xa8a8 + shape.index() as u64);
    let a_host = d.pack_in(&a8, &vec![0i8; K_SIM * PANEL_COLS])[0].clone();
    let mut sim = fresh(&d);

    // Per-family output tensors filled ONLY through ColumnMap::locate.
    let mut out: Vec<Vec<i32>> = fams.iter().map(|f| vec![i32::MIN; WAVE_M * f.family.cols()]).collect();
    let mut written = vec![0usize; fams.len()];
    for p in 0..packed.panels.len() {
        let b_host = truncate_panel(&packed.panels[p], K_SIM / K_CHUNK);
        let canonical = d.pack_in(&a8, &fused_b(shape, mask, &fams, p, K_SIM))[1].clone();
        assert!(b_host == canonical, "{shape:?} panel {p}: truncated pack_shape panel != canonical pack_in of the K-prefix B");
        let c = submit(&mut sim, &d, &a_host, &b_host, K_SIM, &format!("{shape:?} panel {p}/{} K{K_SIM}", packed.panels.len()));
        assert_eq!(c.len(), WAVE_M * PANEL_COLS);
        for j in 0..PANEL_COLS {
            match map.locate(p * PANEL_COLS + j) {
                Some((family, col)) => {
                    let fi = fams.iter().position(|f| f.family == family).expect("family in fixture");
                    for row in 0..WAVE_M {
                        out[fi][row * family.cols() + col] = c[row * PANEL_COLS + j];
                    }
                    written[fi] += 1;
                }
                None => {
                    assert!((0..WAVE_M).all(|row| c[row * PANEL_COLS + j] == 0), "{shape:?} panel {p}: padded column {j} must be 0");
                }
            }
        }
    }
    for (fi, fam) in fams.iter().enumerate() {
        assert_eq!(written[fi], fam.family.cols(), "{:?}: every column scattered exactly once", fam.family);
        for c in 0..fam.family.cols() {
            let w = fam.row(c);
            for row in 0..WAVE_M {
                let want = dot(&a8[row * K_SIM..(row + 1) * K_SIM], w);
                let got = out[fi][row * fam.family.cols() + c];
                assert_eq!(got, want, "{:?} row {row} col {c}: sim {got} != cpu {want}", fam.family);
            }
        }
        eprintln!("PASS {:?}: {} x {} int32 outputs equal the CPU dot through ColumnMap", fam.family, WAVE_M, fam.family.cols());
    }
}

#[test]
fn sim_gdn_in_all_families_k128() {
    sim_all_families(Shape::GdnIn);
}

#[test]
fn sim_qsa_in_all_families_k128() {
    sim_all_families(Shape::QsaIn);
}

// ---------------------------------------------------------------- the real K = 2560 program, small masks
#[test]
fn sim_masked_programs_k2560() {
    use Family::*;
    let mask = FamilyMask::from(GdnA) | FamilyMask::from(GdnB) | FamilyMask::from(QsaIdx);
    let progs = build_programs_for_families(EpilogueContract::I32Fold, WAVE_M, mask).expect("programs");
    assert_eq!(progs.rows_max, WAVE_M);
    for shape in Shape::ALL {
        let prog = progs.get(shape);
        assert_eq!(prog.panel_count, 1, "{shape:?}");
        let fams = fixture(shape, mask, TRUNK_K);
        let packed = pack(shape, &fams, mask);
        let map = assert_map(shape, mask);
        assert_eq!(packed.panels.len(), 1);
        assert_eq!(packed.panels[0].len(), prog.arg_bytes[1], "{shape:?}: panel bytes == program B argument");
        assert_eq!(packed.ws.len(), prog.padded_cols);

        let a8 = random_a8(TRUNK_K, 0x2560 + shape.index() as u64);
        let [a_host, b_ref] = prog.design.pack_in(&a8, &fused_b(shape, mask, &fams, 0, TRUNK_K));
        assert_eq!(a_host.len(), prog.arg_bytes[0]);
        assert!(packed.panels[0] == b_ref, "{shape:?}: pack_shape panel != canonical pack_in of the independent fused B");

        let mut sim = fresh(&prog.design);
        let c = submit(&mut sim, &prog.design, &a_host, &packed.panels[0], TRUNK_K, &format!("{shape:?} program K{TRUNK_K}"));
        assert_eq!(c.len(), WAVE_M * PANEL_COLS);
        let useful = map.useful_cols();
        for j in 0..PANEL_COLS {
            match map.locate(j) {
                Some((family, col)) => {
                    let fam = fams.iter().find(|f| f.family == family).unwrap();
                    let w = fam.row(col);
                    for row in 0..WAVE_M {
                        let want = dot(&a8[row * TRUNK_K..(row + 1) * TRUNK_K], w);
                        assert_eq!(c[row * PANEL_COLS + j], want, "{shape:?} {family:?} row {row} col {col}");
                    }
                }
                None => {
                    assert!(j >= useful);
                    assert!((0..WAVE_M).all(|row| c[row * PANEL_COLS + j] == 0), "{shape:?}: padded column {j} must be 0");
                }
            }
        }
        eprintln!("PASS {shape:?}: {} families, {useful} useful columns x {WAVE_M} rows equal the K2560 CPU dot", map.parts.len());
    }
}

// ---------------------------------------------------------------- canonical B crosscheck (no simulator)
fn crosscheck(shape: Shape, mask: FamilyMask) {
    let fams = fixture(shape, mask, TRUNK_K);
    let packed = pack(shape, &fams, mask);
    let map = assert_map(shape, mask);
    assert_eq!(packed.panels.len(), map.panel_count(), "{shape:?} {mask:?}");
    assert_eq!(packed.ws.len(), map.padded_cols());
    let d = g80(1, TRUNK_K);
    let zero_a = vec![0i8; TRUNK_K];
    for p in 0..packed.panels.len() {
        let canonical = d.pack_in(&zero_a, &fused_b(shape, mask, &fams, p, TRUNK_K))[1].clone();
        assert_eq!(packed.panels[p].len(), canonical.len());
        assert!(packed.panels[p] == canonical, "{shape:?} {mask:?} panel {p}: pack_shape != canonical G80 B");
    }
    for (family, col0, cols) in expected_parts(shape, mask) {
        let fam = fams.iter().find(|f| f.family == family).unwrap();
        for c in 0..cols {
            assert_eq!(packed.ws[col0 + c].to_bits(), fam.ws[c].to_bits(), "{family:?} ws {c}");
        }
    }
    let useful = map.useful_cols();
    assert!(packed.ws[useful..].iter().all(|&w| w == 0.0), "padded ws is 0");
    eprintln!("PASS {shape:?} {mask:?}: {} panels equal the canonical pack_in B", packed.panels.len());
}

fn mask_of(families: &[Family]) -> FamilyMask {
    families.iter().fold(FamilyMask::NONE, |m, &f| m | FamilyMask::from(f))
}

#[test]
fn pack_shape_panels_equal_canonical_b_gdn_in() {
    use Family::*;
    crosscheck(Shape::GdnIn, FamilyMask::GDN);
    crosscheck(Shape::GdnIn, mask_of(&[GdnA, GdnB]));
    // Compaction: Z dropped, a/b land right after qkv and straddle the 8th/9th panel boundary region.
    crosscheck(Shape::GdnIn, mask_of(&[GdnQkv, GdnA, GdnB]));
    crosscheck(Shape::GdnIn, mask_of(&[GdnZ, GdnA]));
}

#[test]
fn pack_shape_panels_equal_canonical_b_qsa_in() {
    use Family::*;
    crosscheck(Shape::QsaIn, FamilyMask::QSA);
    crosscheck(Shape::QsaIn, mask_of(&[QsaIdx]));
    crosscheck(Shape::QsaIn, mask_of(&[QsaK, QsaV, QsaIdx]));
    crosscheck(Shape::QsaIn, mask_of(&[QsaQ, QsaIdx]));
}
