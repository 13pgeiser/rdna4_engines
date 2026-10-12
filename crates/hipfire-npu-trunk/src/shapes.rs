// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused trunk shapes: which projections share an input, how their output columns are
//! concatenated, and the panel arithmetic of the resulting `B` matrix.
//!
//! A *shape* is a set of projections that read the same activation, so one NPU GEMM
//! computes all of them: `GdnIn = qkv|z|a|b` and `QsaIn = q|k|v|idx`. A [`FamilyMask`]
//! selects the projections that run on the NPU; the selected families' columns are
//! concatenated contiguously in fused order (no per-family padding), and only the whole
//! fused width is padded up to a multiple of [`PANEL_COLS`].

use std::fmt;
use std::ops::BitOr;

/// Reduction dimension of every trunk projection.
pub const TRUNK_K: usize = 2560;
/// Output columns of one NPU panel (the G80 wave width).
pub const PANEL_COLS: usize = 1280;

/// Epilogue contract of the NPU path (see INTERFACES.md §0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EpilogueContract {
    /// Raw I32 from the NPU, folded on the GPU: `y = (f32)(((f64)I * (f64)ws_n) * (f64)d_t)`.
    #[default]
    I32Fold,
    /// BF16 straight from the NPU. Defined only after probe B0; until then a typed placeholder.
    Bf16Direct,
}

impl EpilogueContract {
    /// Exact error of the unsupported contract.
    pub const BF16_DIRECT_PENDING: &'static str = "Bf16Direct pending probe B0";

    /// `Ok` for the contract this build implements, the pending-probe error otherwise.
    pub fn ensure_supported(self) -> Result<(), String> {
        match self {
            Self::I32Fold => Ok(()),
            Self::Bf16Direct => Err(Self::BF16_DIRECT_PENDING.to_string()),
        }
    }

    /// Spelling used in load logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::I32Fold => "i32fold",
            Self::Bf16Direct => "bf16direct",
        }
    }
}

impl fmt::Display for EpilogueContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One trunk projection that can be offloaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    QsaQ,
    QsaK,
    QsaV,
    QsaIdx,
}

impl Family {
    /// All families in fused order (GDN `qkv|z|a|b`, then QSA `q|k|v|idx`).
    pub const ALL: [Family; 8] = [
        Family::GdnQkv,
        Family::GdnZ,
        Family::GdnA,
        Family::GdnB,
        Family::QsaQ,
        Family::QsaK,
        Family::QsaV,
        Family::QsaIdx,
    ];

    /// Output features (matrix rows `M`) of the projection.
    pub const fn cols(self) -> usize {
        match self {
            Family::GdnQkv => 10240,
            Family::GdnZ => 6144,
            Family::GdnA => 48,
            Family::GdnB => 48,
            Family::QsaQ => 12288,
            Family::QsaK => 512,
            Family::QsaV => 512,
            Family::QsaIdx => 640,
        }
    }

    /// Position in [`Family::ALL`]; also the bit of the family in a [`FamilyMask`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The shape the family belongs to.
    pub const fn shape(self) -> Shape {
        match self {
            Family::GdnQkv | Family::GdnZ | Family::GdnA | Family::GdnB => Shape::GdnIn,
            _ => Shape::QsaIn,
        }
    }

    /// Short name used in errors and logs.
    pub const fn name(self) -> &'static str {
        match self {
            Family::GdnQkv => "gdn_qkv",
            Family::GdnZ => "gdn_z",
            Family::GdnA => "gdn_a",
            Family::GdnB => "gdn_b",
            Family::QsaQ => "qsa_q",
            Family::QsaK => "qsa_k",
            Family::QsaV => "qsa_v",
            Family::QsaIdx => "qsa_idx",
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Set of families that run on the NPU (bit `Family::index()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FamilyMask(u8);

impl FamilyMask {
    pub const NONE: FamilyMask = FamilyMask(0);
    pub const ALL: FamilyMask = FamilyMask(0xFF);
    /// The four GDN families.
    pub const GDN: FamilyMask = FamilyMask(0x0F);
    /// The four QSA families.
    pub const QSA: FamilyMask = FamilyMask(0xF0);

    pub const fn from_bits(bits: u8) -> Self {
        FamilyMask(bits)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, family: Family) -> bool {
        self.0 & (1 << family.index()) != 0
    }

    /// This mask plus `family`.
    pub const fn with(self, family: Family) -> Self {
        FamilyMask(self.0 | (1 << family.index()))
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for FamilyMask {
    type Output = FamilyMask;
    fn bitor(self, rhs: FamilyMask) -> FamilyMask {
        FamilyMask(self.0 | rhs.0)
    }
}

impl From<Family> for FamilyMask {
    fn from(family: Family) -> Self {
        FamilyMask(1 << family.index())
    }
}

impl BitOr<Family> for FamilyMask {
    type Output = FamilyMask;
    fn bitor(self, rhs: Family) -> FamilyMask {
        self.with(rhs)
    }
}

impl BitOr for Family {
    type Output = FamilyMask;
    fn bitor(self, rhs: Family) -> FamilyMask {
        FamilyMask::from(self).with(rhs)
    }
}

/// Fused column map: for each selected family, `(family, first fused column, columns)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnMap {
    pub parts: Vec<(Family, u32, u32)>,
}

impl ColumnMap {
    /// Live columns (sum of the selected families' columns).
    pub fn useful_cols(&self) -> usize {
        self.parts.iter().map(|&(_, _, cols)| cols as usize).sum()
    }

    /// Live columns rounded up to whole panels; `0` when nothing is selected.
    pub fn padded_cols(&self) -> usize {
        self.panel_count() * PANEL_COLS
    }

    /// Number of [`PANEL_COLS`]-wide panels.
    pub fn panel_count(&self) -> usize {
        self.useful_cols().div_ceil(PANEL_COLS)
    }

    /// Family and family-local column of a fused column; `None` in the padding or out of range.
    pub fn locate(&self, fused_col: usize) -> Option<(Family, usize)> {
        self.parts.iter().find_map(|&(family, col0, cols)| {
            let (col0, cols) = (col0 as usize, cols as usize);
            (fused_col >= col0 && fused_col < col0 + cols).then(|| (family, fused_col - col0))
        })
    }
}

/// A fused trunk input shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// GDN input projections `qkv|z|a|b` (16480 columns).
    GdnIn,
    /// QSA input projections `q|k|v|idx` (13952 columns).
    QsaIn,
}

impl Shape {
    pub const ALL: [Shape; 2] = [Shape::GdnIn, Shape::QsaIn];

    /// Position in [`Shape::ALL`]; indexes `PackedTrunk::map`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The shape's families in fused order.
    pub const fn families(self) -> &'static [Family] {
        match self {
            Shape::GdnIn => &[Family::GdnQkv, Family::GdnZ, Family::GdnA, Family::GdnB],
            Shape::QsaIn => &[Family::QsaQ, Family::QsaK, Family::QsaV, Family::QsaIdx],
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Shape::GdnIn => "gdn_in",
            Shape::QsaIn => "qsa_in",
        }
    }

    /// Fused column map of the families of `mask` that belong to this shape, concatenated in
    /// fused order without per-family padding.
    pub fn column_map(self, mask: FamilyMask) -> ColumnMap {
        let mut parts = Vec::new();
        let mut col0 = 0u32;
        for &family in self.families() {
            if mask.contains(family) {
                let cols = family.cols() as u32;
                parts.push((family, col0, cols));
                col0 += cols;
            }
        }
        ColumnMap { parts }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_table_and_order() {
        let names: Vec<_> = Family::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(
            names,
            ["gdn_qkv", "gdn_z", "gdn_a", "gdn_b", "qsa_q", "qsa_k", "qsa_v", "qsa_idx"]
        );
        for (i, f) in Family::ALL.iter().enumerate() {
            assert_eq!(f.index(), i);
        }
        let gdn: usize = Shape::GdnIn.families().iter().map(|f| f.cols()).sum();
        let qsa: usize = Shape::QsaIn.families().iter().map(|f| f.cols()).sum();
        assert_eq!((gdn, qsa), (16480, 13952));
        for shape in Shape::ALL {
            assert_eq!(Shape::ALL[shape.index()], shape);
            for f in shape.families() {
                assert_eq!(f.shape(), shape);
            }
        }
    }

    #[test]
    fn mask_bits_and_ops() {
        assert_eq!(FamilyMask::default(), FamilyMask::NONE);
        assert!(FamilyMask::NONE.is_empty());
        assert_eq!(FamilyMask::GDN | FamilyMask::QSA, FamilyMask::ALL);
        assert_eq!(FamilyMask::ALL.bits(), 0xFF);
        for f in Family::ALL {
            assert!(FamilyMask::ALL.contains(f));
            assert!(!FamilyMask::NONE.contains(f));
            assert_eq!(FamilyMask::GDN.contains(f), f.shape() == Shape::GdnIn);
            assert_eq!(FamilyMask::QSA.contains(f), f.shape() == Shape::QsaIn);
            let m = FamilyMask::from(f);
            assert_eq!(m.bits().count_ones(), 1);
            assert!(m.contains(f));
        }
        let m = FamilyMask::from(Family::GdnZ).with(Family::QsaIdx);
        assert_eq!(m, FamilyMask::from_bits(0b1000_0010));
        assert_eq!(m, Family::GdnZ | Family::QsaIdx);
        assert!(m.contains(Family::GdnZ) && m.contains(Family::QsaIdx));
        assert!(!m.contains(Family::GdnA));
    }

    #[test]
    fn full_shapes_match_design_arithmetic() {
        let g = Shape::GdnIn.column_map(FamilyMask::ALL);
        assert_eq!(g.useful_cols(), 16480);
        assert_eq!(g.panel_count(), 13);
        assert_eq!(g.padded_cols(), 16640);
        assert_eq!(
            g.parts,
            [
                (Family::GdnQkv, 0, 10240),
                (Family::GdnZ, 10240, 6144),
                (Family::GdnA, 16384, 48),
                (Family::GdnB, 16432, 48),
            ]
        );
        let q = Shape::QsaIn.column_map(FamilyMask::ALL);
        assert_eq!(q.useful_cols(), 13952);
        assert_eq!(q.panel_count(), 11);
        assert_eq!(q.padded_cols(), 14080);
        assert_eq!(
            q.parts,
            [
                (Family::QsaQ, 0, 12288),
                (Family::QsaK, 12288, 512),
                (Family::QsaV, 12800, 512),
                (Family::QsaIdx, 13312, 640),
            ]
        );
    }

    #[test]
    fn masked_maps_are_contiguous_and_shape_local() {
        // The other shape's bits never leak in.
        assert_eq!(Shape::GdnIn.column_map(FamilyMask::QSA), ColumnMap::default());
        let m = Family::GdnZ | Family::GdnB | Family::QsaQ;
        let g = Shape::GdnIn.column_map(m);
        assert_eq!(g.parts, [(Family::GdnZ, 0, 6144), (Family::GdnB, 6144, 48)]);
        assert_eq!(g.useful_cols(), 6192);
        assert_eq!(g.panel_count(), 5);
        assert_eq!(g.padded_cols(), 6400);
        let q = Shape::QsaIn.column_map(m);
        assert_eq!(q.parts, [(Family::QsaQ, 0, 12288)]);
        assert_eq!(q.panel_count(), 10);
        assert_eq!(q.padded_cols(), 12800);
    }

    #[test]
    fn empty_selection_has_zero_panels() {
        for shape in Shape::ALL {
            let m = shape.column_map(FamilyMask::NONE);
            assert!(m.parts.is_empty());
            assert_eq!(m.useful_cols(), 0);
            assert_eq!(m.panel_count(), 0);
            assert_eq!(m.padded_cols(), 0);
            assert_eq!(m.locate(0), None);
        }
    }

    #[test]
    fn exact_panel_boundary_has_no_padding() {
        // qkv alone is eight panels exactly (no padding); a/b alone is one mostly-padded panel.
        let m = Shape::GdnIn.column_map(Family::GdnQkv.into());
        assert_eq!(m.useful_cols(), 10240);
        assert_eq!(m.panel_count(), 8);
        assert_eq!(m.padded_cols(), 10240);
        let m = Shape::GdnIn.column_map(Family::GdnA | Family::GdnB);
        assert_eq!((m.useful_cols(), m.panel_count(), m.padded_cols()), (96, 1, 1280));
    }

    #[test]
    fn locate_maps_fused_columns() {
        let q = Shape::QsaIn.column_map(FamilyMask::QSA);
        assert_eq!(q.locate(0), Some((Family::QsaQ, 0)));
        assert_eq!(q.locate(12287), Some((Family::QsaQ, 12287)));
        assert_eq!(q.locate(12288), Some((Family::QsaK, 0)));
        assert_eq!(q.locate(12800), Some((Family::QsaV, 0)));
        assert_eq!(q.locate(13311), Some((Family::QsaV, 511)));
        assert_eq!(q.locate(13312), Some((Family::QsaIdx, 0)));
        assert_eq!(q.locate(13951), Some((Family::QsaIdx, 639)));
        // Padding and beyond.
        assert_eq!(q.locate(13952), None);
        assert_eq!(q.locate(14079), None);
        assert_eq!(q.locate(usize::MAX), None);
        // Masked: columns shift down.
        let m = Shape::QsaIn.column_map(Family::QsaK | Family::QsaIdx);
        assert_eq!(m.locate(511), Some((Family::QsaK, 511)));
        assert_eq!(m.locate(512), Some((Family::QsaIdx, 0)));
        assert_eq!(m.locate(1151), Some((Family::QsaIdx, 639)));
        assert_eq!(m.locate(1152), None);
    }

    #[test]
    fn bf16_direct_is_rejected_with_exact_message() {
        assert_eq!(EpilogueContract::default(), EpilogueContract::I32Fold);
        assert_eq!(EpilogueContract::I32Fold.ensure_supported(), Ok(()));
        assert_eq!(
            EpilogueContract::Bf16Direct.ensure_supported(),
            Err("Bf16Direct pending probe B0".to_string())
        );
    }
}
