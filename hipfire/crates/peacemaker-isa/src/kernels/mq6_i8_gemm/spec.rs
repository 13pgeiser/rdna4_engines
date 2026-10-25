// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Variant axes and fixed geometry of the MQ6G256V2 x `block_i8_128` GEMM.
use crate::{kernels::iu4_k1::Variant, Arch, KernargLayout};

/// Epilogue: F32 overwrite, or BF16 bits (`isfinite(v) ? RNE : u >> 16`, `v = 0 + y`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi {
    Set,
    SetBf16,
}
/// CTA raster. `Banded` is the product (hipcc `IU4_G12_RASTER`, band 8). `Linear`
/// (`rs = x * 128`, `bs = y * 128`) exists only so the CPU interpreter, which has no
/// SALU float-reciprocal semantics, can run the data path of the same kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Raster {
    Banded,
    Linear,
}
/// CTA tile: weight rows x tokens per workgroup (the wave arrangement follows).
///
/// Every wave keeps the shipped 32-row x 64-token fragment grid (eight 16x16 accumulator
/// fragments: `cacc` 64 + `acc` 64 VGPRs), so the register map, the K-loop bundles and the fold
/// are the same for every tile and the VGPR count stays at the 192 ceiling. A bigger CTA tile is
/// therefore more waves, not bigger waves:
///
/// | tile | waves | threads | per-wave | LDS launch | WG/WGP (VGPR) | WG/WGP (LDS) | waves/WGP |
/// |---|---|---|---|---|---|---|---|
/// | `T128x128` (shipped) | 8 | 256 | 32 x 64 | 36864 B | 4 | 3 | 24 |
/// | `T256x128` | 16 | 512 | 32 x 64 | 55296 B | 2 | 2 | 32 |
/// | `T128x256` | 16 | 512 | 32 x 64 | 55296 B | 2 | 2 | 32 |
///
/// (VGPR column: 192 VGPRs per lane allow 8 waves per SIMD32, 32 per WGP; LDS column: 128 KiB per
/// WGP over the launch bytes.) A 256-wide tile moves one operand's staging to a single round of
/// the 512 threads and doubles the other's; its arithmetic intensity per staged byte is 1.33x the
/// shipped tile's (256x128 outputs per K step over 384 staged rows+tokens vs 128x128 over 256).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Tile {
    T128x128,
    T256x128,
    T128x256,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    pub epi: Epi,
    pub raster: Raster,
    pub tile: Tile,
    pub arch: Arch,
}

/// The shipped tile: 128 weight rows x 128 tokens x 8 waves (32 rows x 64 tokens per wave). The
/// constants below describe it; the other tiles' geometry is [`Tile`]'s methods.
pub const TILE_ROWS: u32 = 128;
pub const TILE_TOKENS: u32 = 128;
pub const THREADS: u32 = 256;
pub const VGPR_CEILING: u16 = 192;
pub const RASTER_BAND: u32 = 8;
/// `block_i8_128`: `{f32 d, i32 s, int8 qs[128]}`.
pub const XBLK_BYTES: u32 = 136;
/// MQ6G256V2 group: 8 header bytes + 192 payload bytes.
pub const GROUP_BYTES: u32 = 200;

/// LDS byte layout (every plane a separate builder slot).
///   slot s: A_s = tokens * 64 (token-group pairs x 4 K16 steps x 512 B),
///           W_s = rows * 64 (row-group pairs x 4 K16 steps x 512 B, decoded codes)
///   DS_p = tokens * 8 per K128 parity: per chunk of 128 tokens, planar token `d` [128 f32]
///          then `float(s)` [128 f32] (chunk `i` at `1024 * i`)
///   SZ_p = rows * 8 per K128 parity: per chunk of 128 rows, planar row `sc` [128 f32] then
///          `z'` [128 f32]
/// The chunking keeps the `ds_load_2addr_b32` offsets of the fold (`d` -> `s` = 128 dwords)
/// inside their 8-bit field for every tile; with one chunk it is the shipped planar layout.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub a: [u32; 2],
    pub w: [u32; 2],
    pub ds: [u32; 2],
    pub sz: [u32; 2],
    pub a_bytes: u32,
    pub w_bytes: u32,
    pub ds_bytes: u32,
    pub sz_bytes: u32,
    pub launch: u32,
}
/// The shipped tile's layout.
pub const LAYOUT: Layout = Layout {
    a: [0, 16384],
    w: [8192, 24576],
    ds: [32768, 33792],
    sz: [34816, 35840],
    a_bytes: 8192,
    w_bytes: 8192,
    ds_bytes: 1024,
    sz_bytes: 1024,
    launch: 36864,
};
/// The dynamic LDS bytes the shipped tile's launch must pass (the same figure as the hipcc
/// twin's `MQ6_I8_GFX12_LDS_BYTES`); see [`Tile::layout`] for the others.
pub const LAUNCH_LDS_BYTES: u32 = 36864;

impl Tile {
    pub const ALL: [Tile; 3] = [Tile::T128x128, Tile::T256x128, Tile::T128x256];
    /// Weight rows per workgroup.
    pub fn rows(self) -> u32 {
        match self {
            Self::T128x128 | Self::T128x256 => 128,
            Self::T256x128 => 256,
        }
    }
    /// Tokens per workgroup.
    pub fn tokens(self) -> u32 {
        match self {
            Self::T128x128 | Self::T256x128 => 128,
            Self::T128x256 => 256,
        }
    }
    pub fn waves(self) -> u32 {
        match self {
            Self::T128x128 => 8,
            Self::T256x128 | Self::T128x256 => 16,
        }
    }
    pub fn threads(self) -> u32 {
        self.waves() * 32
    }
    /// Staging rounds per 64-K slab (each lane moves one 8-byte chunk of its token / one 12-byte
    /// weight quad per round): activations / weights.
    pub fn a_rounds(self) -> u32 {
        self.tokens() * 4 / self.threads()
    }
    pub fn w_rounds(self) -> u32 {
        self.rows() * 4 / self.threads()
    }
    /// Slab rows covered by one staging round (`R = r * round_rows + tid / 4`).
    pub fn round_rows(self) -> u32 {
        self.threads() / 4
    }
    /// Token halves (64 tokens each) and row pairs (32 rows each): wave `w` owns token half
    /// `w % tok_halves` and row pair `w / tok_halves`.
    pub fn tok_halves(self) -> u32 {
        self.tokens() / 64
    }
    pub fn row_pairs(self) -> u32 {
        self.rows() / 32
    }
    /// `ds_store` / `buffer` immediates: LDS bytes between a staging round and the next.
    pub fn round_lds_stride(self) -> u32 {
        self.threads() / 128 * 2048
    }
    /// The CLI / variant grammar name: rows x tokens x waves.
    pub fn name(self) -> &'static str {
        match self {
            Self::T128x128 => "128x128x8",
            Self::T256x128 => "256x128x16",
            Self::T128x256 => "128x256x16",
        }
    }
    /// The shipped tile keeps the product symbol; the others add this suffix.
    pub fn symbol_suffix(self) -> &'static str {
        match self {
            Self::T128x128 => "",
            Self::T256x128 => "_t256x128",
            Self::T128x256 => "_t128x256",
        }
    }
    pub fn layout(self) -> Layout {
        match self {
            Self::T128x128 => LAYOUT,
            // Slot stride 24576: A0 W0 | A1 W1 | DS0 DS1 | SZ0 SZ1.
            Self::T256x128 => Layout {
                a: [0, 24576],
                w: [8192, 32768],
                ds: [49152, 50176],
                sz: [51200, 53248],
                a_bytes: 8192,
                w_bytes: 16384,
                ds_bytes: 1024,
                sz_bytes: 2048,
                launch: 55296,
            },
            Self::T128x256 => Layout {
                a: [0, 24576],
                w: [16384, 40960],
                ds: [49152, 51200],
                sz: [53248, 54272],
                a_bytes: 16384,
                w_bytes: 8192,
                ds_bytes: 2048,
                sz_bytes: 1024,
                launch: 55296,
            },
        }
    }
}
impl std::str::FromStr for Tile {
    type Err = String;
    /// `128x128`, `256x128`, `128x256` (rows x tokens), or the variant-name spellings
    /// `128x128x8`, `256x128x16`, `128x256x16`.
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "128x128" | "128x128x8" => Ok(Self::T128x128),
            "256x128" | "256x128x16" => Ok(Self::T256x128),
            "128x256" | "128x256x16" => Ok(Self::T128x256),
            _ => Err(format!(
                "unknown mq6_i8_gemm tile {s} (128x128|256x128|128x256)"
            )),
        }
    }
}

impl Epi {
    pub fn name(self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::SetBf16 => "set_bf16",
        }
    }
}
impl Spec {
    pub fn product(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Banded,
            tile: Tile::T128x128,
            arch: Arch::Gfx1201,
        }
    }
    /// The interpreter twin: identical data path, linear raster.
    pub fn emulation(epi: Epi) -> Self {
        Self {
            epi,
            raster: Raster::Linear,
            tile: Tile::T128x128,
            arch: Arch::Gfx1201,
        }
    }
    /// The same spec on another tile.
    pub fn with_tile(self, tile: Tile) -> Self {
        Self { tile, ..self }
    }
    pub fn module() -> &'static str {
        "gemm_mq6g256v2_mmq_i8_pm_gfx1201"
    }
    pub fn symbol(self) -> String {
        let base = format!(
            "gemm_mq6g256v2_mmq_i8_pm_{}_gfx1201{}",
            self.epi.name(),
            self.tile.symbol_suffix()
        );
        match self.raster {
            Raster::Banded => base,
            Raster::Linear => format!("{base}_linear"),
        }
    }
    pub fn variant_name(self) -> String {
        format!(
            "{}-{}{}",
            self.tile.name(),
            self.epi.name(),
            match self.raster {
                Raster::Banded => "",
                Raster::Linear => "-linear",
            }
        )
    }
    /// The hipcc twin's ABI `(A, Xq, Y, M, K, N)` followed by K1's hidden block.
    pub fn kernargs(self) -> KernargLayout {
        Variant::FullSet.kernargs()
    }
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 {
            return Err("mq6_i8_gemm targets gfx1201 only".into());
        }
        Ok(())
    }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "set" => Ok(Self::Set),
            "set_bf16" => Ok(Self::SetBf16),
            _ => Err(format!("unknown mq6_i8_gemm epilogue {s}")),
        }
    }
}
