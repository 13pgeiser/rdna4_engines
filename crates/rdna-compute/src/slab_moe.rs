// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab (ctl3-hot) GPU entries, gfx1201, gfx1100 and gfx1151 (two buckets,
//! cold = 0 and MQ4 = 1; the cold codec is W3 (1) or E8P (3) on gfx1201, E8P (3) on gfx1100 and
//! W3 (1) on gfx1151; every arch also serves MQ4-only layers (`cold_codec == 0`); W2 is refused
//! everywhere, gfx1100 refuses W3, gfx1151 refuses E8P; every other arch refuses). The K1 cold-bucket
//! table is inventory ([`slab_cold_k1_metadata`]); serving selection is the admission-gated
//! [`slab_cold_kernels`]. The per-arch modules (with the hot MQ4 K1 bucket) are
//! [`SlabArchKernels`], the per-cold-codec cold K1 blobs and symbols [`SlabCold`].
//!
//! * K1 decode: [`Gpu::qwen4_slab_union_plan`] turns the routed ids into the
//!   fixed-capacity per-codec-bucket lane-union descriptor the four
//!   `mq_i4_moe_gemv_*` symbols read (the data-dependent bucket counts stay on
//!   the device; unused bucket tails are empty experts the kernels skip), and
//!   [`Gpu::mq_i4_moe_gemv_slab`] launches a projection over the non-empty
//!   buckets. Both go through `launch_maybe_blob`, so they are recorded by the
//!   replay recorder and captured by graphs like every other route launch.
//!   The planner's device `u8` table is membership only (cold = 1, MQ4 = 0).
//!   Every K1 symbol (all three archs, MQ4/W3/E8P, narrow and wide) has the 96-byte kernarg
//!   ABI: seven pointers, eight u32 words at `@56..@88`, then `n_experts@88` and a zero
//!   `reserved@92`. A workgroup whose expert id is not `< n_experts` (one unsigned,
//!   workgroup-uniform compare, so a negative id is out of range too) takes the existing
//!   no-load, no-store exit before the id indexes the pointer table, before any
//!   weight or descriptor prefetch and before the first barrier: its output tile stays
//!   untouched. [`Gpu::mq_i4_moe_gemv_slab`] passes `n_experts = SLAB_EXPERTS`, the
//!   extent of the `[512]` pointer table it validates.
//! * K2 prefill: [`Gpu::qwen4_slab_tile_masks`] splits the grouped tile ids by
//!   codec with the table-driven mask: the builder `ep_tile_mask_pm_<arch>` (gfx1100, gfx1201), or on
//!   gfx1151 its hipcc twin `qwen4_slab_tile_mask` (same law, same ids out); gfx1100 down
//!   sidecars come from [`Gpu::qwen4_moe_rotate128_i4_slab`] (K3).

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemv::WideTwinFamily;
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Module key of the embedded gfx1201 K1 certified builder image.
pub const MQ_I4_MOE_GEMV_MODULE: &str = "mq_i4_moe_gemv_pm_gfx1201";
/// Module key of the embedded gfx1151 K1 certified builder image (MQ4 and W3 symbols).
pub const MQ_I4_MOE_GEMV_GFX1151_MODULE: &str = "mq_i4_moe_gemv_pm_gfx1151";

/// Explicit arguments supplied through HIP's kernelParams launch path.
const MQ_I4_MOE_GEMV_KERNARG_COUNT: usize = 17;

/// K1 symbols (gfx1201 and gfx1151) in bucket order: `[bucket][kind]`, bucket 0 = W3 (`MQ3I4`),
/// 1 = MQ4 (`Mq4Sym`), kind 0 = gate/up (R256), 1 = down (R128).
pub const MQ_I4_MOE_GEMV_SYMBOLS: [[&str; 2]; 2] = [
    [
        "mq_i4_moe_gemv_mq3i4_r256_gate_up",
        "mq_i4_moe_gemv_mq3i4_r128_down",
    ],
    [
        "mq_i4_moe_gemv_mq4sym_r256_gate_up",
        "mq_i4_moe_gemv_mq4sym_r128_down",
    ],
];

/// Wide K1 entries, `[capacity (16/32)][codec][kind]`; the same 96-byte ABI as the narrow entries (`n_experts@88`, zero `reserved@92`).
pub const MQ_I4_MOE_GEMV_WIDE_SYMBOLS: [[[&str; 2]; 2]; 2] = [
    [
        [
            "mq_i4_moe_gemv_mq3i4_r256_gate_up_c16",
            "mq_i4_moe_gemv_mq3i4_r128_down_c16",
        ],
        [
            "mq_i4_moe_gemv_mq4sym_r256_gate_up_c16",
            "mq_i4_moe_gemv_mq4sym_r128_down_c16",
        ],
    ],
    [
        [
            "mq_i4_moe_gemv_mq3i4_r256_gate_up_c32",
            "mq_i4_moe_gemv_mq3i4_r128_down_c32",
        ],
        [
            "mq_i4_moe_gemv_mq4sym_r256_gate_up_c32",
            "mq_i4_moe_gemv_mq4sym_r128_down_c32",
        ],
    ],
];

/// Module key of the embedded gfx1100 MQ4-only K1 certified builder image (the hot MQ4 bucket;
/// no W2 entry). Distinct from the retired mixed `mq_i4_moe_gemv_pm_gfx1100` module.
pub const MQ_I4_MOE_GEMV_GFX1100_MODULE: &str = "mq_i4_moe_gemv_mq4only_pm_gfx1100";

/// gfx1100 MQ4 (`Mq4Sym`) K1 symbols (the hot bucket): `[kind]`, 0 = gate/up (R256), 1 = down (R128).
pub const MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS: [&str; 2] = [
    "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
    "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100",
];

/// Module key of the embedded gfx1100 E8P K1 certified builder image (cold bucket metadata).
pub const E8P_I4_MOE_GEMV_GFX1100_MODULE: &str = "e8p_i4_moe_gemv_pm_gfx1100";

/// gfx1100 E8P K1 symbols: `[kind]`, 0 = gate/up (R256), 1 = down (R128), contracts
/// `kernels/e8p_i4_moe_gemv.gfx1100.{r256_gate_up,r128_down}.contract.json`. Same 96-byte
/// kernarg ABI (`n_experts@88`, zero `reserved@92`; the frozen 88-byte F4 layout plus the
/// trailing guard words), grid `[M, U, 1]`, block `[32, 1, 1]` and zero dynamic LDS as the MQ4 K1 symbols
/// (static LDS 1,536 B instead of 0).
pub const E8P_I4_MOE_GEMV_GFX1100_SYMBOLS: [&str; 2] = [
    "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
    "e8p_i4_moe_gemv_r128_down_pm_gfx1100",
];

/// Module key and symbol of the gfx1100 K3 down A4 producer.
pub const MQ_I4_A4_GFX1100_MODULE: &str = "mq_i4_a4_pm_gfx1100";
pub const MQ_I4_A4_GFX1100_SYMBOL: &str = "mq_i4_a4_r128_k640_gfx1100";

/// Module key and symbol of the gfx1100 slab tile mask (module = symbol).
pub const EP_TILE_MASK_GFX1100_MODULE: &str = "ep_tile_mask_pm_gfx1100";
const PLAN_SYMBOL: &str = "qwen4_slab_union_plan";
/// Planner module keys: the shared one (gfx1201 and gfx1151: the historical key) and gfx1100's own.
const PLAN_MODULE_SHARED: &str = "qwen4_slab_union_plan";
const PLAN_MODULE_GFX1100: &str = "qwen4_slab_union_plan_gfx1100";
/// Module key and symbol of the gfx1151 hipcc tile mask (module = symbol).
const MASK_SYMBOL: &str = "qwen4_slab_tile_mask";
const MASK_MODULE: &str = "qwen4_slab_tile_mask";

/// Threads (tiles) per workgroup of the tile mask kernel (the builder masks and the gfx1151 hipcc twin).
const SLAB_TILE_MASK_BLOCK: u32 = 256;

/// One K1 bucket's module, blob and symbols: `[kind]`, kind 0 = gate/up (R256), 1 = down (R128).
pub struct SlabK1 {
    pub module: &'static str,
    pub image: &'static [u8],
    pub symbols: [&'static str; 2],
}

/// The cold-bucket K1 table of one cold codec on one arch (bucket 0).
pub struct SlabCold {
    /// Raw codec id of the cold bucket in the codec tables (also the cold tile-mask rank).
    pub cold_codec: u8,
    pub k1: SlabK1,
}

/// The table-driven tile mask of one arch: an embedded certified builder image or a hipcc source
/// (the repository's embedded/HIP distinction). Allocation-free, so it is a valid static table value.
#[derive(Clone, Copy)]
pub enum SlabTileMask {
    /// Embedded builder image `(module, image, symbol)`: gfx1201 and gfx1100.
    Embedded {
        module: &'static str,
        image: &'static [u8],
        symbol: &'static str,
    },
    /// hipcc-compiled source `(module, source, symbol)`: gfx1151 only.
    Hipcc {
        module: &'static str,
        source: &'static str,
        symbol: &'static str,
    },
}

/// The sealed slab's per-arch modules, blobs and symbols that do not depend on the cold codec,
/// including the hot MQ4 K1 bucket (bucket 1).
pub struct SlabArchKernels {
    pub arch: &'static str,
    /// Lane-union planner: hipcc module key and source.
    pub plan_module: &'static str,
    pub plan_src: fn() -> &'static str,
    /// Hot MQ4 K1 bucket.
    pub mq4_k1: SlabK1,
    /// K3 down A4 producer `(module, image, symbol)`; gfx1100 only (gfx1201 and gfx1151 delegate to
    /// the existing `qwen4_moe_rotate128_i4`).
    pub k3: Option<(&'static str, &'static [u8], &'static str)>,
    /// Table-driven tile mask.
    pub tile_mask: SlabTileMask,
}

fn plan_src_shared() -> &'static str {
    kernels::QWEN4_SLAB_UNION_PLAN_SRC
}

static SLAB_GFX1201: SlabArchKernels = SlabArchKernels {
    arch: "gfx1201",
    plan_module: PLAN_MODULE_SHARED,
    plan_src: plan_src_shared,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[1],
    },
    k3: None,
    tile_mask: SlabTileMask::Embedded {
        module: crate::moe::EP_TILE_MASK_PM_GFX1201_SYMBOL,
        image: kernels::EP_TILE_MASK_PM_GFX1201,
        symbol: crate::moe::EP_TILE_MASK_PM_GFX1201_SYMBOL,
    },
};

static SLAB_GFX1100: SlabArchKernels = SlabArchKernels {
    arch: "gfx1100",
    plan_module: PLAN_MODULE_GFX1100,
    plan_src: kernels::qwen4_slab_union_plan_gfx1100_src,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1100_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
        symbols: MQ_I4_MOE_GEMV_GFX1100_MQ4_SYMBOLS,
    },
    k3: Some((
        MQ_I4_A4_GFX1100_MODULE,
        kernels::MQ_I4_A4_PM_GFX1100,
        MQ_I4_A4_GFX1100_SYMBOL,
    )),
    tile_mask: SlabTileMask::Embedded {
        module: EP_TILE_MASK_GFX1100_MODULE,
        image: kernels::EP_TILE_MASK_PM_GFX1100,
        symbol: EP_TILE_MASK_GFX1100_MODULE,
    },
};

/// gfx1151 (Strix Halo): beta's certified K1 image (MQ4 hot bucket; the W3 cold bucket is
/// [`SLAB_COLD_GFX1151_W3`] on the same image), the shared planner, the hipcc tile mask and no K3
/// (the down sidecar delegates to the existing producer, `qwen4_moe_rotate128_i4`).
static SLAB_GFX1151: SlabArchKernels = SlabArchKernels {
    arch: "gfx1151",
    plan_module: PLAN_MODULE_SHARED,
    plan_src: plan_src_shared,
    mq4_k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1151_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[1],
    },
    k3: None,
    tile_mask: SlabTileMask::Hipcc {
        module: MASK_MODULE,
        source: kernels::QWEN4_SLAB_TILE_MASK_SRC,
        symbol: MASK_SYMBOL,
    },
};

static SLAB_COLD_GFX1201_W3: SlabCold = SlabCold {
    cold_codec: 1,
    k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[0],
    },
};

static SLAB_COLD_GFX1201_E8P: SlabCold = SlabCold {
    cold_codec: 3,
    k1: SlabK1 {
        module: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_MODULE,
        image: kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
        symbols: crate::e8p_moe_decode::E8P_I4_MOE_GEMV_SYMBOLS,
    },
};

/// gfx1100 cold E8P K1 (`e8p_i4_moe_gemv_pm_gfx1100`), served for codec 3 once admitted.
static SLAB_COLD_GFX1100_E8P: SlabCold = SlabCold {
    cold_codec: 3,
    k1: SlabK1 {
        module: E8P_I4_MOE_GEMV_GFX1100_MODULE,
        image: kernels::E8P_I4_MOE_GEMV_PM_GFX1100,
        symbols: E8P_I4_MOE_GEMV_GFX1100_SYMBOLS,
    },
};

/// gfx1151 cold W3 K1: the same module and image as [`SLAB_GFX1151`]'s MQ4 bucket, the W3
/// (`MQ3I4`) symbols.
static SLAB_COLD_GFX1151_W3: SlabCold = SlabCold {
    cold_codec: 1,
    k1: SlabK1 {
        module: MQ_I4_MOE_GEMV_GFX1151_MODULE,
        image: kernels::MQ_I4_MOE_GEMV_PM_GFX1151,
        symbols: MQ_I4_MOE_GEMV_SYMBOLS[0],
    },
};

/// The per-arch slab table of `arch` (exact name), `None` off gfx1201/gfx1100/gfx1151.
pub fn slab_arch_kernels(arch: &str) -> Option<&'static SlabArchKernels> {
    match arch {
        "gfx1201" => Some(&SLAB_GFX1201),
        "gfx1100" => Some(&SLAB_GFX1100),
        "gfx1151" => Some(&SLAB_GFX1151),
        _ => None,
    }
}

/// The cold-bucket K1 table of cold codec `cold_codec` the sealed route SERVES on `arch`:
/// [`slab_cold_k1_metadata`] restricted to what [`crate::qwen4_slab_codec_admitted`] admits, so
/// metadata alone never launches or preloads a codec. `None` for MQ4 (0, never cold), every
/// codec the matrix refuses on that arch, and W2 (2) everywhere. On gfx1100 only E8P (3)
/// resolves (its K1 [`SLAB_COLD_GFX1100_E8P`]; the matching K2 is `slab_k2_entry`); W3 (1) does not.
/// On gfx1151 only W3 (1) resolves ([`SLAB_COLD_GFX1151_W3`]); E8P (3) does not.
pub fn slab_cold_kernels(arch: &str, cold_codec: u8) -> Option<&'static SlabCold> {
    if cold_codec == 0 || !crate::qwen4_slab_codec_admitted(arch, cold_codec) {
        return None;
    }
    slab_cold_k1_metadata(arch, cold_codec)
}

/// The cold K1 entries that exist per (`arch`, cold codec): gfx1201 W3 (1) and E8P (3), gfx1100
/// E8P (3, the qualified `e8p_i4_moe_gemv_pm_gfx1100` image), gfx1151 W3 (1). This is inventory, NOT
/// permission: serving selection goes through [`slab_cold_kernels`] (admission-gated).
pub fn slab_cold_k1_metadata(arch: &str, cold_codec: u8) -> Option<&'static SlabCold> {
    match (arch, cold_codec) {
        ("gfx1201", 1) => Some(&SLAB_COLD_GFX1201_W3),
        ("gfx1201", 3) => Some(&SLAB_COLD_GFX1201_E8P),
        ("gfx1100", 3) => Some(&SLAB_COLD_GFX1100_E8P),
        ("gfx1151", 1) => Some(&SLAB_COLD_GFX1151_W3),
        _ => None,
    }
}

/// Whether `k1` is one of the shared MQ4/W3 K1 entries ([`MQ_I4_MOE_GEMV_SYMBOLS`], gfx1151 and
/// gfx1201) that the wide `_c16/_c32` twins ([`MQ_I4_MOE_GEMV_WIDE_SYMBOLS`]) are exports of the
/// same image for. gfx1100's MQ4-only and E8P K1 entries have no wide twin.
fn slab_k1_wide_capable(k1: &SlabK1) -> bool {
    MQ_I4_MOE_GEMV_SYMBOLS.contains(&k1.symbols)
}

/// Whether every K1 bucket a layer launches on `arch` (the hot MQ4 bucket and `cold`, `None` for an
/// MQ4-only layer) has a wide twin; the wide launch uses one symbol family for both buckets.
fn slab_wide_capable(arch: &SlabArchKernels, cold: Option<&SlabCold>) -> bool {
    slab_k1_wide_capable(&arch.mq4_k1) && cold.is_none_or(|cold| slab_k1_wide_capable(&cold.k1))
}

/// Descriptor/allocation capacity, not the architecture's row admission.
pub const SLAB_MAX_LANES: usize = 32;
/// Capacity of the unchanged decode K1 entries and S1 fallback groups.
pub const SLAB_NARROW_MAX_LANES: usize = 8;
/// Routed experts per lane.
pub const SLAB_TOP_K: usize = 10;
/// Experts per layer.
pub const SLAB_EXPERTS: usize = 512;
/// Gate/up output rows (gate 640 + up 640), reduction width.
pub const SLAB_GATE_UP_M: usize = 1280;
pub const SLAB_GATE_UP_K: usize = 2560;
/// Down output rows and reduction width.
pub const SLAB_DOWN_M: usize = 2560;
pub const SLAB_DOWN_K: usize = 640;

/// Bucket of codec byte `0` (MQ4) and `1` (W3) in the gfx1201 K1 order: W3 first.
pub const fn slab_bucket_of_codec(codec: u8) -> usize {
    if codec == 1 {
        0
    } else {
        1
    }
}

/// Bucket of codec id `codec` on `arch` (cold = 0, MQ4 = 1), `None` when the
/// slab does not admit that codec there ([`crate::qwen4_slab_codec_admitted`]).
pub fn slab_bucket_of_codec_on(arch: &str, codec: u8) -> Option<usize> {
    slab_arch_kernels(arch)?;
    if !crate::qwen4_slab_codec_admitted(arch, codec) {
        return None;
    }
    Some(usize::from(codec == 0))
}

/// How a layer's experts map to the slab's two buckets (cold = 0, MQ4 = 1).
#[derive(Clone, Copy)]
pub enum SlabBuckets<'a> {
    /// Every expert is in one bucket: `0` = the layer's cold codec (W3 or E8P), `1` = MQ4. Only
    /// that bucket launches.
    Uniform(usize),
    /// Per-expert bucket MEMBERSHIP (`[512]` u8), not the real codec id: `1` = cold, `0` = MQ4.
    /// Both buckets launch. (The tile mask's i32 table is the one that holds raw codec ids.)
    Table(&'a GpuTensor),
}

impl Gpu {
    /// Whether a wide K1 launch of `lanes` lanes is admitted for a layer whose cold bucket is
    /// codec `cold_codec` (`0` = none): the developer opt-in is on, the row admission
    /// ([`Gpu::wide_twin_cap`]) holds, and every K1 bucket the layer launches is a shared
    /// MQ4/W3 K1 entry with a `_c16/_c32` twin ([`slab_wide_capable`]). Allocation capacity alone
    /// never widens the shared row-exact admission; gfx1100 and E8P layers keep the unchanged
    /// <=8-lane entries (S1 groups of [`SLAB_NARROW_MAX_LANES`] above 8).
    pub fn qwen4_slab_wide_twin(&self, lanes: usize, cold_codec: u8) -> bool {
        let Some(arch) = slab_arch_kernels(&self.arch) else {
            return false;
        };
        let cold = match cold_codec {
            0 => None,
            codec => match slab_cold_kernels(&self.arch, codec) {
                Some(cold) => Some(cold),
                None => return false,
            },
        };
        self.wide_twin_cap(WideTwinFamily::K1, lanes).is_some() && slab_wide_capable(arch, cold)
    }

    /// Load the sealed slab's kernels (call before graph capture) for a layer whose cold bucket
    /// is codec `cold_codec`: `0` = no cold bucket (an MQ4-only layer, any arch), gfx1201
    /// 1 = W3 or 3 = E8P, gfx1100 3 = E8P, gfx1151 1 = W3; every other value (W2 / codec 2 everywhere,
    /// W3 on gfx1100, E8P on gfx1151, unknown codecs) is refused by the shared admission matrix before
    /// anything loads. Loads the planner and,
    /// on gfx1201, the cold (if any) and MQ4 K1 symbols (the historical W3 preload, unchanged);
    /// on gfx1100 the planner and every serving entry of
    /// [`crate::kernel_registry::slab_embedded_entries`]`("gfx1100", cold_codec)`: MQ4 K1, MQ4 K2
    /// NT1 GEMMs, K3 and the tile mask, plus, for E8P layers only, the cold K1 and the E8P K2
    /// NT1 GEMMs (an MQ4-only layer loads no E8P module). Each entry is ensured once.
    ///
    /// gfx1151 exception to the no-cold-preload rule (gfx1100 and gfx1201 keep it): beta's preload
    /// is unconditional, so even `cold_codec == 0` loads the planner, the HIP tile mask and all four
    /// K1 handles (W3 gate/up, W3 down, MQ4 gate/up, MQ4 down, in that order) of the already-qualified
    /// shared image. That prepares the image only: it is not a W3 bucket launch
    /// ([`Gpu::mq_i4_moe_gemv_slab`] still refuses bucket 0 when `cold_codec == 0`) and admits no other
    /// codec. K2 loading stays lazy.
    pub fn ensure_qwen4_slab_kernels(&mut self, cold_codec: u8) -> HipResult<()> {
        let arch = self.slab_kernels("ensure_qwen4_slab_kernels")?;
        let cold = self.slab_cold_opt("ensure_qwen4_slab_kernels", cold_codec)?;
        self.bind_thread()?;
        self.ensure_kernel(arch.plan_module, (arch.plan_src)(), PLAN_SYMBOL)?;
        match arch.arch {
            "gfx1100" => {
                let inventory = crate::kernel_registry::slab_embedded_entries(
                    arch.arch, cold_codec,
                )
                .map_err(|e| {
                    hip_bridge::HipError::new(0, &format!("ensure_qwen4_slab_kernels: {e:?}"))
                })?;
                for planned in inventory {
                    let crate::kernel_registry::PlannedKernel::Embedded {
                        module,
                        image,
                        symbols,
                        ..
                    } = planned
                    else {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "ensure_qwen4_slab_kernels: the embedded slab inventory holds a hipcc entry",
                        ));
                    };
                    for symbol in symbols {
                        self.ensure_embedded_kernel(module, image, symbol)?;
                    }
                }
            }
            "gfx1151" => {
                // Beta's gfx1151 preload, unconditional (even `cold_codec == 0`): the planner (ensured
                // above), the HIP tile mask, then all four handles of the shared K1 image in beta order
                // (W3 gate/up, W3 down, MQ4 gate/up, MQ4 down). The cold codec was already admission-checked
                // (only 0 and W3 reach here); this prepares the shared image and is not a W3 bucket launch.
                match arch.tile_mask {
                    SlabTileMask::Hipcc {
                        module,
                        source,
                        symbol,
                    } => self.ensure_kernel(module, source, symbol)?,
                    SlabTileMask::Embedded { .. } => {
                        return Err(hip_bridge::HipError::new(
                            0,
                            "ensure_qwen4_slab_kernels: the gfx1151 tile mask is a hipcc source, not an embedded image",
                        ));
                    }
                }
                for k1 in [&SLAB_COLD_GFX1151_W3.k1, &arch.mq4_k1] {
                    for symbol in k1.symbols {
                        self.ensure_embedded_kernel(k1.module, k1.image, symbol)?;
                    }
                }
            }
            _ => {
                for k1 in cold
                    .map(|cold| &cold.k1)
                    .into_iter()
                    .chain(std::iter::once(&arch.mq4_k1))
                {
                    for symbol in k1.symbols {
                        self.ensure_embedded_kernel(k1.module, k1.image, symbol)?;
                    }
                }
            }
        }
        if WideTwinFamily::K1.enabled() && slab_wide_capable(arch, cold) {
            for symbol in MQ_I4_MOE_GEMV_WIDE_SYMBOLS.iter().flatten().flatten() {
                self.ensure_embedded_kernel(arch.mq4_k1.module, arch.mq4_k1.image, symbol)?;
            }
        }
        Ok(())
    }

    /// The sealed slab's per-arch module table of this device, or a refusal for every
    /// arch outside [`slab_arch_kernels`].
    fn slab_kernels(&self, what: &str) -> HipResult<&'static SlabArchKernels> {
        slab_arch_kernels(&self.arch).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "{what}: the sealed expert-slab route is gfx1100/gfx1151/gfx1201-only, got {}",
                    self.arch
                ),
            )
        })
    }

    /// The cold-bucket K1 table of cold codec `cold_codec` on this device
    /// ([`slab_cold_kernels`]); refuses codec 0 (no cold codec) and every codec the shared
    /// admission matrix refuses (W2 / codec 2 everywhere, W3 on gfx1100: a tabled K1 image,
    /// [`slab_cold_k1_metadata`], is not permission).
    fn slab_cold(&self, what: &str, cold_codec: u8) -> HipResult<&'static SlabCold> {
        slab_cold_kernels(&self.arch, cold_codec).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!(
                    "{what}: cold codec {cold_codec} has no sealed expert-slab entry on {}",
                    self.arch
                ),
            )
        })
    }

    /// [`Gpu::slab_cold`] where `cold_codec == 0` means "no cold bucket" (an MQ4-only layer):
    /// `Ok(None)`.
    fn slab_cold_opt(&self, what: &str, cold_codec: u8) -> HipResult<Option<&'static SlabCold>> {
        if cold_codec == 0 {
            return Ok(None);
        }
        self.slab_cold(what, cold_codec).map(Some)
    }

    /// Reserve the slab route's scratch (union descriptor for `SLAB_MAX_LANES`,
    /// and `tiles` masked tile ids per codec) before any capture.
    pub fn reserve_qwen4_slab(&mut self, tiles: usize) -> HipResult<()> {
        self.slab_kernels("reserve_qwen4_slab")?;
        self.bind_thread()?;
        if crate::scratch::scratch_will_grow(
            self.scratch.qwen4_slab_union_scratch_bytes,
            self.scratch.qwen4_slab_union_scratch.is_some(),
            crate::scratch::qwen4_slab_union_words(SLAB_MAX_LANES) * 4,
        ) || (tiles != 0
            && crate::scratch::scratch_will_grow(
                self.scratch.qwen4_slab_tiles_scratch_bytes,
                self.scratch.qwen4_slab_tiles_scratch.is_some(),
                2 * tiles * 4,
            ))
        {
            self.invalidate_for_scratch_growth();
        }
        self.scratch
            .reserve_qwen4_slab(&self.hip, SLAB_MAX_LANES, tiles)
    }

    fn slab_union_base(&self) -> HipResult<*mut c_void> {
        self.scratch
            .qwen4_slab_union_scratch
            .as_ref()
            .map(|buf| buf.as_ptr())
            .ok_or_else(|| hip_bridge::HipError::new(0, "slab union scratch is not reserved"))
    }

    /// Plan the lane union of `lanes` decode lanes from `topk` (`[lanes * 10]`
    /// i32 routed ids, slot = lane * 10 + rank) into the slab union scratch.
    /// Stream-ordered; no readback.
    pub fn qwen4_slab_union_plan(
        &mut self,
        topk: &GpuTensor,
        buckets: SlabBuckets<'_>,
        lanes: usize,
    ) -> HipResult<()> {
        let arch = self.slab_kernels("qwen4_slab_union_plan")?;
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_slab_union_plan: lanes must be 1..=32",
            ));
        }
        let slots = SLAB_TOP_K * lanes;
        if topk.buf.size() < slots * 4 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_slab_union_plan: topk holds fewer than 10 * lanes ids",
            ));
        }
        let (codec_ptr, uniform) = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => {
                (std::ptr::null_mut::<c_void>(), bucket as i32)
            }
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qwen4_slab_union_plan: bucket must be 0 or 1",
                ))
            }
            SlabBuckets::Table(table) => {
                if table.buf.size() < SLAB_EXPERTS {
                    return Err(hip_bridge::HipError::new(
                        0,
                        "qwen4_slab_union_plan: codec table holds fewer than 512 bytes",
                    ));
                }
                (table.buf.as_ptr(), -1)
            }
        };
        self.bind_thread()?;
        self.ensure_kernel(arch.plan_module, (arch.plan_src)(), PLAN_SYMBOL)?;
        if self.scratch.qwen4_slab_union_scratch_bytes
            < crate::scratch::qwen4_slab_union_words(SLAB_MAX_LANES) * 4
            || self.scratch.qwen4_slab_union_scratch.is_none()
        {
            self.reserve_qwen4_slab(0)?;
        }
        let out = self.slab_union_base()?;
        let tp = topk.buf.as_ptr();
        let sv = slots as i32;
        let nexp = SLAB_EXPERTS as i32;
        let mut params = [
            &tp as *const _ as *mut c_void,
            &codec_ptr as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
            &nexp as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(PLAN_SYMBOL, [1, 1, 1], [512, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(codec_ptr);
            b.push_ptr(out);
            b.push_i32(sv);
            b.push_i32(uniform);
            b.push_i32(nexp);
            b
        })
    }

    /// Launch one K1 projection over every non-empty bucket of the union
    /// [`Gpu::qwen4_slab_union_plan`] wrote for `lanes` lanes.
    ///
    /// Gate/up (`down == false`): `xrot` `[lanes, 2560]` F32 rotated
    /// activations, `y0`/`y1` `[lanes, 10, 640]` F32 gate/up. Down: `xrot`
    /// `[10 * lanes, 640]` F32, `y0` `[lanes, 10, 2560]` F32 (unweighted),
    /// `y1` `None`. `weight_ptrs` is the layer's `[512]` u64 pointer table of
    /// the projection. Output rows of every slot are written exactly once. The 96-byte kernarg
    /// ends `n_experts = SLAB_EXPERTS` and a zero reserved word; an expert id outside
    /// `[0, n_experts)` skips its workgroup without any load or store: a fault guard for a
    /// corrupt or non-finite-router id, not a routing rule.
    #[allow(clippy::too_many_arguments)]
    pub fn mq_i4_moe_gemv_slab(
        &mut self,
        down: bool,
        buckets: SlabBuckets<'_>,
        weight_ptrs: &GpuTensor,
        xrot: &GpuTensor,
        y0: &GpuTensor,
        y1: Option<&GpuTensor>,
        lanes: usize,
        cold_codec: u8,
    ) -> HipResult<()> {
        let arch = self.slab_kernels("mq_i4_moe_gemv_slab")?;
        let cold = self.slab_cold_opt("mq_i4_moe_gemv_slab", cold_codec)?;
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: lanes must be 1..=32",
            ));
        }
        let wide = self.qwen4_slab_wide_twin(lanes, cold_codec);
        if lanes > SLAB_NARROW_MAX_LANES && !wide {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: wide lanes require the K1 opt-in, shared architecture admission and an MQ4/W3 K1 layer",
            ));
        }
        let slots = SLAB_TOP_K * lanes;
        let (m, k, x_row_div, x_src_rows, y_stride) = if down {
            (SLAB_DOWN_M, SLAB_DOWN_K, 1usize, slots, SLAB_DOWN_M)
        } else {
            (
                SLAB_GATE_UP_M,
                SLAB_GATE_UP_K,
                SLAB_TOP_K,
                lanes,
                SLAB_GATE_UP_M / 2,
            )
        };
        if down == y1.is_some() {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: gate/up needs the up output, down has none",
            ));
        }
        if weight_ptrs.buf.size() < SLAB_EXPERTS * 8
            || xrot.buf.size() < x_src_rows * k * 4
            || y0.buf.size() < slots * y_stride * 4
            || y1.is_some_and(|y| y.buf.size() < slots * y_stride * 4)
        {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: operand smaller than the launch extent",
            ));
        }
        let present: [bool; 2] = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => [bucket == 0, bucket == 1],
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "mq_i4_moe_gemv_slab: bucket must be 0 or 1",
                ))
            }
            SlabBuckets::Table(_) => [true, true],
        };
        self.bind_thread()?;
        let union = self.slab_union_base()? as usize;
        let s = slots;
        let wp = weight_ptrs.buf.as_ptr();
        let xp = xrot.buf.as_ptr();
        let y0p = y0.buf.as_ptr();
        let y1p = y1.map_or(std::ptr::null_mut::<c_void>(), |y| y.buf.as_ptr());
        for (bucket, present) in present.into_iter().enumerate() {
            if !present {
                continue;
            }
            let k1 = if bucket == 0 {
                &cold
                    .ok_or_else(|| {
                        hip_bridge::HipError::new(
                            0,
                            "mq_i4_moe_gemv_slab: the cold bucket launches but cold_codec is 0 (no cold codec)",
                        )
                    })?
                    .k1
            } else {
                &arch.mq4_k1
            };
            let func = if wide {
                MQ_I4_MOE_GEMV_WIDE_SYMBOLS[usize::from(lanes > 16)][bucket][down as usize]
            } else {
                k1.symbols[down as usize]
            };
            self.ensure_embedded_kernel(k1.module, k1.image, func)?;
            let ids = (union + bucket * s * 4) as *mut c_void;
            let offs = (union + (2 * s + bucket * (s + 1)) * 4) as *mut c_void;
            let slot_ids = (union + (2 * s + 2 * (s + 1) + bucket * s) * 4) as *mut c_void;
            let (mv, kv, uv, cv) = (m as u32, k as u32, s as u32, lanes as u32);
            let (dv, srows, xs, ys) = (
                x_row_div as u32,
                x_src_rows as u32,
                k as u32,
                y_stride as u32,
            );
            let nexp = SLAB_EXPERTS as u32;
            // Metadata lists reserved@92 as argument 17, not implicit struct padding.
            let reserved = 0u32;
            let mut params: [*mut c_void; MQ_I4_MOE_GEMV_KERNARG_COUNT] = [
                &wp as *const _ as *mut c_void,
                &ids as *const _ as *mut c_void,
                &offs as *const _ as *mut c_void,
                &slot_ids as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &y0p as *const _ as *mut c_void,
                &y1p as *const _ as *mut c_void,
                &mv as *const _ as *mut c_void,
                &kv as *const _ as *mut c_void,
                &uv as *const _ as *mut c_void,
                &cv as *const _ as *mut c_void,
                &dv as *const _ as *mut c_void,
                &srows as *const _ as *mut c_void,
                &xs as *const _ as *mut c_void,
                &ys as *const _ as *mut c_void,
                &nexp as *const _ as *mut c_void,
                &reserved as *const _ as *mut c_void,
            ];
            let (grid_m, block) = if wide {
                (mv.div_ceil(8), 256)
            } else {
                (mv, 32)
            };
            self.launch_maybe_blob(func, [grid_m, uv, 1], [block, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(ids);
                b.push_ptr(offs);
                b.push_ptr(slot_ids);
                b.push_ptr(xp);
                b.push_ptr(y0p);
                b.push_ptr(y1p);
                for word in [mv, kv, uv, cv, dv, srows, xs, ys, nexp, 0] {
                    b.push_u32(word);
                }
                b
            })?;
        }
        Ok(())
    }

    /// Split the grouped tile ids `tile_ids` (`tiles` words) by codec with the
    /// table-driven tile mask: `[cold tiles, MQ4 tiles]`, each the source ids with
    /// every tile of an expert of the other codec set to `-1`. `codec_i32` is the
    /// layer's `[512]` i32 codec table of raw codec ids (`0` = MQ4); the masks use ranks
    /// `cold_codec` (gfx1201: 1 = W3 or 3 = E8P; gfx1100: 3 = E8P; gfx1151: 1 = W3) and `0`.
    /// The outputs are views of the slab tile scratch, valid until the next call.
    pub fn qwen4_slab_tile_masks(
        &mut self,
        tile_ids: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
        cold_codec: u8,
    ) -> HipResult<[GpuTensor; 2]> {
        let arch = self.slab_kernels("qwen4_slab_tile_masks")?;
        let cold = self.slab_cold("qwen4_slab_tile_masks", cold_codec)?;
        if tiles == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_slab_tile_masks: no tiles",
            ));
        }
        if self.scratch.qwen4_slab_tiles_scratch_bytes < 2 * tiles * 4
            || self.scratch.qwen4_slab_tiles_scratch.is_none()
        {
            self.reserve_qwen4_slab(tiles)?;
        }
        let base = self.scratch.qwen4_slab_tiles_scratch.as_ref().unwrap();
        let views = [0usize, 1].map(|half| GpuTensor {
            buf: base.byte_view(half * tiles * 4, tiles * 4),
            shape: vec![tiles],
            dtype: DType::Raw,
        });
        // cold codec -> views[0], MQ4 (codec 0) -> views[1].
        self.slab_tile_mask(
            arch,
            tile_ids,
            &views[0],
            codec_i32,
            tiles,
            usize::from(cold.cold_codec),
        )?;
        self.slab_tile_mask(arch, tile_ids, &views[1], codec_i32, tiles, 0)?;
        Ok(views)
    }

    /// One owner mask over the slab's `[512]` codec table: gfx1201 is the EP
    /// [`Gpu::ep_tile_mask`] (unchanged); gfx1100 launches its own certified
    /// build of the same kernel, which the owner-routed EP exchange never reaches
    /// (`ep_tile_mask` itself stays gfx1201-only); gfx1151 launches its hipcc twin
    /// `qwen4_slab_tile_mask` (the typed [`SlabTileMask::Hipcc`] descriptor: one thread
    /// per tile, the same law and ids out).
    fn slab_tile_mask(
        &mut self,
        arch: &'static SlabArchKernels,
        tile_ids: &GpuTensor,
        tile_ids_owned: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
        rank: usize,
    ) -> HipResult<()> {
        if arch.arch == "gfx1201" {
            return self.ep_tile_mask(
                tile_ids,
                tile_ids_owned,
                codec_i32,
                tiles,
                SLAB_EXPERTS,
                rank,
            );
        }
        // Both bodies keep their own refusal text: the gfx1100 embedded mask (`slab tile mask: ..`) and
        // beta's gfx1151 hipcc twin (`qwen4_slab_tile_mask: ..`).
        match arch.tile_mask {
            SlabTileMask::Embedded {
                module,
                image,
                symbol,
            } => {
                let refuse = |what: &str| {
                    Err(hip_bridge::HipError::new(
                        0,
                        &format!("slab tile mask: {what}"),
                    ))
                };
                let word = std::mem::size_of::<i32>();
                if i32::try_from(tiles).is_err() {
                    return refuse("tiles exceed i32");
                }
                if tile_ids.buf.size() < tiles * word || tile_ids_owned.buf.size() < tiles * word {
                    return refuse("tile id buffers hold fewer than `tiles` words");
                }
                if codec_i32.buf.size() < SLAB_EXPERTS * word {
                    return refuse("codec table holds fewer than 512 words");
                }
                if tile_ids.buf.as_ptr() == tile_ids_owned.buf.as_ptr() {
                    return refuse("the masked copy must not alias the source tile ids");
                }
                self.bind_thread()?;
                self.ensure_embedded_kernel(module, image, symbol)?;
                let ids = tile_ids.buf.as_ptr();
                let owned = tile_ids_owned.buf.as_ptr();
                let table = codec_i32.buf.as_ptr();
                let (tiles_i32, experts_i32, rank_i32) =
                    (tiles as i32, SLAB_EXPERTS as i32, rank as i32);
                let mut params = [
                    &ids as *const _ as *mut c_void,
                    &owned as *const _ as *mut c_void,
                    &table as *const _ as *mut c_void,
                    &tiles_i32 as *const _ as *mut c_void,
                    &experts_i32 as *const _ as *mut c_void,
                    &rank_i32 as *const _ as *mut c_void,
                ];
                let timer =
                    crate::profile::begin_timer(&self.hip, "elementwise", symbol, tiles * word * 2);
                let result = self.launch_maybe_blob(
                    symbol,
                    [(tiles as u32).div_ceil(SLAB_TILE_MASK_BLOCK), 1, 1],
                    [SLAB_TILE_MASK_BLOCK, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(ids);
                        b.push_ptr(owned);
                        b.push_ptr(table);
                        b.push_i32(tiles_i32);
                        b.push_i32(experts_i32);
                        b.push_i32(rank_i32);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                result
            }
            SlabTileMask::Hipcc {
                module,
                source,
                symbol,
            } => {
                let (owned, codec) = (tile_ids_owned, rank);
                let refuse = |what: &str| {
                    Err(hip_bridge::HipError::new(
                        0,
                        &format!("qwen4_slab_tile_mask: {what}"),
                    ))
                };
                let word = std::mem::size_of::<i32>();
                if i32::try_from(tiles).is_err() {
                    return refuse("tiles exceed i32");
                }
                if tile_ids.buf.size() < tiles * word || owned.buf.size() < tiles * word {
                    return refuse("tile id buffers hold fewer than `tiles` words");
                }
                if codec_i32.buf.size() < SLAB_EXPERTS * word {
                    return refuse("codec table holds fewer than 512 words");
                }
                if tile_ids.buf.as_ptr() == owned.buf.as_ptr() {
                    return refuse("the masked copy must not alias the source tile ids");
                }
                self.bind_thread()?;
                self.ensure_kernel(module, source, symbol)?;
                let (ids, out, table) = (
                    tile_ids.buf.as_ptr(),
                    owned.buf.as_ptr(),
                    codec_i32.buf.as_ptr(),
                );
                let (tv, ev, cv) = (tiles as i32, SLAB_EXPERTS as i32, codec as i32);
                let mut params = [
                    &ids as *const _ as *mut c_void,
                    &out as *const _ as *mut c_void,
                    &table as *const _ as *mut c_void,
                    &tv as *const _ as *mut c_void,
                    &ev as *const _ as *mut c_void,
                    &cv as *const _ as *mut c_void,
                ];
                let timer =
                    crate::profile::begin_timer(&self.hip, "elementwise", symbol, tiles * word * 2);
                let result = self.launch_maybe_blob(
                    symbol,
                    [(tiles as u32).div_ceil(SLAB_TILE_MASK_BLOCK), 1, 1],
                    [SLAB_TILE_MASK_BLOCK, 1, 1],
                    0,
                    &mut params,
                    || {
                        let mut b = hip_bridge::KernargBlob::new();
                        b.push_ptr(ids);
                        b.push_ptr(out);
                        b.push_ptr(table);
                        b.push_i32(tv);
                        b.push_i32(ev);
                        b.push_i32(cv);
                        b
                    },
                );
                if let Some(t) = timer {
                    t.finish(&self.hip);
                }
                result
            }
        }
    }

    /// K3 down A4 producer of the gfx1100 slab K2 prefill: grouped BF16 SwiGLU rows `h`
    /// `[grouped_rows, 640]` -> the compact flat-slot `block_i4_128` down sidecar
    /// `[5, total_slots]` (record `g * total_slots + sorted[p]`, 72 bytes), padding rows
    /// emit nothing. The row-local `mq_i4_a4_r128_k640_gfx1100` entry (certified builder
    /// module `mq_i4_a4_pm_gfx1100`, 48-byte kernarg, grid `5 * grouped_rows`, block 32);
    /// `k` must be 640. Same handle and ownership as [`Gpu::qwen4_moe_rotate128_i4`]
    /// (which refuses gfx1100). gfx1100 builds the sidecar with K3; gfx1201 and gfx1151 (no K3, `k3` is `None`)
    /// delegate to [`Gpu::qwen4_moe_rotate128_i4`] unchanged, before the K3 lookup.
    pub fn qwen4_moe_rotate128_i4_slab(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<crate::scratch::Int4MmqPrepared> {
        let arch = self.slab_kernels("qwen4_moe_rotate128_i4_slab")?;
        if matches!(arch.arch, "gfx1201" | "gfx1151") {
            return self.qwen4_moe_rotate128_i4(h, sorted, k, grouped_rows, total_slots);
        }
        let Some((module, image, symbol)) = arch.k3 else {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "qwen4_moe_rotate128_i4_slab: no K3 A4 producer on {}",
                    self.arch
                ),
            ));
        };
        if k != SLAB_DOWN_K || grouped_rows == 0 || total_slots == 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "qwen4_moe_rotate128_i4_slab: need k == 640 and nonzero rows",
            ));
        }
        crate::moe::require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i4_slab grouped BF16 rows",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i4_slab sorted slots",
        )?;
        let grid_x = u32::try_from((k / 128) * grouped_rows).map_err(|_| {
            hip_bridge::HipError::new(0, "qwen4_moe_rotate128_i4_slab: grid exceeds u32")
        })?;
        let (pv, lv) = match (u32::try_from(grouped_rows), u32::try_from(total_slots)) {
            (Ok(p), Ok(l)) => (p, l),
            _ => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "qwen4_moe_rotate128_i4_slab: extent overflow",
                ))
            }
        };
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_embedded_kernel(module, image, symbol)?;
        let needed = k / 128 * total_slots * 72;
        if crate::scratch::scratch_will_grow(
            self.scratch.qwen4_moe_down_i4_scratch_bytes,
            self.scratch.qwen4_moe_down_i4_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let reservation = self
            .scratch
            .reserve_qwen4_moe_down_i4(&self.hip, k, total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            symbol,
            grouped_rows * (k * 2 + 4) + needed,
        );
        let result =
            self.launch_maybe_blob(symbol, [grid_x, 1, 1], [32, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_u32(pv);
                b.push_u32(lv);
                b
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(crate::scratch::Int4MmqPrepared::from_reservation(
            reservation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::{envelope::Source, inst::Arch};
    use peacemaker_lift::bundle::SourceCodec;

    fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn arch_table_covers_exactly_gfx1201_gfx1100_and_gfx1151() {
        assert_eq!(
            slab_arch_kernels("gfx1201").map(|k| k.arch),
            Some("gfx1201")
        );
        assert_eq!(
            slab_arch_kernels("gfx1100").map(|k| k.arch),
            Some("gfx1100")
        );
        assert_eq!(
            slab_arch_kernels("gfx1151").map(|k| k.arch),
            Some("gfx1151")
        );
        for other in [
            "gfx942", "gfx906", "gfx1101", "gfx1200", "gfx1100x", "gfx1151x", "GFX1151", "",
        ] {
            assert!(slab_arch_kernels(other).is_none(), "{other}");
        }
    }

    #[test]
    fn gfx1151_table_is_beta_k1_shared_planner_and_hipcc_mask() {
        let k = slab_arch_kernels("gfx1151").unwrap();
        // The shared planner (the same module key and source as gfx1201) and no K3 (the down sidecar
        // delegates to the existing producer).
        assert_eq!(k.plan_module, "qwen4_slab_union_plan");
        assert_eq!((k.plan_src)(), kernels::QWEN4_SLAB_UNION_PLAN_SRC);
        assert!(k.k3.is_none());
        // The real HIP mask descriptor: module/symbol `qwen4_slab_tile_mask`, the hipcc source, not an image.
        let SlabTileMask::Hipcc {
            module,
            source,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1151's tile mask is a hipcc source");
        };
        assert_eq!(
            (module, symbol),
            ("qwen4_slab_tile_mask", "qwen4_slab_tile_mask")
        );
        assert_eq!(source, kernels::QWEN4_SLAB_TILE_MASK_SRC);
        assert!(!source.is_empty());
        // K1: one image for both buckets, MQ4 hot bucket and W3 cold bucket.
        assert_eq!(k.mq4_k1.module, MQ_I4_MOE_GEMV_GFX1151_MODULE);
        assert_eq!(k.mq4_k1.module, "mq_i4_moe_gemv_pm_gfx1151");
        assert_eq!(k.mq4_k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1151);
        assert_eq!(k.mq4_k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[1]);
        let w3 = slab_cold_kernels("gfx1151", 1).unwrap();
        assert_eq!(w3.cold_codec, 1);
        assert_eq!(w3.k1.module, k.mq4_k1.module);
        assert!(std::ptr::eq(w3.k1.image, k.mq4_k1.image));
        assert_eq!(w3.k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[0]);
        for k1 in [&k.mq4_k1, &w3.k1] {
            for symbol in k1.symbols {
                assert!(
                    contains_bytes(k1.image, format!("{symbol}.kd").as_bytes()),
                    "{symbol}.kd missing"
                );
            }
        }
        // E8P has no gfx1151 image or table row; W2 is refused.
        assert!(
            slab_cold_kernels("gfx1151", 3).is_none()
                && slab_cold_k1_metadata("gfx1151", 3).is_none()
        );
        assert!(
            slab_cold_kernels("gfx1151", 2).is_none()
                && slab_cold_k1_metadata("gfx1151", 2).is_none()
        );
    }

    /// The gfx1151 mask is beta's hipcc source (never reinterpreted as an image): its kernel is the
    /// symbol the descriptor names, with the 36-byte ABI (three pointers, i32 tiles/experts/rank), a
    /// 256-thread block, and gfx1151-only guard; gfx1100 never receives it.
    #[test]
    fn gfx1151_tile_mask_source_is_the_beta_hipcc_kernel() {
        let SlabTileMask::Hipcc { source, symbol, .. } =
            slab_arch_kernels("gfx1151").unwrap().tile_mask
        else {
            panic!("gfx1151's tile mask is a hipcc source");
        };
        assert_eq!(SLAB_TILE_MASK_BLOCK, 256);
        assert!(source.contains(&format!("void {symbol}(")));
        assert!(source.contains("__launch_bounds__(256)"));
        assert!(source
            .contains("const int* __restrict__ owner_table, int tiles, int n_experts, int rank)"));
        assert!(source.contains("gfx1151-only"));
        assert!(!source.contains("__gfx1100__"));
        for arch in ["gfx1100", "gfx1201"] {
            assert!(
                matches!(
                    slab_arch_kernels(arch).unwrap().tile_mask,
                    SlabTileMask::Embedded { .. }
                ),
                "{arch}"
            );
        }
    }

    #[test]
    fn gfx1201_table_is_the_historical_constants() {
        let k = slab_arch_kernels("gfx1201").unwrap();
        assert_eq!(k.plan_module, "qwen4_slab_union_plan");
        assert_eq!((k.plan_src)(), kernels::QWEN4_SLAB_UNION_PLAN_SRC);
        assert!(k.k3.is_none());
        let SlabTileMask::Embedded {
            module,
            image,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1201's tile mask is an embedded builder image");
        };
        assert_eq!(module, "ep_tile_mask_pm_gfx1201");
        assert_eq!(symbol, "ep_tile_mask_pm_gfx1201");
        assert_eq!(image, kernels::EP_TILE_MASK_PM_GFX1201);
        // The hot MQ4 bucket and the W3 cold bucket: the historical K1 module, image and symbols.
        assert_eq!(k.mq4_k1.module, MQ_I4_MOE_GEMV_MODULE);
        assert_eq!(k.mq4_k1.module, "mq_i4_moe_gemv_pm_gfx1201");
        assert_eq!(k.mq4_k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(k.mq4_k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[1]);
        let w3 = slab_cold_kernels("gfx1201", 1).unwrap();
        assert_eq!(w3.cold_codec, 1);
        assert_eq!(w3.k1.module, MQ_I4_MOE_GEMV_MODULE);
        assert_eq!(w3.k1.image, kernels::MQ_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(w3.k1.symbols, MQ_I4_MOE_GEMV_SYMBOLS[0]);
        // E8P cold: its own module and blob (the base's launcher constants).
        let e8p = slab_cold_kernels("gfx1201", 3).unwrap();
        assert_eq!(e8p.cold_codec, 3);
        assert_eq!(e8p.k1.module, "e8p_i4_moe_gemv_pm_gfx1201");
        assert_eq!(e8p.k1.image, kernels::E8P_I4_MOE_GEMV_PM_GFX1201);
        assert_eq!(
            e8p.k1.symbols,
            ["e8p_i4_moe_gemv_r256_gate_up", "e8p_i4_moe_gemv_r128_down"]
        );
        for k1 in [&k.mq4_k1, &w3.k1, &e8p.k1] {
            for symbol in k1.symbols {
                assert!(
                    contains_bytes(k1.image, format!("{symbol}.kd").as_bytes()),
                    "{symbol}.kd missing"
                );
            }
        }
    }

    #[test]
    fn gfx1100_table_names_and_images() {
        let k = slab_arch_kernels("gfx1100").unwrap();
        assert_eq!(k.plan_module, "qwen4_slab_union_plan_gfx1100");
        assert_eq!((k.plan_src)(), kernels::qwen4_slab_union_plan_gfx1100_src());
        // The assembled source is the shared planner file minus ONLY its exact arch guard, with no
        // fake compiler arch macro and no wave-sensitive construct in the body.
        let assembled = (k.plan_src)();
        let original = kernels::QWEN4_SLAB_UNION_PLAN_SRC;
        assert_eq!(
            original
                .matches(kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD)
                .count(),
            1
        );
        assert_eq!(
            kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD,
            "#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__) && !defined(__gfx1151__)\n\
             #error \"qwen4_slab_union_plan.hip is gfx1201/gfx1151-only\"\n\
             #endif\n"
        );
        assert_eq!(
            assembled,
            original.replacen(kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD, "", 1),
        );
        assert_eq!(
            original.len() - assembled.len(),
            kernels::QWEN4_SLAB_UNION_PLAN_ARCH_GUARD.len()
        );
        for forbidden in [
            "#define __gfx",
            "#undef __gfx",
            "__shfl",
            "__ballot",
            "__activemask",
            "warpSize",
            "__builtin_amdgcn",
            "__gfx",
        ] {
            assert!(
                !assembled.contains(forbidden),
                "{forbidden} in the assembled planner"
            );
        }
        // gfx1100's hot MQ4 bucket reads the `mq4sym` entries of the NEW MQ4-only K1 module.
        let hot = &k.mq4_k1;
        assert_eq!(hot.module, "mq_i4_moe_gemv_mq4only_pm_gfx1100");
        assert_eq!(hot.image, kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100);
        assert_eq!(
            hot.symbols,
            [
                "mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100",
                "mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100"
            ]
        );
        // The image names no W2 (`mq2i4`) entry, no qualification (`_acc`) twin, and no
        // `mq_i4_moe_gemv_*` name other than the two served symbols (checked below).
        // (A bare `_acc` would match the code-object metadata key `.actual_access`.)
        for needle in [&b"mq2i4"[..], b"_acc.kd", b"_acc\0"] {
            assert!(
                !contains_bytes(hot.image, needle),
                "{needle:?} in the MQ4-only image"
            );
        }
        for symbol in hot.symbols {
            assert!(
                contains_bytes(hot.image, symbol.as_bytes()),
                "{symbol} missing from its K1 image"
            );
            assert!(
                contains_bytes(hot.image, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
            assert!(!contains_bytes(
                kernels::MQ_I4_MOE_GEMV_PM_GFX1201,
                symbol.as_bytes()
            ));
        }
        let prefix = b"mq_i4_moe_gemv_";
        for at in 0..hot.image.len().saturating_sub(prefix.len()) {
            if hot.image[at..].starts_with(prefix) {
                assert!(
                    hot.symbols
                        .iter()
                        .any(|s| hot.image[at..].starts_with(s.as_bytes())),
                    "the MQ4-only image names an unserved mq_i4_moe_gemv_* entry at {at}"
                );
            }
        }
        // E8P K1 (qualified image, two contracts): the gfx1100 cold bucket of codec 3 layers.
        let e8p = slab_cold_k1_metadata("gfx1100", 3).unwrap();
        assert_eq!(e8p.cold_codec, 3);
        assert_eq!(e8p.k1.module, "e8p_i4_moe_gemv_pm_gfx1100");
        assert_eq!(e8p.k1.image, kernels::E8P_I4_MOE_GEMV_PM_GFX1100);
        assert_eq!(
            e8p.k1.symbols,
            [
                "e8p_i4_moe_gemv_r256_gate_up_pm_gfx1100",
                "e8p_i4_moe_gemv_r128_down_pm_gfx1100"
            ]
        );
        for symbol in e8p.k1.symbols {
            assert!(
                contains_bytes(e8p.k1.image, format!("{symbol}.kd").as_bytes()),
                "{symbol}.kd missing"
            );
            assert!(!symbol.contains("mq2i4") && !symbol.ends_with("_acc"));
            // Each symbol is its own contract: kernels/e8p_i4_moe_gemv.gfx1100.<kind>.contract.json.
            let kind = if symbol.contains("r256_gate_up") {
                "r256_gate_up"
            } else {
                "r128_down"
            };
            let contract =
                std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                    format!("../../kernels/e8p_i4_moe_gemv.gfx1100.{kind}.contract.json"),
                ))
                .unwrap();
            assert!(
                contract.contains(&format!("\"symbol\": \"{symbol}\"")),
                "{kind} contract symbol"
            );
            assert!(
                contract.contains("\"group_segment_fixed_bytes\": 1536"),
                "{kind} contract static LDS"
            );
            assert!(
                contract.contains("\"launch_dynamic_lds_bytes\": 0"),
                "{kind} contract dynamic LDS"
            );
        }
        assert!(!e8p.k1.module.contains("mq_i4") && !e8p.k1.module.contains("gfx1201"));
        assert!(!contains_bytes(e8p.k1.image, b"mq_i4_moe_gemv"));
        // The admission matrix admits codec 3 on gfx1100 and runtime selection is the very same
        // table object; W3 (1) stays refused there although it is a gfx1201 codec.
        assert!(crate::qwen4_slab_codec_admitted("gfx1100", 3));
        assert!(std::ptr::eq(
            slab_cold_kernels("gfx1100", 3).unwrap(),
            slab_cold_k1_metadata("gfx1100", 3).unwrap()
        ));
        assert!(slab_cold_kernels("gfx1100", 1).is_none());
        let (module, image, symbol) = k.k3.unwrap();
        assert_eq!(
            (module, symbol),
            ("mq_i4_a4_pm_gfx1100", "mq_i4_a4_r128_k640_gfx1100")
        );
        assert!(contains_bytes(image, format!("{symbol}.kd").as_bytes()));
        let SlabTileMask::Embedded {
            module,
            image,
            symbol,
        } = k.tile_mask
        else {
            panic!("gfx1100's tile mask is an embedded builder image");
        };
        assert_eq!(
            (module, symbol),
            ("ep_tile_mask_pm_gfx1100", "ep_tile_mask_pm_gfx1100")
        );
        assert!(contains_bytes(image, format!("{symbol}.kd").as_bytes()));
        assert!(!contains_bytes(
            kernels::EP_TILE_MASK_PM_GFX1201,
            symbol.as_bytes()
        ));
    }

    #[test]
    fn cold_tables_are_keyed_by_arch_and_cold_codec_and_refuse_w2() {
        for (arch, codec, served) in [
            ("gfx1201", 1u8, true),
            ("gfx1201", 3, true),
            ("gfx1201", 0, false),
            ("gfx1201", 2, false),
            ("gfx1201", 4, false),
            // gfx1100 serves the E8P cold bucket (K1 `slab_cold_k1_metadata` AND the admission
            // matrix); W3 and W2 stay refused there.
            ("gfx1100", 3, true),
            ("gfx1100", 0, false),
            ("gfx1100", 1, false),
            ("gfx1100", 2, false),
            // gfx1151 serves the W3 cold bucket (K1 metadata AND the admission matrix); E8P and W2
            // stay refused there and codec 0 is never cold.
            ("gfx1151", 1, true),
            ("gfx1151", 0, false),
            ("gfx1151", 2, false),
            ("gfx1151", 3, false),
            ("gfx1151", 4, false),
        ] {
            let table = slab_cold_kernels(arch, codec);
            assert_eq!(table.is_some(), served, "{arch} cold codec {codec}");
            if let Some(table) = table {
                assert_eq!(table.cold_codec, codec);
                assert!(crate::qwen4_slab_codec_admitted(arch, codec));
            }
        }
    }

    #[test]
    fn cold_k1_metadata_is_not_permission() {
        // Metadata inventory: exactly the historical gfx1201 pair, the qualified gfx1100 E8P K1 and the
        // gfx1151 W3 K1.
        let mut available = Vec::new();
        for arch in [
            "gfx1201", "gfx1100", "gfx1151", "gfx942", "gfx906", "gfx1101", "",
        ] {
            for codec in 0u8..=255 {
                if let Some(table) = slab_cold_k1_metadata(arch, codec) {
                    assert_eq!(table.cold_codec, codec);
                    available.push((arch, codec));
                }
            }
        }
        assert_eq!(
            available,
            [
                ("gfx1201", 1),
                ("gfx1201", 3),
                ("gfx1100", 3),
                ("gfx1151", 1)
            ]
        );
        // Runtime selection = metadata AND the shared admission matrix (what
        // `Gpu::ensure_qwen4_slab_kernels` and `Gpu::mq_i4_moe_gemv_slab` consult): every tabled
        // entry is admitted today and served by the very same table object, while no other
        // (arch, codec) pair is admitted or served.
        for (arch, codec) in available {
            let admitted = crate::qwen4_slab_codec_admitted(arch, codec);
            assert!(admitted, "{arch} codec {codec}");
            let selected = slab_cold_kernels(arch, codec);
            assert_eq!(selected.is_some(), admitted, "{arch} codec {codec}");
            if let Some(selected) = selected {
                assert!(std::ptr::eq(
                    selected,
                    slab_cold_k1_metadata(arch, codec).unwrap()
                ));
            }
        }
        assert!(
            slab_cold_kernels("gfx1100", 1).is_none() && slab_cold_kernels("gfx1100", 2).is_none()
        );
        // An MQ4-only layer's preload set never names an E8P module: the hot K1 module is its own
        // image and none of its symbols live in the E8P image.
        let hot = &slab_arch_kernels("gfx1100").unwrap().mq4_k1;
        let e8p = &slab_cold_k1_metadata("gfx1100", 3).unwrap().k1;
        assert_ne!(hot.module, e8p.module);
        assert!(!std::ptr::eq(hot.image, e8p.image));
        for symbol in hot.symbols {
            assert!(!contains_bytes(e8p.image, symbol.as_bytes()));
        }
    }

    /// The wide `_c16/_c32` twins live only in the shared gfx1151/gfx1201 MQ4/W3 K1 images: gfx1100 and the
    /// E8P cold bucket never select them, so those layers keep the unchanged <=8-lane entries.
    #[test]
    fn wide_twins_exist_only_for_the_shared_mq4_w3_k1_images() {
        for arch in ["gfx1151", "gfx1201"] {
            let k = slab_arch_kernels(arch).unwrap();
            assert!(slab_wide_capable(k, None), "{arch}: MQ4-only layer");
            assert!(
                slab_wide_capable(k, slab_cold_kernels(arch, 1)),
                "{arch}: W3 cold layer"
            );
            for symbol in MQ_I4_MOE_GEMV_WIDE_SYMBOLS.iter().flatten().flatten() {
                assert!(
                    contains_bytes(k.mq4_k1.image, symbol.as_bytes()),
                    "{arch}: shared image exports {symbol}"
                );
            }
        }
        let e8p = slab_cold_kernels("gfx1201", 3).unwrap();
        assert!(!slab_wide_capable(
            slab_arch_kernels("gfx1201").unwrap(),
            Some(e8p)
        ));
        let gfx1100 = slab_arch_kernels("gfx1100").unwrap();
        assert!(!slab_wide_capable(gfx1100, None));
        assert!(!slab_wide_capable(gfx1100, slab_cold_kernels("gfx1100", 3)));
        assert!(slab_cold_kernels("gfx1151", 3).is_none());
        for symbol in MQ_I4_MOE_GEMV_WIDE_SYMBOLS.iter().flatten().flatten() {
            assert!(!contains_bytes(
                kernels::MQ_I4_MOE_GEMV_MQ4ONLY_PM_GFX1100,
                symbol.as_bytes()
            ));
            assert!(!contains_bytes(
                kernels::E8P_I4_MOE_GEMV_PM_GFX1201,
                symbol.as_bytes()
            ));
        }
    }

    #[test]
    fn bucket_of_codec_on_follows_admission_with_cold_first() {
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 1), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 3), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1201", 2), None);
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 1), None);
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 2), None);
        // The gfx1100 E8P cold bucket is a served bucket 0 (cold = 0, MQ4 = 1).
        assert_eq!(slab_bucket_of_codec_on("gfx1100", 3), Some(0));
        // gfx1151 serves MQ4 (hot bucket 1) and W3 (cold bucket 0); W2 and E8P stay refused there.
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 0), Some(1));
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 1), Some(0));
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 2), None);
        assert_eq!(slab_bucket_of_codec_on("gfx1151", 3), None);
        // The historical gfx1201 byte mapping is unchanged.
        assert_eq!((slab_bucket_of_codec(1), slab_bucket_of_codec(0)), (0, 1));
    }

    #[test]
    fn k1_kernel_params_match_embedded_metadata() {
        let mut mismatches = Vec::new();
        let mut checked = 0;
        for (arch, bytes) in [
            (Arch::Gfx1151, kernels::MQ_I4_MOE_GEMV_PM_GFX1151),
            (Arch::Gfx1201, kernels::MQ_I4_MOE_GEMV_PM_GFX1201),
        ] {
            let (_, images) = Source::read(bytes, arch).expect("embedded K1 metadata must parse");
            for symbol in MQ_I4_MOE_GEMV_SYMBOLS
                .iter()
                .flatten()
                .chain(MQ_I4_MOE_GEMV_WIDE_SYMBOLS.iter().flatten().flatten())
            {
                let image = images
                    .iter()
                    .find(|image| image.name == *symbol)
                    .unwrap_or_else(|| panic!("{arch:?}: missing K1 symbol {symbol}"));
                let metadata = &image.metadata.parsed;
                assert_eq!(metadata.kernarg_segment_size, 96, "{arch:?}: {symbol}");
                let reserved = metadata.args.last().expect("K1 arguments");
                assert_eq!(
                    (
                        reserved.name.as_str(),
                        reserved.offset,
                        reserved.size,
                        reserved.value_kind.as_str()
                    ),
                    ("reserved", 92, 4, "by_value"),
                    "{arch:?}: {symbol}"
                );
                let explicit = metadata
                    .args
                    .iter()
                    .filter(|arg| !arg.value_kind.starts_with("hidden_"))
                    .count();
                if explicit != MQ_I4_MOE_GEMV_KERNARG_COUNT {
                    mismatches.push(format!(
                        "{arch:?}: {symbol}: kernelParams={MQ_I4_MOE_GEMV_KERNARG_COUNT}, declared={explicit}"
                    ));
                }
                checked += 1;
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
        println!("checked {checked} embedded K1 entries against kernelParams");
    }
}
