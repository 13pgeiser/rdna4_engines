// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sealed expert-slab (ctl3-hot) GPU entries, gfx1201 and gfx1151.
//!
//! * K1 decode: [`Gpu::qwen4_slab_union_plan`] turns the routed ids into the
//!   fixed-capacity per-codec-bucket lane-union descriptor the four
//!   `mq_i4_moe_gemv_*` symbols read (the data-dependent bucket counts stay on
//!   the device; unused bucket tails are empty experts the kernels skip), and
//!   [`Gpu::mq_i4_moe_gemv_slab`] launches a projection over the non-empty
//!   buckets. Both go through `launch_maybe_blob`, so they are recorded by the
//!   replay recorder and captured by graphs like every other route launch.
//! * K2 prefill: [`Gpu::qwen4_slab_tile_masks`] splits the grouped tile ids by
//!   codec with the table-driven mask: the builder `ep_tile_mask_pm_gfx1201`, or on
//!   gfx1151 its hipcc twin `qwen4_slab_tile_mask` (same law, same ids out).

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemv::WideTwinFamily;
use crate::kernels;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Module keys of the embedded K1 certified builder images, `[gfx1151, gfx1201]`.
const MQ_I4_MOE_GEMV_MODULES: [&str; 2] = ["mq_i4_moe_gemv_pm_gfx1151", "mq_i4_moe_gemv_pm_gfx1201"];
/// The embedded K1 images, same order.
const MQ_I4_MOE_GEMV_IMAGES: [&[u8]; 2] = [kernels::MQ_I4_MOE_GEMV_PM_GFX1151, kernels::MQ_I4_MOE_GEMV_PM_GFX1201];

/// K1 symbols in bucket order: `[codec][kind]`, codec 0 = W3 (`MQ3I4`),
/// 1 = MQ4 (`Mq4Sym`), kind 0 = gate/up (R256), 1 = down (R128).
pub const MQ_I4_MOE_GEMV_SYMBOLS: [[&str; 2]; 2] = [
    ["mq_i4_moe_gemv_mq3i4_r256_gate_up", "mq_i4_moe_gemv_mq3i4_r128_down"],
    ["mq_i4_moe_gemv_mq4sym_r256_gate_up", "mq_i4_moe_gemv_mq4sym_r128_down"],
];

/// Wide K1 entries, `[capacity (16/32)][codec][kind]`; the 88-byte ABI is unchanged.
pub const MQ_I4_MOE_GEMV_WIDE_SYMBOLS: [[[&str; 2]; 2]; 2] = [
    [
        ["mq_i4_moe_gemv_mq3i4_r256_gate_up_c16", "mq_i4_moe_gemv_mq3i4_r128_down_c16"],
        ["mq_i4_moe_gemv_mq4sym_r256_gate_up_c16", "mq_i4_moe_gemv_mq4sym_r128_down_c16"],
    ],
    [
        ["mq_i4_moe_gemv_mq3i4_r256_gate_up_c32", "mq_i4_moe_gemv_mq3i4_r128_down_c32"],
        ["mq_i4_moe_gemv_mq4sym_r256_gate_up_c32", "mq_i4_moe_gemv_mq4sym_r128_down_c32"],
    ],
];

const PLAN_SYMBOL: &str = "qwen4_slab_union_plan";
const PLAN_MODULE: &str = "qwen4_slab_union_plan";
const MASK_SYMBOL: &str = "qwen4_slab_tile_mask";
const MASK_MODULE: &str = "qwen4_slab_tile_mask";
/// Threads (tiles) per workgroup of the gfx1151 mask kernel.
const MASK_BLOCK: u32 = 256;

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

/// Bucket of codec byte `0` (MQ4) and `1` (W3) in the K1 order: W3 first.
pub const fn slab_bucket_of_codec(codec: u8) -> usize {
    if codec == 1 {
        0
    } else {
        1
    }
}

/// How a layer's experts map to K1 buckets.
#[derive(Clone, Copy)]
pub enum SlabBuckets<'a> {
    /// Every expert has the same codec (bucket 0 = W3, 1 = MQ4): one bucket.
    Uniform(usize),
    /// Per-expert codec bytes (`[512]` u8): both buckets.
    Table(&'a GpuTensor),
}

impl Gpu {
    /// Index of this device in the slab route's per-arch tables (gfx1151,
    /// gfx1201); `None` on every other architecture.
    fn slab_arch_index(&self) -> Option<usize> {
        match self.arch.as_str() {
            "gfx1151" => Some(0),
            "gfx1201" => Some(1),
            _ => None,
        }
    }

    /// Whether the sealed expert-slab route's kernels exist for this device.
    pub fn qwen4_slab_arch(&self) -> bool {
        self.slab_arch_index().is_some()
    }

    /// Whether a wide K1 launch is admitted and the developer opt-in is enabled.
    /// Allocation capacity alone never widens the shared row-exact admission.
    pub fn qwen4_slab_wide_twin(&self, lanes: usize) -> bool {
        self.wide_twin_cap(WideTwinFamily::K1, lanes).is_some()
    }

    /// Load the planner, the mask (gfx1151) and the four K1 symbols (call before graph capture).
    pub fn ensure_qwen4_slab_kernels(&mut self) -> HipResult<()> {
        let arch = self.slab_require_arch("ensure_qwen4_slab_kernels")?;
        self.bind_thread()?;
        self.ensure_kernel(PLAN_MODULE, kernels::QWEN4_SLAB_UNION_PLAN_SRC, PLAN_SYMBOL)?;
        if arch == 0 {
            self.ensure_kernel(MASK_MODULE, kernels::QWEN4_SLAB_TILE_MASK_SRC, MASK_SYMBOL)?;
        }
        for symbol in MQ_I4_MOE_GEMV_SYMBOLS.iter().flatten() {
            self.ensure_embedded_kernel(MQ_I4_MOE_GEMV_MODULES[arch], MQ_I4_MOE_GEMV_IMAGES[arch], symbol)?;
        }
        if WideTwinFamily::K1.enabled() {
            for symbol in MQ_I4_MOE_GEMV_WIDE_SYMBOLS.iter().flatten().flatten() {
                self.ensure_embedded_kernel(MQ_I4_MOE_GEMV_MODULES[arch], MQ_I4_MOE_GEMV_IMAGES[arch], symbol)?;
            }
        }
        Ok(())
    }

    fn slab_require_arch(&self, what: &str) -> HipResult<usize> {
        self.slab_arch_index().ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!("{what}: the sealed expert-slab route is gfx1151/gfx1201-only, got {}", self.arch),
            )
        })
    }

    /// Reserve the slab route's scratch (union descriptor for `SLAB_MAX_LANES`,
    /// and `tiles` masked tile ids per codec) before any capture.
    pub fn reserve_qwen4_slab(&mut self, tiles: usize) -> HipResult<()> {
        self.slab_require_arch("reserve_qwen4_slab")?;
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
        self.scratch.reserve_qwen4_slab(&self.hip, SLAB_MAX_LANES, tiles)
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
        self.slab_require_arch("qwen4_slab_union_plan")?;
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(0, "qwen4_slab_union_plan: lanes must be 1..=32"));
        }
        let slots = SLAB_TOP_K * lanes;
        if topk.buf.size() < slots * 4 {
            return Err(hip_bridge::HipError::new(0, "qwen4_slab_union_plan: topk holds fewer than 10 * lanes ids"));
        }
        let (codec_ptr, uniform) = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => (std::ptr::null_mut::<c_void>(), bucket as i32),
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(0, "qwen4_slab_union_plan: bucket must be 0 or 1"))
            }
            SlabBuckets::Table(table) => {
                if table.buf.size() < SLAB_EXPERTS {
                    return Err(hip_bridge::HipError::new(0, "qwen4_slab_union_plan: codec table holds fewer than 512 bytes"));
                }
                (table.buf.as_ptr(), -1)
            }
        };
        self.bind_thread()?;
        self.ensure_kernel(PLAN_MODULE, kernels::QWEN4_SLAB_UNION_PLAN_SRC, PLAN_SYMBOL)?;
        if self.scratch.qwen4_slab_union_scratch_bytes < crate::scratch::qwen4_slab_union_words(SLAB_MAX_LANES) * 4
            || self.scratch.qwen4_slab_union_scratch.is_none()
        {
            self.reserve_qwen4_slab(0)?;
        }
        let out = self.slab_union_base()?;
        let tp = topk.buf.as_ptr();
        let sv = slots as i32;
        let mut params = [
            &tp as *const _ as *mut c_void,
            &codec_ptr as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &sv as *const _ as *mut c_void,
            &uniform as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(PLAN_SYMBOL, [1, 1, 1], [512, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(codec_ptr);
            b.push_ptr(out);
            b.push_i32(sv);
            b.push_i32(uniform);
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
    /// the projection. Output rows of every slot are written exactly once.
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
    ) -> HipResult<()> {
        let arch = self.slab_require_arch("mq_i4_moe_gemv_slab")?;
        if lanes == 0 || lanes > SLAB_MAX_LANES {
            return Err(hip_bridge::HipError::new(0, "mq_i4_moe_gemv_slab: lanes must be 1..=32"));
        }
        let wide = self.qwen4_slab_wide_twin(lanes);
        if lanes > SLAB_NARROW_MAX_LANES && !wide {
            return Err(hip_bridge::HipError::new(
                0,
                "mq_i4_moe_gemv_slab: wide lanes require the K1 opt-in and shared architecture admission",
            ));
        }
        let slots = SLAB_TOP_K * lanes;
        let (m, k, x_row_div, x_src_rows, y_stride) = if down {
            (SLAB_DOWN_M, SLAB_DOWN_K, 1usize, slots, SLAB_DOWN_M)
        } else {
            (SLAB_GATE_UP_M, SLAB_GATE_UP_K, SLAB_TOP_K, lanes, SLAB_GATE_UP_M / 2)
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
            return Err(hip_bridge::HipError::new(0, "mq_i4_moe_gemv_slab: operand smaller than the launch extent"));
        }
        let present: [bool; 2] = match buckets {
            SlabBuckets::Uniform(bucket) if bucket < 2 => [bucket == 0, bucket == 1],
            SlabBuckets::Uniform(_) => {
                return Err(hip_bridge::HipError::new(0, "mq_i4_moe_gemv_slab: bucket must be 0 or 1"))
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
            let func = if wide {
                MQ_I4_MOE_GEMV_WIDE_SYMBOLS[usize::from(lanes > 16)][bucket][down as usize]
            } else {
                MQ_I4_MOE_GEMV_SYMBOLS[bucket][down as usize]
            };
            self.ensure_embedded_kernel(MQ_I4_MOE_GEMV_MODULES[arch], MQ_I4_MOE_GEMV_IMAGES[arch], func)?;
            let ids = (union + bucket * s * 4) as *mut c_void;
            let offs = (union + (2 * s + bucket * (s + 1)) * 4) as *mut c_void;
            let slot_ids = (union + (2 * s + 2 * (s + 1) + bucket * s) * 4) as *mut c_void;
            let (mv, kv, uv, cv) = (m as u32, k as u32, s as u32, lanes as u32);
            let (dv, srows, xs, ys) = (x_row_div as u32, x_src_rows as u32, k as u32, y_stride as u32);
            let mut params = [
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
            ];
            let (grid_m, block) = if wide { (mv.div_ceil(8), 256) } else { (mv, 32) };
            self.launch_maybe_blob(func, [grid_m, uv, 1], [block, 1, 1], 0, &mut params, || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(ids);
                b.push_ptr(offs);
                b.push_ptr(slot_ids);
                b.push_ptr(xp);
                b.push_ptr(y0p);
                b.push_ptr(y1p);
                for word in [mv, kv, uv, cv, dv, srows, xs, ys] {
                    b.push_u32(word);
                }
                b
            })?;
        }
        Ok(())
    }

    /// Split the grouped tile ids `tile_ids` (`tiles` words) by codec with the
    /// table-driven tile mask: `[W3 tiles, MQ4 tiles]`, each the source ids with
    /// every tile of an expert of the other codec set to `-1`. `codec_i32` is the
    /// layer's `[512]` i32 codec table (`0` = MQ4, `1` = W3). The outputs are
    /// views of the slab tile scratch, valid until the next call.
    pub fn qwen4_slab_tile_masks(
        &mut self,
        tile_ids: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
    ) -> HipResult<[GpuTensor; 2]> {
        self.slab_require_arch("qwen4_slab_tile_masks")?;
        if tiles == 0 {
            return Err(hip_bridge::HipError::new(0, "qwen4_slab_tile_masks: no tiles"));
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
        // codec 1 (W3) -> views[0], codec 0 (MQ4) -> views[1].
        self.slab_tile_mask(tile_ids, &views[0], codec_i32, tiles, 1)?;
        self.slab_tile_mask(tile_ids, &views[1], codec_i32, tiles, 0)?;
        Ok(views)
    }

    /// One codec mask: `owned[t] = tile_ids[t]` when `0 <= tile_ids[t] < 512` and
    /// `codec_i32[tile_ids[t]] == codec`, else `-1`; `tile_ids` is never written.
    /// gfx1201: the certified builder kernel behind [`Gpu::ep_tile_mask`]; gfx1151: the
    /// hipcc twin `qwen4_slab_tile_mask` (one thread per tile, the same law and bytes).
    fn slab_tile_mask(
        &mut self,
        tile_ids: &GpuTensor,
        owned: &GpuTensor,
        codec_i32: &GpuTensor,
        tiles: usize,
        codec: usize,
    ) -> HipResult<()> {
        if self.arch == "gfx1201" {
            return self.ep_tile_mask(tile_ids, owned, codec_i32, tiles, SLAB_EXPERTS, codec);
        }
        let refuse = |what: &str| Err(hip_bridge::HipError::new(0, &format!("qwen4_slab_tile_mask: {what}")));
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
        self.ensure_kernel(MASK_MODULE, kernels::QWEN4_SLAB_TILE_MASK_SRC, MASK_SYMBOL)?;
        let (ids, out, table) = (tile_ids.buf.as_ptr(), owned.buf.as_ptr(), codec_i32.buf.as_ptr());
        let (tv, ev, cv) = (tiles as i32, SLAB_EXPERTS as i32, codec as i32);
        let mut params = [
            &ids as *const _ as *mut c_void,
            &out as *const _ as *mut c_void,
            &table as *const _ as *mut c_void,
            &tv as *const _ as *mut c_void,
            &ev as *const _ as *mut c_void,
            &cv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", MASK_SYMBOL, tiles * word * 2);
        let result = self.launch_maybe_blob(
            MASK_SYMBOL,
            [(tiles as u32).div_ceil(MASK_BLOCK), 1, 1],
            [MASK_BLOCK, 1, 1],
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
