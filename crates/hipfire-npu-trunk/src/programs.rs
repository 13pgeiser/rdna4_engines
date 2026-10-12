// SPDX-License-Identifier: Apache-2.0
//! G80 program construction: one deployment image + command stream per trunk shape, CPU only (no device is opened).
//!
//! Every submit of a trunk program multiplies one 256-row-wave-padded A (`rows_max` rounded up to 256 rows, `K = 2560`) by
//! ONE resident 1280-column B panel with exact int32 accumulation (`Epilogue::I32`, `Control::Fast`). A shape therefore
//! needs `panel_count` submits per row block; the program itself is independent of the panel (B is an argument), so
//! the per-shape design differs between shapes only through the bookkeeping recorded here.

use pm_npu::kernels::gemm_array::ArrayDesign;
use pm_npu::kernels::gemm_core::{Control, Epilogue};
use pm_npu::kernels::gemm_g80::design_g80;

use crate::shapes::{EpilogueContract, FamilyMask, Shape, PANEL_COLS, TRUNK_K};

/// Rows of one G80 M wave: every program is built for a whole number of these.
const WAVE_ROWS: usize = 256;
/// Largest row count a G80 design accepts (256 waves of 256 rows).
const ROWS_LIMIT: usize = 256 * WAVE_ROWS;

/// The program of one trunk shape.
pub struct ShapeProgram {
    pub shape: Shape,
    /// G80 design (PDI + TXN + argument layout), `M = rows_max` rounded, `N = 1280`, `K = 2560`, `Epilogue::I32`, `Control::Fast`.
    pub design: ArrayDesign,
    /// 1280-column B panels (submits per row block) of the selected families of this shape; 0 when none is selected.
    pub panel_count: usize,
    /// `panel_count * 1280`: fused columns including zero padding.
    pub padded_cols: usize,
    /// Bytes of `[A, B, C]` of ONE submit (`design.args`).
    pub arg_bytes: [usize; 3],
    /// Host bytes read by ONE submit (packed A + one packed B panel).
    pub host_bytes_read: u64,
    /// Host bytes written by ONE submit (the packed int32 C arena).
    pub host_bytes_written: u64,
}

/// The programs of both shapes for one epilogue contract.
pub struct NpuPrograms {
    pub contract: EpilogueContract,
    /// Rows the programs are built for: the requested maximum rounded up to a whole 256-row wave.
    pub rows_max: usize,
    /// Indexed by [`Shape::index`] in [`Shape::ALL`] order.
    pub per_shape: [ShapeProgram; 2],
}

impl NpuPrograms {
    /// The program of `shape`.
    pub fn get(&self, shape: Shape) -> &ShapeProgram {
        &self.per_shape[shape.index()]
    }
}

impl std::fmt::Debug for ShapeProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeProgram")
            .field("shape", &self.shape.index())
            .field("panel_count", &self.panel_count)
            .field("padded_cols", &self.padded_cols)
            .field("arg_bytes", &self.arg_bytes)
            .field("host_bytes_read", &self.host_bytes_read)
            .field("host_bytes_written", &self.host_bytes_written)
            .field("pdi_bytes", &self.design.pdi.len())
            .field("txn_bytes", &self.design.insts.len())
            .finish()
    }
}

impl std::fmt::Debug for NpuPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NpuPrograms")
            .field("rows_max", &self.rows_max)
            .field("per_shape", &self.per_shape)
            .finish()
    }
}

/// Programs for every family of both shapes.
pub fn build_programs(contract: EpilogueContract, rows_max: usize) -> Result<NpuPrograms, String> {
    build_programs_for_families(contract, rows_max, FamilyMask::ALL)
}

/// Programs for the families in `families`: the panel counts and padded columns follow
/// [`Shape::column_map`] of the mask, so they match the panels `pack_shape` produces for the same mask.
/// `Err` for an unsupported contract, an empty mask, or `rows_max` outside `1..=65536`.
pub fn build_programs_for_families(
    contract: EpilogueContract,
    rows_max: usize,
    families: FamilyMask,
) -> Result<NpuPrograms, String> {
    contract.ensure_supported()?;
    let epilogue = match contract {
        EpilogueContract::I32Fold => Epilogue::I32,
        EpilogueContract::Bf16Direct => {
            return Err(contract
                .ensure_supported()
                .expect_err("Bf16Direct must stay unsupported until its program exists"))
        }
    };
    if !(1..=ROWS_LIMIT).contains(&rows_max) {
        return Err(format!("npu-trunk: rows_max {rows_max} outside 1..={ROWS_LIMIT}"));
    }
    if families.bits() == 0 {
        return Err("npu-trunk: no families selected, nothing to program".to_string());
    }
    let rows = rows_max.div_ceil(WAVE_ROWS) * WAVE_ROWS;
    let program = |shape: Shape| {
        let map = shape.column_map(families);
        let design = design_g80(rows, PANEL_COLS, TRUNK_K, epilogue, Control::Fast);
        let arg_bytes = [design.args[0].bytes, design.args[1].bytes, design.args[2].bytes];
        ShapeProgram {
            shape,
            panel_count: map.panel_count(),
            padded_cols: map.padded_cols(),
            arg_bytes,
            host_bytes_read: design.host_bytes_read(),
            host_bytes_written: design.host_bytes_written(),
            design,
        }
    };
    Ok(NpuPrograms {
        contract,
        rows_max: rows,
        per_shape: [program(Shape::ALL[0]), program(Shape::ALL[1])],
    })
}
