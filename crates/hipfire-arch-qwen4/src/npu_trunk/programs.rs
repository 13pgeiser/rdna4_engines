// SPDX-License-Identifier: Apache-2.0
//! G80 program construction: one deployment image + command stream per trunk shape, CPU only (no device is opened).
//!
//! Every submit of a trunk program multiplies one 256-row-wave-padded A (`rows_max` rounded up to 256 rows, `K = 2560`) by
//! ONE resident 1280-column B panel, with exact int32 accumulation and either an I32 drain or integer BF16 epilogue.
//! A shape needs `panel_count` submits per row block; the program is independent of the panel (B is an argument).
//! Row-major programs differ only in their fused C row pitch; native-tile programs share a design.

use pm_npu::kernels::gemm_array::ArrayDesign;
use pm_npu::kernels::gemm_core::Epilogue;
use pm_npu::kernels::gemm_g80::{build_panel_program, build_panel_program_rowmajor};

use super::shapes::{EpilogueContract, FamilyMask, Shape};

/// Rows of one G80 M wave: every program is built for a whole number of these.
const WAVE_ROWS: usize = 256;
/// Largest row count a G80 design accepts (256 waves of 256 rows).
const ROWS_LIMIT: usize = 256 * WAVE_ROWS;

/// The program of one trunk shape.
pub struct ShapeProgram {
    pub shape: Shape,
    /// G80 PDI/TXN and argument layout: rounded M, N1280, K2560, selected epilogue, Fast control.
    pub design: ArrayDesign,
    /// 1280-column B panels (submits per row block) of the selected families of this shape; 0 when none is selected.
    pub panel_count: usize,
    /// `panel_count * 1280`: fused columns including zero padding.
    pub padded_cols: usize,
    /// Bytes of `[A, B, C]` of ONE submit (`design.args`).
    pub arg_bytes: [usize; 3],
    /// Host bytes read by ONE submit (packed A + one packed B panel).
    pub host_bytes_read: u64,
    /// Host bytes written by ONE submit (the native I32 or BF16 C arena).
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
/// `Err` for an empty mask or `rows_max` outside `1..=65536`.
pub fn build_programs_for_families(
    contract: EpilogueContract,
    rows_max: usize,
    families: FamilyMask,
) -> Result<NpuPrograms, String> {
    let epilogue = match contract {
        EpilogueContract::I32Fold => Epilogue::I32,
        EpilogueContract::Bf16Direct => Epilogue::Bf16,
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
        let design = build_panel_program(rows, epilogue).expect("validated G80 rows");
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

/// Compacted row-major pitches; zero-panel shapes are not submitted.
pub fn build_rowmajor_programs_for_families(rows_max: usize, families: FamilyMask) -> Result<NpuPrograms, String> {
    if !(1..=ROWS_LIMIT).contains(&rows_max) {
        return Err(format!("npu-trunk: rows_max {rows_max} outside 1..={ROWS_LIMIT}"));
    }
    let rows = rows_max.div_ceil(WAVE_ROWS) * WAVE_ROWS;
    let program = |shape: Shape| -> Result<ShapeProgram, String> {
        let map = shape.column_map(families);
        // A zero-panel record carries a valid unused design, never an arena.
        let design = build_panel_program_rowmajor(rows, map.padded_cols().max(1280), 0)?;
        Ok(ShapeProgram {
            shape, panel_count: map.panel_count(), padded_cols: map.padded_cols(),
            arg_bytes: [design.args[0].bytes, design.args[1].bytes, design.args[2].bytes],
            host_bytes_read: design.host_bytes_read(),
            host_bytes_written: design.host_bytes_written(), design,
        })
    };
    Ok(NpuPrograms {
        contract: EpilogueContract::Bf16Direct, rows_max: rows,
        per_shape: [program(Shape::GdnIn)?, program(Shape::QsaIn)?],
    })
}

#[cfg(test)]
mod rowmajor_tests {
    use super::super::Family;
    use super::*;

    #[test]
    fn shape_programs_declare_fused_arena_sizes() {
        let programs = build_rowmajor_programs_for_families(8192, FamilyMask::ALL).unwrap();
        for p in &programs.per_shape {
            assert_eq!(p.arg_bytes[2], 8192 * p.padded_cols * 2);
            assert_eq!(p.panel_count * 1280, p.padded_cols);
        }
        assert_eq!(programs.per_shape[0].design.pdi, programs.per_shape[1].design.pdi);
    }
    #[test]
    fn qkv_only_compacts_pitch_and_omits_qsa() {
        let programs = build_rowmajor_programs_for_families(256, Family::GdnQkv.into()).unwrap();
        let gdn = programs.get(Shape::GdnIn);
        assert_eq!((gdn.panel_count, gdn.padded_cols), (8, 10240));
        assert_eq!(gdn.arg_bytes[2], 256 * 10240 * 2);
        assert_eq!(programs.get(Shape::QsaIn).panel_count, 0);
    }
}
