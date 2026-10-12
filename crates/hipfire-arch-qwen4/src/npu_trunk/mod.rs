// SPDX-License-Identifier: Apache-2.0
//! Weight preparation, program selection and thin GPU glue for the opt-in
//! Flash-Next NPU trunk. References live in pm-npu; no device opens here.

mod calibration;
mod config;
mod gpu;
mod packing;
mod plan;
mod programs;
mod shapes;

pub(crate) use packing::pack_matrix_direct;
pub(crate) use programs::build_rowmajor_programs;

pub use calibration::{
    CalKey, Calibration, ContentionCalibration, DeviceKey, FamilyCalibration, GlueCalibration,
    GpuCalibration, KernelKey, NpuCalibration, CALIBRATION_SCHEMA_VERSION,
    calibration_build_identity, calibration_key, runtime_calibration_key, model_index_identity,
};
pub use plan::{
    alpha_extra, bank_segments, crossover, gpu_chunk, load_log_line, npu_ms, pair_time, plan,
    request_log_line, select_families, BankSegments, GpuChunk, LoadLogKey, PairTime, Plan,
    SelectionReport, SelectionRow, Step, CHUNK, SELECTION_ROWS,
    auto_admission, executable_plan, planned_pair_rows,
};
pub use config::{NpuTrunkMode, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV};
pub use packing::{
    pack_matrix, pack_shape, pack_shape_with_contract, pack_trunk, pack_trunk_with_contract,
    LayerTrunkRefs, MatrixRef, PackedB, PackedTrunk,
};
pub use programs::{build_programs, build_programs_for_families, NpuPrograms, ShapeProgram};
pub use shapes::{ColumnMap, EpilogueContract, Family, FamilyMask, Shape, PANEL_COLS, TRUNK_K};

#[cfg(test)]
mod plan_tests;
