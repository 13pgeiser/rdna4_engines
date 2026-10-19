// SPDX-License-Identifier: Apache-2.0
//! Weight preparation, program selection and thin GPU glue for the opt-in
//! Flash-Next NPU trunk. References live in peacemaker-npu; no device opens here.

pub(crate) mod admission;
pub(crate) use calibration::runtime_key_with_firmware;
#[cfg(target_os = "linux")]
pub use calibration::{bounded_memlock_bytes, trunk_pin_bytes};
pub use calibration::{host_facts, npu_firmware, HostFacts};
#[cfg(target_os = "linux")]
pub(crate) use calibration::{TRUNK_BANKS, TRUNK_ROWS_MAX};
#[cfg(target_os = "linux")]
pub use railgun::npu::fclk;
mod calibration;
mod config;
pub(crate) mod containment;
mod gpu;
mod packing;
mod plan;
mod programs;
mod rates;
pub mod route;
mod shapes;

pub(crate) use packing::pack_matrix_direct;
pub(crate) use programs::build_rowmajor_programs_for_families;

pub use calibration::{
    calibration_build_identity, calibration_key, model_index_identity, model_trunk_index_sha256,
    runtime_calibration_key, CalKey, Calibration, ContentionCalibration, DeviceKey,
    FamilyCalibration, GlueCalibration, GpuCalibration, KernelKey, NpuCalibration, OutFamily,
    QsaOrder, CALIBRATION_SCHEMA_VERSION,
};
pub use config::{NpuTrunkMode, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV};
pub use packing::{
    pack_matrix, pack_out, pack_out_layers, pack_shape, pack_shape_with_contract, pack_trunk,
    pack_trunk_with_contract, LayerTrunkRefs, MatrixRef, PackedB, PackedOut, PackedTrunk,
};
pub use plan::{
    auto_admission, bank_segments, bank_segments_moved, contention_extra, crossover,
    executable_plan, executable_plan_for, gpu_chunk, load_log_line, npu_ms, pair_time,
    pair_time_moved, plan, plan_executable, plan_moved, planned_pair, planned_pair_rows,
    request_log_line, select_families, BankSegments, GpuChunk, LoadLogKey, Moved, PairTime, Plan,
    SelectionReport, SelectionRow, Step, CHUNK, SELECTION_ROWS,
};
pub use programs::{
    build_out_programs, build_program_for_rows, build_programs, build_programs_for_families,
    wave_padded_rows, NpuOutPrograms, NpuPrograms, ShapeProgram,
};
pub use rates::{
    choose_families, pair_model, scaled_calibration, LayerKind, PairModel, RateEstimator,
    RateSample, RateSnapshot, APPLY_EPS, EWMA_ALPHA, FREEZE_AFTER, FREEZE_EPS, FREEZE_WINDOW,
    MIN_MODEL_MS, REOPEN_CONFIRM, REOPEN_EPS, SCALE_MAX, SCALE_MIN,
};
pub use route::{diagnostic_gpu_route, set_diagnostic_gpu_route};
pub use shapes::{
    ColumnMap, EpilogueContract, Family, FamilyMask, Shape, OUT_COLS, OUT_K, OUT_PANEL_COLS,
    PANEL_COLS, TRUNK_K,
};

#[cfg(test)]
mod plan_tests;
