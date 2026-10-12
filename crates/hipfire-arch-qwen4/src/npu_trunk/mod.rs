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

pub use calibration::{
    CalKey, Calibration, ContentionCalibration, DeviceKey, FamilyCalibration, GlueCalibration,
    GpuCalibration, KernelKey, NpuCalibration, CALIBRATION_SCHEMA_VERSION,
};
pub use plan::{
    alpha_extra, bank_segments, crossover, gpu_chunk, load_log_line, npu_ms, pair_time, plan,
    request_log_line, BankSegments, GpuChunk, LoadLogKey, PairTime, Plan, Step, CHUNK,
};
pub use config::{NpuTrunkMode, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV};
pub use packing::{
    pack_matrix, pack_shape, pack_shape_with_contract, pack_trunk, pack_trunk_with_contract,
    LayerTrunkRefs, MatrixRef, PackedB, PackedTrunk,
};
pub use programs::{build_programs, build_programs_for_families, NpuPrograms, ShapeProgram};
pub use shapes::{ColumnMap, EpilogueContract, Family, FamilyMask, Shape, PANEL_COLS, TRUNK_K};
