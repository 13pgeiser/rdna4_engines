// SPDX-License-Identifier: Apache-2.0
//! Weight preparation, program selection and thin GPU glue for the opt-in
//! Flash-Next NPU trunk. References live in pm-npu; no device opens here.

mod config;
mod gpu;
mod packing;
mod programs;
mod shapes;

pub use config::{NpuTrunkMode, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV};
pub use packing::{
    pack_matrix, pack_shape, pack_shape_with_contract, pack_trunk, pack_trunk_with_contract,
    LayerTrunkRefs, MatrixRef, PackedB, PackedTrunk,
};
pub use programs::{build_programs, build_programs_for_families, NpuPrograms, ShapeProgram};
pub use shapes::{ColumnMap, EpilogueContract, Family, FamilyMask, Shape, PANEL_COLS, TRUNK_K};
