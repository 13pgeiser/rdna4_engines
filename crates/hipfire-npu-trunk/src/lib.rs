// SPDX-License-Identifier: Apache-2.0
//! CPU-only weight preparation and G80 program construction for the opt-in
//! Flash-Next NPU trunk. This crate does not open a device or plan execution.

mod config;
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
