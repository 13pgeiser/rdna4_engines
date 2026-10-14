// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Code-object loading: an AMDGPU ELF64 relocatable/shared object (HSA code
//! object v5/v6) → [`KernelImage`] for one kernel symbol (`name` and
//! `name.kd`), checking the ELF machine (`EM_AMDGPU`, `e_flags` MACH of the
//! [`crate::Arch`] variant).
//!
//! SCAFFOLD CONTRACT (owner: child `SimLoad`).
use super::{KernelImage, SimResult};

/// Load kernel `symbol` from `elf`: its code from the entry to the symbol's
/// end, and its 64-byte descriptor.
pub fn load_code_object(elf: &[u8], symbol: &str) -> SimResult<KernelImage> { let _ = (elf, symbol); todo!("SimLoad: load_code_object") }
