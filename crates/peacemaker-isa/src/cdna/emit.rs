// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Code-object emission for CDNA kernels: the `.amdhsa_*` directive set and
//! descriptor (gfx90a+ layout, `peacemaker_cdna::kd`), the `.amdgpu_metadata`
//! (wavefront size 64, AGPR count), and the HSA code object v5 or v6 through
//! `crate::native::link_encoded`; plus the equivalent `.s` text that ROCm
//! `llvm-mc` + `ld.lld` (test oracle) assemble and link into the same bytes
//! except the `.comment` stamp.
//!
//! SCAFFOLD CONTRACT (owner: child `IsaEmit`). Signatures are frozen.
use super::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeObjectVersion { V5, V6 }

/// The `.amdhsa_*` directives (name, value) of `k`'s descriptor, exactly the
/// set the native descriptor writer accepts for gfx942.
pub fn directives(k: &Kernel) -> Result<Vec<(String, String)>, String> { let _ = k; todo!("IsaEmit: directives") }
/// The `.amdgpu_metadata` YAML body (between `---` and `...`) for `kernels`.
pub fn metadata_yaml(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<String, String> { let _ = (kernels, version); todo!("IsaEmit: metadata_yaml") }
/// The native code object.
pub fn code_object(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<Vec<u8>, String> { let _ = (kernels, version); todo!("IsaEmit: code_object") }
/// `.s` text for the ROCm oracle (instructions printed by their `Display`).
pub fn assembly(kernels: &[&Kernel], version: CodeObjectVersion) -> Result<String, String> { let _ = (kernels, version); todo!("IsaEmit: assembly") }
