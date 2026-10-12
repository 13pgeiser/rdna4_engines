// SPDX-License-Identifier: Apache-2.0
//! Stored measurement inputs. Loading never measures or recalibrates a device.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

pub const CALIBRATION_SCHEMA_VERSION: u32 = 3;
/// Change the contract suffix when serving numerics/transport change.
pub fn calibration_build_identity() -> String {
    format!("hipfire-{}:npu-trunk-bf16-rowmajor-v1", env!("CARGO_PKG_VERSION"))
}
pub(crate) const FAMILY_NAMES: [&str; 4] = ["qkv", "zab", "q", "kvidx"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceKey {
    pub arch: String,
    pub pci_bdf: String,
    pub npu_fw_version: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelKey {
    pub uname_release: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalKey {
    pub device: DeviceKey,
    /// Stable release + NPU numerical/transport contract, never a git SHA.
    #[serde(default = "calibration_build_identity")]
    pub build: String,
    pub kernel: KernelKey,
    /// SHA256 of HFQ bytes [0, data_offset), as in trunk-overlay binding.
    pub model_index_sha256: String,
    /// `i32` or `bf16`, matching NpuCalibration::epilogue.
    pub epilogue: String,
    /// GPU-native trunk identity: `stock` or `overlay:<artifact marker>`.
    pub trunk_route: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuCalibration {
    pub trunk_in_ms_per_8192: BTreeMap<String, f64>,
    pub uniform_per_layer_ms: f64,
    pub gdn_core_per_layer_ms: f64,
    pub qsa_per_layer_ms: f64,
    pub qsa_growth_per_layer_ms_per_8192_prefix: f64,
    pub fixed_per_chunk_ms: f64,
    pub boundary_tax_ms: f64,
    pub min_bank_rows: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyCalibration { pub layer: String, pub cols: usize, pub out_dtype: String }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlueCalibration {
    pub pack_ms_8192: f64,
    pub fold_ms_per_col_8192: f64,
    pub sync_ms: f64,
    pub consume_mode: String,
    pub consume_ms_per_col_8192: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NpuCalibration {
    pub tops_corun: f64,
    pub bytes_per_kop: f64,
    pub epilogue: String,
    pub panel_cols: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentionCalibration { pub alpha_overlap: f64, pub model: String }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub schema_version: u32,
    pub key: CalKey,
    pub gpu: GpuCalibration,
    pub families: BTreeMap<String, FamilyCalibration>,
    pub glue: GlueCalibration,
    pub npu: NpuCalibration,
    pub contention: ContentionCalibration,
    pub margin: f64,
}
impl Calibration {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let cal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        cal.validate()?;
        Ok(cal)
    }
    pub fn store(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let json = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())
    }
    /// First mismatch in stable device/build/kernel/model/epilogue/trunk-route order. Caller
    /// must leave the route off and log this reason; never silently recalibrate.
    pub fn key_matches(&self, key: &CalKey) -> Result<(), String> {
        macro_rules! check { ($($field:ident).+) => {
            if self.key.$($field).+ != key.$($field).+ {
                return Err(format!("calibration key mismatch: {}", stringify!($($field).+)));
            }
        }; }
        check!(device.arch); check!(device.pci_bdf); check!(device.npu_fw_version);
        check!(build); check!(kernel.uname_release);
        check!(model_index_sha256); check!(epilogue); check!(trunk_route);
        Ok(())
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION { return Err("unsupported calibration schema_version".into()); }
        if !self.margin.is_finite() || !(0.0..1.0).contains(&self.margin) { return Err("margin must be finite and in [0,1)".into()); }
        if self.npu.panel_cols == 0 || self.gpu.min_bank_rows == 0 { return Err("panel_cols and min_bank_rows must be positive".into()); }
        if self.npu.epilogue != "i32" && self.npu.epilogue != "bf16" { return Err("npu.epilogue must be i32 or bf16".into()); }
        if self.key.epilogue != self.npu.epilogue { return Err("key.epilogue differs from npu.epilogue".into()); }
        if self.key.trunk_route != "stock" && !self.key.trunk_route.strip_prefix("overlay:").is_some_and(|marker| !marker.trim().is_empty()) {
            return Err("key.trunk_route must be stock or overlay:<marker>".into());
        }
        if self.glue.consume_mode != "retile" && self.glue.consume_mode != "direct" {
            return Err("glue.consume_mode must be retile or direct".into());
        }
        if self.glue.consume_ms_per_col_8192.len() != 2 {
            return Err("expected bf16,f32 consume_ms_per_col_8192".into());
        }
        for dtype in ["bf16", "f32"] {
            let cost = self.glue.consume_ms_per_col_8192.get(dtype).ok_or_else(|| format!("missing glue.consume_ms_per_col_8192.{dtype}"))?;
            if !cost.is_finite() || *cost < 0.0 { return Err(format!("invalid glue.consume_ms_per_col_8192.{dtype}")); }
        }
        for (name, value) in [
            ("device.arch", self.key.device.arch.as_str()), ("device.pci_bdf", &self.key.device.pci_bdf),
            ("device.npu_fw_version", &self.key.device.npu_fw_version), ("build", &self.key.build),
            ("kernel.uname_release", &self.key.kernel.uname_release), ("contention.model", &self.contention.model),
        ] { if value.trim().is_empty() { return Err(format!("{name} must not be empty")); } }
        if self.key.model_index_sha256.len() != 64 || !self.key.model_index_sha256.bytes().all(|c| c.is_ascii_hexdigit()) { return Err("model_index_sha256 must be 64 hex digits".into()); }
        if self.families.len() != 4 || self.gpu.trunk_in_ms_per_8192.len() != 4 { return Err("expected qkv,zab,q,kvidx calibration families".into()); }
        for (i, name) in FAMILY_NAMES.iter().enumerate() {
            let f = self.families.get(*name).ok_or_else(|| format!("missing families.{name}"))?;
            if f.cols == 0 || f.layer != if i < 2 { "gdn" } else { "qsa" } { return Err(format!("invalid families.{name}")); }
            if f.out_dtype != "bf16" && f.out_dtype != "f32" { return Err(format!("invalid families.{name}.out_dtype")); }
            let cost = self.gpu.trunk_in_ms_per_8192.get(*name).ok_or_else(|| format!("missing gpu.trunk_in_ms_per_8192.{name}"))?;
            if !cost.is_finite() || *cost < 0.0 { return Err(format!("invalid gpu.trunk_in_ms_per_8192.{name}")); }
        }
        for (name, value) in [
            ("uniform_per_layer_ms", self.gpu.uniform_per_layer_ms), ("gdn_core_per_layer_ms", self.gpu.gdn_core_per_layer_ms),
            ("qsa_per_layer_ms", self.gpu.qsa_per_layer_ms), ("qsa_growth_per_layer_ms_per_8192_prefix", self.gpu.qsa_growth_per_layer_ms_per_8192_prefix),
            ("fixed_per_chunk_ms", self.gpu.fixed_per_chunk_ms), ("boundary_tax_ms", self.gpu.boundary_tax_ms),
            ("pack_ms_8192", self.glue.pack_ms_8192), ("fold_ms_per_col_8192", self.glue.fold_ms_per_col_8192),
            ("sync_ms", self.glue.sync_ms), ("bytes_per_kop", self.npu.bytes_per_kop),
        ] { if !value.is_finite() || value < 0.0 { return Err(format!("{name} must be finite and nonnegative")); } }
        if !self.npu.tops_corun.is_finite() || self.npu.tops_corun <= 0.0 { return Err("tops_corun must be finite and positive".into()); }
        if !self.contention.alpha_overlap.is_finite() || self.contention.alpha_overlap < 1.0 { return Err("alpha_overlap must be finite and >= 1".into()); }
        Ok(())
    }
}

/// Pure construction shared by runtime and CPU identity tests.
pub fn calibration_key(arch: &str, pci_bdf: &str, firmware: &str, release: &str, index: &str, trunk_route: &str) -> CalKey {
    CalKey {
        device: DeviceKey { arch: arch.into(), pci_bdf: pci_bdf.to_ascii_lowercase(), npu_fw_version: firmware.trim().into() },
        build: calibration_build_identity(),
        kernel: KernelKey { uname_release: release.trim().into() },
        model_index_sha256: index.into(), epilogue: "bf16".into(), trunk_route: trunk_route.into(),
    }
}

/// Read only unprivileged metadata, never open a GPU/NPU or time work.
pub fn runtime_calibration_key(arch: &str, pci_bdf: &str, index: &str, trunk_route: &str) -> Result<CalKey, String> {
    // railgun opens accel0; key that exact device, not the first PCI directory.
    let device = Path::new("/sys/class/accel/accel0/device");
    let firmware = fs::read_to_string(device.join("fw_version")).map_err(|e| format!("NPU fw_version: {e}"))?;
    let release = std::process::Command::new("uname").arg("-r").output().map_err(|e| format!("uname -r: {e}"))?;
    if !release.status.success() { return Err("uname -r failed".into()); }
    let release = std::str::from_utf8(&release.stdout).map_err(|e| e.to_string())?;
    // PerfOpt's armed message is privileged dmesg, with no sysfs/proc ABI.
    // Schema 3 deliberately keys kernel identity on uname_release alone.
    Ok(calibration_key(arch, pci_bdf, &firmware, release, index, trunk_route))
}

/// Capture the base before it moves into the source owner. Off performs no I/O.
pub fn model_index_identity(hfq: &hipfire_runtime::hfq::HfqFile) -> Option<Result<String, String>> {
    if super::NpuTrunkMode::from_env().ok().is_none_or(|mode| mode.is_off()) { return None; }
    Some(hfq.index_sha256().map(|hash| {
        use std::fmt::Write;
        let mut hex = String::with_capacity(64);
        for byte in hash { write!(&mut hex, "{byte:02x}").expect("write to String"); }
        hex
    }).map_err(|e| format!("model index identity: {e}")))
}
