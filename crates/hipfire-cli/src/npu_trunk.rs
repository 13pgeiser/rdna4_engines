// SPDX-License-Identifier: Apache-2.0
//! Offline calibration assembly. This command never runs a benchmark or opens a device.
use anyhow::{bail, Context, Result};
use clap::Args;
use hipfire_arch_qwen4::npu_trunk::{
    CalKey, Calibration, ContentionCalibration, FamilyCalibration, GlueCalibration,
    GpuCalibration, NpuCalibration, CALIBRATION_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

#[derive(Args, Debug)]
pub(crate) struct CalibrateArgs {
    #[arg(long, value_name = "MEASUREMENTS.JSON")]
    from: PathBuf,
    #[arg(long, value_name = "CAL.JSON")]
    out: PathBuf,
}
/// No defaults: missing measured quantities or key fields are errors.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurements {
    schema_version: u32,
    key: CalKey,
    gpu: GpuCalibration,
    families: BTreeMap<String, FamilyCalibration>,
    glue: GlueCalibration,
    npu: NpuCalibration,
    contention: ContentionCalibration,
    margin: f64,
}
impl Measurements {
    fn assemble(self, uname_release: &str) -> Result<Calibration> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION { bail!("unsupported measurements schema_version"); }
        // Device/fw, PerfOpt state and model/build identity are supplied by the
        // measurement collector. Never substitute this CLI's binary for the
        // caller's measured daemon/kernel build identity.
        if self.key.kernel.uname_release != uname_release {
            bail!("measurement key mismatch: kernel.uname_release (measured {}, current {})", self.key.kernel.uname_release, uname_release);
        }
        let cal = Calibration {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            key: self.key, gpu: self.gpu, families: self.families, glue: self.glue,
            npu: self.npu, contention: self.contention, margin: self.margin,
        };
        cal.validate().map_err(anyhow::Error::msg)?;
        Ok(cal)
    }
}
pub(crate) fn calibrate_command(args: CalibrateArgs) -> Result<()> {
    let bytes = fs::read(&args.from).with_context(|| format!("read {}", args.from.display()))?;
    let input: Measurements = serde_json::from_slice(&bytes).context("invalid or incomplete measurement record")?;
    let output = Command::new("uname").arg("-r").output().context("read current kernel release")?;
    if !output.status.success() { bail!("uname -r failed"); }
    let release = std::str::from_utf8(&output.stdout).context("uname output is not UTF-8")?.trim();
    let cal = input.assemble(release)?;
    cal.store(&args.out).map_err(anyhow::Error::msg)?;
    println!("npu-trunk: calibration={} schema_version={}", args.out.display(), cal.schema_version);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn npu_trunk_cli_requires_both_paths() {
        use clap::Parser;
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--from", "in.json", "--out", "out.json"]).is_ok());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--from", "in.json"]).is_err());
    }
    #[test]
    fn npu_trunk_measurements_reject_missing_fields() {
        assert!(serde_json::from_str::<Measurements>("{}").is_err());
        assert!(serde_json::from_str::<Measurements>(r#"{"schema_version":1,"margin":0}"#).is_err());
    }
}
