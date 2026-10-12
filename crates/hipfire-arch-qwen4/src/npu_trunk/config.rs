// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! NPU-trunk mode: the single user-facing switch of the Qwen4 integration.
//!
//! The mode is read through the central developer-config accessor
//! ([`hipfire_config::developer_var`]), so the `HIPFIRE_*` key is registered with the
//! process configuration automatically and no `std::env` access happens here.

use std::fmt;
use std::str::FromStr;

/// Developer environment key of the mode.
pub const NPU_TRUNK_ENV: &str = "HIPFIRE_QWEN4_NPU_TRUNK";
/// Daemon configuration key of the mode.
pub const NPU_TRUNK_DAEMON_KEY: &str = "npu_trunk";

/// How the Flash-Next trunk projections reach the NPU.
///
/// `Off` (the default) never allocates, never opens a device and never packs weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NpuTrunkMode {
    /// Trunk projections stay on the GPU; nothing NPU-related is touched.
    #[default]
    Off,
    /// Use the NPU when it is present and the plan predicts a gain, else fall back to the GPU.
    Auto,
    /// Require the NPU; failure to open it is an error.
    On,
    /// Run the bit-identical GPU emulation of the NPU numerics (no device).
    Emulate,
}

impl NpuTrunkMode {
    /// Every mode, in documentation order.
    pub const ALL: [NpuTrunkMode; 4] = [Self::Off, Self::Auto, Self::On, Self::Emulate];

    /// The canonical lower-case spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
            Self::Emulate => "emulate",
        }
    }

    /// `true` for [`NpuTrunkMode::Off`].
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }

    /// Parse an optional raw value: `None` (key absent) is [`NpuTrunkMode::Off`]; a present value
    /// must be a valid spelling (an empty value is an error, never silently `Off`).
    pub fn from_value(value: Option<&str>) -> Result<Self, String> {
        match value {
            None => Ok(Self::Off),
            Some(raw) => raw
                .parse::<Self>()
                .map_err(|e| format!("{NPU_TRUNK_ENV}: {e}")),
        }
    }

    /// Resolve the mode from the process configuration (`HIPFIRE_QWEN4_NPU_TRUNK`); unset is `Off`.
    pub fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var(NPU_TRUNK_ENV) {
            Ok(raw) => Self::from_value(Some(&raw)),
            Err(std::env::VarError::NotPresent) => Ok(Self::Off),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{NPU_TRUNK_ENV}: value is not valid UTF-8"))
            }
        }
    }
}

impl FromStr for NpuTrunkMode {
    type Err = String;

    /// Case-insensitive, surrounding whitespace ignored: `off`, `auto`, `on`, `emulate`.
    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "emulate" => Ok(Self::Emulate),
            _ => Err(format!(
                "invalid npu trunk mode {s:?} (expected off, auto, on or emulate)"
            )),
        }
    }
}

impl fmt::Display for NpuTrunkMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        assert_eq!(NpuTrunkMode::default(), NpuTrunkMode::Off);
        assert!(NpuTrunkMode::default().is_off());
        assert!(!NpuTrunkMode::On.is_off());
    }

    #[test]
    fn parse_all_modes_roundtrip() {
        for mode in NpuTrunkMode::ALL {
            assert_eq!(mode.to_string().parse::<NpuTrunkMode>(), Ok(mode));
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!("off".parse(), Ok(NpuTrunkMode::Off));
        assert_eq!("auto".parse(), Ok(NpuTrunkMode::Auto));
        assert_eq!("on".parse(), Ok(NpuTrunkMode::On));
        assert_eq!("emulate".parse(), Ok(NpuTrunkMode::Emulate));
    }

    #[test]
    fn parse_is_case_and_space_insensitive() {
        assert_eq!(" EMULATE\n".parse(), Ok(NpuTrunkMode::Emulate));
        assert_eq!("Auto".parse(), Ok(NpuTrunkMode::Auto));
        assert_eq!("\tOn ".parse(), Ok(NpuTrunkMode::On));
    }

    #[test]
    fn parse_rejects_unknown_and_empty() {
        for bad in ["", " ", "1", "0", "true", "yes", "enable", "emulated", "of f"] {
            let err = bad.parse::<NpuTrunkMode>().unwrap_err();
            assert!(err.contains("invalid npu trunk mode"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn from_value_absent_is_off_present_is_strict() {
        assert_eq!(NpuTrunkMode::from_value(None), Ok(NpuTrunkMode::Off));
        assert_eq!(NpuTrunkMode::from_value(Some("on")), Ok(NpuTrunkMode::On));
        let err = NpuTrunkMode::from_value(Some("")).unwrap_err();
        assert!(err.starts_with(NPU_TRUNK_ENV), "{err}");
        let err = NpuTrunkMode::from_value(Some("maybe")).unwrap_err();
        assert!(err.contains("\"maybe\""), "{err}");
    }

    #[test]
    fn env_key_is_the_documented_developer_key() {
        assert_eq!(NPU_TRUNK_ENV, "HIPFIRE_QWEN4_NPU_TRUNK");
        assert_eq!(NPU_TRUNK_DAEMON_KEY, "npu_trunk");
        let field = hipfire_config::field(NPU_TRUNK_DAEMON_KEY).expect("registered npu_trunk schema");
        assert_eq!(field.env_compat, Some(NPU_TRUNK_ENV));
        assert!(matches!(field.default, hipfire_config::DefaultValue::String("off")));
        for mode in NpuTrunkMode::ALL {
            assert!(field.validate(&hipfire_config::ConfigValue::String(mode.to_string())).is_ok());
        }
        assert!(field.validate(&hipfire_config::ConfigValue::String("invalid".into())).is_err());
    }
}
