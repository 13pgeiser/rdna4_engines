// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! In-tree table of validated host profiles and the tier decision.
//!
//! A row is a host the NPU trunk route was validated on. A machine whose facts
//! equal a row (and whose kernel set is one the row lists) is `reference`;
//! anything else that passes every blocking check is `compatible-unvalidated`
//! and reports exactly which fields differ from the nearest row.

use super::report::{Host, ProfileDiff, Tier};

/// One validated host.
pub(crate) struct Profile {
    pub id: &'static str,
    pub kernel: &'static str,
    pub iommu_gpu: &'static str,
    pub iommu_npu: &'static str,
    pub amdxdna: &'static str,
    pub npu_fw: &'static str,
    /// Header index sha256 of the pinned model artifact.
    pub model_index_sha256: &'static str,
    /// Kernel sets (`pack:<tag>@<commit12>` / `jit:<commit12>`) validated on this host.
    pub kernels: &'static [&'static str],
}

/// Validated hosts, first row first. Add a row only from a full `doctor --npu` run.
pub(crate) const PROFILES: &[Profile] = &[Profile {
    id: "hipx",
    kernel: "7.3.0-rc7-perfopt+",
    iommu_gpu: "identity",
    iommu_npu: "identity",
    amdxdna: "in-tree",
    npu_fw: "1.1.2.65",
    model_index_sha256: "8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013",
    // Filled from the Halo P0 smoke of doctor --npu; see docs/npu/npu-trunk.md.
    kernels: &[],
}];

/// The tier decision for one host.
#[derive(Debug, PartialEq)]
pub(crate) struct Assessment {
    pub tier: Tier,
    /// Matched (reference) or nearest (compatible) row; `None` when blocked.
    pub profile: Option<&'static str>,
    pub diff: Vec<ProfileDiff>,
}

fn differences(profile: &Profile, host: &Host, model_sha: Option<&str>) -> Vec<ProfileDiff> {
    let mut diff = Vec::new();
    let mut field = |name: &'static str, reference: &str, found: &str| {
        if reference != found {
            diff.push(ProfileDiff {
                field: name,
                reference: reference.to_owned(),
                found: found.to_owned(),
            });
        }
    };
    field("kernel", profile.kernel, &host.kernel);
    field("iommu.gpu", profile.iommu_gpu, &host.iommu.gpu);
    field("iommu.npu", profile.iommu_npu, &host.iommu.npu);
    field("amdxdna", profile.amdxdna, &host.amdxdna);
    field("npu_fw", profile.npu_fw, &host.npu_fw);
    field(
        "model",
        profile.model_index_sha256,
        model_sha.unwrap_or("unknown"),
    );
    if !profile.kernels.contains(&host.kernels.as_str()) {
        let reference = if profile.kernels.is_empty() {
            "none validated yet".to_owned()
        } else {
            profile.kernels.join(", ")
        };
        diff.push(ProfileDiff {
            field: "kernels",
            reference,
            found: host.kernels.clone(),
        });
    }
    diff
}

/// Tier for `host`: blocked on any failed blocking check, reference when every
/// field equals one row, else compatible-unvalidated against the nearest row
/// (fewest differing fields; the earlier row wins a tie).
pub(crate) fn assess(
    profiles: &[Profile],
    blocked: bool,
    host: &Host,
    model_sha: Option<&str>,
) -> Assessment {
    if blocked {
        return Assessment {
            tier: Tier::Blocked,
            profile: None,
            diff: Vec::new(),
        };
    }
    let mut nearest: Option<(&'static str, Vec<ProfileDiff>)> = None;
    for profile in profiles {
        let diff = differences(profile, host, model_sha);
        if nearest
            .as_ref()
            .is_none_or(|(_, best)| diff.len() < best.len())
        {
            nearest = Some((profile.id, diff));
        }
    }
    match nearest {
        Some((id, diff)) if diff.is_empty() => Assessment {
            tier: Tier::Reference,
            profile: Some(id),
            diff,
        },
        Some((id, diff)) => Assessment {
            tier: Tier::CompatibleUnvalidated,
            profile: Some(id),
            diff,
        },
        None => Assessment {
            tier: Tier::CompatibleUnvalidated,
            profile: None,
            diff: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doctor::report::Iommu;

    const KERNEL_SET: &str = "pack:v0.4.3@0123456789ab";

    static TABLE: &[Profile] = &[
        Profile {
            id: "alpha",
            kernel: "6.17.0",
            iommu_gpu: "DMA-FQ",
            iommu_npu: "DMA-FQ",
            amdxdna: "in-tree",
            npu_fw: "1.1.0.0",
            model_index_sha256: "aa",
            kernels: &[KERNEL_SET],
        },
        Profile {
            id: "beta",
            kernel: "7.3.0",
            iommu_gpu: "identity",
            iommu_npu: "identity",
            amdxdna: "in-tree",
            npu_fw: "1.1.2.65",
            model_index_sha256: "bb",
            kernels: &[KERNEL_SET, "jit:ffffffffffff"],
        },
    ];

    fn beta_host() -> Host {
        Host {
            kernel: "7.3.0".into(),
            iommu: Iommu {
                gpu: "identity".into(),
                npu: "identity".into(),
            },
            amdxdna: "in-tree".into(),
            npu_fw: "1.1.2.65".into(),
            kernels: KERNEL_SET.into(),
            ..Host::default()
        }
    }

    fn fields(assessment: &Assessment) -> Vec<&'static str> {
        assessment.diff.iter().map(|diff| diff.field).collect()
    }

    #[test]
    fn exact_row_with_listed_kernel_set_is_reference() {
        let assessment = assess(TABLE, false, &beta_host(), Some("bb"));
        assert_eq!(assessment.tier, Tier::Reference);
        assert_eq!(assessment.profile, Some("beta"));
        assert!(assessment.diff.is_empty());
    }

    #[test]
    fn kernel_set_alone_is_compatible_unvalidated() {
        let mut host = beta_host();
        host.kernels = "jit:000000000000".into();
        let assessment = assess(TABLE, false, &host, Some("bb"));
        assert_eq!(assessment.tier, Tier::CompatibleUnvalidated);
        assert_eq!(assessment.profile, Some("beta"));
        assert_eq!(fields(&assessment), ["kernels"]);
        assert_eq!(assessment.diff[0].found, "jit:000000000000");
    }

    #[test]
    fn nearest_row_and_every_differing_field_are_reported() {
        let mut host = beta_host();
        host.kernel = "7.2.8".into();
        // Differs from both rows, so beta (4 differences) beats alpha (5) without a tie.
        host.iommu.gpu = "DMA".into();
        host.npu_fw = "1.1.9.9".into();
        let assessment = assess(TABLE, false, &host, Some("cc"));
        assert_eq!(assessment.tier, Tier::CompatibleUnvalidated);
        assert_eq!(assessment.profile, Some("beta"));
        assert_eq!(
            fields(&assessment),
            ["kernel", "iommu.gpu", "npu_fw", "model"]
        );
        assert_eq!(assessment.diff[0].reference, "7.3.0");
        assert_eq!(assessment.diff[0].found, "7.2.8");
    }

    #[test]
    fn blocked_wins_over_a_matching_row() {
        let assessment = assess(TABLE, true, &beta_host(), Some("bb"));
        assert_eq!(assessment.tier, Tier::Blocked);
        assert_eq!(assessment.profile, None);
        assert!(assessment.diff.is_empty());
    }

    #[test]
    fn first_row_has_no_validated_kernel_set_until_filled() {
        let first = &PROFILES[0];
        assert_eq!(first.id, "hipx");
        assert_eq!(first.kernel, "7.3.0-rc7-perfopt+");
        assert_eq!(first.npu_fw, "1.1.2.65");
        assert_eq!(first.model_index_sha256.len(), 64);
        let host = Host {
            kernel: first.kernel.into(),
            iommu: Iommu {
                gpu: first.iommu_gpu.into(),
                npu: first.iommu_npu.into(),
            },
            amdxdna: first.amdxdna.into(),
            npu_fw: first.npu_fw.into(),
            kernels: "jit:abcdefabcdef".into(),
            ..Host::default()
        };
        let assessment = assess(PROFILES, false, &host, Some(first.model_index_sha256));
        assert_eq!(assessment.tier, Tier::CompatibleUnvalidated);
        assert_eq!(fields(&assessment), ["kernels"]);
    }
}
