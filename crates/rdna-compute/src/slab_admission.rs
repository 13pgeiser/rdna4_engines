// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Which expert-slab codecs a GPU architecture serves, and a layer's cold
//! codec (pure CPU matrix).
//!
//! Codec ids are the per-layer codec table bytes of an expert-slab layer:
//! `0` = MQ4 symmetric, `1` = MQ3-I4 (W3), `3` = E8P-I4. `2` (MQ2-I4, W2) is
//! refused on every architecture.
//!
//! | arch    | MQ4 (0) | W3 (1) | W2 (2) | E8P (3) |
//! |---------|---------|--------|--------|---------|
//! | gfx1201 | yes     | yes    | no     | yes     |
//! | gfx1151 | yes     | yes    | no     | no      |
//! | gfx1100 | yes     | no     | no     | yes     |
//! | other   | no      | no     | no     | no      |
//!
//! gfx1100 serves MQ4-only layers (no cold bucket) and MQ4 + E8P layers; W3 is
//! refused there (no gfx1100 W3 kernels). gfx1151 serves MQ4-only layers and
//! MQ4 + W3 layers; E8P is refused there (no gfx1151 E8P kernels). Matching is
//! by exact architecture name, so near misses are refused.
//!
//! Cold codec `0` (MQ4) means NO cold bucket on every architecture, gfx1151
//! included. The "no cold kernel preload" promise for such a layer is scoped to
//! gfx1100/gfx1201. gfx1151 is the one exception: even with cold codec `0` its
//! route unconditionally preloads the shared planner, HIP mask and all four K1
//! function handles (W3 gate/up, W3 down, MQ4 gate/up, MQ4 down) of the
//! already-qualified shared image. That is image preparation, not a W3 bucket
//! launch and not permission for any other cold codec.
//!
//! Every unknown codec id is refused on every architecture. The sealed route
//! has two buckets per layer (cold, MQ4): a layer's cold bucket is one codec,
//! so a layer holds MQ4 plus at most ONE cold codec ([`qwen4_slab_layer_cold_codec`]);
//! different layers may choose different cold codecs.

/// Codec id of an MQ4 symmetric expert.
pub const SLAB_CODEC_MQ4: u8 = 0;
/// Codec id of an MQ3-I4 expert.
pub const SLAB_CODEC_W3: u8 = 1;
/// Codec id of an E8P-I4 expert.
pub const SLAB_CODEC_E8P: u8 = 3;

/// Whether the sealed expert-slab route serves `codec` on GPU architecture
/// `arch`. Exact architecture names only: no family prefix match.
pub fn qwen4_slab_codec_admitted(arch: &str, codec: u8) -> bool {
    match arch {
        "gfx1201" => matches!(codec, SLAB_CODEC_MQ4 | SLAB_CODEC_W3 | SLAB_CODEC_E8P),
        "gfx1151" => matches!(codec, SLAB_CODEC_MQ4 | SLAB_CODEC_W3),
        "gfx1100" => matches!(codec, SLAB_CODEC_MQ4 | SLAB_CODEC_E8P),
        _ => false,
    }
}

/// The cold codec a layer without cold experts (all MQ4) is launched with on
/// `arch`: [`SLAB_CODEC_MQ4`] itself, i.e. NO cold bucket (only the MQ4 bucket
/// launches; requesting bucket 0 with cold codec 0 must fail). It is the same on
/// every slab architecture, gfx1151 included, and never names W3 as the cold
/// codec of an MQ4-only layer. On gfx1100/gfx1201 an MQ4-only layer loads and
/// allocates nothing of any cold codec's kernels; gfx1151 still preloads the
/// shared planner/mask/K1 image (see the table note above), which is not a cold
/// bucket or a cold-codec permission. `None` off the slab architectures.
pub fn qwen4_slab_default_cold_codec(arch: &str) -> Option<u8> {
    qwen4_slab_codec_admitted(arch, SLAB_CODEC_MQ4).then_some(SLAB_CODEC_MQ4)
}

/// Why a layer's codec table has no single cold codec on its architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabLayerCodecError {
    /// `codec` is not served on this architecture (unknown architecture, W2,
    /// an unknown id, or another architecture's cold codec).
    NotAdmitted { codec: u8 },
    /// The layer holds two different cold codecs; its two buckets are cold
    /// and MQ4, so it cannot be served.
    MixedCold { first: u8, second: u8 },
}

impl std::fmt::Display for SlabLayerCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAdmitted { codec } => {
                write!(f, "slab codec {codec} is not served on this architecture")
            }
            Self::MixedCold { first, second } => write!(
                f,
                "a slab layer holds two cold codecs ({first} and {second}); the route has one cold bucket per layer"
            ),
        }
    }
}

impl std::error::Error for SlabLayerCodecError {}

/// The cold codec of one layer's per-expert codec bytes on `arch`: the single
/// admitted non-MQ4 codec the layer holds, or [`qwen4_slab_default_cold_codec`]
/// ([`SLAB_CODEC_MQ4`], no cold bucket) for an MQ4-only layer. Refuses a codec
/// the architecture does not serve and a layer holding two different cold
/// codecs. Computed once per layer at load, never per token.
pub fn qwen4_slab_layer_cold_codec(arch: &str, codecs: &[u8]) -> Result<u8, SlabLayerCodecError> {
    let mut cold: Option<u8> = None;
    for &codec in codecs {
        if !qwen4_slab_codec_admitted(arch, codec) {
            return Err(SlabLayerCodecError::NotAdmitted { codec });
        }
        if codec == SLAB_CODEC_MQ4 {
            continue;
        }
        match cold {
            None => cold = Some(codec),
            Some(first) if first != codec => {
                return Err(SlabLayerCodecError::MixedCold {
                    first,
                    second: codec,
                })
            }
            Some(_) => {}
        }
    }
    cold.or_else(|| qwen4_slab_default_cold_codec(arch))
        .ok_or(SlabLayerCodecError::NotAdmitted {
            codec: SLAB_CODEC_MQ4,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_ids_are_the_codec_table_bytes() {
        assert_eq!((SLAB_CODEC_MQ4, SLAB_CODEC_W3, SLAB_CODEC_E8P), (0, 1, 3));
    }

    #[test]
    fn matrix_is_exactly_the_admitted_pairs() {
        // (arch, admitted among codecs 0, 1, 2, 3 and an unknown byte).
        let expected: [(&str, [bool; 5]); 5] = [
            ("gfx1100", [true, false, false, true, false]),
            ("gfx1201", [true, true, false, true, false]),
            ("gfx1151", [true, true, false, false, false]),
            ("gfx942", [false; 5]),
            ("unknown", [false; 5]),
        ];
        for (arch, row) in expected {
            for (codec, want) in [0u8, 1, 2, 3, 255].into_iter().zip(row) {
                assert_eq!(
                    qwen4_slab_codec_admitted(arch, codec),
                    want,
                    "{arch} codec {codec}"
                );
            }
        }
    }

    #[test]
    fn architecture_match_is_exact() {
        for arch in [
            "", "gfx1100x", "gfx110", "gfx1101", "gfx1200", "gfx12010", "GFX1201", "gfx1201 ",
        ] {
            for codec in 0..=3u8 {
                assert!(
                    !qwen4_slab_codec_admitted(arch, codec),
                    "{arch:?} codec {codec}"
                );
            }
        }
    }

    #[test]
    fn default_cold_codec_is_no_cold_bucket_on_every_slab_arch() {
        for arch in ["gfx1201", "gfx1151", "gfx1100"] {
            assert_eq!(
                qwen4_slab_default_cold_codec(arch),
                Some(SLAB_CODEC_MQ4),
                "{arch}"
            );
        }
        for arch in ["gfx942", "gfx1200", ""] {
            assert_eq!(qwen4_slab_default_cold_codec(arch), None, "{arch}");
        }
    }

    #[test]
    fn layer_cold_codec_is_unambiguous() {
        let (mq4, w3, w2, e8p) = (SLAB_CODEC_MQ4, SLAB_CODEC_W3, 2u8, SLAB_CODEC_E8P);
        // One cold codec next to MQ4, or cold only.
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1201", &[mq4, w3, w3, mq4]),
            Ok(w3)
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1201", &[mq4, e8p, mq4]),
            Ok(e8p)
        );
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1201", &[e8p, e8p]), Ok(e8p));
        // MQ4-only: no cold bucket on either arch, deterministic.
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1201", &[mq4, mq4]), Ok(mq4));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1100", &[mq4, mq4]), Ok(mq4));
        // A layer holding both W3 and E8P has two cold codecs: refused, whatever the order.
        for codecs in [[mq4, w3, e8p], [e8p, w3, mq4], [w3, e8p, w3]] {
            assert!(matches!(
                qwen4_slab_layer_cold_codec("gfx1201", &codecs),
                Err(SlabLayerCodecError::MixedCold { .. })
            ));
        }
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1201", &[mq4, w3, e8p]),
            Err(SlabLayerCodecError::MixedCold {
                first: w3,
                second: e8p
            })
        );
        // gfx1100 serves E8P but no W3 (a layer with E8P alone or next to MQ4 is fine);
        // W2 is nowhere; unknown arches serve nothing.
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1100", &[mq4, e8p, mq4]),
            Ok(e8p)
        );
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1100", &[e8p, e8p]), Ok(e8p));
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1100", &[mq4, w3]),
            Err(SlabLayerCodecError::NotAdmitted { codec: w3 })
        );
        // Mixed W3+E8P on gfx1100 is refused (W3 is not admitted there at all).
        assert!(qwen4_slab_layer_cold_codec("gfx1100", &[mq4, w3, e8p]).is_err());
        assert!(qwen4_slab_layer_cold_codec("gfx1100", &[mq4, e8p, w3]).is_err());
        for arch in ["gfx1100", "gfx1201"] {
            assert_eq!(
                qwen4_slab_layer_cold_codec(arch, &[mq4, w2]),
                Err(SlabLayerCodecError::NotAdmitted { codec: w2 })
            );
        }
        for arch in ["gfx942", ""] {
            assert!(qwen4_slab_layer_cold_codec(arch, &[mq4]).is_err(), "{arch}");
            assert!(qwen4_slab_layer_cold_codec(arch, &[]).is_err(), "{arch}");
        }
        // gfx1151 serves MQ4 and W3 (no E8P): uniform MQ4 has no cold bucket, W3 is the cold
        // codec whether uniform or mixed with MQ4, and E8P is refused in every position.
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1151", &[mq4, mq4]), Ok(mq4));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1151", &[w3, w3]), Ok(w3));
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, w3, mq4, w3]),
            Ok(w3)
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, e8p]),
            Err(SlabLayerCodecError::NotAdmitted { codec: e8p })
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[e8p, e8p]),
            Err(SlabLayerCodecError::NotAdmitted { codec: e8p })
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, w3, e8p]),
            Err(SlabLayerCodecError::NotAdmitted { codec: e8p })
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, w2]),
            Err(SlabLayerCodecError::NotAdmitted { codec: w2 })
        );
    }

    #[test]
    fn merged_slab_admission_matrix_preserves_certified_arch_pairs() {
        let (mq4, w3, w2, e8p) = (SLAB_CODEC_MQ4, SLAB_CODEC_W3, 2u8, SLAB_CODEC_E8P);
        // Full final matrix over codecs 0, 1, 2, 3 and an unknown byte:
        // gfx1100 {0,3}, gfx1151 {0,1}, gfx1201 {0,1,3}; W2 nowhere.
        let matrix: [(&str, [bool; 5]); 3] = [
            ("gfx1100", [true, false, false, true, false]),
            ("gfx1151", [true, true, false, false, false]),
            ("gfx1201", [true, true, false, true, false]),
        ];
        for (arch, row) in matrix {
            for (codec, want) in [0u8, 1, 2, 3, 255].into_iter().zip(row) {
                assert_eq!(
                    qwen4_slab_codec_admitted(arch, codec),
                    want,
                    "{arch} codec {codec}"
                );
            }
            assert!(!qwen4_slab_codec_admitted(arch, w2), "{arch} W2");
            // cold0 means no cold bucket on every admitted arch, Halo included.
            assert_eq!(
                qwen4_slab_default_cold_codec(arch),
                Some(SLAB_CODEC_MQ4),
                "{arch}"
            );
            assert_eq!(
                qwen4_slab_layer_cold_codec(arch, &[mq4, mq4]),
                Ok(mq4),
                "{arch}"
            );
        }
        // gfx1151 never admits E8P, and W3 is the only cold codec it can select.
        assert!(!qwen4_slab_codec_admitted("gfx1151", e8p));
        assert_eq!(qwen4_slab_layer_cold_codec("gfx1151", &[w3, w3]), Ok(w3));
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, w3, mq4]),
            Ok(w3)
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[mq4, e8p]),
            Err(SlabLayerCodecError::NotAdmitted { codec: e8p })
        );
        // Multiple cold codecs: refused where both are admitted, NotAdmitted on the rest.
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1201", &[mq4, w3, e8p]),
            Err(SlabLayerCodecError::MixedCold {
                first: w3,
                second: e8p
            })
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1151", &[w3, e8p]),
            Err(SlabLayerCodecError::NotAdmitted { codec: e8p })
        );
        assert_eq!(
            qwen4_slab_layer_cold_codec("gfx1100", &[e8p, w3]),
            Err(SlabLayerCodecError::NotAdmitted { codec: w3 })
        );
        // Unknown and near-miss architecture names serve nothing and have no default cold codec.
        for arch in [
            "", "unknown", "gfx942", "gfx1150", "gfx1152", "gfx11510", "gfx115", "gfx1151x",
            "gfx1151 ", "GFX1151", "gfx1101", "gfx1200", "gfx1201 ",
        ] {
            for codec in [0u8, 1, 2, 3, 255] {
                assert!(
                    !qwen4_slab_codec_admitted(arch, codec),
                    "{arch:?} codec {codec}"
                );
            }
            assert_eq!(qwen4_slab_default_cold_codec(arch), None, "{arch:?}");
            assert!(
                qwen4_slab_layer_cold_codec(arch, &[mq4]).is_err(),
                "{arch:?}"
            );
        }
    }
}
