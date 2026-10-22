// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `mq6_i8_gemm::Spec` (and the product module) re-emits the bytes recorded
//! before the building blocks (`mq6_i8_gemm::blocks`) were factored out of `emit`: the sha256 of
//! `emit(spec).s_text` per spec and of the `emit_module` text, computed from beta's unrefactored
//! generator. A change to any digest is a change to a production kernel and must be intentional.

use peacemaker_isa::kernels::mq6_i8_gemm::{emit, emit_module, Epi, Raster, Spec, Tile};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "128x128x8-set",
        "cde3c939da7fef1282a34a514ce187a93474e0b5e54b4c9874ca3e9ac178d773",
    ),
    (
        "128x128x8-set-linear",
        "65328b5ba08afd79237fd8577322e1fdd15b04c726523fc264594b8db7e7d87a",
    ),
    (
        "128x128x8-set_bf16",
        "4602e51634daec1f60fdb3ae820b6e1f80d86788f1646a141fc8820d84a2ae29",
    ),
    (
        "128x128x8-set_bf16-linear",
        "1fef4d95fa27fac5c4b646d3677a31d545b63edde4a16652b6079bd8385de4a6",
    ),
];

/// `(variant name, sha256 of s_text)` of the non-shipped tiles.
const PINNED_TILES: &[(&str, &str)] = &[
    (
        "256x128x16-set",
        "1c93780150e77331b2687ebcfe926fc63e165cc9a766b715221faf5d9a60b566",
    ),
    (
        "256x128x16-set-linear",
        "548778c7ff87624854e679ef5dc02b280fefd1e59a58d1d55df4f2c6f644f437",
    ),
    (
        "256x128x16-set_bf16",
        "35206e7cc71638fde065846d8fa7925f1f3e2492ad77f1704d650943fc98ef63",
    ),
    (
        "256x128x16-set_bf16-linear",
        "9873fca002dfb5014cc2fd89ed27b296d8f38f685d039730401bdbe290694165",
    ),
    (
        "128x256x16-set",
        "432d787f344855239aec302e06f73d5d30d4a8ce9f9f1a75a53321a44e4c9acf",
    ),
    (
        "128x256x16-set-linear",
        "dd89af59d4cd69a2115f826a1d27b49e084556cbc4020b9cbcd9b1166aca6b1a",
    ),
    (
        "128x256x16-set_bf16",
        "ee5ff709200f36f4ad91d583e7c44ed28d15649743542d4c6b3a24ccdf07d0be",
    ),
    (
        "128x256x16-set_bf16-linear",
        "498262d4189185a10683d1797f8c73cd5a114c2470ab5f7d3a96b610cab26acc",
    ),
];

/// sha256 of the `emit_module` text (both banded product symbols).
const PINNED_MODULE: &str = "ee29f4a2b680749d24e3552ba1613cd8445f6c5bec2158830fcc647705106d1c";

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn digests(tiles: &[Tile]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for &tile in tiles {
        for epi in [Epi::Set, Epi::SetBf16] {
            for raster in [Raster::Banded, Raster::Linear] {
                let spec = Spec {
                    epi,
                    raster,
                    tile,
                    arch: Arch::Gfx1201,
                };
                if spec.validate().is_err() {
                    continue;
                }
                let e = emit(spec).unwrap_or_else(|e| {
                    panic!("{} is valid but does not emit: {e}", spec.variant_name())
                });
                out.push((spec.variant_name(), sha(&e.s_text)));
            }
        }
    }
    out
}

fn check(tiles: &[Tile], pinned: &[(&str, &str)], what: &str) {
    let got = digests(tiles);
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, pinned, "{what} emitted bytes changed");
}

#[test]
fn mq6_i8_gemm_emit_digests_pinned() {
    check(&[Tile::T128x128], PINNED, "mq6_i8_gemm");
}

/// The tile axis: the 256x128 and 128x256 tiles, pinned once each was bit-equal to the exact
/// reference and to the shipped tile on the interpreter (`tests/mq6_i8_gemm.rs`). These digests
/// guard them; the shipped tile's are `PINNED`.
#[test]
fn mq6_i8_gemm_tile_digests_pinned() {
    check(
        &[Tile::T256x128, Tile::T128x256],
        PINNED_TILES,
        "mq6_i8_gemm tile",
    );
}

#[test]
fn mq6_i8_gemm_module_digest_pinned() {
    let (_, text, proof) = emit_module(Arch::Gfx1201).unwrap();
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        println!("module {}", sha(&text));
    }
    assert_eq!(sha(&text), PINNED_MODULE, "mq6_i8_gemm module text changed");
    assert_eq!(proof.s_text_sha256, sha(&text));
}
