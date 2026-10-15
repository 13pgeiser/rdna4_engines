// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lever1 expert hotset sidecar: parser, validator and enabling policy.
//!
//! A sidecar is an immutable JSON file (`kind = hipfire.qwen4.expert_hotset`,
//! `version = 1`) that ranks every routed expert of the trunk by how often a
//! decode capture routed to it. A global expert id is
//! `layer * experts_per_layer + expert`. The loader fills the longest ranked
//! prefix that fits its runtime byte budget; the budget is never read from
//! the file.
//!
//! # Ranking rule (enforced)
//!
//! `rank` is a permutation of every global id exactly once, ordered by
//! `counts[id]` descending with ties broken by global id ascending.
//! `counts` is indexed by global id and has one entry per expert.
//! `ranking_rule` and `corpus` are provenance strings and are carried, not
//! interpreted.
//!
//! This module hashes only the sidecar bytes ([`HotsetSidecar::sha256`]).
//! `model_sha256`, `counts_sha256`, `format_map_sha256` are validated as full
//! lowercase 64-hex digests and exposed for the loader to compare against its
//! own pins; no model artifact is read here.
//!
//! No GPU access; the only I/O is [`HotsetSidecar::load`] reading one file
//! and [`enabled_hotset_path`] reading the process configuration snapshot.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::expert_residency::EXPERT_VRAM_LAYERS_ENV;

/// Absolute path of the sidecar. Absent: the hotset is off. This is the only
/// variable that enables the hotset.
pub const HOTSET_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOTSET";
/// Kill switch: exactly `0` bypasses [`HOTSET_ENV`] entirely (the sidecar is
/// not even looked at). Unset or `1` leaves [`HOTSET_ENV`] in charge.
pub const HOTCACHE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOTCACHE";
/// `kind` of a v1 sidecar.
pub const SIDECAR_KIND: &str = "hipfire.qwen4.expert_hotset";
/// `version` of the sidecar schema this module reads.
pub const SIDECAR_VERSION: u64 = 1;

/// Pure policy behind [`enabled_hotset_path`]: the three env values in, the
/// sidecar path (or `None` = hotset off) out.
///
/// * `hotcache`: exactly `0` = off, returned before anything else about
///   `hotset` or `vram_layers` is examined. Otherwise it is looked at only
///   once `hotset` is set: unset or `1` = on, anything else is an error.
/// * `hotset`: unset = off. Set = the sidecar path, which must be non-empty
///   and absolute.
/// * `vram_layers`: any set value (the legacy
///   [`EXPERT_VRAM_LAYERS_ENV`], even `auto` or `0`) together with an
///   enabled hotset is refused: the two decide expert placement differently.
pub fn resolve_hotset_path(
    hotset: Option<&str>,
    hotcache: Option<&str>,
    vram_layers: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    if hotcache.map(str::trim) == Some("0") {
        return Ok(None);
    }
    let Some(hotset) = hotset else {
        return Ok(None);
    };
    match hotcache.map(str::trim) {
        None | Some("1") => {}
        Some(other) => {
            return Err(format!("{HOTCACHE_ENV}={other:?} is neither `0` nor `1`"));
        }
    }
    if hotset.trim().is_empty() {
        return Err(format!("{HOTSET_ENV} is set but empty; unset it or give an absolute sidecar path"));
    }
    let path = PathBuf::from(hotset);
    if !path.is_absolute() {
        return Err(format!("{HOTSET_ENV}={hotset:?} is not an absolute path"));
    }
    if let Some(layers) = vram_layers {
        return Err(format!(
            "{HOTSET_ENV} and {EXPERT_VRAM_LAYERS_ENV}={layers:?} are both set; the hotset \
             chooses expert placement itself, so unset {EXPERT_VRAM_LAYERS_ENV} (or set \
             {HOTCACHE_ENV}=0 to bypass the hotset)"
        ));
    }
    Ok(Some(path))
}

/// The sidecar path the process is configured to use: [`HOTSET_ENV`] gated by
/// [`HOTCACHE_ENV`] and checked against an explicit
/// [`EXPERT_VRAM_LAYERS_ENV`], read from the process configuration snapshot.
/// `Ok(None)` = hotset off (today's placement). `Err` = refuse the load.
pub fn enabled_hotset_path() -> Result<Option<PathBuf>, String> {
    let var = |name: &str| hipfire_config::developer_var(name).ok();
    resolve_hotset_path(
        var(HOTSET_ENV).as_deref(),
        var(HOTCACHE_ENV).as_deref(),
        var(EXPERT_VRAM_LAYERS_ENV).as_deref(),
    )
}

/// Wire form; unknown keys are refused so v1 stays closed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSidecar {
    version: u64,
    kind: String,
    model_sha256: String,
    counts_sha256: String,
    format_map: String,
    format_map_sha256: String,
    layers: u64,
    experts_per_layer: u64,
    top_k: u64,
    capture_tokens: u64,
    ranking_rule: String,
    corpus: String,
    rank: Vec<u64>,
    counts: Vec<u64>,
}

/// A validated v1 hotset sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotsetSidecar {
    sha256: String,
    model_sha256: String,
    counts_sha256: String,
    format_map: String,
    format_map_sha256: String,
    layers: usize,
    experts_per_layer: usize,
    top_k: usize,
    capture_tokens: u64,
    ranking_rule: String,
    corpus: String,
    rank: Vec<u32>,
    counts: Vec<u64>,
}

impl HotsetSidecar {
    /// Read and validate the sidecar at `path` against the model's trunk
    /// dimensions. Errors name the path.
    pub fn load(
        path: impl AsRef<Path>,
        layers: usize,
        experts_per_layer: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| format!("{HOTSET_ENV}: cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes, layers, experts_per_layer, top_k)
            .map_err(|e| format!("{HOTSET_ENV}: invalid sidecar {}: {e}", path.display()))
    }

    /// Validate sidecar `bytes` (the pure core of [`Self::load`]).
    pub fn from_bytes(
        bytes: &[u8],
        layers: usize,
        experts_per_layer: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let sha256 = hex(&Sha256::digest(bytes));
        let raw: RawSidecar =
            serde_json::from_slice(bytes).map_err(|e| format!("not a v1 hotset JSON: {e}"))?;

        if raw.kind != SIDECAR_KIND {
            return Err(format!("kind {:?}, expected {SIDECAR_KIND:?}", raw.kind));
        }
        if raw.version != SIDECAR_VERSION {
            return Err(format!("version {}, expected {SIDECAR_VERSION}", raw.version));
        }
        for (name, value) in [
            ("model_sha256", &raw.model_sha256),
            ("counts_sha256", &raw.counts_sha256),
            ("format_map_sha256", &raw.format_map_sha256),
        ] {
            check_sha256(name, value)?;
        }
        for (name, value) in [
            ("format_map", &raw.format_map),
            ("ranking_rule", &raw.ranking_rule),
            ("corpus", &raw.corpus),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} is empty"));
            }
        }

        check_dim("layers", raw.layers, layers)?;
        check_dim("experts_per_layer", raw.experts_per_layer, experts_per_layer)?;
        check_dim("top_k", raw.top_k, top_k)?;
        if raw.capture_tokens == 0 {
            return Err("capture_tokens is 0".to_owned());
        }
        let total = layers
            .checked_mul(experts_per_layer)
            .filter(|&total| total > 0 && u32::try_from(total).is_ok())
            .ok_or_else(|| {
                format!("layers {layers} x experts_per_layer {experts_per_layer} is not a valid expert count")
            })?;

        if raw.counts.len() != total {
            return Err(format!(
                "counts has {} entries, expected {total} (layers {layers} x experts_per_layer \
                 {experts_per_layer})",
                raw.counts.len()
            ));
        }

        // Every id exactly once: range and duplicates in one pass, then the
        // first missing id if the list is short.
        let mut seen = vec![false; total];
        let mut rank = Vec::with_capacity(raw.rank.len());
        for (position, &id) in raw.rank.iter().enumerate() {
            let slot = usize::try_from(id).ok().filter(|&slot| slot < total).ok_or_else(|| {
                format!("rank[{position}] = {id} is out of range 0..{total}")
            })?;
            if std::mem::replace(&mut seen[slot], true) {
                return Err(format!("rank[{position}] = {id} is a duplicate"));
            }
            rank.push(slot as u32);
        }
        if rank.len() != total {
            let missing = seen.iter().position(|&seen| !seen).unwrap_or(total);
            return Err(format!(
                "rank has {} ids, expected {total}; first missing id {missing}",
                rank.len()
            ));
        }

        // Descending count, ties by ascending global id.
        for (position, pair) in rank.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let (count_a, count_b) = (raw.counts[a as usize], raw.counts[b as usize]);
            if count_a < count_b || (count_a == count_b && a > b) {
                return Err(format!(
                    "rank is not (count descending, id ascending) at positions {position}..{}: \
                     id {a} count {count_a} precedes id {b} count {count_b}",
                    position + 1
                ));
            }
        }

        Ok(Self {
            sha256,
            model_sha256: raw.model_sha256,
            counts_sha256: raw.counts_sha256,
            format_map: raw.format_map,
            format_map_sha256: raw.format_map_sha256,
            layers,
            experts_per_layer,
            top_k,
            capture_tokens: raw.capture_tokens,
            ranking_rule: raw.ranking_rule,
            corpus: raw.corpus,
            rank,
            counts: raw.counts,
        })
    }

    /// Every global expert id (`layer * experts_per_layer + expert`) exactly
    /// once, hottest first (count descending, id ascending).
    pub fn rank(&self) -> &[u32] {
        &self.rank
    }

    /// Decode routing counts indexed by global expert id.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Count of global expert `id`. Panics when `id` is out of range.
    pub fn count(&self, id: u32) -> u64 {
        self.counts[id as usize]
    }

    /// `(layer, expert)` of a global expert id.
    pub fn split_id(&self, id: u32) -> (usize, usize) {
        let id = id as usize;
        (id / self.experts_per_layer, id % self.experts_per_layer)
    }

    /// Lowercase hex SHA-256 of the sidecar file bytes (for the load log).
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Full SHA-256 of the model artifact the counts were captured on. The
    /// loader compares it with its own pin; it is not verified here.
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    /// Full SHA-256 of the source counts the ranking derives from.
    pub fn counts_sha256(&self) -> &str {
        &self.counts_sha256
    }

    /// Human-readable expert format map the sidecar was built for.
    pub fn format_map(&self) -> &str {
        &self.format_map
    }

    /// Full SHA-256 of the format map.
    pub fn format_map_sha256(&self) -> &str {
        &self.format_map_sha256
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Tokens of the decode capture the counts come from.
    pub fn capture_tokens(&self) -> u64 {
        self.capture_tokens
    }

    /// Provenance text describing how `rank` was derived.
    pub fn ranking_rule(&self) -> &str {
        &self.ranking_rule
    }

    /// Provenance text describing the capture corpus.
    pub fn corpus(&self) -> &str {
        &self.corpus
    }
}

fn check_dim(name: &str, got: u64, expected: usize) -> Result<(), String> {
    if usize::try_from(got) == Ok(expected) {
        Ok(())
    } else {
        Err(format!("{name} is {got}, the model has {expected}"))
    }
}

fn check_sha256(name: &str, value: &str) -> Result<(), String> {
    let full = value.len() == 64 && value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if full {
        Ok(())
    } else {
        Err(format!("{name} {value:?} is not a full lowercase 64-hex SHA-256"))
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const LAYERS: usize = 2;
    const EXPERTS: usize = 3;
    const TOP_K: usize = 2;
    const SHA_A: &str = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";

    /// Counts with a tie (ids 1 and 4 both 7) and a zero.
    fn counts() -> Vec<u64> {
        vec![5, 7, 0, 9, 7, 1]
    }

    fn ranked(counts: &[u64]) -> Vec<u64> {
        let mut ids: Vec<u64> = (0..counts.len() as u64).collect();
        ids.sort_by_key(|&id| (std::cmp::Reverse(counts[id as usize]), id));
        ids
    }

    fn sidecar() -> Value {
        let counts = counts();
        json!({
            "version": 1,
            "kind": SIDECAR_KIND,
            "model_sha256": SHA_A,
            "counts_sha256": SHA_A,
            "format_map": "all-mq4-xts",
            "format_map_sha256": SHA_A,
            "layers": LAYERS,
            "experts_per_layer": EXPERTS,
            "top_k": TOP_K,
            "capture_tokens": 100,
            "ranking_rule": "descending count, ties by global id ascending",
            "corpus": "fixture",
            "rank": ranked(&counts),
            "counts": counts,
        })
    }

    fn parse(value: &Value) -> Result<HotsetSidecar, String> {
        HotsetSidecar::from_bytes(value.to_string().as_bytes(), LAYERS, EXPERTS, TOP_K)
    }

    fn refused(value: &Value, needle: &str) {
        let err = parse(value).expect_err("sidecar must be refused");
        assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    }

    #[test]
    fn valid_sidecar_loads_with_tie_ordered_rank() {
        let parsed = parse(&sidecar()).unwrap();
        assert_eq!(parsed.rank(), &[3, 1, 4, 0, 5, 2]);
        assert_eq!(parsed.counts(), counts().as_slice());
        assert_eq!(parsed.count(3), 9);
        assert_eq!(parsed.split_id(4), (1, 1));
        assert_eq!(parsed.model_sha256(), SHA_A);
        assert_eq!(parsed.capture_tokens(), 100);
        let bytes = sidecar().to_string();
        assert_eq!(parsed.sha256(), hex(&Sha256::digest(bytes.as_bytes())));
    }

    #[test]
    fn missing_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5]);
        refused(&value, "first missing id 2");
    }

    #[test]
    fn duplicate_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5, 5]);
        refused(&value, "duplicate");
    }

    #[test]
    fn out_of_range_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5, 6]);
        refused(&value, "out of range");
    }

    #[test]
    fn wrong_counts_length_is_refused() {
        let mut value = sidecar();
        value["counts"] = json!([5, 7, 0, 9, 7]);
        refused(&value, "counts has 5 entries");
    }

    #[test]
    fn wrong_dimensions_are_refused() {
        for (key, bad) in [("layers", 3), ("experts_per_layer", 4), ("top_k", 3)] {
            let mut value = sidecar();
            value[key] = json!(bad);
            refused(&value, key);
        }
        // Caller-side dimensions that disagree with the file.
        let bytes = sidecar().to_string();
        assert!(HotsetSidecar::from_bytes(bytes.as_bytes(), 48, 512, 10).is_err());
    }

    #[test]
    fn wrong_kind_or_version_is_refused() {
        let mut value = sidecar();
        value["kind"] = json!("hipfire.qwen4.something_else");
        refused(&value, "kind");
        let mut value = sidecar();
        value["version"] = json!(2);
        refused(&value, "version");
    }

    #[test]
    fn rank_inconsistent_with_counts_is_refused() {
        // Higher count after lower count.
        let mut value = sidecar();
        value["rank"] = json!([1, 3, 4, 0, 5, 2]);
        refused(&value, "not (count descending, id ascending)");
        // Tie broken by descending id instead of ascending.
        let mut value = sidecar();
        value["rank"] = json!([3, 4, 1, 0, 5, 2]);
        refused(&value, "not (count descending, id ascending)");
    }

    #[test]
    fn hashes_must_be_full_lowercase_hex() {
        for key in ["model_sha256", "counts_sha256", "format_map_sha256"] {
            for bad in ["0d8b5dbc", &SHA_A.to_uppercase(), &SHA_A[..63]] {
                let mut value = sidecar();
                value[key] = json!(bad);
                refused(&value, key);
            }
        }
    }

    #[test]
    fn missing_unknown_or_non_integer_fields_are_refused() {
        let mut value = sidecar();
        value.as_object_mut().unwrap().remove("capture_tokens");
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["surprise"] = json!(1);
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["counts"] = json!([5, 7.5, 0, 9, 7, 1]);
        assert!(parse(&value).is_err());
        assert!(HotsetSidecar::from_bytes(b"not json", LAYERS, EXPERTS, TOP_K).is_err());
    }

    #[test]
    fn load_reads_file_and_names_missing_path() {
        let dir = std::env::temp_dir().join(format!("hipfire-hotset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hotset.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        let parsed = HotsetSidecar::load(&path, LAYERS, EXPERTS, TOP_K).unwrap();
        assert_eq!(parsed.rank().len(), LAYERS * EXPERTS);
        let missing = dir.join("absent.json");
        let err = HotsetSidecar::load(&missing, LAYERS, EXPERTS, TOP_K).unwrap_err();
        assert!(err.contains("absent.json"), "{err}");
    }

    /// The real Lever1 decode sidecar, when present on this machine.
    #[test]
    fn real_decode_sidecar_loads_when_present() {
        let path = Path::new("/home/kaden/qcal/release-0.4.3/emu-sku-notes/lever1/hotset-decode-v1.json");
        if !path.exists() {
            return;
        }
        let parsed = HotsetSidecar::load(path, 48, 512, 10).unwrap();
        assert_eq!(
            parsed.sha256(),
            "6ec1615c8ddd4426104961cb525c5ae8acccdff9361671dc447272a5414ec167"
        );
        assert_eq!(parsed.rank().len(), 24576);
        assert_eq!(parsed.capture_tokens(), 3705);
    }

    #[test]
    fn env_policy_off_without_hotset() {
        assert_eq!(resolve_hotset_path(None, None, None), Ok(None));
        assert_eq!(resolve_hotset_path(None, Some("1"), Some("12")), Ok(None));
        // HOTSET unset: nothing changes, even for an unrecognized HOTCACHE value.
        assert_eq!(resolve_hotset_path(None, Some("garbage"), None), Ok(None));
        assert_eq!(resolve_hotset_path(None, Some("off"), Some("8")), Ok(None));
    }

    #[test]
    fn env_policy_enables_only_explicit_absolute_path() {
        assert_eq!(
            resolve_hotset_path(Some("/tmp/hotset.json"), None, None),
            Ok(Some(PathBuf::from("/tmp/hotset.json")))
        );
        assert_eq!(
            resolve_hotset_path(Some("/tmp/hotset.json"), Some("1"), None),
            Ok(Some(PathBuf::from("/tmp/hotset.json")))
        );
        let relative = resolve_hotset_path(Some("hotset.json"), None, None).unwrap_err();
        assert!(relative.contains("absolute"), "{relative}");
        let empty = resolve_hotset_path(Some("  "), None, None).unwrap_err();
        assert!(empty.contains("empty"), "{empty}");
    }

    #[test]
    fn hotcache_zero_bypasses_hotset_entirely() {
        for hotset in [Some("/tmp/hotset.json"), Some("relative.json"), Some(""), None] {
            assert_eq!(resolve_hotset_path(hotset, Some("0"), Some("8")), Ok(None));
        }
        let bad = resolve_hotset_path(Some("/tmp/hotset.json"), Some("off"), None).unwrap_err();
        assert!(bad.contains(HOTCACHE_ENV), "{bad}");
    }

    #[test]
    fn hotset_with_explicit_vram_layers_is_refused() {
        for layers in ["0", "12", "auto"] {
            let err = resolve_hotset_path(Some("/tmp/hotset.json"), None, Some(layers)).unwrap_err();
            assert!(err.contains(HOTSET_ENV) && err.contains(EXPERT_VRAM_LAYERS_ENV), "{err}");
        }
    }
}
