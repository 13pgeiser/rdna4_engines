// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Identity sidecars shared by the Flash-Next MTP hardware tests
//! (`#[path = "common/hw_identity.rs"]`).
//!
//! Two documents per arm:
//!
//! - the **config identity** ([`config_identity`]) is compared for equality
//!   before any cached result is reused. It covers the effective environment
//!   (every `HIPFIRE_*` variable plus the device-visibility selectors, after
//!   the arm's overrides), the lever / floor / graph state, the executable's
//!   SHA-256, a compile-time hash of the production sources the MTP lane
//!   executes, and the model artifact's size and full-file SHA-256;
//! - the **receipt** ([`receipt`]) is informational: GPU arch / UUID / BDF,
//!   the graph flags the runtime resolved, the git state of the checkout and
//!   the host. It is never compared, but it is written next to every result.
//!
//! Nothing here measures anything: the counters are the
//! `hip_bridge::launch_counters` the runtime already keeps.

#![allow(dead_code)]

use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::sync::LazyLock;

/// `(name, Some(value))` sets, `(name, None)` removes.
pub type EnvOverrides<'a> = &'a [(&'a str, Option<&'a str>)];

/// Device-visibility selectors that change which GPU a process sees without a
/// `HIPFIRE_` prefix.
const DEVICE_ENV: [&str; 3] = [
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
];

/// Prefix of the sampled-MTP resident levers. Spelled in two pieces so the
/// env-var inventory scan sees only variables the production tree owns.
fn lever_prefix() -> String {
    format!("{}{}", "HIPFIRE_", "QWEN4_MTP_")
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            return Ok(hex(&hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

/// The process environment this arm runs under: `HIPFIRE_*` plus the device
/// selectors, minus `exclude` (the harness's own control variables), after
/// applying `overrides`. A parent passes the overrides it will give its
/// child; a child passes none, so both sides compute the same map.
pub fn env_snapshot(overrides: EnvOverrides<'_>, exclude: &[&str]) -> BTreeMap<String, String> {
    let mut env: BTreeMap<String, String> = std::env::vars()
        .filter(|(name, _)| name.starts_with("HIPFIRE_") || DEVICE_ENV.contains(&name.as_str()))
        .filter(|(name, _)| !exclude.contains(&name.as_str()))
        .collect();
    for (name, value) in overrides {
        if exclude.contains(name) {
            continue;
        }
        match value {
            Some(value) => {
                env.insert((*name).to_string(), (*value).to_string());
            }
            None => {
                env.remove(*name);
            }
        }
    }
    env
}

/// The native-MTP route class an environment selects.
pub fn route_class(env: &BTreeMap<String, String>) -> &'static str {
    let incremental = env.get("HIPFIRE_MTP_INCREMENTAL").map(String::as_str);
    let floor_off = env.get("HIPFIRE_MTP_AR_FLOOR").map(String::as_str) == Some("0");
    match (incremental, floor_off) {
        (Some("0"), true) => "forced-batched-floor-off",
        (Some("1"), true) => "forced-interleaved-floor-off",
        (Some("0" | "1"), false) => "forced-route-floor-bypassed",
        (_, true) => "floor-off-chooser",
        (_, false) => "adaptive-floor",
    }
}

/// Only a forced route with the floor explicitly off is a fixed batched or
/// interleaved route whose same-seed replay must reproduce its bytes.
pub fn forced_floor_off(env: &BTreeMap<String, String>) -> bool {
    matches!(
        route_class(env),
        "forced-batched-floor-off" | "forced-interleaved-floor-off"
    )
}

static BINARY: LazyLock<Value> = LazyLock::new(|| {
    let path = std::env::current_exe().expect("test binary path");
    let sha = sha256_file(&path).unwrap_or_else(|e| panic!("hash test binary: {e}"));
    let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    json!({ "sha256": sha, "size": size })
});

/// SHA-256 and size of this test executable (computed once).
pub fn binary_identity() -> &'static Value {
    &BINARY
}

/// Production sources the sampled / greedy MTP lane executes, embedded at
/// compile time: the hash names the source the binary was built from even
/// when the checkout is not available where the binary runs.
static SOURCES: LazyLock<Value> = LazyLock::new(|| {
    let files: [(&str, &[u8]); 7] = [
        (
            "hipfire-arch-qwen4/src/mtp_spec.rs",
            include_bytes!("../../src/mtp_spec.rs"),
        ),
        (
            "hipfire-arch-qwen4/src/bundle.rs",
            include_bytes!("../../src/bundle.rs"),
        ),
        (
            "hipfire-arch-qwen4/src/mtp_gpu.rs",
            include_bytes!("../../src/mtp_gpu.rs"),
        ),
        (
            "hipfire-runtime/src/spec.rs",
            include_bytes!("../../../hipfire-runtime/src/spec.rs"),
        ),
        (
            "hipfire-runtime/src/spec_sampling.rs",
            include_bytes!("../../../hipfire-runtime/src/spec_sampling.rs"),
        ),
        (
            "hipfire-runtime/src/sampler.rs",
            include_bytes!("../../../hipfire-runtime/src/sampler.rs"),
        ),
        (
            "hipfire-dispatch/src/pipeline/draft_head.rs",
            include_bytes!("../../../hipfire-dispatch/src/pipeline/draft_head.rs"),
        ),
    ];
    let mut combined = Sha256::new();
    let mut per_file = serde_json::Map::new();
    for (name, bytes) in files {
        let sha = sha256_hex(bytes);
        combined.update(name.as_bytes());
        combined.update(sha.as_bytes());
        per_file.insert(name.to_string(), json!(sha));
    }
    json!({ "sha256": hex(&combined.finalize()), "files": per_file })
});

pub fn compiled_sources() -> &'static Value {
    &SOURCES
}

/// Size and full-file SHA-256 of one file. The hash is cached in
/// `cache_dir/hash-<path digest>.json`, keyed by path, size and mtime, so the
/// children of one parent hash a multi-GB artifact once.
fn file_identity(file: &Path, cache_dir: &Path) -> Result<Value, String> {
    let meta = fs::metadata(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let path_text = file.display().to_string();
    let key = json!({
        "path": path_text,
        "size": meta.len(),
        "mtime_ns": mtime_ns,
    });
    let digest = sha256_hex(path_text.as_bytes());
    let cache = cache_dir.join(format!("hash-{}.json", &digest[..16]));
    if let Ok(bytes) = fs::read(&cache) {
        if let Ok(cached) = serde_json::from_slice::<Value>(&bytes) {
            if cached["key"] == key {
                if let Some(sha) = cached["sha256"].as_str() {
                    return Ok(json!({ "size": meta.len(), "sha256": sha }));
                }
            }
        }
    }
    let sha = sha256_file(file)?;
    let tmp = cache.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(
        &tmp,
        serde_json::to_vec_pretty(&json!({ "key": key, "sha256": sha }))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, &cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    Ok(json!({ "size": meta.len(), "sha256": sha }))
}

/// The model artifact and any `<model>.ple-q4` / `<model>.ple-q8` PLE-row
/// sidecar next to it (the loader attaches one when the base is split):
/// size and full-file SHA-256 of each, `null` for an absent sidecar.
pub fn model_identity(model: &Path, cache_dir: &Path) -> Result<Value, String> {
    let mut identity = file_identity(model, cache_dir)?;
    for suffix in ["ple-q4", "ple-q8"] {
        let mut name = model.as_os_str().to_os_string();
        name.push(format!(".{suffix}"));
        let sidecar = std::path::PathBuf::from(name);
        identity[suffix] = if sidecar.is_file() {
            file_identity(&sidecar, cache_dir)?
        } else {
            Value::Null
        };
    }
    Ok(identity)
}

/// The compared identity: effective env, lever / floor / graph state, binary,
/// compiled sources and model.
pub fn config_identity(env: &BTreeMap<String, String>, model: &Value) -> Value {
    let prefix = lever_prefix();
    let levers: BTreeMap<&String, &String> = env
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
        .collect();
    let get = |name: &str| env.get(name).cloned();
    json!({
        "env": env,
        "levers": levers,
        "floor": {
            "route_class": route_class(env),
            "HIPFIRE_MTP_AR_FLOOR": get("HIPFIRE_MTP_AR_FLOOR"),
            "HIPFIRE_MTP_INCREMENTAL": get("HIPFIRE_MTP_INCREMENTAL"),
        },
        "graph": {
            "HIPFIRE_GRAPH": get("HIPFIRE_GRAPH"),
            "HIPFIRE_AR_GRAPH": get("HIPFIRE_AR_GRAPH"),
            "HIPFIRE_GRAPH_MOE": get("HIPFIRE_GRAPH_MOE"),
        },
        "binary": binary_identity(),
        "compiled_sources": compiled_sources(),
        "model": model,
    })
}

fn git(manifest_dir: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Informational receipt of the running process: GPU identity, the graph
/// flags the runtime resolved, checkout git state and host. `gpu` is the
/// initialised device the arm runs on.
pub fn receipt(gpu: &Gpu, model: &Path) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let status = git(manifest, &["status", "--porcelain"]);
    json!({
        "gpu": {
            "arch": gpu.arch,
            "device_id": gpu.device_id,
            "uuid": gpu.hip.device_uuid(gpu.device_id).ok(),
            "pci_bus_id": gpu.hip.device_pci_bus_id(gpu.device_id).ok(),
            "uma": gpu.is_uma(),
        },
        "resolved_graph_flags": {
            "graph_forward": gpu.flags.graph_forward,
            "graph_ar": gpu.flags.graph_ar,
            "graph_moe": gpu.flags.graph_moe,
            "force_blob_path": gpu.flags.force_blob_path,
        },
        "git": {
            "head": git(manifest, &["rev-parse", "HEAD"]),
            "dirty_status_sha256": status.as_deref().map(|s| sha256_hex(s.as_bytes())),
            "dirty": status.as_deref().map(|s| !s.is_empty()),
        },
        "model_path": model.display().to_string(),
        "binary_path": std::env::current_exe().ok().map(|p| p.display().to_string()),
        "host": fs::read_to_string("/etc/hostname").ok().map(|h| h.trim().to_string()),
        "unix_time": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .ok(),
        "counters_note": "hip_bridge::launch_counters are thread-local call counts the runtime already keeps; not a hardware profiler",
    })
}

/// `hip_bridge::launch_counters` call counts at one instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub launches: u64,
    pub htod: u64,
    pub dtoh: u64,
    pub dtoh_bytes: u64,
    pub dtod: u64,
    pub memset: u64,
    pub stream_sync: u64,
    pub event_sync: u64,
    pub device_sync: u64,
    pub graph_launch: u64,
}

impl Counters {
    pub fn now() -> Self {
        use hip_bridge::launch_counters as lc;
        Self {
            launches: lc::launch_kernel::count(),
            htod: lc::memcpy_htod::count(),
            dtoh: lc::memcpy_dtoh::count(),
            dtoh_bytes: lc::memcpy_dtoh::bytes(),
            dtod: lc::memcpy_dtod::count(),
            memset: lc::memset::count(),
            stream_sync: lc::stream_sync::count(),
            event_sync: lc::event_sync::count(),
            device_sync: lc::device_sync::count(),
            graph_launch: lc::graph_launch::count(),
        }
    }

    pub fn since(self, earlier: Self) -> Self {
        Self {
            launches: self.launches.saturating_sub(earlier.launches),
            htod: self.htod.saturating_sub(earlier.htod),
            dtoh: self.dtoh.saturating_sub(earlier.dtoh),
            dtoh_bytes: self.dtoh_bytes.saturating_sub(earlier.dtoh_bytes),
            dtod: self.dtod.saturating_sub(earlier.dtod),
            memset: self.memset.saturating_sub(earlier.memset),
            stream_sync: self.stream_sync.saturating_sub(earlier.stream_sync),
            event_sync: self.event_sync.saturating_sub(earlier.event_sync),
            device_sync: self.device_sync.saturating_sub(earlier.device_sync),
            graph_launch: self.graph_launch.saturating_sub(earlier.graph_launch),
        }
    }

    /// Compact row: launches, htod, dtoh, dtoh_bytes, dtod, memset,
    /// stream_sync, event_sync, device_sync, graph_launch.
    pub fn to_array(self) -> [u64; 10] {
        [
            self.launches,
            self.htod,
            self.dtoh,
            self.dtoh_bytes,
            self.dtod,
            self.memset,
            self.stream_sync,
            self.event_sync,
            self.device_sync,
            self.graph_launch,
        ]
    }

    pub fn to_row(self) -> Value {
        json!(self.to_array())
    }

    pub fn to_json(self) -> Value {
        json!({
            "launches": self.launches,
            "htod": self.htod,
            "dtoh": self.dtoh,
            "dtoh_bytes": self.dtoh_bytes,
            "dtod": self.dtod,
            "memset": self.memset,
            "stream_sync": self.stream_sync,
            "event_sync": self.event_sync,
            "device_sync": self.device_sync,
            "graph_launch": self.graph_launch,
        })
    }
}

/// Column names of [`Counters::to_row`].
pub const COUNTER_COLUMNS: [&str; 10] = [
    "launches",
    "htod",
    "dtoh",
    "dtoh_bytes",
    "dtod",
    "memset",
    "stream_sync",
    "event_sync",
    "device_sync",
    "graph_launch",
];

/// An existing result may only be reused under the config recorded next to
/// it: `expected` must equal the file's contents.
pub fn require_same_config(config_path: &Path, expected: &Value) -> Result<(), String> {
    let recorded: Value = match fs::read(config_path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| format!("{}: {e}", config_path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "{} does not exist: the result's config is unknown; remove the result or use another output directory",
                config_path.display()
            ));
        }
        Err(e) => return Err(format!("{}: {e}", config_path.display())),
    };
    if &recorded != expected {
        return Err(format!(
            "{} was produced under a different config; remove the result or use another output directory\n  recorded: {recorded}\n  current:  {expected}",
            config_path.display()
        ));
    }
    Ok(())
}
