// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Verification cache: remembers that a large local artifact (the 125 GB pinned
//! Flash-Next monolithic) already hashed to its pin, so resolving it on every
//! load does not read it end to end again.
//!
//! One JSON file under the hipfire home (`<HIPFIRE_HOME>/cache/verified-artifacts.json`)
//! holds one record per verified path: device, inode, size, mtime (seconds and
//! nanoseconds), the verified SHA-256 and when it was verified. A lookup is a
//! hit only when the file's current stat equals the record **and** the record's
//! SHA-256 equals the digest the caller expects, so a record for another pin
//! never satisfies. Anything else hashes, then writes the record atomically
//! (temporary file + rename). The cache is an optimisation: a missing, corrupt
//! or unwritable cache file is ignored and never fails a resolve.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;

/// The stat fields a record is keyed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StatKey {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
}

impl StatKey {
    pub(crate) fn of(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                dev: metadata.dev(),
                ino: metadata.ino(),
                size: metadata.len(),
                mtime_sec: metadata.mtime(),
                mtime_nsec: metadata.mtime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok());
            Self {
                dev: 0,
                ino: 0,
                size: metadata.len(),
                mtime_sec: modified.map_or(0, |time| time.as_secs() as i64),
                mtime_nsec: modified.map_or(0, |time| i64::from(time.subsec_nanos())),
            }
        }
    }
}

/// One verified path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Record {
    #[serde(flatten)]
    pub stat: StatKey,
    pub sha256: String,
    /// Unix seconds when the bytes were hashed.
    pub verified_at: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    records: BTreeMap<String, Record>,
}

/// The on-disk verification cache, or a disabled stand-in.
#[derive(Clone, Debug)]
pub(crate) struct VerifiedCache {
    file: Option<PathBuf>,
}

impl VerifiedCache {
    /// The cache of the hipfire home `root`: `<root>/cache/verified-artifacts.json`.
    pub(crate) fn in_home(root: &Path) -> Self {
        Self::at(root.join("cache").join("verified-artifacts.json"))
    }

    /// A cache stored in `file`.
    pub(crate) fn at(file: PathBuf) -> Self {
        Self { file: Some(file) }
    }

    /// A cache that never hits and never writes.
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self { file: None }
    }

    /// Every record of the cache file; a missing or corrupt file reads as empty.
    fn load(&self) -> BTreeMap<String, Record> {
        let Some(file) = &self.file else {
            return BTreeMap::new();
        };
        fs::read(file)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheFile>(&bytes).ok())
            .filter(|cache| cache.version == CACHE_VERSION)
            .map(|cache| cache.records)
            .unwrap_or_default()
    }

    /// The record of `path`, if any (test and diagnostics access).
    #[cfg(test)]
    pub(crate) fn record_of(&self, path: &Path) -> Option<Record> {
        self.load().remove(&key_of(path))
    }

    /// True when `path` (whose current stat is `current`) was verified against
    /// `expected_sha256`: the record's stat equals `current` and its digest
    /// equals the expected one.
    pub(crate) fn satisfies(&self, path: &Path, current: &StatKey, expected_sha256: &str) -> bool {
        self.load().get(&key_of(path)).is_some_and(|record| {
            record.stat == *current && record.sha256.eq_ignore_ascii_case(expected_sha256)
        })
    }

    /// Record that `path`, whose stat at hashing time was `stat`, hashed to
    /// `sha256`. Written atomically; any failure leaves the cache as it was.
    pub(crate) fn remember(&self, path: &Path, stat: StatKey, sha256: &str) {
        let Some(file) = &self.file else {
            return;
        };
        let mut records = self.load();
        records.insert(
            key_of(path),
            Record {
                stat,
                sha256: sha256.to_ascii_lowercase(),
                verified_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_secs()),
            },
        );
        let _ = write_atomically(
            file,
            &CacheFile {
                version: CACHE_VERSION,
                records,
            },
        );
    }
}

fn key_of(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn write_atomically(file: &Path, cache: &CacheFile) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(cache).map_err(std::io::Error::other)?;
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = file.as_os_str().to_owned();
    temporary.push(format!(".tmp.{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    let result = fs::write(&temporary, bytes).and_then(|()| fs::rename(&temporary, file));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-verified-cache-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    const SHA: &str = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";

    fn stat(ino: u64) -> StatKey {
        StatKey {
            dev: 7,
            ino,
            size: 125,
            mtime_sec: 1_700_000_000,
            mtime_nsec: 123_456_789,
        }
    }

    #[test]
    fn a_remembered_stat_and_digest_satisfy_only_an_exact_match() {
        let home = scratch("exact");
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/models/x.mq4-xts");
        assert!(!cache.satisfies(model, &stat(1), SHA));
        cache.remember(model, stat(1), SHA);
        assert!(cache.satisfies(model, &stat(1), SHA));
        assert!(cache.satisfies(model, &stat(1), &SHA.to_ascii_uppercase()));
        // Any stat field change, or another expected digest, does not satisfy.
        for changed in [
            StatKey { dev: 8, ..stat(1) },
            stat(2),
            StatKey {
                size: 126,
                ..stat(1)
            },
            StatKey {
                mtime_sec: 1_700_000_001,
                ..stat(1)
            },
            StatKey {
                mtime_nsec: 1,
                ..stat(1)
            },
        ] {
            assert!(!cache.satisfies(model, &changed, SHA), "{changed:?}");
        }
        assert!(!cache.satisfies(model, &stat(1), &"a".repeat(64)));
        assert!(!cache.satisfies(Path::new("/models/other"), &stat(1), SHA));
        let record = cache.record_of(model).unwrap();
        assert_eq!(record.stat, stat(1));
        assert_eq!(record.sha256, SHA);
        assert!(record.verified_at > 0);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn records_are_one_per_path_and_replaced_in_place() {
        let home = scratch("replace");
        let cache = VerifiedCache::in_home(&home);
        let (a, b) = (Path::new("/m/a"), Path::new("/m/b"));
        cache.remember(a, stat(1), SHA);
        cache.remember(b, stat(2), SHA);
        cache.remember(a, stat(3), SHA);
        assert_eq!(cache.record_of(a).unwrap().stat, stat(3));
        assert_eq!(cache.record_of(b).unwrap().stat, stat(2));
        let written: serde_json::Value =
            serde_json::from_slice(&fs::read(home.join("cache/verified-artifacts.json")).unwrap())
                .unwrap();
        assert_eq!(written["records"].as_object().unwrap().len(), 2);
        let leftovers: Vec<_> = fs::read_dir(home.join("cache"))
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_corrupt_or_foreign_cache_file_is_ignored_and_replaced() {
        let home = scratch("corrupt");
        let file = home.join("cache/verified-artifacts.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/m/x");
        for bytes in [
            &b"{ this is not json"[..],
            b"",
            b"[]",
            br#"{"version":99,"records":{}}"#,
            br#"{"version":1,"records":{"/m/x":{"dev":"nope"}}}"#,
        ] {
            fs::write(&file, bytes).unwrap();
            assert!(!cache.satisfies(model, &stat(1), SHA), "{bytes:?}");
            cache.remember(model, stat(1), SHA);
            assert!(cache.satisfies(model, &stat(1), SHA), "{bytes:?}");
        }
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn an_unwritable_cache_never_fails_and_a_disabled_cache_never_hits() {
        let home = scratch("unwritable");
        // The cache directory's parent is a regular file: creating it fails.
        let blocker = home.join("blocker");
        fs::write(&blocker, b"x").unwrap();
        let cache = VerifiedCache::in_home(&blocker);
        cache.remember(Path::new("/m/x"), stat(1), SHA);
        assert!(!cache.satisfies(Path::new("/m/x"), &stat(1), SHA));
        let disabled = VerifiedCache::disabled();
        disabled.remember(Path::new("/m/x"), stat(1), SHA);
        assert!(!disabled.satisfies(Path::new("/m/x"), &stat(1), SHA));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn stat_keys_follow_the_file_metadata() {
        let home = scratch("stat");
        let file = home.join("artifact");
        fs::write(&file, b"payload").unwrap();
        let first = StatKey::of(&fs::metadata(&file).unwrap());
        assert_eq!(first.size, 7);
        assert_eq!(first, StatKey::of(&fs::metadata(&file).unwrap()));
        fs::write(&file, b"longer payload").unwrap();
        assert_ne!(first, StatKey::of(&fs::metadata(&file).unwrap()));
        fs::remove_dir_all(&home).unwrap();
    }
}
