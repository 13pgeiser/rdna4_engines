// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The one search rule of the Settings tab and the profile wizard.
//!
//! The matcher itself lives in `hipfire_config::intent` so `hipfire config`
//! and the TUI cannot disagree; this module re-exports it and adds the row
//! level rule (plain title first, then the key's schema text).

use hipfire_config::field;
pub use hipfire_config::intent::{field_matches, key_matches};

/// Whether a Settings row matches `query`: case-insensitive on the plain
/// `title`, or on any of the row's canonical `keys` through the shared schema
/// matcher (key, legacy key, env var, help text, ...). A virtual row names
/// every key its choices write. An empty query matches everything.
#[allow(dead_code)] // consumed by the Settings pane rewrite
pub fn row_matches(title: &str, keys: &[&str], query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() {
        return true;
    }
    if title
        .to_ascii_lowercase()
        .contains(&query.to_ascii_lowercase())
    {
        return true;
    }
    keys.iter().any(|key| match field(key) {
        Some(schema) => field_matches(schema, query),
        None => key_matches(key, query),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_query_matches_everything() {
        assert!(row_matches("Anything", &[], ""));
        assert!(row_matches("Anything", &["memory.kv_cache"], "   "));
    }

    #[test]
    fn title_matches_case_insensitively() {
        assert!(row_matches("Prompt processing", &[], "PROMPT"));
        assert!(!row_matches("Prompt processing", &[], "kv"));
    }

    #[test]
    fn schema_keys_match_by_key_legacy_key_and_env() {
        assert!(row_matches("Cache", &["memory.kv_cache"], "kv_cache"));
        assert!(row_matches(
            "Cache",
            &["memory.kv_cache"],
            "HIPFIRE_KV_MODE"
        ));
        assert!(!row_matches(
            "Cache",
            &["memory.max_seq"],
            "HIPFIRE_KV_MODE"
        ));
    }

    #[test]
    fn developer_keys_match_by_key_and_env_name() {
        let key = "developer.qwen4_moe_sym_iu4";
        assert!(row_matches("Flash-Next experts", &[key], "sym_iu4"));
        assert!(row_matches(
            "Flash-Next experts",
            &[key],
            "hipfire_qwen4_moe"
        ));
        assert!(!row_matches("Flash-Next experts", &[key], "no-such-thing"));
    }
}
