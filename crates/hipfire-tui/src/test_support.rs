// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Hermetic fixtures for the Settings tests: a temporary state root with its
//! own `config.toml`, a controlled registry card, an explicit GPU/model
//! context and an injected environment layer. Nothing reads the real
//! `~/.hipfire`, the process environment or the network.

use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Once,
    },
};

use hipfire_config::{
    effective::{A8Prefill, ArchCtx},
    intent::Intent,
    ConfigLayer, ConfigValue,
};
use hipfire_registry::ModelEntry;

use crate::{
    app::{App, Tab},
    hipfire::{registry::ModelRow, settings_model::RowId, HipfirePaths},
};

/// The model every fixture serves (a dense 27B, `model_arch` 5).
pub const ACTIVE_MODEL: &str = "qwen3.8:27b";

/// A temporary state root plus the [`App`] loaded from it. The root is removed
/// when the fixture drops.
pub struct Fixture {
    pub app: App,
    root: PathBuf,
}

impl Fixture {
    pub fn config_path(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn config_bytes(&self) -> Vec<u8> {
        std::fs::read(self.config_path()).unwrap_or_default()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Builder for a Settings fixture.
pub struct FixtureBuilder {
    gpu: String,
    a8: A8Prefill,
    config: String,
    env: ConfigLayer,
    card_kv_mode: Option<String>,
}

impl FixtureBuilder {
    pub fn new(gpu: &str) -> Self {
        Self {
            gpu: gpu.to_owned(),
            a8: A8Prefill::NotLanded,
            config: String::new(),
            env: ConfigLayer::default(),
            card_kv_mode: None,
        }
    }

    /// The PR #902 state to inject (a gfx1151 fixture mirroring mockup S1
    /// injects [`A8Prefill::OptIn`]).
    pub fn a8(mut self, a8: A8Prefill) -> Self {
        self.a8 = a8;
        self
    }

    /// TOML tables appended to `config.toml` (no `[serve]` table).
    pub fn config(mut self, toml: &str) -> Self {
        self.config.push_str(toml);
        self
    }

    /// One key of the injected `HIPFIRE_*` environment layer.
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env
            .set(key, ConfigValue::String(value.to_owned()))
            .expect("fixture env key is valid");
        self
    }

    /// The active model's registry card sets `default_kv_mode`.
    pub fn card_kv_mode(mut self, mode: &str) -> Self {
        self.card_kv_mode = Some(mode.to_owned());
        self
    }

    pub fn build(self) -> Fixture {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        static NO_FETCH: Once = Once::new();
        // The registry loader would otherwise try the network; the bundled
        // registry is replaced below anyway.
        NO_FETCH.call_once(|| std::env::set_var("HIPFIRE_NO_REGISTRY_FETCH", "1"));

        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-settings-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create fixture home");
        std::fs::write(
            root.join("config.toml"),
            format!(
                "schema_version = 1\n{}\n[serve]\ndefault_model = \"{ACTIVE_MODEL}\"\n",
                self.config
            ),
        )
        .expect("write fixture config");

        let arch = ArchCtx {
            gpu_arch: Some(self.gpu),
            gpu_count: 1,
            model_arch_id: Some(5),
            registry_tag: Some(ACTIVE_MODEL.to_owned()),
            a8_prefill: self.a8,
        };
        let mut app = App::load_from_paths(HipfirePaths::under(root.clone()), arch, self.env)
            .expect("load fixture app");
        app.registry.models = vec![ModelRow {
            tag: ACTIVE_MODEL.to_owned(),
            entry: ModelEntry {
                repo: "hipfire-models/qwen3.8-27b".into(),
                file: "qwen3.8-27b.mq4".into(),
                size_gb: 15.66,
                min_vram_gb: 17.0,
                desc: "Qwen3.8 27B".into(),
                arch_id: Some(5),
                default_kv_mode: self.card_kv_mode,
                ..ModelEntry::default()
            },
            downloaded: false,
            has_mtp: false,
        }];
        app.rebuild_settings();
        app.tab = Tab::Settings;
        // Serve offline, not "checking": the health poller never ran.
        app.status.health_known = true;
        Fixture { app, root }
    }
}

/// Send one key to the Settings handler.
pub fn press(app: &mut App, code: crossterm::event::KeyCode) {
    app.handle_settings_key(crossterm::event::KeyEvent::new(
        code,
        crossterm::event::KeyModifiers::NONE,
    ));
}

/// Type `text` into the Settings handler, one key per char.
pub fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        press(app, crossterm::event::KeyCode::Char(c));
    }
}

/// Put the cursor on the row `id` through the real keys: jump to its group,
/// then step down to it.
pub fn select_row(app: &mut App, id: &RowId) {
    use crossterm::event::KeyCode;
    let group = app.settings.row(id).expect("row exists").group;
    let index = Intent::ALL
        .iter()
        .position(|candidate| *candidate == group)
        .expect("group listed");
    press(app, KeyCode::Char(char::from(b'1' + index as u8)));
    let at = app
        .settings_visible()
        .iter()
        .position(|row| &row.id == id)
        .expect("row visible in its group");
    for _ in 0..at {
        press(app, KeyCode::Down);
    }
}

/// [`select_row`] for a canonical key row.
pub fn select_key(app: &mut App, key: &str) {
    select_row(app, &RowId::Key(key.to_owned()));
}

/// A Dashboard snapshot for a serve that is online with `model` resident and
/// `uptime_s` seconds up. `loaded` is its `/stats` `loaded_config`.
pub fn online_dashboard(
    loaded: Option<crate::hipfire::dashboard::LoadedConfig>,
    uptime_s: u64,
) -> crate::hipfire::dashboard::Dashboard {
    use crate::hipfire::dashboard::{Dashboard, LoadState, ServeStats, VramState};
    Dashboard {
        serve_up: true,
        endpoint: "127.0.0.1:11435".into(),
        health_text: String::new(),
        model: loaded.as_ref().map(|loaded| loaded.model.clone()),
        stats: Some(ServeStats {
            model: loaded.as_ref().map(|loaded| loaded.model.clone()),
            uptime_s,
            queue_depth: 0,
            requests_served: 0,
            recent_tok_s: None,
            loaded_config: loaded,
        }),
        model_ids: Vec::new(),
        vram: VramState::Unavailable("test".into()),
        gpu_load: LoadState::Unavailable("test".into()),
        offline_hint: None,
        system: None,
    }
}
