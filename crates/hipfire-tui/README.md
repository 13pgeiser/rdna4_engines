# hipfire-tui

The hipfire terminal UI (`hipfire tui`, or bare `hipfire`).

Run from the repository root:

```bash
cargo run -p hipfire-tui
```

`hipfire-tui --settings` opens on the Settings tab (what `hipfire config` runs),
`--config-profile-wizard` opens the profile wizard (`hipfire config profile`),
and `--check` loads config, registry and local models without a terminal.

Tabs:

- **Home**: serve state, endpoint, active model, local/registry counts.
- **Dashboard**: live serve stats, GPU/VRAM (`amd-smi`, else `rocm-smi`, found
  through the ROCm-root lookup), live `/metrics`. `s` starts the serve; `x`
  (stop) and `R` (restart, which force-reaps this install's serve daemons) ask
  y/n first.
- **Chat**: streams through the serve's `/v1/chat/completions`. The transcript
  follows new output; Up/Down/PgUp/PgDn/Home/End scroll, also while a reply
  streams. When the serve is offline, Enter queues the prompt, starts
  `hipfire serve -d` if no serve process exists, and sends it once `/health` is
  online. `/help` lists the slash commands.
- **Models**: registry tags grouped by family plus local model files (sidecars
  such as PLE rows, drafts and hot sets are not listed). `Enter` on a local
  model makes it the default model; `p` pulls with `hipfire pull <tag> --yes`
  (a chooser offers PLE rows and draft sidecars when the entry has them), `c`
  cancels the pull, `d` deletes after a y/n.
- **Settings**: easy and advanced views over the typed global config
  (`~/.hipfire/config.toml`); every non-deprecated key is editable inline
  (editor kind comes from the schema's value rule), deprecated keys are
  read-only. Rows come from `SettingsModel` (`src/hipfire/settings_model.rs`):
  intent groups, the value predicted for the active model on this GPU, the
  layer it came from, and a lock on values an environment variable owns.
  Per-model overlays stay with `hipfire config <tag> set ...`.
- **System**: GPU, HIP and path diagnostics; `d` runs `hipfire diag`.
- **Logs**: the tail of `serve.log`.

The serve's `/health` is checked every 2 s on every tab. `?` shows the keys of
the current tab, Esc only closes or cancels, and `q` quits (asking first while a
pull is running).
