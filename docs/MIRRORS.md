<!-- SPDX-License-Identifier: Apache-2.0 -->
# Canonical PeaceMaker mirror

[hipfire](https://github.com/warpfront/hipfire) remains canonical. The public
[PeaceMaker](https://github.com/warpfront/peacemaker) repository receives a
generated `main` branch after pushes to hipfire's `beta`. Submit changes to
hipfire; do not edit the generated branch. No release tag or other branch is
pushed.

Run `scripts/mirror-split.sh peacemaker [destination]` from a committed
checkout with full history, Python 3.11+, and `git-filter-repo==2.47.0` on
`PATH`. The default destination is `mirror-preview/peacemaker`. Existing
destinations are refused. The script never pushes or creates repositories.
The output is a Git repository with a standalone Cargo workspace; check it
with `cargo check --workspace` in that directory.

The mirror contains `peacemaker-ir`, `peacemaker-lift`, `peacemaker-author`,
and `peacemaker-npu`, plus their minimal transitive path dependency closure.
There are currently no Cargo path dependencies outside these four crates.
Tests also read `crates/rocm.rs` and eight `kernels/*.hxaco` fixtures, which
are retained. Source SPDX headers and LICENSE, LICENSE-APACHE, LICENSE-MIT
and NOTICE are preserved.

The split preserves retained revisions, authors, dates, and ancestry.
Historical `pm-npu` paths map to `peacemaker-npu`. Mixed-purpose monorepo
commit messages are replaced with neutral canonical source-commit references.
A deterministic final commit adds the generated workspace, pruned lockfile,
and README pointing to hipfire. Unrelated hipfire root Cargo.toml/README
history is not copied. Existing source and artifact provenance is copied
faithfully; generated tooling and docs contain no private workstation
provenance.

## One-time setup and release cut

The operator supplies `CUT_ROOT` (an external release scratch directory).
Run these only at the cut, after the NPU and EP merges and before the release
tag. Preparation does not create repositories, push, or merge anything.

The public repository and its default `main` branch already exist. Do not
create the repository again. Configure the deploy key and secret once if
they are not already provisioned.

```bash
mkdir -p "$CUT_ROOT/mirror-keys"
ssh-keygen -t ed25519 -N '' -C peacemaker-mirror \
  -f "$CUT_ROOT/mirror-keys/peacemaker"
gh repo deploy-key add "$CUT_ROOT/mirror-keys/peacemaker.pub" \
  --repo warpfront/peacemaker --title hipfire-mirror --allow-write
gh secret set PEACEMAKER_MIRROR_KEY --repo warpfront/hipfire \
  < "$CUT_ROOT/mirror-keys/peacemaker"

# Inside the dedicated chore/peacemaker-split worktree:
git fetch origin
git rebase origin/beta
# Resolve newly merged NPU imports/manifests to peacemaker_npu/peacemaker-npu;
# preserve the merged NPU implementation. Regenerate its maps if needed.
cargo check --workspace --locked
scripts/mirror-split.sh peacemaker "$CUT_ROOT/mirror-cut/peacemaker"
cargo check --workspace --manifest-path "$CUT_ROOT/mirror-cut/peacemaker/Cargo.toml"
```

Run the external publication guard over both the hipfire delta and every
commit delta in the complete filtered history, including generated files.
CPU-only merge gate specs are
`-p peacemaker-npu;-p npu-tools;-p hipfire-arch-qwen4;-p rdna-compute;-p peacemaker-ir;-p peacemaker-lift;-p peacemaker-author`.
The operator's merge gate rebases again, regenerates maps, runs documentation
ratchets and these tests, adds the recording-inventory test, and pushes beta.
Do not run it during preparation. The beta push triggers the first automatic
PeaceMaker publication using `PEACEMAKER_MIRROR_KEY`.

For a manual first push, regenerate from the **merged beta commit**, not the
pre-merge branch, check the standalone tree and full-history publication
guard, then:

```bash
git -C "$CUT_ROOT/mirror-cut/peacemaker" push --force \
  git@github.com:warpfront/peacemaker.git HEAD:refs/heads/main
gh repo edit warpfront/peacemaker --default-branch main
```

Also set PeaceMaker's default branch to `main` after its first automatic push.
Retry the workflow with
`gh workflow run mirror.yml --repo warpfront/hipfire --ref beta`. It checks the
standalone workspace before using its deploy key and skips superseded snapshots.
