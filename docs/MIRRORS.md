<!-- SPDX-License-Identifier: Apache-2.0 -->
# PeaceMaker mirror and Redline overlay

[hipfire](https://github.com/warpfront/hipfire) remains canonical. After a push
to `beta`, the workflow updates two public repositories:

- [PeaceMaker](https://github.com/warpfront/peacemaker) `main` is a full,
  history-preserving crate split. Updating it uses a force-push.
- [Redline](https://github.com/warpfront/redline) `master` receives a normal
  commit on top of its existing history. Only hipfire-owned source paths and
  their minimal standalone dependency configuration are overlaid. No existing
  file is deleted, and master-only integrations and assets remain untouched.
  The push is never forced; a concurrent upstream change rejects the push.

Submit canonical crate changes to hipfire. No release tag or other branch is
pushed. Redline retains its existing default `master`; PeaceMaker uses `main`.

## Local previews

Run `scripts/mirror-split.sh <peacemaker|redline> [destination]` from a
committed checkout with full history, Python 3.11+, and
`git-filter-repo==2.47.0` on `PATH`. The default destination is
`mirror-preview/<name>`. Existing destinations are refused. The script does
not push or create GitHub repositories. Redline's mode reads its existing
public repository, so it also requires network access. Each output is a Git
repository with a standalone Cargo workspace; run `cargo check --workspace`
in it before publishing.

PeaceMaker includes `peacemaker-ir`, `peacemaker-lift`, `peacemaker-author`,
and `peacemaker-npu`, plus any transitive path dependency closure. Its split
preserves retained revisions, authors, dates, and ancestry. Historical
`pm-npu` paths map to `peacemaker-npu`. Mixed-purpose monorepo commit messages
are replaced by neutral source-commit references. A deterministic final
commit adds the generated workspace and canonical README. SPDX headers,
license files, and source fixtures are preserved. Unrelated hipfire root
Cargo.toml/README history is not copied into the split.

Redline overlays `crates/redline`, `crates/redline-rocr`, and
`crates/redline-dispatch`, preserving any existing file absent from hipfire.
The latter two need `hipfire-config`, copied under `vendor/hipfire-config`;
their copied manifests point there. Only missing workspace members and
inherited dependency entries are added. Existing members, excludes, package
metadata, other crates, examples, docs, workflows, and other assets stay in
place. The overlay commit records the hipfire source SHA. Legacy source and
artifact provenance, including existing filenames, is retained faithfully;
generated tooling and docs contain no private workstation provenance.

## One-time setup and release cut

The operator supplies `CUT_ROOT` (an external release scratch directory).
Run these only at the cut, after the NPU and EP merges and before the release
tag. Preparation does not create repositories, push, or merge anything.

```bash
# Redline already exists; only create PeaceMaker.
gh repo create warpfront/peacemaker --public \
  --description 'PeaceMaker crate mirror; canonical source: warpfront/hipfire'
mkdir -p "$CUT_ROOT/mirror-keys"
ssh-keygen -t ed25519 -N '' -C peacemaker-mirror \
  -f "$CUT_ROOT/mirror-keys/peacemaker"
ssh-keygen -t ed25519 -N '' -C redline-mirror \
  -f "$CUT_ROOT/mirror-keys/redline"
gh repo deploy-key add "$CUT_ROOT/mirror-keys/peacemaker.pub" \
  --repo warpfront/peacemaker --title hipfire-mirror --allow-write
gh repo deploy-key add "$CUT_ROOT/mirror-keys/redline.pub" \
  --repo warpfront/redline --title hipfire-mirror --allow-write
gh secret set PEACEMAKER_MIRROR_KEY --repo warpfront/hipfire \
  < "$CUT_ROOT/mirror-keys/peacemaker"
gh secret set REDLINE_MIRROR_KEY --repo warpfront/hipfire \
  < "$CUT_ROOT/mirror-keys/redline"

# Inside the dedicated chore/peacemaker-split worktree:
git fetch origin
git rebase origin/beta
# Resolve newly merged NPU imports/manifests to peacemaker_npu/peacemaker-npu;
# preserve the merged NPU implementation. Regenerate its maps if needed.
cargo check --workspace --locked
scripts/mirror-split.sh peacemaker "$CUT_ROOT/mirror-cut/peacemaker"
scripts/mirror-split.sh redline "$CUT_ROOT/mirror-cut/redline"
cargo check --workspace --manifest-path "$CUT_ROOT/mirror-cut/peacemaker/Cargo.toml"
cargo check --workspace --manifest-path "$CUT_ROOT/mirror-cut/redline/Cargo.toml"
```

Check that Redline's overlay deleted-file count is zero and every changed
path is a canonical-owned crate file, its minimal vendored dependency, or
required root Cargo configuration. Compare against the accepted public master
tree to confirm every original file remains. Run the external publication
guard on the hipfire delta, complete PeaceMaker split history, and Redline
**overlay delta**; retained public master history is not newly authored.

CPU-only merge gate specs are
`-p peacemaker-npu;-p npu-tools;-p hipfire-arch-qwen4;-p rdna-compute;-p peacemaker-ir;-p peacemaker-lift;-p peacemaker-author;-p redline;-p redline-rocr;-p redline-dispatch`.
The operator's gate rebases again, regenerates maps, runs documentation
ratchets and these tests, adds the recording-inventory test, and pushes beta.
Do not run it during preparation. The beta push triggers the first automatic
mirror/overlay publication using the two deploy-key secrets.

For a manual first push, regenerate from the **merged beta commit**, not the
pre-merge branch, check both standalone trees and publication guards, then:

```bash
git -C "$CUT_ROOT/mirror-cut/peacemaker" push --force \
  git@github.com:warpfront/peacemaker.git HEAD:refs/heads/main
git -C "$CUT_ROOT/mirror-cut/redline" push \
  git@github.com:warpfront/redline.git HEAD:refs/heads/master
gh repo edit warpfront/peacemaker --default-branch main
```

Also set PeaceMaker's default branch to `main` after its first automatic push.
Retry the workflow with
`gh workflow run mirror.yml --repo warpfront/hipfire --ref beta`. It checks both
standalone workspaces before using keys and skips superseded beta snapshots.
