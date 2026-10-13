#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Sync a hipfire sub-project to its standalone public repository.
#
#   scripts/mirror-split.sh <peacemaker|redline> [destination]
#
#   peacemaker  history-preserving filter of the committed source tree, branch `main`
#   redline     one overlay commit on a clone of the existing public master
#
# Reads the committed tree and history of the source checkout (HEAD, or
# MIRROR_SOURCE_REV) and never touches it. The destination (default
# mirror-preview/<name> under the source checkout) must not exist. It never
# pushes; pushing is the workflow's job.
#
# Environment: MIRROR_SOURCE_REV, MIRROR_REDLINE_REMOTE (redline only),
# MIRROR_FILTER_REPO (path of git-filter-repo if it is not on PATH).
# Requires: git, python3 >= 3.11, git-filter-repo 2.47.0 (peacemaker), cargo
# (settles the generated/overlaid Cargo.lock).
# See scripts/mirror_split.py for the exact rules.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$here/mirror_split.py" "$@"
