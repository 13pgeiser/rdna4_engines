"""Build the overlay directory from the port worktree.

usage: build_overlay.py <port_worktree> <base_ref> <out_dir>
Writes:
  out/files/...          new files (radiance_*.py, .pth, plugin package, dist-info)
  out/base/vllm/...      <base_ref> versions of every modified vllm file
  out/patched/vllm/...   patched versions
  out/diffs/radiance-vllm-v0.32.0.diff   unified diff (vllm/ only), git-apply/patch -p1 compatible
  out/manifest.tsv       kind  path  radiance_marker_lines  sha256(patched)
"""
import hashlib
import subprocess
import sys
from pathlib import Path

wt, base, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])


def git(*a):
    return subprocess.run(["git", "-C", str(wt), *a], check=True, capture_output=True).stdout


def markers(text: str) -> int:
    return sum("radiance" in line.lower() for line in text.splitlines())


rows = []
for line in git("diff", "--name-status", base, "HEAD").decode().splitlines():
    st, path = line.split("\t", 1)
    data = (wt / path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if st == "A":
        dst = out / "files" / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        rows.append(("new", path, markers(data.decode(errors="replace")), sha))
    elif st == "M":
        for sub, blob in (("base", git("show", f"{base}:{path}")), ("patched", data)):
            dst = out / sub / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob)
        rows.append(("mod", path, markers(data.decode()), sha))
    else:
        raise SystemExit(f"unexpected status {st} {path}")

(out / "diffs").mkdir(parents=True, exist_ok=True)
(out / "diffs" / "radiance-vllm-v0.32.0.diff").write_bytes(git("diff", base, "HEAD", "--", "vllm"))
(out / "diffs" / "radiance-full-v0.32.0.diff").write_bytes(git("diff", "--binary", base, "HEAD"))
with open(out / "manifest.tsv", "w") as f:
    f.write("# kind\tpath\tradiance_marker_lines\tsha256\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(f"{sum(r[0] == 'new' for r in rows)} new, {sum(r[0] == 'mod' for r in rows)} modified -> {out}")
