#!/usr/bin/env python3
"""Flash-Next expert-parallel exactness comparator (stdlib only).

Compares two `fn_ep_oracle` output directories (crates/hipfire-loader/examples/
fn_ep_oracle.rs): the REFERENCE arm (one card, topology.ep = 1, or EP-R with
--ref-ranks R) and an EP arm
(topology.ep = N). Each directory holds meta.json, logits.jsonl, state.jsonl
(and optionally raw logits-p<prompt>-s<step>.f32 dumps).

    fn_ep_compare.py REF_DIR EP_DIR [--expect-ranks N] [--ref-ranks R] [--json OUT.json] [--max-report K]
    fn_ep_compare.py --self-test

The EP arm must equal the reference byte for byte (hashes of bytes):

  * identity: both runs completed; same max_seq / tokens / vocab / model
    sha256 (when both recorded it) / prompt files, token counts, token-id md5;
    the reference has --ref-ranks ranks (default 1) and the EP arm has --expect-ranks
    (default: the EP arm's own meta.json `ranks`, which must be >= 2 and, when
    --ref-ranks > 1, greater than --ref-ranks);
  * logits: for every (prompt, step) the same record set, the same greedy
    token and the same rank-0 logits sha256;
  * mirrors: for every EP record every peer rank's logits mirror sha256 equals
    rank 0's (one entry per rank);
  * state: for every (prompt, step) the EP arm has a digest for EVERY rank and
    every digest row (names and values) equals the reference rank 0's;
  * with --ref-ranks R > 1 the reference (itself EP, R ranks) is also checked
    against itself (families mirrors / state_ranks / state, messages prefixed
    "reference"): R mirror hashes equal to its own logits sha, and state
    digests for ranks 0..R-1 all equal to its rank 0.

Prints one line per failing check family (first mismatch in detail) and a
final one-line summary beginning `PASS` or `FAIL`; `--json` writes the full
verdict (counts, first mismatch, up to --max-report mismatches, per-family
results). Exit status: 0 only when every check passed; 1 on any mismatch,
missing or malformed data; 2 on usage errors.
"""
import argparse
import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile


# ---------------------------------------------------------------- helpers
def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as error:
                raise ValueError("%s:%d: %s" % (path, number, error))
    return out


def read_meta(arm_dir):
    path = os.path.join(arm_dir, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def first_diff_byte(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def raw_logits_report(ref_dir, ep_dir, prompt, step):
    """Max-abs / first differing byte of the raw logits at (prompt, step), if saved."""
    name = "logits-p%d-s%d.f32" % (prompt, step)
    path_a, path_b = os.path.join(ref_dir, name), os.path.join(ep_dir, name)
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return "raw logits not saved at prompt %d step %d in both arms" % (prompt, step)
    with open(path_a, "rb") as handle:
        a = handle.read()
    with open(path_b, "rb") as handle:
        b = handle.read()
    offset = first_diff_byte(a, b)
    if offset is None:
        return "raw logits identical (a sha mismatch is then a recording error)"
    count = min(len(a), len(b)) // 4
    xs = struct.unpack("<%df" % count, a[:count * 4])
    ys = struct.unpack("<%df" % count, b[:count * 4])
    max_abs, max_index = 0.0, -1
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = abs(x - y)
        if math.isnan(d):
            d = math.inf
        if d > max_abs:
            max_abs, max_index = d, i
    word = offset - (offset % 4)
    return "raw logits: max-abs=%g at element %d; first differing byte %d (word@%d %s vs %s); len %d vs %d" % (
        max_abs, max_index, offset, word, a[word:word + 4].hex(), b[word:word + 4].hex(), len(a), len(b))


class Verdict:
    """Collects mismatches per check family; the first one overall is `first`."""

    def __init__(self, max_report):
        self.max_report = max_report
        self.families = {}
        self.first = None
        self.mismatches = []

    def family(self, name):
        return self.families.setdefault(name, {"checked": 0, "failed": 0})

    def checked(self, name, n=1):
        self.family(name)["checked"] += n

    def fail(self, name, text, **detail):
        fam = self.family(name)
        fam["failed"] += 1
        entry = {"check": name, "message": text}
        entry.update(detail)
        if self.first is None:
            self.first = entry
        if len(self.mismatches) < self.max_report:
            self.mismatches.append(entry)

    @property
    def ok(self):
        return self.first is None


# --------------------------------------------------------------- identity
def check_identity(v, ref_meta, ep_meta, expect_ranks, ref_ranks=1):
    for label, meta in (("reference", ref_meta), ("EP", ep_meta)):
        v.checked("identity")
        if meta is None:
            v.fail("identity", "%s arm: meta.json missing" % label)
        elif not meta.get("completed"):
            v.fail("identity", "%s arm: meta.json says the run did not complete" % label)
    if ref_meta is None or ep_meta is None:
        return
    for key in ("max_seq", "tokens", "vocab"):
        v.checked("identity")
        if ref_meta.get(key) != ep_meta.get(key):
            v.fail("identity", "%s differs: reference %r vs EP %r" % (key, ref_meta.get(key), ep_meta.get(key)))
    sha_ref, sha_ep = ref_meta.get("model_sha256"), ep_meta.get("model_sha256")
    v.checked("identity")
    if sha_ref and sha_ep and sha_ref != sha_ep:
        v.fail("identity", "model sha256 differs: %s vs %s" % (sha_ref, sha_ep))

    def prompts(meta):
        return [(p.get("token_count"), p.get("file_md5"), p.get("ids_md5")) for p in meta.get("prompts", [])]

    v.checked("identity")
    if prompts(ref_meta) != prompts(ep_meta):
        v.fail("identity", "prompt identity (token_count, file_md5, ids_md5) differs: %r vs %r"
               % (prompts(ref_meta), prompts(ep_meta)))
    v.checked("identity")
    if ref_meta.get("ranks") != ref_ranks:
        v.fail("identity", "reference arm ran %r ranks, expected %d" % (ref_meta.get("ranks"), ref_ranks))
    ranks = ep_meta.get("ranks")
    v.checked("identity")
    if not isinstance(ranks, int) or ranks < 2:
        v.fail("identity", "EP arm ran %r ranks, expected >= 2" % (ranks,))
    elif ref_ranks > 1 and ranks <= ref_ranks:
        v.fail("identity", "EP arm ran %d ranks, expected more than the %d-rank reference" % (ranks, ref_ranks))
    elif expect_ranks is not None and ranks != expect_ranks:
        v.fail("identity", "EP arm ran %d ranks, expected %d" % (ranks, expect_ranks))


# ----------------------------------------------------------------- logits
def load_logits(arm_dir):
    out = {}
    for rec in read_jsonl(os.path.join(arm_dir, "logits.jsonl")):
        if rec.get("kind") != "logits":
            continue
        out[(rec["prompt"], rec["step"])] = rec
    return out


def check_logits(v, ref_dir, ep_dir, ref, ep, ranks):
    v.checked("record_set")
    if set(ref) != set(ep):
        only_ref = sorted(set(ref) - set(ep))[:3]
        only_ep = sorted(set(ep) - set(ref))[:3]
        v.fail("record_set", "logits record sets differ (only in reference %s, only in EP %s)" % (only_ref, only_ep))
    if not ref:
        v.fail("record_set", "reference logits.jsonl has no logits records")
    for key in sorted(set(ref) & set(ep)):
        prompt, step = key
        ra, rb = ref[key], ep[key]
        v.checked("logits")
        if ra["logits_sha256"] != rb["logits_sha256"] or ra.get("token") != rb.get("token"):
            report = "prompt=%d step=%d: reference token %s sha %s vs EP token %s sha %s" % (
                prompt, step, ra.get("token"), ra["logits_sha256"][:16], rb.get("token"), rb["logits_sha256"][:16])
            v.fail("logits", report, prompt=prompt, step=step, raw=raw_logits_report(ref_dir, ep_dir, prompt, step))
        v.checked("mirrors")
        mirrors = rb.get("rank_logits_sha256")
        if not mirrors or (ranks and len(mirrors) != ranks):
            v.fail("mirrors", "prompt=%d step=%d: EP record has %s logits mirror hashes, expected %s"
                   % (prompt, step, len(mirrors) if mirrors else 0, ranks), prompt=prompt, step=step)
        else:
            bad = [r for r, sha in enumerate(mirrors) if sha != rb["logits_sha256"]]
            if bad:
                v.fail("mirrors", "prompt=%d step=%d: rank(s) %s logits mirror != rank 0" % (prompt, step, bad),
                       prompt=prompt, step=step, ranks=bad)


# ------------------------------------------------------------------ state
def load_state(arm_dir):
    out = {}
    path = os.path.join(arm_dir, "state.jsonl")
    if not os.path.exists(path):
        return out
    for rec in read_jsonl(path):
        if rec.get("kind") != "state":
            continue
        out[(rec["prompt"], rec["step"], rec["rank"])] = [tuple(row) for row in rec["digest"]]
    return out


def diff_digest(ref_rows, ep_rows):
    notes = []
    if [n for n, _ in ref_rows] != [n for n, _ in ep_rows]:
        notes.append("row names differ (%d vs %d rows)" % (len(ref_rows), len(ep_rows)))
    ep_map = dict(ep_rows)
    for name, value in ref_rows:
        if name in ep_map and ep_map[name] != value:
            notes.append("%s: %s vs %s" % (name, value, ep_map[name]))
    return notes


def check_state(v, ref, ep, ranks):
    ref_keys = sorted((p, s) for (p, s, r) in ref if r == 0)
    v.checked("state_set")
    if not ref_keys:
        v.fail("state_set", "reference state.jsonl has no rank-0 state records")
        return
    ep_keys = sorted(set((p, s) for (p, s, r) in ep))
    if ep_keys != ref_keys:
        v.fail("state_set", "state (prompt, step) sets differ: reference %s vs EP %s" % (ref_keys[:6], ep_keys[:6]))
    for prompt, step in ref_keys:
        ref_rows = ref[(prompt, step, 0)]
        have = sorted(r for (p, s, r) in ep if p == prompt and s == step)
        v.checked("state_ranks")
        if ranks and have != list(range(ranks)):
            v.fail("state_ranks", "prompt=%d step=%d: EP state records for ranks %s, expected 0..%d"
                   % (prompt, step, have, ranks - 1), prompt=prompt, step=step)
        for rank in have:
            v.checked("state")
            notes = diff_digest(ref_rows, ep[(prompt, step, rank)])
            if notes:
                v.fail("state", "prompt=%d step=%d rank %d: %d digest row(s) differ from reference rank 0: %s"
                       % (prompt, step, rank, len(notes), "; ".join(notes[:4])),
                       prompt=prompt, step=step, rank=rank)


def check_reference(v, ref_logits, ref_state, ref_ranks):
    """An EP-R reference must itself be internally exact: R mirrors and R state digests equal to rank 0."""
    for (prompt, step), rec in sorted(ref_logits.items()):
        v.checked("mirrors")
        mirrors = rec.get("rank_logits_sha256")
        if not mirrors or len(mirrors) != ref_ranks:
            v.fail("mirrors", "reference prompt=%d step=%d: record has %s logits mirror hashes, expected %d"
                   % (prompt, step, len(mirrors) if mirrors else 0, ref_ranks), prompt=prompt, step=step)
        else:
            bad = [r for r, sha in enumerate(mirrors) if sha != rec["logits_sha256"]]
            if bad:
                v.fail("mirrors", "reference prompt=%d step=%d: rank(s) %s logits mirror != rank 0" % (prompt, step, bad),
                       prompt=prompt, step=step, ranks=bad)
    for prompt, step in sorted(set((p, s) for (p, s, _) in ref_state)):
        have = sorted(r for (p, s, r) in ref_state if p == prompt and s == step)
        v.checked("state_ranks")
        if have != list(range(ref_ranks)):
            v.fail("state_ranks", "reference prompt=%d step=%d: state records for ranks %s, expected 0..%d"
                   % (prompt, step, have, ref_ranks - 1), prompt=prompt, step=step)
        if 0 not in have:
            continue
        base = ref_state[(prompt, step, 0)]
        for rank in have:
            if rank == 0:
                continue
            v.checked("state")
            notes = diff_digest(base, ref_state[(prompt, step, rank)])
            if notes:
                v.fail("state", "reference prompt=%d step=%d rank %d: %d digest row(s) differ from reference rank 0: %s"
                       % (prompt, step, rank, len(notes), "; ".join(notes[:4])),
                       prompt=prompt, step=step, rank=rank)


# ------------------------------------------------------------------- main
def compare(ref_dir, ep_dir, expect_ranks, max_report, ref_ranks=1):
    v = Verdict(max_report)
    try:
        ref_meta, ep_meta = read_meta(ref_dir), read_meta(ep_dir)
    except ValueError as error:
        v.fail("identity", "meta.json unreadable: %s" % error)
        return v, {}
    check_identity(v, ref_meta, ep_meta, expect_ranks, ref_ranks)
    ranks = expect_ranks or (ep_meta or {}).get("ranks")
    try:
        ref_logits, ep_logits = load_logits(ref_dir), load_logits(ep_dir)
        ref_state, ep_state = load_state(ref_dir), load_state(ep_dir)
    except (OSError, ValueError, KeyError) as error:
        v.fail("record_set", "cannot read oracle output: %s" % error)
        return v, {}
    check_logits(v, ref_dir, ep_dir, ref_logits, ep_logits, ranks if isinstance(ranks, int) else None)
    check_state(v, ref_state, ep_state, ranks if isinstance(ranks, int) else None)
    if ref_ranks > 1:
        check_reference(v, ref_logits, ref_state, ref_ranks)
    prompts = sorted(set(p for (p, _) in ref_logits))
    steps = sorted(set(s for (_, s) in ref_logits))
    shape = {
        "ranks": ranks,
        "ref_ranks": ref_ranks,
        "prompts": len(prompts),
        "logits_records": len(ref_logits),
        "max_step": steps[-1] if steps else None,
        "state_records": len(ep_state),
    }
    return v, shape


def summary_line(v, shape, ref_dir, ep_dir):
    counts = {k: v.families[k]["checked"] for k in v.families}
    if v.ok:
        ref_ranks = shape.get("ref_ranks") or 1
        ref_name = "single" if ref_ranks == 1 else "EP%d reference" % ref_ranks
        return ("PASS fn_ep_compare: EP%s == %s, %s prompt(s) x steps 0..%s (%s logits records token+sha, "
                "%s mirror hashes, %s state digests) byte-identical"
                % (shape.get("ranks"), ref_name, shape.get("prompts"), shape.get("max_step"), counts.get("logits", 0),
                   counts.get("mirrors", 0), counts.get("state", 0)))
    failed = ", ".join("%s=%d" % (k, f["failed"]) for k, f in sorted(v.families.items()) if f["failed"])
    return "FAIL fn_ep_compare: %d failing check(s) [%s]; first: %s" % (
        sum(f["failed"] for f in v.families.values()), failed, v.first["message"])


def run(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], add_help=True)
    parser.add_argument("ref_dir", help="reference arm directory (single card, ep = 1; or ep = --ref-ranks)")
    parser.add_argument("ep_dir", help="EP arm directory (ep = N)")
    parser.add_argument("--expect-ranks", type=int, default=None, help="required EP rank count")
    parser.add_argument("--ref-ranks", type=int, default=1, help="reference arm rank count (1 = single card)")
    parser.add_argument("--json", default=None, help="write the verdict JSON here")
    parser.add_argument("--max-report", type=int, default=50, help="mismatches kept in the JSON")
    args = parser.parse_args(argv)
    if args.ref_ranks < 1:
        print("--ref-ranks must be >= 1", file=sys.stderr)
        return 2
    for d in (args.ref_dir, args.ep_dir):
        if not os.path.isdir(d):
            print("not a directory: %s" % d, file=sys.stderr)
            return 2
    v, shape = compare(args.ref_dir, args.ep_dir, args.expect_ranks, args.max_report, args.ref_ranks)
    for name, fam in sorted(v.families.items()):
        print("%s %-10s checked=%d failed=%d" % ("PASS" if not fam["failed"] else "FAIL", name, fam["checked"], fam["failed"]))
    if v.first is not None:
        print("first mismatch [%s]: %s" % (v.first["check"], v.first["message"]))
        if v.first.get("raw"):
            print("     " + v.first["raw"])
    line = summary_line(v, shape, args.ref_dir, args.ep_dir)
    print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "verdict": "PASS" if v.ok else "FAIL",
                "summary": line,
                "reference": os.path.abspath(args.ref_dir),
                "ep": os.path.abspath(args.ep_dir),
                "shape": shape,
                "families": v.families,
                "first_mismatch": v.first,
                "mismatches": v.mismatches,
            }, handle, indent=2)
            handle.write("\n")
    return 0 if v.ok else 1


# -------------------------------------------------------------- self-test
def _write(path, data, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as handle:
        handle.write(data)


def _arm(root, name, shas, tokens, ranks=1, mirrors=None, state=None, flip_state_rank=None, raw=None, completed=True):
    d = os.path.join(root, name)
    meta = {"completed": completed, "ranks": ranks, "max_seq": 512, "tokens": len(shas) - 1, "vocab": 4,
            "model_sha256": "m", "prompts": [{"prompt": 0, "token_count": 3, "file_md5": "f", "ids_md5": "i"}]}
    _write(os.path.join(d, "meta.json"), json.dumps(meta))
    lines = []
    for step, sha in enumerate(shas):
        rec = {"kind": "logits", "prompt": 0, "step": step, "token": None if step == 0 else tokens[step - 1],
               "position": step + 3, "logits_sha256": sha}
        if ranks > 1:
            rec["rank_logits_sha256"] = [sha] + [(mirrors or shas)[step]] * (ranks - 1)
        lines.append(json.dumps(rec))
    _write(os.path.join(d, "logits.jsonl"), "\n".join(lines) + "\n")
    if raw:
        for step, values in raw.items():
            _write(os.path.join(d, "logits-p0-s%d.f32" % step), struct.pack("<%df" % len(values), *values), "wb")
    lines = []
    for rank in range(ranks):
        digest = [["position", "9"], ["gdn[0].recurrent", state or "s0"]]
        if rank == flip_state_rank:
            digest[1][1] = "deadbeef"
        lines.append(json.dumps({"kind": "state", "prompt": 0, "step": 1, "rank": rank, "digest": digest}))
    _write(os.path.join(d, "state.jsonl"), "\n".join(lines) + "\n")
    return d


def _quiet(*argv):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = run(list(argv))
    return code, sink.getvalue()


def self_test():
    failures = []

    def expect(label, got, want):
        if got != want:
            failures.append("%s: got %r want %r" % (label, got, want))

    good = ["a", "b", "c"]
    with tempfile.TemporaryDirectory() as tmp:
        single = _arm(tmp, "single", good, [7, 8], raw={2: [1.0, 2.0]})
        ok = _arm(tmp, "ep_ok", good, [7, 8], ranks=3)
        out_json = os.path.join(tmp, "v.json")
        code, out = _quiet(single, ok, "--expect-ranks", "3", "--json", out_json)
        expect("equal", code, 0)
        if not out.strip().splitlines()[-1].startswith("PASS"):
            failures.append("equal run did not end with PASS: %s" % out)
        with open(out_json) as handle:
            expect("json verdict", json.load(handle)["verdict"], "PASS")

        expect("rank count", _quiet(single, ok, "--expect-ranks", "6")[0], 1)

        bad = _arm(tmp, "ep_bad", ["a", "b", "X"], [7, 8], ranks=2, raw={2: [1.0, 2.5]})
        code, out = _quiet(single, bad)
        expect("sha differs", code, 1)
        if "step=2" not in out or "max-abs=0.5" not in out or not out.strip().splitlines()[-1].startswith("FAIL"):
            failures.append("logits diff report wrong: %s" % out)

        expect("token differs", _quiet(single, _arm(tmp, "ep_tok", good, [7, 9], ranks=2))[0], 1)

        code, out = _quiet(single, _arm(tmp, "ep_mirror", good, [7, 8], ranks=3, mirrors=["a", "Y", "c"]))
        expect("mirror differs", code, 1)
        if "mirrors" not in out:
            failures.append("mirror disagreement not reported: %s" % out)

        code, out = _quiet(single, _arm(tmp, "ep_state", good, [7, 8], ranks=2, flip_state_rank=1))
        expect("state differs", code, 1)
        if "gdn[0].recurrent" not in out:
            failures.append("state diff row not named: %s" % out)

        expect("incomplete", _quiet(single, _arm(tmp, "ep_inc", good, [7, 8], ranks=2, completed=False))[0], 1)
        expect("missing dir", _quiet(single, os.path.join(tmp, "nope"))[0], 2)
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        expect("empty arm", _quiet(single, empty)[0], 1)

        # EP-R reference (--ref-ranks R > 1)
        ref2 = _arm(tmp, "ref_ep2", good, [7, 8], ranks=2)
        ep4 = _arm(tmp, "ep4", good, [7, 8], ranks=4)
        code, out = _quiet(ref2, ep4, "--ref-ranks", "2", "--expect-ranks", "4")
        expect("ref-ranks 2 vs EP4", code, 0)
        if "EP4 == EP2 reference" not in out.strip().splitlines()[-1]:
            failures.append("EP-R summary wording wrong: %s" % out)
        expect("ref-ranks default vs EP-R reference", _quiet(ref2, ep4)[0], 1)
        code, out = _quiet(ref2, ep4, "--ref-ranks", "3")
        expect("reference rank-count mismatch", code, 1)
        if "reference arm ran 2 ranks, expected 3" not in out:
            failures.append("reference rank-count mismatch not reported: %s" % out)
        code, out = _quiet(ref2, _arm(tmp, "ep2", good, [7, 8], ranks=2), "--ref-ranks", "2")
        expect("EP not larger than reference", code, 1)
        if "more than the 2-rank reference" not in out:
            failures.append("EP-not-larger not reported: %s" % out)
        code, out = _quiet(_arm(tmp, "ref_mirror", good, [7, 8], ranks=2, mirrors=["a", "Y", "c"]), ep4, "--ref-ranks", "2")
        expect("reference mirror differs", code, 1)
        if "reference prompt=0 step=1" not in out or "mirrors" not in out:
            failures.append("reference mirror disagreement not reported: %s" % out)
        code, out = _quiet(_arm(tmp, "ref_state", good, [7, 8], ranks=2, flip_state_rank=1), ep4, "--ref-ranks", "2")
        expect("reference rank-1 state differs", code, 1)
        if "reference prompt=0 step=1 rank 1" not in out or "gdn[0].recurrent" not in out:
            failures.append("reference rank-1 state diff not reported: %s" % out)

    for line in failures:
        print("SELF-TEST FAIL:", line)
    if failures:
        return 1
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    sys.exit(run(sys.argv[1:]))
