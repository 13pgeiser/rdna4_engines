#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""CPU tests for scripts/agentic_gate_driver.py (the agentic-gate daemon driver).

A fake daemon enforcing the real wire rules (scripts/tests/fake_agentic_daemon.py)
stands in for target/release/daemon. The Rust side of the proof is
`agentic_gate_driver_requests_pass_daemon_wire_validation` in
crates/hipfire-daemon/src/main.rs, which replays the same fixture through the
daemon's real validation.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import agentic_gate_driver as driver  # noqa: E402

DRIVER = SCRIPTS / "agentic_gate_driver.py"
FAKE = SCRIPTS / "tests" / "fake_agentic_daemon.py"
FIXTURE = SCRIPTS / "tests" / "fixtures" / "agentic-gate-requests.jsonl"
GATE = SCRIPTS / "agentic-gate.sh"

FIXTURE_CELLS = [
    "m.mq4|s1|1|1|clamp-multi",
    "m.mq4|s2|0|0|think-on",
    "m.mq4|s3|0|1|on-multi",
]
FIXTURE_SYSTEMS = ["SYS-1", "SYS-2", "SYS-3"]


def fixture_text():
    msgs = driver.build_session(
        "/models/m.mq4", FIXTURE_CELLS, "USER PROMPT", FIXTURE_SYSTEMS
    )
    return "".join(json.dumps(m) + "\n" for m in msgs)


def test_driver_requests_match_checked_in_fixture():
    if os.environ.get("UPDATE_FIXTURE") == "1":
        FIXTURE.write_text(fixture_text())
    assert FIXTURE.read_text() == fixture_text()


def test_request_shapes():
    msgs = driver.build_requests(FIXTURE_CELLS, "U", FIXTURE_SYSTEMS)
    assert [m["id"] for m in msgs] == ["c0_t1", "c0_t2", "c1_t1", "c2_t1", "c2_t2"]
    assert [m["attempt_id"] for m in msgs] == [1, 2, 3, 4, 5]
    for m in msgs:
        assert isinstance(m["attempt_id"], int) and m["attempt_id"] > 0
        assert "thinking" not in m and "max_think_tokens" not in m
        assert m["thinking_enabled"] is (m["assistant_prefix"] == "open_think")
    # clamp cell and every second turn think off; the other first turns think on
    assert [m["thinking_enabled"] for m in msgs] == [False, False, True, True, False]


def _cells(tmp_path):
    cells = []
    for i, (spec, text) in enumerate(zip(FIXTURE_CELLS, FIXTURE_SYSTEMS)):
        sys_file = tmp_path / f"sys{i}.txt"
        sys_file.write_text(text)
        parts = spec.split("|")
        parts[1] = str(sys_file)
        cells.append("|".join(parts))
    user = tmp_path / "user.txt"
    user.write_text("USER PROMPT")
    return cells, user


def _run_driver(tmp_path, exe, extra=(), env=None, timeout=60):
    cells, user = _cells(tmp_path)
    out = tmp_path / "out.txt"
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--exe", str(exe), "--model", "/models/m.mq4",
         "--user-file", str(user), "--out", str(out), *extra, *cells],
        capture_output=True, text=True, env=full_env, timeout=timeout,
    )
    return proc, out, cells


def _events(path):
    evs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            evs.append(json.loads(line))
    return evs


def _gate_parser():
    """The detector/report heredoc from agentic-gate.sh, verbatim."""
    lines = GATE.read_text().splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.strip().startswith('python3 - "$OUTPUT_FILE" "$OUT"') and "<<'PY'" in l
    )
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "PY")
    return "\n".join(lines[start + 1:end]) + "\n"


def test_driver_end_to_end_against_fake_daemon(tmp_path):
    proc, out, cells = _run_driver(
        tmp_path, FAKE, env={"FAKE_COMMIT_TIMEOUT": "2", "FAKE_READY_DELAY": "0.2"}
    )
    assert proc.returncode == 0, proc.stderr
    evs = _events(out)
    assert not [e for e in evs if e["type"] == "error"], evs
    assert [e["type"] for e in evs if e["type"] in ("loaded", "unloaded")] == [
        "loaded", "unloaded"
    ]
    done = [(e["id"], e["attempt_id"]) for e in evs if e["type"] == "done"]
    assert done == [("c0_t1", 1), ("c0_t2", 2), ("c1_t1", 3), ("c2_t1", 4), ("c2_t2", 5)]
    # Stderr of the daemon is captured into the same file, as `2>&1` was.
    assert "fake-daemon: started" in out.read_text()

    # The gate's own parser scores this output without a hard fail.
    report = tmp_path / "report.md"
    report.write_text("")
    parsed = subprocess.run(
        [sys.executable, "-", str(out), str(report), *cells],
        input=_gate_parser(), capture_output=True, text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    summary = json.loads(parsed.stderr.strip().splitlines()[-1])
    assert summary == {"hard_fail": False, "soft_warn_count": 0}
    text = report.read_text()
    assert text.count("HARD_FAIL") == 0
    assert "zero tokens" not in text
    assert "json ok: name='read_file'" in text


def _fake_session(tmp_path, env):
    full_env = dict(os.environ)
    full_env.update(env)
    return subprocess.Popen(
        [sys.executable, str(FAKE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, env=full_env,
    )


def _send(p, obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()


def _read_until(p, pred, limit=20):
    evs = []
    for _ in range(limit):
        line = p.stdout.readline()
        if not line:
            break
        ev = json.loads(line)
        evs.append(ev)
        if pred(ev):
            break
    return evs


@pytest.mark.parametrize("bad", [None, 0, "7", True, 1.5])
def test_fake_rejects_generate_without_valid_attempt_id(tmp_path, bad):
    p = _fake_session(tmp_path, {})
    try:
        req = {"type": "generate", "id": "x", "prompt": "p"}
        if bad is not None:
            req["attempt_id"] = bad
        _send(p, req)
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert [e["type"] for e in evs] == ["error"]
        assert evs[0]["id"] == "x"
    finally:
        p.kill()
        p.wait()


def test_fake_ignores_early_commit_then_aborts(tmp_path):
    # commit_ready is delayed; the commit sent right after generate is early.
    p = _fake_session(tmp_path, {"FAKE_READY_DELAY": "0.5", "FAKE_COMMIT_TIMEOUT": "1"})
    try:
        _send(p, {"type": "generate", "id": "x", "attempt_id": 1, "prompt": "p"})
        _send(p, {"type": "commit", "id": "x", "attempt_id": 1})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        types = [e["type"] for e in evs]
        assert "commit_ready" in types
        assert types[-1] == "error" and "done" not in types
    finally:
        p.kill()
        p.wait()


def test_fake_aborts_without_commit_and_done_needs_matching_commit(tmp_path):
    p = _fake_session(tmp_path, {"FAKE_COMMIT_TIMEOUT": "1"})
    try:
        _send(p, {"type": "generate", "id": "x", "attempt_id": 1, "prompt": "p"})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "error"
        # wrong attempt_id is not a matching commit
        _send(p, {"type": "generate", "id": "y", "attempt_id": 2, "prompt": "p"})
        _read_until(p, lambda e: e["type"] == "commit_ready")
        _send(p, {"type": "commit", "id": "y", "attempt_id": 99})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "error"
        # a matching commit completes
        _send(p, {"type": "generate", "id": "z", "attempt_id": 3, "prompt": "p"})
        _read_until(p, lambda e: e["type"] == "commit_ready")
        _send(p, {"type": "commit", "id": "z", "attempt_id": 3})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert evs[-1]["type"] == "done" and evs[-1]["attempt_id"] == 3
    finally:
        p.kill()
        p.wait()


def test_pre_protocol_requests_fail_against_fake(tmp_path):
    """The old gate's request shape (no attempt_id) would get an error, not tokens."""
    p = _fake_session(tmp_path, {})
    try:
        _send(p, {"type": "generate", "id": "c0_t1", "prompt": "p", "thinking": False,
                  "max_think_tokens": 1})
        evs = _read_until(p, lambda e: e["type"] in ("error", "done"))
        assert [e["type"] for e in evs] == ["error"]
    finally:
        p.kill()
        p.wait()


def test_driver_passes_the_callers_full_environment_to_the_daemon(tmp_path):
    """A fabric hold's private lock dir and admission vars must reach the daemon."""
    sentinel = {
        "HIPFIRE_LOCK_DIR": str(tmp_path / "fabric-locks"),
        "FABRIC_JOB": "job-123",
        "ROCR_VISIBLE_DEVICES": "1",
        "HIPFIRE_DEVICES": "1",
        "AGENTIC_SENTINEL": "sentinel-value",
        "HIPFIRE_KV_MODE": "asym3",  # the gate's own `env` override
        "HIPFIRE_GRAPH": "1",
    }
    proc, out, _ = _run_driver(
        tmp_path, FAKE,
        env={**sentinel, "FAKE_ECHO_ENV": ",".join(sentinel)},
    )
    assert proc.returncode == 0, proc.stderr
    loaded = next(e for e in _events(out) if e["type"] == "loaded")
    assert loaded["env"] == sentinel


def test_gate_launches_driver_with_its_env_overrides_only():
    """The gate adds HIPFIRE_KV_MODE/HIPFIRE_GRAPH via `env`; no clean env anywhere."""
    text = GATE.read_text()
    assert re.search(
        r"env HIPFIRE_KV_MODE=asym3 \\\n\s+HIPFIRE_GRAPH=1 \\\n"
        r"\s+python3 scripts/agentic_gate_driver\.py",
        text,
    )
    assert "env -i" not in text
    assert "env -i" not in DRIVER.read_text()


def test_driver_deadline_kills_only_its_hung_daemon(tmp_path):
    pidfile = tmp_path / "daemon.pid"
    t0 = time.monotonic()
    proc, out, cells = _run_driver(
        tmp_path, FAKE,
        extra=["--generate-timeout", "1", "--unload-timeout", "1"],
        env={"FAKE_HANG": "1", "FAKE_PIDFILE": str(pidfile)},
    )
    assert time.monotonic() - t0 < 30
    assert proc.returncode == 1
    evs = _events(out)
    errs = [e for e in evs if e["type"] == "error"]
    assert errs and errs[0]["source"] == "agentic_gate_driver"
    assert errs[0]["id"] == "c0_t1" and "deadline" in errs[0]["message"]
    assert not [e for e in evs if e["type"] == "done"]
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the driver's own daemon is gone


def test_driver_records_load_deadline_and_spawn_failure(tmp_path):
    proc, out, _ = _run_driver(tmp_path, tmp_path / "does-not-exist")
    assert proc.returncode == 1
    errs = [e for e in _events(out) if e["type"] == "error"]
    assert errs and "spawn failed" in errs[0]["message"]

    silent = tmp_path / "silent.sh"
    silent.write_text("#!/bin/sh\nexec sleep 30\n")
    silent.chmod(0o755)
    proc, out, _ = _run_driver(tmp_path, silent, extra=["--load-timeout", "1"])
    assert proc.returncode == 1
    errs = [e for e in _events(out) if e["type"] == "error"]
    assert errs and "load deadline" in errs[0]["message"]
