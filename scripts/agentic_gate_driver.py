#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Event-driven daemon session driver for scripts/agentic-gate.sh.

Daemon wire protocol (crates/hipfire-daemon/src/main.rs `require_wire_attempt_id`;
crates/hipfire-engine/src/terminal.rs commit handshake):

  -> {"type":"load","model":M,"params":{...}}          <- {"type":"loaded",...}
  -> {"type":"generate","id":ID,"attempt_id":N,...}    (N numeric, nonzero)
  <- {"type":"commit_ready","id":ID,"attempt_id":N,...}
  -> {"type":"commit","id":ID,"attempt_id":N}
  <- {"type":"done","id":ID,...}
  -> {"type":"unload"}                                 <- {"type":"unloaded"}

A generate without a nonzero numeric attempt_id is refused. A commit sent before
commit_ready is ignored, and a commit_ready that no matching commit answers within
30 s (CLIENT_TERMINAL_COMMIT_TIMEOUT) aborts the request. The driver therefore
sends each request only after the previous one terminated, and answers every
matching commit_ready immediately.

The daemon's stdout and stderr lines are written, whole-line, to the output file
so the gate's parser (which scans raw text for JSON event lines and for the
"already reserved by holder PID" stderr message) works on it unchanged. Driver
failures (deadline, daemon exit, spawn failure) are appended as synthetic
{"type":"error","source":"agentic_gate_driver",...} lines.

Only the daemon this driver spawned is ever signalled: its own process group
(start_new_session), never a pattern match.

Usage:
  agentic_gate_driver.py --exe EXE --model MODEL --user-file FILE --out FILE CELL...
CELL is the gate's "model|system_file|clamp|multi|label" string.
"""
import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

LOAD_PARAMS = {"max_seq": 4096}
LOAD_TIMEOUT_S = 180.0
GENERATE_TIMEOUT_S = 120.0
UNLOAD_TIMEOUT_S = 30.0
EXIT_GRACE_S = 10.0
TERM_GRACE_S = 5.0

TOOL_RESPONSE_PROMPT = (
    "<tool_response>\n"
    "{\"contents\": \"int main() { return 0; }\"}\n"
    "</tool_response>\n"
    "Now describe what the file does in one sentence."
)


def _generate_request(rid, attempt_id, prompt, system, max_tokens, thinking):
    """One generate request shaped like the serve path's Qwen Jinja normalization.

    hipfire-cli `apply_http_reasoning_request` (crates/hipfire-cli/src/main.rs
    ~5708-5720) always sends `thinking_enabled` plus `assistant_prefix`
    (`open_think` when thinking is on, `closed_think` when off) and sends
    `max_think_tokens` only for an explicit positive cap (main.rs ~5883-5884),
    so a thinking-off turn carries no `max_think_tokens`.
    """
    return {
        "type": "generate",
        "id": rid,
        "attempt_id": attempt_id,
        "prompt": prompt,
        "system": system,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "thinking_enabled": bool(thinking),
        "assistant_prefix": "open_think" if thinking else "closed_think",
    }


def build_requests(cells, user_prompt, systems):
    """Return the ordered generate request dicts for a model's cells.

    cells:   "model|system_file|clamp|multi|label" strings (only clamp/multi used)
    systems: system prompt text per cell, same order as cells
    Ids are c{idx}_t1 (and c{idx}_t2 for multi-turn); attempt_ids are 1, 2, ...
    in send order.
    """
    requests = []
    attempt = 0
    for idx, (cell, system) in enumerate(zip(cells, systems)):
        _model, _sys_path, clamp, multi, _label = cell.split("|")
        attempt += 1
        requests.append(
            _generate_request(
                f"c{idx}_t1", attempt, user_prompt, system, 256, clamp != "1"
            )
        )
        if multi == "1":
            attempt += 1
            requests.append(
                _generate_request(
                    f"c{idx}_t2", attempt, TOOL_RESPONSE_PROMPT, "", 128, False
                )
            )
    return requests


def build_load(model_path):
    return {"type": "load", "model": model_path, "params": dict(LOAD_PARAMS)}


def build_session(model_path, cells, user_prompt, systems):
    """The full message sequence: load, every generate, unload."""
    return (
        [build_load(model_path)]
        + build_requests(cells, user_prompt, systems)
        + [{"type": "unload"}]
    )


class Driver:
    def __init__(self, exe, out_path, load_timeout=LOAD_TIMEOUT_S,
                 generate_timeout=GENERATE_TIMEOUT_S, unload_timeout=UNLOAD_TIMEOUT_S,
                 env=None):
        self.exe = exe
        self.out_path = out_path
        self.load_timeout = load_timeout
        self.generate_timeout = generate_timeout
        self.unload_timeout = unload_timeout
        self.env = env
        self.proc = None
        self.errors = []
        self._events = queue.Queue()
        self._out_lock = threading.Lock()
        self._out = None
        self._threads = []

    # -- output ------------------------------------------------------------
    def _write_line(self, text):
        if not text.endswith("\n"):
            text += "\n"
        with self._out_lock:
            self._out.write(text)
            self._out.flush()

    def _record_error(self, message, rid=None):
        self.errors.append(message)
        ev = {"type": "error", "source": "agentic_gate_driver", "message": message}
        if rid is not None:
            ev["id"] = rid
        self._write_line(json.dumps(ev))

    # -- process -----------------------------------------------------------
    def _pump(self, stream, forward):
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", "replace")
            self._write_line(text)
            if forward:
                self._events.put(text)
        if forward:
            self._events.put(None)

    def start(self):
        self._out = open(self.out_path, "w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                [self.exe],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # None inherits the caller's FULL environment (the gate adds only
                # its own overrides via `env`); fabric holds rely on their lock dir
                # and admission vars reaching the daemon. Never build a clean env.
                env=self.env,
                start_new_session=True,
                bufsize=0,
            )
        except OSError as e:
            self._record_error(f"daemon spawn failed: {e}")
            return False
        for stream, forward in ((self.proc.stdout, True), (self.proc.stderr, False)):
            t = threading.Thread(target=self._pump, args=(stream, forward), daemon=True)
            t.start()
            self._threads.append(t)
        return True

    def send(self, obj):
        try:
            self.proc.stdin.write(
                (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
            )
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def _close_stdin(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def kill(self):
        """TERM then KILL the daemon's own process group (pgid == its pid)."""
        if self.proc is None:
            return
        for sig, grace in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, EXIT_GRACE_S)):
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue

    # -- events ------------------------------------------------------------
    def next_event(self, deadline):
        """Next parsed JSON event; "timeout"/"eof" sentinels on those conditions."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout"
            try:
                line = self._events.get(timeout=remaining)
            except queue.Empty:
                return "timeout"
            if line is None:
                return "eof"
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                ev = json.loads(s)
            except ValueError:
                continue
            if isinstance(ev, dict):
                return ev

    def _await_loaded(self):
        deadline = time.monotonic() + self.load_timeout
        while True:
            ev = self.next_event(deadline)
            if ev == "timeout":
                self._record_error(f"load deadline {self.load_timeout:g}s exceeded")
                return False
            if ev == "eof":
                self._record_error("daemon exited before loaded")
                return False
            if ev.get("type") == "loaded":
                return True
            if ev.get("type") == "error":
                self.errors.append(f"load error: {ev.get('message', '')}")
                return False

    def _run_request(self, req):
        """Send one generate, answer its commit_ready, wait for its terminal.

        Returns True when the daemon is still usable for the next request.
        """
        rid, attempt = req["id"], req["attempt_id"]
        if not self.send(req):
            self._record_error("daemon stdin closed before generate", rid)
            return False
        deadline = time.monotonic() + self.generate_timeout
        while True:
            ev = self.next_event(deadline)
            if ev == "timeout":
                self._record_error(
                    f"generate deadline {self.generate_timeout:g}s exceeded", rid
                )
                return False
            if ev == "eof":
                self._record_error("daemon exited during generate", rid)
                return False
            typ, eid = ev.get("type"), ev.get("id")
            if typ == "commit_ready" and eid == rid and ev.get("attempt_id") == attempt:
                if not self.send({"type": "commit", "id": rid, "attempt_id": attempt}):
                    self._record_error("daemon stdin closed before commit", rid)
                    return False
            elif typ == "done" and eid == rid:
                return True
            elif typ == "error" and eid in (rid, None):
                # The daemon reported it; the parser sees the raw event line.
                self.errors.append(f"{rid}: {ev.get('message', '')}")
                return True

    def _unload(self):
        if not self.send({"type": "unload"}):
            return
        deadline = time.monotonic() + self.unload_timeout
        while True:
            ev = self.next_event(deadline)
            if ev == "timeout":
                self._record_error(f"unload deadline {self.unload_timeout:g}s exceeded")
                return
            if ev == "eof":
                return
            if ev.get("type") == "unloaded":
                return

    def run(self, model_path, requests):
        """Drive one daemon session; returns 0 clean, 1 if any error was recorded."""
        if not self.start():
            self._out.close()
            return 1
        try:
            if not self.send(build_load(model_path)):
                self._record_error("daemon stdin closed before load")
            elif self._await_loaded():
                for req in requests:
                    if not self._run_request(req):
                        break
                self._unload()
            self._close_stdin()
            try:
                # A session that already failed gets no long courtesy wait.
                self.proc.wait(timeout=2.0 if self.errors else EXIT_GRACE_S)
            except subprocess.TimeoutExpired:
                self.kill()
        finally:
            if self.proc.poll() is None:
                self.kill()
            for t in self._threads:
                t.join(timeout=5)
            self._out.close()
        return 1 if self.errors else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--user-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-timeout", type=float, default=LOAD_TIMEOUT_S)
    ap.add_argument("--generate-timeout", type=float, default=GENERATE_TIMEOUT_S)
    ap.add_argument("--unload-timeout", type=float, default=UNLOAD_TIMEOUT_S)
    ap.add_argument("cells", nargs="+")
    args = ap.parse_args(argv)

    with open(args.user_file) as f:
        user_prompt = f.read()
    systems = []
    for cell in args.cells:
        with open(cell.split("|")[1]) as f:
            systems.append(f.read())
    requests = build_requests(args.cells, user_prompt, systems)

    driver = Driver(
        args.exe, args.out, args.load_timeout, args.generate_timeout, args.unload_timeout
    )

    def on_signal(signum, _frame):
        driver.kill()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    rc = driver.run(args.model, requests)
    for e in driver.errors:
        print(f"agentic-gate-driver: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
