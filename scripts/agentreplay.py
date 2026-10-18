#!/usr/bin/env python3
"""Several agent conversations with subagents against a live server, and what the prefix cache made
of them.

An agent client sends each turn as the whole transcript so far plus a new message, and hands tasks
to subagents: short conversations that start from the same system prompt while the parent waits.
Here --parents conversations run at once. Each grows by a tool result and a reply per turn until its
prompt reaches --context tokens (or it has taken --turns turns), and every --sub-every turns runs a
subagent of --sub-turns turns before its own next turn, which carries the subagent's answer. Every
conversation starts from one system prompt that opens with a fresh nonce, so a run shares nothing
with an earlier one. With --rewrite a parent's prompt ends with a notes block that its next turn
replaces, as a client that rewrites its last message does, so each turn leaves the previous one's
chain before its end.

The text is slices of this repository's sources and documents, so the token mix is a coding
agent's. Replies are --reply tokens with end-of-text ignored, and the slices follow --seed, so two
servers are sent the same conversations.

Per turn it records usage.prompt_tokens, usage.prompt_tokens_details.cached_tokens and
timings.ttft_ms. At the end it prints, per kind of turn, how much of the prompt the cache served,
the continuing turns that reused less than half of the prompt their conversation had already sent
(a full re-prefill), and the server's prefix-cache counters across the run (/metrics). On a hybrid
model cached_tokens is where the linear state resumed, a checkpoint position, not the attention
hit; radiance:prefix_hit_cut_tokens_total counts what the attention cache held and admission
computed again for want of a snapshot.

  python3 scripts/agentreplay.py --url http://127.0.0.1:8100 --parents 4 --context 200000 \\
      --out run.jsonl

Needs only python3. --api-key-file names a file holding the key, so the key is never on a command
line.
"""
import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARS_PER_TOKEN = 3.6       # this repository's code and markdown, as Qwen tokenizers count it

KINDS = ("parent", "parent after subagent", "subagent first turn", "subagent later turn")
COUNTERS = (
    "vllm:prompt_tokens_total",
    "radiance:prefill_tokens_total",
    "radiance:attn_cached_tokens_total",
    "radiance:linear_cached_tokens_total",
    "radiance:prefix_hit_cut_tokens_total",
    'radiance:prefix_hit_cuts_total{reason="no_snapshot"}',
    'radiance:prefix_cache_ckpt_evictions_by_reason_total{reason="store_full"}',
    'radiance:prefix_cache_ckpt_evictions_by_reason_total{reason="retention"}',
    "radiance:prefix_cache_ckpt_refused_total",
    "radiance:prefix_cache_evictions_total",
    "radiance:linear_checkpoints_written_total",
)


def corpus():
    """This repository's text, concatenated in a fixed order."""
    parts = []
    for top in ("docs", "core", "arch", "libref"):
        for d, _, files in sorted(os.walk(os.path.join(ROOT, top))):
            for f in sorted(files):
                if f.endswith((".md", ".cpp", ".h")):
                    with open(os.path.join(d, f), encoding="utf-8", errors="replace") as h:
                        parts.append(h.read())
    text = "\n".join(parts)
    if len(text) < 1 << 20:
        sys.exit("agentreplay: found under 1 MiB of text in the repository")
    return text


class Server:
    def __init__(self, url, key, timeout):
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _req(self, path, body=None):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError("%s -> HTTP %d: %s" % (path, e.code, e.read()[:400])) from None

    def model(self):
        return json.loads(self._req("/v1/models"))["data"][0]["id"]

    def metrics(self):
        out = {}
        for line in self._req("/metrics").decode().splitlines():
            if line.startswith("#") or " " not in line:
                continue
            k, v = line.rsplit(" ", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
        return out

    def complete(self, model, prompt, reply):
        body = {"model": model, "prompt": prompt, "max_tokens": reply, "temperature": 0,
                "ignore_eos": True, "stream": False}
        r = json.loads(self._req("/v1/completions", body))
        u = r.get("usage", {})
        t = r.get("timings", {})
        return (r["choices"][0]["text"], u.get("prompt_tokens", 0),
                u.get("prompt_tokens_details", {}).get("cached_tokens", 0), t.get("ttft_ms"))


def counter(m, name):
    """A counter by name; a bare name also matches its series that carry only the model label."""
    if name in m:
        return m[name]
    base, _, label = name.partition("{")
    for k, v in m.items():
        kb, _, kl = k.partition("{")
        if kb == base and (not label or label.rstrip("}") in kl):
            return v
    return None


def parent(i, a, srv, model, text, system, log, lock):
    rng = random.Random(a.seed * 1000 + i)

    def chunk(tokens):
        n = int(tokens * CHARS_PER_TOKEN)
        at = rng.randrange(0, len(text) - n)
        return text[at:at + n]

    def turn(kind, convo, prompt, prev):
        reply, n_prompt, cached, ttft = srv.complete(model, prompt, a.reply)
        rec = {"conversation": convo, "kind": kind, "prompt_tokens": n_prompt,
               "cached_tokens": cached, "ttft_ms": ttft, "prev_prompt_tokens": prev,
               "t": time.time()}
        with lock:
            log.append(rec)
            if a.out:
                a.out.write(json.dumps(rec) + "\n")
                a.out.flush()
        return reply, n_prompt

    # The transcript is `kept`, which every later turn sends again, and under --rewrite a last
    # message `tail`, which the next turn replaces.
    kept = system + "\n\nUser (session %d): %s\n" % (i, chunk(200))
    tail = "\nNotes:\n" + chunk(rng.randint(*a.turn) // 2) + "\n" if a.rewrite else ""
    prev, turns, kind = 0, 0, KINDS[0]
    while prev + a.turn[1] < a.context and (not a.turns or turns < a.turns):
        reply, prev = turn(kind, "p%d" % i, kept + tail, prev)
        kind = KINDS[0]
        if not a.rewrite:
            kept += tail
        kept += "\nAssistant: " + reply + "\nTool result:\n" + chunk(rng.randint(*a.turn)) + "\n"
        if a.rewrite:
            tail = "\nNotes:\n" + chunk(rng.randint(*a.turn) // 2) + "\n"
        turns += 1
        if a.sub_every and turns % a.sub_every == 0:
            sub = system + "\n\nTask from session %d: %s\n" % (i, chunk(300))
            sprev = 0
            for k in range(a.sub_turns):
                sreply, sprev = turn(KINDS[2] if k == 0 else KINDS[3], "p%d.s%d" % (i, turns),
                                     sub, sprev)
                sub += "\nAssistant: " + sreply + "\nTool result:\n" + \
                    chunk(rng.randint(*a.turn) // 2) + "\n"
            kept += "\nSubagent answer:\n" + sreply + "\n"
            kind = KINDS[1]


def run_parent(i, a, srv, model, text, system, log, lock, errors):
    try:
        parent(i, a, srv, model, text, system, log, lock)
    except Exception as e:      # a failed request invalidates the run; main reports it
        with lock:
            errors.append("parent %d: %s" % (i, e))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--api-key-file", help="a file holding the server's --api-key")
    p.add_argument("--model", help="default: the first /v1/models lists")
    p.add_argument("--parents", type=int, default=4, help="conversations at once (default 4)")
    p.add_argument("--context", type=int, default=200000,
                   help="a parent stops when its prompt would pass this many tokens")
    p.add_argument("--system", type=int, default=12000, help="system prompt tokens (default 12000)")
    p.add_argument("--turn", default="1000,6000",
                   help="tool result tokens a turn adds, LO,HI (default 1000,6000)")
    p.add_argument("--turns", type=int, default=0,
                   help="a parent stops after this many turns of its own; 0: at --context")
    p.add_argument("--reply", type=int, default=256, help="reply tokens (default 256)")
    p.add_argument("--sub-every", type=int, default=4,
                   help="a parent runs a subagent every N turns; 0 runs none (default 4)")
    p.add_argument("--sub-turns", type=int, default=5, help="turns a subagent runs (default 5)")
    p.add_argument("--rewrite", action="store_true",
                   help="a parent's last message, a notes block, is replaced every turn")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800, help="seconds a request may take")
    p.add_argument("--out", help="write one JSON line per turn here")
    a = p.parse_args()
    a.turn = tuple(int(x) for x in a.turn.split(","))
    key = None
    if a.api_key_file:
        with open(a.api_key_file) as h:
            key = h.read().strip()
    a.out = open(a.out, "w") if a.out else None

    srv = Server(a.url, key, a.timeout)
    try:
        model = a.model or srv.model()
    except (RuntimeError, OSError) as e:
        sys.exit("agentreplay: %s: %s" % (a.url, e))
    text = corpus()
    rng = random.Random(a.seed)
    at = rng.randrange(0, len(text) - int(a.system * CHARS_PER_TOKEN))
    system = "Session nonce %016x.\n" % random.SystemRandom().getrandbits(64) + \
        text[at:at + int(a.system * CHARS_PER_TOKEN)]

    m0 = srv.metrics()
    log, lock, errors = [], threading.Lock(), []
    t0 = time.time()
    th = [threading.Thread(target=run_parent,
                           args=(i, a, srv, model, text, system, log, lock, errors))
          for i in range(a.parents)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    if errors:
        sys.exit("agentreplay: the run failed:\n  " + "\n  ".join(errors))
    m1 = srv.metrics()

    print("agentreplay: %d parents, %d turns in %.0f s against %s (%s)%s" % (
        a.parents, len(log), wall, a.url, model, ", rewriting" if a.rewrite else ""))
    print("%-22s %6s %12s %12s %7s %8s %10s" % ("turn", "turns", "prompt tok", "cached tok",
                                               "cached", "re-pref", "ttft p50 s"))
    tot_p = tot_c = 0
    for k in KINDS:
        rs = [r for r in log if r["kind"] == k]
        if not rs:
            continue
        pt = sum(r["prompt_tokens"] for r in rs)
        ct = sum(r["cached_tokens"] for r in rs)
        tot_p, tot_c = tot_p + pt, tot_c + ct
        full = sum(1 for r in rs if r["prev_prompt_tokens"] > 0 and
                   r["cached_tokens"] * 2 < r["prev_prompt_tokens"])
        tt = [r["ttft_ms"] for r in rs if r["ttft_ms"] is not None]
        print("%-22s %6d %12d %12d %6.1f%% %8d %10s" % (
            k, len(rs), pt, ct, 100.0 * ct / max(pt, 1), full,
            "%.2f" % (statistics.median(tt) / 1000) if tt else "-"))
    print("%-22s %6d %12d %12d %6.1f%%" % ("all", len(log), tot_p, tot_c,
                                            100.0 * tot_c / max(tot_p, 1)))
    print("server counters across the run:")
    for name in COUNTERS:
        a0, a1 = counter(m0, name), counter(m1, name)
        if a1 is None:
            print("  %-76s (not exported)" % name)
        else:
            print("  %-76s %14.0f" % (name, a1 - (a0 or 0)))


if __name__ == "__main__":
    main()
