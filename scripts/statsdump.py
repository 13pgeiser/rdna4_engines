#!/usr/bin/env python3
# statsdump.py -- record /stats to a JSONL file, so the dashboard's figures can be measured over
# time instead of only watched.
#
# The dashboard answers "what is this process doing right now". Answering "what was it doing at
# 3 a.m., and is its best step time worse than last week's" needs the same numbers kept. /stats
# already carries them -- the decode floor, throughput per window, cache behaviour, the draft
# counters -- and /metrics carries most of them for a Prometheus scrape. This is the third option,
# for anyone who does not run a scraper: one JSON object per poll, appended, no dependencies beyond
# python3.
#
#   scripts/statsdump.py [--port 8800] [--every 1] [--seconds 3600] [--out stats.jsonl]
#   scripts/statsdump.py --url http://host:8800 --every 2 --out week.jsonl
#   RADIANCE_API_KEY=... scripts/statsdump.py --port 8800
#
# WHAT IS RECORDED: the engine's own snapshot, verbatim, plus `t` (the poll's own unix time) and
# `dt_s` (seconds since the previous successful poll). Nothing is derived. The page's best/recent/
# now lines are computed in the browser from this same stream, and recomputing them here would put
# a second copy of that arithmetic where it could drift from the first -- so the analysis step
# derives them, from data that cannot disagree with the engine.
#
# Restarts are visible in the rows themselves: `uptime_s` falls and `startup_s` is the cost of the
# restart that just happened. The decode floor resets with the process, so `step_ns_decode_best`
# improving across a restart is a new process finding its own best moment, not a faster machine --
# compare `best` across restarts only against a known-good value for the same hardware.
#
# A poll that fails leaves NO row: a gap in the file is the honest record of a gap in the data, and
# a fabricated row would be a measurement. Consecutive failures are reported on stderr as they
# happen, and the exit summary states how many polls were lost, so an empty file cannot be mistaken
# for a quiet hour.
import argparse, json, os, sys, time, urllib.error, urllib.request

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--url", default=None, help="full base URL; overrides --port")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between polls")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long; 0 = run until interrupted")
    ap.add_argument("--out", default="stats.jsonl")
    ap.add_argument("--max-mib", type=float, default=256.0,
                    help="rotate the file when it passes this size (0 = never); "
                         "keeps 5 generations as .1 (newest) .. .5")
    ap.add_argument("--api-key", default=os.environ.get("RADIANCE_API_KEY"),
                    help="or set RADIANCE_API_KEY; the key is never echoed")
    a = ap.parse_args()

    base = (a.url or "http://127.0.0.1:%d" % a.port).rstrip("/")
    hdr = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
    deadline = time.time() + a.seconds if a.seconds > 0 else None

    # Append, never truncate: a file that already holds last week's data is worth more than a clean
    # start, and the rows carry their own timestamps.
    out = open(a.out, "a")
    prev, ok, lost, streak, rotated = None, 0, 0, 0, 0

    def rotate():
        """Size cap (audit 2026-10-20 §6): an append-only recorder with no cap is a
        disk-full timer. Shift .1->.2 ... keep 5, start fresh; rows carry their own
        timestamps so the chain stays readable in order."""
        nonlocal out, rotated
        try:
            out.close()   # close() flushes; on a full disk the dirty buffer fails to land
        except OSError:
            pass          # those rows are already counted lost -- the fresh handle is the point
        for i in range(4, 0, -1):
            src = "%s.%d" % (a.out, i)
            if os.path.exists(src):
                os.replace(src, "%s.%d" % (a.out, i + 1))
        os.replace(a.out, a.out + ".1")
        out = open(a.out, "a")
        rotated += 1
        sys.stderr.write("statsdump: rotated %s (generation %d)\n" % (a.out, rotated))
    sys.stderr.write("statsdump: %s/stats every %.1fs -> %s\n" % (base, a.every, a.out))
    try:
        while True:
            t0 = time.time()
            try:
                req = urllib.request.Request(base + "/stats", headers=hdr)
                with urllib.request.urlopen(req, timeout=max(5.0, 4 * a.every)) as r:
                    row = json.loads(r.read().decode())
            except (urllib.error.URLError, OSError, ValueError) as e:
                lost += 1; streak += 1
                # Report the first failure and then every tenth: one bad hour should be visible
                # without the file's own signal being buried in identical lines.
                if streak == 1 or streak % 10 == 0:
                    sys.stderr.write("statsdump: poll failed (%s), streak %d, %d lost\n"
                                     % (type(e).__name__, streak, lost))
            else:
                streak = 0
                row["t"] = t0
                # dt_s from the MONOTONIC clock: a wall-clock step (NTP) could make it
                # negative, and a consumer dividing counters by dt_s would inherit the
                # step. t stays wall-clock: it is the row's timestamp, not a delta.
                row["dt_s"] = round(time.monotonic() - prev, 3) if prev is not None else 0.0
                prev = time.monotonic()
                line = json.dumps(row, separators=(",", ":")) + "\n"
                try:
                    out.write(line)
                    out.flush()   # a killed poller must not lose the polls it did get
                    ok += 1
                except OSError as e:
                    # Disk full mid-write: loud, counted, and the poller SURVIVES to
                    # report it (audit §6: this path used to die with a traceback and
                    # no exit summary). The partial line, if any, is detectable JSONL.
                    lost += 1; streak += 1
                    sys.stderr.write("statsdump: write failed (%s), %d lost\n"
                                     % (type(e).__name__, lost))
                    # Drop the dirty buffer and reopen: a flush on the same full disk fails
                    # again, and a later tell() would flush it and die OUTSIDE this handler
                    # (judge slice 23: the cap check's tell() was the actual killer).
                    try:
                        out.close()
                    except OSError:
                        pass
                    try:
                        out = open(a.out, "a")
                    except OSError:
                        pass
                if a.max_mib > 0:
                    try:
                        if out.tell() >= a.max_mib * 1024 * 1024:
                            rotate()
                    except OSError:
                        pass   # a size check must never be the thing that kills the poller
            if deadline is not None and time.time() >= deadline:
                break
            # Sleep from the poll's START, so a slow engine does not walk the schedule later.
            left = a.every - (time.time() - t0)
            if left > 0:
                time.sleep(left)
    except KeyboardInterrupt:
        sys.stderr.write("statsdump: interrupted\n")
    finally:
        out.close()
    sys.stderr.write("statsdump: %d rows written, %d polls lost, %d rotations, %s\n"
                     % (ok, lost, rotated, a.out))
    # An empty file is a failure, not a quiet run: say so with the exit status, not only in prose.
    return 1 if ok == 0 else 0

if __name__ == "__main__":
    sys.exit(main())
