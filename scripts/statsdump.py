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
    ap.add_argument("--api-key", default=os.environ.get("RADIANCE_API_KEY"),
                    help="or set RADIANCE_API_KEY; the key is never echoed")
    a = ap.parse_args()

    base = (a.url or "http://127.0.0.1:%d" % a.port).rstrip("/")
    hdr = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
    deadline = time.time() + a.seconds if a.seconds > 0 else None

    # Append, never truncate: a file that already holds last week's data is worth more than a clean
    # start, and the rows carry their own timestamps.
    out = open(a.out, "a")
    prev, ok, lost, streak = None, 0, 0, 0
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
                row["dt_s"] = round(t0 - prev, 3) if prev is not None else 0.0
                prev = t0
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                out.flush()   # a killed poller must not lose the polls it did get
                ok += 1
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
    sys.stderr.write("statsdump: %d rows written, %d polls lost, %s\n" % (ok, lost, a.out))
    # An empty file is a failure, not a quiet run: say so with the exit status, not only in prose.
    return 1 if ok == 0 else 0

if __name__ == "__main__":
    sys.exit(main())
