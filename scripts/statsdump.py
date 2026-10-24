#!/usr/bin/env python3
# statsdump.py -- record /stats to a JSONL file, so the dashboard's figures can be read back over
# time. /metrics carries most of them for a Prometheus scrape; this is for a machine without one:
# one JSON object per poll, appended, python3's standard library only.
#
#   scripts/statsdump.py [--port 8000] [--every 1] [--seconds 3600] [--out stats.jsonl]
#   scripts/statsdump.py --url http://host:8000 --every 2 --out week.jsonl
#   RADIANCE_API_KEY=... scripts/statsdump.py
#
# A row is the engine's snapshot verbatim plus `t` (the poll's unix time) and `dt_s` (seconds since
# the previous successful poll, from the monotonic clock). Nothing is derived: the dashboard's
# best/recent/now lines are computed from the same stream by whatever reads the file.
#
# A restart shows in the rows: `uptime_s` falls and `startup_s` is the restart's cost. The decode
# floor resets with the process, so `step_ns_decode_best` compares across restarts only against a
# known-good value for the same hardware.
#
# A failed poll leaves no row. Failures are reported on stderr as they happen and counted in the
# exit summary, and a run that wrote no row exits 1.
import argparse, json, os, sys, time, urllib.error, urllib.request

class RecorderDead(Exception):
    """The output file could not be reopened, so recording cannot continue. Raised instead of
    writing to a closed handle, whose ValueError would end the run without the exit summary."""


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--port", type=int, default=8000)
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

    # Append, never truncate: the rows carry their own timestamps.
    out = open(a.out, "a")
    prev, ok, lost, streak, rotated = None, 0, 0, 0, 0
    dead = False

    def rotate():
        """Moves the file to .1 (and .1 to .2, up to .5) and starts a new one."""
        nonlocal out, rotated
        try:
            out.close()   # close() flushes; on a full disk the dirty buffer fails to land
        except OSError:
            pass          # those rows are already counted lost
        try:
            for i in range(4, 0, -1):
                src = "%s.%d" % (a.out, i)
                if os.path.exists(src):
                    os.replace(src, "%s.%d" % (a.out, i + 1))
            os.replace(a.out, a.out + ".1")
            out = open(a.out, "a")
            rotated += 1
            sys.stderr.write("statsdump: rotated %s (generation %d)\n" % (a.out, rotated))
        except OSError:
            # The shift or the reopen failed (ENOSPC when the directory must grow). `out` is
            # closed here, so reopen it before re-raising: opening for append needs no disk
            # space, and the poller keeps counting losses.
            try:
                out = open(a.out, "a")
            except OSError as reopen_err:
                raise RecorderDead("reopen after failed rotation: %s" % reopen_err) from None
            raise   # the caller's size check swallows it
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
                # The first failure of a streak, then every tenth.
                if streak == 1 or streak % 10 == 0:
                    sys.stderr.write("statsdump: poll failed (%s), streak %d, %d lost\n"
                                     % (type(e).__name__, streak, lost))
            else:
                streak = 0
                row["t"] = t0
                # dt_s from the monotonic clock, so a wall-clock step cannot make it negative;
                # t is the row's wall-clock timestamp.
                row["dt_s"] = round(time.monotonic() - prev, 3) if prev is not None else 0.0
                prev = time.monotonic()
                line = json.dumps(row, separators=(",", ":")) + "\n"
                try:
                    out.write(line)
                    out.flush()   # a killed poller must not lose the polls it did get
                    ok += 1
                except OSError as e:
                    # A failed write (disk full) is counted and recording goes on; a partial
                    # line, if any, is an unparseable JSONL row.
                    lost += 1; streak += 1
                    sys.stderr.write("statsdump: write failed (%s), %d lost\n"
                                     % (type(e).__name__, lost))
                    # Drop the unwritten buffer and reopen: on a full disk a later tell() would
                    # flush it and fail outside this handler.
                    try:
                        out.close()
                    except OSError:
                        pass
                    try:
                        out = open(a.out, "a")
                    except OSError as reopen_err:
                        raise RecorderDead("reopen after failed write: %s" % reopen_err) from None
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
    except RecorderDead as e:
        sys.stderr.write("statsdump: stopped, the output cannot be written (%s)\n" % e)
        dead = True
    finally:
        try:
            out.close()
        except OSError:
            pass   # a flush failing at exit must not eat the summary line below
    sys.stderr.write("statsdump: %d rows written, %d polls lost, %d rotations, %s\n"
                     % (ok, lost, rotated, a.out))
    # A run that wrote nothing, or could not go on writing, exits 1.
    return 1 if ok == 0 or dead else 0

if __name__ == "__main__":
    sys.exit(main())
