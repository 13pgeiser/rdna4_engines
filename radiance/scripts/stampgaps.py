#!/usr/bin/env python3
# stampgaps.py -- where a recorded pass leaves the card idle, from RADIANCE_AQL_STAMP's CSV.
#
#   RADIANCE_AQL_STAMP=50 RADIANCE_AQL_STAMP_OUT=/tmp/st.csv build/radiance ...   (drive a load)
#   scripts/stampgaps.py /tmp/st.csv [label] [--top 12]
#
# The engine writes one line a dispatch of every Nth pass on card 0: pass, seq (its record),
# queue, kernel, ordered (the packet waited for the one before it on its queue), chained, start_ns,
# end_ns, play, records. A pass is the plays of one tape from its record 0; its tape length
# (`records`) tells kinds of pass apart (draft, verify, each prefill size), so every figure is per
# kind, the median over that kind's passes.
#
# Per pass: the wall (first start to last end), the busy time as the union of every queue's
# intervals (two queues running at once count once), the idle share (wall minus busy over wall),
# the overlap between queues (the sum of per-queue busy times minus the union), and the gap before
# each ordered start (its start minus the latest end of the earlier dispatches on its queue). The
# gaps are what a chained form would save; the idle share is the most it could. Over several
# queues a gap also holds the time its queue waited for another one (a second lane's dispatch
# waits for its join), so there the summed gaps overstate and the p50 and the idle share are the
# figures to read. Per kernel name: its share of the summed dispatch time.
#
# A pass with fewer lines than the longest of its kind (the stamp's signal pool ran out mid-pass)
# or with a line without times is left out and counted. The stamping costs throughput of its own
# (on GLM-5.3 at TP8, one pass in 50: 0.3 % at one stream, 4 % at eight), so take N larger where
# many streams make the passes short. Standard library only.
import csv
import statistics
import sys
from collections import defaultdict


def union(iv):
    """The length of the union of half-open intervals [a, b)."""
    tot, cur_a, cur_b = 0, None, None
    for a, b in sorted(iv):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                tot += cur_b - cur_a
            cur_a, cur_b = a, b
        elif b > cur_b:
            cur_b = b
    if cur_b is not None:
        tot += cur_b - cur_a
    return tot


def pct(v, p):
    if not v:
        return 0.0
    v = sorted(v)
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def read(path):
    passes = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            passes[int(r["pass"])].append(r)
    return passes


def analyse(rows):
    """One pass's figures; None when a line has no times."""
    if any(not r["start_ns"] or not r["end_ns"] for r in rows):
        return None
    rows = sorted(rows, key=lambda r: int(r["seq"]))
    iv = [(int(r["start_ns"]), int(r["end_ns"])) for r in rows]
    t0 = min(a for a, _ in iv)
    wall = max(b for _, b in iv) - t0
    busy = union(iv)
    per_q = defaultdict(list)
    for r, x in zip(rows, iv):
        per_q[r["queue"]].append(x)
    busy_q = {q: union(v) for q, v in per_q.items()}
    gaps = []
    last_end = {}
    for r, (a, b) in zip(rows, iv):
        q = r["queue"]
        if r["ordered"] == "1" and q in last_end:
            gaps.append(max(0, a - last_end[q]))
        last_end[q] = max(last_end.get(q, 0), b)
    kern = defaultdict(int)
    for r, (a, b) in zip(rows, iv):
        kern[r["kernel"]] += b - a
    return {
        "n": len(rows),
        "ordered": sum(r["ordered"] == "1" for r in rows),
        "chained": sum(r["chained"] == "1" for r in rows),
        "plays": len({r["play"] for r in rows}),
        "queues": len(per_q),
        "wall": wall,
        "busy": busy,
        "idle": wall - busy,
        "overlap": sum(busy_q.values()) - busy,
        "gaps": gaps,
        "kern": kern,
        "dev": sum(b - a for a, b in iv),
    }


def main():
    args = sys.argv[1:]
    top = 12
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit("usage: stampgaps.py CSV [label] [--top K]")
    path = args[0]
    label = args[1] if len(args) > 1 else path
    passes = read(path)
    by_kind = defaultdict(list)
    for p, rows in sorted(passes.items()):
        by_kind[int(rows[0]["records"])].append(rows)
    print(f"{label}: {len(passes)} stamped passes in {path}\n")
    for kind, ps in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        full = max(len(rows) for rows in ps)
        res = [a for a in (analyse(rows) if len(rows) == full else None for rows in ps) if a]
        dropped = len(ps) - len(res)
        print(f"pass of {kind} records")
        if not res:
            print(f"  no complete pass ({dropped} incomplete)\n")
            continue
        med = lambda k: statistics.median(a[k] for a in res)
        ms = lambda ns: ns / 1e6
        gap50 = statistics.median(pct(a["gaps"], 0.5) for a in res)
        gap90 = statistics.median(pct(a["gaps"], 0.9) for a in res)
        gapsum = statistics.median(sum(a["gaps"]) for a in res)
        idle = statistics.median(a["idle"] / a["wall"] for a in res if a["wall"] > 0)
        print(f"  passes complete / incomplete      {len(res)} / {dropped}")
        print(f"  dispatches a pass                 {med('n'):.0f}")
        print(f"    ordered / chained               {med('ordered'):.0f} / {med('chained'):.0f}")
        print(f"  plays / queues                    {med('plays'):.0f} / {med('queues'):.0f}")
        print(f"  wall ms                           {ms(med('wall')):.3f}")
        print(f"  busy ms (union)                   {ms(med('busy')):.3f}")
        print(f"  idle ms / share                   {ms(med('idle')):.3f} / {100 * idle:.1f} %")
        print(f"  overlap between queues ms         {ms(med('overlap')):.3f}")
        print(f"  summed dispatch time ms           {ms(med('dev')):.3f}")
        print(f"  gap before an ordered start us    p50 {gap50 / 1e3:.2f}  p90 {gap90 / 1e3:.2f}")
        print(f"  summed gaps ms                    {ms(gapsum):.3f}")
        kern = defaultdict(int)
        dev = sum(a["dev"] for a in res)
        for a in res:
            for k, v in a["kern"].items():
                kern[k] += v
        print(f"  top {top} kernels (share of dispatch time, ms a pass)")
        for k, v in sorted(kern.items(), key=lambda kv: -kv[1])[:top]:
            print(f"    {100 * v / dev:5.1f} %  {ms(v / len(res)):8.3f}  {k}")
        print()


if __name__ == "__main__":
    main()
