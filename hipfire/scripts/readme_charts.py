#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the README data charts from docs/assets/readme-data.json.

    python3 scripts/readme_charts.py          # write the SVGs
    python3 scripts/readme_charts.py --check  # exit 1 if a committed SVG is stale

Stdlib only; output is byte-deterministic. Each chart is written in a light and
a dark variant (transparent background) for the README's <picture> elements:
docs/assets/readme/gains-042-{light,dark}.svg and
docs/assets/readme/gpu-matrix-{light,dark}.svg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/assets/readme-data.json"
OUT = ROOT / "docs/assets/readme"
WIDTH = 880
FONT = "font-family=\"-apple-system,Segoe UI,Helvetica,Arial,sans-serif\""
# gains-042 follows hipfire.dev (Inter / JetBrains Mono, accent #e11d2a); system fallbacks, no webfonts.
SITE_FONT = "font-family=\"Inter,-apple-system,Segoe UI,Helvetica,Arial,sans-serif\""
SITE_MONO = "font-family=\"JetBrains Mono,SFMono-Regular,Menlo,Consolas,monospace\""

THEMES = {
    "light": {"fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "before": "#8c959f", "after": "#d1242f", "site": "#e11d2a",
              "tuned": "#d1242f", "generic": "#bf8700", "none": "#d1d9e0"},
    "dark": {"fg": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d",
             "before": "#656c76", "after": "#ff7b72", "site": "#e11d2a",
             "tuned": "#ff7b72", "generic": "#d29922", "none": "#3d444d"},
}


def fmt(v: float) -> str:
    return f"{v:,.1f}" if v != int(v) or v < 1000 else f"{int(v):,}"


def text(x, y, s, fill, size=13, anchor="start", weight="normal", font=FONT) -> str:
    return (f'<text x="{x}" y="{y}" {font} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')


def svg(height: int, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">\n'
            f"<title>{escape(title)}</title>\n" + "\n".join(body) + "\n</svg>\n")


def gains(data: dict, t: dict) -> str:
    g = data["gains"]
    label_w, bar_x, bar_max = 230, 250, 470
    acc, F, M = t["site"], SITE_FONT, SITE_MONO
    body, y = [], 28
    body.append(text(0, y, g["title"], t["fg"], 18, weight="bold", font=F))
    y += 18
    body.append(text(0, y, g["subtitle"], t["muted"], 12, font=F))
    y += 16
    body.append(f'<rect x="0" y="{y}" width="12" height="12" fill="{t["before"]}"/>')
    body.append(text(18, y + 11, g["legend"]["before"], t["muted"], 12, font=F))
    body.append(f'<rect x="230" y="{y}" width="12" height="12" fill="{acc}"/>')
    body.append(text(248, y + 11, g["legend"]["after"], t["muted"], 12, font=F))
    body.append(f'<path d="M560 {y + 2}V{y + 10}M560 {y + 6}H576M576 {y + 2}V{y + 10}" stroke="{t["fg"]}" fill="none"/>')
    body.append(text(582, y + 11, "min\u2013max", t["muted"], 12, font=F))
    y += 40
    for p in g["panels"]:
        body.append(text(0, y, f'{p["panel"]} ({p["unit"]}, {"higher" if p["higher_is_better"] else "lower"} is better)',
                         t["fg"], 14, weight="bold", font=F))
        y += 16
        body.append(text(0, y, p["caption"], t["muted"], 11, font=F))
        y += 12
        scale = max(max(c["max"] if "max" in c else c["median"] for c in (r["before"], r["after"]) if c)
                    for r in p["rows"])
        for r in p["rows"]:
            b, a = r["before"], r["after"]
            body.append(text(label_w, y + 24, r["label"], t["fg"], 12, anchor="end", font=F))
            for i, (c, col) in enumerate(((b, t["before"]), (a, acc))):
                yy = y + 8 + i * 18
                if c is None:
                    msg = "no 0.4.1 measurement" if i == 0 else "pending 0.4.2 run"
                    body.append(text(bar_x, yy + 11, msg, t["muted"], 11, font=M))
                    continue
                w = round(bar_max * c["median"] / scale, 1)
                body.append(f'<rect x="{bar_x}" y="{yy}" width="{w}" height="14" rx="2" fill="{col}"/>')
                end = w
                if "min" in c:
                    x0 = round(bar_x + bar_max * c["min"] / scale, 1)
                    x1 = round(bar_x + bar_max * c["max"] / scale, 1)
                    body.append(f'<path d="M{x0} {yy + 3}V{yy + 11}M{x0} {yy + 7}H{x1}M{x1} {yy + 3}V{yy + 11}" '
                                f'stroke="{t["fg"]}" fill="none"/>')
                    end = max(w, x1 - bar_x)
                body.append(text(round(bar_x + end + 6, 1), yy + 12, fmt(c["median"]), t["fg"], 12, font=M))
            if b and a:
                pct = (a["median"] - b["median"]) / b["median"] * 100
                body.append(text(WIDTH - 4, y + 24, f"{pct:+.1f}%".replace("-", "\u2212"),
                                 acc if pct >= 0 else t["fg"], 13, anchor="end", weight="bold", font=F))
            elif a and r.get("new"):
                body.append(text(WIDTH - 4, y + 24, "new in 0.4.2", acc, 12, anchor="end", weight="bold", font=F))
            y += 46
        y += 18
    for line in g["footnotes"]:
        body.append(text(0, y, line, t["muted"], 11, font=F))
        y += 15
    return svg(y, body, g["alt"])


def matrix(data: dict, t: dict) -> str:
    m = data["matrix"]
    col_x, col_w, row_h = 250, 105, 40
    body, y = [], 28
    body.append(text(0, y, "GPU support matrix", t["fg"], 18, weight="bold"))
    y += 34
    for i, f in enumerate(m["features"]):
        body.append(text(col_x + i * col_w + col_w / 2, y, f, t["fg"], 12, anchor="middle", weight="bold"))
    y += 12
    for a in m["archs"]:
        body.append(f'<line x1="0" y1="{y}" x2="{WIDTH}" y2="{y}" stroke="{t["grid"]}"/>')
        body.append(text(0, y + 18, a["label"], t["fg"], 13, weight="bold"))
        body.append(text(0, y + 33, a["detail"], t["muted"], 11))
        for i, c in enumerate(a["cells"]):
            cx = col_x + i * col_w + col_w / 2
            fill = t[c]
            if c == "none":
                body.append(text(cx, y + 26, "\u2014", t["muted"], 14, anchor="middle"))
            else:
                body.append(f'<rect x="{cx - 34}" y="{y + 11}" width="68" height="20" rx="10" fill="{fill}"/>')
                body.append(text(cx, y + 25, c, "#ffffff", 11, anchor="middle", weight="bold"))
        y += row_h
    body.append(f'<line x1="0" y1="{y}" x2="{WIDTH}" y2="{y}" stroke="{t["grid"]}"/>')
    y += 22
    body.append(text(0, y, "tuned: arch-specific kernels or default-on route \u00b7 generic: portable typed-dispatch path",
                     t["muted"], 11))
    y += 16
    body.append(text(0, y, "Capability, not admission or certification. Sources: docs/assets/readme-data.json.",
                     t["muted"], 11))
    return svg(y + 10, body, "hipfire GPU support matrix")


def main() -> int:
    check = "--check" in sys.argv[1:]
    data = json.loads(DATA.read_text())
    stale = []
    for name, fn in (("gains-042", gains), ("gpu-matrix", matrix)):
        for theme, t in THEMES.items():
            path = OUT / f"{name}-{theme}.svg"
            content = fn(data, t)
            if check:
                if not path.exists() or path.read_text() != content:
                    stale.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(content)
    if stale:
        print("readme_charts: stale, rerun scripts/readme_charts.py: " + ", ".join(stale))
        return 1
    print("readme_charts: " + ("charts match data" if check else "wrote 4 SVGs"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
