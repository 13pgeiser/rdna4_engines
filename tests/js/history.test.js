/* The history summary line: how it groups, and how it wraps.
 *
 * The line packs several categories into one strip -- a count, tok/s, prefill, ttft -- and each
 * category carries its own figures. Two rules the page must keep:
 *
 *   a category is ATOMIC. If it does not fit on the line it moves whole to the next line. It
 *     must never break mid-figure: "810" at the end of one line and "ms" on the next is worse
 *     than the wrap it replaced.
 *   the category break is a DOT and the break between figures inside one category is a PIPE.
 *
 * Both rules are structural, not string-level: HTML collapses runs of spaces, so the width of a
 * category break cannot come from spaces in the joined string. It comes from a padded element.
 * That is why these checks inspect the DOM the shipped code builds rather than the string it
 * used to build -- a string assertion here would have passed while the page rendered wrong. */
const fs = require("fs");
const path = require("path");
const src = require("./page.js");           // the page's <script> block
/* The CSS lives in the <style> block, which page.js does not return (the JS tests have no
   reason to read it). Read it from the same shipped file rather than widen page.js's export and
   break the two suites that treat its result as a plain string. */
const cpp = fs.readFileSync(path.join(__dirname, "..", "..", "core", "server", "dashboard.cpp"), "utf8");
const css = (cpp.match(/<style>([\s\S]*?)<\/style>/) || [, ""])[1];
const L = src.split("\n");
let fails = 0, checks = 0;
const ok = (c, m) => { checks++; if (!c) { fails++; console.log("  FAIL " + m); } };
const eq = (a, b, m) => { checks++; if (a !== b) { fails++; console.log("  FAIL " + m + ": got " + JSON.stringify(a) + " want " + JSON.stringify(b)); } };
function lines(a, b) {
  const i = L.findIndex((x) => x.startsWith(a));
  if (i < 0) throw new Error("no start line: " + a);
  let j = -1;
  for (let k = i; k < L.length; k++) if (L[k].startsWith(b)) { j = k; break; }
  if (j < 0) throw new Error("no end line: " + b);
  return L.slice(i, j + 1).join("\n");
}

console.log("== the CSS makes a category atomic and the break actually wide ==");
const grpCss = css.match(/#hist \.hgrp \{([^}]*)\}/);
ok(css.length > 200, "read the page <style> block (" + css.length + " chars)");
ok(grpCss, "#hist .hgrp rule exists");
ok(/white-space:\s*nowrap/.test(grpCss ? grpCss[1] : ""), "a category cannot break mid-figure");
const sepCss = css.match(/#hist \.hsep \{([^}]*)\}/);
ok(sepCss, "#hist .hsep rule exists");
ok(/padding:\s*0\s+\d+px/.test(sepCss ? sepCss[1] : ""), "the category break gets width from padding, not spaces");
const pad = sepCss ? +sepCss[1].match(/padding:0 (\d+)px/)[1] : 0;
ok(pad >= 4, "the padding is wide enough to read as a break (" + pad + "px, want >= 4)");

console.log("== the shipped summary assembly builds one atomic group per category ==");
/* Slice ONLY the summary-line assembly (mean/group -> the forEach that fills #hsum), not the
   whole renderer: the rest of histRender draws the table and needs a dozen module globals that
   have nothing to do with how the line groups or wraps. */
const si = L.findIndex((x) => x.startsWith("  const mean = (a) =>"));
if (si < 0) throw new Error("no summary start");
const ai = L.findIndex((x, k) => k > si && x.includes("hsum.appendChild(g);"));
if (ai < 0) throw new Error("no assembly anchor");
let ei = -1;
for (let k = ai; k < L.length; k++) if (L[k] === "  });") { ei = k; break; }
if (ei < 0) throw new Error("no assembly end");
const BLOCK = L.slice(si, ei + 1).join("\n");
ok(BLOCK.includes("hgrp") && BLOCK.includes("hsep"), "slice captured the grouping code (" + BLOCK.length + " chars)");

function mk(tag, cls, txt) {
  const o = { tag, cls, _t: txt || "", kids: [],
    appendChild(c) { this.kids.push(c); return c; },
    set textContent(v) { this._t = v; if (v === "") this.kids = []; },
    get textContent() { return this._t || ""; } };
  Object.defineProperty(o, "txt", { get() { return o._t || ""; } });
  return o;
}
const hsum = mk("span", "hsum");
const el = (t, c, x) => mk(t, c, x);
const doc = { createTextNode: (t) => ({ tag: "#text", cls: null, txt: t }) };
const rows = [];
for (let i = 0; i < 24; i++) rows.push({ tps: 200 + (i % 9), pf: 690 + (i % 5), ttft: 810, age: i * 3 });
new Function("el", "document", "$", "hRows", "tps", "cnt3", "dur", BLOCK)(
  el, doc, () => hsum, rows,
  (v) => Math.round(v) + " t/s", (v) => String(Math.round(v)), (v) => v + " ms");

const groups = hsum.kids;
eq(groups.length, 4, "four categories: count, tok/s, prefill, ttft");
ok(groups.every((g) => g.cls === "hgrp"), "every category is a .hgrp (the nowrap unit)");
console.log("== the dot introduces its category and travels with it ==");
eq(groups[0].kids.filter((k) => k.cls === "hsep").length, 0, "the first category carries no leading dot");
[1, 2, 3].forEach((i) => {
  const seps = groups[i].kids.filter((k) => k.cls === "hsep");
  eq(seps.length, 1, "category " + i + " carries exactly one dot");
  eq(groups[i].kids[0].cls, "hsep", "the dot is the FIRST child of its own group, not a sibling of it");
  eq(groups[i].kids[0].txt, "\u00b7", "and it is a middle dot, not a pipe");
});
console.log("== within a category the fields are separated by pipes ==");
const textOf = (g) => g.kids.map((k) => (k.txt || "")).join("");
/* The category's own text, excluding the leading dot that introduces it (that dot is the
   category BREAK and is SUPPOSED to be a dot; "no dot inside" is about the figures. */
const ownText = (g) => g.kids.filter((k) => k.cls !== "hsep").map((k) => k.txt || "").join("");
const tpsGroup = textOf(groups[1]);
const tpsOwn = ownText(groups[1]);
ok(tpsGroup.includes("|"), "tok/s separates mean / last 20 / last 5 with pipes: " + JSON.stringify(tpsGroup));
ok(!tpsOwn.includes("\u00b7"), "no dot among a category's own figures (the dot means category change): " + JSON.stringify(tpsOwn));
ok(/last 20:.*\|.*last 5:/.test(tpsGroup), "both windows present and pipe-separated");
console.log("== nothing can orphan a unit ==");
/* The failure this exists to prevent: "ttft median 810" ending a line and "ms" starting the next.
 * That happens when the value and its unit are separate wrap opportunities. */
const ttft = textOf(groups[3]);
const ttftGroup = groups[3];
const textNodes = ttftGroup.kids.filter((k) => k.cls !== "hsep");
eq(textNodes.length, 1, "the ttft category is a single text node, so value and unit cannot split");
ok(/ttft median \d+ ms$/.test(textNodes[0].txt), "value and unit are one string: " + JSON.stringify(textNodes[0].txt));
console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped history line");
process.exit(fails ? 1 : 0);
