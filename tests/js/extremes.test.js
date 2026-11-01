/* Session extremes: the age clause, and the gap/restart split. Executes the shipped
   noteSample / triple / acceptTriple / resetWindows / newEngineRun, sliced by line
   range out of the page and CALLED -- defining a sliced function proves nothing.
   pct() is stubbed (not under test); nowS is declared here because in the page it is
   module-level and so invisible inside new Function. */
const src = require("./page.js");
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
const BLOCK = [
  "let nowS = 0;",
  /* resetWindows() clears these module-level collaborators; they are not under
     test, but it READS hist, so they have to exist for it to run at all. */
  "const hist = {}, moveHist = { up: [], down: [] };",
  "let prev = null, prevAt = 0, accPrev = null;",
  'const DASH = "\\u2013";',
  "const finite = (v) => v != null && isFinite(v);",
  "const pct = (f, d = 1) => (f == null ? DASH : (f * 100).toFixed(d) + '%');",
  lines("function dur(", "}"),
  lines("const RECENT_S =", "const RECENT_PREFILL_S"),
  lines("const extAt =", "let lastActiveC"),
  lines("function noteSample(", "}"),
  lines("function triple(", "}"),
  lines("function acceptTriple(", "}"),
  lines("const accPool =", "const accPool ="),
  lines("function resetWindows", "}"),
  lines("function newEngineRun", "}"),
].join("\n");
function fresh() {
  const box = {};
  new Function("box", BLOCK + "\nbox.api={noteSample,triple,acceptTriple,resetWindows,newEngineRun,extAt,winAt,extAtS,setNow:(v)=>{nowS=v;}};")(box);
  return box.api;
}
const F = (v) => v + " t/s";
console.log("== the age clause appears only when the extreme stands alone ==");
let b = fresh();
b.setNow(100); b.noteSample("dec", 1, 500, 15, true);
let out = b.triple("dec", 1, "best", F);
ok(!out.includes("ago"), "window full: no age clause (got " + JSON.stringify(out) + ")");
ok(out.includes("recent"), "and the window figures still render");
b.resetWindows(); b.setNow(150);
out = b.triple("dec", 1, "best", F);
ok(/best 500 t\/s · 50\.0 s ago$/.test(out), "extreme alone: age names the span (got " + JSON.stringify(out) + ")");
b.setNow(100.4);
ok(!b.triple("dec", 1, "best", F).includes("ago"), "an extreme set 0.4s ago is current: no clause");
console.log("== a stale extreme is distinguishable from a fresh one ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, false); b.setNow(1000);
out = b.triple("dec", 1, "best", F);
ok(out.includes("15m 00s ago"), "15-minute-old extreme says so (got " + JSON.stringify(out) + ")");
ok(out.includes("best 500"), "and still shows the figure itself");
console.log("== a GAP keeps the extreme, and says how old it is ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, true); b.noteSample("accept", 1, 0.9, 15, false);
b.setNow(400); b.resetWindows();
out = b.triple("dec", 1, "best", F);
ok(out.includes("best 500"), "the session extreme survives a gap (got " + JSON.stringify(out) + ")");
ok(out.includes("5m 00s ago"), "and names its age (got " + JSON.stringify(out) + ")");
eq(Object.keys(b.extAt).length, 2, "extAt untouched by a gap");
console.log("== a RESTART drops the extremes: a new process inherits nothing ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, true); b.noteSample("accept", 1, 0.9, 15, false);
b.setNow(400); b.newEngineRun();
eq(Object.keys(b.extAt).length, 0, "extAt cleared by newEngineRun");
eq(Object.keys(b.extAtS).length, 0, "extAtS cleared by newEngineRun");
eq(b.triple("dec", 1, "best", F), "", "the previous run's extreme no longer renders");
eq(b.acceptTriple(1), "", "the previous run's acceptance extreme no longer renders");
b.setNow(200); b.noteSample("dec", 1, 700, 15, false);
out = b.triple("dec", 1, "best", F);
ok(!out.includes("ago"), "a fresh extreme carries no age clause (got " + JSON.stringify(out) + ")");
ok(/best 700 t\/s$/.test(out), "and reads as the new run's own figure");
console.log("== the age tracks the NEWER extreme, not the first ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, false);
b.setNow(400); b.noteSample("dec", 1, 900, 15, false); b.setNow(460);
out = b.triple("dec", 1, "best", F);
ok(out.includes("best 900"), "the surviving figure is the newest extreme (got " + JSON.stringify(out) + ")");
ok(out.includes("1m 00s ago"), "and its age is the newer stamp (got " + JSON.stringify(out) + ")");
console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped extreme code");
process.exit(fails ? 1 : 0);
