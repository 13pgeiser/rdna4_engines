# Devlog: the decode floor, and the advice that had to die

The story behind `step_ns_decode_best`, `step_ns_decode_recent`, and why the
dashboard shows them without a verdict. Written 2026-10-18, the day all of it
happened. The user-facing explanation is §6.1 of the guide; this is the log of
how we got there, including the parts that failed.

## What shipped

Two engine-measured figures on the decode step tile and on `/stats`: the fastest
window of 32 single-sequence decode steps (`best`) and the most recent one
(`recent`), plus `startup_s` (bringup cost) as a raw field. The estimator is
`rad::DecodeFloor` in `core/sched/metrics.h`. No verdict, no restart advice.

## The chain of failures

Every rung below was built, tested, and — for the first and fourth — deployed
before it was refuted. None of them failed because of a coding mistake; each
failed because a premise nobody had measured turned out to be false.

### 1. A raw min over step brackets latched an artifact (deployed, failed live)

The first floor was a plain minimum over host-side decode-step brackets. On a
live TP2 lane it latched **1.16 ms** against a **15.55 ms** mean on the first
request after bringup — and a min never recovers, so the advisory read
"heavily drifted, restart pays back fast" forever.

The artifact: step timing is host-side, and with pipelining on, the next forward
pass is issued behind the current step's commit. Behind a step with long host
work (a prefill chunk, a KV checkpoint save), the decode pass finishes before
its own bracket opens — the bracket holds readback and commit and no GPU work.
It is a real step advancing a real token, so no token-count guard catches it.

### 2. Two plausible guards, refuted by measurement, not built

- *"The bracket is rare — only steps issued ahead are affected."* Refuted:
  `engine_steps_ahead_total == engine_steps_decode_total` (27934 == 27934).
  With pipelining, issue-ahead is universal.
- *"A zero-token decode step produces the degenerate bracket."* Refuted in
  source: `want <= 0` breaks before the bracket opens. The scheduler cannot
  make one.

### 3. The predecessor-phase gate: passed every test, failed live

Hypothesis: a decode step timed behind a *prefill* is the confound; gate the
floor on "previous step was also decode-only." Built, passed **38,820** unit
checks, deployed. Floor still latched: **1.34 ms** vs a 15.70 ms mean.

The confound was predecessor *host work*, not predecessor *phase*. Prefill is
one source of long host work; a KV checkpoint save every 256 tokens is another.
A gate on the phase can see neither. This is the most instructive failure of the
day: a test suite can be green, large, and completely blind, because the fixture
runs steps in microseconds and cannot manufacture the degenerate bracket the
test was about.

### 4. Window averaging: fixed the latch, kept the phantom (deployed, failed live)

Averaging a window of single-row steps before the min dilutes a *lone*
degenerate. But they are not rare — ~1.4% of steps — and a min over hundreds of
windows reliably lands on a degenerate-heavy window. The floor sat ~10% below
the true floor: still unreachable, still phantom drift. Simulation showed this
before deployment; the unit tests, again, could not see it.

The fix that worked: reject candidates at the **step level**, before they enter
a window — a step under half the running mean of accepted steps is a degenerate.
The two populations are ~10x apart (degenerates ~9% of a real step, real steps
~96%), so the threshold is the midpoint of a gap, not a tuned constant. Window
1 is recorded by neither figure: the gate needs a window of accepted samples to
judge one, and recording the un-gated first window latches an unreachable floor.

### 5. The advisory itself: right numbers, wrong premise (deployed, removed live)

With a sound floor, the advisory compared it against the browser's per-poll
statistics. Two instruments, different sample counts — a measured **7.7%**
phantom drift on a lane that had not changed. Fixed by having the engine report
both sides (`recent` added), same stream, same window size.

Then the operator watched the live line: *"best 14.89 · restart pays back in
18m56s"* on an engine minutes from its restart, then *"12 minutes"*. His
reductio — "do I restart every 18 minutes?" — was the falsification, and the
instrument agreed immediately:

- `best` held flat at 14.79 ms all session.
- `recent` moved between 15.2 and 16.5 ms — **3% to 11%** — on a healthy lane.
- It **froze** when the lane went idle and **rose** on a fresh short request:
  neither context length nor engine age. Operating load.

The payback formula assumes a restart's improvement is permanent. The measured
"drift" was transient, so acting on it pays the 49 s bringup and re-drifts
within minutes. The premise underneath the whole advisory — *step time ratchets
up with engine age* — had zero observations in a full day of watching. Every
excursion was load.

## Why the failures ended in simplification

Each failure removed an interpretation layer rather than adding a mechanism:

- The raw min said "the fastest bracket is the truth." It wasn't; the window
  mean replaced it — one statistic instead of a hoped-for exact instrument.
- The phase gate tried to *classify* steps away. Measurement said the
  confounder is unclassifiable from phase; the rejection gate just compares
  magnitudes — simpler, and it works because the populations are far apart.
- The advisory tried to *decide* for the operator. The data said the signal it
  was deciding on is load noise, and no persistence gate or wider band can fix
  a premise that does not exist. So the verdict was deleted, not tuned. The
  honest artifact is two numbers and a guide section that says what the gap
  between them is not.

The temptation at each rung was to add: another gate, another smoothing window,
another confidence band. Every time, the cheaper move — measure the premise
first — was available and was skipped. The day's cost is mostly the price of
fixes built before their premises were probed. The surviving code is the part
that was probed: the estimator (sound, still live) and the raw fields.

## Method notes worth keeping

- **A check that cannot fail is not a check.** The contract test passed at the
  same check count after a new field was added — it had silently not covered
  it. Proving coverage means deleting the emit line and watching the test name
  it. Six vacuous checks were caught this way in one day.
- **Like-for-like means same statistic, same instrument, same operating
  condition.** Two of the three failures were comparisons between different
  instruments wearing the same units.
- **Score predictions on representative encounters.** The advisory was scored
  "working" on one lucky sample at uptime 141 s; the next sample was the
  normal behavior. One claim per prediction, or the score means nothing.
- **Comments are load-bearing.** A stale rationale ("the page needs this to
  say whether a restart pays back") survived a green suite after the code it
  described was deleted; it would have re-taught the phantom to the next
  reader. Sweep added code lines per patch for vocabulary the final tree no
  longer supports.

## Open question

Whether engine age degrades step time at all is still unmeasured — every
excursion seen was load. The discriminating experiment: restart under steady
sustained load and watch whether `recent` re-lands at the same value within
minutes. If it does, no restart advisory is ever defensible and these two
numbers stay un-advised forever. If a floor genuinely ratchets up and stays
up, that signature gets a threshold, and a verdict can come back — designed
from the observation this time, not from the vocabulary of the design.
