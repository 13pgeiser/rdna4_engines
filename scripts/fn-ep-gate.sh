#!/usr/bin/env bash
# fn-ep-gate.sh: Flash-Next (Qwen4, mq4-xts) expert-parallel gate for a host with N homogeneous
# gfx1100 (7900 XTX) or gfx1201 (R9700) cards. Run it from a hipfire checkout; it builds the
# binaries, runs ONE card (topology.ep = 1) and N cards (topology.ep = N) over the same prompts,
# compares them byte for byte, times decode/prefill, and writes a receipt you can send back.
#
#   scripts/fn-ep-gate.sh [options] MODEL.mq4 [N] [OUT_DIR]
#
#   MODEL.mq4    Flash-Next mq4-xts model file (REQUIRED)
#   N            EP degree, 2..=8 (default: length of --devices, else every visible card of the chosen arch, at most 8)
#   OUT_DIR      output directory (default: ./fn-ep-gate-<UTC timestamp>)
#
# Options (also accepted as --n N / --out DIR before or after the positionals):
#   --arch ARCH         card arch to use (default: gfx1100 or gfx1201, whichever has more cards)
#   --devices LIST      comma list of cards in any hardware.devices form (PCI-order index, GPU-<uuid>, PCI
#                       address, gfxNNNN). HIPFIRE_DEVICES for the EP arm is LIST verbatim, the single arm
#                       uses its first entry, N defaults to its length. Default: an inherited HIPFIRE_DEVICES
#                       (e.g. a scheduler's held cards) when set, else the first N cards of --arch from KFD.
#                       ROCR_VISIBLE_DEVICES is cleared in the arms only for the KFD-selected default.
#   --tokens T          greedy tokens per prompt (default 128)
#   --max-seq S         KV context (default 8192)
#   --state-steps L     state digest steps (default 1,64,128)
#   --reps R            fresh processes per speed cell and arm (default 3)
#   --skip-build        reuse existing binaries (cargo target dir: $CARGO_TARGET_DIR or ./target)
#   --skip-speed        exactness only (no `hipfire bench` cells)
#   --skip-model-sha    do not hash the model file (the receipt then has no model sha256)
#   --ref-ranks R       reference arm EP degree, 1 <= R < N (default 1 = one card). Use it when one card
#                       would need more pinned host RAM than you have (a 24 GB card pins ~50 GiB of
#                       experts; EP4 on 24 GB cards keeps every expert layer in VRAM). The reference then
#                       runs EP-R on the first R cards (HIPFIRE_DEVICES = first R entries of the card
#                       list) as arm `ref-epR`, and the EP-N arm is compared against it byte for byte.
#   --no-isolate        keep your own HOME / HIPFIRE_HOME (default: a private one per arm under OUT_DIR
#                       holding only `topology.ep`, so your config cannot change the arms)
#   --target-dir DIR    cargo target dir
#   -h, --help          this text
#
# Environment: EXTRA_ENV="K=V K2=V2" adds variables to every arm; FN_EP_GATE_ARM_TIMEOUT (7200 s,
# oracle arm) and FN_EP_GATE_BENCH_TIMEOUT (3600 s, one bench process) bound each run.
#
# Steps: (a) environment receipt (git/build hash, ROCm, rocm-smi topology + P2P access, per-card
# arch/VRAM/PCIe link speed+width from sysfs, host RAM); (b) cargo build of fn_ep_oracle
# (hipfire-loader, feature `lab`), hipfire, daemon; (c) arms `single` (HIPFIRE_DEVICES = first card,
# HIPFIRE_TOPOLOGY_EP=1) and `epN` (first N cards, HIPFIRE_TOPOLOGY_EP=N) through fn_ep_oracle with
# the two prompts in scripts/fn-ep-gate-prompts (~500 and ~4,400 tokens), 128 greedy tokens, state
# digests at steps 1,64,128 (with --ref-ranks R > 1 the first arm is `ref-epR`: the first R cards,
# HIPFIRE_TOPOLOGY_EP=R, and every `single` below reads `ref-epR`); (d) scripts/fn_ep_compare.py
# (tokens, rank-0 logits sha, every peer logits mirror sha, every rank's state digests: all must
# equal the single card byte for byte; an EP-R reference is also checked against itself);
# (e) speed cells, `hipfire bench --spec off` decode (500-token prompt, 128 tokens) and prefill
# (4,400-token prompt, 1 token), REPS fresh processes per arm, A/B interleaved, raw JSON kept;
# (f) OUT_DIR/receipt.json and OUT_DIR/summary.txt.
#
# A failed EP load (missing peer access, out of VRAM, ...) is recorded in the receipt with the
# stderr tail; the script never exits without writing receipt.json. Exit status: 0 = byte-exact PASS,
# 1 = mismatch or an arm failed, 2 = environment / usage problem (not enough cards, build failed).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PROMPT_DIR=$SCRIPT_DIR/fn-ep-gate-prompts
COMPARE=$SCRIPT_DIR/fn_ep_compare.py
SHORT_PROMPT=$PROMPT_DIR/short-500.txt
LONG_PROMPT=$PROMPT_DIR/long-4400.txt
LONG_PROMPT_TOKENS=4400

MODEL="" N="" OUT="" ARCH=""
REF_RANKS=1
TOKENS=128 MAX_SEQ=8192 STATE_STEPS="1,64,128" REPS=3
SKIP_BUILD=0 SKIP_SPEED=0 SKIP_SHA=0 NO_ISOLATE=0
ARM_TIMEOUT=${FN_EP_GATE_ARM_TIMEOUT:-7200}
BENCH_TIMEOUT=${FN_EP_GATE_BENCH_TIMEOUT:-3600}
EXTRA_ENV=${EXTRA_ENV:-}
DEVLIST="" DEVS_SOURCE=kfd

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "fn-ep-gate: $*" >&2; exit 2; }
log() { echo "[fn-ep-gate $(date -u +%H:%M:%S)] $*"; }

POS=()
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --n|-n) [[ $# -ge 2 ]] || die "$1 needs a value"; N=$2; shift 2 ;;
    --out) [[ $# -ge 2 ]] || die "$1 needs a value"; OUT=$2; shift 2 ;;
    --arch) [[ $# -ge 2 ]] || die "$1 needs a value"; ARCH=$2; shift 2 ;;
    --devices) [[ $# -ge 2 ]] || die "$1 needs a value"; DEVLIST=${2// /}; shift 2 ;;
    --tokens) [[ $# -ge 2 ]] || die "$1 needs a value"; TOKENS=$2; shift 2 ;;
    --max-seq) [[ $# -ge 2 ]] || die "$1 needs a value"; MAX_SEQ=$2; shift 2 ;;
    --state-steps) [[ $# -ge 2 ]] || die "$1 needs a value"; STATE_STEPS=$2; shift 2 ;;
    --reps) [[ $# -ge 2 ]] || die "$1 needs a value"; REPS=$2; shift 2 ;;
    --target-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; export CARGO_TARGET_DIR=$2; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-speed) SKIP_SPEED=1; shift ;;
    --skip-model-sha) SKIP_SHA=1; shift ;;
    --ref-ranks) [[ $# -ge 2 ]] || die "$1 needs a value"; REF_RANKS=$2; shift 2 ;;
    --no-isolate) NO_ISOLATE=1; shift ;;
    --) shift; POS+=("$@"); break ;;
    -*) die "unknown option $1 (try --help)" ;;
    *) POS+=("$1"); shift ;;
  esac
done
((${#POS[@]} <= 3)) || die "too many positional arguments"
[[ -z ${POS[0]:-} ]] || MODEL=${POS[0]}
[[ -z ${POS[1]:-} ]] || N=${POS[1]}
[[ -z ${POS[2]:-} ]] || OUT=${POS[2]}
[[ -n $MODEL ]] || { usage >&2; die "MODEL.mq4 is required"; }
[[ -f $MODEL && -r $MODEL ]] || die "model file not readable: $MODEL"
[[ -z $N || $N =~ ^[0-9]+$ ]] || die "N must be an integer"
if [[ -z $DEVLIST && -n ${HIPFIRE_DEVICES:-} ]]; then
  DEVLIST=${HIPFIRE_DEVICES// /}
  DEVS_SOURCE=inherited
  echo "fn-ep-gate: using inherited HIPFIRE_DEVICES=$DEVLIST as the card list (ROCR_VISIBLE_DEVICES left as inherited)"
elif [[ -n $DEVLIST ]]; then
  DEVS_SOURCE=flag
fi
[[ $TOKENS =~ ^[0-9]+$ && $MAX_SEQ =~ ^[0-9]+$ && $REPS =~ ^[0-9]+$ ]] || die "--tokens/--max-seq/--reps must be integers"
[[ $REF_RANKS =~ ^[0-9]+$ ]] || die "--ref-ranks must be an integer"
[[ $REF_RANKS =~ ^0*([1-7])$ ]] || die "--ref-ranks must be in 1..7 and less than N"
REF_RANKS=${BASH_REMATCH[1]}
((REPS >= 1)) || die "--reps must be >= 1"
[[ -f $SHORT_PROMPT && -f $LONG_PROMPT && -f $COMPARE ]] || die "scripts/fn-ep-gate-prompts or fn_ep_compare.py missing next to this script"
command -v python3 >/dev/null || die "python3 is required"
command -v cargo >/dev/null || [[ $SKIP_BUILD == 1 ]] || die "cargo is required (or pass --skip-build)"

MODEL=$(readlink -f "$MODEL")
[[ -n $OUT ]] || OUT=./fn-ep-gate-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
mkdir -p "$OUT/env" "$OUT/oracle" "$OUT/speed"
exec > >(tee -a "$OUT/gate.log") 2>&1

TARGET_DIR=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
ORACLE=$TARGET_DIR/release/examples/fn_ep_oracle
HIPFIRE=$TARGET_DIR/release/hipfire
DAEMON=$TARGET_DIR/release/daemon

DEVS="" BDFS=() SINGLE_DEVS="" CARD_IDX=""
REF_ARM=single REF_DEVS=""
SAMPLER=""
VERDICT="" REASON=""
BUILD_RC="" ORACLE_RC_SINGLE="" ORACLE_RC_EP="" COMPARE_STATE="not run"
FINAL_RC=""
STARTED=$(date -u +%FT%TZ)
RECEIPT_DONE=0

# ------------------------------------------------------------------ receipt
write_receipt() {
  ((RECEIPT_DONE == 0)) || return 0
  RECEIPT_DONE=1
  export GATE_OUT=$OUT GATE_N=$N GATE_ARCH=$ARCH GATE_DEVS=$DEVS GATE_CARD_IDX=$CARD_IDX GATE_DEVS_SOURCE=$DEVS_SOURCE GATE_MODEL=$MODEL GATE_VERDICT=$VERDICT \
    GATE_REASON=$REASON GATE_STARTED=$STARTED GATE_REPS=$REPS GATE_TOKENS=$TOKENS GATE_MAX_SEQ=$MAX_SEQ \
    GATE_STATE_STEPS=$STATE_STEPS GATE_BUILD_RC=$BUILD_RC GATE_COMPARE_STATE=$COMPARE_STATE \
    GATE_SKIP_SPEED=$SKIP_SPEED GATE_NO_ISOLATE=$NO_ISOLATE GATE_EXTRA_ENV=$EXTRA_ENV \
    GATE_LONG_TOKENS=$LONG_PROMPT_TOKENS GATE_REF_RANKS=$REF_RANKS GATE_REF_ARM=$REF_ARM
  python3 - <<'PY' || echo "fn-ep-gate: WARNING: receipt writer failed" >&2
import json, os, re, statistics, datetime

E = os.environ
out = E["GATE_OUT"]
n = int(E["GATE_N"]) if E.get("GATE_N", "").isdigit() else None
ref_ranks = int(E["GATE_REF_RANKS"]) if E.get("GATE_REF_RANKS", "").isdigit() else 1
ref_arm = E.get("GATE_REF_ARM") or "single"


def rd(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def rdj(path):
    text = rd(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        start = text.find("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except ValueError:
                return None
        return None


def tail(path, count=40):
    text = rd(path)
    return None if text is None else "\n".join(text.splitlines()[-count:])


def seconds(rec):
    try:
        a = datetime.datetime.strptime(rd(rec + "/start.txt").strip(), "%Y-%m-%dT%H:%M:%SZ")
        b = datetime.datetime.strptime(rd(rec + "/end.txt").strip(), "%Y-%m-%dT%H:%M:%SZ")
        return int((b - a).total_seconds())
    except Exception:
        return None


def rc_of(path):
    text = rd(path)
    try:
        return int(text.strip())
    except Exception:
        return None


FAILURE_PATTERNS = [
    ("peer_access", r"peer access|peer-access|canAccessPeer|EnablePeerAccess"),
    ("out_of_memory", r"out of memory|hipErrorOutOfMemory|\bOOM\b|insufficient (?:vram|memory)|does not fit"),
    ("arch_refused", r"requires gfx\d+|not supported on gfx\d+"),
    ("devices", r"HIPFIRE_DEVICES|hardware\.devices|visible devices"),
]


def classify(rc, err):
    if rc == 124:
        return "timeout"
    for name, pattern in FAILURE_PATTERNS:
        if err and re.search(pattern, err, re.I | re.M):
            return name
    return "error"


def peaks(path, bdf_count):
    text = rd(path)
    if not text:
        return None
    best = None
    for line in text.splitlines():
        parts = line.split()[1:]
        if not parts:
            continue
        vals = []
        for p in parts:
            try:
                vals.append(int(p))
            except ValueError:
                vals.append(0)
        best = vals if best is None else [max(a, b) for a, b in zip(best, vals)]
    return None if best is None else [round(v / (1 << 20)) for v in best]


cards = (rdj(out + "/env/cards.json") or {}).get("cards", [])
devs = [int(x) for x in E.get("GATE_CARD_IDX", "").split(",") if x.strip().isdigit()]
by_index = {c["index"]: c for c in cards}
chosen = [by_index[i] for i in devs if i in by_index]

env = {
    "host": rd(out + "/env/host.txt"),
    "cpu": rd(out + "/env/cpu.txt"),
    "meminfo": rd(out + "/env/meminfo.txt"),
    "cmdline": rd(out + "/env/cmdline.txt"),
    "git": rd(out + "/env/git.txt"),
    "hipfire_version": rd(out + "/env/hipfire-version.txt"),
    "rocm_version": rd(out + "/env/rocm-version.txt"),
    "hipcc_version": rd(out + "/env/hipcc-version.txt"),
    "driver": rd(out + "/env/driver.txt"),
    "rocm_smi_topology_and_access": rd(out + "/env/rocm-smi-topo.txt"),
    "inherited_env": rd(out + "/env/inherited-env.txt"),
    "model": rd(out + "/env/model.txt"),
    "cards_all": cards,
}
link_table = [
    {
        "rank": rank,
        "index": c["index"],
        "bdf": c["bdf"],
        "arch": c["arch"],
        "vram_total_mib": (c.get("vram_total_bytes") or 0) // (1 << 20),
        "link_speed": c.get("current_link_speed"),
        "link_width": c.get("current_link_width"),
        "max_link_speed": c.get("max_link_speed"),
        "max_link_width": c.get("max_link_width"),
        "pci_chain": c.get("pci_chain"),
    }
    for rank, c in enumerate(chosen)
]
p2p = None
if chosen:
    p2p = [
        [
            (next((l for l in a.get("p2p_links", []) if l.get("to_node") == b["node"]), None) if a is not b else "self")
            for b in chosen
        ]
        for a in chosen
    ]

arms = {}
for arm in (ref_arm, "ep%d" % n if n else "ep"):
    rec = "%s/oracle/%s-receipt" % (out, arm)
    if not os.path.isdir(rec):
        continue
    rc = rc_of(rec + "/rc.txt")
    err = tail(rec + "/stderr.txt", 40)
    meta = rdj("%s/oracle/%s/meta.json" % (out, arm))
    arms[arm] = {
        "rc": rc,
        "seconds": seconds(rec),
        "command": rd(rec + "/command.txt"),
        "completed": bool(meta and meta.get("completed")),
        "ranks": meta and meta.get("ranks"),
        "devices": meta and meta.get("devices"),
        "prompts": meta and meta.get("prompts"),
        "vram_peak_mib_per_card": peaks(rec + "/vram.txt", None),
        "failure": None if rc == 0 else {"kind": classify(rc, err), "stderr_tail": err},
    }

compare = {"state": E.get("GATE_COMPARE_STATE")}
cj = rdj(out + "/compare.json")
if cj:
    compare.update({"verdict": cj.get("verdict"), "summary": cj.get("summary"), "first_mismatch": cj.get("first_mismatch"),
                    "families": cj.get("families"), "json": "compare.json", "text": "compare.txt"})

speed = {}
sdir = out + "/speed"
if os.path.isdir(sdir):
    for arm in sorted(os.listdir(sdir)):
        cells = {}
        for cell in ("decode", "prefill"):
            runs = []
            for rep in range(1, int(E.get("GATE_REPS", "3")) + 1):
                base = "%s/%s/%s-%d" % (sdir, arm, cell, rep)
                if not os.path.exists(base + ".rc"):
                    continue
                rc = rc_of(base + ".rc")
                doc = rdj(base + ".json")
                samples = (doc or {}).get("samples", {})
                runs.append({
                    "rep": rep,
                    "rc": rc,
                    "decode_tok_s": (samples.get("decode") or [None])[0],
                    "prefill_tok_s": (samples.get("prefill") or [None])[0],
                    "prompt_tokens": (doc or {}).get("prompt_tokens"),
                    "raw": "speed/%s/%s-%d.json" % (arm, cell, rep),
                    "stderr_tail": None if rc == 0 else tail(base + ".stderr", 25),
                })
            key = "decode_tok_s" if cell == "decode" else "prefill_tok_s"
            vals = [r[key] for r in runs if r["rc"] == 0 and r[key] is not None]
            cells[cell] = {"runs": runs, "metric": key, "median": statistics.median(vals) if vals else None, "n_ok": len(vals)}
        speed[arm] = cells
        skip = rd("%s/%s/SKIPPED.txt" % (sdir, arm))
        if skip:
            speed[arm]["skipped"] = skip.strip()

receipt = {
    "schema": "fn-ep-gate-receipt-v1",
    "started": E.get("GATE_STARTED"),
    "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "verdict": E.get("GATE_VERDICT") or "ERROR",
    "reason": E.get("GATE_REASON") or None,
    "config": {
        "n": n, "ref_ranks": ref_ranks, "ref_arm": ref_arm, "arch": E.get("GATE_ARCH"), "hipfire_devices_ep": E.get("GATE_DEVS"),
        "devices_source": E.get("GATE_DEVS_SOURCE"),
        "model": E.get("GATE_MODEL"), "tokens": int(E.get("GATE_TOKENS", "0") or 0),
        "max_seq": int(E.get("GATE_MAX_SEQ", "0") or 0), "state_steps": E.get("GATE_STATE_STEPS"),
        "speed_reps": int(E.get("GATE_REPS", "0") or 0), "isolated_home": E.get("GATE_NO_ISOLATE") != "1",
        "extra_env": E.get("GATE_EXTRA_ENV") or None,
        "long_prompt_truncated_to": int(E.get("GATE_LONG_TOKENS", "0") or 0),
        "oracle_arms_env": "HIPFIRE_GRAPH=0 (EP is eager; %s pinned to the same schedule)" % ("single" if ref_ranks == 1 else "the reference"),
        "speed_arms_env": "default graph setting; EP arms run eager",
    },
    "env": env,
    "link_table": link_table,
    "p2p_links_kfd": p2p,
    "build_rc": int(E["GATE_BUILD_RC"]) if E.get("GATE_BUILD_RC", "").lstrip("-").isdigit() else None,
    "arms": arms,
    "compare": compare,
    "speed": speed,
}
with open(out + "/receipt.json", "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, indent=2)
    handle.write("\n")

# ------------------------------------------------------------- summary.txt
lines = []
lines.append("fn-ep-gate: %s%s" % (receipt["verdict"], " (%s)" % receipt["reason"] if receipt["reason"] else ""))
if ref_ranks == 1:
    lines.append("arch=%s  N=%s  hipfire devices (single=first, EP=all): %s" % (E.get("GATE_ARCH"), n, E.get("GATE_DEVS")))
else:
    lines.append("arch=%s  N=%s  hipfire devices (reference=first %d, EP=all): %s" % (E.get("GATE_ARCH"), n, ref_ranks, E.get("GATE_DEVS")))
    lines.append("reference: EP%d on the first %d cards (one card = --ref-ranks 1)" % (ref_ranks, ref_ranks))
lines.append("ref_ranks=%d" % ref_ranks)
lines.append("Note: EP-N ≡ one card is established separately by our gates.")
lines.append("model: %s" % E.get("GATE_MODEL"))
git = (env["git"] or "").strip().splitlines()
lines.append("build: %s" % (git[0] if git else "unknown"))
_prompts = (arms.get(ref_arm) or {}).get("prompts") or []
if _prompts:
    lines.append("prompts (tokens used / full): " + ", ".join(
        "%s/%s" % (p.get("token_count"), p.get("full_token_count")) for p in _prompts))
lines.append("")
lines.append("rank idx bdf          arch    vram_MiB  link (cur / max)")
for row in link_table:
    lines.append("%-4d %-3d %-12s %-7s %-9d %s x%s / %s x%s" % (
        row["rank"], row["index"], row["bdf"], row["arch"], row["vram_total_mib"],
        row["link_speed"], row["link_width"], row["max_link_speed"], row["max_link_width"]))
lines.append("")
for arm, info in arms.items():
    status = "ok" if info["rc"] == 0 else "FAILED rc=%s (%s)" % (info["rc"], (info["failure"] or {}).get("kind"))
    lines.append("arm %-7s %s  %ss  vram peak MiB/card: %s" % (arm, status, info["seconds"], info["vram_peak_mib_per_card"]))
    if info["failure"]:
        lines.append("    stderr tail: " + "\n    ".join((info["failure"]["stderr_tail"] or "").splitlines()[-6:]))
lines.append("")
lines.append("compare: %s" % (compare.get("summary") or compare.get("state")))
if compare.get("first_mismatch"):
    lines.append("    first mismatch: %s" % compare["first_mismatch"].get("message"))
lines.append("")
if speed:
    lines.append("speed (median of fresh processes, tok/s)       decode   prefill(4,400-token prompt)")
    base_d = (speed.get(ref_arm, {}).get("decode") or {}).get("median")
    base_p = (speed.get(ref_arm, {}).get("prefill") or {}).get("median")
    for arm, cells in speed.items():
        d = (cells.get("decode") or {}).get("median")
        p = (cells.get("prefill") or {}).get("median")
        fmt = lambda v: "n/a" if v is None else "%.1f" % v
        rel = lambda v, b: "" if v is None or not b else " (%.2fx)" % (v / b)
        lines.append("  %-7s n_ok=%s/%s  %s%s   %s%s%s" % (
            arm, (cells.get("decode") or {}).get("n_ok"), (cells.get("prefill") or {}).get("n_ok"),
            fmt(d), rel(d, base_d) if arm != ref_arm else "", fmt(p), rel(p, base_p) if arm != ref_arm else "",
            "   [%s]" % cells["skipped"] if cells.get("skipped") else ""))
lines.append("")
lines.append("Send back: receipt.json, summary.txt, compare.txt, gate.log (and the oracle/*-receipt dirs on a failure).")
with open(out + "/summary.txt", "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY
}

on_exit() {
  local rc=$?
  set +e
  [[ -z $SAMPLER ]] || { kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; }
  if ((RECEIPT_DONE == 0)); then
    [[ -n $VERDICT ]] || { VERDICT=ERROR; [[ -n $REASON ]] || REASON="aborted (exit $rc)"; }
    write_receipt
  fi
  trap - EXIT
  exit "${FINAL_RC:-$rc}"
}
trap on_exit EXIT

# --------------------------------------------------------- (a) environment
collect_env() {
  local d=$OUT/env
  {
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "host=$(hostname 2>/dev/null || true)"
    echo "kernel=$(uname -srvm)"
    [[ ! -r /etc/os-release ]] || grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
  } > "$d/host.txt" 2>&1 || true
  grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo > "$d/meminfo.txt" 2>&1 || true
  { lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Socket|Thread|NUMA node\(s\))' || true; } > "$d/cpu.txt"
  cat /proc/cmdline > "$d/cmdline.txt" 2>&1 || true
  {
    git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "git: unknown"
    git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true
    echo "dirty_files=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l)"
  } > "$d/git.txt" 2>&1 || true
  local rocm="" p
  for p in "${ROCM_PATH:-}" /opt/rocm /opt/rocm-*; do
    [[ -n $p && -r $p/.info/version ]] && { rocm=$p; break; }
  done
  {
    echo "rocm_path=${rocm:-unknown}"
    [[ -z $rocm ]] || echo "rocm_version=$(cat "$rocm/.info/version")"
  } > "$d/rocm-version.txt" 2>&1 || true
  { hipcc --version 2>&1 || "${rocm:-/opt/rocm}/bin/hipcc" --version 2>&1 || echo "hipcc: not found"; } > "$d/hipcc-version.txt" 2>&1 || true
  {
    echo "amdgpu_module_version=$(cat /sys/module/amdgpu/version 2>/dev/null || echo unknown)"
    command -v rocm-smi >/dev/null && rocm-smi --showdriverversion 2>&1 || true
  } > "$d/driver.txt" 2>&1 || true
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showtopo --showtopoaccess > "$d/rocm-smi-topo.txt" 2>&1 || echo "rocm-smi --showtopo --showtopoaccess failed" >> "$d/rocm-smi-topo.txt"
  else
    echo "rocm-smi not found" > "$d/rocm-smi-topo.txt"
  fi
  { env | grep -E '^(HIP|HIPFIRE|ROCR|HSA|GPU_|AMD_|ROC_)' | sort || true; } > "$d/inherited-env.txt"

  python3 - "$d/cards.json" <<'PY'
import json, os, sys

KFD = "/sys/class/kfd/kfd/topology/nodes"


def rd(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def props(path):
    out = {}
    for line in (rd(path) or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]] = parts[1].strip()
    return out


def ival(d, key):
    try:
        return int(d.get(key, 0))
    except ValueError:
        return 0


cards = []
if os.path.isdir(KFD):
    for name in sorted((n for n in os.listdir(KFD) if n.isdigit()), key=int):
        p = props("%s/%s/properties" % (KFD, name))
        if ival(p, "simd_count") == 0:
            continue
        version, loc, dom = ival(p, "gfx_target_version"), ival(p, "location_id"), ival(p, "domain")
        bdf = "%04x:%02x:%02x.%x" % (dom, loc >> 8, (loc >> 3) & 0x1F, loc & 7)
        sysfs = "/sys/bus/pci/devices/" + bdf
        banks = []
        bank_dir = "%s/%s/mem_banks" % (KFD, name)
        if os.path.isdir(bank_dir):
            for bank in sorted(os.listdir(bank_dir)):
                bp = props("%s/%s/properties" % (bank_dir, bank))
                banks.append({"heap_type": ival(bp, "heap_type"), "size_in_bytes": ival(bp, "size_in_bytes")})
        links = []
        for kind in ("p2p_links", "io_links"):
            ldir = "%s/%s/%s" % (KFD, name, kind)
            if os.path.isdir(ldir):
                for link in sorted(os.listdir(ldir)):
                    lp = props("%s/%s/properties" % (ldir, link))
                    if lp:
                        links.append({"kind": kind, "to_node": ival(lp, "node_to"), "type": ival(lp, "type"),
                                      "weight": ival(lp, "weight"), "flags": ival(lp, "flags")})
        chain = [seg for seg in os.path.realpath(sysfs).split("/") if len(seg) == 12 and seg[4] == ":"]
        vram_total = rd(sysfs + "/mem_info_vram_total")
        cards.append({
            "node": int(name),
            "bdf": bdf,
            "arch": "gfx%d%x%x" % (version // 10000, (version // 100) % 100, version % 100),
            "gfx_target_version": version,
            "kfd_name": rd("%s/%s/name" % (KFD, name)),
            "unique_id": p.get("unique_id"),
            "vendor_id": ival(p, "vendor_id"), "device_id": ival(p, "device_id"),
            "kfd_mem_banks": banks,
            "vram_total_bytes": int(vram_total) if vram_total and vram_total.isdigit() else
                max([b["size_in_bytes"] for b in banks] or [0]),
            "current_link_speed": rd(sysfs + "/current_link_speed"),
            "current_link_width": rd(sysfs + "/current_link_width"),
            "max_link_speed": rd(sysfs + "/max_link_speed"),
            "max_link_width": rd(sysfs + "/max_link_width"),
            "pci_chain": chain,
            "p2p_links": [l for l in links if l["kind"] == "p2p_links"],
            "io_links": [l for l in links if l["kind"] == "io_links"],
        })
cards.sort(key=lambda c: c["bdf"])
for index, card in enumerate(cards):
    card["index"] = index
json.dump({"kfd_present": os.path.isdir(KFD), "cards": cards}, open(sys.argv[1], "w"), indent=2)
print("fn-ep-gate: %d GPU node(s): %s" % (len(cards), ", ".join("%d:%s@%s" % (c["index"], c["arch"], c["bdf"]) for c in cards)))
PY
}

select_cards() {
  local sel
  sel=$(python3 - "$OUT/env/cards.json" "$ARCH" "$N" "$DEVLIST" <<'PY'
import collections, json, sys

doc = json.load(open(sys.argv[1]))
cards, want, n, devlist = doc["cards"], sys.argv[2], sys.argv[3], sys.argv[4]
SUPPORTED = ("gfx1100", "gfx1201")


def fail(message):
    print("fn-ep-gate: " + message, file=sys.stderr)
    sys.exit(2)


def table():
    return "visible cards: " + "; ".join("%d:%s@%s uuid=%s" % (c["index"], c["arch"], c["bdf"],
        ("GPU-%016x" % int(c["unique_id"])) if str(c.get("unique_id") or "0").isdigit() and int(c.get("unique_id") or 0) else "none")
        for c in cards)


def parse_bdf(text):
    # DDDD:BB:DD.F or BB:DD.F, hex; returns (domain or None, bus, dev, fn)
    try:
        head, fn = text.rsplit(".", 1)
        parts = head.split(":")
        if len(parts) == 3:
            return (int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), int(fn, 16))
        if len(parts) == 2:
            return (None, int(parts[0], 16), int(parts[1], 16), int(fn, 16))
    except ValueError:
        pass
    return None


def card_bdf(c):
    d, b, rest = c["bdf"].split(":")
    dev, fn = rest.split(".")
    return (int(d, 16), int(b, 16), int(dev, 16), int(fn, 16))


if not doc["kfd_present"] or not cards:
    fail("no GPU found in the KFD topology (/sys/class/kfd/kfd/topology/nodes)")

if devlist:
    # Resolve every entry of a hardware.devices-style list to a KFD card, keeping list order.
    entries = [e.strip() for e in devlist.split(",")]
    chosen, deferred = [None] * len(entries), []
    for slot, entry in enumerate(entries):
        low = entry.lower()
        if not entry:
            fail("empty entry in --devices %r" % devlist)
        if entry.isdigit():
            hits = [c for c in cards if c["index"] == int(entry)]
        elif low.startswith("gpu-"):
            try:
                uid = int(low[4:], 16)
            except ValueError:
                fail("%r is not a GPU-<hex> uuid" % entry)
            hits = [c for c in cards if str(c.get("unique_id") or "0").isdigit() and int(c.get("unique_id") or 0) == uid and uid]
        elif low.startswith("gfx"):
            deferred.append(slot)
            continue
        elif ":" in entry:
            bdf = parse_bdf(entry)
            if bdf is None:
                fail("%r is not a DDDD:BB:DD.F or BB:DD.F PCI address" % entry)
            hits = [c for c in cards if card_bdf(c)[1:] == bdf[1:] and (bdf[0] is None or card_bdf(c)[0] == bdf[0])]
        else:
            fail("unknown device selector %r (index, GPU-uuid, PCI address or gfxNNNN); %s" % (entry, table()))
        if len(hits) != 1:
            fail("--devices entry %r matches %d card(s); %s" % (entry, len(hits), table()))
        chosen[slot] = hits[0]
    for slot in deferred:
        taken = {c["index"] for c in chosen if c}
        hits = [c for c in cards if c["arch"] == entries[slot].lower() and c["index"] not in taken]
        if not hits:
            fail("--devices entry %r: no unused %s card; %s" % (entries[slot], entries[slot], table()))
        chosen[slot] = hits[0]
    if len({c["index"] for c in chosen}) != len(chosen):
        fail("--devices names the same card more than once; %s" % table())
    count = len(chosen)
    if n and int(n) != count:
        fail("N=%s but --devices lists %d card(s)" % (n, count))
    archs = sorted({c["arch"] for c in chosen})
    if len(archs) != 1:
        print("fn-ep-gate: WARNING: --devices mixes archs %s (EP needs a homogeneous set)" % archs, file=sys.stderr)
    arch = chosen[0]["arch"]
    if want and any(c["arch"] != want for c in chosen):
        fail("--arch %s but --devices selects %s" % (want, archs))
    if not 2 <= count <= 8:
        fail("N=%d is outside 2..=8" % count)
else:
    if want:
        arch = want
    else:
        counts = collections.Counter(c["arch"] for c in cards)
        supported = [a for a in counts if a in SUPPORTED]
        if supported:
            arch = max(supported, key=lambda a: (counts[a], -min(c["index"] for c in cards if c["arch"] == a)))
        else:
            arch = cards[0]["arch"]
            print("fn-ep-gate: WARNING: no gfx1100/gfx1201 card; trying %s (EP admission will likely refuse it)" % arch,
                  file=sys.stderr)
    same = [c for c in cards if c["arch"] == arch]
    if not same:
        fail("no %s card found (have: %s)" % (arch, sorted(set(c["arch"] for c in cards))))
    count = int(n) if n else min(len(same), 8)
    if not 2 <= count <= 8:
        fail("N=%d is outside 2..=8 (%d %s card(s) visible)" % (count, len(same), arch))
    if count > len(same):
        fail("N=%d but only %d %s card(s) are visible" % (count, len(same), arch))
    chosen = same[:count]
print(arch, count, ",".join(c["bdf"] for c in chosen), ",".join(str(c["index"]) for c in chosen))
PY
) || { REASON="card selection failed (see gate.log)"; VERDICT=ERROR; FINAL_RC=2; exit 2; }
  local bdfs
  read -r ARCH N bdfs CARD_IDX <<<"$sel"
  IFS=, read -r -a BDFS <<<"$bdfs"
  [[ -n $DEVLIST ]] && DEVS=$DEVLIST || DEVS=$CARD_IDX
  SINGLE_DEVS=${DEVS%%,*}
  log "arch=$ARCH N=$N devices(HIPFIRE_DEVICES)=$DEVS (source: $DEVS_SOURCE) single=$SINGLE_DEVS cards(PCI-order index)=$CARD_IDX bdfs=$bdfs"
  if ((REF_RANKS < 1 || REF_RANKS >= N)); then
    REASON="--ref-ranks $REF_RANKS is outside 1..$((N - 1)) for N=$N"; VERDICT=ERROR; FINAL_RC=2
    die "$REASON"
  fi
  if ((REF_RANKS == 1)); then
    REF_ARM=single REF_DEVS=$SINGLE_DEVS
  else
    local _devs
    IFS=, read -r -a _devs <<<"$DEVS"
    REF_ARM=ref-ep$REF_RANKS
    REF_DEVS=$(IFS=,; echo "${_devs[*]:0:REF_RANKS}")
    log "reference arm $REF_ARM: ep=$REF_RANKS devices=$REF_DEVS"
  fi
}

hash_model() {
  {
    echo "path=$MODEL"
    echo "size_bytes=$(stat -c %s "$MODEL")"
    if ((SKIP_SHA)); then
      echo "sha256=skipped"
    else
      log "hashing model (use --skip-model-sha to skip)"
      echo "sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    fi
  } > "$OUT/env/model.txt"
}

# ------------------------------------------------------------------ (b) build
build_all() {
  if ((SKIP_BUILD)); then
    log "build skipped"
    BUILD_RC=0
  else
    log "building fn_ep_oracle, hipfire, daemon (log: $OUT/build.log)"
    BUILD_RC=0
    (
      cd "$REPO_ROOT"
      cargo build --release -p hipfire-loader --example fn_ep_oracle --features lab \
        && cargo build --release -p hipfire-cli -p hipfire-daemon
    ) > "$OUT/build.log" 2>&1 || BUILD_RC=$?
  fi
  if [[ $BUILD_RC != 0 ]]; then
    REASON="cargo build failed (rc=$BUILD_RC); see build.log"; VERDICT=ERROR; FINAL_RC=2
    tail -n 20 "$OUT/build.log" || true
    exit 2
  fi
  local missing=""
  for bin in "$ORACLE" "$HIPFIRE" "$DAEMON"; do [[ -x $bin ]] || missing+=" $bin"; done
  [[ -z $missing ]] || { REASON="missing binaries:$missing"; VERDICT=ERROR; FINAL_RC=2; die "$REASON"; }
  { "$HIPFIRE" version 2>&1 | sed -n '1,25p'; } > "$OUT/env/hipfire-version.txt" 2>&1 || true
}

# --------------------------------------------------------------- arm helpers
ARM_ENV=()
arm_env() {
  local arm=$1 ep=$2 devs=$3
  # ROCR_VISIBLE_DEVICES: dropped only when the card list came from our own KFD selection; a list the
  # caller supplied (--devices) or inherited (a scheduler's HIPFIRE_DEVICES) leaves ROCR_VISIBLE_DEVICES alone.
  local unset_rocr=()
  [[ $DEVS_SOURCE != kfd ]] || unset_rocr=(-u ROCR_VISIBLE_DEVICES)
  ARM_ENV=(env -u HIP_VISIBLE_DEVICES -u HIPFIRE_QWEN4_EP ${unset_rocr[@]+"${unset_rocr[@]}"}
    "HIPFIRE_DEVICES=$devs" "HIPFIRE_TOPOLOGY_EP=$ep" HIPFIRE_NO_REGISTRY_FETCH=1 "HIPFIRE_DAEMON_BIN=$DAEMON")
  if ((NO_ISOLATE == 0)); then
    local home=$OUT/home-$arm
    mkdir -p "$home/.hipfire"
    printf '[topology]\nep = %s\n' "$ep" > "$home/.hipfire/config.toml"
    ARM_ENV+=("HOME=$home" "HIPFIRE_HOME=$home/.hipfire")
  fi
  [[ -n ${HIPFIRE_KERNEL_CACHE:-} ]] || ARM_ENV+=("HIPFIRE_KERNEL_CACHE=$OUT/kcache")
  if [[ -n $EXTRA_ENV ]]; then
    # shellcheck disable=SC2206
    local extra=($EXTRA_ENV)
    ARM_ENV+=("${extra[@]}")
  fi
}

start_vram_sampler() {
  local file=$1 count=$2
  (
    while :; do
      line=$(date +%s)
      for ((i = 0; i < count; i++)); do
        line+=" $(cat "/sys/bus/pci/devices/${BDFS[$i]}/mem_info_vram_used" 2>/dev/null || echo 0)"
      done
      echo "$line"
      sleep 1
    done
  ) > "$file" 2>/dev/null &
  SAMPLER=$!
}

stop_vram_sampler() {
  [[ -z $SAMPLER ]] || { kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true; SAMPLER=""; }
}

run_oracle_arm() {
  local arm=$1 ep=$2 devs=$3 count=$4
  local rec=$OUT/oracle/$arm-receipt out=$OUT/oracle/$arm rc=0
  rm -rf "$rec" "$out"
  mkdir -p "$rec"
  arm_env "$arm" "$ep" "$devs"
  local cmd=("${ARM_ENV[@]}" HIPFIRE_GRAPH=0 timeout -k 30 "$ARM_TIMEOUT" "$ORACLE"
    --model "$MODEL" --max-seq "$MAX_SEQ" --out "$out" --arm "$arm"
    --prompt-file "$SHORT_PROMPT" --prompt-file "$LONG_PROMPT" --truncate-prompt-tokens "$LONG_PROMPT_TOKENS"
    --tokens "$TOKENS" --state-steps "$STATE_STEPS" --skip-model-hash)
  printf '%q ' "${cmd[@]}" > "$rec/command.txt"
  echo >> "$rec/command.txt"
  date -u +%FT%TZ > "$rec/start.txt"
  start_vram_sampler "$rec/vram.txt" "$count"
  log "oracle arm $arm: ep=$ep devices=$devs"
  "${cmd[@]}" > "$rec/stdout.txt" 2> "$rec/stderr.txt" || rc=$?
  stop_vram_sampler
  date -u +%FT%TZ > "$rec/end.txt"
  echo "$rc" > "$rec/rc.txt"
  log "oracle arm $arm: rc=$rc"
  if ((rc != 0)); then
    log "last stderr lines of arm $arm:"
    tail -n 8 "$rec/stderr.txt" | sed 's/^/    /' || true
  fi
  return 0
}

run_bench_cell() {
  local arm=$1 ep=$2 devs=$3 cell=$4 rep=$5 prompt=$6 max_tokens=$7
  local dir=$OUT/speed/$arm base rc=0
  mkdir -p "$dir"
  base=$dir/$cell-$rep
  arm_env "$arm" "$ep" "$devs"
  local cmd=("${ARM_ENV[@]}" timeout -k 30 "$BENCH_TIMEOUT" "$HIPFIRE" bench "$MODEL"
    --spec off --max-seq "$MAX_SEQ" --runs 1 --max-tokens "$max_tokens" --prompt-file "$prompt" -j)
  printf '%q ' "${cmd[@]}" > "$base.cmd"
  echo >> "$base.cmd"
  "${cmd[@]}" > "$base.json" 2> "$base.stderr" || rc=$?
  echo "$rc" > "$base.rc"
  log "bench $arm $cell rep $rep: rc=$rc"
  return 0
}

oracle_arm_ok() {
  local arm=$1
  [[ $(cat "$OUT/oracle/$arm-receipt/rc.txt" 2>/dev/null || echo x) == 0 ]] && [[ -f $OUT/oracle/$arm/meta.json ]]
}

# ------------------------------------------------------------------------ run
log "output directory: $OUT"
collect_env
select_cards
hash_model
build_all

EPARM=ep$N
run_oracle_arm "$REF_ARM" "$REF_RANKS" "$REF_DEVS" "$REF_RANKS"
run_oracle_arm "$EPARM" "$N" "$DEVS" "$N"

# (d) compare
if oracle_arm_ok "$REF_ARM" && oracle_arm_ok "$EPARM"; then
  crc=0
  python3 "$COMPARE" "$OUT/oracle/$REF_ARM" "$OUT/oracle/$EPARM" --expect-ranks "$N" --ref-ranks "$REF_RANKS" \
    --json "$OUT/compare.json" > "$OUT/compare.txt" 2>&1 || crc=$?
  sed 's/^/    /' "$OUT/compare.txt"
  if ((crc == 0)); then
    VERDICT=PASS; COMPARE_STATE="PASS"
  elif ((crc == 1)); then
    VERDICT=FAIL; COMPARE_STATE="FAIL"
    if ((REF_RANKS == 1)); then REASON="EP$N output differs from the single card"; else REASON="EP$N output differs from the EP$REF_RANKS reference"; fi
  else
    VERDICT=ERROR; COMPARE_STATE="comparator error rc=$crc"; REASON="fn_ep_compare.py failed to run"
  fi
else
  VERDICT=ERROR
  oracle_arm_ok "$REF_ARM" || COMPARE_STATE="$REF_ARM arm failed"
  oracle_arm_ok "$EPARM" || COMPARE_STATE="$EPARM arm failed (see oracle/$EPARM-receipt/stderr.txt)"
  REASON="$COMPARE_STATE; no comparison"
fi

# (e) speed cells, A/B interleaved per repetition
if ((SKIP_SPEED)); then
  log "speed cells skipped"
else
  EP_SPEED=1
  if ! oracle_arm_ok "$EPARM"; then
    EP_SPEED=0
    mkdir -p "$OUT/speed/$EPARM"
    echo "skipped: the $EPARM oracle arm failed to load/run" > "$OUT/speed/$EPARM/SKIPPED.txt"
  fi
  for ((rep = 1; rep <= REPS; rep++)); do
    run_bench_cell "$REF_ARM" "$REF_RANKS" "$REF_DEVS" decode "$rep" "$SHORT_PROMPT" 128
    ((EP_SPEED == 0)) || run_bench_cell "$EPARM" "$N" "$DEVS" decode "$rep" "$SHORT_PROMPT" 128
    run_bench_cell "$REF_ARM" "$REF_RANKS" "$REF_DEVS" prefill "$rep" "$LONG_PROMPT" 1
    ((EP_SPEED == 0)) || run_bench_cell "$EPARM" "$N" "$DEVS" prefill "$rep" "$LONG_PROMPT" 1
  done
fi

# (f) receipt
[[ -n $VERDICT ]] || VERDICT=ERROR
write_receipt
echo
cat "$OUT/summary.txt" || true
case $VERDICT in PASS) FINAL_RC=0 ;; FAIL) FINAL_RC=1 ;; *) FINAL_RC=1 ;; esac
exit "$FINAL_RC"
