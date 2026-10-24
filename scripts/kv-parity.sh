#!/bin/bash
# Prove that a --kv-resident ring that covers the group's context changes no output byte and
# streams nothing at all on the QSA model.
#
# Why this exists: docs/KV-STREAMING.md §5 Phase 4 makes a ring at or above the group's context a
# plain resident pool -- no host mirror, no residency map, no stage view, no kv_resolve and a null
# RadKVStream -- so the base engine's path runs exactly. This script runs one server per arm, sends
# the same request to each, and compares the completion text byte for byte. The "on" arm passes
# --kv-resident 4096 with --max-num-seqs 1 --max-model-len 4096, i.e. exactly the group's context,
# and must say so at startup and issue no resolve at all; the off arm passes no flag. Any overhead
# the covering configuration adds would show up here as either a different completion, a streaming
# startup line, or a nonzero kv_resolve count.
#
# It builds nothing, and it touches only the server it starts: the engine is one process (its TP
# ranks are threads), so each arm is recorded under its own log in $OUT and killed by that pid at the
# end or on any error. Nothing else that matches bin/radiance is signalled -- no pkill anywhere.
#
# Run this INSIDE the toolchain container, with the model mounted, because that is where the binary
# and its kernel libraries live, and because the engine needs io_uring, which Docker's default
# seccomp profile refuses:
#   docker run --rm --security-opt seccomp=unconfined --user 1000:1000 -e HOME=/tmp \
#     --device /dev/kfd --device /dev/dri --group-add 990 --group-add 44 \
#     -v /home/j/src/radiance-kv-cache-streaming:/src -v /home/j/Documents/rad/models:/models \
#     -v /tmp/opencode:/tmp/opencode -w /src rad-toolchain:latest bash -lc 'bash scripts/kv-parity.sh'
#
# Two prompts, both deterministic (a numbered record repeated, then "continue the register"): the
# sparse one is ~3000 tokens because the sparse path only engages above qsa_exact_to = 2051
# (arch/qwen4exp_fp8/qwen4exp_fp8.cpp:621-624) -- below it attention is dense and the selection is
# a different path, so a short prompt alone would prove nothing. Both fit the 4096-token context the
# ring covers. The short arm is there to show that path is untouched too.
#
# Exit status: 0 only if both arms produced a completion, the two are byte-identical, and the
# covering arm reported no streaming and no resolve; 1 on any difference or streaming overhead;
# 2 if an arm did not come up or did not answer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-/tmp/opencode/build-kv2}"
MODEL="${MODEL:-/models/qwen3.8-next-flash-fp8-iq4r-moe.rad}"
PORT="${PORT:-8123}"
OUT="${OUT:-/tmp/opencode/kv-parity}"
SPARSE_LINES="${SPARSE_LINES:-100}"   # ~3028 prompt tokens: above qsa_exact_to, so sparse
DENSE_LINES="${DENSE_LINES:-20}"      # ~600 prompt tokens: below it, so dense, no kv_resolve
MAXTOK="${MAXTOK:-256}"
SEED="${SEED:-12345}"
# Exactly the one-sequence context the server is configured with (--max-num-seqs 1 x
# --max-model-len 4096), so this ring covers the group and nothing streams.
RESIDENT="${RESIDENT:-4096}"

BIN="$BUILD/bin/radiance"
export RADIANCE_HOME="$BUILD/radiance_home"   # absolute: the engine will not find the kernels without it
[ -x "$BIN" ] || { echo "no engine at $BIN (build it first; this script builds nothing)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is needed" >&2; exit 2; }

mkdir -p "$OUT"
SERVER_PID=""
cleanup() {
  # Only our own process, by the pid this script recorded: the engine is one process (its TP ranks
  # are threads), so there is no group to sweep and nothing else can be hit by mistake. No pkill.
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # Generous: with --profile-ops the per-op table is printed by ~Engine on SIGTERM, and killing
    # before it lands would lose the only per-op issue counts the engine ever reports.
    for _ in $(seq 1 240); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

# A one-line prompt, so the request body needs no JSON string escapes at all: the records are
# joined with spaces deliberately (text is identical in both arms either way).
gen_prompt() {
  local n="$1" i first=$(( $1 + 1 )) last=$(( $1 + 10 ))
  for ((i = 1; i <= n; i++)); do
    printf 'Record %d: the archivist checked the shelf, noted the catalogue number, and returned the volume to the east wing of the reading room. ' "$i"
  done
  printf 'Continue the register: write records %d through %d in the same form, one per line.' "$first" "$last"
}
gen_prompt "$SPARSE_LINES" > "$OUT/prompt-sparse.txt"
gen_prompt "$DENSE_LINES"  > "$OUT/prompt-dense.txt"

# The raw JSON-escaped value of the completion's "content". Escaping is a deterministic function of
# the text, so equal escaped bytes mean equal text -- and comparing the escaped form is strictly
# safer than any unescaper this script could carry.
content_of() {
  awk '
    { s = $0
      i = index(s, "\"message\":"); if (i == 0) exit 1
      s = substr(s, i)
      i = index(s, "\"content\":\""); if (i == 0) exit 1
      s = substr(s, i + 11)
      out = ""
      for (j = 1; j <= length(s); j++) {
        c = substr(s, j, 1)
        if (c == "\\") { out = out substr(s, j, 2); j++; continue }
        if (c == "\"") break
        out = out c
      }
      printf "%s", out
    }' "$1"
}

# One arm: start a server, wait for /health (never a blind sleep), ask once, stop the server.
# `$1` is the arm's name, `$2` its prompt, the rest is the streaming flags -- identical otherwise,
# which is the point. Returns non-zero (and leaves the engine's last errors on stderr) if the arm
# could not serve, so every arm gets to report before the script decides.
run_arm() {  local arm="$1" prompt="$2"; shift 2
  local log="$OUT/$arm.log" req="$OUT/$arm.req.json" resp="$OUT/$arm.json"
  echo "== $arm: $* =="
  ( exec "$BIN" --model "$MODEL" \
      --tp 2 --tp-wire exact \
      --max-num-seqs 1 --max-model-len 4096 --placement expert_tiered --host-pool-mib 23652 \
      --kv-cache-dtype fp8 --prefix-cache-mib 4096 --vision-tower-placement ram \
      --num-speculative-tokens 0 --deterministic \
      "$@" --host 127.0.0.1 --port "$PORT" --api-key test ) > "$log" 2>&1 &
  SERVER_PID=$!

  local i code
  for i in $(seq 1 180); do   # 15 min at 5 s: a cold load of this container is under a minute
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FAIL $arm: the engine exited before serving; last errors:" >&2
      grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || tail -12 "$log" >&2
      cleanup
      return 1
    fi
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    [ "$code" = 200 ] && break
    sleep 5
  done
  if [ "${code:-}" != 200 ]; then
    echo "FAIL $arm: /health did not answer" >&2
    cleanup
    return 1
  fi

  # temperature 0 and a fixed seed: greedy, so the only thing that can move a token is the engine.
  local prompt_text
  prompt_text=$(cat "$prompt")
  printf '{"model":"%s","temperature":0,"seed":%s,"max_tokens":%s,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"%s"}]}' \
    "qwen4exp" "$SEED" "$MAXTOK" "$prompt_text" > "$req"
  code=$(curl -s -m 900 -o "$resp" -w '%{http_code}' \
    -H "Authorization: Bearer test" -H "Content-Type: application/json" \
    --data-binary "@$req" "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || true)
  if [ "${code:-}" != 200 ]; then
    echo "FAIL $arm: /v1/chat/completions answered $code; last errors:" >&2
    grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || tail -12 "$log" >&2
    cleanup
    return 1
  fi
  content_of "$resp" > "$OUT/$arm.txt"
  if [ ! -s "$OUT/$arm.txt" ]; then
    echo "FAIL $arm: empty completion text" >&2
    cleanup
    return 1
  fi

  # What the prompt actually cost, straight from the engine: it must be past qsa_exact_to for the
  # sparse arm, and cached_tokens 0 shows the whole prompt was prefilled on this server (a prefix
  # hit would leave the arm's own token count unproven).
  local ptok ctok cache
  ptok=$(grep -o '"prompt_tokens":[0-9]*' "$resp" | head -1 | cut -d: -f2)
  ctok=$(grep -o '"completion_tokens":[0-9]*' "$resp" | head -1 | cut -d: -f2)
  cache=$(grep -o '"cached_tokens":[0-9]*' "$resp" | head -1 | cut -d: -f2)
  echo "   prompt_tokens=$ptok completion_tokens=$ctok cached_tokens=$cache"
  echo "   sha256 $(sha256sum "$OUT/$arm.txt" | cut -d' ' -f1)  ($(wc -c < "$OUT/$arm.txt") bytes)"
  echo "$ptok $ctok $cache" > "$OUT/$arm.counts"

  cleanup
  return 0
}

# What a byte comparison cannot tell you, asserted on the streaming log:
#  - no `kv_resolve` refusal and no GPU fault / fatal.
#  - the "nothing streams" line, and no streamed-group line: the ring covered the context, so the
#    group kept the base engine's resident path rather than a cheapened streamed one.
assert_streaming_log() {
  local arm="$1"
  local log="$OUT/$1.log"
  local bad=0
  if grep -q "refused op 'kv_resolve'" "$log"; then
    echo "FAIL $arm: kv_resolve was refused at issue" >&2; bad=1
  fi
  if grep -q "GPU memory fault" "$log"; then
    echo "FAIL $arm: a GPU memory fault was reported" >&2; bad=1
  fi
  if grep -q "^radiance: fatal:" "$log"; then
    echo "FAIL $arm: the engine reported a fatal error" >&2; bad=1
  fi
  if ! grep -q "^I kv streaming: nothing streams" "$log"; then
    echo "FAIL $arm: no 'nothing streams' startup line: --kv-resident $RESIDENT did not cover the context" >&2; bad=1
  fi
  if grep -q "^I kv streaming: [0-9].*stream (" "$log"; then
    echo "FAIL $arm: a group streamed even though the ring covers its context" >&2; bad=1
  fi
  return $bad
}

# Count the kv_resolve issues the engine itself accounted for, per op instance. `--profile-ops`
# prints a per-op table (calls, total ms, us/call, op, geometry) on shutdown and every
# RADIANCE_PROFILE_EVERY steps, and it is the only place the engine states how many times an op was
# issued: no ctl counter reaches /metrics and no launch is logged. There is one kv_resolve op per
# layer, so every row carries the same count; the largest is this arm's per-layer total.
# A row is `  <calls> <total ms> <us/call> <site> <join> <wait> <op> <geometry> ...`, so the op is
# matched as a whole field and the count is field 1. Prints the count (0 if the op never appears).
resolves_issued() {
  grep -E '^ +[0-9]+ ' "$1" 2>/dev/null |
    awk '{ for (i = 1; i <= NF; i++) if ($i == "kv_resolve") { if ($1 + 0 > mx) mx = $1 + 0; break } }
         END { print mx + 0 }'
}

# Ask one profiled question: how many times was kv_resolve issued per layer on this shape? $1 = arm
# name, $2 = the prompt. The count lands in $OUT/$arm.resolves. The profiled servers are used for
# this evidence only -- their completions are not compared, because --profile-ops restores
# synchronisations that make it a different run (its own caveat prints so).
run_profile_probe() {
  local arm="$1" prompt="$2"
  local log="$OUT/$arm.log" got
  echo "== probe $arm: --profile-ops =="
  run_arm "$arm" "$prompt" --kv-resident "$RESIDENT" --profile-ops || return 1
  got=$(resolves_issued "$log")
  echo "   kv_resolve issues per layer: $got"
  echo "$got" > "$OUT/$arm.resolves"
  return 0
}

# The short arms first, then the arms the change is about: whatever the sparse arm
# does must not hide the evidence the cheap arms produce.
FAILED=0
run_arm off-dense  "$OUT/prompt-dense.txt"                    || FAILED=1
run_arm on-dense   "$OUT/prompt-dense.txt"  --kv-resident "$RESIDENT" || FAILED=1

assert_streaming_log on-dense || FAILED=1

# Phase 2 admits a ring smaller than the context: a prefill or mixed step takes the group's
# one-layer stage view and never resolves, so the union of §3.5 is not a ring requirement. The
# staged-copy evidence (staged copies, in layer order, and byte parity) is
# scripts/kv-stage-parity-64k.sh, which runs the floor ring against a ~64k document; this script
# keeps the context-sized covering configuration it was written for.

run_arm off-sparse "$OUT/prompt-sparse.txt"                    || FAILED=1
run_arm on-sparse  "$OUT/prompt-sparse.txt" --kv-resident "$RESIDENT" || FAILED=1
assert_streaming_log on-sparse || FAILED=1

echo "== kv streaming lines from on-sparse.log =="
grep -E "kv streaming:" "$OUT/on-sparse.log" || echo "  (none: no streaming report line at all)"

# Execution, not configuration: the byte comparison above is worthless if a resolve ran on a group
# this configuration says is resident. A ring that covers the context must issue no kv_resolve at
# all -- on either the dense or the sparse path -- because the batch hands the arch a null stream.
run_profile_probe proof-dense  "$OUT/prompt-dense.txt"  || FAILED=1
run_profile_probe proof-sparse "$OUT/prompt-sparse.txt" || FAILED=1
if [ -s "$OUT/proof-dense.resolves" ] && [ -s "$OUT/proof-sparse.resolves" ]; then
  dense_n=$(cat "$OUT/proof-dense.resolves")
  sparse_n=$(cat "$OUT/proof-sparse.resolves")
  echo "== resolve execution: $sparse_n per layer on the sparse arm, $dense_n on the dense arm =="
  if [ "$sparse_n" -ne 0 ] || [ "$dense_n" -ne 0 ]; then
    echo "FAIL: a covering ring issued $sparse_n/$dense_n resolves per layer: it streamed when it" >&2
    echo "      should have kept the base engine's resident path" >&2
    FAILED=1
  fi
else
  echo "FAIL: a probe arm produced no per-op table, so execution is unproven" >&2
  FAILED=1
fi

# The sparse prompt must be past qsa_exact_to, so the sparse path actually ran (and still issued no
# resolve, because the ring covers the context).
if [ -s "$OUT/on-sparse.counts" ]; then
  read -r ptok _ _ < "$OUT/on-sparse.counts"
  if [ "$ptok" -le 2051 ]; then
    echo "FAIL: the sparse prompt is $ptok tokens, at or below qsa_exact_to 2051: the dense path" >&2
    echo "      ran, so the sparse path this check is about was never exercised. Raise SPARSE_LINES." >&2
    FAILED=1
  fi
fi

if [ "$FAILED" -ne 0 ]; then
  echo "FAIL: an arm did not serve, a log was not clean, or a covering ring streamed anyway" >&2
  echo "      (see above); no zero-overhead claim can be made" >&2
  exit 2
fi

echo "== compare =="
cmp "$OUT/off-sparse.txt" "$OUT/on-sparse.txt" || { echo "FAIL: the sparse arms differ" >&2; exit 1; }
cmp "$OUT/off-dense.txt"  "$OUT/on-dense.txt"  || { echo "FAIL: the dense arms differ" >&2; exit 1; }
echo "PASS: both arms byte-identical, with a covering ring streaming nothing and issuing no resolve"
