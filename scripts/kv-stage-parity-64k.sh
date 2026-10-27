#!/bin/bash
# That the prefill stage is exact: with a ring smaller than the context, a long prompt prefilled
# above qsa_exact_to is byte-identical to streaming off, and every staged layer was copied in layer
# order (docs/KV-STREAMING.md §3.5.1).
#
# The ring is the floor, --kv-resident 32768 (8 192 blocks of 4), and the document ~64k tokens, so
# every chunk whose history passes 32 768 tokens must take the stage. A byte-identical result is
# evidence only if something staged, so do not shorten the document.
#
# Run inside the toolchain container (io_uring and the cards), with the model mounted:
#   docker run --rm --security-opt seccomp=unconfined --user 1000:1000 -e HOME=/tmp \
#     --device /dev/kfd --device /dev/dri --group-add 990 --group-add 44 \
#     -v "$PWD":/src -v /path/to/models:/models -w /src rad-toolchain:latest \
#     bash -lc 'bash scripts/kv-stage-parity-64k.sh'
#
# One fresh server per arm, temperature 0, --deterministic, one request a server. Exit 0 only if
# both arms served, the bytes match, the on arm staged in layer order and the pool report shows the
# streamed configuration; 1 on a byte difference or missing staging; 2 if an arm failed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODEL="${MODEL:-/models/qwen3.8-next-flash-fp8-iq4r-moe.rad}"
PORT="${PORT:-8124}"
OUT="${OUT:-/tmp/radiance-kv-stage-parity-64k}"
LINES="${LINES:-2120}"                 # ~64k prompt tokens: well past the 32 768-token ring
MAXTOK="${MAXTOK:-64}"
SEED="${SEED:-12345}"
MAXLEN="${MAXLEN:-81920}"              # the prompt, the generated tokens and the template
# 8 192 blocks of 4 tokens: the floor. Above decode's one-step union and below the document, so the
# second half of the prefill must stage instead of resolving. Speculation stays 0 here: this script
# measures the stage path, and a spec arm would only change the ring's raise arithmetic.
RESIDENT="${RESIDENT:-32768}"

BIN="$BUILD/bin/radiance"
export RADIANCE_HOME="$BUILD/radiance_home"
[ -x "$BIN" ] || { echo "no engine at $BIN (build it first; this script builds nothing)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "no model at $MODEL" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is needed" >&2; exit 2; }

mkdir -p "$OUT"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 240); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

gen_prompt() {
  local n="$1" i first=$(( $1 + 1 )) last=$(( $1 + 10 ))
  for ((i = 1; i <= n; i++)); do
    printf 'Record %d: the archivist checked the shelf, noted the catalogue number, and returned the volume to the east wing of the reading room. ' "$i"
  done
  printf 'Continue the register: write records %d through %d in the same form, one per line.' "$first" "$last"
}
gen_prompt "$LINES" > "$OUT/prompt.txt"

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

# $1 arm name, $2 the streaming flags, $3.. extra environment VAR=value.
run_arm() {
  # The off arm passes no flags: with `set -u` a bare $2 made the whole script die on its first
  # arm, so the flags and the trailing VAR=value list are both optional.
  local arm="$1" flags="${2:-}"
  shift; [ "$#" -gt 0 ] && shift
  local log="$OUT/$arm.log" req="$OUT/$arm.req.json" resp="$OUT/$arm.json"
  local envs=("$@")
  echo "== $arm: $flags ${envs[*]:-} =="
  ( exec env "${envs[@]}" "$BIN" --model "$MODEL" \
      --tp 2 --tp-wire exact \
      --max-num-seqs 1 --max-model-len "$MAXLEN" --placement expert_tiered --host-pool-mib 23652 \
      --kv-cache-dtype fp8 --prefix-cache-mib 4096 --vision-tower-placement ram \
      --num-speculative-tokens 0 --deterministic \
      $flags --host 127.0.0.1 --port "$PORT" --api-key test ) > "$log" 2>&1 &
  SERVER_PID=$!

  local i code
  for i in $(seq 1 240); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FAIL $arm: the engine exited before serving; last errors:" >&2
      grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || tail -12 "$log" >&2
      cleanup; return 1
    fi
    code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    [ "$code" = 200 ] && break
    sleep 5
  done
  if [ "${code:-}" != 200 ]; then
    echo "FAIL $arm: /health did not answer" >&2
    grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || true
    cleanup; return 1
  fi

  local prompt_text
  prompt_text=$(cat "$OUT/prompt.txt")
  printf '{"model":"%s","temperature":0,"seed":%s,"max_tokens":%s,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"%s"}]}' \
    "qwen4exp" "$SEED" "$MAXTOK" "$prompt_text" > "$req"
  code=$(curl -s -m 1800 -o "$resp" -w '%{http_code}' \
    -H "Authorization: Bearer test" -H "Content-Type: application/json" \
    --data-binary "@$req" "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || true)
  if [ "${code:-}" != 200 ]; then
    echo "FAIL $arm: /v1/chat/completions answered $code; last errors:" >&2
    grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || tail -12 "$log" >&2
    cleanup; return 1
  fi
  content_of "$resp" > "$OUT/$arm.txt"
  if [ ! -s "$OUT/$arm.txt" ]; then echo "FAIL $arm: empty completion text" >&2; cleanup; return 1; fi

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

# The layer sequence the prefill stages. Each staged pass visits the group's layers in order, so
# the trace is runs of one strictly-increasing sequence; every run after the first must equal the
# first (a skipped or repeated stage breaks both properties).
check_stage_order() {
  local log="$OUT/on.log"
  grep 'kv streaming: stage layer ' "$log" |
    sed -E 's/.*stage layer ([0-9]+).*/\1/' > "$OUT/stages.txt" || true
  local n
  n=$(wc -l < "$OUT/stages.txt")
  echo "   stage layers traced: $n"
  if [ "$n" -lt 2 ]; then
    echo "FAIL: no staged layers were traced; the prefill did not take the stage view" >&2
    return 1
  fi
  awk '
    { l[NR] = $1 + 0 }
    END {
      # The first run ends at the first non-increase.
      end = 1
      while (end < NR && l[end + 1] > l[end]) end++
      if (end < 2) { print "FAIL: the first staged run is not increasing"; exit 1 }
      for (i = 2; i <= end; ++i)
        if (l[i] <= l[i-1]) { print "FAIL: layers out of order at line " i; exit 1 }
      # Every later run must repeat the first exactly.
      i = end + 1
      while (i <= NR) {
        for (j = 1; j <= end; ++j) {
          if (i > NR) break
          if (l[i] != l[j]) { print "FAIL: stage run at line " i " differs from the first"; exit 1 }
          ++i
        }
      }
      printf("   one staged run of %d layers, repeated %d time(s), in order\n", end, (NR / end) - 1)
    }' "$OUT/stages.txt" || return 1
  if grep -q "refused op 'kv_resolve'" "$log"; then
    echo "FAIL: kv_resolve was refused at issue on the staged arm" >&2; return 1
  fi
  if grep -q "GPU memory fault" "$log"; then
    echo "FAIL: a GPU memory fault was reported" >&2; return 1
  fi
  return 0
}

FAILED=0
run_arm off || FAILED=1
run_arm on "--kv-resident $RESIDENT" RADIANCE_KV_STAGE_TRACE=1 || FAILED=1

if [ "$FAILED" -eq 0 ]; then
  if ! grep -q "kv streaming: prefill staging" "$OUT/on.log"; then
    echo "FAIL: the streaming configuration is not in the on arm's log" >&2
    FAILED=1
  fi
  read -r ptok _ _ < "$OUT/on.counts"
  if [ "${ptok:-0}" -le 2051 ]; then
    echo "FAIL: the prompt is $ptok tokens, at or below qsa_exact_to 2051; prefill was dense," >&2
    echo "      so raise LINES and rerun: this run would not prove the sparse staged path" >&2
    FAILED=1
  elif [ "${ptok:-0}" -le "$RESIDENT" ]; then
    echo "FAIL: the prompt is $ptok tokens, at or under the $RESIDENT-token ring: it fits, so" >&2
    echo "      nothing streamed and the stage was not taken. Raise LINES, never lower RESIDENT" >&2
    FAILED=1
  fi
  check_stage_order || FAILED=1
fi

echo "== KV pool before and after (from each arm's vram split and kv report) =="
grep -E "vram split|^I kv:|^I kv .*streaming" "$OUT/off.log" | sed 's/^/  off: /' || true
grep -E "vram split|^I kv:|^I kv .*streaming|prefill staging" "$OUT/on.log" | sed 's/^/  on:  /' || true

if [ "$FAILED" -ne 0 ]; then
  echo "FAIL: an arm did not serve, the bytes were not compared, or staging was not proven" >&2
  exit 2
fi

echo "== compare =="
cmp "$OUT/off.txt" "$OUT/on.txt" || { echo "FAIL: the two arms differ" >&2; exit 1; }
echo "PASS: byte-identical with a 32 768-token floor ring, prefill staged in layer order"
