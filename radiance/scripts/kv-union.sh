#!/bin/bash
# How large a KV streaming ring has to be: the union of distinct blocks one kv_resolve launch
# selects, against context length (docs/KV-STREAMING.md §3.5).
#
# At decode a launch selects at most max_seqs * (topk + 1) blocks; above qsa_exact_to prefill
# issues one sparse row per query, so a launch can select much of the history. This runs the
# 32 768-token floor ring on longer contexts, so misses happen, and reads the counters the engine
# logs at prefill completion and request end (ctl[10] and ctl[11]). An arm whose launch overflows is
# failed by the engine with the ring and the union; the union is read from that message.
#
# The last arm checks for stale copies: a prompt below qsa_exact_to (dense prefill, no resolve)
# decoded over a history larger than the ring, so decode evicts and re-copies blocks throughout,
# compared byte for byte with streaming off. A stale copy shows only as wrong text.
#
# Run inside the toolchain container, with the model mounted:
#   docker run --rm --security-opt seccomp=unconfined -e HOME=/tmp \
#     --device /dev/kfd --device /dev/dri --group-add 990 --group-add 44 \
#     -v "$PWD":/src -v /path/to/models:/models -w /src rad-toolchain:latest \
#     bash -lc 'bash scripts/kv-union.sh'
#
# Paths are in-container: MODEL=/models/... and BUILD (default the tree's build/). Exit status: 0 if
# every arm reported, 1 if the stale arm's texts differ, 2 if an arm did not serve or answer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MODEL="${MODEL:-/models/qwen3.8-next-flash-fp8-iq4r-moe.rad}"
PORT="${PORT:-8124}"
OUT="${OUT:-/tmp/radiance-kv-union}"
# The measurement arms climb 32768 -> 131072 tokens (8 192 -> 32 768 blocks of 4), starting at the
# floor. Nothing here runs a ring below 32 768: that is the smallest ring the operator runs, and a
# rung under it would measure a configuration the engine either raises (a decode step's selection
# under speculation) or that nobody deploys. A prompt at or under the ring is not a streaming case
# at all -- the group keeps its cache resident and no resolve is issued -- so every arm's context is
# longer than the floor ring. The stale arm's ring is the floor too, and its history is pushed past
# it by a ~34k-token prompt plus 512 decoded tokens, so decode evicts and re-copies every step.
STALE_RING="${STALE_RING:-32768}"
SEED="${SEED:-12345}"

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
    for _ in $(seq 1 240); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.25; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

# One line of text is ~30 tokens at this tokeniser, so a context of N tokens is about N / 30 lines.
# The records are distinct numbers, which keeps the prompt away from repetitive-text pathologies.
gen_prompt() {  local lines="$1" i first=$(( $1 + 1 )) last=$(( $1 + 10 ))
  for ((i = 1; i <= lines; i++)); do
    printf 'Record %d: the archivist checked the shelf, noted the catalogue number, and returned the volume to the east wing of the reading room. ' "$i"
  done
  printf 'Continue the register: write records %d through %d in the same form, one per line.' "$first" "$last"
}

# The raw JSON-escaped completion text: equal escaped bytes mean equal text, and comparing the
# escaped form is safer than any unescaper this script could carry.
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

# The engine's own report of one streamed group, from the prefill-complete / request-end line.
# Prints: resolves lookups misses union maxmisses overflow
stream_line() {
  grep -E "kv streaming: group '.*' (prefill complete|request end):" "$1" | tail -1 |
    sed -E "s/.*: ([0-9]+) resolves, ([0-9]+) lookups, ([0-9]+) misses.*largest launch union ([0-9]+) distinct blocks, largest launch misses ([0-9]+), overflow (LATCHED|no).*/\1 \2 \3 \4 \5 \6/" |
    { read -r a b c d e f 2>/dev/null && echo "$a $b $c $d $e $f" || echo "- - - - - -"; }
}

# One arm: a server, one request, a stop. $1 name, $2 prompt file, $3 max_model_len, $4 max_tokens,
# the rest are the streaming flags. Writes $OUT/$1.{log,json,txt,counts,wall}.
run_arm() {  local arm="$1" prompt="$2" ctx="$3" maxtok="$4"; shift 4
  local log="$OUT/$arm.log" req="$OUT/$arm.req.json" resp="$OUT/$arm.json"
  echo "== $arm: max-model-len $ctx, max_tokens $maxtok, $* =="
  ( exec "$BIN" --model "$MODEL" \
      --tp 2 --tp-wire exact \
      --max-num-seqs 1 --max-model-len "$ctx" --placement expert_tiered --host-pool-mib 23652 \
      --kv-cache-dtype fp8 --prefix-cache-mib 4096 --vision-tower-placement ram \
      --num-speculative-tokens 0 --deterministic \
      "$@" --host 127.0.0.1 --port "$PORT" --api-key test ) > "$log" 2>&1 &
  SERVER_PID=$!

  local i code
  for i in $(seq 1 240); do   # 20 min at 5 s; a cold load of this container is under a minute
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

  local prompt_text
  prompt_text=$(cat "$prompt")
  printf '{"model":"qwen4exp","temperature":0,"seed":%s,"max_tokens":%s,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"%s"}]}' \
    "$SEED" "$maxtok" "$prompt_text" > "$req"
  # The wall time of the answer: time_starttransfer is the first token, i.e. the prefill's cost, on
  # an arm that succeeds; a failed arm reports the time to the refusal, which is the whole prefill.
  local t
  t=$(curl -s -m 3600 -o "$resp" -w '%{http_code} %{time_starttransfer} %{time_total}' \
      -H "Authorization: Bearer test" -H "Content-Type: application/json" \
      --data-binary "@$req" "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || echo "000 0 0")
  code=${t%% *}
  echo "$t" > "$OUT/$arm.wall"
  local ptok
  ptok=$(grep -o '"prompt_tokens":[0-9]*' "$resp" | head -1 | cut -d: -f2 || true)
  # A failed overflow request has no usage block; the stream line still carries the measurement.
  [ -n "$ptok" ] && echo "$ptok" > "$OUT/$arm.ptok"
  content_of "$resp" > "$OUT/$arm.txt" 2>/dev/null || true
  stream_line "$log" > "$OUT/$arm.stream"
  echo "   http=$code prompt_tokens=${ptok:-?} stream=[$(cat "$OUT/$arm.stream")]"
  cleanup
  return 0
}

FAILED=0
# The curve: contexts of roughly 64k and 128k tokens. The short points of the record (4k, 16k) are
# gone rather than re-run: at the 32 768-token floor the ring covers a 4k or 16k context, the group
# stays resident and no resolve is issued, so there is no union to report. Each prompt is a little
# short of the bound so the chat template and the record numbers do not push it over
# --max-model-len. The bound is twice the prompt, so the KV pool is large enough for the ring beside
# the linear state.
#
# A ring too small for a launch is not served: the arm reports the union of the launch that overflowed
# and fails, which is a lower bound, not the curve. So an arm is retried with a ring twice as deep
# until no launch overflows -- the ring the arm finally ran with is the honest answer for that
# context, and the union the launch's size. Sizes are tokens, one block is 4.
declare -a NAMES=(ctx64k ctx128k)
declare -a LINES=(2100 4200)
declare -a CTX=(69632 139264)
for k in 0 1; do
  gen_prompt "${LINES[$k]}" > "$OUT/${NAMES[$k]}.prompt.txt"
  # max_tokens 8: enough to run a few decode steps (decode is where the union is small) without
  # letting a measurement arm spend its time generating.
  ok=1
  for ring in 32768 65536 131072; do   # the floor first, then the same search in floor-sized steps

    run_arm "${NAMES[$k]}" "$OUT/${NAMES[$k]}.prompt.txt" "${CTX[$k]}" 8 \
      --kv-resident "$ring" || { ok=0; break; }
    echo "$ring" > "$OUT/${NAMES[$k]}.ring"
    if grep -q "kv streaming overflow:" "$OUT/${NAMES[$k]}.log"; then
      echo "   overflowed a $ring-token ring; retrying deeper"
      continue
    fi
    break
  done
  [ "$ok" = 1 ] || FAILED=1
done

# The staleness check: a ~34k-token prompt (above qsa_exact_to, so prefill is the sparse path the
# resolve serves) streamed into the floor ring -- 32 768 tokens, 8 192 blocks -- then 512 decoded
# tokens. The prompt's history is already past the ring when prefill ends and decode pushes it
# further, so decode evicts and re-copies blocks every step: the exact case a stale copy would show
# in, where the block a step writes is often the block the previous step made resident, and the
# store now precedes the resolve. If a miss copied a block the store had not yet written, or an
# eviction raced an attention still reading the slot, the text differs from the control.
#
# The ring is not shrunk to manufacture that pressure: the floor is 32 768 and a ring below it is a
# shape the engine raises (and the operator never runs), so the *context* is what goes past the
# ring. Here that is the prompt plus the 512 generated tokens; nothing about the arm is starved.
#
# A prompt BELOW qsa_exact_to cannot be used: that prefill is dense, and the dense attention reads a
# block id as a pool slot with no resolve at all (docs/KV-STREAMING.md §3.5), so the sparse path the
# resolve serves would not be the one this arm compares.
gen_prompt 1150 > "$OUT/stale.prompt.txt"
# --max-model-len above the prompt plus the generated tokens, and above the ring: at the prompt's own
# length the ring would be the context, which streams nothing.
run_arm stale-off "$OUT/stale.prompt.txt" 45056 512                        || FAILED=1
run_arm stale-on  "$OUT/stale.prompt.txt" 45056 512 \
  --kv-resident "$STALE_RING"                                             || FAILED=1
if [ -s "$OUT/stale-off.txt" ] && [ -s "$OUT/stale-on.txt" ]; then
  # The comparison is only evidence of the copy path if the ring actually evicted: a miss count above
  # the peak union can only be a block copied more than once, i.e. evicted and needed again.
  read -r _ _ smiss _ sunion _ _ < "$OUT/stale-on.stream" || true
  if [ "${smiss:--}" != "-" ] && [ "${smiss:-0}" -le "${sunion:-0}" ]; then
    echo "FAIL: the stale arm made $smiss misses against a union of $sunion: nothing was evicted," >&2
    echo "      so the run says nothing about a miss copying a block the store wrote" >&2
    FAILED=1
  fi
  if cmp -s "$OUT/stale-off.txt" "$OUT/stale-on.txt"; then
    echo "PASS: the floor-ring decode is byte-identical to streaming off ($(wc -c < "$OUT/stale-on.txt") bytes)"
    echo identical > "$OUT/stale.verdict"
  else
    echo "FAIL: the floor-ring decode differs from streaming off -- a stale or torn copy" >&2
    echo different > "$OUT/stale.verdict"
    FAILED=1
  fi
else
  echo "FAIL: a stale arm produced no completion text" >&2
  echo missing > "$OUT/stale.verdict"
  FAILED=1
fi

# A refusal naming the ring and the union is the honest outcome for an arm that overflowed, and it is
# the measurement itself; print it once, verbatim, so the number is in the summary too.
echo "== overflow refusals =="
grep -h -E "^E kv streaming overflow:" "$OUT"/ctx*.log 2>/dev/null | sort -u || true

echo
echo "== union by context (streamed group: kv_attn) =="
printf '%-7s %8s %8s %8s %10s %10s %9s %8s %10s %10s\n' \
  arm prompt ring resolves lookups misses max_union max_miss overflow prefill_s
for arm in "${NAMES[@]}"; do
  read -r res lk mi mu mm ov < "$OUT/$arm.stream" || true
  ptok=$(cat "$OUT/$arm.ptok" 2>/dev/null || echo "?")
  # The prefill's wall time: the first token on an arm that answered, the refusal on one that
  # overflowed (its prefill ran to completion before the engine failed it).
  pt=$(awk '{ print ($1 == 200 ? $2 : $3) }' "$OUT/$arm.wall" 2>/dev/null || echo "?")
  ring=$(cat "$OUT/$arm.ring" 2>/dev/null || echo "?")
  printf '%-7s %8s %8s %8s %10s %10s %9s %8s %10s %10s\n' \
    "$arm" "${ptok:--}" "$ring" "${res:--}" "${lk:--}" "${mi:--}" "${mu:--}" "${mm:--}" "${ov:--}" "${pt:--}"
done

if [ "$FAILED" -ne 0 ]; then
  echo "FAIL: an arm did not serve, or the floor-ring text was not byte-identical" >&2
  exit 2
fi
