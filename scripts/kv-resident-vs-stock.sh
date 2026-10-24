#!/bin/bash
# Measure the streaming-vs-stock decode gap at the canonical under-resident shape, as a paired
# number with an interval. The prompt is generated here, so a run does not depend on a file in /tmp
# surviving a reboot.
#
# The shape:
#   16 110-token prompt (sha256 c34ad060...f3a5) -- gen_prompt 522, generated below
#   8 192 generated tokens, --kv-resident 32768 (8 192 blocks = 200.00 MiB, inside a 204 800-token
#   context, so the group streams and every decode launch resolves)
#   stock image stilldeadcode/radiance:1.10.0 (no --kv-resident) against this build with it.
# Every other flag is identical between arms.
#
# SPEC (default 0) sets --num-speculative-tokens on *both* arms, which is the only thing that
# should differ between them. Two things change when it is on, and both are needed or the numbers
# mean nothing:
#   - ms/step is no longer 1000/predicted_per_second. With speculation a step emits 1 + SPEC
#     tokens, so predicted_per_second is tokens a second, not steps. ms/step is then the engine's
#     own step clock, engine_step_seconds_decode_total / engine_steps_decode_total, which /metrics
#     reports on both arms (the stock 1.10.0 image exposes the same counters). tok/s stays
#     predicted_per_second, tokens a step is MAXTOK / decode steps, and draft acceptance is
#     timings.draft_n_accepted / timings.draft_n -- the bimodal variable, printed per start so the
#     modes are visible instead of averaged.
#   - a start is ~6x shorter (spec 3 decodes 1 024 tokens in ~16 s, not ~206 s), so the same wall
#     time buys many more pairs. SPEC=3 at ROUNDS=12 is ~20 min against ~16 min for ROUNDS=2 off.
#
# Method: interleaved ABBA rounds. Round r sends one request on each arm, the two starts adjacent
# in wall time, and alternates which arm goes first (round 1 stock first, round 2 ours first, ...).
# Each round yields one paired delta (ours ms/step over stock ms/step); the mean of those deltas,
# the per-round values, the spread and a t interval are reported. Interleaving cancels the
# common-mode part of session drift; the residual non-common-mode part is ±1.5 points, which is why
# the interval is wide.
#
# Why one cold start per request rather than one server per arm reused across its rounds: the two
# arms cannot hold the cards at once (each is `--tp 2` with --host-pool-mib 23652), so a server
# reused "across that arm's rounds" would have to be killed and restarted between them anyway. And
# a second request in the same process starts from a partly filled ring, i.e. it is not the same
# workload as the first. So each
# arm's rounds are adjacent cold starts, one request each, and only one server is ever on the cards.
#
# Cost and resolution: a start is ~18 s load + ~6 s prefill + 8 192 x ~26 ms
# decode, i.e. ~4 min, so ROUNDS=2 (two starts per arm, four in all) is ~16 min. At a cross-arm
# sd of 1.4 points the minimum detectable difference is 2.8 x sd / sqrt(n) (95%
# two-sided, 80% power): ~2.8% at 2 rounds, ~2.0% at 4. To resolve **1%** needs ~16 paired rounds
# and to resolve **0.9%** needs ~19-20, i.e. 32-40 cold starts and ~2.5 h of card time. A small
# default here is deliberate: this script states the shipped gap, it does not test a 1% change.
#
# What it proves about the two arms being on the same input: the prompt is regenerated here and its
# sha256 must equal the canonical one; *every* server started is asked POST /tokenize for that
# prompt and must answer the target count before its request runs; and every response's
# usage.prompt_tokens must equal the target again, with cached_tokens 0 (a prefix-cache hit would
# make the two arms' prefill a different workload). A mismatch kills the run.
#
# Run on the host, where the docker CLI talks to the daemon (the images and the build tree live
# here; the engine needs the model mounted and io_uring unconfined, which the script does):
#   bash scripts/kv-resident-vs-stock.sh                 # 2 rounds, ~16 min
#   ROUNDS=4 bash scripts/kv-resident-vs-stock.sh        # ~32 min, ~2.0% resolution
#   SPEC=3 ROUNDS=12 bash scripts/kv-resident-vs-stock.sh  # speculation on, ~20 min, 12 pairs
#
# Env knobs (defaults in parentheses): ROUNDS (2), MAXTOK (8192), RESIDENT (32768), SPEC (0),
# PORT (8141), OUT (/tmp/opencode/kv-resident-vs-stock), BUILD (/tmp/opencode/build-kv3),
# MODEL (/home/j/Documents/rad/models/qwen3.8-next-flash-fp8-iq4r-moe.rad),
# STOCK_IMAGE (stilldeadcode/radiance:1.10.0), TOOLCHAIN (rad-toolchain:latest), SEED (12345).
# TARGET_TOKENS (16110), PROMPT_LINES (522) and PROMPT_SHA are pinned on purpose: they are the
# workload, and a run against a different prompt is a different measurement.
#
# Exit status: 0 if every round measured and the arms agreed on the prompt; 2 if the environment,
# an arm, a token count, a ring check or the model file failed (the reason is on stderr).
set -euo pipefail

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
}

ROUNDS="${ROUNDS:-2}"
MAXTOK="${MAXTOK:-8192}"
RESIDENT="${RESIDENT:-32768}"
SPEC="${SPEC:-0}"
PORT="${PORT:-8141}"
OUT="${OUT:-/tmp/opencode/kv-resident-vs-stock}"
BUILD="${BUILD:-/tmp/opencode/build-kv3}"
MODEL="${MODEL:-/home/j/Documents/rad/models/qwen3.8-next-flash-fp8-iq4r-moe.rad}"
STOCK_IMAGE="${STOCK_IMAGE:-stilldeadcode/radiance:1.10.0}"
TOOLCHAIN="${TOOLCHAIN:-rad-toolchain:latest}"
SEED="${SEED:-12345}"

# The workload, pinned: the canonical prompt, and what the server counted it as.
PROMPT_LINES=522
TARGET_TOKENS=16110
PROMPT_SHA=c34ad06069bc07c791e61789ffbace20f5879781c9fbc2eb7ea009e09cb3f3a5
# Pinned so "the model is the expected file" is a check and not a hope: the whole 122 GB is too
# much to hash every run, so the size plus the first 4 MiB identifies the conversion.
MODEL_BYTES=121969901568
MODEL_HEAD_SHA=f2958962468b9b112ebc4bcf78d6da9b270477c7b636a0ded7418b7298e434b4

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --rounds) [ $# -ge 2 ] || { echo "--rounds needs a value" >&2; exit 2; }; ROUNDS="$2"; shift 2 ;;
    --spec)   [ $# -ge 2 ] || { echo "--spec needs a value" >&2; exit 2; }; SPEC="$2"; shift 2 ;;
    *) echo "unknown option: $1 (--help for usage)" >&2; exit 2 ;;
  esac
done

case "$ROUNDS" in
  ''|*[!0-9]*) echo "ROUNDS must be a positive integer, got '$ROUNDS'" >&2; exit 2 ;;
esac
[ "$ROUNDS" -ge 2 ] || { echo "ROUNDS must be at least 2: one round has no spread, so no interval" >&2; exit 2; }
case "$SPEC" in
  ''|*[!0-9]*) echo "SPEC must be a non-negative integer, got '$SPEC'" >&2; exit 2 ;;
esac
[ "$SPEC" -le 8 ] || { echo "SPEC $SPEC is deeper than the drafter is built for (0-8)" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_IN="/models/$(basename "$MODEL")"
RING_BLOCKS=$(( RESIDENT / 4 ))     # one KV block is 4 tokens (docs/KV-STREAMING.md §3.9)

die() { echo "FAIL: $*" >&2; exit 2; }

command -v docker >/dev/null || die "docker is needed: this script drives both arms as containers"
docker image inspect "$STOCK_IMAGE" >/dev/null 2>&1 || die "no image $STOCK_IMAGE (docker pull it)"
docker image inspect "$TOOLCHAIN"   >/dev/null 2>&1 || die "no image $TOOLCHAIN"
[ -x "$BUILD/bin/radiance" ] || die "no engine at $BUILD/bin/radiance (build it first; this script builds nothing)"
[ -d "$BUILD/radiance_home" ] || die "no RADIANCE_HOME at $BUILD/radiance_home"
[ -f "$MODEL" ] || die "no model at $MODEL"
[ "$(stat -c %s "$MODEL")" = "$MODEL_BYTES" ] || \
  die "$MODEL is $(stat -c %s "$MODEL") bytes, expected $MODEL_BYTES: not the pinned conversion"
head -c 4194304 "$MODEL" | sha256sum | cut -d' ' -f1 | grep -qx "$MODEL_HEAD_SHA" || \
  die "$MODEL's first 4 MiB do not hash to $MODEL_HEAD_SHA: not the pinned conversion"
# The host that drives docker has wget but not necessarily curl (the toolchain container has both);
# either will do, and the request's own `timings` block is the clock, so neither has to report one.
HAVE_CURL=0
if command -v curl >/dev/null; then HAVE_CURL=1
elif ! command -v wget >/dev/null; then die "curl or wget is needed to talk to the engine"
fi
BASE="http://127.0.0.1:$PORT"
AUTH="Authorization: Bearer test"

http_get() {  # $1 path, $2 out file, $3 timeout s; prints the HTTP status
  local url="$BASE$1" out="$2" to="${3:-30}" c
  if [ "$HAVE_CURL" = 1 ]; then
    c=$(curl -s -m "$to" -o "$out" -w '%{http_code}' -H "$AUTH" "$url" 2>/dev/null) || c=000
  else
    wget -S -nv -O "$out" --timeout="$to" --header="$AUTH" "$url" 2>"$out.hdr" || true
    c=$(awk '/HTTP\//{ c = $2 } END { print (c == "" ? "000" : c) }' "$out.hdr")
  fi
  printf '%s\n' "${c:-000}"
}

http_post() {  # $1 path, $2 body file, $3 out file, $4 timeout s; prints the HTTP status
  local url="$BASE$1" body="$2" out="$3" to="${4:-60}" c
  if [ "$HAVE_CURL" = 1 ]; then
    c=$(curl -s -m "$to" -o "$out" -w '%{http_code}' \
          -H "$AUTH" -H "Content-Type: application/json" --data-binary "@$body" "$url" 2>/dev/null) || c=000
  else
    wget -S -nv -O "$out" --timeout="$to" --header="$AUTH" \
      --header="Content-Type: application/json" --post-file="$body" "$url" 2>"$out.hdr" || true
    c=$(awk '/HTTP\//{ c = $2 } END { print (c == "" ? "000" : c) }' "$out.hdr")
  fi
  printf '%s\n' "${c:-000}"
}

# The cards are exclusive: one server at a time, so anything else already running is a refusal and
# not something this script may stop (rules 1-3 of AGENTS.md, which also covers radiance-*).
RUNNING=$(docker ps -q | wc -l)
if [ "$RUNNING" -ne 0 ]; then
  echo "FAIL: $RUNNING container(s) already running; the cards are exclusive to one server:" >&2
  docker ps --format '  {{.Names}}  {{.Image}}' >&2
  echo "      free them (that is the operator's call) and run this again" >&2
  exit 2
fi
mkdir -p "$OUT"

if [ "$(http_get /health "$OUT/health.probe.json" 2)" != 000 ]; then
  die "something already answers on 127.0.0.1:$PORT/health; pick PORT=..."
fi

RUN_NAME=""     # the one container this script may kill, ever
# `docker rm -f` both kills and removes, and is retried until the container is really gone: a
# container left behind holds the cards and blocks the next start (and the next operator).
stop_arm() {
  [ -n "$RUN_NAME" ] || return 0
  local name="$RUN_NAME" i
  docker rm -f "$name" >/dev/null 2>&1 || true
  for i in $(seq 1 60); do
    docker inspect "$name" >/dev/null 2>&1 || { RUN_NAME=""; return 0; }
    sleep 0.5
  done
  echo "FAIL: $name is still present after 30 s; remove it by hand before reusing the cards" >&2
  RUN_NAME=""
  return 0
}
cleanup() { stop_arm; }
trap cleanup EXIT INT TERM

# One line of ~30 tokens, the same generator shape as scripts/kv-union.sh -- and 522 of them is the
# c34ad060 prompt byte for byte. Distinct record numbers keep it away from repetitive-text
# pathologies, and the closing instruction is what the model continues.
gen_prompt() {
  local lines="$1" i first=$(( $1 + 1 )) last=$(( $1 + 10 ))
  for ((i = 1; i <= lines; i++)); do
    printf 'Record %d: the archivist checked the shelf, noted the catalogue number, and returned the volume to the east wing of the reading room. ' "$i"
  done
  printf 'Continue the register: write records %d through %d in the same form, one per line.' "$first" "$last"
}

gen_prompt "$PROMPT_LINES" > "$OUT/prompt.txt"
PROMPT_BYTES=$(sha256sum "$OUT/prompt.txt" | cut -d' ' -f1)
[ "$PROMPT_BYTES" = "$PROMPT_SHA" ] || \
  die "the generated prompt hashes to $PROMPT_BYTES, not the canonical $PROMPT_SHA"

# temperature 0 and a fixed seed: greedy, so the only thing that can move a token is the engine.
# ignore_eos keeps the generated count at MAXTOK on both arms even if the model wants to stop.
printf '{"model":"qwen4exp","temperature":0,"seed":%s,"max_tokens":%s,"ignore_eos":true,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"%s"}]}' \
  "$SEED" "$MAXTOK" "$(cat "$OUT/prompt.txt")" > "$OUT/req.json"

# The two arms differ in exactly one thing: the streaming flags. Everything else is identical,
# including --prefix-cache-ttl 0 (a prefix hit would leave the arm's own prompt uncounted) and
# --num-speculative-tokens $SPEC (0 is the shipped shape; SPEC>0 puts it on on both arms).
# A ring under one decode step's selection is raised, not refused, and a raised ring
# would make the ours arm's block count differ from RESIDENT/4 -- which the ring check below
# fails on, so this script can never silently measure a ring the operator did not ask for. At
# --max-num-seqs 2 and SPEC 3 that ceiling is 2 x 4 x 513 = 4 104 blocks, under 8 192, so 32 768
# tokens is not raised.
FLAGS=(--model "$MODEL_IN" --tp 2 --tp-wire exact --max-num-seqs 2 --max-model-len 204800
       --placement expert_tiered --host-pool-mib 23652 --gpu-headroom-mib 96
       --kv-cache-dtype fp8 --prefix-cache-mib 1024 --prefix-cache-ttl 0
       --vision-tower-placement ram --num-speculative-tokens "$SPEC" --deterministic
       --host 127.0.0.1 --port "$PORT" --api-key test)

start_arm() {  # $1 arm, $2 tag; sets RUN_NAME and writes $OUT/$2.server.log
  local arm="$1" tag="$2"
  RUN_NAME="kv-rvs-$$-$tag"
  case "$arm" in
    ours)
      docker run --rm --name "$RUN_NAME" --user 1000:1000 -e HOME=/tmp \
        --device /dev/kfd --device /dev/dri --group-add 990 --group-add 44 \
        --security-opt seccomp=unconfined --network host \
        -v "$ROOT:/src" -v "$(dirname "$MODEL"):/models" -v /tmp/opencode:/tmp/opencode \
        -w /src "$TOOLCHAIN" \
        bash -lc "RADIANCE_HOME=$BUILD/radiance_home exec $BUILD/bin/radiance ${FLAGS[*]} --kv-resident $RESIDENT" \
        > "$OUT/$tag.server.log" 2>&1 &
      ;;
    stock)
      docker run --rm --name "$RUN_NAME" \
        --device /dev/kfd --device /dev/dri --group-add 990 --group-add 44 \
        --security-opt seccomp=unconfined --network host \
        -v "$(dirname "$MODEL"):/models" \
        "$STOCK_IMAGE" "${FLAGS[@]}" \
        > "$OUT/$tag.server.log" 2>&1 &
      ;;
    *) die "unknown arm '$arm'" ;;
  esac
}

# Readiness is /health, never a blind wait. A container that dies while we poll is a hard failure:
# it is either a bad flag, a card that someone else took, or a refusal worth reading.
wait_health() {  # $1 tag
  local tag="$1" i code status missing=0
  for i in $(seq 1 240); do
    # `docker run` returns before the container is in the running state, and --rm removes a
    # container as soon as its process ends, so only a state that stays gone is a failure.
    status=$(docker inspect -f '{{.State.Status}}' "$RUN_NAME" 2>/dev/null || echo gone)
    case "$status" in
      running|created|restarting) missing=0 ;;
      *)
        missing=$(( missing + 1 ))
        if [ "$missing" -ge 2 ]; then
          echo "FAIL: $tag's engine exited before /health; last errors:" >&2
          grep -E "^(E |radiance: fatal)" "$OUT/$tag.server.log" | tail -12 >&2 || \
            tail -12 "$OUT/$tag.server.log" >&2
          return 2
        fi
        ;;
    esac
    code=$(http_get /health "$OUT/$tag.health.json" 3)
    [ "$code" = 200 ] && { printf '%s\n' "$i" > "$OUT/$tag.health_polls"; return 0; }
    sleep 1
  done
  echo "FAIL: $tag's /health did not answer in 240 s; last errors:" >&2
  tail -12 "$OUT/$tag.server.log" >&2
  return 2
}

# The server's own count of the prompt the chat endpoint will render -- the same template, the same
# variables -- so the two arms are provably on one input before either is timed.
tokenize_count() {
  printf '{"messages":[{"role":"user","content":"%s"}],"chat_template_kwargs":{"enable_thinking":false}}' \
    "$(cat "$OUT/prompt.txt")" > "$OUT/tokenize.req.json"
  http_post /tokenize "$OUT/tokenize.req.json" "$OUT/tokenize.resp.json" 120 >/dev/null
  grep -o '"count":[0-9]*' "$OUT/tokenize.resp.json" | head -1 | cut -d: -f2 || true
}

# A number from a JSON body: `<key>: <number>`. Bounded to the first hit, which is the one the OAI
# usage block carries first (there is no nesting under those keys).
jnum() {  # $1 file, $2 key
  grep -o "\"$2\": *[-0-9.eE+]*" "$1" | head -1 | sed 's/.*: *//' || true
}

# The value of one /metrics counter, by its bare name. Anchored on `{` or a space after the name so
# `engine_steps_total` cannot pick up `engine_steps_decode_total`; the first sample is the only one
# (one model, one engine). Prints nothing when the counter is absent.
metric() {  # $1 metrics file, $2 counter name (without the radiance: prefix)
  [ -s "$1" ] || return 0
  grep -E "^radiance:$2(\\{| )" "$1" 2>/dev/null | head -1 | awk '{print $NF}' || true
}

# The engine's own report of a streamed group at request end, from the log line. Prints
# `resolves lookups misses ring union maxmiss claims overflow`, or six dashes if the line is absent.
stream_end() {  # $1 log
  grep -E "kv streaming: group '.*' request end:" "$1" | tail -1 |
    sed -E "s/.*request end: ([0-9]+) resolves, ([0-9]+) lookups, ([0-9]+) misses.*ring ([0-9]+) blocks, largest launch union ([0-9]+) distinct blocks, largest launch misses ([0-9]+), largest launch claims ([0-9]+), overflow (no|LATCHED).*/\1 \2 \3 \4 \5 \6 \7 \8/" \
    | grep -E '^[0-9]+ ' || echo "- - - - - - - -"
}

# The ring the engine actually carved, from its startup report:
# `streaming:  8192 blocks resident in vram_kv at +6885376 = 200.00 MiB, ...`
ring_line() {  # $1 log -> "blocks MiB"
  grep -oE "streaming: *[0-9]+ blocks resident in vram_kv at \+[0-9]+ = *[0-9.]+ MiB" "$1" |
    head -1 | sed -E 's/.*streaming: *([0-9]+) blocks.*= *([0-9.]+) MiB/\1 \2/' || true
}

# `vram_kv 618.00 MiB the cache's share` -- the placement table, not the "budget ... free" line.
vram_kv() { grep -E "^ *vram_kv +[0-9.]+ (MiB|GiB) +the cache's share" "$1" | head -1 |
              sed -E 's/^ *vram_kv +([0-9.]+ (MiB|GiB)).*/\1/' || true; }

# `experts  11.22 GiB  95% of it` -- residency in the elastic region.
experts() { grep -E "^ *experts +[0-9.]+ GiB +[0-9]+% of it" "$1" | head -1 |
              sed -E 's/^ *experts +([0-9.]+) GiB +([0-9]+)% of it/\1 \2/' || true; }

# One start: a server, one request, a kill. Kills only the container this script named, and always
# before the other arm starts, so the cards never hold two engines. Writes one line of
# $OUT/starts.tsv: round arm order ms_step prefill_tps ttft decode_tps wall ptok ctok cached.
run_start() {  # $1 arm, $2 round, $3 order (1 = first in the round)
  local arm="$1" round="$2" order="$3"
  local tag="$arm$round" log="$OUT/$arm$round.server.log" resp="$OUT/$arm$round.json"
  local t code ptok ctok cached tokencount

  local ord="2nd"; [ "$order" = 1 ] && ord="1st"
  echo "== $tag: $arm, round $round of $ROUNDS, $ord in the round =="
  start_arm "$arm" "$tag"
  wait_health "$tag" || { stop_arm; return 2; }

  tokencount=$(tokenize_count)
  [ "${tokencount:-0}" = "$TARGET_TOKENS" ] || {
    echo "FAIL $tag: /tokenize counts this prompt as ${tokencount:-?} tokens, expected $TARGET_TOKENS" >&2
    stop_arm; return 2
  }
  echo "$tokencount" > "$OUT/$tag.tokens"

  t0=$(date +%s%N)
  code=$(http_post /v1/chat/completions "$OUT/req.json" "$resp" 3600)
  t1=$(date +%s%N)
  printf '%s %s\n' "$code" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1e9}')" \
    > "$OUT/$tag.wall"

  # The counters are read before the kill: /metrics belongs to the live process. Ours' are always
  # read (the ring's byte and overflow counters live there); stock's only when speculation makes
  # its step clock the paired quantity.
  if [ "$arm" = ours ] || [ "$SPEC" -gt 0 ]; then
    http_get /metrics "$OUT/$tag.metrics" 10 >/dev/null || true
  fi
  stop_arm

  if [ "$code" != 200 ]; then
    echo "FAIL $tag: /v1/chat/completions answered $code; last errors:" >&2
    grep -E "^(E |radiance: fatal)" "$log" | tail -12 >&2 || tail -12 "$log" >&2
    return 2
  fi

  ptok=$(jnum "$resp" prompt_tokens); ctok=$(jnum "$resp" completion_tokens)
  cached=$(jnum "$resp" cached_tokens)
  [ "$ptok" = "$TARGET_TOKENS" ] || {
    echo "FAIL $tag: usage.prompt_tokens is ${ptok:-?}, expected $TARGET_TOKENS" >&2; return 2; }
  [ "$ctok" = "$MAXTOK" ] || {
    echo "FAIL $tag: usage.completion_tokens is ${ctok:-?}, expected $MAXTOK" >&2; return 2; }
  [ "${cached:-0}" = 0 ] || {
    echo "FAIL $tag: cached_tokens ${cached:-?} -- a prefix hit would make the arms' prefill" >&2
    echo "      a different workload" >&2; return 2; }

  # The clocks are the engine's own `timings` block, so they do not depend on which HTTP client ran:
  # prompt_per_second is the prefill, ttft_ms the first token, predicted_per_second the decode (one
  # token a step with speculation off), predicted_ms the decode wall. `wall` is the harness's total.
  local ms pre ttft decode_tps dwall wall
  ms=$(awk -v p="$(jnum "$resp" predicted_per_second)" 'BEGIN{ if (p > 0) printf "%.4f", 1000/p; else print "-" }')
  decode_tps=$(jnum "$resp" predicted_per_second)
  pre=$(jnum "$resp" prompt_per_second)
  ttft=$(awk -v t="$(jnum "$resp" ttft_ms)" 'BEGIN{ printf "%.3f", t/1000 }')
  dwall=$(awk -v t="$(jnum "$resp" predicted_ms)" 'BEGIN{ printf "%.1f", t/1000 }')
  wall=$(cut -d' ' -f2 "$OUT/$tag.wall")

  # Speculation on: a step emits up to 1 + SPEC tokens, so 1000/predicted_per_second is ms a token
  # and is not the quantity two arms are compared on. The engine's own decode step clock is, and
  # both arms report it under the same name. Draft acceptance is printed per start because stock
  # starts can be bimodal in it (81.7% and 98.6% on different starts); averaging that hides which
  # mode a start landed in.
  local steps="" tokstep="" dnum="" dacc="" dacc_pct="" spf="" smx=""
  if [ "$SPEC" -gt 0 ]; then
    local secs
    steps=$(metric "$OUT/$tag.metrics" engine_steps_decode_total)
    secs=$(metric "$OUT/$tag.metrics" engine_step_seconds_decode_total)
    spf=$(metric "$OUT/$tag.metrics" engine_steps_prefill_total)
    smx=$(metric "$OUT/$tag.metrics" engine_steps_mixed_total)
    dnum=$(jnum "$resp" draft_n)
    dacc=$(jnum "$resp" draft_n_accepted)
    case "${steps:-x}" in ''|*[!0-9]*) steps="";; esac
    [ -n "$steps" ] && [ "$steps" -gt 0 ] || {
      echo "FAIL $tag: no engine_steps_decode_total in /metrics, so ms/step has no clock" >&2
      return 2; }
    ms=$(awk -v s="$secs" -v n="$steps" 'BEGIN{ if (n > 0 && s > 0) printf "%.4f", 1000*s/n; else print "-" }')
    [ "$ms" != "-" ] || {
      echo "FAIL $tag: engine_step_seconds_decode_total is '${secs:-?}' for $steps steps" >&2
      return 2; }
    tokstep=$(awk -v n="$MAXTOK" -v s="$steps" 'BEGIN{ printf "%.3f", n/s }')
    # A start whose drafter never ran is not a spec-on measurement, and neither is one whose
    # acceptance counters are missing: both are the shape, not the result.
    case "${dnum:-x}${dacc:-x}" in *[!0-9]*) dnum="";; esac
    [ -n "$dnum" ] && [ "$dnum" -gt 0 ] || {
      echo "FAIL $tag: timings carries no draft_n under --num-speculative-tokens $SPEC," >&2
      echo "      so the drafter did not run and this is not a spec-on start" >&2; return 2; }
    [ "${dacc:-0}" -le "$dnum" ] || {
      echo "FAIL $tag: draft_n_accepted ${dacc:-?} exceeds draft_n $dnum" >&2; return 2; }
    dacc_pct=$(awk -v a="$dacc" -v n="$dnum" 'BEGIN{ printf "%.1f", 100*a/n }')
  fi

  if [ "$arm" = ours ]; then
    # The ring checks. A run whose ring did not stream, overflowed, or held a different number of
    # blocks is not the configuration this script claims to measure, so it stops here.
    grep -q "kv streaming: [0-9].*stream (" "$log" || {
      echo "FAIL $tag: no 'group(s) stream' startup line: --kv-resident $RESIDENT did not" >&2
      echo "      leave the group streaming" >&2; return 2; }
    read -r res lk mi rb mu mm cl ov < <(stream_end "$log")
    [ "${ov:-}" = no ] || { echo "FAIL $tag: ring overflow '$ov' on the request-end line" >&2; return 2; }
    [ "${rb:-0}" = "$RING_BLOCKS" ] || {
      echo "FAIL $tag: the ring holds ${rb:-?} blocks, expected $RING_BLOCKS for $RESIDENT tokens" >&2
      return 2; }
    [ "${mu:-0}" -le "${rb:-0}" ] || {
      echo "FAIL $tag: largest launch union ${mu:-?} exceeds the ${rb:-?}-block ring" >&2; return 2; }
    [ "${mi:-0}" -gt 0 ] || {
      echo "FAIL $tag: no misses -- the ring covered the context, so nothing streamed" >&2; return 2; }
    read -r ring_blocks ring_mib < <(ring_line "$log") || true
    [ "${ring_blocks:-0}" = "$RING_BLOCKS" ] || {
      echo "FAIL $tag: startup reports ${ring_blocks:-?} blocks resident, expected $RING_BLOCKS" >&2
      return 2; }
    printf '%s %s %s %s %s %s %s %s\n' "$res" "$lk" "$mi" "$rb" "$mu" "$mm" "$cl" "$ov" \
      > "$OUT/$tag.stream"
    printf '%s %s\n' "${ring_blocks:-?}" "${ring_mib:-?}" > "$OUT/$tag.ring"
  else
    grep -q "kv streaming: [0-9].*stream (" "$log" && {
      echo "FAIL $tag: the stock arm streamed, so it was not the stock configuration" >&2; return 2; }
  fi
  printf '%s\n' "$(vram_kv "$log")" > "$OUT/$tag.vram_kv"
  printf '%s\n' "$(experts "$log")" > "$OUT/$tag.experts"

  # Speculation's own columns go on the same line, after the twelve: every reader of columns 1-12
  # above is unchanged, so a SPEC=0 run writes the twelve-column starts.tsv, byte for byte.
  if [ "$SPEC" -gt 0 ]; then
    printf '%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' \
      "$round" "$arm" "$order" "$ms" "$pre" "$ttft" "$decode_tps" "$dwall" "$ptok" "$ctok" \
      "${cached:-0}" "$wall" "$steps" "$tokstep" "$dnum" "$dacc" "$dacc_pct" "$spf" "$smx" \
      "$SPEC" >> "$OUT/starts.tsv"
  else
    printf '%s %s %s %s %s %s %s %s %s %s %s %s\n' \
      "$round" "$arm" "$order" "$ms" "$pre" "$ttft" "$decode_tps" "$dwall" "$ptok" "$ctok" \
      "${cached:-0}" "$wall" >> "$OUT/starts.tsv"
  fi

  if [ "$SPEC" -gt 0 ]; then
    printf '   ms/step %s, %.2f tok/s, %.3f tok/step, %s decode steps, draft %s/%s = %.1f%% accepted, prefill %.1f tok/s, request %.1f s\n' \
      "$ms" "$decode_tps" "$tokstep" "$steps" "$dacc" "$dnum" "$dacc_pct" "$pre" "$wall"
  else
    printf '   ms/step %s, decode %.2f tok/s, prefill %.1f tok/s, TTFT %.3f s, decode wall %.1f s, request %.1f s\n' \
      "$ms" "$decode_tps" "$pre" "$ttft" "$dwall" "$wall"
  fi
  return 0
}

# t quantiles, two-sided. The table covers the round counts this script is usable at; beyond 30 the
# normal value is within 0.2% and the interval is dominated by the spread, not the quantile.
tval() {  # $1 df, $2 95|90
  local df="$1" q="$2"
  if [ "$q" = 95 ]; then
    case "$df" in
      1) echo 12.706 ;; 2) echo 4.303 ;; 3) echo 3.182 ;; 4) echo 2.776 ;; 5) echo 2.571 ;;
      6) echo 2.447 ;; 7) echo 2.365 ;; 8) echo 2.306 ;; 9) echo 2.262 ;; 10) echo 2.228 ;;
      11) echo 2.201 ;; 12) echo 2.179 ;; 13) echo 2.160 ;; 14) echo 2.145 ;; 15) echo 2.131 ;;
      16) echo 2.120 ;; 17) echo 2.110 ;; 18) echo 2.101 ;; 19) echo 2.093 ;; 20) echo 2.086 ;;
      21) echo 2.080 ;; 22) echo 2.074 ;; 23) echo 2.069 ;; 24) echo 2.064 ;; 25) echo 2.060 ;;
      26) echo 2.056 ;; 27) echo 2.052 ;; 28) echo 2.048 ;; 29) echo 2.045 ;; 30) echo 2.042 ;;
      *) echo 1.960 ;;
    esac
  else
    case "$df" in
      1) echo 6.314 ;; 2) echo 2.920 ;; 3) echo 2.353 ;; 4) echo 2.132 ;; 5) echo 2.015 ;;
      6) echo 1.943 ;; 7) echo 1.895 ;; 8) echo 1.860 ;; 9) echo 1.833 ;; 10) echo 1.812 ;;
      11) echo 1.796 ;; 12) echo 1.782 ;; 13) echo 1.771 ;; 14) echo 1.761 ;; 15) echo 1.753 ;;
      16) echo 1.746 ;; 17) echo 1.740 ;; 18) echo 1.734 ;; 19) echo 1.729 ;; 20) echo 1.725 ;;
      21) echo 1.721 ;; 22) echo 1.717 ;; 23) echo 1.714 ;; 24) echo 1.711 ;; 25) echo 1.708 ;;
      26) echo 1.706 ;; 27) echo 1.703 ;; 28) echo 1.701 ;; 29) echo 1.699 ;; 30) echo 1.697 ;;
      *) echo 1.645 ;;
    esac
  fi
}

printf 'prompt sha256 %s (%s bytes), %s tokens, request sha256 %s\n' \
  "$PROMPT_BYTES" "$(wc -c < "$OUT/prompt.txt")" "$TARGET_TOKENS" \
  "$(sha256sum "$OUT/req.json" | cut -d' ' -f1)"
printf 'arms: stock=%s  ours=%s --kv-resident %s (%s blocks)\n' \
  "$STOCK_IMAGE" "$BUILD/bin/radiance" "$RESIDENT" "$RING_BLOCKS"
if [ "$SPEC" -gt 0 ]; then
  printf 'speculation: --num-speculative-tokens %s on both arms; ms/step = engine_step_seconds_decode_total /\n' "$SPEC"
  printf '  engine_steps_decode_total (one step emits up to %s tokens), acceptance = timings.draft_n_accepted / draft_n\n' \
    "$(( SPEC + 1 ))"
  printf 'rounds: %s per arm, %s starts, ~%s min\n\n' "$ROUNDS" "$(( ROUNDS * 2 ))" "$(( ROUNDS * 9 / 6 ))"
else
  printf 'rounds: %s per arm, %s starts, ~%s min\n\n' "$ROUNDS" "$(( ROUNDS * 2 ))" "$(( ROUNDS * 9 ))"
fi

: > "$OUT/starts.tsv"
STARTED=$(date +%s)
r=1
while [ "$r" -le "$ROUNDS" ]; do
  # ABBA: the arm that goes first alternates by round, so the two arms share first position and no
  # arm is always measured on the cooler or hotter side of a round.
  if [ $(( r % 2 )) -eq 1 ]; then
    run_start stock "$r" 1 || exit 2
    run_start ours  "$r" 2 || exit 2
  else
    run_start ours  "$r" 1 || exit 2
    run_start stock "$r" 2 || exit 2
  fi
  r=$(( r + 1 ))
done
ELAPSED=$(( $(date +%s) - STARTED ))

# ---- ours' counters, per round ----------------------------------------------------------------
echo
echo "== ours, per round (the engine's own request-end line; bytes and overflow from /metrics) =="
printf '%-5s %9s %11s %9s %6s %9s %10s %9s %8s\n' \
  round resolves lookups misses ring union maxmiss bytes overflow
for r in $(seq 1 "$ROUNDS"); do
  read -r res lk mi rb mu mm cl ov < "$OUT/ours$r.stream"
  bytes=$(grep -E "^radiance:kv_stream_bytes_total" "$OUT/ours$r.metrics" 2>/dev/null | head -1 |
            awk '{print $NF}' || true)
  ovm=$(grep -E "^radiance:stream_overflow_total" "$OUT/ours$r.metrics" 2>/dev/null | head -1 |
          awk '{print $NF}' || true)
  printf '%-5s %9s %11s %9s %6s %9s %10s %9s %8s\n' \
    "$r" "$res" "$lk" "$mi" "$rb" "$mu" "$mm" "${bytes:--}" "${ovm:-$ov}"
done
read -r ring_blocks ring_mib < "$OUT/ours1.ring"
echo "ring (ours): ${ring_blocks} blocks = ${ring_mib} MiB resident in vram_kv, ${RESIDENT} tokens"
echo "vram_kv:     ours $(cat "$OUT/ours1.vram_kv")   stock $(cat "$OUT/stock1.vram_kv")"
echo "experts:     ours $(awk '{print $1 " GiB (" $2 "%)"}' "$OUT/ours1.experts") of the elastic region,"
echo "             stock $(awk '{print $1 " GiB (" $2 "%)"}' "$OUT/stock1.experts")"

# ---- both arms, per round ---------------------------------------------------------------------
echo
if [ "$SPEC" -gt 0 ]; then
  echo "== both arms, per round (ms/step from the engine's decode step clock; delta = ours over stock) =="
else
  echo "== both arms, per round (ms/step from predicted_per_second; delta = ours over stock) =="
fi
printf '%-5s %-6s %10s %10s %9s %10s %10s\n' round first ms_stock ms_ours delta% stock_tps ours_tps
for r in $(seq 1 "$ROUNDS"); do
  sms=$(awk -v r="$r" '$1 == r && $2 == "stock" {print $4}' "$OUT/starts.tsv")
  oms=$(awk -v r="$r" '$1 == r && $2 == "ours"  {print $4}' "$OUT/starts.tsv")
  stp=$(awk -v r="$r" '$1 == r && $2 == "stock" {print $7}' "$OUT/starts.tsv")
  otp=$(awk -v r="$r" '$1 == r && $2 == "ours"  {print $7}' "$OUT/starts.tsv")
  first=$(awk -v r="$r" '$1 == r && $3 == 1 {print $2}' "$OUT/starts.tsv")
  awk -v r="$r" -v f="$first" -v s="$sms" -v o="$oms" -v st="$stp" -v ot="$otp" 'BEGIN{
    printf "%-5s %-6s %10.3f %10.3f %+8.2f%% %10.2f %10.2f\n", r, f, s, o, 100*(o/s-1), st, ot }'
done

# ---- speculation: per start, because acceptance is the bimodal variable ----------------------
if [ "$SPEC" -gt 0 ]; then
  echo
  echo "== speculation, per start (acceptance per start: stock starts can be bimodal) =="
  printf '%-7s %-6s %8s %9s %8s %6s %14s %7s %7s\n' \
    start arm ms_step tok/s tok/step steps accepted/drafted accept% p/m/d
  while read -r rnd arm ord ms pre ttft dtps dwall ptok ctok cached wall \
                   steps tokstep dnum dacc dpct spf smx spec; do
    printf '%-7s %-6s %8.4f %9.2f %8.3f %6s %14s %7.1f%% %s/%s/%s\n' \
      "$arm$rnd" "$arm" "$ms" "$dtps" "$tokstep" "$steps" "$dacc/$dnum" "$dpct" \
      "$spf" "$smx" "$steps"
  done < "$OUT/starts.tsv"

  echo
  echo "== draft acceptance per arm, sorted (two clusters here = a bimodal winner) =="
  for arm in stock ours; do
    list=$(awk -v a="$arm" '$2 == a { printf "%.1f ", $17 }' "$OUT/starts.tsv")
    stats=$(awk -v a="$arm" '$2 == a { v[n++] = $17; s += $17 }
      END { if (!n) { print "no starts"; exit }
            m = s/n; mn = v[0]; mx = v[0]
            for (i = 0; i < n; i++) { if (v[i] < mn) mn = v[i]; if (v[i] > mx) mx = v[i] }
            ss = 0; for (i = 0; i < n; i++) ss += (v[i]-m)^2
            printf "n=%d  mean %.1f%%  min %.1f%%  max %.1f%%  sd %.2f points", n, m, mn, mx,
                   (n > 1 ? sqrt(ss/(n-1)) : 0) }' "$OUT/starts.tsv")
    printf '%-6s %s   %s\n' "$arm" "$list" "$stats"
  done

  echo
  echo "== paired tok/s (secondary: under speculation the arms do not emit the same tokens, so a"
  echo "   tok/s ratio mixes step time with tokens a step and is not a speed ratio)"
  awk '$2 == "stock" { s[$1] = $7 } $2 == "ours" { o[$1] = $7 }
    END { for (r = 1; r <= '"$ROUNDS"'; r++) { d = 100*(o[r]/s[r]-1); n = r;
          t += d; if (r == 1 || d < mn) mn = d; if (r == 1 || d > mx) mx = d }
          printf "  mean %+.2f%%, range %+.2f%% .. %+.2f%%\n", t/n, mn, mx }' "$OUT/starts.tsv"
fi

# ---- arm summaries ----------------------------------------------------------------------------
echo
echo "== per-arm means over the run =="
printf '%-7s %12s %10s %12s %12s %10s\n' arm prefill_tps TTFT_s decode_tps decode_wall_s ms_step
for arm in stock ours; do
  awk -v a="$arm" '$2 == a { p+=$5; t+=$6; d+=$7; w+=$8; m+=$4; n++ }
    END { printf "%-7s %12.1f %10.3f %12.2f %12.1f %10.3f\n", a, p/n, t/n, d/n, w/n, m/n }' \
    "$OUT/starts.tsv"
done

# ---- paired statistics ------------------------------------------------------------------------
awk -v a=stock '$2 == a { s[$1] = $4 } $2 == "ours" { o[$1] = $4 } END {
  for (r = 1; r <= '"$ROUNDS"'; r++) printf "%.6f\n", 100*(o[r]/s[r] - 1) }' \
  "$OUT/starts.tsv" > "$OUT/deltas.txt"

read -r n mean sd sem ci95 ci90 min max mdb n1 n09 < <(
  awk -v t95="$(tval $(( ROUNDS - 1 )) 95)" \
      -v t90="$(tval $(( ROUNDS - 1 )) 90)" '
    { d[NR] = $1; s += $1 }
    END {
      m = s/NR; ss = 0
      for (i = 1; i <= NR; i++) ss += (d[i] - m)^2
      sd = NR > 1 ? sqrt(ss/(NR-1)) : 0
      sem = sd/sqrt(NR)
      mn = d[1]; mx = d[1]
      for (i = 2; i <= NR; i++) { if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i] }
      a1 = (2.8*sd/1.0)^2; a9 = (2.8*sd/0.9)^2
      r1 = int(a1) + (a1 > int(a1) ? 1 : 0); if (r1 < 2) r1 = 2
      r9 = int(a9) + (a9 > int(a9) ? 1 : 0); if (r9 < 2) r9 = 2
      printf "%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.1f %.1f\n",
             NR, m, sd, sem, t95*sem, t90*sem, mn, mx, 2.8*sd/sqrt(NR), r1, r9
    }' "$OUT/deltas.txt")

echo
echo "== the paired deltas (ours against stock, same round; + = ours slower) =="
r=1
while [ "$r" -le "$ROUNDS" ]; do
  printf 'round %-3s %+7.2f%%\n' "$r" "$(sed -n "${r}p" "$OUT/deltas.txt")"
  r=$(( r + 1 ))
done
echo "paired mean  $mean%   sd $sd   sem $sem   range $min% .. $max%"
echo "95% CI       $mean ± $ci95   (t, $(( ROUNDS - 1 )) df)  ->  $(awk -v m="$mean" -v c="$ci95" \
     'BEGIN{printf "%.2f%% .. %.2f%%", m-c, m+c}')"
echo "90% CI       $mean ± $ci90"
echo "unpaired:    $(awk '$2 == "stock" { s += $4; ns++ } $2 == "ours" { o += $4; no++ }
     END { printf "%+.2f%%", 100*((o/no)/(s/ns) - 1) }' "$OUT/starts.tsv") (meaning what pairing bought or cost)"
echo
# Two readings of the same n. The run's own interval is the honest resolution: at n=2 the sd has one
# degree of freedom, so the t interval is wide and the sd is barely estimated. The sd-based minimum
# detectable difference at a cross-arm sd of 1.4 points is what the header's round counts come from,
# so both are printed rather than whichever flatters this run.
echo "at n=$n this run's interval is ±$(awk -v c="$ci95" 'BEGIN{printf "%.2f%%", c}') (95%), and from its own"
echo "  sd ($sd points) the minimum detectable difference would be $(awk -v m="$mdb" 'BEGIN{printf "%.2f%%", m}')"
echo "  (2.8 x sd / sqrt(n), 95% two-sided, 80% power) -- with $(( ROUNDS - 1 )) df read the interval, not the sd."
echo "  At that cross-arm sd of 1.4 points: n=2 detects 2.8%, n=4 2.0%, n=16 1.0%, n=19-20 0.9%."
echo "  This run's own sd would need $(awk -v n="$n1" 'BEGIN{printf "%.0f", n}') paired rounds for 1% and $(awk -v n="$n09" 'BEGIN{printf "%.0f", n}') for 0.9%."
awk -v m="$mean" -v c="$ci95" 'BEGIN{
  where = (m - c > 0 || m + c < 0) ? "excludes" : "includes"
  printf "reading: %+.2f%% %s zero (95%%): ", m, where
}' 
if awk -v m="$mean" -v c="$ci95" 'BEGIN{ exit !(m - c > 0 || m + c < 0) }'; then
  echo "ours is slower by a margin this run can resolve"
else
  echo "the gap is inside this run's noise; state it as a band, and raise ROUNDS to narrow it"
fi
if [ "$SPEC" -gt 0 ]; then
  echo
  echo "speculation read: the sd above carries the draft-acceptance mix, which is the bimodal"
  echo "  variable (81.7% and 98.6% on different stock starts). The acceptance table is the"
  echo "  evidence on whether the modes split evenly across the arms; whichever arm drew the high"
  echo "  mode more often has part of this delta from acceptance, not from streaming."
fi
echo "wall: ${ELAPSED} s for $(( ROUNDS * 2 )) starts; raw under $OUT"
# A container that is being removed still shows in `docker ps` for a moment after `docker inspect`
# starts failing, so let the count settle instead of reporting a leak that is not there.
LEFT=1
for i in $(seq 1 20); do
  LEFT=$(docker ps -q | wc -l)
  if [ "$LEFT" -eq 0 ]; then break; fi
  sleep 0.5
done
echo "leftover containers: $LEFT"

exit 0
