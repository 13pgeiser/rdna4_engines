#!/bin/sh
# Qwen3.8-Flash-Next prefill under rocprofv3 kernel tracing, with the link counters beside it.
#
# `moe_gemm_q` is the largest item in a prefill step, and it mostly waits on non-resident expert
# weight, which for a host-pooled unit is read across the link inside the kernel. The trace gives
# the GEMM's time and /stats the bytes streamed; together they give the achieved streaming rate,
# which shows whether the kernel reaches the link's ceiling.
#
# tsum.sh keys its steady-state window on a once-per-decode-step marker; with max_tokens 1 there is
# none, so it sums the whole trace and every number is per run. The model load's kernels are in the
# total, so the prompt must be long enough to make them negligible.
#
#   N=20000 OUT=<trace dir> scripts/fntrace.sh
#
# The default container is the production one, w4nl64-i8lm.rad (see scripts/fnserve.sh).
P="${P:-8164}"
M="${M:-$HOME/models/rad/w4nl64-i8lm.rad}"
N="${N:-20000}"
BPT="${BPT:-4}"
OUT="${OUT:-$HOME/traces/fnpf}"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CORPUS=/dev/shm/fntrace-corpus.txt
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1

want=$((N * BPT + 4096))
if [ "$(wc -c < "$CORPUS" 2>/dev/null || echo 0)" -lt "$want" ]; then
    : > "$CORPUS"
    while [ "$(wc -c < "$CORPUS")" -lt "$want" ]; do
        cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$CORPUS" 2>/dev/null || exit 1
    done
fi

pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 $EXTRA_ENV \
  "$ROOT/libr4d/isa/trace.sh" "$OUT" -- \
  ./bin/radiance --model "$M" \
    --radiance-home radiance_home \
    --host 127.0.0.1 --port "$P" \
    --max-num-seqs ${SEQS:-8} --max-model-len ${MLEN:-200000} \
    --tp ${TP:-2} --placement expert_tiered --host-pool-mib ${HOSTPOOL:-12288} \
    --gpu-headroom-mib ${HR:-96} --kv-cache-dtype ${KVD:-fp8} \
    --num-speculative-tokens ${NS:-3} \
    --max-num-batched-tokens ${MNBT:-2048} \
    $EXTRA > "$HOME/fntrace.log" 2>&1 &
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED"; grep -iE "^E |error" "$HOME/fntrace.log" | tail -10; exit 1; }
  sleep 4
done
[ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:$P/health)" = 200 ] || { echo "DID NOT START"; exit 1; }

st() { curl -s -m 5 "http://127.0.0.1:$P/stats" | jq -r ".link.$1 // 0"; }
s0=$(st stream_bytes); a0=$(st ar_bytes); h0=$(st h2d_bytes)
{   printf 'trace %s\n\n' "$(date +%s%N)"; head -c $((N * BPT)) "$CORPUS"; } > /dev/shm/fntrace-p.txt
jq -Rs '{prompt: ., max_tokens: 1, temperature: 0}' < /dev/shm/fntrace-p.txt > /dev/shm/fntrace-b.json
t0=$(date +%s.%N)
out=$(curl -s -m 1800 -X POST "http://127.0.0.1:$P/v1/completions" \
          -H 'Content-Type: application/json' --data-binary @/dev/shm/fntrace-b.json)
t1=$(date +%s.%N)
s1=$(st stream_bytes); a1=$(st ar_bytes); h1=$(st h2d_bytes)
pt=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens // 0')

# Under rocprofv3 the server can hang in finalisation after SIGTERM (the profiler chains its own
# handler). The trace CSV is already on disk by then, so escalate to SIGKILL after a bounded wait.
pkill -f "bin/radianc[e]"
i=0; while pgrep -f "bin/radianc[e]" >/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done
pgrep -f "bin/radianc[e]" >/dev/null && { echo "did not exit on TERM under the profiler, killing"; pkill -9 -f "bin/radianc[e]"; sleep 2; }
awk -v t0="$t0" -v t1="$t1" -v pt="$pt" -v s="$((s1 - s0))" -v a="$((a1 - a0))" -v h="$((h1 - h0))" 'BEGIN {
    d = t1 - t0;
    printf "\nprefill %d tok in %.3f s = %.1f tok/s\n", pt, d, pt / d;
    printf "  streamed   %10.2f GB over the request  (%.2f GB/s over its wall, ALL ranks)\n", s/1e9, s/1e9/d;
    printf "  promotions %10.2f GB   all-reduce %8.2f GB\n", h/1e9, a/1e9;
    printf "\nDivide the streamed GB by moe_gemm_q device time below for the rate the read achieves.\n";
}'
sh "$ROOT/libr4d/isa/tsum.sh" "$OUT/k_kernel_trace.csv"
