#!/bin/sh
# Prefill throughput of the server already running on $P.
#
# Prefill speed depends on the deployment's flags (chunk, tensor-parallel width, expert/cache
# split), so this attaches to a running server instead of starting its own; scripts/fnserve.sh
# starts one.
#
# The prompt is the repository's markdown, repeated to length. Random-letter words tokenize at
# about three tokens each and route through the experts differently from real text, which skews a
# MoE prefill measurement.
#
# Every rep starts with a different nonce so the prefix cache cannot serve it: the cache matches
# from the start of the prompt, so a trailing nonce would leave all but the last block a hit.
# `cached_tokens` is printed to show this.
#
# Prefill slows as the card heats, within and across runs, so temperature and sclk are printed
# beside every rep; a number without them is not comparable across runs.
#
#   P=8100 N=8000 REPS=3 scripts/fnpf.sh
#
# N is a token target reached through an assumed bytes-per-token; the rate uses the server's
# reported `prompt_tokens`, so the approximation does not affect it.
P="${P:-8100}"
H="${H:-127.0.0.1}"
N="${N:-8000}"
REPS="${REPS:-3}"
BPT="${BPT:-4}"                 # bytes of this corpus per token, measured at about 4.0
ROOT="${ROOT:-$(dirname "$0")/..}"
URL="http://$H:$P"
CORPUS="${CORPUS:-/dev/shm/fnpf-corpus.txt}"

# One corpus, built once and reused, so successive runs prefill the same bytes. It is grown when a
# run needs more: `head -c` on a short corpus silently returns less, and the run would report a
# rate for a shorter prompt than intended.
want=$((N * BPT + 4096))
if [ "$(wc -c < "$CORPUS" 2>/dev/null || echo 0)" -lt "$want" ]; then
    : > "$CORPUS"
    while [ "$(wc -c < "$CORPUS")" -lt "$want" ]; do
        cat "$ROOT"/spec.md "$ROOT"/docs/*.md >> "$CORPUS" 2>/dev/null || exit 1
    done
fi

card() {
    rocm-smi --showtemp --showclocks --csv 2>/dev/null |
        awk -F, 'NR>1 && $1 ~ /card/ { printf "%s ", $2 }' | head -c 48
}

stats() { curl -s --max-time 5 "$URL/stats"; }

echo "server $URL, target $N tokens, $REPS reps, corpus $(wc -c < "$CORPUS") bytes"
for r in $(seq 0 "$REPS"); do
    {   printf 'request %s %s\n\n' "$r" "$(date +%s%N)"
        head -c $((N * BPT)) "$CORPUS"
    } > /dev/shm/fnpf-prompt.txt
    jq -Rs '{prompt: ., max_tokens: 1, temperature: 0}' < /dev/shm/fnpf-prompt.txt \
        > /dev/shm/fnpf-body.json
    t0=$(date +%s.%N)
    out=$(curl -s --max-time 1800 -X POST "$URL/v1/completions" \
              -H 'Content-Type: application/json' --data-binary @/dev/shm/fnpf-body.json)
    t1=$(date +%s.%N)
    pt=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens // "ERR"')
    ct=$(printf '%s' "$out" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0')
    [ "$pt" = "ERR" ] && { echo "  failed: $(printf '%s' "$out" | head -c 300)"; exit 1; }
    awk -v t0="$t0" -v t1="$t1" -v pt="$pt" -v ct="$ct" -v r="$r" -v c="$(card)" 'BEGIN {
        d = t1 - t0;
        printf "  %-6s %7d tok  %6d cached  %8.3f s  %8.1f tok/s   %s\n",
               (r == 0 ? "warm" : "rep " r), pt, ct, d, pt / d, c;
    }'
done
