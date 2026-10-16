#!/bin/sh
# ident.sh -- server reproducibility check. Prints one hash per (temperature, question) for
# comparison against another run of the same configuration.
#
# It tests run-to-run determinism at a fixed configuration, across restarts: on
# Qwen3.8-Flash-Next at tp2 with the MTP head at depth 3, two fresh servers give the same six
# hashes (temperature 0, 0.7 and 1.0, both questions).
#
# It does not test that a speculative run matches a non-speculative one byte for byte.
# `narrow_ksplit` picks the fp8 GEMM finisher's split count from M, so a row computed at M=1 and
# at M=8 (a verify block) is not bit-identical, and a long answer can flip a near-tie past its
# first few hundred characters. `--deterministic` pins placement only and does not affect kernels.
#
# A short common prefix is a defect, not a tie. On Qwen3.8-Flash-Next at tp2, depth 3 and
# --num-speculative-tokens 0 give the same six hashes, and longer answers agree for hundreds of
# characters. A speculative run that departs within the first sentence or two is computing
# something different after a rejected draft (e.g. rolling state read from a rejected token) and
# is worth bisecting. Byte identity across that boundary would need a pinned split count in the
# fp8 GEMM, at a throughput cost for one M or the other.
#
#   P=8100 scripts/ident.sh          # against a server already up
P=${P:-8100}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ident.XXXXXX"); trap 'rm -rf "$WORK"' EXIT INT TERM
ask() {
  jq -n --arg p "$2" --argjson t "$1" \
     '{model:"m",messages:[{role:"user",content:$p}],max_tokens:200,temperature:$t,seed:12345,
       chat_template_kwargs:{enable_thinking:false}}' > "$WORK/iq.json"
  curl -s -m 900 http://localhost:$P/v1/chat/completions -H "Content-Type: application/json" \
       -d @"$WORK/iq.json" | jq -r '.choices[0].message.content' | sha256sum | cut -c1-16
}
Q1="Implement an LRU cache in Go with a fixed capacity and a small test."
Q2="Explain why merge sort is O(n log n), briefly."
for T in 0 0.7 1.0; do
  printf "temp %-4s  %s  %s\n" "$T" "$(ask $T "$Q1")" "$(ask $T "$Q2")"
done
