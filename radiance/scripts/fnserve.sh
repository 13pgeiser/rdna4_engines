#!/bin/sh
# Start Qwen3.8-Flash-Next as a long-lived server: wait for /health, print the endpoints and exit
# with the server still up. The default container is the production w4nl64-i8lm.rad, built by
# data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe (scripts/convfn.sh).
#
# Context: --max-model-len reserves nothing per request; it sets the worst case the paged pool is
# sized against. Under expert_tiered the routed experts keep on the card what the host pool cannot
# hold, the cache takes the rest of the elastic pool (up to --max-num-seqs x --max-model-len), and
# blocks no live request holds are lent back to the expert slab. --host-pool-mib moves that split:
# a larger pool moves experts to pinned host RAM and gives the cache more VRAM.
#
# The pool is a capacity, not a per-request cap. A token costs 7.31 KiB across the two paged
# groups (kv_attn 26.00 KiB and kv_qsa_bkey 3.25 KiB per 4-token block), so one 200K sequence is
# 1.43 GiB. With prefix caching a finished sequence's blocks stay held for the next turn, so demand
# is the live sessions: a context of T tokens occupies about 1.8 x T pool tokens. If the pool
# cannot hold them, the cache evicts what the next turn needs and every turn re-prefills its whole
# context (0% prefix hit, minutes per turn). Watch kv_util; the hit rate drops sharply as the pool
# fills.
#
# Defaults are the production shape (gpu-headroom-mib 96, kv-cache-dtype fp8,
# max-num-batched-tokens 2048, tp 2, speculation depth 3), with the prefix cache on and
# --checkpoint-slots at its default, as multi-turn chat needs both.
#
# Chunk 2048: `--max-num-batched-tokens` is the prefill chunk when checkpoints are off and sets how
# many routed rows the MoE GEMM sees at once. Part of the expert weight is pinned host memory read
# over the link inside the kernel, so a smaller chunk re-reads every expert for fewer tokens; 2048
# prefills about 55% faster than 512.
#   - 4096 does not give a 4096 chunk: plan_chunk splits at --checkpoint-interval (2048) first,
#     while the activation arena grows with the budget and is charged before the expert/KV split,
#     costing resident experts for nothing. Startup names which limit bound the chunk. A real 4096
#     chunk (--checkpoint-interval 4096) is no faster: the larger arena costs more resident experts
#     than the amortisation returns.
#   - Co-running decoders queued behind a prefill finish sooner at 2048 than at 512, since the
#     prefill does (scripts/fnconc.sh).
#   - The KV pool loses about 3% to the larger arena; two ~180K sessions still hit 99% with no
#     evictions.
#   - Decode step time rises about 0.6% at one stream (scripts/cstep2.sh) from the experts the
#     arena displaces; at eight streams the difference is below the harness's resolution.
#   - The chunk changes the text: a different batch shape sums prefill in a different order, so a
#     tie can resolve differently. Each chunk size is stable on its own; pin it before recording a
#     reference.
#
# Prefix cache: the engine's host tier is off by default (--prefix-cache-mib 0), so a conversation
# is dropped as soon as the pool needs its blocks. This deployment keeps finished turns in host
# memory for an hour of disuse (--prefix-cache-ttl's default). The elastic VRAM half needs no flag
# and cannot be turned off: the pool holds the blocks in use plus a buffer derived from the chunk,
# and the expert slab takes the VRAM it releases. That relies on --gpu-headroom-mib (96 above);
# startup names any shortfall and holds it back on top of the reserve.
#
#   --prefix-cache-mib  finished conversations are copied here at once and their VRAM goes to the
#                       expert slab; a return restores instead of re-prefilling, and a full tier
#                       evicts least recently used. It holds attention blocks only: linear-state
#                       snapshots live in the --checkpoint-slots store (113 MiB each on this model,
#                       counted for the whole server, not per sequence).
#
# The host tier is kept small because of the n-gram table: the tier's arena is pinned, and the
# 47.68 GiB n-gram (PLE) table is read through the container's mmap, so both draw on the same host
# RAM. Too large a tier turns a decode step's minor faults into major ones. Watch
# /proc/<pid>/stat majflt across a prefill before raising it.
#
# --tp-wire wht6: an all-reduce at or above --tp-wire-min-kb (128 KiB: every prefill chunk, and
# decode from seven streams up) carries a Walsh-Hadamard-rotated 6-bit payload instead of bf16,
# 2.6x fewer bytes over PCIe, inside the gated residual's fused writes. Against the exact wire at
# a 2048-token chunk it prefills about 6% faster and decodes about 1% faster at eight streams; one
# stream stays under the floor and is exact. It is lossy: over greedy continuations of 160 code and
# prose prompts, wht6 agrees with the exact wire at 91.4% of positions, the same as the E4M3
# hyper-connection matrices the container already carries versus bf16 (91.3%). WIRE=exact serves
# the exact wire.
#
# Binds 0.0.0.0 so other hosts can reach it. Stop it with scripts/fnstop.sh.
P="${P:-8100}"
M="${M:-$HOME/models/rad/w4nl64-i8lm.rad}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 2
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 $EXTRA_ENV \
nohup ./bin/radiance --model "$M" \
    --radiance-home radiance_home \
    --host 0.0.0.0 --port "$P" \
    --max-num-seqs ${SEQS:-8} --max-model-len ${MLEN:-200000} \
    --tp ${TP:-2} --placement expert_tiered --host-pool-mib ${HOSTPOOL:-12288} \
    --gpu-headroom-mib ${HR:-96} --kv-cache-dtype ${KVD:-fp8} \
    --prefix-cache-mib ${KVHOST:-4096} \
    --num-speculative-tokens ${NS:-3} \
    --max-num-batched-tokens ${MNBT:-2048} \
    --tp-wire ${WIRE:-wht6} \
    $EXTRA > "$HOME/fnserve.log" 2>&1 &
echo "starting, log: ~/fnserve.log"
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED after ${i}x4s"; grep -iE "^E |error|refus" "$HOME/fnserve.log" | tail -20; exit 1; }
  sleep 4
done
[ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] || {
  echo "DID NOT START in 10 min"; grep -iE "^E |error" "$HOME/fnserve.log" | tail -20; exit 1; }
IP=$(ip -4 -o addr show scope global | awk '{split($4,a,"/"); print a[1]; exit}')
echo "UP  pid $(pgrep -f "bin/radianc[e]" | tr "\n" " ")"
echo "  completions  http://$IP:$P/v1/completions"
echo "  chat         http://$IP:$P/v1/chat/completions"
echo "  dashboard    http://$IP:$P/"
echo "  metrics      http://$IP:$P/metrics"
grep -iE "serving on|listening on|KV cache|placement" "$HOME/fnserve.log" | tail -8
