#!/bin/sh
# Compare every matching op against libref on the operands the device kernel was handed.
#
# rad-kbench tests one op at a time on operands it invents (dense, rank 2, freshly filled). The
# engine's operands are row slices of a max_tok buffer, column slices of a fused projection, rank-3
# views and weights loaded from a file; a kernel can be right at every kbench shape and wrong at
# the one the model issues. This runs the host row beside the device one on every issue and
# reports per operand.
#
# Cost: two device-to-host copies of every operand (a fused gate_up weight is 170 MiB), a host
# synchronisation per issue on every lane and a slow reference kernel. Not a performance
# measurement; keep MAXN small and the prompt short. A routed expert GEMM is the most expensive:
# its weight operand is the layer's whole expert table, snapshotted twice plus the inverted copy,
# some gigabytes per issue, so MAXN=2 is a sensible sweep there.
#
# Large containers: a 166 GiB container with a 12 GiB host pool can be OOM-killed part way
# through. That is a host limit, not a defect; the verdicts printed before it died are valid.
# Lower HOSTPOOL or window the sweep with FROM/TO. An operand over RADIANCE_OP_ORACLE_MAXBYTES
# (4 GiB), such as Flash-Next's 47.68 GiB PLE n-gram table, is reported NOT CHECKED with its name
# and size and the sweep continues; raise the cap if the host can hold three copies.
#
# Relayouted weights: a kernel that claims a weight may relayout its canonical planes at load (e.g.
# libr4d's fp8a8 weights in WMMA fragment order). The oracle maps them back through the device
# row's RadUnrelayoutFn and decodes them before comparing; a row that relayouts without publishing
# an inverse is reported NOT CHECKED by name, which is a coverage gap, not a pass. The fp8 rows,
# the int8 (i8a8) block rows, the MoE 4-bit expert plane and its E4M3 scale views (both grouped
# GEMMs, every class) and the dense w4 rows are checked; the w8a16, w8a8, w2a8 and mxfp4 rows are
# not.
#
# `gemm_w2a8_nt` (the 2-bit draft head on every speculating container: mtp.draft_head.weight,
# dspark.draft_head.weight, dflash.draft_head.weight) is a missing zero point, not a missing
# inverse. It uses an unsigned asymmetric grid, value = (code - zero) * scale per (row, group of
# 128), and the zero plane lives in `b_scale` beside the scale (r4d.w2a8.sz.g128, a packed pair a
# dword). The reference reads `b` as codes and `b_scale` as a symmetric scale, so closing this
# needs a zero point in `gemm_nt_q`'s schema and an asymmetric reader in libref. r4d_selftest's
# gemm_w2a8_nt case covers it against a host decode of the canonical planes, and since the draft
# head only proposes tokens the target verifies, an error here costs acceptance, never output.
#
# Collectives are checked only when both ranks arm on the same issue. libref's all_reduce is a
# rendezvous: it waits on a barrier for every rank's message. Each rank arms on its own issue
# counter and spends its own budget, so the ranks must arrive together. Unfiltered, about one
# collective in ten is refused over a tp2 decode. A narrow OPS list makes it worse: the ranks do
# not issue the same named ops (an expert-parallel split alone ensures that), so one rank arrives
# armed and the other does not, giving
#
#   oracle #609 op 'all_reduce': the host kernel REFUSED ... with device error
#   oracle #620 op 'all_reduce': the host kernel REFUSED ... with shape this kernel does not serve
#
# (the 30-second barrier timeout, then the peer's stale message length). Neither is a finding about
# all_reduce. Sweep collectives with OPS=*, or at TP=1 where world_size is 1 and the reference
# answers in place; the two-rank wire has its own tests (tests/wire_test.cpp and the tp2
# determinism gate).
#
# Ops with no libref row report "0 op(s) checked, N skipped for want of a reference kernel": the five
# QSA ops (qsa_block_key, qsa_score, qsa_select, qsa_work, qsa_tail_store), dflash_conv,
# gate_quant_fp8, router_topk_scatter, qk_norm_rope_gate and ar_rmsnorm_quant_fp8. Each has a
# numeric case in r4d_selftest (the QSA ops against f64 evaluations of their formulae, a host sort
# and a host walk; the four fusions byte-identical against the unfused pair they replace). What is
# missing is only the check on the engine's own operands.
#
# Coverage across Flash-Next (four-bit MoE, QSA, PLE), Qwen3.6-35B-A3B (fp8 MoE), the 27B dense
# with the DFlash2 drafter and MiniCPM5-2B with DSpark: 42 distinct op types, every comparison
# made agrees. Every speculating container leaves NOT CHECKED only `gdn_recurrent_update` under
# the anchor state form (covered by r4d_selftest) and the w2a8 draft head weight above.
#
# At MAXN=400 the budget is spent inside the prefill of a deep model; reach the drafter and the
# sampler chain with FROM past the prefill or an OPS list that skips the GEMMs. Run both for a
# full picture.
#
#   OPS=a,b   op names, or * for every op        MAXN=8    stop after this many checks
#   FROM/TO   issue-index window                 TOL=0.02  relative tolerance
#   M=path    container                          PROMPT/MT the request to run
P="${P:-8302}"
M="${M:-$HOME/models/rad/minicpm5-2b-dspark.v2.rad}"
pkill -f "port $P"; while pgrep -f "port $P" >/dev/null; do sleep 1; done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
cd "$BUILD" || exit 1
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 \
    RADIANCE_OP_ORACLE="${OPS:-*}" \
    RADIANCE_OP_ORACLE_MAX="${MAXN:-8}" \
    RADIANCE_OP_ORACLE_FROM="${FROM:-0}" \
    RADIANCE_OP_ORACLE_TOL="${TOL:-0.02}" \
    ${TO:+RADIANCE_OP_ORACLE_TO=$TO} \
nohup ./bin/radiance --model "$M" \
    --radiance-home radiance_home \
    --host 127.0.0.1 --port "$P" \
    --max-num-seqs 1 --max-model-len "${MLEN:-8192}" --tp "${TP:-1}" \
    --gpu-headroom-mib "${HR:-512}" --kv-cache-dtype "${KVD:-bf16}" \
    --max-num-batched-tokens "${MNBT:-512}" $EXTRA \
    > "$HOME/oracle.log" 2>&1 &
for i in $(seq 1 90); do
  c=$(curl -s -m 3 -o /dev/null -w %{http_code} "http://127.0.0.1:$P/health" 2>/dev/null)
  [ "$c" = 200 ] && break
  pgrep -f "port $P" >/dev/null || { echo "DIED AT LOAD"; grep -E "^E " "$HOME/oracle.log" | tail -6; exit 1; }
  sleep 4
done
[ "$c" = 200 ] || { echo "NO START"; grep -E "^E " "$HOME/oracle.log" | tail -6; exit 1; }
curl -s -m 900 "http://127.0.0.1:$P/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":\"${PROMPT:-The capital of France is}\",\"max_tokens\":${MT:-2},\"temperature\":0}" \
  | head -c 300
echo
pkill -f "port $P"; while pgrep -f "port $P" >/dev/null; do sleep 1; done
echo "=== verdicts ==="
# Per-operand detail stays in the log. A MISMATCH line names the op and the kernel; NOT CHECKED
# lines are the coverage gaps.
grep -E "oracle #|op oracle|NOT CHECKED" "$HOME/oracle.log" | sed 's/^W //'
