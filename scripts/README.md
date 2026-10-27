# scripts/

Harnesses that drive a running engine: bring a model up, measure it, and assert on what comes back
over the wire. They are versioned with the engine so measurements stay reproducible, and each
header records its method (why a window is sampled the way it is, which counter to trust), which is
hard to reconstruct from the number alone.

Scripts that start a server or a conversion run the build at `$BUILD`, by default `build/` in the
checkout the script is in, and default to containers under `$HOME/models/rad/` and checkpoints under
`$HOME/models/<org>/<name>`. A script that starts a server first stops any `radiance` already
running (`oracle.sh` only the one on its own port) and logs to `$HOME/<script>.log`. Most defaults
are environment overrides (`M=` for the container, `P=` for the port, and so on); each script's
header lists its own. Scripts that only attach to a live server need `curl` and `jq`; `cardclock.py`,
`agentreplay.py` and `stampgaps.py` need `python3`. Scripts find each other through `$(dirname "$0")`.

## Bringing a model up

| | |
|---|---|
| `mcserve.sh` | MiniCPM5-2B with its DSpark drafter (`minicpm5-2b-dspark.v2.rad`), left running on :8100. Records why `--max-model-len` is the checkpoint's `n_ctx_train` (131072) and what a token of KV costs. |
| `fnserve.sh` | Qwen3.8-Flash-Next, the production container `w4nl64-i8lm.rad`, left running on :8100. Records the production flags and their reasons: the 200K context and how the expert/cache split sizes the pool for it, the 2048-token chunk, the host prefix-cache tier and the lossy `wht6` cross-rank wire. |
| `fnstop.sh` | Stops whichever server is running and waits for the process to exit. |
| `convdspark.sh` | Rebuilds the container `mcserve.sh` serves, with the drafter merged, as `minicpm5-2b-dspark.new.rad` beside it. The checkpoint and drafter paths are fixed; `RECIPE=` overrides the recipe. |
| `convfn.sh` | Rebuilds the container `fnserve.sh` serves, as `w4nl64-i8lm.new.rad` beside it, from `data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe`. Needs `CALIB=<dir>`: the recipe's expert rule runs GPTQ against the calibration Grams there (`docs/MOE-W4.md`). |
| `hfget.sh` | Fetches a Hugging Face repo with `curl` and `jq`. Resumable, and skips the duplicate weight formats a repo often carries beside its safetensors. `radiance --model <repo id>` runs it. |
| `krea2_native.py` | Rewrites a Krea 2 DiT single file in Krea's own key names (`turbo.safetensors`, civitai and ComfyUI finetunes) as diffusers' transformer file, bytes unchanged, so it converts as the `transformer/` of a Krea 2 pipeline directory (`docs/GUIDE.md`). |
| `krea2_rebalance_calib.py` | Measures the constant of each Krea 2 rebalance preset (`rebalance`: balanced, detail, subtle): the taps' mean squares over twenty prompts through a pipeline's text encoder, and the renormalising factor arch/krea2 holds for them (`docs/GUIDE.md`). Needs torch and transformers. |

## Measuring

| | |
|---|---|
| `cstep2.sh` | Steady-state decode ms/step at each concurrency, against a live server. Waits until exactly C sequences are running before sampling and re-checks at the end, since a window with fewer than C at either edge measures no single concurrency. Model-agnostic (`P=` picks the port). |
| `dprofd.sh` | Per-op device timing (`--profile-ops`) at depth and concurrency on the 27B dense model with its DFlash2 drafter (`qwen3.8-27b-fp8.rad`); starts its own server on :8129 and needs a ~16K-token prompt (`PROMPT=<file>`). The counters are cumulative and a run starts with prefill, so one dump mixes prefill and decode; difference two dumps taken after prefill has ended. |
| `fnpf.sh` | Prefill throughput of the server on `$P`, over the repository's own markdown, with a fresh nonce per rep so the prefix cache cannot serve it. Prints card temperature and clock beside every rep. |
| `agentreplay.py` | Several agent conversations at once against a live server, each growing turn by turn to `--context` tokens and handing tasks to subagents that share its system prompt while it waits (`--rewrite`: a client that replaces its last message every turn). Prints per kind of turn how much of the prompt the cache served and how many continuing turns prefilled their history again, and the prefix-cache counters across the run. The conversations follow `--seed` and carry a fresh nonce, so two servers (one flag apart, restarted between) are sent the same work and share nothing with an earlier run. Takes the key from `--api-key-file`, never the command line. |
| `fnconc.sh` | What a long prefill costs the decoders running beside it, and the prefill's own time, measured together: alone, then contended, with a warm pass discarded. |
| `fntrace.sh` | A Flash-Next prefill under `rocprofv3`, on its own server, with the `/stats` link counters across the same request, giving the MoE GEMM's streamed-read rate. |
| `mctrace.sh` | A MiniCPM5-2B decode trace under `rocprofv3`, on its own server, formatted for `libr4d/isa/tsum.sh`, mainly for its `blocks` column. |
| `ident.sh` | Run-to-run reproducibility at a fixed configuration: one hash per (temperature, question), to compare against another run of the same configuration. A speculative run need not match a non-speculative one byte for byte (the split-K count is chosen from M, so reduction orders differ and a near-tie can flip late in a long answer), but a divergence within the first sentence or two is a defect. |
| `cardclock.py` | What holds each card's clock under sustained load, and which card a tensor-parallel step waits for (the slowest rank ends the step). Reads every AMD card's `gpu_metrics` table ten times a second and counts a throttler bit only when the clock is lower while it is set (some firmware keeps a bit on at idle). Shows the engine's step counters per window, so a slow window is attributed to slower steps, fewer tokens per step or host time between steps. Drives its own greedy load, or with `--watch` only samples a server under real traffic; `--compare` shows two saved runs side by side. Read-only on the cards. |
| `kv-resident-vs-stock.sh` | This build with `--kv-resident 32768` against the stock 1.10.0 image without it, in interleaved ABBA rounds of one cold start a request; the mean paired delta with per-round values and a t interval (`ROUNDS` sets the resolution, ~2.0 % at 4). Both arms must tokenize the 16 110-token prompt alike and report `cached_tokens` 0. |
| `kv-union.sh` | How large a ring must be: the union of distinct blocks one `kv_resolve` launch selects at 64k/128k-token prompts, the ring climbed from the 32 768-token floor until no launch overflows. Reports resolves, lookups, misses, the largest union and prefill time, and the overflow message verbatim; its last arm decodes a ~34k-token history past the floor ring against streaming off and passes only if the ring evicted. |
| `kv-parity.sh` | That a ring covering the group's context streams nothing and changes no output byte: a dense and a sparse prompt without the flag and with `--kv-resident 4096` (the whole `--max-model-len x --max-num-seqs`), compared byte for byte, with `--profile-ops` showing no `kv_resolve` issued. |
| `kv-multiturn-recall.py` | Whether planted facts survive twenty turns whose history outgrows the ring: the same transcript with streaming off and with `--kv-resident 32768` under `--max-model-len 65536`, a code planted at turn 2 and a surname at turn 5 asked at turn 20 (~49k prompt tokens). Fails if the off arm missed the cache on continuing turns or the on arm streamed nothing. |
| `kv-stage-parity-64k.sh` | The prefill stage: streaming off against `--kv-resident 32768` on a ~64k-token document, so every chunk past 32 768 tokens takes the one-layer stage; byte-identical output, the staged copies checked in layer order, and both arms' VRAM split printed. A run that takes the ring instead fails. |
| `statsdump.py` | Records `/stats` to a JSONL file, one row per poll (`--every`, default 1 s), so the dashboard's figures can be read back over time on a machine without a Prometheus scraper. A row is the engine's snapshot verbatim plus the poll's time `t` and the gap since the last good poll `dt_s`; nothing is derived. The file is appended to and rotated at `--max-mib` (default 256, five generations). A failed poll leaves no row; a run that wrote no row, or can no longer write its file, exits 1. The key is taken from `--api-key` or `RADIANCE_API_KEY`. |
| `stampgaps.py` | Reads the CSV that `RADIANCE_AQL_STAMP=N` writes (every dispatch of one recorded pass in N on card 0, start and end) and prints per kind of pass (told apart by tape length) the median wall, the busy time as the union over the queues, the idle share, the gap before an ordered start (p50, p90, sum) and the kernels by share of the dispatch time. The idle share is what chained launches could win at most, the gaps what they would remove. Stamping costs throughput of its own when passes are short (one in 50: 0.3 % at one GLM-5.3 stream, 4 % at eight), so take N larger at high concurrency. |
| `oracle.sh` | Every op compared against libref on the operands the model issued, which rad-kbench cannot do: its operands are its own (dense, rank 2), while the engine's are arena slices and weights loaded from a file. Starts its own server on :8302. Read it for `MISMATCH` lines, and for `NOT CHECKED` lines naming rows whose weight layout has no published inverse and so cannot be compared. |

## Regression checks against a live server

These cover what `tests/chat_test.cpp` cannot: the server, the wire format, and a real model's
output. `mcagent.sh` and `mcdecl.sh` start their own MiniCPM5-2B server through `mcserve.sh`.

| | |
|---|---|
| `fndrop.sh` | Clients that disconnect mid-request, at points derived from a timed full request: the server must stay up and keep answering a reference question identically. |
| `tierident.sh` | Whether a session that left VRAM comes back byte for byte, attention blocks and recurrent state both. The repeat from VRAM is the control: without it a differing hash proves nothing, since a cache hit could be tie-divergent. A run where nothing moved is a failure. On a hybrid model the linear-state checks are counters, not hashes: a lost snapshot does not show in the text, because the engine replays from further back and produces the same answer. Needs a server started with `--prefix-cache-mib 512`. |
| `tiercross.sh` | The same check with two sessions in flight, the only way to catch one being served the other's context or state. Compares each restore against that session's device-hit run rather than its cold one, because a hybrid session's reuse is clamped to the last reachable checkpoint and a cold run reuses nothing. |
| `mctools.sh` | The tool-call shapes hardest to parse: an array argument, two calls in one reply, prose before a call, plain chat with tools available. Asserts `finish_reason` and that no `<function`/`<param`/CDATA marker reaches content. |
| `mcstream.sh` | Tool-call replies streamed, asserting on raw SSE fragments rather than the assembled message. Streamed content cannot be retracted, so a repair applied at the end is not a fix. |
| `mcagent.sh` | What speculation is worth on an agentic workload: `/v1/chat/completions` with tools, a constrained decode. Speculation on and off alternate across restarts because the speculation depth is a server-level setting. |
| `mcdecl.sh` | How often the drafter proposes, by context length. `draft_acceptance_ratio` is conditional on having drafted, so it can look healthy while tokens per step collapse; blocks per step separates the two. |
