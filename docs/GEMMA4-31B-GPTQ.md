# The Gemma 4 31B at w4a8 by GPTQ

The 31B's w4a8 container (`data/recipes/gemma4-31b-dense-bf16-w4a8-df2.recipe`) rounds the trunk's
four linears to the nearest int4 on a group-128 grid. That moves the model's distribution most
where a reply is most constrained. After a string argument in a tool call, `sku:<|"|>MS-110`, the
bf16 model closes the string (`<|"|>`, probability ~1.0); the rounded trunk prefers `**,` (58%
against 41%), so a streamed call stalls inside the string and the client never sees its end.

`gemma4-31b-dense-bf16-w4a8g-df2.recipe` makes the same container (the same grid, kernels and size)
with the int4 codes chosen by GPTQ against the bf16 model's own activations.

## The route

1. **The Hessians, from the bf16 checkpoint.** arch/gemma4 serves the 31B's bf16 checkpoint as it
   ships, and with `RADIANCE_CALIB_DIR` set it sums the Gram of every dense linear's bf16 input over
   the corpus it is fed. The files carry the names the w4a8 container gives the weights
   (gemma4_bf16's `blk.N.attn_qkv`, `attn_o`, `ffn_gate_up`, `ffn_down`), so the recipe reads them
   as they are. o_proj's and down's inputs are split across ranks at `--tp 4`, and each rank writes
   its block (`.part`), which libquant places on the diagonal.
2. **The convert.** `CALIB=DIR rad-convert` with the recipe: libquant's `gptq` on the trunk's four
   linears, everything else as the -df2 recipe makes it. libquant factors each Hessian through
   OpenBLAS's LAPACK where the build finds OpenBLAS (the container image does once libopenblas-dev
   is in its build stage, a separate change); without it, libquant's own single-threaded loops take
   hours for each 21504-wide down projection.

The w4a8 container's own plugin cannot record them: its linears read int8 activations, which
`gram_accum` does not take, and a quantised container's activations would come through its
quantisers.

## The corpus

Whole conversations, rendered with the model's own chat template (Hugging Face's
`apply_chat_template`, thinking off) as token ids, so the Hessians see the reply format the server
writes: 2.78M tokens in 4,724 conversations, each at most 8192 tokens, of which a tenth is held out
and the other 2.47M tokens (4,252 conversations) is the calibration corpus.

| share | tokens | conversations | from |
|---|---|---|---|
| tool calls | 834K | 2,627 | glaive-function-calling-v2 (Apache-2.0): its functions as tools, its calls as `tool_calls` (the template writes them as `<\|tool_call>call:NAME{k:<\|"\|>v<\|"\|>}<tool_call\|>`), its function responses as tool turns |
| chat | 695K | 730 | UltraChat 200k (MIT) prompts with the served 31B's greedy replies (the round-to-nearest w4a8 container's) |
| code | 417K | 589 | self-oss-instruct-sc2 (ODC-BY) prompts, the 31B's replies |
| math | 417K | 493 | OpenMathInstruct-2 (CC-BY-4.0) prompts, the 31B's replies |
| reasoning | 417K | 285 | rStar-Coder (CC-BY-4.0) prompts, the 31B's replies |

The held-out tenth (305K tokens) is never fitted to; 44.5K tokens of it are the KL mode's corpus
below. BetterBench's corpora, on which the stall was found and is measured, are in neither.

## Running it

```sh
# 1. The Hessians: three slices of 20 layers (60 layers' Grams, ~364 MiB a rank each at --tp 4,
#    do not fit beside the bf16 weights), the corpus through prefill once per slice.
for sl in 0:20 20:40 40:60; do
  RADIANCE_CALIB_DIR=/calib RADIANCE_CALIB_TOKENS=2300000 RADIANCE_G4_CALIB_LAYERS=$sl \
    radiance --model gemma-4-31B-it --tp 4 --no-prefix-cache --num-speculative-tokens 0 \
      --max-num-batched-tokens 8192 --max-num-seqs 8 --max-model-len 10240 --kv-cache-dtype bf16
  # ...feed the corpus at max_tokens 1 until every rank prints
  #    radiance: gemma4: rank N wrote the Hessians of layers lo:hi after ... tokens to /calib
done

# 2. The container.
CALIB=/calib rad-convert gemma-4-31B-it --draft-model gemma-4-31B-it-speculator.dflash2 \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8g-df2.recipe \
    --set quantization_config.quant_method=compressed-tensors -o gemma4-31b-dense-w4a8g-df2-qkv.rad
```

On four R9700s each slice took about 25 minutes, the corpus through prefill at ~1,500 tokens a
second with the taps live, and the three wrote 45 GB of Grams. The convert took 75 minutes on a
48-thread EPYC.

## Measuring it

- **KL against the bf16 model.** The engine's KL mode: the bf16 checkpoint (arch/gemma4) records its
  full distribution at every position of the held-out 44.5K tokens, and each container is scored
  against it with the same flags (`--tp 4`, bf16 KV). That reference is Hugging Face's: on 41 of
  its documents (21,407 positions), Transformers' bf16 31B is a mean 0.0195 nats from it (median
  0.00001), and at each container's 25 worst positions the two agree.
- **Where it counts: the model's turns.** At system and user positions, text the model never
  writes, the bf16 model's own guesses are untrained and often odd (` same` at 1.000 after
  `<|tool>declaration`, `---` at 0.997 inside a user's problem statement). Any quantisation moves
  them, so they carry most of the mean: of RTN's and Q8's 50 worst positions, 46 are system or
  user positions, each a confident bf16 guess that both implementations make. The tables give every
  position and the model's own turns apart.

## Results

The containers are the same size (24.9 GB) on the same kernels, so speed does not move; what
moves is how close each is to the bf16 model.

**The stall.** The next token after `sku:<|"|>MS-110` in BetterBench's order request, and
BetterBench's two order requests (a single call, and a round trip) each streamed 20 times, sampled
as BetterBench sends them, `--tp 2`, thinking off:

| | `<|"|>` (the string closes) | `**,` | single calls that stalled | round trips that stalled |
|---|---|---|---|---|
| bf16 (Hugging Face) | ~1.0 | | | |
| RTN (`-df2-qkv`) | 41% | 58% | 10 of 20 | 1 of 20 |
| **GPTQ** | **99.8%** | 0.2% | **0 of 20** | **0 of 20** |

**KL against the bf16 model**, the 44.5K held-out positions (Q8 is the w8a8 container, 38.2 GB):

**every position** (44456 positions)

| | mean KL | median | p99 | over 1 nat | top-1 |
|---|---|---|---|---|---|
| RTN | 0.988 | 0.0039 | 14.81 | 16.7% | 81.9% |
| GPTQ | 0.659 | 0.0020 | 11.15 | 12.8% | 84.5% |
| Q8 | 0.620 | 0.0014 | 12.08 | 11.7% | 85.6% |

**the model's turns** (31975 positions)

| | mean KL | median | p99 | over 1 nat | top-1 |
|---|---|---|---|---|---|
| RTN | 0.341 | 0.0003 | 7.49 | 6.7% | 90.5% |
| GPTQ | 0.251 | 0.0001 | 5.46 | 5.7% | 91.3% |
| Q8 | 0.226 | 0.0001 | 5.28 | 4.8% | 92.3% |

**the model's turns by source** (mean KL / top-1)

| | chat | math | oss | rstar | tool |
|---|---|---|---|---|---|
| RTN | 0.810 / 81.8% | 0.093 / 94.5% | 0.166 / 92.9% | 0.087 / 94.4% | 0.521 / 88.9% |
| GPTQ | 0.572 / 82.7% | 0.062 / 95.4% | 0.134 / 93.4% | 0.067 / 95.2% | 0.396 / 89.7% |
| Q8 | 0.536 / 84.7% | 0.046 / 95.9% | 0.130 / 94.3% | 0.043 / 95.8% | 0.356 / 90.8% |

GPTQ closes 78% of the distance from RTN to Q8 on the model's turns and 89% over every position, on
every source; on tool-call replies the mean falls from 0.521 to 0.396.

**The drafter** (DFlash2, block 8), specbench at 8 users, thinking off, tokens a step:

| | HumanEval | math | qa | question | rag | summarization | translation | all |
|---|---|---|---|---|---|---|---|---|
| RTN | 5.97 | 6.16 | 2.91 | 3.44 | 4.40 | 3.21 | 4.02 | 3.96 |
| **GPTQ** | 5.98 | 6.19 | 3.25 | 3.56 | 4.37 | 3.21 | 4.08 | **4.11** |

(The set's `writing` subset is byte for byte its `question` subset and is left out.) Decode speed
on BetterBench's 7 multi-turn conversations is the same: first turns 159.6 tok/s against 159.8,
3.85 tokens a step against 3.88.

**BetterBench's `tool_call` and `multi_turn` categories at 1-16 users**, sampled at 0.7, `--tp 2`,
thinking off, the two containers served at once and the card pairs swapped for a second run (both
runs given):

| | main's container | this one |
|---|---|---|
| multi-turn: the right tool | 15/18, 15/18 | **18/18, 18/18** |
| multi-turn: a turn after the tool's result | 16/21, 17/21 | **18/18, 18/18** |
| tool calls: a valid call | 16/17, 15/17 | **17/17, 17/17** |
| tool calls: the right tool | 16/17, 15/17 | **17/17, 17/17** |
| weighted decode, tok/s | 176.7, 175.0 | 183.8, 172.1 |
| aggregate at 1 / 2 / 4 / 8 / 16 users, tok/s | 116 / 174 / 252 / 253 / 308, 113 / 175 / 240 / 261 / 314 | 111 / 169 / 240 / 260 / 309, 115 / 168 / 235 / 244 / 299 |

Throughput is level within the runs' noise (about 3%; the sessions differ too, since the calls that
now succeed lead to other turns).
