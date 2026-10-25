# HYV3 W4 decode optimizations

These changes optimize the existing HYV3 path and shared asymmetric W4/W5 and signed-I4 expert
kernels. They do not change the canonical `.rad` weight encodings, precision, default split-K
selection, or speculative decoding settings.

## Scope and affected paths

| Change | Affected paths | What remains to measure |
| --- | --- | --- |
| Compact one-row W4/W5 epilogue | Callers of the asymmetric W4/W5 GEMM at one live row and a compatible tile; includes Gemma callers | Fresh whole-model W4/W5 comparisons outside HYV3 |
| BF16-to-F16 narrowing inside W4 GEMM | Non-rotated group-128 asymmetric W4 `LinearFP8` decode through 64 rows | Other models using this path; Gemma's separate a4 graph still keeps its cast |
| Fragment-tiled expert I4 and one-row register loads | Per-128-scale signed-I4 experts, including plain and Hadamard-rotated formats | Other models/recipes that actually use this encoding, plus prefill and TP regressions |
| Fused Q/K prologue and residual/shared-combine folding | HYV3's graph; TP2 keeps separate routed/shared reductions | Fresh HYV3 throughput, text/tool quality and TP1/TP2 checks on the rebased engine |

Subgroup-scaled codebooks retain their row-major packing. In particular, the served Flash-Next
W5DN64 recipe is not evidence for the per-128 I4 kernel: its routed experts use a different
subgroup-scaled path. No speedup is claimed for it or for any unmeasured model.

## Historical HYV3 measurements

These are earlier measurements of the original optimization branch, **not a benchmark of this
rebased PR**. They establish motivation only.

- One Radeon AI PRO R9700 (`gfx1201`), TP1, full-GPTQ W4 HYV3.
- BF16 KV, 65,536 maximum context, one sequence, 4,096 batched tokens, no drafter or launch cache.
- Matched 10,030-token prompt and 2,000 generated tokens, temperature zero; reported decode
  medians are from three repetitions.
- The frozen request SHA-256 is
  `e04af05fd7952b5c1e8a38927bb53ab8621ca3e90a2713285eb98122c849ed47`.
  The first repetition was uncached; later repetitions reused a 16-token prefix.
- The frozen prompt repeatedly requests the word `alpha`, and the generated answer repeats it.
  This is a narrow synthetic workload, not representative prose, code or tool-use throughput.

| Historical stage | Median decode tokens/s |
| --- | ---: |
| Initial model path | 90.290 |
| Fresh pre-packing control after earlier graph work | 95.392 |
| Coalesced per-128 I4 expert packing | 103.364 |
| Rebuilt cast-free graph | 111.922 |
| Final untuned optimization stack | 115.678 |

The frozen answer SHA-256 was
`caca914b1bced70b2f9d8bf32d476179f18cef456201b02f55f16b9863bedbf0` in every repetition.
That establishes parity for this answer, not numerical or semantic equivalence on arbitrary input. An intermediate 104.744 tokens/s
cast-fusion result used a stale architecture plugin and must not be used to quantify removal of
graph casts. A separate launch-cache experiment near 120.5 tokens/s changed reduction order and
is not part of this patch.

The historical native-tool check also needs a new, isolated A/B. The untuned stack scored 102/113
exact and 112/113 executable cases versus 105/113 and 113/113 in an earlier baseline that predated
intervening engine changes. This was a small curated executable subset, not a broad quality test;
the unmatched baseline prevents attributing the difference, but it does not clear a quality gate.

## Validation on upstream 2d2f638

- Host architecture suite, including existing other-model graph pins and HYV3 TP1/TP2 and
  multi-pass checks: passed.
- Architecture suite under ASan/UBSan/leak checking: passed.
- `libr4d`, `r4d_selftest`, and all three HYV3 plugins: compiled for `gfx1201`; no GPU execution.
- Stored-layout parity suite: 1,145 checks over 20 cases passed, including independent canonical
  nibble addresses, GPTQ and rotated experts, subgroup codebooks and unaffected dense formats.
  `libquant` was present and the cases were exercised rather than skipped.
- Reference arithmetic: 136,622 checks over 254 cases passed with the two incomplete upstream
  inventory/sweep cases excluded. The new exhaustive BF16-add/cast-chain case passed separately.
  The full reference suite's three failed assertions were reproduced using unchanged upstream
  test source: it lists 208 operations while the plugin ships 210 and lacks sweep geometry for
  `kv_store_streamed` and `kv_resolve`.

Current upstream runtime sources also include missing `kvstream/bind.h` and `kvstream/stage.h`.
That blocks a fresh full engine/unit build. The layout/reference checks above compile the existing
host format/plugin/device sources in a standalone harness, without modifying production sources
or substituting a runtime. They are not a claim that the complete upstream suite passes.

No serving process was stopped, replaced or used for these checks. No GPU correctness run or new
inference/performance measurement was made during this rebasing pass.

## Before this PR is ready

1. Clear the upstream full-build and reference-inventory blockers and rerun the full unit suite.
2. Rerun GPU numerical/guard suites, including asymmetric W4/W5, rotated/mixed-class MoE and
   normalization/KV-store cases, on the rebased kernels.
3. Run matched current-base HYV3 control/candidate comparisons on ordinary prose, code and tools,
   with cold/uncached prompts and unchanged model, precision and deployment settings.
4. Benchmark and quality-check the other affected W4/W5 and per-128 I4 model paths, including
   prefill and TP configurations. Unchanged subgroup paths need regression coverage, not an
   assumed throughput improvement.

The cold-weight diagnostics are `r4d_selftest --perfw4`, `--perfw4bf16`, `--perfbf16` and
`--perfhyv3moe`; see [libr4d/README.md](../../libr4d/README.md). Their byte/canonical checks and
shape-specific timings do not replace whole-model performance or quality comparisons.
