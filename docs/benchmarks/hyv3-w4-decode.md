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
| Fused Q/K prologue and residual/shared-combine folding | HYV3's graph; TP2 keeps separate routed/shared reductions | Fresh TP2 whole-model A/B; other models affected by shared-kernel changes |

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

The earlier native-tool comparison was unmatched: the untuned stack scored 102/113 exact and
112/113 executable cases versus 105/113 and 113/113 in an older baseline. The matched release-tag
A/B below supersedes it for this PR validation: both arms scored 102/113 and 112/113, with identical
per-case results. This remains a small curated subset, not a broad quality test.

## Validation on the rebased optimization branch

The GPU, layout and reference results below were recorded with the candidate based on merged main
`2d2f638`. Main later advanced to `d650eca` with two Gemma4 changes, and the PR branch was rebased
on it. On that final rebased tip, `arch_test` was rebuilt and passed (`ctest -R arch_test`, 1/1).
I then retried the `radiance` build target from this latest source; it still fails on the missing
KV-stream headers noted below. The full unit, GPU and inference suites were not rerun after rebasing.

- Release and ASan/UBSan/leak architecture suites, including HYV3 TP1/TP2 graph checks: passed
  before the base advance.
- `gfx1201` `libr4d`, `r4d_selftest`, and all three HYV3 plugins: compiled.
- Stored-layout parity: 1,145 checks over 20 cases passed, including independent canonical nibble
  addresses, GPTQ and rotated experts, subgroup codebooks and unaffected dense formats.
- Reference arithmetic: 136,622 checks over 254 cases passed with the incomplete upstream
  inventory/sweep cases excluded. The new exhaustive BF16-add/cast-chain case passed separately.
  The unfiltered reference suite's three failed assertions were reproduced with unchanged upstream
  test source: an operation-count mismatch and missing sweep geometry for `kv_store_streamed` and
  `kv_resolve`.
- Full rebased GPU selftest on the designated last R9700 (`gfx1201`, UUID
  `GPU-7f082536d27449f3`): 9,057 `ok` results, 194 skips, and two failing assertions in one
  `hc_read_e4m3` test case at M=2048; overall exit 1. The same untouched HC-read test family
  failed with the pre-follow-up baseline plugin at M=2048 and M=700 (the particular stream/form
  varied). The PR does not change that kernel or test. This is not a clean full-suite pass.
- Focused GPU checks passed for `gemm_w4a16_asym_nt_m64`, `moe_gather_bf16`,
  `qkv_norm_rope_kv_store_h128_bf16` (including byte parity to the split norm/RoPE/store chain),
  and `rmsnorm_bf16`. The signed-I4 `moe_gemm_w4a8` case had no failures; its exit status was the
  harness's intentional-skip status for unsupported large-expert aligned tiles. The full run also
  exercised its HYV3 top-8 gate/up and down cases without failures.

The latest-source `radiance` target build stops at `ctx.cpp:6`, `kvbind.cpp:4`, `issue.cpp:22`,
and `tierexec.cpp:4`: the runtime includes `kvstream/bind.h`, `kvstream/stage.h` and
`kvstream/plan.h`, but current `main` has no engine-side `core/kvstream/` module. This blocks
building the matching ABI-30 HIP engine needed for whole-model tests. The older available GPU
engine is ABI 27 and refuses the ABI-30 plugin; the current-main host build has no HIP backend.
Accordingly, no fresh HYV3 or Qwen inference A/B ran.

The user's local Qwen `.rad` was inspected without downloading anything:
`/mnt/gostar_data/models/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad` is a 19.4 GB
`Qwen3_5ForConditionalGeneration` container with `n_expert=0` and group-32 MXFP4 weights (SHA-256
`a4908940c568f5f9dd24e88f46aefe9ac2ae07dfec023a2eaa1d14dcaa3df4f8`). It does not exercise this
PR's asymmetric W4/W5 or per-128 I4 paths; on the release-tag build it was used only as an MXFP4
compatibility smoke (answered `42`), not as an affected-model A/B. Attempts against the current-main
candidate exited before model load because that engine could not be built. No model was downloaded
and no llama-swap route was edited.

### Release-tag plus PR build experiment

To isolate the later mainline KV-stream integration, a temporary worktree was based on release tag
`v1.12.0`, then given the HYV3 model-support patch (from `32881da` to `79487d2`) and the PR #237
delta (`d650eca` to this PR head). One context conflict in `libr4d/r4d_layout.cpp` came from a
post-release MXFP4-gated wrapper absent from the tag; the resolution kept the release's unrelated
MXFP4 code and applied the PR's I4 layout changes. No changes were made to the published branch for
this experiment.

- Full Release host build: passed (694 targets).
- Full `gfx1201` HIP build: passed (809 targets).
- Focused `arch_test`, `chat_test`, `chatparse_test`, and `checkpoint_test`: 4/4 passed.

Follow-up validation used only the third R9700 (GPU index 2, UUID
`GPU-7f082536d27449f3`):

- `gemm_w4a16_asym_nt_m64`: 341 checks passed, zero failures or skips.
- `moe_gemm_w4a8`: 647 checks passed, zero failures, 12 expected skips for unsupported aligned
  prefill tiles at 640 experts (the run-map limit is 512); HYV3 top-8 gate/up and down geometries
  were exercised. The selftest exits 2 when skips are present.
- The release+PR server loaded the local full-GPTQ W4 HYV3 container at 4096 context on loopback
  port 18100. `/health` was OK and a chat request asking for 6×7 returned `42`. The process was
  stopped afterward and the port was confirmed free; no route was changed.

### Full-suite, performance and quality A/B (completed on the release-tag stack)

The full CTest suite on the release+HYV3-support+PR candidate passed **121/121** tests, including
83 unit tests and all 10 GPU-labelled tests. The full `r4d_selftest` passed on a standalone rerun
and again in the final complete suite. An earlier full-suite attempt had an intermittent
`hc_read_e4m3` assertion; the same untouched family also failed on the baseline run, while four
isolated baseline/candidate reruns passed 38/38 each. The final full-suite run was clean.

A matched TP1 full-GPTQ W4 A/B used the exact frozen 10,030-token prompt and 2,000-token decode
request (`e04af05f...`), on the third R9700, BF16 KV, one sequence, 65,536 context, 4,096 batched
tokens, and no launch tune or speculative decoding. The reference was release `v1.12.0` plus HYV3
model support; the candidate added only PR #237's delta. Each arm had three repetitions, with the
same 16-token prefix-cache behavior after the first run.

| Arm | Median decode tokens/s | Median prefill tokens/s | Answer SHA-256 |
| --- | ---: | ---: | --- |
| v1.12.0 + HYV3 support (baseline) | 76.567 | 7,876.292 | `caca914b...` |
| v1.12.0 + HYV3 support + PR #237 | 109.618 | 7,926.569 | `caca914b...` |

That is **+43.17% median decode throughput** and **+0.64% prefill throughput** on this narrow
synthetic workload; all six generated-answer hashes were identical. The speed result is specific
to this matched release-tag comparison, not a claim about other models or current-main performance.

Quality comparisons on the same two builds:

- Greedy output hashes matched **12/12** fixed probes.
- Teacher-forced evaluation on the frozen eight-document, 578-position corpus measured mean KL
  `7.61e-8`, maximum KL `1.57e-6`, **100% top-1 agreement**, and identical PPL `2.875200`.
- The 113-case native-tool BFCL subset was identical row-for-row: **102/113** exact calls,
  **112/113** execution successes, and zero API errors on both arms. This is a curated subset, not
  broad capability evidence.

#### Same-build cross-card diagnostic

At the user's request, I reran the same candidate executable, `.rad` artifact, launch flags, and
three 2,000-token repetitions on two individual R9700s. The frozen 10,030-token benchmark used the
same per-repetition prefix scheme on both cards; no launch tune or drafter was enabled. The model
was loaded separately on each card, with no other Radiance process using either device.

| Device | Median decode tok/s | Median prefill tok/s | Answer SHA-256 |
| --- | ---: | ---: | --- |
| PCI `0000:06:00.0`, UUID `GPU-048200e09844fd51` | 118.380 | 8,077.674 | `caca914b...` |
| PCI `0000:0f:00.0`, UUID `GPU-7f082536d27449f3` | 109.834 | 8,011.282 | `caca914b...` |

The same candidate was **7.2% slower on the designated PCI `0f:00.0` card**. A fresh confirmation
on that card measured **110.311 tok/s**, within 0.44% of the initial 109.834 tok/s result. The older
115.678 tok/s untuned run was on PCI `06:00.0`, so the earlier 115-versus-109 comparison was
confounded by card/operating-point differences, not evidence that splitting the branch caused a
regression. This isolates a material card-dependent decode difference; it does **not** identify
whether power limit, clocks, thermals, or silicon variation caused it. Per-run power/clock
telemetry was not captured, and PCIe bandwidth is not established as the cause. Raw-request repeats
that reused 10,000 cached prompt tokens are retained separately and excluded from this comparison.

Additional compatibility smokes using the same candidate build passed for the local Qwen4Exp W5
container (returned `42`; prompt-lookup drafted 6 tokens and accepted 1), Qwen Flash-Next IQ4R
(TP1 and TP2 launchers, answered `42`), Qwen3.8 27B MXFP4 (answered `42`), Breeze-TTS-2 (valid
24 kHz mono PCM), and HYV3 (answered `42`). The verified deployment build was moved intact to
`/mnt/gostar_data/radiance-builds/radiance-v1.12-hyv3-pr237/build-gfx1201`; all six local model
launchers were syntax-checked and updated to use it. The W5 launcher dropped a stale
`--draft-coupling on` option rejected by this container (it declares no linear drafter); its
six-token prompt-lookup setting remains. These are smoke tests, not matched performance/quality
results for those models. No model was downloaded and no llama-swap route was changed.

Logs, summaries and raw rows are under
`/mnt/gostar_data/radiance-calib/hyv3-pr-validation-20261025/pr237/` and
`/mnt/gostar_data/radiance-calib/hyv3-w4-round2-20261024/` (labels `release-v112-baseline` and
`release-v112-pr`).

## Integration caveat and scope of claims

The successful full builds, full test suite and A/B above use the release-tag stack because the
current-main source still lacks the engine-side `core/kvstream/` headers introduced by upstream
`486c933`; it therefore cannot produce the matching ABI-30 HIP engine. That unrelated upstream
build break remains outside this PR, so the A/B must not be described as a current-main-vs-PR run.
The PR is validated and ready for review against the buildable release-tag stack, but after the
upstream KV-stream integration is restored, current-main builds/tests should be rerun.

Performance and quality claims are limited to the HYV3 TP1 full-GPTQ W4 A/B above. The Qwen W5,
IQ4R and MXFP4 runs were compatibility smokes only; no broad speedup or quality claims are made for
other model families or formats. The native BFCL corpus is also a small safe subset, not a broad
quality benchmark.

The cold-weight diagnostics are `r4d_selftest --perfw4`, `--perfw4bf16`, `--perfbf16` and
`--perfhyv3moe`; see [libr4d/README.md](../../libr4d/README.md). Their byte/canonical checks and
shape-specific timings do not replace whole-model performance or quality comparisons.
