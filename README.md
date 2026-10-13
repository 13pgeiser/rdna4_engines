<p align="center">
  <img width="1774" height="887" alt="hipfire — local LLM inference for AMD GPUs" src="https://github.com/user-attachments/assets/2013c1ae-6011-477c-a8de-5e90751bed74" />
</p>

<h1 align="center">hipfire</h1>

<p align="center">
  <strong>Fast, AMD-native LLM inference — RDNA-first, CDNA-supported.</strong><br />
  Rust + HIP + Redline. No Python in the hot path. Simple pull, serve, and chat UX.
</p>

<p align="center">
  <a href="https://github.com/warpfront/hipfire/releases"><img alt="Stable release v0.4.1" src="https://img.shields.io/badge/stable-v0.4.1-24292f?style=flat-square" /></a>
  <a href="CHANGELOG.md"><img alt="Changelog" src="https://img.shields.io/badge/changelog-v0.4.1-f04b24?style=flat-square" /></a>
  <a href="docs/MODELS.md"><img alt="90 curated registry entries" src="https://img.shields.io/badge/registry-90%20curated%20entries-ff8a1f?style=flat-square" /></a>
  <a href="https://discord.gg/F3BaywB8Rs"><img alt="Join Discord" src="https://img.shields.io/badge/chat-Discord-5865F2?style=flat-square" /></a>
</p>

<p align="center">
  <a href="#mq4r--redline">MQ4R + Redline</a> ·
  <a href="#curated-model-registry">Models</a> ·
  <a href="#gpu-support">GPU support</a> ·
  <a href="#install">Install</a> ·
  <a href="#documentation">Docs</a>
</p>

```bash
hipfire pull qwen3.8:27b-mq4-xts
hipfire serve qwen3.8:27b-mq4-xts -d
hipfire chat qwen3.8:27b-mq4-xts
```

That pulls about 15 GB and wants a 24 GB+ card or Strix Halo. On smaller cards,
start with `qwen3.5:4b` (or `qwen3.5:9b`) in the same commands.

Qwen3.8 Flash-Next (MoE; about 125 GB of base weights, plus sidecars):

```bash
hipfire pull qwen3.8:flash-next
hipfire serve qwen3.8:flash-next -d
```

Flash-Next needs a large memory budget, not just a 32 GB card: use a
128 GB Strix Halo system or an R9700 with sufficient host RAM for mapped
experts. Up to 262K native context is supported on these routes subject to
available memory; see [GPU-tier recommendations](docs/PICK-A-MODEL.md).

One-shot inference uses the same model registry and serving stack:

```bash
hipfire run qwen3.8:27b-mq4-xts "What is the capital of France?"
```

Image generation (first release; RDNA3/3.5 measured):

```bash
hipfire pull flux.schnell:1
hipfire img --out x.png flux.schnell:1 "a red cube on a wooden table"
```

The daemon exposes an OpenAI-compatible API on `127.0.0.1:11435` (loopback
only by default; set `serve.host` to listen on other interfaces):

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8:27b-mq4-xts",
       "messages": [{"role": "user", "content": "What is 2+2?"}],
       "max_tokens": 256,
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen models think by default, and `max_tokens` covers the `<think>` span as
well as the answer: with a small cap and thinking on, `content` can come back
empty (`finish_reason: "length"`). Turn thinking off as above, or raise
`max_tokens`. See [docs/SERVE.md](docs/SERVE.md).

Not sure which model fits your card? See
[Pick a model for your GPU](docs/PICK-A-MODEL.md).

Current stable release: **[v0.4.1](https://github.com/warpfront/hipfire/releases/tag/v0.4.1)**.

## Coming in 0.4.2 (beta)

- **Flash-Next EP2:** owner-routed prefill and expert staging default on
  when admitted on two R9700s (gfx1201). EP4 owner-routed prefill remains
  refused; this is not a refusal of every EP4 route. See
  [multi-GPU serving](docs/multi-gpu.md).
- **Flash-Next NPU route:** `fusions.npu_trunk = "auto"` defaults on for
  eligible Strix Halo (gfx1151) systems with an open XDNA2 NPU and matching
  calibration. Refused or unsupported systems use GPU-only execution.
  Set `HIPFIRE_QWEN4_NPU_TRUNK=off` to disable it. See
  [configuration](docs/CONFIG.md).
- **Engine YaRN:** checkpoint YaRN RoPE scaling and the
  `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR` override make extended
  context available where the model and memory budget admit it. See
  [configuration](docs/CONFIG.md).
- **Halo sidecars:** `hipfire pull` fetches the Flash-Next zab trunk
  overlay, applied automatically on gfx1151; q4 PLE also defaults on
  there when its sidecar is present.
- **Qwen3.8-27B paired DFlash:** defaults on when the registry-paired draft
  is installed; without it, MTP stays `auto`. See the
  [model-scoped default and kill switch](docs/CONFIG.md#qwen38-27b-paired-dflash-default).

See [CHANGELOG.md](CHANGELOG.md) for the complete beta feature list and
release measurements. [BENCHMARKS.md](docs/BENCHMARKS.md) explains fixture
identity and admission requirements; its retained tables and older release
measurements are historical, not current product baselines.

Curated weights are published through
[huggingface.co/hipfire-models](https://huggingface.co/hipfire-models)
and the per-model repositories recorded in the dynamic registry.

Discord: <https://discord.gg/F3BaywB8Rs>

## MQ4R + Redline

MQ4R is the performance-oriented Qwen 3.6 35B-A3B SKU. It combines
uniform MQ4 attention and gate-side weights with graded routed experts
and the fused gate path.

```bash
hipfire pull qwen3.6:35b-a3b-mq4r
hipfire run qwen3.6:35b-a3b-mq4r \
  "Implement a bounded lock-free queue in Rust."
```

The model is 18.7 GB and requires approximately 22 GB of available
VRAM. MQ4R prioritizes throughput; use the default MQ4P model or MFP4,
MQ5, or MQ6 when quality matters more than maximum decode speed.

Redline is hipfire's in-tree dispatch and retained-replay substrate. It
records the actual kernel graph, derives resource dependencies, retains
invariant command state, and lowers validated paths through public ROCr
queue interfaces.

Redline supports the RDNA architecture family from RDNA1 through RDNA4.
Optimized routes remain architecture- and workload-specific. Unsupported
graphs, failed shadow validation, ABI mismatches, queue faults, and model
changes fail closed to ordinary HIP dispatch.

<p align="center">
  <img width="1200" alt="Hipfire runtime and Redline validated replay path, with fail-closed ordinary HIP fallback" src="docs/assets/readme/redline-runtime.svg" />
</p>

### MQ4R measurement history

The sealed MQ4R measurements and their methods are preserved in the
[gfx1201 campaign report](docs/perf-checkpoints/2026-07-13-redline-mq4r-110-to-204.md).
These are historical fixtures, not current product performance claims.
See [BENCHMARKS.md](docs/BENCHMARKS.md) for claim requirements and the
[Redline integration boundary](crates/redline-dispatch/HIPFIRE-GRAFT.md).
To reproduce the sealed TG128 route, model, and sampling fixture on a supported
GPU, run `python3 -m tools.redline golden`; the
[Golden Redline guide](docs/GOLDEN-REDLINE.md) also covers making the validated
fixture the serve default and connecting Hermes, Pi, or another
OpenAI-compatible client.

## Curated model registry

The bundled registry currently contains 90 curated entries (models, quant variants and
draft sidecars) plus 82 aliases. Run
`hipfire list -r` to see the authoritative live list.

| Registry family | Pull tags and variants |
|---|---|
| Qwen 3.5 dense | Primary: `qwen3.5:0.8b`, `qwen3.5:2b`, `qwen3.5:4b`, `qwen3.5:9b`, `qwen3.5:27b`; MQ3: `qwen3.5:2b-mq3`, `qwen3.5:4b-mq3`, `qwen3.5:9b-mq3`, `qwen3.5:27b-mq3`; MQ6: `qwen3.5:0.8b-mq6`, `qwen3.5:2b-mq6`, `qwen3.5:4b-mq6`, `qwen3.5:9b-mq6`, `qwen3.5:27b-mq6`; legacy HF6: `qwen3.5:2b-hf6`; drafts: `qwen3.5:9b-draft`, `qwen3.5:27b-draft`, `qwen3.5:27b-draft-mq3` |
| Qwen 3.5 MoE | `qwen3.5:35b-a3b` |
| Qwen 3.6 dense | `qwen3.6:27b`, `qwen3.6:27b-mq3`, `qwen3.6:27b-draft`, `qwen3.6:27b-draft-mq3` |
| Qwen 3.6 35B-A3B | `qwen3.6:35b-a3b` (MQ4P default), `qwen3.6:35b-a3b-mq2`, `qwen3.6:35b-a3b-mq3p`, `qwen3.6:35b-a3b-mq4p`, `qwen3.6:35b-a3b-mfp4`, `qwen3.6:35b-a3b-mq4r`, `qwen3.6:35b-a3b-mq5`, `qwen3.6:35b-a3b-mq6` |
| Qwen 3.8 dense | MQ V2 ladder: `qwen3.8:27b-mq3-xt`, `qwen3.8:27b-mq3`, `qwen3.8:27b-mq3-pro`; `qwen3.8:27b-mq4-xt`, `qwen3.8:27b-mq4-xts` (symmetric XT, flagship), `qwen3.8:27b` (MQ4V2 default), `qwen3.8:27b-mq4-pro`; corresponding MQ5 and MQ6 `-xt` / base / `-pro` tags; drafts `qwen3.8:27b-draft-mq3` through `-mq6` (MQ4 recommended) |
| Qwen 3.8 Flash-Next | `qwen3.8:flash-next` (canonical, MQ4 XTS); 262K context on one R9700 (gfx1201, tp=1, routed experts host-mapped) and on Strix Halo (gfx1151) |
| Muse Glimmer | `muse-glimmer` (MQ4 quality trunk), `muse-glimmer:fast` (MQ4R speed SKU), `muse-glimmer:draft` |
| Ornith 1.5 | `ornith-1.5:35b-a3b` (MQ4 default), `ornith-1.5:35b-a3b-mq4r` / `ornith-1.5:fast` (MQ4R) |
| DeepSeek V4 Flash | `deepseek-v4-flash` |
| MiniMax-M2.7 | `minimax-m2.7` |
| North-Mini-Code-1.0 | `north-mini-code` |
| Qwen3 standard attention | `qwen3:0.6b`, `qwen3:8b` |
| Carnice tool-use | `carnice:9b`, `carnice:27b`, `carnice:9b-mq6`, `carnice:27b-mq6` |
| Qwopus | `qwopus:4b`, `qwopus:9b`, `qwopus:27b`, `qwopus:4b-mq6`, `qwopus:9b-mq6`, `qwopus:27b-mq6`, `qwopus3.6:27b-coder` |
| LFM2.5 | `lfm2.5:350m`, `lfm2.5:1.2b`, `lfm2.5:1.2b-thinking`, `lfm2.5:8b-a1b` |
| NEX N2 Mini | `nex-n2:mini` |
| VibeThinker-3B | `vibethinker:3b`, `vibethinker:3b-mq6` |

Common aliases include `qwen3.5`, `qwen3.6`, `qwen3.8`, `qwen3`, `carnice`,
`qwopus`, `deepseek4`, `deepseek-v4`, `muse-glimmer`, `ornith`, `ornith-1.5`, `ornith-1.5:fast`, and `vibethinker`.

Carnice uses the Hermes tool-call format. Plain Qwen 3.5 and 3.6 use
their native Qwen XML tool-call format.

See [docs/MODELS.md](docs/MODELS.md) for sizes, minimum VRAM,
recommended sampling settings, sidecars, artifact provenance, and
bring-your-own-model flows through `hipfire quantize`.

## GPU support

| Family | Representative architectures | Notes |
|---|---|---|
| Vega/CDNA | `gfx906`, `gfx908`, `gfx940`-`gfx942` | Wave64 HIP; MQ3 and similar run via per-token GEMV fallback (correct, slower prefill) |
| RDNA1 | `gfx1010`-`gfx1013` | Portable HIP and Redline dispatch support |
| RDNA2 | `gfx1030`-`gfx1032` | Portable HIP and Redline dispatch support |
| RDNA3 | `gfx1100`-`gfx1103` | Architecture-tuned kernels and validated MQ4R route |
| RDNA3.5 | `gfx1150`-`gfx1152` | Architecture-tuned kernels and validated MQ4R route |
| RDNA4 | `gfx1200`, `gfx1201` | WMMA paths and validated retained-PM4 MQ4R route |

Optimized kernel families are RDNA3/3.5 (gfx11) and RDNA4 (gfx12) via WMMA;
other architectures run through portable or architecture-specific fallback
paths. Support for a GPU family does not imply every optimized route is admitted.

Architecture-specific kernels are selected through typed dispatch tables.
Unsupported specializations return to the correct portable or architecture
fallback instead of being applied to neighboring GPU families.

## Why

AMD GPUs are capable inference devices, but tuning and runtime support vary
widely across consumer, professional, APU, and datacenter products.

hipfire supplies its own Rust runtime, model implementations, quantization
formats, dispatch layer, and HIP kernels. ROCm is loaded dynamically; there
is no Python, PyTorch, CUDA translation layer, or third-party inference
engine in the hot path.

Redline removes launch overhead after the ordinary graph has been proven
safe. It does not replace the model implementation or bypass hipfire's
correctness gates.

## Performance and lifecycle

See [BENCHMARKS.md](docs/BENCHMARKS.md) for historical measurements and the
requirements for fixture identity, measurement methods, and admission. Historical
release results remain in [CHANGELOG.md](CHANGELOG.md) and the immutable
[performance checkpoint ledger](docs/perf-checkpoints/).

PFlash prefill compression, CASK / TriAttention KV eviction, Givens/asymmetric
KV modes, and legacy DFlash knobs are deprecated since 0.4.0, unsupported,
and scheduled for removal in 0.5.0. They are not recommended settings or
current performance baselines. See [configuration](docs/CONFIG.md) for the
supported settings and [validation](docs/VALIDATION.md) for current gates.

## Install

Linux with ROCm 6 or newer:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh | bash

# Or install the integration branch for testing:
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/install.sh \
  | bash -s -- --branch beta
```

RDNA4 requires ROCm 6.4 or newer. gfx1151 requires ROCm 7.2 or newer.
Release-tag installs (`install.sh` / `install.ps1`, or `hipfire update --tag`)
download a verified prebuilt kernel pack for the detected arch when it admits;
otherwise they fall back to local hipcc. See
[Prebuilt kernel packs](docs/GETTING_STARTED.md#prebuilt-kernel-packs).
Run `hipfire --version` for a concise build ID or `hipfire version` to compare
the installed binary, managed source checkout, and daemon. Managed Linux
installs can switch revisions with `hipfire update @beta`,
`hipfire update --tag v0.4.0`, or `hipfire update --commit <sha>`.

To uninstall a managed Linux install while keeping downloaded models and
settings:

```bash
curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/master/scripts/uninstall.sh | bash
```

Add `--dry-run` to preview it. `--purge` also deletes all data under
`~/.hipfire` and requires explicit confirmation.

For Windows, source builds, and verifying the install:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## NixOS

First-class support via Nix flake. See [docs/NIXOS.md](docs/NIXOS.md).

```bash
nix develop github:warpfront/hipfire  # dev shell with Rust + ROCm
nix build github:warpfront/hipfire    # build package
```

NixOS module:

```nix
{
  inputs.hipfire.url = "github:warpfront/hipfire";
  # then in configuration.nix:
  services.hipfire.enable = true;
  services.hipfire.gpuTargets = [ "gfx1100" ];
}
```

## Container (podman/docker)

A multi-stage `Containerfile` builds a slim deliverable inference image and a
full-toolchain GPU gate-runner for reproducible PR/dev-build validation. See
[docs/CONTAINER.md](docs/CONTAINER.md).

```bash
podman build -f Containerfile --target runtime -t hipfire .
podman run --rm -it --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --security-opt seccomp=unconfined \
  -v hipfire-models:/root/.hipfire/models \
  -v hipfire-kcache:/var/cache/hipfire \
  hipfire run qwen3.5:4b "2+2="
```

## Documentation

| Page | Topic |
|---|---|
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Install, first run, what to read next |
| [NIXOS.md](docs/NIXOS.md) | NixOS flake, module, dev shell |
| [CLI.md](docs/CLI.md) | Every subcommand, flags, file locations |
| [MODELS.md](docs/MODELS.md) | Curated tags, BYO models, file extensions |
| [QUANTIZE.md](docs/QUANTIZE.md) | `hipfire quantize` for HF / safetensors (GGUF input deprecated) |
| [CONFIG.md](docs/CONFIG.md) | Every config key and env override |
| [SERVE.md](docs/SERVE.md) | OpenAI-compatible HTTP API |
| [IMAGEGEN.md](docs/IMAGEGEN.md) | FLUX.1 / FLUX.2 Klein image generation — local test guide |
| [BENCHMARKS.md](docs/BENCHMARKS.md) | Historical measurements and performance claim requirements |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Engine layout, dispatch, two model paths |
| [QUANTIZATION.md](docs/QUANTIZATION.md) | MQ4 / HF4 design, KV cache modes (q8 / fp8 / fwht; legacy asym), FWHT math |
| [CONTAINER.md](docs/CONTAINER.md) | Runtime and GPU gate-runner containers |
| [multi-gpu.md](docs/multi-gpu.md) | Multi-GPU serving — EP, TP, pipeline parallelism, budgets and refusals |
| [methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md) | Bench protocol — read before claiming a perf win |
| [HIPFIRE-GRAFT.md](crates/redline-dispatch/HIPFIRE-GRAFT.md) | Redline integration and enablement boundary |

## License

hipfire is licensed under **Apache-2.0** as of v0.3.0. See
[LICENSE](LICENSE), [LICENSE-APACHE](LICENSE-APACHE), and
[NOTICE](NOTICE) for details. Releases up to and including v0.2.1 were
dual MIT/Apache-2.0; that grant is not revoked.

Individual files whose substantive authors have not elected Apache-2.0
remain MIT-licensed (see [LICENSE-MIT](LICENSE-MIT)) and are identified
by their per-file `SPDX-License-Identifier`. No contributor's file was
relicensed in absentia. New contributions default to Apache-2.0 via DCO
sign-off. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor
side and
[docs/governance/relicense-2026-05.md](docs/governance/relicense-2026-05.md)
for the decision record (including the 2026-05-19 course correction and
the v0.3.0 move to outbound Apache-2.0).

Original architectural innovations originating in hipfire are
catalogued in [PRIOR-ART.md](PRIOR-ART.md); derivative works
(including reimplementations informed by hipfire's design) should
attribute the corresponding inventions per [AGENTS.md](AGENTS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), including
[Release cadence and PR cutoff](CONTRIBUTING.md#release-cadence-and-pr-cutoff)
(weekly releases alternating Tuesday/Sunday). Install local hooks with
`./scripts/install-hooks.sh`. The no-GPU CI subset is
`./scripts/no-gpu-ci.sh`; it does not replace the hardware gates.
**There is no single canonical correctness gate** — the retired
`scripts/coherence-gate*.sh` batteries are not acceptance evidence and
`coherence-gate-dflash.sh` no longer exists. Select the route your change
owes from [docs/VALIDATION.md](docs/VALIDATION.md): `scripts/serve_harness.py`
for generation and state-lifecycle changes, `scripts/redline_daemon_harness.py`
for kernel, dispatch, graph, or Redline-replay changes. Architecture-specific
work must also pass its channel test, anti-bleed checks, and relevant speed
gate. Don't bypass the gates with `--no-verify` — see
[methodology/perf-benchmarking.md](docs/methodology/perf-benchmarking.md).
