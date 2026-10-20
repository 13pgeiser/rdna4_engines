# RDNA4 Engines

Host-orchestration repo for serving LLMs on an **AMD Radeon AI PRO R9700**
(gfx1201 / RDNA4, 32 GiB) box. It is *not* an engine: it is the set of
top-level entry-point scripts that prepare docker images and models, then serve
one of the engines below on port **9931**.

The five big directories are **git-subtree checkouts of external repos — not
first-party code**. Do not hand-edit them; changes belong in the upstream repos.
Sync all five with `./git.sh`.

## Engines (upstream sources)

| Directory | Engine | Upstream |
| --- | --- | --- |
| `radiance-vllm-mxfp4/` | vLLM + MXFP4 stack | <https://codeberg.org/ggz14/radiance-vllm-mxfp4> (`main`) |
| `radiance/` | C++/HIP inference engine (distinct from the vllm one) | <https://codeberg.org/StillDeadcode/radiance> (`main`) |
| `llama.cpp/` | llama.cpp | <https://github.com/ggml-org/llama.cpp> (`master`) |
| `hipfire/` | RDNA-native LLM inference engine (Rust + HIP + Redline) | <https://github.com/warpfront/hipfire> (`master`) |
| `libr4d/` | HIP kernel library for gfx1201 (used by radiance-vllm) | <https://codeberg.org/StillDeadcode/libr4d> (`main`) |

## Entry points

All four serve on **port 9931**, build their docker image **only if missing**,
and (where active) write a 210 W GPU power cap to sysfs via `sudo` at start.

| Script | Serves | Notes |
| --- | --- | --- |
| `./run_radiance-vllm-mxfp4.sh [rocm7\|rocm10] [--bench]` | `radiance-vllm-mxfp4` | default `rocm10`; prepares `/models/radiance-vllm-mxfp4` (clones the checkpoints below, builds the derived variants), TP=1 foreground serve; `--bench` serves detached, waits for `/health`, runs `vllm bench serve`, then stops the container |
| `./run_radiance.sh [model]` | `radiance` | `.rad` checkpoints (27B MXFP4 + DFlash2 drafter, or flash-next fp8 trunk + 4-bit codebook MoE); model menu; `VMEMOFF=1` runs the KMT workaround shim (see below) |
| `./run_llama.sh [model]` | `llama.cpp` | menu over GGUFs under `/models/llama.cpp`, interactive or positional arg |
| `./run_hipfire.sh [tag]` | `hipfire` | registry tags (default `qwen3.8:27b-mq4-xts`); needs GPU passthrough (`/dev/kfd`, `/dev/dri`, `seccomp=unconfined`) |

`AGENTS.md` is the detailed reference for the local flow: the checkpoint
pipeline, serving knobs, TP auto-detection, the libr4d build, and troubleshooting.

## Checkpoints (models)

| Checkpoint | Source |
| --- | --- |
| `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | <https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4> |
| `tcclaviger/Qwen3.8-27B-DFlash2-FP8` | <https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8> |
| `StillDeadcode/qwen3.8-27b-mxfp4` (`.rad`) | <https://huggingface.co/StillDeadcode/qwen3.8-27b-mxfp4> |
| `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe` (`.rad`) | <https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe> |
| `froggeric/Qwen-Fixed-Chat-Templates` (chat template) | <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates> |

AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` **does not load as-is**: its bf16 MTP head
is mis-declared in the quark config (vLLM fails with
`AssertionError: Attempted to load weight (5120, 10240) into (5120, 5120)`).
The rewrite in `radiance-vllm-mxfp4-scripts/` is mandatory:

1. `fp8_mtp.py <src> <dst>` → `Qwen3.8-27B-MXFP4-mtpfp8` (needs torch)
2. `mtpfp8-pertoken.py <src config.json> <dst config.json>` → `-pertoken` variant (config-only)

## Requirements

- Linux with ROCm + an AMD gfx1201 card, docker, and `sudo` (power cap)
- `/models` (filesystem root) for checkpoint storage
- `gfx1201` build targets (`GPU_TARGETS`/`AMDGPU_TARGETS`) are baked into the
  local image builds
