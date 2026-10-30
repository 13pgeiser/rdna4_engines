# AGENTS.md

Host-orchestration repo for serving LLMs on a local AMD Radeon AI PRO R9700 (gfx1201 / RDNA4) box. The top-level `.sh` entry points and `radiance-vllm-mxfp4-scripts/` are the local (first-party) flow; the five big directories are **git-subtree checkouts of external repos — not first-party code**.

## Subtrees (do not hand-edit)

- `llama.cpp/` — ggml-org/llama.cpp (master)
- `radiance-vllm-mxfp4/` — ggz14/radiance-vllm-mxfp4 (main) — the vLLM + MXFP4 stack
- `radiance/` — StillDeadcode/radiance (main) — a C++/HIP engine, distinct from `radiance-vllm-mxfp4/`
- `libr4d/` — StillDeadcode/libr4d (main)
- `hipfire/` — warpfront/hipfire (master)

Sync all five with `./git.sh` (runs `git subtree add` or `pull` per dir). Changes belong in the upstream repos. `radiance-vllm-mxfp4-scripts/` is local, not a subtree.

## The local flow (top-level scripts)

All four servers listen on port **9931** (not the upstream defaults). The vllm/llama/hipfire entry scripts start by writing a 210 W GPU power cap to sysfs via `sudo` (that line is commented out in `run_radiance.sh`; `run_hipfire_gfx1152.sh` writes no cap — that sysfs line targets the discrete R9700 and would throttle a whole APU SoC).

- `./run_radiance-vllm-mxfp4.sh [rocm7|rocm10] [--bench]` — the radiance-vllm entry point (default **rocm10**). Prepares **`/models/radiance-vllm-mxfp4`** (git-clones `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and `tcclaviger/Qwen3.8-27B-DFlash2-FP8`, then builds the two derived variants below; `/models` is filesystem root — absolute, not `~/models`). Builds the docker image **only if missing** (`radiance-vllm-mxfp4:$(cat radiance-vllm-mxfp4/VERSION)` for rocm7; `rocm10-radiance-vllm-mxfp4:…` for rocm10, via a sed-rewritten `Dockerfile.rocm10` with a ROCm 10.0 base), writes the power cap, and serves via `radiance-vllm-mxfp4/serve-mxfp4.sh` (TP=1, foreground). Docker, not podman. With `--bench` it instead serves detached (`DETACH=1`), waits for `/health` (900 s timeout), runs the vLLM bench client (`vllm bench serve`, 100 random prompts), then stops the container.
- `./run_radiance.sh` — drives the **C++ `radiance` engine** (the `radiance/` subtree, not the vllm one). Builds `local/radiance` via `radiance/Dockerfile` (`GPU_TARGETS=gfx1201`) **only if missing**; during the build it sed-bumps host-test CPU budgets in `radiance/tests/CMakeLists.txt` (tokenizer_test 20→25 s, gbnf_test 15 s, arch_test 20 s) and reverts them after (a failed build leaves the edits in place). Downloads `qwen3.8-27b-mxfp4.rad` from `StillDeadcode/qwen3.8-27b-mxfp4` into `/models/rad` and serves on 9931.
- `./run_llama.sh [model]` — builds `local/llama.cpp:latest` via `llama.cpp/.devops/rocm.Dockerfile` **only if missing** (`AMDGPU_TARGETS=gfx1201`), writes the power cap. The model is a menu (positional arg, or interactive; non-interactive default is index 1) over GGUFs under `/models/llama.cpp`; serves foreground (`docker -it`, container `llama_cpp_server`) on 9931.
- `./run_hipfire.sh [tag]` — the hipfire entry point (default tag **`qwen3.8:27b-mq4-xts`**; a menu now offers two tags). Builds `local/hipfire` via `hipfire/Containerfile --target runtime` **only if missing** (multi-stage; no GPU needed at build time). Pulls the registry model into host `/models/hipfire` (bind-mounted at `/home/hipfire/.hipfire/models`), writes a default `reasoning_effort` (env `REASONING_EFFORT`, default **medium**) into the persistent `hipfire-home` volume, writes the power cap, and serves foreground (container `hipfire_server`) on 9931.
- `./run_hipfire_gfx1152.sh [tag]` — hipfire on a **gfx1152 (Krackan Point) APU box**. Same flow as `run_hipfire.sh`, but: pre-flight fails early unless the host KFD exposes a `gfx_target_version 110502` node; builds `local/hipfire:full` via `hipfire/Containerfile --target full --build-arg HIPFIRE_GPU_ARCHES="gfx1152"` **only if missing** (`full` = hipcc present, since gfx1152 has no prebuilt kernel pack and the slim `runtime` image cannot JIT — it dies with "no usable cached kernel image exists … hipcc is unavailable"; the build-arg bakes in the exact-source gfx1152 registry pack so the first request skips JIT); menu defaults to small models (`qwen3.5:9b` — 27B only fits ~32 GB UMA boxes); writes **no** power cap. Upstream issue #50 (gfx1152 segfault) is open — report crashes there.

## Checkpoint pipeline (radiance-vllm)

AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` **does not load as-is**: its bf16 MTP head is mis-declared in the quark config (fails with `AssertionError: Attempted to load weight (5120, 10240) into (5120, 5120)`). The rewrite is mandatory:

1. `radiance-vllm-mxfp4-scripts/fp8_mtp.py <src> <dst>` → `Qwen3.8-27B-MXFP4-mtpfp8` (needs **torch**; `run_radiance-vllm-mxfp4.sh` spins a throwaway venv in `/models` for it)
2. `radiance-vllm-mxfp4-scripts/mtpfp8-pertoken.py <src config.json> <dst config.json>` → `-pertoken` variant (config-only, no torch; rewrites the 8 `mtp.*` `input_tensors` to per-channel; prints "expect 8")

Note: `radiance-vllm-mxfp4-scripts/fp8_mtp.py` and the copy in `radiance-vllm-mxfp4/fp8_mtp.py` **differ**; the entry script uses the `radiance-vllm-mxfp4-scripts/` one.

`radiance-vllm-mxfp4-scripts/chat_template.jinja` is auto-downloaded from `froggeric/Qwen-Fixed-Chat-Templates` if missing; `run_radiance-vllm-mxfp4.sh` mounts it into the container.

## Radiance stack

The local flow builds its own image (`radiance-vllm-mxfp4:$(cat radiance-vllm-mxfp4/VERSION)`, currently 0.13.0) via `radiance-vllm-mxfp4/Dockerfile` — this differs from the subtree README's quickstart (pulls published `stilldeadcode/vllm-radiance:0.9.3`). For the radiance internals (knobs, troubleshooting, performance, ParoQuant, TP=3 padding), the subtree docs are the authoritative reference: `radiance-vllm-mxfp4/README.md`, `PERFORMANCE.md`, `PAROQUANT.md`, `DOCKERHUB.md`, `TP3_PADDING_PLAN.md`, `MXFP4-NOTES.md`, `FASTBOOT.md`.

- All serving knobs are `${VAR:-default}` env overrides; nothing needs editing. `DRY_RUN=1 ./serve-mxfp4.sh` prints the container command; `./serve-mxfp4.sh --help` is the short reference.
- `run_radiance-vllm-mxfp4.sh` `sed`-rewrites the `--override-generation-config` line inside `serve-mxfp4.sh` on every start (top_k 20), then reverts the file via an `EXIT` trap (`git checkout`) when the script terminates — the local diff is transient, present only during the serve run.
- TP is auto-detected (largest of 8/4/2/1 that fits); 3/6/12 are never auto-picked (GDN head divisibility) and TP=3 is available only via explicit dummy-head padding (`RADIANCE_TP_PAD=3`). Cards below `MIN_GPU_MIB=8192` are excluded so the box's 2 GiB iGPU render node is never picked.
- The stock libr4d kernel in published images **NaNs this model's GDN path** (perplexity in the hundreds of thousands); the launcher builds a pinned libr4d (`AUTO_R4D=1`, default). Don't run with `AUTO_R4D=0` unless you know why.

## libr4d

HIP kernel library for gfx1201. Entry-point names encode the exact geometry (compile-time constants; mismatches are rejected, not run). `r4d.select()` / `r4d.kernels()` answer "is there a kernel for this shape" — `None` means fall back to non-R4D paths. Build: `./build.sh` (needs `hipcc`, same ROCm as the runtime; `radiance-vllm-mxfp4/Makefile`'s `make r4d` clones the pinned libr4d and compiles it inside the image so it matches). Adding a kernel: 6-step process in `libr4d/README.md` (registry row in `r4d_registry.hip` is what makes it visible).

## hipfire

RDNA-native LLM inference engine (Rust + HIP + Redline; ROCm is dlopen'd at runtime). See the `./run_hipfire.sh [tag]` entry point above for the build/pull/serve flow. The `local/hipfire` image bundles the daemon, native CLI, and exact-source registry kernel packages compiled for all five archs. Runs need GPU passthrough (`--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined`); the JIT kernel cache is the named volume `hipfire-kcache` (`/var/cache/hipfire`); the OpenAI-compatible serve API is on port **9931**. The `Containerfile` base is `rocm/dev-ubuntu-24.04:7.2.4`; do not downgrade (gfx1151 needs ROCm 7.2+; gfx1201 needs 6.4+). If a build reports missing `hipcc`/HIP headers, switch the base tag to `7.2.4-complete`.

## Verification

No CI, no pre-commit, no top-level test suite. Subtree checks that need no GPU: `tp3pad_selftest.py` (radiance-vllm-mxfp4), `make verify IMAGE=<rocm-image>` (libr4d). Otherwise the check is "does the server answer a request" — see the subtree READMEs.
