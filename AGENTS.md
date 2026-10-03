# AGENTS.md

Host-orchestration repo for serving LLMs on a local AMD Radeon AI PRO R9700 (gfx1201 / RDNA4) box. Top-level `.sh`/`.py` files are the local flow; the four big directories are **git-subtree checkouts of external repos — not first-party code**.

## Subtrees (do not hand-edit)

- `llama.cpp/` — ggml-org/llama.cpp (master)
- `radiance-vllm-mxfp4/` — ggz14/radiance-vllm-mxfp4 (main)
- `libr4d/` — StillDeadcode/libr4d (main)
- `hipfire/` — warpfront/hipfire (master)

Sync all four with `./git.sh` (runs `git subtree add` or `pull` per dir). Changes belong in the upstream repos.

## The local flow (top-level scripts)

- `./run_radiance.sh [rocm7|rocm10]` — the single radiance entry point (default **rocm10**). It prepares **`/models`** (filesystem root, absolute path not `~/models`), builds the docker image **only if missing** (`radiance-vllm-mxfp4:$(cat VERSION)` for rocm7; `rocm10-radiance-vllm-mxfp4:$(cat VERSION)` for rocm10, via a sed-rewritten `Dockerfile.rocm10` with ROCm 10.0 base), writes the 210 W GPU power cap (sudo), and serves via `serve-mxfp4.sh`. Docker, not podman (the subtree README prefers podman).
- `./build_llama.sh` — builds llama.cpp via `.devops/rocm.Dockerfile` with `AMDGPU_TARGETS=gfx1201`. `./run_llama.sh` serves the GGUF model.
- `./build_hipfire.sh` — builds the `local/hipfire` deliverable image via `hipfire/Containerfile` (multi-stage, podman-native but docker-compatible; `--target runtime`; no GPU needed at build time).
- The llama.cpp/radiance servers listen on port **9931** (not the upstream default 8080) and start by writing a 210 W power cap to sysfs via `sudo`.

## Checkpoint pipeline (radiance)

AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` **does not load as-is**: its bf16 MTP head is mis-declared in the quark config (fails with `AssertionError: Attempted to load weight (5120, 10240) into (5120, 5120)`). The rewrite is mandatory:

1. `radiance/fp8_mtp.py <src> <dst>` → `Qwen3.8-27B-MXFP4-mtpfp8` (needs **torch**; `run_radiance.sh` spins a throwaway venv in `/models` for it)
2. `radiance/mtpfp8-pertoken.py <src config.json> <dst config.json>` → `-pertoken` variant (config-only, no torch; rewrites the 8 `mtp.*` `input_tensors` to per-channel; prints "expect 8")

Note: `radiance/fp8_mtp.py` and the copy in `radiance-vllm-mxfp4/` **differ**; `run_radiance.sh` uses the `radiance/` one.

`radiance/chat_template.jinja` is auto-downloaded from `froggeric/Qwen-Fixed-Chat-Templates` if missing; `run_radiance.sh` mounts it into the container.

## Radiance stack

The local flow builds its own image (`radiance-vllm-mxfp4:$(cat VERSION)`, currently 0.13.0) via `radiance-vllm-mxfp4/Dockerfile` — this differs from the subtree README's quickstart (pulls published `stilldeadcode/vllm-radiance:0.9.3`). For the radiance internals (knobs, troubleshooting, performance, ParoQuant, TP=3 padding), the subtree docs are the authoritative reference: `radiance-vllm-mxfp4/README.md`, `PERFORMANCE.md`, `PAROQUANT.md`, `DOCKERHUB.md`, `TP3_PADDING_PLAN.md`.

- All serving knobs are `${VAR:-default}` env overrides; nothing needs editing. `DRY_RUN=1 ./serve-mxfp4.sh` prints the container command; `./serve-mxfp4.sh --help` is the short reference.
- `run_radiance.sh` `sed`-rewrites the `--override-generation-config` line inside `serve-mxfp4.sh` on every start (top_k 20), then reverts the file via an `EXIT` trap (`git checkout`) when the script terminates — the local diff is transient, present only during the serve run.
- TP is auto-detected; 3/6/12 are never auto-picked (GDN head divisibility), and cards below 8 GiB are excluded (the box also has a 2 GiB iGPU render node).
- The stock libr4d kernel in published images **NaNs this model's GDN path** (perplexity in the hundreds of thousands); the launcher builds a pinned libr4d (`AUTO_R4D=1`, default). Don't run with `AUTO_R4D=0` unless you know why.

## libr4d

HIP kernel library for gfx1201. Entry-point names encode the exact geometry (compile-time constants; mismatches are rejected, not run). `r4d.select()` / `r4d.kernels()` answer "is there a kernel for this shape" — `None` means fall back to non-R4D paths. Build: `./build.sh` (needs `hipcc`, same ROCm as the runtime; the `radiance-vllm-mxfp4/Makefile` compiles it inside the image so it matches). Adding a kernel: 6-step process in `libr4d/README.md` (registry row in `r4d_registry.hip` is what makes it visible).

## hipfire

RDNA-native LLM inference engine (Rust + HIP + Redline; ROCm is dlopen'd at runtime). `./build_hipfire.sh` builds the `local/hipfire` deliverable image (daemon + native CLI + exact-source registry kernel packages compiled for all five archs) — no GPU needed at build time. The `Containerfile` base is `rocm/dev-ubuntu-24.04:7.2.4`; do not downgrade (gfx1151 needs ROCm 7.2+; gfx1201 needs 6.4+). If a build reports missing `hipcc`/HIP headers, switch the base tag to `7.2.4-complete`. Models are never baked in — mount a volume at `/root/.hipfire/models` (JIT kernel cache: `/var/cache/hipfire`). Runs need GPU passthrough (`--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined`); the OpenAI-compatible serve API is on port **11435**, loopback-bound by default.

## Verification

No CI, no pre-commit, no top-level test suite. Subtree checks that need no GPU: `tp3pad_selftest.py` (radiance), `make verify IMAGE=<rocm-image>` (libr4d). Otherwise the check is "does the server answer a request" — see the subtree READMEs.
