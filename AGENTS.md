# AGENTS.md

Host-orchestration repo for serving LLMs on a local AMD Radeon AI PRO R9700 (gfx1201 / RDNA4) box. Top-level `.sh`/`.py` files are the local flow; the three big directories are **git-subtree checkouts of external repos — not first-party code**.

## Subtrees (do not hand-edit)

- `llama.cpp/` — ggml-org/llama.cpp (master)
- `radiance-vllm-mxfp4/` — ggz14/radiance-vllm-mxfp4 (main)
- `libr4d/` — StillDeadcode/libr4d (main)

Sync all three with `./git.sh` (runs `git subtree add` or `pull` per dir). Changes belong in the upstream repos.

## The local flow (top-level scripts)

- `./build_radiance.sh` — then `./run_radiance.sh`. Both assume a **`/models` directory at filesystem root** (absolute path, not `~/models`) and **docker** (the subtree README prefers podman; the top-level scripts use docker). `build_radiance.sh` also writes the GPU power cap — sudo is required.
- `./build_llama.sh` — builds llama.cpp via `.devops/rocm.Dockerfile` with `AMDGPU_TARGETS=gfx1201`. `./run_llama.sh` serves the GGUF model.
- Both servers listen on port **9931** (not the upstream default 8080) and start by writing a 210 W power cap to sysfs via `sudo`.

## Checkpoint pipeline (radiance)

AMD's `Qwen3.8-27B-Quark-AWQ-MXFP4` **does not load as-is**: its bf16 MTP head is mis-declared in the quark config (fails with `AssertionError: Attempted to load weight (5120, 10240) into (5120, 5120)`). The rewrite is mandatory:

1. `fp8_mtp.py <src> <dst>` → `Qwen3.8-27B-MXFP4-mtpfp8` (needs **torch**; `build_radiance.sh` spins a throwaway venv in `/models` for it)
2. `mtpfp8-pertoken.py <src config.json> <dst config.json>` → `-pertoken` variant (config-only, no torch; rewrites the 8 `mtp.*` `input_tensors` to per-channel; prints "expect 8")

Note: `fp8_mtp.py` exists in two copies (top level and in `radiance-vllm-mxfp4/`) and **they differ**; `build_radiance.sh` uses the top-level one.

`chat_template.jinja` (top level) is auto-downloaded from `froggeric/Qwen-Fixed-Chat-Templates` if missing; `run_radiance.sh` mounts it into the container.

## Radiance stack

The local flow builds its own image (`radiance-vllm-mxfp4:$(cat VERSION)`, currently 0.13.0) via `radiance-vllm-mxfp4/Dockerfile` — this differs from the subtree README's quickstart (pulls published `stilldeadcode/vllm-radiance:0.9.3`). For the radiance internals (knobs, troubleshooting, performance, ParoQuant, TP=3 padding), the subtree docs are the authoritative reference: `radiance-vllm-mxfp4/README.md`, `PERFORMANCE.md`, `PAROQUANT.md`, `DOCKERHUB.md`, `TP3_PADDING_PLAN.md`.

- All serving knobs are `${VAR:-default}` env overrides; nothing needs editing. `DRY_RUN=1 ./serve-mxfp4.sh` prints the container command; `./serve-mxfp4.sh --help` is the short reference.
- `run_radiance.sh` `sed`-rewrites the `--override-generation-config` line inside `serve-mxfp4.sh` on every start (top_k 20) — a local diff there is expected, not drift.
- TP is auto-detected; 3/6/12 are never auto-picked (GDN head divisibility), and cards below 8 GiB are excluded (the box also has a 2 GiB iGPU render node).
- The stock libr4d kernel in published images **NaNs this model's GDN path** (perplexity in the hundreds of thousands); the launcher builds a pinned libr4d (`AUTO_R4D=1`, default). Don't run with `AUTO_R4D=0` unless you know why.

## libr4d

HIP kernel library for gfx1201. Entry-point names encode the exact geometry (compile-time constants; mismatches are rejected, not run). `r4d.select()` / `r4d.kernels()` answer "is there a kernel for this shape" — `None` means fall back to non-R4D paths. Build: `./build.sh` (needs `hipcc`, same ROCm as the runtime; the `radiance-vllm-mxfp4/Makefile` compiles it inside the image so it matches). Adding a kernel: 6-step process in `libr4d/README.md` (registry row in `r4d_registry.hip` is what makes it visible).

## Verification

No CI, no pre-commit, no top-level test suite. Subtree checks that need no GPU: `tp3pad_selftest.py` (radiance), `make verify IMAGE=<rocm-image>` (libr4d). Otherwise the check is "does the server answer a request" — see the subtree READMEs.
