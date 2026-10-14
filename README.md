# radiance

An LLM and image-generation inference server for AMD GPUs, in C++ and HIP. Its HTTP API speaks OpenAI's
chat, completions, Responses and images APIs, Anthropic's Messages API and llama-server's native one,
so their clients and SDKs work unchanged. Model architectures, kernel libraries and quantisers are `.so` plugins; the ones in this tree are built
and loaded the same way a third party's are ([`docs/PLUGIN.md`](docs/PLUGIN.md)).

The bundled device kernels (`libr4d`) target **AMD RDNA4** (`gfx1200`, `gfx1201`). The serve flags
below are for **two 32 GB cards** (Radeon AI PRO R9700). The engine itself starts on any AMD GPU
family; another card needs a kernel library for it. A machine with no GPU builds the host backend,
which runs the full test suite but does not serve these models at any useful speed.

If you wish to support the developement of the radiance engine you can do so [here](https://patreon.com/StillDeadCode?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)

## Models

| model | container | get it | compose file |
|---|---|---|---|
| Qwen3.8-Flash-Next: 4-bit experts, int8 trunk, MTP, vision | `qwen3.8-next-flash-fp8-iq4r-moe.rad` (114 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe) | `flashnext.yaml` |
| Qwen3.8-27B FP8, DFlash2 drafter, vision | `qwen3.8-27b-fp8.rad` (29 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-fp8) | `qwen3.8-27b-fp8.yaml` |
| Qwen3.8-27B MXFP4 (AMD Quark AWQ), DFlash2 drafter, vision | `qwen3.8-27b-mxfp4.rad` (19 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.8-27b-mxfp4) | `qwen3.8-27b-mxfp4.yaml` |
| Qwen3.6-35B-A3B FP8, MTP | `qwen3.6-35b-a3b-fp8.rad` (35 GiB) | [download](https://huggingface.co/StillDeadcode/qwen3.6-35b-a3b-fp8) | `qwen3.6-35b-a3b.yaml` |
| MiniCPM5-2B FP8, DSpark drafter | `minicpm5-2b-fp8.rad` (3 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-fp8) | `minicpm5-2b.yaml` |
| MiniCPM5-2B bf16, DSpark drafter | `minicpm5-2b-bf16.rad` (5 GiB) | [download](https://huggingface.co/StillDeadcode/minicpm5-2b-bf16) | `minicpm5-2b-bf16.yaml` |
| Gemma 4 31B: GPTQ int4 trunk, int8 activations, MTP, vision | `gemma4-31b-w4a8.rad` (20 GiB) | [download](https://huggingface.co/keptn/gemma4-31b-w4a8) | `gemma4-31b-w4a8.yaml` |
| Qwen-Image 2.1 FP8: text to image and image edits | `qwen-image-2.1-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-fp8) | `qwen-image-2.1.yaml` |
| Qwen-Image 2.1 Turbo FP8: the same in 8 steps | `qwen-image-2.1-turbo-fp8.rad` (16 GiB) | [download](https://huggingface.co/StillDeadcode/radiance-qwen-image-2.1-turbo-fp8) | `qwen-image-2.1-turbo.yaml` |
| Qwen3.8-27B ParoQuant int4, int5 or MXFP4 (learned rotations), DFlash2 drafter, vision | `qwen3.8-27b-paro-int4.rad` (19 GiB; int5 22 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-paro.yaml` |
| Qwen3.8-27B bf16 | `qwen3.8-27b-bf16.rad` (50 GiB) | [convert](#convert-a-checkpoint) | `qwen3.8-27b-bf16.yaml` |
| Qwen3.8-Flash-Next bf16, a disk-streamed reference | `qwen3.8-flash-next-bf16.rad` (335 GiB) | [convert](#convert-a-checkpoint) | `flashnext-bf16.yaml` |
| EmbeddingGemma 2 bf16: text, image, video and audio embeddings, one card | `embeddinggemma2-bf16.rad` (1.4 GiB) | [convert](#convert-a-checkpoint) | `embeddinggemma2.yaml` |

A container holds everything the server needs: the weights, the tokeniser, the chat template and
any drafter. Each download link's repository has the recipe and the exact command that made the
file. Download one into your models directory (the transfer resumes if interrupted):

```sh
mkdir -p /srv/models/rad && cd /srv/models/rad
curl -LC - -O https://huggingface.co/StillDeadcode/minicpm5-2b-fp8/resolve/main/minicpm5-2b-fp8.rad
```

The Gemma 4 container is keptn42's: GPTQ calibrated in transformers, with an int8 lm_head; its
repository has the recipe and the calibration step. To convert Google's bf16 checkpoint yourself,
`data/recipes/gemma4-31b-dense-bf16-w4a8.recipe` quantises without calibration (round-to-nearest
with a scale search, bf16 lm_head); see [Convert a checkpoint](#convert-a-checkpoint). On two R9700s
that container decodes at 103 tok/s on prose and 141 on code, against 112 and 150 for keptn42's,
and its drafts are accepted a little less often (0.46 against 0.50 of them on prose).

The ParoQuant containers keep the trunk exactly as the ParoQuant releases ship it, with the learned
Givens rotations it was trained in. Convert int4 from `z-lab/Qwen3.8-27B-PARO`, int5 from
`Launch80/Qwen3.8-27B-PARO-int5` and MXFP4 from `MisterSnrub/Qwen3.8-27B-PARO-MXFP4-v2`, each with
`data/recipes/q38-27b-paro-mxfp4-df2.recipe` and the DFlash2 drafter. A ParoQuant container made
before 1.5.0 is refused at startup. Convert it again and pass the old one as `--reuse` to copy every
tensor that did not change.

The Qwen-Image containers serve OpenAI's images API (`/v1/images/generations`, `/v1/images/edits`):
pictures from a prompt, and edits of up to ten pictures with an optional mask, 2048x2048 by default.
On two R9700s a 2048x2048 picture takes 61 s at the base model's 40 steps and 13 s with Turbo (56 and
12 s with `--tp-wire wht6`); 1024x1024 takes 6 s and 2.6 s. Qwen-Image 2.1 is under Qwen's research
license, which is non-commercial.

## Run with Docker

The host needs Docker and an `amdgpu` driver that supports ROCm 10.1; it does not need ROCm. The
card needs Resizable BAR on (Above 4G Decoding and Resizable BAR in the firmware setup). The
image is [`stilldeadcode/radiance`](https://hub.docker.com/r/stilldeadcode/radiance), with its
kernels built for `gfx1201`. Pull it, clone this repository for its compose files, then start a
model from one of them:

```sh
docker pull stilldeadcode/radiance
git clone https://codeberg.org/StillDeadcode/radiance.git
cp -r radiance/deploy/compose ~/radiance-compose && cd ~/radiance-compose
cp .env.example .env                         # set RADIANCE_MODELS, RADIANCE_API_KEY, the GIDs
docker compose -f minicpm5-2b.yaml up -d     # or any file from the table
docker compose -f minicpm5-2b.yaml logs -f
docker compose -f minicpm5-2b.yaml down
```

Every model uses both cards and the same port, so run one at a time. Running without compose, and
the reason for each `docker run` option, is in [`docs/DOCKER.md`](docs/DOCKER.md).

To build the image yourself, for example for other cards, run `docker/build.sh -t 'gfx1200;gfx1201'`
in the clone (it needs `docker buildx` and tests the build as it goes). Then set
`RADIANCE_IMAGE=radiance:latest` in `.env`.

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+ and Ninja. For the device kernels you also need
ROCm 10.1 or 7.2 (`hipcc`; set `ROCM_PATH` if it is not `/opt/rocm`). Clone the repository and
build in it:

```sh
git clone https://codeberg.org/StillDeadcode/radiance.git && cd radiance
./build.sh                                   # configure, build for the cards present, run the tests
./build.sh --no-hip                          # host backend only
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201'   # build for cards this machine lacks
cmake --install build --prefix /opt/radiance # optional; bin/, include/, share/radiance/
```

With ROCm 10.1 the build turns the compiler's VOPD pairing (dual-issue instruction pairs) off: its
LLVM 24 pairs in an order that follows heap addresses, so two builds of the same source can differ.
The Docker image builds its compiler with the fix and keeps the pairing, which is 1-3% faster on
the fp8 prefill GEMMs; a compiler that has the same fix is declared with
`-DRAD_HIP_VOPD_REPEATABLE=ON`. ROCm 7.2 is unaffected.

## Run without Docker

```sh
export RADIANCE_HOME=$PWD/build/radiance_home   # the plugins; an installed prefix finds its own
build/bin/radiance --model /srv/models/rad/minicpm5-2b-fp8.rad --port 8000 <flags>
build/bin/radiance --model StillDeadcode/minicpm5-2b-fp8 --port 8000 <flags>
```

`--model` also takes a Hugging Face repo id, as vLLM does. The server fetches the repo with `curl`
and `jq` into `~/.cache/radiance/hf` (`--download-dir` puts it elsewhere), serves the container it
holds, and checks it for changes at every start; a start that cannot reach the hub serves the copy
it has. The Docker image has neither tool, so with Docker download the file as shown above.

These are the flags each model is served with. They match its compose file, minus Flash-Next's
prefix-cache tiers:

| model | flags |
|---|---|
| Qwen3.8-Flash-Next | `--tp 2 --tp-wire wht6 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, four ranks | `--tp 4 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 12288 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-Flash-Next, three ranks (the shipped container as it is, [`docs/TP3.md`](docs/TP3.md)) | `--tp 3 --max-num-seqs 8 --max-model-len 200000 --placement expert_tiered --host-pool-mib 8192 --gpu-headroom-mib 96 --kv-cache-dtype fp8 --num-speculative-tokens 3 --max-num-batched-tokens 2048` |
| Qwen3.8-27B FP8, Qwen3.6-35B-A3B | `--tp 2 --max-num-seqs 32 --max-model-len 200000 --kv-cache-dtype fp8` |
| Qwen3.8-27B bf16 | `--tp 2 --max-num-seqs 8 --max-model-len 200000 --kv-cache-dtype fp8` |
| MiniCPM5-2B FP8 and bf16 | `--tp 2 --max-num-seqs 32 --max-model-len 131072 --kv-cache-dtype fp8 --gpu-headroom-mib 160 --max-num-batched-tokens 512` |
| Qwen3.8-Flash-Next bf16 | `--tp 2 --max-model-len 32768 --placement expert_tiered --weights-disk-tier --host-pool-mib 12288 --gpu-headroom-mib 768 --deterministic --max-num-seqs 32 --max-num-batched-tokens 32768 --num-speculative-tokens 0` |
| Qwen-Image 2.1 and 2.1 Turbo | `--tp 2 --tp-wire wht6` |

Every model serves up to 32 sequences a step. Two rows keep 8, because on a hybrid model each
sequence holds its own recurrent state, and 32 of them take that memory from the attention cache
and, on Flash-Next, from the expert slab: Flash-Next and the 27B in bf16. See
[`docs/GUIDE.md`](docs/GUIDE.md) §4.4.

At `--tp 4` the gated residual's write carries the exact all-reduce, with the MoE gather in front
of it, on decode and verify steps; a prefill chunk runs the unfused chain. That chain's exact
all-reduce is one-shot up to 1 MiB a message and two-shot above that, which sends half the bytes
on a prefill chunk; both produce the same bits. The row above runs exact.
`--tp-wire wht6` at three and four ranks selects a different lossy wire, tiered int8 over the
two-shot, above `--tp-wire-min-kb`. Eight ranks have no lossy wire, so `--tp 8 --tp-wire wht6`
stops at startup naming the all-reduce. Four ranks is a capacity configuration; its per-step rate
against `--tp 2` on the same cards has not been measured. The
tiering flags are the tp2 recipe carried over. Above two ranks each rank still owns a **slice of
every expert's width** — each parity's 128-column blocks are cut contiguously over the ranks and the
odd block alternates with the parity, so every rank's share is whole blocks and every block has
exactly one owner — and the resident expert bytes per rank are smaller, so the elastic window is a
different shape: read the boot log's `elastic` line and re-derive `--host-pool-mib` before
trusting it. **`--host-pool-mib` is pinned on every card**, so the row
above pins 4 × 12 GiB = 48 GiB of RAM that nothing else can use and the kernel cannot swap out. The
amdgpu driver caps what every process together pins at the machine's RAM less 1/64 and 1.5 GiB; the
pool is locked user memory, which `ttm.pages_limit` does not count. A 64 GiB machine serves
`flashnext-tp1.yaml` with 49 GiB pinned ([`docs/GUIDE.md`](docs/GUIDE.md) §7).

Add `--api-key KEY` to require `Authorization: Bearer KEY` (or `X-Api-Key: KEY`). Add `--debug-graph` to print every op
and the kernel it resolved to, then exit without serving. Then:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

An image model answers OpenAI's images API instead, with base64 pictures:

```sh
curl -s localhost:8000/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse on a cliff at sunset, oil painting","size":"1024x1024"}' \
  | python3 -c 'import sys,json,base64; open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`http://localhost:8000/` is the dashboard. [`docs/GUIDE.md`](docs/GUIDE.md) covers the API, memory
budgets, long context, speculative decoding and operating a server.

## Convert a checkpoint

`rad-convert` turns a Hugging Face checkpoint into a container, quantising what a recipe in
`data/recipes/` names. Download checkpoints with `scripts/hfget.sh <org/repo> <dir>`.

```sh
build/bin/rad-convert ~/models/Qwen/Qwen3.8-27B -o qwen3.8-27b-bf16.rad \
    --recipe data/recipes/q38-27b-bf16.recipe
build/bin/rad-convert ~/models/Qwen/Qwen3.8-Flash-Next -o qwen3.8-flash-next-bf16.rad --max-ctx 4096
build/bin/rad-convert ~/models/google/gemma-4-31B-it -o gemma4-31b-w4a8.rad \
    --recipe data/recipes/gemma4-31b-dense-bf16-w4a8.recipe \
    --set quantization_config.quant_method=compressed-tensors \
    --draft-model ~/models/google/gemma-4-31B-it-assistant
build/bin/rad-convert ~/models/z-lab/Qwen3.8-27B-PARO -o qwen3.8-27b-paro-int4.rad \
    --recipe data/recipes/q38-27b-paro-mxfp4-df2.recipe \
    --draft-model ~/models/z-lab/Qwen3.8-27B-DFlash2
```

With Docker, use `docker run --rm -v ~/models:/models --entrypoint rad-convert stilldeadcode/radiance ...`. In
the image the recipes are in `/opt/radiance/share/radiance/recipes/`. A checkpoint served at its own
precision needs no container: `radiance --model ~/models/Qwen/Qwen3.8-27B-FP8`.

## Test

```sh
ctest --test-dir build -LE 'gpu|oot'   # everything that runs without a card (a few minutes)
ctest --test-dir build -L gpu          # device suites; they need the cards to themselves
ctest --test-dir build -L oot          # rebuilds every bundled plugin out of tree against an install
```

## Documentation

| | |
|---|---|
| [`docs/GUIDE.md`](docs/GUIDE.md) | installing, converting, serving, the HTTP API, operating a server |
| [`docs/TOOLS.md`](docs/TOOLS.md) | every flag of every tool, and the recipe syntax |
| [`docs/DOCKER.md`](docs/DOCKER.md) | the image and the compose files |
| [`docs/PLUGIN.md`](docs/PLUGIN.md) | writing a kernel library, architecture or quantiser plugin, including one for another AMD card |
| [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) | the supported model families and their geometry |
| [`docs/TP3.md`](docs/TP3.md) | any world size: the attention world and uneven delta-net heads; three ranks |
| [`docs/OPS.md`](docs/OPS.md), [`docs/KERNELS.md`](docs/KERNELS.md) | the op vocabulary, and what `rad-kbench` last measured |
| [`spec.md`](spec.md) | the design |

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from. The Docker image also holds third-party libraries under their own licenses, GPL and LGPL
among them; `/usr/share/doc/THIRD-PARTY` in the image lists each one with its version and source.
