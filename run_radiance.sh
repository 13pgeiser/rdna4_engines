#!/bin/bash
# radiance .rad containers: 27b MXFP4 + DFlash2 drafter, flash-next fp8 trunk + 4-bit codebook MoE, or a 35B-A3B MoE (qwen3.6 / ornith).
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="/models/rad"

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# parallel arrays, one entry per model, indexed together:
#   FILES  - .rad file under /models/rad (absent = downloaded)
#   REPOS  - Hugging Face repo the file is served from
#   EXTRAS - extra serve args for this model
FILES=(
      "qwen3.8-27b-mxfp4.rad"
      "qwen3.8-next-flash-fp8-iq4r-moe.rad"
      "qwen3.6-35b-a3b-fp8.rad"
      "ornith-1.5-35b-a3b-fp8.rad"
)
REPOS=(
      "StillDeadcode/qwen3.8-27b-mxfp4"
      "StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe"
      "StillDeadcode/qwen3.6-35b-a3b-fp8"
      "keptn/ornith-1.5-35b-a3b-fp8"
)
EXTRAS=(
      "--max-num-seqs 1 --max-model-len 262144"
      "--max-num-seqs 1 --max-model-len 132144 --weights-disk-tier --host-pool-mib 115000 --embedding-placement ram --kv-cache-dtype fp8"
      "--max-num-seqs 1 --max-model-len 200000 --host-pool-mib 16384 --embedding-placement ram --kv-cache-dtype fp8"
      "--max-num-seqs 1 --max-model-len 262144 --num-speculative-tokens 3 --max-num-batched-tokens 2048 --host-pool-mib 16384 --embedding-placement ram --kv-cache-dtype fp8"
)
# Per-model docker -e args. The two 35B-A3B models (16 query heads over 2 KV heads, head_dim
# 256, gqa 8) abort at a 3-4 row speculative verify: the attn_paged band q_len<=5 (its edge is
# the gqa-12 rows' q_len*gqa<=64 bound) sizes its split-KV scratch at the band top, where the
# split law (q_len*gqa 40 > 32 rows) picks 64 segments, but a 4-row verify (32 rows) picks 128
# and the kernel refuses the undersized buffer (RAD_E_SCRATCH). Pinning splits to 64 keeps the
# declare-time sizing and the launch in agreement. Workaround until the radiance scratch hook
# (g8_decode_scratch) takes the band's q_len worst case instead of its top; costs some decode
# speed while the law would use 128 segments (256 vs 128 workgroups at q_len <= 4).
ENV=(
      ""
      ""
      "RADIANCE_TUNE=attn_paged:splits=64"
      "RADIANCE_TUNE=attn_paged:splits=64"
)

resolve_model() {    # $1 -> model index on stdout; rc!=0 if no unique match
      local sel="$1" i hits=()
      if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#FILES[@]} )); then
           printf '%s\n' $((sel-1)); return 0
      fi
      for i in "${!FILES[@]}"; do
           [[ "${FILES[$i]}" == *"$sel"* ]] && hits+=("$i")
      done
      if (( ${#hits[@]} == 1 )); then
           printf '%s\n' "${hits[0]}"; return 0
      fi
      return 1
}

if [ -n "$1" ]; then
      # positional arg bypasses the menu (index or name)
      IDX="$(resolve_model "$1")" || {
           echo "no model matching '$1'; options:" >&2
           for i in "${!FILES[@]}"; do echo "     $((i+1))) ${FILES[$i]}" >&2; done
           exit 1
      }
elif [ ! -t 0 ]; then
      # non-interactive, no arg: keep the current default
      IDX=0
else
      echo "Available radiance models in $MODEL_DIR:"
      for i in "${!FILES[@]}"; do echo "     $((i+1))) ${FILES[$i]}"; done
      read -rp "Select model [1-${#FILES[@]}] (default 1): " pick
      pick=${pick:-1}
      IDX="$(resolve_model "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
MODEL_FILE="${FILES[$IDX]}"
REPO="${REPOS[$IDX]}"
EXTRA_FLAGS="${EXTRAS[$IDX]}"
EXTRA_ENV=()
[ -n "${ENV[$IDX]}" ] && EXTRA_ENV=(-e "${ENV[$IDX]}")
echo "serving $MODEL_FILE with EXTRA_FLAGS=$EXTRA_FLAGS"
[ -n "${ENV[$IDX]}" ] && echo "with ${ENV[$IDX]}"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/radiance"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    # The in-image host test suite has tight CPU budgets that this machine runs over: relax
    # tokenizer_test 20 -> 25 s, give the gbnf_test shards and glm5next_test 15 s (default 10 s),
    # and give the upgraded arch_test 20 s (it spends ~11 s of CPU here, over the 10 s default)
    # for the build.
    # Upstream 6c33ee4e (DFlash2 block coupling) reworded the engine's --draft-coupling refusal
    # without updating the expected string in engine_e2e_test, so
    # the_coupling_flag_refuses_what_it_cannot_serve_at_startup fails on every build, not just
    # slow machines: make it expect a prefix of the new message until the test is fixed upstream.
    # Budgets are looked up by binary name, so gbnf_test covers all four shards. Revert on a
    # successful build to keep the subtree clean; a failed build leaves the edits in place.
    cmake="$CURDIR/radiance/tests/CMakeLists.txt"
    e2e="$CURDIR/radiance/tests/engine_e2e_test.cpp"
    sed -i 's/^set(RAD_TEST_CPU_BUDGET_tokenizer_test 20)/set(RAD_TEST_CPU_BUDGET_tokenizer_test 25)/' "$cmake"
    grep -q '^set(RAD_TEST_CPU_BUDGET_gbnf_test 15)$' "$cmake" || \
	sed -i '/^set(RAD_TEST_CPU_BUDGET_tokenizer_test 25)/a set(RAD_TEST_CPU_BUDGET_gbnf_test 15)' "$cmake"
    grep -q '^set(RAD_TEST_CPU_BUDGET_arch_test 20)$' "$cmake" || \
	sed -i '/^set(RAD_TEST_CPU_BUDGET_gbnf_test 15)/a set(RAD_TEST_CPU_BUDGET_arch_test 20)' "$cmake"
    grep -q '^set(RAD_TEST_CPU_BUDGET_glm5next_test 15)$' "$cmake" || \
	sed -i '/^set(RAD_TEST_CPU_BUDGET_arch_test 20)/a set(RAD_TEST_CPU_BUDGET_glm5next_test 15)' "$cmake"
    sed -i 's/"--draft-coupling on requires the linear serial draft chain "/"--draft-coupling on requires a linear draft chain (a serial head or a block drafter with a coupled walk, depth > 0)";/' "$e2e"
    sed -i '/"(RAD_DRAFT_SERIAL with depth > 0)";$/d' "$e2e"
    (
	cd "$CURDIR/radiance" || exit
	docker build --progress=plain -t "$IMAGE" --build-arg GPU_TARGETS=gfx1201 .
    ) || { echo "image build failed; $cmake and $e2e left modified" >&2; exit 1; }
    git -C "$CURDIR/radiance" checkout -- tests/CMakeLists.txt tests/engine_e2e_test.cpp
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# model (downloaded into host $MODEL_DIR)
# -------------------------------
if [ ! -e "$MODEL_DIR/$MODEL_FILE" ]; then
    (
	mkdir -p "$MODEL_DIR" || exit
	cd "$MODEL_DIR" || exit
	curl -LC - -O "https://huggingface.co/$REPO/resolve/main/$MODEL_FILE"
    )
fi

# -------------------------------
# vmemoff shim (KMT workaround, see radiance-vmemoff/vmemoff.c)
# -------------------------------
# WHY THIS EXISTS -- radiance hangs at startup on this host, with nothing wrong in radiance.
#
# Symptom: the server dies after loading weights and printing "chunk geometry", before
# /health ever answers. GPU sits idle (~15 W), the stallwatch never fires (it skips
# threads blocked in a driver ioctl), and every amdgpu ring shows clean fences.
#
# The hang point (gdb stack of the hung process, all inside libhsa-runtime64.so.1.21.0):
#
#   core/mem/kv.cpp  KV pool "backing" commit      (elastic vmem pool, configure_kv)
#     -> core/device/aql.cpp  a_vmem_commit()
#        -> hsa_amd_vmem_set_access(GPU)
#           -> rocr::core::Runtime::VMemorySetAccess
#              -> MappedHandleAllowedAgent::EnableAccess
#                 -> rocr::AMD::KfdDriver::Map
#                    -> hsaKmtMemoryVaMap                      (KMT = KFD Memory Table)
#                       -> DRM_IOCTL_SYNCOBJ_ARRAY_WAIT        (renderD128) -- never returns
#
# KMT maps the pool's virtual range into the GPU page tables by having KIQ issue a TLB
# flush, and then waits on the resulting fence through a DRM syncobj. On this box
# (kernel 6.12.113+deb13, gfx1201/RDNA4) that fence is never signalled: ftrace shows a
# kernel worker spinning on MES/KIQ registers forever, the flush never completes, and the
# driver has no timeout or reset on this path.
#
# How it was established (2026-10): reproduced without any radiance code by a 64 MiB
# standalone HSA program doing reserve -> map -> set_access; the GPU-access call hangs,
# the CPU-access call on the same mapping returns instantly; every size tested hangs; the
# hang is intermittent (one success out of eight tries), which is why the log looks random.
# hipfire, llama.cpp and the vLLM stack all work fine on this same HSA runtime because
# none of them use the HSA vmem (KMT) path -- they use plain hipMalloc.
#
# THE WORKAROUND: radiance has a supported "no virtual memory management" mode.
# VMem::supported() (core/device/dispatch.cpp:231) is false when the VRAM pool reports a
# zero RUNTIME_ALLOC_REC_GRANULE, and Pool::init_elastic() then falls back to init(): the
# KV pool is allocated whole, once, through the plain allocation path -- which works here.
# The vmemoff.so shim (radiance-vmemoff/) intercepts hsa_amd_memory_pool_get_info and
# zeroes exactly that one field; nothing else is touched.
#
# COST: the KV pool holds its full budget for the whole run instead of committing elastically.
# Irrelevant at TP=1 with one model served, this is a memory-flexibility trade, not a
# correctness one.
#
# TESTING A REAL FIX: VMEMOFF=0 ./run_radiance.sh runs plain local/radiance without the shim.
# Candidates: a ROCm 7.14.1 build (Dockerfile's commented ARG), a newer host kernel, or an
# upstream fix in the ROCm HSA runtime's KMT code (this is worth reporting to AMD: fence
# never signalled, no driver timeout). Once one is verified, drop this block and the
# PRELOAD line below it.
#
# MECHANICS: vmemoff.so is gitignored (only the source is tracked); if missing it is
# rebuilt from vmemoff.c with host gcc (self-contained source), falling back to a build
# inside the radiance-build image. local/radiance-vmemoff is derived from local/radiance
# with the shim at /opt/vmemoff.so, only while it is missing, so it survives docker
# system prune -a.
VMEMOFF="${VMEMOFF:-0}"
if [ "$VMEMOFF" = 1 ]; then
    shim="$CURDIR/radiance-vmemoff/vmemoff.so"
    if [ ! -e "$shim" ]; then
        if command -v gcc >/dev/null 2>&1; then
            # vmemoff.c is self-contained (no ROCm headers): plain host gcc suffices.
            gcc -shared -fPIC -O2 -o "$shim" "$CURDIR/radiance-vmemoff/vmemoff.c" -ldl
        elif docker image inspect radiance-build >/dev/null 2>&1; then
            docker rm -f vmemoff_build >/dev/null 2>&1
            docker create --name vmemoff_build radiance-build sleep 60 >/dev/null
            docker cp "$CURDIR/radiance-vmemoff/vmemoff.c" vmemoff_build:/tmp/vmemoff.c
            docker start vmemoff_build >/dev/null
            docker exec vmemoff_build gcc -shared -fPIC -O2 -o /tmp/vmemoff.so /tmp/vmemoff.c -I/opt/rocm/include \
                || { docker rm -f vmemoff_build >/dev/null; exit 1; }
            docker cp vmemoff_build:/tmp/vmemoff.so "$shim" || exit 1
            docker rm -f vmemoff_build >/dev/null
        fi
    fi
    [ -e "$shim" ] || { echo "vmemoff shim missing: $shim (no host gcc, no radiance-build image to rebuild it)" >&2; exit 1; }
    vmemoff_image="local/radiance-vmemoff"
    if ! docker image inspect "$vmemoff_image" >/dev/null 2>&1; then
        docker rm -f vmemoff_img >/dev/null 2>&1
        docker create --name vmemoff_img "$IMAGE" sleep 60 >/dev/null || exit 1
        docker cp "$shim" vmemoff_img:/opt/vmemoff.so || exit 1
        docker commit vmemoff_img "$vmemoff_image" >/dev/null || exit 1
        docker rm -f vmemoff_img >/dev/null
        echo "derived $vmemoff_image from $IMAGE + vmemoff shim"
    fi
    IMAGE="$vmemoff_image"
fi
PRELOAD=()
[ "$VMEMOFF" = 1 ] && PRELOAD=(-e LD_PRELOAD=/opt/vmemoff.so)

# -------------------------------
# run
# -------------------------------
#sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/power1_cap)"
docker rm -f radiance_server
#IMAGE="stilldeadcode/radiance:1.7.0"
docker run --name radiance_server --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v "$MODEL_DIR":/models:ro \
    -e RADIANCE_LOG=info \
    "${EXTRA_ENV[@]}" \
    "${PRELOAD[@]}" \
    "$IMAGE" --model /models/"$MODEL_FILE" \
    --host :: --port 9931 --tp 1 $EXTRA_FLAGS --kv-cache-dtype fp8 --debug-selection --debug-placement

# qwen3.8-27b-mxfp4.rad
# 
# llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9931/v1 --models openai:qwen3.8:27b-mq4-xts --prompts "Write a rust function to parse JSON safely."
# → Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests
# 
# Results: 3 succeeded, 0 failed (100% success rate)
# 
# ╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
# │ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
# ├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
# │ openai   │ qwen3.8:27b-mq4-xts │     6613 │     8175 │  73.5 │      n/a │        n/a │    100% │
# ╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯
