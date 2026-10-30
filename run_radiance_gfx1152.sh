#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# gfx1152 (Krackan Point, Radeon 860M/840M/820M) entry point for the radiance engine.
#
# Same flow as run_radiance.sh, with load-bearing differences:
# - built with GPU_TARGETS=gfx1152: libr4d serves gfx12 only and is dropped, so the
#   device kernel library is libr3d (RDNA3/RDNA3.5, ^gfx11 family filter; the chip
#   object is preferred over gfx11-generic by the loader). The engine's own kernels
#   carry the generic family targets.
# - the test stage is skipped: its llvm-check.sh reruns libr4d's compile command
#   (ninja -t commands libr4d), which does not exist in a gfx11-only build, and the
#   ctest host suite is the slow part of the build. llvm-check is seded to `true`
#   for the duration of the build (reverted after) and TEST_EXCLUDE='.*' leaves
#   zero ctest entries.
# - minicpm5-2b is the model this library was built and measured for (its row
#   tables' "served geometry" is minicpm5-2b's exact shape: 42 layers, 16/2 GQA 8,
#   head_dim 128, block 16, fp8 cache, 3-layer dspark drafter) -- fully on-device.
# - qwen3.8-27b-mxfp4 is kept as an experiment only: its 48 GDN layers and 16
#   head_dim-256 gated-attention layers have no libr3d rows (libr4d-only), so by
#   default startup stops with RAD_E_NOKERNEL; --debug-accept-reference-kernels
#   runs those ops on the CPU libref. GEMMs still run on the GPU; expect it to be
#   slow, especially prefill.
# - writes NO power cap: the 210 W sysfs line targets the discrete R9700;
#   capping the APU would throttle the whole SoC (CPU included).

# -------------------------------
# pre-flight: host must expose a gfx1152 KFD node
# -------------------------------
# gfx_target_version encoding is major * 10000 + minor * 100 + stepping;
# 110502 = gfx1152. A host whose kernel/driver predates Krackan never enumerates
# the node, so this catches too-old hosts early instead of mid-build.
KFD_NODES=/sys/class/kfd/kfd/topology/nodes
if [ ! -d "$KFD_NODES" ]; then
    echo "no KFD topology at $KFD_NODES (amdgpu driver/KFD not loaded?)" >&2
    exit 1
fi
GPU_ARCHS=""
for nodedir in "$KFD_NODES"/*/; do
    props="$nodedir/properties"
    [ -r "$props" ] || continue
    simd=$(awk '/^simd_count/ {print $2; exit}' "$props")
    gfx=$(awk '/^gfx_target_version/ {print $2; exit}' "$props")
    [ -n "$simd" ] || continue
    [ "$simd" != "0" ] || continue
    [ -n "$gfx" ] || continue
    [ "$gfx" != "0" ] || continue
    GPU_ARCHS="$GPU_ARCHS $gfx"
done
case "$GPU_ARCHS" in
    *110502*) echo "host exposes gfx1152 (gfx_target_version 110502)" ;;
    "") echo "no GPU KFD node found (no /sys/class/kfd/kfd/topology/nodes/*/properties with simd_count > 0)" >&2
        exit 1 ;;
    *) echo "no gfx1152 KFD node; found gfx_target_version:$GPU_ARCHS" >&2
       echo "this script is for Krackan Point (gfx1152) boxes" >&2
       exit 1 ;;
esac

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# parallel arrays, one entry per model, indexed together:
#   FILES  - .rad file under /models/rad (absent = downloaded)
#   REPOS  - Hugging Face repo the file is served from
#   EXTRAS - extra serve args for this model
FILES=(
      "minicpm5-2b-fp8.rad"
      "minicpm5-2b-bf16.rad"
      "qwen3.8-27b-mxfp4.rad"
)
REPOS=(
      "StillDeadcode/minicpm5-2b-fp8"
      "StillDeadcode/minicpm5-2b-bf16"
      "StillDeadcode/qwen3.8-27b-mxfp4"
)
EXTRAS=(
      "--max-num-seqs 1 --max-model-len 32768"
      "--max-num-seqs 1 --max-model-len 32768"
      # experiment: GDN (48 layers) and h256 gated attention (16 layers) resolve to
      # the CPU libref; drafter off for clean AR numbers; the 27B vision tower lives
      # in system RAM to keep the GPU pool for text + KV; small context (15 GB of
      # weights leave little headroom on a 16.6 GB UMA card).
      "--max-num-seqs 1 --max-model-len 8192 --num-speculative-tokens 0 --debug-accept-reference-kernels --vision-tower-placement ram"
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
           printf '%s\n' "${hits[0]}"
           return 0
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
      echo "Available radiance models in /models/rad:"
      for i in "${!FILES[@]}"; do echo "     $((i+1))) ${FILES[$i]}"; done
      read -rp "Select model [1-${#FILES[@]}] (default 1): " pick
      pick=${pick:-1}
      IDX="$(resolve_model "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
MODEL_FILE="${FILES[$IDX]}"
REPO="${REPOS[$IDX]}"
EXTRA_FLAGS="${EXTRAS[$IDX]}"
echo "serving $MODEL_FILE with EXTRA_FLAGS=$EXTRA_FLAGS"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/radiance:gfx1152"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    # Skip the test stage: llvm-check.sh needs libr4d's compile command, which a
    # gfx11-only build has no target for, and the ctest suite is the slow part of
    # the build. The sed is transient: reverted after a successful build (a failed
    # build leaves the edit in place, like run_radiance.sh's budget edits). The
    # first build on a box is long regardless (ffmpeg + libLLVM stages).
    df="$CURDIR/radiance/Dockerfile"
    grep -q 'bash /src/docker/llvm-check.sh /build' "$df" || {
        echo "Dockerfile test stage changed upstream; adjust the sed below before building" >&2
        exit 1
    }
    sed -i 's|bash /src/docker/llvm-check.sh /build|true|' "$df"
    (
        cd "$CURDIR/radiance" || exit
        docker build --progress=plain -t "$IMAGE" \
            --build-arg GPU_TARGETS=gfx1152 \
            --build-arg TEST_EXCLUDE='.*' .
    ) || { echo "image build failed; $df left modified" >&2; exit 1; }
    git -C "$CURDIR/radiance" checkout -- Dockerfile
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# model (downloaded into host $MODEL_DIR)
# -------------------------------
MODEL_DIR="/models/rad"
if [ ! -e "$MODEL_DIR/$MODEL_FILE" ]; then
    (
        mkdir -p "$MODEL_DIR" || exit
        cd "$MODEL_DIR" || exit
        curl -LC - -O "https://huggingface.co/$REPO/resolve/main/$MODEL_FILE"
    )
fi

# -------------------------------
# run (no power cap: APU, not the discrete R9700)
# -------------------------------
docker rm -f radiance_server
docker run --name radiance_server --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add video --group-add render \
    -v "$MODEL_DIR":/models:ro \
    -e RADIANCE_LOG=info \
    "$IMAGE" --model /models/"$MODEL_FILE" \
    --host :: --port 9931 --tp 1 --kv-cache-dtype fp8 $EXTRA_FLAGS

# curl -s http://localhost:9931/v1/models     # served model name for the bench
# llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 \
#     --models openai:minicpm5-2b-fp8 --prompts "Write a rust function to parse JSON safely."
