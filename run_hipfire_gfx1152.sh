#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# gfx1152 (Krackan Point, Radeon 860M/840M/820M) entry point.
#
# Same flow as run_hipfire.sh, with four load-bearing differences:
# - builds --target full: the slim runtime image has no hipcc, and gfx1152
#   ships no prebuilt kernel pack, so runtime-only dies with "no usable
#   cached kernel image exists ... hipcc is unavailable" (docs/CONTAINER.md).
# - no prebuilt kernel pack exists for gfx1152: the registry pack builder's
#   SUPPORTED_ARCHES is gfx1201/gfx1100/gfx1151/gfx906/gfx942 (kernel_registry.rs:83)
#   and refuses anything else. Kernels therefore JIT-compile via hipcc on
#   first use and land in the hipfire-kcache volume; the first request is slow.
# - writes NO power cap: the 210 W sysfs line targets the discrete R9700;
#   capping the APU would throttle the whole SoC (CPU included).
# - small models by default: gfx1152 is an 8-CU UMA iGPU drawing from
#   system RAM (16-32 GB on Krackan laptops); the 27B only fits ~32 GB.
#
# Known upstream issue: #50 (gfx1152 segfault, open). If the daemon
# segfaults, report it upstream with a backtrace instead of debugging here.

# -------------------------------
# pre-flight: host must expose a gfx1152 KFD node
# -------------------------------
# gfx_target_version encoding is major * 10000 + minor * 100 + stepping
# (same table as hipfire/scripts/pp-gate.sh); 110502 = gfx1152. A host whose
# kernel/driver predates Krackan never enumerates the node, so this catches
# too-old hosts early instead of mid-build.
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
#   TAGS  - registry tag (`hipfire serve`/`pull` take this)
#   FILES - main model file under /models/hipfire (present = pulled)
TAGS=(
      "qwen3.5:9b"
      "qwen3.5:4b"
      "qwen3.8:27b-mq4-xts"
)
FILES=(
      "qwen3.5-9b.mq4"
      "qwen3.5-4b.mq4"
      "qwen3.8-27b.mq4-xts"
)

resolve_tag() {    # $1 -> model index on stdout; rc!=0 if no unique match
      local sel="$1" i hits=()
      if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#TAGS[@]} )); then
           printf '%s\n' $((sel-1)); return 0
      fi
      for i in "${!TAGS[@]}"; do
           [[ "${TAGS[$i]}" == *"$sel"* ]] && hits+=("$i")
      done
      if (( ${#hits[@]} == 1 )); then
           printf '%s\n' "${hits[0]}"
           return 0
      fi
      return 1
}

if [ -n "$1" ]; then
      # positional arg bypasses the menu (index or name)
      IDX="$(resolve_tag "$1")" || {
           echo "no model matching '$1'; options:" >&2
           for i in "${!TAGS[@]}"; do echo "     $((i+1))) ${TAGS[$i]}" >&2; done
           exit 1
      }
elif [ ! -t 0 ]; then
      # non-interactive, no arg: keep the current default
      IDX=0
else
      echo "Available models in the registry:"
      for i in "${!TAGS[@]}"; do echo "     $((i+1))) ${TAGS[$i]}" >&2; done
      read -rp "Select model [1-${#TAGS[@]}] (default 1): " pick
      pick=${pick:-1}
      IDX="$(resolve_tag "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
TAG="${TAGS[$IDX]}"
FILE="${FILES[$IDX]}"
echo "serving $TAG"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/hipfire:full"
MODELS_DIR=/models/hipfire
# the image's runtime stage sets HOME=/home/hipfire and HIPFIRE_HOME=/home/hipfire/.hipfire
HIPFIRE_HOME_DIR=/home/hipfire/.hipfire
mkdir -p "$MODELS_DIR"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    (
	cd "$CURDIR/hipfire" || exit
	docker build --progress=plain -f Containerfile --target full -t "$IMAGE" .
    )
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# model (pulled into host /models)
# -------------------------------
if [ ! -f "$MODELS_DIR/$FILE" ]; then
    docker run --rm --device /dev/kfd --device /dev/dri -v "$MODELS_DIR":"$HIPFIRE_HOME_DIR/models" "$IMAGE" pull "$TAG"
fi

# -------------------------------
# default reasoning effort (persistent config)
# -------------------------------
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
HOME_VOLUME=hipfire-home
echo "setting reasoning_effort=$REASONING_EFFORT in $HOME_VOLUME"
docker run --rm --device /dev/kfd --device /dev/dri -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" "$IMAGE" config set reasoning_effort "$REASONING_EFFORT"

# -------------------------------
# run (no power cap: APU, not the discrete R9700)
# -------------------------------
docker rm -f hipfire_server
docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" -v "$MODELS_DIR":"$HIPFIRE_HOME_DIR/models" -v hipfire-kcache:/var/cache/hipfire -e HIPFIRE_NGRAM_STORAGE=system -e HIPFIRE_VISION_MODE=off "$IMAGE" serve "$TAG" 0.0.0.0:9931
#docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" -v "$MODELS_DIR":"$HIPFIRE_HOME_DIR/models" -v hipfire-kcache:/var/cache/hipfire -e HIPFIRE_NGRAM_STORAGE=system -e HIPFIRE_VISION_MODE=off "$IMAGE" run "$TAG" "Explain FFT in one line"
