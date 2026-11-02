#!/bin/bash
# FastFlowLM (FLM): chat models running on the Ryzen AI XDNA2 NPU, built from the FastFlowLM/ subtree.
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURDIR/common.sh" || exit 1

# -------------------------------
# GPU arch gate: gfx1152 only (early exit; detection in common.sh)
# -------------------------------
gpu_arch_gate run_fastflowlm.sh gfx1152

# -------------------------------
# NPU gate: FLM runs on the XDNA2 NPU, not the GPU
# -------------------------------
if ! lsmod | grep -q '^amdxdna'; then
    echo "amdxdna kernel module not loaded: FLM needs the NPU driver (kernel 7.0+ in-tree, or amdxdna-dkms)" >&2
    exit 1
fi
if ! ls /dev/accel/accel* >/dev/null 2>&1; then
    echo "no NPU device node under /dev/accel: nothing to serve on" >&2
    exit 1
fi

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# TAGS - registry tags (`flm serve`/`pull` take this)
TAGS=(
      "gemma4-it:12b"
      "qwen3.5:9b"
      "nanbeige4.1:3b"
      "phi4-mini-it-core:4b"
      "gpt-oss:20b"
      "qwen3.6-moe:35b-a3b"
      "qwen3.8-dflash2:27b"   # source catalog only (unreleased in the 1.0.11 tarballs)
      "qwen3.8-mtp:27b"       # the 27B with MTP speculative decoding
)

select_model "Available FastFlowLM models:" "$1" "${TAGS[@]}"
TAG="${TAGS[$MODEL_IDX]}"
echo "serving $TAG"

# -------------------------------
# docker image (built from the FastFlowLM/ subtree; the tag is the subtree's git
# tree hash, so a ./git.sh sync that moves FastFlowLM/ builds a fresh image, old
# ones survive for A/B testing, and unrelated commits reuse the existing one)
# -------------------------------
FLM_SRC_REV="$(git -C "$CURDIR" rev-parse --short=8 HEAD:FastFlowLM)"
IMAGE="local/fastflowlm-src:$FLM_SRC_REV"
HAVE_REV="$(docker image inspect --format '{{index .Config.Labels "flm.src-rev"}}' "$IMAGE" 2>/dev/null)"
if [ "$HAVE_REV" = "$FLM_SRC_REV" ]; then
    echo "image $IMAGE already present (subtree rev $FLM_SRC_REV), skipping build"
else
    # the third_party submodules are gitlinks and arrive empty: clone them at the pinned revisions
    docker build --progress=plain -t "$IMAGE" \
        --build-arg TOKENIZERS_REVISION="$(git -C "$CURDIR" ls-tree HEAD FastFlowLM/third_party/tokenizers-cpp | awk '{print $3}')" \
        --build-arg XGRAMMAR_REVISION="$(git -C "$CURDIR" ls-tree HEAD FastFlowLM/third_party/xgrammar | awk '{print $3}')" \
        --label flm.src-rev="$FLM_SRC_REV" \
        -f "$CURDIR/Dockerfile.fastflowlm-src" "$CURDIR/FastFlowLM" \
        || { echo "image build failed" >&2; exit 1; }
fi

# -------------------------------
# model (pulled into host /models/flm; `flm pull` is a no-op when present)
# -------------------------------
MODELS_DIR=/models/flm
mkdir -p "$MODELS_DIR" || exit 1
docker run --rm -v "$MODELS_DIR":/models "$IMAGE" flm pull "$TAG" \
    || { echo "flm pull $TAG failed (tag not in this FLM build's catalog - check 'flm list')" >&2; exit 1; }

# -------------------------------
# run
# -------------------------------
docker rm -f fastflowlm_server
# memlock must be unlimited for NPU execution; seccomp unconfined like the GPU servers
docker run --name fastflowlm_server --network host --init --ulimit memlock=-1 \
    --device /dev/accel --security-opt seccomp=unconfined \
    -v "$MODELS_DIR":/models \
    "$IMAGE" flm serve "$TAG" --host 0.0.0.0 -p 9931
