#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURDIR/common.sh" || exit 1

# -------------------------------
# GPU arch gate: gfx1201 or gfx1152 only (early exit; detection in common.sh)
# -------------------------------
gpu_arch_gate run_hipfire.sh gfx1201 gfx1152

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# parallel arrays, one entry per model, indexed together:
#   TAGS  - registry tag (`hipfire serve`/`pull` take this)
#   FILES - main model file under /models/hipfire (present = pulled)
#   EXTRAS- extra serve args for this model
TAGS=(
      "qwen3.8:27b-mq4-xts"
      "qwen3.8:flash-next-mq4-xts"
      "qwen3.6:35b-a3b"
      "ornith-1.5:35b-a3b"
)
FILES=(
      "qwen3.8-27b.mq4-xts"
      "qwen3.8-flash-next.mq4-xts"
      "qwen3.6-35b-a3b.mq4p"
      "ornith-1.5-35b-a3b.mq4"
)
EXTRAS=(
      ""                   # dense 27B: omitted max_seq auto-sizes to card capacity (free VRAM after
                           # weights / K+V stride), so it always fits - no flag needed
      "--max-seq 262144"   # MoE models on a discrete GPU (gfx1201) get NO auto-sizing: it is not
      "--max-seq 262144"   # validated against expert placement, so an omitted max_seq falls back
      "--max-seq 262144"   # to a fixed 32768 default and can OOM - pin 262144 explicitly
)

select_model "Available models in the registry:" "$1" "${TAGS[@]}"
TAG="${TAGS[$MODEL_IDX]}"
FILE="${FILES[$MODEL_IDX]}"
EXTRA_FLAGS="${EXTRAS[$MODEL_IDX]}"
echo "serving $TAG with EXTRA_FLAGS=$EXTRA_FLAGS"

# -------------------------------
# docker image (build only if missing or wrong variant)
# -------------------------------
IMAGE="local/hipfire"
MODELS_DIR=/models/hipfire
# the image's runtime stage sets HOME=/home/hipfire and HIPFIRE_HOME=/home/hipfire/.hipfire
HIPFIRE_HOME_DIR=/home/hipfire/.hipfire
# gfx1152 has no baked kernel pack (baked arches: gfx1100 gfx1151 gfx1201) and the slim
# runtime image has no hipcc, so a missing kernel is a hard error there: the full target
# carries hipcc and JITs each such route once into the hipfire-kcache volume.
if [ "$GPU_ARCH" = gfx1152 ]; then TARGET=full; else TARGET=runtime; fi
HAVE_VARIANT="$(docker image inspect --format '{{index .Config.Labels "dev.hipfire.variant"}}' "$IMAGE" 2>/dev/null)"
if [ "$HAVE_VARIANT" = "$TARGET" ]; then
    echo "image $IMAGE already present (variant $TARGET), skipping build"
else
    (
	cd "$CURDIR/hipfire" || exit
	docker build --progress=plain -f Containerfile --target "$TARGET" -t "$IMAGE" .
    )
fi

# -------------------------------
# model (pulled into host /models)
# -------------------------------
if [ ! -f "$MODELS_DIR/$FILE" ]; then
    docker run --rm --device /dev/kfd --device /dev/dri -v "$MODELS_DIR":"$HIPFIRE_HOME_DIR/models" "$IMAGE" pull "$TAG"
fi

# -------------------------------
# persistent config (default reasoning effort, DFlash off on gfx1152)
# -------------------------------
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
HOME_VOLUME=hipfire-home
echo "setting reasoning_effort=$REASONING_EFFORT in $HOME_VOLUME"
docker run --rm --device /dev/kfd --device /dev/dri -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" "$IMAGE" config set reasoning_effort "$REASONING_EFFORT"
# the registry auto-enables the 27B's DFlash draft, but its batched lm_head verify
# (MQ4G256V2) is not admitted on gfx1152: pre-warm fails and every request 400s,
# so force AR decoding for the selected model there
if [ "$GPU_ARCH" = gfx1152 ]; then
    echo "disabling DFlash for $TAG on gfx1152"
    docker run --rm --device /dev/kfd --device /dev/dri -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" "$IMAGE" config "$TAG" set dflash_mode off
fi

# -------------------------------
# run
# -------------------------------
# power cap is only for the 210 W gfx1201 card
if [ "$GPU_ARCH" = gfx1201 ]; then
    sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
fi
docker rm -f hipfire_server
docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$HOME_VOLUME":"$HIPFIRE_HOME_DIR" -v "$MODELS_DIR":"$HIPFIRE_HOME_DIR/models" -v hipfire-kcache:/var/cache/hipfire -e HIPFIRE_NGRAM_STORAGE=system -e HIPFIRE_VISION_MODE=off "$IMAGE" serve "$TAG" 0.0.0.0:9931 $EXTRA_FLAGS
# docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /models/hipfire:/home/hipfire/.hipfire/models -v hipfire-kcache:/var/cache/hipfire "$IMAGE" run qwen3.8:27b-mq4-xts "Explain FFT in one line"

# qwen3.8:27b-mq4-xts
#
# llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9931/v1 --models openai:qwen3.8:27b-mq4-xts --prompts "Write a rust function to parse JSON safely."
# → Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests
# 
# Results: 3 succeeded, 0 failed (100% success rate)
# 
# ╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
# │ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
# ├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
# │ openai   │ qwen3.8:27b-mq4-xts │     6189 │     6308 │  83.4 │      n/a │        n/a │    100% │
# ╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯
