#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# GPU arch gate: gfx1201 or gfx1152 only (early exit)
# Card identity by PCI ID from /sys; the amdgpu dmesg GFX IP block version
# ("amdgpu <bdf>: ... <gfx_vX_Y_Z> ...") is a family fallback for unknown IDs.
# -------------------------------
DMSGS="$( { dmesg 2>/dev/null || sudo dmesg 2>/dev/null; } 2>/dev/null)"
SUPPORTED_ARCHES=""
AMD_IDS=""
for dev in /sys/class/drm/card*/device; do
    [ "$(cat "$dev/vendor" 2>/dev/null)" = "0x1002" ] || continue
    id="$(cat "$dev/device" 2>/dev/null)"
    case " $AMD_IDS " in *" $id "*) ;; *) AMD_IDS="$AMD_IDS $id" ;; esac
    bdf="$(basename "$(readlink -f "$dev")")"
    ip="$(printf '%s\n' "$DMSGS" | awk -v b="amdgpu $bdf:" 'index($0, b) && match($0, /gfx_v[0-9]+_[0-9]+_[0-9]+/) { print substr($0, RSTART, RLENGTH); exit }')"
    case "$id" in
        0x7551) arch=gfx1201 ;;        # Radeon AI PRO R9700
        0x1114) arch=gfx1152 ;;        # Ryzen AI 7 350 (Krackan Point)
        *)
            case "$ip" in
                gfx_v12_0_*) arch=gfx1201 ;;   # RDNA4 family
                gfx_v11_5_*) arch=gfx1152 ;;   # RDNA3.5 family (e.g. RX 9060)
                *) continue ;;
            esac ;;
    esac
    case " $SUPPORTED_ARCHES " in *" $arch "*) ;; *) SUPPORTED_ARCHES="$SUPPORTED_ARCHES $arch" ;; esac
done
SUPPORTED_ARCHES="${SUPPORTED_ARCHES# }"
AMD_IDS="${AMD_IDS# }"
if [ -z "$SUPPORTED_ARCHES" ]; then
    echo "no supported GPU: run_hipfire.sh needs gfx1201 or gfx1152 (AMD cards found: ${AMD_IDS:-none})" >&2
    exit 1
fi
if [ "$(echo $SUPPORTED_ARCHES | wc -w)" -gt 1 ]; then
    echo "ambiguous: multiple supported GPUs found ($SUPPORTED_ARCHES); narrow to one first" >&2
    exit 1
fi
GPU_ARCH="$SUPPORTED_ARCHES"
echo "GPU arch: $GPU_ARCH"

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

resolve_tag() {    # $1 -> model index on stdout; rc!=0 if no unique match
      local sel="$1" i hits=()
      if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#TAGS[@]} )); then
           printf '%s\n' $((sel-1)); return 0
      fi
      for i in "${!TAGS[@]}"; do
           [[ "${TAGS[$i]}" == *"$sel"* ]] && hits+=("$i")
      done
      if (( ${#hits[@]} == 1 )); then
           printf '%s\n' "${hits[0]}"; return 0
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
      for i in "${!TAGS[@]}"; do echo "     $((i+1))) ${TAGS[$i]}"; done
      read -rp "Select model [1-${#TAGS[@]}] (default 1): " pick
      pick=${pick:-1}
      IDX="$(resolve_tag "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
TAG="${TAGS[$IDX]}"
FILE="${FILES[$IDX]}"
EXTRA_FLAGS="${EXTRAS[$IDX]}"
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
	docker build -f Containerfile --target "$TARGET" -t "$IMAGE" .
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
