#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# model selection (multiple choice)
# -------------------------------
# Servable gguf models under /models/llama.cpp. Multi-part sets list their 00001 part;
# llama.cpp auto-loads the siblings. mtp-* auxiliary heads are not servable
# standalone and are intentionally omitted.
FILES=(
      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
      "Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
      "Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf"
      "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
      "Qwen3.8-27B-UD-Q4_K_XL.gguf"
)
LABELS=()
for f in "${FILES[@]}"; do
      if [[ "$f" =~ ^(.+)-[0-9]{5}-of-([0-9]{5})\.gguf$ ]]; then
           LABELS+=("${BASH_REMATCH[1]}     [$((10#${BASH_REMATCH[2]})) parts]")
      else
           LABELS+=("${f%.gguf}")
      fi
done

resolve_model() {    # $1 -> basename on stdout; rc!=0 if no unique match
      local sel="$1" i hits=()
      if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#FILES[@]} )); then
           printf '%s\n' "${FILES[sel-1]}"; return 0
      fi
      for i in "${!FILES[@]}"; do
           [[ "${FILES[$i]}" == *"$sel"* ]] && hits+=("$i")
      done
      if (( ${#hits[@]} == 1 )); then
           printf '%s\n' "${FILES[hits[0]]}"; return 0
      fi
      return 1
}

if [ -n "$1" ]; then
      # positional arg bypasses the menu (index or name)
      MODEL_FILE="$(resolve_model "$1")" || {
           echo "no model matching '$1'; options:" >&2
           for i in "${!FILES[@]}"; do echo "     $((i+1))) ${LABELS[$i]}" >&2; done
           exit 1
      }
elif [ ! -t 0 ]; then
      # non-interactive, no arg: keep the current default
      MODEL_FILE="${FILES[0]}"
else
      echo "Available gguf models in /models/llama.cpp:"
      for i in "${!FILES[@]}"; do echo "     $((i+1))) ${LABELS[$i]}"; done
      read -rp "Select model [1-${#FILES[@]}] (default 1): " pick
      pick=${pick:-1}
      MODEL_FILE="$(resolve_model "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
[ -e "/models/llama.cpp/$MODEL_FILE" ] || { echo "model not found: /models/llama.cpp/$MODEL_FILE" >&2; exit 1; }
echo "serving /models/llama.cpp/$MODEL_FILE"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/llama.cpp:latest"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    (
	cd "$CURDIR/llama.cpp" || exit
	docker build -f .devops/rocm.Dockerfile -t "$IMAGE" --build-arg AMDGPU_TARGETS=gfx1201 .
    )
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# run
# -------------------------------
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f llama_cpp_server
docker run -it --name llama_cpp_server --network host --device=/dev/kfd --device=/dev/dri -v /models/llama.cpp/:/models "$IMAGE" -m "/models/$MODEL_FILE" --host 0.0.0.0 --port 9931 -np 1  --jinja --cache-type-k q8_0 --cache-type-v q8_0 \
 -c 262144 --ctx-checkpoints 64 -b 1024 -ub 512 --reasoning-preserve --reasoning-effort high -ncmoe 36 --flash-attn on \
 --temperature 1.0 --top-p 0.95 --top-k 40

# $ llm-bench run --openai-base-url http://localhost:9931/v1 --models openai:qwen3.8:27b-mq4-xts --prompts "Write a rust function to parse JSON safely."
#→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests
#
#Results: 3 succeeded, 0 failed (100% success rate)
#
#╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
#│ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
#├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
#│ openai   │ qwen3.8:27b-mq4-xts │    38119 │    57313 │  16.4 │      n/a │        n/a │    100% │
#╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯
