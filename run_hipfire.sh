#!/bin/bash
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# model selection (multiple choice)
# -------------------------------
TAGS=(
      "qwen3.8:27b-mq4-xts"
      "qwen3.8:flash-next"
)

resolve_tag() {    # $1 -> tag on stdout; rc!=0 if no unique match
      local sel="$1" i hits=()
      if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#TAGS[@]} )); then
           printf '%s\n' "${TAGS[sel-1]}"; return 0
      fi
      for i in "${!TAGS[@]}"; do
           [[ "${TAGS[$i]}" == *"$sel"* ]] && hits+=("$i")
      done
      if (( ${#hits[@]} == 1 )); then
           printf '%s\n' "${TAGS[hits[0]]}"; return 0
      fi
      return 1
}

if [ -n "$1" ]; then
      # positional arg bypasses the menu (index or name)
      TAG="$(resolve_tag "$1")" || {
           echo "no model matching '$1'; options:" >&2
           for i in "${!TAGS[@]}"; do echo "     $((i+1))) ${TAGS[$i]}" >&2; done
           exit 1
      }
elif [ ! -t 0 ]; then
      # non-interactive, no arg: keep the current default
      TAG="${TAGS[0]}"
else
      echo "Available models in the registry:"
      for i in "${!TAGS[@]}"; do echo "     $((i+1))) ${TAGS[$i]}"; done
      read -rp "Select model [1-${#TAGS[@]}] (default 1): " pick
      pick=${pick:-1}
      TAG="$(resolve_tag "$pick")" || { echo "invalid selection: $pick" >&2; exit 1; }
fi
echo "serving $TAG"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/hipfire"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    (
	cd "$CURDIR/hipfire" || exit
	docker build -f Containerfile --target runtime -t "$IMAGE" .
    )
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# model (pulled into host /models)
# -------------------------------
if ! docker run --rm -v /models/hipfire:/root/.hipfire/models "$IMAGE" list 2>/dev/null | grep -q "$TAG"; then
    docker run --rm -v /models/hipfire:/root/.hipfire/models "$IMAGE" pull "$TAG"
fi

# -------------------------------
# run
# -------------------------------
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f hipfire_server
docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /models/hipfire:/root/.hipfire/models -v hipfire-kcache:/var/cache/hipfire -e HIPFIRE_VISION_MODE=off "$IMAGE" serve "$TAG" 0.0.0.0:9931 --spec auto
# docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /models/hipfire:/root/.hipfire/models -v hipfire-kcache:/var/cache/hipfire "$IMAGE" run qwen3.8:27b-mq4-xts "Explain FFT in one line"


#phoenix@debian00d861dabdae:~/ai/my_ai_scripts$ llm-bench run --openai-base-url http://localhost:9931/v1 --models openai:qwen3.8:27b-mq4-xts --prompts "Write a rust function to parse JSON safely."
#→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests
#
#Results: 3 succeeded, 0 failed (100% success rate)
#
#╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
#│ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
#├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
#│ openai   │ qwen3.8:27b-mq4-xts │    12246 │    17813 │  49.0 │      n/a │        n/a │    100% │
#╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

# ============ Serving Benchmark Result ============
# Successful requests:                     100       
# Failed requests:                         0         
# Maximum request concurrency:             1         
# Benchmark duration (s):                  383.00    
# Total input tokens:                      107626    
# Total generated tokens:                  12792     
# Request throughput (req/s):              0.26      
# Output token throughput (tok/s):         33.40     
# Peak output token throughput (tok/s):    66.00     
# Peak concurrent requests:                2.00      
# Total token throughput (tok/s):          314.41    
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          725.60    
# Median TTFT (ms):                        664.28    
# P99 TTFT (ms):                           769.68    
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          24.46     
# Median TPOT (ms):                        24.95     
# P99 TPOT (ms):                           29.40     
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           24.11     
# Median ITL (ms):                         0.01      
# P99 ITL (ms):                            61.16     
# ==================================================

