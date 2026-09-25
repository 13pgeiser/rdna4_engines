#!/bin/bash
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f llama_cpp_server
IMAGE="local/llama.cpp:latest"
docker run -d --name llama_cpp_server --network host --device=/dev/kfd --device=/dev/dri -v /models/:/models "$IMAGE" -m "/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" --host 0.0.0.0 --port 9931 -np 1  --jinja --cache-type-k q8_0 --cache-type-v q8_0 \
 -c 262144 --ctx-checkpoints 64 -b 1024 -ub 512 --reasoning-preserve --reasoning-effort high -ncmoe 36 --flash-attn on \
 --temperature 1.0 --top-p 0.95 --top-k 40
