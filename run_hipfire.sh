#!/bin/bash
TAG="${1:-qwen3.8:27b-mq4-xts}"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /models/hipfire:/root/.hipfire/models -v hipfire-kcache:/var/cache/hipfire "$IMAGE" serve "$TAG" 0.0.0.0:9931
# docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /models/hipfire:/root/.hipfire/models -v hipfire-kcache:/var/cache/hipfire "$IMAGE" run qwen3.8:27b-mq4-xts "Explain FFT in one line"

============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  383.00    
Total input tokens:                      107626    
Total generated tokens:                  12792     
Request throughput (req/s):              0.26      
Output token throughput (tok/s):         33.40     
Peak output token throughput (tok/s):    66.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          314.41    
---------------Time to First Token----------------
Mean TTFT (ms):                          725.60    
Median TTFT (ms):                        664.28    
P99 TTFT (ms):                           769.68    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.46     
Median TPOT (ms):                        24.95     
P99 TPOT (ms):                           29.40     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.11     
Median ITL (ms):                         0.01      
P99 ITL (ms):                            61.16     
==================================================

