#!/bin/bash
IMAGE="radiance-vllm-mxfp4:0.13.0"
docker run -it --network host --entrypoint /opt/vllm/bin/vllm -v /models:/models "$IMAGE" bench serve \
    --backend openai-chat \
    --model models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken \
    --served-model-name qwen3.8 \
    --base-url "http://127.0.0.1:9931" \
    --endpoint "/v1/chat/completions" \
    --dataset-name random \
    --num-prompts 100 \
    --max-concurrency 1 \
    --request-rate inf


