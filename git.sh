#!/bin/bash
subtree() {
    if [ ! -d "$1" ]; then
        git subtree add --prefix="$1" "$2" "$3"
    else
        git subtree pull --prefix="$1" "$2" "$3"
    fi
}
subtree libr4d https://codeberg.org/StillDeadcode/libr4d main
subtree radiance-vllm-mxfp4 https://codeberg.org/ggz14/radiance-vllm-mxfp4 main
subtree llama.cpp https://github.com/ggml-org/llama.cpp.git master
subtree hipfire https://github.com/warpfront/hipfire.git master
subtree radiance https://codeberg.org/StillDeadcode/radiance.git main
