#!/bin/bash
if [ ! -d libr4d ]; then
    git_cmd="git subtree add"
else
    git_cmd="git subtree pull"
fi
$git_cmd --prefix=libr4d https://codeberg.org/StillDeadcode/libr4d main
$git_cmd --prefix=radiance-vllm-mxfp4 https://codeberg.org/ggz14/radiance-vllm-mxfp4 main
$git_cmd --prefix=llama.cpp https://github.com/ggml-org/llama.cpp.git master
