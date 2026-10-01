#!/bin/bash
export IMAGE="radiance-vllm-mxfp4:0.13.0"
source base_run_radiance.sh
# systemctl stop gdm3
# IMAGE="radiance-vllm-mxfp4:0.13.0" CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=245760 MAXSEQS=1 CHUNK=4096 KV_MEM=9977856000 SINGLE_GPU_PROFILE=1 RADIANCE_NORMQUANT_FUSION=0 RADIANCE_FP8_STREAM=0 RADIANCE_GDN_MERGE_INPROJ=0 FAST_DRAFT=0 RADIANCE_VERIFY_HEAD=0 DRAFT_SAMPLE=probabilistic ./serve-mxfp4.sh
