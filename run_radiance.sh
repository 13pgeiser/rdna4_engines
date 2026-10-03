#!/bin/bash
ROCM="rocm10"
BENCH=0
for a in "$@"; do
    case "$a" in
        rocm7|rocm10) ROCM="$a" ;;
        --bench) BENCH=1 ;;
        *) echo "usage: $0 [rocm7|rocm10] [--bench]" >&2; exit 1 ;;
    esac
done
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# model preparation
# -------------------------------
if [ -d /models ]; then
	if [ ! -d /models/Qwen3.8-27B-Quark-AWQ-MXFP4 ]; then
	    (
		cd /models || exit
		git clone https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-DFlash2-FP8 ]; then
	    (
		cd /models || exit
		git clone https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-MXFP4-mtpfp8 ]; then
	    mtp="$CURDIR/radiance/fp8_mtp.py"
	    (
		cd /models || exit
		python3 -m venv venv
		./venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
		./venv/bin/python3 "$mtp" "Qwen3.8-27B-Quark-AWQ-MXFP4" "Qwen3.8-27B-MXFP4-mtpfp8"
		rm -rf venv
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken ]; then
		(
		cd /models || exit
		mkdir -p Qwen3.8-27B-MXFP4-mtpfp8-pertoken
		for f in model.safetensors tokenizer.json tokenizer_config.json merges.txt \
		    generation_config.json chat_template.jinja preprocessor_config.json processor_config.json; do
		    [ -e "/models/Qwen3.8-27B-MXFP4-mtpfp8/$f" ] && ln -sf "/models/Qwen3.8-27B-MXFP4-mtpfp8/$f" "/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken/$f"
		done
		python3 "$CURDIR/radiance/mtpfp8-pertoken.py" "Qwen3.8-27B-MXFP4-mtpfp8/config.json" "Qwen3.8-27B-MXFP4-mtpfp8-pertoken/config.json"
		)
	fi
else
	echo "/models does not exists..."
fi
if [ ! -e "$CURDIR/radiance/chat_template.jinja" ]; then
	curl -fsSL https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja -o "$CURDIR/radiance/chat_template.jinja"
fi

# -------------------------------
# docker image (build only if missing)
# -------------------------------
VERSION="$(cat "$CURDIR/radiance-vllm-mxfp4/VERSION")"
case "$ROCM" in
    rocm7)  IMAGE="radiance-vllm-mxfp4:$VERSION" ;;
    rocm10) IMAGE="rocm10-radiance-vllm-mxfp4:$VERSION" ;;
esac
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    (
	cd "$CURDIR/radiance-vllm-mxfp4" || exit
	if [ "$ROCM" = "rocm10" ]; then
	    cat Dockerfile \
		| sed 's|^ARG ROCM_BASE=.*$|ARG ROCM_BASE=docker.io/rocm/dev-ubuntu-24.04:10.0.0-full|' \
		| sed 's|^ARG RELEASE_BASE=.*$|ARG RELEASE_BASE=docker.io/library/ubuntu:24.04|' \
		>Dockerfile.rocm10
	    docker build -t "$IMAGE" -f Dockerfile.rocm10 .
	    rm -f Dockerfile.rocm10
	else
	    docker build -t "$IMAGE" -f Dockerfile .
	fi
    )
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# run
# -------------------------------
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f r9700-qwen3.8-mxfp4
HOME="$CURDIR"
export IMAGE
cd "$CURDIR/radiance-vllm-mxfp4"
sed -i '/override-generation-config/c\    --override-generation-config '\''{"temperature":1.0,"top_p":0.95,"top_k":20}'\'' \\' serve-mxfp4.sh # temperature":1.0,"top_p":0.95,"top_k":20 are recommended settings.
# the sed tweak is only for this serve run: revert serve-mxfp4.sh whenever this script exits
trap 'git -C "$CURDIR/radiance-vllm-mxfp4" checkout -- serve-mxfp4.sh' EXIT

# -------------------------------
if [ "$BENCH" = 1 ]; then
	# detached serve: the container is RUNNING when this returns, the engine is not ready yet
	DETACH=1 CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 FAST_DRAFT=0 ./serve-mxfp4.sh
	deadline=$((SECONDS + 900))
	until curl -fsS "http://127.0.0.1:9931/health" >/dev/null 2>&1; do
	    if [ "$SECONDS" -ge "$deadline" ]; then
		docker rm -f r9700-qwen3.8-mxfp4
		echo "server did not become healthy within 900 s" >&2
		exit 1
	    fi
	    sleep 5
	done
	# vLLM benchmark client (formerly bench.sh)
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
	docker rm -f r9700-qwen3.8-mxfp4
else
	CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 FAST_DRAFT=0 ./serve-mxfp4.sh
fi
# ============ Serving Benchmark Result ============
# Successful requests:                     100       
# Failed requests:                         0         
# Maximum request concurrency:             1         
# Benchmark duration (s):                  226.79    
# Total input tokens:                      103438    
# Total generated tokens:                  12800     
# Request throughput (req/s):              0.44      
# Output token throughput (tok/s):         56.44     
# Peak output token throughput (tok/s):    25.00     
# Peak concurrent requests:                2.00      
# Total token throughput (tok/s):          512.53    
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          449.08    
# Median TTFT (ms):                        451.17    
# P99 TTFT (ms):                           460.25    
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          14.32     
# Median TPOT (ms):                        14.14     
# P99 TPOT (ms):                           17.88     
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           41.80     
# Median ITL (ms):                         42.62     
# P99 ITL (ms):                            47.17     
# ---------------Speculative Decoding---------------
# Acceptance rate (%):                     29.00     
# Acceptance length:                       3.03      
# Drafts:                                  4251      
# Draft tokens:                            29757     
# Accepted tokens:                         8629      
# Per-position acceptance (%):
#   Position 0:                            78.52     
#   Position 1:                            50.51     
#   Position 2:                            33.03     
#   Position 3:                            19.31     
#   Position 4:                            11.95     
#   Position 5:                            6.35      
#   Position 6:                            3.32      
# ==================================================
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=180000 MAXSEQS=1 CHUNK=4096 KV_MEM=6900000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# ============ Serving Benchmark Result ============
#Successful requests:                     100       
#Failed requests:                         0         
#Maximum request concurrency:             1         
#Benchmark duration (s):                  349.94    
#Total input tokens:                      103438    
#Total generated tokens:                  12800     
#Request throughput (req/s):              0.29      
#Output token throughput (tok/s):         36.58     
#Peak output token throughput (tok/s):    16.00     
#Peak concurrent requests:                2.00      
#Total token throughput (tok/s):          332.16    
#---------------Time to First Token----------------
#Mean TTFT (ms):                          464.06    
#Median TTFT (ms):                        457.87    
#P99 TTFT (ms):                           472.27    
#-----Time per Output Token (excl. 1st token)------
#Mean TPOT (ms):                          23.90     
#Median TPOT (ms):                        24.07     
#P99 TPOT (ms):                           30.30     
#---------------Inter-token Latency----------------
#Mean ITL (ms):                           69.37     
#Median ITL (ms):                         71.66     
#P99 ITL (ms):                            72.52     
#---------------Speculative Decoding---------------
#Acceptance rate (%):                     28.76     
#Acceptance length:                       3.01      
#Drafts:                                  4275      
#Draft tokens:                            29925     
#Accepted tokens:                         8606      
#Per-position acceptance (%):
#  Position 0:                            78.53     
#  Position 1:                            50.08     
#  Position 2:                            32.12     
#  Position 3:                            18.48     
#  Position 4:                            11.70     
#  Position 5:                            6.62      
#  Position 6:                            3.79     
#==================================================
# -------------------------------

# -------------------------------
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.50 GiB. GPU 0 has a total capacity of 31.86 GiB of which 7.10 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=8300000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.72 GiB. GPU 0 has a total capacity of 31.86 GiB of which 7.10 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=200000 MAXSEQS=1 CHUNK=4096 KV_MEM=7900000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.35 GiB. GPU 0 has a total capacity of 31.86 GiB of which 6.84 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../radiance/chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with:
# (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.50 GiB. GPU 0 has a total capacity of 31.86 GiB of which 6.87 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------
