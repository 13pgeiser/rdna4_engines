#!/bin/bash
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f r9700-qwen3.8-mxfp4
HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd radiance-vllm-mxfp4
sed -i '/override-generation-config/c\    --override-generation-config '\''{"temperature":1.0,"top_p":0.95,"top_k":20}'\'' \\' serve-mxfp4.sh # temperature":1.0,"top_p":0.95,"top_k":20 are recommended settings.

# -------------------------------
#CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with:
# (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.50 GiB. GPU 0 has a total capacity of 31.86 GiB of which 6.87 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 FAST_DRAFT=0 ./serve-mxfp4.sh
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
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=9126805504 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.50 GiB. GPU 0 has a total capacity of 31.86 GiB of which 7.10 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=220000 MAXSEQS=1 CHUNK=4096 KV_MEM=8300000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.72 GiB. GPU 0 has a total capacity of 31.86 GiB of which 7.10 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=200000 MAXSEQS=1 CHUNK=4096 KV_MEM=7900000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
# Fails with: (EngineCore pid=572) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.35 GiB. GPU 0 has a total capacity of 31.86 GiB of which 6.84 GiB is free. Of the allocated memory 20.60 GiB is allocated by PyTorch, and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
# -------------------------------

# -------------------------------
# CHAT_TEMPLATE="../chat_template.jinja" MODELS=/models SNAP=/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken SERVED_NAMES="qwen3.8" NAME="r9700-qwen3.8-mxfp4" PORT=9931 TP=1 MAXLEN=180000 MAXSEQS=1 CHUNK=4096 KV_MEM=6900000000 SINGLE_GPU_PROFILE=1 SPEC_METHOD=dflash ./serve-mxfp4.sh
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
