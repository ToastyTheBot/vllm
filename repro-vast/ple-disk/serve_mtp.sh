#!/bin/bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export TORCH_CUDA_ARCH_LIST=12.0a
export VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel
export VLLM_PLE_PROBE=1
export VLLM_PLE_MMAP=1
export VLLM_PLE_MMAP_WORKERS=32
export VLLM_PLE_MMAP_CHUNK=2048
export VLLM_LOGGING_LEVEL=INFO
exec /workspace/vllmenv/bin/vllm serve /workspace/model \
  --served-model-name q38fn \
  --tensor-parallel-size 1 \
  --max-model-len 180224 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  -cc.cudagraph_mode=PIECEWISE \
  --port 8000
