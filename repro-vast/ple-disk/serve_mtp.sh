#!/bin/bash
# Final verified config: estonia 30/30, lavd 29/30, MTP acceptance 3.00,
# 0 IMAs. Prefix caching ON, engrams served from NVMe, single RTX PRO 6000 96GB.
# NOTE: --kv-cache-dtype is deliberately left at the bf16 default; qsa.py:114
# refuses any other value for this model.
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export TORCH_CUDA_ARCH_LIST=12.0a
export VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel
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
