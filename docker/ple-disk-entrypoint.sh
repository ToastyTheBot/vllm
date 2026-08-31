#!/usr/bin/env bash
# Serve a Qwen3.8-Flash-Next checkpoint with the n-gram table on disk.
#
#   ple-disk-entrypoint.sh /model [extra vllm serve args...]
#
# Anything after the model path is passed through to `vllm serve`, so the
# defaults below can be overridden per deployment.
set -euo pipefail

MODEL_PATH="${1:?usage: ple-disk-entrypoint.sh <model-path> [vllm serve args...]}"
shift || true

# Checkpoints exported under the downstream naming (model_type
# "qwen3_8_flash_next", architecture "Qwen3_8FlashNextForConditionalGeneration")
# do not match the names upstream registers, which are "qwen4_exp" /
# "Qwen4ExpForConditionalGeneration" — the same names the official
# Qwen/Qwen3.8-Flash-Next checkpoint uses. Remap in place; the weights are
# identical, only the config labels differ. The original is kept alongside.
python3 - "$MODEL_PATH" <<'PY'
import json, os, shutil, sys

cfg_path = os.path.join(sys.argv[1], "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)

if cfg.get("model_type") == "qwen3_8_flash_next":
    backup = cfg_path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(cfg_path, backup)
    cfg["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    cfg["model_type"] = "qwen4_exp"
    if "text_config" in cfg:
        cfg["text_config"]["model_type"] = "qwen4_exp_text"
    if "vision_config" in cfg:
        cfg["vision_config"]["model_type"] = "qwen4_exp"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"remapped {cfg_path} to upstream qwen4_exp naming", file=sys.stderr)
PY

# PIECEWISE is mandatory: the PLE mmap forward is a blocking host round-trip
# (ids D2H, page-cache gather, rows H2D) and cannot be captured. ple_mmap
# refuses FULL/FULL_AND_PIECEWISE at startup rather than failing later.
#
# The KV cache dtype is deliberately left at the bf16 default: upstream's
# Qwen4Exp QSA kernel raises "Qwen4Exp QSA requires a BF16 main KV cache" for
# any other value (vllm/models/qwen4_exp/nvidia/qsa.py).
exec vllm serve "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "${MAX_MODEL_LEN:-180224}" \
    --max-num-seqs "${MAX_NUM_SEQS:-16}" \
    -cc.cudagraph_mode=PIECEWISE \
    --port "${PORT:-8000}" \
    "$@"
