# Qwen3.8-Flash-Next-NVFP4 on one RTX PRO 6000 96 GB, engrams on disk

Session 2026-09-01. Hardware: vast.ai instance 49421402, machine 56397 (Texas),
1x RTX PRO 6000 Blackwell Server Edition 96 GB (sm_120), driver 590.48.01,
256 cores, 1 TB host RAM, 8.7 GB/s NVMe. ~7.5 h, ~$13.

Runtime: vLLM `0.28.1rc1.dev164+g39e276eae` wheel + torch 2.13.0+cu130, with
this branch's pure-Python changes overlaid onto site-packages.

## What works

The goal — serving this checkpoint on a single 96 GB GPU with the n-gram
("engram") table on disk and small host RAM — is met.

```
PLE mmap: layer 1 attached (nvfp4), 128 shards, 320001536 rows x 80 B (23.84 GiB on disk), 32 workers
Model loading took 73.07 GiB memory and 86.7 seconds
GPU KV cache size: 454,594 tokens
```

| metric | value |
|---|---|
| engram table, served from NVMe | 23.84 GiB |
| backbone on GPU | 71.65 GiB (73.07 with MTP) |
| checkpoint total | 98.53 GiB — **does not fit in 96 GB VRAM** without this |
| engine host RSS | 3.67–3.92 GiB (PSS 3.47–3.72) |
| all vLLM processes RSS | 5.77 GiB |

Host RSS is far below the 28 GB target. The table is never charged to RSS,
only to evictable page cache.

Correctness of the offload path is established two ways:

1. **Gather oracle** (`oracle.py`): 512 random row ids gathered through
   `MmapPleTable` and compared byte-for-byte against `safetensors.safe_open`.
   **0/64 mismatches on weights, 0/64 on block scales.** Dequant output finite,
   absmean 0.0061, range ±0.035.
2. **End-to-end**: the 136,562-token estonia prompt answers correctly and
   terminates cleanly — `finish=stop`, 2,267 completion tokens, final line
   *"...V-441 corresponds to Mirel Instrument, headquartered in Estonia."*

## What does not work

### estonia scored test — FAILS the 29/30 bar

| config | result |
|---|---|
| Marlin MXFP8, MTP 3, c=6 | **12/30 correct** (`estonia_mtp3.json`) |
| Emulation MXFP8, no MTP, c=6 | engine died mid-run (`estonia_emu.json`) |
| Emulation MXFP8, no MTP, c=2 | engine died mid-run (`estonia_c2.json`) |

Neither failure is in the engram path.

### 1. Marlin MXFP8 is numerically wrong for this model

flashinfer's cutlass `mm_mxfp8` rejects this model's projection shapes on
sm120 (`Problem size is not supported for mm_mxfp8`), so kernel selection falls
through to `MarlinMxfp8LinearKernel`. Under Marlin, 18 of 30 estonia runs
produced **degenerate repetition** — `'ductductductduct...'`, `'名单'` —
looping to the 40,000-token cap. Correct runs finished in ~2,955 tokens
(p50 2955, p90/max 40000).

Swapping to `EmulationMxfp8LinearKernel` (via
`VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel,MarlinMxfp8LinearKernel`)
makes the same prompt answer correctly in 2,267 tokens. **Marlin MXFP8 is
producing wrong results here and should be treated as unusable for this model.**

### 2. CUDA illegal memory access at long context — UNRESOLVED

`CUDA error: an illegal memory access was encountered`, async, no useful
traceback. Reproduced:

- with MTP on **and** with MTP off → not MTP
- with Marlin **and** with Emulation → not the MXFP8 kernel
- at concurrency 6 **and** at concurrency 2 → not a specific concurrency level
- **not** on a single sequential 136k request, which completes cleanly

So: long context plus more than one in-flight request. The engram gather is
CPU/numpy and byte-exact, so it is not the source. Prime remaining suspect is
upstream's Qwen4Exp QSA sparse-attention/indexer path at long context.

Next step would be the standard IMA hunt: `--enforce-eager` plus
`CUDA_LAUNCH_BLOCKING=1` to name the kernel (CLB alone is masked inside
cudagraph replay), then instrument the suspect Triton kernel in place.

### 3. MTP 3 acceptance is exactly 0

`Mean acceptance length: 1.00, Drafted throughput: ~380 tokens/s, Accepted: 0
tokens`, sustained for the whole run. MTP drafts and nothing is ever accepted —
pure overhead, no speedup. Correctness is unaffected (rejected drafts are
discarded). Likely the same defect fixed downstream by
`local-inference-lab/vllm` commit `4c1f7b2c3` *"fix(mtp): preserve
position-zero embeddings (#539)"*.

### 4. fp8 KV cache is unavailable

`vllm/models/qwen4_exp/nvidia/qsa.py:114` raises
`Qwen4Exp QSA requires a BF16 main KV cache` for any `kv_cache_dtype` other
than `auto`/`bfloat16`. Blanket refusal, not a knob — the sparse-attention
kernel has no fp8 path. The downstream fork has this
(`701985284 "Support FP8 KV cache for Qwen3.8 Flash Next"`) but via its b12x
QSA implementation, which upstream does not have.

## Performance notes

- Decode at c=6 with Marlin: ~185 tok/s aggregate, ~31 tok/s/stream, p50
  per-stream 43.9 tok/s.
- Per-stream decode was *faster* at batch 6 than at batch 2 (2.6 tok/s/stream),
  which tracks QSA indexer cost growing with context rather than the PLE
  gather.
- At small batch, consider `VLLM_PLE_MMAP_SERIAL`: a batch-1 gather scatters
  across all 128 shards by construction and degenerates to ~one pool task per
  row.

## Files

| file | what |
|---|---|
| `oracle.py` | gather-vs-safetensors byte-exactness check |
| `direct.py` | single estonia request straight at the server |
| `rep.py` | summarise a bench result json |
| `serve.sh` / `estonia.sh` | exact serve and bench invocations used |
| `estonia_*.json` | raw bench results per config |
