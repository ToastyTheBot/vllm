# Serving the Qwen3.8-Flash-Next PLE n-gram table from disk

Qwen3.8-Flash-Next carries a very large learned n-gram ("engram") embedding
table. It is a pure gather: each decoded token touches only a few KB of rows,
so almost all of it is cold at any instant. `VLLM_PLE_MMAP=1` swaps the
GPU-resident table for `np.memmap` views over the checkpoint's own safetensors
shards, and lets the kernel page cache serve the working set.

This makes single-GPU serving possible on hosts with far less RAM than the
table, and — because nothing is converted or rewritten at load — it costs no
extra disk and no first-boot conversion pass.

## Quick start

```bash
VLLM_PLE_MMAP=1 vllm serve <checkpoint> \
    --tensor-parallel-size 1 \
    -cc.cudagraph_mode=PIECEWISE
```

`-cc.cudagraph_mode=PIECEWISE` is mandatory. The gather is a blocking host
round-trip (ids D2H, page-cache gather, rows H2D) and cannot run inside a
capture, so startup refuses `FULL` and `FULL_AND_PIECEWISE` rather than failing
later.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_PLE_MMAP` | `0` | Serve the table from disk. Unset means byte-identical stock behavior. |
| `VLLM_PLE_MMAP_WORKERS` | `32` | Gather thread-pool size. |
| `VLLM_PLE_MMAP_CHUNK` | `2048` | Rows per gather task. |
| `VLLM_PLE_MMAP_PREWARM` | `0` | Stream the table once at load to warm the page cache, bounded by available memory. |
| `VLLM_PLE_MMAP_READAHEAD` | `0` | Max coalesced ranges handed to `posix_fadvise(WILLNEED)` before copying. `0` disables the pre-pass. |
| `VLLM_PLE_MMAP_PINNED` | `0` | Stage gathered rows through a per-call pinned host buffer before the H2D copy. |
| `VLLM_PLE_MMAP_SERIAL` | `0` | Run gathers touching at most N distinct rows inline instead of through the pool. |

Only `VLLM_PLE_MMAP` is a compile factor; the rest touch only the host body of
an already-split-out op.

### Tuning at small batch

At batch 1 a gather's rows land scattered across shards by construction — that
is what hashing into a wide n-gram table does — so a small gather degenerates
into roughly one task per row and pays full pool dispatch for each. If decode
throughput at low concurrency looks far worse than at high concurrency, set
`VLLM_PLE_MMAP_SERIAL` to a few hundred so small gathers run inline on the
calling thread.

## Supported table formats

| Checkpoint dtype | Layout | Scales |
|---|---|---|
| `BF16` | one tensor per shard, 2 B/element | none |
| `F8_E4M3` | one tensor per shard, 1 B/element | one global scalar per layer |
| **NVFP4** | `U8`, **2 values per byte** | per-row group-16 block scales sharded alongside the weights, plus one fp32 `weight_scale_2` |

`F8_E5M2` is deliberately refused: `is_fp8()` does not recognize it, so the
dequant gate would silently never fire.

For NVFP4 the block scales share the weights' row layout exactly, so they are
served by a second mmap table keyed on the same row ids and reusing the same
coalescing, readahead and thread-pool machinery. Both gathers land on the
device before `dequantize_to_dtype` unpacks the nibbles and applies the scales,
so no 4-bit arithmetic happens on the host. `weight_scale_2` is consumed
un-reciprocated, matching `ModelOptNvFp4LinearMethod`.

## Measured: Qwen3.8-Flash-Next-NVFP4 on one RTX PRO 6000 96 GB

Checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`, 98.53 GiB total:

| component | size | placement |
|---|---|---|
| n-gram table (128 shards, `(2500012, 80)` U8 + block scales) | 23.84 GiB | disk (page cache) |
| backbone, MoE, embeddings, lm_head | 71.65 GiB | GPU |

With `--gpu-memory-utilization 0.95 --max-model-len 180224` and MTP 3:

```
PLE mmap: layer 1 attached (nvfp4), 128 shards, 320001536 rows x 80 B (23.84 GiB on disk), 32 workers
Model loading took 73.07 GiB memory and 83.257126 seconds
GPU KV cache size: 454,594 tokens, Maximum concurrency for 180,224 tokens per request: 2.52x
```

(Verbatim from one run; see `repro-vast/ple-disk/final_metrics.txt`. Load time
varies run to run -- 70.8 s to 86.7 s observed -- with page-cache warmth.)

Host memory splits into two very different parts, and they should be read
separately:

| | |
|---|---|
| anonymous (non-evictable) | **~3.1 GiB** |
| file-backed, clean, evictable (the mmap'd table) | grows toward the table size |

Only the anonymous figure is a hard requirement. The file-backed portion is
mmap-backed page cache the kernel reclaims under pressure, so it expands to
whatever the host can spare — on a 1 TB host it grows to hold nearly the whole
23.84 GiB table, and on a small host it simply holds less and serves more
gathers from NVMe. Measure `Pss_Anon` in `/proc/<pid>/smaps_rollup`, not `Rss`,
when sizing a deployment.

The stock path cannot serve this checkpoint on one 96 GB GPU at all: 98.53 GiB
of weights does not fit in 96 GB of VRAM.

## Deployment notes

- **Cap the serving container's memory** when host RAM is comfortably larger
  than the table. Otherwise the load's own multi-GiB checkpoint streaming
  passes through the global page cache and evicts the table it is about to
  serve. A container memory cap makes reclaim cgroup-local.
- **Prefix caching works, but needs the hybrid-mamba seeding fix.** Before that
  fix, enabling prefix caching on this model produced degenerate repetition and
  then a CUDA illegal memory access (upstream #54173), because
  `MambaHybridModelState.add_request` seeded the mamba state column with
  `cache_config.block_size` — which `EngineCore` rewrites to the *smallest* KV
  group's block size on a hybrid model (4 here, versus a mamba block size of
  1568). With the fix in this branch, the 30-run estonia long-context suite
  scores 30/30 with prefix caching enabled.
- **`--kv-cache-dtype fp8` is supported** (E4M3 only). The QSA Triton kernel
  dequantizes fp8 pages to the query dtype on load with the same per-tensor
  `k_scale`/`v_scale` used to write them. On one RTX PRO 6000 96 GB it gives
  780,970 KV tokens against 454,594 for bf16 -- 1.72x -- and scores the same
  30/30 on the estonia long-context suite. E5M2 is refused at startup: the
  kernel could handle it, but nothing here has validated it.
- **Checkpoints exported under downstream naming** (`model_type:
  qwen3_8_flash_next`, `Qwen3_8FlashNextForConditionalGeneration`) must be
  remapped to the names upstream registers — `qwen4_exp` /
  `Qwen4ExpForConditionalGeneration`, which is what the official
  `Qwen/Qwen3.8-Flash-Next` checkpoint uses. Only the config labels differ; the
  weights are identical. `docker/ple-disk-entrypoint.sh` does this
  automatically.

### sm120 (Blackwell) prerequisites

- **CUDA >= 12.9.** flashinfer refuses to JIT any sm120 kernel below that
  (`SM 12.x requires CUDA >= 12.9`), and the failure surfaces only *after* the
  model has loaded.
- **`libcurand-dev`** must be present, or the sm120 MXFP8 GEMM JIT fails on a
  missing `curand_kernel.h` pulled in via CUTLASS's `tensor_fill.h`.
- **`VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel`.** flashinfer's
  cutlass `mm_mxfp8` rejects this model's projection shapes on sm120
  (`Problem size is not supported for mm_mxfp8`); disabling it falls through to
  Marlin.

`docker/Dockerfile.ple-disk` encodes all of the above.
