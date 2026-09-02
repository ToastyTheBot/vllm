# Qwen3.8-Flash-Next-NVFP4 on one RTX PRO 6000 96 GB, engrams on disk

Sessions 2026-09-01 / 2026-09-02. Hardware: vast.ai instance 49421402, machine 56397 (Texas),
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
| engine anonymous memory (non-evictable) | **3.03 GiB** (`Pss_Anon`) |
| engine file-backed RSS (evictable page cache) | 25.27 GiB (`Pss_File`), of which 24.77 GiB is the safetensors mmap |
| engine total RSS | 28.59 GiB |

Source: `fp8kv/fp8_metrics.txt` (fp8-KV run). `final_metrics.txt` from the
bf16 run records only the totals (28.65 GiB RSS), not the split.

**Read these two numbers separately.** The hard requirement is the ~3.1 GiB of
anonymous memory. The file-backed portion is clean, mmap-backed page cache that
the kernel reclaims under pressure -- it grew to hold nearly the whole 23.84 GiB
table only because this box has 1 TB of RAM. On a 28 GB host the kernel simply
keeps fewer table pages resident and serves more gathers from NVMe.

An earlier revision of this document quoted 3.67-3.92 GiB RSS. That was measured
shortly after startup, before the page cache warmed, and understated steady-state
RSS; the anon/file split above is the accurate picture.

Correctness of the offload path is established two ways:

1. **Gather oracle** (`oracle.py`): 512 random row ids gathered through
   `MmapPleTable`; the first **64 of them** byte-compared against
   `safetensors.safe_open`. **0/64 mismatches on weights, 0/64 on block
   scales.** (The gather is 512 wide to exercise the chunking; the byte
   comparison covers 64.) Dequant output finite,
   absmean 0.0061, range ±0.035.
2. **End-to-end**: the 136,562-token estonia prompt answers correctly and
   terminates cleanly — `finish=stop`, 2,267 completion tokens, final line
   *"...V-441 corresponds to Mirel Instrument, headquartered in Estonia."*

## The long-context crash: root-caused and fixed

### Symptom

With prefix caching enabled, long-context runs produced degenerate repetition
(`'ductductduct...'`) and then a `CUDA error: an illegal memory access was
encountered`, killing the engine. Reported upstream as **#54173** with
`--no-enable-prefix-caching` as the known workaround.

### Root cause

`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`, `add_request`, seeded the
mamba running-state column as:

```python
(new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
```

`cache_config.block_size` is **not** the mamba group's block size at runtime.
`EngineCore` rewrites it to `min(g.kv_cache_spec.block_size for g in
kv_cache_groups)` once the KV cache config is known
(`vllm/v1/engine/core.py:336`), so on a hybrid model it becomes the *smallest*
group's block size. On Qwen3.8-Flash-Next that is the PLE short-conv group's
**4**, against a mamba block size of **1568** — a 392x overshoot.

That column indexes the mamba group's block table. Overshooting walks off the
row, reads a garbage block id, multiplies it by the state stride, and faults
inside `precopy_mamba_align_fused_kernel`.

### Evidence

A temporary host-side probe on the pre-copy launch caught it directly:

```
MAMBA-PRECOPY-PROBE: bt_width=42 num_reqs=3
src_col=[7, 6, 2743]  dst_col=[7, 7, 7]  bad_src=[False, False, True]
```

`2743 / 7 = 391.86`, i.e. exactly `mamba_block_size / cache_config.block_size =
1568 / 4`. A second capture gave `2351 / 6 = 391.83` — the same constant ratio,
which is what identified the divisor.

`num_computed_tokens == 0` yields `-1` for any positive divisor, so a fresh
request never copies. That is precisely why **only prefix-cache hits crashed**,
and why toggling prefix caching appeared to fix it.

### Attribution

The traceback initially pointed into `ple_mmap._forward_nvfp4`, because that
function's `ids.detach().to("cpu")` is the first synchronizing point in the
whole forward, so any earlier async fault surfaced there. Forcing an explicit
`torch.cuda.synchronize()` ahead of it moved the fault to
`vllm/compilation/cuda_graph.py`, and `--enforce-eager` plus
`CUDA_LAUNCH_BLOCKING=1` then named the real site:
`mamba_hybrid.py:preprocess_state` -> `mamba_utils.py:run_fused_precopy`.

### Fix

Read the mamba spec's block size, falling back to
`cache_config.mamba_block_size` for the window before the spec is resolved
(`add_request` can run before the first `preprocess_state`), and refuse to fall
back to `cache_config.block_size`, which silently restores the bug. This is the
same defect class as **#53142**, fixed in `mamba_utils.py`'s V1 path but never
in this V2 align path.

### Verification

| check | result |
|---|---|
| 8 rounds x 3 concurrent shared-prefix requests, up to 125,328 prompt tokens | **0 IMAs** (previously crashed in round 0, every time) |
| estonia, 30 runs, c=6, **prefix caching ON** | **30/30 correct**, 0 errors, tokens p50/p90/max 2711/3679/5033 |
| estonia, 30 runs, c=6, **prefix caching ON + MTP 3** | **30/30 correct**, 0 errors (`estonia_final.json`) |
| lavd (context consistency), 30 runs, c=6, prefix caching ON | **29/30**, 0 errors |
| estonia, 30 runs, c=3, prefix caching off | 30/30 correct (pre-fix control) |

### Withdrawn: the Marlin MXFP8 attribution

An earlier revision of this document claimed `MarlinMxfp8LinearKernel` was
numerically wrong for this model. **That was wrong and is withdrawn.** The
comparison behind it changed three variables at once (kernel, concurrency, and
prefix-cache reuse); the kernel was simply the one changed most recently. The
30/30 run above uses Marlin, with prefix caching on. Marlin is fine.

What remains true: flashinfer's cutlass `mm_mxfp8` genuinely cannot serve this
model's projection shapes on sm120 (`Problem size is not supported for
mm_mxfp8`), so it must be disabled and selection falls through to Marlin.

## fp8 KV cache: implemented

Previously recorded here as unavailable. The QSA sparse-attention path now
reads fp8 pages directly: the Triton kernel dequantizes them to the query
dtype on load using the same per-tensor `k_scale`/`v_scale` that
`reshape_and_cache_flash` used to write them.

Five separate refusals had to be cleared, each in a different place:

1. `Qwen4ExpQSABackend.supported_kv_cache_dtypes` (backend advertisement)
2. `Qwen4ExpQSAFlashAttentionImpl.__init__` (`kv_cache_dtype` string)
3. `Qwen4ExpQSAAttention.__init__` (`cache_config.cache_dtype`)
4. `Qwen4ExpQSAAttention.__init__` (`kv_cache_torch_dtype`, which is
   `torch.uint8` -- vLLM stores every fp8 mode as uint8 and kernels re-view it)
5. `FlashAttentionImpl.__init__`, which refuses fp8 whenever FlashAttention
   lacks device support (sm120). QSA inherits that class only for
   `do_kv_cache_update` and metadata and never runs FA's attention kernel, so
   the check does not apply to it.

One real bug surfaced during bring-up: the dequantized BF16 tile lives
alongside the fp8 tile it came from, and with 2-stage pipelining that exceeded
sm120's shared-memory budget (`Required: 106496, Hardware limit: 101376`). The
fp8 path uses `num_stages=1`.

| | bf16 KV | **fp8 KV** |
|---|---|---|
| GPU KV cache | 454,594 tokens | **780,970 tokens** (1.72x) |
| Max concurrency @ 180,224 | 2.52x | **4.33x** |
| estonia, 30 runs, c=6, prefix caching + MTP 3 | 30/30 | **30/30** |

fp8 KV is accuracy-neutral on this benchmark and buys 1.72x KV capacity.
Only E4M3 is accepted; E5M2 is refused at init rather than at first request.

## Vision tower

Verified working under the full configuration (fp8 KV, prefix caching, MTP 3,
engrams on disk). Two synthetic images with controlled content -- a red circle
top-left plus a blue square bottom-right, and the digit 7 -- both answered
correctly and with correct spatial relations
(`fp8kv/vision_test_output.txt`). Synthetic rather than stock imagery so a
correct answer cannot come from a prior.

## Other findings

### MTP 3 zero-acceptance -- also caused by the same bug, now fixed

Before the fix, MTP 3 reported `Mean acceptance length: 1.00, Accepted: 0
tokens`, sustained: it drafted at ~380 tok/s and nothing was ever accepted.
That was attributed here to the downstream fork's `4c1f7b2c3` *"fix(mtp):
preserve position-zero embeddings (#539)"*. **That attribution is withdrawn.**

With the mamba seeding fix, MTP 3 reports a mean acceptance length of
**2.65-3.13** across runs (`fp8kv/fp8_metrics.txt` captures 2.65-2.96 under
load on the fp8-KV run).
The corrupted mamba state was making every draft mismatch the target, so zero
acceptance was a second symptom of the same defect.

### fp8 KV cache is unavailable

`vllm/models/qwen4_exp/nvidia/qsa.py:114` raises `Qwen4Exp QSA requires a BF16
main KV cache` for any `kv_cache_dtype` other than `auto`/`bfloat16`. Blanket
refusal, not a knob -- the sparse-attention kernel has no fp8 path. The
downstream fork has it (`701985284`) but via its b12x QSA implementation, which
upstream does not have.

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
