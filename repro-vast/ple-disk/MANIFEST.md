# Which artifact backs which claim

Written because an earlier revision of `FINDINGS.md` quoted numbers without
saying which file they came from, which made several of them unverifiable.
Every empirical claim in `FINDINGS.md` and `docs/features/ple_disk_offload.md`
should be traceable to a row here.

Reproduce any summary with:

```bash
python repro-vast/ple-disk/rep.py <json> "<label>"
```

## Accuracy runs

| artifact | config | result |
|---|---|---|
| `estonia_mtp3.json` | bf16 KV, prefix caching **on**, MTP 3, c=6, **pre-fix** | 12/30 — 18 degenerate repetition runs (`ductductduct...`); the bug this branch fixes |
| `estonia_noprefix.json` | bf16 KV, prefix caching **off**, no MTP, c=3, pre-fix | 30/30 — the control that localised the bug to the prefix-caching path |
| `estonia_prefix.json` | bf16 KV, prefix caching **on**, no MTP, c=6, **post-fix** | 30/30 |
| `estonia_mtp.json` | bf16 KV, prefix caching on, MTP 3, c=2, post-fix, 6 runs | 6/6 (smoke) |
| `estonia_final.json` | bf16 KV, prefix caching on, MTP 3, c=6, post-fix | 30/30 |
| `lavd.json` | bf16 KV, prefix caching on, MTP 3, c=6, post-fix | 29/30 (26 exact / 3 near / 1 fail) |
| `fp8kv/estonia_fp8.json` | **fp8 KV**, prefix caching on, MTP 3, c=6 | 30/30 |

`estonia_emu.json`, `estonia_c2.json`, `estonia_nomtp.json` are **crash
records**, not accuracy results: every run reports `ok=false` with an empty
error string because the engine died mid-run (the IMA). They are kept only as
evidence that those configurations could not complete, and carry no
diagnostic content of their own — the diagnosis is in `FINDINGS.md`.

## Metrics

| artifact | contains |
|---|---|
| `final_metrics.txt` | bf16-KV run: engine `Rss`/`Pss` totals, all-process RSS, PLE attach line, model load time, KV cache size |
| `fp8kv/fp8_metrics.txt` | fp8-KV run: full serve config, KV cache size, **MTP acceptance**, **`Pss_Anon` vs `Pss_File` split**, safetensors mmap RSS, IMA count |
| `serve_key_lines.txt` | PLE attach / model load / KV size lines from the bf16 runs |

The anon-vs-file memory split and the MTP acceptance figures are in
`fp8kv/fp8_metrics.txt` only; the earlier `final_metrics.txt` has totals alone,
which is why prose quoting a split must cite the fp8 file.

## Vision

| artifact | contains |
|---|---|
| `fp8kv/vision_test.py` | the harness: two synthetic images, expected substrings, PASS/FAIL |
| `fp8kv/mkimg.py` | generates them deterministically (red circle + blue square; the digit 7) |
| `fp8kv/vision_test_output.txt` | captured output — both PASS, with spatial detail |

Synthetic rather than a stock photo so a correct answer cannot be a lucky
prior: the model has to report *which* colour is in *which* corner.

## Scripts

| artifact | purpose |
|---|---|
| `oracle.py` | gathers 512 rows through `MmapPleTable`, byte-compares the **first 64** against a `safetensors` reference. The 512/64 split is deliberate (gather width vs verified width) and prose should say "64 verified". |
| `repro_prefix.py` | the shared-prefix IMA reproducer (rounds of 3 concurrent, growing prefix) |
| `direct.py` | one estonia request straight at the server, no bench harness |
| `rep.py` | summarises any bench JSON |
| `serve.sh`, `serve_mtp.sh`, `fp8kv/serve_fp8kv.sh` | exact serve invocations |
| `estonia.sh` | exact bench invocation |

## Known gap

Accuracy scoring is performed by `llm-inference-bench`
(`llm_decode_bench.py`, version 0.4.29 per the JSON metadata), which is **not
vendored here**. The pass/fail definition behind every `N/30` above therefore
cannot be re-derived from this repo alone. The raw per-run model outputs are in
the JSONs, so the scoring can be independently recomputed, but the harness
itself is external.
