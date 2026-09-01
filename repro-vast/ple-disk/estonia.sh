#!/bin/bash
cd /workspace/bench
/workspace/vllmenv/bin/python llm_decode_bench.py --port 8000 --model q38fn \
  --test-profile estonia \
  --profile-concurrency ${CONC:-2} \
  --profile-runs ${RUNS:-30} \
  --max-tokens 40000 \
  --contexts 0 \
  --display-mode plain \
  --output ${OUT:-/workspace/estonia_noprefix.json}
