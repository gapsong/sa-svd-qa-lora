#!/usr/bin/env bash
# v18 queue (journal v18, 2026-06-05): activation-whitened SA-SVD, seed 0,
# SmolLM2 + TinyLlama. Budget identical to the v14/v15 arms so the numbers
# are directly comparable (750 steps, batch 2x8, per-model default lr).
# VRAM rule: gpu_guard before each run, 8 GB reserved.
set -u
PY=/home/gap/miniforge3/envs/sasvd/bin/python
cd "$(dirname "$0")/.."
mkdir -p research/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  local tag=$1 model=$2
  bash research/gpu_guard.sh 15000
  echo "[queue] START $tag $(date '+%H:%M:%S')"
  env -u PYTHONPATH "$PY" scripts/reproduce.py \
    --model-id "$model" --method sa_svd_white --seed 0 \
    --batch-size 2 --grad-accum 8 \
    --results-path research/whitened.json \
    > "research/logs/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[queue] FAILED $tag rc=$rc $(date '+%H:%M:%S')"
    tail -5 "research/logs/${tag}.log"
  else
    echo "[queue] DONE $tag $(date '+%H:%M:%S')"
    grep -E "perplexity|Wrote" "research/logs/${tag}.log" | tail -2
  fi
}

run whitened_smollm2   HuggingFaceTB/SmolLM2-1.7B
run whitened_tinyllama TinyLlama/TinyLlama_v1.1
echo "[queue] ALL DONE $(date '+%H:%M:%S')"
