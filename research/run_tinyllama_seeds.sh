#!/usr/bin/env bash
# v15 seed-hardening queue (journal v15, 2026-06-05): TinyLlama seeds 1-2,
# both arms (kaiming + sa_svd), identical budget to the v14 seed-0 runs.
# One results file PER SEED because results are keyed [model][method] and
# seeds of the same method would overwrite each other.
# VRAM rule: gpu_guard before each run, batch 2 x accum 8, 8 GB reserved.
set -u
PY=/home/gap/miniforge3/envs/sasvd/bin/python
cd "$(dirname "$0")/.."
mkdir -p research/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  local tag=$1; shift
  bash research/gpu_guard.sh 15000
  echo "[queue] START $tag $(date '+%H:%M:%S')"
  env -u PYTHONPATH "$PY" scripts/reproduce.py \
    --model-id TinyLlama/TinyLlama_v1.1 --batch-size 2 --grad-accum 8 "$@" \
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

for seed in 1 2; do
  run "tinyllama_kaiming_s${seed}" --method kaiming --seed "$seed" \
      --results-path "research/tinyllama_seed${seed}.json"
  run "tinyllama_sasvd_s${seed}"   --method sa_svd  --seed "$seed" \
      --results-path "research/tinyllama_seed${seed}.json"
done
echo "[queue] ALL DONE $(date '+%H:%M:%S')"
