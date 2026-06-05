#!/usr/bin/env bash
# v16 seed-hardening queue (journal v16, 2026-06-05): Qwen2 seeds 1-2,
# both arms (kaiming + sa_svd), identical budget to the v14 seed-0 runs
# (lr 3e-5 comes from MODEL_LR in reproduce.py, no --lr needed).
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
    --model-id Qwen/Qwen2-1.5B --batch-size 2 --grad-accum 8 "$@" \
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
  run "qwen2_kaiming_s${seed}" --method kaiming --seed "$seed" \
      --results-path "research/qwen2_seed${seed}.json"
  run "qwen2_sasvd_s${seed}"   --method sa_svd  --seed "$seed" \
      --results-path "research/qwen2_seed${seed}.json"
done
echo "[queue] ALL DONE $(date '+%H:%M:%S')"
