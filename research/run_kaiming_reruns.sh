#!/usr/bin/env bash
# Corrected-baseline rerun queue (journal v12 follow-up, 2026-06-05).
#
# Re-measures every invalidated "baseline (random init)" column with an
# explicitly written Kaiming init (research/kaiming_init.py), because peft
# 0.19.1 never initializes the QA-LoRA default adapter (journal.md v11).
#
# Order:
#   1. SmolLM2-1.7B   crosscheck vs the harness v12 arm (expect ~34-35)
#   2. TinyLlama-1.1B replaces the invalidated 34.63
#   3. Qwen2-1.5B     replaces the invalidated 53.15 (lr auto -> 3e-5)
#   4. Qwen2-1.5B     LR check at 1e-4: was the 3e-5 downgrade only ever
#      needed because of the junk init? Separate results file so run 3's
#      entry is not overwritten.
#
# VRAM: batch 2 x grad-accum 8 (effective 16, the published budget) and a
# gpu_guard wait before each run, keeping ~8 GB reserved for other apps.
set -u
PY=/home/gap/miniforge3/envs/sasvd/bin/python
cd "$(dirname "$0")/.."
mkdir -p research/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(--method kaiming --batch-size 2 --grad-accum 8 --seed 0
        --results-path research/kaiming_reruns.json)

run() {
  local tag=$1; shift
  bash research/gpu_guard.sh 15000
  echo "[queue] START $tag $(date '+%H:%M:%S')"
  env -u PYTHONPATH "$PY" scripts/reproduce.py "$@" \
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

run smollm2_kaiming   "${COMMON[@]}" --model-id HuggingFaceTB/SmolLM2-1.7B
run tinyllama_kaiming "${COMMON[@]}" --model-id TinyLlama/TinyLlama_v1.1
run qwen2_kaiming     "${COMMON[@]}" --model-id Qwen/Qwen2-1.5B
run qwen2_kaiming_lr1e4 --method kaiming --batch-size 2 --grad-accum 8 \
    --seed 0 --model-id Qwen/Qwen2-1.5B --lr 1e-4 \
    --results-path research/kaiming_qwen2_lr1e4.json
echo "[queue] ALL DONE $(date '+%H:%M:%S')"
