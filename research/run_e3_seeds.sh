#!/usr/bin/env bash
# E3: seed robustness for the mechanism decomposition (red-team.md E3, §10).
#
# Runs the four informative arms at seeds 1 and 2 (seed 0 already exists),
# sequentially, all at batch 2 x grad-accum 8 so the sweep is internally
# consistent and robust to GPU sharing. ~8 runs x ~1.2 h.
#
# The seed changes: baseline adapter init, E4/E6 random directions, data
# order. Per-seed results files keep the {model: {method: ...}} schema from
# colliding across seeds.
#
# Usage:  bash research/run_e3_seeds.sh   (after the GPU is free)
set -u
PY=/home/gap/miniforge3/envs/sasvd/bin/python
cd "$(dirname "$0")/.."

for seed in 1 2; do
  for method in baseline sa_svd random_matched random_flat; do
    log=/tmp/e3_${method}_seed${seed}.log
    echo "=== $method seed $seed -> $log ==="
    env -u PYTHONPATH "$PY" scripts/reproduce.py --method "$method" \
      --batch-size 2 --grad-accum 8 --seed "$seed" \
      --results-path "research/e3_seed${seed}_results.json" > "$log" 2>&1 \
      || echo "FAILED: $method seed $seed (see $log)"
  done
done
echo "E3 sweep done. Results: research/e3_seed1_results.json, research/e3_seed2_results.json"
