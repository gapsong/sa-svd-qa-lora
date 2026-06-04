#!/usr/bin/env bash
# One autoresearch cycle: evaluate the current autoresearch/init_fn.py at
# seeds 0 and 1, print each METRIC line and the mean. Extra args pass
# through to harness.py (do not change budget knobs mid-journal).
set -u
PY=/home/gap/miniforge3/envs/sasvd/bin/python
cd "$(dirname "$0")/.."

out=$(mktemp)
for seed in 0 1; do
  env -u PYTHONPATH "$PY" autoresearch/harness.py --seed "$seed" "$@" 2>&1 \
    | grep -E "^METRIC" | tee -a "$out"
done
awk -F'val_ppl=' '{split($2, a, " "); s += a[1]; n++}
                  END {if (n > 0) printf "MEAN val_ppl=%.4f over %d seeds\n", s/n, n;
                       else print "MEAN unavailable: no METRIC lines (run failed?)"}' "$out"
rm -f "$out"
