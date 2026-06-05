#!/usr/bin/env bash
# Block until the GPU has at least $1 MiB free (default 15000).
#
# Standing rule for this machine: keep ~8 GB VRAM reserved for other
# applications. Our 1.7B-class runs need ~13-14 GiB at batch 2 x accum 8,
# so 24 GiB total - 8 GiB reserved leaves just enough; this guard makes
# sure we never START a run into a GPU that cannot fit it. It does not
# protect against another app claiming memory mid-run; the batch size is
# chosen so both fit (run + 8 GiB < 24 GiB).
set -u
need=${1:-15000}
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -ge "$need" ]; then
    break
  fi
  echo "[gpu_guard] ${free} MiB free < ${need} MiB needed; waiting 60s"
  sleep 60
done
