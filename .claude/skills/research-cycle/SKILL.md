---
name: research-cycle
description: Run one disciplined experiment cycle for this repo (register prediction, guard VRAM, run, interpret, journal, verdict). Use when the user wants to run an experiment, test a hypothesis, try an init idea, or start a research run.
---

# Research cycle

Protocol source: research/WORKFLOW.md. One invocation = one cycle. Do the
steps in order; do not skip the writing steps to get to the run faster.

## 1. Register (before any GPU command)

Write into the journal (autoresearch/journal.md for loop experiments,
research/red-team.md for claim attacks) BEFORE launching:
- hypothesis, one line
- the change, one line
- the registered prediction: what each arm will show, with numbers
- the decision rule: what result KEEPs, what REVERTs

If the experiment uses a new proxy (smaller model, shorter budget, subset
eval), STOP and run a discrimination check first: the proxy must reproduce
one ordering we already trust (current anchor: written-Kaiming ~34.1 beats
sa_svd ~35.9 on SmolLM2-1.7B, 750 steps, full eval). If the proxy inverts
it, the proxy is invalid; record and abort.

## 2. Launch guarded

- Always through the VRAM guard: `bash research/gpu_guard.sh 15000` before
  the run, or use research/run_kaiming_reruns.sh as the template for a
  queue. Keep ~8 GB free for other apps; 1.7B fits at batch 2 x accum 8.
- Loop experiments: edit ONLY autoresearch/init_fn.py, run
  `bash autoresearch/run.sh`. Budget knobs are frozen by program.md.
- Pipeline experiments: `scripts/reproduce.py --method ... --results-path
  research/<exp>.json` (never clobber results/results.json from research).
- Run in background with a Monitor whose filter catches failures too
  (START/DONE/FAILED/Traceback/OOM/inf), not just the success line.

## 3. Interpret

Invoke /research-interpret on the finished log before quoting any number.
A run whose anomaly checklist fails is FAILED, not a data point.

## 4. Verdict and journal

- Compare means against the calibrated noise threshold (loop: 5.0 PPL per
  program.md; pipeline single runs: treat <2.5 PPL deltas as noise until
  seeds 1-2 confirm).
- Append the entry: id, date, hypothesis, change, per-seed metrics, mean,
  verdict (KEEP/REVERT/FAILED), one-line takeaway. Append-only; correct
  old entries forward, never edit them.
- KEEP for a loop candidate means: update the current-best block at the
  top of the journal and leave init_fn.py as the new reference.

## Stop conditions

Stop and report instead of starting another cycle if: 3 consecutive FAILED
cycles, less than 8 GiB GPU free, a cycle exceeds 30 minutes on a free
GPU, or the journal hits 25 entries without a KEEP.
