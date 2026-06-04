# Autoresearch program: better QA-LoRA adapter initializations

Human-edited research brief, in the spirit of karpathy/autoresearch
(one mutable file, one metric, fixed budget, keep or revert).

## Goal

Minimize **mean WikiText-subset perplexity over seeds 0 and 1** of a 2-bit
GPTQ SmolLM2-135M fine-tuned with QA-LoRA for a fixed budget, by changing
**only** `autoresearch/init_fn.py::build_init`.

## The contract

- `build_init(W, rank, group_size, seed) -> (lora_A, lora_B)`. The harness
  derives the quantized base as `W - B*expand(A)`, so the step-0 function is
  always W: candidates cannot cheat by changing the model, only the adapter
  parameterization. Any randomness must use `seed`.
- Everything else is FIXED: harness.py, the budget knobs (150 steps, batch
  16, lr 1e-4, rank 16, group 16, 2400 Alpaca samples, 60k eval tokens), the
  model, the data. Changing a budget knob invalidates the journal.
- One cycle = `bash autoresearch/run.sh` (two seeds, prints MEAN).

## Keep / revert rule

Keep a candidate only if its MEAN improves on the current best MEAN by more
than the noise threshold. Provisional threshold: **0.3 PPL**, to be
calibrated by running the SA-SVD reference cycle twice before the first real
experiment (the E3 lesson from the 1.7B runs: single-seed deltas up to 2.7
PPL were pure noise). Record every experiment in `journal.md` regardless of
verdict: failures are data.

## Journal protocol (journal.md, append-only)

Per experiment: id, date, hypothesis (one line), change summary (one line),
seed-0 and seed-1 metrics, mean, verdict (KEEP / REVERT / FAILED), one-line
takeaway. Keep the current-best mean and its init description at the top.

## What we already know (do not re-discover; see results/ablation-2026-06-05.md)

On SmolLM2-1.7B, 3 seeds: the SVD's singular values carry a consistent gain
(concentrated beats flat at matched norm, 3/3 paired); the singular vectors
buy reliability, not a unique optimum; zero-contribution init is strictly
worse; scale alone (flat spectrum) is weakest. The transfer assumption that
135M results predict 1.7B results is UNVERIFIED: promote any kept candidate
to one full 1.7B run (scripts/reproduce.py) before believing it.

## Seeded ideas (ranked; the loop may add its own)

1. **Spectrum tempering:** use s_k^alpha instead of s_k (alpha in
   {0.5, 0.75, 1.25, 1.5, 2}), norm-matched or not. We know the spectrum
   shape matters; nobody has asked which shape is best.
2. **Factor asymmetry:** put the scale in B only (B = U*S, A = Vh) or A only
   instead of the symmetric sqrt split. Same function, different optimizer
   dynamics (LoRA+ suggests asymmetric learning rates help; asymmetric init
   is the zero-cost cousin).
3. **Global scale multiplier:** sqrt(S) * c for c in {0.5, 1.5, 2}. Is the
   SA-SVD magnitude near the optimum or just in the right ballpark?
4. **Effective-rank reallocation:** zero out trailing components (keep top
   4/8/12 of 16), or per-layer-type ranks (attention vs MLP). E6 suggests a
   few strong components do the work.
5. **Per-layer-type scaling:** different global multipliers for attention vs
   MLP projections (their spectra and energies differ; E1 showed component
   energy 7-18% varying by depth/type).
6. **Non-pooled spectra:** take singular values from the full W instead of
   the pooled matrix (different magnitudes, same directions question).
7. **Direction structure:** Gaussian (non-orthonormal) directions at the
   SA-SVD spectrum; tests whether orthonormality itself matters or only the
   per-component scales.

## Stop conditions (for any unattended session)

Stop and report if: 3 consecutive FAILED cycles, GPU has less than 8 GiB
free, a cycle exceeds 30 minutes, or the journal reaches 25 entries without
a KEEP.
