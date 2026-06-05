---
name: research-interpret
description: Interpret a finished training run for this repo (anomaly checklist, known PPL anchors, bottleneck detection, paper-grounded explanation). Use when a run finishes, when results look odd, or when the user asks what a number means.
---

# Research interpret

Input: a run log (research/logs/*.log or a harness METRIC line) and/or a
results JSON. Output: an interpretation the user can trust, plus what to
do next. Protocol source: research/WORKFLOW.md section 6.

## Anomaly checklist (run BEFORE quoting any number)

1. **Trainability:** grep the log for `grad_norm`: healthy band here is
   ~9-17, finite. Any `inf`/`nan` step is a finding; probe init tensors
   before blaming the method (journal v11-v13: it was upstream bugs).
2. **Start state:** loss at step ~10. Anchors (SmolLM2-1.7B, 2-bit):
   healthy ~7-10; 13+ means the starting function is damaged (zero-point
   non-equivariance) and short budgets will mislead.
3. **Final PPL anchors (WikiText-2, this pipeline):** ~12200 = unadapted
   2-bit model, the adapter learned nothing (saddle: A=B=0); ~36 = sa_svd;
   ~34 = written-Kaiming baseline; ~30 = "usable LM" threshold; ~427 =
   135M-scale reference. A number near an anchor is a diagnosis, not a
   coincidence.
4. **Determinism:** identical config rerun should land within 2.3 PPL
   (calibrated GPU noise). Larger unseeded spread means uncontrolled
   state somewhere; find it before comparing methods.
5. **Comparability:** same steps, batch x accum, lr, eval protocol as the
   row it is compared against? Subset eval and full eval are different
   protocols; never mix (journal v6-v8).

## Bottleneck detection

Expected on a free RTX 4090 at 1.7B: GPTQ ~2 min, training ~0.8 s/step,
full eval ~2-3 min, whole 750-step arm ~15-25 min.
- Much slower: check `nvidia-smi` for contention (earlier sessions saw 3x
  inflation from a sim using 5.6 GiB). Contention changes timing, not
  correctness; do not retune hyperparameters because of a slow run.
- OOM: another app probably claimed VRAM mid-run; the fix is the guard +
  batch 2 x accum 8, not a smaller model.
- Eval dominating: the known double-count quirk makes eval pessimistic
  but fair across methods; do not "fix" it piecemeal (CLAUDE.md sec 9).

## Explain with the literature (the learning track)

When interpreting, connect the observation to the relevant paper from
research/WORKFLOW.md's reading list (LoRA saddle structure, QA-LoRA group
constraint, PiSSA init, LoRA+ asymmetry). One paragraph, plain language,
so the user learns the mechanism, not just the verdict.

## Output format

- verdict line: TRUSTWORTHY / SUSPECT (which checklist item failed)
- the number in context: nearest anchor, delta vs its comparison row,
  whether the delta clears the noise threshold
- bottleneck note if timing was off
- one "what we learned" line, paper-linked when possible
- recommended next step (more seeds? probe? journal verdict?)
