# Autoresearch journal (append-only)

## v2/v3 (rank 4, budget probes): 135M PROXY REJECTED

v2 validation (rank 4, 150 steps): sa_svd reference 632.30/613.59 and
612.55/612.43 (MEAN ~617.7), zero-init 310.86/301.71 (MEAN ~306.3). The
sign flip PERSISTS at ratio 0.11, same ~2x gap as rank 16. The
rank/n_groups hypothesis from the v1 diagnosis is falsified: the ratio is
not the driver.

v3 budget probe (rank 4, 600 steps, seed 0, batch 8x2 after an OOM at 16x1
under GPU sharing): sa_svd 227.88 vs zero 169.92. Still flipped, but the
gap narrows with budget: 2.0x at 150 steps, 1.34x at 600. Consistent with
the known mechanism: zero-point quantization breaks shift-equivariance and
hands the sa_svd arm a damaged start (~3 nats at step 10); its optimization
advantage claws back slowly, and at 135M scale the crossover (if any) lies
beyond practical proxy budgets.

**Verdict: SmolLM2-135M cannot proxy the 1.7B ordering at any tested rank
or budget. Loop remains halted pending a redesign decision.**

**Research finding worth keeping regardless of the loop:** on the stock
quantizer, SA-SVD inverts (hurts) on a small, heavily damaged model under
short budgets, and the deficit shrinks as budget grows. This adds a model
size / budget axis to the regime map and strengthens the case for testing
the thesis fork (qzero_unquantized), which removes exactly the zero-point
channel implicated here.

## v1 (rank 16): ARCHIVED, proxy invalid (sign flip)

**Current best:** exact SA-SVD (reference), MEAN 435.22 (avg of exp 0 and 0b)
**Keep rule:** KEEP only if candidate MEAN < 430.2 (reference - 5.0)

Calibration: two identical reference cycles differed by 2.30 PPL in MEAN at
fixed seeds (pure GPU nondeterminism; per-seed deltas 2.63 and 1.97).
Threshold set to 5.0 PPL, roughly 2x the observed cycle noise. Cycle time
~8.3 min for both seeds. Note: 135M at 2-bit is far more damaged than 1.7B
(PPL ~435 vs ~36); expected, and fine for a proxy with headroom.

| id | date | hypothesis | change | seed0 | seed1 | mean | verdict |
|----|------|------------|--------|-------|-------|------|---------|
| 0 | 2026-06-05 | reference: exact SA-SVD | none (initial init_fn.py) | 430.46 | 442.28 | 436.37 | REFERENCE |
| 0b | 2026-06-05 | noise calibration: identical rerun of exp 0 | none | 427.83 | 440.32 | 434.07 | REFERENCE |
| 0c | 2026-06-05 | discrimination check: zero init (PEFT default, B=0) should be WORSE per the 1.7B evidence | temporary init_fn swap | 199.21 | 190.71 | 194.96 | **SIGN FLIP** |

**HALT (2026-06-05): the proxy is invalid as designed.** On the 1.7B target,
SA-SVD beats zero-init in 9/9 pairings; on this 135M/150-step proxy,
zero-init is 2.2x BETTER (195 vs 435). A comparator that inverts the one
difference we trust cannot rank new candidates. No idea experiments until
this is resolved. Debugging notes follow in the next entry.

**Diagnosis (same day):** single-seed debug runs with full logs show the
sa_svd arm starts broken (loss 10.08 / accuracy 4% at step 10) while the
zero arm starts at 7.19. Cause: at rank 16 on 576-wide layers, rank/n_groups
= 16/36 = 0.44 (target: 16/128 = 0.125), so the adapter component carries
~half the pooled spectrum and the zero-point non-equivariance (stock
gptqmodel quantizes zero-points) becomes first-order damage that 150 steps
cannot repair. Not a harness bug: the run is deterministic (427.83
reproduced exactly) and the earlier 245s-vs-66s timing difference was GPU
contention, not SVD cost. New method constraint recorded in program.md:
SA-SVD assumes rank << n_groups. Fix: proxy v2 at rank 4.
