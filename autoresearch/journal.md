# Autoresearch journal (append-only)

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
