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

v4 (360M, rank 8 = ratio 0.133, 150 steps, batch 8x2, seed 0): sa_svd
221.72 vs zero 171.81. Still flipped, ratio 1.29x. Size series at 150
steps: 135M 2.0x, 360M 1.29x, monotone toward the 1.7B/750-step win
(0.87x). Open discriminator: 1.7B at 150 steps. If sa_svd wins there, the
flip is a small-model effect and a slow 1.7B loop is valid; if it loses,
SA-SVD's advantage is budget-dependent and no fast proxy exists, which
would itself be a headline finding (crossover budget decreasing with model
size, consistent with the zero-point start-damage mechanism).

v5 (1.7B, rank 16 = the target config, 150 steps, batch 2x8, seed 0):
sa_svd 48.55 vs zero 45.87. **The flip exists at the target model too.**
Complete series at 150 steps: 135M 2.00x, 360M 1.29x, 1.7B 1.06x; at 750
steps on 1.7B: 0.87x (sa_svd wins, 3 seeds). Step-10 losses: sa_svd 13.1
vs zero 9.9, so the zero-point start damage is universal, not a
small-model artifact; what varies is recovery speed vs budget.

**HEADLINE FINDING of the proxy validation: SA-SVD's advantage on the
stock quantizer is budget-dependent. Crossover at 1.7B lies between 150
and 750 steps; at 360M/135M beyond practical budgets. Practitioners with
short fine-tuning budgets should not use SA-SVD on stock gptqmodel. The
qzero_unquantized fork removes exactly the implicated channel and may
eliminate the crossover entirely; the fork test rises to top priority.**

Operational silver lining: on an uncontended GPU a full 1.7B/150-step arm
takes ~190s (GPTQ ~2 min, not ~12; earlier estimates were inflated by GPU
sharing). A loop AT THE TARGET MODEL is feasible once the smallest budget
with the correct ordering is found (crossover bisection: 450, then 300).

## v6-v11 (2026-06-05): the detective arc, ending in an upstream PEFT bug

All runs 1.7B, rank 16, 2x8, seed 0 unless noted. Full eval = whole
WikiText-2 test via --eval-tokens 999999 (identical protocol to pipeline).

| v | arm | steps | eval | PPL |
|---|-----|-------|------|-----|
| v6 | sa_svd / zero(written) | 450 | subset | 37.02 / 32.95 |
| v7 | sa_svd / zero(written) | 750 | subset | 36.03 / 32.01 |
| v8 | zero(written) | 750 | FULL | 33.46 |
| v9 | zero(written) seeds 1,2 | 750 | FULL | 33.16 / 33.81 |
| v9 | full-scale-A control | 750 | FULL | 33.87 |
| v10 | untied full-scale A | 750 | FULL | 34.78 |
| v11 | PEFT default (untouched) seed 1 | 750 | FULL | **12230.68, did not train** |

Falsified along the way: the A-scale hypothesis (v9 control: 0.125x and 1x
both ~33.5-33.9), the subset-eval-artifact hypothesis for 750 steps (v8:
full eval confirms 33.5), layer-tying as main driver (v10: untied still
34.78, tying worth ~1 PPL at most).

**v11 + tensor probe found the root cause.** The untouched PEFT-default
QA-LoRA adapter has lora_A = 0 AND lora_B = 0 (probed directly: all zeros,
fp32, cuda). That is an exact saddle point: grad(A) ~ B = 0, grad(B) ~ A =
0, the adapter can never train. v11's PPL equals the unadapted quantized
model (the known ~12.2k init PPL). Source of the bug (peft 0.19.1):
lora/layer.py update_layer kaiming-inits lora_A (line 243), THEN the
QALoRA variant init REPLACES lora_A with a fresh nn.Linear (line 248 ->
variants.py:486) whose initialization never lands on the GPTQ layer path;
the result is uninitialized memory: zero pages on a fresh GPU (today),
arbitrary junk otherwise. reset is never re-run after the replacement.

**Consequences, in order of severity:**
1. Every "baseline (random init)" number in this project was measured
   against an UNINITIALIZED adapter, not a Kaiming init: the E3 baselines
   (38-42, spread 4.2) were junk-memory inits, unseeded by construction,
   which explains their anomalous spread. The honest Kaiming baseline is
   the harness-written v10 arm: 34.78 (1 seed), which BEATS sa_svd
   (35.87 +/- 0.2, 3 seeds). Verification seeds running.
2. The Qwen2 "did not train, inf grads" baseline and E2's no-train arm are
   reinterpreted: dead or pathological junk adapters, not a regime
   property. The budget-dependence story (v5/v6) collapses too: its
   "zero" arms were written (valid) but its comparison target was always
   confounded.
3. sa_svd's own numbers are unaffected (its adapters were always written
   via write_adapter_weights), but its 9/9 win was against a broken
   opponent. The corrected comparison is sa_svd vs written-Kaiming, and
   on current evidence sa_svd LOSES it by ~1 PPL.
4. Upstream bug report due against peft (QALoRA variant init order),
   with the probe and v11 as evidence. Ironic note: the QA-LoRA
   integration originates from this project's own PRs (#2571, #2664).
5. The 135M/360M size-series flips (v1-v4) compared written arms against
   written arms and remain internally valid.

## v12 (2026-06-05): verdict. The corrected baseline beats SA-SVD 9/9.

Written-Kaiming baseline (untied, full scale, the init PEFT *intended* as
default), 1.7B, 750 steps, full eval, seeds 0/1/2: **34.78 / 33.51 / 33.96**
(mean 34.08, spread 1.27). Against sa_svd's 36.02 / 35.61 / 35.99 (mean
35.87, spread 0.41): the corrected baseline wins **all 9 pairings**, worst
Kaiming seed (34.78) vs best sa_svd seed (35.61), margin 0.83 PPL.

**Corrected headline for the pinned stack: a properly initialized random
QA-LoRA adapter beats SA-SVD by ~1.8 PPL mean. SA-SVD's previously
measured 9/9 advantage was an artifact of comparing against PEFT's broken
(uninitialized) default adapter.** What remains to SA-SVD on this stack:
lower seed spread (0.41 vs 1.27) and determinism; not the mean.

Scope notes, carefully: (1) This concerns the pinned Option-A stack
(upstream peft 0.19.1 + stock gptqmodel). The thesis-era stack used a
different QA-LoRA implementation; the thesis collapse-regime claims are
not automatically affected and the fork test remains open. (2) The
TinyLlama and Qwen2 baseline columns in results.json are invalidated the
same way (junk-init opponents); the sa_svd columns stand as absolute
numbers. (3) README/CLAUDE.md/results corrections on main are required but
should land as one coherent rewrite after the upstream bug is reported and
the E-series mechanism findings are re-read against the corrected
baseline (the E4/E6 within-written comparisons remain valid since all
arms there were written).

To do, in order: upstream PEFT issue (variant init order; fix is to re-run
reset_lora_parameters after the variant replaces lora_A, or init inside
the variant), re-run TinyLlama/Qwen2 baselines with written Kaiming,
then the public-text rewrite.

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
