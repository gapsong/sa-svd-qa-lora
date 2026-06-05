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

## v13 (2026-06-05): root cause refined. The leak is gptqmodel's, PEFT is the second link.

Two stage probes on SmolLM2-135M (logs: /tmp/peft_bug_stage_probe.py,
/tmp/init_leak_probe.py) replace v11's inferred "meta materialization"
link with a proven mechanism:

1. `GPTQModel.from_pretrained` (gptqmodel 6.0.3 loader.py:305-310; still
   on main at 685-690) globally replaces torch.nn.init.kaiming_uniform_,
   uniform_, normal_ with a no-op `skip` and NEVER restores them. Function
   identity probe: the stub is live after from_pretrained, quantize,
   save_quantized, AND from_quantized (whose own suspend_hf_weight_init
   CM captures the stub as "original" at entry and faithfully restores
   the stub at exit). Control: a fresh nn.Linear(36,4) created after the
   pipeline has weight std 1e-4 instead of kaiming's ~0.096.
2. Therefore PEFT's reset_lora_parameters is ALSO a no-op at adapter
   creation time: the probe shows junk (std=inf in bf16) in lora_A even
   BEFORE the QALoRA variant replacement. The variant replacement
   (variants.py:486 in 0.19.1, 489-496 on main) then swaps in a second
   uninitialized tensor and never re-inits: two independent bugs, both
   needed for the v11 disaster, each reportable on its own.
3. Our written inits (sa_svd, kaiming, all E-series arms) are immune:
   they use Tensor.uniform_ and copy_, not torch.nn.init.

Crosscheck landed: `reproduce.py --method kaiming` (new, public pipeline,
batch 2x8, seed 0) gives SmolLM2-1.7B **34.76** vs the harness v12 arm's
34.78. The corrected-baseline result is now reproducible through the same
script that produced the published table. TinyLlama/Qwen2 reruns queued
(research/run_kaiming_reruns.sh, VRAM-guarded).

Issue drafts for both upstreams: research/upstream-bug-reports.md (NOT
filed; awaiting review). Neither bug has an existing upstream issue.

## v14 (2026-06-05): corrected 3-model table. The verdict is 1-1-1, model-dependent.

Kaiming rerun queue complete (research/run_kaiming_reruns.sh, all runs
seed 0, batch 2x8, 750 steps, full eval, healthy by checklist: zero
inf/nan grad steps everywhere):

| model | kaiming (corrected baseline) | sa_svd | verdict |
|-------|------------------------------|--------|---------|
| SmolLM2-1.7B | **34.76** | 36.02 | kaiming wins (3-seed confirmed, v12) |
| TinyLlama-1.1B | 40.90 | **28.68** | sa_svd wins by 12.2 PPL |
| Qwen2-1.5B | 27.90 | 27.61 | tie (0.29 < 2.3 noise) |

Results: research/kaiming_reruns.json. The old baseline column (42.45 /
34.63 / 53.15) is retired: junk-init opponents, not Kaiming.

Reinterpretations this forces:
1. The Qwen2 "did not train, inf grads at every step" regime is DEAD as a
   regime: written-Kaiming trains cleanly at 3e-5 (0 inf steps, loss
   5.09 -> 1.51) and matches sa_svd. The -48% row was entirely the
   uninitialized-adapter artifact.
2. LR check: Qwen2 at 1e-4 also trains cleanly (0 inf steps) but
   generalizes worse: train loss 1.30 (lower than 3e-5's 1.51), WikiText
   33.23 (worse than 27.90). The per-model 3e-5 stays justified, but on
   generalization grounds, not stability; mini "illusion of convergence".
3. TinyLlama is the new interesting case: sa_svd beats a HEALTHY Kaiming
   baseline by 12 PPL. Single seed on both sides; seeds 1-2 are the
   obvious next runs before any public claim. Note the old junk baseline
   (34.63) happened to BEAT proper Kaiming (40.90) here, which is why
   junk-vs-written comparisons are uninterpretable in either direction.
4. The corrected story is neither "SA-SVD always wins" (old table) nor
   "Kaiming beats SA-SVD" (v12, which generalized from SmolLM2 alone).
   It is model-dependent: 1 win, 1 loss, 1 tie. The regime question
   (WHEN does the init help?) is back at the center, now on a clean
   comparison. v12's verdict sentence is corrected forward by this entry:
   "corrected baseline beats SA-SVD" holds for SmolLM2 only.

Open next: TinyLlama/Qwen2 kaiming seeds 1-2, TinyLlama sa_svd seeds 1-2
(its 28.68 is also single-seed), upstream issue filing (drafts ready),
fork test, then the one coherent public-text rewrite with this table.

## v15 (2026-06-05): REGISTERED. TinyLlama seeds 1-2, both arms.

Hypothesis: the TinyLlama ordering (sa_svd 28.68 beats written-Kaiming
40.90 by 12.2 PPL at seed 0) is a property of the model, not seed luck.

Change: 4 runs, seeds 1 and 2, methods kaiming and sa_svd, identical
budget to v14 (750 steps, batch 2x8, lr 1e-4, full eval), results to
research/tinyllama_seed{1,2}.json.

Registered prediction (written before launch): based on the SmolLM2 seed
behavior, sa_svd lands 28.7 +/- 1.0 (it is deterministic given W; only
data order and GPU noise vary) and kaiming lands 40.9 +/- 2.5 (Kaiming
spread was 1.27 on SmolLM2, and TinyLlama looks noisier). Predicted
outcome: sa_svd wins all 9 cross-seed pairings with a minimum gap > 5.

Decision rule: CONFIRMED if sa_svd wins >= 8/9 pairings and the worst-case
gap exceeds 2.5 (the single-run noise bound); UNCLEAR if pairings are
mixed; REFUTED if kaiming wins the means. If CONFIRMED, the public rewrite
presents TinyLlama as the regime where SA-SVD genuinely helps on the
stock quantizer, with 3-seed evidence on both arms.

**v15 RESULT: CONFIRMED, 9/9.** All runs healthy (0 inf/nan steps).

| seed | kaiming | sa_svd |
|------|---------|--------|
| 0 (v14) | 40.90 | 28.68 |
| 1 | 36.27 | 28.35 |
| 2 | 38.88 | 28.31 |
| mean (spread) | 38.68 (4.63) | **28.45 (0.37)** |

sa_svd wins all 9 cross-seed pairings; worst case 28.68 vs 36.27, gap
7.59 > 2.5. Prediction scorecard: sa_svd band hit (predicted +/- 1.0,
got 0.37 spread); kaiming band MISSED (predicted 40.9 +/- 2.5, seed 1
came in at 36.27): the written-Kaiming arm itself is far noisier on
TinyLlama (spread 4.63) than on SmolLM2 (1.27), which is its own finding.
Minimum-gap prediction (>5) held: 7.59.

Two extra observations worth keeping:
1. The illusion-of-convergence pattern generalizes: kaiming reaches LOWER
   train loss (1.25 vs sa_svd's 1.53) in all seeds yet is ~10 PPL worse
   on WikiText. Fitting Alpaca better, generalizing worse; exactly why
   WikiText (never seen in training) stays the headline metric.
2. SA-SVD's seed robustness now replicates across models: spread 0.41
   (SmolLM2) and 0.37 (TinyLlama) vs Kaiming's 1.27 and 4.63. The
   13x spread reduction on TinyLlama exceeds the SmolLM2 effect.

Corrected 3-model summary as of v15 (3 seeds where marked):
- SmolLM2-1.7B: kaiming 34.08* beats sa_svd 35.87* (3 seeds each)
- TinyLlama-1.1B: sa_svd 28.45* beats kaiming 38.68* (3 seeds each)
- Qwen2-1.5B: tie at ~27.6-27.9 (1 seed each)
The regime question is the story; per-model evidence is now solid for
two of three models.

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
