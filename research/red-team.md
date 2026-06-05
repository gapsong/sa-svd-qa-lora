# Red-team analysis: challenging the 2026-06-03 findings

Branch-only document (`auto-research`). Purpose: attack the repo's public
claims before building on them, rank the attacks by severity, and derive the
cheapest experiments that would falsify or confirm each claim. This is
adversarial by design; section 6 records what survives.

Scope chosen for this pass: (1) the headline 3-model deltas, (2) the causal
"optimization benefit, not head start" story. The Qwen2 "trains vs does not
train" claim and the thesis collapse regime are out of scope here except where
they intersect.

All references are to the repo state at commit f78b1c8.

---

## 1. Claims under attack

**C1 (headline deltas).** README.md:14-20: SmolLM2 -15%, TinyLlama -17%,
Qwen2 -48%, and the sentence "The only thing that changed is how the adapter
was initialized."

**C2 (causal story).** README.md:104-107 and results/run-2026-06-03.md:18-23:
SA-SVD "won despite starting worse: its init-only PPL was ~25.8k vs the
baseline's ~12.2k. The advantage is not 'starts closer to the solution'; the
principal-component structure in the adapter gives optimization a better
direction even from a worse starting point."

---

## 2. Attacks on C1: the headline deltas

### A1. "Only the initialization differs" is strictly false. Severity: HIGH.

For the sa_svd arm, `apply_sa_svd_to_base` replaces every target weight with
its residual `W - B·expand(A)` **before** GPTQ quantization
(reproduce.py:87-92, integration.py:77-78). So the two arms quantize
**different matrices**: the baseline quantizes `W`, sa_svd quantizes the
residual.

This matters because removing the top-r principal components plausibly shrinks
the dynamic range of each quantization group. At 2 bits there are only 4
levels per group; the repo's own collapse narrative (README.md:71-74) says
outliers stretching the grid is exactly what destroys 2-bit quantization. If
the residual has fewer or smaller outliers, the sa_svd arm gets a
**better-quantized frozen base**, independent of what the adapter is
initialized to.

Consequence: the experiment design cannot distinguish "good adapter init"
from "easier quantization target". The README sentence overstates the
controlled-ness of the comparison. Note that this is arguably the method
working as intended (the thesis frames SA-SVD as changing what gets
quantized), but then the public framing should say "initialization plus
quantization target", not "only the initialization".

### A2. Single seed, and only one arm is stochastic at init. Severity: HIGH for the percentages, MEDIUM for the sign.

Every number is one run at seed 0 (run-2026-06-03.md:87). The baseline's
adapter init is seed-dependent; SA-SVD's is deterministic given `W`. So the
baseline column is a single draw from a distribution whose variance is
unknown, compared against a point. The -15% / -17% / -48% percentages have no
error bars and the run note itself says to quote percentages only after 2-3
seeds (run-2026-06-03.md:30-31).

The 3-out-of-3 sign consistency is weaker evidence than it looks: the three
model runs share one pipeline, one seed, one calibration set, one eval
implementation, and one GPU/session. They are three measurements with heavily
correlated systematics, not three independent replications.

### A3. Batch-shape confound on SmolLM2. Severity: LOW.

The SmolLM2 sa_svd run used per-device batch 2 x grad-accum 8; the baseline
used 4 x 4 (run-2026-06-03.md:32-34). Same global batch of 16, but different
data order and gradient-noise profile. Probably footnote-level, as the run
note says, but it means even the SmolLM2 row is not a perfectly controlled
pair. (Qwen2 ran both arms at 2 x 8, so that row is internally consistent.)

### A4. The eval double-counting is only fair on average. Severity: LOW-MEDIUM.

`evaluate_wikitext` advances by stride 512 over 1024-token windows, but the
masking line is a no-op (pipeline.py:122: `target_ids[:, :-trg_len] = -100`
with `trg_len == end - begin`, the full window). Every overlap token is scored
twice, once with under 512 tokens of context. This is already a recorded
open issue, and it is identical code for both arms.

The red-team point is subtler: "identical code" guarantees identical
*treatment*, not identical *bias*. The double-counted tokens are scored at
short context; if the two trained models degrade differently at short context
(plausible, since their frozen bases differ per A1), the pessimism is not
exactly equal across arms. Likely a small effect, but the standard defense
("fair across methods") is an assumption, not a measurement.

### A5. Calibration coupling and undersized calibration set. Severity: LOW.

GPTQ calibrates on 128 WikiText-train snippets averaging ~116 tokens
(pipeline.py:132-143), below gptqmodel's own recommendation, and the eval is
WikiText-test. Both arms share this, but (a) absolute PPLs are flattered by
calibrating on the eval distribution, and (b) calibration noise at this size
could interact with A1: GPTQ error-compensation behaves differently on
matrices with different outlier structure, so an undersized calibration set
may not hurt both arms equally.

### A6. Flag only (out of scope): Qwen2 LR selection.

Qwen2 runs at 3e-5 because it is "unstable at 1e-4" (reproduce.py:44-48). If
that instability was diagnosed on one arm's runs, the LR choice may favor
that arm. Worth one sentence in any future write-up of the Qwen2 result:
state which method exhibited the 1e-4 instability that motivated 3e-5.

---

## 3. Attacks on C2: "optimization benefit, not a head start"

### B1. The decisive confound is A1. Severity: HIGH.

The claim "the benefit is better optimization from the adapter's
principal-component structure" has an unexcluded alternative: **the benefit
is the residual-quantized base**, and the adapter init direction is
irrelevant or minor. The current design changes both at once, so the repo
cannot currently distinguish:

- H_init: principal-component adapter directions make QA-LoRA optimization
  work better.
- H_base: the residual quantizes better at 2 bits, and any adapter
  (including random) would fine-tune that base to a similar PPL.

Until H_base is excluded, C2 is a hypothesis, not a finding.

### B2. The "started worse" evidence is weak in both directions. Severity: MEDIUM.

The supporting fact is init-only PPL ~25.8k (sa_svd) vs ~12.2k (baseline)
(run-2026-06-03.md:20-21). Both numbers are garbage-level; at that magnitude
PPL differences carry almost no information about the loss landscape near the
init or about trainability. "Won despite starting worse" is rhetorically
attractive but not load-bearing; it should not be used as evidence *for* the
optimization story, and a critic should not use it as evidence *against*
SA-SVD either.

### B3. The mechanism claim has zero direct measurements. Severity: MEDIUM.

run-2026-06-03.md:21-23 states the principal-component structure "gives
optimization a better direction" as a finding. The repo contains no gradient
norms, no loss-curve comparison, no landscape probe for the SmolLM2/TinyLlama
runs that would support a directional-quality mechanism. (The Qwen2 grad_norm
observations are the only gradient evidence in the repo, and they belong to a
different regime.) As written, this is an interpretation presented with the
confidence of a measurement.

### B4. Alternative mechanism: scale, not direction. Severity: MEDIUM.

`write_adapter_weights` rescales lora_A by `group_size / (scaling * n_groups)`
to match PEFT's QA-LoRA forward (integration.py:115-126). The resulting
adapter magnitudes are set by the spectrum of `W_pooled`, which is nothing
like PEFT's default init (Kaiming A, zero B: the adapter contributes exactly
zero at step 0). Under AdamW the parameter-space starting magnitude changes
the effective step dynamics. So even if SA-SVD's *directions* were replaced
with random ones at the *same norms*, training might improve. If that
reproduced most of the gain, "principal components" would be the wrong story;
"non-zero, well-scaled init" would be the right one.

---

## 4. Severity ranking

| Rank | Attack | Threatens | Why it leads |
|------|--------|-----------|--------------|
| 1 | A1/B1 residual-base confound | C1 wording, all of C2 | Single root cause, undermines the "controlled comparison" framing and the causal story at once |
| 2 | A2 single seed | C1 percentages | The exact deltas are one draw; sign is probably safe, magnitude is not |
| 3 | B4 scale vs direction | C2 mechanism | Cheap alternative explanation that nobody has excluded |
| 4 | B2/B3 weak mechanism evidence | C2 wording | Fixable by rewording even without new runs |
| 5 | A4 eval bias not provably arm-neutral | C1 margins | Small expected effect, but currently an assumption |
| 6 | A3/A5 batch shape, calibration | C1 margins | Footnote-level, already partly disclosed |

---

## 5. Falsification experiments (derived, not yet run)

Ordered by information per unit cost. E1 and E2 together resolve rank 1; E1
alone is free.

### E1. Quantization-quality probe (free, CPU, run first)

No training. For one model (SmolLM2), compare quantizing `W` vs quantizing the
SA-SVD residual:

- per-group dynamic range and outlier statistics before/after component
  removal;
- 2-bit round-trip error (`||W - Q(W)||` vs `||R - Q(R)||`, relative);
- WikiText PPL of the two quantized bases with **no adapter at all** (for the
  sa_svd base, also score it with the frozen SA-SVD init adapter added back,
  so the reconstruction is comparable).

Decision rule: if the residual base (plus its frozen init adapter) already
sits well below the baseline's quantized base before any training, H_base is
alive and C2 as stated is in serious trouble. If the two bases are comparable,
A1/B1 lose most of their force.

### E2. Residual base + random adapter ablation (decisive, ~1-2 h GPU per run)

A third arm: run `apply_sa_svd_to_base` (so the quantized base is the
residual), then **skip** `write_adapter_weights` and train from PEFT's default
random/zero init. One run on SmolLM2 first.

- If this arm lands near sa_svd's 36.0: the benefit is the base. C2 is
  falsified; reword README and run notes.
- If it lands near the baseline's 42.5 (or worse): the init direction matters.
  C2 survives its strongest challenge and can be stated with much more
  confidence.

This is the single most informative training run available to the project.

### E3. Seeds 1 and 2 for SmolLM2, both arms (4 runs, ~1-2 h each)

Gives a crude spread for the -15% and tests whether the baseline's seed-0 draw
was unlucky. Decision rule: if baseline seed-variance is comparable to the
6.4-PPL gap, retire the percentage claims until more seeds exist; if the gap
holds across 3 seeds, the README claim gets real support.

### E4. Norm-matched random init (1-2 runs)

Replace each SA-SVD `A`/`B` with random matrices scaled to the same Frobenius
norms (per layer), on the residual base. Distinguishes B4's scale story from
the direction story, conditional on E2 showing the init matters at all. Run
only after E2.

### E5. Fixed eval masking (deferred)

Correcting pipeline.py:122 changes absolute PPLs and requires re-running every
model in one batch (trained models were not saved). Already an open task in
CLAUDE.md section 9; do not fix piecemeal. Note for later: when it is fixed,
E1-E4 numbers must be produced under the fixed eval too.

Suggested order: E1 now (free), E2 next GPU session, then E3, then E4.

---

## 6. What survives even if every attack lands

- The residual identity `B·expand(A) + residual == W` is a mathematical
  guarantee, tested in tests/test_core.py. No experiment threatens it.
- The direction of the effect (sa_svd arm beat the baseline arm in all three
  pairs, under shared conditions) survives A2-A5; only its magnitude and its
  attribution are in question.
- The Qwen2 observation (random-init arm had inf grad_norm at every logged
  step in two runs; sa_svd arm trained normally) is a reproducible-looking
  qualitative difference, though A1 applies to its attribution too: the
  stable arm also had a different quantized base.
- The repo's own disclosures (single seed, eval no-op, calibration size,
  batch-shape note) are accurate and already public in
  results/run-2026-06-03.md. The red-team findings sharpen the framing; they
  do not reveal hidden misconduct.

The two sentences most in need of rewording, independent of any new runs:

1. README.md:20 "The only thing that changed is how the adapter was
   initialized." Should acknowledge the quantized base also differs.
2. run-2026-06-03.md:21-23 "the principal-component structure in the adapter
   gives optimization a better direction". Should be marked as a hypothesis
   pending E2.

Do not edit those files from this branch until E1/E2 results exist; the
rewording should cite evidence, not just suspicion.

---

## 7. E1 results (2026-06-04): the residual-base confound is structurally dead

Ran `research/e1_quant_probe.py` on SmolLM2-1.7B, all 168 target layers,
rank 16, group_size 16, 2-bit asymmetric per-group RTN as the GPTQ proxy.
Raw numbers in `research/e1_results.json`.

### Measured

| metric (numel-weighted, all layers) | W (baseline) | R (sa_svd) |
|--------------------------------------|--------------|------------|
| reconstruction error of W, rel. Frobenius | 0.3286 | 0.3286 |
| mean distinct levels used per group (max 4) | 3.991 | 3.991 |
| mean modal-level occupancy | 0.460 | 0.460 |

The ratio is not approximately 1, it is exactly 1: per-layer spot checks show
the error difference is bitwise 0.0. Meanwhile the subtracted component is not
small: `||B·expand(A)||_F / ||W||_F` ranges from ~0.18 (layer-0 attention) to
~0.07 (late down_proj), so this is not "residual equals W".

### Why: a shift-equivariance theorem

`expand(A)` repeats each pooled column `group_size` times, so `B·expand(A)`
is **constant within every quantization group**. For any per-group asymmetric
quantizer whose grid is anchored to the group (min/max or MSE-optimal scale
with a continuous zero-point), subtracting a constant c from a group shifts
min and max equally: range, scale, and codes are unchanged and
`Q(g - c) = Q(g) - c` exactly. Therefore:

- The per-group dynamic range of R **equals** that of W. A1's premise
  (component removal shrinks group range and tames outliers) is not just
  unsupported, it is mathematically impossible. This is the same structural
  property that lets QA-LoRA merge the adapter into the zero-point.
- Under the idealized quantizer, `Q(R) + B·expand(A) = Q(W)`: the sa_svd
  arm's reconstruction at init is *identical* to the baseline's quantized
  model. There is no "easier quantization target".

### Caveat: the real stack is not exactly equivariant

Stock gptqmodel quantizes the zero-points themselves (the thesis fork is
named `qzero_unquantized` precisely because it removes this), and GPTQ's
calibration-driven error compensation acts on top. The measured init-only
PPLs (baseline ~12.2k vs sa_svd ~25.8k, run-2026-06-03.md:20-21) prove the
deviation is real at PPL level. Direction of that data point: the residual
base came out **worse** at init, not better. So the only known deviation from
equivariance points *against* H_base, not for it.

**Forward note (2026-06-05, journal v17): the theorem's premise does not
match the pipeline.** The pipeline's QuantizeConfig defaults to sym=True
(verified on the installed gptqmodel 6.0.3), and a symmetric grid (scale
anchored to max|g|, zero fixed at 2) is NOT shift-equivariant, so the
equivariance proof above covers a grid the pipeline does not use. The
empirical conclusion survives anyway: re-probing all three models with the
real sym grid plus a rotation arm (e1_quant_probe.py --sym --rotate,
research/v17_*.json) shows the residual swap moves relative error by only
~0.0025 and levels-used by ~0.005, identically across models. H_base stays
dead in practice, but cite the theorem as asym-only and this measurement
as the sym-grid evidence.

### Verdict and updated ranking

- **A1/B1 downgraded HIGH to LOW.** The textual point stands (the matrix fed
  to GPTQ does differ, and README.md:20 could still note the zero-point
  caveat in one clause), but the substantive confound is dead: to first order
  the quantized base is the same, so the measured benefit must come from the
  adapter side. H_base has no mechanism left and one opposing data point.
- C2's surviving threats are now **A2 (single seed)** and **B4 (scale vs
  direction)**, in that order. Updated ranking: (1) A2, (2) B4, (3) B2/B3
  wording, (4) A4, (5) A3/A5.
- Side observation: no collapse signature on this stack's proxy (groups use
  ~3.99 of 4 levels; modal occupancy 0.46). Consistent with the run note's
  "no collapse on the modern stack" finding, now also visible in weight space
  without any training.
- Side observation for the README narrative: the adapter component carries
  7-18% of W's Frobenius energy. "Carries the dominant structure of the
  original weights" is true in the pooled space SA-SVD decomposes, but in
  full weight space the init is a modest correction, not the bulk of W.

### Registered prediction for E2

Under stock gptqmodel, residual-base + PEFT-default random adapter (E2)
should land near the baseline's 42.5, because the base is (to first order)
the same quantized model and the random init contributes zero at step 0. If
it instead lands near sa_svd's 36.0, the zero-point quantization path is
doing real work and this section's first-order analysis is wrong in an
interesting way. Either outcome is informative; the prediction is logged
before the run.

---

## 8. E2 results (2026-06-04): prediction falsified, H_base dead, C2 sharpened

Ran `scripts/reproduce.py --method residual_random` (added on this branch,
commit d35af40): SA-SVD residual swap on the base, PEFT default adapter init
(A Kaiming, B zero), otherwise the exact baseline conditions (SmolLM2-1.7B,
lr 1e-4, seed 0, 750 steps, batch 4x4). Raw record in
`research/e2_results.json`; log in /tmp/e2_residual_random.log.

| arm | base fed to GPTQ | adapter init | WikiText PPL |
|-----|------------------|--------------|--------------|
| baseline (2026-06-03) | W | random | 42.45 |
| sa_svd (2026-06-03) | R | SA-SVD A/B | 36.02 |
| residual_random (E2) | R | random | **2524.88, did not train** |

The E2 arm showed grad_norm inf at every logged step, loss flat at 16.5-16.6,
mean token accuracy exactly 0 for all 750 steps. With inf gradient norm,
TRL's gradient clipping scales updates by 1/inf = 0, so the final model is
effectively the init model. 2524.88 is the perplexity of the bare
residual-quantized base, not of anything trained.

### The registered prediction was wrong, and the error is instructive

Section 7 predicted ~42.5 "because the base is (to first order) the same
quantized model". That conflated quantization **error** with the base
**function**. The theorem gives `Q(R) = Q(W) - B*expand(A)`: identical error,
but a different function. The baseline starts at the function Q(W); the E2
arm starts at Q(W) minus the principal pooled component (7-18% of weight
energy per layer), which is a catastrophically broken model, and from there
random-init QA-LoRA cannot train at all.

### What the triple actually establishes

1. **H_base is conclusively dead, from both sides.** E1: the residual earns
   no quantization-error advantage (provably zero under the proxy). E2: the
   residual base on its own is not a better substrate but a vastly worse one.
   Nothing of SA-SVD's benefit comes from "an easier quantization target".
2. **C2 survives and is sharpened.** By the shift theorem, sa_svd's init
   function is Q(R) + B*expand(A) = Q(W) up to zero-point rounding, i.e.
   approximately the same function the baseline starts at (its B = 0). Same
   starting function, different parameter-space coordinates, 36.02 vs 42.45
   after identical budgets. The benefit is therefore a parameterization and
   optimization effect, which is what C2 claimed. What remains open is only
   the mechanism inside that effect: principal directions (H_init) vs merely
   a large, structured, nonzero starting point (B4's scale story).
3. **The Qwen2 open question gets a data point.** The inf-grad signature
   (grad_norm inf every step, zero learning, loss pinned) was reproduced on
   SmolLM2 by handing it a sufficiently broken init function. This supports
   the hypothesis that random-init QA-LoRA fails by gradient overflow
   whenever the effective model at step 0 is bad enough, and that Qwen2's
   2-bit Q(W) sits past that threshold while SmolLM2's and TinyLlama's do
   not. The failure tracks init-function badness, not model identity.
4. **Oddity flagged, not resolved:** bare Q(R) evals at ~2.5k PPL here while
   the previously quoted init-only PPLs were higher (Q(W) + zero adapter
   ~12.2k, Q(R) + SA-SVD adapter ~25.8k, measured in an earlier session).
   In the garbage regime PPL ordering carries little signal, but if init
   PPLs are ever quoted again they should be re-measured under one protocol
   in one session.

### E4 redesigned: now the decisive mechanism experiment

The shift theorem enables a clean control that section 5's E4 lacked: ANY
group-constant low-rank component can be moved from the base into the
adapter with (to first order) no change to the init function. Concretely:
draw random A' (r, n_groups) and B' (out, r), rescale so ||B'A'||_F matches
the SA-SVD component per layer, set the base to W - B'*expand(A') and the
adapter to (A', B'). Init function is again approximately Q(W), so it should
train. Then:

- lands near 36 (sa_svd): the benefit is having a large structured nonzero
  init at all; "principal components" is the wrong story (B4 wins).
- lands near 42.5 (baseline): the principal directions themselves carry the
  benefit; H_init wins and C2 can be stated at full strength.

Updated priority: E4 (mechanism) and E3 (seeds, for the percentages) are the
two remaining runs that matter. Registered prediction for E4, before any
run: no confident prediction; genuinely uncertain between the two outcomes,
which is what makes it worth running.

---

## 9. E4 results (2026-06-04): directions do not matter, the scale story wins

Ran `scripts/reproduce.py --method random_matched` (commit 7b9ef9d): SA-SVD's
per-layer singular-value spectrum, random orthonormal U'/V'h directions,
moved into the base exactly like SA-SVD (verified: residual identity to float
precision, exact match of ||B||, ||A||, ||BA||, group-constant component).
Conditions: SmolLM2-1.7B, lr 1e-4, seed 0, 750 steps, per-device batch 2 x
grad-accum 8 (GPU shared with a simulator; same batch shape as the sa_svd
arm, first attempt at 4x4 OOMed at step ~552 when the simulator started).
Record in `research/e4_results.json`; log /tmp/e4_random_matched_try2.log.

| arm | base | adapter init | WikiText PPL |
|-----|------|--------------|--------------|
| baseline | W | PEFT default (B=0) | 42.45 |
| sa_svd | R | SVD directions, SVD spectrum | 36.02 |
| residual_random (E2) | R | PEFT default (B=0) | 2524.88, did not train |
| random_matched (E4) | R' | random directions, SVD spectrum | **35.65** |

random_matched trained indistinguishably from sa_svd (finite grads 4-7, loss
2.0 to 1.9, accuracy 56-58%) and finished at 35.65, marginally *better* than
sa_svd's 36.02. Single seed, so 35.65 vs 36.02 is a tie; both are clearly
separated from 42.45.

### Verdict: B4 wins, H_init falsified (one model, one seed)

The principal-component directions contribute nothing measurable. An adapter
with the same singular-value spectrum but random orthonormal directions
reproduces the full SA-SVD benefit. The run-note sentence "the
principal-component structure in the adapter gives optimization a better
direction" is falsified as a mechanism claim on this evidence.

What the active ingredients actually are, per the four-arm table:
1. A **nonzero, spectrum-scaled adapter init** (sa_svd and random_matched
   have it, baseline does not): worth ~16% PPL.
2. The **shift property** that keeps the init function at ~Q(W) (residual
   swap and adapter component cancel): without restoring the component, the
   model does not even train (E2).
3. The **SVD directions**: worth ~nothing (36.02 vs 35.65).

Note the spectrum itself still comes from the pooled SVD, so SA-SVD's
decomposition is still *used*, just not its directions. Whether the spectrum
shape matters either (vs any reasonable scale) is untested; see E6 below.

### Consequences for the public claims

- C2 must be reworded. Supported version: "the benefit is an optimization
  effect of a large, well-scaled, group-constant adapter initialization, not
  a head start in function space". Unsupported version (current README and
  run note): "the principal-component structure gives optimization a better
  direction".
- The method keeps its practical value: SA-SVD is a deterministic, principled
  construction of such an init, and computing the spectrum requires the SVD
  anyway, so nothing simpler is cheaper. But the *story* of why it works must
  change from "principal components" to "init scale and parameterization".
- The thesis collapse-regime claims are untouched by this; in the collapse
  regime the function-space reconstruction may genuinely matter. This verdict
  applies to the mild regime on the pinned stack.

### New follow-ups surfaced

- **E6 (cheap, decisive for the remaining mechanism):** flat spectrum
  control: random orthonormal directions with all component scales set to
  the same value (product norm ||B'A'||_F matched to SA-SVD per layer; a
  flat spectrum cannot match product and factor norms simultaneously, and
  the product norm is what the base subtraction and the merge see). If it
  also lands ~36, even the spectrum shape is irrelevant and the story is
  purely "init magnitude". If it degrades, the spectrum carries real
  information. Wired as `--method random_flat` (flat_spectrum=True in
  research/e4_random_matched.py). Registered prediction, before the run:
  ~36, i.e. spectrum shape also does not matter, medium-low confidence.
  Reasoning: if the mechanism is optimizer leverage from factor magnitude,
  the per-component allocation should be second-order at rank 16, where the
  SA-SVD spectrum is fairly flat to begin with on these layers.
- **E7:** run random_matched on Qwen2-1.5B. Prediction (registered now,
  medium confidence): it rescues the inf-grad failure just like sa_svd did,
  because the rescue is about init parameterization, not directions.
- E3 (seeds) unchanged and still needed before quoting any percentage.

---

## 10. E6 results (2026-06-04): the spectrum matters, the directions do not

Ran `scripts/reproduce.py --method random_flat` (commit 9d2e2ea): random
orthonormal directions AND a flat spectrum (every component scale equal,
product norm ||B'A'||_F matched to SA-SVD per layer). Same conditions as E4
(2x8, seed 0, lr 1e-4). Record in `research/e6_results.json`; log
/tmp/e6_random_flat.log. Result: **38.45**. The registered prediction (~36)
was wrong; that is two falsified predictions (E2, E6) out of three registered,
which says the experiments were informative rather than confirmatory.

### The five-arm table

| arm | directions | spectrum | WikiText PPL |
|-----|------------|----------|--------------|
| baseline | none (B=0) | none | 42.45 |
| random_flat (E6) | random | flat, norm-matched | 38.45 |
| random_matched (E4) | random | SVD values | 35.65 |
| sa_svd | SVD vectors | SVD values | 36.02 |
| residual_random (E2) | none (B=0) | none, base broken | 2524.88, no train |

### Mechanism decomposition (single seed, SmolLM2, pinned stack)

- Having any well-scaled group-constant nonzero init: 42.45 to 38.45,
  roughly 60% of the total benefit.
- Allocating that scale per the pooled SVD spectrum (a few large components
  instead of equal shares): 38.45 to ~35.8, the remaining roughly 40%.
- The singular vectors themselves: nothing (36.02 vs 35.65).

Slogan for the corrected story: **the spectrum, not the subspace**. SA-SVD's
SVD is doing real work, through its singular values, not its singular
vectors.

A detail that kills the simplest scale story: with the product norm matched,
the flat spectrum's *factor* norms are larger than SA-SVD's
(Cauchy-Schwarz: sum(s_k) <= r*c), yet E6 did worse. So "bigger factors,
more leverage" is not monotonically true; the concentration of scale into a
few dominant components is what helps. Consistent with an optimizer-dynamics
account (a few high-gain coordinates) but not proven by it.

### Caveats and consequences

- The E4-vs-E6 gap (35.65 vs 38.45, ~2.8 PPL) is the load-bearing new
  number and it is single-seed. E3 (seeds) is now needed not just for the
  headline percentages but to confirm this gap survives noise.
- C2's supported rewording is refined once more: "the benefit is an
  optimization effect of a group-constant adapter init whose magnitude is
  allocated by the pooled weight spectrum; the singular vectors are
  replaceable, the singular values are not (on this evidence)".
- For the method: SA-SVD is partially vindicated relative to section 9's
  verdict. Its decomposition is not decoration; you need the spectrum, and
  computing it IS the SVD. The honest summary is that SA-SVD computes
  exactly the right thing and uses half of it (values) for the part that
  matters and half (vectors) for a part that does not hurt but does not
  help.

---

## 11. E3 results (2026-06-04/05): seeds rewrite every single-seed verdict

Ran `research/run_e3_seeds.sh`: baseline, sa_svd, random_matched,
random_flat at seeds 1 and 2 (seed 0 from earlier sections), all at 2x8,
SmolLM2, zero failures. Records: `research/e3_seed{1,2}_results.json`.

| arm | seed 0 | seed 1 | seed 2 | mean | spread |
|-----|--------|--------|--------|------|--------|
| baseline | 42.45* | 42.44 | 38.24 | 41.04 | 4.21 |
| sa_svd | 36.02 | 35.61 | 35.99 | 35.87 | 0.41 |
| random_matched | 35.65 | 38.31 | 36.53 | 36.83 | 2.66 |
| random_flat | 38.45 | 39.12 | 40.11 | 39.23 | 1.66 |

*seed-0 baseline ran 4x4; all other 11 runs 2x8. Seed-1 baseline (2x8)
reproduced it to 0.01, so batch shape is negligible.

### Verdicts, in order of how much they changed

1. **A2 fully vindicated; retire the point percentages.** The baseline's
   seed spread (4.21) is two-thirds of the original headline gap (6.43).
   Its best draw (38.24) lands inside the scaled-init cluster. The
   section-5 decision rule fires: stop quoting "-15%" as a number. The
   supportable statements: sa_svd beats baseline in **9/9 seed pairings**,
   minimum margin 2.2 PPL (sa_svd's worst 36.02 vs baseline's best 38.24),
   mean improvement 41.04 to 35.87, about -13%.
2. **Section 9's "directions are worth nothing" is OVERTURNED**, a
   casualty of the same single-seed fallacy it was investigating. With
   three direction draws, random_matched spans 35.65-38.31 (spread 2.66)
   while sa_svd sits at 35.87 +/- 0.2. The corrected claim: **SVD
   directions are not uniquely necessary (the best random draw matched
   them) but they deterministically achieve what random directions achieve
   only at the top of their lottery.** Worth ~1 PPL in mean and a 6x spread
   reduction relative to random directions.
3. **The spectrum claim survives seeds, and is the cleanest finding.**
   random_matched and random_flat share identical directions within each
   seed (same generator), so the within-seed difference isolates the
   spectrum: +2.80, +0.81, +3.58 PPL across seeds 0/1/2. Three of three
   positive, mean ~2.4. Concentrated SVD-shaped scale beats flat scale at
   matched product norm, under direction-controlled pairing.
4. **Section 10's "any scaled init buys ~60% of the benefit" is
   downgraded.** random_flat's mean (39.23) vs baseline's mean (41.04) is
   a 1.8 margin sitting inside the baseline's own 4.2 spread, and the
   baseline's best seed beats random_flat's mean. Flat scale alone is the
   weakest and least robust ingredient.
5. **New headline property: variance collapse.** sa_svd's outcome spread
   is 0.41 vs the baseline's 4.21, a 10x reduction, with a fully
   deterministic init. For a QAT pipeline, "reliably 35.9" may be worth
   more than the mean improvement itself. This property was invisible in
   every single-seed comparison.

### Corrected story for the repo (what README/run notes should say)

SA-SVD initialization, on the pinned stack, mild regime, SmolLM2, 3 seeds:
improves mean WikiText PPL by about 13% (41.0 to 35.9), wins every observed
pairing with a worst-case margin of 2.2 PPL, and reduces outcome spread by
about 10x. Mechanistically, the SVD's singular values (the spectrum
allocation) carry a consistent paired gain of roughly 1-3.5 PPL, and the
singular vectors buy reliability rather than a unique optimum. The previous
single-seed narratives (sections 9-10) understated direction value and
overstated scale value; both corrections came from seed replication, which
is the methodological lesson of this whole document.

### Epistemic scorecard for the exercise

Every section that drew a mechanism conclusion from one seed got revised by
E3 (sections 9 and 10), exactly as attack A2 predicted for the original
README. Registered predictions: 3 made, 2 falsified (E2, E6), 1 partially
right for the wrong reasons. The experiments were informative throughout;
the priors were not.

### Remaining open items

- E7 (random_matched on Qwen2): does the inf-grad rescue need the
  directions, the spectrum, or any scaled init? Now also interesting
  whether the rescue is seed-robust.
- TinyLlama/Qwen2 seeds for the multi-model table (the public table is
  still 1 seed for those rows).
- The README/run-note rewording can now proceed with E1-E6 evidence; the
  concrete supportable sentences are above.

---

## 12. E7 result (2026-06-05, truncated): the Qwen2 rescue needs no SVD directions

Ran `scripts/reproduce.py --method random_matched --model-id Qwen/Qwen2-1.5B`
(lr 3e-5 auto, 2x8, seed 0). The process was externally SIGTERMed at step
588/750 (78%); training had been healthy the entire time: grad_norm finite
throughout (9-17, the sa_svd run's reported band), loss 2.1 to 1.97, token
accuracy ~55%, zero inf-grad steps. Log: /tmp/e7_qwen2_random_matched.log.
No checkpoints exist (save_strategy="no"), so there is no final PPL.

Verdict: the registered prediction (section 9) is **confirmed on the
trainability question**. The Qwen2 random-init baseline fails with inf
gradients at every step from step 1; the random-directions arm trained
cleanly for 588 steps. The gradient-overflow rescue is a property of the
scaled, group-constant init parameterization, not of the SVD directions.
Combined with E2 (a broken-enough init function reproduces the inf-grad
signature on SmolLM2), the unified picture: random-init QA-LoRA fails when
the effective step-0 model is past a badness threshold, and any
function-preserving scaled init steps around the failure.

Not established, deliberately skipped for now (user call, GPU time): the
final PPL of this arm (expected near sa_svd's 27.61), and whether its
*quality* shows the same direction-lottery variance as SmolLM2's E3. Rerun
is one command if needed:
`python scripts/reproduce.py --method random_matched --model-id Qwen/Qwen2-1.5B --batch-size 2 --grad-accum 8 --results-path research/e7_results.json`
