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
