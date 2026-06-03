# CLAUDE.md — SA-SVD QA-LoRA Project Context

This file gives Claude Code the full context for the `sa-svd-qa-lora`
repository. Read it before making changes.

---

## 1. What this project is

A clean, public, reproducible reference implementation of **Structure-Aware SVD
(SA-SVD)**, a quantization-group-aware adapter-initialization method for QA-LoRA
fine-tuning of 2-bit quantized language models. It is the public artifact for the
author's M.Sc. thesis (TU Berlin, *Accelerating Quantization-Aware Training of
2-bit Compact Language Models*, supervised by Prof. W. Samek and Prof. K.-R.
Müller, Fraunhofer HHI).

The repository has two audiences and both must stay satisfied:
1. **An engineer skimming the project.** They spend ~90 seconds. The README
   hero chart and the cleanliness of `core.py` are what they judge. Do not let
   the codebase become sprawling or hard to read.
2. **A researcher** who wants to reproduce or build on the result.

It is **not** the research paper. A future paper (a predictive diagnostic for
*when* SA-SVD helps, with a spectral-rank mechanism) is separate work; do not mix
that scope in here unless explicitly asked.

---

## 2. The scientific idea (so you reason correctly about the code)

QA-LoRA fine-tunes a quantized model by constraining the LoRA adapter to be
constant within each quantization group, so it merges into the group's zero-point
with **zero re-quantization loss**. Standard QA-LoRA initializes the adapter
randomly.

At 2-bit, a few high-magnitude **outlier weights** stretch the quantization grid
so far that all other weights round to the same level → **resolution collapse**.
A random adapter cannot recover from this.

**SA-SVD** initializes the adapter with the **principal components of the weight
matrix**, computed to align with quantization groups:
1. **Pool**: reshape `W` to `(out, n_groups, group_size)`, average over the last
   axis → `W_pooled (out, n_groups)`.
2. **Decompose**: SVD of `W_pooled`, keep top-`r` components.
3. **Init adapter**: `B = U_r · sqrt(S_r)`, `A = sqrt(S_r) · Vh_r` (A lives in
   pooled space, shape `(r, n_groups)`).
4. **Residual**: `residual = W - B · expand(A)`, where `expand` is a Kronecker
   repeat of each pooled column `group_size` times. The residual is what gets
   GPTQ-quantized and frozen.

It is a quantization-aware variant of **PiSSA** (Meng et al., 2024).

**Critical correctness invariant:** `B · expand(A) + residual == W` to float
precision. Any change to `core.py` must preserve this. The test
`tests/test_core.py::test_residual_identity` guards it.

**When it helps:** SA-SVD was designed as a *repair* mechanism for
collapse-prone models with high weight dynamic range (e.g. SmolLM2-1.7B).
Three regimes are observed: (1) collapse regime (thesis-era stack), repairs a
broken model (~172 to ~26); (2) gradient-failure regime (Qwen2 on the pinned
stack), random-init training diverges with inf gradients while SA-SVD trains
normally (53 to 28); (3) mild regime (SmolLM2, TinyLlama on the pinned
stack), baseline trains and SA-SVD still gives 15-17%, despite a worse
init-only perplexity, so the benefit is optimization, not a head start. All
measured numbers are single-seed. Never claim SA-SVD is a universal
improvement; the honest framing is load-bearing for credibility.

---

## 3. Headline result (do not misstate these numbers)

2-bit GPTQ, QA-LoRA fine-tune, rank=16, group_size=16, WikiText-2 perplexity
(lower is better). Measured 2026-06-03 on the pinned stack in
`requirements.txt` (stock gptqmodel 6.0.3, upstream peft 0.19.1, single seed);
full conditions and caveats in `results/run-2026-06-03.md`:

| Model          | QA-LoRA (random init) | SA-SVD (ours) | delta |
|----------------|----------------------|---------------|-------|
| SmolLM2-1.7B   | 42.45                | 36.02         | -15%  |
| TinyLlama-1.1B | 34.63                | 28.68         | -17%  |
| Qwen2-1.5B     | 53.15 (did not train) | 27.61        | -48%  |

Same rank, same group size, same training budget per model. **Only the
initialization differs.** The "usable LM" threshold is ~30 PPL. The Qwen2
baseline had inf gradients at every step (two independent runs) and learned
nothing; its 53.15 is effectively the unadapted quantized model, so the Qwen2
row is "trains vs does not train", not "better vs worse".

Context that must not get lost: the thesis reported ~172 (broken) vs ~26
(usable) for SmolLM2 on an older stack (gptqmodel fork
`gapsong/GPTQModel@qzero_unquantized`) where 2-bit quantization caused
resolution collapse. On the modern stock quantizer the baseline does not
collapse, so the dramatic repair effect does not reproduce; the consistent
improvement does. Present the thesis numbers as thesis results, never as
reproducible on the pinned stack. `results/results.json` holds the measured
values keyed by model.

---

## 4. Repository layout

```
sa-svd-qa-lora/
├── README.md               # leads with assets/hero.png, then the idea + quickstart
├── pyproject.toml          # pip install -e .  (core dep: torch only)
├── requirements.txt
├── LICENSE                 # Apache-2.0
├── assets/hero.png         # committed; generated by plot_results.py
├── results/results.json    # method -> {rank, group_size, wikitext_ppl}
├── src/sa_svd/
│   ├── __init__.py         # public API surface
│   ├── core.py             # THE NOVEL PART. sa_svd_init(), reconstruction_error()
│   └── integration.py      # apply to a model + write into a PEFT adapter
├── scripts/
│   ├── demo_fast.py        # ~5 min CPU smoke test; verifies residual identity
│   ├── reproduce.py        # full SmolLM2 train+eval; --method {baseline,sa_svd,both}
│   ├── pipeline.py         # thin GPTQ + QA-LoRA + train + WikiText-eval helpers
│   └── plot_results.py     # results.json -> assets/hero.png
└── tests/test_core.py      # pytest; residual identity + low-rank capture + guards
```

**Design rule:** the SA-SVD idea must stay isolated and readable in
`src/sa_svd/core.py`. GPTQ/PEFT/TRL plumbing belongs in `scripts/pipeline.py`.
Do not leak training/quantization complexity into `core.py`.

---

## 5. Public API (`src/sa_svd`)

```python
sa_svd_init(weight, rank, group_size, *, dtype=torch.float32) -> SASVDResult
# SASVDResult has .lora_A (r, n_groups), .lora_B (out, r), .residual (out, in)

reconstruction_error(weight, result, group_size) -> float
# relative Frobenius error of the rank-r approximation

find_target_linears(model, patterns) -> list[str]
apply_sa_svd_to_base(model, target_names, rank, group_size) -> dict[name -> {lora_A, lora_B}]
write_adapter_weights(peft_model, adapters, adapter_name="default") -> int  # count written
```

Constraints enforced by `sa_svd_init` (keep these):
- `in_features % group_size == 0` else `ValueError("... divisible ...")`
- `rank <= min(out_features, n_groups)` else `ValueError("rank ...")`
- weight must be 2D.

---

## 6. Reproduction pipeline (`scripts/reproduce.py` + `pipeline.py`)

Per method in {baseline, sa_svd}:
1. Load SmolLM2-1.7B FP16.
2. (sa_svd only) `apply_sa_svd_to_base` → swaps each target weight for its
   residual, returns adapter components.
3. 2-bit GPTQ quantize (`gptqmodel`, group_size matches).
4. Attach QA-LoRA adapter (`peft.LoraConfig` with `use_qalora=True`,
   `qalora_group_size=group_size`, `lora_alpha=rank`, dropout 0.0 per thesis
   Table 4.1).
5. (sa_svd only) `write_adapter_weights` overwrites A/B with SA-SVD init.
6. Fine-tune on 10k Alpaca samples (TRL `SFTTrainer`, AdamW, cosine, warmup 0.03,
   bf16, batch 4 × grad-accum 4, 750 steps). LR is per-model via `MODEL_LR` in
   `reproduce.py` (1e-4 default, 3e-5 for Qwen2-1.5B which is unstable at 1e-4);
   an explicit `--lr` overrides it.
7. Evaluate WikiText-2 perplexity via sliding window (stride 512, max_len 1024).

Target modules: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`.

**Why WikiText, not held-out instruction data:** WikiText is never seen in
fine-tuning, so it measures genuine representational repair. Held-out
instruction-tuning perplexity can be "faked" by learning the instruction format
(the thesis "illusion of convergence" finding) and would hide quantization
damage. Keep WikiText as the primary metric. AlpacaEval, if added, is only a
small confirmatory check on 2-3 models — it does not drive the headline.

---

## 7. Conventions and constraints

- **Python 3.10+.** Use `from __future__ import annotations`, type hints, and
  `list[...] | None` style.
- **torch is the only core dependency.** Everything else (transformers, peft,
  gptqmodel, trl, datasets, matplotlib) is an optional `[reproduce]` extra. Do
  not add core deps without a strong reason.
- **SVD in float32 even when the model is bf16** (numerical stability). The
  `dtype` arg in `sa_svd_init` controls this.
- **Determinism:** scripts take `--seed` (default 0). Preserve seeding when
  editing training code.
- **Docstrings:** every public function has a numpy-style docstring explaining
  the *why*, not just the *what*. Match the existing tone — precise, plain, no
  marketing.
- **No em dashes in prose / comments / README** (author preference). Use commas,
  colons, or parentheses.
- **Keep `core.py` small.** It is the first file an engineer reads. If a change
  would bloat it, put the addition elsewhere.
- **The hero chart is committed.** If you change results, regenerate it
  (`python scripts/plot_results.py`) and keep the log-scale + threshold line.

---

## 8. How to verify changes

```bash
pip install -e ".[dev]"
pytest tests/                 # residual identity, low-rank capture, guards
python scripts/demo_fast.py   # end-to-end pipeline on a tiny model, CPU ok
```

`demo_fast.py` uses random Gaussian weights, so its reconstruction error is
expected to be ~0.98 (random matrices have no low-rank structure). That is
**correct and not a bug** — the only thing the smoke test asserts is the
`identity == OK` column. Real LLM weights compress far better, which is where the
repair effect comes from. The full signal is the SmolLM2 172 → 26 result.

For the full reproduction you need a ~24 GB GPU (RTX 4090). `--max-steps 200`
gives a faster, noisier sanity run.

---

## 9. Known follow-ups / open tasks

- Resolved 2026-06-03: multi-model comparison ran (SmolLM2, TinyLlama, Qwen2,
  both methods each, RTX 4090, pinned stack). `results/results.json` is keyed
  by model; chart shows grouped bars. See `results/run-2026-06-03.md`.
- Resolved: convention mismatch against upstream peft 0.19.1 fixed in
  `integration.py::write_adapter_weights` (lora_A scaled by
  `group_size/(scaling*n_groups)`); `core.py` left canonical (Option A).
- Open: the WikiText eval masking line in `pipeline.py::evaluate_wikitext` is
  a no-op (trg_len equals the window), so overlap tokens are scored twice and
  absolute PPLs are pessimistic vs the canonical HF stride recipe. Fair
  across methods. Fixing it changes absolute numbers, so it requires
  re-running all models in one batch; do not fix it piecemeal.
- Open: investigate the Qwen2 inf-gradient failure of random-init QA-LoRA
  (likely bf16 overflow in the backward path); interesting in its own right.
- Open: test the collapse regime by installing the thesis gptqmodel fork
  (`gapsong/GPTQModel@qzero_unquantized`) and re-running; this is what should
  reproduce the ~172 vs ~26 thesis numbers.
- Optional: 2-3 seeds so the percentages get error bars; FP16 reference PPL
  per model for context; larger GPTQ calibration set (>=256 samples of >=256
  tokens, gptqmodel warns about the current 128 short samples).
- Optional: fetch the full Apache-2.0 license text into `LICENSE` (currently a
  header stub).

---

## 10. What NOT to do

- Do not turn this into the research paper (predictor + mechanism). Different repo,
  different scope.
- Do not oversell. SA-SVD is a repair mechanism, not a universal win. Keep the
  "when it helps and when it doesn't" framing intact.
- Do not move GPTQ/PEFT plumbing into `core.py`.
- Do not change the headline numbers to look better than a real run produces.
- Do not add heavyweight dependencies to the core library.
