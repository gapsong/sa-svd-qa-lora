# SA-SVD: Structure-Aware SVD Initialization for 2-bit QA-LoRA

**One sentence:** initializing the QA-LoRA adapter with a
**quantization-group-aware SVD of the weights**, instead of random noise,
improves 2-bit fine-tuned language models at the same rank, group size, and
training budget.

![SA-SVD vs baseline on 2-bit SmolLM2-1.7B](assets/hero.png)

Three models quantized to 2-bit with GPTQ, then fine-tuned with QA-LoRA under
identical budgets (rank 16, group 16, 750 steps, single seed). WikiText-2
perplexity, lower is better:

| Model          | QA-LoRA (random init) | SA-SVD (ours) | delta |
|----------------|----------------------|---------------|-------|
| SmolLM2-1.7B   | 42.5                 | 36.0          | -15%  |
| TinyLlama-1.1B | 34.6                 | 28.7          | -17%  |
| Qwen2-1.5B     | 53.2*                | 27.6          | -48%  |

The only thing that changed is how the adapter was initialized.

*The Qwen2 baseline did not train at all: its gradients overflowed to inf at
every step (two independent runs), so 53.2 is effectively the unadapted
quantized model. With SA-SVD initialization the identical setup trains
normally. On Qwen2 the difference is not "better", it is "trains vs does not
train". Full conditions and caveats: `results/run-2026-06-03.md`.

For context, the thesis observed an even larger effect on its older
quantization stack, where 2-bit quantization caused resolution collapse: the
random baseline stayed broken at PPL ~172 and SA-SVD repaired it to ~26.

This repository accompanies my M.Sc. thesis at TU Berlin (*Accelerating
Quantization-Aware Training of 2-bit Compact Language Models*, supervised by
Prof. W. Samek and Prof. K.-R. Müller, Fraunhofer HHI). It builds on the
official QA-LoRA implementation I contributed to Hugging Face PEFT
([#2571](https://github.com/huggingface/peft/pull/2571),
[#2664](https://github.com/huggingface/peft/pull/2664)).

---

## Quickstart

```bash
git clone https://github.com/gapsong/sa-svd-qa-lora
cd sa-svd-qa-lora
pip install -e .                   # core library (needs torch only)

# 1. Verify your setup runs the SA-SVD pipeline end to end (~5 min, CPU ok).
python scripts/demo_fast.py

# 2. Reproduce the headline SmolLM2 result (~1-2 h on a 24 GB GPU).
#    Use the pinned stack: the loose version floors can resolve to
#    mutually incompatible packages.
pip install -r requirements.txt
python scripts/reproduce.py --method both
python scripts/plot_results.py        # regenerates assets/hero.png
```

You do **not** need to run anything to see the result; the chart above is
checked in. The commands are for reproducing it yourself.

---

## The idea

QA-LoRA fine-tunes a quantized model by forcing the LoRA adapter to be constant
within each quantization group, which lets the adapter merge into the group's
zero-point with no re-quantization loss. The standard recipe initializes that
adapter **randomly**.

At 2-bit, a handful of high-magnitude outlier weights can stretch the
quantization grid so far that every other weight rounds to the same level. The
model suffers *resolution collapse*, and a random adapter has no path back.

**SA-SVD** initializes the adapter with the **principal components of the
weights**, computed so they align with the quantization groups:

1. **Pool** the weight matrix along each quantization group (average within
   groups).
2. **Decompose** the pooled matrix with SVD; keep the top-`r` singular vectors.
3. **Initialize** the adapter `B, A` from those vectors.
4. **Quantize the residual** `W - B·expand(A)` as the frozen base.

The adapter now carries the dominant structure of the original weights, so
fine-tuning starts from a healthy point instead of from noise. It is a
quantization-aware variant of [PiSSA](https://arxiv.org/abs/2404.02948).

The whole method is ~80 lines: [`src/sa_svd/core.py`](src/sa_svd/core.py).

---

## When it helps (and when it doesn't)

SA-SVD was designed as a **repair mechanism** for models that collapse under
2-bit quantization. Three failure regimes have been observed, and SA-SVD
helped in all of them, by different amounts:

- **Collapse regime** (thesis, older quantization stack): the 2-bit baseline is
  broken (PPL ~172) and SA-SVD repairs it to a usable ~26.
- **Gradient-failure regime** (Qwen2-1.5B on this repo's pinned stack): the
  quantized model is healthy but random-init QA-LoRA training diverges
  (inf gradients, zero learning). SA-SVD initialization makes the same setup
  train, 53.2 to 27.6.
- **Mild regime** (SmolLM2, TinyLlama on the pinned stack): the baseline
  trains fine and SA-SVD still gives a consistent 15-17% improvement at zero
  extra cost. Notably it can win despite a *worse* pre-training perplexity,
  so the benefit comes from better optimization, not a head start.

All numbers are single-seed; treat the percentages as indicative. Mapping the
regime boundary across models and quantizers is ongoing work.

## Configuration

All runs share the thesis configuration (Section 4.3.4 / Table 4.1):

| Hyperparameter | Value |
|----------------|-------|
| Quantization   | 2-bit GPTQ, group_size 16, 128 WikiText calibration samples |
| Adapter        | QA-LoRA, rank 16, alpha = rank, dropout 0.0 |
| Target modules | q, k, v, o, gate, up, down projections |
| Optimizer      | AdamW, cosine schedule, warmup ratio 0.03 |
| Learning rate  | 1e-4 (3e-5 for Qwen2-1.5B, which is unstable at 1e-4) |
| Batch          | global 16 (per-device 4 x grad-accum 4) |
| Steps / data   | 750 steps on a 10k-sample Alpaca subset (~1.2 epochs) |
| Precision      | bf16 compute, float32 SVD |
| Evaluation     | WikiText-2 test, sliding window (window 1024, stride 512) |
| Seed           | 0 |

---

## How the pieces fit

| File | Role |
|------|------|
| `src/sa_svd/core.py` | The SA-SVD decomposition. The novel part. |
| `src/sa_svd/integration.py` | Apply SA-SVD to a model and write it into a PEFT adapter. |
| `scripts/demo_fast.py` | Fast environment check, no large download. |
| `scripts/reproduce.py` | Full SmolLM2 reproduction (train + eval). |
| `scripts/pipeline.py` | Thin GPTQ + QA-LoRA + train + WikiText-eval helpers. |
| `scripts/plot_results.py` | Renders the hero chart from `results/results.json`. |
| `tests/test_core.py` | Verifies the residual identity and low-rank capture. |

Run the tests with `pytest tests/`.

---

## Citation

```bibtex
@mastersthesis{tonthat2025sasvd,
  title  = {Accelerating Quantization-Aware Training of 2-bit Compact Language Models},
  author = {Ton-That, Khiem},
  school = {Technische Universit\"at Berlin},
  year   = {2025}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
