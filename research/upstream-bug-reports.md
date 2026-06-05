# Upstream bug report drafts (2026-06-05, NOT yet filed)

Status: drafts for review. Nothing has been submitted. Both bugs verified
on the pinned stack (gptqmodel 6.0.3, peft 0.19.1) AND on both projects'
current `main` (fetched 2026-06-05: peft main commit baa6a04, variants.py
last changed 2026-05-21; gptqmodel main loader.py lines 685-690).

The discovery chain is in autoresearch/journal.md v11-v13. The 2026-06-05
stage probes replaced the earlier "meta-device materialization" hypothesis
with a fully proven mechanism: a global torch.nn.init monkeypatch leak.

GitHub search found no existing issue for either bug (queries: "kaiming",
"nn.init" on ModelCloud/GPTQModel; "qalora" on huggingface/peft).

---

## Issue 1 (root cause): GPTQModel - `from_pretrained` permanently no-ops `torch.nn.init` for the whole process

Repo: ModelCloud/GPTQModel
Affected: 6.0.3 (pinned) and current main (gptqmodel/models/loader.py:685-690)

### Title

`ModelLoader.from_pretrained` globally replaces `torch.nn.init.kaiming_uniform_/uniform_/normal_` with a no-op and never restores them

### Body draft

`ModelLoader.from_pretrained` does this unconditionally (loader.py:685-690 on
main, 305-310 in 6.0.3):

```python
def skip(*args, **kwargs):
    pass

torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip
```

The originals are never restored. From that point on, EVERY
`torch.nn.Linear` (or any module whose `reset_parameters` uses these
functions) created anywhere in the process is born with uninitialized
memory instead of its intended init. The same file already imports and
uses `suspend_hf_weight_init()` (utils/hf.py), a context manager that does
the same suppression correctly with try/finally restore; `from_pretrained`
just does not use it.

A second-order effect makes the leak permanent even across later guarded
sections: `from_quantized` wraps its work in `suspend_hf_weight_init()`,
which captures the CURRENT function as "original" at entry. If
`from_pretrained` ran earlier in the process, the captured "original" is
already the `skip` stub, so the restore at exit reinstalls the stub.

Minimal repro (no GPU work needed beyond the load):

```python
import torch
from gptqmodel import GPTQModel, QuantizeConfig

print(torch.nn.init.kaiming_uniform_)
# <function kaiming_uniform_ at 0x...>

GPTQModel.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                          QuantizeConfig(bits=4, group_size=128))
print(torch.nn.init.kaiming_uniform_)
# <function ModelLoader.from_pretrained.<locals>.skip at 0x...>

lin = torch.nn.Linear(36, 4, bias=False)
print(lin.weight.std())
# ~0.0001 or 0.0 (uninitialized memory); kaiming would give ~0.096
```

Real-world impact we measured: in a GPTQ + PEFT QA-LoRA fine-tuning
pipeline (quantize with gptqmodel, then `get_peft_model`), the LoRA
adapters created after `from_pretrained` contain raw uninitialized memory.
On a freshly used GPU these pages are zeros, so lora_A = lora_B = 0, which
is an exact saddle point of the LoRA parameterization: the adapter trains
to nothing, silently (750 fine-tuning steps, loss never moved, final
WikiText-2 PPL 12230 = the unadapted model). On a busy GPU the pages are
arbitrary junk (we probed values up to +inf), giving unseeded,
non-reproducible behavior, including inf gradients.

Suggested fix: wrap the body of `from_pretrained` in the existing
`suspend_hf_weight_init()` context manager (and make that CM re-entrant or
idempotent so nested use cannot capture a stub as the original), or
restore the three functions in a `finally:` before returning.

We can submit a PR if useful.

---

## Issue 2 (defense in depth): PEFT - QA-LoRA variant replaces lora_A without re-initializing it

Repo: huggingface/peft
Affected: 0.19.1 (pinned) and current main (src/peft/tuners/lora/variants.py,
`QALoraLinearVariant.init`, lines 489-496 on main)

### Title

QA-LoRA variant init replaces `lora_A` with a fresh `nn.Linear` after `reset_lora_parameters` ran, discarding the configured initialization

### Body draft

Order of operations in `LoraLayer.update_layer` (layer.py, main):

1. `lora_A`/`lora_B` are created and `reset_lora_parameters(adapter_name,
   init_lora_weights)` applies the configured init (line ~246).
2. `self.lora_variant[adapter_name].init(...)` runs (line ~251).
3. For `use_qalora=True`, `QALoraLinearVariant.init` REPLACES
   `module.lora_A[adapter_name]` with a brand-new `nn.Linear` of pooled
   shape `(in_features // qalora_group_size, r)` (variants.py:489-496) and
   never re-initializes it.

Consequences:

- Whatever `init_lora_weights` produced in step 1 is silently discarded.
  The replacement relies entirely on `nn.Linear`'s constructor default.
  Non-default settings (`gaussian`, `init_lora_weights=False`, etc.) are
  ignored for QA-LoRA without warning.
- The constructor default is fragile: under any environment where
  `torch.nn.init` is suppressed (accelerate meta-device contexts,
  `low_cpu_mem_usage` paths, or the GPTQModel bug where
  `from_pretrained` leaks a global no-op of `kaiming_uniform_`, see
  ModelCloud/GPTQModel issue <link>), the replacement tensor is raw
  uninitialized memory. Combined with the intended `lora_B = 0`, zeros in
  lora_A give A = B = 0, an exact saddle point (`grad A ~ B = 0`,
  `grad B ~ A = 0`): the adapter can never train, and nothing fails
  loudly. We measured exactly this: a 750-step QA-LoRA fine-tune of a
  2-bit GPTQ SmolLM2-1.7B where the default adapter learned nothing
  (final PPL 12230, identical to the unadapted quantized model), because
  every `lora_A` was all zeros (probed directly).

Since QA-LoRA targets quantized GPTQ models, the GPTQModel interaction is
not exotic; it is the canonical usage (the qalora_finetuning example).

Suggested fix: after constructing the replacement layer in
`QALoraLinearVariant.init`, explicitly initialize it, e.g. re-run the
configured init logic on the new pooled shape:

```python
module.lora_A[adapter_name] = new_lora_A_layer
module.reset_lora_parameters(adapter_name, config.init_lora_weights)
```

or at minimum `nn.init.kaiming_uniform_(new_lora_A_layer.weight,
a=math.sqrt(5))` with a comment that constructor init cannot be relied on.
(A direct `Tensor.uniform_` call would additionally be immune to the
monkeypatch class of problems, since only `torch.nn.init` functions get
patched by loaders.)

Happy to submit a PR; we hit this while benchmarking adapter
initializations for QA-LoRA (the integration from PRs #2571/#2664).

### Evidence appendix (for either issue, on request)

- Stage probe: post-reset lora_A on the GPTQ path contains junk
  (min -7.7e35, max +2e29, std inf in bf16) BEFORE the variant
  replacement; the replacement nn.Linear contains junk after construction
  (std 1e-4, one layer with +inf) despite `device="cuda:0"`.
- Function identity probe: `torch.nn.init.kaiming_uniform_` is
  `ModelLoader.from_pretrained.<locals>.skip` after
  `GPTQModel.from_pretrained` and stays so after quantize,
  save_quantized, and from_quantized.
- Control: fresh `nn.Linear(36, 4)` created after the pipeline has
  weight std 0.0001 (expected ~0.096).
- Outcome A/B: corrected, explicitly written Kaiming baseline reaches
  WikiText-2 PPL 34.08 (mean, 3 seeds) where the broken default adapter
  gave 38-42 (junk init, unseeded spread 4.2) or 12230 (zero init, no
  training).
