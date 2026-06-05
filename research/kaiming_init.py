"""Corrected baseline init: explicitly written Kaiming A, zero B.

Why this exists (autoresearch/journal.md v11/v12): PEFT
0.19.1's QA-LoRA variant replaces lora_A with a fresh nn.Linear AFTER
update_layer's kaiming init ran, and on the GPTQ layer path that replacement
is never initialized. The "baseline" rows measured before 2026-06-05 trained
an UNINITIALIZED adapter (zero pages or junk memory). This module builds the
init PEFT *intended* as default, so it can be written explicitly via
write_adapter_weights, the same trusted path SA-SVD always used.

The init: per layer, lora_A ~ kaiming_uniform(a=sqrt(5)) on shape
(rank, n_groups) exactly as peft.tuners.lora.layer.reset_lora_parameters
would produce, lora_B = 0. Each layer gets its own seeded generator
(untied), matching the journal v10/v12 arm.
"""

from __future__ import annotations

import torch
from torch import nn


def build_kaiming_adapters(
    model: nn.Module,
    target_names: list[str],
    rank: int,
    group_size: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build PEFT-default Kaiming adapters for explicit writing.

    Does NOT touch the base weights: lora_B = 0 means the adapter
    contributes nothing at step 0, so the quantized base is the plain
    GPTQ model, exactly what the random-init baseline is supposed to be.

    write_adapter_weights rescales lora_A by group_size/(scaling*n_groups)
    to convert from core.py's canonical convention. We want the RAW Kaiming
    values to land in the adapter (that is what PEFT's own init would have
    produced), so pre-scale by the inverse. scaling = lora_alpha/rank = 1
    in this repo (lora_alpha=rank everywhere).
    """
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for i, name in enumerate(target_names):
        layer = model.get_submodule(name)
        out_f, in_f = layer.weight.shape
        n_groups = in_f // group_size

        # kaiming_uniform_(a=sqrt(5)) on a (rank, n_groups) weight:
        # gain = sqrt(2/(1+5)) = sqrt(1/3), fan_in = n_groups,
        # bound = gain * sqrt(3/fan_in) = 1/sqrt(n_groups).
        gen = torch.Generator().manual_seed(seed * 100_003 + i)
        bound = n_groups ** -0.5
        lora_A = torch.empty(rank, n_groups).uniform_(-bound, bound, generator=gen)
        lora_A = lora_A * (n_groups / group_size)  # cancels the qalora_scale
        lora_B = torch.zeros(out_f, rank)

        adapters[name] = {"lora_A": lora_A, "lora_B": lora_B}
    return adapters
