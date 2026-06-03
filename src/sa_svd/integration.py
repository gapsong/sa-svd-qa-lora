"""
Glue between :func:`sa_svd_init` and a PEFT QA-LoRA model.

The flow this module supports
-----------------------------
1. Load a full-precision base model.
2. For every target linear layer, compute the SA-SVD decomposition and replace
   the layer's weight with the *residual* (W - low-rank approximation).
3. Quantize the residual-weighted model with GPTQ at 2-bit.
4. Attach a QA-LoRA adapter and overwrite its A/B with the SA-SVD components.

Steps 3-4 use the public PEFT / GPTQModel APIs and live in the example scripts;
this module owns steps 1-2 and the adapter overwrite, which is the part that is
specific to SA-SVD.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn

from .core import sa_svd_init


def find_target_linears(model: nn.Module, patterns: Iterable[str]) -> list[str]:
    """Return fully-qualified names of nn.Linear modules matching any pattern.

    Parameters
    ----------
    model : nn.Module
        The base model to scan.
    patterns : Iterable[str]
        Regex fragments, e.g. ``["q_proj", "v_proj", "gate_proj"]``.

    Returns
    -------
    list[str]
        Module names suitable for ``model.get_submodule(name)``.
    """
    compiled = [re.compile(p) for p in patterns]
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(c.search(name) for c in compiled):
            names.append(name)
    return names


def apply_sa_svd_to_base(
    model: nn.Module,
    target_names: list[str],
    rank: int,
    group_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Replace each target layer's weight with its SA-SVD residual.

    The adapter matrices are returned (not attached) so the caller can write
    them into the QA-LoRA adapter after quantization, when the adapter modules
    actually exist.

    Returns
    -------
    dict
        Maps layer name -> {"lora_A": ..., "lora_B": ...}.
    """
    adapters: dict[str, dict[str, torch.Tensor]] = {}

    for name in target_names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear (got {type(layer).__name__}).")

        result = sa_svd_init(layer.weight.data, rank=rank, group_size=group_size)

        # Swap in the residual; the original full weight is no longer needed.
        layer.weight.data = result.residual.to(layer.weight.dtype)

        adapters[name] = {
            "lora_A": result.lora_A,
            "lora_B": result.lora_B,
        }

    return adapters


@torch.no_grad()
def write_adapter_weights(
    peft_model: nn.Module,
    adapters: dict[str, dict[str, torch.Tensor]],
    adapter_name: str = "default",
) -> int:
    """Overwrite a PEFT model's LoRA A/B tensors with SA-SVD components.

    Matches stored adapters to the model's LoRA modules by layer-name suffix.
    Returns the number of layers successfully written, so the caller can assert
    it matched everything it expected.
    """
    written = 0
    for layer_name, weights in adapters.items():
        for mod_name, module in peft_model.named_modules():
            if not mod_name.endswith(layer_name):
                continue
            lora_A = getattr(module, "lora_A", None)
            lora_B = getattr(module, "lora_B", None)
            if lora_A is None or lora_B is None:
                continue
            if adapter_name not in lora_A:
                continue

            a_dst = lora_A[adapter_name].weight
            b_dst = lora_B[adapter_name].weight
            a_src = weights["lora_A"].to(a_dst.dtype, a_dst.device)
            b_src = weights["lora_B"].to(b_dst.dtype, b_dst.device)

            if a_dst.shape != a_src.shape or b_dst.shape != b_src.shape:
                raise ValueError(
                    f"Shape mismatch on {layer_name}: "
                    f"A {tuple(a_dst.shape)} vs {tuple(a_src.shape)}, "
                    f"B {tuple(b_dst.shape)} vs {tuple(b_src.shape)}."
                )
            a_dst.copy_(a_src)
            b_dst.copy_(b_src)
            written += 1
            break

    return written
