"""
THE MUTABLE FILE. The autoresearch loop edits this file and nothing else.

Contract (enforced by harness.py):
  build_init(weight, rank, group_size, seed) -> (lora_A, lora_B)
    weight     : (out_features, in_features) float32, the FP weight matrix
    lora_A     : (rank, in_features // group_size) float32
    lora_B     : (out_features, rank) float32

The harness sets the quantized base to W - B*expand(A), so the step-0
function always equals W. Only the adapter parameterization is searched.
Use `seed` for any randomness so runs are reproducible.

Current candidate: exact SA-SVD (the project baseline init).
"""

from __future__ import annotations

import torch

from sa_svd import sa_svd_init


def build_init(weight: torch.Tensor, rank: int, group_size: int,
               seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    result = sa_svd_init(weight, rank=rank, group_size=group_size)
    return result.lora_A, result.lora_B
