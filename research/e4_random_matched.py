"""
E4: norm-matched random init (red-team.md, experiment E4, redesigned in §8).

Tests direction vs scale: build an adapter with SA-SVD's exact singular-value
spectrum but random orthonormal directions, move it into the base the same way
SA-SVD does (group-constant low-rank component, so the init function is again
approximately Q(W) by the shift theorem), and train under baseline conditions.

  SA-SVD        : B  = U  sqrt(S),  A  = sqrt(S) Vh      (U, Vh from SVD)
  random_matched: B' = U' sqrt(S),  A' = sqrt(S) V'h     (U', V'h random orthonormal)

Per layer this matches ||B'||_F = ||B||_F, ||A'||_F = ||A||_F, every component
scale sqrt(s_k), and ||B'A'||_F = ||BA||_F = ||diag(S_r)||_F. The only thing
that differs from SA-SVD is the directions.

Outcome reading (red-team.md §8):
  - lands near sa_svd's 36.0  -> scale story wins (B4): any large structured
    nonzero init helps; "principal components" is the wrong mechanism.
  - lands near baseline's 42.5 -> direction story wins (H_init): the principal
    components themselves carry the benefit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def apply_random_matched_to_base(
    model: nn.Module,
    target_names: list[str],
    rank: int,
    group_size: int,
    seed: int,
    flat_spectrum: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Like sa_svd.apply_sa_svd_to_base, but with random orthonormal directions.

    Mirrors the SA-SVD recipe (pool, SVD in float32, keep top-r singular
    values) and then discards U, Vh in favour of random orthonormal factors.
    Replaces each target weight with W - B'*expand(A') and returns the
    (A', B') adapter components for write_adapter_weights.

    With ``flat_spectrum=True`` (E6), the singular values are additionally
    replaced by a single constant chosen so the *product* norm ||B'A'||_F
    matches SA-SVD's ||diag(S_r)||_F per layer. Note a flat spectrum cannot
    match the product norm and the factor norms simultaneously; the product
    norm is matched because it is the function-space size of the component
    that the base subtraction and the eventual zero-point merge both see.
    """
    gen = torch.Generator().manual_seed(seed)
    adapters: dict[str, dict[str, torch.Tensor]] = {}

    for name in target_names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear (got {type(layer).__name__}).")

        W = layer.weight.data.to(torch.float32)
        out_features, in_features = W.shape
        n_groups = in_features // group_size

        # Same pooling + SVD as sa_svd_init, but only the spectrum is kept.
        pooled = W.reshape(out_features, n_groups, group_size).mean(dim=-1)
        S = torch.linalg.svdvals(pooled)[:rank]
        if flat_spectrum:
            # Constant spectrum with the same product norm: c*sqrt(r) =
            # ||diag(S_r)||_F  ->  c = sqrt(sum(s_k^2) / r).
            S = torch.full_like(S, ((S**2).sum() / rank).sqrt())
        sqrt_s = S.sqrt()

        # Random orthonormal directions (QR of Gaussian), float32 on CPU for
        # determinism, then moved to the weight's device.
        U_r = torch.linalg.qr(
            torch.randn(out_features, rank, generator=gen)
        ).Q.to(W.device)
        V_r = torch.linalg.qr(
            torch.randn(n_groups, rank, generator=gen)
        ).Q.to(W.device)

        lora_B = U_r * sqrt_s.to(W.device)              # (out, r)
        lora_A = (V_r * sqrt_s.to(W.device)).T          # (r, n_groups)

        component = (lora_B @ lora_A).repeat_interleave(group_size, dim=1)
        layer.weight.data = (W - component).to(layer.weight.dtype)

        adapters[name] = {"lora_A": lora_A, "lora_B": lora_B}

    return adapters
