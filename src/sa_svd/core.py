"""
Structure-Aware SVD (SA-SVD) initialization for QA-LoRA.

The idea in one paragraph
-------------------------
QA-LoRA fine-tunes a quantized base model by constraining the LoRA adapter to be
constant within each quantization group, so it can be merged into the group's
zero-point with zero re-quantization loss. Standard QA-LoRA initializes the
adapter randomly. At 2-bit precision, some models suffer "resolution collapse":
a few high-magnitude outlier weights stretch the quantization grid so far that
every other weight rounds to the same level, destroying the model. A randomly
initialized adapter cannot find its way out of that hole.

SA-SVD instead initializes the adapter with the *principal components of the
weight matrix*, computed in a way that respects the quantization group structure.
Concretely: pool the weight matrix along quantization groups, run SVD on the
pooled matrix, and use the top-r singular vectors as the adapter. The leftover
(weight minus low-rank approximation) becomes the residual that gets quantized.
Because the adapter now carries the dominant structure of the original weights,
the model starts fine-tuning from a far healthier point.

This is a quantization-group-aware variant of PiSSA (Meng et al., 2024).

Reference
---------
K. Ton-That, "Accelerating Quantization-Aware Training of 2-bit Compact Language
Models", M.Sc. thesis, TU Berlin, 2025. Supervised by Prof. W. Samek and
Prof. K.-R. Müller (Fraunhofer HHI).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SASVDResult:
    """Output of one SA-SVD decomposition for a single linear layer.

    Attributes
    ----------
    lora_A : torch.Tensor
        Adapter down-projection, shape (r, in_features // group_size).
        Stored in the pooled (group) space; expanded at forward time.
    lora_B : torch.Tensor
        Adapter up-projection, shape (out_features, r).
    residual : torch.Tensor
        W - B @ expand(A). This is what gets quantized and frozen.
        Shape (out_features, in_features), same as the original weight.
    """

    lora_A: torch.Tensor
    lora_B: torch.Tensor
    residual: torch.Tensor


def _expand_pooled(a_pooled: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand a pooled adapter back to full input width.

    Each pooled column represents one quantization group; we repeat it
    ``group_size`` times so the product B @ A_full has the original shape.
    This is the Kronecker expansion A_full = A_pooled (x) 1_{1 x g}.
    """
    return a_pooled.repeat_interleave(group_size, dim=1)


def sa_svd_init(
    weight: torch.Tensor,
    rank: int,
    group_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> SASVDResult:
    """Compute a Structure-Aware SVD initialization for one weight matrix.

    Parameters
    ----------
    weight : torch.Tensor
        The full-precision weight, shape (out_features, in_features).
    rank : int
        LoRA rank r (number of singular components to keep).
    group_size : int
        Quantization group size g. ``in_features`` must be divisible by g.
    dtype : torch.dtype
        Precision for the SVD. float32 is recommended for numerical stability
        even when the model runs in bfloat16.

    Returns
    -------
    SASVDResult
        Adapter matrices (in pooled space) and the residual to be quantized.

    Raises
    ------
    ValueError
        If ``in_features`` is not divisible by ``group_size``, or if ``rank``
        exceeds the pooled dimension.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}.")

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by "
            f"group_size ({group_size})."
        )

    n_groups = in_features // group_size
    if rank > min(out_features, n_groups):
        raise ValueError(
            f"rank ({rank}) cannot exceed min(out_features={out_features}, "
            f"n_groups={n_groups})."
        )

    w = weight.detach().to(dtype)

    # Phase 1 - structural pooling.
    # Reshape to (out, n_groups, group_size) and average within each group so
    # the decomposition aligns with the quantization blocks.
    w_grouped = w.reshape(out_features, n_groups, group_size)
    w_pooled = w_grouped.mean(dim=2)  # (out_features, n_groups)

    # Phase 2 - SVD on the pooled matrix.
    U, S, Vh = torch.linalg.svd(w_pooled, full_matrices=False)
    Ur = U[:, :rank]                  # (out_features, r)
    Sr = S[:rank]                     # (r,)
    Vhr = Vh[:rank, :]                # (r, n_groups)

    # Phase 3 - adapter initialization. Split the singular values evenly
    # between B and A so the product B @ A reconstructs the rank-r component.
    sqrt_S = torch.sqrt(Sr)
    lora_B = Ur * sqrt_S.unsqueeze(0)          # (out_features, r)
    lora_A = sqrt_S.unsqueeze(1) * Vhr         # (r, n_groups), pooled space

    # Phase 4 - residual. Subtract the full-width reconstruction from W.
    a_full = _expand_pooled(lora_A, group_size)   # (r, in_features)
    w_approx = lora_B @ a_full                     # (out_features, in_features)
    residual = w - w_approx

    return SASVDResult(
        lora_A=lora_A.contiguous(),
        lora_B=lora_B.contiguous(),
        residual=residual.contiguous(),
    )


def reconstruction_error(weight: torch.Tensor, result: SASVDResult,
                         group_size: int) -> float:
    """Relative Frobenius error of the rank-r SA-SVD approximation.

    Useful as a sanity check: a small value means the adapter captured most
    of the weight's energy, which is the regime where SA-SVD helps most.
    """
    a_full = _expand_pooled(result.lora_A, group_size)
    w_approx = result.lora_B @ a_full
    w = weight.detach().to(w_approx.dtype)
    return (torch.linalg.norm(w - w_approx) / torch.linalg.norm(w)).item()
