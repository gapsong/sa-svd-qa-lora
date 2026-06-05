"""Activation-whitened SA-SVD init (journal v18, 2026-06-05).

Plain SA-SVD is data-free: the SVD of the pooled weight spends adapter rank
on directions that are large in W, even if real inputs never excite them.
GPTQ itself is activation-aware (it minimizes ||(W - Q(W)) X||^2 via the
calibration Hessian), so the adapter init is currently the only data-blind
stage of the pipeline. This variant whitens the pooled weight with input
second moments before the SVD (diagonal ASVD / SVD-LLM idea, restricted to
the QA-LoRA group-constant structure).

Derivation: we approximate W with B * expand(A) minimizing the
activation-weighted error ||(W - B expand(A)) diag(d)||_F where
d_j = sqrt(E[x_j^2]) from calibration data. Within group k the expanded
adapter is a constant v per row, and the weighted least-squares optimum is
the d^2-weighted group mean (not the plain mean core.py uses). Stacking
those means gives W_tilde (out, n_groups); the remaining low-rank problem
is standard SVD of W_tilde * diag(g) with group weight g_k =
sqrt(sum_{j in k} d_j^2). Then

    B = U_r sqrt(S_r),   A = sqrt(S_r) Vh_r diag(1/g)

and residual = W - B expand(A) holds exactly by construction, so the
residual identity invariant is preserved regardless of the whitening.
"""
from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def collect_input_sq_moments(
    model: nn.Module,
    target_names: list[str],
    calib: list[dict],
) -> dict[str, torch.Tensor]:
    """Per-input-channel E[x_j^2] for each target linear, from calibration.

    ``calib`` is the same tokenized format the GPTQ stage uses
    (pipeline._calibration_samples): a list of dicts with ``input_ids``.
    Samples run one at a time (no padding needed).
    """
    device = next(model.parameters()).device
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, args):
            x = args[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            if name not in sums:
                sums[name] = torch.zeros(flat.shape[-1], device=flat.device)
                counts[name] = 0
            sums[name] += flat.pow(2).sum(dim=0)
            counts[name] += flat.shape[0]
        return hook

    for name in target_names:
        hooks.append(model.get_submodule(name).register_forward_pre_hook(make_hook(name)))
    try:
        was_training = model.training
        model.eval()
        for sample in calib:
            ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0)
            model(input_ids=ids)
        if was_training:
            model.train()
    finally:
        for h in hooks:
            h.remove()

    return {n: (sums[n] / max(counts[n], 1)).cpu() for n in target_names}


@torch.no_grad()
def apply_whitened_sa_svd_to_base(
    model: nn.Module,
    target_names: list[str],
    rank: int,
    group_size: int,
    moments: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Whitened analogue of sa_svd.apply_sa_svd_to_base.

    Swaps each target weight for its residual and returns the adapter
    components in the same canonical convention core.py uses, so
    write_adapter_weights applies its usual qalora rescale unchanged.
    """
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for name in target_names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear (got {type(layer).__name__}).")
        w = layer.weight.data.float()
        out_f, in_f = w.shape
        if in_f % group_size != 0:
            raise ValueError(f"{name}: in_features {in_f} not divisible by {group_size}")
        n_groups = in_f // group_size

        d2 = moments[name].to(w.device).float()
        # Guard dead channels so the unwhitening 1/g stays finite.
        d2 = d2.clamp(min=1e-8 * d2.mean().clamp(min=1e-12))
        d2g = d2.reshape(n_groups, group_size)
        denom = d2g.sum(dim=-1)                      # (n_groups,)
        # d^2-weighted group means (weighted LS optimum per group).
        w_tilde = (w.reshape(out_f, n_groups, group_size) * d2g).sum(-1) / denom
        g = denom.sqrt()                             # (n_groups,)

        u, s, vh = torch.linalg.svd(w_tilde * g, full_matrices=False)
        sr = s[:rank].sqrt()
        lora_B = u[:, :rank] * sr                    # (out, r)
        lora_A = (sr.unsqueeze(1) * vh[:rank]) / g   # (r, n_groups)

        expanded = torch.repeat_interleave(lora_A, group_size, dim=1)
        residual = w - lora_B @ expanded
        layer.weight.data = residual.to(layer.weight.dtype)
        adapters[name] = {"lora_A": lora_A.cpu(), "lora_B": lora_B.cpu()}
    return adapters
