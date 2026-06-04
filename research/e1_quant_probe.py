#!/usr/bin/env python
"""
E1: quantization-quality probe (red-team.md, experiment E1).

Tests the residual-base confound (red-team.md A1/B1) without any training:
does the SA-SVD residual quantize better at 2 bits than the full weight W?

For every target linear layer of the model, compare two reconstructions of W:

  baseline : Q(W)                      error = ||W - Q(W)||_F      / ||W||_F
  sa_svd   : Q(R) + B*expand(A)        error = ||R - Q(R)||_F      / ||W||_F

where R = W - B*expand(A) is the SA-SVD residual and the adapter term is exact
in float, so the sa_svd reconstruction error reduces to the residual's own
quantization error. Both errors share the denominator ||W||_F, which makes
them directly comparable.

Q() is 2-bit asymmetric per-group round-to-nearest (min/max grid, 4 levels),
a proxy for GPTQ without error compensation. CAVEAT: GPTQ redistributes
rounding error using calibration data, so absolute errors here overestimate
GPTQ's; the red-team question is only the *relative* ordering of W vs R.

Also reports per-group grid health: median group dynamic range, mean distinct
levels used per group (max 4), and mean modal-level occupancy (fraction of a
group's weights landing on its most popular level; high = collapse-like).

Usage:
    python research/e1_quant_probe.py [--model-id HuggingFaceTB/SmolLM2-1.7B]
                                      [--rank 16] [--group-size 16]
                                      [--device cuda] [--out research/e1_results.json]

CPU works (slower). No training, no GPTQ, no datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sa_svd import find_target_linears, sa_svd_init  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def quantize_2bit_rtn(w: torch.Tensor, group_size: int):
    """2-bit asymmetric per-group RTN. Returns (dequantized, codes, group_range)."""
    out, inp = w.shape
    g = w.reshape(out, inp // group_size, group_size)
    wmin = g.min(dim=-1, keepdim=True).values
    wmax = g.max(dim=-1, keepdim=True).values
    rng = wmax - wmin
    scale = torch.where(rng > 0, rng / 3.0, torch.ones_like(rng))
    codes = torch.round((g - wmin) / scale).clamp_(0, 3)
    deq = (codes * scale + wmin).reshape(out, inp)
    return deq, codes, rng.squeeze(-1)


def grid_health(codes: torch.Tensor):
    """Mean distinct levels per group and mean modal-level occupancy."""
    # codes: (out, n_groups, group_size) with integer values in {0,1,2,3}
    counts = torch.stack([(codes == v).sum(dim=-1) for v in range(4)], dim=-1)
    distinct = (counts > 0).sum(dim=-1).float()
    modal = counts.max(dim=-1).values.float() / codes.shape[-1]
    return distinct.mean().item(), modal.mean().item()


def probe_layer(W: torch.Tensor, rank: int, group_size: int) -> dict:
    res = sa_svd_init(W, rank=rank, group_size=group_size)
    R = res.residual.to(W.device)
    w_norm = W.norm()

    qW, cW, rngW = quantize_2bit_rtn(W, group_size)
    qR, cR, rngR = quantize_2bit_rtn(R, group_size)

    distW, modW = grid_health(cW)
    distR, modR = grid_health(cR)

    return {
        "relerr_baseline": ((W - qW).norm() / w_norm).item(),
        "relerr_sa_svd": ((R - qR).norm() / w_norm).item(),
        "median_group_range_W": rngW.median().item(),
        "median_group_range_R": rngR.median().item(),
        "mean_distinct_levels_W": distW,
        "mean_distinct_levels_R": distR,
        "mean_modal_occupancy_W": modW,
        "mean_modal_occupancy_R": modR,
        "numel": W.numel(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=16)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(ROOT / "research" / "e1_results.json"))
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model_id} (float32, cpu) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32
    )
    names = find_target_linears(model, TARGETS)
    print(f"Probing {len(names)} layers on {args.device} "
          f"(rank={args.rank}, group_size={args.group_size}) ...")

    per_layer: dict[str, dict] = {}
    for i, name in enumerate(names):
        W = model.get_submodule(name).weight.data.to(args.device)
        per_layer[name] = probe_layer(W, args.rank, args.group_size)
        if (i + 1) % 28 == 0:
            print(f"  {i + 1}/{len(names)}")

    # Aggregate per module type, numel-weighted.
    by_type: dict[str, dict] = {}
    buckets = defaultdict(list)
    for name, m in per_layer.items():
        buckets[name.rsplit(".", 1)[-1]].append(m)
    keys = [k for k in next(iter(per_layer.values())) if k != "numel"]
    for t, ms in buckets.items():
        tot = sum(m["numel"] for m in ms)
        by_type[t] = {k: sum(m[k] * m["numel"] for m in ms) / tot for k in keys}
        by_type[t]["n_layers"] = len(ms)
    tot = sum(m["numel"] for m in per_layer.values())
    overall = {k: sum(m[k] * m["numel"] for m in per_layer.values()) / tot
               for k in keys}

    print(f"\n{'module':<10} {'relerr W':>9} {'relerr R':>9} {'ratio R/W':>9} "
          f"{'levels W':>9} {'levels R':>9} {'modal W':>8} {'modal R':>8}")
    for t in TARGETS:
        m = by_type[t]
        print(f"{t:<10} {m['relerr_baseline']:>9.4f} {m['relerr_sa_svd']:>9.4f} "
              f"{m['relerr_sa_svd'] / m['relerr_baseline']:>9.3f} "
              f"{m['mean_distinct_levels_W']:>9.3f} "
              f"{m['mean_distinct_levels_R']:>9.3f} "
              f"{m['mean_modal_occupancy_W']:>8.3f} "
              f"{m['mean_modal_occupancy_R']:>8.3f}")
    print(f"{'OVERALL':<10} {overall['relerr_baseline']:>9.4f} "
          f"{overall['relerr_sa_svd']:>9.4f} "
          f"{overall['relerr_sa_svd'] / overall['relerr_baseline']:>9.3f} "
          f"{overall['mean_distinct_levels_W']:>9.3f} "
          f"{overall['mean_distinct_levels_R']:>9.3f} "
          f"{overall['mean_modal_occupancy_W']:>8.3f} "
          f"{overall['mean_modal_occupancy_R']:>8.3f}")

    out = Path(args.out)
    out.write_text(json.dumps({
        "model": args.model_id, "rank": args.rank,
        "group_size": args.group_size,
        "quantizer": "2-bit asymmetric per-group RTN (GPTQ proxy, no error compensation)",
        "overall": overall, "by_type": by_type, "per_layer": per_layer,
    }, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
