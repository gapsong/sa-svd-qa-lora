#!/usr/bin/env python
"""
Fast smoke demo (~5 min on a 4090, runs on CPU too).

This does NOT reproduce the headline result. Its only job is to prove the
SA-SVD pipeline runs end to end on your machine: load a tiny model, compute the
SA-SVD decomposition for its linear layers, verify the residual identity, and
report the reconstruction error per layer. If this passes, your environment is
correctly set up for the full reproduction.

Usage:
    python scripts/demo_fast.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from sa_svd import (  # noqa: E402
    apply_sa_svd_to_base,
    find_target_linears,
    reconstruction_error,
    sa_svd_init,
)

RANK = 16
GROUP_SIZE = 16


def main():
    print("=" * 64)
    print("SA-SVD fast smoke demo")
    print("=" * 64)

    # A tiny stand-in 'model' of a few linear layers, so this runs anywhere
    # without a multi-GB download. The real pipeline uses the same code path
    # on actual transformer layers (see scripts/reproduce.py).
    torch.manual_seed(0)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(512, 512, bias=False)
            self.v_proj = torch.nn.Linear(512, 512, bias=False)
            self.gate_proj = torch.nn.Linear(512, 1024, bias=False)

    model = TinyModel()

    targets = find_target_linears(model, ["q_proj", "v_proj", "gate_proj"])
    print(f"\nTarget layers found: {targets}")

    # Snapshot originals to verify the residual identity after the swap.
    originals = {n: model.get_submodule(n).weight.data.clone() for n in targets}

    print(f"\nComputing SA-SVD (rank={RANK}, group_size={GROUP_SIZE}) ...")
    adapters = apply_sa_svd_to_base(model, targets, rank=RANK, group_size=GROUP_SIZE)

    print("\nPer-layer check:")
    print(f"  {'layer':<12} {'rel. recon. error':>18}   identity")
    all_ok = True
    for name in targets:
        W = originals[name]
        res = sa_svd_init(W, rank=RANK, group_size=GROUP_SIZE)
        err = reconstruction_error(W, res, GROUP_SIZE)
        # residual + reconstruction must equal the original
        a_full = res.lora_A.repeat_interleave(GROUP_SIZE, dim=1)
        recon = res.lora_B @ a_full
        ok = torch.allclose(recon + res.residual, W, atol=1e-4)
        all_ok &= ok
        print(f"  {name:<12} {err:>18.4f}   {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 64)
    if all_ok:
        print("SMOKE DEMO PASSED - environment is ready.")
        print("Next: python scripts/reproduce.py   (full SmolLM2 result)")
    else:
        print("SMOKE DEMO FAILED - residual identity broke.")
        sys.exit(1)
    print("=" * 64)


if __name__ == "__main__":
    main()
