"""Tests for SA-SVD core math. Run with: pytest tests/

These need torch installed (they run on your machine / 4090, not in CI without
a torch wheel). They verify the two properties that matter:
  1. The residual identity W = B @ expand(A) + residual holds exactly.
  2. On genuinely low-rank weights, SA-SVD captures almost all the energy.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sa_svd.core import _expand_pooled, reconstruction_error, sa_svd_init  # noqa: E402


def test_shapes():
    W = torch.randn(64, 128)
    res = sa_svd_init(W, rank=8, group_size=16)
    assert res.lora_A.shape == (8, 8)      # (r, n_groups)
    assert res.lora_B.shape == (64, 8)     # (out, r)
    assert res.residual.shape == (64, 128)


def test_residual_identity():
    """B @ expand(A) + residual must reconstruct W to float precision."""
    W = torch.randn(64, 128)
    res = sa_svd_init(W, rank=8, group_size=16)
    a_full = _expand_pooled(res.lora_A, 16)
    recon = res.lora_B @ a_full
    assert torch.allclose(recon + res.residual, W, atol=1e-4)


def test_captures_low_rank_structure():
    """A rank-4 weight should be almost perfectly captured at rank >= 4."""
    out, n_groups, g, true_rank = 64, 8, 16, 4
    B_true = torch.randn(out, true_rank)
    A_true = torch.randn(true_rank, n_groups)
    pooled = B_true @ A_true
    W = pooled.repeat_interleave(g, dim=1)  # genuinely low-rank, group-aligned
    res = sa_svd_init(W, rank=true_rank, group_size=g)
    assert reconstruction_error(W, res, g) < 1e-3


def test_divisibility_guard():
    W = torch.randn(64, 128)
    with pytest.raises(ValueError, match="divisible"):
        sa_svd_init(W, rank=8, group_size=17)


def test_rank_guard():
    W = torch.randn(64, 128)
    with pytest.raises(ValueError, match="rank"):
        sa_svd_init(W, rank=999, group_size=16)
