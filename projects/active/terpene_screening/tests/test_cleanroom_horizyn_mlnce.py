from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "external/horizyn"))
from horizyn.losses import FullBatchMLNCELoss
from projects.active.terpene_screening.train_cleanroom_horizyn_mlnce import _unique_with_inverse


def test_unique_with_inverse_matches_positive_pair_semantics() -> None:
    unique, inverse = _unique_with_inverse(["a", "b", "a", "c"])
    assert unique == ["a", "b", "c"]
    assert inverse.tolist() == [0, 1, 0, 2]


def test_official_horizyn_loss_matches_documented_formula() -> None:
    d = torch.tensor([[0.1, 0.8], [0.7, 0.2]], dtype=torch.float32)
    q = torch.tensor([0, 1], dtype=torch.long); t = torch.tensor([0, 1], dtype=torch.long)
    loss = FullBatchMLNCELoss(beta=10.0, learn_beta=False)(d, q, t)
    expected = 10.0 * torch.tensor([0.1, 0.2]).mean() + torch.logsumexp(-10.0 * d, dim=(0, 1))
    assert float(loss) == pytest.approx(float(expected))
