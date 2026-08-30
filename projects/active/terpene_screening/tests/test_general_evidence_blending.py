from __future__ import annotations

import torch
import pytest

from projects.active.terpene_screening.blend_general_evidence_models import blend_state_dicts


def test_blend_state_dicts_interpolates_floating_tensors():
    base = {"weight": torch.tensor([0.0, 2.0]), "counter": torch.tensor([3], dtype=torch.int64)}
    adapted = {"weight": torch.tensor([2.0, 4.0]), "counter": torch.tensor([3], dtype=torch.int64)}
    mixed = blend_state_dicts(base, adapted, 0.25)
    assert torch.allclose(mixed["weight"], torch.tensor([0.5, 2.5]))
    assert mixed["counter"].item() == 3


def test_blend_state_dicts_rejects_nonfloating_state_drift():
    base = {"counter": torch.tensor([3], dtype=torch.int64)}
    adapted = {"counter": torch.tensor([4], dtype=torch.int64)}
    with pytest.raises(ValueError, match="non-floating state differs"):
        blend_state_dicts(base, adapted, 0.5)


def test_blend_state_dicts_rejects_out_of_range_alpha():
    state = {"weight": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="alpha"):
        blend_state_dicts(state, state, 1.01)
