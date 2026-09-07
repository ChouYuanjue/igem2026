from __future__ import annotations

import torch
import pytest

from projects.active.terpene_screening.merge_general_evidence_ties import merge_state_dicts_ties
from projects.active.terpene_screening.third_party.ties_merge import disjoint_mean, resolve_sign_mass, topk_values_mask


def test_ties_topk20_keeps_largest_twenty_percent():
    values = torch.arange(1, 11, dtype=torch.float32).unsqueeze(0)
    masked = topk_values_mask(values, keep_fraction=0.2)
    # Faithfully preserve the upstream inclusive kth-value threshold: for this
    # tiny 10-value example topk20 retains 8,9,10 rather than forcing exactly 2.
    assert int((masked != 0).sum()) == 3
    assert masked[0, -1] == 10
    assert masked[0, -2] == 9
    assert masked[0, -3] == 8


def test_ties_elects_mass_sign_and_disjoint_mean():
    vectors = torch.tensor([[2.0, -4.0, 1.0], [1.0, 3.0, -2.0]])
    signs = resolve_sign_mass(vectors)
    assert signs.tolist() == [1.0, -1.0, -1.0]
    merged = disjoint_mean(vectors, signs)
    assert torch.allclose(merged, torch.tensor([1.5, -4.0, -2.0]))


def test_checkpoint_ties_only_changes_selected_directional_tower():
    base = {
        "protein_tower.weight": torch.tensor([5.0, 6.0]),
        "reaction_tower.weight": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "counter": torch.tensor([1], dtype=torch.int64),
    }
    first = {key: value.clone() for key, value in base.items()}
    second = {key: value.clone() for key, value in base.items()}
    first["reaction_tower.weight"] = torch.tensor([3.0, -2.0, 0.1, 0.0])
    second["reaction_tower.weight"] = torch.tensor([1.0, 4.0, 0.2, 0.0])
    merged = merge_state_dicts_ties(
        base, [first, second], keep_fraction=1.0, scale=1.0, prefixes=("reaction_tower.",)
    )
    assert torch.equal(merged["protein_tower.weight"], base["protein_tower.weight"])
    assert torch.equal(merged["counter"], base["counter"])
    assert torch.allclose(merged["reaction_tower.weight"], torch.tensor([2.0, 4.0, 0.15, 0.0]))


def test_checkpoint_ties_rejects_drift_outside_selected_tower():
    base = {"protein_tower.weight": torch.tensor([1.0]), "reaction_tower.weight": torch.tensor([2.0])}
    expert = {key: value.clone() for key, value in base.items()}
    expert["protein_tower.weight"] += 1
    with pytest.raises(ValueError, match="outside selected prefixes"):
        merge_state_dicts_ties(base, [expert], keep_fraction=0.2, scale=1.0, prefixes=("reaction_tower.",))
