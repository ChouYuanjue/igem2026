from __future__ import annotations

import torch
import pytest

from projects.active.terpene_screening.train_retrieval_adamerging import materialize_layerwise_merge, multipositive_bidirectional_loss


def test_materialize_layerwise_merge_uses_per_tensor_coefficients_and_directional_experts():
    base = {
        "protein_tower.a": torch.tensor([1.0, 1.0]),
        "reaction_tower.b": torch.tensor([2.0]),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    protein = {key: value.clone() for key, value in base.items()}
    reaction = {key: value.clone() for key, value in base.items()}
    protein["protein_tower.a"] = torch.tensor([3.0, 5.0])
    reaction["reaction_tower.b"] = torch.tensor([6.0])
    merged = materialize_layerwise_merge(
        base, protein, reaction,
        {"protein_tower.a": 0.25, "reaction_tower.b": 0.5},
    )
    assert torch.allclose(merged["protein_tower.a"], torch.tensor([1.5, 2.0]))
    assert torch.allclose(merged["reaction_tower.b"], torch.tensor([4.0]))
    assert torch.equal(merged["counter"], base["counter"])


def test_materialize_rejects_directional_expert_cross_tower_drift():
    base = {"protein_tower.a": torch.tensor([1.0]), "reaction_tower.b": torch.tensor([2.0])}
    protein = {key: value.clone() for key, value in base.items()}
    reaction = {key: value.clone() for key, value in base.items()}
    protein["reaction_tower.b"] += 1
    with pytest.raises(ValueError, match="outside protein_tower"):
        materialize_layerwise_merge(base, protein, reaction, {"protein_tower.a": 1.0, "reaction_tower.b": 1.0})


def test_multipositive_loss_rewards_higher_known_pair_scores_in_both_directions():
    positive = torch.eye(2, dtype=torch.bool)
    good = torch.tensor([[4.0, -1.0], [-1.0, 4.0]])
    bad = torch.tensor([[-1.0, 4.0], [4.0, -1.0]])
    good_loss, good_e2r, good_r2e = multipositive_bidirectional_loss(good, positive)
    bad_loss, _, _ = multipositive_bidirectional_loss(bad, positive)
    assert good_loss < bad_loss
    assert good_e2r.item() > 0
    assert good_r2e.item() > 0
