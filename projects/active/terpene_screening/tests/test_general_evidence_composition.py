from __future__ import annotations

import torch
import pytest

from projects.active.terpene_screening.compose_general_evidence_towers import compose_state_dicts


def test_compose_state_dicts_takes_each_directional_tower():
    base = {
        "protein_tower.weight": torch.tensor([1.0]),
        "reaction_tower.weight": torch.tensor([2.0]),
        "shared_counter": torch.tensor([3], dtype=torch.int64),
    }
    protein = {key: value.clone() for key, value in base.items()}
    reaction = {key: value.clone() for key, value in base.items()}
    protein["protein_tower.weight"] = torch.tensor([10.0])
    reaction["reaction_tower.weight"] = torch.tensor([20.0])
    composed = compose_state_dicts(base, protein, reaction)
    assert composed["protein_tower.weight"].item() == 10.0
    assert composed["reaction_tower.weight"].item() == 20.0
    assert composed["shared_counter"].item() == 3


def test_compose_state_dicts_rejects_non_tower_drift():
    base = {"protein_tower.weight": torch.tensor([1.0]), "shared": torch.tensor([2.0])}
    protein = {key: value.clone() for key, value in base.items()}
    reaction = {key: value.clone() for key, value in base.items()}
    reaction["shared"] = torch.tensor([4.0])
    with pytest.raises(ValueError, match="non-tower state drifted"):
        compose_state_dicts(base, protein, reaction)
