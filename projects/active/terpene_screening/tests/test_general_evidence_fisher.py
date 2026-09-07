from __future__ import annotations

import torch

from projects.active.terpene_screening.third_party.fusionbench_fisher import (
    merging_with_fisher_weights,
)
from projects.active.terpene_screening.merge_general_evidence_fisher import _sample_queries


def test_fisher_merge_prefers_model_with_larger_parameter_importance():
    params = {"w": [torch.tensor([0.0]), torch.tensor([10.0])]}
    fishers = [{"w": torch.tensor([100.0])}, {"w": torch.tensor([1.0])}]
    merged = merging_with_fisher_weights(
        params,
        fishers,
        fisher_scaling_coefficients=torch.tensor([1.0, 1.0]),
        normalize_fisher_weight=False,
        minimal_fisher_weight=0.0,
    )
    assert merged["w"].item() < 0.2


def test_fisher_source_scale_can_increase_source_priority():
    params = {"w": [torch.tensor([0.0]), torch.tensor([10.0])]}
    fishers = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([1.0])}]
    equal = merging_with_fisher_weights(params, fishers, torch.tensor([1.0, 1.0]), normalize_fisher_weight=False)
    source_heavy = merging_with_fisher_weights(params, fishers, torch.tensor([5.0, 1.0]), normalize_fisher_weight=False)
    assert source_heavy["w"].item() < equal["w"].item()


def test_query_sampling_is_deterministic_and_bounded():
    ids = [f"R{i}" for i in range(20)]
    first = _sample_queries(ids, 5, 17)
    second = _sample_queries(list(reversed(ids)), 5, 17)
    assert first == second
    assert len(first) == 5
