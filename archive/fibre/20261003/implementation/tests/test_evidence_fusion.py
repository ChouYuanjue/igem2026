from __future__ import annotations

import torch

from projects.active.fibre.kernel.evidence_fusion import (
    AnchoredEvidenceFusion,
    EvidenceChannelSpec,
    query_standardize,
)


def build_model() -> AnchoredEvidenceFusion:
    return AnchoredEvidenceFusion(
        (
            EvidenceChannelSpec("protein_view", "molecular_view", has_quality=True),
            EvidenceChannelSpec("structure", "structural"),
        ),
        initial_strength=0.2,
        initial_quality_slope=0.3,
    )


def test_missing_evidence_is_exactly_neutral() -> None:
    model = build_model()
    core = torch.tensor([0.2, -0.4, 1.3])
    evidence = torch.tensor([[2.0, -1.0], [1.0, 4.0], [-3.0, 2.0]])
    available = torch.zeros_like(evidence, dtype=torch.bool)
    fused, diag = model(core, evidence, available)
    assert torch.equal(fused, core)
    assert torch.count_nonzero(diag["channel_contributions"]) == 0


def test_available_channels_do_not_compete_for_mass() -> None:
    model = build_model()
    core = torch.zeros(1)
    evidence = torch.tensor([[2.0, 3.0]])
    both, diag_both = model(core, evidence, torch.tensor([[True, True]]))
    only_first, diag_one = model(core, evidence, torch.tensor([[True, False]]))
    first_contribution_both = diag_both["channel_contributions"][0, 0]
    first_contribution_one = diag_one["channel_contributions"][0, 0]
    assert torch.allclose(first_contribution_both, first_contribution_one)
    assert both > only_first


def test_channel_order_is_semantically_declared_not_softmax_competition() -> None:
    model = build_model()
    core = torch.tensor([0.0])
    evidence = torch.tensor([[0.4, 0.7]])
    available = torch.tensor([[True, True]])
    quality = torch.tensor([[0.8, 0.0]])
    fused, diag = model(core, evidence, available, quality)
    manual = core + diag["channel_contributions"].sum(dim=-1)
    assert torch.allclose(fused, manual)


def test_higher_quality_cannot_reduce_quality_aware_weight() -> None:
    model = build_model()
    low = model.channel_weights(torch.tensor([[0.1, 0.0]]))[0, 0]
    high = model.channel_weights(torch.tensor([[0.9, 0.0]]))[0, 0]
    assert high >= low


def test_non_quality_channel_ignores_quality_input() -> None:
    model = build_model()
    a = model.channel_weights(torch.tensor([[0.2, 0.0]]))[0, 1]
    b = model.channel_weights(torch.tensor([[0.2, 1.0]]))[0, 1]
    assert torch.allclose(a, b)


def test_query_standardize_is_zero_mean_on_available_candidates() -> None:
    score = torch.tensor([[1.0, 2.0, 100.0, 4.0]])
    available = torch.tensor([[True, True, False, True]])
    z = query_standardize(score, available)
    supported = z[available]
    assert torch.allclose(supported.mean(), torch.tensor(0.0), atol=1e-6)
    assert z[0, 2] == 0
