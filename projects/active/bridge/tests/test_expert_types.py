from __future__ import annotations

import numpy as np

from projects.active.bridge.runtime.expert_types import (
    EXPERT_TYPE_IDS,
    R2EExpertTypeRuntime,
)
from projects.active.bridge.runtime.scientific_evidence import EvidenceOutput


def test_expert_type_ids_are_coarse_and_stable():
    assert EXPERT_TYPE_IDS == (
        "functional_foundation",
        "structural_geometry",
        "mechanistic",
        "family_domain",
        "contextual_observational",
    )


def test_type_aggregation_averages_only_available_members():
    a = EvidenceOutput(
        score=np.asarray([1.0, 2.0, 3.0]),
        available=np.asarray([True, True, False]),
    )
    b = EvidenceOutput(
        score=np.asarray([7.0, 9.0, 11.0]),
        available=np.asarray([False, True, True]),
    )
    out = R2EExpertTypeRuntime._aggregate_type(
        "family_domain",
        [("a", a), ("b", b)],
        3,
    )
    assert out.available.tolist() == [True, True, True]
    assert set(out.member_scores) == {"a", "b"}
    assert out.member_available["a"].tolist() == [True, True, False]
    assert out.member_available["b"].tolist() == [False, True, True]
    assert np.isfinite(out.score).all()


def test_empty_context_type_is_strictly_unavailable():
    out = R2EExpertTypeRuntime._aggregate_type(
        "contextual_observational",
        [],
        4,
    )
    assert out.available.tolist() == [False, False, False, False]
    assert np.allclose(out.score, 0.0)
