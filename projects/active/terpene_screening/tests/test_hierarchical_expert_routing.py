from __future__ import annotations

import numpy as np
import pytest

from projects.active.terpene_screening.hierarchical_expert_routing import (
    ExpertExecutionSpec,
    plan_expert_execution,
    stable_union_prefix,
)


def test_cached_global_scores_full_universe():
    spec = ExpertExecutionSpec("ESM-C", "global_cached", fallback="base")
    d = plan_expert_execution(
        spec,
        candidate_count=7,
        query_supported=True,
        candidate_features_cached=True,
    )
    assert d.mode == "full"
    assert d.candidate_count == 7
    assert d.candidate_fraction == 1.0
    assert np.array_equal(d.candidate_rows, np.arange(7))


def test_conditional_cached_falls_back_when_query_is_unsupported():
    spec = ExpertExecutionSpec("CLIPZyme", "query_conditional_cached", fallback="base_bime")
    d = plan_expert_execution(
        spec,
        candidate_count=10,
        query_supported=False,
        candidate_features_cached=True,
    )
    assert d.mode == "fallback"
    assert d.fallback == "base_bime"
    assert d.candidate_count == 0


def test_on_demand_never_materializes_full_universe_without_cache():
    spec = ExpertExecutionSpec(
        "EnzGFM-temporary", "shortlist_on_demand", fallback="cached_bime",
        default_shortlist_k=100, max_shortlist_k=500,
    )
    rows = np.array([9, 2, 2, 5], dtype=np.int32)
    d = plan_expert_execution(
        spec,
        candidate_count=1000,
        query_supported=True,
        candidate_features_cached=False,
        shortlist_rows=rows,
    )
    assert d.mode == "shortlist"
    assert np.array_equal(d.candidate_rows, np.array([2, 5, 9], dtype=np.int32))
    assert d.candidate_count == 3
    assert d.candidate_fraction == pytest.approx(0.003)


def test_on_demand_cached_specialist_may_score_full_universe():
    spec = ExpertExecutionSpec(
        "cached-specialist", "shortlist_on_demand", fallback="base",
        default_shortlist_k=50, max_shortlist_k=500,
    )
    d = plan_expert_execution(
        spec,
        candidate_count=12,
        query_supported=True,
        candidate_features_cached=True,
        shortlist_rows=np.array([1, 2]),
    )
    assert d.mode == "full"
    assert d.candidate_count == 12


def test_shortlist_contract_rejects_widening():
    spec = ExpertExecutionSpec(
        "on-demand", "shortlist_on_demand", fallback="base",
        default_shortlist_k=2, max_shortlist_k=3,
    )
    with pytest.raises(ValueError, match="exceeds"):
        plan_expert_execution(
            spec,
            candidate_count=10,
            query_supported=True,
            candidate_features_cached=False,
            shortlist_rows=np.array([0, 1, 2, 3]),
        )


def test_stable_union_prefix_is_bounded_and_deduplicated():
    a = np.array([3, 1, 0, 2, 4], dtype=np.int32)
    b = np.array([1, 2, 4, 3, 0], dtype=np.int32)
    rows = stable_union_prefix([a, b], 2, 5)
    assert np.array_equal(rows, np.array([1, 2, 3], dtype=np.int32))
    assert set(rows).issubset(set(a[:2]) | set(b[:2]))
