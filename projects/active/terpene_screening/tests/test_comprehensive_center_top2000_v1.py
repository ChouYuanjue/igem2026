from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projects.active.terpene_screening.evaluate_comprehensive_center_top2000_v1_gate import (
    _fallback_invariant,
    _fold_stability,
    _material_gain,
)

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_CENTER_TOP2000_V1.json"


def test_protocol_freezes_inherited_component_values_before_performance() -> None:
    p = json.loads(PROTOCOL.read_text())
    assert p["status"] == "frozen_before_any_new_split_performance_materialization"
    assert p["global_rules"]["external_or_revealed_outer_metrics_used_for_selection"] is False
    assert p["development_split"]["split_salt"] == "comprehensive_center_top2000_v1_dev_20260901"
    assert p["confirmation_split"]["split_salt"] == "comprehensive_center_top2000_v1_confirm_20260901"
    assert p["confirmation_split"]["dev_fold"] == 6
    assert p["coarse_model"]["center_max_residual_ratio"] == 0.1
    assert p["top2000_refinement"]["shortlist_size"] == 2000
    assert p["top2000_refinement"]["pair_residual_scale"] == 0.03
    assert p["top2000_refinement"]["router_min_reaction_similarity"] == 0.9
    assert p["top2000_refinement"]["no_scale_or_threshold_sweep"] is True


def test_material_gain_matches_frozen_threshold() -> None:
    assert _material_gain({"mrr": 0.01, "map": 0.0, "hit_at_10": 0.0})
    assert _material_gain({"mrr": 0.0, "map": 0.01, "hit_at_10": 0.0})
    assert _material_gain({"mrr": 0.0, "map": 0.0, "hit_at_10": 0.02})
    assert not _material_gain({"mrr": 0.0099, "map": 0.0099, "hit_at_10": 0.0199})


def test_fold_stability_requires_multifold_gain_and_no_large_regression() -> None:
    base = {f: {"mrr": 0.1, "map": 0.1} for f in range(3)}
    good = {0: {"mrr": 0.11, "map": 0.10}, 1: {"mrr": 0.12, "map": 0.10}, 2: {"mrr": 0.099, "map": 0.10}}
    bad = {0: {"mrr": 0.11, "map": 0.10}, 1: {"mrr": 0.12, "map": 0.10}, 2: {"mrr": 0.09, "map": 0.10}}
    assert _fold_stability(good, base)["pass"] is True
    assert _fold_stability(bad, base)["pass"] is False


def test_fallback_invariant_checks_exact_rank_signatures_and_metrics() -> None:
    coarse = pd.DataFrame([
        {"direction": "reaction_to_enzyme", "query_id": "R1", "candidate_count": 10, "positive_count": 1, "has_positive": 1, "best_positive_rank": 3, "reciprocal_rank": 1/3, "average_precision": 1/3},
        {"direction": "reaction_to_enzyme", "query_id": "R2", "candidate_count": 10, "positive_count": 1, "has_positive": 1, "best_positive_rank": 1, "reciprocal_rank": 1.0, "average_precision": 1.0},
    ])
    routed = coarse.copy()
    support = pd.DataFrame([
        {"query_id": "R1", "max_train_drfp_tanimoto": 0.5, "routed_residual_scale": 0.0, "pair_reranker_selected": 0, "coarse_positive_rank_signature": "P1:3", "reranked_positive_rank_signature": "P1:3", "positive_ranks_exactly_preserved": 1},
        {"query_id": "R2", "max_train_drfp_tanimoto": 0.95, "routed_residual_scale": 0.03, "pair_reranker_selected": 1, "coarse_positive_rank_signature": "P2:2", "reranked_positive_rank_signature": "P2:1", "positive_ranks_exactly_preserved": 0},
    ])
    result = _fallback_invariant(coarse, routed, support, threshold=0.9)
    assert result["query_count"] == 1
    assert result["positive_rank_signatures_exact"] is True
    assert result["query_metrics_exact"] is True
    assert result["pass"] is True
