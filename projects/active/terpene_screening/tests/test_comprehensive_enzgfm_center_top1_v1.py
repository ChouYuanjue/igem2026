from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projects.active.terpene_screening.evaluate_comprehensive_enzgfm_center_top1_v1_gate import (
    _low_slice_checks,
    _rank1_protection_invariant,
)

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_ENZGFM_CENTER_TOP1_V1.json"


def test_protocol_is_frozen_with_rank1_protection_before_performance() -> None:
    p = json.loads(PROTOCOL.read_text())
    assert p["status"] == "frozen_before_any_new_split_performance_materialization"
    assert p["global_rules"]["external_or_revealed_outer_metrics_used_for_selection"] is False
    assert p["development_split"]["split_salt"] == "comprehensive_enzgfm_center_top1_v1_dev_20260901"
    assert p["confirmation_split"]["split_salt"] == "comprehensive_enzgfm_center_top1_v1_confirm_20260901"
    assert p["confirmation_split"]["dev_fold"] == 6
    assert p["top2000_refinement"]["shortlist_size"] == 2000
    assert p["top2000_refinement"]["pair_residual_scale"] == 0.03
    assert p["top2000_refinement"]["router_min_reaction_similarity"] == 0.9
    assert p["top2000_refinement"]["protected_coarse_prefix"] == 1
    assert p["development_gate"]["material_gain"].startswith("At least one pooled overall metric")


def test_rank1_invariant_requires_selected_queries_to_preserve_exact_id() -> None:
    support = pd.DataFrame([
        {"query_id": "R1", "max_train_drfp_tanimoto": 0.95, "protected_coarse_prefix": 1, "pair_reranker_selected": 1, "routed_residual_scale": 0.03, "coarse_top1_id": "P1", "routed_top1_id": "P1", "coarse_top1_preserved": 1},
        {"query_id": "R2", "max_train_drfp_tanimoto": 0.5, "protected_coarse_prefix": 1, "pair_reranker_selected": 0, "routed_residual_scale": 0.0, "coarse_top1_id": "P2", "routed_top1_id": "P2", "coarse_top1_preserved": 1},
    ])
    assert _rank1_protection_invariant(support, threshold=0.9, protected_prefix=1)["pass"] is True
    support.loc[0, "routed_top1_id"] = "PX"
    assert _rank1_protection_invariant(support, threshold=0.9, protected_prefix=1)["pass"] is False


def test_low_slice_gate_uses_frozen_hit10_tolerance() -> None:
    baseline = {"mrr": 0.1, "map": 0.08, "hit_at_10": 0.2, "median_best_positive_rank": 100.0}
    good = {"mrr": 0.1, "map": 0.08, "hit_at_10": 0.195, "median_best_positive_rank": 99.0}
    bad = {"mrr": 0.1, "map": 0.08, "hit_at_10": 0.1949, "median_best_positive_rank": 99.0}
    assert all(_low_slice_checks(good, baseline).values())
    assert not all(_low_slice_checks(bad, baseline).values())
