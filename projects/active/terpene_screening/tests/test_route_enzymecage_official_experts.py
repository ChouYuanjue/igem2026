from __future__ import annotations

import pandas as pd
import pytest

from projects.active.terpene_screening.route_enzymecage_official_experts import (
    cage_query_confidence,
    neural_query_confidence,
    route_direction,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"reaction_id": "R1", "protein_id": "P1", "label": 1, "cage_score": 0.5, "neural_score": 0.9, "neural_member_0_score": 0.9, "neural_member_1_score": 0.88},
        {"reaction_id": "R1", "protein_id": "P2", "label": 0, "cage_score": 0.5, "neural_score": 0.2, "neural_member_0_score": 0.2, "neural_member_1_score": 0.21},
        {"reaction_id": "R1", "protein_id": "P3", "label": 0, "cage_score": 0.5, "neural_score": 0.1, "neural_member_0_score": 0.1, "neural_member_1_score": 0.11},
        {"reaction_id": "R2", "protein_id": "P1", "label": 0, "cage_score": 0.95, "neural_score": 0.1, "neural_member_0_score": 0.1, "neural_member_1_score": 0.11},
        {"reaction_id": "R2", "protein_id": "P2", "label": 1, "cage_score": 0.1, "neural_score": 0.11, "neural_member_0_score": 0.11, "neural_member_1_score": 0.1},
        {"reaction_id": "R2", "protein_id": "P3", "label": 0, "cage_score": 0.05, "neural_score": 0.12, "neural_member_0_score": 0.12, "neural_member_1_score": 0.13},
    ])


def test_cage_saturated_scores_have_low_confidence():
    diag = cage_query_confidence([0.5, 0.5, 0.5, 0.5])
    assert diag["confidence"] < 0.25
    assert diag["top_tie_fraction"] == 1.0


def test_neural_agreement_produces_high_confidence():
    group = _frame()[lambda x: x.reaction_id.eq("R1")]
    diag = neural_query_confidence(group, "protein_id")
    assert diag["top1_vote_fraction"] == 1.0
    assert diag["top10_jaccard"] == 1.0
    assert diag["confidence"] > 0.7


def test_routing_does_not_depend_on_labels():
    frame = _frame()
    routed_a, diag_a = route_direction(frame, query_col="reaction_id", candidate_col="protein_id")
    flipped = frame.copy(); flipped["label"] = 1 - flipped["label"]
    routed_b, diag_b = route_direction(flipped, query_col="reaction_id", candidate_col="protein_id")
    assert routed_a["routed_score"].tolist() == pytest.approx(routed_b["routed_score"].tolist())
    assert diag_a[["query_id", "cage_weight", "neural_weight", "route"]].to_dict("records") == diag_b[["query_id", "cage_weight", "neural_weight", "route"]].to_dict("records")


def test_saturated_cage_routes_toward_consistent_neural_expert():
    _, diag = route_direction(_frame(), query_col="reaction_id", candidate_col="protein_id")
    r1 = diag[diag.query_id.eq("R1")].iloc[0]
    assert r1.neural_weight > r1.cage_weight
    assert r1.route == "neural_dominant"
