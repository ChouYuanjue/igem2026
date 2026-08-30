from __future__ import annotations

import pandas as pd
import pytest

from projects.active.terpene_screening.evaluate_reaction_novelty_router import route_query_metrics


def frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "direction": ["reaction_to_enzyme"] * 3,
        "query_id": ["R1", "R2", "R3"],
        "candidate_count": [100] * 3,
        "positive_count": [1] * 3,
        "reciprocal_rank": scores,
    })


def test_router_uses_expert_only_below_train_similarity_threshold():
    backbone = frame([0.1, 0.2, 0.3])
    expert = frame([0.9, 0.8, 0.7])
    similarity = pd.DataFrame({
        "reaction_id": ["R1", "R2", "R3"],
        "max_train_drfp_tanimoto": [0.2, 0.5, 0.9],
    })
    routed = route_query_metrics(backbone, expert, similarity, threshold=0.5)
    assert routed["reciprocal_rank"].tolist() == [0.9, 0.2, 0.3]
    assert routed["route_source"].tolist() == ["novelty_expert", "backbone", "backbone"]


def test_router_requires_complete_train_distance_without_label_fallback():
    similarity = pd.DataFrame({"reaction_id": ["R1"], "max_train_drfp_tanimoto": [0.2]})
    with pytest.raises(ValueError, match="missing train-distance"):
        route_query_metrics(frame([0.1, 0.2, 0.3]), frame([0.9, 0.8, 0.7]), similarity, threshold=0.5)
