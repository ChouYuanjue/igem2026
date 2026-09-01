import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.evaluate_interaction_retriever_marts import PairResidualHead
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (
    blend_coarse_and_residual,
    exact_fallback_positive_ranks,
    positive_rank_signature,
    query_metrics_from_positive_rank_frame,
    reconstruct_positive_ranks,
    routed_residual_scale,
    zero_initialize_residual,
)


def test_residual_blend_scale_zero_is_exact_coarse() -> None:
    coarse = np.array([0.7, -0.1, 0.2])
    residual = np.array([4.0, -3.0, 8.0])
    assert np.array_equal(blend_coarse_and_residual(coarse, residual, 0.0), coarse)
    assert np.allclose(
        blend_coarse_and_residual(coarse, residual, 0.01),
        coarse + 0.01 * residual,
    )


def test_train_distance_gate_preserves_coarse_below_threshold() -> None:
    assert routed_residual_scale(
        0.03, reaction_similarity=0.8999, min_reaction_similarity=0.9
    ) == 0.0
    assert routed_residual_scale(
        0.03, reaction_similarity=0.9, min_reaction_similarity=0.9
    ) == 0.03
    assert routed_residual_scale(
        0.03, reaction_similarity=0.1, min_reaction_similarity=None
    ) == 0.03


def test_zero_initialized_residual_is_exact_zero() -> None:
    torch.manual_seed(7)
    head = PairResidualHead(8, 5, 0.0)
    zero_initialize_residual(head)
    r = torch.randn(11, 8)
    p = torch.randn(11, 8)
    assert torch.equal(head(r, p), torch.zeros(11))


def test_reconstruct_positive_ranks_only_changes_shortlisted_positives() -> None:
    positives = {"pA", "pB", "pC"}
    reranked = ["x", "pB", "y", "pA"]
    coarse = {"pA": 3, "pB": 4, "pC": 91}
    ranks = reconstruct_positive_ranks(
        positives=positives,
        reranked_top_ids=reranked,
        coarse_positive_ranks=coarse,
    )
    assert np.array_equal(ranks, np.array([4, 2, 91]))


def test_query_metrics_are_rebuilt_with_current_cutoffs() -> None:
    positive_ranks = pd.DataFrame(
        [
            {"query_id": "r1", "positive_rank": 2, "candidate_count": 100},
            {"query_id": "r1", "positive_rank": 7, "candidate_count": 100},
            {"query_id": "r2", "positive_rank": 4, "candidate_count": 100},
        ]
    )
    metrics = query_metrics_from_positive_rank_frame(positive_ranks)
    assert {"hit_at_2", "hit_at_4", "ndcg_at_2", "ndcg_at_4"} <= set(metrics.columns)
    r1 = metrics.set_index("query_id").loc["r1"]
    r2 = metrics.set_index("query_id").loc["r2"]
    assert r1["hit_at_2"] == 1
    assert r1["hit_at_4"] == 1
    assert r2["hit_at_2"] == 0
    assert r2["hit_at_4"] == 1


def test_positive_rank_signature_is_order_independent() -> None:
    assert positive_rank_signature({"P2": 7, "P1": 3}) == "P1:3|P2:7"
    assert positive_rank_signature({"P1": 3, "P2": 7}) == "P1:3|P2:7"


def test_exact_fallback_rank_vector_uses_coarse_mapping_order() -> None:
    coarse = {"P3": 1713, "P1": 3, "P2": 7}
    ranks = exact_fallback_positive_ranks(coarse, {"P1", "P2", "P3"})
    assert ranks.tolist() == [3, 7, 1713]
