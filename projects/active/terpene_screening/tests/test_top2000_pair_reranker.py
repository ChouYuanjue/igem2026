import numpy as np
import torch

from projects.active.terpene_screening.evaluate_interaction_retriever_marts import PairResidualHead
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (
    reconstruct_positive_ranks,
    zero_initialize_residual,
)


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
