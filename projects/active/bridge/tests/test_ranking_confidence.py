from __future__ import annotations

import torch

from projects.active.bridge.kernel.ranking_confidence import (
    pairwise_dominance_probability,
    rank_quantile_interval,
    sampled_average_ranks,
    topk_inclusion_probability,
)


def test_sampled_average_ranks_assign_midrank_to_exact_ties() -> None:
    scores = torch.tensor([[1.0, 1.0, 0.0]])
    ranks = sampled_average_ranks(scores)
    torch.testing.assert_close(ranks, torch.tensor([[1.5, 1.5, 3.0]]))


def test_topk_probability_reports_budget_stability() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.8, 0.9, 0.1],
            [0.9, 0.2, 0.8],
            [0.9, 0.1, 0.8],
        ]
    )
    p = topk_inclusion_probability(scores, 1)
    torch.testing.assert_close(p, torch.tensor([0.75, 0.25, 0.0]))


def test_pairwise_dominance_is_complementary_and_ties_are_half() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.8, 0.8, 0.2],
        ]
    )
    p = pairwise_dominance_probability(scores)
    torch.testing.assert_close(torch.diag(p), torch.full((3,), 0.5))
    torch.testing.assert_close(p + p.T, torch.ones_like(p))


def test_rank_quantile_interval_collapses_for_stable_order() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.1],
            [0.95, 0.7, 0.0],
            [0.85, 0.75, 0.2],
        ]
    )
    low, high = rank_quantile_interval(scores)
    torch.testing.assert_close(low, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(high, torch.tensor([1.0, 2.0, 3.0]))
