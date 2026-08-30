from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from projects.active.terpene_screening.fair_benchmark import (
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
)


def candidate_ranking_context(candidate_ids: list[str]) -> tuple[dict[str, int], np.ndarray]:
    """Return candidate row lookup and deterministic lexical tie-break ranks."""
    index = {value: row for row, value in enumerate(candidate_ids)}
    if len(index) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    ids_array = np.asarray(candidate_ids, dtype=object)
    lexical_order = np.empty(len(candidate_ids), dtype=np.int64)
    for rank, row in enumerate(np.argsort(ids_array, kind="mergesort")):
        lexical_order[int(row)] = rank
    return index, lexical_order


def positive_rank_map(
    scores: np.ndarray,
    candidate_ids: list[str],
    positive_ids: set[str],
    *,
    candidate_index: dict[str, int] | None = None,
    lexical_order: np.ndarray | None = None,
) -> dict[str, int]:
    """Return exact rank for every labelled positive under deterministic tie semantics."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(candidate_ids):
        raise ValueError("scores must be one-dimensional and align with candidate_ids")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    if not positive_ids:
        raise ValueError("positive_ids must be non-empty")
    if candidate_index is None or lexical_order is None:
        candidate_index, lexical_order = candidate_ranking_context(candidate_ids)
    result: dict[str, int] = {}
    for candidate_id in sorted(positive_ids):
        row = candidate_index.get(candidate_id)
        if row is None:
            continue
        score = values[row]
        better = int(np.count_nonzero(values > score))
        tied_before = int(
            np.count_nonzero((values == score) & (lexical_order < lexical_order[row]))
        )
        result[candidate_id] = better + tied_before + 1
    if not result:
        raise ValueError("none of the positives are in candidate_ids")
    return result


def positive_ranks(
    scores: np.ndarray,
    candidate_ids: list[str],
    positive_ids: set[str],
    *,
    candidate_index: dict[str, int] | None = None,
    lexical_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact positive ranks under score-descending, candidate-id-ascending ordering.

    This avoids sorting the entire candidate universe. Runtime is O(P*N), where P is
    the number of labelled positives for the query and is typically very small for
    enzyme-reaction association benchmarks.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(candidate_ids):
        raise ValueError("scores must be one-dimensional and align with candidate_ids")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    if not positive_ids:
        raise ValueError("positive_ids must be non-empty")

    if candidate_index is None or lexical_order is None:
        candidate_index, lexical_order = candidate_ranking_context(candidate_ids)
    positive_rows = np.asarray(
        [candidate_index[value] for value in positive_ids if value in candidate_index],
        dtype=np.int64,
    )
    if len(positive_rows) == 0:
        raise ValueError("none of the positives are in candidate_ids")

    ranks: list[int] = []
    for row in positive_rows:
        score = values[row]
        better = int(np.count_nonzero(values > score))
        tied_before = int(
            np.count_nonzero((values == score) & (lexical_order < lexical_order[row]))
        )
        ranks.append(better + tied_before + 1)
    return np.sort(np.asarray(ranks, dtype=np.int64)), positive_rows


def _average_precision_from_ranks(ranks: np.ndarray) -> float:
    if len(ranks) == 0:
        return 0.0
    numerators = np.arange(1, len(ranks) + 1, dtype=float)
    return float(np.mean(numerators / ranks.astype(float)))


def _dcg_from_ranks(ranks: np.ndarray, k: int) -> float:
    selected = ranks[ranks <= min(int(k), np.iinfo(np.int64).max)]
    if len(selected) == 0:
        return 0.0
    return float(np.sum(1.0 / np.log2(selected.astype(float) + 1.0)))


def _ideal_dcg(positive_count: int, k: int) -> float:
    n = min(int(positive_count), int(k))
    if n <= 0:
        return 0.0
    positions = np.arange(1, n + 1, dtype=float)
    return float(np.sum(1.0 / np.log2(positions + 1.0)))


def _roc_auc_from_scores(scores: np.ndarray, positive_rows: np.ndarray) -> float | None:
    """Binary AUROC with average score ranks for ties, without full sorting."""
    values = np.asarray(scores, dtype=float)
    positives = int(len(positive_rows))
    negatives = int(len(values) - positives)
    if positives <= 0 or negatives <= 0:
        return None

    rank_sum = 0.0
    for row in positive_rows:
        score = values[row]
        less = int(np.count_nonzero(values < score))
        tied = int(np.count_nonzero(values == score))
        rank_sum += less + (tied + 1.0) / 2.0
    u = rank_sum - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def evaluate_full_candidate_scores(
    scores: np.ndarray,
    candidate_ids: list[str],
    positive_ids: set[str],
    *,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    top_percents: Iterable[float] = DEFAULT_TOP_PERCENTS,
    candidate_index: dict[str, int] | None = None,
    lexical_order: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Compute the common benchmark metrics from one complete candidate score vector.

    Metric semantics intentionally match ``fair_benchmark.evaluate_ranking_frame``.
    ReactZyme's native mean-positive rank/reciprocal-rank quantities are kept separate
    from standard first-positive MRR.
    """
    budgets = tuple(sorted({int(value) for value in budgets}))
    top_percents = tuple(sorted({float(value) for value in top_percents}))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    if any(value <= 0 or value > 1 for value in top_percents):
        raise ValueError("top_percents must lie in (0, 1]")

    values = np.asarray(scores, dtype=float)
    ranks, positive_rows = positive_ranks(
        values,
        candidate_ids,
        positive_ids,
        candidate_index=candidate_index,
        lexical_order=lexical_order,
    )
    positive_count = int(len(ranks))
    candidate_count = int(len(values))
    best_rank = int(ranks[0])

    out: dict[str, float | int | None] = {
        "candidate_count": candidate_count,
        "positive_count": positive_count,
        "has_positive": 1,
        "best_positive_rank": best_rank,
        "best_positive_rank_fraction": float(best_rank / candidate_count),
        "mean_positive_rank": float(ranks.mean()),
        "mean_positive_reciprocal_rank": float(np.mean(1.0 / ranks.astype(float))),
        "reciprocal_rank": float(1.0 / best_rank),
        "average_precision": _average_precision_from_ranks(ranks),
        "roc_auc": _roc_auc_from_scores(values, positive_rows),
        "dcg_at_10": _dcg_from_ranks(ranks, 10),
    }

    for k in sorted(set(budgets) | {10, 20}):
        ideal = _ideal_dcg(positive_count, k)
        dcg = _dcg_from_ranks(ranks, k)
        out[f"ndcg_at_{k}"] = float(dcg / ideal) if ideal > 0 else 0.0
    for k in budgets:
        effective_k = min(k, candidate_count)
        hits = int(np.count_nonzero(ranks <= effective_k))
        out[f"hit_at_{k}"] = int(hits > 0)
        out[f"positive_hits_at_{k}"] = hits
        out[f"precision_at_{k}"] = float(hits / effective_k) if effective_k else 0.0
        out[f"positive_recall_at_{k}"] = float(hits / positive_count)
    for percent in top_percents:
        top_k = max(1, int(candidate_count * percent))
        out[f"success_at_{percent:g}_fraction"] = float(np.any(ranks <= top_k))
    for percent in (0.01, 0.02):
        top_k = min(max(int(candidate_count * percent), 5), candidate_count)
        hits = int(np.count_nonzero(ranks <= top_k))
        expected = positive_count * (top_k / candidate_count)
        out[f"ef_at_{percent:g}_fraction"] = float(hits / expected) if expected > 0 else 0.0
    return out


def summarize_query_metrics(
    frame: pd.DataFrame,
    *,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    top_percents: Iterable[float] = DEFAULT_TOP_PERCENTS,
) -> dict[str, float | int | None]:
    if frame.empty:
        raise ValueError("No query metrics to summarize")
    budgets = tuple(sorted({int(value) for value in budgets}))
    top_percents = tuple(sorted({float(value) for value in top_percents}))

    summary: dict[str, float | int | None] = {
        "query_count": int(len(frame)),
        "evaluable_query_count": int(len(frame)),
        "query_positive_coverage": 1.0,
        "candidate_rows": int(frame["candidate_count"].sum()),
        "positive_rows": int(frame["positive_count"].sum()),
        "mean_candidate_pool_size": float(frame["candidate_count"].mean()),
        "median_candidate_pool_size": float(frame["candidate_count"].median()),
        "mrr": float(frame["reciprocal_rank"].mean()),
        "map": float(frame["average_precision"].mean()),
        "macro_roc_auc": (
            float(frame["roc_auc"].dropna().mean()) if frame["roc_auc"].notna().any() else None
        ),
        "roc_auc_query_count": int(frame["roc_auc"].notna().sum()),
        "top10_dcg": float(frame["dcg_at_10"].mean()),
        "ndcg_at_10": float(frame["ndcg_at_10"].mean()),
        "ndcg_at_20": float(frame["ndcg_at_20"].mean()),
        "top1_percent_ef": float(frame["ef_at_0.01_fraction"].mean()),
        "top2_percent_ef": float(frame["ef_at_0.02_fraction"].mean()),
        "median_best_positive_rank": float(frame["best_positive_rank"].median()),
        "mean_positive_rank": float(frame["mean_positive_rank"].mean()),
        "mean_positive_reciprocal_rank": float(frame["mean_positive_reciprocal_rank"].mean()),
        "mean_best_positive_rank_fraction": float(frame["best_positive_rank_fraction"].mean()),
        "median_best_positive_rank_fraction": float(frame["best_positive_rank_fraction"].median()),
    }
    total_positives = int(frame["positive_count"].sum())
    for k in budgets:
        summary[f"hit_at_{k}"] = float(frame[f"hit_at_{k}"].mean())
        summary[f"precision_at_{k}"] = float(frame[f"precision_at_{k}"].mean())
        summary[f"positive_recall_at_{k}"] = float(frame[f"positive_recall_at_{k}"].mean())
        summary[f"micro_positive_recall_at_{k}"] = (
            float(frame[f"positive_hits_at_{k}"].sum() / total_positives)
            if total_positives else 0.0
        )
        summary[f"ndcg_at_{k}"] = float(frame[f"ndcg_at_{k}"].mean())
    for percent in top_percents:
        summary[f"success_at_{percent:g}_fraction"] = float(
            frame[f"success_at_{percent:g}_fraction"].mean()
        )
    return summary
