from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = (1, 3, 5, 10, 20)
DEFAULT_TOP_PERCENTS = (0.01, 0.02, 0.03, 0.05)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dcg(labels: np.ndarray, k: int) -> float:
    values = labels[: min(k, len(labels))].astype(float)
    if not len(values):
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2, dtype=float))
    return float(np.sum((np.power(2.0, values) - 1.0) / discounts))


def _average_precision(labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives <= 0:
        return 0.0
    positions = np.flatnonzero(labels > 0) + 1
    precisions = np.arange(1, len(positions) + 1, dtype=float) / positions
    return float(precisions.sum() / positives)


def _enrichment_factor(labels: np.ndarray, *, top_percent: float) -> float:
    positives = int(labels.sum())
    if positives <= 0 or len(labels) == 0:
        return 0.0
    # Matches EnzymeCAGE evaluate.py: percentage-based EF has a minimum panel of 5.
    top_k = max(int(top_percent * len(labels)), 5)
    top_k = min(top_k, len(labels))
    expected = positives * (top_k / len(labels))
    return float(labels[:top_k].sum() / expected) if expected > 0 else 0.0


def evaluate_ranking_frame(
    frame: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
    score_col: str = "score",
    label_col: str = "label",
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    top_percents: Iterable[float] = DEFAULT_TOP_PERCENTS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {query_col, candidate_col, score_col, label_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ranking frame missing columns: {sorted(missing)}")
    data = frame.copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if data[score_col].isna().any():
        raise ValueError("Ranking frame contains non-numeric/missing scores")
    if data.duplicated([query_col, candidate_col]).any():
        raise ValueError("Ranking frame contains duplicate query-candidate rows")

    budgets = tuple(sorted({int(value) for value in budgets}))
    top_percents = tuple(sorted({float(value) for value in top_percents}))
    rows: list[dict[str, object]] = []
    for query_id, group in data.groupby(query_col, sort=True):
        group = group.sort_values([score_col, candidate_col], ascending=[False, True], kind="mergesort")
        labels = group[label_col].to_numpy(dtype=np.int8)
        positives = int(labels.sum())
        hit_positions = np.flatnonzero(labels > 0) + 1
        best_rank = int(hit_positions[0]) if len(hit_positions) else None
        row: dict[str, object] = {
            "query_id": str(query_id),
            "candidate_count": int(len(group)),
            "positive_count": positives,
            "has_positive": positives > 0,
            "best_positive_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "average_precision": _average_precision(labels),
            "dcg_at_10": _dcg(labels, 10),
        }
        ideal = np.sort(labels)[::-1]
        for k in (10, 20):
            ideal_dcg = _dcg(ideal, k)
            row[f"ndcg_at_{k}"] = _dcg(labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0
        for k in budgets:
            row[f"hit_at_{k}"] = float(best_rank is not None and best_rank <= k)
            row[f"positive_recall_at_{k}"] = float(labels[:k].sum() / positives) if positives else 0.0
        for percent in top_percents:
            # Matches EnzymeCAGE external-test evaluator: floor(n*p), minimum 1.
            k = max(1, int(len(labels) * percent))
            row[f"success_at_{percent:g}_fraction"] = float(labels[:k].sum() > 0)
        for percent in (0.01, 0.02):
            row[f"ef_at_{percent:g}_fraction"] = _enrichment_factor(labels, top_percent=percent)
        rows.append(row)

    per_query = pd.DataFrame(rows)
    if per_query.empty:
        raise ValueError("No ranking queries to evaluate")
    evaluable = per_query[per_query["has_positive"]].copy()
    summary: dict[str, object] = {
        "query_count": int(len(per_query)),
        "evaluable_query_count": int(len(evaluable)),
        "query_positive_coverage": float(per_query["has_positive"].mean()),
        "candidate_rows": int(len(data)),
        "positive_rows": int(data[label_col].sum()),
        "mean_candidate_pool_size": float(per_query["candidate_count"].mean()),
        "median_candidate_pool_size": float(per_query["candidate_count"].median()),
        "mrr": float(per_query["reciprocal_rank"].mean()),
        "map": float(per_query["average_precision"].mean()),
        "top10_dcg": float(per_query["dcg_at_10"].mean()),
        "ndcg_at_10": float(per_query["ndcg_at_10"].mean()),
        "ndcg_at_20": float(per_query["ndcg_at_20"].mean()),
        "top1_percent_ef": float(per_query["ef_at_0.01_fraction"].mean()),
        "top2_percent_ef": float(per_query["ef_at_0.02_fraction"].mean()),
    }
    for k in budgets:
        summary[f"hit_at_{k}"] = float(per_query[f"hit_at_{k}"].mean())
        summary[f"positive_recall_at_{k}"] = float(per_query[f"positive_recall_at_{k}"].mean())
    for percent in top_percents:
        summary[f"success_at_{percent:g}_fraction"] = float(
            per_query[f"success_at_{percent:g}_fraction"].mean()
        )
    return per_query, summary


def audit_exact_overlap(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    *,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
) -> tuple[pd.DataFrame, dict[str, object]]:
    for frame, name in ((train_pairs, "train"), (test_pairs, "test")):
        missing = {query_col, candidate_col} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
    train_q = set(train_pairs[query_col].astype(str))
    train_c = set(train_pairs[candidate_col].astype(str))
    train_p = set(zip(train_pairs[query_col].astype(str), train_pairs[candidate_col].astype(str)))
    audit = test_pairs[[query_col, candidate_col]].copy()
    audit[query_col] = audit[query_col].astype(str)
    audit[candidate_col] = audit[candidate_col].astype(str)
    audit["query_seen"] = audit[query_col].isin(train_q)
    audit["candidate_seen"] = audit[candidate_col].isin(train_c)
    audit["exact_pair_seen"] = [pair in train_p for pair in zip(audit[query_col], audit[candidate_col])]
    summary = {
        "test_pair_rows": int(len(audit)),
        "exact_pair_seen_rows": int(audit["exact_pair_seen"].sum()),
        "exact_pair_seen_fraction": float(audit["exact_pair_seen"].mean()) if len(audit) else 0.0,
        "query_seen_rows": int(audit["query_seen"].sum()),
        "query_seen_fraction": float(audit["query_seen"].mean()) if len(audit) else 0.0,
        "candidate_seen_rows": int(audit["candidate_seen"].sum()),
        "candidate_seen_fraction": float(audit["candidate_seen"].mean()) if len(audit) else 0.0,
        "generalization_claim_safe_exact_pair": bool(not audit["exact_pair_seen"].any()),
    }
    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Common leakage-aware ranking evaluator for enzyme-reaction baselines.")
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-col", default="query_id")
    parser.add_argument("--candidate-col", default="candidate_id")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--train-pairs", type=Path)
    parser.add_argument("--require-clean-exact-pairs", action="store_true")
    args = parser.parse_args()
    ranking = pd.read_csv(args.ranking)
    per_query, metrics = evaluate_ranking_frame(
        ranking, query_col=args.query_col, candidate_col=args.candidate_col,
        score_col=args.score_col, label_col=args.label_col,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(output / "query_metrics.csv", index=False)
    payload: dict[str, object] = {
        "ranking_file": str(args.ranking),
        "ranking_sha256": sha256_file(args.ranking),
        "metrics": metrics,
    }
    if args.train_pairs:
        train = pd.read_csv(args.train_pairs)
        _, leakage = audit_exact_overlap(
            train, ranking[ranking[args.label_col].astype(int).eq(1)],
            query_col=args.query_col, candidate_col=args.candidate_col,
        )
        payload["train_pairs"] = str(args.train_pairs)
        payload["train_pairs_sha256"] = sha256_file(args.train_pairs)
        payload["leakage"] = leakage
        if args.require_clean_exact_pairs and not leakage["generalization_claim_safe_exact_pair"]:
            raise RuntimeError("Exact train-test positive pair overlap detected; generalization claim is blocked")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
