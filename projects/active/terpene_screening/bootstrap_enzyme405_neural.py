from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def native_query_metrics(frame: pd.DataFrame, score_col: str = "neural_score") -> pd.DataFrame:
    data = frame.drop_duplicates(["reaction_id", "protein_id"], keep="first").copy()
    rows = []
    for query_id, group in data.groupby("reaction_id", sort=False):
        ranked = group.sort_values(score_col, ascending=False, kind="mergesort")
        labels = ranked["label"].astype(int).to_numpy()
        hits = np.flatnonzero(labels > 0) + 1
        best = int(hits[0]) if len(hits) else -1
        values = labels[: min(10, len(labels))].astype(float)
        discounts = np.log2(np.arange(2, len(values) + 2, dtype=float))
        dcg10 = float(np.sum((np.power(2.0, values) - 1.0) / discounts))
        total_active = int(labels.sum())
        native_ef1 = 0.0
        if total_active > 0:
            topk = max(int(0.01 * len(labels)), 5)
            native_ef1 = float(int(labels[:topk].sum()) / (total_active * 0.01))
        rows.append({
            "query_id": str(query_id),
            "native_sr1": float(0 < best <= 1),
            "native_sr3": float(0 < best <= 3),
            "native_sr5": float(0 < best <= 5),
            "native_sr10": float(0 < best <= 10),
            "native_dcg10": dcg10,
            "native_ef1": native_ef1,
        })
    return pd.DataFrame(rows)


def bootstrap_means(frame: pd.DataFrame, columns: list[str], *, samples: int, seed: int) -> dict[str, dict[str, float]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    values = frame[columns].to_numpy(dtype=float)
    n = len(values)
    if n <= 1:
        raise ValueError("at least two queries are required")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(columns)), dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        idx = rng.integers(0, n, size=(stop - start, n))
        draws[start:stop] = values[idx].mean(axis=1)
    result = {}
    for i, column in enumerate(columns):
        lo, hi = np.quantile(draws[:, i], [0.025, 0.975])
        result[column] = {
            "estimate": float(values[:, i].mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Query bootstrap intervals for pure neural Enzyme-405 metrics.")
    parser.add_argument("--pair-scores", type=Path, required=True)
    parser.add_argument("--query-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    pair = pd.read_csv(args.pair_scores, dtype={"reaction_id": str, "protein_id": str})
    common = pd.read_csv(args.query_metrics)
    common = common[common["direction"].eq("reaction_to_enzyme")].copy()
    native = native_query_metrics(pair)
    merged = native.merge(common, on="query_id", how="inner", validate="one_to_one")
    if len(merged) != pair["reaction_id"].nunique():
        raise ValueError("native/common query metrics do not cover identical reaction queries")
    columns = [
        "native_sr1", "native_sr3", "native_sr5", "native_sr10", "native_dcg10", "native_ef1",
        "reciprocal_rank", "average_precision", "roc_auc", "ndcg_at_10",
    ]
    intervals = bootstrap_means(merged, columns, samples=args.samples, seed=args.seed)
    payload = {
        "method": "nonparametric_query_bootstrap",
        "score": "neural_score_only",
        "queries": int(len(merged)),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "interval": "percentile 95%",
        "metrics": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
