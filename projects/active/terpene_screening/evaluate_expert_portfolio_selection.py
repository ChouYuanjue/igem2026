from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/terpene_cage_neural_common_reservoir_specialists_v1/query_metrics.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_expert_portfolio_selection"


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    token = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    local = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")
    return np.random.default_rng(local)


def choose_method(tune: pd.DataFrame, methods: list[str], metric: str) -> str:
    scored = []
    for method in methods:
        group = tune[tune["method"].eq(method)]
        if group.empty:
            continue
        scored.append((float(group[metric].mean()), float(group["reciprocal_rank"].mean()), method))
    if not scored:
        raise ValueError("No candidate methods available in tuning split")
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][2]


def evaluate_selection(
    query_metrics: pd.DataFrame,
    *,
    candidate_methods: list[str],
    repeats: int,
    seed: int,
    tune_fraction: float,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_cols = ["scenario", "direction", "n_seed"]
    for scenario_key, frame in query_metrics.groupby(scenario_cols, sort=True):
        scenario, direction, n_seed = scenario_key
        available = sorted(set(frame["method"].astype(str)) & set(candidate_methods))
        if not available:
            continue
        query_ids = np.asarray(sorted(frame["query_id"].astype(str).unique()))
        if len(query_ids) < 4:
            continue
        for metric in metrics:
            for rep in range(repeats):
                rng = stable_rng(seed, scenario, direction, n_seed, metric, rep)
                perm = rng.permutation(query_ids)
                n_tune = min(len(perm) - 1, max(1, int(round(len(perm) * tune_fraction))))
                tune_ids = set(perm[:n_tune]); test_ids = set(perm[n_tune:])
                tune = frame[frame["query_id"].astype(str).isin(tune_ids)]
                test = frame[frame["query_id"].astype(str).isin(test_ids)]
                selected = choose_method(tune, available, metric)
                selected_test = test[test["method"].eq(selected)]
                for baseline in ["direct:legacy", "pure_cage"]:
                    base_test = test[test["method"].eq(baseline)]
                    if base_test.empty:
                        continue
                    rows.append({
                        "scenario": scenario,
                        "direction": direction,
                        "n_seed": int(n_seed),
                        "selection_metric": metric,
                        "repeat": rep,
                        "selected_method": selected,
                        "baseline": baseline,
                        "n_tune_queries": len(tune_ids),
                        "n_test_queries": len(test_ids),
                        "selected_test_score": float(selected_test[metric].mean()),
                        "baseline_test_score": float(base_test[metric].mean()),
                        "delta": float(selected_test[metric].mean() - base_test[metric].mean()),
                    })
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["scenario", "direction", "n_seed", "selection_metric", "baseline"]
    for key, group in records.groupby(keys, sort=True):
        scenario, direction, n_seed, metric, baseline = key
        counts = group["selected_method"].value_counts()
        rows.append({
            "scenario": scenario,
            "direction": direction,
            "n_seed": int(n_seed),
            "selection_metric": metric,
            "baseline": baseline,
            "repeats": int(len(group)),
            "mean_selected_test_score": float(group["selected_test_score"].mean()),
            "mean_baseline_test_score": float(group["baseline_test_score"].mean()),
            "mean_delta": float(group["delta"].mean()),
            "median_delta": float(group["delta"].median()),
            "improvement_probability": float((group["delta"] > 0).mean()),
            "non_degradation_probability": float((group["delta"] >= 0).mean()),
            "q05_delta": float(group["delta"].quantile(0.05)),
            "q95_delta": float(group["delta"].quantile(0.95)),
            "modal_method": str(counts.index[0]),
            "modal_method_fraction": float(counts.iloc[0] / counts.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated query-disjoint validation of route-level expert portfolio selection.")
    parser.add_argument("--query-metrics", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method-prefix", action="append", default=["direct:", "rrf:direct+seed:"])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--metrics", default="reciprocal_rank,hit_at_1,hit_at_3,hit_at_5,hit_at_10,hit_at_20")
    args = parser.parse_args()
    if args.repeats <= 0 or not 0 < args.tune_fraction < 1:
        raise ValueError("repeats must be positive and tune-fraction in (0,1)")
    query = pd.read_csv(args.query_metrics)
    methods = sorted({
        value for value in query["method"].astype(str).unique()
        if any(value.startswith(prefix) for prefix in args.method_prefix)
    })
    # Pure CAGE is always reported as a baseline, not allowed to disappear from the comparison.
    if "direct:legacy" not in methods:
        raise ValueError("direct:legacy must be present among candidate methods")
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    records = evaluate_selection(
        query,
        candidate_methods=methods,
        repeats=args.repeats,
        seed=args.seed,
        tune_fraction=args.tune_fraction,
        metrics=metrics,
    )
    summary = summarize(records)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "selection_runs.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    payload = {
        "protocol": "repeated_query_disjoint_route_selection",
        "query_metrics": str(args.query_metrics.resolve()),
        "candidate_methods": methods,
        "pure_cage_baseline": True,
        "repeats": args.repeats,
        "tune_fraction": args.tune_fraction,
        "selection_metrics": metrics,
        "constraint": "expert is selected using only tuning-query aggregate performance for a route cell; no test-query labels enter selection",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
