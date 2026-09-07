from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_expert_portfolio_selection import stable_rng  # noqa: E402

DEFAULT_METRICS = (
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
)


def choose_joint_method(
    tune: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    *,
    backbone: str = "direct:legacy",
    max_guard_drop: float = 0.0,
) -> tuple[str, dict[str, float]]:
    """Choose one expert/fusion by joint rank subject to per-metric guards.

    Selection uses tune queries only.  Every guarded metric must be no worse than
    the backbone by more than ``max_guard_drop``.  Among feasible methods we use
    mean percentile rank across metrics, avoiding arbitrary metric-scale weights.
    If no non-backbone method is feasible, the backbone is the safe fallback.
    """
    means = tune[tune["method"].isin(methods)].groupby("method")[metrics].mean()
    if backbone not in means.index:
        raise ValueError(f"backbone {backbone!r} is unavailable in tuning split")
    base = means.loc[backbone]
    deltas = means.subtract(base, axis="columns")
    feasible = deltas.index[(deltas >= -float(max_guard_drop)).all(axis=1)].tolist()
    if not feasible:
        return backbone, {"joint_score": 0.0, "min_guard_delta": 0.0, "feasible_count": 0.0}
    ranks = means.loc[feasible, metrics].rank(axis=0, method="average", pct=True, ascending=True)
    joint = ranks.mean(axis=1)
    min_delta = deltas.loc[feasible, metrics].min(axis=1)
    rr_delta = deltas.loc[feasible, "reciprocal_rank"] if "reciprocal_rank" in metrics else pd.Series(0.0, index=feasible)
    scored = pd.DataFrame({"joint_score": joint, "min_guard_delta": min_delta, "rr_delta": rr_delta})
    scored["method_name"] = scored.index.astype(str)
    scored = scored.sort_values(
        ["joint_score", "min_guard_delta", "rr_delta", "method_name"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = str(scored.index[0])
    row = scored.iloc[0]
    return selected, {
        "joint_score": float(row["joint_score"]),
        "min_guard_delta": float(row["min_guard_delta"]),
        "feasible_count": float(len(feasible)),
    }


def evaluate_joint_selection(
    query_metrics: pd.DataFrame,
    *,
    candidate_methods: list[str],
    repeats: int,
    seed: int,
    tune_fraction: float,
    metrics: list[str],
    backbone: str,
    max_guard_drop: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario_cols = ["scenario", "direction", "n_seed"]
    for scenario_key, frame in query_metrics.groupby(scenario_cols, sort=True):
        scenario, direction, n_seed = scenario_key
        available = sorted(set(frame["method"].astype(str)) & set(candidate_methods))
        if backbone not in available:
            continue
        query_ids = np.asarray(sorted(frame["query_id"].astype(str).unique()))
        if len(query_ids) < 4:
            continue
        for rep in range(repeats):
            rng = stable_rng(seed, scenario, direction, n_seed, "joint", rep)
            perm = rng.permutation(query_ids)
            n_tune = min(len(perm) - 1, max(1, int(round(len(perm) * tune_fraction))))
            tune_ids = set(perm[:n_tune]); test_ids = set(perm[n_tune:])
            tune = frame[frame["query_id"].astype(str).isin(tune_ids)]
            test = frame[frame["query_id"].astype(str).isin(test_ids)]
            selected, diagnostics = choose_joint_method(
                tune,
                available,
                metrics,
                backbone=backbone,
                max_guard_drop=max_guard_drop,
            )
            selected_test = test[test["method"].eq(selected)]
            if selected_test.empty:
                raise ValueError(f"selected method {selected!r} absent from held-out split")
            for baseline in [backbone, "pure_cage"]:
                base_test = test[test["method"].eq(baseline)]
                if base_test.empty:
                    continue
                for metric in metrics:
                    selected_score = float(selected_test[metric].mean())
                    baseline_score = float(base_test[metric].mean())
                    rows.append({
                        "scenario": scenario,
                        "direction": direction,
                        "n_seed": int(n_seed),
                        "repeat": rep,
                        "selected_method": selected,
                        "baseline": baseline,
                        "metric": metric,
                        "n_tune_queries": len(tune_ids),
                        "n_test_queries": len(test_ids),
                        "tune_joint_score": diagnostics["joint_score"],
                        "tune_min_guard_delta": diagnostics["min_guard_delta"],
                        "tune_feasible_count": int(diagnostics["feasible_count"]),
                        "selected_test_score": selected_score,
                        "baseline_test_score": baseline_score,
                        "delta": selected_score - baseline_score,
                    })
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if records.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["scenario", "direction", "n_seed", "baseline", "metric"]
    for key, group in records.groupby(keys, sort=True):
        scenario, direction, n_seed, baseline, metric = key
        rows.append({
            "scenario": scenario,
            "direction": direction,
            "n_seed": int(n_seed),
            "baseline": baseline,
            "metric": metric,
            "repeats": int(group["repeat"].nunique()),
            "mean_selected_test_score": float(group["selected_test_score"].mean()),
            "mean_baseline_test_score": float(group["baseline_test_score"].mean()),
            "mean_delta": float(group["delta"].mean()),
            "median_delta": float(group["delta"].median()),
            "improvement_probability": float((group["delta"] > 0).mean()),
            "non_degradation_probability": float((group["delta"] >= 0).mean()),
            "q05_delta": float(group["delta"].quantile(0.05)),
            "q95_delta": float(group["delta"].quantile(0.95)),
        })
    summary = pd.DataFrame(rows)

    joint_rows: list[dict[str, object]] = []
    rep_keys = ["scenario", "direction", "n_seed", "baseline", "repeat"]
    per_rep = records.groupby(rep_keys, sort=True)
    for key, group in per_rep:
        scenario, direction, n_seed, baseline, rep = key
        metric_delta = group.set_index("metric")["delta"].reindex(metrics)
        selected = str(group["selected_method"].iloc[0])
        joint_rows.append({
            "scenario": scenario,
            "direction": direction,
            "n_seed": int(n_seed),
            "baseline": baseline,
            "repeat": int(rep),
            "selected_method": selected,
            "all_metrics_non_degraded": bool((metric_delta >= 0).all()),
            "all_metrics_improved": bool((metric_delta > 0).all()),
            "worst_metric_delta": float(metric_delta.min()),
            "mean_metric_delta": float(metric_delta.mean()),
        })
    joint = pd.DataFrame(joint_rows)
    return summary, joint


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated query-disjoint, multi-metric constrained expert portfolio selection.")
    parser.add_argument("--query-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-prefix", action="append", default=[])
    parser.add_argument("--backbone", default="direct:legacy")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--max-guard-drop", type=float, default=0.0)
    args = parser.parse_args()
    if args.repeats <= 0 or not 0 < args.tune_fraction < 1:
        raise ValueError("repeats must be positive and tune-fraction in (0,1)")
    query = pd.read_csv(args.query_metrics)
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    prefixes = args.method_prefix or ["direct:", "rrf:cage+", "rrf:direct+seed:"]
    methods = sorted({
        value for value in query["method"].astype(str).unique()
        if any(value.startswith(prefix) for prefix in prefixes)
    })
    if args.backbone not in methods:
        raise ValueError(f"backbone {args.backbone!r} must be among candidate methods")
    if "pure_cage" not in set(query["method"].astype(str)):
        raise ValueError("pure_cage is mandatory as EnzymeCAGE baseline")
    records = evaluate_joint_selection(
        query,
        candidate_methods=methods,
        repeats=args.repeats,
        seed=args.seed,
        tune_fraction=args.tune_fraction,
        metrics=metrics,
        backbone=args.backbone,
        max_guard_drop=args.max_guard_drop,
    )
    summary, joint = summarize(records, metrics)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "selection_metric_runs.csv", index=False)
    summary.to_csv(output / "metric_summary.csv", index=False)
    joint.to_csv(output / "joint_runs.csv", index=False)
    payload = {
        "protocol": "repeated_query_disjoint_joint_metric_guarded_selection",
        "query_metrics": str(args.query_metrics.resolve()),
        "candidate_methods": methods,
        "backbone": args.backbone,
        "mandatory_baseline": "EnzymeCAGE/pure_cage",
        "metrics": metrics,
        "max_guard_drop": args.max_guard_drop,
        "target_test_labels_used_for_selection": False,
        "selection_rule": "all tune metrics must pass backbone guard; choose highest mean cross-metric percentile rank; fallback backbone",
        "repeats": args.repeats,
        "tune_fraction": args.tune_fraction,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    if not joint.empty:
        print("\nJoint held-out non-degradation rates vs backbone:")
        print(joint[joint["baseline"].eq(args.backbone)].groupby(["scenario", "direction", "n_seed"])[["all_metrics_non_degraded", "all_metrics_improved", "worst_metric_delta", "mean_metric_delta"]].mean().to_string())


if __name__ == "__main__":
    main()
