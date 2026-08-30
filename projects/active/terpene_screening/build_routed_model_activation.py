from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BACKBONE = "direct:legacy"


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return float((centre - radius) / denom)


def activation_table(
    metric_summary: pd.DataFrame,
    joint_runs: pd.DataFrame,
    *,
    backbone: str = DEFAULT_BACKBONE,
    lower_bound_threshold: float = 0.5,
    min_metric_mean_delta: float = 0.0,
    require_strict_gain: bool = True,
) -> pd.DataFrame:
    keys = ["scenario", "direction", "n_seed"]
    metric = metric_summary[metric_summary["baseline"].eq(backbone)].copy()
    joint = joint_runs[joint_runs["baseline"].eq(backbone)].copy()
    rows: list[dict[str, object]] = []
    for key, group in metric.groupby(keys, sort=True):
        scenario, direction, n_seed = key
        matching = joint
        for column, value in zip(keys, key):
            matching = matching[matching[column].eq(value)]
        repeats = int(matching["repeat"].nunique())
        successes = int(matching.groupby("repeat")["all_metrics_non_degraded"].first().sum()) if repeats else 0
        lower = wilson_lower_bound(successes, repeats)
        deltas = pd.to_numeric(group["mean_delta"], errors="raise")
        worst_mean = float(deltas.min())
        mean_delta = float(deltas.mean())
        positive_metrics = int((deltas > 0).sum())
        metrics_count = int(len(deltas))
        all_metric_guard = bool((deltas >= float(min_metric_mean_delta)).all())
        evidence_guard = bool(lower > float(lower_bound_threshold))
        strict_gain_guard = bool(positive_metrics > 0) if require_strict_gain else True
        active = all_metric_guard and evidence_guard and strict_gain_guard
        methods = matching.groupby("repeat")["selected_method"].first() if repeats else pd.Series(dtype=str)
        modal_method = str(methods.value_counts().index[0]) if len(methods) else backbone
        modal_fraction = float(methods.value_counts().iloc[0] / len(methods)) if len(methods) else 1.0
        reasons: list[str] = []
        if not all_metric_guard:
            reasons.append("one_or_more_mean_metric_deltas_below_guard")
        if not evidence_guard:
            reasons.append("simultaneous_non_degradation_wilson_lower_bound_not_above_half")
        if not strict_gain_guard:
            reasons.append("no_guarded_metric_has_strict_positive_mean_gain")
        rows.append({
            "scenario": scenario,
            "direction": direction,
            "n_seed": int(n_seed),
            "backbone": backbone,
            "active": active,
            "selected_route": modal_method if active else backbone,
            "modal_candidate_method": modal_method,
            "modal_candidate_fraction": modal_fraction,
            "guarded_metrics": metrics_count,
            "positive_mean_gain_metrics": positive_metrics,
            "worst_mean_metric_delta": worst_mean,
            "mean_metric_delta": mean_delta,
            "joint_repeats": repeats,
            "joint_non_degradation_successes": successes,
            "joint_non_degradation_rate": float(successes / repeats) if repeats else 0.0,
            "joint_non_degradation_wilson95_lower": lower,
            "activation_reasons": "activated" if active else ";".join(reasons),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a development-only activation guard for the routed Catalyst model.")
    parser.add_argument("--metric-summary", type=Path, required=True)
    parser.add_argument("--joint-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--lower-bound-threshold", type=float, default=0.5)
    parser.add_argument("--min-metric-mean-delta", type=float, default=0.0)
    args = parser.parse_args()
    metric = pd.read_csv(args.metric_summary)
    joint = pd.read_csv(args.joint_runs)
    table = activation_table(
        metric,
        joint,
        backbone=args.backbone,
        lower_bound_threshold=args.lower_bound_threshold,
        min_metric_mean_delta=args.min_metric_mean_delta,
    )
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "activation.csv", index=False)
    payload = {
        "protocol": "development_only_joint_metric_activation_guard",
        "backbone": args.backbone,
        "target_test_labels_used": False,
        "rules": {
            "all_guarded_metric_mean_deltas_gte": args.min_metric_mean_delta,
            "at_least_one_guarded_metric_strictly_improves": True,
            "simultaneous_non_degradation_wilson95_lower_gt": args.lower_bound_threshold,
        },
        "active_cells": table.loc[table["active"], ["scenario", "direction", "n_seed", "selected_route"]].to_dict("records"),
        "inactive_cells": table.loc[~table["active"], ["scenario", "direction", "n_seed", "activation_reasons"]].to_dict("records"),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
