from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RETENTION_STRATA = ("historical_training_pair", "project_catalog")


def _delta_violations(frame: pd.DataFrame, *, max_drop: float, context_columns: tuple[str, ...]) -> tuple[list[dict[str, object]], float]:
    delta_columns = [column for column in frame.columns if column.startswith("delta_")]
    if not delta_columns:
        raise ValueError("comparison has no delta_* columns")
    violations: list[dict[str, object]] = []
    worst = float("inf")
    for _, row in frame.iterrows():
        for column in delta_columns:
            value = float(row[column])
            worst = min(worst, value)
            if value < -max_drop:
                item: dict[str, object] = {
                    "metric": column.removeprefix("delta_"),
                    "delta": value,
                    "allowed_drop": float(max_drop),
                }
                for key in context_columns:
                    if key in row:
                        item[key] = str(row[key])
                violations.append(item)
    return violations, worst


def evaluate_pareto(
    known_summary: dict[str, object],
    known_comparison: pd.DataFrame,
    current_comparison: pd.DataFrame,
    *,
    max_current_drop: float = 0.005,
    max_known_drop: float = 0.01,
    min_primary_hit10_ratio: float = 1.25,
    min_primary_mrr_ratio: float = 1.10,
    min_primary_hit20_ratio: float = 1.0,
    min_secondary_hit10_ratio: float = 1.0,
    primary_direction: str = "reaction_to_enzyme",
    secondary_direction: str = "enzyme_to_reaction",
) -> dict[str, object]:
    for name, value in [("max_current_drop", max_current_drop), ("max_known_drop", max_known_drop)]:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    for name, value in [
        ("min_primary_hit10_ratio", min_primary_hit10_ratio),
        ("min_primary_mrr_ratio", min_primary_mrr_ratio),
        ("min_primary_hit20_ratio", min_primary_hit20_ratio),
        ("min_secondary_hit10_ratio", min_secondary_hit10_ratio),
    ]:
        if value < 1.0:
            raise ValueError(f"{name} must be >= 1")

    current_violations, current_worst = _delta_violations(
        current_comparison, max_drop=max_current_drop, context_columns=("direction", "evaluation_level")
    )
    retained = known_comparison[
        known_comparison.get("stratum", pd.Series(index=known_comparison.index, dtype=str)).isin(DEFAULT_RETENTION_STRATA)
    ].copy()
    if retained.empty:
        raise ValueError("known comparison has no historical/project retention strata")
    known_violations, known_worst = _delta_violations(
        retained, max_drop=max_known_drop, context_columns=("direction", "stratum")
    )

    unseen = known_summary.get("unseen_directional_gain") or {}
    if not isinstance(unseen, dict):
        unseen = {}

    def ratio(direction: str, metric: str) -> float | None:
        payload = unseen.get(direction)
        if not isinstance(payload, dict):
            return None
        metric_payload = payload.get(metric)
        if not isinstance(metric_payload, dict) or metric_payload.get("ratio") is None:
            return None
        return float(metric_payload["ratio"])

    gains = {
        "primary_hit10_ratio": ratio(primary_direction, "hit_at_10"),
        "primary_mrr_ratio": ratio(primary_direction, "reciprocal_rank"),
        "primary_hit20_ratio": ratio(primary_direction, "hit_at_20"),
        "secondary_hit10_ratio": ratio(secondary_direction, "hit_at_10"),
    }
    requirements = {
        "primary_hit10_ratio": float(min_primary_hit10_ratio),
        "primary_mrr_ratio": float(min_primary_mrr_ratio),
        "primary_hit20_ratio": float(min_primary_hit20_ratio),
        "secondary_hit10_ratio": float(min_secondary_hit10_ratio),
    }
    gain_violations = [
        {"metric": key, "ratio": gains[key], "required_ratio": required}
        for key, required in requirements.items()
        if gains[key] is None or gains[key] < required
    ]
    passed = not current_violations and not known_violations and not gain_violations
    return {
        "pareto_passed": passed,
        "policy": "small_retention_tradeoff_for_material_unseen_gain",
        "primary_direction": primary_direction,
        "secondary_direction": secondary_direction,
        "retention": {
            "max_current_drop": float(max_current_drop),
            "max_known_drop": float(max_known_drop),
            "current_worst_delta": current_worst,
            "known_worst_delta": known_worst,
            "current_violations": current_violations,
            "known_violations": known_violations,
        },
        "gains": gains,
        "gain_requirements": requirements,
        "gain_violations": gain_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto gate for broad general-evidence continuation with explicitly bounded retention tradeoff.")
    parser.add_argument("--known-comparison-dir", type=Path, required=True)
    parser.add_argument("--current-retention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-current-drop", type=float, default=0.005)
    parser.add_argument("--max-known-drop", type=float, default=0.01)
    parser.add_argument("--min-primary-hit10-ratio", type=float, default=1.25)
    parser.add_argument("--min-primary-mrr-ratio", type=float, default=1.10)
    parser.add_argument("--min-primary-hit20-ratio", type=float, default=1.0)
    parser.add_argument("--min-secondary-hit10-ratio", type=float, default=1.0)
    args = parser.parse_args()

    known_summary = json.loads((args.known_comparison_dir / "summary.json").read_text(encoding="utf-8"))
    known_comparison = pd.read_csv(args.known_comparison_dir / "comparison.csv")
    current_comparison = pd.read_csv(args.current_retention_dir / "comparison.csv")
    result = evaluate_pareto(
        known_summary,
        known_comparison,
        current_comparison,
        max_current_drop=args.max_current_drop,
        max_known_drop=args.max_known_drop,
        min_primary_hit10_ratio=args.min_primary_hit10_ratio,
        min_primary_mrr_ratio=args.min_primary_mrr_ratio,
        min_primary_hit20_ratio=args.min_primary_hit20_ratio,
        min_secondary_hit10_ratio=args.min_secondary_hit10_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pareto_passed"] else 2)


if __name__ == "__main__":
    main()
