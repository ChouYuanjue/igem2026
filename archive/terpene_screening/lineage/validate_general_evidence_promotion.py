from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def evaluate_promotion(
    known_summary: dict[str, object],
    current_comparison: pd.DataFrame,
    *,
    max_current_drop: float = 0.0,
    min_unseen_hit10_ratio: float = 1.0,
) -> dict[str, object]:
    if max_current_drop < 0:
        raise ValueError("max_current_drop must be non-negative")
    if min_unseen_hit10_ratio < 1.0:
        raise ValueError("min_unseen_hit10_ratio must be >= 1")

    current_delta_columns = [column for column in current_comparison.columns if column.startswith("delta_")]
    if not current_delta_columns:
        raise ValueError("current retention comparison has no delta_* columns")

    current_violations: list[dict[str, object]] = []
    for _, row in current_comparison.iterrows():
        for column in current_delta_columns:
            value = float(row[column])
            if value < -max_current_drop:
                current_violations.append(
                    {
                        "direction": str(row.get("direction", "")),
                        "evaluation_level": str(row.get("evaluation_level", "")),
                        "metric": column.removeprefix("delta_"),
                        "delta": value,
                    }
                )

    unseen = known_summary.get("unseen_directional_gain") or {}
    gain_violations: list[dict[str, object]] = []
    for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
        payload = unseen.get(direction) if isinstance(unseen, dict) else None
        if not isinstance(payload, dict) or "hit_at_10" not in payload:
            gain_violations.append({"direction": direction, "reason": "missing_hit_at_10"})
            continue
        hit10 = payload["hit_at_10"]
        ratio = hit10.get("ratio") if isinstance(hit10, dict) else None
        if ratio is None or float(ratio) < min_unseen_hit10_ratio:
            gain_violations.append(
                {
                    "direction": direction,
                    "metric": "hit_at_10",
                    "ratio": None if ratio is None else float(ratio),
                    "required_ratio": float(min_unseen_hit10_ratio),
                }
            )

    known_pass = bool(known_summary.get("retention_guard_passed"))
    passed = known_pass and not current_violations and not gain_violations
    return {
        "promotion_passed": passed,
        "known_recovery_retention_passed": known_pass,
        "known_recovery_retention_violations": known_summary.get("retention_violations", []),
        "current_retention_passed": not current_violations,
        "current_retention_violations": current_violations,
        "broad_gain_passed": not gain_violations,
        "broad_gain_violations": gain_violations,
        "max_current_drop": float(max_current_drop),
        "min_unseen_hit10_ratio": float(min_unseen_hit10_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified promotion gate for general-evidence continuation checkpoints.")
    parser.add_argument("--known-comparison-dir", type=Path, required=True)
    parser.add_argument("--current-retention-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-current-drop", type=float, default=0.0)
    parser.add_argument("--min-unseen-hit10-ratio", type=float, default=1.0)
    args = parser.parse_args()

    known = json.loads((args.known_comparison_dir / "summary.json").read_text(encoding="utf-8"))
    current = pd.read_csv(args.current_retention_dir / "comparison.csv")
    result = evaluate_promotion(
        known,
        current,
        max_current_drop=args.max_current_drop,
        min_unseen_hit10_ratio=args.min_unseen_hit10_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["promotion_passed"] else 2)


if __name__ == "__main__":
    main()
