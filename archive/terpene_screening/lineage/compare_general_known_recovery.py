from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("direction", "stratum", "query_id")
METRIC_COLUMNS = (
    "reciprocal_rank",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
)
DEFAULT_GUARD_STRATA = ("historical_training_pair", "project_catalog")


def evaluation_model_signature(evaluation_dir: Path) -> tuple[str, ...]:
    summary_path = evaluation_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing evaluation summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    model_dir = Path(str(payload.get("model_dir", "")))
    if not model_dir.is_dir():
        raise ValueError(f"evaluation model directory is unavailable: {model_dir}")
    checkpoints = tuple(sorted(path.name for path in (model_dir / "models").glob("production_seed*.pt")))
    if not checkpoints:
        raise ValueError(f"no production checkpoints found under {model_dir / 'models'}")
    return checkpoints


def _validate_unique(frame: pd.DataFrame, label: str) -> None:
    missing = [column for column in (*KEY_COLUMNS, "n_positives", *METRIC_COLUMNS) if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    duplicates = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"{label} contains duplicate query keys: {sample}")


def compare_matched_queries(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    guard_strata: tuple[str, ...] = DEFAULT_GUARD_STRATA,
    max_guard_drop: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare two known-recovery runs only when their query populations match exactly."""
    _validate_unique(baseline, "baseline")
    _validate_unique(candidate, "candidate")

    baseline_keys = set(map(tuple, baseline[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    candidate_keys = set(map(tuple, candidate[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    if baseline_keys != candidate_keys:
        missing = sorted(baseline_keys - candidate_keys)[:5]
        extra = sorted(candidate_keys - baseline_keys)[:5]
        raise ValueError(
            "query populations do not match exactly; "
            f"baseline_only={missing}, candidate_only={extra}, "
            f"n_baseline={len(baseline_keys)}, n_candidate={len(candidate_keys)}"
        )

    joined = baseline.merge(
        candidate,
        on=list(KEY_COLUMNS),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    positive_mismatch = joined["n_positives_baseline"].astype(int) != joined["n_positives_candidate"].astype(int)
    if positive_mismatch.any():
        sample = joined.loc[positive_mismatch, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"positive-label counts differ for matched queries: {sample}")

    rows: list[dict[str, object]] = []
    for (direction, stratum), group in joined.groupby(["direction", "stratum"], sort=True):
        row: dict[str, object] = {
            "direction": direction,
            "stratum": stratum,
            "n_queries": int(len(group)),
        }
        for metric in METRIC_COLUMNS:
            before = float(group[f"{metric}_baseline"].mean())
            after = float(group[f"{metric}_candidate"].mean())
            row[f"baseline_{metric}"] = before
            row[f"candidate_{metric}"] = after
            row[f"delta_{metric}"] = after - before
            row[f"ratio_{metric}"] = (after / before) if before != 0 else None
        rows.append(row)
    comparison = pd.DataFrame(rows)

    guard_rows = comparison[comparison["stratum"].isin(guard_strata)]
    violations: list[dict[str, object]] = []
    for _, row in guard_rows.iterrows():
        for metric in METRIC_COLUMNS:
            delta = float(row[f"delta_{metric}"])
            if delta < -float(max_guard_drop):
                violations.append(
                    {
                        "direction": str(row["direction"]),
                        "stratum": str(row["stratum"]),
                        "metric": metric,
                        "delta": delta,
                    }
                )

    unseen = comparison[comparison["stratum"].eq("unseen_to_historical_training")]
    summary: dict[str, object] = {
        "matched_query_keys": int(len(joined)),
        "guard_strata": list(guard_strata),
        "max_guard_drop": float(max_guard_drop),
        "retention_guard_passed": not violations,
        "retention_violations": violations,
        "unseen_directional_gain": {
            str(row["direction"]): {
                metric: {
                    "baseline": float(row[f"baseline_{metric}"]),
                    "candidate": float(row[f"candidate_{metric}"]),
                    "delta": float(row[f"delta_{metric}"]),
                    "ratio": (None if pd.isna(row[f"ratio_{metric}"]) else float(row[f"ratio_{metric}"])),
                }
                for metric in METRIC_COLUMNS
            }
            for _, row in unseen.iterrows()
        },
    }
    return comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict matched-query comparison for broad known-recovery checkpoints.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-strata", default=",".join(DEFAULT_GUARD_STRATA))
    parser.add_argument(
        "--max-guard-drop",
        type=float,
        default=0.0,
        help="Maximum absolute mean drop allowed for every guarded metric; default 0 requires non-degradation.",
    )
    args = parser.parse_args()

    baseline_signature = evaluation_model_signature(args.baseline_dir)
    candidate_signature = evaluation_model_signature(args.candidate_dir)
    if baseline_signature != candidate_signature:
        raise ValueError(
            "model ensemble signatures do not match; "
            f"baseline={baseline_signature}, candidate={candidate_signature}"
        )

    baseline = pd.read_csv(args.baseline_dir / "query_metrics.csv")
    candidate = pd.read_csv(args.candidate_dir / "query_metrics.csv")
    strata = tuple(value.strip() for value in args.guard_strata.split(",") if value.strip())
    comparison, summary = compare_matched_queries(
        baseline,
        candidate,
        guard_strata=strata,
        max_guard_drop=args.max_guard_drop,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    summary["model_signature"] = list(baseline_signature)
    summary["baseline_dir"] = str(args.baseline_dir.resolve())
    summary["candidate_dir"] = str(args.candidate_dir.resolve())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
