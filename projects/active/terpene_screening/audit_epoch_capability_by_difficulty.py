from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import summarize_query_metrics
from projects.active.terpene_screening.compare_broad_rhea_full_candidate import (
    HIGHER_BETTER,
    LOWER_BETTER,
)

DEFAULT_SOURCE = (
    ROOT
    / "results/broad_rhea_difficulty_performance_nested_selected_v1"
    / "reactzyme_enzyme_projected_protein_cold"
)
DEFAULT_OUTPUT = ROOT / "results/capability_epoch_difficulty_audit_v1"
DIRECTION_AXIS = {
    "enzyme_to_reaction": "protein_identity_bucket",
    "reaction_to_enzyme": "reaction_similarity_bucket",
}
METRIC_SIGNS = {**{name: 1.0 for name in HIGHER_BETTER}, **{name: -1.0 for name in LOWER_BETTER}}


def compare_metric_dicts(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    direction: str,
    slice_name: str,
    slice_value: str,
    n_queries: int,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, sign in METRIC_SIGNS.items():
        if metric not in baseline or metric not in candidate:
            continue
        before = baseline[metric]
        after = candidate[metric]
        if before is None or after is None:
            continue
        before_f = float(before)
        after_f = float(after)
        raw = after_f - before_f
        improvement = sign * raw
        scale = abs(before_f)
        rows.append(
            {
                "direction": direction,
                "slice_name": slice_name,
                "slice_value": slice_value,
                "n_queries": int(n_queries),
                "metric": metric,
                "higher_is_better": sign > 0,
                "baseline": before_f,
                "candidate": after_f,
                "raw_delta": raw,
                "improvement_delta": improvement,
                "relative_improvement": improvement / scale if scale > 0 else None,
                "status": (
                    "improved"
                    if improvement > tolerance
                    else "regressed"
                    if improvement < -tolerance
                    else "tied"
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_pairing(one: pd.DataFrame, two: pd.DataFrame) -> None:
    keys = ["direction", "query_id"]
    if one.duplicated(keys).any() or two.duplicated(keys).any():
        raise ValueError("query metrics contain duplicate direction/query_id rows")
    one_keys = set(map(tuple, one[keys].astype(str).to_numpy()))
    two_keys = set(map(tuple, two[keys].astype(str).to_numpy()))
    if one_keys != two_keys:
        raise ValueError("1-epoch and 2-epoch query sets differ")
    for direction, axis in DIRECTION_AXIS.items():
        a = one.loc[one["direction"].eq(direction), ["query_id", axis]].copy()
        b = two.loc[two["direction"].eq(direction), ["query_id", axis]].copy()
        merged = a.merge(b, on="query_id", suffixes=("_1", "_2"), validate="one_to_one")
        left = merged[f"{axis}_1"].fillna("<NA>").astype(str)
        right = merged[f"{axis}_2"].fillna("<NA>").astype(str)
        if not left.equals(right):
            raise ValueError(f"difficulty labels changed between epochs for {direction}/{axis}")


def build_audit(one: pd.DataFrame, two: pd.DataFrame, *, tolerance: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    _validate_pairing(one, two)
    overall_rows: list[pd.DataFrame] = []
    difficulty_rows: list[pd.DataFrame] = []
    direction_summary: dict[str, object] = {}

    for direction, axis in DIRECTION_AXIS.items():
        one_dir = one[one["direction"].eq(direction)].copy()
        two_dir = two[two["direction"].eq(direction)].copy()
        if one_dir.empty or two_dir.empty:
            raise ValueError(f"missing direction {direction}")
        baseline = summarize_query_metrics(one_dir)
        candidate = summarize_query_metrics(two_dir)
        whole = compare_metric_dicts(
            baseline,
            candidate,
            direction=direction,
            slice_name="all",
            slice_value="all",
            n_queries=len(one_dir),
            tolerance=tolerance,
        )
        overall_rows.append(whole)

        bucket_summary: dict[str, object] = {}
        for bucket, one_bucket in one_dir.groupby(axis, dropna=False, sort=True):
            bucket_label = "<NA>" if pd.isna(bucket) else str(bucket)
            ids = set(one_bucket["query_id"].astype(str))
            two_bucket = two_dir[two_dir["query_id"].astype(str).isin(ids)].copy()
            if len(two_bucket) != len(one_bucket):
                raise ValueError(f"pairing mismatch for {direction}/{axis}/{bucket_label}")
            b = summarize_query_metrics(one_bucket)
            c = summarize_query_metrics(two_bucket)
            delta = compare_metric_dicts(
                b,
                c,
                direction=direction,
                slice_name=axis,
                slice_value=bucket_label,
                n_queries=len(one_bucket),
                tolerance=tolerance,
            )
            difficulty_rows.append(delta)
            bucket_summary[bucket_label] = {
                "n_queries": int(len(one_bucket)),
                "metric_rows": int(len(delta)),
                "improved": int(delta["status"].eq("improved").sum()),
                "tied": int(delta["status"].eq("tied").sum()),
                "regressed": int(delta["status"].eq("regressed").sum()),
                "mrr_delta": float(delta.loc[delta["metric"].eq("mrr"), "improvement_delta"].iloc[0]),
                "hit_at_10_delta": float(
                    delta.loc[delta["metric"].eq("hit_at_10"), "improvement_delta"].iloc[0]
                ),
            }

        direction_summary[direction] = {
            "difficulty_axis": axis,
            "n_queries": int(len(one_dir)),
            "metric_rows": int(len(whole)),
            "improved": int(whole["status"].eq("improved").sum()),
            "tied": int(whole["status"].eq("tied").sum()),
            "regressed": int(whole["status"].eq("regressed").sum()),
            "buckets": bucket_summary,
        }

    overall = pd.concat(overall_rows, ignore_index=True)
    difficulty = pd.concat(difficulty_rows, ignore_index=True)
    payload: dict[str, object] = {
        "audit_role": "posthoc_descriptive_capability_audit",
        "model_selection_allowed": False,
        "benchmark_labels_read_by_this_audit": False,
        "input_type": "already_materialized_query_metrics_with_train_only_difficulty_metadata",
        "comparison": "paired 1-epoch vs 2-epoch full-candidate evaluation on identical query sets",
        "tolerance": float(tolerance),
        "overall": {
            "metric_rows": int(len(overall)),
            "expected_metric_rows": 72,
            "improved": int(overall["status"].eq("improved").sum()),
            "tied": int(overall["status"].eq("tied").sum()),
            "regressed": int(overall["status"].eq("regressed").sum()),
        },
        "direction_summary": direction_summary,
    }
    if len(overall) != 72:
        raise RuntimeError(f"expected the registered 72-metric audit, got {len(overall)} rows")
    return overall, difficulty, payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc paired 1→2 epoch capability audit using only existing query metrics and train-only difficulty labels."
    )
    parser.add_argument("--one-epoch", type=Path, default=DEFAULT_SOURCE / "one_epoch_query_metrics_with_difficulty.csv")
    parser.add_argument("--two-epoch", type=Path, default=DEFAULT_SOURCE / "two_epoch_query_metrics_with_difficulty.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    one = pd.read_csv(args.one_epoch, low_memory=False)
    two = pd.read_csv(args.two_epoch, low_memory=False)
    overall, difficulty, payload = build_audit(one, two, tolerance=args.tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output_dir / "overall_72_metric_deltas.csv", index=False)
    difficulty.to_csv(args.output_dir / "difficulty_metric_deltas.csv", index=False)
    payload.update(
        {
            "one_epoch_query_metrics": str(args.one_epoch.resolve()),
            "two_epoch_query_metrics": str(args.two_epoch.resolve()),
            "overall_metric_output": str((args.output_dir / "overall_72_metric_deltas.csv").resolve()),
            "difficulty_metric_output": str((args.output_dir / "difficulty_metric_deltas.csv").resolve()),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
