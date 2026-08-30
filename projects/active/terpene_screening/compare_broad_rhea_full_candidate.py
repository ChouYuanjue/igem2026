from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

HIGHER_BETTER = (
    "mrr", "map", "macro_roc_auc", "top10_dcg", "ndcg_at_10", "ndcg_at_20", "ndcg_at_50",
    "top1_percent_ef", "top2_percent_ef", "mean_positive_reciprocal_rank",
    "hit_at_1", "hit_at_2", "hit_at_3", "hit_at_4", "hit_at_5", "hit_at_10", "hit_at_20", "hit_at_50",
    "positive_recall_at_1", "positive_recall_at_5", "positive_recall_at_10", "positive_recall_at_20", "positive_recall_at_50",
    "micro_positive_recall_at_1", "micro_positive_recall_at_5", "micro_positive_recall_at_10", "micro_positive_recall_at_20", "micro_positive_recall_at_50",
    "success_at_0.01_fraction", "success_at_0.02_fraction", "success_at_0.03_fraction", "success_at_0.05_fraction",
)
LOWER_BETTER = (
    "median_best_positive_rank", "mean_positive_rank", "mean_best_positive_rank_fraction", "median_best_positive_rank_fraction",
)
DIRECTIONS = ("reaction_to_enzyme", "enzyme_to_reaction")


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in payload:
        raise ValueError(f"summary missing metrics: {path}")
    return payload


def compare_summaries(baseline: Mapping[str, object], candidate: Mapping[str, object], tolerance: float = 0.0) -> tuple[pd.DataFrame, dict[str, object]]:
    bmetrics = baseline["metrics"]
    cmetrics = candidate["metrics"]
    rows: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        b = bmetrics[direction]
        c = cmetrics[direction]
        for metric, sign in [(m, 1.0) for m in HIGHER_BETTER] + [(m, -1.0) for m in LOWER_BETTER]:
            if metric not in b or metric not in c or b[metric] is None or c[metric] is None:
                continue
            before = float(b[metric]); after = float(c[metric]); raw = after - before
            improvement = sign * raw
            scale = abs(before)
            relative_improvement = improvement / scale if scale > 0 else None
            rows.append({
                "direction": direction,
                "metric": metric,
                "higher_is_better": sign > 0,
                "baseline": before,
                "candidate": after,
                "raw_delta": raw,
                "improvement_delta": improvement,
                "relative_improvement": relative_improvement,
                "status": "improved" if improvement > tolerance else ("regressed" if improvement < -tolerance else "tied"),
            })
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "tolerance": float(tolerance),
        "metric_rows": int(len(frame)),
        "improved": int(frame["status"].eq("improved").sum()),
        "tied": int(frame["status"].eq("tied").sum()),
        "regressed": int(frame["status"].eq("regressed").sum()),
        "regressions": frame.loc[frame["status"].eq("regressed"), ["direction", "metric", "baseline", "candidate", "improvement_delta", "relative_improvement"]].to_dict("records"),
        "direction_summary": {},
    }
    for direction, group in frame.groupby("direction", sort=True):
        summary["direction_summary"][str(direction)] = {
            "improved": int(group["status"].eq("improved").sum()),
            "tied": int(group["status"].eq("tied").sum()),
            "regressed": int(group["status"].eq("regressed").sum()),
            "mrr_improvement": float(group.loc[group["metric"].eq("mrr"), "improvement_delta"].iloc[0]),
            "map_improvement": float(group.loc[group["metric"].eq("map"), "improvement_delta"].iloc[0]),
        }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two broad-RHEA full-candidate summaries without cherry-picking metrics.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    baseline = _load(args.baseline); candidate = _load(args.candidate)
    frame, summary = compare_summaries(baseline, candidate, tolerance=args.tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "metric_deltas.csv", index=False)
    summary.update({"baseline": str(args.baseline.resolve()), "candidate": str(args.candidate.resolve())})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
