from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.select_cleanroom_bidirectional_multifold import load_summary

DIRECTION_METRICS = {
    "r2e": (
        "r2e_hit_at_10",
        "r2e_mrr",
        "r2e_map",
        "r2e_macro_roc_auc",
        "r2e_ndcg_at_10",
    ),
    "e2r": (
        "e2r_hit_at_10",
        "e2r_mrr",
        "e2r_map",
        "e2r_macro_roc_auc",
        "e2r_ndcg_at_10",
    ),
}


def select_direction(frame: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = DIRECTION_METRICS[direction]
    missing = (set(metrics) | {"candidate", "fold"}) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    counts = frame.groupby("candidate")["fold"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"candidates have unequal fold counts: {counts.to_dict()}")
    work = frame.copy()
    rank_cols: list[str] = []
    for metric in metrics:
        column = f"{metric}_percentile"
        work[column] = work.groupby("fold")[metric].rank(method="average", pct=True, ascending=True)
        rank_cols.append(column)
    work[f"{direction}_joint_percentile"] = work[rank_cols].mean(axis=1)
    rows: list[dict[str, object]] = []
    for candidate, group in work.groupby("candidate", sort=True):
        joint = group[f"{direction}_joint_percentile"]
        rows.append(
            {
                "candidate": candidate,
                "folds": int(group.fold.nunique()),
                "mean_joint_percentile": float(joint.mean()),
                "worst_fold_joint_percentile": float(joint.min()),
                "joint_percentile_std": float(joint.std(ddof=0)),
                **{f"mean_{metric}": float(group[metric].mean()) for metric in metrics},
            }
        )
    aggregate = pd.DataFrame(rows).sort_values(
        ["mean_joint_percentile", "worst_fold_joint_percentile", "joint_percentile_std", "candidate"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    aggregate["selected"] = False
    if len(aggregate):
        aggregate.loc[0, "selected"] = True
    return work, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direction-specific train-only multi-fold expert selection using five common-IR metrics per direction."
    )
    parser.add_argument("--run", action="append", required=True, help="CANDIDATE:FOLD:summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for spec in args.run:
        candidate, fold, path = spec.split(":", 2)
        rows.append(load_summary(Path(path), candidate, int(fold)))
    frame = pd.DataFrame(rows)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected: dict[str, str | None] = {}
    for direction in ("r2e", "e2r"):
        scored, aggregate = select_direction(frame, direction)
        scored.to_csv(out / f"{direction}_fold_scores.csv", index=False)
        aggregate.to_csv(out / f"{direction}_candidate_summary.csv", index=False)
        selected[direction] = None if aggregate.empty else str(aggregate.iloc[0].candidate)
    payload = {
        "protocol": "target-label-free_directional_multi-fold_expert_selection",
        "target_benchmark_labels_used": False,
        "metrics": {key: list(value) for key, value in DIRECTION_METRICS.items()},
        "selection_rule": (
            "For each direction independently: highest mean fold-wise five-metric percentile; "
            "then highest worst-fold percentile; then lower variability; deterministic candidate-name tie break."
        ),
        "selected_r2e_candidate": selected["r2e"],
        "selected_e2r_candidate": selected["e2r"],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
