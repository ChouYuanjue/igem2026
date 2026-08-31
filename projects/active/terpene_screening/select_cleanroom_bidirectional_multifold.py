from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = (
    "r2e_hit_at_10",
    "r2e_mrr",
    "r2e_map",
    "r2e_macro_roc_auc",
    "r2e_ndcg_at_10",
    "e2r_hit_at_10",
    "e2r_mrr",
    "e2r_map",
    "e2r_macro_roc_auc",
    "e2r_ndcg_at_10",
)


def load_summary(path: Path, candidate: str, fold: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dev = payload.get("dev_metrics") or {}
    r2e = dev.get("common_ir_r2e")
    e2r = dev.get("common_ir_e2r")
    if r2e is None or e2r is None:
        raise ValueError(f"summary lacks bidirectional dev metrics: {path}")
    row: dict[str, object] = {
        "candidate": candidate,
        "fold": int(fold),
        "summary_path": str(path.resolve()),
    }
    for prefix, metrics in [("r2e", r2e), ("e2r", e2r)]:
        for key in ["hit_at_10", "mrr", "map", "macro_roc_auc", "ndcg_at_10"]:
            row[f"{prefix}_{key}"] = float(metrics[key])
    return row


def select(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = (set(METRICS) | {"candidate", "fold"}) - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    counts = frame.groupby("candidate")["fold"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"candidates have unequal fold counts: {counts.to_dict()}")
    work = frame.copy(); rank_cols=[]
    for metric in METRICS:
        column=f"{metric}_percentile"
        work[column]=work.groupby("fold")[metric].rank(method="average",pct=True,ascending=True)
        rank_cols.append(column)
    work["joint_percentile"] = work[rank_cols].mean(axis=1)
    work["r2e_joint_percentile"] = work[[c for c in rank_cols if c.startswith("r2e_")]].mean(axis=1)
    work["e2r_joint_percentile"] = work[[c for c in rank_cols if c.startswith("e2r_")]].mean(axis=1)
    rows=[]
    for candidate, group in work.groupby("candidate",sort=True):
        rows.append({
            "candidate":candidate,
            "folds":int(group.fold.nunique()),
            "mean_joint_percentile":float(group.joint_percentile.mean()),
            "worst_fold_joint_percentile":float(group.joint_percentile.min()),
            "mean_r2e_joint_percentile":float(group.r2e_joint_percentile.mean()),
            "mean_e2r_joint_percentile":float(group.e2r_joint_percentile.mean()),
            "joint_percentile_std":float(group.joint_percentile.std(ddof=0)),
            **{f"mean_{metric}":float(group[metric].mean()) for metric in METRICS},
        })
    aggregate=pd.DataFrame(rows).sort_values(
        ["mean_joint_percentile","worst_fold_joint_percentile","joint_percentile_std","candidate"],
        ascending=[False,False,True,True],kind="mergesort",
    ).reset_index(drop=True)
    aggregate["selected"] = False
    if len(aggregate): aggregate.loc[0,"selected"] = True
    return work, aggregate


def main() -> None:
    parser=argparse.ArgumentParser(description="Balanced bidirectional internal-fold selection for protein/reaction representations.")
    parser.add_argument("--run",action="append",required=True,help="CANDIDATE:FOLD:summary.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for spec in args.run:
        candidate,fold,path=spec.split(":",2); rows.append(load_summary(Path(path),candidate,int(fold)))
    scored,aggregate=select(pd.DataFrame(rows))
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    scored.to_csv(out/"fold_scores.csv",index=False); aggregate.to_csv(out/"candidate_summary.csv",index=False)
    payload={
        "protocol":"target-label-free_balanced_bidirectional_multi-fold_selection",
        "target_benchmark_labels_used":False,
        "metrics":list(METRICS),
        "direction_weighting":"5 R2E and 5 E2R common-IR metrics, equal weight",
        "selection_rule":"highest mean fold-wise cross-metric percentile; then highest worst-fold percentile; then lower variability",
        "selected_candidate":None if aggregate.empty else str(aggregate.iloc[0].candidate),
    }
    (out/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(aggregate.to_string(index=False))

if __name__ == "__main__": main()
