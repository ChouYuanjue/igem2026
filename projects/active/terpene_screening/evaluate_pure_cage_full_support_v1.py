from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(frame: pd.DataFrame) -> dict[str, object]:
    required = {"reaction_id", "uniprot_id", "label", "pred_logit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    for reaction_id, group in frame.groupby("reaction_id", sort=True):
        group = group.sort_values(["pred_logit", "uniprot_id"], ascending=[False, True])
        positive_ranks = np.flatnonzero(pd.to_numeric(group["label"]).to_numpy() > 0) + 1
        if not len(positive_ranks):
            continue
        best = int(positive_ranks.min())
        rows.append(
            {
                "reaction_id": str(reaction_id),
                "positive_count": int((pd.to_numeric(group["label"]) > 0).sum()),
                "candidate_count": len(group),
                "best_positive_rank": best,
                "reciprocal_rank": 1.0 / best,
                "hit1": int(best <= 1),
                "hit3": int(best <= 3),
                "hit5": int(best <= 5),
                "hit10": int(best <= 10),
                "hit20": int(best <= 20),
                "positive_hits1": int((positive_ranks <= 1).sum()),
                "positive_hits3": int((positive_ranks <= 3).sum()),
                "positive_hits5": int((positive_ranks <= 5).sum()),
                "positive_hits10": int((positive_ranks <= 10).sum()),
                "positive_hits20": int((positive_ranks <= 20).sum()),
            }
        )
    q = pd.DataFrame(rows)
    if q.empty:
        raise ValueError("No evaluable reaction queries with positive labels")
    return {
        "query_metrics": q,
        "summary": {
            "n_evaluable_reactions": len(q),
            "candidate_count_per_reaction_min": int(q["candidate_count"].min()),
            "candidate_count_per_reaction_max": int(q["candidate_count"].max()),
            "mrr": float(q["reciprocal_rank"].mean()),
            "median_best_positive_rank": float(q["best_positive_rank"].median()),
            "hit1": float(q["hit1"].mean()),
            "hit3": float(q["hit3"].mean()),
            "hit5": float(q["hit5"].mean()),
            "hit10": float(q["hit10"].mean()),
            "hit20": float(q["hit20"].mean()),
            "mrr_at_1": float(np.where(q["best_positive_rank"] <= 1, q["reciprocal_rank"], 0.0).mean()),
            "mrr_at_3": float(np.where(q["best_positive_rank"] <= 3, q["reciprocal_rank"], 0.0).mean()),
            "mrr_at_5": float(np.where(q["best_positive_rank"] <= 5, q["reciprocal_rank"], 0.0).mean()),
            "mrr_at_10": float(np.where(q["best_positive_rank"] <= 10, q["reciprocal_rank"], 0.0).mean()),
            "mrr_at_20": float(np.where(q["best_positive_rank"] <= 20, q["reciprocal_rank"], 0.0).mean()),
            "expected_positive_hits_at_1": float(q["positive_hits1"].mean()),
            "expected_positive_hits_at_3": float(q["positive_hits3"].mean()),
            "expected_positive_hits_at_5": float(q["positive_hits5"].mean()),
            "expected_positive_hits_at_10": float(q["positive_hits10"].mean()),
            "expected_positive_hits_at_20": float(q["positive_hits20"].mean()),
            "macro_positive_recall_at_1": float((q["positive_hits1"] / q["positive_count"]).mean()),
            "macro_positive_recall_at_3": float((q["positive_hits3"] / q["positive_count"]).mean()),
            "macro_positive_recall_at_5": float((q["positive_hits5"] / q["positive_count"]).mean()),
            "macro_positive_recall_at_10": float((q["positive_hits10"] / q["positive_count"]).mean()),
            "macro_positive_recall_at_20": float((q["positive_hits20"] / q["positive_count"]).mean()),
            "micro_positive_recall_at_10": float(q["positive_hits10"].sum() / q["positive_count"].sum()),
            "micro_positive_recall_at_20": float(q["positive_hits20"].sum() / q["positive_count"].sum()),
            "ranking_score": "raw pre-sigmoid pred_logit",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    result = evaluate(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["query_metrics"].to_csv(args.output_dir / "query_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
