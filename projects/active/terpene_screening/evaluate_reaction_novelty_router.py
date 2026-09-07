from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import summarize_query_metrics


def route_query_metrics(
    backbone: pd.DataFrame,
    expert: pd.DataFrame,
    reaction_similarity: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    required_metrics = {"direction", "query_id", "candidate_count", "positive_count"}
    for name, frame in [("backbone", backbone), ("expert", expert)]:
        missing = required_metrics - set(frame.columns)
        if missing:
            raise ValueError(f"{name} query metrics missing {sorted(missing)}")
        if frame["query_id"].duplicated().any():
            raise ValueError(f"{name} query IDs must be unique after direction filtering")
    required_similarity = {"reaction_id", "max_train_drfp_tanimoto"}
    missing = required_similarity - set(reaction_similarity.columns)
    if missing:
        raise ValueError(f"reaction similarity file missing {sorted(missing)}")

    b = backbone[backbone["direction"].eq("reaction_to_enzyme")].copy()
    e = expert[expert["direction"].eq("reaction_to_enzyme")].copy()
    if set(b["query_id"]) != set(e["query_id"]):
        raise ValueError("backbone and expert must score identical R2E queries")
    b = b.sort_values("query_id").reset_index(drop=True)
    e = e.set_index("query_id").loc[b["query_id"]].reset_index()
    for column in ("candidate_count", "positive_count"):
        if not b[column].equals(e[column]):
            raise ValueError(f"backbone/expert {column} differ")

    sim = reaction_similarity[["reaction_id", "max_train_drfp_tanimoto"]].drop_duplicates(
        "reaction_id"
    )
    routed = b.merge(sim, left_on="query_id", right_on="reaction_id", how="left", validate="one_to_one")
    if routed["max_train_drfp_tanimoto"].isna().any():
        missing_ids = routed.loc[routed["max_train_drfp_tanimoto"].isna(), "query_id"].head().tolist()
        raise ValueError(f"missing train-distance for routed queries: {missing_ids}")
    use_expert = routed["max_train_drfp_tanimoto"].astype(float) < float(threshold)
    expert_aligned = e.set_index("query_id").loc[routed["query_id"]].reset_index()
    protected = {"direction", "query_id", "reaction_id", "max_train_drfp_tanimoto"}
    for column in b.columns:
        if column in protected:
            continue
        routed.loc[use_expert, column] = expert_aligned.loc[use_expert, column].to_numpy()
    routed["route_source"] = "backbone"
    routed.loc[use_expert, "route_source"] = "novelty_expert"
    return routed.drop(columns=["reaction_id"])


def guard_summary(backbone: dict[str, float | int | None], routed: dict[str, float | int | None]) -> dict[str, object]:
    higher = [
        "mrr", "map", "macro_roc_auc", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "hit_at_20", "hit_at_50", "ndcg_at_10", "ndcg_at_20", "ndcg_at_50",
        "top1_percent_ef", "success_at_0.01_fraction",
    ]
    lower = ["median_best_positive_rank", "mean_best_positive_rank_fraction"]
    deltas: dict[str, float] = {}
    violations: list[str] = []
    for metric in higher:
        if backbone.get(metric) is None or routed.get(metric) is None:
            continue
        delta = float(routed[metric]) - float(backbone[metric])
        deltas[metric] = delta
        if delta < -1e-12:
            violations.append(metric)
    for metric in lower:
        if backbone.get(metric) is None or routed.get(metric) is None:
            continue
        improvement = float(backbone[metric]) - float(routed[metric])
        deltas[metric] = improvement
        if improvement < -1e-12:
            violations.append(metric)
    return {"pass": not violations, "violations": violations, "improvement_deltas": deltas}


def main() -> None:
    parser = argparse.ArgumentParser(description="Route leakage-clean R2E queries by train-only reaction novelty.")
    parser.add_argument("--backbone-query-metrics", type=Path, required=True)
    parser.add_argument("--expert-query-metrics", type=Path, required=True)
    parser.add_argument("--reaction-similarity-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    backbone = pd.read_csv(args.backbone_query_metrics)
    expert = pd.read_csv(args.expert_query_metrics)
    similarity = pd.read_csv(args.reaction_similarity_csv)
    routed = route_query_metrics(backbone, expert, similarity, threshold=args.threshold)
    backbone_r2e = backbone[backbone["direction"].eq("reaction_to_enzyme")].copy()
    expert_r2e = expert[expert["direction"].eq("reaction_to_enzyme")].copy()
    summaries = {
        "backbone": summarize_query_metrics(backbone_r2e),
        "expert": summarize_query_metrics(expert_r2e),
        "routed": summarize_query_metrics(routed),
    }
    summary = {
        "protocol": "train_distance_label_free_expert_routing",
        "target_labels_used_for_routing": False,
        "routing_feature": "max_train_binary_drfp_tanimoto",
        "threshold": float(args.threshold),
        "expert_query_count": int(routed["route_source"].eq("novelty_expert").sum()),
        "expert_query_fraction": float(routed["route_source"].eq("novelty_expert").mean()),
        "summaries": summaries,
        "guard": guard_summary(summaries["backbone"], summaries["routed"]),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    routed.to_csv(output / "routed_query_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
