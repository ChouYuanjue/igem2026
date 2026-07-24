from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import load_protein_library  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_UNIPROT = ROOT / "data/terpene_embeddings/uniprot_tps_primary_esmc600m"
DEFAULT_SHORT_MODELS = ROOT / "results/terpene_marts_domain_adaptation_r2e075"
DEFAULT_LONG_MODELS = ROOT / "results/terpene_marts_domain_adaptation_cartesian_pu"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_expanded_double_cold"
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def load_fold_models(directory: Path, split_id: str, device: torch.device) -> list[TerpeneDualTower]:
    paths = sorted((directory / "models").glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No models for {split_id} under {directory}")
    models = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def ensemble_scores(
    models: list[TerpeneDualTower],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    total = None
    with torch.no_grad():
        for model in models:
            proteins = model.encode_proteins(protein_tensor)
            reactions = model.encode_reactions(reaction_tensor)
            scores = (reactions @ proteins.T).cpu().numpy()
            total = scores if total is None else total + scores
    if total is None:
        raise ValueError("No models supplied")
    return total / len(models)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["candidate_universe", "budget"])
        .agg(
            n_queries=("query_id", "size"),
            hit_probability=("hit", "mean"),
            positive_recall=("positive_recall", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_best_positive_rank=("best_positive_rank", "median"),
            mean_uniprot_above_best_positive=("uniprot_above_best_positive", "mean"),
            median_uniprot_above_best_positive=("uniprot_above_best_positive", "median"),
            uniprot_top1_fraction=("top1_is_uniprot", "mean"),
        )
        .reset_index()
    )


def paired_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget, group in frame.groupby("budget", sort=True):
        pivot = group.pivot(index=["split_id", "query_id"], columns="candidate_universe", values="hit")
        canonical = pivot["canonical"].astype(int)
        expanded = pivot["expanded"].astype(int)
        rows.append(
            {
                "budget": budget,
                "n_queries": len(pivot),
                "canonical_hits": int(canonical.sum()),
                "expanded_hits": int(expanded.sum()),
                "retained_hits": int(((canonical == 1) & (expanded == 1)).sum()),
                "hits_lost_to_expansion": int(((canonical == 1) & (expanded == 0)).sum()),
                "new_hits_after_expansion": int(((canonical == 0) & (expanded == 1)).sum()),
                "hit_retention_fraction": (
                    float(((canonical == 1) & (expanded == 1)).sum() / canonical.sum())
                    if canonical.sum()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def rescue_slot_retention(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame[frame["candidate_universe"].eq("canonical")].copy()
    rows = []
    for budget, group in canonical.groupby("budget", sort=True):
        budget = int(budget)
        baseline_hits = int(group["hit"].sum())
        for rescue_slots in range(budget + 1):
            canonical_slots = budget - rescue_slots
            quota_hits = (
                int((group["best_positive_rank"] <= canonical_slots).sum())
                if canonical_slots > 0
                else 0
            )
            rows.append(
                {
                    "budget": budget,
                    "rescue_slots": rescue_slots,
                    "canonical_slots": canonical_slots,
                    "n_queries": len(group),
                    "baseline_canonical_hits": baseline_hits,
                    "quota_hits": quota_hits,
                    "hit_probability": quota_hits / len(group),
                    "hit_retention_fraction": (
                        quota_hits / baseline_hits if baseline_hits else np.nan
                    ),
                    "hits_lost": baseline_hits - quota_hits,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict double-cold stress test with UniProt candidate expansion.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--uniprot-protein-dir", type=Path, default=DEFAULT_UNIPROT)
    parser.add_argument("--short-model-dir", type=Path, default=DEFAULT_SHORT_MODELS)
    parser.add_argument("--long-model-dir", type=Path, default=DEFAULT_LONG_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    canonical_features = normalize_rows(
        np.load(cache_dir / "protein_features.npy").astype(np.float32)
    )
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].astype(str).str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].astype(str).str.lower().eq("true")
    canonical_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    uniprot_features, uniprot_ids = load_protein_library(args.uniprot_protein_dir.resolve())
    if set(canonical_ids) & set(uniprot_ids):
        raise ValueError("UniProt stress-test IDs overlap canonical protein IDs")
    expanded_features = np.concatenate([canonical_features, uniprot_features], axis=0)
    expanded_ids = canonical_ids + uniprot_ids
    canonical_count = len(canonical_ids)
    records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            if test.empty:
                continue
            score_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for budget, model_directory in [
                (3, args.short_model_dir.resolve()),
                (10, args.short_model_dir.resolve()),
                (20, args.long_model_dir.resolve()),
            ]:
                key = 0 if budget <= 10 else 1
                if key not in score_cache:
                    models = load_fold_models(model_directory, split_id, device)
                    expanded_scores = ensemble_scores(
                        models, expanded_features, reaction_features, device
                    )
                    canonical_scores = expanded_scores[:, :canonical_count]
                    score_cache[key] = (canonical_scores, expanded_scores)
                canonical_scores, expanded_scores = score_cache[key]
                for reaction_id, group in test.groupby("rhea_id", sort=True):
                    positives = set(group["Entry"].astype(str))
                    reaction_row = reaction_to_row[reaction_id]
                    for universe, scores, candidate_ids in [
                        ("canonical", canonical_scores[reaction_row], canonical_ids),
                        ("expanded", expanded_scores[reaction_row], expanded_ids),
                    ]:
                        metrics = rank_metrics(
                            scores,
                            candidate_ids,
                            positives,
                            set(),
                            (budget,),
                        )
                        best_rank = int(metrics["best_positive_rank"])
                        order = np.lexsort((np.asarray(candidate_ids), -scores))
                        top1_is_uniprot = bool(order[0] >= canonical_count)
                        uniprot_above = (
                            int(np.sum(order[: max(best_rank - 1, 0)] >= canonical_count))
                            if universe == "expanded"
                            else 0
                        )
                        records.append(
                            {
                                "split_id": split_id,
                                "query_id": reaction_id,
                                "budget": budget,
                                "candidate_universe": universe,
                                "candidate_count": len(candidate_ids),
                                "n_positives": len(positives),
                                "hit": int(metrics[f"hit_at_{budget}"]),
                                "positive_recall": float(
                                    metrics[f"positive_recall_at_{budget}"]
                                ),
                                "best_positive_rank": best_rank,
                                "reciprocal_rank": float(metrics["reciprocal_rank"]),
                                "top1_is_uniprot": top1_is_uniprot,
                                "uniprot_above_best_positive": uniprot_above,
                            }
                        )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics)
    paired = paired_comparison(query_metrics)
    quota = rescue_slot_retention(query_metrics)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_hit_retention.csv", index=False)
    quota.to_csv(output_dir / "rescue_slot_retention.csv", index=False)
    quota[quota["rescue_slots"].isin([0, 1, 2, 3, 4, 5])].to_csv(
        output_dir / "rescue_slot_retention_practical.csv", index=False
    )
    summary = {
        "canonical_candidate_count": len(canonical_ids),
        "uniprot_candidate_count": len(uniprot_ids),
        "expanded_candidate_count": len(expanded_ids),
        "strict_external_double_cold": True,
        "uniprot_candidates_are_unlabelled_decoys_for_this_stress_test": True,
        "metrics": metrics.to_dict("records"),
        "paired_hit_retention": paired.to_dict("records"),
        "controlled_rescue_slots": {
            "interpretation": "Canonical candidates occupy the prefix; unlabelled UniProt candidates use only reserved tail slots. This measures preservation of known external positives, not UniProt discovery yield.",
            "recommended_defaults": {"top3": 0, "top10": 1, "top20": 2},
            "recommended_rows": quota[
                ((quota["budget"] == 3) & (quota["rescue_slots"] == 0))
                | ((quota["budget"] == 10) & (quota["rescue_slots"] == 1))
                | ((quota["budget"] == 20) & (quota["rescue_slots"] == 2))
            ].to_dict("records"),
        },
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "paired": str(output_dir / "paired_hit_retention.csv"),
            "rescue_slots": str(output_dir / "rescue_slot_retention.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
