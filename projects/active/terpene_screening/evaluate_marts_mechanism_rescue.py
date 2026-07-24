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

from projects.active.terpene_screening.rank_open_world import tied_rank_percentile  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_STEPS = ROOT / "data/terpene_marts/marts_mechanism_steps.tsv"
DEFAULT_MODEL_DIR = ROOT / "results/terpene_marts_domain_adaptation_cartesian_drfp"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_mechanism_rescue"
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def build_mechanism_vectors(
    marts_path: Path,
    steps_path: Path,
    reaction_table: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    marts = pd.read_csv(marts_path, sep="\t", dtype=str).fillna("")
    steps = pd.read_csv(steps_path, sep="\t", dtype=str).fillna("")
    reaction_types = sorted(value for value in steps["Reaction_type"].astype(str).unique() if value)
    evidence_values = sorted(value for value in steps["Evidence"].astype(str).unique() if value)
    type_to_row = {value: index for index, value in enumerate(reaction_types)}
    evidence_to_row = {value: index for index, value in enumerate(evidence_values)}
    vector_dim = len(reaction_types) + len(evidence_values) + 3

    mechanism_vectors: dict[str, np.ndarray] = {}
    for mechanism_id, group in steps[steps["Mechanism_marts_id"] != ""].groupby("Mechanism_marts_id", sort=True):
        vector = np.zeros(vector_dim, dtype=np.float32)
        for value, count in group["Reaction_type"].value_counts().items():
            if value in type_to_row:
                vector[type_to_row[value]] = float(count)
        offset = len(reaction_types)
        for value, count in group["Evidence"].value_counts().items():
            if value in evidence_to_row:
                vector[offset + evidence_to_row[value]] = float(count)
        step_count = max(1, len(group))
        vector[: offset + len(evidence_values)] /= step_count
        vector[-3] = min(step_count, 20) / 20.0
        vector[-2] = float(group["Reaction_type"].eq("cyclization").sum()) / step_count
        vector[-1] = float(group["Reaction_type"].isin(["hydride shift", "methyl shift", "WM rearrangement"]).sum()) / step_count
        mechanism_vectors[str(mechanism_id)] = vector

    signature_to_mechanisms = (
        marts[(marts["reaction_signature"] != "") & (marts["mechanism_marts_id"] != "")]
        .groupby("reaction_signature")["mechanism_marts_id"]
        .apply(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    rows: list[np.ndarray] = []
    mechanism_counts: list[int] = []
    for row in reaction_table[["reaction_signature"]].itertuples(index=False):
        mechanism_ids = signature_to_mechanisms.get(str(row.reaction_signature), [])
        vectors = [mechanism_vectors[value] for value in mechanism_ids if value in mechanism_vectors]
        rows.append(np.mean(vectors, axis=0) if vectors else np.zeros(vector_dim, dtype=np.float32))
        mechanism_counts.append(len(vectors))
    matrix = normalize_rows(np.stack(rows).astype(np.float32))
    metadata = {
        "reaction_types": reaction_types,
        "evidence_values": evidence_values,
        "vector_dimension": vector_dim,
        "reactions_with_mechanism": int(sum(value > 0 for value in mechanism_counts)),
        "total_reactions": len(reaction_table),
    }
    return matrix, mechanism_counts, metadata


def load_adapted_models(model_dir: Path, split_id: str, device: torch.device) -> list[TerpeneDualTower]:
    paths = sorted((model_dir / "models").glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No adapted models for {split_id}")
    models: list[TerpeneDualTower] = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def direct_score_matrix(
    models: list[TerpeneDualTower],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    proteins = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reactions = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    total = np.zeros((len(reaction_features), len(protein_features)), dtype=np.float32)
    with torch.no_grad():
        for model in models:
            total += model.encode_reactions(reactions).cpu().numpy() @ model.encode_proteins(proteins).cpu().numpy().T
    return total / len(models)


def mechanism_transfer_scores(
    query_reaction_id: str,
    train_pairs: pd.DataFrame,
    reaction_ids: list[str],
    protein_ids: list[str],
    mechanism_matrix: np.ndarray,
    protein_features: np.ndarray,
    topk_reactions: int,
) -> np.ndarray | None:
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    query_row = reaction_to_row[query_reaction_id]
    if not mechanism_matrix[query_row].any():
        return None
    train_reactions = sorted(set(train_pairs["rhea_id"].astype(str)))
    train_rows = np.asarray([reaction_to_row[value] for value in train_reactions], dtype=np.int64)
    valid = np.linalg.norm(mechanism_matrix[train_rows], axis=1) > 0
    train_rows = train_rows[valid]
    train_reactions = [value for value, keep in zip(train_reactions, valid) if keep]
    if not len(train_rows):
        return None
    similarities = mechanism_matrix[train_rows] @ mechanism_matrix[query_row]
    order = np.lexsort((np.asarray(train_reactions), -similarities))
    selected: list[tuple[str, float]] = []
    for index in order:
        score = float(similarities[int(index)])
        if score <= 0:
            continue
        selected.append((train_reactions[int(index)], score))
        if len(selected) >= topk_reactions:
            break
    if not selected:
        return None

    protein_weights: dict[str, float] = {}
    proteins_by_reaction = train_pairs.groupby("rhea_id")["Entry"].apply(lambda values: sorted(set(values.astype(str)))).to_dict()
    for reaction_id, reaction_weight in selected:
        for protein_id in proteins_by_reaction.get(reaction_id, []):
            if protein_id in protein_to_row:
                protein_weights[protein_id] = max(protein_weights.get(protein_id, 0.0), reaction_weight)
    if not protein_weights:
        return None
    seed_ids = sorted(protein_weights)
    seed_rows = np.asarray([protein_to_row[value] for value in seed_ids], dtype=np.int64)
    weights = np.asarray([protein_weights[value] for value in seed_ids], dtype=np.float32)
    return (protein_features @ protein_features[seed_rows].T * weights[None, :]).max(axis=1)


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["coverage", "method"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism-aware rescue for MARTS reaction-to-enzyme double-cold retrieval.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk-reactions", type=int, default=5)
    parser.add_argument("--direct-weights", default="0.25,0.5,0.75")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    direct_weights = tuple(float(value) for value in args.direct_weights.split(",") if value)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = args.cache_dir.resolve()
    protein_table = pd.read_csv(cache / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    for column in ["protein_fold", "reaction_fold"]:
        pairs[column] = pd.to_numeric(pairs[column], errors="raise").astype(int)
    for column in ["protein_seen", "reaction_seen"]:
        pairs[column] = pairs[column].astype(str).str.lower().eq("true")
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_features = normalize_rows(np.load(cache / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache / "reaction_features.npy").astype(np.float32)
    mechanism_matrix, mechanism_counts, mechanism_metadata = build_mechanism_vectors(
        args.marts.resolve(), args.steps.resolve(), reaction_table
    )
    mechanism_available = {
        reaction_ids[index]: mechanism_counts[index] > 0 for index in range(len(reaction_ids))
    }
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    records: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for protein_fold in sorted(pairs["protein_fold"].unique()):
        for reaction_fold in sorted(pairs["reaction_fold"].unique()):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train_pairs = pairs[(pairs["protein_fold"] != protein_fold) & (pairs["reaction_fold"] != reaction_fold)]
            test_pairs = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            if test_pairs.empty:
                continue
            models = load_adapted_models(args.model_dir.resolve(), split_id, device)
            direct_matrix = direct_score_matrix(models, protein_features, reaction_features, device)
            covered_queries = 0
            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                direct = direct_matrix[reaction_to_row[reaction_id]]
                transfer = mechanism_transfer_scores(
                    reaction_id,
                    train_pairs,
                    reaction_ids,
                    protein_ids,
                    mechanism_matrix,
                    protein_features,
                    args.topk_reactions,
                )
                coverage = "mechanism_available" if transfer is not None else "all_queries"
                if transfer is not None:
                    covered_queries += 1
                score_map: dict[str, np.ndarray] = {"adapted_direct": direct}
                if transfer is not None:
                    score_map["mechanism_transfer"] = transfer
                    direct_rank = tied_rank_percentile(direct, protein_ids)
                    transfer_rank = tied_rank_percentile(transfer, protein_ids)
                    for direct_weight in direct_weights:
                        score_map[f"rank_hybrid_direct_{direct_weight:g}"] = (
                            direct_weight * direct_rank + (1 - direct_weight) * transfer_rank
                        )
                for method, scores in score_map.items():
                    metrics = rank_metrics(scores, protein_ids, positives, set(), budgets)
                    records.append(
                        {
                            "split_id": split_id,
                            "coverage": coverage,
                            "method": method,
                            "query_id": reaction_id,
                            **metrics,
                        }
                    )
            split_rows.append(
                {
                    "split_id": split_id,
                    "test_reaction_queries": test_pairs["rhea_id"].nunique(),
                    "mechanism_covered_queries": covered_queries,
                }
            )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    summary = {
        "model_dir": str(args.model_dir.resolve()),
        "topk_reactions": args.topk_reactions,
        "direct_weights": direct_weights,
        "budgets": budgets,
        "mechanism_metadata": mechanism_metadata,
        "n_query_method_rows": len(query_metrics),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "split_summary": str(output_dir / "split_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
