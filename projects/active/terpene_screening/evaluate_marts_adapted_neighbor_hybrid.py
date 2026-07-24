from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (  # noqa: E402
    build_reaction_similarity_matrix,
)
from projects.active.terpene_screening.rank_open_world import tied_rank_percentile  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_MODEL_DIR = ROOT / "results/terpene_marts_domain_adaptation_cartesian_pu"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_adapted_neighbor_hybrid"
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def load_models(model_dir: Path, split_id: str, device: torch.device) -> list[TerpeneDualTower]:
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


def encode_models(
    models: list[TerpeneDualTower],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    protein_embeddings: list[np.ndarray] = []
    reaction_embeddings: list[np.ndarray] = []
    direct = np.zeros((len(reaction_features), len(protein_features)), dtype=np.float32)
    with torch.no_grad():
        for model in models:
            proteins = model.encode_proteins(protein_tensor).cpu().numpy()
            reactions = model.encode_reactions(reaction_tensor).cpu().numpy()
            protein_embeddings.append(proteins)
            reaction_embeddings.append(reactions)
            direct += reactions @ proteins.T
    return protein_embeddings, reaction_embeddings, direct / len(models)


def reaction_to_enzyme_transfer(
    reaction_id: str,
    train_by_reaction: dict[str, list[str]],
    reaction_similarity: np.ndarray,
    reaction_to_row: dict[str, int],
    protein_features: np.ndarray,
    protein_to_row: dict[str, int],
    protein_ids: list[str],
    topk_neighbors: int,
) -> np.ndarray | None:
    target_row = reaction_to_row.get(reaction_id)
    if target_row is None:
        return None
    train_reactions = [value for value in train_by_reaction if value in reaction_to_row and value != reaction_id]
    train_reactions.sort(key=lambda value: (-float(reaction_similarity[target_row, reaction_to_row[value]]), value))
    protein_weights: dict[str, float] = defaultdict(float)
    selected = 0
    for seed_reaction in train_reactions:
        weight = float(reaction_similarity[target_row, reaction_to_row[seed_reaction]])
        if weight <= 0:
            continue
        selected += 1
        for protein_id in train_by_reaction[seed_reaction]:
            if protein_id in protein_to_row:
                protein_weights[protein_id] = max(protein_weights[protein_id], weight)
        if selected >= topk_neighbors:
            break
    if not protein_weights:
        return None
    seed_ids = sorted(protein_weights)
    seed_rows = np.asarray([protein_to_row[value] for value in seed_ids], dtype=np.int64)
    weights = np.asarray([protein_weights[value] for value in seed_ids], dtype=np.float32)
    return (protein_features @ protein_features[seed_rows].T * weights[None, :]).max(axis=1)


def enzyme_to_reaction_transfer(
    protein_id: str,
    train_by_protein: dict[str, list[str]],
    protein_features: np.ndarray,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    reaction_embedding_sets: list[np.ndarray],
    topk_neighbors: int,
) -> np.ndarray | None:
    query_row = protein_to_row.get(protein_id)
    if query_row is None:
        return None
    train_proteins = [value for value in train_by_protein if value in protein_to_row and value != protein_id]
    if not train_proteins:
        return None
    train_rows = np.asarray([protein_to_row[value] for value in train_proteins], dtype=np.int64)
    similarities = protein_features[train_rows] @ protein_features[query_row]
    order = np.lexsort((np.asarray(train_proteins), -similarities))
    reaction_weights: dict[str, float] = defaultdict(float)
    selected = 0
    for local_index in order:
        weight = float(similarities[int(local_index)])
        if weight <= 0:
            continue
        selected += 1
        neighbor = train_proteins[int(local_index)]
        for reaction_id in train_by_protein[neighbor]:
            if reaction_id in reaction_to_row:
                reaction_weights[reaction_id] = max(reaction_weights[reaction_id], weight)
        if selected >= topk_neighbors:
            break
    if not reaction_weights:
        return None
    seed_ids = sorted(reaction_weights)
    seed_rows = np.asarray([reaction_to_row[value] for value in seed_ids], dtype=np.int64)
    weights = np.asarray([reaction_weights[value] for value in seed_ids], dtype=np.float32)
    total = np.zeros(len(reaction_to_row), dtype=np.float32)
    for embeddings in reaction_embedding_sets:
        total += (embeddings @ embeddings[seed_rows].T * weights[None, :]).max(axis=1)
    return total / len(reaction_embedding_sets)


def prefix_rescue_scores(
    primary_scores: np.ndarray,
    rescue_scores: np.ndarray,
    ids: list[str],
    prefix_size: int,
) -> np.ndarray:
    primary_order = np.lexsort((np.asarray(ids), -primary_scores))
    rescue_order = np.lexsort((np.asarray(ids), -rescue_scores))
    selected = list(primary_order[:prefix_size])
    selected_set = set(selected)
    for index in rescue_order:
        if int(index) not in selected_set:
            selected.append(int(index))
            selected_set.add(int(index))
    for index in primary_order[prefix_size:]:
        if int(index) not in selected_set:
            selected.append(int(index))
            selected_set.add(int(index))
    scores = np.empty(len(ids), dtype=np.float32)
    for position, index in enumerate(selected):
        scores[index] = float(len(ids) - position)
    return scores


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["direction", "method"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapted MARTS bidirectional neighbor-transfer benchmark.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk-neighbors", type=int, default=5)
    parser.add_argument("--direct-weights", default="0.25,0.5,0.75")
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    direct_weights = tuple(float(value) for value in args.direct_weights.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()

    protein_table = pd.read_csv(cache / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"], errors="raise").astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"], errors="raise").astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].astype(str).str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].astype(str).str.lower().eq("true")
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_features = normalize_rows(np.load(cache / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache / "reaction_features.npy").astype(np.float32)
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    similarity_input = reaction_table.rename(columns={"reaction_id": "rhea_id", "reaction_smiles": "smiles_seq"})
    reaction_similarity, _ = build_reaction_similarity_matrix(similarity_input[["rhea_id", "smiles_seq"]])

    records: list[dict[str, object]] = []
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
            models = load_models(args.model_dir.resolve(), split_id, device)
            _, reaction_embedding_sets, direct_matrix = encode_models(
                models, protein_features, reaction_features, device
            )
            train_by_reaction = {
                reaction_id: sorted(set(group["Entry"].astype(str)))
                for reaction_id, group in train_pairs.groupby("rhea_id")
            }
            train_by_protein = {
                protein_id: sorted(set(group["rhea_id"].astype(str)))
                for protein_id, group in train_pairs.groupby("Entry")
            }

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                direct = direct_matrix[reaction_to_row[reaction_id]]
                transfer = reaction_to_enzyme_transfer(
                    reaction_id,
                    train_by_reaction,
                    reaction_similarity,
                    reaction_to_row,
                    protein_features,
                    protein_to_row,
                    protein_ids,
                    args.topk_neighbors,
                )
                score_map: dict[str, np.ndarray] = {"adapted_direct": direct}
                if transfer is not None:
                    score_map["reaction_neighbor_esmc_transfer"] = transfer
                    score_map["direct_top5_neighbor_rescue"] = prefix_rescue_scores(
                        direct, transfer, protein_ids, 5
                    )
                    score_map["direct_top10_neighbor_rescue"] = prefix_rescue_scores(
                        direct, transfer, protein_ids, 10
                    )
                    direct_rank = tied_rank_percentile(direct, protein_ids)
                    transfer_rank = tied_rank_percentile(transfer, protein_ids)
                    for direct_weight in direct_weights:
                        score_map[f"rank_hybrid_direct_{direct_weight:g}"] = (
                            direct_weight * direct_rank + (1 - direct_weight) * transfer_rank
                        )
                for method, scores in score_map.items():
                    records.append(
                        {
                            "split_id": split_id,
                            "direction": "reaction_to_enzyme",
                            "method": method,
                            "query_id": reaction_id,
                            **rank_metrics(scores, protein_ids, positives, set(), budgets),
                        }
                    )

            for protein_id, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                direct = direct_matrix[:, protein_to_row[protein_id]]
                transfer = enzyme_to_reaction_transfer(
                    protein_id,
                    train_by_protein,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    reaction_embedding_sets,
                    args.topk_neighbors,
                )
                score_map = {"adapted_direct": direct}
                if transfer is not None:
                    score_map["protein_neighbor_reaction_transfer"] = transfer
                    score_map["direct_top5_neighbor_rescue"] = prefix_rescue_scores(
                        direct, transfer, reaction_ids, 5
                    )
                    score_map["direct_top10_neighbor_rescue"] = prefix_rescue_scores(
                        direct, transfer, reaction_ids, 10
                    )
                    direct_rank = tied_rank_percentile(direct, reaction_ids)
                    transfer_rank = tied_rank_percentile(transfer, reaction_ids)
                    for direct_weight in direct_weights:
                        score_map[f"rank_hybrid_direct_{direct_weight:g}"] = (
                            direct_weight * direct_rank + (1 - direct_weight) * transfer_rank
                        )
                for method, scores in score_map.items():
                    records.append(
                        {
                            "split_id": split_id,
                            "direction": "enzyme_to_reaction",
                            "method": method,
                            "query_id": protein_id,
                            **rank_metrics(scores, reaction_ids, positives, set(), budgets),
                        }
                    )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best = (
        metrics.sort_values(
            ["direction", "hit_probability_at_10", "mean_reciprocal_rank"],
            ascending=[True, False, False],
        )
        .groupby("direction", as_index=False)
        .head(1)
    )
    best.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "model_dir": str(args.model_dir.resolve()),
        "topk_neighbors": args.topk_neighbors,
        "direct_weights": direct_weights,
        "budgets": budgets,
        "n_query_method_rows": len(query_metrics),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
