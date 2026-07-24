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

from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
    split_pairs,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PROTEIN_DIR = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_DUAL_TOWER_DIR = ROOT / "results/terpene_dual_tower_cold"
DEFAULT_REACTION_SIMILARITY_DIR = ROOT / "results/terpene_zero_shot_cold"
DEFAULT_OUTPUT = ROOT / "results/terpene_hybrid_bidirectional_cold"
DEFAULT_SCOPES = ("protein_cold", "reaction_cold", "double_cold")
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def tied_rank_percentile(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    order = np.lexsort((np.asarray(ids), -scores))
    sorted_scores = scores[order]
    result = np.empty(len(scores), dtype=np.float32)
    if len(scores) == 1:
        result[0] = 1.0
        return result
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_position = (start + end - 1) / 2
        result[order[start:end]] = 1.0 - average_position / (len(scores) - 1)
        start = end
    return result


def load_model(path: Path, device: torch.device) -> TerpeneDualTower:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_protein_features(protein_dir: Path) -> tuple[np.ndarray, list[str]]:
    entries = pd.read_csv(protein_dir / "entries.csv", dtype={"Entry": str}).sort_values("row")
    matrix = normalize_rows(np.load(protein_dir / "embeddings.npy").astype(np.float32))
    if len(entries) != len(matrix):
        raise ValueError("Protein feature matrix and entries file differ in length.")
    return matrix, entries["Entry"].astype(str).tolist()


def load_reaction_similarity(path: Path) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(path / "reaction_similarity_matrix.npy").astype(np.float32)
    entries = pd.read_csv(path / "reaction_similarity_entries.csv", dtype={"rhea_id": str}).sort_values("row")
    if len(entries) != len(matrix):
        raise ValueError("Reaction similarity matrix and entries file differ in length.")
    return matrix, entries["rhea_id"].astype(str).tolist()


def reaction_to_enzyme_transfer(
    reaction_id: str,
    train_by_reaction: dict[str, list[str]],
    reaction_similarity: np.ndarray,
    reaction_to_similarity_row: dict[str, int],
    protein_features: np.ndarray,
    protein_to_row: dict[str, int],
    protein_ids: list[str],
    topk_neighbors: int,
) -> np.ndarray:
    if reaction_id not in reaction_to_similarity_row:
        return np.zeros(len(protein_ids), dtype=np.float32)
    target_row = reaction_to_similarity_row[reaction_id]
    seed_reactions = [value for value in train_by_reaction if value in reaction_to_similarity_row and value != reaction_id]
    seed_reactions.sort(
        key=lambda value: (
            -float(reaction_similarity[target_row, reaction_to_similarity_row[value]]),
            value,
        )
    )
    seed_weights: dict[str, float] = defaultdict(float)
    for seed_reaction in seed_reactions[:topk_neighbors]:
        weight = float(reaction_similarity[target_row, reaction_to_similarity_row[seed_reaction]])
        for entry in train_by_reaction[seed_reaction]:
            if entry in protein_to_row:
                seed_weights[entry] = max(seed_weights[entry], weight)
    if not seed_weights:
        return np.zeros(len(protein_ids), dtype=np.float32)
    seed_ids = sorted(seed_weights)
    seed_rows = np.asarray([protein_to_row[value] for value in seed_ids], dtype=np.int64)
    weights = np.asarray([seed_weights[value] for value in seed_ids], dtype=np.float32)
    similarities = protein_features @ protein_features[seed_rows].T
    return (similarities * weights[None, :]).max(axis=1).astype(np.float32)


def enzyme_to_reaction_transfer(
    entry: str,
    train_by_protein: dict[str, list[str]],
    protein_features: np.ndarray,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    n_reactions: int,
    topk_neighbors: int,
) -> np.ndarray:
    if entry not in protein_to_row:
        return np.zeros(n_reactions, dtype=np.float32)
    train_proteins = [value for value in train_by_protein if value in protein_to_row and value != entry]
    if not train_proteins:
        return np.zeros(n_reactions, dtype=np.float32)
    query = protein_features[protein_to_row[entry]]
    train_rows = np.asarray([protein_to_row[value] for value in train_proteins], dtype=np.int64)
    similarities = protein_features[train_rows] @ query
    order = np.argsort(-similarities, kind="stable")[:topk_neighbors]
    scores = np.zeros(n_reactions, dtype=np.float32)
    for local_index in order:
        neighbor = train_proteins[int(local_index)]
        weight = float(similarities[int(local_index)])
        for reaction_id in train_by_protein[neighbor]:
            row = reaction_to_row.get(reaction_id)
            if row is not None:
                scores[row] = max(scores[row], weight)
    return scores


def metrics_record(
    scope: str,
    fold: int,
    direction: str,
    query_id: str,
    method: str,
    metrics: dict[str, float | int | None],
    known_count: int,
) -> dict[str, object]:
    return {
        "scope": scope,
        "fold": fold,
        "direction": direction,
        "query_id": query_id,
        "method": method,
        "known_associations_masked": known_count,
        **metrics,
    }


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["scope", "direction", "method"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual-tower plus bidirectional neighbor-transfer retrieval.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument("--dual-tower-dir", type=Path, default=DEFAULT_DUAL_TOWER_DIR)
    parser.add_argument("--reaction-similarity-dir", type=Path, default=DEFAULT_REACTION_SIMILARITY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scopes", default=",".join(DEFAULT_SCOPES))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--topk-neighbors", type=int, default=10)
    parser.add_argument("--direct-weights", default="0.25,0.5,0.75")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scopes = tuple(value.strip() for value in args.scopes.split(",") if value.strip())
    if set(scopes) - set(DEFAULT_SCOPES):
        raise ValueError(f"Unsupported scopes: {sorted(set(scopes) - set(DEFAULT_SCOPES))}")
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    direct_weights = tuple(float(value) for value in args.direct_weights.split(",") if value)
    if not direct_weights or any(value < 0 or value > 1 for value in direct_weights):
        raise ValueError("direct weights must be within [0, 1]")
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_features, protein_ids = load_protein_features(args.protein_dir.resolve())
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_features = np.load(args.dual_tower_dir / "reaction_feature_matrix.npy").astype(np.float32)
    schema = json.loads((args.dual_tower_dir / "feature_schema.json").read_text(encoding="utf-8"))
    reaction_ids = [str(value) for value in schema["reaction_ids"]]
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    reaction_similarity, similarity_ids = load_reaction_similarity(args.reaction_similarity_dir.resolve())
    reaction_to_similarity_row = {value: index for index, value in enumerate(similarity_ids)}

    split_frame = pd.read_csv(args.splits, dtype=str).fillna("")
    split_frame["protein_fold"] = pd.to_numeric(split_frame["protein_fold"]).astype(int)
    split_frame["reaction_fold"] = pd.to_numeric(split_frame["reaction_fold"]).astype(int)
    split_frame = split_frame[
        split_frame["Entry"].isin(protein_to_row) & split_frame["rhea_id"].isin(reaction_to_row)
    ].drop_duplicates(["rhea_id", "Entry"])
    folds = sorted(set(split_frame["protein_fold"]) | set(split_frame["reaction_fold"]))
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)

    records: list[dict[str, object]] = []
    for scope in scopes:
        for fold in folds:
            train_pairs, test_pairs = split_pairs(split_frame, scope, fold)
            model_path = args.dual_tower_dir / "models" / f"{scope}_fold{fold}.pt"
            model = load_model(model_path, device)
            with torch.no_grad():
                dual_proteins = model.encode_proteins(protein_tensor).cpu().numpy()
                dual_reactions = model.encode_reactions(reaction_tensor).cpu().numpy()
            train_by_reaction = {
                reaction_id: sorted(set(group["Entry"].astype(str)))
                for reaction_id, group in train_pairs.groupby("rhea_id")
            }
            train_by_protein = {
                entry: sorted(set(group["rhea_id"].astype(str)))
                for entry, group in train_pairs.groupby("Entry")
            }

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str)) & set(protein_to_row)
                if not positives or reaction_id not in reaction_to_row:
                    continue
                known = set(train_by_reaction.get(reaction_id, []))
                direct = dual_reactions[reaction_to_row[reaction_id]] @ dual_proteins.T
                transfer = reaction_to_enzyme_transfer(
                    reaction_id,
                    train_by_reaction,
                    reaction_similarity,
                    reaction_to_similarity_row,
                    protein_features,
                    protein_to_row,
                    protein_ids,
                    args.topk_neighbors,
                )
                direct_rank = tied_rank_percentile(direct, protein_ids)
                transfer_rank = tied_rank_percentile(transfer, protein_ids)
                score_map = {
                    "dual_tower": direct,
                    "reaction_neighbor_esmc_transfer": transfer,
                }
                for direct_weight in direct_weights:
                    score_map[f"rank_hybrid_direct_{direct_weight:g}"] = (
                        direct_weight * direct_rank + (1 - direct_weight) * transfer_rank
                    )
                for method, scores in score_map.items():
                    metrics = rank_metrics(scores, protein_ids, positives, known, budgets)
                    records.append(metrics_record(scope, fold, "reaction_to_enzyme", reaction_id, method, metrics, len(known)))

            for entry, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str)) & set(reaction_to_row)
                if not positives or entry not in protein_to_row:
                    continue
                known = set(train_by_protein.get(entry, []))
                direct = dual_proteins[protein_to_row[entry]] @ dual_reactions.T
                transfer = enzyme_to_reaction_transfer(
                    entry,
                    train_by_protein,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    len(reaction_ids),
                    args.topk_neighbors,
                )
                direct_rank = tied_rank_percentile(direct, reaction_ids)
                transfer_rank = tied_rank_percentile(transfer, reaction_ids)
                score_map = {
                    "dual_tower": direct,
                    "protein_neighbor_reaction_transfer": transfer,
                }
                for direct_weight in direct_weights:
                    score_map[f"rank_hybrid_direct_{direct_weight:g}"] = (
                        direct_weight * direct_rank + (1 - direct_weight) * transfer_rank
                    )
                for method, scores in score_map.items():
                    metrics = rank_metrics(scores, reaction_ids, positives, known, budgets)
                    records.append(metrics_record(scope, fold, "enzyme_to_reaction", entry, method, metrics, len(known)))

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best = (
        metrics.sort_values(
            ["scope", "direction", "hit_probability_at_10", "mean_reciprocal_rank"],
            ascending=[True, True, False, False],
        )
        .groupby(["scope", "direction"], as_index=False)
        .head(1)
    )
    best.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "dual_tower_dir": str(args.dual_tower_dir.resolve()),
        "topk_neighbors": args.topk_neighbors,
        "direct_weights": direct_weights,
        "scopes": scopes,
        "budgets": budgets,
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
