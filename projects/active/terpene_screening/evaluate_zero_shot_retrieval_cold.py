from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    best_match_similarity,
    canonical_or_raw_reaction,
    mol_fp,
    precursor_class_from_reaction,
    product_skeleton_class,
    split_reaction_smiles,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_zero_shot_cold"
DEFAULT_TOPK_REACTIONS = 20
DEFAULT_BUDGETS = (1, 5, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def rank_percentile(scores: np.ndarray, entries: np.ndarray) -> np.ndarray:
    """Return descending percentile ranks while assigning equal scores equal ranks."""
    order = np.lexsort((entries, -scores))
    result = np.empty(len(scores), dtype=np.float32)
    if len(scores) == 1:
        result[order] = 1.0
        return result
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_position = (start + end - 1) / 2
        percentile = 1.0 - average_position / (len(scores) - 1)
        result[order[start:end]] = percentile
        start = end
    return result


def reaction_features(reaction_smiles: str) -> dict[str, object]:
    canonical = canonical_or_raw_reaction(reaction_smiles)
    reactants, products = split_reaction_smiles(canonical)
    return {
        "canonical": canonical,
        "reactant_fps": [fp for fp in (mol_fp(value) for value in reactants) if fp is not None],
        "product_fps": [fp for fp in (mol_fp(value) for value in products) if fp is not None],
        "precursor_class": precursor_class_from_reaction(canonical),
        "product_skeleton_class": product_skeleton_class(canonical),
    }


def reaction_similarity(first: dict[str, object], second: dict[str, object]) -> float:
    substrate = best_match_similarity(first["reactant_fps"], second["reactant_fps"])
    product = best_match_similarity(first["product_fps"], second["product_fps"])
    precursor_bonus = float(
        first["precursor_class"] == second["precursor_class"] and first["precursor_class"] != "unknown"
    )
    skeleton_bonus = float(
        first["product_skeleton_class"] == second["product_skeleton_class"]
        and first["product_skeleton_class"] != "unknown"
    )
    return 0.4 * substrate + 0.4 * product + 0.1 * precursor_bonus + 0.1 * skeleton_bonus


def build_reaction_similarity_matrix(reactions: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    ids = reactions["rhea_id"].astype(str).tolist()
    features = {
        row.rhea_id: reaction_features(row.smiles_seq)
        for row in reactions[["rhea_id", "smiles_seq"]].itertuples(index=False)
    }
    matrix = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for first_index, first_id in enumerate(ids):
        matrix[first_index, first_index] = 1.0
        for second_index in range(first_index):
            second_id = ids[second_index]
            score = reaction_similarity(features[first_id], features[second_id])
            matrix[first_index, second_index] = score
            matrix[second_index, first_index] = score
    return matrix, features


def sorted_ranking(scores: np.ndarray, entries: np.ndarray) -> np.ndarray:
    return np.lexsort((entries, -scores))


def evaluate_scores(
    records: list[dict[str, object]],
    scope: str,
    fold: int,
    reaction_id: str,
    method: str,
    scores: np.ndarray,
    entries: np.ndarray,
    positive_entries: set[str],
    budgets: tuple[int, ...],
    n_seed_reactions: int,
    n_seed_enzymes: int,
    max_seed_reaction_similarity: float,
) -> None:
    order = sorted_ranking(scores, entries)
    ranked_entries = entries[order]
    positive_mask = np.array([entry in positive_entries for entry in ranked_entries], dtype=bool)
    positive_positions = np.flatnonzero(positive_mask)
    best_rank = int(positive_positions[0] + 1) if len(positive_positions) else None
    base = {
        "scope": scope,
        "fold": fold,
        "reaction_id": reaction_id,
        "method": method,
        "n_positives": len(positive_entries),
        "best_positive_rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
        "n_seed_reactions": n_seed_reactions,
        "n_seed_enzymes": n_seed_enzymes,
        "max_seed_reaction_similarity": max_seed_reaction_similarity,
    }
    for budget in budgets:
        top_mask = positive_mask[:budget]
        hits = int(top_mask.sum())
        row = dict(base)
        row.update(
            {
                "B": budget,
                "hits": hits,
                "hit": hits > 0,
                "precision": hits / budget,
                "positive_recall": hits / len(positive_entries) if positive_entries else 0.0,
            }
        )
        records.append(row)


def evaluate_scope(
    scope: str,
    fold: int,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    reaction_ids: list[str],
    reaction_index: dict[str, int],
    similarity_matrix: np.ndarray,
    features: dict[str, dict[str, object]],
    candidate_entries: np.ndarray,
    entry_to_row: dict[str, int],
    embeddings: np.ndarray,
    topk_reactions: int,
    budgets: tuple[int, ...],
    records: list[dict[str, object]],
) -> None:
    train_by_reaction = {
        reaction_id: sorted(set(group["Entry"].astype(str)) & set(entry_to_row))
        for reaction_id, group in train_pairs.groupby("rhea_id")
    }
    train_reactions = sorted(train_by_reaction)
    train_indices = np.array([reaction_index[reaction_id] for reaction_id in train_reactions], dtype=np.int64)

    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        positive_entries = set(group["Entry"].astype(str)) & set(entry_to_row)
        if not positive_entries or reaction_id not in reaction_index or len(train_indices) == 0:
            continue
        target_index = reaction_index[reaction_id]
        similarities = similarity_matrix[target_index, train_indices]
        target_canonical = str(features[reaction_id]["canonical"])
        valid_positions = [
            position
            for position, seed_reaction in enumerate(train_reactions)
            if seed_reaction != reaction_id
            and not (
                target_canonical
                and str(features[seed_reaction]["canonical"])
                and target_canonical == str(features[seed_reaction]["canonical"])
            )
        ]
        valid_positions.sort(key=lambda position: (-float(similarities[position]), train_reactions[position]))
        selected_positions = valid_positions[:topk_reactions]
        selected_reactions = [train_reactions[position] for position in selected_positions]
        selected_similarities = [float(similarities[position]) for position in selected_positions]

        seed_weight: dict[str, float] = defaultdict(float)
        for seed_reaction, weight in zip(selected_reactions, selected_similarities):
            for entry in train_by_reaction.get(seed_reaction, []):
                seed_weight[entry] = max(seed_weight[entry], weight)

        reaction_transfer = np.zeros(len(candidate_entries), dtype=np.float32)
        for entry, weight in seed_weight.items():
            reaction_transfer[entry_to_row[entry]] = max(reaction_transfer[entry_to_row[entry]], weight)

        if seed_weight:
            seed_entries = sorted(seed_weight)
            seed_rows = np.array([entry_to_row[entry] for entry in seed_entries], dtype=np.int64)
            weights = np.array([seed_weight[entry] for entry in seed_entries], dtype=np.float32)
            cosine = embeddings @ embeddings[seed_rows].T
            weighted_cosine = cosine * weights[None, :]
            esmc_max = weighted_cosine.max(axis=1)
            centroid = (embeddings[seed_rows] * weights[:, None]).sum(axis=0, keepdims=True)
            centroid = normalize_rows(centroid)[0]
            esmc_centroid = embeddings @ centroid
        else:
            seed_entries = []
            esmc_max = np.zeros(len(candidate_entries), dtype=np.float32)
            esmc_centroid = np.zeros(len(candidate_entries), dtype=np.float32)

        reaction_rank = rank_percentile(reaction_transfer, candidate_entries)
        esmc_max_rank = rank_percentile(esmc_max, candidate_entries)
        esmc_centroid_rank = rank_percentile(esmc_centroid, candidate_entries)
        score_map = {
            "reaction_transfer": reaction_transfer,
            "esmc_seed_max": esmc_max,
            "esmc_seed_centroid": esmc_centroid,
            "rank_fusion_reaction_esmc_max": 0.5 * reaction_rank + 0.5 * esmc_max_rank,
            "rank_fusion_reaction_esmc_centroid": 0.5 * reaction_rank + 0.5 * esmc_centroid_rank,
        }
        max_similarity = max(selected_similarities) if selected_similarities else 0.0
        for method, scores in score_map.items():
            evaluate_scores(
                records,
                scope,
                fold,
                reaction_id,
                method,
                scores,
                candidate_entries,
                positive_entries,
                budgets,
                len(selected_reactions),
                len(seed_entries),
                max_similarity,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reaction-aware zero-shot TPS retrieval under cold splits.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk-reactions", type=int, default=DEFAULT_TOPK_REACTIONS)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_frame = pd.read_csv(args.embedding_dir / "entries.csv", dtype={"Entry": str}).sort_values("row")
    embeddings = normalize_rows(np.load(args.embedding_dir / "embeddings.npy").astype(np.float32))
    candidate_entries = entry_frame["Entry"].astype(str).to_numpy()
    if len(candidate_entries) != len(embeddings):
        raise ValueError("Embedding matrix and entry map have different lengths.")
    entry_to_row = {entry: index for index, entry in enumerate(candidate_entries)}

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry", "smiles_seq"]].drop_duplicates(["rhea_id", "Entry"])
    reactions = positives.groupby("rhea_id", as_index=False)["smiles_seq"].first().sort_values("rhea_id")
    reaction_ids = reactions["rhea_id"].astype(str).tolist()
    reaction_index = {reaction_id: index for index, reaction_id in enumerate(reaction_ids)}
    similarity_matrix, features = build_reaction_similarity_matrix(reactions)
    np.save(output_dir / "reaction_similarity_matrix.npy", similarity_matrix)
    pd.DataFrame({"row": range(len(reaction_ids)), "rhea_id": reaction_ids}).to_csv(
        output_dir / "reaction_similarity_entries.csv", index=False
    )

    split_pairs = pd.read_csv(args.splits, dtype=str).fillna("")
    split_pairs["protein_fold"] = pd.to_numeric(split_pairs["protein_fold"]).astype(int)
    split_pairs["reaction_fold"] = pd.to_numeric(split_pairs["reaction_fold"]).astype(int)
    records: list[dict[str, object]] = []

    evaluate_scope(
        "leave_one_reaction_out",
        -1,
        positives,
        positives,
        reaction_ids,
        reaction_index,
        similarity_matrix,
        features,
        candidate_entries,
        entry_to_row,
        embeddings,
        args.topk_reactions,
        budgets,
        records,
    )

    folds = sorted(set(split_pairs["reaction_fold"]) | set(split_pairs["protein_fold"]))
    for fold in folds:
        reaction_train = split_pairs[split_pairs["reaction_fold"] != fold]
        reaction_test = split_pairs[split_pairs["reaction_fold"] == fold]
        evaluate_scope(
            "reaction_cold",
            int(fold),
            reaction_train,
            reaction_test,
            reaction_ids,
            reaction_index,
            similarity_matrix,
            features,
            candidate_entries,
            entry_to_row,
            embeddings,
            args.topk_reactions,
            budgets,
            records,
        )

        double_train = split_pairs[
            (split_pairs["reaction_fold"] != fold) & (split_pairs["protein_fold"] != fold)
        ]
        double_test = split_pairs[
            (split_pairs["reaction_fold"] == fold) & (split_pairs["protein_fold"] == fold)
        ]
        evaluate_scope(
            "double_cold",
            int(fold),
            double_train,
            double_test,
            reaction_ids,
            reaction_index,
            similarity_matrix,
            features,
            candidate_entries,
            entry_to_row,
            embeddings,
            args.topk_reactions,
            budgets,
            records,
        )

    long = pd.DataFrame(records)
    long.to_csv(output_dir / "metrics_long.csv", index=False)
    reaction_level = long.drop_duplicates(["scope", "fold", "reaction_id", "method", "B"])
    aggregate = (
        reaction_level.groupby(["scope", "method", "B"])
        .agg(
            n_reactions=("reaction_id", "size"),
            hit_probability=("hit", "mean"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            expected_hits=("hits", "mean"),
            precision=("precision", "mean"),
            positive_recall=("positive_recall", "mean"),
            median_best_positive_rank=("best_positive_rank", "median"),
            mean_seed_reactions=("n_seed_reactions", "mean"),
            mean_seed_enzymes=("n_seed_enzymes", "mean"),
            mean_max_seed_reaction_similarity=("max_seed_reaction_similarity", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "metrics.csv", index=False)
    best = (
        aggregate.sort_values(
            ["scope", "B", "hit_probability", "mean_reciprocal_rank"],
            ascending=[True, True, False, False],
        )
        .groupby(["scope", "B"], as_index=False)
        .head(1)
    )
    best.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "n_candidates": int(len(candidate_entries)),
        "n_reactions": int(len(reaction_ids)),
        "n_positive_pairs": int(len(positives)),
        "topk_seed_reactions": args.topk_reactions,
        "budgets": budgets,
        "scopes": sorted(long["scope"].unique().tolist()),
        "outputs": {
            "metrics_long": str(output_dir / "metrics_long.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
            "reaction_similarity_matrix": str(output_dir / "reaction_similarity_matrix.npy"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
