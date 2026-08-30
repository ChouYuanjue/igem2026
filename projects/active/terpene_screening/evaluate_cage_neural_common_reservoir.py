from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (
    ensemble_similarity,
    load_auxiliary_reaction_library,
    load_feature_schema,
    load_models_runtime,
    load_protein_library,
    load_reaction_library,
    models_require_auxiliary_reaction_features,
    reaction_embedding_ensemble,
)

DEFAULT_CAGE = ROOT / "results/terpene_old_new_comparison/legacy_historical/fair_cage_all_scores.csv"
DEFAULT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_BASE = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_OUTPUT = ROOT / "results/terpene_cage_neural_common_reservoir"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def parse_named_paths(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate model label: {name!r}")
        result[name] = Path(path).resolve()
    if not result:
        raise ValueError("At least one --model NAME=PATH is required")
    return result


def stable_seed(base_seed: int, direction: str, query_id: str, n_seed: int, rep: int) -> int:
    payload = f"{base_seed}|{direction}|{query_id}|{n_seed}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def reciprocal_rank_fusion(score_arrays: list[np.ndarray], ids: np.ndarray, k: float = 60.0) -> np.ndarray:
    if not score_arrays:
        raise ValueError("RRF requires at least one score array")
    out = np.zeros(len(ids), dtype=np.float64)
    for scores in score_arrays:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(ids),):
            raise ValueError("RRF score arrays must align with ids")
        order = np.lexsort((ids, -scores))
        rank = np.empty(len(ids), dtype=np.int64)
        rank[order] = np.arange(1, len(ids) + 1)
        out += 1.0 / (float(k) + rank)
    return out


def rank_metrics(
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    positives: set[str],
    budgets: tuple[int, ...],
) -> dict[str, float]:
    order = np.lexsort((candidate_ids, -np.asarray(scores, dtype=np.float64)))
    ranked = candidate_ids[order]
    positive_positions = [i + 1 for i, value in enumerate(ranked) if str(value) in positives]
    best = min(positive_positions) if positive_positions else None
    result: dict[str, float] = {
        "reciprocal_rank": 0.0 if best is None else 1.0 / best,
        "best_positive_rank": float("nan") if best is None else float(best),
        "n_positives": float(len(positives)),
        "candidate_count": float(len(candidate_ids)),
    }
    for budget in budgets:
        top = set(map(str, ranked[:budget]))
        hits = len(top & positives)
        result[f"hit_at_{budget}"] = float(hits > 0)
        result[f"expected_hits_at_{budget}"] = float(hits)
        result[f"positive_recall_at_{budget}"] = float(hits / len(positives)) if positives else 0.0
    return result


def aggregate(query_metrics: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["scenario", "direction", "n_seed", "method"]
    for keys, group in query_metrics.groupby(group_cols, sort=True, dropna=False):
        scenario, direction, n_seed, method = keys
        row: dict[str, object] = {
            "scenario": scenario,
            "direction": direction,
            "n_seed": int(n_seed),
            "method": method,
            "n_trials": int(len(group)),
            "n_unique_queries": int(group["query_id"].nunique()),
            "mrr": float(group["reciprocal_rank"].mean()),
            "mean_candidate_count": float(group["candidate_count"].mean()),
        }
        for budget in budgets:
            row[f"hit_at_{budget}"] = float(group[f"hit_at_{budget}"].mean())
            row[f"expected_hits_at_{budget}"] = float(group[f"expected_hits_at_{budget}"].mean())
            row[f"positive_recall_at_{budget}"] = float(group[f"positive_recall_at_{budget}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def load_model_scores(
    model_paths: dict[str, Path],
    *,
    protein_dir: Path,
    common_proteins: list[str],
    common_reactions: list[str],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]], np.ndarray, dict[str, int], dict[str, int]]:
    protein_features, protein_ids = load_protein_library(protein_dir)
    pindex = {value: i for i, value in enumerate(protein_ids)}
    p_rows = np.asarray([pindex[value] for value in common_proteins], dtype=np.int64)
    selected_protein_features = protein_features[p_rows]
    dense_scores: dict[str, np.ndarray] = {}
    reaction_embeddings: dict[str, list[np.ndarray]] = {}
    common_rindex: dict[str, int] | None = None
    for label, model_root in model_paths.items():
        schema = load_feature_schema(model_root)
        reaction_features, reaction_ids = load_reaction_library(model_root, schema)
        rindex = {value: i for i, value in enumerate(reaction_ids)}
        missing = [value for value in common_reactions if value not in rindex]
        if missing:
            raise ValueError(f"Model {label} misses common reactions: {missing[:10]}")
        r_rows = np.asarray([rindex[value] for value in common_reactions], dtype=np.int64)
        selected_reaction_features = reaction_features[r_rows]
        models = load_models_runtime(model_root / "models", "production", device)
        auxiliary = None
        if models_require_auxiliary_reaction_features(models):
            full_auxiliary = load_auxiliary_reaction_library(model_root, reaction_ids)
            auxiliary = full_auxiliary[r_rows]
        dense_scores[label] = ensemble_similarity(
            models,
            selected_protein_features,
            selected_reaction_features,
            device,
            auxiliary_reaction_features=auxiliary,
        )
        reaction_embeddings[label] = reaction_embedding_ensemble(
            models,
            selected_reaction_features,
            device,
            auxiliary_reaction_features=auxiliary,
        )
        if common_rindex is None:
            common_rindex = {value: i for i, value in enumerate(common_reactions)}
    assert common_rindex is not None
    return dense_scores, reaction_embeddings, selected_protein_features, {v: i for i, v in enumerate(common_proteins)}, common_rindex


def build_pair_frame(
    cage_path: Path,
    model_scores: dict[str, np.ndarray],
    common_proteins: list[str],
    common_reactions: list[str],
) -> pd.DataFrame:
    frame = pd.read_csv(cage_path, dtype=str).fillna("")
    required = {"reaction_id", "uniprot_id", "label", "cage_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CAGE score file misses columns: {sorted(missing)}")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int)
    frame["cage_score"] = pd.to_numeric(frame["cage_score"], errors="coerce")
    frame = frame[np.isfinite(frame["cage_score"])].copy()
    pindex = {value: i for i, value in enumerate(common_proteins)}
    rindex = {value: i for i, value in enumerate(common_reactions)}
    frame = frame[frame["uniprot_id"].isin(pindex) & frame["reaction_id"].isin(rindex)].copy()
    frame["protein_row"] = frame["uniprot_id"].map(pindex).astype(int)
    frame["reaction_row"] = frame["reaction_id"].map(rindex).astype(int)
    frame["pure_cage"] = frame["cage_score"].astype(float)
    for label, matrix in model_scores.items():
        frame[f"direct:{label}"] = matrix[
            frame["reaction_row"].to_numpy(), frame["protein_row"].to_numpy()
        ]
    return frame


def score_zero_shot(
    pair_frame: pd.DataFrame,
    model_labels: list[str],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    specs = [
        ("reaction_to_enzyme", "reaction_id", "uniprot_id"),
        ("enzyme_to_reaction", "uniprot_id", "reaction_id"),
    ]
    base_methods = ["pure_cage", *[f"direct:{label}" for label in model_labels]]
    for direction, qcol, ccol in specs:
        for query_id, group in pair_frame.groupby(qcol, sort=True):
            positives = set(group.loc[group["label"].eq(1), ccol].astype(str))
            if not positives:
                continue
            ids = group[ccol].astype(str).to_numpy()
            method_scores: dict[str, np.ndarray] = {
                method: group[method].to_numpy(dtype=float) for method in base_methods
            }
            for label in model_labels:
                method_scores[f"rrf:cage+{label}"] = reciprocal_rank_fusion(
                    [method_scores["pure_cage"], method_scores[f"direct:{label}"]], ids
                )
            for method, scores in method_scores.items():
                records.append({
                    "scenario": "zero_shot_common_reservoir",
                    "direction": direction,
                    "n_seed": 0,
                    "query_id": str(query_id),
                    "rep": 0,
                    "method": method,
                    **rank_metrics(ids, scores, positives, budgets),
                })
    return pd.DataFrame(records)


def _fewshot_seed_scores(
    *,
    direction: str,
    group: pd.DataFrame,
    seeds: list[str],
    candidate_ids: np.ndarray,
    model_labels: list[str],
    protein_features: np.ndarray,
    protein_index: dict[str, int],
    reaction_embeddings: dict[str, list[np.ndarray]],
    reaction_index: dict[str, int],
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if direction == "reaction_to_enzyme":
        candidate_rows = np.asarray([protein_index[value] for value in candidate_ids], dtype=np.int64)
        seed_rows = np.asarray([protein_index[value] for value in seeds], dtype=np.int64)
        scores["seed_similarity"] = (protein_features[candidate_rows] @ protein_features[seed_rows].T).max(axis=1)
    else:
        candidate_rows = np.asarray([reaction_index[value] for value in candidate_ids], dtype=np.int64)
        seed_rows = np.asarray([reaction_index[value] for value in seeds], dtype=np.int64)
        for label in model_labels:
            accumulated = np.zeros(len(candidate_ids), dtype=np.float32)
            for embeddings in reaction_embeddings[label]:
                accumulated += (embeddings[candidate_rows] @ embeddings[seed_rows].T).max(axis=1)
            scores[f"seed_similarity:{label}"] = accumulated / len(reaction_embeddings[label])
    return scores


def score_fewshot(
    pair_frame: pd.DataFrame,
    model_labels: list[str],
    budgets: tuple[int, ...],
    seed_counts: tuple[int, ...],
    repeats: int,
    base_seed: int,
    protein_features: np.ndarray,
    protein_index: dict[str, int],
    reaction_embeddings: dict[str, list[np.ndarray]],
    reaction_index: dict[str, int],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    specs = [
        ("reaction_to_enzyme", "reaction_id", "uniprot_id"),
        ("enzyme_to_reaction", "uniprot_id", "reaction_id"),
    ]
    for direction, qcol, ccol in specs:
        for query_id, group in pair_frame.groupby(qcol, sort=True):
            all_positive = sorted(set(group.loc[group["label"].eq(1), ccol].astype(str)))
            ids_all = group[ccol].astype(str).to_numpy()
            for n_seed in seed_counts:
                if len(all_positive) < n_seed + 1:
                    continue
                for rep in range(repeats):
                    rng = np.random.default_rng(stable_seed(base_seed, direction, str(query_id), n_seed, rep))
                    seeds = sorted(rng.choice(np.asarray(all_positive), size=n_seed, replace=False).tolist())
                    hidden = set(all_positive) - set(seeds)
                    keep = ~np.isin(ids_all, np.asarray(seeds))
                    ids = ids_all[keep]
                    local = group.loc[keep]
                    method_scores: dict[str, np.ndarray] = {
                        "pure_cage": local["pure_cage"].to_numpy(dtype=float),
                    }
                    for label in model_labels:
                        method_scores[f"direct:{label}"] = local[f"direct:{label}"].to_numpy(dtype=float)
                    seed_scores = _fewshot_seed_scores(
                        direction=direction,
                        group=local,
                        seeds=seeds,
                        candidate_ids=ids,
                        model_labels=model_labels,
                        protein_features=protein_features,
                        protein_index=protein_index,
                        reaction_embeddings=reaction_embeddings,
                        reaction_index=reaction_index,
                    )
                    method_scores.update(seed_scores)
                    for label in model_labels:
                        seed_key = "seed_similarity" if direction == "reaction_to_enzyme" else f"seed_similarity:{label}"
                        method_scores[f"rrf:cage+seed:{label}"] = reciprocal_rank_fusion(
                            [method_scores["pure_cage"], method_scores[seed_key]], ids
                        )
                        method_scores[f"rrf:direct+seed:{label}"] = reciprocal_rank_fusion(
                            [method_scores[f"direct:{label}"], method_scores[seed_key]], ids
                        )
                        method_scores[f"rrf:cage+direct+seed:{label}"] = reciprocal_rank_fusion(
                            [method_scores["pure_cage"], method_scores[f"direct:{label}"], method_scores[seed_key]], ids
                        )
                    for method, scores in method_scores.items():
                        records.append({
                            "scenario": "few_shot_hidden_positive_common_reservoir",
                            "direction": direction,
                            "n_seed": n_seed,
                            "query_id": str(query_id),
                            "rep": rep,
                            "method": method,
                            **rank_metrics(ids, scores, hidden, budgets),
                        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pure CAGE, neural experts and simple expert fusion on one common scored pair reservoir.")
    parser.add_argument("--cage-scores", type=Path, default=DEFAULT_CAGE)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--model", action="append", default=[], help="Repeat NAME=PATH for every neural expert")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seed-counts", default="1,2,3,5")
    parser.add_argument("--fewshot-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    seed_counts = tuple(sorted({int(value) for value in args.seed_counts.split(",") if value}))
    if not budgets or not seed_counts or args.fewshot_repeats <= 0:
        raise ValueError("budgets, seed counts and repeats must be positive")
    model_paths = parse_named_paths(args.model)
    cage = pd.read_csv(args.cage_scores, dtype=str).fillna("")
    cage_proteins = sorted(set(cage["uniprot_id"].astype(str)))
    cage_reactions = sorted(set(cage["reaction_id"].astype(str)))
    protein_features_all, protein_ids_all = load_protein_library(args.protein_dir.resolve())
    protein_set = set(protein_ids_all)
    common_proteins = [value for value in cage_proteins if value in protein_set]
    # Every model must contain every common reaction; retain only their intersection.
    reaction_sets = []
    for root in model_paths.values():
        reaction_sets.append(set(map(str, load_feature_schema(root)["reaction_ids"])))
    common_reactions = [value for value in cage_reactions if all(value in values for values in reaction_sets)]
    device = torch.device(args.device)
    model_scores, reaction_embeddings, protein_features, protein_index, reaction_index = load_model_scores(
        model_paths,
        protein_dir=args.protein_dir.resolve(),
        common_proteins=common_proteins,
        common_reactions=common_reactions,
        device=device,
    )
    pair_frame = build_pair_frame(
        args.cage_scores.resolve(), model_scores, common_proteins, common_reactions
    )
    zero = score_zero_shot(pair_frame, list(model_paths), budgets)
    few = score_fewshot(
        pair_frame,
        list(model_paths),
        budgets,
        seed_counts,
        args.fewshot_repeats,
        args.seed,
        protein_features,
        protein_index,
        reaction_embeddings,
        reaction_index,
    )
    query_metrics = pd.concat([zero, few], ignore_index=True)
    metrics = aggregate(query_metrics, budgets)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    pair_frame.to_csv(output / "pair_scores.csv", index=False)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = {
        "protocol": "common_cage_scored_pair_reservoir",
        "cage_scores": str(args.cage_scores.resolve()),
        "protein_dir": str(args.protein_dir.resolve()),
        "models": {key: str(value) for key, value in model_paths.items()},
        "n_pairs": int(len(pair_frame)),
        "n_proteins": int(pair_frame["uniprot_id"].nunique()),
        "n_reactions": int(pair_frame["reaction_id"].nunique()),
        "n_positive_pairs": int(pair_frame["label"].sum()),
        "pure_cage_definition": "raw EnzymeCAGE pair probability, no reaction/sequence/neural feature; few-shot only removes supplied seed positives from candidates",
        "budgets": list(budgets),
        "fewshot_seed_counts": list(seed_counts),
        "fewshot_repeats": args.fewshot_repeats,
        "limitations": [
            "The common reservoir contains only pairs historically scored by EnzymeCAGE; it is not the full general candidate universe.",
            "Broad-adapted neural experts have seen many integrated associations, so this benchmark measures retrieval/recovery capability rather than strict unseen generalization.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
