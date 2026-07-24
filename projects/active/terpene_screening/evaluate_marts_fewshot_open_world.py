from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_marts_open_world import (  # noqa: E402
    build_current_maps,
    stable_external_reaction_id,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_reaction,
    ensemble_similarity,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_reaction_library,
    normalize_rows,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_MARTS_PAIRS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_EXTERNAL_PROTEIN_DIR = ROOT / "data/terpene_embeddings/marts_unseen_esmc600m"
DEFAULT_CURRENT_PROTEIN_DIR = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CURRENT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_R2E_DIR = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_E2R_DIR = ROOT / "results/terpene_production_models/multiview"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_fewshot_open_world"
DEFAULT_M_VALUES = (1, 2, 3)
DEFAULT_BUDGETS = (3, 10, 20)
DEFAULT_REPEATS = 20
DEFAULT_SEED = 20260723


def stable_seed(base_seed: int, direction: str, query_id: str, m: int, rep: int) -> int:
    payload = f"{base_seed}|{direction}|{query_id}|{m}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


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


def append_records(
    records: list[dict[str, object]],
    direction: str,
    category: str,
    query_id: str,
    m: int,
    rep: int,
    method: str,
    scores: np.ndarray,
    candidate_ids: list[str],
    hidden: set[str],
    masked: set[str],
    budgets: tuple[int, ...],
) -> None:
    metrics = rank_metrics(scores, candidate_ids, hidden, masked, budgets)
    records.append(
        {
            "direction": direction,
            "category": category,
            "query_id": query_id,
            "m": m,
            "rep": rep,
            "method": method,
            "candidate_count": len(candidate_ids),
            "n_hidden": len(hidden),
            "n_masked": len(masked),
            **metrics,
        }
    )


def load_reaction_embedding_ensembles(
    production_dir: Path,
    reaction_features: np.ndarray,
    device: torch.device,
) -> tuple[list[object], list[np.ndarray]]:
    models = load_models(production_dir / "models", "production", device)
    tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            embeddings.append(model.encode_reactions(tensor).cpu().numpy())
    return models, embeddings


def evaluate_r2e(
    marts: pd.DataFrame,
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    external_protein_features: np.ndarray,
    external_protein_ids: list[str],
    current_proteins_by_signature: dict[str, set[str]],
    production_dir: Path,
    m_values: tuple[int, ...],
    repeats: int,
    budgets: tuple[int, ...],
    base_seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    schema = load_feature_schema(production_dir)
    models = load_models(production_dir / "models", "production", device)
    candidate_features = np.concatenate([current_protein_features, external_protein_features], axis=0)
    candidate_ids = current_protein_ids + external_protein_ids
    candidate_to_row = {value: index for index, value in enumerate(candidate_ids)}
    external_id_set = set(external_protein_ids)

    pairs = marts[(~marts["enzyme_seen"]) & (marts["reaction_signature"] != "")].copy()
    query_rows: list[dict[str, object]] = []
    query_features: list[np.ndarray] = []
    for signature, group in pairs.groupby("reaction_signature", sort=True):
        positives = sorted(set(group["enzyme_id"].astype(str)) & external_id_set)
        if len(positives) < 2:
            continue
        query_id = stable_external_reaction_id(signature)
        query_rows.append(
            {
                "query_id": query_id,
                "category": "seen_reaction" if bool(group["reaction_seen"].any()) else "external_reaction",
                "positive_ids": positives,
                "known_ids": sorted(current_proteins_by_signature.get(signature, set())),
            }
        )
        query_features.append(encode_reaction(str(group.iloc[0]["reaction_smiles"]), schema))
    query_matrix = np.stack(query_features).astype(np.float32)
    direct_scores = ensemble_similarity(models, candidate_features, query_matrix, device)

    records: list[dict[str, object]] = []
    for query_index, query in enumerate(query_rows):
        positives = list(query["positive_ids"])
        for m in m_values:
            if len(positives) < m + 1:
                continue
            for rep in range(repeats):
                rng = random.Random(stable_seed(base_seed, "r2e", str(query["query_id"]), m, rep))
                seeds = tuple(sorted(rng.sample(positives, m)))
                hidden = set(positives) - set(seeds)
                seed_rows = np.asarray([candidate_to_row[value] for value in seeds], dtype=np.int64)
                seed_scores = (candidate_features @ candidate_features[seed_rows].T).max(axis=1)
                direct = direct_scores[query_index]
                hybrid = 0.5 * tied_rank_percentile(direct, candidate_ids) + 0.5 * tied_rank_percentile(seed_scores, candidate_ids)
                masked = set(query["known_ids"]) | set(seeds)
                for method, scores in {
                    "production_direct": direct,
                    "seed_esmc_max": seed_scores,
                    "fixed_rank_hybrid": hybrid,
                }.items():
                    append_records(
                        records,
                        "reaction_to_enzyme",
                        str(query["category"]),
                        str(query["query_id"]),
                        m,
                        rep,
                        method,
                        scores,
                        candidate_ids,
                        hidden,
                        masked,
                        budgets,
                    )
    return records


def build_e2r_query_feature(
    enzyme_id: str,
    sequence: str,
    current_features: np.ndarray,
    current_ids: list[str],
    external_features: np.ndarray,
    external_ids: list[str],
    entries_by_sequence: dict[str, list[str]],
) -> tuple[np.ndarray | None, list[str]]:
    current_to_row = {value: index for index, value in enumerate(current_ids)}
    external_to_row = {value: index for index, value in enumerate(external_ids)}
    if enzyme_id in current_to_row:
        return current_features[current_to_row[enzyme_id]], [enzyme_id]
    if enzyme_id in external_to_row:
        return external_features[external_to_row[enzyme_id]], []
    matching_entries = entries_by_sequence.get(sequence, [])
    if matching_entries:
        return current_features[current_to_row[matching_entries[0]]], matching_entries
    return None, []


def evaluate_e2r(
    marts: pd.DataFrame,
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    external_protein_features: np.ndarray,
    external_protein_ids: list[str],
    current_reactions_by_entry: dict[str, set[str]],
    entries_by_sequence: dict[str, list[str]],
    production_dir: Path,
    m_values: tuple[int, ...],
    repeats: int,
    budgets: tuple[int, ...],
    base_seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    schema = load_feature_schema(production_dir)
    current_reaction_features, current_reaction_ids = load_reaction_library(production_dir, schema)
    pairs = marts[(~marts["reaction_seen"]) & (marts["reaction_signature"] != "")].copy()
    signatures = sorted(set(pairs["reaction_signature"].astype(str)))
    external_reaction_ids = [stable_external_reaction_id(value) for value in signatures]
    signature_to_id = dict(zip(signatures, external_reaction_ids))
    representative = pairs.drop_duplicates("reaction_signature").set_index("reaction_signature")
    external_reaction_features = np.stack(
        [encode_reaction(str(representative.loc[value, "reaction_smiles"]), schema) for value in signatures]
    ).astype(np.float32)
    candidate_features = np.concatenate([current_reaction_features, external_reaction_features], axis=0)
    candidate_ids = current_reaction_ids + external_reaction_ids
    candidate_to_row = {value: index for index, value in enumerate(candidate_ids)}
    models, reaction_embedding_sets = load_reaction_embedding_ensembles(production_dir, candidate_features, device)

    query_rows: list[dict[str, object]] = []
    query_features: list[np.ndarray] = []
    for enzyme_id, group in pairs.groupby("enzyme_id", sort=True):
        positives = sorted({signature_to_id[value] for value in set(group["reaction_signature"].astype(str))})
        if len(positives) < 2:
            continue
        sequence = str(group.iloc[0]["sequence"])
        feature, matching_current_entries = build_e2r_query_feature(
            enzyme_id,
            sequence,
            current_protein_features,
            current_protein_ids,
            external_protein_features,
            external_protein_ids,
            entries_by_sequence,
        )
        if feature is None:
            continue
        known_reactions: set[str] = set()
        for entry in matching_current_entries:
            known_reactions.update(current_reactions_by_entry.get(entry, set()))
        query_rows.append(
            {
                "query_id": enzyme_id,
                "category": "seen_enzyme" if bool(group["enzyme_seen"].any()) else "external_enzyme",
                "positive_ids": positives,
                "known_ids": sorted(known_reactions),
            }
        )
        query_features.append(feature)
    query_matrix = normalize_rows(np.stack(query_features).astype(np.float32))
    direct_scores = ensemble_similarity(models, query_matrix, candidate_features, device).T

    records: list[dict[str, object]] = []
    for query_index, query in enumerate(query_rows):
        positives = list(query["positive_ids"])
        for m in m_values:
            if len(positives) < m + 1:
                continue
            for rep in range(repeats):
                rng = random.Random(stable_seed(base_seed, "e2r", str(query["query_id"]), m, rep))
                seeds = tuple(sorted(rng.sample(positives, m)))
                hidden = set(positives) - set(seeds)
                seed_rows = np.asarray([candidate_to_row[value] for value in seeds], dtype=np.int64)
                seed_scores = np.zeros(len(candidate_ids), dtype=np.float32)
                for embeddings in reaction_embedding_sets:
                    seed_scores += (embeddings @ embeddings[seed_rows].T).max(axis=1)
                seed_scores /= len(reaction_embedding_sets)
                direct = direct_scores[query_index]
                hybrid = 0.5 * tied_rank_percentile(direct, candidate_ids) + 0.5 * tied_rank_percentile(seed_scores, candidate_ids)
                masked = set(query["known_ids"]) | set(seeds)
                for method, scores in {
                    "production_direct": direct,
                    "seed_reaction_embedding_max": seed_scores,
                    "fixed_rank_hybrid": hybrid,
                }.items():
                    append_records(
                        records,
                        "enzyme_to_reaction",
                        str(query["category"]),
                        str(query["query_id"]),
                        m,
                        rep,
                        method,
                        scores,
                        candidate_ids,
                        hidden,
                        masked,
                        budgets,
                    )
    return records


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_trials": ("query_id", "size"),
        "n_unique_queries": ("query_id", "nunique"),
        "mean_hidden_positives": ("n_hidden", "mean"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["direction", "category", "m", "method"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MARTS open-world few-shot bidirectional retrieval.")
    parser.add_argument("--marts-pairs", type=Path, default=DEFAULT_MARTS_PAIRS)
    parser.add_argument("--external-protein-dir", type=Path, default=DEFAULT_EXTERNAL_PROTEIN_DIR)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEIN_DIR)
    parser.add_argument("--current-positives", type=Path, default=DEFAULT_CURRENT_POSITIVES)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--r2e-production-dir", type=Path, default=DEFAULT_R2E_DIR)
    parser.add_argument("--e2r-production-dir", type=Path, default=DEFAULT_E2R_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m-values", default=",".join(str(value) for value in DEFAULT_M_VALUES))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    m_values = tuple(int(value) for value in args.m_values.split(",") if value)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marts = pd.read_csv(args.marts_pairs, sep="\t", dtype=str).fillna("")
    for column in ["enzyme_seen", "reaction_seen"]:
        marts[column] = marts[column].astype(str).str.lower().eq("true")
    current_features, current_ids = load_protein_library(args.current_protein_dir.resolve())
    external_features, external_ids = load_protein_library(args.external_protein_dir.resolve())
    current_proteins_by_signature, current_reactions_by_entry, _, entries_by_sequence = build_current_maps(
        args.current_positives.resolve(),
        args.current_candidates.resolve(),
    )

    records = evaluate_r2e(
        marts,
        current_features,
        current_ids,
        external_features,
        external_ids,
        current_proteins_by_signature,
        args.r2e_production_dir.resolve(),
        m_values,
        args.repeats,
        budgets,
        args.seed,
        device,
    )
    records.extend(
        evaluate_e2r(
            marts,
            current_features,
            current_ids,
            external_features,
            external_ids,
            current_reactions_by_entry,
            entries_by_sequence,
            args.e2r_production_dir.resolve(),
            m_values,
            args.repeats,
            budgets,
            args.seed,
            device,
        )
    )
    long = pd.DataFrame(records)
    long.to_csv(output_dir / "metrics_long.csv", index=False)
    metrics = aggregate(long, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    best = (
        metrics.sort_values(
            ["direction", "category", "m", "hit_probability_at_10", "mean_reciprocal_rank"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["direction", "category", "m"], as_index=False)
        .head(1)
    )
    best.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "m_values": m_values,
        "budgets": budgets,
        "repeats": args.repeats,
        "n_trials": int(len(long)),
        "n_unique_queries": int(long["query_id"].nunique()),
        "outputs": {
            "metrics_long": str(output_dir / "metrics_long.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
