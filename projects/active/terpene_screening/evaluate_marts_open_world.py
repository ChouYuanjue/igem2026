from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_reaction,
    ensemble_similarity,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_reaction_library,
    normalize_rows,
    prepare_reaction_neighbor_index,
    protein_neighbor_reaction_transfer_scores,
    reaction_embedding_ensemble,
    reaction_neighbor_transfer_scores,
    tied_rank_percentile,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_MARTS_PAIRS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_EXTERNAL_PROTEIN_DIR = ROOT / "data/terpene_embeddings/marts_unseen_esmc600m"
DEFAULT_CURRENT_PROTEIN_DIR = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CURRENT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_R2E_DIR = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_E2R_DIR = ROOT / "results/terpene_production_models/multiview"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_open_world"
DEFAULT_BUDGETS = (3, 10, 20)


def stable_external_reaction_id(signature: str) -> str:
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"MARTS_EXT_RXN_{digest}"


def random_hit_probability(candidate_count: int, positive_count: int, budget: int) -> float:
    if positive_count <= 0 or candidate_count <= 0 or budget <= 0:
        return 0.0
    budget = min(budget, candidate_count)
    if budget > candidate_count - positive_count:
        return 1.0
    log_miss = 0.0
    for offset in range(budget):
        log_miss += math.log(candidate_count - positive_count - offset) - math.log(candidate_count - offset)
    return 1.0 - math.exp(log_miss)


def append_random_metrics(metrics: dict[str, object], candidate_count: int, budgets: tuple[int, ...]) -> None:
    positive_count = int(metrics["n_positives"])
    for budget in budgets:
        random_probability = random_hit_probability(candidate_count, positive_count, budget)
        metrics[f"random_hit_probability_at_{budget}"] = random_probability
        observed = float(metrics[f"hit_at_{budget}"])
        metrics[f"hit_enrichment_at_{budget}"] = observed / random_probability if random_probability > 0 else None


def aggregate_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_positive_count": ("n_positives", "mean"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
        "candidate_count": ("candidate_count", "first"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
        aggregations[f"random_hit_probability_at_{budget}"] = (f"random_hit_probability_at_{budget}", "mean")
    result = frame.groupby(["direction", "category", "method"]).agg(**aggregations).reset_index()
    for budget in budgets:
        denominator = result[f"random_hit_probability_at_{budget}"].replace(0, np.nan)
        result[f"enrichment_over_random_at_{budget}"] = result[f"hit_probability_at_{budget}"] / denominator
    return result


def build_current_maps(
    current_positives_path: Path,
    current_candidates_path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    positives = pd.read_csv(current_positives_path, sep="\t", dtype=str).fillna("")
    positives["reaction_signature"] = positives["smiles_seq"].map(reaction_signature)
    proteins_by_signature = (
        positives[positives["reaction_signature"] != ""]
        .groupby("reaction_signature")["Entry"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    reactions_by_entry = positives.groupby("Entry")["rhea_id"].apply(lambda values: set(values.astype(str))).to_dict()

    candidates = pd.read_csv(current_candidates_path, sep="\t", dtype=str).fillna("")
    candidates["Sequence"] = candidates["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    sequence_by_entry = dict(zip(candidates["Entry"].astype(str), candidates["Sequence"].astype(str)))
    entries_by_sequence = candidates.groupby("Sequence")["Entry"].apply(lambda values: sorted(set(values.astype(str)))).to_dict()
    return proteins_by_signature, reactions_by_entry, sequence_by_entry, entries_by_sequence


def evaluate_reaction_to_enzyme(
    marts: pd.DataFrame,
    external_protein_features: np.ndarray,
    external_protein_ids: list[str],
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    current_proteins_by_signature: dict[str, set[str]],
    production_dir: Path,
    positives_path: Path,
    topk_neighbor_reactions: int,
    budgets: tuple[int, ...],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_feature_schema(production_dir)
    models = load_models(production_dir / "models", "production", device)
    all_protein_features = np.concatenate([current_protein_features, external_protein_features], axis=0)
    all_protein_ids = current_protein_ids + external_protein_ids
    external_id_set = set(external_protein_ids)

    unseen_pairs = marts[(~marts["enzyme_seen"]) & (marts["reaction_signature"] != "")].copy()
    query_rows: list[dict[str, object]] = []
    query_features: list[np.ndarray] = []
    for signature, group in unseen_pairs.groupby("reaction_signature", sort=True):
        positives = sorted(set(group["enzyme_id"].astype(str)) & external_id_set)
        if not positives:
            continue
        reaction_smiles = str(group.iloc[0]["reaction_smiles"])
        category = "external_enzyme_seen_reaction" if bool(group["reaction_seen"].astype(bool).any()) else "external_enzyme_external_reaction"
        query_rows.append(
            {
                "query_id": stable_external_reaction_id(signature),
                "reaction_signature": signature,
                "reaction_smiles": reaction_smiles,
                "category": category,
                "positive_ids": positives,
                "known_ids": sorted(current_proteins_by_signature.get(signature, set())),
            }
        )
        query_features.append(encode_reaction(reaction_smiles, schema))
    query_matrix = np.stack(query_features).astype(np.float32)
    score_matrix = ensemble_similarity(models, all_protein_features, query_matrix, device)

    records: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []
    protein_to_index = {value: position for position, value in enumerate(all_protein_ids)}
    neighbor_index = prepare_reaction_neighbor_index(positives_path)
    for index, query in enumerate(query_rows):
        direct = score_matrix[index]
        neighbor = reaction_neighbor_transfer_scores(
            str(query["reaction_smiles"]),
            all_protein_features,
            all_protein_ids,
            positives_path,
            topk_neighbor_reactions,
            neighbor_index=neighbor_index,
        )
        score_map: dict[str, np.ndarray] = {"production_direct": direct}
        if neighbor is not None:
            score_map["reaction_neighbor_esmc_transfer"] = neighbor
            score_map["fixed_rank_neighbor_hybrid"] = (
                0.5 * tied_rank_percentile(direct, all_protein_ids)
                + 0.5 * tied_rank_percentile(neighbor, all_protein_ids)
            )
        for method, method_scores in score_map.items():
            metrics = rank_metrics(
                method_scores,
                all_protein_ids,
                set(query["positive_ids"]),
                set(query["known_ids"]),
                budgets,
            )
            append_random_metrics(metrics, len(all_protein_ids), budgets)
            records.append(
                {
                    "direction": "reaction_to_enzyme",
                    "category": query["category"],
                    "method": method,
                    "query_id": query["query_id"],
                    "candidate_count": len(all_protein_ids),
                    "known_associations_masked": len(query["known_ids"]),
                    **metrics,
                }
            )
            ranking_scores = method_scores.copy()
            for known_id in set(query["known_ids"]) - set(query["positive_ids"]):
                position = protein_to_index.get(known_id)
                if position is not None:
                    ranking_scores[position] = -np.inf
            order = [
                value
                for value in np.lexsort((np.asarray(all_protein_ids), -ranking_scores))
                if np.isfinite(ranking_scores[value])
            ][:20]
            rankings.append(
                pd.DataFrame(
                    {
                        "direction": "reaction_to_enzyme",
                        "category": query["category"],
                        "method": method,
                        "query_id": query["query_id"],
                        "rank": np.arange(1, len(order) + 1),
                        "candidate_id": [all_protein_ids[value] for value in order],
                        "score": ranking_scores[order],
                        "is_positive": [all_protein_ids[value] in set(query["positive_ids"]) for value in order],
                        "is_external_candidate": [all_protein_ids[value] in external_id_set for value in order],
                    }
                )
            )
    return pd.DataFrame(records), pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()


def evaluate_enzyme_to_reaction(
    marts: pd.DataFrame,
    external_protein_features: np.ndarray,
    external_protein_ids: list[str],
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    current_reactions_by_entry: dict[str, set[str]],
    sequence_by_entry: dict[str, str],
    entries_by_sequence: dict[str, list[str]],
    production_dir: Path,
    topk_neighbor_proteins: int,
    budgets: tuple[int, ...],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = load_feature_schema(production_dir)
    models = load_models(production_dir / "models", "production", device)
    current_reaction_features, current_reaction_ids = load_reaction_library(production_dir, schema)
    current_protein_to_row = {value: index for index, value in enumerate(current_protein_ids)}
    external_protein_to_row = {value: index for index, value in enumerate(external_protein_ids)}

    unseen_reaction_pairs = marts[(~marts["reaction_seen"]) & (marts["reaction_signature"] != "")].copy()
    signatures = sorted(set(unseen_reaction_pairs["reaction_signature"].astype(str)))
    external_reaction_ids = [stable_external_reaction_id(signature) for signature in signatures]
    signature_to_external_id = dict(zip(signatures, external_reaction_ids))
    representative = unseen_reaction_pairs.drop_duplicates("reaction_signature").set_index("reaction_signature")
    external_reaction_features = np.stack(
        [encode_reaction(str(representative.loc[signature, "reaction_smiles"]), schema) for signature in signatures]
    ).astype(np.float32)
    all_reaction_features = np.concatenate([current_reaction_features, external_reaction_features], axis=0)
    all_reaction_ids = current_reaction_ids + external_reaction_ids
    external_reaction_id_set = set(external_reaction_ids)

    query_rows: list[dict[str, object]] = []
    query_features: list[np.ndarray] = []
    for enzyme_id, group in unseen_reaction_pairs.groupby("enzyme_id", sort=True):
        relevant_signatures = sorted(set(group["reaction_signature"].astype(str)))
        positive_ids = [signature_to_external_id[value] for value in relevant_signatures]
        sequence = str(group.iloc[0]["sequence"])
        enzyme_seen = bool(group["enzyme_seen"].astype(bool).any())
        known_reactions: set[str] = set()
        if enzyme_id in current_protein_to_row:
            feature = current_protein_features[current_protein_to_row[enzyme_id]]
            known_reactions.update(current_reactions_by_entry.get(enzyme_id, set()))
        elif enzyme_id in external_protein_to_row:
            feature = external_protein_features[external_protein_to_row[enzyme_id]]
        else:
            matching_entries = entries_by_sequence.get(sequence, [])
            if not matching_entries:
                continue
            feature = current_protein_features[current_protein_to_row[matching_entries[0]]]
            for entry in matching_entries:
                known_reactions.update(current_reactions_by_entry.get(entry, set()))
        query_rows.append(
            {
                "query_id": enzyme_id,
                "category": "seen_enzyme_external_reaction" if enzyme_seen else "external_enzyme_external_reaction",
                "positive_ids": positive_ids,
                "known_ids": sorted(known_reactions),
            }
        )
        query_features.append(feature)
    query_matrix = normalize_rows(np.stack(query_features).astype(np.float32))
    score_matrix = ensemble_similarity(models, query_matrix, all_reaction_features, device).T
    reaction_embedding_sets = reaction_embedding_ensemble(models, all_reaction_features, device)
    protein_reaction_index = {
        entry: sorted(values)
        for entry, values in current_reactions_by_entry.items()
    }

    records: list[dict[str, object]] = []
    rankings: list[pd.DataFrame] = []
    reaction_to_index = {value: position for position, value in enumerate(all_reaction_ids)}
    for index, query in enumerate(query_rows):
        direct = score_matrix[index]
        neighbor = protein_neighbor_reaction_transfer_scores(
            query_matrix[index],
            current_protein_features,
            current_protein_ids,
            all_reaction_ids,
            reaction_embedding_sets,
            None,
            topk_neighbor_proteins,
            exclude_protein_id=str(query["query_id"]),
            protein_reaction_index=protein_reaction_index,
        )
        score_map: dict[str, np.ndarray] = {"production_direct": direct}
        if neighbor is not None:
            score_map["protein_neighbor_reaction_transfer"] = neighbor
            score_map["fixed_rank_neighbor_hybrid"] = (
                0.5 * tied_rank_percentile(direct, all_reaction_ids)
                + 0.5 * tied_rank_percentile(neighbor, all_reaction_ids)
            )
        for method, method_scores in score_map.items():
            metrics = rank_metrics(
                method_scores,
                all_reaction_ids,
                set(query["positive_ids"]),
                set(query["known_ids"]),
                budgets,
            )
            append_random_metrics(metrics, len(all_reaction_ids), budgets)
            records.append(
                {
                    "direction": "enzyme_to_reaction",
                    "category": query["category"],
                    "method": method,
                    "query_id": query["query_id"],
                    "candidate_count": len(all_reaction_ids),
                    "known_associations_masked": len(query["known_ids"]),
                    **metrics,
                }
            )
            ranking_scores = method_scores.copy()
            for known_id in set(query["known_ids"]) - set(query["positive_ids"]):
                position = reaction_to_index.get(known_id)
                if position is not None:
                    ranking_scores[position] = -np.inf
            order = [
                value
                for value in np.lexsort((np.asarray(all_reaction_ids), -ranking_scores))
                if np.isfinite(ranking_scores[value])
            ][:20]
            rankings.append(
                pd.DataFrame(
                    {
                        "direction": "enzyme_to_reaction",
                        "category": query["category"],
                        "method": method,
                        "query_id": query["query_id"],
                        "rank": np.arange(1, len(order) + 1),
                        "candidate_id": [all_reaction_ids[value] for value in order],
                        "score": ranking_scores[order],
                        "is_positive": [all_reaction_ids[value] in set(query["positive_ids"]) for value in order],
                        "is_external_candidate": [all_reaction_ids[value] in external_reaction_id_set for value in order],
                    }
                )
            )
    return pd.DataFrame(records), pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current production TPS models on MARTS-DB open-world entities.")
    parser.add_argument("--marts-pairs", type=Path, default=DEFAULT_MARTS_PAIRS)
    parser.add_argument("--external-protein-dir", type=Path, default=DEFAULT_EXTERNAL_PROTEIN_DIR)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEIN_DIR)
    parser.add_argument("--current-positives", type=Path, default=DEFAULT_CURRENT_POSITIVES)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--r2e-production-dir", type=Path, default=DEFAULT_R2E_DIR)
    parser.add_argument("--e2r-production-dir", type=Path, default=DEFAULT_E2R_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--topk-neighbor-reactions", type=int, default=5)
    parser.add_argument("--topk-neighbor-proteins", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marts = pd.read_csv(args.marts_pairs, sep="\t", dtype=str).fillna("")
    for column in ["enzyme_seen", "reaction_seen"]:
        marts[column] = marts[column].astype(str).str.lower().eq("true")
    current_protein_features, current_protein_ids = load_protein_library(args.current_protein_dir.resolve())
    external_protein_features, external_protein_ids = load_protein_library(args.external_protein_dir.resolve())
    current_proteins_by_signature, current_reactions_by_entry, sequence_by_entry, entries_by_sequence = build_current_maps(
        args.current_positives.resolve(),
        args.current_candidates.resolve(),
    )

    r2e_metrics, r2e_rankings = evaluate_reaction_to_enzyme(
        marts,
        external_protein_features,
        external_protein_ids,
        current_protein_features,
        current_protein_ids,
        current_proteins_by_signature,
        args.r2e_production_dir.resolve(),
        args.current_positives.resolve(),
        args.topk_neighbor_reactions,
        budgets,
        device,
    )
    e2r_metrics, e2r_rankings = evaluate_enzyme_to_reaction(
        marts,
        external_protein_features,
        external_protein_ids,
        current_protein_features,
        current_protein_ids,
        current_reactions_by_entry,
        sequence_by_entry,
        entries_by_sequence,
        args.e2r_production_dir.resolve(),
        args.topk_neighbor_proteins,
        budgets,
        device,
    )
    query_metrics = pd.concat([r2e_metrics, e2r_metrics], ignore_index=True)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    rankings = pd.concat([r2e_rankings, e2r_rankings], ignore_index=True)
    rankings.to_csv(output_dir / "top20_rankings.csv", index=False)
    aggregate = aggregate_metrics(query_metrics, budgets)
    aggregate.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "marts_pairs": str(args.marts_pairs.resolve()),
        "current_candidate_proteins": len(current_protein_ids),
        "external_candidate_proteins": len(external_protein_ids),
        "r2e_candidate_proteins": len(current_protein_ids) + len(external_protein_ids),
        "current_candidate_reactions": 513,
        "external_candidate_reactions": int(marts.loc[~marts["reaction_seen"], "reaction_signature"].replace("", pd.NA).nunique()),
        "budgets": budgets,
        "topk_neighbor_reactions": args.topk_neighbor_reactions,
        "topk_neighbor_proteins": args.topk_neighbor_proteins,
        "n_query_method_rows": int(len(query_metrics)),
        "n_unique_queries": int(query_metrics[["direction", "query_id"]].drop_duplicates().shape[0]),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "top20_rankings": str(output_dir / "top20_rankings.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
