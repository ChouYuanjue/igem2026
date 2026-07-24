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

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_DIR,
    DEFAULT_R2E_DUAL_TOWER_DIR,
    DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    annotate_candidate_uncertainty,
    encode_reaction,
    ensemble_similarity_members,
    load_external_reaction_rows,
    load_feature_schema,
    load_models,
    load_protein_library,
    sort_scores,
)
from projects.active.terpene_screening.rank_registry_batch import (  # noqa: E402
    DEFAULT_MARTS,
    build_known_association_maps,
)

DEFAULT_UNIPROT_PROTEINS = ROOT / "data/terpene_embeddings/uniprot_tps_primary_esmc600m"
DEFAULT_UNIPROT_METADATA = (
    ROOT / "data/terpene_uniprot_expansion/uniprot_tps_primary_embedding_candidates.tsv"
)
DEFAULT_CANONICAL_BATCH = ROOT / "results/terpene_registry_batch"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_expanded_r2e"
DEFAULT_OBJECTIVES = (3, 10, 20)


def load_candidate_universe(
    current_dir: Path,
    registered_dir: Path,
    uniprot_dir: Path,
    uniprot_metadata_path: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    current_features, current_ids = load_protein_library(current_dir)
    registered_features, registered_ids = load_protein_library(registered_dir)
    uniprot_features, uniprot_ids = load_protein_library(uniprot_dir)
    all_ids = current_ids + registered_ids + uniprot_ids
    if len(all_ids) != len(set(all_ids)):
        duplicates = pd.Series(all_ids).value_counts()
        raise ValueError(
            f"Duplicate protein IDs across candidate layers: {duplicates[duplicates > 1].index[:20].tolist()}"
        )
    all_features = np.concatenate(
        [current_features, registered_features, uniprot_features], axis=0
    ).astype(np.float32, copy=False)
    metadata = pd.DataFrame(
        {
            "candidate_id": all_ids,
            "candidate_source": (
                ["current"] * len(current_ids)
                + ["marts_registered"] * len(registered_ids)
                + ["uniprot_primary"] * len(uniprot_ids)
            ),
        }
    )
    uniprot_metadata = pd.read_csv(
        uniprot_metadata_path, sep="\t", dtype=str
    ).fillna("")
    uniprot_metadata = uniprot_metadata.rename(columns={"accession": "candidate_id"})
    available_columns = [
        "candidate_id",
        "entry_name",
        "reviewed",
        "protein_name",
        "gene_names",
        "organism_name",
        "organism_id",
        "length",
        "pfam_combination",
        "domain_family",
        "protein_existence",
        "evidence_quality_tier",
        "cluster_id",
        "cluster_size",
        "selection_reason",
    ]
    available_columns = [
        column for column in available_columns if column in uniprot_metadata.columns
    ]
    metadata = metadata.merge(
        uniprot_metadata[available_columns], on="candidate_id", how="left"
    ).fillna("")
    metadata["evidence_quality_tier"] = np.where(
        metadata["candidate_source"].eq("current"),
        "current_curated",
        np.where(
            metadata["candidate_source"].eq("marts_registered"),
            "marts_registered",
            metadata["evidence_quality_tier"],
        ),
    )
    return all_features, all_ids, metadata


def build_reaction_queries(
    model_dir: Path,
    registered_reactions_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    schema = load_feature_schema(model_dir)
    reactions = load_external_reaction_rows(registered_reactions_path)
    features = np.stack(
        [encode_reaction(value, schema) for value in reactions["reaction_smiles"]]
    ).astype(np.float32)
    return features, reactions, schema


def score_model(
    model_dir: Path,
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    models = load_models(model_dir / "models", "production", device)
    return ensemble_similarity_members(
        models, protein_features, reaction_features, device
    )


def annotate_results(
    result: pd.DataFrame,
    candidate_metadata: pd.DataFrame,
    query_id: str,
    objective: str,
    model_dir: Path,
    known_count: int,
) -> pd.DataFrame:
    result.insert(0, "query_id", query_id)
    result.insert(1, "direction", "reaction_to_enzyme")
    result.insert(2, "ranking_objective", objective)
    result.insert(3, "score_source", "direct")
    result.insert(4, "model_directory", str(model_dir.resolve()))
    result.insert(5, "known_associations_masked", known_count)
    result.insert(6, "empirical_reliability_status", "not_applicable_candidate_universe_expanded")
    return result.merge(candidate_metadata, on="candidate_id", how="left")


def canonical_sets(batch_dir: Path) -> dict[tuple[str, str], set[str]]:
    path = batch_dir / "reaction_to_enzyme_rankings.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        (str(query_id), str(objective)): set(group["candidate_id"].astype(str))
        for (query_id, objective), group in frame.groupby(
            ["query_id", "ranking_objective"], sort=False
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-tiered UniProt expansion for R2E retrieval."
    )
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR)
    parser.add_argument("--uniprot-protein-dir", type=Path, default=DEFAULT_UNIPROT_PROTEINS)
    parser.add_argument("--uniprot-metadata", type=Path, default=DEFAULT_UNIPROT_METADATA)
    parser.add_argument("--registered-reactions", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--r2e-shared-model-dir", type=Path, default=DEFAULT_R2E_DUAL_TOWER_DIR)
    parser.add_argument("--r2e-short-model-dir", type=Path, default=DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR)
    parser.add_argument("--canonical-batch-dir", type=Path, default=DEFAULT_CANONICAL_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rescue-top-k", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protein_features, protein_ids, candidate_metadata = load_candidate_universe(
        args.current_protein_dir.resolve(),
        args.registered_protein_dir.resolve(),
        args.uniprot_protein_dir.resolve(),
        args.uniprot_metadata.resolve(),
    )
    registered_reactions = load_external_reaction_rows(args.registered_reactions.resolve())
    registered_reaction_ids = set(registered_reactions["reaction_id"].astype(str))
    _, known_enzymes_by_reaction = build_known_association_maps(
        args.marts.resolve(), args.positives.resolve(), registered_reaction_ids
    )
    protein_id_set = set(protein_ids)
    canonical = canonical_sets(args.canonical_batch_dir.resolve())

    all_rankings: list[pd.DataFrame] = []
    rescue_rankings: list[pd.DataFrame] = []
    query_rows: list[dict[str, object]] = []
    model_specs = [
        (args.r2e_short_model_dir.resolve(), (3, 10)),
        (args.r2e_shared_model_dir.resolve(), (20,)),
    ]
    score_cache: dict[Path, tuple[np.ndarray, pd.DataFrame]] = {}
    for model_dir, objectives in model_specs:
        reaction_features, reaction_table, _ = build_reaction_queries(
            model_dir, args.registered_reactions.resolve()
        )
        member_scores = score_model(
            model_dir, protein_features, reaction_features, device
        )
        score_cache[model_dir] = (member_scores, reaction_table)
        for query_index, row in enumerate(reaction_table.itertuples(index=False)):
            query_id = str(row.reaction_id)
            known = set(known_enzymes_by_reaction.get(query_id, set())) & protein_id_set
            query_member_scores = member_scores[:, query_index, :]
            mean_scores = query_member_scores.mean(axis=0)
            for top_k in objectives:
                objective = f"top{top_k}"
                result = sort_scores(protein_ids, mean_scores, known, top_k)
                result = annotate_candidate_uncertainty(
                    result, protein_ids, query_member_scores, known, top_k
                )
                result = annotate_results(
                    result,
                    candidate_metadata,
                    query_id,
                    objective,
                    model_dir,
                    len(known),
                )
                all_rankings.append(result)
                source_counts = result["candidate_source"].value_counts().to_dict()
                canonical_set = canonical.get((query_id, objective), set())
                expanded_set = set(result["candidate_id"].astype(str))
                uniprot_rows = result[result["candidate_source"].eq("uniprot_primary")]
                query_rows.append(
                    {
                        "query_id": query_id,
                        "ranking_objective": objective,
                        "candidate_count": len(protein_ids) - len(known),
                        "known_associations_masked": len(known),
                        "canonical_overlap_count": len(canonical_set & expanded_set),
                        "canonical_overlap_fraction": (
                            len(canonical_set & expanded_set) / len(canonical_set)
                            if canonical_set
                            else np.nan
                        ),
                        "current_candidates": int(source_counts.get("current", 0)),
                        "marts_registered_candidates": int(
                            source_counts.get("marts_registered", 0)
                        ),
                        "uniprot_primary_candidates": int(
                            source_counts.get("uniprot_primary", 0)
                        ),
                        "uniprot_primary_fraction": float(
                            source_counts.get("uniprot_primary", 0) / len(result)
                        ),
                        "best_uniprot_rank": (
                            int(uniprot_rows["rank"].min()) if len(uniprot_rows) else None
                        ),
                        "uniprot_evidence_tiers": ";".join(
                            sorted(set(uniprot_rows["evidence_quality_tier"].astype(str)) - {""})
                        ),
                    }
                )

    shared_model_dir = args.r2e_shared_model_dir.resolve()
    shared_members, shared_reactions = score_cache[shared_model_dir]
    for query_index, row in enumerate(shared_reactions.itertuples(index=False)):
        query_id = str(row.reaction_id)
        known = set(known_enzymes_by_reaction.get(query_id, set())) & protein_id_set
        query_member_scores = shared_members[:, query_index, :]
        result = sort_scores(
            protein_ids, query_member_scores.mean(axis=0), known, args.rescue_top_k
        )
        result = annotate_candidate_uncertainty(
            result,
            protein_ids,
            query_member_scores,
            known,
            args.rescue_top_k,
        )
        result = annotate_results(
            result,
            candidate_metadata,
            query_id,
            f"top{args.rescue_top_k}_rescue",
            shared_model_dir,
            len(known),
        )
        rescue_rankings.append(result)

    rankings = pd.concat(all_rankings, ignore_index=True)
    rescue = pd.concat(rescue_rankings, ignore_index=True)
    queries = pd.DataFrame(query_rows)
    rankings.to_csv(output_dir / "expanded_rankings.csv", index=False)
    rescue.to_csv(output_dir / "expanded_top100_rescue_rankings.csv", index=False)
    queries.to_csv(output_dir / "expanded_query_summary.csv", index=False)
    candidate_metadata.to_csv(output_dir / "expanded_candidate_registry.csv", index=False)

    objective_summary = (
        queries.groupby("ranking_objective")
        .agg(
            n_queries=("query_id", "size"),
            mean_canonical_overlap_fraction=("canonical_overlap_fraction", "mean"),
            median_canonical_overlap_fraction=("canonical_overlap_fraction", "median"),
            mean_uniprot_primary_fraction=("uniprot_primary_fraction", "mean"),
            queries_with_uniprot_candidate=(
                "uniprot_primary_candidates",
                lambda values: int((values > 0).sum()),
            ),
            median_best_uniprot_rank=("best_uniprot_rank", "median"),
        )
        .reset_index()
    )
    objective_summary.to_csv(output_dir / "objective_summary.csv", index=False)
    uniprot_top = rankings[rankings["candidate_source"].eq("uniprot_primary")].copy()
    uniprot_frequency = (
        uniprot_top.groupby(
            ["ranking_objective", "candidate_id", "evidence_quality_tier", "domain_family"]
        )
        .agg(
            appearance_count=("query_id", "size"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            mean_ensemble_score=("ensemble_score_mean", "mean"),
        )
        .reset_index()
        .sort_values(
            ["ranking_objective", "appearance_count", "best_rank"],
            ascending=[True, False, True],
        )
    )
    uniprot_frequency.to_csv(output_dir / "uniprot_candidate_frequency.csv", index=False)
    summary = {
        "candidate_universe": {
            "total": len(protein_ids),
            **candidate_metadata["candidate_source"].value_counts().to_dict(),
        },
        "registered_reactions": len(registered_reactions),
        "known_associations_masked": int(
            sum(len(set(value) & protein_id_set) for value in known_enzymes_by_reaction.values())
        ),
        "empirical_reliability_status": "not_applicable_candidate_universe_expanded",
        "objectives": objective_summary.to_dict("records"),
        "uniprot_evidence_tiers_in_top_lists": uniprot_top[
            "evidence_quality_tier"
        ].value_counts().to_dict(),
        "uniprot_domain_families_in_top_lists": uniprot_top[
            "domain_family"
        ].value_counts().to_dict(),
        "outputs": {
            "rankings": str(output_dir / "expanded_rankings.csv"),
            "query_summary": str(output_dir / "expanded_query_summary.csv"),
            "rescue_rankings": str(
                output_dir / "expanded_top100_rescue_rankings.csv"
            ),
            "objective_summary": str(output_dir / "objective_summary.csv"),
            "candidate_registry": str(output_dir / "expanded_candidate_registry.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(objective_summary.to_string(index=False))


if __name__ == "__main__":
    main()
