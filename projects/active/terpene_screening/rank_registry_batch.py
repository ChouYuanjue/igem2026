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

from projects.active.terpene_screening.evaluate_marts_open_world import (  # noqa: E402
    stable_external_reaction_id,
)
from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (  # noqa: E402
    reaction_features as zero_shot_reaction_features,
    reaction_similarity as zero_shot_reaction_similarity,
)
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_E2R_DUAL_TOWER_DIR,
    DEFAULT_E2R_HARDNEG_DUAL_TOWER_DIR,
    E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
    E2R_TOP10_RRF_CONSTANT,
    E2R_TOP10_RRF_PRIMARY_WEIGHT,
    E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
    E2R_TOP10_SECONDARY_NEIGHBOR_K,
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_DIR,
    DEFAULT_R2E_DUAL_TOWER_DIR,
    DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR,
    DEFAULT_R2E_TOP10_20_DUAL_TOWER_DIR,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    DEFAULT_UNCERTAINTY_CALIBRATORS,
    annotate_candidate_uncertainty,
    apply_empirical_reliability,
    choose_retrieval_scores,
    encode_reaction,
    encode_exact_horizyn_reactions,
    ensemble_similarity_members,
    load_auxiliary_reaction_library,
    load_external_reaction_rows,
    load_feature_schema,
    load_models,
    load_protein_library,
    models_require_auxiliary_reaction_features,
    load_reaction_library,
    nearest_protein_similarity,
    prepare_reaction_neighbor_index,
    protein_neighbor_reaction_transfer_scores,
    reaction_embedding_ensemble,
    reciprocal_rank_fusion_members,
    reciprocal_rank_fusion_scores,
    route_member_scores,
    sort_scores,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_registry_batch"
DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_OBJECTIVES = (3, 10, 20)




def build_known_association_maps(
    marts_path: Path,
    positives_path: Path,
    registered_reaction_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    positives = pd.read_csv(positives_path, sep="	", dtype=str).fillna("")
    positives["reaction_signature"] = positives["smiles_seq"].map(reaction_signature)
    current_by_signature = (
        positives[positives["reaction_signature"] != ""]
        .groupby("reaction_signature")["rhea_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    marts = pd.read_csv(marts_path, sep="	", dtype=str).fillna("")
    reactions_by_enzyme: dict[str, set[str]] = {}
    enzymes_by_reaction: dict[str, set[str]] = {}
    for row in marts[["enzyme_id", "reaction_signature"]].itertuples(index=False):
        enzyme_id = str(row.enzyme_id)
        signature = str(row.reaction_signature)
        if not enzyme_id or not signature:
            continue
        reaction_ids = set(current_by_signature.get(signature, set()))
        external_id = stable_external_reaction_id(signature)
        if external_id in registered_reaction_ids:
            reaction_ids.add(external_id)
        for reaction_id in reaction_ids:
            reactions_by_enzyme.setdefault(enzyme_id, set()).add(reaction_id)
            enzymes_by_reaction.setdefault(reaction_id, set()).add(enzyme_id)
    return reactions_by_enzyme, enzymes_by_reaction


def objective_name(top_k: int) -> str:
    return f"top{top_k}"


def policy_accepts(status: str, tier: str, policy: str) -> bool:
    if policy == "annotate":
        return True
    calibrated = status == "validated_external_double_cold"
    if policy == "require_calibrated":
        return calibrated
    if policy == "require_intermediate":
        return calibrated and tier in {"intermediate", "higher_evidence"}
    if policy == "require_higher":
        return calibrated and tier == "higher_evidence"
    raise ValueError(f"Unknown reliability policy: {policy}")


def add_common_columns(
    result: pd.DataFrame,
    *,
    query_id: str,
    direction: str,
    score_source: str,
    ranking_objective: str,
    model_directory: Path,
    secondary_model_directory: Path | None = None,
    nearest_id: str | None,
    nearest_similarity: float,
    external_candidates: set[str],
) -> pd.DataFrame:
    result.insert(0, "query_id", query_id)
    result.insert(1, "direction", direction)
    result.insert(2, "score_source", score_source)
    result.insert(3, "ranking_objective", ranking_objective)
    result.insert(4, "model_directory", str(model_directory.resolve()))
    result.insert(5, "secondary_model_directory", (
        str(secondary_model_directory.resolve()) if secondary_model_directory is not None else ""
    ))
    result.insert(6, "query_nearest_library_id", nearest_id)
    result.insert(7, "query_nearest_library_similarity", nearest_similarity)
    result.insert(8, "query_is_current_entity", False)
    result["is_external_candidate"] = result["candidate_id"].isin(external_candidates)
    return result


def query_summary(result: pd.DataFrame, accepted: bool) -> dict[str, object]:
    row = result.iloc[0]
    return {
        "query_id": row["query_id"],
        "direction": row["direction"],
        "ranking_objective": row["ranking_objective"],
        "model_directory": row["model_directory"],
        "secondary_model_directory": row.get("secondary_model_directory", ""),
        "score_source": row["score_source"],
        "top1_candidate_id": row["candidate_id"],
        "top1_score": row["score"],
        "query_nearest_library_id": row["query_nearest_library_id"],
        "query_nearest_library_similarity": row["query_nearest_library_similarity"],
        "ensemble_top1_vote_fraction": row["ensemble_top1_vote_fraction"],
        "ensemble_top1_rank_std": row["ensemble_top1_rank_std"],
        "ensemble_top1_margin_z": row["ensemble_top1_margin_z"],
        "ensemble_topk_jaccard": row["ensemble_topk_jaccard"],
        "ensemble_topk_vote_mean": row["ensemble_topk_vote_mean"],
        "empirical_reliability_score": row["empirical_reliability_score"],
        "empirical_reliability_tier": row["empirical_reliability_tier"],
        "empirical_reliability_status": row["empirical_reliability_status"],
        "reliability_recommendation": row["reliability_recommendation"],
        "known_associations_masked": int(row["known_associations_masked"]),
        "accepted_by_policy": bool(accepted),
    }


def nearest_registered_reaction(
    reaction_smiles: str,
    neighbor_index: tuple[list[str], dict[str, dict[str, object]], dict[str, list[str]]],
) -> tuple[str | None, float]:
    reaction_ids, features, _ = neighbor_index
    query = zero_shot_reaction_features(reaction_smiles)
    values = [
        (reaction_id, float(zero_shot_reaction_similarity(query, features[reaction_id])))
        for reaction_id in reaction_ids
    ]
    if not values:
        return None, float("nan")
    values.sort(key=lambda item: (-item[1], item[0]))
    return values[0]


def rank_registered_enzymes(
    *,
    objectives: tuple[int, ...],
    max_queries: int | None,
    output_dir: Path,
    current_protein_dir: Path,
    registered_protein_dir: Path,
    registered_reactions_csv: Path,
    positives: Path,
    calibrators: Path,
    model_dir: Path,
    secondary_model_dir: Path,
    reliability_policy: str,
    topk_neighbor_proteins: int,
    known_reactions_by_enzyme: dict[str, set[str]],
    mask_known_associations: bool,
    device: torch.device,
) -> list[dict[str, object]]:
    current_proteins, current_ids = load_protein_library(current_protein_dir)
    registered_proteins, registered_ids = load_protein_library(registered_protein_dir)
    if max_queries is not None:
        registered_proteins = registered_proteins[:max_queries]
        registered_ids = registered_ids[:max_queries]

    schema = load_feature_schema(model_dir)
    models = load_models(model_dir / "models", "production", device)
    secondary_schema = load_feature_schema(secondary_model_dir)
    secondary_models = load_models(secondary_model_dir / "models", "production", device)
    if [str(value) for value in secondary_schema.get("reaction_ids", [])] != [
        str(value) for value in schema.get("reaction_ids", [])
    ]:
        raise ValueError("Primary and hard-negative E2R batch deployments use different reaction IDs")
    if len(secondary_models) != len(models):
        raise ValueError("Primary and hard-negative E2R batch ensembles differ in size")
    reaction_features, reaction_ids = load_reaction_library(model_dir, schema)
    registered_reactions = load_external_reaction_rows(registered_reactions_csv)
    existing = set(reaction_ids)
    external_rows = [
        (row.reaction_id, encode_reaction(row.reaction_smiles, schema))
        for row in registered_reactions.itertuples(index=False)
        if row.reaction_id not in existing
    ]
    if external_rows:
        reaction_ids.extend([value[0] for value in external_rows])
        reaction_features = np.concatenate(
            [reaction_features, np.stack([value[1] for value in external_rows])], axis=0
        )
    external_reaction_ids = set(registered_reactions["reaction_id"].astype(str))

    direct_members = ensemble_similarity_members(
        models, registered_proteins, reaction_features, device
    ).transpose(0, 2, 1)
    reaction_embedding_sets = reaction_embedding_ensemble(models, reaction_features, device)
    secondary_direct_members = ensemble_similarity_members(
        secondary_models, registered_proteins, reaction_features, device
    ).transpose(0, 2, 1)
    secondary_reaction_embedding_sets = reaction_embedding_ensemble(
        secondary_models, reaction_features, device
    )
    all_rankings: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for query_index, query_id in enumerate(registered_ids):
        query_feature = registered_proteins[query_index]
        neighbor_scores = protein_neighbor_reaction_transfer_scores(
            query_feature,
            current_proteins,
            current_ids,
            reaction_ids,
            reaction_embedding_sets,
            positives,
            topk_neighbor_proteins,
            exclude_protein_id=None,
        )
        secondary_neighbor_scores = protein_neighbor_reaction_transfer_scores(
            query_feature,
            current_proteins,
            current_ids,
            reaction_ids,
            secondary_reaction_embedding_sets,
            positives,
            E2R_TOP10_SECONDARY_NEIGHBOR_K,
            exclude_protein_id=None,
        )
        nearest_id, nearest_similarity = nearest_protein_similarity(
            query_feature, current_proteins, current_ids
        )
        known_reaction_ids = (
            set(known_reactions_by_enzyme.get(query_id, set()))
            if mask_known_associations
            else set()
        )
        for top_k in objectives:
            ranking_objective = objective_name(top_k)
            direct_weight = {
                3: 0.75,
                10: E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
                20: 0.75,
            }[top_k]
            retrieval_mode = "neighbor_hybrid" if neighbor_scores is not None else "direct"
            primary_direct = direct_members[:, query_index, :]
            routed_members = route_member_scores(
                primary_direct,
                None,
                reaction_ids,
                retrieval_mode,
                neighbor_scores,
                direct_weight,
            )
            scores, score_source = choose_retrieval_scores(
                primary_direct.mean(axis=0),
                None,
                reaction_ids,
                retrieval_mode,
                neighbor_scores=neighbor_scores,
                hybrid_direct_weight=direct_weight,
            )
            secondary_output_dir: Path | None = None
            uncertainty_consensus: np.ndarray | None = None
            if top_k == 10:
                secondary_mode = (
                    "neighbor_hybrid" if secondary_neighbor_scores is not None else "direct"
                )
                secondary_direct = secondary_direct_members[:, query_index, :]
                secondary_scores, _ = choose_retrieval_scores(
                    secondary_direct.mean(axis=0),
                    None,
                    reaction_ids,
                    secondary_mode,
                    neighbor_scores=secondary_neighbor_scores,
                    hybrid_direct_weight=E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
                )
                secondary_routed_members = route_member_scores(
                    secondary_direct,
                    None,
                    reaction_ids,
                    secondary_mode,
                    secondary_neighbor_scores,
                    E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
                )
                scores = reciprocal_rank_fusion_scores(
                    scores,
                    secondary_scores,
                    reaction_ids,
                    E2R_TOP10_RRF_PRIMARY_WEIGHT,
                    E2R_TOP10_RRF_CONSTANT,
                )
                routed_members = reciprocal_rank_fusion_members(
                    routed_members,
                    secondary_routed_members,
                    reaction_ids,
                    E2R_TOP10_RRF_PRIMARY_WEIGHT,
                    E2R_TOP10_RRF_CONSTANT,
                )
                uncertainty_consensus = scores
                secondary_output_dir = secondary_model_dir
                score_source = "rrf_e2r_top10_primary0.35_secondary0.65_c60"
            result = sort_scores(reaction_ids, scores, known_reaction_ids, top_k)
            result = annotate_candidate_uncertainty(
                result,
                reaction_ids,
                routed_members,
                known_reaction_ids,
                top_k,
                consensus_scores=uncertainty_consensus,
            )
            result = add_common_columns(
                result,
                query_id=query_id,
                direction="enzyme_to_reaction",
                score_source=score_source,
                ranking_objective=ranking_objective,
                model_directory=model_dir,
                secondary_model_directory=secondary_output_dir,
                nearest_id=nearest_id,
                nearest_similarity=nearest_similarity,
                external_candidates=external_reaction_ids,
            )
            result["known_associations_masked"] = len(known_reaction_ids)
            result = apply_empirical_reliability(
                result,
                "enzyme_to_reaction",
                ranking_objective,
                calibrators,
                applicable=not bool(known_reaction_ids),
                not_applicable_reason="not_applicable_known_associations_masked",
            )
            accepted = policy_accepts(
                str(result.iloc[0]["empirical_reliability_status"]),
                str(result.iloc[0]["empirical_reliability_tier"]),
                reliability_policy,
            )
            result["accepted_by_policy"] = accepted
            all_rankings.append(result)
            summaries.append(query_summary(result, accepted))

    rankings = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    rankings.to_csv(output_dir / "enzyme_to_reaction_rankings.csv", index=False)
    summary.to_csv(output_dir / "enzyme_to_reaction_queries.csv", index=False)
    return summaries


def rank_registered_reactions(
    *,
    objectives: tuple[int, ...],
    max_queries: int | None,
    output_dir: Path,
    current_protein_dir: Path,
    registered_protein_dir: Path,
    registered_reactions_csv: Path,
    positives: Path,
    calibrators: Path,
    short_model_dir: Path,
    top10_20_model_dir: Path,
    reliability_policy: str,
    known_enzymes_by_reaction: dict[str, set[str]],
    mask_known_associations: bool,
    device: torch.device,
) -> list[dict[str, object]]:
    current_proteins, current_ids = load_protein_library(current_protein_dir)
    registered_proteins, registered_ids = load_protein_library(registered_protein_dir)
    candidate_proteins = np.concatenate([current_proteins, registered_proteins], axis=0)
    candidate_ids = current_ids + registered_ids
    external_protein_ids = set(registered_ids)
    registered_reactions = load_external_reaction_rows(registered_reactions_csv)
    if max_queries is not None:
        registered_reactions = registered_reactions.head(max_queries).copy()
    neighbor_index = prepare_reaction_neighbor_index(positives)

    all_rankings: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    model_groups = {
        "top3": (tuple(value for value in objectives if value == 3), short_model_dir),
        "top10_20": (
            tuple(value for value in objectives if value in {10, 20}),
            top10_20_model_dir,
        ),
    }
    for model_label, (grouped_objectives, selected_model_dir) in model_groups.items():
        if not grouped_objectives:
            continue
        schema = load_feature_schema(selected_model_dir)
        models = load_models(selected_model_dir / "models", "production", device)
        library_features, library_ids = load_reaction_library(selected_model_dir, schema)
        library_to_row = {value: index for index, value in enumerate(library_ids)}
        requires_auxiliary = models_require_auxiliary_reaction_features(models)
        library_auxiliary = (
            load_auxiliary_reaction_library(selected_model_dir, library_ids)
            if requires_auxiliary
            else None
        )
        base_rows: list[np.ndarray] = []
        missing_smiles: list[str] = []
        missing_base_rows: list[np.ndarray] = []
        missing_positions: list[int] = []
        auxiliary_rows: list[np.ndarray | None] = []
        for position, row in enumerate(registered_reactions.itertuples(index=False)):
            reaction_id = str(row.reaction_id)
            library_row = library_to_row.get(reaction_id)
            if library_row is not None:
                base_rows.append(library_features[library_row])
                auxiliary_rows.append(
                    library_auxiliary[library_row]
                    if library_auxiliary is not None
                    else None
                )
            else:
                base = encode_reaction(str(row.reaction_smiles), schema)
                base_rows.append(base)
                auxiliary_rows.append(None)
                if requires_auxiliary:
                    missing_smiles.append(str(row.reaction_smiles))
                    missing_base_rows.append(base)
                    missing_positions.append(position)
        query_features = np.stack(base_rows).astype(np.float32)
        query_auxiliary: np.ndarray | None = None
        if requires_auxiliary:
            if missing_positions:
                encoded_missing = encode_exact_horizyn_reactions(
                    missing_smiles,
                    selected_model_dir,
                    device,
                    np.stack(missing_base_rows),
                )
                for local_index, position in enumerate(missing_positions):
                    auxiliary_rows[position] = encoded_missing[local_index]
            if any(value is None for value in auxiliary_rows):
                raise ValueError("Missing auxiliary reaction rows after exact encoding")
            query_auxiliary = np.stack(auxiliary_rows).astype(np.float32)  # type: ignore[arg-type]
        direct_members = ensemble_similarity_members(
            models,
            candidate_proteins,
            query_features,
            device,
            query_auxiliary,
        )
        for query_index, row in enumerate(registered_reactions.itertuples(index=False)):
            known_enzyme_ids = (
                set(known_enzymes_by_reaction.get(str(row.reaction_id), set()))
                if mask_known_associations
                else set()
            )
            nearest_id, nearest_similarity = nearest_registered_reaction(
                str(row.reaction_smiles), neighbor_index
            )
            for top_k in grouped_objectives:
                ranking_objective = objective_name(top_k)
                routed_members = direct_members[:, query_index, :]
                scores = routed_members.mean(axis=0)
                result = sort_scores(candidate_ids, scores, known_enzyme_ids, top_k)
                result = annotate_candidate_uncertainty(
                    result, candidate_ids, routed_members, known_enzyme_ids, top_k
                )
                result = add_common_columns(
                    result,
                    query_id=str(row.reaction_id),
                    direction="reaction_to_enzyme",
                    score_source="direct",
                    ranking_objective=ranking_objective,
                    model_directory=selected_model_dir,
                    nearest_id=nearest_id,
                    nearest_similarity=nearest_similarity,
                    external_candidates=external_protein_ids,
                )
                result["known_associations_masked"] = len(known_enzyme_ids)
                result = apply_empirical_reliability(
                    result,
                    "reaction_to_enzyme",
                    ranking_objective,
                    calibrators,
                    applicable=not bool(known_enzyme_ids),
                    not_applicable_reason="not_applicable_known_associations_masked",
                )
                accepted = policy_accepts(
                    str(result.iloc[0]["empirical_reliability_status"]),
                    str(result.iloc[0]["empirical_reliability_tier"]),
                    reliability_policy,
                )
                result["accepted_by_policy"] = accepted
                all_rankings.append(result)
                summaries.append(query_summary(result, accepted))

    rankings = pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    rankings.to_csv(output_dir / "reaction_to_enzyme_rankings.csv", index=False)
    summary.to_csv(output_dir / "reaction_to_enzyme_queries.csv", index=False)
    return summaries


def write_discovery_audits(
    output_dir: Path,
    known_reactions_by_enzyme: dict[str, set[str]],
    known_enzymes_by_reaction: dict[str, set[str]],
    mask_known_associations: bool,
) -> dict[str, object]:
    ranking_specs = [
        (
            "enzyme_to_reaction",
            output_dir / "enzyme_to_reaction_rankings.csv",
            known_reactions_by_enzyme,
        ),
        (
            "reaction_to_enzyme",
            output_dir / "reaction_to_enzyme_rankings.csv",
            known_enzymes_by_reaction,
        ),
    ]
    leak_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    total_ranking_rows = 0
    for direction, path, known_map in ranking_specs:
        if not path.exists():
            continue
        rankings = pd.read_csv(path, dtype=str).fillna("")
        if rankings.empty:
            continue
        rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
        rankings["is_external_candidate"] = (
            rankings["is_external_candidate"].astype(str).str.lower().eq("true")
        )
        total_ranking_rows += len(rankings)
        for row in rankings[["query_id", "candidate_id", "ranking_objective", "rank"]].itertuples(index=False):
            if str(row.candidate_id) in known_map.get(str(row.query_id), set()):
                leak_rows.append(
                    {
                        "direction": direction,
                        "ranking_objective": str(row.ranking_objective),
                        "query_id": str(row.query_id),
                        "candidate_id": str(row.candidate_id),
                        "rank": int(row.rank),
                    }
                )
        for objective, group in rankings.groupby("ranking_objective", sort=True):
            top1 = group[group["rank"].eq(1)].copy()
            n_queries = int(top1["query_id"].nunique())
            top1_counts = top1["candidate_id"].value_counts()
            if len(top1_counts):
                probabilities = top1_counts.to_numpy(dtype=float) / float(top1_counts.sum())
                entropy = float(-(probabilities * np.log(probabilities)).sum())
                effective_top1 = float(np.exp(entropy))
                largest_top1_share = float(top1_counts.iloc[0] / top1_counts.sum())
            else:
                effective_top1 = 0.0
                largest_top1_share = 0.0
            all_counts = group["candidate_id"].value_counts()
            top10_share = (
                float(all_counts.head(10).sum() / len(group)) if len(group) else 0.0
            )
            concentration_rows.append(
                {
                    "direction": direction,
                    "objective": str(objective),
                    "n_queries": n_queries,
                    "unique_top1": int(top1["candidate_id"].nunique()),
                    "top1_top_candidate_share": largest_top1_share,
                    "top10_candidates_share": top10_share,
                    "effective_top1_candidates": effective_top1,
                    "top1_external_share": (
                        float(top1["is_external_candidate"].mean()) if len(top1) else 0.0
                    ),
                }
            )
    leak_frame = pd.DataFrame(
        leak_rows,
        columns=[
            "direction",
            "ranking_objective",
            "query_id",
            "candidate_id",
            "rank",
        ],
    )
    leak_frame.to_csv(output_dir / "known_association_leaks.csv", index=False)
    concentration = pd.DataFrame(concentration_rows)
    concentration.to_csv(output_dir / "discovery_concentration_summary.csv", index=False)
    audit = {
        "mask_known_associations": bool(mask_known_associations),
        "ranking_rows_checked": int(total_ranking_rows),
        "known_association_leaks": int(len(leak_frame)),
        "leaks_by_direction": (
            leak_frame["direction"].value_counts().to_dict() if len(leak_frame) else {}
        ),
        "outputs": {
            "leaks": str(output_dir / "known_association_leaks.csv"),
            "concentration": str(output_dir / "discovery_concentration_summary.csv"),
        },
    }
    (output_dir / "discovery_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def summarize(summaries: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame(summaries)
    if frame.empty:
        return {"n_queries": 0}
    return {
        "n_query_objectives": len(frame),
        "n_unique_queries": int(frame["query_id"].nunique()),
        "accepted_by_policy": int(frame["accepted_by_policy"].sum()),
        "reliability_tiers": frame["empirical_reliability_tier"].value_counts(dropna=False).to_dict(),
        "reliability_status": frame["empirical_reliability_status"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized ranking for the persistent TPS open-world registry.")
    parser.add_argument("--direction", choices=["both", "enzyme_to_reaction", "reaction_to_enzyme"], default="both")
    parser.add_argument("--objectives", default=",".join(str(value) for value in DEFAULT_OBJECTIVES))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR)
    parser.add_argument("--registered-reactions-csv", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument(
        "--include-known-associations",
        action="store_true",
        help="Retain MARTS-labelled pairs for regression/audit. Discovery mode masks them by default.",
    )
    parser.add_argument("--calibrators", type=Path, default=DEFAULT_UNCERTAINTY_CALIBRATORS)
    parser.add_argument(
        "--r2e-shared-model-dir",
        type=Path,
        default=DEFAULT_R2E_DUAL_TOWER_DIR,
        help="Deprecated for registered external R2E; retained for CLI compatibility.",
    )
    parser.add_argument("--r2e-short-model-dir", type=Path, default=DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR)
    parser.add_argument(
        "--r2e-top10-20-model-dir",
        type=Path,
        default=DEFAULT_R2E_TOP10_20_DUAL_TOWER_DIR,
    )
    parser.add_argument("--e2r-model-dir", type=Path, default=DEFAULT_E2R_DUAL_TOWER_DIR)
    parser.add_argument(
        "--e2r-secondary-model-dir",
        type=Path,
        default=DEFAULT_E2R_HARDNEG_DUAL_TOWER_DIR,
    )
    parser.add_argument("--topk-neighbor-proteins", type=int, default=5)
    parser.add_argument(
        "--reliability-policy",
        choices=["annotate", "require_calibrated", "require_intermediate", "require_higher"],
        default="annotate",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    objectives = tuple(sorted({int(value) for value in args.objectives.split(",") if value}))
    if any(value not in DEFAULT_OBJECTIVES for value in objectives):
        raise ValueError("Batch objectives must be selected from 3,10,20")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("max-queries must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    registered_reaction_ids = set(
        load_external_reaction_rows(args.registered_reactions_csv.resolve())["reaction_id"].astype(str)
    )
    known_reactions_by_enzyme, known_enzymes_by_reaction = build_known_association_maps(
        args.marts.resolve(), args.positives.resolve(), registered_reaction_ids
    )
    mask_known_associations = not args.include_known_associations
    summary: dict[str, object] = {
        "direction": args.direction,
        "objectives": objectives,
        "reliability_policy": args.reliability_policy,
        "mask_known_associations": mask_known_associations,
    }
    if args.direction in {"both", "enzyme_to_reaction"}:
        enzyme_summaries = rank_registered_enzymes(
            objectives=objectives,
            max_queries=args.max_queries,
            output_dir=output_dir,
            current_protein_dir=args.protein_dir.resolve(),
            registered_protein_dir=args.registered_protein_dir.resolve(),
            registered_reactions_csv=args.registered_reactions_csv.resolve(),
            positives=args.positives.resolve(),
            calibrators=args.calibrators.resolve(),
            model_dir=args.e2r_model_dir.resolve(),
            secondary_model_dir=args.e2r_secondary_model_dir.resolve(),
            reliability_policy=args.reliability_policy,
            topk_neighbor_proteins=args.topk_neighbor_proteins,
            known_reactions_by_enzyme=known_reactions_by_enzyme,
            mask_known_associations=mask_known_associations,
            device=device,
        )
        summary["enzyme_to_reaction"] = summarize(enzyme_summaries)
    if args.direction in {"both", "reaction_to_enzyme"}:
        reaction_summaries = rank_registered_reactions(
            objectives=objectives,
            max_queries=args.max_queries,
            output_dir=output_dir,
            current_protein_dir=args.protein_dir.resolve(),
            registered_protein_dir=args.registered_protein_dir.resolve(),
            registered_reactions_csv=args.registered_reactions_csv.resolve(),
            positives=args.positives.resolve(),
            calibrators=args.calibrators.resolve(),
            short_model_dir=args.r2e_short_model_dir.resolve(),
            top10_20_model_dir=args.r2e_top10_20_model_dir.resolve(),
            reliability_policy=args.reliability_policy,
            known_enzymes_by_reaction=known_enzymes_by_reaction,
            mask_known_associations=mask_known_associations,
            device=device,
        )
        summary["reaction_to_enzyme"] = summarize(reaction_summaries)
    audit = write_discovery_audits(
        output_dir,
        known_reactions_by_enzyme,
        known_enzymes_by_reaction,
        mask_known_associations,
    )
    summary["discovery_audit"] = audit
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
