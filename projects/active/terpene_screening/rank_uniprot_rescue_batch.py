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

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (  # noqa: E402
    pfam_architecture,
    reaction_metadata,
)
from projects.active.terpene_screening.evaluate_uniprot_expanded_r2e import (  # noqa: E402
    DEFAULT_UNIPROT_METADATA,
    DEFAULT_UNIPROT_PROTEINS,
    build_reaction_queries,
    load_candidate_universe,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_DIR,
    DEFAULT_R2E_DUAL_TOWER_DIR,
    DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR,
    DEFAULT_R2E_TOP10_20_DUAL_TOWER_DIR,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    annotate_candidate_uncertainty,
    encode_exact_horizyn_reactions,
    ensemble_similarity_members,
    load_auxiliary_reaction_library,
    load_external_reaction_rows,
    load_feature_schema,
    load_models,
    load_reaction_library,
    models_require_auxiliary_reaction_features,
    sort_scores,
)
from projects.active.terpene_screening.rank_registry_batch import (  # noqa: E402
    DEFAULT_MARTS,
    build_known_association_maps,
)
from projects.active.terpene_screening.rank_uniprot_rescue import (  # noqa: E402
    DEFAULT_CONTRACTS,
    DEFAULT_HUB_FREQUENCY,
    DEFAULT_RESCUE_SLOTS,
    DEFAULT_STRESS_DIR,
    load_hub_fraction,
    load_quota_evidence,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_controlled_rescue_batch"


def reaction_type_table(marts_path: Path, registry_path: Path) -> pd.DataFrame:
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    metadata = reaction_metadata(marts_path)
    columns = [
        "reaction_signature",
        "terpene_type",
        "tps_class",
        "substrate_name",
        "product_name",
    ]
    return registry.merge(metadata[columns], on="reaction_signature", how="left").fillna("")


def pool_result(
    ids: list[str],
    scores: np.ndarray,
    member_scores: np.ndarray,
    masked: set[str],
    top_k: int,
    pool_name: str,
) -> pd.DataFrame:
    if top_k == 0:
        return pd.DataFrame()
    result = sort_scores(ids, scores, masked, top_k)
    result = annotate_candidate_uncertainty(
        result, ids, member_scores, masked, top_k
    )
    result[f"{pool_name}_pool_rank"] = pd.to_numeric(
        result["rank"], errors="raise"
    ).astype(int)
    return result.drop(columns=["rank"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch controlled UniProt rescue ranking for all registered reactions."
    )
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument(
        "--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR
    )
    parser.add_argument("--uniprot-protein-dir", type=Path, default=DEFAULT_UNIPROT_PROTEINS)
    parser.add_argument("--uniprot-metadata", type=Path, default=DEFAULT_UNIPROT_METADATA)
    parser.add_argument("--registered-reactions", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--shared-model-dir", type=Path, default=DEFAULT_R2E_DUAL_TOWER_DIR)
    parser.add_argument(
        "--short-model-dir", type=Path, default=DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR
    )
    parser.add_argument(
        "--top10-20-model-dir",
        type=Path,
        default=DEFAULT_R2E_TOP10_20_DUAL_TOWER_DIR,
    )
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS_DIR)
    parser.add_argument("--hub-frequency", type=Path, default=DEFAULT_HUB_FREQUENCY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    candidate_metadata["pfam_architecture"] = candidate_metadata[
        "pfam_combination"
    ].map(pfam_architecture)
    source_array = candidate_metadata["candidate_source"].astype(str).to_numpy()
    canonical_indices = np.flatnonzero(source_array != "uniprot_primary")
    uniprot_indices = np.flatnonzero(source_array == "uniprot_primary")
    canonical_ids = [protein_ids[int(index)] for index in canonical_indices]
    uniprot_ids = [protein_ids[int(index)] for index in uniprot_indices]
    canonical_id_set = set(canonical_ids)

    registry = load_external_reaction_rows(args.registered_reactions.resolve())
    registered_ids = set(registry["reaction_id"].astype(str))
    _, known_by_reaction = build_known_association_maps(
        args.marts.resolve(), args.positives.resolve(), registered_ids
    )
    reaction_info = reaction_type_table(
        args.marts.resolve(), args.registered_reactions.resolve()
    ).set_index("reaction_id")
    contracts = pd.read_csv(args.contracts, dtype=str).fillna("")
    contract_map = contracts.set_index("reaction_id")
    hub_fraction = load_hub_fraction(args.hub_frequency.resolve())

    ranking_frames: list[pd.DataFrame] = []
    query_rows: list[dict[str, object]] = []
    model_specs = [
        (args.short_model_dir.resolve(), (3,)),
        (args.top10_20_model_dir.resolve(), (10, 20)),
    ]
    for model_dir, objectives in model_specs:
        reaction_features, reaction_table, _ = build_reaction_queries(
            model_dir, args.registered_reactions.resolve()
        )
        models = load_models(model_dir / "models", "production", device)
        query_auxiliary: np.ndarray | None = None
        if models_require_auxiliary_reaction_features(models):
            schema = load_feature_schema(model_dir)
            library_features, library_ids = load_reaction_library(model_dir, schema)
            library_auxiliary = load_auxiliary_reaction_library(model_dir, library_ids)
            library_to_row = {value: index for index, value in enumerate(library_ids)}
            auxiliary_rows: list[np.ndarray | None] = []
            missing_positions: list[int] = []
            missing_smiles: list[str] = []
            missing_base: list[np.ndarray] = []
            for position, row in enumerate(reaction_table.itertuples(index=False)):
                library_row = library_to_row.get(str(row.reaction_id))
                if library_row is None:
                    auxiliary_rows.append(None)
                    missing_positions.append(position)
                    missing_smiles.append(str(row.reaction_smiles))
                    missing_base.append(reaction_features[position])
                else:
                    auxiliary_rows.append(library_auxiliary[library_row])
            if missing_positions:
                encoded = encode_exact_horizyn_reactions(
                    missing_smiles,
                    model_dir,
                    device,
                    np.stack(missing_base),
                )
                for local_index, position in enumerate(missing_positions):
                    auxiliary_rows[position] = encoded[local_index]
            if any(value is None for value in auxiliary_rows):
                raise ValueError("Missing exact auxiliary rows in UniProt rescue batch")
            query_auxiliary = np.stack(auxiliary_rows).astype(np.float32)  # type: ignore[arg-type]
        all_member_scores = ensemble_similarity_members(
            models,
            protein_features,
            reaction_features,
            device,
            query_auxiliary,
        )
        for query_index, reaction_row in enumerate(reaction_table.itertuples(index=False)):
            query_id = str(reaction_row.reaction_id)
            info = reaction_info.loc[query_id]
            terpene_type = str(info["terpene_type"])
            if query_id not in contract_map.index:
                raise KeyError(f"Architecture contract missing for {query_id}")
            contract = contract_map.loc[query_id]
            contract_status = str(contract["contract_status"])
            allowed_architectures = {
                value
                for value in str(contract["allowed_candidate_architectures"]).split(";")
                if value
            }
            known = set(known_by_reaction.get(query_id, set())) & canonical_id_set
            query_member_scores = all_member_scores[:, query_index, :]
            mean_scores = query_member_scores.mean(axis=0)
            uniprot_metadata = candidate_metadata.iloc[uniprot_indices].copy().reset_index(drop=True)
            compatibility = np.where(
                uniprot_metadata["pfam_architecture"].isin(allowed_architectures),
                "compatible",
                "architecture_mismatch",
            )
            eligible_mask = compatibility == "compatible"
            eligible_local = np.flatnonzero(eligible_mask)
            eligible_uniprot_ids = [uniprot_ids[int(index)] for index in eligible_local]
            eligible_global = uniprot_indices[eligible_local]

            for top_k in objectives:
                requested_rescue_slots = int(DEFAULT_RESCUE_SLOTS[top_k])
                rescue_slots = requested_rescue_slots if allowed_architectures else 0
                if rescue_slots and len(eligible_uniprot_ids) < rescue_slots:
                    raise ValueError(
                        f"Only {len(eligible_uniprot_ids)} eligible candidates for {query_id} Top-{top_k}"
                    )
                canonical_slots = top_k - rescue_slots
                canonical = pool_result(
                    canonical_ids,
                    mean_scores[canonical_indices],
                    query_member_scores[:, canonical_indices],
                    known,
                    canonical_slots,
                    "canonical",
                )
                canonical["selection_source"] = "canonical_primary"
                rescue = pool_result(
                    eligible_uniprot_ids,
                    mean_scores[eligible_global],
                    query_member_scores[:, eligible_global],
                    set(),
                    rescue_slots,
                    "uniprot",
                )
                if not rescue.empty:
                    rescue["selection_source"] = "uniprot_rescue"
                    result = pd.concat([canonical, rescue], ignore_index=True, sort=False)
                else:
                    result = canonical.copy()
                result.insert(0, "query_id", query_id)
                result.insert(1, "direction", "reaction_to_enzyme")
                result.insert(2, "ranking_objective", f"top{top_k}")
                result.insert(3, "rank", np.arange(1, len(result) + 1))
                result.insert(4, "canonical_slots", canonical_slots)
                result.insert(5, "uniprot_rescue_slots", rescue_slots)
                result.insert(6, "known_associations_masked", len(known))
                result.insert(7, "terpene_type", terpene_type)
                result.insert(8, "tps_class", str(info["tps_class"]))
                result.insert(9, "substrate_name", str(info["substrate_name"]))
                result.insert(10, "product_name", str(info["product_name"]))
                result.insert(11, "architecture_contract_status", contract_status)
                result.insert(12, "allowed_candidate_architectures", ";".join(sorted(allowed_architectures)))
                result.insert(13, "requested_uniprot_rescue_slots", requested_rescue_slots)
                result.insert(14, "model_directory", str(model_dir))
                result.insert(
                    15,
                    "empirical_reliability_status",
                    "not_applicable_controlled_candidate_expansion",
                )
                result = result.merge(candidate_metadata, on="candidate_id", how="left")
                result["pfam_architecture"] = result["pfam_combination"].map(
                    pfam_architecture
                )
                result["family_compatibility"] = [
                    (
                        "compatible"
                        if architecture in allowed_architectures
                        else "architecture_mismatch"
                    )
                    if candidate_source == "uniprot_primary"
                    else "not_applicable_non_uniprot"
                    for architecture, candidate_source in zip(
                        result["pfam_architecture"].astype(str),
                        result["candidate_source"].astype(str),
                    )
                ]
                result["uniprot_registry_query_fraction"] = (
                    result["candidate_id"].map(hub_fraction).fillna(0.0)
                )
                quota = load_quota_evidence(
                    args.stress_dir.resolve(), top_k, rescue_slots
                )
                for column, value in quota.items():
                    result[column] = value
                ranking_frames.append(result)
                rescue_rows = result[result["selection_source"].eq("uniprot_rescue")]
                query_rows.append(
                    {
                        "query_id": query_id,
                        "ranking_objective": f"top{top_k}",
                        "canonical_slots": canonical_slots,
                        "requested_uniprot_rescue_slots": requested_rescue_slots,
                        "uniprot_rescue_slots": rescue_slots,
                        "architecture_contract_status": contract_status,
                        "allowed_candidate_architectures": ";".join(sorted(allowed_architectures)),
                        "known_associations_masked": len(known),
                        "eligible_uniprot_candidates": len(eligible_uniprot_ids),
                        "selected_uniprot_candidates": len(rescue_rows),
                        "selected_uniprot_evidence_tiers": ";".join(
                            sorted(set(rescue_rows["evidence_quality_tier"].astype(str)) - {""})
                        ),
                        "maximum_selected_hub_fraction": (
                            float(rescue_rows["uniprot_registry_query_fraction"].max())
                            if len(rescue_rows)
                            else 0.0
                        ),
                        **quota,
                    }
                )

    rankings = pd.concat(ranking_frames, ignore_index=True)
    queries = pd.DataFrame(query_rows)
    rankings.to_csv(output_dir / "controlled_rankings.csv", index=False)
    queries.to_csv(output_dir / "query_summary.csv", index=False)
    rescue = rankings[rankings["selection_source"].eq("uniprot_rescue")].copy()
    frequency = (
        rescue.groupby(
            ["candidate_id", "evidence_quality_tier", "domain_family"],
            dropna=False,
        )
        .agg(
            query_appearance_count=("query_id", "nunique"),
            total_appearance_count=("query_id", "size"),
            best_final_rank=("rank", "min"),
            mean_model_score=("ensemble_score_mean", "mean"),
            maximum_registry_hub_fraction=("uniprot_registry_query_fraction", "max"),
        )
        .reset_index()
        .sort_values(
            ["query_appearance_count", "best_final_rank"], ascending=[False, True]
        )
    )
    frequency.to_csv(output_dir / "selected_uniprot_frequency.csv", index=False)
    summary = {
        "registered_reactions": len(registry),
        "candidate_universe": candidate_metadata["candidate_source"].value_counts().to_dict(),
        "validated_default_quotas": {f"top{key}": value for key, value in DEFAULT_RESCUE_SLOTS.items()},
        "ranking_rows": len(rankings),
        "uniprot_rescue_rows": len(rescue),
        "unique_uniprot_rescue_candidates": rescue["candidate_id"].nunique(),
        "queries_with_complete_quota": int(
            queries["selected_uniprot_candidates"].eq(queries["uniprot_rescue_slots"]).sum()
        ),
        "query_objectives": len(queries),
        "contract_supported_reactions": int(
            contracts["rescue_supported"].astype(str).str.lower().eq("true").sum()
        ),
        "contract_unsupported_reactions": int(
            contracts["rescue_supported"].astype(str).str.lower().eq("false").sum()
        ),
        "uniprot_evidence_tiers": rescue["evidence_quality_tier"].value_counts().to_dict(),
        "uniprot_domain_families": rescue["domain_family"].value_counts().to_dict(),
        "maximum_selected_candidate_query_appearances": (
            int(frequency["query_appearance_count"].max()) if len(frequency) else 0
        ),
        "outputs": {
            "rankings": str(output_dir / "controlled_rankings.csv"),
            "query_summary": str(output_dir / "query_summary.csv"),
            "frequency": str(output_dir / "selected_uniprot_frequency.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
