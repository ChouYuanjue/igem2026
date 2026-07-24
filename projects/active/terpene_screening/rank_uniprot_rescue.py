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
    encode_reaction,
    ensemble_similarity_members,
    load_external_reaction_rows,
    load_feature_schema,
    load_models,
    models_require_auxiliary_reaction_features,
    sort_scores,
)
from projects.active.terpene_screening.rank_registry_batch import (  # noqa: E402
    DEFAULT_MARTS,
    build_known_association_maps,
)

DEFAULT_STRESS_DIR = ROOT / "results/terpene_uniprot_expanded_double_cold"
DEFAULT_HUB_FREQUENCY = (
    ROOT
    / "results/terpene_uniprot_expansion_quality/uniprot_rescue_candidate_frequency.csv"
)
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_rescue_ranking.csv"
DEFAULT_CONTRACTS = (
    ROOT
    / "data/terpene_uniprot_expansion/reaction_architecture_contracts/reaction_architecture_contracts.csv"
)
DEFAULT_RESCUE_SLOTS = {3: 0, 10: 1, 20: 2}
EVIDENCE_ORDER = {
    "A_reviewed": 0,
    "B_experimental_or_transcript_named": 1,
    "C_homology_named": 2,
    "D_named_predicted": 3,
}


def resolve_rescue_slots(top_k: int, explicit: int | None) -> int:
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    if explicit is None:
        if top_k not in DEFAULT_RESCUE_SLOTS:
            raise ValueError(
                "No validated default rescue quota for this top-k. Pass --rescue-slots explicitly."
            )
        return DEFAULT_RESCUE_SLOTS[top_k]
    if explicit < 0 or explicit > top_k:
        raise ValueError("rescue-slots must be between zero and top-k")
    return explicit


def choose_model_dir(
    top_k: int,
    shared_model_dir: Path,
    short_model_dir: Path,
    top10_20_model_dir: Path,
) -> Path:
    if top_k == 3:
        return short_model_dir
    if top_k in {10, 20}:
        return top10_20_model_dir
    return shared_model_dir


def resolve_reaction(
    reaction_id: str | None,
    reaction_smiles: str | None,
    registry_path: Path,
) -> tuple[str, str, bool]:
    if reaction_smiles:
        query_id = str(reaction_id or "external_reaction").strip()
        return query_id, str(reaction_smiles).strip(), False
    if not reaction_id:
        raise ValueError("Provide --reaction-id or --reaction-smiles")
    registry = load_external_reaction_rows(registry_path)
    matched = registry[registry["reaction_id"].astype(str).eq(str(reaction_id))]
    if len(matched) != 1:
        raise KeyError(f"Reaction ID not found in registry: {reaction_id}")
    return str(reaction_id), str(matched.iloc[0]["reaction_smiles"]), True


def load_reaction_type_map(marts_path: Path, registry_path: Path) -> dict[str, str]:
    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    metadata = reaction_metadata(marts_path)
    merged = registry[["reaction_id", "reaction_signature"]].merge(
        metadata[["reaction_signature", "terpene_type"]],
        on="reaction_signature",
        how="left",
    )
    return dict(zip(merged["reaction_id"].astype(str), merged["terpene_type"].fillna("")))


def load_hub_fraction(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "query_fraction" not in frame.columns:
        return {}
    fraction = pd.to_numeric(frame["query_fraction"], errors="coerce").fillna(0.0)
    return dict(zip(frame["candidate_id"].astype(str), fraction.astype(float)))


def load_quota_evidence(stress_dir: Path, top_k: int, rescue_slots: int) -> dict[str, object]:
    path = stress_dir / "rescue_slot_retention.csv"
    if not path.exists():
        return {
            "strict_double_cold_status": "quota_evidence_missing",
            "strict_double_cold_hit_retention_fraction": np.nan,
            "strict_double_cold_quota_hit_probability": np.nan,
        }
    frame = pd.read_csv(path)
    selected = frame[
        frame["budget"].eq(top_k) & frame["rescue_slots"].eq(rescue_slots)
    ]
    if len(selected) != 1:
        return {
            "strict_double_cold_status": "quota_not_evaluated",
            "strict_double_cold_hit_retention_fraction": np.nan,
            "strict_double_cold_quota_hit_probability": np.nan,
        }
    row = selected.iloc[0]
    return {
        "strict_double_cold_status": "validated_unlabelled_decoy_stress_test",
        "strict_double_cold_hit_retention_fraction": float(row["hit_retention_fraction"]),
        "strict_double_cold_quota_hit_probability": float(row["hit_probability"]),
        "strict_double_cold_baseline_hits": int(row["baseline_canonical_hits"]),
        "strict_double_cold_quota_hits": int(row["quota_hits"]),
    }


def annotate_pool_ranks(result: pd.DataFrame, pool_name: str) -> pd.DataFrame:
    result = result.copy()
    result[f"{pool_name}_pool_rank"] = pd.to_numeric(result["rank"], errors="raise").astype(int)
    return result.drop(columns=["rank"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled UniProt rescue ranking with a canonical prefix and validated tail quota."
    )
    parser.add_argument("--reaction-id")
    parser.add_argument("--reaction-smiles")
    parser.add_argument("--terpene-type", default="")
    parser.add_argument(
        "--allowed-architectures",
        default="",
        help="Comma-separated explicit architectures for a raw external reaction; omitted means canonical-only.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rescue-slots", type=int)
    parser.add_argument(
        "--family-policy",
        choices=["compatible_only", "annotate"],
        default="compatible_only",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    top_k = int(args.top_k)
    requested_rescue_slots = resolve_rescue_slots(top_k, args.rescue_slots)
    query_id, reaction_smiles, is_registered = resolve_reaction(
        args.reaction_id,
        args.reaction_smiles,
        args.registered_reactions.resolve(),
    )
    terpene_type = str(args.terpene_type).strip()
    if not terpene_type and is_registered:
        terpene_type = load_reaction_type_map(
            args.marts.resolve(), args.registered_reactions.resolve()
        ).get(query_id, "")
    contracts = pd.read_csv(args.contracts, dtype=str).fillna("")
    contract_status = "external_architecture_not_supplied"
    allowed_architectures = {
        value.strip()
        for value in str(args.allowed_architectures).split(",")
        if value.strip()
    }
    if is_registered:
        matched_contract = contracts[contracts["reaction_id"].astype(str).eq(query_id)]
        if len(matched_contract) != 1:
            raise KeyError(f"Architecture contract missing for registered reaction: {query_id}")
        contract_row = matched_contract.iloc[0]
        contract_status = str(contract_row["contract_status"])
        allowed_architectures = {
            value
            for value in str(contract_row["allowed_candidate_architectures"]).split(";")
            if value
        }
    rescue_slots = requested_rescue_slots if allowed_architectures else 0
    canonical_slots = top_k - rescue_slots

    model_dir = choose_model_dir(
        top_k,
        args.shared_model_dir.resolve(),
        args.short_model_dir.resolve(),
        args.top10_20_model_dir.resolve(),
    )
    schema = load_feature_schema(model_dir)
    reaction_feature = encode_reaction(reaction_smiles, schema)[None, :]
    protein_features, protein_ids, metadata = load_candidate_universe(
        args.current_protein_dir.resolve(),
        args.registered_protein_dir.resolve(),
        args.uniprot_protein_dir.resolve(),
        args.uniprot_metadata.resolve(),
    )
    device = torch.device(args.device)
    models = load_models(model_dir / "models", "production", device)
    auxiliary_feature = (
        encode_exact_horizyn_reactions(
            [reaction_smiles],
            model_dir,
            device,
            reaction_feature,
        )
        if models_require_auxiliary_reaction_features(models)
        else None
    )
    member_scores = ensemble_similarity_members(
        models,
        protein_features,
        reaction_feature,
        device,
        auxiliary_feature,
    )[:, 0, :]
    mean_scores = member_scores.mean(axis=0)

    registered_ids = set(
        load_external_reaction_rows(args.registered_reactions.resolve())["reaction_id"].astype(str)
    )
    _, known_by_reaction = build_known_association_maps(
        args.marts.resolve(), args.positives.resolve(), registered_ids
    )
    known = set(known_by_reaction.get(query_id, set())) if is_registered else set()

    source = metadata["candidate_source"].astype(str).to_numpy()
    canonical_indices = np.flatnonzero(source != "uniprot_primary")
    uniprot_indices = np.flatnonzero(source == "uniprot_primary")
    canonical_ids = [protein_ids[int(index)] for index in canonical_indices]
    uniprot_ids = [protein_ids[int(index)] for index in uniprot_indices]
    canonical_known = known & set(canonical_ids)

    canonical = sort_scores(
        canonical_ids,
        mean_scores[canonical_indices],
        canonical_known,
        canonical_slots,
    )
    canonical = annotate_candidate_uncertainty(
        canonical,
        canonical_ids,
        member_scores[:, canonical_indices],
        canonical_known,
        canonical_slots,
    )
    canonical = annotate_pool_ranks(canonical, "canonical")
    canonical["selection_source"] = "canonical_primary"

    uniprot_metadata = metadata.iloc[uniprot_indices].copy().reset_index(drop=True)
    uniprot_metadata["pfam_architecture"] = uniprot_metadata[
        "pfam_combination"
    ].map(pfam_architecture)
    uniprot_metadata["family_compatibility"] = np.where(
        uniprot_metadata["pfam_architecture"].isin(allowed_architectures),
        "compatible",
        "architecture_mismatch",
    )
    eligible = np.ones(len(uniprot_metadata), dtype=bool)
    if args.family_policy == "compatible_only":
        eligible = uniprot_metadata["family_compatibility"].eq("compatible").to_numpy()
    eligible_indices = np.flatnonzero(eligible)
    if rescue_slots and len(eligible_indices) < rescue_slots:
        raise ValueError(
            f"Only {len(eligible_indices)} eligible UniProt candidates for {rescue_slots} rescue slots"
        )

    if rescue_slots:
        eligible_ids = [uniprot_ids[int(index)] for index in eligible_indices]
        rescue = sort_scores(
            eligible_ids,
            mean_scores[uniprot_indices[eligible_indices]],
            set(),
            rescue_slots,
        )
        rescue = annotate_candidate_uncertainty(
            rescue,
            eligible_ids,
            member_scores[:, uniprot_indices[eligible_indices]],
            set(),
            rescue_slots,
        )
        rescue = annotate_pool_ranks(rescue, "uniprot")
        rescue["selection_source"] = "uniprot_rescue"
    else:
        rescue = pd.DataFrame(columns=canonical.columns)

    result = canonical.copy() if rescue.empty else pd.concat([canonical, rescue], ignore_index=True, sort=False)
    result.insert(0, "query_id", query_id)
    result.insert(1, "direction", "reaction_to_enzyme")
    result.insert(2, "ranking_objective", f"top{top_k}")
    result.insert(3, "rank", np.arange(1, len(result) + 1))
    result.insert(4, "canonical_slots", canonical_slots)
    result.insert(5, "uniprot_rescue_slots", rescue_slots)
    result.insert(6, "reaction_is_registered", is_registered)
    result.insert(7, "known_associations_masked", len(canonical_known))
    result.insert(8, "terpene_type", terpene_type)
    result.insert(9, "family_policy", args.family_policy)
    result.insert(10, "architecture_contract_status", contract_status)
    result.insert(11, "allowed_candidate_architectures", ";".join(sorted(allowed_architectures)))
    result.insert(12, "requested_uniprot_rescue_slots", requested_rescue_slots)
    result.insert(13, "model_directory", str(model_dir.resolve()))
    result.insert(14, "empirical_reliability_status", "not_applicable_controlled_candidate_expansion")
    result = result.merge(metadata, on="candidate_id", how="left")
    result["pfam_architecture"] = result["pfam_combination"].map(pfam_architecture)
    result["family_compatibility"] = [
        (
            "compatible"
            if architecture in allowed_architectures
            else "architecture_mismatch"
        )
        if source_name == "uniprot_primary"
        else "not_applicable_non_uniprot"
        for architecture, source_name in zip(
            result["pfam_architecture"].astype(str), result["candidate_source"].astype(str)
        )
    ]
    hub = load_hub_fraction(args.hub_frequency.resolve())
    result["uniprot_registry_query_fraction"] = result["candidate_id"].map(hub).fillna(0.0)
    result["evidence_priority"] = result["evidence_quality_tier"].map(EVIDENCE_ORDER).fillna(-1).astype(int)
    quota = load_quota_evidence(args.stress_dir.resolve(), top_k, rescue_slots)
    for column, value in quota.items():
        result[column] = value

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "query_id": query_id,
        "reaction_is_registered": is_registered,
        "terpene_type": terpene_type,
        "top_k": top_k,
        "canonical_slots": canonical_slots,
        "requested_uniprot_rescue_slots": requested_rescue_slots,
        "uniprot_rescue_slots": rescue_slots,
        "architecture_contract_status": contract_status,
        "allowed_candidate_architectures": sorted(allowed_architectures),
        "known_associations_masked": len(canonical_known),
        "family_policy": args.family_policy,
        "candidate_sources": result["candidate_source"].value_counts().to_dict(),
        "quota_evidence": quota,
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
