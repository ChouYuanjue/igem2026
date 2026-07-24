from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPANDED = ROOT / "results/terpene_uniprot_expanded_r2e"
DEFAULT_REACTIONS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_expansion_quality"
DEFAULT_CONTRACTS = (
    ROOT
    / "data/terpene_uniprot_expansion/reaction_architecture_contracts/reaction_architecture_contracts.csv"
)

CORE_CLASS_I_TYPES = {"mono", "sesq", "di", "sester"}
EXTENDED_TYPES = {"psy", "sqs", "tetra", "pt"}


def pfam_architecture(pfam_combination: str) -> str:
    domains = {value for value in str(pfam_combination).split(";") if value}
    if {"PF13243", "PF13249"}.issubset(domains):
        return "osc_full"
    if "PF13243" in domains:
        return "classII_cyclase_single_domain"
    if "PF13249" in domains:
        return "osc_domain_fragment"
    if "PF19086" in domains and {"PF01397", "PF03936"}.issubset(domains):
        return "classI_hybrid_full"
    if "PF19086" in domains:
        return "bacterial_classI"
    if {"PF01397", "PF03936"}.issubset(domains):
        return "plant_tps_full"
    if "PF01397" in domains:
        return "plant_tps_single_PF01397"
    if "PF03936" in domains:
        return "plant_tps_single_PF03936"
    return "unsupported_architecture"


def reaction_architecture_compatibility(
    terpene_type: str,
    substrate_name: str,
    product_name: str,
    tps_class: str,
    pfam_combination: str,
    domain_family: str = "",
) -> str:
    terpene_type = str(terpene_type)
    substrate = str(substrate_name).lower()
    product = str(product_name).lower()
    architecture = pfam_architecture(pfam_combination)
    if not str(pfam_combination):
        return "not_applicable_non_uniprot"
    if "presqualene diphosphate" in substrate:
        return "unsupported_expansion_family"
    if "epoxysqualene" in substrate or terpene_type in {"tri", "sesquar"}:
        return "compatible" if architecture == "osc_full" else "architecture_mismatch"
    if terpene_type in EXTENDED_TYPES:
        return "extended_pathway_uncertain"
    if str(tps_class) == "2":
        return (
            "compatible"
            if architecture in {"classII_cyclase_single_domain", "osc_full", "plant_tps_full"}
            else "architecture_mismatch"
        )
    if terpene_type in CORE_CLASS_I_TYPES:
        return (
            "compatible"
            if architecture in {"bacterial_classI", "plant_tps_full", "classI_hybrid_full"}
            else "architecture_mismatch"
        )
    return "unknown_reaction_type"


def family_compatibility(terpene_type: str, domain_family: str) -> str:
    """Legacy coarse compatibility retained for older callers and tests."""
    family = str(domain_family)
    if not family:
        return "not_applicable_non_uniprot"
    if str(terpene_type) in {"tri", "sesquar"}:
        return "compatible" if "triterpene_cyclase" in family else "family_mismatch"
    if str(terpene_type) in CORE_CLASS_I_TYPES:
        return (
            "compatible"
            if "plant_like_classI_II" in family or "bacterial_classI" in family
            else "family_mismatch"
        )
    if str(terpene_type) in EXTENDED_TYPES:
        return "extended_pathway_uncertain"
    return "unknown_reaction_type"


def reaction_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return (
        frame[
            [
                "reaction_signature",
                "substrate_name",
                "product_name",
                "terpene_type",
                "tps_class",
                "has_mechanism",
            ]
        ]
        .drop_duplicates("reaction_signature")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UniProt-expanded TPS retrieval quality.")
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED)
    parser.add_argument("--marts", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expanded_dir = args.expanded_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_csv(expanded_dir / "expanded_rankings.csv", dtype=str).fillna("")
    rescue = pd.read_csv(
        expanded_dir / "expanded_top100_rescue_rankings.csv", dtype=str
    ).fillna("")
    registry = pd.read_csv(
        "data/terpene_open_world_registry/reactions.csv", dtype=str
    ).fillna("")
    metadata = reaction_metadata(args.marts.resolve())
    reaction_table = registry[["reaction_id", "reaction_signature"]].merge(
        metadata, on="reaction_signature", how="left"
    ).fillna("")
    reaction_map = reaction_table.set_index("reaction_id")
    contracts = pd.read_csv(args.contracts, dtype=str).fillna("")
    contract_map = contracts.set_index("reaction_id")

    for frame in (rankings, rescue):
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame["ensemble_score_mean"] = pd.to_numeric(
            frame["ensemble_score_mean"], errors="coerce"
        )
        frame["terpene_type"] = frame["query_id"].map(
            reaction_map["terpene_type"].to_dict()
        ).fillna("")
        frame["tps_class"] = frame["query_id"].map(
            reaction_map["tps_class"].to_dict()
        ).fillna("")
        frame["substrate_name"] = frame["query_id"].map(
            reaction_map["substrate_name"].to_dict()
        ).fillna("")
        frame["product_name"] = frame["query_id"].map(
            reaction_map["product_name"].to_dict()
        ).fillna("")
        frame["pfam_architecture"] = frame["pfam_combination"].map(
            pfam_architecture
        )
        frame["contract_status"] = frame["query_id"].map(
            contract_map["contract_status"].to_dict()
        ).fillna("contract_missing")
        frame["allowed_candidate_architectures"] = frame["query_id"].map(
            contract_map["allowed_candidate_architectures"].to_dict()
        ).fillna("")
        compatibility = []
        for query_id, architecture, source, allowed_text, contract_status in zip(
            frame["query_id"],
            frame["pfam_architecture"],
            frame["candidate_source"],
            frame["allowed_candidate_architectures"],
            frame["contract_status"],
        ):
            if source != "uniprot_primary":
                compatibility.append("not_applicable_non_uniprot")
                continue
            allowed = {value for value in str(allowed_text).split(";") if value}
            if not allowed:
                compatibility.append("unsupported_expansion_family")
            elif architecture in allowed:
                compatibility.append("compatible")
            else:
                compatibility.append("architecture_mismatch")
        frame["family_compatibility"] = compatibility

    rankings.to_csv(output_dir / "expanded_rankings_annotated.csv", index=False)
    rescue.to_csv(output_dir / "expanded_top100_annotated.csv", index=False)
    uniprot = rankings[rankings["candidate_source"].eq("uniprot_primary")].copy()
    uniprot_rescue = rescue[rescue["candidate_source"].eq("uniprot_primary")].copy()

    objective_rows = []
    for objective, group in rankings.groupby("ranking_objective", sort=True):
        uni = group[group["candidate_source"].eq("uniprot_primary")]
        queries = group["query_id"].nunique()
        objective_rows.append(
            {
                "ranking_objective": objective,
                "n_queries": queries,
                "ranking_rows": len(group),
                "uniprot_rows": len(uni),
                "uniprot_fraction": len(uni) / len(group),
                "queries_with_uniprot": uni["query_id"].nunique(),
                "queries_with_uniprot_fraction": uni["query_id"].nunique() / queries,
                "compatible_uniprot_fraction": (
                    uni["family_compatibility"].eq("compatible").mean()
                    if len(uni)
                    else np.nan
                ),
                "architecture_mismatch_or_unsupported_fraction": (
                    uni["family_compatibility"].isin(
                        ["architecture_mismatch", "unsupported_expansion_family"]
                    ).mean()
                    if len(uni)
                    else np.nan
                ),
                "experimental_transcript_or_reviewed_fraction": (
                    uni["evidence_quality_tier"].isin(
                        ["A_reviewed", "B_experimental_or_transcript_named"]
                    ).mean()
                    if len(uni)
                    else np.nan
                ),
            }
        )
    objective_summary = pd.DataFrame(objective_rows)
    objective_summary.to_csv(output_dir / "objective_quality_summary.csv", index=False)

    evidence_summary = (
        uniprot.groupby(
            [
                "ranking_objective",
                "evidence_quality_tier",
                "domain_family",
                "pfam_architecture",
                "family_compatibility",
            ],
            dropna=False,
        )
        .agg(
            appearance_count=("candidate_id", "size"),
            unique_candidates=("candidate_id", "nunique"),
            unique_queries=("query_id", "nunique"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        )
        .reset_index()
        .sort_values(
            ["ranking_objective", "appearance_count"], ascending=[True, False]
        )
    )
    evidence_summary.to_csv(output_dir / "evidence_family_summary.csv", index=False)

    frequency = (
        uniprot_rescue.groupby(
            [
                "candidate_id",
                "evidence_quality_tier",
                "domain_family",
                "pfam_architecture",
                "protein_name",
                "organism_name",
            ]
        )
        .agg(
            query_appearance_count=("query_id", "nunique"),
            total_appearance_count=("query_id", "size"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            compatible_appearances=(
                "family_compatibility",
                lambda values: int((values == "compatible").sum()),
            ),
            mismatch_or_unsupported_appearances=(
                "family_compatibility",
                lambda values: int(
                    values.isin(
                        ["architecture_mismatch", "unsupported_expansion_family"]
                    ).sum()
                ),
            ),
        )
        .reset_index()
        .sort_values(
            ["query_appearance_count", "best_rank"], ascending=[False, True]
        )
    )
    frequency["query_fraction"] = frequency["query_appearance_count"] / rankings[
        "query_id"
    ].nunique()
    frequency.to_csv(output_dir / "uniprot_rescue_candidate_frequency.csv", index=False)

    mismatch = uniprot[
        uniprot["family_compatibility"].isin(
            ["family_mismatch", "architecture_mismatch", "unsupported_expansion_family"]
        )
    ].copy()
    mismatch.to_csv(output_dir / "family_mismatch_top_lists.csv", index=False)
    priority_rescue = uniprot_rescue[
        uniprot_rescue["family_compatibility"].isin(
            ["compatible", "extended_pathway_uncertain"]
        )
    ].copy()
    priority_rescue["evidence_priority"] = priority_rescue[
        "evidence_quality_tier"
    ].map(
        {
            "A_reviewed": 0,
            "B_experimental_or_transcript_named": 1,
            "C_homology_named": 2,
            "D_named_predicted": 3,
        }
    ).fillna(9)
    priority_rescue = priority_rescue.sort_values(
        ["query_id", "evidence_priority", "rank", "candidate_id"]
    )
    priority_rescue.to_csv(
        output_dir / "evidence_tiered_compatible_rescue_candidates.csv", index=False
    )

    top_hub = frequency.iloc[0] if len(frequency) else None
    summary = {
        "candidate_universe_expansion_is_unlabelled": True,
        "architecture_compatibility_is_known_positive_contract_based": True,
        "contract_supported_reactions": int(contracts["rescue_supported"].astype(str).str.lower().eq("true").sum()),
        "contract_unsupported_reactions": int(contracts["rescue_supported"].astype(str).str.lower().eq("false").sum()),
        "fragment_architectures_are_not_eligible_for_controlled_rescue": True,
        "objective_quality": objective_summary.to_dict("records"),
        "uniprot_top_list_evidence_tiers": uniprot[
            "evidence_quality_tier"
        ].value_counts().to_dict(),
        "uniprot_top_list_domain_families": uniprot[
            "domain_family"
        ].value_counts().to_dict(),
        "uniprot_top_list_compatibility": uniprot[
            "family_compatibility"
        ].value_counts().to_dict(),
        "top_uniprot_rescue_hub": (
            {
                "candidate_id": str(top_hub["candidate_id"]),
                "query_appearance_count": int(top_hub["query_appearance_count"]),
                "query_fraction": float(top_hub["query_fraction"]),
                "evidence_quality_tier": str(top_hub["evidence_quality_tier"]),
                "domain_family": str(top_hub["domain_family"]),
            }
            if top_hub is not None
            else None
        ),
        "outputs": {
            "annotated_rankings": str(
                output_dir / "expanded_rankings_annotated.csv"
            ),
            "objective_summary": str(
                output_dir / "objective_quality_summary.csv"
            ),
            "evidence_summary": str(
                output_dir / "evidence_family_summary.csv"
            ),
            "compatible_rescue": str(
                output_dir / "evidence_tiered_compatible_rescue_candidates.csv"
            ),
            "mismatch": str(output_dir / "family_mismatch_top_lists.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(objective_summary.to_string(index=False))
    if len(frequency):
        print("\nMost frequent UniProt rescue candidates:")
        print(
            frequency.head(20)[
                [
                    "candidate_id",
                    "evidence_quality_tier",
                    "domain_family",
                    "query_appearance_count",
                    "query_fraction",
                    "best_rank",
                    "protein_name",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
