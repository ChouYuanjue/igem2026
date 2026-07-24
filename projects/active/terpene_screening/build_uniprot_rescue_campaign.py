from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.audit_uniprot_rescue_sequence_integrity import (  # noqa: E402
    conservative_sequence_risk,
)
from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (  # noqa: E402
    pfam_architecture,
)

DEFAULT_QUALITY = ROOT / "results/terpene_uniprot_expansion_quality"
DEFAULT_CAMPAIGN = ROOT / "results/terpene_wetlab_discovery_panels/campaign_reactions.csv"
DEFAULT_REACTION_SUMMARY = ROOT / "results/terpene_wetlab_discovery_panels/reaction_panel_summary.csv"
DEFAULT_CONTROLS = ROOT / "results/terpene_wetlab_discovery_panels/reaction_positive_controls.csv"
DEFAULT_CONTRACTS = (
    ROOT
    / "data/terpene_uniprot_expansion/reaction_architecture_contracts/reaction_architecture_contracts.csv"
)
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/uniprot_tps_primary_esmc600m"
DEFAULT_METADATA = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_primary_embedding_candidates.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_rescue_campaign"
EVIDENCE_PRIORITY = {
    "A_reviewed": 0,
    "B_experimental_or_transcript_named": 1,
    "C_homology_named": 2,
    "D_named_predicted": 3,
}
ROWS = tuple("ABCDEFGH")


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str]]:
    entries = pd.read_csv(path / "entries.csv", dtype=str).sort_values("row")
    matrix = np.load(path / "embeddings.npy").astype(np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    if len(entries) != len(matrix):
        raise ValueError("UniProt embedding entries and matrix differ in length")
    return matrix / denominator, entries["Entry"].astype(str).tolist()


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    for row in frame.itertuples(index=False):
        lines.append(
            f">{row.sequence_construct_id}|aliases={row.candidate_id_aliases}|aa={len(str(row.sequence))}"
        )
        sequence = str(row.sequence)
        lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def select_supported_campaign(
    base_campaign: pd.DataFrame,
    reaction_summary: pd.DataFrame,
    contracts: pd.DataFrame,
    eligible_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = set(
        contracts.loc[
            contracts["rescue_supported"].astype(str).str.lower().eq("true"),
            "reaction_id",
        ].astype(str)
    )
    pool_size = eligible_pool.groupby("query_id")["candidate_id"].nunique().to_dict()
    all_rows = reaction_summary.copy()
    all_rows["campaign_priority_rank"] = pd.to_numeric(
        all_rows["campaign_priority_rank"], errors="coerce"
    ).fillna(10**9)
    all_rows = all_rows[
        all_rows["reaction_id"].isin(supported)
        & all_rows["positive_control_available"].astype(str).str.lower().eq("true")
        & all_rows["reaction_id"].map(pool_size).fillna(0).ge(4)
    ].copy()
    all_by_id = all_rows.drop_duplicates("reaction_id").set_index("reaction_id")
    selected_ids: set[str] = set()
    reserved_base_supported = {
        str(value)
        for value in base_campaign["reaction_id"].astype(str)
        if value in supported and pool_size.get(str(value), 0) >= 4
    }
    selected_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    used_substrates: dict[str, int] = {}
    used_controls: dict[str, int] = {}

    def add_row(row: pd.Series, order: int, replaced: str = "") -> None:
        reaction_id = str(row["reaction_id"])
        selected_ids.add(reaction_id)
        substrate = str(row.get("substrate_name", ""))
        control = str(row.get("positive_control_id", ""))
        used_substrates[substrate] = used_substrates.get(substrate, 0) + 1
        used_controls[control] = used_controls.get(control, 0) + 1
        payload = row.to_dict()
        payload["balanced_campaign_order"] = order
        payload["rescue_target_source"] = (
            "base_supported" if not replaced else "supported_replacement"
        )
        payload["replaced_reaction_id"] = replaced
        payload["compatible_candidate_pool_size"] = int(pool_size.get(reaction_id, 0))
        selected_rows.append(payload)

    base_campaign = base_campaign.sort_values("balanced_campaign_order")
    for base in base_campaign.itertuples(index=False):
        reaction_id = str(base.reaction_id)
        if reaction_id in supported and pool_size.get(reaction_id, 0) >= 4:
            row = all_by_id.loc[reaction_id].copy() if reaction_id in all_by_id.index else pd.Series(base._asdict())
            row["reaction_id"] = reaction_id
            add_row(row, int(base.balanced_campaign_order))
            continue
        candidates = all_rows[
            ~all_rows["reaction_id"].isin(selected_ids | reserved_base_supported)
        ].copy()
        same_type = candidates[candidates["terpene_type"].eq(str(base.terpene_type))]
        if len(same_type):
            candidates = same_type
        candidates["substrate_reuse"] = candidates["substrate_name"].map(
            used_substrates
        ).fillna(0)
        candidates["control_reuse"] = candidates["positive_control_id"].map(
            used_controls
        ).fillna(0)
        candidates["candidate_pool_size"] = candidates["reaction_id"].map(pool_size).fillna(0)
        candidates = candidates.sort_values(
            [
                "substrate_reuse",
                "control_reuse",
                "campaign_priority_rank",
                "candidate_pool_size",
                "reaction_id",
            ],
            ascending=[True, True, True, False, True],
        )
        if candidates.empty:
            raise ValueError(f"No supported replacement for {reaction_id}")
        replacement = candidates.iloc[0]
        add_row(replacement, int(base.balanced_campaign_order), reaction_id)
        excluded_rows.append(
            {
                "reaction_id": reaction_id,
                "terpene_type": str(base.terpene_type),
                "substrate_name": str(base.substrate_name),
                "product_name": str(base.product_name),
                "replacement_reaction_id": str(replacement["reaction_id"]),
                "reason": "reference_family_not_supported_by_five_pfam_expansion",
            }
        )
    selected = pd.DataFrame(selected_rows).sort_values("balanced_campaign_order")
    if len(selected) != len(base_campaign) or selected["reaction_id"].nunique() != len(selected):
        raise ValueError("Supported rescue campaign must preserve size with unique reactions")
    return selected, pd.DataFrame(excluded_rows)


def choose_ranked(
    pool: pd.DataFrame,
    allowed_tiers: set[str],
    selected: set[str],
    usage: dict[str, int],
    hub_frequency: dict[str, int],
    maximum_global_usage: int,
) -> str | None:
    candidates = pool[
        pool["evidence_quality_tier"].isin(allowed_tiers)
        & ~pool["candidate_id"].isin(selected)
    ].copy()
    if candidates.empty:
        return None
    candidates["usage"] = candidates["candidate_id"].map(usage).fillna(0).astype(int)
    candidates["hub_frequency"] = (
        candidates["candidate_id"].map(hub_frequency).fillna(0).astype(int)
    )
    candidates["usage_over_cap"] = candidates["usage"].ge(maximum_global_usage).astype(int)
    candidates["evidence_priority"] = candidates["evidence_quality_tier"].map(
        EVIDENCE_PRIORITY
    ).fillna(9)
    row = candidates.sort_values(
        [
            "usage_over_cap",
            "evidence_priority",
            "usage",
            "hub_frequency",
            "rank",
            "candidate_id",
        ],
        ascending=[True, True, True, True, True, True],
    ).iloc[0]
    return str(row["candidate_id"])


def choose_diverse(
    pool: pd.DataFrame,
    selected: set[str],
    usage: dict[str, int],
    hub_frequency: dict[str, int],
    feature_matrix: np.ndarray,
    id_to_row: dict[str, int],
    maximum_global_usage: int,
) -> str | None:
    candidates = pool[~pool["candidate_id"].isin(selected)].copy()
    if candidates.empty:
        return None
    selected_rows = np.asarray(
        [id_to_row[value] for value in selected if value in id_to_row], dtype=np.int64
    )
    values = []
    max_hub = max(hub_frequency.values()) if hub_frequency else 1
    for row in candidates.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        feature = feature_matrix[id_to_row[candidate_id]]
        min_distance = (
            float(np.min(1.0 - feature_matrix[selected_rows] @ feature))
            if len(selected_rows)
            else 1.0
        )
        evidence_bonus = 0.04 * (3 - EVIDENCE_PRIORITY.get(str(row.evidence_quality_tier), 3))
        rank_bonus = 0.10 * (1.0 - (float(row.rank) - 1.0) / 99.0)
        usage_penalty = 0.15 * usage.get(candidate_id, 0)
        if usage.get(candidate_id, 0) >= maximum_global_usage:
            usage_penalty += 0.50
        hub_penalty = 0.08 * hub_frequency.get(candidate_id, 0) / max_hub
        values.append(
            (
                min_distance + evidence_bonus + rank_bonus - usage_penalty - hub_penalty,
                -float(row.rank),
                candidate_id,
            )
        )
    return max(values)[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-tiered UniProt TPS rescue plates.")
    parser.add_argument("--quality-dir", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--reaction-summary", type=Path, default=DEFAULT_REACTION_SUMMARY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-global-usage", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_campaign = pd.read_csv(args.campaign, dtype=str).fillna("")
    base_campaign["balanced_campaign_order"] = pd.to_numeric(
        base_campaign["balanced_campaign_order"], errors="raise"
    ).astype(int)
    reaction_summary = pd.read_csv(args.reaction_summary, dtype=str).fillna("")
    contracts = pd.read_csv(args.contracts, dtype=str).fillna("")
    controls = pd.read_csv(args.controls, dtype=str).fillna("")
    control_by_reaction = controls.drop_duplicates("reaction_id").set_index("reaction_id")
    pool = pd.read_csv(
        args.quality_dir.resolve() / "evidence_tiered_compatible_rescue_candidates.csv",
        dtype=str,
    ).fillna("")
    pool["rank"] = pd.to_numeric(pool["rank"], errors="coerce")
    metadata = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
    metadata = metadata.rename(columns={"accession": "candidate_id"})
    pool = pool.merge(
        metadata[["candidate_id", "sequence"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    if pool["sequence"].eq("").any():
        raise ValueError("Contract-compatible rescue pool contains missing sequences")
    risk_values = [
        conservative_sequence_risk(sequence, pfam)
        for sequence, pfam in zip(pool["sequence"], pool["pfam_combination"])
    ]
    pool["high_confidence_sequence_risk"] = [value[0] for value in risk_values]
    pool["sequence_risk_reason"] = [value[1] for value in risk_values]
    excluded_sequence_risks = pool[pool["high_confidence_sequence_risk"]].copy()
    pool = pool[~pool["high_confidence_sequence_risk"]].copy()
    campaign, excluded_targets = select_supported_campaign(
        base_campaign, reaction_summary, contracts, pool
    )
    pool = pool[pool["query_id"].isin(set(campaign["reaction_id"]))].copy()
    feature_matrix, feature_ids = load_embeddings(args.embedding_dir.resolve())
    id_to_row = {value: index for index, value in enumerate(feature_ids)}
    pool = pool[pool["candidate_id"].isin(id_to_row)].copy()
    hub_frequency = pool.groupby("candidate_id")["query_id"].nunique().astype(int).to_dict()
    usage: dict[str, int] = {}
    selected_rows: list[dict[str, object]] = []

    for reaction in campaign.itertuples(index=False):
        reaction_id = str(reaction.reaction_id)
        reaction_pool = pool[pool["query_id"].eq(reaction_id)].drop_duplicates(
            "candidate_id"
        )
        selected: set[str] = set()
        role_candidates = [
            (
                "evidence_anchor",
                {"A_reviewed", "B_experimental_or_transcript_named"},
            ),
            ("homology_named", {"C_homology_named"}),
            ("named_predicted", {"D_named_predicted"}),
        ]
        for role, tiers in role_candidates:
            candidate_id = choose_ranked(
                reaction_pool,
                tiers,
                selected,
                usage,
                hub_frequency,
                args.maximum_global_usage,
            )
            if candidate_id is None and role == "evidence_anchor":
                candidate_id = choose_ranked(
                    reaction_pool,
                    {"C_homology_named"},
                    selected,
                    usage,
                    hub_frequency,
                    args.maximum_global_usage,
                )
                role = "best_available_evidence"
            if candidate_id is None:
                candidate_id = choose_ranked(
                    reaction_pool,
                    set(EVIDENCE_PRIORITY),
                    selected,
                    usage,
                    hub_frequency,
                    args.maximum_global_usage,
                )
            if candidate_id is None:
                raise ValueError(f"No candidate available for {reaction_id} role {role}")
            selected.add(candidate_id)
            usage[candidate_id] = usage.get(candidate_id, 0) + 1
            row = reaction_pool[reaction_pool["candidate_id"].eq(candidate_id)].iloc[0]
            selected_rows.append(
                {
                    "reaction_id": reaction_id,
                    "reaction_order": int(reaction.balanced_campaign_order),
                    "terpene_type": reaction.terpene_type,
                    "tps_class": reaction.tps_class,
                    "substrate_name": reaction.substrate_name,
                    "product_name": reaction.product_name,
                    "candidate_id": candidate_id,
                    "rescue_role": role,
                    "expanded_rank": int(row["rank"]),
                    "evidence_quality_tier": row["evidence_quality_tier"],
                    "domain_family": row["domain_family"],
                    "family_compatibility": row["family_compatibility"],
                    "global_core_pool_frequency": hub_frequency.get(candidate_id, 0),
                }
            )
        candidate_id = choose_diverse(
            reaction_pool,
            selected,
            usage,
            hub_frequency,
            feature_matrix,
            id_to_row,
            args.maximum_global_usage,
        )
        if candidate_id is None:
            raise ValueError(f"No diversity candidate for {reaction_id}")
        selected.add(candidate_id)
        usage[candidate_id] = usage.get(candidate_id, 0) + 1
        row = reaction_pool[reaction_pool["candidate_id"].eq(candidate_id)].iloc[0]
        selected_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_order": int(reaction.balanced_campaign_order),
                "terpene_type": reaction.terpene_type,
                "tps_class": reaction.tps_class,
                "substrate_name": reaction.substrate_name,
                "product_name": reaction.product_name,
                "candidate_id": candidate_id,
                "rescue_role": "sequence_diversity",
                "expanded_rank": int(row["rank"]),
                "evidence_quality_tier": row["evidence_quality_tier"],
                "domain_family": row["domain_family"],
                "family_compatibility": row["family_compatibility"],
                "global_core_pool_frequency": hub_frequency.get(candidate_id, 0),
            }
        )

    selected = pd.DataFrame(selected_rows)
    selected["rescue_order"] = selected.groupby("reaction_id").cumcount() + 1
    selected = selected.merge(
        metadata[
            [
                "candidate_id",
                "sequence",
                "length",
                "protein_name",
                "gene_names",
                "organism_name",
                "protein_existence",
                "pfam_combination",
                "cluster_id",
                "cluster_size",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    if selected["sequence"].eq("").any():
        raise ValueError("Selected UniProt rescue candidate lacks sequence")

    assay_rows: list[dict[str, object]] = []
    for reaction_position, reaction in enumerate(
        campaign.itertuples(index=False), start=0
    ):
        plate_number = reaction_position // 12 + 1
        column = reaction_position % 12 + 1
        plate_id = f"TPS_UNIPROT_RESCUE_P{plate_number:02d}"
        reaction_id = str(reaction.reaction_id)
        group = selected[selected["reaction_id"].eq(reaction_id)].sort_values(
            "rescue_order"
        )
        for row_letter, candidate in zip(ROWS[:4], group.itertuples(index=False)):
            assay_rows.append(
                {
                    "plate_id": plate_id,
                    "well": f"{row_letter}{column}",
                    "reaction_id": reaction_id,
                    "reaction_order": int(reaction.balanced_campaign_order),
                    "terpene_type": reaction.terpene_type,
                    "tps_class": reaction.tps_class,
                    "substrate_name": reaction.substrate_name,
                    "product_name": reaction.product_name,
                    "assay_role": "uniprot_rescue_candidate",
                    "candidate_id": candidate.candidate_id,
                    "rescue_role": candidate.rescue_role,
                    "expanded_rank": candidate.expanded_rank,
                    "evidence_quality_tier": candidate.evidence_quality_tier,
                    "domain_family": candidate.domain_family,
                    "sequence": candidate.sequence,
                }
            )
        control = control_by_reaction.loc[reaction_id]
        for row_letter, assay_role in [
            ("E", "positive_control_primary"),
            ("F", "positive_control_replicate"),
        ]:
            assay_rows.append(
                {
                    "plate_id": plate_id,
                    "well": f"{row_letter}{column}",
                    "reaction_id": reaction_id,
                    "reaction_order": int(reaction.balanced_campaign_order),
                    "terpene_type": reaction.terpene_type,
                    "tps_class": reaction.tps_class,
                    "substrate_name": reaction.substrate_name,
                    "product_name": reaction.product_name,
                    "assay_role": assay_role,
                    "candidate_id": control.candidate_id,
                    "rescue_role": "positive_control",
                    "expanded_rank": "",
                    "evidence_quality_tier": "known_positive_control",
                    "domain_family": "",
                    "sequence": control.sequence,
                }
            )
        for row_letter, assay_role, candidate_id in [
            ("G", "empty_vector_negative", "EMPTY_VECTOR"),
            ("H", "substrate_process_blank", "NO_ENZYME"),
        ]:
            assay_rows.append(
                {
                    "plate_id": plate_id,
                    "well": f"{row_letter}{column}",
                    "reaction_id": reaction_id,
                    "reaction_order": int(reaction.balanced_campaign_order),
                    "terpene_type": reaction.terpene_type,
                    "tps_class": reaction.tps_class,
                    "substrate_name": reaction.substrate_name,
                    "product_name": reaction.product_name,
                    "assay_role": assay_role,
                    "candidate_id": candidate_id,
                    "rescue_role": assay_role,
                    "expanded_rank": "",
                    "evidence_quality_tier": "",
                    "domain_family": "",
                    "sequence": "",
                }
            )

    assays = pd.DataFrame(assay_rows).sort_values(["plate_id", "well"])
    if assays.duplicated(["plate_id", "well"]).any():
        raise ValueError("Duplicate wells in UniProt rescue layout")
    plate_sizes = assays.groupby("plate_id").size()
    if not plate_sizes.eq(96).all():
        raise ValueError(f"Rescue plates must have 96 wells: {plate_sizes.to_dict()}")

    protein_assays = assays[
        assays["assay_role"].isin(
            [
                "uniprot_rescue_candidate",
                "positive_control_primary",
                "positive_control_replicate",
            ]
        )
    ]
    sequence_rows = []
    candidate_to_sequence_id: dict[str, str] = {}
    for index, (sequence, group) in enumerate(
        protein_assays.groupby("sequence", sort=True), start=1
    ):
        aliases = sorted(set(group["candidate_id"].astype(str)))
        sequence_id = f"TPSRESCUESEQ_{index:04d}"
        for alias in aliases:
            candidate_to_sequence_id[alias] = sequence_id
        sequence_rows.append(
            {
                "sequence_construct_id": sequence_id,
                "candidate_id_aliases": ";".join(aliases),
                "n_candidate_ids": len(aliases),
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_sha1": hashlib.sha1(sequence.encode("utf-8")).hexdigest(),
                "assay_well_count": len(group),
                "reaction_count": group["reaction_id"].nunique(),
            }
        )
    sequence_constructs = pd.DataFrame(sequence_rows)
    assays["sequence_construct_id"] = assays["candidate_id"].map(
        candidate_to_sequence_id
    ).fillna("")

    campaign.to_csv(output_dir / "rescue_campaign_reactions.csv", index=False)
    excluded_targets.to_csv(output_dir / "replaced_unsupported_reactions.csv", index=False)
    excluded_sequence_risks.to_csv(
        output_dir / "excluded_high_confidence_sequence_risks.csv", index=False
    )
    selected.to_csv(output_dir / "selected_uniprot_rescue_candidates.csv", index=False)
    assays.to_csv(output_dir / "assay_manifest.csv", index=False)
    sequence_constructs.to_csv(
        output_dir / "sequence_deduplicated_constructs.csv", index=False
    )
    write_fasta(sequence_constructs, output_dir / "sequence_deduplicated_constructs.fasta")
    for plate_id, group in assays.groupby("plate_id", sort=True):
        group.to_csv(output_dir / f"{plate_id}.csv", index=False)
        layout = pd.DataFrame(index=ROWS, columns=range(1, 13), dtype=object)
        for row in group.itertuples(index=False):
            layout.loc[str(row.well)[0], int(str(row.well)[1:])] = (
                f"{row.reaction_id}|{row.assay_role}|{row.candidate_id}"
            )
        layout.index.name = "row"
        layout.to_csv(output_dir / f"{plate_id}_layout.csv")

    selected["pfam_architecture"] = selected["pfam_combination"].map(
        pfam_architecture
    )
    summary = {
        "candidate_universe": "uniprot_primary_named_cluster_representatives",
        "empirical_reliability_status": "not_applicable_candidate_universe_expanded",
        "architecture_eligibility_is_known_positive_contract_based": True,
        "base_campaign_reactions": int(base_campaign["reaction_id"].nunique()),
        "supported_base_reactions_retained": int(
            campaign["rescue_target_source"].eq("base_supported").sum()
        ),
        "unsupported_base_reactions_replaced": int(len(excluded_targets)),
        "contract_supported_registry_reactions": int(
            contracts["rescue_supported"].astype(str).str.lower().eq("true").sum()
        ),
        "contract_unsupported_registry_reactions": int(
            contracts["rescue_supported"].astype(str).str.lower().eq("false").sum()
        ),
        "excluded_high_confidence_risk_rows": int(len(excluded_sequence_risks)),
        "excluded_high_confidence_risk_candidates": int(
            excluded_sequence_risks["candidate_id"].nunique()
        ),
        "selected_high_confidence_sequence_risks": 0,
        "n_reactions": selected["reaction_id"].nunique(),
        "n_selected_candidates": len(selected),
        "n_unique_selected_candidates": selected["candidate_id"].nunique(),
        "maximum_selected_candidate_usage": int(
            selected["candidate_id"].value_counts().max()
        ),
        "selection_roles": selected["rescue_role"].value_counts().to_dict(),
        "evidence_tiers": selected["evidence_quality_tier"].value_counts().to_dict(),
        "domain_families": selected["domain_family"].value_counts().to_dict(),
        "pfam_architectures": selected["pfam_architecture"].value_counts().to_dict(),
        "replacements": excluded_targets.to_dict("records"),
        "mean_expanded_rank": float(selected["expanded_rank"].mean()),
        "median_expanded_rank": float(selected["expanded_rank"].median()),
        "n_plates": assays["plate_id"].nunique(),
        "wells_per_plate": 96,
        "n_sequence_deduplicated_constructs_including_controls": len(
            sequence_constructs
        ),
        "n_uniprot_sequences": selected["sequence"].nunique(),
        "total_uniprot_amino_acids": int(
            selected.drop_duplicates("sequence")["sequence"].str.len().sum()
        ),
        "outputs": {
            "campaign_reactions": str(output_dir / "rescue_campaign_reactions.csv"),
            "replacements": str(output_dir / "replaced_unsupported_reactions.csv"),
            "excluded_sequence_risks": str(
                output_dir / "excluded_high_confidence_sequence_risks.csv"
            ),
            "selected_candidates": str(
                output_dir / "selected_uniprot_rescue_candidates.csv"
            ),
            "assay_manifest": str(output_dir / "assay_manifest.csv"),
            "constructs": str(
                output_dir / "sequence_deduplicated_constructs.csv"
            ),
            "fasta": str(
                output_dir / "sequence_deduplicated_constructs.fasta"
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(
        selected[
            [
                "reaction_order",
                "reaction_id",
                "candidate_id",
                "rescue_role",
                "expanded_rank",
                "evidence_quality_tier",
                "domain_family",
                "global_core_pool_frequency",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
