from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PANELS = ROOT / "results/terpene_wetlab_discovery_panels"
DEFAULT_OUTPUT = ROOT / "results/terpene_wetlab_plate_manifest"
ROWS = tuple("ABCDEFGH")


def clean_sequence(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def sequence_flags(sequence: str) -> tuple[str, bool, str]:
    length = len(sequence)
    if length < 200:
        length_flag = "very_short"
    elif length > 1000:
        length_flag = "very_long"
    elif length > 800:
        length_flag = "long"
    else:
        length_flag = "standard"
    noncanonical = bool(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    review_reasons = []
    if length_flag in {"very_short", "very_long"}:
        review_reasons.append(length_flag)
    if noncanonical:
        review_reasons.append("noncanonical_residue")
    return length_flag, noncanonical, ";".join(review_reasons)


def well(row: str, column: int) -> str:
    return f"{row}{column}"


def assay_row(
    *,
    plate_id: str,
    well_id: str,
    reaction: pd.Series,
    assay_role: str,
    candidate_id: str,
    candidate_source: str = "",
    panel_role: str = "",
    original_rank: object = "",
    sequence: str = "",
    sequence_length: object = "",
    enzyme_name: str = "",
    species: str = "",
) -> dict[str, object]:
    return {
        "plate_id": plate_id,
        "well": well_id,
        "reaction_order": int(reaction["balanced_campaign_order"]),
        "reaction_id": reaction["reaction_id"],
        "terpene_type": reaction.get("terpene_type", ""),
        "tps_class": reaction.get("tps_class", ""),
        "substrate_name": reaction.get("substrate_name", ""),
        "product_name": reaction.get("product_name", ""),
        "assay_role": assay_role,
        "candidate_id": candidate_id,
        "candidate_source": candidate_source,
        "panel_role": panel_role,
        "original_rank": original_rank,
        "sequence": sequence,
        "sequence_length": sequence_length,
        "enzyme_name": enzyme_name,
        "species": species,
    }


def write_fasta(frame: pd.DataFrame, path: Path, id_column: str, aliases_column: str | None = None) -> None:
    lines: list[str] = []
    for row in frame.itertuples(index=False):
        identifier = str(getattr(row, id_column))
        sequence = str(row.sequence)
        aliases = str(getattr(row, aliases_column)) if aliases_column else identifier
        lines.append(f">{identifier}|aliases={aliases}|aa={len(sequence)}")
        lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TPS wet-lab construct and 96-well plate manifests.")
    parser.add_argument("--panels-dir", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    panels_dir = args.panels_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reactions = pd.read_csv(panels_dir / "campaign_reactions.csv", dtype=str).fillna("")
    candidates = pd.read_csv(panels_dir / "campaign_discovery_candidates.csv", dtype=str).fillna("")
    controls = pd.read_csv(panels_dir / "campaign_positive_controls.csv", dtype=str).fillna("")
    reactions["balanced_campaign_order"] = pd.to_numeric(
        reactions["balanced_campaign_order"], errors="raise"
    ).astype(int)
    candidates["panel_order"] = pd.to_numeric(candidates["panel_order"], errors="raise").astype(int)
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="coerce")
    reactions = reactions.sort_values("balanced_campaign_order").reset_index(drop=True)

    candidate_groups = {
        reaction_id: group.sort_values("panel_order")
        for reaction_id, group in candidates.groupby("reaction_id", sort=False)
    }
    control_by_reaction = controls.drop_duplicates("reaction_id").set_index("reaction_id")
    plate_rows: list[dict[str, object]] = []
    for reaction_position, reaction in reactions.iterrows():
        plate_number = reaction_position // 6 + 1
        block = reaction_position % 6
        first_column = 2 * block + 1
        second_column = first_column + 1
        plate_id = f"TPS_DISCOVERY_P{plate_number:02d}"
        reaction_id = str(reaction["reaction_id"])
        group = candidate_groups[reaction_id]
        if len(group) != 12:
            raise ValueError(f"Expected 12 discovery candidates for {reaction_id}, found {len(group)}")
        for local_index, (_, candidate) in enumerate(group.iterrows()):
            if local_index < 8:
                well_id = well(ROWS[local_index], first_column)
            else:
                well_id = well(ROWS[local_index - 8], second_column)
            plate_rows.append(
                assay_row(
                    plate_id=plate_id,
                    well_id=well_id,
                    reaction=reaction,
                    assay_role="discovery_candidate",
                    candidate_id=str(candidate["candidate_id"]),
                    candidate_source=str(candidate.get("candidate_source", "")),
                    panel_role=str(candidate.get("panel_role", "")),
                    original_rank=candidate.get("rank", ""),
                    sequence=clean_sequence(candidate.get("sequence", "")),
                    sequence_length=candidate.get("sequence_length", ""),
                    enzyme_name=str(candidate.get("enzyme_name", "")),
                    species=str(candidate.get("species", "")),
                )
            )
        control = control_by_reaction.loc[reaction_id]
        control_kwargs = {
            "candidate_id": str(control["candidate_id"]),
            "candidate_source": str(control.get("candidate_source", "")),
            "panel_role": "positive_control",
            "sequence": clean_sequence(control.get("sequence", "")),
            "sequence_length": control.get("sequence_length", ""),
            "enzyme_name": str(control.get("enzyme_name", "")),
            "species": str(control.get("species", "")),
        }
        plate_rows.append(
            assay_row(
                plate_id=plate_id,
                well_id=well("E", second_column),
                reaction=reaction,
                assay_role="positive_control_primary",
                **control_kwargs,
            )
        )
        plate_rows.append(
            assay_row(
                plate_id=plate_id,
                well_id=well("F", second_column),
                reaction=reaction,
                assay_role="positive_control_replicate",
                **control_kwargs,
            )
        )
        plate_rows.append(
            assay_row(
                plate_id=plate_id,
                well_id=well("G", second_column),
                reaction=reaction,
                assay_role="empty_vector_negative",
                candidate_id="EMPTY_VECTOR",
            )
        )
        plate_rows.append(
            assay_row(
                plate_id=plate_id,
                well_id=well("H", second_column),
                reaction=reaction,
                assay_role="substrate_process_blank",
                candidate_id="NO_ENZYME",
            )
        )

    assays = pd.DataFrame(plate_rows).sort_values(["plate_id", "well"])
    if assays.duplicated(["plate_id", "well"]).any():
        raise ValueError("Plate layout contains duplicate wells")
    plate_sizes = assays.groupby("plate_id").size()
    if not plate_sizes.eq(96).all():
        raise ValueError(f"Every plate must contain 96 wells; found {plate_sizes.to_dict()}")

    construct_assays = assays[
        assays["assay_role"].isin(
            ["discovery_candidate", "positive_control_primary", "positive_control_replicate"]
        )
    ].copy()
    constructs = []
    for candidate_id, group in construct_assays.groupby("candidate_id", sort=True):
        sequence_values = [value for value in group["sequence"].astype(str).unique() if value]
        if len(sequence_values) > 1:
            raise ValueError(f"Conflicting sequences for {candidate_id}")
        sequence = sequence_values[0] if sequence_values else ""
        length_flag, noncanonical, review_reason = sequence_flags(sequence)
        roles = sorted(set(group["assay_role"].astype(str)))
        panel_roles = sorted(value for value in set(group["panel_role"].astype(str)) if value)
        constructs.append(
            {
                "candidate_id": candidate_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "candidate_source": ";".join(sorted(set(group["candidate_source"].astype(str)) - {""})),
                "enzyme_name": next((value for value in group["enzyme_name"].astype(str) if value), ""),
                "species": next((value for value in group["species"].astype(str) if value), ""),
                "assay_roles": ";".join(roles),
                "panel_roles": ";".join(panel_roles),
                "reaction_count": int(group["reaction_id"].nunique()),
                "assay_well_count": len(group),
                "sequence_length_flag": length_flag,
                "contains_noncanonical_residue": noncanonical,
                "manual_review_reason": review_reason,
                "construct_readiness": "needs_manual_review" if review_reason else "sequence_ready",
            }
        )
    constructs = pd.DataFrame(constructs)
    constructs["sequence_sha1"] = constructs["sequence"].map(
        lambda value: hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    )
    sequence_groups = []
    candidate_to_sequence_construct: dict[str, str] = {}
    for sequence_index, (sequence, group) in enumerate(
        constructs.groupby("sequence", sort=True), start=1
    ):
        aliases = sorted(group["candidate_id"].astype(str))
        sequence_construct_id = f"TPSSEQ_{sequence_index:04d}"
        for candidate_id in aliases:
            candidate_to_sequence_construct[candidate_id] = sequence_construct_id
        sequence_groups.append(
            {
                "sequence_construct_id": sequence_construct_id,
                "candidate_id_aliases": ";".join(aliases),
                "n_candidate_ids": len(aliases),
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_sha1": hashlib.sha1(str(sequence).encode("utf-8")).hexdigest(),
                "candidate_sources": ";".join(
                    sorted(set(group["candidate_source"].astype(str)) - {""})
                ),
                "assay_roles": ";".join(sorted(set(";".join(group["assay_roles"]).split(";")) - {""})),
                "panel_roles": ";".join(sorted(set(";".join(group["panel_roles"]).split(";")) - {""})),
                "reaction_count": int(
                    construct_assays[
                        construct_assays["candidate_id"].isin(aliases)
                    ]["reaction_id"].nunique()
                ),
                "assay_well_count": int(group["assay_well_count"].sum()),
            }
        )
    sequence_constructs = pd.DataFrame(sequence_groups)
    constructs["sequence_construct_id"] = constructs["candidate_id"].map(
        candidate_to_sequence_construct
    )
    assays["sequence_construct_id"] = assays["candidate_id"].map(
        candidate_to_sequence_construct
    ).fillna("")
    assays.to_csv(output_dir / "assay_manifest.csv", index=False)
    constructs.to_csv(output_dir / "unique_constructs.csv", index=False)
    sequence_constructs.to_csv(
        output_dir / "sequence_deduplicated_constructs.csv", index=False
    )
    write_fasta(
        constructs.sort_values("candidate_id"),
        output_dir / "candidate_id_constructs.fasta",
        "candidate_id",
    )
    write_fasta(
        sequence_constructs.sort_values("sequence_construct_id"),
        output_dir / "sequence_deduplicated_constructs.fasta",
        "sequence_construct_id",
        "candidate_id_aliases",
    )
    for plate_id, group in assays.groupby("plate_id", sort=True):
        group.to_csv(output_dir / f"{plate_id}.csv", index=False)
        matrix = group.pivot(index="well", columns=[] if False else None) if False else None
        layout = pd.DataFrame(index=ROWS, columns=range(1, 13), dtype=object)
        for row in group.itertuples(index=False):
            layout.loc[str(row.well)[0], int(str(row.well)[1:])] = (
                f"{row.reaction_id}|{row.assay_role}|{row.candidate_id}"
            )
        layout.index.name = "row"
        layout.to_csv(output_dir / f"{plate_id}_layout.csv")
        plate_sequence_ids = sorted(
            set(group["sequence_construct_id"].astype(str)) - {""}
        )
        write_fasta(
            sequence_constructs[
                sequence_constructs["sequence_construct_id"].isin(plate_sequence_ids)
            ].sort_values("sequence_construct_id"),
            output_dir / f"{plate_id}_constructs.fasta",
            "sequence_construct_id",
            "candidate_id_aliases",
        )

    plate_summary = (
        assays.groupby("plate_id")
        .agg(
            n_wells=("well", "size"),
            n_reactions=("reaction_id", "nunique"),
            n_discovery_assays=("assay_role", lambda values: int((values == "discovery_candidate").sum())),
            n_positive_control_wells=("assay_role", lambda values: int(values.str.startswith("positive_control").sum())),
            n_negative_control_wells=("assay_role", lambda values: int((values == "empty_vector_negative").sum())),
            n_process_blank_wells=("assay_role", lambda values: int((values == "substrate_process_blank").sum())),
            unique_protein_constructs=("candidate_id", lambda values: len(set(values) - {"EMPTY_VECTOR", "NO_ENZYME"})),
        )
        .reset_index()
    )
    plate_summary.to_csv(output_dir / "plate_summary.csv", index=False)
    review = constructs[constructs["construct_readiness"].eq("needs_manual_review")].copy()
    review.to_csv(output_dir / "constructs_needing_manual_review.csv", index=False)
    summary = {
        "n_plates": int(assays["plate_id"].nunique()),
        "wells_per_plate": 96,
        "n_reactions": int(assays["reaction_id"].nunique()),
        "n_discovery_assays": int(assays["assay_role"].eq("discovery_candidate").sum()),
        "n_positive_control_wells": int(assays["assay_role"].str.startswith("positive_control").sum()),
        "n_empty_vector_negative_wells": int(assays["assay_role"].eq("empty_vector_negative").sum()),
        "n_substrate_process_blank_wells": int(assays["assay_role"].eq("substrate_process_blank").sum()),
        "n_candidate_id_constructs": len(constructs),
        "n_sequence_deduplicated_constructs": len(sequence_constructs),
        "redundant_candidate_ids_collapsed": len(constructs) - len(sequence_constructs),
        "total_amino_acids_sequence_deduplicated": int(sequence_constructs["sequence_length"].sum()),
        "total_coding_nucleotides_without_stop": int(3 * sequence_constructs["sequence_length"].sum()),
        "n_sequence_ready": int(constructs["construct_readiness"].eq("sequence_ready").sum()),
        "n_constructs_needing_manual_review": len(review),
        "sequence_length_flags": constructs["sequence_length_flag"].value_counts().to_dict(),
        "outputs": {
            "assay_manifest": str(output_dir / "assay_manifest.csv"),
            "candidate_id_constructs": str(output_dir / "unique_constructs.csv"),
            "sequence_deduplicated_constructs": str(output_dir / "sequence_deduplicated_constructs.csv"),
            "sequence_deduplicated_fasta": str(output_dir / "sequence_deduplicated_constructs.fasta"),
            "plate_summary": str(output_dir / "plate_summary.csv"),
            "manual_review": str(output_dir / "constructs_needing_manual_review.csv"),
        },
        "codon_optimization_performed": False,
        "codon_optimization_note": "Only protein sequences are exported. Expression host and vector architecture must be fixed before host-specific codon optimization or synthesis ordering.",
        "layout_note": "Each reaction occupies two columns: candidates 1-8 in A-H of the first column; candidates 9-12 in A-D of the second; positive control primary/replicate in E/F; empty-vector negative in G; substrate/process blank in H.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(plate_summary.to_string(index=False))
    if len(review):
        print("\nConstructs needing manual review:")
        print(review[["candidate_id", "sequence_length", "sequence_length_flag", "contains_noncanonical_residue", "manual_review_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
