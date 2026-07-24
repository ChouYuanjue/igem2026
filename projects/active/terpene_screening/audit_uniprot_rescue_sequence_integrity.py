from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import pfam_architecture  # noqa: E402
DEFAULT_NORMALIZED = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_normalized.tsv"
DEFAULT_RESCUE = ROOT / "results/terpene_uniprot_rescue_campaign/selected_uniprot_rescue_candidates.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_rescue_sequence_integrity"

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
MINIMUM_COMPLETE_ARCHITECTURE_LENGTH = {
    "bacterial_classI": 280,
    "plant_tps_full": 400,
    "classI_hybrid_full": 400,
    "osc_full": 600,
    "classII_cyclase_single_domain": 450,
}


def motif_features(sequence: str) -> dict[str, object]:
    sequence = str(sequence).upper().replace(" ", "").rstrip("*")
    length = len(sequence)
    counts = Counter(sequence)
    entropy = 0.0
    if length:
        for count in counts.values():
            p = count / length
            entropy -= p * math.log2(p)
    max_run = 0
    current_run = 0
    previous = ""
    for residue in sequence:
        if residue == previous:
            current_run += 1
        else:
            current_run = 1
            previous = residue
        max_run = max(max_run, current_run)
    window = min(19, length)
    max_hydrophobic_fraction = 0.0
    if window:
        max_hydrophobic_fraction = max(
            sum(residue in HYDROPHOBIC for residue in sequence[start : start + window])
            / window
            for start in range(max(1, length - window + 1))
        )
    nterm = sequence[:50]
    return {
        "sequence_length": length,
        "canonical_amino_acids": set(sequence).issubset(CANONICAL_AA),
        "ddxxd_count": len(re.findall(r"DD[A-Z]{2}D", sequence)),
        "nse_like_count": len(re.findall(r"[ND]D[A-Z]{2}[ST][A-Z]{3}E", sequence)),
        "dxdd_count": len(re.findall(r"D[A-Z]DD", sequence)),
        "dctae_like_count": len(re.findall(r"D[CST][A-Z]{2}E", sequence)),
        "qw_near_count": len(re.findall(r"Q[A-Z]{0,4}W", sequence)),
        "literal_qw_count": sequence.count("QW"),
        "max_homopolymer_run": max_run,
        "sequence_entropy_bits": entropy,
        "most_common_residue_fraction": (
            max(counts.values()) / length if length else np.nan
        ),
        "max_19aa_hydrophobic_fraction": max_hydrophobic_fraction,
        "nterm_acidic_fraction": (
            sum(residue in "DE" for residue in nterm) / len(nterm) if nterm else np.nan
        ),
        "nterm_strar_fraction": (
            sum(residue in "STAR" for residue in nterm) / len(nterm) if nterm else np.nan
        ),
    }


def conservative_sequence_risk(
    sequence: str, pfam_combination: str
) -> tuple[bool, str]:
    features = motif_features(sequence)
    architecture = pfam_architecture(pfam_combination)
    minimum_length = MINIMUM_COMPLETE_ARCHITECTURE_LENGTH.get(architecture, 10**9)
    reasons = []
    if not bool(features["canonical_amino_acids"]):
        reasons.append("invalid_residue")
    if (
        float(features["sequence_entropy_bits"]) < 3.2
        or float(features["most_common_residue_fraction"]) > 0.20
        or int(features["max_homopolymer_run"]) >= 8
    ):
        reasons.append("low_complexity")
    if float(features["max_19aa_hydrophobic_fraction"]) >= 0.79:
        reasons.append("hydrophobic_segment")
    if int(features["sequence_length"]) < minimum_length:
        reasons.append("architecture_length")
    return bool(reasons), ";".join(reasons)


def motif_contract(architecture: str, row: pd.Series) -> tuple[bool, str]:
    class_i_complete = int(row["ddxxd_count"]) > 0 and int(row["nse_like_count"]) > 0
    class_ii_complete = int(row["dxdd_count"]) > 0
    osc_complete = int(row["dctae_like_count"]) > 0 and int(row["qw_near_count"]) >= 2
    if architecture == "osc_full":
        return osc_complete, "OSC:D[CST]xxE+>=2_QW-near"
    if architecture == "classII_cyclase_single_domain":
        return class_ii_complete, "class-II:DxDD"
    if architecture == "bacterial_classI":
        return class_i_complete, "class-I:DDxxD+NSE-like"
    if architecture == "plant_tps_full":
        return class_i_complete or class_ii_complete, "plant-TPS:(class-I-complete)_or_DxDD"
    if architecture == "classI_hybrid_full":
        return class_i_complete, "hybrid-class-I:DDxxD+NSE-like"
    return False, "unsupported_or_fragment_architecture"


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame([motif_features(value) for value in frame["sequence"]])
    result = pd.concat([frame.reset_index(drop=True), features], axis=1)
    result["pfam_architecture"] = result["pfam_combination"].map(pfam_architecture)
    contracts = [
        motif_contract(architecture, row)
        for architecture, (_, row) in zip(
            result["pfam_architecture"], result.iterrows()
        )
    ]
    result["expected_motif_contract_met"] = [value[0] for value in contracts]
    result["expected_motif_contract"] = [value[1] for value in contracts]
    result["low_complexity_risk"] = (
        (result["sequence_entropy_bits"] < 3.2)
        | (result["most_common_residue_fraction"] > 0.20)
        | (result["max_homopolymer_run"] >= 8)
    )
    result["hydrophobic_segment_risk"] = result["max_19aa_hydrophobic_fraction"] >= 0.79
    result["minimum_complete_architecture_length"] = result[
        "pfam_architecture"
    ].map(MINIMUM_COMPLETE_ARCHITECTURE_LENGTH).fillna(10**9).astype(int)
    result["hard_length_risk"] = (
        result["sequence_length"] < result["minimum_complete_architecture_length"]
    )
    result["high_confidence_sequence_risk"] = (
        ~result["canonical_amino_acids"]
        | result["low_complexity_risk"]
        | result["hydrophobic_segment_risk"]
        | result["hard_length_risk"]
    )
    result["sequence_integrity_status"] = np.select(
        [
            ~result["canonical_amino_acids"],
            result["low_complexity_risk"] | result["hydrophobic_segment_risk"],
            result["hard_length_risk"],
            ~result["expected_motif_contract_met"],
        ],
        [
            "invalid_residue",
            "sequence_composition_risk",
            "architecture_length_risk",
            "expected_motif_not_detected_annotation_only",
        ],
        default="motif_supported",
    )
    return result


def summarize(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    for family, group in frame.groupby("pfam_architecture", sort=True):
        rows.append(
            {
                "cohort": cohort,
                "pfam_architecture": family,
                "n_sequences": len(group),
                "median_length": float(group["sequence_length"].median()),
                "q05_length": float(group["sequence_length"].quantile(0.05)),
                "q95_length": float(group["sequence_length"].quantile(0.95)),
                "motif_contract_rate": float(group["expected_motif_contract_met"].mean()),
                "low_complexity_risk_rate": float(group["low_complexity_risk"].mean()),
                "hydrophobic_segment_risk_rate": float(group["hydrophobic_segment_risk"].mean()),
                "median_ddxxd_count": float(group["ddxxd_count"].median()),
                "median_nse_like_count": float(group["nse_like_count"].median()),
                "median_dxdd_count": float(group["dxdd_count"].median()),
                "median_dctae_like_count": float(group["dctae_like_count"].median()),
                "median_qw_near_count": float(group["qw_near_count"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit motif and sequence integrity of UniProt rescue candidates.")
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--rescue", type=Path, default=DEFAULT_RESCUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = pd.read_csv(args.normalized, sep="\t", dtype=str).fillna("")
    normalized["existing_sequence"] = normalized["existing_sequence"].astype(str).str.lower().eq("true")
    normalized["reviewed"] = normalized["reviewed"].astype(str).str.lower().eq("true")
    reference_all = normalized[normalized["existing_sequence"]].drop_duplicates("sequence").copy()
    reference_reviewed = reference_all[reference_all["reviewed"]].copy()

    rescue_rows = pd.read_csv(args.rescue, dtype=str).fillna("")
    aggregation = {
        "reaction_ids": ("reaction_id", lambda values: ";".join(sorted(set(values)))),
        "reaction_count": ("reaction_id", "nunique"),
        "rescue_roles": ("rescue_role", lambda values: ";".join(sorted(set(values)))),
        "terpene_types": ("terpene_type", lambda values: ";".join(sorted(set(values)))),
        "minimum_expanded_rank": ("expanded_rank", lambda values: pd.to_numeric(values).min()),
    }
    rescue_unique = (
        rescue_rows.groupby(
            [
                "candidate_id",
                "sequence",
                "length",
                "evidence_quality_tier",
                "domain_family",
                "protein_name",
                "organism_name",
                "protein_existence",
                "pfam_combination",
            ],
            as_index=False,
        )
        .agg(**aggregation)
    )

    annotated_reference_all = annotate(reference_all)
    annotated_reference_reviewed = annotate(reference_reviewed)
    annotated_rescue = annotate(rescue_unique)

    reference_summary = pd.concat(
        [
            summarize(annotated_reference_all, "existing_sequence_all"),
            summarize(annotated_reference_reviewed, "existing_sequence_reviewed"),
        ],
        ignore_index=True,
    )
    rescue_summary = summarize(annotated_rescue, "selected_rescue")
    summary = pd.concat([reference_summary, rescue_summary], ignore_index=True)

    reviewed_rates = (
        reference_summary[
            reference_summary["cohort"].eq("existing_sequence_reviewed")
        ]
        .set_index("pfam_architecture")["motif_contract_rate"]
        .to_dict()
    )
    reviewed_lengths = reference_summary[
        reference_summary["cohort"].eq("existing_sequence_reviewed")
    ].set_index("pfam_architecture")
    annotated_rescue["reviewed_reference_motif_rate"] = annotated_rescue["pfam_architecture"].map(reviewed_rates)
    annotated_rescue["below_reviewed_q05_length"] = [
        bool(
            family in reviewed_lengths.index
            and row["sequence_length"] < reviewed_lengths.loc[family, "q05_length"]
        )
        for family, (_, row) in zip(annotated_rescue["pfam_architecture"], annotated_rescue.iterrows())
    ]
    annotated_rescue["above_reviewed_q95_length"] = [
        bool(
            family in reviewed_lengths.index
            and row["sequence_length"] > reviewed_lengths.loc[family, "q95_length"]
        )
        for family, (_, row) in zip(annotated_rescue["pfam_architecture"], annotated_rescue.iterrows())
    ]
    annotated_rescue["priority_review"] = (
        annotated_rescue["high_confidence_sequence_risk"]
        | (
            ~annotated_rescue["expected_motif_contract_met"]
            & annotated_rescue["below_reviewed_q05_length"]
        )
    )

    annotated_rescue.to_csv(output_dir / "rescue_sequence_integrity.csv", index=False)
    annotated_rescue[annotated_rescue["priority_review"]].to_csv(
        output_dir / "priority_review_candidates.csv", index=False
    )
    summary.to_csv(output_dir / "family_reference_summary.csv", index=False)
    result_summary = {
        "reference_existing_unique_sequences": len(annotated_reference_all),
        "reference_reviewed_unique_sequences": len(annotated_reference_reviewed),
        "selected_rescue_unique_sequences": len(annotated_rescue),
        "motif_supported": int(annotated_rescue["sequence_integrity_status"].eq("motif_supported").sum()),
        "expected_motif_not_detected_annotation_only": int(annotated_rescue["sequence_integrity_status"].eq("expected_motif_not_detected_annotation_only").sum()),
        "architecture_length_risk": int(annotated_rescue["sequence_integrity_status"].eq("architecture_length_risk").sum()),
        "high_confidence_sequence_risk": int(annotated_rescue["high_confidence_sequence_risk"].sum()),
        "sequence_composition_risk": int(annotated_rescue["sequence_integrity_status"].eq("sequence_composition_risk").sum()),
        "invalid_residue": int(annotated_rescue["sequence_integrity_status"].eq("invalid_residue").sum()),
        "below_reviewed_q05_length": int(annotated_rescue["below_reviewed_q05_length"].sum()),
        "above_reviewed_q95_length": int(annotated_rescue["above_reviewed_q95_length"].sum()),
        "priority_review": int(annotated_rescue["priority_review"].sum()),
        "motif_contract_is_heuristic_not_activity_validation": True,
        "outputs": {
            "audit": str(output_dir / "rescue_sequence_integrity.csv"),
            "priority_review": str(output_dir / "priority_review_candidates.csv"),
            "family_reference": str(output_dir / "family_reference_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    print(json.dumps(result_summary, indent=2))
    print(rescue_summary.to_string(index=False))
    if result_summary["priority_review"]:
        print("\nPriority review candidates")
        print(
            annotated_rescue[annotated_rescue["priority_review"]][
                [
                    "candidate_id",
                    "domain_family",
                    "pfam_architecture",
                    "evidence_quality_tier",
                    "sequence_length",
                    "sequence_integrity_status",
                    "expected_motif_contract",
                    "ddxxd_count",
                    "nse_like_count",
                    "dxdd_count",
                    "dctae_like_count",
                    "qw_near_count",
                    "below_reviewed_q05_length",
                    "above_reviewed_q95_length",
                    "reaction_ids",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
