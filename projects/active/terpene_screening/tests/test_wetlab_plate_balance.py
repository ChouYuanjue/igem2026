from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.balance_wetlab_reactions_across_plates import (
    CampaignSpec,
    attach_rescue_architecture,
    feature_vectors,
    optimize_assignment,
    reaction_features,
)


ROWS = list("ABCDEFGH")


def build_rescue_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, str]] = []
    architectures = [
        ("PF13243;PF13249", "tri", "2", 760),
        ("PF13243;PF13249", "tri", "2", 720),
        ("PF19086", "sesq", "1", 330),
        ("PF19086", "sesq", "1", 350),
    ]
    for reaction_index, (pfam, terpene_type, tps_class, length) in enumerate(architectures):
        plate_id = "R1" if reaction_index < 2 else "R2"
        reaction_id = f"RXN_{reaction_index}"
        column = reaction_index % 2 + 1
        for candidate_index, row in enumerate(ROWS[:4]):
            candidate_id = f"U_{reaction_index}_{candidate_index}"
            manifest_rows.append(
                {
                    "plate_id": plate_id,
                    "well": f"{row}{column}",
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "terpene_type": terpene_type,
                    "tps_class": tps_class,
                    "substrate_name": f"substrate_{reaction_index}",
                    "product_name": f"product_{reaction_index}",
                    "assay_role": "uniprot_rescue_candidate",
                    "candidate_id": candidate_id,
                    "rescue_role": "evidence_anchor",
                    "evidence_quality_tier": "B_experimental_or_transcript_named",
                    "sequence": "M" * (length + candidate_index),
                }
            )
            metadata_rows.append(
                {
                    "reaction_id": reaction_id,
                    "candidate_id": candidate_id,
                    "pfam_combination": pfam,
                }
            )
        for row, assay_role in zip(
            ROWS[4:],
            [
                "positive_control_primary",
                "positive_control_replicate",
                "empty_vector_negative",
                "substrate_process_blank",
            ],
        ):
            manifest_rows.append(
                {
                    "plate_id": plate_id,
                    "well": f"{row}{column}",
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "terpene_type": terpene_type,
                    "tps_class": tps_class,
                    "substrate_name": f"substrate_{reaction_index}",
                    "product_name": f"product_{reaction_index}",
                    "assay_role": assay_role,
                    "candidate_id": f"CTRL_{reaction_index}" if assay_role.startswith("positive") else "",
                    "rescue_role": "",
                    "evidence_quality_tier": "",
                    "sequence": "M" * 400 if assay_role.startswith("positive") else "",
                }
            )
    return pd.DataFrame(manifest_rows), pd.DataFrame(metadata_rows)


def test_rescue_architecture_is_attached_and_used(tmp_path: Path):
    manifest, metadata = build_rescue_manifest()
    metadata_path = tmp_path / "selected.csv"
    metadata.to_csv(metadata_path, index=False)

    enriched = attach_rescue_architecture(manifest, metadata_path)
    candidates = enriched[enriched["assay_role"].eq("uniprot_rescue_candidate")]
    assert candidates["pfam_architecture"].ne("").all()
    assert candidates["pfam_architecture"].value_counts().to_dict() == {
        "osc_full": 8,
        "bacterial_classI": 8,
    }

    spec = CampaignSpec(
        scope="synthetic_rescue",
        candidate_role="uniprot_rescue_candidate",
        role_column="rescue_role",
        reactions_per_plate=2,
        columns_per_reaction=1,
    )
    reactions = reaction_features(enriched, spec)
    names = [name for name, _, _ in feature_vectors(reactions, spec)]
    assert "osc_full_count" in names
    assert "bacterial_classI_count" in names

    assigned, audit, summary = optimize_assignment(
        reactions,
        ["R1", "R2"],
        spec,
        seed=20260723,
    )
    assert summary["solver_success"]
    assert assigned.groupby("balanced_plate_id").size().to_dict() == {"R1": 2, "R2": 2}
    after = audit[audit["layout"].eq("after")]
    assert after["osc_full_total"].tolist() == [4, 4]
    assert after["tps_class_counts"].tolist() == ['{"1": 1, "2": 1}'] * 2


def test_missing_rescue_architecture_metadata_fails(tmp_path: Path):
    manifest, metadata = build_rescue_manifest()
    metadata = metadata.iloc[:-1]
    metadata_path = tmp_path / "incomplete.csv"
    metadata.to_csv(metadata_path, index=False)
    with pytest.raises(ValueError, match="Missing Pfam architecture"):
        attach_rescue_architecture(manifest, metadata_path)
