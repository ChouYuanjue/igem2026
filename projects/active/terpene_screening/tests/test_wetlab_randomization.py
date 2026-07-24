from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.randomize_wetlab_candidate_positions import (
    randomize_campaign,
)


ROWS = list("ABCDEFGH")


def canonical_manifest(n_reactions: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reaction_index in range(n_reactions):
        first_column = 2 * reaction_index + 1
        second_column = first_column + 1
        reaction_id = f"RXN_{reaction_index:02d}"
        candidate_wells = [f"{row}{first_column}" for row in ROWS] + [
            f"{row}{second_column}" for row in ROWS[:4]
        ]
        roles = ["exploitation"] * 6 + ["uncertainty"] * 3 + ["diversity"] * 3
        for candidate_index, (well, role) in enumerate(zip(candidate_wells, roles)):
            rows.append(
                {
                    "plate_id": "P1",
                    "well": well,
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "assay_role": "discovery_candidate",
                    "candidate_id": f"C_{reaction_index:02d}_{candidate_index:02d}",
                    "panel_role": role,
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
            rows.append(
                {
                    "plate_id": "P1",
                    "well": f"{row}{second_column}",
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "assay_role": assay_role,
                    "candidate_id": f"CTRL_{reaction_index:02d}" if assay_role.startswith("positive") else "",
                    "panel_role": "",
                }
            )
    return pd.DataFrame(rows)


def rescue_manifest(n_reactions: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = ["evidence_anchor", "homology_named", "named_predicted", "sequence_diversity"]
    for reaction_index in range(n_reactions):
        column = reaction_index + 1
        reaction_id = f"RESCUE_{reaction_index:02d}"
        for candidate_index, (row, role) in enumerate(zip(ROWS[:4], roles)):
            rows.append(
                {
                    "plate_id": "R1",
                    "well": f"{row}{column}",
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "assay_role": "uniprot_rescue_candidate",
                    "candidate_id": f"U_{reaction_index:02d}_{candidate_index:02d}",
                    "rescue_role": role,
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
            rows.append(
                {
                    "plate_id": "R1",
                    "well": f"{row}{column}",
                    "reaction_order": reaction_index + 1,
                    "reaction_id": reaction_id,
                    "assay_role": assay_role,
                    "candidate_id": f"RCTRL_{reaction_index:02d}" if assay_role.startswith("positive") else "",
                    "rescue_role": "",
                }
            )
    return pd.DataFrame(rows)


def assert_randomization_invariants(
    original: pd.DataFrame,
    randomized: pd.DataFrame,
    candidate_role: str,
) -> None:
    assert len(original) == len(randomized)
    assert not randomized.duplicated(["plate_id", "well"]).any()

    controls_original = original[~original["assay_role"].eq(candidate_role)].sort_values(
        ["plate_id", "reaction_id", "assay_role", "candidate_id"]
    )
    controls_randomized = randomized[
        ~randomized["assay_role"].eq(candidate_role)
    ].sort_values(["plate_id", "reaction_id", "assay_role", "candidate_id"])
    assert controls_original[["plate_id", "reaction_id", "assay_role", "candidate_id", "well"]].reset_index(
        drop=True
    ).equals(
        controls_randomized[["plate_id", "reaction_id", "assay_role", "candidate_id", "well"]].reset_index(
            drop=True
        )
    )

    for reaction_id, group in original[original["assay_role"].eq(candidate_role)].groupby(
        "reaction_id"
    ):
        observed = randomized[
            randomized["reaction_id"].eq(reaction_id)
            & randomized["assay_role"].eq(candidate_role)
        ]
        assert set(group["candidate_id"]) == set(observed["candidate_id"])


def test_canonical_randomization_is_deterministic_and_balanced():
    original = canonical_manifest()
    first, assignments, audit = randomize_campaign(
        original,
        "canonical_discovery",
        "discovery_candidate",
        "panel_role",
        20260723,
    )
    second, _, _ = randomize_campaign(
        original,
        "canonical_discovery",
        "discovery_candidate",
        "panel_role",
        20260723,
    )

    assert_randomization_invariants(original, first, "discovery_candidate")
    assert first[["candidate_id", "well"]].sort_values("candidate_id").reset_index(drop=True).equals(
        second[["candidate_id", "well"]].sort_values("candidate_id").reset_index(drop=True)
    )
    assert len(assignments) == 6 * 12

    before = audit[audit["layout"].eq("before")]
    after = audit[audit["layout"].eq("after")]
    assert after["normalized_slot_entropy"].mean() > before["normalized_slot_entropy"].mean()
    assert after["slot_count_range"].max() < before["slot_count_range"].max()
    assert after["occupied_slots"].min() == 12


def test_rescue_randomization_breaks_fixed_row_roles():
    original = rescue_manifest()
    randomized, assignments, audit = randomize_campaign(
        original,
        "uniprot_rescue",
        "uniprot_rescue_candidate",
        "rescue_role",
        20260723,
    )

    assert_randomization_invariants(original, randomized, "uniprot_rescue_candidate")
    assert len(assignments) == 12 * 4

    before = audit[audit["layout"].eq("before")]
    after = audit[audit["layout"].eq("after")]
    assert before["occupied_rows"].max() == 1
    assert after["occupied_rows"].min() == 4
    assert after["slot_count_range"].max() <= 1
    assert after["normalized_slot_entropy"].min() > 0.99
