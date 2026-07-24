from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.build_wetlab_discovery_panels import (
    select_balanced_campaign,
)
from projects.active.terpene_screening.manage_wetlab_feedback import (
    classify_discovery_rows,
    reaction_qc,
)


def test_wetlab_qc_and_labels_do_not_turn_failed_expression_into_negative():
    rows = [
        {
            "reaction_id": "RXN1",
            "assay_role": "positive_control_primary",
            "candidate_id": "POS",
            "target_product_detected": "true",
            "product_identity_confidence": "0.95",
            "technical_issue": "false",
            "expression_status": "not_measured",
            "soluble_expression": "",
            "assay_signal": "10",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "positive_control_replicate",
            "candidate_id": "POS",
            "target_product_detected": "true",
            "product_identity_confidence": "0.92",
            "technical_issue": "false",
            "expression_status": "not_measured",
            "soluble_expression": "",
            "assay_signal": "9",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "empty_vector_negative",
            "candidate_id": "EMPTY_VECTOR",
            "target_product_detected": "false",
            "product_identity_confidence": "0",
            "technical_issue": "false",
            "expression_status": "not_measured",
            "soluble_expression": "",
            "assay_signal": "1",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "substrate_process_blank",
            "candidate_id": "NO_ENZYME",
            "target_product_detected": "false",
            "product_identity_confidence": "0",
            "technical_issue": "false",
            "expression_status": "not_measured",
            "soluble_expression": "",
            "assay_signal": "1",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "discovery_candidate",
            "candidate_id": "HIT",
            "target_product_detected": "true",
            "product_identity_confidence": "0.9",
            "technical_issue": "false",
            "expression_status": "adequate",
            "soluble_expression": "true",
            "assay_signal": "8",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "discovery_candidate",
            "candidate_id": "NEG",
            "target_product_detected": "false",
            "product_identity_confidence": "0",
            "technical_issue": "false",
            "expression_status": "adequate",
            "soluble_expression": "true",
            "assay_signal": "1",
            "background_signal": "1",
        },
        {
            "reaction_id": "RXN1",
            "assay_role": "discovery_candidate",
            "candidate_id": "FAILED",
            "target_product_detected": "false",
            "product_identity_confidence": "0",
            "technical_issue": "false",
            "expression_status": "failed",
            "soluble_expression": "false",
            "assay_signal": "",
            "background_signal": "",
        },
    ]
    frame = pd.DataFrame(rows)
    qc = reaction_qc(frame, identity_threshold=0.8)
    assert qc["reaction_qc_pass"] is True
    labels = classify_discovery_rows(frame, {"RXN1": True}, identity_threshold=0.8)
    observed = dict(zip(labels["candidate_id"], labels["feedback_label"]))
    assert observed == {
        "HIT": "confirmed_positive",
        "NEG": "expression_qualified_negative",
        "FAILED": "inconclusive",
    }

    failed_qc_labels = classify_discovery_rows(frame, {"RXN1": False}, identity_threshold=0.8)
    assert set(failed_qc_labels["feedback_label"]) == {"inconclusive"}


def test_balanced_campaign_covers_every_core_type():
    core_types = {"mono", "sesq", "di", "sester", "tri", "sesquar"}
    rows = []
    for index, terpene_type in enumerate(sorted(core_types)):
        for replicate in range(2):
            rows.append(
                {
                    "reaction_id": f"R{index}_{replicate}",
                    "terpene_type": terpene_type,
                    "tps_class": "2" if index < 2 else "1",
                    "substrate_name": f"S{index}",
                    "positive_control_id": f"P{index}_{replicate}",
                    "campaign_priority_score": 1.0 - 0.05 * replicate - 0.01 * index,
                }
            )
    frame = pd.DataFrame(rows)
    selected = select_balanced_campaign(
        frame,
        count=8,
        core_types=core_types,
        class2_minimum=2,
    )
    assert len(selected) == 8
    assert set(selected["terpene_type"]) == core_types
    assert selected["tps_class"].eq("2").sum() >= 2
