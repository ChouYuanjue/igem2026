from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (
    family_compatibility,
    pfam_architecture,
    reaction_architecture_compatibility,
)
from projects.active.terpene_screening.audit_uniprot_rescue_sequence_integrity import (
    conservative_sequence_risk,
)
from projects.active.terpene_screening.rank_uniprot_rescue import (
    load_quota_evidence,
    resolve_rescue_slots,
)


def test_validated_default_rescue_quotas_are_fixed():
    assert resolve_rescue_slots(3, None) == 0
    assert resolve_rescue_slots(10, None) == 1
    assert resolve_rescue_slots(20, None) == 2


def test_unvalidated_top_k_requires_explicit_quota():
    with pytest.raises(ValueError, match="No validated default"):
        resolve_rescue_slots(15, None)
    assert resolve_rescue_slots(15, 2) == 2


def test_invalid_rescue_quota_is_rejected():
    with pytest.raises(ValueError):
        resolve_rescue_slots(10, -1)
    with pytest.raises(ValueError):
        resolve_rescue_slots(10, 11)


def test_tps_family_compatibility_contract():
    assert family_compatibility("mono", "bacterial_classI") == "compatible"
    assert family_compatibility("di", "plant_like_classI_II") == "compatible"
    assert family_compatibility("tri", "triterpene_cyclase") == "compatible"
    assert family_compatibility("tri", "bacterial_classI") == "family_mismatch"
    assert family_compatibility("sesq", "triterpene_cyclase") == "family_mismatch"
    assert family_compatibility("psy", "bacterial_classI") == "extended_pathway_uncertain"


def test_pfam_architecture_separates_complete_enzymes_and_fragments():
    assert pfam_architecture("PF13243;PF13249") == "osc_full"
    assert pfam_architecture("PF13243") == "classII_cyclase_single_domain"
    assert pfam_architecture("PF13249") == "osc_domain_fragment"
    assert pfam_architecture("PF01397;PF03936") == "plant_tps_full"
    assert pfam_architecture("PF01397") == "plant_tps_single_PF01397"
    assert pfam_architecture("PF03936") == "plant_tps_single_PF03936"
    assert pfam_architecture("PF19086") == "bacterial_classI"


def test_reaction_architecture_compatibility_rejects_fragments_and_unsupported_family():
    assert (
        reaction_architecture_compatibility(
            "tri", "(S)-2,3-epoxysqualene", "friedelin", "2", "PF13243;PF13249"
        )
        == "compatible"
    )
    assert (
        reaction_architecture_compatibility(
            "tri", "(S)-2,3-epoxysqualene", "friedelin", "2", "PF13249"
        )
        == "architecture_mismatch"
    )
    assert (
        reaction_architecture_compatibility(
            "tri", "presqualene diphosphate", "presqualene alcohol", "1", "PF13243;PF13249"
        )
        == "unsupported_expansion_family"
    )


def test_conservative_sequence_risk_uses_complete_architecture_length():
    balanced = "ACDEFGHIKLMNPQRSTVWY"
    safe, reason = conservative_sequence_risk(balanced * 16, "PF19086")
    assert not safe
    assert reason == ""
    risky, reason = conservative_sequence_risk((balanced * 10)[:199], "PF19086")
    assert risky
    assert "architecture_length" in reason
    safe, reason = conservative_sequence_risk((balanced * 33)[:650], "PF13243")
    assert not safe
    risky, reason = conservative_sequence_risk((balanced * 28)[:550], "PF13243;PF13249")
    assert risky
    assert "architecture_length" in reason


def test_quota_evidence_is_loaded_from_stress_result(tmp_path: Path):
    pd.DataFrame(
        [
            {
                "budget": 20,
                "rescue_slots": 2,
                "canonical_slots": 18,
                "n_queries": 237,
                "baseline_canonical_hits": 43,
                "quota_hits": 42,
                "hit_probability": 42 / 237,
                "hit_retention_fraction": 42 / 43,
                "hits_lost": 1,
            }
        ]
    ).to_csv(tmp_path / "rescue_slot_retention.csv", index=False)
    evidence = load_quota_evidence(tmp_path, 20, 2)
    assert evidence["strict_double_cold_status"] == "validated_unlabelled_decoy_stress_test"
    assert evidence["strict_double_cold_baseline_hits"] == 43
    assert evidence["strict_double_cold_quota_hits"] == 42
    assert evidence["strict_double_cold_hit_retention_fraction"] == pytest.approx(42 / 43)
