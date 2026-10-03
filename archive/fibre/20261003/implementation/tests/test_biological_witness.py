from projects.active.fibre.evidence.biological_witness import (
    _alignment_identity,
    _family_specific_observables,
    _reaction_center_edit_summary,
)


def test_alignment_identity_exposes_global_and_local_views():
    x = _alignment_identity("ACDEFGHIK", "ACDEYGHIK")
    assert 0.0 <= x["global_sequence_identity"] <= 1.0
    assert 0.0 <= x["local_sequence_identity"] <= 1.0
    assert x["global_alignment_length"] >= 9
    assert x["local_alignment_length"] > 0


def test_tps_motifs_are_not_emitted_for_generic_protein():
    x = _family_specific_observables(
        {"domain_family": "", "pfam": ""},
        "MDDXXDUMMYSEQ".replace("X", "A"),
    )
    assert x["scope"] == "not_applicable_without_tps_family_annotation"
    assert "tps_motif_observables" not in x


def test_tps_motifs_are_emitted_only_for_annotated_tps_family():
    x = _family_specific_observables(
        {"domain_family": "bacterial_classI", "pfam": "PF19086"},
        "MDDDADAAAAAAAAAAAAAANDLAAA",
    )
    assert x["scope"] == "class_I_terpene_synthase_observables"
    assert x["domain_family"] == "bacterial_classI"
    assert "motifs" in x


def test_non_tps_domain_name_does_not_enable_tps_motifs():
    x = _family_specific_observables(
        {"domain_family": "generic_oxidoreductase", "pfam": "PF00348"},
        "MDDDADAAAAAAAAAAAAAANDLAAA",
    )
    assert x["scope"] == "not_applicable_without_tps_family_annotation"
    assert "tps_motif_observables" not in x


def test_reaction_center_edit_summary_reports_bond_order_change():
    x = _reaction_center_edit_summary("[CH3:1][OH:2]>>[CH2:1]=[O:2]")
    assert x["available"] is True
    assert len(x["bond_changes"]) == 1
    b = x["bond_changes"][0]
    assert b["before"] == "single"
    assert b["after"] == "double"
    assert "C-O" in b["signature"]
