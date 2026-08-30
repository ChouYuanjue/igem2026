from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import (
    evaluate_enzymecage_native_r2e,
    evaluate_scores,
    exposure_audit,
    map_official_pairs,
)


def _universe(tmp_path: Path) -> Path:
    root = tmp_path / "universe"
    root.mkdir()
    pd.DataFrame([
        {"protein_id": "P1", "canonical_accession": "U1", "aliases": "OLD1"},
        {"protein_id": "P2", "canonical_accession": "U2", "aliases": ""},
        {"protein_id": "P3", "canonical_accession": "U3", "aliases": ""},
    ]).to_csv(root / "protein_metadata.csv", index=False)
    pd.DataFrame([
        {"reaction_id": "RHEA:1", "reaction_smiles": "CC>>CO", "source_layer": "x"},
        {"reaction_id": "RHEA:2", "reaction_smiles": "CCC>>CCO", "source_layer": "x"},
    ]).to_csv(root / "reactions.csv", index=False)
    return root


def test_mapping_uses_aliases_and_canonical_reaction_smiles(tmp_path: Path):
    universe = _universe(tmp_path)
    official = pd.DataFrame([
        {"UniprotID": "U1", "CANO_RXN_SMILES": "CC>>CO", "Label": 1},
        {"UniprotID": "P2", "CANO_RXN_SMILES": "CCC>>CCO", "Label": 0},
        {"UniprotID": "MISSING", "CANO_RXN_SMILES": "CC>>CO", "Label": 0},
    ])
    mapped, audit = map_official_pairs(official, universe)
    assert mapped.loc[0, "protein_id"] == "P1"
    assert mapped.loc[0, "reaction_id"] == "RHEA:1"
    assert mapped.loc[1, "reaction_id"] == "RHEA:2"
    assert bool(mapped.loc[2, "mapped"]) is False
    assert audit["mapped_rows"] == 2
    assert audit["mapped_positive_rows"] == 1


def test_strict_query_audit_uses_positive_exposure_not_negative_candidates(tmp_path: Path):
    universe = _universe(tmp_path)
    base = tmp_path / "base"; base.mkdir()
    # P2 is a seen negative candidate. The positive P3/RHEA:2 is still strict-cold.
    pd.DataFrame([
        {"Entry": "U1", "rhea_id": "RHEA:1", "source": "train"},
        {"Entry": "U2", "rhea_id": "RHEA:1", "source": "train"},
    ]).to_csv(base / "training_pairs.csv", index=False)
    mapped = pd.DataFrame([
        {"mapped": True, "label": 1, "protein_id": "P3", "reaction_id": "RHEA:2"},
        {"mapped": True, "label": 0, "protein_id": "P2", "reaction_id": "RHEA:2"},
    ])
    positives, audit, strict = exposure_audit(mapped, base, universe)
    assert len(positives) == 1
    assert strict == {"RHEA:2"}
    assert audit["base_protein_seen_rows"] == 0
    assert audit["base_reaction_seen_rows"] == 0


def test_native_sr_and_dcg_match_shared_when_no_ties():
    frame = pd.DataFrame([
        {"reaction_id": "R1", "protein_id": "P1", "label": 1, "score": 0.9},
        {"reaction_id": "R1", "protein_id": "P2", "label": 0, "score": 0.8},
        {"reaction_id": "R1", "protein_id": "P3", "label": 0, "score": 0.7},
        {"reaction_id": "R2", "protein_id": "P1", "label": 0, "score": 0.9},
        {"reaction_id": "R2", "protein_id": "P2", "label": 1, "score": 0.8},
        {"reaction_id": "R2", "protein_id": "P3", "label": 0, "score": 0.7},
    ])
    metrics, _ = evaluate_scores(frame, "score")
    native = metrics["enzymecage_native_r2e"]
    shared = metrics["reaction_to_enzyme"]
    assert native["top1_sr"] == shared["hit_at_1"] == pytest.approx(0.5)
    assert native["top3_sr"] == shared["hit_at_3"] == pytest.approx(1.0)
    assert native["top10_dcg"] == shared["top10_dcg"]
    # EnzymeCAGE's EF convention intentionally differs from standard panel-size EF
    # whenever its minimum Top-5 floor activates.
    assert native["top1_percent_ef"] != shared["top1_percent_ef"]
    assert native["top2_percent_ef"] != shared["top2_percent_ef"]


def test_native_ef_keeps_requested_percent_under_top5_floor():
    frame = pd.DataFrame([
        {"reaction_id": "R1", "protein_id": f"P{i}", "label": int(i == 0), "score": 10 - i}
        for i in range(10)
    ])
    native = evaluate_enzymecage_native_r2e(frame, "score")
    # The author code selects at least five rows but still uses total_active*0.01
    # and total_active*0.02 as the random expectation.
    assert native["top1_percent_ef"] == pytest.approx(100.0)
    assert native["top2_percent_ef"] == pytest.approx(50.0)


def test_native_score_ties_preserve_input_order():
    frame = pd.DataFrame([
        {"reaction_id": "R1", "protein_id": "Z_NEG", "label": 0, "score": 1.0},
        {"reaction_id": "R1", "protein_id": "A_POS", "label": 1, "score": 1.0},
        {"reaction_id": "R1", "protein_id": "M_NEG", "label": 0, "score": 0.5},
    ])
    native = evaluate_enzymecage_native_r2e(frame, "score")
    # Alphabetical tie-breaking would put A_POS first; EnzymeCAGE's stable sort
    # preserves the file order, so the positive is rank 2.
    assert native["top1_sr"] == 0.0
    assert native["top3_sr"] == 1.0
