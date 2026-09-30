from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.active.fibre.runtime.cli import (
    annotate_runtime_scientific_evidence,
    apply_runtime_scientific_evidence,
    scientific_evidence_requested,
)


def _admission(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "fibre-local-scientific-evidence-admission-v1",
                "admission_policy": "all-heldout-folds-and-query-bootstrap95-v1",
                "baseline": {"id": "test-core"},
                "descriptor": {
                    "name": "local_structure",
                    "kind": "structural",
                    "role": "rerank",
                    "directions": ["r2e"],
                    "score_semantics": "higher means stronger structural support",
                    "availability_semantics": "local structure score exists",
                    "quality_semantics": None,
                    "provenance": "synthetic frozen local model",
                },
                "final": {
                    "strength": 1.0,
                    "quality_slope": 0.0,
                    "all_holdouts_improved_pairwise_log_loss": True,
                    "query_bootstrap_lower_95_positive": True,
                    "admitted": True,
                },
            }
        )
    )


def test_runtime_cli_glue_applies_before_topk_and_audits_contribution(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.csv"
    admission = tmp_path / "admission.json"
    pd.DataFrame(
        {
            "direction": ["r2e", "r2e"],
            "query_id": ["R1", "R1"],
            "candidate_id": ["P1", "P2"],
            "score": [0.0, 2.0],
            "available": [True, True],
        }
    ).to_csv(evidence, index=False)
    _admission(admission)
    args = Namespace(
        scientific_evidence_csv=[evidence],
        scientific_evidence_admission=[admission],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="test-core",
    )
    assert scientific_evidence_requested(args)
    fused, contributions, audit = apply_runtime_scientific_evidence(
        np.asarray([0.5, 0.4, 0.3]),
        ["P1", "P2", "P3"],
        args=args,
        direction="reaction_to_enzyme",
        query_id="R1",
    )
    assert fused[1] > fused[0]
    result = pd.DataFrame(
        {
            "rank": [1, 2],
            "candidate_id": ["P2", "P1"],
            "score": [fused[1], fused[0]],
        }
    )
    annotated = annotate_runtime_scientific_evidence(
        result,
        ["P1", "P2", "P3"],
        contributions,
        audit,
    )
    assert annotated["scientific_evidence_applied"].all()
    assert annotated["scientific_evidence_modules"].iloc[0] == "local_structure"
    assert annotated["scientific_evidence:local_structure"].iloc[0] > 0


def test_runtime_cli_glue_rejects_unpaired_evidence_arguments(tmp_path: Path) -> None:
    args = Namespace(
        scientific_evidence_csv=[tmp_path / "evidence.csv"],
        scientific_evidence_admission=[],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="test-core",
    )
    with pytest.raises(ValueError):
        apply_runtime_scientific_evidence(
            np.asarray([0.0]),
            ["P1"],
            args=args,
            direction="reaction_to_enzyme",
            query_id="R1",
        )


def test_runtime_cli_rejects_two_independently_admitted_modules(
    tmp_path: Path,
) -> None:
    evidence_a = tmp_path / "a.csv"
    evidence_b = tmp_path / "b.csv"
    admission_a = tmp_path / "a.json"
    admission_b = tmp_path / "b.json"
    pd.DataFrame(
        {
            "direction": ["r2e"],
            "query_id": ["R1"],
            "candidate_id": ["P1"],
            "score": [1.0],
            "available": [True],
        }
    ).to_csv(evidence_a, index=False)
    pd.DataFrame(
        {
            "direction": ["r2e"],
            "query_id": ["R1"],
            "candidate_id": ["P1"],
            "score": [2.0],
            "available": [True],
        }
    ).to_csv(evidence_b, index=False)
    _admission(admission_a)
    _admission(admission_b)
    args = Namespace(
        scientific_evidence_csv=[evidence_a, evidence_b],
        scientific_evidence_admission=[admission_a, admission_b],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="test-core",
    )
    with pytest.raises(ValueError, match="jointly cross-fitted"):
        apply_runtime_scientific_evidence(
            np.asarray([0.0]),
            ["P1"],
            args=args,
            direction="reaction_to_enzyme",
            query_id="R1",
        )


def test_runtime_cli_rejects_wrong_core_baseline(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    admission = tmp_path / "admission.json"
    pd.DataFrame(
        {
            "direction": ["r2e"],
            "query_id": ["R1"],
            "candidate_id": ["P1"],
            "score": [1.0],
            "available": [True],
        }
    ).to_csv(evidence, index=False)
    _admission(admission)
    args = Namespace(
        scientific_evidence_csv=[evidence],
        scientific_evidence_admission=[admission],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="different-core",
    )
    with pytest.raises(ValueError, match="baseline does not match"):
        apply_runtime_scientific_evidence(
            np.asarray([0.0]),
            ["P1"],
            args=args,
            direction="reaction_to_enzyme",
            query_id="R1",
        )


def test_runtime_cli_rejects_baseline_metadata_without_evidence() -> None:
    args = Namespace(
        scientific_evidence_csv=[],
        scientific_evidence_admission=[],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="test-core",
    )
    assert scientific_evidence_requested(args)
    with pytest.raises(ValueError, match="without an evidence CSV"):
        apply_runtime_scientific_evidence(
            np.asarray([0.0]),
            ["P1"],
            args=args,
            direction="reaction_to_enzyme",
            query_id="R1",
        )


def test_runtime_cli_glue_supports_e2r_direction(tmp_path: Path) -> None:
    evidence = tmp_path / "e2r.csv"
    admission = tmp_path / "e2r.json"
    pd.DataFrame(
        {
            "direction": ["e2r", "e2r"],
            "query_id": ["P1", "P1"],
            "candidate_id": ["R1", "R2"],
            "score": [0.0, 2.0],
            "available": [True, True],
        }
    ).to_csv(evidence, index=False)
    admission.write_text(
        json.dumps(
            {
                "schema": "fibre-local-scientific-evidence-admission-v1",
                "admission_policy": "all-heldout-folds-and-query-bootstrap95-v1",
                "baseline": {"id": "e2r-core"},
                "descriptor": {
                    "name": "reaction_context",
                    "kind": "experimental_context",
                    "role": "rerank",
                    "directions": ["e2r"],
                    "score_semantics": "higher means stronger support",
                    "availability_semantics": "context score exists",
                    "quality_semantics": None,
                    "provenance": "synthetic frozen context model",
                },
                "final": {
                    "strength": 1.0,
                    "quality_slope": 0.0,
                    "all_holdouts_improved_pairwise_log_loss": True,
                    "query_bootstrap_lower_95_positive": True,
                    "admitted": True,
                },
            }
        )
    )
    args = Namespace(
        scientific_evidence_csv=[evidence],
        scientific_evidence_admission=[admission],
        scientific_evidence_bundle=None,
        scientific_evidence_baseline_id="e2r-core",
    )
    fused, contributions, audit = apply_runtime_scientific_evidence(
        np.asarray([0.5, 0.4]),
        ["R1", "R2"],
        args=args,
        direction="enzyme_to_reaction",
        query_id="P1",
    )
    assert fused[1] > fused[0]
    assert audit["modules"] == ["reaction_context"]
    assert set(contributions) == {"reaction_context"}
