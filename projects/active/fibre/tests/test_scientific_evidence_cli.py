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
    )
    with pytest.raises(ValueError):
        apply_runtime_scientific_evidence(
            np.asarray([0.0]),
            ["P1"],
            args=args,
            direction="reaction_to_enzyme",
            query_id="R1",
        )
