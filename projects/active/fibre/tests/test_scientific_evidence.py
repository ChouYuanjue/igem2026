from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projects.active.fibre.runtime.scientific_evidence import (
    AdmittedEvidence,
    EvidenceDescriptor,
    EvidenceOutput,
    TabularEvidenceModule,
    fuse_admitted_evidence,
    query_zscore,
)


def test_query_zscore_uses_supported_candidates_only() -> None:
    out = EvidenceOutput(
        score=np.asarray([1.0, 3.0, 1000.0]),
        available=np.asarray([True, True, False]),
    )
    calibrated = query_zscore(out)
    assert np.allclose(calibrated.score[:2], [-1.0, 1.0])
    assert calibrated.score[2] == 0.0
    assert not calibrated.available[2]


def test_missing_scores_may_be_nonfinite_but_supported_scores_may_not() -> None:
    EvidenceOutput(
        score=np.asarray([0.2, np.nan]),
        available=np.asarray([True, False]),
    ).validate(2)
    with pytest.raises(ValueError):
        EvidenceOutput(
            score=np.asarray([np.nan, 0.2]),
            available=np.asarray([True, False]),
        ).validate(2)


def test_descriptor_requires_provenance_and_availability_semantics() -> None:
    good = EvidenceDescriptor(
        name="structure",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means stronger reaction-protein structural compatibility",
        availability_semantics="both reaction and protein structure assets exist",
        quality_semantics="optional structure confidence",
        provenance="frozen CLIPZyme checkpoint and internal clean-development calibration",
    )
    good.validate()

    with pytest.raises(ValueError):
        EvidenceDescriptor(
            name="structure",
            kind="structural",
            role="rerank",
            directions=("r2e",),
            score_semantics="higher is better",
            availability_semantics="",
            quality_semantics=None,
            provenance="frozen source",
        ).validate()


def test_descriptor_rejects_unknown_runtime_kind_and_role() -> None:
    with pytest.raises(ValueError):
        EvidenceDescriptor(
            name="bad",
            kind="unknown",  # type: ignore[arg-type]
            role="rerank",
            directions=("r2e",),
            score_semantics="higher is better",
            availability_semantics="always",
            quality_semantics=None,
            provenance="synthetic",
        ).validate()
    with pytest.raises(ValueError):
        EvidenceDescriptor(
            name="bad",
            kind="structural",
            role="unknown",  # type: ignore[arg-type]
            directions=("r2e",),
            score_semantics="higher is better",
            availability_semantics="always",
            quality_semantics=None,
            provenance="synthetic",
        ).validate()


def test_multiple_admitted_evidence_channels_add_without_competing() -> None:
    structure = EvidenceDescriptor(
        name="structure",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means more compatible",
        availability_semantics="structure is available",
        quality_semantics=None,
        provenance="frozen structure model",
    )
    context = EvidenceDescriptor(
        name="known_positive",
        kind="experimental_context",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means closer to a supplied positive enzyme",
        availability_semantics="at least one known positive was supplied",
        quality_semantics=None,
        provenance="user-supplied positive with frozen sequence encoder",
    )
    core = np.asarray([0.0, 0.0, 0.0])
    outputs = [
        EvidenceOutput(
            score=np.asarray([0.0, 1.0, 2.0]),
            available=np.asarray([True, True, True]),
        ),
        EvidenceOutput(
            score=np.asarray([2.0, 1.0, 0.0]),
            available=np.asarray([True, True, True]),
        ),
    ]
    fused, contributions = fuse_admitted_evidence(
        core,
        outputs,
        [
            AdmittedEvidence(structure, strength=0.5),
            AdmittedEvidence(context, strength=1.0),
        ],
    )
    assert np.allclose(
        fused,
        contributions["structure"] + contributions["known_positive"],
    )


def test_missing_module_does_not_change_other_contribution() -> None:
    descriptor = EvidenceDescriptor(
        name="structure",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means more compatible",
        availability_semantics="structure is available",
        quality_semantics=None,
        provenance="frozen structure model",
    )
    core = np.asarray([0.2, 0.4, 0.1])
    registration = AdmittedEvidence(descriptor, strength=0.5)
    present = EvidenceOutput(
        score=np.asarray([0.0, 1.0, 2.0]),
        available=np.asarray([True, True, True]),
    )
    missing = EvidenceOutput(
        score=np.asarray([10.0, -4.0, 7.0]),
        available=np.asarray([False, False, False]),
    )
    fused_present, _ = fuse_admitted_evidence(core, [present], [registration])
    fused_missing, _ = fuse_admitted_evidence(core, [missing], [registration])
    assert np.allclose(fused_missing, core)
    assert not np.allclose(fused_present, core)


def test_tabular_module_is_missing_neutral_and_can_retrieve() -> None:
    descriptor = EvidenceDescriptor(
        name="local_assay_model",
        kind="experimental_context",
        role="retrieve_and_rerank",
        directions=("r2e",),
        score_semantics="higher means stronger local catalytic support",
        availability_semantics="local model emitted a score for the pair",
        quality_semantics="cross-validated local confidence in [0,1]",
        provenance="laboratory-owned cross-fitted model",
    )
    module = TabularEvidenceModule(
        pd.DataFrame(
            {
                "direction": ["r2e", "r2e", "r2e"],
                "query_id": ["R1", "R1", "R1"],
                "candidate_id": ["P2", "P1", "P3"],
                "score": [0.7, 0.9, np.nan],
                "available": ["true", "1", "False"],
                "quality": [0.8, 0.6, np.nan],
            }
        ),
        descriptor,
    )
    output = module.score(
        direction="r2e",
        query_id="R1",
        candidate_ids=["P1", "P2", "P3"],
    )
    assert np.allclose(output.score, [0.9, 0.7, 0.0])
    assert output.available.tolist() == [True, True, False]
    assert np.allclose(output.quality, [0.6, 0.8, 0.0])
    assert module.retrieve(direction="r2e", query_id="R1", top_k=3) == ["P1", "P2"]
