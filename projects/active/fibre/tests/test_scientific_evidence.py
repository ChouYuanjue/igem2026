from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.active.fibre.runtime.scientific_evidence import (
    AdmittedEvidence,
    EvidenceDescriptor,
    EvidenceOutput,
    TabularEvidenceModule,
    apply_tabular_scientific_evidence,
    apply_tabular_scientific_evidence_bundle,
    fuse_admitted_evidence,
    load_admitted_evidence_bundle,
    load_admitted_evidence,
    load_evidence_descriptor,
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


def test_non_discriminative_query_only_evidence_is_neutral() -> None:
    out = EvidenceOutput(
        score=np.asarray([3.0, 3.0, 3.0]),
        available=np.asarray([True, True, True]),
    )
    calibrated = query_zscore(out)
    assert np.allclose(calibrated.score, 0.0)


def test_lower_is_better_scores_are_canonicalized() -> None:
    descriptor = EvidenceDescriptor(
        name="energy_like",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="lower means stronger support",
        availability_semantics="score exists",
        quality_semantics=None,
        provenance="frozen external tool",
        score_direction="lower_is_better",
    )
    module = TabularEvidenceModule(
        pd.DataFrame(
            {
                "direction": ["r2e", "r2e"],
                "query_id": ["R1", "R1"],
                "candidate_id": ["P1", "P2"],
                "score": [-8.0, -2.0],
                "available": [True, True],
            }
        ),
        descriptor,
    )
    output = module.score(
        direction="r2e",
        query_id="R1",
        candidate_ids=["P1", "P2"],
    )
    assert output.score[0] > output.score[1]


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


def test_admitted_table_can_modify_complete_candidate_vector(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.csv"
    pd.DataFrame(
        {
            "direction": ["r2e", "r2e"],
            "query_id": ["R1", "R1"],
            "candidate_id": ["P1", "P2"],
            "score": [0.0, 2.0],
            "available": [True, True],
        }
    ).to_csv(evidence, index=False)
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "fibre-local-scientific-evidence-admission-v1",
                "admission_policy": "pairwise-mrr95-topk-nonreg-crossfit-affine-v4",
                "baseline": {"id": "test-core"},
                "calibration": {"method": "fixed_global_affine_v1", "score_center": 0.0, "score_scale": 1.0},
                "descriptor": {
                    "name": "structure",
                    "kind": "structural",
                    "role": "rerank",
                    "directions": ["r2e"],
                    "score_semantics": "higher means stronger compatibility",
                    "availability_semantics": "structure score exists",
                    "quality_semantics": None,
                    "provenance": "frozen synthetic model",
                },
                "final": {
                    "strength": 1.0,
                    "quality_slope": 0.0,
                    "all_holdouts_improved_pairwise_log_loss": True,
                    "query_bootstrap_lower_95_positive": True,
                    "mrr_bootstrap_lower_95_positive": True,
                    "protected_topk_nonnegative": True,
                    "admitted": True,
                },
            }
        )
    )
    registration = load_admitted_evidence(admission)
    assert registration.descriptor.name == "structure"
    fused, contributions, registrations = apply_tabular_scientific_evidence(
        np.asarray([0.5, 0.4, 0.3]),
        ["P1", "P2", "P3"],
        direction="r2e",
        query_id="R1",
        evidence_csvs=[evidence],
        admission_jsons=[admission],
    )
    assert [item.descriptor.name for item in registrations] == ["structure"]
    assert contributions["structure"][2] == 0.0
    assert fused[1] > fused[0]
    assert fused[2] == 0.3


def test_declared_quality_requires_runtime_quality_column() -> None:
    descriptor = EvidenceDescriptor(
        name="mechanism",
        kind="mechanistic",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means stronger mechanism support",
        availability_semantics="mapping succeeded",
        quality_semantics="mapping confidence",
        provenance="frozen mechanism model",
    )
    with pytest.raises(ValueError):
        TabularEvidenceModule(
            pd.DataFrame(
                {
                    "direction": ["r2e"],
                    "query_id": ["R1"],
                    "candidate_id": ["P1"],
                    "score": [0.5],
                    "available": [True],
                }
            ),
            descriptor,
        )


def test_joint_bundle_applies_joint_coefficients(tmp_path: Path) -> None:
    structure_csv = tmp_path / "structure.csv"
    mechanism_csv = tmp_path / "mechanism.csv"
    pd.DataFrame(
        {
            "direction": ["r2e", "r2e"],
            "query_id": ["R1", "R1"],
            "candidate_id": ["P1", "P2"],
            "score": [0.0, 1.0],
            "available": [True, True],
        }
    ).to_csv(structure_csv, index=False)
    pd.DataFrame(
        {
            "direction": ["r2e", "r2e"],
            "query_id": ["R1", "R1"],
            "candidate_id": ["P1", "P2"],
            "score": [0.0, 2.0],
            "available": [True, True],
            "quality": [0.5, 0.5],
        }
    ).to_csv(mechanism_csv, index=False)
    bundle_json = tmp_path / "bundle.json"
    bundle_json.write_text(
        json.dumps(
            {
                "schema": "fibre-scientific-evidence-bundle-admission-v1",
                "admission_policy": "pairwise-mrr95-topk-nonreg-crossfit-affine-v4",
                "bundle_id": "joint-test",
                "baseline": {"id": "test-core"},
                "members": [
                    {
                        "descriptor": {
                            "name": "structure",
                            "kind": "structural",
                            "role": "rerank",
                            "directions": ["r2e"],
                            "score_semantics": "higher is better",
                            "availability_semantics": "structure exists",
                            "quality_semantics": None,
                            "provenance": "frozen structure model",
                        },
                        "calibration": {"method": "fixed_global_affine_v1", "score_center": 0.5, "score_scale": 0.5},
                        "strength": 0.5,
                        "quality_slope": 0.0,
                    },
                    {
                        "descriptor": {
                            "name": "mechanism",
                            "kind": "mechanistic",
                            "role": "rerank",
                            "directions": ["r2e"],
                            "score_semantics": "higher is better",
                            "availability_semantics": "mapping exists",
                            "quality_semantics": "mapping confidence",
                            "provenance": "frozen mechanism model",
                        },
                        "calibration": {"method": "fixed_global_affine_v1", "score_center": 1.0, "score_scale": 1.0},
                        "strength": 0.25,
                        "quality_slope": 0.1,
                    },
                ],
                "final": {
                    "all_holdouts_improved_pairwise_log_loss": True,
                    "query_bootstrap_lower_95_positive": True,
                    "mrr_bootstrap_lower_95_positive": True,
                    "protected_topk_nonnegative": True,
                    "admitted": True,
                },
            }
        )
    )
    bundle = load_admitted_evidence_bundle(bundle_json)
    assert bundle.bundle_id == "joint-test"
    fused, contributions, loaded = apply_tabular_scientific_evidence_bundle(
        np.asarray([0.4, 0.3]),
        ["P1", "P2"],
        direction="r2e",
        query_id="R1",
        evidence_csvs=[structure_csv, mechanism_csv],
        bundle_json=bundle_json,
    )
    assert loaded.bundle_id == "joint-test"
    assert set(contributions) == {"structure", "mechanism"}
    assert fused[1] > fused[0]


def test_joint_bundle_descriptor_does_not_require_solo_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synergy_only.json"
    path.write_text(
        json.dumps(
            {
                "descriptor": {
                    "name": "synergy_only",
                    "kind": "mechanistic",
                    "role": "rerank",
                    "directions": ["r2e"],
                    "score_semantics": "higher means more support",
                    "availability_semantics": "mechanistic score exists",
                    "quality_semantics": None,
                    "provenance": "cross-fitted local model",
                },
                "final": {
                    "strength": 0.0,
                    "quality_slope": 0.0,
                    "admitted": False,
                },
            }
        )
    )
    assert load_evidence_descriptor(path).name == "synergy_only"
    with pytest.raises(ValueError):
        load_admitted_evidence(path)


def test_fixed_affine_contribution_is_candidate_subset_invariant() -> None:
    descriptor = EvidenceDescriptor(
        name="stable_scale",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means stronger support",
        availability_semantics="score exists",
        quality_semantics=None,
        provenance="synthetic frozen model",
    )
    registration = AdmittedEvidence(
        descriptor=descriptor,
        strength=0.5,
        score_center=2.0,
        score_scale=2.0,
    )
    _, full = fuse_admitted_evidence(
        np.zeros(3),
        [
            EvidenceOutput(
                score=np.asarray([1.0, 3.0, 5.0]),
                available=np.asarray([True, True, True]),
            )
        ],
        [registration],
    )
    _, subset = fuse_admitted_evidence(
        np.zeros(2),
        [
            EvidenceOutput(
                score=np.asarray([1.0, 5.0]),
                available=np.asarray([True, True]),
            )
        ],
        [registration],
    )
    assert np.allclose(
        subset["stable_scale"],
        full["stable_scale"][[0, 2]],
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_admitted_evidence_rejects_nonfinite_reliability(value: float) -> None:
    descriptor = EvidenceDescriptor(
        name="bad",
        kind="structural",
        role="rerank",
        directions=("r2e",),
        score_semantics="higher means more support",
        availability_semantics="score exists",
        quality_semantics=None,
        provenance="synthetic",
    )
    with pytest.raises(ValueError, match="finite"):
        AdmittedEvidence(
            descriptor=descriptor,
            strength=value,
            quality_slope=0.0,
        ).validate()
