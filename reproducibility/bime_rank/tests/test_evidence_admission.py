from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reproducibility.bime_rank.scripts.fit_scientific_evidence import parse_bool_series
from reproducibility.bime_rank.support.evidence_admission import (
    bootstrap_mean_interval,
    core_score_signature,
    fit_nonnegative_pairwise_logistic,
    ranking_deltas,
)


def test_useful_evidence_receives_positive_strength() -> None:
    core = np.zeros(6, dtype=np.float64)
    evidence = np.column_stack(
        [
            np.asarray([1.0, 0.8, 1.2, 0.7, 0.9, 1.1]),
            np.zeros(6, dtype=np.float64),
        ]
    )
    fit = fit_nonnegative_pairwise_logistic(core, evidence, l2=0.05)
    assert fit.coefficients[0] > 0
    assert abs(float(fit.coefficients[1])) < 1e-8


def test_harmful_evidence_is_projected_to_zero() -> None:
    core = np.full(8, 0.5, dtype=np.float64)
    harmful = -np.ones((8, 1), dtype=np.float64)
    fit = fit_nonnegative_pairwise_logistic(core, harmful, l2=0.01)
    assert abs(float(fit.coefficients[0])) < 1e-8


def test_query_equal_weights_are_supported() -> None:
    core = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    evidence = np.asarray([[1.0], [1.0], [-1.0]], dtype=np.float64)
    weight = np.asarray([0.25, 0.25, 0.5], dtype=np.float64)
    fit = fit_nonnegative_pairwise_logistic(
        core,
        evidence,
        sample_weight=weight,
        l2=0.1,
    )
    assert fit.success
    assert fit.coefficients.shape == (1,)


def test_csv_boolean_parser_does_not_treat_false_string_as_true() -> None:
    values = parse_bool_series(pd.Series(["true", "False", "1", "0", ""]))
    assert values.tolist() == [True, False, True, False, False]
    with pytest.raises(ValueError):
        parse_bool_series(pd.Series(["maybe"]))


def test_l1_joint_admission_does_not_reward_duplicate_channel() -> None:
    core = np.asarray([0.2, -0.1, 0.05, 0.0, 0.1, -0.2], dtype=np.float64)
    signal = np.asarray([1.0, 0.4, -0.2, 0.8, -0.5, 0.3], dtype=np.float64)
    single = fit_nonnegative_pairwise_logistic(
        core,
        signal[:, None],
        l1=0.05,
        l2=0.0,
    )
    duplicated = fit_nonnegative_pairwise_logistic(
        core,
        np.column_stack([signal, signal]),
        l1=0.05,
        l2=0.0,
    )
    assert np.isclose(
        float(duplicated.coefficients.sum()),
        float(single.coefficients[0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_core_signature_ignores_evidence_labels_and_row_order() -> None:
    base = pd.DataFrame(
        {
            "query_id": ["Q1", "Q1", "Q2"],
            "candidate_id": ["P1", "P2", "P3"],
            "core_score": [0.2, 0.1, -0.4],
            "evidence_score": [10.0, 20.0, 30.0],
            "label": [1, 0, 1],
        }
    )
    changed_metadata = base.iloc[::-1].copy()
    changed_metadata["evidence_score"] *= -7.0
    changed_metadata["label"] = 1 - changed_metadata["label"]
    assert core_score_signature(base) == core_score_signature(changed_metadata)

    changed_core = base.copy()
    changed_core.loc[0, "core_score"] += 1e-3
    assert core_score_signature(base) != core_score_signature(changed_core)


def test_bootstrap_interval_detects_consistently_positive_query_improvement() -> None:
    values = np.asarray([0.02, 0.03, 0.01, 0.04, 0.025, 0.018])
    interval = bootstrap_mean_interval(values, seed=7, replicates=1000)
    assert interval.n == len(values)
    assert interval.lower_95 > 0.0
    assert interval.upper_95 >= interval.mean
    assert interval.fraction_positive == 1.0


def test_bootstrap_interval_exposes_uncertain_query_improvement() -> None:
    values = np.asarray([0.03, -0.04, 0.02, -0.01, 0.01, -0.03])
    interval = bootstrap_mean_interval(values, seed=7, replicates=1000)
    assert interval.lower_95 < 0.0 < interval.upper_95


def test_ranking_deltas_follow_best_positive_rank() -> None:
    core = np.asarray([0.9, 0.8, 0.7, 0.6])
    fused = np.asarray([0.9, 1.0, 0.7, 0.6])
    labels = np.asarray([0, 1, 0, 0], dtype=np.int8)
    delta = ranking_deltas(
        core,
        fused,
        labels,
        candidate_ids=np.asarray(["A", "B", "C", "D"]),
        topk=(1, 3),
    )
    assert delta["reciprocal_rank"] == pytest.approx(0.5)
    assert delta["hit_at_1"] == 1.0
    assert delta["hit_at_3"] == 0.0


def test_ranking_deltas_expose_early_rank_regression() -> None:
    core = np.asarray([1.0, 0.9, 0.8, 0.7])
    fused = np.asarray([1.0, 0.6, 0.95, 0.9])
    labels = np.asarray([0, 1, 0, 0], dtype=np.int8)
    delta = ranking_deltas(
        core,
        fused,
        labels,
        candidate_ids=np.asarray(["A", "B", "C", "D"]),
        topk=(3,),
    )
    assert delta["reciprocal_rank"] < 0.0
    assert delta["hit_at_3"] == -1.0


def test_ranking_deltas_ties_ignore_input_row_order() -> None:
    core = np.asarray([1.0, 1.0, 0.5])
    fused = np.asarray([1.0, 1.0, 0.5])
    labels = np.asarray([0, 1, 0], dtype=np.int8)
    ids = np.asarray(["B", "A", "C"])
    first = ranking_deltas(
        core,
        fused,
        labels,
        candidate_ids=ids,
        topk=(1,),
    )
    perm = np.asarray([1, 0, 2])
    second = ranking_deltas(
        core[perm],
        fused[perm],
        labels[perm],
        candidate_ids=ids[perm],
        topk=(1,),
    )
    assert first == second
