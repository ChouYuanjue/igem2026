from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reproducibility.bime_rank.scripts.fit_scientific_evidence import parse_bool_series
from reproducibility.bime_rank.support.evidence_admission import (
    fit_nonnegative_pairwise_logistic,
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
