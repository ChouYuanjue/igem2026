from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projects.active.terpene_screening.bootstrap_general_known_recovery import (
    compare_bootstrap,
    paired_bootstrap_delta,
)


def test_paired_bootstrap_detects_uniform_positive_gain():
    base = np.zeros(20)
    candidate = np.ones(20)
    result = paired_bootstrap_delta(base, candidate, samples=2000, seed=7, chunk_size=200)
    assert result["delta"] == 1.0
    assert result["bootstrap_ci_low"] == 1.0
    assert result["bootstrap_probability_nonpositive"] == 0.0


def test_compare_bootstrap_is_query_paired():
    rows = []
    for q in range(10):
        rows.append({
            "direction": "reaction_to_enzyme",
            "stratum": "unseen_to_historical_training",
            "query_id": f"q{q}",
            "n_positives": 1,
            "reciprocal_rank": 0.1,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "hit_at_10": 0,
            "hit_at_20": 0,
        })
    base = pd.DataFrame(rows)
    candidate = base.copy()
    candidate["reciprocal_rank"] += 0.05
    candidate["hit_at_10"] = 1
    candidate["hit_at_20"] = 1
    result = compare_bootstrap(base, candidate, stratum="unseen_to_historical_training", samples=1000, seed=11)
    assert set(result["metric"]) == {"reciprocal_rank", "hit_at_10", "hit_at_20"}
    assert (result["delta"] > 0).all()


def test_paired_bootstrap_rejects_unpaired_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap_delta(np.zeros(3), np.zeros(2), samples=10, seed=1)
