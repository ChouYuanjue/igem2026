from __future__ import annotations

import pandas as pd
import pytest

from projects.active.terpene_screening.bootstrap_enzyme405_neural import bootstrap_means, native_query_metrics


def test_native_query_metrics_respects_stable_score_order_and_top5_ef_floor() -> None:
    frame = pd.DataFrame({
        "reaction_id": ["R1"] * 6,
        "protein_id": [f"P{i}" for i in range(6)],
        "label": [0, 1, 0, 0, 0, 0],
        "neural_score": [1.0, 1.0, 0.5, 0.4, 0.3, 0.2],
    })
    q = native_query_metrics(frame).iloc[0]
    assert q.native_sr1 == 0
    assert q.native_sr3 == 1
    assert q.native_ef1 == pytest.approx(100.0)


def test_bootstrap_point_estimate_matches_observed_mean() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0, 1.0], "b": [1.0, 3.0, 5.0]})
    out = bootstrap_means(frame, ["a", "b"], samples=2000, seed=7)
    assert out["a"]["estimate"] == pytest.approx(2 / 3)
    assert out["b"]["estimate"] == pytest.approx(3.0)
    assert out["a"]["ci95_low"] <= out["a"]["estimate"] <= out["a"]["ci95_high"]
