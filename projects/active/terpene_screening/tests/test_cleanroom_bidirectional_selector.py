from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.select_cleanroom_bidirectional_multifold import METRICS, select


def row(candidate: str, fold: int, r2e: float, e2r: float) -> dict[str, object]:
    values: dict[str, object] = {"candidate": candidate, "fold": fold}
    for metric in METRICS:
        values[metric] = r2e if metric.startswith("r2e_") else e2r
    return values


def test_balanced_selector_does_not_allow_r2e_only_winner_to_dominate():
    frame = pd.DataFrame(
        [
            row("balanced", fold, 0.8, 0.8)
            for fold in range(3)
        ]
        + [
            row("r2e_only", fold, 1.0, 0.2)
            for fold in range(3)
        ]
    )
    _, aggregate = select(frame)
    assert aggregate.iloc[0]["candidate"] == "balanced"
    assert aggregate.iloc[0]["mean_r2e_joint_percentile"] < 1.0
    assert aggregate.iloc[0]["mean_e2r_joint_percentile"] == 1.0


def test_balanced_selector_requires_equal_fold_counts():
    frame = pd.DataFrame(
        [row("a", 0, 0.5, 0.5), row("a", 1, 0.5, 0.5), row("b", 0, 0.6, 0.6)]
    )
    import pytest
    with pytest.raises(ValueError, match="unequal fold counts"):
        select(frame)
