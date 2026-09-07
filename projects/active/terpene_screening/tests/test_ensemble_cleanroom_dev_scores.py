from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.active.terpene_screening.ensemble_cleanroom_dev_scores import ensemble_frames


def write(path: Path, scores: list[float], labels: list[int] | None = None) -> None:
    pd.DataFrame({
        "reaction_id": ["R1", "R1", "R2"],
        "protein_id": ["P1", "P2", "P3"],
        "label": labels or [1, 0, 1],
        "score": scores,
    }).to_csv(path, index=False)


def test_ensemble_is_exact_mean_on_identical_reservoir(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, [0.8, 0.2, 0.4]); write(b, [0.6, 0.4, 0.8])
    out = ensemble_frames([a, b])
    assert out["score"].tolist() == pytest.approx([0.7, 0.3, 0.6])
    assert out["label"].tolist() == [1, 0, 1]


def test_ensemble_rejects_label_disagreement(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, [0.8, 0.2, 0.4]); write(b, [0.6, 0.4, 0.8], [0, 0, 1])
    with pytest.raises(ValueError, match="disagree on labels"):
        ensemble_frames([a, b])


def test_ensemble_rejects_reservoir_mismatch(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, [0.8, 0.2, 0.4]); write(b, [0.6, 0.4, 0.8])
    frame = pd.read_csv(b).iloc[:-1]
    frame.to_csv(b, index=False)
    with pytest.raises(ValueError, match="identical query-candidate reservoirs"):
        ensemble_frames([a, b])
