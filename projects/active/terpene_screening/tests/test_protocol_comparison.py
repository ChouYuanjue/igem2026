from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (
    masked_rank_metrics,
)
from projects.active.terpene_screening.evaluate_legacy_cage_double_cold import (
    parse_cells,
    recompute_local_cage_ranks,
)


def test_parse_cells_supports_all_and_explicit_cells():
    cells = parse_cells("all")
    assert len(cells) == 25
    assert cells[0] == (0, 0)
    assert cells[-1] == (4, 4)
    assert parse_cells("p0_r1,p4_r3") == [(0, 1), (4, 3)]
    with pytest.raises(ValueError, match="Invalid cell token"):
        parse_cells("0_1")
    with pytest.raises(ValueError, match="outside"):
        parse_cells("p5_r0")


def test_local_cage_ranks_depend_only_on_local_reservoir():
    frame = pd.DataFrame(
        {
            "uniprot_id": ["B", "A", "C", "D"],
            "cage_score": [0.9, 0.9, 0.2, np.nan],
        }
    )
    ranked = recompute_local_cage_ranks(frame)
    by_id = ranked.set_index("uniprot_id")["cage_rank_score"]
    assert by_id["A"] == pytest.approx(1.0)
    assert by_id["B"] == pytest.approx(0.5)
    assert by_id["C"] == pytest.approx(0.0)
    assert np.isnan(by_id["D"])

    subset = recompute_local_cage_ranks(frame[frame["uniprot_id"].isin(["B", "C"])]).set_index(
        "uniprot_id"
    )["cage_rank_score"]
    assert subset["B"] == pytest.approx(1.0)
    assert subset["C"] == pytest.approx(0.0)


def test_masked_rank_metrics_masks_known_associations_and_breaks_ties_by_id():
    scores = np.asarray([0.9, 0.8, 0.8, 0.7], dtype=np.float32)
    candidate_ids = ["known", "positive_b", "positive_a", "other"]
    metrics = masked_rank_metrics(
        scores,
        candidate_ids,
        positives={"positive_a"},
        masked={"known"},
        budgets=(1, 2, 3),
    )
    assert metrics["n_masked_known_positives"] == 1
    assert metrics["best_positive_rank"] == 1.0
    assert metrics["hit_at_1"] == 1
    assert metrics["hits_at_2"] == 1
    assert metrics["positive_recall_at_3"] == 1.0
