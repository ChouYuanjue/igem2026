"""Frozen query-disjoint gate, missing-evidence fallback and metric contracts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.e2r_query_gate import (
    CHANNELS,
    FEATURE_NAMES,
    filtered_target_ranks,
    reorder_head,
)
from reproducibility.bime_rank.scripts.fit_bridge_e2r_query_gate_v3 import (
    gate_relations,
)
from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import (
    balanced,
)


def test_frozen_reuses_complete_original_parent_with_protein_query_isolation():
    selected, report = gate_relations()
    final = pd.read_csv(
        ROOT / "results/bridge_gate_split_v2/evaluation_pairs.csv.gz",
        dtype=str,
    )
    assert len(selected) == 5216
    assert selected.source.value_counts().to_dict() == {
        "rhea_independent": 2948,
        "parent_dev": 2268,
    }
    assert not set(selected.protein_id) & set(final.protein_id)
    assert not set(zip(selected.protein_id, selected.reaction_id)) & set(
        zip(final.protein_id, final.reaction_id)
    )
    assert selected.groupby("protein_id").partition.nunique().eq(1).all()
    assert selected.novelty.nunique() == 4
    assert len(final) == 21505
    assert report["original_parent_edges"] == 23773
    assert report["external_pair_unique_but_evaluation_query_overlapping_rejected"] == 989


def test_missing_channels_preserve_broad_and_bounded_rerank():
    broad = np.asarray([5, 2, 4, 0, 3, 1], dtype=np.int64)
    lex = np.arange(6)
    core = np.asarray([0.3, -0.3, 0.2, 0.1, 0.0, -0.2])
    # Core ranking follows frozen broad order and labels are only for scoring.
    core[broad] = np.arange(6, 0, -1, dtype=float)
    v = np.arange(6, dtype=float)
    neutral = reorder_head(
        broad_order=broad, lexical=lex, core=core,
        functional=v, structure=-v, relation=v / 2,
        authority=np.zeros(3), k=4,
    )
    np.testing.assert_array_equal(neutral, broad)
    active = reorder_head(
        broad_order=broad, lexical=lex, core=core,
        functional=v, structure=-v, relation=v / 2,
        authority=np.array([1.0, 0.0, 1.0]), k=4,
    )
    assert set(active[:4]) == set(broad[:4])
    np.testing.assert_array_equal(active[4:], broad[4:])
    assert len(FEATURE_NAMES) == 41
    assert CHANNELS == ("functional", "structure", "relation")


def test_positive_rank_filter_and_monotone_group_balancing():
    order = np.array([0, 5, 2, 4, 3])
    target = np.array([5, 4])
    np.testing.assert_array_equal(
        filtered_target_ranks(order, target), [2, 3]
    )
    rows = []
    for novelty in ("both_seen", "protein_cold", "reaction_cold", "double_cold"):
        for difficulty in ("a", "b"):
            for rank in (2, 11, 102):
                rows.append({
                    "novelty": novelty, "difficulty_stratum": novelty + difficulty,
                    "rank": rank,
                })
    summary = balanced(pd.DataFrame(rows), "rank")
    for section in ("direct", "balanced"):
        m = summary[section]
        assert 0 <= m["hit3"] <= m["hit10"] <= m["hit100"] <= 1
        assert m["mrr"] > 0
