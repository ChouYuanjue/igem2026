from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import (
    AuthorPairwiseNative, DualTowerNative, balanced_selection_score, normalize_reaction_bag, query_metrics, split_is_dev,
)

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "projects/active/terpene_screening/ENZGFM_NATIVE_SAME_SUPPORT_CATALYST_V1.json"


def test_protocol_freezes_unique_authoritative_baseline_and_no_test_selection():
    d = json.loads(PROTOCOL.read_text())
    assert d["authoritative_external_baseline"]["name"] == "EnzGFM-1.5B"
    assert d["data_isolation"]["native_test_performance_for_candidate_selection"] is False
    assert d["data_isolation"]["post_native_test_reveal_retuning_on_same_test"] is False
    assert set(d["candidate_family"]) == {"dual_tower", "author_pairwise"}
    assert d["selection_rule"]["candidate_addition_after_dev_reveal"] is False
    assert d["selection_rule"]["alternative_candidate_native_test_scoring"] is False
    assert d["final_reveal"]["native_support_expected"] == {"positive_pairs": 8739, "unique_sequences": 8734, "unique_reaction_bags": 1573}


def test_hash_split_and_reaction_normalization_are_deterministic():
    seq = "MABCDEFGHIKLMNPQRSTVWY"
    assert split_is_dev(seq) == split_is_dev("  mabcdefghiklmnpqrstvwy*  ")
    assert normalize_reaction_bag("O.C*.CC") == normalize_reaction_bag("CC.O.CC")


def test_candidate_forward_shapes():
    torch.manual_seed(1)
    r = torch.randn(3, 4096)
    p = torch.randn(3, 2048)
    dual = DualTowerNative(dropout=0.0).eval()
    assert dual(r, p).shape == (3,)
    pair = AuthorPairwiseNative().eval()
    assert pair(r, p).shape == (3,)


def test_query_metrics_standard_and_author_semantics():
    scores = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype=np.float32)
    labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    m = query_metrics(scores, labels)
    assert m["n_queries"] == 2
    assert np.isclose(m["map"], 1.0)
    assert np.isclose(m["best_mrr"], 1.0)
    assert np.isclose(m["hit_at_1"], 1.0)
    # Author code averages reciprocal ranks over every positive rather than best-positive reciprocal rank.
    assert m["author_avg_positive_rr"] < m["best_mrr"]
    assert np.isclose(balanced_selection_score({"map": 0.5}, {"map": 1.0}), 2.0 / 3.0)


def test_final_evaluator_contains_guarded_selection_and_no_ndcg_paper_delta():
    text = (ROOT / "projects/active/terpene_screening/evaluate_enzgfm_native_same_support_v1.py").read_text()
    assert "selected_candidate" in text
    assert "refusing to overwrite frozen evaluation" in text
    assert '"e2r_map"' in text and '"r2e_map"' in text
    assert "ndcg" not in text.split('"direct_same_support_paper_deltas"', 1)[1].split('}', 1)[0].lower()
