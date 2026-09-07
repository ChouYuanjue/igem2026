from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "select_functional_prototype_residual_v1.py"
spec = importlib.util.spec_from_file_location("functional_selector", MODULE_PATH)
mod = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)


def _record(primary, all_delta, fold_deltas):
    return {
        "pooled_primary_delta": primary,
        "pooled_all_delta": all_delta,
        "folds": [{"fold": i, "primary_delta": d} for i, d in enumerate(fold_deltas)],
    }


def test_gate_passes_only_frozen_conditions():
    primary = {"mrr": .01, "map": .01, "ndcg_at_10": 0., "macro_roc_auc": 0., "hit_at_10": .01, "hit_at_20": 0., "hit_at_50": 0., "median_best_positive_rank": -1.}
    all_delta = {"mrr": 0., "map": 0., "ndcg_at_10": 0., "macro_roc_auc": -9., "hit_at_10": 0., "hit_at_20": 0., "hit_at_50": 0., "median_best_positive_rank": 9.}
    folds = [
        {"mrr": .01, "map": .01},
        {"mrr": .001, "map": .002},
        {"mrr": -.004, "map": -.003},
    ]
    ok, reasons = mod.candidate_passes(_record(primary, all_delta, folds))
    assert ok and not reasons


def test_gate_rejects_primary_and_global_regression():
    primary = {"mrr": 0., "map": .01, "ndcg_at_10": -.001, "macro_roc_auc": 0., "hit_at_10": 0., "hit_at_20": 0., "hit_at_50": 0., "median_best_positive_rank": 0.}
    all_delta = {"mrr": -.0001, "map": 0., "ndcg_at_10": 0., "macro_roc_auc": 0., "hit_at_10": 0., "hit_at_20": 0., "hit_at_50": 0., "median_best_positive_rank": 0.}
    folds = [{"mrr": -.006, "map": .001}] * 3
    ok, reasons = mod.candidate_passes(_record(primary, all_delta, folds))
    assert not ok
    assert "primary_mrr_not_strictly_improved" in reasons
    assert "primary_ndcg_at_10_regressed" in reasons
    assert "primary_median_best_positive_rank_not_decreased" in reasons
    assert "all_mrr_regressed" in reasons
    assert "fold0_primary_mrr_regressed_gt_0p005" in reasons
    assert "primary_mrr_and_map_not_improved_in_2_of_3_folds" in reasons


def test_tie_break_matches_preregistered_order():
    base = {"pooled_primary_delta": {"hit_at_10": .1, "mrr": .02}, "pooled_all_delta": {"mrr": .003}}
    a = {**base, "residual_scale": .1, "confidence_margin": .02}
    b = {**base, "residual_scale": .05, "confidence_margin": 0.}
    c = {**base, "residual_scale": .05, "confidence_margin": .05}
    assert sorted([a, b, c], key=mod._candidate_sort_key) == [c, b, a]


def test_selector_source_does_not_reference_forbidden_external_benchmarks():
    text = MODULE_PATH.read_text().lower()
    for forbidden in ["enzyme-405", "orphan-335", "reactzyme native", "rhea128", "temporal_post2020"]:
        assert forbidden not in text
