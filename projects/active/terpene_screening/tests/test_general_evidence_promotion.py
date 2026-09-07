from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.validate_general_evidence_promotion import evaluate_promotion


def _known(pass_retention: bool = True, ratio: float = 1.2):
    return {
        "retention_guard_passed": pass_retention,
        "retention_violations": [] if pass_retention else [{"metric": "hit_at_1"}],
        "unseen_directional_gain": {
            "reaction_to_enzyme": {"hit_at_10": {"ratio": ratio}},
            "enzyme_to_reaction": {"hit_at_10": {"ratio": ratio}},
        },
    }


def _current(delta: float = 0.0):
    return pd.DataFrame([
        {"direction": "reaction_to_enzyme", "evaluation_level": "query", "delta_mrr": delta, "delta_hit_at_10": 0.0},
        {"direction": "enzyme_to_reaction", "evaluation_level": "pair", "delta_mrr": 0.0, "delta_hit_at_10": 0.01},
    ])


def test_promotion_passes_only_when_both_retention_families_and_gain_pass():
    result = evaluate_promotion(_known(), _current())
    assert result["promotion_passed"] is True


def test_promotion_rejects_any_current_metric_drop_by_default():
    result = evaluate_promotion(_known(), _current(-1e-4))
    assert result["promotion_passed"] is False
    assert result["current_retention_violations"][0]["metric"] == "mrr"


def test_promotion_can_use_explicit_current_tolerance_but_not_weak_gain():
    result = evaluate_promotion(_known(ratio=1.01), _current(-1e-4), max_current_drop=1e-3, min_unseen_hit10_ratio=1.05)
    assert result["current_retention_passed"] is True
    assert result["broad_gain_passed"] is False
    assert result["promotion_passed"] is False
