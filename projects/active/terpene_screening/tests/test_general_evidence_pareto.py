from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.validate_general_evidence_pareto import evaluate_pareto


def _summary(r2e_h10=1.5, r2e_mrr=1.2, r2e_h20=1.1, e2r_h10=1.05):
    return {
        "unseen_directional_gain": {
            "reaction_to_enzyme": {
                "hit_at_10": {"ratio": r2e_h10},
                "reciprocal_rank": {"ratio": r2e_mrr},
                "hit_at_20": {"ratio": r2e_h20},
            },
            "enzyme_to_reaction": {"hit_at_10": {"ratio": e2r_h10}},
        }
    }


def _known(delta=-0.008):
    return pd.DataFrame([
        {"direction": "reaction_to_enzyme", "stratum": "historical_training_pair", "delta_mrr": delta, "delta_hit_at_10": 0.01},
        {"direction": "enzyme_to_reaction", "stratum": "project_catalog", "delta_mrr": 0.0, "delta_hit_at_10": 0.0},
        {"direction": "reaction_to_enzyme", "stratum": "unseen_to_historical_training", "delta_mrr": -0.5, "delta_hit_at_10": -0.5},
    ])


def _current(delta=-0.004):
    return pd.DataFrame([
        {"direction": "reaction_to_enzyme", "evaluation_level": "query", "delta_mrr": delta, "delta_hit_at_10": 0.0},
        {"direction": "enzyme_to_reaction", "evaluation_level": "pair", "delta_mrr": 0.0, "delta_hit_at_10": 0.0},
    ])


def test_pareto_accepts_small_tradeoff_for_material_gain():
    result = evaluate_pareto(_summary(), _known(), _current())
    assert result["pareto_passed"] is True
    assert result["retention"]["current_worst_delta"] == -0.004
    assert result["retention"]["known_worst_delta"] == -0.008


def test_pareto_rejects_large_retention_drop_even_with_gain():
    result = evaluate_pareto(_summary(), _known(-0.02), _current())
    assert result["pareto_passed"] is False
    assert result["retention"]["known_violations"]


def test_pareto_rejects_weak_primary_gain_even_when_retention_is_perfect():
    result = evaluate_pareto(_summary(r2e_h10=1.2), _known(0.0), _current(0.0))
    assert result["pareto_passed"] is False
    assert any(item["metric"] == "primary_hit10_ratio" for item in result["gain_violations"])
