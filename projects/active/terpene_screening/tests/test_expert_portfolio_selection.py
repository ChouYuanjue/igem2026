from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.evaluate_expert_portfolio_selection import choose_method, summarize


def test_choose_method_uses_primary_then_mrr_tie_break():
    frame = pd.DataFrame([
        {"method":"a","hit_at_10":1.0,"reciprocal_rank":0.5},
        {"method":"b","hit_at_10":1.0,"reciprocal_rank":0.8},
    ])
    assert choose_method(frame,["a","b"],"hit_at_10") == "b"


def test_summary_reports_selection_stability_and_delta():
    records = pd.DataFrame([
        {"scenario":"s","direction":"d","n_seed":0,"selection_metric":"hit_at_10","baseline":"direct:legacy","selected_method":"x","selected_test_score":.8,"baseline_test_score":.7,"delta":.1},
        {"scenario":"s","direction":"d","n_seed":0,"selection_metric":"hit_at_10","baseline":"direct:legacy","selected_method":"x","selected_test_score":.7,"baseline_test_score":.7,"delta":0.0},
    ])
    row=summarize(records).iloc[0]
    assert row.modal_method == 'x'
    assert row.non_degradation_probability == 1.0
    assert abs(row.mean_delta-.05)<1e-12
