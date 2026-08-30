from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.evaluate_domain_expert_router import aggregate_routed, route_query_metrics


def _rows(delta: float) -> pd.DataFrame:
    rows = []
    for direction, query in [("reaction_to_enzyme", "R_TPS"), ("reaction_to_enzyme", "R_GEN"), ("enzyme_to_reaction", "P_TPS"), ("enzyme_to_reaction", "P_GEN")]:
        row = {"direction": direction, "stratum": "all_known", "query_id": query, "n_positives": 1}
        for metric in ["reciprocal_rank", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "hit_at_20"]:
            row[metric] = 0.5 + delta
        rows.append(row)
    return pd.DataFrame(rows)


def test_router_keeps_legacy_for_tps_and_general_elsewhere():
    legacy = _rows(0.0)
    general = _rows(0.2)
    routed = route_query_metrics(
        legacy,
        general,
        protein_ids={"P_TPS"},
        reaction_ids={"R_TPS"},
    )
    by_id = routed.set_index("query_id")
    assert by_id.loc["R_TPS", "expert"] == "tps_legacy"
    assert by_id.loc["P_TPS", "hit_at_10"] == 0.5
    assert by_id.loc["R_GEN", "expert"] == "general"
    assert by_id.loc["P_GEN", "hit_at_10"] == 0.7


def test_router_aggregate_includes_combined_micro_all_known():
    routed = route_query_metrics(
        _rows(0.0), _rows(0.2), protein_ids={"P_TPS"}, reaction_ids={"R_TPS"}
    )
    metrics = aggregate_routed(routed)
    both = metrics[(metrics.direction == "both_micro") & (metrics.stratum == "all_known")].iloc[0]
    assert both.n_queries == 4
    assert both.tps_legacy_fraction == 0.5
    assert abs(both.routed_hit_at_10 - 0.6) < 1e-9
    assert abs(both.legacy_hit_at_10 - 0.5) < 1e-9
