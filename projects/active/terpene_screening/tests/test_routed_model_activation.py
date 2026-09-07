import pandas as pd

from projects.active.terpene_screening.build_routed_model_activation import activation_table, wilson_lower_bound


def test_wilson_lower_bound_requires_reliable_majority() -> None:
    assert wilson_lower_bound(70, 100) > 0.5
    assert wilson_lower_bound(54, 100) < 0.5


def _inputs(deltas: list[float], successes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["reciprocal_rank", "hit_at_1", "hit_at_20"]
    summary = pd.DataFrame([
        {"scenario": "s", "direction": "r2e", "n_seed": 5, "baseline": "direct:legacy", "metric": metric, "mean_delta": delta}
        for metric, delta in zip(metrics, deltas)
    ])
    joint = pd.DataFrame([
        {"scenario": "s", "direction": "r2e", "n_seed": 5, "baseline": "direct:legacy", "repeat": i,
         "selected_method": "expert" if i < 80 else "direct:legacy", "all_metrics_non_degraded": i < successes}
        for i in range(100)
    ])
    return summary, joint


def test_activation_requires_all_metric_guard_and_reliable_joint_non_degradation() -> None:
    metric, joint = _inputs([0.05, 0.01, 0.02], 70)
    row = activation_table(metric, joint).iloc[0]
    assert bool(row.active)
    assert row.selected_route == "expert"

    metric, joint = _inputs([0.05, -0.001, 0.02], 70)
    assert not bool(activation_table(metric, joint).iloc[0].active)

    metric, joint = _inputs([0.05, 0.01, 0.02], 54)
    assert not bool(activation_table(metric, joint).iloc[0].active)


def test_zero_gain_backbone_only_cell_does_not_activate() -> None:
    metric, joint = _inputs([0.0, 0.0, 0.0], 100)
    row = activation_table(metric, joint).iloc[0]
    assert not bool(row.active)
    assert row.selected_route == "direct:legacy"
