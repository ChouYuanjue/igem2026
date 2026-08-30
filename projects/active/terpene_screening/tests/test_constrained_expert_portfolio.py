import pandas as pd

from projects.active.terpene_screening.evaluate_constrained_expert_portfolio import choose_joint_method


def _frame() -> pd.DataFrame:
    rows = []
    values = {
        "direct:legacy": (0.40, 0.30, 0.50),
        "good": (0.45, 0.32, 0.55),
        "tradeoff": (0.60, 0.29, 0.70),
    }
    for method, vals in values.items():
        for query in range(4):
            rows.append({"method": method, "query_id": str(query), "reciprocal_rank": vals[0], "hit_at_1": vals[1], "hit_at_20": vals[2]})
    return pd.DataFrame(rows)


def test_joint_guard_rejects_single_metric_tradeoff() -> None:
    frame = _frame()
    selected, diag = choose_joint_method(
        frame,
        ["direct:legacy", "good", "tradeoff"],
        ["reciprocal_rank", "hit_at_1", "hit_at_20"],
    )
    assert selected == "good"
    assert diag["min_guard_delta"] >= 0


def test_backbone_is_safe_fallback() -> None:
    frame = _frame()
    frame.loc[frame["method"].eq("good"), "hit_at_1"] = 0.1
    selected, _ = choose_joint_method(
        frame,
        ["direct:legacy", "good", "tradeoff"],
        ["reciprocal_rank", "hit_at_1", "hit_at_20"],
    )
    assert selected == "direct:legacy"
