import pandas as pd

from projects.active.terpene_screening.select_cleanroom_directional_multifold import select_direction


def _row(candidate: str, fold: int, r2e: float, e2r: float) -> dict[str, object]:
    return {
        "candidate": candidate,
        "fold": fold,
        "r2e_hit_at_10": r2e,
        "r2e_mrr": r2e,
        "r2e_map": r2e,
        "r2e_macro_roc_auc": r2e,
        "r2e_ndcg_at_10": r2e,
        "e2r_hit_at_10": e2r,
        "e2r_mrr": e2r,
        "e2r_map": e2r,
        "e2r_macro_roc_auc": e2r,
        "e2r_ndcg_at_10": e2r,
    }


def test_directional_selector_can_choose_different_experts() -> None:
    frame = pd.DataFrame(
        [
            _row("r2e_expert", 0, 0.9, 0.2),
            _row("r2e_expert", 1, 0.8, 0.3),
            _row("e2r_expert", 0, 0.3, 0.9),
            _row("e2r_expert", 1, 0.2, 0.8),
        ]
    )
    _, r2e = select_direction(frame, "r2e")
    _, e2r = select_direction(frame, "e2r")
    assert r2e.iloc[0]["candidate"] == "r2e_expert"
    assert e2r.iloc[0]["candidate"] == "e2r_expert"
    assert bool(r2e.iloc[0]["selected"])
    assert bool(e2r.iloc[0]["selected"])


def test_directional_selector_requires_equal_fold_coverage() -> None:
    frame = pd.DataFrame(
        [
            _row("complete", 0, 0.9, 0.2),
            _row("complete", 1, 0.8, 0.3),
            _row("partial", 0, 0.3, 0.9),
        ]
    )
    try:
        select_direction(frame, "r2e")
    except ValueError as exc:
        assert "unequal fold counts" in str(exc)
    else:
        raise AssertionError("expected unequal fold coverage to be rejected")
