from projects.active.terpene_screening.run_internal_top2000_bounded_residual_v2 import (
    select_residual_scale,
    scale_slug,
)


PRIMARY = ["mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50"]


def _metrics(base: float, delta: float) -> dict[str, float]:
    return {metric: base + delta for metric in PRIMARY}


def test_scale_slug_is_path_safe() -> None:
    assert scale_slug(0.0) == "scale_0"
    assert scale_slug(0.003) == "scale_0p003"


def test_selector_promotes_robust_nonzero_scale() -> None:
    folds = {
        fold: {
            0.0: _metrics(0.20 + fold * 0.01, 0.0),
            0.01: _metrics(0.20 + fold * 0.01, 0.01),
            0.03: _metrics(0.20 + fold * 0.01, 0.005),
        }
        for fold in range(3)
    }
    selected = select_residual_scale(
        folds,
        scales=[0.0, 0.01, 0.03],
        primary_metrics=PRIMARY,
    )
    assert selected["pair_residual_expert_promoted"] is True
    assert selected["selected_residual_scale"] == 0.01


def test_selector_falls_back_when_any_primary_mean_regresses() -> None:
    folds = {
        fold: {
            0.0: _metrics(0.20, 0.0),
            0.01: _metrics(0.20, 0.01),
        }
        for fold in range(3)
    }
    # Make one primary metric worse on average, so the strict Pareto gate must reject.
    for fold in range(3):
        folds[fold][0.01]["hit_at_50"] = 0.19
    selected = select_residual_scale(
        folds,
        scales=[0.0, 0.01],
        primary_metrics=PRIMARY,
    )
    assert selected["pair_residual_expert_promoted"] is False
    assert selected["selected_residual_scale"] == 0.0


def test_selector_rejects_large_single_fold_core_regression() -> None:
    folds = {
        fold: {
            0.0: _metrics(0.20, 0.0),
            0.01: _metrics(0.20, 0.01),
        }
        for fold in range(3)
    }
    # Mean remains positive, but one fold violates the 2% MRR robustness guard.
    folds[0][0.01]["mrr"] = 0.19
    folds[1][0.01]["mrr"] = 0.23
    folds[2][0.01]["mrr"] = 0.23
    selected = select_residual_scale(
        folds,
        scales=[0.0, 0.01],
        primary_metrics=PRIMARY,
    )
    assert selected["pair_residual_expert_promoted"] is False
    assert selected["selected_residual_scale"] == 0.0
