from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from projects.active.terpene_screening.dual_kernel_runtime import (
    DualKernelAssets,
    align_reaction_scores,
    protein_affinity,
    score_batch,
    score_query,
)
from projects.active.terpene_screening.rank_open_world import (
    DEFAULT_E2R_DUAL_TOWER_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    should_use_e2r_top20_dual_kernel,
)
from projects.active.terpene_screening.validate_dual_kernel_deployment import (
    references_packaged_asset,
)


def toy_assets() -> DualKernelAssets:
    protein_features = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    protein_features /= np.linalg.norm(protein_features, axis=1, keepdims=True)
    support = csr_matrix(
        np.asarray(
            [
                [1.0, 0.8, 0.0, 0.1],
                [0.0, 0.2, 1.0, 0.7],
                [0.2, 0.4, 0.3, 0.5],
            ],
            dtype=np.float32,
        )
    )
    return DualKernelAssets(
        root=Path("/tmp/toy-dual-kernel"),
        reaction_ids=("R1", "R2", "R3"),
        protein_ids=("P0", "P1", "P2", "P3"),
        train_protein_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
        protein_features=protein_features,
        reaction_protein_support=support,
        metadata={"protein_k": 1, "temperature": 0.03},
    )


def test_dual_kernel_excludes_exact_query_self_neighbor():
    assets = toy_assets()
    selected_without_exclusion, _ = protein_affinity(
        assets.protein_features[0], assets
    )
    selected_with_exclusion, weights = protein_affinity(
        assets.protein_features[0], assets, query_id="P0"
    )
    assert selected_without_exclusion.tolist() == [0]
    assert selected_with_exclusion.tolist() == [1]
    np.testing.assert_allclose(weights, np.asarray([1.0], dtype=np.float32))


def test_dual_kernel_reaction_scores_align_by_identifier():
    assets = toy_assets()
    scores = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    aligned = align_reaction_scores(scores, assets, ["R3", "R1", "R2"])
    np.testing.assert_allclose(aligned, np.asarray([3.0, 1.0, 2.0]))


def test_dual_kernel_single_and_batch_scores_are_identical():
    assets = toy_assets()
    query_features = np.stack(
        [assets.protein_features[0], assets.protein_features[2]]
    )
    query_ids = ["P0", "P2"]
    batch = score_batch(query_features, assets, query_ids=query_ids)
    expected = np.stack(
        [
            score_query(query_features[0], assets, query_id="P0"),
            score_query(query_features[1], assets, query_id="P2"),
        ]
    )
    np.testing.assert_allclose(batch, expected, rtol=0, atol=0)
    np.testing.assert_allclose(batch[0], np.asarray([0.8, 0.2, 0.4]))
    np.testing.assert_allclose(batch[1], np.asarray([0.1, 0.7, 0.5]))


def eligible(**overrides: object) -> bool:
    arguments: dict[str, object] = {
        "ranking_objective": "top20",
        "is_current_enzyme": False,
        "has_seed_reactions": False,
        "requested_retrieval_mode": "auto",
        "model_dir_override": None,
        "dual_tower_dir": DEFAULT_E2R_DUAL_TOWER_DIR,
        "has_temporary_external_reactions": False,
        "registered_reactions_csv": DEFAULT_REGISTERED_REACTIONS,
    }
    arguments.update(overrides)
    return should_use_e2r_top20_dual_kernel(**arguments)  # type: ignore[arg-type]


def test_dual_kernel_route_boundary_is_strict():
    assert eligible()
    assert not eligible(ranking_objective="top3")
    assert not eligible(ranking_objective="top10")
    assert not eligible(is_current_enzyme=True)
    assert not eligible(has_seed_reactions=True)
    assert not eligible(requested_retrieval_mode="direct")
    assert not eligible(model_dir_override=Path("manual-model"))
    assert not eligible(dual_tower_dir=Path("different-model"))
    assert not eligible(has_temporary_external_reactions=True)
    assert not eligible(registered_reactions_csv=Path("different-registry.csv"))
    assert not eligible(registered_reactions_csv=None)


def test_dual_kernel_asset_reference_survives_repository_relocation():
    expected = Path(
        "/new/server/repo/results/terpene_production_models/"
        "marts_dual_kernel_e2r_top20"
    )
    recorded = (
        "/old/server/repo/results/terpene_production_models/"
        "marts_dual_kernel_e2r_top20"
    )
    assert references_packaged_asset(recorded, expected)
    assert not references_packaged_asset("", expected)
    assert not references_packaged_asset(
        "/old/server/results/other_assets", expected
    )
