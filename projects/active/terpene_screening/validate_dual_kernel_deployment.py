from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.dual_kernel_runtime import (  # noqa: E402
    LOCKED_DEGREE_POWER,
    LOCKED_PROTEIN_K,
    LOCKED_REACTION_K,
    LOCKED_TEMPERATURE,
    align_reaction_scores,
    load_assets,
    protein_affinity,
    score_query,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_E2R_DUAL_TOWER_DIR,
    DEFAULT_E2R_TOP20_DUAL_KERNEL_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_UNCERTAINTY_CALIBRATORS,
    E2R_TOP20_RRF_CONSTANT,
    E2R_TOP20_RRF_PRIMARY_WEIGHT,
    load_external_reaction_rows,
    load_feature_schema,
    load_protein_library,
    load_reaction_library,
)

DEFAULT_REGISTRY_RESULTS = ROOT / "results/terpene_registry_batch"
DEFAULT_OUTPUT = ROOT / "results/terpene_deployment_validation_e2r_top20_dual_kernel.json"
EXPECTED_SCORE_SOURCE = "rrf_e2r_top20_primary0.7_dual_kernel0.3_c60"


def references_packaged_asset(recorded: str, expected: Path) -> bool:
    """Accept a relocated clone while still requiring the locked asset package."""
    value = str(recorded).strip()
    return bool(value) and Path(value).name == expected.resolve().name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the packaged E2R Top-20 dual-kernel production route."
    )
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_E2R_TOP20_DUAL_KERNEL_DIR)
    parser.add_argument("--e2r-dir", type=Path, default=DEFAULT_E2R_DUAL_TOWER_DIR)
    parser.add_argument(
        "--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR
    )
    parser.add_argument(
        "--registered-reactions-csv", type=Path, default=DEFAULT_REGISTERED_REACTIONS
    )
    parser.add_argument(
        "--calibrators", type=Path, default=DEFAULT_UNCERTAINTY_CALIBRATORS
    )
    parser.add_argument("--registry-results", type=Path, default=DEFAULT_REGISTRY_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assets = load_assets(args.asset_dir)
    metadata = assets.metadata
    locked = {
        "reaction_k": int(metadata["reaction_k"]) == LOCKED_REACTION_K,
        "protein_k": int(metadata["protein_k"]) == LOCKED_PROTEIN_K,
        "temperature": float(metadata["temperature"]) == LOCKED_TEMPERATURE,
        "degree_power": float(metadata["degree_power"]) == LOCKED_DEGREE_POWER,
        "rrf_primary_weight": E2R_TOP20_RRF_PRIMARY_WEIGHT == 0.70,
        "rrf_constant": E2R_TOP20_RRF_CONSTANT == 60.0,
    }
    if not all(locked.values()):
        raise ValueError(f"Locked dual-kernel parameters drifted: {locked}")

    schema = load_feature_schema(args.e2r_dir)
    _, runtime_reaction_ids = load_reaction_library(args.e2r_dir, schema)
    registered_reactions = load_external_reaction_rows(args.registered_reactions_csv)
    seen = set(runtime_reaction_ids)
    runtime_reaction_ids.extend(
        value
        for value in registered_reactions["reaction_id"].astype(str)
        if value not in seen
    )
    if set(runtime_reaction_ids) != set(assets.reaction_ids):
        raise ValueError("Dual-kernel and E2R runtime reaction candidate sets differ")

    registered_features, registered_ids = load_protein_library(
        args.registered_protein_dir
    )
    if not len(registered_ids):
        raise ValueError("No registered enzyme queries are available")
    query_id = registered_ids[0]
    query_feature = registered_features[0]
    selected_without, _ = protein_affinity(query_feature, assets)
    selected_with, weights = protein_affinity(
        query_feature, assets, query_id=query_id
    )
    query_row = assets.protein_to_row.get(query_id)
    if query_row is not None and query_row in set(selected_with.tolist()):
        raise ValueError("Dual-kernel query self-exclusion failed")
    scores = align_reaction_scores(
        score_query(query_feature, assets, query_id=query_id),
        assets,
        runtime_reaction_ids,
    )
    if scores.shape != (len(runtime_reaction_ids),) or not np.isfinite(scores).all():
        raise ValueError("Dual-kernel runtime scores are invalid")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Dual-kernel protein affinity weights are not normalized")

    calibrators = json.loads(args.calibrators.read_text(encoding="utf-8"))
    calibrator = calibrators.get("enzyme_to_reaction_top20", {})
    calibration_metrics = calibrator.get("cross_validated", {})
    if not calibrator.get("deployable") or float(
        calibration_metrics.get("roc_auc_ci_low", 0.0)
    ) <= 0.5:
        raise ValueError("E2R Top-20 reliability calibrator is not deployable")

    rankings = pd.read_csv(
        args.registry_results / "enzyme_to_reaction_rankings.csv", dtype=str
    ).fillna("")
    top20 = rankings[rankings["ranking_objective"].eq("top20")]
    if len(top20) != 694 * 20:
        raise ValueError(f"Unexpected E2R Top-20 registry size: {len(top20)}")
    if set(top20["score_source"]) != {EXPECTED_SCORE_SOURCE}:
        raise ValueError("Registry Top-20 rows do not use the locked score source")
    expected_asset_dir = args.asset_dir.resolve()
    recorded_asset_dirs = set(top20["auxiliary_score_directory"].astype(str))
    if not recorded_asset_dirs or not all(
        references_packaged_asset(value, expected_asset_dir)
        for value in recorded_asset_dirs
    ):
        raise ValueError("Registry Top-20 rows do not record the packaged assets")
    leaks = pd.read_csv(args.registry_results / "known_association_leaks.csv")
    if len(leaks):
        raise ValueError(f"Known association leaks remain: {len(leaks)}")

    summary = {
        "status": "valid",
        "asset_dir": str(expected_asset_dir),
        "registry_asset_references": sorted(recorded_asset_dirs),
        "registry_asset_reference_policy": "portable_directory_name_match",
        "e2r_dir": str(args.e2r_dir.resolve()),
        "locked_parameters": {
            "reaction_k": LOCKED_REACTION_K,
            "protein_k": LOCKED_PROTEIN_K,
            "temperature": LOCKED_TEMPERATURE,
            "degree_power": LOCKED_DEGREE_POWER,
            "primary_weight": E2R_TOP20_RRF_PRIMARY_WEIGHT,
            "auxiliary_weight": 1.0 - E2R_TOP20_RRF_PRIMARY_WEIGHT,
            "rrf_constant": E2R_TOP20_RRF_CONSTANT,
        },
        "n_reactions": len(assets.reaction_ids),
        "n_proteins": len(assets.protein_ids),
        "n_training_pairs": int(metadata["n_training_pairs"]),
        "n_train_proteins": int(metadata["n_train_proteins"]),
        "support_shape": list(assets.reaction_protein_support.shape),
        "support_nnz": int(assets.reaction_protein_support.nnz),
        "support_zero_rows": int(metadata["support_zero_rows"]),
        "runtime_reaction_set_match": True,
        "query_self_exclusion": True,
        "sample_query": query_id,
        "sample_neighbors_without_exclusion": selected_without.astype(int).tolist(),
        "sample_neighbors_with_exclusion": selected_with.astype(int).tolist(),
        "sample_score_range": [float(scores.min()), float(scores.max())],
        "calibrator": {
            "deployable": True,
            **calibration_metrics,
        },
        "registry_top20_rows": int(len(top20)),
        "registry_top20_queries": int(top20["query_id"].nunique()),
        "known_association_leaks": 0,
        "score_source": EXPECTED_SCORE_SOURCE,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
