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

from projects.active.terpene_screening.dual_kernel_runtime import build_assets  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_E2R_DUAL_TOWER_DIR,
    DEFAULT_PROTEIN_DIR,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    load_protein_library,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_production_models/marts_dual_kernel_e2r_top20"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare locked sparse dual-kernel assets for production E2R Top-20 routing."
    )
    parser.add_argument(
        "--production-dir", type=Path, default=DEFAULT_E2R_DUAL_TOWER_DIR
    )
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument(
        "--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    production_dir = args.production_dir.resolve()
    protein_registry = pd.read_csv(
        production_dir / "protein_registry.csv", dtype=str
    ).fillna("")
    reaction_registry = pd.read_csv(
        production_dir / "reaction_registry.csv", dtype=str
    ).fillna("")
    training_pairs = pd.read_csv(
        production_dir / "training_pairs.csv", dtype=str
    ).fillna("")

    current_features, current_ids = load_protein_library(
        args.current_protein_dir.resolve()
    )
    registered_features, registered_ids = load_protein_library(
        args.registered_protein_dir.resolve()
    )
    combined_ids = list(current_ids) + list(registered_ids)
    combined_features = np.concatenate(
        [current_features, registered_features], axis=0
    ).astype(np.float32)
    expected_ids = protein_registry["protein_id"].astype(str).tolist()
    if combined_ids != expected_ids:
        mismatch = next(
            (
                (index, left, right)
                for index, (left, right) in enumerate(
                    zip(combined_ids, expected_ids)
                )
                if left != right
            ),
            None,
        )
        raise ValueError(
            "Runtime protein feature order differs from production registry: "
            f"counts={len(combined_ids)}/{len(expected_ids)}, first_mismatch={mismatch}"
        )

    metadata = build_assets(
        reaction_registry=reaction_registry,
        protein_ids=combined_ids,
        protein_features=combined_features,
        training_pairs=training_pairs,
        output_dir=args.output_dir,
    )
    metadata.update(
        {
            "production_dir": str(production_dir),
            "current_protein_dir": str(args.current_protein_dir.resolve()),
            "registered_protein_dir": str(
                args.registered_protein_dir.resolve()
            ),
            "protein_registry": str(production_dir / "protein_registry.csv"),
            "reaction_registry": str(production_dir / "reaction_registry.csv"),
            "training_pairs": str(production_dir / "training_pairs.csv"),
        }
    )
    (args.output_dir.resolve() / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
