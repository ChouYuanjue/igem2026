from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.prepare_broad_rhea_benchmarks import (
    load_base_exposure,
    stable_bucket,
    write_cell,
)

DEFAULT_BENCHMARK_ROOT = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "results/broad_rhea_nested_dev_v1"
DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_BASE_TRAIN = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/training_pairs.csv"


def _pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"protein_id", "reaction_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pair frame missing columns: {sorted(missing)}")
    return (
        frame[["protein_id", "reaction_id"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["protein_id", "reaction_id"])
        .reset_index(drop=True)
    )


def nested_double_cold(
    pairs: pd.DataFrame,
    *,
    base_proteins: set[str],
    base_reactions: set[str],
    base_pairs: set[tuple[str, str]],
    modulo: int,
    holdout_bucket: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a deterministic development split with both entity types unseen.

    The split is derived only from the *outer training pairs*. A seed set is formed
    from independently hashed protein/reaction entities, excluding anything exposed
    by the frozen base checkpoint. Once an entity enters the dev set, every pair
    containing that entity is removed from inner training. This makes dev proteins
    and dev reactions strictly absent from continuation training.
    """
    pairs = _pair_frame(pairs)
    if modulo <= 1 or not 0 <= holdout_bucket < modulo:
        raise ValueError("modulo must be >1 and holdout_bucket must be in range")

    eligible = pairs[
        ~pairs["protein_id"].isin(base_proteins)
        & ~pairs["reaction_id"].isin(base_reactions)
    ].copy()
    eligible = eligible[
        [pair not in base_pairs for pair in zip(eligible["protein_id"], eligible["reaction_id"])]
    ]
    p_hold = eligible["protein_id"].map(
        lambda value: stable_bucket("nested-dev-protein", value, modulo=modulo) == holdout_bucket
    )
    r_hold = eligible["reaction_id"].map(
        lambda value: stable_bucket("nested-dev-reaction", value, modulo=modulo) == holdout_bucket
    )
    seeds = eligible[p_hold & r_hold]
    dev_proteins = set(seeds["protein_id"])
    dev_reactions = set(seeds["reaction_id"])
    if not dev_proteins or not dev_reactions:
        raise ValueError("nested double-cold seed set is empty")

    inner_train = pairs[
        ~pairs["protein_id"].isin(dev_proteins)
        & ~pairs["reaction_id"].isin(dev_reactions)
    ].copy()
    dev = eligible[
        eligible["protein_id"].isin(dev_proteins)
        & eligible["reaction_id"].isin(dev_reactions)
    ].copy()
    if dev.empty or inner_train.empty:
        raise ValueError("nested double-cold split is empty")
    return _pair_frame(inner_train), _pair_frame(dev)


def nested_protein_cold(
    pairs: pd.DataFrame,
    *,
    base_proteins: set[str],
    base_pairs: set[tuple[str, str]],
    modulo: int,
    holdout_bucket: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a protein-cold dev split while requiring every dev reaction in train."""
    pairs = _pair_frame(pairs)
    if modulo <= 1 or not 0 <= holdout_bucket < modulo:
        raise ValueError("modulo must be >1 and holdout_bucket must be in range")

    candidate_proteins = {
        value
        for value in set(pairs["protein_id"])
        if value not in base_proteins
        and stable_bucket("nested-dev-protein", value, modulo=modulo) == holdout_bucket
    }
    if not candidate_proteins:
        raise ValueError("nested protein-cold protein set is empty")
    inner_train = pairs[~pairs["protein_id"].isin(candidate_proteins)].copy()
    train_reactions = set(inner_train["reaction_id"])
    dev = pairs[
        pairs["protein_id"].isin(candidate_proteins)
        & pairs["reaction_id"].isin(train_reactions)
    ].copy()
    dev = dev[
        [pair not in base_pairs for pair in zip(dev["protein_id"], dev["reaction_id"])]
    ]
    if dev.empty or inner_train.empty:
        raise ValueError("nested protein-cold split is empty")
    return _pair_frame(inner_train), _pair_frame(dev)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train-only nested development cells for leakage-safe method selection."
    )
    parser.add_argument("--source-cell", default="reactzyme_reaction_projected_double_cold")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--modes", default="double_cold,protein_cold")
    parser.add_argument("--modulo", type=int, default=5)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    args = parser.parse_args()

    source_cell = args.benchmark_root.resolve() / args.source_cell
    outer_train_path = source_cell / "train_pairs.csv"
    outer_test_path = source_cell / "test_pairs.csv"
    if not outer_train_path.is_file() or not outer_test_path.is_file():
        raise FileNotFoundError(f"missing outer benchmark assets under {source_cell}")

    # Deliberately read only outer train labels for split construction. The outer
    # test path is hashed for provenance below, never loaded or used for selection.
    outer_train = pd.read_csv(outer_train_path, dtype=str).fillna("")
    base_proteins, base_reactions, base_pairs = load_base_exposure(
        args.base_train.resolve(), args.universe_dir.resolve()
    )
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - {"double_cold", "protein_cold"}
    if unknown:
        raise ValueError(f"unknown nested dev modes: {sorted(unknown)}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    suffix = f"m{args.modulo}b{args.holdout_bucket}"

    if "double_cold" in modes:
        train, dev = nested_double_cold(
            outer_train,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            modulo=args.modulo,
            holdout_bucket=args.holdout_bucket,
        )
        manifests.append(
            write_cell(
                output_root,
                name=f"{args.source_cell}__nested_double_cold_{suffix}",
                train=train,
                test=dev,
                base_proteins=base_proteins,
                base_reactions=base_reactions,
                base_pairs=base_pairs,
                expected={
                    "protein_unseen": True,
                    "reaction_unseen": True,
                    "base_protein_unseen": True,
                    "base_reaction_unseen": True,
                },
                source_protocol=(
                    f"Nested train-only deterministic double-cold hash split of {args.source_cell}; "
                    "outer test labels are not read during construction"
                ),
                claim_tier="development_only",
            )
        )

    if "protein_cold" in modes:
        train, dev = nested_protein_cold(
            outer_train,
            base_proteins=base_proteins,
            base_pairs=base_pairs,
            modulo=args.modulo,
            holdout_bucket=args.holdout_bucket,
        )
        manifests.append(
            write_cell(
                output_root,
                name=f"{args.source_cell}__nested_protein_cold_{suffix}",
                train=train,
                test=dev,
                base_proteins=base_proteins,
                base_reactions=base_reactions,
                base_pairs=base_pairs,
                expected={"protein_unseen": True, "base_protein_unseen": True},
                source_protocol=(
                    f"Nested train-only deterministic protein-cold hash split of {args.source_cell}; "
                    "dev reactions are required to remain seen in inner train"
                ),
                claim_tier="development_only",
            )
        )

    provenance = {
        "source_cell": args.source_cell,
        "outer_train_path": str(outer_train_path.resolve()),
        "outer_test_path": str(outer_test_path.resolve()),
        "outer_test_usage": "path recorded only; labels are never read by this builder",
        "modulo": args.modulo,
        "holdout_bucket": args.holdout_bucket,
        "manifests": manifests,
    }
    (output_root / f"{args.source_cell}__nested_dev_summary_{suffix}.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
