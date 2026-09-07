from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.prepare_broad_rhea_benchmarks import (  # noqa: E402
    DEFAULT_BASE_TRAIN,
    DEFAULT_RAW,
    DEFAULT_UNIVERSE,
    load_base_exposure,
    load_rhea_pairs,
    pair_frame,
    write_cell,
)

DEFAULT_PARENT_CELL = ROOT / "results/broad_rhea_fair_benchmarks_v1/temporal_post2020_double_cold"
DEFAULT_OUTPUT = ROOT / "results/broad_rhea_nested_dev_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_nested_temporal_double_cold(
    parent_train: pd.DataFrame,
    dated_pairs: pd.DataFrame,
    *,
    cutoff: int,
    base_proteins: set[str],
    base_reactions: set[str],
    base_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(
        map(tuple, pair_frame(parent_train)[["protein_id", "reaction_id"]].itertuples(index=False, name=None))
    )
    scoped = dated_pairs[
        [pair in allowed for pair in zip(dated_pairs["protein_id"], dated_pairs["reaction_id"])]
    ].copy()
    scoped_pairs = set(
        map(tuple, pair_frame(scoped)[["protein_id", "reaction_id"]].itertuples(index=False, name=None))
    )
    missing = allowed - scoped_pairs
    if missing:
        raise ValueError(f"Parent train contains {len(missing)} pairs without dated Rhea provenance")

    train = scoped[scoped["creation_date"].between(1, cutoff)].copy()
    later = scoped[scoped["creation_date"] > cutoff].copy()
    train_proteins = set(train["protein_id"])
    train_reactions = set(train["reaction_id"])
    dev = later[
        ~later["protein_id"].isin(train_proteins)
        & ~later["reaction_id"].isin(train_reactions)
        & ~later["protein_id"].isin(base_proteins)
        & ~later["reaction_id"].isin(base_reactions)
    ].copy()
    dev = dev[
        [pair not in base_pairs for pair in zip(dev["protein_id"], dev["reaction_id"])]
    ]
    return train, dev


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict temporal double-cold nested development cell entirely inside an existing "
            "outer benchmark training partition. The outer test partition is never read."
        )
    )
    parser.add_argument("--parent-cell", type=Path, default=DEFAULT_PARENT_CELL)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inner-cutoff-year", type=int, default=2016)
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    parent_cell = args.parent_cell.resolve()
    parent_train_path = parent_cell / "train_pairs.csv"
    parent_manifest_path = parent_cell / "manifest.json"
    if not parent_train_path.is_file() or not parent_manifest_path.is_file():
        raise FileNotFoundError(f"Parent cell must contain train_pairs.csv and manifest.json: {parent_cell}")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if not bool(parent_manifest.get("valid")):
        raise RuntimeError(f"Parent benchmark cell is not valid: {parent_manifest.get('violations')}")

    parent_train = pd.read_csv(parent_train_path, dtype=str).fillna("")
    dated_pairs = load_rhea_pairs(args.raw.resolve(), args.universe.resolve())
    base_proteins, base_reactions, base_pairs = load_base_exposure(
        args.base_train.resolve(), args.universe.resolve()
    )
    cutoff = int(args.inner_cutoff_year) * 10000 + 1231
    train, dev = build_nested_temporal_double_cold(
        parent_train,
        dated_pairs,
        cutoff=cutoff,
        base_proteins=base_proteins,
        base_reactions=base_reactions,
        base_pairs=base_pairs,
    )
    name = args.name or f"{parent_cell.name}_inner_post{args.inner_cutoff_year}_double_cold"
    metadata = write_cell(
        args.output.resolve(),
        name=name,
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
            f"Nested development inside parent train {parent_cell.name}: inner train <= {args.inner_cutoff_year}; "
            "later parent-train pairs are dev only when both entities are unseen in inner train and absent from base exposure"
        ),
        claim_tier="nested_dev_only",
    )
    cell_dir = args.output.resolve() / name
    metadata.update(
        {
            "parent_cell": str(parent_cell),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_train_sha256": sha256_file(parent_train_path),
            "inner_cutoff_year": int(args.inner_cutoff_year),
            "outer_test_read": False,
        }
    )
    (cell_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
