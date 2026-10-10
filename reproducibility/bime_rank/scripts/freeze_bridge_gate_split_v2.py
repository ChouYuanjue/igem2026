from __future__ import annotations

"""Freeze a query-isolated E2R gate development split without rewriting the parent test.

Every original score/rank asset remains available. Fixed evaluation queries are
never used to train or tune the E2R gate; the original edge list is immutable.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_standardized_v3 import annotate

SOURCE = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUT = ROOT / "results/bridge_gate_split_v2"
DEV_SALT = "IGEM26:E2R:DEV:20261010"
VALID_SALT = "IGEM26:E2R:INTERNALVALID:20261010"
DEV_FRACTION = 0.10
VALID_FRACTION = 0.25
KEYS = ("protein_id", "reaction_id")
PARTITIONS = ("gate_learn", "gate_validation", "evaluation")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def proportion(text: str, salt: str) -> float:
    return int.from_bytes(
        hashlib.blake2b((salt + str(text)).encode(), digest_size=8).digest(),
        "big",
    ) / 2**64


def main() -> None:
    t = pd.read_csv(SOURCE, dtype=str).drop_duplicates(list(KEYS))
    train = pd.read_csv(TRAIN, dtype=str).drop_duplicates(list(KEYS))
    if len(t) != 23773:
        raise ValueError("Expected canonical 23,773 relation edge benchmark")
    if len(set(map(tuple, t[list(KEYS)].to_numpy())) & set(map(tuple, train[list(KEYS)].to_numpy()))):
        raise AssertionError("Benchmark contains train relation")
    frame = annotate(
        t,
        train.groupby("protein_id").size().to_dict(),
        train.groupby("reaction_id").size().to_dict(),
    )
    partition = {}
    for q in frame.protein_id.unique():
        if proportion(q, DEV_SALT) >= DEV_FRACTION:
            partition[q] = "evaluation"
        elif proportion(q, VALID_SALT) < VALID_FRACTION:
            partition[q] = "gate_validation"
        else:
            partition[q] = "gate_learn"
    frame["partition"] = frame.protein_id.map(partition)
    assert frame.groupby("protein_id").partition.nunique().eq(1).all()
    if frame.partition.nunique() != 3:
        raise ValueError("Need exactly three partitions")
    counts = frame.partition.value_counts()
    fraction = 1 - counts["evaluation"] / len(frame)
    if not 0.075 <= fraction <= 0.125:
        raise AssertionError(f"Development fraction unexpectedly imbalanced: {fraction}")
    original_distribution = frame.novelty.value_counts(normalize=True)
    dev_distribution = frame.loc[frame.partition.ne("evaluation")].novelty.value_counts(normalize=True)
    if (original_distribution - dev_distribution).abs().max() > 0.035:
        raise AssertionError("Holdout novelty distribution materially changed")
    if min(counts["gate_learn"], counts["gate_validation"]) < 400:
        raise AssertionError("Gate query-validation too small")
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [*KEYS, "partition", "novelty", "difficulty_stratum", "protein_degree", "reaction_degree"]
    frame[columns].sort_values([*KEYS]).to_csv(OUT / "membership.csv.gz", index=False)
    for part in PARTITIONS:
        frame.loc[frame.partition.eq(part), list(KEYS)].sort_values(
            list(KEYS)
        ).to_csv(OUT / f"{part}_pairs.csv.gz", index=False)
    manifest = {
        "schema": "bridge-query-group-heldout-split-v2",
        "source": "unchanged frozen bridge_relation_unseen_max_v3/targets.csv",
        "original_relations": len(frame),
        "original_queries_e2r": int(frame.protein_id.nunique()),
        "original_sha256": sha256(SOURCE),
        "clean2023_sha256": sha256(TRAIN),
        "splitting": "protein-query-group by hash before viewing model scores; novelty and degree audit only",
        "dev_hash_salt": DEV_SALT,
        "internal_validation_hash_salt": VALID_SALT,
        "target_development_fraction": DEV_FRACTION,
        "actual_development_fraction": fraction,
        "parts": {
            part: {
                "edges": int(counts[part]),
                "unique_e2r_queries": int(frame.loc[frame.partition.eq(part), "protein_id"].nunique()),
                "novelty_edge_counts": {
                    str(k): int(v)
                    for k, v in frame.loc[frame.partition.eq(part)].novelty.value_counts().items()
                },
                "degree_stratum_edge_counts": {
                    str(k): int(v)
                    for k, v in frame.loc[frame.partition.eq(part)].difficulty_stratum.value_counts().items()
                },
                "csv_sha256": sha256(OUT / f"{part}_pairs.csv.gz"),
            }
            for part in PARTITIONS
        },
        "evaluation_labels_used_for_gate": False,
        "all_original_rank_assets_unchanged": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "parent_sha256": manifest["original_sha256"],
        "total_relations": len(frame),
        "partition_summaries": manifest["parts"],
        "dev_novelty_fraction": dev_distribution.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
