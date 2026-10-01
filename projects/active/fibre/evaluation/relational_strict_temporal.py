from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from projects.active.fibre.evaluation.relational_double_cold import _evaluate_direction
from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.runtime.relational import FibreRelationalRuntime


DEFAULT_CHECKPOINT = ROOT / "results/fibre_relational_main_v1/best.pt"
DEFAULT_PAIRS = (
    ROOT
    / "results/rhea128_to141_external_v2/"
    "rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv"
)
DEFAULT_OUTPUT = ROOT / "results/fibre_relational_main_v1/strict_temporal_rhea128_to141"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate the single final FIBRE runtime on frozen Rhea128-to-141 strict double-cold pairs."
    )
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--shortlist", type=int, default=4096)
    ap.add_argument("--relation-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    required = {"protein_id", "reaction_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"strict-temporal pairs require columns {sorted(required)}")

    runtime = FibreRelationalRuntime(
        args.checkpoint,
        device=args.device,
        shortlist=args.shortlist,
        relation_batch=args.relation_batch,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "schema": "fibre-relational-rhea128-to141-strict-temporal-v1",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "pair_source": str(args.pairs.relative_to(ROOT)),
        "protocol": (
            "frozen Rhea release128->141 Swiss-Prot strict double-cold pairs; "
            "no target labels used for training or checkpoint selection"
        ),
        "runtime": "full-universe dual-encoder shortlist then FIBRE relational rerank",
        "shortlist": int(args.shortlist),
        "test_pairs": int(len(pairs)),
        "directions": {},
    }
    for direction in ("r2e", "e2r"):
        frame, summary = _evaluate_direction(runtime, pairs, direction=direction)
        frame.to_csv(args.output / f"{direction}_queries.csv", index=False)
        result["directions"][direction] = summary
        print(direction, json.dumps(summary), flush=True)

    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
