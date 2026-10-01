from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.runtime.ranking_metrics import (
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.fibre.runtime.relational import FibreRelationalRuntime
from projects.active.fibre.training.train import split_double_cold


DEFAULT_CHECKPOINT = ROOT / "results/fibre_relational_main_v1/best.pt"
DEFAULT_PAIRS = (
    ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
)
DEFAULT_OUTPUT = ROOT / "results/fibre_relational_main_v1/internal_double_cold"


def _final_order(
    runtime: FibreRelationalRuntime,
    *,
    direction: str,
    query_id: str,
) -> list[str] | None:
    if direction == "r2e":
        if not runtime.assets.tokens.available(query_id):
            return None
        base, _ = runtime.index.proteins_for_reaction(
            query_id,
            k=len(runtime.index.protein_ids),
        )
        head = base[: runtime.shortlist]
        relation = runtime._scores(head, [query_id] * len(head))
    elif direction == "e2r":
        base, _ = runtime.index.reactions_for_protein(
            query_id,
            k=len(runtime.index.reaction_ids),
        )
        original_head = base[: runtime.shortlist]
        head = [rid for rid in original_head if runtime.assets.tokens.available(rid)]
        relation = runtime._scores([query_id] * len(head), head)
        unsupported = [rid for rid in original_head if rid not in set(head)]
    else:
        raise ValueError(direction)

    order = np.argsort(-relation, kind="stable")
    reranked = [head[int(i)] for i in order]
    if direction == "e2r":
        reranked.extend(unsupported)
    reranked.extend(base[runtime.shortlist :])
    if len(reranked) != len(base) or len(set(reranked)) != len(base):
        raise AssertionError("final two-stage order is not a full permutation")
    return reranked


def _evaluate_direction(
    runtime: FibreRelationalRuntime,
    test: pd.DataFrame,
    *,
    direction: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if direction == "r2e":
        grouped = test.groupby("reaction_id").protein_id.apply(set)
    else:
        grouped = test.groupby("protein_id").reaction_id.apply(set)

    rows: list[dict[str, object]] = []
    unsupported = 0
    for query_id, positives in grouped.items():
        order = _final_order(runtime, direction=direction, query_id=str(query_id))
        if order is None:
            unsupported += 1
            continue
        rank = {value: i + 1 for i, value in enumerate(order)}
        positive_ranks = np.asarray(
            sorted(rank[value] for value in positives if value in rank),
            dtype=np.int64,
        )
        if len(positive_ranks) == 0:
            continue
        metrics = evaluate_full_candidate_ranks(positive_ranks, len(order))
        rows.append({"query_id": str(query_id), **metrics})

    frame = pd.DataFrame(rows)
    summary = summarize_query_metrics(frame) if not frame.empty else {"query_count": 0}
    summary["labelled_query_count"] = int(len(grouped))
    summary["unsupported_query_count"] = int(unsupported)
    summary["relation_query_coverage"] = (
        float((len(grouped) - unsupported) / len(grouped)) if len(grouped) else 0.0
    )
    summary["shortlist"] = int(runtime.shortlist)
    return frame, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate the single final FIBRE runtime on the held-out double-cold test split."
    )
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--shortlist", type=int, default=4096)
    ap.add_argument("--relation-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    _, _, test = split_double_cold(pairs)
    runtime = FibreRelationalRuntime(
        args.checkpoint,
        device=args.device,
        shortlist=args.shortlist,
        relation_batch=args.relation_batch,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema": "fibre-relational-internal-double-cold-v1",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "protocol": "fixed 10-fold entity hash; fold 1 protein and reaction entities held out",
        "runtime": "full-universe dual-encoder shortlist then FIBRE relational rerank",
        "shortlist": int(args.shortlist),
        "directions": {},
    }
    for direction in ("r2e", "e2r"):
        frame, summary = _evaluate_direction(runtime, test, direction=direction)
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
