from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_model_rank_fusion_double_cold import (  # noqa: E402
    load_score_matrix,
    parse_source,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_dual_tower_rankings"


def append_ranking(
    rows: list[dict[str, object]],
    *,
    source: str,
    split_id: str,
    direction: str,
    query_id: str,
    candidate_ids: list[str],
    scores: np.ndarray,
    positives: set[str],
    depth: int,
) -> None:
    order = np.lexsort((np.asarray(candidate_ids), -scores))[:depth]
    rows.extend(
        {
            "source": source,
            "split_id": split_id,
            "direction": direction,
            "query_id": query_id,
            "rank": rank,
            "candidate_id": candidate_ids[int(index)],
            "score": float(scores[int(index)]),
            "is_positive": int(candidate_ids[int(index)] in positives),
        }
        for rank, index in enumerate(order, start=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export strict MARTS Top-N rankings from one or more adapted dual-tower sources."
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.ranking_depth <= 0:
        raise ValueError("ranking-depth must be positive")
    sources = dict(parse_source(value) for value in args.source)
    cache = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protein_matrix = np.load(cache / "protein_features.npy").astype(np.float32)
    reaction_matrix = np.load(cache / "reaction_features.npy").astype(np.float32)
    proteins = pd.read_csv(cache / "protein_entities.csv", dtype=str).fillna("")
    reactions = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    for column in ("protein_seen", "reaction_seen"):
        pairs[column] = pairs[column].astype(str).str.lower().eq("true")
    protein_ids = proteins["protein_id"].astype(str).tolist()
    reaction_ids = reactions["reaction_id"].astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    device = torch.device(args.device)
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)

    ranking_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ].copy()
            if test.empty:
                continue
            for label, source_dir in sources.items():
                scores = load_score_matrix(
                    source_dir, split_id, protein_tensor, reaction_tensor, device
                )
                for reaction_id, group in test.groupby("rhea_id", sort=True):
                    reaction_id = str(reaction_id)
                    append_ranking(
                        ranking_rows,
                        source=label,
                        split_id=split_id,
                        direction="reaction_to_enzyme",
                        query_id=reaction_id,
                        candidate_ids=protein_ids,
                        scores=scores[reaction_to_row[reaction_id]],
                        positives=set(group["Entry"].astype(str)),
                        depth=args.ranking_depth,
                    )
                for protein_id, group in test.groupby("Entry", sort=True):
                    protein_id = str(protein_id)
                    append_ranking(
                        ranking_rows,
                        source=label,
                        split_id=split_id,
                        direction="enzyme_to_reaction",
                        query_id=protein_id,
                        candidate_ids=reaction_ids,
                        scores=scores[:, protein_to_row[protein_id]],
                        positives=set(group["rhea_id"].astype(str)),
                        depth=args.ranking_depth,
                    )
                del scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            split_rows.append(
                {
                    "split_id": split_id,
                    "test_pairs": len(test),
                    "test_proteins": test["Entry"].nunique(),
                    "test_reactions": test["rhea_id"].nunique(),
                }
            )

    rankings = pd.DataFrame(ranking_rows)
    keys = ["source", "split_id", "direction", "query_id"]
    counts = rankings.groupby(keys).size()
    if counts.empty or counts.min() != args.ranking_depth or counts.max() != args.ranking_depth:
        raise ValueError("Ranking export did not produce a fixed depth for every source/query")
    if rankings.duplicated(keys + ["candidate_id"]).any():
        raise ValueError("Ranking export contains duplicate candidates within a query")
    rankings.to_csv(output / "rankings.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_summary.csv", index=False)
    summary = {
        "sources": {label: str(path) for label, path in sources.items()},
        "ranking_depth": args.ranking_depth,
        "n_sources": len(sources),
        "n_query_sources": int(len(counts)),
        "n_ranking_rows": int(len(rankings)),
        "outputs": {
            "rankings": str(output / "rankings.csv"),
            "split_summary": str(output / "split_summary.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
