from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT.parent / "igem2026"
if not SOURCE_ROOT.exists():
    SOURCE_ROOT = ROOT

BASE_FEATURES = [
    "primary_raw_score",
    "secondary_raw_score",
    "primary_query_zscore",
    "secondary_query_zscore",
    "primary_log_rank_fraction",
    "secondary_log_rank_fraction",
    "primary_reciprocal_rank",
    "secondary_reciprocal_rank",
    "zscore_difference",
    "log_rank_difference",
    "best_log_rank",
    "worst_log_rank",
    "fallback_log_rank_fraction",
    "alternate_log_rank_fraction",
    "fallback_zscore",
    "alternate_zscore",
    "primary_top10",
    "secondary_top10",
    "primary_top50",
    "secondary_top50",
    "primary_top200",
    "secondary_top200",
    "max_train_binary_drfp_tanimoto",
    "low_similarity_router_flag",
]
EXTRA_FEATURES = [
    "clip_raw_score",
    "clip_query_zscore",
    "clip_log_rank_fraction",
    "clip_reciprocal_rank",
    "clip_candidate_supported",
    "clip_query_supported",
    "clip_top10",
    "clip_top50",
    "clip_top100",
    "clip_z_minus_fallback",
    "clip_logrank_minus_fallback",
    "top10_votes3",
    "top50_votes3",
    "top100_votes3",
    "best3_log_rank",
    "best3_zscore",
]
FEATURES = [*BASE_FEATURES, *EXTRA_FEATURES]
IDX = {name: index for index, name in enumerate(FEATURES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen three-fold BiME-Rank/CLIPZyme development cache "
            "into the generic FIBRE scientific-evidence admission-table contract."
        )
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=(
            SOURCE_ROOT
            / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/prepared"
        ),
    )
    parser.add_argument(
        "--protein-entries",
        type=Path,
        default=(
            SOURCE_ROOT
            / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/fibre_structure_scientific_evidence_v4/admission_table.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results/fibre_structure_scientific_evidence_v4/export_summary.json",
    )
    args = parser.parse_args()

    entries = pd.read_csv(
        args.protein_entries,
        dtype={"row": int, "Entry": str},
    ).sort_values("row", kind="stable")
    if not np.array_equal(
        entries["row"].to_numpy(np.int64),
        np.arange(len(entries), dtype=np.int64),
    ):
        raise ValueError("protein entries rows must be contiguous and zero-based")
    protein_ids = entries["Entry"].astype(str).to_numpy()

    parts: list[pd.DataFrame] = []
    fold_summary: list[dict[str, int]] = []
    for fold in (0, 1, 2):
        folder = args.prepared_root / f"fold{fold}"
        queries = (
            pd.read_csv(folder / "queries.csv", dtype=str)["query_id"]
            .astype(str)
            .tolist()
        )
        with np.load(folder / "cache.npz") as cache:
            if cache["X"].shape[1] != len(FEATURES):
                raise ValueError("prepared CLIPZyme feature contract drifted")
            features = cache["X"]
            labels = cache["labels"].astype(np.int8)
            candidate_rows = cache["candidate_rows"].astype(np.int64)
            query_ptr = cache["query_ptr"].astype(np.int64)
            if len(query_ptr) != len(queries) + 1:
                raise ValueError("query pointer count does not match queries.csv")
            fold_parts: list[pd.DataFrame] = []
            for query_index, query_id in enumerate(queries):
                start = int(query_ptr[query_index])
                stop = int(query_ptr[query_index + 1])
                local = features[start:stop]
                local_rows = candidate_rows[start:stop]
                available = (
                    (local[:, IDX["clip_candidate_supported"]] > 0.5)
                    & (local[:, IDX["clip_query_supported"]] > 0.5)
                )
                fold_parts.append(
                    pd.DataFrame(
                        {
                            "query_id": str(query_id),
                            "candidate_id": protein_ids[local_rows],
                            "core_score": local[
                                :, IDX["fallback_zscore"]
                            ].astype(np.float64),
                            "evidence_score": local[
                                :, IDX["clip_raw_score"]
                            ].astype(np.float64),
                            "available": available,
                            "label": labels[start:stop],
                            "fold": int(fold),
                        }
                    )
                )
            fold_frame = pd.concat(fold_parts, ignore_index=True)
            parts.append(fold_frame)
            fold_summary.append(
                {
                    "fold": int(fold),
                    "queries": int(fold_frame["query_id"].nunique()),
                    "rows": int(len(fold_frame)),
                    "positive_rows": int(fold_frame["label"].sum()),
                    "available_rows": int(fold_frame["available"].sum()),
                }
            )

    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("export contains duplicate query/candidate rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    summary = {
        "schema": "fibre-clipzyme-structure-evidence-export-v1",
        "prepared_root": str(args.prepared_root),
        "protein_entries": str(args.protein_entries),
        "feature_contract": {
            "core_score": "fallback_zscore",
            "evidence_score": "clip_raw_score",
            "available": (
                "clip_candidate_supported AND clip_query_supported; shortlist "
                "membership itself is not treated as evidence availability"
            ),
        },
        "folds": fold_summary,
        "rows": int(len(output)),
        "queries": int(output["query_id"].nunique()),
        "positive_rows": int(output["label"].sum()),
        "available_rows": int(output["available"].sum()),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
