from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.evaluate_enzymecage_405_cleanroom import (
    DEFAULT_NEW_PROTEINS,
    DEFAULT_UNIVERSE,
    build_official_protein_library,
    load_query_reaction_library,
    official_query_id,
    score_models,
)
from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import evaluate_scores
from projects.active.terpene_screening.fair_benchmark import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
DEFAULT_QUERY_REACTIONS = (
    ROOT
    / "data/external/enzymecage_current/catalyst_features/query_reaction_rdkitplus_center_v1"
)


def prepare_fixed_support(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"UniprotID", "sequence", "CANO_RXN_SMILES", "Label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fixed support missing columns: {sorted(missing)}")
    local = frame.copy()
    local["_support_row"] = np.arange(len(local), dtype=np.int64)
    local["reaction_id"] = local["CANO_RXN_SMILES"].astype(str).map(official_query_id)
    local["protein_id"] = local["UniprotID"].astype(str)
    local["label"] = pd.to_numeric(local["Label"], errors="raise").astype(int).clip(0, 1)
    if local.groupby("reaction_id")["label"].sum().le(0).any():
        bad = int(local.groupby("reaction_id")["label"].sum().le(0).sum())
        raise ValueError(f"fixed support contains {bad} reaction queries without any positive")
    return local


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score an already-frozen Enzyme-405 pair reservoir with the frozen Catalyst production model. "
            "No support filtering, routing, or benchmark-specific model adaptation is performed."
        )
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--new-protein-dir", type=Path, default=DEFAULT_NEW_PROTEINS)
    parser.add_argument("--query-reaction-dir", type=Path, default=DEFAULT_QUERY_REACTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int)
    parser.add_argument("--expected-row-count", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pairs_path = args.pairs.resolve()
    raw = pd.read_csv(pairs_path, dtype=str).fillna("")
    frame = prepare_fixed_support(raw)
    if args.expected_query_count is not None and frame["reaction_id"].nunique() != args.expected_query_count:
        raise ValueError(
            f"query-count drift: {frame['reaction_id'].nunique()} != {args.expected_query_count}"
        )
    if args.expected_row_count is not None and len(frame) != args.expected_row_count:
        raise ValueError(f"row-count drift: {len(frame)} != {args.expected_row_count}")

    protein_features, protein_ids, protein_audit = build_official_protein_library(
        frame, args.universe_dir.resolve(), args.new_protein_dir.resolve()
    )
    reaction_features, reaction_ids = load_query_reaction_library(args.query_reaction_dir.resolve())
    missing_reactions = sorted(set(frame["reaction_id"]) - set(reaction_ids))
    if missing_reactions:
        raise ValueError(f"query feature library misses reactions: {missing_reactions[:5]}")

    scored = score_models(
        frame,
        model_dir=args.model_dir.resolve(),
        protein_features=protein_features,
        protein_ids=protein_ids,
        reaction_features=reaction_features,
        reaction_ids=reaction_ids,
        device=torch.device(args.device),
    )
    metrics, query_metrics = evaluate_scores(scored, "neural_score")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored.sort_values("_support_row").to_csv(output / "pair_scores.csv", index=False)
    protein_audit.to_csv(output / "protein_feature_audit.csv", index=False)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    model_summary_path = args.model_dir.resolve() / "summary.json"
    summary = {
        "protocol": "frozen Enzyme-405 exact pair-support Catalyst evaluation",
        "support_selection_allowed_here": False,
        "target_labels_used_for_model_or_routing": False,
        "pairs": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "model_dir": str(args.model_dir.resolve()),
        "model_summary": (
            json.loads(model_summary_path.read_text(encoding="utf-8"))
            if model_summary_path.is_file()
            else None
        ),
        "rows_raw": int(len(frame)),
        "rows_canonical": int(len(frame.drop_duplicates(["reaction_id", "protein_id"]))),
        "queries": int(frame["reaction_id"].nunique()),
        "candidate_uids": int(frame["protein_id"].nunique()),
        "positive_rows_raw": int(frame["label"].sum()),
        "protein_feature_coverage": int(len(protein_ids)),
        "reaction_feature_coverage": int(len(reaction_ids)),
        "metrics": metrics,
        "fairness_boundary": (
            "The input pair table is treated as immutable support. Catalyst receives only its frozen production "
            "reaction/protein feature pipeline; no candidate/query removal, benchmark-specific routing, novelty "
            "shortcut, score fusion, or target-dependent hyperparameter selection is applied."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
