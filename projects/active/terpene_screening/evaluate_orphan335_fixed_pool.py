from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from projects.active.terpene_screening.build_label_blind_query_reaction_base import stable_query_id
from projects.active.terpene_screening.evaluate_enzymecage_405_cleanroom import (
    DEFAULT_UNIVERSE,
    build_official_protein_library,
    load_query_reaction_library,
    score_models,
)
from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import evaluate_scores
from projects.active.terpene_screening.fair_benchmark import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ROOT / "results/orphan335_author_retrieval_v1/Orphan-335_retrievel_cands_selenzyme_eval.csv"
DEFAULT_AUTHOR_NATIVE = ROOT / "results/orphan335_author_retrieval_v1/selenzyme_native_metrics.json"
DEFAULT_MODEL = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
DEFAULT_EXTRA_PROTEINS = ROOT / "data/external/enzymecage_current/catalyst_features/orphan335_new_protein_esmc"
DEFAULT_REACTIONS = ROOT / "data/external/enzymecage_current/catalyst_features/orphan335_query_rdkitplus_center_v1"
DEFAULT_OUTPUT = ROOT / "results/orphan335_fixed_pool_v1"


def prepare_pool(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"CANO_RXN_SMILES", "UniprotID", "sequence", "Label", "Score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Orphan-335 evaluation pool missing columns: {sorted(missing)}")
    frame = raw.copy()
    frame["reaction_id"] = frame["CANO_RXN_SMILES"].astype(str).map(
        lambda value: stable_query_id("O335", value)
    )
    frame["protein_id"] = frame["UniprotID"].astype(str)
    frame["label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int).clip(0, 1)
    frame["selenzyme_score"] = pd.to_numeric(frame["Score"], errors="raise")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Catalyst V3 and native Selenzyme-style Score on the exact Orphan-335 author candidate pool."
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--author-native-summary", type=Path, default=DEFAULT_AUTHOR_NATIVE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--extra-protein-dir", type=Path, default=DEFAULT_EXTRA_PROTEINS)
    parser.add_argument("--reaction-feature-dir", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pairs_path = args.pairs.resolve()
    raw = pd.read_csv(pairs_path, dtype=str).fillna("")
    frame = prepare_pool(raw)
    if len(frame) != 101771 or frame["reaction_id"].nunique() != 335:
        raise ValueError(
            f"Orphan support drift: rows={len(frame)}, queries={frame['reaction_id'].nunique()}"
        )

    selenzyme_metrics, selenzyme_query = evaluate_scores(frame, "selenzyme_score")
    author_path = args.author_native_summary.resolve()
    if author_path.is_file():
        author = json.loads(author_path.read_text(encoding="utf-8"))
        native = selenzyme_metrics["enzymecage_native_r2e"]
        mapping = {
            "top10_dcg": "top10_dcg",
            "top1_percent_ef": "top1_percent_ef",
            "top2_percent_ef": "top2_percent_ef",
            "top1_sr": "top1_sr",
            "top3_sr": "top3_sr",
            "top5_sr": "top5_sr",
            "top10_sr": "top10_sr",
        }
        deltas = {key: float(native[field]) - float(author[key]) for key, field in mapping.items()}
        max_abs = max(abs(value) for value in deltas.values())
        sr_fields = ["top1_sr", "top3_sr", "top5_sr", "top10_sr"]
        sr_max_abs = max(abs(deltas[key]) for key in sr_fields)
        if sr_max_abs > 1e-12:
            raise ValueError(f"native Selenzyme SR reproduction drift: max_abs_delta={sr_max_abs}, deltas={deltas}")
    else:
        deltas = None
        max_abs = None
        sr_max_abs = None

    protein_features, protein_ids, protein_audit = build_official_protein_library(
        frame,
        args.universe_dir.resolve(),
        args.extra_protein_dir.resolve(),
    )
    reaction_features, reaction_ids = load_query_reaction_library(args.reaction_feature_dir.resolve())
    expected_reactions = set(frame["reaction_id"])
    missing_reactions = sorted(expected_reactions - set(reaction_ids))
    if missing_reactions:
        raise ValueError(f"reaction feature library misses {len(missing_reactions)} queries: {missing_reactions[:5]}")
    if set(frame["protein_id"]) != set(protein_ids):
        missing = sorted(set(frame["protein_id"]) - set(protein_ids))
        extra = sorted(set(protein_ids) - set(frame["protein_id"]))
        raise ValueError(f"protein support drift: missing={missing[:5]}, extra={extra[:5]}")

    scored = score_models(
        frame,
        model_dir=args.model_dir.resolve(),
        protein_features=protein_features,
        protein_ids=protein_ids,
        reaction_features=reaction_features,
        reaction_ids=reaction_ids,
        device=torch.device(args.device),
    )
    catalyst_metrics, catalyst_query = evaluate_scores(scored, "neural_score")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "pair_scores.csv", index=False)
    protein_audit.to_csv(output / "protein_feature_audit.csv", index=False)
    selenzyme_query.to_csv(output / "selenzyme_query_metrics.csv", index=False)
    catalyst_query.to_csv(output / "catalyst_query_metrics.csv", index=False)
    query_positive = frame.groupby("reaction_id")["label"].sum()
    summary = {
        "protocol": "Orphan-335 immutable author candidate-pool comparison",
        "pairs": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "rows_raw": int(len(frame)),
        "rows_canonical": int(len(frame.drop_duplicates(["reaction_id", "protein_id"]))),
        "queries": int(frame["reaction_id"].nunique()),
        "candidate_uids": int(frame["protein_id"].nunique()),
        "positive_rows_raw": int(frame["label"].sum()),
        "queries_with_positive_in_candidate_pool": int((query_positive > 0).sum()),
        "queries_without_positive_in_candidate_pool": int((query_positive <= 0).sum()),
        "all_queries_retained": True,
        "author_candidate_pool_unchanged": True,
        "selenzyme_score_identity": "author Score unchanged",
        "selenzyme_native_reproduction_max_abs_delta": max_abs,
        "selenzyme_native_sr_reproduction_max_abs_delta": sr_max_abs,
        "selenzyme_native_reproduction_deltas": deltas,
        "selenzyme_native_metric_boundary": "Author SR@1/3/5/10 reproduces exactly. Author raw-row DCG/EF are retained as provenance; direct Catalyst-vs-Selenzyme deltas use this script's single canonical-pair evaluator for both score columns.",
        "selenzyme_metrics": selenzyme_metrics,
        "catalyst_model_dir": str(args.model_dir.resolve()),
        "catalyst_reaction_feature_dir": str(args.reaction_feature_dir.resolve()),
        "catalyst_extra_protein_dir": str(args.extra_protein_dir.resolve()),
        "catalyst_protein_feature_count": int(len(protein_ids)),
        "catalyst_metrics": catalyst_metrics,
        "fairness_boundary": (
            "The author retrieval candidate pool and 2025 evaluation truth are immutable. Selenzyme uses the author Score exactly. "
            "Catalyst uses its frozen V3 model and native ESM-C/RDKit+/reaction-center preprocessing; no candidate deletion, score fusion, "
            "target-dependent routing, or benchmark-specific model selection is performed. Queries with no 2025 positive inside the author "
            "candidate pool remain in the 335-query native denominator as failures."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
