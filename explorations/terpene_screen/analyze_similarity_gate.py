from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENZYMECAGE_ROOT = PROJECT_ROOT / "external_repos" / "EnzymeCAGE"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENZYMECAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENZYMECAGE_ROOT))

from retrieve import run_retrieval  # type: ignore

from explorations.terpene_screen.common import (
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    SOURCE_FILES,
    SOURCE_DATA_DIR,
    canonicalize_reaction_smiles,
    coerce_text,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_table,
)


DEFAULT_QUERY_PATH = SOURCE_DATA_DIR / "10rhea_selected.tsv"
DEFAULT_PAIR_PATH = TERPENE_RESULTS_DIR / "terpene_candidate_pairs.csv"
DEFAULT_POSITIVE_LABELS_PATH = SOURCE_FILES["positive_labels"]
DEFAULT_PRED_PATH = TERPENE_RESULTS_DIR / "predictions" / "all_pair_scores.csv"
DEFAULT_OUT_JSON = TERPENE_RESULTS_DIR / "similarity_gate_metrics.json"
DEFAULT_OUT_CSV = TERPENE_RESULTS_DIR / "similarity_gate_reaction_level.csv"


def _load_query_reactions(query_path: Path, pair_df: pd.DataFrame) -> pd.DataFrame:
    query_df = read_table(query_path)
    if "RHEA_ID" not in query_df.columns or "SMILES" not in query_df.columns:
        raise ValueError(f"Unexpected query columns in {query_path}: {query_df.columns.tolist()}")

    pair_lookup = (
        pair_df[["reaction_id", "rhea_id", "CANO_RXN_SMILES"]]
        .drop_duplicates()
        .set_index("rhea_id")
        .to_dict(orient="index")
    )

    rows: list[dict[str, Any]] = []
    for idx, row in query_df.iterrows():
        rhea_id = coerce_text(row.get("RHEA_ID"))
        raw_smiles = coerce_text(row.get("SMILES"))
        cano = canonicalize_reaction_smiles(raw_smiles) or raw_smiles
        mapped = pair_lookup.get(rhea_id, {})
        reaction_id = coerce_text(mapped.get("reaction_id")) or f"reaction_{idx + 1:02d}"
        cano_from_pairs = coerce_text(mapped.get("CANO_RXN_SMILES"))
        if cano_from_pairs:
            cano = cano_from_pairs
        rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": rhea_id,
                "raw_smiles": raw_smiles,
                "CANO_RXN_SMILES": cano,
            }
        )
    return pd.DataFrame(rows)


def _load_positive_db(positive_labels_path: Path) -> pd.DataFrame:
    raw_df = read_table(positive_labels_path)
    columns = identify_terpene_columns(raw_df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    seq_col = columns["sequence"]["column"]
    rxn_col = columns["reaction_smiles"]["column"]
    if id_col is None or seq_col is None or rxn_col is None:
        raise ValueError(f"Could not identify columns in {positive_labels_path}: {raw_df.columns.tolist()}")

    rows: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        raw_id = coerce_text(row.get(id_col))
        uniprot_id = parse_uniprot_id(raw_id)
        raw_rxn = coerce_text(row.get(rxn_col))
        sequence = coerce_text(row.get(seq_col))
        if not uniprot_id or not raw_rxn or not sequence:
            continue
        rows.append(
            {
                "CANO_RXN_SMILES": canonicalize_reaction_smiles(raw_rxn) or raw_rxn,
                "UniprotID": uniprot_id,
                "sequence": sequence,
            }
        )

    df_pos = pd.DataFrame(rows).drop_duplicates(subset=["CANO_RXN_SMILES", "UniprotID"]).reset_index(drop=True)
    return df_pos


def _build_retrieval_candidates(query_df: pd.DataFrame, positive_db: pd.DataFrame, topk: int) -> pd.DataFrame:
    retrieval_input = query_df[["CANO_RXN_SMILES"]].drop_duplicates().reset_index(drop=True)
    return run_retrieval(
        retrieval_input,
        positive_db,
        smiles_col="CANO_RXN_SMILES",
        uid_to_proevi={},
        uid_to_taxdis={},
        topk=topk,
    )


def _reaction_level_from_gate(
    pair_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    query_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
) -> pd.DataFrame:
    query_lookup = query_df.set_index("CANO_RXN_SMILES")[["reaction_id", "rhea_id"]].to_dict(orient="index")
    positive_lookup = (
        pair_df[pair_df["label"].astype(int) == 1]
        .groupby("reaction_id")["uniprot_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    retrieved_lookup = (
        retrieval_df.groupby("reaction")["enzyme"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    pred_by_reaction = {reaction_id: group.copy() for reaction_id, group in pred_df.groupby("reaction_id")}
    rows: list[dict[str, Any]] = []

    for cano_rxn, meta in query_lookup.items():
        reaction_id = meta["reaction_id"]
        rhea_id = meta["rhea_id"]
        candidate_enzymes = retrieved_lookup.get(cano_rxn, set())
        true_enzymes = positive_lookup.get(reaction_id, set())
        scored_subset = pred_by_reaction.get(reaction_id, pd.DataFrame()).copy()

        if not scored_subset.empty:
            scored_subset = scored_subset[scored_subset["uniprot_id"].astype(str).isin(candidate_enzymes)].copy()
            scored_subset = scored_subset.sort_values(
                by=["cage_score", "uniprot_id", "enzyme_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).reset_index(drop=True)
            scored_subset["rank_within_reaction"] = range(1, len(scored_subset) + 1)

        positives_scored = scored_subset[scored_subset["label"].astype(int) == 1].copy() if not scored_subset.empty else pd.DataFrame()
        best_positive_rank = None
        best_positive_enzyme_id = ""
        best_positive_score = None
        reciprocal_rank = 0.0
        if not positives_scored.empty:
            best_positive_row = positives_scored.iloc[0]
            best_positive_rank = int(best_positive_row["rank_within_reaction"])
            best_positive_enzyme_id = coerce_text(best_positive_row.get("enzyme_id"))
            best_positive_score = float(best_positive_row.get("cage_score"))
            reciprocal_rank = 1.0 / best_positive_rank

        candidate_hit_count = len(candidate_enzymes & true_enzymes)
        n_candidates = int(len(candidate_enzymes))
        n_scored_candidates = int(len(scored_subset))
        n_positive_enzymes = int(len(true_enzymes))
        n_positive_candidates = int(candidate_hit_count)
        n_scored_positive_candidates = int(
            len(positives_scored[positives_scored["uniprot_id"].astype(str).isin(true_enzymes)])
        ) if not positives_scored.empty else 0

        top10 = scored_subset.head(10)
        rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": rhea_id,
                "n_candidates": n_candidates,
                "n_scored_candidates": n_scored_candidates,
                "n_positive_enzymes": n_positive_enzymes,
                "n_positive_candidates_in_gate": n_positive_candidates,
                "n_scored_positive_candidates": n_scored_positive_candidates,
                "positive_pair_coverage": (n_positive_candidates / n_positive_enzymes) if n_positive_enzymes else None,
                "gate_hit": bool(n_positive_candidates > 0),
                "best_positive_rank": best_positive_rank,
                "best_positive_enzyme_id": best_positive_enzyme_id,
                "best_positive_score": best_positive_score,
                "top1_hit": bool(best_positive_rank is not None and best_positive_rank <= 1),
                "top3_hit": bool(best_positive_rank is not None and best_positive_rank <= 3),
                "top5_hit": bool(best_positive_rank is not None and best_positive_rank <= 5),
                "top10_hit": bool(best_positive_rank is not None and best_positive_rank <= 10),
                "top10_enzyme_ids": json.dumps(top10["enzyme_id"].astype(str).tolist(), ensure_ascii=False),
                "top10_scores": json.dumps([float(v) for v in top10["cage_score"].tolist()], ensure_ascii=False),
                "reciprocal_rank": reciprocal_rank,
            }
        )

    return pd.DataFrame(rows).sort_values("reaction_id", kind="mergesort").reset_index(drop=True)


def analyze_similarity_gate(
    query_path: Path,
    pair_path: Path,
    positive_labels_path: Path,
    pred_path: Path,
    out_json: Path,
    out_csv: Path,
    topk: int = 10,
) -> dict[str, Any]:
    pair_df = read_table(pair_path)
    if "label" not in pair_df.columns and "Label" in pair_df.columns:
        pair_df["label"] = pair_df["Label"]
    if "UniprotID" not in pair_df.columns and "uniprot_id" in pair_df.columns:
        pair_df["UniprotID"] = pair_df["uniprot_id"]
    if "sequence" not in pair_df.columns and "Sequence" in pair_df.columns:
        pair_df["sequence"] = pair_df["Sequence"]

    query_df = _load_query_reactions(query_path, pair_df)
    positive_db = _load_positive_db(positive_labels_path)
    retrieval_df = _build_retrieval_candidates(query_df, positive_db, topk=topk)

    pred_df = pd.read_csv(pred_path)
    if "cage_score" not in pred_df.columns and "pred" in pred_df.columns:
        pred_df = pred_df.rename(columns={"pred": "cage_score"})
    if "uniprot_id" not in pred_df.columns and "UniprotID" in pred_df.columns:
        pred_df["uniprot_id"] = pred_df["UniprotID"]
    if "reaction_id" not in pred_df.columns:
        raise ValueError(f"Prediction file missing reaction_id: {pred_path}")

    reaction_level_df = _reaction_level_from_gate(pair_df, pred_df, query_df, retrieval_df)
    write_table(reaction_level_df, out_csv, sep=",")

    eligible = reaction_level_df[reaction_level_df["n_positive_enzymes"] > 0].copy()
    if eligible.empty:
        eligible = reaction_level_df.copy()

    total_positive_pairs = int(eligible["n_positive_enzymes"].sum())
    covered_positive_pairs = int(eligible["n_positive_candidates_in_gate"].sum())
    gate_hit_rate = float(eligible["gate_hit"].mean()) if len(eligible) else 0.0
    pair_coverage = float(covered_positive_pairs / total_positive_pairs) if total_positive_pairs else 0.0
    mean_pair_coverage_per_reaction = (
        float(eligible["positive_pair_coverage"].dropna().astype(float).mean()) if eligible["positive_pair_coverage"].notna().any() else 0.0
    )

    scored_eligible = eligible.copy()
    metrics = {
        "similarity_topk": int(topk),
        "n_query_reactions": int(len(reaction_level_df)),
        "n_reactions_with_positive_label": int((reaction_level_df["n_positive_enzymes"] > 0).sum()),
        "n_reactions_without_positive_label": int((reaction_level_df["n_positive_enzymes"] == 0).sum()),
        "n_candidate_enzymes_in_full_pool": int(pd.read_csv(pred_path)["uniprot_id"].astype(str).nunique()),
        "n_retrieved_candidate_enzymes_total": int(reaction_level_df["n_candidates"].sum()),
        "n_scored_retrieved_candidate_enzymes_total": int(reaction_level_df["n_scored_candidates"].sum()),
        "n_positive_pairs_total": int(total_positive_pairs),
        "n_positive_pairs_covered_by_gate": int(covered_positive_pairs),
        "candidate_pool_hit_rate_by_reaction": gate_hit_rate,
        "positive_pair_coverage": pair_coverage,
        "mean_positive_pair_coverage_per_reaction": mean_pair_coverage_per_reaction,
        "n_reactions_with_scored_positive_candidates": int((reaction_level_df["n_scored_positive_candidates"] > 0).sum()),
        "n_reactions_missing_scored_positive_candidates": int(
            ((reaction_level_df["n_positive_enzymes"] > 0) & (reaction_level_df["n_scored_positive_candidates"] == 0)).sum()
        ),
        "top1_recall_after_gate": float(scored_eligible["top1_hit"].astype(bool).mean()) if len(scored_eligible) else 0.0,
        "top3_recall_after_gate": float(scored_eligible["top3_hit"].astype(bool).mean()) if len(scored_eligible) else 0.0,
        "top5_recall_after_gate": float(scored_eligible["top5_hit"].astype(bool).mean()) if len(scored_eligible) else 0.0,
        "top10_recall_after_gate": float(scored_eligible["top10_hit"].astype(bool).mean()) if len(scored_eligible) else 0.0,
        "mean_reciprocal_rank_after_gate": float(scored_eligible["reciprocal_rank"].astype(float).mean()) if len(scored_eligible) else 0.0,
        "median_best_positive_rank_after_gate": float(scored_eligible["best_positive_rank"].dropna().astype(float).median())
        if scored_eligible["best_positive_rank"].notna().any()
        else None,
    }

    safe_json_dump(metrics, out_json)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the original similarity gate for terpene screening.")
    parser.add_argument("--query_path", type=str, default=str(DEFAULT_QUERY_PATH))
    parser.add_argument("--pair_path", type=str, default=str(DEFAULT_PAIR_PATH))
    parser.add_argument("--positive_labels_path", type=str, default=str(DEFAULT_POSITIVE_LABELS_PATH))
    parser.add_argument("--pred_path", type=str, default=str(DEFAULT_PRED_PATH))
    parser.add_argument("--out_json", type=str, default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out_csv", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    metrics = analyze_similarity_gate(
        query_path=Path(args.query_path),
        pair_path=Path(args.pair_path),
        positive_labels_path=Path(args.positive_labels_path),
        pred_path=Path(args.pred_path),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        topk=args.topk,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
