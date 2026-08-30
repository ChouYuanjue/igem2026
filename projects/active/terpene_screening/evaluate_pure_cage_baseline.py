from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_pure_cage_baseline_v1"
DEFAULT_SCORE_SOURCES = [
    ROOT / "results/terpene_cage_screen/all_rhea_gate/all_pair_scores.csv",
    ROOT / "results/terpene_cage_screen/predictions/terpene_candidate_pairs_epoch_9.csv",
    ROOT / "results/terpene_old_new_comparison/legacy_historical/fair_cage_all_scores.csv",
]
BUDGETS = (1, 3, 5, 10, 20)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"None of {candidates} is present in columns {list(frame.columns)}")


def load_cage_union(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for priority, path in enumerate(paths):
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=str)
        reaction_col = _pick_column(frame, ("reaction_id", "rhea_id"))
        protein_col = _pick_column(frame, ("uniprot_id", "UniprotID", "enzyme_id", "Entry"))
        score_col = _pick_column(frame, ("cage_score", "pred"))
        part = frame[[reaction_col, protein_col, score_col]].rename(
            columns={reaction_col: "reaction_id", protein_col: "uniprot_id", score_col: "cage_score"}
        )
        part["cage_score"] = pd.to_numeric(part["cage_score"], errors="coerce")
        part = part.dropna(subset=["reaction_id", "uniprot_id", "cage_score"]).copy()
        part["reaction_id"] = part["reaction_id"].astype(str)
        part["uniprot_id"] = part["uniprot_id"].astype(str)
        part["source"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        part["source_priority"] = priority
        parts.append(part)
    if not parts:
        raise FileNotFoundError("No usable CAGE score source was found")
    merged = pd.concat(parts, ignore_index=True)
    # Later sources are treated as higher-priority canonical outputs. They are not
    # averaged: all inputs are the same EnzymeCAGE checkpoint/protocol family and
    # averaging repeated inference would silently change the pure baseline.
    merged = merged.sort_values(
        ["reaction_id", "uniprot_id", "source_priority"], kind="mergesort"
    ).drop_duplicates(["reaction_id", "uniprot_id"], keep="last")
    merged = merged.drop(columns=["source_priority"]).reset_index(drop=True)
    return merged


def _query_metrics(
    scored: pd.DataFrame,
    positives: dict[str, set[str]],
    *,
    query_col: str,
    candidate_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows: list[dict[str, object]] = []
    scored_groups = {str(q): g.copy() for q, g in scored.groupby(query_col, sort=False)}
    all_queries = sorted(positives)
    for query_id in all_queries:
        known = positives[query_id]
        group = scored_groups.get(query_id)
        if group is None or group.empty:
            best_rank = None
            support_size = 0
            positive_support = 0
        else:
            group = group.sort_values(
                ["cage_score", candidate_col], ascending=[False, True], kind="mergesort"
            )
            ranked = group[candidate_col].astype(str).tolist()
            positive_ranks = [idx + 1 for idx, candidate in enumerate(ranked) if candidate in known]
            best_rank = min(positive_ranks) if positive_ranks else None
            support_size = len(group)
            positive_support = len(set(ranked) & known)
        row: dict[str, object] = {
            "query_id": query_id,
            "support_size": support_size,
            "known_positive_count": len(known),
            "scored_positive_count": positive_support,
            "positive_coverage_fraction": positive_support / len(known) if known else 0.0,
            "has_scored_positive": positive_support > 0,
            "best_positive_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        for budget in BUDGETS:
            row[f"hit_at_{budget}"] = float(best_rank is not None and best_rank <= budget)
        rows.append(row)
    per_query = pd.DataFrame(rows)
    covered = per_query[per_query["has_scored_positive"]].copy()
    summary: dict[str, float | int] = {
        "query_count_end_to_end": len(per_query),
        "query_count_native_evaluable": len(covered),
        "query_positive_coverage": float(per_query["has_scored_positive"].mean()),
        "mean_pair_positive_coverage": float(per_query["positive_coverage_fraction"].mean()),
        "median_candidate_support": float(per_query["support_size"].median()),
        "median_candidate_support_native": float(covered["support_size"].median()) if len(covered) else 0.0,
        "mrr_end_to_end": float(per_query["reciprocal_rank"].mean()),
        "mrr_native": float(covered["reciprocal_rank"].mean()) if len(covered) else 0.0,
    }
    for budget in BUDGETS:
        summary[f"hit_at_{budget}_end_to_end"] = float(per_query[f"hit_at_{budget}"].mean())
        summary[f"hit_at_{budget}_native"] = (
            float(covered[f"hit_at_{budget}"].mean()) if len(covered) else 0.0
        )
    return per_query, summary


def evaluate(score_paths: list[Path], positives_path: Path, output_dir: Path) -> dict[str, object]:
    cage = load_cage_union(score_paths)
    positives_frame = pd.read_csv(positives_path, sep="\t", dtype=str).fillna("")
    positives_frame = positives_frame[["rhea_id", "Entry"]].drop_duplicates()
    positive_pairs = set(map(tuple, positives_frame[["rhea_id", "Entry"]].itertuples(index=False, name=None)))
    cage["label"] = [
        int((reaction, protein) in positive_pairs)
        for reaction, protein in cage[["reaction_id", "uniprot_id"]].itertuples(index=False, name=None)
    ]

    r2e_positives = (
        positives_frame.groupby("rhea_id")["Entry"].apply(lambda values: set(values.astype(str))).to_dict()
    )
    e2r_positives = (
        positives_frame.groupby("Entry")["rhea_id"].apply(lambda values: set(values.astype(str))).to_dict()
    )
    r2e_query, r2e_summary = _query_metrics(
        cage, r2e_positives, query_col="reaction_id", candidate_col="uniprot_id"
    )
    e2r_query, e2r_summary = _query_metrics(
        cage, e2r_positives, query_col="uniprot_id", candidate_col="reaction_id"
    )
    r2e_query.insert(0, "direction", "reaction_to_enzyme")
    e2r_query.insert(0, "direction", "enzyme_to_reaction")

    output_dir.mkdir(parents=True, exist_ok=True)
    cage.sort_values(["reaction_id", "uniprot_id"]).to_csv(output_dir / "cage_score_union.csv", index=False)
    pd.concat([r2e_query, e2r_query], ignore_index=True).to_csv(output_dir / "query_metrics.csv", index=False)
    summary: dict[str, object] = {
        "protocol": "pure EnzymeCAGE on native scored support, with coverage reported separately",
        "score_source_count": len([path for path in score_paths if path.is_file()]),
        "scored_pair_count": len(cage),
        "scored_reaction_count": int(cage["reaction_id"].nunique()),
        "scored_protein_count": int(cage["uniprot_id"].nunique()),
        "scored_positive_pair_count": int(cage["label"].sum()),
        "positive_pair_count": len(positive_pairs),
        "positive_pair_coverage": float(cage["label"].sum() / len(positive_pairs)),
        "reaction_to_enzyme": r2e_summary,
        "enzyme_to_reaction": e2r_summary,
        "interpretation": {
            "native": "Ranking quality conditional on at least one known positive having a real CAGE score.",
            "end_to_end": "Same pure-CAGE ranking with unsupported queries counted as misses; this measures coverage plus ranking.",
            "missing_scores": "Never imputed with neural scores and never interpreted as negative EnzymeCAGE evidence.",
        },
        "sources": [str(path) for path in score_paths if path.is_file()],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pure EnzymeCAGE without treating unscored pairs as negative evidence.")
    parser.add_argument("--score-source", action="append", type=Path, default=None)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    score_paths = args.score_source or DEFAULT_SCORE_SOURCES
    summary = evaluate(score_paths, args.positives, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
