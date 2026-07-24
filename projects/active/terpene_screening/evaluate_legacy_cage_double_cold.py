from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    GateSpec,
    TerpeneGateMatrix,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_CAGE = ROOT / "results/terpene_old_new_comparison/legacy_historical/fair_cage_all_scores.csv"
DEFAULT_STRICT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_old_new_comparison/legacy_cage_double_cold"
DEFAULT_BUDGETS = (5, 10, 20)
DEFAULT_RESCUE_SLOTS = {5: 1, 10: 5, 20: 10}
SEED = 20260707
RECALL_UNION_SPEC = GateSpec("recall_union_core", "recall_union", max_candidates=300)

NUMERIC_COLS = [
    "gate_score",
    "reaction_similarity",
    "sequence_kmer",
    "motif_score",
    "precursor_match",
    "product_skeleton",
    "mechanism",
    "evidence_channels",
    "cage_score_fill0",
    "cage_rank_score_fill0",
    "has_cage",
    "rxn_x_cage",
    "seq_x_cage",
    "motif_x_cage",
    "evidence_x_cage",
    "rxn_minus_cage",
    "cage_top_80",
    "cage_top_90",
    "cage_top_95",
]
CAT_COLS = ["gate_id"]


def parse_cells(value: str) -> list[tuple[int, int]]:
    if value.strip().lower() == "all":
        return [(protein_fold, reaction_fold) for protein_fold in range(5) for reaction_fold in range(5)]
    result: list[tuple[int, int]] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if not token.startswith("p") or "_r" not in token:
            raise ValueError(f"Invalid cell token: {part}")
        protein_text, reaction_text = token[1:].split("_r", 1)
        protein_fold = int(protein_text)
        reaction_fold = int(reaction_text)
        if protein_fold not in range(5) or reaction_fold not in range(5):
            raise ValueError(f"Cell outside [0,4]: {part}")
        result.append((protein_fold, reaction_fold))
    if not result:
        raise ValueError("No cells selected")
    return result


def clear_association_caches(matrix: TerpeneGateMatrix) -> None:
    matrix._seed_cache.clear()
    matrix._sequence_seed_score_cache.clear()
    matrix._precursor_gate_cache.clear()
    matrix._product_skeleton_gate_cache.clear()
    matrix._mechanism_gate_cache.clear()
    matrix._weighted_gate_cache.clear()
    matrix._built_gate_cache.clear()


def set_fold_state(
    matrix: TerpeneGateMatrix,
    train_positive_rows: pd.DataFrame,
    available_reactions: pd.DataFrame,
) -> None:
    matrix.positive_rows = train_positive_rows.copy()
    matrix.known_positive_ids = set(train_positive_rows["uniprot_id"].astype(str))
    matrix.true_map = (
        train_positive_rows.groupby("rhea_id")["uniprot_id"]
        .apply(lambda values: set(values.astype(str)))
        .to_dict()
    )
    matrix.reaction_to_positive_rows = {
        reaction_id: group.copy()
        for reaction_id, group in train_positive_rows.groupby("rhea_id", sort=False)
    }
    matrix.positive_seq_lookup = (
        train_positive_rows.drop_duplicates("uniprot_id")
        .set_index("uniprot_id")["sequence"]
        .astype(str)
        .to_dict()
    )
    matrix.reactions = available_reactions.copy()
    clear_association_caches(matrix)


def load_cage_lookup(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["cage_score"] = pd.to_numeric(frame.get("cage_score", 0), errors="coerce")
    lookup: dict[tuple[str, str], float] = {}
    for row in frame[["reaction_id", "uniprot_id", "cage_score"]].itertuples(index=False):
        lookup[(str(row.reaction_id), str(row.uniprot_id))] = (
            float(row.cage_score) if pd.notna(row.cage_score) else float("nan")
        )
    return lookup


def recompute_local_cage_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cage_rank_score"] = float("nan")
    available = result[result["cage_score"].notna()].copy()
    if available.empty:
        return result
    available = available.sort_values(
        ["cage_score", "uniprot_id"], ascending=[False, True], kind="mergesort"
    )
    denominator = max(1, len(available) - 1)
    available["cage_rank_score"] = 1.0 - np.arange(len(available)) / denominator
    result.loc[available.index, "cage_rank_score"] = available["cage_rank_score"]
    return result


def gate_rows_for_reaction(
    matrix: TerpeneGateMatrix,
    reaction_id: str,
    cage_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    candidates = matrix.build_gate_for_reaction(reaction_id, RECALL_UNION_SPEC)
    records: list[dict[str, object]] = []
    positives = matrix.true_map.get(reaction_id, set())
    for uniprot_id, scores in candidates.items():
        cage_score = cage_lookup.get((reaction_id, uniprot_id), float("nan"))
        motif = float(matrix.candidate_motif_scores.get(uniprot_id, 0.0))
        reaction_similarity = float(scores.get("reaction_similarity", 0.0))
        sequence_kmer = float(scores.get("sequence_kmer", 0.0))
        precursor_match = float(scores.get("precursor_match", 0.0))
        product_skeleton = float(scores.get("product_skeleton", 0.0))
        mechanism = float(scores.get("mechanism", 0.0))
        evidence_channels = int(reaction_similarity > 0)
        evidence_channels += int(sequence_kmer > 0)
        evidence_channels += int(precursor_match > 0)
        evidence_channels += int(product_skeleton > 0)
        evidence_channels += int(motif > 0)
        evidence_channels += int(mechanism > 0)
        records.append(
            {
                "reaction_id": reaction_id,
                "uniprot_id": uniprot_id,
                "gate_id": "recall_union_core",
                "label": int(uniprot_id in positives),
                "gate_score": float(
                    matrix.candidate_rerank_score(
                        reaction_id, uniprot_id, scores, "gate_score"
                    )
                ),
                "reaction_similarity": reaction_similarity,
                "sequence_kmer": sequence_kmer,
                "motif_score": motif,
                "precursor_match": precursor_match,
                "product_skeleton": product_skeleton,
                "mechanism": mechanism,
                "evidence_channels": evidence_channels,
                "cage_score": cage_score,
                "cage_rank_score": float("nan"),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return add_meta_features(frame)
    return add_meta_features(recompute_local_cage_ranks(frame))


def add_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "reaction_id",
                "uniprot_id",
                "gate_id",
                "label",
                *NUMERIC_COLS,
            ]
        )
    result = frame.copy()
    for column in [
        "label",
        "gate_score",
        "reaction_similarity",
        "sequence_kmer",
        "motif_score",
        "precursor_match",
        "product_skeleton",
        "mechanism",
        "evidence_channels",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["has_cage"] = result["cage_rank_score"].notna().astype(float)
    result["cage_score_fill0"] = pd.to_numeric(result["cage_score"], errors="coerce").fillna(0.0)
    result["cage_rank_score_fill0"] = pd.to_numeric(
        result["cage_rank_score"], errors="coerce"
    ).fillna(0.0)
    result["rxn_x_cage"] = result["reaction_similarity"] * result["cage_rank_score_fill0"]
    result["seq_x_cage"] = result["sequence_kmer"] * result["cage_rank_score_fill0"]
    result["motif_x_cage"] = result["motif_score"] * result["cage_rank_score_fill0"]
    result["evidence_x_cage"] = result["evidence_channels"] * result["cage_rank_score_fill0"]
    result["rxn_minus_cage"] = result["reaction_similarity"] - result["cage_rank_score_fill0"]
    result["cage_top_80"] = (result["cage_rank_score_fill0"] >= 0.80).astype(float)
    result["cage_top_90"] = (result["cage_rank_score_fill0"] >= 0.90).astype(float)
    result["cage_top_95"] = (result["cage_rank_score_fill0"] >= 0.95).astype(float)
    return result


def build_model(random_state: int, n_jobs: int, n_estimators: int) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=15,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def panel_for_query(
    rows: pd.DataFrame,
    *,
    budget: int,
    rescue_slots: int,
    masked_ids: set[str],
) -> list[str]:
    available = rows[~rows["uniprot_id"].isin(masked_ids)].copy()
    main_count = max(0, budget - rescue_slots)
    main = available.sort_values(
        ["reaction_similarity", "uniprot_id"], ascending=[False, True]
    ).head(main_count)
    used = set(main["uniprot_id"].astype(str))
    rescue = available[~available["uniprot_id"].isin(used)].sort_values(
        ["strict_rf_score", "uniprot_id"], ascending=[False, True]
    ).head(rescue_slots)
    return pd.concat([main, rescue], ignore_index=True)["uniprot_id"].astype(str).tolist()


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    row: dict[str, object] = {
        "protocol": "double_cold_25cell",
        "method": "legacy_fold_local_recall_union_core_plus_rf_rescue",
        "n_query_cells": int(len(frame)),
        "n_unique_reactions": int(frame["reaction_id"].nunique()),
        "mean_candidate_pool_size": float(frame["candidate_pool_size"].mean()),
        "empty_pool_rate": float((frame["candidate_pool_size"] == 0).mean()),
        "mean_masked_known_positives": float(frame["n_masked_known_positives"].mean()),
    }
    for budget in budgets:
        row[f"hit_probability_at_{budget}"] = float(frame[f"hit_at_{budget}"].mean())
        row[f"expected_hits_at_{budget}"] = float(frame[f"hits_at_{budget}"].mean())
        row[f"precision_at_{budget}"] = float(frame[f"precision_at_{budget}"].mean())
        row[f"positive_recall_at_{budget}"] = float(frame[f"positive_recall_at_{budget}"].mean())
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the original recall_union_core + RF rescue method with fold-local features."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cage", type=Path, default=DEFAULT_CAGE)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--cells", default="all")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--n-estimators", type=int, default=250)
    args = parser.parse_args()

    budgets = tuple(int(part.strip()) for part in args.budgets.split(",") if part.strip())
    unsupported = set(budgets) - set(DEFAULT_RESCUE_SLOTS)
    if unsupported:
        raise ValueError(f"No original-document rescue allocation for budgets {sorted(unsupported)}")
    cells = parse_cells(args.cells)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = TerpeneGateMatrix(
        args.positives.resolve(),
        args.candidates.resolve(),
        cage_score_path=None,
        exclude_same_reaction_smiles=True,
    )
    full_positive_rows = matrix.positive_rows.copy()
    full_reactions = matrix.reactions.copy()
    cage_lookup = load_cage_lookup(args.cage.resolve())

    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict["protein_fold"] = pd.to_numeric(strict["protein_fold"]).astype(int)
    strict["reaction_fold"] = pd.to_numeric(strict["reaction_fold"]).astype(int)
    strict = strict[
        [
            "Entry",
            "rhea_id",
            "protein_cluster",
            "reaction_cluster",
            "protein_fold",
            "reaction_fold",
        ]
    ].drop_duplicates(["Entry", "rhea_id"])
    fold_by_pair = strict[["Entry", "rhea_id", "protein_fold", "reaction_fold"]]
    positives_with_folds = full_positive_rows.merge(
        fold_by_pair,
        left_on=["uniprot_id", "rhea_id"],
        right_on=["Entry", "rhea_id"],
        how="left",
        validate="one_to_one",
    )
    if positives_with_folds[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict folds do not cover every positive association")
    positives_with_folds["protein_fold"] = positives_with_folds["protein_fold"].astype(int)
    positives_with_folds["reaction_fold"] = positives_with_folds["reaction_fold"].astype(int)

    cluster_fold = (
        strict[["protein_cluster", "protein_fold"]]
        .drop_duplicates()
        .set_index("protein_cluster")["protein_fold"]
        .astype(int)
        .to_dict()
    )
    clusters = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    cluster_by_entry = dict(zip(clusters["entry"].astype(str), clusters["cluster_id"].astype(str)))
    candidate_fold_by_id = {
        entry: cluster_fold.get(cluster_by_entry.get(entry, ""))
        for entry in matrix.candidate_ids
    }
    positives_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in strict.groupby("rhea_id", sort=True)
    }

    query_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []
    for protein_fold, reaction_fold in cells:
        train_positive_rows = positives_with_folds[
            (positives_with_folds["protein_fold"] != protein_fold)
            & (positives_with_folds["reaction_fold"] != reaction_fold)
        ][full_positive_rows.columns].copy()
        train_reaction_ids = sorted(set(train_positive_rows["rhea_id"].astype(str)))
        train_reactions = full_reactions[
            full_reactions["rhea_id"].astype(str).isin(train_reaction_ids)
        ].copy()
        set_fold_state(matrix, train_positive_rows, train_reactions)

        train_gate_frames: list[pd.DataFrame] = []
        for reaction_id in train_reaction_ids:
            local = gate_rows_for_reaction(matrix, reaction_id, cage_lookup)
            if not local.empty:
                local["candidate_protein_fold"] = local["uniprot_id"].map(candidate_fold_by_id)
                local = local[
                    local["candidate_protein_fold"].isna()
                    | (local["candidate_protein_fold"].astype(float) != float(protein_fold))
                ].copy()
                train_gate_frames.append(local)
        train_rows = (
            pd.concat(train_gate_frames, ignore_index=True)
            if train_gate_frames
            else pd.DataFrame()
        )
        if train_rows.empty or int(train_rows["label"].sum()) == 0:
            raise ValueError(f"No fold-local legacy training data for p{protein_fold}_r{reaction_fold}")
        model = build_model(
            SEED + 10 * protein_fold + reaction_fold,
            args.n_jobs,
            args.n_estimators,
        )
        model.fit(
            train_rows[NUMERIC_COLS + CAT_COLS],
            train_rows["label"].astype(int).to_numpy(),
        )

        target_pairs = strict[
            (strict["protein_fold"] == protein_fold)
            & (strict["reaction_fold"] == reaction_fold)
        ]
        for reaction_id, group in target_pairs.groupby("rhea_id", sort=True):
            available = pd.concat(
                [
                    train_reactions,
                    full_reactions[full_reactions["rhea_id"].astype(str).eq(str(reaction_id))],
                ],
                ignore_index=True,
            ).drop_duplicates("rhea_id")
            set_fold_state(matrix, train_positive_rows, available)
            local = gate_rows_for_reaction(matrix, str(reaction_id), cage_lookup)
            if not local.empty:
                local["strict_rf_score"] = model.predict_proba(
                    local[NUMERIC_COLS + CAT_COLS]
                )[:, 1]
            positives = set(group["Entry"].astype(str))
            known_other = positives_by_reaction.get(str(reaction_id), set()) - positives
            record: dict[str, object] = {
                "protocol": "double_cold_25cell",
                "method": "legacy_fold_local_recall_union_core_plus_rf_rescue",
                "protein_fold": protein_fold,
                "reaction_fold": reaction_fold,
                "reaction_id": str(reaction_id),
                "n_positives": len(positives),
                "n_masked_known_positives": len(known_other),
                "candidate_pool_size": int(len(local)),
            }
            for budget in budgets:
                panel = (
                    panel_for_query(
                        local,
                        budget=budget,
                        rescue_slots=DEFAULT_RESCUE_SLOTS[budget],
                        masked_ids=known_other,
                    )
                    if not local.empty
                    else []
                )
                hits = len(set(panel) & positives)
                record[f"hit_at_{budget}"] = int(hits > 0)
                record[f"hits_at_{budget}"] = int(hits)
                record[f"precision_at_{budget}"] = float(hits / budget)
                record[f"positive_recall_at_{budget}"] = float(hits / len(positives)) if positives else 0.0
                record[f"selected_at_{budget}"] = ";".join(panel)
            query_records.append(record)

        fit_records.append(
            {
                "protein_fold": protein_fold,
                "reaction_fold": reaction_fold,
                "n_train_positive_pairs": int(len(train_positive_rows)),
                "n_train_reactions": int(len(train_reaction_ids)),
                "n_train_gate_rows": int(len(train_rows)),
                "n_train_gate_positives": int(train_rows["label"].sum()),
                "n_test_query_cells": int(target_pairs["rhea_id"].nunique()),
            }
        )
        print(
            f"finished p{protein_fold}_r{reaction_fold}: "
            f"train_pairs={len(train_positive_rows)} train_gate_rows={len(train_rows)} "
            f"test_queries={target_pairs['rhea_id'].nunique()}",
            flush=True,
        )

    query_metrics = pd.DataFrame(query_records)
    metrics = aggregate(query_metrics, budgets)
    fit_summary = pd.DataFrame(fit_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    fit_summary.to_csv(output_dir / "fit_summary.csv", index=False)
    summary = {
        "method": "original_document_recall_union_core_plus_rf_rescue",
        "gate": "recall_union_core",
        "main_ranker": "reaction_similarity",
        "rescue_model": "RandomForest with original-document feature set",
        "n_estimators": args.n_estimators,
        "rescue_slots": {str(key): value for key, value in DEFAULT_RESCUE_SLOTS.items()},
        "protocol": "25-cell protein-cluster × reaction-cluster double-cold",
        "fold_local_regeneration": True,
        "train_only_association_features": True,
        "pairwise_cage_reused_without_labels": True,
        "cage_rank_recomputed_inside_fold_local_reservoir": True,
        "known_positive_masking": True,
        "cells": [f"p{protein_fold}_r{reaction_fold}" for protein_fold, reaction_fold in cells],
        "n_query_cells": int(len(query_metrics)),
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "fit_summary": str(output_dir / "fit_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
