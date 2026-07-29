from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_current_pfam_architecture_reranking import (  # noqa: E402
    DEFAULT_PFAM,
    DEFAULT_POSITIVES,
    DEFAULT_RANKINGS,
    DEFAULT_STRICT,
    architecture_group,
    build_reaction_labels,
    choose_architecture_vocabulary,
    fit_predictors,
    normalize_pfam,
    tied_percentile,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_current_pfam_fixed_rankings"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed fold-local Pfam reranked Top-N lists.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--pfam-annotations", type=Path, default=DEFAULT_PFAM)
    parser.add_argument("--base-rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=0.05)
    parser.add_argument("--minimum-class-count", type=int, default=8)
    parser.add_argument("--maximum-classes", type=int, default=16)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if not 0 <= args.fusion_weight <= 1:
        raise ValueError("fusion weight must be inside [0, 1]")

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    reaction_features, reaction_ids, _, feature_schema = build_reaction_features(
        positives, "multiview"
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    strict = strict[["Entry", "rhea_id", "protein_fold", "reaction_fold"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )

    annotations = pd.read_csv(args.pfam_annotations, dtype=str).fillna("")
    annotations["pfam_combination"] = annotations.pfam_combination.map(normalize_pfam)
    vocabulary = choose_architecture_vocabulary(
        annotations,
        args.minimum_class_count,
        args.maximum_classes,
    )
    classes = vocabulary + ("__OTHER_PFAM__",)
    annotations["architecture_group"] = annotations.pfam_combination.map(
        lambda value: architecture_group(value, vocabulary)
    )
    protein_architecture = dict(
        zip(annotations.Entry.astype(str), annotations.architecture_group.astype(str))
    )
    class_to_column = {value: index for index, value in enumerate(classes)}

    rankings = pd.read_csv(args.base_rankings, dtype=str).fillna("")
    rankings = rankings[rankings.protocol.eq("double_cold_25cell")].copy()
    rankings[["protein_fold", "reaction_fold", "rank"]] = rankings[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    base_lists = {
        key: group.sort_values(["rank", "candidate_id"])[
            ["candidate_id", "rank"]
        ].copy()
        for key, group in rankings.groupby(keys, sort=True)
    }

    output_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            train_pairs = strict[
                strict.protein_fold.ne(protein_fold)
                & strict.reaction_fold.ne(reaction_fold)
            ]
            labels, labelled = build_reaction_labels(
                train_pairs,
                protein_architecture,
                reaction_ids,
                classes,
            )
            probabilities = fit_predictors(
                reaction_features,
                labels,
                labelled,
                args.c_value,
                args.seed + protein_fold * 100 + reaction_fold,
            )
            fit_rows.append(
                {
                    "protein_fold": protein_fold,
                    "reaction_fold": reaction_fold,
                    "partition": "development" if protein_fold == 4 or reaction_fold == 4 else "frozen",
                    "train_pairs": len(train_pairs),
                    "labelled_train_reactions": int(labelled.sum()),
                }
            )
            local_keys = [
                key for key in base_lists if key[:2] == (protein_fold, reaction_fold)
            ]
            for key in local_keys:
                reaction = key[2]
                base = base_lists[key]
                candidates = base.candidate_id.astype(str).tolist()
                base_percentile = tied_percentile(-base["rank"].to_numpy(float))
                reaction_probability = probabilities[reaction_to_row[reaction]]
                neutral = float(reaction_probability.mean())
                architecture_scores = np.asarray(
                    [
                        reaction_probability[
                            class_to_column[protein_architecture[candidate]]
                        ]
                        if protein_architecture.get(candidate, "") in class_to_column
                        else neutral
                        for candidate in candidates
                    ],
                    dtype=np.float32,
                )
                architecture_percentile = tied_percentile(architecture_scores)
                combined = (
                    (1.0 - args.fusion_weight) * base_percentile
                    + args.fusion_weight * architecture_percentile
                )
                order = np.lexsort((np.asarray(candidates), -combined))
                for rank, local_index in enumerate(order[: args.ranking_depth], start=1):
                    output_rows.append(
                        {
                            "partition": "development" if protein_fold == 4 or reaction_fold == 4 else "frozen",
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction,
                            "candidate_id": candidates[int(local_index)],
                            "rank": rank,
                            "score": float(combined[int(local_index)]),
                        }
                    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_dir / "rankings.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_audit.csv", index=False)
    summary = {
        "method": "fixed_fold_local_exact_pfam_reranking",
        "base_rankings": str(args.base_rankings.resolve()),
        "c_value": args.c_value,
        "fusion_weight": args.fusion_weight,
        "vocabulary": list(vocabulary),
        "classes": list(classes),
        "ranking_depth": args.ranking_depth,
        "reaction_feature_schema": feature_schema,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(output_rows).groupby("partition").size().to_string())


if __name__ == "__main__":
    main()
