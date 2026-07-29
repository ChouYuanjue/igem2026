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
    choose_architecture_vocabulary,
    fit_predictors,
    normalize_pfam,
    parse_float_tuple,
    parse_int_tuple,
    reciprocal_rank_and_hits,
    tied_percentile,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_current_pfam_hierarchical_reranking"


def split_domains(value: object) -> tuple[str, ...]:
    return tuple(sorted({part for part in str(value).split(";") if part}))


def choose_domain_vocabulary(
    annotations: pd.DataFrame,
    minimum_count: int,
    maximum_classes: int,
) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for domains in annotations["pfam_domains"]:
        for domain in domains:
            counts[domain] = counts.get(domain, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        domain
        for domain, count in ordered
        if count >= minimum_count
    )[:maximum_classes]


def build_hierarchical_labels(
    train_pairs: pd.DataFrame,
    protein_combinations: dict[str, str],
    protein_domains: dict[str, tuple[str, ...]],
    reaction_ids: list[str],
    combination_classes: tuple[str, ...],
    domain_classes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    combo_to_column = {value: index for index, value in enumerate(combination_classes)}
    domain_to_column = {value: index for index, value in enumerate(domain_classes)}
    combo_labels = np.zeros((len(reaction_ids), len(combination_classes)), dtype=np.int8)
    domain_labels = np.zeros((len(reaction_ids), len(domain_classes)), dtype=np.int8)
    combo_labelled = np.zeros(len(reaction_ids), dtype=bool)
    domain_labelled = np.zeros(len(reaction_ids), dtype=bool)
    for reaction, group in train_pairs.groupby("rhea_id", sort=False):
        row = reaction_to_row.get(str(reaction))
        if row is None:
            continue
        proteins = group.Entry.astype(str)
        combinations = {protein_combinations.get(value, "") for value in proteins} - {""}
        domains = {
            domain
            for protein in proteins
            for domain in protein_domains.get(protein, ())
            if domain in domain_to_column
        }
        if combinations:
            combo_labelled[row] = True
            for value in combinations:
                column = combo_to_column.get(value)
                if column is not None:
                    combo_labels[row, column] = 1
        if domains:
            domain_labelled[row] = True
            for value in domains:
                domain_labels[row, domain_to_column[value]] = 1
    return combo_labels, combo_labelled, domain_labels, domain_labelled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-local hierarchical single-domain plus Pfam-combination reranking."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--pfam-annotations", type=Path, default=DEFAULT_PFAM)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--fusion-weights", default="0,0.03,0.05,0.1,0.2")
    parser.add_argument("--combination-weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--budgets", default="10")
    parser.add_argument("--minimum-combination-count", type=int, default=8)
    parser.add_argument("--maximum-combinations", type=int, default=16)
    parser.add_argument("--minimum-domain-count", type=int, default=5)
    parser.add_argument("--maximum-domains", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    c_values = parse_float_tuple(args.c_values)
    fusion_weights = parse_float_tuple(args.fusion_weights)
    combination_weights = parse_float_tuple(args.combination_weights)
    budgets = parse_int_tuple(args.budgets)
    if any(not 0 <= value <= 1 for value in fusion_weights + combination_weights):
        raise ValueError("Fusion and combination weights must be inside [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    annotations["pfam_domains"] = annotations.pfam_combination.map(split_domains)
    combo_vocabulary = choose_architecture_vocabulary(
        annotations,
        args.minimum_combination_count,
        args.maximum_combinations,
    )
    combination_classes = combo_vocabulary + ("__OTHER_PFAM__",)
    domain_classes = choose_domain_vocabulary(
        annotations,
        args.minimum_domain_count,
        args.maximum_domains,
    )
    annotations["combination_group"] = annotations.pfam_combination.map(
        lambda value: value if value in combo_vocabulary else "__OTHER_PFAM__" if value else ""
    )
    protein_combinations = dict(
        zip(annotations.Entry.astype(str), annotations.combination_group.astype(str))
    )
    protein_domains = dict(
        zip(annotations.Entry.astype(str), annotations.pfam_domains)
    )

    rankings = pd.read_csv(args.rankings, dtype=str).fillna("")
    rankings = rankings[rankings.protocol.eq("double_cold_25cell")].copy()
    rankings[["protein_fold", "reaction_fold", "rank"]] = rankings[
        ["protein_fold", "reaction_fold", "rank"]
    ].astype(int)
    rankings["score"] = pd.to_numeric(rankings.score, errors="coerce").fillna(0.0)
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    base_lists = {
        key: group.sort_values(["rank", "candidate_id"])[
            ["candidate_id", "rank", "score"]
        ].copy()
        for key, group in rankings.groupby(keys, sort=True)
    }
    positives_by_query = {
        (pfold, rfold, str(reaction)): set(group.Entry.astype(str))
        for (pfold, rfold, reaction), group in strict.rename(
            columns={"rhea_id": "reaction_id"}
        ).groupby(keys, sort=True)
    }
    if set(base_lists) != set(positives_by_query):
        raise ValueError("Ranking and strict query sets differ")

    query_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            partition = "development" if protein_fold == 4 or reaction_fold == 4 else "frozen"
            train_pairs = strict[
                strict.protein_fold.ne(protein_fold)
                & strict.reaction_fold.ne(reaction_fold)
            ]
            combo_y, combo_labelled, domain_y, domain_labelled = build_hierarchical_labels(
                train_pairs,
                protein_combinations,
                protein_domains,
                reaction_ids,
                combination_classes,
                domain_classes,
            )
            fit_rows.append(
                {
                    "split_id": split_id,
                    "partition": partition,
                    "train_pairs": len(train_pairs),
                    "combo_labelled_reactions": int(combo_labelled.sum()),
                    "domain_labelled_reactions": int(domain_labelled.sum()),
                    "combination_classes": len(combination_classes),
                    "domain_classes": len(domain_classes),
                }
            )
            local_keys = [key for key in base_lists if key[:2] == (protein_fold, reaction_fold)]
            for c_value in c_values:
                combo_prob = fit_predictors(
                    reaction_features,
                    combo_y,
                    combo_labelled,
                    c_value,
                    args.seed + protein_fold * 100 + reaction_fold,
                )
                domain_prob = fit_predictors(
                    reaction_features,
                    domain_y,
                    domain_labelled,
                    c_value,
                    args.seed + 10000 + protein_fold * 100 + reaction_fold,
                )
                combo_to_column = {value: index for index, value in enumerate(combination_classes)}
                domain_to_column = {value: index for index, value in enumerate(domain_classes)}
                for combo_weight in combination_weights:
                    for fusion_weight in fusion_weights:
                        method = f"hier_c{c_value:g}_cw{combo_weight:g}_w{fusion_weight:g}"
                        for key in local_keys:
                            reaction = key[2]
                            row = reaction_to_row[reaction]
                            base = base_lists[key]
                            candidates = base.candidate_id.astype(str).tolist()
                            base_percentile = tied_percentile(-base["rank"].to_numpy(float))
                            combo_neutral = float(combo_prob[row].mean())
                            domain_neutral = float(domain_prob[row].mean())
                            architecture_scores: list[float] = []
                            for candidate in candidates:
                                combination = protein_combinations.get(candidate, "")
                                combo_score = (
                                    float(combo_prob[row, combo_to_column[combination]])
                                    if combination in combo_to_column
                                    else combo_neutral
                                )
                                columns = [
                                    domain_to_column[value]
                                    for value in protein_domains.get(candidate, ())
                                    if value in domain_to_column
                                ]
                                domain_score = (
                                    float(domain_prob[row, columns].mean())
                                    if columns
                                    else domain_neutral
                                )
                                architecture_scores.append(
                                    combo_weight * combo_score
                                    + (1.0 - combo_weight) * domain_score
                                )
                            architecture_percentile = tied_percentile(
                                np.asarray(architecture_scores, dtype=np.float32)
                            )
                            score = (
                                (1.0 - fusion_weight) * base_percentile
                                + fusion_weight * architecture_percentile
                            )
                            order = np.lexsort((np.asarray(candidates), -score))
                            ranked = [candidates[index] for index in order]
                            query_rows.append(
                                {
                                    "split_id": split_id,
                                    "partition": partition,
                                    "protein_fold": protein_fold,
                                    "reaction_fold": reaction_fold,
                                    "reaction_id": reaction,
                                    "c_value": c_value,
                                    "combination_weight": combo_weight,
                                    "fusion_weight": fusion_weight,
                                    "method": method,
                                    **reciprocal_rank_and_hits(
                                        ranked,
                                        positives_by_query[key],
                                        budgets,
                                    ),
                                }
                            )

    query_frame = pd.DataFrame(query_rows)
    selected_rows: list[dict[str, object]] = []
    frozen_rows: list[dict[str, object]] = []
    development = query_frame[query_frame.partition.eq("development")]
    for budget in budgets:
        column = f"hit_at_{budget}"
        summary = (
            development.groupby(
                ["c_value", "combination_weight", "fusion_weight", "method"],
                as_index=False,
            )
            .agg(hit_probability=(column, "mean"), mrr=("reciprocal_rank", "mean"))
            .sort_values(
                ["hit_probability", "mrr", "fusion_weight", "combination_weight", "c_value"],
                ascending=[False, False, True, True, True],
            )
        )
        selected = summary.iloc[0]
        selected_rows.append(
            {
                "budget": budget,
                "c_value": float(selected.c_value),
                "combination_weight": float(selected.combination_weight),
                "fusion_weight": float(selected.fusion_weight),
                "method": str(selected.method),
                "development_hit_probability": float(selected.hit_probability),
                "development_mrr": float(selected.mrr),
            }
        )
        frozen = query_frame[
            query_frame.partition.eq("frozen") & query_frame.method.eq(str(selected.method))
        ]
        baseline = query_frame[
            query_frame.partition.eq("frozen")
            & query_frame.c_value.eq(float(selected.c_value))
            & query_frame.combination_weight.eq(float(selected.combination_weight))
            & query_frame.fusion_weight.eq(0.0)
        ]
        merge_keys = ["split_id", "reaction_id"]
        paired = frozen[merge_keys + [column]].merge(
            baseline[merge_keys + [column]],
            on=merge_keys,
            suffixes=("_selected", "_baseline"),
            validate="one_to_one",
        )
        difference = paired[f"{column}_selected"] - paired[f"{column}_baseline"]
        frozen_rows.append(
            {
                "budget": budget,
                "selected_method": str(selected.method),
                "n_queries": len(paired),
                "baseline_hit_probability": float(paired[f"{column}_baseline"].mean()),
                "selected_hit_probability": float(paired[f"{column}_selected"].mean()),
                "difference": float(difference.mean()),
                "new_hits": int((difference == 1).sum()),
                "lost_hits": int((difference == -1).sum()),
            }
        )

    selected_frame = pd.DataFrame(selected_rows)
    frozen_frame = pd.DataFrame(frozen_rows)
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    selected_frame.to_csv(output_dir / "selected_parameters.csv", index=False)
    frozen_frame.to_csv(output_dir / "frozen_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fit_audit.csv", index=False)
    annotations.drop(columns=["pfam_domains"]).to_csv(
        output_dir / "candidate_pfam_hierarchy.csv", index=False
    )
    summary = {
        "method": "fold_local_hierarchical_pfam_reranking",
        "base_rankings": str(args.rankings.resolve()),
        "combination_classes": list(combination_classes),
        "domain_classes": list(domain_classes),
        "c_values": list(c_values),
        "combination_weights": list(combination_weights),
        "fusion_weights": list(fusion_weights),
        "reaction_feature_schema": feature_schema,
        "selection": "all parameters selected on the nine development cells; frozen cells never used for selection",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SELECTED")
    print(selected_frame.to_string(index=False))
    print("FROZEN")
    print(frozen_frame.to_string(index=False))


if __name__ == "__main__":
    main()
