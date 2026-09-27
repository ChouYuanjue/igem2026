from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.runtime.base_model import load_protein_features
from projects.active.terpene_screening.runtime.pair_protocol import (
    DEFAULT_BUDGETS,
    DEFAULT_EMBEDDINGS,
    DEFAULT_EXACT_FOLDS,
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_CLUSTERS,
    DEFAULT_REACTION_CLUSTERS,
    DEFAULT_STRICT_SPLITS,
    masked_rank_metrics,
    parse_int_tuple,
)
from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    MultiExpertConfig,
    parse_topk_terms,
    prepare_data,
    train_multi_expert,
)

DEFAULT_GENERAL_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
DEFAULT_GENERAL_METADATA = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
DEFAULT_OUTPUT = ROOT / "results/fibre_interaction_atlas_tps_broad_universe_v1"


def canonical_alias_map(metadata_path: Path) -> tuple[dict[str, str], set[str]]:
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    result: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in metadata[["protein_id", "aliases"]].itertuples(index=False):
        canonical = str(row.protein_id)
        result[canonical] = canonical
        for alias in str(row.aliases).split(";"):
            alias = alias.strip()
            if alias:
                if alias in ambiguous:
                    continue
                previous = result.get(alias)
                if previous is not None and previous != canonical:
                    result.pop(alias, None)
                    ambiguous.add(alias)
                    continue
                result[alias] = canonical
    return result, ambiguous


def mapped_ids(values: set[str], alias_map: dict[str, str]) -> set[str]:
    missing = sorted(value for value in values if value not in alias_map)
    if missing:
        raise ValueError(
            f"{len(missing)} positive proteins are absent from the general universe; "
            f"examples={missing[:10]}"
        )
    return {alias_map[value] for value in values}


def aggregate_methods(
    frame: pd.DataFrame,
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (protocol, method), group in frame.groupby(
        ["protocol", "method"], sort=True
    ):
        row: dict[str, object] = {
            "protocol": protocol,
            "method": method,
            "n_query_cells": int(len(group)),
            "n_unique_reactions": int(group["reaction_id"].nunique()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "median_best_positive_rank": float(
                group["best_positive_rank"].median()
            ),
            "mean_masked_known_positives": float(
                group["n_masked_known_positives"].mean()
            ),
        }
        for budget in budgets:
            row[f"hit_probability_at_{budget}"] = float(
                group[f"hit_at_{budget}"].mean()
            )
            row[f"expected_hits_at_{budget}"] = float(
                group[f"hits_at_{budget}"].mean()
            )
            row[f"precision_at_{budget}"] = float(
                group[f"precision_at_{budget}"].mean()
            )
            row[f"positive_recall_at_{budget}"] = float(
                group[f"positive_recall_at_{budget}"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the FIBRE TPS interaction atlas on strict held-out TPS queries "
            "against the full general_merged protein universe."
        )
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--general-proteins", type=Path, default=DEFAULT_GENERAL_PROTEINS)
    parser.add_argument("--general-metadata", type=Path, default=DEFAULT_GENERAL_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="20260723")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument(
        "--reaction-feature-mode",
        choices=["drfp_categorical", "multiview"],
        default="multiview",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--global-dim", type=int, default=128)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--expert-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--expert-mix-init", type=float, default=0.5)
    parser.add_argument("--hard-negative-k", type=int, default=0)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=20)
    parser.add_argument("--topk-terms", default="3:0.10,10:0.05,20:0.025")
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--glue-weight", type=float, default=0.02)
    parser.add_argument(
        "--strict-partition",
        choices=["all", "development", "frozen"],
        default="all",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = parse_int_tuple(args.seeds)
    budgets = parse_int_tuple(args.budgets)
    topk_terms = parse_topk_terms(args.topk_terms)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare_data only reads the fixed TPS reproduction assets.
    data = prepare_data(args)
    tps_protein_matrix = data["protein_matrix"]
    reaction_matrix = data["reaction_matrix"]
    reaction_ids = data["reaction_ids"]
    reaction_to_row = data["reaction_to_row"]
    pairs = data["pairs"]
    protein_groups = data["protein_groups"]
    reaction_groups = data["reaction_groups"]
    reaction_precursor_map = data["reaction_precursor_map"]
    reaction_skeleton_map = data["reaction_skeleton_map"]
    mechanism_values = data["mechanism_values"]

    general_matrix, general_ids = load_protein_features(args.general_proteins.resolve())
    if int(general_matrix.shape[1]) != int(tps_protein_matrix.shape[1]):
        raise ValueError(
            "TPS and general protein feature dimensions differ: "
            f"{tps_protein_matrix.shape[1]} vs {general_matrix.shape[1]}"
        )
    alias_map, ambiguous_aliases = canonical_alias_map(args.general_metadata.resolve())
    general_id_set = set(general_ids)
    if set(alias_map.values()) != general_id_set:
        raise ValueError("general metadata canonical IDs do not match embedding entries")

    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in pairs.groupby("rhea_id", sort=True)
    }
    all_positive_canonical = {
        reaction_id: mapped_ids(values, alias_map)
        for reaction_id, values in all_positive_by_reaction.items()
    }

    protein_tensor = torch.as_tensor(
        tps_protein_matrix, dtype=torch.float32, device=device
    )
    reaction_tensor = torch.as_tensor(
        reaction_matrix, dtype=torch.float32, device=device
    )
    general_tensor = torch.as_tensor(
        general_matrix, dtype=torch.float32, device=device
    )

    config = MultiExpertConfig(
        protein_input_dim=int(tps_protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        expert_mix_init=args.expert_mix_init,
    )

    records: list[dict[str, object]] = []
    partition_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            is_development = protein_fold == 4 or reaction_fold == 4
            if args.strict_partition == "development" and not is_development:
                continue
            if args.strict_partition == "frozen" and is_development:
                continue
            test_pairs = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
            ].copy()
            if test_pairs.empty:
                continue
            train_pairs = pairs[
                (pairs["protein_fold"] != protein_fold)
                & (pairs["reaction_fold"] != reaction_fold)
            ][["Entry", "rhea_id"]].drop_duplicates()
            query_ids = sorted(test_pairs["rhea_id"].astype(str).unique())
            query_rows = torch.as_tensor(
                [reaction_to_row[value] for value in query_ids],
                dtype=torch.long,
                device=device,
            )
            score_matrices: list[np.ndarray] = []
            universal_score_matrices: list[np.ndarray] = []
            partitions: list[np.ndarray] = []
            split_id = f"p{protein_fold}_r{reaction_fold}"
            for seed in seeds:
                model, history = train_multi_expert(
                    protein_tensor=protein_tensor,
                    reaction_tensor=reaction_tensor,
                    train_pairs=train_pairs,
                    protein_to_row=data["protein_to_row"],
                    reaction_to_row=reaction_to_row,
                    protein_groups=protein_groups,
                    reaction_groups=reaction_groups,
                    config=config,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    temperature=args.temperature,
                    reaction_loss_weight=args.reaction_loss_weight,
                    hard_negative_k=args.hard_negative_k,
                    hard_negative_start_epoch=args.hard_negative_start_epoch,
                    topk_terms=topk_terms,
                    topk_margin=args.topk_margin,
                    balance_weight=args.balance_weight,
                    entropy_weight=args.entropy_weight,
                    diversity_weight=args.diversity_weight,
                    glue_weight=args.glue_weight,
                    reaction_precursor_map=reaction_precursor_map,
                    reaction_skeleton_map=reaction_skeleton_map,
                    mechanism_values=mechanism_values,
                    mechanism_auxiliary_weight=0.0,
                    seed=seed,
                    device=device,
                )
                model.eval()
                with torch.no_grad():
                    score_matrix, _, diagnostics = model.score_matrices(
                        general_tensor, reaction_tensor[query_rows]
                    )
                score_matrices.append(score_matrix.cpu().numpy())
                universal_score_matrices.append(
                    (
                        diagnostics["reaction_global"]
                        @ diagnostics["protein_global"].T
                    )
                    .cpu()
                    .numpy()
                )
                partitions.append(diagnostics["r2e_partition"].cpu().numpy())
                training_records.append(
                    {
                        "split_id": split_id,
                        "seed": seed,
                        "n_train_pairs": len(train_pairs),
                        **history[-1],
                        "best_loss": min(item["loss"] for item in history),
                    }
                )
                del model, score_matrix, diagnostics
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            score_matrix = np.mean(score_matrices, axis=0)
            universal_score_matrix = np.mean(
                universal_score_matrices, axis=0
            )
            partition = np.mean(partitions, axis=0)
            local_query = {value: index for index, value in enumerate(query_ids)}

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = mapped_ids(set(group["Entry"].astype(str)), alias_map)
                known_other = all_positive_canonical[reaction_id] - positives
                query_index = local_query[reaction_id]
                for method, local_scores in (
                    ("full_atlas", score_matrix[query_index]),
                    (
                        "universal_chart_only",
                        universal_score_matrix[query_index],
                    ),
                ):
                    records.append(
                        {
                            "protocol": "tps_strict_double_cold_general_merged",
                            "method": method,
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction_id,
                            **masked_rank_metrics(
                                local_scores,
                                general_ids,
                                positives,
                                known_other,
                                budgets,
                            ),
                        }
                    )
                row = {
                    "protein_fold": protein_fold,
                    "reaction_fold": reaction_fold,
                    "reaction_id": reaction_id,
                    "global_chart": float(partition[query_index, 0]),
                }
                for expert_index in range(args.n_experts):
                    row[f"expert_{expert_index}"] = float(
                        partition[query_index, expert_index + 1]
                    )
                partition_records.append(row)

    query_metrics = pd.DataFrame(records)
    metrics = aggregate_methods(query_metrics, budgets)
    partition_table = pd.DataFrame(partition_records)
    training = pd.DataFrame(training_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    partition_table.to_csv(output_dir / "partition_weights.csv", index=False)
    training.to_csv(output_dir / "training_summary.csv", index=False)

    summary = {
        "schema": "fibre-interaction-atlas-tps-broad-universe-v1",
        "method": "FIBRE interaction atlas",
        "evaluation_scope": "reproduction_only",
        "protocol": (
            "TPS strict protein+reaction double-cold queries scored against the full "
            "general_merged 185,918-protein universe; exact-sequence canonical aliases "
            "preserve all TPS positives."
        ),
        "candidate_count": len(general_ids),
        "tps_positive_proteins": len(
            set(pairs["Entry"].astype(str))
        ),
        "tps_positive_canonical_proteins": len(
            mapped_ids(set(pairs["Entry"].astype(str)), alias_map)
        ),
        "general_universe_ambiguous_alias_count": len(ambiguous_aliases),
        "config": asdict(config),
        "epochs": args.epochs,
        "seeds": list(seeds),
        "reaction_feature_mode": args.reaction_feature_mode,
        "reaction_loss_weight": args.reaction_loss_weight,
        "glue_weight": args.glue_weight,
        "strict_partition": args.strict_partition,
        "universal_chart_in_partition": True,
        "same_model_universal_chart_comparator": True,
        "external_labels_used_for_tuning": False,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "partition_weights": str(output_dir / "partition_weights.csv"),
            "training_summary": str(output_dir / "training_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
