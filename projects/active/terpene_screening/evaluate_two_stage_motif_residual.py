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

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    product_carbon_skeleton_signature,
)
from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (  # noqa: E402
    DEFAULT_EXACT_FOLDS,
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_CLUSTERS,
    DEFAULT_REACTION_CLUSTERS,
    DEFAULT_STRICT_SPLITS,
)
from projects.active.terpene_screening.evaluate_motif_channel_reranker import (  # noqa: E402
    DEFAULT_GLOBAL_EMBEDDINGS,
    DEFAULT_MOTIF_EMBEDDINGS,
    TrainingConfig,
    load_motif_blocks,
    score_candidates,
    train_model as train_motif_model,
)
from projects.active.terpene_screening.evaluate_multi_expert_protocol_comparison import (  # noqa: E402
    MultiExpertConfig,
    prepare_data,
    train_multi_expert,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_two_stage_motif_residual"
DEFAULT_BUDGETS = (3, 5, 10, 20)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one float")
    if any(item < 0 for item in result):
        raise ValueError("Residual scales must be non-negative")
    return result


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Expected positive integers")
    return result


def build_base_hard_triples(
    *,
    train_pairs: pd.DataFrame,
    base_scores: np.ndarray,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    protein_groups: dict[str, str],
    precursor_by_reaction: dict[str, str],
    skeleton_by_reaction: dict[str, str],
    shortlist_depth: int,
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine safe query-specific errors from the base model's fold-local shortlist."""
    known_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in train_pairs.groupby("rhea_id", sort=True)
    }
    classes_by_protein: dict[str, set[tuple[str, str]]] = {}
    for row in train_pairs[["Entry", "rhea_id"]].drop_duplicates().itertuples(index=False):
        label = (
            precursor_by_reaction.get(str(row.rhea_id), ""),
            skeleton_by_reaction.get(str(row.rhea_id), ""),
        )
        if all(label):
            classes_by_protein.setdefault(str(row.Entry), set()).add(label)
    identifiers = np.asarray(protein_ids)
    reaction_rows: list[int] = []
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    for reaction_id, group in train_pairs.groupby("rhea_id", sort=True):
        reaction_id = str(reaction_id)
        reaction_row = reaction_to_row.get(reaction_id)
        precursor = precursor_by_reaction.get(reaction_id, "")
        skeleton = skeleton_by_reaction.get(reaction_id, "")
        if reaction_row is None or not precursor or not skeleton:
            continue
        positives = set(group.Entry.astype(str))
        positive_groups = {protein_groups.get(value, value) for value in positives}
        order = np.lexsort((identifiers, -base_scores[reaction_row]))
        candidates: list[int] = []
        for candidate_row in order:
            candidate = protein_ids[int(candidate_row)]
            if candidate in positives:
                continue
            if protein_groups.get(candidate, candidate) in positive_groups:
                continue
            candidate_skeletons = {
                local_skeleton
                for local_precursor, local_skeleton in classes_by_protein.get(candidate, set())
                if local_precursor == precursor
            }
            if not candidate_skeletons or skeleton in candidate_skeletons:
                continue
            candidates.append(int(candidate_row))
            if len(candidates) >= shortlist_depth:
                break
        for positive in sorted(positives):
            positive_row = protein_to_row.get(positive)
            if positive_row is None:
                continue
            for candidate_row in candidates[:negatives_per_positive]:
                reaction_rows.append(reaction_row)
                positive_rows.append(positive_row)
                negative_rows.append(candidate_row)
    return (
        np.asarray(reaction_rows, dtype=np.int64),
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
    )


def rank_top_shortlist(
    base_scores: np.ndarray,
    residual_scores: np.ndarray,
    candidate_ids: list[str],
    masked_ids: set[str],
    shortlist_depth: int,
    scale: float,
) -> list[str]:
    adjusted = np.asarray(base_scores, dtype=np.float64).copy()
    id_to_row = {value: index for index, value in enumerate(candidate_ids)}
    for value in masked_ids:
        row = id_to_row.get(value)
        if row is not None:
            adjusted[row] = -np.inf
    order = [
        int(index)
        for index in np.lexsort((np.asarray(candidate_ids), -adjusted))
        if np.isfinite(adjusted[index])
    ][:shortlist_depth]
    combined = adjusted[order] + scale * np.asarray(residual_scores, dtype=np.float64)[order]
    local = np.lexsort((np.asarray([candidate_ids[index] for index in order]), -combined))
    return [candidate_ids[order[int(index)]] for index in local]


def query_metrics(ranking: list[str], positives: set[str], budgets: tuple[int, ...]) -> dict[str, float | int]:
    positions = [index + 1 for index, candidate in enumerate(ranking) if candidate in positives]
    best = min(positions) if positions else None
    row: dict[str, float | int] = {
        "best_positive_rank": float(best) if best is not None else np.nan,
        "reciprocal_rank": 0.0 if best is None else 1.0 / best,
        "n_positives": len(positives),
    }
    for budget in budgets:
        hits = sum(candidate in positives for candidate in ranking[:budget])
        row[f"hit_at_{budget}"] = int(hits > 0)
        row[f"hits_at_{budget}"] = hits
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage TPS motif residual reranker over an 8-expert Top-N shortlist.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_GLOBAL_EMBEDDINGS)
    parser.add_argument("--motif-embedding-dir", type=Path, default=DEFAULT_MOTIF_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-partition", choices=["development", "frozen"], default="development")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--residual-scales", default="0,0.05,0.1,0.2,0.3")
    parser.add_argument("--shortlist-depth", type=int, default=100)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
    parser.add_argument(
        "--residual-skeleton-source",
        choices=["coarse", "reaction_cluster", "carbon_graph"],
        default="coarse",
    )
    parser.add_argument("--base-epochs", type=int, default=80)
    parser.add_argument("--residual-epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.shortlist_depth <= 0 or args.negatives_per_positive <= 0:
        raise ValueError("shortlist depth and negatives per positive must be positive")
    budgets = parse_int_tuple(args.budgets)
    scales = parse_float_tuple(args.residual_scales)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare_data expects these standard evaluator attributes.
    args.reaction_feature_mode = "multiview"
    data = prepare_data(args)
    protein_matrix = np.asarray(data["protein_matrix"], dtype=np.float32)
    reaction_matrix = np.asarray(data["reaction_matrix"], dtype=np.float32)
    protein_ids = list(data["protein_ids"])
    reaction_ids = list(data["reaction_ids"])
    protein_to_row = dict(data["protein_to_row"])
    reaction_to_row = dict(data["reaction_to_row"])
    precursor_by_reaction = dict(data["reaction_precursor_map"])
    coarse_skeleton_by_reaction = dict(data["reaction_skeleton_map"])
    reaction_cluster_by_reaction = dict(data["reaction_groups"])
    reaction_table = data["reaction_table"]
    carbon_graph_by_reaction = dict(
        zip(
            reaction_table["rhea_id"].astype(str),
            reaction_table["canonical_reaction"].astype(str).map(
                product_carbon_skeleton_signature
            ),
        )
    )
    skeleton_by_reaction = (
        reaction_cluster_by_reaction
        if args.residual_skeleton_source == "reaction_cluster"
        else carbon_graph_by_reaction
        if args.residual_skeleton_source == "carbon_graph"
        else coarse_skeleton_by_reaction
    )
    pairs = data["pairs"].copy()
    protein_groups = dict(data["protein_groups"])
    reaction_groups = dict(data["reaction_groups"])
    motif_matrix, descriptors, presence, motif_schema = load_motif_blocks(
        args.motif_embedding_dir.resolve(), protein_ids
    )
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    base_config = MultiExpertConfig(
        protein_input_dim=protein_matrix.shape[1],
        reaction_input_dim=reaction_matrix.shape[1],
        hidden_dim=512,
        global_dim=128,
        n_experts=8,
        expert_dim=32,
        dropout=0.1,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    residual_config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=0.1,
        epochs=args.residual_epochs,
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=1e-4,
        margin=args.margin,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    all_positive_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in pairs.groupby("rhea_id", sort=True)
    }

    candidate_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    completed = 0
    stop = False
    for protein_fold in range(5):
        for reaction_fold in range(5):
            development = protein_fold == 4 or reaction_fold == 4
            if args.strict_partition == "development" and not development:
                continue
            if args.strict_partition == "frozen" and development:
                continue
            if args.max_cells and completed >= args.max_cells:
                stop = True
                break
            train_pairs = pairs[
                (pairs.protein_fold != protein_fold) & (pairs.reaction_fold != reaction_fold)
            ][["Entry", "rhea_id"]].drop_duplicates()
            test_pairs = pairs[
                (pairs.protein_fold == protein_fold) & (pairs.reaction_fold == reaction_fold)
            ].copy()
            if test_pairs.empty:
                completed += 1
                continue
            base_model, base_history = train_multi_expert(
                protein_tensor=protein_tensor,
                reaction_tensor=reaction_tensor,
                train_pairs=train_pairs,
                protein_to_row=protein_to_row,
                reaction_to_row=reaction_to_row,
                protein_groups=protein_groups,
                reaction_groups=reaction_groups,
                config=base_config,
                epochs=args.base_epochs,
                learning_rate=3e-4,
                weight_decay=1e-4,
                temperature=0.07,
                reaction_loss_weight=0.75,
                hard_negative_k=0,
                hard_negative_start_epoch=20,
                topk_terms=((3, 0.10), (10, 0.05), (20, 0.025)),
                topk_margin=0.0,
                balance_weight=0.05,
                entropy_weight=0.005,
                diversity_weight=0.01,
                reaction_precursor_map=precursor_by_reaction,
                reaction_skeleton_map=skeleton_by_reaction,
                mechanism_values=(),
                mechanism_auxiliary_weight=0.0,
                seed=args.seed,
                device=device,
            )
            base_model.eval()
            with torch.no_grad():
                base_scores_t, _, _ = base_model.score_matrices(protein_tensor, reaction_tensor)
            base_scores = base_scores_t.cpu().numpy()
            triples = build_base_hard_triples(
                train_pairs=train_pairs,
                base_scores=base_scores,
                protein_ids=protein_ids,
                reaction_ids=reaction_ids,
                protein_to_row=protein_to_row,
                reaction_to_row=reaction_to_row,
                protein_groups=protein_groups,
                precursor_by_reaction=precursor_by_reaction,
                skeleton_by_reaction=skeleton_by_reaction,
                shortlist_depth=args.shortlist_depth,
                negatives_per_positive=args.negatives_per_positive,
            )
            if not len(triples[0]):
                raise ValueError(f"No base-hard residual triples for p{protein_fold}_r{reaction_fold}")
            residual_model, residual_history = train_motif_model(
                reaction_features=reaction_matrix,
                motif_features=motif_matrix,
                descriptors=descriptors,
                presence=presence,
                triples=triples,
                config=residual_config,
                device=device,
            )
            split_id = f"p{protein_fold}_r{reaction_fold}"
            training_rows.append(
                {
                    "split_id": split_id,
                    "n_train_pairs": len(train_pairs),
                    "n_residual_triples": len(triples[0]),
                    "base_final_loss": base_history[-1]["loss"],
                    "base_best_loss": min(row["loss"] for row in base_history),
                    "residual_final_loss": residual_history[-1]["mean_loss"],
                    "residual_best_loss": min(row["mean_loss"] for row in residual_history),
                }
            )
            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                reaction_id = str(reaction_id)
                positives = set(group.Entry.astype(str))
                known_other = all_positive_by_reaction.get(reaction_id, set()) - positives
                reaction_row = reaction_to_row[reaction_id]
                residual_scores = score_candidates(
                    residual_model,
                    reaction_matrix,
                    motif_matrix,
                    descriptors,
                    presence,
                    reaction_row,
                    device,
                )
                adjusted = base_scores[reaction_row].copy()
                for candidate in known_other:
                    row = protein_to_row.get(candidate)
                    if row is not None:
                        adjusted[row] = -np.inf
                base_order = [
                    int(index)
                    for index in np.lexsort((np.asarray(protein_ids), -adjusted))
                    if np.isfinite(adjusted[index])
                ][: args.shortlist_depth]
                for rank, protein_row in enumerate(base_order, start=1):
                    candidate_rows.append(
                        {
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction_id,
                            "candidate_id": protein_ids[protein_row],
                            "base_rank": rank,
                            "base_score": float(base_scores[reaction_row, protein_row]),
                            "residual_score": float(residual_scores[protein_row]),
                            "is_positive": int(protein_ids[protein_row] in positives),
                        }
                    )
            completed += 1
        if stop:
            break

    candidates = pd.DataFrame(candidate_rows)
    training = pd.DataFrame(training_rows)
    candidates.to_csv(output_dir / "candidate_scores.csv", index=False)
    training.to_csv(output_dir / "training_summary.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    keys = ["protein_fold", "reaction_fold", "reaction_id"]
    for scale in scales:
        scale_queries: list[dict[str, object]] = []
        for key, group in candidates.groupby(keys, sort=True):
            group = group.copy()
            group["combined"] = group.base_score + scale * group.residual_score
            group = group.sort_values(["combined", "candidate_id"], ascending=[False, True])
            ranking = group.candidate_id.astype(str).tolist()
            positives = set(group.loc[group.is_positive.eq(1), "candidate_id"].astype(str))
            row = {**dict(zip(keys, key)), "scale": scale, **query_metrics(ranking, positives, budgets)}
            scale_queries.append(row)
            query_rows.append(row)
        frame = pd.DataFrame(scale_queries)
        metric: dict[str, object] = {
            "scale": scale,
            "n_queries": len(frame),
            "mrr": frame.reciprocal_rank.mean(),
            "median_rank": frame.best_positive_rank.median(),
        }
        for budget in budgets:
            metric[f"hit{budget}"] = frame[f"hit_at_{budget}"].mean()
        metric["pareto_score"] = sum(metric[f"hit{budget}"] for budget in budgets) + metric["mrr"]
        metric_rows.append(metric)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["pareto_score", "mrr", "scale"], ascending=[False, False, True]
    )
    pd.DataFrame(query_rows).to_csv(output_dir / "query_metrics_by_scale.csv", index=False)
    metrics.to_csv(output_dir / "scale_metrics.csv", index=False)
    summary = {
        "method": "eight_expert_topn_then_motif_residual",
        "strict_partition": args.strict_partition,
        "base_config": asdict(base_config),
        "base_epochs": args.base_epochs,
        "residual_config": asdict(residual_config),
        "shortlist_depth": args.shortlist_depth,
        "residual_skeleton_source": args.residual_skeleton_source,
        "residual_scales": list(scales),
        "completed_cells": completed,
        "motif_schema": motif_schema,
        "selected_by_pareto": metrics.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
