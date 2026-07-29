from __future__ import annotations

import argparse
import hashlib
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

from projects.active.terpene_screening.evaluate_multi_expert_protocol_comparison import (  # noqa: E402
    DirectionalMultiExpertDualTower,
    MultiExpertConfig,
    parse_topk_terms,
    train_multi_expert,
)
from projects.active.terpene_screening.gate_matrix import canonical_or_raw_reaction  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    load_protein_features,
    rank_metrics,
)

CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
CURRENT_SEQUENCES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
CURRENT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
CURRENT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
MARTS_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OUTPUT = ROOT / "results/terpene_multi_expert_marts_adaptation"
DEFAULT_BUDGETS = (3, 5, 10, 20)


def parse_str_tuple(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one value")
    return result


def exact_group(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


def build_unified_reaction_features(
    current_positives: pd.DataFrame,
    marts_reactions: pd.DataFrame,
    feature_mode: str,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, object]]:
    current = current_positives[["rhea_id", "smiles_seq"]].drop_duplicates("rhea_id")
    external = marts_reactions[["reaction_id", "reaction_smiles"]].rename(
        columns={"reaction_id": "rhea_id", "reaction_smiles": "smiles_seq"}
    )
    combined = pd.concat([current, external], ignore_index=True)
    if combined["rhea_id"].duplicated().any():
        duplicates = combined.loc[combined["rhea_id"].duplicated(), "rhea_id"].tolist()
        raise ValueError(f"Current and MARTS reaction identifiers overlap: {duplicates[:5]}")
    return build_reaction_features(combined, feature_mode)


def build_combined_protein_groups(
    current_ids: list[str],
    current_sequences_path: Path,
    current_clusters_path: Path,
    marts_proteins: pd.DataFrame,
    marts_pairs: pd.DataFrame,
) -> dict[str, str]:
    current_sequences = pd.read_csv(current_sequences_path, sep="\t", dtype=str).fillna("")
    current_sequences["normalized_sequence"] = (
        current_sequences["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    )
    current_sequence_by_id = dict(
        zip(current_sequences["Entry"].astype(str), current_sequences["normalized_sequence"])
    )
    marts_sequence_by_id = dict(
        zip(
            marts_proteins["protein_id"].astype(str),
            marts_proteins["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper(),
        )
    )
    sequence_counts = pd.Series(
        [current_sequence_by_id.get(value, "") for value in current_ids]
        + list(marts_sequence_by_id.values())
    ).value_counts()
    duplicated_sequences = {value for value, count in sequence_counts.items() if value and count > 1}

    current_clusters = pd.read_csv(current_clusters_path, dtype=str).fillna("")
    current_cluster_by_id = dict(
        zip(current_clusters["entry"].astype(str), current_clusters["cluster_id"].astype(str))
    )
    marts_cluster_by_id = dict(
        zip(marts_pairs["Entry"].astype(str), marts_pairs["protein_cluster"].astype(str))
    )
    groups: dict[str, str] = {}
    for protein_id in current_ids:
        sequence = current_sequence_by_id.get(protein_id, "")
        groups[protein_id] = (
            exact_group("EXACTSEQ", sequence)
            if sequence in duplicated_sequences
            else f"CURRENT::{current_cluster_by_id.get(protein_id, protein_id)}"
        )
    for protein_id, sequence in marts_sequence_by_id.items():
        groups[protein_id] = (
            exact_group("EXACTSEQ", sequence)
            if sequence in duplicated_sequences
            else f"MARTS::{marts_cluster_by_id.get(protein_id, protein_id)}"
        )
    return groups


def build_combined_reaction_groups(
    reaction_table: pd.DataFrame,
    current_clusters_path: Path,
    marts_pairs: pd.DataFrame,
) -> dict[str, str]:
    canonical_counts = reaction_table["canonical_reaction"].astype(str).value_counts()
    duplicated = {value for value, count in canonical_counts.items() if value and count > 1}
    current_clusters = pd.read_csv(current_clusters_path, dtype=str).fillna("")
    current_cluster_by_id = dict(
        zip(
            current_clusters["reaction_id"].astype(str),
            current_clusters["reaction_cluster"].astype(str),
        )
    )
    marts_cluster_by_id = dict(
        zip(marts_pairs["rhea_id"].astype(str), marts_pairs["reaction_cluster"].astype(str))
    )
    groups: dict[str, str] = {}
    for row in reaction_table[["rhea_id", "canonical_reaction"]].itertuples(index=False):
        reaction_id = str(row.rhea_id)
        canonical = str(row.canonical_reaction)
        if canonical in duplicated:
            groups[reaction_id] = exact_group("EXACTRXN", canonical)
        elif reaction_id.startswith("MARTS_RXN_"):
            groups[reaction_id] = f"MARTS::{marts_cluster_by_id.get(reaction_id, reaction_id)}"
        else:
            groups[reaction_id] = f"CURRENT::{current_cluster_by_id.get(reaction_id, reaction_id)}"
    return groups


def evaluate_test_pairs(
    records: list[dict[str, object]],
    ranking_rows: list[dict[str, object]] | None,
    ranking_depth: int,
    method: str,
    split_id: str,
    r2e_scores: np.ndarray,
    e2r_scores: np.ndarray,
    test_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    budgets: tuple[int, ...],
) -> None:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        reaction_id = str(reaction_id)
        positives = set(group["Entry"].astype(str))
        metrics = rank_metrics(
            r2e_scores[reaction_to_row[reaction_id]], protein_ids, positives, set(), budgets
        )
        records.append(
            {
                "method": method,
                "split_id": split_id,
                "direction": "reaction_to_enzyme",
                "query_id": reaction_id,
                **metrics,
            }
        )
        if ranking_rows is not None and ranking_depth > 0:
            scores = r2e_scores[reaction_to_row[reaction_id]]
            order = np.lexsort((np.asarray(protein_ids), -scores))[:ranking_depth]
            ranking_rows.extend(
                {
                    "method": method,
                    "split_id": split_id,
                    "direction": "reaction_to_enzyme",
                    "query_id": reaction_id,
                    "rank": rank,
                    "candidate_id": protein_ids[int(index)],
                    "score": float(scores[int(index)]),
                    "is_positive": int(protein_ids[int(index)] in positives),
                }
                for rank, index in enumerate(order, start=1)
            )
    for protein_id, group in test_pairs.groupby("Entry", sort=True):
        protein_id = str(protein_id)
        positives = set(group["rhea_id"].astype(str))
        metrics = rank_metrics(
            e2r_scores[:, protein_to_row[protein_id]], reaction_ids, positives, set(), budgets
        )
        records.append(
            {
                "method": method,
                "split_id": split_id,
                "direction": "enzyme_to_reaction",
                "query_id": protein_id,
                **metrics,
            }
        )
        if ranking_rows is not None and ranking_depth > 0:
            scores = e2r_scores[:, protein_to_row[protein_id]]
            order = np.lexsort((np.asarray(reaction_ids), -scores))[:ranking_depth]
            ranking_rows.extend(
                {
                    "method": method,
                    "split_id": split_id,
                    "direction": "enzyme_to_reaction",
                    "query_id": protein_id,
                    "rank": rank,
                    "candidate_id": reaction_ids[int(index)],
                    "score": float(scores[int(index)]),
                    "is_positive": int(reaction_ids[int(index)] in positives),
                }
                for rank, index in enumerate(order, start=1)
            )


def score_marts(
    model: DirectionalMultiExpertDualTower,
    marts_proteins: torch.Tensor,
    marts_reactions: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        r2e_scores, e2r_scores, _ = model.score_matrices(marts_proteins, marts_reactions)
    return r2e_scores.cpu().numpy(), e2r_scores.cpu().numpy()


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}",
            "mean",
        )
    return frame.groupby(["method", "direction"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Current-pretrained directional multi-expert adaptation on MARTS external double-cold folds."
    )
    parser.add_argument("--current-positives", type=Path, default=CURRENT_POSITIVES)
    parser.add_argument("--current-protein-dir", type=Path, default=CURRENT_PROTEINS)
    parser.add_argument("--current-sequences", type=Path, default=CURRENT_SEQUENCES)
    parser.add_argument("--current-protein-clusters", type=Path, default=CURRENT_PROTEIN_CLUSTERS)
    parser.add_argument("--current-reaction-clusters", type=Path, default=CURRENT_REACTION_CLUSTERS)
    parser.add_argument("--marts-cache", type=Path, default=MARTS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-mode", choices=["drfp_categorical", "multiview"], default="multiview")
    parser.add_argument("--adaptation-modes", default="marts_only,joint_current_marts")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--adapt-epochs", type=int, default=50)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapt-learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--hard-negative-start-epoch", type=int, default=16)
    parser.add_argument("--topk-terms", default="3:0.10,10:0.05,20:0.025")
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--ranking-depth", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    adaptation_modes = parse_str_tuple(args.adaptation_modes)
    unknown_modes = set(adaptation_modes) - {"marts_only", "joint_current_marts"}
    if unknown_modes:
        raise ValueError(f"Unknown adaptation modes: {sorted(unknown_modes)}")
    topk_terms = parse_topk_terms(args.topk_terms)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_protein_matrix, current_protein_ids = load_protein_features(
        args.current_protein_dir.resolve()
    )
    current_positives = pd.read_csv(args.current_positives, sep="\t", dtype=str).fillna("")
    current_positives = current_positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    current_positives = current_positives[
        current_positives["Entry"].isin(set(current_protein_ids))
    ].copy()

    cache = args.marts_cache.resolve()
    marts_protein_matrix = np.load(cache / "protein_features.npy").astype(np.float32)
    marts_proteins = pd.read_csv(cache / "protein_entities.csv", dtype=str).fillna("")
    marts_protein_ids = marts_proteins["protein_id"].astype(str).tolist()
    if len(marts_protein_matrix) != len(marts_protein_ids):
        raise ValueError("MARTS protein feature matrix and entity table differ")
    marts_reactions = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    marts_pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    for column in ("protein_fold", "reaction_fold"):
        marts_pairs[column] = pd.to_numeric(marts_pairs[column]).astype(int)
    for column in ("protein_seen", "reaction_seen"):
        marts_pairs[column] = marts_pairs[column].astype(str).str.lower().eq("true")

    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_unified_reaction_features(
        current_positives, marts_reactions, args.feature_mode
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    current_reaction_ids = sorted(set(current_positives["rhea_id"].astype(str)))
    marts_reaction_ids = marts_reactions["reaction_id"].astype(str).tolist()
    missing_reactions = (set(current_reaction_ids) | set(marts_reaction_ids)) - set(reaction_to_row)
    if missing_reactions:
        raise ValueError(f"Unified reaction features miss identifiers: {sorted(missing_reactions)[:5]}")

    current_protein_to_row = {value: index for index, value in enumerate(current_protein_ids)}
    combined_protein_ids = current_protein_ids + marts_protein_ids
    if len(set(combined_protein_ids)) != len(combined_protein_ids):
        raise ValueError("Current and MARTS protein identifiers overlap")
    combined_protein_matrix = np.concatenate(
        [current_protein_matrix, marts_protein_matrix], axis=0
    ).astype(np.float32)
    combined_protein_to_row = {
        value: index for index, value in enumerate(combined_protein_ids)
    }

    protein_groups = build_combined_protein_groups(
        current_protein_ids,
        args.current_sequences.resolve(),
        args.current_protein_clusters.resolve(),
        marts_proteins,
        marts_pairs,
    )
    reaction_groups = build_combined_reaction_groups(
        reaction_table,
        args.current_reaction_clusters.resolve(),
        marts_pairs,
    )

    config = MultiExpertConfig(
        protein_input_dim=int(combined_protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        expert_mix_init=args.expert_mix_init,
    )
    combined_protein_tensor = torch.as_tensor(
        combined_protein_matrix, dtype=torch.float32, device=device
    )
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    marts_protein_rows = torch.as_tensor(
        [combined_protein_to_row[value] for value in marts_protein_ids],
        dtype=torch.long,
        device=device,
    )
    marts_reaction_rows = torch.as_tensor(
        [reaction_to_row[value] for value in marts_reaction_ids],
        dtype=torch.long,
        device=device,
    )

    current_pairs = current_positives[["Entry", "rhea_id"]].drop_duplicates().copy()
    pretrained_model, pretrain_history = train_multi_expert(
        protein_tensor=combined_protein_tensor,
        reaction_tensor=reaction_tensor,
        train_pairs=current_pairs,
        protein_to_row=combined_protein_to_row,
        reaction_to_row=reaction_to_row,
        protein_groups=protein_groups,
        reaction_groups=reaction_groups,
        config=config,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_learning_rate,
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
        seed=args.seed,
        device=device,
    )
    pretrained_state = {
        name: value.detach().cpu().clone()
        for name, value in pretrained_model.state_dict().items()
    }
    baseline_r2e, baseline_e2r = score_marts(
        pretrained_model,
        combined_protein_tensor[marts_protein_rows],
        reaction_tensor[marts_reaction_rows],
    )

    records: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    model_dir = output_dir / "models"
    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": pretrained_state,
                "model_config": asdict(config),
                "feature_schema": feature_schema,
                "seed": args.seed,
            },
            model_dir / "current_pretrained.pt",
        )

    for split_index, (protein_fold, reaction_fold) in enumerate(
        (p, r) for p in range(5) for r in range(5)
    ):
        split_id = f"p{protein_fold}_r{reaction_fold}"
        marts_train = marts_pairs[
            (marts_pairs["protein_fold"] != protein_fold)
            & (marts_pairs["reaction_fold"] != reaction_fold)
        ][["Entry", "rhea_id"]].drop_duplicates()
        test_pairs = marts_pairs[
            (marts_pairs["protein_fold"] == protein_fold)
            & (marts_pairs["reaction_fold"] == reaction_fold)
            & (~marts_pairs["protein_seen"])
            & (~marts_pairs["reaction_seen"])
        ][["Entry", "rhea_id"]].drop_duplicates()
        split_rows.append(
            {
                "split_id": split_id,
                "marts_train_pairs": len(marts_train),
                "test_pairs": len(test_pairs),
                "test_proteins": test_pairs["Entry"].nunique(),
                "test_reactions": test_pairs["rhea_id"].nunique(),
            }
        )
        if test_pairs.empty:
            continue
        evaluate_test_pairs(
            records,
            ranking_rows if args.ranking_depth > 0 else None,
            args.ranking_depth,
            "current_pretrained_multi_expert",
            split_id,
            baseline_r2e,
            baseline_e2r,
            test_pairs,
            marts_protein_ids,
            marts_reaction_ids,
            budgets,
        )
        for mode_index, mode in enumerate(adaptation_modes):
            train_pairs = (
                marts_train
                if mode == "marts_only"
                else pd.concat([current_pairs, marts_train], ignore_index=True).drop_duplicates(
                    ["Entry", "rhea_id"]
                )
            )
            model, history = train_multi_expert(
                protein_tensor=combined_protein_tensor,
                reaction_tensor=reaction_tensor,
                train_pairs=train_pairs,
                protein_to_row=combined_protein_to_row,
                reaction_to_row=reaction_to_row,
                protein_groups=protein_groups,
                reaction_groups=reaction_groups,
                config=config,
                epochs=args.adapt_epochs,
                learning_rate=args.adapt_learning_rate,
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
                seed=args.seed + split_index * 100 + mode_index,
                device=device,
                initial_state_dict=pretrained_state,
            )
            r2e_scores, e2r_scores = score_marts(
                model,
                combined_protein_tensor[marts_protein_rows],
                reaction_tensor[marts_reaction_rows],
            )
            method = f"multi_expert_{mode}"
            evaluate_test_pairs(
                records,
                ranking_rows if args.ranking_depth > 0 else None,
                args.ranking_depth,
                method,
                split_id,
                r2e_scores,
                e2r_scores,
                test_pairs,
                marts_protein_ids,
                marts_reaction_ids,
                budgets,
            )
            history_frame = pd.DataFrame(history)
            history_frame.insert(0, "adaptation_mode", mode)
            history_frame.insert(0, "split_id", split_id)
            histories.append(history_frame)
            if args.save_models:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(config),
                        "feature_schema": feature_schema,
                        "split_id": split_id,
                        "adaptation_mode": mode,
                        "seed": args.seed + split_index * 100 + mode_index,
                    },
                    model_dir / f"adapted_{mode}_{split_id}.pt",
                )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics, budgets)
    split_summary = pd.DataFrame(split_rows)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    if ranking_rows:
        pd.DataFrame(ranking_rows).to_csv(output_dir / "rankings.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame(pretrain_history).to_csv(output_dir / "pretrain_history.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            output_dir / "adaptation_history.csv", index=False
        )
    summary = {
        "strict_external_double_cold": True,
        "feature_mode": args.feature_mode,
        "feature_schema": feature_schema,
        "model_config": asdict(config),
        "adaptation_modes": list(adaptation_modes),
        "n_current_proteins": len(current_protein_ids),
        "n_marts_proteins": len(marts_protein_ids),
        "n_current_reactions": len(current_reaction_ids),
        "n_marts_reactions": len(marts_reaction_ids),
        "n_current_pairs": len(current_pairs),
        "n_marts_pairs": len(marts_pairs),
        "pretrain_epochs": args.pretrain_epochs,
        "adapt_epochs": args.adapt_epochs,
        "pretrain_learning_rate": args.pretrain_learning_rate,
        "adapt_learning_rate": args.adapt_learning_rate,
        "temperature": args.temperature,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "topk_terms": list(topk_terms),
        "seed": args.seed,
        "ranking_depth": args.ranking_depth,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "split_summary": str(output_dir / "split_summary.csv"),
            "pretrain_history": str(output_dir / "pretrain_history.csv"),
            "adaptation_history": str(output_dir / "adaptation_history.csv"),
            "rankings": str(output_dir / "rankings.csv") if ranking_rows else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
