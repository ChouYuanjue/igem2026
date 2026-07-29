from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_multi_expert_marts_adaptation import (  # noqa: E402
    CURRENT_POSITIVES,
    CURRENT_PROTEIN_CLUSTERS,
    CURRENT_PROTEINS,
    CURRENT_REACTION_CLUSTERS,
    CURRENT_SEQUENCES,
    MARTS_CACHE,
    build_combined_protein_groups,
    build_combined_reaction_groups,
    build_unified_reaction_features,
)
from projects.active.terpene_screening.evaluate_multi_expert_protocol_comparison import (  # noqa: E402
    build_denominator_masks,
    directional_multi_positive_loss,
    directional_topk_surrogate,
    parse_topk_terms,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_training_mask,
    load_protein_features,
    rank_metrics,
    seed_everything,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_interaction_retriever_marts"
DEFAULT_BUDGETS = (3, 5, 10, 20)


@dataclass(frozen=True)
class InteractionConfig:
    protein_input_dim: int
    reaction_input_dim: int
    hidden_dim: int
    embedding_dim: int
    interaction_hidden_dim: int
    dropout: float
    residual_scale_init: float
    interaction_chunk_size: int


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


class PairResidualHead(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, reaction: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        product = reaction * protein
        difference = torch.abs(reaction - protein)
        return self.network(torch.cat([product, difference], dim=-1)).squeeze(-1)


class DirectionalInteractionRetriever(nn.Module):
    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        self.config = config
        self.protein_encoder = Encoder(
            config.protein_input_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.dropout,
        )
        self.reaction_encoder = Encoder(
            config.reaction_input_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.dropout,
        )
        self.r2e_head = PairResidualHead(
            config.embedding_dim, config.interaction_hidden_dim, config.dropout
        )
        self.e2r_head = PairResidualHead(
            config.embedding_dim, config.interaction_hidden_dim, config.dropout
        )
        scale = min(max(config.residual_scale_init, 1e-4), 1 - 1e-4)
        initial_logit = math.log(scale / (1 - scale))
        self.r2e_scale_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.e2r_scale_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_encoder(values)

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        return self.reaction_encoder(values)

    def score_encoded(
        self,
        proteins: torch.Tensor,
        reactions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        base = reactions @ proteins.T
        r2e_blocks: list[torch.Tensor] = []
        e2r_blocks: list[torch.Tensor] = []
        chunk = max(1, int(self.config.interaction_chunk_size))
        for start in range(0, reactions.shape[0], chunk):
            local = reactions[start : start + chunk]
            local_expanded = local[:, None, :].expand(-1, proteins.shape[0], -1)
            protein_expanded = proteins[None, :, :].expand(local.shape[0], -1, -1)
            r2e_blocks.append(self.r2e_head(local_expanded, protein_expanded))
            e2r_blocks.append(self.e2r_head(local_expanded, protein_expanded))
        r2e_residual = torch.cat(r2e_blocks, dim=0)
        e2r_residual = torch.cat(e2r_blocks, dim=0)
        r2e_scale = torch.sigmoid(self.r2e_scale_logit)
        e2r_scale = torch.sigmoid(self.e2r_scale_logit)
        r2e = base + r2e_scale * r2e_residual
        e2r = base + e2r_scale * e2r_residual
        diagnostics = {
            "r2e_scale": r2e_scale,
            "e2r_scale": e2r_scale,
            "r2e_residual_rms": r2e_residual.square().mean().sqrt(),
            "e2r_residual_rms": e2r_residual.square().mean().sqrt(),
        }
        return r2e, e2r, diagnostics

    def score_matrices(
        self,
        protein_features: torch.Tensor,
        reaction_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        proteins = self.encode_proteins(protein_features)
        reactions = self.encode_reactions(reaction_features)
        return self.score_encoded(proteins, reactions)


def train_interaction_model(
    *,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    protein_groups: dict[str, str],
    reaction_groups: dict[str, str],
    config: InteractionConfig,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    hard_negative_k: int,
    hard_negative_start_epoch: int,
    topk_terms: tuple[tuple[int, float], ...],
    topk_margin: float,
    residual_penalty: float,
    seed: int,
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[DirectionalInteractionRetriever, list[dict[str, float]]]:
    seed_everything(seed)
    model = DirectionalInteractionRetriever(config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    reaction_rows, protein_rows, positive_mask = build_training_mask(
        train_pairs, reaction_to_row, protein_to_row
    )
    reaction_denominator, protein_denominator = build_denominator_masks(
        train_pairs,
        reaction_rows,
        protein_rows,
        positive_mask,
        reaction_to_row,
        protein_to_row,
        reaction_groups,
        protein_groups,
    )
    reaction_rows_t = torch.as_tensor(reaction_rows, dtype=torch.long, device=device)
    protein_rows_t = torch.as_tensor(protein_rows, dtype=torch.long, device=device)
    positive_t = torch.as_tensor(positive_mask, dtype=torch.bool, device=device)
    reaction_denominator_t = torch.as_tensor(
        reaction_denominator, dtype=torch.bool, device=device
    )
    protein_denominator_t = torch.as_tensor(
        protein_denominator, dtype=torch.bool, device=device
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        r2e_scores, e2r_scores, diagnostics = model.score_matrices(
            protein_tensor[protein_rows_t], reaction_tensor[reaction_rows_t]
        )
        r2e_logits = r2e_scores / temperature
        e2r_logits = e2r_scores.T / temperature
        active_hard = hard_negative_k if epoch >= hard_negative_start_epoch else 0
        r2e_loss = directional_multi_positive_loss(
            r2e_logits, positive_t, reaction_denominator_t, active_hard
        )
        e2r_loss = directional_multi_positive_loss(
            e2r_logits, positive_t.T, protein_denominator_t.T, active_hard
        )
        contrastive = reaction_loss_weight * r2e_loss + (1 - reaction_loss_weight) * e2r_loss
        r2e_topk = directional_topk_surrogate(
            r2e_logits, positive_t, reaction_denominator_t, topk_terms, topk_margin
        )
        e2r_topk = directional_topk_surrogate(
            e2r_logits, positive_t.T, protein_denominator_t.T, topk_terms, topk_margin
        )
        topk = reaction_loss_weight * r2e_topk + (1 - reaction_loss_weight) * e2r_topk
        residual = 0.5 * (
            diagnostics["r2e_residual_rms"].square()
            + diagnostics["e2r_residual_rms"].square()
        )
        loss = contrastive + topk + residual_penalty * residual
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": current,
                    "contrastive_loss": float(contrastive.detach().cpu()),
                    "r2e_loss": float(r2e_loss.detach().cpu()),
                    "e2r_loss": float(e2r_loss.detach().cpu()),
                    "topk_loss": float(topk.detach().cpu()),
                    "residual_penalty_loss": float(residual.detach().cpu()),
                    "r2e_scale": float(diagnostics["r2e_scale"].detach().cpu()),
                    "e2r_scale": float(diagnostics["e2r_scale"].detach().cpu()),
                    "r2e_residual_rms": float(
                        diagnostics["r2e_residual_rms"].detach().cpu()
                    ),
                    "e2r_residual_rms": float(
                        diagnostics["e2r_residual_rms"].detach().cpu()
                    ),
                    "active_hard_negative_k": float(active_hard),
                }
            )
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


def score_marts(
    model: DirectionalInteractionRetriever,
    proteins: torch.Tensor,
    reactions: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        r2e, e2r, _ = model.score_matrices(proteins, reactions)
    return r2e.cpu().numpy(), e2r.cpu().numpy()


def evaluate_test_pairs(
    records: list[dict[str, object]],
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
        records.append(
            {
                "split_id": split_id,
                "direction": "reaction_to_enzyme",
                "query_id": reaction_id,
                **rank_metrics(
                    r2e_scores[reaction_to_row[reaction_id]],
                    protein_ids,
                    set(group["Entry"].astype(str)),
                    set(),
                    budgets,
                ),
            }
        )
    for protein_id, group in test_pairs.groupby("Entry", sort=True):
        protein_id = str(protein_id)
        records.append(
            {
                "split_id": split_id,
                "direction": "enzyme_to_reaction",
                "query_id": protein_id,
                **rank_metrics(
                    e2r_scores[:, protein_to_row[protein_id]],
                    reaction_ids,
                    set(group["rhea_id"].astype(str)),
                    set(),
                    budgets,
                ),
            }
        )


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}", "mean"
        )
    return frame.groupby("direction").agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direction-specific nonlinear pair-interaction retriever on strict MARTS double-cold cells."
    )
    parser.add_argument("--current-positives", type=Path, default=CURRENT_POSITIVES)
    parser.add_argument("--current-protein-dir", type=Path, default=CURRENT_PROTEINS)
    parser.add_argument("--current-sequences", type=Path, default=CURRENT_SEQUENCES)
    parser.add_argument("--current-protein-clusters", type=Path, default=CURRENT_PROTEIN_CLUSTERS)
    parser.add_argument("--current-reaction-clusters", type=Path, default=CURRENT_REACTION_CLUSTERS)
    parser.add_argument("--marts-cache", type=Path, default=MARTS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partition", choices=["development", "frozen", "all"], default="development")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--adapt-epochs", type=int, default=20)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-k", type=int, default=0)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=10)
    parser.add_argument("--topk-terms", default="3:0.10,10:0.05,20:0.025")
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--residual-penalty", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--interaction-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale-init", type=float, default=0.1)
    parser.add_argument("--interaction-chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    topk_terms = parse_topk_terms(args.topk_terms)
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

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
    marts_reactions = pd.read_csv(cache / "reaction_entities.csv", dtype=str).fillna("")
    marts_reaction_ids = marts_reactions["reaction_id"].astype(str).tolist()
    marts_pairs = pd.read_csv(cache / "marts_pair_folds.csv", dtype=str).fillna("")
    for column in ("protein_fold", "reaction_fold"):
        marts_pairs[column] = pd.to_numeric(marts_pairs[column]).astype(int)
    for column in ("protein_seen", "reaction_seen"):
        marts_pairs[column] = marts_pairs[column].astype(str).str.lower().eq("true")

    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_unified_reaction_features(
        current_positives, marts_reactions, "multiview"
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    combined_protein_ids = current_protein_ids + marts_protein_ids
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
    config = InteractionConfig(
        protein_input_dim=int(combined_protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        interaction_hidden_dim=args.interaction_hidden_dim,
        dropout=args.dropout,
        residual_scale_init=args.residual_scale_init,
        interaction_chunk_size=args.interaction_chunk_size,
    )
    protein_tensor = torch.as_tensor(
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
    current_pairs = current_positives[["Entry", "rhea_id"]].drop_duplicates()
    pretrained, pretrain_history = train_interaction_model(
        protein_tensor=protein_tensor,
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
        residual_penalty=args.residual_penalty,
        seed=args.seed,
        device=device,
    )
    pretrained_state = {
        name: value.detach().cpu().clone() for name, value in pretrained.state_dict().items()
    }

    records: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    cells = [(p, r) for p in range(5) for r in range(5)]
    if args.partition == "development":
        cells = [(p, r) for p, r in cells if p == 4 or r == 4]
    elif args.partition == "frozen":
        cells = [(p, r) for p, r in cells if p != 4 and r != 4]

    for cell_index, (protein_fold, reaction_fold) in enumerate(cells):
        split_id = f"p{protein_fold}_r{reaction_fold}"
        train_pairs = marts_pairs[
            (marts_pairs["protein_fold"] != protein_fold)
            & (marts_pairs["reaction_fold"] != reaction_fold)
        ][["Entry", "rhea_id"]].drop_duplicates()
        test_pairs = marts_pairs[
            (marts_pairs["protein_fold"] == protein_fold)
            & (marts_pairs["reaction_fold"] == reaction_fold)
            & (~marts_pairs["protein_seen"])
            & (~marts_pairs["reaction_seen"])
        ][["Entry", "rhea_id"]].drop_duplicates()
        if test_pairs.empty:
            continue
        model, history = train_interaction_model(
            protein_tensor=protein_tensor,
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
            residual_penalty=args.residual_penalty,
            seed=args.seed + cell_index,
            device=device,
            initial_state_dict=pretrained_state,
        )
        r2e_scores, e2r_scores = score_marts(
            model,
            protein_tensor[marts_protein_rows],
            reaction_tensor[marts_reaction_rows],
        )
        evaluate_test_pairs(
            records,
            split_id,
            r2e_scores,
            e2r_scores,
            test_pairs,
            marts_protein_ids,
            marts_reaction_ids,
            budgets,
        )
        local_history = pd.DataFrame(history)
        local_history.insert(0, "split_id", split_id)
        histories.append(local_history)
        split_rows.append(
            {
                "split_id": split_id,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "test_proteins": test_pairs["Entry"].nunique(),
                "test_reactions": test_pairs["rhea_id"].nunique(),
            }
        )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics, budgets)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(pretrain_history).to_csv(output / "pretrain_history.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(output / "adaptation_history.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_summary.csv", index=False)
    summary = {
        "partition": args.partition,
        "strict_external_double_cold": True,
        "model_type": "directional_pair_interaction_retriever",
        "model_config": asdict(config),
        "feature_schema": feature_schema,
        "pretrain_epochs": args.pretrain_epochs,
        "adapt_epochs": args.adapt_epochs,
        "reaction_loss_weight": args.reaction_loss_weight,
        "topk_terms": list(topk_terms),
        "hard_negative_k": args.hard_negative_k,
        "residual_penalty": args.residual_penalty,
        "n_cells": len(split_rows),
        "outputs": {
            "metrics": str(output / "metrics.csv"),
            "query_metrics": str(output / "query_metrics.csv"),
            "pretrain_history": str(output / "pretrain_history.csv"),
            "adaptation_history": str(output / "adaptation_history.csv"),
            "split_summary": str(output / "split_summary.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
