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

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_EMBEDDINGS,
    DEFAULT_EXACT_FOLDS,
    DEFAULT_POSITIVES,
    DEFAULT_PROTEIN_CLUSTERS,
    DEFAULT_REACTION_CLUSTERS,
    DEFAULT_STRICT_SPLITS,
    aggregate,
    masked_rank_metrics,
    parse_int_tuple,
    ranked_candidate_rows,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    build_training_mask,
    load_protein_features,
    seed_everything,
    tps_skeleton_attributes,
)

DEFAULT_OUTPUT = ROOT / "results/terpene_multi_expert_protocol_comparison"


@dataclass(frozen=True)
class MultiExpertConfig:
    protein_input_dim: int
    reaction_input_dim: int
    hidden_dim: int
    global_dim: int
    n_experts: int
    expert_dim: int
    dropout: float
    gate_temperature: float
    expert_mix_init: float
    mechanism_dims: tuple[int, ...] = ()
    mechanism_score_weight: float = 0.0


class ExpertTower(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        global_dim: int,
        n_experts: int,
        expert_dim: int,
        dropout: float,
        gate_temperature: float,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.gate_temperature = gate_temperature
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_projection = nn.Linear(hidden_dim, global_dim)
        self.expert_projection = nn.Linear(hidden_dim, n_experts * expert_dim)
        self.gate = nn.Linear(hidden_dim, n_experts)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(values)
        global_embedding = F.normalize(self.global_projection(hidden), p=2, dim=-1)
        expert_embeddings = self.expert_projection(hidden).reshape(
            len(values), self.n_experts, self.expert_dim
        )
        expert_embeddings = F.normalize(expert_embeddings, p=2, dim=-1)
        gates = F.softmax(self.gate(hidden) / self.gate_temperature, dim=-1)
        return global_embedding, expert_embeddings, gates


class DirectionalMultiExpertDualTower(nn.Module):
    def __init__(self, config: MultiExpertConfig) -> None:
        super().__init__()
        if config.n_experts <= 1:
            raise ValueError("n_experts must be greater than one")
        if config.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if not 0 < config.expert_mix_init < 1:
            raise ValueError("expert_mix_init must be inside (0, 1)")
        if config.mechanism_score_weight < 0:
            raise ValueError("mechanism_score_weight must be non-negative")
        self.config = config
        self.protein_tower = ExpertTower(
            config.protein_input_dim,
            config.hidden_dim,
            config.global_dim,
            config.n_experts,
            config.expert_dim,
            config.dropout,
            config.gate_temperature,
        )
        self.reaction_tower = ExpertTower(
            config.reaction_input_dim,
            config.hidden_dim,
            config.global_dim,
            config.n_experts,
            config.expert_dim,
            config.dropout,
            config.gate_temperature,
        )
        initial_logit = math.log(config.expert_mix_init / (1 - config.expert_mix_init))
        self.r2e_mix_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.e2r_mix_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.mechanism_heads = nn.ModuleList(
            [nn.Linear(config.global_dim, dimension) for dimension in config.mechanism_dims]
        )

    def encode_proteins(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.protein_tower(values)

    def encode_reactions(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reaction_tower(values)

    def score_matrices(
        self,
        protein_values: torch.Tensor,
        reaction_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        protein_global, protein_experts, protein_gates = self.encode_proteins(protein_values)
        reaction_global, reaction_experts, reaction_gates = self.encode_reactions(reaction_values)
        global_scores = reaction_global @ protein_global.T
        expert_scores = torch.einsum("rhd,phd->rph", reaction_experts, protein_experts)
        r2e_expert = (expert_scores * reaction_gates[:, None, :]).sum(dim=-1)
        e2r_expert = (expert_scores * protein_gates[None, :, :]).sum(dim=-1)
        r2e_mix = torch.sigmoid(self.r2e_mix_logit)
        e2r_mix = torch.sigmoid(self.e2r_mix_logit)
        mechanism_scores = torch.zeros_like(global_scores)
        protein_mechanism_logits: list[torch.Tensor] = []
        reaction_mechanism_logits: list[torch.Tensor] = []
        if self.mechanism_heads:
            local_scores: list[torch.Tensor] = []
            for head in self.mechanism_heads:
                protein_logits = head(protein_global)
                reaction_logits = head(reaction_global)
                protein_mechanism_logits.append(protein_logits)
                reaction_mechanism_logits.append(reaction_logits)
                protein_semantics = torch.tanh(protein_logits / 2.0)
                reaction_semantics = torch.tanh(reaction_logits / 2.0)
                protein_semantics = F.normalize(protein_semantics, p=2, dim=-1)
                reaction_semantics = F.normalize(reaction_semantics, p=2, dim=-1)
                local_scores.append(reaction_semantics @ protein_semantics.T)
            mechanism_scores = torch.stack(local_scores).mean(dim=0)
        r2e = (
            (1 - r2e_mix) * global_scores
            + r2e_mix * r2e_expert
            + self.config.mechanism_score_weight * mechanism_scores
        )
        e2r = (
            (1 - e2r_mix) * global_scores
            + e2r_mix * e2r_expert
            + self.config.mechanism_score_weight * mechanism_scores
        )
        diagnostics = {
            "protein_global": protein_global,
            "reaction_global": reaction_global,
            "protein_gates": protein_gates,
            "reaction_gates": reaction_gates,
            "protein_experts": protein_experts,
            "reaction_experts": reaction_experts,
            "protein_mechanism_logits": protein_mechanism_logits,
            "reaction_mechanism_logits": reaction_mechanism_logits,
            "mechanism_scores": mechanism_scores,
            "r2e_mix": r2e_mix,
            "e2r_mix": e2r_mix,
        }
        return r2e, e2r, diagnostics


def parse_topk_terms(value: str) -> tuple[tuple[int, float], ...]:
    if not value.strip():
        return ()
    terms: list[tuple[int, float]] = []
    for part in value.split(","):
        k_text, separator, weight_text = part.strip().partition(":")
        if not separator:
            raise ValueError("Top-K terms must use K:WEIGHT")
        k = int(k_text)
        weight = float(weight_text)
        if k <= 0 or weight < 0:
            raise ValueError("Top-K K must be positive and weight non-negative")
        if weight > 0:
            terms.append((k, weight))
    return tuple(terms)


def _retain_topk_negatives(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    hard_negative_k: int,
) -> torch.Tensor:
    if hard_negative_k <= 0:
        return denominator_mask | positive_mask
    negative_mask = denominator_mask & ~positive_mask
    k = min(int(hard_negative_k), int(logits.shape[1]))
    negative_logits = logits.masked_fill(~negative_mask, torch.finfo(logits.dtype).min)
    indices = torch.topk(negative_logits, k=k, dim=1).indices
    selected = torch.zeros_like(negative_mask, dtype=torch.bool)
    selected.scatter_(1, indices, True)
    return positive_mask | (selected & negative_mask)


def directional_multi_positive_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    hard_negative_k: int,
) -> torch.Tensor:
    negative_infinity = torch.finfo(logits.dtype).min
    denominator_mask = _retain_topk_negatives(
        logits, positive_mask, denominator_mask, hard_negative_k
    )
    valid = positive_mask.any(dim=1)
    numerator = logits.masked_fill(~positive_mask, negative_infinity)
    denominator = logits.masked_fill(~denominator_mask, negative_infinity)
    return (
        torch.logsumexp(denominator[valid], dim=1)
        - torch.logsumexp(numerator[valid], dim=1)
    ).mean()


def directional_topk_surrogate(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    terms: tuple[tuple[int, float], ...],
    margin: float,
) -> torch.Tensor:
    if not terms:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    negative_infinity = torch.finfo(logits.dtype).min
    valid_positive = positive_mask.any(dim=1)
    positive_best = logits.masked_fill(~positive_mask, negative_infinity).max(dim=1).values
    negative_mask = denominator_mask & ~positive_mask
    negative_logits = logits.masked_fill(~negative_mask, negative_infinity)
    losses: list[torch.Tensor] = []
    total_weight = 0.0
    for k, weight in terms:
        local_k = min(k, int(negative_logits.shape[1]))
        kth = torch.topk(negative_logits, k=local_k, dim=1).values[:, -1]
        valid = valid_positive & torch.isfinite(kth)
        if valid.any():
            losses.append(weight * F.softplus(kth[valid] - positive_best[valid] + margin).mean())
            total_weight += weight
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return sum(losses) / total_weight


def gate_regularization(
    gates: torch.Tensor,
    experts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_experts = gates.shape[1]
    mean_gate = gates.mean(dim=0)
    balance = n_experts * ((mean_gate - 1.0 / n_experts) ** 2).sum()
    entropy = -(gates.clamp_min(1e-8).log() * gates).sum(dim=1).mean()
    gram = torch.einsum("nhd,nkd->nhk", experts, experts)
    eye = torch.eye(n_experts, dtype=torch.bool, device=experts.device)[None, :, :]
    diversity = gram.masked_select(~eye).pow(2).mean()
    return balance, entropy, diversity


def build_denominator_masks(
    train_pairs: pd.DataFrame,
    reaction_rows: np.ndarray,
    protein_rows: np.ndarray,
    positive_mask: np.ndarray,
    reaction_to_row: dict[str, int],
    protein_to_row: dict[str, int],
    reaction_groups: dict[str, str],
    protein_groups: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    inverse_protein = {row: value for value, row in protein_to_row.items()}
    inverse_reaction = {row: value for value, row in reaction_to_row.items()}
    local_protein_groups = np.asarray(
        [protein_groups.get(inverse_protein[int(row)], "") for row in protein_rows],
        dtype=object,
    )
    local_reaction_groups = np.asarray(
        [reaction_groups.get(inverse_reaction[int(row)], "") for row in reaction_rows],
        dtype=object,
    )
    reaction_denominator = np.ones_like(positive_mask, dtype=bool)
    protein_denominator = np.ones_like(positive_mask, dtype=bool)
    for reaction_index in range(positive_mask.shape[0]):
        positive_groups = set(local_protein_groups[positive_mask[reaction_index]]) - {""}
        if positive_groups:
            hidden = np.isin(local_protein_groups, list(positive_groups)) & ~positive_mask[reaction_index]
            reaction_denominator[reaction_index, hidden] = False
    for protein_index in range(positive_mask.shape[1]):
        positive_groups = set(local_reaction_groups[positive_mask[:, protein_index]]) - {""}
        if positive_groups:
            hidden = np.isin(local_reaction_groups, list(positive_groups)) & ~positive_mask[:, protein_index]
            protein_denominator[hidden, protein_index] = False
    reaction_denominator |= positive_mask
    protein_denominator |= positive_mask
    return reaction_denominator, protein_denominator


def train_multi_expert(
    *,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    protein_groups: dict[str, str],
    reaction_groups: dict[str, str],
    config: MultiExpertConfig,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    hard_negative_k: int,
    hard_negative_start_epoch: int,
    topk_terms: tuple[tuple[int, float], ...],
    topk_margin: float,
    balance_weight: float,
    entropy_weight: float,
    diversity_weight: float,
    reaction_precursor_map: dict[str, str],
    reaction_skeleton_map: dict[str, str],
    mechanism_values: tuple[tuple[str, ...], ...],
    mechanism_auxiliary_weight: float,
    seed: int,
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[DirectionalMultiExpertDualTower, list[dict[str, float]]]:
    if mechanism_auxiliary_weight < 0:
        raise ValueError("mechanism_auxiliary_weight must be non-negative")
    if config.mechanism_dims and len(mechanism_values) != len(config.mechanism_dims):
        raise ValueError("Mechanism value groups and configured dimensions differ")
    seed_everything(seed)
    model = DirectionalMultiExpertDualTower(config).to(device)
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
    mechanism_targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    if model.mechanism_heads:
        inverse_protein = {row: value for value, row in protein_to_row.items()}
        inverse_reaction = {row: value for value, row in reaction_to_row.items()}
        train_protein_ids = [inverse_protein[int(row)] for row in protein_rows]
        train_reaction_ids = [inverse_reaction[int(row)] for row in reaction_rows]
        local_protein = {value: index for index, value in enumerate(train_protein_ids)}
        local_reaction = {value: index for index, value in enumerate(train_reaction_ids)}
        indices = [
            {value: index for index, value in enumerate(values)}
            for values in mechanism_values
        ]

        def attributes(reaction_id: str) -> tuple[str, str, str]:
            topology, oxidation = tps_skeleton_attributes(
                reaction_skeleton_map.get(reaction_id, "")
            )
            return reaction_precursor_map.get(reaction_id, ""), topology, oxidation

        reaction_target_arrays = [
            np.zeros((len(train_reaction_ids), len(values)), dtype=np.float32)
            for values in mechanism_values
        ]
        protein_target_arrays = [
            np.zeros((len(train_protein_ids), len(values)), dtype=np.float32)
            for values in mechanism_values
        ]
        for reaction_id, row_index in local_reaction.items():
            for group_index, value in enumerate(attributes(reaction_id)):
                value_index = indices[group_index].get(value)
                if value_index is not None:
                    reaction_target_arrays[group_index][row_index, value_index] = 1.0
        for row in train_pairs[["Entry", "rhea_id"]].drop_duplicates().itertuples(index=False):
            protein_index = local_protein.get(str(row.Entry))
            if protein_index is None:
                continue
            for group_index, value in enumerate(attributes(str(row.rhea_id))):
                value_index = indices[group_index].get(value)
                if value_index is not None:
                    protein_target_arrays[group_index][protein_index, value_index] = 1.0
        mechanism_targets = [
            (
                torch.as_tensor(reaction_target, dtype=torch.float32, device=device),
                torch.as_tensor(protein_target, dtype=torch.float32, device=device),
            )
            for reaction_target, protein_target in zip(
                reaction_target_arrays, protein_target_arrays
            )
        ]
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
        active_hard_k = hard_negative_k if epoch >= hard_negative_start_epoch else 0
        r2e_loss = directional_multi_positive_loss(
            r2e_logits, positive_t, reaction_denominator_t, active_hard_k
        )
        e2r_loss = directional_multi_positive_loss(
            e2r_logits, positive_t.T, protein_denominator_t.T, active_hard_k
        )
        contrastive = reaction_loss_weight * r2e_loss + (1 - reaction_loss_weight) * e2r_loss
        r2e_topk = directional_topk_surrogate(
            r2e_logits, positive_t, reaction_denominator_t, topk_terms, topk_margin
        )
        e2r_topk = directional_topk_surrogate(
            e2r_logits, positive_t.T, protein_denominator_t.T, topk_terms, topk_margin
        )
        topk_loss = reaction_loss_weight * r2e_topk + (1 - reaction_loss_weight) * e2r_topk
        protein_balance, protein_entropy, protein_diversity = gate_regularization(
            diagnostics["protein_gates"], diagnostics["protein_experts"]
        )
        reaction_balance, reaction_entropy, reaction_diversity = gate_regularization(
            diagnostics["reaction_gates"], diagnostics["reaction_experts"]
        )
        balance = 0.5 * (protein_balance + reaction_balance)
        entropy = 0.5 * (protein_entropy + reaction_entropy)
        diversity = 0.5 * (protein_diversity + reaction_diversity)
        mechanism_auxiliary_loss = torch.zeros(
            (), dtype=contrastive.dtype, device=device
        )
        if mechanism_targets:
            local_losses: list[torch.Tensor] = []
            for group_index, (reaction_target, protein_target) in enumerate(
                mechanism_targets
            ):
                local_losses.append(
                    F.binary_cross_entropy_with_logits(
                        diagnostics["reaction_mechanism_logits"][group_index],
                        reaction_target,
                    )
                )
                local_losses.append(
                    F.binary_cross_entropy_with_logits(
                        diagnostics["protein_mechanism_logits"][group_index],
                        protein_target,
                    )
                )
            mechanism_auxiliary_loss = torch.stack(local_losses).mean()
        loss = (
            contrastive
            + topk_loss
            + balance_weight * balance
            + entropy_weight * entropy
            + diversity_weight * diversity
            + mechanism_auxiliary_weight * mechanism_auxiliary_loss
        )
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
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": current,
                    "contrastive_loss": float(contrastive.detach().cpu()),
                    "r2e_loss": float(r2e_loss.detach().cpu()),
                    "e2r_loss": float(e2r_loss.detach().cpu()),
                    "topk_loss": float(topk_loss.detach().cpu()),
                    "balance_loss": float(balance.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "diversity_loss": float(diversity.detach().cpu()),
                    "mechanism_auxiliary_weight": float(mechanism_auxiliary_weight),
                    "mechanism_auxiliary_loss": float(
                        mechanism_auxiliary_loss.detach().cpu()
                    ),
                    "mechanism_score_weight": float(config.mechanism_score_weight),
                    "mean_mechanism_score": float(
                        diagnostics["mechanism_scores"].mean().detach().cpu()
                    ),
                    "r2e_expert_mix": float(diagnostics["r2e_mix"].detach().cpu()),
                    "e2r_expert_mix": float(diagnostics["e2r_mix"].detach().cpu()),
                    "active_hard_negative_k": float(active_hard_k),
                }
            )
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history


def prepare_data(args: argparse.Namespace) -> dict[str, object]:
    protein_matrix, protein_ids = load_protein_features(args.embedding_dir.resolve())
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(
        positives, args.reaction_feature_mode
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    reaction_precursor_map = dict(
        zip(
            reaction_table["rhea_id"].astype(str),
            reaction_table["precursor_class"].astype(str),
        )
    )
    reaction_skeleton_map = dict(
        zip(
            reaction_table["rhea_id"].astype(str),
            reaction_table["product_skeleton_class"].astype(str),
        )
    )
    mechanism_values = (
        tuple(sorted(set(reaction_precursor_map.values()) - {""})),
        ("acyclic", "mono", "bicyclic", "polycyclic", "unknown"),
        ("hydrocarbon", "oxygenated", "highly_oxygenated", "unknown"),
    )
    positives = positives[
        positives["Entry"].isin(protein_to_row)
        & positives["rhea_id"].isin(reaction_to_row)
    ].copy()
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
    pairs = positives[["Entry", "rhea_id"]].merge(
        strict, on=["Entry", "rhea_id"], how="left", validate="one_to_one"
    )
    if pairs[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict fold assignments do not cover every positive pair")
    pairs["protein_fold"] = pairs["protein_fold"].astype(int)
    pairs["reaction_fold"] = pairs["reaction_fold"].astype(int)
    protein_clusters = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    raw_protein_groups = dict(
        zip(
            protein_clusters["entry"].astype(str),
            protein_clusters["cluster_id"].astype(str),
        )
    )
    protein_groups = {
        value: raw_protein_groups.get(value, value) for value in protein_ids
    }
    reaction_clusters = pd.read_csv(args.reaction_clusters, dtype=str).fillna("")
    raw_reaction_groups = dict(
        zip(
            reaction_clusters["reaction_id"].astype(str),
            reaction_clusters["reaction_cluster"].astype(str),
        )
    )
    reaction_groups = {
        value: raw_reaction_groups.get(value, value) for value in reaction_ids
    }
    exact = pd.read_csv(args.exact_folds, dtype=str).fillna("")
    exact["legacy_exact_fold"] = pd.to_numeric(exact["legacy_exact_fold"]).astype(int)
    exact_fold_by_reaction = dict(
        zip(exact["reaction_id"].astype(str), exact["legacy_exact_fold"].astype(int))
    )
    return {
        "protein_matrix": protein_matrix,
        "protein_ids": protein_ids,
        "reaction_matrix": reaction_matrix,
        "reaction_ids": reaction_ids,
        "reaction_table": reaction_table,
        "feature_schema": feature_schema,
        "protein_to_row": protein_to_row,
        "reaction_to_row": reaction_to_row,
        "reaction_precursor_map": reaction_precursor_map,
        "reaction_skeleton_map": reaction_skeleton_map,
        "mechanism_values": mechanism_values,
        "pairs": pairs,
        "protein_groups": protein_groups,
        "reaction_groups": reaction_groups,
        "exact_fold_by_reaction": exact_fold_by_reaction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a directional multi-expert TPS dual tower under exact and strict protocols."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocols", default="legacy_exact,double_cold_25cell")
    parser.add_argument("--seeds", default="20260723")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--reaction-feature-mode", choices=["drfp_categorical", "multiview"], default="drfp_categorical")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.75)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--global-dim", type=int, default=128)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--expert-dim", type=int, default=64)
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
    parser.add_argument("--mechanism-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--mechanism-score-weight", type=float, default=0.0)
    parser.add_argument("--ranking-depth", type=int, default=0)
    parser.add_argument(
        "--strict-partition",
        choices=["all", "development", "frozen"],
        default="all",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocols = tuple(part.strip() for part in args.protocols.split(",") if part.strip())
    unknown = set(protocols) - {"legacy_exact", "double_cold_25cell"}
    if unknown:
        raise ValueError(f"Unknown protocols: {sorted(unknown)}")
    seeds = parse_int_tuple(args.seeds)
    budgets = parse_int_tuple(args.budgets)
    topk_terms = parse_topk_terms(args.topk_terms)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args)
    protein_matrix = data["protein_matrix"]
    reaction_matrix = data["reaction_matrix"]
    protein_ids = data["protein_ids"]
    reaction_ids = data["reaction_ids"]
    protein_to_row = data["protein_to_row"]
    reaction_to_row = data["reaction_to_row"]
    reaction_precursor_map = data["reaction_precursor_map"]
    reaction_skeleton_map = data["reaction_skeleton_map"]
    mechanism_values = data["mechanism_values"]
    pairs = data["pairs"]
    protein_groups = data["protein_groups"]
    reaction_groups = data["reaction_groups"]
    exact_fold_by_reaction = data["exact_fold_by_reaction"]
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    config = MultiExpertConfig(
        protein_input_dim=int(protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        expert_mix_init=args.expert_mix_init,
        mechanism_dims=(
            tuple(len(values) for values in mechanism_values)
            if args.mechanism_auxiliary_weight > 0 or args.mechanism_score_weight > 0
            else ()
        ),
        mechanism_score_weight=args.mechanism_score_weight,
    )
    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in pairs.groupby("rhea_id", sort=True)
    }
    records: list[dict[str, object]] = []
    ranking_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []

    def fit(train_pairs: pd.DataFrame, split_id: str) -> list[DirectionalMultiExpertDualTower]:
        models: list[DirectionalMultiExpertDualTower] = []
        for seed in seeds:
            model, history = train_multi_expert(
                protein_tensor=protein_tensor,
                reaction_tensor=reaction_tensor,
                train_pairs=train_pairs,
                protein_to_row=protein_to_row,
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
                reaction_precursor_map=reaction_precursor_map,
                reaction_skeleton_map=reaction_skeleton_map,
                mechanism_values=mechanism_values,
                mechanism_auxiliary_weight=args.mechanism_auxiliary_weight,
                seed=seed,
                device=device,
            )
            models.append(model)
            training_records.append(
                {
                    "split_id": split_id,
                    "seed": seed,
                    "n_train_pairs": len(train_pairs),
                    **history[-1],
                    "best_loss": min(item["loss"] for item in history),
                }
            )
        return models

    def ensemble_r2e(models: list[DirectionalMultiExpertDualTower]) -> np.ndarray:
        matrices=[]
        for model in models:
            model.eval()
            with torch.no_grad():
                r2e, _, _ = model.score_matrices(protein_tensor, reaction_tensor)
            matrices.append(r2e.cpu().numpy())
        return np.mean(matrices, axis=0)

    if "legacy_exact" in protocols:
        for fold in range(5):
            test_reactions = {
                value for value, local_fold in exact_fold_by_reaction.items() if local_fold == fold
            }
            train_pairs = pairs[~pairs["rhea_id"].isin(test_reactions)][
                ["Entry", "rhea_id"]
            ].drop_duplicates()
            score_matrix = ensemble_r2e(fit(train_pairs, f"exact_r{fold}"))
            for reaction_id in sorted(test_reactions):
                positives = all_positive_by_reaction.get(reaction_id, set())
                if positives:
                    query_scores = score_matrix[reaction_to_row[reaction_id]]
                    records.append(
                        {
                            "protocol": "legacy_exact",
                            "protein_fold": "",
                            "reaction_fold": fold,
                            "reaction_id": reaction_id,
                            **masked_rank_metrics(
                                query_scores,
                                protein_ids,
                                positives,
                                set(),
                                budgets,
                            ),
                        }
                    )
                    for rank, candidate_id, score in ranked_candidate_rows(
                        query_scores, protein_ids, set(), args.ranking_depth
                    ):
                        ranking_records.append(
                            {
                                "protocol": "legacy_exact",
                                "protein_fold": "",
                                "reaction_fold": fold,
                                "reaction_id": reaction_id,
                                "candidate_id": candidate_id,
                                "rank": rank,
                                "score": score,
                            }
                        )

    if "double_cold_25cell" in protocols:
        for protein_fold in range(5):
            for reaction_fold in range(5):
                is_development = protein_fold == 4 or reaction_fold == 4
                if args.strict_partition == "development" and not is_development:
                    continue
                if args.strict_partition == "frozen" and is_development:
                    continue
                train_pairs = pairs[
                    (pairs["protein_fold"] != protein_fold)
                    & (pairs["reaction_fold"] != reaction_fold)
                ][["Entry", "rhea_id"]].drop_duplicates()
                test_pairs = pairs[
                    (pairs["protein_fold"] == protein_fold)
                    & (pairs["reaction_fold"] == reaction_fold)
                ].copy()
                if test_pairs.empty:
                    continue
                split_id = f"p{protein_fold}_r{reaction_fold}"
                score_matrix = ensemble_r2e(fit(train_pairs, split_id))
                for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                    positives = set(group["Entry"].astype(str))
                    known_other = all_positive_by_reaction.get(reaction_id, set()) - positives
                    query_scores = score_matrix[reaction_to_row[reaction_id]]
                    records.append(
                        {
                            "protocol": "double_cold_25cell",
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction_id,
                            **masked_rank_metrics(
                                query_scores,
                                protein_ids,
                                positives,
                                known_other,
                                budgets,
                            ),
                        }
                    )
                    for rank, candidate_id, score in ranked_candidate_rows(
                        query_scores, protein_ids, known_other, args.ranking_depth
                    ):
                        ranking_records.append(
                            {
                                "protocol": "double_cold_25cell",
                                "protein_fold": protein_fold,
                                "reaction_fold": reaction_fold,
                                "reaction_id": reaction_id,
                                "candidate_id": candidate_id,
                                "rank": rank,
                                "score": score,
                            }
                        )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics, budgets)
    training = pd.DataFrame(training_records)
    rankings = pd.DataFrame(ranking_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    training.to_csv(output_dir / "training_summary.csv", index=False)
    if args.ranking_depth > 0:
        rankings.to_csv(output_dir / "rankings.csv", index=False)
    summary = {
        "method": "directional_multi_expert_dual_tower",
        "config": asdict(config),
        "protocols": list(protocols),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "hard_negative_start_epoch": args.hard_negative_start_epoch,
        "topk_terms": list(topk_terms),
        "topk_margin": args.topk_margin,
        "balance_weight": args.balance_weight,
        "entropy_weight": args.entropy_weight,
        "diversity_weight": args.diversity_weight,
        "mechanism_auxiliary_weight": args.mechanism_auxiliary_weight,
        "mechanism_score_weight": args.mechanism_score_weight,
        "mechanism_values": [list(values) for values in mechanism_values],
        "ranking_depth": args.ranking_depth,
        "strict_partition": args.strict_partition,
        "reaction_feature_mode": args.reaction_feature_mode,
        "pu_group_mask": True,
        "feature_schema": data["feature_schema"],
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "training_summary": str(output_dir / "training_summary.csv"),
            "rankings": str(output_dir / "rankings.csv") if args.ranking_depth > 0 else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
