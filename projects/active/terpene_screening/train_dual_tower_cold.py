from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder
from rdkit import DataStructs
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    canonical_or_raw_reaction,
    carbon_count,
    largest_organic_component,
    mol_fp,
    oxygen_count,
    phosphorus_count,
    precursor_class_from_reaction,
    product_skeleton_class,
    ring_count,
    split_reaction_smiles,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_dual_tower_cold"
DEFAULT_SCOPES = ("protein_cold", "reaction_cold", "double_cold")
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class ModelConfig:
    protein_input_dim: int
    reaction_input_dim: int
    hidden_dim: int
    embedding_dim: int
    dropout: float


class ProjectionTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), p=2, dim=-1)


class TerpeneDualTower(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.protein_tower = ProjectionTower(
            config.protein_input_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.dropout,
        )
        self.reaction_tower = ProjectionTower(
            config.reaction_input_dim,
            config.hidden_dim,
            config.embedding_dim,
            config.dropout,
        )

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_tower(values)

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        return self.reaction_tower(values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Expected a comma-separated list of positive integers.")
    return result


def parse_str_tuple(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected a non-empty comma-separated list.")
    unknown = set(result) - set(DEFAULT_SCOPES)
    if unknown:
        raise ValueError(f"Unknown scopes: {sorted(unknown)}")
    return result


def load_protein_features(embedding_dir: Path) -> tuple[np.ndarray, list[str]]:
    entries = pd.read_csv(embedding_dir / "entries.csv", dtype={"Entry": str}).sort_values("row")
    matrix = np.load(embedding_dir / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError("Protein embedding matrix and entry map have different lengths.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms, entries["Entry"].astype(str).tolist()


def load_aligned_feature_augmentation(
    directory: Path,
    identifiers: list[str],
) -> np.ndarray:
    entries = pd.read_csv(directory / "entries.csv", dtype=str).fillna("")
    if "row" not in entries.columns:
        raise ValueError(f"Feature augmentation entries lack row column: {directory}")
    id_columns = [
        column
        for column in ("reaction_id", "rhea_id", "id", "Entry")
        if column in entries.columns
    ]
    if len(id_columns) != 1:
        raise ValueError(
            f"Feature augmentation must expose exactly one identifier column; "
            f"found {id_columns} under {directory}"
        )
    id_column = id_columns[0]
    entries["row"] = pd.to_numeric(entries["row"]).astype(int)
    matrix = np.load(directory / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError(f"Feature augmentation matrix and entries differ under {directory}")
    row_by_id = dict(zip(entries[id_column].astype(str), entries["row"].astype(int)))
    missing = [value for value in identifiers if value not in row_by_id]
    if missing:
        raise ValueError(
            f"Feature augmentation misses {len(missing)} identifiers under {directory}; "
            f"examples={missing[:10]}"
        )
    aligned = np.stack([matrix[row_by_id[value]] for value in identifiers]).astype(np.float32)
    if not np.isfinite(aligned).all():
        raise ValueError(f"Feature augmentation contains non-finite values: {directory}")
    return aligned


def fingerprint_array(fingerprint: object, dimension: int = 2048) -> np.ndarray:
    values = np.zeros(dimension, dtype=np.float32)
    if fingerprint is not None:
        DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


def reaction_multiview_features(reaction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reactants, products = split_reaction_smiles(reaction)
    substrate = largest_organic_component(reactants)
    product = largest_organic_component(products)
    substrate_fp = fingerprint_array(mol_fp(substrate))
    product_fp = fingerprint_array(mol_fp(product))
    signed_difference = product_fp - substrate_fp
    descriptors = np.asarray(
        [
            carbon_count(substrate) / 50.0,
            carbon_count(product) / 50.0,
            (carbon_count(product) - carbon_count(substrate)) / 50.0,
            oxygen_count(substrate) / 20.0,
            oxygen_count(product) / 20.0,
            (oxygen_count(product) - oxygen_count(substrate)) / 20.0,
            phosphorus_count(substrate) / 5.0,
            phosphorus_count(product) / 5.0,
            ring_count(substrate) / 10.0,
            ring_count(product) / 10.0,
            (ring_count(product) - ring_count(substrate)) / 10.0,
        ],
        dtype=np.float32,
    )
    return substrate_fp, product_fp, signed_difference, descriptors


def build_reaction_features(
    positives: pd.DataFrame,
    feature_mode: str,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, object]]:
    reactions = (
        positives.groupby("rhea_id", as_index=False)["smiles_seq"]
        .first()
        .sort_values("rhea_id")
        .reset_index(drop=True)
    )
    reactions["canonical_reaction"] = reactions["smiles_seq"].map(canonical_or_raw_reaction)
    reactions["has_reaction_smiles"] = reactions["canonical_reaction"].astype(str).str.contains(">>", regex=False)
    reactions["precursor_class"] = reactions["canonical_reaction"].map(precursor_class_from_reaction)
    reactions["product_skeleton_class"] = reactions["canonical_reaction"].map(product_skeleton_class)

    drfp_rows: list[np.ndarray] = []
    drfp_dimension = 2048
    for reaction in reactions["canonical_reaction"].astype(str):
        if ">>" not in reaction:
            drfp_rows.append(np.zeros(drfp_dimension, dtype=np.float32))
            continue
        try:
            encoded = DrfpEncoder.encode([reaction])[0].astype(np.float32, copy=False)
            drfp_dimension = int(encoded.shape[0])
            drfp_rows.append(encoded)
        except Exception:
            drfp_rows.append(np.zeros(drfp_dimension, dtype=np.float32))
    drfp = np.stack(drfp_rows)
    precursor_values = sorted(reactions["precursor_class"].astype(str).unique())
    skeleton_values = sorted(reactions["product_skeleton_class"].astype(str).unique())
    precursor_index = {value: index for index, value in enumerate(precursor_values)}
    skeleton_index = {value: index for index, value in enumerate(skeleton_values)}
    categorical = np.zeros((len(reactions), len(precursor_values) + len(skeleton_values)), dtype=np.float32)
    for row_index, row in reactions.iterrows():
        categorical[row_index, precursor_index[str(row["precursor_class"])]] = 1.0
        categorical[row_index, len(precursor_values) + skeleton_index[str(row["product_skeleton_class"])]] = 1.0
    feature_blocks = [drfp]
    multiview_dimensions: dict[str, int] = {}
    if feature_mode == "multiview":
        substrate_rows: list[np.ndarray] = []
        product_rows: list[np.ndarray] = []
        difference_rows: list[np.ndarray] = []
        descriptor_rows: list[np.ndarray] = []
        for reaction in reactions["canonical_reaction"].astype(str):
            substrate_fp, product_fp, signed_difference, descriptors = reaction_multiview_features(reaction)
            substrate_rows.append(substrate_fp)
            product_rows.append(product_fp)
            difference_rows.append(signed_difference)
            descriptor_rows.append(descriptors)
        substrate_matrix = np.stack(substrate_rows)
        product_matrix = np.stack(product_rows)
        difference_matrix = np.stack(difference_rows)
        descriptor_matrix = np.stack(descriptor_rows)
        feature_blocks.extend([substrate_matrix, product_matrix, difference_matrix, descriptor_matrix])
        multiview_dimensions = {
            "substrate_morgan_dimension": int(substrate_matrix.shape[1]),
            "product_morgan_dimension": int(product_matrix.shape[1]),
            "signed_difference_dimension": int(difference_matrix.shape[1]),
            "descriptor_dimension": int(descriptor_matrix.shape[1]),
        }
    elif feature_mode != "drfp_categorical":
        raise ValueError(f"Unknown reaction feature mode: {feature_mode}")
    feature_blocks.append(categorical)
    matrix = np.concatenate(feature_blocks, axis=1)
    schema = {
        "feature_mode": feature_mode,
        "precursor_classes": precursor_values,
        "product_skeleton_classes": skeleton_values,
        "drfp_dimension": int(drfp.shape[1]),
        "n_reactions_without_parseable_smiles": int((~reactions["has_reaction_smiles"]).sum()),
        **multiview_dimensions,
    }
    return matrix, reactions["rhea_id"].astype(str).tolist(), reactions, schema


def _retain_topk_negatives(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
    hard_negative_k: int,
) -> torch.Tensor:
    """Keep every positive and only the highest-scoring allowed negatives per query."""
    if hard_negative_k <= 0:
        return denominator_mask | positive_mask
    negative_mask = denominator_mask & ~positive_mask
    if logits.shape[1] == 0:
        return positive_mask
    k = min(int(hard_negative_k), int(logits.shape[1]))
    negative_logits = logits.masked_fill(~negative_mask, torch.finfo(logits.dtype).min)
    indices = torch.topk(negative_logits, k=k, dim=1).indices
    selected = torch.zeros_like(negative_mask, dtype=torch.bool)
    selected.scatter_(1, indices, True)
    selected &= negative_mask
    return positive_mask | selected


def multi_positive_contrastive_loss(
    reaction_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
    reaction_denominator_mask: torch.Tensor | None = None,
    protein_denominator_mask: torch.Tensor | None = None,
    reaction_loss_weight: float = 0.5,
    loss_mode: str = "bidirectional_infonce",
    hard_negative_k: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= reaction_loss_weight <= 1:
        raise ValueError("reaction_loss_weight must be within [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if hard_negative_k < 0:
        raise ValueError("hard_negative_k must be non-negative")
    if loss_mode not in {"bidirectional_infonce", "global_mlnce"}:
        raise ValueError(f"Unknown loss mode: {loss_mode}")

    logits = reaction_embeddings @ protein_embeddings.T / temperature
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    if reaction_denominator_mask is None:
        reaction_denominator_mask = torch.ones_like(positive_mask, dtype=torch.bool)
    if protein_denominator_mask is None:
        protein_denominator_mask = torch.ones_like(positive_mask, dtype=torch.bool)
    reaction_denominator_mask = _retain_topk_negatives(
        logits,
        positive_mask,
        reaction_denominator_mask,
        hard_negative_k,
    )
    protein_denominator_mask_t = _retain_topk_negatives(
        logits.T,
        positive_mask.T,
        protein_denominator_mask.T,
        hard_negative_k,
    )
    protein_denominator_mask = protein_denominator_mask_t.T

    if loss_mode == "global_mlnce":
        global_denominator = reaction_denominator_mask & protein_denominator_mask
        global_denominator |= positive_mask
        global_logits = logits.masked_fill(~global_denominator, negative_infinity)
        positive_values = logits[positive_mask]
        if positive_values.numel() == 0:
            raise ValueError("MLNCE requires at least one positive pair")
        global_loss = torch.logsumexp(global_logits.reshape(-1), dim=0) - positive_values.mean()
        return global_loss, global_loss, global_loss

    reaction_logits = logits.masked_fill(~reaction_denominator_mask, negative_infinity)
    valid_reactions = positive_mask.any(dim=1)
    valid_proteins = positive_mask.any(dim=0)
    reaction_loss = (
        torch.logsumexp(reaction_logits[valid_reactions], dim=1)
        - torch.logsumexp(positive_logits[valid_reactions], dim=1)
    ).mean()
    protein_logits = logits.T.masked_fill(~protein_denominator_mask.T, negative_infinity)
    protein_positive_logits = positive_logits.T
    protein_loss = (
        torch.logsumexp(protein_logits[valid_proteins], dim=1)
        - torch.logsumexp(protein_positive_logits[valid_proteins], dim=1)
    ).mean()
    total_loss = reaction_loss_weight * reaction_loss + (1 - reaction_loss_weight) * protein_loss
    return total_loss, reaction_loss, protein_loss


def topk_hit_surrogate_loss(
    reaction_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    reaction_denominator_mask: torch.Tensor,
    protein_denominator_mask: torch.Tensor,
    temperature: float,
    target_k: int,
    margin: float,
    reaction_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smoothly require at least one positive to outrank the K-th allowed negative."""
    if target_k <= 0:
        raise ValueError("topk surrogate target_k must be positive")
    if margin < 0:
        raise ValueError("topk surrogate margin must be non-negative")
    logits = reaction_embeddings @ protein_embeddings.T / temperature
    negative_infinity = torch.finfo(logits.dtype).min

    def directional(
        directional_logits: torch.Tensor,
        directional_positive: torch.Tensor,
        directional_denominator: torch.Tensor,
    ) -> torch.Tensor:
        valid = directional_positive.any(dim=1)
        positive_best = directional_logits.masked_fill(
            ~directional_positive, negative_infinity
        ).max(dim=1).values
        negative_mask = directional_denominator & ~directional_positive
        negative_logits = directional_logits.masked_fill(
            ~negative_mask, negative_infinity
        )
        k = min(int(target_k), int(negative_logits.shape[1]))
        kth_negative = torch.topk(negative_logits, k=k, dim=1).values[:, -1]
        valid &= torch.isfinite(kth_negative)
        if not valid.any():
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        return torch.nn.functional.softplus(
            kth_negative[valid] - positive_best[valid] + margin
        ).mean()

    reaction_loss = directional(
        logits,
        positive_mask,
        reaction_denominator_mask | positive_mask,
    )
    protein_loss = directional(
        logits.T,
        positive_mask.T,
        protein_denominator_mask.T | positive_mask.T,
    )
    total = reaction_loss_weight * reaction_loss + (1 - reaction_loss_weight) * protein_loss
    return total, reaction_loss, protein_loss


def split_pairs(frame: pd.DataFrame, scope: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scope == "protein_cold":
        train = frame[frame["protein_fold"] != fold]
        test = frame[frame["protein_fold"] == fold]
    elif scope == "reaction_cold":
        train = frame[frame["reaction_fold"] != fold]
        test = frame[frame["reaction_fold"] == fold]
    elif scope == "double_cold":
        train = frame[(frame["protein_fold"] != fold) & (frame["reaction_fold"] != fold)]
        test = frame[(frame["protein_fold"] == fold) & (frame["reaction_fold"] == fold)]
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return train.drop_duplicates(["rhea_id", "Entry"]), test.drop_duplicates(["rhea_id", "Entry"])


def build_training_mask(
    train_pairs: pd.DataFrame,
    reaction_to_row: dict[str, int],
    protein_to_row: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_reactions = sorted(set(train_pairs["rhea_id"].astype(str)) & set(reaction_to_row))
    train_proteins = sorted(set(train_pairs["Entry"].astype(str)) & set(protein_to_row))
    local_reaction = {value: index for index, value in enumerate(train_reactions)}
    local_protein = {value: index for index, value in enumerate(train_proteins)}
    mask = np.zeros((len(train_reactions), len(train_proteins)), dtype=bool)
    for row in train_pairs.itertuples(index=False):
        reaction_id = str(row.rhea_id)
        entry = str(row.Entry)
        if reaction_id in local_reaction and entry in local_protein:
            mask[local_reaction[reaction_id], local_protein[entry]] = True
    return (
        np.array([reaction_to_row[value] for value in train_reactions], dtype=np.int64),
        np.array([protein_to_row[value] for value in train_proteins], dtype=np.int64),
        mask,
    )


def scheduled_hard_negative_k(
    hard_negative_k: int,
    epoch: int,
    start_epoch: int,
    end_epoch: int,
) -> int:
    if hard_negative_k < 0:
        raise ValueError("hard_negative_k must be non-negative")
    if epoch <= 0 or start_epoch <= 0 or end_epoch < 0:
        raise ValueError("epoch/start must be positive and end must be non-negative")
    return (
        hard_negative_k
        if epoch >= start_epoch and (end_epoch == 0 or epoch <= end_epoch)
        else 0
    )


def train_model(
    protein_features: torch.Tensor,
    reaction_features: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    config: ModelConfig,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    seed: int,
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    protein_group_map: dict[str, str] | None = None,
    reaction_group_map: dict[str, str] | None = None,
    exclude_same_group_negatives: bool = False,
    anchor_protein_rows: np.ndarray | None = None,
    anchor_protein_targets: torch.Tensor | None = None,
    anchor_reaction_rows: np.ndarray | None = None,
    anchor_reaction_targets: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    freeze_protein_tower: bool = False,
    freeze_reaction_tower: bool = False,
    reaction_loss_weight: float = 0.5,
    loss_mode: str = "bidirectional_infonce",
    hard_negative_k: int = 0,
    hard_negative_start_epoch: int = 1,
    hard_negative_end_epoch: int = 0,
    model_selection: str = "min_loss",
    topk_surrogate_weight: float = 0.0,
    topk_surrogate_k: int = 10,
    topk_surrogate_margin: float = 0.0,
) -> tuple[TerpeneDualTower, list[dict[str, float]]]:
    if hard_negative_start_epoch <= 0:
        raise ValueError("hard_negative_start_epoch must be positive")
    if hard_negative_end_epoch < 0:
        raise ValueError("hard_negative_end_epoch must be non-negative")
    if model_selection not in {"min_loss", "final"}:
        raise ValueError("model_selection must be min_loss or final")
    seed_everything(seed)
    model = TerpeneDualTower(config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    if freeze_protein_tower:
        for parameter in model.protein_tower.parameters():
            parameter.requires_grad = False
    if freeze_reaction_tower:
        for parameter in model.reaction_tower.parameters():
            parameter.requires_grad = False
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("At least one tower must remain trainable")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    reaction_rows, protein_rows, mask = build_training_mask(train_pairs, reaction_to_row, protein_to_row)
    if not len(reaction_rows) or not len(protein_rows) or not mask.any():
        raise ValueError("Training split has no usable positive pairs.")
    reaction_rows_tensor = torch.as_tensor(reaction_rows, dtype=torch.long, device=device)
    protein_rows_tensor = torch.as_tensor(protein_rows, dtype=torch.long, device=device)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)

    reaction_denominator = np.ones_like(mask, dtype=bool)
    protein_denominator = np.ones_like(mask, dtype=bool)
    if exclude_same_group_negatives:
        inverse_protein = {row: value for value, row in protein_to_row.items()}
        inverse_reaction = {row: value for value, row in reaction_to_row.items()}
        local_protein_groups = np.asarray(
            [protein_group_map.get(inverse_protein[int(row)], "") if protein_group_map else "" for row in protein_rows],
            dtype=object,
        )
        local_reaction_groups = np.asarray(
            [reaction_group_map.get(inverse_reaction[int(row)], "") if reaction_group_map else "" for row in reaction_rows],
            dtype=object,
        )
        for reaction_index in range(mask.shape[0]):
            positive_groups = set(local_protein_groups[mask[reaction_index]]) - {""}
            if positive_groups:
                potential = np.isin(local_protein_groups, list(positive_groups)) & ~mask[reaction_index]
                reaction_denominator[reaction_index, potential] = False
        for protein_index in range(mask.shape[1]):
            positive_groups = set(local_reaction_groups[mask[:, protein_index]]) - {""}
            if positive_groups:
                potential = np.isin(local_reaction_groups, list(positive_groups)) & ~mask[:, protein_index]
                protein_denominator[potential, protein_index] = False
    reaction_denominator_tensor = torch.as_tensor(reaction_denominator, dtype=torch.bool, device=device)
    protein_denominator_tensor = torch.as_tensor(protein_denominator, dtype=torch.bool, device=device)
    anchor_protein_rows_tensor = (
        torch.as_tensor(anchor_protein_rows, dtype=torch.long, device=device)
        if anchor_protein_rows is not None and len(anchor_protein_rows)
        else None
    )
    anchor_reaction_rows_tensor = (
        torch.as_tensor(anchor_reaction_rows, dtype=torch.long, device=device)
        if anchor_reaction_rows is not None and len(anchor_reaction_rows)
        else None
    )
    if anchor_protein_targets is not None:
        anchor_protein_targets = anchor_protein_targets.to(device)
    if anchor_reaction_targets is not None:
        anchor_reaction_targets = anchor_reaction_targets.to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reaction_embeddings = model.encode_reactions(reaction_features[reaction_rows_tensor])
        protein_embeddings = model.encode_proteins(protein_features[protein_rows_tensor])
        active_hard_negative_k = scheduled_hard_negative_k(
            hard_negative_k,
            epoch,
            hard_negative_start_epoch,
            hard_negative_end_epoch,
        )
        contrastive_loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
            reaction_embeddings,
            protein_embeddings,
            mask_tensor,
            temperature,
            reaction_denominator_tensor,
            protein_denominator_tensor,
            reaction_loss_weight,
            loss_mode,
            active_hard_negative_k,
        )
        topk_surrogate = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        topk_reaction_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        topk_protein_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        if topk_surrogate_weight > 0:
            topk_surrogate, topk_reaction_loss, topk_protein_loss = topk_hit_surrogate_loss(
                reaction_embeddings,
                protein_embeddings,
                mask_tensor,
                reaction_denominator_tensor,
                protein_denominator_tensor,
                temperature,
                topk_surrogate_k,
                topk_surrogate_margin,
                reaction_loss_weight,
            )
        protein_anchor_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        reaction_anchor_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        if anchor_weight > 0 and anchor_protein_rows_tensor is not None and anchor_protein_targets is not None:
            current_anchor_proteins = model.encode_proteins(protein_features[anchor_protein_rows_tensor])
            protein_anchor_loss = (1 - (current_anchor_proteins * anchor_protein_targets).sum(dim=1)).mean()
        if anchor_weight > 0 and anchor_reaction_rows_tensor is not None and anchor_reaction_targets is not None:
            current_anchor_reactions = model.encode_reactions(reaction_features[anchor_reaction_rows_tensor])
            reaction_anchor_loss = (1 - (current_anchor_reactions * anchor_reaction_targets).sum(dim=1)).mean()
        anchor_loss = 0.5 * (protein_anchor_loss + reaction_anchor_loss)
        loss = (
            contrastive_loss
            + topk_surrogate_weight * topk_surrogate
            + anchor_weight * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        current = float(loss.detach().cpu())
        if model_selection == "final" or current < best_loss:
            best_loss = current
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": current,
                    "contrastive_loss": float(contrastive_loss.detach().cpu()),
                    "reaction_loss": float(reaction_loss.detach().cpu()),
                    "protein_loss": float(protein_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "protein_anchor_loss": float(protein_anchor_loss.detach().cpu()),
                    "reaction_anchor_loss": float(reaction_anchor_loss.detach().cpu()),
                    "anchor_weight": float(anchor_weight),
                    "freeze_protein_tower": bool(freeze_protein_tower),
                    "freeze_reaction_tower": bool(freeze_reaction_tower),
                    "reaction_loss_weight": float(reaction_loss_weight),
                    "loss_mode": loss_mode,
                    "hard_negative_k": int(hard_negative_k),
                    "active_hard_negative_k": int(active_hard_negative_k),
                    "hard_negative_start_epoch": int(hard_negative_start_epoch),
                    "hard_negative_end_epoch": int(hard_negative_end_epoch),
                    "model_selection": model_selection,
                    "topk_surrogate_weight": float(topk_surrogate_weight),
                    "topk_surrogate_k": int(topk_surrogate_k),
                    "topk_surrogate_margin": float(topk_surrogate_margin),
                    "topk_surrogate_loss": float(topk_surrogate.detach().cpu()),
                    "topk_reaction_loss": float(topk_reaction_loss.detach().cpu()),
                    "topk_protein_loss": float(topk_protein_loss.detach().cpu()),
                    "reaction_negative_exclusion_rate": float(1.0 - reaction_denominator.mean()),
                    "protein_negative_exclusion_rate": float(1.0 - protein_denominator.mean()),
                }
            )
    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    return model, history


def rank_metrics(
    scores: np.ndarray,
    candidate_ids: list[str],
    positive_ids: set[str],
    masked_ids: set[str],
    budgets: tuple[int, ...],
) -> dict[str, float | int | None]:
    adjusted = scores.copy()
    candidate_to_index = {value: index for index, value in enumerate(candidate_ids)}
    for value in masked_ids - positive_ids:
        index = candidate_to_index.get(value)
        if index is not None:
            adjusted[index] = -np.inf
    order = np.lexsort((np.asarray(candidate_ids), -adjusted))
    ranked = [candidate_ids[index] for index in order if np.isfinite(adjusted[index])]
    positions = [index + 1 for index, value in enumerate(ranked) if value in positive_ids]
    best_rank = min(positions) if positions else None
    result: dict[str, float | int | None] = {
        "n_positives": len(positive_ids),
        "best_positive_rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
    }
    for budget in budgets:
        panel = ranked[:budget]
        hits = sum(value in positive_ids for value in panel)
        result[f"hits_at_{budget}"] = hits
        result[f"hit_at_{budget}"] = int(hits > 0)
        result[f"precision_at_{budget}"] = hits / budget
        result[f"positive_recall_at_{budget}"] = hits / len(positive_ids) if positive_ids else 0.0
    return result


def evaluate_bidirectional(
    model: TerpeneDualTower,
    protein_features: torch.Tensor,
    reaction_features: torch.Tensor,
    protein_ids: list[str],
    reaction_ids: list[str],
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    budgets: tuple[int, ...],
    scope: str,
    fold: int,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    with torch.no_grad():
        protein_embeddings = model.encode_proteins(protein_features).cpu().numpy()
        reaction_embeddings = model.encode_reactions(reaction_features).cpu().numpy()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    known_by_reaction = train_pairs.groupby("rhea_id")["Entry"].apply(lambda values: set(values.astype(str))).to_dict()
    known_by_protein = train_pairs.groupby("Entry")["rhea_id"].apply(lambda values: set(values.astype(str))).to_dict()
    records: list[dict[str, object]] = []

    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        if reaction_id not in reaction_to_row:
            continue
        positives = set(group["Entry"].astype(str)) & set(protein_to_row)
        if not positives:
            continue
        scores = reaction_embeddings[reaction_to_row[reaction_id]] @ protein_embeddings.T
        metrics = rank_metrics(scores, protein_ids, positives, known_by_reaction.get(reaction_id, set()), budgets)
        records.append(
            {
                "scope": scope,
                "fold": fold,
                "direction": "reaction_to_enzyme",
                "query_id": reaction_id,
                "candidate_count": len(protein_ids),
                "known_associations_masked": len(known_by_reaction.get(reaction_id, set())),
                **metrics,
            }
        )

    for entry, group in test_pairs.groupby("Entry", sort=True):
        if entry not in protein_to_row:
            continue
        positives = set(group["rhea_id"].astype(str)) & set(reaction_to_row)
        if not positives:
            continue
        scores = protein_embeddings[protein_to_row[entry]] @ reaction_embeddings.T
        metrics = rank_metrics(scores, reaction_ids, positives, known_by_protein.get(entry, set()), budgets)
        records.append(
            {
                "scope": scope,
                "fold": fold,
                "direction": "enzyme_to_reaction",
                "query_id": entry,
                "candidate_count": len(reaction_ids),
                "known_associations_masked": len(known_by_protein.get(entry, set())),
                **metrics,
            }
        )
    return records


def aggregate_metrics(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
        "mean_known_associations_masked": ("known_associations_masked", "mean"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"expected_hits_at_{budget}"] = (f"hits_at_{budget}", "mean")
        aggregations[f"precision_at_{budget}"] = (f"precision_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["scope", "direction"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bidirectional TPS reaction/protein dual towers under cold splits.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scopes", default=",".join(DEFAULT_SCOPES))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--reaction-feature-mode", choices=["drfp_categorical", "multiview"], default="drfp_categorical")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scopes = parse_str_tuple(args.scopes)
    budgets = parse_int_tuple(args.budgets)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    protein_matrix, protein_ids = load_protein_features(args.embedding_dir.resolve())
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry", "smiles_seq"]].drop_duplicates(["rhea_id", "Entry"])
    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(
        positives,
        args.reaction_feature_mode,
    )
    split_frame = pd.read_csv(args.splits, dtype=str).fillna("")
    split_frame["protein_fold"] = pd.to_numeric(split_frame["protein_fold"]).astype(int)
    split_frame["reaction_fold"] = pd.to_numeric(split_frame["reaction_fold"]).astype(int)

    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    split_frame = split_frame[
        split_frame["Entry"].isin(protein_to_row) & split_frame["rhea_id"].isin(reaction_to_row)
    ].copy()

    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    config = ModelConfig(
        protein_input_dim=protein_matrix.shape[1],
        reaction_input_dim=reaction_matrix.shape[1],
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    feature_schema.update(
        {
            "reaction_ids": reaction_ids,
            "protein_ids_file": str((args.embedding_dir / "entries.csv").resolve()),
            "reaction_feature_dimension": reaction_matrix.shape[1],
            "protein_feature_dimension": protein_matrix.shape[1],
        }
    )
    (output_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    reaction_table.to_csv(output_dir / "reaction_features.csv", index=False)
    np.save(output_dir / "reaction_feature_matrix.npy", reaction_matrix)

    all_records: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    folds = sorted(set(split_frame["protein_fold"]) | set(split_frame["reaction_fold"]))
    for scope in scopes:
        for fold in folds:
            train_pairs, test_pairs = split_pairs(split_frame, scope, fold)
            if train_pairs.empty or test_pairs.empty:
                continue
            model, history = train_model(
                protein_tensor,
                reaction_tensor,
                train_pairs,
                protein_to_row,
                reaction_to_row,
                config,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.temperature,
                args.seed + fold + 100 * DEFAULT_SCOPES.index(scope),
                device,
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "scope": scope,
                "fold": int(fold),
                "temperature": args.temperature,
                "feature_schema": feature_schema,
            }
            torch.save(checkpoint, model_dir / f"{scope}_fold{fold}.pt")
            for item in history:
                training_rows.append(
                    {
                        "scope": scope,
                        "fold": fold,
                        "n_train_pairs": len(train_pairs),
                        "n_test_pairs": len(test_pairs),
                        **item,
                    }
                )
            all_records.extend(
                evaluate_bidirectional(
                    model,
                    protein_tensor,
                    reaction_tensor,
                    protein_ids,
                    reaction_ids,
                    train_pairs,
                    test_pairs,
                    budgets,
                    scope,
                    int(fold),
                    device,
                )
            )

    query_metrics = pd.DataFrame(all_records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    training_history = pd.DataFrame(training_rows)
    training_history.to_csv(output_dir / "training_history.csv", index=False)
    aggregate = aggregate_metrics(query_metrics, budgets)
    aggregate.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "model_config": asdict(config),
        "device": str(device),
        "scopes": scopes,
        "folds": folds,
        "budgets": budgets,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "reaction_feature_mode": args.reaction_feature_mode,
        "n_positive_pairs": int(len(split_frame)),
        "n_proteins": int(len(protein_ids)),
        "n_reactions": int(len(reaction_ids)),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "training_history": str(output_dir / "training_history.csv"),
            "models": str(model_dir),
            "feature_schema": str(output_dir / "feature_schema.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
