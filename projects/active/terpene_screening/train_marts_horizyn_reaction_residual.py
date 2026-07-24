from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_dual_tower_cold import (
    ModelConfig,
    ProjectionTower,
    TerpeneDualTower,
    build_training_mask,
    multi_positive_contrastive_loss,
    rank_metrics,
    seed_everything,
    topk_hit_surrogate_loss,
)
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (
    build_denominator_masks,
)
from projects.active.terpene_screening.train_marts_domain_adaptation import (
    DEFAULT_BUDGETS,
    DEFAULT_CACHE,
    DEFAULT_CURRENT_CANDIDATES,
    DEFAULT_CURRENT_PROTEINS,
    DEFAULT_EXTERNAL_PROTEINS,
    DEFAULT_MARTS,
    DEFAULT_MMSEQS,
    DEFAULT_PRODUCTION_DIR,
    aggregate,
    assign_folds,
    build_mmseqs_clusters,
    build_pair_table,
    build_protein_entities,
    build_reaction_clusters,
    build_reaction_entities,
    ensemble_scores,
    evaluate_fold,
    load_feature_schema,
    load_production_payloads,
)

DEFAULT_AUX = ROOT / "results/terpene_horizyn_adapter_full/horizyn_reaction_embeddings.npy"
DEFAULT_AUX_ENTRIES = ROOT / "data/terpene_marts_adaptation/reaction_entities.csv"
DEFAULT_PAIR_FOLDS = ROOT / "data/terpene_marts_adaptation/marts_pair_folds.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_horizyn_reaction_residual"


class ResidualReactionDualTower(nn.Module):
    def __init__(
        self,
        base_config: ModelConfig,
        aux_input_dim: int,
        aux_hidden_dim: int,
        gate_init: float,
        vector_gate: bool = False,
    ) -> None:
        super().__init__()
        self.base_config = base_config
        self.protein_tower = ProjectionTower(
            base_config.protein_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.base_reaction_tower = ProjectionTower(
            base_config.reaction_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.aux_reaction_tower = ProjectionTower(
            aux_input_dim,
            aux_hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        gate_shape = (base_config.embedding_dim,) if vector_gate else ()
        self.gate_logit = nn.Parameter(torch.full(gate_shape, float(gate_init)))

    def load_base_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for key, value in state_dict.items():
            if key.startswith("protein_tower."):
                target = key
            elif key.startswith("reaction_tower."):
                target = "base_reaction_tower." + key[len("reaction_tower.") :]
            else:
                continue
            if target not in own or own[target].shape != value.shape:
                raise ValueError(f"Cannot map production parameter {key} -> {target}")
            own[target].copy_(value)
        self.load_state_dict(own)

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_tower(values)

    def encode_reactions(
        self,
        base_values: torch.Tensor,
        aux_values: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_reaction_tower(base_values)
        auxiliary = self.aux_reaction_tower(aux_values)
        gate = torch.sigmoid(self.gate_logit)
        return F.normalize(base + gate * auxiliary, dim=-1)

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit).detach().mean().cpu())


def load_auxiliary_features(
    path: Path,
    entries_path: Path,
    reaction_ids: list[str],
) -> np.ndarray:
    matrix = np.load(path.resolve()).astype(np.float32)
    entries = pd.read_csv(entries_path.resolve(), dtype=str).fillna("")
    id_column = "reaction_id" if "reaction_id" in entries.columns else "rhea_id"
    if len(entries) != len(matrix):
        raise ValueError("Auxiliary reaction entries and matrix length differ")
    mapping = {value: index for index, value in enumerate(entries[id_column].astype(str))}
    missing = [value for value in reaction_ids if value not in mapping]
    if missing:
        raise ValueError(f"Auxiliary reaction matrix missing IDs: {missing[:10]}")
    aligned = np.stack([matrix[mapping[value]] for value in reaction_ids]).astype(np.float32)
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return aligned / norms


def train_residual_model(
    protein_features: torch.Tensor,
    base_reaction_features: torch.Tensor,
    aux_reaction_features: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_clusters: dict[str, str],
    reaction_clusters: dict[str, str],
    base_config: ModelConfig,
    base_state: dict[str, torch.Tensor],
    aux_hidden_dim: int,
    gate_init: float,
    vector_gate: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    hard_negative_k: int,
    topk_surrogate_weight: float,
    topk_surrogate_k: int,
    topk_surrogate_margin: float,
    pu_group_mask: bool,
    freeze_base_reaction: bool,
    seed: int,
    device: torch.device,
) -> tuple[ResidualReactionDualTower, list[dict[str, object]]]:
    seed_everything(seed)
    model = ResidualReactionDualTower(
        base_config,
        aux_reaction_features.shape[1],
        aux_hidden_dim,
        gate_init,
        vector_gate,
    ).to(device)
    model.load_base_state(base_state)
    if freeze_base_reaction:
        for parameter in model.base_reaction_tower.parameters():
            parameter.requires_grad = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    reaction_rows, protein_rows, positive_mask = build_training_mask(
        train_pairs, reaction_to_row, protein_to_row
    )
    local_reactions = [reaction_ids[int(row)] for row in reaction_rows]
    local_proteins = [protein_ids[int(row)] for row in protein_rows]
    reaction_denominator, protein_denominator = build_denominator_masks(
        positive_mask,
        local_reactions,
        local_proteins,
        reaction_clusters,
        protein_clusters,
        pu_group_mask,
    )
    rr = torch.as_tensor(reaction_rows, dtype=torch.long, device=device)
    pr = torch.as_tensor(protein_rows, dtype=torch.long, device=device)
    positives = torch.as_tensor(positive_mask, dtype=torch.bool, device=device)
    rden = torch.as_tensor(reaction_denominator, dtype=torch.bool, device=device)
    pden = torch.as_tensor(protein_denominator, dtype=torch.bool, device=device)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        proteins = model.encode_proteins(protein_features[pr])
        reactions = model.encode_reactions(
            base_reaction_features[rr], aux_reaction_features[rr]
        )
        contrastive_loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
            reactions,
            proteins,
            positives,
            temperature,
            rden,
            pden,
            reaction_loss_weight,
            "bidirectional_infonce",
            hard_negative_k,
        )
        topk_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        topk_reaction_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        topk_protein_loss = torch.zeros((), dtype=contrastive_loss.dtype, device=device)
        if topk_surrogate_weight > 0:
            logits = reactions @ proteins.T / temperature
            topk_loss, topk_reaction_loss, topk_protein_loss = topk_hit_surrogate_loss(
                logits,
                positives,
                rden,
                pden,
                topk_surrogate_k,
                topk_surrogate_margin,
                reaction_loss_weight,
            )
        loss = contrastive_loss + topk_surrogate_weight * topk_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": value,
                    "contrastive_loss": float(contrastive_loss.detach().cpu()),
                    "reaction_loss": float(reaction_loss.detach().cpu()),
                    "protein_loss": float(protein_loss.detach().cpu()),
                    "topk_surrogate_loss": float(topk_loss.detach().cpu()),
                    "topk_reaction_loss": float(topk_reaction_loss.detach().cpu()),
                    "topk_protein_loss": float(topk_protein_loss.detach().cpu()),
                    "topk_surrogate_weight": topk_surrogate_weight,
                    "topk_surrogate_k": topk_surrogate_k,
                    "topk_surrogate_margin": topk_surrogate_margin,
                    "gate_value": model.gate_value(),
                    "hard_negative_k": hard_negative_k,
                    "freeze_base_reaction": freeze_base_reaction,
                }
            )
    if best_state is None:
        raise RuntimeError("No residual model state selected")
    model.load_state_dict(best_state)
    return model, history


def encode_residual_models(
    models: list[ResidualReactionDualTower],
    protein_tensor: torch.Tensor,
    base_reaction_tensor: torch.Tensor,
    aux_reaction_tensor: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    proteins: list[np.ndarray] = []
    reactions: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            model.eval()
            proteins.append(model.encode_proteins(protein_tensor).cpu().numpy())
            reactions.append(
                model.encode_reactions(base_reaction_tensor, aux_reaction_tensor)
                .cpu()
                .numpy()
            )
    return proteins, reactions


def main() -> None:
    parser = argparse.ArgumentParser(description="MARTS double-cold adaptation with a zero-initialized Horizyn reaction residual branch.")
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--external-protein-dir", type=Path, default=DEFAULT_EXTERNAL_PROTEINS)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--aux-features", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--aux-entries", type=Path, default=DEFAULT_AUX_ENTRIES)
    parser.add_argument(
        "--pair-folds",
        type=Path,
        default=DEFAULT_PAIR_FOLDS,
        help="Canonical strict double-cold pair/cluster/fold manifest; pair equality is mandatory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-mode", choices=["paired", "cartesian"], default="cartesian")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.75)
    parser.add_argument("--hard-negative-k", type=int, default=0)
    parser.add_argument("--topk-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--topk-surrogate-k", type=int, default=10)
    parser.add_argument("--topk-surrogate-margin", type=float, default=0.1)
    parser.add_argument("--pu-group-mask", action="store_true")
    parser.add_argument("--aux-hidden-dim", type=int, default=512)
    parser.add_argument("--gate-init", type=float, default=-4.0)
    parser.add_argument("--vector-gate", action="store_true")
    parser.add_argument("--freeze-base-reaction", action="store_true")
    parser.add_argument("--protein-identity", type=float, default=0.5)
    parser.add_argument("--reaction-threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)

    marts = pd.read_csv(args.marts, sep="\t", dtype=str).fillna("")
    schema = load_feature_schema(args.production_dir.resolve())
    protein_matrix, protein_ids, protein_table, enzyme_to_entity = build_protein_entities(
        marts,
        args.current_protein_dir.resolve(),
        args.external_protein_dir.resolve(),
        args.current_candidates.resolve(),
    )
    reaction_matrix, reaction_ids, reaction_table, signature_to_entity = build_reaction_entities(marts, schema)
    aux_matrix = load_auxiliary_features(args.aux_features, args.aux_entries, reaction_ids)
    protein_seen = dict(zip(protein_table["protein_id"], protein_table["enzyme_seen"].astype(bool)))
    reaction_seen = dict(zip(reaction_table["reaction_id"], reaction_table["reaction_seen"].astype(bool)))
    pairs = build_pair_table(marts, enzyme_to_entity, signature_to_entity, protein_seen, reaction_seen)
    pair_folds = pd.read_csv(args.pair_folds.resolve(), dtype=str).fillna("")
    required_fold_columns = {
        "Entry",
        "rhea_id",
        "protein_cluster",
        "reaction_cluster",
        "protein_fold",
        "reaction_fold",
    }
    if not required_fold_columns.issubset(pair_folds.columns):
        raise ValueError(
            f"Pair-fold manifest lacks columns: {sorted(required_fold_columns - set(pair_folds.columns))}"
        )
    pair_keys = set(map(tuple, pairs[["Entry", "rhea_id"]].astype(str).to_numpy()))
    manifest_keys = set(
        map(tuple, pair_folds[["Entry", "rhea_id"]].astype(str).to_numpy())
    )
    if pair_keys != manifest_keys:
        raise ValueError(
            "Current MARTS pairs differ from frozen pair-fold manifest: "
            f"current_only={len(pair_keys - manifest_keys)}, "
            f"manifest_only={len(manifest_keys - pair_keys)}"
        )
    fold_columns = [
        "Entry",
        "rhea_id",
        "protein_cluster",
        "reaction_cluster",
        "protein_fold",
        "reaction_fold",
    ]
    pairs = pairs.drop(
        columns=[
            value
            for value in ["protein_cluster", "reaction_cluster", "protein_fold", "reaction_fold"]
            if value in pairs.columns
        ]
    ).merge(
        pair_folds[fold_columns].drop_duplicates(["Entry", "rhea_id"]),
        on=["Entry", "rhea_id"],
        how="left",
        validate="one_to_one",
    )
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    if pairs[["protein_cluster", "reaction_cluster"]].eq("").any().any():
        raise ValueError("Frozen pair-fold manifest contains empty cluster assignments")
    protein_clusters = (
        pairs.groupby("Entry")["protein_cluster"].first().astype(str).to_dict()
    )
    reaction_clusters = (
        pairs.groupby("rhea_id")["reaction_cluster"].first().astype(str).to_dict()
    )

    payloads = load_production_payloads(args.production_dir.resolve(), device)
    base_config = ModelConfig(**payloads[0]["model_config"])
    if base_config.protein_input_dim != protein_matrix.shape[1] or base_config.reaction_input_dim != reaction_matrix.shape[1]:
        raise ValueError("Production dimensions do not match MARTS features")
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    aux_tensor = torch.as_tensor(aux_matrix, dtype=torch.float32, device=device)

    baseline_models: list[TerpeneDualTower] = []
    for payload in payloads:
        model = TerpeneDualTower(base_config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        baseline_models.append(model)
    with torch.no_grad():
        baseline_proteins = [model.encode_proteins(protein_tensor).cpu().numpy() for model in baseline_models]
        baseline_reactions = [model.encode_reactions(reaction_tensor).cpu().numpy() for model in baseline_models]
    baseline_scores = ensemble_scores(baseline_proteins, baseline_reactions)

    split_specs = (
        [(fold, fold) for fold in range(args.n_folds)]
        if args.fold_mode == "paired"
        else [(p, r) for p in range(args.n_folds) for r in range(args.n_folds)]
    )
    records: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    for split_index, (protein_test_fold, reaction_test_fold) in enumerate(split_specs):
        split_id = f"p{protein_test_fold}_r{reaction_test_fold}"
        train_pairs = pairs[
            pairs["protein_fold"].ne(protein_test_fold)
            & pairs["reaction_fold"].ne(reaction_test_fold)
        ].copy()
        test_pairs = pairs[
            pairs["protein_fold"].eq(protein_test_fold)
            & pairs["reaction_fold"].eq(reaction_test_fold)
            & (~pairs["protein_seen"])
            & (~pairs["reaction_seen"])
        ].copy()
        split_rows.append(
            {
                "split_id": split_id,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "test_proteins": test_pairs["Entry"].nunique(),
                "test_reactions": test_pairs["rhea_id"].nunique(),
            }
        )
        if test_pairs.empty:
            continue
        evaluate_fold(records, "current_production", split_id, baseline_scores, test_pairs, protein_ids, reaction_ids, budgets)
        adapted_models: list[ResidualReactionDualTower] = []
        for payload_index, payload in enumerate(payloads):
            seed = int(payload.get("seed", 20260723 + payload_index)) + split_index * 1000
            model, history = train_residual_model(
                protein_tensor,
                reaction_tensor,
                aux_tensor,
                train_pairs,
                protein_ids,
                reaction_ids,
                protein_clusters,
                reaction_clusters,
                base_config,
                payload["model_state_dict"],
                args.aux_hidden_dim,
                args.gate_init,
                args.vector_gate,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.temperature,
                args.reaction_loss_weight,
                args.hard_negative_k,
                args.topk_surrogate_weight,
                args.topk_surrogate_k,
                args.topk_surrogate_margin,
                args.pu_group_mask,
                args.freeze_base_reaction,
                seed,
                device,
            )
            adapted_models.append(model)
            frame = pd.DataFrame(history)
            frame.insert(0, "checkpoint_index", payload_index)
            frame.insert(0, "split_id", split_id)
            histories.append(frame)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_model_config": asdict(base_config),
                    "aux_input_dim": aux_matrix.shape[1],
                    "aux_hidden_dim": args.aux_hidden_dim,
                    "gate_init": args.gate_init,
                    "vector_gate": args.vector_gate,
                "topk_surrogate_weight": args.topk_surrogate_weight,
                "topk_surrogate_k": args.topk_surrogate_k,
                "topk_surrogate_margin": args.topk_surrogate_margin,
                    "split_id": split_id,
                },
                model_dir / f"residual_{split_id}_model{payload_index}.pt",
            )
        adapted_proteins, adapted_reactions = encode_residual_models(
            adapted_models, protein_tensor, reaction_tensor, aux_tensor
        )
        adapted_scores = ensemble_scores(adapted_proteins, adapted_reactions)
        evaluate_fold(records, "horizyn_reaction_residual", split_id, adapted_scores, test_pairs, protein_ids, reaction_ids, budgets)

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
    summary = {
        "production_dir": str(args.production_dir.resolve()),
        "pair_folds": str(args.pair_folds.resolve()),
        "aux_features": str(args.aux_features.resolve()),
        "aux_dimension": int(aux_matrix.shape[1]),
        "aux_hidden_dim": args.aux_hidden_dim,
        "gate_init": args.gate_init,
        "vector_gate": args.vector_gate,
        "freeze_base_reaction": args.freeze_base_reaction,
        "epochs": args.epochs,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "topk_surrogate_weight": args.topk_surrogate_weight,
        "topk_surrogate_k": args.topk_surrogate_k,
        "topk_surrogate_margin": args.topk_surrogate_margin,
        "pu_group_mask": args.pu_group_mask,
        "n_splits": len(split_specs),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "history": str(output_dir / "training_history.csv"),
            "models": str(model_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
