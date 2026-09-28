from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.bime_rank.support import evaluate_multi_expert_protocol_comparison as base

CENTER_DIR = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"


class ReactionCenterModeEvidence(nn.Module):
    def __init__(self, center_dim: int, n_experts: int) -> None:
        super().__init__()
        self.head = nn.Linear(center_dim, n_experts, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, base_gates: torch.Tensor, center_values: torch.Tensor) -> torch.Tensor:
        evidence = self.head(center_values)
        return torch.softmax(base_gates.clamp_min(1e-12).log() + evidence, dim=-1)


def load_center_features(reaction_ids: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    manifest = json.loads((CENTER_DIR / "manifest.json").read_text())
    center_dim = int(manifest["reaction_center_dimension"])
    entries = pd.read_csv(CENTER_DIR / "entries.csv", dtype=str).fillna("")
    matrix = np.load(CENTER_DIR / "reaction_feature_matrix.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError("reaction-center entries/matrix mismatch")
    center = matrix[:, -center_dim:]
    row_by_id = {
        str(r.reaction_id): int(i)
        for i, r in enumerate(entries.itertuples(index=False))
    }
    out = np.zeros((len(reaction_ids), center_dim), dtype=np.float32)
    observed = 0
    for i, rid in enumerate(reaction_ids):
        row = row_by_id.get(str(rid))
        if row is None:
            continue
        out[i] = center[row]
        if np.count_nonzero(out[i]):
            observed += 1
    return out, {
        "center_dim": center_dim,
        "reaction_count": len(reaction_ids),
        "nonzero_center_count": observed,
        "source_manifest": str(CENTER_DIR / "manifest.json"),
    }


def train_evidence_head(
    model: base.DirectionalMultiExpertDualTower,
    center_tensor: torch.Tensor,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    protein_groups: dict[str, str],
    reaction_groups: dict[str, str],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    topk_terms: tuple[tuple[int, float], ...],
    topk_margin: float,
    device: torch.device,
) -> tuple[ReactionCenterModeEvidence, list[dict[str, float]]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    reaction_rows, protein_rows, positive_mask = base.build_training_mask(
        train_pairs, reaction_to_row, protein_to_row
    )
    reaction_denominator, _ = base.build_denominator_masks(
        train_pairs,
        reaction_rows,
        protein_rows,
        positive_mask,
        reaction_to_row,
        protein_to_row,
        reaction_groups,
        protein_groups,
    )
    rr = torch.as_tensor(reaction_rows, dtype=torch.long, device=device)
    pr = torch.as_tensor(protein_rows, dtype=torch.long, device=device)
    positive = torch.as_tensor(positive_mask, dtype=torch.bool, device=device)
    denominator = torch.as_tensor(reaction_denominator, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, _, diagnostics = model.score_matrices(
            protein_tensor[pr], reaction_tensor[rr]
        )
        chart_scores = diagnostics["chart_scores"].detach()
        base_gates = diagnostics["reaction_gates"].detach()
        mix = diagnostics["r2e_mix"].detach()
        center_values = center_tensor[rr].detach()

    adapter = ReactionCenterModeEvidence(
        center_values.shape[1],
        model.config.n_experts,
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        refined = adapter(base_gates, center_values)
        partition = torch.cat(
            [
                torch.ones_like(refined[:, :1]) * (1 - mix),
                refined * mix,
            ],
            dim=-1,
        )
        scores = (chart_scores * partition[:, None, :]).sum(dim=-1)
        logits = scores / temperature
        contrastive = base.directional_multi_positive_loss(
            logits, positive, denominator, hard_negative_k=0
        )
        topk = base.directional_topk_surrogate(
            logits, positive, denominator, topk_terms, topk_margin
        )
        loss = contrastive + topk
        loss.backward()
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss:
            best_loss = current
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            with torch.no_grad():
                shift = (
                    refined.clamp_min(1e-12)
                    * (
                        refined.clamp_min(1e-12).log()
                        - base_gates.clamp_min(1e-12).log()
                    )
                ).sum(dim=-1)
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": current,
                    "contrastive_loss": float(contrastive.detach().cpu()),
                    "topk_loss": float(topk.detach().cpu()),
                    "mean_kl_refined_to_base": float(shift.mean().detach().cpu()),
                }
            )
    if best_state is None:
        raise RuntimeError("reaction-center evidence head did not train")
    adapter.load_state_dict(best_state)
    return adapter, history


def score_with_center(
    model: base.DirectionalMultiExpertDualTower,
    adapter: ReactionCenterModeEvidence,
    center_tensor: torch.Tensor,
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    adapter.eval()
    with torch.no_grad():
        _, e2r, diagnostics = model.score_matrices(protein_tensor, reaction_tensor)
        refined = adapter(diagnostics["reaction_gates"], center_tensor)
        mix = diagnostics["r2e_mix"]
        partition = torch.cat(
            [
                torch.ones_like(refined[:, :1]) * (1 - mix),
                refined * mix,
            ],
            dim=-1,
        )
        r2e = (diagnostics["chart_scores"] * partition[:, None, :]).sum(dim=-1)
        kl = (
            refined.clamp_min(1e-12)
            * (
                refined.clamp_min(1e-12).log()
                - diagnostics["reaction_gates"].clamp_min(1e-12).log()
            )
        ).sum(dim=-1).mean()
    return (
        r2e.cpu().numpy(),
        e2r.cpu().numpy(),
        float(kl.cpu()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=Path, default=base.DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=base.DEFAULT_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=base.DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=base.DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=base.DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=base.DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-partition", choices=["development", "frozen"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--evidence-epochs", type=int, default=80)
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
    parser.add_argument("--topk-terms", default="3:0.10,10:0.05,20:0.025")
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # prepare_data expects this selector.
    args.reaction_feature_mode = "multiview"
    args.glue_weight = 0.0
    args.mechanism_auxiliary_weight = 0.0
    args.mechanism_score_weight = 0.0
    args.hard_negative_k = 0
    args.hard_negative_start_epoch = 20

    data = base.prepare_data(args)
    protein_matrix = data["protein_matrix"]
    reaction_matrix = data["reaction_matrix"]
    protein_ids = data["protein_ids"]
    reaction_ids = data["reaction_ids"]
    protein_to_row = data["protein_to_row"]
    reaction_to_row = data["reaction_to_row"]
    pairs = data["pairs"]
    protein_groups = data["protein_groups"]
    reaction_groups = data["reaction_groups"]
    center_matrix, center_meta = load_center_features(reaction_ids)

    device = torch.device(args.device)
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    center_tensor = torch.as_tensor(center_matrix, dtype=torch.float32, device=device)
    topk_terms = base.parse_topk_terms(args.topk_terms)
    config = base.MultiExpertConfig(
        protein_input_dim=int(protein_matrix.shape[1]),
        reaction_input_dim=int(reaction_matrix.shape[1]),
        hidden_dim=args.hidden_dim,
        global_dim=args.global_dim,
        n_experts=args.n_experts,
        expert_dim=args.expert_dim,
        dropout=args.dropout,
        gate_temperature=args.gate_temperature,
        expert_mix_init=args.expert_mix_init,
    )

    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in pairs.groupby("rhea_id", sort=True)
    }
    all_positive_by_protein = {
        protein_id: set(group["rhea_id"].astype(str))
        for protein_id, group in pairs.groupby("Entry", sort=True)
    }
    budgets = (1, 3, 5, 10, 20, 50, 100)
    records: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

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

            model, history = base.train_multi_expert(
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
                hard_negative_k=0,
                hard_negative_start_epoch=20,
                topk_terms=topk_terms,
                topk_margin=args.topk_margin,
                balance_weight=args.balance_weight,
                entropy_weight=args.entropy_weight,
                diversity_weight=args.diversity_weight,
                glue_weight=0.0,
                reaction_precursor_map=data["reaction_precursor_map"],
                reaction_skeleton_map=data["reaction_skeleton_map"],
                mechanism_values=data["mechanism_values"],
                mechanism_auxiliary_weight=0.0,
                seed=args.seed,
                device=device,
            )
            adapter, evidence_history = train_evidence_head(
                model,
                center_tensor,
                protein_tensor,
                reaction_tensor,
                train_pairs,
                protein_to_row,
                reaction_to_row,
                protein_groups,
                reaction_groups,
                epochs=args.evidence_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                topk_terms=topk_terms,
                topk_margin=args.topk_margin,
                device=device,
            )
            r2e_matrix, e2r_matrix, mean_kl = score_with_center(
                model, adapter, center_tensor, protein_tensor, reaction_tensor
            )
            training_rows.append(
                {
                    "protein_fold": protein_fold,
                    "reaction_fold": reaction_fold,
                    "base_best_loss": min(x["loss"] for x in history),
                    "evidence_best_loss": min(x["loss"] for x in evidence_history),
                    "mean_kl_refined_to_base_all_reactions": mean_kl,
                }
            )

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                known_other = all_positive_by_reaction.get(reaction_id, set()) - positives
                records.append(
                    {
                        "protocol": "double_cold_25cell",
                        "direction": "reaction_to_enzyme",
                        "query_id": reaction_id,
                        "protein_fold": protein_fold,
                        "reaction_fold": reaction_fold,
                        **base.masked_rank_metrics(
                            r2e_matrix[reaction_to_row[reaction_id]],
                            protein_ids,
                            positives,
                            known_other,
                            budgets,
                        ),
                    }
                )
            for protein_id, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                known_other = all_positive_by_protein.get(protein_id, set()) - positives
                records.append(
                    {
                        "protocol": "double_cold_25cell",
                        "direction": "enzyme_to_reaction",
                        "query_id": protein_id,
                        "protein_fold": protein_fold,
                        "reaction_fold": reaction_fold,
                        **base.masked_rank_metrics(
                            e2r_matrix[:, protein_to_row[protein_id]],
                            reaction_ids,
                            positives,
                            known_other,
                            budgets,
                        ),
                    }
                )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    query = pd.DataFrame(records)
    metrics = base.aggregate_directional(query, budgets)
    query.to_csv(out / "query_metrics.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(out / "evidence_training.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "method": "fibre_reaction_center_mode_evidence_v1",
                "strict_partition": args.strict_partition,
                "center_meta": center_meta,
                "base": "v2 no-glue directional conditional modes",
                "evidence_update": "q_plus proportional to q_base * exp(W_center c), W zero initialized and bias free",
                "base_parameters_frozen_during_evidence_training": True,
                "e2r_score_path_changed": False,
                "evidence_epochs": args.evidence_epochs,
                "seed": args.seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
