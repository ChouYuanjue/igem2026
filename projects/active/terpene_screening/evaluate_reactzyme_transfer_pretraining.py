from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (
    aggregate,
    masked_rank_metrics,
)
from projects.active.terpene_screening.train_dual_tower_cold import (
    ModelConfig,
    TerpeneDualTower,
    build_reaction_features,
    load_protein_features,
    multi_positive_contrastive_loss,
    seed_everything,
    train_model,
)

DEFAULT_CLEAN = ROOT / "data/external/reactzyme_transfer/global_clean_v2"
DEFAULT_RZ_EMBEDDINGS = ROOT / "data/external/reactzyme_transfer/esmc600m_mean"
DEFAULT_TPS_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_TPS_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_reactzyme_transfer_pretraining"
DEFAULT_BUDGETS = (3, 5, 10, 20)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def load_aligned_embeddings(directory: Path) -> tuple[np.ndarray, list[str]]:
    entries = pd.read_csv(directory / "entries.csv", dtype=str).fillna("")
    entries["row"] = pd.to_numeric(entries["row"]).astype(int)
    entries = entries.sort_values("row")
    matrix = np.load(directory / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError(f"Embedding rows differ under {directory}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms, entries["Entry"].astype(str).tolist()


def make_batch_mask(
    pair_rows: np.ndarray,
    protein_indices: np.ndarray,
    reaction_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_proteins = np.unique(protein_indices[pair_rows])
    local_reactions = np.unique(reaction_indices[pair_rows])
    protein_local = {int(row): index for index, row in enumerate(local_proteins)}
    reaction_local = {int(row): index for index, row in enumerate(local_reactions)}
    positive = np.zeros((len(local_reactions), len(local_proteins)), dtype=bool)
    for row in pair_rows:
        positive[
            reaction_local[int(reaction_indices[row])],
            protein_local[int(protein_indices[row])],
        ] = True
    return local_proteins, local_reactions, positive


def pretrain_model(
    *,
    protein_matrix: np.ndarray,
    reaction_matrix: np.ndarray,
    protein_indices: np.ndarray,
    reaction_indices: np.ndarray,
    config: ModelConfig,
    epochs: int,
    batch_pairs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    hard_negative_k: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    if epochs <= 0 or batch_pairs <= 0:
        raise ValueError("Pretraining epochs and batch-pairs must be positive")
    seed_everything(seed)
    model = TerpeneDualTower(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    pair_order = np.arange(len(protein_indices), dtype=np.int64)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        rng.shuffle(pair_order)
        losses: list[float] = []
        reaction_losses: list[float] = []
        protein_losses: list[float] = []
        model.train()
        for start in range(0, len(pair_order), batch_pairs):
            batch = pair_order[start : start + batch_pairs]
            local_proteins, local_reactions, positive = make_batch_mask(
                batch, protein_indices, reaction_indices
            )
            protein_values = torch.as_tensor(
                protein_matrix[local_proteins], dtype=torch.float32, device=device
            )
            reaction_values = torch.as_tensor(
                reaction_matrix[local_reactions], dtype=torch.float32, device=device
            )
            positive_tensor = torch.as_tensor(
                positive, dtype=torch.bool, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            protein_embeddings = model.encode_proteins(protein_values)
            reaction_embeddings = model.encode_reactions(reaction_values)
            loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
                reaction_embeddings,
                protein_embeddings,
                positive_tensor,
                temperature,
                reaction_loss_weight=reaction_loss_weight,
                hard_negative_k=hard_negative_k,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            reaction_losses.append(float(reaction_loss.detach().cpu()))
            protein_losses.append(float(protein_loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append(
            {
                "epoch": float(epoch),
                "mean_loss": mean_loss,
                "mean_reaction_loss": float(np.mean(reaction_losses)),
                "mean_protein_loss": float(np.mean(protein_losses)),
                "steps": float(len(losses)),
            }
        )
    if best_state is None:
        raise RuntimeError("ReactZyme pretraining did not produce a state")
    return best_state, history


def build_data(args: argparse.Namespace) -> dict[str, object]:
    clean_pairs = pd.read_csv(args.clean_dir / "clean_pairs.csv", dtype=str).fillna("")
    clean_reactions = pd.read_csv(
        args.clean_dir / "clean_reactions.csv", dtype=str
    ).fillna("")
    rz_matrix, rz_ids = load_aligned_embeddings(args.reactzyme_embeddings)
    rz_to_row = {value: index for index, value in enumerate(rz_ids)}
    clean_pairs = clean_pairs[clean_pairs["Entry"].isin(rz_to_row)].copy()

    tps_matrix, tps_ids = load_protein_features(args.tps_embeddings)
    tps = pd.read_csv(args.tps_positives, sep="\t", dtype=str).fillna("")
    tps = tps[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    tps_to_row = {value: index for index, value in enumerate(tps_ids)}
    tps = tps[tps["Entry"].isin(tps_to_row)].copy()

    union_reactions = pd.concat(
        [
            clean_reactions[["rhea_id", "smiles_seq"]],
            tps[["rhea_id", "smiles_seq"]],
        ],
        ignore_index=True,
    ).drop_duplicates("rhea_id")
    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(
        union_reactions.assign(Entry="external"), "multiview"
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    clean_pairs = clean_pairs[
        clean_pairs["rhea_id"].isin(reaction_to_row)
    ].drop_duplicates(["Entry", "rhea_id"])

    rz_protein_indices = clean_pairs["Entry"].map(rz_to_row).to_numpy(np.int64)
    rz_reaction_indices = clean_pairs["rhea_id"].map(reaction_to_row).to_numpy(np.int64)
    tps_reaction_indices = np.asarray(
        [reaction_to_row[value] for value in sorted(tps["rhea_id"].unique())],
        dtype=np.int64,
    )
    tps_reaction_ids = [reaction_ids[index] for index in tps_reaction_indices]
    tps_reaction_matrix = reaction_matrix[tps_reaction_indices]
    tps_reaction_to_row = {
        value: index for index, value in enumerate(tps_reaction_ids)
    }

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
    tps_pairs = tps[["Entry", "rhea_id"]].merge(
        strict, on=["Entry", "rhea_id"], how="left", validate="one_to_one"
    )
    if tps_pairs[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict split misses TPS pairs")
    tps_pairs["protein_fold"] = tps_pairs["protein_fold"].astype(int)
    tps_pairs["reaction_fold"] = tps_pairs["reaction_fold"].astype(int)

    protein_groups_frame = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_groups = dict(
        zip(
            protein_groups_frame["entry"].astype(str),
            protein_groups_frame["cluster_id"].astype(str),
        )
    )
    protein_groups = {value: protein_groups.get(value, value) for value in tps_ids}
    reaction_groups_frame = pd.read_csv(args.reaction_clusters, dtype=str).fillna("")
    reaction_groups = dict(
        zip(
            reaction_groups_frame["reaction_id"].astype(str),
            reaction_groups_frame["reaction_cluster"].astype(str),
        )
    )
    reaction_groups = {
        value: reaction_groups.get(value, value) for value in tps_reaction_ids
    }
    return {
        "clean_pairs": clean_pairs,
        "rz_matrix": rz_matrix,
        "reaction_matrix": reaction_matrix,
        "rz_protein_indices": rz_protein_indices,
        "rz_reaction_indices": rz_reaction_indices,
        "tps_matrix": tps_matrix,
        "tps_ids": tps_ids,
        "tps_reaction_matrix": tps_reaction_matrix,
        "tps_reaction_ids": tps_reaction_ids,
        "tps_pairs": tps_pairs,
        "protein_groups": protein_groups,
        "reaction_groups": reaction_groups,
        "feature_schema": feature_schema,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Globally leakage-clean ReactZyme pretraining for TPS retrieval."
    )
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--reactzyme-embeddings", type=Path, default=DEFAULT_RZ_EMBEDDINGS)
    parser.add_argument("--tps-positives", type=Path, default=DEFAULT_TPS_POSITIVES)
    parser.add_argument("--tps-embeddings", type=Path, default=DEFAULT_TPS_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--reaction-clusters", type=Path, default=DEFAULT_REACTION_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--partition", choices=["development", "frozen", "all"], default="development")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=60)
    parser.add_argument("--batch-pairs", type=int, default=1024)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--finetune-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.75)
    parser.add_argument("--hard-negative-k", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = parse_int_tuple(args.budgets)
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = build_data(args)
    config = ModelConfig(
        protein_input_dim=int(data["rz_matrix"].shape[1]),
        reaction_input_dim=int(data["reaction_matrix"].shape[1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    state, pretrain_history = pretrain_model(
        protein_matrix=data["rz_matrix"],
        reaction_matrix=data["reaction_matrix"],
        protein_indices=data["rz_protein_indices"],
        reaction_indices=data["rz_reaction_indices"],
        config=config,
        epochs=args.pretrain_epochs,
        batch_pairs=args.batch_pairs,
        learning_rate=args.pretrain_learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        reaction_loss_weight=args.reaction_loss_weight,
        hard_negative_k=args.hard_negative_k,
        seed=args.seed,
        device=device,
    )
    torch.save(
        {"model_state_dict": state, "model_config": asdict(config)},
        output / "reactzyme_pretrained.pt",
    )
    pd.DataFrame(pretrain_history).to_csv(output / "pretraining_history.csv", index=False)

    tps_protein_tensor = torch.as_tensor(
        data["tps_matrix"], dtype=torch.float32, device=device
    )
    tps_reaction_tensor = torch.as_tensor(
        data["tps_reaction_matrix"], dtype=torch.float32, device=device
    )
    tps_protein_to_row = {
        value: index for index, value in enumerate(data["tps_ids"])
    }
    tps_reaction_to_row = {
        value: index for index, value in enumerate(data["tps_reaction_ids"])
    }
    all_positive_by_reaction = {
        reaction_id: set(group["Entry"].astype(str))
        for reaction_id, group in data["tps_pairs"].groupby("rhea_id", sort=True)
    }
    cells = [
        (p, r)
        for p in range(5)
        for r in range(5)
        if args.partition == "all"
        or (args.partition == "development" and (p == 4 or r == 4))
        or (args.partition == "frozen" and p != 4 and r != 4)
    ]
    records: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    for protein_fold, reaction_fold in cells:
        train_pairs = data["tps_pairs"][(data["tps_pairs"]["protein_fold"] != protein_fold) & (data["tps_pairs"]["reaction_fold"] != reaction_fold)][["Entry", "rhea_id"]].drop_duplicates()
        test_pairs = data["tps_pairs"][(data["tps_pairs"]["protein_fold"] == protein_fold) & (data["tps_pairs"]["reaction_fold"] == reaction_fold)].copy()
        if test_pairs.empty:
            continue
        for method, initial_state in [
            ("scratch", None),
            ("reactzyme_pretrained", state),
        ]:
            model, history = train_model(
                tps_protein_tensor,
                tps_reaction_tensor,
                train_pairs,
                tps_protein_to_row,
                tps_reaction_to_row,
                config,
                args.finetune_epochs,
                args.finetune_learning_rate,
                args.weight_decay,
                args.temperature,
                args.seed + protein_fold * 100 + reaction_fold,
                device,
                initial_state_dict=initial_state,
                protein_group_map=data["protein_groups"],
                reaction_group_map=data["reaction_groups"],
                exclude_same_group_negatives=True,
                reaction_loss_weight=args.reaction_loss_weight,
            )
            histories.append(
                {
                    "method": method,
                    "protein_fold": protein_fold,
                    "reaction_fold": reaction_fold,
                    "final_loss": history[-1]["loss"],
                    "best_loss": min(row["loss"] for row in history),
                }
            )
            model.eval()
            with torch.no_grad():
                proteins = model.encode_proteins(tps_protein_tensor).cpu().numpy()
                reactions = model.encode_reactions(tps_reaction_tensor).cpu().numpy()
            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                masked = all_positive_by_reaction.get(reaction_id, set()) - positives
                score = reactions[tps_reaction_to_row[reaction_id]] @ proteins.T
                records.append(
                    {
                        "method": method,
                        "protein_fold": protein_fold,
                        "reaction_fold": reaction_fold,
                        "reaction_id": reaction_id,
                        **masked_rank_metrics(
                            score, data["tps_ids"], positives, masked, budgets
                        ),
                    }
                )
    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "finetuning_summary.csv", index=False)
    metric_rows=[]
    for method, group in query_metrics.groupby("method", sort=True):
        row={
            "method":method,
            "n_queries":len(group),
            "mean_reciprocal_rank":group.reciprocal_rank.mean(),
            "median_best_positive_rank":group.best_positive_rank.median(),
        }
        for budget in budgets:
            row[f"hit_probability_at_{budget}"]=group[f"hit_at_{budget}"].mean()
        metric_rows.append(row)
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary={
        "clean_dir":str(args.clean_dir.resolve()),
        "partition":args.partition,
        "pretrain_pairs":int(len(data["clean_pairs"])),
        "pretrain_proteins":int(data["clean_pairs"]["Entry"].nunique()),
        "pretrain_reactions":int(data["clean_pairs"]["rhea_id"].nunique()),
        "pretrain_epochs":args.pretrain_epochs,
        "finetune_epochs":args.finetune_epochs,
        "batch_pairs":args.batch_pairs,
        "model_config":asdict(config),
        "feature_schema":data["feature_schema"],
    }
    (output / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
