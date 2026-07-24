from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from horizyn.lightning_module import HorizynLitModule  # noqa: E402
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (  # noqa: E402
    AdapterConfig,
    ProteinAdapter,
    normalize_rows,
)
from projects.active.terpene_screening.train_dual_tower_cold import seed_everything  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_OVERLAP = ROOT / "results/terpene_horizyn_protein_overlap.csv"
DEFAULT_H5 = HORIZYN_ROOT / "data/sota/prots_t5.h5"
DEFAULT_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_dev.ckpt"
DEFAULT_OUTPUT = ROOT / "results/terpene_horizyn_protein_space_distillation"


def parse_first_hit(value: str) -> str:
    return next((token for token in str(value).split("|") if token), "")


def cluster_validation_split(
    table: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_sizes = table.groupby("cluster_id").size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    cluster_ids = cluster_sizes.index.astype(str).tolist()
    rng.shuffle(cluster_ids)
    target = max(1, int(round(len(table) * validation_fraction)))
    validation_clusters: set[str] = set()
    count = 0
    for cluster_id in cluster_ids:
        validation_clusters.add(cluster_id)
        count += int(cluster_sizes.loc[cluster_id])
        if count >= target:
            break
    validation = table["cluster_id"].astype(str).isin(validation_clusters).to_numpy()
    train = ~validation
    if not train.any() or not validation.any():
        raise ValueError("Cluster validation split is empty")
    return np.flatnonzero(train), np.flatnonzero(validation)


def batch_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    relational_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pointwise = (1.0 - (student * teacher).sum(dim=1)).mean()
    if len(student) > 1 and relational_weight > 0:
        student_rel = student @ student.T
        teacher_rel = teacher @ teacher.T
        relational = F.smooth_l1_loss(student_rel, teacher_rel)
    else:
        relational = torch.zeros((), dtype=student.dtype, device=student.device)
    return pointwise + relational_weight * relational, pointwise, relational


def evaluate(
    model: ProteinAdapter,
    features: torch.Tensor,
    targets: torch.Tensor,
    rows: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    cosine_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = torch.as_tensor(rows[start : start + batch_size], dtype=torch.long, device=features.device)
            student = model(features[batch_rows])
            cosine_values.append((student * targets[batch_rows]).sum(dim=1).cpu().numpy())
    values = np.concatenate(cosine_values)
    return {
        "mean_cosine": float(values.mean()),
        "median_cosine": float(np.median(values)),
        "p10_cosine": float(np.quantile(values, 0.1)),
        "min_cosine": float(values.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Horizyn's protein target space into an ESM-C adapter using current-library proteins only.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--relational-weight", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache_dir = args.cache_dir.resolve()
    features = normalize_rows(np.load(cache_dir / "protein_features.npy"))
    proteins = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    overlap = pd.read_csv(args.overlap, dtype=str).fillna("")
    overlap["teacher_id"] = overlap["all_hits"].map(parse_first_hit)
    table = proteins.merge(
        overlap[["protein_id", "teacher_id"]],
        on="protein_id",
        how="left",
        validate="one_to_one",
    )
    current = table[
        table["enzyme_seen"].astype(str).str.lower().isin({"true", "1"})
        & table["teacher_id"].astype(str).ne("")
    ].copy()
    if current.empty:
        raise ValueError("No current proteins overlap Horizyn")

    with h5py.File(args.h5.resolve(), "r") as h5:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in h5["ids"][:]]
        id_to_row = {value: index for index, value in enumerate(ids)}
        missing = sorted(set(current["teacher_id"]) - set(id_to_row))
        if missing:
            raise ValueError(f"Teacher H5 missing IDs: {missing[:10]}")
        teacher_inputs = np.stack(
            [h5["vectors"][id_to_row[value]] for value in current["teacher_id"]],
        ).astype(np.float32)

    horizyn = HorizynLitModule.load_from_checkpoint(
        str(args.checkpoint.resolve()),
        map_location=device,
    ).to(device).eval()
    with torch.no_grad():
        teacher_targets = []
        teacher_input_tensor = torch.as_tensor(teacher_inputs, dtype=torch.float32, device=device)
        for start in range(0, len(teacher_input_tensor), args.batch_size):
            teacher_targets.append(
                horizyn.model.target_encoder(teacher_input_tensor[start : start + args.batch_size]).cpu().numpy()
            )
    teacher_targets_array = normalize_rows(np.concatenate(teacher_targets, axis=0))

    protein_to_row = {value: index for index, value in enumerate(proteins["protein_id"].astype(str))}
    current["feature_row"] = current["protein_id"].map(protein_to_row).astype(int)
    current = current.reset_index(drop=True)
    student_features = features[current["feature_row"].to_numpy()]
    train_rows, validation_rows = cluster_validation_split(current, args.validation_fraction, args.seed)

    feature_tensor = torch.as_tensor(student_features, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(teacher_targets_array, dtype=torch.float32, device=device)
    config = AdapterConfig(
        input_dim=student_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=teacher_targets_array.shape[1],
        dropout=args.dropout,
        linear_only=False,
    )
    model = ProteinAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_rows)
        epoch_losses: list[float] = []
        pointwise_losses: list[float] = []
        relational_losses: list[float] = []
        for start in range(0, len(shuffled), args.batch_size):
            rows = torch.as_tensor(shuffled[start : start + args.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            student = model(feature_tensor[rows])
            loss, pointwise, relational = batch_loss(student, target_tensor[rows], args.relational_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            pointwise_losses.append(float(pointwise.detach().cpu()))
            relational_losses.append(float(relational.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, feature_tensor, target_tensor, train_rows, args.batch_size)
            validation_metrics = evaluate(model, feature_tensor, target_tensor, validation_rows, args.batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "pointwise_loss": float(np.mean(pointwise_losses)),
                    "relational_loss": float(np.mean(relational_losses)),
                    "train_mean_cosine": train_metrics["mean_cosine"],
                    "validation_mean_cosine": validation_metrics["mean_cosine"],
                    "validation_median_cosine": validation_metrics["median_cosine"],
                }
            )
            if validation_metrics["mean_cosine"] > best_validation:
                best_validation = validation_metrics["mean_cosine"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No distillation checkpoint selected")
    model.load_state_dict(best_state)
    final_train = evaluate(model, feature_tensor, target_tensor, train_rows, args.batch_size)
    final_validation = evaluate(model, feature_tensor, target_tensor, validation_rows, args.batch_size)

    current["distillation_split"] = "train"
    current.loc[validation_rows, "distillation_split"] = "validation"
    current.to_csv(output_dir / "matched_current_proteins.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save(
        {
            "adapter_state_dict": best_state,
            "adapter_config": asdict(config),
            "teacher_checkpoint": str(args.checkpoint.resolve()),
            "seed": args.seed,
            "n_current_teacher_pairs": len(current),
            "n_train": len(train_rows),
            "n_validation": len(validation_rows),
            "relational_weight": args.relational_weight,
            "train_metrics": final_train,
            "validation_metrics": final_validation,
        },
        output_dir / "distilled_adapter.pt",
    )
    summary = {
        "adapter_config": asdict(config),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "n_current_proteins": int(proteins["enzyme_seen"].astype(str).str.lower().isin({"true", "1"}).sum()),
        "n_current_teacher_pairs": len(current),
        "n_train": len(train_rows),
        "n_validation": len(validation_rows),
        "n_train_clusters": current.iloc[train_rows]["cluster_id"].nunique(),
        "n_validation_clusters": current.iloc[validation_rows]["cluster_id"].nunique(),
        "best_validation_mean_cosine": best_validation,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "epochs": args.epochs,
        "relational_weight": args.relational_weight,
        "outputs": {
            "checkpoint": str(output_dir / "distilled_adapter.pt"),
            "history": str(output_dir / "training_history.csv"),
            "matched_proteins": str(output_dir / "matched_current_proteins.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
