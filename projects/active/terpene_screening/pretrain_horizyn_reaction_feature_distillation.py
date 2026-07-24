from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_reaction,
    load_feature_schema,
)
from projects.active.terpene_screening.train_dual_tower_cold import seed_everything  # noqa: E402
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (  # noqa: E402
    build_horizyn_fingerprints,
    encode_horizyn_reactions,
    normalize_rows,
)

DEFAULT_TRAIN = HORIZYN_ROOT / "data/sota/train_rxns.csv"
DEFAULT_VALIDATION = HORIZYN_ROOT / "data/sota/test_rxns.csv"
DEFAULT_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_dev.ckpt"
DEFAULT_HORIZYN_CONFIG = HORIZYN_ROOT / "configs/sota.yaml"
DEFAULT_PRODUCTION_DIR = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_MARTS_OVERLAP = ROOT / "results/terpene_horizyn_reaction_overlap.csv"
DEFAULT_CACHE = ROOT / "data/terpene_horizyn_reaction_distillation"
DEFAULT_STANDARDIZED_TRAIN = ROOT / "data/terpene_horizyn_adapter_v2/train_standardized_reactions.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_horizyn_reaction_feature_distillation"


@dataclass(frozen=True)
class ReactionDistillerConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int = 512
    dropout: float = 0.1
    residual_blocks: int = 2


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class ReactionFeatureDistiller(nn.Module):
    def __init__(self, config: ReactionDistillerConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        layers.extend(ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.residual_blocks))
        layers.extend([nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.output_dim)])
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


def load_overlap_signatures(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "standardized_reaction" not in frame.columns:
        return set()
    return set(frame.loc[frame["standardized_reaction"].ne(""), "standardized_reaction"].astype(str))


def prepare_split(
    frame: pd.DataFrame,
    split_name: str,
    schema: dict[str, object],
    fingerprint_cache: Path,
    checkpoint: Path,
    horizyn_config: Path,
    device: torch.device,
    batch_size: int,
    append_rdkit_plus: bool,
    rdkit_plus_dim: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    working = frame[["reaction_id", "reaction_smiles"]].drop_duplicates("reaction_id").copy()
    fingerprint_dir = fingerprint_cache / split_name
    fingerprint_dir.mkdir(parents=True, exist_ok=True)

    tps_feature_cache = fingerprint_dir / "tps_reaction_features.npy"
    tps_audit_cache = fingerprint_dir / "tps_reaction_feature_audit.csv"
    if tps_feature_cache.exists() and tps_audit_cache.exists():
        base_features = np.load(tps_feature_cache).astype(np.float32)
        tps_audit = pd.read_csv(tps_audit_cache, dtype=str).fillna("")
        if len(base_features) != len(working) or len(tps_audit) != len(working):
            raise ValueError(f"Cached TPS features for {split_name} differ in row count")
        if tps_audit["reaction_id"].astype(str).tolist() != working["reaction_id"].astype(str).tolist():
            raise ValueError(f"Cached TPS feature IDs for {split_name} differ in order")
        feature_success = tps_audit["tps_feature_success"].astype(str).str.lower().eq("true").tolist()
        feature_error = tps_audit["tps_feature_error"].astype(str).tolist()
    else:
        feature_rows: list[np.ndarray] = []
        feature_success: list[bool] = []
        feature_error: list[str] = []
        for reaction in working["reaction_smiles"].astype(str):
            try:
                feature_rows.append(encode_reaction(reaction, schema).astype(np.float32))
                feature_success.append(True)
                feature_error.append("")
            except Exception as exc:
                feature_rows.append(
                    np.zeros(int(schema["reaction_feature_dimension"]), dtype=np.float32)
                )
                feature_success.append(False)
                feature_error.append(repr(exc))
        base_features = np.stack(feature_rows).astype(np.float32)
        np.save(tps_feature_cache, base_features)
        pd.DataFrame(
            {
                "reaction_id": working["reaction_id"].astype(str),
                "tps_feature_success": feature_success,
                "tps_feature_error": feature_error,
            }
        ).to_csv(tps_audit_cache, index=False)
    working["tps_feature_success"] = feature_success
    working["tps_feature_error"] = feature_error

    horizyn_fingerprints, fingerprint_audit = build_horizyn_fingerprints(
        working,
        horizyn_config,
        fingerprint_dir,
    )
    teacher_cache = fingerprint_dir / "teacher_reaction_embeddings.npy"
    if teacher_cache.exists():
        teacher = np.load(teacher_cache).astype(np.float32)
        if len(teacher) != len(working):
            raise ValueError(f"Cached teacher embeddings for {split_name} differ in row count")
    else:
        teacher = encode_horizyn_reactions(
            horizyn_fingerprints,
            checkpoint,
            device,
            batch_size,
        )
        np.save(teacher_cache, teacher.astype(np.float32))
    fingerprint_audit = fingerprint_audit.rename(
        columns={"success": "horizyn_fingerprint_success", "error": "horizyn_fingerprint_error"}
    )
    audit = working.merge(
        fingerprint_audit[["reaction_id", "horizyn_fingerprint_success", "horizyn_fingerprint_error"]],
        on="reaction_id",
        how="left",
        validate="one_to_one",
    )
    valid = audit["tps_feature_success"].astype(bool) & audit["horizyn_fingerprint_success"].astype(str).str.lower().eq("true")
    audit["used_for_distillation"] = valid
    features = base_features
    if append_rdkit_plus:
        if rdkit_plus_dim <= 0 or rdkit_plus_dim > horizyn_fingerprints.shape[1]:
            raise ValueError("Invalid RDKit+ dimension for enriched distillation")
        features = np.concatenate(
            [features, horizyn_fingerprints[:, :rdkit_plus_dim].astype(np.float32)],
            axis=1,
        )
    features = features[valid.to_numpy()]
    teacher = normalize_rows(teacher[valid.to_numpy()])
    return features, teacher, audit

def batch_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    relational_weight: float,
    variance_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pointwise = (1.0 - (student * teacher).sum(dim=1)).mean()
    if len(student) > 1:
        relational = F.smooth_l1_loss(student @ student.T, teacher @ teacher.T)
        student_std = torch.sqrt(student.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(0.02 - student_std).mean()
    else:
        relational = torch.zeros((), device=student.device, dtype=student.dtype)
        variance = torch.zeros((), device=student.device, dtype=student.dtype)
    total = pointwise + relational_weight * relational + variance_weight * variance
    return total, pointwise, relational, variance


def evaluate(
    model: ReactionFeatureDistiller,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    cosine_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            student = model(features[start : start + batch_size])
            cosine_rows.append((student * targets[start : start + batch_size]).sum(dim=1).cpu().numpy())
    values = np.concatenate(cosine_rows)
    return {
        "n": int(len(values)),
        "mean_cosine": float(values.mean()),
        "median_cosine": float(np.median(values)),
        "p10_cosine": float(np.quantile(values, 0.1)),
        "p01_cosine": float(np.quantile(values, 0.01)),
        "min_cosine": float(values.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Horizyn reaction embeddings into the existing TPS reaction feature representation.")
    parser.add_argument("--train-reactions", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-reactions", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--horizyn-config", type=Path, default=DEFAULT_HORIZYN_CONFIG)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--marts-overlap", type=Path, default=DEFAULT_MARTS_OVERLAP)
    parser.add_argument(
        "--standardized-train-cache",
        type=Path,
        default=DEFAULT_STANDARDIZED_TRAIN,
        help="Validated standardized Horizyn training reactions used for exact MARTS overlap exclusion.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument(
        "--append-rdkit-plus",
        action="store_true",
        help="Concatenate the official RDKit+ structural fingerprint to TPS features.",
    )
    parser.add_argument("--rdkit-plus-dim", type=int, default=1024)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--relational-weight", type=float, default=0.25)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema(args.production_dir.resolve())

    train_frame = pd.read_csv(args.train_reactions.resolve(), dtype=str).fillna("")
    validation_frame = pd.read_csv(args.validation_reactions.resolve(), dtype=str).fillna("")
    train_features, train_teacher, train_audit = prepare_split(
        train_frame,
        "train",
        schema,
        cache_dir,
        args.checkpoint.resolve(),
        args.horizyn_config.resolve(),
        device,
        args.batch_size,
        args.append_rdkit_plus,
        args.rdkit_plus_dim,
    )
    validation_features, validation_teacher, validation_audit = prepare_split(
        validation_frame,
        "validation",
        schema,
        cache_dir,
        args.checkpoint.resolve(),
        args.horizyn_config.resolve(),
        device,
        args.batch_size,
        args.append_rdkit_plus,
        args.rdkit_plus_dim,
    )

    # Exclude exact standardized MARTS overlap from teacher training.
    overlap_signatures = load_overlap_signatures(args.marts_overlap.resolve())
    standardized_cache = args.standardized_train_cache.resolve()
    if not standardized_cache.exists():
        raise FileNotFoundError(
            f"Standardized Horizyn training cache is required for overlap exclusion: {standardized_cache}"
        )
    if not overlap_signatures:
        raise ValueError("No standardized MARTS overlap signatures were loaded")
    standardized = pd.read_csv(standardized_cache, dtype=str).fillna("")
    required_standardized = {"reaction_id", "standardized_reaction"}
    if not required_standardized.issubset(standardized.columns):
        raise ValueError(
            f"Standardized cache lacks columns: {sorted(required_standardized - set(standardized.columns))}"
        )
    usable_ids = train_audit.loc[
        train_audit["used_for_distillation"].astype(bool), "reaction_id"
    ].astype(str).tolist()
    standard_by_id = dict(
        zip(
            standardized["reaction_id"].astype(str),
            standardized["standardized_reaction"].astype(str),
        )
    )
    missing_standardized = [value for value in usable_ids if value not in standard_by_id]
    if missing_standardized:
        raise ValueError(
            f"Standardized cache is missing usable training reactions: {missing_standardized[:10]}"
        )
    keep = np.asarray(
        [standard_by_id[value] not in overlap_signatures for value in usable_ids],
        dtype=bool,
    )
    excluded_exact_overlap = int((~keep).sum())
    train_features = train_features[keep]
    train_teacher = train_teacher[keep]

    config = ReactionDistillerConfig(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=train_teacher.shape[1],
        dropout=args.dropout,
        residual_blocks=args.residual_blocks,
    )
    model = ReactionFeatureDistiller(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(train_teacher, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(validation_teacher, dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        pointwise_losses: list[float] = []
        relational_losses: list[float] = []
        variance_losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            rows = torch.as_tensor(order[start : start + args.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            student = model(train_x[rows])
            loss, pointwise, relational, variance = batch_distillation_loss(
                student,
                train_y[rows],
                args.relational_weight,
                args.variance_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pointwise_losses.append(float(pointwise.detach().cpu()))
            relational_losses.append(float(relational.detach().cpu()))
            variance_losses.append(float(variance.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_x, train_y, args.batch_size)
            validation_metrics = evaluate(model, validation_x, validation_y, args.batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "pointwise_loss": float(np.mean(pointwise_losses)),
                    "relational_loss": float(np.mean(relational_losses)),
                    "variance_loss": float(np.mean(variance_losses)),
                    "train_mean_cosine": train_metrics["mean_cosine"],
                    "validation_mean_cosine": validation_metrics["mean_cosine"],
                    "validation_p10_cosine": validation_metrics["p10_cosine"],
                }
            )
            if validation_metrics["mean_cosine"] > best_validation:
                best_validation = validation_metrics["mean_cosine"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("No reaction distillation checkpoint selected")
    model.load_state_dict(best_state)
    final_train = evaluate(model, train_x, train_y, args.batch_size)
    final_validation = evaluate(model, validation_x, validation_y, args.batch_size)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": asdict(config),
            "append_rdkit_plus": args.append_rdkit_plus,
            "rdkit_plus_dim": args.rdkit_plus_dim if args.append_rdkit_plus else 0,
            "feature_schema": schema,
            "teacher_checkpoint": str(args.checkpoint.resolve()),
            "train_metrics": final_train,
            "validation_metrics": final_validation,
            "excluded_exact_marts_overlap": excluded_exact_overlap,
        },
        output_dir / "reaction_feature_distiller.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    train_audit.to_csv(output_dir / "train_reaction_audit.csv", index=False)
    validation_audit.to_csv(output_dir / "validation_reaction_audit.csv", index=False)
    summary = {
        "model_config": asdict(config),
        "append_rdkit_plus": args.append_rdkit_plus,
        "rdkit_plus_dim": args.rdkit_plus_dim if args.append_rdkit_plus else 0,
        "n_train_usable": int(len(train_features)),
        "n_validation_usable": int(len(validation_features)),
        "excluded_exact_marts_overlap": excluded_exact_overlap,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "best_validation_mean_cosine": best_validation,
        "epochs": args.epochs,
        "outputs": {
            "checkpoint": str(output_dir / "reaction_feature_distiller.pt"),
            "history": str(output_dir / "training_history.csv"),
            "train_audit": str(output_dir / "train_reaction_audit.csv"),
            "validation_audit": str(output_dir / "validation_reaction_audit.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
