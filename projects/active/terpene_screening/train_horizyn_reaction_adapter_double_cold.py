from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder
from rdkit import Chem
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from horizyn.config import load_config  # noqa: E402
from horizyn.datasets.base import BaseDataset  # noqa: E402
from horizyn.datasets.collection import MergeDataset  # noqa: E402
from horizyn.datasets.fingerprints import (  # noqa: E402
    DRFPFingerprintDataset,
    RDKitPlusFingerprintDataset,
)
from horizyn.datasets.transform import ConcatTensorTransform  # noqa: E402
from horizyn.lightning_module import HorizynLitModule  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_training_mask,
    multi_positive_contrastive_loss,
    rank_metrics,
    seed_everything,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_dev.ckpt"
DEFAULT_CONFIG = HORIZYN_ROOT / "configs/sota.yaml"
DEFAULT_OUTPUT = ROOT / "results/terpene_horizyn_reaction_adapter_double_cold"
DEFAULT_FINGERPRINT_CACHE = ROOT / "data/terpene_horizyn_adapter"
DEFAULT_BUDGETS = (3, 10, 20)


@dataclass(frozen=True)
class AdapterConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int = 512
    dropout: float = 0.1
    linear_only: bool = False


class ProteinAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        if config.linear_only:
            self.network = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.output_dim),
            )
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(config.input_dim),
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(features), dim=-1)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def remove_reaction_stereochemistry(reaction_smiles: str) -> str:
    if ">>" not in reaction_smiles:
        return reaction_smiles
    left, right = reaction_smiles.split(">>", 1)

    def convert_side(side: str) -> str:
        converted: list[str] = []
        for part in side.split("."):
            molecule = Chem.MolFromSmiles(part)
            converted.append(
                Chem.MolToSmiles(molecule, isomericSmiles=False)
                if molecule is not None
                else part
            )
        return ".".join(converted)

    return f"{convert_side(left)}>>{convert_side(right)}"


def build_horizyn_fingerprints(
    reaction_table: pd.DataFrame,
    config_path: Path,
    cache_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "reaction_fingerprints.npy"
    audit_path = cache_dir / "reaction_fingerprint_audit.csv"
    if matrix_path.exists() and audit_path.exists():
        matrix = np.load(matrix_path).astype(np.float32)
        audit = pd.read_csv(audit_path, dtype=str).fillna("")
        if len(matrix) == len(reaction_table) == len(audit):
            return matrix, audit

    config = load_config(str(config_path))
    keys = reaction_table["reaction_id"].astype(str).tolist()
    data = [
        {"reaction_smiles": str(value)}
        for value in reaction_table["reaction_smiles"].astype(str)
    ]
    reactions = BaseDataset(keys=keys, array_data=data)
    common = {
        "reaction_dataset": reactions,
        "standardize": config.data.get("standardize_reactions", True),
        "standardize_hypervalent": config.data.get("standardize_hypervalent", True),
        "standardize_remove_hs": config.data.get("standardize_remove_hs", True),
        "standardize_kekulize": config.data.get("standardize_kekulize", False),
        "standardize_uncharge": config.data.get("standardize_uncharge", True),
        "standardize_metals": config.data.get("standardize_metals", True),
    }
    rdkit = RDKitPlusFingerprintDataset(
        vec_dim=config.data.get("rdkit_fp_dim", 1024),
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        **common,
    )
    drfp = DRFPFingerprintDataset(
        vec_dim=config.data.get("drfp_dim", 1024),
        radius=3,
        rings=True,
        **common,
    )
    merged = MergeDataset(datasets={"rdkit": rdkit, "drfp": drfp}, add_prefix=False)
    merged.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))

    rows: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    raw_by_key = dict(zip(keys, reaction_table["reaction_smiles"].astype(str)))
    for key in keys:
        try:
            fingerprint = merged[key].detach().cpu().numpy().astype(np.float32)
            rows.append(fingerprint)
            audit_rows.append(
                {
                    "reaction_id": key,
                    "success": True,
                    "fingerprint_mode": "official_standardized",
                    "error": "",
                }
            )
        except Exception as primary_error:
            try:
                rdkit_vector = rdkit[key].detach().cpu().numpy().astype(np.float32)
                fallback_smiles = remove_reaction_stereochemistry(raw_by_key[key])
                drfp_vector = DrfpEncoder.encode(
                    [fallback_smiles],
                    n_folded_length=config.data.get("drfp_dim", 1024),
                    min_radius=0,
                    radius=3,
                    rings=True,
                    mapping=False,
                    atom_index_mapping=False,
                    root_central_atom=True,
                    include_hydrogens=False,
                    show_progress_bar=False,
                )[0].astype(np.float32)
                rows.append(np.concatenate([rdkit_vector, drfp_vector]))
                audit_rows.append(
                    {
                        "reaction_id": key,
                        "success": True,
                        "fingerprint_mode": "stereo_stripped_drfp_fallback",
                        "error": str(primary_error)[:1000],
                    }
                )
            except Exception as fallback_error:
                rows.append(np.zeros(2048, dtype=np.float32))
                audit_rows.append(
                    {
                        "reaction_id": key,
                        "success": False,
                        "fingerprint_mode": "failed",
                        "error": f"primary={primary_error}; fallback={fallback_error}"[:1000],
                    }
                )
    matrix = np.stack(rows)
    audit = pd.DataFrame(audit_rows)
    np.save(matrix_path, matrix)
    audit.to_csv(audit_path, index=False)
    return matrix, audit


def encode_horizyn_reactions(
    fingerprints: np.ndarray,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    module = HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    encoder = module.model.query_encoder.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fingerprints), batch_size):
            tensor = torch.as_tensor(
                fingerprints[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(encoder(tensor).cpu().numpy())
    return normalize_rows(np.concatenate(outputs, axis=0))


def build_denominator_masks(
    positive_mask: np.ndarray,
    local_reaction_ids: list[str],
    local_protein_ids: list[str],
    reaction_clusters: dict[str, str],
    protein_clusters: dict[str, str],
    pu_group_mask: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reaction_denominator = np.ones_like(positive_mask, dtype=bool)
    protein_denominator = np.ones_like(positive_mask, dtype=bool)
    if not pu_group_mask:
        return reaction_denominator, protein_denominator
    local_protein_groups = np.asarray(
        [protein_clusters.get(value, value) for value in local_protein_ids], dtype=object
    )
    local_reaction_groups = np.asarray(
        [reaction_clusters.get(value, value) for value in local_reaction_ids], dtype=object
    )
    for reaction_index in range(positive_mask.shape[0]):
        groups = set(local_protein_groups[positive_mask[reaction_index]])
        potential = np.isin(local_protein_groups, list(groups)) & ~positive_mask[reaction_index]
        reaction_denominator[reaction_index, potential] = False
    for protein_index in range(positive_mask.shape[1]):
        groups = set(local_reaction_groups[positive_mask[:, protein_index]])
        potential = np.isin(local_reaction_groups, list(groups)) & ~positive_mask[:, protein_index]
        protein_denominator[potential, protein_index] = False
    return reaction_denominator, protein_denominator


def train_adapter(
    protein_features: torch.Tensor,
    reaction_embeddings: torch.Tensor,
    train_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_clusters: dict[str, str],
    reaction_clusters: dict[str, str],
    config: AdapterConfig,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    hard_negative_k: int,
    pu_group_mask: bool,
    seed: int,
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[ProteinAdapter, list[dict[str, object]]]:
    seed_everything(seed)
    adapter = ProteinAdapter(config).to(device)
    if initial_state_dict is not None:
        adapter.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    reaction_rows, protein_rows, positive_mask = build_training_mask(
        train_pairs, reaction_to_row, protein_to_row
    )
    local_reaction_ids = [reaction_ids[int(row)] for row in reaction_rows]
    local_protein_ids = [protein_ids[int(row)] for row in protein_rows]
    reaction_denominator, protein_denominator = build_denominator_masks(
        positive_mask,
        local_reaction_ids,
        local_protein_ids,
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
    fixed_reactions = reaction_embeddings[rr]
    for epoch in range(1, epochs + 1):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        proteins = adapter(protein_features[pr])
        loss, reaction_loss, protein_loss = multi_positive_contrastive_loss(
            fixed_reactions,
            proteins,
            positives,
            temperature,
            rden,
            pden,
            reaction_loss_weight,
            "bidirectional_infonce",
            hard_negative_k,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in adapter.state_dict().items()
            }
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": value,
                    "reaction_loss": float(reaction_loss.detach().cpu()),
                    "protein_loss": float(protein_loss.detach().cpu()),
                    "reaction_loss_weight": reaction_loss_weight,
                    "hard_negative_k": hard_negative_k,
                    "pu_group_mask": pu_group_mask,
                }
            )
    if best_state is None:
        raise RuntimeError("No adapter state was trained")
    adapter.load_state_dict(best_state)
    return adapter, history


def evaluate_scores(
    records: list[dict[str, object]],
    split_id: str,
    score_matrix: np.ndarray,
    test_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    budgets: tuple[int, ...],
) -> None:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        positives = set(group["Entry"].astype(str))
        metrics = rank_metrics(
            score_matrix[reaction_to_row[str(reaction_id)]],
            protein_ids,
            positives,
            set(),
            budgets,
        )
        records.append(
            {
                "split_id": split_id,
                "direction": "reaction_to_enzyme",
                "query_id": reaction_id,
                **metrics,
            }
        )
    for protein_id, group in test_pairs.groupby("Entry", sort=True):
        positives = set(group["rhea_id"].astype(str))
        metrics = rank_metrics(
            score_matrix[:, protein_to_row[str(protein_id)]],
            reaction_ids,
            positives,
            set(),
            budgets,
        )
        records.append(
            {
                "split_id": split_id,
                "direction": "enzyme_to_reaction",
                "query_id": protein_id,
                **metrics,
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
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby("direction").agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the Horizyn reaction encoder and adapt ESM-C proteins under strict double-cold splits."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FINGERPRINT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-mode", choices=["paired", "cartesian"], default="cartesian")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.75)
    parser.add_argument("--hard-negative-k", type=int, default=128)
    parser.add_argument("--pu-group-mask", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        default=None,
        help="Optional distilled ProteinAdapter checkpoint used to initialize every outer split.",
    )
    parser.add_argument("--ensemble-seeds", default="20260723,20260724,20260725")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    ensemble_seeds = tuple(int(value) for value in args.ensemble_seeds.split(",") if value)

    cache_dir = args.cache_dir.resolve()
    proteins = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reactions = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = boolean_series(pairs["protein_seen"])
    pairs["reaction_seen"] = boolean_series(pairs["reaction_seen"])
    protein_ids = proteins["protein_id"].astype(str).tolist()
    reaction_ids = reactions["reaction_id"].astype(str).tolist()
    protein_clusters = dict(zip(proteins["protein_id"], proteins["cluster_id"]))
    reaction_clusters = dict(zip(reactions["reaction_id"], reactions["cluster_id"]))
    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy"))

    fingerprints, fingerprint_audit = build_horizyn_fingerprints(
        reactions,
        args.config.resolve(),
        args.fingerprint_cache.resolve(),
    )
    reaction_embeddings = encode_horizyn_reactions(
        fingerprints,
        args.checkpoint.resolve(),
        device,
        args.batch_size,
    )
    np.save(output_dir / "horizyn_reaction_embeddings.npy", reaction_embeddings)
    fingerprint_audit.to_csv(output_dir / "reaction_fingerprint_audit.csv", index=False)

    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_embeddings, dtype=torch.float32, device=device)
    adapter_config = AdapterConfig(
        input_dim=protein_features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        linear_only=args.linear_only,
    )
    initial_adapter_state: dict[str, torch.Tensor] | None = None
    initial_adapter_path: str | None = None
    if args.initial_adapter is not None:
        initial_payload = torch.load(
            args.initial_adapter.resolve(), map_location="cpu", weights_only=False
        )
        initial_config = AdapterConfig(**initial_payload["adapter_config"])
        if initial_config != adapter_config:
            raise ValueError(
                f"Initial adapter config {initial_config} does not match requested {adapter_config}"
            )
        initial_adapter_state = initial_payload["adapter_state_dict"]
        initial_adapter_path = str(args.initial_adapter.resolve())
    split_specs = (
        [(fold, fold) for fold in range(args.n_folds)]
        if args.fold_mode == "paired"
        else [
            (protein_fold, reaction_fold)
            for protein_fold in range(args.n_folds)
            for reaction_fold in range(args.n_folds)
        ]
    )

    records: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    for split_index, (protein_fold, reaction_fold) in enumerate(split_specs):
        split_id = f"p{protein_fold}_r{reaction_fold}"
        train_pairs = pairs[
            (pairs["protein_fold"] != protein_fold)
            & (pairs["reaction_fold"] != reaction_fold)
        ].drop_duplicates(["rhea_id", "Entry"])
        test_pairs = pairs[
            (pairs["protein_fold"] == protein_fold)
            & (pairs["reaction_fold"] == reaction_fold)
            & (~pairs["protein_seen"])
            & (~pairs["reaction_seen"])
        ].drop_duplicates(["rhea_id", "Entry"])
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
        adapters: list[ProteinAdapter] = []
        for seed_index, base_seed in enumerate(ensemble_seeds):
            adapter, history = train_adapter(
                protein_tensor,
                reaction_tensor,
                train_pairs,
                protein_ids,
                reaction_ids,
                protein_clusters,
                reaction_clusters,
                adapter_config,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.temperature,
                args.reaction_loss_weight,
                args.hard_negative_k,
                args.pu_group_mask,
                base_seed + split_index * 1000,
                device,
                initial_state_dict=initial_adapter_state,
            )
            adapters.append(adapter)
            frame = pd.DataFrame(history)
            frame.insert(0, "seed_index", seed_index)
            frame.insert(0, "split_id", split_id)
            histories.append(frame)
            torch.save(
                {
                    "adapter_state_dict": adapter.state_dict(),
                    "adapter_config": asdict(adapter_config),
                    "split_id": split_id,
                    "seed": base_seed,
                    "horizyn_checkpoint": str(args.checkpoint.resolve()),
                    "initial_adapter": initial_adapter_path,
                },
                model_dir / f"adapter_{split_id}_seed{seed_index}.pt",
            )
        with torch.no_grad():
            score_matrix = np.zeros((len(reaction_ids), len(protein_ids)), dtype=np.float32)
            for adapter in adapters:
                adapter.eval()
                protein_embeddings = adapter(protein_tensor).cpu().numpy()
                score_matrix += reaction_embeddings @ protein_embeddings.T
            score_matrix /= len(adapters)
        evaluate_scores(
            records,
            split_id,
            score_matrix,
            test_pairs,
            protein_ids,
            reaction_ids,
            budgets,
        )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            output_dir / "training_history.csv", index=False
        )
    summary = {
        "horizyn_checkpoint": str(args.checkpoint.resolve()),
        "horizyn_config": str(args.config.resolve()),
        "adapter_config": asdict(adapter_config),
        "initial_adapter": initial_adapter_path,
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "n_pairs": len(pairs),
        "fingerprint_success": int(boolean_series(fingerprint_audit["success"]).sum()),
        "fingerprint_failures": int((~boolean_series(fingerprint_audit["success"])).sum()),
        "fold_mode": args.fold_mode,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "pu_group_mask": args.pu_group_mask,
        "ensemble_seeds": ensemble_seeds,
        "budgets": budgets,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "training_history": str(output_dir / "training_history.csv"),
            "models": str(model_dir),
            "reaction_embeddings": str(output_dir / "horizyn_reaction_embeddings.npy"),
            "fingerprint_audit": str(output_dir / "reaction_fingerprint_audit.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
