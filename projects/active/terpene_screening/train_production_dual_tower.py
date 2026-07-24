from __future__ import annotations

import argparse
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

from projects.active.terpene_screening.train_dual_tower_cold import (
    ModelConfig,
    build_reaction_features,
    load_aligned_feature_augmentation,
    load_protein_features,
    train_model,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_PROTEIN_DIR = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_OUTPUT = ROOT / "results/terpene_production_models"
DEFAULT_MODES = ("drfp_categorical", "multiview")
DEFAULT_SEEDS = (20260723, 20260724, 20260725)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("At least one seed is required.")
    return result


def parse_str_tuple(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(result) - set(DEFAULT_MODES)
    if not result or unknown:
        raise ValueError(f"Unsupported feature modes: {sorted(unknown)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all-data production dual towers for open-world TPS retrieval.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument("--reaction-augmentation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    modes = parse_str_tuple(args.feature_modes)
    seeds = parse_int_tuple(args.seeds)
    device = torch.device(args.device)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    protein_matrix, protein_ids = load_protein_features(args.protein_dir.resolve())
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry", "smiles_seq"]].drop_duplicates(["rhea_id", "Entry"])
    positives = positives[positives["Entry"].isin(protein_to_row)].copy()
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)

    mode_summaries: list[dict[str, object]] = []
    for mode in modes:
        mode_dir = output_root / mode
        model_dir = mode_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(positives, mode)
        if args.reaction_augmentation_dir is not None:
            augmentation = load_aligned_feature_augmentation(
                args.reaction_augmentation_dir.resolve(), reaction_ids
            )
            reaction_matrix = np.concatenate([reaction_matrix, augmentation], axis=1)
            feature_schema["reaction_augmentation"] = {
                "directory": str(args.reaction_augmentation_dir.resolve()),
                "dimension": int(augmentation.shape[1]),
            }
        reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
        train_pairs = positives[
            positives["rhea_id"].isin(reaction_to_row) & positives["Entry"].isin(protein_to_row)
        ].drop_duplicates(["rhea_id", "Entry"])
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
                "protein_ids_file": str((args.protein_dir / "entries.csv").resolve()),
                "reaction_feature_dimension": reaction_matrix.shape[1],
                "protein_feature_dimension": protein_matrix.shape[1],
                "production_training_pairs": int(len(train_pairs)),
            }
        )
        (mode_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
        reaction_table.to_csv(mode_dir / "reaction_features.csv", index=False)
        np.save(mode_dir / "reaction_feature_matrix.npy", reaction_matrix)

        histories: list[pd.DataFrame] = []
        final_losses: list[float] = []
        for seed in seeds:
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
                seed,
                device,
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "feature_mode": mode,
                "seed": seed,
                "temperature": args.temperature,
                "feature_schema": feature_schema,
                "n_training_pairs": int(len(train_pairs)),
            }
            torch.save(checkpoint, model_dir / f"production_seed{seed}.pt")
            history_frame = pd.DataFrame(history)
            history_frame.insert(0, "seed", seed)
            histories.append(history_frame)
            final_losses.append(float(history_frame.iloc[-1]["loss"]))
        pd.concat(histories, ignore_index=True).to_csv(mode_dir / "training_history.csv", index=False)
        mode_summary = {
            "feature_mode": mode,
            "model_config": asdict(config),
            "seeds": seeds,
            "epochs": args.epochs,
            "n_training_pairs": int(len(train_pairs)),
            "n_proteins": int(len(protein_ids)),
            "n_reactions": int(len(reaction_ids)),
            "final_recorded_losses": final_losses,
            "models": str(model_dir),
            "feature_schema": str(mode_dir / "feature_schema.json"),
            "reaction_feature_matrix": str(mode_dir / "reaction_feature_matrix.npy"),
        }
        (mode_dir / "summary.json").write_text(json.dumps(mode_summary, indent=2), encoding="utf-8")
        mode_summaries.append(mode_summary)

    summary = {
        "device": str(device),
        "feature_modes": modes,
        "seeds": seeds,
        "epochs": args.epochs,
        "reaction_augmentation_dir": (
            str(args.reaction_augmentation_dir.resolve())
            if args.reaction_augmentation_dir is not None
            else None
        ),
        "modes": mode_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
