from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_reaction_library,
)

DEFAULT_DEPLOYMENT = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_REGISTERED_PROTEINS = ROOT / "data/terpene_open_world_registry/proteins"
DEFAULT_OUTPUT = ROOT / "results/terpene_deployment_validation.json"


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a deployable open-world TPS dual-tower directory.")
    parser.add_argument("--deployment-dir", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEINS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    deployment = args.deployment_dir.resolve()
    required_paths = [
        deployment / "feature_schema.json",
        deployment / "reaction_feature_matrix.npy",
        deployment / "reaction_features.csv",
        deployment / "protein_registry.csv",
        deployment / "reaction_registry.csv",
        deployment / "training_pairs.csv",
        deployment / "summary.json",
        deployment / "models",
    ]
    deployment_summary = json.loads(
        (deployment / "summary.json").read_text(encoding="utf-8")
    ) if (deployment / "summary.json").exists() else {}
    if deployment_summary.get("model_type") == "horizyn_reaction_residual":
        required_paths.extend(
            [
                deployment / "reaction_feature_distiller.pt",
                deployment / "auxiliary_reaction_feature_matrix.npy",
            ]
        )
    elif deployment_summary.get("model_type") == "horizyn_reaction_residual_exact":
        required_paths.extend(
            [
                deployment / "auxiliary_reaction_feature_matrix.npy",
                deployment / "horizyn_v1_0_dev.ckpt",
                deployment / "horizyn_sota.yaml",
                deployment / "reaction_feature_distiller.pt",
                deployment / "HORIZYN_LICENSE",
            ]
        )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing deployment assets: {missing_paths}")

    schema = load_feature_schema(deployment)
    reaction_features, current_reaction_ids = load_reaction_library(deployment, schema)
    current_protein_features, current_protein_ids = load_protein_library(args.current_protein_dir.resolve())
    registered_protein_features, registered_protein_ids = load_protein_library(args.registered_protein_dir.resolve())
    overlap = set(current_protein_ids) & set(registered_protein_ids)
    if overlap:
        raise ValueError(f"Current and registered protein libraries overlap: {sorted(overlap)[:5]}")

    protein_registry = pd.read_csv(deployment / "protein_registry.csv", dtype=str).fillna("")
    reaction_registry = pd.read_csv(deployment / "reaction_registry.csv", dtype=str).fillna("")
    training_pairs = pd.read_csv(deployment / "training_pairs.csv", dtype=str).fillna("")
    require_columns(protein_registry, {"protein_id", "source"}, "protein_registry")
    require_columns(reaction_registry, {"reaction_id", "reaction_smiles", "source"}, "reaction_registry")
    require_columns(training_pairs, {"Entry", "rhea_id", "source"}, "training_pairs")

    expected_proteins = set(current_protein_ids) | set(registered_protein_ids)
    registry_proteins = set(protein_registry["protein_id"].astype(str))
    if not registry_proteins <= expected_proteins:
        missing = sorted(registry_proteins - expected_proteins)
        raise ValueError(f"Deployment training proteins absent from active registries: {missing[:5]}")
    matrix_reaction_set = set(current_reaction_ids)
    registered_reaction_set = set(reaction_registry["reaction_id"].astype(str))
    if not matrix_reaction_set <= registered_reaction_set:
        missing_from_registry = sorted(matrix_reaction_set - registered_reaction_set)
        raise ValueError(
            "Packaged reaction matrix contains IDs absent from the deployment registry: "
            f"{missing_from_registry[:5]}"
        )
    runtime_reactions = reaction_registry[
        ~reaction_registry["reaction_id"].astype(str).isin(matrix_reaction_set)
    ]
    if runtime_reactions["reaction_smiles"].astype(str).str.strip().eq("").any():
        examples = runtime_reactions.loc[
            runtime_reactions["reaction_smiles"].astype(str).str.strip().eq(""),
            "reaction_id",
        ].astype(str).head().tolist()
        raise ValueError(
            "Runtime registry reactions require reaction SMILES for feature encoding: "
            f"{examples}"
        )
    if not set(training_pairs["Entry"].astype(str)) <= registry_proteins:
        raise ValueError("Training pairs contain proteins absent from registry")
    if not set(training_pairs["rhea_id"].astype(str)) <= registered_reaction_set:
        raise ValueError("Training pairs contain reactions absent from registry")

    if deployment_summary.get("model_type") in {
        "horizyn_reaction_residual",
        "horizyn_reaction_residual_exact",
    }:
        auxiliary = np.load(deployment / "auxiliary_reaction_feature_matrix.npy")
        if len(auxiliary) != len(reaction_features):
            raise ValueError("Auxiliary and base reaction matrices differ in row count")
        expected_aux_dim = int(schema.get("auxiliary_reaction_feature_dimension", auxiliary.shape[1]))
        if auxiliary.shape[1] != expected_aux_dim:
            raise ValueError("Auxiliary reaction feature dimension differs from schema")

    models = load_models(deployment / "models", "production", torch.device(args.device))
    model_configs = []
    for model in models:
        config = model.config
        if config.protein_input_dim != current_protein_features.shape[1]:
            raise ValueError("Protein input dimension does not match ESM-C features")
        if config.reaction_input_dim != reaction_features.shape[1]:
            raise ValueError("Reaction input dimension does not match feature matrix")
        model_configs.append(
            {
                "protein_input_dim": config.protein_input_dim,
                "reaction_input_dim": config.reaction_input_dim,
                "hidden_dim": config.hidden_dim,
                "embedding_dim": config.embedding_dim,
            }
        )

    report = {
        "status": "valid",
        "deployment_dir": str(deployment),
        "n_models": len(models),
        "n_current_proteins": len(current_protein_ids),
        "n_registered_external_proteins": len(registered_protein_ids),
        "n_deployment_training_proteins": len(registry_proteins),
        "n_total_active_proteins": len(expected_proteins),
        "n_current_reactions": int(
            reaction_registry["source"].astype(str).eq("current").sum()
        ),
        "n_registered_external_reactions": int(
            reaction_registry["source"].astype(str).eq("marts_external").sum()
        ),
        "n_total_reactions": len(registered_reaction_set),
        "n_training_pairs": len(training_pairs),
        "reaction_feature_shape": list(reaction_features.shape),
        "protein_feature_shapes": {
            "current": list(current_protein_features.shape),
            "registered_external": list(registered_protein_features.shape),
        },
        "feature_mode": schema.get("feature_mode"),
        "model_type": deployment_summary.get("model_type", "dual_tower"),
        "license": deployment_summary.get("license"),
        "model_configs": model_configs,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
