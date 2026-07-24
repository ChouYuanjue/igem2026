from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.pretrain_horizyn_reaction_feature_distillation import (  # noqa: E402
    ReactionDistillerConfig,
    ReactionFeatureDistiller,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
)
from projects.active.terpene_screening.train_marts_adapted_production import (  # noqa: E402
    DEFAULT_CURRENT_POSITIVES,
    DEFAULT_CURRENT_PROTEIN_CLUSTERS,
    DEFAULT_CURRENT_REACTION_CLUSTERS,
    DEFAULT_MARTS_PROTEIN_ENTITIES,
    DEFAULT_MARTS_REACTION_ENTITIES,
    build_union_group_maps,
    build_union_protein_library,
    build_union_reaction_library,
    build_union_training_pairs,
)
from projects.active.terpene_screening.train_marts_domain_adaptation import (  # noqa: E402
    DEFAULT_CURRENT_CANDIDATES,
    DEFAULT_CURRENT_PROTEINS,
    DEFAULT_EXTERNAL_PROTEINS,
    DEFAULT_MARTS,
    load_feature_schema,
    load_production_payloads,
)
from projects.active.terpene_screening.train_marts_horizyn_reaction_residual import (  # noqa: E402
    ResidualReactionDualTower,
    train_residual_model,
)

DEFAULT_BASE = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_DISTILLER = ROOT / "results/terpene_horizyn_reaction_feature_distillation/reaction_feature_distiller.pt"
DEFAULT_OUTPUT = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_r2e_residual"


def encode_distilled_auxiliary(
    reaction_matrix: np.ndarray,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    config = ReactionDistillerConfig(**payload["model_config"])
    if config.input_dim != reaction_matrix.shape[1]:
        raise ValueError(
            f"Distiller input {config.input_dim} does not match reaction matrix {reaction_matrix.shape[1]}"
        )
    model = ReactionFeatureDistiller(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows: list[np.ndarray] = []
    tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            rows.append(model(tensor[start : start + batch_size]).cpu().numpy())
    matrix = np.concatenate(rows, axis=0).astype(np.float32)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "config": asdict(config),
        "teacher_checkpoint": payload.get("teacher_checkpoint"),
        "validation_metrics": payload.get("validation_metrics"),
        "excluded_exact_marts_overlap": payload.get("excluded_exact_marts_overlap"),
    }
    return matrix, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three-seed production TPS R2E models with a distilled Horizyn reaction residual branch.")
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--current-positives", type=Path, default=DEFAULT_CURRENT_POSITIVES)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--external-protein-dir", type=Path, default=DEFAULT_EXTERNAL_PROTEINS)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--current-protein-clusters", type=Path, default=DEFAULT_CURRENT_PROTEIN_CLUSTERS)
    parser.add_argument("--current-reaction-clusters", type=Path, default=DEFAULT_CURRENT_REACTION_CLUSTERS)
    parser.add_argument("--marts-protein-entities", type=Path, default=DEFAULT_MARTS_PROTEIN_ENTITIES)
    parser.add_argument("--marts-reaction-entities", type=Path, default=DEFAULT_MARTS_REACTION_ENTITIES)
    parser.add_argument("--base-production-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--distiller", type=Path, default=DEFAULT_DISTILLER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    marts = pd.read_csv(args.marts.resolve(), sep="\t", dtype=str).fillna("")
    schema = load_feature_schema(args.base_production_dir.resolve())
    protein_matrix, protein_ids, protein_table, enzyme_mapping = build_union_protein_library(
        marts,
        args.current_protein_dir.resolve(),
        args.external_protein_dir.resolve(),
        args.current_candidates.resolve(),
    )
    reaction_matrix, reaction_ids, reaction_table, signature_mapping = build_union_reaction_library(
        args.current_positives.resolve(),
        marts,
        schema,
    )
    pairs = build_union_training_pairs(
        args.current_positives.resolve(),
        marts,
        enzyme_mapping,
        signature_mapping,
        set(protein_ids),
        set(reaction_ids),
    )
    protein_groups, reaction_groups = build_union_group_maps(
        protein_ids,
        reaction_table,
        args.current_protein_clusters.resolve(),
        args.current_reaction_clusters.resolve(),
        args.marts_protein_entities.resolve(),
        args.marts_reaction_entities.resolve(),
    )
    auxiliary_matrix, distiller_metadata = encode_distilled_auxiliary(
        reaction_matrix,
        args.distiller,
        device,
        args.batch_size,
    )
    deployment_schema = dict(schema)
    deployment_schema["reaction_ids"] = reaction_ids
    deployment_schema["reaction_feature_dimension"] = int(reaction_matrix.shape[1])
    deployment_schema["auxiliary_reaction_feature_dimension"] = int(auxiliary_matrix.shape[1])
    deployment_schema["model_type"] = "horizyn_reaction_residual"

    payloads = load_production_payloads(args.base_production_dir.resolve(), device)
    base_config = ModelConfig(**payloads[0]["model_config"])
    if base_config.protein_input_dim != protein_matrix.shape[1]:
        raise ValueError("Production protein dimension does not match union matrix")
    if base_config.reaction_input_dim != reaction_matrix.shape[1]:
        raise ValueError("Production reaction dimension does not match union matrix")

    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    auxiliary_tensor = torch.as_tensor(auxiliary_matrix, dtype=torch.float32, device=device)
    histories: list[pd.DataFrame] = []
    checkpoints: list[str] = []
    gate_values: list[float] = []

    for payload_index, payload in enumerate(payloads):
        seed = int(payload.get("seed", 20260723 + payload_index))
        model, history = train_residual_model(
            protein_tensor,
            reaction_tensor,
            auxiliary_tensor,
            pairs,
            protein_ids,
            reaction_ids,
            protein_groups,
            reaction_groups,
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
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "seed", seed)
        histories.append(history_frame)
        gate_values.append(model.gate_value())
        checkpoint = model_dir / f"production_seed{seed}.pt"
        torch.save(
            {
                "model_type": "horizyn_reaction_residual",
                "model_state_dict": model.state_dict(),
                "base_model_config": asdict(base_config),
                "aux_input_dim": int(auxiliary_matrix.shape[1]),
                "aux_hidden_dim": args.aux_hidden_dim,
                "gate_init": args.gate_init,
                "vector_gate": args.vector_gate,
                "topk_surrogate_weight": args.topk_surrogate_weight,
                "topk_surrogate_k": args.topk_surrogate_k,
                "topk_surrogate_margin": args.topk_surrogate_margin,
                "freeze_base_reaction": args.freeze_base_reaction,
                "seed": seed,
                "feature_schema": deployment_schema,
                "distiller_checkpoint": str(output_dir / "reaction_feature_distiller.pt"),
                "protein_registry": str(output_dir / "protein_registry.csv"),
                "reaction_registry": str(output_dir / "reaction_registry.csv"),
                "n_training_pairs": int(len(pairs)),
            },
            checkpoint,
        )
        checkpoints.append(str(checkpoint))

    pd.concat(histories, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
    protein_table.to_csv(output_dir / "protein_registry.csv", index=False)
    reaction_table.to_csv(output_dir / "reaction_registry.csv", index=False)
    pairs.to_csv(output_dir / "training_pairs.csv", index=False)
    np.save(output_dir / "reaction_feature_matrix.npy", reaction_matrix.astype(np.float32))
    np.save(output_dir / "auxiliary_reaction_feature_matrix.npy", auxiliary_matrix.astype(np.float32))
    reaction_table.to_csv(output_dir / "reaction_features.csv", index=False)
    (output_dir / "feature_schema.json").write_text(
        json.dumps(deployment_schema, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.distiller.resolve(), output_dir / "reaction_feature_distiller.pt")
    distiller_source_dir = args.distiller.resolve().parent
    for name in ["summary.json", "training_history.csv", "train_reaction_audit.csv", "validation_reaction_audit.csv"]:
        source = distiller_source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / f"distiller_{name}")

    summary = {
        "model_type": "horizyn_reaction_residual",
        "base_production_dir": str(args.base_production_dir.resolve()),
        "distiller": distiller_metadata,
        "n_current_proteins": int(protein_table["source"].astype(str).eq("current").sum()),
        "n_external_proteins": int(protein_table["source"].astype(str).eq("marts_external").sum()),
        "n_current_reactions": int(reaction_table["source"].astype(str).eq("current").sum()),
        "n_external_reactions": int(reaction_table["source"].astype(str).eq("marts_external").sum()),
        "n_training_pairs": int(len(pairs)),
        "n_current_only_pairs": int(pairs["source"].eq("current").sum()),
        "n_marts_only_pairs": int(pairs["source"].eq("marts").sum()),
        "n_shared_pairs": int(pairs["source"].eq("current;marts").sum()),
        "n_protein_groups": len(set(protein_groups.values())),
        "n_reaction_groups": len(set(reaction_groups.values())),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "topk_surrogate_weight": args.topk_surrogate_weight,
        "topk_surrogate_k": args.topk_surrogate_k,
        "topk_surrogate_margin": args.topk_surrogate_margin,
        "pu_group_mask": args.pu_group_mask,
        "aux_hidden_dim": args.aux_hidden_dim,
        "gate_init": args.gate_init,
        "vector_gate": args.vector_gate,
        "freeze_base_reaction": args.freeze_base_reaction,
        "final_gate_values": gate_values,
        "base_model_config": asdict(base_config),
        "checkpoints": checkpoints,
        "deployment_assets": {
            "reaction_feature_matrix": str(output_dir / "reaction_feature_matrix.npy"),
            "auxiliary_reaction_feature_matrix": str(output_dir / "auxiliary_reaction_feature_matrix.npy"),
            "reaction_feature_distiller": str(output_dir / "reaction_feature_distiller.pt"),
            "protein_registry": str(output_dir / "protein_registry.csv"),
            "reaction_registry": str(output_dir / "reaction_registry.csv"),
            "training_pairs": str(output_dir / "training_pairs.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
