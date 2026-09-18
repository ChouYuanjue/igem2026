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

from projects.active.terpene_screening.evaluate_marts_open_world import (  # noqa: E402
    stable_external_reaction_id,
)
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_reaction,
    load_feature_schema,
    load_protein_library,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    train_model,
)
from projects.active.terpene_screening.train_marts_domain_adaptation import (  # noqa: E402
    DEFAULT_CURRENT_CANDIDATES,
    DEFAULT_CURRENT_PROTEINS,
    DEFAULT_EXTERNAL_PROTEINS,
    DEFAULT_MARTS,
    load_production_payloads,
)

DEFAULT_CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CURRENT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_CURRENT_REACTION_CLUSTERS = ROOT / "data/terpene_cold_splits/reaction_cluster_folds.csv"
DEFAULT_MARTS_PROTEIN_ENTITIES = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_MARTS_REACTION_ENTITIES = ROOT / "data/terpene_marts_adaptation/reaction_entities.csv"
DEFAULT_BASE = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_OUTPUT = ROOT / "results/terpene_production_models/marts_adapted_drfp"


def normalize_sequence(value: object) -> str:
    return "".join(str(value).split()).upper()


def build_union_protein_library(
    marts: pd.DataFrame,
    current_dir: Path,
    external_dir: Path,
    current_candidates_path: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, str]]:
    current_features, current_ids = load_protein_library(current_dir)
    external_features, external_ids = load_protein_library(external_dir)
    duplicate_ids = set(current_ids) & set(external_ids)
    if duplicate_ids:
        raise ValueError(f"Current/external protein IDs overlap: {sorted(duplicate_ids)[:5]}")
    protein_ids = current_ids + external_ids
    protein_matrix = np.concatenate([current_features, external_features], axis=0).astype(np.float32)
    protein_id_set = set(protein_ids)

    candidates = pd.read_csv(current_candidates_path, sep="\t", dtype=str).fillna("")
    candidates["normalized_sequence"] = candidates["Sequence"].map(normalize_sequence)
    current_by_sequence = (
        candidates[candidates["Entry"].isin(current_ids)]
        .groupby("normalized_sequence")["Entry"]
        .apply(lambda values: sorted(set(values.astype(str)))[0])
        .to_dict()
    )
    marts_unique = marts[["enzyme_id", "sequence"]].drop_duplicates("enzyme_id")
    enzyme_mapping: dict[str, str] = {}
    sequence_by_external: dict[str, str] = {}
    for row in marts_unique.itertuples(index=False):
        enzyme_id = str(row.enzyme_id)
        sequence = normalize_sequence(row.sequence)
        if enzyme_id in protein_id_set:
            enzyme_mapping[enzyme_id] = enzyme_id
        elif sequence in current_by_sequence:
            enzyme_mapping[enzyme_id] = current_by_sequence[sequence]
        if enzyme_id in set(external_ids):
            sequence_by_external[enzyme_id] = sequence

    rows = [
        {"protein_id": value, "source": "current", "sequence": ""}
        for value in current_ids
    ] + [
        {"protein_id": value, "source": "marts_external", "sequence": sequence_by_external.get(value, "")}
        for value in external_ids
    ]
    return protein_matrix, protein_ids, pd.DataFrame(rows), enzyme_mapping


def build_union_reaction_library(
    current_positives_path: Path,
    marts: pd.DataFrame,
    schema: dict[str, object],
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, str]]:
    current = pd.read_csv(current_positives_path, sep="\t", dtype=str).fillna("")
    current = current.drop_duplicates("rhea_id").copy()
    current["reaction_signature"] = current["smiles_seq"].map(reaction_signature)
    current = current.sort_values("rhea_id").reset_index(drop=True)
    reaction_ids = current["rhea_id"].astype(str).tolist()
    reaction_features = [encode_reaction(value, schema) for value in current["smiles_seq"].astype(str)]
    signature_to_current = (
        current[current["reaction_signature"] != ""]
        .groupby("reaction_signature")["rhea_id"]
        .apply(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    signature_mapping = {value: ids[0] for value, ids in signature_to_current.items()}
    rows = [
        {
            "reaction_id": row.rhea_id,
            "reaction_signature": row.reaction_signature,
            "reaction_smiles": row.smiles_seq,
            "source": "current",
        }
        for row in current[["rhea_id", "reaction_signature", "smiles_seq"]].itertuples(index=False)
    ]

    unseen = marts[(marts["reaction_signature"] != "") & (~marts["reaction_signature"].isin(signature_mapping))]
    for signature, group in unseen.groupby("reaction_signature", sort=True):
        reaction_id = stable_external_reaction_id(str(signature))
        reaction_smiles = str(group.iloc[0]["reaction_smiles"])
        signature_mapping[str(signature)] = reaction_id
        reaction_ids.append(reaction_id)
        reaction_features.append(encode_reaction(reaction_smiles, schema))
        rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_signature": str(signature),
                "reaction_smiles": reaction_smiles,
                "source": "marts_external",
            }
        )
    return np.stack(reaction_features).astype(np.float32), reaction_ids, pd.DataFrame(rows), signature_mapping


def build_union_training_pairs(
    current_positives_path: Path,
    marts: pd.DataFrame,
    enzyme_mapping: dict[str, str],
    signature_mapping: dict[str, str],
    protein_ids: set[str],
    reaction_ids: set[str],
) -> pd.DataFrame:
    current = pd.read_csv(current_positives_path, sep="\t", dtype=str).fillna("")
    current = current[["Entry", "rhea_id"]].drop_duplicates()
    current = current[current["Entry"].isin(protein_ids) & current["rhea_id"].isin(reaction_ids)].copy()
    current["source"] = "current"

    marts_rows = []
    for row in marts[["enzyme_id", "reaction_signature"]].itertuples(index=False):
        protein_id = enzyme_mapping.get(str(row.enzyme_id))
        reaction_id = signature_mapping.get(str(row.reaction_signature))
        if protein_id in protein_ids and reaction_id in reaction_ids:
            marts_rows.append({"Entry": protein_id, "rhea_id": reaction_id, "source": "marts"})
    combined = pd.concat([current, pd.DataFrame(marts_rows)], ignore_index=True)
    source_summary = (
        combined.groupby(["Entry", "rhea_id"])["source"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .reset_index()
    )
    return source_summary


def build_union_group_maps(
    protein_ids: list[str],
    reaction_table: pd.DataFrame,
    current_protein_clusters_path: Path,
    current_reaction_clusters_path: Path,
    marts_protein_entities_path: Path,
    marts_reaction_entities_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    current_proteins = pd.read_csv(current_protein_clusters_path, dtype=str).fillna("")
    protein_groups = dict(zip(current_proteins["entry"].astype(str), current_proteins["cluster_id"].astype(str)))
    marts_proteins = pd.read_csv(marts_protein_entities_path, dtype=str).fillna("")
    for row in marts_proteins[["aliases", "cluster_id"]].itertuples(index=False):
        for alias in str(row.aliases).split(";"):
            alias = alias.strip()
            if alias and alias not in protein_groups:
                protein_groups[alias] = str(row.cluster_id)
    protein_groups = {value: protein_groups.get(value, value) for value in protein_ids}

    current_reactions = pd.read_csv(current_reaction_clusters_path, dtype=str).fillna("")
    reaction_groups = dict(
        zip(current_reactions["reaction_id"].astype(str), current_reactions["reaction_cluster"].astype(str))
    )
    marts_reactions = pd.read_csv(marts_reaction_entities_path, dtype=str).fillna("")
    cluster_by_signature = dict(
        zip(marts_reactions["reaction_signature"].astype(str), marts_reactions["cluster_id"].astype(str))
    )
    for row in reaction_table[["reaction_id", "reaction_signature"]].itertuples(index=False):
        reaction_id = str(row.reaction_id)
        if reaction_id not in reaction_groups:
            reaction_groups[reaction_id] = cluster_by_signature.get(str(row.reaction_signature), reaction_id)
    return protein_groups, reaction_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune production TPS dual towers on current+MARTS rehearsal pairs.")
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-k", type=int, default=0)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=1)
    parser.add_argument("--hard-negative-end-epoch", type=int, default=0)
    parser.add_argument(
        "--model-selection",
        choices=["min_loss", "final"],
        default="min_loss",
    )
    parser.add_argument("--pu-group-mask", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--freeze-protein-tower", action="store_true")
    parser.add_argument("--freeze-reaction-tower", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    marts = pd.read_csv(args.marts, sep="\t", dtype=str).fillna("")
    marts["reaction_seen"] = marts["reaction_seen"].astype(str).str.lower().eq("true")
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

    payloads = load_production_payloads(args.base_production_dir.resolve(), device)
    config = ModelConfig(**payloads[0]["model_config"])
    if config.protein_input_dim != protein_matrix.shape[1] or config.reaction_input_dim != reaction_matrix.shape[1]:
        raise ValueError("Base production dimensions do not match union feature matrices")

    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    anchor_protein_rows = np.flatnonzero(protein_table["source"].astype(str).eq("current").to_numpy()).astype(np.int64)
    anchor_reaction_rows = np.flatnonzero(reaction_table["source"].astype(str).eq("current").to_numpy()).astype(np.int64)
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}

    histories: list[pd.DataFrame] = []
    checkpoints: list[str] = []
    for payload_index, payload in enumerate(payloads):
        seed = int(payload.get("seed", 20260723 + payload_index))
        base_model = TerpeneDualTower(config).to(device)
        base_model.load_state_dict(payload["model_state_dict"])
        base_model.eval()
        with torch.no_grad():
            anchor_protein_targets = base_model.encode_proteins(
                protein_tensor[torch.as_tensor(anchor_protein_rows, dtype=torch.long, device=device)]
            ).detach()
            anchor_reaction_targets = base_model.encode_reactions(
                reaction_tensor[torch.as_tensor(anchor_reaction_rows, dtype=torch.long, device=device)]
            ).detach()
        model, history = train_model(
            protein_tensor,
            reaction_tensor,
            pairs,
            protein_to_row,
            reaction_to_row,
            config,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            args.temperature,
            seed,
            device,
            initial_state_dict=payload["model_state_dict"],
            protein_group_map=protein_groups,
            reaction_group_map=reaction_groups,
            exclude_same_group_negatives=args.pu_group_mask,
            anchor_protein_rows=anchor_protein_rows,
            anchor_protein_targets=anchor_protein_targets,
            anchor_reaction_rows=anchor_reaction_rows,
            anchor_reaction_targets=anchor_reaction_targets,
            anchor_weight=args.anchor_weight,
            freeze_protein_tower=args.freeze_protein_tower,
            freeze_reaction_tower=args.freeze_reaction_tower,
            reaction_loss_weight=args.reaction_loss_weight,
            hard_negative_k=args.hard_negative_k,
            hard_negative_start_epoch=args.hard_negative_start_epoch,
            hard_negative_end_epoch=args.hard_negative_end_epoch,
            model_selection=args.model_selection,
        )
        history_frame = pd.DataFrame(history)
        history_frame.insert(0, "seed", seed)
        histories.append(history_frame)
        checkpoint = model_dir / f"production_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "feature_schema": schema,
                "seed": seed,
                "base_production_dir": str(args.base_production_dir.resolve()),
                "training_sources": [str(args.current_positives.resolve()), str(args.marts.resolve())],
                "n_training_pairs": int(len(pairs)),
                "hard_negative_k": args.hard_negative_k,
                "hard_negative_start_epoch": args.hard_negative_start_epoch,
                "hard_negative_end_epoch": args.hard_negative_end_epoch,
                "model_selection": args.model_selection,
                "protein_registry": str(output_dir / "protein_registry.csv"),
                "reaction_registry": str(output_dir / "reaction_registry.csv"),
            },
            checkpoint,
        )
        checkpoints.append(str(checkpoint))

    pd.concat(histories, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
    for asset_name in ["reaction_feature_matrix.npy", "reaction_features.csv"]:
        source_asset = args.base_production_dir.resolve() / asset_name
        if source_asset.exists():
            shutil.copy2(source_asset, output_dir / asset_name)
    protein_table.to_csv(output_dir / "protein_registry.csv", index=False)
    reaction_table.to_csv(output_dir / "reaction_registry.csv", index=False)
    pairs.to_csv(output_dir / "training_pairs.csv", index=False)
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {
        "base_production_dir": str(args.base_production_dir.resolve()),
        "feature_mode": schema.get("feature_mode"),
        "n_current_proteins": int((protein_table["source"] == "current").sum()),
        "n_external_proteins": int((protein_table["source"] == "marts_external").sum()),
        "n_current_reactions": int((reaction_table["source"] == "current").sum()),
        "n_external_reactions": int((reaction_table["source"] == "marts_external").sum()),
        "n_training_pairs": int(len(pairs)),
        "n_current_only_pairs": int(pairs["source"].eq("current").sum()),
        "n_marts_only_pairs": int(pairs["source"].eq("marts").sum()),
        "n_shared_pairs": int(pairs["source"].eq("current;marts").sum()),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "pu_group_mask": args.pu_group_mask,
        "anchor_weight": args.anchor_weight,
        "n_anchor_proteins": int(len(anchor_protein_rows)),
        "n_anchor_reactions": int(len(anchor_reaction_rows)),
        "freeze_protein_tower": args.freeze_protein_tower,
        "freeze_reaction_tower": args.freeze_reaction_tower,
        "reaction_loss_weight": args.reaction_loss_weight,
        "hard_negative_k": args.hard_negative_k,
        "hard_negative_start_epoch": args.hard_negative_start_epoch,
        "hard_negative_end_epoch": args.hard_negative_end_epoch,
        "model_selection": args.model_selection,
        "n_protein_groups": len(set(protein_groups.values())),
        "n_reaction_groups": len(set(reaction_groups.values())),
        "checkpoints": checkpoints,
        "deployment_assets": {
            "reaction_feature_matrix": str(output_dir / "reaction_feature_matrix.npy"),
            "protein_registry": str(output_dir / "protein_registry.csv"),
            "reaction_registry": str(output_dir / "reaction_registry.csv"),
        },
        "model_config": asdict(config),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
