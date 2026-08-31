from __future__ import annotations

import argparse
import json
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

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    IdentityHiddenResidualReactionDualTower,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    seed_everything,
)
from projects.active.terpene_screening.train_general_evidence_retriever import (  # noqa: E402
    _directional_full_candidate_loss,
    _query_positive_rows,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_FEATURES = DEFAULT_UNIVERSE / "reaction_features/drfp_categorical_rdkitplus_center_v1"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_identity_aux_residual"


def configure_r2e_identity_residual_trainables(
    model: IdentityHiddenResidualReactionDualTower,
) -> list[torch.nn.Parameter]:
    """Freeze both source towers; train only the zero-init auxiliary projection."""
    for parameter in model.protein_tower.parameters():
        parameter.requires_grad = False
    for parameter in model.base_reaction_tower.parameters():
        parameter.requires_grad = False
    model.aux_to_hidden.weight.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if trainable != [model.aux_to_hidden.weight]:
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        raise RuntimeError(f"Unexpected trainable parameters: {names}")
    return trainable


def _encode_base_proteins(
    model: IdentityHiddenResidualReactionDualTower,
    values: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        model.eval()
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            rows.append(model.encode_proteins(batch).detach())
    return torch.cat(rows, dim=0)


def _encode_base_reactions(
    base_model: TerpeneDualTower,
    values: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        base_model.eval()
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            rows.append(base_model.encode_reactions(batch).detach())
    return torch.cat(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a strict-clean zero-init auxiliary reaction residual on one frozen RDKit+ fold checkpoint.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--training-pairs", type=Path, required=True)
    parser.add_argument("--reaction-feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk-k", type=int, default=10)
    parser.add_argument("--topk-weight", type=float, default=0.10)
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--all-positive-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--anchor-batch-size", type=int, default=256)
    parser.add_argument("--historical-query-repeat", type=int, default=2)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.anchor_weight < 0 or args.historical_query_repeat < 0:
        raise ValueError("anchor weight/repeat must be non-negative")

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    base_dir = args.base_dir.resolve(); feature_dir = args.reaction_feature_dir.resolve()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted((base_dir / "models").glob("production_seed*.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one fold base checkpoint, found {len(checkpoints)}")
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"])
    if int(payload.get("seed", args.seed)) != args.seed:
        raise ValueError("Base checkpoint seed differs from frozen residual seed")

    protein_features, protein_ids = load_protein_library(args.universe_dir.resolve() / "proteins")
    feature_schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    reaction_features, reaction_ids = load_registered_reaction_feature_library(feature_dir, feature_schema)
    aux_dim = int(reaction_features.shape[1] - config.reaction_input_dim)
    if aux_dim <= 0:
        raise ValueError("Residual feature library must append auxiliary columns to the base checkpoint width")
    if config.protein_input_dim != protein_features.shape[1]:
        raise ValueError("Protein feature width differs from base checkpoint")

    associations = pd.read_csv(args.training_pairs.resolve(), dtype=str).fillna("")
    required = {"protein_id", "reaction_id"}
    if not required <= set(associations.columns):
        raise ValueError(f"Training pairs missing columns: {sorted(required-set(associations.columns))}")
    associations = associations[["protein_id", "reaction_id"]].drop_duplicates().copy()
    pindex = {value: row for row, value in enumerate(protein_ids)}
    rindex = {value: row for row, value in enumerate(reaction_ids)}
    if not set(associations.protein_id) <= set(pindex) or not set(associations.reaction_id) <= set(rindex):
        raise ValueError("Training pairs fall outside registered feature universe")

    train_proteins = sorted(set(associations.protein_id))
    candidate_rows = np.asarray([pindex[value] for value in train_proteins], dtype=np.int64)
    candidate_index = {value: row for row, value in enumerate(train_proteins)}
    query_ids, positives = _query_positive_rows(
        associations, direction="r2e", query_index=rindex, candidate_index=candidate_index
    )
    positive_by_query = dict(zip(query_ids, positives, strict=True))

    model = IdentityHiddenResidualReactionDualTower(config, aux_dim).to(device)
    model.load_base_state(payload["model_state_dict"])
    trainable = configure_r2e_identity_residual_trainables(model)
    base_model = TerpeneDualTower(config).to(device)
    base_model.load_state_dict(payload["model_state_dict"]); base_model.eval()

    # Exact identity audit before training on a deterministic prefix.
    audit_ids = reaction_ids[: min(512, len(reaction_ids))]
    audit_rows = np.asarray([rindex[x] for x in audit_ids], dtype=np.int64)
    full_audit = torch.as_tensor(reaction_features[audit_rows], dtype=torch.float32, device=device)
    base_audit = full_audit[:, : config.reaction_input_dim]
    with torch.no_grad():
        expected = base_model.encode_reactions(base_audit)
        actual = model.encode_reactions(full_audit)
    max_identity_diff = float((expected - actual).abs().max().cpu())
    mean_identity_diff = float((expected - actual).abs().mean().cpu())
    if max_identity_diff > 1e-7:
        raise RuntimeError(f"Residual initialization is not identity: max diff {max_identity_diff}")

    candidate_embeddings = _encode_base_proteins(
        model, protein_features[candidate_rows], device=device, chunk_size=args.feature_chunk_size
    ).detach()
    query_rows = np.asarray([rindex[value] for value in query_ids], dtype=np.int64)
    teacher_queries = _encode_base_reactions(
        base_model, reaction_features[query_rows, : config.reaction_input_dim],
        device=device, chunk_size=args.feature_chunk_size,
    ).detach()
    teacher_by_query = {value: teacher_queries[row] for row, value in enumerate(query_ids)}

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule_base = list(query_ids) * (1 + int(args.historical_query_repeat))
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        schedule = list(schedule_base); rng.shuffle(schedule); model.train()
        totals=[]; contrastives=[]; topks=[]; anchors=[]
        for start in range(0, len(schedule), args.batch_size):
            batch_ids = schedule[start : start + args.batch_size]
            rows = np.asarray([rindex[value] for value in batch_ids], dtype=np.int64)
            batch = torch.as_tensor(reaction_features[rows], dtype=torch.float32, device=device)
            query_embeddings = model.encode_reactions(batch)
            batch_positives = [positive_by_query[value] for value in batch_ids]
            loss, components = _directional_full_candidate_loss(
                query_embeddings, candidate_embeddings, batch_positives,
                temperature=args.temperature, topk_k=args.topk_k,
                topk_weight=args.topk_weight, topk_margin=args.topk_margin,
                all_positive_weight=args.all_positive_weight,
            )
            anchor_loss = torch.zeros((), device=device)
            if args.anchor_weight > 0:
                unique_batch = sorted(set(batch_ids))
                n = min(args.anchor_batch_size, len(unique_batch))
                anchor_ids = rng.sample(unique_batch, n) if n < len(unique_batch) else unique_batch
                ar = np.asarray([rindex[value] for value in anchor_ids], dtype=np.int64)
                av = torch.as_tensor(reaction_features[ar], dtype=torch.float32, device=device)
                current = model.encode_reactions(av)
                target = torch.stack([teacher_by_query[value] for value in anchor_ids])
                anchor_loss = (1.0 - (current * target).sum(dim=1)).mean()
                loss = loss + args.anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            totals.append(float(loss.detach().cpu())); contrastives.append(components["contrastive_loss"])
            topks.append(components["topk_loss"]); anchors.append(float(anchor_loss.detach().cpu()))
        row={"epoch":float(epoch),"loss":float(np.mean(totals)),"contrastive_loss":float(np.mean(contrastives)),
             "topk_loss":float(np.mean(topks)),"anchor_loss":float(np.mean(anchors))}
        history.append(row); print(json.dumps(row),flush=True)

    target = model_dir / f"production_seed{args.seed}.pt"
    torch.save({
        "model_type":"rdkitplus_identity_hidden_residual",
        "model_state_dict":model.state_dict(),
        "base_model_config":asdict(config),
        "model_config":asdict(config),
        "aux_input_dim":aux_dim,
        "seed":args.seed,
        "base_checkpoint":str(checkpoints[0].resolve()),
        "dev_fold":args.dev_fold,
        "target_benchmark_labels_read":False,
        "target_benchmark_metadata_used_for_training":False,
        "training_pairs":str(args.training_pairs.resolve()),
        "loss_candidate_scope":"training_entities",
        "identity_preserving_initialization":True,
        "freeze_base_protein":True,
        "freeze_base_reaction":True,
    }, target)
    pd.DataFrame(history).to_csv(output/"training_history.csv",index=False)
    associations.to_csv(output/"training_pairs.csv",index=False)
    (output/"feature_schema.json").write_text(json.dumps({**feature_schema,
        "model_type":"rdkitplus_identity_hidden_residual","base_reaction_feature_dimension":config.reaction_input_dim,
        "auxiliary_reaction_feature_dimension":aux_dim},indent=2),encoding="utf-8")
    summary={
        "model_type":"rdkitplus_identity_hidden_residual","dev_fold":args.dev_fold,"seed":args.seed,
        "base_dir":str(base_dir),"base_checkpoint":str(checkpoints[0].resolve()),
        "reaction_feature_dir":str(feature_dir),"reaction_input_dim":int(reaction_features.shape[1]),
        "base_reaction_input_dim":config.reaction_input_dim,"aux_input_dim":aux_dim,
        "n_train_pairs":int(len(associations)),"n_train_reactions":int(len(query_ids)),"n_train_proteins":int(len(train_proteins)),
        "loss_candidate_scope":"training_entities","target_benchmark_labels_read":False,
        "target_benchmark_metadata_used_for_training":False,"freeze_base_protein":True,"freeze_base_reaction":True,
        "trainable_parameter_names":[name for name,p in model.named_parameters() if p.requires_grad],
        "identity_audit":{"rows":len(audit_rows),"max_abs_diff":max_identity_diff,"mean_abs_diff":mean_identity_diff},
        "training":{"epochs":args.epochs,"learning_rate":args.learning_rate,"weight_decay":args.weight_decay,
          "temperature":args.temperature,"batch_size":args.batch_size,"topk_k":args.topk_k,"topk_weight":args.topk_weight,
          "topk_margin":args.topk_margin,"all_positive_weight":args.all_positive_weight,"anchor_weight":args.anchor_weight,
          "anchor_batch_size":args.anchor_batch_size,"historical_query_repeat":args.historical_query_repeat},
        "checkpoint":str(target),
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__":
    main()
