from __future__ import annotations

import argparse
import hashlib
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
    BoundedIdentityHiddenResidualReactionDualTower,
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
DEFAULT_PROTEINS = DEFAULT_UNIVERSE / "proteins"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_directional_identity_aux_residual"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_identity_residual_trainables(
    model: BoundedIdentityHiddenResidualReactionDualTower,
) -> list[torch.nn.Parameter]:
    """Freeze both source towers; only the zero-init reaction-center projection may train."""
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


def encode_proteins(
    model: BoundedIdentityHiddenResidualReactionDualTower,
    values: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            rows.append(model.encode_proteins(batch).detach())
    return torch.cat(rows, dim=0)


def encode_base_reactions(
    base_model: TerpeneDualTower,
    values: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    base_model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            rows.append(base_model.encode_reactions(batch).detach())
    return torch.cat(rows, dim=0)


def _sample_global_teacher_candidates(
    query_embeddings: torch.Tensor,
    teacher_candidate_embeddings: torch.Tensor,
    positive_rows: list[np.ndarray],
    *,
    hard_candidates: int,
    random_candidates: int,
    rng: random.Random,
) -> np.ndarray:
    """Select an E2R reaction minibatch using only the frozen base teacher and train positives.

    The union always contains every positive for the current protein queries.  Hard rows are the
    reactions with the largest teacher score for any query in the batch, independent of target
    benchmark data. Random rows add coverage without requiring all ~10k reaction candidates to
    pass through the trainable residual every optimizer step.
    """
    if hard_candidates < 0 or random_candidates < 0:
        raise ValueError("candidate counts must be non-negative")
    positive_union = sorted({int(row) for values in positive_rows for row in values.tolist()})
    selected = set(positive_union)
    if hard_candidates > 0:
        with torch.no_grad():
            teacher_scores = query_embeddings @ teacher_candidate_embeddings.T
            per_candidate = teacher_scores.max(dim=0).values
            k = min(int(hard_candidates), int(per_candidate.numel()))
            selected.update(map(int, torch.topk(per_candidate, k=k).indices.cpu().tolist()))
    if random_candidates > 0 and len(selected) < teacher_candidate_embeddings.shape[0]:
        remaining = [row for row in range(teacher_candidate_embeddings.shape[0]) if row not in selected]
        selected.update(rng.sample(remaining, min(int(random_candidates), len(remaining))))
    return np.asarray(sorted(selected), dtype=np.int64)


def _local_positive_rows(positive_rows: list[np.ndarray], selected_rows: np.ndarray) -> list[np.ndarray]:
    local = {int(source): local_row for local_row, source in enumerate(selected_rows.tolist())}
    output: list[np.ndarray] = []
    for positives in positive_rows:
        mapped = sorted({local[int(row)] for row in positives.tolist() if int(row) in local})
        if not mapped:
            raise AssertionError("sampled E2R candidate pool dropped all positives for a query")
        output.append(np.asarray(mapped, dtype=np.int64))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a strict-clean direction-aware zero-init bounded reaction-center residual on one frozen "
            "RDKit+ fold checkpoint. R2E preserves the historical full-training-entity loss; E2R uses a "
            "teacher-selected train-only reaction minibatch for tractable gradients."
        )
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--training-pairs", type=Path, required=True)
    parser.add_argument("--protein-feature-dir", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--reaction-feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direction", choices=("r2e", "e2r"), required=True)
    parser.add_argument("--dev-fold", type=int, required=True)
    parser.add_argument("--max-residual-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
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
    parser.add_argument("--e2r-hard-candidates", type=int, default=512)
    parser.add_argument("--e2r-random-candidates", type=int, default=128)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs <= 0 or args.steps_per_epoch <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, steps-per-epoch, batch-size and learning-rate must be positive")
    if not 0 < args.max_residual_ratio <= 1:
        raise ValueError("max-residual-ratio must be in (0,1]")
    if args.anchor_weight < 0 or args.historical_query_repeat < 0:
        raise ValueError("anchor weight/repeat must be non-negative")

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    base_dir = args.base_dir.resolve()
    protein_dir = args.protein_feature_dir.resolve()
    feature_dir = args.reaction_feature_dir.resolve()
    training_pairs_path = args.training_pairs.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted((base_dir / "models").glob("production_seed*.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one fold base checkpoint, found {len(checkpoints)}")
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"])
    if int(payload.get("seed", args.seed)) != args.seed:
        raise ValueError("Base checkpoint seed differs from frozen residual seed")

    protein_features, protein_ids = load_protein_library(protein_dir)
    feature_schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    reaction_features, reaction_ids = load_registered_reaction_feature_library(feature_dir, feature_schema)
    aux_dim = int(reaction_features.shape[1] - config.reaction_input_dim)
    if aux_dim <= 0:
        raise ValueError("Residual feature library must append auxiliary columns to base checkpoint width")
    if int(config.protein_input_dim) != int(protein_features.shape[1]):
        raise ValueError(
            f"Protein feature width differs from base checkpoint: {protein_features.shape[1]} != {config.protein_input_dim}"
        )

    associations = pd.read_csv(training_pairs_path, dtype=str).fillna("")
    required = {"protein_id", "reaction_id"}
    if not required <= set(associations.columns):
        raise ValueError(f"Training pairs missing columns: {sorted(required-set(associations.columns))}")
    associations = associations[["protein_id", "reaction_id"]].drop_duplicates().copy()
    pindex = {value: row for row, value in enumerate(protein_ids)}
    rindex = {value: row for row, value in enumerate(reaction_ids)}
    if not set(associations.protein_id) <= set(pindex) or not set(associations.reaction_id) <= set(rindex):
        raise ValueError("Training pairs fall outside registered feature universe")

    model = BoundedIdentityHiddenResidualReactionDualTower(config, aux_dim, args.max_residual_ratio).to(device)
    model.load_base_state(payload["model_state_dict"])
    trainable = configure_identity_residual_trainables(model)
    base_model = TerpeneDualTower(config).to(device)
    base_model.load_state_dict(payload["model_state_dict"])
    base_model.eval()

    # Exact identity before training. Evaluation mode is essential because both source towers include dropout.
    model.eval()
    audit_ids = reaction_ids[: min(512, len(reaction_ids))]
    audit_rows = np.asarray([rindex[value] for value in audit_ids], dtype=np.int64)
    full_audit = torch.as_tensor(reaction_features[audit_rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        expected = base_model.encode_reactions(full_audit[:, : config.reaction_input_dim])
        actual = model.encode_reactions(full_audit)
    max_identity_diff = float((expected - actual).abs().max().cpu())
    mean_identity_diff = float((expected - actual).abs().mean().cpu())
    if max_identity_diff > 1e-7:
        raise RuntimeError(f"Residual initialization is not identity: max diff {max_identity_diff}")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []

    if args.direction == "r2e":
        train_candidates = sorted(set(associations.protein_id))
        candidate_rows = np.asarray([pindex[value] for value in train_candidates], dtype=np.int64)
        candidate_index = {value: row for row, value in enumerate(train_candidates)}
        query_ids, positives = _query_positive_rows(
            associations, direction="r2e", query_index=rindex, candidate_index=candidate_index
        )
        positive_by_query = dict(zip(query_ids, positives, strict=True))
        candidate_embeddings = encode_proteins(
            model, protein_features[candidate_rows], device=device, chunk_size=args.feature_chunk_size
        ).detach()
        query_rows = np.asarray([rindex[value] for value in query_ids], dtype=np.int64)
        teacher_queries = encode_base_reactions(
            base_model,
            reaction_features[query_rows, : config.reaction_input_dim],
            device=device,
            chunk_size=args.feature_chunk_size,
        ).detach()
        teacher_by_query = {value: teacher_queries[row] for row, value in enumerate(query_ids)}
        schedule_base = list(query_ids) * (1 + int(args.historical_query_repeat))

        for epoch in range(1, args.epochs + 1):
            schedule = list(schedule_base)
            rng.shuffle(schedule)
            model.train()
            totals: list[float] = []
            contrastives: list[float] = []
            topks: list[float] = []
            anchors: list[float] = []
            for start in range(0, len(schedule), args.batch_size):
                batch_ids = schedule[start : start + args.batch_size]
                rows = np.asarray([rindex[value] for value in batch_ids], dtype=np.int64)
                batch = torch.as_tensor(reaction_features[rows], dtype=torch.float32, device=device)
                query_embeddings = model.encode_reactions(batch)
                batch_positives = [positive_by_query[value] for value in batch_ids]
                loss, components = _directional_full_candidate_loss(
                    query_embeddings,
                    candidate_embeddings,
                    batch_positives,
                    temperature=args.temperature,
                    topk_k=args.topk_k,
                    topk_weight=args.topk_weight,
                    topk_margin=args.topk_margin,
                    all_positive_weight=args.all_positive_weight,
                )
                anchor_loss = torch.zeros((), device=device)
                if args.anchor_weight > 0:
                    unique_batch = sorted(set(batch_ids))
                    n = min(args.anchor_batch_size, len(unique_batch))
                    anchor_ids = rng.sample(unique_batch, n) if n < len(unique_batch) else unique_batch
                    anchor_rows = np.asarray([rindex[value] for value in anchor_ids], dtype=np.int64)
                    current = model.encode_reactions(
                        torch.as_tensor(reaction_features[anchor_rows], dtype=torch.float32, device=device)
                    )
                    target = torch.stack([teacher_by_query[value] for value in anchor_ids])
                    anchor_loss = (1.0 - (current * target).sum(dim=1)).mean()
                    loss = loss + args.anchor_weight * anchor_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals.append(float(loss.detach().cpu()))
                contrastives.append(components["contrastive_loss"])
                topks.append(components["topk_loss"])
                anchors.append(float(anchor_loss.detach().cpu()))
            row = {
                "epoch": float(epoch),
                "loss": float(np.mean(totals)),
                "contrastive_loss": float(np.mean(contrastives)),
                "topk_loss": float(np.mean(topks)),
                "anchor_loss": float(np.mean(anchors)),
                "optimizer_steps": float(len(totals)),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        n_queries = len(query_ids)
        n_candidates = len(train_candidates)
        candidate_scope = "all_training_proteins"

    else:
        train_candidates = sorted(set(associations.reaction_id))
        candidate_source_rows = np.asarray([rindex[value] for value in train_candidates], dtype=np.int64)
        candidate_index = {value: row for row, value in enumerate(train_candidates)}
        query_ids, positives = _query_positive_rows(
            associations, direction="e2r", query_index=pindex, candidate_index=candidate_index
        )
        positive_by_query = dict(zip(query_ids, positives, strict=True))
        query_source_rows = np.asarray([pindex[value] for value in query_ids], dtype=np.int64)
        query_embeddings = encode_proteins(
            model, protein_features[query_source_rows], device=device, chunk_size=args.feature_chunk_size
        ).detach()
        teacher_candidate_embeddings = encode_base_reactions(
            base_model,
            reaction_features[candidate_source_rows, : config.reaction_input_dim],
            device=device,
            chunk_size=args.feature_chunk_size,
        ).detach()
        query_embedding_by_id = {value: query_embeddings[row] for row, value in enumerate(query_ids)}
        query_pool = list(query_ids)

        for epoch in range(1, args.epochs + 1):
            model.train()
            totals: list[float] = []
            contrastives: list[float] = []
            topks: list[float] = []
            anchors: list[float] = []
            candidate_sizes: list[int] = []
            for _step in range(args.steps_per_epoch):
                batch_ids = rng.sample(query_pool, min(args.batch_size, len(query_pool)))
                batch_query = torch.stack([query_embedding_by_id[value] for value in batch_ids])
                batch_positives = [positive_by_query[value] for value in batch_ids]
                selected_local_rows = _sample_global_teacher_candidates(
                    batch_query,
                    teacher_candidate_embeddings,
                    batch_positives,
                    hard_candidates=args.e2r_hard_candidates,
                    random_candidates=args.e2r_random_candidates,
                    rng=rng,
                )
                local_positives = _local_positive_rows(batch_positives, selected_local_rows)
                selected_source_rows = candidate_source_rows[selected_local_rows]
                current_candidates = model.encode_reactions(
                    torch.as_tensor(reaction_features[selected_source_rows], dtype=torch.float32, device=device)
                )
                loss, components = _directional_full_candidate_loss(
                    batch_query,
                    current_candidates,
                    local_positives,
                    temperature=args.temperature,
                    topk_k=args.topk_k,
                    topk_weight=args.topk_weight,
                    topk_margin=args.topk_margin,
                    all_positive_weight=args.all_positive_weight,
                )
                anchor_loss = torch.zeros((), device=device)
                if args.anchor_weight > 0:
                    n = min(args.anchor_batch_size, len(selected_local_rows))
                    anchor_local = rng.sample(selected_local_rows.tolist(), n) if n < len(selected_local_rows) else selected_local_rows.tolist()
                    anchor_local_np = np.asarray(anchor_local, dtype=np.int64)
                    anchor_source = candidate_source_rows[anchor_local_np]
                    current_anchor = model.encode_reactions(
                        torch.as_tensor(reaction_features[anchor_source], dtype=torch.float32, device=device)
                    )
                    teacher_anchor = teacher_candidate_embeddings[
                        torch.as_tensor(anchor_local_np, dtype=torch.long, device=device)
                    ]
                    anchor_loss = (1.0 - (current_anchor * teacher_anchor).sum(dim=1)).mean()
                    loss = loss + args.anchor_weight * anchor_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals.append(float(loss.detach().cpu()))
                contrastives.append(components["contrastive_loss"])
                topks.append(components["topk_loss"])
                anchors.append(float(anchor_loss.detach().cpu()))
                candidate_sizes.append(int(len(selected_local_rows)))
            row = {
                "epoch": float(epoch),
                "loss": float(np.mean(totals)),
                "contrastive_loss": float(np.mean(contrastives)),
                "topk_loss": float(np.mean(topks)),
                "anchor_loss": float(np.mean(anchors)),
                "optimizer_steps": float(len(totals)),
                "mean_sampled_reaction_candidates": float(np.mean(candidate_sizes)),
                "max_sampled_reaction_candidates": float(np.max(candidate_sizes)),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        n_queries = len(query_ids)
        n_candidates = len(train_candidates)
        candidate_scope = "teacher_hard_plus_random_training_reactions_per_step"

    target = model_dir / f"production_seed{args.seed}.pt"
    torch.save(
        {
            "model_type": "rdkitplus_bounded_identity_hidden_residual",
            "model_state_dict": model.state_dict(),
            "base_model_config": asdict(config),
            "model_config": asdict(config),
            "aux_input_dim": aux_dim,
            "max_residual_ratio": float(args.max_residual_ratio),
            "seed": args.seed,
            "base_checkpoint": str(checkpoints[0].resolve()),
            "dev_fold": args.dev_fold,
            "direction": args.direction,
            "target_benchmark_labels_read": False,
            "target_benchmark_metadata_used_for_training": False,
            "training_pairs": str(training_pairs_path),
            "protein_feature_dir": str(protein_dir),
            "loss_candidate_scope": candidate_scope,
            "identity_preserving_initialization": True,
            "freeze_base_protein": True,
            "freeze_base_reaction": True,
        },
        target,
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    associations.to_csv(output / "training_pairs.csv", index=False)
    (output / "feature_schema.json").write_text(
        json.dumps(
            {
                **feature_schema,
                "model_type": "rdkitplus_bounded_identity_hidden_residual",
                "base_reaction_feature_dimension": config.reaction_input_dim,
                "auxiliary_reaction_feature_dimension": aux_dim,
                "max_residual_ratio": float(args.max_residual_ratio),
                "direction": args.direction,
                "protein_feature_dir": str(protein_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_type": "rdkitplus_bounded_identity_hidden_residual",
        "direction": args.direction,
        "dev_fold": args.dev_fold,
        "seed": args.seed,
        "base_dir": str(base_dir),
        "base_checkpoint": str(checkpoints[0].resolve()),
        "base_checkpoint_sha256": sha256_file(checkpoints[0]),
        "training_pairs": str(training_pairs_path),
        "training_pairs_sha256": sha256_file(training_pairs_path),
        "protein_feature_dir": str(protein_dir),
        "reaction_feature_dir": str(feature_dir),
        "reaction_input_dim": int(reaction_features.shape[1]),
        "base_reaction_input_dim": int(config.reaction_input_dim),
        "protein_input_dim": int(config.protein_input_dim),
        "aux_input_dim": aux_dim,
        "max_residual_ratio": float(args.max_residual_ratio),
        "n_train_pairs": int(len(associations)),
        "n_train_queries": int(n_queries),
        "n_train_candidates": int(n_candidates),
        "loss_candidate_scope": candidate_scope,
        "target_benchmark_labels_read": False,
        "target_benchmark_metadata_used_for_training": False,
        "freeze_base_protein": True,
        "freeze_base_reaction": True,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "identity_audit": {
            "rows": len(audit_rows),
            "max_abs_diff": max_identity_diff,
            "mean_abs_diff": mean_identity_diff,
        },
        "training": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch if args.direction == "e2r" else None,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "topk_k": args.topk_k,
            "topk_weight": args.topk_weight,
            "topk_margin": args.topk_margin,
            "all_positive_weight": args.all_positive_weight,
            "anchor_weight": args.anchor_weight,
            "anchor_batch_size": args.anchor_batch_size,
            "historical_query_repeat": args.historical_query_repeat if args.direction == "r2e" else None,
            "e2r_hard_candidates": args.e2r_hard_candidates if args.direction == "e2r" else None,
            "e2r_random_candidates": args.e2r_random_candidates if args.direction == "e2r" else None,
        },
        "checkpoint": str(target),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
