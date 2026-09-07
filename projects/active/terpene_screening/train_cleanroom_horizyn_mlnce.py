from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from horizyn.losses import FullBatchMLNCELoss  # noqa: E402
from projects.active.terpene_screening.fair_benchmark import sha256_file  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.train_cleanroom_rhea_retriever import (  # noqa: E402
    DEFAULT_ASSOCIATIONS,
    DEFAULT_SCHEMA_DIR,
    DEFAULT_UNIVERSE,
    _tensor_rows,
    build_author_like_dev_reservoir,
    build_reaction_neighbors,
    evaluate_dev,
    positive_maps,
    score_reservoir,
    split_double_cold,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    seed_everything,
)


def _unique_with_inverse(values: list[str]) -> tuple[list[str], torch.Tensor]:
    mapping: dict[str, int] = {}
    unique: list[str] = []
    inverse: list[int] = []
    for value in values:
        index = mapping.get(value)
        if index is None:
            index = len(unique)
            mapping[value] = index
            unique.append(value)
        inverse.append(index)
    return unique, torch.as_tensor(inverse, dtype=torch.long)


def train_mlnce(
    protein_features: np.ndarray,
    protein_ids: list[str],
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    train_pairs: pd.DataFrame,
    *,
    config: ModelConfig,
    epochs: int,
    batch_pairs: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    seed: int,
    device: torch.device,
) -> tuple[TerpeneDualTower, list[dict[str, float]]]:
    if epochs <= 0 or batch_pairs <= 0 or beta <= 0:
        raise ValueError("epochs, batch_pairs, and beta must be positive")
    seed_everything(seed)
    rng = random.Random(seed)
    pindex = {value: i for i, value in enumerate(protein_ids)}
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    pairs = train_pairs[
        train_pairs["protein_id"].isin(pindex) & train_pairs["reaction_id"].isin(rindex)
    ][["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)
    if pairs.empty:
        raise ValueError("no clean training pairs remain")

    model = TerpeneDualTower(config).to(device)
    loss_fn = FullBatchMLNCELoss(beta=beta, learn_beta=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    order = list(range(len(pairs)))
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        losses: list[float] = []
        pair_counts: list[int] = []
        query_counts: list[int] = []
        target_counts: list[int] = []
        model.train()
        for start in range(0, len(order), batch_pairs):
            batch = pairs.iloc[order[start : start + batch_pairs]]
            if batch.empty:
                continue
            reactions = batch["reaction_id"].astype(str).tolist()
            proteins = batch["protein_id"].astype(str).tolist()
            unique_r, q_inverse_cpu = _unique_with_inverse(reactions)
            unique_p, t_inverse_cpu = _unique_with_inverse(proteins)
            q = model.encode_reactions(_tensor_rows(reaction_features, unique_r, rindex, device))
            t = model.encode_proteins(_tensor_rows(protein_features, unique_p, pindex, device))
            dists = 1.0 - q @ t.T
            loss = loss_fn(
                dists,
                q_inverse_cpu.to(device=device),
                t_inverse_cpu.to(device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pair_counts.append(int(len(batch)))
            query_counts.append(int(len(unique_r)))
            target_counts.append(int(len(unique_p)))
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            "steps": float(len(losses)),
            "mean_pairs_per_step": float(np.mean(pair_counts)),
            "mean_unique_reactions": float(np.mean(query_counts)),
            "mean_unique_proteins": float(np.mean(target_counts)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-clean training using Horizyn's official FullBatchMLNCELoss.")
    parser.add_argument("--associations-csv", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-pairs", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--embedding-dim", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--dev-neighbor-reactions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    universe = args.universe_dir.resolve(); schema_dir = args.schema_dir.resolve()
    schema = load_feature_schema(schema_dir)
    protein_features, protein_ids = load_protein_library(universe / "proteins")
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe / "reaction_features/drfp_categorical_v1", schema
    )
    pset, rset = set(protein_ids), set(reaction_ids)
    source = args.associations_csv.resolve()
    pairs = pd.read_csv(source, dtype=str).fillna("")
    pairs = pairs[pairs["protein_id"].isin(pset) & pairs["reaction_id"].isin(rset)]
    pairs = pairs[["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)
    train_pairs, dev_pairs = split_double_cold(pairs, dev_fold=args.dev_fold, folds=args.folds)
    config = ModelConfig(
        protein_input_dim=int(protein_features.shape[1]), reaction_input_dim=int(reaction_features.shape[1]),
        hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim, dropout=args.dropout,
    )
    model, history = train_mlnce(
        protein_features, protein_ids, reaction_features, reaction_ids, train_pairs,
        config=config, epochs=args.epochs, batch_pairs=args.batch_pairs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, beta=args.beta,
        seed=args.seed, device=device,
    )

    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / f"production_seed{args.seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "model_config": asdict(config), "model_type": "dual_tower",
        "seed": args.seed, "cleanroom_random_init": True, "base_checkpoint": None,
        "training_source": str(source), "training_source_sha256": sha256_file(source),
        "dev_fold": args.dev_fold, "folds": args.folds,
        "loss_provenance": str((HORIZYN_ROOT / 'horizyn/losses.py').resolve()),
    }, checkpoint)
    shutil.copy2(schema_dir / "feature_schema.json", output / "feature_schema.json")
    train_pairs.to_csv(output / "training_pairs.csv", index=False)
    dev_pairs.to_csv(output / "dev_pairs.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

    train_by_reaction, _ = positive_maps(train_pairs)
    neighbor_queries = set(dev_pairs["reaction_id"].astype(str))
    neighbors = build_reaction_neighbors(
        reaction_features, reaction_ids, train_reactions=set(train_by_reaction), query_reactions=neighbor_queries,
        topk=args.dev_neighbor_reactions, device=device,
    )
    reservoir = build_author_like_dev_reservoir(
        dev_pairs, neighbors=neighbors, train_proteins_by_reaction=train_by_reaction,
        neighbor_reactions=args.dev_neighbor_reactions,
    )
    scored = score_reservoir(
        model, reservoir, protein_features, protein_ids, reaction_features, reaction_ids, device=device,
    )
    dev_metrics, query_metrics = evaluate_dev(scored)
    scored.to_csv(output / "dev_pair_scores.csv", index=False)
    query_metrics.to_csv(output / "dev_query_metrics.csv", index=False)
    summary = {
        "method": "cleanroom_random_init_horizyn_official_fullbatch_mlnce",
        "external_code_reused": "dayhofflabs/horizyn FullBatchMLNCELoss",
        "target_benchmark_labels_read": False,
        "target_benchmark_metadata_used_for_training": False,
        "association_source": str(source), "association_source_sha256": sha256_file(source),
        "dev_protocol": "strict protein+reaction hash double-cold inside 2023 snapshot",
        "dev_fold": args.dev_fold, "folds": args.folds,
        "n_source_pairs": int(len(pairs)), "n_train_pairs": int(len(train_pairs)), "n_dev_pairs": int(len(dev_pairs)),
        "model_config": asdict(config),
        "training": {
            "epochs": args.epochs, "batch_pairs": args.batch_pairs, "beta": args.beta,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "gradient_clip_norm": 1.0,
            "batch_semantics": "official Horizyn-style positive-pair batch; deduplicate entities; all query-target combinations enter MLNCE partition",
            "feature_change_from_cleanroom_baseline": False,
        },
        "dev_metrics": dev_metrics, "checkpoint": str(checkpoint),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
