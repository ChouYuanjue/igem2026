from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.build_cold_splits import (  # noqa: E402
    butina_clusters,
    product_fingerprint,
)
from projects.active.terpene_screening.gate_matrix import precursor_class_from_reaction  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_reaction,
    load_feature_schema,
    load_protein_library,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    load_aligned_feature_augmentation,
    rank_metrics,
    train_model,
)

DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_EXTERNAL_PROTEINS = ROOT / "data/terpene_embeddings/marts_unseen_esmc600m"
DEFAULT_CURRENT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_PRODUCTION_DIR = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_domain_adaptation"
DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_MMSEQS = ROOT / "data/assets/mmseqs2/mmseqs/bin/mmseqs"
DEFAULT_BUDGETS = (3, 10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]}"


def load_embedding_map(directory: Path) -> dict[str, np.ndarray]:
    matrix, ids = load_protein_library(directory)
    return {value: matrix[index] for index, value in enumerate(ids)}


def build_protein_entities(
    marts: pd.DataFrame,
    current_dir: Path,
    external_dir: Path,
    current_candidates_path: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, str]]:
    current = load_embedding_map(current_dir)
    external = load_embedding_map(external_dir)
    candidates = pd.read_csv(current_candidates_path, sep="\t", dtype=str).fillna("")
    candidates["normalized_sequence"] = (
        candidates["Sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    )
    current_by_sequence: dict[str, str] = {}
    for row in candidates[["Entry", "normalized_sequence"]].itertuples(index=False):
        if row.Entry in current and row.normalized_sequence not in current_by_sequence:
            current_by_sequence[row.normalized_sequence] = row.Entry

    unique = marts[["enzyme_id", "sequence", "enzyme_seen"]].drop_duplicates(["enzyme_id", "sequence"]).copy()
    unique["sequence"] = unique["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    unique["enzyme_seen"] = unique["enzyme_seen"].astype(str).str.lower().eq("true")
    enzyme_to_entity: dict[str, str] = {}
    feature_by_entity: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, object]] = {}
    for row in unique.itertuples(index=False):
        if not row.sequence:
            continue
        entity_id = stable_id("MARTS_SEQ", row.sequence)
        enzyme_to_entity[str(row.enzyme_id)] = entity_id
        feature = current.get(str(row.enzyme_id))
        source = "current_id"
        if feature is None:
            feature = external.get(str(row.enzyme_id))
            source = "external_id"
        if feature is None:
            matching_entry = current_by_sequence.get(row.sequence)
            feature = current.get(matching_entry) if matching_entry else None
            source = "current_sequence"
        if feature is None:
            continue
        feature_by_entity.setdefault(entity_id, feature)
        previous = rows.get(entity_id)
        seen = bool(row.enzyme_seen) or bool(previous and previous["enzyme_seen"])
        aliases = set(previous["aliases"].split(";")) if previous else set()
        aliases.add(str(row.enzyme_id))
        rows[entity_id] = {
            "protein_id": entity_id,
            "sequence": row.sequence,
            "enzyme_seen": seen,
            "aliases": ";".join(sorted(value for value in aliases if value)),
            "feature_source": source if previous is None else previous["feature_source"],
        }
    protein_ids = sorted(feature_by_entity)
    matrix = normalize_rows(np.stack([feature_by_entity[value] for value in protein_ids]).astype(np.float32))
    table = pd.DataFrame([rows[value] for value in protein_ids])
    return matrix, protein_ids, table, enzyme_to_entity


def build_reaction_entities(
    marts: pd.DataFrame,
    schema: dict[str, object],
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, str]]:
    valid = marts[marts["reaction_signature"].astype(str) != ""].copy()
    representative = valid.drop_duplicates("reaction_signature").set_index("reaction_signature")
    signatures = sorted(representative.index.astype(str))
    signature_to_entity = {value: stable_id("MARTS_RXN", value) for value in signatures}
    features = np.stack(
        [encode_reaction(str(representative.loc[value, "reaction_smiles"]), schema) for value in signatures]
    ).astype(np.float32)
    rows = []
    for signature in signatures:
        group = valid[valid["reaction_signature"].eq(signature)]
        rows.append(
            {
                "reaction_id": signature_to_entity[signature],
                "reaction_signature": signature,
                "reaction_smiles": str(representative.loc[signature, "reaction_smiles"]),
                "reaction_seen": bool(group["reaction_seen"].astype(str).str.lower().eq("true").any()),
            }
        )
    return features, [signature_to_entity[value] for value in signatures], pd.DataFrame(rows), signature_to_entity


def build_pair_table(
    marts: pd.DataFrame,
    enzyme_to_entity: dict[str, str],
    signature_to_entity: dict[str, str],
    protein_seen: dict[str, bool],
    reaction_seen: dict[str, bool],
) -> pd.DataFrame:
    rows = []
    for row in marts[["enzyme_id", "reaction_signature"]].itertuples(index=False):
        protein_id = enzyme_to_entity.get(str(row.enzyme_id))
        reaction_id = signature_to_entity.get(str(row.reaction_signature))
        if protein_id and reaction_id:
            rows.append(
                {
                    "Entry": protein_id,
                    "rhea_id": reaction_id,
                    "protein_seen": bool(protein_seen[protein_id]),
                    "reaction_seen": bool(reaction_seen[reaction_id]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["rhea_id", "Entry"]).reset_index(drop=True)


def build_mmseqs_clusters(
    protein_table: pd.DataFrame,
    output_dir: Path,
    mmseqs: Path,
    min_identity: float,
    threads: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_csv = output_dir / f"protein_clusters_id{int(round(min_identity * 100))}.csv"
    if cluster_csv.exists():
        frame = pd.read_csv(cluster_csv, dtype=str)
        return dict(zip(frame["protein_id"], frame["cluster_id"]))
    fasta = output_dir / "marts_sequences.fasta"
    with fasta.open("w", encoding="utf-8") as handle:
        for row in protein_table[["protein_id", "sequence"]].itertuples(index=False):
            handle.write(f">{row.protein_id}\n{row.sequence}\n")
    prefix = output_dir / f"mmseqs_id{int(round(min_identity * 100))}"
    temporary = output_dir / f"tmp_id{int(round(min_identity * 100))}"
    shutil.rmtree(temporary, ignore_errors=True)
    subprocess.run(
        [
            str(mmseqs), "easy-cluster", str(fasta), str(prefix), str(temporary),
            "--min-seq-id", str(min_identity), "-c", "0.8", "--cov-mode", "2",
            "--threads", str(threads), "--remove-tmp-files", "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw = pd.read_csv(f"{prefix}_cluster.tsv", sep="\t", header=None, names=["representative", "protein_id"], dtype=str)
    mapping = dict(zip(raw["protein_id"], raw["representative"]))
    frame = protein_table[["protein_id"]].copy()
    frame["cluster_id"] = frame["protein_id"].map(mapping).fillna(frame["protein_id"])
    frame.to_csv(cluster_csv, index=False)
    return dict(zip(frame["protein_id"], frame["cluster_id"]))


def build_reaction_clusters(
    reaction_table: pd.DataFrame,
    output_dir: Path,
    threshold: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_csv = output_dir / f"reaction_clusters_butina_t{threshold:.2f}.csv"
    if cluster_csv.exists():
        frame = pd.read_csv(cluster_csv, dtype=str)
        return dict(zip(frame["reaction_id"], frame["cluster_id"]))

    working = reaction_table[["reaction_id", "reaction_smiles"]].copy()
    products = working["reaction_smiles"].map(product_fingerprint)
    working["fingerprint"] = products.map(lambda value: value[0])
    working["precursor_class"] = working["reaction_smiles"].map(precursor_class_from_reaction)
    assignments: dict[str, str] = {}
    for precursor_class, group in working.groupby("precursor_class", sort=True):
        ids = group["reaction_id"].astype(str).tolist()
        fingerprints = group["fingerprint"].tolist()
        local = butina_clusters(ids, fingerprints, threshold)
        for reaction_id, cluster_id in local.items():
            assignments[reaction_id] = f"{precursor_class}::{cluster_id}"
    frame = reaction_table[["reaction_id"]].copy()
    frame["cluster_id"] = frame["reaction_id"].map(assignments).fillna(frame["reaction_id"])
    frame.to_csv(cluster_csv, index=False)
    return dict(zip(frame["reaction_id"], frame["cluster_id"]))


def assign_folds(
    cluster_values: pd.Series,
    pair_weights: pd.Series,
    n_folds: int,
    seed: int = 0,
) -> dict[str, int]:
    frame = pd.DataFrame(
        {"cluster": cluster_values.astype(str), "weight": pair_weights.astype(float)}
    )
    if seed == 0:
        legacy_weights = (
            frame.groupby("cluster")["weight"].sum().sort_values(ascending=False)
        )
        ordered = [(str(cluster_id), float(weight)) for cluster_id, weight in legacy_weights.items()]
    else:
        weights = frame.groupby("cluster")["weight"].sum().reset_index()
        rng = np.random.default_rng(seed)
        weights["tie_break"] = rng.random(len(weights))
        weights = weights.sort_values(
            ["weight", "tie_break", "cluster"], ascending=[False, True, True]
        )
        ordered = [
            (str(row.cluster), float(row.weight)) for row in weights.itertuples(index=False)
        ]
    loads = [0.0] * n_folds
    assignment: dict[str, int] = {}
    for cluster_id, weight in ordered:
        fold = min(range(n_folds), key=lambda value: (loads[value], value))
        assignment[cluster_id] = fold
        loads[fold] += weight
    return assignment


def encode_models(
    models: list[TerpeneDualTower],
    protein_tensor: torch.Tensor,
    reaction_tensor: torch.Tensor,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    proteins, reactions = [], []
    with torch.no_grad():
        for model in models:
            model.eval()
            proteins.append(model.encode_proteins(protein_tensor).cpu().numpy())
            reactions.append(model.encode_reactions(reaction_tensor).cpu().numpy())
    return proteins, reactions


def ensemble_scores(
    protein_embeddings: list[np.ndarray],
    reaction_embeddings: list[np.ndarray],
) -> np.ndarray:
    total = np.zeros((len(reaction_embeddings[0]), len(protein_embeddings[0])), dtype=np.float32)
    for proteins, reactions in zip(protein_embeddings, reaction_embeddings):
        total += reactions @ proteins.T
    return total / len(protein_embeddings)


def load_production_payloads(production_dir: Path, device: torch.device) -> list[dict[str, object]]:
    paths = sorted((production_dir / "models").glob("production_seed*.pt"))
    if not paths:
        raise FileNotFoundError(f"No production checkpoints under {production_dir / 'models'}")
    return [torch.load(path, map_location=device, weights_only=False) for path in paths]


def evaluate_fold(
    records: list[dict[str, object]],
    method: str,
    fold: int,
    scores: np.ndarray,
    test_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    budgets: tuple[int, ...],
) -> None:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
        positives = set(group["Entry"].astype(str))
        metrics = rank_metrics(scores[reaction_to_row[reaction_id]], protein_ids, positives, set(), budgets)
        records.append({"method": method, "fold": fold, "direction": "reaction_to_enzyme", "query_id": reaction_id, **metrics})
    for protein_id, group in test_pairs.groupby("Entry", sort=True):
        positives = set(group["rhea_id"].astype(str))
        metrics = rank_metrics(scores[:, protein_to_row[protein_id]], reaction_ids, positives, set(), budgets)
        records.append({"method": method, "fold": fold, "direction": "enzyme_to_reaction", "query_id": protein_id, **metrics})


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (f"positive_recall_at_{budget}", "mean")
    return frame.groupby(["method", "direction"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="MARTS domain adaptation with protein/reaction double-cold folds.")
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--external-protein-dir", type=Path, default=DEFAULT_EXTERNAL_PROTEINS)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument("--reaction-augmentation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--protein-identity", type=float, default=0.5)
    parser.add_argument("--reaction-threshold", type=float, default=0.5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-mode", choices=["paired", "cartesian"], default="cartesian")
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=0,
        help="Nonzero seed randomizes equal-weight cluster ordering for a confirmatory split.",
    )
    parser.add_argument("--pu-group-mask", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--freeze-protein-tower", action="store_true")
    parser.add_argument("--freeze-reaction-tower", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--reaction-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--loss-mode",
        choices=["bidirectional_infonce", "global_mlnce"],
        default="bidirectional_infonce",
    )
    parser.add_argument("--hard-negative-k", type=int, default=0)
    parser.add_argument("--hard-negative-start-epoch", type=int, default=1)
    parser.add_argument(
        "--hard-negative-end-epoch",
        type=int,
        default=0,
        help="Zero keeps hard negatives active through the final epoch.",
    )
    parser.add_argument(
        "--model-selection",
        choices=["min_loss", "final"],
        default="min_loss",
    )
    parser.add_argument("--topk-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--topk-surrogate-k", type=int, default=10)
    parser.add_argument("--topk-surrogate-margin", type=float, default=0.0)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    marts = pd.read_csv(args.marts, sep="\t", dtype=str).fillna("")
    schema = load_feature_schema(args.production_dir.resolve())
    protein_matrix, protein_ids, protein_table, enzyme_to_entity = build_protein_entities(
        marts,
        args.current_protein_dir.resolve(),
        args.external_protein_dir.resolve(),
        args.current_candidates.resolve(),
    )
    reaction_matrix, reaction_ids, reaction_table, signature_to_entity = build_reaction_entities(marts, schema)
    if args.reaction_augmentation_dir is not None:
        augmentation = load_aligned_feature_augmentation(
            args.reaction_augmentation_dir.resolve(), reaction_ids
        )
        reaction_matrix = np.concatenate([reaction_matrix, augmentation], axis=1)
    protein_seen = dict(zip(protein_table["protein_id"], protein_table["enzyme_seen"].astype(bool)))
    reaction_seen = dict(zip(reaction_table["reaction_id"], reaction_table["reaction_seen"].astype(bool)))
    pairs = build_pair_table(marts, enzyme_to_entity, signature_to_entity, protein_seen, reaction_seen)

    protein_clusters = build_mmseqs_clusters(
        protein_table,
        cache_dir,
        args.mmseqs.resolve(),
        args.protein_identity,
        args.threads,
    )
    reaction_clusters = build_reaction_clusters(reaction_table, cache_dir, args.reaction_threshold)
    pairs["protein_cluster"] = pairs["Entry"].map(protein_clusters)
    pairs["reaction_cluster"] = pairs["rhea_id"].map(reaction_clusters)
    protein_fold = assign_folds(
        pairs["protein_cluster"],
        pd.Series(np.ones(len(pairs))),
        args.n_folds,
        args.fold_seed,
    )
    reaction_fold = assign_folds(
        pairs["reaction_cluster"],
        pd.Series(np.ones(len(pairs))),
        args.n_folds,
        args.fold_seed + 1 if args.fold_seed else 0,
    )
    pairs["protein_fold"] = pairs["protein_cluster"].map(protein_fold).astype(int)
    pairs["reaction_fold"] = pairs["reaction_cluster"].map(reaction_fold).astype(int)
    pairs.to_csv(cache_dir / "marts_pair_folds.csv", index=False)
    protein_table.assign(cluster_id=protein_table["protein_id"].map(protein_clusters)).to_csv(cache_dir / "protein_entities.csv", index=False)
    reaction_table.assign(cluster_id=reaction_table["reaction_id"].map(reaction_clusters)).to_csv(cache_dir / "reaction_entities.csv", index=False)
    np.save(cache_dir / "protein_features.npy", protein_matrix)
    np.save(cache_dir / "reaction_features.npy", reaction_matrix)

    payloads = load_production_payloads(args.production_dir.resolve(), device)
    config = ModelConfig(**payloads[0]["model_config"])
    if config.protein_input_dim != protein_matrix.shape[1] or config.reaction_input_dim != reaction_matrix.shape[1]:
        raise ValueError("Production checkpoint dimensions do not match MARTS features")
    protein_tensor = torch.as_tensor(protein_matrix, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_matrix, dtype=torch.float32, device=device)
    anchor_protein_rows = np.flatnonzero(protein_table["enzyme_seen"].astype(bool).to_numpy()).astype(np.int64)
    anchor_reaction_rows = np.flatnonzero(reaction_table["reaction_seen"].astype(bool).to_numpy()).astype(np.int64)
    baseline_models: list[TerpeneDualTower] = []
    anchor_targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    for payload in payloads:
        model = TerpeneDualTower(config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        baseline_models.append(model)
        with torch.no_grad():
            protein_target = model.encode_proteins(
                protein_tensor[torch.as_tensor(anchor_protein_rows, dtype=torch.long, device=device)]
            ).detach()
            reaction_target = model.encode_reactions(
                reaction_tensor[torch.as_tensor(anchor_reaction_rows, dtype=torch.long, device=device)]
            ).detach()
        anchor_targets.append((protein_target, reaction_target))
    baseline_proteins, baseline_reactions = encode_models(baseline_models, protein_tensor, reaction_tensor)
    baseline_score_matrix = ensemble_scores(baseline_proteins, baseline_reactions)

    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    histories: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []

    if args.fold_mode == "paired":
        split_specs = [(fold, fold) for fold in range(args.n_folds)]
    else:
        split_specs = [
            (protein_test_fold, reaction_test_fold)
            for protein_test_fold in range(args.n_folds)
            for reaction_test_fold in range(args.n_folds)
        ]

    for split_index, (protein_test_fold, reaction_test_fold) in enumerate(split_specs):
        split_id = f"p{protein_test_fold}_r{reaction_test_fold}"
        train_pairs = pairs[
            (pairs["protein_fold"] != protein_test_fold)
            & (pairs["reaction_fold"] != reaction_test_fold)
        ].copy()
        test_pairs = pairs[
            (pairs["protein_fold"] == protein_test_fold)
            & (pairs["reaction_fold"] == reaction_test_fold)
            & (~pairs["protein_seen"])
            & (~pairs["reaction_seen"])
        ].copy()
        split_rows.append(
            {
                "split_id": split_id,
                "protein_test_fold": protein_test_fold,
                "reaction_test_fold": reaction_test_fold,
                "train_pairs": len(train_pairs),
                "external_external_test_pairs": len(test_pairs),
                "test_proteins": test_pairs["Entry"].nunique(),
                "test_reactions": test_pairs["rhea_id"].nunique(),
            }
        )
        if test_pairs.empty:
            continue
        evaluate_fold(records, "current_production", split_id, baseline_score_matrix, test_pairs, protein_ids, reaction_ids, budgets)

        adapted_models: list[TerpeneDualTower] = []
        for payload_index, payload in enumerate(payloads):
            seed = int(payload.get("seed", 20260723 + payload_index)) + split_index * 1000
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
                initial_state_dict=payload["model_state_dict"],
                protein_group_map=protein_clusters,
                reaction_group_map=reaction_clusters,
                exclude_same_group_negatives=args.pu_group_mask,
                anchor_protein_rows=anchor_protein_rows,
                anchor_protein_targets=anchor_targets[payload_index][0],
                anchor_reaction_rows=anchor_reaction_rows,
                anchor_reaction_targets=anchor_targets[payload_index][1],
                anchor_weight=args.anchor_weight,
                freeze_protein_tower=args.freeze_protein_tower,
                freeze_reaction_tower=args.freeze_reaction_tower,
                reaction_loss_weight=args.reaction_loss_weight,
                loss_mode=args.loss_mode,
                hard_negative_k=args.hard_negative_k,
                hard_negative_start_epoch=args.hard_negative_start_epoch,
                hard_negative_end_epoch=args.hard_negative_end_epoch,
                model_selection=args.model_selection,
                topk_surrogate_weight=args.topk_surrogate_weight,
                topk_surrogate_k=args.topk_surrogate_k,
                topk_surrogate_margin=args.topk_surrogate_margin,
            )
            adapted_models.append(model)
            history_frame = pd.DataFrame(history)
            history_frame.insert(0, "checkpoint_index", payload_index)
            history_frame.insert(0, "split_id", split_id)
            histories.append(history_frame)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "split_id": split_id,
                    "protein_test_fold": protein_test_fold,
                    "reaction_test_fold": reaction_test_fold,
                    "source_seed": payload.get("seed"),
                    "loss_mode": args.loss_mode,
                    "hard_negative_k": args.hard_negative_k,
                    "hard_negative_start_epoch": args.hard_negative_start_epoch,
                    "hard_negative_end_epoch": args.hard_negative_end_epoch,
                    "model_selection": args.model_selection,
                    "topk_surrogate_weight": args.topk_surrogate_weight,
                    "topk_surrogate_k": args.topk_surrogate_k,
                    "topk_surrogate_margin": args.topk_surrogate_margin,
                    "feature_schema": schema,
                },
                model_dir / f"adapted_{split_id}_model{payload_index}.pt",
            )
        adapted_proteins, adapted_reactions = encode_models(adapted_models, protein_tensor, reaction_tensor)
        adapted_score_matrix = ensemble_scores(adapted_proteins, adapted_reactions)
        evaluate_fold(records, "marts_adapted", split_id, adapted_score_matrix, test_pairs, protein_ids, reaction_ids, budgets)

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)
    summary = {
        "production_dir": str(args.production_dir.resolve()),
        "feature_mode": schema.get("feature_mode"),
        "reaction_augmentation_dir": (
            str(args.reaction_augmentation_dir.resolve())
            if args.reaction_augmentation_dir is not None
            else None
        ),
        "reaction_augmentation_dimension": (
            int(augmentation.shape[1]) if args.reaction_augmentation_dir is not None else 0
        ),
        "n_protein_entities": len(protein_ids),
        "n_reaction_entities": len(reaction_ids),
        "n_pairs": len(pairs),
        "n_protein_clusters": len(set(protein_clusters.values())),
        "n_reaction_clusters": len(set(reaction_clusters.values())),
        "protein_identity": args.protein_identity,
        "reaction_threshold": args.reaction_threshold,
        "n_folds": args.n_folds,
        "fold_mode": args.fold_mode,
        "fold_seed": args.fold_seed,
        "pu_group_mask": args.pu_group_mask,
        "anchor_weight": args.anchor_weight,
        "n_anchor_proteins": int(len(anchor_protein_rows)),
        "n_anchor_reactions": int(len(anchor_reaction_rows)),
        "freeze_protein_tower": args.freeze_protein_tower,
        "freeze_reaction_tower": args.freeze_reaction_tower,
        "reaction_loss_weight": args.reaction_loss_weight,
        "loss_mode": args.loss_mode,
        "hard_negative_k": args.hard_negative_k,
        "hard_negative_start_epoch": args.hard_negative_start_epoch,
        "hard_negative_end_epoch": args.hard_negative_end_epoch,
        "model_selection": args.model_selection,
        "topk_surrogate_weight": args.topk_surrogate_weight,
        "topk_surrogate_k": args.topk_surrogate_k,
        "topk_surrogate_margin": args.topk_surrogate_margin,
        "n_split_specs": len(split_specs),
        "epochs": args.epochs,
        "budgets": budgets,
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "split_summary": str(output_dir / "split_summary.csv"),
            "models": str(model_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(split_rows).to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
