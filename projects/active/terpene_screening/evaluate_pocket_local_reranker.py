from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_dual_tower_protocol_comparison import (  # noqa: E402
    aggregate,
    masked_rank_metrics,
    ranked_candidate_rows,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    load_protein_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_GLOBAL_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_POCKET_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_pocket_local"
DEFAULT_POCKET_AUDIT = ROOT / "data/terpene_pocket_sequence/pocket_sequence_audit.csv"
DEFAULT_MOTIF_AUDIT = (
    ROOT / "data/terpene_embeddings/esmc600m_motif_context_combined/motif_context_audit.csv"
)
DEFAULT_STRICT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_EXACT_FOLDS = (
    ROOT
    / "projects/active/terpene_screening/comparison_assets/legacy_exact_reaction_folds.csv"
)
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_pocket_local_reranker"
DEFAULT_BUDGETS = (3, 5, 10, 20)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def load_aligned_embedding(directory: Path, identifiers: list[str]) -> np.ndarray:
    matrix, values = load_protein_features(directory)
    row_by_id = {value: index for index, value in enumerate(values)}
    missing = [value for value in identifiers if value not in row_by_id]
    if missing:
        raise ValueError(f"Embedding directory {directory} misses identifiers: {missing[:10]}")
    return np.stack([matrix[row_by_id[value]] for value in identifiers]).astype(np.float32)


def build_local_descriptors(
    protein_ids: list[str],
    pocket_audit_path: Path,
    motif_audit_path: Path,
) -> tuple[np.ndarray, list[str]]:
    pocket = pd.read_csv(pocket_audit_path, dtype=str).fillna("")
    motif = pd.read_csv(motif_audit_path, dtype=str).fillna("")
    table = pd.DataFrame({"Entry": protein_ids}).merge(
        pocket, on="Entry", how="left", validate="one_to_one"
    ).merge(motif, on="Entry", how="left", validate="one_to_one")
    columns = [
        "pocket_available",
        "valid_pocket_positions",
        "segment_count",
        "pocket_sequence_length",
        "coverage_fraction",
        "classI_pair_present",
        "classI_pair_distance",
        "ddxxd_count",
        "nse_dte_count",
        "dxdd_count",
        "qw_count",
        "sequence_length",
    ]
    values = np.zeros((len(table), len(columns)), dtype=np.float32)
    for index, column in enumerate(columns):
        raw = table[column] if column in table else pd.Series([""] * len(table))
        if column in {"pocket_available", "classI_pair_present"}:
            values[:, index] = raw.astype(str).str.lower().eq("true").astype(np.float32)
        else:
            values[:, index] = pd.to_numeric(raw, errors="coerce").fillna(0).astype(np.float32)
    # Stable scales; no learned statistics from a held-out split are required.
    scales = np.asarray(
        [1, 100, 25, 300, 1, 1, 250, 4, 4, 4, 6, 2000], dtype=np.float32
    )
    values = np.clip(values / scales, -4, 4)
    return values, columns


class PocketLocalTower(nn.Module):
    def __init__(
        self,
        reaction_dim: int,
        pocket_dim: int,
        descriptor_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.reaction = nn.Sequential(
            nn.LayerNorm(reaction_dim),
            nn.Linear(reaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.pocket = nn.Sequential(
            nn.LayerNorm(pocket_dim),
            nn.Linear(pocket_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.descriptor = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, max(32, embedding_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, embedding_dim // 4), embedding_dim),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(descriptor_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.reaction(values), p=2, dim=-1)

    def encode_proteins(
        self, pocket_values: torch.Tensor, descriptors: torch.Tensor
    ) -> torch.Tensor:
        pocket = self.pocket(pocket_values)
        descriptor = self.descriptor(descriptors)
        gate = self.local_gate(descriptors)
        return F.normalize(pocket + gate * descriptor, p=2, dim=-1)

    def score(
        self,
        reaction_values: torch.Tensor,
        pocket_values: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        reaction = self.encode_reactions(reaction_values)
        protein = self.encode_proteins(pocket_values, descriptors)
        return (reaction * protein).sum(dim=-1)


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int
    embedding_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    margin: float
    negatives_per_positive: int
    seed: int


def build_global_neighbor_order(global_features: np.ndarray) -> np.ndarray:
    normalized = global_features / np.maximum(
        np.linalg.norm(global_features, axis=1, keepdims=True), 1e-8
    )
    similarities = normalized @ normalized.T
    identifiers = np.arange(len(global_features))
    order = np.empty_like(similarities, dtype=np.int32)
    for index in range(len(global_features)):
        order[index] = np.lexsort((identifiers, -similarities[index])).astype(np.int32)
    return order


def build_pairwise_training_rows(
    train_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    neighbor_order: np.ndarray,
    protein_groups: dict[str, str],
    precursor_by_reaction: dict[str, str],
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pairs = train_pairs[["Entry", "rhea_id"]].drop_duplicates().copy()
    positive_by_reaction = {
        reaction: set(group["Entry"].astype(str))
        for reaction, group in train_pairs.groupby("rhea_id", sort=True)
    }
    reactions_by_protein = {
        protein: set(group["rhea_id"].astype(str))
        for protein, group in train_pairs.groupby("Entry", sort=True)
    }
    proteins_by_precursor: dict[str, set[str]] = {}
    for protein, reactions in reactions_by_protein.items():
        for precursor in {precursor_by_reaction.get(reaction, "") for reaction in reactions}:
            if precursor:
                proteins_by_precursor.setdefault(precursor, set()).add(protein)
    train_proteins = set(train_pairs["Entry"].astype(str))
    reaction_rows: list[int] = []
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    for pair in train_pairs.sort_values(["rhea_id", "Entry"]).itertuples(index=False):
        reaction = str(pair.rhea_id)
        positive = str(pair.Entry)
        if reaction not in reaction_to_row or positive not in protein_to_row:
            continue
        precursor = precursor_by_reaction.get(reaction, "")
        preferred = proteins_by_precursor.get(precursor, set()) & train_proteins
        positives = positive_by_reaction.get(reaction, set())
        positive_group = protein_groups.get(positive, positive)
        candidates: list[str] = []
        for candidate_row in neighbor_order[protein_to_row[positive]]:
            candidate = protein_ids[int(candidate_row)]
            if candidate == positive or candidate not in train_proteins or candidate in positives:
                continue
            if protein_groups.get(candidate, candidate) == positive_group:
                continue
            if preferred and candidate not in preferred:
                continue
            candidates.append(candidate)
            if len(candidates) >= negatives_per_positive:
                break
        if len(candidates) < negatives_per_positive:
            for candidate_row in neighbor_order[protein_to_row[positive]]:
                candidate = protein_ids[int(candidate_row)]
                if candidate == positive or candidate not in train_proteins or candidate in positives:
                    continue
                if protein_groups.get(candidate, candidate) == positive_group:
                    continue
                if candidate in candidates:
                    continue
                candidates.append(candidate)
                if len(candidates) >= negatives_per_positive:
                    break
        for negative in candidates:
            reaction_rows.append(reaction_to_row[reaction])
            positive_rows.append(protein_to_row[positive])
            negative_rows.append(protein_to_row[negative])
    if not reaction_rows:
        raise ValueError("No pairwise training triples were generated")
    return (
        np.asarray(reaction_rows, dtype=np.int64),
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
    )


def train_pocket_model(
    *,
    reaction_features: np.ndarray,
    pocket_features: np.ndarray,
    descriptors: np.ndarray,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[PocketLocalTower, list[dict[str, float]]]:
    seed_everything(config.seed)
    model = PocketLocalTower(
        reaction_dim=reaction_features.shape[1],
        pocket_dim=pocket_features.shape[1],
        descriptor_dim=descriptors.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    pocket_tensor = torch.as_tensor(pocket_features, dtype=torch.float32, device=device)
    descriptor_tensor = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
    reaction_rows, positive_rows, negative_rows = triples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = generator.permutation(len(reaction_rows))
        losses: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            selection = permutation[start : start + config.batch_size]
            r = torch.as_tensor(reaction_rows[selection], dtype=torch.long, device=device)
            p = torch.as_tensor(positive_rows[selection], dtype=torch.long, device=device)
            n = torch.as_tensor(negative_rows[selection], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores = model.score(
                reaction_tensor[r], pocket_tensor[p], descriptor_tensor[p]
            )
            negative_scores = model.score(
                reaction_tensor[r], pocket_tensor[n], descriptor_tensor[n]
            )
            pairwise = F.softplus(negative_scores - positive_scores + config.margin).mean()
            # Small norm-free separation term prevents an all-tie local expert.
            separation = F.softplus(-positive_scores).mean() + F.softplus(negative_scores).mean()
            loss = pairwise + 0.05 * separation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            history.append({"epoch": float(epoch), "mean_loss": mean_loss})
    if best_state is None:
        raise RuntimeError("Pocket model training produced no state")
    model.load_state_dict(best_state)
    return model, history


def score_all_candidates(
    model: PocketLocalTower,
    reaction_features: np.ndarray,
    pocket_features: np.ndarray,
    descriptors: np.ndarray,
    reaction_row: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        reaction = torch.as_tensor(
            reaction_features[reaction_row : reaction_row + 1],
            dtype=torch.float32,
            device=device,
        )
        pockets = torch.as_tensor(pocket_features, dtype=torch.float32, device=device)
        descriptor_tensor = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
        reaction_embedding = model.encode_reactions(reaction)
        protein_embeddings = model.encode_proteins(pockets, descriptor_tensor)
        return (reaction_embedding @ protein_embeddings.T).squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPS-specialized pocket-local pairwise reranker under exact and strict double-cold protocols."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--global-embedding-dir", type=Path, default=DEFAULT_GLOBAL_EMBEDDINGS)
    parser.add_argument("--pocket-embedding-dir", type=Path, default=DEFAULT_POCKET_EMBEDDINGS)
    parser.add_argument("--pocket-audit", type=Path, default=DEFAULT_POCKET_AUDIT)
    parser.add_argument("--motif-audit", type=Path, default=DEFAULT_MOTIF_AUDIT)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocols", default="legacy_exact,double_cold_25cell")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument(
        "--strict-partition",
        choices=["all", "development", "frozen"],
        default="all",
        help="development means protein_fold==4 or reaction_fold==4; frozen means both folds are 0..3",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocols = tuple(part.strip() for part in args.protocols.split(",") if part.strip())
    unknown = set(protocols) - {"legacy_exact", "double_cold_25cell"}
    if unknown:
        raise ValueError(f"Unknown protocols: {sorted(unknown)}")
    budgets = parse_int_tuple(args.budgets)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_matrix, protein_ids = load_protein_features(args.global_embedding_dir.resolve())
    pocket_matrix = load_aligned_embedding(args.pocket_embedding_dir.resolve(), protein_ids)
    descriptors, descriptor_columns = build_local_descriptors(
        protein_ids, args.pocket_audit.resolve(), args.motif_audit.resolve()
    )
    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    reaction_matrix, reaction_ids, reaction_table, feature_schema = build_reaction_features(
        positives, "multiview"
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    positives = positives[
        positives.Entry.isin(protein_to_row) & positives.rhea_id.isin(reaction_to_row)
    ].copy()

    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict["protein_fold"] = pd.to_numeric(strict["protein_fold"]).astype(int)
    strict["reaction_fold"] = pd.to_numeric(strict["reaction_fold"]).astype(int)
    strict = strict[
        [
            "Entry",
            "rhea_id",
            "protein_cluster",
            "reaction_cluster",
            "protein_fold",
            "reaction_fold",
        ]
    ].drop_duplicates(["Entry", "rhea_id"])
    pairs = positives[["Entry", "rhea_id"]].merge(
        strict, on=["Entry", "rhea_id"], how="left", validate="one_to_one"
    )
    if pairs[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict assignments do not cover every current positive pair")
    pairs[["protein_fold", "reaction_fold"]] = pairs[
        ["protein_fold", "reaction_fold"]
    ].astype(int)

    exact = pd.read_csv(args.exact_folds, dtype=str).fillna("")
    exact["legacy_exact_fold"] = pd.to_numeric(exact.legacy_exact_fold).astype(int)
    exact_fold_by_reaction = dict(zip(exact.reaction_id, exact.legacy_exact_fold))
    clusters = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_groups = dict(zip(clusters.entry.astype(str), clusters.cluster_id.astype(str)))
    protein_groups = {value: protein_groups.get(value, value) for value in protein_ids}
    precursor_by_reaction = dict(
        zip(reaction_table.rhea_id.astype(str), reaction_table.precursor_class.astype(str))
    )
    all_positive_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in pairs.groupby("rhea_id", sort=True)
    }
    neighbor_order = build_global_neighbor_order(global_matrix)
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )

    records: list[dict[str, object]] = []
    ranking_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []
    completed_cells = 0

    if "legacy_exact" in protocols:
        for fold in range(5):
            test_reactions = {
                reaction for reaction, local_fold in exact_fold_by_reaction.items() if local_fold == fold
            }
            train_pairs = pairs[~pairs.rhea_id.isin(test_reactions)].copy()
            triples = build_pairwise_training_rows(
                train_pairs,
                protein_ids,
                reaction_ids,
                protein_to_row,
                reaction_to_row,
                neighbor_order,
                protein_groups,
                precursor_by_reaction,
                args.negatives_per_positive,
            )
            model, history = train_pocket_model(
                reaction_features=reaction_matrix,
                pocket_features=pocket_matrix,
                descriptors=descriptors,
                triples=triples,
                config=config,
                device=device,
            )
            training_records.append(
                {
                    "protocol": "legacy_exact",
                    "protein_fold": "",
                    "reaction_fold": fold,
                    "n_train_pairs": len(train_pairs),
                    "n_training_triples": len(triples[0]),
                    "final_loss": history[-1]["mean_loss"],
                    "best_loss": min(row["mean_loss"] for row in history),
                }
            )
            for reaction in sorted(test_reactions):
                positives_for_query = all_positive_by_reaction.get(reaction, set())
                if not positives_for_query:
                    continue
                scores = score_all_candidates(
                    model,
                    reaction_matrix,
                    pocket_matrix,
                    descriptors,
                    reaction_to_row[reaction],
                    device,
                )
                records.append(
                    {
                        "protocol": "legacy_exact",
                        "protein_fold": "",
                        "reaction_fold": fold,
                        "reaction_id": reaction,
                        **masked_rank_metrics(
                            scores, protein_ids, positives_for_query, set(), budgets
                        ),
                    }
                )
                for rank, candidate, score in ranked_candidate_rows(
                    scores, protein_ids, set(), args.ranking_depth
                ):
                    ranking_records.append(
                        {
                            "protocol": "legacy_exact",
                            "protein_fold": "",
                            "reaction_fold": fold,
                            "reaction_id": reaction,
                            "candidate_id": candidate,
                            "rank": rank,
                            "score": score,
                        }
                    )

    if "double_cold_25cell" in protocols:
        stop = False
        for protein_fold in range(5):
            for reaction_fold in range(5):
                is_development = protein_fold == 4 or reaction_fold == 4
                if args.strict_partition == "development" and not is_development:
                    continue
                if args.strict_partition == "frozen" and is_development:
                    continue
                if args.max_cells and completed_cells >= args.max_cells:
                    stop = True
                    break
                train_pairs = pairs[
                    (pairs.protein_fold != protein_fold)
                    & (pairs.reaction_fold != reaction_fold)
                ].copy()
                test_pairs = pairs[
                    (pairs.protein_fold == protein_fold)
                    & (pairs.reaction_fold == reaction_fold)
                ].copy()
                if test_pairs.empty:
                    completed_cells += 1
                    continue
                triples = build_pairwise_training_rows(
                    train_pairs,
                    protein_ids,
                    reaction_ids,
                    protein_to_row,
                    reaction_to_row,
                    neighbor_order,
                    protein_groups,
                    precursor_by_reaction,
                    args.negatives_per_positive,
                )
                model, history = train_pocket_model(
                    reaction_features=reaction_matrix,
                    pocket_features=pocket_matrix,
                    descriptors=descriptors,
                    triples=triples,
                    config=config,
                    device=device,
                )
                training_records.append(
                    {
                        "protocol": "double_cold_25cell",
                        "protein_fold": protein_fold,
                        "reaction_fold": reaction_fold,
                        "n_train_pairs": len(train_pairs),
                        "n_training_triples": len(triples[0]),
                        "final_loss": history[-1]["mean_loss"],
                        "best_loss": min(row["mean_loss"] for row in history),
                    }
                )
                for reaction, group in test_pairs.groupby("rhea_id", sort=True):
                    positives_for_query = set(group.Entry.astype(str))
                    known_other = all_positive_by_reaction.get(reaction, set()) - positives_for_query
                    scores = score_all_candidates(
                        model,
                        reaction_matrix,
                        pocket_matrix,
                        descriptors,
                        reaction_to_row[reaction],
                        device,
                    )
                    records.append(
                        {
                            "protocol": "double_cold_25cell",
                            "protein_fold": protein_fold,
                            "reaction_fold": reaction_fold,
                            "reaction_id": reaction,
                            **masked_rank_metrics(
                                scores,
                                protein_ids,
                                positives_for_query,
                                known_other,
                                budgets,
                            ),
                        }
                    )
                    for rank, candidate, score in ranked_candidate_rows(
                        scores, protein_ids, known_other, args.ranking_depth
                    ):
                        ranking_records.append(
                            {
                                "protocol": "double_cold_25cell",
                                "protein_fold": protein_fold,
                                "reaction_fold": reaction_fold,
                                "reaction_id": reaction,
                                "candidate_id": candidate,
                                "rank": rank,
                                "score": score,
                            }
                        )
                completed_cells += 1
            if stop:
                break

    query_frame = pd.DataFrame(records)
    rankings = pd.DataFrame(ranking_records)
    training = pd.DataFrame(training_records)
    metrics = aggregate(query_frame, budgets) if not query_frame.empty else pd.DataFrame()
    query_frame.to_csv(output_dir / "query_metrics.csv", index=False)
    rankings.to_csv(output_dir / "rankings.csv", index=False)
    training.to_csv(output_dir / "training_summary.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "method": "tps_pocket_local_pairwise_dual_tower",
        "protocols": list(protocols),
        "budgets": list(budgets),
        "config": config.__dict__,
        "ranking_depth": args.ranking_depth,
        "strict_partition": args.strict_partition,
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "n_positive_pairs": len(pairs),
        "descriptor_columns": descriptor_columns,
        "reaction_feature_schema": feature_schema,
        "negative_mining": {
            "primary": "nearest global ESM-C proteins associated with the same precursor class",
            "false_negative_control": "exclude known positives and the positive protein's 50% identity cluster",
            "fallback": "nearest labeled training proteins outside the positive cluster",
        },
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "rankings": str(output_dir / "rankings.csv"),
            "training_summary": str(output_dir / "training_summary.csv"),
            "metrics": str(output_dir / "metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
