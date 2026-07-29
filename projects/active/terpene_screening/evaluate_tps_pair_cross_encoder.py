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
from projects.active.terpene_screening.evaluate_motif_channel_reranker import (  # noqa: E402
    load_motif_blocks,
)
from projects.active.terpene_screening.evaluate_pocket_local_reranker import (  # noqa: E402
    build_global_neighbor_order,
    build_local_descriptors,
    load_aligned_embedding,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    load_protein_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_GLOBAL = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_MOTIF = ROOT / "data/terpene_embeddings/esmc600m_motif_context_combined"
DEFAULT_POCKET = ROOT / "data/terpene_embeddings/esmc600m_pocket_local"
DEFAULT_POCKET_AUDIT = ROOT / "data/terpene_pocket_sequence/pocket_sequence_audit.csv"
DEFAULT_MOTIF_AUDIT = DEFAULT_MOTIF / "motif_context_audit.csv"
DEFAULT_STRICT = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_tps_pair_cross_encoder"
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


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), p=2, dim=-1)


class TPSPairCrossEncoder(nn.Module):
    """Pair-level TPS scorer over reaction, global, motif, and pocket views."""

    def __init__(
        self,
        reaction_dim: int,
        global_dim: int,
        motif_dim: int,
        pocket_dim: int,
        descriptor_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.reaction_projection = Projection(
            reaction_dim, hidden_dim, latent_dim, dropout
        )
        self.global_projection = Projection(global_dim, hidden_dim, latent_dim, dropout)
        self.pocket_projection = Projection(pocket_dim, hidden_dim, latent_dim, dropout)
        self.motif_projections = nn.ModuleList(
            [Projection(motif_dim, hidden_dim, latent_dim, dropout) for _ in range(4)]
        )
        self.motif_gate = nn.Sequential(
            nn.LayerNorm(reaction_dim),
            nn.Linear(reaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        pair_dim = latent_dim * 10 + descriptor_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_pairs(
        self,
        reaction_values: torch.Tensor,
        global_values: torch.Tensor,
        motif_values: torch.Tensor,
        pocket_values: torch.Tensor,
        descriptors: torch.Tensor,
        motif_presence: torch.Tensor,
    ) -> torch.Tensor:
        reaction = self.reaction_projection(reaction_values)
        global_protein = self.global_projection(global_values)
        pocket = self.pocket_projection(pocket_values)
        pocket_available = descriptors[:, :1].clamp(0, 1)
        pocket = pocket_available * pocket + (1 - pocket_available) * global_protein

        motif_channels = torch.stack(
            [
                projection(motif_values[:, index, :])
                for index, projection in enumerate(self.motif_projections)
            ],
            dim=1,
        )
        motif_weights = torch.softmax(self.motif_gate(reaction_values), dim=-1)
        motif_weights = motif_weights * motif_presence
        motif_denominator = motif_weights.sum(dim=-1, keepdim=True)
        motif_weights = motif_weights / motif_denominator.clamp_min(1e-8)
        motif = (motif_weights[:, :, None] * motif_channels).sum(dim=1)
        no_motif = motif_denominator.squeeze(-1).eq(0)
        if no_motif.any():
            motif = motif.clone()
            motif[no_motif] = global_protein[no_motif]

        pair_features = torch.cat(
            [
                reaction,
                global_protein,
                motif,
                pocket,
                reaction * global_protein,
                reaction * motif,
                reaction * pocket,
                torch.abs(reaction - global_protein),
                torch.abs(reaction - motif),
                torch.abs(reaction - pocket),
                descriptors,
            ],
            dim=-1,
        )
        return self.scorer(pair_features).squeeze(-1)


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int
    latent_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    margin: float
    negatives_per_positive: int
    seed: int


def build_specialized_triples(
    train_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_to_row: dict[str, int],
    reaction_to_row: dict[str, int],
    neighbor_order: np.ndarray,
    protein_groups: dict[str, str],
    precursor_by_reaction: dict[str, str],
    skeleton_by_reaction: dict[str, str],
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = train_pairs[["Entry", "rhea_id"]].drop_duplicates().copy()
    positive_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in working.groupby("rhea_id", sort=True)
    }
    reactions_by_protein = {
        protein: set(group.rhea_id.astype(str))
        for protein, group in working.groupby("Entry", sort=True)
    }
    train_proteins = set(working.Entry.astype(str))
    reaction_rows: list[int] = []
    positive_rows: list[int] = []
    negative_rows: list[int] = []

    for row in working.sort_values(["rhea_id", "Entry"]).itertuples(index=False):
        reaction = str(row.rhea_id)
        positive = str(row.Entry)
        if reaction not in reaction_to_row or positive not in protein_to_row:
            continue
        precursor = precursor_by_reaction.get(reaction, "")
        skeleton = skeleton_by_reaction.get(reaction, "")
        positives = positive_by_reaction.get(reaction, set())
        positive_group = protein_groups.get(positive, positive)
        selected: list[str] = []

        def valid(candidate: str, require_different_skeleton: bool) -> bool:
            if candidate == positive or candidate not in train_proteins or candidate in positives:
                return False
            if protein_groups.get(candidate, candidate) == positive_group:
                return False
            candidate_reactions = reactions_by_protein.get(candidate, set())
            candidate_skeletons = {
                skeleton_by_reaction.get(local_reaction, "")
                for local_reaction in candidate_reactions
                if precursor_by_reaction.get(local_reaction, "") == precursor
            } - {""}
            if not candidate_skeletons:
                return False
            if require_different_skeleton and skeleton in candidate_skeletons:
                return False
            return True

        for strict in (True, False):
            for candidate_index in neighbor_order[protein_to_row[positive]]:
                candidate = protein_ids[int(candidate_index)]
                if candidate in selected or not valid(candidate, strict):
                    continue
                selected.append(candidate)
                if len(selected) >= negatives_per_positive:
                    break
            if len(selected) >= negatives_per_positive:
                break
        if len(selected) < negatives_per_positive:
            for candidate_index in neighbor_order[protein_to_row[positive]]:
                candidate = protein_ids[int(candidate_index)]
                if (
                    candidate in selected
                    or candidate == positive
                    or candidate not in train_proteins
                    or candidate in positives
                    or protein_groups.get(candidate, candidate) == positive_group
                ):
                    continue
                selected.append(candidate)
                if len(selected) >= negatives_per_positive:
                    break
        for negative in selected:
            reaction_rows.append(reaction_to_row[reaction])
            positive_rows.append(protein_to_row[positive])
            negative_rows.append(protein_to_row[negative])
    if not reaction_rows:
        raise ValueError("No TPS-specialized pairwise triples were generated")
    return (
        np.asarray(reaction_rows, dtype=np.int64),
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
    )


def train_model(
    *,
    reaction_features: np.ndarray,
    global_features: np.ndarray,
    motif_features: np.ndarray,
    pocket_features: np.ndarray,
    descriptors: np.ndarray,
    motif_presence: np.ndarray,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[TPSPairCrossEncoder, list[dict[str, float]]]:
    seed_everything(config.seed)
    model = TPSPairCrossEncoder(
        reaction_dim=reaction_features.shape[1],
        global_dim=global_features.shape[1],
        motif_dim=motif_features.shape[2],
        pocket_dim=pocket_features.shape[1],
        descriptor_dim=descriptors.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)
    tensors = {
        "reaction": torch.as_tensor(reaction_features, dtype=torch.float32, device=device),
        "global": torch.as_tensor(global_features, dtype=torch.float32, device=device),
        "motif": torch.as_tensor(motif_features, dtype=torch.float32, device=device),
        "pocket": torch.as_tensor(pocket_features, dtype=torch.float32, device=device),
        "descriptor": torch.as_tensor(descriptors, dtype=torch.float32, device=device),
        "presence": torch.as_tensor(motif_presence, dtype=torch.float32, device=device),
    }
    reaction_rows, positive_rows, negative_rows = triples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    def score(r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return model.score_pairs(
            tensors["reaction"][r],
            tensors["global"][p],
            tensors["motif"][p],
            tensors["pocket"][p],
            tensors["descriptor"][p],
            tensors["presence"][p],
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = generator.permutation(len(reaction_rows))
        losses: list[float] = []
        accuracies: list[float] = []
        for start in range(0, len(permutation), config.batch_size):
            selection = permutation[start : start + config.batch_size]
            r = torch.as_tensor(reaction_rows[selection], dtype=torch.long, device=device)
            p = torch.as_tensor(positive_rows[selection], dtype=torch.long, device=device)
            n = torch.as_tensor(negative_rows[selection], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            positive_score = score(r, p)
            negative_score = score(r, n)
            pairwise = F.softplus(
                negative_score - positive_score + config.margin
            ).mean()
            separation = F.softplus(-positive_score).mean() + F.softplus(
                negative_score
            ).mean()
            loss = pairwise + 0.05 * separation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float((positive_score > negative_score).float().mean().cpu()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "mean_loss": mean_loss,
                    "pairwise_accuracy": float(np.mean(accuracies)),
                }
            )
    if best_state is None:
        raise RuntimeError("Cross-encoder training produced no state")
    model.load_state_dict(best_state)
    return model, history


def score_all_candidates(
    model: TPSPairCrossEncoder,
    reaction_features: np.ndarray,
    global_features: np.ndarray,
    motif_features: np.ndarray,
    pocket_features: np.ndarray,
    descriptors: np.ndarray,
    motif_presence: np.ndarray,
    reaction_row: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(global_features), batch_size):
            end = min(len(global_features), start + batch_size)
            count = end - start
            reaction = torch.as_tensor(
                np.repeat(
                    reaction_features[reaction_row : reaction_row + 1], count, axis=0
                ),
                dtype=torch.float32,
                device=device,
            )
            score = model.score_pairs(
                reaction,
                torch.as_tensor(global_features[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(motif_features[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(pocket_features[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(descriptors[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(motif_presence[start:end], dtype=torch.float32, device=device),
            )
            rows.append(score.cpu().numpy())
    return np.concatenate(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TPS-specialized pair cross-encoder over global, motif, pocket, and reaction views."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--global-embedding-dir", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--motif-embedding-dir", type=Path, default=DEFAULT_MOTIF)
    parser.add_argument("--pocket-embedding-dir", type=Path, default=DEFAULT_POCKET)
    parser.add_argument("--pocket-audit", type=Path, default=DEFAULT_POCKET_AUDIT)
    parser.add_argument("--motif-audit", type=Path, default=DEFAULT_MOTIF_AUDIT)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-partition", choices=["all", "development", "frozen"], default="all")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--ranking-depth", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = parse_int_tuple(args.budgets)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_matrix, protein_ids = load_protein_features(args.global_embedding_dir.resolve())
    pocket_matrix = load_aligned_embedding(args.pocket_embedding_dir.resolve(), protein_ids)
    motif_matrix, motif_descriptors, motif_presence, motif_schema = load_motif_blocks(
        args.motif_embedding_dir.resolve(), protein_ids
    )
    local_descriptors, local_descriptor_columns = build_local_descriptors(
        protein_ids, args.pocket_audit.resolve(), args.motif_audit.resolve()
    )
    descriptors = np.concatenate([local_descriptors, motif_descriptors], axis=1).astype(
        np.float32
    )

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id", "smiles_seq"]].drop_duplicates(
        ["Entry", "rhea_id"]
    )
    reaction_matrix, reaction_ids, reaction_table, reaction_schema = build_reaction_features(
        positives, "multiview"
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    positives = positives[
        positives.Entry.isin(protein_to_row) & positives.rhea_id.isin(reaction_to_row)
    ].copy()
    strict = pd.read_csv(args.strict_splits, dtype=str).fillna("")
    strict[["protein_fold", "reaction_fold"]] = strict[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    pairs = positives[["Entry", "rhea_id"]].merge(
        strict[
            [
                "Entry",
                "rhea_id",
                "protein_cluster",
                "reaction_cluster",
                "protein_fold",
                "reaction_fold",
            ]
        ].drop_duplicates(["Entry", "rhea_id"]),
        on=["Entry", "rhea_id"],
        how="left",
        validate="one_to_one",
    )
    if pairs[["protein_fold", "reaction_fold"]].isna().any().any():
        raise ValueError("Strict split assignments are incomplete")
    pairs[["protein_fold", "reaction_fold"]] = pairs[
        ["protein_fold", "reaction_fold"]
    ].astype(int)
    clusters = pd.read_csv(args.protein_clusters, dtype=str).fillna("")
    protein_groups = dict(zip(clusters.entry.astype(str), clusters.cluster_id.astype(str)))
    protein_groups = {value: protein_groups.get(value, value) for value in protein_ids}
    precursor_by_reaction = dict(
        zip(reaction_table.rhea_id.astype(str), reaction_table.precursor_class.astype(str))
    )
    skeleton_by_reaction = dict(
        zip(
            reaction_table.rhea_id.astype(str),
            reaction_table.product_skeleton_class.astype(str),
        )
    )
    all_positive_by_reaction = {
        reaction: set(group.Entry.astype(str))
        for reaction, group in pairs.groupby("rhea_id", sort=True)
    }
    neighbor_order = build_global_neighbor_order(global_matrix)
    config = TrainingConfig(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
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
    completed = 0
    stop = False
    for protein_fold in range(5):
        for reaction_fold in range(5):
            is_development = protein_fold == 4 or reaction_fold == 4
            if args.strict_partition == "development" and not is_development:
                continue
            if args.strict_partition == "frozen" and is_development:
                continue
            if args.max_cells and completed >= args.max_cells:
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
                completed += 1
                continue
            triples = build_specialized_triples(
                train_pairs,
                protein_ids,
                reaction_ids,
                protein_to_row,
                reaction_to_row,
                neighbor_order,
                protein_groups,
                precursor_by_reaction,
                skeleton_by_reaction,
                args.negatives_per_positive,
            )
            model, history = train_model(
                reaction_features=reaction_matrix,
                global_features=global_matrix,
                motif_features=motif_matrix,
                pocket_features=pocket_matrix,
                descriptors=descriptors,
                motif_presence=motif_presence,
                triples=triples,
                config=config,
                device=device,
            )
            training_records.append(
                {
                    "protein_fold": protein_fold,
                    "reaction_fold": reaction_fold,
                    "n_train_pairs": len(train_pairs),
                    "n_training_triples": len(triples[0]),
                    "final_loss": history[-1]["mean_loss"],
                    "best_loss": min(row["mean_loss"] for row in history),
                    "final_pairwise_accuracy": history[-1]["pairwise_accuracy"],
                }
            )
            for reaction, group in test_pairs.groupby("rhea_id", sort=True):
                positives_for_query = set(group.Entry.astype(str))
                known_other = all_positive_by_reaction.get(reaction, set()) - positives_for_query
                scores = score_all_candidates(
                    model,
                    reaction_matrix,
                    global_matrix,
                    motif_matrix,
                    pocket_matrix,
                    descriptors,
                    motif_presence,
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
                            scores, protein_ids, positives_for_query, known_other, budgets
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
            completed += 1
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
        "method": "tps_pair_cross_encoder",
        "strict_partition": args.strict_partition,
        "budgets": list(budgets),
        "config": config.__dict__,
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "n_positive_pairs": len(pairs),
        "ranking_depth": args.ranking_depth,
        "reaction_schema": reaction_schema,
        "motif_schema": motif_schema,
        "descriptor_columns": local_descriptor_columns,
        "negative_mining": {
            "primary": "same precursor, different known product skeleton, sequence-near, outside positive 50% cluster",
            "secondary": "same precursor sequence-near",
            "fallback": "sequence-near labeled TPS outside positive cluster",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
