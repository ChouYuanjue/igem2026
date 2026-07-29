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
from projects.active.terpene_screening.evaluate_pocket_local_reranker import (  # noqa: E402
    build_global_neighbor_order,
    build_pairwise_training_rows,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    build_reaction_features,
    load_protein_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_GLOBAL_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_MOTIF_EMBEDDINGS = (
    ROOT / "data/terpene_embeddings/esmc600m_motif_context_combined"
)
DEFAULT_STRICT_SPLITS = ROOT / "data/terpene_cold_splits/positive_pair_fold_assignments.csv"
DEFAULT_PROTEIN_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_motif_channel_reranker"
DEFAULT_BUDGETS = (3, 5, 10, 20)
MOTIF_NAMES = ("ddxxd", "nse_dte", "dxdd", "qw")


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


def load_motif_blocks(
    directory: Path,
    protein_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    global_dim = int(summary["global_dimension"])
    descriptor_dim = int(summary["descriptor_dimension"])
    entries = pd.read_csv(directory / "entries.csv", dtype=str).fillna("")
    entries["row"] = pd.to_numeric(entries["row"]).astype(int)
    matrix = np.load(directory / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError("Motif feature matrix and entries differ")
    expected = global_dim * (1 + len(MOTIF_NAMES)) + descriptor_dim
    if matrix.shape[1] != expected:
        raise ValueError(f"Unexpected motif feature dimension {matrix.shape[1]} != {expected}")
    row_by_id = dict(zip(entries.Entry.astype(str), entries.row.astype(int)))
    missing = [value for value in protein_ids if value not in row_by_id]
    if missing:
        raise ValueError(f"Motif features miss identifiers: {missing[:10]}")
    aligned = np.stack([matrix[row_by_id[value]] for value in protein_ids])
    start = global_dim
    blocks = []
    for _ in MOTIF_NAMES:
        blocks.append(aligned[:, start : start + global_dim])
        start += global_dim
    descriptors = aligned[:, start : start + descriptor_dim]
    presence = descriptors[:, [0, 3, 6, 9]].astype(np.float32)
    motif_blocks = np.stack(blocks, axis=1).astype(np.float32)
    return motif_blocks, descriptors.astype(np.float32), presence, summary


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


class ReactionConditionedMotifTower(nn.Module):
    def __init__(
        self,
        reaction_dim: int,
        motif_dim: int,
        descriptor_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.reaction_channels = nn.ModuleList(
            [
                Projection(reaction_dim, hidden_dim, embedding_dim, dropout)
                for _ in range(len(MOTIF_NAMES) + 1)
            ]
        )
        self.motif_channels = nn.ModuleList(
            [
                Projection(motif_dim, hidden_dim, embedding_dim, dropout)
                for _ in MOTIF_NAMES
            ]
        )
        self.descriptor_fallback = Projection(
            descriptor_dim, max(64, hidden_dim // 2), embedding_dim, dropout
        )
        self.reaction_gate = nn.Sequential(
            nn.LayerNorm(reaction_dim),
            nn.Linear(reaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(MOTIF_NAMES) + 1),
        )

    def channel_scores(
        self,
        reaction_values: torch.Tensor,
        motif_values: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        reaction_channels = torch.stack(
            [projection(reaction_values) for projection in self.reaction_channels], dim=1
        )
        protein_channels = [
            projection(motif_values[:, index, :])
            for index, projection in enumerate(self.motif_channels)
        ]
        protein_channels.append(self.descriptor_fallback(descriptors))
        protein = torch.stack(protein_channels, dim=1)
        return (reaction_channels * protein).sum(dim=-1)

    def score(
        self,
        reaction_values: torch.Tensor,
        motif_values: torch.Tensor,
        descriptors: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        channel_scores = self.channel_scores(reaction_values, motif_values, descriptors)
        fallback = torch.ones(
            (len(presence), 1), dtype=presence.dtype, device=presence.device
        )
        active = torch.cat([presence, fallback], dim=1)
        gates = torch.softmax(self.reaction_gate(reaction_values), dim=-1) * active
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (gates * channel_scores).sum(dim=-1)


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


def train_model(
    *,
    reaction_features: np.ndarray,
    motif_features: np.ndarray,
    descriptors: np.ndarray,
    presence: np.ndarray,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[ReactionConditionedMotifTower, list[dict[str, float]]]:
    seed_everything(config.seed)
    model = ReactionConditionedMotifTower(
        reaction_dim=reaction_features.shape[1],
        motif_dim=motif_features.shape[2],
        descriptor_dim=descriptors.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    motif_tensor = torch.as_tensor(motif_features, dtype=torch.float32, device=device)
    descriptor_tensor = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
    presence_tensor = torch.as_tensor(presence, dtype=torch.float32, device=device)
    reaction_rows, positive_rows, negative_rows = triples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
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
                reaction_tensor[r], motif_tensor[p], descriptor_tensor[p], presence_tensor[p]
            )
            negative_scores = model.score(
                reaction_tensor[r], motif_tensor[n], descriptor_tensor[n], presence_tensor[n]
            )
            pairwise = F.softplus(negative_scores - positive_scores + config.margin).mean()
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
        raise RuntimeError("Motif model training produced no state")
    model.load_state_dict(best_state)
    return model, history


def score_candidates(
    model: ReactionConditionedMotifTower,
    reaction_features: np.ndarray,
    motif_features: np.ndarray,
    descriptors: np.ndarray,
    presence: np.ndarray,
    reaction_row: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        count = len(motif_features)
        reaction = torch.as_tensor(
            np.repeat(reaction_features[reaction_row : reaction_row + 1], count, axis=0),
            dtype=torch.float32,
            device=device,
        )
        motifs = torch.as_tensor(motif_features, dtype=torch.float32, device=device)
        descriptor_tensor = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
        presence_tensor = torch.as_tensor(presence, dtype=torch.float32, device=device)
        scores = model.score(reaction, motifs, descriptor_tensor, presence_tensor)
        return scores.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reaction-conditioned catalytic-motif channel reranker for strict TPS retrieval."
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--global-embedding-dir", type=Path, default=DEFAULT_GLOBAL_EMBEDDINGS)
    parser.add_argument("--motif-embedding-dir", type=Path, default=DEFAULT_MOTIF_EMBEDDINGS)
    parser.add_argument("--strict-splits", type=Path, default=DEFAULT_STRICT_SPLITS)
    parser.add_argument("--protein-clusters", type=Path, default=DEFAULT_PROTEIN_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--strict-partition", choices=["all", "development", "frozen"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
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
    motif_matrix, descriptors, presence, motif_schema = load_motif_blocks(
        args.motif_embedding_dir.resolve(), protein_ids
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
            model, history = train_model(
                reaction_features=reaction_matrix,
                motif_features=motif_matrix,
                descriptors=descriptors,
                presence=presence,
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
                scores = score_candidates(
                    model,
                    reaction_matrix,
                    motif_matrix,
                    descriptors,
                    presence,
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
        "method": "reaction_conditioned_catalytic_motif_channels",
        "motif_names": list(MOTIF_NAMES),
        "strict_partition": args.strict_partition,
        "budgets": list(budgets),
        "config": config.__dict__,
        "ranking_depth": args.ranking_depth,
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "n_positive_pairs": len(pairs),
        "motif_schema": motif_schema,
        "reaction_feature_schema": feature_schema,
        "negative_mining": {
            "primary": "same-precursor labeled TPS proteins nearest in global ESM-C space",
            "false_negative_control": "exclude known positives and the positive 50% identity cluster",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
